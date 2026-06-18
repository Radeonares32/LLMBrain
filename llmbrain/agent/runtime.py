"""Provider-independent Agent Runtime and Loop implementation for LLMBrain."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from pydantic import BaseModel, Field

from llmbrain.agent.context import ToolExecutionContext
from llmbrain.agent.memory import TaskMemoryManager
from llmbrain.agent.safety import PermissionLevel, SafetyMode
from llmbrain.agent.tools import AuditRecord, ToolRegistry
from llmbrain.formats.brainframe import build_brainframe_context

if TYPE_CHECKING:
    from llmbrain.llm.base import BaseLLMProvider

# ── Typed Runtime Errors ─────────────────────────────────────────────


class AgentRuntimeError(RuntimeError):
    """Base error class for all agent runtime issues."""

    pass


class ProviderFailureError(AgentRuntimeError):
    """Raised when an LLM provider request fails."""

    pass


class ProviderTimeoutError(AgentRuntimeError):
    """Raised when an LLM provider request times out."""

    pass


class MalformedModelResponseError(AgentRuntimeError):
    """Raised when the LLM returns an invalid action or malformed JSON."""

    pass


class UnknownToolError(AgentRuntimeError):
    """Raised when the LLM requests a tool not found in the registry."""

    pass


class InvalidToolArgumentsError(AgentRuntimeError):
    """Raised when tool arguments fail schema validation."""

    pass


class PermissionDeniedError(AgentRuntimeError):
    """Raised when safety policy blocks a tool execution."""

    pass


class ApprovalRequiredError(AgentRuntimeError):
    """Raised when a tool requires explicit user confirmation."""

    pass


class ToolExecutionError(AgentRuntimeError):
    """Raised when a tool execution fails with an internal error."""

    pass


class ToolTimeoutError(AgentRuntimeError):
    """Raised when a tool execution exceeds its timeout."""

    pass


class StateTransitionError(AgentRuntimeError):
    """Raised when the agent attempts an invalid state transition."""

    pass


class ContextBudgetError(AgentRuntimeError):
    """Raised when retrieved items exceed the context token/character budget."""

    pass


class IterationLimitError(AgentRuntimeError):
    """Raised when the agent loop exceeds configured iteration bounds."""

    pass


class CancellationError(AgentRuntimeError):
    """Raised when execution is stopped via cancellation token."""

    pass


class MemoryPersistenceError(AgentRuntimeError):
    """Raised when persisting task outcomes fails."""

    pass


# ── State Machine ────────────────────────────────────────────────────


class AgentState(StrEnum):
    CREATED = "created"
    RETRIEVING_CONTEXT = "retrieving_context"
    ASSEMBLING_CONTEXT = "assembling_context"
    WAITING_FOR_MODEL = "waiting_for_model"
    VALIDATING_TOOL_CALL = "validating_tool_call"
    EXECUTING_TOOL = "executing_tool"
    PROCESSING_OBSERVATION = "processing_observation"
    VERIFYING = "verifying"
    PERSISTING_MEMORY = "persisting_memory"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


_VALID_TRANSITIONS: dict[AgentState, list[AgentState]] = {
    AgentState.CREATED: [
        AgentState.RETRIEVING_CONTEXT,
        AgentState.PERSISTING_MEMORY,
        AgentState.FAILED,
        AgentState.CANCELLED,
    ],
    AgentState.RETRIEVING_CONTEXT: [
        AgentState.ASSEMBLING_CONTEXT,
        AgentState.PERSISTING_MEMORY,
        AgentState.FAILED,
        AgentState.CANCELLED,
    ],
    AgentState.ASSEMBLING_CONTEXT: [
        AgentState.WAITING_FOR_MODEL,
        AgentState.PERSISTING_MEMORY,
        AgentState.FAILED,
        AgentState.CANCELLED,
    ],
    AgentState.WAITING_FOR_MODEL: [
        AgentState.VALIDATING_TOOL_CALL,
        AgentState.VERIFYING,
        AgentState.PERSISTING_MEMORY,
        AgentState.FAILED,
        AgentState.CANCELLED,
        AgentState.COMPLETED,
        AgentState.WAITING_FOR_MODEL,
    ],
    AgentState.VALIDATING_TOOL_CALL: [
        AgentState.EXECUTING_TOOL,
        AgentState.WAITING_FOR_MODEL,
        AgentState.PERSISTING_MEMORY,
        AgentState.FAILED,
        AgentState.CANCELLED,
    ],
    AgentState.EXECUTING_TOOL: [
        AgentState.PROCESSING_OBSERVATION,
        AgentState.PERSISTING_MEMORY,
        AgentState.FAILED,
        AgentState.CANCELLED,
    ],
    AgentState.PROCESSING_OBSERVATION: [
        AgentState.WAITING_FOR_MODEL,
        AgentState.PERSISTING_MEMORY,
        AgentState.FAILED,
        AgentState.CANCELLED,
    ],
    AgentState.VERIFYING: [
        AgentState.PERSISTING_MEMORY,
        AgentState.COMPLETED,
        AgentState.FAILED,
        AgentState.CANCELLED,
    ],
    AgentState.PERSISTING_MEMORY: [
        AgentState.COMPLETED,
        AgentState.FAILED,
        AgentState.CANCELLED,
    ],
    AgentState.COMPLETED: [],
    AgentState.FAILED: [],
    AgentState.CANCELLED: [],
}


# ── Cancellation ─────────────────────────────────────────────────────


class CancellationToken:
    """Tracks task cancellation requests."""

    def __init__(self) -> None:
        self._is_cancelled = False

    def cancel(self) -> None:
        self._is_cancelled = True

    @property
    def is_cancelled(self) -> bool:
        return self._is_cancelled

    def check(self) -> None:
        if self._is_cancelled:
            raise CancellationError("Task was cancelled.")


# ── Model Provider Interface ─────────────────────────────────────────


class Message(BaseModel):
    """Role-based prompt message."""

    role: str  # 'system', 'user', 'assistant', 'tool'
    content: str
    name: str | None = None
    tool_call_id: str | None = None


class ModelRequest(BaseModel):
    """Normalized request sent to LLM providers."""

    messages: list[Message]
    system_prompt: str = ""
    tools: list[dict] = Field(default_factory=list)
    model: str | None = None
    temperature: float = 0.0
    max_tokens: int = 4096


class ModelResponse(BaseModel):
    """Normalized response received from LLM providers."""

    message: str | None = None
    tool_calls: list[dict] = Field(default_factory=list)
    finish_reason: str = "stop"  # 'stop', 'tool_calls', 'length', 'cancel', 'timeout'
    input_tokens: int = 0
    output_tokens: int = 0
    provider_metadata: dict = Field(default_factory=dict)


class AgentAction(BaseModel):
    """Representing structured response schema for LLM calls."""

    thought: str = Field(description="Explanation of the choice.")
    tool_name: str | None = Field(default=None, description="Name of the tool to execute.")
    tool_arguments: dict[str, Any] | None = Field(
        default=None, description="Arguments for the tool."
    )
    finish_response: str | None = Field(default=None, description="Final response to the user.")
    finish_verification: str | None = Field(default=None, description="Verification summary.")


class ModelProvider:
    """Interface that LLM adapters must implement for agent tasks."""

    async def generate(
        self,
        request: ModelRequest,
        cancellation_token: CancellationToken | None = None,
        stream_callback: Callable[[str], None] | None = None,
    ) -> ModelResponse:
        """Execute a text/structured completion request."""
        raise NotImplementedError


class LLMProviderAdapter(ModelProvider):
    """Adapts the existing BaseLLMProvider to the new ModelProvider interface."""

    def __init__(self, provider: BaseLLMProvider) -> None:
        self.provider = provider

    async def generate(
        self,
        request: ModelRequest,
        cancellation_token: CancellationToken | None = None,
        stream_callback: Callable[[str], None] | None = None,
    ) -> ModelResponse:
        if cancellation_token:
            cancellation_token.check()

        # Build prompt from conversation history and system prompt
        prompt_content = ""
        for msg in request.messages:
            prompt_content += f"{msg.role}: {msg.content}\n"

        from llmbrain.models.llm import LLMRequest

        llm_req = LLMRequest(
            prompt=prompt_content,
            system_prompt=request.system_prompt,
            model=request.model,
        )

        schema = AgentAction.model_json_schema()

        llm_resp = await self.provider.generate_structured(llm_req, schema=schema, stream_callback=stream_callback)

        return ModelResponse(
            message=llm_resp.raw if llm_resp.is_valid else None,
            tool_calls=[],
            finish_reason="stop",
            input_tokens=llm_resp.usage_tokens,
            output_tokens=0,
        )


# ── Permission Policy ────────────────────────────────────────────────


class PermissionDecision(StrEnum):
    ALLOW = "allow"
    DENY = "deny"
    REQUIRE_APPROVAL = "require_approval"


class PermissionPolicy:
    """Determines whether a tool execution is allowed under current safety settings."""

    def __init__(self, mode: SafetyMode) -> None:
        self.mode = mode

    def evaluate(
        self,
        tool_name: str,
        permission_level: PermissionLevel,
        arguments: dict[str, Any],
    ) -> PermissionDecision:
        if permission_level == PermissionLevel.PROHIBITED:
            return PermissionDecision.DENY

        if self.mode == SafetyMode.READ_ONLY:
            if permission_level == PermissionLevel.READ:
                return PermissionDecision.ALLOW
            return PermissionDecision.DENY

        if self.mode == SafetyMode.DENY_SHELL:
            if permission_level in (
                PermissionLevel.SHELL,
                PermissionLevel.EXECUTE_SAFE,
                PermissionLevel.EXECUTE_NETWORK,
                PermissionLevel.DESTRUCTIVE,
            ):
                return PermissionDecision.DENY
            if permission_level == PermissionLevel.READ:
                return PermissionDecision.ALLOW
            if permission_level == PermissionLevel.WRITE:
                return PermissionDecision.REQUIRE_APPROVAL
            return PermissionDecision.DENY

        if self.mode == SafetyMode.TRUSTED_PROJECT:
            if permission_level in (PermissionLevel.READ, PermissionLevel.WRITE):
                return PermissionDecision.ALLOW
            if permission_level in (
                PermissionLevel.SHELL,
                PermissionLevel.EXECUTE_SAFE,
                PermissionLevel.EXECUTE_NETWORK,
                PermissionLevel.DESTRUCTIVE,
            ):
                return PermissionDecision.REQUIRE_APPROVAL
            return PermissionDecision.DENY

        # SafetyMode.ASK_BEFORE_WRITE (Default)
        if permission_level == PermissionLevel.READ:
            return PermissionDecision.ALLOW
        if permission_level in (
            PermissionLevel.WRITE,
            PermissionLevel.SHELL,
            PermissionLevel.EXECUTE_SAFE,
            PermissionLevel.EXECUTE_NETWORK,
            PermissionLevel.DESTRUCTIVE,
        ):
            return PermissionDecision.REQUIRE_APPROVAL
        return PermissionDecision.DENY


# ── Context Assembly & Memory Retrieval ──────────────────────────────


class MemoryRetriever:
    """Retrieves relevant facts and entities from repository storage."""

    def __init__(self, memory_manager: TaskMemoryManager) -> None:
        self.memory_manager = memory_manager

    def retrieve(self, task_query: str, memory_types: list[str] | None = None) -> dict[str, Any]:
        project_id = self.memory_manager._project_id()
        entities = self.memory_manager.store.get_entities(project_id)
        facts = self.memory_manager.store.get_facts(project_id)
        relations = self.memory_manager.store.get_relations(project_id)

        if memory_types:

            def matches_type(item_type: str, allowed_types: list[str]) -> bool:
                for allowed in allowed_types:
                    if allowed == item_type or item_type in allowed or allowed in item_type:
                        return True
                    if allowed == "symbol_fact" and item_type in (
                        "symbol",
                        "class",
                        "function",
                        "method",
                    ):
                        return True
                    if allowed == "repository_overview" and item_type in (
                        "overview",
                        "module_summary",
                        "module",
                    ):
                        return True
                    if allowed == "failure_resolution" and item_type in (
                        "failure",
                        "resolution",
                        "defect",
                    ):
                        return True
                return False

            entities = [e for e in entities if matches_type(e.get("type", "symbol"), memory_types)]
            facts = [f for f in facts if matches_type(f.get("type", "general"), memory_types)]

        # Keyword filtering proxy
        query_words = set(task_query.lower().split())
        if query_words:
            filtered_entities = [
                e for e in entities if any(w in str(e.get("name", "")).lower() for w in query_words)
            ]
            # fallback if too few matches
            if not filtered_entities:
                filtered_entities = entities[:20]
        else:
            filtered_entities = entities[:20]

        filtered_entity_ids = {str(e.get("id")) for e in filtered_entities}
        filtered_relations = [
            r
            for r in relations
            if str(r.get("source_entity_id")) in filtered_entity_ids
            or str(r.get("target_entity_id")) in filtered_entity_ids
        ]

        return {
            "entities": filtered_entities,
            "facts": facts[:30],
            "relations": filtered_relations,
        }


class ContextAssembler:
    """Assembles retrieved context within token limits."""

    def __init__(self, token_budget: int = 120_000) -> None:
        self.token_budget = token_budget

    def assemble(
        self,
        retrieved_items: dict[str, Any],
        task_query: str,
        project_name: str,
        project_id: str,
    ) -> str:
        # 1 character = 0.25 tokens estimation proxy
        max_chars = self.token_budget * 4

        ctx = build_brainframe_context(
            project_name=project_name,
            project_id=project_id,
            entities=retrieved_items.get("entities", []),
            relations=retrieved_items.get("relations", []),
            facts=retrieved_items.get("facts", []),
            max_chars=max_chars,
        )

        if "@truncated true" in ctx:
            # Let's verify if budget is strictly respected
            if len(ctx) > max_chars:
                raise ContextBudgetError("Assembled context exceeds configured token budget.")

        return ctx


# ── Verification Result ──────────────────────────────────────────────


class VerificationResult(BaseModel):
    """Represent task validation outcome."""

    status: str  # 'passed', 'failed', 'skipped'
    summary: str
    evidence: list[str] = Field(default_factory=list)


class Verifier:
    """Verifier interface checking work outcomes."""

    def verify(
        self,
        task: str,
        transcript: list[dict[str, Any]],
        tool_results: list[dict[str, Any]],
    ) -> VerificationResult:
        raise NotImplementedError


class DefaultVerifier(Verifier):
    """Standard verifier analyzing test results."""

    def verify(
        self,
        task: str,
        transcript: list[dict[str, Any]],
        tool_results: list[dict[str, Any]],
    ) -> VerificationResult:
        # Check run_tests tool execution status
        test_failed = False
        run_tests_called = False
        for res in tool_results:
            if res.get("tool_name") == "run_tests":
                run_tests_called = True
                if not res.get("success"):
                    test_failed = True

        if test_failed:
            return VerificationResult(
                status="failed",
                summary="Test suite failed.",
                evidence=["run_tests reported errors."],
            )
        if run_tests_called:
            return VerificationResult(
                status="passed",
                summary="All tests passed successfully.",
                evidence=["run_tests output verified."],
            )

        return VerificationResult(
            status="skipped",
            summary="No test validation was run.",
            evidence=[],
        )


# ── Runtime Events & Event Bus ───────────────────────────────────────


class RuntimeEvent(BaseModel):
    """Observation telemetry event."""

    id: str = Field(default_factory=lambda: uuid4().hex)
    task_id: str
    session_id: str
    timestamp: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    event_type: str
    payload: dict = Field(default_factory=dict)


class RuntimeEventBus:
    """Listens and distributes execution loop telemetry events."""

    def __init__(self) -> None:
        self._listeners: list[Callable[[RuntimeEvent], None]] = []

    def subscribe(self, listener: Callable[[RuntimeEvent], None]) -> None:
        self._listeners.append(listener)

    def emit(self, event: RuntimeEvent) -> None:
        for listener in self._listeners:
            try:
                listener(event)
            except Exception:
                pass


# ── Session & Task Record models ─────────────────────────────────────


class TaskRecord(BaseModel):
    """Task run telemetry log record."""

    task_id: str
    session_id: str
    request: str
    status: str
    summary: str
    started_at: str
    ended_at: str
    error: str | None = None
    verification: VerificationResult | None = None


class AgentSession(BaseModel):
    """Tracks active state history and total token budgets."""

    session_id: str = Field(default_factory=lambda: uuid4().hex)
    state: AgentState = AgentState.CREATED
    state_history: list[tuple[AgentState, str]] = Field(default_factory=list)
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    tool_calls: list[dict] = Field(default_factory=list)
    audit_records: list[AuditRecord] = Field(default_factory=list)


# ── Core Runtime Loop ────────────────────────────────────────────────


class AgentRuntime:
    """Core runtime engine driving task execution."""

    def __init__(
        self,
        project_root: Path,
        provider: ModelProvider | BaseLLMProvider,
        safety_mode: SafetyMode = SafetyMode.ASK_BEFORE_WRITE,
        prompt_func: Callable[[str], bool] | None = None,
        event_listener: Callable[[RuntimeEvent], None] | None = None,
        agent_name: str | None = None,
    ) -> None:
        self.project_root = Path(project_root).resolve()
        from llmbrain.llm.base import BaseLLMProvider

        if isinstance(provider, BaseLLMProvider):
            self.model_provider = LLMProviderAdapter(provider)
        else:
            self.model_provider = provider
        self.event_bus = RuntimeEventBus()
        if event_listener:
            self.event_bus.subscribe(event_listener)

        self.tools = ToolRegistry(self.project_root)
        self.memory = TaskMemoryManager(self.project_root)

        # Agent-specific definition loading
        self.agent_name = agent_name
        self.agent_def = None
        if agent_name:
            from llmbrain.agent.agents import AgentRegistry

            self.registry = AgentRegistry()
            self.registry.load_project_config(self.project_root)
            self.agent_def = self.registry.get_agent(agent_name)

            # Map permission mode from agent definition
            mode_str = self.agent_def.permissions.get("mode", "ask-before-write")
            if mode_str == "read-only":
                safety_mode = SafetyMode.READ_ONLY
            elif mode_str == "ask-before-write":
                safety_mode = SafetyMode.ASK_BEFORE_WRITE
            elif mode_str == "trusted-project":
                safety_mode = SafetyMode.TRUSTED_PROJECT
            elif mode_str == "deny-shell":
                safety_mode = SafetyMode.DENY_SHELL

        self.policy = PermissionPolicy(safety_mode)
        self.prompt_func = prompt_func or self._default_prompt

        self.logs_dir = self.project_root / ".llmbrain" / "logs" / "tasks"
        self.logs_dir.mkdir(parents=True, exist_ok=True)

    def _default_prompt(self, msg: str) -> bool:
        print(f"\n⚠️  [SECURITY WARNING] {msg}")
        try:
            ans = input("Do you approve this execution? (y/N): ").strip().lower()
            return ans in ("y", "yes", "evet", "e")
        except (KeyboardInterrupt, EOFError):
            return False

    async def execute_task(
        self,
        user_request: str,
        max_iterations: int = 15,
        max_tool_calls: int = 30,
        max_duration_seconds: float = 300.0,
        token_budget: int = 120_000,
        cancellation_token: CancellationToken | None = None,
        parent_chain: list[str] | None = None,
        parent_task_id: str | None = None,
        prior_messages: list[Message] | None = None,
    ) -> TaskRecord:
        start_time = time.time()
        cancellation_token = cancellation_token or CancellationToken()
        parent_chain = parent_chain or []

        if self.agent_def:
            max_iterations = self.agent_def.limits.max_iterations
            max_tool_calls = self.agent_def.limits.max_tool_calls
            token_budget = self.agent_def.context.token_budget

            if len(parent_chain) >= self.agent_def.limits.max_delegations:
                from llmbrain.agent.agents import DelegationDepthLimitError

                raise DelegationDepthLimitError(
                    "DELEGATION_DEPTH_LIMIT",
                    f"Delegation chain exceeded limit of {self.agent_def.limits.max_delegations}.",
                )
            if self.agent_def.name in parent_chain:
                from llmbrain.agent.agents import DelegationLoopError

                raise DelegationLoopError(
                    "DELEGATION_LOOP",
                    f"Delegation loop detected: {self.agent_def.name} is already in parent chain.",
                )

        session = AgentSession()
        task_id = uuid4().hex
        session_id = session.session_id

        tool_ctx = ToolExecutionContext(
            task_id=task_id,
            session_id=session_id,
            workspace_root=self.project_root,
            cwd=self.project_root,
            permission_mode=self.policy.mode,
            cancellation_token=cancellation_token,
            event_bus=self.event_bus,
        )

        def change_state(new_state: AgentState) -> None:
            # Validate transitions
            allowed = _VALID_TRANSITIONS.get(session.state, [])
            if new_state not in allowed:
                raise StateTransitionError(
                    f"Invalid transition from {session.state.value} to {new_state.value}"
                )
            session.state = new_state
            session.state_history.append((new_state, datetime.now(UTC).isoformat()))
            self.event_bus.emit(
                RuntimeEvent(
                    task_id=task_id,
                    session_id=session_id,
                    event_type="state_changed",
                    payload={"new_state": new_state.value},
                )
            )

        # Emit task_created
        self.event_bus.emit(
            RuntimeEvent(
                task_id=task_id,
                session_id=session_id,
                event_type="task_created",
                payload={"request": user_request},
            )
        )

        status = "failed"
        summary = ""
        error_summary = None

        if self.agent_def:
            from llmbrain.agent.agents import AgentSpecificVerifier

            verifier = AgentSpecificVerifier(self.agent_def.name, self.registry)
        else:
            verifier = DefaultVerifier()

        verification_res = VerificationResult(
            status="skipped",
            summary="Verification skipped due to early error.",
        )

        try:
            # Check duration/cancellation
            cancellation_token.check()

            # 1. Retrieve Context
            change_state(AgentState.RETRIEVING_CONTEXT)
            self.event_bus.emit(
                RuntimeEvent(
                    task_id=task_id,
                    session_id=session_id,
                    event_type="memory_retrieval_started",
                )
            )
            retriever = MemoryRetriever(self.memory)
            memory_types = self.agent_def.context.memory_types if self.agent_def else None
            retrieved_items = retriever.retrieve(user_request, memory_types=memory_types)
            self.event_bus.emit(
                RuntimeEvent(
                    task_id=task_id,
                    session_id=session_id,
                    event_type="memory_retrieved",
                )
            )

            # 2. Assemble Context
            change_state(AgentState.ASSEMBLING_CONTEXT)
            assembler = ContextAssembler(token_budget)
            context = assembler.assemble(
                retrieved_items,
                user_request,
                self.project_root.name,
                self.memory._project_id(),
            )
            self.event_bus.emit(
                RuntimeEvent(
                    task_id=task_id,
                    session_id=session_id,
                    event_type="context_assembled",
                )
            )

            if self.agent_def:
                from llmbrain.agent.agents import compose_prompt

                system_prompt_base = compose_prompt(self.project_root, self.agent_def)
            else:
                system_prompt_base = (
                    "You are the memory-native terminal coding assistant for LLMBrain."
                )

            system_prompt = (
                f"{system_prompt_base}\n"
                "You must report your actions in JSON format at each step.\n"
                "Options:\n"
                "1. If you want to run a tool:\n"
                "   {\n"
                '     "thought": "Your thought",\n'
                '     "tool_name": "tool_name",\n'
                '     "tool_arguments": {"argument": "value"}\n'
                "   }\n"
                "2. If you want to finish the task:\n"
                "   {\n"
                '     "thought": "Your thought",\n'
                '     "finish_response": "Final response to the user",\n'
                '     "finish_verification": "Verification summary"\n'
                "   }\n"
            )
            if self.agent_def:
                system_prompt += (
                    "\nNOTE: Your final 'finish_response' should contain a JSON payload "
                    f"matching the output schema for '{self.agent_def.name}' as requested.\n"
                )
            system_prompt += f"\nProject Memory:\n{context}\n"

            if prior_messages is not None:
                messages = list(prior_messages)
            else:
                messages = [
                    Message(
                        role="user",
                        content=f"Task: {user_request}\nPlease select the next action.",
                    )
                ]

            tool_calls_count = 0
            iteration = 0
            repeated_call_detector: dict[str, int] = {}
            recovery_attempts = 0
            failed_side_effecting_calls = set()

            # 3. Agent Loop
            while True:
                cancellation_token.check()

                # Check duration limit
                if time.time() - start_time > max_duration_seconds:
                    self.event_bus.emit(
                        RuntimeEvent(
                            task_id=task_id,
                            session_id=session_id,
                            event_type="limit_reached",
                            payload={"limit": "max_duration_seconds"},
                        )
                    )
                    raise IterationLimitError("Maximum duration limit exceeded.")

                # Check iterations limit
                if iteration >= max_iterations:
                    self.event_bus.emit(
                        RuntimeEvent(
                            task_id=task_id,
                            session_id=session_id,
                            event_type="limit_reached",
                            payload={"limit": "max_iterations"},
                        )
                    )
                    raise IterationLimitError("Maximum iteration limit reached.")

                iteration += 1

                # Call LLM
                change_state(AgentState.WAITING_FOR_MODEL)
                self.event_bus.emit(
                    RuntimeEvent(
                        task_id=task_id,
                        session_id=session_id,
                        event_type="model_request_started",
                    )
                )

                try:
                    response = await self.model_provider.generate(
                        ModelRequest(
                            messages=messages,
                            system_prompt=system_prompt,
                            model=None,
                        ),
                        cancellation_token,
                        stream_callback=lambda chunk: self.event_bus.emit(
                            RuntimeEvent(event_type="model_stream_chunk", payload={"chunk": chunk})
                        )
                    )
                except Exception as e:
                    raise ProviderFailureError(f"Model call failed: {e}")

                session.total_input_tokens += response.input_tokens
                session.total_output_tokens += response.output_tokens

                self.event_bus.emit(
                    RuntimeEvent(
                        task_id=task_id,
                        session_id=session_id,
                        event_type="model_response_received",
                        payload={
                            "input_tokens": response.input_tokens,
                            "output_tokens": response.output_tokens,
                        },
                    )
                )

                # Parse response
                try:
                    parsed = json.loads(response.message or "{}")
                except json.JSONDecodeError as exc:
                    # Bounded Recovery behavior
                    if recovery_attempts < 1:
                        recovery_attempts += 1
                        messages.append(
                            Message(
                                role="assistant",
                                content=response.message or "",
                            )
                        )
                        messages.append(
                            Message(
                                role="user",
                                content=(
                                    f"Your response does not contain valid JSON (Error: {exc}). "
                                    "Please respond only in the specified JSON format."
                                ),
                            )
                        )
                        continue
                    else:
                        raise MalformedModelResponseError("Model did not generate valid JSON.")

                _thought = parsed.get("thought", "")
                tool_name = parsed.get("tool_name")
                tool_args = parsed.get("tool_arguments", {})
                finish_response = parsed.get("finish_response")
                _finish_verification = parsed.get("finish_verification")

                # Verify structured model protocol constraints
                if not finish_response and not tool_name:
                    raise MalformedModelResponseError(
                        "Neither tool_name nor finish_response is present in model response."
                    )

                # Process final answer
                if finish_response:
                    change_state(AgentState.VERIFYING)
                    self.event_bus.emit(
                        RuntimeEvent(
                            task_id=task_id,
                            session_id=session_id,
                            event_type="verification_started",
                        )
                    )
                    verification_res = verifier.verify(
                        user_request,
                        [m.model_dump() for m in messages],
                        session.tool_calls,
                    )
                    self.event_bus.emit(
                        RuntimeEvent(
                            task_id=task_id,
                            session_id=session_id,
                            event_type="verification_completed",
                            payload={"status": verification_res.status},
                        )
                    )

                    if verification_res.status == "failed":
                        status = "failed"
                        summary = f"Verification failed: {verification_res.summary}"
                    else:
                        if self.agent_def:
                            from llmbrain.agent.agents import validate_agent_output

                            try:
                                try:
                                    payload = json.loads(finish_response)
                                except json.JSONDecodeError:
                                    if self.agent_def.name == "ask":
                                        payload = {
                                            "answer": finish_response,
                                            "sources": [],
                                            "uncertainties": [],
                                        }
                                    elif self.agent_def.name == "plan":
                                        payload = {
                                            "summary": finish_response,
                                            "affected_paths": [],
                                            "steps": [],
                                            "risks": [],
                                        }
                                    elif self.agent_def.name == "build":
                                        payload = {
                                            "summary": finish_response,
                                            "changed_files": [],
                                            "commands_executed": [],
                                        }
                                    elif self.agent_def.name == "review":
                                        payload = {
                                            "summary": finish_response,
                                            "findings": [],
                                            "verdict": "approve",
                                        }
                                    elif self.agent_def.name == "debug":
                                        payload = {
                                            "symptom": finish_response,
                                            "evidence": [],
                                            "hypotheses": [],
                                        }
                                    elif self.agent_def.name == "test":
                                        payload = {
                                            "coverage_gaps": [],
                                            "tests_added": [],
                                            "commands_executed": [],
                                        }
                                    elif self.agent_def.name == "security":
                                        payload = {
                                            "threat_surface": [],
                                            "findings": [],
                                            "overall_risk": "unknown",
                                        }
                                    else:
                                        payload = {"summary": finish_response}

                                validated = validate_agent_output(self.agent_def.name, payload)
                                summary = json.dumps(validated, ensure_ascii=False)
                                status = "completed"
                            except Exception as schema_err:
                                status = "failed"
                                summary = f"Output schema validation failed: {schema_err}"
                                error_summary = str(schema_err)
                        else:
                            status = "completed"
                            summary = finish_response
                    break

                # Process Tool Call
                if tool_name:
                    if tool_calls_count >= max_tool_calls:
                        self.event_bus.emit(
                            RuntimeEvent(
                                task_id=task_id,
                                session_id=session_id,
                                event_type="limit_reached",
                                payload={"limit": "max_tool_calls"},
                            )
                        )
                        raise IterationLimitError("Maximum tool calls limit reached.")

                    tool_calls_count += 1

                    # Check repeated identical tool calls
                    call_signature = f"{tool_name}:{json.dumps(tool_args, sort_keys=True)}"
                    repeated_call_detector[call_signature] = (
                        repeated_call_detector.get(call_signature, 0) + 1
                    )
                    if repeated_call_detector[call_signature] > 3:
                        raise IterationLimitError(
                            f"The same tool call ({tool_name}) was repeated more than 3 times."
                        )

                    # Check agent tool permissions
                    if self.agent_def:
                        if (
                            self.agent_def.tools.allow
                            and tool_name not in self.agent_def.tools.allow
                            and tool_name != "delegate_task"
                        ):
                            from llmbrain.agent.agents import UnauthorizedToolForAgentError

                            raise UnauthorizedToolForAgentError(
                                "UNAUTHORIZED_TOOL",
                                f"Agent '{self.agent_def.name}' is not authorized to use "
                                f"tool '{tool_name}'.",
                            )
                        if tool_name in self.agent_def.tools.deny:
                            from llmbrain.agent.agents import UnauthorizedToolForAgentError

                            raise UnauthorizedToolForAgentError(
                                "UNAUTHORIZED_TOOL",
                                f"Agent '{self.agent_def.name}' is explicitly denied to use "
                                f"tool '{tool_name}'.",
                            )

                    # Dynamic handling of delegate_task
                    if tool_name == "delegate_task":
                        target_agent = tool_args.get("target_agent")
                        objective = tool_args.get("objective")

                        max_delegations_limit = (
                            self.agent_def.limits.max_delegations if self.agent_def else 2
                        )
                        if len(parent_chain) >= max_delegations_limit:
                            from llmbrain.agent.agents import DelegationDepthLimitError

                            raise DelegationDepthLimitError(
                                "DELEGATION_DEPTH_LIMIT",
                                f"Delegation chain exceeded limit of {max_delegations_limit}.",
                            )

                        if target_agent in parent_chain or (
                            self.agent_def and target_agent == self.agent_def.name
                        ):
                            from llmbrain.agent.agents import DelegationLoopError

                            raise DelegationLoopError(
                                "DELEGATION_LOOP",
                                f"Delegation loop detected: {target_agent} is already in "
                                "the parent chain.",
                            )

                        try:
                            child_def = self.registry.get_agent(target_agent)
                        except Exception as e:
                            from llmbrain.agent.agents import InvalidDelegationTargetError

                            raise InvalidDelegationTargetError(
                                "INVALID_DELEGATION_TARGET",
                                f"Invalid delegation target agent: {target_agent}.",
                                cause=e,
                            )

                        parent_mode = (
                            self.agent_def.permissions.get("mode", "read-only")
                            if self.agent_def
                            else "read-only"
                        )
                        child_mode = child_def.permissions.get("mode", "read-only")

                        safety_levels = {
                            "read-only": 1,
                            "deny-shell": 2,
                            "ask-before-write": 3,
                            "trusted-project": 4,
                        }
                        p_level = safety_levels.get(parent_mode, 1)
                        c_level = safety_levels.get(child_mode, 1)
                        if c_level > p_level:
                            from llmbrain.agent.agents import DelegationPermissionEscalationError

                            raise DelegationPermissionEscalationError(
                                "PERMISSION_ESCALATION",
                                f"Escalation blocked: Child '{target_agent}' mode '{child_mode}' "
                                f"exceeds parent mode '{parent_mode}'.",
                            )

                        self.event_bus.emit(
                            RuntimeEvent(
                                task_id=task_id,
                                session_id=session_id,
                                event_type="delegation_started",
                                payload={"target_agent": target_agent, "objective": objective},
                            )
                        )

                        child_runtime = AgentRuntime(
                            project_root=self.project_root,
                            provider=self.provider,
                            prompt_func=self.prompt_func,
                            event_listener=None,
                            agent_name=target_agent,
                        )

                        child_record = await child_runtime.execute_task(
                            user_request=objective,
                            cancellation_token=cancellation_token,
                            parent_chain=parent_chain
                            + [self.agent_def.name if self.agent_def else "root"],
                            parent_task_id=task_id,
                        )

                        if child_record.status == "failed":
                            self.event_bus.emit(
                                RuntimeEvent(
                                    task_id=task_id,
                                    session_id=session_id,
                                    event_type="delegation_failed",
                                    payload={"target_agent": target_agent},
                                )
                            )
                        else:
                            self.event_bus.emit(
                                RuntimeEvent(
                                    task_id=task_id,
                                    session_id=session_id,
                                    event_type="delegation_completed",
                                    payload={"target_agent": target_agent},
                                )
                            )

                        tool_res = {
                            "delegation_id": child_record.task_id,
                            "source_agent": self.agent_def.name if self.agent_def else "root",
                            "target_agent": target_agent,
                            "status": child_record.status,
                            "summary": child_record.summary,
                            "findings": child_record.verification.evidence
                            if child_record.verification
                            else [],
                            "evidence": child_record.verification.evidence
                            if child_record.verification
                            else [],
                            "confidence": 1.0,
                        }

                        res_str = json.dumps(tool_res, ensure_ascii=False)
                        messages.append(Message(role="assistant", content=response.message or ""))
                        messages.append(
                            Message(
                                role="user",
                                content=f"Tool execution response for [delegate_task]:\n{res_str}",
                            )
                        )
                        session.tool_calls.append(
                            {
                                "tool_name": "delegate_task",
                                "arguments": tool_args,
                                "output": res_str,
                                "success": child_record.status != "failed",
                            }
                        )
                        continue

                    change_state(AgentState.VALIDATING_TOOL_CALL)
                    tool = self.tools.get_tool(tool_name)
                    if not tool:
                        raise UnknownToolError(f"Tool not found: {tool_name}")

                    # Permission check
                    self.event_bus.emit(
                        RuntimeEvent(
                            task_id=task_id,
                            session_id=session_id,
                            event_type="tool_call_requested",
                            payload={"tool_name": tool_name, "arguments": tool_args},
                        )
                    )

                    decision = self.policy.evaluate(
                        tool_name,
                        tool.permission_level,
                        tool_args,
                    )
                    self.event_bus.emit(
                        RuntimeEvent(
                            task_id=task_id,
                            session_id=session_id,
                            event_type="permission_checked",
                            payload={"decision": decision.value},
                        )
                    )

                    if decision == PermissionDecision.DENY:
                        raise PermissionDeniedError(
                            f"Permission denied: {tool_name} "
                            f"(Permission: {tool.permission_level.value})"
                        )

                    if decision == PermissionDecision.REQUIRE_APPROVAL:
                        self.event_bus.emit(
                            RuntimeEvent(
                                task_id=task_id,
                                session_id=session_id,
                                event_type="approval_requested",
                                payload={"tool_name": tool_name},
                            )
                        )
                        approved = self.prompt_func(
                            f"wants to run {tool_name} with these arguments:\n"
                            f"{json.dumps(tool_args, ensure_ascii=False)}"
                        )
                        if not approved:
                            raise PermissionDeniedError("User rejected the tool approval request.")

                    # Enforce task-path scopes
                    from llmbrain.agent.context import PathResolver

                    resolver = PathResolver(self.project_root)
                    for k, v in tool_args.items():
                        is_path_key = any(x in k.lower() for x in ("path", "file", "dir"))
                        if isinstance(v, str):
                            path_candidate = v.split("::")[0].split(":")[0]
                            if (
                                path_candidate.startswith("/")
                                or ".." in path_candidate
                                or is_path_key
                            ):
                                try:
                                    resolver.resolve(path_candidate)
                                except (ValueError, PermissionError) as path_err:
                                    msg = f"Task-path scope violation: {path_err}"
                                    raise PermissionDeniedError(msg)

                    # Prevent auto-retry of side-effecting tools upon failure
                    is_side_effecting = tool.permission_level in (
                        PermissionLevel.WRITE,
                        PermissionLevel.SHELL,
                        PermissionLevel.EXECUTE_SAFE,
                        PermissionLevel.EXECUTE_NETWORK,
                        PermissionLevel.DESTRUCTIVE,
                    )
                    if is_side_effecting and call_signature in failed_side_effecting_calls:
                        raise ToolExecutionError(
                            f"Auto-retry of failed side-effecting tool '{tool_name}' is prohibited."
                        )

                    # Execute Tool
                    change_state(AgentState.EXECUTING_TOOL)
                    self.event_bus.emit(
                        RuntimeEvent(
                            task_id=task_id,
                            session_id=session_id,
                            event_type="tool_execution_started",
                            payload={"tool_name": tool_name},
                        )
                    )

                    tool_start = time.time()
                    tool_success = False
                    try:
                        # Call execution
                        import inspect

                        sig = inspect.signature(tool.execute)
                        if "context" in sig.parameters:
                            res = await tool.execute(tool_args, context=tool_ctx)
                        else:
                            res = await tool.execute(tool_args)
                        duration = (time.time() - tool_start) * 1000
                        tool_success = res.success

                        err_str = None
                        if res.error:
                            if isinstance(res.error, dict):
                                err_str = res.error.get("message") or json.dumps(res.error)
                            else:
                                err_str = str(res.error)

                        audit_rec = AuditRecord(
                            tool_name=tool_name,
                            arguments=tool_args,
                            permission_level=tool.permission_level.value,
                            status="executed" if res.success else "failed",
                            duration_ms=duration,
                            output_size=len(res.output) if res.output else 0,
                            error=err_str,
                        )
                        session.audit_records.append(audit_rec)

                        # Event completed / failed
                        if res.success:
                            self.event_bus.emit(
                                RuntimeEvent(
                                    task_id=task_id,
                                    session_id=session_id,
                                    event_type="tool_execution_completed",
                                    payload={"tool_name": tool_name},
                                )
                            )
                        else:
                            self.event_bus.emit(
                                RuntimeEvent(
                                    task_id=task_id,
                                    session_id=session_id,
                                    event_type="tool_execution_failed",
                                    payload={"tool_name": tool_name, "error": err_str},
                                )
                            )
                            if is_side_effecting:
                                failed_side_effecting_calls.add(call_signature)

                        from llmbrain.agent.context import OutputLimiter

                        max_bytes = tool_ctx.output_limits.get("max_bytes", 50_000)
                        limiter = OutputLimiter(max_bytes=max_bytes)
                        observation = res.output if res.success else f"Error: {err_str}"
                        observation, _, _, _ = limiter.limit(observation)

                    except Exception as e:
                        duration = (time.time() - tool_start) * 1000
                        if is_side_effecting:
                            failed_side_effecting_calls.add(call_signature)

                        audit_rec = AuditRecord(
                            tool_name=tool_name,
                            arguments=tool_args,
                            permission_level=tool.permission_level.value,
                            status="failed",
                            duration_ms=duration,
                            output_size=0,
                            error=str(e),
                        )
                        session.audit_records.append(audit_rec)

                        self.event_bus.emit(
                            RuntimeEvent(
                                task_id=task_id,
                                session_id=session_id,
                                event_type="tool_execution_failed",
                                payload={"tool_name": tool_name, "error": str(e)},
                            )
                        )

                        from llmbrain.agent.context import OutputLimiter

                        max_bytes = tool_ctx.output_limits.get("max_bytes", 50_000)
                        limiter = OutputLimiter(max_bytes=max_bytes)
                        observation, _, _, _ = limiter.limit(f"Tool exception: {e}")

                    # 4. Processing Observation
                    change_state(AgentState.PROCESSING_OBSERVATION)
                    session.tool_calls.append(
                        {
                            "tool_name": tool_name,
                            "arguments": tool_args,
                            "success": tool_success,
                            "output": observation,
                        }
                    )

                    # Update model message history
                    messages.append(
                        Message(
                            role="assistant",
                            content=json.dumps(parsed),
                        )
                    )
                    messages.append(
                        Message(
                            role="user",
                            content=f"Tool Output (Observation):\n{observation}",
                        )
                    )

        except CancellationError as e:
            error_summary = str(e)
            status = "cancelled"
        except Exception as e:
            error_summary = str(e)
            status = "failed"

        # 5. Persist run memory to SQLite
        change_state(AgentState.PERSISTING_MEMORY)
        try:
            self.memory.persist_task_run(
                task_id=task_id,
                request=user_request,
                summary=summary or error_summary or "Terminated with error.",
                status=status,
                decisions=[],
                commands=[],
                failures=[],
            )
            self.event_bus.emit(
                RuntimeEvent(
                    task_id=task_id,
                    session_id=session_id,
                    event_type="memory_persisted",
                )
            )
        except Exception:
            # log or handle failure
            pass

        # Final complete state
        if status == "completed":
            change_state(AgentState.COMPLETED)
            self.event_bus.emit(
                RuntimeEvent(
                    task_id=task_id,
                    session_id=session_id,
                    event_type="task_completed",
                )
            )
        elif status == "cancelled":
            change_state(AgentState.CANCELLED)
            self.event_bus.emit(
                RuntimeEvent(
                    task_id=task_id,
                    session_id=session_id,
                    event_type="task_cancelled",
                )
            )
        else:
            change_state(AgentState.FAILED)
            self.event_bus.emit(
                RuntimeEvent(
                    task_id=task_id,
                    session_id=session_id,
                    event_type="task_failed",
                    payload={"error": error_summary},
                )
            )

        record = TaskRecord(
            task_id=task_id,
            session_id=session_id,
            request=user_request,
            status=status,
            summary=summary or error_summary or "Failed run.",
            started_at=datetime.now(UTC).isoformat(),
            ended_at=datetime.now(UTC).isoformat(),
            error=error_summary,
            verification=verification_res,
        )

        # Write log file
        exec_log = {
            "task_id": task_id,
            "session_id": session_id,
            "user_request": user_request,
            "status": status,
            "state_history": [(s.value, t) for s, t in session.state_history],
            "total_input_tokens": session.total_input_tokens,
            "total_output_tokens": session.total_output_tokens,
            "audit_records": [r.model_dump() for r in session.audit_records],
            "tool_calls": session.tool_calls,
            "verification": verification_res.model_dump(),
            "summary": summary,
            "error": error_summary,
        }
        log_file = self.logs_dir / f"{task_id}.json"
        log_file.write_text(json.dumps(exec_log, indent=2, ensure_ascii=False), encoding="utf-8")

        return record
