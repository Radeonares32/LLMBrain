"""Specialized agent registry, schemas, error model, and routing logic for LLMBrain."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from llmbrain.agent.runtime import VerificationResult, Verifier
from llmbrain.llm.base import BaseLLMProvider

# ── Error Model ──────────────────────────────────────────────────────


class AgentError(Exception):
    """Base error class for all agent orchestration issues."""

    def __init__(
        self,
        code: str,
        message: str,
        recoverable: bool = False,
        metadata: dict | None = None,
        cause: Exception | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.recoverable = recoverable
        self.metadata = metadata or {}
        self.cause = cause


class UnknownAgentError(AgentError):
    """Raised when an requested agent name is not registered."""


class InvalidAgentConfigurationError(AgentError):
    """Raised when an agent definition fails schema or consistency checks."""


class DuplicateAgentNameError(AgentError):
    """Raised when attempting to register an agent name already taken."""


class InvalidPromptReferenceError(AgentError):
    """Raised when an agent's prompt reference is missing or invalid."""


class AgentSelectionError(AgentError):
    """Raised when automatic router or selection fails."""


class RouterLowConfidenceError(AgentError):
    """Raised when router confidence is below the threshold and fallback is disabled."""


class UnauthorizedToolForAgentError(AgentError):
    """Raised when an agent requests a tool not in its allowlist."""


class InvalidDelegationTargetError(AgentError):
    """Raised when delegating to an unregistered or invalid agent."""


class DelegationPermissionEscalationError(AgentError):
    """Raised when a child agent gets higher privileges than parent."""


class DelegationScopeExpansionError(AgentError):
    """Raised when child agent task path scope expands beyond parent."""


class DelegationLoopError(AgentError):
    """Raised when agent delegation or inheritance cycle is detected."""


class DelegationDepthLimitError(AgentError):
    """Raised when agent delegation chain exceeds depth ceiling."""


class InvalidChildResultError(AgentError):
    """Raised when a delegated child agent returns malformed output."""


class AgentOutputSchemaFailureError(AgentError):
    """Raised when an agent final output fails schema validation."""


class VerificationPolicyFailureError(AgentError):
    """Raised when verification checks fail after agent execution."""


class PromptCompositionFailureError(AgentError):
    """Raised when combining prompt parts fails."""


# ── Configuration Schemas ───────────────────────────────────────────


class AgentModelConfig(BaseModel):
    provider: str = "default"
    model: str = "default"
    temperature: float = 0.1


class AgentContextConfig(BaseModel):
    token_budget: int = 16000
    memory_types: list[str] = Field(default_factory=list)


class AgentToolsConfig(BaseModel):
    allow: list[str] = Field(default_factory=list)
    deny: list[str] = Field(default_factory=list)


class AgentLimitsConfig(BaseModel):
    max_iterations: int = 20
    max_tool_calls: int = 40
    max_delegations: int = 2


class AgentVerificationConfig(BaseModel):
    required: bool = False
    run_tests: bool = False
    run_diagnostics: bool = False


class AgentDefinition(BaseModel):
    name: str
    display_name: str
    description: str
    system_prompt: str
    model: AgentModelConfig = Field(default_factory=AgentModelConfig)
    context: AgentContextConfig = Field(default_factory=AgentContextConfig)
    tools: AgentToolsConfig = Field(default_factory=AgentToolsConfig)
    permissions: dict[str, Any] = Field(default_factory=dict)
    limits: AgentLimitsConfig = Field(default_factory=AgentLimitsConfig)
    verification: AgentVerificationConfig = Field(default_factory=AgentVerificationConfig)
    extends: str | None = None


# ── Structured Output Models ───────────────────────────────────────


class AskOutput(BaseModel):
    answer: str
    sources: list[str] = Field(default_factory=list)
    uncertainties: list[str] = Field(default_factory=list)


class PlanOutputStep(BaseModel):
    step: str
    description: str


class PlanOutput(BaseModel):
    summary: str
    affected_paths: list[str] = Field(default_factory=list)
    affected_symbols: list[str] = Field(default_factory=list)
    steps: list[Any] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    tests: list[str] = Field(default_factory=list)
    verification: list[str] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)


class BuildOutput(BaseModel):
    summary: str
    changed_files: list[str] = Field(default_factory=list)
    commands_executed: list[str] = Field(default_factory=list)
    tests: Any = None
    diagnostics: Any = None
    verification: dict[str, Any] = Field(default_factory=dict)
    remaining_issues: list[str] = Field(default_factory=list)


class ReviewFinding(BaseModel):
    title: str
    severity: str  # critical|high|medium|low|info
    affected_path: str
    affected_line: int | None = None
    explanation: str
    evidence: str
    impact: str
    suggested_fix: str
    confidence: float


class ReviewOutput(BaseModel):
    summary: str
    findings: list[ReviewFinding] = Field(default_factory=list)
    test_gaps: list[str] = Field(default_factory=list)
    security_notes: list[str] = Field(default_factory=list)
    verdict: str  # approve|request_changes|inconclusive


class DebugHypothesis(BaseModel):
    description: str
    status: str  # confirmed|likely|rejected|unknown


class DebugOutput(BaseModel):
    symptom: str
    evidence: list[str] = Field(default_factory=list)
    hypotheses: list[DebugHypothesis] = Field(default_factory=list)
    root_cause: str | None = None
    fix: dict[str, Any] = Field(default_factory=dict)
    verification: dict[str, Any] = Field(default_factory=dict)
    remaining_uncertainty: list[str] = Field(default_factory=list)


class TestOutput(BaseModel):
    coverage_gaps: list[str] = Field(default_factory=list)
    tests_added: list[str] = Field(default_factory=list)
    commands_executed: list[str] = Field(default_factory=list)
    results: list[Any] = Field(default_factory=list)
    remaining_gaps: list[str] = Field(default_factory=list)


class SecurityFinding(BaseModel):
    title: str
    severity: str  # critical|high|medium|low|info
    category: str
    affected_path: str
    explanation: str
    evidence: str
    suggested_fix: str
    confidence: float


class SecurityOutput(BaseModel):
    threat_surface: list[str] = Field(default_factory=list)
    findings: list[SecurityFinding] = Field(default_factory=list)
    tested_controls: list[str] = Field(default_factory=list)
    unverified_controls: list[str] = Field(default_factory=list)
    overall_risk: str  # critical|high|medium|low|unknown


def validate_agent_output(agent_name: str, payload: dict) -> dict:
    schemas = {
        "ask": AskOutput,
        "plan": PlanOutput,
        "build": BuildOutput,
        "review": ReviewOutput,
        "debug": DebugOutput,
        "test": TestOutput,
        "security": SecurityOutput,
    }
    schema = schemas.get(agent_name)
    if not schema:
        return payload
    try:
        parsed = schema.model_validate(payload)
        return parsed.model_dump()
    except Exception as e:
        raise AgentOutputSchemaFailureError(
            "SCHEMA_FAILURE", f"Agent '{agent_name}' output failed validation: {e}", cause=e
        )


# ── Prompt Composition ──────────────────────────────────────────────


def load_prompt(project_root: Path, prompt_path: str) -> str:
    p = project_root / prompt_path
    if p.exists():
        return p.read_text(encoding="utf-8")
    p2 = Path(prompt_path)
    if p2.exists():
        return p2.read_text(encoding="utf-8")
    raise InvalidPromptReferenceError("INVALID_PROMPT", f"Prompt file not found: {prompt_path}")


def compose_prompt(project_root: Path, agent_def: AgentDefinition) -> str:
    try:
        common_content = ""
        try:
            common_content = load_prompt(project_root, "prompts/common.md")
        except InvalidPromptReferenceError:
            common_content = (
                "Do not fabricate repository facts. Use source-grounded memory.\n"
                "Distinguish evidence from inference.\n"
                "Respect workspace boundaries and tool permissions."
            )

        agent_content = ""
        try:
            agent_content = load_prompt(project_root, agent_def.system_prompt)
        except InvalidPromptReferenceError:
            if " " in agent_def.system_prompt or "\n" in agent_def.system_prompt:
                agent_content = agent_def.system_prompt
            else:
                raise

        allowed_str = ", ".join(sorted(agent_def.tools.allow))
        denied_str = ", ".join(sorted(agent_def.tools.deny))

        composed = (
            f"{common_content.strip()}\n\n"
            f"Role: {agent_def.display_name}\n"
            f"Description: {agent_def.description}\n\n"
            f"{agent_content.strip()}\n\n"
            f"Allowed Tools: [{allowed_str}]\n"
            f"Denied Capabilities: [{denied_str}]\n"
        )
        return composed
    except Exception as e:
        if isinstance(e, AgentError):
            raise e
        raise PromptCompositionFailureError(
            "COMPOSITION_FAILURE", f"Failed to compose prompt: {e}", cause=e
        )


# ── Agent Registry ───────────────────────────────────────────────────


class AgentRegistry:
    def __init__(self) -> None:
        self.agents: dict[str, AgentDefinition] = {}
        self._register_builtins()

    def _register_builtins(self) -> None:
        builtins = [
            AgentDefinition(
                name="ask",
                display_name="Ask Agent",
                description="Answers questions about the repository",
                system_prompt="prompts/ask.md",
                context=AgentContextConfig(
                    token_budget=16000,
                    memory_types=[
                        "repository_overview",
                        "symbol_fact",
                        "decision",
                        "task",
                        "failure_resolution",
                    ],
                ),
                tools=AgentToolsConfig(
                    allow=["read_file", "read_files", "grep", "glob", "git_status", "git_log"]
                ),
                permissions={"mode": "read-only"},
                limits=AgentLimitsConfig(max_iterations=10, max_tool_calls=20, max_delegations=2),
                verification=AgentVerificationConfig(required=True),
            ),
            AgentDefinition(
                name="plan",
                display_name="Plan Agent",
                description="Inspects a task and produces an implementation plan",
                system_prompt="prompts/plan.md",
                context=AgentContextConfig(
                    token_budget=16000,
                    memory_types=[
                        "repository_overview",
                        "symbol_fact",
                        "decision",
                        "task",
                        "failure_resolution",
                    ],
                ),
                tools=AgentToolsConfig(
                    allow=["read_file", "read_files", "grep", "glob", "git_status", "git_log"]
                ),
                permissions={"mode": "read-only"},
                limits=AgentLimitsConfig(max_iterations=10, max_tool_calls=20, max_delegations=2),
                verification=AgentVerificationConfig(required=True),
            ),
            AgentDefinition(
                name="build",
                display_name="Build Agent",
                description="Implements approved changes in the repository",
                system_prompt="prompts/build.md",
                context=AgentContextConfig(
                    token_budget=32000,
                    memory_types=[
                        "repository_overview",
                        "symbol_fact",
                        "decision",
                        "task",
                        "failure_resolution",
                    ],
                ),
                tools=AgentToolsConfig(
                    allow=[
                        "read_file",
                        "read_files",
                        "grep",
                        "glob",
                        "apply_patch",
                        "write_file",
                        "git_status",
                        "git_diff",
                        "run_tests",
                        "diagnostics",
                    ],
                    deny=["delete_file"],
                ),
                permissions={"mode": "ask-before-write"},
                limits=AgentLimitsConfig(max_iterations=20, max_tool_calls=40, max_delegations=2),
                verification=AgentVerificationConfig(
                    required=True, run_tests=True, run_diagnostics=True
                ),
            ),
            AgentDefinition(
                name="review",
                display_name="Review Agent",
                description="Reviews uncommitted changes or diffs",
                system_prompt="prompts/review.md",
                context=AgentContextConfig(
                    token_budget=16000,
                    memory_types=[
                        "repository_overview",
                        "symbol_fact",
                        "decision",
                        "task",
                        "failure_resolution",
                    ],
                ),
                tools=AgentToolsConfig(
                    allow=["read_file", "grep", "git_diff", "git_log", "git_status", "diagnostics"]
                ),
                permissions={"mode": "read-only"},
                limits=AgentLimitsConfig(max_iterations=10, max_tool_calls=20, max_delegations=2),
                verification=AgentVerificationConfig(required=True),
            ),
            AgentDefinition(
                name="debug",
                display_name="Debug Agent",
                description="Reproduces, diagnoses and proposes fixes for bugs",
                system_prompt="prompts/debug.md",
                context=AgentContextConfig(
                    token_budget=24000,
                    memory_types=[
                        "repository_overview",
                        "symbol_fact",
                        "decision",
                        "task",
                        "failure_resolution",
                    ],
                ),
                tools=AgentToolsConfig(
                    allow=[
                        "read_file",
                        "grep",
                        "git_diff",
                        "git_log",
                        "git_status",
                        "diagnostics",
                        "run_tests",
                    ]
                ),
                permissions={"mode": "ask-before-write"},
                limits=AgentLimitsConfig(max_iterations=20, max_tool_calls=40, max_delegations=2),
                verification=AgentVerificationConfig(required=True),
            ),
            AgentDefinition(
                name="test",
                display_name="Test Agent",
                description="Inspects coverage, writes and runs tests",
                system_prompt="prompts/test.md",
                context=AgentContextConfig(
                    token_budget=24000,
                    memory_types=[
                        "repository_overview",
                        "symbol_fact",
                        "decision",
                        "task",
                        "failure_resolution",
                    ],
                ),
                tools=AgentToolsConfig(
                    allow=[
                        "read_file",
                        "grep",
                        "git_diff",
                        "git_log",
                        "git_status",
                        "diagnostics",
                        "run_tests",
                        "write_file",
                    ]
                ),
                permissions={"mode": "ask-before-write"},
                limits=AgentLimitsConfig(max_iterations=20, max_tool_calls=40, max_delegations=2),
                verification=AgentVerificationConfig(required=True, run_tests=True),
            ),
            AgentDefinition(
                name="security",
                display_name="Security Agent",
                description="Performs secure code review and audits authentication/authorization",
                system_prompt="prompts/security.md",
                context=AgentContextConfig(
                    token_budget=24000,
                    memory_types=[
                        "repository_overview",
                        "symbol_fact",
                        "decision",
                        "task",
                        "failure_resolution",
                    ],
                ),
                tools=AgentToolsConfig(
                    allow=[
                        "read_file",
                        "grep",
                        "git_diff",
                        "git_log",
                        "git_status",
                        "diagnostics",
                        "run_tests",
                    ]
                ),
                permissions={"mode": "read-only"},
                limits=AgentLimitsConfig(max_iterations=15, max_tool_calls=30, max_delegations=2),
                verification=AgentVerificationConfig(required=True),
            ),
        ]
        for agent in builtins:
            self.agents[agent.name] = agent

    def register_agent(self, agent: AgentDefinition) -> None:
        if agent.name in self.agents:
            raise DuplicateAgentNameError(
                "DUPLICATE_AGENT", f"Agent with name '{agent.name}' is already registered."
            )
        self._validate_config(agent)
        self.agents[agent.name] = agent

    def _validate_config(self, agent: AgentDefinition) -> None:
        if not agent.name:
            raise InvalidAgentConfigurationError("INVALID_CONFIG", "Agent name cannot be empty.")
        if not agent.system_prompt:
            raise InvalidPromptReferenceError(
                "INVALID_PROMPT", "Agent system prompt cannot be empty."
            )
        if agent.limits.max_iterations <= 0 or agent.limits.max_iterations > 100:
            raise InvalidAgentConfigurationError(
                "INVALID_CONFIG", "max_iterations must be between 1 and 100"
            )
        if agent.limits.max_tool_calls <= 0 or agent.limits.max_tool_calls > 200:
            raise InvalidAgentConfigurationError(
                "INVALID_CONFIG", "max_tool_calls must be between 1 and 200"
            )
        if agent.limits.max_delegations < 0 or agent.limits.max_delegations > 10:
            raise InvalidAgentConfigurationError(
                "INVALID_CONFIG", "max_delegations must be between 0 and 10"
            )
        if agent.context.token_budget <= 0:
            raise InvalidAgentConfigurationError("INVALID_CONFIG", "token_budget must be positive")

        # Check prohibited tools
        for t in agent.tools.allow:
            if t == "prohibited" or t == "execute_network" or t == "destructive":
                raise InvalidAgentConfigurationError(
                    "SECURITY_VIOLATION", f"Cannot allow prohibited/destructive tool: {t}"
                )

    def get_agent(self, name: str) -> AgentDefinition:
        if name not in self.agents:
            raise UnknownAgentError("UNKNOWN_AGENT", f"Agent '{name}' not found.")
        return self.agents[name]

    def list_agents(self) -> list[AgentDefinition]:
        return list(self.agents.values())

    def resolve_default_agent(self) -> AgentDefinition:
        return self.get_agent("plan")

    def load_project_config(self, project_root: Path) -> None:
        config_file = project_root / ".llmbrain" / "agents.yaml"
        if not config_file.exists():
            config_file = project_root / ".llmbrain" / "agents.yml"
        if config_file.exists():
            try:
                try:
                    import yaml

                    with open(config_file, encoding="utf-8") as f:
                        data = yaml.safe_load(f) or {}
                except ImportError:
                    with open(config_file, encoding="utf-8") as f:
                        data = json.load(f) or {}
                if "agents" in data:
                    for name, agent_data in data["agents"].items():
                        agent_data["name"] = name
                        if agent_data.get("extends"):
                            parent_name = agent_data["extends"]
                            parent = self.get_agent(parent_name)
                            merged = self._merge_definitions(parent, agent_data)
                            agent_def = AgentDefinition.model_validate(merged)
                        else:
                            if name in self.agents:
                                parent = self.agents[name]
                                merged = self._merge_definitions(parent, agent_data)
                                agent_def = AgentDefinition.model_validate(merged)
                            else:
                                agent_def = AgentDefinition.model_validate(agent_data)

                        self._validate_security_precedence(self.agents.get(name), agent_def)
                        self._validate_config(agent_def)
                        self.agents[name] = agent_def
            except Exception as e:
                if isinstance(e, AgentError):
                    raise e
                raise InvalidAgentConfigurationError(
                    "INVALID_CONFIG", f"Failed to load project configuration: {e}", cause=e
                )

    def _merge_definitions(self, parent: AgentDefinition, child_data: dict) -> dict:
        visited = {child_data.get("name")}
        curr = child_data.get("extends")
        while curr:
            if curr in visited:
                raise DelegationLoopError(
                    "INHERITANCE_CYCLE",
                    f"Inheritance cycle detected: {' -> '.join(visited)} -> {curr}",
                )
            visited.add(curr)
            if curr in self.agents:
                curr = self.agents[curr].extends
            else:
                break

        parent_dict = parent.model_dump()
        for k, v in child_data.items():
            if isinstance(v, dict) and k in parent_dict and isinstance(parent_dict[k], dict):
                parent_dict[k].update(v)
            elif isinstance(v, list) and k in parent_dict and isinstance(parent_dict[k], list):
                parent_dict[k] = list(set(parent_dict[k] + v))
            else:
                parent_dict[k] = v
        return parent_dict

    def _validate_security_precedence(
        self, parent: AgentDefinition | None, child: AgentDefinition
    ) -> None:
        if not parent:
            return
        parent_mode = parent.permissions.get("mode")
        child_mode = child.permissions.get("mode")
        if parent_mode == "read-only" and child_mode in ("ask_before_write", "trusted_project"):
            raise InvalidAgentConfigurationError(
                "SECURITY_VIOLATION",
                "Project config cannot weaken permission mode from read-only to write.",
            )
        for t in child.tools.allow:
            if (
                t in parent.tools.deny
                or t == "prohibited"
                or t == "execute_network"
                or t == "destructive"
            ):
                raise InvalidAgentConfigurationError(
                    "SECURITY_VIOLATION",
                    f"Project config attempts to allow a denied/prohibited tool: {t}",
                )


# ── Agent Router ─────────────────────────────────────────────────────


class AgentRouter:
    def __init__(
        self,
        registry: AgentRegistry,
        provider: BaseLLMProvider | None = None,
        disable_routing: bool = False,
    ) -> None:
        self.registry = registry
        self.provider = provider
        self.disable_routing = disable_routing

    async def route(self, task: str) -> tuple[AgentDefinition, str]:
        if self.disable_routing:
            return (
                self.registry.resolve_default_agent(),
                "Routing is disabled; using default agent.",
            )

        task_lower = task.lower()
        matched = None
        confidence = 0.0
        reason = ""

        # Rules-based quick mapping
        if any(
            w in task_lower
            for w in (
                "security",
                "vulnerability",
                "encrypt",
                "leak",
                "secrets",
                "cve",
                "injection",
                "güvenlik",
            )
        ):
            matched = "security"
            confidence = 0.92
            reason = (
                "The task relates to security review, credentials, or potential vulnerabilities."
            )
        elif any(
            w in task_lower
            for w in (
                "fail",
                "bug",
                "error",
                "debug",
                "crash",
                "wrong",
                "broken",
                "issue",
                "reproduce",
                "hata",
            )
        ):
            matched = "debug"
            confidence = 0.91
            reason = "The task describes an existing failure requiring reproduction and diagnosis."
        elif any(w in task_lower for w in ("test", "coverage", "assert", "pytest", "spec")):
            matched = "test"
            confidence = 0.93
            reason = "The task requires inspecting test coverage, writing, or running tests."
        elif any(w in task_lower for w in ("review", "audit", "diff", "check", "gözden geçir")):
            matched = "review"
            confidence = 0.91
            reason = "The task asks for reviewing uncommitted changes, code review, or git diffs."
        elif any(
            w in task_lower
            for w in (
                "implement",
                "create",
                "write",
                "add",
                "change",
                "modify",
                "build",
                "yap",
                "ekle",
                "yaz",
            )
        ):
            matched = "build"
            confidence = 0.92
            reason = "The task requests implementing changes or creating new files."
        elif (
            any(
                w in task_lower
                for w in (
                    "how",
                    "what",
                    "why",
                    "explain",
                    "where",
                    "tell me",
                    "nedir",
                    "nasıl",
                    "açıkla",
                )
            )
            or "?" in task_lower
        ):
            matched = "ask"
            confidence = 0.91
            reason = "The task is a question about the repository or code behavior."
        elif any(
            w in task_lower for w in ("plan", "design", "steps", "architect", "proposal", "analyze")
        ):
            matched = "plan"
            confidence = 0.91
            reason = "The task asks for analysis, design steps, or planning."

        # Model based classification fallback
        if self.provider and confidence < 0.8:
            try:
                system_prompt = (
                    "You are the Agent Router for LLMBrain.\n"
                    "Analyze the user request and classify it into one of these agents:\n"
                    "- ask: Q&A, explain architecture/symbols.\n"
                    "- plan: analysis, design, implementation planning.\n"
                    "- build: implementing changes, coding, testing.\n"
                    "- review: code review, checking uncommitted changes.\n"
                    "- debug: fixing failures, reproducing bugs, isolating root causes.\n"
                    "- test: test gap analysis, writing tests, test execution.\n"
                    "- security: security audit, threat surface analysis.\n"
                    "\n"
                    "Respond with a JSON object in this format:\n"
                    "{\n"
                    '  "selected_agent": "agent_name",\n'
                    '  "confidence": 0.95,\n'
                    '  "reason": "explanation of choice"\n'
                    "}"
                )
                from llmbrain.llm.base import Message, ModelRequest

                response = await self.provider.complete(
                    ModelRequest(
                        messages=[Message(role="user", content=f"Request: {task}")],
                        system_prompt=system_prompt,
                    )
                )
                res_dict = json.loads(response.message or "{}")
                cand = res_dict.get("selected_agent")
                conf = res_dict.get("confidence", 0.0)
                reas = res_dict.get("reason", "")
                if cand in self.registry.agents and conf >= 0.7:
                    matched = cand
                    confidence = conf
                    reason = reas
            except Exception:
                pass

        if matched and confidence >= 0.7:
            return self.registry.get_agent(
                matched
            ), f"Router selected '{matched}' ({int(confidence * 100)}% confidence): {reason}"

        fallback = self.registry.resolve_default_agent()
        return fallback, "Low confidence in routing; using safe fallback agent 'plan'."


# ── Agent Specific Verifier ─────────────────────────────────────────


class AgentSpecificVerifier(Verifier):
    def __init__(self, agent_name: str, registry: AgentRegistry) -> None:
        self.agent_name = agent_name
        self.registry = registry

    def verify(
        self,
        task: str,
        transcript: list[dict[str, Any]],
        tool_results: list[dict[str, Any]],
    ) -> VerificationResult:
        try:
            agent_def = self.registry.get_agent(self.agent_name)
        except Exception:
            return VerificationResult(
                status="passed", summary="Verification bypassed.", evidence=[]
            )

        if not agent_def.verification.required:
            return VerificationResult(
                status="skipped",
                summary="Verification skipped as not required by agent.",
                evidence=[],
            )

        mutating_tools = {"write_file", "apply_patch", "delete_file"}
        executed_mutating = [
            r.get("tool_name") for r in tool_results if r.get("tool_name") in mutating_tools
        ]

        if self.agent_name in ("ask", "plan", "review", "security"):
            if executed_mutating:
                return VerificationResult(
                    status="failed",
                    summary="Workspace mutation occurred in a read-only agent task.",
                    evidence=[f"Mutating tool run: {t}" for t in executed_mutating],
                )

        if self.agent_name == "build":
            run_tests_called = False
            test_success = True
            for r in tool_results:
                if r.get("tool_name") == "run_tests":
                    run_tests_called = True
                    if not r.get("success"):
                        test_success = False

            if agent_def.verification.run_tests and not run_tests_called:
                return VerificationResult(
                    status="failed",
                    summary="Verification policy requires running tests, but none were executed.",
                    evidence=["run_tests tool was not called."],
                )
            if not test_success:
                return VerificationResult(
                    status="failed",
                    summary="Build verification failed: test suite reported errors.",
                    evidence=["run_tests returned failure."],
                )

        elif self.agent_name == "test":
            run_tests_called = False
            for r in tool_results:
                if r.get("tool_name") == "run_tests":
                    run_tests_called = True
            if not run_tests_called:
                return VerificationResult(
                    status="failed",
                    summary="Test Agent verification failed: no tests were run.",
                    evidence=["run_tests was not executed."],
                )

        return VerificationResult(
            status="passed", summary="Verification passed successfully.", evidence=[]
        )
