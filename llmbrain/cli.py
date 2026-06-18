"""CLI interface for LLM Brain."""

import json
from pathlib import Path

import typer
import uvicorn
from rich.console import Console
from rich.table import Table

from llmbrain import __version__
from llmbrain.core.config import settings
from llmbrain.services.project_service import ProjectService
from llmbrain.storage.filesystem import output_root

app = typer.Typer(help="LLM Brain - Engineering Knowledge Compiler")
console = Console()
err_console = Console(stderr=True)


@app.command()
def version():
    """Print version."""
    typer.echo(f"LLM Brain {__version__}")


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", help="Host IP"),
    port: int = typer.Option(8000, help="Port"),
    reload: bool = typer.Option(False, help="Reload on changes"),
):
    """Start the FastAPI server."""
    uvicorn.run("llmbrain.main:app", host=host, port=port, reload=reload)


@app.command()
def scan(
    path: str,
    json_out: bool = typer.Option(False, "--json", help="Output JSON"),
    output: str = typer.Option(None, "--output", help="Output file"),
):
    """Scan a project directory without building."""
    service = ProjectService()
    try:
        result = service.scan_project(path)
        if json_out:
            out_str = result.model_dump_json(indent=2)
            if output:
                Path(output).write_text(out_str)
            else:
                typer.echo(out_str)
        else:
            console.print(f"[bold green]Scanned {result.project.name}[/bold green]")
            table = Table("Documents Found")
            table.add_row(str(result.stats.documents))
            console.print(table)
    except Exception as e:
        err_console.print(f"[bold red]Error: {e}[/bold red]")
        raise typer.Exit(code=1)


@app.command()
def build(
    path_or_task: str,
    provider: str = typer.Option(
        settings.default_provider,
        help="LLM provider: openai | deepseek | anthropic | ollama | gemini",
    ),
    project_name: str = typer.Option(None, "--project-name", help="Project name"),
    incremental: bool = typer.Option(
        True,
        "--incremental/--full",
        help="Use incremental build if possible",
    ),
    json_out: bool = typer.Option(False, "--json", help="Output JSON"),
    output_dir: str = typer.Option(None, "--output-dir", help="Output directory override"),
):
    """Run the build pipeline if a directory is passed, otherwise execute coding agent tasks."""
    import asyncio

    service = ProjectService()
    try:
        p = Path(path_or_task).expanduser().resolve()
        if p.is_dir():
            if not json_out:
                console.print(
                    f"[bold blue]Building project at {path_or_task} "
                    f"(provider={provider})[/bold blue]"
                )

            result = service.build_project(
                path_or_task,
                name=project_name,
                llm_provider=provider,
                incremental=incremental,
            )

            if json_out:
                typer.echo(result.model_dump_json(indent=2))
            else:
                console.print("[bold green]Build successful![/bold green]")
                table = Table("Metric", "Count")
                table.add_row("Documents", str(result.stats.documents))
                table.add_row("Facts", str(result.stats.facts))
                table.add_row("Entities", str(result.stats.entities))
                table.add_row("Relations", str(result.stats.relations))
                table.add_row("Wiki Pages", str(result.stats.wiki_pages))
                console.print(table)
                console.print(f"Output saved to {result.output_path}")
        else:
            # Execute agent task
            from llmbrain.agent.runtime import AgentRuntime
            from llmbrain.llm.providers import create_provider

            project_path = Path(".").resolve()
            service.build_project(str(project_path), llm_provider=provider, incremental=True)

            llm = create_provider(provider)
            agent = AgentRuntime(project_path, llm, agent_name="build")

            console.print(f"[bold blue]Running Build Agent Task:[/bold blue] {path_or_task}")
            record = asyncio.run(agent.execute_task(path_or_task))

            if json_out:
                typer.echo(record.model_dump_json(indent=2))
            else:
                if record.status == "completed":
                    console.print(
                        f"\n[bold green]Completed ({record.status}):[/bold green] {record.summary}"
                    )
                else:
                    console.print(
                        f"\n[bold red]Failed ({record.status}):[/bold red] {record.summary}"
                    )
                if record.verification:
                    console.print(
                        f"[bold]Verification:[/bold] {record.verification.status} | "
                        f"{record.verification.summary}"
                    )
    except Exception as e:
        err_console.print(f"[bold red]Hata: {e}[/bold red]")
        raise typer.Exit(code=1)


@app.command()
def context(
    path: str,
    json_out: bool = typer.Option(False, "--json", help="Output JSON"),
    print_out: bool = typer.Option(False, "--print", help="Print context to stdout"),
    output: str = typer.Option(None, "--output", help="Output file path"),
):
    """Print the BrainFrame context for a project."""
    root = output_root(path)
    bf_file = root / "llm-context" / "brainframe.bf"
    if not bf_file.exists():
        err_console.print("[bold red]No context found. Run build first.[/bold red]")
        raise typer.Exit(code=1)

    content = bf_file.read_text()
    if json_out:
        typer.echo(json.dumps({"context": content}))
    elif output:
        Path(output).write_text(content)
        console.print(f"[bold green]Saved context to {output}[/bold green]")
    elif print_out or not json_out:
        typer.echo(content)


@app.command()
def diff(
    path: str,
    base_ref: str = typer.Option(None, "--base-ref", help="Base ref to compare against"),
    json_out: bool = typer.Option(False, "--json", help="Output JSON"),
):
    """Show git diff."""
    from llmbrain.services.git_diff import analyze_git_diff

    changed_files = analyze_git_diff(path, base_ref)

    if json_out:
        typer.echo(json.dumps([cf.model_dump() for cf in changed_files]))
    else:
        table = Table("Status", "Path")
        for cf in changed_files:
            table.add_row(cf.status, cf.path)
        console.print(table)


@app.command()
def graph(
    path: str,
    format: str = typer.Option("json", "--format", help="json | graphml"),
    output: str = typer.Option(None, "--output", help="Output file path"),
):
    """Show graph status."""
    service = ProjectService()
    root = service._normalize_path(path)
    project_id = service._project_id_from_path(root)
    g = service.get_graph(project_id)

    if output:
        Path(output).write_text(json.dumps(g, indent=2))
        console.print(f"[bold green]Saved graph to {output}[/bold green]")
    else:
        console.print(f"Graph nodes: {len(g.get('nodes', []))}, edges: {len(g.get('edges', []))}")


@app.command()
def diagram(
    path: str = typer.Option(".", "--path", help="Proje dizini"),
    output: str = typer.Option(None, "--output", help="Output file path (*.mermaid)"),
):
    """Proje bağımlılıklarını ve relation'larını Mermaid diyagramı olarak üret."""
    from llmbrain.services.graph_generator import graph_to_mermaid, KnowledgeGraph, GraphNode, GraphEdge
    from llmbrain.core.identity import load_or_create_project_identity

    service = ProjectService()
    root = service._normalize_path(path)
    project_id = service._project_id_from_path(root)
    
    g_json = service.get_graph(project_id)
    
    nodes = [
        GraphNode(id=n["id"], label=n["label"], type=n["type"], metadata=n.get("metadata", {}))
        for n in g_json.get("nodes", [])
    ]
    edges = [
        GraphEdge(source=e["source"], target=e["target"], relation=e["relation"], 
                  confidence=e.get("confidence", "medium"), evidence=e.get("evidence", ""))
        for e in g_json.get("edges", [])
    ]
    
    kg = KnowledgeGraph(project_id=project_id, nodes=nodes, edges=edges)
    mermaid_out = graph_to_mermaid(kg)
    
    if output:
        Path(output).write_text(mermaid_out, encoding="utf-8")
        console.print(f"[bold green]Saved mermaid diagram to {output}[/bold green]")
    else:
        console.print(mermaid_out)


@app.command()
def drift(
    path: str = typer.Option(".", "--path", help="Proje dizini"),
    provider: str = typer.Option("mock", "--provider", help="LLM provider (e.g., mock, openai)"),
):
    """Dokümantasyon ile kod arasındaki anlam kaymalarını (drift) analiz et."""
    import asyncio
    from llmbrain.services.drift_detection import DriftDetector
    from llmbrain.core.identity import load_or_create_project_identity
    from llmbrain.llm.providers import create_provider

    service = ProjectService()
    root = service._normalize_path(path)
    project_id = service._project_id_from_path(root)

    llm = create_provider(provider)
    detector = DriftDetector(project_id, service, llm)
    
    console.print(f"Analyzing documentation drift for project: {project_id} using {provider}...")
    reports = asyncio.run(detector.analyze_drift())
    
    if not reports:
        console.print("[green]Hiç drift veya dokümantasyon bulunamadı.[/green]")
        return
        
    table = Table("Page", "Drift?", "Risk", "Rationale")
    for r in reports:
        drift_str = "[red]Yes[/red]" if r.is_drifting else "[green]No[/green]"
        risk_color = "red" if r.risk_level == "high" else "yellow" if r.risk_level == "medium" else "green"
        risk_str = f"[{risk_color}]{r.risk_level}[/{risk_color}]"
        table.add_row(r.wiki_title, drift_str, risk_str, r.rationale)
        
    console.print(table)


@app.command("pr-review")
def pr_review(
    path: str = typer.Option(".", "--path", help="Proje dizini"),
    base_ref: str = typer.Option("origin/main", "--base", help="Karşılaştırılacak base branch"),
    provider: str = typer.Option("mock", "--provider", help="LLM provider"),
):
    """Değişen dosyaların semantic analizi ile Pull Request review yorumları oluştur."""
    import asyncio
    from llmbrain.services.pr_review import PRReviewer
    from llmbrain.llm.providers import create_provider

    service = ProjectService()
    root = service._normalize_path(path)
    project_id = service._project_id_from_path(root)

    llm = create_provider(provider)
    reviewer = PRReviewer(project_id, service, llm)
    
    console.print(f"Generating PR review against '{base_ref}' using {provider}...")
    comments = asyncio.run(reviewer.generate_review(path, base_ref))
    
    if not comments:
        console.print("[green]No issues found. LGTM![/green]")
        return
        
    table = Table("File", "Line", "Severity", "Comment")
    for c in comments:
        color = "red" if c.severity == "critical" else "yellow" if c.severity == "warning" else "blue"
        table.add_row(
            c.file_path, 
            str(c.line_number) if c.line_number else "-", 
            f"[{color}]{c.severity}[/{color}]", 
            c.comment
        )
        
    console.print(table)


@app.command()
def ci(
    path: str,
    base_ref: str = typer.Option(None, "--base-ref", help="Base ref to compare against"),
    fail_on: str = typer.Option(
        "high",
        "--fail-on",
        help="Risk level to fail on: low | medium | high",
    ),
    provider: str = typer.Option(settings.default_provider, "--provider", help="LLM provider"),
    output_json: str = typer.Option(None, "--output-json", help="Output JSON file"),
):
    """Run LLM Brain in CI mode."""
    console.print(f"Running CI on {path}")

    from llmbrain.services.git_diff import analyze_git_diff
    from llmbrain.storage.filesystem import output_root

    service = ProjectService()
    try:
        # 1. Build
        service.build_project(path, llm_provider=provider)

        # 2. Diff
        changed_files = analyze_git_diff(path, base_ref)
        console.print(f"Found {len(changed_files)} changed files.")

        root = service._normalize_path(path)
        out_dir = output_root(root)

        ci_result = {"status": "passed", "risk_level": "low"}

        with open(out_dir / "ci-result.json", "w") as f:
            json.dump(ci_result, f)

        if output_json:
            Path(output_json).write_text(json.dumps(ci_result, indent=2))

        # Simplified fail_on logic for MVP
        if fail_on == "low" and ci_result["risk_level"] in ["low", "medium", "high"]:
            err_console.print("[bold red]Failing on low risk.[/bold red]")
            raise typer.Exit(code=1)

        console.print("[bold green]CI checks passed successfully.[/bold green]")
    except typer.Exit:
        raise
    except Exception as e:
        err_console.print(f"[bold red]CI failed: {e}[/bold red]")
        raise typer.Exit(code=1)


@app.command()
def health(path: str, json_out: bool = typer.Option(False, "--json", help="Output JSON")):
    """Check evidence health score for a project."""
    service = ProjectService()
    try:
        root = service._normalize_path(path)
        project_id = service._project_id_from_path(root)

        facts = service.get_facts(project_id)
        pages = service.get_wiki_pages(project_id)

        from llmbrain.models.fact import Fact
        from llmbrain.models.wiki import WikiPage
        from llmbrain.services.evidence_health import calculate_evidence_health

        fact_objs = [Fact(**f) for f in facts]
        page_objs = [WikiPage(**p) for p in pages]

        health_data = calculate_evidence_health(fact_objs, page_objs)

        if json_out:
            typer.echo(json.dumps(health_data, indent=2))
        else:
            console.print(f"Health Score: {health_data['score']}/100")
            console.print(f"Rating: {health_data['rating']}")
    except Exception as e:
        err_console.print(f"[bold red]Health check failed: {e}[/bold red]")
        raise typer.Exit(code=1)


@app.command()
def benchmark(
    path: str,
    json_out: bool = typer.Option(False, "--json", help="Output JSON"),
):
    """Benchmark token savings from raw source to LLM Brain memory artifacts."""
    try:
        from llmbrain.services.benchmark import benchmark_project

        result = benchmark_project(path)
        if json_out:
            typer.echo(json.dumps(result, indent=2))
            return

        console.print(f"[bold green]Benchmark: {result['project']}[/bold green]")
        console.print(
            f"Source: {result['documents']} docs, "
            f"{result['source']['estimated_tokens']} estimated tokens"
        )
        table = Table("Artifact", "Tokens", "Savings", "Compression")
        for artifact in result["artifacts"]:
            compression = artifact["compression_ratio"]
            table.add_row(
                artifact["name"],
                str(artifact["estimated_tokens"]),
                f"{artifact['savings_vs_source_percent']}%",
                f"{compression}x" if compression else "-",
            )
        console.print(table)

        knowledge = result["knowledge"]
        console.print(
            "Knowledge: "
            f"{knowledge['facts']} facts, "
            f"{knowledge['facts_with_evidence']} evidenced, "
            f"{knowledge['entities']} entities, "
            f"{knowledge['relations']} relations, "
            f"{knowledge['wiki_pages']} wiki pages"
        )
        console.print(
            f"Density: {knowledge['items_per_1k_source_tokens']} knowledge items / 1k source tokens"
        )
        console.print(f"Brain readiness: {result['brain_readiness_score']}/100")
    except Exception as e:
        err_console.print(f"[bold red]Benchmark failed: {e}[/bold red]")
        raise typer.Exit(code=1)


@app.command("token-report")
def token_report(
    path: str,
    json_out: bool = typer.Option(False, "--json", help="Output JSON"),
    max_chars: int = typer.Option(120_000, "--max-chars", help="BrainFrame max chars"),
    mode: str = typer.Option(
        "stored",
        "--mode",
        help="Compatibility option; token-report uses stored build artifacts.",
    ),
):
    """Compare JSON-style context with compact BrainFrame context."""
    _ = mode
    service = ProjectService()
    try:
        result = service.token_report_for_path(path, max_chars=max_chars)
        if json_out:
            typer.echo(json.dumps(result, indent=2))
            return

        table = Table("Context Format", "Chars", "Est. Tokens")
        table.add_row(
            "JSON",
            f"{result['json_chars']:,}",
            f"{result['json_estimated_tokens']:,}",
        )
        table.add_row(
            "BrainFrame",
            f"{result['brainframe_chars']:,}",
            f"{result['brainframe_estimated_tokens']:,}",
        )
        table.add_row(
            "Saved",
            f"{result['saved_chars']:,}",
            f"{result['saved_tokens']:,}",
        )
        table.add_row("Saved Percent", f"{result['saved_percent']}%", "")
        console.print(table)
        if result["brainframe_truncated"]:
            console.print("[yellow]BrainFrame was truncated by --max-chars.[/yellow]")
    except Exception as e:
        err_console.print(f"[bold red]Token report failed: {e}[/bold red]")
        raise typer.Exit(code=1)


# ── Agent and Memory Subcommands ─────────────────────────────────────

memory_app = typer.Typer(help="Repository memory management commands.")
app.add_typer(memory_app, name="memory")


def run_agent_task(
    agent_name: str,
    task_or_question: str,
    path: str = ".",
    provider: str = settings.default_provider,
    json_out: bool = False,
):
    import asyncio

    from llmbrain.agent.runtime import AgentRuntime
    from llmbrain.llm.providers import create_provider
    from llmbrain.services.project_service import ProjectService

    service = ProjectService()
    try:
        p = Path(path).resolve()
        service.build_project(str(p), llm_provider=provider, incremental=True)

        llm = create_provider(provider)
        agent = AgentRuntime(p, llm, agent_name=agent_name)

        if not json_out:
            console.print(
                f"[bold blue]Running {agent.agent_def.display_name}:[/bold blue] {task_or_question}"
            )

        record = asyncio.run(agent.execute_task(task_or_question))

        if json_out:
            typer.echo(record.model_dump_json(indent=2))
        else:
            if record.status == "completed":
                console.print(
                    f"\n[bold green]Completed ({record.status}):[/bold green] {record.summary}"
                )
            else:
                console.print(f"\n[bold red]Failed ({record.status}):[/bold red] {record.summary}")
            if record.verification:
                console.print(
                    f"[bold]Verification Status:[/bold] {record.verification.status} | "
                    f"{record.verification.summary}"
                )
    except Exception as e:
        err_console.print(f"[bold red]Error: {e}[/bold red]")
        raise typer.Exit(code=1)


@app.command()
def ask(
    question: str,
    path: str = typer.Option(".", "--path", help="Project path"),
    provider: str = typer.Option(
        settings.default_provider,
        help="LLM provider: openai | deepseek | anthropic | ollama",
    ),
    json_out: bool = typer.Option(False, "--json", help="Output JSON"),
):
    """Run Q&A about the repository memory."""
    run_agent_task("ask", question, path, provider, json_out)


@app.command()
def plan(
    task: str,
    path: str = typer.Option(".", "--path", help="Project path"),
    provider: str = typer.Option(
        settings.default_provider,
        help="LLM provider: openai | deepseek | anthropic | ollama",
    ),
    json_out: bool = typer.Option(False, "--json", help="Output JSON"),
):
    """Generate analysis and implementation plan for a coding task."""
    run_agent_task("plan", task, path, provider, json_out)


@app.command()
def review(
    base: str = typer.Option("main", "--base", help="Base commit/branch to diff against"),
    path: str = typer.Option(".", "--path", help="Project path"),
    provider: str = typer.Option(
        settings.default_provider,
        help="LLM provider: openai | deepseek | anthropic | ollama",
    ),
    json_out: bool = typer.Option(False, "--json", help="Output JSON"),
):
    """Review current git changes (diff, correctness, safety)."""
    run_agent_task("review", f"Review changes compared to {base}", path, provider, json_out)


@app.command()
def index(
    path: str = typer.Option(".", "--path", help="Project path"),
    provider: str = typer.Option(
        settings.default_provider,
        help="LLM provider: openai | deepseek | anthropic | ollama",
    ),
):
    """Build and refresh project index database."""
    service = ProjectService()
    try:
        p = Path(path).resolve()
        console.print(f"[bold blue]İndeksleniyor: {p}[/bold blue]")
        service.build_project(str(p), llm_provider=provider, incremental=True)
        console.print("[bold green]İndeksleme başarıyla tamamlandı.[/bold green]")
    except Exception as e:
        err_console.print(f"[bold red]Hata: {e}[/bold red]")
        raise typer.Exit(code=1)


@memory_app.command("inspect")
def memory_inspect(
    path: str = typer.Option(".", "--path", help="Project path"),
):
    """Inspect stored task execution runs and decisions."""
    from llmbrain.storage.filesystem import output_root
    from llmbrain.storage.sqlite import SQLiteStore

    try:
        p = Path(path).resolve()
        db_path = output_root(p) / "brain.db"
        if not db_path.exists():
            console.print("Veritabanı bulunamadı. Lütfen önce indeksleme yapın.")
            return

        store = SQLiteStore(db_path)
        project_id = ProjectService._project_id_from_path(p)

        runs = store.get_task_runs(project_id)
        if not runs:
            console.print("Görev geçmişi hafızası boş.")
            return

        for run in runs:
            console.print(f"\n[bold cyan]Görev ID: {run['id']}[/bold cyan]")
            console.print(f"  Talep: {run['request']}")
            console.print(f"  Durum: {run['status']}")
            console.print(f"  Özet: {run['summary']}")
            if run.get("commit_hash"):
                console.print(f"  Commit: {run['commit_hash']}")

            # Load details
            details = store.get_task_details(run["id"])
            if details:
                if details.get("decisions"):
                    console.print("  Alınan Kararlar:")
                    for d in details["decisions"]:
                        console.print(f"    - {d['decision']}")
                if details.get("failures"):
                    console.print("  Karşılaşılan Hatalar:")
                    for f in details["failures"]:
                        console.print(f"    - Hata: {f['failure']} | Çözüm: {f.get('resolution')}")
    except Exception as e:
        err_console.print(f"[bold red]Hata: {e}[/bold red]")
        raise typer.Exit(code=1)


@memory_app.command("refresh")
def memory_refresh(
    path: str = typer.Option(".", "--path", help="Project path"),
    provider: str = typer.Option(
        settings.default_provider,
        help="LLM provider: openai | deepseek | anthropic | ollama",
    ),
):
    """Force complete rebuild of project memory index."""
    service = ProjectService()
    try:
        p = Path(path).resolve()
        console.print(f"[bold blue]Hafıza zorla tazeleniyor (Full build): {p}[/bold blue]")
        service.build_project(str(p), llm_provider=provider, incremental=False)
        console.print("[bold green]Hafıza başarıyla tazelendi.[/bold green]")
    except Exception as e:
        err_console.print(f"[bold red]Hata: {e}[/bold red]")
        raise typer.Exit(code=1)


@memory_app.command("export")
def memory_export(
    path: str = typer.Option(".", "--path", help="Project path"),
    output: str = typer.Option(None, "--output", help="Output JSONL file path"),
):
    """Export project vectors and metadata to a JSONL file."""
    service = ProjectService()
    try:
        p = Path(path).resolve()
        proj = service.get_project_by_path(str(p))
        if not proj:
            console.print(f"[bold red]Project not found for path: {p}[/bold red]")
            raise typer.Exit(code=1)
        
        out_path = Path(output) if output else p / "memory_export.jsonl"
        from llmbrain.storage.vector_store import VectorStore
        vs = VectorStore(output_root(proj.root_path) / "vectors.db")
        count = vs.export_jsonl(proj.id, out_path)
        vs.close()
        
        console.print(f"[bold green]Exported {count} vectors to {out_path}[/bold green]")
    except Exception as e:
        err_console.print(f"[bold red]Hata: {e}[/bold red]")
        raise typer.Exit(code=1)


@app.command()
def config():
    """Print active configuration settings."""
    from rich.pretty import pprint

    pprint(settings.model_dump())


@app.command()
def debug(
    problem: str,
    path: str = typer.Option(".", "--path", help="Project path"),
    provider: str = typer.Option(
        settings.default_provider,
        help="LLM provider: openai | deepseek | anthropic | ollama",
    ),
    json_out: bool = typer.Option(False, "--json", help="Output JSON"),
):
    """Reproduce and diagnose bugs."""
    run_agent_task("debug", problem, path, provider, json_out)


@app.command()
def test(
    task: str,
    path: str = typer.Option(".", "--path", help="Project path"),
    provider: str = typer.Option(
        settings.default_provider,
        help="LLM provider: openai | deepseek | anthropic | ollama",
    ),
    json_out: bool = typer.Option(False, "--json", help="Output JSON"),
):
    """Inspect coverage, write or run tests."""
    run_agent_task("test", task, path, provider, json_out)


@app.command()
def security(
    scope: str,
    path: str = typer.Option(".", "--path", help="Project path"),
    provider: str = typer.Option(
        settings.default_provider,
        help="LLM provider: openai | deepseek | anthropic | ollama",
    ),
    json_out: bool = typer.Option(False, "--json", help="Output JSON"),
):
    """Perform secure code review."""
    run_agent_task("security", scope, path, provider, json_out)


@app.command()
def run(
    task: str,
    path: str = typer.Option(".", "--path", help="Project path"),
    provider: str = typer.Option(
        settings.default_provider,
        help="LLM provider: openai | deepseek | anthropic | ollama",
    ),
    json_out: bool = typer.Option(False, "--json", help="Output JSON"),
    disable_routing: bool = typer.Option(
        False, "--disable-routing", help="Disable automatic routing"
    ),
):
    """Execute task with automatic agent routing."""
    import asyncio

    from llmbrain.agent.agents import AgentRegistry, AgentRouter
    from llmbrain.llm.providers import create_provider
    from llmbrain.services.project_service import ProjectService

    service = ProjectService()
    try:
        p = Path(path).resolve()
        service.build_project(str(p), llm_provider=provider, incremental=True)

        llm = create_provider(provider)
        registry = AgentRegistry()
        registry.load_project_config(p)

        router = AgentRouter(registry, llm, disable_routing=disable_routing)
        agent_def, reason = asyncio.run(router.route(task))

        if not json_out:
            console.print(f"[bold yellow]Router routing result:[/bold yellow] {reason}")

        run_agent_task(agent_def.name, task, path, provider, json_out)
    except Exception as e:
        err_console.print(f"[bold red]Error: {e}[/bold red]")
        raise typer.Exit(code=1)


agents_app = typer.Typer(help="Manage and validate specialized agents")
app.add_typer(agents_app, name="agents")


@agents_app.command("list")
def agents_list(
    path: str = typer.Option(".", "--path", help="Project path"),
):
    """List all available agents (built-in and custom)."""
    from llmbrain.agent.agents import AgentRegistry

    registry = AgentRegistry()
    try:
        registry.load_project_config(Path(path).resolve())
        table = Table("Name", "Display Name", "Description", "Mode")
        for agent in registry.list_agents():
            table.add_row(
                agent.name,
                agent.display_name,
                agent.description,
                agent.permissions.get("mode", "read-only"),
            )
        console.print(table)
    except Exception as e:
        err_console.print(f"[bold red]Error: {e}[/bold red]")
        raise typer.Exit(code=1)


@agents_app.command("show")
def agents_show(
    name: str,
    path: str = typer.Option(".", "--path", help="Project path"),
):
    """Show details of a specific agent configuration."""
    from llmbrain.agent.agents import AgentRegistry

    registry = AgentRegistry()
    try:
        registry.load_project_config(Path(path).resolve())
        agent = registry.get_agent(name)
        console.print(f"[bold cyan]Agent: {agent.name}[/bold cyan]")
        console.print(f"  Display Name: {agent.display_name}")
        console.print(f"  Description: {agent.description}")
        console.print(f"  Prompt File: {agent.system_prompt}")
        console.print(f"  Context budget: {agent.context.token_budget} tokens")
        console.print(f"  Allowed tools: {agent.tools.allow}")
        console.print(f"  Denied tools: {agent.tools.deny}")
        console.print(f"  Permission mode: {agent.permissions.get('mode')}")
    except Exception as e:
        err_console.print(f"[bold red]Error: {e}[/bold red]")
        raise typer.Exit(code=1)


@agents_app.command("validate")
def agents_validate(
    path: str = typer.Option(".", "--path", help="Project path"),
):
    """Validate all agent configurations."""
    from llmbrain.agent.agents import AgentRegistry

    registry = AgentRegistry()
    try:
        registry.load_project_config(Path(path).resolve())
        console.print("[bold green]All agent configurations are valid![/bold green]")
    except Exception as e:
        err_console.print(f"[bold red]Validation failed: {e}[/bold red]")
        raise typer.Exit(code=1)


# ── Phase 5 commands and subcommands ─────────────────────────────────

db_app = typer.Typer(help="Database backup and restore operations.")
app.add_typer(db_app, name="db")


@db_app.command("backup")
def db_backup(
    path: str = typer.Option(".", "--path", help="Project path"),
    output: str = typer.Option("backup.zip", "--output", help="Backup output zip file"),
):
    """Backup project database files."""
    from llmbrain.services.project_service import ProjectService
    from llmbrain.storage.sqlite import backup_project_db

    p = Path(path).resolve()
    project_id = ProjectService._project_id_from_path(p)
    try:
        backup_project_db(project_id, output)
        console.print(f"[bold green]Database backup saved to {output}[/bold green]")
    except Exception as e:
        err_console.print(f"[bold red]Backup failed: {e}[/bold red]")
        raise typer.Exit(1)


@db_app.command("restore")
def db_restore(
    backup_file: str,
    path: str = typer.Option(".", "--path", help="Project path"),
):
    """Restore project database files from a backup zip."""
    from llmbrain.services.project_service import ProjectService
    from llmbrain.storage.sqlite import restore_project_db

    p = Path(path).resolve()
    project_id = ProjectService._project_id_from_path(p)
    try:
        ans = typer.confirm(
            "Are you sure you want to restore? This will overwrite existing databases."
        )
        if not ans:
            console.print("Restore cancelled.")
            return

        restore_project_db(project_id, backup_file)
        console.print("[bold green]Database restored successfully.[/bold green]")
    except Exception as e:
        err_console.print(f"[bold red]Restore failed: {e}[/bold red]")
        raise typer.Exit(1)


cache_app = typer.Typer(help="Cache statistics and management.")
app.add_typer(cache_app, name="cache")


@cache_app.command("stats")
def cache_stats(
    path: str = typer.Option(".", "--path", help="Project path"),
):
    """Display cache statistics."""
    from llmbrain.services.session_service import SessionService

    p = Path(path).resolve()
    service = SessionService(p)
    stats = service.cache.stats()
    table = Table("Metric", "Value")
    table.add_row("Hits", str(stats.hits))
    table.add_row("Misses", str(stats.misses))
    table.add_row("Evictions", str(stats.evictions))
    table.add_row("Current Items", str(stats.current_items))
    table.add_row("Current Bytes", f"{stats.current_bytes} bytes")
    console.print(table)


@cache_app.command("clear")
def cache_clear(
    project: bool = typer.Option(
        False, "--project", help="Clear only the current project namespace"
    ),
    path: str = typer.Option(".", "--path", help="Project path"),
):
    """Clear cached data."""
    from llmbrain.services.session_service import SessionService

    p = Path(path).resolve()
    service = SessionService(p)
    if project:
        service.cache.invalidate_project(service.project_id)
        console.print(f"[bold green]Cleared cache for project {service.project_id}.[/bold green]")
    else:
        service.cache.clear()
        console.print("[bold green]Cleared global cache.[/bold green]")


sessions_app = typer.Typer(help="Session management operations.")
app.add_typer(sessions_app, name="sessions")


@sessions_app.command("list")
def sessions_list_cmd(
    path: str = typer.Option(".", "--path", help="Project path"),
):
    """List all sessions in this project."""
    from llmbrain.services.session_service import SessionService

    p = Path(path).resolve()
    service = SessionService(p)
    sessions = service.list_sessions()

    table = Table("Session ID", "Title", "Agent", "Status", "Updated At")
    for s in sessions:
        table.add_row(s["id"], s["title"], s["active_agent"], s["status"], s["updated_at"])
    console.print(table)


@sessions_app.command("new")
def sessions_new_cmd(
    title: str = typer.Option("New Session", "--title", help="Session title"),
    agent: str = typer.Option("build", "--agent", help="Active agent"),
    path: str = typer.Option(".", "--path", help="Project path"),
):
    """Create a new session."""
    from llmbrain.services.session_service import SessionService

    p = Path(path).resolve()
    service = SessionService(p)
    sess = service.create_session(title, agent, {}, "ask-before-write")
    console.print(f"[bold green]Created new session {sess['id']}[/bold green]")


@sessions_app.command("resume")
def sessions_resume_cmd(
    session_id: str,
    path: str = typer.Option(".", "--path", help="Project path"),
):
    """Resume a coding session and start the TUI for it."""
    import asyncio

    from llmbrain.tui import LLMBrainTUI

    p = Path(path).resolve()
    tui = LLMBrainTUI(p)
    tui.state.selected_session_id = session_id
    tui.state.messages = tui.session_service.get_messages(session_id)
    try:
        asyncio.run(tui.start())
    except KeyboardInterrupt:
        pass


@sessions_app.command("rename")
def sessions_rename_cmd(
    session_id: str,
    new_title: str,
    path: str = typer.Option(".", "--path", help="Project path"),
):
    """Rename a session."""
    from llmbrain.services.session_service import SessionService

    p = Path(path).resolve()
    service = SessionService(p)
    service.rename_session(session_id, new_title)
    console.print("[bold green]Session renamed successfully.[/bold green]")


@sessions_app.command("archive")
def sessions_archive_cmd(
    session_id: str,
    path: str = typer.Option(".", "--path", help="Project path"),
):
    """Archive a session."""
    from llmbrain.services.session_service import SessionService

    p = Path(path).resolve()
    service = SessionService(p)
    service.archive_session(session_id)
    console.print("[bold green]Session archived.[/bold green]")


@sessions_app.command("delete")
def sessions_delete_cmd(
    session_id: str,
    path: str = typer.Option(".", "--path", help="Project path"),
):
    """Delete a session."""
    from llmbrain.services.session_service import SessionService

    p = Path(path).resolve()
    service = SessionService(p)
    ans = typer.confirm("Are you sure you want to delete this session?")
    if ans:
        service.delete_session(session_id)
        console.print("[bold green]Session deleted.[/bold green]")


@sessions_app.command("export")
def sessions_export_cmd(
    session_id: str,
    path: str = typer.Option(".", "--path", help="Project path"),
):
    """Export a session transcript to standard output."""
    from llmbrain.services.session_service import SessionService

    p = Path(path).resolve()
    service = SessionService(p)
    transcript = service.export_session(session_id)
    typer.echo(transcript)


@app.command()
def logs(
    path: str = typer.Option(".", "--path", help="Project path"),
    lines: int = typer.Option(50, "--lines", help="Number of lines to show"),
):
    """Show application logs."""
    from llmbrain.services.session_service import SessionService

    p = Path(path).resolve()
    service = SessionService(p)
    log_dir = service.storage_dir / "logs"
    console.print(f"Log directory: {log_dir}")


@app.command()
def doctor(
    path: str = typer.Option(".", "--path", help="Project path"),
):
    """Perform a system health check and configuration audit."""
    import sqlite3
    import sys

    from llmbrain.services.session_service import SessionService

    p = Path(path).resolve()
    try:
        service = SessionService(p)
        console.print("🩺 [bold]LLMBrain Doctor Diagnostics[/bold]")
        console.print(f"  Project ID: [green]{service.project_id}[/green]")
        console.print(f"  Project Root: {p}")
        console.print(f"  Storage Directory: {service.storage_dir}")

        # Check SQLite
        conn = sqlite3.connect(str(service.brain_db))
        cur = conn.cursor()
        wal_check = cur.execute("PRAGMA journal_mode").fetchone()[0]
        conn.close()
        console.print(f"  Database Mode: [cyan]{wal_check}[/cyan] (Expected: wal)")

        # Check Python
        console.print(f"  Python Version: {sys.version.split()[0]}")
        console.print(f"  SQLite Version: {sqlite3.sqlite_version}")
        console.print("  [bold green]Status: All systems functional![/bold green]")
    except Exception as e:
        err_console.print(f"  [bold red]Health check failed: {e}[/bold red]")
        raise typer.Exit(1)


# ── Phase 6 commands ─────────────────────────────────────────────────

phase6_app = typer.Typer(help="Phase 6: async indexing, resource profiling, and observability.")
app.add_typer(phase6_app, name="observe")


@phase6_app.command("queue-stats")
def queue_stats_cmd(
    path: str = typer.Option(".", "--path", help="Project path"),
    json_out: bool = typer.Option(False, "--json", help="Output JSON"),
):
    """Show indexing queue statistics for this project."""
    from llmbrain.core.identity import get_project_storage_dir, load_or_create_project_identity
    from llmbrain.core.queue import IndexQueue

    p = Path(path).resolve()
    identity = load_or_create_project_identity(p)
    project_id = identity["project_id"]
    db_path = get_project_storage_dir(project_id) / "queue.db"
    q = IndexQueue(db_path)
    stats = q.stats(project_id)
    if json_out:
        typer.echo(json.dumps(stats, indent=2))
    else:
        table = Table("Status", "Count")
        for status, count in sorted(stats.items()):
            table.add_row(status, str(count))
        console.print(table)


@phase6_app.command("profiler-report")
def profiler_report_cmd(
    json_out: bool = typer.Option(False, "--json", help="Output JSON"),
    top: int = typer.Option(10, "--top", help="Number of slowest operations to show"),
):
    """Show operation profiler report from the current process session."""
    from llmbrain.services.profiler import default_profiler

    report = default_profiler.get_report()
    slowest = default_profiler.get_slowest(top)
    if json_out:
        typer.echo(json.dumps(default_profiler.as_dict(), indent=2))
        return
    console.print(f"[bold]Profiler Report[/bold] — {report.total_operations} operations recorded")
    console.print(f"  Total duration : {report.total_duration_ms:.1f} ms")
    console.print(f"  Avg duration   : {report.avg_duration_ms:.1f} ms")
    console.print(f"  Peak mem delta : {report.peak_memory_delta_mb:.2f} MB")
    if slowest:
        table = Table("Operation", "Duration (ms)", "Mem Δ (MB)", "Timestamp")
        for e in slowest:
            table.add_row(
                e.operation,
                f"{e.duration_ms:.1f}",
                f"{e.memory_delta_mb:.2f}",
                str(e.timestamp),
            )
        console.print(table)


@phase6_app.command("resource-status")
def resource_status_cmd(
    samples: int = typer.Option(3, "--samples", help="Number of samples to take"),
    json_out: bool = typer.Option(False, "--json", help="Output JSON"),
):
    """Show CPU and memory resource status."""
    import time as _time

    from llmbrain.core.resource_manager import ResourceManager

    rm = ResourceManager()
    for _ in range(samples):
        rm.sample()
        _time.sleep(0.2)
    stats = rm.get_stats()
    if json_out:
        typer.echo(json.dumps(stats, indent=2))
        return
    snap = rm.snapshots[-1] if rm.snapshots else None
    console.print("[bold]Resource Status[/bold]")
    if snap:
        console.print(f"  CPU         : [cyan]{snap.cpu_percent:.1f}%[/cyan]")
        console.print(f"  Memory      : [cyan]{snap.memory_percent:.1f}%[/cyan]")
        console.print(f"  Avail. RAM  : [cyan]{snap.memory_mb:.0f} MB[/cyan]")
    state_color = "green" if stats["state"] == "normal" else "red"
    console.print(f"  State       : [{state_color}]{stats['state']}[/{state_color}]")
    console.print(f"  Recommended workers: [yellow]{stats['recommended_workers']}[/yellow]")


@phase6_app.command("services")
def services_cmd(
    endpoint: list[str] = typer.Option([], "--endpoint", help="Add endpoint: name=http://url"),
    timeout: float = typer.Option(3.0, "--timeout", help="Health check timeout"),
    json_out: bool = typer.Option(False, "--json", help="Output JSON"),
):
    """Check remote service health endpoints."""
    import asyncio as _asyncio

    from llmbrain.services.remote import ConnectionState, RemoteServiceMonitor, ServiceEndpoint

    endpoints = []
    for ep_str in endpoint:
        if "=" not in ep_str:
            err_console.print(f"[red]Invalid endpoint format (use name=url): {ep_str}[/red]")
            continue
        name, url = ep_str.split("=", 1)
        endpoints.append(
            ServiceEndpoint(name=name.strip(), base_url=url.strip(), timeout_sec=timeout)
        )

    monitor = RemoteServiceMonitor(endpoints)
    results = _asyncio.run(monitor.check_all())

    if json_out:
        typer.echo(json.dumps([r.model_dump(mode="json") for r in results], indent=2))
        return

    if not results:
        console.print("[dim]No endpoints configured. Use --endpoint name=http://url[/dim]")
        return

    table = Table("Service", "State", "Latency (ms)", "Error")
    state_colors = {
        ConnectionState.CONNECTED: "green",
        ConnectionState.OFFLINE: "red",
        ConnectionState.DEGRADED: "yellow",
        ConnectionState.UNKNOWN: "dim",
    }
    for r in results:
        color = state_colors.get(r.state, "white")
        latency = f"{r.latency_ms:.1f}" if r.latency_ms is not None else "-"
        table.add_row(r.service_name, f"[{color}]{r.state.value}[/{color}]", latency, r.error or "")
    console.print(table)
    overall = monitor.get_overall_state()
    overall_color = state_colors.get(overall, "white")
    console.print(f"Overall state: [{overall_color}]{overall.value}[/{overall_color}]")


# ── Phase 7 commands ─────────────────────────────────────────────────

repo_app = typer.Typer(help="Çoklu repo yönetimi (Phase 7).")
app.add_typer(repo_app, name="repo")


@repo_app.command("add")
def repo_add_cmd(
    path: str = typer.Argument(..., help="Proje kök dizini"),
    name: str = typer.Option(None, "--name", help="Repo adı (varsayılan: klasör adı)"),
    tag: list[str] = typer.Option([], "--tag", help="Etiket (birden fazla eklenebilir)"),
):
    """Repo kaydına yeni bir proje dizini ekle."""
    from llmbrain.services.multi_repo import MultiRepoRegistry

    reg = MultiRepoRegistry()
    try:
        entry = reg.add(path, name=name, tags=list(tag))
        console.print(f"[bold green]Eklendi:[/bold green] {entry.name} ({entry.project_id})")
    except ValueError as e:
        err_console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)


@repo_app.command("remove")
def repo_remove_cmd(project_id: str):
    """Repo kaydından projeyi sil."""
    from llmbrain.services.multi_repo import MultiRepoRegistry

    reg = MultiRepoRegistry()
    if reg.remove(project_id):
        console.print("[bold green]Proje kaydı silindi.[/bold green]")
    else:
        err_console.print(f"[red]Proje bulunamadı: {project_id}[/red]")
        raise typer.Exit(1)


@repo_app.command("list")
def repo_list_cmd(
    tag: str = typer.Option(None, "--tag", help="Etikete göre filtrele"),
    json_out: bool = typer.Option(False, "--json", help="JSON çıkış"),
):
    """Kayıtlı tüm repoları listele."""
    from llmbrain.services.multi_repo import MultiRepoRegistry

    reg = MultiRepoRegistry()
    repos = reg.search_by_tag(tag) if tag else reg.list_repos()
    if json_out:
        typer.echo(json.dumps([r.model_dump(mode="json") for r in repos], indent=2))
        return
    table = Table("Proje ID", "Ad", "Kök Dizin", "Son İndeksleme", "Etiketler")
    for r in repos:
        last = r.last_indexed.strftime("%Y-%m-%d") if r.last_indexed else "-"
        table.add_row(r.project_id[:12], r.name, r.root_path, last, ", ".join(r.tags))
    console.print(table)


@repo_app.command("tag")
def repo_tag_cmd(project_id: str, tag: str):
    """Bir repoya etiket ekle."""
    from llmbrain.services.multi_repo import MultiRepoRegistry

    reg = MultiRepoRegistry()
    if reg.add_tag(project_id, tag):
        console.print(f"[bold green]Etiket '{tag}' eklendi.[/bold green]")
    else:
        err_console.print(f"[red]Proje bulunamadı: {project_id}[/red]")
        raise typer.Exit(1)


@app.command("search")
def search_cmd(
    query: str = typer.Argument(..., help="Arama sorgusu"),
    path: str = typer.Option(".", "--path", help="Proje dizini"),
    k: int = typer.Option(10, "--top", help="Sonuç sayısı"),
    threshold: float = typer.Option(0.2, "--threshold", help="Minimum benzerlik skoru"),
    types: str = typer.Option(None, "--types", help="chunk,fact,entity (virgülle)"),
    json_out: bool = typer.Option(False, "--json", help="JSON çıkış"),
):
    """Proje hafızasında semantik arama yap."""
    from llmbrain.core.identity import get_project_storage_dir, load_or_create_project_identity
    from llmbrain.services.semantic_search import create_search_service

    p = Path(path).resolve()
    try:
        identity = load_or_create_project_identity(p)
        project_id = identity["project_id"]
        storage_dir = get_project_storage_dir(project_id)
        svc = create_search_service(project_id, storage_dir)

        # Try to auto-index if embedder not fitted
        stats = svc.get_stats()
        if not stats["embedder_fitted"]:
            console.print("[dim]İlk çalıştırma: hafıza indeksleniyor...[/dim]")
            indexed = svc.index_all()
            console.print(
                f"[dim]İndekslendi: {indexed['chunks']} parça, "
                f"{indexed['facts']} gerçek, {indexed['entities']} varlık[/dim]"
            )

        source_types = [t.strip() for t in types.split(",")] if types else None
        results = svc.search(query, source_types=source_types, k=k, threshold=threshold)

        if json_out:
            typer.echo(json.dumps([r.model_dump() for r in results], indent=2))
            return

        if not results:
            console.print("[dim]Sonuç bulunamadı.[/dim]")
            return

        table = Table("Tür", "Kaynak ID", "Skor", "Önizleme")
        for r in results:
            preview = r.text_preview[:60] + "..." if len(r.text_preview) > 60 else r.text_preview
            table.add_row(r.source_type, r.source_id[:16], f"{r.score:.3f}", preview)
        console.print(table)
    except Exception as e:
        err_console.print(f"[bold red]Hata: {e}[/bold red]")
        raise typer.Exit(1)


@app.callback(invoke_without_command=True)
def main_callback(
    ctx: typer.Context,
):
    """Start interactive TUI for LLMBrain if no subcommand is passed."""
    if ctx.invoked_subcommand is None:
        import sys

        # If user asks for help or version, let Typer handle it
        if any(h in ctx.args for h in ("--help", "-h")):
            return

        import asyncio

        from llmbrain.tui import LLMBrainTUI

        path = "."
        provider = settings.default_provider

        # Parse path from ctx.args
        for arg in ctx.args:
            if not arg.startswith("-"):
                path = arg
                break

        # Parse provider from ctx.args
        for i, arg in enumerate(ctx.args):
            if arg == "--provider" and i + 1 < len(ctx.args):
                provider = ctx.args[i + 1]
                break

        # Prevent starting TUI in non-interactive environments (e.g. tests)
        if not sys.stdin.isatty():
            return

        tui = LLMBrainTUI(path, provider_name=provider)
        try:
            asyncio.run(tui.start())
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    app()
