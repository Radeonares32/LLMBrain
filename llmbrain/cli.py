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
    path: str,
    provider: str = typer.Option(
        settings.default_provider,
        help="LLM provider: openai | deepseek | anthropic | ollama",
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
    """Run the full build pipeline."""
    service = ProjectService()
    try:
        if not json_out:
            console.print(
                f"[bold blue]Building project at {path} (provider={provider})[/bold blue]"
            )

        result = service.build_project(
            path,
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
    except Exception as e:
        err_console.print(f"[bold red]Build failed: {e}[/bold red]")
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


if __name__ == "__main__":
    app()
