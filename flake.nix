{
  description = "LLM Brain — Engineering Knowledge Compiler";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};

        python = pkgs.python312;

        pythonEnv = python.withPackages (ps: with ps; [
          # — core —
          fastapi
          uvicorn
          pydantic
          pydantic-settings
          python-multipart
          aiofiles
          typer
          rich
          python-dotenv
          jsonschema

          # — dev / test —
          pytest
          pytest-asyncio
          httpx
          ruff

          # — utilities —
          pip
        ]);
      in
      {
        devShells.default = pkgs.mkShell {
          name = "llm-brain";

          buildInputs = [
            pythonEnv
            pkgs.sqlite
          ];

          shellHook = ''
            echo ""
            echo "🧠  LLM Brain dev shell activated"
            echo "   Python : $(python3 --version)"
            echo "   SQLite : $(sqlite3 --version | cut -d' ' -f1)"
            echo ""
            echo "   Run:  uvicorn app.main:app --reload --port 8000"
            echo ""
          '';
        };
      }
    );
}
