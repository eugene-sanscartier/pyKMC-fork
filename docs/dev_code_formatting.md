# Developer Guide

## Code Formatting

pyKMC uses [Ruff](https://docs.astral.sh/ruff/) for Python formatting. Formatting is enforced by CI on every push and pull request — the check runs `ruff format --check` and fails if any file is not formatted.

### Installation

```bash
pip install ruff
# or, if using uv:
uv sync --extra dev
```

### Formatting

```bash
# Format all files
ruff format .

# Check without modifying (same as CI)
ruff format --check .
```

Configuration lives in `ruff.toml` at the repo root. Key settings: line length 88, Python 3.10 target, double quotes.

### Editor Setup

=== "VSCode"

    Install the [Ruff extension](https://marketplace.visualstudio.com/items?itemName=charliermarsh.ruff) (`charliermarsh.ruff`).

    The repo includes `.vscode/settings.json` which sets Ruff as the default Python formatter and enables format-on-save automatically.

=== "PyCharm / JetBrains"

    Install the [Ruff plugin](https://plugins.jetbrains.com/plugin/20574-ruff) from the JetBrains Marketplace.

    Then enable format-on-save: **Settings → Tools → Ruff → Format on save**.

    The plugin automatically picks up `ruff.toml` from the project root.

=== "Neovim / other editors"

    Any editor with an LSP client can use `ruff server`. See the [Ruff editor integrations docs](https://docs.astral.sh/ruff/editors/) for setup instructions.
