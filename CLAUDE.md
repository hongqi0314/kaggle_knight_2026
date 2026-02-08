# CLAUDE.md

## Commands
- **Run Script:** `uv run python <script>`
- **Start Notebook:** `uv run marimo edit notebooks/<name>.py --watch`
- **Install Package:** `uv add <package>`
- **Update Lockfile:** `uv sync`
- **Agent Commit:** `/commit "<message>"`
- **Publish Repo:** `gh repo create --private --source=. --remote=origin`
- **Push:** `git push origin main`

## Guidelines
- **Git Style:** Use Conventional Commits.
  - `feat:` New features, models, or pipelines.
  - `fix:` Bug fixes.
  - `data:` Data processing, feature engineering, or dataset changes.
  - `docs:` Documentation or log updates.
  - `chore:` Tooling, configuration, or maintenance.
- **Tooling:**
  - **Execution:** Always prioritize `uv run` for consistent environments.
  - **Data Processing:** Use `import polars as pl` for memory efficiency and speed. Prefer Polars' lazy API (`.lazy()`) for heavy operations.
- **Project Structure:**
  - `data/`: Ignored by Git. Contains `raw/` and `processed/`.
  - `notebooks/`: Interactive development. Use `.py` format via Marimo or Jupytext for better diffs.
  - `src/`: Reusable modules (e.g., `src/preprocessing`, `src/models`).
- **Experiment Tracking:**
  - Summarize all key experiment results (Model, CV Score, LB Score, Params) in `EXPERIMENTS.md` before committing.
