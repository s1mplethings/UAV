# CURRENT

- task: Make source-mode OpenAI config prefer the repo-root `openai.env` over inherited shell environment variables so the workbench actually uses the intended key instead of an unrelated ambient value like `OPENAI_API_KEY=ollama`.
- scope: Touch only runtime OpenAI default resolution and restart the source workbench; no packaging changes.
- constraints: Keep the API flow intact, preserve environment-variable fallback when no sidecar file exists, and make the smallest possible change.
- current step: Runtime OpenAI precedence fix, verification, and source-workbench restart are complete.
- next action: Hand off the restarted source workbench and tell the user the repo-root `openai.env` now overrides the inherited `OPENAI_API_KEY=ollama`.
- status: done
