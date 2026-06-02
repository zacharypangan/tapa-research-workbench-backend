# stling_backend/AGENTS.md

## Backend Rules

- Follow existing route, controller, service, and validation patterns.
- Preserve existing API response format.
- Add validation for new inputs.
- Avoid changing database schema unless explicitly requested.
- Avoid changing authentication or authorization unless explicitly requested.
- Keep changes focused on the endpoint or service relevant to the task.

## Credit-Saving Workflow

For API work:
1. Identify the existing route pattern.
2. Inspect only the relevant route, controller, service, schema/model, and test files.
3. Do not scan unrelated modules.
4. Propose the exact files to edit before editing.
5. Make the smallest working change.
6. Run the most focused test available before full test suite.

## Testing

- Prefer focused backend tests.
- Run the full backend test command only after targeted tests pass or if no focused test exists.
- Do not repeatedly run expensive integration tests without narrowing the cause.

