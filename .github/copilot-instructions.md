---
applyTo: '**'
---

# General Instructions (PYTHON)

## Python Coding Guidelines

### Comments

- Comments should never be on the right-hand side of code. They should always be on their own line above the code they
  refer to.

### General Style

- Adhere to PEP 8, the official style guide for Python code. Use a linter (e.g., Ruff, Flake8) to enforce this.
- Keep lines under 79 characters for maximum readability.

### Naming Conventions

- Use `snake_case` for function and variable names.
- Use `PascalCase` for class names.
- Use `UPPER_SNAKE_CASE` for constants.
- Avoid single-letter variable names unless they are loop counters or widely understood (e.g., `x`, `y` for
  coordinates).

### Docstrings

- All modules, functions, classes, and methods should have docstrings.
- Use triple double quotes `"""Docstring content"""`.
- After the opening triple quotes, the first line should be a short summary of the object's purpose and should be placed
  on the next line.
- The order is:
  - Summary line.
  - Blank line.
  - More detailed description (if necessary).
  - Blank line.
  - Any parameters and return value descriptions (if applicable) (`Params:`).
  - Blank line.
  - Return value description (if applicable) (`Returns:`).
  - Blank line.
  - Any exceptions raised (if applicable) (`Raises:`).
- Use imperative mood in the comments.
- Follow reStructuredText or Google style for formatting docstrings.
- The arguments name, type and description should be formatted as follows:

  ```python
  Args/Returns/Raises:
      <variable_name> (<variable_type>):
          <variable_description>
  ```

### Imports

- Imports should always be at the top of the file, after any module docstrings and `__future__` imports.
- Imports should be grouped in the following order:
    1. Standard library imports.
    2. Third-party imports.
    3. Local application/library specific imports.
- Each group should be separated by a blank line.
- Use absolute imports over relative imports.

### Functions and Methods

- Functions and methods should ideally be short and perform a single, well-defined task.
- Avoid excessive parameters; consider passing an object or dictionary if there are many related parameters.

### Error Handling

- Use `try-except` blocks for handling expected errors gracefully.
- Avoid broad `except` clauses (e.g., `except Exception:`). Be specific about the exceptions you catch.
- Do not suppress errors silently. Log them or re-raise them if appropriate.

### Type Hinting

- Use type hints for function arguments and return values to improve code readability and maintainability.
- Use type hints for variables where clarity is needed.
- Prefer the modern union syntax (`X | Y`) over `Union[X, Y]` for Python 3.10+.
- Use `X | None` instead of `Optional[X]` for optional types.
- Only import from `typing` when necessary (e.g., for `List`, `Dict` in older Python versions, or specialized types like `Callable`).

## GIT Repository Management

### Commits

Use the Conventional Commits format: `<type>(scope): short summary`

Examples:

- `feature(config): support dynamic environment loading`
- `fix(core): handle missing config file gracefully`
- `test(utils): add unit tests for retry logic`

Allowed types (use these as `<type>` in your commit messages):

- `feature` – New features
- `fix` – Bug fixes
- `documentation` – Documentation changes only
- `style` – Code style, formatting, missing semi-colons, etc. (no code meaning changes)
- `refactor` – Code changes that neither fix a bug nor add a feature
- `performance` – Code changes that improve performance
- `test` – Adding or correcting tests
- `build` – Changes to build system or external dependencies
- `ci` – Changes to CI configuration files and scripts
- `chore` – Maintenance tasks (e.g., updating dependencies, minor tooling)
- `revert` – Reverting previous commits
- `security` – Security-related improvements or fixes
- `ux` – User experience or UI improvements

Other Notes:

- Prefer simple, linear Git history. Use rebase over merge where possible.
- Use `pre-commit` hooks to enforce formatting, linting, and checks before commits.
- If unsure about a change, open a draft PR with a summary and rationale.

### Release Guidelines

We follow a simplified Git-flow model for releases:

#### Branches

- `main`: Represents the latest stable, released version. Only hotfixes and release merges are committed directly to `main`.
- `develop`: Integration branch for ongoing development. All new features and bug fixes are merged into `develop`.
- `feature/<feature-name>`: Used for developing new features. Branch off `develop` and merge back into `develop` upon completion.

Here is the release process:

1. Prepare `develop` for Release:
    - Ensure all desired features and bug fixes are merged into `develop`.
    - Update `CHANGELOG.md` with changes for the new version, with the help of the command: `git log --pretty=format:"- (%h) %s" ...`
    - Update version numbers in relevant project files (e.g., `pyproject.toml`, `package.json`).
2. Make sure we start from a clean state:
    - Make sure you are on the `develop`, and that we start from there.
    - Perform final testing and bug fixing on this branch.
3. Merge to `main` and Tag:
    - Once the develop branch is stable, merge it into `main`:
      1. `git checkout main`
      2. `git merge --no-ff develop`  
    - Tag the release on `main`: `git tag -a v<version-number> -m "Release v<version-number>"`
    - Ask the user to push the changes to the `main` branch, including tags: `git push origin main --tags`
4. Merge back to `develop`:
    - Merge the main branch back into `develop` to ensure `develop` has all release changes:
      1. `git checkout develop`
      2. `git merge --no-ff main`
    - Ask the user to push the changes to `develop` branch: `git push origin develop`

## Tooling and Automation

- Use Black for automatic code formatting to ensure consistent style across the codebase.
- Integrate mypy for static type checking to catch type-related errors early.
- Set up CI/CD pipelines (e.g., GitHub Actions) for automated linting, testing, and building on every push/PR.

## Testing requirement (mandatory)

- Write comprehensive unit tests using pytest; aim for at least 80% code coverage.
- Include integration and end-to-end tests to validate full workflows.
- Run tests automatically before commits using pre-commit hooks.

- For every code change, add or update at least one targeted automated test that would fail before the change and pass after it.  
- Run the smallest relevant test command and report the exact command and pass/fail outcome.  
- If behavior affects pipeline integration, run one integration-level smoke check in addition to unit tests.  
- Do not claim completion if tests were not run; instead state exactly what remains unverified.  
- For bug fixes, include a regression test that covers the specific failure mode from logs/tracebacks.  
- Any change to branching/flags/modes must include tests for both default and non-default paths where feasible.
- Do not publish Kaggle assets or push kernel versions until targeted tests for changed logic pass.

## Security and Performance

- Never hardcode secrets; use environment variables or secure management tools (e.g., AWS Secrets Manager).
- Validate and sanitize all user inputs to prevent injection attacks and other vulnerabilities.
- Profile performance-critical code using tools like cProfile or line_profiler; optimize bottlenecks as needed.

## Dependencies and Environment

- Pin dependency versions in `pyproject.toml` or `requirements.txt` to ensure reproducible builds.
- Use virtual environments (e.g., venv or conda) for project isolation.
- Regularly update dependencies and perform security audits (e.g., via `pip-audit`).

## Advanced Topics

- For asynchronous code, use asyncio with proper type hints for coroutines and awaitables.
- Implement structured logging using Python's `logging` module, setting appropriate levels (DEBUG, INFO, WARNING, ERROR).
- If no project-wide logger is configured, use per-file loggers: `logger = logging.getLogger(__name__)`

## Enforcement and Metrics

- Enforce coding guidelines through CI failures (e.g., PRs must pass linting and tests).
- Track key metrics like test coverage percentage, cyclomatic complexity, and code duplication using tools like coverage.py and radon.