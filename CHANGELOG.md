# Changelog

## Unreleased

- Added execution timeout handling for `HaskellInterpreter` to prevent indefinite blocking.
- Improved SQL statement parsing to handle semicolons inside strings/comments safely.
- Refreshed SQL tool registration behavior so reused interpreter instances honor updated callables.
- Added regression tests for SQL parsing/tool refresh and Haskell timeout behavior.
- Added CI and Trusted Publishing workflows for automated lint/test/build and PyPI release.
- Added release documentation and repository hygiene updates (`.gitignore`, README release validation section).

## 0.1.0

- Initial extraction of non-Python REPL engines for DSPy:
  - `SchemeRLM`
  - `SQLRLM`
  - `HaskellRLM`
- Added shared non-Python base RLM implementation and compatibility helpers.
- Added benchmark integration entry points for side-by-side engine comparisons.
