---
description: Generated code (codegen output) must not be hand-edited
globs:
  - "**/gen/**"
  - "**/generated/**"
---

# Generated Code

Files under these directories are emitted by codegen and must not be hand-edited.

## To change them

1. Edit the source model or schema (e.g., Alloy `models/*.als`, OpenAPI spec).
2. Run the project's regeneration script (see project README).
3. Commit the regenerated diff alongside the source change.

## Detection

CI runs a drift check (regenerate and diff against committed output). Regenerate locally before pushing.
