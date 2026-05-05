# Style Guide

## Imports

- Prefer a single import per module source rather than repeated one-line imports from the same module.
- If importing only a few names from the same module, keep them on one line:
  - `from package.module import A, B, C`
- If that import would exceed a reasonable line length, switch to a parenthesized multi-line import:
  - `from package.module import (`
  - `    A,`
  - `    B,`
  - `    C,`
  - `)`
- If a module contributes many names to a file, prefer importing the module itself and referencing names through the module namespace:
  - `from claim.datagen import constants`
  - `constants.DEFAULT_SAMPLE_SIZE`
- Prefer module-qualified constant access over importing constants individually when a constants module is shared broadly.
- Apply the same rule to other large helper surfaces: if many symbols from the same file are used, import the file and reference `module.name` instead of importing each symbol.
