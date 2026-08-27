# Repository Agent Guidelines

These instructions apply to the entire repository unless a more specific
`AGENTS.md` exists in a subdirectory.

## Priorities

- Optimize for correctness, clarity, maintainability, and a small review surface.
- Make the smallest cohesive change that fully solves the problem.
- Preserve public behavior and compatibility unless the task explicitly calls for
  a breaking change.
- Prefer straightforward code over clever code. Make invariants obvious.
- Do not add speculative abstractions, dependencies, configuration, or features.

## Rust Quality Gates

Before handing off a Rust change, run the relevant targeted tests while iterating,
then run all of these from the workspace root:

```text
python scripts/check_repository_policy.py
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace --all-targets --all-features --exclude dxbc-fuzz
```

Also build documentation with warnings denied when public APIs or docs change:

```text
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --all-features --no-deps
```

In PowerShell, set `RUSTDOCFLAGS` with
`$env:RUSTDOCFLAGS = "-D warnings"` before running the documentation command.
If `python` is a local shim that does not forward arguments, run the policy check
with `uv run --no-project python scripts/check_repository_policy.py`.

The workspace lint table already denies Clippy's `pedantic` group; a plain
`-D warnings` run enforces it.

### Clippy

- Treat the workspace's `clippy::pedantic` policy and every compiler or Clippy
  warning as an error.
- Fix the cause of a lint instead of hiding it or weakening workspace lint
  configuration.
- Do not add crate-, module-, or file-wide lint allowances.
- A narrow `#[expect(...)]` is acceptable only when the lint makes the code less
  correct or less clear. Keep it on the smallest item and add a short reason
  explaining why it is justified.

## Decompiler Correctness

- The decompiler's contract is *proven* semantic equivalence, not plausible
  output. `crates/decompiler/tests/recompile.rs` decompiles every corpus shader,
  recompiles it under `fxc /Gis`, and executes both programs in the reference
  interpreter (`crates/decompiler/src/interp.rs`), comparing outputs
  bit-for-bit. A change to the lift, renderer, or interpreter is not done until
  this gate reports every decompiled shader proven equivalent.
- The interpreter executes the decoded program independently of the lift so
  lift and interpreter bugs cannot cancel. Do not share per-opcode semantics
  between them.
- Renderer output must preserve bit-level semantics: `precise` locals pin
  evaluation order, integer data never round-trips float-typed storage
  (fxc's constant folder flushes provable denormals), and the DXBC-defined
  out-of-bounds behavior of `icb` reads is guarded explicitly.
- An unsupported shader shape must fail loudly with a message naming the
  construct — never render silently wrong output.

## File and Module Size

- Keep every hand-written source file at or below 1,000 physical lines, including
  inline tests.
- Split files before they cross the limit. Extract modules by responsibility, not
  by arbitrary line ranges.
- Do not create catch-all modules or move complexity into a differently named
  oversized file.
- Files listed in the policy script's grandfathered set predate the limit: do not
  make them larger. A substantial edit to one must include a cohesive split that
  moves it toward compliance.
- Generated files are exempt only when they are clearly marked as generated and
  are not intended for manual editing.

## Cargo Project Layout

- Follow the [Cargo Book package-layout conventions](https://doc.rust-lang.org/cargo/guide/project-layout.html)
  within every workspace package.
- Keep package manifests at the package root and Rust implementation code under
  `src/`. Use `src/lib.rs` and `src/main.rs` for the default library and binary
  targets.
- Put integration tests in `tests/`, examples in `examples/`, and benchmarks in
  `benches/` at the package root. The `dxbc-fuzz` package follows cargo-fuzz's
  own layout and is exempt.
- Name binary, example, benchmark, and integration-test targets in `kebab-case`.
  Name Rust modules within those targets in `snake_case`.
- Give a multi-file binary, example, benchmark, or integration test its own
  `kebab-case` directory with a `main.rs` entry point and `snake_case` module
  files.

## Naming and Layout Conventions

- Module files use the named-file style — `parent.rs` beside a `parent/`
  directory. `mod.rs` is forbidden (enforced by the policy script).
- Compound words in file and identifier names are underscore-separated and
  spelled in full; domain-standard acronyms (`ssa`, `cfg`, `hlsl`, `dxbc`,
  `rdef`, `isgn`, `icb`, `uav`, `expr`) stay.
- Counting accessors use the `*_count` suffix; do not introduce `num_*` names.
- Error types end in `Error` (or name the failure, like `Unsupported` and
  `Halt`) and implement `core::error::Error`.
- Use American English spelling in identifiers, documentation, and comments.
- Unit tests live inline in a `#[cfg(test)] mod tests` in the file they test;
  split them into `<parent>/tests.rs` when the file approaches the 1,000-line
  cap. Test modules are always named `tests`.

## Clean and Ergonomic Code

- Keep functions focused and control flow shallow. Prefer guard clauses when they
  make the happy path easier to follow.
- Use descriptive domain names and explicit domain types. Avoid boolean
  parameters, magic values, and ambiguous tuples when a small enum or struct
  communicates the contract better.
- Preserve the `dxbc` crate's `no_std` plus `alloc` compatibility; its `std`
  feature stays opt-in.
- Keep the decompiler's cfglib dialect language-owned: operations, effects,
  value shapes, and storage vocabulary belong to `crates/decompiler`, while
  generic function storage, web lifting, and structuring stay in cfglib. Do not
  push shader-specific variants into cfglib's generic layers.
- Avoid unnecessary cloning, allocation, collection, dynamic dispatch, and
  generic indirection. Optimize for a clear ownership story before
  micro-optimizing.
- Return structured errors for recoverable failures. Do not panic in public APIs
  unless the panic is part of a documented contract.
- Use `unwrap` or `expect` only for a proven invariant; make an `expect` message
  explain the violated invariant rather than repeat the immediate operation.
- Comments should explain intent, invariants, tradeoffs, or safety reasoning. Do
  not narrate code that is already self-explanatory.
- Do not use decorative section-divider comments made from repeated dashes,
  equals signs, or box-drawing characters, or standalone comments that merely
  name a section. Express meaningful structure with cohesive modules, types,
  functions, and interface boundaries.
- Keep public items documented, and update examples and crate-level docs when
  their behavior changes.

## DRY and Abstraction Boundaries

- Maintain one source of truth for each rule, representation, and conversion.
- Reuse existing domain types and helpers before adding a parallel
  implementation.
- Extract repeated behavior when it represents the same policy or invariant.
- Do not merge code that is merely syntactically similar but has different
  domain semantics or is likely to evolve independently.
- Keep conversions and normalization at explicit boundaries instead of
  scattering them across callers.

## Tests

- Add a regression test for every bug fix and every externally observable
  behavior change.
- Decode/encode changes must preserve the byte-identical round-trip invariant
  verified by the conformance examples and integration tests.
- Decompiler changes are validated by the recompile-and-differential gate; a
  corpus regression there is a failing test.
- Keep tests deterministic, focused, and readable.
- Do not remove or weaken a test merely to make a change pass.

## Repository Hygiene

- Inspect the working tree before editing and preserve unrelated user changes.
- Keep formatting-only churn and opportunistic refactors out of focused changes.
- Do not use destructive Git commands, discard local work, commit, push, or
  rewrite history unless the user explicitly requests it.
- Shader binaries and proprietary corpora stay untracked (`shader-raw/`,
  `shaders/`, `shaders-pc/`).
- Add a dependency only when the benefit clearly outweighs its maintenance and
  compatibility cost. Confirm that it supports the repository's targets and the
  `dxbc` crate's `no_std` requirement.
- Report the checks actually run and any checks that could not be run.
