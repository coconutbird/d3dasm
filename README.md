# d3dasm

Direct3D shader bytecode disassembler and encoder written in Rust.

Parses DXBC containers, decodes SM4/SM5 shader bytecode into a fully typed intermediate representation, and produces `fxc.exe`-compatible disassembly output. The IR can be re-encoded to byte-identical bytecode, enabling shader inspection and modification workflows.

## Features

- **Full SM4/SM5 bytecode round-trip** — Decode → IR → encode produces bit-identical output across all 217+ opcodes. Verified against 1,152 real-world shader chunks with zero failures.
- **Typed IR with no raw bit hacks** — Every opcode field (GS primitives, topologies, component selection modes, sample counts, test conditions) is represented as a proper Rust enum. No raw bit buckets.
- **18 chunk types parsed** — RDEF, ISGN/ISG1, OSGN/OSG1/OSG5, PCSG/PSG1, SHEX/SHDR, STAT, RTS0, SFI0, HASH/XHSH, ILDN, ILDB, PRIV, DXIL, PSV0, RDAT, LIBF, LFS0, LIBH. Unknown chunks are preserved for lossless container rebuilds.
- **Zero-copy reads, copy-on-write mutation** — Metadata strings and raw blobs borrow directly from the input buffer via `Cow<'a, T>`. Assigning `Cow::Owned(...)` transparently enables shader patching without touching the original data.
- **Lazy chunk parsing** — Non-program chunks (RDEF, signatures, STAT, etc.) are parsed on first access via `OnceCell`. Callers that only need the shader program pay zero overhead for metadata.
- **Container scanning** — Automatically finds DXBC containers in arbitrary byte streams (raw files, `fxb0` wrappers, custom game archives).
- **`no_std` core** — The `dxbc` crate uses only `core` and `alloc`, no filesystem or I/O.
- **All shader stages** — VS, PS, GS, HS, DS, CS.

## Documentation

- [`docs/d3dasm-grammar.md`](docs/d3dasm-grammar.md) — the full grammar of the
  **current** `.d3dasm` text format (container document, RDEF-as-HLSL,
  signatures, STAT, and the SHEX instruction body), with the known
  inconsistencies a future parser must handle.
- [`docs/d3dasm-desired-grammar.md`](docs/d3dasm-desired-grammar.md) — the
  **target** redesign that resolves those inconsistencies into a consistent,
  context-free grammar (a future `v2` of the format).

## Crates

| Crate        | Type           | Purpose                                                                                       |
| ------------ | -------------- | --------------------------------------------------------------------------------------------- |
| `dxbc`       | lib (`no_std`) | DXBC container parser/writer, chunk decoders, SM4/SM5 bytecode decode/encode/format           |
| `d3dasm`     | lib            | High-level disassembly interface — wraps `dxbc` with typed accessors and `Display` formatting |
| `d3dasm-cli` | bin            | Command-line disassembler                                                                     |

## Usage

```sh
cargo run -- <shader.bin>
```

### As a library

```rust
let data = std::fs::read("shader.bin").unwrap();
for shader in d3dasm::parse(&data) {
    // Typed access to parsed chunks.
    if let Some(program) = shader.program() {
        println!("SM {}.{}", program.version_major, program.version_minor);
    }

    // fxc.exe-compatible disassembly via Display.
    print!("{shader}");
}
```

### Lower-level access via `dxbc`

```rust
let container = dxbc::DxbcContainer::parse(bytes).unwrap();
for chunk in &container.chunks {
    match chunk.parse() {
        dxbc::ChunkData::Shader(program) => {
            // Round-trip: decode → encode → identical bytes.
            let encoded = dxbc::shex::encode(&program);
            assert_eq!(chunk.data, &encoded[..]);
        }
        _ => {}
    }
}
```

## Example Output

```hlsl
// Compiled with: Microsoft (R) HLSL Shader Compiler 10.0.10011.0
//
// Resource Bindings:
// Name            Type         Dim      Slot Count Flags
// --------------- ------------ -------- ---- ----- -----
// gPointSampler0  sampler      NA       0    1
// gPointTexture0  texture      2dMS     0    1     comparisonSampler;texComp0
//
// Input Signature:
//   TEXCOORD      float v0.xy
//   TEXCOORD1     float v0.zw
//   SV_Position   float v1.xyzw
//
// Output Signature:
//   SV_Target   float v0.xyzw
//
ps_5_0
dcl_globalFlags refactoringAllowed
dcl_sampler s0, mode_default
dcl_resource_texture2d (float,float,float,float) t0
dcl_input_ps linear v0.xy
dcl_input_ps linear v0.zw
dcl_output o0.xyzw
dcl_temps 1
sample r0.xyzw, v0.xyxx, t0.xyzw, s0
mov o0.xyzw, r0.xyzw
ret
```

## Project Structure

```
d3dasm/
├── Cargo.toml                        # Virtual workspace and shared policy
├── Cargo.lock                        # Workspace-wide dependency lockfile
├── crates/
│   ├── dxbc/                         # no_std DXBC library package
│   │   ├── Cargo.toml
│   │   ├── src/
│   │   │   ├── lib.rs                # Library target
│   │   │   ├── container.rs          # Container parser, writer, scanner
│   │   │   ├── chunks/               # Per-chunk parsers and writers
│   │   │   └── shex/                 # SM4/SM5 IR, codec, and formatter
│   │   └── fuzz/                     # cargo-fuzz workspace package
│   │       ├── Cargo.toml
│   │       └── fuzz_targets/
│   ├── d3dasm/                       # High-level disassembly API
│   │   ├── Cargo.toml
│   │   ├── src/lib.rs                # Library target
│   │   └── tests/                    # Integration tests
│   └── d3dasm-cli/                   # CLI binary
│       ├── Cargo.toml
│       ├── src/main.rs               # Binary target
│       └── examples/                 # Example targets
├── .github/workflows/ci.yml          # Workspace quality gates
└── .pre-commit-config.yaml           # Matching local quality gates
```

The root is a virtual workspace; every package uses Cargo's conventional
`src/lib.rs`, `src/main.rs`, `tests/`, `examples/`, or `fuzz_targets/` target
locations. The workspace topology is listed explicitly in `Cargo.toml`, with
the fuzz package included in workspace checks but excluded from default builds.

Shared package metadata and every dependency declaration live in the root
`[workspace.package]` and `[workspace.dependencies]` tables. Member manifests
inherit them with `workspace = true`. The same pattern applies to
`[workspace.lints]`: every member enables `workspace = true`, and Clippy's
`pedantic` group is denied without per-crate Clippy exceptions.

## Building

```sh
cargo build
```

Requires Rust 1.85+ (edition 2024, `core::cell::OnceCell` stable since 1.70).

## Testing

```sh
cargo test --workspace --all-targets --all-features --exclude dxbc-fuzz
cargo clippy --workspace --all-targets --all-features -- -D warnings
```

The fuzz harness is checked and linted with the workspace, but is excluded from
`cargo test --all-targets` because libFuzzer supplies its entry point only when
run through `cargo fuzz`.

Round-trip verification across all test shaders:

```sh
cargo run --release --example roundtrip -- shaders/shaders-pc/**/*.bin
```
