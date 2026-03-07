# d3dasm

A Direct3D shader disassembler written in Rust (edition 2024).

Parses and disassembles DXBC (DirectX Bytecode) shader binaries, supporting the full SM4/SM5 (Shader Model 4/5) instruction set used by Direct3D 11.

## Features

- **DXBC container parsing** — Reads standard DXBC containers with chunk-based layout (RDEF, ISGN, OSGN, SHEX/SHDR, STAT)
- **fxb0 container support** — Automatically extracts DXBC blobs from custom `fxb0` wrapper files
- **Full SM5 opcode coverage** — All 217+ D3D11 opcodes mapped from the authoritative `d3d12TokenizedProgramFormat.hpp` spec
- **Resource definitions** — Parses RDEF chunks (constant buffers, resource bindings, variables)
- **Input/Output signatures** — Parses ISGN/OSGN chunks (semantics, register assignments, masks)
- **All shader stages** — VS, PS, GS, HS, DS, CS

## Usage

```sh
cargo run -- <shader.bin>
```

Supports both raw DXBC files and `fxb0` container files containing multiple shaders.

## Example Output

```
vs_5_0
dcl_globalFlags refactoringAllowed
dcl_input, v0.xy
dcl_output_siv o0.xyzw, position
mov o0.xy, v0.xyx
mov o0.zw, l(0.000000, 0.000000, 0.000000, 1.000000)
ret
```

## Project Structure

```
d3dasm/
├── Cargo.toml                        # Workspace root (resolver = "3")
├── crates/
│   ├── dxbc/                         # DXBC parsing & disassembly library
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── container.rs          # DXBC container parser + fxb0 scanning
│   │       ├── rdef.rs               # Resource definition (RDEF) parser
│   │       ├── signature.rs          # Input/Output signature parser
│   │       └── shex/
│   │           ├── shex.rs           # SHEX/SHDR bytecode disassembler
│   │           ├── opcodes.rs        # SM4/SM5 opcode enum and mappings
│   │           └── operand.rs        # Operand decoding
│   ├── d3dasm/                       # Generic disassembly interface
│   │   ├── Cargo.toml
│   │   └── src/
│   │       └── lib.rs               # Traits + re-exports (dxbc, future dxil)
│   └── d3dasm-cli/                   # CLI binary
│       ├── Cargo.toml
│       └── src/
│           └── main.rs
└── shaders/                          # Test shader data
```

### Crate Responsibilities

| Crate | Type | Purpose |
|-------|------|---------|
| `dxbc` | lib | Low-level DXBC container parsing, SM4/SM5 bytecode disassembly |
| `d3dasm` | lib | Generic disassembly interface over backends (dxbc, future dxil) |
| `d3dasm-cli` | bin | Command-line disassembler tool |

## Building

```sh
cargo build
```

Requires Rust with edition 2024 support.

## Development

Pre-commit hooks are managed with [prek](https://github.com/j178/prek). After cloning:

```sh
prek install
```

This installs hooks that run on every commit:

- `cargo fmt --check` — formatting
- `cargo clippy -- -D warnings` — linting
- `cargo check` — compilation
- Trailing whitespace, end-of-file, YAML, and merge conflict checks
