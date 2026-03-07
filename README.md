# d3dasm

A zero-dependency Direct3D shader bytecode disassembler written in Rust.

Parses DXBC (DirectX Bytecode) shader binaries and produces human-readable assembly output matching the output of Microsoft's `fxc.exe` compiler. Supports the full SM4/SM5 instruction set used by Direct3D 11 across all shader stages.

## Features

- **DXBC container parsing** — Reads standard DXBC containers with chunk-based layout (RDEF, ISGN, OSGN, SHEX/SHDR, STAT)
- **fxb0 container support** — Automatically scans and extracts DXBC blobs from `fxb0` wrapper files containing multiple shaders
- **Full SM5 opcode coverage** — All 217+ opcodes mapped from the `d3d12TokenizedProgramFormat.hpp` specification
- **Parser → IR → Formatter pipeline** — Clean three-stage architecture: binary decoding into a structured IR, then formatted to text
- **Resource definitions** — Parses RDEF chunks including constant buffers, resource bindings, and SM5 variable descriptors
- **Input/Output signatures** — Parses ISGN/OSGN/ISG1/OSG1/OSG5 chunks with semantic names, register assignments, and component masks
- **All shader stages** — VS, PS, GS, HS, DS, CS
- **Operand decoding** — Handles swizzles, masks, scalar selects, relative indexing, nested operands, immediate constants (32-bit and 64-bit), and extended modifiers (negate, absolute value)
- **Declaration support** — Constant buffers (immediate/dynamic indexed), samplers, resources, UAVs, temps, indexable temps, thread groups, tessellation parameters, GS topology, and system values

## Usage

```sh
cargo run -- <shader.bin>
```

Accepts both raw DXBC files and `fxb0` container files containing multiple shaders.

## Example Output

```hlsl
// Compiled with: Microsoft (R) HLSL Shader Compiler 10.0.10011.0
//
// Resource Bindings:
// Name                         Type         Dim      Slot
// ---------------------------- ------------ -------- ----
// $Globals                       cbuffer               4
// TerrainCompositePSC            cbuffer               5
//
// cbuffer $Globals (16 bytes)
//   g_explicitMipValue             offset=0    size=4
//   g_RCPnumLayers                 offset=4    size=4
//
// Input Signature:
//   SV_Position                     float v0.xyzw
//   TEXCOORD                     float v1.xy
//
vs_5_0
dcl_globalFlags refactoringAllowed
dcl_constantbuffer cb4[1].xyzw, immediateIndexed
dcl_constantbuffer cb5[32].xyzw, dynamicIndexed
dcl_input_sgv v0.x, vertex_id
dcl_input v1.xyzw
dcl_output_siv o0.xyzw, position
dcl_output o1.xy
dcl_temps 1
mov o0.xyzw, v1.xyzw
mov o1.xy, v3.xy
utof r0.x, v0.x
mul r0.x, r0.x, l(0.041667)
mul o1.zw, v2.xxxy, cb5[r0.x].yyyz
ret
```

## Architecture

```
  ┌──────────┐     ┌────────────┐     ┌───────────┐
  │  Binary   │────▶│  Decoder   │────▶│    IR     │────▶  Assembly text
  │  (DXBC)   │     │ decode.rs  │     │  ir.rs    │      fmt.rs
  └──────────┘     └────────────┘     └───────────┘
```

The disassembler uses a three-stage pipeline:

1. **Decode** (`decode.rs`) — Reads raw token streams and produces a structured intermediate representation. Handles operand token layout, extended operand modifiers, index representations (immediate, relative, relative+offset), and component selection modes.
2. **IR** (`ir.rs`) — Type-safe representation of the full instruction set: opcodes, operands, register types, component selects, index modes, and declaration metadata.
3. **Format** (`fmt.rs`) — Walks the IR and emits `fxc.exe`-compatible assembly text. Handles instruction-aware swizzle truncation (e.g., `dp3` always shows 3 source components regardless of destination mask), operand modifier formatting, and immediate value display.

## Project Structure

```
d3dasm/
├── Cargo.toml                        # Workspace root (resolver = "3")
├── crates/
│   ├── dxbc/                         # DXBC parsing & disassembly library
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── container.rs          # DXBC container parser + fxb0 scanning
│   │       ├── rdef.rs               # Resource definition (RDEF) parser
│   │       ├── signature.rs          # Input/Output signature parser
│   │       └── shex/
│   │           ├── decode.rs         # Token stream → IR decoder
│   │           ├── ir.rs             # Structured intermediate representation
│   │           ├── fmt.rs            # IR → assembly text formatter
│   │           └── opcodes.rs        # SM4/SM5 opcode enum (217+ opcodes)
│   ├── d3dasm/                       # Generic disassembly interface
│   │   └── src/
│   │       └── lib.rs               # Shader/ShaderContainer traits + scan()
│   └── d3dasm-cli/                   # CLI binary
│       └── src/
│           └── main.rs
```

| Crate | Type | Purpose |
|-------|------|---------|
| `dxbc` | lib | Low-level DXBC container parsing, RDEF/signature/SHEX decoding, SM4/SM5 bytecode disassembly |
| `d3dasm` | lib | Backend-agnostic disassembly traits and scanning interface |
| `d3dasm-cli` | bin | Command-line disassembler tool |

## Building

```sh
cargo build
```

Requires Rust 1.85+ (edition 2024).

## Testing

```sh
cargo test --all
```

## Development

Pre-commit hooks are configured via [pre-commit](https://pre-commit.com/). After cloning:

```sh
pre-commit install
```

Hooks run automatically on every commit:

- `cargo fmt --check` — formatting
- `cargo clippy -- -D warnings` — linting
- `cargo check` — compilation
- Trailing whitespace, end-of-file fixes, merge conflict detection

## Reference

The decoder is implemented against the operand token layout defined in Microsoft's `d3d12TokenizedProgramFormat.hpp`. Key bit-field positions:

| Field | Bits | Description |
|-------|------|-------------|
| Num Components | [1:0] | 0=none, 1=scalar, 2=four-component |
| Selection Mode | [3:2] | 0=mask, 1=swizzle, 2=select_1 |
| Component Data | [11:4] | Mask, swizzle, or scalar index |
| Operand Type | [19:12] | Register type (temp, input, output, CB, etc.) |
| Index Dimension | [21:20] | Number of index levels (0–3) |
| Index Repr | [30:22] | Encoding per index (imm32, imm64, relative, relative+imm) |
| Extended | [31] | Extended operand token follows |
