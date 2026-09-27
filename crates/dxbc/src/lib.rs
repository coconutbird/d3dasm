//! `dxbc` — a `no_std` DXBC container parser, SM4/SM5 disassembler, and encoder.
//!
//! This crate can parse DXBC containers (the binary format produced by `fxc`
//! and consumed by Direct3D 10–12), decode individual chunk types (signatures,
//! resource definitions, statistics, root signatures, shader bytecode, etc.),
//! and round-trip shader bytecode through an intermediate representation.
//!
//! Metadata strings and raw blobs borrow from the input buffer via
//! `Cow<'a, T>` — zero allocation on read, transparent ownership on mutation.
//!
//! # Quick start
//!
//! ```rust,no_run
//! use dxbc::{ChunkData, scan_dxbc};
//!
//! # let dxbc_bytes = &[0u8];
//! let containers = scan_dxbc(dxbc_bytes);
//! for container in &containers {
//!     for chunk in &container.chunks {
//!         match chunk.parse() {
//!             ChunkData::Shader(program) => {
//!                 // Program implements Display for fxc-compatible disassembly.
//!                 println!("{program}");
//!             }
//!             _ => {}
//!         }
//!     }
//! }
//! ```
//!
//! # Crate features
//!
//! * **`no_std`** — the crate depends only on `core` and `alloc`; no filesystem or I/O.
//! * **Zero-copy metadata** — chunk strings and blobs borrow from the input
//!   via `Cow<'a, T>`, supporting read-then-patch workflows.
//! * **Decode** — `shex::decode` turns raw SHEX/SHDR bytes into a [`Program`].
//! * **Display** — `Program` and all chunk types implement `Display` for
//!   `fxc.exe`-compatible disassembly output.
//! * **Encode** — `shex::encode` serialises a `Program` back to
//!   bytes, enabling shader modification and round-trip workflows.
//!
//! The lossless `.d3dasm` *text* grammar (serialize/parse) lives in the
//! `d3dasm` crate (`d3dasm::grammar`), built on this crate's IR.

#![no_std]
extern crate alloc;

/// DXBC container checksum (the modified-MD5 header digest).
pub mod checksum;
/// Typed parsers for recognized DXBC chunk FourCCs; unknown data stays opaque.
pub mod chunks;
/// DXBC container (header + chunk table) parser and writer.
pub mod container;
/// SM4/SM5 shader bytecode decoder, encoder, and formatter.
pub mod shex;
/// Shared helpers (C-string reading, string table builder).
pub mod util;

// Re-export the most commonly used types at the crate root.
pub use chunks::ChunkData;
pub use container::{ContainerVersion, DxbcChunk, DxbcContainer, build_dxbc, scan_dxbc};
pub use shex::Program;
pub use shex::disassemble;
