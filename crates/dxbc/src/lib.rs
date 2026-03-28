//! `dxbc` — a `no_std` DXBC container parser, SM4/SM5 disassembler, and encoder.
//!
//! This crate can parse DXBC containers (the binary format produced by `fxc`
//! and consumed by Direct3D 10–12), decode individual chunk types (signatures,
//! resource definitions, statistics, root signatures, shader bytecode, etc.),
//! and round-trip shader bytecode through an intermediate representation.
//!
//! # Quick start
//!
//! ```rust,ignore
//! use dxbc::{DxbcContainer, parse_chunk, ChunkData};
//!
//! let container = DxbcContainer::parse(dxbc_bytes).unwrap();
//! for chunk in &container.chunks {
//!     match parse_chunk(chunk) {
//!         ChunkData::Shader(program) => {
//!             println!("{}", dxbc::shex::format_program(&program));
//!         }
//!         _ => {}
//!     }
//! }
//! ```
//!
//! # Crate features
//!
//! * **`no_std`** — the crate depends only on `alloc`; no filesystem or I/O.
//! * **Decode** — `shex::decode` turns raw SHEX/SHDR bytes into
//!   a [`Program`].
//! * **Format** — `shex::format_program` renders a `Program` as
//!   human-readable disassembly text.
//! * **Encode** — `shex::encode` serialises a `Program` back to
//!   bytes, enabling shader modification and round-trip workflows.

#![no_std]
extern crate alloc;

/// DXBC chunk parsers for every known chunk FourCC.
pub mod chunks;
/// DXBC container (header + chunk table) parser and writer.
pub mod container;
/// SM4/SM5 shader bytecode decoder, encoder, and formatter.
pub mod shex;
/// Shared helpers (C-string reading, string table builder).
pub mod util;

// Re-export the most commonly used types at the crate root.
pub use chunks::{ChunkData, parse_chunk};
pub use container::{DxbcChunk, DxbcContainer, build_dxbc, scan_dxbc};
pub use shex::Program;
pub use shex::disassemble;
