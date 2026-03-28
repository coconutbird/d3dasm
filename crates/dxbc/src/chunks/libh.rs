//! LIBH chunk parser — library header.
//!
//! The LIBH chunk appears alongside LIBF and LFS0 in D3D11 shader
//! library containers.  It stores top-level metadata for the library
//! (creator info, version, etc.).
//!
//! The internal binary format is undocumented — we expose the raw size.

use alloc::vec::Vec;
use core::fmt;

use super::{ChunkParser, ChunkWriter};

/// Parsed LIBH (library header) chunk.
#[derive(Debug, Clone)]
pub struct LibraryHeader {
    /// Total size of the chunk payload in bytes.
    pub size: usize,
    /// Raw chunk payload for round-trip serialization.
    pub raw: Vec<u8>,
}

/// Parse a LIBH chunk.
pub fn parse_libh(data: &[u8]) -> Option<LibraryHeader> {
    Some(LibraryHeader {
        size: data.len(),
        raw: Vec::from(data),
    })
}

impl ChunkParser for LibraryHeader {
    fn parse(data: &[u8]) -> Option<Self> {
        parse_libh(data)
    }
}

impl ChunkWriter for LibraryHeader {
    fn fourcc(&self) -> [u8; 4] {
        *b"LIBH"
    }

    fn write_payload(&self) -> Vec<u8> {
        self.raw.clone()
    }
}

impl fmt::Display for LibraryHeader {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "// Library Header: {} bytes", self.size)
    }
}
