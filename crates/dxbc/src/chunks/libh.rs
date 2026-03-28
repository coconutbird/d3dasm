//! LIBH chunk parser — library header.
//!
//! The LIBH chunk appears alongside LIBF and LFS0 in D3D11 shader
//! library containers.  It stores top-level metadata for the library
//! (creator info, version, etc.).
//!
//! The internal binary format is undocumented — we expose the raw size.

use core::fmt;

use super::ChunkParser;

/// Parsed LIBH (library header) chunk.
#[derive(Debug, Clone)]
pub struct LibraryHeader {
    /// Total size of the chunk payload in bytes.
    pub size: usize,
}

/// Parse a LIBH chunk.
pub fn parse_libh(data: &[u8]) -> Option<LibraryHeader> {
    Some(LibraryHeader { size: data.len() })
}

impl ChunkParser for LibraryHeader {
    fn parse(data: &[u8]) -> Option<Self> {
        parse_libh(data)
    }
}

impl fmt::Display for LibraryHeader {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "// Library Header: {} bytes", self.size)
    }
}
