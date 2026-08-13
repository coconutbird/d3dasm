//! LIBF chunk parser — library function table.
//!
//! The LIBF chunk appears in D3D11 shader libraries compiled with the
//! `lib_*` target profile.  Each entry describes one exported function
//! that can be linked via `ID3D11Linker`.
//!
//! The internal binary format has no published contract. The payload remains
//! opaque instead of assigning speculative meaning to its first dword.

use alloc::vec::Vec;
use core::fmt;

use super::{ChunkParser, ChunkWriter};
/// Parsed LIBF (library function table) chunk.
#[derive(Debug, Clone)]
pub struct LibraryFunction {
    /// Total size of the chunk payload in bytes.
    pub size: usize,
    /// Raw chunk payload for round-trip serialization.
    pub raw: Vec<u8>,
}

/// Parse a LIBF chunk.
#[must_use]
pub fn parse_libf(data: &[u8]) -> Option<LibraryFunction> {
    Some(LibraryFunction {
        size: data.len(),
        raw: Vec::from(data),
    })
}

impl ChunkParser<'_> for LibraryFunction {
    fn parse(data: &[u8]) -> Option<Self> {
        parse_libf(data)
    }
}

impl ChunkWriter for LibraryFunction {
    fn fourcc(&self) -> [u8; 4] {
        *b"LIBF"
    }

    fn write_payload(&self) -> Vec<u8> {
        self.raw.clone()
    }
}

impl fmt::Display for LibraryFunction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "// Library Functions (opaque): {} bytes", self.size)
    }
}
