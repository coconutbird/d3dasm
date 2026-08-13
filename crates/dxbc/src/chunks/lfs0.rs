//! LFS0 chunk parser — library function signatures.
//!
//! The LFS0 chunk accompanies LIBF in D3D11 shader library containers.
//! It holds per-function input/output signature data used by the linker
//! to validate function calls across separately-compiled modules.
//!
//! The internal binary format has no published contract. The payload remains
//! opaque instead of assigning speculative meaning to its first dword.

use alloc::vec::Vec;
use core::fmt;

use super::{ChunkParser, ChunkWriter};
/// Parsed LFS0 (library function signatures) chunk.
#[derive(Debug, Clone)]
pub struct LibraryFunctionSignatures {
    /// Total size of the chunk payload in bytes.
    pub size: usize,
    /// Raw chunk payload for round-trip serialization.
    pub raw: Vec<u8>,
}

/// Parse an LFS0 chunk.
#[must_use]
pub fn parse_lfs0(data: &[u8]) -> Option<LibraryFunctionSignatures> {
    Some(LibraryFunctionSignatures {
        size: data.len(),
        raw: Vec::from(data),
    })
}

impl ChunkParser<'_> for LibraryFunctionSignatures {
    fn parse(data: &[u8]) -> Option<Self> {
        parse_lfs0(data)
    }
}

impl ChunkWriter for LibraryFunctionSignatures {
    fn fourcc(&self) -> [u8; 4] {
        *b"LFS0"
    }

    fn write_payload(&self) -> Vec<u8> {
        self.raw.clone()
    }
}

impl fmt::Display for LibraryFunctionSignatures {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "// Library Function Signatures (opaque): {} bytes",
            self.size
        )
    }
}
