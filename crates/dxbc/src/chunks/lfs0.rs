//! LFS0 chunk parser — library function signatures.
//!
//! The LFS0 chunk accompanies LIBF in D3D11 shader library containers.
//! It holds per-function input/output signature data used by the linker
//! to validate function calls across separately-compiled modules.
//!
//! The internal binary format is undocumented — we expose the raw size
//! and attempt to read a leading count (u32).

use alloc::vec::Vec;
use core::fmt;

use super::{ChunkParser, ChunkWriter};
use nostdio::{ReadLe, SliceCursor};

/// Parsed LFS0 (library function signatures) chunk.
#[derive(Debug, Clone)]
pub struct LibraryFunctionSignatures {
    /// Signature count (first u32), if parseable.
    pub signature_count: Option<u32>,
    /// Total size of the chunk payload in bytes.
    pub size: usize,
    /// Raw chunk payload for round-trip serialization.
    pub raw: Vec<u8>,
}

/// Parse an LFS0 chunk.
pub fn parse_lfs0(data: &[u8]) -> Option<LibraryFunctionSignatures> {
    let signature_count = if data.len() >= 4 {
        let mut c = SliceCursor::new(data);
        Some(c.read_u32_le().ok()?)
    } else {
        None
    };
    Some(LibraryFunctionSignatures {
        signature_count,
        size: data.len(),
        raw: Vec::from(data),
    })
}

impl ChunkParser for LibraryFunctionSignatures {
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
        write!(f, "// Library Function Signatures: ")?;
        if let Some(count) = self.signature_count {
            write!(f, "{} entries", count)?;
        } else {
            write!(f, "empty")?;
        }
        writeln!(f, " ({} bytes)", self.size)
    }
}
