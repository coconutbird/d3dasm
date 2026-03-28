//! LIBF chunk parser — library function table.
//!
//! The LIBF chunk appears in D3D11 shader libraries compiled with the
//! `lib_*` target profile.  Each entry describes one exported function
//! that can be linked via `ID3D11Linker`.
//!
//! The internal binary format is undocumented — we expose the raw size
//! and attempt to read a leading function count (u32) when the chunk
//! is large enough.

use alloc::vec::Vec;
use core::fmt;

use super::{ChunkParser, ChunkWriter};
use nostdio::{ReadLe, SliceCursor};

/// Parsed LIBF (library function table) chunk.
#[derive(Debug, Clone)]
pub struct LibraryFunction {
    /// Number of exported functions (first u32), if parseable.
    pub function_count: Option<u32>,
    /// Total size of the chunk payload in bytes.
    pub size: usize,
    /// Raw chunk payload for round-trip serialization.
    pub raw: Vec<u8>,
}

/// Parse a LIBF chunk.
pub fn parse_libf(data: &[u8]) -> Option<LibraryFunction> {
    let function_count = if data.len() >= 4 {
        let mut c = SliceCursor::new(data);
        Some(c.read_u32_le().ok()?)
    } else {
        None
    };
    Some(LibraryFunction {
        function_count,
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
        write!(f, "// Library Functions: ")?;
        if let Some(count) = self.function_count {
            write!(f, "{} entries", count)?;
        } else {
            write!(f, "empty")?;
        }
        writeln!(f, " ({} bytes)", self.size)
    }
}
