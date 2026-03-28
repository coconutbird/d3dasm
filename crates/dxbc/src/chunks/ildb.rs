//! ILDB chunk parser — embedded debug data blob.
//!
//! The ILDB chunk contains an embedded PDB or debug info blob.
//! We don't parse the PDB contents — just expose the raw bytes via
//! `Cow<'a, [u8]>` (zero-copy borrow on read, owned on mutation).

use alloc::borrow::Cow;
use alloc::vec::Vec;
use core::fmt;

use super::{ChunkParser, ChunkWriter};

/// Parsed ILDB (debug data) chunk.
#[derive(Debug, Clone)]
pub struct DebugData<'a> {
    /// Raw chunk payload for round-trip serialization.
    pub raw: Cow<'a, [u8]>,
}

/// Parse an ILDB chunk.
pub fn parse_ildb<'a>(data: &'a [u8]) -> Option<DebugData<'a>> {
    Some(DebugData {
        raw: Cow::Borrowed(data),
    })
}

impl<'a> ChunkParser<'a> for DebugData<'a> {
    fn parse(data: &'a [u8]) -> Option<Self> {
        parse_ildb(data)
    }
}

impl ChunkWriter for DebugData<'_> {
    fn fourcc(&self) -> [u8; 4] {
        *b"ILDB"
    }

    fn write_payload(&self) -> Vec<u8> {
        self.raw.to_vec()
    }
}

impl fmt::Display for DebugData<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "// Debug Data: {} bytes", self.raw.len())
    }
}
