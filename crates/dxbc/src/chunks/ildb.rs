//! ILDB chunk parser — embedded debug data blob.
//!
//! The ILDB chunk contains an embedded PDB or debug info blob.
//! We don't parse the PDB contents — just expose the raw bytes and size.

use alloc::vec::Vec;
use core::fmt;

use super::{ChunkParser, ChunkWriter};

/// Parsed ILDB (debug data) chunk.
#[derive(Debug, Clone)]
pub struct DebugData {
    /// Size of the debug data in bytes.
    pub size: usize,
    /// Raw chunk payload for round-trip serialization.
    pub raw: Vec<u8>,
}

/// Parse an ILDB chunk.
pub fn parse_ildb(data: &[u8]) -> Option<DebugData> {
    Some(DebugData {
        size: data.len(),
        raw: Vec::from(data),
    })
}

impl ChunkParser for DebugData {
    fn parse(data: &[u8]) -> Option<Self> {
        parse_ildb(data)
    }
}

impl ChunkWriter for DebugData {
    fn fourcc(&self) -> [u8; 4] {
        *b"ILDB"
    }

    fn write_payload(&self) -> Vec<u8> {
        self.raw.clone()
    }
}

impl fmt::Display for DebugData {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "// Debug Data: {} bytes", self.size)
    }
}
