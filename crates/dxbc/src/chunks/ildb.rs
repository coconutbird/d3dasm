//! ILDB chunk parser — embedded debug data blob.
//!
//! The ILDB chunk contains an embedded PDB or debug info blob.
//! We don't parse the PDB contents — just expose the raw bytes and size.

use core::fmt;

use super::ChunkParser;

/// Parsed ILDB (debug data) chunk.
#[derive(Debug, Clone)]
pub struct DebugData {
    /// Size of the debug data in bytes.
    pub size: usize,
}

/// Parse an ILDB chunk.
pub fn parse_ildb(data: &[u8]) -> Option<DebugData> {
    Some(DebugData { size: data.len() })
}

impl ChunkParser for DebugData {
    fn parse(data: &[u8]) -> Option<Self> {
        parse_ildb(data)
    }
}

impl fmt::Display for DebugData {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "// Debug Data: {} bytes", self.size)
    }
}
