//! PRIV chunk parser — private / debug data.
//!
//! The PRIV chunk carries tool-specific private data with no shared payload
//! contract. It is exposed only as opaque bytes.

use alloc::borrow::Cow;
use alloc::vec::Vec;
use core::fmt;

use super::{ChunkParser, ChunkWriter};

/// Parsed PRIV (private data) chunk.
#[derive(Debug, Clone)]
pub struct PrivateData<'a> {
    /// Raw chunk payload for round-trip serialization.
    pub raw: Cow<'a, [u8]>,
}

/// Parse a PRIV chunk.
#[must_use]
pub fn parse_priv(data: &[u8]) -> Option<PrivateData<'_>> {
    Some(PrivateData {
        raw: Cow::Borrowed(data),
    })
}

impl<'a> ChunkParser<'a> for PrivateData<'a> {
    fn parse(data: &'a [u8]) -> Option<Self> {
        parse_priv(data)
    }
}

impl ChunkWriter for PrivateData<'_> {
    fn fourcc(&self) -> [u8; 4] {
        *b"PRIV"
    }

    fn write_payload(&self) -> Vec<u8> {
        self.raw.to_vec()
    }
}

impl fmt::Display for PrivateData<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "// Private Data (opaque): {} bytes", self.raw.len())
    }
}
