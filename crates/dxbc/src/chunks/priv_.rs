//! PRIV chunk parser — private / debug data.
//!
//! The PRIV chunk carries tool-specific private data. It may contain a
//! GUID identifying the tool, followed by opaque payload bytes.
//! We expose the raw bytes via `Cow<'a, [u8]>` and, when present, the
//! leading GUID bytes.

use alloc::borrow::Cow;
use alloc::vec::Vec;
use core::fmt;

use super::{ChunkParser, ChunkWriter};

/// Parsed PRIV (private data) chunk.
#[derive(Debug, Clone)]
pub struct PrivateData<'a> {
    /// First 16 bytes (GUID) if present.
    pub guid: Option<[u8; 16]>,
    /// Raw chunk payload for round-trip serialization.
    pub raw: Cow<'a, [u8]>,
}

/// Parse a PRIV chunk.
pub fn parse_priv<'a>(data: &'a [u8]) -> Option<PrivateData<'a>> {
    let guid = if data.len() >= 16 {
        let mut g = [0u8; 16];
        g.copy_from_slice(&data[..16]);
        Some(g)
    } else {
        None
    };
    Some(PrivateData {
        guid,
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
        write!(f, "// Private Data: {} bytes", self.raw.len())?;
        if let Some(g) = &self.guid {
            write!(
                f,
                " (GUID: {:02x}{:02x}{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}{:02x}{:02x}{:02x}{:02x})",
                g[0],
                g[1],
                g[2],
                g[3],
                g[4],
                g[5],
                g[6],
                g[7],
                g[8],
                g[9],
                g[10],
                g[11],
                g[12],
                g[13],
                g[14],
                g[15]
            )?;
        }
        writeln!(f)
    }
}
