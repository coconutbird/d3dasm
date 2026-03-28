//! PRIV chunk parser — private / debug data.
//!
//! The PRIV chunk carries tool-specific private data. It may contain a
//! GUID identifying the tool, followed by opaque payload bytes.
//! We expose the raw size and, when present, the leading GUID bytes.

use core::fmt;

use super::ChunkParser;

/// Parsed PRIV (private data) chunk.
#[derive(Debug, Clone)]
pub struct PrivateData {
    /// Total size of the private data in bytes.
    pub size: usize,
    /// First 16 bytes (GUID) if present.
    pub guid: Option<[u8; 16]>,
}

/// Parse a PRIV chunk.
pub fn parse_priv(data: &[u8]) -> Option<PrivateData> {
    let guid = if data.len() >= 16 {
        let mut g = [0u8; 16];
        g.copy_from_slice(&data[..16]);
        Some(g)
    } else {
        None
    };
    Some(PrivateData {
        size: data.len(),
        guid,
    })
}

impl ChunkParser for PrivateData {
    fn parse(data: &[u8]) -> Option<Self> {
        parse_priv(data)
    }
}

impl fmt::Display for PrivateData {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "// Private Data: {} bytes", self.size)?;
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
