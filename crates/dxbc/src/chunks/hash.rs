//! HASH / XHSH chunk parser — shader content hash.

use core::fmt;

use super::ChunkParser;

/// Shader hash stored in a HASH or XHSH chunk.
#[derive(Debug, Clone)]
pub struct ShaderHash {
    /// Raw hash bytes (typically 16 bytes for MD5 in HASH, 8 bytes in XHSH).
    pub bytes: [u8; 16],
    /// Number of valid bytes in `bytes` (8 for XHSH, 16 for HASH).
    pub len: usize,
}

/// Parse a HASH or XHSH chunk.
///
/// HASH chunks contain a 4-byte flags field followed by a 16-byte MD5 digest.
/// XHSH chunks contain an 8-byte hash with no flags prefix.
/// We handle both by checking the data length.
pub fn parse_hash(data: &[u8]) -> Option<ShaderHash> {
    let mut bytes = [0u8; 16];

    if data.len() >= 20 {
        // HASH chunk: 4 bytes flags + 16 bytes MD5
        bytes.copy_from_slice(&data[4..20]);
        Some(ShaderHash { bytes, len: 16 })
    } else if data.len() >= 16 {
        // Could be a 16-byte hash without flags
        bytes.copy_from_slice(&data[..16]);
        Some(ShaderHash { bytes, len: 16 })
    } else if data.len() >= 8 {
        // XHSH: 8-byte hash
        bytes[..8].copy_from_slice(&data[..8]);
        Some(ShaderHash { bytes, len: 8 })
    } else {
        None
    }
}

impl ChunkParser for ShaderHash {
    fn parse(data: &[u8]) -> Option<Self> {
        parse_hash(data)
    }
}

impl fmt::Display for ShaderHash {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "// Hash: ")?;
        for i in 0..self.len {
            write!(f, "{:02x}", self.bytes[i])?;
        }

        writeln!(f)
    }
}
