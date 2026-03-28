//! HASH / XHSH chunk parser — shader content hash.

use core::fmt;

use super::{ChunkParser, ChunkWriter};

/// Shader hash stored in a HASH or XHSH chunk.
#[derive(Debug, Clone)]
pub struct ShaderHash {
    /// Raw hash bytes (typically 16 bytes for MD5 in HASH, 8 bytes in XHSH).
    pub bytes: [u8; 16],
    /// Number of valid bytes in `bytes` (8 for XHSH, 16 for HASH).
    pub len: usize,
    /// FourCC that was used to parse this chunk (HASH or XHSH).
    pub fourcc: [u8; 4],
}

/// Parse a HASH or XHSH chunk.
///
/// `fourcc` should be the original FourCC from the container so it is
/// preserved correctly for round-trip serialization. HASH chunks contain
/// a 4-byte flags field followed by a 16-byte MD5 digest. XHSH chunks
/// contain an 8-byte hash with no flags prefix.
pub fn parse_hash(data: &[u8], fourcc: [u8; 4]) -> Option<ShaderHash> {
    let mut bytes = [0u8; 16];
    let is_xhsh = &fourcc == b"XHSH";

    if is_xhsh {
        // XHSH: 8-byte hash, no flags prefix
        if data.len() >= 8 {
            bytes[..8].copy_from_slice(&data[..8]);
            Some(ShaderHash {
                bytes,
                len: 8,
                fourcc,
            })
        } else {
            None
        }
    } else if data.len() >= 20 {
        // HASH chunk: 4 bytes flags + 16 bytes MD5
        bytes.copy_from_slice(&data[4..20]);
        Some(ShaderHash {
            bytes,
            len: 16,
            fourcc,
        })
    } else if data.len() >= 16 {
        // Could be a 16-byte hash without flags
        bytes.copy_from_slice(&data[..16]);
        Some(ShaderHash {
            bytes,
            len: 16,
            fourcc,
        })
    } else {
        None
    }
}

impl ChunkParser for ShaderHash {
    fn parse(data: &[u8]) -> Option<Self> {
        // Default to HASH when called via the generic ChunkParser trait.
        // Prefer calling parse_hash(data, fourcc) directly when the FourCC is known.
        parse_hash(data, *b"HASH")
    }
}

impl ChunkWriter for ShaderHash {
    fn fourcc(&self) -> [u8; 4] {
        self.fourcc
    }

    fn write_payload(&self) -> alloc::vec::Vec<u8> {
        let mut buf = alloc::vec::Vec::new();
        if self.len == 16 {
            // HASH: 4-byte flags (0) + 16-byte digest
            buf.extend_from_slice(&0u32.to_le_bytes());
            buf.extend_from_slice(&self.bytes[..16]);
        } else {
            // XHSH: 8-byte hash
            buf.extend_from_slice(&self.bytes[..8]);
        }
        buf
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
