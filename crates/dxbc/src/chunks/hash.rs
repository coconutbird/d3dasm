//! HASH chunk parser — DXIL shader content hash.

use core::fmt;

use super::{ChunkParser, ChunkWriter};

/// The published `DxilShaderHash` payload stored in a HASH chunk.
#[derive(Debug, Clone)]
pub struct ShaderHash {
    /// HASH flags (`DxilShaderHash::Flags`).
    pub flags: u32,
    /// The 16-byte digest.
    pub bytes: [u8; 16],
}

/// Parse a HASH chunk.
///
/// The payload is exactly a four-byte flags field followed by a 16-byte
/// digest, matching Microsoft's `DxilShaderHash` structure. No public
/// Microsoft contract was found for `XHSH`, so that FourCC is deliberately
/// left opaque by the container dispatcher.
#[must_use]
pub fn parse_hash(data: &[u8], fourcc: [u8; 4]) -> Option<ShaderHash> {
    if &fourcc != b"HASH" || data.len() != 20 {
        None
    } else {
        let flags = u32::from_le_bytes(data[..4].try_into().ok()?);
        let bytes = data[4..20].try_into().ok()?;
        Some(ShaderHash { flags, bytes })
    }
}

impl ChunkParser<'_> for ShaderHash {
    fn parse(data: &[u8]) -> Option<Self> {
        parse_hash(data, *b"HASH")
    }
}

impl ChunkWriter for ShaderHash {
    fn fourcc(&self) -> [u8; 4] {
        *b"HASH"
    }

    fn write_payload(&self) -> alloc::vec::Vec<u8> {
        let mut buf = alloc::vec::Vec::new();
        buf.extend_from_slice(&self.flags.to_le_bytes());
        buf.extend_from_slice(&self.bytes);
        buf
    }
}

impl fmt::Display for ShaderHash {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "// Hash: ")?;
        for byte in self.bytes {
            write!(f, "{byte:02x}")?;
        }

        if self.flags != 0 {
            write!(f, " (flags=0x{:08X})", self.flags)?;
        }
        writeln!(f)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_published_dxil_shader_hash_layout() {
        let mut bytes = [0u8; 20];
        bytes[..4].copy_from_slice(&1u32.to_le_bytes());
        bytes[4..].copy_from_slice(&[0x5a; 16]);

        let parsed = parse_hash(&bytes, *b"HASH").expect("valid HASH");
        assert_eq!(parsed.flags, 1);
        assert_eq!(parsed.bytes, [0x5a; 16]);
        assert_eq!(parsed.write_payload(), bytes);
        assert!(parse_hash(&bytes[..8], *b"XHSH").is_none());
    }
}
