use alloc::vec::Vec;
use core::fmt;

use crate::util::read_u32;

const DXBC_MAGIC: &[u8; 4] = b"DXBC";

#[derive(Debug)]
pub struct DxbcContainer<'a> {
    pub offset_in_file: usize,
    pub total_size: u32,
    pub chunks: Vec<DxbcChunk<'a>>,
}

#[derive(Debug)]
pub struct DxbcChunk<'a> {
    pub fourcc: [u8; 4],
    pub size: u32,
    pub data: &'a [u8],
}

impl DxbcChunk<'_> {
    pub fn fourcc_str(&self) -> &str {
        core::str::from_utf8(&self.fourcc).unwrap_or("????")
    }
}

impl fmt::Display for DxbcContainer<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "DXBC at 0x{:X}, size={}, chunks={}",
            self.offset_in_file,
            self.total_size,
            self.chunks.len()
        )
    }
}

/// Scan a byte slice for all DXBC containers and parse them.
pub fn scan_dxbc<'a>(data: &'a [u8]) -> Vec<DxbcContainer<'a>> {
    let mut results = Vec::new();
    let mut pos = 0;
    while pos + 4 <= data.len() {
        if let Some(offset) = find_magic(data, pos)
            && let Some(container) = parse_dxbc(data, offset)
        {
            let end = offset + container.total_size as usize;
            results.push(container);
            pos = end;
            continue;
        }
        break;
    }
    results
}

fn find_magic(data: &[u8], start: usize) -> Option<usize> {
    let needle = DXBC_MAGIC;
    data[start..]
        .windows(4)
        .position(|w| w == needle)
        .map(|p| start + p)
}

fn parse_dxbc<'a>(data: &'a [u8], offset: usize) -> Option<DxbcContainer<'a>> {
    if offset + 0x20 > data.len() {
        return None;
    }
    let total_size = read_u32(data, offset + 0x18);
    let chunk_count = read_u32(data, offset + 0x1C) as usize;

    if offset + total_size as usize > data.len() {
        return None;
    }

    let mut chunks = Vec::with_capacity(chunk_count);
    for i in 0..chunk_count {
        let chunk_offset_rel = read_u32(data, offset + 0x20 + i * 4) as usize;
        let chunk_abs = offset + chunk_offset_rel;
        if chunk_abs + 8 > data.len() {
            break;
        }
        let mut fourcc = [0u8; 4];
        fourcc.copy_from_slice(&data[chunk_abs..chunk_abs + 4]);
        let chunk_size = read_u32(data, chunk_abs + 4) as usize;
        let chunk_data_start = chunk_abs + 8;
        let chunk_data_end = (chunk_data_start + chunk_size).min(data.len());

        chunks.push(DxbcChunk {
            fourcc,
            size: chunk_size as u32,
            data: &data[chunk_data_start..chunk_data_end],
        });
    }

    Some(DxbcContainer {
        offset_in_file: offset,
        total_size,
        chunks,
    })
}

#[cfg(test)]
mod tests {
    use alloc::format;
    use alloc::vec::Vec;

    use super::*;

    /// Build a minimal valid DXBC container with the given chunks.
    /// Each chunk is (fourcc, data).
    fn build_dxbc(chunks: &[(&[u8; 4], &[u8])]) -> Vec<u8> {
        // DXBC header: magic(4) + hash(16) + 1(4) + total_size(4) + chunk_count(4) = 32 bytes
        // Then chunk_count * 4 bytes for chunk offsets
        // Then each chunk: fourcc(4) + size(4) + data
        let header_size = 0x20 + chunks.len() * 4;
        let mut chunk_data_sections: Vec<Vec<u8>> = Vec::new();
        for &(_, data) in chunks {
            let mut section = Vec::new();
            // We'll fill fourcc+size later in the final assembly
            section.extend_from_slice(data);
            chunk_data_sections.push(section);
        }

        // Calculate total size
        let chunks_total: usize = chunks
            .iter()
            .map(|(_, data)| 8 + data.len()) // fourcc + size + data
            .sum();
        let total_size = header_size + chunks_total;

        let mut buf = Vec::with_capacity(total_size);
        // Magic
        buf.extend_from_slice(b"DXBC");
        // Hash (16 bytes of zeros — we don't validate it)
        buf.extend_from_slice(&[0u8; 16]);
        // Version = 1
        buf.extend_from_slice(&1u32.to_le_bytes());
        // Total size
        buf.extend_from_slice(&(total_size as u32).to_le_bytes());
        // Chunk count
        buf.extend_from_slice(&(chunks.len() as u32).to_le_bytes());

        // Chunk offset table
        let mut offset = header_size;
        for &(_, data) in chunks {
            buf.extend_from_slice(&(offset as u32).to_le_bytes());
            offset += 8 + data.len();
        }

        // Chunk data
        for (i, &(fourcc, data)) in chunks.iter().enumerate() {
            buf.extend_from_slice(fourcc);
            buf.extend_from_slice(&(data.len() as u32).to_le_bytes());
            buf.extend_from_slice(data);
            let _ = i;
        }

        buf
    }

    #[test]
    fn parse_single_chunk() {
        let shex_data = [0u8; 16]; // dummy SHEX payload
        let dxbc = build_dxbc(&[(b"SHEX", &shex_data)]);
        let containers = scan_dxbc(&dxbc);
        assert_eq!(containers.len(), 1);
        assert_eq!(containers[0].chunks.len(), 1);
        assert_eq!(containers[0].chunks[0].fourcc_str(), "SHEX");
        assert_eq!(containers[0].chunks[0].size, 16);
        assert_eq!(containers[0].chunks[0].data.len(), 16);
    }

    #[test]
    fn parse_multiple_chunks() {
        let rdef_data = [1u8; 32];
        let isgn_data = [2u8; 24];
        let osgn_data = [3u8; 24];
        let shex_data = [4u8; 64];
        let dxbc = build_dxbc(&[
            (b"RDEF", &rdef_data),
            (b"ISGN", &isgn_data),
            (b"OSGN", &osgn_data),
            (b"SHEX", &shex_data),
        ]);
        let containers = scan_dxbc(&dxbc);
        assert_eq!(containers.len(), 1);
        assert_eq!(containers[0].chunks.len(), 4);
        assert_eq!(containers[0].chunks[0].fourcc_str(), "RDEF");
        assert_eq!(containers[0].chunks[1].fourcc_str(), "ISGN");
        assert_eq!(containers[0].chunks[2].fourcc_str(), "OSGN");
        assert_eq!(containers[0].chunks[3].fourcc_str(), "SHEX");
        assert_eq!(containers[0].chunks[3].data.len(), 64);
    }

    #[test]
    fn parse_back_to_back_containers() {
        let shex1 = [0u8; 8];
        let shex2 = [1u8; 8];
        let dxbc1 = build_dxbc(&[(b"SHEX", &shex1)]);
        let dxbc2 = build_dxbc(&[(b"SHDR", &shex2)]);
        let mut combined = dxbc1.clone();
        combined.extend_from_slice(&dxbc2);
        let containers = scan_dxbc(&combined);
        assert_eq!(containers.len(), 2);
        assert_eq!(containers[0].chunks[0].fourcc_str(), "SHEX");
        assert_eq!(containers[1].chunks[0].fourcc_str(), "SHDR");
    }

    #[test]
    fn empty_input() {
        assert!(scan_dxbc(&[]).is_empty());
    }

    #[test]
    fn no_dxbc_magic() {
        assert!(scan_dxbc(b"not a dxbc file at all").is_empty());
    }

    #[test]
    fn truncated_header() {
        // Just the magic, not enough for a full header
        assert!(scan_dxbc(b"DXBC").is_empty());
    }

    #[test]
    fn display_format() {
        let dxbc = build_dxbc(&[(b"SHEX", &[0u8; 8])]);
        let containers = scan_dxbc(&dxbc);
        let display = format!("{}", containers[0]);
        assert!(display.contains("DXBC at 0x0"), "got: {display}");
        assert!(display.contains("chunks=1"), "got: {display}");
    }

    #[test]
    fn chunk_data_integrity() {
        let payload: Vec<u8> = (0..128).collect();
        let dxbc = build_dxbc(&[(b"TEST", &payload)]);
        let containers = scan_dxbc(&dxbc);
        assert_eq!(containers[0].chunks[0].data, payload);
    }
}
