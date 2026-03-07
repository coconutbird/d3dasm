use std::fmt;

const DXBC_MAGIC: &[u8; 4] = b"DXBC";

#[derive(Debug)]
pub struct DxbcContainer {
    pub offset_in_file: usize,
    pub total_size: u32,
    pub chunks: Vec<DxbcChunk>,
}

#[derive(Debug)]
pub struct DxbcChunk {
    pub fourcc: [u8; 4],
    pub size: u32,
    pub data: Vec<u8>,
}

impl DxbcChunk {
    pub fn fourcc_str(&self) -> &str {
        std::str::from_utf8(&self.fourcc).unwrap_or("????")
    }
}

impl fmt::Display for DxbcContainer {
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

fn read_u32(data: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(data[offset..offset + 4].try_into().unwrap())
}

/// Scan a byte slice for all DXBC containers and parse them.
pub fn scan_dxbc(data: &[u8]) -> Vec<DxbcContainer> {
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

fn parse_dxbc(data: &[u8], offset: usize) -> Option<DxbcContainer> {
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
        let chunk_data = data[chunk_data_start..chunk_data_end].to_vec();

        chunks.push(DxbcChunk {
            fourcc,
            size: chunk_size as u32,
            data: chunk_data,
        });
    }

    Some(DxbcContainer {
        offset_in_file: offset,
        total_size,
        chunks,
    })
}
