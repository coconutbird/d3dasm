//! Debug a single shader's round-trip diff at the dword level.
//!
//! Usage: cargo run --release --example debug_roundtrip -- <file> [container_index]
fn main() {
    let path = std::env::args().nth(1).expect("path");
    let ci: usize = std::env::args()
        .nth(2)
        .unwrap_or("0".into())
        .parse()
        .unwrap();
    let data = std::fs::read(&path).unwrap();
    let containers = dxbc::scan_dxbc(&data);
    let c = &containers[ci];
    for (chi, chunk) in c.chunks.iter().enumerate() {
        let cc = chunk.fourcc_str();
        if cc != "SHEX" && cc != "SHDR" {
            continue;
        }
        let prog = dxbc::shex::decode_with_fourcc(chunk.data, chunk.fourcc).unwrap();
        let encoded = dxbc::shex::encode(&prog);
        if encoded == chunk.data {
            println!("chunk {chi}: OK");
            continue;
        }
        let orig_dwords: Vec<u32> = chunk
            .data
            .chunks(4)
            .map(|c| {
                u32::from_le_bytes([
                    c[0],
                    c.get(1).copied().unwrap_or(0),
                    c.get(2).copied().unwrap_or(0),
                    c.get(3).copied().unwrap_or(0),
                ])
            })
            .collect();
        let enc_dwords: Vec<u32> = encoded
            .chunks(4)
            .map(|c| {
                u32::from_le_bytes([
                    c[0],
                    c.get(1).copied().unwrap_or(0),
                    c.get(2).copied().unwrap_or(0),
                    c.get(3).copied().unwrap_or(0),
                ])
            })
            .collect();
        println!(
            "chunk {chi}: orig={} dwords, enc={} dwords (diff={})",
            orig_dwords.len(),
            enc_dwords.len(),
            orig_dwords.len() as isize - enc_dwords.len() as isize
        );
        let max = std::cmp::max(orig_dwords.len(), enc_dwords.len());
        for i in 0..std::cmp::min(max, 80) {
            let o = orig_dwords.get(i).copied().unwrap_or(0xDEADDEAD);
            let e = enc_dwords.get(i).copied().unwrap_or(0xDEADDEAD);
            let mark = if o != e { " <-- DIFF" } else { "" };
            println!("  [{i:4}] orig={o:08X}  enc={e:08X}{mark}");
        }
        if max > 80 {
            println!("  ... ({} more dwords)", max - 80);
        }
        break; // just first failing chunk
    }
}
