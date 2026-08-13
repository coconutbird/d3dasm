//! Check typed chunk parsers and writers against one or more DXBC files.
//!
//! Usage: `cargo run --release --example chunk_conformance -- <files...>`

use std::collections::BTreeMap;

#[derive(Default)]
struct Counts {
    parsed: usize,
    unknown: usize,
    mismatched: usize,
}

fn main() {
    let mut counts: BTreeMap<String, Counts> = BTreeMap::new();
    let mut failed = false;

    for path in std::env::args().skip(1) {
        let data = match std::fs::read(&path) {
            Ok(data) => data,
            Err(error) => {
                eprintln!("skip {path}: {error}");
                failed = true;
                continue;
            }
        };
        for (container_index, container) in dxbc::scan_dxbc(&data).iter().enumerate() {
            for (chunk_index, chunk) in container.chunks.iter().enumerate() {
                let fourcc = String::from_utf8_lossy(&chunk.fourcc).into_owned();
                let count = counts.entry(fourcc.clone()).or_default();
                let parsed = chunk.parse();
                if matches!(parsed, dxbc::ChunkData::Unknown { .. }) {
                    count.unknown += 1;
                    if matches!(&chunk.fourcc, b"SHEX" | b"SHDR") {
                        match dxbc::shex::decode_with_fourcc(chunk.data, chunk.fourcc) {
                            Ok(_) => eprintln!(
                                "UNKNOWN {path}:c{container_index}:ch{chunk_index} ({fourcc})"
                            ),
                            Err(error) => eprintln!(
                                "REJECTED {path}:c{container_index}:ch{chunk_index} ({fourcc}): {error}"
                            ),
                        }
                    } else {
                        eprintln!(
                            "UNKNOWN {path}:c{container_index}:ch{chunk_index} ({fourcc}, {} bytes)",
                            chunk.data.len()
                        );
                    }
                    continue;
                }
                count.parsed += 1;
                let Some(encoded) = parsed.to_writable() else {
                    continue;
                };
                if encoded.fourcc != chunk.fourcc || encoded.data != chunk.data {
                    count.mismatched += 1;
                    failed = true;
                    let difference = encoded
                        .data
                        .iter()
                        .zip(chunk.data)
                        .position(|(left, right)| left != right)
                        .unwrap_or(encoded.data.len().min(chunk.data.len()));
                    eprintln!(
                        "MISMATCH {path}:c{container_index}:ch{chunk_index} ({fourcc}) original={} encoded={} first_diff={difference}",
                        chunk.data.len(),
                        encoded.data.len()
                    );
                    let start = difference.saturating_sub(difference % 4);
                    let end = (start + 24)
                        .min(chunk.data.len())
                        .min(encoded.data.len());
                    eprintln!(
                        "  original[{start}..{end}]={:02x?}\n  encoded [{start}..{end}]={:02x?}",
                        &chunk.data[start..end],
                        &encoded.data[start..end]
                    );
                }
            }
        }
    }

    for (fourcc, count) in counts {
        println!(
            "{fourcc}: parsed={} unknown={} mismatched={}",
            count.parsed, count.unknown, count.mismatched
        );
    }
    if failed {
        std::process::exit(1);
    }
}
