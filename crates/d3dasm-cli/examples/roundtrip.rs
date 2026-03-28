//! Round-trip test: decode every SHEX/SHDR chunk and re-encode, compare bytes.
//!
//! Usage: cargo run --release --example roundtrip -- shaders/shaders-pc/**/*.bin

fn main() {
    let mut total = 0usize;
    let mut failed = 0usize;

    for arg in std::env::args().skip(1) {
        let data = match std::fs::read(&arg) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("skip {arg}: {e}");
                continue;
            }
        };
        let containers = dxbc::scan_dxbc(&data);
        for (ci, c) in containers.iter().enumerate() {
            for (chi, chunk) in c.chunks.iter().enumerate() {
                let cc = chunk.fourcc_str();
                if cc != "SHEX" && cc != "SHDR" {
                    continue;
                }
                total += 1;
                match dxbc::shex::decode_with_fourcc(chunk.data, chunk.fourcc) {
                    Ok(prog) => {
                        let encoded = dxbc::shex::encode(&prog);
                        if encoded != chunk.data {
                            failed += 1;
                            let pos = encoded
                                .iter()
                                .zip(chunk.data.iter())
                                .position(|(a, b)| a != b)
                                .unwrap_or(std::cmp::min(encoded.len(), chunk.data.len()));
                            eprintln!(
                                "FAIL {arg}:c{ci}:ch{chi} ({cc}) orig={}B enc={}B first_diff@{pos}",
                                chunk.data.len(),
                                encoded.len()
                            );
                        }
                    }
                    Err(e) => {
                        failed += 1;
                        eprintln!("FAIL {arg}:c{ci}:ch{chi} decode err: {e:?}");
                    }
                }
            }
        }
    }

    println!("{total} SHEX/SHDR chunks tested, {failed} round-trip failures");
    if failed > 0 {
        std::process::exit(1);
    }
}
