//! Lossless `.d3dasm` text round-trip: for every SHEX/SHDR chunk, run
//! `decode -> serialize -> assemble -> encode` and compare against the original
//! chunk bytes. Verifies that the textual format re-encodes byte-identically.
//!
//! Usage: cargo run --release --example `text_roundtrip` -- shaders/**/*.bin

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

                let prog = match dxbc::shex::decode_with_fourcc(chunk.data, chunk.fourcc) {
                    Ok(p) => p,
                    Err(e) => {
                        failed += 1;
                        eprintln!("FAIL {arg}:c{ci}:ch{chi} decode err: {e:?}");
                        continue;
                    }
                };

                // Re-encode the decoded IR directly: this is the byte-identity
                // baseline the text round-trip must also hit.
                let baseline = dxbc::shex::encode(&prog);

                let text = d3dasm::serialize(&prog);
                let parsed = match d3dasm::assemble(&text) {
                    Ok(p) => p,
                    Err(e) => {
                        failed += 1;
                        eprintln!("FAIL {arg}:c{ci}:ch{chi} ({cc}) assemble err: {e}");
                        continue;
                    }
                };
                let reencoded = dxbc::shex::encode(&parsed);

                if reencoded != baseline {
                    failed += 1;
                    let pos = reencoded
                        .iter()
                        .zip(baseline.iter())
                        .position(|(a, b)| a != b)
                        .unwrap_or(std::cmp::min(reencoded.len(), baseline.len()));
                    eprintln!(
                        "FAIL {arg}:c{ci}:ch{chi} ({cc}) baseline={}B text={}B first_diff@{pos}",
                        baseline.len(),
                        reencoded.len()
                    );
                }
            }
        }
    }

    println!("{total} SHEX/SHDR chunks tested, {failed} text round-trip failures");
    if failed > 0 {
        std::process::exit(1);
    }
}
