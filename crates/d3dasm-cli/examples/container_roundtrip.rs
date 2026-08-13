//! Whole-file round-trip gate: serialize each input file to a forensic
//! `.d3dasm` document (containers + `.raw` wrapper segments) and assemble it
//! back, asserting byte-identity with the entire original file.
//!
//! Usage: cargo run --release --example `container_roundtrip` -- shaders/**/*.bin

fn main() {
    let mut files = 0usize;
    let mut containers = 0usize;
    let mut failed = 0usize;

    for arg in std::env::args().skip(1) {
        let data = match std::fs::read(&arg) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("skip {arg}: {e}");
                continue;
            }
        };
        files += 1;
        containers += d3dasm::parse(&data).len();

        let text = d3dasm::container_doc::serialize_file(&data);
        let rebuilt = match d3dasm::container_doc::assemble_file(&text) {
            Ok(b) => b,
            Err(e) => {
                failed += 1;
                eprintln!("FAIL {arg}: assemble error: {e}");
                continue;
            }
        };

        if rebuilt != data {
            failed += 1;
            let pos = rebuilt
                .iter()
                .zip(data.iter())
                .position(|(a, b)| a != b)
                .unwrap_or(data.len().min(rebuilt.len()));
            eprintln!(
                "FAIL {arg}: orig={}B rebuilt={}B first_diff@{pos}",
                data.len(),
                rebuilt.len()
            );
        }
    }

    println!("{files} files ({containers} containers) round-tripped, {failed} failures");
    if failed > 0 {
        std::process::exit(1);
    }
}
