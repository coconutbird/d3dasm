//! Verify the DXBC checksum implementation against real shaders: recompute the
//! header digest from each container's content and compare to the stored hash.
//!
//! Usage: cargo run --release --example `hash_check` -- shaders/**/*.bin

use std::fmt::Write as _;

fn main() {
    let mut total = 0usize;
    let mut matched = 0usize;

    for arg in std::env::args().skip(1) {
        let Ok(data) = std::fs::read(&arg) else {
            continue;
        };
        for c in dxbc::scan_dxbc(&data) {
            total += 1;
            // Hash covers container bytes from offset 20 to the end.
            let start = c.offset_in_file + 20;
            let end = c.offset_in_file + c.total_size as usize;
            let computed = dxbc::checksum::dxbc_checksum(&data[start..end]);
            if computed == c.header_hash {
                matched += 1;
            } else if total - matched <= 3 {
                eprintln!(
                    "MISMATCH {arg} @0x{:X}\n  stored:   {}\n  computed: {}",
                    c.offset_in_file,
                    hex(&c.header_hash),
                    hex(&computed)
                );
            }
        }
    }

    println!("{matched}/{total} container hashes reproduced");
    if matched != total {
        std::process::exit(1);
    }
}

fn hex(b: &[u8]) -> String {
    let mut encoded = String::with_capacity(b.len() * 2);
    for byte in b {
        let _ = write!(encoded, "{byte:02x}");
    }
    encoded
}
