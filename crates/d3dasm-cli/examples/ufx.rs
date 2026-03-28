//! Example: parse Halo Wars 2 UFXS container files (.ufx).
//!
//! This is a temporary tool for feature development — it will not be committed.
//!
//! Usage: cargo run --example ufx -- test/ambientocclusion.ufx

use std::path::PathBuf;
use std::process;

use dxbc::scan_dxbc;
use nostdio::{ReadLe, Seek, SeekFrom, SliceCursor};

fn main() {
    let path = match std::env::args_os().nth(1) {
        Some(p) => PathBuf::from(p),
        None => {
            eprintln!("Usage: ufx <file.ufx>");
            process::exit(1);
        }
    };

    let data = match std::fs::read(&path) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("Error reading {}: {e}", path.display());
            process::exit(1);
        }
    };

    if data.len() < 4 || &data[0..4] != b"UFXS" {
        eprintln!("Not a UFXS file: {}", path.display());
        process::exit(1);
    }

    let mut hdr = SliceCursor::new(&data);
    hdr.seek(SeekFrom::Start(0x04)).unwrap();
    let version = hdr.read_u32_le().unwrap();
    let hash = hdr.read_u32_le().unwrap();

    println!("// File: {}", path.display());
    println!("// Format: UFXS v{version}, hash=0x{hash:08X}");
    println!();

    // Header layout (version 9):
    //   0x10: RTS0 offset, 0x14: RTS0 size
    //   0x18: VS offset,   0x1C: VS size
    //   0x20: HS offset,   0x2C: HS size (0 if absent)
    //   0x30: DS offset,   0x34: DS size (0 if absent)
    //   0x38: PS offset,   0x3C: PS size
    let slots: &[(&str, usize, usize)] = &[
        ("Root Signature", 0x10, 0x14),
        ("Vertex Shader", 0x18, 0x1C),
        ("Hull Shader", 0x20, 0x2C),
        ("Domain Shader", 0x30, 0x34),
        ("Pixel Shader", 0x38, 0x3C),
    ];

    for &(label, off_pos, size_pos) in slots {
        if size_pos + 4 > data.len() {
            continue;
        }
        hdr.seek(SeekFrom::Start(off_pos as u64)).unwrap();
        let offset = hdr.read_u32_le().unwrap() as usize;
        hdr.seek(SeekFrom::Start(size_pos as u64)).unwrap();
        let size = hdr.read_u32_le().unwrap() as usize;
        if offset == 0 || size == 0 {
            continue;
        }
        let end = offset + size;
        if end > data.len() {
            continue;
        }

        let containers = scan_dxbc(&data[offset..end]);
        for mut c in containers {
            c.offset_in_file += offset;
            let shader = d3dasm::parse_container(&c);
            println!("// ============================================================");
            println!(
                "// {label}: DXBC at 0x{:X}, size={}",
                shader.offset, shader.size
            );
            println!("// ============================================================");
            print!("{shader}");
            println!();
        }
    }
}
