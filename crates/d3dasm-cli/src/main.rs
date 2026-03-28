use std::path::{Path, PathBuf};
use std::process;

use clap::Parser;

/// Direct3D shader bytecode disassembler.
///
/// Supports DXBC containers (SM4/SM5) found in raw .bin files.
#[derive(Parser)]
#[command(name = "d3dasm", version, about)]
struct Cli {
    /// Path to the shader binary file.
    file: PathBuf,
}

fn main() {
    let cli = Cli::parse();
    let path = &cli.file;

    let data = match std::fs::read(path) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("Error reading {}: {e}", path.display());
            process::exit(1);
        }
    };

    print_dxbc(path, &data);
}

fn print_dxbc(path: &Path, data: &[u8]) {
    let shaders = d3dasm::parse(data);
    if shaders.is_empty() {
        eprintln!("No DXBC shader bytecode found in {}", path.display());
        process::exit(1);
    }

    println!("// File: {}", path.display());
    println!("// Found {} DXBC shader(s)", shaders.len());
    println!();

    for (i, shader) in shaders.iter().enumerate() {
        println!("// ============================================================");
        println!(
            "// Shader #{i}: DXBC at 0x{:X}, size={}",
            shader.offset(),
            shader.size()
        );
        println!("// ============================================================");
        print!("{shader}");
        println!();
    }
}
