use std::path::PathBuf;
use std::process;

use clap::Parser;

/// Direct3D shader bytecode disassembler.
///
/// Supports DXBC containers (SM4/SM5) found in raw .bin files
/// and fxb0 container bundles.
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

    let output = d3dasm::disassemble(&data);
    if output.is_empty() {
        eprintln!("No DXBC shader bytecode found in {}", path.display());
        process::exit(1);
    }

    println!("// File: {}", path.display());
    println!("// Found {} DXBC shader(s)", d3dasm::scan(&data).len());
    println!();
    print!("{output}");
}
