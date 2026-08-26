//! Command-line interface for disassembling DXBC binaries and assembling
//! lossless `.d3dasm` documents.

use std::fmt::Write as _;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::process;

use clap::Parser;

/// Direct3D shader bytecode disassembler and `.d3dasm` assembler.
///
/// Default mode disassembles a DXBC binary to human text. `--emit d3dasm`
/// writes the full forensic `.d3dasm` document (metadata header + every chunk,
/// shader program editable); `--emit hlsl` decompiles each shader program to
/// HLSL source; `--assemble` rebuilds the byte-identical DXBC container(s)
/// from a `.d3dasm` document.
#[derive(Parser)]
#[command(name = "d3dasm", version, about)]
struct Cli {
    /// Input file: a DXBC binary, or a `.d3dasm` text file with `--assemble`.
    file: PathBuf,

    /// Assemble a `.d3dasm` document back into the byte-identical DXBC container(s).
    #[arg(long)]
    assemble: bool,

    /// Output format when disassembling: `human` (default) or `d3dasm` (lossless).
    #[arg(long, value_name = "FORMAT", default_value = "human")]
    emit: String,

    /// Write output to a file instead of stdout.
    #[arg(short, long)]
    output: Option<PathBuf>,
}

fn main() {
    let cli = Cli::parse();

    let result = if cli.assemble {
        assemble(&cli)
    } else {
        disassemble(&cli)
    };

    if let Err(msg) = result {
        eprintln!("{msg}");
        process::exit(1);
    }
}

/// Disassemble a DXBC binary to human text or lossless `.d3dasm`.
fn disassemble(cli: &Cli) -> Result<(), String> {
    let data = std::fs::read(&cli.file)
        .map_err(|e| format!("Error reading {}: {e}", cli.file.display()))?;
    let shaders = d3dasm::parse(&data);
    if shaders.is_empty() {
        return Err(format!(
            "No DXBC shader bytecode found in {}",
            cli.file.display()
        ));
    }

    let mut out = String::new();
    match cli.emit.as_str() {
        "human" => {
            let _ = writeln!(out, "// File: {}", cli.file.display());
            let _ = writeln!(out, "// Found {} DXBC shader(s)\n", shaders.len());
            for (i, shader) in shaders.iter().enumerate() {
                out.push_str("// ============================================================\n");
                let _ = writeln!(
                    out,
                    "// Shader #{i}: DXBC at 0x{:X}, size={}",
                    shader.offset(),
                    shader.size()
                );
                out.push_str("// ============================================================\n");
                let _ = writeln!(out, "{shader}");
            }
        }
        "hlsl" => {
            for (i, shader) in shaders.iter().enumerate() {
                let _ = writeln!(out, "// ===== shader #{i} =====");
                match decompiler::decompile_shader(shader) {
                    Ok(source) => out.push_str(&source),
                    Err(error) => {
                        let _ = writeln!(out, "// not decompiled: {error}");
                    }
                }
                out.push('\n');
            }
        }
        "d3dasm" => {
            // Whole-file forensic document: a container document per DXBC
            // container, plus `.raw` segments for any archive/wrapper bytes
            // around them, so the entire file reassembles byte-identically.
            out.push_str(&d3dasm::container_doc::serialize_file(&data));
        }
        other => {
            return Err(format!(
                "unknown --emit format: {other:?} (use human|d3dasm|hlsl)"
            ));
        }
    }

    write_out(cli.output.as_deref(), out.as_bytes())
}

/// Assemble a `.d3dasm` document back into byte-identical DXBC container(s).
fn assemble(cli: &Cli) -> Result<(), String> {
    let text = std::fs::read_to_string(&cli.file)
        .map_err(|e| format!("Error reading {}: {e}", cli.file.display()))?;
    let bytes = d3dasm::container_doc::assemble_file(&text).map_err(|e| e.to_string())?;
    write_out(cli.output.as_deref(), &bytes)
}

fn write_out(output: Option<&Path>, bytes: &[u8]) -> Result<(), String> {
    match output {
        Some(path) => std::fs::write(path, bytes)
            .map_err(|e| format!("Error writing {}: {e}", path.display())),
        None => std::io::stdout()
            .write_all(bytes)
            .map_err(|e| format!("Error writing stdout: {e}")),
    }
}
