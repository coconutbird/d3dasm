use std::env;
use std::fs;
use std::process;

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: d3dasm <shader.bin> [--raw-dxbc]");
        eprintln!("  Disassembles D3D shader bytecode from .bin files.");
        eprintln!("  Supports both fxb0 container files and raw DXBC blobs.");
        process::exit(1);
    }

    let path = &args[1];
    let data = match fs::read(path) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("Error reading {path}: {e}");
            process::exit(1);
        }
    };

    let containers = d3dasm::scan(&data);
    if containers.is_empty() {
        eprintln!("No DXBC shader bytecode found in {path}");
        process::exit(1);
    }

    println!("// File: {path}");
    println!("// Found {} DXBC shader(s)", containers.len());
    println!();

    for (i, container) in containers.iter().enumerate() {
        println!("// ============================================================");
        println!("// Shader #{i}: {container}");
        println!("// ============================================================");

        for chunk in &container.chunks {
            match chunk.fourcc_str() {
                "RDEF" => print_rdef(chunk.data),
                "ISGN" | "ISG1" => print_signature("Input", chunk.data),
                "OSGN" | "OSG1" | "OSG5" => print_signature("Output", chunk.data),
                "SHEX" | "SHDR" => print_shex(chunk.data),
                "STAT" => print_stat(chunk.data),
                _ => {
                    println!("// Chunk: {} ({} bytes)", chunk.fourcc_str(), chunk.size);
                }
            }
        }
        println!();
    }
}

fn print_rdef(data: &[u8]) {
    if let Some(rd) = dxbc::rdef::parse_rdef(data) {
        if !rd.creator.is_empty() {
            println!("// Compiled with: {}", rd.creator);
        }
        if !rd.bindings.is_empty() {
            println!("//");
            println!("// Resource Bindings:");
            println!("// {:<28} {:<12} {:<8} Slot", "Name", "Type", "Dim");
            println!("// {:-<28} {:-<12} {:-<8} ----", "", "", "");
            for b in &rd.bindings {
                println!("// {b}");
            }
        }
        for cb in &rd.constant_buffers {
            println!("//");
            println!("// cbuffer {} ({} bytes)", cb.name, cb.size);
            for v in &cb.variables {
                println!("//   {:<30} offset={:<4} size={}", v.name, v.offset, v.size);
            }
        }
        println!("//");
    }
}

fn print_signature(label: &str, data: &[u8]) {
    let elements = dxbc::signature::parse_signature(data);
    if !elements.is_empty() {
        println!("// {label} Signature:");
        for e in &elements {
            println!("//   {e}");
        }
        println!("//");
    }
}

fn print_shex(data: &[u8]) {
    let asm = dxbc::shex::disassemble(data);
    print!("{asm}");
}

fn print_stat(data: &[u8]) {
    if let Some(stats) = dxbc::stat::parse_stat(data) {
        print!("{stats}");
    }
}
