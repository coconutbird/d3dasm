use std::fmt::Write;

/// Re-export the DXBC backend.
pub use dxbc;

/// Scan a byte buffer and extract all shader containers.
///
/// Currently supports DXBC (SM4/SM5). Will support DXIL in the future.
pub fn scan(data: &[u8]) -> Vec<dxbc::container::DxbcContainer<'_>> {
    dxbc::container::scan_dxbc(data)
}

/// Disassemble all shaders found in a raw byte buffer.
///
/// Scans for DXBC containers and formats each one into human-readable
/// disassembly text, including resource definitions, signatures,
/// shader instructions, and statistics.
///
/// Returns an empty string if no DXBC containers are found.
pub fn disassemble(data: &[u8]) -> String {
    let containers = scan(data);
    if containers.is_empty() {
        return String::new();
    }

    let mut out = String::new();
    for (i, container) in containers.iter().enumerate() {
        let _ = writeln!(
            out,
            "// ============================================================"
        );
        let _ = writeln!(out, "// Shader #{i}: {container}");
        let _ = writeln!(
            out,
            "// ============================================================"
        );

        for chunk in &container.chunks {
            match chunk.fourcc_str() {
                "RDEF" => format_rdef(&mut out, chunk.data),
                "ISGN" | "ISG1" => format_signature(&mut out, "Input", chunk.data),
                "OSGN" | "OSG1" | "OSG5" => format_signature(&mut out, "Output", chunk.data),
                "SHEX" | "SHDR" => format_shex(&mut out, chunk.data),
                "STAT" => format_stat(&mut out, chunk.data),
                _ => {
                    let _ = writeln!(
                        out,
                        "// Chunk: {} ({} bytes)",
                        chunk.fourcc_str(),
                        chunk.size
                    );
                }
            }
        }
        let _ = writeln!(out);
    }
    out
}

/// Disassemble a single DXBC container into text.
pub fn disassemble_container(container: &dxbc::container::DxbcContainer<'_>) -> String {
    let mut out = String::new();
    for chunk in &container.chunks {
        match chunk.fourcc_str() {
            "RDEF" => format_rdef(&mut out, chunk.data),
            "ISGN" | "ISG1" => format_signature(&mut out, "Input", chunk.data),
            "OSGN" | "OSG1" | "OSG5" => format_signature(&mut out, "Output", chunk.data),
            "SHEX" | "SHDR" => format_shex(&mut out, chunk.data),
            "STAT" => format_stat(&mut out, chunk.data),
            _ => {
                let _ = writeln!(
                    out,
                    "// Chunk: {} ({} bytes)",
                    chunk.fourcc_str(),
                    chunk.size
                );
            }
        }
    }
    out
}

fn format_rdef(out: &mut String, data: &[u8]) {
    if let Some(rd) = dxbc::rdef::parse_rdef(data) {
        if !rd.creator.is_empty() {
            let _ = writeln!(out, "// Compiled with: {}", rd.creator);
        }
        if !rd.bindings.is_empty() {
            let _ = writeln!(out, "//");
            let _ = writeln!(out, "// Resource Bindings:");
            let _ = writeln!(out, "// {:<28} {:<12} {:<8} Slot", "Name", "Type", "Dim");
            let _ = writeln!(out, "// {:-<28} {:-<12} {:-<8} ----", "", "", "");
            for b in &rd.bindings {
                let _ = writeln!(out, "// {b}");
            }
        }
        for cb in &rd.constant_buffers {
            let _ = writeln!(out, "//");
            let _ = writeln!(out, "// cbuffer {} ({} bytes)", cb.name, cb.size);
            for v in &cb.variables {
                let _ = writeln!(
                    out,
                    "//   {:<30} offset={:<4} size={}",
                    v.name, v.offset, v.size
                );
            }
        }
        let _ = writeln!(out, "//");
    }
}

fn format_signature(out: &mut String, label: &str, data: &[u8]) {
    let elements = dxbc::signature::parse_signature(data);
    if !elements.is_empty() {
        let _ = writeln!(out, "// {label} Signature:");
        for e in &elements {
            let _ = writeln!(out, "//   {e}");
        }
        let _ = writeln!(out, "//");
    }
}

fn format_shex(out: &mut String, data: &[u8]) {
    let asm = dxbc::shex::disassemble(data);
    out.push_str(&asm);
}

fn format_stat(out: &mut String, data: &[u8]) {
    if let Some(stats) = dxbc::stat::parse_stat(data) {
        let _ = write!(out, "{stats}");
    }
}
