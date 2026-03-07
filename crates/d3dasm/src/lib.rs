use std::fmt;

/// Re-export the DXBC backend.
pub use dxbc;
use dxbc::container::DxbcContainer;
use dxbc::rdef::ResourceDef;
use dxbc::shex::ir::Program;
use dxbc::signature::SignatureElement;
use dxbc::stat::ShaderStats;

/// A fully parsed shader from a DXBC container.
///
/// All chunk data is parsed eagerly. String fields in `resource_def` and
/// signatures borrow directly from the input byte slice (zero-copy).
/// The instruction IR (`program`) is decoded from packed bit-fields and
/// always allocates.
#[derive(Debug)]
pub struct Shader<'a> {
    /// Byte offset of this container in the source file.
    pub offset: usize,
    /// Total size of the DXBC container in bytes.
    pub size: u32,
    /// Decoded shader program (instructions, version, warnings).
    pub program: Option<Program>,
    /// Resource definitions (constant buffers, bindings, creator string).
    pub resource_def: Option<ResourceDef<'a>>,
    /// Input signature elements.
    pub input_signature: Vec<SignatureElement<'a>>,
    /// Output signature elements.
    pub output_signature: Vec<SignatureElement<'a>>,
    /// Shader statistics (instruction counts, register usage, etc.).
    pub stats: Option<ShaderStats>,
}

/// Parse all DXBC shaders found in a raw byte buffer.
///
/// Scans for DXBC containers and parses each one into a structured
/// [`Shader`]. Returns an empty `Vec` if no containers are found.
pub fn parse(data: &[u8]) -> Vec<Shader<'_>> {
    dxbc::container::scan_dxbc(data)
        .iter()
        .map(parse_container)
        .collect()
}

/// Parse a single DXBC container into a [`Shader`].
pub fn parse_container<'a>(container: &DxbcContainer<'a>) -> Shader<'a> {
    let mut shader = Shader {
        offset: container.offset_in_file,
        size: container.total_size,
        program: None,
        resource_def: None,
        input_signature: Vec::new(),
        output_signature: Vec::new(),
        stats: None,
    };

    for chunk in &container.chunks {
        match chunk.fourcc_str() {
            "RDEF" => shader.resource_def = dxbc::rdef::parse_rdef(chunk.data),
            "ISGN" | "ISG1" => {
                shader.input_signature = dxbc::signature::parse_signature(chunk.data);
            }
            "OSGN" | "OSG1" | "OSG5" => {
                shader.output_signature = dxbc::signature::parse_signature(chunk.data);
            }
            "SHEX" | "SHDR" => shader.program = dxbc::shex::decode::decode(chunk.data),
            "STAT" => shader.stats = dxbc::stat::parse_stat(chunk.data),
            _ => {}
        }
    }

    shader
}

// ---------------------------------------------------------------------------
// Display — produces the same disassembly text as before
// ---------------------------------------------------------------------------

impl fmt::Display for Shader<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Resource definitions
        if let Some(rd) = &self.resource_def {
            if !rd.creator.is_empty() {
                writeln!(f, "// Compiled with: {}", rd.creator)?;
            }
            if !rd.bindings.is_empty() {
                writeln!(f, "//")?;
                writeln!(f, "// Resource Bindings:")?;
                writeln!(f, "// {:<28} {:<12} {:<8} Slot", "Name", "Type", "Dim")?;
                writeln!(f, "// {:-<28} {:-<12} {:-<8} ----", "", "", "")?;
                for b in &rd.bindings {
                    writeln!(f, "// {b}")?;
                }
            }
            for cb in &rd.constant_buffers {
                writeln!(f, "//")?;
                writeln!(f, "// cbuffer {} ({} bytes)", cb.name, cb.size)?;
                for v in &cb.variables {
                    writeln!(
                        f,
                        "//   {:<30} offset={:<4} size={}",
                        v.name, v.offset, v.size
                    )?;
                }
            }
            writeln!(f, "//")?;
        }

        // Input signature
        if !self.input_signature.is_empty() {
            writeln!(f, "// Input Signature:")?;
            for e in &self.input_signature {
                writeln!(f, "//   {e}")?;
            }
            writeln!(f, "//")?;
        }

        // Output signature
        if !self.output_signature.is_empty() {
            writeln!(f, "// Output Signature:")?;
            for e in &self.output_signature {
                writeln!(f, "//   {e}")?;
            }
            writeln!(f, "//")?;
        }

        // Shader instructions
        if let Some(program) = &self.program {
            write!(f, "{}", dxbc::shex::fmt::format_program(program))?;
        }

        // Statistics
        if let Some(stats) = &self.stats {
            write!(f, "{stats}")?;
        }

        Ok(())
    }
}
