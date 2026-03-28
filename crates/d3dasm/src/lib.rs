use std::fmt;

/// Re-export the DXBC backend.
pub use dxbc;
use dxbc::chunks::{
    self, ChunkData, DebugData, DebugName, DxilData, LibraryFunction, LibraryFunctionSignatures,
    LibraryHeader, PipelineStateValidation, PrivateData, ResourceDef, RootSignature, RuntimeData,
    ShaderFeatureInfo, ShaderHash, ShaderStats, SignatureElement,
};
use dxbc::container::DxbcContainer;
use dxbc::shex::ir::Program;

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
    /// Patch constant signature elements (hull/domain shaders).
    pub patch_constant_signature: Vec<SignatureElement<'a>>,
    /// Shader statistics (instruction counts, register usage, etc.).
    pub stats: Option<ShaderStats>,
    /// Shader hash (HASH or XHSH chunk).
    pub hash: Option<ShaderHash>,
    /// Root Signature (RTS0 chunk).
    pub root_sig: Option<RootSignature>,
    /// Shader feature info (SFI0 chunk).
    pub feature_info: Option<ShaderFeatureInfo>,
    /// Debug name (ILDN chunk).
    pub debug_name: Option<DebugName>,
    /// Debug data (ILDB chunk).
    pub debug_data: Option<DebugData>,
    /// Private data (PRIV chunk).
    pub private_data: Option<PrivateData>,
    /// DXIL shader bytecode stub (DXIL chunk).
    pub dxil: Option<DxilData>,
    /// Pipeline state validation (PSV0 chunk).
    pub psv: Option<PipelineStateValidation>,
    /// Runtime data (RDAT chunk).
    pub rdat: Option<RuntimeData>,
    /// Library function table (LIBF chunk).
    pub library_functions: Option<LibraryFunction>,
    /// Library function signatures (LFS0 chunk).
    pub library_signatures: Option<LibraryFunctionSignatures>,
    /// Library header (LIBH chunk).
    pub library_header: Option<LibraryHeader>,
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
        patch_constant_signature: Vec::new(),
        stats: None,
        hash: None,
        root_sig: None,
        feature_info: None,
        debug_name: None,
        debug_data: None,
        private_data: None,
        dxil: None,
        psv: None,
        rdat: None,
        library_functions: None,
        library_signatures: None,
        library_header: None,
    };

    for chunk in &container.chunks {
        match chunks::parse_chunk(chunk) {
            ChunkData::RootSignature(rs) => shader.root_sig = Some(rs),
            ChunkData::Rdef(rd) => shader.resource_def = Some(rd),
            ChunkData::InputSignature(s) => shader.input_signature = s,
            ChunkData::OutputSignature(s) => shader.output_signature = s,
            ChunkData::PatchConstantSignature(s) => shader.patch_constant_signature = s,
            ChunkData::Shader(p) => shader.program = Some(p),
            ChunkData::Stats(s) => shader.stats = Some(s),
            ChunkData::Hash(h) => shader.hash = Some(h),
            ChunkData::FeatureInfo(fi) => shader.feature_info = Some(fi),
            ChunkData::DebugName(dn) => shader.debug_name = Some(dn),
            ChunkData::DebugData(dd) => shader.debug_data = Some(dd),
            ChunkData::PrivateData(pd) => shader.private_data = Some(pd),
            ChunkData::Dxil(d) => shader.dxil = Some(d),
            ChunkData::PipelineStateValidation(p) => shader.psv = Some(p),
            ChunkData::RuntimeData(r) => shader.rdat = Some(r),
            ChunkData::LibraryFunction(lf) => shader.library_functions = Some(lf),
            ChunkData::LibraryFunctionSignatures(ls) => shader.library_signatures = Some(ls),
            ChunkData::LibraryHeader(lh) => shader.library_header = Some(lh),
            ChunkData::Unknown { .. } => {}
        }
    }

    shader
}

// ---------------------------------------------------------------------------
// Display — produces the same disassembly text as before
// ---------------------------------------------------------------------------

impl fmt::Display for Shader<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Root Signature
        if let Some(rs) = &self.root_sig {
            write!(f, "{rs}")?;
        }

        // Resource definitions
        if let Some(rd) = &self.resource_def {
            if !rd.creator.is_empty() {
                writeln!(f, "// Compiled with: {}", rd.creator)?;
            }
            if !rd.bindings.is_empty() {
                let nw = rd
                    .bindings
                    .iter()
                    .map(|b| b.name.len())
                    .max()
                    .unwrap_or(4)
                    .max(4);
                writeln!(f, "//")?;
                writeln!(f, "// Resource Bindings:")?;
                writeln!(
                    f,
                    "// {:<nw$} {:<12} {:<8} {:<4} {:<5} Flags",
                    "Name", "Type", "Dim", "Slot", "Count"
                )?;
                writeln!(
                    f,
                    "// {:-<nw$} {:-<12} {:-<8} {:-<4} {:-<5} -----",
                    "", "", "", "", ""
                )?;
                for b in &rd.bindings {
                    writeln!(f, "// {:<nw$} {}", b.name, b.format_columns())?;
                }
            }
            for cb in &rd.constant_buffers {
                let nw = cb
                    .variables
                    .iter()
                    .map(|v| v.name.len())
                    .max()
                    .unwrap_or(4)
                    .max(4);
                writeln!(f, "//")?;
                writeln!(f, "// cbuffer {} ({} bytes)", cb.name, cb.size)?;
                writeln!(f, "//   {:<nw$} {:<6}  Size", "Name", "Offset")?;
                writeln!(f, "//   {:-<nw$} {:-<6}  ----", "", "")?;
                for v in &cb.variables {
                    writeln!(f, "//   {:<nw$} {:<6}  {}", v.name, v.offset, v.size)?;
                }
            }
            writeln!(f, "//")?;
        }

        // Input signature
        if !self.input_signature.is_empty() {
            let nw = self
                .input_signature
                .iter()
                .map(|e| e.name_with_index().len())
                .max()
                .unwrap_or(4)
                .max(4);
            writeln!(f, "// Input Signature:")?;
            for e in &self.input_signature {
                writeln!(
                    f,
                    "//   {:<nw$} {}",
                    e.name_with_index(),
                    e.format_columns()
                )?;
            }
            writeln!(f, "//")?;
        }

        // Output signature
        if !self.output_signature.is_empty() {
            let nw = self
                .output_signature
                .iter()
                .map(|e| e.name_with_index().len())
                .max()
                .unwrap_or(4)
                .max(4);
            writeln!(f, "// Output Signature:")?;
            for e in &self.output_signature {
                writeln!(
                    f,
                    "//   {:<nw$} {}",
                    e.name_with_index(),
                    e.format_columns()
                )?;
            }
            writeln!(f, "//")?;
        }

        // Patch constant signature
        if !self.patch_constant_signature.is_empty() {
            let nw = self
                .patch_constant_signature
                .iter()
                .map(|e| e.name_with_index().len())
                .max()
                .unwrap_or(4)
                .max(4);
            writeln!(f, "// Patch Constant Signature:")?;
            for e in &self.patch_constant_signature {
                writeln!(
                    f,
                    "//   {:<nw$} {}",
                    e.name_with_index(),
                    e.format_columns()
                )?;
            }
            writeln!(f, "//")?;
        }

        // Debug name
        if let Some(dn) = &self.debug_name {
            write!(f, "{dn}")?;
        }

        // Shader feature info
        if let Some(fi) = &self.feature_info {
            write!(f, "{fi}")?;
            writeln!(f, "//")?;
        }

        // Shader hash
        if let Some(h) = &self.hash {
            write!(f, "{h}")?;
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

        // DXIL
        if let Some(d) = &self.dxil {
            write!(f, "{d}")?;
        }

        // PSV0
        if let Some(p) = &self.psv {
            write!(f, "{p}")?;
        }

        // RDAT
        if let Some(r) = &self.rdat {
            write!(f, "{r}")?;
        }

        // Debug data / private data (at the end, less important)
        if let Some(dd) = &self.debug_data {
            write!(f, "{dd}")?;
        }
        if let Some(pd) = &self.private_data {
            write!(f, "{pd}")?;
        }

        // Library chunks
        if let Some(lh) = &self.library_header {
            write!(f, "{lh}")?;
        }
        if let Some(lf) = &self.library_functions {
            write!(f, "{lf}")?;
        }
        if let Some(ls) = &self.library_signatures {
            write!(f, "{ls}")?;
        }

        Ok(())
    }
}
