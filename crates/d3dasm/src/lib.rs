//! `d3dasm` — a high-level Direct3D shader disassembly interface.
//!
//! This crate provides a convenient API on top of [`dxbc`] for scanning,
//! parsing, and formatting DXBC shader containers. It is the primary
//! interface for tools that need structured access to shader metadata
//! (signatures, resource definitions, statistics) and human-readable
//! disassembly output.
//!
//! # Quick start
//!
//! ```rust,ignore
//! let data = std::fs::read("shader.bin").unwrap();
//! let shaders = d3dasm::parse(&data);
//!
//! for shader in &shaders {
//!     // Structured access to parsed chunks.
//!     // Chunks are parsed lazily — only the shader program is decoded here.
//!     if let Some(program) = shader.program() {
//!         println!("SM {}.{}", program.version_major, program.version_minor);
//!     }
//!
//!     // Full disassembly via Display (parses remaining chunks on demand).
//!     print!("{shader}");
//! }
//! ```

#![no_std]
extern crate alloc;

use alloc::vec::Vec;
use core::cell::OnceCell;
use core::fmt;

use dxbc::chunks::ChunkData;
use dxbc::container::DxbcContainer;

/// Re-export the DXBC backend for callers that need lower-level access.
pub use dxbc;

/// A parsed DXBC shader container with typed chunk access and formatting.
///
/// Created by [`parse`] or [`Shader::from_container`]. Chunks are parsed
/// lazily on first access through the typed accessor methods, so
/// constructing a `Shader` is cheap.
pub struct Shader<'a> {
    /// The underlying DXBC container (chunk table + raw data).
    container: DxbcContainer<'a>,
    /// Lazily parsed chunks, one cell per container chunk index.
    parsed: Vec<OnceCell<ChunkData<'a>>>,
}

impl fmt::Debug for Shader<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Shader")
            .field("container", &self.container)
            .field(
                "parsed_count",
                &self.parsed.iter().filter(|c| c.get().is_some()).count(),
            )
            .field("total_chunks", &self.container.chunks.len())
            .finish()
    }
}

impl<'a> Shader<'a> {
    /// Wrap a [`DxbcContainer`]. No chunks are parsed until accessed.
    pub fn from_container(container: DxbcContainer<'a>) -> Self {
        let parsed = (0..container.chunks.len())
            .map(|_| OnceCell::new())
            .collect();
        Self { container, parsed }
    }

    /// Parse and return the chunk at `index`, caching the result.
    fn chunk_at(&self, index: usize) -> &ChunkData<'a> {
        self.parsed[index].get_or_init(|| self.container.chunks[index].parse())
    }

    /// Find the first chunk whose fourcc matches, parsing it on demand.
    fn find_by_fourcc(&self, fourccs: &[&[u8; 4]]) -> Option<&ChunkData<'a>> {
        for (i, raw) in self.container.chunks.iter().enumerate() {
            if fourccs.iter().any(|f| raw.fourcc == **f) {
                return Some(self.chunk_at(i));
            }
        }
        None
    }

    /// Byte offset of this container in the source file.
    pub fn offset(&self) -> usize {
        self.container.offset_in_file
    }

    /// Total size of the DXBC container in bytes.
    pub fn size(&self) -> u32 {
        self.container.total_size
    }

    /// The decoded shader program (SM4/SM5 instructions), if present.
    pub fn program(&self) -> Option<&dxbc::shex::Program> {
        match self.find_by_fourcc(&[b"SHEX", b"SHDR"])? {
            ChunkData::Shader(p) => Some(p),
            _ => None,
        }
    }

    /// Resource definitions (constant buffers, bindings, creator string).
    pub fn resource_def(&self) -> Option<&dxbc::chunks::ResourceDef<'a>> {
        match self.find_by_fourcc(&[b"RDEF"])? {
            ChunkData::Rdef(rd) => Some(rd),
            _ => None,
        }
    }

    /// Input signature elements.
    pub fn input_signature(&self) -> Option<&dxbc::chunks::Signature<'a>> {
        match self.find_by_fourcc(&[b"ISGN", b"ISG1"])? {
            ChunkData::InputSignature(s) => Some(s),
            _ => None,
        }
    }

    /// Output signature elements.
    pub fn output_signature(&self) -> Option<&dxbc::chunks::Signature<'a>> {
        match self.find_by_fourcc(&[b"OSGN", b"OSG1", b"OSG5"])? {
            ChunkData::OutputSignature(s) => Some(s),
            _ => None,
        }
    }

    /// Patch constant signature elements (hull/domain shaders).
    pub fn patch_constant_signature(&self) -> Option<&dxbc::chunks::Signature<'a>> {
        match self.find_by_fourcc(&[b"PCSG", b"PSG1"])? {
            ChunkData::PatchConstantSignature(s) => Some(s),
            _ => None,
        }
    }

    /// Shader statistics (instruction counts, register usage, etc.).
    pub fn stats(&self) -> Option<&dxbc::chunks::ShaderStats> {
        match self.find_by_fourcc(&[b"STAT"])? {
            ChunkData::Stats(s) => Some(s),
            _ => None,
        }
    }

    /// Shader hash (HASH or XHSH chunk).
    pub fn hash(&self) -> Option<&dxbc::chunks::ShaderHash> {
        match self.find_by_fourcc(&[b"HASH", b"XHSH"])? {
            ChunkData::Hash(h) => Some(h),
            _ => None,
        }
    }

    /// D3D12 root signature (RTS0 chunk).
    pub fn root_signature(&self) -> Option<&dxbc::chunks::RootSignature> {
        match self.find_by_fourcc(&[b"RTS0"])? {
            ChunkData::RootSignature(rs) => Some(rs),
            _ => None,
        }
    }

    /// Shader feature info flags (SFI0 chunk).
    pub fn feature_info(&self) -> Option<&dxbc::chunks::ShaderFeatureInfo> {
        match self.find_by_fourcc(&[b"SFI0"])? {
            ChunkData::FeatureInfo(fi) => Some(fi),
            _ => None,
        }
    }

    /// Debug name (ILDN chunk).
    pub fn debug_name(&self) -> Option<&dxbc::chunks::DebugName<'_>> {
        match self.find_by_fourcc(&[b"ILDN"])? {
            ChunkData::DebugName(dn) => Some(dn),
            _ => None,
        }
    }

    /// DXIL bytecode (SM6.0+, DXIL chunk).
    pub fn dxil(&self) -> Option<&dxbc::chunks::DxilData<'_>> {
        match self.find_by_fourcc(&[b"DXIL"])? {
            ChunkData::Dxil(d) => Some(d),
            _ => None,
        }
    }

    /// Access to the underlying [`DxbcContainer`] for advanced use.
    pub fn container(&self) -> &DxbcContainer<'a> {
        &self.container
    }

    /// Iterate over all parsed chunks, parsing any that haven't been yet.
    pub fn chunks(&self) -> impl Iterator<Item = &ChunkData<'a>> {
        (0..self.container.chunks.len()).map(move |i| self.chunk_at(i))
    }
}

/// Parse all DXBC shaders found in a raw byte buffer.
///
/// Scans for DXBC container headers and parses each one into a
/// [`Shader`]. Returns an empty `Vec` if no containers are found.
///
/// # Example
///
/// ```rust,ignore
/// let data = std::fs::read("shader.bin").unwrap();
/// for shader in d3dasm::parse(&data) {
///     print!("{shader}");
/// }
/// ```
pub fn parse(data: &[u8]) -> Vec<Shader<'_>> {
    dxbc::scan_dxbc(data)
        .into_iter()
        .map(Shader::from_container)
        .collect()
}

impl fmt::Display for Shader<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(rs) = self.root_signature() {
            write!(f, "{rs}")?;
        }

        if let Some(rd) = self.resource_def() {
            fmt_resource_def(f, rd)?;
        }

        if let Some(sig) = self.input_signature() {
            fmt_signature(f, "Input Signature", &sig.elements)?;
        }
        if let Some(sig) = self.output_signature() {
            fmt_signature(f, "Output Signature", &sig.elements)?;
        }
        if let Some(sig) = self.patch_constant_signature() {
            fmt_signature(f, "Patch Constant Signature", &sig.elements)?;
        }

        if let Some(dn) = self.debug_name() {
            write!(f, "{dn}")?;
        }
        if let Some(fi) = self.feature_info() {
            write!(f, "{fi}")?;
            writeln!(f, "//")?;
        }
        if let Some(h) = self.hash() {
            write!(f, "{h}")?;
            writeln!(f, "//")?;
        }
        if let Some(program) = self.program() {
            dxbc::shex::write_program(f, program)?;
        }
        if let Some(stats) = self.stats() {
            write!(f, "{stats}")?;
        }
        if let Some(d) = self.dxil() {
            write!(f, "{d}")?;
        }

        // Remaining chunks: PSV0, RDAT, debug data, private, library.
        for chunk in self.chunks() {
            match chunk {
                ChunkData::PipelineStateValidation(p) => write!(f, "{p}")?,
                ChunkData::RuntimeData(r) => write!(f, "{r}")?,
                ChunkData::DebugData(dd) => write!(f, "{dd}")?,
                ChunkData::PrivateData(pd) => write!(f, "{pd}")?,
                ChunkData::LibraryHeader(lh) => write!(f, "{lh}")?,
                ChunkData::LibraryFunction(lf) => write!(f, "{lf}")?,
                ChunkData::LibraryFunctionSignatures(ls) => write!(f, "{ls}")?,
                _ => {}
            }
        }

        Ok(())
    }
}

/// Format resource definitions as commented disassembly text.
fn fmt_resource_def(f: &mut fmt::Formatter<'_>, rd: &dxbc::chunks::ResourceDef<'_>) -> fmt::Result {
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
            write!(f, "// {:<nw$} ", b.name)?;
            b.fmt_columns(f)?;
            writeln!(f)?;
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
    Ok(())
}

/// Format a signature block as commented disassembly text.
fn fmt_signature(
    f: &mut fmt::Formatter<'_>,
    header: &str,
    elements: &[dxbc::chunks::SignatureElement<'_>],
) -> fmt::Result {
    if elements.is_empty() {
        return Ok(());
    }
    let nw = elements
        .iter()
        .map(|e| e.name_with_index_len())
        .max()
        .unwrap_or(4)
        .max(4);
    writeln!(f, "// {header}:")?;
    for e in elements {
        write!(f, "//   ")?;
        // Pad the name+index to nw columns, then columns.
        let name_len = e.name_with_index_len();
        e.fmt_name_with_index(f)?;
        for _ in name_len..nw {
            f.write_str(" ")?;
        }
        write!(f, " ")?;
        e.fmt_columns(f)?;
        writeln!(f)?;
    }
    writeln!(f, "//")?;
    Ok(())
}
