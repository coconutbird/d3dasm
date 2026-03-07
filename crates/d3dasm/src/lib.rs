/// Re-export the DXBC backend.
pub use dxbc;

use std::fmt;

/// A parsed shader ready for disassembly.
pub trait Shader: fmt::Debug {
    /// The shader stage (e.g. "vs", "ps", "gs", "cs", "hs", "ds").
    fn stage(&self) -> &str;

    /// Shader model version as (major, minor).
    fn version(&self) -> (u32, u32);

    /// Disassemble the shader bytecode into human-readable text.
    fn disassemble(&self) -> String;
}

/// A container holding one or more shaders extracted from a binary blob.
pub trait ShaderContainer: fmt::Debug + fmt::Display {
    /// The individual shader chunks within this container.
    fn shaders(&self) -> &[Box<dyn Shader>];
}

/// Scan a byte buffer and extract all shader containers.
///
/// Currently supports DXBC. Will support DXIL in the future.
pub fn scan(data: &[u8]) -> Vec<dxbc::container::DxbcContainer<'_>> {
    dxbc::container::scan_dxbc(data)
}
