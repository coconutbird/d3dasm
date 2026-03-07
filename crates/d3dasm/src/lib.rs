/// Re-export the DXBC backend.
pub use dxbc;

/// Scan a byte buffer and extract all shader containers.
///
/// Currently supports DXBC (SM4/SM5). Will support DXIL in the future.
pub fn scan(data: &[u8]) -> Vec<dxbc::container::DxbcContainer<'_>> {
    dxbc::container::scan_dxbc(data)
}
