//! Forensic metadata header: a `//`-commented dump of everything an analyst
//! would want from a DXBC container, emitted ahead of the reconstructable
//! chunk body in a `.d3dasm` document.
//!
//! Every line is guaranteed to start with `//`, so the assembler ignores the
//! whole block — it is informational and is always regenerated from the chunks.

use alloc::format;
use alloc::string::String;
use core::fmt::{self, Write};

use dxbc::chunks::{ResourceDef, SignatureElement};

use crate::Shader;

const BANNER: &str =
    "// ============================================================================\n";

/// Write the full forensic metadata header for `shader` into `out`.
pub fn write_metadata_header(out: &mut String, shader: &Shader<'_>) {
    out.push_str(BANNER);
    out.push_str("// DXBC forensic metadata — informational; ignored on reassembly\n");
    out.push_str(BANNER);

    let container = shader.container();
    let _ = writeln!(
        out,
        "// Container: offset=0x{:X} size={} version={} chunks={}",
        shader.offset(),
        shader.size(),
        container.version,
        container.chunks.len()
    );
    out.push_str("// Header hash: ");
    for b in &container.header_hash {
        let _ = write!(out, "{b:02x}");
    }
    out.push('\n');

    out.push_str("// Chunks:");
    for chunk in &container.chunks {
        let _ = write!(out, " {}({})", chunk.fourcc_str(), chunk.size);
    }
    out.push_str("\n//\n");

    // Decoded chunk metadata. Each block is forced to `//`-prefixed lines.
    if let Some(rd) = shader.resource_def() {
        commented(out, &RdefBlock(rd));
    }
    if let Some(sig) = shader.input_signature() {
        commented(out, &SigBlock("Input Signature", &sig.elements));
    }
    if let Some(sig) = shader.output_signature() {
        commented(out, &SigBlock("Output Signature", &sig.elements));
    }
    if let Some(sig) = shader.patch_constant_signature() {
        commented(out, &SigBlock("Patch Constant Signature", &sig.elements));
    }
    if let Some(stats) = shader.stats() {
        commented(out, stats);
    }
    if let Some(h) = shader.hash() {
        commented(out, h);
    }
    if let Some(fi) = shader.feature_info() {
        commented(out, fi);
    }
    if let Some(dn) = shader.debug_name() {
        commented(out, dn);
    }
    if let Some(rs) = shader.root_signature() {
        out.push_str("// Root Signature:\n");
        commented(out, rs);
    }

    out.push_str(BANNER);
}

/// Append `item`'s `Display` output to `out`, forcing every line to start with
/// `//` (so chunks whose `Display` is not already commented — e.g. the root
/// signature — stay inside the ignored metadata block).
fn commented(out: &mut String, item: &dyn fmt::Display) {
    let rendered = format!("{item}");
    for line in rendered.lines() {
        if line.is_empty() {
            out.push_str("//\n");
        } else if line.starts_with("//") {
            out.push_str(line);
            out.push('\n');
        } else {
            out.push_str("// ");
            out.push_str(line);
            out.push('\n');
        }
    }
}

// Display adapters so the crate-private `fmt_resource_def` / `fmt_signature`
// header writers can be captured into a string.

struct RdefBlock<'a>(&'a ResourceDef<'a>);

impl fmt::Display for RdefBlock<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        crate::fmt_resource_def(f, self.0)
    }
}

struct SigBlock<'a>(&'a str, &'a [SignatureElement<'a>]);

impl fmt::Display for SigBlock<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        crate::fmt_signature(f, self.0, self.1)
    }
}
