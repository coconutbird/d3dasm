//! RDEF-derived naming for constant buffers and resource bindings.
//!
//! The RDEF chunk records the original constant-buffer names, their
//! variable names with byte offsets, and every resource binding's name.
//! Rendering with these names reproduces the original interface instead
//! of register-derived placeholders. Layout stays register-exact: fields
//! pin their registers with `packoffset`, and any constant buffer whose
//! packing is not register-aligned falls back to the anonymous
//! register-array form.

use std::collections::{BTreeMap, BTreeSet};

use dxbc::chunks::rdef::{CBufferDef, ResourceDef, ResourceInputType};

/// One named constant-buffer field pinned to its register range.
#[derive(Debug, Clone)]
pub struct CbField {
    /// Sanitized field name.
    pub name: String,
    /// First register of the field.
    pub register: u32,
    /// Registers spanned (1 renders as `float4 name`, more as an array).
    pub span: u32,
}

/// Naming for one constant-buffer slot.
#[derive(Debug, Clone)]
pub struct CbNames {
    /// Sanitized block name.
    pub block: String,
    /// Register-aligned fields, or `None` to keep the anonymous array.
    pub fields: Option<Vec<CbField>>,
}

/// RDEF-derived names by bind slot.
#[derive(Debug, Clone, Default)]
pub struct ResourceNames {
    /// `b#` slot → constant-buffer naming.
    pub constant_buffers: BTreeMap<u32, CbNames>,
    /// `t#` slot → SRV name (textures, buffers, structured buffers).
    pub textures: BTreeMap<u32, String>,
    /// `s#` slot → sampler name.
    pub samplers: BTreeMap<u32, String>,
    /// `u#` slot → UAV name.
    pub uavs: BTreeMap<u32, String>,
}

impl ResourceNames {
    /// Derives naming from one parsed RDEF chunk.
    #[must_use]
    pub fn from_rdef(rdef: &ResourceDef<'_>) -> Self {
        let mut used = Used::default();
        let mut names = Self::default();
        for binding in &rdef.bindings {
            let slot = binding.bind_point;
            // A multi-register binding range has one name for many
            // registers; the register-derived spelling stays exact.
            if binding.bind_count != 1 {
                continue;
            }
            match ResourceInputType::from_u32(binding.input_type) {
                Some(ResourceInputType::CBuffer) => {
                    let block = used.claim(&binding.name);
                    let fields = rdef
                        .constant_buffers
                        .iter()
                        .find(|buffer| buffer.name == binding.name)
                        .and_then(|buffer| aligned_fields(buffer, &mut used));
                    names
                        .constant_buffers
                        .insert(slot, CbNames { block, fields });
                }
                Some(
                    ResourceInputType::Texture
                    | ResourceInputType::Structured
                    | ResourceInputType::ByteAddress,
                ) => {
                    let name = used.claim(&binding.name);
                    names.textures.insert(slot, name);
                }
                Some(ResourceInputType::Sampler) => {
                    let name = used.claim(&binding.name);
                    names.samplers.insert(slot, name);
                }
                Some(
                    ResourceInputType::UavRwTyped
                    | ResourceInputType::UavRwStructured
                    | ResourceInputType::UavRwByteAddress
                    | ResourceInputType::UavAppendStructured
                    | ResourceInputType::UavConsumeStructured
                    | ResourceInputType::UavRwStructuredWithCounter,
                ) => {
                    let name = used.claim(&binding.name);
                    names.uavs.insert(slot, name);
                }
                _ => {}
            }
        }
        names
    }

    /// Drops every constant buffer's field naming, keeping block and
    /// resource names — the retry form when an access resolves outside
    /// the named fields.
    #[must_use]
    pub fn without_cb_fields(mut self) -> Self {
        for naming in self.constant_buffers.values_mut() {
            naming.fields = None;
        }
        self
    }
}

/// Field naming for one buffer, when every variable is register-aligned
/// and no register is shared between variables.
fn aligned_fields(buffer: &CBufferDef<'_>, used: &mut Used) -> Option<Vec<CbField>> {
    let mut spans: Vec<(u32, u32, &str)> = Vec::new();
    for variable in &buffer.variables {
        if variable.offset % 16 != 0 || variable.size == 0 {
            return None;
        }
        let register = variable.offset / 16;
        let span = variable.size.div_ceil(16);
        spans.push((register, span, variable.name.as_ref()));
    }
    spans.sort_by_key(|&(register, ..)| register);
    for pair in spans.windows(2) {
        if pair[0].0 + pair[0].1 > pair[1].0 {
            return None;
        }
    }
    let fields = spans
        .into_iter()
        .map(|(register, span, name)| CbField {
            name: used.claim(name),
            register,
            span,
        })
        .collect();
    Some(fields)
}

/// The set of claimed identifiers, seeded with everything the renderer
/// generates and the HLSL words a field must not shadow.
#[derive(Debug, Default)]
struct Used {
    names: BTreeSet<String>,
}

impl Used {
    /// Sanitizes `raw` into a fresh HLSL identifier and claims it.
    fn claim(&mut self, raw: &str) -> String {
        let mut name: String = raw
            .chars()
            .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
            .collect();
        if name.is_empty() || name.starts_with(|c: char| c.is_ascii_digit()) {
            name.insert(0, '_');
        }
        if reserved(&name) {
            name.push('_');
        }
        while !self.names.insert(name.clone()) {
            name.push('_');
        }
        name
    }
}

/// Whether an identifier collides with an HLSL word or a spelling the
/// renderer generates itself (registers, temps, resinfo locals).
fn reserved(name: &str) -> bool {
    const WORDS: &[&str] = &[
        "asm",
        "bool",
        "break",
        "case",
        "cbuffer",
        "centroid",
        "column_major",
        "const",
        "continue",
        "default",
        "discard",
        "do",
        "double",
        "else",
        "extern",
        "false",
        "float",
        "for",
        "groupshared",
        "half",
        "icb",
        "if",
        "in",
        "inline",
        "inout",
        "int",
        "line",
        "linear",
        "main",
        "matrix",
        "min16float",
        "min16int",
        "min16uint",
        "nointerpolation",
        "noperspective",
        "oDepth",
        "oMask",
        "out",
        "packoffset",
        "pass",
        "point",
        "precise",
        "register",
        "return",
        "rd",
        "rh",
        "rm",
        "row_major",
        "rw",
        "sample",
        "sampler",
        "shared",
        "snorm",
        "static",
        "string",
        "struct",
        "switch",
        "tbuffer",
        "technique",
        "texture",
        "triangle",
        "true",
        "typedef",
        "uniform",
        "unorm",
        "uint",
        "vector",
        "void",
        "volatile",
        "while",
    ];
    if WORDS.contains(&name) {
        return true;
    }
    // Vector/matrix spellings: float4, int2, uint4x4, bool3, …
    for stem in ["float", "int", "uint", "bool", "half", "double"] {
        if let Some(rest) = name.strip_prefix(stem) {
            let mut chars = rest.chars();
            match (chars.next(), chars.next(), chars.next(), chars.next()) {
                (Some('1'..='4'), None, ..)
                | (Some('1'..='4'), Some('x'), Some('1'..='4'), None) => return true,
                _ => {}
            }
        }
    }
    // Register spellings the renderer emits: r0, v1, o2, x0, t3, s0, u1,
    // cb4, tmp7.
    for stem in ["r", "v", "o", "x", "t", "s", "u", "cb", "tmp", "b"] {
        if let Some(rest) = name.strip_prefix(stem)
            && !rest.is_empty()
            && rest.chars().all(|c| c.is_ascii_digit())
        {
            return true;
        }
    }
    false
}
