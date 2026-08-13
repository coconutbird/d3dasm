//! Lossless `.d3dasm` text format: a round-trippable textual encoding of a
//! `dxbc::shex::Program`.
//!
//! Unlike the human-readable disassembly produced by `dxbc::shex` `Display`
//! (which trims swizzles, prettifies immediates, and drops fields the encoder
//! needs), the `.d3dasm` format is **injective**: [`serialize`] writes every
//! field the `dxbc::shex` encoder consumes, and [`parse`] is its inverse. The
//! design property is:
//!
//! ```text
//! bytes -> decode -> serialize -> parse -> encode -> bytes
//! ```
//!
//! re-produces byte-identical bytecode (bounded only by the existing
//! `decode`/`encode` fidelity).
//!
//! ## Grammar (informal)
//!
//! ```text
//! program      := profile NL (instr NL)*
//! profile      := shadertype '_' major '_' minor          ; e.g. ps_5_0
//! instr        := generic | decl | customdata
//! generic      := mnemonic modifiers? (' ' operand (', ' operand)*)?
//! mnemonic     := <opcode name> | 'op' <value>            ; op<value> = unknown opcode
//! modifiers    := ('_sat' | '_nz' | '_ri' N | '_pm' HEX | '_sf' HEX
//!                 | '_off(' i ',' i ',' i ')'
//!                 | '_res(' dim (',stride=' N)? ')' | '_rd' HEX8     ; resource dim (+ hex fallback)
//!                 | '_rt(' ret ',' ret ',' ret ',' ret ')' | '_rr' HEX8)*  ; return types (+ fallback)
//! operand      := '-'? '|'? core components? '|'? operand_tag*
//! operand_tag  := '@min16f' | '@min2_8f' | '@min16i' | '@min16u'
//!               | '@minprec' digit | '@nonuniform'
//! core         := imm | reg
//! imm          := ('l'|'d') '(' value (', ' value)* ')'   ; value = float | int | 0xhex
//! reg          := (prefix | '?reg(' value ')') index0? ('[' index ']')*
//! reg          := prefix index0? ('[' index ']')*
//! index0       := digits 'L'?                              ; first Imm index, unbracketed
//! index        := digits 'L'? | operand (' + ' digits)?    ; L = Imm64, operand = relative
//! components   := ':' letter* | '.1' | '.' letter | '.' letter{4}
//! ```
//!
//! Component encoding is explicit so each `dxbc::shex::ComponentSelect`
//! variant is distinguishable: `ZeroComponent` -> (none), `OneComponent` -> `.1`,
//! `Scalar(c)` -> `.<letter>`, `Swizzle(s)` -> `.<4 letters>`, `Mask(m)` ->
//! `:<letters>` (the `:` marks a write-mask vs a `.` read-swizzle).

mod parse;
mod serialize;

pub use self::parse::{AsmError, parse};
pub use self::serialize::{operand_string, serialize};

/// Component axis letters, indexed by component number (x=0..w=3).
pub(crate) const AXES: [char; 4] = ['x', 'y', 'z', 'w'];

/// Map a component axis letter to its index (0..=3).
pub(crate) fn axis_index(c: char) -> Option<u8> {
    match c {
        'x' => Some(0),
        'y' => Some(1),
        'z' => Some(2),
        'w' => Some(3),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Shared string<->value tables.
//
// These mirror the canonical literals produced by `decode` so that a parsed
// string is interned to the exact same `&'static str` the decoder would yield,
// keeping round-trips content-equal.
// ---------------------------------------------------------------------------

/// Shader stage names, indexed by `D3D10_SB_TOKENIZED_PROGRAM_TYPE`.
pub(crate) const SHADER_TYPES: &[(&str, u32)] = &[
    ("ps", 0),
    ("vs", 1),
    ("gs", 2),
    ("hs", 3),
    ("ds", 4),
    ("cs", 5),
];

/// Pixel-shader interpolation modes (token0 bits 11-14).
pub(crate) const INTERPOLATIONS: &[(&str, u32)] = &[
    ("undefined", 0),
    ("constant", 1),
    ("linear", 2),
    ("linearCentroid", 3),
    ("linearNoperspective", 4),
    ("linearNoperspectiveCentroid", 5),
    ("linearSample", 6),
    ("linearNoperspectiveSample", 7),
];

/// Resource dimension names (token0 bits 11-15).
pub(crate) const DIMENSIONS: &[(&str, u32)] = &[
    ("buffer", 1),
    ("texture1d", 2),
    ("texture2d", 3),
    ("texture2dms", 4),
    ("texture3d", 5),
    ("texturecube", 6),
    ("texture1darray", 7),
    ("texture2darray", 8),
    ("texture2dmsarray", 9),
    ("texturecubearray", 10),
    ("raw_buffer", 11),
    ("structured_buffer", 12),
];

/// Sampler modes (token0 bits 11-14).
pub(crate) const SAMPLER_MODES: &[(&str, u32)] = &[("default", 0), ("comparison", 1), ("mono", 2)];

/// Constant-buffer access patterns (token0 bit 11).
pub(crate) const CB_ACCESS: &[(&str, u32)] = &[("immediateIndexed", 0), ("dynamicIndexed", 1)];

/// Tessellator domains (token0 bits 11-12).
pub(crate) const TESS_DOMAINS: &[(&str, u32)] =
    &[("undefined", 0), ("isoline", 1), ("tri", 2), ("quad", 3)];

/// Tessellator partitioning modes (token0 bits 11-13).
pub(crate) const TESS_PARTITIONINGS: &[(&str, u32)] = &[
    ("undefined", 0),
    ("integer", 1),
    ("pow2", 2),
    ("fractional_odd", 3),
    ("fractional_even", 4),
];

/// Tessellator output primitive types (token0 bits 11-13).
pub(crate) const TESS_OUTPUT_PRIMS: &[(&str, u32)] = &[
    ("undefined", 0),
    ("point", 1),
    ("line", 2),
    ("triangle_cw", 3),
    ("triangle_ccw", 4),
];

/// `dcl_globalFlags` flag names, indexed by bit position (token0 bits 11+).
pub(crate) const GLOBAL_FLAGS: &[&str] = &[
    "refactoringAllowed",
    "enableDoublePrecisionFloatOps",
    "forceEarlyDepthStencil",
    "enableRawAndStructuredBuffers",
    "skipOptimization",
    "enableMinPrecision",
    "enable11_1DoubleExtensions",
    "enable11_1ShaderExtensions",
];

/// Intern `s` to the canonical `&'static str` in a `(name, value)` table.
pub(crate) fn intern(table: &[(&'static str, u32)], s: &str) -> Option<&'static str> {
    table.iter().find(|(n, _)| *n == s).map(|(n, _)| *n)
}

/// Look up the `&'static str` for `val` in a `(name, value)` table.
pub(crate) fn name_of(table: &[(&'static str, u32)], val: u32) -> Option<&'static str> {
    table.iter().find(|(_, v)| *v == val).map(|(n, _)| *n)
}

/// Look up the `u32` value for `name` in a `(name, value)` table.
pub(crate) fn value_of(table: &[(&'static str, u32)], name: &str) -> Option<u32> {
    table.iter().find(|(n, _)| *n == name).map(|(_, v)| *v)
}
