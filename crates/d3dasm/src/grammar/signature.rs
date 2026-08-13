//! `.d3dasm` text codec for the signature chunks (ISGN/OSGN/PCSG/ISG1/OSG1/
//! OSG5/PSG1).
//!
//! The byte parser ([`dxbc::chunks::signature::parse_signature`]) and the
//! forensic `Display` live in `dxbc`; this is the editable one-line-per-element
//! text form.

use alloc::borrow::Cow;
use alloc::collections::BTreeMap;
use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;
use core::fmt::Write as _;

use dxbc::chunks::signature::{
    ComponentType, MinPrecision, Signature, SignatureElement, SignatureVersion,
};
use dxbc::shex::{system_value_name, system_value_to_u32};

const SIGNATURE_FIELDS: &[&str] = &["idx", "reg", "type", "mask", "rw", "sv", "stream", "prec"];

/// Component type `D3D_REGISTER_COMPONENT_TYPE` → editable name (or number).
fn comp_type_to_str(v: u32) -> String {
    match ComponentType::from_u32(v) {
        Some(ct) => String::from(ct.name()),
        None => format!("{v}"),
    }
}

/// Parse a component type: a known name or a raw number.
fn comp_type_from(s: &str) -> Option<u32> {
    Some(match s {
        "unknown" => 0,
        "uint" => 1,
        "int" => 2,
        "float" => 3,
        "uint16" => 4,
        "int16" => 5,
        "float16" => 6,
        "uint64" => 7,
        "int64" => 8,
        "float64" => 9,
        _ => return s.parse().ok(),
    })
}

/// Component mask → `.xyzw` letters, `none` when empty (or hex for the rare
/// non-standard masks with high bits set).
fn mask_to_str(m: u8) -> String {
    if m == 0 {
        return String::from("none");
    }
    if m & 0xf0 != 0 {
        return format!("{m:02x}");
    }
    let mut s = String::from(".");
    for (bit, ch) in [(1u8, 'x'), (2, 'y'), (4, 'z'), (8, 'w')] {
        if m & bit != 0 {
            s.push(ch);
        }
    }
    s
}

/// Parse a component mask: `none`, `.xyzw` letters, or a hex number.
fn mask_from(s: &str) -> Option<u8> {
    if s == "none" {
        return Some(0);
    }
    let body = s.strip_prefix('.').unwrap_or(s);
    if !body.is_empty() && body.bytes().all(|b| matches!(b, b'x' | b'y' | b'z' | b'w')) {
        let mut m = 0u8;
        for c in body.bytes() {
            m |= match c {
                b'x' => 1,
                b'y' => 2,
                b'z' => 4,
                _ => 8,
            };
        }
        Some(m)
    } else if s.starts_with('.') {
        None // dotted but not valid component letters
    } else {
        u8::from_str_radix(body, 16).ok()
    }
}

/// System value `D3D_NAME` → editable name (or number for unrecognised values).
fn sysval_to_str(v: u32) -> String {
    let n = system_value_name(v);
    if n == "unknown_sv" {
        format!("{v}")
    } else {
        String::from(n)
    }
}

/// Parse a system value: a known name or a raw number. Unknown names error.
fn sysval_from(s: &str) -> Option<u32> {
    if let Ok(n) = s.parse::<u32>() {
        return Some(n);
    }
    let v = system_value_to_u32(s);
    if v == 0 && s != "undefined" {
        None // unrecognised name — surface a loud error rather than silently 0
    } else {
        Some(v)
    }
}

/// Minimum precision → editable name (or number).
fn minprec_to_str(v: u32) -> String {
    String::from(match v {
        1 => "min16f",
        2 => "min2_8f",
        3 => "reserved",
        4 => "min16i",
        5 => "min16u",
        0xf0 => "any16",
        0xf1 => "any10",
        _ => return format!("{v}"),
    })
}

/// Parse a minimum precision: a known name or a raw number.
fn minprec_from(s: &str) -> Option<u32> {
    Some(match s {
        "min16f" => 1,
        "min2_8f" => 2,
        "reserved" => 3,
        "min16i" => 4,
        "min16u" => 5,
        "any16" => 0xf0,
        "any10" => 0xf1,
        _ => return s.parse().ok(),
    })
}

/// Serialize a signature to editable text, one element per line:
/// `<name|-> idx=N reg=N type=<t> mask=<m> rw=<m> [sv=<name>] [stream=N] [prec=<name>]`.
/// Enum fields use names (component type, masks, system value, precision);
/// `sv`/`stream`/`prec` appear only when non-default.
#[must_use]
pub fn signature_to_text(sig: &Signature<'_>) -> String {
    let fourcc_str = core::str::from_utf8(&sig.fourcc).unwrap_or("ISGN");
    let ver = SignatureVersion::from_fourcc(fourcc_str);
    let mut out = String::new();
    for e in &sig.elements {
        let name = if e.semantic_name.is_empty() {
            "-"
        } else {
            &e.semantic_name
        };
        let _ = write!(
            out,
            "{name} idx={} reg={} type={} mask={} rw={}",
            e.semantic_index,
            e.register,
            comp_type_to_str(e.component_type),
            mask_to_str(e.mask),
            mask_to_str(e.rw_mask),
        );
        if e.system_value != 0 {
            let _ = write!(out, " sv={}", sysval_to_str(e.system_value));
        }
        if let (true, Some(s)) = (ver.has_stream(), e.stream.filter(|stream| *stream != 0)) {
            let _ = write!(out, " stream={s}");
        }
        if let (true, Some(mp)) = (
            ver.has_min_precision(),
            e.min_precision.filter(|precision| precision.to_u32() != 0),
        ) {
            let _ = write!(out, " prec={}", minprec_to_str(mp.to_u32()));
        }
        out.push('\n');
    }
    out
}

/// Parse the editable text form produced by [`signature_to_text`]. Accepts both
/// enum names and raw numbers; unrecognised names are an error (returns `None`).
#[must_use]
pub fn signature_from_text(fourcc: [u8; 4], text: &str) -> Option<Signature<'static>> {
    let fourcc_str = core::str::from_utf8(&fourcc).unwrap_or("ISGN");
    let ver = SignatureVersion::from_fourcc(fourcc_str);
    let mut elements = Vec::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let mut f = line.split_whitespace();
        let name = f.next()?;
        let semantic_name = if name == "-" {
            Cow::Owned(String::new())
        } else {
            Cow::Owned(String::from(name))
        };
        let mut m: BTreeMap<&str, &str> = BTreeMap::new();
        for tok in f {
            // Every token after the name must be a known `key=value` tag.
            let (k, v) = tok.split_once('=')?;
            if !SIGNATURE_FIELDS.contains(&k) {
                return None;
            }
            m.insert(k, v);
        }
        let semantic_index = m.get("idx")?.parse().ok()?;
        let register = m.get("reg")?.parse().ok()?;
        let component_type = comp_type_from(m.get("type")?)?;
        let mask = mask_from(m.get("mask")?)?;
        let rw_mask = mask_from(m.get("rw")?)?;
        let system_value = match m.get("sv") {
            Some(s) => sysval_from(s)?,
            None => 0,
        };
        let stream = if ver.has_stream() {
            Some(match m.get("stream") {
                Some(s) => s.parse().ok()?,
                None => 0,
            })
        } else {
            None
        };
        let min_precision = if ver.has_min_precision() {
            Some(MinPrecision::from_u32(match m.get("prec") {
                Some(s) => minprec_from(s)?,
                None => 0,
            }))
        } else {
            None
        };
        elements.push(SignatureElement {
            semantic_name,
            semantic_index,
            system_value,
            component_type,
            register,
            mask,
            rw_mask,
            stream,
            min_precision,
        });
    }
    Some(Signature { fourcc, elements })
}
