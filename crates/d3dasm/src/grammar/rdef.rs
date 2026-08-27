//! `.d3dasm` text codecs for the RDEF (resource definitions) chunk.
//!
//! Two forms, both lossless:
//!
//! * [`hlsl`] — the preferred editable HLSL reconstruction.
//! * [`rdef_to_text`] / [`rdef_from_text`] — a flat `key=value` fallback for
//!   RDEF shapes the HLSL form can't express.
//!
//! The byte parser ([`dxbc::chunks::rdef::parse_rdef`]) and the IR types live in
//! `dxbc`.

pub mod hlsl;

use alloc::borrow::Cow;
use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::vec::Vec;

use dxbc::chunks::rdef::{
    CBufferDef, CBufferVariable, MemberDesc, Rd11Header, ResourceBinding, ResourceDef, TypeDesc,
};

/// Serialize an RDEF to a flat, editable `key=value` text form. Returns `None`
/// for shapes the codec does not model (e.g. nested struct members).
#[must_use]
pub fn rdef_to_text(rd: &ResourceDef<'_>) -> Option<String> {
    use core::fmt::Write as _;

    fn write_type(o: &mut String, t: &TypeDesc<'_>) {
        let _ = write!(
            o,
            " class={} base={} rows={} cols={} elements={}",
            t.class, t.var_type, t.rows, t.columns, t.elements
        );
        if !t.name.is_empty() {
            let _ = write!(o, " typename={}", t.name);
        }
        if let Some(e) = &t.sm5_extra {
            let _ = write!(o, " sm5={:x},{:x},{:x},{:x}", e[0], e[1], e[2], e[3]);
        }
    }

    // Nested struct members (a member whose own type has members) are not yet
    // supported by the text codec; defer those RDEFs to raw hex.
    for cb in &rd.constant_buffers {
        for v in &cb.variables {
            for m in &v.var_type.members {
                if !m.member_type.members.is_empty() {
                    return None;
                }
            }
        }
    }

    let mut o = String::new();
    let _ = writeln!(o, "version=0x{:08x}", rd.target_version);
    let _ = writeln!(o, "flags=0x{:x}", rd.compile_flags);
    let _ = writeln!(o, "creator={}", rd.creator);
    if let Some(rd11) = rd.rd11 {
        let _ = write!(o, "rd11=");
        for (i, x) in rd11.to_words().iter().enumerate() {
            let _ = write!(o, "{}{x:x}", if i == 0 { "" } else { "," });
        }
        let _ = writeln!(o);
    }
    for b in &rd.bindings {
        let _ = writeln!(
            o,
            "binding {} input={} return={} dim={} samples={} slot={} count={} flags={:x} space={} id={}",
            b.name,
            b.input_type,
            b.return_type,
            b.dimension,
            b.num_samples,
            b.bind_point,
            b.bind_count,
            b.flags,
            b.space,
            b.id
        );
    }
    for cb in &rd.constant_buffers {
        let _ = writeln!(
            o,
            "cbuffer {} size={} flags={:x} kind={}",
            cb.name, cb.size, cb.flags, cb.cb_type
        );
        for v in &cb.variables {
            let _ = write!(
                o,
                "  var {} offset={} size={} flags={:x}",
                v.name, v.offset, v.size, v.flags
            );
            write_type(&mut o, &v.var_type);
            if let Some(ts) = v.texture_start {
                let start = i32::from_ne_bytes(ts.to_ne_bytes());
                let _ = write!(o, " tex={start},{}", v.texture_size.unwrap_or(0));
            }
            if let Some(ss) = v.sampler_start {
                let start = i32::from_ne_bytes(ss.to_ne_bytes());
                let _ = write!(o, " samp={start},{}", v.sampler_size.unwrap_or(0));
            }
            if !v.default_value.is_empty() {
                let _ = write!(o, " default=");
                for byte in v.default_value.iter() {
                    let _ = write!(o, "{byte:02x}");
                }
            }
            let _ = writeln!(o);
            for m in &v.var_type.members {
                let _ = write!(o, "    member {} offset={}", m.name, m.offset);
                write_type(&mut o, &m.member_type);
                let _ = writeln!(o);
            }
        }
    }
    Some(o)
}

fn parse_fields<'a>(tokens: impl Iterator<Item = &'a str>) -> BTreeMap<&'a str, &'a str> {
    let mut fields = BTreeMap::new();
    for token in tokens {
        if let Some((key, value)) = token.split_once('=') {
            fields.insert(key, value);
        }
    }
    fields
}

fn parse_decimal(fields: &BTreeMap<&str, &str>, key: &str) -> Option<u32> {
    fields.get(key)?.parse().ok()
}

fn parse_hex_field(fields: &BTreeMap<&str, &str>, key: &str) -> Option<u32> {
    u32::from_str_radix(fields.get(key)?, 16).ok()
}

fn parse_hex_bytes(hex: &str) -> Option<Vec<u8>> {
    if hex.len() % 2 != 0 {
        return None;
    }
    let bytes = hex.as_bytes();
    let mut decoded = Vec::with_capacity(hex.len() / 2);
    let mut position = 0;
    while position < bytes.len() {
        let high = (bytes[position] as char).to_digit(16)?;
        let low = (bytes[position + 1] as char).to_digit(16)?;
        decoded.push(u8::try_from(high * 16 + low).ok()?);
        position += 2;
    }
    Some(decoded)
}

fn parse_flat_type(fields: &BTreeMap<&str, &str>) -> Option<TypeDesc<'static>> {
    let sm5_extra = match fields.get("sm5") {
        Some(value) => {
            let mut values = value.split(',');
            let mut extra = [0u32; 4];
            for slot in &mut extra {
                *slot = u32::from_str_radix(values.next()?, 16).ok()?;
            }
            Some(extra)
        }
        None => None,
    };
    Some(TypeDesc {
        class: u16::try_from(parse_decimal(fields, "class")?).ok()?,
        var_type: u16::try_from(parse_decimal(fields, "base")?).ok()?,
        rows: u16::try_from(parse_decimal(fields, "rows")?).ok()?,
        columns: u16::try_from(parse_decimal(fields, "cols")?).ok()?,
        elements: u16::try_from(parse_decimal(fields, "elements")?).ok()?,
        members: Vec::new(),
        sm5_extra,
        name: match fields.get("typename") {
            Some(value) => Cow::Owned(String::from(*value)),
            None => Cow::Borrowed(""),
        },
    })
}

fn parse_signed_pair(value: &str) -> Option<(u32, u32)> {
    let (signed, count) = value.split_once(',')?;
    let signed = signed.parse::<i32>().ok()?;
    Some((
        u32::from_ne_bytes(signed.to_ne_bytes()),
        count.parse().ok()?,
    ))
}

fn optional_pair(fields: &BTreeMap<&str, &str>, key: &str) -> Option<(Option<u32>, Option<u32>)> {
    match fields.get(key) {
        Some(value) => {
            let (start, count) = parse_signed_pair(value)?;
            Some((Some(start), Some(count)))
        }
        None => Some((None, None)),
    }
}

fn parse_flat_record(resource_def: &mut ResourceDef<'static>, line: &str) -> Option<()> {
    let (head, rest) = line.split_once(' ').unwrap_or((line, ""));
    let mut tokens = rest.split_whitespace();
    let name = tokens.next()?;
    let fields = parse_fields(tokens);
    match head {
        "binding" => resource_def.bindings.push(ResourceBinding {
            name: Cow::Owned(String::from(name)),
            input_type: parse_decimal(&fields, "input")?,
            return_type: parse_decimal(&fields, "return")?,
            dimension: parse_decimal(&fields, "dim")?,
            num_samples: parse_decimal(&fields, "samples")?,
            bind_point: parse_decimal(&fields, "slot")?,
            bind_count: parse_decimal(&fields, "count")?,
            flags: parse_hex_field(&fields, "flags")?,
            space: fields
                .get("space")
                .map_or(Some(0), |value| value.parse().ok())?,
            id: fields
                .get("id")
                .map_or(Some(0), |value| value.parse().ok())?,
        }),
        "cbuffer" => resource_def.constant_buffers.push(CBufferDef {
            name: Cow::Owned(String::from(name)),
            variables: Vec::new(),
            size: parse_decimal(&fields, "size")?,
            flags: parse_hex_field(&fields, "flags")?,
            cb_type: parse_decimal(&fields, "kind")?,
        }),
        "member" => {
            let member_type = parse_flat_type(&fields)?;
            let offset = parse_decimal(&fields, "offset")?;
            let variable = resource_def
                .constant_buffers
                .last_mut()?
                .variables
                .last_mut()?;
            variable.var_type.members.push(MemberDesc {
                name: Cow::Owned(String::from(name)),
                member_type,
                offset,
            });
        }
        "var" => {
            let (texture_start, texture_size) = optional_pair(&fields, "tex")?;
            let (sampler_start, sampler_size) = optional_pair(&fields, "samp")?;
            let default_value = match fields.get("default") {
                Some(hex) => parse_hex_bytes(hex)?,
                None => Vec::new(),
            };
            let variable = CBufferVariable {
                name: Cow::Owned(String::from(name)),
                offset: parse_decimal(&fields, "offset")?,
                size: parse_decimal(&fields, "size")?,
                flags: parse_hex_field(&fields, "flags")?,
                var_type: parse_flat_type(&fields)?,
                default_value: Cow::Owned(default_value),
                texture_start,
                texture_size,
                sampler_start,
                sampler_size,
            };
            resource_def
                .constant_buffers
                .last_mut()?
                .variables
                .push(variable);
        }
        _ => return None,
    }
    Some(())
}

/// Parse the text produced by [`rdef_to_text`] back into an owned RDEF.
#[must_use]
pub fn rdef_from_text(text: &str) -> Option<ResourceDef<'static>> {
    let mut rd = ResourceDef {
        constant_buffers: Vec::new(),
        bindings: Vec::new(),
        creator: Cow::Owned(String::new()),
        target_version: 0,
        compile_flags: 0,
        rd11: None,
        original_data: None,
        original_canonical: None,
    };

    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        // Header tags (`key=value`).
        let hex0x =
            |s: &str| u32::from_str_radix(s.trim().strip_prefix("0x").unwrap_or(s.trim()), 16);
        if let Some(v) = line.strip_prefix("version=") {
            rd.target_version = hex0x(v).ok()?;
            continue;
        }
        if let Some(v) = line.strip_prefix("flags=") {
            rd.compile_flags = hex0x(v).ok()?;
            continue;
        }
        if let Some(v) = line.strip_prefix("creator=") {
            rd.creator = Cow::Owned(String::from(v));
            continue;
        }
        if let Some(v) = line.strip_prefix("rd11=") {
            let mut a = [0u32; 8];
            let mut it = v.split(',');
            for slot in &mut a {
                *slot = u32::from_str_radix(it.next()?.trim(), 16).ok()?;
            }
            rd.rd11 = Some(Rd11Header::from_words(a));
            continue;
        }

        parse_flat_record(&mut rd, line)?;
    }
    Some(rd)
}
