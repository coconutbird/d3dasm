//! Parse editable HLSL back into an RDEF.

use alloc::borrow::Cow;
use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::vec::Vec;

use super::{
    bind_flags_from, derived_binding_flags, derived_var_size, dim_from_hlsl, hlsl_type_name,
    is_sm5, is_texture, natural_size, packoffset_parse, parse_default, parse_hlsl_type,
    ret_from_hlsl, split_name, type_size,
};
use dxbc::chunks::rdef::{
    CBufferDef, CBufferVariable, MemberDesc, ResourceBinding, ResourceDef, SLOT_UNUSED, SVF_USED,
    TypeDesc,
};

/// Parse a `0x`-prefixed (or bare) hex u32.
fn parse_hex0x(s: &str) -> Option<u32> {
    u32::from_str_radix(s.strip_prefix("0x").unwrap_or(s), 16).ok()
}

/// Collect `key=value` tokens (ignoring bare words) from an iterator.
fn kv<'a>(it: impl Iterator<Item = &'a str>) -> BTreeMap<&'a str, &'a str> {
    let mut m = BTreeMap::new();
    for tok in it {
        if let Some((k, v)) = tok.split_once('=') {
            m.insert(k, v);
        }
    }
    m
}

fn parse_sm5(s: &str) -> Option<[u32; 4]> {
    let mut it = s.split(',');
    let mut a = [0u32; 4];
    for slot in &mut a {
        *slot = u32::from_str_radix(it.next()?, 16).ok()?;
    }
    Some(a)
}

fn hex_bytes(h: &str) -> Option<Vec<u8>> {
    if h.len() % 2 != 0 {
        return None;
    }
    let b = h.as_bytes();
    let mut out = Vec::with_capacity(h.len() / 2);
    let mut i = 0;
    while i < b.len() {
        let hi = (b[i] as char).to_digit(16)?;
        let lo = (b[i + 1] as char).to_digit(16)?;
        out.push(u8::try_from(hi * 16 + lo).ok()?);
        i += 2;
    }
    Some(out)
}

fn parse_signed_u32(value: &str) -> Option<u32> {
    if let Ok(unsigned) = value.parse::<u32>() {
        return Some(unsigned);
    }
    let signed = value.parse::<i32>().ok()?;
    Some(u32::from_ne_bytes(signed.to_ne_bytes()))
}

/// Resolve a type reference token (possibly `row_major`-qualified, with `[N]`)
/// against the known struct table. Returns the reconstructed `TypeDesc`.
fn resolve_type(
    tref: &str,
    row_major: bool,
    structs: &BTreeMap<String, TypeDesc<'static>>,
    sm5: bool,
) -> Option<TypeDesc<'static>> {
    // Array suffix.
    let (core, elements) = match tref.split_once('[') {
        Some((c, rest)) => (c, rest.strip_suffix(']')?.parse().ok()?),
        None => (tref, 0u16),
    };
    if let Some(st) = structs.get(core) {
        let mut td = st.clone();
        td.elements = elements;
        return Some(td);
    }
    let (class, base, rows, cols, _e) = parse_hlsl_type(tref)?;
    let class = if row_major && (class == 3) { 2 } else { class };
    // Canonical element name (no array suffix).
    let core_td = TypeDesc {
        class,
        var_type: base,
        rows,
        columns: cols,
        elements: 0,
        members: Vec::new(),
        sm5_extra: if sm5 { Some([0; 4]) } else { None },
        name: Cow::Borrowed(""),
    };
    let name = hlsl_type_name(&core_td).unwrap_or_default();
    Some(TypeDesc {
        class,
        var_type: base,
        rows,
        columns: cols,
        elements,
        members: Vec::new(),
        sm5_extra: if sm5 { Some([0; 4]) } else { None },
        name: Cow::Owned(name),
    })
}

fn parse_binding(line: &str) -> Option<ResourceBinding<'static>> {
    // `<typespec> <name> : register(<x><slot>)[count] [annotations];`
    let line = line.strip_suffix(';').unwrap_or(line);
    let mut it = line.split_whitespace();
    let spec = it.next()?;
    let name = it.next()?;
    // After the name, find `register(...)`.
    let rest: Vec<&str> = it.collect();
    let reg_tok = rest.iter().find(|t| t.starts_with("register("))?;
    let inner = reg_tok.strip_prefix("register(")?.split(')').next()?;
    let regclass = inner.chars().next()?;
    // slot then optional `[count]` (count may be appended to the register token).
    let after = &reg_tok[reg_tok.find(')')? + 1..];
    let mut bind_count = 1u32;
    if let Some(c) = after.strip_prefix('[').and_then(|s| s.strip_suffix(']')) {
        bind_count = c.parse().ok()?;
    }
    let slot: u32 = inner[1..].parse().ok()?;
    let m = kv(rest.iter().copied());

    // Decode the typespec into input_type / dimension / return_type.
    let (input_type, mut dimension, mut return_type) = decode_typespec(spec, regclass)?;
    let structured = matches!(input_type, 5 | 6 | 9 | 10 | 11);
    if structured {
        dimension = 1; // Buffer
        return_type = 6; // mixed
    }
    if let Some(d) = m.get("dim") {
        dimension = d.parse().ok()?;
    }
    if let Some(r) = m.get("ret") {
        return_type = r.parse().ok()?;
    }
    let num_samples = if structured {
        match m.get("stride") {
            Some(s) => s.parse().ok()?,
            None => 0,
        }
    } else if is_texture(input_type) {
        match m.get("samples") {
            Some(s) => s.parse().ok()?,
            None => 0xFFFF_FFFF, // not multisampled
        }
    } else {
        match m.get("samples") {
            Some(s) => parse_signed_u32(s)?,
            None => 0,
        }
    };
    let flags = match m.get("flags") {
        Some(s) => bind_flags_from(s)?,
        None => derived_binding_flags(input_type, return_type),
    };
    Some(ResourceBinding {
        name: Cow::Owned(String::from(name)),
        input_type,
        return_type,
        dimension,
        num_samples,
        bind_point: slot,
        bind_count,
        flags,
        space: m.get("space").map_or(Some(0), |value| value.parse().ok())?,
        id: m.get("id").map_or(Some(0), |value| value.parse().ok())?,
    })
}

/// Map an HLSL typespec + register class to `(input_type, dimension, return_type)`.
fn decode_typespec(spec: &str, regclass: char) -> Option<(u32, u32, u32)> {
    // Strip any `<...>` element/return wrapper.
    let (stem, inner) = match spec.split_once('<') {
        Some((s, rest)) => (s, Some(rest.strip_suffix('>')?)),
        None => (spec, None),
    };
    let ret = |inner: Option<&str>| inner.and_then(ret_from_hlsl).unwrap_or(0);
    Some(match stem {
        "cbuffer" => (0, 0, 0),
        "tbuffer" => (1, 0, 0),
        "SamplerState" => (3, 0, 0),
        "StructuredBuffer" => (5, 0, 0),
        "RWStructuredBuffer" => (if regclass == 'u' { 6 } else { 5 }, 0, 0),
        "ByteAddressBuffer" => (7, 0, 0),
        "RWByteAddressBuffer" => (8, 0, 0),
        "AppendStructuredBuffer" => (9, 0, 0),
        "ConsumeStructuredBuffer" => (10, 0, 0),
        other => {
            // Texture / RWTexture forms.
            let (prefix_rw, dim_name) = if let Some(d) = other.strip_prefix("RW") {
                (true, d)
            } else {
                (false, other)
            };
            let dim = dim_from_hlsl(dim_name)?;
            let input = if prefix_rw { 4 } else { 2 };
            (input, dim, ret(inner))
        }
    })
}

/// Parse a `key=value`-and-flags variable/member tail into the optional fields.
struct VarTail {
    size: Option<u32>,
    /// `None` means no flag tag was present — derive the used bit on parse.
    flags: Option<u32>,
    sm5: Option<[u32; 4]>,
    tex: Option<(u32, u32)>,
    samp: Option<(u32, u32)>,
    default: Vec<u8>,
    /// HLSL initializer literal after `=`, resolved against the type later.
    default_init: Option<String>,
}

fn parse_var_tail<'a>(toks: impl Iterator<Item = &'a str> + Clone) -> Option<VarTail> {
    let toks: Vec<&str> = toks.collect();
    let m = kv(toks.iter().copied());
    // `used=true|false` (always emitted) plus any other variable-flag bits in
    // `var_flags=`. RDEF now parses without the SHEX program.
    let extra = match m.get("var_flags") {
        Some(v) => u32::from_str_radix(v, 16).ok()?,
        None => 0,
    };
    let flags: Option<u32> = match m.get("used") {
        Some(&"true") => Some(SVF_USED | extra),
        Some(&"false") => Some(extra),
        Some(_) => return None,
        None if extra != 0 => Some(extra),
        None => None,
    };
    let default_init = toks
        .iter()
        .position(|t| *t == "=")
        .and_then(|i| toks.get(i + 1))
        .map(|s| String::from(*s));
    let pair = |s: &str| -> Option<(u32, u32)> {
        let (start, count) = s.split_once(',')?;
        Some((parse_signed_u32(start)?, count.parse().ok()?))
    };
    Some(VarTail {
        size: match m.get("size") {
            Some(s) => Some(s.parse().ok()?),
            None => None,
        },
        flags,
        sm5: match m.get("sm5") {
            Some(s) => Some(parse_sm5(s)?),
            None => None,
        },
        tex: match m.get("tex") {
            Some(s) => Some(pair(s)?),
            None => None,
        },
        samp: match m.get("samp") {
            Some(s) => Some(pair(s)?),
            None => None,
        },
        default: match m.get("default") {
            Some(h) => hex_bytes(h)?,
            None => Vec::new(),
        },
        default_init,
    })
}

struct StructHeader {
    name: String,
    class: u16,
    var_type: u16,
    rows: Option<u16>,
    columns: Option<u16>,
    sm5: Option<[u32; 4]>,
}

enum Block {
    Struct(StructHeader, Vec<MemberDesc<'static>>),
    Cbuffer,
}

struct HlslParser {
    resource_def: ResourceDef<'static>,
    structs: BTreeMap<String, TypeDesc<'static>>,
    block: Option<Block>,
    sm5: bool,
    pending_buffers: Vec<CBufferDef<'static>>,
    buffer_order: Option<Vec<String>>,
    merged_mode: bool,
}

fn parse_struct_member(
    line: &str,
    members: &mut Vec<MemberDesc<'static>>,
    structs: &BTreeMap<String, TypeDesc<'static>>,
    sm5: bool,
) -> Option<()> {
    let body = line.strip_suffix(';').unwrap_or(line);
    let mut tokens = body.split_whitespace();
    let mut type_token = tokens.next()?;
    let row_major = type_token == "row_major";
    if row_major {
        type_token = tokens.next()?;
    }
    let member_token = tokens.next()?;
    let (member_name, array_count) = split_name(member_token)?;
    let rest: Vec<&str> = tokens.collect();
    let offset = match rest.iter().find_map(|token| token.strip_prefix('+')) {
        Some(value) => value.parse().ok()?,
        None => members.last().map_or(0, |member| {
            member.offset + natural_size(&member.member_type).unwrap_or(0)
        }),
    };
    let fields = kv(rest.iter().copied());
    let mut member_type = resolve_type(type_token, row_major, structs, sm5)?;
    if array_count > 0 {
        member_type.elements = array_count;
    }
    if let Some(value) = fields.get("sm5") {
        member_type.sm5_extra = Some(parse_sm5(value)?);
    }
    members.push(MemberDesc {
        name: Cow::Owned(String::from(member_name)),
        member_type,
        offset,
    });
    Some(())
}

fn parse_cbuffer_variable(
    line: &str,
    resource_def: &mut ResourceDef<'static>,
    structs: &BTreeMap<String, TypeDesc<'static>>,
    sm5: bool,
) -> Option<()> {
    let body = line.strip_suffix(';').unwrap_or(line);
    let mut tokens = body.split_whitespace();
    let mut type_token = tokens.next()?;
    let row_major = type_token == "row_major";
    if row_major {
        type_token = tokens.next()?;
    }
    let variable_token = tokens.next()?;
    let (variable_name, array_count) = split_name(variable_token)?;
    tokens.next()?;
    let packoffset_token = tokens.next()?;
    let packoffset = packoffset_token
        .strip_prefix("packoffset(")
        .and_then(|value| value.split(')').next())?;
    let offset = packoffset_parse(packoffset)?;
    let tail = parse_var_tail(tokens)?;
    let mut variable_type = resolve_type(type_token, row_major, structs, sm5)?;
    if array_count > 0 {
        variable_type.elements = array_count;
    }
    if let Some(extra) = tail.sm5 {
        variable_type.sm5_extra = Some(extra);
    }
    let size = match tail.size {
        Some(size) => size,
        None => derived_var_size(&variable_type)?,
    };
    let default_value = match &tail.default_init {
        Some(initializer) => parse_default(&variable_type, initializer)?,
        None => tail.default,
    };
    let buffer = resource_def.constant_buffers.last_mut()?;
    buffer.variables.push(CBufferVariable {
        name: Cow::Owned(String::from(variable_name)),
        offset,
        size,
        flags: tail.flags.unwrap_or(0),
        var_type: variable_type,
        default_value: Cow::Owned(default_value),
        texture_start: tail
            .tex
            .map(|value| value.0)
            .or(if sm5 { Some(SLOT_UNUSED) } else { None }),
        texture_size: tail
            .tex
            .map(|value| value.1)
            .or(if sm5 { Some(0) } else { None }),
        sampler_start: tail.samp.map(|value| value.0).or(if sm5 {
            Some(SLOT_UNUSED)
        } else {
            None
        }),
        sampler_size: tail
            .samp
            .map(|value| value.1)
            .or(if sm5 { Some(0) } else { None }),
    });
    Some(())
}

impl HlslParser {
    fn new(text: &str) -> Self {
        let merged_mode = !text.lines().any(|line| {
            let line = line.trim();
            line.starts_with("cbuffer ") && line.ends_with('{') && !line.contains(": register(")
        });
        Self {
            resource_def: ResourceDef {
                constant_buffers: Vec::new(),
                bindings: Vec::new(),
                creator: Cow::Owned(String::new()),
                target_version: 0,
                compile_flags: 0,
                rd11: None,
                original_data: None,
                original_canonical: None,
            },
            structs: BTreeMap::new(),
            block: None,
            sm5: false,
            pending_buffers: Vec::new(),
            buffer_order: None,
            merged_mode,
        }
    }

    fn parse_line(&mut self, line: &str) -> Option<()> {
        if line == "}" || line == "};" {
            return self.close_block();
        }
        if self.block.is_some() {
            return self.parse_block_line(line);
        }
        if self.parse_metadata(line)? {
            return Some(());
        }
        self.parse_top_level(line)
    }

    fn close_block(&mut self) -> Option<()> {
        match self.block.take()? {
            Block::Struct(header, members) => {
                let struct_size = members
                    .iter()
                    .filter_map(|member| {
                        natural_size(&member.member_type).map(|size| member.offset + size)
                    })
                    .max()
                    .unwrap_or(0);
                let columns = match header.columns {
                    Some(columns) => columns,
                    None => u16::try_from(struct_size / 4).ok()?,
                };
                let descriptor = TypeDesc {
                    class: header.class,
                    var_type: header.var_type,
                    rows: header.rows.unwrap_or(1),
                    columns,
                    elements: 0,
                    members,
                    sm5_extra: header.sm5.or(if self.sm5 { Some([0; 4]) } else { None }),
                    name: Cow::Owned(header.name.clone()),
                };
                self.structs.insert(header.name, descriptor);
            }
            Block::Cbuffer => {}
        }
        Some(())
    }

    fn parse_block_line(&mut self, line: &str) -> Option<()> {
        if let Some(Block::Struct(_, members)) = &mut self.block {
            return parse_struct_member(line, members, &self.structs, self.sm5);
        }
        parse_cbuffer_variable(line, &mut self.resource_def, &self.structs, self.sm5)
    }

    fn parse_metadata(&mut self, line: &str) -> Option<bool> {
        if let Some(value) = line.strip_prefix("target=") {
            self.resource_def.target_version = parse_hex0x(value.trim())?;
            self.sm5 = is_sm5(self.resource_def.target_version);
        } else if let Some(value) = line.strip_prefix("flags=") {
            self.resource_def.compile_flags = parse_hex0x(value.trim())?;
        } else if let Some(value) = line.strip_prefix("creator=") {
            self.resource_def.creator = Cow::Owned(String::from(value));
        } else if let Some(value) = line.strip_prefix("rd11=") {
            let mut extra = [0u32; 8];
            let mut values = value.split(',');
            for slot in &mut extra {
                *slot = u32::from_str_radix(values.next()?.trim(), 16).ok()?;
            }
            self.resource_def.rd11 = Some(dxbc::chunks::rdef::Rd11Header::from_words(extra));
        } else {
            return Some(false);
        }
        Some(true)
    }

    fn parse_top_level(&mut self, line: &str) -> Option<()> {
        let (head, rest) = line.split_once(' ').unwrap_or((line, ""));
        match head {
            "cborder" => {
                self.buffer_order = Some(rest.split_whitespace().map(String::from).collect());
                Some(())
            }
            "struct" => self.open_struct(rest),
            "cbuffer" => self.parse_cbuffer(line, rest),
            _ => self.parse_resource(line),
        }
    }

    fn open_struct(&mut self, rest: &str) -> Option<()> {
        let inner = rest.trim_end_matches('{').trim();
        let mut tokens = inner.split_whitespace();
        let name = tokens.next()?;
        let fields = kv(tokens);
        let rows = match fields.get("rows") {
            Some(value) => Some(value.parse().ok()?),
            None => None,
        };
        let columns = match fields.get("cols") {
            Some(value) => Some(value.parse().ok()?),
            None => None,
        };
        let sm5 = match fields.get("sm5") {
            Some(value) => Some(parse_sm5(value)?),
            None => None,
        };
        self.block = Some(Block::Struct(
            StructHeader {
                name: String::from(name),
                class: fields
                    .get("class")
                    .map_or(Some(5), |value| value.parse().ok())?,
                var_type: fields
                    .get("vtype")
                    .map_or(Some(0), |value| value.parse().ok())?,
                rows,
                columns,
                sm5,
            },
            Vec::new(),
        ));
        Some(())
    }

    fn parse_cbuffer(&mut self, line: &str, rest: &str) -> Option<()> {
        if !line.ends_with('{') {
            self.resource_def.bindings.push(parse_binding(line)?);
            return Some(());
        }
        let inner = rest.trim_end_matches('{').trim();
        let mut tokens = inner.split_whitespace();
        let name = tokens.next()?;
        let tokens: Vec<&str> = tokens.collect();
        let fields = kv(tokens.iter().copied());
        if let Some(register) = tokens.iter().find_map(|token| {
            token
                .strip_prefix("register(")
                .and_then(|value| value.strip_suffix(')'))
        }) {
            let slot = register.get(1..)?.parse().ok()?;
            self.resource_def.bindings.push(ResourceBinding {
                name: Cow::Owned(String::from(name)),
                input_type: 0,
                return_type: 0,
                dimension: 0,
                num_samples: 0,
                bind_point: slot,
                bind_count: 1,
                flags: fields
                    .get("flags")
                    .map_or(Some(0), |value| bind_flags_from(value))?,
                space: fields
                    .get("space")
                    .map_or(Some(0), |value| value.parse().ok())?,
                id: fields
                    .get("id")
                    .map_or(Some(0), |value| value.parse().ok())?,
            });
        }
        self.resource_def.constant_buffers.push(CBufferDef {
            name: Cow::Owned(String::from(name)),
            variables: Vec::new(),
            size: 0,
            flags: fields
                .get("cbflags")
                .map_or(Some(0), |value| u32::from_str_radix(value, 16).ok())?,
            cb_type: fields
                .get("kind")
                .map_or(Some(0), |value| value.parse().ok())?,
        });
        self.block = Some(Block::Cbuffer);
        Some(())
    }

    fn parse_resource(&mut self, line: &str) -> Option<()> {
        self.resource_def.bindings.push(parse_binding(line)?);
        let binding = self.resource_def.bindings.last()?;
        if !self.merged_mode || !matches!(binding.input_type, 5 | 6 | 9 | 10 | 11) {
            return Some(());
        }
        let element = line
            .split_whitespace()
            .next()?
            .split_once('<')
            .and_then(|(_, rest)| rest.strip_suffix('>'))?;
        let element_type = resolve_type(element, false, &self.structs, self.sm5)?;
        let size = type_size(&element_type);
        self.pending_buffers.push(CBufferDef {
            name: binding.name.clone(),
            variables: alloc::vec![CBufferVariable {
                name: Cow::Owned(String::from("$Element")),
                offset: 0,
                size,
                flags: SVF_USED,
                var_type: element_type,
                default_value: Cow::Owned(Vec::new()),
                texture_start: if self.sm5 { Some(SLOT_UNUSED) } else { None },
                texture_size: if self.sm5 { Some(0) } else { None },
                sampler_start: if self.sm5 { Some(SLOT_UNUSED) } else { None },
                sampler_size: if self.sm5 { Some(0) } else { None },
            }],
            size,
            flags: 0,
            cb_type: 3,
        });
        Some(())
    }

    fn finish(mut self) -> ResourceDef<'static> {
        self.resource_def
            .constant_buffers
            .append(&mut self.pending_buffers);
        if let Some(order) = &self.buffer_order {
            self.resource_def.constant_buffers.sort_by_key(|buffer| {
                order
                    .iter()
                    .position(|name| name.as_str() == buffer.name.as_ref())
                    .unwrap_or(usize::MAX)
            });
        }
        update_buffer_sizes(&mut self.resource_def.constant_buffers);
        update_binding_strides(&mut self.resource_def);
        self.resource_def
    }
}

fn update_buffer_sizes(buffers: &mut [CBufferDef<'_>]) {
    for buffer in buffers {
        let end = buffer
            .variables
            .iter()
            .map(|variable| variable.offset + variable.size)
            .max()
            .unwrap_or(0);
        buffer.size = if buffer.cb_type == 0 {
            end.div_ceil(16) * 16
        } else {
            end
        };
    }
}

fn update_binding_strides(resource_def: &mut ResourceDef<'_>) {
    let strides: Vec<(usize, u32)> = resource_def
        .bindings
        .iter()
        .enumerate()
        .filter(|(_, binding)| {
            matches!(binding.input_type, 5 | 6 | 9 | 10 | 11) && binding.num_samples == 0
        })
        .filter_map(|(index, binding)| {
            let buffer = resource_def
                .constant_buffers
                .iter()
                .find(|buffer| buffer.name == binding.name)?;
            let variable = buffer.variables.first()?;
            Some((index, type_size(&variable.var_type)))
        })
        .collect();
    for (index, stride) in strides {
        resource_def.bindings[index].num_samples = stride;
    }
}

/// Parse HLSL text produced by [`super::rdef_to_hlsl`] back into an owned RDEF.
#[must_use]
pub fn rdef_from_hlsl(text: &str) -> Option<ResourceDef<'static>> {
    let mut parser = HlslParser::new(text);
    for raw_line in text.lines() {
        let line = raw_line.trim();
        if !line.is_empty() {
            parser.parse_line(line)?;
        }
    }
    Some(parser.finish())
}
