//! Full-container `.d3dasm` document: forensic metadata header + every chunk,
//! such that a container disassembles and reassembles byte-identically.
//!
//! Layout (after the `//` forensic header, which is ignored on parse):
//! ```text
//! .dxbc version=1      ; header hash is recomputed on reassembly, not stored here
//! .code SHEX            ; editable shader program (lossless disassembly body)
//! ps_5_0
//! ...
//! .chunk RDEF           ; every other chunk, raw bytes as hex (32/line)
//!   0a0b0c...
//! .end
//! ```
//!
//! See `docs/d3dasm-grammar.md` for the full text-format grammar.
//! `.code <fourcc>` bodies are assembled via [`crate::assemble`] and
//! re-encoded; `.chunk <fourcc>` bodies are raw hex. The container is rebuilt
//! with [`dxbc::container::build_dxbc_with_header`], preserving the original
//! version and 16-byte header hash.

use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;
use core::fmt::Write;

use crate::AsmError;
use dxbc::chunks::WritableChunk;

use crate::{Shader, forensic};

const HEX_PER_LINE: usize = 32;

/// Serialize a whole DXBC container to a forensic `.d3dasm` document.
#[must_use]
pub fn serialize(shader: &Shader<'_>) -> String {
    let mut out = String::new();
    forensic::write_metadata_header(&mut out, shader);

    let container = shader.container();
    // The header hash is recomputed on reassembly (see `assemble`), so it is
    // not part of the directive — the original is recorded in the comment above.
    out.push_str(".dxbc version=");
    let _ = write!(out, "{}", container.version);
    out.push('\n');

    // Resource definitions, used to symbolicate cbuffer/resource operands in the
    // SHEX body (as `//` comments — stripped on parse, so round-trip is safe).
    let rdef = container
        .chunks
        .iter()
        .find(|c| c.fourcc_str() == "RDEF")
        .and_then(|c| dxbc::chunks::rdef::parse_rdef(c.data));

    for chunk in &container.chunks {
        // A chunk becomes an editable `.code` block only when we have a text
        // codec for it AND that codec is verified to round-trip these exact
        // bytes (see `chunk_to_body`); otherwise it is preserved as raw hex.
        if let Some(mut body) = chunk_to_body(chunk.fourcc, chunk.data) {
            // RDEF carries a `form=` tag naming which editable form was used.
            if &chunk.fourcc == b"RDEF" {
                let _ = writeln!(out, ".code RDEF form={}", rdef_form(&body));
            } else {
                let _ = writeln!(out, ".code {}", chunk.fourcc_str());
            }
            // Annotate the shader program's resource operands with their names.
            if let (true, Some(rd), Ok(program)) = (
                matches!(chunk.fourcc_str(), "SHEX" | "SHDR"),
                &rdef,
                dxbc::shex::decode_with_fourcc(chunk.data, chunk.fourcc),
            ) {
                body = crate::symbolicate::annotate_shex(&body, &program, rd);
            }
            out.push_str(&body);
        } else {
            let _ = writeln!(out, ".chunk {}", chunk.fourcc_str());
            write_hex_block(&mut out, chunk.data);
        }
    }

    out.push_str(".end\n");
    out
}

// ---------------------------------------------------------------------------
// Per-chunk text codecs. A chunk is emitted as an editable `.code` block only
// when `chunk_to_body` confirms its text round-trips to the exact bytes.
// ---------------------------------------------------------------------------

const SIGNATURE_FOURCCS: &[&[u8; 4]] = &[
    b"ISGN", b"ISG1", b"OSGN", b"OSG1", b"OSG5", b"PCSG", b"PSG1",
];

/// Which editable RDEF form a body uses: the HLSL form opens with `target=`,
/// the flat `key=value` form with `version=`.
fn rdef_form(body: &str) -> &'static str {
    let hlsl = body
        .lines()
        .map(str::trim)
        .find(|l| !l.is_empty())
        .is_some_and(|l| l.starts_with("target"));
    if hlsl { "hlsl" } else { "kv" }
}

/// Editable text for a chunk, if a codec exists for its `FourCC`.
fn encode_to_text(fourcc: [u8; 4], data: &[u8]) -> Option<String> {
    if &fourcc == b"SHEX" || &fourcc == b"SHDR" {
        let program = dxbc::shex::decode_with_fourcc(data, fourcc).ok()?;
        return Some(crate::grammar::shex::serialize(&program));
    }
    if SIGNATURE_FOURCCS.contains(&&fourcc) {
        let fs = core::str::from_utf8(&fourcc).ok()?;
        let sig = dxbc::chunks::signature::Signature {
            fourcc,
            elements: dxbc::chunks::signature::parse_signature(fs, data),
        };
        return Some(crate::grammar::signature::signature_to_text(&sig));
    }
    if &fourcc == b"STAT" {
        let stats = dxbc::chunks::stat::parse_stat(data)?;
        return Some(crate::grammar::stat::stat_to_text(&stats));
    }
    if &fourcc == b"RDEF" {
        let rd = dxbc::chunks::rdef::parse_rdef(data)?;
        // Prefer the HLSL reconstruction when it round-trips byte-exactly;
        // otherwise the explicit key=value form (also lossless).
        let hlsl_roundtrip = crate::grammar::rdef::hlsl::rdef_to_hlsl(&rd).and_then(|hlsl| {
            crate::grammar::rdef::hlsl::rdef_from_hlsl(&hlsl).map(|roundtrip| (hlsl, roundtrip))
        });
        if let Some((hlsl, rd2)) = hlsl_roundtrip {
            use dxbc::chunks::ChunkWriter;
            if rd2.to_writable().data == data {
                return Some(hlsl);
            }
        }
        return crate::grammar::rdef::rdef_to_text(&rd);
    }
    None
}

/// Encode an editable chunk body back to raw chunk bytes.
fn body_to_chunk(fourcc: [u8; 4], body: &str) -> Result<Vec<u8>, AsmError> {
    if &fourcc == b"SHEX" || &fourcc == b"SHDR" {
        let program = crate::grammar::shex::parse(body)?;
        return Ok(dxbc::shex::encode(&program));
    }
    if SIGNATURE_FOURCCS.contains(&&fourcc) {
        let sig = crate::grammar::signature::signature_from_text(fourcc, body)
            .ok_or_else(|| err("malformed signature text"))?;
        return Ok(dxbc::chunks::signature::write_signature(fourcc, &sig.elements).data);
    }
    if &fourcc == b"STAT" {
        use dxbc::chunks::ChunkWriter;
        let stats =
            crate::grammar::stat::stat_from_text(body).ok_or_else(|| err("malformed stat text"))?;
        return Ok(stats.to_writable().data);
    }
    if &fourcc == b"RDEF" {
        use dxbc::chunks::ChunkWriter;
        // Auto-detect: the HLSL form opens with `target`, key=value with `version`.
        let is_hlsl = body
            .lines()
            .map(str::trim)
            .find(|l| !l.is_empty())
            .is_some_and(|l| l.starts_with("target"));
        let rd = if is_hlsl {
            crate::grammar::rdef::hlsl::rdef_from_hlsl(body)
                .ok_or_else(|| err("malformed rdef hlsl"))?
        } else {
            crate::grammar::rdef::rdef_from_text(body).ok_or_else(|| err("malformed rdef text"))?
        };
        return Ok(rd.to_writable().data);
    }
    Err(err("no text codec for chunk"))
}

/// Editable body for a chunk, but only when re-encoding it reproduces the
/// original bytes exactly — so an imperfect codec silently falls back to raw
/// hex and byte-identity is never at risk.
fn chunk_to_body(fourcc: [u8; 4], data: &[u8]) -> Option<String> {
    let body = encode_to_text(fourcc, data)?;
    let rebuilt = body_to_chunk(fourcc, &body).ok()?;
    if rebuilt == data { Some(body) } else { None }
}

fn write_hex_block(out: &mut String, data: &[u8]) {
    for row in data.chunks(HEX_PER_LINE) {
        out.push_str("  ");
        for b in row {
            let _ = write!(out, "{b:02x}");
        }
        out.push('\n');
    }
}

/// Assemble a forensic `.d3dasm` document back into byte-identical container
/// bytes. Expects a `.dxbc` directive (produced by [`serialize`]).
///
/// # Errors
///
/// Returns an error when the document contains malformed directives, encoded
/// chunk text, or hexadecimal data.
pub fn assemble(text: &str) -> Result<Vec<u8>, AsmError> {
    // Strip `//` comments and blanks (mirrors shex::assemble).
    let lines: Vec<&str> = text
        .lines()
        .map(|l| l.split("//").next().unwrap_or(l).trim())
        .filter(|l| !l.is_empty())
        .collect();

    let mut idx = 0;
    let header = lines
        .first()
        .ok_or_else(|| err("empty container document"))?;
    let version = parse_dxbc_directive(header)?;
    idx += 1;

    let mut chunks: Vec<WritableChunk> = Vec::new();
    while idx < lines.len() {
        let line = lines[idx];
        if line == ".end" {
            break;
        }
        if let Some(rest) = line.strip_prefix(".code ") {
            // `.code <FOURCC> [tag=value ...]` — only the FourCC is significant;
            // trailing tags (e.g. RDEF's `form=`) are informational.
            let fourcc = parse_fourcc(rest.split_whitespace().next().unwrap_or("").trim())?;
            idx += 1;
            let mut body = String::new();
            while idx < lines.len() && !lines[idx].starts_with('.') {
                body.push_str(lines[idx]);
                body.push('\n');
                idx += 1;
            }
            chunks.push(WritableChunk {
                fourcc,
                data: body_to_chunk(fourcc, &body)?,
            });
        } else if let Some(fourcc) = line.strip_prefix(".chunk ") {
            let fourcc = parse_fourcc(fourcc.trim())?;
            idx += 1;
            let mut hex = String::new();
            while idx < lines.len() && !lines[idx].starts_with('.') {
                hex.push_str(lines[idx]);
                idx += 1;
            }
            chunks.push(WritableChunk {
                fourcc,
                data: hex_decode(&hex)?,
            });
        } else {
            return Err(err(format!("unexpected directive: {line:?}")));
        }
    }

    // Build, then recompute the header checksum over the assembled content so
    // edited shaders carry a valid digest (and unedited ones reproduce the
    // original, keeping byte-identity).
    let mut bytes = dxbc::container::build_dxbc_with_header(&chunks, version, &[0u8; 16]);
    let digest = dxbc::checksum::dxbc_checksum(&bytes[20..]);
    bytes[4..20].copy_from_slice(&digest);
    Ok(bytes)
}

/// Assemble a document containing one *or more* `.dxbc` containers (e.g. an
/// archive `.bin` with many shaders) back into the concatenated container
/// bytes. Each `.dxbc` directive starts a new container.
///
/// # Errors
///
/// Returns an error when no container is present or any contained document is
/// malformed.
pub fn assemble_all(text: &str) -> Result<Vec<u8>, AsmError> {
    let mut out = Vec::new();
    let mut current: Option<String> = None;
    for raw in text.lines() {
        let stripped = raw.split("//").next().unwrap_or(raw).trim();
        if stripped.starts_with(".dxbc") {
            if let Some(doc) = current.take() {
                out.extend_from_slice(&assemble(&doc)?);
            }
            current = Some(String::new());
        }
        if let Some(doc) = current.as_mut() {
            doc.push_str(raw);
            doc.push('\n');
        }
    }
    if let Some(doc) = current.take() {
        out.extend_from_slice(&assemble(&doc)?);
    }
    if out.is_empty() {
        return Err(err("no .dxbc container directive found"));
    }
    Ok(out)
}

/// Serialize a whole input file — which may be a game archive wrapping several
/// DXBC containers with non-container bytes around/between them — preserving
/// **every** byte. Container regions become forensic documents; the wrapper
/// bytes are emitted verbatim as `.raw` hex segments.
#[must_use]
pub fn serialize_file(data: &[u8]) -> String {
    let shaders = crate::parse(data);
    let mut out = String::new();
    let mut pos = 0usize;
    for shader in &shaders {
        let start = shader.offset();
        if start > pos {
            out.push_str(".raw\n");
            write_hex_block(&mut out, &data[pos..start]);
        }
        out.push_str(&serialize(shader));
        pos = start + shader.size() as usize;
    }
    if pos < data.len() {
        out.push_str(".raw\n");
        write_hex_block(&mut out, &data[pos..]);
    }
    out
}

/// Reassemble a whole-file document (see [`serialize_file`]) back into the
/// byte-identical original file, wrapper bytes and all.
///
/// # Errors
///
/// Returns an error when a raw segment or embedded container document is
/// malformed.
pub fn assemble_file(text: &str) -> Result<Vec<u8>, AsmError> {
    let lines: Vec<&str> = text
        .lines()
        .map(|l| l.split("//").next().unwrap_or(l).trim())
        .filter(|l| !l.is_empty())
        .collect();

    let mut out = Vec::new();
    let mut idx = 0;
    while idx < lines.len() {
        let line = lines[idx];
        if line == ".raw" {
            idx += 1;
            let mut hex = String::new();
            while idx < lines.len() && !lines[idx].starts_with('.') {
                hex.push_str(lines[idx]);
                idx += 1;
            }
            out.extend_from_slice(&hex_decode(&hex)?);
        } else if line.starts_with(".dxbc") {
            // Collect this container document up to and including its `.end`.
            let mut doc = String::new();
            doc.push_str(lines[idx]);
            doc.push('\n');
            idx += 1;
            while idx < lines.len() {
                let is_end = lines[idx] == ".end";
                doc.push_str(lines[idx]);
                doc.push('\n');
                idx += 1;
                if is_end {
                    break;
                }
            }
            out.extend_from_slice(&assemble(&doc)?);
        } else {
            return Err(err(format!("unexpected directive: {line:?}")));
        }
    }
    Ok(out)
}

/// True if `text` is a container document (vs. a bare program disassembly).
#[must_use]
pub fn is_container(text: &str) -> bool {
    text.lines()
        .map(|l| l.split("//").next().unwrap_or(l).trim())
        .find(|l| !l.is_empty())
        .is_some_and(|l| l.starts_with(".dxbc"))
}

fn parse_dxbc_directive(line: &str) -> Result<u32, AsmError> {
    // `.dxbc version=N`. The header hash is always recomputed on reassembly,
    // so it is not carried in the directive.
    for tok in line.split_whitespace() {
        if let Some(v) = tok.strip_prefix("version=") {
            return v.parse::<u32>().map_err(|_| err("bad version"));
        }
    }
    Err(err("malformed .dxbc directive (missing version=)"))
}

fn parse_fourcc(s: &str) -> Result<[u8; 4], AsmError> {
    let b = s.as_bytes();
    if b.len() != 4 {
        return Err(err(format!("fourcc must be 4 bytes: {s:?}")));
    }
    Ok([b[0], b[1], b[2], b[3]])
}

fn hex_decode(s: &str) -> Result<Vec<u8>, AsmError> {
    let s = s.trim();
    if s.len() % 2 != 0 {
        return Err(err("hex block has odd length"));
    }
    let bytes = s.as_bytes();
    let mut out = Vec::with_capacity(s.len() / 2);
    let mut i = 0;
    while i < bytes.len() {
        let hi = hex_nibble(bytes[i])?;
        let lo = hex_nibble(bytes[i + 1])?;
        out.push((hi << 4) | lo);
        i += 2;
    }
    Ok(out)
}

fn hex_nibble(b: u8) -> Result<u8, AsmError> {
    match b {
        b'0'..=b'9' => Ok(b - b'0'),
        b'a'..=b'f' => Ok(b - b'a' + 10),
        b'A'..=b'F' => Ok(b - b'A' + 10),
        _ => Err(err("bad hex digit")),
    }
}

fn err(msg: impl Into<String>) -> AsmError {
    AsmError {
        message: msg.into(),
    }
}
