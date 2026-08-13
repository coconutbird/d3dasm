# `.d3dasm` Grammar Specification

A `.d3dasm` file is a **lossless, round-trippable text rendering of DXBC shader
bytecode** (the binary format `fxc` emits and Direct3D 10–12 consumes). It is
produced by this project's disassembler and is designed so that
`assemble(serialize(bytes)) == bytes`, byte-for-byte.

This document is the **specification for that text format**, reverse-engineered
from the serializer/parser that currently define it. It exists so that a
standalone parser (in any language) can be written against a stable grammar
rather than against the Rust source. Where the current implementation is
internally inconsistent, this document says so explicitly (see
[§12 Inconsistencies](#12-inconsistencies--ambiguities)) and proposes the
intended, consistent rule.

> **Companion:** [`d3dasm-desired-grammar.md`](d3dasm-desired-grammar.md) takes
> the §12 inconsistencies and commits to a clean, consistent **target** grammar
> (a future `v2`). This document describes what *is*; that one describes what
> *should be*.

> **Status.** Descriptive of the implementation as of this writing. The
> *Recommended grammar* notes in §12 are forward-looking; the *Observed* rules
> are what the current code does. A future parser should follow **Observed**
> for compatibility and may additionally accept the **Recommended** relaxations.
>
> **Note (migrated to the new grammar).** The implementation has been migrated
> to the consistent grammar described in
> [`d3dasm-desired-grammar.md`](d3dasm-desired-grammar.md): uniform `key=value`
> tags everywhere (`profile=`, `access=`, `dim=`/`rt=`, `samples=`, …),
> whitespace-separated operands with `,`-only list values, `.xyzw`/`none`
> component masks, a self-contained RDEF (`used=true|false` always emitted, no
> dependence on the SHEX program), `key=value` RDEF headers (`target=`/
> `version=`), and a `.code RDEF form=hlsl|kv` marker. The **worked example in
> §10 is current**; many of the per-section *Observed* forms below predate the
> migration — treat the desired-grammar doc as the authoritative spec and §10 as
> the ground truth, pending a full rewrite of §§4–9 here.

## Source of truth

The format is defined by these modules (the de-facto spec):

| Layer | File |
| --- | --- |
| Container document, whole-file archive, hex | `crates/d3dasm/src/container_doc.rs` |
| Forensic metadata header (informational) | `crates/d3dasm/src/forensic.rs` |
| SHEX instruction body | `crates/dxbc/src/shex/asm/{serialize,parse,mod}.rs`, `shex/ir.rs` |
| RDEF (resources) as editable HLSL | `crates/dxbc/src/chunks/rdef_hlsl/{mod,encode,decode}.rs` |
| RDEF `key=value` fallback | `crates/dxbc/src/chunks/rdef.rs` |
| Signatures (ISGN/OSGN/…) | `crates/dxbc/src/chunks/signature.rs` |
| Statistics (STAT) | `crates/dxbc/src/chunks/stat.rs` |

---

## 1. Notation

Grammar is given in EBNF:

```
rule      = production ;          a named rule
A B       = A then B              concatenation
A | B     = A or B                alternation
[ A ]     = optional
{ A }     = zero or more
( A )     = grouping
"lit"     = literal text
SP        = a single ASCII space (0x20)
NL        = end of line (the line framing; see §2)
INT       = decimal integer        e.g. 0, 19, 120
HEX       = lowercase hex digits    e.g. ffff0500, 3f80
IDENT     = a name token (see §2)
```

Terminals quoted `"like_this"` are matched literally and are **case-sensitive**.
Separators are significant — `", "` (comma-space) is a different terminal than
`","` or `" "`. This matters; see [§12.1](#121-separator-zoo).

---

## 2. Lexical conventions

These rules apply across **every** layer and are the most important thing to get
right.

### 2.1 Line framing

The format is **line-oriented**. Every consumer performs the same preprocessing
on each physical line, in order (`container_doc.rs:199-203`,
`shex/asm/parse.rs:43-46`):

1. **Comment strip:** truncate the line at the first occurrence of `//`
   (`line.split("//").next()`). `//` is **not escapable** and applies *anywhere*
   in the line, including inside what would otherwise be a data token.
2. **Trim:** strip leading and trailing ASCII whitespace.
3. **Drop blanks:** a line that is empty after steps 1–2 is removed entirely.

Consequences a parser must honor:

- Indentation is **purely cosmetic** — it is removed before parsing. The 4-space
  indents in struct/cbuffer bodies and the 2-space indents in hex blocks carry
  no meaning.
- The serializer uses trailing `//` comments to **symbolicate** SHEX lines —
  resolving cbuffer/texture/sampler/UAV register operands to their RDEF source
  names, e.g. `mad r0:y r0.y r0.z cb0[21].w  // cb0[21].w = gDirLightColor.w`.
  These are an analysis aid only; being comments, they are dropped on parse and
  the faithful operand round-trips unchanged.
- A `//` comment runs to end-of-line and can appear on its own line or trailing
  a content line.
- Blank lines are insignificant separators only.

### 2.2 Whitespace *within* a line is significant

After trimming the ends, interior whitespace is **strict** and varies by
construct. The tokenizers split on single spaces and many require an *exact*
separator:

- `", "` (comma + one space) separates SHEX source operands and immediate values
  (`parse.rs` `eat_str(", ")`).
- `" + "` (space-plus-space) separates a relative index from its constant
  (`cb0[r0.y + 11]`).
- `" "` (single space) separates signature/STAT tokens and SHEX declaration
  operands.
- `","` (bare comma, **no** space) separates the elements of return-type lists
  `(float,float,float,float)`, texel offsets `_off(0,0,0)`, and HLSL default
  initializers `float4(0,0,4,0.25)`.
- Tabs are **not** accepted where a space is expected (`skip_spaces` eats `0x20`
  only).

This separator inconsistency is the single biggest hazard for a hand-written
parser; it is catalogued in [§12.1](#121-separator-zoo).

### 2.3 Numbers

```
INT   = [ "-" ] digit { digit }
HEX   = hexdigit { hexdigit }            lowercase on output; both cases accepted on input
HEX0X = [ "0x" ] HEX                     "0x" optional on input, present on output for SHEX flags
```

- Hex is emitted lowercase. Parsers accept `0-9a-fA-F`.
- Float literals are emitted in Rust's shortest round-tripping debug form, which
  **always** carries a `.`, `e`, `inf`, or `nan` so they can never be mistaken
  for integers (e.g. `1.0`, `0.05`, `0.00048828125`, `inf`). A value whose
  pretty form would not re-parse to identical bits falls back to raw hex bits
  (`0xXXXXXXXX`). See [§9.5](#95-immediates).

### 2.4 Identifiers

`IDENT` is a maximal run of non-space characters used for names (semantic names,
resource/variable names, struct names, FourCCs). Names are emitted verbatim.
Because of §2.1 a name must not contain `//`, and because of the body-delimiter
rule ([§3.3](#33-section-bodies)) a *line* must not begin with `.`.

---

## 3. Document grammar (container layer)

### 3.1 Top level

A `.d3dasm` file is either a **single container document** or a **whole-file
archive** that wraps one or more containers plus raw filler bytes.

```
file        = { raw_segment | container } ;        whole-file form
container   = metadata_header dxbc_line { section } end_line ;
```

`is_container(text)` is true iff the first non-comment line begins with `.dxbc`
(`container_doc.rs:358`).

### 3.2 Directives

```
dxbc_line   = ".dxbc" SP "version=" INT
              [ SP "hash=" HEX32 SP "checksum=" HEX32 ] NL
end_line    = ".end" NL
raw_segment = ".raw" NL hex_block
section     = code_section | chunk_section
code_section  = ".code"  SP FOURCC NL body
chunk_section = ".chunk" SP FOURCC NL hex_block
FOURCC      = <exactly 4 bytes>                    e.g. SHEX, RDEF, ISGN, STAT
```

Rules:

- **`.dxbc`** must be the first non-comment line of a container. `hash=` is the
  exact 16-byte digest stored in the original header; `checksum=` is the
  computed digest of the original bytes from offset 20 onward. If the content
  is unchanged, the assembler restores `hash=` byte-for-byte (including a
  deliberately stale digest). If it changed, the assembler writes a newly
  computed digest. Omitting both tags retains the legacy recompute behavior.
  Unknown or duplicate tokens are errors. See [§12.6](#126-header-hash-preservation).
- **`.code <FOURCC>`** introduces an *editable* chunk body whose grammar is
  selected by the FourCC ([§5](#5-rdef-resource-definitions)–[§9](#9-shex-shader-program)).
- **`.chunk <FOURCC>`** introduces a chunk preserved as **raw hex**
  ([§4](#4-raw--hex-encoding)).
- **`.raw`** (whole-file form only) introduces raw filler bytes between/around
  containers, as hex.
- **`.end`** terminates a container. It is always emitted; on parse the
  container also ends at EOF, but in whole-file mode a container is collected
  *up to and including* its `.end`, so omitting it there is unsafe.
- Any other directive line is an error (`unexpected directive`).

Which of `.code`/`.chunk` a serializer chooses is **not fixed per FourCC**: a
chunk is emitted as `.code` only when its editable text re-encodes to the exact
original bytes; otherwise it falls back to `.chunk` hex
(`container_doc.rs:157-161`). A parser must therefore accept **either** form for
any codec-capable FourCC.

### 3.3 Section bodies

A section body is every following line **up to (not including) the next line
that begins with `.`** (`container_doc.rs:226`, `239`). There is no explicit
"end of body" token; the next directive delimits it. Because lines are trimmed
first (§2.1), a body line that begins with `.` would prematurely end the
section — body content must not start a line with `.`.

### 3.4 Ordering

- `.dxbc` first.
- The SHEX/SHDR program is logically assembled **first regardless of its
  position**, because RDEF reconstruction depends on it (to derive cbuffer
  `used` flags). Chunks are otherwise emitted and rebuilt in document order.

### 3.5 FourCCs with editable codecs

| FourCC(s) | Body grammar |
| --- | --- |
| `SHEX`, `SHDR` | SHEX program ([§9](#9-shex-shader-program)) |
| `ISGN ISG1 OSGN OSG1 OSG5 PCSG PSG1` | Signature ([§7](#7-signatures)) |
| `STAT` | Statistics ([§8](#8-statistics-stat)) |
| `RDEF` | RDEF, HLSL form or `key=value` form ([§5](#5-rdef-resource-definitions), [§6](#6-rdef-keyvalue-fallback)) |
| anything else | raw hex (`.chunk`) |

---

## 4. Raw / hex encoding

Used by `.chunk` bodies and `.raw` segments.

```
hex_block = { hex_line }
hex_line  = "  " { byte } NL              two-space indent on output
byte      = hexdigit hexdigit             lowercase on output
```

- Output: 32 bytes per line, lowercase, **no separator between bytes**,
  two-space indent.
- Input: all body lines are concatenated and the indent/grouping discarded; the
  total nibble count must be even. Both hex cases accepted. Any non-hex
  character is an error.

---

## 5. RDEF (resource definitions)

The RDEF body is rendered as **editable HLSL** with inline annotations for
anything HLSL can't express. The encoder prefers this form and only keeps it if
it round-trips byte-exactly; otherwise it falls back to the `key=value` form
([§6](#6-rdef-keyvalue-fallback)). The container layer auto-detects which is
present: **HLSL if the first non-blank line starts with `target`, key=value if
it starts with `version`** (`container_doc.rs:138-142`).

### 5.1 Header

```
rdef_hlsl   = header { struct_def } { binding } { cbuffer_block } ;
header      = "target" SP HEX           NL    ; 8-digit, e.g. "target ffff0500"
              "flags"  SP HEX           NL    ; e.g. "flags 100"
              "creator" SP rest_of_line NL    ; verbatim, may contain spaces
              [ "rd11" 8*( SP HEX ) NL ]      ; SM5 sub-header, 8 u32
              [ "cborder" { SP IDENT } NL ]   ; explicit cbuffer order (see §5.4)
```

- `target` is the shader-model/target version. Its high byte ≥ 5 marks **SM5**,
  which changes variable-descriptor layout (enables `tex=`/`samp=`).
- `creator` takes the entire rest of the line.
- `rd11` (8 hex u32) appears only for SM5 RDEFs.

> Note the field is spelled **`target`** here but **`version`** in the
> `key=value` fallback ([§6](#6-rdef-keyvalue-fallback)) — same datum, two
> spellings. See [§12.7](#127-rdef-field-name-divergence).

### 5.2 Struct definitions

Emitted once per named struct type (used by structured buffers), before bindings.

```
struct_def  = "struct" SP IDENT { SP struct_attr } SP "{" NL
                 { member_line }
              "};" NL
struct_attr = "class=" INT | "vtype=" INT | "rows=" INT | "cols=" INT
            | "sm5=" hex4
member_line = [ "row_major" SP ] type_stem SP IDENT [ "[" INT "]" ]
                 [ SP "+" INT ] [ SP "sm5=" hex4 ] ";" NL
hex4        = HEX "," HEX "," HEX "," HEX
```

All `struct_attr`s are **omitted when equal to their derived default**
(`class=5`, `vtype=0`, `rows=1`, `cols=struct_size/4`); a parser must apply
those defaults when the attribute is absent. `+INT` on a member is its byte
offset, emitted only when it differs from the tight-packed running position.

### 5.3 Resource bindings

```
binding   = typespec SP IDENT SP ":" SP "register(" class slot ")"
              [ "[" INT "]" ]                 ; bind count, only when ≠ 1
              [ SP "dim="     INT ]           ; non-textures only
              [ SP "ret="     INT ]           ; non-textures only
              [ SP "samples=" INT ]
              [ SP "flags="   flagspec ]
              ";" NL
class     = "b" | "t" | "s" | "u"
slot      = INT
flagspec  = "0" | flagtok { "|" flagtok }
flagtok   = "userPacked" | "comparisonSampler" | "texComp0" | "texComp1"
          | "unused" | HEX0X
```

`typespec` by resource kind (input_type):

| typespec | kind | reg class |
| --- | --- | --- |
| `cbuffer` / `tbuffer` | constant/texture buffer | `b` / `t` |
| `<Dim><...>` e.g. `Texture2DArray<float4>` | SRV texture | `t` |
| `RW<Dim><...>` | UAV typed texture | `u` |
| `SamplerState` | sampler | `s` |
| `StructuredBuffer<T>` / `RWStructuredBuffer<T>` | structured | `t` / `u` |
| `ByteAddressBuffer` / `RWByteAddressBuffer` | raw | `t` / `u` |
| `AppendStructuredBuffer<T>` / `ConsumeStructuredBuffer<T>` | append/consume | `u` |

Texture dimension stems (`Dim`) and their underlying `D3D_SRV_DIMENSION` values
are in [Appendix A.1](#a1-resource-dimensions-rdef--srv). The `<...>` element is
the return type for textures (`float4`, `uint4`, …) or the struct/scalar element
for structured buffers.

**Annotation emission is derivation-conditional** — an annotation is omitted when
it equals a value the parser can recompute, so *absence is meaningful*. To parse
correctly you must reproduce the same derivations:

- `dim=` / `ret=`: emitted only for **non-textures** and only when not equal to
  the structured-type defaults (`dim=1`, `ret=6`). Textures carry these in the
  typespec.
- `samples=`: three regimes — **textures**: absent ⇒ `0xFFFFFFFF` ("not
  multisampled"); **structured**: never emitted (the field is the stride, derived
  from element size); **other**: absent ⇒ `0`.
- `flags=`: absent ⇒ `derived_binding_flags`, which is `texComp0|texComp1`
  (`0xC`) for a typed texture/UAV with a return type, else `0`. An explicit
  `flags=0` is therefore *distinct* from absence.

Flag bit values are in [Appendix A.2](#a2-binding-flags-d3d_shader_input_flags).

### 5.4 cbuffer blocks

```
cbuffer_block = "cbuffer" SP IDENT [ SP ":" SP "register(b" slot ")" ]
                  [ SP "flags="   flagspec ]     ; binding SIF_* flags
                  [ SP "kind="    INT ]
                  [ SP "cbflags=" HEX ]
                  SP "{" NL
                    { var_line }
                "};" NL
var_line      = [ "row_major" SP ] type_stem SP IDENT [ "[" INT "]" ]
                  SP ":" SP "packoffset(" packoff ")"
                  [ SP "size=" INT ]
                  [ SP ( "used" | "unused" | "vflags=" HEX ) ]
                  [ SP "sm5="  hex4 ]
                  [ SP "tex="  INT "," INT ]
                  [ SP "samp=" INT "," INT ]
                  [ SP ( "= " initializer | "default=" HEX ) ]
                  ";" NL
packoff       = "c" INT [ "." ("x"|"y"|"z"|"w") ]   ; register + component
```

The presence of `: register(b…)` on the `cbuffer` line distinguishes the two
overall body modes — see [§12.4](#124-global-mode-detection).

**Variable `used` flag** is the same derivation hazard as binding flags: a
variable with **none** of `used`/`unused`/`vflags=` is marked *derive from the
SHEX program* (the analysis recomputes whether the byte range is read). Without
the program it resolves to "not used". `used` = `SVF_USED` = `0x2`.

> ⚠ `0x2` means **`used`** on a *variable* but **`comparisonSampler`** on a
> *binding/cbuffer* `flags=`. Same bit, two fields, two meanings. See
> [§12.3](#123-flags-token-overloaded-across-two-fields).

**Default-value initializers** (`= initializer`) must be a **single token with no
internal spaces**:

```
initializer = scalar_lit | vector_init | array_init
scalar_lit  = INT | float_lit | "true" | "false"
vector_init = type_stem "(" lit { "," lit } ")"     e.g. float4(0,0,4,0.25)
array_init  = "{" lit { "," lit } "}"               e.g. {0,1,4,5,2,3,6,7,...}
```

When a value can't be rendered this way, `default=<hex>` carries the raw bytes
instead (mutually exclusive with `= initializer`).

### 5.5 Type spellings

```
type_stem = base [ digits ] [ "x" digits ] [ "[" INT "]" ] | struct_name
base      = "bool" | "int" | "float" | "uint" | "double"
```

`float4` = vector, `float4x4` = matrix (`RxC`), trailing `[N]` = array.
`row_major` is emitted as a **separate token** before the stem, never inside it.

---

## 6. RDEF `key=value` fallback

A separate, non-HLSL codec used when the HLSL form can't round-trip. Detected by
a first line starting with `version`.

```
rdef_kv   = "version" SP HEX NL
            "flags"   SP HEX NL
            "creator" SP rest_of_line NL
            [ "rd11" 8*( SP HEX ) NL ]
            { binding_kv | cbuffer_kv }
binding_kv = "binding" SP IDENT { SP key "=" value } NL     ; input,return,dim,samples,slot,count,flags
cbuffer_kv = "cbuffer" SP IDENT { SP key "=" value } NL     ; size,flags,kind
              { var_kv }
var_kv     = "var"    SP IDENT { SP key "=" value } NL       ; offset,size,flags,class,base,rows,cols,elements,typename,sm5,tex,samp,default
              { member_kv }
member_kv  = "member" SP IDENT { SP key "=" value } NL
```

All values are decimal except `flags`/`sm5`/`default` (hex). Within a line the
`key=value` tokens are order-free (bare words are ignored). Unknown head
keywords are a hard error. This form does **not** support nested struct members.

---

## 7. Signatures

One element per line. The grammar depends on the **FourCC from the `.code`
line** (V0 = ISGN/OSGN/PCSG; V5 = OSG5; V1 = ISG1/OSG1/PSG1), which gates the
optional `stream`/`prec` tokens.

```
sig_line = name SP "idx=" INT SP "reg=" INT SP "type=" comptype
                SP "mask=" compmask SP "rw=" compmask
                [ SP "sv="     sysval ]      ; only when ≠ 0
                [ SP "stream=" INT ]         ; V5/V1 only, when ≠ 0
                [ SP "prec="   minprec ]     ; V1 only, when ≠ 0
name     = IDENT | "-"                       ; "-" = empty semantic name
compmask = [ "." ] ("x"|"y"|"z"|"w"){1,4} | HEX | "0"
```

- Separator is a **single space**; there are no commas.
- The mandatory tokens are emitted in the order shown, but on parse the
  `key=value` tokens form an order-free map; **tokens without `=` are silently
  ignored**, and unknown keys are ignored.
- Required keys: `idx`, `reg`, `type`, `mask`, `rw`. Missing any is an error.
- `compmask` is emitted **without** a leading `.` (e.g. `xyzw`); the parser also
  accepts a leading `.` and accepts a bare hex byte (when high bits are set).
- `comptype`, `sysval`, `minprec` vocabularies: [Appendix A.5](#a5-signature-vocabularies).

---

## 8. Statistics (STAT)

One `key value` pair per line, single-space separated, all decimal. Order is
fixed on output but order-free on parse; absent keys default to 0; unknown keys
are an error.

```
stat_line = "size" SP INT NL
          | "sample_frequency" SP INT NL
          | "reserved" SP INT SP INT SP INT SP INT NL
          | counter_key SP INT NL
```

`size` reproduces the exact byte length and `reserved` carries four dwords
written to non-contiguous payload offsets — both must be preserved verbatim for
byte-identity. Full key list: [Appendix A.6](#a6-stat-keys).

---

## 9. SHEX (shader program)

The SHEX body (after the `.code SHEX`/`SHDR` line) is:

```
shex      = profile NL { instruction NL }
```

### 9.1 Profile line

```
profile   = shadertype "_" major "_" minor [ SP FOURCC ]
shadertype = "ps" | "vs" | "gs" | "hs" | "ds" | "cs"
```

The FourCC suffix appears only when it is not the default `SHEX` (e.g.
`ps_5_0 SHDR`). `major`/`minor` are decimal.

### 9.2 Instruction line

```
instruction = mnemonic { modifier } [ SP operand { ", " operand } ]
mnemonic    = opcode_name | "op" INT          ; "op<n>" = unknown opcode
```

- The mnemonic is matched by **longest opcode-name prefix**; the remainder is
  the modifier tail and must be empty or begin with `_`.
- **Source operands are separated by `", "`** (comma + exactly one space).
  Declaration operands and `dcl_input`/`dcl_output` operands are
  **space-separated** instead ([§9.7](#97-declarations)).

### 9.3 Modifiers

Emitted in this fixed order, each only when non-default; on parse they are
order-tolerant. The tail is split on `_` **at paren depth 0** only.

```
modifier = "_ri" INT                 ; resinfo return type
         | "_sat"                    ; saturate
         | "_nz"                     ; test non-zero
         | "_pm" HEX                 ; precise-component mask
         | "_sf" HEX                 ; sync flags
         | "_off(" i8 "," i8 "," i8 ")"   ; texel offsets (no spaces)
         | "_res(" resdim ")" | "_rd" HEX8   ; resource-dim ext. token
         | "_rt(" rettype{4 ×, no spaces} ")" | "_rr" HEX8  ; return-type ext. token
resdim   = dimname [ "," "stride=" INT ]
```

`_res`/`_rd` are two encodings of the same field (readable vs hex fallback);
likewise `_rt`/`_rr`. A parser must treat them as alternatives.

### 9.4 Operands

```
operand    = [ "-" ] [ "|" ] core [ components ] [ "|" ]
core       = immediate | "?reg(" INT ")" indices | prefix indices
prefix     = <register prefix, Appendix A.3>
```

- Modifier order is fixed: negate `-` precedes abs `|…|` (so `-|r0.x|`;
  `|-r0.x|` is **not** accepted).
- `?reg(<n>)` is the escape hatch for an unrecognized register type (preserves
  the raw value).

### 9.5 Immediates

```
immediate  = ( "l" | "d" ) "(" [ value { ", " value } ] ")"
value      = float_lit | INT | "0x" HEX8
```

- `l(...)` = 32-bit immediate, `d(...)` = 64-bit (always rendered as raw hex).
- **Value type on output** is chosen per opcode (float vs signed-int vs
  unsigned-hex). **Value type on input** is purely syntactic: `0x…` ⇒ raw bits;
  a token containing `.`/`e`/`inf`/`nan` ⇒ float; otherwise ⇒ signed int. The two
  agree because the float writer always emits a `.`/`e` and the uint writer
  always emits `0x`.
- **Redundant component suppression:** a bare (un-indexed) immediate omits its
  component selection entirely; the parser restores it from the value count
  (1 value ⇒ scalar `.1`; ≥2 ⇒ empty write-mask). This is why a vector immediate
  reads `l(1.0, 1.0, 0.0, 0.0)` with **no** trailing `:` or `.xyzw`.

### 9.6 Components and indices

```
components = ":" { axis }          ; WRITE-MASK (destination); 0 axes ⇒ empty mask
           | ".1"                  ; the implied 1-component (OneComponent)
           | "." axis              ; SCALAR read-select
           | "." axis axis axis axis   ; SWIZZLE (read)
axis       = "x" | "y" | "z" | "w"
indices    = [ INT [ "L" ] ] { "[" index "]" }
index      = INT [ "L" ]           ; immediate (L ⇒ 64-bit)
           | operand               ; relative, e.g. cb0[r0.y]
           | operand " + " INT     ; relative + constant, e.g. cb0[r0.y + 11]
```

**`:` vs `.` is the core disambiguator:** `:` introduces a **destination
write-mask** (an unordered on/off set of lanes), `.` introduces a **source read**
(`.1`, a scalar `.x`, or an ordered 4-lane swizzle `.zxwz` where repeats are
legal). The first index of an operand is unbracketed (`r0`, `cb0`); all further
indices are bracketed.

### 9.7 Declarations

Each `dcl_*` mnemonic has a fixed trailing syntax. Tags are `name(value)` with no
inner spaces and the inner text must not contain `)`. Selected forms:

```
dcl_globalFlags  flag { "|" flag }
dcl_temps        INT
dcl_indexableTemp  INT INT INT
dcl_constantbuffer  operand SP "access(" cbaccess ")"
dcl_sampler      operand SP "mode(" sampmode ")"
dcl_resource     dimname SP "(" rettype "," rettype "," rettype "," rettype ")"
                   SP operand [ SP "samples(" INT ")" ]      ; samples only for MS dims
dcl_resource_structured  operand SP "stride(" INT ")"
dcl_resource_raw operand
dcl_uav_typed    dimname SP "(" rettype×4 ")" SP operand SP "flags(0x" HEX ")"
dcl_uav_structured  operand SP "stride(" INT ")" SP "flags(0x" HEX ")"
dcl_uav_raw      operand SP "flags(0x" HEX ")"
dcl_input_ps     "interp(" interp ")" SP operand
dcl_input*       [ "interp(" interp ")" ] { SP operand } [ SP "sv(" sysval ")" ]
dcl_output*      { SP operand } [ SP "sv(" sysval ")" ]
dcl_hsMaxTessFactor  "0x" HEX8            ; float bits as hex, NOT a decimal float
dcl_thread_group     INT SP INT SP INT
```

`samples(N)` is **optional and only emitted for multisampled resource
dimensions** (`texture2dms`/`texture2dmsarray`); absent ⇒ 0. Tessellation / GS /
HS / CS / function / interface declarations follow the same `tag(value)` pattern;
see `serialize.rs`/`parse.rs` for the exhaustive list. `dcl_stream`,
`dcl_tgsm_raw`, and `dcl_tgsm_structured` have **no dedicated syntax** and
serialize as generic `", "`-separated operand lists.

### 9.8 customdata

```
customdata "icb" SP "{" icb_row { "," icb_row } SP "}"     ; icb_row = " 0xHEX8 0xHEX8 0xHEX8 0xHEX8"
customdata ("comment"|"debuginfo"|"opaque"|"other(" INT ")") SP "{" { SP "0x" HEX8 } SP "}"
```

---

## 10. A worked example

The opening of a pixel-shader container (abridged), showing the layering:

```
// ===== forensic header (informational, ignored on parse) =====
.dxbc version=1 hash=00000000000000000000000000000000 checksum=00000000000000000000000000000000
.code RDEF form=hlsl
target=0xffff0500
flags=0x100
creator=Microsoft (R) HLSL Shader Compiler 10.0.10011.0
rd11=31314452,3c,18,20,28,24,c,0
SamplerState gDirShadowMapSampler : register(s1);
Texture2DArray<float4> gDirShadowMapTexture : register(t0);
cbuffer cbShared : register(b0) flags=userPacked {
    float4x4 gWorldToProj : packoffset(c0) used=false;
    bool gDirLightEnabled : packoffset(c4) used=false;
};
.code ISGN
SV_Position idx=0 reg=0 type=float mask=.xyzw rw=none sv=position
.code SHEX
profile=ps_5_0
dcl_globalFlags refactoringAllowed
dcl_constantbuffer cb0[32].xyzw access=dynamicIndexed
dcl_resource t0 dim=texture2darray rt=float,float,float,float samples=0
mad_sat r0:y r4.z cb0[r0.y+11].x cb0[r0.y+11].y
sample_res(texture2darray)_rt(float,float,float,float) r6:z r7.zxwz t0.yzxw s1
ret
.code STAT
size=148
sample_frequency=false
instructions=191
.end
```

Note the operand forms: write-mask `r6:z`, read-swizzle `r7.zxwz`, vector
immediate `l(a,b,c,d)` (comma, no space), and relative index `cb0[r0.y+11]`
(`+`, no space) — operands are whitespace-separated tokens, list values use `,`.

Note `r6:z` (write-mask `z`) vs `r7.zxwz` (read swizzle) vs `t0.yzxw` (resource
component order) on the `sample` line — see [§9.6](#96-components-and-indices).

---

## 11. Design invariants

1. **Lossless round-trip.** `assemble(serialize(x)) == x` byte-for-byte for every
   chunk. Editable codecs are only used when verified to round-trip; otherwise
   raw hex preserves the bytes.
2. **The forensic `//` header is informational.** It is regenerated on
   serialize and discarded on parse. Nothing in it is authoritative.
3. **Self-validation by re-encode.** The serializer checks each editable body
   round-trips before choosing it. A consequence: the editable grammars are
   *lossy by construction* and not every binary is representable in HLSL/text
   form — some chunks will always be raw hex.

---

## 12. Inconsistencies & ambiguities

These are places where the current format is hard to parse, internally
inconsistent, or context-sensitive. Each lists the **Observed** behavior (what a
compatible parser must accept today) and a **Recommended** direction for a
cleaner, consistent grammar. Recommendations are *not yet implemented*.

### 12.1 Separator zoo

**Observed.** The same conceptual "list" uses four different separators
depending on context: `", "` (SHEX source operands, immediate values), `" "`
(signature/STAT tokens, SHEX declaration operands), `","` (return-type lists,
texel offsets, HLSL initializers), `"|"` (flag labels). Several tokenizers
require the *exact* separator (`", "`, `" + "`) and reject tab or extra spaces.

**Why it's a problem.** A parser cannot use one generic "split on commas/spaces"
routine; it must know, per production, which exact separator applies. Hand-edits
that add a stray space silently fail to parse.

**Recommended.** Pick one list separator (`", "`) and one token separator
(single space), normalize all producers to them, and make all consumers
whitespace-tolerant (collapse runs of spaces/tabs except inside quoted names).
At minimum, accept `, ` / `,` / ` , ` interchangeably in list positions.

### 12.2 `//` is line-truncating and non-escapable

**Observed.** `//` anywhere in a line deletes the rest of the line, even
mid-token, with no escape. Applied uniformly including to data lines.

**Why it's a problem.** Any name or value containing `//` is silently corrupted.
Today this is latent (no current token legitimately contains `//`), but it
constrains the grammar and is a footgun for generators.

**Recommended.** Restrict comment stripping to `//` that is preceded by
whitespace or start-of-line, or require comments to occupy their own line. Document
that names may not contain `//`.

### 12.3 `flags=` token overloaded across two fields

**Observed.** On a binding/cbuffer header, `flags=` uses
`D3D_SHADER_INPUT_FLAGS` (`userPacked=1, comparisonSampler=2, texComp0=4,
texComp1=8, unused=16`). On a variable line, the flag surface is
`used`/`unused`/`vflags=` using `D3D_SHADER_VARIABLE_FLAGS` (`used=2`). **Bit
`0x2` means `comparisonSampler` in one field and `used` in the other.**

**Why it's a problem.** A reader that shares a flag-decoding routine across the
two contexts will mislabel bits. (This class of bug was already found and fixed
once in the metadata header.)

**Recommended.** Keep the two vocabularies textually distinct (they already are:
`flags=` labels vs `used`/`unused`/`vflags=`). Document the bit tables side by
side ([Appendix A.2](#a2-binding-flags-d3d_shader_input_flags),
[A.4](#a4-variable-flags-d3d_shader_variable_flags)) and never reuse a single
decoder.

### 12.4 Global mode detection (merged vs two-section RDEF)

**Observed.** Whether the RDEF HLSL body is in "merged" or "two-section" form is
decided by scanning **all** lines for a `cbuffer …{` that lacks `": register("`.
Editing one block's register token flips the parsing of the *entire* body.

**Recommended.** Add an explicit marker (e.g. a header token `rdef_form=merged`)
rather than inferring mode from the absence of a substring.

### 12.5 Derivation-conditional annotations

**Observed.** Several annotations are omitted when they equal a value the parser
must recompute, so **absence is significant**:
- binding `flags=` absent ⇒ `derived_binding_flags(input_type, return_type)`.
- binding `dim=`/`ret=` absent ⇒ structured defaults (and never present for
  textures).
- `samples=` absent ⇒ a *type-dependent* sentinel (`0xFFFFFFFF` for textures,
  `0` otherwise, stride for structured).
- variable `used` absent ⇒ **recompute from the SHEX program** (cbuffer read
  analysis). Parsing RDEF correctly therefore *requires the SHEX chunk*.

**Why it's a problem.** A standalone RDEF parser cannot be local: it must
reproduce nontrivial derivations and, for `used`, must have the decoded program.

**Recommended.** For a clean grammar, make the text **self-contained**: always
emit `flags`/`dim`/`ret`/`samples`/`used` explicitly (accept their omission for
backward compatibility, but don't rely on cross-chunk derivation). This trades a
little verbosity for a context-free RDEF grammar.

### 12.6 Header-hash preservation

**Observed.** Some official compiler test fixtures intentionally carry a stale
header digest. Recomputing it during an otherwise unchanged reassembly violates
the format's byte-identity invariant.

**Implemented.** The directive records both the stored `hash=` and the computed
content `checksum=`. Their separation lets unchanged documents restore even a
stale hash, while edits are detected and receive a valid newly computed hash.
Legacy documents that omit both fields continue to recompute the hash.

### 12.7 RDEF field-name divergence (`target` vs `version`)

**Observed.** The same shader-model field is spelled `target` in the HLSL form
and `version` in the `key=value` fallback. This doubles as the form-detection
heuristic, so it can't simply be unified without changing detection.

**Recommended.** If the forms are unified later, use one spelling and a separate
explicit form marker (see §12.4).

### 12.8 Longest-prefix tokenization (opcodes and registers)

**Observed.** Both opcode mnemonics and register prefixes are matched by
**longest prefix**, scanning all known names. A naive first-match parser
mis-splits e.g. `dcl_input_ps_sgv` (also a prefix-superset of `dcl_input`) or
reads `v` instead of `vThreadID`.

**Recommended.** Document that tokenization is maximal-munch over the name
tables in [Appendix A.3](#a3-register-prefixes); generate the tables from a
single source.

### 12.9 Optional/positional vs named tags

**Observed.** Some trailing tags are named and order-free (`interp(...)`,
`sv(...)` within a `dcl_input`), while others are positional-at-end
(`samples(...)`). Signature/STAT tokens are order-free on parse but fixed on
emit; tokens without `=` (signatures) are silently dropped.

**Recommended.** Treat all trailing tags as named and order-free, and make
unknown/`=`-less tokens a *warning* rather than silent drop, so typos surface.

### 12.10 Format-specific encodings that look like other types

**Observed.** `dcl_hsMaxTessFactor` and customdata ICB rows encode **floats as
`0x`-hex bits**, not decimal floats. Signature `mask=`/`rw=` switch between
letter-form and hex-byte form based on whether high bits are set.

**Recommended.** Document these explicitly (done here); consider a distinct
sigil (e.g. `f32:0x…`) so a value's type is self-describing.

---

## Appendix A: Token tables

### A.1 Resource dimensions

Two **distinct** enumerations exist and must not be conflated (this was the
source of a real mislabeling bug):

| Name (RDEF/HLSL stem) | `D3D_SRV_DIMENSION` (RDEF binding) | SHEX `dcl_resource` token | `D3D10_SB_RESOURCE_DIMENSION` |
| --- | --- | --- | --- |
| Buffer | 1 | `buffer` | 1 |
| Texture1D | 2 | `texture1d` | 2 |
| Texture1DArray | 3 | `texture1darray` | 7 |
| Texture2D | 4 | `texture2d` | 3 |
| Texture2DArray | 5 | `texture2darray` | 8 |
| Texture2DMS | 6 | `texture2dms` | 4 |
| Texture2DMSArray | 7 | `texture2dmsarray` | 9 |
| Texture3D | 8 | `texture3d` | 5 |
| TextureCube | 9 | `texturecube` | 6 |
| TextureCubeArray | 10 | `texturecubearray` | 10 |
| (SHEX-only) | — | `raw_buffer` | 11 |
| (SHEX-only) | — | `structured_buffer` | 12 |

The RDEF binding's dimension field uses the **SRV** numbering; the SHEX
`dcl_resource` token uses the **SB** numbering. They are different integers for
the same dimension (e.g. Texture2DArray is 5 in RDEF but 8 in SHEX).

### A.2 Binding flags (`D3D_SHADER_INPUT_FLAGS`)

| Bit | Label |
| --- | --- |
| `0x1` | `userPacked` |
| `0x2` | `comparisonSampler` |
| `0x4` | `texComp0` |
| `0x8` | `texComp1` |
| `0x10` | `unused` |

`texComp0|texComp1` (`0xC`) encodes a 4-component return type and is the derived
default for typed textures.

### A.3 Register prefixes

`r v o x l d s t cb icb label vPrim oDepth null rasterizer oMask stream
function_body function_table interface function_input function_output
vOutputControlPointID vForkInstanceID vJoinInstanceID vicp vocp vpc vDomain
thisPointer u g vThreadID vThreadGroupID vThreadIDInGroup vCoverage
vThreadIDInGroupFlattened vGSInstanceID oDepthGE oDepthLE vCycleCounter`, plus the
escape form `?reg(<n>)`. Tokenized by **longest match**.

### A.4 Variable flags (`D3D_SHADER_VARIABLE_FLAGS`)

| Bit | Text |
| --- | --- |
| `0x2` | `used` (absent token ⇒ `unused` or derive-from-program) |

Other bits, if present, are emitted as `vflags=<hex>`.

### A.5 Signature vocabularies

- `type=` (component type): `unknown uint int float uint16 int16 float16 uint64
  int64 float64`, else raw decimal.
- `sv=` (system value): `position clip_distance cull_distance
  render_target_array_index viewport_array_index vertex_id primitive_id
  instance_id is_front_face sample_index` + tessellation factor names +
  `target depth coverage depth_greater_equal depth_less_equal stencil_ref
  inner_coverage`, else raw decimal.
- `prec=` (min precision): `min16f min2_8f reserved min16i min16u any16 any10`,
  else raw decimal.

### A.6 STAT keys

Header: `size`, `sample_frequency`, `reserved` (×4). Counters (payload order):
`instructions temps defines declarations float_ops int_ops uint_ops static_flow
dynamic_flow macros temp_arrays array_ops cuts emits tex_normal tex_load tex_comp
tex_bias tex_gradient movs movcs conversions gs_input_prim gs_output_topo
gs_max_verts gs_instances hs_control_points hs_output_prim hs_partitioning
ds_domain barriers interlocked tex_store`.

### A.7 SHEX enum tables

- Shader types: `ps`=0 `vs`=1 `gs`=2 `hs`=3 `ds`=4 `cs`=5.
- Interpolation: `undefined constant linear linearCentroid linearNoperspective
  linearNoperspectiveCentroid linearSample linearNoperspectiveSample` (0–7).
- Sampler modes: `default`=0 `comparison`=1 `mono`=2.
- CB access: `immediateIndexed`=0 `dynamicIndexed`=1.
- Tess domains: `undefined isoline tri quad` (0–3).
- Tess partitioning: `undefined integer pow2 fractional_odd fractional_even` (0–4).
- Tess output prim: `undefined point line triangle_cw triangle_ccw` (0–4).
- Return types (resource): `unorm snorm sint uint float mixed double continued
  unused` (1–9), else `unknown<v>`.
- Global flags (bit-indexed): `refactoringAllowed enableDoublePrecisionFloatOps
  forceEarlyDepthStencil enableRawAndStructuredBuffers skipOptimization
  enableMinPrecision enable11_1DoubleExtensions enable11_1ShaderExtensions`.
