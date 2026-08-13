# `.d3dasm` Desired Grammar (target design)

This is a **forward-looking redesign** of the `.d3dasm` text format: the way it
*should* work, not the way it currently does. Its companion,
[`d3dasm-grammar.md`](d3dasm-grammar.md), describes the **current** format and
catalogues its inconsistencies in §12. This document takes each of those and
commits to a concrete, consistent resolution.

The two documents share notation (see `d3dasm-grammar.md` §1) so they can be
diffed section by section.

> **Status: implemented.** Treated as still-`v1` (no compatibility with older
> `.d3dasm` text is kept; artifacts are regenerated). **Done:** uniform
> `key=value` tags and `profile=`/`fourcc=` in SHEX; always-emit `samples=`;
> `dim=`/`rt=` resource tags; whitespace-separated operands with `,`-only list
> values and `+`-no-space relative indices; STAT and signatures in `key=value`
> form with `.xyzw`/`none` masks; unknown tags are errors; **self-contained
> RDEF** — `used=true|false` is always emitted and the SHEX-program derivation is
> gone (RDEF parses standalone); `key=value` RDEF headers (`target=`/`version=`)
> and a `.code RDEF form=hlsl|kv` marker. **Intentionally not done** (low value /
> already unambiguous): the `@reg` shorthand (HLSL `register(...)` kept), typed
> `f32(...)` immediate literals (bare `0x` raw-bits already round-trip), and the
> `//`-comment-hardening / name-quoting (no current data triggers it). Section
> references like *(fixes §12.1)* point at the inconsistency the rule resolves.

---

## 1. Goals

In priority order:

1. **Losslessly round-trippable** — the non-negotiable invariant. The redesign
   keeps every escape hatch (raw hex chunks, `default=`, `vflags=`, `?reg`,
   `op<n>`) so any binary is still representable.
2. **Context-free per section** — any section can be parsed in isolation, with no
   dependence on another chunk (today RDEF needs the decoded SHEX program to know
   the `used` bit). *(fixes §12.5)*
3. **Self-describing** — a token's type and a body's form are stated, never
   inferred from a heuristic. *(fixes §12.4, §12.7, §12.10)*
4. **One way to write a list, one way to write a tag** — uniform separators and a
   single `key=value` tag syntax everywhere. *(fixes §12.1, §12.9)*
5. **Whitespace-tolerant** — indentation and run-length of inner whitespace never
   change meaning. *(fixes §12.2 partially)*
6. **Diff-friendly and greppable** — stable token order, one statement per line,
   names quoted only when necessary.

Non-goals: matching `fxc` syntax (this is its own format), and human
*editability* beyond what round-trip safety allows.

---

## 2. Decisions at a glance

| # | Current problem (`d3dasm-grammar.md` §12) | Desired rule |
| --- | --- | --- |
| 1 | Four list separators; exact-separator tokenizers | **One** list separator `,` (whitespace around it optional); tokens separated by any run of spaces/tabs |
| 2 | `//` truncates mid-token, non-escapable | `//` is a comment only at line start or after whitespace; names with specials are **quoted** |
| 3 | `flags=` bit `0x2` means two different things | Binding flags and variable flags are **separate keywords** (`flags=` vs `var_flags=`) with documented, non-overlapping vocab |
| 4 | RDEF merged/two-section inferred from a substring | Explicit `form=` marker on the `.code RDEF` line |
| 5 | Annotations omitted when "derivable"; RDEF needs SHEX | **Always emit** `flags`/`dim`/`ret`/`samples`/`used`; derivation is opt-in via `derive` |
| 6 | Stale header hashes broke byte identity | Store `hash=` plus the original content `checksum=`; rehash only after edits |
| 7 | `target` vs `version` for the same field | Always `target=` |
| 8 | Longest-prefix opcode/register tokenization | Keep maximal-munch, but **document the token tables** as the contract |
| 9 | Mixed positional/named tags; silent token drops | Operands positional-first, then **named** `key=value` tags; unknown tags are an **error**, not dropped |
| 10 | Floats encoded as hex bits (tess factor, ICB) | Render real values as typed literals; `0x…` only as an explicit raw-bits fallback with a sigil |

---

## 3. Lexical conventions

### 3.1 Lines and comments *(fixes §12.2)*

Still line-oriented, but comments are safe:

- A `//` begins a comment **only** at the start of a line or when immediately
  preceded by whitespace. A `//` inside a token (e.g. within a quoted string) is
  literal.
- Indentation and trailing whitespace are insignificant.
- Blank lines are insignificant.

### 3.2 Whitespace and separators *(fixes §12.1)*

- **Token separator:** any non-empty run of spaces/tabs. Producers emit a single
  space; consumers accept any run.
- **List separator:** `,` with **optional** surrounding whitespace. `a,b`,
  `a, b`, and `a , b` are equivalent.
- No production requires an *exact* separator string.

### 3.3 Tokens

```
INT     = [ "-" ] digit { digit }
HEX     = "0x" hexdigit { hexdigit }              ; "0x" is mandatory and is the type marker
FLOAT   = ... a literal containing "." or "e"/"E", or one of inf, -inf, nan
RAWF32  = "f32" HEX                                ; raw 32-bit float pattern, e.g. f32:0x7fc00000 written f32(0x7fc00000)
NAME    = bare_name | quoted
bare_name = (letter | "_") { letter | digit | "_" }
quoted    = '"' { any-char-except-unescaped-quote } '"'
```

Rules:

- **Integers are decimal**, hex is always `0x`-prefixed. The two are never
  ambiguous, so a reader never needs context to type a number. *(fixes §12.10)*
- **Floats always carry `.` or `e`** (or are `inf`/`nan`). A value whose decimal
  form would not round-trip to identical bits is written `f32(0x........)` — an
  explicit, self-describing raw-bits literal — never a bare `0x…`. *(fixes
  §12.10)*
- **Names are quoted** iff they contain whitespace, `//`, `"`, or a leading `.`.
  This lets the `creator` string and any odd semantic name be a single token
  without special "rest of line" handling. *(fixes §12.2)*

### 3.4 Tags *(fixes §12.9)*

Every optional/auxiliary value is a **named tag**:

```
tag      = key "=" value
value    = INT | HEX | FLOAT | RAWF32 | NAME | list
list     = value { "," value }
```

There is exactly one tag syntax. The current `tag(value)` paren form
(`access(...)`, `samples(...)`) and the `key=value` form are unified to
`key=value`. Tag order within a statement is free. An **unknown tag key is an
error** (typos surface immediately) rather than being silently ignored.

---

## 4. Document grammar

```
document    = magic_line { raw_segment | container }
magic_line  = ".d3dasm" SP INT NL                  ; format version, e.g. ".d3dasm 2"
container   = dxbc_line { section } end_line
dxbc_line   = ".dxbc" SP "version=" INT
              [ SP "hash=" HEX32 SP "checksum=" HEX32 ] NL
end_line    = ".end" NL
raw_segment = ".raw" NL hex_block
section     = ".code"  SP FOURCC { SP tag } NL body
            | ".chunk" SP FOURCC NL hex_block
```

Changes from current:

- A leading **`.d3dasm <version>`** line declares the grammar version so a parser
  branches explicitly instead of sniffing. *(fixes §12.4/§12.7 at the document
  level)*
- `.code <FOURCC>` may carry **section tags** — notably `form=` for RDEF
  *(fixes §12.4)*.
- `.end` is **required** to close every container (no "EOF also works" special
  case), so the whole-file and single-container parsers behave identically
  *(removes the §12 "`.end` optional" asymmetry)*.
- `.code` vs `.chunk` is still serializer-chosen by round-trip success; a parser
  accepts either for any FourCC (unchanged — this one is fine).

### 4.1 Hex blocks

Unchanged in spirit, but whitespace-tolerant:

```
hex_block = { hex_line }
hex_line  = { hexbyte } NL
hexbyte   = hexdigit hexdigit
```

Bytes may be separated by whitespace or not; rows may be any length. Output
stays 32 bytes/line, lowercase, for readability.

---

## 5. RDEF *(fixes §12.3, §12.4, §12.5, §12.7)*

The body form is **explicit** on the section line:

```
.code RDEF form=hlsl    ; editable HLSL reconstruction
.code RDEF form=kv      ; flat key=value (fallback for shapes HLSL can't express)
```

No more sniffing `target` vs `version`, and no scanning for a register-less
`cbuffer` to pick merged/two-section. If a future split is still needed it is a
second tag (`layout=merged|split`), never inferred.

### 5.1 Header

```
rdef_hlsl = header { struct_def } { binding } { cbuffer_block }
header    = "target="  HEX                          ; SM/target, e.g. target=0xffff0500
            "flags="   HEX
            "creator=" quoted
            [ "rd11="  HEX{8, comma-list} ]
```

- One field name, **`target`**, in both forms. *(fixes §12.7)*
- `creator` is a quoted string, not a rest-of-line. *(fixes §12.2)*
- Header fields are tags on one logical line (wrappable), not positional
  keywords.

### 5.2 Bindings — self-contained *(fixes §12.5)*

```
binding = typespec SP NAME SP "@" reg
            [ SP "count="   INT ]
            SP "flags="   flaglist            ; ALWAYS present
            [ SP "dim="    dimname ]          ; present for non-textures
            [ SP "ret="    rettype ]          ; present for non-textures
            [ SP "samples=" INT ]             ; ALWAYS present for textures/MS-capable
            ";" NL
reg      = ("b"|"t"|"s"|"u") INT
flaglist = "none" | flagtok { "," flagtok }
flagtok  = "userPacked" | "comparisonSampler" | "texComp0" | "texComp1"
         | "unused" | HEX                     ; HEX only for unknown bits
```

Key changes:

- `register(s1)` becomes the terser, unambiguous **`@s1`**. (Optional — keeping
  `register(...)` is fine; the point is one consistent spelling.)
- **`flags=` is always emitted** (with `none` when empty), so the reader never has
  to recompute `derived_binding_flags` to learn the value. `derive` may be
  written instead to opt back into the old behavior for compactness. *(fixes
  §12.5)*
- Binding flag labels are the `SIF_*` vocabulary only; the variable `used` bit
  lives in a different keyword (§5.4), so `0x2` is never ambiguous. *(fixes
  §12.3)*
- Dimension is written by **name** (`texture2darray`), never as an integer, so
  the SRV-vs-SB numbering split is invisible at the text level. The single
  canonical name table is [Appendix A.1 of the descriptive doc].

### 5.3 Structs

As today (already reasonable), but attributes use the uniform tag syntax and are
**always emitted when non-default is possible**; `derive` opts into omission.
Member offsets are always written (`offset=`), not the conditional `+N`.

```
struct_def = "struct" SP NAME { SP tag } SP "{" NL { member } "};" NL
member     = [ "row_major" SP ] type SP NAME SP "offset=" INT { SP tag } ";" NL
```

### 5.4 cbuffer blocks *(fixes §12.3, §12.5)*

```
cbuffer_block = "cbuffer" SP NAME SP "@b" INT { SP tag } SP "{" NL { var } "};" NL
var = [ "row_major" SP ] type SP NAME SP "@" packoff
        SP "used="  ("true"|"false")          ; ALWAYS present (no derive-from-program)
        [ SP "size=" INT ]
        [ SP "var_flags=" HEX ]               ; only the non-USED bits
        [ SP tag ]                            ; sm5, tex, samp
        [ SP "init=" initializer | SP "raw=" HEX ]
        ";" NL
packoff = "c" INT [ "." ("x"|"y"|"z"|"w") ]
```

- The variable "used" state is the explicit boolean **`used=true|false`**, named
  differently from binding `flags=` so the two flag fields can never be confused.
  *(fixes §12.3)* It is **always written**, so RDEF no longer needs the decoded
  program to parse. *(fixes §12.5)*
- Default values: `init=` for the readable initializer, `raw=` for the hex
  fallback — both use the one-token rule, and initializers use the canonical `,`
  list separator with optional spaces.

```
initializer = scalar | type "(" list ")" | "{" list "}"
scalar      = INT | FLOAT | RAWF32 | "true" | "false"
```

### 5.5 `form=kv` (flat fallback)

A single flat shape (no second spelling of any field name), one statement per
record, all tags named:

```
rdef_kv = "target=" HEX NL "flags=" HEX NL "creator=" quoted NL [ "rd11=" ... NL ]
          { "binding" SP NAME { SP tag } NL
          | "cbuffer" SP NAME { SP tag } NL { "var" SP NAME { SP tag } NL
                                              { "member" SP NAME { SP tag } NL } } }
```

Same keyword (`target`) and same tag vocab as the HLSL form, so the two RDEF
forms differ only in *structure*, never in *spelling*. *(fixes §12.7)*

---

## 6. Signatures *(fixes §12.1, §12.9)*

```
sig_line = NAME SP "idx=" INT SP "reg=" INT SP "type=" comptype
              SP "mask=" compmask SP "rw=" compmask
              [ SP "sv="     sysval ]
              [ SP "stream=" INT ]
              [ SP "prec="   minprec ]
NAME     = quoted-if-needed | "-"            ; "-" still means empty
compmask = "." ("x"|"y"|"z"|"w"){1,4} | HEX | "none"
```

- All tokens are named tags after the name; order-free on read, fixed on write.
- An unrecognized tag is an **error** (today they are silently dropped).
- Masks are written with a **leading `.`** consistently (matching the SHEX read
  convention) and `none` for empty (instead of `0`). Whether a token is valid for
  the FourCC's layout is validated, not silently ignored.

---

## 7. STAT

Already clean (one `key value` per line). The only change: use the uniform
`key=value` tag form for symmetry, and keep `size`/`reserved` verbatim for
byte-identity.

```
stat_line = "size="  INT NL
          | "sample_frequency=" ("true"|"false") NL
          | "reserved=" INT "," INT "," INT "," INT NL
          | counter_key "=" INT NL
```

---

## 8. SHEX *(fixes §12.1, §12.9, §12.10)*

The instruction model is good; the changes are separators, tags, and typed
literals. Operand `:` (write-mask) vs `.` (read swizzle) is **kept** — it is the
one piece of context-sensitivity that genuinely aids readability and is
unambiguous.

### 8.1 Profile and instructions

```
shex        = "profile=" shadertype "_" major "_" minor [ SP "fourcc=" FOURCC ] NL
              { instruction NL }
instruction = mnemonic { modifier } { SP operand } { SP tag }
```

- **Operands are space-separated and positional; tags follow, named.** This single
  rule replaces today's split where source operands use `", "` but declaration
  operands use `" "`. *(fixes §12.1, §12.9)* A list *value inside a tag* (e.g.
  `rt=float,float,float,float`) uses the canonical `,`.
- The `,`-as-operand-separator is gone; operands are always space-separated. (The
  reader still recovers operand boundaries because every operand is a single
  whitespace-free token — register+indices+components have no internal spaces.)

### 8.2 Modifiers

Unchanged grammar (`_sat`, `_nz`, `_ri…`, `_pm…`, `_sf…`, `_off(...)`,
`_res(...)`/`_rd…`, `_rt(...)`/`_rr…`), with the list-separator and whitespace
rules of §3. `_res`/`_rt` stay the readable forms; `_rd`/`_rr` stay the explicit
hex fallbacks.

### 8.3 Operands, immediates, components

Identical to the current grammar (`d3dasm-grammar.md` §9.4–9.6), with two
clarifications:

- Immediate raw-bits use the typed `f32(0x…)` / `d64(0x…)` literal, never a bare
  `0x…` that a reader must type by opcode. *(fixes §12.10)* Decimal-friendly
  values still print as `1.0`, `0.05`, etc.
- The redundant-component suppression on bare immediates is kept (it is
  unambiguous and is reconstructed from the value count).

### 8.4 Declarations *(fixes §12.1, §12.9, §12.10)*

Every declaration is `mnemonic { operand } { key=value }` — positional operands,
then named tags. Examples in the target form:

```
dcl_temps count=10
dcl_constantbuffer cb0[32].xyzw access=dynamicIndexed
dcl_sampler s1 mode=default
dcl_resource t0 dim=texture2darray rt=float,float,float,float samples=0
dcl_resource_structured t1 stride=120
dcl_uav_typed u0 dim=texture2d rt=float,float,float,float flags=0x0
dcl_input_ps v1:xyzw interp=linear
dcl_output o0:xyzw
dcl_hsMaxTessFactor value=64.0           ; a real float, not 0x42800000
dcl_thread_group x=8 y=8 z=1
dcl_globalFlags flags=refactoringAllowed
```

- `tag(value)` → `tag=value` everywhere. *(fixes §12.1)*
- `samples=` is **always** emitted for resources (self-contained), not only for
  MS dims. *(fixes §12.5 at the SHEX level)*
- `dcl_hsMaxTessFactor` and customdata ICB values render as real floats; a `0x`
  pattern only appears via `f32(...)` when not round-trippable. *(fixes §12.10)*
- `dim=` uses the canonical name; `rt=` is a normal `,`-list tag.

### 8.5 customdata

```
customdata kind=icb { row { "," row } }                ; row = float,float,float,float
customdata kind=(comment|debuginfo|opaque|other) [ id=INT ] { value { "," value } }
```

ICB rows are real floats (with `f32(...)` fallback), comma-separated, consistent
with every other list.

---

## 9. Worked example (same shader, target form)

Compare with `d3dasm-grammar.md` §10 (current form):

```
.d3dasm 2
.dxbc version=1 hash=00000000000000000000000000000000 checksum=00000000000000000000000000000000
.code RDEF form=hlsl
target=0xffff0500 flags=0x100 creator="Microsoft (R) HLSL Shader Compiler 10.0.10011.0"
rd11=0x31314452,0x3c,0x18,0x20,0x28,0x24,0xc,0x0
SamplerState gDirShadowMapSampler @s1 flags=none samples=0xffffffff;
Texture2DArray<float4> gDirShadowMapTexture @t0 flags=texComp0,texComp1 samples=0xffffffff;
cbuffer cbShared @b0 flags=userPacked {
    float4x4 gWorldToProj @c0 used=true;
    bool     gDirLightEnabled @c4 used=true;
};
.code ISGN
SV_Position idx=0 reg=0 type=float mask=.xyzw rw=none sv=position
.code SHEX
profile=ps_5_0
dcl_globalFlags flags=refactoringAllowed
dcl_constantbuffer cb0[32].xyzw access=dynamicIndexed
dcl_resource t0 dim=texture2darray rt=float,float,float,float samples=0
sample_res(texture2darray)_rt(float,float,float,float) r6:z r7.zxwz t0.yzxw s1
ret
.code STAT
instructions=191
.end
```

Everything a parser needs is local to each section: no `samples=` sentinel to
decode against a type table, no `used` bit to recover from the program, no
substring sniff to pick a body mode, one separator for lists, one tag syntax,
and floats that look like floats.

---

## 10. What stays the same (deliberately)

- **`:` write-mask vs `.` read-swizzle** on operands — unambiguous and readable.
- **Raw escape hatches** — `.chunk` hex, `raw=`/`var_flags=`, `?reg(n)`, `op<n>`,
  `_rd`/`_rr` — required for lossless round-trip; they are explicit, not
  inconsistencies.
- **Longest-prefix tokenization** of opcode/register names — kept, but the token
  tables (descriptive doc Appendix A.3, A.7) are the documented contract.
- **Serializer chooses `.code` vs `.chunk`** by round-trip success.

---

## 11. Migration

Because this changes the wire text, it is a **format version bump** (`.d3dasm 2`,
§4). Suggested path:

1. Land the descriptive grammar (`d3dasm-grammar.md`) and this target as the
   contract. *(done)*
2. Write the standalone parser against **v2** while keeping the v1
   serializer/parser for existing artifacts.
3. Add a v2 serializer behind a flag; gate it on the same byte-identity
   round-trip check the current code already uses, so v2 output is provably
   lossless before it is trusted.
4. Provide a `v1 → v2` text transcoder (parse v1 to IR, emit v2) so existing
   `.d3dasm` dumps can be upgraded without the original binaries.
5. Flip the default once v2 round-trips the whole shader corpus with zero
   failures (the existing sweep: 1197 chunks across 78 binaries).

Each decision in §2 is independent, so they can also be adopted piecemeal within
v1 where they are backward-compatible (e.g. always-emitting `flags=`/`used`,
accepting `,`/`, ` interchangeably) ahead of a full v2 cutover.
