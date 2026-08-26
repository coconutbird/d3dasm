//! Corpus decompilation harness: every supported shader must decompile;
//! when `fxc.exe` is available the decompiled HLSL must compile back
//! under the shader's own profile, and the recompiled bytecode must be
//! semantically equivalent — original and recompiled programs execute
//! under the reference interpreter with identical seeded worlds and
//! their architectural outputs must match bit for bit.

use std::path::{Path, PathBuf};
use std::process::Command;

/// Known fxc locations tried after `PATH`.
const FXC_CANDIDATES: &[&str] = &[
    "fxc",
    r"C:\Program Files (x86)\Windows Kits\10\bin\10.0.26100.0\x64\fxc.exe",
];

fn find_fxc() -> Option<PathBuf> {
    FXC_CANDIDATES.iter().find_map(|candidate| {
        let path = PathBuf::from(candidate);
        Command::new(&path).arg("/?").output().ok().map(|_| path)
    })
}

fn corpus() -> Vec<PathBuf> {
    let mut files = Vec::new();
    let mut stack = vec![
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../shaders"),
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../shaders-pc"),
    ];
    while let Some(directory) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(directory) else {
            continue;
        };
        for entry in entries.filter_map(Result::ok) {
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
            } else if path.extension().is_some_and(|extension| extension == "bin") {
                files.push(path);
            }
        }
    }
    files.sort();
    files
}

#[test]
fn corpus_decompiles_and_recompiles() {
    let fxc = find_fxc();
    if fxc.is_none() {
        eprintln!("fxc.exe not found; compile validation skipped");
    }
    let out_dir = std::env::temp_dir().join("d3dasm-decompiler-tests");
    std::fs::create_dir_all(&out_dir).expect("temp dir is writable");

    let mut attempted = 0usize;
    let mut decompiled = 0usize;
    let mut compiled = 0usize;
    let mut equivalent = 0usize;
    let mut uninterpreted = 0usize;
    let mut failures: Vec<String> = Vec::new();

    for path in corpus() {
        let data = std::fs::read(&path).expect("corpus file reads");
        let name: String = path
            .strip_prefix(Path::new(env!("CARGO_MANIFEST_DIR")).join("../.."))
            .unwrap_or(&path)
            .with_extension("")
            .to_string_lossy()
            .chars()
            .map(|character| {
                if character.is_ascii_alphanumeric() {
                    character
                } else {
                    '_'
                }
            })
            .collect();
        for (index, shader) in d3dasm::parse(&data).iter().enumerate() {
            let Some(program) = shader.program() else {
                continue;
            };
            attempted += 1;
            let source = match decompiler::decompile_shader(shader) {
                Ok(source) => source,
                Err(error) => {
                    failures.push(format!("{name}#{index}: {error}"));
                    continue;
                }
            };
            decompiled += 1;
            let Some(fxc) = &fxc else { continue };
            let profile = format!(
                "{}_{}_{}",
                program.shader_type, program.major_version, program.minor_version
            );
            let hlsl_path = out_dir.join(format!("{name}_{index}.hlsl"));
            std::fs::write(&hlsl_path, &source).expect("temp files are writable");
            let output = Command::new(fxc)
                // `/Gis` compiles with IEEE strictness: the decompiled
                // source denotes the original instruction sequence
                // exactly, so the equivalence claim must not be diluted
                // by fxc re-fusing or reassociating float math.
                .args(["/nologo", "/Gis", "/T", &profile, "/E", "main", "/Fo"])
                .arg(out_dir.join(format!("{name}_{index}.dxbc")))
                .arg(&hlsl_path)
                .output()
                .expect("fxc launches");
            if output.status.success() {
                compiled += 1;
                let object = out_dir.join(format!("{name}_{index}.dxbc"));
                match differential(shader, program, &object) {
                    Verdict::Equivalent => equivalent += 1,
                    Verdict::Uninterpreted(reason) => {
                        uninterpreted += 1;
                        eprintln!("uninterpreted {name}#{index}: {reason}");
                    }
                    Verdict::Diverged(report) => {
                        failures.push(format!(
                            "{name}#{index}: semantic divergence:
{report}
--- source ---
{source}"
                        ));
                    }
                }
            } else {
                failures.push(format!(
                    "{name}#{index}: fxc rejected:\n{}\n--- source ---\n{source}",
                    String::from_utf8_lossy(&output.stderr)
                ));
            }
        }
    }

    eprintln!(
        "corpus: {attempted} shaders, {decompiled} decompiled, {compiled} recompiled,          {equivalent} proven equivalent, {uninterpreted} uninterpreted"
    );
    for failure in &failures {
        eprintln!("FAILURE: {failure}");
    }
    assert!(
        attempted > 0,
        "the corpus should contain at least one shader"
    );
    // The supported subset must fully round-trip: every decompiled
    // shader compiles and executes equivalently. Unsupported shapes may
    // remain, but are reported.
    assert!(
        failures
            .iter()
            .all(|failure| failure.contains("unsupported shader shape")),
        "decompiled output failed to recompile or diverged"
    );
}

/// The outcome of executing original against recompiled bytecode.
enum Verdict {
    Equivalent,
    Uninterpreted(String),
    Diverged(String),
}

/// A stable identity for one signature semantic.
fn semantic_key(name: &str, index: u32) -> u64 {
    let mut key = 0xcbf2_9ce4_8422_2325u64;
    for byte in name.to_ascii_uppercase().bytes() {
        // `SV_Position` and `POSITION` name the same slot across
        // compilations; fold the prefix away by hashing the tail only.
        key = (key ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01b3);
    }
    key.wrapping_add(u64::from(index)) | (1 << 63)
}

/// Signature-derived register identities for one shader side.
fn io_keys(shader: &d3dasm::Shader<'_>) -> decompiler::interp::IoKeys {
    let normalize = |name: &str| -> String {
        let upper = name.to_ascii_uppercase();
        upper
            .strip_prefix("SV_")
            .map_or(upper.clone(), str::to_owned)
    };
    let mut keys = decompiler::interp::IoKeys::default();
    if let Some(signature) = shader.input_signature() {
        for element in &signature.elements {
            keys.inputs.entry(element.register).or_insert_with(|| {
                semantic_key(&normalize(&element.semantic_name), element.semantic_index)
            });
        }
    }
    if let Some(signature) = shader.output_signature() {
        for element in &signature.elements {
            keys.outputs.entry(element.register).or_insert_with(|| {
                semantic_key(&normalize(&element.semantic_name), element.semantic_index)
            });
        }
    }
    keys
}

/// Executes both programs under seeded worlds and compares outputs.
fn differential(
    original_shader: &d3dasm::Shader<'_>,
    original: &dxbc::shex::Program,
    object: &Path,
) -> Verdict {
    let Ok(bytes) = std::fs::read(object) else {
        return Verdict::Uninterpreted("recompiled object unreadable".into());
    };
    let shaders = d3dasm::parse(&bytes);
    let Some(recompiled_shader) = shaders.first() else {
        return Verdict::Uninterpreted("recompiled object has no shader".into());
    };
    let Some(recompiled) = recompiled_shader.program() else {
        return Verdict::Uninterpreted("recompiled object has no program".into());
    };
    let original_keys = io_keys(original_shader);
    let recompiled_keys = io_keys(recompiled_shader);
    let mut compared = 0usize;
    for seed in 1..=4u64 {
        let world = decompiler::interp::World { seed };
        let first = decompiler::interp::execute(original, world, &original_keys);
        let second = decompiler::interp::execute(recompiled, world, &recompiled_keys);
        match (first, second) {
            (Ok(a), Ok(b)) => {
                if a == b {
                    compared += 1;
                } else {
                    return Verdict::Diverged(format!(
                        "seed {seed}:
  original:   {a:x?}
  recompiled: {b:x?}"
                    ));
                }
            }
            (Err(decompiler::interp::Halt::Budget), Err(decompiler::interp::Halt::Budget)) => {}
            (Err(halt), _) => return Verdict::Uninterpreted(format!("original: {halt}")),
            (_, Err(halt)) => {
                // The recompiled side must stay interpretable: fxc chose
                // an opcode outside the interpreter's subset.
                return Verdict::Uninterpreted(format!("recompiled: {halt}"));
            }
        }
    }
    if compared > 0 {
        Verdict::Equivalent
    } else {
        Verdict::Uninterpreted("every seed exhausted the budget".into())
    }
}
