extern crate napi_build;

use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    napi_build::setup();
    println!("cargo:rustc-check-cfg=cfg(engpe_metal)");

    let target_arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    if target_arch == "aarch64" {
        println!("cargo:rustc-cfg=aarch64_neon");
    }

    if target_os != "macos" {
        return;
    }

    println!("cargo:rustc-cfg=engpe_metal");
    println!("cargo:rerun-if-changed=metal/");
    println!("cargo:rerun-if-changed=fhe_shaders.metallib");

    let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let include = manifest.join("metal/include");
    let mm_compute = manifest.join("metal/src/metal_compute.mm");
    let mm_bridge = manifest.join("metal/src/metal_bridge.mm");

    // Compile ObjC++ Metal host.
    cc::Build::new()
        .cpp(true)
        .file(&mm_compute)
        .file(&mm_bridge)
        .include(&include)
        .flag("-fobjc-arc")
        .flag("-std=c++17")
        .flag("-Wno-unused-parameter")
        .compile("engpe_metal");

    println!("cargo:rustc-link-lib=framework=Metal");
    println!("cargo:rustc-link-lib=framework=Foundation");

    let metallib = manifest.join("fhe_shaders.metallib");
    // Rebuild metallib whenever Metal sources change (cargo:rerun-if-changed above).
    if let Err(e) = compile_metallib(&manifest) {
        eprintln!("cargo:warning=metallib rebuild failed: {e}");
        if !metallib.exists() {
            panic!("fhe_shaders.metallib missing and compile failed: {e}");
        }
    }
}

fn compile_metallib(manifest: &PathBuf) -> Result<(), String> {
    let shader_dir = manifest.join("metal/shaders");
    let out = manifest.join("fhe_shaders.metallib");
    let build = manifest.join("target/shader-air");
    std::fs::create_dir_all(&build).map_err(|e| e.to_string())?;

    let metals = [
        "common/fhe_common.metal",
        "ntt/ntt_forward.metal",
        "ntt/ntt_inverse.metal",
        "modular/modmul_direct.metal",
        "modular/modmul_batch.metal",
        "modular/keyswitch_mac.metal",
        "modular/mod_down_batch.metal",
    ];
    let mut airs = Vec::new();
    for rel in metals {
        let src = shader_dir.join(rel);
        if !src.exists() {
            continue;
        }
        let air = build.join(format!("{}.air", rel.replace('/', "_")));
        let status = Command::new("xcrun")
            .args([
                "-sdk",
                "macosx",
                "metal",
                "-std=metal3.0",
                "-O3",
                "-target",
                "air64-apple-macos14.0",
                "-I",
            ])
            .arg(&shader_dir)
            .arg("-c")
            .arg(&src)
            .arg("-o")
            .arg(&air)
            .status()
            .map_err(|e| e.to_string())?;
        if !status.success() {
            return Err(format!("metal compile failed for {rel}"));
        }
        airs.push(air);
    }
    if airs.is_empty() {
        return Err("no shaders compiled".into());
    }
    let mut cmd = Command::new("xcrun");
    cmd.args(["-sdk", "macosx", "metallib"]);
    for a in &airs {
        cmd.arg(a);
    }
    cmd.arg("-o").arg(&out);
    let status = cmd.status().map_err(|e| e.to_string())?;
    if status.success() {
        Ok(())
    } else {
        Err("metallib link failed".into())
    }
}
