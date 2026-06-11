fn main() {
    #[cfg(feature = "cuda")]
    cuda::compile();

    #[cfg(feature = "hip")]
    hip::compile();
}

#[cfg(feature = "cuda")]
mod cuda {
    use std::path::PathBuf;
    use std::process::Command;

    pub fn compile() {
        let kernel_src = PathBuf::from("src/compute/cuda/kernel.cu");
        let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());
        let ptx_out = out_dir.join("kernel.ptx");

        println!("cargo:rerun-if-changed=src/compute/cuda/kernel.cu");
        println!("cargo:rerun-if-env-changed=CUDA_ARCH");

        if !kernel_src.exists() {
            panic!(
                "src/compute/cuda/kernel.cu not found. \
                 Create the file before building with --features cuda."
            );
        }

        let arch = std::env::var("CUDA_ARCH").unwrap_or_else(|_| "sm_89".into());
        println!("cargo:warning=compiling kernel.cu with -arch={arch}");

        let output = Command::new("nvcc")
            .args([
                "--ptx",
                &format!("-arch={arch}"),
                "-O3",
                "--use_fast_math",
                kernel_src.to_str().unwrap(),
                "-o",
                ptx_out.to_str().unwrap(),
            ])
            .output()
            .expect("nvcc not found — install CUDA Toolkit and ensure nvcc is on PATH");

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            panic!("nvcc PTX compilation failed:\n{stderr}");
        }

        println!("cargo:warning=PTX written to {}", ptx_out.display());
        println!("cargo:rustc-env=KERNEL_PTX_PATH={}", ptx_out.display());
    }
}

#[cfg(feature = "hip")]
mod hip {
    use std::path::{Path, PathBuf};
    use std::process::Command;

    pub fn compile() {
        let kernel_src = PathBuf::from("src/compute/amd/kernel.hip");
        let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());
        let hsaco_out = out_dir.join("kernel.hsaco");

        println!("cargo:rerun-if-changed=src/compute/amd/kernel.hip");
        println!("cargo:rerun-if-env-changed=HIP_ARCH");
        println!("cargo:rerun-if-env-changed=ROCM_PATH");
        println!("cargo:rerun-if-env-changed=HIP_LIB_DIR");

        if !kernel_src.exists() {
            panic!(
                "src/compute/amd/kernel.hip not found. \
                 Create the file before building with --features hip."
            );
        }

        let arch = std::env::var("HIP_ARCH").unwrap_or_else(|_| "gfx1201".into());
        println!("cargo:warning=compiling kernel.hip for --offload-arch={arch}");

        // Build arguments dynamically so we don't hardcode broken absolute paths
        let mut args = vec![
            "--genco".to_string(),
            format!("--offload-arch={arch}"),
            "-O3".to_string(),
            "-ffast-math".to_string(),
            "-mno-wavefrontsize64".to_string(),
            "-DHIP_ENABLE_WARP_SYNC_BUILTINS".to_string(),
            // for gpu assembly debuging
            //"--save-temps".to_string(),

        ];

        // Safely inject the device library path ONLY if the environment variable is present
        if let Ok(hip_lib) = std::env::var("HIP_LIB_DIR") {
            args.push(format!("--rocm-device-lib-path={hip_lib}"));
        }

        args.push(kernel_src.to_str().unwrap().to_string());
        args.push("-o".to_string());
        args.push(hsaco_out.to_str().unwrap().to_string());

        let mut cmd = Command::new("hipcc");
        cmd.args(&args);

        // Pass ROCM_PATH directly to the hipcc process environment
        if let Ok(rocm_path) = std::env::var("ROCM_PATH") {
            cmd.env("ROCM_PATH", rocm_path);
        }

        let output = cmd
            .output()
            .expect("hipcc not found — install ROCm and ensure hipcc is on PATH");

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            panic!("hipcc HSACO compilation failed:\n{stderr}");
        }

        let rocm_path = std::env::var("ROCM_PATH").unwrap_or_else(|_| "/opt/rocm".into());
        if let Ok(hip_lib_dir) = std::env::var("HIP_LIB_DIR") {
            println!("cargo:rustc-link-search=native={hip_lib_dir}");
        } else {
            let rocm_lib64 = format!("{rocm_path}/lib64");
            let rocm_lib = format!("{rocm_path}/lib");

            if Path::new(&rocm_lib64).join("libamdhip64.so").exists() {
                println!("cargo:rustc-link-search=native={rocm_lib64}");
            } else if Path::new(&rocm_lib).join("libamdhip64.so").exists() {
                println!("cargo:rustc-link-search=native={rocm_lib}");
            } else {
                println!("cargo:rustc-link-search=native={rocm_lib}");
            }
        }

        println!("cargo:rustc-link-lib=amdhip64");

        println!("cargo:warning=HSACO written to {}", hsaco_out.display());
        println!("cargo:rustc-env=KERNEL_HSACO_PATH={}", hsaco_out.display());
    }
}
