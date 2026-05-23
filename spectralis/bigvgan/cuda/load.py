# Copyright (c) 2024 NVIDIA CORPORATION.
#   Licensed under the MIT license.
#
# Modifications:
#   - Lazy torch/cpp_extension imports (allows structural import without GPU)
#   - MSVC auto-discovery on Windows
#   - Per-GPU cache suffix (avoid recompilation across GPU architectures)
#   - Sanitized cache tokens for safe filesystem paths

import os
import pathlib
import subprocess


def _sanitize_cache_token(token: str) -> str:
    """Replace characters that are unsafe for directory names."""
    return token.replace(" ", "_").replace("/", "_").replace("\\", "_")


def _get_gpu_cache_suffix() -> str:
    """Return a per-GPU cache suffix based on compute capability and CUDA version."""
    import torch
    try:
        if torch.cuda.is_available():
            major = torch.cuda.get_device_capability(0)
            cc = f"sm{major[0]}{major[1]}"
        else:
            cc = "cpu"
    except Exception:
        cc = "cpu"
    try:
        import torch.utils.cpp_extension as cpp_ext
        _, bare_major, bare_minor = _get_cuda_bare_metal_version(cpp_ext.CUDA_HOME)
        cuda_ver = f"cuda{bare_major}{bare_minor}"
    except Exception:
        cuda_ver = "nocuda"
    return _sanitize_cache_token(f"{cc}_{cuda_ver}")


def _get_cuda_bare_metal_version(cuda_dir):
    raw_output = subprocess.check_output(
        [cuda_dir + "/bin/nvcc", "-V"], universal_newlines=True
    )
    output = raw_output.split()
    release_idx = output.index("release") + 1
    release = output[release_idx].split(".")
    bare_metal_major = release[0]
    bare_metal_minor = release[1][0]
    return raw_output, bare_metal_major, bare_metal_minor


def _create_build_dir(buildpath):
    try:
        os.mkdir(buildpath)
    except OSError:
        if not os.path.isdir(buildpath):
            print(f"Creation of the build directory {buildpath} failed")


def load():
    """Load the fused anti-alias activation CUDA kernel.

    Handles MSVC auto-discovery on Windows and caches the compiled .pyd per GPU
    architecture (under build_smXX_cudaXX/) to avoid recompilation.
    """
    from torch.utils import cpp_extension

    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "")

    gpu_suffix = _get_gpu_cache_suffix()
    srcpath = pathlib.Path(__file__).parent.absolute()
    buildpath = srcpath / f"build_{gpu_suffix}"
    _create_build_dir(buildpath)

    # --- MSVC auto-discovery on Windows ---
    extra_cflags = ["-O3"]
    if os.name == "nt":
        msvc_found = False
        for vswhere in [
            r"C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe",
            r"C:\Program Files\Microsoft Visual Studio\Installer\vswhere.exe",
        ]:
            if os.path.exists(vswhere):
                import subprocess as _sp
                try:
                    result = _sp.check_output(
                        [vswhere, "-latest", "-products", "*",
                         "-requires", "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                         "-property", "installationPath"],
                        universal_newlines=True
                    ).strip()
                    if result:
                        msvc_path = os.path.join(result, "VC", "Tools", "MSVC")
                        if os.path.isdir(msvc_path):
                            versions = sorted(os.listdir(msvc_path), reverse=True)
                            if versions:
                                msvc_found = True
                                break
                except Exception:
                    pass
        if not msvc_found:
            print("Warning: MSVC not detected via vswhere; CUDA compilation may fail on Windows.")

    # --- CUDA arch flags ---
    cc_flag = []
    try:
        _, bare_metal_major, _ = _get_cuda_bare_metal_version(cpp_extension.CUDA_HOME)
        if int(bare_metal_major) >= 11:
            cc_flag.append("-gencode")
            cc_flag.append("arch=compute_80,code=sm_80")
    except Exception:
        pass

    extra_cuda_flags = [
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
    ]

    def _cpp_extention_load_helper(name, sources, extra_cuda_flags_inner):
        from torch.utils import cpp_extension as cpp_ext
        return cpp_ext.load(
            name=name,
            sources=sources,
            build_directory=buildpath,
            extra_cflags=extra_cflags,
            extra_cuda_cflags=[
                "-O3",
                "-gencode",
                "arch=compute_70,code=sm_70",
                "--use_fast_math",
            ]
            + extra_cuda_flags_inner
            + cc_flag,
            verbose=True,
        )

    sources = [
        srcpath / "anti_alias_activation.cpp",
        srcpath / "anti_alias_activation_cuda.cu",
    ]
    anti_alias_activation_cuda = _cpp_extention_load_helper(
        "anti_alias_activation_cuda", sources, extra_cuda_flags
    )
    return anti_alias_activation_cuda
