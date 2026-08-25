"""ctypes loader for optional ``lut2_gather`` native extension."""

from __future__ import annotations

import ctypes
import logging
import os
import platform
import subprocess
import sys
from pathlib import Path

_LIB = None
_LIB_PATH: Path | None = None
_DLL_DIR_HANDLES = []
_LOAD_WARNED = False
_log = logging.getLogger(__name__)


def native_lib_path() -> Path | None:
    """Return path to shared library if built."""
    here = Path(__file__).resolve().parent
    if sys.platform == "win32":
        for name in ("lut2_gather.dll", "lut2_gather.pyd"):
            p = here / name
            if p.is_file():
                return p
    else:
        for name in ("lut2_gather.so", "lut2_gather.dylib"):
            p = here / name
            if p.is_file():
                return p
    return None


def build_native(*, verbose: bool = True) -> Path:
    """Compile lut2_gather.c in-place (requires MSVC or gcc)."""
    here = Path(__file__).resolve().parent
    src = here / "lut2_gather.c"
    if not src.is_file():
        raise FileNotFoundError(src)
    if sys.platform == "win32":
        out = here / "lut2_gather.dll"
        cc = os.environ.get("CC", "")
        use_msvc = cc.lower() in ("cl", "cl.exe") or (
            cc and Path(cc).name.lower() in ("cl", "cl.exe")
        )
        if not cc and not use_msvc:
            for cand in ("gcc", "cc"):
                import shutil

                if shutil.which(cand):
                    cc = cand
                    break
        if use_msvc:
            cmd = [
                "cl",
                "/nologo",
                "/O2",
                "/openmp",
                "/LD",
                f"/Fe{out}",
                str(src),
            ]
            fast_math = os.environ.get("RWKV_LUT2_FAST_MATH", "auto").strip().lower()
            if fast_math in ("1", "true", "yes", "on"):
                cmd.insert(3, "/fp:fast")
        else:
            cc = cc or "gcc"
            # OpenMP needs libgomp on PATH on Windows; use -fopenmp when MSYS bin is visible.
            import shutil

            use_omp = os.environ.get("RWKV_LUT2_OPENMP", "").strip().lower() in (
                "1",
                "true",
                "yes",
            )
            omp = ["-fopenmp"] if use_omp else []
            avx2 = os.environ.get("RWKV_LUT2_AVX2", "").strip().lower() in (
                "1",
                "true",
                "yes",
            )
            arch = ["-mavx2", "-mfma"] if avx2 else []
            fast_math = os.environ.get("RWKV_LUT2_FAST_MATH", "auto").strip().lower()
            # Fast math is limited to the opt-in AVX2 build.  The grouped-U8
            # kernel only operates on finite quantized weights/activations and
            # benefits substantially from reassociated reductions; portable
            # scalar builds retain strict IEEE behavior unless explicitly
            # requested.
            fast_math_flag = (
                ["-ffast-math"]
                if fast_math in ("1", "true", "yes", "on")
                or (fast_math == "auto" and avx2)
                else []
            )
            cmd = [cc, "-shared", "-O3", *fast_math_flag, *arch, *omp, "-o", str(out), str(src)]
    else:
        out = here / "lut2_gather.so"
        avx2 = os.environ.get("RWKV_LUT2_AVX2", "").strip().lower() in (
            "1",
            "true",
            "yes",
        )
        arch = ["-mavx2", "-mfma"] if avx2 else []
        cmd = [
            os.environ.get("CC", "cc"),
            "-shared",
            "-O3",
            *arch,
            "-fPIC",
            "-fopenmp",
            "-o",
            str(out),
            str(src),
        ]
    if verbose:
        print(" ".join(cmd))
    subprocess.run(cmd, cwd=here, check=True)
    return out


def _windows_dll_dependency_dirs(path: Path) -> list[Path]:
    """Return safe directories for transitive native DLL dependencies.

    Do not walk the whole process ``PATH`` here.  On Windows, merely probing
    an inaccessible PATH entry can make ``ctypes`` fail with ``WinError 5``
    before it reaches the actual DLL.  Native dependencies can be supplied
    explicitly with ``RWKV_LUT2_DLL_DIRS`` (semicolon-separated on Windows).
    Do not guess compiler installation locations: doing so makes a wheel or
    application load an unrelated OpenMP runtime from the host machine.
    """
    candidates = [path.parent]
    raw_dirs = os.environ.get("RWKV_LUT2_DLL_DIRS", "")
    candidates.extend(Path(raw) for raw in raw_dirs.split(os.pathsep) if raw)
    out: list[Path] = []
    seen: set[str] = set()
    try:
        own_dir = path.parent.resolve()
    except OSError:
        own_dir = path.parent
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
            if not resolved.is_dir():
                continue
        except OSError:
            continue
        key = os.path.normcase(str(resolved))
        if key in seen:
            continue
        seen.add(key)
        # The DLL's own directory is always useful.  For external directories
        # only retain locations that actually contain the GCC dependency; this
        # keeps an unrelated or inaccessible directory out of the loader path.
        try:
            has_gomp = (resolved / "libgomp-1.dll").is_file()
        except OSError:
            has_gomp = False
        if resolved == own_dir or has_gomp:
            out.append(resolved)
    return out


def load_native(lib_path: Path | None = None) -> ctypes.CDLL:
    global _LIB, _LIB_PATH
    path = lib_path or native_lib_path()
    if path is None:
        raise FileNotFoundError(
            "lut2_gather native library not built. Run: python -m rwkv_ssd.native.lut2_gather_loader --build"
        )
    if sys.platform == "win32" and hasattr(os, "add_dll_directory"):
        # Python 3.8+ no longer searches PATH for transitive DLL dependencies
        # loaded via ctypes. GCC/OpenMP builds need libgomp-1.dll from the
        # compiler bin directory, so add only known/safe dependency dirs.
        for resolved in _windows_dll_dependency_dirs(path):
            try:
                _DLL_DIR_HANDLES.append(os.add_dll_directory(str(resolved)))
            except OSError:
                pass
    lib = ctypes.CDLL(str(path))
    lib.lut2_gather_packed_export.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_size_t,
    ]
    lib.lut2_gather_packed_export.restype = None
    lib.lut2_layer_gather_export.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.POINTER(ctypes.c_uint8)),
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.c_int,
    ]
    lib.lut2_layer_gather_export.restype = None
    lib.lut2_gather_bf16_packed_export.argtypes = [
        ctypes.POINTER(ctypes.c_uint16),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_size_t,
    ]
    lib.lut2_gather_bf16_packed_export.restype = None
    lib.lut2_layer_gather_bf16_export.argtypes = [
        ctypes.POINTER(ctypes.c_uint16),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.POINTER(ctypes.c_uint8)),
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.c_int,
    ]
    lib.lut2_layer_gather_bf16_export.restype = None
    lib.lut2_gemv_f32_export.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_int,
    ]
    lib.lut2_gemv_f32_export.restype = None
    if hasattr(lib, "trinity_grouped_lut2_gemv_f32_export"):
        lib.trinity_grouped_lut2_gemv_f32_export.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
            ctypes.c_int,
        ]
        lib.trinity_grouped_lut2_gemv_f32_export.restype = None
    if hasattr(lib, "scale_u8_grouped_gemv_f32_export"):
        lib.scale_u8_grouped_gemv_f32_export.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
            ctypes.c_int,
        ]
        lib.scale_u8_grouped_gemv_f32_export.restype = None
    if hasattr(lib, "scale_u8_grouped_cmix_gemv_f32_export"):
        lib.scale_u8_grouped_cmix_gemv_f32_export.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        ]
        lib.scale_u8_grouped_cmix_gemv_f32_export.restype = None
    if hasattr(lib, "scale_u8_grouped_i8_gemv_f32_export"):
        lib.scale_u8_grouped_i8_gemv_f32_export.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
            ctypes.c_int,
        ]
        lib.scale_u8_grouped_i8_gemv_f32_export.restype = None
    if hasattr(lib, "scale_u8_grouped_transposed_gemv_f32_export"):
        lib.scale_u8_grouped_transposed_gemv_f32_export.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
            ctypes.c_int,
        ]
        lib.scale_u8_grouped_transposed_gemv_f32_export.restype = None
    if hasattr(lib, "scale_u8_grouped_transposed_tmix_gemv_f32_export"):
        lib.scale_u8_grouped_transposed_tmix_gemv_f32_export.argtypes = [
            ctypes.POINTER(ctypes.POINTER(ctypes.c_float)),
            ctypes.POINTER(ctypes.POINTER(ctypes.c_uint8)),
            ctypes.POINTER(ctypes.POINTER(ctypes.c_float)),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_int,
        ]
        lib.scale_u8_grouped_transposed_tmix_gemv_f32_export.restype = None
    if hasattr(lib, "scale_u8_grouped_transposed_tmix_fused_f32_export"):
        lib.scale_u8_grouped_transposed_tmix_fused_f32_export.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.POINTER(ctypes.c_uint8)),
            ctypes.POINTER(ctypes.POINTER(ctypes.c_float)),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_int,
        ]
        lib.scale_u8_grouped_transposed_tmix_fused_f32_export.restype = None
    if hasattr(lib, "scale_u8_grouped_sparse_gemv_f32_export"):
        lib.scale_u8_grouped_sparse_gemv_f32_export.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        ]
        lib.scale_u8_grouped_sparse_gemv_f32_export.restype = None
    if hasattr(lib, "scale_u8_grouped_tmix_gemv_f32_export"):
        lib.scale_u8_grouped_tmix_gemv_f32_export.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.POINTER(ctypes.c_uint8)),
            ctypes.POINTER(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_int,
            ctypes.c_int,
        ]
        lib.scale_u8_grouped_tmix_gemv_f32_export.restype = None
    if hasattr(lib, "scale_u8_grouped_tmix_qkv_gemv_f32_export"):
        lib.scale_u8_grouped_tmix_qkv_gemv_f32_export.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.POINTER(ctypes.c_uint8)),
            ctypes.POINTER(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_int,
            ctypes.c_int,
        ]
        lib.scale_u8_grouped_tmix_qkv_gemv_f32_export.restype = None
    if hasattr(lib, "scale_u8_grouped_tmix_qkv_i8_gemv_f32_export"):
        lib.scale_u8_grouped_tmix_qkv_i8_gemv_f32_export.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.POINTER(ctypes.c_uint8)),
            ctypes.POINTER(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_int,
            ctypes.c_int,
        ]
        lib.scale_u8_grouped_tmix_qkv_i8_gemv_f32_export.restype = None
    lib.lut2_tmix_gemv_f32_export.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.POINTER(ctypes.c_uint8)),
        ctypes.POINTER(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_int,
        ctypes.c_int,
    ]
    lib.lut2_tmix_gemv_f32_export.restype = None
    if hasattr(lib, "lut2_tmix_qkv_gemv_f32_export"):
        lib.lut2_tmix_qkv_gemv_f32_export.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.POINTER(ctypes.c_uint8)),
            ctypes.POINTER(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_int,
            ctypes.c_int,
        ]
        lib.lut2_tmix_qkv_gemv_f32_export.restype = None
    _LIB = lib
    _LIB_PATH = path
    return lib


def lib() -> ctypes.CDLL | None:
    global _LIB, _LOAD_WARNED
    if _LIB is not None:
        return _LIB
    path = native_lib_path()
    if path is None:
        if not _LOAD_WARNED:
            _LOAD_WARNED = True
            _log.warning(
                "native lut2_gather library not found — falling back to NumPy/Numba "
                "(rebuild with rwkv_ssd.native.lut2_gather_loader.build_native)"
            )
        return None
    try:
        return load_native(path)
    except OSError as exc:
        if not _LOAD_WARNED:
            _LOAD_WARNED = True
            _log.warning(
                "failed to load native lut2_gather from %s (%s) — falling back to NumPy/Numba",
                path,
                exc,
            )
        return None


def gather_packed(
    out: memoryview, codebook: memoryview, packed: memoryview, n: int
) -> None:
    """Fused unpack+gather into preallocated float32 ``out``."""
    native = lib()
    if native is None:
        raise RuntimeError("native lut2_gather not loaded")
    out_f = (ctypes.c_float * n).from_buffer(out)
    cb_f = (ctypes.c_float * 4).from_buffer(codebook)
    pk = (ctypes.c_uint8 * len(packed)).from_buffer(packed)
    native.lut2_gather_packed_export(out_f, cb_f, pk, n)


def gather_packed_bf16(
    out: memoryview, codebook: memoryview, packed: memoryview, n: int
) -> None:
    """Fused unpack+gather into preallocated bf16 bit pattern ``out``."""
    native = lib()
    if native is None:
        raise RuntimeError("native lut2_gather not loaded")
    out_u16 = (ctypes.c_uint16 * n).from_buffer(out)
    cb_f = (ctypes.c_float * 4).from_buffer(codebook)
    pk = (ctypes.c_uint8 * len(packed)).from_buffer(packed)
    native.lut2_gather_bf16_packed_export(out_u16, cb_f, pk, n)


def gather_layer_bf16(
    flat: memoryview,
    codebooks: memoryview,
    packed_ptrs: list[memoryview],
    offsets: list[int],
    numels: list[int],
) -> None:
    """Batched native bf16 gather into one flat slab."""
    native = lib()
    if native is None:
        raise RuntimeError("native lut2_gather not loaded")
    n_tensors = len(packed_ptrs)
    ptrs = (ctypes.POINTER(ctypes.c_uint8) * n_tensors)()
    pk_arrays: list[memoryview] = []
    for i, pk in enumerate(packed_ptrs):
        pk_arrays.append(pk)
        ptrs[i] = (ctypes.c_uint8 * len(pk)).from_buffer(pk)
    off_a = (ctypes.c_size_t * n_tensors)(*offsets)
    nel_a = (ctypes.c_size_t * n_tensors)(*numels)
    n_total = sum(numels)
    out_u16 = (ctypes.c_uint16 * n_total).from_buffer(flat)
    cb_f = (ctypes.c_float * (n_tensors * 4)).from_buffer(codebooks)
    native.lut2_layer_gather_bf16_export(out_u16, cb_f, ptrs, off_a, nel_a, n_tensors)
    del pk_arrays, ptrs


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Build or probe lut2_gather native lib")
    p.add_argument("--build", action="store_true")
    args = p.parse_args()
    if args.build:
        path = build_native()
        print(f"built {path}")
    path = native_lib_path()
    print(f"platform={platform.platform()}")
    print(f"lib={path}")
    print(f"loaded={lib() is not None}")


if __name__ == "__main__":
    main()
