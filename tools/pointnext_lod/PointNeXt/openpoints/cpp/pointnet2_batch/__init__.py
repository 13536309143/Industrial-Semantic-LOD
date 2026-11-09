import torch
import os
import sys
from pathlib import Path

if os.name == "nt":
    torch_lib = Path(torch.__file__).resolve().parent / "lib"
    if torch_lib.exists():
        os.add_dll_directory(str(torch_lib))

    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if cuda_home:
        cuda_bin = Path(cuda_home) / "bin"
        if cuda_bin.exists():
            os.add_dll_directory(str(cuda_bin))

    package_dir = Path(__file__).resolve().parent
    for build_lib in package_dir.glob("build/lib.*"):
        if build_lib.exists():
            sys.path.insert(0, str(build_lib))

import pointnet2_batch_cuda as pointnet2_cuda

