"""Run unchanged flat-inference tests on CPU with complete-module imports.

Unrelated parent initializers are bypassed in the main and spawned processes.
Only the runtime's device-selection hook is fixed to CPU; no test is rewritten.
"""
import argparse
import importlib
from pathlib import Path
import sys
import tempfile

import pytest
import torch

from validate_flat_data_loop import load_module


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--tests", type=Path, nargs="+", required=True)
    return parser.parse_args()


def cpu_device():
    return torch.device("cpu")


def prepare_module(args):
    load_module(args.source,args.source_root,"module")
    # Match normal import attribute binding for pytest's dotted monkeypatches.
    for name, module in tuple(sys.modules.items()):
        if name.startswith("vesuvius."):
            parent, _, child = name.rpartition(".")
            if parent in sys.modules:
                setattr(sys.modules[parent], child, module)
    runtime = importlib.import_module(
        "vesuvius.ink_detection.inference.inference_runtime"
    )
    runtime.get_accelerator = cpu_device


def main():
    args = arguments()
    prepare_module(args)
    with tempfile.TemporaryDirectory(prefix="flat_module_tests_") as temporary:
        result = pytest.main([
            *map(str, args.tests),"-q","-c","/dev/null",
            "-o",f"cache_dir={temporary}/cache",f"--basetemp={temporary}/tmp",
        ])
    raise SystemExit(result)


if __name__ == "__main__":
    main()
elif __name__ == "__mp_main__":
    # Spawn imports this runner before unpickling the actual inference dataset.
    prepare_module(arguments())
