"""Verify the built wheel from outside the source tree, using this interpreter.

Reference: MetPy/WRF-Python separation of package builds and tests.
Climate-Base improvement: no sys.path injection in tests, no network or global
environment mutation, and an explicit assertion about the imported package path.
"""

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


def check_wheel(dist):
    root = Path(__file__).resolve().parents[1]
    wheels = sorted(Path(dist).resolve().glob("climate_preprocessing-*.whl"))
    if len(wheels) != 1:
        raise ValueError("Pass a build directory containing exactly one climate-preprocessing wheel")
    with tempfile.TemporaryDirectory(prefix="climate-wheel-") as directory:
        work = Path(directory)
        installed = work / "installed"
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--no-deps", "--no-index", "--target", str(installed), str(wheels[0])],
            check=True,
        )
        shutil.copytree(root / "tests", work / "tests", ignore=shutil.ignore_patterns("__pycache__"))
        shutil.copytree(root / "examples", work / "examples", ignore=shutil.ignore_patterns("__pycache__"))
        shutil.copytree(root / "tools", work / "tools", ignore=shutil.ignore_patterns("__pycache__"))
        # Preserve pytest marker/options registration in the tree-external run.
        # Package imports still resolve only to the isolated installed wheel.
        shutil.copy2(root / "pyproject.toml", work / "pyproject.toml")
        environment = os.environ.copy()
        inherited = [part for part in environment.get("PYTHONPATH", "").split(os.pathsep)
                     if part and Path(part).resolve() != (root / "src").resolve()]
        environment["PYTHONPATH"] = os.pathsep.join([str(installed), *inherited])
        assertion = (
            "from pathlib import Path; import climate_preprocessing as cp; "
            f"assert Path(cp.__file__).resolve().is_relative_to(Path({str(installed)!r})); "
            "print('Installed wheel:', cp.__version__, cp.__file__)"
        )
        subprocess.run([sys.executable, "-c", assertion], cwd=work, env=environment, check=True)
        subprocess.run(
            [sys.executable, "-m", "pytest", "tests", "-q", "-o", "faulthandler_timeout=30",
             "--basetemp", str(work / "pytest-tmp")],
            cwd=work, env=environment, check=True, timeout=300,
        )
        subprocess.run([sys.executable, "-m", "climate_preprocessing", "--help"],
                       cwd=work, env=environment, check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dist", type=Path)
    check_wheel(parser.parse_args().dist)
