"""Report project imports, versions and upstream metadata conflicts separately."""
import importlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
from packaging.version import Version

ROOT = Path(__file__).resolve().parents[1]
COMMIT = "97f5b9f58d9e445a8ecb0218037af7465c3708c0"


def check():
    modules = ("numpy", "pandas", "scipy", "sklearn", "xgboost", "torch", "RNA",
               "oligogym.data", "oligogym.features")
    imports = {}
    for name in modules:
        try:
            module = importlib.import_module(name)
            imports[name] = {"status": "PASS", "path": module.__file__, "version": getattr(module, "__version__", None)}
        except Exception as exc:
            imports[name] = {"status": "FAIL", "error": str(exc)}
    pins = {}
    module_names = {"scikit-learn": "sklearn", "ViennaRNA": "RNA"}
    for line in (ROOT / "requirements.txt").read_text().splitlines():
        if '==' not in line or line.lstrip().startswith('#'):
            continue
        package, expected = line.split('==')
        try:
            actual = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            actual = None
        try:
            runtime = getattr(importlib.import_module(module_names.get(package, package)), '__version__', actual)
        except ImportError:
            runtime = None
        pins[package] = {"expected": expected, "installed_metadata": actual, "runtime": runtime,
                         "matches": bool(actual and runtime) and Version(expected) == Version(actual) == Version(runtime)}
    pip = subprocess.run([sys.executable, '-m', 'pip', 'check'], capture_output=True, text=True)
    checkout = ROOT / 'external/OligoGym'
    revision = subprocess.run(['git', '-C', str(checkout), 'rev-parse', 'HEAD'], capture_output=True, text=True)
    clean = subprocess.run(['git', '-C', str(checkout), 'diff', '--quiet', 'HEAD', '--', 'oligogym'], capture_output=True)
    source_path = imports.get('oligogym.features', {}).get('path', '')
    source_check = {"expected_commit": COMMIT, "actual_commit": revision.stdout.strip(),
                    "source_modules_unmodified": clean.returncode == 0,
                    "imports_from_pinned_checkout": bool(source_path) and Path(source_path).resolve().is_relative_to(checkout.resolve())}
    source_check['passes'] = (revision.returncode == 0 and revision.stdout.strip() == COMMIT and
                              clean.returncode == 0 and source_check['imports_from_pinned_checkout'])
    torch_version = imports.get('torch', {}).get('version')
    pins['torch'] = {"expected": "2.11.0 (CPU or CUDA build)", "runtime": torch_version,
                     "matches": bool(torch_version) and torch_version.split('+')[0] == '2.11.0'}
    return {"imports": imports, "pins": pins, "pip_check": {"exit_code": pip.returncode, "output": pip.stdout + pip.stderr},
            "upstream_source": source_check,
            "project_imports_pass": all(r["status"] == "PASS" for r in imports.values()),
            "pinned_requirements_match": all(r["matches"] for r in pins.values()),
            "dependency_declarations_pass": pip.returncode == 0,
            "scope": "Import and metadata checks, not a full training or optional upstream-feature test"}


if __name__ == '__main__':
    result = check()
    print(json.dumps(result, indent=2))
    if not (result['project_imports_pass'] and result['pinned_requirements_match'] and result['dependency_declarations_pass'] and result['upstream_source']['passes']):
        raise SystemExit(1)
