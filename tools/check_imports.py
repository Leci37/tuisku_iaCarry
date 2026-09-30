"""Verify that every local import in the repository resolves to a real file.

This exists because of a specific gap: the step scripts import each other by
bare module name (`from label_gui_utils import ...`), so moving or renaming a
file breaks an import that nothing catches until someone runs that exact script
-- possibly months later. Reorganising the tree without this was guesswork.

The check is **static**: it parses each file and resolves every import against
the tree, so it needs no third-party package installed. That matters here, since
no single environment has both pipelines' stacks (TensorFlow for track A, torch
and ultralytics for track B) and a missing `ultralytics` says nothing about
whether the tree is wired correctly.

    python3 tools/check_imports.py
    python3 tools/check_imports.py --runtime   # also try importing for real

What it checks
  1. Every `import X` / `from X import ...` where X is a module that lives in
     this repository points at a file that exists.
  2. No imported module has a name Python cannot import -- a leading digit, or
     a dot in the stem. Numbered files are entry points; anything imported must
     carry a plain identifier name.

What it does not check: behaviour. A module can resolve every import and still
be wrong.
"""
import argparse
import ast
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Folders whose scripts import their siblings by bare name -- which is what
# happens when you run `python ingestion/video/02_label_gui.py`, since Python
# puts the script's own folder on sys.path.
SCRIPT_DIRS = [
    "common",
    "ingestion/synthetic",
    "ingestion/video",
    "training/tensorflow",
    "training/yolo",
    "training/tflite",
    "serving",
    "serving/verify",
]

# Archived code, kept for reference. Expected to reference modules that moved.
SKIP_DIRS = {"legacy", "tools", "outputs", "models"}

# Not importable standalone under --runtime: entry points that read sys.argv or
# open files relative to a directory they are handed. Their imports are still
# checked statically, which is the part that can break during a rename.
RUNTIME_SKIP = {
    "serving/verify/stub_server.py",   # takes its template/fixture dir as argv[1]
}

# Imported but absent from the repository, and documented as such in the phase
# READMEs. Listed here so their absence reads as a known gap rather than as
# damage from a move. Removing a name from this set turns it back into an error.
KNOWN_MISSING = {
    "utils_bbox":              "ingestion/synthetic -- COCO/bbox helpers, never committed",
    "Utils":                   "ingestion/synthetic/07 -- COCO_json_format_validator, never committed",
    "utils_transfer_learning": "training/tensorflow/03 -- checkpoint helpers, never committed",
}


def local_modules():
    """Map every importable local module name to the file it resolves to."""
    found = {}
    for d in SCRIPT_DIRS:
        full = os.path.join(ROOT, d)
        if not os.path.isdir(full):
            continue
        for f in os.listdir(full):
            if f.endswith(".py"):
                found.setdefault(os.path.splitext(f)[0], []).append(os.path.join(d, f))
    found["common"] = ["common/__init__.py"]
    return found


def top_level_imports(path):
    """(module, lineno) for every import in the file, dotted names kept whole."""
    with open(path, encoding="utf-8", errors="ignore") as fh:
        try:
            tree = ast.parse(fh.read(), filename=path)
        except SyntaxError as e:
            return None, e
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                out.append((a.name, node.lineno))
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:
                out.append((node.module, node.lineno))
    return out, None


def importable(name):
    """Could Python import a module with this stem at all?"""
    return name.isidentifier()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runtime", action="store_true",
                    help="also try importing each module for real (needs the deps)")
    args = ap.parse_args()

    modules = local_modules()
    errors, warnings, checked, resolved = [], [], 0, 0

    for d in SCRIPT_DIRS:
        full = os.path.join(ROOT, d)
        if not os.path.isdir(full):
            warnings.append(("(folder)", d, "listed in SCRIPT_DIRS but absent"))
            continue
        for f in sorted(os.listdir(full)):
            if not f.endswith(".py"):
                continue
            rel = os.path.join(d, f)
            checked += 1

            stem = os.path.splitext(f)[0]
            imports, syntax_error = top_level_imports(os.path.join(full, f))
            if syntax_error is not None:
                errors.append((rel, syntax_error.lineno, "SyntaxError: %s" % syntax_error.msg))
                continue

            for mod, lineno in imports:
                head = mod.split(".")[0]

                if head in KNOWN_MISSING:
                    warnings.append((rel, lineno, "known missing: %s (%s)"
                                     % (head, KNOWN_MISSING[head])))
                    continue

                if head == "common":
                    tail = mod.split(".")[1:] or []
                    target = os.path.join(ROOT, "common", *(tail[:-1] + [(tail[-1] + ".py") if tail else "__init__.py"]))
                    if not os.path.exists(target):
                        errors.append((rel, lineno, "`%s` -> %s does not exist"
                                       % (mod, os.path.relpath(target, ROOT))))
                    else:
                        resolved += 1
                    continue

                if head not in modules:
                    continue                      # third party, or stdlib

                siblings = [p for p in modules[head] if os.path.dirname(p) == d]
                if not siblings:
                    where = ", ".join(modules[head])
                    errors.append((rel, lineno,
                                   "`%s` is not a sibling -- it lives in %s" % (mod, where)))
                    continue
                if not importable(head):
                    errors.append((rel, lineno,
                                   "`%s` cannot be imported: a module name must be a valid "
                                   "identifier (numbered files are entry points, not imports)"
                                   % mod))
                    continue
                resolved += 1

            # An imported module must itself have an importable name.
            imported_by_someone = any(
                head == stem
                for other in os.listdir(full) if other.endswith(".py") and other != f
                for head, _ in (top_level_imports(os.path.join(full, other))[0] or [])
                for head in [head.split(".")[0]]
            )
            if imported_by_someone and not importable(stem):
                errors.append((rel, 0, "this file is imported but its name is not a valid "
                                       "identifier -- rename it without the leading digit"))

    print("checked %d file(s); %d local import(s) resolved" % (checked, resolved))

    if warnings:
        print("\nKNOWN GAPS (documented in the phase READMEs, not caused by a move):")
        seen = set()
        for rel, lineno, msg in warnings:
            key = (rel, msg)
            if key in seen:
                continue
            seen.add(key)
            print("   %s:%s  %s" % (rel, lineno, msg))

    if errors:
        print("\nBROKEN -- these are real:")
        for rel, lineno, msg in errors:
            print("   %s:%s  %s" % (rel, lineno, msg))
        print("\nFAIL")
        return 1

    print("\nOK -- every local import resolves")

    if args.runtime:
        print("\n(--runtime) importing for real; third-party misses are expected:")
        import importlib.util
        for d in SCRIPT_DIRS:
            full = os.path.join(ROOT, d)
            if not os.path.isdir(full):
                continue
            sys.path.insert(0, full)
            saved_argv = sys.argv
            try:
                for f in sorted(os.listdir(full)):
                    if not f.endswith(".py") or f == "__init__.py":
                        continue
                    rel = os.path.join(d, f)
                    if rel.replace(os.sep, "/") in RUNTIME_SKIP:
                        print("   %-52s skipped (needs argv)" % rel)
                        continue
                    path = os.path.join(full, f)
                    # Scripts read sys.argv at import; hide this tool's own flags.
                    sys.argv = [path]
                    spec = importlib.util.spec_from_file_location(
                        "chk_" + f[:-3].replace(".", "_"), path)
                    module = importlib.util.module_from_spec(spec)
                    try:
                        spec.loader.exec_module(module)
                    except ModuleNotFoundError as e:
                        print("   %-52s needs %s" % (rel, e.name))
                    except BaseException as e:      # noqa: BLE001 - scripts run at import
                        print("   %-52s %s: %s" % (rel, type(e).__name__, e))
            finally:
                sys.argv = saved_argv
                sys.path.remove(full)
    return 0


if __name__ == "__main__":
    sys.exit(main())
