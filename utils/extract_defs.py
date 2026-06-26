# /usr/bin/env python

import argparse
import ast
import sys
from pathlib import Path


def extract_defs(source):
    """Extract imports & function and class definitions, discarding other statements."""
    lines = source.splitlines(keepends=True)
    tree = ast.parse(source)
    import_snippets = []
    def_snippets = []
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            import_snippets.append("".join(lines[node.lineno - 1 : node.end_lineno]))
        elif isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            lineno = node.decorator_list[0].lineno if node.decorator_list else node.lineno
            def_snippets.append("".join(lines[lineno - 1 : node.end_lineno]))
    out = "".join(sorted(set(import_snippets))) + "\n\n" + "\n\n".join(def_snippets)
    return out


if __name__ == "__main__":
    repo = Path(__file__).parents[1]
    for script_path in (repo / "content" / "python_files").glob("*.py"):
        if script_path.stem.endswith("_lib"):
            continue
        source = script_path.read_text("utf-8")
        output_path = script_path.with_stem(script_path.stem + "_lib")
        output_path.write_text(extract_defs(source), "utf-8")
        sys.stderr.write(
            f"Extracted {script_path.relative_to(repo)} -> {output_path.relative_to(repo)}\n"
        )
