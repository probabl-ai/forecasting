import ast
import sys
from pathlib import Path


def extract_defs(source, names=None):
    """Extract imports and function/class definitions, discarding other statements."""
    selected = set(names or [])
    lines = source.splitlines(keepends=True)
    tree = ast.parse(source)
    selected_nodes = []

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            if selected and node.name not in selected:
                continue
            selected_nodes.append(node)

    used_names = set()
    if selected:
        for node in selected_nodes:
            for child in ast.walk(node):
                if isinstance(child, ast.Name):
                    used_names.add(child.id)

    import_snippets = []
    def_snippets = []
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            if not selected:
                import_snippets.append("".join(lines[node.lineno - 1 : node.end_lineno]))
                continue

            imported_names = []
            for alias in node.names:
                if isinstance(node, ast.Import):
                    imported_names.append(alias.asname or alias.name.split(".")[0])
                else:
                    imported_names.append(alias.asname or alias.name)
            if any(name in used_names for name in imported_names):
                import_snippets.append("".join(lines[node.lineno - 1 : node.end_lineno]))
        elif isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            if selected and node.name not in selected:
                continue
            lineno = node.decorator_list[0].lineno if node.decorator_list else node.lineno
            def_snippets.append("".join(lines[lineno - 1 : node.end_lineno]))
    out = "".join(sorted(set(import_snippets))) + "\n\n" + "\n\n".join(def_snippets)
    return out


if __name__ == "__main__":
    args = sys.argv[1:]
    file_name = args[0] if args else "feature_engineering.py"
    out_file = None
    names = None

    i = 1
    while i < len(args):
        if args[i] == "--out" and i + 1 < len(args):
            out_file = args[i + 1]
            i += 2
            continue
        if args[i] == "--names" and i + 1 < len(args):
            names = [name for name in args[i + 1].split(",") if name]
            i += 2
            continue
        i += 1

    source = Path(file_name).read_text(encoding="utf-8")
    extracted = extract_defs(source, names=names)
    if out_file:
        Path(out_file).write_text(extracted, encoding="utf-8")
    else:
        print(extracted)
