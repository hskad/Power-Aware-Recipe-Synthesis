import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path


NODE_TYPES = {
    "PI": 0,
    "PO": 1,
    "AND": 2,
    "NOT": 3,
    "BUF": 4,
    "CONST": 5,
    "LATCH": 6,
    "UNKNOWN": 7,
}


ASSIGN_RE = re.compile(r"^(?P<lhs>[^=]+)=\s*(?P<rhs>.+)$")


def strip_comment(line: str) -> str:
    return line.split("#", 1)[0].strip()


def normalize_signal(signal: str) -> str:
    signal = signal.strip()
    if signal.startswith("!"):
        return signal[1:].strip()
    return signal


def signal_type(signal: str) -> str:
    signal = signal.strip().upper()
    if signal in {"0", "CONST0", "GND"}:
        return "CONST"
    if signal in {"1", "CONST1", "VDD"}:
        return "CONST"
    return "UNKNOWN"


class BenchGraph:
    def __init__(self, source_path: str):
        self.source_path = source_path
        self.nodes = {}
        self.edges = []
        self.primary_inputs = []
        self.primary_outputs = []
        self.latches = []
        self.fanin_map = defaultdict(list)
        self.fanout_map = defaultdict(list)

    def ensure_node(self, name: str, node_kind: str = "UNKNOWN"):
        name = name.strip()
        if name not in self.nodes:
            self.nodes[name] = {"name": name, "kind": node_kind, "expr": None}
        elif self.nodes[name]["kind"] == "UNKNOWN" and node_kind != "UNKNOWN":
            self.nodes[name]["kind"] = node_kind
        return self.nodes[name]

    def add_edge(self, src: str, dst: str):
        src = normalize_signal(src)
        dst = normalize_signal(dst)
        self.ensure_node(src)
        self.ensure_node(dst)
        self.edges.append((src, dst))
        self.fanin_map[dst].append(src)
        self.fanout_map[src].append(dst)

    def parse(self):
        lines = Path(self.source_path).read_text(encoding="utf-8", errors="ignore").splitlines()
        for raw_line in lines:
            line = strip_comment(raw_line)
            if not line:
                continue

            upper = line.upper()
            if upper.startswith("INPUT(") and line.endswith(")"):
                name = line[line.find("(") + 1 : line.rfind(")")].strip()
                self.ensure_node(name, "PI")
                self.primary_inputs.append(name)
                continue

            if upper.startswith("OUTPUT(") and line.endswith(")"):
                name = line[line.find("(") + 1 : line.rfind(")")].strip()
                self.ensure_node(name, "PO")
                self.primary_outputs.append(name)
                continue

            assignment = ASSIGN_RE.match(line)
            if not assignment:
                continue

            lhs = assignment.group("lhs").strip()
            rhs = assignment.group("rhs").strip()
            self.ensure_node(lhs)
            self.nodes[lhs]["expr"] = rhs

            rhs_upper = rhs.upper()
            if rhs_upper.startswith("DFF(") or rhs_upper.startswith("LATCH("):
                self.nodes[lhs]["kind"] = "LATCH"
                inner = rhs[rhs.find("(") + 1 : rhs.rfind(")")]
                if inner.strip():
                    self.add_edge(inner, lhs)
                self.latches.append(lhs)
                continue

            if rhs in {"0", "1"}:
                self.nodes[lhs]["kind"] = "CONST"
                continue

            if "*" in rhs:
                self.nodes[lhs]["kind"] = "AND"
                for token in re.split(r"[\s\*\+\,\(\)]+", rhs):
                    token = token.strip()
                    if token and token != lhs:
                        self.add_edge(token, lhs)
                continue

            if rhs.startswith("!"):
                self.nodes[lhs]["kind"] = "NOT"
                self.add_edge(rhs[1:].strip(), lhs)
                continue

            if rhs.startswith("(") and rhs.endswith(")"):
                inner = rhs[1:-1].strip()
                if inner:
                    self.add_edge(inner, lhs)
                    if any(op in inner for op in ["*", "&"]):
                        self.nodes[lhs]["kind"] = "AND"
                    else:
                        self.nodes[lhs]["kind"] = "BUF"
                continue

            tokens = [token for token in re.split(r"[\s\*\+\,\(\)]+", rhs) if token]
            if len(tokens) == 1:
                self.nodes[lhs]["kind"] = "BUF"
                self.add_edge(tokens[0], lhs)
            elif len(tokens) > 1:
                self.nodes[lhs]["kind"] = "AND"
                for token in tokens:
                    if token != lhs:
                        self.add_edge(token, lhs)

    def topological_depths(self):
        depths = {name: 0 for name, node in self.nodes.items() if node["kind"] == "PI"}

        changed = True
        while changed:
            changed = False
            for name in self.nodes:
                if name in depths:
                    continue
                fanins = self.fanin_map.get(name, [])
                if not fanins:
                    continue
                if all(fanin in depths for fanin in fanins):
                    depths[name] = 1 + max(depths[fanin] for fanin in fanins)
                    changed = True

        return depths

    def build_node_features(self):
        depths = self.topological_depths()
        names = list(self.nodes.keys())
        index_of = {name: index for index, name in enumerate(names)}

        node_features = []
        node_types = []
        for name in names:
            kind = self.nodes[name]["kind"] if self.nodes[name]["kind"] != "UNKNOWN" else signal_type(name)
            node_types.append(kind)
            node_features.append(
                {
                    "name": name,
                    "type": kind,
                    "type_id": NODE_TYPES.get(kind, NODE_TYPES["UNKNOWN"]),
                    "fanin": len(self.fanin_map.get(name, [])),
                    "fanout": len(self.fanout_map.get(name, [])),
                    "depth": depths.get(name, 0),
                }
            )

        edge_index = [[index_of[src], index_of[dst]] for src, dst in self.edges if src in index_of and dst in index_of]

        graph_features = {
            "num_nodes": len(names),
            "num_edges": len(edge_index),
            "num_primary_inputs": len(self.primary_inputs),
            "num_primary_outputs": len(self.primary_outputs),
            "num_latches": len(self.latches),
            "num_and_nodes": sum(1 for node in self.nodes.values() if node["kind"] == "AND"),
            "max_depth": max(depths.values()) if depths else 0,
            "avg_fanin": (sum(len(self.fanin_map.get(name, [])) for name in names) / len(names)) if names else 0.0,
            "avg_fanout": (sum(len(self.fanout_map.get(name, [])) for name in names) / len(names)) if names else 0.0,
        }

        return {
            "source_path": self.source_path,
            "nodes": node_features,
            "edges": edge_index,
            "graph_features": graph_features,
            "node_type_vocab": NODE_TYPES,
        }


def parse_label_csv(csv_path: str):
    required = {"design_id", "recipe_id", "output_name", "power_switch"}
    forbidden = {"and", "lev", "lat", "pi", "po", "and_count", "levels", "latches"}

    with open(csv_path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        headers = set(reader.fieldnames or [])

        missing = sorted(required - headers)
        if missing:
            raise ValueError(f"Label CSV missing required columns: {missing}")

        present_forbidden = sorted(headers & forbidden)
        if present_forbidden:
            raise ValueError(
                "Label CSV contains forbidden optimized non-power columns: "
                f"{present_forbidden}. Keep only power_switch as target label."
            )

        rows = []
        for row in reader:
            key = (row["design_id"].strip(), row["recipe_id"].strip())
            try:
                power_switch = float(row["power_switch"])
            except ValueError as error:
                raise ValueError(f"Invalid power_switch for key {key}: {row['power_switch']}") from error

            tech_lib = row.get("tech_lib", "").strip() or None
            tech_lib_path = row.get("tech_lib_path", "").strip() or None
            tech_super_path = row.get("tech_super_path", "").strip() or None
            tech_map = row.get("tech_map", "").strip() or None
            tech_lib_command = row.get("tech_lib_command", "").strip() or None

            rows.append(
                {
                    "key": key,
                    "design_id": key[0],
                    "recipe_id": key[1],
                    "output_name": row["output_name"].strip(),
                    "power_switch": power_switch,
                    "tech_lib": tech_lib,
                    "tech_lib_path": tech_lib_path,
                    "tech_super_path": tech_super_path,
                    "tech_map": tech_map,
                    "tech_lib_command": tech_lib_command,
                }
            )

    return rows


def parse_recipe_index(recipe_index_path: str):
    if not recipe_index_path:
        return {}

    rows = json.loads(Path(recipe_index_path).read_text(encoding="utf-8"))
    recipe_map = {}
    for row in rows:
        recipe_id = row.get("recipe_id")
        if recipe_id:
            recipe_map[recipe_id] = row
    return recipe_map


def export_graph(
    input_path: Path,
    output_dir: Path,
    design_id: str,
    recipe_id: str,
    power_switch: float,
    recipe_meta: dict | None,
    tech_lib: str | None,
    tech_lib_path: str | None,
    tech_super_path: str | None,
    tech_map: str | None,
):
    graph = BenchGraph(str(input_path))
    graph.parse()
    payload = graph.build_node_features()

    payload["design_id"] = design_id
    payload["recipe_id"] = recipe_id
    payload["recipe"] = recipe_meta or {"recipe_id": recipe_id}
    payload["tech_lib"] = tech_lib
    payload["tech_lib_path"] = tech_lib_path
    payload["tech_super_path"] = tech_super_path
    payload["tech_map"] = tech_map
    payload["label"] = {"power_switch": power_switch}
    payload["target"] = power_switch

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{design_id}__{recipe_id}.json"
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Convert BENCH AIGs into recipe-conditioned graph dataset JSON files")
    parser.add_argument("--input-dir", default=str(Path(__file__).resolve().parent / "orig_designs"), help="Directory containing unoptimized .bench files")
    parser.add_argument("--output-dir", default=str(Path(__file__).resolve().parent / "graph_dataset"), help="Directory to write JSON graph files")
    parser.add_argument("--label-csv", default=str(Path(__file__).resolve().parent / "labels_psp.csv"), help="CSV containing strict labels with power_switch")
    parser.add_argument("--recipe-index", default=str(Path(__file__).resolve().parent / "recipe_index.json"), help="Recipe metadata JSON generated by synthesis.py")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    label_rows = parse_label_csv(args.label_csv)
    recipe_map = parse_recipe_index(args.recipe_index) if args.recipe_index else {}

    design_file_map = {bench_file.stem: bench_file for bench_file in sorted(input_dir.glob("*.bench"))}
    if not design_file_map:
        print(f"No .bench files found in {input_dir}")
        return

    output_paths = []
    seen_keys = set()
    missing_designs = []
    tech_fingerprints = set()

    print(f"Building dataset rows for {len(label_rows)} (design, recipe) label pairs...")
    for row in label_rows:
        key = row["key"]
        if key in seen_keys:
            raise ValueError(f"Duplicate label key detected: {key}")
        seen_keys.add(key)

        design_id = row["design_id"]
        recipe_id = row["recipe_id"]
        power_switch = row["power_switch"]
        tech_lib = row.get("tech_lib")
        tech_lib_path = row.get("tech_lib_path")
        tech_super_path = row.get("tech_super_path")
        tech_map = row.get("tech_map")
        tech_fingerprints.add((tech_lib, tech_lib_path, tech_super_path, tech_map))

        bench_file = design_file_map.get(design_id)
        if bench_file is None:
            missing_designs.append(design_id)
            continue

        recipe_meta = recipe_map.get(recipe_id)
        out_path = export_graph(bench_file, output_dir, design_id, recipe_id, power_switch, recipe_meta, tech_lib, tech_lib_path, tech_super_path, tech_map)
        output_paths.append(out_path)
        print(f"- ({design_id}, {recipe_id}) -> {out_path.name}")

    if missing_designs:
        missing_designs = sorted(set(missing_designs))
        raise ValueError(f"Missing unoptimized design files for design_id values: {missing_designs}")

    if len(tech_fingerprints) > 1:
        raise ValueError(f"Mixed technology-mapping metadata detected in label CSV: {sorted(tech_fingerprints)}")

    print(f"Done. Wrote {len(output_paths)} dataset rows to {output_dir}.")


if __name__ == "__main__":
    main()
