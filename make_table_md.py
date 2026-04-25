import argparse
import json
import math
import os


METRIC_KEYS = ["RRMSE", "RMAE", "tRMSE", "tMAE"]
ACC_KEYS = ["L1", "L2"]
ALL_KEYS = METRIC_KEYS + ACC_KEYS
HIGHLIGHT_KEYS = ALL_KEYS

HIGHER_IS_BETTER = {k: False for k in METRIC_KEYS}
HIGHER_IS_BETTER.update({k: True for k in ACC_KEYS})

# (save_name, log_prefix, display_name)
DATASETS = [
    ("kitti", "kitti", "KITTI"),
    ("nuscenes", "nusc", "nuScenes"),
]

DEFAULT_METHODS = [
    "corfil2p",
    "directcalib",
    "calibanything",
    "calibnet",
    "rggnet",
    "lccnet",
    "lccraft_large",
    "calibdepth",
    "projdualfusion_harmonic",
]

DEFAULT_METHOD_NAMES = [
    "CoFiI2P",
    "DirectCalib",
    "CalibAnything",
    "CalibNet",
    "RGGNet",
    "LCCNet",
    "LCCRAFT",
    "CalibDepth",
    "Ours",
]

SCENARIOS = ["r15_t0.15", "r10_t0.25", "r10_t0.5"]
RANGE_FG_COLORS = ["#FFFFFF", "#FFFFFF", "#FFFFFF"]
RANGE_BG_COLORS = ["#000000", "#3A3939", "#000000"]

def scenario_to_title(s):
    try:
        r, t = s.split("_")
        deg = float(r[1:])
        meters = float(t[1:])
        cm = meters * 100.0

        def fmt(x):
            if float(x).is_integer():
                return str(int(x))
            return ("{:.2f}".format(x)).rstrip("0").rstrip(".")

        return f"{fmt(deg)}° / {fmt(cm)}cm"
    except Exception:
        return s


def load_summary(path):
    if not os.path.isfile(path):
        print(f"[WARN] missing file: {path}")
        return None
    try:
        data = json.load(open(path, "r"))
        return data[-1] if isinstance(data, list) else data
    except Exception as e:
        print(f"[WARN] bad json: {path} ({e})")
        return None


def get_data_pair(summary, key):
    if summary is None:
        return (None, None)

    if key in ACC_KEYS:
        sr = summary.get("success_rate", {})
        val = sr.get(key, None)
        return (float(val), None) if val is not None else (None, None)

    item = summary.get(key, {})
    if not isinstance(item, dict):
        return (None, None)

    mean_val = item.get("mean", None)
    std_val = item.get("std", None)
    if mean_val is None:
        return (None, None)

    mean_val = float(mean_val)
    std_val = float(std_val) if std_val is not None else 0.0

    if key.startswith("t"):
        mean_val *= 100.0
        std_val *= 100.0

    return (mean_val, std_val)


def fmt_display(key, mean_val, std_val):
    def cal_fmt(val):
        abs_m = abs(val)
        if abs_m < 10:
            prec = 2
        elif abs_m < 100:
            prec = 1
        else:
            prec = 0
        return f"{{:.{prec}f}}"

    if mean_val is None:
        return "--"

    if key in ACC_KEYS:
        return f"{100.0 * mean_val:.1f}%"

    mean_fmt = cal_fmt(mean_val)
    std_fmt = cal_fmt(std_val)
    m_str = mean_fmt.format(mean_val)
    s_str = std_fmt.format(std_val)
    return f"{m_str} +- {s_str}"


def decorate(text, mark):
    if text == "--":
        return text
    if mark == "best":
        return f"<b>{text}</b>"
    if mark == "second":
        return f"<u>{text}</u>"
    return text


def rank_marks(data_pairs, higher_is_better):
    marks = [None] * len(data_pairs)

    valid_pairs = []
    for i, (mean_val, _) in enumerate(data_pairs):
        if mean_val is not None and not math.isnan(mean_val):
            valid_pairs.append((i, mean_val))

    if not valid_pairs:
        return marks

    valid_pairs.sort(key=lambda x: x[1], reverse=higher_is_better)
    best_idx = valid_pairs[0][0]
    best_val = valid_pairs[0][1]
    marks[best_idx] = "best"

    if len(valid_pairs) >= 2:
        for i, val in valid_pairs[1:]:
            if not math.isclose(val, best_val, rel_tol=1e-12, abs_tol=1e-12):
                marks[i] = "second"
                break

    return marks


def build_dataset_table(root, methods, method_names, dataset_display, dataset_prefix):
    lines = []
    lines.append(f"# Calibration Results on {dataset_display}")
    lines.append("")
    lines.append("- Best: bold")
    lines.append("- Second: underline")
    lines.append("")
    lines.append("<table>")
    lines.append("  <thead>")
    lines.append("    <tr>")
    lines.append("      <th>Dataset</th>")
    lines.append("      <th>Range</th>")
    lines.append("      <th>Method</th>")
    lines.append("      <th>RRMSE (°)</th>")
    lines.append("      <th>RMAE (°)</th>")
    lines.append("      <th>tRMSE (cm)</th>")
    lines.append("      <th>tMAE (cm)</th>")
    lines.append("      <th>L1 (%)</th>")
    lines.append("      <th>L2 (%)</th>")
    lines.append("    </tr>")
    lines.append("  </thead>")
    lines.append("  <tbody>")

    for s_idx, scenario in enumerate(SCENARIOS):
        bg = RANGE_BG_COLORS[s_idx % len(RANGE_BG_COLORS)]
        fg = RANGE_FG_COLORS[s_idx % len(RANGE_FG_COLORS)]
        summaries = []
        for method in methods:
            folder = f"{dataset_prefix}_{scenario}"
            path = os.path.join(root, folder, f"{method}_{scenario}.json")
            summaries.append(load_summary(path))

        row_data = []
        for m_idx in range(len(methods)):
            row_data.append({})
            for key in ALL_KEYS:
                row_data[m_idx][key] = get_data_pair(summaries[m_idx], key)

        marks_map = {}
        for key in HIGHLIGHT_KEYS:
            vals = [row_data[i][key] for i in range(len(methods))]
            marks_map[key] = rank_marks(vals, HIGHER_IS_BETTER[key])

        range_title = scenario_to_title(scenario)
        for m_idx, method_name in enumerate(method_names):
            cells = []
            for key in ALL_KEYS:
                mean_v, std_v = row_data[m_idx][key]
                disp = fmt_display(key, mean_v, std_v)
                mark = marks_map[key][m_idx]
                cells.append(decorate(disp, mark))

            td = f' style="background:{bg}; color:{fg};"'
            lines.append("    <tr>")
            lines.append(f"      <td{td}>{dataset_display}</td>")
            lines.append(f"      <td{td}>{range_title}</td>")
            lines.append(f"      <td{td}>{method_name}</td>")
            lines.append(f"      <td{td}>{cells[0]}</td>")
            lines.append(f"      <td{td}>{cells[1]}</td>")
            lines.append(f"      <td{td}>{cells[2]}</td>")
            lines.append(f"      <td{td}>{cells[3]}</td>")
            lines.append(f"      <td{td}>{cells[4]}</td>")
            lines.append(f"      <td{td}>{cells[5]}</td>")
            lines.append("    </tr>")

    lines.append("  </tbody>")
    lines.append("</table>")
    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="log")
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    parser.add_argument("--method_names", nargs="+", default=DEFAULT_METHOD_NAMES)
    parser.add_argument("--save_dir", type=str, default="tables")
    args = parser.parse_args()

    if len(args.methods) != len(args.method_names):
        raise ValueError("--methods and --method_names must have the same length")

    os.makedirs(args.save_dir, exist_ok=True)

    for save_name, dataset_prefix, dataset_display in DATASETS:
        md = build_dataset_table(
            root=args.root,
            methods=args.methods,
            method_names=args.method_names,
            dataset_display=dataset_display,
            dataset_prefix=dataset_prefix,
        )

        out_path = os.path.join(args.save_dir, f"calibration_table_{save_name}.md")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(md)
        print(f"[OK] saved {out_path}")


if __name__ == "__main__":
    main()
