# make_calib_table_v2.py
import argparse, os, json

# ---- 固定列：Dataset | Range | Method | (Rx,Ry,Rz,RRMSE | tx,ty,tz,tRMSE | 3d3c,5d5c) = 13列
COLS_SPEC = r"lllrrrr|rrrr|ll"

ROT_KEYS   = ["Rx", "Ry", "Rz", "RRMSE"]
TRANS_KEYS = ["tx", "ty", "tz", "tRMSE"]
ACC_KEYS   = ["0.5d2.5c", "1d5c"]
ALL_KEYS   = ROT_KEYS + TRANS_KEYS + ACC_KEYS

DATASETS = [
    ("KITTI~\\cite{KITTI}", "kitti"),
    ("nuScenes~\\cite{nuScenes}", "nusc"),
]

DEFAULT_METHODS = [
    "calibnet", "rggnet", "lccnet", "lccraft_large", "calibdepth", "projdualfusion_harmonic"
]
DEFAULT_METHOD_NAMES = [
    "CalibNet", "RGGNet", "LCCNet", "LCCRAFT", "CalibDepth", "Ours"
]

# 三个误差组（每组6个方法，共18行）；目录与文件名后缀用它
SCENARIOS = ["r5_t0.5", "r10_t0.25", "r10_t0.5"]  # 5°0.5m, 10°0.25m, 10°0.5m

def scenario_to_title(s):
    """ 'r5_t0.5' -> '$5^\\circ 50\\mathrm{cm}$' """
    try:
        r, t = s.split("_")
        deg = float(r[1:])
        meters = float(t[1:])
        cm = meters * 100.0
        def fmt(x):
            return str(int(x)) if float(x).is_integer() else (("{:.2f}".format(x)).rstrip("0").rstrip("."))
        return rf"${fmt(deg)}^\circ\,{fmt(cm)}\mathrm{{cm}}$"
    except Exception:
        return rf"${s}$"

def load_summary(p):
    if not os.path.isfile(p):
        print(f"[WARN] missing file: {p}")
        return None
    try:
        data = json.load(open(p, "r"))
        return data[-1] if isinstance(data, list) else data
    except Exception as e:
        print(f"[WARN] bad json: {p} ({e})")
        return None

def fmt_cell(key, val):
    if val is None: return "--"
    if key in ['tx', 'ty', 'tz', 'tRMSE']:             # m -> cm
        return f"{100.0*val:.3f}"
    if key in ("3d3c","5d5c"):     # ratio -> %
        return f"{100.0*val:.2f}\\%"
    return f"{val:.3f}"

def row_vals(summary):
    return [fmt_cell(k, None if summary is None else summary.get(k)) for k in ALL_KEYS]

def build_table(root, methods, method_names):
    L = []
    L += [
        r"\begin{table*}[htbp]",
        r"\centering",
        r"\caption{Calibration Results on KITTI and nuScenes Dataset.}",
        r"\label{Table.calib_metrics}",
        rf"\begin{{tabular}}{{{COLS_SPEC}}}",
        r"\toprule",
        r"\multirow{2}*{Dataset} & \multirow{2}*{Range} & \multirow{2}*{Method} & "
        r"\multicolumn{4}{c}{Rotation ($^\circ$)$\downarrow$} & "
        r"\multicolumn{4}{c}{Translation (cm)$\downarrow$} & "
        r"\multirow{2}*{$0.5^\circ2.5\mathrm{cm}$\,$\uparrow$} & \multirow{2}*{$1^\circ5\mathrm{cm}$\,$\uparrow$} \\",
        r"~ & ~ & ~ & Roll & Pitch & Yaw & RMSE & X & Y & Z & RMSE & ~ & ~ \\",
        r"\midrule",
    ]

    for ds_caption, ds_prefix in DATASETS:
        total_rows = len(SCENARIOS) * len(methods)  # 18
        first_ds_row = True
        for si, sc in enumerate(SCENARIOS):
            range_title = scenario_to_title(sc)
            # Range 列跨 6 行
            first_range_row = True
            for method, mname in zip(methods, method_names):
                # Dataset 列跨 18 行
                if first_ds_row:
                    left = rf"\multirow{{{total_rows}}}*{{{ds_caption}}}"
                    first_ds_row = False
                else:
                    left = r"~"

                # Range 列跨 6 行
                if first_range_row:
                    mid = rf"& \multirow{{{len(methods)}}}*{{{range_title}}}"
                    first_range_row = False
                else:
                    mid = r"& ~"

                folder = f"{ds_prefix}_{sc}"  # e.g., kitti_r5_t0.5
                path = os.path.join(root, folder, f"{method}_{sc}.json")
                vals = row_vals(load_summary(path))

                L.append(left + f" {mid} & {mname} & " + " & ".join(vals) + r" \\")
            # 组内分隔线：不要横跨 Dataset 列，但要横跨 Range 以右边所有列 => 3-13
            if si < len(SCENARIOS) - 1:
                L.append(r"\cdashline{3-13}[1pt/1pt]")
        # 数据集分隔
        L.append(r"\midrule")
    L.pop()  # 将最后一个midrule弹出
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(L)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="log",
                    help="root folder containing dataset_scenario subfolders")
    ap.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    ap.add_argument("--method_names", nargs="+", default=DEFAULT_METHOD_NAMES)
    ap.add_argument("--save_table", type=str, default="calibration_table.tex")
    args = ap.parse_args()
    assert len(args.methods) == len(args.method_names)
    tex = build_table(args.root, args.methods, args.method_names)
    with open(args.save_table, "w", encoding="utf-8") as f:
        f.write(tex)
    print(f"[OK] saved {args.save_table}")
    print("Expect folders like: log/kitti_r5_t0.5/method_r5_t0.5.json, log/nusc_r10_t0.25/..., etc.")

if __name__ == "__main__":
    main()
