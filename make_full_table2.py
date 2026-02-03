# make_calib_table_v7.py
import argparse, os, json, math

# ---- 列定义：Dataset | Range | Method | (Rx,Ry,Rz,RRMSE | tx,ty,tz,tRMSE | L1,L2) = 13列
COLS_SPEC = r"lllcccc|cccc|cc"

ROT_KEYS   = ["Rx", "Ry", "Rz", "RRMSE"]
TRANS_KEYS = ["tx", "ty", "tz", "tRMSE"]
ACC_JSON_KEYS = ["1d2.5c", "2d5c"]   # JSON中的键
ACC_HEADER = ["$L_1$", "$L_2$"]  # 表头显示

ALL_KEYS = ROT_KEYS + TRANS_KEYS + ACC_JSON_KEYS
HIGHLIGHT_KEYS = ["RRMSE", "tRMSE"] + ACC_JSON_KEYS

HIGHER_IS_BETTER = {k: False for k in (ROT_KEYS + TRANS_KEYS)}
HIGHER_IS_BETTER.update({k: True for k in ACC_JSON_KEYS})

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

# 只在 KITTI 的第一个 scenario（SCENARIOS[0]）里为方法名追加 \cite{...}
METHOD_CITES = {
    "calibnet": "CalibNet",
    "rggnet": "RGGNet",
    "lccnet": "LCCNet",
    "lccraft_large": "LCCRAFT",
    "calibdepth": "CalibDepth",
    # ours 不加引用
}

SCENARIOS = ["r15_t0.15", "r10_t0.25", "r10_t0.5"]  # 15°0.15m, 10°0.25m, 10°0.5m

def scenario_to_title(s):
    try:
        r, t = s.split("_")
        deg = float(r[1:]); meters = float(t[1:]); cm = meters * 100.0
        def fmt(x): return str(int(x)) if float(x).is_integer() else (("{:.2f}".format(x)).rstrip("0").rstrip("."))
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
        print(f"[WARN] bad json: {p} ({e})"); return None

def get_numeric(summary, key):
    if summary is None: 
        return None
    v = summary.get(key, None)
    return None if v is None else float(v)

def fmt_display(key, val):
    if val is None:
        return "--"
    if key in ['tx', 'ty', 'tz', 'tRMSE']:
        return f"{100.0*val:.3f}"               # m -> cm
    if key in ACC_JSON_KEYS:
        return f"{100.0*val:.2f}\\%" # ratio -> %
    return f"{val:.3f}"

def decorate(text, which):
    if which == "best":
        return r"\textbf{" + text + "}"
    if which == "second":
        return r"\underline{" + text + "}"
    return text

def rank_marks(values, higher_is_better):
    marks = [None]*len(values)
    pairs = [(i,v) for i,v in enumerate(values) if v is not None and not math.isnan(v)]
    if not pairs: return marks
    pairs.sort(key=lambda x: x[1], reverse=higher_is_better)
    marks[pairs[0][0]] = "best"
    if len(pairs) >= 2:
        top = pairs[0][1]
        for i,v in pairs[1:]:
            if not math.isclose(v, top, rel_tol=1e-12, abs_tol=1e-12):
                marks[i] = "second"; break
    return marks

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
        r"\multicolumn{2}{c}{Success Rate (\%)$\uparrow$} \\",
        r"~ & ~ & ~ & Roll & Pitch & Yaw & RMSE & X & Y & Z & RMSE & "
        + " & ".join(ACC_HEADER) + r" \\",
        r"\midrule",
    ]

    for ds_caption, ds_prefix in DATASETS:
        total_rows = len(SCENARIOS) * len(methods)
        first_ds_row = True

        for si, sc in enumerate(SCENARIOS):
            summaries = []
            for method in methods:
                folder = f"{ds_prefix}_{sc}"
                path = os.path.join(root, folder, f"{method}_{sc}.json")
                summaries.append(load_summary(path))

            marks_for_keys = {}
            for k in HIGHLIGHT_KEYS:
                vals = [get_numeric(s, k) for s in summaries]
                marks_for_keys[k] = rank_marks(vals, HIGHER_IS_BETTER[k])

            range_title = scenario_to_title(sc)
            first_range_row = True

            for m_idx, (method, mname) in enumerate(zip(methods, method_names)):
                if first_ds_row:
                    left = rf"\multirow{{{total_rows}}}*{{{ds_caption}}}"
                    first_ds_row = False
                else:
                    left = r"~"

                if first_range_row:
                    mid = rf"& \multirow{{{len(methods)}}}*{{{range_title}}}"
                    first_range_row = False
                else:
                    mid = r"& ~"

                # 仅在 KITTI 的第一个 scenario 中给方法名追加 \cite{...}
                disp_name = mname
                if ds_prefix == "kitti" and si == 0:
                    cite_key = METHOD_CITES.get(method)
                    if cite_key:
                        disp_name = f"{mname}\\cite{{{cite_key}}}"

                s = summaries[m_idx]
                cells = []
                for k in ALL_KEYS:
                    raw = get_numeric(s, k)
                    disp = fmt_display(k, raw)
                    mark = marks_for_keys.get(k, [None]*len(methods))[m_idx] if k in HIGHLIGHT_KEYS else None
                    cells.append(decorate(disp, mark))

                L.append(left + f" {mid} & {disp_name} & " + " & ".join(cells) + r" \\")
            if si < len(SCENARIOS) - 1:
                L.append(r"\cdashline{3-13}[1pt/1pt]")

        L.append(r"\midrule")

    if L[-1] == r"\midrule":
        L.pop()

    L += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
    ]
    return "\n".join(L)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="log")
    ap.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    ap.add_argument("--method_names", nargs="+", default=DEFAULT_METHOD_NAMES)
    ap.add_argument("--save_table", type=str, default="calibration_table_v2.tex")
    args = ap.parse_args()

    assert len(args.methods) == len(args.method_names)
    tex = build_table(args.root, args.methods, args.method_names)
    with open(args.save_table, "w", encoding="utf-8") as f:
        f.write(tex)
    print(f"[OK] saved {args.save_table}")
    print("Expect folders like: log/kitti_r15_t0.15/method_r15_t0.15.json, log/nusc_r10_t0.25/..., etc.")

if __name__ == "__main__":
    main()
