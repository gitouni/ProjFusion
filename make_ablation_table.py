# make_ablation_table_v13_rmse_only.py
import json, os, argparse
import numpy as np

# -------- 配置 --------
LOG_DIRS = [
    'log/ablation/projdualfusion_harmonic_r10_t0.5.json',
    'log/ablation/projdualfusion_r10_t0.5.json',
    'log/ablation/projdualfusion_harmonic_f2_r10_t0.5.json',
    'log/ablation/projdualfusion_harmonic_f10_r10_t0.5.json',
    'log/ablation/projdualfusion_harmonic_m0_mask_r10_t0.5.json',
    'log/ablation/projdualfusion_rope_r10_t0.5.json',
    'log/ablation/projfusion_harmonic_r10_t0.5.json',
    'log/ablation/projdualfusion_harmonic_resnet_r10_t0.5.json',
]

# 现在的定义： (Dual, ProjectionMargin, PositionalEmbedding, ImageEncoder)
OPTION_ROWS_ORIG = [
    (r"\cmark", r"\cmark", "harmonic ($n_h=6$)",  "DINOv2"),
    (r"\cmark", r"\cmark", "harmonic ($n_h=0$)",  "DINOv2"),
    (r"\cmark", r"\cmark", "harmonic ($n_h=2$)",  "DINOv2"),
    (r"\cmark", r"\cmark", "harmonic ($n_h=10$)", "DINOv2"),
    (r"\cmark", r"\xmark", "harmonic ($n_h=6$)",  "DINOv2"),  # Projection Margin = ×
    (r"\cmark", r"\cmark", "RoPE-2D ($f_B=10^3$)","DINOv2"),
    (r"\xmark", r"\cmark", "harmonic ($n_h=6$)",  "DINOv2"),
    (r"\cmark", r"\cmark", "harmonic ($n_h=6$)",  "ResNet-18"),
]

# 4 个选项列：Dual / Projection Margin / Positional Embedding / Image Encoder
OPTION_ROWS = [(dual, pm, pe, enc) for (dual, pm, pe, enc) in OPTION_ROWS_ORIG]

# 只保留 RMSE 两列
ROT_KEYS   = ["RRMSE"]
TRANS_KEYS = ["tRMSE"]
ACC_JSON_KEYS = ["1d2.5c", "2d5c"]  # L1, L2
ALL_KEYS = ROT_KEYS + TRANS_KEYS + ACC_JSON_KEYS

# 需要做 “最好加粗、次好下划线” 的指标
MARK_KEYS = {"RRMSE", "tRMSE", "1d2.5c", "2d5c"}

# 列格式: 4 个选项列 + 1 rotation RMSE + 1 translation RMSE + 2 success rate
COLS_SPEC = r"cccccccc"

# ---------------- 辅助函数 ----------------
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

def get_num(summary, key):
    if summary is None:
        return None
    v = summary.get(key)
    return None if v is None else float(v)

# ---------- LaTeX 格式化 ----------
def fmt_val(key, val):
    if val is None:
        return None
    if key == 'tRMSE':  # m -> cm
        return 100.0 * val
    if key in ACC_JSON_KEYS:
        return 100.0 * val
    return val

def fmt_cell(key, val):
    if val is None:
        return "--"
    if key == 'tRMSE':
        return f"{val:.3f}"
    if key in ACC_JSON_KEYS:
        return f"{val:.2f}\\%"
    return f"{val:.3f}"

# ---------- 主表构建 ----------
def build_table():
    summaries = [load_summary(p) for p in LOG_DIRS]

    # 提取所有数值矩阵 (rows × 4 指标)
    mat = []
    for s in summaries:
        row_vals = [fmt_val(k, get_num(s, k)) for k in ALL_KEYS]
        mat.append(row_vals)
    mat = np.array(mat, dtype=object)

    # 仅针对 MARK_KEYS 计算最优和次优索引
    best_idx, second_idx = {}, {}
    for j, key in enumerate(ALL_KEYS):
        if key not in MARK_KEYS:
            continue
        col = np.array([x for x in mat[:, j] if x is not None], dtype=float)
        if len(col) == 0:
            continue
        if key in ["RRMSE", "tRMSE"]:      # 越小越好
            order = np.argsort(col)
        else:                              # L1/L2 越大越好
            order = np.argsort(-col)
        best_idx[key] = order[0]
        if len(order) > 1:
            second_idx[key] = order[1]

    # 开始写表格
    L = [
        r"\begin{table*}[htbp]",
        r"\centering",
        r"\caption{Ablation on KITTI at $10^\circ\,50\mathrm{cm}$ (r10\_t0.5).}",
        r"\label{Table.ablation}",
        rf"\begin{{tabular}}{{{COLS_SPEC}}}",
        r"\toprule",
        # 表头第 1 行
        r"\multirow{2}{*}{\makecell{Dual \\ Branches}} & "
        r"\multirow{2}{*}{\makecell{Projection \\ Margin}} & "
        r"\multirow{2}{*}{\makecell{Positional \\ Embedding}} & "
        r"\multirow{2}{*}{\makecell{Image \\ Encoder}} & "
        r"\multirow{2}{*}{\makecell{Rotation \\ RMSE ($^\circ$)$\downarrow$}} & "
        r"\multirow{2}{*}{\makecell{Translation \\ RMSE (cm)$\downarrow$}} & "
        r"\multicolumn{2}{c}{Success Rate (\%)$\uparrow$} \\",
        # 表头第 2 行
        r" & & & & & & $L_1$ & $L_2$ \\",
        r"\midrule",
    ]

    for i, (path, opts) in enumerate(zip(LOG_DIRS, OPTION_ROWS)):
        opts = list(opts)  # [dual, pm, pe, enc]
        cells = []
        for j, key in enumerate(ALL_KEYS):
            val = mat[i, j]
            cell = fmt_cell(key, val)
            if val is not None and cell != "--" and key in MARK_KEYS:
                if best_idx.get(key) == i:
                    cell = r"\textbf{" + cell + "}"
                elif second_idx.get(key) == i:
                    cell = r"\underline{" + cell + "}"
            cells.append(cell)
        L.append(" & ".join(opts + cells) + r" \\")

    L += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
    ]
    return "\n".join(L)

# ---------- 主入口 ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save_table", type=str, default="ablation_kitti_r10_t0p5.tex")
    args = ap.parse_args()
    tex = build_table()
    with open(args.save_table, "w", encoding="utf-8") as f:
        f.write(tex)
    print(f"[OK] saved {args.save_table}")

if __name__ == "__main__":
    main()
