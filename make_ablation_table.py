# make_ablation_table_v10.py
import json, os, argparse

# -------- 配置 --------
LOG_DIRS = [
    'log/ablation/projdualfusion_harmonic_r10_t0.5.json',
    'log/ablation/projdualfusion_r10_t0.5.json',
    'log/ablation/projdualfusion_harmonic_f2_r10_t0.5.json',
    'log/ablation/projdualfusion_harmonic_f10_r10_t0.5.json',
    'log/ablation/projfusion_harmonic_r10_t0.5.json',
    'log/ablation/projdualfusion_harmonic_resnet_r10_t0.5.json',
]

# 原 OPTION_ROWS: (Dual, HarmonicEmb, n_h, EncoderFlag)
OPTION_ROWS_ORIG = [
    (r"\cmark", r"\cmark", "6",  r"\cmark"),
    (r"\cmark", r"\xmark", "0",  r"\cmark"),
    (r"\cmark", r"\cmark", "2",  r"\cmark"),
    (r"\cmark", r"\cmark", "10", r"\cmark"),
    (r"\xmark", r"\cmark", "6",  r"\cmark"),
    (r"\cmark", r"\cmark", "6",  r"\xmark"),
]

# 删除 Harmonic Embedding 列后，生成新的三列配置
OPTION_ROWS = []
for dual, harmonic, nh, enc in OPTION_ROWS_ORIG:
    # 如果 Harmonic Embedding 为打叉，则 Harmonic Functions=0
    if harmonic == r"\xmark":
        nh = "0"
    OPTION_ROWS.append((dual, nh, enc))

ROT_KEYS   = ["Rx", "Ry", "Rz", "RRMSE"]
TRANS_KEYS = ["tx", "ty", "tz", "tRMSE"]
ACC_JSON_KEYS = ["0.5d2.5c", "1d5c"]
ALL_KEYS = ROT_KEYS + TRANS_KEYS + ACC_JSON_KEYS

# 列格式: 3个选项列 + 4旋转 + 4平移 + 2成功率
COLS_SPEC = r"ccc|cccc|cccc|cc"


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

def fmt_cell(key, val):
    if val is None:
        return "--"
    if key in ['tx', 'ty', 'tz', 'tRMSE']:
        return f"{100.0*val:.3f}"         # m → cm
    if key in ("0.5d2.5c", "1d5c"):
        return f"{100.0*val:.2f}\\%"      # 比例 → %
    return f"{val:.3f}"

def map_image_encoder(flag):
    # 将 Image Encoder 列的 \cmark/\xmark 映射为具体骨干名称
    if flag == r"\cmark":
        return "DINO-V2"
    if flag == r"\xmark":
        return "ResNet-18"
    return flag  # fallback

def build_table():
    L = []
    L += [
        r"\begin{table*}[htbp]",
        r"\centering",
        r"\caption{Ablation on KITTI at $10^\circ\,50\mathrm{cm}$ (r10\_t0.5).}",
        r"\label{Table.ablation}",
        rf"\begin{{tabular}}{{{COLS_SPEC}}}",
        r"\toprule",
        # ------- 表头第1行 -------
        r"\multirow{2}{*}{\makecell{Dual \\ Branches}} & "
        r"\multirow{2}{*}{\makecell{\#Harmonic \\ Functions}} & "
        r"\multirow{2}{*}{\makecell{Image \\ Encoder}} & "
        r"\multicolumn{4}{c|}{Rotation ($^\circ$)$\downarrow$} & "
        r"\multicolumn{4}{c|}{Translation (cm)$\downarrow$} & "
        r"\multicolumn{2}{c}{Success Rate (\%)$uparrow$} \\",
        # ------- 表头第2行 -------
        r" &  &  & "
        r"Roll & Pitch & Yaw & RMSE & "
        r"X & Y & Z & RMSE & "
        r"$L_1$ & $L_2$ \\",
        r"\midrule",
    ]

    assert len(LOG_DIRS) == len(OPTION_ROWS), "LOG_DIRS and OPTION_ROWS length mismatch"

    for path, opts in zip(LOG_DIRS, OPTION_ROWS):
        s = load_summary(path)
        metric_cells = [fmt_cell(k, get_num(s, k)) for k in ALL_KEYS]
        opts = list(opts)
        opts[2] = map_image_encoder(opts[2])  # 第3列是Image Encoder
        row = " & ".join(list(opts) + metric_cells) + r" \\"
        L.append(row)

    L += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
    ]
    return "\n".join(L)

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
