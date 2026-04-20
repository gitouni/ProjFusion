# make_ablation_table_v13_rmse_only.py
import json, os, argparse, math
import numpy as np

# -------- 配置 --------
LOG_DIRS = [
    'log/ablation/projdualfusion_concat_r10_t0.5.json',
    'log/ablation/projdualfusion_harmonic_depth_r10_t0.5.json',
    'log/ablation/projdualfusion_harmonic_depth_m0_r10_t0.5.json',
    'log/ablation/projdualfusion_r10_t0.5.json',
    'log/ablation/projdualfusion_harmonic_f2_r10_t0.5.json',
    'log/ablation/projdualfusion_harmonic_r10_t0.5.json',
    'log/ablation/projdualfusion_harmonic_f10_r10_t0.5.json',
    'log/ablation/projdualfusion_harmonic_m0_mask_r10_t0.5.json',
    'log/ablation/projdualfusion_rope_r10_t0.5.json',
    'log/ablation/projfusion_harmonic_r10_t0.5.json',
    'log/ablation/projdualfusion_harmonic_resnet_r10_t0.5.json'
]

# 现在的定义： (Dual, ProjectionMargin, PositionalEmbedding, ImageEncoder)
OPTION_ROWS_ORIG = [
    (r"\cmark", r"\xmark", "3D", "concatenation",  "DINOv2"),
    (r"\cmark", r"\cmark", "2D", "harmonic ($n_h=6$)",  "DINOv2"),
    (r"\cmark", r"\xmark", "2D", "harmonic ($n_h=6$)",  "DINOv2"),
    (r"\cmark", r"\cmark", "3D", "harmonic ($n_h=0$)",  "DINOv2"),
    (r"\cmark", r"\cmark", "3D", "harmonic ($n_h=2$)",  "DINOv2"),
    (r"\cmark", r"\cmark", "3D", "harmonic ($n_h=6$)",  "DINOv2"),
    (r"\cmark", r"\cmark", "3D", "harmonic ($n_h=10$)", "DINOv2"),
    (r"\cmark", r"\xmark", "3D", "harmonic ($n_h=6$)",  "DINOv2"),  # Projection Margin = ×
    (r"\cmark", r"\cmark", "3D", "RoPE-2D ($f_B=10^3$)","DINOv2"),
    (r"\xmark", r"\cmark", "3D", "harmonic ($n_h=6$)",  "DINOv2"),
    (r"\cmark", r"\cmark", "3D", "harmonic ($n_h=6$)",  "ResNet-18"),
]

# 4 个选项列：Dual / Projection Margin / Positional Embedding / Image Encoder
OPTION_ROWS = [(dual, pm, space, pe, enc) for (dual, pm, space, pe, enc) in OPTION_ROWS_ORIG]

# 只保留 RMSE 两列
ROT_KEYS   = ["RRMSE"]
TRANS_KEYS = ["tRMSE"]
# 注意：如果 metrics.py 生成的键是 "L1", "L2"，请在此处修改为 ["L1", "L2"]
ACC_JSON_KEYS = ["L1", "L2"]  
ALL_KEYS = ROT_KEYS + TRANS_KEYS + ACC_JSON_KEYS

# 需要做 “最好加粗、次好下划线” 的指标
MARK_KEYS = {"RRMSE", "tRMSE", "L1", "L2"}

# 列格式: 4 个选项列 + 1 rotation RMSE + 1 translation RMSE + 2 success rate
COLS_SPEC = r"ccccccccc"

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

def get_data(summary, key):
    """
    返回 (mean, std) 元组。
    """
    if summary is None:
        return (None, None)
    
    # 1. 尝试处理 Success Rate (通常在 success_rate 字典下)
    if key in ACC_JSON_KEYS or key in ["L1", "L2"]:
        # 优先尝试从 success_rate 字段读取
        sr = summary.get("success_rate", {})
        if isinstance(sr, dict) and key in sr:
            val = sr[key]
            return (float(val), None) if val is not None else (None, None)
        # 兼容旧格式：直接在根目录
        if key in summary:
            return (float(summary[key]), None)
    
    # 2. 处理 Rotation / Translation (通常有 mean, std)
    val_obj = summary.get(key)
    if isinstance(val_obj, dict):
        mean_v = val_obj.get('mean')
        std_v = val_obj.get('std', 0.0)
        if mean_v is None: return (None, None)
        return float(mean_v), float(std_v)
    elif val_obj is not None:
        # 旧格式只有数值
        return float(val_obj), 0.0
        
    return (None, None)

# ---------- LaTeX 格式化 ----------
def fmt_val_tuple(key, val_tuple):
    """
    对数值进行单位换算 (m -> cm, ratio -> %)，不处理字符串格式。
    """
    mean_v, std_v = val_tuple
    if mean_v is None:
        return (None, None)
    
    if key in TRANS_KEYS:  # m -> cm
        mean_v *= 100.0
        if std_v is not None: std_v *= 100.0
    
    if key in ACC_JSON_KEYS or key in ["L1", "L2"]:
        mean_v *= 100.0
        # success rate 通常没有 std 或者不需要转换 std (因为是None)
    
    return (mean_v, std_v)

def fmt_cell(key, val_tuple):
    """
    将 (mean, std) 格式化为 LaTeX 字符串。
    Success Rate: "98.50%"
    Metrics: "Mean \pm Std" (自适应精度)
    """
    def cal_fmt(val):
        abs_m = abs(val)
        if abs_m < 10:
            prec = 2
        elif abs_m < 100:
            prec = 1
        # elif abs_m < 1000:
        #     prec = 1
        else:
            prec = 0
            
        # 构建格式化字符串，例如 "{:.3f}"
        return f"{{:.{prec}f}}"
    mean_v, std_v = val_tuple
    if mean_v is None:
        return "--"
    
    # Success Rate: 固定两位小数百分比
    if key in ACC_JSON_KEYS or key in ["L1", "L2"]:
        return f"{mean_v:.1f}\\%"
    
    # Metrics: 自适应精度
    # Mean < 10: 3位
    # 10 <= Mean < 100: 2位
    # Mean >= 100: 1位
    m_fmt = cal_fmt(mean_v)
    s_fmt = cal_fmt(std_v) if std_v is not None else "0.0"
    m_str = m_fmt.format(mean_v)
    
    if std_v is not None:
        s_str = s_fmt.format(std_v)
        return f"{m_str}$\\pm${s_str}"
    else:
        return m_str

# ---------- 主表构建 ----------
def build_table():
    summaries = [load_summary(p) for p in LOG_DIRS]

    # 提取所有数值 (rows × cols)，存储为 (mean, std) tuple
    mat = []
    for s in summaries:
        row_vals = []
        for k in ALL_KEYS:
            raw = get_data(s, k)
            processed = fmt_val_tuple(k, raw)
            row_vals.append(processed)
        mat.append(row_vals)
    
    # 计算最优和次优索引 (基于 mean 值)
    best_idx, second_idx = {}, {}
    for j, key in enumerate(ALL_KEYS):
        if key not in MARK_KEYS:
            continue
        
        # 提取该列所有的 mean 值用于排序
        means = []
        indices = []
        for i in range(len(mat)):
            m_val = mat[i][j][0] # mean
            if m_val is not None:
                means.append(m_val)
                indices.append(i)
        
        if not means:
            continue
            
        means = np.array(means)
        indices = np.array(indices)
        
        if key in ROT_KEYS or key in TRANS_KEYS: # 越小越好
            sorted_args = np.argsort(means)
        else: # Success Rate 越大越好
            sorted_args = np.argsort(-means)
            
        best_i = indices[sorted_args[0]]
        best_idx[key] = best_i
        
        if len(sorted_args) > 1:
            # 简单去重逻辑：如果第二名和第一名数值极其接近，可能需要处理，这里简化为直接取第二名
            second_i = indices[sorted_args[1]]
            # 只有当数值确实不同时才标记(可选)，这里直接标记
            second_idx[key] = second_i

    # 开始写表格
    L = [
        r"\begin{table*}[htbp]",
        r"\centering",
        r"\caption{Ablation on KITTI at $10^\circ\,50\mathrm{cm}$ (r10\_t0.5).}",
        r"\label{Table.ablation}",
        r"\resizebox{\textwidth}{!}{",  # 防止表格过宽
        rf"\begin{{tabular}}{{{COLS_SPEC}}}",
        r"\toprule",
        # 表头第 1 行
        r"\multirow{2}{*}{Index} & "
        r"\multirow{2}{*}{\makecell{Dual \\ Branches}} & "
        r"\multirow{2}{*}{\makecell{Projection \\ Margin}} & "
        r"\multirow{2}{*}{\makecell{Encoding \\ Space}} & "
        r"\multirow{2}{*}{\makecell{Positional \\ Embedding}} & "
        r"\multirow{2}{*}{\makecell{Image \\ Encoder}} & "
        r"\multirow{2}{*}{\makecell{Rotation \\ RMSE ($^\circ$)$\downarrow$}} & "
        r"\multirow{2}{*}{\makecell{Translation \\ RMSE (cm)$\downarrow$}} & "
        r"\multicolumn{2}{c}{Success Rate (\%)$\uparrow$} \\",
        # 表头第 2 行
        r"& & & & & & & & $L_1$ & $L_2$ \\",
        r"\midrule",
    ]

    for i, (path, opts) in enumerate(zip(LOG_DIRS, OPTION_ROWS)):
        opts = list(opts)  # [dual, pm, pe, enc]
        cells = []
        for j, key in enumerate(ALL_KEYS):
            val_tuple = mat[i][j]
            cell_str = fmt_cell(key, val_tuple)
            
            # 加粗/下划线处理
            if val_tuple[0] is not None and cell_str != "--" and key in MARK_KEYS:
                if best_idx.get(key) == i:
                    cell_str = r"\textbf{" + cell_str + "}"
                elif second_idx.get(key) == i:
                    cell_str = r"\underline{" + cell_str + "}"
            cells.append(cell_str)
        index_str = str(i + 1)
        L.append(" & ".join([index_str] + opts + cells) + r" \\")

    L += [
        r"\bottomrule",
        r"\end{tabular}",
        r"}", # end resizebox
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