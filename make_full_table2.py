import argparse, os, json, math

# ---- 列定义：Dataset | Range | Method | (RRMSE, RMAE | tRMSE, tMAE | L1, L2) = 9列
# 格式：Dataset, Range, Method (lll) | Rot (cc) | Trans (cc) | Success (cc)
COLS_SPEC = r"lllcc|cc|cc"

# 定义需要提取的 Key
# metrics.py 生成的 JSON 中，RMSE/MAE 是字典 {'mean':..., 'std':...}
# success_rate 是嵌套字典 {'L1':..., 'L2':...}

METRIC_KEYS = ["RRMSE", "RMAE", "tRMSE", "tMAE"]
ACC_KEYS = ["L1", "L2"]

ALL_KEYS = METRIC_KEYS + ACC_KEYS
# 需要高亮的指标 (全部)
HIGHLIGHT_KEYS = ALL_KEYS

# 定义指标方向：False表示越低越好(误差)，True表示越高越好(成功率)
HIGHER_IS_BETTER = {k: False for k in METRIC_KEYS}
HIGHER_IS_BETTER.update({k: True for k in ACC_KEYS})

DATASETS = [
    ("KITTI~\\cite{KITTI}", "kitti"),
    ("nuScenes~\\cite{nuScenes}", "nusc"),
]

# 默认方法名配置 (保持原样)
DEFAULT_METHODS = [
   "corfil2p", "directcalib","calibanything", "calibnet", "rggnet", "lccnet", "lccraft_large", "calibdepth", "projdualfusion_harmonic"
]
DEFAULT_METHOD_NAMES = [
   "CoFiI2P", "DirectCalib", "CalibAnything", "CalibNet", "RGGNet", "LCCNet", "LCCRAFT", "CalibDepth",  "Ours"
]

METHOD_CITES = {
    "calibnet": "CalibNet",
    "rggnet": "RGGNet",
    "lccnet": "LCCNet",
    "lccraft_large": "LCCRAFT",
    "calibdepth": "CalibDepth",
    "corfil2p": "CoFiI2P",
    "directcalib": "DirectCalib",
    "calibanything": "CalibAnything",
}

SCENARIOS = ["r15_t0.15", "r10_t0.25", "r10_t0.5"] 

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
        # metrics.py 现在直接 dump 字典，不是 list，但也可能是 list 兼容旧版
        return data[-1] if isinstance(data, list) else data
    except Exception as e:
        print(f"[WARN] bad json: {p} ({e})"); return None

def get_data_pair(summary, key):
    """
    从 JSON 中提取数据。
    返回 tuple: (mean_value, std_value)
    对于 L1/L2，std_value 为 None。
    此处统一进行单位换算（m -> cm, ratio -> % 待显示时处理）。
    """
    if summary is None: 
        return (None, None)
    
    # 处理 Success Rate (L1, L2)
    if key in ACC_KEYS:
        # 结构: summary['success_rate']['L1'] -> float
        sr = summary.get("success_rate", {})
        val = sr.get(key, None)
        return (float(val), None) if val is not None else (None, None)

    # 处理 Rotation / Translation Metrics (RMSE, MAE)
    # 结构: summary[key]['mean'] / summary[key]['std']
    item = summary.get(key, {})
    if not isinstance(item, dict):
        return (None, None)
    
    mean_val = item.get('mean', None)
    std_val = item.get('std', None)

    if mean_val is None:
        return (None, None)

    mean_val = float(mean_val)
    std_val = float(std_val) if std_val is not None else 0.0

    # 如果是平移指标 (tx, ty, tz, tRMSE, tMAE)，将 m 转换为 cm
    if key.startswith('t'):
        mean_val *= 100.0
        std_val *= 100.0

    return (mean_val, std_val)

def fmt_display(key, mean_val, std_val):
    """
    格式化显示字符串。
    逻辑：
    1. Success Rate (L1, L2): 固定显示百分比后两位小数 (xx.xx%)。
    2. Metrics (RMSE/MAE): 
       - Mean < 10   -> 3位小数 (e.g., 4.112)
       - 10 <= Mean < 100 -> 2位小数 (e.g., 12.34)
       - Mean >= 100 -> 1位小数 (e.g., 113.2)
       Std 的小数位数跟随 Mean 保持一致。
    """
    def cal_fmt(val):
        abs_m = abs(val)
        if abs_m < 10:
            prec = 3
        elif abs_m < 100:
            prec = 2
        elif abs_m < 1000:
            prec = 1
        else:
            prec = 0
            
        # 构建格式化字符串，例如 "{:.3f}"
        return f"{{:.{prec}f}}"
    if mean_val is None:
        return "--"
    
    # Success Rate: 显示百分比，无方差
    if key in ACC_KEYS:
        return f"{100.0*mean_val:.2f}\\%"
    
    # Metrics: 自适应精度
    mean_fmt = cal_fmt(mean_val)
    std_fmt = cal_fmt(std_val)
    # 构建格式化字符串，例如 "{:.3f}"
    m_str = mean_fmt.format(mean_val)
    # 标准差使用相同的精度，以保证 mean ± std 的对齐
    s_str = std_fmt.format(std_val)
    
    return f"{m_str}$\\pm${s_str}"

def decorate(text, mark):
    if mark == "best":
        return r"\textbf{" + text + "}"
    if mark == "second":
        return r"\underline{" + text + "}"
    return text

def rank_marks(data_pairs, higher_is_better):
    """
    根据 Mean 值进行排序打标。
    data_pairs: list of (mean, std)
    """
    marks = [None] * len(data_pairs)
    # 提取 (index, mean_value) 用于排序，忽略 None
    valid_pairs = []
    for i, (m, s) in enumerate(data_pairs):
        if m is not None and not math.isnan(m):
            valid_pairs.append((i, m))
    
    if not valid_pairs: return marks

    # 排序
    valid_pairs.sort(key=lambda x: x[1], reverse=higher_is_better)

    # Best
    best_idx = valid_pairs[0][0]
    best_val = valid_pairs[0][1]
    marks[best_idx] = "best"

    # Second
    if len(valid_pairs) >= 2:
        for i, val in valid_pairs[1:]:
            # 简单判断浮点数不等，避免相等值也被标记为 second
            if not math.isclose(val, best_val, rel_tol=1e-12, abs_tol=1e-12):
                marks[i] = "second"
                break
    return marks

def build_table(root, methods, method_names):
    L = []
    L += [
        r"\begin{table*}[htbp]",
        r"\centering",
        r"\caption{Calibration Results on KITTI and nuScenes Dataset (Mean $\pm$ Std).}",
        r"\label{Table.calib_metrics}",
        # 调整字体大小如果表格太宽
        r"\resizebox{\textwidth}{!}{", 
        rf"\begin{{tabular}}{{{COLS_SPEC}}}",
        r"\toprule",
        # 表头第一行
        r"\multirow{2}*{Dataset} & \multirow{2}*{Range} & \multirow{2}*{Method} & "
        r"\multicolumn{2}{c}{Rotation ($^\circ$)$\downarrow$} & "
        r"\multicolumn{2}{c}{Translation (cm)$\downarrow$} & "
        r"\multicolumn{2}{c}{Success Rate (\%)$\uparrow$} \\",
        # 表头第二行
        r"~ & ~ & ~ & RMSE & MAE & RMSE & MAE & $L_1$ & $L_2$ \\",
        r"\midrule",
    ]

    for ds_caption, ds_prefix in DATASETS:
        total_rows = len(SCENARIOS) * len(methods)
        first_ds_row = True

        for si, sc in enumerate(SCENARIOS):
            summaries = []
            for method in methods:
                folder = f"{ds_prefix}_{sc}"
                # 假设 metric.py 生成的文件名格式依然是 method_sc.json，或者 log_file 参数指定的路径
                # 这里沿用原逻辑：root/folder/method_sc.json
                path = os.path.join(root, folder, f"{method}_{sc}.json")
                summaries.append(load_summary(path))

            # 预计算每一列的数据和排名
            # row_data[method_index][key] = (mean, std)
            row_data = [] 
            for m_idx in range(len(methods)):
                row_data.append({})
                for k in ALL_KEYS:
                    row_data[m_idx][k] = get_data_pair(summaries[m_idx], k)

            # 计算排名标记
            marks_map = {} # key -> [mark_for_method_0, mark_for_method_1, ...]
            for k in HIGHLIGHT_KEYS:
                # 提取所有方法的该指标均值列表用于排序
                vals = [row_data[i][k] for i in range(len(methods))] # list of (mean, std)
                marks_map[k] = rank_marks(vals, HIGHER_IS_BETTER[k])

            range_title = scenario_to_title(sc)
            first_range_row = True

            for m_idx, (method, mname) in enumerate(zip(methods, method_names)):
                # 处理合并单元格
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

                # 处理引用
                disp_name = mname
                if ds_prefix == "kitti" and si == 0:
                    cite_key = METHOD_CITES.get(method)
                    if cite_key:
                        disp_name = f"{mname}\\cite{{{cite_key}}}"

                # 拼接一行的数据
                cells = []
                for k in ALL_KEYS:
                    mean_v, std_v = row_data[m_idx][k]
                    disp = fmt_display(k, mean_v, std_v)
                    mark = marks_map[k][m_idx]
                    cells.append(decorate(disp, mark))

                L.append(left + f" {mid} & {disp_name} & " + " & ".join(cells) + r" \\")
            
            # Scenario 之间的分隔线
            if si < len(SCENARIOS) - 1:
                L.append(r"\cdashline{3-9}[1pt/1pt]") # 调整列索引 3-9

        L.append(r"\midrule")

    if L[-1] == r"\midrule":
        L.pop()

    L += [
        r"\bottomrule",
        r"\end{tabular}",
        r"}", # End resizebox
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

if __name__ == "__main__":
    main()