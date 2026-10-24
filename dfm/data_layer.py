import os
import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LinearRegression
import libpysal
from esda.moran import Moran_Lag
from config import RESULT_DIR

def calculate_spatial_lag_correlation(df, factor_col, target_col='IoU', k=5):
    """
    计算空间滞后相关性 (Spatial-lag correlation)
    基于 k-nearest neighbors (k=5) 构建空间权重矩阵
    """
    if 'lon' not in df.columns or 'lat' not in df.columns:
        print("⚠️ 缺少经纬度列，跳过空间滞后分析")
        return np.nan

    # 构建 KNN 空间权重矩阵
    coords = list(zip(df['lon'], df['lat']))
    w = libpysal.weights.KNN.from_array(coords, k=k)
    w.transform = 'r' # 行标准化

    # 计算空间滞后值
    lag_factor = libpysal.weights.lag_spatial(w, df[factor_col].values)
    lag_target = libpysal.weights.lag_spatial(w, df[target_col].values)

    # 计算滞后变量之间的相关系数
    r, _ = pearsonr(lag_factor, lag_target)
    return r

def run_data_layer_attribution():
    print("="*40)
    print("开始数据层归因分析 (Data-Layer Attribution)")
    print("="*40)

    metrics_file = os.path.join(RESULT_DIR, "baseline_metrics.csv")
    out_corr_file = os.path.join(RESULT_DIR, "table3_data_correlations.csv")
    out_sampling_file = os.path.join(RESULT_DIR, "table4_stratified_sampling.csv")
    
    # 防覆盖保护
    if os.path.exists(out_corr_file) or os.path.exists(out_sampling_file):
        raise FileExistsError(f"⚠️ 结果文件已存在于 {RESULT_DIR}，请先备份或重命名，拒绝覆盖！")

    if not os.path.exists(metrics_file):
        raise FileNotFoundError(f"找不到 {metrics_file}，请先运行 01_inference.py")

    df = pd.read_csv(metrics_file)
    
    # ==========================================
    # 1. 相关性分析 (Table 3)
    # ==========================================
    print("\n[1] 正在计算 Table 3: 数据层因素与 IoU 的相关性...")
    
    # 假设你的 csv 里有以下列（你需要根据实际情况修改列名）
    factors = ['training_density', 'geo_coverage', 'resolution', 'label_quality', 'building_density']
    results = []

    for factor in factors:
        if factor not in df.columns:
            print(f"  - 警告: 列 '{factor}' 不存在，跳过...")
            continue
            
        # Pearson
        r_p, _ = pearsonr(df[factor], df['IoU'])
        # Spearman
        r_s, _ = spearmanr(df[factor], df['IoU'])
        # Spatial lag
        r_w = calculate_spatial_lag_correlation(df, factor)
        
        results.append({
            "Factor": factor,
            "Pearson r": round(r_p, 3),
            "Spearman ρ": round(r_s, 3),
            "Spatial-lag r": round(r_w, 3) if not np.isnan(r_w) else "N/A"
        })

    df_corr = pd.DataFrame(results)
    df_corr.to_csv(out_corr_file, index=False)
    print(f"  ✅ Table 3 已保存至: {out_corr_file}")
    print(df_corr.to_string(index=False))

    # ==========================================
    # 2. 计算数据层方差解释率 (η_d²) -> 对应 5.1.3
    # ==========================================
    print("\n[2] 正在计算数据层贡献度 η_d²...")
    # 假设 U_data(s) 可以用 training_density 和 geo_coverage 来代理
    # 这里的计算取决于你真实的残差 U(s)。此处演示使用训练密度预测 IoU 的 R² 作为代理。
    
    X = df[['training_density', 'geo_coverage']].dropna()
    y = df.loc[X.index, 'IoU']
    
    model = LinearRegression()
    model.fit(X, y)
    r2 = model.score(X, y)
    print(f"  ✅ 数据层解释方差占比 (η_d²): {r2:.3f} ({r2*100:.1f}%)")
    print("  *(注：这是基于回归 R² 的近似代理，最终论文需根据 OLS 融合模型确定)*")

    # ==========================================
    # 3. 分层采样结果统计 (Table 4)
    # ==========================================
    print("\n[3] 正在统计 Table 4: 各大洲 IoU (用于分层采样对比)...")
    
    # 按大洲计算 mean IoU
    df_continent = df.groupby('continent')['IoU'].mean().reset_index()
    df_continent = df_continent.sort_values(by='IoU', ascending=False)
    df_continent.columns = ['Region', 'Baseline IoU']
    
    # 如果你重训了平衡模型，把新的结果加载进来合并
    # balanced_metrics_file = os.path.join(RESULT_DIR, "balanced_baseline_metrics.csv")
    # if os.path.exists(balanced_metrics_file):
    #     df_bal = pd.read_csv(balanced_metrics_file).groupby('continent')['IoU'].mean().reset_index()
    #     df_continent['Balanced-training IoU'] = df_continent['Region'].map(df_bal.set_index('continent')['IoU'])
    #     df_continent['Δ IoU'] = df_continent['Balanced-training IoU'] - df_continent['Baseline IoU']
    
    df_continent.to_csv(out_sampling_file, index=False)
    print(f"  ✅ Table 4 已保存至: {out_sampling_file}")
    print(df_continent.to_string(index=False))

    print("\n✅ 数据层归因分析全部完成！")

if __name__ == "__main__":
    run_data_layer_attribution()