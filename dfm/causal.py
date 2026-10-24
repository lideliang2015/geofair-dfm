import os
import pandas as pd
import numpy as np
from config import RESULT_DIR

def check_exists(filepath):
    if os.path.exists(filepath):
        raise FileExistsError(f"⚠️ {filepath} 已存在，拒绝覆盖！")

def run_causal_inference():
    print("="*40)
    print("开始因果分离与反事实分析 (Causal Separation)")
    print("="*40)

    in_file = os.path.join(RESULT_DIR, "baseline_metrics.csv")
    out_ate_file = os.path.join(RESULT_DIR, "table8_ate_estimates.csv")
    out_cate_file = os.path.join(RESULT_DIR, "fig7_cate_forest_plot.csv")

    check_exists(out_ate_file)
    check_exists(out_cate_file)

    if not os.path.exists(in_file):
        raise FileNotFoundError(f"找不到 {in_file}")
    
    df = pd.read_csv(in_file)
    
    # 构造变量：
    # T: 是否属于历史欠发达地区 (如 South America, Africa = 1, Europe, N. America, Oceania = 0)
    # Y: IoU
    # X: 客观难度协变量 (building density, terrain roughness 等)
    
    T = df['continent'].isin(['S_ America', 'Africa']).astype(int).values
    Y = df['IoU'].values
    
    # 假设你的 csv 里有客观难度特征
    X_cols = ['building_density', 'terrain_roughness'] 
    if not all(col in df.columns for col in X_cols):
        print("⚠️ 缺少客观难度特征列，使用训练密度作为代理协变量...")
        X = df[['training_density']].values
    else:
        X = df[X_cols].values

    # ==========================================
    # 1. ATE 估计 (Table 8)
    # ==========================================
    print("\n[1] 正在使用因果森林估计 ATE...")
    
    # 未调整的 ATE
    unadjusted_ate = np.mean(Y[T == 1]) - np.mean(Y[T == 0])
    
    try:
        from econml.dml import CausalForestDML
        from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
        
        print("  - 拟合 Causal Forest (可能需要几分钟)...")
        est = CausalForestDML(
            model_y=RandomForestRegressor(n_estimators=100, min_samples_leaf=5),
            model_t=RandomForestClassifier(n_estimators=100, min_samples_leaf=5),
            discrete_treatment=True,
            random_state=42
        )
        est.fit(Y, T, X=X)
        
        # 计算 Adjusted ATE
        adjusted_ate = est.ate(X)
        cate = est.effect(X) # 条件平均处理效应
        
        # 计算 ABR (Algorithmic Bias Ratio)
        # 假设 未调整的 ATE 包含 100% 偏差，调整后的 ATE 是纯算法偏差
        abr = adjusted_ate / unadjusted_ate if unadjusted_ate != 0 else 0
        
        df_ate = pd.DataFrame({
            "Estimate": ["Unadjusted ATE", "Adjusted ATE", "Algorithmic bias ratio (ABR)"],
            "Value": [f"{unadjusted_ate:.3f}", f"{adjusted_ate:.3f}", f"{abr:.2f}"]
        })
        df_ate.to_csv(out_ate_file, index=False)
        print(f"  ✅ Table 8 已保存至: {out_ate_file}")
        print(df_ate.to_string(index=False))

        # ==========================================
        # 2. CATE 异质性 (Figure 7)
        # ==========================================
        print("\n[2] 输出 CATE 森林图数据...")
        df_cate = pd.DataFrame({
            "tile_id": df['tile_id'],
            "continent": df['continent'],
            "CATE": cate
        })
        df_cate.to_csv(out_cate_file, index=False)
        print(f"  ✅ Figure 7 原始数据已保存至: {out_cate_file}")

    except ImportError:
        print("⚠️ 未安装 econml 库，请先 pip install econml。")
    
    print("\n✅ 因果分析完成！请根据输出结果修改论文中的 60/40 比例。")

if __name__ == "__main__":
    run_causal_inference()