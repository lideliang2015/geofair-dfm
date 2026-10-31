import os
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_predict
from config import RESULT_DIR

def run_causal_inference():
    print("="*40)
    print("开始因果分离与反事实分析 (Causal Separation)")
    print("="*40)

    in_file = os.path.join(RESULT_DIR, "baseline_metrics.csv")
    out_ate_file = os.path.join(RESULT_DIR, "table8_ate_estimates.csv")
    out_cate_file = os.path.join(RESULT_DIR, "fig7_cate_forest_plot.csv")

    if os.path.exists(out_ate_file) or os.path.exists(out_cate_file):
        raise FileExistsError("⚠️ 结果文件已存在，请先备份或重命名，拒绝覆盖！")

    df = pd.read_csv(in_file)
    
    # 1. 定义处理变量 (T)
    # 根据论文设定，欠发达地区 (非洲、南美) T=1，发达/发展中 (欧洲、北美、大洋洲) T=0
    # 亚洲 (Asia) 因为是零样本，我们把它剔除出因果分析，避免极端值干扰
    df_causal = df[df['continent'] != 'Asia'].copy()
    
    df_causal['T'] = df_causal['continent'].isin(['Africa', 'South America']).astype(int)
    df_causal['Y'] = df_causal['IoU']
    
    # 2. 协变量 (X)：客观地理难度
    # 我们用 building_density 作为控制变量 (D(s))
    X_cols = ['building_density']
    X = df_causal[X_cols].values
    
    Y = df_causal['Y'].values
    T = df_causal['T'].values
    
    print(f"参与因果分析的样本数: {len(df_causal)}")

    # ==========================================
    # 1. 计算未调整的 ATE (无控制变量)
    # ==========================================
    unadjusted_ate = np.mean(Y[T == 1]) - np.mean(Y[T == 0])
    print(f"\n[1] 未调整 ATE (Unadjusted ATE): {unadjusted_ate:.4f}")

    # ==========================================
    # 2. 计算调整后的 ATE (控制建筑密度)
    # 我们使用 RandomForest 拟合 Y ~ T + X，然后通过 g-computation 估算 ATE
    # ==========================================
    print("[2] 正在训练随机森林进行因果推断 (G-computation)...")
    
    # 构建特征矩阵
    feature_matrix = np.column_stack((T, X))
    
    # 用随机森林预测 IoU
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(feature_matrix, Y)
    
    # 构造反事实：把所有样本都假设为 T=1 和 T=0
    T1_features = np.column_stack((np.ones(len(T)), X))
    T0_features = np.column_stack((np.zeros(len(T)), X))
    
    Y1_pred = rf.predict(T1_features)
    Y0_pred = rf.predict(T0_features)
    
    # 调整后的 ATE
    adjusted_ate = np.mean(Y1_pred - Y0_pred)
    print(f"    调整后 ATE (Adjusted ATE): {adjusted_ate:.4f}")

    # 计算算法偏见比例 ABR
    # ABR = 调整后ATE / 未调整ATE
    abr = adjusted_ate / unadjusted_ate if unadjusted_ate != 0 else 0
    print(f"    算法偏见比例 (ABR): {abr:.2f} ({abr*100:.1f}%)")

    # 保存 Table 8
    df_ate = pd.DataFrame({
        "Estimate": ["Unadjusted ATE", "Adjusted ATE", "Algorithmic Bias Ratio (ABR)"],
        "Value": [f"{unadjusted_ate:.4f}", f"{adjusted_ate:.4f}", f"{abr:.2f}"]
    })
    df_ate.to_csv(out_ate_file, index=False)
    print(f"  ✅ Table 8 已保存至: {out_ate_file}")

    # ==========================================
    # 3. 计算 CATE (异质性处理效应)
    # ==========================================
    print("\n[3] 计算各区域 CATE...")
    # 我们按大洲分组，分别计算每个大洲的调整后偏差
    cate_results = []
    for cont in df_causal['continent'].unique():
        idx = df_causal['continent'] == cont
        X_cont = X[idx]
        
        # 假设该洲处于 T=1 和 T=0 的预测差异
        T1_cont = np.column_stack((np.ones(len(X_cont)), X_cont))
        T0_cont = np.column_stack((np.zeros(len(X_cont)), X_cont))
        
        cate = np.mean(rf.predict(T1_cont) - rf.predict(T0_cont))
        cate_results.append({"Continent": cont, "CATE": cate})

    df_cate = pd.DataFrame(cate_results)
    df_cate.to_csv(out_cate_file, index=False)
    print(f"  ✅ Figure 7 数据已保存至: {out_cate_file}")
    print(df_cate.to_string(index=False))

if __name__ == "__main__":
    run_causal_inference()
