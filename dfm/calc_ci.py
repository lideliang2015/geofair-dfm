import os
import numpy as np
import pandas as pd
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.ensemble import RandomForestRegressor
from config import RESULT_DIR

def run_bootstrap_ci():
    print("="*40)
    print("开始计算 Bootstrap 95% 置信区间 (1000次)")
    print("="*40)
    
    in_file = os.path.join(RESULT_DIR, "baseline_metrics.csv")
    df = pd.read_csv(in_file)
    
    # 剔除亚洲 (无坐标，IoU 异常)
    df = df[df['continent'] != 'Asia'].copy()
    
    # 定义变量
    df['T'] = df['continent'].isin(['Africa', 'South America']).astype(int)
    df['Y'] = df['IoU']
    X = df[['building_density']].values
    
    Y = df['Y'].values
    T = df['T'].values
    
    n_bootstraps = 1000
    boot_unadj = []
    boot_adj = []
    
    print("正在重采样 (可能需要 1-2 分钟)...")
    for i in range(n_bootstraps):
        # 有放回地重采样索引
        idx = np.random.choice(len(Y), len(Y), replace=True)
        Y_b, T_b, X_b = Y[idx], T[idx], X[idx]
        
        # 1. 计算未调整的 ATE
        if np.sum(T_b == 1) > 0 and np.sum(T_b == 0) > 0:
            unadj_b = np.mean(Y_b[T_b == 1]) - np.mean(Y_b[T_b == 0])
            boot_unadj.append(unadj_b)
        
        # 2. 计算调整后的 ATE
        rf_b = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
        rf_b.fit(np.column_stack((T_b, X_b)), Y_b)
        
        T1_b = np.column_stack((np.ones(len(T_b)), X_b))
        T0_b = np.column_stack((np.zeros(len(T_b)), X_b))
        adj_b = np.mean(rf_b.predict(T1_b) - rf_b.predict(T0_b))
        boot_adj.append(adj_b)
        
        if (i+1) % 200 == 0:
            print(f"  已完成 {i+1} / 1000 次...")

    # 计算 2.5% 和 97.5% 分位数
    ci_unadj = np.percentile(boot_unadj, [2.5, 97.5])
    ci_adj = np.percentile(boot_adj, [2.5, 97.5])
    
    print("\n" + "="*40)
    print("✅ 计算完成！真实 95% 置信区间如下：")
    print(f"Unadjusted ATE 95% CI: [{ci_unadj[0]:.3f}, {ci_unadj[1]:.3f}]")
    print(f"Adjusted ATE 95% CI:   [{ci_adj[0]:.3f}, {ci_adj[1]:.3f}]")
    print("="*40)
    print("\n请把上面这两个区间填入论文的 Table 7 中！")

if __name__ == "__main__":
    run_bootstrap_ci()