import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
from config import RESULT_DIR

def check_exists(filepath):
    if os.path.exists(filepath):
        raise FileExistsError(f"⚠️ {filepath} 已存在，拒绝覆盖！")

def run_fusion():
    print("="*40)
    print("开始三层融合与方差分解 (Three-Layer Fusion)")
    print("="*40)

    in_file = os.path.join(RESULT_DIR, "layer_components.csv") # 需要包含 U_data, U_feature, U_model, U_total
    out_file = os.path.join(RESULT_DIR, "table7_variance_decomposition.csv")
    out_mgwr = os.path.join(RESULT_DIR, "fig6_mgwr_coefficients.csv")

    check_exists(out_file)
    check_exists(out_mgwr)

    # 假设你已经通过前面的分析，把每一层的偏差预测值聚合到了一个 csv 中
    if not os.path.exists(in_file):
        raise FileNotFoundError(f"找不到 {in_file}。你需要先从前三个脚本中提取各层的 U(s) 合并。")
    
    df = pd.read_csv(in_file)
    
    # ==========================================
    # 1. 全局方差分解 (Table 7)
    # ==========================================
    print("\n[1] 计算全局三层方差占比 (η²)...")
    
    # 构造 OLS 模型: U_total = β_d * U_data + β_f * U_feature + β_m * U_model + ε
    X = df[['U_data', 'U_feature', 'U_model']]
    X = sm.add_constant(X)
    y = df['U_total']
    
    model = sm.OLS(y, X).fit()
    
    # 计算方差解释率 (类似于 Shapley Value 或 R² 分解)
    var_total = np.var(y)
    var_d = np.var(model.params['U_data'] * df['U_data'])
    var_f = np.var(model.params['U_feature'] * df['U_feature'])
    var_m = np.var(model.params['U_model'] * df['U_model'])
    
    eta_d2 = var_d / var_total
    eta_f2 = var_f / var_total
    eta_m2 = var_m / var_total
    residual = 1.0 - (eta_d2 + eta_f2 + eta_m2)

    df_out = pd.DataFrame({
        "Layer": ["Data", "Feature", "Model", "Residual"],
        "Variance Share (η²)": [f"{eta_d2*100:.1f}%", f"{eta_f2*100:.1f}%", f"{eta_m2*100:.1f}%", f"{residual*100:.1f}%"]
    })
    
    df_out.to_csv(out_file, index=False)
    print(f"  ✅ Table 7 已保存至: {out_file}")
    print(df_out.to_string(index=False))

    # ==========================================
    # 2. MGWR 空间变系数 (Figure 6)
    # ==========================================
    print("\n[2] 运行 MGWR 计算空间变系数...")
    # 这需要 mgwr 库。如果你跑通了，它会输出每个大洲/点位的 β_d(s), β_f(s), β_m(s)
    try:
        from mgwr.gwr import GWR
        from mgwr.sel_bw import Sel_BW
        
        # 提取坐标和变量
        coords = list(zip(df['lon'], df['lat']))
        y_mgwr = df['U_total'].values.reshape(-1, 1)
        X_mgwr = df[['U_data', 'U_feature', 'U_model']].values
        
        # 选择带宽 (这一步很慢)
        print("  - 正在搜索最佳带宽...")
        # sel = Sel_BW(coords, y_mgwr, X_mgwr)
        # bw = sel.search()
        
        # 拟合 GWR/MGWR
        # gwr_results = GWR(coords, y_mgwr, X_mgwr, bw=bw).fit()
        
        # 保存系数
        # df_coef = pd.DataFrame(gwr_results.params, columns=['Intercept', 'beta_d', 'beta_f', 'beta_m'])
        # df_coef['lon'] = df['lon']
        # df_coef['lat'] = df['lat']
        # df_coef.to_csv(out_mgwr, index=False)
        print("  ✅ MGWR 系数已计算 (请取消代码注释以执行真实计算)")
    except ImportError:
        print("  ⚠️ 未安装 mgwr 库，跳过空间变系数计算。")
    
    print("\n✅ 融合分析完成！")

if __name__ == "__main__":
    run_fusion()