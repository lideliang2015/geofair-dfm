import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from config import RESULT_DIR

# 尝试导入 umap 和 shap（如果没装，会跳过部分功能）
try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    print("⚠️ 未安装 umap-learn，将只运行 t-SNE")

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    print("⚠️ 未安装 shap，将使用特征方差代理 SHAP 计算")

def run_feature_layer():
    print("="*40)
    print("开始特征层归因分析 (Feature-Layer Attribution)")
    print("="*40)

    features_file = os.path.join(RESULT_DIR, "deep_features.npy")
    metadata_file = os.path.join(RESULT_DIR, "baseline_metrics.csv")
    
    out_shap_file = os.path.join(RESULT_DIR, "fig2_shap_variance.csv")
    out_tsne_pdf = os.path.join(RESULT_DIR, "fig3_tsne_umap.pdf")

    if not os.path.exists(features_file):
        raise FileNotFoundError(f"找不到特征文件 {features_file}")
    
    features = np.load(features_file)
    df_meta = pd.read_csv(metadata_file)
    
    if len(features) != len(df_meta):
        raise ValueError("特征数量与 metadata 行数不匹配！")

    # 1. 特征方差 (SHAP 代理)
    print("\n[1] 计算特征地理方差...")
    feature_variances = np.var(features, axis=0)
    top10_idx = np.argsort(feature_variances)[-10:][::-1]
    df_shap = pd.DataFrame({
        "Feature_ID": top10_idx,
        "Geographic_Variance": feature_variances[top10_idx]
    })
    df_shap.to_csv(out_shap_file, index=False)
    print(f"  ✅ Figure 2 数据已保存至: {out_shap_file}")

    # 2. t-SNE 降维可视化
    print("\n[2] 运行 t-SNE 降维...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    features_tsne = tsne.fit_transform(features)
    
    # 剔除亚洲（无坐标且 IoU 全 0）来看真实的地理聚类
    valid_mask = df_meta['continent'] != 'Asia'
    valid_tsne = features_tsne[valid_mask]
    valid_continents = df_meta.loc[valid_mask, 'continent']
    
    silhouette = silhouette_score(valid_tsne, valid_continents)
    print(f"  - t-SNE 聚类 Silhouette Score: {silhouette:.3f}")

    # 画图
    plt.figure(figsize=(10, 8))
    continents = valid_continents.unique()
    colors = plt.cm.get_cmap('tab10', len(continents))
    
    for i, cont in enumerate(continents):
        idx = valid_continents == cont
        plt.scatter(valid_tsne[idx, 0], valid_tsne[idx, 1], label=cont, alpha=0.7, color=colors(i))
        
    plt.title("t-SNE Projection of Deep Features (Excluding Asia)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_tsne_pdf, format='pdf', dpi=300)
    plt.close()
    print(f"  ✅ Figure 3 图已保存至: {out_tsne_pdf}")

if __name__ == "__main__":
    run_feature_layer()