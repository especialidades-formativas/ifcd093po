import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score

c = pd.read_csv("clientes_rfm.csv", index_col="cliente")
Z = StandardScaler().fit_transform(np.log1p(c))

def perfil(etiquetas):
    p = c.groupby(etiquetas).median()
    p["clientes"] = pd.Series(etiquetas).value_counts()
    return p.round(0)

if __name__ == "__main__":
    km = KMeans(4, n_init=10, random_state=0).fit_predict(Z)
    print(perfil(km))
    je = AgglomerativeClustering(4, linkage="ward").fit_predict(Z)
    print(perfil(je))
    print("ARI k-medias/jerárquico:",
          round(adjusted_rand_score(km, je), 3))
    for eps in (0.2, 0.3, 0.5):
        db = DBSCAN(eps=eps, min_samples=10).fit_predict(Z)
        print(eps, "grupos", db.max() + 1,
              "ruido", (db == -1).sum())
    print(perfil(db))                    # el último: eps = 0,5
    print("una sola compra:", (c.frecuencia == 1).sum())
    for k in range(2, 9):
        g = GaussianMixture(k, n_init=3, random_state=0).fit(Z)
        print(k, "BIC", round(g.bic(Z)))
    import matplotlib.pyplot as plt
    from scipy.cluster.hierarchy import linkage, dendrogram
    L = linkage(Z, method="ward")
    plt.figure(figsize=(4.4, 2.0))
    dendrogram(L, truncate_mode="lastp", p=30, no_labels=True,
               color_threshold=L[-3, 2],  # corte: 4 grupos
               above_threshold_color="black")
    plt.ylabel("distancia de Ward"); plt.savefig("dendro.pdf")
