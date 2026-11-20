import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import (silhouette_score, davies_bouldin_score,
    calinski_harabasz_score, adjusted_rand_score)

m = pd.read_csv("mayoristas.csv")
gasto = m.drop(columns=["canal", "region"])
X = StandardScaler().fit_transform(np.log1p(gasto))

def agrupar(X, k, semilla=0):
    return KMeans(k, n_init=10, random_state=semilla).fit_predict(X)

if __name__ == "__main__":
    print(gasto.describe().round(0).T[["mean", "50%", "max"]])
    for k in range(2, 7):
        e = agrupar(X, k)
        print(k, round(silhouette_score(X, e), 3),
              round(davies_bouldin_score(X, e), 2),
              round(calinski_harabasz_score(X, e), 1))
    e2 = agrupar(X, 2)
    print(pd.crosstab(e2, m.canal))
    print("ARI con el canal:",
          round(adjusted_rand_score(m.canal, e2), 3))
    # el mismo k-means sin logaritmo
    Xb = StandardScaler().fit_transform(gasto)
    eb = agrupar(Xb, 2)
    print("sin log:", np.bincount(eb),
          round(silhouette_score(Xb, eb), 3),
          round(adjusted_rand_score(m.canal, eb), 3))
    # estabilidad: reagrupar submuestras del 80 %
    rng = np.random.default_rng(1)
    for k in (2, 3, 4):
        ref, ari = agrupar(X, k), []
        for _ in range(20):
            i = rng.choice(len(X), int(.8 * len(X)), replace=False)
            ari.append(adjusted_rand_score(
                ref[i], agrupar(X[i], k, semilla=7)))
        print(k, "estabilidad", round(np.mean(ari), 3),
              round(np.min(ari), 3))
