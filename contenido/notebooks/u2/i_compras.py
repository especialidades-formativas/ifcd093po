import time
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

d = pd.read_csv("compras.csv")                 # 12 330 visitas
y = d.pop("Revenue").astype(int)               # 1 = acabó en compra
cat = ["Month", "VisitorType", "Weekend", "OperatingSystems",
       "Browser", "Region", "TrafficType"]
num = d.columns.difference(cat)

def prep(escalar):              # los árboles no necesitan escalar
    return make_column_transformer(
        (OneHotEncoder(handle_unknown="ignore", sparse_output=False),
         cat),
        (StandardScaler() if escalar else "passthrough", num),
        verbose_feature_names_out=False)

X_ent, X_pru, y_ent, y_pru = train_test_split(
    d, y, test_size=0.3, stratify=y, random_state=0)

def evalua(nombre, mod):
    """Entrena y muestra el acierto equilibrado y el AUC en prueba."""
    t = time.perf_counter()
    mod.fit(X_ent, y_ent)
    if hasattr(mod, "decision_function"):            # la SVM
        s = mod.decision_function(X_pru)
    else:                                            # los demás
        s = mod.predict_proba(X_pru)[:, 1]
    eq = balanced_accuracy_score(y_pru, mod.predict(X_pru))
    print(f"{nombre:<22} entrenamiento {mod.score(X_ent, y_ent):.3f}"
          f" ({time.perf_counter() - t:.0f} s)")
    print(f"   prueba: acierto equilibrado {eq:.3f}, "
          f"AUC {roc_auc_score(y_pru, s):.3f}")
    return mod

if __name__ == "__main__":
    print(f"compran {y.mean():.1%} de las visitas")
    arbol = evalua("árbol, profundidad 3", make_pipeline(
        prep(False), DecisionTreeClassifier(max_depth=3,
                                            random_state=0)))
    evalua("árbol, sin límite", make_pipeline(
        prep(False), DecisionTreeClassifier(random_state=0)))
    evalua("SVM (núcleo RBF)", make_pipeline(prep(True), SVC()))
    evalua("red neuronal 32-16", make_pipeline(
        prep(True), MLPClassifier((32, 16), early_stopping=True,
                                  max_iter=500, random_state=0)))
    plt.figure(figsize=(5.5, 2.4))
    plot_tree(arbol[-1], max_depth=2, fontsize=5,
              feature_names=arbol[0].get_feature_names_out(),
              class_names=["no compra", "compra"], filled=True,
              impurity=False, proportion=True)
    plt.savefig("arbol.pdf", bbox_inches="tight")
