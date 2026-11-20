import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier

d = pd.read_csv("corazon.csv")          # 303 pacientes de Cleveland
y = (d.pop("diagnostico") > 0).astype(int)   # 1 = enfermo
categoricas = ["sexo", "dolor", "glucosa", "ecg", "angina_esf",
               "pendiente", "talio"]
numericas = ["edad", "presion", "colesterol", "frec_max", "st_dep",
             "vasos"]

def preparacion():
    """Rellena los datos que faltan, codifica y escala."""
    cat = make_pipeline(SimpleImputer(strategy="most_frequent"),
                        OneHotEncoder(handle_unknown="ignore"))
    num = make_pipeline(SimpleImputer(strategy="median"),
                        StandardScaler())
    return make_column_transformer((cat, categoricas),
                                   (num, numericas))

X_ent, X_pru, y_ent, y_pru = train_test_split(
    d, y, test_size=0.3, stratify=y, random_state=0)

if __name__ == "__main__":
    print(d.shape, f"enfermos {y.mean():.1%}, datos que faltan "
          f"{d.isna().sum().sum()}")
    for nombre, m in [("referencia (clase mayoritaria)",
                       DummyClassifier()),
                      ("regresión logística", LogisticRegression())]:
        modelo = make_pipeline(preparacion(), m).fit(X_ent, y_ent)
        print(f"{nombre}: acierto en prueba "
              f"{modelo.score(X_pru, y_pru):.3f}")
