import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (mean_absolute_error, r2_score,
                             root_mean_squared_error)

v = pd.read_csv("ventas_diarias.csv", parse_dates=["fecha"],
                index_col="fecha")["ventas"] / 1000  # miles de £

def variables(v):
    """Lo que se sabe la víspera: ventas pasadas y el calendario."""
    X = pd.DataFrame({"ayer": v.shift(1), "hace_7": v.shift(7),
                      "media_7": v.shift(1).rolling(7).mean(),
                      "media_28": v.shift(1).rolling(28).mean()})
    dias = pd.get_dummies(v.index.dayofweek, prefix="dia",
                          dtype=float).set_index(v.index)
    return X.join(dias).dropna()

X = variables(v)
y = v.loc[X.index]
ent, pru = X.index < "2011-07-01", X.index >= "2011-07-01"
print(f"entrenamiento {ent.sum()} días, prueba {pru.sum()} días")

def mide(nombre, pred):
    yp = y[pru]
    print(f"{nombre:<26} EAM {mean_absolute_error(yp, pred):5.2f}  "
          f"RECM {root_mean_squared_error(yp, pred):5.2f}  "
          f"R2 {r2_score(yp, pred):5.2f}")

if __name__ == "__main__":
    mide("referencia: hace 7 días", X.loc[pru, "hace_7"])
    lineal = LinearRegression().fit(X[ent], y[ent])
    mide("regresión lineal", lineal.predict(X[pru]))
