"""Comparativo de modelos sobre el dataset SmartDesk (LinearRegression, RandomForest, GradientBoosting y, si esta instalado, XGBoost)."""
import re, time, unicodedata, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.metrics import mean_absolute_error, median_absolute_error, r2_score, f1_score, accuracy_score, roc_auc_score, average_precision_score, recall_score, precision_score
try:
    from xgboost import XGBRegressor, XGBClassifier; HAY_XGB = True
except ImportError:
    HAY_XGB = False

def norm(t):
    t = unicodedata.normalize("NFD", str(t).lower()); t = "".join(c for c in t if unicodedata.category(c) != "Mn")
    return re.sub(r"[^a-z0-9]+", " ", t).strip()

from pathlib import Path
BASE = Path(__file__).resolve().parent
RUTA = next((p for p in (BASE / "data" / "tickets_smartdesk_es.csv", BASE / "tickets_smartdesk_es.csv") if p.exists()), None)
if RUTA is None:
    raise SystemExit("No encuentro tickets_smartdesk_es.csv: ponlo en la carpeta data/ o junto a este script.")
df = pd.read_csv(RUTA)
df["texto"] = (df.titulo.fillna("") + " " + df.descripcion_falla.fillna("")).map(norm)
df["sistema_operativo"] = df.sistema_operativo.fillna("desconocido")
CAT = ["canal", "sede", "departamento_solicitante", "sistema_operativo"]
CAT_POST = CAT + ["subcategoria", "criticidad"]   # disponibles despues del triaje

idx = np.arange(len(df))
tr, te = train_test_split(idx, test_size=0.2, random_state=42, stratify=df.criticidad)
A, B = df.iloc[tr], df.iloc[te]

def features(cols_cat, extra_num):
    tf = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    Ta, Tb = tf.fit_transform(A.texto), tf.transform(B.texto)
    svd = TruncatedSVD(60, random_state=0); Sa, Sb = svd.fit_transform(Ta), svd.transform(Tb)
    oh = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    Oa, Ob = oh.fit_transform(A[cols_cat]), oh.transform(B[cols_cat])
    Na = np.log1p(A[extra_num].values.astype(float)); Nb = np.log1p(B[extra_num].values.astype(float))
    return np.hstack([Sa, Oa, Na]), np.hstack([Sb, Ob, Nb]), Ta, Tb

# ---------- 1) REGRESION: tiempo de resolucion (minutos) ----------
print("\n=== 1) Regresion: tiempo de resolucion (minutos) ===")
Xa, Xb, *_ = features(CAT_POST, ["usuarios_afectados"])
ya, yb = np.log1p(A.tiempo_resolucion_min.values), B.tiempo_resolucion_min.values
base = np.median(A.tiempo_resolucion_min)
print(f"{'Modelo':22s} {'MAE':>8s} {'MedAE':>7s} {'R2(log)':>8s} {'seg':>6s}")
print(f"{'Mediana global':22s} {np.mean(np.abs(yb-base)):8.1f} {np.median(np.abs(yb-base)):7.1f} {'-':>8s} {'-':>6s}")
regs = {"Regresion Lineal": LinearRegression(),
        "Random Forest": RandomForestRegressor(300, min_samples_leaf=3, random_state=0, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=0)}
if HAY_XGB: regs["XGBoost"] = XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, random_state=0)
imp = {}
for n, m in regs.items():
    t = time.time(); m.fit(Xa, ya); p = np.expm1(m.predict(Xb)).clip(1)
    print(f"{n:22s} {mean_absolute_error(yb,p):8.1f} {median_absolute_error(yb,p):7.1f} {r2_score(np.log1p(yb),np.log1p(p)):8.3f} {time.time()-t:6.1f}")
    if hasattr(m, "feature_importances_"): imp[n] = m.feature_importances_

# ---------- 2) CLASIFICACION: criticidad ----------
print("\n=== 2) Clasificacion: criticidad (Baja/Media/Alta/Critica) ===")
Xa2, Xb2, Ta, Tb = features(CAT, ["usuarios_afectados"])
ya2, yb2 = A.criticidad.values, B.criticidad.values
print(f"{'Modelo':32s} {'Acc':>6s} {'F1 macro':>9s} {'Recall Crit.':>13s} {'seg':>6s}")
t = time.time(); lr = LogisticRegression(C=5, max_iter=500, class_weight="balanced").fit(Ta, ya2); p = lr.predict(Tb)
print(f"{'Log. (solo texto, actual)':32s} {accuracy_score(yb2,p):6.3f} {f1_score(yb2,p,average='macro'):9.3f} {recall_score(yb2,p,labels=['Crítica'],average=None)[0]:13.3f} {time.time()-t:6.1f}")
clfs = {"Random Forest": RandomForestClassifier(300, min_samples_leaf=2, class_weight="balanced", random_state=0, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=150, max_depth=3, learning_rate=0.1, random_state=0)}
if HAY_XGB: clfs["XGBoost"] = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1, subsample=0.8, random_state=0)
cls = sorted(set(ya2)); enc = {c: i for i, c in enumerate(cls)}
for n, m in clfs.items():
    t = time.time()
    if n == "XGBoost": m.fit(Xa2, [enc[c] for c in ya2]); p = np.array([cls[i] for i in m.predict(Xb2)])
    else: m.fit(Xa2, ya2); p = m.predict(Xb2)
    print(f"{n+' (texto+datos)':32s} {accuracy_score(yb2,p):6.3f} {f1_score(yb2,p,average='macro'):9.3f} {recall_score(yb2,p,labels=['Crítica'],average=None)[0]:13.3f} {time.time()-t:6.1f}")

# ---------- 3) CLASIFICACION: riesgo de incumplir SLA ----------
print("\n=== 3) Riesgo de incumplir SLA (clase positiva = incumple, 5.5%) ===")
Xa3, Xb3, *_ = features(CAT_POST, ["usuarios_afectados"])
ya3, yb3 = (~A.cumple_sla).astype(int).values, (~B.cumple_sla).astype(int).values
print(f"{'Modelo':22s} {'ROC-AUC':>8s} {'PR-AUC':>7s} {'Recall':>7s} {'Prec.':>6s} {'seg':>6s}")
c3 = {"Reg. Logistica (ref.)": LogisticRegression(max_iter=1000, class_weight="balanced"),
      "Random Forest": RandomForestClassifier(300, min_samples_leaf=3, class_weight="balanced", random_state=0, n_jobs=-1),
      "Gradient Boosting": GradientBoostingClassifier(n_estimators=150, max_depth=3, learning_rate=0.1, random_state=0)}
if HAY_XGB: c3["XGBoost"] = XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.1, subsample=0.8, scale_pos_weight=(ya3==0).sum()/ya3.sum(), random_state=0)
for n, m in c3.items():
    t = time.time(); m.fit(Xa3, ya3); pr = m.predict_proba(Xb3)[:, 1]
    thr = 0.5 if n != "Gradient Boosting" else np.quantile(pr, 1 - 2*yb3.mean())  # umbral ajustado (GB sin pesos)
    p = (pr >= thr).astype(int)
    print(f"{n:22s} {roc_auc_score(yb3,pr):8.3f} {average_precision_score(yb3,pr):7.3f} {recall_score(yb3,p):7.3f} {precision_score(yb3,p,zero_division=0):6.3f} {time.time()-t:6.1f}")
print("\nXGBoost instalado:", HAY_XGB, "| positivos en test:", int(yb3.sum()), "de", len(yb3))
