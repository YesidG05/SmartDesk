#!/usr/bin/env python3
"""
SmartDesk ML - Motor de recomendacion.

Dado el texto de un caso, devuelve:
  1. La falla probable (top 3 hipotesis con probabilidad) y su procedimiento.
  2. Criticidad (impacto x urgencia, matriz ITIL).
  3. Tiempo estimado de solucion (mediana y rango).
  4. Que agente (de los que haya en agentes.json) puede terminarlo antes,
     considerando turno, almuerzo, ausencias, cola actual y habilidades.

Aprende solo: al cerrar un ticket real (resolver) se guarda en
data/tickets_reales.csv y el modelo se reentrena. Los tickets reales pesan
mas que los sinteticos y, cuando hay suficientes, los tiempos se estiman
unicamente con ellos.

Uso rapido por consola:
    python smartdesk_engine.py "no puedo conectarme a la vpn desde casa"
    python smartdesk_engine.py "el erp esta muy lento" --ahora "2026-10-12 17:30"

Dependencias: pandas, numpy, scipy, scikit-learn. Opcional: xgboost (se usa si esta instalado).
"""
from __future__ import annotations

import argparse
import json
import re
import threading
import unicodedata
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import (HistGradientBoostingClassifier, HistGradientBoostingRegressor,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import f1_score, median_absolute_error, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

try:
    from xgboost import XGBClassifier, XGBRegressor
    HAY_XGB = True
except ImportError:          # el programa funciona igual sin xgboost
    HAY_XGB = False

BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
AGENTES_JSON = BASE / "agentes.json"
COLA_JSON = DATA / "cola.json"
REALES_CSV = DATA / "tickets_reales.csv"

PESO_REAL = 5.0           # un ticket real pesa 5 veces uno sintetico al entrenar
MIN_REALES_TIEMPO = 5     # con >=5 tickets reales de la subcategoria, el tiempo se estima solo con ellos
UMBRAL_CONFIANZA = 0.45   # por debajo, se avisa que la falla no es clara
SVD_COMPONENTES = 80      # el texto se resume en 80 numeros para los modelos de arboles y la regresion lineal
MARGEN_F1 = 0.02          # un modelo de criticidad debe superar a la matriz ITIL por al menos 0,02 de F1 macro
MARGEN_TIEMPO = 0.97      # un modelo de tiempo debe bajar el error mediano al menos 3 % frente a la estadistica
# Para forzar un modelo aunque no supere la validacion (pruebas), por ejemplo:
#   FORZAR = {"criticidad": "XGBoost", "tiempo": "Regresión Lineal"}
FORZAR: dict = {}
RANGO = {"Crítica": 0, "Alta": 1, "Media": 2, "Baja": 3}
DIAS = ["lun", "mar", "mié", "jue", "vie", "sáb", "dom"]
MATRIZ = {("Alto", "Alto"): "Crítica", ("Alto", "Medio"): "Alta", ("Alto", "Bajo"): "Media",
          ("Medio", "Alto"): "Alta", ("Medio", "Medio"): "Media", ("Medio", "Bajo"): "Baja",
          ("Bajo", "Alto"): "Media", ("Bajo", "Medio"): "Baja", ("Bajo", "Bajo"): "Baja"}
COLS_REALES = ["ticket_id", "fecha_creacion", "fecha_resolucion", "titulo", "descripcion_falla",
               "categoria", "subcategoria", "criticidad", "documentacion_solucion",
               "tiempo_resolucion_min", "agente"]


# ---------------------------------------------------------------------------
# Utilidades de texto y tiempo
# ---------------------------------------------------------------------------
def normalizar(texto) -> str:
    t = unicodedata.normalize("NFD", str(texto).lower())
    t = "".join(c for c in t if unicodedata.category(c) != "Mn")
    return re.sub(r"[^a-z0-9]+", " ", t).strip()


def fmt_min(m) -> str:
    m = int(round(m))
    if m < 60:
        return f"{m} min"
    h, r = divmod(m, 60)
    return f"{h} h" + (f" {r} min" if r else "")


def fmt_dt(dt: datetime, ahora: datetime) -> str:
    d = (dt.date() - ahora.date()).days
    hora = dt.strftime("%H:%M")
    if d == 0:
        return f"hoy {hora}"
    if d == 1:
        return f"mañana {hora}"
    return f"{DIAS[dt.weekday()]} {dt:%d/%m} {hora}"


def _hm(s: str) -> int:
    h, m = s.split(":")
    return int(h) * 60 + int(m)


# ---------------------------------------------------------------------------
# Calendario laboral de un agente
# ---------------------------------------------------------------------------
def ventanas(ag: dict, fecha) -> list[tuple[datetime, datetime]]:
    """Franjas de trabajo del agente en una fecha (sin almuerzo, sin fines de semana)."""
    if fecha.weekday() not in ag.get("dias_laborales", [0, 1, 2, 3, 4]):
        return []
    base = datetime(fecha.year, fecha.month, fecha.day)
    ini, fin = _hm(ag["turno"][0]), _hm(ag["turno"][1])
    tramos = [(ini, fin)]
    if ag.get("almuerzo"):
        a, b = _hm(ag["almuerzo"][0]), _hm(ag["almuerzo"][1])
        tramos = [(ini, a), (b, fin)]
    return [(base + timedelta(minutes=x), base + timedelta(minutes=y)) for x, y in tramos if y > x]


def sumar_minutos_laborales(ag: dict, desde: datetime, minutos: float) -> datetime:
    """Instante en que se completan `minutos` de trabajo, contando solo horas laborales."""
    restante, t = float(minutos), desde
    for i in range(45):
        for a, b in ventanas(ag, (desde + timedelta(days=i)).date()):
            if b <= t:
                continue
            a = max(a, t)
            disp = (b - a).total_seconds() / 60
            if restante <= disp:
                return a + timedelta(minutes=restante)
            restante -= disp
    return desde + timedelta(days=45)  # agente sin turnos configurados


def estado_ahora(ag: dict, ahora: datetime, pendiente_min: float) -> tuple[str, bool]:
    if ag.get("ausente"):
        return "Ausente", False
    if any(a <= ahora < b for a, b in ventanas(ag, ahora.date())):
        if pendiente_min <= 0:
            return "Disponible ahora", True
        return f"Ocupado (cola de {fmt_min(pendiente_min)})", True
    alm = ag.get("almuerzo")
    if alm and ventanas(ag, ahora.date()):
        m = ahora.hour * 60 + ahora.minute
        if _hm(alm[0]) <= m < _hm(alm[1]):
            return f"En almuerzo (vuelve {alm[1]})", False
    vuelve = sumar_minutos_laborales(ag, ahora, 0)
    return f"Fuera de turno (vuelve {fmt_dt(vuelve, ahora)})", False


# ---------------------------------------------------------------------------
# Motor
# ---------------------------------------------------------------------------
class SmartDesk:
    def __init__(self):
        self.lock = threading.RLock()
        DATA.mkdir(exist_ok=True)
        self.info_modelos = {}
        self.m_crit = None
        self.m_tiempo = None
        self.cargar_agentes()
        self.entrenar()

    # ---- agentes ----------------------------------------------------------
    def cargar_agentes(self):
        with self.lock:
            self.cfg = json.loads(AGENTES_JSON.read_text(encoding="utf-8"))

    def guardar_agentes(self):
        with self.lock:
            AGENTES_JSON.write_text(json.dumps(self.cfg, ensure_ascii=False, indent=2), encoding="utf-8")

    def set_ausente(self, agente_id: str, ausente: bool):
        with self.lock:
            for ag in self.cfg["agentes"]:
                if ag["id"] == agente_id:
                    ag["ausente"] = ausente
            self.guardar_agentes()

    # ---- entrenamiento ----------------------------------------------------
    def _leer_reales(self) -> pd.DataFrame:
        if not REALES_CSV.exists():
            return pd.DataFrame(columns=COLS_REALES + ["origen"])
        r = pd.read_csv(REALES_CSV)
        r = r.dropna(subset=["descripcion_falla", "subcategoria", "criticidad", "categoria"])
        r["origen"] = "real"
        return r

    def entrenar(self, evaluar: bool = True) -> dict:
        """Entrena los modelos. Con evaluar=True (arranque y boton Reentrenar) tambien valida los modelos
        candidatos contra los metodos actuales; al cerrar un ticket (evaluar=False) solo reajusta los ya elegidos."""
        with self.lock:
            sint = pd.read_csv(DATA / "tickets_smartdesk_es.csv")
            sint["origen"] = "sintetico"
            self.kb = pd.read_csv(DATA / "kb_articulos_es.csv")
            reales = self._leer_reales()
            df = pd.concat([sint, reales], ignore_index=True)
            df["num_reasignaciones"] = df["num_reasignaciones"].fillna(0)
            df["titulo"] = df["titulo"].fillna("")
            df["documentacion_solucion"] = df["documentacion_solucion"].fillna("")
            df["texto"] = (df["titulo"] + " " + df["descripcion_falla"].fillna("")).map(normalizar)
            peso = np.where(df["origen"] == "real", PESO_REAL, 1.0)

            self.vw = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
            self.vc = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3,
                                      sublinear_tf=True, max_features=60000)
            X = hstack([self.vw.fit_transform(df["texto"]), self.vc.fit_transform(df["texto"])]).tocsr()

            self.c_sub = LogisticRegression(C=10, max_iter=400).fit(X, df["subcategoria"], sample_weight=peso)
            # impacto y urgencia se predicen por separado y se combinan con la matriz ITIL;
            # si los tickets reales no traen esas columnas, se usan solo los sinteticos.
            m = df["impacto"].notna() & df["urgencia"].notna()
            self.c_imp = LogisticRegression(C=5, max_iter=400, class_weight="balanced").fit(
                X[m.values], df.loc[m, "impacto"], sample_weight=peso[m.values])
            self.c_urg = LogisticRegression(C=5, max_iter=400, class_weight="balanced").fit(
                X[m.values], df.loc[m, "urgencia"], sample_weight=peso[m.values])

            self.sub_a_cat = df.groupby("subcategoria")["categoria"].agg(lambda s: s.mode().iloc[0]).to_dict()
            self.X, self.df = X, df
            self.stats = {"sinteticos": int((df.origen == "sintetico").sum()),
                          "reales": int((df.origen == "real").sum())}
            self._entrenar_modelos_extra(evaluar)
            self.stats["modelos"] = self.info_modelos
            return self.stats

    # ---- criticidad y tiempo: modelos candidatos con validacion -------------------
    @staticmethod
    def _onehot():
        try:
            return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        except TypeError:            # scikit-learn < 1.2
            return OneHotEncoder(handle_unknown="ignore", sparse=False)

    @staticmethod
    def _candidatos_crit() -> dict:
        c = {"Random Forest": lambda: RandomForestClassifier(150, min_samples_leaf=2, class_weight="balanced",
                                                             random_state=0, n_jobs=-1),
             "Gradient Boosting": lambda: HistGradientBoostingClassifier(max_iter=150, learning_rate=0.1,
                                                                         random_state=0)}
        if HAY_XGB:
            c["XGBoost"] = lambda: XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1, subsample=0.8,
                                                 random_state=0, verbosity=0)
        return c

    @staticmethod
    def _candidatos_tiempo() -> dict:
        c = {"Regresión Lineal": lambda: LinearRegression(),
             "Random Forest": lambda: RandomForestRegressor(150, min_samples_leaf=3, random_state=0, n_jobs=-1),
             "Gradient Boosting": lambda: HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05,
                                                                        random_state=0)}
        if HAY_XGB:
            c["XGBoost"] = lambda: XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
                                                random_state=0, verbosity=0)
        return c

    @staticmethod
    def _x_crit(S, usuarios):
        return np.c_[S, np.log1p(np.asarray(usuarios, dtype=float))]

    @staticmethod
    def _df_cat(sub, crit) -> pd.DataFrame:
        return pd.DataFrame({"subcategoria": list(sub), "criticidad": list(crit),
                             "par": [f"{a}|{b}" for a, b in zip(sub, crit)]})

    def _x_tiempo(self, S, sub, crit):
        return np.c_[S, self.oh_t.transform(self._df_cat(sub, crit))]

    def _puntaje_crit(self, y_ok, y_pred) -> dict:
        crit = self.crit_clases.index("Crítica")
        return {"f1": float(f1_score(y_ok, y_pred, average="macro")),
                "rec_critica": float(recall_score(y_ok, y_pred, labels=[crit], average=None, zero_division=0)[0])}

    def _entrenar_modelos_extra(self, evaluar: bool):
        self.svd = TruncatedSVD(SVD_COMPONENTES, random_state=0).fit(self.X)
        self.S = self.svd.transform(self.X)
        self._entrenar_crit(evaluar)
        self._entrenar_tiempo(evaluar)

    def _entrenar_crit(self, evaluar: bool):
        """Criticidad directa (texto + usuarios afectados) con Random Forest, Gradient Boosting o XGBoost.
        Solo se activa si supera a la matriz ITIL en validacion."""
        df = self.df
        m = (df["criticidad"].notna() & df["usuarios_afectados"].notna()
             & df["impacto"].notna() & df["urgencia"].notna()).values
        self.crit_clases = sorted(RANGO)
        cod = {c: i for i, c in enumerate(self.crit_clases)}
        y = df.loc[m, "criticidad"].map(cod).values
        Xc = self._x_crit(self.S[m], df.loc[m, "usuarios_afectados"].values)
        w = np.where(df.loc[m, "origen"].values == "real", PESO_REAL, 1.0)
        info = self.info_modelos.get("criticidad")
        self.m_crit = None
        if evaluar or info is None:
            itr, ite = train_test_split(np.arange(len(y)), test_size=0.2, random_state=42, stratify=y)
            Xm = self.X[m]
            ci = LogisticRegression(C=5, max_iter=400, class_weight="balanced").fit(
                Xm[itr], df.loc[m, "impacto"].values[itr], sample_weight=w[itr])
            cu = LogisticRegression(C=5, max_iter=400, class_weight="balanced").fit(
                Xm[itr], df.loc[m, "urgencia"].values[itr], sample_weight=w[itr])
            itil = [cod[MATRIZ[(i, u)]] for i, u in zip(ci.predict(Xm[ite]), cu.predict(Xm[ite]))]
            ref = self._puntaje_crit(y[ite], itil)
            cand = {}
            for nombre, make in self._candidatos_crit().items():
                mod = make().fit(Xc[itr], y[itr], sample_weight=w[itr])
                cand[nombre] = self._puntaje_crit(y[ite], np.asarray(mod.predict(Xc[ite])).astype(int))
            mejor = max(cand, key=lambda k: cand[k]["f1"])
            gana = (cand[mejor]["f1"] >= ref["f1"] + MARGEN_F1
                    and cand[mejor]["rec_critica"] >= ref["rec_critica"] - 0.05)
            if FORZAR.get("criticidad") in cand:
                mejor, gana = FORZAR["criticidad"], True
            motivo = (f'{mejor} superó a la matriz ITIL en validación (F1 {cand[mejor]["f1"]:.2f} frente a {ref["f1"]:.2f}).'
                      if gana else
                      f'El mejor candidato ({mejor}, F1 {cand[mejor]["f1"]:.2f}) no superó a la matriz ITIL '
                      f'(F1 {ref["f1"]:.2f}) con el margen exigido; se mantiene la matriz.')
            info = {"usa_modelo": bool(gana), "activo": mejor if gana else "Matriz ITIL", "mejor_candidato": mejor,
                    "referencia": ref, "candidatos": cand, "motivo": motivo}
        if info["usa_modelo"]:
            self.m_crit = self._candidatos_crit()[info["activo"]]().fit(Xc, y, sample_weight=w)
        self.info_modelos["criticidad"] = info

    def _estadistico_en(self, tr: pd.DataFrame, te: pd.DataFrame) -> np.ndarray:
        """El metodo actual de tiempo (mediana por falla y criticidad, con respaldos) calculado sobre `tr`."""
        glob = tr["tiempo_resolucion_min"].median()
        g1 = tr.groupby(["subcategoria", "criticidad"])["tiempo_resolucion_min"].agg(["median", "count"])
        g2 = tr.groupby("subcategoria")["tiempo_resolucion_min"].agg(["median", "count"])
        g3 = tr.groupby("categoria")["tiempo_resolucion_min"].agg(["median", "count"])
        out = []
        for s_, c_, k_ in zip(te["subcategoria"], te["criticidad"], te["categoria"]):
            v = glob
            if (s_, c_) in g1.index and g1.loc[(s_, c_), "count"] >= 8:
                v = g1.loc[(s_, c_), "median"]
            elif s_ in g2.index and g2.loc[s_, "count"] >= 8:
                v = g2.loc[s_, "median"]
            elif k_ in g3.index and g3.loc[k_, "count"] >= 8:
                v = g3.loc[k_, "median"]
            out.append(v)
        return np.array(out, dtype=float)

    def _entrenar_tiempo(self, evaluar: bool):
        """Tiempo de solucion con Regresion Lineal, Random Forest, Gradient Boosting o XGBoost (sobre log-minutos).
        Solo se activa si supera a la estadistica actual en validacion."""
        df = self.df
        m = ((df["num_reasignaciones"] == 0) & df["tiempo_resolucion_min"].notna()).values
        dd = df[m]
        self.oh_t = self._onehot().fit(self._df_cat(dd["subcategoria"].values, dd["criticidad"].values))
        Xt = self._x_tiempo(self.S[m], dd["subcategoria"].values, dd["criticidad"].values)
        y = np.log1p(dd["tiempo_resolucion_min"].values.astype(float))
        w = np.where(dd["origen"].values == "real", PESO_REAL, 1.0)
        info = self.info_modelos.get("tiempo")
        self.m_tiempo = None
        if evaluar or info is None:
            itr, ite = train_test_split(np.arange(len(y)), test_size=0.2, random_state=42)
            real_min = dd["tiempo_resolucion_min"].values[ite].astype(float)
            ref = float(median_absolute_error(real_min, self._estadistico_en(dd.iloc[itr], dd.iloc[ite])))
            cand, resid = {}, {}
            for nombre, make in self._candidatos_tiempo().items():
                mod = make().fit(Xt[itr], y[itr], sample_weight=w[itr])
                pred = mod.predict(Xt[ite])
                cand[nombre] = {"error_mediano_min": float(median_absolute_error(real_min, np.clip(np.expm1(pred), 1, None)))}
                resid[nombre] = y[ite] - pred
            mejor = min(cand, key=lambda k: cand[k]["error_mediano_min"])
            gana = cand[mejor]["error_mediano_min"] <= ref * MARGEN_TIEMPO
            if FORZAR.get("tiempo") in cand:
                mejor, gana = FORZAR["tiempo"], True
            motivo = (f'{mejor} superó a la estadística en validación (error mediano {cand[mejor]["error_mediano_min"]:.0f} '
                      f'frente a {ref:.0f} min).' if gana else
                      f'El mejor candidato ({mejor}, error mediano {cand[mejor]["error_mediano_min"]:.0f} min) no superó a la '
                      f'estadística actual ({ref:.0f} min); se mantiene la estadística.')
            info = {"usa_modelo": bool(gana), "activo": mejor if gana else "Estadística (mediana)", "mejor_candidato": mejor,
                    "referencia": {"error_mediano_min": ref}, "candidatos": cand, "motivo": motivo,
                    "r20": float(np.quantile(resid[mejor], 0.2)), "r80": float(np.quantile(resid[mejor], 0.8))}
        if info["usa_modelo"]:
            self.m_tiempo = self._candidatos_tiempo()[info["activo"]]().fit(Xt, y, sample_weight=w)
            self.n_tiempo = int(len(y))
        self.info_modelos["tiempo"] = info

    # ---- analisis de un caso ---------------------------------------------
    def _vectorizar(self, texto: str):
        q = normalizar(texto)
        return hstack([self.vw.transform([q]), self.vc.transform([q])]).tocsr()

    def estimar_tiempo(self, sub: str, cat: str, crit: str, Sq=None) -> dict:
        """Si hay >=5 tickets reales de la falla se usan solo ellos; si no, el modelo (si paso la validacion)
        o, en su defecto, la estadistica de casos pasados."""
        est = self._estimar_tiempo_estadistico(sub, cat, crit)
        if "solo tickets reales" in est["base"] or self.m_tiempo is None or Sq is None:
            return est
        info = self.info_modelos["tiempo"]
        p = max(float(np.expm1(self.m_tiempo.predict(self._x_tiempo(Sq, [sub], [crit]))[0])), 1.0)
        return {"mediana": p, "p20": min(p, p * float(np.exp(info["r20"]))), "p80": max(p, p * float(np.exp(info["r80"]))),
                "n": self.n_tiempo, "base": f'modelo {info["activo"]}'}

    def _estimar_tiempo_estadistico(self, sub: str, cat: str, crit: str) -> dict:
        d = self.df[self.df["num_reasignaciones"] == 0]  # tiempo "limpio": sin reasignaciones por mal triaje
        niveles = [("subcategoría y criticidad", (d.subcategoria == sub) & (d.criticidad == crit)),
                   ("subcategoría", d.subcategoria == sub),
                   ("categoría", d.categoria == cat)]
        for nombre, mask in niveles:
            real = d.loc[mask & (d.origen == "real"), "tiempo_resolucion_min"]
            todos = d.loc[mask, "tiempo_resolucion_min"]
            serie, base = (real, nombre + ", solo tickets reales") if len(real) >= MIN_REALES_TIEMPO else (todos, nombre)
            if len(serie) >= 8 or (len(real) >= MIN_REALES_TIEMPO and serie is real):
                return {"mediana": float(serie.median()), "p20": float(serie.quantile(0.2)),
                        "p80": float(serie.quantile(0.8)), "n": int(len(serie)), "base": base}
        s = d["tiempo_resolucion_min"]
        return {"mediana": float(s.median()), "p20": float(s.quantile(0.2)), "p80": float(s.quantile(0.8)),
                "n": int(len(s)), "base": "global"}

    @staticmethod
    def _a_usuarios(v):
        try:
            n = int(float(v))
            return n if n >= 1 else None
        except (TypeError, ValueError):
            return None

    def analizar(self, texto: str, criticidad_manual: str | None = None, ahora: datetime | None = None,
                 usuarios_afectados=None) -> dict:
        ahora = ahora or datetime.now()
        with self.lock:
            X = self._vectorizar(texto)
            Sq = self.svd.transform(X)
            usuarios = self._a_usuarios(usuarios_afectados)
            p = self.c_sub.predict_proba(X)[0]
            top = np.argsort(p)[::-1][:3]
            hipotesis = [{"subcategoria": self.c_sub.classes_[i],
                          "categoria": self.sub_a_cat.get(self.c_sub.classes_[i], ""),
                          "probabilidad": float(p[i])} for i in top]
            sub, cat = hipotesis[0]["subcategoria"], hipotesis[0]["categoria"]

            imp = self.c_imp.classes_[int(np.argmax(self.c_imp.predict_proba(X)[0]))]
            urg = self.c_urg.classes_[int(np.argmax(self.c_urg.predict_proba(X)[0]))]
            crit_auto = MATRIZ[(imp, urg)]
            metodo_crit = "Matriz ITIL (impacto × urgencia)"
            if self.m_crit is not None and usuarios is not None:
                crit_auto = self.crit_clases[int(self.m_crit.predict(self._x_crit(Sq, [usuarios]))[0])]
                metodo_crit = f'{self.info_modelos["criticidad"]["activo"]} (texto + usuarios afectados)'
            crit = criticidad_manual if criticidad_manual in RANGO else crit_auto

            tiempo = self.estimar_tiempo(sub, cat, crit, Sq)

            sims = (self.X @ X.T).toarray().ravel() / 2.0   # coseno medio palabra + caracteres
            vistos, similares = set(), []
            for i in np.argsort(sims)[::-1][:200]:
                fila = self.df.iloc[i]
                doc = str(fila["documentacion_solucion"]).strip()
                clave = normalizar(fila["titulo"])
                if not doc or clave in vistos:
                    continue
                vistos.add(clave)
                similares.append({"ticket_id": fila.get("ticket_id", ""), "titulo": fila["titulo"],
                                  "similitud": float(sims[i]), "subcategoria": fila["subcategoria"],
                                  "solucion": doc, "minutos": float(fila["tiempo_resolucion_min"]),
                                  "origen": fila["origen"]})
                if len(similares) == 3:
                    break

            kb = self.kb[self.kb["subcategoria"] == sub]
            kb = kb.iloc[0].to_dict() if len(kb) else None

            agentes = self.recomendar_agentes(cat, crit, tiempo["mediana"], ahora)
            rec = next((a for a in agentes if a["recomendado"]), None)
            alerta = None
            if not any(a["en_turno"] for a in agentes):
                alerta = ("Ningún agente está en turno ahora. "
                          + ("Es un caso crítico: contacta a un agente de guardia o escala el caso. " if RANGO[crit] <= 1 else "")
                          + (f'Si se asigna, empezaría {rec["inicio_txt"]}.' if rec else ""))
            elif rec is None:
                alerta = "Todos los agentes están marcados como ausentes."
            elif crit == "Crítica" and (rec["espera_min"] > 0 or not rec["en_turno"]):
                alerta = ("Es un caso crítico y el agente más apto no está libre ahora; "
                          "considera pausar un ticket de menor prioridad.")
            elif RANGO[crit] <= 1 and rec["cola_min"] > 0 and rec["espera_min"] == 0:
                alerta = (f'{rec["nombre"]} tiene {fmt_min(rec["cola_min"])} de tickets de menor prioridad en cola; '
                          f'este caso pasaría por delante de ellos.')
            return {"alerta": alerta, "descripcion": texto, "hipotesis": hipotesis, "categoria": cat, "subcategoria": sub,
                    "impacto": imp, "urgencia": urg, "criticidad": crit, "criticidad_automatica": crit_auto,
                    "criticidad_manual": crit != crit_auto, "tiempo": tiempo, "kb": kb, "similares": similares,
                    "agentes": agentes, "baja_confianza": hipotesis[0]["probabilidad"] < UMBRAL_CONFIANZA,
                    "metodo_criticidad": metodo_crit, "ahora": ahora.isoformat(timespec="minutes")}

    # ---- cola y disponibilidad --------------------------------------------
    def _leer_cola(self) -> dict:
        if COLA_JSON.exists():
            return json.loads(COLA_JSON.read_text(encoding="utf-8"))
        return {"siguiente": 1, "abiertos": []}

    def _guardar_cola(self, estado: dict):
        COLA_JSON.write_text(json.dumps(estado, ensure_ascii=False, indent=2), encoding="utf-8")

    def cola(self) -> list[dict]:
        with self.lock:
            return self._leer_cola()["abiertos"]

    def _habilidad(self, ag: dict, cat: str) -> float:
        return float(ag.get("habilidades", {}).get(cat, self.cfg.get("habilidad_por_defecto", 0.5)))

    def recomendar_agentes(self, cat: str, crit: str, minutos: float, ahora: datetime) -> list[dict]:
        cola = self.cola()
        filas = []
        for ag in self.cfg["agentes"]:
            propios = [t for t in cola if t["agente_id"] == ag["id"]]
            total = sum(t["minutos_est"] for t in propios)
            # un ticket solo espera a los que se atienden antes (misma o mayor prioridad)
            antes = sum(t["minutos_est"] for t in propios if RANGO[t["criticidad"]] <= RANGO[crit])
            hab = self._habilidad(ag, cat)
            estado, en_turno = estado_ahora(ag, ahora, total)
            ausente = bool(ag.get("ausente"))
            dur = minutos * (1 + 0.6 * (1 - hab))  # menos habilidad -> mas lento
            inicio = fin = None
            if not ausente:
                inicio = sumar_minutos_laborales(ag, ahora, antes)
                fin = sumar_minutos_laborales(ag, ahora, antes + dur)
            filas.append({"id": ag["id"], "nombre": ag["nombre"], "habilidad": hab, "estado": estado,
                          "en_turno": en_turno, "ausente": ausente, "cola_min": total, "espera_min": antes,
                          "duracion_min": dur, "inicio": inicio, "fin": fin, "tickets": len(propios)})
        # criticos/altos: solo agentes con buena habilidad, si existe al menos uno disponible con ella
        elegibles = [f for f in filas if not f["ausente"]]
        if RANGO[crit] <= 1 and any(f["habilidad"] >= 0.6 for f in elegibles):
            elegibles = [f for f in elegibles if f["habilidad"] >= 0.6]
        elegibles.sort(key=lambda f: (f["fin"], -f["habilidad"]))
        rec_id = elegibles[0]["id"] if elegibles else None
        orden = {f["id"]: i for i, f in enumerate(elegibles)}
        filas.sort(key=lambda f: (orden.get(f["id"], 99), f["nombre"]))
        for f in filas:
            f["recomendado"] = f["id"] == rec_id
            f["inicio_txt"] = fmt_dt(f["inicio"], ahora) if f["inicio"] else "—"
            f["fin_txt"] = fmt_dt(f["fin"], ahora) if f["fin"] else "—"
            f["inicio"] = f["inicio"].isoformat(timespec="minutes") if f["inicio"] else None
            f["fin"] = f["fin"].isoformat(timespec="minutes") if f["fin"] else None
        return filas

    def estado_agentes(self, ahora: datetime | None = None) -> list[dict]:
        ahora = ahora or datetime.now()
        cola = self.cola()
        out = []
        for ag in self.cfg["agentes"]:
            propios = [t for t in cola if t["agente_id"] == ag["id"]]
            total = sum(t["minutos_est"] for t in propios)
            estado, en_turno = estado_ahora(ag, ahora, total)
            libre = None if ag.get("ausente") else sumar_minutos_laborales(ag, ahora, total)
            out.append({"id": ag["id"], "nombre": ag["nombre"], "estado": estado, "en_turno": en_turno,
                        "ausente": bool(ag.get("ausente")), "cola_min": total, "tickets": propios,
                        "turno": f'{ag["turno"][0]}–{ag["turno"][1]}',
                        "libre_txt": fmt_dt(libre, ahora) if libre else "—"})
        return out

    def asignar(self, descripcion, categoria, subcategoria, criticidad, minutos_est, agente_id,
                ahora: datetime | None = None) -> str:
        ahora = ahora or datetime.now()
        with self.lock:
            estado = self._leer_cola()
            tid = f"SD-{estado['siguiente']:04d}"
            estado["siguiente"] += 1
            estado["abiertos"].append({"ticket_id": tid, "descripcion": descripcion, "categoria": categoria,
                                       "subcategoria": subcategoria, "criticidad": criticidad,
                                       "minutos_est": float(minutos_est), "agente_id": agente_id,
                                       "asignado_en": ahora.isoformat(timespec="minutes")})
            self._guardar_cola(estado)
            return tid

    def resolver(self, ticket_id, categoria, subcategoria, criticidad, minutos_reales, solucion,
                 ahora: datetime | None = None) -> bool:
        """Cierra un ticket: lo registra como dato real y reentrena el modelo."""
        ahora = ahora or datetime.now()
        with self.lock:
            estado = self._leer_cola()
            t = next((x for x in estado["abiertos"] if x["ticket_id"] == ticket_id), None)
            if t is None:
                return False
            estado["abiertos"] = [x for x in estado["abiertos"] if x["ticket_id"] != ticket_id]
            self._guardar_cola(estado)
            agente = next((a["nombre"] for a in self.cfg["agentes"] if a["id"] == t["agente_id"]), "")
            fila = {"ticket_id": ticket_id, "fecha_creacion": t["asignado_en"],
                    "fecha_resolucion": ahora.isoformat(timespec="minutes"),
                    "titulo": t["descripcion"].split(".")[0][:80], "descripcion_falla": t["descripcion"],
                    "categoria": categoria, "subcategoria": subcategoria, "criticidad": criticidad,
                    "documentacion_solucion": solucion, "tiempo_resolucion_min": float(minutos_reales),
                    "agente": agente}
            nuevo = pd.DataFrame([fila], columns=COLS_REALES)
            nuevo.to_csv(REALES_CSV, mode="a", header=not REALES_CSV.exists(), index=False, encoding="utf-8")
            self.entrenar(evaluar=False)
            return True

    def opciones_catalogo(self) -> dict:
        return {"categorias": sorted(set(self.sub_a_cat.values())), "subcategorias": sorted(self.sub_a_cat),
                "criticidades": list(RANGO)}


# ---------------------------------------------------------------------------
# Consola
# ---------------------------------------------------------------------------
def reporte(r: dict) -> str:
    ahora = datetime.fromisoformat(r["ahora"])
    L = [f'Caso: {r["descripcion"]}', ""]
    L.append("FALLA PROBABLE")
    for i, h in enumerate(r["hipotesis"], 1):
        L.append(f'  {i}. {h["subcategoria"]} ({h["categoria"]}) - {h["probabilidad"]:.0%}')
    if r["baja_confianza"]:
        L.append("  ! Confianza baja: conviene que un agente revise el caso antes de actuar.")
    manual = " (definida manualmente)" if r["criticidad_manual"] else ""
    L.append(f'\nCRITICIDAD: {r["criticidad"]}{manual}  [impacto {r["impacto"]}, urgencia {r["urgencia"]}]')
    L.append(f'  método de criticidad: {r["metodo_criticidad"]}')
    t = r["tiempo"]
    L.append(f'TIEMPO ESTIMADO: {fmt_min(t["mediana"])} (rango típico {fmt_min(t["p20"])} a {fmt_min(t["p80"])}; '
             f'base: {t["base"]}, {t["n"]} casos)')
    if r.get("alerta"):
        L.append(f'\n!! ALERTA: {r["alerta"]}')
    L.append("\nAGENTES (mejor opción primero)")
    for a in r["agentes"]:
        marca = "->" if a["recomendado"] else "  "
        L.append(f'  {marca} {a["nombre"]:<10} habilidad {a["habilidad"]:.0%} | {a["estado"]:<42} | '
                 f'inicia {a["inicio_txt"]:<16} termina {a["fin_txt"]}')
    if r["kb"]:
        L.append(f'\nPROCEDIMIENTO SUGERIDO ({r["kb"]["kb_id"]})\n{r["kb"]["procedimiento"]}')
    L.append("\nCASOS SIMILARES")
    for s in r["similares"]:
        L.append(f'  - {s["titulo"]} (similitud {s["similitud"]:.0%}, {fmt_min(s["minutos"])})')
    return "\n".join(L)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("texto")
    ap.add_argument("--ahora", help='fecha/hora simulada, ej. "2026-10-12 17:30"')
    ap.add_argument("--criticidad", choices=list(RANGO))
    ap.add_argument("--usuarios", type=int, help="usuarios afectados (opcional; activa el modelo de criticidad si fue validado)")
    a = ap.parse_args()
    ahora = datetime.strptime(a.ahora, "%Y-%m-%d %H:%M") if a.ahora else None
    print(reporte(SmartDesk().analizar(a.texto, a.criticidad, ahora, a.usuarios)))
