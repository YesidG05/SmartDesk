#!/usr/bin/env python3
"""
SmartDesk ML - Interfaz web local.

    python smartdesk_server.py            # abre http://127.0.0.1:8000
    python smartdesk_server.py --port 9000 --no-abrir

Pantallas:
  /        Nuevo caso: escribes la falla y ves falla probable, criticidad,
           tiempo estimado y que agente puede atenderlo antes. Aqui se asigna.
  /cola    Cola por agente, marcar ausencias y cerrar tickets (el cierre
           alimenta el aprendizaje del modelo).
  /api/analizar   (POST JSON {"descripcion": "..."}) para integrarlo con otros sistemas.

Solo usa la libreria estandar + las dependencias del motor. Sin autenticacion:
pensado para uso local. No lo expongas a internet.
"""
import argparse
import html
import json
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from smartdesk_engine import RANGO, SmartDesk, fmt_min

e = html.escape
CLASES = {"Crítica": "c", "Alta": "a", "Media": "m", "Baja": "l"}

CSS = """
:root{--bg:#f6f7f9;--card:#fff;--tx:#1d2330;--mut:#6a7385;--bd:#e1e5ec;--ac:#2b5fd9;--acbg:#e8efff;
--c:#c62828;--a:#e65100;--m:#b58900;--l:#2e7d32;--warn:#fff4e5;--warnbd:#f0b45a}
@media(prefers-color-scheme:dark){:root{--bg:#14171d;--card:#1c2029;--tx:#e8eaf0;--mut:#9aa3b5;--bd:#2c3240;
--ac:#7aa2ff;--acbg:#23304f;--c:#ff6b6b;--a:#ffa05a;--m:#e6c15a;--l:#6fcf7a;--warn:#33291a;--warnbd:#8a6a2c}}
*{box-sizing:border-box}body{margin:0;font:15px/1.5 system-ui,-apple-system,Segoe UI,Roboto,sans-serif;background:var(--bg);color:var(--tx)}
header{background:var(--card);border-bottom:1px solid var(--bd);padding:12px 20px;display:flex;gap:24px;align-items:center}
header b{font-size:17px}header a{color:var(--mut);text-decoration:none;padding:4px 0}header a.on{color:var(--ac);border-bottom:2px solid var(--ac)}
main{max-width:980px;margin:0 auto;padding:20px}h2{margin:0 0 10px;font-size:17px}
.card{background:var(--card);border:1px solid var(--bd);border-radius:10px;padding:16px;margin-bottom:16px}
.row{display:grid;grid-template-columns:1fr 1fr;gap:16px}@media(max-width:700px){.row{grid-template-columns:1fr}}
textarea,input,select{width:100%;padding:9px;border:1px solid var(--bd);border-radius:8px;background:var(--bg);color:var(--tx);font:inherit}
textarea{min-height:110px;resize:vertical}label{display:block;color:var(--mut);font-size:13px;margin:10px 0 4px}
button{background:var(--ac);color:#fff;border:0;border-radius:8px;padding:9px 16px;font:inherit;cursor:pointer}
button.sec{background:var(--acbg);color:var(--ac)}button:hover{filter:brightness(1.08)}
table{width:100%;border-collapse:collapse}th,td{text-align:left;padding:8px 6px;border-bottom:1px solid var(--bd);font-size:14px;vertical-align:top}
th{color:var(--mut);font-weight:500;font-size:12px;text-transform:uppercase;letter-spacing:.03em}
tr.rec td{background:var(--acbg)}.tag{background:var(--ac);color:#fff;border-radius:10px;padding:1px 8px;font-size:11px;margin-left:6px}
.b{display:inline-block;border-radius:10px;padding:2px 10px;font-size:12px;font-weight:600;border:1px solid currentColor}
.b-c{color:var(--c)}.b-a{color:var(--a)}.b-m{color:var(--m)}.b-l{color:var(--l)}
.bar{background:var(--bd);border-radius:6px;height:8px;margin-top:4px}.bar i{display:block;height:8px;border-radius:6px;background:var(--ac)}
.alert{background:var(--warn);border:1px solid var(--warnbd);border-radius:10px;padding:12px 16px;margin-bottom:16px}
.big{font-size:28px;font-weight:700}.mut{color:var(--mut);font-size:13px}.ok{color:var(--l)}.no{color:var(--c)}
ol{margin:6px 0 0 20px;padding:0}details{border-top:1px solid var(--bd);padding:8px 0}summary{cursor:pointer}
pre{white-space:pre-wrap;font:13px/1.45 inherit;margin:6px 0 0;color:var(--mut)}.inline{display:inline}
"""


def pagina(cuerpo, activo="/"):
    nav = "".join(f'<a href="{u}" class="{"on" if u == activo else ""}">{n}</a>'
                  for u, n in [("/", "Nuevo caso"), ("/cola", "Cola y agentes")])
    return (f'<!doctype html><html lang="es"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width,initial-scale=1"><title>SmartDesk</title>'
            f'<style>{CSS}</style></head><body><header><b>SmartDesk</b>{nav}</header>'
            f'<main>{cuerpo}</main></body></html>').encode("utf-8")


def badge(crit):
    return f'<span class="b b-{CLASES.get(crit, "m")}">{e(crit)}</span>'


def tabla_estado(estados):
    filas = ""
    for a in estados:
        cls = "ok" if a["en_turno"] and not a["ausente"] else "no"
        filas += (f'<tr><td><b>{e(a["nombre"])}</b></td><td>{e(a["turno"])}</td>'
                  f'<td class="{cls}">{e(a["estado"])}</td><td>{len(a["tickets"])}</td><td>{e(a["libre_txt"])}</td></tr>')
    return ('<table><tr><th>Agente</th><th>Turno</th><th>Estado ahora</th><th>Tickets</th>'
            f'<th>Libre desde</th></tr>{filas}</table>')


def vista_inicio(sd):
    return pagina(f"""
<div class="card"><h2>Nuevo caso</h2>
<form method="post" action="/analizar">
<label>Describe la falla con tus palabras</label>
<textarea name="descripcion" required autofocus placeholder="Ej.: No puedo conectarme a la VPN desde mi casa, se queda en conectando..."></textarea>
<label>Criticidad (opcional: si la dejas en automática, el sistema la estima)</label>
<select name="criticidad"><option value="">Automática</option>{"".join(f'<option>{c}</option>' for c in RANGO)}</select>
<label>Usuarios afectados (opcional: mejora la estimación de criticidad si hay un modelo validado)</label>
<input type="number" name="usuarios" min="1" placeholder="Ej.: 1, 5, 40...">
<p><button type="submit">Analizar caso</button></p></form></div>
<div class="card"><h2>Agentes ahora</h2>{tabla_estado(sd.estado_agentes())}</div>""", "/")


def vista_resultado(r):
    alerta = f'<div class="alert"><b>Atención:</b> {e(r["alerta"])}</div>' if r.get("alerta") else ""
    hip = ""
    for h in r["hipotesis"]:
        hip += (f'<div style="margin-bottom:10px"><b>{e(h["subcategoria"])}</b> '
                f'<span class="mut">· {e(h["categoria"])} · {h["probabilidad"]:.0%}</span>'
                f'<div class="bar"><i style="width:{max(2, h["probabilidad"] * 100):.0f}%"></i></div></div>')
    if r["baja_confianza"]:
        hip += ('<p class="mut"><b>Confianza baja:</b> el texto no se parece claramente a un caso conocido. '
                'Conviene que un agente lo revise antes de actuar.</p>')
    manual = " (definida por ti)" if r["criticidad_manual"] else ""
    manual += f' · método: {r["metodo_criticidad"]}'
    t = r["tiempo"]

    filas = ""
    for a in r["agentes"]:
        tag = '<span class="tag">Recomendado</span>' if a["recomendado"] else ""
        filas += (f'<tr class="{"rec" if a["recomendado"] else ""}"><td>{e(a["nombre"])}{tag}</td>'
                  f'<td>{a["habilidad"]:.0%}</td><td>{e(a["estado"])}</td>'
                  f'<td>{e(a["inicio_txt"])}</td><td>{e(a["fin_txt"])}</td></tr>')
    opciones = "".join(
        f'<option value="{e(a["id"])}"{" selected" if a["recomendado"] else ""}>{e(a["nombre"])} — termina {e(a["fin_txt"])}</option>'
        for a in r["agentes"] if not a["ausente"])
    ocultos = "".join(f'<input type="hidden" name="{k}" value="{e(str(v))}">' for k, v in [
        ("descripcion", r["descripcion"]), ("categoria", r["categoria"]), ("subcategoria", r["subcategoria"]),
        ("criticidad", r["criticidad"]), ("minutos", f'{t["mediana"]:.1f}')])
    asignar = (f'<form method="post" action="/asignar" style="margin-top:12px">{ocultos}'
               f'<label>Asignar a</label><select name="agente_id">{opciones}</select>'
               f'<p><button type="submit">Asignar ticket</button></p></form>') if opciones else ""

    kb = ""
    if r["kb"]:
        pasos = "".join(f"<li>{e(p.split('. ', 1)[-1])}</li>" for p in r["kb"]["procedimiento"].split("\n"))
        kb = (f'<div class="card"><h2>Procedimiento sugerido</h2><div class="mut">{e(r["kb"]["kb_id"])} · '
              f'causa probable: {e(r["kb"]["causa_raiz"])}</div><ol>{pasos}</ol></div>')
    sim = "".join(
        f'<details><summary>{e(s["titulo"])} <span class="mut">· similitud {s["similitud"]:.0%} · '
        f'{fmt_min(s["minutos"])} · {e(s["origen"])}</span></summary><pre>{e(s["solucion"])}</pre></details>'
        for s in r["similares"])
    sim = f'<div class="card"><h2>Casos similares ya resueltos</h2>{sim}</div>' if sim else ""

    return pagina(f"""{alerta}
<div class="card"><h2>Caso</h2><div class="mut">{e(r["descripcion"])}</div></div>
<div class="row">
<div class="card"><h2>Falla probable</h2>{hip}</div>
<div class="card"><h2>Criticidad y tiempo</h2>
<div>{badge(r["criticidad"])}<span class="mut"> impacto {e(r["impacto"])}, urgencia {e(r["urgencia"])}{manual}</span></div>
<p class="big" style="margin:12px 0 0">{fmt_min(t["mediana"])}</p>
<div class="mut">tiempo típico de solución · rango {fmt_min(t["p20"])} a {fmt_min(t["p80"])}<br>
base: {e(t["base"])} ({t["n"]} casos)</div></div></div>
<div class="card"><h2>¿Quién puede atenderlo?</h2>
<table><tr><th>Agente</th><th>Habilidad</th><th>Estado ahora</th><th>Inicia</th><th>Termina</th></tr>{filas}</table>
{asignar}</div>{kb}{sim}""", "/")


def resumen_modelos(s):
    info = s.get("modelos", {})
    nombres = {"criticidad": "Criticidad", "tiempo": "Tiempo estimado"}
    filas = ""
    for k, v in info.items():
        if k == "criticidad":
            cand = ", ".join(f'{n} F1 {sc["f1"]:.2f}' for n, sc in v["candidatos"].items())
            ref = f'matriz ITIL F1 {v["referencia"]["f1"]:.2f}'
        else:
            cand = ", ".join(f'{n} {sc["error_mediano_min"]:.0f} min' for n, sc in v["candidatos"].items())
            ref = f'estadística {v["referencia"]["error_mediano_min"]:.0f} min'
        filas += (f'<tr><td><b>{nombres.get(k, k)}</b></td><td>{e(v["activo"])}</td>'
                  f'<td class="mut">{e(v["motivo"])}<br>Validación: {e(ref)}; candidatos: {e(cand)}</td></tr>')
    return (f'<h3 style="margin:14px 0 6px">Modelos en uso</h3><table><tr><th>Tarea</th><th>Método activo</th><th>Por qué</th></tr>{filas}</table>'
            if filas else "")


def vista_cola(sd, aviso=""):
    estados = sd.estado_agentes()
    tarjetas = ""
    for a in estados:
        filas = "".join(
            f'<tr><td>{e(t["ticket_id"])}</td><td>{badge(t["criticidad"])}</td><td>{e(t["subcategoria"])}</td>'
            f'<td>{fmt_min(t["minutos_est"])}</td><td class="mut">{e(t["descripcion"][:90])}</td>'
            f'<td><a href="/resolver?id={e(t["ticket_id"])}">Resolver</a></td></tr>' for t in a["tickets"])
        tabla = (f'<table><tr><th>Ticket</th><th>Crit.</th><th>Falla</th><th>Est.</th><th>Caso</th><th></th></tr>{filas}</table>'
                 if filas else '<div class="mut">Sin tickets abiertos.</div>')
        boton = ("Marcar disponible", "0") if a["ausente"] else ("Marcar ausente", "1")
        tarjetas += (f'<div class="card"><div style="display:flex;justify-content:space-between;gap:10px;flex-wrap:wrap">'
                     f'<h2>{e(a["nombre"])} <span class="mut">· turno {e(a["turno"])}</span></h2>'
                     f'<form method="post" action="/ausente" class="inline"><input type="hidden" name="agente_id" value="{e(a["id"])}">'
                     f'<input type="hidden" name="valor" value="{boton[1]}"><button class="sec">{boton[0]}</button></form></div>'
                     f'<div class="{"ok" if a["en_turno"] and not a["ausente"] else "no"}">{e(a["estado"])}</div>'
                     f'<div class="mut" style="margin-bottom:8px">Libre desde: {e(a["libre_txt"])}</div>{tabla}</div>')
    s = sd.stats
    modelo = (f'<div class="card"><h2>Modelo</h2><div class="mut">Entrenado con {s["sinteticos"]} tickets sintéticos y '
              f'<b>{s["reales"]} reales</b>. Cada ticket que cierres con «Resolver» mejora las predicciones y los tiempos.</div>'
              f'{resumen_modelos(s)}<form method="post" action="/reentrenar" style="margin-top:10px"><button class="sec">Reentrenar ahora</button></form></div>')
    ok = f'<div class="alert">{e(aviso)}</div>' if aviso else ""
    return pagina(f"{ok}{tarjetas}{modelo}", "/cola")


def vista_resolver(sd, tid):
    t = next((x for x in sd.cola() if x["ticket_id"] == tid), None)
    if not t:
        return pagina('<div class="card">Ticket no encontrado o ya cerrado. <a href="/cola">Volver</a></div>', "/cola")
    cat = sd.opciones_catalogo()
    dl = lambda id_, vals: f'<datalist id="{id_}">' + "".join(f'<option value="{e(v)}">' for v in vals) + "</datalist>"
    return pagina(f"""<div class="card"><h2>Resolver {e(tid)}</h2><div class="mut">{e(t["descripcion"])}</div>
<form method="post" action="/resolver">
<input type="hidden" name="id" value="{e(tid)}">
<label>Categoría real</label><input name="categoria" list="cats" value="{e(t["categoria"])}" required>
<label>Falla real (subcategoría)</label><input name="subcategoria" list="subs" value="{e(t["subcategoria"])}" required>
<label>Criticidad real</label><select name="criticidad">{"".join(f'<option{" selected" if c == t["criticidad"] else ""}>{c}</option>' for c in RANGO)}</select>
<label>Tiempo real de trabajo (minutos)</label><input name="minutos" type="number" min="1" step="1" value="{int(t["minutos_est"])}" required>
<label>¿Qué hiciste para resolverlo? (esto alimenta el procedimiento sugerido)</label>
<textarea name="solucion" required placeholder="1. ... 2. ... 3. ..."></textarea>
<p><button type="submit">Cerrar ticket y aprender</button></p></form>
{dl("cats", cat["categorias"])}{dl("subs", cat["subcategorias"])}</div>""", "/cola")


class Handler(BaseHTTPRequestHandler):
    sd: SmartDesk = None

    def _enviar(self, cuerpo, tipo="text/html; charset=utf-8", codigo=200):
        self.send_response(codigo)
        self.send_header("Content-Type", tipo)
        self.send_header("Content-Length", str(len(cuerpo)))
        self.end_headers()
        self.wfile.write(cuerpo)

    def _redirigir(self, destino):
        self.send_response(303)
        self.send_header("Location", destino)
        self.end_headers()

    def _body(self) -> bytes:
        n = min(int(self.headers.get("Content-Length", 0)), 200_000)
        return self.rfile.read(n)

    def _form(self) -> dict:
        return {k: v[0] for k, v in parse_qs(self._body().decode("utf-8"), keep_blank_values=True).items()}

    def do_GET(self):
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        sd = self.sd
        if u.path == "/":
            self._enviar(vista_inicio(sd))
        elif u.path == "/cola":
            aviso = (f'Ticket {q["nuevo"]} asignado.' if "nuevo" in q else
                     f'Ticket {q["cerrado"]} cerrado. El modelo se reentrenó con ese caso.' if "cerrado" in q else "")
            self._enviar(vista_cola(sd, aviso))
        elif u.path == "/resolver":
            self._enviar(vista_resolver(sd, q.get("id", "")))
        elif u.path == "/api/cola":
            self._enviar(json.dumps(sd.estado_agentes(), ensure_ascii=False, default=str).encode(), "application/json")
        else:
            self._enviar(pagina('<div class="card">Página no encontrada.</div>'), codigo=404)

    def do_POST(self):
        sd, u = self.sd, urlparse(self.path)
        if u.path == "/api/analizar":
            try:
                datos = json.loads(self._body().decode("utf-8"))
                r = sd.analizar(datos["descripcion"], datos.get("criticidad"), usuarios_afectados=datos.get("usuarios_afectados"))
                self._enviar(json.dumps(r, ensure_ascii=False, default=str).encode(), "application/json")
            except (ValueError, KeyError):
                self._enviar(b'{"error": "enviar JSON con el campo descripcion"}', "application/json", 400)
            return
        f = self._form()
        if u.path == "/analizar":
            desc = f.get("descripcion", "").strip()
            if not desc:
                return self._redirigir("/")
            self._enviar(vista_resultado(sd.analizar(desc, f.get("criticidad") or None, usuarios_afectados=f.get("usuarios") or None)))
        elif u.path == "/asignar":
            tid = sd.asignar(f["descripcion"], f["categoria"], f["subcategoria"], f["criticidad"],
                             float(f["minutos"]), f["agente_id"])
            self._redirigir(f"/cola?nuevo={tid}")
        elif u.path == "/ausente":
            sd.set_ausente(f["agente_id"], f.get("valor") == "1")
            self._redirigir("/cola")
        elif u.path == "/resolver":
            if sd.resolver(f["id"], f["categoria"].strip(), f["subcategoria"].strip(), f["criticidad"],
                           float(f["minutos"]), f["solucion"].strip()):
                self._redirigir(f"/cola?cerrado={f['id']}")
            else:
                self._redirigir("/cola")
        elif u.path == "/reentrenar":
            sd.entrenar()
            self._redirigir("/cola")
        else:
            self._enviar(pagina('<div class="card">Página no encontrada.</div>'), codigo=404)

    def log_message(self, fmt, *args):
        pass  # sin ruido en consola


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--no-abrir", action="store_true", help="no abrir el navegador")
    a = ap.parse_args()

    print("Entrenando modelo inicial...")
    Handler.sd = SmartDesk()
    print(f"Listo: {Handler.sd.stats['sinteticos']} tickets sintéticos, {Handler.sd.stats['reales']} reales.")
    servidor = ThreadingHTTPServer((a.host, a.port), Handler)
    url = f"http://{a.host}:{a.port}"
    print(f"SmartDesk en {url}  (Ctrl+C para salir)")
    if not a.no_abrir:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        servidor.serve_forever()
    except KeyboardInterrupt:
        print("\nCerrando...")
