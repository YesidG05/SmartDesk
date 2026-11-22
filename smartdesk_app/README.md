# SmartDesk — asistente de asignación para mesa de ayuda

Escribes un caso y el sistema te dice:

1. **Falla probable** (top 3 con probabilidad) y el **procedimiento sugerido**.
2. **Criticidad** (impacto × urgencia, matriz ITIL).
3. **Tiempo estimado de solución** (mediana y rango típico).
4. **Qué agente lo termina antes**, según su turno, almuerzo, ausencias, cola actual y habilidades.

No necesita ninguna herramienta de tickets: la cola y el historial viven en la carpeta `data/`.

## Instalación y arranque

```bash
pip install -r requirements.txt        # Python 3.9 o superior
python smartdesk_server.py             # abre http://127.0.0.1:8000
```

Prueba rápida por consola, sin interfaz:

```bash
python smartdesk_engine.py "no puedo conectarme a la vpn desde casa"
python smartdesk_engine.py "el erp esta muy lento" --ahora "2026-10-12 17:30"   # simula otra hora
```

## Antes de usarlo: configura tus 4 agentes

Edita `agentes.json`. Los 4 que vienen son **ejemplos**: pon nombres, turnos, hora de almuerzo y habilidades reales (0 a 1 por categoría; 1 = experto). Sin esto, las recomendaciones de agente no reflejan tu equipo.

## Uso diario

1. **Nuevo caso:** escribe la falla, revisa el análisis y pulsa *Asignar ticket* (por defecto elige al recomendado, pero puedes cambiarlo).
2. **Cola y agentes:** ves los tickets abiertos de cada agente y puedes marcar a alguien como ausente.
3. **Resolver:** cuando el agente termina, cierra el ticket con la falla real, el tiempo real y lo que hizo. Eso reentrena el modelo al instante.

## Cómo aprende

Arranca con `data/tickets_smartdesk_es.csv` (4.000 tickets **sintéticos**). Cada ticket cerrado se guarda en `data/tickets_reales.csv`; pesa 5 veces más que uno sintético, y cuando una falla acumula 5 o más cierres reales, su tiempo se estima solo con ellos. Si el agente escribe una falla nueva al cerrar, el sistema la aprende (necesita varios ejemplos para reconocerla con confianza).

## Integración

`POST /api/analizar` con `{"descripcion": "...", "criticidad": "Alta"}` (criticidad opcional) devuelve el análisis completo en JSON. `GET /api/cola` devuelve el estado de los agentes.

## Limitaciones que debes conocer

- **Las probabilidades iniciales son optimistas.** Los datos sintéticos salen de 33 plantillas, así que el modelo se muestra muy seguro con casos parecidos a ellas. Con casos fuera de ese catálogo baja la confianza y el sistema lo avisa; hazle caso a ese aviso.
- **Los tiempos iniciales son ficticios** hasta que registres cierres reales. La pantalla indica en qué base se calculó cada estimación.
- **La criticidad es la predicción más débil.** El texto solo insinúa impacto y urgencia; usa el selector manual cuando conozcas el impacto real.
- **La disponibilidad asume trabajo secuencial** (un agente, un ticket a la vez), tiempos de trabajo continuos dentro del turno y que el tiempo estimado es la mediana, no el peor caso.
- **Sin autenticación.** Está pensado para uso local; no lo expongas a internet. Con `--host 0.0.0.0` queda visible para toda tu red.
- Para fallas que no están en el catálogo inicial, el sistema solo las reconocerá después de que cierres tickets reales de ese tipo.
