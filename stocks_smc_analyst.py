#!/usr/bin/env python3
"""
================================================================================
NATHANIEL RIDGE — RIDGECREST EQUITIES | ANALISTA SMC/ICT DE ACCIONES (Bitget)
================================================================================
Asistente de análisis técnico (NO ejecuta órdenes, NO gestiona fondos) basado
en metodología Smart Money Concepts (SMC) + Price Action + ICT, aplicado a
contratos "USDT-FUTURES" de acciones (RWA / stock perpetuals) listados en
Bitget (ej. TSLAUSDT = Tesla, NVDAUSDT = Nvidia) — API pública, sin necesidad
de cuenta ni API key, con velas completas (OHLC) para el análisis SMC/ICT
real. Es el cuarto bot de la familia Ridgecrest (cripto, forex, materias
primas y ahora acciones), con la misma arquitectura que el de materias
primas — Bitget también es la mejor fuente pública para esto: lista más de
300 acciones/perpetuos tokenizados (contra ~20 en Bybit), con TSLA, NVDA,
AAPL, META y MSTR como los de mayor volumen.

TOP 10 DE ACCIONES EN BITGET (confirmadas como contrato USDT-FUTURES real,
elegidas por ser las de mayor volumen/liquidez y relevancia dentro del
listado de más de 300 acciones de Bitget):
    TSLAUSDT   Tesla
    NVDAUSDT   Nvidia
    AAPLUSDT   Apple
    METAUSDT   Meta
    MSFTUSDT   Microsoft
    AMZNUSDT   Amazon
    GOOGLUSDT  Alphabet (Google)
    MSTRUSDT   MicroStrategy (Strategy) — alta volatilidad, proxy de Bitcoin
    AMDUSDT    AMD
    COINUSDT   Coinbase

Igual que en los bots de materias primas y cripto, no se asume que un
símbolo seguirá existiendo para siempre: el bot prueba cada símbolo de
SYMBOLS y si Bitget no lo tiene (o lo retira), lo salta automáticamente sin
afectar a los demás.

DOS DIFERENCIAS IMPORTANTES FRENTE A CRIPTO/MATERIAS PRIMAS (por eso este
bot no es una simple copia, sino que tiene lógica propia):

1) HORARIO DE MERCADO: las acciones en Bitget NO cotizan 24/7 como el
   cripto. Siguen el horario bursátil de EE. UU.: de lunes 00:00 a sábado
   00:00 (hora UTC-4/Nueva York), cerrado el resto del fin de semana y en
   feriados bursátiles de EE. UU. Este bot detecta cuándo el mercado está
   cerrado y lo marca claramente en el reporte para que no se lea una señal
   de fin de semana como si fuera del momento — perseguir un precio con el
   mercado cerrado no tiene sentido y puede llevar a error.

2) GAPS DE APERTURA: al reabrir tras el cierre (o tras noticias/resultados
   trimestrales), el precio puede saltar de un cierre a una apertura sin
   velas intermedias — algo raro en cripto 24/7 pero normal en acciones.
   Estos gaps distorsionan la lectura de Order Blocks/FVG si no se
   señalan aparte, así que este bot detecta el gap más reciente y lo
   reporta como una sección propia con su magnitud en ATR.

Bitget, igual que para materias primas, restringe el acceso desde IPs de
EE. UU. a nivel de plataforma — este bot debe desplegarse en una región de
Render fuera de EE. UU. (Frankfurt o Singapore), igual que los otros tres
bots de la familia.

------------------------------------------------------------------------------
AVISO IMPORTANTE / DISCLAIMER
------------------------------------------------------------------------------
- Herramienta EDUCATIVA e INFORMATIVA. No es asesoría financiera.
- No ejecuta órdenes ni tiene acceso a tus fondos.
- El "plan de trading" es un ESCENARIO HIPOTÉTICO calculado con reglas
  automatizadas y simplificadas de SMC/ICT. No sustituye el juicio de un
  analista humano ni garantiza resultados.
- Estos productos son derivados tokenizados (RWA) sobre acciones, no
  acciones reales — suelen operarse con apalancamiento: el riesgo de
  pérdida puede superar el capital depositado. Además, resultados
  trimestrales y noticias corporativas pueden generar saltos de precio
  (gaps) que ninguna estructura SMC/ICT previa anticipa.
------------------------------------------------------------------------------
Requisitos:
    pip install requests pandas numpy

Uso básico:
    python stocks_smc_analyst.py --interval 60 --limit 300

Variables de entorno:
    export SYMBOLS="TSLAUSDT,NVDAUSDT,AAPLUSDT,METAUSDT,MSFTUSDT,AMZNUSDT,GOOGLUSDT,MSTRUSDT,AMDUSDT,COINUSDT"
    export BITGET_PRODUCT_TYPE="USDT-FUTURES"
    export SWING_ORDER=4
    export OB_LOOKBACK=8
    export LIQUIDITY_TOLERANCE_PCT=0.05
    export NEARBY_ZONE_PCT=0.03
    export RSI_PERIOD=14
    export VOLUME_LOOKBACK=20
    export VOLUME_ZSCORE_THRESHOLD=2.0
    export ATR_PERIOD=14
    export ATR_BUFFER_MULT=0.5
    export MIN_OB_BODY_ATR_MULT=0.15
    export BREAK_VOLUME_CONFIRM_MULT=1.2
    export GAP_ATR_MULT=0.5
    export MARKET_CLOSE_WEEKDAY_UTC4=5   # 0=lunes ... 5=sábado (cierre semanal)

Notas sobre --interval (mismo formato que los otros bots de la familia):
    Acepta tanto el formato nativo de Bitget (1m,5m,15m,30m,1H,4H,6H,12H,
    1D,3D,1W,1M) como el formato numérico heredado en minutos (60 -> 1H,
    etc.), para reutilizar la misma configuración de ANALYST_ARGS en Render.

Envío por Telegram (para despliegue 24/7 en Render, región Frankfurt/Singapore):
    export TELEGRAM_BOT_TOKEN="tu-token-de-botfather"
    export TELEGRAM_CHAT_ID="tu-chat-id"
    python stocks_smc_analyst.py --interval 60 --watch 60 --telegram
================================================================================
"""
import argparse
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Literal

import numpy as np
import pandas as pd
import requests

# ==============================================================================
# CONFIGURACIÓN (todo ajustable por variables de entorno)
# ==============================================================================
BITGET_BASE_URL = "https://api.bitget.com"
BITGET_PRODUCT_TYPE = os.environ.get("BITGET_PRODUCT_TYPE", "USDT-FUTURES").strip()

# Top 10 de acciones (RWA / stock perpetuals) confirmadas como contrato
# USDT-FUTURES en Bitget, elegidas por ser las de mayor volumen/liquidez.
# El bot salta automáticamente cualquier símbolo que Bitget no devuelva.
DEFAULT_SYMBOLS = (
    "TSLAUSDT,NVDAUSDT,AAPLUSDT,METAUSDT,MSFTUSDT,"
    "AMZNUSDT,GOOGLUSDT,MSTRUSDT,AMDUSDT,COINUSDT"
)
SYMBOLS = [s.strip().upper() for s in os.environ.get("SYMBOLS", DEFAULT_SYMBOLS).split(",") if s.strip()]

SWING_ORDER = int(os.environ.get("SWING_ORDER", "4"))
OB_LOOKBACK = int(os.environ.get("OB_LOOKBACK", "8"))
LIQUIDITY_TOLERANCE_PCT = float(os.environ.get("LIQUIDITY_TOLERANCE_PCT", "0.05"))
NEARBY_ZONE_PCT = float(os.environ.get("NEARBY_ZONE_PCT", "0.03"))
RSI_PERIOD = int(os.environ.get("RSI_PERIOD", "14"))
VOLUME_LOOKBACK = int(os.environ.get("VOLUME_LOOKBACK", "20"))
VOLUME_ZSCORE_THRESHOLD = float(os.environ.get("VOLUME_ZSCORE_THRESHOLD", "2.0"))

# --- Parámetros de mejora de señal (heredados del bot de materias primas) ---
ATR_PERIOD = int(os.environ.get("ATR_PERIOD", "14"))
ATR_BUFFER_MULT = float(os.environ.get("ATR_BUFFER_MULT", "0.5"))
MIN_OB_BODY_ATR_MULT = float(os.environ.get("MIN_OB_BODY_ATR_MULT", "0.15"))
BREAK_VOLUME_CONFIRM_MULT = float(os.environ.get("BREAK_VOLUME_CONFIRM_MULT", "1.2"))

# --- Parámetros específicos de ACCIONES (nuevos en este bot) ---------------
# Un gap se reporta cuando la apertura de la última vela se separa del
# cierre anterior más de este múltiplo del ATR — típico tras un cierre de
# fin de semana o resultados trimestrales, y algo que casi no ocurre en
# cripto/materias primas 24/7.
GAP_ATR_MULT = float(os.environ.get("GAP_ATR_MULT", "0.5"))
# Día de la semana (hora UTC-4, estilo Nueva York) en que cierra el mercado
# hasta el lunes: 0=lunes ... 5=sábado. Bitget cierra sus acciones de
# sábado 00:00 a lunes 00:00 (UTC-4).
MARKET_CLOSE_WEEKDAY_UTC4 = int(os.environ.get("MARKET_CLOSE_WEEKDAY_UTC4", "5"))

API_CALL_DELAY_SECONDS = float(os.environ.get("API_CALL_DELAY_SECONDS", "1"))

DEPLOY_LABEL = os.environ.get("DEPLOY_LABEL", "").strip()

OTE_LEVELS = (0.618, 0.705, 0.79)

# ── Supabase (Fase 3: guardar cada señal, además de mandarla a Telegram) ────
# Usa la clave SECRETA (service_role), no la pública — este bot corre en un
# servidor de confianza (Render), no en el navegador de un cliente. Si estas
# 2 variables no están configuradas, el bot simplemente no guarda nada en
# Supabase y sigue funcionando igual que antes (no rompe nada).
SUPABASE_URL = os.environ.get("SUPABASE_URL", "").strip().rstrip("/")
SUPABASE_SERVICE_ROLE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "").strip()
BOT_NAME = "acciones"  # debe coincidir con el check constraint de "bot" en la tabla senales

# ── Zona horaria (agregado a pedido del usuario) ────────────────────────────
# Los reportes siempre muestran la hora en UTC (así todos los bots son
# consistentes entre sí), pero además muestran la hora local aproximada
# entre paréntesis, para que tus clientes no tengan que hacer la cuenta a
# mano. Configurable por si en algún momento cambia el huso horario del
# negocio o de la mayoría de los clientes.
TIMEZONE_OFFSET_HORAS = float(os.environ.get("TIMEZONE_OFFSET_HORAS", "-5"))


def formatear_fecha_con_hora_local(dt) -> str:
    local = dt + timedelta(hours=TIMEZONE_OFFSET_HORAS)
    signo = "+" if TIMEZONE_OFFSET_HORAS >= 0 else "-"
    return f"{dt:%Y-%m-%d %H:%M} UTC ({local:%H:%M} hora local, UTC{signo}{abs(TIMEZONE_OFFSET_HORAS):g})"

BOT_NAME_LEGIBLE = "Acciones"

# ── Explicación hablada en español simple (para los clientes del canal) ────
# Punto agregado a pedido del usuario: además del reporte técnico de
# siempre, se manda un audio explicando la señal en lenguaje fácil, usando
# gTTS (motor de voz gratuito de Google Translate — la voz suena algo
# robótica, pero se entiende perfecto y no tiene ningún costo).
CONFIANZA_SIMPLE = {
    "alta": "está bastante seguro de esto",
    "media-alta": "confía bastante, aunque no del todo",
    "media": "tiene una confianza media, ni mucha ni poca",
    "baja": "no está muy seguro, hay que tomarlo con pinzas",
}


def construir_explicacion_hablada(symbol: str, report) -> str:
    tp = report.trade_plan
    sube = report.bias == "alcista"
    frases = [f"El bot de {BOT_NAME_LEGIBLE} piensa que el precio de {symbol} va a {'subir' if sube else 'bajar'}."]

    confianza_texto = CONFIANZA_SIMPLE.get(report.confidence, "tiene una confianza media")
    frases.append(f"El bot {confianza_texto}.")

    pd_info = report.premium_discount
    if pd_info:
        zona_texto = (
            "el precio está caro, comparado con lo último que se movió" if pd_info.zone == "premium"
            else "el precio está barato, comparado con lo último que se movió"
        )
        frases.append(f"Ahora mismo, {zona_texto}.")

    if report.liquidity_zones:
        n = len(report.liquidity_zones)
        frases.append(
            f"Cerca hay {'una zona' if n == 1 else f'{n} zonas'} donde mucha gente tiene puestas sus "
            f"apuestas, esto se llama liquidez. A veces el precio va justo ahí a sacudir esas apuestas "
            f"antes de moverse de verdad, como una trampa antes del movimiento real."
        )

    if report.ob_zones:
        n = len(report.ob_zones)
        frases.append(
            f"También el bot encontró {'una zona' if n == 1 else f'{n} zonas'} donde inversores grandes "
            f"compraron o vendieron fuerte antes, esto se llama Order Block. Son como huellas en la arena, "
            f"y el precio a veces vuelve a pisarlas antes de seguir su camino."
        )

    if tp:
        frases.append(
            f"Si alguien quisiera copiar esta idea: entraría cerca de {tp.entry_low:.4f}, pondría una "
            f"alarma de emergencia, el stop loss, en {tp.stop_loss:.4f} para salir si el bot se equivoca, "
            f"y la primera meta de ganancia sería {tp.take_profit_1:.4f}."
        )

    frases.append("Recordá: esto es un análisis técnico automático, no una promesa ni un consejo financiero.")
    return " ".join(frases)


def generar_audio_explicacion(texto: str):
    print(f"[Audio] Generando audio con gTTS ({len(texto)} caracteres de texto)...")
    try:
        from gtts import gTTS
        import io
        buf = io.BytesIO()
        # tld="com.mx" da un acento de español latinoamericano, generalmente
        # más neutro y claro que el acento de España que sale por defecto.
        gTTS(text=texto, lang="es", tld="com.mx").write_to_fp(buf)
        audio_bytes = buf.getvalue()
        print(f"[Audio] Audio generado OK ({len(audio_bytes)} bytes).")
        return audio_bytes
    except Exception as exc:
        print(f"[Aviso] No se pudo generar el audio de explicación: {type(exc).__name__}: {exc}")
        return None


def subir_audio_supabase(audio_bytes, symbol: str):
    """Sube el audio a Supabase Storage y devuelve el link público, para que
    el backoffice pueda reproducir la MISMA voz que se manda a Telegram (en
    vez de depender de la voz que traiga cada celular)."""
    if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
        print("[Audio] Falta SUPABASE_URL o SUPABASE_SERVICE_ROLE_KEY, no se sube el audio.")
        return None
    nombre_archivo = f"{BOT_NAME}/{symbol.replace('/', '-')}-{int(time.time())}.mp3"
    url = f"{SUPABASE_URL}/storage/v1/object/audios-senales/{nombre_archivo}"
    try:
        resp = requests.post(
            url,
            data=audio_bytes,
            headers={
                "apikey": SUPABASE_SERVICE_ROLE_KEY,
                "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}",
                "Content-Type": "audio/mpeg",
            },
            timeout=30,
        )
        if resp.status_code >= 300:
            print(f"[Aviso] No se pudo subir el audio de {symbol}: {resp.status_code} {resp.text}")
            return None
        link = f"{SUPABASE_URL}/storage/v1/object/public/audios-senales/{nombre_archivo}"
        print(f"[Audio] Subido a Supabase OK: {link}")
        return link
    except Exception as exc:
        print(f"[Aviso] Error subiendo el audio de {symbol}: {type(exc).__name__}: {exc}")
        return None


def enviar_audio_telegram(audio_bytes, symbol: str) -> None:
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        print("[Audio] Falta TELEGRAM_BOT_TOKEN o TELEGRAM_CHAT_ID, no se manda el audio.")
        return
    url = f"https://api.telegram.org/bot{token}/sendAudio"
    try:
        resp = requests.post(
            url,
            data={"chat_id": chat_id, "caption": f"🔊 Explicación simple: {symbol}"},
            files={"audio": (f"{symbol}.mp3", audio_bytes, "audio/mpeg")},
            timeout=30,
        )
        result = resp.json()
        if not result.get("ok"):
            print(f"[Error Telegram audio] {result}")
        else:
            print(f"[Audio] Enviado a Telegram OK (message_id {result['result']['message_id']}).")
    except Exception as exc:
        print(f"[Error enviando audio a Telegram] {type(exc).__name__}: {exc}")



def _construir_detalle(report) -> dict:
    """Arma un resumen en JSON de todo el análisis (no solo los números del
    plan de trading), para que el backoffice pueda armar una explicación en
    lenguaje simple más adelante, sin depender de leer el texto ya armado."""
    pd_info = report.premium_discount
    tp = report.trade_plan
    return {
        "rsi_note": report.rsi_note,
        "confianza": report.confidence,
        "alerta_flujo_capital": report.capital_flow_alert,
        "invalidacion_rota": report.invalidation_broken,
        "premium_discount": (
            {"zona": pd_info.zone, "rango_bajo": pd_info.range_low, "rango_alto": pd_info.range_high}
            if pd_info else None
        ),
        "liquidez": [z.note for z in (report.liquidity_zones or [])],
        "order_blocks": [
            {"tipo": z.kind, "bottom": z.bottom, "top": z.top, "nota": z.note}
            for z in (report.ob_zones or [])
        ],
        "fvg": [{"tipo": z.kind, "bottom": z.bottom, "top": z.top} for z in (report.fvg_zones or [])],
        "objetivos_extension": (
            {str(k): v for k, v in report.extension_targets.items()} if report.extension_targets else None
        ),
        "plan": (
            {
                "rr1": tp.rr1, "rr2": tp.rr2, "rr3": tp.rr3, "fuente_tp1": tp.tp1_source,
            } if tp else None
        ),
    }


def enviar_latido() -> None:
    """Punto 1 (agregado a pedido del usuario): deja constancia de que este
    bot completó un ciclo, para que ridgecrest-pagos pueda avisar si algún
    bot deja de correr sin que nadie se entere hasta que un cliente se
    queje."""
    if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
        return
    try:
        resp = requests.post(
            f"{SUPABASE_URL}/rest/v1/heartbeats",
            json={
                "bot": BOT_NAME,
                "ultima_corrida": datetime.now(timezone.utc).isoformat(),
                "aviso_enviado": False,
            },
            headers={
                "apikey": SUPABASE_SERVICE_ROLE_KEY,
                "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}",
                "Content-Type": "application/json",
                "Prefer": "resolution=merge-duplicates",
            },
            params={"on_conflict": "bot"},
            timeout=15,
        )
        if resp.status_code >= 300:
            print(f"[Aviso] No se pudo actualizar el latido: {resp.status_code} {resp.text}")
    except Exception as exc:
        print(f"[Aviso] Error actualizando el latido: {exc}")


def guardar_senal_supabase(symbol: str, report, audio_url=None) -> None:
    if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
        return
    tp = report.trade_plan
    if tp is None:
        return
    direccion = "long" if report.bias == "alcista" else "short"
    entrada = (tp.entry_low + tp.entry_high) / 2
    payload = {
        "bot": BOT_NAME,
        "simbolo": symbol,
        "direccion": direccion,
        "entrada": entrada,
        "stop_loss": tp.stop_loss,
        "tp1": tp.take_profit_1,
        "tp2": tp.take_profit_2,
        "tp3": tp.take_profit_3,
        "confianza": report.confidence,
        "detalle": _construir_detalle(report),
    }
    if audio_url:
        payload["audio_url"] = audio_url
    try:
        resp = requests.post(
            f"{SUPABASE_URL}/rest/v1/senales",
            json=payload,
            headers={
                "apikey": SUPABASE_SERVICE_ROLE_KEY,
                "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}",
                "Content-Type": "application/json",
                "Prefer": "return=minimal",
            },
            timeout=15,
        )
        if resp.status_code >= 300:
            print(f"[Aviso] No se pudo guardar la señal de {symbol} en Supabase: {resp.status_code} {resp.text}")
    except Exception as exc:
        print(f"[Aviso] Error guardando señal de {symbol} en Supabase: {exc}")

ANALYST_PERSONA = """
Eres Nathaniel Ridge, analista principal de Ridgecrest Equities, trader institucional y analista de finanzas con
más de 20 años de experiencia en mercados financieros. Eres especialista de alto nivel en Smart Money Concepts
(SMC), Price Action e ICT (Inner Circle Trader) aplicado a acciones (perpetuos tokenizados sobre acciones de EE. UU.).
Tu análisis es riguroso, objetivo y directo: identificas la huella de las instituciones (liquidez, order blocks,
desequilibrios) y evitas términos imprecisos o lenguaje de hype. Presentas siempre: estructura y sesgo, liquidez,
zonas de oferta y demanda, un plan de trading hipotético con entrada/invalidación/objetivos/ratio riesgo:beneficio,
y una nota de gestión de riesgo. Aclaras siempre que esto es un escenario técnico automatizado, no asesoría
financiera personalizada, que estos productos son derivados tokenizados (no acciones reales) que suelen operarse
con apalancamiento, y que resultados trimestrales o noticias corporativas pueden generar gaps de apertura que
invalidan cualquier lectura técnica previa.
"""


# ==============================================================================
# 1. DESCARGA DE DATOS (Bitget API pública — /api/v2/mix/market/candles)
# ==============================================================================
VALID_BITGET_GRANULARITIES = {
    "1m", "3m", "5m", "15m", "30m",
    "1H", "4H", "6H", "12H",
    "1D", "3D", "1W", "1M",
}
LEGACY_MINUTES_TO_BITGET = {
    "1": "1m", "3": "3m", "5": "5m", "15": "15m", "30": "30m",
    "60": "1H", "120": "4H", "240": "4H", "360": "6H", "720": "12H",
    "D": "1D", "W": "1W", "M": "1M",
}


def resolve_granularity(interval: str) -> str:
    key = interval.strip()
    if key in VALID_BITGET_GRANULARITIES:
        return key
    if key in LEGACY_MINUTES_TO_BITGET:
        return LEGACY_MINUTES_TO_BITGET[key]
    raise ValueError(
        f"Intervalo '{interval}' no reconocido. Usa el formato de Bitget "
        f"({', '.join(sorted(VALID_BITGET_GRANULARITIES))}) o el formato "
        f"numérico heredado en minutos (1,3,5,15,30,60,120,240,360,720,D,W,M)."
    )


def fetch_klines(symbol: str, interval: str = "60", limit: int = 300) -> pd.DataFrame:
    granularity = resolve_granularity(interval)
    url = f"{BITGET_BASE_URL}/api/v2/mix/market/candles"
    params = {
        "symbol": symbol,
        "granularity": granularity,
        "limit": min(limit, 1000),
        "productType": BITGET_PRODUCT_TYPE,
    }
    resp = requests.get(url, params=params, timeout=20)
    resp.raise_for_status()
    data = resp.json()

    if data.get("code") != "00000":
        raise RuntimeError(f"Bitget API error para {symbol}: {data.get('msg', data)}")

    rows = data.get("data", [])
    if not rows:
        raise RuntimeError(
            f"Bitget no devolvió datos para {symbol} (productType={BITGET_PRODUCT_TYPE}). "
            f"Es posible que este símbolo no exista, haya sido retirado, o el mercado "
            f"lleve cerrado mucho tiempo — revisa https://www.bitget.com/futures/usdt/{symbol}"
        )

    # Bitget devuelve: [timestamp, open, high, low, close, baseVolume, quoteVolume]
    df = pd.DataFrame(
        rows, columns=["start", "open", "high", "low", "close", "volume", "quote_volume"]
    )
    df["timestamp"] = pd.to_datetime(df["start"].astype(np.int64), unit="ms", utc=True)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)

    df = df.sort_values("timestamp").reset_index(drop=True)
    return df[["timestamp", "open", "high", "low", "close", "volume"]]


# ==============================================================================
# 1b. HORARIO DE MERCADO Y GAPS (específico de acciones)
# ==============================================================================
def market_status(now_utc: Optional[datetime] = None) -> Dict[str, object]:
    """Bitget cotiza sus acciones de lunes 00:00 a sábado 00:00 hora de
    Nueva York (UTC-4). Fuera de esa ventana el mercado está cerrado (fin
    de semana) — no es un fallo del bot, es el comportamiento esperado."""
    now_utc = now_utc or datetime.now(timezone.utc)
    ny_time = now_utc - timedelta(hours=4)
    weekday = ny_time.weekday()  # 0=lunes ... 6=domingo
    is_closed = weekday >= MARKET_CLOSE_WEEKDAY_UTC4
    return {"is_open": not is_closed, "ny_time": ny_time, "weekday": weekday}


def detect_opening_gap(df: pd.DataFrame, atr: float, mult: float = GAP_ATR_MULT) -> Optional[str]:
    """Compara el cierre de la penúltima vela con la apertura de la última.
    Un salto mayor a `mult` veces el ATR es un gap real (fin de semana o
    noticia), no ruido intra-vela — y hay que señalarlo aparte porque
    distorsiona la lectura normal de Order Blocks/FVG."""
    if len(df) < 2 or atr <= 0:
        return None
    prev_close = float(df["close"].iloc[-2])
    last_open = float(df["open"].iloc[-1])
    gap = last_open - prev_close
    if abs(gap) < atr * mult:
        return None
    direction = "ALCISTA" if gap > 0 else "BAJISTA"
    gap_atr = abs(gap) / atr
    return (
        f"↕ GAP DE APERTURA {direction}: {prev_close:.4f} → {last_open:.4f} "
        f"({abs(gap):.4f}, {gap_atr:.1f}x ATR). Los niveles SMC/ICT previos al gap "
        f"pierden fiabilidad hasta que el precio confirme una nueva estructura."
    )


# ==============================================================================
# 2. ESTRUCTURA DE MERCADO (Swing points, BOS, CHoCH)
# ==============================================================================
@dataclass
class SwingPoint:
    index: int
    timestamp: pd.Timestamp
    price: float
    kind: Literal["high", "low"]


@dataclass
class StructureEvent:
    index: int
    timestamp: pd.Timestamp
    price: float
    kind: Literal["BOS_bullish", "BOS_bearish", "CHoCH_bullish", "CHoCH_bearish"]
    volume_confirmed: bool = True


def find_swing_points(df: pd.DataFrame, order: int = SWING_ORDER) -> List[SwingPoint]:
    swings: List[SwingPoint] = []
    highs, lows = df["high"].values, df["low"].values
    for i in range(order, len(df) - order):
        window_h = highs[i - order: i + order + 1]
        window_l = lows[i - order: i + order + 1]
        if highs[i] == window_h.max() and highs[i] == window_h[order]:
            swings.append(SwingPoint(i, df["timestamp"][i], highs[i], "high"))
        if lows[i] == window_l.min() and lows[i] == window_l[order]:
            swings.append(SwingPoint(i, df["timestamp"][i], lows[i], "low"))
    swings.sort(key=lambda s: s.index)
    return swings


def _is_break_volume_confirmed(df: pd.DataFrame, index: int, lookback: int = VOLUME_LOOKBACK,
                                mult: float = BREAK_VOLUME_CONFIRM_MULT) -> bool:
    start = max(0, index - lookback)
    prior = df["volume"].iloc[start:index]
    if prior.empty or prior.mean() == 0:
        return True
    return bool(df["volume"].iloc[index] >= prior.mean() * mult)


def detect_structure_events(df: pd.DataFrame, swings: List[SwingPoint]) -> List[StructureEvent]:
    events: List[StructureEvent] = []
    trend: Optional[str] = None
    last_high: Optional[SwingPoint] = None
    last_low: Optional[SwingPoint] = None
    for s in sorted(swings, key=lambda s: s.index):
        if s.kind == "high":
            if last_high is not None:
                if s.price > last_high.price:
                    kind = None
                    if trend == "down":
                        kind = "CHoCH_bullish"
                    elif trend == "up":
                        kind = "BOS_bullish"
                    if kind:
                        events.append(StructureEvent(
                            s.index, s.timestamp, s.price, kind,
                            volume_confirmed=_is_break_volume_confirmed(df, s.index),
                        ))
                    trend = "up"
            last_high = s
        else:
            if last_low is not None:
                if s.price < last_low.price:
                    kind = None
                    if trend == "up":
                        kind = "CHoCH_bearish"
                    elif trend == "down":
                        kind = "BOS_bearish"
                    if kind:
                        events.append(StructureEvent(
                            s.index, s.timestamp, s.price, kind,
                            volume_confirmed=_is_break_volume_confirmed(df, s.index),
                        ))
                    trend = "down"
            last_low = s
    return events


# ==============================================================================
# 3. ORDER BLOCKS, BREAKER BLOCKS, FVG Y LIQUIDEZ
# ==============================================================================
@dataclass
class Zone:
    kind: str
    top: float
    bottom: float
    index: int
    timestamp: pd.Timestamp
    note: str = ""
    mitigated: bool = False


def find_order_blocks(df: pd.DataFrame, events: List[StructureEvent], lookback: int = OB_LOOKBACK,
                       min_body: float = 0.0) -> List[Zone]:
    zones: List[Zone] = []
    for ev in events:
        start = max(0, ev.index - lookback)
        segment = df.iloc[start:ev.index]
        if segment.empty:
            continue
        bullish_event = "bullish" in ev.kind
        if bullish_event:
            bearish_candles = segment[
                (segment["close"] < segment["open"])
                & ((segment["open"] - segment["close"]) >= min_body)
            ]
            if not bearish_candles.empty:
                c = bearish_candles.iloc[-1]
                zones.append(Zone(
                    "order_block_bullish", top=c["open"], bottom=c["low"],
                    index=int(c.name), timestamp=c["timestamp"],
                    note=f"OB alcista previo a {ev.kind} ({ev.timestamp:%Y-%m-%d %H:%M})",
                ))
        else:
            bullish_candles = segment[
                (segment["close"] > segment["open"])
                & ((segment["close"] - segment["open"]) >= min_body)
            ]
            if not bullish_candles.empty:
                c = bullish_candles.iloc[-1]
                zones.append(Zone(
                    "order_block_bearish", top=c["high"], bottom=c["open"],
                    index=int(c.name), timestamp=c["timestamp"],
                    note=f"OB bajista previo a {ev.kind} ({ev.timestamp:%Y-%m-%d %H:%M})",
                ))
    return zones


def tag_breaker_blocks(df: pd.DataFrame, zones: List[Zone]) -> None:
    for z in zones:
        if z.kind not in ("order_block_bullish", "order_block_bearish"):
            continue
        after = df.iloc[z.index + 1:]
        if after.empty:
            continue
        if z.kind == "order_block_bullish":
            broken = after[after["close"] < z.bottom]
            if not broken.empty:
                z.mitigated = True
                z.note += " → posible BREAKER (bajista si el precio regresa aquí)"
        else:
            broken = after[after["close"] > z.top]
            if not broken.empty:
                z.mitigated = True
                z.note += " → posible BREAKER (alcista si el precio regresa aquí)"


def find_fair_value_gaps(df: pd.DataFrame) -> List[Zone]:
    zones: List[Zone] = []
    for i in range(2, len(df)):
        c1, c3 = df.iloc[i - 2], df.iloc[i]
        if c1["high"] < c3["low"]:
            zones.append(Zone(
                "fvg_bullish", top=c3["low"], bottom=c1["high"],
                index=i, timestamp=df.iloc[i - 1]["timestamp"],
                note="FVG alcista (desequilibrio / posible zona de reequilibrio)",
            ))
        elif c1["low"] > c3["high"]:
            zones.append(Zone(
                "fvg_bearish", top=c1["low"], bottom=c3["high"],
                index=i, timestamp=df.iloc[i - 1]["timestamp"],
                note="FVG bajista (desequilibrio / posible zona de reequilibrio)",
            ))
    return zones


def find_liquidity_pools(swings: List[SwingPoint], tolerance_pct: float = LIQUIDITY_TOLERANCE_PCT) -> List[Zone]:
    zones: List[Zone] = []
    for kind, pts in (("high", [s for s in swings if s.kind == "high"]),
                       ("low", [s for s in swings if s.kind == "low"])):
        pts = sorted(pts, key=lambda s: s.price)
        used = set()
        for i, a in enumerate(pts):
            if a.index in used:
                continue
            cluster = [a]
            for b in pts[i + 1:]:
                if abs(b.price - a.price) / a.price * 100 <= tolerance_pct:
                    cluster.append(b)
            if len(cluster) >= 2:
                for p in cluster:
                    used.add(p.index)
                avg_price = float(np.mean([p.price for p in cluster]))
                zones.append(Zone(
                    f"liquidity_{kind}", top=avg_price, bottom=avg_price,
                    index=cluster[-1].index, timestamp=cluster[-1].timestamp,
                    note=f"{len(cluster)} equal {kind}s agrupados cerca de {avg_price:.4f} "
                         f"(liquidez minorista — posible objetivo de barrido)",
                ))
    return zones


# ==============================================================================
# 4. PREMIUM / DISCOUNT PRICING + NIVELES OTE (ICT)
# ==============================================================================
@dataclass
class PremiumDiscount:
    range_high: float
    range_low: float
    pct_in_range: float
    zone: str
    impulse_up: bool
    ote_levels: Dict[float, float]


def compute_premium_discount(df: pd.DataFrame, swings: List[SwingPoint]) -> Optional[PremiumDiscount]:
    highs = [s for s in swings if s.kind == "high"]
    lows = [s for s in swings if s.kind == "low"]
    if not highs or not lows:
        return None
    last_high, last_low = highs[-1], lows[-1]
    range_high = max(last_high.price, last_low.price)
    range_low = min(last_high.price, last_low.price)
    rng = range_high - range_low
    if rng <= 0:
        return None
    price = float(df["close"].iloc[-1])
    pct = (price - range_low) / rng
    if pct > 1.0:
        extra = price - range_high
        zone = (f"EXTENSIÓN ALCISTA — {extra:.4f} por encima del último swing high "
                f"confirmado ({range_high:.4f}), equivalente al {(pct - 1) * 100:.0f}% del rango")
    elif pct < 0.0:
        extra = range_low - price
        zone = (f"EXTENSIÓN BAJISTA — {extra:.4f} por debajo del último swing low "
                f"confirmado ({range_low:.4f}), equivalente al {(-pct) * 100:.0f}% del rango")
    elif pct < 0.45:
        zone = "descuento (discount)"
    elif pct > 0.55:
        zone = "premium"
    else:
        zone = "equilibrio (50%)"
    impulse_up = last_high.index > last_low.index
    ote_levels = {}
    for fib in OTE_LEVELS:
        ote_levels[fib] = (range_high - rng * fib) if impulse_up else (range_low + rng * fib)
    return PremiumDiscount(range_high, range_low, pct, zone, impulse_up, ote_levels)


EXTENSION_RATIOS = (1.272, 1.618, 2.0)


def compute_extension_targets(pd_info: PremiumDiscount) -> Optional[Dict[float, float]]:
    rng = pd_info.range_high - pd_info.range_low
    if rng <= 0:
        return None
    targets: Dict[float, float] = {}
    if pd_info.pct_in_range > 1.0:
        for ratio in EXTENSION_RATIOS:
            targets[ratio] = pd_info.range_high + rng * (ratio - 1.0)
    elif pd_info.pct_in_range < 0.0:
        for ratio in EXTENSION_RATIOS:
            targets[ratio] = pd_info.range_low - rng * (ratio - 1.0)
    else:
        return None
    return targets


# ==============================================================================
# 5. INDICADORES DE CONFLUENCIA: RSI Y ATR
# ==============================================================================
def calculate_rsi(df: pd.DataFrame, period: int = RSI_PERIOD) -> float:
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return float(rsi.iloc[-1]) if not rsi.empty and not pd.isna(rsi.iloc[-1]) else 50.0


def calculate_atr(df: pd.DataFrame, period: int = ATR_PERIOD) -> float:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    true_range = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr = true_range.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    if not atr.empty and not pd.isna(atr.iloc[-1]):
        return float(atr.iloc[-1])
    return float((high - low).mean())


# ==============================================================================
# 6. ALERTA DE FLUJO DE CAPITAL (anomalía de volumen)
# ==============================================================================
def detect_capital_flow_alert(df: pd.DataFrame, lookback: int = VOLUME_LOOKBACK,
                               z_threshold: float = VOLUME_ZSCORE_THRESHOLD) -> Optional[str]:
    vol = df["volume"]
    if len(vol) < lookback + 1 or vol.sum() == 0:
        return None
    recent = vol.iloc[-(lookback + 1):-1]
    mean, std = recent.mean(), recent.std()
    if std == 0 or pd.isna(std):
        return None
    current = df.iloc[-1]
    z = (current["volume"] - mean) / std
    if z >= z_threshold:
        direction = "ENTRADA fuerte de capital (compra)" if current["close"] > current["open"] else "SALIDA fuerte de capital (venta)"
        return f"⚡ {direction} detectada — volumen {z:.1f}σ sobre el promedio de {lookback} velas."
    return None


# ==============================================================================
# 7. PLAN DE TRADING HIPOTÉTICO
# ==============================================================================
@dataclass
class TradePlan:
    entry_low: float
    entry_high: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    take_profit_3: float
    rr1: float
    rr2: float
    rr3: float
    tp1_source: str


MAX_TP1_DISTANCE_MULT = 2.0


def compute_trade_plan(bias: str, price: float, pd_info: Optional[PremiumDiscount],
                        zones: List[Zone], atr: float) -> Optional[TradePlan]:
    if pd_info is None or bias not in ("alcista", "bajista"):
        return None
    if pd_info.pct_in_range < 0.0 or pd_info.pct_in_range > 1.0:
        return None
    if bias == "alcista" and not pd_info.impulse_up:
        return None
    if bias == "bajista" and pd_info.impulse_up:
        return None

    level_618 = pd_info.ote_levels[0.618]
    level_79 = pd_info.ote_levels[0.79]
    entry_low, entry_high = sorted([level_618, level_79])
    entry_mid = (entry_low + entry_high) / 2
    rng = pd_info.range_high - pd_info.range_low
    buffer = max(rng * 0.02, atr * ATR_BUFFER_MULT)
    max_tp1_distance = rng * MAX_TP1_DISTANCE_MULT
    tp1_source = "límite del rango"

    if bias == "alcista":
        sl = pd_info.range_low - buffer
        targets = sorted(
            [z for z in zones if z.kind == "liquidity_high" and z.top > entry_mid],
            key=lambda z: z.top,
        )
        if targets and (targets[0].top - entry_mid) <= max_tp1_distance:
            tp1 = targets[0].top
            tp1_source = f"liquidez en {tp1:.4f}"
        else:
            tp1 = pd_info.range_high
        tp2 = pd_info.range_high
        risk = entry_mid - sl
        tp3 = entry_mid + risk * 3
    else:
        sl = pd_info.range_high + buffer
        targets = sorted(
            [z for z in zones if z.kind == "liquidity_low" and z.bottom < entry_mid],
            key=lambda z: -z.bottom,
        )
        if targets and (entry_mid - targets[0].bottom) <= max_tp1_distance:
            tp1 = targets[0].bottom
            tp1_source = f"liquidez en {tp1:.4f}"
        else:
            tp1 = pd_info.range_low
        tp2 = pd_info.range_low
        risk = sl - entry_mid
        tp3 = entry_mid - risk * 3

    if risk <= 0:
        return None

    rr1 = abs(tp1 - entry_mid) / risk
    if rr1 < 0.8:
        return None
    rr2 = abs(tp2 - entry_mid) / risk
    rr3 = abs(tp3 - entry_mid) / risk

    return TradePlan(entry_low, entry_high, sl, tp1, tp2, tp3, rr1, rr2, rr3, tp1_source)


# ==============================================================================
# 8. GENERACIÓN DE SEÑAL / SESGO
# ==============================================================================
@dataclass
class SignalReport:
    symbol: str
    price: float
    bias: str
    confidence: str
    last_event: Optional[StructureEvent]
    liquidity_zones: List[Zone]
    ob_zones: List[Zone]
    fvg_zones: List[Zone]
    invalidation: Optional[float]
    invalidation_broken: bool
    rsi: float
    rsi_note: str
    atr: float
    premium_discount: Optional[PremiumDiscount]
    extension_targets: Optional[Dict[float, float]]
    trade_plan: Optional[TradePlan]
    capital_flow_alert: Optional[str]
    gap_alert: Optional[str]
    market_open: bool
    generated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


def find_structural_invalidation(swings: List[SwingPoint], last_event: Optional[StructureEvent],
                                  bias: str) -> Optional[float]:
    if last_event is None or bias not in ("alcista", "bajista"):
        return None
    if bias == "alcista":
        candidates = [s for s in swings if s.kind == "low" and s.index <= last_event.index]
    else:
        candidates = [s for s in swings if s.kind == "high" and s.index <= last_event.index]
    if not candidates:
        return last_event.price
    return candidates[-1].price


def generate_signal(symbol: str, df: pd.DataFrame, events: List[StructureEvent], zones: List[Zone]) -> SignalReport:
    price = float(df["close"].iloc[-1])
    last_event = events[-1] if events else None

    if last_event is None:
        bias, confidence = "neutral / rango", "baja"
    elif "bullish" in last_event.kind:
        bias, confidence = "alcista", "media-alta" if last_event.kind.startswith("BOS") else "media"
    else:
        bias, confidence = "bajista", "media-alta" if last_event.kind.startswith("BOS") else "media"

    volume_note = ""
    if last_event is not None and not last_event.volume_confirmed:
        confidence = "media" if confidence == "media-alta" else "baja"
        volume_note = " (sin confirmación de volumen — señal más débil de lo habitual)"

    swings_all = find_swing_points(df)
    invalidation = find_structural_invalidation(swings_all, last_event, bias)

    invalidation_broken = False
    if invalidation is not None:
        if bias == "bajista" and price > invalidation:
            invalidation_broken = True
        elif bias == "alcista" and price < invalidation:
            invalidation_broken = True
    if invalidation_broken:
        confidence = "muy baja — posible lectura obsoleta"

    rsi = calculate_rsi(df)
    atr = calculate_atr(df)
    rsi_note = f"{rsi:.1f} (neutral)"
    if bias == "alcista" and not invalidation_broken:
        if rsi < 35:
            confidence, rsi_note = "alta", f"{rsi:.1f} (sobreventa — refuerza el sesgo alcista)"
        elif rsi > 75:
            confidence, rsi_note = "baja", f"{rsi:.1f} (sobrecompra — precaución, posible agotamiento)"
    elif bias == "bajista" and not invalidation_broken:
        if rsi > 65:
            confidence, rsi_note = "alta", f"{rsi:.1f} (sobrecompra — refuerza el sesgo bajista)"
        elif rsi < 25:
            confidence, rsi_note = "baja", f"{rsi:.1f} (sobreventa — precaución, posible agotamiento)"
    rsi_note += volume_note

    def zone_mid(z: Zone) -> float:
        return (z.top + z.bottom) / 2

    liquidity_zones = sorted(
        [z for z in zones if z.kind.startswith("liquidity_")],
        key=lambda z: abs(zone_mid(z) - price),
    )
    ob_zones = sorted(
        [z for z in zones if "order_block" in z.kind and not z.mitigated],
        key=lambda z: abs(zone_mid(z) - price),
    )
    fvg_zones = sorted(
        [z for z in zones if z.kind.startswith("fvg_")],
        key=lambda z: abs(zone_mid(z) - price),
    )

    pd_info = compute_premium_discount(df, swings_all)
    extension_targets = compute_extension_targets(pd_info) if pd_info else None

    status = market_status()
    gap_alert = detect_opening_gap(df, atr)
    # Con el mercado cerrado (fin de semana) o justo tras un gap grande, un
    # plan de entrada hipotético pierde sentido: no hay forma de ejecutar
    # nada hasta la reapertura, y el gap puede haber invalidado el rango.
    trade_plan = None
    if not invalidation_broken and status["is_open"] and gap_alert is None:
        trade_plan = compute_trade_plan(bias, price, pd_info, zones, atr)
    capital_flow_alert = detect_capital_flow_alert(df)

    return SignalReport(
        symbol=symbol, price=price, bias=bias, confidence=confidence,
        last_event=last_event, liquidity_zones=liquidity_zones[:4],
        ob_zones=ob_zones[:3],
        fvg_zones=fvg_zones[:3], invalidation=invalidation,
        invalidation_broken=invalidation_broken,
        rsi=rsi, rsi_note=rsi_note, atr=atr, premium_discount=pd_info,
        extension_targets=extension_targets,
        trade_plan=trade_plan, capital_flow_alert=capital_flow_alert,
        gap_alert=gap_alert, market_open=bool(status["is_open"]),
    )


# ==============================================================================
# 9. NARRATIVA — plantilla determinista
# ==============================================================================
def render_template_report(report: SignalReport, interval: str) -> str:
    L = []
    label_prefix = f"[{DEPLOY_LABEL}] " if DEPLOY_LABEL else ""
    L.append(f"═══ {label_prefix}RIDGECREST EQUITIES | NATHANIEL RIDGE | {report.symbol} | TF {interval} ═══")
    L.append(formatear_fecha_con_hora_local(report.generated_at))
    L.append("")

    if not report.market_open:
        L.append(
            "🔒 MERCADO CERRADO (fin de semana bursátil). Esta lectura corresponde "
            "al último cierre disponible — no persigas el precio hasta la "
            "reapertura y vigila el gap de apertura del lunes."
        )
        L.append("")

    if report.gap_alert:
        L.append(report.gap_alert)
        L.append("")

    if report.capital_flow_alert:
        L.append(report.capital_flow_alert)
        L.append("")

    if report.invalidation_broken:
        L.append(
            "🚨 ATENCIÓN: el precio ya cerró más allá del nivel de invalidación de "
            "esta lectura. La estructura probablemente ya está cambiando de nuevo — "
            "trata este sesgo como OBSOLETO hasta la próxima confirmación."
        )
        L.append("")

    L.append("1) ESTRUCTURA Y SESGO")
    L.append(f"   Precio: {report.price:.4f} | Sesgo: {report.bias.upper()} | Confianza: {report.confidence}")
    L.append(f"   RSI({RSI_PERIOD}): {report.rsi_note}")
    L.append(f"   ATR({ATR_PERIOD}): {report.atr:.4f} (volatilidad reciente, usada para la invalidación)")
    if report.last_event:
        vol_tag = "✓ volumen" if report.last_event.volume_confirmed else "✗ sin volumen"
        L.append(f"   Último evento: {report.last_event.kind} en {report.last_event.price:.4f} "
                  f"({report.last_event.timestamp:%m-%d %H:%M}) [{vol_tag}]")
        L.append(f"   Invalidación de esta lectura: {report.invalidation:.4f}")
    else:
        L.append("   Sin eventos de estructura claros todavía.")

    if report.premium_discount:
        pd_i = report.premium_discount
        L.append("")
        L.append("2) PREMIUM / DISCOUNT (ICT)")
        L.append(f"   Rango activo (último swing): {pd_i.range_low:.4f} — {pd_i.range_high:.4f}")
        L.append(f"   Precio: {pd_i.zone}")
        levels_txt = " | ".join(f"{fib}: {p:.4f}" for fib, p in pd_i.ote_levels.items())
        L.append(f"   Niveles OTE: {levels_txt}")

    L.append("")
    L.append("3) LIQUIDEZ")
    if report.liquidity_zones:
        for z in report.liquidity_zones:
            L.append(f"   - {z.note}")
    else:
        L.append("   - Sin agrupaciones claras de liquidez cerca del rango analizado.")

    L.append("")
    L.append("4) OFERTA / DEMANDA (Order Blocks & FVG)")
    if report.ob_zones:
        for z in report.ob_zones:
            L.append(f"   - [{z.kind}] {z.bottom:.4f}-{z.top:.4f} | {z.note}")
    else:
        L.append("   - Sin Order Blocks no mitigados relevantes.")
    if report.fvg_zones:
        for z in report.fvg_zones:
            L.append(f"   - [{z.kind}] {z.bottom:.4f}-{z.top:.4f}")

    if report.trade_plan:
        tp = report.trade_plan
        L.append("")
        L.append("5) PLAN DE TRADING HIPOTÉTICO")
        L.append(f"   Entrada (POI): {tp.entry_low:.4f} — {tp.entry_high:.4f}")
        L.append(f"   Invalidación (SL): {tp.stop_loss:.4f}")
        L.append(f"   TP1: {tp.take_profit_1:.4f} (R:R {tp.rr1:.2f}) — fuente: {tp.tp1_source}")
        L.append(f"   TP2: {tp.take_profit_2:.4f} (R:R {tp.rr2:.2f})")
        L.append(f"   TP3: {tp.take_profit_3:.4f} (R:R {tp.rr3:.2f})")
    elif report.extension_targets and report.market_open and not report.gap_alert:
        L.append("")
        L.append("5) OBJETIVOS DE EXTENSIÓN (movimiento fuerte, sin retroceso aún)")
        L.append("   No hay entrada recomendada aquí — el precio ya se movió sin retroceso, "
                  "entrar ahora implica peor R:R y mayor riesgo de reversión. Estos son "
                  "niveles de referencia por si el movimiento continúa:")
        for ratio, level in report.extension_targets.items():
            L.append(f"   Extensión {ratio}: {level:.4f}")
        L.append("   Si buscas entrar, lo prudente es esperar un retroceso hacia las zonas "
                  "de la sección 4, no perseguir el precio aquí.")
    else:
        L.append("")
        L.append("5) PLAN DE TRADING HIPOTÉTICO")
        if not report.market_open:
            L.append("   Sin plan mientras el mercado esté cerrado — espera la reapertura "
                      "y confirma que el gap de apertura no cambió la estructura.")
        elif report.gap_alert:
            L.append("   Sin plan justo después de un gap — espera a que se formen velas "
                      "nuevas que confirmen estructura post-gap antes de operar.")
        else:
            L.append("   Sin plan accionable en este momento (precio en extensión fuera del "
                      "último rango confirmado, o el Riesgo:Beneficio disponible no es "
                      "favorable). Se recomienda esperar un retroceso o una nueva "
                      "confirmación de estructura.")

    L.append("")
    L.append(
        "⚠ Escenario técnico automatizado (SMC/ICT simplificado) sobre un derivado "
        "tokenizado, no la acción real. No es asesoría financiera ni garantía de "
        "resultados. Suele operarse con apalancamiento y resultados trimestrales o "
        "noticias corporativas pueden generar gaps que invaliden esta lectura. "
        "Define tu propia gestión de riesgo."
    )
    return "\n".join(L)


# ==============================================================================
# 10. TELEGRAM
# ==============================================================================
def send_telegram_message(text: str) -> None:
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        print("[Aviso] Falta TELEGRAM_BOT_TOKEN o TELEGRAM_CHAT_ID. No se envió mensaje.")
        return
    if len(text) > 4000:
        text = text[:3990] + "\n...(recortado)"
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    try:
        resp = requests.post(url, data={"chat_id": chat_id, "text": text}, timeout=15)
        resp.raise_for_status()
        result = resp.json()
        if not result.get("ok"):
            print(f"[Error Telegram] {result}")
    except Exception as exc:
        print(f"[Error enviando a Telegram] {exc}")


# ==============================================================================
# 11. CLI
# ==============================================================================
def analyze_symbol(symbol: str, interval: str, limit: int):
    df = fetch_klines(symbol=symbol, interval=interval, limit=limit)
    atr = calculate_atr(df)
    min_body = atr * MIN_OB_BODY_ATR_MULT
    swings = find_swing_points(df)
    events = detect_structure_events(df, swings)
    zones = []
    zones += find_order_blocks(df, events, min_body=min_body)
    tag_breaker_blocks(df, zones)
    zones += find_fair_value_gaps(df)
    zones += find_liquidity_pools(swings)
    report = generate_signal(symbol, df, events, zones)
    return render_template_report(report, interval), report


def run_once(symbols: List[str], interval: str, limit: int, use_telegram: bool) -> None:
    for i, symbol in enumerate(symbols):
        try:
            text, report = analyze_symbol(symbol, interval, limit)
            print(text)
            if use_telegram:
                send_telegram_message(text)
            audio_url = None
            if report.trade_plan is not None:
                print(f"[Audio] {symbol} tiene plan de trading, generando explicación hablada...")
                audio = generar_audio_explicacion(construir_explicacion_hablada(symbol, report))
                if audio:
                    if use_telegram:
                        enviar_audio_telegram(audio, symbol)
                    audio_url = subir_audio_supabase(audio, symbol)
            guardar_senal_supabase(symbol, report, audio_url)
        except Exception as exc:
            print(f"[Error analizando {symbol}] {exc}")
        if i < len(symbols) - 1:
            time.sleep(API_CALL_DELAY_SECONDS)
    enviar_latido()


def seconds_until_next_aligned_run(watch_minutes: int) -> float:
    now = datetime.now(timezone.utc)
    total_minutes_since_midnight = now.hour * 60 + now.minute
    next_slot = ((total_minutes_since_midnight // watch_minutes) + 1) * watch_minutes
    next_run = now.replace(hour=0, minute=0, second=0, microsecond=0)
    next_run += pd.Timedelta(minutes=next_slot)
    return max((next_run - now).total_seconds(), 0)


def main():
    parser = argparse.ArgumentParser(description="Analista SMC/ICT de acciones (Bitget USDT-FUTURES)")
    parser.add_argument("--symbols", default=None,
                         help="Lista separada por comas, ej. 'TSLAUSDT,NVDAUSDT'. Si no se especifica, usa SYMBOLS.")
    parser.add_argument("--interval", default="60",
                         help="Timeframe: formato Bitget (1m,5m,15m,30m,1H,4H,6H,12H,1D,3D,1W,1M) "
                              "o formato numérico heredado en minutos (1,3,5,15,30,60,120,240,360,720,D,W,M).")
    parser.add_argument("--limit", type=int, default=300, help="Número de velas a analizar (máx 1000)")
    parser.add_argument("--telegram", action="store_true", help="Envía el reporte a Telegram")
    parser.add_argument("--watch", type=int, default=0, help="Repite cada N minutos (0 = una sola vez)")
    parser.add_argument("--no-align", action="store_true",
                         help="Desactiva la alineación al reloj.")
    args = parser.parse_args()

    symbols = [s.strip().upper() for s in args.symbols.split(",")] if args.symbols else SYMBOLS

    print(
        "⚠ Herramienta educativa. No ejecuta órdenes ni gestiona fondos. No es asesoría financiera.\n"
        f"Exchange: Bitget | Product type: {BITGET_PRODUCT_TYPE} (acciones tokenizadas, no acciones reales)\n"
        f"Símbolos candidatos ({len(symbols)}): {', '.join(symbols)}\n"
    )

    if args.watch <= 0:
        run_once(symbols, args.interval, args.limit, args.telegram)
        return

    while True:
        if not args.no_align:
            wait = seconds_until_next_aligned_run(args.watch)
            if wait > 1:
                mins = int(wait // 60)
                print(f"Esperando {mins} min para alinear el próximo reporte a un horario redondo...\n")
                time.sleep(wait)
        try:
            run_once(symbols, args.interval, args.limit, args.telegram)
        except KeyboardInterrupt:
            print("\nDetenido por el usuario.")
            sys.exit(0)
        except Exception as exc:
            print(f"[Error en el ciclo de análisis] {exc}")

        if args.no_align:
            print(f"\n(Próxima actualización en {args.watch} min)\n")
            time.sleep(args.watch * 60)
        else:
            time.sleep(5)


if __name__ == "__main__":
    main()
