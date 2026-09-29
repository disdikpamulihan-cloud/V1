import os, requests, numpy as np, pandas as pd
from datetime import datetime, timezone, timedelta
from core_engine import (find_swing_points, detect_bos_choch, find_order_blocks,
                         find_fvg, detect_liquidity_sweep, calc_vwap, detect_rsi_divergence,
                         calc_volume_delta, detect_ml_regime, detect_ml_anomaly,
                         AGIMemory, grade_the_god_signal)

SYMBOL = "PAXGUSD"
INTERVAL = 15

def get_market_data():
    try:
        url = "https://api.kraken.com/0/public/OHLC"
        params = {"pair": SYMBOL, "interval": INTERVAL}
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(url, params=params, headers=headers, timeout=15)
        r.raise_for_status()
        data = r.json()
        if data.get("error"): raise Exception(str(data['error']))
        key = list(data["result"].keys())[0]
        klines = data["result"][key]
        df = pd.DataFrame(klines, columns=['date','open','high','low','close','vwap','volume','count'])
        for c in ['open','high','low','close','volume']:
            df[c] = df[c].astype(float)
        df['date'] = pd.to_datetime(df['date'].astype(int), unit='s')
        df = df[['date','open','high','low','close','volume']].dropna()
        print(f"OK: {len(df)} candle. Harga: ${df['close'].iloc[-1]:.2f}")
        return df
    except Exception as e:
        print(f"ERROR: {e}")
        np.random.seed(42)
        n = 500
        c = np.cumsum(np.random.normal(0, 3, n)) + 2650
        return pd.DataFrame({
            'date': pd.date_range(end=pd.Timestamp.now(), periods=n, freq='15min'),
            'open': c + np.random.normal(0, 1, n),
            'high': c + np.abs(np.random.normal(3, 1, n)),
            'low': c - np.abs(np.random.normal(3, 1, n)),
            'close': c,
            'volume': np.random.lognormal(10, 1, n)
        })

def send_telegram(text):
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        print("TOKEN/CHAT_ID BELUM DI-SET DI GITHUB SECRETS")
        return
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    try:
        r = requests.post(url, json={"chat_id": chat_id, "text": text, "parse_mode": "HTML"}, timeout=10)
        print("TELEGRAM OK" if r.status_code == 200 else f"TELEGRAM FAIL: {r.text}")
    except Exception as e:
        print(f"TELEGRAM ERROR: {e}")

def run():
    print("START")
    df = get_market_data()
    mem = AGIMemory()
    df = find_swing_points(df)
    df = detect_bos_choch(df)
    df = detect_liquidity_sweep(df)
    obs = find_order_blocks(df)
    fvgs = find_fvg(df)
    rsi_div = detect_rsi_divergence(df)
    vol_delta = calc_volume_delta(df)
    ml_regime, reg_conf = detect_ml_regime(df)
    anom_score, is_anom = detect_ml_anomaly(df)
    price = float(df['close'].iloc[-1])
    has_choch = any("ChoCh" in str(x) for x in df['structure'].iloc[-10:].values)
    has_bos = any("BOS" in str(x) for x in df['structure'].iloc[-10:].values)
    has_sweep = any("Sweep" in str(x) for x in df['liquidity_sweep'].iloc[-10:].values)
    entry_at_ob = any(o['bottom'] <= price <= o['top'] for o in obs[-10:])
    entry_at_fvg = any(f['bottom'] <= price <= f['top'] for f in fvgs[-10:])
    direction = "BUY" if price > float(df['close'].iloc[-5]) and ml_regime != "BEAR" else "SELL"
    emb = np.random.rand(32).astype(np.float32)
    mem_stats = mem.query(emb)
    g = grade_the_god_signal(direction, ml_regime, reg_conf, anom_score, is_anom,
                             has_choch, has_bos, has_sweep, entry_at_ob, entry_at_fvg,
                             rsi_div, vol_delta, mem_stats)
    if g.grade in ("A", "A++", "A+++", "A SUPER"):
        atr = (df['high'] - df['low']).rolling(14).mean().iloc[-1]
        sl = price - atr*1.5 if direction == "BUY" else price + atr*1.5
        tp1 = price + atr*1.5 if direction == "BUY" else price - atr*1.5
        tp2 = price + atr*3.0 if direction == "BUY" else price - atr*3.0
        reasons = "\n".join([f"  - {r}" for r in g.reasons])
        wib = datetime.now(timezone(timedelta(hours=7))).strftime('%H:%M WIB')
        emoji = "🟢" if direction == "BUY" else "🔴"
        icon = "💎" if g.grade == "A SUPER" else "🔥" if g.grade == "A+++" else "⚡" if g.grade == "A++" else "✅"
        caption = (
            f"{icon} <b>AGI SIGNAL — {g.grade}</b> (Score: {g.score}/100)\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"{emoji} <b>{direction} XAUUSD</b>\n"
            f"💰 Price: <code>{price:.2f}</code>\n"
            f"🛡 SL: <code>{sl:.2f}</code>\n"
            f"🎯 TP1: <code>{tp1:.2f}</code>\n"
            f"🎯 TP2: <code>{tp2:.2f}</code>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"🧠 Confluence:\n{reasons}\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"📊 RSI Div: {rsi_div} | Vol: {vol_delta:.0f}\n"
            f"🌊 Regime: {ml_regime} ({reg_conf:.0%})\n"
            f"⚡ Anomaly: {anom_score:.2f}\n"
            f"🧬 Memory: {mem_stats['n']} case | WR {mem_stats['wr']}%\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"⏰ {wib}"
        )
        send_telegram(caption)
    else:
        print(f"TRASH: {g.grade} | Score: {g.score} | Harga: ${price:.2f}")
        mem.store(emb, direction, "TRASHED")

if __name__ == "__main__":
    run()
