import os, requests, numpy as np, pandas as pd, yfinance as yf
from datetime import datetime, timezone, timedelta
from core_engine import (find_swing_points, detect_bos_choch, find_order_blocks, 
                         find_fvg, detect_liquidity_sweep, calc_vwap, detect_rsi_divergence, 
                         calc_volume_delta, detect_ml_regime, detect_ml_anomaly, 
                         AGIMemory, grade_the_god_signal)

# KONFIGURASI (JANGAN DIUBAH KALAU GAK PAHAM)
SYMBOL = "XAUUSD=X"  # Gold
INTERVAL = "15m"     # Timeframe
PERIOD = "60d"       # Data history

def get_market_data():
    try:
        df = yf.download(SYMBOL, period=PERIOD, interval=INTERVAL, progress=False)
        if df.empty or len(df) < 100: raise Exception("Data kosong")
        df = df.reset_index()
        # Flatten multi-index columns jika ada
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]
        return df[['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume']].rename(columns={'Datetime': 'date'})
    except Exception as e:
        print(f"⚠️ yfinance gagal: {e}. Generating fallback data...")
        # Fallback dummy data biar bot gak crash
        np.random.seed(42)
        length = 300
        close = np.cumsum(np.random.normal(0, 2, length)) + 2000
        return pd.DataFrame({
            'date': pd.date_range(end=pd.Timestamp.now(), periods=length, freq='15min'),
            'open': close + np.random.normal(0, 1, length),
            'high': close + np.abs(np.random.normal(2, 1, length)),
            'low': close - np.abs(np.random.normal(2, 1, length)),
            'close': close,
            'volume': np.random.lognormal(10, 1, length)
        })

def send_telegram(text):
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        print("❌ TELEGRAM_BOT_TOKEN atau TELEGRAM_CHAT_ID belum di-set di GitHub Secrets!")
        print("Caption yang seharusnya dikirim:\n", text)
        return
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": text, "parse_mode": "HTML"}
    try:
        r = requests.post(url, json=payload, timeout=10)
        if r.status_code == 200: print("✅ Sinyal berhasil dikirim ke Telegram!")
        else: print(f"❌ Telegram API Error: {r.text}")
    except Exception as e:
        print(f"❌ Gagal kirim Telegram: {e}")

def run_god_pipeline():
    print("🚀 Initializing AGI God-Tier Pipeline...")
    df = get_market_data()
    mem = AGIMemory()
    
    # 1. Process Structure & Quant
    df = find_swing_points(df)
    df = detect_bos_choch(df)
    df = detect_liquidity_sweep(df)
    obs = find_order_blocks(df)
    fvgs = find_fvg(df)
    vwap = calc_vwap(df)
    rsi_div = detect_rsi_divergence(df)
    vol_delta = calc_volume_delta(df)
    
    # 2. Process ML
    ml_regime, reg_conf = detect_ml_regime(df)
    anom_score, is_anom = detect_ml_anomaly(df)
    
    # 3. Evaluate Current State
    price = df['close'].iloc[-1]
    has_choch = any("ChoCh" in str(x) for x in df['structure'].iloc[-5:].values)
    has_bos = any("BOS" in str(x) for x in df['structure'].iloc[-5:].values)
    has_sweep = any("Sweep" in str(x) for x in df['liquidity_sweep'].iloc[-5:].values)
    
    entry_at_ob = any(ob['bottom'] <= price <= ob['top'] for ob in obs[-5:])
    entry_at_fvg = any(fvg['bottom'] <= price <= fvg['top'] for fvg in fvgs[-5:])
    
    direction = "BUY" if price > df['close'].iloc[-2] and ml_regime != "BEAR" else "SELL"
    
    # 4. Memory Query (Dummy embedding for now, can be upgraded later)
    emb = np.random.rand(32).astype(np.float32) 
    mem_stats = mem.query(emb)
    
    # 5. GRADING
    grade_obj = grade_the_god_signal(
        direction=direction, ml_regime=ml_regime, reg_conf=reg_conf,
        anom_score=anom_score, is_anom=is_anom, has_choch=has_choch, 
        has_bos=has_bos, has_sweep=has_sweep, entry_at_ob=entry_at_ob, 
        entry_at_fvg=entry_at_fvg, rsi_div=rsi_div, vol_delta=vol_delta, 
        mem_stats=mem_stats
    )
    
    # 6. OUTPUT & TELEGRAM
    if "A" in grade_obj.grade:
        sl = price - 15 if direction == "BUY" else price + 15
        tp1 = price + 20 if direction == "BUY" else price - 20
        tp2 = price + 40 if direction == "BUY" else price - 40
        
        reasons_text = "\n".join([f"  • {r}" for r in grade_obj.reasons])
        wib_time = datetime.now(timezone(timedelta(hours=7))).strftime('%H:%M WIB')
        
        caption = f"""
💎 <b>AGI GOD-TIER SIGNAL</b> 💎
━━━━━━━━━━━━━━━━━━━━━━
🏆 <b>GRADE: {grade_obj.grade}</b> (Score: {grade_obj.score}/100)
🧭 <b>Direction:</b> {"🟢 BUY" if direction == "BUY" else "🔴 SELL"}
💰 <b>Price:</b> <code>{price:.2f}</code>

🛡 <b>SL:</b> <code>{sl:.2f}</code>
🎯 <b>TP1:</b> <code>{tp1:.2f}</code>
🎯 <b>TP2:</b> <code>{tp2:.2f}</code>

━━━━━━━━━━━━━━━━━━━━━━
🧠 <b>CONFLUENCE:</b>
{reasons_text}

📊 <b>QUANT & ML:</b>
  • RSI Div: {rsi_div}
  • Vol Delta: {vol_delta:.0f}
  • ML Regime: {ml_regime} ({reg_conf:.0%})
  • Anomaly: {anom_score:.2f} {'⚠️' if is_anom else '✅'}
  
🧬 <b>MEMORY:</b>
  • Cases: {mem_stats['n']} | WR: {mem_stats['wr']}%
━━━━━━━━━━━━━━━━━━━━━━
⏰ {wib_time} | {SYMBOL}
        """
        send_telegram(caption)
    else:
        print(f"🗑 Signal Trashed. Grade: {grade_obj.grade} | Score: {grade_obj.score}")
        # Simpan ke memory biar bot belajar dari sinyal yang dibuang
        mem.store(emb, direction, "TRASHED")

if __name__ == "__main__":
    run_god_pipeline()
