import os, requests, numpy as np, pandas as pd
from datetime import datetime, timezone, timedelta
from core_engine import (find_swing_points, detect_bos_choch, find_order_blocks, 
                         find_fvg, detect_liquidity_sweep, calc_vwap, detect_rsi_divergence, 
                         calc_volume_delta, detect_ml_regime, detect_ml_anomaly, 
                         AGIMemory, grade_the_god_signal)

# ==============================================================================
# KONFIGURASI
# ==============================================================================
SYMBOL = "PAXGUSDT"  # Pax Gold (Sangat dekat dengan harga Spot XAUUSD MT5)
INTERVAL = "15"      # Bybit menggunakan format angka untuk menit (15 = 15 menit)
LIMIT = 500          # 500 candle

# ==============================================================================
# DATA FETCHER (BYBIT PUBLIC API - TIDAK BLOKIR GITHUB ACTIONS)
# ==============================================================================
def get_market_data():
    try:
        print(f"📡 Mengambil data market untuk {SYMBOL} dari Bybit...")
        # Bybit V5 Public API Endpoint
        url = "https://api.bybit.com/v5/market/kline"
        params = {
            "category": "spot",
            "symbol": SYMBOL,
            "interval": INTERVAL,
            "limit": LIMIT
        }
        
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()
        
        if data.get("retCode") != 0:
            raise Exception(f"Bybit API Error: {data.get('retMsg')}")
        
        # Bybit mengembalikan data dari yang TERBARU ke TERLAMA, jadi kita harus reverse
        klines = data["result"]["list"]
        klines.reverse() 
        
        df = pd.DataFrame(klines, columns=[
            'open_time', 'Open', 'High', 'Low', 'Close', 'Volume', 'turnover'
        ])
        
        # Konversi tipe data
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            df[col] = df[col].astype(float)
            
        df['date'] = pd.to_datetime(df['open_time'].astype(int), unit='ms')
        df = df[['date', 'Open', 'High', 'Low', 'Close', 'Volume']].dropna()
        
        print(f"✅ Data berhasil diambil: {len(df)} candle. Harga terakhir: ${df['Close'].iloc[-1]:.2f}")
        return df
        
    except Exception as e:
        print(f"❌ Gagal mengambil data dari Bybit: {e}")
        print("⚠️ PERINGATAN: Menggunakan data DUMMY. Sinyal TIDAK AKURAT!")
        
        # Fallback darurat jika semua API diblokir (Harga realistis ~2650)
        np.random.seed(42)
        length = 500
        close = np.cumsum(np.random.normal(0, 3, length)) + 2650
        return pd.DataFrame({
            'date': pd.date_range(end=pd.Timestamp.now(), periods=length, freq='15min'),
            'Open': close + np.random.normal(0, 1, length),
            'High': close + np.abs(np.random.normal(3, 1, length)),
            'Low': close - np.abs(np.random.normal(3, 1, length)),
            'Close': close,
            'Volume': np.random.lognormal(10, 1, length)
        })

# ==============================================================================
# TELEGRAM SENDER
# ==============================================================================
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
        if r.status_code == 200: 
            print("✅ Sinyal berhasil dikirim ke Telegram!")
        else: 
            print(f"❌ Telegram API Error: {r.text}")
    except Exception as e:
        print(f"❌ Gagal kirim Telegram: {e}")

# ==============================================================================
# MAIN PIPELINE
# ==============================================================================
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
    price = float(df['Close'].iloc[-1])
    has_choch = any("ChoCh" in str(x) for x in df['structure'].iloc[-10:].values)
    has_bos = any("BOS" in str(x) for x in df['structure'].iloc[-10:].values)
    has_sweep = any("Sweep" in str(x) for x in df['liquidity_sweep'].iloc[-10:].values)
    
    entry_at_ob = any(ob['bottom'] <= price <= ob['top'] for ob in obs[-10:])
    entry_at_fvg = any(fvg['bottom'] <= price <= fvg['top'] for fvg in fvgs[-10:])
    
    direction = "BUY" if price > float(df['Close'].iloc[-5]) and ml_regime != "BEAR" else "SELL"
    
    # 4. Memory Query
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
        atr = (df['High'] - df['Low']).rolling(14).mean().iloc[-1]
        sl = price - (atr * 1.5) if direction == "BUY" else price + (atr * 1.5)
        tp1 = price + (atr * 1.5) if direction == "BUY" else price - (atr * 1.5)
        tp2 = price + (atr * 3.0) if direction == "BUY" else price - (atr * 3.0)
        
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
⏰ {wib_time} | XAUUSD (via PAXG)
        """
        send_telegram(caption)
    else:
        print(f"🗑 Signal Trashed. Grade: {grade_obj.grade} | Score: {grade_obj.score}")
        print(f"📊 Harga Saat Ini: ${price:.2f} | Regime: {ml_regime}")
        mem.store(emb, direction, "TRASHED")

if __name__ == "__main__":
    run_god_pipeline()
