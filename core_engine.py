import os, json, sqlite3, numpy as np, pandas as pd
from contextlib import closing
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import IsolationForest
from scipy.signal import argrelextrema
from dataclasses import dataclass
from typing import List, Tuple

STATE_DIR = ".state_cache"
os.makedirs(STATE_DIR, exist_ok=True)

# ==============================================================================
# 1. MARKET STRUCTURE (SMC)
# ==============================================================================
def find_swing_points(df: pd.DataFrame, left=5, right=5) -> pd.DataFrame:
    df = df.copy()
    highs, lows = df['high'].values, df['low'].values
    sh = np.full(len(df), np.nan); sl = np.full(len(df), np.nan)
    for i in range(left, len(df) - right):
        if all(highs[i] > highs[i-j] for j in range(1, left+1)) and all(highs[i] > highs[i+j] for j in range(1, right+1)): sh[i] = highs[i]
        if all(lows[i] < lows[i-j] for j in range(1, left+1)) and all(lows[i] < lows[i+j] for j in range(1, right+1)): sl[i] = lows[i]
    df['swing_high'], df['swing_low'] = sh, sl
    return df

def detect_bos_choch(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    last_hh, last_ll = np.nan, np.nan
    bos_choch = [''] * len(df)
    current_trend = 1 if df['close'].iloc[-1] > df['close'].iloc[-50] else -1
    for i in range(50, len(df)):
        if not np.isnan(df['swing_high'].iloc[i]): last_hh = df['swing_high'].iloc[i]
        if not np.isnan(df['swing_low'].iloc[i]): last_ll = df['swing_low'].iloc[i]
        if current_trend == -1 and df['close'].iloc[i] > last_hh: bos_choch[i] = 'ChoCh_Bull'; current_trend = 1
        elif current_trend == 1 and df['close'].iloc[i] > last_hh: bos_choch[i] = 'BOS_Bull'
        elif current_trend == 1 and df['close'].iloc[i] < last_ll: bos_choch[i] = 'ChoCh_Bear'; current_trend = -1
        elif current_trend == -1 and df['close'].iloc[i] < last_ll: bos_choch[i] = 'BOS_Bear'
    df['structure'] = bos_choch
    return df

def find_order_blocks(df: pd.DataFrame) -> List[dict]:
    obs = []
    for i in range(2, len(df)):
        if df['close'].iloc[i-1] < df['open'].iloc[i-1] and df['close'].iloc[i] > df['high'].iloc[i-2]:
            obs.append({'type': 'OB_BULL', 'top': df['open'].iloc[i-1], 'bottom': df['low'].iloc[i-1]})
        if df['close'].iloc[i-1] > df['open'].iloc[i-1] and df['close'].iloc[i] < df['low'].iloc[i-2]:
            obs.append({'type': 'OB_BEAR', 'top': df['high'].iloc[i-1], 'bottom': df['close'].iloc[i-1]})
    return obs

def find_fvg(df: pd.DataFrame) -> List[dict]:
    fvgs = []
    for i in range(2, len(df)):
        if df['low'].iloc[i] > df['high'].iloc[i-2]: fvgs.append({'type': 'FVG_BULL', 'top': df['low'].iloc[i], 'bottom': df['high'].iloc[i-2]})
        if df['high'].iloc[i] < df['low'].iloc[i-2]: fvgs.append({'type': 'FVG_BEAR', 'top': df['low'].iloc[i-2], 'bottom': df['high'].iloc[i]})
    return fvgs

def detect_liquidity_sweep(df: pd.DataFrame, lookback=20) -> pd.DataFrame:
    df = df.copy()
    sweep = [''] * len(df)
    for i in range(lookback, len(df)):
        rh = df['high'].iloc[i-lookback:i].max()
        rl = df['low'].iloc[i-lookback:i].min()
        if df['high'].iloc[i] > rh and df['close'].iloc[i] < rh: sweep[i] = 'Sweep_High'
        elif df['low'].iloc[i] < rl and df['close'].iloc[i] > rl: sweep[i] = 'Sweep_Low'
    df['liquidity_sweep'] = sweep
    return df

# ==============================================================================
# 2. QUANT INDICATORS
# ==============================================================================
def calc_vwap(df: pd.DataFrame) -> pd.Series:
    tp = (df['high'] + df['low'] + df['close']) / 3
    return (tp * df['volume']).cumsum() / df['volume'].cumsum()

def detect_rsi_divergence(df: pd.DataFrame, period=14, lookback=30) -> str:
    try:
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = -delta.where(delta < 0, 0).rolling(period).mean()
        rsi = 100 - (100 / (1 + gain / (loss + 1e-9)))
        if len(rsi) < lookback: return "NONE"
        p_vals = df['close'].values[-lookback:]
        r_vals = rsi.values[-lookback:]
        p_lows = argrelextrema(p_vals, np.less, order=5)[0]
        p_highs = argrelextrema(p_vals, np.greater, order=5)[0]
        if len(p_lows) >= 2 and p_vals[p_lows[-1]] < p_vals[p_lows[-2]] and r_vals[p_lows[-1]] > r_vals[p_lows[-2]]: return "BULL_DIV"
        if len(p_highs) >= 2 and p_vals[p_highs[-1]] > p_vals[p_highs[-2]] and r_vals[p_highs[-1]] < r_vals[p_highs[-2]]: return "BEAR_DIV"
    except: pass
    return "NONE"

def calc_volume_delta(df: pd.DataFrame) -> float:
    try: return float((df['volume'] * np.sign(df['close'] - df['open'])).iloc[-1])
    except: return 0.0

# ==============================================================================
# 3. MACHINE LEARNING (REGIME & ANOMALY)
# ==============================================================================
def detect_ml_regime(df: pd.DataFrame):
    if len(df) < 100: return "CHOP", 0.5
    try:
        feats = df[['close']].pct_change().dropna().values.reshape(-1, 1)
        if len(feats) < 50: return "CHOP", 0.5
        gmm = GaussianMixture(n_components=3, random_state=42).fit(feats)
        labels = gmm.predict(feats)
        means = gmm.means_.flatten()
        sorted_idx = np.argsort(means)
        current_label = labels[-1]
        prob = gmm.predict_proba(feats[-1:])[0]
        if current_label == sorted_idx[2]: return "BULL", prob[current_label]
        elif current_label == sorted_idx[0]: return "BEAR", prob[current_label]
        else: return "CHOP", prob[current_label]
    except: return "CHOP", 0.5

def detect_ml_anomaly(df: pd.DataFrame):
    if len(df) < 50: return 0.0, False
    try:
        feats = pd.DataFrame({
            'ret': df['close'].pct_change(),
            'range': (df['high'] - df['low']) / df['close'],
            'vol_spike': df['volume'] / df['volume'].rolling(20).mean()
        }).dropna()
        if len(feats) < 20: return 0.0, False
        iso = IsolationForest(contamination=0.05, random_state=42).fit(feats)
        scores = iso.score_samples(feats)
        norm_scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)
        current_anomaly = 1.0 - norm_scores[-1]
        return current_anomaly, current_anomaly > 0.8
    except: return 0.0, False

# ==============================================================================
# 4. EPISODIC MEMORY (SQLITE)
# ==============================================================================
class AGIMemory:
    def __init__(self):
        self.db = os.path.join(STATE_DIR, "god_memory.db")
        with closing(sqlite3.connect(self.db)) as c:
            c.execute("""CREATE TABLE IF NOT EXISTS mem (
                id INTEGER PRIMARY KEY, ts REAL, emb BLOB, signal TEXT, 
                outcome TEXT, pnl REAL, structure TEXT)""")
            c.commit()

    def store(self, emb, signal, structure_tag):
        with closing(sqlite3.connect(self.db)) as c:
            c.execute("INSERT INTO mem (ts, emb, signal, outcome, pnl, structure) VALUES (?,?,?,?,?,?)",
                      (pd.Timestamp.now().timestamp(), emb.tobytes(), signal, "OPEN", 0.0, structure_tag))
            c.commit()

    def query(self, emb, k=15):
        with closing(sqlite3.connect(self.db)) as c:
            rows = c.execute("SELECT id, emb, signal, outcome, pnl, structure FROM mem WHERE outcome != 'OPEN'").fetchall()
        if not rows: return {"n": 0, "wr": 50.0, "avg_pnl": 0.0}
        embs = np.array([np.frombuffer(r[1], np.float32) for r in rows])
        norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-9
        sims = np.dot(embs / norms, emb / (np.linalg.norm(emb) + 1e-9))
        top_idx = np.argsort(sims)[::-1][:k]
        top_rows = [rows[i] for i in top_idx]
        wins = sum(1 for r in top_rows if r[3] == "WIN")
        losses = sum(1 for r in top_rows if r[3] == "LOSS")
        pnls = [r[4] for r in top_rows if r[3] in ("WIN", "LOSS")]
        return {"n": wins + losses, "wr": round(wins / max(1, wins+losses) * 100, 1), "avg_pnl": round(np.mean(pnls), 2) if pnls else 0.0}

    def update_outcome(self, signal_id, outcome, pnl):
        with closing(sqlite3.connect(self.db)) as c:
            c.execute("UPDATE mem SET outcome=?, pnl=? WHERE id=?", (outcome, pnl, signal_id))
            c.commit()

# ==============================================================================
# 5. THE GOD GRADER (A, A++, A+++, A SUPER)
# ==============================================================================
@dataclass
class SignalGrade:
    grade: str
    score: float
    reasons: list

def grade_the_god_signal(direction, ml_regime, reg_conf, anom_score, is_anom,
                         has_choch, has_bos, has_sweep, entry_at_ob, entry_at_fvg,
                         rsi_div, vol_delta, mem_stats):
    score, reasons = 0.0, []
    
    if (direction == "BUY" and ml_regime == "BULL") or (direction == "SELL" and ml_regime == "BEAR"):
        score += 20 * reg_conf; reasons.append(f"✅ ML Regime: {ml_regime} ({reg_conf:.0%})")
    elif ml_regime == "CHOP": score -= 10; reasons.append("⚠️ ML Regime: CHOP")
        
    if has_sweep: score += 15; reasons.append("✅ Liquidity Sweep (Judas Swing)")
    if has_choch: score += 10; reasons.append("✅ Change of Character (ChoCh)")
    if has_bos: score += 5; reasons.append("✅ Break of Structure (BOS)")
        
    if entry_at_ob: score += 15; reasons.append("✅ Entry at Order Block (OB)")
    elif entry_at_fvg: score += 10; reasons.append("✅ Entry at Fair Value Gap (FVG)")
        
    if rsi_div == "BULL_DIV" and direction == "BUY": score += 10; reasons.append("✅ Bullish RSI Divergence")
    elif rsi_div == "BEAR_DIV" and direction == "SELL": score += 10; reasons.append("✅ Bearish RSI Divergence")
        
    if vol_delta > 0 and direction == "BUY": score += 5; reasons.append("✅ Positive Volume Delta")
    elif vol_delta < 0 and direction == "SELL": score += 5; reasons.append("✅ Negative Volume Delta")
    
    if mem_stats['n'] >= 5 and mem_stats['wr'] >= 60: score += 10; reasons.append(f"✅ Memory WR: {mem_stats['wr']}%")
    if not is_anom: score += 5; reasons.append("✅ Clean Market")
    else: score -= 15; reasons.append("❌ High Anomaly")

    score = max(0, min(100, score))
    
    if score >= 95 and has_sweep and (entry_at_ob or entry_at_fvg) and mem_stats['wr'] >= 60:
        grade = "A SUPER 💎"; reasons.insert(0, "🏆 HOLY GRAIL: Perfect Confluence")
    elif score >= 85: grade = "A+++"
    elif score >= 75: grade = "A++"
    elif score >= 65: grade = "A"
    else: grade = "TRASH"
        
    return SignalGrade(grade=grade, score=round(score, 1), reasons=reasons)
