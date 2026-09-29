#!/usr/bin/env python3
"""
AGI Core Engine (Regime, Anomaly, Embedding, Memory, Calibrator, Meta, Grader, Caption)
"""

import os
import json
import logging
import sqlite3
import datetime
import html
from contextlib import closing
from dataclasses import dataclass
from enum import Enum
from typing import List, Dict, Optional, Tuple
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("AGICore")

STATE_DIR = ".state_cache"
os.makedirs(STATE_DIR, exist_ok=True)
DB_FILE = os.path.join(STATE_DIR, "agi_memory.db")
CAL_FILE = os.path.join(STATE_DIR, "calibration.json")
META_FILE = os.path.join(STATE_DIR, "meta.json")

EMB_DIM = 32


class Regime(str, Enum):
    TRENDING_UP = "TRENDING_UP"
    TRENDING_DOWN = "TRENDING_DOWN"
    RANGING = "RANGING"
    VOLATILE = "VOLATILE"
    QUIET = "QUIET"
    TRANSITION = "TRANSITION"


@dataclass
class RegimeState:
    regime: Regime
    confidence: float
    adx: float
    atr_pct: float
    bb_pct: float
    slope: float
    description: str


def _adx_wilder(df: pd.DataFrame, p: int = 14) -> float:
    if len(df) < p * 2:
        return 20.0
    h, l, c = df["high"], df["low"], df["close"]
    up = h.diff()
    dn = -l.diff()
    pdm = np.where((up > dn) & (up > 0), up, 0.0)
    mdm = np.where((dn > up) & (dn > 0), dn, 0.0)
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/p, adjust=False).mean()
    pdi = 100 * pd.Series(pdm, index=df.index).ewm(alpha=1/p, adjust=False).mean() / (atr + 1e-9)
    mdi = 100 * pd.Series(mdm, index=df.index).ewm(alpha=1/p, adjust=False).mean() / (atr + 1e-9)
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi + 1e-9)
    adx_series = dx.ewm(alpha=1/p, adjust=False).mean()
    v = adx_series.iloc[-1]
    return float(v) if not pd.isna(v) else 20.0


def _atr_pct(df: pd.DataFrame, p: int = 14, lb: int = 100) -> float:
    if len(df) < p + 10:
        return 0.5
    h, l, c = df["high"], df["low"], df["close"]
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(p).mean().dropna()
    if len(atr) == 0:
        return 0.5
    window = atr.tail(min(len(atr), lb))
    return float((window < atr.iloc[-1]).mean())


def _bb_pct(df: pd.DataFrame, p: int = 20, lb: int = 100) -> float:
    if len(df) < p + 10:
        return 0.5
    ma = df["close"].rolling(p).mean()
    sd = df["close"].rolling(p).std()
    w = ((4 * sd) / (ma + 1e-9)).dropna()
    if len(w) == 0:
        return 0.5
    window = w.tail(min(len(w), lb))
    return float((window < w.iloc[-1]).mean())


def _slope(df: pd.DataFrame, span: int = 50, lb: int = 10) -> float:
    if len(df) < span:
        return 0.0
    e = df["close"].ewm(span=span).mean()
    return float((e.iloc[-1] - e.iloc[-lb]) / (abs(e.iloc[-lb]) + 1e-9))


def detect_regime(df: pd.DataFrame) -> RegimeState:
    if df is None or len(df) < 50:
        return RegimeState(Regime.TRANSITION, 0.0, 20.0, 0.5, 0.5, 0.0, "insufficient_data")
    try:
        adx = _adx_wilder(df)
        ap = _atr_pct(df)
        bp = _bb_pct(df)
        sl = _slope(df)
        regime, conf = Regime.TRANSITION, 0.4
        if ap >= 0.85 and bp >= 0.85:
            regime, conf = Regime.VOLATILE, min(0.95, 0.6 + (ap - 0.85) * 2)
        elif adx >= 25 and abs(sl) > 0.002:
            regime = Regime.TRENDING_UP if sl > 0 else Regime.TRENDING_DOWN
            conf = min(0.95, 0.5 + (adx - 25) / 50)
        elif ap <= 0.25 and bp <= 0.30:
            regime, conf = Regime.QUIET, min(0.90, 0.5 + (0.25 - ap) * 2)
        elif adx < 20 and bp <= 0.5:
            regime, conf = Regime.RANGING, min(0.90, 0.5 + (20 - adx) / 40)
        return RegimeState(
            regime=regime, confidence=round(conf, 3),
            adx=round(adx, 2), atr_pct=round(ap, 3), bb_pct=round(bp, 3),
            slope=round(sl, 5),
            description=f"{regime.value} ADX={adx:.1f} ATR%={ap:.2f} BB%={bp:.2f}"
        )
    except Exception as e:
        logger.error(f"Error detect_regime: {e}")
        return RegimeState(Regime.TRANSITION, 0.0, 20.0, 0.5, 0.5, 0.0, "error")


@dataclass
class AnomalyReport:
    score: float
    is_anomaly: bool
    reason: str


def detect_anomaly(df: pd.DataFrame, z_th: float = 3.0) -> AnomalyReport:
    if df is None or len(df) < 50:
        return AnomalyReport(0.0, False, "insufficient_data")
    try:
        c = df["close"].astype(float)
        h = df["high"].astype(float)
        l = df["low"].astype(float)
        v = df["volume"].astype(float) if "volume" in df else pd.Series(np.ones(len(df)), index=df.index)
        ret = c.pct_change()
        ret_std = ret.rolling(50).std().iloc[-1] + 1e-9
        ret_z = abs(ret.iloc[-1]) / ret_std
        rng = (h - l) / (c + 1e-9)
        rng_std = rng.rolling(50).std().iloc[-1] + 1e-9
        rng_z = abs(rng.iloc[-1] - rng.rolling(50).mean().iloc[-1]) / rng_std
        vol_mean = v.rolling(50).mean().iloc[-1]
        vol_std = v.rolling(50).std().iloc[-1] + 1e-9
        vol_z = abs(v.iloc[-1] - vol_mean) / vol_std
        score = float(np.clip(0.35 * min(1.0, ret_z / z_th) + 0.30 * min(1.0, rng_z / z_th) + 0.35 * min(1.0, vol_z / z_th), 0, 1))
        reasons = []
        if ret_z > z_th: reasons.append(f"ret_z={ret_z:.1f}")
        if rng_z > z_th: reasons.append(f"rng_z={rng_z:.1f}")
        if vol_z > z_th: reasons.append(f"vol_z={vol_z:.1f}")
        return AnomalyReport(round(score, 3), score >= 0.65, "; ".join(reasons) or "normal")
    except Exception as e:
        logger.error(f"Error detect_anomaly: {e}")
        return AnomalyReport(0.0, False, "error")


def _moments(x: np.ndarray, k: int = 4) -> List[float]:
    if len(x) < 2:
        return [0.0] * k
    m = float(np.mean(x))
    s = float(np.std(x)) + 1e-9
    z = (x - m) / s
    return [m, s, float(np.mean(z ** 3)), float(np.mean(z ** 4)) - 3]


def encode(df: pd.DataFrame) -> np.ndarray:
    if df is None or len(df) < 20:
        return np.zeros(EMB_DIM, dtype=np.float32)
    try:
        c = df["close"].astype(float).values
        h = df["high"].astype(float).values
        l = df["low"].astype(float).values
        v = df["volume"].astype(float).values if "volume" in df else np.ones(len(c))
        ret = np.diff(c) / (c[:-1] + 1e-9)
        rng = (h - l) / (c + 1e-9)
        feats: List[float] = []
        feats += _moments(ret)
        feats += _moments(rng)
        feats += _moments(np.log1p(np.abs(v)))
        x = ret - ret.mean()
        d = np.dot(x, x) + 1e-9
        for lag in (1, 2, 3, 5, 8, 13):
            feats.append(float(np.dot(x[:-lag], x[lag:]) / d) if lag < len(x) else 0.0)
        x_fft = c - c.mean()
        fft = np.abs(np.fft.rfft(x_fft))
        top = np.sort(fft)[::-1][:8]
        tot = fft.sum() + 1e-9
        fft_feats = [float(t / tot) for t in top]
        fft_feats += [0.0] * (8 - len(fft_feats))
        feats += fft_feats
        qs = np.percentile(c, np.linspace(0, 100, 9))
        hist, _ = np.histogram(c, bins=qs)
        tot_hist = hist.sum() + 1e-9
        feats += [float(hh / tot_hist) for hh in hist]
        vec = np.array(feats, dtype=np.float32)
        vec = np.nan_to_num(vec, nan=0.0, posinf=1.0, neginf=-1.0)
        n = np.linalg.norm(vec) + 1e-9
        vec = vec / n
        if len(vec) < EMB_DIM:
            vec = np.pad(vec, (0, EMB_DIM - len(vec)))
        return vec[:EMB_DIM].astype(np.float32)
    except Exception as e:
        logger.error(f"Error encode embedding: {e}")
        return np.zeros(EMB_DIM, dtype=np.float32)


class Memory:
    def __init__(self, path: str = DB_FILE, max_entries: int = 2000):
        self.path = path
        self.max_entries = max_entries
        self._init()

    def _init(self):
        with closing(sqlite3.connect(self.path)) as c:
            c.execute("""
                CREATE TABLE IF NOT EXISTS mem (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts REAL, embedding BLOB, signal TEXT, entry REAL,
                    outcome TEXT, pnl_r REAL, regime TEXT, extra TEXT
                )
            """)
            c.commit()

    def store(self, df: pd.DataFrame, regime_state: RegimeState, signal: str, entry: float,
              outcome: str = "OPEN", pnl_r: float = 0.0, extra: Optional[dict] = None) -> int:
        emb = encode(df)
        utc_now = datetime.datetime.now(datetime.timezone.utc).timestamp()
        with closing(sqlite3.connect(self.path)) as c:
            cur = c.execute(
                "INSERT INTO mem (ts, embedding, signal, entry, outcome, pnl_r, regime, extra) "
                "VALUES (?,?,?,?,?,?,?,?)",
                (utc_now, emb.tobytes(), signal, float(entry), outcome, float(pnl_r),
                 regime_state.regime.value, json.dumps(extra or {}))
            )
            c.commit()
            eid = cur.lastrowid
        self._prune()
        return eid

    def update(self, eid: int, outcome: str, pnl_r: float):
        with closing(sqlite3.connect(self.path)) as c:
            c.execute("UPDATE mem SET outcome=?, pnl_r=? WHERE id=?", (outcome, float(pnl_r), eid))
            c.commit()

    def _prune(self):
        with closing(sqlite3.connect(self.path)) as c:
            (n,) = c.execute("SELECT COUNT(*) FROM mem").fetchone()
            if n > self.max_entries:
                c.execute(f"DELETE FROM mem WHERE id IN (SELECT id FROM mem ORDER BY id ASC LIMIT {n - self.max_entries})")
                c.commit()

    def query(self, df: pd.DataFrame, k: int = 20) -> Tuple[List[dict], dict]:
        q = encode(df)
        with closing(sqlite3.connect(self.path)) as c:
            rows = c.execute(
                "SELECT id, embedding, signal, entry, outcome, pnl_r, regime "
                "FROM mem WHERE outcome IN ('WIN','LOSS','DRAW')"
            ).fetchall()
        if not rows:
            return [], {"n": 0, "winrate": 0.0, "avg_r": 0.0}
        sims = []
        for r in rows:
            e = np.frombuffer(r[1], dtype=np.float32)
            if len(e) != EMB_DIM: continue
            s = float(np.dot(q, e) / (np.linalg.norm(q) * np.linalg.norm(e) + 1e-9))
            sims.append((s, r))
        sims.sort(key=lambda x: -x[0])
        top = sims[:k]
        wins = sum(1 for _, r in top if r[4] == "WIN")
        losses = sum(1 for _, r in top if r[4] == "LOSS")
        rs = [r[5] for _, r in top if r[4] in ("WIN", "LOSS")]
        return (
            [{"id": r[0], "outcome": r[4], "pnl_r": r[5], "regime": r[6]} for _, r in top],
            {
                "n": wins + losses,
                "winrate": round(wins / max(1, wins + losses) * 100, 2),
                "avg_r": round(float(np.mean(rs)) if rs else 0.0, 4),
            },
        )


class Calibrator:
    def __init__(self, path: str = CAL_FILE):
        self.path = path
        self.a, self.b, self.fitted, self.n_samples = 1.0, 0.0, False, 0
        if os.path.exists(path):
            try:
                with open(path) as f: d = json.load(f)
                self.a = d.get("a", 1.0)
                self.b = d.get("b", 0.0)
                self.fitted = d.get("fitted", False)
                self.n_samples = d.get("n_samples", 0)
            except Exception as e:
                logger.error(f"Failed to load calibration file: {e}")

    def save(self):
        tmp = self.path + ".tmp"
        with open(tmp, "w") as f:
            json.dump({"a": self.a, "b": self.b, "fitted": self.fitted, "n_samples": self.n_samples}, f)
        os.replace(tmp, self.path)

    def calibrate(self, p: float) -> float:
        if not self.fitted: return float(p)
        z = np.clip(self.a * float(p) + self.b, -15, 15)
        return float(1.0 / (1.0 + np.exp(-z)))


class MetaLearner:
    def __init__(self, path: str = META_FILE, alpha: float = 0.15):
        self.path, self.alpha, self.state = path, alpha, {}
        if os.path.exists(path):
            try:
                with open(path) as f: self.state = json.load(f)
            except Exception: self.state = {}

    def _save(self):
        tmp = self.path + ".tmp"
        with open(tmp, "w") as f: json.dump(self.state, f, indent=2)
        os.replace(tmp, self.path)

    def update(self, regime: str, win: bool, pnl_r: float):
        s = self.state.setdefault(regime, {"n": 0, "wins": 0, "ema_wr": 0.5, "ema_r": 0.0, "penalty": 1.0})
        s["n"] += 1
        if win: s["wins"] += 1
        y = 1.0 if win else 0.0
        s["ema_wr"] = (1 - self.alpha) * s["ema_wr"] + self.alpha * y
        s["ema_r"] = (1 - self.alpha) * s["ema_r"] + self.alpha * float(pnl_r)
        if s["ema_wr"] < 0.45 or s["ema_r"] < -0.1: s["penalty"] = max(0.5, s["penalty"] * 0.97)
        elif s["ema_wr"] > 0.55 and s["ema_r"] > 0.1: s["penalty"] = min(1.3, s["penalty"] * 1.02)
        self._save()

    def penalty(self, regime: str) -> float:
        return self.state.get(regime, {}).get("penalty", 1.0)


def grade_signal(consensus: float, signal: str, engine_states: dict,
                 h1_trend: str, h4_trend: str, atr: float,
                 memory_stats: dict, anomaly_score: float) -> dict:
    score = 0.0
    score += max(0.0, min(25.0, (consensus - 60) / 40 * 25))
    h1_ok = ("BULLISH" in h1_trend and signal == "BUY") or ("BEARISH" in h1_trend and signal == "SELL")
    h4_ok = ("BULLISH" in h4_trend and signal == "BUY") or ("BEARISH" in h4_trend and signal == "SELL")
    if h1_ok and h4_ok: score += 20
    elif h1_ok: score += 12
    elif h4_ok: score += 6
    want_bull = signal == "BUY"
    agree = tot = 0
    for st in engine_states.values():
        sc = st.get("sc", 0)
        if sc == 0: continue
        tot += 1
        if (sc > 0) == want_bull: agree += 1
    if tot: score += agree / tot * 20
    if 5 <= atr <= 12: score += 15
    elif atr < 5: score += max(0, atr / 5 * 15)
    else: score += max(0, 15 - (atr - 12) * 2)
    if memory_stats.get("n", 0) >= 5:
        wr = memory_stats.get("winrate", 50)
        score += min(10, max(0, (wr - 40) / 30 * 10))
    if anomaly_score >= 0.65: score -= 15
    score = max(0, min(100, score))
    if score >= 90: grade = "A Super"
    elif score >= 85: grade = "A+++"
    elif score >= 75: grade = "A++"
    elif score >= 65: grade = "A"
    elif score >= 55: grade = "B"
    elif score >= 45: grade = "C"
    else: grade = "D"
    return {"score": round(score, 2), "grade": grade}


def compose_caption(signal: str, entry: float, sl: float, tps: List[float],
                    atr: float, consensus: float, grade: str, grade_score: float,
                    regime: RegimeState, anomaly: AnomalyReport,
                    memory_stats: dict, debate_verdict: str,
                    debate_notes: str, source: str, offset: float,
                    ai_insight: str, symbol: str = "XAUUSD") -> str:
    bar_f = int(consensus / 10)
    bar = "█" * bar_f + "░" * (10 - bar_f)
    regime_tag = {
        "TRENDING_UP": "📈 TREND UP", "TRENDING_DOWN": "📉 TREND DOWN",
        "RANGING": "↔️ RANGING", "VOLATILE": "⚡ VOLATILE",
        "QUIET": "😴 QUIET", "TRANSITION": "🔄 TRANSITION",
    }.get(regime.regime.value, "❓")
    emoji = "🟢" if signal == "BUY" else "🔴"
    mem_line = "—"
    if memory_stats.get("n", 0) >= 5:
        mem_line = f"{memory_stats['n']} case | WR {memory_stats['winrate']:.0f}% | avg {memory_stats['avg_r']:+.2f}R"
    anom_line = f"{anomaly.score:.2f}" + (" ⚠️" if anomaly.is_anomaly else "")
    debate_line = ""
    if debate_verdict and debate_verdict != "ABSTAIN":
        de = {"AGREE": "✅", "DISAGREE": "❌"}.get(debate_verdict, "⚪")
        debate_line = f"\n{de} <b>AI Council:</b> {debate_verdict}"
        if debate_notes: debate_line += f" — <i>{html.escape(debate_notes[:120])}</i>"
    tp_line = " | ".join(f"<code>{t:.2f}</code>" for t in tps)
    off_line = f" | offset {offset:+.2f}" if offset else ""
    ai_safe = html.escape(ai_insight[:400]) if ai_insight else "-"
    now_wib = datetime.datetime.now(ZoneInfo("Asia/Jakarta")).strftime('%H:%M WIB')

    return (
        f"{emoji} <b>{symbol} {signal}</b> — Grade <b>{grade}</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━━\n"
        f"📊 Confidence: [{bar}] {consensus:.0f}%  (score {grade_score:.0f}/100)\n"
        f"🌊 Regime: {regime_tag} (conf {regime.confidence:.2f})\n"
        f"📡 Source: {html.escape(source)}{off_line}\n"
        f"━━━━━━━━━━━━━━━━━━━━━\n"
        f"<b>Entry :</b> <code>{entry:.2f}</code>\n"
        f"<b>SL    :</b> <code>{sl:.2f}</code> (risk {abs(entry - sl):.2f})\n"
        f"<b>TP    :</b> {tp_line}\n"
        f"<b>ATR   :</b> {atr:.2f}\n"
        f"━━━━━━━━━━━━━━━━━━━━━\n"
        f"🧠 Memory: {mem_line}\n"
        f"⚡ Anomaly: {anom_line}"
        f"{debate_line}\n"
        f"🤖 <i>{ai_safe}</i>\n"
        f"━━━━━━━━━━━━━━━━━━━━━\n"
        f"⏰ {now_wib}"
    )
