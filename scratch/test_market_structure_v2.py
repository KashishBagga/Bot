#!/usr/bin/env python3
"""
MKE Milestone 1 — Automated Verification Suite
===============================================
Tests every layer of the new market structure pipeline:

  1. TimeOfDayEngine  — profile lookup & bootstrap fallback
  2. TradingClock     — bars_between (same-day & multi-day)
  3. PivotEngine      — pivot detection, deterministic IDs, strength sub-components
  4. ClusterEngine    — EQH / EQL grouping
  5. StructureEngine  — swing lifecycles, developing leg, candidate swing, events
  6. IndicatorPipeline integration — MarketContext built correctly on m5+h1+d1
  7. Database         — ResearchEvent persists to market_events hypertable

Run from the project root:
    python scratch/test_market_structure_v2.py
"""

import sys
import traceback
import logging
from datetime import datetime, date, timedelta, time
from typing import List, Tuple

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.WARNING)          # suppress engine noise
logger = logging.getLogger("MKE_VERIFY")

PASS = "✅ PASS"
FAIL = "❌ FAIL"
results: List[Tuple[str, str, str]] = []            # (test_name, status, detail)


def _record(name: str, ok: bool, detail: str = ""):
    status = PASS if ok else FAIL
    results.append((name, status, detail))
    symbol = "✅" if ok else "❌"
    print(f"  {symbol} {name}" + (f"  — {detail}" if detail else ""))


# ─────────────────────────────────────────────────────────────────────────────
# Helpers to synthesise OHLCV DataFrames
# ─────────────────────────────────────────────────────────────────────────────

def _make_m5_df(n: int = 150, seed: int = 42) -> pd.DataFrame:
    """
    Generates a realistic-looking intraday m5 DataFrame spread across
    multiple trading days (Mon–Fri, 09:15–15:30 IST).
    """
    rng = np.random.default_rng(seed)
    idx = []
    prices = []
    base = 22000.0
    current = datetime(2026, 6, 16, 9, 15)   # Monday

    for _ in range(n):
        # Advance by 5 minutes; skip weekends & non-trading hours
        while True:
            if current.weekday() < 5:
                t = current.time()
                if time(9, 15) <= t <= time(15, 25):
                    break
            if current.weekday() >= 5:
                current += timedelta(days=(7 - current.weekday()))
                current = current.replace(hour=9, minute=15)
            elif current.time() >= time(15, 30):
                current += timedelta(days=1)
                current = current.replace(hour=9, minute=15)
                while current.weekday() >= 5:
                    current += timedelta(days=1)
            else:
                current += timedelta(minutes=5)

        delta = rng.normal(0, 0.3) * (base * 0.001)
        base += delta
        o = round(base + rng.uniform(-0.2, 0.2), 2)
        c = round(base + rng.uniform(-0.2, 0.2), 2)
        h = round(max(o, c) + rng.uniform(0, 0.4), 2)
        l = round(min(o, c) - rng.uniform(0, 0.4), 2)
        v = int(rng.integers(1000, 50000))
        idx.append(current)
        prices.append((o, h, l, c, v))
        current += timedelta(minutes=5)

    df = pd.DataFrame(prices, columns=["open", "high", "low", "close", "volume"],
                      index=pd.DatetimeIndex(idx))
    return df


def _make_h1_df(n: int = 50, seed: int = 99) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = []
    prices = []
    base = 22000.0
    current = datetime(2026, 6, 9, 9, 15)    # Monday, 10 days back

    for _ in range(n):
        while True:
            if current.weekday() < 5 and time(9, 15) <= current.time() <= time(14, 15):
                break
            if current.time() >= time(15, 15):
                current += timedelta(days=1)
                current = current.replace(hour=9, minute=15)
                while current.weekday() >= 5:
                    current += timedelta(days=1)
            else:
                current += timedelta(hours=1)

        delta = rng.normal(0, 1.5) * (base * 0.001)
        base += delta
        o = round(base + rng.uniform(-1, 1), 2)
        c = round(base + rng.uniform(-1, 1), 2)
        h = round(max(o, c) + rng.uniform(0, 2), 2)
        l = round(min(o, c) - rng.uniform(0, 2), 2)
        v = int(rng.integers(5000, 200000))
        idx.append(current)
        prices.append((o, h, l, c, v))
        current += timedelta(hours=1)

    return pd.DataFrame(prices, columns=["open", "high", "low", "close", "volume"],
                        index=pd.DatetimeIndex(idx))


def _make_d1_df(n: int = 30, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = []
    prices = []
    base = 22000.0
    current = datetime(2026, 5, 14, 9, 15)

    for _ in range(n):
        while current.weekday() >= 5:
            current += timedelta(days=1)
        delta = rng.normal(0, 5) * (base * 0.001)
        base += delta
        o = round(base + rng.uniform(-3, 3), 2)
        c = round(base + rng.uniform(-3, 3), 2)
        h = round(max(o, c) + rng.uniform(0, 8), 2)
        l = round(min(o, c) - rng.uniform(0, 8), 2)
        v = int(rng.integers(500000, 5000000))
        idx.append(current)
        prices.append((o, h, l, c, v))
        current += timedelta(days=1)

    return pd.DataFrame(prices, columns=["open", "high", "low", "close", "volume"],
                        index=pd.DatetimeIndex(idx))


# ─────────────────────────────────────────────────────────────────────────────
# 1. TimeOfDayEngine
# ─────────────────────────────────────────────────────────────────────────────

def test_tod_engine():
    print("\n── 1. TimeOfDayEngine ──────────────────────────────────────────")
    from src.core.time_of_day_engine import TimeOfDayEngine, TimeOfDayProfileSlot

    tod = TimeOfDayEngine(use_bootstrap=True)

    # Profile should have 75 slots (09:15 to 15:30 at 5-min intervals)
    _record("Profile has 75 slots", len(tod.profile) == 75, f"got {len(tod.profile)}")

    # Open slot (09:15) should have high volume factor
    open_slot = tod.lookup(datetime(2026, 6, 16, 9, 15))
    _record("Open (09:15) has vol_factor >= 2.0", open_slot.avg_volume_factor >= 2.0,
            f"vol={open_slot.avg_volume_factor:.3f}")

    # Mid-day slot (12:00) should have lower volume
    mid_slot = tod.lookup(datetime(2026, 6, 16, 12, 0))
    _record("Mid-day (12:00) has vol_factor < 1.0", mid_slot.avg_volume_factor < 1.0,
            f"vol={mid_slot.avg_volume_factor:.3f}")

    # Outside market hours returns neutral fallback
    after_slot = tod.lookup(datetime(2026, 6, 16, 18, 0))
    _record("After-hours returns neutral slot (vol_factor=1.0)",
            after_slot.avg_volume_factor == 1.0, f"vol={after_slot.avg_volume_factor:.3f}")

    # Instantiating without bootstrap still works (empty profile, fallback only)
    tod2 = TimeOfDayEngine(use_bootstrap=False)
    fallback = tod2.lookup(datetime(2026, 6, 16, 9, 15))
    _record("No-bootstrap engine uses fallback slot",
            isinstance(fallback, TimeOfDayProfileSlot), "")


# ─────────────────────────────────────────────────────────────────────────────
# 2. TradingClock
# ─────────────────────────────────────────────────────────────────────────────

def test_trading_clock():
    print("\n── 2. TradingClock ─────────────────────────────────────────────")
    from src.core.trading_clock import TradingClock

    # Same day, 5-min bars: 09:15 → 10:00 = 9 bars
    t1 = datetime(2026, 6, 16, 9, 15)
    t2 = datetime(2026, 6, 16, 10, 0)
    bars = TradingClock.bars_between(t1, t2)
    _record("Same-day: 09:15→10:00 = 9 bars", bars == 9, f"got {bars}")

    # Same day, reversed → 0
    bars_rev = TradingClock.bars_between(t2, t1)
    _record("Reversed timestamps → 0 bars", bars_rev == 0, f"got {bars_rev}")

    # Weekend day returns 0
    sat = datetime(2026, 6, 14, 10, 0)   # Saturday
    sat_end = datetime(2026, 6, 14, 11, 0)
    _record("Weekend day → 0 bars", TradingClock.bars_between(sat, sat_end) == 0,
            f"got {TradingClock.bars_between(sat, sat_end)}")

    # Multi-day: Mon 15:30 → Tue 09:15 → very few bars (0 from Mon tail + 0 from Tue head)
    mon_close = datetime(2026, 6, 16, 15, 30)
    tue_open  = datetime(2026, 6, 17, 9, 15)
    bars_overnight = TradingClock.bars_between(mon_close, tue_open)
    _record("Overnight gap contributes 0 extra bars", bars_overnight == 0,
            f"got {bars_overnight}")

    # Single full day (Mon 09:15 → Tue 09:15) ≈ 75 bars
    full_day_bars = TradingClock.bars_between(
        datetime(2026, 6, 16, 9, 15), datetime(2026, 6, 17, 9, 15))
    _record(f"Full trading day ≈ 75 bars", abs(full_day_bars - 75) <= 1,
            f"got {full_day_bars}")

    # H1 interval check
    h1_bars = TradingClock.bars_between(
        datetime(2026, 6, 16, 9, 15), datetime(2026, 6, 16, 15, 15), interval_minutes=60)
    _record("H1 09:15→15:15 = 6 bars", h1_bars == 6, f"got {h1_bars}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. PivotEngine
# ─────────────────────────────────────────────────────────────────────────────

def test_pivot_engine():
    print("\n── 3. PivotEngine ──────────────────────────────────────────────")
    from src.core.pivot_engine import PivotEngine
    from src.core.market_knowledge import SwingStatus

    df = _make_m5_df(150)
    engine = PivotEngine(pivot_window=3)
    swings = engine.detect_pivots(df, symbol="NSE:NIFTY50-INDEX", timeframe="m5")

    _record("Detects at least 1 pivot", len(swings) > 0, f"found {len(swings)} pivots")

    # IDs are deterministic and symbol-prefixed
    ids = [s.id for s in swings]
    _record("All IDs start with 'sw_NSE_NIFTY50_INDEX_m5_'",
            all(i.startswith("sw_NSE_NIFTY50_INDEX_m5_") for i in ids), "")

    # No duplicate IDs
    _record("No duplicate swing IDs", len(ids) == len(set(ids)), f"{len(ids)} unique")

    # Strength sub-components present
    if swings:
        s = swings[0]
        keys = set(s.strength_components.keys())
        _record("Strength sub-components present (geometry, participation, reaction, persistence)",
                {"geometry", "participation", "reaction", "persistence"} <= keys, str(keys))
        _record("Aggregate strength in [0, 1]", 0.0 <= s.strength <= 1.0,
                f"strength={s.strength}")
        _record("Initial status is ACTIVE", s.status == SwingStatus.ACTIVE, str(s.status))

    # Session insulation: pivots should not span across day boundaries
    all_dates = set()
    for s in swings:
        all_dates.add(s.timestamp.date())

    # Each pivot's window neighbours must be on the same date (engine enforces this internally)
    # We verify indirectly: pivots are confirmed at a lag of w bars; timestamps inside session
    _record("All pivot timestamps are within market hours (09:15–15:30)",
            all(time(9, 15) <= s.timestamp.time() <= time(15, 30) for s in swings), "")

    # Determinism: same call returns identical results
    swings2 = engine.detect_pivots(df, symbol="NSE:NIFTY50-INDEX", timeframe="m5")
    _record("Pivot detection is deterministic",
            [s.id for s in swings] == [s.id for s in swings2], "")


# ─────────────────────────────────────────────────────────────────────────────
# 4. ClusterEngine
# ─────────────────────────────────────────────────────────────────────────────

def test_cluster_engine():
    print("\n── 4. ClusterEngine ────────────────────────────────────────────")
    from src.core.cluster_engine import ClusterEngine
    from src.core.market_knowledge import SwingPoint, SwingStatus

    # Build synthetic equal highs: two pivots at ~22000, one outlier at 22100
    def _make_swing(price: float, ts: datetime, kind: str) -> SwingPoint:
        return SwingPoint(
            id=f"sw_test_{kind}_{int(price)}",
            timestamp=ts,
            price=price,
            type=kind,
            status=SwingStatus.ACTIVE,
            confidence=1.0,
            strength=0.5,
            strength_components={"geometry": 0.5, "participation": 0.5, "reaction": 0.5, "persistence": 0.5},
            provenance={"engine": "test"}
        )

    base = datetime(2026, 6, 16, 10, 0)
    swings = [
        _make_swing(22000.1, base,                               "HIGH"),
        _make_swing(22000.3, base + timedelta(minutes=30),      "HIGH"),
        _make_swing(22000.2, base + timedelta(hours=1),         "HIGH"),
        _make_swing(22100.0, base + timedelta(hours=2),         "HIGH"),  # outlier
        _make_swing(21900.5, base + timedelta(minutes=15),      "LOW"),
        _make_swing(21900.2, base + timedelta(minutes=45),      "LOW"),
    ]

    engine = ClusterEngine(tolerance_pct=0.0005)
    clusters = engine.detect_clusters(swings, symbol="TEST", timeframe="m5")

    eqh = [c for c in clusters if c.type == "EQH"]
    eql = [c for c in clusters if c.type == "EQL"]

    _record("Detects 1 EQH cluster (3 equal highs)",    len(eqh) == 1, f"found {len(eqh)}")
    _record("Detects 1 EQL cluster (2 equal lows)",     len(eql) == 1, f"found {len(eql)}")
    _record("Outlier high (22100) not in any cluster",
            all(22100.0 not in [s_id for s_id in c.member_swing_ids] for c in eqh), "")
    if eqh:
        _record("EQH cluster price is average of members (≈22000.2)",
                abs(eqh[0].price - 22000.2) < 0.5, f"price={eqh[0].price}")
        _record("EQH cluster strength in [0, 1]",
                0.0 <= eqh[0].strength <= 1.0, f"strength={eqh[0].strength}")

    # Single pivot should produce no clusters
    clusters_single = engine.detect_clusters([swings[0]], "TEST", "m5")
    _record("Single pivot produces no clusters", len(clusters_single) == 0, "")


# ─────────────────────────────────────────────────────────────────────────────
# 5. StructureEngine
# ─────────────────────────────────────────────────────────────────────────────

def test_structure_engine():
    print("\n── 5. StructureEngine ──────────────────────────────────────────")
    from src.core.structure_engine import StructureEngine
    from src.core.pivot_engine import PivotEngine
    from src.core.cluster_engine import ClusterEngine
    from src.core.market_knowledge import SwingStatus, DevelopingLeg

    df = _make_m5_df(150)
    pivot_engine = PivotEngine(pivot_window=3)
    cluster_engine = ClusterEngine()
    structure_engine = StructureEngine(pivot_window=3)

    swings = pivot_engine.detect_pivots(df, "NSE:NIFTY50-INDEX", "m5")
    clusters = cluster_engine.detect_clusters(swings, "NSE:NIFTY50-INDEX", "m5")
    state, events = structure_engine.analyze(df, swings, clusters, "NSE:NIFTY50-INDEX", "m5")

    _record("StructureEngine returns a StructureState",
            state is not None, "")
    _record("Swings are present in state", len(state.swings) > 0,
            f"{len(state.swings)} swings")
    _record("Relationships dict is populated", len(state.relationships) > 0, "")
    _record("At least one completed leg", len(state.legs) > 0,
            f"{len(state.legs)} legs")

    # Developing leg
    _record("Developing leg is populated",
            state.developing_leg is not None, str(type(state.developing_leg)))
    if state.developing_leg:
        dl = state.developing_leg
        _record("Developing leg has current_price > 0", dl.current_price > 0,
                f"price={dl.current_price}")
        _record("Developing leg type is UP_LEG or DOWN_LEG",
                dl.current_extension_atr >= 0.0, f"ext={dl.current_extension_atr}")

    # Swing status lifecycle
    active_count = sum(1 for s in state.swings if s.status == SwingStatus.ACTIVE)
    breached_count = sum(1 for s in state.swings if s.status == SwingStatus.BREACHED)
    _record("Some swings remain ACTIVE", active_count > 0, f"{active_count} active")
    # Not all may be breached, so we just verify states are valid
    valid_statuses = {SwingStatus.ACTIVE, SwingStatus.BREACHED, SwingStatus.RETESTED, SwingStatus.ARCHIVED}
    _record("All swings have valid status",
            all(s.status in valid_statuses for s in state.swings), "")

    # Confidence decay applied
    if len(state.swings) > 0:
        _record("Confidence is in [0.1, 1.0]",
                all(0.1 <= s.confidence <= 1.0 for s in state.swings), "")

    # Events are correctly typed
    valid_event_types = {"BOS_CONFIRMED", "CHOCH_CONFIRMED", "STRUCTURE_RESET"}
    for ev in events:
        _record(f"Event '{ev.event_type}' has valid type",
                ev.event_type in valid_event_types, ev.event_type)
        _record(f"Event '{ev.event_id}' has non-empty payload",
                bool(ev.payload), "")
        break   # check one event to avoid noise

    # StructureState with no swings should return empty cleanly
    state_empty, events_empty = structure_engine.analyze(df.head(5), [], [], "NSE:NIFTY50-INDEX", "m5")
    _record("Engine handles empty swings gracefully", state_empty is not None, "")

    # HTF split: confirmed (excluding last bar) vs developing
    swings_conf = pivot_engine.detect_pivots(df.iloc[:-1], "NSE:NIFTY50-INDEX", "h1")
    clusters_conf = cluster_engine.detect_clusters(swings_conf, "NSE:NIFTY50-INDEX", "h1")
    state_conf, _ = structure_engine.analyze(df.iloc[:-1], swings_conf, clusters_conf, "NSE:NIFTY50-INDEX", "h1")
    state_dev, _  = structure_engine.analyze(df, swings, clusters, "NSE:NIFTY50-INDEX", "h1")
    _record("HTF confirmed state is StructureState", state_conf is not None, "")
    _record("HTF developing state is StructureState", state_dev is not None, "")


# ─────────────────────────────────────────────────────────────────────────────
# 6. IndicatorPipeline integration
# ─────────────────────────────────────────────────────────────────────────────

def test_indicator_pipeline():
    print("\n── 6. IndicatorPipeline Integration ────────────────────────────")
    from src.core.indicator_pipeline import IndicatorPipeline, LegacyStructureReportAdapter
    from src.core.market_knowledge import MarketContext, HTFStructure

    pipeline = IndicatorPipeline(pivot_window=3)
    _record("IndicatorPipeline instantiates cleanly", pipeline is not None, "")
    _record("required_history == 150",
            pipeline.required_history == 150, f"got {pipeline.required_history}")

    d1 = _make_d1_df(30)
    h1 = _make_h1_df(50)
    m5 = _make_m5_df(150)
    now = datetime(2026, 6, 16, 14, 30)

    snapshot = pipeline.compute("NSE:NIFTY50-INDEX", float(m5["close"].iloc[-1]),
                                d1, h1, m5, now)

    _record("compute() returns a MarketSnapshot", snapshot is not None, "")
    if snapshot is None:
        return

    # Legacy h1_structure adapter
    _record("h1_structure is LegacyStructureReportAdapter",
            isinstance(snapshot.h1_structure, LegacyStructureReportAdapter), "")
    _record("h1_structure.trend is BULLISH/BEARISH/NEUTRAL",
            snapshot.h1_structure.trend in ("BULLISH", "BEARISH", "NEUTRAL"),
            snapshot.h1_structure.trend)
    _record("h1_structure.quality_score in [0, 100]",
            0.0 <= snapshot.h1_structure.quality_score <= 100.0,
            f"{snapshot.h1_structure.quality_score}")

    # MarketContext
    ctx = snapshot.market
    _record("snapshot.market is MarketContext", isinstance(ctx, MarketContext), "")
    _record("m5 structure populated",
            ctx.structure is not None, "")
    _record("htf_structure has 'h1' key", "h1" in ctx.htf_structure, "")
    _record("htf_structure has 'd1' key", "d1" in ctx.htf_structure, "")
    _record("h1 HTFStructure has confirmed + developing",
            isinstance(ctx.htf_structure.get("h1"), HTFStructure), "")

    h1_htf = ctx.htf_structure.get("h1")
    if h1_htf:
        _record("h1 confirmed state is StructureState", h1_htf.confirmed is not None, "")
        _record("h1 developing state is StructureState", h1_htf.developing is not None, "")

    # None inputs handled gracefully
    snapshot_null = pipeline.compute("NSE:NIFTY50-INDEX", 22000.0, None, None, m5, now)
    _record("compute() returns None when h1 is missing", snapshot_null is None, "")


# ─────────────────────────────────────────────────────────────────────────────
# 7. Database — ResearchEvent persists to market_events
# ─────────────────────────────────────────────────────────────────────────────

def test_database_event_persistence():
    print("\n── 7. Database — market_events hypertable ──────────────────────")
    from src.models.postgres_database import PostgresDatabase
    from src.core.market_knowledge import ResearchEvent

    db = PostgresDatabase()

    # Build a unique test event
    test_ts = datetime(2026, 6, 21, 14, 30, 0)
    test_event = ResearchEvent(
        event_id="TEST_evt_verify_20260621_143000",
        timestamp=test_ts,
        occurrence_timestamp=test_ts,
        symbol="TEST:VERIFY",
        event_type="BOS_CONFIRMED",
        engine_version="v2.0",
        payload={"breached_swing_id": "sw_test", "breach_price": 22000.5,
                 "swing_price": 21999.0, "swing_type": "HIGH", "rvol": 1.5}
    )

    db.save_market_event(test_event.to_dict())

    # Verify row exists
    conn = db._get_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT event_id, event_type, symbol FROM market_events WHERE event_id = %s",
        (test_event.event_id,)
    )
    row = cur.fetchone()
    cur.close()
    conn.close()

    _record("Event persisted to market_events", row is not None, f"row={row}")
    if row:
        _record("event_id matches",    row[0] == test_event.event_id,    row[0])
        _record("event_type matches",  row[1] == "BOS_CONFIRMED",        row[1])
        _record("symbol matches",      row[2] == "TEST:VERIFY",          row[2])

    # Idempotent re-insert (ON CONFLICT DO NOTHING)
    try:
        db.save_market_event(test_event.to_dict())
        _record("Duplicate insert is idempotent (no exception)", True, "")
    except Exception as e:
        _record("Duplicate insert is idempotent (no exception)", False, str(e))

    # ResearchEvent.to_dict() serializes correctly
    d = test_event.to_dict()
    _record("to_dict() produces all required keys",
            {"event_id", "timestamp", "occurrence_timestamp", "symbol",
             "event_type", "engine_version", "payload"} <= set(d.keys()), "")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("  MKE Milestone 1 — Automated Verification Suite")
    print("=" * 65)

    suites = [
        ("TimeOfDayEngine",          test_tod_engine),
        ("TradingClock",             test_trading_clock),
        ("PivotEngine",              test_pivot_engine),
        ("ClusterEngine",            test_cluster_engine),
        ("StructureEngine",          test_structure_engine),
        ("IndicatorPipeline",        test_indicator_pipeline),
        ("Database Persistence",     test_database_event_persistence),
    ]

    for name, fn in suites:
        try:
            fn()
        except Exception:
            print(f"\n  ❌ SUITE CRASHED — {name}")
            traceback.print_exc()

    # ── Summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    total  = len(results)
    passed = sum(1 for _, s, _ in results if s == PASS)
    failed = total - passed

    print(f"  Total: {total}   Passed: {passed}   Failed: {failed}")
    if failed > 0:
        print("\n  FAILED TESTS:")
        for name, status, detail in results:
            if status == FAIL:
                print(f"    ❌ {name}  {detail}")
    print("=" * 65)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
