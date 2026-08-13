# 📊 Weekly Trading Log Analysis Report

This report compiles a detailed, day-by-day analysis of the system's paper-trading logs and execution journals for the week of August 10, 2026.

## 📅 Summary Timeline

* **Monday, Aug 10:** No log file was generated; trading systems were inactive.
* **Tuesday, Aug 11:** Main session ran from 09:35 to 15:55 IST. Balanced day ending near break-even.
* **Wednesday, Aug 12:** Session ran from 08:56 to 14:14 IST. System terminated prematurely.
* **Thursday, Aug 13:** Extended session ran from 09:13 to 20:15 IST. High trading frequency with multiple re-connections.

---

## 📅 Tuesday, August 11, 2026

### ⚙️ System Status & Initializations
* **Session Window:** `09:35:46` to `15:55:10 IST` (Full day session).
* **Fyers Client Connection:** 
  * Initially failed twice at `09:35` with error: `Your token has expired. Please generate a token`.
  * Successfully initialized at `09:36:11` using local JSON cache token file (`token_2026-08-11.json`).
* **PostgreSQL / TimescaleDB:** Checked and fully initialized (10 distinct connection check-ins).
* **Option Warehouse:** Background thread started successfully at `09:35` and `09:36` monitoring `NSE:NIFTY50-INDEX` and `NSE:NIFTYBANK-INDEX` depth 5 strikes.
* **Experiments Registered:** 19 active strategy pipelines running in parallel.

### 📈 Trading Session Statistics

#### Real Trades
| Metrics | Total Trades | Wins | Losses | Win Rate | Net PnL | Expectancy |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Stats** | 19 | 8 | 11 | 42% | **-0.01R** | **+-0.00R** |

* **Best Experiment:** `VWAP_Reversion_1.5ATR_RVOL1.0` (+1.55R)
* **Worst Experiment:** `CreditSpread_v1.0_PCRFade` (+0.00R expectancy)

#### Counterfactual Research (Shadow Trades)
| Metrics | Total Trades | Wins (Positive) | Losses (Negative) | Positive Rate | Net PnL | Expectancy |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Stats** | 158 (Log counts 172) | 37 | 121 | 23% | **-60.92R** | **-0.39R** |

> [!NOTE]
> The massive negative PnL in counterfactual shadow trades validates the effectiveness of the filters (e.g. RVOL, Efficiency, Wickiness) which kept the real portfolio out of these loss-making setups.

### 📝 Live Trade Details

1. **Trade #1 & #2 (Confluence Bounce Put on NIFTY50):** Entry at `10:10:13` (24,479.55). Hit trailing stop at `10:30:10` (24,477.43) for **`+0.17R`** profit.
2. **Trade #3 (OI Wall Fade Call on NIFTY50):** Entry at `10:30:13` (24,479.00). Hit trailing stop at `10:40:11` (24,472.84) for a loss of **`-0.83R`**.
3. **Trade #4 (OI Wall Fade Call on NIFTY50):** Entry at `10:55:12` (24,476.70). Hit initial stop at `11:00:10` (24,469.81) for a loss of **`-1.05R`**.
4. **Trade #5 (VWAP Reclaim Call on BANKNIFTY):** Entry at `10:55:15` (57,301.80). Hit trailing stop at `13:15:10` (57,356.03) for a profit of **`+0.80R`**.
5. **Trade #6 (Structural Call on NIFTY50):** Entry at `12:05:13` (24,431.05). Hit trailing stop at `12:20:10` (24,440.32) for a profit of **`+1.55R`**.
6. **Trade #7 (OrderFlow Put on NIFTY50):** Entry at `12:25:14` (24,431.85). Hit initial stop at `12:30:12` (24,437.56) for a loss of **`-1.05R`**.
7. **Trade #8 (ATR Squeeze Put on BANKNIFTY):** Entry at `12:45:17` (57,328.80). Hit initial stop at `12:50:10` (57,347.79) for a loss of **`-1.05R`**.
8. **Trade #9 (OrderFlow Call on BANKNIFTY):** Entry at `12:45:17` (57,328.80). Hit trailing stop at `13:00:10` (57,337.76) for a profit of **`+0.42R`**.
9. **Trade #10 (Geometry Put on BANKNIFTY):** Entry at `13:00:17` (57,327.90). Hit initial stop at `13:05:10` (57,347.99) for a loss of **`-1.05R`**.
10. **Trade #11 (Structural Put on NIFTY50):** Entry at `13:15:14` (24,453.15). Hit trailing stop at `13:30:11` (24,462.20) for a loss of **`-0.79R`**.
11. **Trade #12 (Structural Call on NIFTY50):** Entry at `13:30:15` (24,455.05). Hit trailing stop at `13:40:11` (24,464.29) for a profit of **`+1.29R`**.
12. **Option Strategies:**
    * **CreditSpread Bear Call Spread (NIFTY50):** Entry at `09:40`. Exited at `15:25` (session end) for **`+0.36R`**.
    * **VerticalSpread Bear Put Spread (BANKNIFTY):** Entry at `09:45`. Exited at `15:25` for a loss of **`-0.17R`**.
    * **Straddle Vol Compression (NIFTY50):** Entry at `11:00`. Hit stop loss at `15:20` for **`-0.69R`**.
    * **Strangle Vol Compression (NIFTY50):** Entry at `11:00`. Hit stop loss at `13:15` for **`-0.50R`**.
    * **Straddle / Strangle (BANKNIFTY):** Entered at `12:05`. Exited at `15:25` for **`-0.04R`** and **`-0.04R`** respectively.

### ⚠️ Warnings & Anomalies
* **Risk Governor Restriction:** A warning occurred at `10:30` blocking a potential real trade on `NSE:NIFTY50-INDEX` due to `LEVEL_REPEAT_CAP(2x@24500)`.

---

## 📅 Wednesday, August 12, 2026

### ⚙️ System Status & Initializations
* **Session Window:** `08:56:25` to `14:14:53 IST`. 

> [!WARNING]
> The logs for this day cut off at 14:14 IST. This indicates a premature system termination or crash. There was no EOD report generated.

* **Fyers Client Connection:** Successfully initialized at `08:56:26` and re-established at `09:03:07` from cache.
* **PostgreSQL / TimescaleDB:** 10 connection check-ins succeeded.
* **Option Warehouse:** Started successfully for both indices.
* **Experiments Registered:** 21 active experiments.

### 📈 Trading Session Statistics

#### Real Trades
* **Total Entries logged:** 27
* **Total Exits logged before crash:** 19
* Net expectancy and total PnL were not compiled due to missing EOD reporting.

#### Counterfactual Research (Shadow Trades)
* **Total shadow entries logged:** 155
* **Total shadow exits logged:** 145

### 📝 Live Trade Details (Logged Actions)
1. **Trade #1 (EMA Pullback Call on BANKNIFTY):** Entered at `09:15:45` (57,446.25). Hit initial stop at `09:20:10` (57,356.37) for a loss of **`-1.05R`**.
2. **Trade #2 & #3 (Confluence Bounce Call on NIFTY50):** Entered at `09:25:13` (24,423.00). Hit trailing stop at `09:35:11` (24,448.16) for a profit of **`+2.67R`** each.
3. **Trade #4 & #5 (Structural Call on NIFTY50):** Entered at `09:50:14` (24,402.55). Hit trailing stop at `10:00:11` (24,398.16) for a loss of **`-0.42R`** each.
4. **Trade #6 (OI Wall Break Put on NIFTY50):** Entered at `10:00:14` (24,378.55). Hit trailing stop at `10:35:10` (24,387.84) for a loss of **`-0.42R`**.
5. **Trade #7 (Confluence Bounce Put on NIFTY50):** Entered at `10:30:14` (24,371.50). Hit initial stop at `10:35:10` (24,381.95) for a loss of **`-1.05R`**.
6. **Trade #8 & #9 (Structural Call on NIFTY50):** Entered at `10:55:13` (24,359.30). Hit initial stop at `11:00:10` (24,355.45) for a loss of **`-1.05R`** each.
7. **Trade #10 (PrevDay Extremes Reversal Put on BANKNIFTY):** Entered at `11:25:17` (57,587.55). Hit trailing stop at `12:35:10` (57,535.67) for a profit of **`+0.77R`**.
8. **Trade #11 (Structural Call on NIFTY50):** Entered at `11:45:14` (24,294.20). Hit initial stop at `12:00:11` (24,277.25) for a loss of **`-1.05R`** each.
9. **Trade #12 (OI Wall Break Put on NIFTY50):** Entered at `11:55:14` (24,282.10). Hit trailing stop at `12:35:11` (24,287.85) for a loss of **`-0.33R`**.
10. **Trade #13 (Geometry Put on NIFTY50):** Entered at `12:15:17` (24,267.10). Hit initial stop at `12:20:10` (24,277.85) for a loss of **`-1.05R`**.
11. **Trade #14 & #15 (Structural Call on BANKNIFTY):** Entered at `12:30:18` (57,516.50). Hit trailing stop at `12:55:10` (57,506.00) for a loss of **`-0.33R`** each.
12. **Trade #16 (ATR Squeeze Put on BANKNIFTY):** Entered at `13:05:18` (57,517.10). Hit trailing stop at `13:20:10` (57,519.64) for a loss of **`-0.18R`**.
13. **Trade #17 (OI Wall Fade Call on NIFTY50):** Entered at `13:15:14` (24,288.25). Hit trailing stop at `13:30:10` (24,290.37) for a profit of **`+0.32R`**.
14. **Trade #18 (ATR Squeeze Put on BANKNIFTY):** Entered at `13:20:18` (57,510.25). Hit trailing stop at `13:40:10` (57,526.87) for a loss of **`-0.92R`**.

### ⚠️ Warnings & Anomalies
* Option chain strategies (`CreditSpread_v1.0_PCRFade`, `OIWallReaction_v1.0`, `PCRExtremeReversal_v1.0`) regularly reported `OPTIONS_DATA_STALE` errors.

---

## 📅 Thursday, August 13, 2026

### ⚙️ System Status & Initializations
* **Session Window:** `09:13:37` to `20:15:05 IST` (Extended session).
* **Fyers Client Connection:** Verbose initializations; the system had to reconnect numerous times due to network drops/API call timeouts.
* **PostgreSQL / TimescaleDB:** Succeeded with 89 connections but registered 6 connection refused errors at startup:
  `Postgres Init Failed: connection to server at "127.0.0.1", port 5433 failed: Connection refused`
  A database deadlock warning occurred at `09:57:40` (`Postgres Init Failed: deadlock detected`).
* **Option Warehouse:** Re-launched multiple times in sync with network reconnects.
* **Experiments Registered:** 24 active experiments (highest of the week).

### 📈 Trading Session Statistics

#### Real Trades
| Metrics | Total Trades | Wins | Losses | Win Rate | Net PnL | Expectancy |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Stats** | 33 | 10 | 21 | 30% | **-2.05R** | **-0.06R** |

* **Best Experiment:** `OIWallReaction_v1.0` (+0.28R)
* **Worst Experiment:** `ORB_60m_IB_RVOL1.2` (+0.00R expectancy)

#### Counterfactual Research (Shadow Trades)
| Metrics | Total Trades | Wins (Positive) | Losses (Negative) | Positive Rate | Net PnL | Expectancy |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Stats** | 145 (Log counts 260) | 33 | 112 | 23% | **-47.46R** | **-0.33R** |

### 📝 Live Trade Details

#### Directional Index Trades
1. **Trade #1 (Structural Sweep Call on NIFTY50):** Entry at `10:40:14` (24,327.10). Hit trailing stop at `10:50:12` (24,330.90) for a profit of **`+0.34R`**.
2. **Trade #2 (OrderFlow Sweep Call on NIFTY50):** Entry at `12:00:15` (24,386.45). Hit trailing stop at `12:05:11` (24,387.60) for a profit of **`+0.12R`**.
3. **Trade #3 & #4 (Geometry Trendline Retest Put on NIFTY50):** Entry at `12:20:16` (24,392.15). Hit trailing stop at `12:24:35` (24,400.45) for a loss of **`-0.96R`** each.
4. **Trade #5 (OI Wall Fade Call on NIFTY50):** Entry at `12:25:16` (24,399.20). Hit trailing stop at `12:28:36` (24,393.80) for a loss of **`-0.80R`**.
5. **Trade #6 (VWAP Reclaim Put on NIFTY50):** Entry at `13:20:16` (24,355.25). Hit trailing stop at `13:42:05` (24,370.70) for a loss of **`-0.84R`**.

#### Option Combination Trades (Butterfly Spreads on BANKNIFTY)
* **09:35 Entry:** Stopped out at `10:05` (Loss: **`-0.68R`**).
* **09:35 Entry (re-run):** Stopped out at `10:05` (Loss: **`-0.68R`**).
* **10:05 Entry:** Stopped out at `10:15` (Loss: **`-0.51R`**).
* **10:20 Entry:** Stopped out at `10:42` (Loss: **`-0.50R`**).
* **10:50 Entry:** Stopped out at `10:54` (Loss: **`-0.69R`**).
* **10:55 Entry:** Hit Target Profit at `10:59` (Profit: **`+1.62R`**).
* **11:00 Entry:** Stopped out at `11:10` (Loss: **`-0.55R`**).
* **11:10 Entry:** Hit Target Profit at `11:22` (Profit: **`+1.77R`**).
* **13:50 Entry:** Hit Target Profit at `13:52` (Profit: **`+1.91R`**).
* **13:55 Entry:** Stopped out at `14:13` (Loss: **`-0.58R`**).

### ⚠️ Warnings & Anomalies
* **Premarket Fetch Errors:** The `PreMarketCollector` regularly warned that the previous day's OHLC data fetch failed:
  `get_historical_data() got an unexpected keyword argument 'days'`
  This caused it to skip premarket calculations for `NSE:NIFTY50-INDEX` and abort the pre-open window.
* **Insufficient OI Data:** Strategy `OI_Scalping_v1.0` triggered 292 warnings due to stale or missing order flow records: `errors: ['INSUFFICIENT_OI_DATA']`.
