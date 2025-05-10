-- Create a view that combines all strategy results with performance metrics
-- Run this file with: sqlite3 trading_signals.db < strategy_performance_view.sql

-- Drop existing views
DROP VIEW IF EXISTS strategy_performance;
DROP VIEW IF EXISTS strategy_performance_summary;
DROP VIEW IF EXISTS strategy_signal_distribution;
DROP VIEW IF EXISTS strategy_signal_criteria;
DROP VIEW IF EXISTS recent_signals;

-- Create the consolidated view
CREATE VIEW strategy_performance AS
WITH all_signals AS (
    -- Breakout RSI
    SELECT 
        'breakout_rsi' AS strategy_name,
        signal_time,
        index_name,
        signal,
        price,
        confidence,
        rsi,
        NULL AS macd,
        NULL AS macd_signal,
        NULL AS ema_20,
        NULL AS supertrend_direction,
        NULL AS bollinger_width,
        NULL AS channel_width,
        rsi_reason,
        price_reason,
        NULL AS macd_reason,
        CAST(stop_loss AS TEXT) AS stop_loss,
        CAST(target AS TEXT) AS target,
        outcome,
        pnl,
        targets_hit,
        stoploss_count,
        failure_reason
    FROM breakout_rsi
    WHERE signal != 'NO TRADE'
    
    UNION ALL
    
    -- Inside Bar RSI
    SELECT 
        'insidebar_rsi' AS strategy_name,
        signal_time,
        index_name,
        signal,
        price,
        confidence,
        rsi,
        NULL AS macd,
        NULL AS macd_signal,
        ema_20,
        NULL AS supertrend_direction,
        NULL AS bollinger_width,
        NULL AS channel_width,
        rsi_reason,
        price_reason,
        NULL AS macd_reason,
        CAST(stop_loss AS TEXT) AS stop_loss,
        CAST(target AS TEXT) AS target,
        outcome,
        pnl,
        targets_hit,
        stoploss_count,
        failure_reason
    FROM insidebar_rsi
    WHERE signal != 'NO TRADE'
    
    UNION ALL
    
    -- Inside Bar Bollinger
    SELECT 
        'insidebar_bollinger' AS strategy_name,
        signal_time,
        index_name,
        signal,
        price,
        confidence,
        NULL AS rsi,
        NULL AS macd,
        NULL AS macd_signal,
        NULL AS ema_20,
        NULL AS supertrend_direction,
        bollinger_width,
        NULL AS channel_width,
        rsi_reason,
        price_reason,
        NULL AS macd_reason,
        'N/A' AS stop_loss,
        'N/A' AS target,
        outcome,
        pnl,
        targets_hit,
        stoploss_count,
        failure_reason
    FROM insidebar_bollinger
    WHERE signal != 'NO TRADE'
    
    UNION ALL
    
    -- Supertrend EMA
    SELECT 
        'supertrend_ema' AS strategy_name,
        signal_time,
        index_name,
        signal,
        price,
        confidence,
        rsi,
        macd,
        macd_signal,
        ema_20,
        NULL AS supertrend_direction,
        NULL AS bollinger_width,
        NULL AS channel_width,
        rsi_reason,
        price_reason,
        macd_reason,
        CAST(stop_loss AS TEXT) AS stop_loss,
        CAST(target AS TEXT) AS target,
        outcome,
        pnl,
        targets_hit,
        stoploss_count,
        failure_reason
    FROM supertrend_ema
    WHERE signal != 'NO TRADE'
    
    UNION ALL
    
    -- Supertrend MACD RSI EMA
    SELECT 
        'supertrend_macd_rsi_ema' AS strategy_name,
        signal_time,
        index_name,
        signal,
        price,
        confidence,
        rsi,
        macd,
        macd_signal,
        ema_20,
        supertrend_direction,
        NULL AS bollinger_width,
        NULL AS channel_width,
        rsi_reason,
        price_reason,
        macd_reason,
        CAST(stop_loss AS TEXT) AS stop_loss,
        CAST(target AS TEXT) AS target,
        outcome,
        pnl,
        targets_hit,
        stoploss_count,
        failure_reason
    FROM supertrend_macd_rsi_ema
    WHERE signal != 'NO TRADE'
    
    UNION ALL
    
    -- Donchian Breakout
    SELECT 
        'donchian_breakout' AS strategy_name,
        signal_time,
        index_name,
        signal,
        price,
        confidence,
        NULL AS rsi,
        NULL AS macd,
        NULL AS macd_signal,
        NULL AS ema_20,
        NULL AS supertrend_direction,
        NULL AS bollinger_width,
        channel_width,
        rsi_reason,
        price_reason,
        NULL AS macd_reason,
        CAST(stop_loss AS TEXT) AS stop_loss,
        CAST(target AS TEXT) AS target,
        outcome,
        pnl,
        targets_hit,
        stoploss_count,
        failure_reason
    FROM donchian_breakout
    WHERE signal != 'NO TRADE'
    
    UNION ALL
    
    -- Range Breakout Volatility
    SELECT 
        'range_breakout_volatility' AS strategy_name,
        signal_time,
        index_name,
        signal,
        price,
        confidence,
        NULL AS rsi,
        NULL AS macd,
        NULL AS macd_signal,
        NULL AS ema_20,
        NULL AS supertrend_direction,
        NULL AS bollinger_width,
        NULL AS channel_width,
        rsi_reason,
        price_reason,
        NULL AS macd_reason,
        CAST(stop_loss AS TEXT) AS stop_loss,
        CAST(target AS TEXT) AS target,
        outcome,
        pnl,
        targets_hit,
        stoploss_count,
        failure_reason
    FROM range_breakout_volatility
    WHERE signal != 'NO TRADE'
    
    UNION ALL
    
    -- EMA Crossover
    SELECT 
        'ema_crossover' AS strategy_name,
        signal_time,
        index_name,
        signal,
        price,
        confidence,
        rsi,
        macd,
        macd_signal,
        ema_20,
        NULL AS supertrend_direction,
        NULL AS bollinger_width,
        NULL AS channel_width,
        rsi_reason,
        price_reason,
        macd_reason,
        CAST(stop_loss AS TEXT) AS stop_loss,
        CAST(target AS TEXT) AS target,
        outcome,
        pnl,
        targets_hit,
        stoploss_count,
        failure_reason
    FROM ema_crossover
    WHERE signal != 'NO TRADE'
)

SELECT 
    strategy_name,
    signal_time,
    index_name,
    signal,
    price,
    confidence,
    rsi,
    macd,
    macd_signal,
    ema_20,
    supertrend_direction,
    bollinger_width,
    channel_width,
    rsi_reason,
    price_reason,
    macd_reason,
    stop_loss,
    target,
    outcome,
    pnl,
    targets_hit,
    stoploss_count,
    failure_reason
FROM all_signals;

-- Create a summary view for strategy performance metrics
CREATE VIEW strategy_performance_summary AS
SELECT 
    strategy_name,
    COUNT(*) AS total_trades,
    SUM(CASE WHEN outcome = 'Win' THEN 1 ELSE 0 END) AS wins,
    SUM(CASE WHEN outcome = 'Loss' THEN 1 ELSE 0 END) AS losses,
    SUM(CASE WHEN outcome = 'Pending' THEN 1 ELSE 0 END) AS pending,
    ROUND(SUM(CASE WHEN outcome = 'Win' THEN 1 ELSE 0 END) * 100.0 / 
          NULLIF(SUM(CASE WHEN outcome IN ('Win', 'Loss') THEN 1 ELSE 0 END), 0), 2) AS win_rate,
    ROUND(SUM(pnl), 2) AS total_pnl,
    ROUND(AVG(CASE WHEN outcome = 'Win' THEN pnl ELSE NULL END), 2) AS avg_win,
    ROUND(AVG(CASE WHEN outcome = 'Loss' THEN pnl ELSE NULL END), 2) AS avg_loss,
    ROUND(SUM(CASE WHEN outcome = 'Win' THEN pnl ELSE 0 END) / 
          ABS(NULLIF(SUM(CASE WHEN outcome = 'Loss' THEN pnl ELSE 0 END), 0)), 2) AS profit_factor
FROM strategy_performance
GROUP BY strategy_name
ORDER BY total_pnl DESC;

-- Create a view for strategy signal distribution by index
CREATE VIEW strategy_signal_distribution AS
SELECT 
    strategy_name,
    index_name,
    signal,
    COUNT(*) AS count,
    ROUND(COUNT(*) * 100.0 / 
          SUM(COUNT(*)) OVER (PARTITION BY strategy_name, index_name), 2) AS percentage,
    ROUND(SUM(CASE WHEN outcome = 'Win' THEN 1 ELSE 0 END) * 100.0 / 
          NULLIF(COUNT(CASE WHEN outcome IN ('Win', 'Loss') THEN 1 ELSE NULL END), 0), 2) AS win_rate,
    ROUND(SUM(pnl), 2) AS total_pnl
FROM strategy_performance
GROUP BY strategy_name, index_name, signal
ORDER BY strategy_name, index_name, count DESC;

-- Create a view showing the criteria/conditions for each strategy's signals
CREATE VIEW strategy_signal_criteria AS
SELECT DISTINCT
    strategy_name,
    signal,
    COALESCE(rsi_reason, '') AS rsi_criteria,
    COALESCE(macd_reason, '') AS macd_criteria,
    COALESCE(price_reason, '') AS price_criteria
FROM strategy_performance
WHERE signal != 'NO TRADE'
GROUP BY strategy_name, signal
ORDER BY strategy_name, signal;

-- Create a view for recent signals with performance (last 7 days)
CREATE VIEW recent_signals AS
SELECT 
    strategy_name,
    signal_time,
    index_name,
    signal,
    price,
    confidence,
    outcome,
    pnl,
    targets_hit,
    stoploss_count,
    failure_reason,
    rsi_reason,
    price_reason,
    macd_reason
FROM strategy_performance
WHERE signal_time >= datetime('now', '-7 days')
ORDER BY signal_time DESC; 