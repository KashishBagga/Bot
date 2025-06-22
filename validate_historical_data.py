#!/usr/bin/env python3
"""
Historical Data Validator and Quick Access
Validates fetched 20-year data and provides easy access for backtesting
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import json
import logging
from pathlib import Path
import pickle
import gzip

class HistoricalDataValidator:
    """Validate and provide quick access to historical data"""
    
    def __init__(self, data_dir: str = "historical_data_20yr"):
        """Initialize the data validator"""
        self.data_dir = Path(data_dir)
        self.setup_logging()
        
        # Expected timeframes
        self.timeframes = ['1min', '3min', '5min', '15min', '30min', '1hour', '4hour', '1day', '1week']
        
        # Data quality metrics
        self.quality_report = {
            'validation_time': datetime.now().isoformat(),
            'symbols': {},
            'overall_quality': {},
            'recommendations': []
        }
    
    def setup_logging(self):
        """Setup logging"""
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger('DataValidator')
    
    def load_data(self, symbol: str, timeframe: str, format_type: str = 'parquet') -> Optional[pd.DataFrame]:
        """Load historical data for a symbol and timeframe
        
        Args:
            symbol: Symbol name (e.g., 'NIFTY50')
            timeframe: Timeframe (e.g., '5min')
            format_type: 'parquet' (fastest), 'pickle' (medium), 'csv' (slowest)
        
        Returns:
            DataFrame with historical data
        """
        symbol_dir = self.data_dir / symbol
        
        if format_type == 'parquet':
            file_path = symbol_dir / f"{symbol}_{timeframe}_20yr.parquet"
            if file_path.exists():
                return pd.read_parquet(file_path)
        elif format_type == 'pickle':
            file_path = symbol_dir / f"{symbol}_{timeframe}_20yr.pkl.gz"
            if file_path.exists():
                with gzip.open(file_path, 'rb') as f:
                    return pickle.load(f)
        elif format_type == 'csv':
            file_path = symbol_dir / f"{symbol}_{timeframe}_20yr.csv"
            if file_path.exists():
                df = pd.read_csv(file_path, index_col='time', parse_dates=True)
                return df
        
        self.logger.warning(f"Data file not found: {symbol} {timeframe} ({format_type})")
        return None
    
    def get_available_symbols(self) -> List[str]:
        """Get list of available symbols"""
        symbols = []
        if self.data_dir.exists():
            for item in self.data_dir.iterdir():
                if item.is_dir() and not item.name.startswith('.'):
                    symbols.append(item.name)
        return sorted(symbols)
    
    def get_available_timeframes(self, symbol: str) -> List[str]:
        """Get available timeframes for a symbol"""
        symbol_dir = self.data_dir / symbol
        timeframes = []
        
        if symbol_dir.exists():
            for file_path in symbol_dir.glob(f"{symbol}_*_20yr.parquet"):
                # Extract timeframe from filename
                parts = file_path.stem.split('_')
                if len(parts) >= 3:
                    timeframe = parts[-2]  # Second to last part
                    timeframes.append(timeframe)
        
        return sorted(timeframes)
    
    def validate_data_quality(self, symbol: str, timeframe: str) -> Dict:
        """Validate data quality for a symbol and timeframe"""
        df = self.load_data(symbol, timeframe)
        
        if df is None:
            return {
                'available': False,
                'error': 'Data file not found'
            }
        
        if df.empty:
            return {
                'available': True,
                'quality': 'POOR',
                'error': 'DataFrame is empty'
            }
        
        # Calculate quality metrics
        total_rows = len(df)
        
        # Check for missing values
        missing_data = df.isnull().sum()
        missing_percentage = (missing_data.sum() / (total_rows * len(df.columns))) * 100
        
        # Check for duplicate timestamps
        duplicate_timestamps = df.index.duplicated().sum()
        
        # Check date range
        date_range = {
            'start': df.index.min(),
            'end': df.index.max(),
            'total_days': (df.index.max() - df.index.min()).days
        }
        
        # Check for gaps in data
        expected_frequency = self.get_expected_frequency(timeframe)
        if expected_frequency:
            expected_periods = pd.date_range(
                start=df.index.min(),
                end=df.index.max(),
                freq=expected_frequency
            )
            # Filter for market hours if intraday
            if timeframe != '1day' and timeframe != '1week':
                expected_periods = expected_periods[
                    (expected_periods.hour >= 9) & 
                    (expected_periods.hour < 16) &
                    (expected_periods.weekday < 5)
                ]
            
            missing_periods = len(expected_periods) - len(df)
            data_completeness = (len(df) / len(expected_periods)) * 100 if len(expected_periods) > 0 else 0
        else:
            missing_periods = 0
            data_completeness = 100
        
        # Price data validation
        price_issues = 0
        if 'open' in df.columns and 'high' in df.columns and 'low' in df.columns and 'close' in df.columns:
            # Check OHLC logic
            invalid_ohlc = (
                (df['high'] < df['open']) | 
                (df['high'] < df['close']) | 
                (df['low'] > df['open']) | 
                (df['low'] > df['close']) |
                (df['high'] < df['low'])
            ).sum()
            price_issues += invalid_ohlc
        
        # Zero or negative prices
        if 'close' in df.columns:
            invalid_prices = (df['close'] <= 0).sum()
            price_issues += invalid_prices
        
        # Volume validation
        volume_issues = 0
        if 'volume' in df.columns:
            negative_volume = (df['volume'] < 0).sum()
            volume_issues += negative_volume
        
        # Quality score calculation
        quality_score = 100
        quality_score -= min(missing_percentage * 2, 20)  # Max 20 points for missing data
        quality_score -= min((duplicate_timestamps / total_rows) * 100, 10)  # Max 10 points for duplicates
        quality_score -= min((price_issues / total_rows) * 100, 15)  # Max 15 points for price issues
        quality_score -= min((volume_issues / total_rows) * 100, 5)  # Max 5 points for volume issues
        quality_score -= min((100 - data_completeness) * 0.5, 20)  # Max 20 points for incompleteness
        
        # Determine quality rating
        if quality_score >= 90:
            quality_rating = 'EXCELLENT'
        elif quality_score >= 80:
            quality_rating = 'GOOD'
        elif quality_score >= 70:
            quality_rating = 'FAIR'
        elif quality_score >= 60:
            quality_rating = 'POOR'
        else:
            quality_rating = 'VERY_POOR'
        
        return {
            'available': True,
            'quality': quality_rating,
            'quality_score': round(quality_score, 2),
            'metrics': {
                'total_rows': total_rows,
                'missing_data_percentage': round(missing_percentage, 2),
                'duplicate_timestamps': duplicate_timestamps,
                'price_issues': price_issues,
                'volume_issues': volume_issues,
                'data_completeness': round(data_completeness, 2),
                'missing_periods': missing_periods,
                'date_range': {
                    'start': date_range['start'].isoformat(),
                    'end': date_range['end'].isoformat(),
                    'total_days': date_range['total_days']
                }
            }
        }
    
    def get_expected_frequency(self, timeframe: str) -> Optional[str]:
        """Get expected pandas frequency for timeframe"""
        freq_map = {
            '1min': '1min',
            '3min': '3min',
            '5min': '5min',
            '15min': '15min',
            '30min': '30min',
            '1hour': '1H',
            '4hour': '4H',
            '1day': '1D',
            '1week': '1W'
        }
        return freq_map.get(timeframe)
    
    def validate_all_data(self) -> Dict:
        """Validate all available data"""
        symbols = self.get_available_symbols()
        
        self.logger.info(f"🔍 Validating data for {len(symbols)} symbols")
        
        overall_stats = {
            'total_datasets': 0,
            'excellent_datasets': 0,
            'good_datasets': 0,
            'fair_datasets': 0,
            'poor_datasets': 0,
            'very_poor_datasets': 0,
            'missing_datasets': 0
        }
        
        for symbol in symbols:
            self.logger.info(f"📊 Validating {symbol}...")
            self.quality_report['symbols'][symbol] = {}
            
            timeframes = self.get_available_timeframes(symbol)
            
            for timeframe in self.timeframes:
                validation_result = self.validate_data_quality(symbol, timeframe)
                self.quality_report['symbols'][symbol][timeframe] = validation_result
                
                overall_stats['total_datasets'] += 1
                
                if not validation_result['available']:
                    overall_stats['missing_datasets'] += 1
                else:
                    quality = validation_result.get('quality', 'UNKNOWN')
                    if quality == 'EXCELLENT':
                        overall_stats['excellent_datasets'] += 1
                    elif quality == 'GOOD':
                        overall_stats['good_datasets'] += 1
                    elif quality == 'FAIR':
                        overall_stats['fair_datasets'] += 1
                    elif quality == 'POOR':
                        overall_stats['poor_datasets'] += 1
                    else:
                        overall_stats['very_poor_datasets'] += 1
        
        # Calculate overall quality percentage
        total_available = overall_stats['total_datasets'] - overall_stats['missing_datasets']
        if total_available > 0:
            overall_quality_score = (
                (overall_stats['excellent_datasets'] * 100 +
                 overall_stats['good_datasets'] * 80 +
                 overall_stats['fair_datasets'] * 70 +
                 overall_stats['poor_datasets'] * 60 +
                 overall_stats['very_poor_datasets'] * 40) / total_available
            )
        else:
            overall_quality_score = 0
        
        self.quality_report['overall_quality'] = {
            'score': round(overall_quality_score, 2),
            'statistics': overall_stats
        }
        
        # Generate recommendations
        self.generate_recommendations()
        
        return self.quality_report
    
    def generate_recommendations(self):
        """Generate recommendations based on validation results"""
        recommendations = []
        stats = self.quality_report['overall_quality']['statistics']
        
        if stats['missing_datasets'] > 0:
            recommendations.append(
                f"⚠️ {stats['missing_datasets']} datasets are missing. "
                f"Consider re-running the data fetch for missing timeframes."
            )
        
        if stats['very_poor_datasets'] > 0:
            recommendations.append(
                f"🔴 {stats['very_poor_datasets']} datasets have very poor quality. "
                f"These should be re-fetched or excluded from backtesting."
            )
        
        if stats['poor_datasets'] > 0:
            recommendations.append(
                f"🟡 {stats['poor_datasets']} datasets have poor quality. "
                f"Review and consider re-fetching these datasets."
            )
        
        excellent_percentage = (stats['excellent_datasets'] / stats['total_datasets']) * 100
        if excellent_percentage >= 80:
            recommendations.append(
                f"✅ {excellent_percentage:.1f}% of datasets are excellent quality. "
                f"Data is ready for comprehensive backtesting."
            )
        elif excellent_percentage >= 60:
            recommendations.append(
                f"✅ {excellent_percentage:.1f}% of datasets are excellent quality. "
                f"Data is suitable for backtesting with some limitations."
            )
        else:
            recommendations.append(
                f"⚠️ Only {excellent_percentage:.1f}% of datasets are excellent quality. "
                f"Consider improving data quality before extensive backtesting."
            )
        
        self.quality_report['recommendations'] = recommendations
    
    def save_validation_report(self, filename: str = None):
        """Save validation report to file"""
        if filename is None:
            filename = f"data_validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        report_path = self.data_dir / filename
        with open(report_path, 'w') as f:
            json.dump(self.quality_report, f, indent=2, default=str)
        
        self.logger.info(f"📄 Validation report saved to: {report_path}")
        return report_path
    
    def print_summary_report(self):
        """Print a summary of the validation results"""
        print("\n📈 HISTORICAL DATA VALIDATION REPORT")
        print("="*60)
        
        stats = self.quality_report['overall_quality']['statistics']
        score = self.quality_report['overall_quality']['score']
        
        print(f"Overall Quality Score: {score:.1f}/100")
        print(f"Total Datasets: {stats['total_datasets']}")
        print(f"✅ Excellent: {stats['excellent_datasets']} ({(stats['excellent_datasets']/stats['total_datasets']*100):.1f}%)")
        print(f"✅ Good: {stats['good_datasets']} ({(stats['good_datasets']/stats['total_datasets']*100):.1f}%)")
        print(f"⚠️ Fair: {stats['fair_datasets']} ({(stats['fair_datasets']/stats['total_datasets']*100):.1f}%)")
        print(f"🔴 Poor: {stats['poor_datasets']} ({(stats['poor_datasets']/stats['total_datasets']*100):.1f}%)")
        print(f"🔴 Very Poor: {stats['very_poor_datasets']} ({(stats['very_poor_datasets']/stats['total_datasets']*100):.1f}%)")
        print(f"❌ Missing: {stats['missing_datasets']} ({(stats['missing_datasets']/stats['total_datasets']*100):.1f}%)")
        
        print(f"\n📋 RECOMMENDATIONS:")
        for rec in self.quality_report['recommendations']:
            print(f"  {rec}")
        
        print(f"\n📊 SYMBOLS AVAILABLE:")
        symbols = self.get_available_symbols()
        for symbol in symbols:
            available_timeframes = self.get_available_timeframes(symbol)
            print(f"  {symbol}: {len(available_timeframes)} timeframes")
        
        print("="*60)
    
    def get_data_summary(self, symbol: str) -> Dict:
        """Get comprehensive data summary for a symbol"""
        summary = {
            'symbol': symbol,
            'timeframes': {},
            'total_candles': 0,
            'date_range': {'start': None, 'end': None},
            'file_sizes_mb': 0
        }
        
        for timeframe in self.timeframes:
            df = self.load_data(symbol, timeframe)
            if df is not None and not df.empty:
                summary['timeframes'][timeframe] = {
                    'candles': len(df),
                    'start_date': df.index.min().isoformat(),
                    'end_date': df.index.max().isoformat(),
                    'columns': list(df.columns)
                }
                summary['total_candles'] += len(df)
                
                # Update overall date range
                if summary['date_range']['start'] is None or df.index.min() < pd.to_datetime(summary['date_range']['start']):
                    summary['date_range']['start'] = df.index.min().isoformat()
                if summary['date_range']['end'] is None or df.index.max() > pd.to_datetime(summary['date_range']['end']):
                    summary['date_range']['end'] = df.index.max().isoformat()
        
        # Calculate file sizes
        symbol_dir = self.data_dir / symbol
        if symbol_dir.exists():
            total_size = sum(f.stat().st_size for f in symbol_dir.glob('*.parquet'))
            summary['file_sizes_mb'] = round(total_size / 1024 / 1024, 2)
        
        return summary


def main():
    """Main execution function"""
    print("🔍 HISTORICAL DATA VALIDATOR")
    print("="*50)
    
    validator = HistoricalDataValidator()
    
    # Check if data directory exists
    if not validator.data_dir.exists():
        print(f"❌ Data directory not found: {validator.data_dir}")
        print("Run fetch_20_year_historical_data.py first to download data.")
        return 1
    
    # Validate all data
    print("🔄 Validating all historical data...")
    validation_report = validator.validate_all_data()
    
    # Print summary
    validator.print_summary_report()
    
    # Save report
    report_path = validator.save_validation_report()
    
    print(f"\n✅ Validation completed!")
    print(f"📄 Detailed report saved to: {report_path}")
    
    return 0


if __name__ == "__main__":
    exit(main()) 