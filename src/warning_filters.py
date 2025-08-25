import warnings
import pandas as pd
 
# Suppress the common SettingWithCopyWarning that floods logs during backtests
warnings.simplefilter("ignore", pd.errors.SettingWithCopyWarning) 