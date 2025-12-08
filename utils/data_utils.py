# UNUSED - This entire module is not imported anywhere in the codebase
# Also contains bugs: mt5.datetime doesn't exist, symbol_info.timezone doesn't exist

import pandas as pd
import MetaTrader5 as mt5


# UNUSED - Function never called anywhere in codebase
def load_data(file_path):
    data = pd.read_csv(file_path)
    return data


# UNUSED - Function never called anywhere in codebase
def get_historical_data(symbol, timeframe, start, end):
    mt5.initialize()

    timezone = mt5.symbol_info(symbol).timezone
    utc_from = pd.Timestamp(start, tz='UTC').isoformat()
    utc_to = pd.Timestamp(end, tz='UTC').isoformat()

    rates = mt5.copy_rates_range(symbol, timeframe, mt5.datetime.strptime(utc_from, '%Y-%m-%dT%H:%M:%S.%f'), mt5.datetime.strptime(utc_to, '%Y-%m-%dT%H:%M:%S.%f'))

    mt5.shutdown()

    return pd.DataFrame(rates)[['time', 'open', 'high', 'low', 'close', 'tick_volume', 'spread', 'real_volume']]
