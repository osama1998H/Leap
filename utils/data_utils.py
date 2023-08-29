import pandas as pd
import MetaTrader5 as mt5

def load_data(file_path):
    return pd.read_csv(file_path)


def get_historical_data(symbol, timeframe, start, end):
    mt5.initialize()

    timezone = mt5.symbol_info(symbol).timezone
    utc_from = pd.Timestamp(start, tz='UTC').isoformat()
    utc_to = pd.Timestamp(end, tz='UTC').isoformat()

    rates = mt5.copy_rates_range(symbol, timeframe, mt5.datetime.strptime(utc_from, '%Y-%m-%dT%H:%M:%S.%f'), mt5.datetime.strptime(utc_to, '%Y-%m-%dT%H:%M:%S.%f'))

    mt5.shutdown()

    return pd.DataFrame(rates)[['time', 'open', 'high', 'low', 'close', 'tick_volume', 'spread', 'real_volume']]
