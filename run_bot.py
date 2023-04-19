import MetaTrader5 as mt5
from trading_bot import TradingBot

# Connect to MetaTrader 5 terminal
if not mt5.initialize():
    print("initialize() failed")
    mt5.shutdown()

# Set symbol, lot size, stop loss and take profit
symbol = "EURUSD"
lot_size = 0.01
stop_loss = 100  # in pips
take_profit = 100  # in pips

# Initialize trading bot
bot = TradingBot(symbol, lot_size, stop_loss, take_profit)

# Run trading bot
bot.run()

# Disconnect from MetaTrader 5 terminal
mt5.shutdown()
