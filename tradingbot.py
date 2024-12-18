import os
from dotenv import load_dotenv
from lumibot.brokers import Alpaca
from lumibot.backtesting import YahooDataBacktesting
from lumibot.strategies.strategy import Strategy
from lumibot.traders import Trader
from datetime import datetime

load_dotenv()

API_KEY = os.getenv('API_KEY')
SECRET_API_KEY = os.getenv('SECRET_API_KEY')
BASE_URL = "https://paper-api.alpaca.markets"

print("API Key:", API_KEY)
print("API Secret:", SECRET_API_KEY)

ALPACA_CREDS = {
    "API_KEY":API_KEY, 
    "API_SECRET": SECRET_API_KEY, 
    "PAPER": True
}

class MLTrader(Strategy):
    """Initializes the object by delegating to the parent class initialization method. 

Provides a thin wrapper around the parent class initialization process, allowing optional parameter configuration.

Args:
    parameters: Optional configuration parameters for initialization.

Returns:
    The result of the parent class initialization method.
"""

    def initialize(self, symbol:str="SPY"):
        self.symbol = symbol
        self.sleeptime = "24"
        self.last_trade = None

    def on_trading_iteration(self):
        if self.last_trade == None:
            order = self.create_order(
                self.symbol,
                10,
                "buy",
                type="market"
            )
            self.submit_order(order)
            self.last_trade = "buy"


start_date = datetime(2020,1,1)
end_date = datetime(2023,12,31) 
broker = Alpaca(ALPACA_CREDS) 

strategy = MLTrader(name='mlstrat', broker=broker, parameters={"symbol":"SPY"})

strategy.backtest(
    YahooDataBacktesting, 
    start_date, 
    end_date, 
    parameters={"symbol":"SPY"}
)


