import os
from dotenv import load_dotenv
from lumibot.brokers import Alpaca
from lumibot.backtesting import YahooDataBacktesting
from lumibot.strategies.strategy import Strategy
from lumibot.traders import Trader
from datetime import datetime
from alpaca_trade_api import REST
from timedelta import Timedelta
from finbert_utils import estimate_sentiment

load_dotenv()

# Leaning more https://forum.alpaca.markets/t/manually-trading-stocks-using-postman-and-the-alpaca-api/166

API_KEY = "PK4ADA600KF3A6UARD49"
API_SECRET = "MsrKOH9TNhezArSCJoKM9ZONwn7GvuUm5g0hpmAw"
# API_KEY = os.getenv('API_KEY')
# SECRET_API_KEY = os.getenv('SECRET_API_KEY')
BASE_URL = "https://paper-api.alpaca.markets"

ALPACA_CREDS = {
    "API_KEY":API_KEY, 
    "API_SECRET": API_SECRET, 
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

    def initialize(self, symbol:str="SPY", cash_at_risk:float=0.5):
        self.symbol = symbol
        self.sleeptime = "24"
        self.last_trade = None
        self.cash_at_risk = cash_at_risk
        self.api = REST(base_url=BASE_URL, key_id=API_KEY, secret_key=API_SECRET)
        
    def position_sizing(self):
        """This formula guides how much of our cash balance we use per trade. cash_at_risk of 0.5 means that for each trade we're using 50% of our remaining cash balance."""
        cash = float(self.get_cash())
        last_price =  self.get_last_price(self.symbol)
        quantity = round(cash * self.cash_at_risk / last_price, 0)
        return cash, last_price, quantity
    
    def get_dates(self):
        today = self.get_datetime()
        three_days_prior = today - Timedelta(days=3)
        return today.strftime("%Y-%M-%d"),  three_days_prior.strftime("%Y-%M-%d"),
    
    def get_sentiment(self):
        today, three_days_prior = self.get_dates()
        news = self.api.get_news(symbol=self.symbol, start=three_days_prior, end=today)
        news = [ev.__dict__["_raw"]["headline"] for ev in news]
        probability, sentiment = estimate_sentiment(news)
        return probability, sentiment

    def on_trading_iteration(self):
        cash, last_price, quantity = self.position_sizing()
        probability, sentiment = self.get_sentiment()
        
        if cash > last_price: 
            if sentiment == 'positive' and probability > .999:
                if self.last_trade == "sell":
                    self.sell_all()
                order = self.create_order(
                    self.symbol,
                    quantity,
                    "buy",
                    type="bracket", 
                    take_profit_price=last_price*1.20, 
                    stop_loss_price=last_price*.95
                )
                self.submit_order(order) 
                self.last_trade = "buy"
            if sentiment == 'negative' and probability > .999:
                if self.last_trade == "buy":
                    self.sell_all()
                order = self.create_order(
                    self.symbol,
                    quantity,
                    "sell",
                    type="bracket", 
                    take_profit_price=last_price*.8, 
                    stop_loss_price=last_price*1.05
                )
                self.submit_order(order) 
                self.last_trade = "sell"


start_date = datetime(2019-12-30)
end_date = datetime(2024-12-1)
broker = Alpaca(ALPACA_CREDS)

strategy = MLTrader(name='mlstrat', broker=broker, parameters={"symbol":"SPY", 
                                                                "cash_at_risk":0.5})

strategy.backtest(
    YahooDataBacktesting,
    start_date,
    end_date,
    parameters={"symbol":"SPY", 
                "cash_at_risk":0.5}
)


# trader = Trader()
# trader.add_strategy(strategy)
# trader.run_all()