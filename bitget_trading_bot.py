import ccxt
import numpy as np
import time
import json
import logging
from datetime import datetime
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ==============================================
# CONFIGURATION SETTINGS
# ==============================================
CONFIG = {
    "demo_mode": True,
    "api_key": os.getenv('BITGET_DEMO_API_KEY', 'YOUR_DEMO_API_KEY'),
    "secret_key": os.getenv('BITGET_DEMO_SECRET_KEY', 'YOUR_DEMO_SECRET_KEY'),
    "passphrase": os.getenv('BITGET_DEMO_PASSPHRASE', 'YOUR_DEMO_PASSPHRASE'),
    "symbol": "BTCUSDT_UMCBL",
    "timeframe": "1m",
    "initial_balance": 10000,
    "max_positions": 3,
    "trade_size": 0.001,
    "max_daily_trades": 100,
    "strategy_selection": "combined",
    "log_level": logging.INFO,
}

# Set up logging
logging.basicConfig(
    level=CONFIG['log_level'],
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("bitget_trading_bot.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("BitgetTradingBot")

class BitgetFuturesBot:
    def __init__(self):
        # API configuration
        self.api_key = os.getenv("BITGET_API_KEY")
        self.secret_key = os.getenv("BITGET_SECRET_KEY")
        self.passphrase = os.getenv("BITGET_PASSPHRASE")
        self.base_url = "https://api.bitget.com"
        
        # Trading parameters
        self.symbol = "BTCUSDT_UMCBL"
        self.risk_per_trade = 1.0  # $1 per trade
        self.max_loss_streak = 3
        self.loss_streak = 0
        self.cooldown_until = None
        
        # Strategy toggles
        self.strategy_toggles = {
            "liquidity_hunt": True,
            "trend_steal": True,
            "break_retest": True
        }
        
        # Timeframes to monitor
        self.timeframes = ["5m", "15m"]
        
        # Initialize session
        self.session = requests.Session()
        self.session.headers.update({
            "Content-Type": "application/json",
            "locale": "en-US"
        })
        
        logger.info("Bitget Futures Bot initialized")

    def generate_signature(self, method: str, path: str, body: str = "") -> Dict[str, str]:
        """Generate authentication signature for Bitget API"""
        timestamp = str(int(time.time() * 1000))
        message = timestamp + method.upper() + path + body
        signature = hmac.new(
            self.secret_key.encode(), 
            message.encode(), 
            hashlib.sha256
        ).hexdigest()
        
        return {
            "ACCESS-KEY": self.api_key,
            "ACCESS-SIGN": signature,
            "ACCESS-TIMESTAMP": timestamp,
            "ACCESS-PASSPHRASE": self.passphrase
        }

    def make_request(self, method: str, endpoint: str, params: Dict = None) -> Optional[Dict]:
        """Make authenticated request to Bitget API"""
        try:
            # Generate signature
            signature_headers = self.generate_signature(method, endpoint, str(params) if params else "")
            
            # Make request
            if method.upper() == "GET":
                response = self.session.get(
                    self.base_url + endpoint,
                    params=params,
                    headers=signature_headers
                )
            elif method.upper() == "POST":
                response = self.session.post(
                    self.base_url + endpoint,
                    json=params,
                    headers=signature_headers
                )
            else:
                logger.error(f"Unsupported HTTP method: {method}")
                return None
            
            # Parse response
            data = response.json()
            if data.get("code") == "00000":
                return data.get("data")
            else:
                logger.error(f"API error: {data.get('msg')}")
                return None
                
        except Exception as e:
            logger.error(f"Request failed: {str(e)}")
            return None

    def get_account_balance(self) -> Optional[float]:
        """Get available USDT balance"""
        data = self.make_request("GET", "/api/mix/v1/account/accounts")
        if data and len(data) > 0:
            for account in data:
                if account.get("marginCoin") == "USDT":
                    return float(account.get("available"))
        return None

    def get_market_data(self, timeframe: str, limit: int = 100) -> Optional[pd.DataFrame]:
        """Get OHLCV data for the specified timeframe"""
        params = {
            "symbol": self.symbol,
            "granularity": timeframe,  # 5m, 15m, etc.
            "limit": limit
        }
        
        data = self.make_request("GET", "/api/mix/v1/market/candles", params)
        if data:
            # Convert to DataFrame
            df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close", "volume", "quote_volume"])
            df = df.iloc[::-1].reset_index(drop=True)  # Reverse to chronological order
            
            # Convert types
            for col in ["open", "high", "low", "close", "volume", "quote_volume"]:
                df[col] = pd.to_numeric(df[col])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            
            return df
        return None

    def get_orderbook(self) -> Optional[Dict]:
        """Get current orderbook data"""
        params = {"symbol": self.symbol}
        data = self.make_request("GET", "/api/mix/v1/market/depth", params)
        return data

    def calculate_position_size(self, entry_price: float, stop_loss: float) -> float:
        """Calculate position size based on risk per trade"""
        risk_amount = self.risk_per_trade
        price_difference = abs(entry_price - stop_loss)
        
        if price_difference == 0:
            return 0
            
        position_size = risk_amount / price_difference
        return position_size

    def place_order(self, side: str, entry_price: float, stop_loss: float, take_profit: float) -> bool:
        """Place a futures order with stop loss and take profit"""
        # Check if in cooldown period
        if self.cooldown_until and datetime.now() < self.cooldown_until:
            logger.info("In cooldown period, skipping trade")
            return False
            
        # Calculate position size
        position_size = self.calculate_position_size(entry_price, stop_loss)
        if position_size == 0:
            logger.error("Cannot calculate position size (division by zero)")
            return False
            
        # Prepare order parameters
        order_params = {
            "symbol": self.symbol,
            "marginCoin": "USDT",
            "size": str(position_size),
            "price": str(entry_price),
            "side": side,
            "orderType": "limit",
            "timeInForce": "GTC"
        }
        
        # Place order
        order_result = self.make_request("POST", "/api/mix/v1/order/placeOrder", order_params)
        if order_result:
            order_id = order_result.get("orderId")
            logger.info(f"Order placed successfully: {order_id}")
            
            # Set stop loss and take profit
            sl_tp_params = {
                "symbol": self.symbol,
                "orderId": order_id,
                "stopLossPrice": str(stop_loss),
                "takeProfitPrice": str(take_profit)
            }
            sl_tp_result = self.make_request("POST", "/api/mix/v1/order/modifyOrder", sl_tp_params)
            
            if sl_tp_result:
                logger.info("Stop loss and take profit set successfully")
                return True
            else:
                logger.error("Failed to set stop loss and take profit")
                return False
        else:
            logger.error("Failed to place order")
            return False

    def liquidity_hunt_strategy(self, df: pd.DataFrame) -> Optional[Tuple[str, float, float, float]]:
        """Liquidity Hunt strategy implementation"""
        if not self.strategy_toggles["liquidity_hunt"]:
            return None
            
        # Look for liquidity sweeps (price making new highs/lows then reversing)
        latest_price = df["close"].iloc[-1]
        latest_high = df["high"].iloc[-1]
        latest_low = df["low"].iloc[-1]
        
        # Check for bullish liquidity hunt (sweep of lows then reversal)
        if (latest_low < df["low"].iloc[-2] and  # New low
            latest_close > df["open"].iloc[-1] and  # Bullish candle
            latest_close > df["close"].iloc[-2]):   # Close above previous close
            
            entry = latest_close
            stop_loss = latest_low - (latest_low * 0.001)  # Slightly below the low
            take_profit = entry + (entry - stop_loss) * 2  # 1:2 risk-reward ratio
            
            logger.info("Liquidity Hunt: Bullish signal detected")
            return ("buy", entry, stop_loss, take_profit)
        
        # Check for bearish liquidity hunt (sweep of highs then reversal)
        elif (latest_high > df["high"].iloc[-2] and  # New high
              latest_close < df["open"].iloc[-1] and  # Bearish candle
              latest_close < df["close"].iloc[-2]):   # Close below previous close
            
            entry = latest_close
            stop_loss = latest_high + (latest_high * 0.001)  # Slightly above the high
            take_profit = entry - (stop_loss - entry) * 2  # 1:2 risk-reward ratio
            
            logger.info("Liquidity Hunt: Bearish signal detected")
            return ("sell", entry, stop_loss, take_profit)
            
        return None

    def trend_steal_strategy(self, df: pd.DataFrame) -> Optional[Tuple[str, float, float, float]]:
        """Trend Steal strategy implementation"""
        if not self.strategy_toggles["trend_steal"]:
            return None
            
        # Calculate indicators
        df["ema_fast"] = df["close"].ewm(span=12).mean()
        df["ema_slow"] = df["close"].ewm(span=26).mean()
        df["rsi"] = self.calculate_rsi(df["close"], 14)
        
        latest_close = df["close"].iloc[-1]
        prev_close = df["close"].iloc[-2]
        
        # Check for bullish trend (fast EMA above slow EMA)
        if (df["ema_fast"].iloc[-1] > df["ema_slow"].iloc[-1] and
            df["rsi"].iloc[-1] < 40 and  # Oversold condition
            latest_close > prev_close):   # Price starting to move up
            
            entry = latest_close
            stop_loss = df["low"].iloc[-1] - (df["low"].iloc[-1] * 0.002)
            take_profit = entry + (entry - stop_loss) * 1.5  # 1:1.5 risk-reward ratio
            
            logger.info("Trend Steal: Bullish signal detected")
            return ("buy", entry, stop_loss, take_profit)
        
        # Check for bearish trend (fast EMA below slow EMA)
        elif (df["ema_fast"].iloc[-1] < df["ema_slow"].iloc[-1] and
              df["rsi"].iloc[-1] > 60 and  # Overbought condition
              latest_close < prev_close):   # Price starting to move down
            
            entry = latest_close
            stop_loss = df["high"].iloc[-1] + (df["high"].iloc[-1] * 0.002)
            take_profit = entry - (stop_loss - entry) * 1.5  # 1:1.5 risk-reward ratio
            
            logger.info("Trend Steal: Bearish signal detected")
            return ("sell", entry, stop_loss, take_profit)
            
        return None

    def break_retest_strategy(self, df: pd.DataFrame) -> Optional[Tuple[str, float, float, float]]:
        """Break and Retest strategy implementation"""
        if not self.strategy_toggles["break_retest"]:
            return None
            
        # Identify support and resistance levels
        support, resistance = self.identify_support_resistance(df)
        
        latest_close = df["close"].iloc[-1]
        prev_close = df["close"].iloc[-2]
        prev_low = df["low"].iloc[-2]
        prev_high = df["high"].iloc[-2]
        
        # Check for break above resistance and retest
        for level in resistance:
            if (prev_close < level and  # Previously below resistance
                latest_close > level and  # Now above resistance
                df["low"].iloc[-1] <= level and  # Retested the level
                df["close"].iloc[-1] > level):   # Closed above after retest
                
                entry = latest_close
                stop_loss = level - (level * 0.002)
                take_profit = entry + (entry - stop_loss) * 2  # 1:2 risk-reward ratio
                
                logger.info("Break & Retest: Bullish signal detected")
                return ("buy", entry, stop_loss, take_profit)
        
        # Check for break below support and retest
        for level in support:
            if (prev_close > level and  # Previously above support
                latest_close < level and  # Now below support
                df["high"].iloc[-1] >= level and  # Retested the level
                df["close"].iloc[-1] < level):   # Closed below after retest
                
                entry = latest_close
                stop_loss = level + (level * 0.002)
                take_profit = entry - (stop_loss - entry) * 2  # 1:2 risk-reward ratio
                
                logger.info("Break & Retest: Bearish signal detected")
                return ("sell", entry, stop_loss, take_profit)
                
        return None

    def calculate_rsi(self, series: pd.Series, period: int) -> pd.Series:
        """Calculate RSI indicator"""
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def identify_support_resistance(self, df: pd.DataFrame, window: int = 20) -> Tuple[List[float], List[float]]:
        """Identify support and resistance levels"""
        support_levels = []
        resistance_levels = []
        
        # Identify local minima and maxima
        for i in range(window, len(df) - window):
            if df["low"].iloc[i] == df["low"].iloc[i-window:i+window].min():
                support_levels.append(df["low"].iloc[i])
            if df["high"].iloc[i] == df["high"].iloc[i-window:i+window].max():
                resistance_levels.append(df["high"].iloc[i])
        
        # Filter levels that are too close together
        support_levels = self.consolidate_levels(support_levels)
        resistance_levels = self.consolidate_levels(resistance_levels)
        
        return support_levels, resistance_levels

    def consolidate_levels(self, levels: List[float], threshold: float = 0.005) -> List[float]:
        """Consolidate levels that are too close to each other"""
        if not levels:
            return []
            
        consolidated = []
        levels.sort()
        
        current_group = [levels[0]]
        for i in range(1, len(levels)):
            if levels[i] - current_group[0] < current_group[0] * threshold:
                current_group.append(levels[i])
            else:
                consolidated.append(sum(current_group) / len(current_group))
                current_group = [levels[i]]
        
        consolidated.append(sum(current_group) / len(current_group))
        return consolidated

    def update_loss_streak(self, is_winning_trade: bool):
        """Update loss streak and cooldown period"""
        if is_winning_trade:
            self.loss_streak = 0
            if self.cooldown_until:
                self.cooldown_until = None
                logger.info("Win reset loss streak and cooldown")
        else:
            self.loss_streak += 1
            logger.info(f"Loss streak: {self.loss_streak}")
            
            if self.loss_streak >= self.max_loss_streak:
                self.cooldown_until = datetime.now() + timedelta(minutes=3)
                logger.info(f"Max loss streak reached. Cooldown until {self.cooldown_until}")

    def run(self):
        """Main trading loop"""
        logger.info("Starting trading bot...")
        
        while True:
            try:
                # Check if in cooldown period
                if self.cooldown_until and datetime.now() < self.cooldown_until:
                    time.sleep(10)
                    continue
                
                # Get market data for different timeframes
                signals = []
                for timeframe in self.timeframes:
                    df = self.get_market_data(timeframe)
                    if df is None or len(df) < 50:  # Need enough data for indicators
                        continue
                    
                    # Check each strategy
                    if self.strategy_toggles["liquidity_hunt"]:
                        signal = self.liquidity_hunt_strategy(df)
                        if signal:
                            signals.append(("Liquidity Hunt", timeframe) + signal)
                    
                    if self.strategy_toggles["trend_steal"]:
                        signal = self.trend_steal_strategy(df)
                        if signal:
                            signals.append(("Trend Steal", timeframe) + signal)
                    
                    if self.strategy_toggles["break_retest"]:
                        signal = self.break_retest_strategy(df)
                        if signal:
                            signals.append(("Break & Retest", timeframe) + signal)
                
                # Execute the first valid signal
                if signals:
                    strategy_name, timeframe, side, entry, sl, tp = signals[0]
                    
                    logger.info(f"Executing {strategy_name} on {timeframe}: {side} at {entry}, SL: {sl}, TP: {tp}")
                    
                    # Place order
                    success = self.place_order(side, entry, sl, tp)
                    
                    # Simulate trade outcome (in real trading, you'd monitor the order)
                    # For demo purposes, we'll simulate a random outcome
                    import random
                    is_win = random.random() > 0.5
                    
                    # Update loss streak
                    self.update_loss_streak(is_win)
                    
                    # Log trade result
                    result = "WIN" if is_win else "LOSS"
                    logger.info(f"Trade completed: {result}")
                    
                    # Wait before next iteration
                    time.sleep(60)
                else:
                    # No signals found, wait before checking again
                    time.sleep(30)
                    
            except Exception as e:
                logger.error(f"Error in main loop: {str(e)}")
                time.sleep(60)

if __name__ == "__main__":
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    # Initialize and run bot
    bot = BitgetFuturesBot()
    bot.run()

