"""
draco/live_draco.py — Robust Threaded Interactive Production Bot.
"""
import os
import time
import json
import logging
import requests
import pandas as pd
import numpy as np
import xgboost as xgb
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from threading import Thread
from dotenv import load_dotenv

load_dotenv()
TELE_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# Tiers
TITAN = ["BTC", "ETH", "SOL", "BNB", "XRP", "ADA", "LINK", "DOT", "LTC", "BCH"]
NAV = ["MATIC", "AVAX", "NEAR", "ATOM", "FIL", "ICP", "STX", "RNDR", "FET", "GRT", 
       "INJ", "THETA", "AR", "OP", "ARB", "TIA", "SEI", "SUI", "APT", "VET"]

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-5s | %(message)s")
logger = logging.getLogger("draco_live")

def send_telegram(msg: str):
    if not TELE_TOKEN or not CHAT_ID: return
    url = f"https://api.telegram.org/bot{TELE_TOKEN}/sendMessage"
    try: requests.post(url, json={"chat_id": CHAT_ID, "text": msg, "parse_mode": "Markdown"}, timeout=10)
    except: pass

class DracoLive:
    def __init__(self):
        self.model = xgb.Booster()
        self.model.load_model("models/draco_meta_xgb.json")
        with open('results/draco_tiered_configs.json', 'r') as f:
            self.cfgs = json.load(f)
        
        self.wallet = 1000.0
        self.active_trades = {}
        self.total_trades = 0
        self.last_update_id = 0
        self.running = True
        
        self.assets = TITAN + NAV + ["PEPE", "SHIB", "DOGE", "BONK", "WIF", "FLOKI", "MEME", "ORDI", "1000SATS", "AAVE", "MKR", "LDO", "UNI", "DYDX", "CRV", "IMX", "PYTH", "JUP", "ENA", "W"]
        
        # Start command listener in a thread
        self.listener_thread = Thread(target=self.check_commands, daemon=True)
        self.listener_thread.start()
        
        send_telegram("🐉 *DRACO V11 MULTI-THREADED ACTIVE*\nWallet: $1,000.00 | Listening for commands...")
        logger.info("Bot initialized and listening.")

    def check_commands(self):
        while self.running:
            url = f"https://api.telegram.org/bot{TELE_TOKEN}/getUpdates"
            params = {"timeout": 10, "offset": self.last_update_id + 1}
            try:
                resp = requests.get(url, params=params, timeout=15).json()
                if resp.get("ok"):
                    for update in resp.get("result", []):
                        self.last_update_id = update["update_id"]
                        if "message" in update and "text" in update["message"]:
                            text = update["message"]["text"].lower()
                            logger.info(f"Command received: {text}")
                            if "status" in text:
                                self.send_status()
                            else:
                                send_telegram(f"👋 *Draco is Listening!*\nI received: '{text}'\nTry sending 'status' for a portfolio update.", self.chat_id)
            except Exception as e:
                logger.error(f"Listener Error: {e}")
            time.sleep(1)

    def send_status(self):
        unrealized = 0; pos_str = ""
        for asset, data in self.active_trades.items():
            try:
                url = "https://fapi.binance.com/fapi/v1/ticker/price"
                p = float(requests.get(url, params={"symbol": asset+"USDT"}, timeout=5).json()["price"])
                pnl = (p / data['entry_p'] - 1) * 100
                unrealized += (p - data['entry_p']) * data['qty']
                pos_str += f"• {asset}: {pnl:+.2f}%\n"
            except: pass
        equity = self.wallet + unrealized
        msg = f"📊 *DRACO STATUS*\n\n💰 *Equity:* ${equity:,.2f}\n💵 *Balance:* ${self.wallet:,.2f}\n📈 *Trades:* {self.total_trades}\n\n🏦 *Positions:*\n{pos_str if pos_str else 'None'}"
        send_telegram(msg)

    def fetch_and_scan(self, asset):
        if asset in self.active_trades: return
        url = "https://fapi.binance.com/fapi/v1/klines"
        params = {"symbol": asset + "USDT", "interval": "1h", "limit": 100}
        try:
            r = requests.get(url, params=params, timeout=5).json()
            df = pd.DataFrame(r, columns=['ot','o','h','l','c','v','ct','qv','tr','tb','tq','i'])
            close = df['c'].astype(float); vol = df['v'].astype(float)
            log_c = np.log(close)
            r_val = log_c.rolling(48).corr(pd.Series(np.arange(len(close)))).iloc[-1]
            pvt_r = log_c.rolling(48).corr(np.log(vol.replace(0, 1e-9))).iloc[-1]
            feat = np.array([[abs(r_val), abs(pvt_r), 50.0, 0.0, 0.0]])
            d = xgb.DMatrix(feat); conf = self.model.predict(d)[0]
            tier = "TITAN" if asset in TITAN else ("NAVIGATOR" if asset in NAV else "VOLT")
            cfg = self.cfgs[tier]
            if conf >= cfg['conf_min'] and abs(pvt_r) >= cfg['pvt_r_min']:
                entry_p = close.iloc[-1]
                qty = (self.wallet * 0.12 * 3.0) / entry_p
                self.active_trades[asset] = {'entry_p': entry_p, 'qty': qty, 'tier': tier}
                self.wallet -= (self.wallet * 0.12); self.total_trades += 1
                send_telegram(f"🐉 *BUY:* #{asset}\nPrice: ${entry_p:,.4f}")
        except: pass

    def run_cycle(self):
        logger.info("Starting hourly scan...")
        with ThreadPoolExecutor(max_workers=20) as executor:
            executor.map(self.fetch_and_scan, self.assets)

if __name__ == "__main__":
    bot = DracoLive()
    while True:
        now = datetime.now()
        if now.minute == 0 and now.second < 15:
            bot.run_cycle()
            time.sleep(20)
        time.sleep(1)
