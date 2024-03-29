import json
import logging
import threading
from datetime import datetime

import requests
from freqtrade.persistence import Trade
from freqtrade.strategy.interface import IStrategy

logger = logging.getLogger(__name__)

"""
Usage:
StrategyAnalysis(YourStrategyClass)

Note:
If you're using a hyperopt json file, change the name of the class to "StrategyAnalysis" in the file.
"""


def colored(param, color):
    color_codes = {
        'red': '\033[91m',
        'green': '\033[92m',
    }
    return color_codes[color] + param + '\033[0m'


class StrategyAnalysis(IStrategy):
    def confirm_trade_exit(self, pair: str, trade: Trade, order_type: str, amount: float,
                           rate: float, time_in_force: str, exit_reason: str,
                           current_time: datetime, **kwargs) -> bool:
        trade_exit_parent = super().confirm_trade_exit(pair, trade, order_type, amount, rate, time_in_force,
                                                       exit_reason, current_time, **kwargs)
        if self.config['runmode'].value in ['backtest']:
            # log a table for the pair with following columns:
            # - pair
            # - min_profit percentage
            # - max_profit percentage
            # - duration (in minutes)
            # - profit (in percentage)
            # - stoploss (in percentage)
            min_profit_rate = trade.max_rate if trade.is_short else trade.min_rate
            max_profit_rate = trade.min_rate if trade.is_short else trade.max_rate

            min_profit = (trade.calc_profit_ratio(min_profit_rate) * 100)
            max_profit = (trade.calc_profit_ratio(max_profit_rate) * 100)
            duration = (trade.close_date - trade.open_date).total_seconds() / 60
            profit = (trade.calc_profit_ratio(rate) * 100)
            # log the table with columns above
            # if profit is negative, then log in red color
            if profit < 0:
                self.log(
                    f'{pair} min_profit: {min_profit}%, max_profit: {max_profit}%, duration: {duration}min, profit: {profit}%',
                    color='red')
            else:
                self.log(
                    f'{pair} min_profit: {min_profit}%, max_profit: {max_profit}%, duration: {duration}min, profit: {profit}%',
                    color='green')

        if self.config['telegram']['enabled'] == True:
            min_profit_rate = trade.max_rate if trade.is_short else trade.min_rate
            max_profit_rate = trade.min_rate if trade.is_short else trade.max_rate

            min_profit = (trade.calc_profit_ratio(min_profit_rate) * 100)
            max_profit = (trade.calc_profit_ratio(max_profit_rate) * 100)

            start_time = datetime.now()
            self.telegram_send(
                f"↕️ {pair} Min profit:  {min_profit:.2f}%  Max profit:  {max_profit:.2f}%\n"
                f"💰 {pair} Open fee:  {trade.fee_open * 100:.4f}%  Close fee:  {trade.fee_close * 100:.4f}%"
            )
            # logger.info(f"{pair} took {datetime.now() - start_time} to send telegram message")

        if 'discord' in self.config and self.config['discord']['enabled'] == True:
            self.discord_send(trade)

        return trade_exit_parent

    def telegram_send(self, message):
        if self.config['runmode'].value in ('dry_run', 'live') and self.config['telegram']['enabled'] == True:
            bot_token = self.config['telegram']['token']
            bot_chatID = self.config['telegram']['chat_id']
            send_text = 'https://api.telegram.org/bot' + bot_token + '/sendMessage?chat_id=' + bot_chatID + '&parse_mode=Markdown&text=' + message

            threading.Thread(target=requests.get, args=(send_text,)).start()

    def log(self, param, color):
        logger.info(colored(param, color))

    def discord_send(self, trade: Trade):
        if self.config['runmode'].value in ('dry_run', 'live') and self.config['discord']['enabled'] == True:
            webhook_url = self.config['discord']['webhook_url']
            # TODO: fix profit and sell_reason
            profit = trade.calc_profit_ratio()
            gain = "profit" if profit > 0 else "loss"
            embeds = [
                {
                    "title": f"{trade.pair} {gain}",
                    "description": f"{trade.pair} {gain} {profit * 100:.2f}%",
                    "color": 0x00ff00 if gain == "profit" else 0xff0000,
                    "fields": [
                        {
                            "name": "Open",
                            "value": f"{trade.open_rate:.8f}",
                            "inline": True
                        },
                        {
                            "name": "Profit",
                            "value": f"{profit * 100:.2f}%",
                            "inline": True
                        },
                        {
                            "name": "Buy Tag",
                            "value": f"{trade.buy_tag}",
                            "inline": True
                        },
                        {
                            "name": "Sell Tag",
                            "value": f"{trade.sell_reason}",
                            "inline": True
                        }
                    ]
                }
            ]

            payload = {
                "embeds": embeds
            }

            threading.Thread(target=requests.post, args=(webhook_url,), kwargs={"data": json.dumps(payload),
                                                                                "headers": {
                                                                                    "Content-Type": "application/json"
                                                                                }}).start()
