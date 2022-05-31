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
            profit_rate = trade.close_rate if trade.close_rate else trade.close_rate_requested
            profit_trade = trade.calc_profit(rate=profit_rate)
            profit_ratio = trade.calc_profit_ratio(profit_rate)
            gain = "profit" if profit_ratio > 0 else "loss"

            msg = {
                'trade_id': trade.id,
                'exchange': trade.exchange.capitalize(),
                'pair': trade.pair,
                'gain': gain,
                'limit': profit_rate,
                'order_type': order_type,
                'amount': trade.amount,
                'open_rate': trade.open_rate,
                'close_rate': trade.close_rate,
                'profit_amount': profit_trade,
                'profit_ratio': profit_ratio,
                'buy_tag': trade.buy_tag,
                'sell_reason': trade.sell_reason,
                'open_date': trade.open_date,
                'close_date': trade.close_date or datetime.utcnow(),
                'stake_currency': self.config['stake_currency'],
                'fiat_currency': self.config.get('fiat_display_currency', None),
            }
            self.discord_send(msg)

        return trade_exit_parent

    def telegram_send(self, message):
        if self.config['runmode'].value in ('dry_run', 'live') and self.config['telegram']['enabled'] == True:
            bot_token = self.config['telegram']['token']
            bot_chatID = self.config['telegram']['chat_id']
            send_text = 'https://api.telegram.org/bot' + bot_token + '/sendMessage?chat_id=' + bot_chatID + '&parse_mode=Markdown&text=' + message

            threading.Thread(target=requests.get, args=(send_text,)).start()

    def discord_send(self, message):
        if self.config['runmode'].value in ('dry_run', 'live') and 'discord' in self.config and \
                self.config['discord']['enabled'] == True:
            bot_token = self.config['discord']['token']
            bot_chatID = self.config['discord']['chat_id']
            send_text = 'https://discordapp.com/api/webhooks/' + bot_chatID + '/' + bot_token + '?content=' + message

            threading.Thread(target=requests.get, args=(send_text,)).start()

    def log(self, param, color):
        logger.info(colored(param, color))
