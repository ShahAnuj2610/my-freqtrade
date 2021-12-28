import math
import logging
from datetime import datetime

from freqtrade.persistence import Trade
from freqtrade.strategy import IntParameter
from pandas import DataFrame

import talib.abstract as ta

from user_data.strategies.tbedit import tbedit

logger = logging.getLogger(__name__)


class strat_dca(tbedit):
    sell_profit_only = True
    sell_profit_offset = 0.005

    initial_safety_order_trigger = -0.057
    max_safety_orders = 3
    safety_order_step_scale = 2
    safety_order_volume_scale = 2

    buy_params = {
        "dca_min_rsi": 36,
    }

    # append buy_params of parent class
    buy_params.update(tbedit.buy_params)

    dca_min_rsi = IntParameter(35, 75, default=buy_params['dca_min_rsi'], space='buy', optimize=True)

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = super().populate_indicators(dataframe, metadata)
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        return dataframe

    def adjust_trade_position(self, pair: str, trade: Trade,
                              current_time: datetime, current_rate: float, current_profit: float,
                              **kwargs):
        if current_profit > self.initial_safety_order_trigger:
            return None

        # credits to reinuvader for not blindly executing safety orders
        # Obtain pair dataframe.
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        # Only buy when it seems it's climbing back up
        last_candle = dataframe.iloc[-1].squeeze()
        if last_candle['rsi'] < self.dca_min_rsi.value:
            logger.info(f"DCA for {pair} waiting for RSI({last_candle['rsi']}) to rise above {self.dca_min_rsi.value}")
            return None

        count_of_buys = 0
        for order in trade.orders:
            if order.ft_is_open or order.ft_order_side != 'buy':
                continue
            if order.status == "closed":
                count_of_buys += 1

        if 1 <= count_of_buys <= self.max_safety_orders:

            safety_order_trigger = abs(self.initial_safety_order_trigger) + (
                    abs(self.initial_safety_order_trigger) * self.safety_order_step_scale * (
                    math.pow(self.safety_order_step_scale, (count_of_buys - 1)) - 1) / (
                            self.safety_order_step_scale - 1))

            if current_profit <= (-1 * abs(safety_order_trigger)):
                try:
                    stake_amount = self.wallets.get_trade_stake_amount(pair, None)
                    stake_amount = stake_amount * math.pow(self.safety_order_volume_scale, (count_of_buys - 1))
                    amount = stake_amount / current_rate
                    logger.info(
                        f"Initiating safety order buy #{count_of_buys} for {pair} with stake amount of {stake_amount} which equals {amount}")
                    return stake_amount
                except Exception as exception:
                    logger.info(f'Error occured while trying to get stake amount for {pair}: {str(exception)}')
                    return None

        return None

    def confirm_trade_exit(self, pair: str, trade: Trade, order_type: str, amount: float,
                           rate: float, time_in_force: str, sell_reason: str, **kwargs) -> bool:
        # call parent confirm_trade_exit
        if not super().confirm_trade_exit(pair, trade, order_type, amount, rate, time_in_force, sell_reason, **kwargs):
            return False

        # check if profit is positive
        if trade.calc_profit_ratio(rate) > 0:
            return True

        return False
