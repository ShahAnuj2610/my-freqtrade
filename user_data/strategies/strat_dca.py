import math
import logging
from datetime import datetime
from typing import Optional

from freqtrade.persistence import Trade
from freqtrade.strategy import IntParameter
from pandas import DataFrame

import talib.abstract as ta

from user_data.strategies.tbedit import tbedit

logger = logging.getLogger(__name__)


class strat_dca(tbedit):
    position_adjustment_enable = True

    # disable stoploss
    stoploss = -1

    sell_profit_only = True
    sell_profit_offset = 0.005

    initial_safety_order_trigger = -0.057
    max_safety_orders = 4
    safety_order_step_scale = 2
    safety_order_volume_scale = 2

    max_dca_multiplier = (1 + max_safety_orders)
    if max_safety_orders > 0:
        if safety_order_volume_scale > 1:
            max_dca_multiplier = (2 + (safety_order_volume_scale * (
                    math.pow(safety_order_volume_scale, (max_safety_orders - 1)) - 1) / (
                                               safety_order_volume_scale - 1)))
        elif safety_order_volume_scale < 1:
            max_dca_multiplier = (2 + (safety_order_volume_scale * (
                    1 - math.pow(safety_order_volume_scale, (max_safety_orders - 1))) / (
                                               1 - safety_order_volume_scale)))

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

    # Let unlimited stakes leave funds open for DCA orders
    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                            proposed_stake: float, min_stake: float, max_stake: float,
                            **kwargs) -> float:
        if self.config['stake_amount'] == 'unlimited':
            return proposed_stake / self.max_dca_multiplier

        # Use default stake amount.
        return proposed_stake

    def adjust_trade_position(self, trade: Trade, current_time: datetime,
                              current_rate: float, current_profit: float, min_stake: float,
                              max_stake: float, **kwargs) -> Optional[float]:
        pair = trade.pair

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

        filled_buys = trade.select_filled_orders('buy')
        count_of_buys = len(filled_buys)

        if 1 <= count_of_buys <= self.max_safety_orders:
            safety_order_trigger = (abs(self.initial_safety_order_trigger) * count_of_buys)
            if self.safety_order_step_scale > 1:
                safety_order_trigger = abs(self.initial_safety_order_trigger) + (
                        abs(self.initial_safety_order_trigger) * self.safety_order_step_scale * (
                        math.pow(self.safety_order_step_scale, (count_of_buys - 1)) - 1) / (
                                self.safety_order_step_scale - 1))
            elif self.safety_order_step_scale < 1:
                safety_order_trigger = abs(self.initial_safety_order_trigger) + (
                        abs(self.initial_safety_order_trigger) * self.safety_order_step_scale * (
                        1 - math.pow(self.safety_order_step_scale, (count_of_buys - 1))) / (
                                1 - self.safety_order_step_scale))

            if current_profit <= (-1 * abs(safety_order_trigger)):
                try:
                    stake_amount = filled_buys[0].cost
                    # calculate when stake amount will be unlimited
                    if self.config['stake_amount'] == 'unlimited':
                        # This calculates base order size
                        stake_amount = stake_amount / self.max_dca_multiplier
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
        if trade.calc_profit_ratio(rate) > 0.005:
            return True

        return False

    def confirm_trade_entry(self, pair: str, order_type: str, amount: float, rate: float,
                            time_in_force: str, current_time: datetime, entry_tag: Optional[str],
                            **kwargs) -> bool:
        # no more trades
        return False
