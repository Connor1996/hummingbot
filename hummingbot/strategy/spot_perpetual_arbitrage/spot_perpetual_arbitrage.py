import asyncio
import logging
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Tuple

import pandas as pd

from hummingbot.connector.connector_base import ConnectorBase
from hummingbot.connector.derivative.position import Position
from hummingbot.connector.perpetual_derivative_py_base import PerpetualDerivativePyBase
from hummingbot.core.clock import Clock
from hummingbot.core.data_type.common import OrderType, PositionAction, PositionMode, TradeType
from hummingbot.core.data_type.limit_order import LimitOrder
from hummingbot.core.data_type.market_order import MarketOrder
from hummingbot.core.data_type.order_candidate import OrderCandidate, PerpetualOrderCandidate
from hummingbot.core.event.events import (
    BuyOrderCompletedEvent,
    FundingPaymentCompletedEvent,
    MarketOrderFailureEvent,
    OrderFilledEvent,
    PositionModeChangeEvent,
    SellOrderCompletedEvent,
)
from hummingbot.core.utils.async_utils import safe_ensure_future, safe_gather
from hummingbot.logger import HummingbotLogger
from hummingbot.strategy.market_trading_pair_tuple import MarketTradingPairTuple
from hummingbot.strategy.spot_perpetual_arbitrage.arb_proposal import ArbProposal, ArbProposalSide
from hummingbot.strategy.strategy_py_base import StrategyPyBase

NaN = float("nan")
s_decimal_zero = Decimal(0)
spa_logger = None


class StrategyState(Enum):
    NotReady = 0
    Ready = 1
    Opening = 2
    Closing = 4


class Stats:
    def __init__(self):
        self._fee_paid = Decimal(0)
        self._overall_spread = Decimal(0)
        self._spread_earned = Decimal(0)
        self._funding_earned = Decimal(0)


class SpotPerpetualArbitrageStrategy(StrategyPyBase):
    """
    This strategy arbitrages between a spot and a perpetual exchange.
    For a given order amount, the strategy checks for price discrepancy between buy and sell price on the 2 exchanges.
    Since perpetual contract requires closing position before profit is realised, there are 2 stages to this arbitrage
    operation - first to open and second to close.
    """

    @classmethod
    def logger(cls) -> HummingbotLogger:
        global spa_logger
        if spa_logger is None:
            spa_logger = logging.getLogger(__name__)
        return spa_logger

    def init_params(self,
                    spot_market_info: MarketTradingPairTuple,
                    perp_market_info: MarketTradingPairTuple,
                    total_amount: Decimal,
                    order_amount: Decimal,
                    perp_leverage: int,
                    min_opening_arbitrage_pct: Decimal,
                    min_closing_arbitrage_pct: Decimal,
                    spot_market_slippage_buffer: Decimal = Decimal("0"),
                    perp_market_slippage_buffer: Decimal = Decimal("0"),
                    next_arbitrage_opening_delay: float = 120,
                    status_report_interval: float = 10,
                    near_liquidation_pct: Decimal = Decimal("0.1")):
        """
        :param spot_market_info: The spot market info
        :param perp_market_info: The perpetual market info
        :param total_amount: The total amount of base asset to use for arbitrage
        :param order_amount: The amount of quote asset per order
        :param perp_leverage: The leverage level to use on perpetual market
        :param min_opening_arbitrage_pct: The minimum spread to open arbitrage position (e.g. 0.0003 for 0.3%)
        :param min_closing_arbitrage_pct: The minimum spread to close arbitrage position (e.g. 0.0003 for 0.3%)
        :param spot_market_slippage_buffer: The buffer for which to adjust order price for higher chance of
        the order getting filled on spot market.
        :param perp_market_slippage_buffer: The slipper buffer for perpetual market.
        :param next_arbitrage_opening_delay: The number of seconds to delay before the next arb position can be opened
        :param status_report_interval: Amount of seconds to wait to refresh the status report
        :param near_liquidation_pct: The percentage of liquidation price to consider closing positions
        """
        self._spot_market_info = spot_market_info
        self._perp_market_info = perp_market_info
        self._min_opening_arbitrage_pct = min_opening_arbitrage_pct
        self._min_closing_arbitrage_pct = min_closing_arbitrage_pct
        self._total_amount = total_amount
        self._order_amount = order_amount
        self._perp_leverage = perp_leverage
        self._spot_market_slippage_buffer = spot_market_slippage_buffer
        self._perp_market_slippage_buffer = perp_market_slippage_buffer
        self._next_arbitrage_opening_delay = next_arbitrage_opening_delay
        self._next_arbitrage_opening_ts = 0  # next arbitrage opening timestamp
        self._all_markets_ready = False
        self._ev_loop = asyncio.get_event_loop()
        self._last_timestamp = 0
        self._status_report_interval = status_report_interval
        self._near_liquidation_pct = near_liquidation_pct
        self._stats = Stats()
        self.add_markets([spot_market_info.market, perp_market_info.market])

        self._main_task = None
        self._completed_buy_order_id = 0
        self._completed_sell_order_id = 0
        self._completed_order_fills = []
        self._strategy_state = StrategyState.NotReady
        self._position_action = PositionAction.OPEN
        self._last_arb_op_reported_ts = 0
        self._insufficient_balance = False
        self._position_mode_ready = False
        self._position_mode_not_ready_counter = 0
        self._trading_started = False

    def all_markets_ready(self):
        return all([market.ready for market in self.active_markets])

    @property
    def strategy_state(self) -> StrategyState:
        return self._strategy_state

    @property
    def min_opening_arbitrage_pct(self) -> Decimal:
        return self._min_opening_arbitrage_pct

    @property
    def min_closing_arbitrage_pct(self) -> Decimal:
        return self._min_closing_arbitrage_pct

    @property
    def order_amount(self) -> Decimal:
        return self._order_amount

    @order_amount.setter
    def order_amount(self, value):
        self._order_amount = value

    @property
    def market_info_to_active_orders(self) -> Dict[MarketTradingPairTuple, List[LimitOrder]]:
        return self._sb_order_tracker.market_pair_to_active_orders

    @property
    def perp_positions(self) -> List[Position]:
        return [s for s in self._perp_market_info.market.account_positions.values() if
                s.trading_pair == self._perp_market_info.trading_pair and s.amount != s_decimal_zero]

    def apply_initial_settings(self):
        self._perp_market_info.market.set_leverage(self._perp_market_info.trading_pair, self._perp_leverage)
        self._perp_market_info.market.set_position_mode(PositionMode.ONEWAY)

    def tick(self, timestamp: float):
        """
        Clock tick entry point, is run every second (on normal tick setting).
        :param timestamp: current tick timestamp
        """
        if not self._all_markets_ready or not self._position_mode_ready or not self._trading_started:
            self._all_markets_ready = self.all_markets_ready()
            if not self._all_markets_ready:
                return
            else:
                self.logger().info("Markets are ready.")

            if not self._position_mode_ready:
                self._position_mode_not_ready_counter += 1
                # Attempt to switch position mode every 10 ticks only to not to spam and DDOS
                if self._position_mode_not_ready_counter == 10:
                    self._perp_market_info.market.set_position_mode(PositionMode.ONEWAY)
                    self._position_mode_not_ready_counter = 0
                return
            self._position_mode_not_ready_counter = 0

            self.logger().info("Trading started.")
            self._trading_started = True

            # if not self.check_budget_available():
            #     self.logger().info("Trading not possible.")
            #     return

            if self._perp_market_info.market.position_mode != PositionMode.ONEWAY or \
                    len(self.perp_positions) > 1:
                self.logger().info("This strategy supports only Oneway position mode. Attempting to switch ...")
                self._perp_market_info.market.set_position_mode(PositionMode.ONEWAY)
                return

            spot = abs(self._spot_market_info.base_balance)
            perp = abs(self.perp_positions[0].amount) if len(self.perp_positions) == 1 else s_decimal_zero
            if spot > s_decimal_zero or perp > s_decimal_zero:
                if abs(spot - perp) / max(spot, perp) <= Decimal("0.01"):
                    if self._spot_market_info.base_balance > 0 and self.perp_positions[0].amount > 0 or \
                            self._spot_market_info.base_balance < 0 and self.perp_positions[0].amount < 0:
                        self.logger().info("unmatched position type")
                        return
                    self.logger().info(f"There is an existing {self._perp_market_info.trading_pair} matched "
                                       f"position amount {perp} and balance of amount {spot}.")
                else:
                    self.logger().warning(f"There is an existing {self._perp_market_info.trading_pair} unmatched "
                                          f"position amount {perp} and balance {spot}. "
                                          f"Please manually close out the position before starting this strategy.")
                    return
            self._strategy_state = StrategyState.Ready

        if self._strategy_state != StrategyState.NotReady and (self._main_task is None or self._main_task.done()):
            try:
                self._main_task = safe_ensure_future(self.main(timestamp))
            except Exception as e:
                msg = f"Error during main task: {e}"
                self.logger().error(msg, exc_info=True)
                self.notify_hb_app_with_timestamp(msg)

    async def main(self, timestamp):
        """
        The main procedure for the arbitrage strategy.
        """
        self.update_strategy_state()
        if self._strategy_state in (StrategyState.Opening, StrategyState.Closing):
            self.logger().info("Waiting for orders to complete.")
            return
        if self._next_arbitrage_opening_ts > timestamp:
            return
        proposals = await self.get_proposal_and_update_position_action()
        if len(proposals) == 0:
            return
        proposal = proposals[0]
        if self._last_arb_op_reported_ts + 60 < self.current_timestamp:
            pos_txt = "closing" if self._position_action == PositionAction.CLOSE else "opening"
            self.logger().info(f"Arbitrage position {pos_txt} opportunity found.")
            self.logger().info(f"Profitability ({proposal.profit_pct():.2%}) is now above min_{pos_txt}_arbitrage_pct.")
            self._last_arb_op_reported_ts = self.current_timestamp
        self.apply_slippage_buffers(proposal)
        if self.check_budget_constraint(proposal):
            self._insufficient_balance = False
            self.execute_arb_proposal(proposal)

    def near_liquidation_price(self):
        if len(self.perp_positions) != 0:
            if self.perp_positions[0].liquidation_price is None:
                return None
            return self.perp_positions[0].liquidation_price * (1 - self._near_liquidation_pct)
        return None

    def near_liquidation(self):
        liq_price = self.near_liquidation_price()
        return liq_price is not None and self._perp_market_info.get_mid_price() > liq_price

    def near_liquidation_buffer_price(self):
        if len(self.perp_positions) != 0:
            if self.perp_positions[0].liquidation_price is None:
                return None
            return self.perp_positions[0].liquidation_price * (1 - self._near_liquidation_pct * Decimal(1.5))
        return None

    def near_liquidation_emergent(self):
        liq_price_emergent = self.near_liquidation_emergent_price()
        return liq_price_emergent is not None and self._perp_market_info.get_mid_price() > liq_price_emergent

    def near_liquidation_emergent_price(self):
        if len(self.perp_positions) != 0:
            if self.perp_positions[0].liquidation_price is None:
                return None
            return self.perp_positions[0].liquidation_price * (1 - self._near_liquidation_pct * Decimal(0.5))
        return None

    def near_liquidation_buffer(self):
        liq_buffer_price = self.near_liquidation_buffer_price()
        return liq_buffer_price is not None and self._perp_market_info.get_mid_price() > liq_buffer_price

    async def get_proposal_and_update_position_action(self) -> List[ArbProposal]:
        proposals = await self.create_base_proposals()
        perp_is_buy = False if self.perp_positions[0].amount > 0 else True

        if self.near_liquidation():
            msg = f"Current price {self._perp_market_info.get_mid_price()} is near liquidation price {self.perp_positions[0].liquidation_price}, closing position."
            self.logger().info(msg)
            self.notify_hb_app_with_timestamp(msg)
            self._position_action = PositionAction.CLOSE
            return [p for p in proposals if p.perp_side.is_buy == perp_is_buy and
                    (self.near_liquidation_emergent or p.profit_pct() >= self._min_closing_arbitrage_pct)]

        close_proposals = [p for p in proposals if p.perp_side.is_buy == perp_is_buy and
                           p.profit_pct() >= self._min_closing_arbitrage_pct]
        open_proposals = [p for p in proposals if p.perp_side.is_buy is False and p.profit_pct() >= self._min_opening_arbitrage_pct]

        price = self._spot_market_info.get_mid_price()
        opened = self.total_amount_opened
        # Already opened and min closing arbitrage pct is met
        # Or opened amount is larger than expected
        if (opened * price >= self.order_amount * 2 and len(close_proposals) != 0) or (opened > self._total_amount):
            self._position_action = PositionAction.CLOSE
            return close_proposals
        # TODO: make sure don't back and forth
        # Requested amount is not met yet and not near liquidation
        elif (self._total_amount - opened) * price >= self.order_amount * 2 and not self.near_liquidation_buffer():
            self._position_action = PositionAction.OPEN
            return open_proposals
        else:
            self._position_action = PositionAction.NIL
            return []

    @property
    def total_amount_opened(self):
        spot = self._spot_market_info.base_balance
        perp = self.perp_positions[0].amount if len(self.perp_positions) == 1 else s_decimal_zero
        return abs(spot - perp)

    def update_strategy_state(self):
        """
        Updates strategy state to either Opened or Closed if the condition is right.
        """

        if self._completed_buy_order_id == 0 or self._completed_sell_order_id == 0:
            return
        self.logger().info("Complete one round. buy order id: %s, sell order id: %s", self._completed_buy_order_id,
                           self._completed_sell_order_id)
        self._next_arbitrage_opening_ts = self.current_timestamp + self._next_arbitrage_opening_delay

        buy_volume = Decimal(0)
        sell_volume = Decimal(0)
        for fill in self._completed_order_fills:
            self._stats._fee_paid += fill.trade_fee.fee_amount_in_token(
                trading_pair=fill.trading_pair,
                price=fill.price,
                order_amount=fill.amount,
                token="USDC",
            )
            if fill.order_id == self._completed_buy_order_id:
                buy_volume += fill.amount * fill.price
            elif fill.order_id == self._completed_sell_order_id:
                sell_volume += fill.amount * fill.price
            else:
                self.logger().warning(f"Unknown order id {fill.order_id} in order fills.")

        if self._strategy_state == StrategyState.Opening:
            self._strategy_state = StrategyState.Ready
            # buy spot and sell perp
            self._stats._overall_spread = (buy_volume - sell_volume) / sell_volume
        elif self._strategy_state == StrategyState.Closing:
            self._strategy_state = StrategyState.Ready
            # sell spot and buy perp
            self._next_arbitrage_opening_ts = self.current_timestamp + self._next_arbitrage_opening_delay

        self._completed_buy_order_id = 0
        self._completed_sell_order_id = 0
        self._completed_order_fills.clear()

    async def create_base_proposals(self) -> List[ArbProposal]:
        """
        Creates a list of 2 base proposals, no filter.
        :return: A list of 2 base proposals.
        """
        tasks = [self._spot_market_info.market.get_price_for_quote_volume_async(self._spot_market_info.trading_pair, True,
                                                                                self._order_amount),
                 self._spot_market_info.market.get_price_for_quote_volume_async(self._spot_market_info.trading_pair, False,
                                                                                self._order_amount),
                 self._perp_market_info.market.get_price_for_quote_volume_async(self._perp_market_info.trading_pair, True,
                                                                                self._order_amount),
                 self._perp_market_info.market.get_price_for_quote_volume_async(self._perp_market_info.trading_pair, False,
                                                                                self._order_amount)]
        prices = await safe_gather(*tasks, return_exceptions=True)
        spot_buy, spot_sell, perp_buy, perp_sell = [*prices]

        return [
            ArbProposal(ArbProposalSide(self._spot_market_info, True, spot_buy),
                        ArbProposalSide(self._perp_market_info, False, perp_sell),
                        self._order_amount / min(spot_buy, perp_sell)),
            ArbProposal(ArbProposalSide(self._spot_market_info, False, spot_sell),
                        ArbProposalSide(self._perp_market_info, True, perp_buy),
                        self._order_amount / min(spot_sell, perp_buy))
        ]

    def apply_slippage_buffers(self, proposal: ArbProposal):
        """
        Updates arb_proposals by adjusting order price for slipper buffer percentage.
        E.g. if it is a buy order, for an order price of 100 and 1% slipper buffer, the new order price is 101,
        for a sell order, the new order price is 99.
        :param proposal: the arbitrage proposal
        """
        for arb_side in (proposal.spot_side, proposal.perp_side):
            market = arb_side.market_info.market
            # arb_side.amount = market.quantize_order_amount(arb_side.market_info.trading_pair, arb_side.amount)
            s_buffer = self._spot_market_slippage_buffer if market == self._spot_market_info.market \
                else self._perp_market_slippage_buffer
            if not arb_side.is_buy:
                s_buffer *= Decimal("-1")
            arb_side.order_price *= Decimal("1") + s_buffer
            arb_side.order_price = market.quantize_order_price(arb_side.market_info.trading_pair,
                                                               arb_side.order_price)

    def check_budget_available(self) -> bool:
        """
        Checks if there's any balance for trading to be possible at all
        :return: True if user has available balance enough for orders submission.
        """

        spot_base, spot_quote = self._spot_market_info.trading_pair.split("-")
        perp_base, perp_quote = self._perp_market_info.trading_pair.split("-")

        balance_spot_base = self._spot_market_info.market.get_available_balance(spot_base)
        balance_spot_quote = self._spot_market_info.market.get_available_balance(spot_quote)

        balance_perp_quote = self._perp_market_info.market.get_available_balance(perp_quote)

        if balance_spot_base == s_decimal_zero and balance_spot_quote == s_decimal_zero:
            self.logger().info(f"Cannot arbitrage, {self._spot_market_info.market.display_name} {spot_base} balance "
                               f"({balance_spot_base}) is 0 and {self._spot_market_info.market.display_name} {spot_quote} balance "
                               f"({balance_spot_quote}) is 0.")
            return False

        if balance_perp_quote == s_decimal_zero:
            self.logger().info(f"Cannot arbitrage, {self._perp_market_info.market.display_name} {perp_quote} balance "
                               f"({balance_perp_quote}) is 0.")
            return False

        return True

    def check_budget_constraint(self, proposal: ArbProposal) -> bool:
        """
        Check balances on both exchanges if there is enough to submit both orders in a proposal.
        :param proposal: An arbitrage proposal
        :return: True if user has available balance enough for both orders submission.
        """
        return self.check_spot_budget_constraint(proposal) and self.check_perpetual_budget_constraint(proposal)

    def check_spot_budget_constraint(self, proposal: ArbProposal) -> bool:
        """
        Check balance on spot exchange.
        :param proposal: An arbitrage proposal
        :return: True if user has available balance enough for both orders submission.
        """
        proposal_side = proposal.spot_side
        order_amount = proposal.order_amount
        market_info = proposal_side.market_info
        budget_checker = market_info.market.budget_checker
        order_candidate = OrderCandidate(
            trading_pair=market_info.trading_pair,
            is_maker=False,
            order_type=OrderType.LIMIT,
            order_side=TradeType.BUY if proposal_side.is_buy else TradeType.SELL,
            amount=order_amount,
            price=proposal_side.order_price,
        )

        all_or_none = False if self._position_action == PositionAction.CLOSE else True
        adjusted_candidate_order = budget_checker.adjust_candidate(order_candidate, all_or_none)

        if adjusted_candidate_order.amount < order_amount:
            if self._position_action == PositionAction.CLOSE:
                proposal.order_amount = adjusted_candidate_order.amount
                self.logger().info(f"Adjusting order amount from {order_amount} to {adjusted_candidate_order.amount}")
            else:
                if not self._insufficient_balance:
                    self.logger().info(
                        f"Cannot arbitrage, {proposal_side.market_info.market.display_name} balance"
                        f" is insufficient to place the order candidate {order_candidate}."
                    )
                    self._insufficient_balance = True
                return False

        return True

    def check_perpetual_budget_constraint(self, proposal: ArbProposal) -> bool:
        """
        Check balance on spot exchange.
        :param proposal: An arbitrage proposal
        :return: True if user has available balance enough for both orders submission.
        """
        proposal_side = proposal.perp_side
        order_amount = proposal.order_amount
        market_info = proposal_side.market_info
        budget_checker = market_info.market.budget_checker

        # position_close = False
        # if self.perp_positions and abs(self.perp_positions[0].amount) <= order_amount:
        #     perp_side = proposal.perp_side
        #     cur_perp_pos_is_buy = True if self.perp_positions[0].amount > 0 else False
        #     if perp_side.is_buy != cur_perp_pos_is_buy:
        #         position_close = True
        position_close = self._position_action == PositionAction.CLOSE
        order_candidate = PerpetualOrderCandidate(
            trading_pair=market_info.trading_pair,
            is_maker=False,
            order_type=OrderType.LIMIT,
            order_side=TradeType.BUY if proposal_side.is_buy else TradeType.SELL,
            amount=order_amount,
            price=proposal_side.order_price,
            leverage=Decimal(self._perp_leverage),
            position_close=position_close,
        )

        all_or_none = False if position_close else True
        adjusted_candidate_order = budget_checker.adjust_candidate(order_candidate, all_or_none)

        # TODO: check logic of perptual bus
        # if adjusted_candidate_order.amount < order_amount:
        #     if self._position_action == PositionAction.CLOSE:
        #         proposal.order_amount = adjusted_candidate_order.amount
        #         self.logger().info(f"Adjusting order amount from {order_amount} to {adjusted_candidate_order.amount}")
        #     else:
        #         self.logger().info(
        #             f"Cannot arbitrage, {proposal_side.market_info.market.display_name} balance"
        #             f" is insufficient to place the order candidate {order_candidate}."
        #         )
        #         return False

        if adjusted_candidate_order.amount < order_amount:
            if not self._insufficient_balance:
                self.logger().info(
                    f"Cannot arbitrage, {proposal_side.market_info.market.display_name} balance"
                    f" is insufficient to place the order candidate {order_candidate}."
                    f" Adjusted order amount from {order_amount} to {adjusted_candidate_order.amount}."
                )
                self._insufficient_balance = True
            return False

        return True

    def execute_arb_proposal(self, proposal: ArbProposal):
        """
        Execute both sides of the arbitrage trades concurrently.
        :param proposal: the arbitrage proposal
        """
        if proposal.order_amount == s_decimal_zero:
            return
        spot_side = proposal.spot_side
        spot_order_fn = self.buy_with_specific_market if spot_side.is_buy else self.sell_with_specific_market
        side = "BUY" if spot_side.is_buy else "SELL"
        self.log_with_clock(
            logging.INFO,
            f"Placing {side} order for {proposal.order_amount} {spot_side.market_info.base_asset} "
            f"at {spot_side.market_info.market.display_name} at {spot_side.order_price} price"
        )
        spot_order_fn(
            spot_side.market_info,
            proposal.order_amount,
            spot_side.market_info.market.get_taker_order_type(),
            spot_side.order_price,
        )
        perp_side = proposal.perp_side
        perp_order_fn = self.buy_with_specific_market if perp_side.is_buy else self.sell_with_specific_market
        side = "BUY" if perp_side.is_buy else "SELL"
        self.log_with_clock(
            logging.INFO,
            f"Placing {side} order for {proposal.order_amount} {perp_side.market_info.base_asset} "
            f"at {perp_side.market_info.market.display_name} at {perp_side.order_price} price to "
            f"{self._position_action.name} position."
        )
        perp_order_fn(
            perp_side.market_info,
            proposal.order_amount,
            perp_side.market_info.market.get_taker_order_type(),
            perp_side.order_price,
            position_action=self._position_action
        )
        if self._position_action == PositionAction.OPEN:
            self._strategy_state = StrategyState.Opening
        else:
            self._strategy_state = StrategyState.Closing

    def active_positions_df(self) -> pd.DataFrame:
        """
        Returns a new dataframe on current active perpetual positions.
        """
        columns = ["Symbol", "Type", "Entry Price", "Amount", "Leverage", "Unrealized PnL", "Liquidation"]
        data = []
        for pos in self.perp_positions:
            data.append([
                pos.trading_pair,
                "LONG" if pos.amount > 0 else "SHORT",
                pos.entry_price,
                pos.amount,
                pos.leverage,
                pos.unrealized_pnl,
                pos.liquidation_price
            ])

        return pd.DataFrame(data=data, columns=columns)

    async def format_status(self) -> str:
        """
        Returns a status string formatted to display nicely on terminal. The strings composes of 4 parts: markets,
        assets, spread and warnings(if any).
        """

        columns = ["Exchange", "Market", "Sell Price", "Buy Price", "Mid Price", "Funding"]
        data = []
        for market_info in [self._spot_market_info, self._perp_market_info]:
            market, trading_pair, base_asset, quote_asset = market_info
            buy_price = await market.get_price_for_quote_volume_async(trading_pair, True, self._order_amount)
            sell_price = await market.get_price_for_quote_volume_async(trading_pair, False, self._order_amount)
            if isinstance(market, PerpetualDerivativePyBase):
                rate = f"{(market.get_funding_info(trading_pair).rate * Decimal(365 * 24 * 100)):.2f}%"
            else:
                rate = "N/A"

            mid_price = (buy_price + sell_price) / 2
            data.append([
                market.display_name,
                trading_pair,
                float(sell_price),
                float(buy_price),
                float(mid_price),
                rate
            ])
        markets_df = pd.DataFrame(data=data, columns=columns)
        lines = []
        lines.extend(["", "  Markets:"] + ["    " + line for line in markets_df.to_string(index=False).split("\n")])

        price = self._spot_market_info.get_mid_price()
        base = self._spot_market_info.base_asset
        opened = self.total_amount_opened
        lines.extend(["", "  Info:"])
        lines.extend(["    " + f"Strategy State: {self._strategy_state.name}"])
        lines.extend(["    " + f"Position Action: {self._position_action.name}"])
        lines.extend(["    " + f"Amount: {opened:.2f} {base}({opened * price:.2f}$) / {self._total_amount} {base}({self._total_amount * price:.2f}$)"])
        lines.extend(["    " + f"Fee Paid: {self._stats._fee_paid:.2f}"])
        lines.extend(["    " + f"Overall Opening Spread Rate: {self._stats._overall_spread:.2f}"])
        lines.extend(["    " + f"Spread Earned: {self._stats._spread_earned:.2f}"])
        lines.extend(["    " + f"Funding Earned: {self._stats._funding_earned:.2f}"])
        lines.extend(["    " + f"Near Liquidation: {self.near_liquidation_price()}"])
        lines.extend(["    " + f"Near Liquidation Buffer: {self.near_liquidation_buffer_price()}"])
        lines.extend(["    " + f"Near Liquidation Emergent: {self.near_liquidation_emergent_price()}"])

        # See if there're any active positions.
        if len(self.perp_positions) > 0:
            df = self.active_positions_df()
            lines.extend(["", "  Positions:"] + ["    " + line for line in df.to_string(index=False).split("\n")])
        else:
            lines.extend(["", "  Positions:"] + ["    " + "No active positions."])

        assets_df = self.wallet_balance_data_frame([self._spot_market_info, self._perp_market_info])
        lines.extend(["", "  Assets:"] +
                     ["    " + line for line in str(assets_df).split("\n")])

        proposals = await self.create_base_proposals()
        lines.extend(["", "  Opportunity:"] + self.short_proposal_msg(proposals))

        warning_lines = self.network_warning([self._spot_market_info])
        warning_lines.extend(self.network_warning([self._perp_market_info]))
        warning_lines.extend(self.balance_warning([self._spot_market_info]))
        warning_lines.extend(self.balance_warning([self._perp_market_info]))
        if len(warning_lines) > 0:
            lines.extend(["", "*** WARNINGS ***"] + warning_lines)

        return "\n".join(lines)

    def short_proposal_msg(self, arb_proposal: List[ArbProposal], indented: bool = True) -> List[str]:
        """
        Composes a short proposal message.
        :param arb_proposal: The arbitrage proposal
        :param indented: If the message should be indented (by 4 spaces)
        :return A list of messages
        """
        lines = []
        for proposal in arb_proposal:
            spot_side = "buy" if proposal.spot_side.is_buy else "sell"
            perp_side = "buy" if proposal.perp_side.is_buy else "sell"
            profit_pct = proposal.profit_pct()
            lines.append(f"{'    ' if indented else ''}{spot_side} at "
                         f"{proposal.spot_side.market_info.market.display_name}"
                         f", {perp_side} at {proposal.perp_side.market_info.market.display_name}: "
                         f"{profit_pct:.2%}")
        return lines

    @property
    def tracked_market_orders(self) -> List[Tuple[ConnectorBase, MarketOrder]]:
        return self._sb_order_tracker.tracked_market_orders

    @property
    def tracked_limit_orders(self) -> List[Tuple[ConnectorBase, LimitOrder]]:
        return self._sb_order_tracker.tracked_limit_orders

    def start(self, clock: Clock, timestamp: float):
        self._strategy_state = StrategyState.NotReady
        self.apply_initial_settings()

    def stop(self, clock: Clock):
        if self._main_task is not None:
            self._main_task.cancel()
            self._main_task = None
        self._strategy_state = StrategyState.NotReady

    def did_complete_buy_order(self, event: BuyOrderCompletedEvent):
        self.logger().info(f"Buy order completed. Order ID: {event.order_id}")
        self._completed_buy_order_id = event.order_id

    def did_complete_sell_order(self, event: SellOrderCompletedEvent):
        self.logger().info(f"Sell order completed. Order ID: {event.order_id}")
        self._completed_sell_order_id = event.order_id

    def did_fill_order(self, order_filled_event: OrderFilledEvent):
        self._completed_order_fills.append(order_filled_event)

    def did_fail_order(self, order_failed_event: MarketOrderFailureEvent):
        msg = f"Order failed. Order ID: {order_failed_event.order_id}"
        self.logger().error(msg)
        self.notify_hb_app_with_timestamp(msg)
        self._strategy_state = StrategyState.NotReady
        # TODO: Add logic to handle failed orders

    def did_change_position_mode_succeed(self, position_mode_changed_event: PositionModeChangeEvent):
        if position_mode_changed_event.position_mode is PositionMode.ONEWAY:
            self.logger().info(
                f"Changing position mode to {PositionMode.ONEWAY.name} succeeded.")
            self._position_mode_ready = True
        else:
            self.logger().warning(
                f"Changing position mode to {PositionMode.ONEWAY.name} did not succeed.")
            self._position_mode_ready = False

    def did_change_position_mode_fail(self, position_mode_changed_event: PositionModeChangeEvent):
        self.logger().error(
            f"Changing position mode to {PositionMode.ONEWAY.name} failed. "
            f"Reason: {position_mode_changed_event.message}.")
        self._position_mode_ready = False
        self.logger().warning("Cannot continue. Please resolve the issue in the account.")

    def did_complete_funding_payment(self, funding_payment_completed_event: FundingPaymentCompletedEvent):
        self._stats._funding_earned += funding_payment_completed_event.amount


# TODO:
# - Make sure order is completed before exiting
# - Persist state
# - Calculate spread earned
# - Make amount equal, considering fee
# - Make sure close all position
# - Support negative spread for opening
