from decimal import Decimal
from pathlib import Path

from hummingbot.client.settings import DEFAULT_LOG_FILE_PATH
from hummingbot.strategy.market_trading_pair_tuple import MarketTradingPairTuple
from hummingbot.strategy.spot_perpetual_arbitrage.spot_perpetual_arbitrage import SpotPerpetualArbitrageStrategy
from hummingbot.strategy.spot_perpetual_arbitrage.spot_perpetual_arbitrage_config_map import (
    spot_perpetual_arbitrage_config_map,
)


def _safe_history_file_name(strategy_file_name: str) -> str:
    strategy_name = Path(strategy_file_name or "spot_perpetual_arbitrage").stem
    safe_name = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in strategy_name)
    return f"{safe_name}_arbitrage_history.csv"


async def start(self):
    spot_connector = spot_perpetual_arbitrage_config_map.get("spot_connector").value.lower()
    spot_market = spot_perpetual_arbitrage_config_map.get("spot_market").value
    perpetual_connector = spot_perpetual_arbitrage_config_map.get("perpetual_connector").value.lower()
    perpetual_market = spot_perpetual_arbitrage_config_map.get("perpetual_market").value
    total_amount = spot_perpetual_arbitrage_config_map.get("total_amount").value
    extra_spot_base_amount = spot_perpetual_arbitrage_config_map.get("extra_spot_base_amount").value or Decimal("0")
    dryrun = spot_perpetual_arbitrage_config_map.get("dryrun").value or False
    order_amount = spot_perpetual_arbitrage_config_map.get("order_amount").value
    perpetual_leverage = spot_perpetual_arbitrage_config_map.get("perpetual_leverage").value
    min_opening_arbitrage_pct = spot_perpetual_arbitrage_config_map.get("min_opening_arbitrage_pct").value / Decimal("100")
    min_closing_arbitrage_pct = spot_perpetual_arbitrage_config_map.get("min_closing_arbitrage_pct").value / Decimal("100")
    spot_market_slippage_buffer = spot_perpetual_arbitrage_config_map.get("spot_market_slippage_buffer").value / Decimal("100")
    perpetual_market_slippage_buffer = spot_perpetual_arbitrage_config_map.get("perpetual_market_slippage_buffer").value / Decimal("100")
    next_arbitrage_opening_delay = spot_perpetual_arbitrage_config_map.get("next_arbitrage_opening_delay").value
    near_liquidation_pct = spot_perpetual_arbitrage_config_map.get("near_liquidation_pct").value / Decimal("100")

    await self.initialize_markets([(spot_connector, [spot_market]), (perpetual_connector, [perpetual_market])])
    base_1, quote_1 = spot_market.split("-")
    base_2, quote_2 = perpetual_market.split("-")

    spot_market_info = MarketTradingPairTuple(self.markets[spot_connector], spot_market, base_1, quote_1)
    perpetual_market_info = MarketTradingPairTuple(self.markets[perpetual_connector], perpetual_market, base_2, quote_2)

    self.market_trading_pair_tuples = [spot_market_info, perpetual_market_info]
    log_file_path = Path(getattr(self.client_config_map, "log_file_path", DEFAULT_LOG_FILE_PATH) or DEFAULT_LOG_FILE_PATH)
    history_file_path = log_file_path / _safe_history_file_name(getattr(self, "strategy_file_name", None))
    self.strategy = SpotPerpetualArbitrageStrategy()
    self.strategy.init_params(
        spot_market_info=spot_market_info,
        perp_market_info=perpetual_market_info,
        total_amount=total_amount,
        extra_spot_base_amount=extra_spot_base_amount,
        dryrun=dryrun,
        order_amount=order_amount,
        perp_leverage=perpetual_leverage,
        min_opening_arbitrage_pct=min_opening_arbitrage_pct,
        min_closing_arbitrage_pct=min_closing_arbitrage_pct,
        spot_market_slippage_buffer=spot_market_slippage_buffer,
        perp_market_slippage_buffer=perpetual_market_slippage_buffer,
        next_arbitrage_opening_delay=next_arbitrage_opening_delay,
        near_liquidation_pct=near_liquidation_pct,
        history_file_path=str(history_file_path),
    )
