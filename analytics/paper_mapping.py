import uuid
from typing import Optional

from .signal_logger import SignalLogger


class PaperTradeMapper:
    """
    Minimal safe mapping for current PaperBroker:
    - There is no explicit trade_id in broker state.
    - We generate local_trade_id on OPEN and reuse it on CLOSE.
    """

    def __init__(self, logger: SignalLogger):
        self.logger = logger

    def map_open(
        self,
        signal_id: Optional[str],
        ticker: str,
        qty: float,
        price: float,
        local_trade_id: Optional[str] = None,
        comment: str = "paper_open",
    ) -> str:
        trade_id = local_trade_id or str(uuid.uuid4())
        self.logger.link_paper_trade(
            signal_id=signal_id,
            local_trade_id=trade_id,
            ticker=ticker,
            side="OPEN",
            qty=qty,
            price=price,
            paper_position_id=ticker,
            comment=comment,
        )
        return trade_id

    def map_close(
        self,
        signal_id: Optional[str],
        local_trade_id: Optional[str],
        ticker: str,
        qty: float,
        price: float,
        comment: str = "paper_close",
    ) -> str:
        return self.logger.link_paper_trade(
            signal_id=signal_id,
            local_trade_id=local_trade_id,
            ticker=ticker,
            side="CLOSE",
            qty=qty,
            price=price,
            paper_position_id=ticker,
            comment=comment,
        )

