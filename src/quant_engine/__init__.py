"""Crypto/derivatives quant engine.

Bootstrap vertical slice: market data (Crypto.com public REST) -> strategy
signals -> funding-aware vectorized backtest -> paper execution behind a
pre-trade guard layer. Not investment advice; no live capital until backtest
and paper-trade both pass.
"""

__version__ = "0.1.0"
