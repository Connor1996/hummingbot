from decimal import Decimal

CRYPTO_QUOTES = [
    "XBT",
    "ETH",
    "USDT",
    "DAI",
    "USDC",
]

ADDED_CRYPTO_QUOTES = [
    "XXBT",
    "XETH",
]

FIAT_QUOTES = [
    "USD",
    "EUR",
    "CAD",
    "JPY",
    "GBP",
    "CHF",
]

FIAT_QUOTES = FIAT_QUOTES + ["Z" + quote for quote in FIAT_QUOTES]

QUOTES = CRYPTO_QUOTES + ADDED_CRYPTO_QUOTES + FIAT_QUOTES

BASE_ORDER_MIN = {
    "ALGO": Decimal("50"),
    "XREP": Decimal("0.3"),
    "BAT": Decimal("50"),
    "BTC": Decimal("0.002"),
    "BCH": Decimal("0.000002"),
    "ADA": Decimal("1"),
    "LINK": Decimal("10"),
    "ATOM": Decimal("1"),
    "DAI": Decimal("10"),
    "DASH": Decimal("0.03"),
    "XDG": Decimal("3000"),
    "EOS": Decimal("3"),
    "ETH": Decimal("0.02"),
    "ETC": Decimal("0.3"),
    "GNO": Decimal("0.02"),
    "ICX": Decimal("50"),
    "LSK": Decimal("10"),
    "LTC": Decimal("0.1"),
    "XMR": Decimal("0.1"),
    "NANO": Decimal("10"),
    "OMG": Decimal("10"),
    "PAXG": Decimal("0.01"),
    "QTUM": Decimal("0.1"),
    "XRP": Decimal("30"),
    "SC": Decimal("5000"),
    "XLM": Decimal("30"),
    "USDT": Decimal("5"),
    "XTZ": Decimal("1"),
    "USDC": Decimal("5"),
    "MLN": Decimal("0.1"),
    "WAVES": Decimal("10"),
    "ZEC": Decimal("0.03"),
    "TRX": Decimal("500")
}

HB_PAIR_TO_KRAKEN_PAIR = {
    'ADA-XETH': 'ADAETH',
    'ADA-ZEUR': 'ADAEUR',
    'ADA-ZUSD': 'ADAUSD',
    'ADA-XXBT': 'ADAXBT',
    'ALGO-XETH': 'ALGOETH',
    'ALGO-ZEUR': 'ALGOEUR',
    'ALGO-ZUSD': 'ALGOUSD',
    'ALGO-XXBT': 'ALGOXBT',
    'ATOM-XETH': 'ATOMETH',
    'ATOM-ZEUR': 'ATOMEUR',
    'ATOM-ZUSD': 'ATOMUSD',
    'ATOM-XXBT': 'ATOMXBT',
    'BAT-XETH': 'BATETH',
    'BAT-ZEUR': 'BATEUR',
    'BAT-ZUSD': 'BATUSD',
    'BAT-XXBT': 'BATXBT',
    'BCH-ZEUR': 'BCHEUR',
    'BCH-ZUSD': 'BCHUSD',
    'BCH-XXBT': 'BCHXBT',
    'DAI-ZEUR': 'DAIEUR',
    'DAI-ZUSD': 'DAIUSD',
    'DAI-USDT': 'DAIUSDT',
    'DASH-ZEUR': 'DASHEUR',
    'DASH-ZUSD': 'DASHUSD',
    'DASH-XXBT': 'DASHXBT',
    'EOS-XETH': 'EOSETH',
    'EOS-ZEUR': 'EOSEUR',
    'EOS-ZUSD': 'EOSUSD',
    'EOS-XXBT': 'EOSXBT',
    'XETH-CHF': 'ETHCHF',
    'XETH-DAI': 'ETHDAI',
    'XETH-USDC': 'ETHUSDC',
    'XETH-USDT': 'ETHUSDT',
    'GNO-XETH': 'GNOETH',
    'GNO-ZEUR': 'GNOEUR',
    'GNO-ZUSD': 'GNOUSD',
    'GNO-XXBT': 'GNOXBT',
    'ICX-XETH': 'ICXETH',
    'ICX-ZEUR': 'ICXEUR',
    'ICX-ZUSD': 'ICXUSD',
    'ICX-XXBT': 'ICXXBT',
    'LINK-XETH': 'LINKETH',
    'LINK-ZEUR': 'LINKEUR',
    'LINK-ZUSD': 'LINKUSD',
    'LINK-XXBT': 'LINKXBT',
    'LSK-XETH': 'LSKETH',
    'LSK-ZEUR': 'LSKEUR',
    'LSK-ZUSD': 'LSKUSD',
    'LSK-XXBT': 'LSKXBT',
    'NANO-XETH': 'NANOETH',
    'NANO-ZEUR': 'NANOEUR',
    'NANO-ZUSD': 'NANOUSD',
    'NANO-XXBT': 'NANOXBT',
    'OMG-XETH': 'OMGETH',
    'OMG-ZEUR': 'OMGEUR',
    'OMG-ZUSD': 'OMGUSD',
    'OMG-XXBT': 'OMGXBT',
    'PAXG-XETH': 'PAXGETH',
    'PAXG-ZEUR': 'PAXGEUR',
    'PAXG-ZUSD': 'PAXGUSD',
    'PAXG-XXBT': 'PAXGXBT',
    'QTUM-XETH': 'QTUMETH',
    'QTUM-ZEUR': 'QTUMEUR',
    'QTUM-ZUSD': 'QTUMUSD',
    'QTUM-XXBT': 'QTUMXBT',
    'SC-XETH': 'SCETH',
    'SC-ZEUR': 'SCEUR',
    'SC-ZUSD': 'SCUSD',
    'SC-XXBT': 'SCXBT',
    "TRX-XETH": "TRXETH",
    "TRX-ZEUR": "TRXEUR",
    "TRX-ZUSD": "TRXUSD",
    "TRX-XXBT": "TRXXBT",
    'USDC-ZEUR': 'USDCEUR',
    'USDC-ZUSD': 'USDCUSD',
    'USDC-USDT': 'USDCUSDT',
    'USDT-ZCAD': 'USDTCAD',
    'USDT-ZEUR': 'USDTEUR',
    'USDT-ZGBP': 'USDTGBP',
    'USDT-ZUSD': 'USDTZUSD',
    'WAVES-XETH': 'WAVESETH',
    'WAVES-ZEUR': 'WAVESEUR',
    'WAVES-ZUSD': 'WAVESUSD',
    'WAVES-XXBT': 'WAVESXBT',
    'XXBT-CHF': 'XBTCHF',
    'XXBT-DAI': 'XBTDAI',
    'XXBT-USDC': 'XBTUSDC',
    'XXBT-USDT': 'XBTUSDT',
    'XXDG-ZEUR': 'XDGEUR',
    'XXDG-ZUSD': 'XDGUSD',
    'XETC-XETH': 'XETCXETH',
    'XETC-XXBT': 'XETCXXBT',
    'XETC-ZEUR': 'XETCZEUR',
    'XETC-ZUSD': 'XETCZUSD',
    'XETH-XXBT': 'XETHXXBT',
    'XETH-ZCAD': 'XETHZCAD',
    'XETH-ZEUR': 'XETHZEUR',
    'XETH-ZGBP': 'XETHZGBP',
    'XETH-ZJPY': 'XETHZJPY',
    'XETH-ZUSD': 'XETHZUSD',
    'XLTC-XXBT': 'XLTCXXBT',
    'XLTC-ZEUR': 'XLTCZEUR',
    'XLTC-ZUSD': 'XLTCZUSD',
    'XMLN-XETH': 'XMLNXETH',
    'XMLN-XXBT': 'XMLNXXBT',
    'XMLN-ZEUR': 'XMLNZEUR',
    'XMLN-ZUSD': 'XMLNZUSD',
    'XREP-XETH': 'XREPXETH',
    'XREP-XXBT': 'XREPXXBT',
    'XREP-ZEUR': 'XREPZEUR',
    'XREP-ZUSD': 'XREPZUSD',
    'XTZ-XETH': 'XTZETH',
    'XTZ-ZEUR': 'XTZEUR',
    'XTZ-ZUSD': 'XTZUSD',
    'XTZ-XXBT': 'XTZXBT',
    'XXBT-ZCAD': 'XXBTZCAD',
    'XXBT-ZEUR': 'XXBTZEUR',
    'XXBT-ZGBP': 'XXBTZGBP',
    'XXBT-ZJPY': 'XXBTZJPY',
    'XXBT-ZUSD': 'XXBTZUSD',
    'XXDG-XXBT': 'XXDGXXBT',
    'XXLM-XXBT': 'XXLMXXBT',
    'XXLM-ZEUR': 'XXLMZEUR',
    'XXLM-ZUSD': 'XXLMZUSD',
    'XXMR-XXBT': 'XXMRXXBT',
    'XXMR-ZEUR': 'XXMRZEUR',
    'XXMR-ZUSD': 'XXMRZUSD',
    'XXRP-XXBT': 'XXRPXXBT',
    'XXRP-ZCAD': 'XXRPZCAD',
    'XXRP-ZEUR': 'XXRPZEUR',
    'XXRP-ZJPY': 'XXRPZJPY',
    'XXRP-ZUSD': 'XXRPZUSD',
    'XZEC-XXBT': 'XZECXXBT',
    'XZEC-ZEUR': 'XZECZEUR',
    'XZEC-ZUSD': 'XZECZUSD',
    'ZEUR-ZCAD': 'EURCAD',
    'ZEUR-CHF': 'EURCHF',
    'ZEUR-ZGBP': 'EURGBP',
    'ZEUR-ZJPY': 'EURJPY',
    'ZEUR-ZUSD': 'ZEURZUSD',
    'ZGBP-ZUSD': 'ZGBPZUSD',
    'ZUSD-ZCAD': 'ZUSDZCAD',
    'ZUSD-CHF': 'USDCHF',
    'ZUSD-ZJPY': 'ZUSDZJPY',
}
