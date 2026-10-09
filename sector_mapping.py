"""The stock list and its sector / company-name maps: the ONE place to add or remove a stock.

Add a stock: put it in `stock_symbols` with its sector in `symbol_sector` and its name in `symbol_name`, then run
`python run_all.py`. Everything else (scoring, backtest, trading, dashboard, fundamentals, news) reads this file.
A stock with fewer than 200 trading days is listed but not scored or traded until it has them (backtest_engine.MIN_BARS).
"""
import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
REPORTS_DIR = os.path.join(PROJECT_ROOT, "Reports")

# The stocks the strategy ranks and trades (QQQ is the benchmark; it is listed for the dashboard but never traded).
stock_symbols = [
    'AAPL', 'AMZN', 'ANET', 'AMD', 'APP', 'AVAV', 'AVGO', 'BIIB', 'BKR', 'CDNS', 'COIN', 'CRM',
    'CRWV', 'CVLT', 'DASH', 'ENPH', 'FIG', 'FTNT', 'GOOGL', 'GTLB', 'HAL', 'HIMS', 'HOOD', 'INTU',
    'IREN', 'LRCX', 'MCK', 'MRK', 'META', 'MRVL', 'MSFT', 'NFLX', 'NOW', 'NVDA', 'ORCL', 'PANW',
    'RCL', 'REGN', 'SLB', 'SNOW', 'TEAM', 'TSLA', 'UBER', 'UI', 'UNH', 'UPST', 'VEEV', 'VRT',
    'ZS', 'UMAC', 'NOC', 'LMT', 'U', 'CRCL', 'TWLO', 'ACHR', 'ALAB', 'APLD', 'ARM', 'ASTS',
    'CIFR', 'CVNA', 'IONQ', 'JOBY', 'MARA', 'NVTS', 'RDDT', 'RKLB', 'SHOP', 'SMCI', 'SOUN', 'TEM',
    'TTWO', 'SPCX', 'APA', 'OXY', 'TRGP', 'DVN', 'FANG', 'BE', 'BA', 'URI',
    'PH', 'FCX', 'LYB', 'CRDO', 'NBIS', 'LITE', 'CLS', 'RBRK', 'CORZ', 'HIMX', 'LUNR', 'NU',
    'QQQ'
]

# Your manual overrides for the live Mon/Wed/Fri trades and the 9 AM fill check (paper_trade.py), e.g. ['TSLA', 'NVDA'].
# do_not_buy: never bought or topped up (on Friday its slot goes to the next eligible stock, down to rank 20).
# do_not_sell: never sold or trimmed (the Mon/Wed rank-20 rule keeps it too). The earnings-day stop still sells.
do_not_buy = ['SMCI', 'UBER']
do_not_sell = ['SNOW', 'TWLO', 'LITE']

# SPDR sector ETFs: each stock's relative strength is measured against its sector's ETF.
sector_etfs = {
    'XLC': 'Communication Services', 'XLY': 'Consumer Discretionary', 'XLP': 'Consumer Staples', 'XLE': 'Energy',
    'XLF': 'Financials', 'XLV': 'Health Care', 'XLI': 'Industrials', 'XLB': 'Materials',
    'XLRE': 'Real Estate', 'XLK': 'Technology', 'XLU': 'Utilities',
}

# Sector of every stock (also the stocks only used for the fundamentals list below).
symbol_sector = {
    'AAPL': 'Technology', 'ADBE': 'Technology', 'ADSK': 'Technology',
    'AFRM': 'Financials', 'AMD': 'Technology', 'AMZN': 'Consumer Discretionary',
    'ANET': 'Technology', 'APH': 'Technology', 'APP': 'Communication Services',
    'ARM': 'Technology', 'ASML': 'Technology', 'AVAV': 'Industrials',
    'AVGO': 'Technology', 'AXON': 'Industrials', 'BAC': 'Financials',
    'BBAI': 'Technology', 'BIIB': 'Health Care', 'BKR': 'Energy',
    'BLK': 'Financials', 'BTC-USD': 'Commodities', 'CDNS': 'Technology',
    'CDW': 'Technology', 'CELH': 'Consumer Staples', 'COF': 'Financials',
    'COIN': 'Financials', 'CRM': 'Technology', 'CRWD': 'Technology',
    'CRWV': 'Technology', 'CRCL': 'Technology', 'CRSP': 'Health Care',
    'CVLT': 'Technology', 'CVX': 'Energy', 'DASH': 'Consumer Discretionary',
    'DDOG': 'Technology', 'ELV': 'Health Care', 'ENPH': 'Technology',
    'ETH-USD': 'Commodities', 'FANG': 'Energy', 'FCX': 'Materials',
    'FIG': 'Technology', 'FLEX': 'Industrials', 'FTNT': 'Technology',
    'GLW': 'Materials', 'GOOGL': 'Communication Services', 'GTLB': 'Technology',
    'HAL': 'Energy', 'HD': 'Consumer Discretionary', 'HG=F': 'Commodities',
    'HIMS': 'Health Care', 'HOOD': 'Financials', 'HUBS': 'Technology',
    'INTC': 'Technology', 'INTU': 'Technology', 'IONQ': 'Technology',
    'IREN': 'Energy', 'ISRG': 'Health Care', 'IT': 'Technology',
    'JOBY': 'Industrials', 'KLAC': 'Technology', 'KVYO': 'Technology',
    'LIN': 'Materials', 'LLY': 'Health Care', 'LRCX': 'Technology',
    'LMT': 'Industrials', 'MCK': 'Health Care', 'MELI': 'Consumer Discretionary',
    'META': 'Communication Services', 'MNDY': 'Technology', 'MPWR': 'Technology',
    'MRK': 'Health Care', 'MRVL': 'Technology', 'MSFT': 'Technology',
    'MSTR': 'Technology', 'MU': 'Technology', 'NBIS': 'Technology',
    'NEM': 'Materials', 'NET': 'Technology', 'NFLX': 'Communication Services',
    'NKE': 'Consumer Discretionary', 'NOC': 'Industrials', 'NOW': 'Technology',
    'NSC': 'Industrials', 'NUE': 'Materials', 'NVDA': 'Technology',
    'NVO': 'Health Care', 'ORCL': 'Technology', 'PA=F': 'Commodities',
    'PANW': 'Technology', 'PATH': 'Technology', 'PAYX': 'Industrials',
    'PCTY': 'Technology', 'PLTR': 'Technology', 'POOL': 'Consumer Discretionary',
    'PWR': 'Industrials', 'QQQ': 'Technology', 'RCL': 'Consumer Discretionary',
    'REGN': 'Health Care', 'RTX': 'Industrials', 'SHOP': 'Consumer Discretionary',
    'SI=F': 'Commodities', 'SLB': 'Energy', 'SNOW': 'Technology',
    'SOFI': 'Financials', 'SOL-USD': 'Commodities', 'SOUN': 'Technology',
    'SYM': 'Industrials', 'TEAM': 'Technology', 'TEM': 'Health Care',
    'TMO': 'Health Care', 'TMUS': 'Communication Services', 'TSLA': 'Consumer Discretionary',
    'TTD': 'Communication Services', 'TTWO': 'Communication Services', 'TWLO': 'Technology',
    'TXN': 'Technology', 'TYL': 'Technology', 'UBER': 'Industrials',
    'UI': 'Technology', 'UMAC': 'Technology', 'UNH': 'Health Care',
    'UPST': 'Financials', 'VEEV': 'Technology', 'VRT': 'Industrials',
    'WAT': 'Health Care', 'WDAY': 'Technology', 'ZENA': 'Technology',
    'ZS': 'Technology', 'ABNB': 'Consumer Discretionary', 'ALAB': 'Technology',
    'CAVA': 'Consumer Discretionary', 'CPNG': 'Consumer Discretionary', 'DUOL': 'Technology',
    'ELF': 'Consumer Staples', 'MDB': 'Technology', 'NU': 'Financials',
    'TMDX': 'Health Care', 'TOST': 'Technology', 'COST': 'Consumer Staples',
    'JNJ': 'Health Care', 'JPM': 'Financials', 'MA': 'Financials',
    'MCD': 'Consumer Discretionary', 'PEP': 'Consumer Staples', 'PG': 'Consumer Staples',
    'V': 'Financials', 'WM': 'Industrials', 'WMT': 'Consumer Staples',
    'AMT': 'Real Estate', 'NEE': 'Utilities', 'PLD': 'Real Estate',
    'SHW': 'Materials', 'SO': 'Utilities', 'XOM': 'Energy',
    'U': 'Technology', 'CBRE': 'Real Estate', 'CSGP': 'Real Estate',
    'CTSH': 'Technology', 'ZBRA': 'Technology', 'ACHR': 'Industrials',
    'APLD': 'Technology', 'ASTS': 'Communication Services', 'CIFR': 'Technology',
    'CVNA': 'Consumer Discretionary', 'MARA': 'Technology', 'NVTS': 'Technology',
    'RDDT': 'Communication Services', 'RKLB': 'Industrials', 'SMCI': 'Technology',
    'SPCX': 'Technology', 'WBD': 'Communication Services', 'T': 'Communication Services',
    'DIS': 'Communication Services', 'TJX': 'Consumer Discretionary', 'BKNG': 'Consumer Discretionary',
    'COP': 'Energy', 'PSX': 'Energy', 'MPC': 'Energy',
    'VLO': 'Energy', 'BRK.B': 'Financials', 'ABBV': 'Health Care',
    'CAT': 'Industrials', 'GE': 'Industrials', 'KO': 'Consumer Staples',
    'PM': 'Consumer Staples', 'MO': 'Consumer Staples', 'TGT': 'Consumer Staples',
    'MDLZ': 'Consumer Staples', 'CL': 'Consumer Staples', 'ECL': 'Materials',
    'VMC': 'Materials', 'STLD': 'Materials', 'APD': 'Materials',
    'MLM': 'Materials', 'WELL': 'Real Estate', 'EQIX': 'Real Estate',
    'SPG': 'Real Estate', 'PSA': 'Real Estate', 'VTR': 'Real Estate',
    'DLR': 'Real Estate', 'VMRK': 'Real Estate', 'DUK': 'Utilities',
    'CEG': 'Utilities', 'AEP': 'Utilities', 'D': 'Utilities',
    'SRE': 'Utilities', 'ETR': 'Utilities', 'XEL': 'Utilities',
    'VST': 'Utilities', 'CRDO': 'Technology', 'LITE': 'Technology',
    'CLS': 'Technology', 'RBRK': 'Technology', 'APA': 'Energy',
    'OXY': 'Energy', 'TRGP': 'Energy', 'DVN': 'Energy',
    'C': 'Financials', 'APO': 'Financials', 'MS': 'Financials',
    'AXP': 'Financials', 'BE': 'Industrials', 'BA': 'Industrials',
    'URI': 'Industrials', 'PH': 'Industrials', 'TDG': 'Industrials',
    'LYB': 'Materials', 'MOS': 'Materials', 'SW': 'Materials',
    'CORZ': 'Technology', 'HIMX': 'Technology', 'LUNR': 'Industrials',
}

# Company names (dashboard and news search).
symbol_name = {
    'AAPL': 'Apple Inc.', 'AFRM': 'Affirm Holdings, Inc.',
    'QQQ': 'Invesco QQQ Trust Series 1', 'BKR': 'Baker Hughes Company',
    'HG=F': 'Gold', 'SI=F': 'Silver',
    'BTC-USD': 'Bitcoin', 'ETH-USD': 'Ethereum',
    'SOL-USD': 'Solana', 'PA=F': 'Palladium',
    'ADSK': 'Autodesk, Inc.', 'ADBE': 'Adobe Inc.',
    'AMD': 'Advanced Micro Devices, Inc.', 'AMZN': 'Amazon.com, Inc.',
    'ANET': 'Arista Networks, Inc.', 'APH': 'Amphenol Corporation',
    'APP': 'AppLovin Corporation', 'ARM': 'Arm Holdings plc',
    'ASML': 'ASML Holding N.V.', 'AVAV': 'AeroVironment, Inc.',
    'AVGO': 'Broadcom Inc.', 'AXON': 'Axon Enterprise, Inc.',
    'BAC': 'Bank of America Corporation', 'BBAI': 'BigBear',
    'BIIB': 'Biogen Inc.', 'BLK': 'BlackRock, Inc.',
    'CDNS': 'Cadence Design Systems, Inc.', 'CDW': 'CDW Corporation',
    'CELH': 'Celsius Holdings, Inc.', 'COF': 'Capital One Financial Corporation',
    'COIN': 'Coinbase Global, Inc.', 'CRWD': 'CrowdStrike Holdings, Inc.',
    'CRM': 'Salesforce, Inc.', 'CRWV': 'CoreWeave, Inc.',
    'CRCL': 'Circle Internet', 'CRSP': 'CRISPR Therapeutics AG',
    'CVLT': 'Commvault Systems, Inc.', 'CVX': 'Chevron',
    'DASH': 'DoorDash, Inc.', 'DDOG': 'Datadog, Inc.',
    'ELV': 'Elevance Health, Inc.', 'ENPH': 'Enphase Energy, Inc.',
    'FANG': 'Diamondback Energy, Inc.', 'FCX': 'Freeport-McMoRan Inc.',
    'FIG': 'Figma, Inc.', 'FLEX': 'Flex Ltd.',
    'FTNT': 'Fortinet, Inc.', 'GLW': 'Corning Incorporated',
    'GOOGL': 'Alphabet Inc.', 'GTLB': 'GitLab Inc.',
    'HAL': 'Halliburton Company', 'HD': 'The Home Depot, Inc.',
    'HIMS': 'Hims & Hers Health, Inc.', 'HOOD': 'Robinhood Markets, Inc.',
    'HUBS': 'HubSpot, Inc.', 'INTC': 'Intel Corporation',
    'INTU': 'Intuit Inc.', 'IONQ': 'IonQ, Inc.',
    'IREN': 'Iris Energy Limited', 'ISRG': 'Intuitive Surgical, Inc.',
    'IT': 'Gartner, Inc.', 'JOBY': 'Joby Aviation, Inc.',
    'KLAC': 'KLA Corporation', 'KVYO': 'Klaviyo, Inc.',
    'LIN': 'Linde plc', 'LLY': 'Eli Lilly and Company',
    'LMT': 'Lockheed', 'LRCX': 'Lam Research Corporation',
    'MCK': 'McKesson Corporation', 'MELI': 'MercadoLibre, Inc.',
    'MRK': 'Merck & Co., Inc.', 'META': 'Meta Platforms, Inc.',
    'MNDY': 'monday.com Ltd.', 'MPWR': 'Monolithic Power Systems, Inc.',
    'MRVL': 'Marvell Technology, Inc.', 'MSFT': 'Microsoft Corporation',
    'MSTR': 'Strategy, Inc.', 'MU': 'Micron Technology, Inc.',
    'NBIS': 'Nebius Group N.V.', 'NEM': 'Newmont Corporation',
    'NET': 'Cloudflare, Inc.', 'NFLX': 'Netflix, Inc.',
    'NKE': 'Nike, Inc.', 'NVO': 'Novo Nordisk',
    'NOW': 'ServiceNow, Inc.', 'NOC': 'Northrop Grumman Corporation',
    'NSC': 'Norfolk Southern Corporation', 'NUE': 'Nucor Corporation',
    'NVDA': 'NVIDIA Corporation', 'ORCL': 'Oracle Corporation',
    'PANW': 'Palo Alto Networks, Inc.', 'PAYX': 'Paychex, Inc.',
    'PATH': 'UiPath, Inc.', 'PCTY': 'Paylocity Holding Corporation',
    'PLTR': 'Palantir Technologies Inc.', 'POOL': 'Pool Corporation',
    'PWR': 'Quanta Services, Inc.', 'RCL': 'Royal Caribbean Cruises Ltd.',
    'REGN': 'Regeneron Pharmaceuticals, Inc.', 'RTX': 'RTX Corporation',
    'SLB': 'Schlumberger N.V.', 'SHOP': 'Shopify Inc.',
    'SNOW': 'Snowflake Inc.', 'SOFI': 'SoFi Technologies, Inc.',
    'SOUN': 'SoundHound AI, Inc.', 'SYM': 'Symbotic Inc.',
    'TEAM': 'Atlassian Corporation', 'TEM': 'Tempus AI, Inc.',
    'TMO': 'Thermo Fisher Scientific Inc.', 'TMUS': 'T-Mobile US, Inc.',
    'TSLA': 'Tesla, Inc.', 'TTD': 'The Trade Desk, Inc.',
    'TXN': 'Texas Instruments Incorporated', 'TYL': 'Tyler Technologies, Inc.',
    'TWLO': 'Twilio Inc.', 'UBER': 'Uber Technologies, Inc.',
    'UI': 'Ubiquiti Inc.', 'UMAC': 'Unusual Machines, Inc.',
    'UNH': 'UnitedHealth Group Incorporated', 'UPST': 'Upstart Holdings, Inc.',
    'VEEV': 'Veeva Systems Inc.', 'VRT': 'Vertiv Holdings Co',
    'WAT': 'Waters Corporation', 'WDAY': 'Workday, Inc.',
    'ZENA': 'Zenatech, Inc.', 'ZS': 'Zscaler, Inc.',
    'ABNB': 'Airbnb, Inc.', 'ALAB': 'Astera Labs, Inc.',
    'CAVA': 'CAVA Group, Inc.', 'CPNG': 'Coupang, Inc.',
    'DUOL': 'Duolingo, Inc.', 'ELF': 'e.l.f. Beauty, Inc.',
    'MDB': 'MongoDB, Inc.', 'NU': 'Nu Holdings Ltd.',
    'TMDX': 'TransMedics Group, Inc.', 'TOST': 'Toast, Inc.',
    'COST': 'Costco Wholesale Corporation', 'JNJ': 'Johnson & Johnson',
    'JPM': 'JPMorgan Chase & Co.', 'MA': 'Mastercard Incorporated',
    'MCD': "McDonald's Corporation", 'PEP': 'PepsiCo, Inc.',
    'PG': 'Procter & Gamble Company', 'V': 'Visa Inc.',
    'WM': 'Waste Management, Inc.', 'WMT': 'Walmart Inc.',
    'AMT': 'American Tower Corporation', 'NEE': 'NextEra Energy, Inc.',
    'PLD': 'Prologis, Inc.', 'SHW': 'The Sherwin-Williams Company',
    'SO': 'The Southern Company', 'XOM': 'Exxon Mobil Corporation',
    'U': 'Unity Software', 'CBRE': 'CBRE Group, Inc.',
    'CSGP': 'CoStar Group, Inc.', 'CTSH': 'Cognizant Technology Solutions Corporation',
    'ZBRA': 'Zebra Technologies Corporation', 'ACHR': 'Archer Aviation Inc.',
    'APLD': 'Applied Digital Corporation', 'ASTS': 'AST SpaceMobile, Inc.',
    'CIFR': 'Cipher Digital Inc.', 'CVNA': 'Carvana Co.',
    'MARA': 'MARA Holdings, Inc.', 'NVTS': 'Navitas Semiconductor Corporation',
    'RDDT': 'Reddit, Inc.', 'RKLB': 'Rocket Lab Corporation',
    'SMCI': 'Super Micro Computer, Inc.', 'TTWO': 'Take-Two Interactive Software, Inc.',
    'SPCX': 'SpaceX, Inc.', 'WBD': 'Warner Bros. Discovery, Inc.',
    'T': 'AT&T Inc.', 'DIS': 'The Walt Disney Company',
    'TJX': 'The TJX Companies, Inc.', 'BKNG': 'Booking Holdings Inc.',
    'COP': 'ConocoPhillips', 'PSX': 'Phillips 66',
    'MPC': 'Marathon Petroleum Corporation', 'VLO': 'Valero Energy Corporation',
    'BRK.B': 'Berkshire Hathaway Inc. (Class B)', 'ABBV': 'AbbVie Inc.',
    'CAT': 'Caterpillar Inc.', 'GE': 'General Electric (GE Aerospace)',
    'KO': 'The Coca-Cola Company', 'PM': 'Philip Morris International Inc.',
    'MO': 'Altria Group, Inc.', 'TGT': 'Target Corporation',
    'MDLZ': 'Mondelez International, Inc.', 'CL': 'Colgate-Palmolive Company',
    'ECL': 'Ecolab Inc.', 'VMC': 'Vulcan Materials Company',
    'STLD': 'Steel Dynamics, Inc.', 'APD': 'Air Products and Chemicals, Inc.',
    'MLM': 'Martin Marietta Materials, Inc.', 'WELL': 'Welltower Inc.',
    'EQIX': 'Equinix, Inc.', 'SPG': 'Simon Property Group, Inc.',
    'PSA': 'Public Storage', 'VTR': 'Ventas, Inc.',
    'DLR': 'Digital Realty Trust, Inc.', 'VMRK': 'Vivmark Residential',
    'DUK': 'Duke Energy Corporation', 'CEG': 'Constellation Energy Corporation',
    'AEP': 'American Electric Power Company, Inc.', 'D': 'Dominion Energy, Inc.',
    'SRE': 'Sempra', 'ETR': 'Entergy Corporation',
    'XEL': 'Xcel Energy Inc.', 'VST': 'Vistra Corp.',
    'CRDO': 'Credo Technology Group Holding Ltd', 'LITE': 'Lumentum Holdings Inc.',
    'CLS': 'Celestica Inc.', 'RBRK': 'Rubrik, Inc.',
    'APA': 'APA Corporation', 'OXY': 'Occidental Petroleum Corporation',
    'TRGP': 'Targa Resources Corp.', 'DVN': 'Devon Energy Corporation',
    'C': 'Citigroup Inc.', 'APO': 'Apollo Global Management, Inc.',
    'MS': 'Morgan Stanley', 'AXP': 'American Express Company',
    'BE': 'Bloom Energy Corporation', 'BA': 'The Boeing Company',
    'URI': 'United Rentals, Inc.', 'PH': 'Parker-Hannifin Corporation',
    'TDG': 'TransDigm Group Incorporated', 'LYB': 'LyondellBasell Industries N.V.',
    'MOS': 'The Mosaic Company', 'SW': 'Smurfit Westrock plc',
    'CORZ': 'Core Scientific, Inc.', 'HIMX': 'Himax Technologies, Inc.', 'LUNR': 'Intuitive Machines, Inc.',
}

# First usable bar date per symbol (earlier vendor bars belong to a different business or a trading halt).
HISTORY_START = {'NBIS': '2024-10-21', 'CORZ': '2024-01-24'}   # CORZ: relisted after its bankruptcy (flat 0.0751 stub before)

# Benchmarks: never traded, used for the market regime and relative strength.
BENCHMARK_SYMBOLS = ['SPY', 'QQQ']

# Tradable stocks = stock_symbols minus the benchmarks.
tradable_symbols = [s for s in stock_symbols if s not in BENCHMARK_SYMBOLS]

# Fundamentals-only watchlist (company reports, never traded). company_report_autofetch.py fetches the
# tradable stocks plus these.
fundamentals_extra = [
    'ABNB', 'ASML', 'AXON', 'BAC', 'BBAI', 'CDW', 'CELH', 'CRSP', 'CRWD', 'CVX', 'DDOG', 'DUOL',
    'ELF', 'GLW', 'HD', 'HUBS', 'INTC', 'ISRG', 'IT', 'KLAC', 'KVYO', 'LIN', 'LLY', 'MA',
    'MELI', 'MNDY', 'NEE', 'NEM', 'NET', 'NKE', 'NUE', 'NVO', 'PATH', 'PAYX', 'PCTY',
    'PLTR', 'POOL', 'RTX', 'SYM', 'TMO', 'TOST', 'TTD', 'WDAY', 'XOM', 'ZBRA', 'ZENA',
]
fundamentals_symbols = sorted(set(tradable_symbols) | set(fundamentals_extra))

def sector_etf_for(symbol):
    """SPDR sector ETF ticker for a symbol (None when its sector has no ETF)."""
    by_name = {name: etf for etf, name in sector_etfs.items()}
    return by_name.get(symbol_sector.get(symbol))

if __name__ == "__main__":
    for s in stock_symbols:
        print(f"{s:6s} {symbol_sector.get(s, '?'):24s} {symbol_name.get(s, '')}")
