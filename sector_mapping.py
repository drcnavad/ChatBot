import pandas as pd
import os

# Centralized directory paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
REPORTS_DIR = os.path.join(PROJECT_ROOT, "Reports")

## symbols to analyze
stock_symbols = ['AAPL', 'ADBE', 'AFRM', 'AMZN', 'ANET', "AMD", 'APP', 'AVAV',
'AVGO', 'BIIB', 'BKR', 'CDNS', 'COIN', 'CRM',
'CRWV', 'CVLT', 'DASH', 'ENPH', 'FIG',
'FTNT', 'GOOGL', 'GTLB', 'HAL', 'HIMS', 'HOOD', 'INTU',
'IREN', 'LRCX', 'MCK', 'MRK', 'META', 'MRVL',
'MSFT', 'MU', 'NFLX', 'NOW', 'NVDA', 'ORCL', 'PANW',
'RCL', 'REGN',
'SLB', 'SNOW', 'SOFI', 'TEAM', 'TSLA', 'UBER',
'UI', 'UNH', 'UPST', 'VEEV', 'VRT', 'ZS', 'UMAC', 'NOC', 'QQQ',
'MDB', 'LMT', 'U', 'CRCL', 'TWLO',
# High beta (> 2)
'ACHR', 'ALAB', 'APLD', 'ARM', 'ASTS', 'CIFR', 'CVNA', 'IONQ', 'JOBY',
'MARA', 'MSTR', 'NVTS', 'RDDT', 'RKLB', 'SHOP', 'SMCI', 'SOUN', 'TEM']
stock_symbols = list(dict.fromkeys(stock_symbols))  # de-duplicate, keep order
# NOTE: IREN is mapped to 'Energy' below (bitcoin miner / AI data centers) - questionable, kept as-is.

# GICS sector ETFs (SPDR) - canonical sector names for integration with main_signal_analysis
sector_etfs = {
    "XLC": "Communication Services",
    "XLY": "Consumer Discretionary",
    "XLP": "Consumer Staples",
    "XLE": "Energy",
    "XLF": "Financials",
    "XLV": "Health Care",
    "XLI": "Industrials",
    "XLB": "Materials",
    "XLRE": "Real Estate",
    "XLK": "Technology",
    "XLU": "Utilities",
}

# GICS sector mapping (alphabetically by symbol) - must match sector_etfs values
symbol_sector = {
    'AAPL': 'Technology',
    'ADBE': 'Technology',
    'ADSK': 'Technology',
    'AFRM': 'Financials',
    'AMD': 'Technology',
    'AMZN': 'Consumer Discretionary',
    'ANET': 'Technology',
    'APH': 'Technology',
    'APP': 'Communication Services',
    'ARM': 'Technology',
    'ASML': 'Technology',
    'AVAV': 'Industrials',
    'AVGO': 'Technology',
    'AXON': 'Industrials',
    'BAC': 'Financials',
    'BBAI': "Technology",
    'BIIB': 'Health Care',
    'BKR': 'Energy',
    'BLK': 'Financials',
    'BTC-USD': 'Commodities',
    'CDNS': 'Technology',
    'CDW': 'Technology',
    'CELH': 'Consumer Staples',
    'COF': 'Financials',
    'COIN': 'Financials',
    'CRM': 'Technology',
    'CRWD': 'Technology',
    'CRWV': 'Technology',
    'CRCL': 'Technology',
    'CRSP': 'Health Care',
    'CVLT': 'Technology',
    'CVX': 'Energy',
    'DASH': 'Communication Services',
    'DDOG': 'Technology',
    'ELV': 'Health Care',
    'ENPH': 'Technology',
    'ETH-USD': 'Commodities',
    'FANG': 'Energy',
    'FCX': 'Materials',
    'FIG': 'Technology',
    'FLEX': 'Industrials',
    'FTNT': 'Technology',
    'GLW': 'Materials',
    'GOOGL': 'Communication Services',
    'GTLB': 'Technology',
    'HAL': 'Energy',
    'HD': 'Consumer Discretionary',
    'HG=F': 'Commodities',
    'HIMS': 'Health Care',
    'HOOD': 'Financials',
    'HUBS': 'Technology',
    'INTC': 'Technology',
    'INTU': 'Technology',
    'IONQ': 'Technology',
    'IREN': 'Energy',
    'ISRG': 'Health Care',
    'IT': 'Technology',
    'JOBY': 'Industrials',
    'KLAC': 'Technology',
    'KVYO': 'Technology',
    'LIN': 'Materials',
    'LLY': 'Health Care',
    'LRCX': 'Technology',
    'LMT' : 'Industrials',
    'MCK': 'Health Care',
    'MELI': 'Consumer Discretionary',
    'META': 'Communication Services',
    'MNDY': 'Technology',
    'MPWR': 'Technology',
    'MRK': 'Health Care',
    'MRVL': 'Technology',
    'MSFT': 'Technology',
    'MSTR': 'Technology',
    'MU': 'Technology',
    'NBIS': 'Technology',
    'NEM': 'Materials',
    'NET': 'Technology',
    'NFLX': 'Communication Services',
    'NKE': 'Consumer Discretionary',
    'NOC': 'Industrials',
    'NOW': 'Technology',
    'NSC': 'Industrials',
    'NUE': 'Materials',
    'NVDA': 'Technology',
    'NVO': 'Health Care',
    'ORCL': 'Technology',
    'PA=F': 'Commodities',
    'PANW': 'Technology',
    'PATH': 'Technology',
    'PAYX': 'Industrials',
    'PCTY': 'Technology',
    'PLTR': 'Technology',
    'POOL': 'Consumer Discretionary',
    'PWR': 'Industrials',
    'QQQ': 'Technology',
    'RCL': 'Consumer Discretionary',
    'REGN': 'Health Care',
    'RTX': 'Industrials',
    'SHOP': 'Consumer Discretionary',
    'SI=F': 'Commodities',
    'SLB': 'Energy',
    'SNOW': 'Technology',
    'SOFI': 'Financials',
    'SOL-USD': 'Commodities',
    'SOUN': 'Technology',
    'SYM': 'Industrials',
    'TEAM': 'Technology',
    'TEM': 'Health Care',
    'TMO': 'Health Care',
    'TMUS': 'Communication Services',
    'TSLA': 'Consumer Discretionary',
    'TTD': 'Communication Services',
    'TWLO': 'Technology',
    'TXN': 'Technology',
    'TYL': 'Technology',
    'UBER': 'Consumer Discretionary',
    'UI': 'Technology',
    'UMAC': 'Technology',
    'UNH': 'Health Care',
    'UPST': 'Financials',
    'VEEV': 'Technology',
    'VRT': 'Industrials',
    'WAT': 'Health Care',
    'WDAY': 'Technology',
    'ZENA': 'Technology',
    'ZS': 'Technology',
    'ABNB': 'Consumer Discretionary',
    'ALAB': 'Technology',
    'CAVA': 'Consumer Discretionary',
    'CPNG': 'Consumer Discretionary',
    'DUOL': 'Technology',
    'ELF': 'Consumer Staples',
    'MDB': 'Technology',
    'NU': 'Financials',
    'TMDX': 'Health Care',
    'TOST': 'Technology',
    'COST': 'Consumer Staples',
    'JNJ': 'Health Care',
    'JPM': 'Financials',
    'MA': 'Financials',
    'MCD': 'Consumer Discretionary',
    'PEP': 'Consumer Staples',
    'PG': 'Consumer Staples',
    'V': 'Financials',
    'WM': 'Industrials',
    'WMT': 'Consumer Staples',
    'AMT': 'Real Estate',
    'NEE': 'Utilities',
    'PLD': 'Real Estate',
    'SHW': 'Materials',
    'SO': 'Utilities',
    'XOM': 'Energy',
    'U': 'Technology',
    'CBRE': 'Real Estate',
    'CSGP': 'Real Estate',
    'CTSH': 'Technology',
    'ZBRA': 'Technology',
    'ACHR': 'Industrials',
    'APLD': 'Technology',
    'ASTS': 'Communication Services',
    'CIFR': 'Technology',
    'CVNA': 'Consumer Discretionary',
    'MARA': 'Technology',
    'NVTS': 'Technology',
    'RDDT': 'Communication Services',
    'RKLB': 'Industrials',
    'SMCI': 'Technology',
}


symbol_name = {
    'AAPL': 'Apple Inc.',
    'AFRM': 'Affirm Holdings, Inc.',
    'QQQ': 'Invesco QQQ Trust Series 1',
    'BKR': 'Baker Hughes Company',
    'HG=F': 'Gold',
    'SI=F': 'Silver',
    'BTC-USD': 'Bitcoin',
    'ETH-USD': 'Ethereum',
    'SOL-USD': 'Solana',
    'PA=F': 'Palladium',
    'ADSK': 'Autodesk, Inc.',
    'ADBE': 'Adobe Inc.',
    'AMD': 'Advanced Micro Devices, Inc.',
    'AMZN': 'Amazon.com, Inc.',
    'ANET': 'Arista Networks, Inc.',
    'APH': 'Amphenol Corporation',
    'APP': 'AppLovin Corporation',
    'ARM': 'Arm Holdings plc',
    'ASML': 'ASML Holding N.V.',
    'AVAV': 'AeroVironment, Inc.',
    'AVGO': 'Broadcom Inc.',
    'AXON': 'Axon Enterprise, Inc.',
    'BAC': 'Bank of America Corporation',
    'BBAI': 'BigBear',
    'BIIB': 'Biogen Inc.',
    'BLK': 'BlackRock, Inc.',
    'CDNS': 'Cadence Design Systems, Inc.',
    'CDW': 'CDW Corporation',
    'CELH': 'Celsius Holdings, Inc.',
    'COF': 'Capital One Financial Corporation',
    'COIN': 'Coinbase Global, Inc.',
    'CRWD': 'CrowdStrike Holdings, Inc.',
    'CRM': 'Salesforce, Inc.',
    'CRWV': 'CoreWeave, Inc.',
    'CRCL' : 'Circle Internet',
    'CRSP': 'CRISPR Therapeutics AG',
    'CVLT': 'Commvault Systems, Inc.',
    'CVX': 'Chevron',
    'DASH': 'DoorDash, Inc.',
    'DDOG': 'Datadog, Inc.',
    'ELV': 'Elevance Health, Inc.',
    'ENPH': 'Enphase Energy, Inc.',
    'FANG': 'Diamondback Energy, Inc.',
    'FCX': 'Freeport-McMoRan Inc.',
    'FIG': 'Figma, Inc.',
    'FLEX': 'Flex Ltd.',
    'FTNT': 'Fortinet, Inc.',
    'GLW': 'Corning Incorporated',
    'GOOGL': 'Alphabet Inc.',
    'GTLB': 'GitLab Inc.',
    'HAL': 'Halliburton Company',
    'HD': 'The Home Depot, Inc.',
    'HIMS': 'Hims & Hers Health, Inc.',
    'HOOD': 'Robinhood Markets, Inc.',
    'HUBS': 'HubSpot, Inc.',
    'INTC': 'Intel Corporation',
    'INTU': 'Intuit Inc.',
    'IONQ': 'IonQ, Inc.',
    'IREN': 'Iris Energy Limited',
    'ISRG': 'Intuitive Surgical, Inc.',
    'IT': 'Gartner, Inc.',
    'JOBY': 'Joby Aviation, Inc.',
    'KLAC': 'KLA Corporation',
    'KVYO': 'Klaviyo, Inc.',
    'LIN': 'Linde plc',
    'LLY': 'Eli Lilly and Company',
    'LMT': 'Lockheed',
    'LRCX': 'Lam Research Corporation',
    'MCK': 'McKesson Corporation',
    'MELI': 'MercadoLibre, Inc.',
    'MRK': 'Merck & Co., Inc.',
    'META': 'Meta Platforms, Inc.',
    'MNDY': 'monday.com Ltd.',
    'MPWR': 'Monolithic Power Systems, Inc.',
    'MRVL': 'Marvell Technology, Inc.',
    'MSFT': 'Microsoft Corporation',
    'MSTR': 'Strategy, Inc.',
    'MU': 'Micron Technology, Inc.',
    'NBIS': 'Nebius Group',
    'NEM': 'Newmont Corporation',
    'NET': 'Cloudflare, Inc.',
    'NFLX': 'Netflix, Inc.',
    'NKE': 'Nike, Inc.',
    'NVO': 'Novo Nordisk',
    'NOW': 'ServiceNow, Inc.',
    'NOC': 'Northrop Grumman Corporation',
    'NSC': 'Norfolk Southern Corporation',
    'NUE': 'Nucor Corporation',
    'NVDA': 'NVIDIA Corporation',
    'ORCL': 'Oracle Corporation',
    'PANW': 'Palo Alto Networks, Inc.',
    'PAYX': 'Paychex, Inc.',
    'PATH': 'UiPath, Inc.',
    'PCTY': 'Paylocity Holding Corporation',
    'PLTR': 'Palantir Technologies Inc.',
    'POOL': 'Pool Corporation',
    'PWR': 'Quanta Services, Inc.',
    'RCL': 'Royal Caribbean Cruises Ltd.',
    'REGN': 'Regeneron Pharmaceuticals, Inc.',
    'RTX': 'RTX Corporation',
    'SLB': 'Schlumberger N.V.',
    'SHOP': 'Shopify Inc.',
    'SNOW': 'Snowflake Inc.',
    'SOFI': 'SoFi Technologies, Inc.',
    'SOUN': 'SoundHound AI, Inc.',
    'SYM': 'Symbotic Inc.',
    'TEAM': 'Atlassian Corporation',
    'TEM': 'Tempus AI, Inc.',
    'TMO': 'Thermo Fisher Scientific Inc.',
    'TMUS': 'T-Mobile US, Inc.',
    'TSLA': 'Tesla, Inc.',
    'TTD': 'The Trade Desk, Inc.',
    'TXN': 'Texas Instruments Incorporated',
    'TYL': 'Tyler Technologies, Inc.',
    'TWLO': 'Twilio Inc.',
    'UBER': 'Uber Technologies, Inc.',
    'UI': 'Ubiquiti Inc.',
    'UMAC': 'Unusual Machines, Inc.',
    'UNH': 'UnitedHealth Group Incorporated',
    'UPST': 'Upstart Holdings, Inc.',
    'VEEV': 'Veeva Systems Inc.',
    'VRT': 'Vertiv Holdings Co',
    'WAT': 'Waters Corporation',
    'WDAY': 'Workday, Inc.',
    'ZENA': 'Zenatech, Inc.',
    'ZS': 'Zscaler, Inc.',
    'ABNB': 'Airbnb, Inc.',
    'ALAB': 'Astera Labs, Inc.',
    'CAVA': 'CAVA Group, Inc.',
    'CPNG': 'Coupang, Inc.',
    'DUOL': 'Duolingo, Inc.',
    'ELF': 'e.l.f. Beauty, Inc.',
    'MDB': 'MongoDB, Inc.',
    'NU': 'Nu Holdings Ltd.',
    'TMDX': 'TransMedics Group, Inc.',
    'TOST': 'Toast, Inc.',
    'COST': 'Costco Wholesale Corporation',
    'JNJ': 'Johnson & Johnson',
    'JPM': 'JPMorgan Chase & Co.',
    'MA': 'Mastercard Incorporated',
    'MCD': "McDonald's Corporation",
    'PEP': 'PepsiCo, Inc.',
    'PG': 'Procter & Gamble Company',
    'V': 'Visa Inc.',
    'WM': 'Waste Management, Inc.',
    'WMT': 'Walmart Inc.',
    'AMT': 'American Tower Corporation',
    'NEE': 'NextEra Energy, Inc.',
    'PLD': 'Prologis, Inc.',
    'SHW': 'The Sherwin-Williams Company',
    'SO': 'The Southern Company',
    'XOM': 'Exxon Mobil Corporation', 
    'U': 'Unity Software',
    'CBRE': 'CBRE Group, Inc.',
    'CSGP': 'CoStar Group, Inc.',
    'CTSH': 'Cognizant Technology Solutions Corporation',
    'ZBRA': 'Zebra Technologies Corporation',
    'ACHR': 'Archer Aviation Inc.',
    'APLD': 'Applied Digital Corporation',
    'ASTS': 'AST SpaceMobile, Inc.',
    'CIFR': 'Cipher Digital Inc.',
    'CVNA': 'Carvana Co.',
    'MARA': 'MARA Holdings, Inc.',
    'NVTS': 'Navitas Semiconductor Corporation',
    'RDDT': 'Reddit, Inc.',
    'RKLB': 'Rocket Lab Corporation',
    'SMCI': 'Super Micro Computer, Inc.',
}


# --- Universe expansion: tested 2026-09-24; "u96" (= high_beta_91 + 5 emerging tech) LIVE by user decision (see EXPANDED_UNIVERSE) ------------------------------------------------------------------
# Fixed rule (not hand-picked): for every sector with fewer than 10 stocks above, add the largest holdings of that sector's
# SPDR ETF by ETF weight (SSGA daily holdings files, "As of 23-Sep-2026"), skipping names already listed and second share
# classes (GOOG, FOX, NWS); Technology already had 42, so nothing was added there. All added names have Alpaca daily bars
# (CEG from 2022-02, WBD from 2022-04; the 200-bar eligibility rule handles late listings). Survivorship bias remains:
# these are TODAY's largest names. Details and backtest: docs/universe_expansion.md, Reports/universe_expansion_added.csv,
# Reports/universe_expansion_comparison.csv.
# ONE-STEP GO-LIVE: set EXPANDED_UNIVERSE = "fast_sectors_98" (see below), "existing_sectors" (+24 names in the 7 sectors already covered) or "all_sectors"
# (+64: also 10 each for Consumer Staples, Materials, Real Estate, Utilities), then run `python run_all.py`.
EXPANDED_UNIVERSE = "u96"   # LIVE since 2026-09-24 (user decision): tag C6-U96 = "high_beta_91" + 5 emerging-tech names = 96 stocks.
# (With backtest_engine.WINNER["midweek_swap"] on - live since 2026-09-24 - the tag becomes C6-U96-MW.)
#   History of this switch (all 2026-09-24, user decisions):
#     None           -> the original 78 stocks (tag C6)
#     "high_beta_91" -> 78 + 13 names with 2019-2021 beta >= 1.5 (tag C6-U91). It FAILED the pre-declared never-seen 2022-04..2024-09
#                       test (Sharpe 0.72 vs 0.92 for the 78) and was adopted by user choice for its recent strength.
#     "u96"          -> U91 + CRDO NBIS LITE CLS RBRK (Technology; tag C6-U96). HINDSIGHT: picked on 2026-09-24 news after large
#                       run-ups, so any backtest including them is flattered.
#   REVERT: EXPANDED_UNIVERSE = "high_beta_91" (U91) or None (78), then `python run_all.py` (tracking rows keep their Rules label).
#   "fast_sectors_98" (+20 = 98, user revision 2026-09-24): keep the 78, add 5 names to each of the 4 sector ETFs with the highest
#   beta to SPY over 2019-2021 among XLE/XLF/XLI/XLY/XLV/XLB (XLE 1.32, XLF 1.19, XLI 1.10, XLB 1.08; XLY 1.00 and XLV 0.82 not
#   chosen; Communication Services / Staples / Utilities / Real Estate / Technology get nothing). Per sector: top 30 holdings not
#   already listed (no second share classes), full Alpaca history 2019-2021 without a >100% one-day move (spliced history, e.g.
#   EXE), then the 5 highest 2019-2021 betas. See docs/universe_expansion.md and Reports/universe_expansion_u98_added.csv.
EXPANSION_FAST_SECTORS = {
    'Energy': ['APA', 'OXY', 'TRGP', 'DVN', 'FANG'],
    'Financials': ['COF', 'C', 'APO', 'MS', 'AXP'],
    'Industrials': ['BE', 'BA', 'URI', 'PH', 'TDG'],
    'Materials': ['FCX', 'LYB', 'STLD', 'MOS', 'SW'],
}
_FAST_NAMES = {'APA': 'APA Corporation', 'OXY': 'Occidental Petroleum Corporation', 'TRGP': 'Targa Resources Corp.', 'DVN': 'Devon Energy Corporation', 'C': 'Citigroup Inc.', 'APO': 'Apollo Global Management, Inc.', 'MS': 'Morgan Stanley', 'AXP': 'American Express Company', 'BE': 'Bloom Energy Corporation', 'BA': 'The Boeing Company', 'URI': 'United Rentals, Inc.', 'PH': 'Parker-Hannifin Corporation', 'TDG': 'TransDigm Group Incorporated', 'LYB': 'LyondellBasell Industries N.V.', 'MOS': 'The Mosaic Company', 'SW': 'Smurfit Westrock plc'}
EXPANSION_EXISTING_SECTORS = {
    'Communication Services': ['WBD', 'T', 'DIS'],
    'Consumer Discretionary': ['HD', 'MCD', 'TJX', 'BKNG'],
    'Energy': ['XOM', 'CVX', 'COP', 'PSX', 'MPC', 'VLO'],
    'Financials': ['BRK.B', 'JPM', 'V', 'MA', 'BAC'],
    'Health Care': ['LLY', 'JNJ', 'ABBV'],
    'Industrials': ['CAT', 'GE', 'RTX'],
}
EXPANSION_NEW_SECTORS = {
    'Consumer Staples': ['WMT', 'COST', 'PG', 'KO', 'PM', 'MO', 'TGT', 'MDLZ', 'CL', 'PEP'],
    'Materials': ['LIN', 'NEM', 'FCX', 'SHW', 'ECL', 'VMC', 'STLD', 'APD', 'MLM', 'NUE'],
    'Real Estate': ['WELL', 'PLD', 'EQIX', 'AMT', 'SPG', 'PSA', 'VTR', 'CBRE', 'DLR', 'VMRK'],
    'Utilities': ['NEE', 'SO', 'DUK', 'CEG', 'AEP', 'D', 'SRE', 'ETR', 'XEL', 'VST'],
}
_EXPANSION_NAMES = {'WBD': 'Warner Bros. Discovery, Inc.', 'T': 'AT&T Inc.', 'DIS': 'The Walt Disney Company', 'TJX': 'The TJX Companies, Inc.', 'BKNG': 'Booking Holdings Inc.', 'KO': 'The Coca-Cola Company', 'PM': 'Philip Morris International Inc.', 'MO': 'Altria Group, Inc.', 'TGT': 'Target Corporation', 'MDLZ': 'Mondelez International, Inc.', 'CL': 'Colgate-Palmolive Company', 'COP': 'ConocoPhillips', 'PSX': 'Phillips 66', 'MPC': 'Marathon Petroleum Corporation', 'VLO': 'Valero Energy Corporation', 'BRK.B': 'Berkshire Hathaway Inc. (Class B)', 'ABBV': 'AbbVie Inc.', 'CAT': 'Caterpillar Inc.', 'GE': 'General Electric (GE Aerospace)', 'ECL': 'Ecolab Inc.', 'VMC': 'Vulcan Materials Company', 'STLD': 'Steel Dynamics, Inc.', 'APD': 'Air Products and Chemicals, Inc.', 'MLM': 'Martin Marietta Materials, Inc.', 'WELL': 'Welltower Inc.', 'EQIX': 'Equinix, Inc.', 'SPG': 'Simon Property Group, Inc.', 'PSA': 'Public Storage', 'VTR': 'Ventas, Inc.', 'DLR': 'Digital Realty Trust, Inc.', 'VMRK': 'Vivmark Residential', 'DUK': 'Duke Energy Corporation', 'CEG': 'Constellation Energy Corporation', 'AEP': 'American Electric Power Company, Inc.', 'D': 'Dominion Energy, Inc.', 'SRE': 'Sempra', 'ETR': 'Entergy Corporation', 'XEL': 'Xcel Energy Inc.', 'VST': 'Vistra Corp.'}
for _sector, _syms in {**EXPANSION_EXISTING_SECTORS, **EXPANSION_NEW_SECTORS}.items():
    for _s in _syms:
        symbol_sector[_s] = _sector            # mapping only; harmless while EXPANDED_UNIVERSE is None
        symbol_name.setdefault(_s, _EXPANSION_NAMES.get(_s, _s))
#   "high_beta_91" / "high_beta_84" (user revision 2026-09-24): only the U98 additions with 2019-2021 beta >= 1.5 (13 names -> 91)
#   or >= 1.75 (6 names -> 84); betas in Reports/universe_expansion_u98_added.csv.
# "u96" (user decision 2026-09-24): U91 + 5 emerging-tech names, all Technology (XLK). Short histories are handled by the 200-bar
# eligibility rule (CRDO listed 2022-01-27, RBRK IPO 2024-04-25); NBIS bars before 2024-10-21 are Yandex N.V. history incl. a flat,
# zero-volume 2022-24 trading halt, so they are cut by HISTORY_START (applied in backtest_engine.fetch_daily_bars / load_bars).
EXPANSION_EMERGING_TECH = ['CRDO', 'NBIS', 'LITE', 'CLS', 'RBRK']
_EMERGING_NAMES = {'CRDO': 'Credo Technology Group Holding Ltd', 'NBIS': 'Nebius Group N.V.', 'LITE': 'Lumentum Holdings Inc.',
                   'CLS': 'Celestica Inc.', 'RBRK': 'Rubrik, Inc.'}
for _s in EXPANSION_EMERGING_TECH:
    symbol_sector[_s] = 'Technology'
    symbol_name[_s] = _EMERGING_NAMES[_s]
# First usable bar date per symbol (earlier vendor bars belong to a different business / a halt). Applied before any indicator.
HISTORY_START = {'NBIS': '2024-10-21'}
_HIGH_BETA = {
    "high_beta_91": ["APA", "OXY", "BE", "TRGP", "DVN", "BA", "FCX", "FANG", "URI", "COF", "C", "PH", "LYB"],
    "high_beta_84": ["APA", "OXY", "BE", "TRGP", "DVN", "BA"],
}
for _sector, _syms in EXPANSION_FAST_SECTORS.items():
    for _s in _syms:
        symbol_sector[_s] = _sector
        symbol_name.setdefault(_s, _FAST_NAMES.get(_s, _EXPANSION_NAMES.get(_s, _s)))
if EXPANDED_UNIVERSE:
    assert EXPANDED_UNIVERSE in ("existing_sectors", "all_sectors", "fast_sectors_98", "high_beta_91", "high_beta_84", "u96"), EXPANDED_UNIVERSE
    _groups = ([{k: [x for x in v if x in _HIGH_BETA["high_beta_91"]] for k, v in EXPANSION_FAST_SECTORS.items()},
                {'Technology': EXPANSION_EMERGING_TECH}]
               if EXPANDED_UNIVERSE == "u96" else
               [{k: [x for x in v if x in _HIGH_BETA[EXPANDED_UNIVERSE]] for k, v in EXPANSION_FAST_SECTORS.items()}]
               if EXPANDED_UNIVERSE in _HIGH_BETA else
               [EXPANSION_FAST_SECTORS] if EXPANDED_UNIVERSE == "fast_sectors_98" else
               [EXPANSION_EXISTING_SECTORS] + ([EXPANSION_NEW_SECTORS] if EXPANDED_UNIVERSE == "all_sectors" else []))
    _qqq = [s for s in stock_symbols if s == "QQQ"]
    stock_symbols = list(dict.fromkeys([s for s in stock_symbols if s != "QQQ"] + [s for g in _groups for v in g.values() for s in v] + _qqq))


# Benchmarks / ETFs: never traded by the strategies, used for regime + relative strength
BENCHMARK_SYMBOLS = ["SPY", "QQQ"]

# Tradable universe = stock_symbols minus ETFs (QQQ is a benchmark, not a stock)
tradable_symbols = [s for s in dict.fromkeys(stock_symbols) if s not in BENCHMARK_SYMBOLS]

# Extra fundamentals watchlist (formerly hard-coded in company_report_autofetch.py)
fundamentals_watchlist = [
    "AAPL", "ADBE", "AFRM", "AMD", "AMZN", "ANET", "APP", "AVAV", "AVGO", "AXON",
    "BBAI", "BIIB", "BKR", "CDNS", "COIN", "CRM", "CRSP", "CRWD", "CRWV", "CVLT",
    "DASH", "DDOG", "DUOL", "ELF", "ENPH", "FANG", "FIG", "FTNT", "GOOGL", "GTLB",
    "HAL", "HIMS", "HOOD", "HUBS", "INTC", "INTU", "IONQ", "IREN", "ISRG", "KLAC",
    "KVYO", "LIN", "LLY", "LMT", "LRCX", "MA", "MCK", "MDB", "MELI", "META",
    "MNDY", "MRK", "MRVL", "MSFT", "MSTR", "MU", "NEE", "NET", "NFLX", "NOC",
    "NOW", "NU", "NVDA", "ORCL", "PANW", "PLTR", "POOL", "RCL", "REGN", "RTX",
    "SLB", "SNOW", "SOFI", "TEAM", "TEM", "TMO", "TOST", "TSLA", "TTD", "UBER",
    "UI", "UNH", "UPST", "VEEV", "VRT", "WDAY", "XOM", "ZS", "ZENA", "UMAC",
    "SHOP", "NBIS", "NEM", "NKE", "PAYX", "PCTY", "PATH", "ZBRA", "SYM", "ABNB",
    "ARM", "ASML", "BAC", "CDW", "CELH", "CVX", "FCX", "GLW", "HD", "IT",
    "JOBY", "NUE", "NVO",
]


def fundamentals_symbols():
    """Sorted, de-duplicated fundamentals universe: watchlist + every tradable stock."""
    return sorted(set(fundamentals_watchlist) | set(tradable_symbols))


def sector_etf_for(symbol):
    """SPDR sector ETF ticker for a symbol (None when the sector has no ETF, e.g. 'Commodities')."""
    by_name = {name: etf for etf, name in sector_etfs.items()}
    return by_name.get(symbol_sector.get(symbol))


# Create stock_sector_df from stock_symbols list to ensure exact match
stock_sector_df = pd.DataFrame(stock_symbols, columns=['Symbol'])

# Add Sector and Company Name columns
stock_sector_df["Sector"] = stock_sector_df["Symbol"].map(symbol_sector)
stock_sector_df["Company Name"] = stock_sector_df["Symbol"].map(symbol_name)

# sector_map = {
#     "Consumer Cyclical": "Consumer Discretionary",
#     "Consumer Defensive": "Consumer Staples",
#     "Information Technology": "Technology",
#     "Healthcare": "Health Care"
# }

# stock_sector_df["Sector"] = stock_sector_df["Sector"].replace(sector_map)

if __name__ == "__main__":
    print(stock_sector_df)