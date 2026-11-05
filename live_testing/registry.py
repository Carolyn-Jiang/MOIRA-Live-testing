from zoneinfo import ZoneInfo

NY = ZoneInfo('America/New_York')
TICKERS = ('NVDA','GOOG','META','MSFT','JPM','AMZN','AAPL','AMD','WMT','TSLA')
MODEL_IDS = ('codex','finmem','moira','moira2-top1','moira2-top3','moira2-top5')
CODEX_BACKBONE = 'gpt-6.1-sol'
BACKBONES = {name: CODEX_BACKBONE if name == 'codex' else 'gpt-4.1-mini' for name in MODEL_IDS}
# November 2 imports must retain their original provider identity even when the
# current reference backbone changes. Never relabel a historical source seal.
LEGACY_BACKBONES = {name: 'gpt-5.3-codex' if name == 'codex' else 'gpt-4.1-mini' for name in MODEL_IDS}
TOP_K = {'moira2-top1':1,'moira2-top3':3,'moira2-top5':5}
DATASET = 'CarolynJiang/MOIRA-Live-testing'
GITHUB_REPO = 'Carolyn-Jiang/MOIRA-Live-testing'
