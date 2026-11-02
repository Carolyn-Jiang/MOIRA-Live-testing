"""Exact Yahoo boundary valuations and individual-news releases, locally cached."""
from datetime import datetime,timedelta,timezone
import json
import math
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request,urlopen
from .registry import TICKERS,NY
from .calendar import DEFAULT_CALENDAR,aware
from .storage import read_json,write_private,now,digest
from .news import collect_articles
from .summary_batches import prepare_summaries
from .publish import publish_release

def release_id(at): return aware(at).strftime('%Y-%m-%d_%H-%M_NY')
def round_id(at): return aware(at).strftime('%Y-%m-%d_%H:%M')
def window_start(at):
    i=DEFAULT_CALENDAR.release_nodes.index(aware(at))
    return DEFAULT_CALENDAR.release_nodes[i-1] if i else datetime(2026,10,30,16,tzinfo=NY)

def prices(at):
    at=aware(at)
    if datetime.now(timezone.utc)<at:raise ValueError('Never collect an unreached price boundary')
    field='open' if (at.hour,at.minute)==(9,30) else 'close'
    target=at if field=='open' else at-timedelta(minutes=1);epoch=int(target.timestamp());values={};rows=[]
    for ticker in TICKERS:
        value=None;found=False
        paths=(f'v8/finance/chart/{ticker}?'+urlencode({'interval':'1m','range':'5d','includePrePost':'true'}),
               f'v8/finance/chart/{ticker}?'+urlencode({'interval':'1m','period1':epoch,'period2':epoch+60,'includePrePost':'true'}))
        for path in paths:
            for host in ('query1','query2'):
                try:
                    with urlopen(Request('https://'+host+'.finance.yahoo.com/'+path,headers={'User-Agent':'Mozilla/5.0'}),timeout=20) as response:
                        data=json.load(response)['chart']['result'][0]
                    stamps=data.get('timestamp') or []
                    if epoch not in stamps:continue
                    index=stamps.index(epoch);found=True;value=data['indicators']['quote'][0][field][index]
                    if type(value) in (float,int) and math.isfinite(value) and value>0:break
                    value=None
                except (OSError,ValueError,TypeError,KeyError,IndexError):continue
            if value is not None:break
        values[ticker]=value;rows.append({'ticker':ticker,'timestamp':at.isoformat(),'price':value,'currency':'USD',
            'price_basis':'09:30 minute open' if field=='open' else 'previous minute close',
            'bar_start':target.isoformat() if found else None,'status':'available' if value is not None else 'missing'})
    return values,rows

def _valid(values):
    return set(values)==set(TICKERS) and all(type(v) in (float,int) and math.isfinite(v) and v>0 for v in values.values())

def collect_release(root,at,gateway,*,historical=False,valuation=None):
    root=Path(root);at=aware(at);start=window_start(at);rid=release_id(at);path=root/'releases'/f'{rid}.json'
    cached=read_json(path)
    if cached is not None:
        if cached['timestamp']!=at.isoformat():raise ValueError('Immutable cached release timestamp changed')
        return cached
    raw_path=root/'inputs'/f'{rid}.json';raw=read_json(raw_path)
    if raw is None:
        if historical:
            i=DEFAULT_CALENDAR.release_nodes.index(at)+1
            if i not in (1,2):raise ValueError('Only the two explicitly authorized initial observations may be replayed')
            frozen=read_json(root/'seed'/f'legacy-inputs-{i}.json')
            # This is an already merged archive, not a newly capped search feed.
            # Preserve every frozen item even when one ticker has >200 records.
            write_private(root/'news/archives'/f'legacy-inputs-{i}.json',
                {'symbols':{ticker:{'articles':rows} for ticker,rows in frozen['groups'].items()}})
            def fetcher(ticker):
                return {'articles':[],'source':'frozen_original_release_archive_only','capped':False}
            collection=collect_articles(start,at,root/'news',fetcher=fetcher)
            values,rows=frozen['prices'],frozen['price_rows']
            observed=read_json(root/'book.json')['valuations'][at.isoformat()]['observed_at']
        else:
            if datetime.now(timezone.utc)<at:raise ValueError('No collection before the actual publication boundary')
            if valuation is None:
                values,rows=prices(at);observed=now()
            else:
                if aware(valuation['at']) != at or aware(valuation['observed_at']) < at:
                    raise ValueError('The reached immutable valuation must match this release')
                values=valuation['prices'];observed=valuation['observed_at']
                rows=valuation['provenance']['price_rows']
            collection=collect_articles(start,at,root/'news')
        raw={'timestamp':at.isoformat(),'prices':values,'price_rows':rows,'observed_at':observed,'collection':collection}
        write_private(raw_path,raw)
    if not _valid(raw['prices']):
        values,rows=prices(at);mapped={row['ticker']:row for row in rows}
        for ticker,value in values.items():
            if raw['prices'].get(ticker) is None and value is not None:
                raw['prices'][ticker]=value
                raw['price_rows']=[mapped[ticker] if r['ticker']==ticker else r for r in raw['price_rows']]
        raw['observed_at']=now();write_private(raw_path,raw)
    if not _valid(raw['prices']):raise RuntimeError('Exact price coverage incomplete; release and scoring remain pending')
    import tiktoken
    summarized=prepare_summaries(raw['collection'],gateway,root/'news',tiktoken.get_encoding('cl100k_base'))
    data={'timestamp':at.isoformat(),'window_start':start.isoformat(),'window_end':at.isoformat(),
        'prices':raw['prices'],'price_rows':raw['price_rows'],'observed_at':raw['observed_at'],
        'news_rows':summarized['news_rows'],'coverage':summarized['coverage'],
        'input_sha256':digest({'prices':raw['prices'],'articles':raw['collection']['articles']})}
    write_private(path,data);return data

def publish(root,data):
    return publish_release(data['price_rows'],data['news_rows'],release_id=release_id(data['timestamp']),
        window_start=data['window_start'],window_end=data['window_end'],cache_root=Path(root)/'news',coverage=data['coverage'])

def observation(data):
    news={ticker:[] for ticker in TICKERS}
    for row in data['news_rows']:
        news[row['ticker']].append(f"Published {row['published_at']} [{row['article_id']}]: {row['summary']}")
    return {'round_id':round_id(data['timestamp']),'timestamp':data['timestamp'],'prices':data['prices'],'news':news,
        'news_coverage':data['coverage'],'news_protocol':'individual_10_to_50_token_article_summaries'}
