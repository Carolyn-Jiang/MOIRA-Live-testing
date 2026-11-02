"""Atomic, immutable publication to the explicitly private experiment Dataset."""
from datetime import datetime
import hashlib
import io
import json
from pathlib import Path
import re

from .calendar import aware
from .registry import DATASET,TICKERS
from .storage import write_private,now,digest
from .news import private_root


class PublicationConflict(RuntimeError):pass


def _bytes(value):return (json.dumps(value,sort_keys=True,ensure_ascii=False,allow_nan=False,indent=2)+'\n').encode()
def _sha(value):return hashlib.sha256(value).hexdigest()


def _read_remote(api,dataset,filename,revision):
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError
    try:
        path=hf_hub_download(dataset,filename,repo_type='dataset',revision=revision,token=api.token)
        return Path(path).read_bytes()
    except EntryNotFoundError:return None


def _private_info(api,dataset):
    info=api.repo_info(dataset,repo_type='dataset')
    if info.private is not True:raise RuntimeError('Refuse publication: experiment Dataset must remain private')
    if not info.sha:raise RuntimeError('Private Dataset commit is unavailable')
    return info


def _parquet(rows,*,news=False):
    import pyarrow as pa
    import pyarrow.parquet as pq
    if rows:table=pa.Table.from_pylist(rows)
    elif news:table=pa.table({name:pa.array([],type=pa.string()) for name in
        ('ticker','article_id','published_at','title','summary','content_basis','window_start','window_end','summary_generated_at')})
    else:raise ValueError('Ten actual price rows required')
    buffer=io.BytesIO();pq.write_table(table,buffer,compression='zstd');return buffer.getvalue()


def publish_release(price_rows,news_rows,*,release_id,window_start,window_end,cache_root,
                    api=None,dataset=DATASET,reader=None,coverage=None):
    """One CAS commit contains price, per-article news, release and global manifest.

    A retry reads back the exact saved byte hashes. Changed data under an existing
    release ID is rejected; no later run can overwrite a historical publication.
    """
    from huggingface_hub import HfApi,CommitOperationAdd
    if dataset!=DATASET:raise ValueError('Only the named private experiment Dataset is permitted')
    if not isinstance(release_id,str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,100}',release_id):raise ValueError('Safe immutable release ID required')
    start,end=aware(window_start),aware(window_end)
    if start>=end:raise ValueError('Ordered source-publication window required')
    if (not isinstance(price_rows,list) or len(price_rows)!=10 or {row.get('ticker') for row in price_rows}!=set(TICKERS)):
        raise ValueError('Exactly ten eligible price rows required')
    import math
    for row in price_rows:
        if type(row.get('price')) not in (int,float) or not math.isfinite(row['price']) or row['price']<=0 or aware(row['timestamp'])!=end:
            raise ValueError('All ten released prices must be finite and match this boundary')
    if not isinstance(news_rows,list):raise ValueError('Per-article news rows required')
    for row in news_rows:
        if row.get('ticker') not in TICKERS or not start<=aware(row['published_at'])<end or not row.get('article_id') or not row.get('summary'):
            raise ValueError('News must contain actual articles inside this publication window')
        if aware(row['window_start'])!=start or aware(row['window_end'])!=end:raise ValueError('News row window changed')
        if type(row.get('summary_tokens')) is not int or not 10<=row['summary_tokens']<=50:
            raise ValueError('Every news row requires its verified 10–50 token article summary count')
    root=private_root(cache_root);folder='data/'+release_id
    assets={folder+'/price.parquet':_parquet(price_rows),folder+'/news.parquet':_parquet(news_rows,news=True)}
    data_hashes={name.rsplit('/',1)[-1]:_sha(blob) for name,blob in assets.items()}
    api=api or HfApi();reader=reader or (lambda name,revision:_read_remote(api,dataset,name,revision))
    info=_private_info(api,dataset);raw=reader('releases.json',info.sha)
    manifest=json.loads(raw) if raw is not None else {'schema':'moira.live-testing.releases/1','releases':[]}
    if set(manifest)!={'schema','releases'} or manifest['schema']!='moira.live-testing.releases/1' or not isinstance(manifest['releases'],list):raise ValueError('Invalid existing experiment release manifest')
    records={row['release_id']:row for row in manifest['releases']}
    if len(records)!=len(manifest['releases']):raise ValueError('Duplicate existing release identity')
    existing=records.get(release_id)
    if existing:
        if existing['file_sha256']!=data_hashes or aware(existing['window_start'])!=start or aware(existing['window_end'])!=end:
            raise PublicationConflict('Published release is immutable; different bytes or source window rejected')
        for name,blob in assets.items():
            remote=reader(name,info.sha)
            if remote is None or _sha(remote)!=_sha(blob):raise PublicationConflict('Existing release byte verification failed')
        release_raw=reader(folder+'/release.json',info.sha)
        if release_raw is None or json.loads(release_raw)!=existing:raise PublicationConflict('Existing release metadata differs')
        return {**existing,'commit':info.sha,'idempotent':True}
    record={'release_id':release_id,'published_at':now(),'window_start':start.isoformat(),'window_end':end.isoformat(),
        'folder':folder,'file_sha256':data_hashes,'price_rows':len(price_rows),'news_rows':len(news_rows),'coverage':coverage}
    assets[folder+'/release.json']=_bytes(record)
    manifest['releases'].append(record);assets['releases.json']=_bytes(manifest)
    for name,blob in assets.items():
        path=root/'hf-staging'/name;path.parent.mkdir(parents=True,exist_ok=True,mode=0o700);path.write_bytes(blob);path.chmod(0o600)
    # Visibility is checked again immediately before the only remote mutation.
    current=_private_info(api,dataset)
    if current.sha!=info.sha:raise PublicationConflict('Dataset advanced during preparation; retry the read-only checks')
    api.auth_check(repo_id=dataset,repo_type='dataset',write=True)
    result=api.create_commit(repo_id=dataset,repo_type='dataset',parent_commit=info.sha,
        commit_message='Publish verified experiment release '+release_id,
        operations=[CommitOperationAdd(path_in_repo=name,path_or_fileobj=blob) for name,blob in assets.items()])
    commit=result.oid
    if not commit:raise RuntimeError('Dataset publication returned no commit')
    if _private_info(api,dataset).private is not True:raise RuntimeError('Private visibility changed after commit')
    for name,blob in assets.items():
        remote=reader(name,commit)
        if remote is None or _sha(remote)!=_sha(blob):raise PublicationConflict('Committed Dataset bytes failed exact hash verification')
    write_private(root/'publication-audit'/f'{release_id}.json',{'commit':commit,'release':record,
        'verified_at':now(),'asset_sha256':{name:_sha(blob) for name,blob in assets.items()},'dataset_private':True})
    return {**record,'commit':commit,'idempotent':False}
