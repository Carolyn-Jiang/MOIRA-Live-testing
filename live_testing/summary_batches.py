"""Batched transport, independent article summaries, private resumable caches."""
import copy
import hashlib
import json
import uuid
from concurrent.futures import ThreadPoolExecutor,as_completed

from .calendar import aware
from .news import SUMMARY_VERSION,SUMMARY_INSTRUCTIONS,private_root,summarize_articles
from .storage import digest,read_json,write_private,now

MAX_ARTICLES=15
MAX_PAYLOAD_BYTES=80_000
MAX_SINGLE_RETRIES=2


class SummariesPending(RuntimeError):pass


def _input(article):
    return {key:copy.deepcopy(article.get(key)) for key in
        ('article_id','title','published_at','publisher','body','content_basis','content_sha256')}


def _payload(rows):return {'articles':[_input(article) for article in rows]}
def _size(rows):return len(json.dumps(_payload(rows),ensure_ascii=False,allow_nan=False).encode())
def _path(root,article):return root/'summaries'/f'{digest([SUMMARY_VERSION,article["article_id"],article["content_sha256"]])}.json'


def _valid_text(text,encoder):return isinstance(text,str) and 10<=len(encoder.encode(text))<=50


def _store(root,article,text,encoder):
    if not _valid_text(text,encoder):raise ValueError('Article summary is not 10–50 cl100k_base tokens')
    record={'article_id':article['article_id'],'content_sha256':article['content_sha256'],
        'summary_version':SUMMARY_VERSION,'summary':text,'summary_tokens':len(encoder.encode(text)),
        'summary_sha256':hashlib.sha256(text.encode()).hexdigest(),'summary_generated_at':now(),
        'source_published_at':article['published_at'],'content_basis':article['content_basis']}
    write_private(_path(root,article),record)


def _cached(root,article,encoder):
    record=read_json(_path(root,article))
    if record is None:return False
    if (record.get('article_id')!=article['article_id'] or record.get('content_sha256')!=article['content_sha256']
            or record.get('summary_version')!=SUMMARY_VERSION or not _valid_text(record.get('summary'),encoder)
            or record.get('summary_tokens')!=len(encoder.encode(record['summary']))
            or record.get('summary_sha256')!=hashlib.sha256(record['summary'].encode()).hexdigest()
            or aware(record['source_published_at'])!=aware(article['published_at'])):
        raise ValueError('Existing summary cache failed validation; no automatic paid regeneration')
    return True


def _batches(rows):
    batch=[]
    for article in rows:
        if _size([article])>MAX_PAYLOAD_BYTES:raise SummariesPending('A single article exceeds the bounded summary payload; review required')
        if batch and (len(batch)>=MAX_ARTICLES or _size(batch+[article])>MAX_PAYLOAD_BYTES):yield batch;batch=[]
        batch.append(article)
    if batch:yield batch


def _schema(rows):
    return {'type':'object','properties':{'summaries':{'type':'array','minItems':len(rows),'maxItems':len(rows),
        'items':{'type':'object','properties':{'article_id':{'type':'string','enum':[a['article_id'] for a in rows]},
            'summary':{'type':'string'}},'required':['article_id','summary'],'additionalProperties':False}}},
        'required':['summaries'],'additionalProperties':False}


def prepare_summaries(collection,gateway,cache_root,encoder,*,max_workers=4):
    """Prime the exact news.py cache; never summarize more than one source as one.

    Only malformed individual summaries get up to two single-article retries.
    Up to four independent batches run concurrently. Every batch finishes or
    records its error before a pending error is returned; successes are retained.
    """
    root=private_root(cache_root);articles=collection['articles']
    if type(max_workers) is not int or not 1<=max_workers<=4:raise ValueError('At most four concurrent summary batches are permitted')
    if len({a['article_id'] for a in articles})!=len(articles):raise ValueError('Global article deduplication required before batching')
    start,end=aware(collection['window_start']),aware(collection['window_end'])
    for article in articles:
        if not start<=aware(article['published_at'])<end:raise ValueError('Source publication lies outside this release window')
    pending=[article for article in articles if not _cached(root,article,encoder)]
    event_id='news:'+collection['window_end'];scoped=gateway.scoped('data','data_summary',event_id)
    def process_batch(batch):
        identity=uuid.uuid4().hex;audit=root/'summary-batch-audit'/f'{identity}.json';payload=_payload(batch)
        record={'schema':'moira.live-testing.summary-batch/1','batch_id':identity,'event_id':event_id,
            'article_ids':[a['article_id'] for a in batch],'request_payload_sha256':digest(payload),
            'payload_bytes':_size(batch),'started_at':now(),'status':'started','completed_article_ids':[],'single_retry_calls':0}
        write_private(audit,record)
        try:
            response=scoped.json(model='gpt-4.1-mini',payload=payload,
                instructions=SUMMARY_INSTRUCTIONS+' Each array item is a separate article. Produce exactly one independent summary per article_id; never combine articles or transfer claims between articles. Preserve each exact article_id.',
                schema=_schema(batch),name='independent_article_summaries',max_output_tokens=4096)
        except Exception as error:
            record.update(status='pending',finished_at=now(),error_type=type(error).__name__);write_private(audit,record)
            raise SummariesPending('Summary batch transport failed; completed caches preserved') from error
        result=response.get('summaries',[]) if isinstance(response,dict) and set(response)=={'summaries'} else []
        candidates={};duplicates=set()
        if isinstance(result,list):
            for item in result:
                if not isinstance(item,dict) or set(item)!={'article_id','summary'} or not isinstance(item['article_id'],str):continue
                if item['article_id'] in candidates:duplicates.add(item['article_id'])
                candidates[item['article_id']]=item['summary']
        invalid=[]
        for article in batch:
            summary=candidates.get(article['article_id'])
            if article['article_id'] not in duplicates and _valid_text(summary,encoder):
                _store(root,article,summary,encoder);record['completed_article_ids'].append(article['article_id']);write_private(audit,record)
            else:invalid.append(article)
        failed_ids=[]
        for article in invalid:
            success=False;previous=candidates.get(article['article_id'])
            for attempt in range(MAX_SINGLE_RETRIES):
                record['single_retry_calls']+=1;write_private(audit,record)
                retry={**_input(article),'summary_instructions':SUMMARY_INSTRUCTIONS,
                    'previous_summary':previous if isinstance(previous,str) else '',
                    'previous_token_count':len(encoder.encode(previous)) if isinstance(previous,str) else 0,
                    'revision_request':{'kind':'length_bound','attempt':attempt+1,'target_min_tokens':25,'target_max_tokens':35}}
                try:summary=scoped.summary(retry)
                except Exception as error:
                    record.update(error_type=type(error).__name__);write_private(audit,record)
                    break
                if _valid_text(summary,encoder):
                    _store(root,article,summary,encoder);record['completed_article_ids'].append(article['article_id']);write_private(audit,record);success=True;break
                previous=summary
            if not success:
                failed_ids.append(article['article_id'])
        if failed_ids:
            record.update(status='pending',finished_at=now(),failed_article_ids=failed_ids,
                error_type=record.get('error_type','SummaryTokenBounds'));write_private(audit,record)
            raise SummariesPending('Article summaries remain pending after at most two individual retries')
        record.update(status='completed',finished_at=now());write_private(audit,record)
    batches=list(_batches(pending));errors=[]
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures={executor.submit(process_batch,batch):batch for batch in batches}
        for future in as_completed(futures):
            try:future.result()
            except Exception as error:errors.append(type(error).__name__)
    if errors:raise SummariesPending(f'{len(errors)} independent summary batch(es) remain pending; all completed article caches preserved')
    def cached_only(article,_stage):
        raise RuntimeError('Unprimed summary unexpectedly reached; no implicit extra model call')
    return summarize_articles(collection,cached_only,root,encoder=encoder)
