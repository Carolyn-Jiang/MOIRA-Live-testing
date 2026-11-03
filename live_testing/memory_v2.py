"""Versioned quarter-scale, normalized and explicitly attributed RRI memory.

This module never reads a key, file, clock or provider by itself. Schema 1 and
its recorded ranks are never converted or mutated. All evidence is private.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
import hashlib
import json
import math
import re

from .calendar import NY, FULL_HOLIDAYS, HALF_DAYS, aware, is_trading_day
from .memory import MemoryUnit, Namespace, cosine, number

SCHEMA = 'moira.counterfactual-memory/2'
PROTOCOL_VERSION = 'quarter-normalized-attributed-ema/2'
TAU_MARKET_HOURS = 409.5
EMA = {'helpful':(1.,.02), 'harmful':(0.,.04), 'neutral':(.5,.01)}


def _stamp(value): return aware(value).isoformat()
def _digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def _sha(value):
    if not isinstance(value,str) or re.fullmatch('[0-9a-f]{64}',value) is None:
        raise ValueError('A real immutable SHA256 evidence reference is required')
    return value
def _unit(raw):
    if isinstance(raw,MemoryUnit):return copy.deepcopy(raw)
    value=copy.deepcopy(raw)
    for field in ('decision_at','entry_at','exit_at','available_at'):value[field]=aware(value[field])
    value['entities']=tuple(value['entities'])
    return MemoryUnit(**value)


class NYSEMarketHoursClock:
    """Actual open hours, not release counts; half-days close at 13:00 NY.

    These verified dates cover the historical training and full live period.
    A close boundary is integrated only; this creates no trading/valuation node.
    """
    first_date=date(2026,10,19)
    last_date=date(2027,5,24)
    version='nyse-open-hours-2026-10-19_2027-05-24/1'

    def __init__(self):self._hours_cache={}

    def to_dict(self):
        return {'version':self.version,'timezone':'America/New_York',
            'first_date':self.first_date.isoformat(),'last_date':self.last_date.isoformat(),
            'regular_open':'09:30','regular_close':'16:00','half_day_close':'13:00',
            'holidays':sorted(d.isoformat() for d in FULL_HOLIDAYS),
            'half_days':sorted(d.isoformat() for d in HALF_DAYS)}

    def hours(self,then,as_of):
        begin,end=aware(then),aware(as_of)
        lower=datetime.combine(self.first_date,time(),NY)
        upper=datetime.combine(self.last_date+timedelta(days=1),time(),NY)
        if begin>end or begin<lower or end>upper:
            raise ValueError('Ordered times inside the verified market-clock coverage are required')
        signature=(self.version,self.first_date,self.last_date,tuple(sorted(FULL_HOLIDAYS)),tuple(sorted(HALF_DAYS)))
        key=(signature,begin.timestamp(),end.timestamp())
        cached=self._hours_cache.get(key)
        if cached is not None:return cached
        day=begin.date();seconds=0.
        while day<=end.date():
            if is_trading_day(day):
                opening=datetime.combine(day,time(9,30),NY)
                closing=datetime.combine(day,time(13) if day in HALF_DAYS else time(16),NY)
                start=max(begin,opening);finish=min(end,closing)
                if finish>start:seconds+=finish.timestamp()-start.timestamp()
            day+=timedelta(days=1)
        result=seconds/3600.
        if len(self._hours_cache)>=65536:self._hours_cache.clear()
        self._hours_cache[key]=result
        return result


MarketHoursClock=NYSEMarketHoursClock
_DEFAULT_CLOCK=NYSEMarketHoursClock()


@dataclass(frozen=True)
class RetrievalHitV2:
    unit:MemoryUnit
    score:float
    raw_cosine:float
    relevance:float
    recency:float
    importance:float
    market_hours_age:float
    feedback_count:int
    updated_at:str
    clock:dict

    def to_prompt(self):
        return {**self.unit.to_prompt(),'retrieval':{
            **{key:getattr(self,key) for key in ('score','raw_cosine','relevance','recency',
                'importance','market_hours_age','feedback_count','updated_at')},
            'protocol_version':PROTOCOL_VERSION,'clock':copy.deepcopy(self.clock)}}


@dataclass(frozen=True)
class RetrievalBatchV2:
    retrieval_id:str
    namespace:Namespace
    as_of:datetime
    source_cutoff:datetime
    top_k:int
    hits:tuple[RetrievalHitV2,...]
    embedding_id:str
    feedback_revision:int


class CounterfactualMemoryBankV2:
    def __init__(self,namespace:Namespace,*,embedding_id,dimension,embed=None,clock=None):
        if not isinstance(namespace,Namespace) or not embedding_id or type(dimension)is not int or dimension<=0:
            raise ValueError('Explicit independent namespace, embedding identity and dimension required')
        self.namespace=namespace;self.embedding_id=embedding_id;self.dimension=dimension;self.embed=embed
        self.clock=clock or _DEFAULT_CLOCK
        self._units={};self._vectors={};self._bases={};self._action_ids={}
        self._retrievals={};self._feedback={};self._feedback_by_retrieval={};self._seed_bases={}
        self._events_by_unit={}

    def _vector(self,text,supplied):
        if supplied is None:
            if self.embed is None:raise RuntimeError('An explicit embedding provider or actual cached vector is required')
            supplied=self.embed(text)
        vector=tuple(number(value) for value in supplied)
        if len(vector)!=self.dimension:raise ValueError('Embedding dimension changed')
        cosine(vector,vector)
        return vector

    def _add(self,unit,vector,base):
        if unit.provenance.get('model_id',self.namespace.version)!=self.namespace.version:
            raise ValueError('Finalized units cannot cross independent model/version namespaces')
        if unit.unit_id in self._units:
            if (self._units[unit.unit_id].to_prompt()!=unit.to_prompt()
                    or self._vectors[unit.unit_id]!=tuple(vector) or self._bases[unit.unit_id]!=base):
                raise ValueError('An accepted immutable memory unit or seed changed')
            return unit.unit_id
        if unit.action_id in self._action_ids:raise ValueError('Only one memory per genuinely exited action is allowed')
        self._units[unit.unit_id]=copy.deepcopy(unit);self._vectors[unit.unit_id]=tuple(vector)
        self._bases[unit.unit_id]=copy.deepcopy(base);self._action_ids[unit.action_id]=unit.unit_id
        self._events_by_unit[unit.unit_id]=[]
        return unit.unit_id

    def add_finalized(self,unit,*,as_of,embedding=None):
        unit=_unit(unit);knowledge=aware(as_of)
        if aware(unit.available_at)>knowledge:raise ValueError('Unfinished or unavailable evidence cannot enter memory')
        self.clock.hours(unit.exit_at,unit.exit_at)
        if unit.unit_id in self._units:
            if (self._units[unit.unit_id].to_prompt()!=unit.to_prompt()
                    or embedding is not None and tuple(embedding)!=self._vectors[unit.unit_id]):
                raise ValueError('An accepted immutable memory unit or embedding changed')
            return unit.unit_id  # An idempotent retry never pays for another embedding.
        vector=self._vector(json.dumps(unit.to_prompt(),sort_keys=True),embedding)
        base={'importance':.5,'feedback_count':0,'updated_at':_stamp(unit.available_at),'seed_basis':None}
        return self._add(unit,vector,base)

    def importance_state_at(self,unit_id,as_of,*,source_cutoff=None,feedback_revision=None):
        if unit_id not in self._units:raise KeyError(unit_id)
        knowledge=aware(as_of);cutoff=aware(source_cutoff or as_of);unit=self._units[unit_id]
        revision=len(self._feedback) if feedback_revision is None else feedback_revision
        if type(revision)is not int or not 0<=revision<=len(self._feedback) or cutoff>knowledge:
            raise ValueError('A valid accepted feedback revision and causal cutoff are required')
        if aware(unit.available_at)>knowledge or aware(unit.exit_at)>cutoff:
            raise ValueError('An unavailable or unexited memory has no usable importance')
        base=self._bases[unit_id];value=base['importance'];count=base['feedback_count'];updated=base['updated_at']
        for event in self._events_by_unit[unit_id]:
            if event['revision']>revision:break
            attribution=event['attributions'].get(unit_id)
            if (attribution is None or aware(event['exit_at'])>cutoff or aware(event['available_at'])>knowledge
                    or aware(event['applied_at'])>knowledge):continue
            target,rate=EMA[attribution['label']];value=(1-rate)*value+rate*target
            count+=1;updated=event['applied_at']
        if not 0<=value<=1:raise ValueError('Normalized EMA importance escaped its valid domain')
        return {'importance':value,'feedback_count':count,'updated_at':updated}

    def importance_at(self,unit_id,as_of,*,source_cutoff=None,feedback_revision=None):
        return self.importance_state_at(unit_id,as_of,source_cutoff=source_cutoff,
            feedback_revision=feedback_revision)['importance']

    def _rank(self,vector,knowledge,cutoff,top_k,entities,revision):
        hits=[]
        for key,unit in self._units.items():
            if (aware(unit.available_at)>knowledge or aware(unit.exit_at)>cutoff
                    or entities and not set(entities).intersection(unit.entities)):continue
            hours=self.clock.hours(unit.exit_at,cutoff);recency=math.exp(-hours/TAU_MARKET_HOURS)
            raw=cosine(vector,self._vectors[key]);relevance=(max(-1.,min(1.,raw))+1)/2
            state=self.importance_state_at(key,knowledge,source_cutoff=cutoff,feedback_revision=revision)
            importance=state['importance'];score=recency+relevance+importance
            if not all(0<=v<=1 for v in (recency,relevance,importance)) or not 0<=score<=3:
                raise ValueError('Every normalized component and total must remain in the declared range')
            hits.append(RetrievalHitV2(copy.deepcopy(unit),score,raw,relevance,recency,importance,hours,
                state['feedback_count'],state['updated_at'],self.clock.to_dict()))
        hits.sort(key=lambda hit:(-hit.score,-hit.unit.exit_at.timestamp(),hit.unit.unit_id))
        return tuple(hits[:top_k])

    def retrieve(self,query,*,as_of,source_cutoff,top_k,retrieval_id,query_embedding=None,entities=None):
        knowledge=aware(as_of);cutoff=aware(source_cutoff)
        if (cutoff>knowledge or type(top_k)is not int or top_k not in (1,3,5)
                or not isinstance(query,str) or not query.strip() or not isinstance(retrieval_id,str) or not retrieval_id):
            raise ValueError('A real query, top K 1/3/5 and causal source/knowledge times are required')
        entity_filter=tuple(sorted(set(entities or ())))
        if retrieval_id in self._retrievals:
            old=self._retrievals[retrieval_id]
            if ((query,_stamp(knowledge),_stamp(cutoff),top_k,entity_filter)!=
                    (old['query'],old['as_of'],old['source_cutoff'],old['top_k'],tuple(old['entities']))
                    or query_embedding is not None and tuple(query_embedding)!=tuple(old['query_vector'])):
                raise ValueError('An accepted retrieval is immutable')
            return copy.deepcopy(old['batch'])
        latest=[aware(row['as_of'])for row in self._retrievals.values()]
        latest += [aware(row['applied_at'])for row in self._feedback.values()]
        if latest and knowledge<max(latest):
            raise ValueError('A new retrieval cannot backdate accepted knowledge history')
        vector=self._vector(query,query_embedding);revision=len(self._feedback)
        batch=RetrievalBatchV2(retrieval_id,self.namespace,knowledge,cutoff,top_k,
            self._rank(vector,knowledge,cutoff,top_k,entity_filter,revision),self.embedding_id,revision)
        self._retrievals[retrieval_id]={'query':query,'as_of':_stamp(knowledge),'source_cutoff':_stamp(cutoff),
            'top_k':top_k,'entities':list(entity_filter),'query_vector':list(vector),
            'feedback_revision':revision,'batch':copy.deepcopy(batch)}
        return batch

    def apply_attributed_feedback(self,feedback_id,*,retrieval_id,used_memory_ids,attributions,
            action_evidence_sha256,outcome_evidence_sha256,exit_at,available_at,as_of,source_cutoff):
        if not isinstance(feedback_id,str) or not feedback_id or retrieval_id not in self._retrievals:
            raise ValueError('Feedback must identify an actual accepted retrieval and immutable event')
        used=list(used_memory_ids);provided={hit.unit.unit_id for hit in self._retrievals[retrieval_id]['batch'].hits}
        if (len(set(used))!=len(used) or not set(used)<=provided or not isinstance(attributions,dict)
                or set(attributions)!=set(used)):
            raise ValueError('Only exact, actually cited provided memory IDs may receive credit')
        normalized={}
        for key in sorted(attributions):
            row=attributions[key]
            if not isinstance(row,dict) or set(row)!={'label','evidence_sha256'} or row['label']not in EMA:
                raise ValueError('Explicit per-unit helpful/harmful/neutral attribution evidence required')
            normalized[key]={'label':row['label'],'evidence_sha256':_sha(row['evidence_sha256'])}
        exit,available,knowledge,cutoff=map(aware,(exit_at,available_at,as_of,source_cutoff))
        query=self._retrievals[retrieval_id]
        if (not aware(query['source_cutoff'])<=exit<=cutoff<=knowledge or not exit<=available<=knowledge
                or available<aware(query['as_of'])):
            raise ValueError('Only genuinely exited, source-known, analysis-available outcomes may update memory')
        payload={'feedback_id':feedback_id,'retrieval_id':retrieval_id,'used_memory_ids':sorted(used),
            'attributions':normalized,'action_evidence_sha256':_sha(action_evidence_sha256),
            'outcome_evidence_sha256':_sha(outcome_evidence_sha256),'exit_at':_stamp(exit),
            'available_at':_stamp(available),'source_cutoff':_stamp(cutoff)}
        if feedback_id in self._feedback:
            existing={key:value for key,value in self._feedback[feedback_id].items()
                      if key not in ('applied_at','revision','unit_updates')}
            if existing!=payload:raise ValueError('A completed attribution event cannot change')
            return False
        if retrieval_id in self._feedback_by_retrieval:
            raise ValueError('One action retrieval may receive only one completed atomic attribution')
        latest=[aware(row['as_of'])for row in self._retrievals.values()]
        latest += [aware(row['applied_at'])for row in self._feedback.values()]
        if latest and knowledge<max(latest):
            raise ValueError('Attribution application cannot backdate accepted bank knowledge')
        event={**payload,'applied_at':_stamp(knowledge),'revision':len(self._feedback)+1,'unit_updates':{}}
        for key,row in normalized.items():
            before=self.importance_state_at(key,knowledge,source_cutoff=cutoff)
            target,rate=EMA[row['label']];after=(1-rate)*before['importance']+rate*target
            event['unit_updates'][key]={'before':before['importance'],'after':after,'target':target,'eta':rate,
                'feedback_count_before':before['feedback_count'],'feedback_count_after':before['feedback_count']+1,
                'updated_at':_stamp(knowledge)}
        self._feedback[feedback_id]=event;self._feedback_by_retrieval[retrieval_id]=feedback_id
        for key in normalized:self._events_by_unit[key].append(event)
        return True

    def export_seed(self,*,as_of,source_cutoff):
        result=[]
        for key,unit in self._units.items():
            if aware(unit.available_at)>aware(as_of) or aware(unit.exit_at)>aware(source_cutoff):continue
            state=self.importance_state_at(key,as_of,source_cutoff=source_cutoff)
            result.append({'unit':unit.to_prompt(),'vector':list(self._vectors[key]),**state})
        return result

    def seed_from_training_bank(self,snapshot,*,checkpoint_sha256,completed_at,source_cutoff,as_of):
        if self.namespace.scope!='live' or snapshot.get('seed_bases'):
            raise ValueError('Only a fresh trained schema-2 bank may seed a live namespace')
        source_namespace=Namespace(**snapshot['namespace'])
        if source_namespace.scope!='training' or source_namespace.version!=self.namespace.version:
            raise ValueError('Trained priors cannot cross model/version namespaces')
        completed,cutoff,knowledge=map(aware,(completed_at,source_cutoff,as_of))
        if not cutoff<=completed<=knowledge:raise ValueError('Actual training completion must precede live knowledge')
        source=type(self).from_dict(snapshot,namespace=source_namespace,clock=self.clock)
        if source.embedding_id!=self.embedding_id or source.dimension!=self.dimension:
            raise ValueError('The actual trained embedding provider/dimension must remain unchanged')
        if any(aware(unit.available_at)>completed or aware(unit.exit_at)>cutoff for unit in source._units.values()):
            raise ValueError('Every imported trained unit must be genuinely completed and source-known')
        for query in snapshot['retrievals']:
            if aware(query['as_of'])>completed or aware(query['source_cutoff'])>cutoff:
                raise ValueError('Future training query evidence cannot seed live memory')
        for event in snapshot['feedback']:
            if aware(event['applied_at'])>completed or aware(event['exit_at'])>cutoff:
                raise ValueError('Future trained feedback cannot seed live memory')
        basis={'snapshot':copy.deepcopy(snapshot),'source_snapshot_sha256':_digest(snapshot),
            'checkpoint_sha256':_sha(checkpoint_sha256),'completed_at':_stamp(completed),'source_cutoff':_stamp(cutoff)}
        key=_digest(basis)
        if key in self._seed_bases:return key
        planned=[]
        for row in source.export_seed(as_of=completed,source_cutoff=cutoff):
            raw=copy.deepcopy(row['unit']);raw['available_at']=_stamp(completed)
            raw['provenance']['v2_training_seed']={'basis_sha256':key,'source_unit_available_at':row['unit']['available_at'],
                'learned_prior_available_at':_stamp(completed),'embedding_preserved_from_training':True}
            base={'importance':row['importance'],'feedback_count':row['feedback_count'],
                  'updated_at':_stamp(completed),'seed_basis':key}
            planned.append((_unit(raw),row['vector'],base))
        if any(unit.unit_id in self._units or unit.action_id in self._action_ids for unit,_,_ in planned):
            raise ValueError('Trained memory seed collides with an existing live action')
        self._seed_bases[key]=basis
        for unit,vector,base in planned:self._add(unit,vector,base)
        return key

    def to_dict(self):
        units=[]
        for key,unit in self._units.items():
            knowledge=max([aware(unit.available_at)]+[aware(x['applied_at'])for x in self._feedback.values()])
            state=self.importance_state_at(key,knowledge,source_cutoff=knowledge)
            units.append({'unit':unit.to_prompt(),'vector':list(self._vectors[key]),'initial_state':copy.deepcopy(self._bases[key]),
                          'state':state})
        queries=[]
        for key,row in self._retrievals.items():
            queries.append({'retrieval_id':key,**{name:copy.deepcopy(value)for name,value in row.items()if name!='batch'},
                'hits':[{'unit_id':hit.unit.unit_id,**{name:getattr(hit,name)for name in ('score','raw_cosine','relevance',
                    'recency','importance','market_hours_age','feedback_count','updated_at')}}for hit in row['batch'].hits]})
        result={'schema':SCHEMA,'protocol_version':PROTOCOL_VERSION,'namespace':vars(self.namespace),
            'embedding_id':self.embedding_id,'dimension':self.dimension,'clock':self.clock.to_dict(),
            'tau_market_hours':TAU_MARKET_HOURS,'initial_importance':.5,
            'ema_policy':{label:{'target':target,'eta':eta}for label,(target,eta)in EMA.items()},
            'seed_bases':copy.deepcopy(self._seed_bases),'units':units,'retrievals':queries,
            'feedback':copy.deepcopy(list(self._feedback.values()))}
        json.dumps(result,allow_nan=False)
        return result

    @classmethod
    def from_dict(cls,data,*,namespace,clock=None,embed=None,embedding_id=None):
        if (data.get('schema')!=SCHEMA or data.get('protocol_version')!=PROTOCOL_VERSION
                or Namespace(**data['namespace'])!=namespace):
            raise ValueError('Schema-1 or cross-namespace snapshots cannot enter the normalized v2 bank')
        bank=cls(namespace,embedding_id=data['embedding_id'],dimension=data['dimension'],clock=clock,embed=embed)
        if embedding_id is not None and embedding_id!=bank.embedding_id:raise ValueError('Embedding model changed')
        if (data.get('clock')!=bank.clock.to_dict() or data.get('tau_market_hours')!=TAU_MARKET_HOURS
                or data.get('initial_importance')!=.5
                or data.get('ema_policy')!={label:{'target':target,'eta':eta}for label,(target,eta)in EMA.items()}):
            raise ValueError('A frozen v2 market clock or normalization/EMA policy changed')
        for key,basis in data['seed_bases'].items():
            if key!=_digest(basis) or basis['source_snapshot_sha256']!=_digest(basis['snapshot']):
                raise ValueError('A learned training seed basis changed')
            restored=bank.seed_from_training_bank(basis['snapshot'],checkpoint_sha256=basis['checkpoint_sha256'],
                completed_at=basis['completed_at'],source_cutoff=basis['source_cutoff'],as_of=basis['completed_at'])
            if restored!=key:raise ValueError('Trained prior transfer proof changed')
        for row in data['units']:
            unit=_unit(row['unit']);base=row['initial_state']
            if base['seed_basis'] is None:
                bank.add_finalized(unit,as_of=unit.available_at,embedding=row['vector'])
            if (unit.unit_id not in bank._units or bank._units[unit.unit_id].to_prompt()!=unit.to_prompt()
                    or bank._vectors[unit.unit_id]!=tuple(row['vector']) or bank._bases[unit.unit_id]!=base):
                raise ValueError('An immutable unit, embedding or verified initial prior changed')
        if len(bank._units)!=len(data['units']):raise ValueError('Missing or duplicate memory units')
        events=data['feedback'];queries=data['retrievals'];event_ids=set();query_ids=set()
        # Revisions reconstruct accepted knowledge even when a feedback arrives
        # later than its outcome or shares a timestamp with an earlier query.
        for revision in range(len(events)+1):
            for row in queries:
                if row['feedback_revision']!=revision:continue
                if row['retrieval_id']in query_ids:raise ValueError('Duplicate retrieval provenance')
                query_ids.add(row['retrieval_id'])
                batch=bank.retrieve(row['query'],as_of=row['as_of'],source_cutoff=row['source_cutoff'],
                    top_k=row['top_k'],retrieval_id=row['retrieval_id'],query_embedding=row['query_vector'],entities=row['entities'])
                expected=[{'unit_id':hit.unit.unit_id,**{name:getattr(hit,name)for name in ('score','raw_cosine','relevance',
                    'recency','importance','market_hours_age','feedback_count','updated_at')}}for hit in batch.hits]
                if expected!=row['hits']:raise ValueError('Stored v2 rank, components or causal time provenance changed')
            if revision==len(events):break
            row=events[revision]
            if row['feedback_id']in event_ids or row['revision']!=revision+1:raise ValueError('Duplicate or unordered feedback event')
            event_ids.add(row['feedback_id'])
            bank.apply_attributed_feedback(row['feedback_id'],retrieval_id=row['retrieval_id'],
                used_memory_ids=row['used_memory_ids'],attributions=row['attributions'],
                action_evidence_sha256=row['action_evidence_sha256'],outcome_evidence_sha256=row['outcome_evidence_sha256'],
                exit_at=row['exit_at'],available_at=row['available_at'],as_of=row['applied_at'],source_cutoff=row['source_cutoff'])
            if bank._feedback[row['feedback_id']]!=row:raise ValueError('Stored per-unit EMA update evidence changed')
        if len(query_ids)!=len(queries):raise ValueError('A retrieval refers to a nonexistent feedback revision')
        restored=bank.to_dict()
        if restored!=data:raise ValueError('The complete causal v2 snapshot failed deterministic restoration')
        return bank
