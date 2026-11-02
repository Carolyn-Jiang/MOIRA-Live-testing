"""Exact-score snapshot restoration for the persistent experiment.

The snapshot schema and inherited full causal replay validation are unchanged.
NumPy only selects a conservative candidate superset; every returned score is
computed by memory.cosine and the original scalar arithmetic/tie-break order.
Extreme numerical scales fall back to exact scoring of every eligible unit.
"""
from datetime import datetime,time,timedelta
import copy
import math
import threading

import numpy as np

from .calendar import NY,aware,is_trading_day
from .memory import CounterfactualMemoryBank,RetrievalHit,cosine,instant


class IndexedExchangeSessionAge:
    """Count actual opens in (then, as_of], caching one ordinal per local date.

    Uses the same frozen trading-day predicate as the existing native helper.
    It is opt-in: the memory-bank subclass does not replace arbitrary supplied
    session_age functions with a different calendar.
    """
    def __init__(self,trading_day=is_trading_day):
        self.trading_day=trading_day;self._days={};self._ordinals={};self._lock=threading.RLock()

    def _trading(self,day):
        if day not in self._days:self._days[day]=bool(self.trading_day(day))
        return self._days[day]

    def _ordinal(self,at):
        day=at.date()
        if not self._ordinals:self._ordinals[day]=0;self._lo=self._hi=day
        while day<self._lo:
            previous=self._lo-timedelta(days=1)
            self._ordinals[previous]=self._ordinals[self._lo]-int(self._trading(self._lo));self._lo=previous
        while day>self._hi:
            following=self._hi+timedelta(days=1)
            self._ordinals[following]=self._ordinals[self._hi]+int(self._trading(following));self._hi=following
        return self._ordinals[day]-int(self._trading(day) and at<datetime.combine(day,time(9,30),NY))

    def __call__(self,then,as_of):
        then,as_of=aware(then),aware(as_of)
        if then>as_of:raise ValueError('Future session age prohibited')
        with self._lock:return self._ordinal(as_of)-self._ordinal(then)


class FastCounterfactualMemoryBank(CounterfactualMemoryBank):
    """API/schema-compatible kernel retaining exact accepted hit values."""
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self._feedback_by_unit={};self._age_cache={};self._matrix=None;self._matrix_keys=()
        self._fast_diagnostics={'rank_calls':0,'exact_cosines':0,'eligible_units':0,'fallback_ranks':0}

    def add_finalized(self,unit,*,as_of,embedding=None):
        existed=unit.unit_id in self._units
        result=super().add_finalized(unit,as_of=as_of,embedding=embedding)
        if not existed:self._matrix=None;self._feedback_by_unit[unit.unit_id]=[]
        return result

    def apply_outcome_feedback(self,retrieval_id,*,actual_return,exit_at,available_at,as_of):
        existed=retrieval_id in self._feedback
        super().apply_outcome_feedback(retrieval_id,actual_return=actual_return,exit_at=exit_at,
            available_at=available_at,as_of=as_of)
        if not existed:
            event=self._feedback[retrieval_id]
            for key in event['unit_ids']:self._feedback_by_unit[key].append(event)

    def importance_at(self,unit_id,as_of,*,source_cutoff=None):
        as_of=instant(as_of);cutoff=instant(source_cutoff)if source_cutoff is not None else as_of
        if unit_id not in self._units:raise KeyError(unit_id)
        return self.initial_importance+self.feedback_step*sum(event['direction']
            for event in self._feedback_by_unit[unit_id]
            if event['available_at']<=as_of and event['exit_at']<=cutoff)

    def _prepare_matrix(self):
        if self._matrix is None:
            self._matrix_keys=tuple(self._units)
            self._matrix=np.array([self._vectors[key]for key in self._matrix_keys],dtype=np.float64)
            with np.errstate(over='ignore',under='ignore',invalid='ignore'):
                self._matrix_norms=np.sqrt(np.einsum('ij,ij->i',self._matrix,self._matrix,optimize=False))

    def _rank(self,vector,as_of,top_k,entity_filter,cutoff):
        self._fast_diagnostics['rank_calls']+=1
        if type(top_k)is not int or top_k not in (1,3,5):
            return super()._rank(vector,as_of,top_k,entity_filter,cutoff)
        if not self._units:return ()
        self._prepare_matrix();eligible=[];entities=set(entity_filter)
        for index,key in enumerate(self._matrix_keys):
            unit=self._units[key]
            if unit.available_at>as_of or unit.exit_at>cutoff or(entities and not entities.intersection(unit.entities)):continue
            age_key=(unit.exit_at,cutoff)
            if age_key not in self._age_cache:
                age=self.session_age(unit.exit_at,cutoff)
                if type(age)is not int or age<0:raise ValueError('Causal integer exchange-session age required')
                # Historical replay must not retain an O(units × queries) cache.
                # The opt-in ordinal helper retains only one entry per date.
                if len(self._age_cache)>=8192:self._age_cache.clear()
                self._age_cache[age_key]=age
            age=self._age_cache[age_key];recency=math.exp(-age/self.tau)
            importance=self.importance_at(key,as_of,source_cutoff=cutoff);normalized=min(importance,100.)/100.
            eligible.append((index,key,age,recency,importance,normalized))
        self._fast_diagnostics['eligible_units']+=len(eligible)
        if not eligible:return ()
        chosen=eligible
        query=np.asarray(vector,dtype=np.float64)
        with np.errstate(over='ignore',under='ignore',invalid='ignore',divide='ignore'):
            qnorm=float(np.sqrt(np.einsum('i,i->',query,query,optimize=False)))
            norms=self._matrix_norms
            safe=(self.dimension*np.finfo(np.float64).eps<1e-4 and math.isfinite(qnorm)
                and 1e-100<=qnorm<=1e100 and np.all(np.isfinite(norms))
                and np.all((norms>=1e-100)&(norms<=1e100)))
            if safe:
                approximate=np.clip(np.einsum('ij,j->i',self._matrix,query,optimize=False)/(norms*qnorm),-1.,1.)
                scores=np.asarray([float(approximate[i])+normalized+recency
                    for i,_key,_age,recency,_importance,normalized in eligible])
                eps=np.finfo(np.float64).eps;gamma=self.dimension*eps/(1-self.dimension*eps)
                # Conservative float64 dot/norm error allowance, also covering
                # scalar score additions. All boundary-near ties are exact.
                band=max(1e-10,64*gamma)+8*eps*max(1.,float(np.max(np.abs(scores))))
                if np.all(np.isfinite(scores))and math.isfinite(band):
                    boundary=float(np.partition(scores,-min(top_k,len(scores)))[-min(top_k,len(scores))])
                    chosen=[row for row,score in zip(eligible,scores)if score>=boundary-band]
                else:safe=False
        if not safe:self._fast_diagnostics['fallback_ranks']+=1
        hits=[]
        for _index,key,age,recency,importance,normalized in chosen:
            relevance=cosine(vector,self._vectors[key]);self._fast_diagnostics['exact_cosines']+=1
            hits.append(RetrievalHit(copy.deepcopy(self._units[key]),relevance+normalized+recency,
                relevance,recency,importance,normalized,age))
        hits.sort(key=lambda h:(-h.score,-h.unit.exit_at.timestamp(),h.unit.unit_id))
        return tuple(hits[:top_k])


def configure_native_kernel():
    """Opt into an equivalent implementation without editing frozen trainers.

    Full historical rank validation still executes on every restore. Only the
    candidate search and calendar/feedback indexes are accelerated; final
    scores, ties, namespace and information-time guards remain exact.
    """
    from . import moira_native
    if moira_native.CounterfactualMemoryBank is not FastCounterfactualMemoryBank:
        moira_native.CounterfactualMemoryBank=FastCounterfactualMemoryBank
        moira_native.session_age=IndexedExchangeSessionAge()
    return {'kernel':'exact-score-indexed-memory/1','full_historical_causal_validation':True,
            'snapshot_schema_unchanged':True,'final_scores_use_original_scalar_cosine':True}
