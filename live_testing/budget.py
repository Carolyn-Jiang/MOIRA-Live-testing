"""One project-wide, NY-month $100 guard; provider token usage is authoritative.

Reserve a conservative byte-BPE input bound and bounded maximum output BEFORE
each HTTP request, including retries and embeddings. Unknown/failed requests keep
their reservation. No API/account-wide billing claim is made by this local ledger.
Standard rates verified against official OpenAI documentation on 2026-11-02.
"""
from datetime import datetime, timezone
from decimal import Decimal
import json
import re
from pathlib import Path
import uuid
from .registry import NY
from .storage import locked, read_json, write_private, now, digest

PRICING_SOURCE='https://developers.openai.com/api/docs/pricing'
RATES={'gpt-5.3-codex':(1.75,.175,14.),'gpt-4.1-mini':(.40,.10,1.60),
       'text-embedding-3-small':(.02,.02,0.),'text-embedding-3-large':(.13,.13,0.),
       'gpt-6.1-sol':(2.,.10,10.)}
SOL_PRICING_SOURCE='https://developers.openai.com/api/docs/models/gpt-6.1-sol'
SOL_LONG_INPUT=272000

class BudgetExceeded(RuntimeError): pass
class PricingUnavailable(RuntimeError): pass

def aware(value):
    value=datetime.fromisoformat(value) if isinstance(value,str) else value
    if value.tzinfo is None: raise ValueError('Aware budget time required')
    return value

def model_rate(model):
    for base,rates in RATES.items():
        if model==base or model.startswith(base+'-20'): return rates
    raise PricingUnavailable('No verified price for requested model')

def _is_sol(model):
    return model=='gpt-6.1-sol' or model.startswith('gpt-6.1-sol-20')

def _partition_cost(model,incoming,cached,outgoing,write_30m=0,write_12h=0):
    if (not all(type(x) is int and x>=0 for x in (incoming,cached,outgoing,write_30m,write_12h))
            or cached+write_30m+write_12h>incoming):
        raise ValueError('Invalid actual provider token counts')
    a,c,b=map(lambda v:Decimal(str(v)),model_rate(model))
    if _is_sol(model) and incoming>SOL_LONG_INPUT:
        a*=2;c*=2;b*=Decimal('1.5')
    ordinary=incoming-cached-write_30m-write_12h
    return float((Decimal(ordinary)*a+Decimal(cached)*c+Decimal(write_30m)*a*Decimal('1.25')
        +Decimal(write_12h)*a*Decimal('2.5')+Decimal(outgoing)*b)/Decimal(1000000))

def cost(model, incoming, cached, outgoing):
    """Legacy three-way pricing, or Sol with explicitly zero cache writes.

    Completed Sol responses must use usage_cost so billed write subsets cannot
    disappear into ordinary input. Existing callers retain the four-arg API.
    """
    return _partition_cost(model,incoming,cached,outgoing)

def tokens(usage):
    incoming=usage.get('input_tokens',usage.get('prompt_tokens'))
    outgoing=usage.get('output_tokens',usage.get('completion_tokens',0))
    if incoming is None and 'total_tokens' in usage: incoming=usage['total_tokens']
    detail=usage.get('input_tokens_details') or usage.get('prompt_tokens_details') or {}
    cached=detail.get('cached_tokens',0)
    if incoming is None: raise ValueError('Provider usage missing; reservation retained')
    values=(incoming,cached,outgoing)
    if not all(type(x) is int and x>=0 for x in values) or cached>incoming: raise ValueError('Malformed provider usage')
    return values

def _usage_tokens(model,usage):
    measured=tokens(usage)
    if _is_sol(model):
        details=usage.get('input_tokens_details') or usage.get('prompt_tokens_details') or {}
        if (not isinstance(details,dict) or 'cached_tokens' not in details
                or not any(key in usage for key in ('input_tokens','prompt_tokens'))
                or not any(key in usage for key in ('output_tokens','completion_tokens'))):
            raise ValueError('Sol requires reported input/output and cached-read usage')
        if 'total_tokens' in usage and (type(usage['total_tokens']) is not int
                or usage['total_tokens']!=measured[0]+measured[2]):
            raise ValueError('Sol provider token total disagrees with actual input/output')
    return measured

def usage_cost(model,usage,*,cache_write_ttl=None):
    """Price actual response usage without inventing unreported Sol writes.

    Reads, 30-minute writes, 12-hour writes and ordinary input partition input.
    A request-level TTL is usable only when its breakpoints have one known TTL.
    These are standard-rate estimates, before any account-level incentives.
    """
    incoming,cached,outgoing=_usage_tokens(model,usage)
    if not _is_sol(model):return cost(model,incoming,cached,outgoing)
    details=usage.get('input_tokens_details') or usage.get('prompt_tokens_details') or {}
    writes=details.get('cache_write_tokens',usage.get('cache_write_tokens'))
    if 'cache_write_tokens' in details and 'cache_write_tokens' in usage and writes!=usage['cache_write_tokens']:
        raise ValueError('Conflicting cache-write totals')
    split=details.get('cache_write_tokens_details',usage.get('cache_write_tokens_details'))
    if ('cache_write_tokens_details' in details and 'cache_write_tokens_details' in usage
            and split!=usage['cache_write_tokens_details']):
        raise ValueError('Conflicting cache-write lifetimes')
    if split is not None:
        if not isinstance(split,dict) or not {'ttl_30m_tokens','ttl_12h_tokens'}<=set(split):
            raise ValueError('Both actual cache-write lifetime counts are required')
        short,long=split['ttl_30m_tokens'],split['ttl_12h_tokens']
        if not all(type(v) is int and v>=0 for v in (short,long)):
            raise ValueError('Invalid cache-write lifetime counts')
        if writes is not None and (type(writes) is not int or writes!=short+long):
            raise ValueError('Cache-write lifetime counts do not sum to the reported total')
    else:
        if type(writes) is not int or writes<0:
            raise ValueError('Actual cache-write usage is unavailable')
        if writes==0:short=long=0
        elif cache_write_ttl=='30m':short,long=writes,0
        elif cache_write_ttl=='12h':short,long=0,writes
        else:raise ValueError('Actual cache-write lifetime allocation is unavailable')
    return _partition_cost(model,incoming,cached,outgoing,short,long)

def _request_write_ttl(payload):
    """Return a proven single TTL; mixed explicit overrides stay unknown."""
    options=payload.get('prompt_cache_options') or {}
    if not isinstance(options,dict):return None
    default=options.get('ttl','30m')
    if default not in ('30m','12h'):return None
    ttls={default}
    def visit(value):
        if isinstance(value,dict):
            marker=value.get('prompt_cache_breakpoint')
            if isinstance(marker,dict):ttls.add(marker.get('ttl',default))
            for item in value.values():visit(item)
        elif isinstance(value,list):
            for item in value:visit(item)
    visit(payload)
    return default if ttls=={default} else None

def _reserve_cost(model,incoming,outgoing):
    if not _is_sol(model):return cost(model,incoming,0,outgoing)
    # Full 12h writes are the most expensive possible input partition. Applying
    # the long-context multiplier to the byte bound is deliberately conservative.
    return _partition_cost(model,incoming,0,outgoing,write_12h=incoming)

class Budget:
    def __init__(self, root, *, model_id='data', stage='inference', event_id=None, clock=None):
        self.root=Path(root);self.path=self.root/'api-budget.json';self.lock=self.root/'api-budget.lock'
        self.model_id=model_id;self.stage=stage;self.event_id=event_id
        self.clock=clock or (lambda:datetime.now(timezone.utc))

    def scoped(self,model_id,stage='inference',event_id=None):
        return Budget(self.root,model_id=model_id,stage=stage,event_id=event_id,clock=self.clock)

    def _state(self):
        state=read_json(self.path,{'schema':'moira.live-testing.budget/1','limit_usd':100.,'warning_usd':20.,
            'pricing_source':PRICING_SOURCE,'pricing_verified_at':'2026-11-02T15:45:00+00:00','calls':{},'warnings':{},'blocks':{}})
        if state['limit_usd']!=100. or state['warning_usd']!=20.: raise BudgetExceeded('Budget policy changed unexpectedly')
        state.setdefault('blocks',{})
        if not isinstance(state['blocks'],dict) or not isinstance(state['warnings'],dict):
            raise BudgetExceeded('Budget control metadata is invalid')
        return state

    def reserve(self,payload,phase):
        instant=aware(self.clock());month=instant.astimezone(NY).strftime('%Y-%m')
        model=payload.get('model','');model_rate(model)
        upper_input=len(json.dumps(payload,ensure_ascii=False).encode())+4096
        upper_output=payload.get('max_output_tokens') or payload.get('max_completion_tokens') or payload.get('max_tokens') or 0
        if not model.startswith('text-embedding-') and (type(upper_output) is not int or not 1<=upper_output<=4096):
            raise BudgetExceeded('Every model call must have a bounded output of 1–4096 tokens')
        upper=_reserve_cost(model,upper_input,upper_output)
        identity=uuid.uuid4().hex
        with locked(self.lock):
            state=self._state()
            verified=aware(state['pricing_verified_at'])
            if instant>verified and (instant-verified).total_seconds()>32*86400:
                raise PricingUnavailable('Official model rates require monthly verification before spending')
            if month in state['blocks']:
                raise BudgetExceeded('This New York month is persistently blocked; new API requests are prohibited')
            guard=sum(Decimal(str(r['guard_cost_usd'])) for r in state['calls'].values() if r['month']==month)
            if guard+Decimal(str(upper))>Decimal('100'):
                state['blocks'][month]={'created_at':instant.isoformat(),
                    'reason':'next_conservative_reservation_exceeds_monthly_limit',
                    'guard_total_usd':float(guard),'requested_upper_usd':upper,'limit_usd':100.}
                write_private(self.path,state)
                raise BudgetExceeded('Monthly $100 project API cap reached; month persistently blocked')
            state['calls'][identity]={'month':month,'model_id':self.model_id,'stage':self.stage,'event_id':self.event_id,
                'phase':phase,'requested_model':model,'started_at':instant.isoformat(),'status':'reserved',
                'reserved_upper_usd':upper,'guard_cost_usd':upper,'request_sha256':digest(payload)}
            if _is_sol(model):
                state['calls'][identity]['cache_write_ttl']=_request_write_ttl(payload)
            write_private(self.path,state)
        return identity

    def finish(self,identity,*,response=None,error_type=None,status_code=None):
        with locked(self.lock):
            state=self._state();row=state['calls'][identity]
            row.update(finished_at=aware(self.clock()).isoformat(),status='failed' if error_type else 'completed')
            if error_type: row.update(error_type=error_type,status_code=status_code)
            elif response is not None:
                usage=response.get('usage') or {}
                if _is_sol(row['requested_model']):
                    # Even ambiguous cost allocation retains the exact provider
                    # evidence; the unresolved reservation is never refunded.
                    row.update(usage=usage,response_model=response.get('model'),response_id=response.get('id'))
                try:
                    inc,cached,out=_usage_tokens(row['requested_model'],usage)
                    if _is_sol(row['requested_model']):
                        row.update(input_tokens=inc,cached_input_tokens=cached,output_tokens=out,total_tokens=inc+out)
                        try:
                            actual=usage_cost(row['requested_model'],usage,cache_write_ttl=row.get('cache_write_ttl'))
                            row['cost_status']='complete'
                        except ValueError:
                            # Price ambiguity does not erase genuine provider token counts.
                            actual=None;row['cost_status']='unavailable'
                    else:
                        actual=usage_cost(row['requested_model'],usage)
                    if actual is not None:
                        if actual>row['reserved_upper_usd']+1e-9: raise BudgetExceeded('Provider cost exceeded reserved bound; stop for audit')
                        row.update(usage=usage,input_tokens=inc,cached_input_tokens=cached,output_tokens=out,
                            total_tokens=inc+out,actual_cost_usd=actual,guard_cost_usd=actual,
                            response_model=response.get('model'),response_id=response.get('id'))
                except ValueError:
                    row['status']='usage-unavailable'
            spent=sum(r.get('actual_cost_usd',0.) for r in state['calls'].values() if r['month']==row['month'])
            if spent>=20. and row['month'] not in state['warnings']:
                state['warnings'][row['month']]={'created_at':now(),'spent_usd':spent,'notification_pending':True}
            write_private(self.path,state)

    def ack_warning(self,month):
        """Acknowledge a notification only; never refund or unblock spending."""
        if not isinstance(month,str) or not re.fullmatch(r'\d{4}-(?:0[1-9]|1[0-2])',month):
            raise ValueError('New York calendar month must be YYYY-MM')
        with locked(self.lock):
            state=self._state();warning=state['warnings'].get(month)
            if warning is None:return {'month':month,'acknowledged':False,'notification_pending':False}
            warning['notification_pending']=False
            warning.setdefault('acknowledged_at',aware(self.clock()).isoformat())
            write_private(self.path,state)
            return {'month':month,'acknowledged':True,'acknowledged_at':warning['acknowledged_at'],
                'notification_pending':False}

    def confirm_pricing(self,evidence):
        """Record an actual official-source review; never alter caps/calls/blocks.

        The human/automation must first read the linked official pricing page.
        Changed rates require a reviewed code update, not an unchecked override.
        """
        if (not isinstance(evidence,dict) or evidence.get('source_url')!=PRICING_SOURCE
                or evidence.get('rates')!={key:list(value) for key,value in RATES.items()}):
            raise PricingUnavailable('Reviewed official rates must exactly match the implemented pricing table')
        verified=aware(evidence['verified_at']);instant=aware(self.clock())
        if verified>instant or (instant-verified).total_seconds()>86400:
            raise PricingUnavailable('Pricing verification must be an actual review within the previous day')
        with locked(self.lock):
            state=self._state()
            if verified<aware(state['pricing_verified_at']):
                raise PricingUnavailable('A pricing review cannot move verification backwards')
            state.setdefault('pricing_reviews',[]).append({**evidence,'recorded_at':instant.isoformat()})
            state['pricing_verified_at']=verified.isoformat();write_private(self.path,state)
        return {'status':'pricing_confirmed','verified_at':verified.isoformat(),'limit_usd':100.}

    def summary(self):
        month=aware(self.clock()).astimezone(NY).strftime('%Y-%m')
        with locked(self.lock):
            state=self._state();rows=[r for r in state['calls'].values() if r['month']==month]
        spent=sum(r.get('actual_cost_usd',0.) for r in rows);guard=sum(r['guard_cost_usd'] for r in rows)
        return {'month':month,'spent_usd':spent,'reserved_usd':max(0.,guard-spent),'guard_total_usd':guard,
            'warning_usd':20.,'limit_usd':100.,'status':'blocked' if month in state['blocks'] or guard>=100. else 'warning' if spent>=20. else 'ok',
            'warning_pending':bool(state['warnings'].get(month,{}).get('notification_pending',False)),
            'attempts':len(rows),'completed':sum(r['status']=='completed' for r in rows),
            'failed':sum(r['status']=='failed' for r in rows),'scope':'this_experiment',
            'pricing_verified_at':state['pricing_verified_at'],'pricing_source':state['pricing_source'],
            'prompt_tokens':sum(r.get('input_tokens',0) for r in rows),'completion_tokens':sum(r.get('output_tokens',0) for r in rows)}
