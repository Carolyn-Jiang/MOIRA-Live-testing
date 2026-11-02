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
       'text-embedding-3-small':(.02,.02,0.),'text-embedding-3-large':(.13,.13,0.)}

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

def cost(model, incoming, cached, outgoing):
    if not all(type(x) is int and x>=0 for x in (incoming,cached,outgoing)) or cached>incoming:
        raise ValueError('Invalid actual provider token counts')
    a,c,b=map(lambda v:Decimal(str(v)),model_rate(model))
    return float((Decimal(incoming-cached)*a+Decimal(cached)*c+Decimal(outgoing)*b)/Decimal(1000000))

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
        upper=cost(model,upper_input,0,upper_output)
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
            write_private(self.path,state)
        return identity

    def finish(self,identity,*,response=None,error_type=None,status_code=None):
        with locked(self.lock):
            state=self._state();row=state['calls'][identity]
            row.update(finished_at=aware(self.clock()).isoformat(),status='failed' if error_type else 'completed')
            if error_type: row.update(error_type=error_type,status_code=status_code)
            elif response is not None:
                usage=response.get('usage') or {}
                try:
                    inc,cached,out=tokens(usage);actual=cost(row['requested_model'],inc,cached,out)
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
