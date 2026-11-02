"""Pure long-test pair-selection policy; never imports an old native runtime.

Initial selection is allowed at a real observation. Every subsequent ordered
pair change, including one accompanying Flat, requires a real NY 15:30 release.
No prices, file operations, model calls, clocks, or repairs occur in this module.
"""
from collections.abc import Mapping

from .calendar import DEFAULT_CALENDAR,aware
from .registry import TOP_K,TICKERS

MOIRA_MODELS=('moira',*TOP_K)


class PairLifecycleViolation(ValueError):pass


def _pair(action):
    if (not isinstance(action,Mapping) or action.get('stock1')not in TICKERS
            or action.get('stock2')not in TICKERS or action['stock1']==action['stock2']
            or action.get('action')not in ('LS','SL','F')):
        raise PairLifecycleViolation('A genuine eligible ordered pair action is required')
    return action['stock1'],action['stock2']


def validate_pair_lifecycle(model_id,decisions,*,calendar=DEFAULT_CALENDAR):
    """Validate existing accounting/native-shaped decisions without mutation."""
    if model_id not in MOIRA_MODELS or not isinstance(decisions,Mapping):
        raise PairLifecycleViolation('An independent registered Moira decision chain is required')
    actual_nodes=set(calendar.release_nodes);ordered=[];seen=set()
    for key,row in decisions.items():
        if not isinstance(row,Mapping):raise PairLifecycleViolation('Invalid accepted decision record')
        try:
            at=aware(row.get('release_at',row.get('timestamp',key)))
            if 'release_at'in row and 'timestamp'in row and aware(row['timestamp'])!=at:
                raise PairLifecycleViolation('Decision timestamps disagree')
        except (ValueError,TypeError,AttributeError)as error:
            raise PairLifecycleViolation('An actual timezone-aware release is required')from error
        if at not in actual_nodes or at in seen:
            raise PairLifecycleViolation('Unique real calendar releases are required')
        seen.add(at);ordered.append((at,_pair(row.get('action'))))
    ordered.sort(key=lambda item:item[0].timestamp());previous=None;changes=[]
    for at,pair in ordered:
        if previous is not None and pair!=previous:
            if(at.hour,at.minute)!=(15,30):
                raise PairLifecycleViolation(f'{model_id}: ordered pair changed outside a real NY 15:30 release ({at.isoformat()})')
            changes.append(at.isoformat())
        previous=pair
    return {'model_id':model_id,'checked_decisions':len(ordered),'pair_changes':len(changes),'change_releases':changes}


def validate_book_pair_lifecycle(book,*,calendar=DEFAULT_CALENDAR):
    if not isinstance(book,Mapping)or not isinstance(book.get('decisions'),Mapping):
        raise PairLifecycleViolation('A verified accounting decision map is required')
    return {model:validate_pair_lifecycle(model,book['decisions'][model],calendar=calendar)
        for model in MOIRA_MODELS if model in book['decisions']}


def validate_next_pair(model_id,decisions,release_at,action,*,calendar=DEFAULT_CALENDAR):
    at=aware(release_at);prospective=dict(decisions)
    prospective[at.isoformat()]={'release_at':at.isoformat(),'action':action}
    return validate_pair_lifecycle(model_id,prospective,calendar=calendar)


def validate_legacy_moira_seals(source_state,*,limit=2,calendar=DEFAULT_CALENDAR):
    """Preflight old originals before any new bootstrap/native file copy."""
    if type(limit)is not int or not 1<=limit<=2:
        raise PairLifecycleViolation('Only the first two explicitly authorized seals may be preflighted')
    try:
        rows={at.isoformat():{'release_at':at.isoformat(),
            'action':source_state['sealed'][at.strftime('%Y-%m-%d_%H:%M')]['Moira']['action']}
            for at in calendar.release_nodes[:limit]}
    except(KeyError,TypeError)as error:
        raise PairLifecycleViolation('Complete original Moira seals are required before copying')from error
    return validate_pair_lifecycle('moira',rows,calendar=calendar)
