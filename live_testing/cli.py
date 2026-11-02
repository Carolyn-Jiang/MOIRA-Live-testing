"""Run local nodes only after their actual NY boundaries."""
import argparse
import json
from pathlib import Path
from .storage import locked

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--state-dir',required=True,type=Path)
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--due',action='store_true');group.add_argument('--backfill-initial',action='store_true')
    group.add_argument('--publish-status',action='store_true');parser.add_argument('--wait-until-due',action='store_true')
    group.add_argument('--budget-status',action='store_true')
    group.add_argument('--backfill-preparation',action='store_true')
    group.add_argument('--ack-budget-warning',metavar='YYYY-MM')
    group.add_argument('--confirm-pricing-file',type=Path)
    parser.add_argument('--event-kind',action='append',choices=['release','valuation','training','pair_reselection','cache'])
    args=parser.parse_args()
    if args.budget_status or args.ack_budget_warning or args.confirm_pricing_file:
        from .budget import Budget
        from .storage import read_json
        budget=Budget(args.state_dir)
        result=(budget.ack_warning(args.ack_budget_warning) if args.ack_budget_warning else
            budget.confirm_pricing(read_json(args.confirm_pricing_file)) if args.confirm_pricing_file else budget.summary())
        print(json.dumps(result,ensure_ascii=False),flush=True);return
    from .runner import Runtime,dispatch
    if args.due: result=dispatch(args.state_dir,wait_until_due=args.wait_until_due,event_kinds=args.event_kind)
    elif args.backfill_initial or args.backfill_preparation:
        from .calendar import DEFAULT_CALENDAR
        with locked(args.state_dir/'run.lock',nonblocking=True):
            runtime=Runtime(args.state_dir)
            nodes=DEFAULT_CALENDAR.release_nodes[:2]
            if args.backfill_preparation:
                from .calendar import aware
                from .storage import read_json
                permission=read_json(args.state_dir/'preparation-replay-authorization.json')
                if not permission or not permission.get('training_completion_cutoff'):
                    raise RuntimeError('Training must finish before preparation replay')
                cutoff=aware(permission['training_completion_cutoff'])
                nodes=[at for at in DEFAULT_CALENDAR.release_nodes if at.date().isoformat()==permission['date'] and at<=cutoff]
            result={'results':[runtime.release(at,replay=True,publish_site=False) for at in nodes],
                    'website':runtime.feed()}
    else:
        with locked(args.state_dir/'run.lock',nonblocking=True):result=Runtime(args.state_dir).feed()
    print(json.dumps(result,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
