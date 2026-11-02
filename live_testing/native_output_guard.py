"""Native Moira output schemas using real, project-budgeted API calls only.

Unlike recovery, this factory has no saved responses and never patches a clock.
It preserves Gateway/scoped interfaces for both live and training dispatch.
"""
from pathlib import Path

from .resume_initial_training import ReuseClock,gateway_class


def create_guarded_gateway(key,budget,audit_dir):
    """Drop-in ``Gateway(key,budget,audit_dir)`` with native enum constraints.

    Codex JSON calls and embeddings retain their normal Gateway implementation.
    Native Moira pair selection and direction invoke actual strict-schema calls;
    monitoring/reflection invoke actual native Chat Completions. Every request
    uses the same shared Budget. An empty response queue cannot replay anything.
    """
    audit_dir=Path(audit_dir)
    cls=gateway_class(ReuseClock([]),audit_dir/'native-output-guard.jsonl')
    return cls(key,budget,audit_dir)
