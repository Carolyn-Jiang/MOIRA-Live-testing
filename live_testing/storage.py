"""Atomic private records and process locks, outside tracked public assets."""
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path

def now(): return datetime.now(timezone.utc).isoformat()
def digest(value): return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False).encode()).hexdigest()
def file_hash(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def read_json(path, default=None):
    path=Path(path)
    return json.loads(path.read_text()) if path.exists() else default

def write_private(path, value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True,mode=0o700);path.parent.chmod(0o700)
    temporary=path.with_suffix(path.suffix+'.tmp')
    with os.fdopen(os.open(temporary,os.O_CREAT|os.O_TRUNC|os.O_WRONLY,0o600),'w') as handle:
        json.dump(value,handle,indent=2,ensure_ascii=False,allow_nan=False);handle.write('\n');handle.flush();os.fsync(handle.fileno())
    temporary.replace(path);path.chmod(0o600)

def append_private(path, value):
    line=json.dumps(value,ensure_ascii=False,allow_nan=False)+'\n'
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True,mode=0o700);path.parent.chmod(0o700)
    # The lock spans the full serialized line and fsync, including short writes
    # in the underlying text buffer, so independent API workers cannot interleave.
    with locked(path.with_name(path.name+'.lock')):
        with os.fdopen(os.open(path,os.O_CREAT|os.O_APPEND|os.O_WRONLY,0o600),'a') as handle:
            handle.write(line);handle.flush();os.fsync(handle.fileno())
        path.chmod(0o600)

@contextmanager
def locked(path, *, nonblocking=False):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
    with path.open('a+') as handle:
        path.chmod(0o600);fcntl.flock(handle,fcntl.LOCK_EX|(fcntl.LOCK_NB if nonblocking else 0))
        try: yield
        finally: fcntl.flock(handle,fcntl.LOCK_UN)
