"""Only the strict aggregate feed is committed to the public Pages repository."""
import hashlib
import json
from pathlib import Path
import subprocess
import time
from urllib.request import Request,urlopen
from .registry import GITHUB_REPO
from .storage import read_json,write_private,now

SITE='https://carolyn-jiang.github.io/MOIRA-Live-testing/'
FORBIDDEN={'action','actions','stock1','stock2','pair','prompt','payload','api_key','token','response_id','receipt_id','generated_at'}

def assert_public(value):
    if isinstance(value,dict):
        if FORBIDDEN.intersection(value):raise ValueError('Private field in public projection')
        for item in value.values():assert_public(item)
    elif isinstance(value,list):
        for item in value:assert_public(item)

def _git(repo,*args):return subprocess.check_output(['git','-C',str(repo),*args],text=True).strip()
def _gh(path):return json.loads(subprocess.check_output(['gh','api',path],text=True))

def publish_feed(feed,state_dir,*,verify=True):
    assert_public(feed);repo=Path(__file__).resolve().parents[1];root=Path(state_dir)
    text=json.dumps(feed,ensure_ascii=False,allow_nan=False,indent=2)+'\n';target=repo/'docs/state.json'
    # The frontend validator independently recomputes path metrics before publication.
    check='import {validateState} from "./docs/app.js";let s="";for await(const c of process.stdin)s+=c;validateState(JSON.parse(s));'
    subprocess.run(['node','--input-type=module','-e',check],input=text,text=True,cwd=repo,check=True,capture_output=True)
    previous=read_json(root/'website-publication.json')
    digest=hashlib.sha256(text.encode()).hexdigest()
    if previous and previous.get('state_sha256')==digest and previous.get('verified'):return previous
    target.write_text(text)
    if _git(repo,'status','--porcelain','--','docs/state.json'):
        subprocess.run(['git','-C',str(repo),'add','--','docs/state.json'],check=True,capture_output=True)
        subprocess.run(['git','-C',str(repo),'commit','-m','Update verified live-testing aggregates','--only','--','docs/state.json'],check=True,capture_output=True)
        subprocess.run(['git','-C',str(repo),'push','origin','main'],check=True,capture_output=True)
    commit=_git(repo,'rev-parse','HEAD')
    record={'commit':commit,'state_sha256':digest,'created_at':now(),'verified':False,'url':SITE}
    write_private(root/'website-publication.json',record)
    if verify:
        end=time.monotonic()+240
        while time.monotonic()<end:
            try:
                build=_gh('repos/'+GITHUB_REPO+'/pages/builds/latest')
                if build.get('status')=='built' and build.get('commit')==commit:
                    with urlopen(Request(SITE+'state.json?verify='+commit,headers={'Cache-Control':'no-cache'}),timeout=20) as response:
                        actual=response.read()
                    if hashlib.sha256(actual).hexdigest()==digest:
                        record.update(verified=True,verified_at=now());write_private(root/'website-publication.json',record);return record
            except Exception:pass
            time.sleep(10)
        raise RuntimeError('GitHub commit saved; exact deployed Pages verification remains pending')
    return record
