"""Aggregate distinct completed episode results; reject duplicates and mixed models."""
import argparse
import json
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('directories',nargs='+',type=Path)
p.add_argument('--output',required=True,type=Path)
a=p.parse_args()
seen=set();identity=None;success=0;by_suite={}
for directory in a.directories:
 for path in sorted(directory.rglob('result.json')):
  r=json.loads(path.read_text())
  key=(r['suite'],r['task_index'],r['init_state_index'])
  if key in seen:raise ValueError(f'Duplicate episode: {key}')
  seen.add(key)
  protocol=(r['method'],r['checkpoint_sha256'],r['manifest_sha256'],r['k'])
  if identity is not None and protocol!=identity:raise ValueError('Mixed evaluation protocols')
  identity=protocol
  if r.get('error') is not None:raise ValueError(f'Unresolved error: {path}')
  success+=int(r['success'])
  values=by_suite.setdefault(r['suite'],{'successful':0,'total':0})
  values['successful']+=int(r['success']);values['total']+=1
if not seen:raise ValueError('No completed episodes found')
report={'method':identity[0],'checkpoint_sha256':identity[1],'manifest_sha256':identity[2],
        'k':identity[3],'successful':success,'total':len(seen),'success_rate_percent':100*success/len(seen),
        'complete_full_manifest':len(seen)==10030,'by_suite':by_suite}
a.output.parent.mkdir(parents=True,exist_ok=True)
a.output.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
