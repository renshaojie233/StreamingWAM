"""Write an isolated LIBERO-Plus path configuration without changing user defaults."""
import argparse
from pathlib import Path
import yaml
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--libero-plus',type=Path,required=True)
p.add_argument('--output',type=Path,default=Path('outputs/libero-config'))
a=p.parse_args()
root=a.libero_plus.resolve(strict=True)/'libero/libero'
paths={'benchmark_root':root,'bddl_files':root/'bddl_files','init_states':root/'init_files',
       'assets':root/'assets','datasets':root.parent/'datasets'}
for key in ('bddl_files','init_states','assets'):
    if not paths[key].is_dir():raise FileNotFoundError(f'Missing {key}: follow LIBERO-Plus asset installation first')
a.output.mkdir(parents=True,exist_ok=True)
(a.output/'config.yaml').write_text(yaml.safe_dump({k:str(v) for k,v in paths.items()}))
print(a.output.resolve())
