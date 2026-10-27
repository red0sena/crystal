"""Restore the exact public raw snapshots used by the recorded fresh holdback.

python -m qwen_decision.v3.download_fresh_sources
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

from huggingface_hub import hf_hub_download


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,default=Path('wiki/project/g6-soup-holdback-results.json'))
    args=p.parse_args()
    data=json.loads(args.manifest.read_text())
    snapshots=data.get('manifest',data)['raw_dataset_snapshots']
    def download(item):
        repo,spec=item
        for name in spec['files']:
            hf_hub_download(repo,name,repo_type='dataset',revision=spec['revision'],
                local_dir=Path('data/raw/fresh')/repo.replace('/','__'))
        return repo
    with ThreadPoolExecutor(max_workers=4) as pool:
        for repo in pool.map(download,snapshots.items()):print('RESTORED',repo,flush=True)
    target=Path('runs/robust/fresh-raw-snapshots.json')
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(snapshots,indent=2))


if __name__=='__main__':main()
