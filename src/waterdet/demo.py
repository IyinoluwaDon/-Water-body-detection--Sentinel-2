"""Reproducible offline demonstration. All imagery is artificial, not Lagos observations."""
import argparse
import json
from pathlib import Path

import numpy as np
import rasterio
import yaml

from .pipeline import main as pipeline_main
from .synthetic import demo_config, make_bands, make_world, write_stack


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out-dir', default='outputs/demo')
    args = ap.parse_args(argv)
    root = Path(args.out_dir)
    data = root / 'data'
    data.mkdir(parents=True, exist_ok=True)
    world = make_world()
    truth = {}
    for period in ('dry', 'wet'):
        bands = make_bands(world, period)
        truth[period] = bands['_truth']
        write_stack(data / f's2_{period}.tif', bands)
    cfg = demo_config()
    cfg['reference']['csv'] = str(root / 'not_labelled.csv')
    config = root / 'config.yaml'
    config.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding='utf-8')
    pipeline_main(['--config', str(config), '--data-dir', str(data), '--out-dir', str(root), '--presentation'])
    checks = {'data_kind': 'synthetic; not measured Lagos accuracy', 'periods': {}}
    for period in truth:
        with rasterio.open(root / 'maps' / f'water_{period}.tif') as src:
            mapped = src.read(1)
        valid = mapped != 255
        checks['periods'][period] = {'known_truth_agreement': float(np.mean(mapped[valid] == truth[period][valid])),
                                     'valid_pixels': int(valid.sum())}
    (root / 'known_truth_checks.json').write_text(json.dumps(checks, indent=2), encoding='utf-8')
    (root / 'README.md').write_text('# Synthetic demonstration\n\nAll rasters and figures in this folder use artificial data. '
                                  'Use them to explain tests, not as evidence of Lagos mapping accuracy.\n', encoding='utf-8')


if __name__ == '__main__':
    main()
