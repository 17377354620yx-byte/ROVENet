"""Evaluate rigid R,t on held-out 3Dircadb models 1 and 3."""

import argparse
import csv
import json
import os
import os.path as osp
import sys
from collections import defaultdict

ROOT_DIR = osp.realpath(osp.join(osp.dirname(__file__), '..', '..'))
if ROOT_DIR in sys.path:
    sys.path.remove(ROOT_DIR)
sys.path.insert(0, ROOT_DIR)

import numpy as np
import torch
from geotransformer.engine import SingleTester
from geotransformer.modules.ops.transformation import apply_transform
from geotransformer.modules.registration.metrics import isotropic_transform_error
from geotransformer.utils.torch import release_cuda
from config import make_cfg
from dataset import test_data_loader
from model import create_model


def summarize(rows):
    tre = np.asarray([r['rms_tre_mm'] for r in rows])
    return dict(count=len(rows), mean_rms_tre_mm=float(tre.mean()),
                median_rms_tre_mm=float(np.median(tre)),
                std_rms_tre_mm=float(tre.std()),
                mean_oracle_rms_tre_mm=float(np.mean([r['oracle_rms_tre_mm'] for r in rows])),
                mean_rre_deg=float(np.mean([r['rre_deg'] for r in rows])),
                mean_rte_mm=float(np.mean([r['rte_mm'] for r in rows])),
                sr_5mm=float((tre < 5).mean()), sr_10mm=float((tre < 10).mean()),
                sr_20mm=float((tre < 20).mean()))


class Tester(SingleTester):
    def __init__(self, cfg, parser):
        super().__init__(cfg, parser=parser)
        self.checkpoint = torch.load(self.args.snapshot, map_location='cpu', weights_only=False)
        protocol = self.checkpoint.get('metadata', {}).get('ircadb_v3_protocol')
        if not protocol:
            raise ValueError('Checkpoint has no dataset_v3 protocol')
        if protocol.get('dual_encoder', False):
            raise ValueError('Dual-encoder checkpoints are not supported by fixed ROVENet')
        if protocol['dataset_root'] != cfg.data.root:
            raise ValueError('Checkpoint dataset root differs')
        if protocol.get('architecture') not in ('rtor_a3', 'ROVENet'):
            raise ValueError('Checkpoint is not a complete RTOR+A3 model')
        for key, value in [('interaction_profile', cfg.model.interaction_profile),
                           ('registration_profile', cfg.model.registration_profile)]:
            if protocol[key] != value:
                raise ValueError(f'Checkpoint {key}={protocol[key]} differs from requested {value}')
        cfg.data.neighbor_limits = protocol['neighbor_limits']
        cfg.fine_refiner.geometry_weight = float(protocol['geometry_weight'])
        loader, limits = test_data_loader(cfg)
        self.logger.info(f'Test loader: {len(loader)} pairs; neighbors {limits}')
        self.register_loader(loader)
        self.register_model(create_model(cfg).cuda())
        self.cfg = cfg
        self.records = []
        self.output_path = self.args.output or osp.join(cfg.output_dir, 'test_summary.json')

    def load_snapshot(self, snapshot):
        self.model.load_state_dict(self.checkpoint['model'], strict=True)
        del self.checkpoint
        self.logger.info(f'Strictly loaded {snapshot}')

    def test_step(self, iteration, data):
        return self.model(data)

    def eval_step(self, iteration, data, output):
        estimated = output['estimated_transform']
        markers = apply_transform(data['source_markers'], estimated)
        tre_mm = torch.linalg.norm(markers - data['target_markers'], dim=1) * 1000
        rms_tre = torch.sqrt(torch.mean(tre_mm.square()))
        rre, rte = isotropic_transform_error(data['transform'], estimated)
        return {'RMS_TRE_mm': rms_tre, 'RRE_deg': rre, 'RTE_mm': rte * 1000,
                'SR_5mm': (rms_tre < 5).float(),
                'SR_10mm': (rms_tre < 10).float(),
                'SR_20mm': (rms_tre < 20).float()}

    def after_test_step(self, iteration, data, output, result):
        self.records.append({
            'index': int(data['index']), 'model_id': int(data['model_id']),
            'sample': data['sample_name'], 'visibility': float(data['overlap']),
            'max_nonrigid_mm': float(data['deformation_mm']),
            'rms_tre_mm': float(result['RMS_TRE_mm']),
            'oracle_rms_tre_mm': float(torch.sqrt(torch.mean(data['rigid_marker_tre_mm'].square()))),
            'rre_deg': float(result['RRE_deg']), 'rte_mm': float(result['RTE_mm']),
            'success_5mm': int(result['SR_5mm']),
            'success_10mm': int(result['SR_10mm']),
            'success_20mm': int(result['SR_20mm']),
        })
        if self.args.save_predictions:
            np.savez_compressed(
                osp.join(self.cfg.registration_dir, f'{int(data["index"]):05d}.npz'),
                sample=data['sample_name'],
                estimated_transform=release_cuda(output['estimated_transform']),
            )

    def summary_string(self, iteration, data, output, result):
        return (f'{data["sample_name"]}: RMS-TRE {float(result["RMS_TRE_mm"]):.3f} mm, '
                f'RRE {float(result["RRE_deg"]):.2f} deg, '
                f'RTE {float(result["RTE_mm"]):.2f} mm')

    def after_test_epoch(self):
        if not self.records:
            raise RuntimeError('No test records')
        by_model = defaultdict(list)
        for row in self.records:
            by_model[row['model_id']].append(row)
        payload = {
            'protocol': '3Dircadb dataset_v3 patient-disjoint held-out test',
            'checkpoint': osp.abspath(self.args.snapshot),
            'unit': 'mm for TRE and RTE; degrees for RRE',
            'task': 'Estimate rigid R,t only; target retains nonrigid deformation',
            'overall': summarize(self.records),
            'by_model': {str(k): summarize(v) for k, v in sorted(by_model.items())},
        }
        output = osp.abspath(self.output_path)
        os.makedirs(osp.dirname(output), exist_ok=True)
        with open(output, 'w') as handle:
            json.dump(payload, handle, indent=2)
        with open(osp.splitext(output)[0] + '.csv', 'w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(self.records[0]))
            writer.writeheader()
            writer.writerows(self.records)
        self.logger.critical(f'Test RMS-TRE {payload["overall"]["mean_rms_tre_mm"]:.3f} mm; '
                             f'SR@20mm {payload["overall"]["sr_20mm"]:.3f}')
        self.logger.info(f'Saved {output}')


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--test_limit', type=int, default=0)
    parser.add_argument('--output', default=None)
    parser.add_argument('--save_predictions', action='store_true')
    parser.add_argument('--architecture', default=None, help=argparse.SUPPRESS)
    parser.add_argument('--registration_profile', choices=['legacy', 'tight', 'robust'], default='legacy')
    parser.add_argument('--dual_encoder', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--interaction_profile', choices=['legacy', 'cooperative', 'soft_overlap'], default='legacy')
    known, _ = parser.parse_known_args()
    if known.architecture is not None:
        parser.error('--architecture is retired: ROVENet always includes RTOR and A3')
    if known.dual_encoder:
        parser.error('--dual_encoder is retired: ROVENet uses one backbone')
    if known.test_limit > 0:
        os.environ['IRCADB_V3_TEST_LIMIT'] = str(known.test_limit)
    cfg = make_cfg(known.registration_profile, known.interaction_profile)
    Tester(cfg, parser).run()


if __name__ == '__main__':
    main()
