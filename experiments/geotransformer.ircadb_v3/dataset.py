"""Read dataset_v3 without altering its physical rigid labels or patient split."""

import json
import os
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset

from geotransformer.utils.data import (
    build_dataloader_stack_mode,
    calibrate_neighbors_stack_mode,
    registration_collate_fn_stack_mode,
)


class IrcadbV3Dataset(Dataset):
    def __init__(self, cfg, split):
        if split not in ('train', 'val', 'test'):
            raise ValueError(f'Unknown split: {split}')
        self.root = Path(cfg.data.root)
        self.split = split
        split_file = self.root / 'splits.json'
        manifest_file = self.root / 'manifest.jsonl'
        if not split_file.is_file() or not manifest_file.is_file():
            raise FileNotFoundError(f'Expected splits.json and manifest.jsonl under {self.root}')
        allowed_models = set(json.loads(split_file.read_text())['splits'][split])
        self.records = []
        for line in manifest_file.open():
            row = json.loads(line)
            if row['split'] == split:
                if row['model_id'] not in allowed_models:
                    raise ValueError(f'Patient split mismatch: {row}')
                self.records.append(row)
        expected = len(allowed_models) * 30 * 31
        if len(self.records) != expected:
            raise ValueError(f'{split}: expected {expected} pairs, found {len(self.records)}')
        limit_name = {'train': 'IRCADB_V3_TRAIN_LIMIT',
                      'val': 'IRCADB_V3_VAL_LIMIT',
                      'test': 'IRCADB_V3_TEST_LIMIT'}[split]
        limit = int(os.environ.get(limit_name, '0'))
        if limit > 0:
            self.records = self.records[:limit]
        if split == 'val' and cfg.data.validation_size is not None:
            self.records = self.records[:int(cfg.data.validation_size)]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        row = self.records[index]
        path = self.root / row['pair']
        with np.load(path, allow_pickle=False) as entry:
            source = np.ascontiguousarray(entry['source_xyz_m'], dtype=np.float32)
            target = np.ascontiguousarray(entry['target_xyz_m'], dtype=np.float32)
            rotation = np.asarray(entry['R'], dtype=np.float64)
            translation = np.asarray(entry['t_m'], dtype=np.float64)
        if source.shape != (4096, 3) or target.shape != (1024, 3):
            raise ValueError(f'Unexpected point cloud shape: {path}')
        if rotation.shape != (3, 3) or translation.shape != (3,):
            raise ValueError(f'Unexpected rigid label shape: {path}')
        transform = np.eye(4, dtype=np.float32)
        transform[:3, :3] = rotation
        transform[:3, 3] = translation
        sample = {
            'scene_name': f'3Dircadb_model_{row["model_id"]:02d}',
            'ref_frame': row['pair'],
            'src_frame': 'complete_preoperative_liver',
            'sample_name': row['pair'],
            'index': index,
            'model_id': row['model_id'],
            'overlap': np.float32(row['visibility']),
            'deformation_mm': np.float32(row['nonrigid_max_mm']),
            'ref_points': target,
            'src_points': source,
            'ref_feats': np.ones((len(target), 1), dtype=np.float32),
            'src_feats': np.ones((len(source), 1), dtype=np.float32),
            'transform': transform,
        }
        if self.split != 'train':
            with np.load(self.root / row['evaluation'], allow_pickle=False) as entry:
                sample['source_markers'] = np.ascontiguousarray(
                    entry['internal_preop_xyz_m'], dtype=np.float32)
                sample['target_markers'] = np.ascontiguousarray(
                    entry['internal_nonrigid_target_xyz_m'], dtype=np.float32)
                sample['rigid_marker_tre_mm'] = np.ascontiguousarray(
                    entry['internal_rigid_tre_mm'], dtype=np.float32)
        return sample


def _calibrate(cfg, dataset):
    if cfg.data.neighbor_limits is not None:
        return [int(value) for value in cfg.data.neighbor_limits]
    return calibrate_neighbors_stack_mode(
        dataset, registration_collate_fn_stack_mode,
        cfg.backbone.num_stages, cfg.backbone.init_voxel_size,
        cfg.backbone.init_radius,
        sample_threshold=int(cfg.data.neighbor_calibration_samples),
    )


def _loader(cfg, dataset, limits, train, distributed=False):
    section = cfg.train if train else cfg.test
    return build_dataloader_stack_mode(
        dataset, registration_collate_fn_stack_mode,
        cfg.backbone.num_stages, cfg.backbone.init_voxel_size,
        cfg.backbone.init_radius, limits,
        batch_size=section.batch_size, num_workers=section.num_workers,
        shuffle=train, drop_last=False, distributed=distributed,
    )


def train_valid_data_loader(cfg, distributed=False):
    train = IrcadbV3Dataset(cfg, 'train')
    limits = _calibrate(cfg, train)
    return (_loader(cfg, train, limits, True, distributed),
            _loader(cfg, IrcadbV3Dataset(cfg, 'val'), limits, False), limits)


def test_data_loader(cfg):
    test = IrcadbV3Dataset(cfg, 'test')
    limits = _calibrate(cfg, test)
    return _loader(cfg, test, limits, False), limits
