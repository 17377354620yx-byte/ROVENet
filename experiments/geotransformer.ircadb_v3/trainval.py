"""Train RTOR+A3 on 3Dircadb dataset_v3 with patient-disjoint validation."""

import argparse
import json
import os
import os.path as osp
import sys
import time
from collections import OrderedDict


_ROOT_DIR = osp.realpath(osp.join(osp.dirname(__file__), "..", ".."))
if _ROOT_DIR in sys.path:
    sys.path.remove(_ROOT_DIR)
sys.path.insert(0, _ROOT_DIR)

import torch
import torch.optim as optim

from geotransformer.engine import EpochBasedTrainer
from geotransformer.modules.ops.transformation import apply_transform

from config import make_cfg
from dataset import train_valid_data_loader
from loss import Evaluator, OverallLoss
from model import create_model
from rovenet.modules.cooperative_matching import predicted_ratio_at_epoch


class Trainer(EpochBasedTrainer):
    def __init__(self, cfg, parser=None):
        super().__init__(cfg, max_epoch=cfg.optim.max_epoch, parser=parser,
                         save_all_snapshots=False)
        self.logger.info('Architecture: ROVENet (RTOR+A3)')
        self.cfg = cfg
        self.best_validation_error = float('inf')
        start = time.time()
        train_loader, val_loader, limits = train_valid_data_loader(cfg, self.distributed)
        self.logger.info(f"Data loader created: {time.time() - start:.3f}s collapsed.")
        self.logger.info(f"Calibrate neighbors: {limits}.")
        protocol = dict(dataset_root=cfg.data.root,
                        dataset='3Dircadb dataset_v3',
                        coordinate_unit='m',
                        label='source @ R.T + t; nonrigid residual remains in target',
                        architecture='ROVENet',
                        registration_profile=cfg.model.registration_profile,
                        interaction_profile=cfg.model.interaction_profile,
                        geometry_weight=float(cfg.fine_refiner.geometry_weight),
                        neighbor_limits=[int(x) for x in limits], seed=int(cfg.seed),
                        validation='model 17 only; model-disjoint from train and test')
        self.save_state('ircadb_v3_protocol', protocol)
        with open(osp.join(cfg.output_dir, 'run_manifest.json'), 'w') as handle:
            json.dump(dict(protocol=protocol, command=sys.argv, config=cfg), handle, indent=2)
        self.register_loader(train_loader, val_loader)
        model = self.register_model(create_model(cfg).cuda())
        optimizer = optim.Adam(
            model.parameters(), lr=cfg.optim.lr, weight_decay=cfg.optim.weight_decay
        )
        self.register_optimizer(optimizer)
        self.register_scheduler(
            optim.lr_scheduler.StepLR(
                optimizer, cfg.optim.lr_decay_steps, gamma=cfg.optim.lr_decay
            )
        )
        self.loss_func = OverallLoss(cfg).cuda()
        self.evaluator = Evaluator(cfg).cuda()

    def load_snapshot(self, snapshot, fix_prefix=True):
        """Resume RTOR+A3 or warm-start only its model parameters."""
        state_dict = torch.load(snapshot, map_location=torch.device("cpu"))
        model_dict = state_dict["model"]
        saved_protocol = state_dict.get('metadata', {}).get('ircadb_v3_protocol', {})
        if saved_protocol.get('dual_encoder', False):
            raise ValueError('Dual-encoder checkpoints are not supported by fixed ROVENet')
        if not self.args.warm_start:
            if saved_protocol.get('dataset_root', self.cfg.data.root) != self.cfg.data.root:
                raise ValueError('Resume dataset root differs from checkpoint')
            saved_profile = saved_protocol.get('interaction_profile')
            if saved_profile is not None and saved_profile != self.cfg.model.interaction_profile:
                raise ValueError('Resume must use the checkpoint interaction profile; use --warm_start for an intentional new run')
            saved_architecture = saved_protocol.get('architecture')
            if saved_architecture is not None and saved_architecture not in ('rtor_a3', 'ROVENet'):
                raise ValueError('Checkpoint is not a complete RTOR+A3 model')
            saved_geometry_weight = saved_protocol.get('geometry_weight')
            if saved_geometry_weight is not None and float(saved_geometry_weight) != float(self.cfg.fine_refiner.geometry_weight):
                raise ValueError('Resume must use the checkpoint geometry weight; use --warm_start for an intentional new run')
            saved_seed = saved_protocol.get('seed')
            if saved_seed is not None and int(saved_seed) != int(self.cfg.seed):
                raise ValueError('Resume must use the checkpoint seed; use --warm_start for an intentional new run')
            self.best_validation_error = state_dict.get('metadata', {}).get('best_validation_error', float('inf'))
            strict_dict = (OrderedDict(('module.' + k, v) for k, v in model_dict.items())
                           if fix_prefix and self.distributed else model_dict)
            self.model.load_state_dict(strict_dict, strict=True)
            return super().load_snapshot(snapshot, fix_prefix=fix_prefix)

        self.logger.info(
            f'Warm-starting model weights only from "{snapshot}".'
        )
        if fix_prefix and self.distributed:
            model_dict = OrderedDict(
                [("module." + key, value) for key, value in model_dict.items()]
            )
        incompatibility = self.model.load_state_dict(model_dict, strict=False)
        allowed_prefixes = (
            'module.topology_overlap_refiner.',
            'module.fine_local_refiner.',
        ) if self.distributed else (
            'topology_overlap_refiner.',
            'fine_local_refiner.',
        )
        missing = list(incompatibility.missing_keys)
        required_missing = [
            key for key in missing if not key.startswith(allowed_prefixes)
        ]
        unexpected = list(incompatibility.unexpected_keys)
        if unexpected:
            raise RuntimeError(
                f"Unexpected keys in warm start: {sorted(unexpected)}"
            )
        if required_missing:
            raise RuntimeError(
                f"Required keys missing from warm start: {sorted(required_missing)}"
            )
        self.logger.info(
            f"Loaded weights; initialized {len(missing)} RTOR/A3 tensors. "
            "Epoch, optimizer and scheduler start fresh."
        )

    def _step(self, data):
        output = self.model(data)
        result = self.loss_func(output, data)
        result.update(self.evaluator(output, data))
        return output, result

    def train_step(self, epoch, iteration, data):
        model = self.model.module if self.distributed else self.model
        model.predicted_coarse_ratio = predicted_ratio_at_epoch(
            epoch, self.cfg.coarse_matching.exposure_start_epoch,
            self.cfg.coarse_matching.exposure_end_epoch,
            self.cfg.coarse_matching.predicted_ratio_max)
        return self._step(data)

    def val_step(self, epoch, iteration, data):
        output, result = self._step(data)
        aligned = apply_transform(data['source_markers'], output['estimated_transform'])
        distances_mm = torch.linalg.norm(aligned - data['target_markers'], dim=1) * 1000
        result['RMS_TRE_mm'] = torch.sqrt(torch.mean(distances_mm.square()))
        self.validation_errors.append(float(result['RMS_TRE_mm']))
        return output, result

    def before_val_epoch(self, epoch):
        self.validation_errors = []

    def after_val_epoch(self, epoch):
        if self.validation_errors:
            error = sum(self.validation_errors) / len(self.validation_errors)
            if error < self.best_validation_error:
                self.best_validation_error = error
                self.save_state('best_validation_error', error)
                self.save_snapshot('best.pth.tar')
                self.logger.info(f'Best model-17 validation RMS-TRE: {error:.6f} mm, epoch {epoch}')


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--architecture', default=None, help=argparse.SUPPRESS)
    parser.add_argument('--registration_profile', choices=['legacy', 'tight', 'robust'], default='legacy')
    parser.add_argument('--dual_encoder', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--interaction_profile', choices=['legacy', 'cooperative', 'soft_overlap'], default='legacy')
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--geometry_weight', type=float, default=None)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument("--train_limit", type=int, default=0)
    parser.add_argument("--validation_size", type=int, default=None)
    parser.add_argument("--max_epoch", type=int, default=None)
    parser.add_argument(
        "--warm_start",
        action="store_true",
        help="Load --snapshot model weights only; start epoch/optimizer fresh.",
    )
    known, _ = parser.parse_known_args()
    if known.architecture is not None:
        parser.error('--architecture is retired: ROVENet always includes RTOR and A3')
    if known.dual_encoder:
        parser.error('--dual_encoder is retired: ROVENet uses one backbone')
    if known.train_limit > 0:
        os.environ["IRCADB_V3_TRAIN_LIMIT"] = str(known.train_limit)
    cfg = make_cfg(known.registration_profile, known.interaction_profile)
    if known.seed is not None:
        if known.seed < 0:
            parser.error('--seed must be nonnegative')
        cfg.seed = known.seed
    if known.geometry_weight is not None:
        if known.geometry_weight < 0:
            parser.error('--geometry_weight must be nonnegative')
        cfg.fine_refiner.geometry_weight = known.geometry_weight
    if known.lr is not None:
        if known.lr <= 0:
            parser.error('--lr must be positive')
        cfg.optim.lr = known.lr
    if known.validation_size is not None:
        if known.validation_size <= 0:
            parser.error("--validation_size must be positive")
        cfg.data.validation_size = known.validation_size
    if known.max_epoch is not None:
        if known.max_epoch <= 0:
            parser.error("--max_epoch must be positive")
        cfg.optim.max_epoch = known.max_epoch
    Trainer(cfg, parser=parser).run()


if __name__ == "__main__":
    main()
