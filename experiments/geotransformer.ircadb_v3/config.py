"""RTOR+A3 configuration for the 3Dircadb complete-to-partial rigid task."""

import os
import os.path as osp
import sys


ROOT_DIR = osp.realpath(osp.join(osp.dirname(__file__), '..', '..'))
if ROOT_DIR in sys.path:
    sys.path.remove(ROOT_DIR)
sys.path.insert(0, ROOT_DIR)

from geotransformer.config import make_rtor_a3_cfg
from geotransformer.utils.common import ensure_dir


DEFAULT_DATA_ROOT = (
    '/home/yangx/code/two_work/Liver_surface_data_process-main/'
    'liver_deform_output/dataset_v3'
)


def make_cfg(registration_profile='legacy', interaction_profile='legacy'):
    cfg = make_rtor_a3_cfg()
    if registration_profile not in ('legacy', 'tight', 'robust'):
        raise ValueError(f'Unknown registration profile: {registration_profile}')
    if interaction_profile not in ('legacy', 'cooperative', 'soft_overlap'):
        raise ValueError(f'Unknown interaction profile: {interaction_profile}')
    if interaction_profile == 'cooperative' and registration_profile != 'legacy':
        raise ValueError('Cooperative profile requires legacy registration')

    cfg.model.registration_profile = registration_profile
    cfg.model.interaction_profile = interaction_profile
    cfg.coarse_matching.predicted_ratio_max = .25 if interaction_profile == 'cooperative' else 0.
    cfg.coarse_matching.exposure_start_epoch = 5
    cfg.coarse_matching.exposure_end_epoch = 20
    if interaction_profile in ('cooperative', 'soft_overlap'):
        cfg.overlap_selection.enabled = False

    cfg.working_dir = osp.dirname(osp.realpath(__file__))
    cfg.root_dir = ROOT_DIR
    run_name = os.environ.get('IRCADB_V3_RUN_NAME', 'rtor_a3_legacy').strip()
    if not run_name or '/' in run_name or '\\' in run_name or run_name in ('.', '..'):
        raise ValueError('IRCADB_V3_RUN_NAME must be a simple directory name')
    cfg.exp_name = f'geotransformer.ircadb_v3.{run_name}'
    cfg.output_dir = osp.join(ROOT_DIR, 'output', cfg.exp_name)
    for key, dirname in [('snapshot_dir', 'snapshots'), ('log_dir', 'logs'),
                         ('event_dir', 'events'), ('feature_dir', 'features'),
                         ('registration_dir', 'registration')]:
        cfg[key] = osp.join(cfg.output_dir, dirname)
    for dirname in (cfg.output_dir, cfg.snapshot_dir, cfg.log_dir,
                    cfg.event_dir, cfg.feature_dir, cfg.registration_dir):
        ensure_dir(dirname)

    cfg.data.root = osp.realpath(os.environ.get('IRCADB_V3_DATA_ROOT', DEFAULT_DATA_ROOT))
    cfg.data.neighbor_calibration_samples = 64
    cfg.data.neighbor_limits = None
    cfg.data.validation_size = None  # use all 930 held-out model-17 pairs

    # All coordinates and transforms are in metres. No independent centering is
    # applied: source @ R.T + t is the synthetic rigid label in physical space.
    cfg.backbone.init_voxel_size = .004
    cfg.backbone.init_radius = cfg.backbone.base_radius * cfg.backbone.init_voxel_size
    cfg.backbone.init_sigma = cfg.backbone.base_sigma * cfg.backbone.init_voxel_size
    cfg.model.ground_truth_matching_radius = .014
    cfg.fine_loss.positive_radius = .010
    cfg.fine_matching.acceptance_radius = .014
    cfg.eval.acceptance_radius = .010
    cfg.eval.rmse_threshold = .020
    cfg.eval.rte_threshold = .020
    cfg.geotransformer.sigma_d = .04
    cfg.fine_refiner.geometry_sigma = .02
    cfg.overlap_selection.min_spread = .01
    if registration_profile == 'tight':
        cfg.fine_matching.acceptance_radius = .010
    elif registration_profile == 'robust':
        cfg.fine_matching.robust_refinement_radius = .014
    return cfg
