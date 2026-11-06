# 历史分支与源工作树的逐行差异

```diff
diff --git a/experiments/geotransformer.p2p_liver/config.py b/experiments/geotransformer.p2p_liver/config.py
index a026b7c..ab0bc41 100644
--- a/experiments/geotransformer.p2p_liver/config.py
+++ b/experiments/geotransformer.p2p_liver/config.py
@@ -31,10 +31,11 @@ _C.registration_dir = osp.join(_C.output_dir, 'registration')
 _DATA_ROOT = os.environ.get('P2P_DATA_ROOT', '/mnt/data3/yangx/P2P/Dataset')
 _C.data.train_root = osp.join(_DATA_ROOT, 'Deform_mesh_npz')
 _C.data.train_list = osp.join(_C.data.train_root, 'dict.json')
-_C.data.test_root = osp.join(_DATA_ROOT, 'Deform_mesh_npz_test', 'Test')
-_C.data.test_list = osp.join(_DATA_ROOT, 'Deform_mesh_npz_test', 'list.npz')
+_TEST_DATA_ROOT = os.environ.get('P2P_TEST_DATA_ROOT', _DATA_ROOT)
+_C.data.test_root = osp.join(_TEST_DATA_ROOT, 'Deform_mesh_npz_test', 'Test')
+_C.data.test_list = osp.join(_TEST_DATA_ROOT, 'Deform_mesh_npz_test', 'list.npz')
 _C.data.statistics = osp.join(
-    _DATA_ROOT,
+    _TEST_DATA_ROOT,
     'Deform_mesh_npz_test',
     'stat_svd.npz',
 )
@@ -75,8 +76,8 @@ def make_cfg(architecture='rtor_a3', registration_profile='legacy', dual_encoder
     cfg.model.dual_encoder = bool(dual_encoder)
     if interaction_profile not in ('legacy', 'cooperative', 'soft_overlap'):
         raise ValueError(f'Unknown interaction profile: {interaction_profile}')
-    if interaction_profile == 'cooperative' and (architecture != 'rtor_a3' or dual_encoder or registration_profile != 'legacy'):
-        raise ValueError('Cooperative profile uses single-encoder RTOR+A3 and the original LGR settings')
+    if interaction_profile == 'cooperative' and (dual_encoder or registration_profile != 'legacy'):
+        raise ValueError('Cooperative profile uses a single encoder and the original LGR settings')
     cfg.model.interaction_profile = interaction_profile
     cfg.coarse_matching.predicted_ratio_max = .25 if interaction_profile == 'cooperative' else 0.
     cfg.coarse_matching.exposure_start_epoch = 5
diff --git a/experiments/geotransformer.p2p_liver/test.py b/experiments/geotransformer.p2p_liver/test.py
index 128c27e..bb3edcf 100644
--- a/experiments/geotransformer.p2p_liver/test.py
+++ b/experiments/geotransformer.p2p_liver/test.py
@@ -16,6 +16,7 @@ sys.path.insert(0, _ROOT_DIR)

 import numpy as np
 import torch
+from scipy.spatial.transform import Rotation

 from geotransformer.engine import SingleTester
 from geotransformer.modules.ops.transformation import apply_transform
@@ -102,6 +103,7 @@ class Tester(SingleTester):
         protocol = self._checkpoint.get('metadata', {}).get('p2p_protocol', {})
         if protocol.get('interaction_profile', cfg.model.interaction_profile) != cfg.model.interaction_profile:
             raise ValueError('Checkpoint interaction profile differs; supply its --interaction_profile explicitly')
+        cfg.fine_refiner.geometry_weight = float(protocol.get('geometry_weight', cfg.fine_refiner.geometry_weight))
         if 'neighbor_limits' in protocol:
             cfg.data.neighbor_limits = protocol['neighbor_limits']
         noise_mm = None if self.args.noise == 'none' else int(self.args.noise)
@@ -197,6 +199,29 @@ class Tester(SingleTester):
     def after_test_step(self, iteration, data, output, result):
         index = int(data['index'])
         rms_tre = float(result['RMS_TRE_mm'].detach().cpu())
+        physical_scale = float(data['physical_scale'])
+        rre_deg = float(result['RRE'].detach().cpu())
+        rte_mm = float(result['RTE'].detach().cpu()) * physical_scale
+        estimated_transform = release_cuda(output['estimated_transform'])
+        oracle_transform = release_cuda(data['transform'])
+        source_error = (
+            apply_transform(output['src_points'], output['estimated_transform'])
+            - apply_transform(output['src_points'], data['transform'])
+        )
+        point_rmse_mm = float(
+            torch.sqrt(torch.mean(torch.sum(source_error.square(), dim=1))).detach().cpu()
+        ) * physical_scale
+        estimated_euler = Rotation.from_matrix(
+            np.asarray(estimated_transform[:3, :3], dtype=np.float64)
+        ).as_euler('xyz', degrees=True)
+        oracle_euler = Rotation.from_matrix(
+            np.asarray(oracle_transform[:3, :3], dtype=np.float64)
+        ).as_euler('xyz', degrees=True)
+        rotation_delta = (estimated_euler - oracle_euler + 180.0) % 360.0 - 180.0
+        translation_delta_mm = (
+            np.asarray(estimated_transform[:3, 3], dtype=np.float64)
+            - np.asarray(oracle_transform[:3, 3], dtype=np.float64)
+        ) * physical_scale
         focused_ref = int(output['num_focused_ref_nodes'])
         focused_src = int(output['num_focused_src_nodes'])
         valid_ref = int(output['ref_node_masks'].sum())
@@ -209,9 +234,17 @@ class Tester(SingleTester):
                 'visibility': float(self.visibility[index]),
                 'deformation_mm': float(self.deformation[index]),
                 'rms_tre_mm': rms_tre,
+                'success_5mm': int(rms_tre < 5.0),
                 'success_20mm': int(rms_tre < 20.0),
-                'rre_deg': float(result['RRE'].detach().cpu()),
+                'rre_deg': rre_deg,
                 'rte': float(result['RTE'].detach().cpu()),
+                'rte_mm': rte_mm,
+                'point_rmse_mm': point_rmse_mm,
+                'rr_5deg_5mm': int(rre_deg < 5.0 and rte_mm < 5.0),
+                'r_rmse_deg': float(np.sqrt(np.mean(np.square(rotation_delta)))),
+                'r_mae_deg': float(np.mean(np.abs(rotation_delta))),
+                't_rmse_mm': float(np.sqrt(np.mean(np.square(translation_delta_mm)))),
+                't_mae_mm': float(np.mean(np.abs(translation_delta_mm))),
                 'pir': float(result['PIR']),
                 'ir': float(result['IR']),
                 'mean_point_displacement': float(result['RMSE']),
@@ -268,6 +301,18 @@ class Tester(SingleTester):
                 ),
             }
         )
+        strict_metrics = {
+            'RMS_TRE_mm': float(values.mean()),
+            'SR_5_pct': 100.0 * float(np.mean([row['success_5mm'] for row in self.records])),
+            'RRE_deg': float(np.mean([row['rre_deg'] for row in self.records])),
+            'RTE_mm': float(np.mean([row['rte_mm'] for row in self.records])),
+            'RMSE_mm': float(np.mean([row['point_rmse_mm'] for row in self.records])),
+            'RR_pct': 100.0 * float(np.mean([row['rr_5deg_5mm'] for row in self.records])),
+            'r_RMSE_deg': float(np.mean([row['r_rmse_deg'] for row in self.records])),
+            'r_MAE_deg': float(np.mean([row['r_mae_deg'] for row in self.records])),
+            't_RMSE_mm': float(np.mean([row['t_rmse_mm'] for row in self.records])),
+            't_MAE_mm': float(np.mean([row['t_mae_mm'] for row in self.records])),
+        }
         payload = {
             'metric_schema_version': 3,
             'metric_note': 'RR uses mean normalized source displacement < eval.rmse_threshold; SR@20mm uses volumetric RMS-TRE. Historical GeoTransformer RMSE is mean point displacement, not root mean square.',
@@ -280,6 +325,10 @@ class Tester(SingleTester):
             'interaction_profile': self.cfg.model.interaction_profile,
             'fine_matching': dict(self.cfg.fine_matching),
             'noise_mm': None if self.args.noise == 'none' else int(self.args.noise),
+            'dataset_root': (self.cfg.data.in_vitro_root if self.args.dataset == 'in_vitro'
+                             else self.cfg.data.test_root),
+            'strict_metric_note': 'RTE and point RMSE are in millimetres; RR requires RRE < 5 degrees and RTE < 5 mm. Euler component errors use wrapped XYZ angles. The reference transform is the marker-fitted rigid oracle.',
+            'strict_metrics': strict_metrics,
             'rtor': {
                 'num_neighbors': int(self.cfg.topology_overlap.num_neighbors),
                 'score_floor': float(self.cfg.topology_overlap.score_floor),
@@ -306,6 +355,7 @@ class Tester(SingleTester):
             f'RTOR metrics: RMS-TRE={values.mean():.4f}+/-{values.std():.4f} mm, '
             f'SR@20mm={(values < 20).mean():.4f}'
         )
+        self.logger.critical(f'Strict metrics: {strict_metrics}')
         self.logger.info(f'Results written to {output_path} and {csv_path}')


diff --git a/experiments/geotransformer.p2p_liver/trainval.py b/experiments/geotransformer.p2p_liver/trainval.py
index 015c9e0..324a388 100644
--- a/experiments/geotransformer.p2p_liver/trainval.py
+++ b/experiments/geotransformer.p2p_liver/trainval.py
@@ -45,6 +45,7 @@ class Trainer(EpochBasedTrainer):
                         dual_encoder=cfg.model.dual_encoder,
                         registration_profile=cfg.model.registration_profile,
                         interaction_profile=cfg.model.interaction_profile,
+                        geometry_weight=float(cfg.fine_refiner.geometry_weight),
                         neighbor_limits=[int(x) for x in limits], seed=int(cfg.seed),
                         validation='disjoint deformation groups')
         self.save_state('p2p_protocol', protocol)
@@ -69,9 +70,19 @@ class Trainer(EpochBasedTrainer):
         state_dict = torch.load(snapshot, map_location=torch.device("cpu"))
         model_dict = state_dict["model"]
         if not self.args.warm_start:
-            saved_profile = state_dict.get('metadata', {}).get('p2p_protocol', {}).get('interaction_profile')
+            saved_protocol = state_dict.get('metadata', {}).get('p2p_protocol', {})
+            saved_profile = saved_protocol.get('interaction_profile')
             if saved_profile is not None and saved_profile != self.cfg.model.interaction_profile:
                 raise ValueError('Resume must use the checkpoint interaction profile; use --warm_start for an intentional new run')
+            saved_architecture = saved_protocol.get('architecture')
+            if saved_architecture is not None and saved_architecture != self.cfg.ablation.architecture:
+                raise ValueError('Resume must use the checkpoint architecture; use --warm_start for an intentional new run')
+            saved_geometry_weight = saved_protocol.get('geometry_weight')
+            if saved_geometry_weight is not None and float(saved_geometry_weight) != float(self.cfg.fine_refiner.geometry_weight):
+                raise ValueError('Resume must use the checkpoint geometry weight; use --warm_start for an intentional new run')
+            saved_seed = saved_protocol.get('seed')
+            if saved_seed is not None and int(saved_seed) != int(self.cfg.seed):
+                raise ValueError('Resume must use the checkpoint seed; use --warm_start for an intentional new run')
             self.best_validation_error = state_dict.get('metadata', {}).get('best_validation_error', float('inf'))
             strict_dict = (OrderedDict(('module.' + k, v) for k, v in model_dict.items())
                            if fix_prefix and self.distributed else model_dict)
@@ -150,6 +161,8 @@ def main():
     parser.add_argument('--dual_encoder', action='store_true')
     parser.add_argument('--interaction_profile', choices=['legacy', 'cooperative', 'soft_overlap'], default='legacy')
     parser.add_argument('--lr', type=float, default=None)
+    parser.add_argument('--geometry_weight', type=float, default=None)
+    parser.add_argument('--seed', type=int, default=None)
     parser.add_argument("--train_limit", type=int, default=0)
     parser.add_argument("--validation_size", type=int, default=None)
     parser.add_argument("--max_epoch", type=int, default=None)
@@ -162,6 +175,14 @@ def main():
     if known.train_limit > 0:
         os.environ["P2P_TRAIN_LIMIT"] = str(known.train_limit)
     cfg = make_cfg(known.architecture, known.registration_profile, known.dual_encoder, known.interaction_profile)
+    if known.seed is not None:
+        if known.seed < 0:
+            parser.error('--seed must be nonnegative')
+        cfg.seed = known.seed
+    if known.geometry_weight is not None:
+        if known.geometry_weight < 0:
+            parser.error('--geometry_weight must be nonnegative')
+        cfg.fine_refiner.geometry_weight = known.geometry_weight
     if known.lr is not None:
         if known.lr <= 0:
             parser.error('--lr must be positive')
```
