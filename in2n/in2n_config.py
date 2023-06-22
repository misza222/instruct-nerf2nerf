# Copyright 2022 The Nerfstudio Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Instruct-NeRF2NeRF configuration file.
"""

from nerfstudio.cameras.camera_optimizers import CameraOptimizerConfig
from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.plugins.types import MethodSpecification

from in2n.in2n_datamanager import InstructNeRF2NeRFDataManagerConfig
from in2n.in2n import InstructNeRF2NeRFModelConfig
from in2n.in2n_pipeline import InstructNeRF2NeRFPipelineConfig
from in2n.in2n_trainer import InstructNeRF2NeRFTrainerConfig

in2n_method = MethodSpecification(
    config=InstructNeRF2NeRFTrainerConfig(
        method_name="in2n",
        steps_per_eval_batch=1000,
        steps_per_eval_image=100,
        steps_per_save=250,
        max_num_iterations=15000,
        save_only_latest_checkpoint=True,
        mixed_precision=True,
        pipeline=InstructNeRF2NeRFPipelineConfig(
            datamanager=InstructNeRF2NeRFDataManagerConfig(
                dataparser=NerfstudioDataParserConfig(),
                train_num_rays_per_batch=16384,
                eval_num_rays_per_batch=4096,
                patch_size=32,
                camera_optimizer=CameraOptimizerConfig(
                    mode="SO3xR3", optimizer=AdamOptimizerConfig(lr=1e-30, eps=1e-8, weight_decay=1e-2)
                ),
            ),
            model=InstructNeRF2NeRFModelConfig(
                eval_num_rays_per_chunk=1 << 15,
                use_lpips=True,
            ),
            ip2p_use_full_precision=True
        ),
        optimizers={
            "proposal_networks": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": None,
            },
            "fields": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": None,
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
    ),
    description="Instruct-NeRF2NeRF primary method: uses LPIPS, IP2P at full precision",
)

in2n_method_big = MethodSpecification(
    config=InstructNeRF2NeRFTrainerConfig(
        method_name="in2n-big",
        steps_per_eval_batch=1000,
        steps_per_eval_image=100,
        steps_per_save=250,
        max_num_iterations=15000,
        save_only_latest_checkpoint=True,
        mixed_precision=True,
        pipeline=InstructNeRF2NeRFPipelineConfig(
            datamanager=InstructNeRF2NeRFDataManagerConfig(
                dataparser=NerfstudioDataParserConfig(),
                train_num_rays_per_batch=16384,
                eval_num_rays_per_batch=4096,
                patch_size=32,
                camera_optimizer=CameraOptimizerConfig(
                    mode="SO3xR3", optimizer=AdamOptimizerConfig(lr=1e-30, eps=1e-8, weight_decay=1e-2)
                ),
            ),
            model=InstructNeRF2NeRFModelConfig(
                eval_num_rays_per_chunk=1 << 15,
                use_lpips=True,

                num_nerf_samples_per_ray=128,
                num_proposal_samples_per_ray=(512, 256),
                hidden_dim=128,
                hidden_dim_color=128,
                appearance_embed_dim=128,
                base_res=32,
                max_res=4096,
                proposal_weights_anneal_max_num_iters=5000,
                log2_hashmap_size=21,
            ),
            ip2p_use_full_precision=True
        ),
        optimizers={
            "proposal_networks": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": None,
            },
            "fields": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": None,
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
    ),
    description="Instruct-NeRF2NeRF primary method: Big",
)

in2n_method_small = MethodSpecification(
    config=InstructNeRF2NeRFTrainerConfig(
        method_name="in2n-small",
        steps_per_eval_batch=1000,
        steps_per_eval_image=100,
        steps_per_save=250,
        max_num_iterations=30000,
        save_only_latest_checkpoint=True,
        mixed_precision=True,
        pipeline=InstructNeRF2NeRFPipelineConfig(
            datamanager=InstructNeRF2NeRFDataManagerConfig(
                dataparser=NerfstudioDataParserConfig(),
                train_num_rays_per_batch=16384,
                eval_num_rays_per_batch=4096,
                patch_size=32,
                camera_optimizer=CameraOptimizerConfig(
                    mode="SO3xR3", optimizer=AdamOptimizerConfig(lr=1e-30, eps=1e-8, weight_decay=1e-2)
                ),
            ),
            model=InstructNeRF2NeRFModelConfig(
                eval_num_rays_per_chunk=1 << 15,
                use_lpips=True,
            ),
            ip2p_use_full_precision=False,
        ),
        optimizers={
            "proposal_networks": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": None,
            },
            "fields": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": None,
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
    ),
    description="Instruct-NeRF2NeRF small method, uses LPIPs, IP2P at half precision",
)

in2n_method_tiny = MethodSpecification(
    config=InstructNeRF2NeRFTrainerConfig(
        method_name="in2n-tiny",
        steps_per_eval_batch=1000,
        steps_per_eval_image=100,
        steps_per_save=250,
        max_num_iterations=30000,
        save_only_latest_checkpoint=True,
        mixed_precision=True,
        pipeline=InstructNeRF2NeRFPipelineConfig(
            datamanager=InstructNeRF2NeRFDataManagerConfig(
                dataparser=NerfstudioDataParserConfig(),
                train_num_rays_per_batch=4096,
                eval_num_rays_per_batch=4096,
                patch_size=1,
                camera_optimizer=CameraOptimizerConfig(
                    mode="SO3xR3", optimizer=AdamOptimizerConfig(lr=1e-30, eps=1e-8, weight_decay=1e-2)
                ),
            ),
            model=InstructNeRF2NeRFModelConfig(
                eval_num_rays_per_chunk=1 << 15,
                use_lpips=False,
            ),
            ip2p_use_full_precision=False,
        ),
        optimizers={
            "proposal_networks": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": None,
            },
            "fields": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": None,
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
    ),
    description="Instruct-NeRF2NeRF tiny method, does not use LPIPs, IP2P at half precision",
)
