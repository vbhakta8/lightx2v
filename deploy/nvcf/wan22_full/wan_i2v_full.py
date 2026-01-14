"""
WAN 2.2 Full Model (Vanilla/Unquantized) PyTriton Server for NVCF
Uses LightX2V with on-demand model loading for 48GB VRAM GPUs

This runs the FULL 40-step model (not distilled 4-step) for maximum quality.
Models are loaded from disk on-demand to work within memory constraints.
"""

import os
import gc
import sys
import json
import time
import base64
import tempfile
import shutil
import logging
from datetime import datetime
from typing import Optional
from urllib.parse import urlparse

import requests

sys.path.append('.')
sys.path.append('/app/LightX2V')

from PIL import Image
import numpy as np
from pytriton.model_config import ModelConfig, Tensor
from pytriton.triton import Triton, TritonConfig
import torch


# ======== Inlined utilities (avoids importing wombo package with heavy deps) ========

def get_logger_for_file(name: str) -> logging.Logger:
    """Get a configured logger for a file/module."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        log_level = os.environ.get('WOMBO_LOG_LEVEL', 'DEBUG').upper()
        logger.setLevel(getattr(logging, log_level, logging.INFO))
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(getattr(logging, log_level, logging.INFO))
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False
    return logger


def uppercase_keys(input_dict):
    """Uppercase keys in a dictionary, for NVCF headers."""
    return {key.upper(): value for key, value in input_dict.items()}


def numpy_array_to_variable(numpy_array):
    """Convert received numpy values into actual python primitive values."""
    if not isinstance(numpy_array, np.ndarray):
        return numpy_array
    if numpy_array.size != 1:
        raise ValueError("Input array must have a single element")
    value = numpy_array.item()
    if numpy_array.dtype.kind in ("S", "O"):
        value = value.decode("utf-8")
    return value


# ======== End inlined utilities ========

logger = get_logger_for_file(__name__)
logger.info('Running WAN 2.2 Full Model (Vanilla/Unquantized) via LightX2V')


def truthy(s) -> bool:
    """Parse boolean-like env values"""
    return str(s).lower() in ('1', 'true', 'yes', 'y', 'on')


class WanI2VFullServer:
    def __init__(self):
        from lightx2v import LightX2VPipeline

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")

        # Log GPU info
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            logger.info(f"GPU: {gpu_name}, VRAM: {gpu_mem:.1f} GB")

        # Get configuration from environment
        model_path = os.environ.get('WAN_MODEL_PATH', '/app/models/Wan2.2-I2V-A14B')
        model_cls = os.environ.get('WAN_MODEL_CLS', 'wan2.2_moe')  # Full model, not distilled
        task = os.environ.get('WAN_TASK', 'i2v')

        logger.info(f"Initializing WAN 2.2 FULL pipeline: model_path={model_path}, model_cls={model_cls}, task={task}")

        # Initialize pipeline
        start_time = time.time()
        self.pipe = LightX2VPipeline(
            model_path=model_path,
            model_cls=model_cls,
            task=task,
        )

        # NO quantization - vanilla FP16/BF16
        if truthy(os.environ.get('WAN_DIT_QUANTIZED', 'false')):
            quant_scheme = os.environ.get('WAN_QUANT_SCHEME', 'fp8-q8f')
            logger.info(f"Enabling quantization: scheme={quant_scheme}")
            self.pipe.enable_quantize(
                dit_quantized=True,
                dit_quant_scheme=quant_scheme,
                text_encoder_quantized=truthy(os.environ.get('WAN_T5_QUANTIZED', 'false')),
            )
        else:
            logger.info("Running VANILLA (unquantized) model")

        # Memory management: offload T5 after encoding, load models on-demand
        text_encoder_offload = truthy(os.environ.get('WAN_TEXT_ENCODER_OFFLOAD', 'true'))
        vae_offload = truthy(os.environ.get('WAN_VAE_OFFLOAD', 'false'))
        cpu_offload = truthy(os.environ.get('WAN_CPU_OFFLOAD', 'false'))

        logger.info(f"Offload settings: text_encoder={text_encoder_offload}, vae={vae_offload}, cpu={cpu_offload}")

        self.pipe.enable_offload(
            cpu_offload=cpu_offload,
            text_encoder_offload=text_encoder_offload,
            vae_offload=vae_offload,
        )

        # Enable on-demand model loading (critical for 48GB VRAM with unquantized models)
        unload_modules = truthy(os.environ.get('WAN_UNLOAD_MODULES', 'true'))
        if unload_modules:
            logger.info("Enabling unload_modules: models will be loaded from disk on-demand")
            self.pipe.update({
                'unload_modules': True,
            })

        # Full model settings
        infer_steps = int(os.environ.get('WAN_INFER_STEPS', '40'))  # Full model = 40 steps
        enable_cfg = truthy(os.environ.get('WAN_ENABLE_CFG', 'true'))  # CFG required for full model
        guidance_scale = float(os.environ.get('WAN_GUIDANCE_SCALE', '5.0'))  # Default CFG strength
        boundary = float(os.environ.get('WAN_BOUNDARY', '0.900'))  # Switch at 90%

        # Resolution and frame settings
        default_height = int(os.environ.get('WAN_HEIGHT', '720'))
        default_width = int(os.environ.get('WAN_WIDTH', '1280'))
        default_num_frames = int(os.environ.get('WAN_NUM_FRAMES', '81'))
        default_fps = int(os.environ.get('WAN_FPS', '24'))

        logger.info(f"Creating generator: infer_steps={infer_steps}, enable_cfg={enable_cfg}, "
                    f"guidance_scale={guidance_scale}, boundary={boundary}, "
                    f"resolution={default_width}x{default_height}, frames={default_num_frames}, fps={default_fps}")

        self.pipe.create_generator(
            attn_mode="flash_attn3",  # Best for L40/A100
            infer_steps=infer_steps,
            height=default_height,
            width=default_width,
            num_frames=default_num_frames,
            guidance_scale=guidance_scale if enable_cfg else 1.0,  # CFG enabled by default for full model
            sample_shift=5.0,
            fps=default_fps,
            boundary=boundary,
        )

        load_time = time.time() - start_time
        logger.info(f"WAN 2.2 FULL pipeline initialized in {load_time:.2f}s")
        logger.info(f"NOTE: First inference will take longer due to model loading from disk")

    def inference_function(self, triton_requests):
        logger.info(f"Received {len(triton_requests)} inference requests")
        responses = []

        for req in triton_requests:
            request_start_time = time.time()
            request_parameters = uppercase_keys(req.parameters)
            asset_dir = request_parameters.get('NVCF-ASSET-DIR')
            large_output_dir = request_parameters.get('NVCF-LARGE-OUTPUT-DIR')

            # Parse inputs
            input_image_path = numpy_array_to_variable(req.data.get('input_image_path'))
            user_prompt = numpy_array_to_variable(req.data.get('user_prompt')) or ''
            negative_prompt = numpy_array_to_variable(req.data.get('negative_prompt')) or ''
            seed = numpy_array_to_variable(req.data.get('seed'))

            # Build absolute path for input image (supports URLs and local paths)
            temp_image_path = None
            if input_image_path and input_image_path.startswith(('http://', 'https://')):
                # Download image from URL
                try:
                    logger.info(f"Downloading image from URL: {input_image_path}")
                    response = requests.get(input_image_path, timeout=30)
                    response.raise_for_status()

                    # Determine extension from URL or content-type
                    parsed_url = urlparse(input_image_path)
                    ext = os.path.splitext(parsed_url.path)[1] or '.jpg'

                    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp_file:
                        tmp_file.write(response.content)
                        temp_image_path = tmp_file.name
                        full_image_path = temp_image_path
                    logger.info(f"Downloaded image to: {full_image_path}")
                except Exception as e:
                    logger.error(f"Failed to download image from URL: {e}")
                    responses.append({
                        'generated_video': np.array(b''),
                        'profile': np.array(json.dumps({'error': f'Failed to download image: {e}'})),
                    })
                    continue
            elif input_image_path and asset_dir:
                full_image_path = os.path.join(asset_dir, input_image_path)
            else:
                full_image_path = input_image_path

            if not full_image_path or not os.path.exists(full_image_path):
                logger.error(f"Input image not found: {full_image_path}")
                responses.append({
                    'generated_video': np.array(b''),
                    'profile': np.array(json.dumps({'error': 'Input image not found'})),
                })
                continue

            actual_seed = int(seed) if seed is not None else np.random.randint(0, 2**31)
            logger.info(f"Running WAN 2.2 FULL I2V inference: prompt='{user_prompt[:50]}...', seed={actual_seed}")
            logger.info(f"NOTE: Full model with 40 steps - expect ~10-15 minutes inference time")

            # Create temp output path
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_file:
                temp_output_path = tmp_file.name

            try:
                # Run inference
                inference_start_time = time.time()

                self.pipe.generate(
                    seed=actual_seed,
                    image_path=full_image_path,
                    prompt=user_prompt,
                    negative_prompt=negative_prompt,
                    save_result_path=temp_output_path,
                )

                inference_time = time.time() - inference_start_time
                logger.info(f"Video generation completed in {inference_time:.2f}s ({inference_time/60:.1f} min)")

                # Save video with timestamp to persistent output directory
                output_dir = os.environ.get('WAN_OUTPUT_DIR', '/app/outputs')
                os.makedirs(output_dir, exist_ok=True)
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                video_filename = f"wan_i2v_full_{timestamp}_seed{actual_seed}.mp4"
                final_output_path = os.path.join(output_dir, video_filename)
                shutil.copy(temp_output_path, final_output_path)
                os.chmod(final_output_path, 0o644)
                logger.info(f"Video saved to: {final_output_path}")

                # Also copy to NVCF large output dir if provided
                if large_output_dir:
                    nvcf_output_path = os.path.join(large_output_dir, 'video.mp4')
                    shutil.copy(temp_output_path, nvcf_output_path)
                    os.chmod(nvcf_output_path, 0o644)
                    logger.info(f"Video also saved to NVCF dir: {nvcf_output_path}")

                # Read video and encode as base64
                with open(temp_output_path, 'rb') as f:
                    video_bytes = f.read()
                video_base64 = base64.b64encode(video_bytes).decode('utf-8')
                logger.info(f"Video encoded as base64: {len(video_base64)} chars")

                # Build response profile
                synth_profile = json.dumps({
                    'model': 'wan2.2_moe_full',
                    'model_type': 'vanilla_unquantized',
                    'infer_steps': int(os.environ.get('WAN_INFER_STEPS', '40')),
                    'seed': actual_seed,
                    'timing': {
                        'inference_time_seconds': float(inference_time),
                        'inference_time_minutes': float(inference_time / 60),
                        'total_time': float(time.time() - request_start_time),
                    },
                })

                responses.append({
                    'generated_video': np.array(video_base64.encode('utf-8')),
                    'profile': np.array(synth_profile),
                })

            except Exception as e:
                logger.exception(f"Inference failed: {e}")
                responses.append({
                    'generated_video': np.array(b''),
                    'profile': np.array(json.dumps({'error': str(e)})),
                })
            finally:
                # Cleanup temp files
                if os.path.exists(temp_output_path):
                    try:
                        os.remove(temp_output_path)
                    except:
                        pass
                if temp_image_path and os.path.exists(temp_image_path):
                    try:
                        os.remove(temp_image_path)
                    except:
                        pass

                # Force garbage collection after each request
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            total_request_time = time.time() - request_start_time
            logger.info(f"Total request processing time: {total_request_time:.2f}s ({total_request_time/60:.1f} min)")

        logger.debug(f"Completed {len(responses)} inference requests")
        return responses

    def run(self):
        with Triton(
            config=TritonConfig(
                http_header_forward_pattern='NVCF-*',
                http_port=8000,
                grpc_port=8001,
                metrics_port=8002,
            )
        ) as triton:
            triton.bind(
                model_name='wombo-diffusion',
                infer_func=self.inference_function,
                inputs=[
                    Tensor(name='negative_prompt', dtype=object, shape=(1,), optional=True),
                    Tensor(name='input_image_path', dtype=object, shape=(1,), optional=True),
                    Tensor(name='user_prompt', dtype=object, shape=(1,), optional=True),
                    Tensor(name='seed', dtype=np.uint32, shape=(1,), optional=True),
                ],
                outputs=[
                    Tensor(name='generated_video', dtype=bytes, shape=(1,)),
                    Tensor(name='profile', dtype=np.bytes_, shape=(1,)),
                ],
                config=ModelConfig(batching=False),
            )
            logger.info('Serving WAN 2.2 FULL model (vanilla/unquantized)')
            logger.info('Expected inference time: ~10-15 minutes per video')
            triton.serve()


if __name__ == '__main__':
    server = WanI2VFullServer()
    server.run()
