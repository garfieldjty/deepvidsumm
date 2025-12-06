"""
Training script for Wan video inbetweening (TI2V variant).

This script trains a Wan video model to generate video frames given
both starting and ending frames as conditioning inputs.

Features:
    - TensorBoard logging for loss and learning rate
    - Checkpoint saving every N steps (default: 200)
    - Automatic resume from latest checkpoint

Usage:
    accelerate launch examples/wanvideo/model_training/train_inbetween_ti2v.py \
        --dataset_base_path /path/to/videos \
        --dataset_metadata_path /path/to/metadata.csv \
        --model_id_with_origin_paths "Wan-AI/Wan2.2-TI2V-5B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.2-TI2V-5B:diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.2-TI2V-5B:Wan2.2_VAE.pth" \
        --output_path "./models/train/Wan2.2-TI2V-InBetween_lora" \
        --num_start_frames 8 \
        --num_end_frames 8 \
        --save_steps 200 \
        --tensorboard_log_dir "./logs/tensorboard"
"""

import torch
import os
import glob
import re
import json
import shutil
from einops import rearrange
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig
from diffsynth.models.wan_video_dit import sinusoidal_embedding_1d
from diffsynth.trainers.utils import DiffusionTrainingModule, ModelLogger, wan_parser
from diffsynth.trainers.unified_dataset import UnifiedDataset, LoadVideo, LoadAudio, ImageCropAndResize, ToAbsolutePath

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Enable cuDNN autotuning for faster convolutions (finds optimal algorithm)
torch.backends.cudnn.benchmark = True
# Use TF32 for faster matrix multiplications on Ampere+ GPUs (minimal precision loss)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
# Optimize memory allocator for less fragmentation
torch.cuda.set_per_process_memory_fraction(0.95)  # Use more GPU memory
# Enable flash attention if available (much faster attention)
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)


def model_fn_wan_ti2v_inbetween(
    dit,
    latents: torch.Tensor = None,
    timestep: torch.Tensor = None,
    context: torch.Tensor = None,
    ti2v_start_latents: torch.Tensor = None,
    ti2v_end_latents: torch.Tensor = None,
    use_gradient_checkpointing=False,
    use_gradient_checkpointing_offload=False,
    **kwargs
):
    """
    Model function for Wan video in-between task.
    This function conditions on both starting and ending frames for video interpolation.
    
    Args:
        dit: Wan DiT model
        latents: Full video latents [B, C, T, H, W]
        timestep: Timestep for diffusion
        context: Text context embeddings
        ti2v_start_latents: Starting frame latents for conditioning [B, C, T_start, H, W]
        ti2v_end_latents: Ending frame latents for conditioning [B, C, T_end, H, W]
        use_gradient_checkpointing: Whether to use gradient checkpointing
        use_gradient_checkpointing_offload: Whether to offload gradient checkpointing
        **kwargs: Additional models (motion_controller, vace, etc.) that are not used
    
    Returns:
        Model output (noise prediction)
    """
    # Replace start portion of latents with clean conditioned latents
    if ti2v_start_latents is not None:
        num_start = ti2v_start_latents.shape[2]
        latents[:, :, :num_start] = ti2v_start_latents
    
    # Replace end portion of latents with clean conditioned latents
    if ti2v_end_latents is not None:
        num_end = ti2v_end_latents.shape[2]
        latents[:, :, -num_end:] = ti2v_end_latents
    
    # Timestep embedding
    t = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, timestep).to(latents.dtype))
    t_mod = dit.time_projection(t).unflatten(1, (6, dit.dim))
    
    # Text embedding
    context = dit.text_embedding(context)
    
    x = latents
    
    # Patchify
    x = dit.patchify(x, None)
    f, h, w = x.shape[2:]
    x = rearrange(x, 'b c f h w -> b (f h w) c').contiguous()
    
    # Position embeddings - use contiguous() for better memory access patterns
    freqs = torch.cat([
        dit.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
        dit.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
        dit.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
    ], dim=-1).reshape(f * h * w, 1, -1).contiguous().to(x.device, non_blocking=True)
    
    # Run transformer blocks
    def create_custom_forward(module):
        def custom_forward(*inputs):
            return module(*inputs)
        return custom_forward
    
    for block in dit.blocks:
        if use_gradient_checkpointing:
            if use_gradient_checkpointing_offload:
                with torch.autograd.graph.save_on_cpu():
                    x = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        x, context, t_mod, freqs,
                        use_reentrant=False,
                    )
            else:
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    x, context, t_mod, freqs,
                    use_reentrant=False,
                )
        else:
            x = block(x, context, t_mod, freqs)
    
    # Head and unpatchify
    x = dit.head(x, t)
    x = dit.unpatchify(x, (f, h, w))
    
    output = -x
    output = output.to(latents.dtype)
    return output


class WanTI2VTrainingModuleInBetween(DiffusionTrainingModule):
    """Training module for Wan video inbetweening."""
    
    def __init__(
        self,
        model_paths=None,
        model_id_with_origin_paths=None,
        audio_processor_config=None,
        trainable_models=None,
        lora_base_model=None,
        lora_target_modules="q,k,v,o,ffn.0,ffn.2",
        lora_rank=32,
        lora_checkpoint=None,
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=False,
        extra_inputs=None,
        max_timestep_boundary=1.0,
        min_timestep_boundary=0.0,
        num_start_frames=1,
        num_end_frames=1,
    ):
        super().__init__()
        
        # Load models
        model_configs = self.parse_model_configs(model_paths, model_id_with_origin_paths, enable_fp8_training=False)
        if audio_processor_config is not None:
            audio_processor_config = ModelConfig(
                model_id=audio_processor_config.split(":")[0],
                origin_file_pattern=audio_processor_config.split(":")[1]
            )
        
        self.pipe = WanVideoPipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device="cpu",
            model_configs=model_configs,
            audio_processor_config=audio_processor_config
        )
        
        # Training mode
        self.switch_pipe_to_training_mode(
            self.pipe,
            trainable_models,
            lora_base_model,
            lora_target_modules,
            lora_rank,
            lora_checkpoint=lora_checkpoint,
            enable_fp8_training=False,
        )
        
        # Store other configs
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.extra_inputs = extra_inputs.split(",") if extra_inputs is not None else []
        self.max_timestep_boundary = max_timestep_boundary
        self.min_timestep_boundary = min_timestep_boundary
        self.num_start_frames = num_start_frames
        self.num_end_frames = num_end_frames
        
        # Override the model function to use the in-between version
        self.pipe.model_fn = model_fn_wan_ti2v_inbetween
        
        # Note: torch.compile() is disabled for training because it conflicts with
        # accelerate's model unwrapping during checkpoint saving. The error occurs
        # because torch.compile wraps the model and accelerate can't find '_orig_mod'.
        # If you need torch.compile, consider using it only for inference.
    
    def forward_preprocess(self, data):
        # CFG-sensitive parameters
        inputs_posi = {"prompt": data.get("prompt", "")}
        inputs_nega = {}
        
        # CFG-unsensitive parameters
        video_frames = data["video"]
        num_frames = len(video_frames)
        
        # Extract start and end frames for conditioning
        ti2v_start_video = video_frames[:self.num_start_frames]
        ti2v_end_video = video_frames[-self.num_end_frames:]
        
        inputs_shared = {
            # Assume you are using this pipeline for inference,
            # please fill in the input parameters.
            "input_video": video_frames,
            "height": video_frames[0].size[1],
            "width": video_frames[0].size[0],
            "num_frames": num_frames,
            # In-between task specific parameters
            "ti2v_start_video": ti2v_start_video,
            "ti2v_end_video": ti2v_end_video,
            # Please do not modify the following parameters
            # unless you clearly know what this will cause.
            "cfg_scale": 1,
            "tiled": False,
            "rand_device": self.pipe.device,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
            "cfg_merge": False,
            "vace_scale": 1,
            "max_timestep_boundary": self.max_timestep_boundary,
            "min_timestep_boundary": self.min_timestep_boundary,
        }
        
        # Extra inputs
        for extra_input in self.extra_inputs:
            if extra_input == "input_image":
                inputs_shared["input_image"] = video_frames[0]
            elif extra_input == "end_image":
                inputs_shared["end_image"] = video_frames[-1]
            elif extra_input == "reference_image" or extra_input == "vace_reference_image":
                inputs_shared[extra_input] = data[extra_input][0]
            else:
                inputs_shared[extra_input] = data[extra_input]
        
        # Pipeline units will automatically process the input parameters.
        for unit in self.pipe.units:
            inputs_shared, inputs_posi, inputs_nega = self.pipe.unit_runner(unit, self.pipe, inputs_shared, inputs_posi, inputs_nega)
        
        # Encode start and end videos to latents for conditioning
        # Use torch.no_grad() since VAE is frozen and we don't need gradients
        if self.pipe.vae is not None:
            self.pipe.load_models_to_device(["vae"])
            
            with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
                # Process starting frames
                if "ti2v_start_video" in inputs_shared:
                    ti2v_start_video = inputs_shared.pop("ti2v_start_video")
                    ti2v_start_video = self.pipe.preprocess_video(ti2v_start_video)
                    ti2v_start_latents = self.pipe.vae.encode(ti2v_start_video, device=self.pipe.device)
                    ti2v_start_latents = ti2v_start_latents.to(dtype=self.pipe.torch_dtype, device=self.pipe.device, non_blocking=True)
                    inputs_shared["ti2v_start_latents"] = ti2v_start_latents
                
                # Process ending frames
                if "ti2v_end_video" in inputs_shared:
                    ti2v_end_video = inputs_shared.pop("ti2v_end_video")
                    ti2v_end_video = self.pipe.preprocess_video(ti2v_end_video)
                    ti2v_end_latents = self.pipe.vae.encode(ti2v_end_video, device=self.pipe.device)
                    ti2v_end_latents = ti2v_end_latents.to(dtype=self.pipe.torch_dtype, device=self.pipe.device, non_blocking=True)
                    inputs_shared["ti2v_end_latents"] = ti2v_end_latents
        
        return {**inputs_shared, **inputs_posi}
    
    def forward(self, data, inputs=None):
        if inputs is None:
            inputs = self.forward_preprocess(data)
        models = {name: getattr(self.pipe, name) for name in self.pipe.in_iteration_models}
        loss = self.pipe.training_loss(**models, **inputs)
        return loss


def find_latest_checkpoint(output_path):
    """Find the latest checkpoint in the output directory."""
    checkpoint_pattern = os.path.join(output_path, "checkpoint-*")
    checkpoints = glob.glob(checkpoint_pattern)
    
    if not checkpoints:
        return None, 0, 0
    
    # Extract step numbers and find the latest
    latest_checkpoint = None
    latest_step = 0
    
    for ckpt in checkpoints:
        match = re.search(r'checkpoint-(\d+)', ckpt)
        if match:
            step = int(match.group(1))
            if step > latest_step:
                latest_step = step
                latest_checkpoint = ckpt
    
    # Load training state to get epoch info
    epoch = 0
    if latest_checkpoint:
        state_path = os.path.join(latest_checkpoint, "training_state.json")
        if os.path.exists(state_path):
            with open(state_path, "r") as f:
                state = json.load(f)
                epoch = state.get("epoch", 0)
                latest_step = state.get("global_step", latest_step)
    
    return latest_checkpoint, latest_step, epoch


def save_checkpoint(accelerator, model, optimizer, scheduler, output_path, global_step, epoch, model_logger, max_checkpoints=3):
    """Save a full checkpoint for resuming training."""
    accelerator.wait_for_everyone()
    
    checkpoint_dir = os.path.join(output_path, f"checkpoint-{global_step}")
    
    # First, save accelerator state (model, optimizer, scheduler, RNG states)
    # This must be called on all processes and handles distributed sync internally
    accelerator.save_state(checkpoint_dir)
    
    # Wait for accelerator save to complete on all processes
    accelerator.wait_for_everyone()
    
    # Now save additional files on main process only
    if accelerator.is_main_process:
        # Save model weights (LoRA weights) in a more portable format
        state_dict = accelerator.get_state_dict(model)
        state_dict = accelerator.unwrap_model(model).export_trainable_state_dict(
            state_dict, remove_prefix=model_logger.remove_prefix_in_ckpt
        )
        state_dict = model_logger.state_dict_converter(state_dict)
        model_path = os.path.join(checkpoint_dir, "model.safetensors")
        accelerator.save(state_dict, model_path, safe_serialization=True)
        
        # Save training state
        training_state = {
            "global_step": global_step,
            "epoch": epoch,
        }
        state_path = os.path.join(checkpoint_dir, "training_state.json")
        with open(state_path, "w") as f:
            json.dump(training_state, f, indent=2)
        
        print(f"Checkpoint saved at step {global_step} to {checkpoint_dir}")
    
    # Wait for all processes to finish saving before cleanup
    accelerator.wait_for_everyone()
    
    # Remove old checkpoints, keeping only the most recent max_checkpoints (only on main process)
    if accelerator.is_main_process:
        checkpoint_pattern = os.path.join(output_path, "checkpoint-*")
        checkpoints = glob.glob(checkpoint_pattern)
        
        if len(checkpoints) > max_checkpoints:
            # Sort checkpoints by step number
            checkpoint_steps = []
            for ckpt in checkpoints:
                match = re.search(r'checkpoint-(\d+)', ckpt)
                if match:
                    checkpoint_steps.append((int(match.group(1)), ckpt))
            
            checkpoint_steps.sort(key=lambda x: x[0])
            
            # Remove oldest checkpoints (exclude the current one being saved)
            checkpoints_to_remove = checkpoint_steps[:-max_checkpoints]
            for step, ckpt_path in checkpoints_to_remove:
                # Double-check we're not removing the current checkpoint
                if ckpt_path != checkpoint_dir:
                    shutil.rmtree(ckpt_path)
                    print(f"Removed old checkpoint: {ckpt_path}")
    
    return checkpoint_dir


def load_checkpoint(accelerator, checkpoint_path):
    """Load checkpoint for resuming training.
    
    This loads the accelerator state which includes:
    - Model weights (including LoRA weights)
    - Optimizer state
    - Scheduler state  
    - RNG states for reproducibility
    
    Note: The separate model.safetensors file saved in the checkpoint is for
    inference/sharing purposes and uses processed keys. The accelerator saves
    the raw model state which is what we load here.
    """
    if checkpoint_path and os.path.exists(checkpoint_path):
        # Validate checkpoint has required files before attempting to load
        # accelerator.save_state() creates model files with specific names
        model_files = glob.glob(os.path.join(checkpoint_path, "pytorch_model*.bin")) + \
                      glob.glob(os.path.join(checkpoint_path, "model*.safetensors")) + \
                      glob.glob(os.path.join(checkpoint_path, "model_*.safetensors"))
        
        # Also check for optimizer state which is always saved
        optimizer_files = glob.glob(os.path.join(checkpoint_path, "optimizer*.bin"))
        
        if not model_files and not optimizer_files:
            print(f"Warning: Checkpoint {checkpoint_path} appears to be incomplete or corrupted. Skipping resume.")
            return False
        
        print(f"Loading checkpoint from {checkpoint_path}")
        try:
            accelerator.load_state(checkpoint_path)
            return True
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            print("Starting training from scratch.")
            return False
    return False


def launch_training_task_with_logging(
    dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    learning_rate: float = 1e-5,
    weight_decay: float = 1e-2,
    num_workers: int = 8,
    save_steps: int = 200,
    num_epochs: int = 1,
    gradient_accumulation_steps: int = 1,
    find_unused_parameters: bool = False,
    tensorboard_log_dir: str = None,
    auto_resume: bool = True,
    args=None,
):
    """
    Launch training with TensorBoard logging and checkpoint resume support.
    
    Args:
        dataset: Training dataset
        model: Training module
        model_logger: Model logger for saving checkpoints
        learning_rate: Learning rate
        weight_decay: Weight decay for optimizer
        num_workers: Number of data loader workers
        save_steps: Save checkpoint every N steps (default: 200)
        num_epochs: Number of training epochs
        gradient_accumulation_steps: Gradient accumulation steps
        find_unused_parameters: Whether to find unused parameters in DDP
        tensorboard_log_dir: Directory for TensorBoard logs
        auto_resume: Whether to automatically resume from latest checkpoint
        args: Argument namespace (overrides other parameters if provided)
    """
    if args is not None:
        learning_rate = args.learning_rate
        weight_decay = args.weight_decay
        num_workers = args.dataset_num_workers
        save_steps = getattr(args, 'save_steps', 200) or 200
        num_epochs = args.num_epochs
        gradient_accumulation_steps = args.gradient_accumulation_steps
        find_unused_parameters = args.find_unused_parameters
        tensorboard_log_dir = getattr(args, 'tensorboard_log_dir', None)
        auto_resume = getattr(args, 'auto_resume', True)
    
    # Set default tensorboard log dir if not specified
    if tensorboard_log_dir is None:
        tensorboard_log_dir = os.path.join(model_logger.output_path, "tensorboard")
    
    # Initialize optimizer and scheduler
    # Use fused AdamW for better performance on CUDA
    optimizer = torch.optim.AdamW(
        model.trainable_modules(), 
        lr=learning_rate, 
        weight_decay=weight_decay,
        fused=torch.cuda.is_available()  # Use fused kernel for faster optimizer step
    )
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
    
    # Optimize DataLoader for faster data loading
    dataloader_kwargs = {
        "shuffle": True,
        "collate_fn": lambda x: x[0],
        "num_workers": num_workers,
    }
    # Add optimizations when using multiple workers
    if num_workers > 0:
        dataloader_kwargs.update({
            "pin_memory": True,  # Faster CPU->GPU transfer
            "prefetch_factor": 2,  # Prefetch batches in advance
            "persistent_workers": True,  # Keep workers alive between epochs
        })
    dataloader = torch.utils.data.DataLoader(dataset, **dataloader_kwargs)
    
    # Initialize accelerator with mixed precision for faster training
    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision="bf16",  # Use bfloat16 mixed precision for speed
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=find_unused_parameters)],
    )
    
    # Prepare for distributed training
    model, optimizer, dataloader, scheduler = accelerator.prepare(model, optimizer, dataloader, scheduler)
    
    # Initialize TensorBoard writer (only on main process)
    writer = None
    if accelerator.is_main_process:
        os.makedirs(tensorboard_log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=tensorboard_log_dir)
        print(f"TensorBoard logging to: {tensorboard_log_dir}")
    
    # Check for existing checkpoints and resume if available
    global_step = 0
    start_epoch = 0
    steps_per_epoch = len(dataloader)
    
    if auto_resume:
        latest_checkpoint, resumed_step, resumed_epoch = find_latest_checkpoint(model_logger.output_path)
        if latest_checkpoint:
            if load_checkpoint(accelerator, latest_checkpoint):
                global_step = resumed_step
                # Calculate the correct epoch and steps to skip based on global_step
                # This is more reliable than using the saved epoch value
                start_epoch = global_step // steps_per_epoch
                print(f"Resumed from checkpoint: step={global_step}, epoch={start_epoch}")
    
    # Calculate steps to skip in current epoch if resuming mid-epoch
    steps_to_skip = global_step % steps_per_epoch if global_step > 0 else 0
    
    # Training loop
    total_steps = num_epochs * steps_per_epoch
    
    if accelerator.is_main_process:
        print(f"Starting training:")
        print(f"  - Total epochs: {num_epochs}")
        print(f"  - Steps per epoch: {steps_per_epoch}")
        print(f"  - Total steps: {total_steps}")
        print(f"  - Save every {save_steps} steps")
        print(f"  - Starting from epoch {start_epoch}, step {global_step}")
    
    for epoch_id in range(start_epoch, num_epochs):
        epoch_loss = 0.0
        epoch_steps = 0
        
        # Set dataloader seed for reproducibility on resume
        # This helps ensure similar data ordering when resuming
        if hasattr(dataloader, 'sampler') and hasattr(dataloader.sampler, 'set_epoch'):
            dataloader.sampler.set_epoch(epoch_id)
        
        # Create progress bar
        progress_bar = tqdm(
            enumerate(dataloader),
            total=steps_per_epoch,
            desc=f"Epoch {epoch_id + 1}/{num_epochs}",
            disable=not accelerator.is_main_process,
        )
        
        for batch_idx, data in progress_bar:
            # Skip steps if resuming mid-epoch (only for the first epoch after resume)
            if epoch_id == start_epoch and batch_idx < steps_to_skip:
                continue
            
            with accelerator.accumulate(model):
                # Use set_to_none=True for faster memory clearing
                optimizer.zero_grad(set_to_none=True)
                
                # Forward pass
                if hasattr(dataset, 'load_from_cache') and dataset.load_from_cache:
                    loss = model({}, inputs=data)
                else:
                    loss = model(data)
                
                # Backward pass
                accelerator.backward(loss)
                
                # Gradient clipping for stability (optional but recommended)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                scheduler.step()
                
                global_step += 1
                epoch_steps += 1
                
                # Get loss value - only call .item() once to minimize GPU sync
                loss_val = loss.detach().float().item()
                epoch_loss += loss_val
                
                # Update progress bar (less frequently to reduce overhead)
                if global_step % 5 == 0:
                    current_lr = scheduler.get_last_lr()[0]
                    progress_bar.set_postfix({
                        'loss': f'{loss_val:.4f}',
                        'avg_loss': f'{epoch_loss / epoch_steps:.4f}',
                        'lr': f'{current_lr:.2e}',
                        'step': global_step,
                    })
                
                # Log to TensorBoard less frequently (every 10 steps) to reduce I/O overhead
                if writer is not None and global_step % 10 == 0:
                    current_lr = scheduler.get_last_lr()[0]
                    writer.add_scalar('train/loss', loss_val, global_step)
                    writer.add_scalar('train/learning_rate', current_lr, global_step)
                    writer.add_scalar('train/epoch', epoch_id + batch_idx / steps_per_epoch, global_step)
                
                # Save checkpoint every save_steps
                if global_step % save_steps == 0:
                    save_checkpoint(
                        accelerator, model, optimizer, scheduler,
                        model_logger.output_path, global_step, epoch_id, model_logger
                    )
                    
                    # Also log average loss at checkpoint
                    if writer is not None:
                        writer.add_scalar('train/avg_loss_at_checkpoint', epoch_loss / epoch_steps, global_step)
        
        # End of epoch logging
        avg_epoch_loss = epoch_loss / max(epoch_steps, 1)
        if accelerator.is_main_process:
            print(f"Epoch {epoch_id + 1} completed. Average loss: {avg_epoch_loss:.4f}")
            if writer is not None:
                writer.add_scalar('train/epoch_avg_loss', avg_epoch_loss, epoch_id + 1)
        
        # Save checkpoint at end of each epoch (if not already saved)
        if global_step % save_steps != 0:
            save_checkpoint(
                accelerator, model, optimizer, scheduler,
                model_logger.output_path, global_step, epoch_id + 1, model_logger
            )
        
        # Reset skip counter after first epoch
        steps_to_skip = 0
    
    # Final save
    if global_step % save_steps != 0:
        save_checkpoint(
            accelerator, model, optimizer, scheduler,
            model_logger.output_path, global_step, num_epochs, model_logger
        )
    
    # Save final model
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        state_dict = accelerator.get_state_dict(model)
        state_dict = accelerator.unwrap_model(model).export_trainable_state_dict(
            state_dict, remove_prefix=model_logger.remove_prefix_in_ckpt
        )
        state_dict = model_logger.state_dict_converter(state_dict)
        os.makedirs(model_logger.output_path, exist_ok=True)
        final_path = os.path.join(model_logger.output_path, "final_model.safetensors")
        accelerator.save(state_dict, final_path, safe_serialization=True)
        print(f"Final model saved to: {final_path}")
    
    # Close TensorBoard writer
    if writer is not None:
        writer.close()
    
    print(f"Training completed! Total steps: {global_step}")


if __name__ == "__main__":
    parser = wan_parser()
    parser.add_argument("--num_start_frames", type=int, default=1, help="Number of starting frames to use as condition")
    parser.add_argument("--num_end_frames", type=int, default=1, help="Number of ending frames to use as condition")
    parser.add_argument("--tensorboard_log_dir", type=str, default=None, help="Directory for TensorBoard logs. Defaults to output_path/tensorboard")
    parser.add_argument("--auto_resume", action="store_true", default=True, help="Automatically resume from latest checkpoint")
    parser.add_argument("--no_auto_resume", action="store_false", dest="auto_resume", help="Disable automatic resume from checkpoint")
    args = parser.parse_args()
    
    # Set default save_steps if not provided
    if args.save_steps is None:
        args.save_steps = 200
    
    dataset = UnifiedDataset(
        base_path=args.dataset_base_path,
        metadata_path=args.dataset_metadata_path,
        repeat=args.dataset_repeat,
        data_file_keys=args.data_file_keys.split(","),
        main_data_operator=UnifiedDataset.default_video_operator(
            base_path=args.dataset_base_path,
            max_pixels=args.max_pixels,
            height=args.height,
            width=args.width,
            height_division_factor=16,
            width_division_factor=16,
            num_frames=args.num_frames,
            time_division_factor=4,
            time_division_remainder=1,
            start_frame_index=args.start_frame_index,
        ),
        special_operator_map={
            "animate_face_video": ToAbsolutePath(args.dataset_base_path) >> LoadVideo(args.num_frames, 4, 1, frame_processor=ImageCropAndResize(512, 512, None, 16, 16)),
            "input_audio": ToAbsolutePath(args.dataset_base_path) >> LoadAudio(sr=16000),
        }
    )
    model = WanTI2VTrainingModuleInBetween(
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        audio_processor_config=args.audio_processor_config,
        trainable_models=args.trainable_models,
        lora_base_model=args.lora_base_model,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        lora_checkpoint=args.lora_checkpoint,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        extra_inputs=args.extra_inputs,
        max_timestep_boundary=args.max_timestep_boundary,
        min_timestep_boundary=args.min_timestep_boundary,
        num_start_frames=args.num_start_frames,
        num_end_frames=args.num_end_frames,
    )
    model_logger = ModelLogger(
        args.output_path,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt
    )
    
    # Launch training with TensorBoard logging and checkpoint resume
    launch_training_task_with_logging(dataset, model, model_logger, args=args)
