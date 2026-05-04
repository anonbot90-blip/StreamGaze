"""
EgoGPT-7b-EgoIT Model Wrapper for StreamGaze Benchmark

Note: This is a simplified wrapper that uses EgoGPT for video-only inference.
Audio features are set to zero since StreamGaze doesn't have audio.
"""

from model.modelclass import Model
import torch
import torch.distributed as dist
import copy
import os
from decord import VideoReader, cpu
import numpy as np
from PIL import Image
import time
import sys
import warnings

# Add EgoGPT to path
egogpt_path = "/home/colligo/sensei-fs-link/main/StreamGaze/EgoLife/EgoGPT"
if egogpt_path not in sys.path:
    sys.path.insert(0, egogpt_path)

from egogpt.model.builder import load_pretrained_model
from egogpt.conversation import conv_templates
from egogpt.constants import IMAGE_TOKEN_INDEX, SPEECH_TOKEN_INDEX


def setup_distributed(rank=0, world_size=1):
    """Initialize distributed process group for EgoGPT"""
    # Check if already initialized (for parallel execution)
    if dist.is_initialized():
        print("🔧 Distributed group already initialized, skipping...")
        return
    
    # Use process ID for unique port to allow parallel execution
    port = 12355 + (os.getpid() % 1000)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    
    try:
        dist.init_process_group("gloo", rank=rank, world_size=world_size)
    except RuntimeError as e:
        if "Address already in use" in str(e):
            # Try alternative port
            port = 12355 + (os.getpid() % 1000) + 1000
            os.environ["MASTER_PORT"] = str(port)
            dist.init_process_group("gloo", rank=rank, world_size=world_size)
        else:
            raise


class EgoGPT(Model):
    def __init__(self):
        """Initialize EgoGPT-7b-EgoIT model"""
        print("=" * 60)
        print("🔧 Loading EgoGPT-7b-EgoIT...")
        print("=" * 60)
        
        try:
            warnings.filterwarnings("ignore")
            
            # Initialize distributed process group (required by EgoGPT)
            print("🔧 Initializing distributed process group...")
            setup_distributed(0, 1)
            
            # Load model
            model_name = "lmms-lab/EgoGPT-7b-EgoIT"
            self.tokenizer, self.model, self.max_length = load_pretrained_model(
                model_name,
                device_map="cuda"  # Explicit device
            )
            
            self.model.eval()
            self.processor = self.model.get_vision_tower().image_processor
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            
            print("✅ EgoGPT-7b-EgoIT loaded successfully!")
            print(f"📊 Max length: {self.max_length}")
            print(f"📊 Device: {self.device}")
            print("=" * 60)
            
        except Exception as e:
            print(f"❌ Error loading EgoGPT: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def extract_frames(self, video_path, start_time, end_time, max_frames=16):
        """Extract frames uniformly from video segment"""
        try:
            vr = VideoReader(video_path, ctx=cpu(0))
            fps = vr.get_avg_fps()
            
            start_frame = int(start_time * fps)
            end_frame = int(end_time * fps)
            total_frames = end_frame - start_frame
            
            if total_frames > max_frames:
                indices = np.linspace(start_frame, end_frame - 1, max_frames, dtype=int)
            else:
                indices = list(range(start_frame, end_frame))
            
            frames = vr.get_batch(indices).asnumpy()
            
            print(f"📹 Extracted {len(frames)} frames from {start_time:.1f}s to {end_time:.1f}s (fps={fps:.1f})")
            
            return frames
            
        except Exception as e:
            print(f"❌ Error extracting frames from {video_path}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def Run(self, file, inp, start_time, end_time, question_time, 
            omni=False, proactive=False, salience_map_path=None):
        """Run EgoGPT inference on video segment"""
        try:
            T_start = time.time()
            
            print(f"\n{'='*60}")
            print(f"📹 Processing: {file}")
            print(f"⏰ Time range: {start_time:.1f}s → {end_time:.1f}s")
            print(f"❓ Question: {inp[:100]}..." if len(inp) > 100 else f"❓ Question: {inp}")
            print(f"{'='*60}")
            
            # Extract frames
            video = self.extract_frames(file, start_time, end_time, max_frames=16)
            
            if video is None or len(video) == 0:
                print("❌ Failed to extract frames")
                return " ", 0
            
            # Convert frames to PIL Images (needed for size extraction)
            pil_frames = [Image.fromarray(frame) for frame in video]
            
            # Process video
            processed_video = self.processor.preprocess(pil_frames, return_tensors="pt")["pixel_values"]
            image_size = pil_frames[0].size  # (width, height)
            
            # Prepare image data (following EgoGPT inference.py format)
            image_tensor = [processed_video.half().to(self.device)]
            image_sizes = [image_size]
            
            # Create dummy speech (no audio in StreamGaze)
            speech = torch.zeros(3000, 128).half().to(self.device)
            speech_lengths = torch.LongTensor([3000])
            speech = torch.stack([speech])
            
            # Prepare conversation
            conv_template = "qwen_1_5"
            question = f"<image>\n<speech>\n\n{inp}"
            conv = copy.deepcopy(conv_templates[conv_template])
            conv.append_message(conv.roles[0], question)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            
            # Tokenize with special tokens (following EgoGPT inference.py)
            def split_text(text, markers):
                """Split text by markers"""
                result = []
                current = ""
                i = 0
                while i < len(text):
                    found = False
                    for marker in markers:
                        if text[i:i+len(marker)] == marker:
                            if current:
                                result.append(current)
                                current = ""
                            result.append(marker)
                            i += len(marker)
                            found = True
                            break
                    if not found:
                        current += text[i]
                        i += 1
                if current:
                    result.append(current)
                return result
            
            parts = split_text(prompt, ["<image>", "<speech>"])
            input_ids = []
            for part in parts:
                if part == "<image>":
                    input_ids.append(IMAGE_TOKEN_INDEX)
                elif part == "<speech>":
                    input_ids.append(SPEECH_TOKEN_INDEX)
                else:
                    input_ids.extend(self.tokenizer(part).input_ids)
            
            input_ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(self.device)
            
            # Generate
            print("🤖 Generating response...")
            with torch.inference_mode():
                output_ids = self.model.generate(
                    input_ids,
                    images=image_tensor,
                    image_sizes=image_sizes,
                    speech=speech,
                    speech_lengths=speech_lengths,
                    do_sample=False,
                    max_new_tokens=512,
                    use_cache=True,
                    modalities=["video"],
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            
            # Decode
            response = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
            
            # Remove prompt from response (the prompt is part of the output)
            # Extract only the assistant's response after the last message separator
            if conv.sep in response:
                parts = response.split(conv.sep)
                if len(parts) > 1:
                    response = parts[-1].strip()
            
            T_end = time.time()
            response_time = T_end - T_start
            
            print(f"💬 Response: {response}")
            print(f"⏱️  Response time: {response_time:.2f}s")
            print(f"{'='*60}\n")
            
            return response, response_time
            
        except Exception as e:
            print(f"❌ Error in EgoGPT inference: {e}")
            import traceback
            traceback.print_exc()
            return " ", 0
    
    def name(self):
        """Return model name"""
        return "EgoGPT-7b-EgoIT"
