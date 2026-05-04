#!/usr/bin/env python3
"""
Quick test to verify EgoGPT is generating responses
"""
import sys
sys.path.insert(0, "src")

from model.EgoGPT import EgoGPT

print("=" * 80)
print("Testing EgoGPT Model")
print("=" * 80)

# Initialize model
model = EgoGPT()

# Test with a simple video
video_path = "dataset/videos/original_video/OP01-R01-PastaSalad.mp4"
question = "What objects do you see in this video?"
start_time = 0.0
end_time = 10.0

print(f"\nVideo: {video_path}")
print(f"Question: {question}")
print(f"Time: {start_time}s - {end_time}s\n")

# Run inference
response, time_taken = model.Run(
    video_path, 
    question, 
    start_time, 
    end_time, 
    question_time=5.0
)

print("\n" + "=" * 80)
print(f"RESPONSE: '{response}'")
print(f"LENGTH: {len(response.strip())} characters")
print(f"TIME: {time_taken:.2f}s")
print("=" * 80)

# Check if response is empty
if not response.strip():
    print("\n❌ ERROR: Model returned empty response!")
    sys.exit(1)
else:
    print("\n✅ SUCCESS: Model generated a response!")
    sys.exit(0)

