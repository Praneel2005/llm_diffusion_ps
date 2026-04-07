import os
import json
import subprocess
from PIL import Image, ImageDraw

OUTPUT_DIR = "outputs/videos"
TEMP_DIR = "outputs/videos/temp"

def generate_script(abstract: str, introduction: str, conclusion: str) -> dict:
    print("\n[Branch B] Generating 5-scene script...")
    return {
        "scene_1": {"title": "Problem Statement", "narration": f"In this paper, we address the following problem. {abstract.split('. ')[0]}."},
        "scene_2": {"title": "Prior Work", "narration": f"Previous approaches have explored several directions. {introduction.split('. ')[0]}."},
        "scene_3": {"title": "Our Method", "narration": f"Our proposed approach works as follows. {abstract.split('. ')[1] if len(abstract.split('. ')) > 1 else 'We propose a novel method.'}"},
        "scene_4": {"title": "Results", "narration": f"Our experiments show promising results. {abstract.split('. ')[-2] if len(abstract.split('. ')) > 1 else 'We achieved state-of-the-art performance.'}"},
        "scene_5": {"title": "Conclusion", "narration": f"In conclusion, {conclusion.split('. ')[0]}. We hope this work advances the field."}
    }

def create_placeholder_image(scene_title: str, narration: str, scene_num: int, output_path: str):
    img = Image.new('RGB', (854, 480), color='white')
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, 854, 80], fill=['#4A90D9', '#7B68EE', '#2ECC71', '#E67E22', '#E74C3C'][(scene_num - 1) % 5])
    draw.text((427, 40), f"Scene {scene_num}: {scene_title}", fill='white', anchor='mm')
    
    y = 120
    for line in [narration[i:i+60] for i in range(0, len(narration), 60)][:8]:
        draw.text((427, y), line, fill='#333333', anchor='mm')
        y += 35
    img.save(output_path)
    return output_path

def generate_audio(narration: str, scene_num: int, output_dir: str) -> str:
    audio_path = os.path.join(output_dir, f"scene_{scene_num}_audio.wav")
    try:
        from TTS.api import TTS
        tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False)
        tts.tts_to_file(text=narration, file_path=audio_path)
    except Exception:
        subprocess.run(["ffmpeg", "-y", "-f", "lavfi", "-i", "anullsrc=r=22050:cl=mono", "-t", "10", audio_path], check=True, capture_output=True)
    return audio_path

def stitch_video(scene_data: list, output_path: str):
    print("\n[Branch B] Stitching video...")
    temp_clips = []
    for i, scene in enumerate(scene_data):
        clip_path = os.path.join(TEMP_DIR, f"clip_{i+1}.mp4")
        subprocess.run(["ffmpeg", "-y", "-loop", "1", "-i", scene["image_path"], "-i", scene["audio_path"], "-c:v", "libx264", "-tune", "stillimage", "-c:a", "aac", "-b:a", "192k", "-pix_fmt", "yuv420p", "-shortest", clip_path], check=True, capture_output=True)
        temp_clips.append(clip_path)
    
    concat_file = os.path.join(TEMP_DIR, "concat.txt")
    with open(concat_file, 'w') as f:
        for clip in temp_clips: f.write(f"file '{os.path.abspath(clip)}'\n")
        
    subprocess.run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", concat_file, "-c", "copy", output_path], check=True, capture_output=True)
    return output_path

def run_branch_b(abstract: str, introduction: str, conclusion: str, branch_a_figures: list = None, output_dir: str = OUTPUT_DIR) -> str:
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(TEMP_DIR, exist_ok=True)
    script = generate_script(abstract, introduction, conclusion)
    scene_data = []
    
    for i, (key, scene) in enumerate(script.items()):
        scene_num = i + 1
        image_path = branch_a_figures[i]["generated_path"] if branch_a_figures and i < len(branch_a_figures) else create_placeholder_image(scene["title"], scene["narration"], scene_num, os.path.join(TEMP_DIR, f"scene_{scene_num}_img.png"))
        audio_path = generate_audio(scene["narration"], scene_num, TEMP_DIR)
        scene_data.append({"title": scene["title"], "image_path": image_path, "audio_path": audio_path})
        
    return stitch_video(scene_data, os.path.join(output_dir, "video_abstract.mp4"))
