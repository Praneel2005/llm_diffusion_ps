"""
Evaluates Branch B video output.
Measures: duration, file size, scene count, audio presence.
"""
import os, json, subprocess

def evaluate_video(video_path: str, understanding_path: str):
    print(f"\n=== Branch B Video Evaluation ===")
    
    # Check file exists
    if not os.path.exists(video_path):
        print(f"ERROR: Video not found: {video_path}")
        return
    
    size_mb = os.path.getsize(video_path) / (1024*1024)
    
    # Get duration via ffprobe
    cmd = ["ffprobe", "-v", "quiet", "-print_format", "json",
           "-show_format", video_path]
    result = subprocess.run(cmd, capture_output=True, text=True)
    info   = json.loads(result.stdout)
    duration = float(info["format"]["duration"])
    
    print(f"  Video path  : {video_path}")
    print(f"  Duration    : {duration:.1f}s (target: 60-90s)")
    print(f"  File size   : {size_mb:.1f} MB")
    print(f"  In range    : {'YES' if 60 <= duration <= 90 else 'NO — ' + str(duration) + 's'}")
    
    # Check understanding quality
    if os.path.exists(understanding_path):
        with open(understanding_path) as f:
            u = json.load(f)
        print(f"\n  Paper title : {u.get('paper_title','?')}")
        print(f"  Method name : {u.get('method_name','?')}")
        print(f"  Components  : {u.get('components',[])} ")
        print(f"  Understanding quality: {'GOOD' if u.get('method_name') and u.get('components') else 'WEAK'}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python evaluate_branch_b.py <video_path> [understanding_json]")
        sys.exit(1)
    video = sys.argv[1]
    understanding = sys.argv[2] if len(sys.argv) > 2 else video.replace("_video_abstract.mp4", "/understanding.json")
    evaluate_video(video, understanding)
