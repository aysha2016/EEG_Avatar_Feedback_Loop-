import subprocess
import os

def generate_avatar(video_input="media/input_video.mp4"):
    output_path = "output/avatar_talk.mp4"

    model_path = "Wav2Lip/checkpoints/wav2lip_gan.pth"
    if not os.path.exists(model_path):
        print(" Wav2Lip model not found. Downloading...")
        subprocess.run(["python", "scripts/download_wav2lip_models.py"])

    subprocess.run([
        "python", "Wav2Lip/inference.py", "--checkpoint_path", model_path,
        "--input_path", video_input, "--output_path", output_path
    ])
    print(f" Avatar video generated at {output_path}")
