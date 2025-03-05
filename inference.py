import argparse

import torch

from utils import compute_spectrogram
from model import BeatTrackingNet

from madmom.features.beats import DBNBeatTrackingProcessor


def beatTracker(inputFile, modelPath = 'epoch200_fuz.pt'):
    """
    Takes the path name of an audio file (inputFile), and returns a vector of beat times (beats) in seconds

    Args:
        inputFile (string): path name of an audio file

    Returns:
        beats (numpy array): Vector of beat times (in seconds).
    """
    spectrogram = compute_spectrogram(inputFile)
    spectrogram = torch.tensor(spectrogram, dtype=torch.float32).T.unsqueeze(0).unsqueeze(1)
    
    dbn_processor = DBNBeatTrackingProcessor(min_bpm=55, max_bpm=215, threshold=0.4, fps=100)
    model = BeatTrackingNet(input_dim=81, num_filters=16, kernel_size=5, num_layers=11)
    
    model.load_state_dict(torch.load(modelPath))
    
    with torch.no_grad():
        activation = model(spectrogram).squeeze(1).cpu().numpy().flatten()

    detected_beats = dbn_processor(activation)
    
    return detected_beats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Given the path of an audio file, and returns a vector of beat times in second")
    parser.add_argument("audio_path", type=str, help="Path to the input audio file")
    parser.add_argument("--model_path", type=str, default="epoch200_fuz.pt", help="Path to the trained model file")
    args = parser.parse_args()
    
    beats = beatTracker(args.audio_path, args.model_path)

    print("Predicted Beats (seconds):")
    print(beats)