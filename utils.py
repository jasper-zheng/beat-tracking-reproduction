import numpy as np
import librosa
import os
import glob

def compute_spectrogram(audio_file, sr=44100, n_fft=2048, hop_length=441, n_mels=81):

    y, sr = librosa.load(audio_file, sr=sr, mono=True)
    S = np.abs(librosa.feature.melspectrogram(y=y, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, fmin=30, fmax=17000))
    return S

def compute_all_spectrograms(input_folder, output_folder):
    if not os.path.exists(output_folder):
        print(f"Creating output folder: {output_folder}")
        os.makedirs(output_folder)

    for i, file_name in enumerate(glob.glob(os.path.join(input_folder, '**/*'),recursive=True)):
        if file_name.endswith(('.wav')):
            # audio_path = os.path.join(input_folder, file_name)
            spectrogram = compute_spectrogram(file_name)

            if spectrogram is not None:
                save_path = os.path.join(output_folder, file_name.split('/')[-1].replace('.wav', '.npy').split('/')[-1])
                np.save(save_path, spectrogram)

                if i % 10 == 0:
                    print(f"Processed {i} files: {save_path}")
