import torch
from torch.utils.data import Dataset
import os
import numpy as np

class BallroomDataset(Dataset):
    def __init__(self, spectrogram_dir, annotation_dir, sr=44100, trim = 3000, fuzziness = True):
        self.spectrogram_dir = spectrogram_dir
        self.annotation_dir = annotation_dir
        self.trim = trim
        self.fuzziness = fuzziness

        self.file_names = [f.replace('.npy','') for f in os.listdir(spectrogram_dir) if f.endswith('.npy')]
        self.sr = sr
        self.hop_size = int(np.floor(0.01 * sr))

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        file_name = self.file_names[idx]
        spectrogram = torch.tensor(np.load(os.path.join(self.spectrogram_dir, file_name)+'.npy'), dtype=torch.float32)

        with open(os.path.join(self.annotation_dir, file_name + '.beats'), 'r') as f:
            beat_in_sec = []
            beat_id = []
            for line in f:
                line = line.strip().replace('\t',' ').split(' ')
                beat_in_sec.append(float(line[0]))
                beat_id.append(int(line[1]))

            beat_times = np.array(beat_in_sec) * self.sr

            beat_vector = np.zeros((spectrogram.shape[-1]))

            # for i, beat_time in enumerate(beat_times):
            #     beat_vector[min(int(time/self.hop_size), beat_vector.shape[0]-1)] = 1.0

            # the for-loop below is borrowed from:
            # https://github.com/ben-hayes/beat-tracking-tcn/blob/d984e97d0d366072af2499a4ba224e66be7bf70a/beat_tracking_tcn/datasets/ballroom_dataset.py#L223C9-L231C80
            # because I was confused by the "widening the temporal activation region around the annotations" step in the paper... thanks Ben
            for time in beat_times:
                spec_frame = min(int(time / self.hop_size), beat_vector.shape[0] - 1)
                for n in range(-2, 3):
                    if 0 <= spec_frame + n < beat_vector.shape[0]:
                        if self.fuzziness:
                          beat_vector[spec_frame + n] = 1.0 if n == 0 else 0.5
                        else:
                          beat_vector[spec_frame + n] = 1.0 if n == 0 else 0.0

            beat_vector = torch.tensor(beat_vector, dtype=torch.float32)

        return spectrogram[:,:self.trim].T, beat_vector[:self.trim]