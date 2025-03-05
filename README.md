# Beat Tracking Reproduction  

This is an implementation of: Davies, M., Böck, S., 2019. Temporal convolutional networks for musical audio beat tracking, in: 2019 27th European Signal Processing Conference (EUSIPCO). pp. 1–5. https://doi.org/10.23919/EUSIPCO.2019.8902578  


## Training  

See the [`inference.ipynb`](inference.ipynb) notebook. The audio data `BallroomData` (available from: http://mtg.upf.edu/ismir2004/contest/tempoContest/node5.html) and annotation data `BallroomAnnotations` (available from: https://github.com/CPJKU/BallroomAnnotations) are required.

## Inference

Please use the `beats = beatTracker(inputFile)` function in `inference.py`.

## Acknowledgement  

The annotation processing loop in the `dataset.py` file is taken from https://github.com/ben-hayes/beat-tracking-tcn/blob/d984e97d0d366072af2499a4ba224e66be7bf70a/beat_tracking_tcn/datasets/ballroom_dataset.py#L223C9-L231C80  

