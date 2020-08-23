# End-to-End Deep Learning
- Example: Speech recognition
    - First extract features of the audio with MFCC
    - Then apply ML algorithm to extract the phonemes
    - Then string the phonemes together to output transcript
    - In end-to-end, we go directly from audio to transcript
- **End-to-end** - using a machine learning model to directly bypass the pipeline
    - Usually works better with large datasets; with smaller datasets the traditional pipeline approach is better
- Know when to break down your problem into separate tasks depending on the available data
- Know when to use deep learning and when not to

## Pros & Cons
- Advantages
    - Sometimes human judgement is not that great; we might think some data is important when it's not
        - i.e. phonemes
    - Less hand-designing of components
- Cons
    - May need large amount of data
    - Excludes potentially useful hand-designed components (since we have knowledge)