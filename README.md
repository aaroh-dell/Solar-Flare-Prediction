# Solar Flare Prediction (ResNet-18)

This project explores predicting solar flare intensity from NASA SDO magnetogram data using deep learning.

## Why this problem?

Solar flare prediction isn’t a typical image classification task.

- The dataset is **highly imbalanced** (X-class flares are rare but critical)
- The model needs to understand **magnetic field patterns**, not just textures
- Missing a major flare (M/X) is much worse than a false positive

Because of this, optimizing only for accuracy doesn’t make sense. The focus here is on improving **recall for high-intensity flares**.

## Approach

I used a multi-task setup based on :contentReference[oaicite:0]{index=0}.

**Input:** 128×128 single-channel magnetograms  

The model has two outputs:
- **Classification head** → predicts flare class (B, C, M, X)  
- **Regression head** → predicts magnetic flux  

The idea was to let the shared backbone learn better representations by combining both tasks.

## Handling Imbalance

This turned out to be the main challenge.

- Applied **weighted loss** so mistakes on M/X classes matter more  
- Focused on **recall instead of accuracy** during evaluation  
- Tuned thresholds to avoid missing high-intensity events  


## Results

Compared to a simple CNN baseline:

- **Accuracy**
  - CNN: ~68%  
  - ResNet-18: ~84%

- **M-class recall**
  - CNN: 18%  
  - ResNet-18: 77%

- **X-class recall**
  - CNN: 0%  
  - ResNet-18: improved (model starts detecting rare events)

A key observation:
> The baseline model looked decent on accuracy but completely failed on the cases that actually matter.


## Tech Stack

- PyTorch, Torchvision  
- NumPy, Pandas, Scikit-learn  
- Matplotlib, Seaborn  
- Training done on Kaggle (Tesla T4 GPU)


## What I learned

- Accuracy can be misleading in imbalanced problems  
- Multi-task learning can help with representation quality  
- Small design choices (loss weighting, thresholds) matter a lot more than architecture here  


## Next Steps

- Use temporal data instead of single images  
- Try transformer-based models  
- Improve detection consistency for X-class flares  

---

## Note

Kaggle dataset paths (`/kaggle/input/...`) need to be updated for local runs.
