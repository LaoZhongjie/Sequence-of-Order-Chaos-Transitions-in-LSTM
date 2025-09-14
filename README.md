# Multiple Descents in Deep Learning Reproduction

This repository reproduces the experiments from the paper **"Multiple Descents in Deep Learning as a Sequence of Order-Chaos Transitions"** by Wei Wenbo et al.

## Paper Summary

The paper discovers a novel 'multiple-descent' phenomenon in LSTM training where test loss exhibits multiple cycles of increase followed by sharp decreases during overfitting. Key findings:

1. **Multiple descent cycles** occur in the overfitting regime
2. **Global optimum** occurs at the first order-to-chaos transition  
3. Each descent corresponds to **order-chaos phase transitions**
4. LSTM dynamics resemble **tanh map bifurcation patterns**

## Repository Structure

```
├── config.py                 # Configuration and hyperparameters
├── data_loader.py            # IMDB data loading and preprocessing  
├── model.py                  # LSTM model definition
├── chaos_analysis.py         # Asymptotic stability analysis
├── train.py                  # Training script with checkpointing
├── analyze_chaos.py          # Main chaos analysis script
├── visualize_results.py      # Generate paper figures
├── run_experiment.py         # Complete experiment pipeline
├── requirements.txt          # Python dependencies
└── README.md                # This file
```

## Dataset

The experiments use the **IMDB Large Movie Review Dataset** containing 50,000 movie reviews:

- **Official source**: http://ai.stanford.edu/~amaas/data/sentiment/
- **Kaggle**: https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews
- **Hugging Face**: https://huggingface.co/datasets/stanfordnlp/imdb

The code will automatically download the dataset using Hugging Face `datasets` library.

## Installation

1. **Clone repository** (create files from artifacts above)

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Configure settings** in `config.py`:
   - Set `DEVICE = 'cuda'` if you have GPU
   - Adjust `MAX_EPOCHS` (paper uses 10,000, start with 1,000)
   - Modify paths if needed

## Quick Start

### Option 1: Full Reproduction
```bash
python run_experiment.py
```

### Option 2: Quick Test (Reduced epochs)
```bash
python run_experiment.py --quick-test
```

### Option 3: Step-by-Step

1. **Train LSTM model**:
```bash
python train.py
```

2. **Analyze chaos dynamics**:
```bash
python analyze_chaos.py
```

3. **Generate visualizations**:
```bash
python visualize_results.py
```

## Model Architecture

Following the paper's Table I:

| Layer | Output Dimension |
|-------|-----------------|
| Embedding Layer | 32 |
| LSTM Layer | 60 |
| Fully Connected Layer | 1 |

**Training Details:**
- Optimizer: Adam (lr=0.0005)
- Loss: Binary Cross Entropy  
- Sequence Length: 500 tokens
- Batch Size: 32

## Chaos Analysis Methodology

The asymptotic stability analysis follows Figure 1 of the paper:

1. **Process review** through embedding (first 500 timesteps)
2. **Continue LSTM** with zero inputs (timesteps 500-1599)  
3. **Add perturbation** to initial hidden state
4. **Calculate distance** between original and perturbed trajectories
5. **Compute geometric mean** of distances across 500 test samples

**Order vs Chaos Detection:**
- **Order phase**: Asymptotic distance ≈ -15 (machine precision)
- **Chaos phase**: Asymptotic distance > -10
- **Transitions**: Sharp changes in asymptotic distance

## Expected Results

The reproduction should demonstrate:

### 1. Multiple Descent Cycles
- Test loss exhibits ~8 cycles of increase-decrease in overfitting regime (epochs 500-1000)
- Each cycle ends with sharp descent within single epoch

### 2. Order-Chaos Transitions  
- Asymptotic distance increases during loss increases (chaos)
- Sharp drops in distance coincide with loss decreases (order)
- Bifurcation diagram shows convergence (order) vs scattering (chaos)

### 3. Optimal Epoch Findings
- **Global optimum** occurs around epoch 114
- **First order-chaos transition** also around epoch 114  
- Confirms paper's key finding: optimal performance at first transition

### 4. Tanh Map Similarity
- LSTM bifurcation pattern resembles tanh map: k_t = r·k_{t-1}·(1-tanh(k_{t-1}))
- First transition is "widest", allowing best exploration

## Generated Outputs

### Files Created:
```
results/
├── training_history.json         # Loss and accuracy over epochs
├── chaos_analysis_results.json   # Asymptotic distances and transitions
└── figures/                      # Reproduction of paper figures
    ├── multiple_descents_overview.png    # Figure 2(a) reproduction
    ├── bifurcation_diagram.png          # Figure 2(b) reproduction  
    ├── descent_cycles_analysis.png      # Detailed cycle analysis
    ├── optimal_epoch_analysis.png       # Optimal epochs vs transitions
    └── tanh_map_comparison.png          # Figure 3 reproduction

checkpoints/
├── model_epoch_1.pt             # Model checkpoints for analysis
├── model_epoch_2.pt
├── ...
└── best_model.pt                # Best performing model
```

## Computational Requirements

**Training Time:**
- 1,000 epochs: ~2-6 hours (depending on hardware)
- 10,000 epochs: ~20-60 hours

**Chaos Analysis Time:**  
- 200 epochs: ~4-8 hours
- 1,000 epochs: ~20-40 hours

**Memory:**
- GPU: 4GB+ VRAM recommended
- RAM: 8GB+ system memory

**Recommendations:**
- Start with `--quick-test` for verification
- Use GPU acceleration if available
- Consider analyzing fewer epochs initially

## Troubleshooting

### Common Issues:

1. **CUDA out of memory**:
   - Reduce `BATCH_SIZE` in `config.py`
   - Use `DEVICE = 'cpu'`

2. **Dataset download fails**:
   - Manually download from Kaggle
   - Place `IMDB_Dataset.csv` in `data/` directory

3. **Long analysis time**:
   - Reduce `NUM_TEST_SAMPLES` in `config.py`
   - Analyze fewer epochs initially

4. **Missing checkpoints**:
   - Ensure training completed successfully
   - Check `checkpoints/` directory exists

## Key Parameters to Adjust

In `config.py`:

```python
# For faster experimentation
MAX_EPOCHS = 100              # Instead of 1000
NUM_TEST_SAMPLES = 100        # Instead of 500  
BATCH_SIZE = 16               # If memory limited

# For full reproduction  
MAX_EPOCHS = 1000             # Or 10000 as in paper
NUM_TEST_SAMPLES = 500        # As in paper
ASYMPTOTIC_TIMESTEPS = 1600   # As in paper
```

## Verification Checklist

- [ ] Training converges and overfits after ~100-200 epochs
- [ ] Multiple descent cycles appear in overfitting regime  
- [ ] Asymptotic distances show order-chaos transitions
- [ ] Global optimum occurs near first order-chaos transition
- [ ] Bifurcation diagram shows order (convergence) vs chaos (scattering)
- [ ] Generated figures match paper's Figure 2 patterns

## Citation

If you use this reproduction code, please cite the original paper:

```bibtex
@article{wei2025multiple,
  title={Multiple Descents in Deep Learning as a Sequence of Order-Chaos Transitions},
  author={Wei, Wenbo and Chong, Nicholas Jia Le and Lai, Choy Heng and Feng, Ling},
  journal={arXiv preprint arXiv:2505.20030},
  year={2025}
}
```

## Contact

For issues with this reproduction code, please check:
1. Configuration settings in `config.py`
2. Hardware requirements and computational time
3. Dataset download and preprocessing steps

The reproduction aims to faithfully implement the paper's methodology while providing clear documentation and modular code structure for educational purposes.