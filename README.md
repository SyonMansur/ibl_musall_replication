# Attempted recreation of Musall et al., 2019


WORK IN PROGRESS  /  NOT COMPLETE (2/6/26)

Musall, S., Kaufman, M. T., Juavinett, A. L., Gluf, S., & Churchland, A. K. (2019). Single-trial neural dynamics are dominated by richly varied movements. Nature neuroscience.

A replication of the finding that spontaneous uninstructed movement dominates neural activity across the cortex. This project uses the International Brain Laboratory (IBL) dataset to model the relationship between behavioral state and single-neuron firing rates.

## Project Structure
- `src/`: Core library for IBL data loading, alignment, and preprocessing.
- `scripts/`: Executable experiments.
  - `01_baseline_ridge.py`: Linear encoding model (Ridge Regression) using temporal design matrices (Toeplitz).
  - `02_deep_gru.py`: [In Progress] Non-linear encoding model using Gated Recurrent Units (GRU).
- `results/`: Generated plots and model performance metrics.

## Usage
```bash
# Install dependencies
pip install -r requirements.txt

# Run the linear baseline
python scripts/01_baseline_ridge.py