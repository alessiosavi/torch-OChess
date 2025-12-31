# Installation Guide

Detailed instructions for installing Torch o'Chess.

## System Requirements

### Minimum Requirements

| Component | Requirement |
|-----------|-------------|
| OS | Windows 10+, macOS 10.15+, Linux |
| Python | 3.8 or higher |
| RAM | 8 GB |
| Storage | 2 GB free space |

### Recommended Requirements

| Component | Requirement |
|-----------|-------------|
| Python | 3.10+ |
| RAM | 16 GB |
| GPU | NVIDIA with 8GB+ VRAM |
| Storage | 10 GB free space |

## Installation Methods

### Method 1: pip install (Recommended)

```bash
# Clone repository
git clone https://github.com/yourrepo/torch-ochess.git
cd torch-ochess

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install package
pip install -e .
```

### Method 2: Requirements File

```bash
# Clone repository
git clone https://github.com/yourrepo/torch-ochess.git
cd torch-ochess

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Method 3: Conda

```bash
# Clone repository
git clone https://github.com/yourrepo/torch-ochess.git
cd torch-ochess

# Create conda environment
conda create -n ochess python=3.10
conda activate ochess

# Install PyTorch (with CUDA)
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia

# Install other dependencies
pip install -r requirements.txt
```

## Installing Stockfish

Stockfish is required for data generation and evaluation.

### macOS

```bash
# Using Homebrew
brew install stockfish

# Verify
stockfish --version
which stockfish  # Usually /usr/local/bin/stockfish
```

### Ubuntu/Debian

```bash
# Using apt
sudo apt update
sudo apt install stockfish

# Verify
stockfish --version
which stockfish  # Usually /usr/bin/stockfish
```

### Windows

1. Download from <https://stockfishchess.org/download/>
2. Extract to a folder (e.g., `C:\stockfish`)
3. Add to PATH or use full path in commands

```powershell
# Verify
C:\stockfish\stockfish.exe --version
```

### From Source

```bash
# Clone Stockfish
git clone https://github.com/official-stockfish/Stockfish.git
cd Stockfish/src

# Build
make build ARCH=x86-64-modern

# Install
sudo cp stockfish /usr/local/bin/
```

## GPU Support (Optional but Recommended)

### NVIDIA CUDA

```bash
# Check if CUDA is available
python -c "import torch; print(torch.cuda.is_available())"

# If False, install CUDA-enabled PyTorch
pip uninstall torch
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### Apple Silicon (M1/M2)

```bash
# PyTorch supports MPS (Metal Performance Shaders)
python -c "import torch; print(torch.backends.mps.is_available())"

# Install PyTorch for M1/M2
pip install torch torchvision
```

## Dependencies

### Core Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| torch | >=2.0.0 | Deep learning framework |
| python-chess | >=1.999 | Chess logic and board representation |
| pandas | >=1.5.0 | Data manipulation |
| numpy | >=1.21.0 | Numerical operations |
| pyarrow | >=10.0.0 | Parquet file support |

### Training Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| tqdm | >=4.64.0 | Progress bars |
| tensorboard | >=2.11.0 | Training visualization |
| pyyaml | >=6.0 | Configuration files |

### Optional Dependencies

| Package | Purpose |
|---------|---------|
| matplotlib | Plotting and visualization |
| jupyter | Interactive notebooks |
| pytest | Running tests |

## Verification

After installation, verify everything works:

```bash
# 1. Check Python package
python -c "from ochess.model import ChessResNet; print('Models: OK')"
python -c "from ochess.data import FenParser; print('Data: OK')"
python -c "from ochess.datagen import StockfishWrapper; print('Datagen: OK')"

# 2. Check Stockfish
python -c "
from ochess.datagen import StockfishWrapper
with StockfishWrapper('/usr/local/bin/stockfish') as sf:
    import chess
    result = sf.analyze(chess.Board())
    print(f'Stockfish: OK (best move: {result[\"best_move\"]})')
"

# 3. Check GPU
python -c "
import torch
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
elif torch.backends.mps.is_available():
    print('GPU: Apple Silicon MPS')
else:
    print('GPU: None (using CPU)')
"

# 4. Run tests
pytest tests/ -v --tb=short
```

## Troubleshooting

### ImportError: No module named 'ochess'

```bash
# Make sure you installed the package
pip install -e .

# Or add to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### Stockfish permission denied

```bash
# Make stockfish executable
chmod +x /path/to/stockfish
```

### CUDA out of memory

```bash
# Reduce batch size in training
python scripts/train.py --batch-size 64

# Or use CPU
python scripts/train.py --device cpu
```

### torch.cuda.is_available() returns False

```bash
# Check NVIDIA driver
nvidia-smi

# Reinstall PyTorch with CUDA
pip uninstall torch
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

## Directory Structure After Installation

```
torch-ochess/
├── ochess/              # Python package
├── scripts/             # CLI scripts
├── tests/               # Test suite
├── docs/                # Documentation
├── datasets/            # Generated datasets (created after generation)
├── checkpoints/         # Model checkpoints (created during training)
└── runs/                # TensorBoard logs (created during training)
```

## Updating

```bash
# Pull latest changes
git pull origin main

# Update dependencies
pip install -e . --upgrade
```

## Uninstalling

```bash
# Remove package
pip uninstall ochess

# Remove virtual environment
deactivate
rm -rf venv

# Remove generated files
rm -rf datasets/ checkpoints/ runs/
```
