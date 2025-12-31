# Troubleshooting Guide

Solutions to common issues when using Torch o'Chess.

## Installation Issues

### ImportError: No module named 'ochess'

**Problem:** Python can't find the ochess package.

**Solutions:**

```bash
# Install the package
pip install -e .

# Or add to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Verify installation
python -c "import ochess; print('OK')"
```

### Stockfish Not Found

**Problem:** `FileNotFoundError: Stockfish not found`

**Solutions:**

```bash
# Check if installed
which stockfish

# macOS
brew install stockfish

# Ubuntu/Debian
sudo apt install stockfish

# Specify path explicitly
python scripts/generate_data.py --stockfish-path /path/to/stockfish
```

### Permission Denied for Stockfish

**Problem:** `PermissionError: [Errno 13] Permission denied`

**Solution:**

```bash
chmod +x /path/to/stockfish
```

### torch.cuda.is_available() Returns False

**Problem:** PyTorch doesn't detect GPU.

**Solutions:**

```bash
# Check NVIDIA driver
nvidia-smi

# Reinstall PyTorch with CUDA
pip uninstall torch
pip install torch --index-url https://download.pytorch.org/whl/cu118

# Verify
python -c "import torch; print(torch.cuda.is_available())"
```

---

## Data Generation Issues

### Generation is Very Slow

**Problem:** Data generation takes too long.

**Solutions:**

```bash
# Lower analysis depth (faster but lower quality)
python scripts/generate_data.py --depth 10

# Use more threads
python scripts/generate_data.py --threads 4

# Skip bad move generation
python scripts/generate_data.py --no-bad-moves
```

### Stockfish Crashes

**Problem:** Stockfish process dies during generation.

**Solutions:**

```bash
# Reduce hash size
python scripts/generate_data.py --hash 64

# Reduce threads
python scripts/generate_data.py --threads 1

# Update Stockfish to latest version
```

### Dataset File is Empty

**Problem:** Generated parquet file has no data.

**Solutions:**

```bash
# Check for errors
python scripts/generate_data.py --verbose

# Verify Stockfish works
stockfish <<< "uci"

# Check disk space
df -h
```

---

## Training Issues

### CUDA Out of Memory

**Problem:** `RuntimeError: CUDA out of memory`

**Solutions:**

```bash
# Reduce batch size
python scripts/train.py --batch-size 64

# Enable mixed precision (uses less memory)
python scripts/train.py --mixed-precision

# Use gradient accumulation
python scripts/train.py --batch-size 32 --accumulation-steps 4

# Use smaller model
python scripts/train.py --model resnet --num-blocks 4 --channels 128
```

### Loss Not Decreasing

**Problem:** Training loss stays flat.

**Solutions:**

1. **Lower learning rate:**

   ```bash
   python scripts/train.py --lr 0.0001
   ```

2. **Add warmup:**

   ```bash
   python scripts/train.py --warmup-steps 1000
   ```

3. **Check data:**

   ```python
   import pandas as pd
   df = pd.read_parquet("datasets/train.parquet")
   print(df.head())
   print(df['best_move_uci'].value_counts())
   ```

4. **Try different optimizer:**

   ```bash
   python scripts/train.py --optimizer sgd --lr 0.01
   ```

### Loss is NaN

**Problem:** `loss: nan`

**Solutions:**

1. **Lower learning rate:**

   ```bash
   python scripts/train.py --lr 0.00001
   ```

2. **Add gradient clipping:**

   ```bash
   python scripts/train.py --gradient-clip 0.5
   ```

3. **Disable mixed precision:**

   ```bash
   python scripts/train.py  # without --mixed-precision
   ```

4. **Check for bad data:**

   ```python
   import pandas as pd
   import numpy as np
   df = pd.read_parquet("datasets/train.parquet")
   print(df.isnull().sum())
   print((df['score_cp'] == np.inf).sum())
   ```

### Overfitting (Train << Val Loss)

**Problem:** Training loss much lower than validation loss.

**Solutions:**

1. **Add regularization:**

   ```bash
   python scripts/train.py --weight-decay 0.1 --dropout 0.2
   ```

2. **Use smaller model:**

   ```bash
   python scripts/train.py --num-blocks 4 --channels 128
   ```

3. **Get more data:**

   ```bash
   python scripts/generate_data.py --num-positions 100000
   ```

4. **Use early stopping:**

   ```bash
   python scripts/train.py --early-stopping --patience 10
   ```

### Slow Training

**Problem:** Training takes too long.

**Solutions:**

```bash
# Enable mixed precision
python scripts/train.py --mixed-precision

# Use more workers
python scripts/train.py --num-workers 4

# Enable tensor caching
python scripts/train.py --cache-dir datasets/cache

# Reduce sequence length
python scripts/train.py --sequence-length 3
```

---

## Evaluation Issues

### Model Plays Poorly

**Problem:** Model makes bad moves.

**Solutions:**

1. **Check model is in eval mode:**

   ```python
   model.eval()
   ```

2. **Use temperature 0:**

   ```bash
   python scripts/evaluate.py --temperature 0.0
   ```

3. **Verify legal move filtering:**

   ```python
   from ochess.data import MoveEncoder
   encoder = MoveEncoder()
   filtered = encoder.filter_to_legal(logits, board)
   ```

4. **Train on more data:**
   - Generate larger dataset
   - Train for more epochs

### Evaluation Games Timeout

**Problem:** Games take too long.

**Solutions:**

```bash
# Limit moves per game
python scripts/evaluate.py --max-moves 100

# Use lower Stockfish level for testing
python scripts/evaluate.py --levels 1,3,5
```

---

## Play Interface Issues

### Invalid Move Error

**Problem:** `ValueError: Invalid move`

**Solutions:**

- Use correct notation (SAN or UCI):

  ```
  e4 (correct)
  E4 (wrong - lowercase)
  e2e4 (correct - UCI)
  ```

- Check if move is legal:

  ```
  moves  # Shows all legal moves
  ```

### Board Display Issues

**Problem:** Board characters don't display correctly.

**Solutions:**

```bash
# Use ASCII mode
python scripts/play.py --no-unicode

# Check terminal encoding
echo $LANG  # Should be UTF-8
```

---

## General Issues

### Memory Leak

**Problem:** Memory usage grows over time.

**Solutions:**

```python
# Clear CUDA cache periodically
import torch
torch.cuda.empty_cache()

# Use with torch.no_grad() for inference
with torch.no_grad():
    outputs = model(inputs)
```

### Reproducibility Issues

**Problem:** Results differ between runs.

**Solutions:**

```python
import torch
import numpy as np
import random

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
```

```bash
python scripts/train.py --seed 42
python scripts/generate_data.py --seed 42
```

---

## Getting Help

If you still have issues:

1. **Check logs:**

   ```bash
   python scripts/train.py --verbose 2>&1 | tee training.log
   ```

2. **Create minimal example:**

   ```python
   # Isolate the problem to a small script
   from ochess.model import ChessResNet
   model = ChessResNet()
   # ... reproduce issue
   ```

3. **Report bug:**
   - Include Python version, PyTorch version
   - Include full error traceback
   - Include minimal reproduction code
