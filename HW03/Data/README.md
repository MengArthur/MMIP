# Data

CIFAR-10 is downloaded automatically the first time `HW03_main.py` or the notebook runs (`Code/data.py`), so the raw files are not committed to git (~140 MB):

- Source: official CIFAR-10 release (Krizhevsky, 2009, <https://www.cs.toronto.edu/~kriz/cifar.html>), downloaded from its authors' group upload on the Hugging Face Hub: <https://huggingface.co/datasets/uoft-cs/cifar10>
- `cifar10_train.parquet` / `cifar10_test.parquet`: original download (PNG bytes + label)
- `cifar10_train.npz` / `cifar10_test.npz`: decoded uint8 arrays, cached to speed up later runs
