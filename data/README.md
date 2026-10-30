# Data directory

The datasets in this folder are **downloaded automatically** by `torchvision` the first time you
run `train_mnist.py`, `train_emnist.py`, `evaluate.py` or `inspect_datasets.py`. They are not
committed to git (see `.gitignore`); only this README and `.gitkeep` are tracked.

| Mode (web app) | Dataset | Classes | Train / test images | Download | On disk |
|---|---|---|---|---|---|
| `digits` | MNIST | 10 (`0`–`9`) | 60,000 / 10,000 | ~11 MB | ~64 MB |
| `characters` | EMNIST **ByClass** | 62 (`0`–`9`, `A`–`Z`, `a`–`z`) | 697,932 / 116,323 | ~562 MB zip | ~2.2 GB |

All images are 28×28 grayscale, white ink on a black background, stored as `uint8`.

```
data/
├── MNIST/raw/     train-images-idx3-ubyte, t10k-images-idx3-ubyte, ... (+ the .gz originals)
└── EMNIST/raw/    emnist-<split>-{train,test}-{images,labels}-*-ubyte for all six EMNIST splits
                   gzip.zip  (the original archive, which also contains the mapping files)
```

`torchvision` extracts all six EMNIST splits from the single archive; this project only uses
**ByClass**. To verify the downloaded data (shapes, label mapping, orientation, pixel statistics,
glyph geometry) run:

```bash
python inspect_datasets.py            # writes results/dataset_inspection.json + sample grids
```

## Offline / manual download

If the automatic download fails (firewall, flaky connection), place the files manually:

* MNIST: the four `*-ubyte.gz` files from <https://ossci-datasets.s3.amazonaws.com/mnist/>
  (mirror used by torchvision) into `data/MNIST/raw/`, then run any of the scripts once with
  downloading enabled - torchvision verifies the checksums and extracts them without downloading.
* EMNIST: `gzip.zip` from <https://biometrics.nist.gov/cs_links/EMNIST/gzip.zip> into
  `data/EMNIST/raw/`, then run any of the scripts once with downloading enabled - torchvision
  verifies the archive's checksum and extracts it instead of downloading it again.

Scripts accept `--data-dir` to use another location, and `inspect_datasets.py --no-download`
fails with a clear message instead of downloading.

## EMNIST orientation

The raw EMNIST tensors returned by `torchvision.datasets.EMNIST` are stored **transposed**
(rows and columns swapped), so letters look mirrored and rotated. The project fixes this in
exactly one place, `src.datasets.to_upright`, which applies a single `transpose(1, 2)` - no
extra flip or rotation. Training, evaluation and inference therefore all work with upright
glyphs, and the web app does not need to change the orientation of drawings.
`results/samples_emnist_byclass.png` (upright) and
`results/samples_emnist_byclass_raw_untransposed.png` (raw) show the difference.

## Class mapping (EMNIST ByClass)

Class indices 0–9 are the digits, 10–35 are `A`–`Z` and 36–61 are `a`–`z`. The official mapping
file `emnist-byclass-mapping.txt` (inside `data/EMNIST/raw/gzip.zip`, member
`gzip/emnist-byclass-mapping.txt`) lists `"<class index> <ASCII code>"` pairs, e.g. `10 65` → `A`.
The class index is **not** the ASCII code. `src/class_mapping.py` holds the table and
`inspect_datasets.py` checks it against both the official file and torchvision's `.classes`.

ByClass is strongly imbalanced (smallest class `j` has 1,896 training images, largest `1` has
38,374; digits are ~49% of the data), and several letters have upper- and lowercase forms that
look almost identical when handwritten (`C/c`, `O/o/0`, `S/s`, `V/v`, `W/w`, `X/x`, `Z/z`, ...).
Expect lower accuracy than on MNIST.

## Attribution and licensing

The datasets are third-party works and are **not** covered by this repository's license. Check
the original sources before redistributing the data or using it commercially.

### MNIST

* Created by **Yann LeCun, Corinna Cortes and Christopher J.C. Burges**, from images in NIST
  Special Databases 1 and 3. Original page: <http://yann.lecun.com/exdb/mnist/>.
* The original page states no explicit license. MNIST is commonly redistributed (for example by
  public dataset catalogues and mirrors) under the **Creative Commons Attribution-Share Alike 3.0
  (CC BY-SA 3.0)** license. Treat that as the customary terms rather than an official grant.
* Usual citation: Y. LeCun, L. Bottou, Y. Bengio and P. Haffner, "Gradient-based learning applied
  to document recognition," *Proceedings of the IEEE*, 86(11):2278–2324, 1998.

### EMNIST

* Please cite: G. Cohen, S. Afshar, J. Tapson and A. van Schaik, "EMNIST: an extension of MNIST
  to handwritten letters," arXiv:1702.05373, 2017 (also published at IJCNN 2017 as "EMNIST:
  Extending MNIST to handwritten letters").
* EMNIST is derived from **NIST Special Database 19** (Handprinted Forms and Characters) and is
  distributed by NIST (<https://www.nist.gov/itl/products-and-services/emnist-dataset>). The NIST
  Special Database 19 terms of use apply. NIST provides its data "as is", and its terms should be
  reviewed directly before redistribution or commercial use.
