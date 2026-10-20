# DIV2K validation data

Download the **validation high-resolution images** from the [official DIV2K page](https://data.vision.ee.ethz.ch/cvl/DIV2K/) (`DIV2K_valid_HR.zip`). The [direct archive link](https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip) is also provided there. DIV2K is made available for academic research; check the terms on the official page before using it.

Extract it so the directory is:

```text
datasets/DIV2K/DIV2K_valid_HR/
├── 0801.png
├── 0802.png
├── ...
└── 0900.png
```

The archive downloaded for this experiment was 448,993,893 bytes and had SHA-256 `20dd31fd84d777bc1cf5d6b7654a3f569c0aec74458ae094122ad1d0489900fc`. Its ZIP integrity check passed and it contained 100 validation PNGs. These values identify the exact local copy used for the reported measurements.

Run commands from the project root. The dataset is intentionally ignored by Git and is not downloaded by the build or benchmark scripts. If it is absent, the synthetic experiments still run independently.

