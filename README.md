# ethos_TDS

## Installation

We recommend installation via Anaconda (refer to [Anaconda website and follow the instructions](https://docs.anaconda.com/anaconda/install/)).

* Linux and macOS

```bash
git clone git@github.com:rchan26/ethos_TDS.git
cd ethos_TDS
conda env create
conda activate ethosTDSenv
```

* Windows

We recommend first installing the [Windows Subsystem for Linux](https://docs.microsoft.com/en-us/windows/wsl/install) (WSL). You can then use the following to install Conda (following the Linux instructions [here](https://docs.conda.io/en/latest/miniconda.html#linux-installers)):

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

You can then follow the installation instructions for Linux and macOS above.

* For using within Jupyter, you can create a kernel with:

```bash
python -m ipykernel install --user --name ethosTDS
```
