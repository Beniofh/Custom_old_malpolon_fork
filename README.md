# Old Malpolon fork that has been customised

This is a modified and unofficial version of a `very old fork of the Malpolon framework`. You can find the current version of Malpolon here: [https://github.com/plantnet/malpolon](https://github.com/plantnet/malpolon). 

## Installation

Currently, only the development version is available.
First make sure that the dependences listed in the `requirements.txt` file are installed.

One way to do so is to use `conda`

```script
conda env create -n <name> -f environment.yml
conda activate <name>
```

`malpolon` can then be installed via `pip` using

```script
git clone https://github.com/Beniofh/Custom_old_malpolon_fork.git
cd malpolon
pip install -e .
```

To check that the installation went well, use the following command

```script
python -m malpolon.check_install
```

which, if you have CUDA properly installed, should output something similar to

```script
Using PyTorch version 1.13.0
CUDA available: True (version: 11.6)
cuDNN available: True (version: 8302)
Number of CUDA-compatible devices found: 1
```



## Examples

Examples using the GeoLifeCLEF 2022 dataset is provided in the `examples` folder.


## Documentation

To generate the documention, additional dependences contained in `docs/docs_requirements.txt` must be installed using

```script
pip install -r docs/docs_requirements.txt
```

The documentation can then be generated using

```script
make -C docs html
```

The result can be found in `docs/_build/html`.
