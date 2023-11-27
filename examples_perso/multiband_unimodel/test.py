
import yaml
from yaml import SafeLoader 

def read(path):
    with open(path, 'r') as f:
        return list(yaml.load_all(f, Loader=SafeLoader))[0]

yaml = read("/home/bbourel/Documents/IA/malpolon/examples_perso/multiband_unimodel/jean_zay/checkpoint_for_load/2023-10-06_16-49-33_956642/hparams.yaml")

print("a")