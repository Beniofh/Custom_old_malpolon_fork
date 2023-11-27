import os
import shutil

import hydra
import torch
from omegaconf import DictConfig
import pandas as pd
from datetime import datetime
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from malpolon.logging import Summary

from transforms import *
from auto_plot import Autoplot

from init_elements import Init_of_secondary_parameters

from pytorch_lightning.callbacks import LearningRateMonitor

from transfer_learning import Transfer_learning_ia_biodiv
from auto_lr_finder import Auto_lr_find
from datamodule import MicroGeoLifeCLEF2022DataModule, ClassificationSystem


@hydra.main(version_base="1.1", config_path="config", config_name="cnn_multi_band_config")
def main(cfg: DictConfig) -> None:
    logger = pl.loggers.CSVLogger(".", name=False, version="")
    logger.log_hyperparams(cfg)
    
    cls_num_list_train, patch_data_ext, cfg, cfg_modif = Init_of_secondary_parameters(cfg=cfg)

    datamodule = MicroGeoLifeCLEF2022DataModule(**cfg.data,
                                                patch_data_ext = patch_data_ext,
                                                patch_data=cfg.patch.patch_data, 
                                                patch_band_mean = cfg.patch.patch_band_mean,
                                                patch_band_sd = cfg.patch.patch_band_sd)
        
    cfg_model = hydra.utils.instantiate(cfg_modif.model)
    model = ClassificationSystem(cfg_model, **cfg.optimizer.SGD, **cfg.optimizer.scheduler, **cfg.optimizer.loss, cls_num_list_train=cls_num_list_train)

    callbacks = [
        Summary(),
        ModelCheckpoint(
            dirpath=os.getcwd(),
            filename="checkpoint-{epoch:02d}-{step}-{val_accuracy:.4f}", 
            monitor= cfg.callbacks.monitor,
            mode=cfg.callbacks.mode,),
        LearningRateMonitor(logging_interval=cfg.optimizer.scheduler.logging_interval), #epoch'),
        EarlyStopping(monitor=cfg.callbacks.monitor, min_delta=0.00, patience=cfg.callbacks.patience, verbose=False, mode=cfg.callbacks.mode)]                
    
    trainer = pl.Trainer(logger=logger, callbacks=callbacks, **cfg.trainer)



    ### adaptation des paramettres pour le chargement du nouveau jeu de donnée
    data_tf = cfg.data.copy()
    #data_tf.dataset_path = '/home/bbourel/Data/malpolon_datasets/Galaxy117-Sort_on_data_82'
    #data_tf.csv_occurence_path = '/home/bbourel/Data/malpolon_datasets/Galaxy117-Sort_on_data_82/Galaxy117-Sort_on_data_82_n_vec.csv'
    data_tf.dataset_path = '/gpfsscratch/rech/cvn/uva31ry/Plancton_FLORTOT_ia_biodiv'
    data_tf.csv_occurence_path = '/gpfsscratch/rech/cvn/uva31ry/Plancton_FLORTOT_ia_biodiv/Plancton_FLORTOT_ia_biodiv_clean_vec_subset.csv'
    data_tf.csv_separator = ','
    data_tf.csv_col_occurence_id = 'id' # 'SurveyID'
    data_tf.csv_col_class_id = 'SurveyID' #ne pas modifier
    data_tf.train_batch_size = 32
    data_tf.inference_batch_size = 256
    data_tf.num_workers = 8

    ### configuration d'un datamodule pour le nouveau jeu de donnée sur la base de l'adaptation des paramettres pour le chargement du nouveau jeu de donnée 
    datamodule_tf = MicroGeoLifeCLEF2022DataModule(**data_tf,
                                                   patch_data_ext = patch_data_ext,
                                                   patch_data=cfg.patch.patch_data, 
                                                   patch_band_mean = cfg.patch.patch_band_mean,
                                                   patch_band_sd = cfg.patch.patch_band_sd)
        
    #chk_path ='/home/bbourel/Documents/IA/malpolon/examples_perso/multiband_unimodel/outputs/cnn_multi_band/2023-04-05_18-21-28_367789/checkpoint-epoch=05-step=7764-val_accuracy=0.0959.ckpt'
    #chk_path ='/home/bbourel/Documents/IA/malpolon/examples_perso/multiband_unimodel/outputs/cnn_multi_band/2023-04-20_15-02-40_108293/checkpoint-epoch=18-step=23939-val_accuracy=0.0948.ckpt'
    chk_path = '/gpfswork/rech/cvn/uva31ry/checkpoint_for_load/2023-04-28_12-14-11_553436/checkpoint-epoch=02-step=3882-val_accuracy=0.0972.ckpt'

    checkpoint = torch.load(chk_path) 
    model.load_state_dict(checkpoint['state_dict'])
    datamodule_tf.setup(stage="fit")

    data_loader = datamodule_tf.val_dataloader()
    y_pred = []
    y_true = []  
    for inputs, labels in data_loader:
        output = model(inputs) # Feed Network
        y_pred.extend(output.tolist()) # Save Prediction
        y_true.extend(labels) 

    y_true_val_list = []
    for tensor in y_true:
        y_true_val_list.append(tensor.item())

    data_loader = datamodule_tf.train_dataloader()
    for inputs, labels in data_loader:
        output = model(inputs) # Feed Network
        y_pred.extend(output.tolist()) # Save Prediction
        y_true.extend(labels) 

    y_true_list = []
    for tensor in y_true:
        y_true_list.append(tensor.item())
   
    df = pd.DataFrame(y_pred)
    
    df.insert(0, 'subset', 'train')
    df.insert(0, 'SurveyID', y_true_list)
    df.subset[df.SurveyID.isin(y_true_val_list)]="val"
    
    if not os.path.exists(os.getcwd()[:-len(Path(os.getcwd()).name)] + 'tranfer_learning_rf/'):
        os.makedirs(os.getcwd()[:-len(Path(os.getcwd()).name)] + 'tranfer_learning_rf/')

    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    df.to_csv(os.getcwd()[:-len(Path(os.getcwd()).name)] + 'tranfer_learning_rf/' + now + '.csv', index=False)
    shutil.rmtree(os.getcwd())

if __name__ == "__main__":
    main()