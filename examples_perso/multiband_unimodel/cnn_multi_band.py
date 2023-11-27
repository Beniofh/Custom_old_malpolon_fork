import os
import shutil
import sys

import hydra
from omegaconf import DictConfig

import torch
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


from pathlib import Path
from omegaconf import OmegaConf

@hydra.main(version_base="1.1", config_path="config", config_name="cnn_multi_band_config")
def main(cfg: DictConfig) -> None:
    logger = pl.loggers.CSVLogger(".", name=False, version="")
    logger.log_hyperparams(cfg)
    
    # Alternative 3.1 : mise en place pour l'Alternative 3.2
    if cfg.visualization.validate_metric == True : 
        path_chk=Path(cfg.visualization.chk_path_validate_metric)
        path_yaml=path_chk.parent / "hparams.yaml"
        import yaml
        with open(path_yaml, 'r') as file:
            cfg = yaml.safe_load(file)

        cfg=OmegaConf.create(cfg)
        cfg.visualization.validate_metric = True
        cfg.visualization.chk_path_validate_metric = path_chk
        
    cls_num_list_train, patch_data_ext, cfg, cfg_modif, patch_band = Init_of_secondary_parameters(cfg=cfg)

    datamodule = MicroGeoLifeCLEF2022DataModule(**cfg.data,
                                                patch_data_ext = patch_data_ext,
                                                patch_data=cfg.patch.patch_data, 
                                                patch_band_mean = cfg.patch.patch_band_mean,
                                                patch_band_sd = cfg.patch.patch_band_sd,
                                                train_augmentation = cfg.train_augmentation,
                                                test_augmentation = cfg.test_augmentation,
                                                dataloader = cfg.dataloader, )
        
    # Alternative 1 : vérification du dataloader puis STOP
    if cfg.visualization.check_dataloader == True :   
        from check_dataloader import Check_dataloader
        Check_dataloader(datamodule, cfg, patch_data_ext, patch_band)
        sys.exit()
    
    cfg_model = hydra.utils.instantiate(cfg_modif.model)
    
    model = ClassificationSystem(cfg_model, **cfg.optimizer.SGD, **cfg.optimizer.scheduler, **cfg.optimizer.loss, dropout_proba = cfg.dropout_proba, cls_num_list_train=cls_num_list_train)
    #model.model.fc = torch.nn.Sequential(torch.nn.Dropout(cfg.dropout_proba), model.model.fc)
    
    # Alternative 2 : recherche du lr optimal puis STOP
    if cfg.visualization.auto_lr_finder == True :
        Auto_lr_find(model, datamodule, cfg.trainer.accelerator, cfg.trainer.devices)
        sys.exit()
    
    # entrainement 
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

    # fait un transfert leraning si activé
    if cfg.transfer_learning.transfer_learning_activated == True :
        model, datamodule, trainer = Transfer_learning_ia_biodiv(model, cfg, cfg_model, cls_num_list_train, patch_data_ext, logger)
        if cfg.transfer_learning.model_tf.load_checkpoint_for_model_tf == True and cfg.visualization.validate_metric != True :
        #if cfg.transfer_learning.model_tf.load_checkpoint_for_model_tf == True :
            chk_path = cfg.transfer_learning.model_tf.chk_tf_path
            checkpoint = torch.load(chk_path) 
            model.load_state_dict(checkpoint['state_dict'])

    # Alternative 3.2 : permet de voir les métriques assossié à un checkpoint puis STOP
    if cfg.visualization.validate_metric == True :                       
        chk_path = cfg.visualization.chk_path_validate_metric
        checkpoint = torch.load(chk_path) 
        model.load_state_dict(checkpoint['state_dict'])
        datamodule.setup(stage="fit")
        
        trainer.validate(model, datamodule=datamodule)
        trainer.test(model, datamodule=datamodule)

        shutil.rmtree(os.getcwd())
        sys.exit()   
        
    # lance l'entrainement
    trainer.fit(model, datamodule=datamodule)   # pour charger un model et continuer l'entrainement : trainer.fit(..., ckpt_path="some/path/to/my_checkpoint.ckpt")
    trainer.validate(model, datamodule=datamodule)
    Autoplot(os.getcwd(), cfg.visualization.graph)


if __name__ == "__main__":
    main()