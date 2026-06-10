import os
import argparse
import yaml
import wandb
import time

from pathlib import Path
from dotenv import load_dotenv

import torch

import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import (
    DeviceStatsMonitor, 
    EarlyStopping, 
    ModelCheckpoint, 
    LearningRateMonitor, 
    RichProgressBar, 
    RichModelSummary, 
    StochasticWeightAveraging
)

from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.losses import DiceCELoss
from monai.transforms import AsDiscrete
from monai.data import decollate_batch

from training.utils import get_dataset
from models.utils import get_model

from utils import get_latest_version

class Net(L.LightningModule):
    def __init__(self, args):
        super().__init__()
        # Guarda toda la configuración (YAML + Parser) en el checkpoint
        self.save_hyperparameters(args)
        
        # Inicializa el modelo usando tu factoría
        self.model = get_model(args=self.hparams)

        # Configuración de métricas y pérdida
        self.loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
        self.dice_metric = DiceMetric(include_background=False, reduction="mean")
        
        # Post-procesamiento para métricas
        self.post_pred = AsDiscrete(argmax=True, to_onehot=self.hparams.out_channels)
        self.post_label = AsDiscrete(to_onehot=self.hparams.out_channels)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, labels = batch["image"], batch["label"]
        output = self(images)
        loss = self.loss_function(output, labels)
        
        # Registro automático en W&B
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch["image"], batch["label"]
        
        # Inferencia 3D profesional
        outputs = sliding_window_inference(
            images, self.hparams.roi_size, self.hparams.inference_batch_size, self.forward
        )
        
        loss = self.loss_function(outputs, labels)
        
        # Procesamiento de métricas
        outputs = [self.post_pred(i) for i in decollate_batch(outputs)]
        labels = [self.post_label(i) for i in decollate_batch(labels)]
        self.dice_metric(y_pred=outputs, y=labels)
        
        self.log("val_loss", loss, prog_bar=True)
        return {"val_loss": loss}

    def on_validation_epoch_end(self):
        # Agregación del Dice al final de la época
        mean_dice = self.dice_metric.aggregate().item()
        self.dice_metric.reset()
        self.log("val_dice", mean_dice, prog_bar=True)

    def configure_optimizers(self):
        # Optimizado para AdamW con el LR del YAML
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)

class MainModule:

    def __init__(self, args):

        self.args = args
        self.root = '.'
        self.log = os.path.join(self.root, 'logs', args.dataset, args.model, args.dimension)
        self.device = 'gpu' if torch.cuda.is_available() else 'cpu'


        print(f"\nNumber of GPUs avaliable {torch.cuda.device_count()}.")
        print(f"Using {self.device}.")


    def __call__(self):

        if self.args.mode == 'train':
            self.train()
        
        elif self.args.mode == 'test':
            self.test()

        elif self.args.mode == 'predict':
            self.predict()

        else:
            raise ValueError("")
        
    def train(self):

        net = Net(args=self.args)
       
        if self.args.resume:
            latest = get_latest_version(dir=self.log)
            checkpoint = os.path.join(self.log, latest, 'checkpoints', 'best.ckpt')
            net.load_from_checkpoint(checkpoint)


        # Logger
        run_name = f"{self.args.model}_{self.args.dataset}_{self.args.dimension}_lr{self.args.learning_rate}"
        
        wandb_logger = WandbLogger(
            project=self.args.project_name, 
            name=run_name,          
            group=self.args.model,  
            config=self.args        
        )

        checkpoint_dir = os.path.join(self.log, 'checkpoints')
        
        # Callbacks
        checkpoint_callback = ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename='best-{epoch:02d}-{val_loss:.2f}', 
            monitor='val_loss',
            mode='min',          
            save_top_k=3,        
            save_last=True,      
            verbose=True
        )

        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            mode='min'
        )

        callbacks = [
            DeviceStatsMonitor(),
            early_stopping,
            checkpoint_callback,
            LearningRateMonitor(logging_interval='epoch'),
            RichProgressBar(),
            RichModelSummary(max_depth=2),
            StochasticWeightAveraging(swa_lrs=1e-2)
        ]

        # Define Trainer
        trainer = L.Trainer(
            accelerator=self.device,
            precision='bf16-mixed',
            logger=wandb_logger,
            callbacks=callbacks,
            max_epochs=self.args.max_epochs,
            check_val_every_n_epoch=1,
            num_sanity_val_steps=2
        )

        dataset = get_dataset(args=self.args)

        start = time.time()
        print("init training...")
        trainer.fit(model=net, datamodule=dataset)
        print(f"duration = {(time.time() - start) / 3600} hours")

        

    def test(self):
        pass

    def predict(self):
        pass



def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='amos22', help="")
    parser.add_argument('--model', type=str, default='unet', help="")
    parser.add_argument('--dimension', type=str, default='3d', help="")
    parser.add_argument('--data_dir', type=str, default='../dataset', help="")
    parser.add_argument('--mode', type=str, default='train', help="")
    parser.add_argument('--resume', action='store_true', help="")
    parser.add_argument('--project_name', type=str, default='tfg', help="")
    parser.add_argument('--batch_size', type=int, default=1, help="")
    
    parser.add_argument('--num_workers', type=int, default=2, help="")
    parser.add_argument('--cache_rate', type=float, default=0.0, help="")
    
    args = parser.parse_args()

    config_path = f'config/{args.dataset}/{args.model}_{args.dimension}.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        for key, value in config.items():
            setattr(args, key, value)

    return args

if __name__ == "__main__":
    load_dotenv(Path('../.env'), override=True)
    WANDB_KEY = os.getenv("WANDB_API_KEY")

    if WANDB_KEY:
        wandb.login(key=WANDB_KEY)
    else: 
        print("No api key found in .env file, logging disable")

    args = get_args()
    MainModule(args=args)()
    wandb.finish()