import lightning.pytorch as pl
from lightning.pytorch.cli import LightningCLI

import torch
from datetime import datetime
from loss import GeoCLIPLoss
from model import GeoCLIP
from datamodules.naipgeo_dataset import NAIPGeoDataModule

torch.set_float32_matmul_precision('medium')

class GeoCLIPLightningModule(pl.LightningModule):
    def __init__(
            self,
            embed_dim=512,
            image_resolution=256,
            vision_layers=12,
            vision_width=768,
            vision_patch_size=32,
            in_channels=4,
            le_type="grid",
            frequency_num=16,
            max_radius=0.01,
            min_radius=0.00001,
            learning_rate=1e-4,
            num_hidden_layers = 2,
            capacity = 256
    ) -> None:
        super().__init__()

        self.model = GeoCLIP(
            embed_dim=embed_dim,
            image_resolution=image_resolution,
            vision_layers=vision_layers,
            vision_width=vision_width,
            vision_patch_size=vision_patch_size,
            in_channels=in_channels,
            le_type=le_type,
            frequency_num = frequency_num,
            max_radius = max_radius,
            min_radius = min_radius,
            num_hidden_layers=num_hidden_layers,
            capacity=capacity
        )

        self.loss_fun = GeoCLIPLoss()
        self.learning_rate = learning_rate

        self.save_hyperparameters()

    def common_step(self, batch, batch_idx):
        images = batch["image"]
        t_points = batch["point"]
        logits_per_image, logits_per_coord = self.model(images, t_points)
        return self.loss_fun(logits_per_image, logits_per_coord)

    def training_step(self, batch, batch_idx):
        loss = self.common_step(batch, batch_idx)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.common_step(batch, batch_idx)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(params=self.model.parameters(),lr=self.learning_rate) # specify in configs/default.yaml

def cli_main():

    # modify configs/default.yaml for learning rate etc.
    cli = LightningCLI(model_class=GeoCLIPLightningModule,
                       datamodule_class=NAIPGeoDataModule,
                       save_config_kwargs=dict(
                           config_filename="configs/test.yaml",
                           overwrite=True
                       ),
                       parser_kwargs={"default_config_files": ["configs/use_small_rcf.yaml"]},
                       seed_everything_default=0,
                       run=False,
                       )

    cli.datamodule.setup()

    ts = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    run_name = f"GeoCLIP_ER_{ts}"
    if cli.trainer.logger is not None:
        cli.trainer.logger.experiment.name = run_name

        # this seems to be necessary to force logging of datamodule hyperparams
        cli.trainer.logger.log_hyperparams(cli.datamodule.hparams)

    cli.trainer.fit(model=cli.model,
                train_dataloaders=cli.datamodule.train_dataloader(),
                val_dataloaders=cli.datamodule.val_dataloader())

if __name__ == "__main__":
    cli_main()
