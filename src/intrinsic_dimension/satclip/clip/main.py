import argparse
import os
from datetime import datetime

import lightning.pytorch
import torch
import wandb
from datamodules.s2geo_dataset import S2GeoDataModule
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.cli import LightningCLI
from loss import GeoCLIPLoss
from model import GeoCLIP
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

torch.set_float32_matmul_precision('high')

class GeoCLIPLightningModule(lightning.pytorch.LightningModule):
    def __init__(
        self,
        embed_dim=512,
        image_resolution=256,
        vision_layers=12,
        vision_width=768,
        vision_patch_size=32,
        in_channels=4,
        le_type="grid",
        pe_type="siren",
        frequency_num=16,
        max_radius=260,
        min_radius=1,
        legendre_polys=16,
        harmonics_calculation="analytic",
        sh_embedding_dims=32,
        learning_rate=1e-4,
        weight_decay=0.01,
        num_hidden_layers=2,
        capacity=256,
        # Downstream eval
        eval_downstream=False,
        air_temp_data_path="./data/air_temp_us",
        election_data_path="./data/election",
        rcf_empirical_data_dir=None, 
        rcf_seed=0,
        rcf_empirical_transform='pretrained',
        
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
            pe_type=pe_type,
            frequency_num=frequency_num,
            max_radius=max_radius,
            min_radius=min_radius,
            legendre_polys=legendre_polys,
            harmonics_calculation=harmonics_calculation,
            sh_embedding_dims=sh_embedding_dims,
            num_hidden_layers=num_hidden_layers,
            capacity=capacity,
            rcf_empirical_data_dir=rcf_empirical_data_dir, 
            rcf_seed=rcf_seed,
            rcf_empirical_transform=rcf_empirical_transform
        )

        self.loss_fun = GeoCLIPLoss()
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.eval_downstream = eval_downstream
        if eval_downstream == True:
            (
                self.air_train_y,
                self.air_train_x,
                self.air_train_c,
                self.air_test_y,
                self.air_test_x,
                self.air_test_c,
            ) = get_air_temp_data(air_temp_data_path)
            (
                self.ele_train_y,
                self.ele_train_x,
                self.ele_train_c,
                self.ele_test_y,
                self.ele_test_x,
                self.ele_test_c,
            ) = get_election_data(election_data_path)
        self.save_hyperparameters()

    def common_step(self, batch, batch_idx):
        images = batch["image"]
        t_points = batch["point"].float()
        logits_per_image, logits_per_coord = self.model(images, t_points)
        return self.loss_fun(logits_per_image, logits_per_coord)

    def training_step(self, batch, batch_idx):
        loss = self.common_step(batch, batch_idx)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.common_step(batch, batch_idx)
        self.log("val_loss", loss)
        if self.eval_downstream == True and batch_idx == 0:
            with torch.no_grad():
                # Air Temp
                c_emb_train = self.model.encode_location(
                    self.air_train_c.double().flip(1).cuda()
                )
                c_emb_test = self.model.encode_location(
                    self.air_test_c.double().flip(1).cuda()
                )
                reg = LinearRegression().fit(
                    c_emb_train.detach().cpu().numpy(), self.air_train_y.numpy()
                )
                test_y_pred = reg.predict(c_emb_test.detach().cpu().numpy())
                air_temp_mse = mean_squared_error(self.air_test_y.numpy(), test_y_pred)
                self.log("air_temp_mse", air_temp_mse)
                # Election
                c_emb_train = self.model.encode_location(
                    self.ele_train_c.double().flip(1).cuda()
                )
                c_emb_test = self.model.encode_location(
                    self.ele_test_c.double().flip(1).cuda()
                )
                reg = LinearRegression().fit(
                    c_emb_train.detach().cpu().numpy(), self.ele_train_y.numpy()
                )
                test_y_pred = reg.predict(c_emb_test.detach().cpu().numpy())
                election_mse = mean_squared_error(self.ele_test_y.numpy(), test_y_pred)
                self.log("election_mse", election_mse)

            return loss, air_temp_mse, election_mse
        else:
            return loss

    def configure_optimizers(self):
        exclude = (
            lambda n, p: p.ndim < 2
            or "bn" in n
            or "ln" in n
            or "bias" in n
            or "logit_scale" in n
        )
        include = lambda n, p: not exclude(n, p)

        named_parameters = list(self.model.named_parameters())
        gain_or_bias_params = [
            p for n, p in named_parameters if exclude(n, p) and p.requires_grad
        ]
        rest_params = [
            p for n, p in named_parameters if include(n, p) and p.requires_grad
        ]

        optimizer = torch.optim.AdamW(
            [
                {"params": gain_or_bias_params, "weight_decay": 0.0},
                {
                    "params": rest_params,
                    "weight_decay": self.weight_decay,
                },  # specify in configs/default.yaml
            ],
            lr=self.learning_rate,  # specify in configs/default.yaml
        )

        return optimizer


class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.add_argument("--watchmodel", action="store_true")


def cli_main(
    default_config_filename="/home/esther/geoclip/GeoCLIP/clip/configs/esther-vit16-L40.yaml",
):
    #     checkpoint_callback = ModelCheckpoint(
    #         monitor="val_loss", dirpath='geoclip_models', save_top_k=1, save_last=True
    #     )

    #     accelerator = "auto"
    #     if torch.cuda.is_available: accelerator = "gpu"

    save_config_fn = default_config_filename.replace(".yaml", "-latest.yaml")

    # modify configs/default.yaml for learning rate etc.
    cli = MyLightningCLI(
        model_class=GeoCLIPLightningModule,
        datamodule_class=S2GeoDataModule,
        save_config_kwargs=dict(
            # config_filename="configs/latest.yaml",
            config_filename=save_config_fn,
            overwrite=True,
        ),
        trainer_defaults={
            "accumulate_grad_batches": 16,
            "log_every_n_steps": 10,
            # "default_root_dir": 'geoclip_models',
            # "callbacks": checkpoint_callback,
            # "accelerator": accelerator,
            # "devices": 2,
            # "strategy": "ddp"
        },
        # parser_kwargs={"default_config_files": ["configs/default.yaml"]},
        parser_kwargs={"default_config_files": [default_config_filename]},
        seed_everything_default=0,
        run=False,
    )

    ts = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    run_name = f"SatCLIP-300K_{ts}"
    if cli.trainer.logger is not None:
        cli.trainer.logger.experiment.name = run_name

        # this seems to be necessary to force logging of datamodule hyperparams
        cli.trainer.logger.log_hyperparams(cli.datamodule.hparams)

        # DEBUG
        if hasattr(cli.config, "watchmodel"):
            if cli.config.watchmodel:
                wandb.watch(cli.model, log="all", log_graph=True)

    cli.trainer.fit(
        model=cli.model,
        datamodule=cli.datamodule,
    )


if __name__ == "__main__":
    #config_fn = "clip/configs/resnet18-L-10.yaml"
    #config_fn = "clip/configs/resnet18-L-40.yaml"
    #config_fn = "clip/configs/vit16-L-10.yaml"
    #config_fn = "clip/configs/vit16-L-40.yaml"
    #config_fn = "clip/configs/resnet50-L-10.yaml"
    #config_fn = "clip/configs/resnet50-L-40.yaml"
    #config_fn = "clip/configs/rcf-L-10_new.yaml"
    #config_fn = "clip/configs/rcf-L-40_new.yaml"
    config_fn = "clip/configs/konstantin.yaml"


    #A100 go vroom vroom 🚗💨
    if torch.cuda.get_device_name(device=0)=='NVIDIA A100 80GB PCIe':
        torch.backends.cuda.matmul.allow_tf32 = True
        print('Superfastmode! 🚀')
    else:
        torch.backends.cuda.matmul.allow_tf32 = False
    cli_main(config_fn)
