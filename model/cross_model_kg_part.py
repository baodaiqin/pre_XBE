from transformers import AdamW

from .surface_model_bert import SurfaceTransformer
from .symbolic_model import TripleTransformer

import pytorch_lightning as pl
import torchmetrics
import torch

class CrossModel_KG(pl.LightningModule):
    
    def __init__(self, conf):
        super().__init__()
        self.surface_trf = SurfaceTransformer(conf)
        self.symbol_trf = TripleTransformer(conf)    
        self.surface_loss_weight = conf.surface_loss_weight
        self.symbolic_loss_weight = conf.symbolic_loss_weight
        self.alignment_loss_weight = conf.alignment_loss_weight
        self.cross_stitch_active = True
        self.conf = conf

        self.train_acc = torchmetrics.Accuracy()
        self.valid_acc = torchmetrics.Accuracy()
        
        from .layers import (
            GatedCrossStitch,
            GatedCrossStitch_Multihead,
            AverageCrossStitch,
            FirstTokenCrossStitch,
            NoCrossStitch,
            ResiCrossStitch,
            Alignment,
            )
        stitch_model_cls = {
            'gated': GatedCrossStitch,
            'gated_multihead': GatedCrossStitch_Multihead,
            'average': AverageCrossStitch,
            'firsttoken': FirstTokenCrossStitch,
            'none': NoCrossStitch,
            'resi': ResiCrossStitch
            }[conf.stitch_model]
        n_layers1 = len(set(conf.surface_cross_stitch_send_layers))
        n_layers2 = len(set(conf.symbol_cross_stitch_send_layers))
        n_layers3 = len(set(conf.surface_cross_stitch_receive_layers))
        n_layers4 = len(set(conf.symbol_cross_stitch_receive_layers))
        assert n_layers1 == n_layers2 == n_layers3 == n_layers4
        n_cross_stitch_layers = n_layers1
        self.cross_stitch = stitch_model_cls(
            n_cross_stitch_layers,
            input1_dim=self.surface_trf.trf.config.hidden_size,
            input2_dim=self.symbol_trf.trf.config.hidden_size,
            attn_dim=conf.attn_dim,
            )

        if self.conf.cross_stitch_start_epoch > 0:
            self.cross_stitch_active = False
            for param in self.cross_stitch.parameters():
                param.requires_grad = False
                

        if self.conf.freeze_symbol_until_epoch > 0:
            for param in self.symbol_trf.parameters():
                param.requires_grad = False
        if self.conf.freeze_surface_until_epoch > 0:
            for param in self.surface_trf.parameters():
                param.requires_grad = False

        self.align = Alignment(
            repr1_dim=self.surface_trf.trf.config.hidden_size,
            repr2_dim=self.symbol_trf.trf.config.hidden_size,
            attn_dim=self.conf.alignment_attn_dim,
            )

        #to save hyperparameters()
        self.save_hyperparameters()
        
    def forward(self, batch, return_instances=False):
        """This method implements the main cross-encoder functionality.
        We have two transformer-based encoders, namely a symbol encoder
        (symbol_trf) and a surface encoder (surface_trf).  The forward
        methods of those two encoders are coroutines.
        The role of this method is to drive those two coroutines.
        That is, whenever one of the encoders reaches a layer specified
        as cross-stitch layer, we receive the current hidden states from
        that encoder layer, pass it into the cross stitch layer, and then
        send the updated hidden states to the other encoder, and vice versa.
        """
        stitch_idx = 0
        # initialize the two encoder coroutines
        symbol_send_layers = set(self.conf.symbol_cross_stitch_send_layers)
        symbol_rcv_layers = set(self.conf.symbol_cross_stitch_receive_layers)
        symbol_gen = self.symbol_trf.encoder(
            batch['sym_input_ids'],
            batch['sym_attention_mask'],
            cross_stitch_send_layers=symbol_send_layers,
            cross_stitch_receive_layers=symbol_rcv_layers,
            )
        
        symbol_repr = None
        for stitch_idx in range(self.cross_stitch.n_layers):
            # send current surface and symbol representations to the
            # encoding coroutines of their resepective encoders. The
            # encoders will process those representations layer-wise
            # until they reach a layer that is specified as "send" layer.
            # Once a "send" layer is reached, the coroutines will yield
            # the representations, which we receive here.

            symbol_repr = symbol_gen.send(symbol_repr)

        symbol_out = symbol_gen.send(symbol_repr)
        symbol_results = self.symbol_trf.predict(batch, symbol_out, return_instances=True)
        
        return symbol_results


    def training_step(self, batch, batch_idx):
        dict_out = self(batch)
        sym_loss = dict_out['loss']
        self.log('sym_train_loss', sym_loss, prog_bar=True, logger=True)

        sym_pred = torch.argmax(dict_out['relation_pred'], axis=-1)[:, 1]
        sym_y = dict_out['relation_target'][:, 1]
        train_acc = self.train_acc(sym_pred, sym_y)
        self.log('sym_train_acc', train_acc, prog_bar=True, logger=True)

        return sym_loss

    def training_epoch_end(self, outputs):
        self.train_acc.reset()
        
    def validation_step(self, batch, batch_idx):
        dict_out = self(batch)
        loss = dict_out['loss']
        self.log('val_loss', loss, prog_bar=True, logger=True)
        
        sym_pred = torch.argmax(dict_out['relation_pred'], axis=-1)[:, 1]
        sym_y =	dict_out['relation_target'][:, 1]

        self.valid_acc.update(sym_pred, sym_y)
        return loss

    def validation_epoch_end(self, outputs):
        self.log('sym_valid_acc', self.valid_acc.compute())
        self.valid_acc.reset()
    
    def test_step(self, batch, batch_idx):
        loss = self(batch)['loss']
        self.log('test_loss', loss, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=self.conf.lr)
