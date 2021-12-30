import torch
import numpy as np
from models.transformer import BasicTransformerModel
from models import BaseModel
from models.flowplusplus import FlowPlusPlus
import ast
from torch import nn
import math

from .util.generation import autoregressive_generation_multimodal

from models.cdvae import ConditionalDiscretizedModel

class DiscreteModel(BaseModel):
    def __init__(self, opt):
        super().__init__(opt)
        input_mods = self.input_mods
        output_mods = self.output_mods
        dins = self.dins
        douts = self.douts
        input_lengths = self.input_lengths
        output_lengths = self.output_lengths
        self.conditioning_seq_lens = [int(x) for x in str(self.opt.conditioning_seq_lens).split(",")]
        if len(self.conditioning_seq_lens) < len(self.conditioning_seq_lens):
            if len(self.conditioning_seq_lens) == 1:
                self.conditioning_seq_lens = self.conditioning_seq_lens*len(self.conditioning_seq_lens)
            else:
                raise Exception("number of output_time_offsets doesnt match number of output_mods")

        self.outputs_chunking = outputs_chunking = [int(x) for x in str(self.opt.outputs_chunking).split(",")]

        self.input_mod_nets = []
        self.output_mod_nets = []
        self.output_mod_mean_nets = []
        self.output_mod_trans = []
        #self.output_mod_vaes = []
        self.module_names = []
        self.out_temp = opt.out_temp
        for i, mod in enumerate(input_mods):
            net = BasicTransformerModel(opt.dhid, dins[i], opt.nhead, opt.dhid, 2, opt.dropout, self.device, use_pos_emb=True, input_length=input_lengths[i], use_x_transformers=opt.use_x_transformers, opt=opt)
            name = "_input_"+mod
            setattr(self,"net"+name, net)
            self.input_mod_nets.append(net)
            self.module_names.append(name)
        for i, mod in enumerate(output_mods):

            assert douts[i] % outputs_chunking[i] == 0 #TODO: implement padding to avoid this restrictiion
            assert output_lengths[i] == 1

            # import pdb;pdb.set_trace()
            trans = ConditionalDiscretizedModel(
                input_shape = (output_lengths[i], 1),
                output_dim = douts[i],
                cond_len=self.conditioning_seq_lens[i],
                num_tokens = opt.num_tokens,        # number of visual tokens. in the paper, they used 8192, but could be smaller for downsized projects
                temperature = opt.out_temp,        # gumbel softmax temperature, the lower this is, the harder the discretization
                cond_dim = opt.dhid,
                nhead = opt.out_nhead,
                dhid = opt.out_dhid,
                nlayers = opt.out_nlayers,
                dropout = opt.out_dropout,
                use_pos_emb = not opt.out_no_use_pos_emb,
                use_x_transformers = opt.out_use_x_transformers,
                opt = opt
            )
            #name = "_output_trans_"+mod
            name = "_output_vae_"+mod
            setattr(self, "net"+name, trans)
            self.output_mod_trans.append(trans)
            #self.output_mod_vaes.append(trans)

            net = BasicTransformerModel(opt.dhid, opt.dhid, opt.nhead, opt.dhid, opt.nlayers, opt.dropout, self.device, use_pos_emb=opt.use_pos_emb_output, input_length=sum(input_lengths), use_x_transformers=opt.use_x_transformers, opt=opt)
            name = "_output_"+mod
            setattr(self, "net"+name, net)
            self.output_mod_nets.append(net)
            self.module_names.append(name)
            #if opt.residual:
            #    if self.opt.cond_concat_dims:
            #        net = nn.Linear(opt.dhid,douts[i])
            #    else:
            #        net = nn.Linear(opt.dhid,opt.douts[i])
            #    name="_output_mean_encoder"
            #    setattr(self, "net"+name, net)
            #    self.output_mod_mean_nets.append(net)


        self.mean_loss = nn.MSELoss()
        self.inputs = []
        self.targets = []
        self.mse_loss = 0
        self.nll_loss = 0

    def name(self):
        return "discrete"

    @staticmethod
    def modify_commandline_options(parser, opt):
        parser.add_argument('--dhid', type=int, default=512)
        parser.add_argument('--out_dhid', type=int, default=512)
        parser.add_argument('--out_emb_dim', type=int, default=512)
        # parser.add_argument('--conditioning_seq_lens', type=str, default=None, help="the number of outputs of the conditioning transformers to feed (meaning the number of elements along the sequence dimension)")
        parser.add_argument('--num_tokens', type=int, default=2048)
        parser.add_argument('--nhead', type=int, default=8)
        parser.add_argument('--out_nhead', type=int, default=8)
        parser.add_argument('--nlayers', type=int, default=8)
        parser.add_argument('--out_nlayers', type=int, default=8)
        parser.add_argument('--conditioning_seq_lens', type=str, default="8")
        parser.add_argument('--loss_weight_initial', type=float, default=0)
        parser.add_argument('--loss_weight_warmup_epochs', type=float, default=100)
        parser.add_argument('--max_prior_loss_weight', type=float, default=0, help="max value of prior loss weight during stage 1 (e.g. 0.01 is a good value)")
        parser.add_argument('--anneal_rate', type=float, default=1e-6)
        parser.add_argument('--temp_min', type=float, default=0.5)
        parser.add_argument('--dropout', type=float, default=0.1)
        parser.add_argument('--out_dropout', type=float, default=0.1)
        parser.add_argument('--out_temp', type=float, default=1.0)
        #parser.add_argument('--scales', type=str, default="[[10,0]]")
        #parser.add_argument('--residual', action='store_true', help="whether to use the vae to predict the residual around a determnisitic mean")
        parser.add_argument('--use_pos_emb_output', action='store_true', help="whether to use positional embeddings for output modality transformers")
        parser.add_argument('--use_rotary_pos_emb', action='store_true', help="whether to use rotary position embeddings")
        parser.add_argument('--use_x_transformers', action='store_true', help="whether to use rotary position embeddings")
        parser.add_argument('--out_use_x_transformers', action='store_true', help="whether to use rotary position embeddings")
        parser.add_argument('--no_use_pos_emb', action='store_true', help="dont use positional embeddings for the prior transformer")
        parser.add_argument('--out_no_use_pos_emb', action='store_true', help="dont use positional embeddings for the prior transformer")
        parser.add_argument('--outputs_chunking', type=str, default="1", help="number of chunks in which the inputs are chunked")
        return parser

    def forward(self, data, temp=1.0):
        # in lightning, forward defines the prediction/inference actions
        opt=self.opt
        latents = []
        for i, mod in enumerate(self.input_mods):
            latents.append(self.input_mod_nets[i].forward(data[i]))
        latent = torch.cat(latents)
        outputs = []
        #if self.opt.residual:
        #    for i, mod in enumerate(self.output_mods):
        #        trans_output = self.output_mod_nets[i].forward(latent)[:self.conditioning_seq_lens[i]]
        #        trans_predicted_mean_latents = self.output_mod_nets[i].forward(latent)[self.conditioning_seq_lens[i]:self.conditioning_seq_lens[i]+self.output_lengths[i]]
        #        predicted_mean = self.output_mod_mean_nets[i](trans_predicted_mean_latents)
        #        # residual, _ = self.output_mod_glows[i](x=None, cond=trans_output.permute(1,0,2), reverse=True)
        #        residual = self.output_mod_vaes[i].generate(trans_output.permute(1,2,0), temp=temp)
        #        residual = residual.squeeze(-1)
        #        shap = residual.shape
        #        residual = residual.view(shap[0],shap[1]*self.outputs_chunking[i],shap[2]//self.outputs_chunking[i])
        #        output = predicted_mean + residual.permute(2,0,1)
        #        outputs.append(output)
        #else:
        for i, mod in enumerate(self.output_mods):
            trans_output = self.output_mod_nets[i].forward(latent)[:self.conditioning_seq_lens[i]]
            output = self.output_mod_trans[i].generate(trans_output.permute(1,2,0), temp=temp)
            #import pdb;pdb.set_trace()
            #output = output.squeeze(-1)
            #shap = output.shape
            #output = output.view(shap[0],shap[1]*self.outputs_chunking[i],shap[2]//self.outputs_chunking[i])
            #import pdb;pdb.set_trace()
            outputs.append(output.permute(2,0,1)) #out: time, batch, features
        return outputs

    def training_step(self, batch, batch_idx):
        opt = self.opt
        self.set_inputs(batch)
        targets = []
        for i, mod in enumerate(self.output_mods):
            targets.append(self.targets[i])
            targets_shape = targets[i].shape
            targets[i] = torch.stack(torch.chunk(targets[i],self.outputs_chunking[i], dim=2), dim=2)
            targets[i] = targets[i].permute(1,0,2,3).reshape(targets_shape[1], targets_shape[0]*self.outputs_chunking[i], targets_shape[2]//self.outputs_chunking[i]).permute(1,0,2)
        # print(self.input_mod_nets[0].encoder1.weight.data)
        # print(self.targets[0])
        latents = []
        for i, mod in enumerate(self.input_mods):
            latents.append(self.input_mod_nets[i].forward(self.inputs[i]))

        latent = torch.cat(latents)
    # print(latent)
        #if self.opt.residual:
        #    nll_loss = 0
        #    mse_loss = 0
        #    accuracies = []
        #    for i, mod in enumerate(self.output_mods):
        #        trans_output = self.output_mod_nets[i].forward(latent)
        #        latents1 = trans_output[:self.conditioning_seq_lens[i]]
        #        latents2 = latents1
        #        trans_predicted_mean_latents = trans_output[self.conditioning_seq_lens[i]:self.conditioning_seq_lens[i]+self.output_lengths[i]]
        #        predicted_mean = self.output_mod_mean_nets[i](trans_predicted_mean_latents)
        #        vae = self.output_mod_vaes[i]
        #        if not self.opt.stage2:
        #            nll_loss += vae((targets[i] - predicted_mean).permute(1,2,0), cond=latents1.permute(1,2,0), return_loss=True, temp=self.vae_temp) #time, batch, features -> batch, features, time
        #            if self.opt.max_prior_loss_weight > 0:
        #                prior_loss, accuracy = vae.prior_logp((targets[i] - predicted_mean).permute(1,2,0), cond=latents2.permute(1,2,0), return_accuracy=True)
        #                accuracies.append(accuracy)
        #                nll_loss += self.prior_loss_weight * prior_loss
        #        else:
        #            prior_loss, accuracy = vae.prior_logp((targets[i] - predicted_mean).permute(1,2,0), cond=latents2.permute(1,2,0), return_accuracy=True, detach_cond=True)
        #            nll_loss += prior_loss
        #            accuracies.append(accuracy)
        #        mse_loss += 100*self.mean_loss(predicted_mean[i], targets[i])
        #    loss = nll_loss + mse_loss
        #    self.mse_loss = mse_loss
        #    self.nll_loss = nll_loss
        #    self.log('mse_loss', mse_loss)
        #    self.log('nll_loss', nll_loss)
        #    if len(accuracies) > 0:
        #        self.log('accuracy', torch.mean(torch.stack(accuracies)))
        #else:
        loss = 0
        accuracies = []
        # import pdb;pdb.set_trace()
        for i, mod in enumerate(self.output_mods):
            output1 = self.output_mod_nets[i].forward(latent)[:self.conditioning_seq_lens[i]]
            trans = self.output_mod_trans[i]
            trans_loss, accuracy = trans(targets[i].permute(1,2,0), cond=output1.permute(1,2,0), return_accuracy=True)
            ##prior_loss, accuracy = trans.prior_logp(targets[i].permute(1,2,0), return_accuracy=True, detach_cond=True)
            loss += trans_loss
            accuracies.append(accuracy)

        self.log('loss', loss)
        if len(accuracies) > 0:
           self.log('accuracy', torch.mean(torch.stack(accuracies)))
        # print(loss)
        # for p in self.output_mod_nets[0].parameters():
        #     print(p.norm())
        return loss

    #def test_step(self, batch, batch_idx):
    #    if self.opt.residual:
    #        self.eval()
    #        loss = self.training_step(batch, batch_idx)
    #        # print(loss)
    #        return {"test_loss": loss, "test_mse_loss": self.mse_loss, "test_nll_loss": self.nll_loss}
    #    else:
    #        return super().test_step(batch, batch_idx)

    #def test_epoch_end(self, outputs):
    #    if self.opt.residual:
    #        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
    #        avg_mse_loss = torch.stack([x['test_mse_loss'] for x in outputs]).mean()
    #        avg_nll_loss = torch.stack([x['test_nll_loss'] for x in outputs]).mean()
    #        logs = {'test_loss': avg_loss, 'test_mse_loss': avg_mse_loss, 'test_nll_loss': avg_nll_loss}

    #        return {'log': logs}
    #    else:
    #        return super().test_epoch_end(outputs)

    #to help debug XLA stuff, like missing ops, or data loading/compiling bottlenecks
    # see https://youtu.be/iwtpwQRdb3Y?t=1056
    # def on_epoch_end(self):
    #    import torch_xla.core.xla_model as xm
    #    import torch_xla.debug.metrics as met
    #    xm.master_print(met.metrics_report())


    #def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx,
    #                           optimizer_closure, on_tpu, using_native_amp, using_lbfgs):
    #    optimizer.zero_grad()
