import torch
import logging
from torch import nn
from torch.nn import functional as F

import pytorch_lightning as pl
from pytorch_lightning.utilities.memory import garbage_collection_cuda

from transformers import LlamaForCausalLM, AutoTokenizer
from pandas.core.frame import DataFrame
import os.path as op
import os
from optims import LinearWarmupCosineLRScheduler
from peft import get_peft_model, LoraConfig, TaskType
from typing import Optional
from .sasrec import SASRec

logger = logging.getLogger(__name__)


class IntervalInfusedAttention(nn.Module):
    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.temperature = d_k**-0.5
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        attn = torch.matmul(q*self.temperature, k.transpose(2, 3))

        if mask is not None:
            attn = attn.masked_fill(mask.to(torch.bool), torch.finfo(attn.dtype).min)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        output = output.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        output = self.fc(output)

        return output, attn


class TemporalProjector(nn.Module):
    def __init__(self, llm_size):
        super().__init__()
        self.temporal_mlp_proj = nn.Sequential(
            nn.Linear(1, llm_size//4),
            nn.GELU(),
            nn.Linear(llm_size//4, llm_size),
            nn.LayerNorm(llm_size)
        )

    def forward(self, x):
        return self.temporal_mlp_proj(x)
    

class ItemImbedder(nn.Module):
    def __init__(self, rec_size=64, llm_size=4096):
        super().__init__()
        self.mlp_proj = nn.Sequential(
            nn.Linear(rec_size, llm_size),
            nn.GELU(),
            nn.Linear(llm_size, llm_size)
        )

    def forward(self, x):
        x = self.mlp_proj(x)
        return x


class MInterface(pl.LightningModule):
    def __init__(self,
                 lr,
                 llm_path,
                 llm_tuning,
                 output_dir,
                 weight_decay,
                 lr_scheduler,
                 lr_decay_min_lr,
                 save,
                 lora_r,
                 lora_alpha,
                 lora_dropout,
                 lr_warmup_start_lr,
                 temporal_projector_config,
                 interval_infused_attention_config,
                 pretrained_item_config
                 ):
        super().__init__()
        self.save_hyperparameters()
        self.llm_tuning = llm_tuning
        self.output_dir = output_dir
        self.weight_decay = weight_decay
        self.lr = lr
        self.lr_scheduler = lr_scheduler
        self.lr_decay_min_lr = lr_decay_min_lr
        self.save = save
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.lr_warmup_start_lr = lr_warmup_start_lr
        self.pretrained_rec_used = pretrained_item_config.pretrained_rec_used
        self.temporal_projector_config = temporal_projector_config
        self.interval_infused_attention = interval_infused_attention_config.interval_infused_attention

        self.load_llm(llm_path)

        if self.pretrained_rec_used:
            self.load_rec_model(pretrained_item_config)

        if temporal_projector_config:
            self.temporal_projector = TemporalProjector(self.llama_model.config.hidden_size)

        if self.interval_infused_attention:
            self.interval_attention = IntervalInfusedAttention(
                n_head=interval_infused_attention_config.n_head, 
                d_model=interval_infused_attention_config.d_model, 
                d_k=interval_infused_attention_config.d_k, 
                d_v=interval_infused_attention_config.d_v,
                dropout=interval_infused_attention_config.dropout
            )
        if self.interval_infused_attention or self.pretrained_rec_used:
            self.MAX_ITEM_NUM = 1024

    def forward(self, batch):
        targets = batch["tokens"]["input_ids"].masked_fill(
            batch["tokens"]["input_ids"] == self.llama_tokenizer.pad_token_id, -100
        )
        targets = targets.masked_fill((batch["tokens"]["token_type_ids"] == 0), -100)

        input_embeds = self.wrap_emb(batch)

        outputs = self.llama_model(
            inputs_embeds=input_embeds,
            attention_mask=batch["tokens"]["attention_mask"],
            return_dict=True,
            labels=targets,
            use_cache=False
        )
        return outputs

    def generate(self, batch, do_sample=False, num_beams=1, max_gen_length=2, min_gen_length=1, repetition_penalty=1.0, length_penalty=1.0, num_return_sequences=1):
        input_embeds = self.wrap_emb(batch)
        generate_ids = self.llama_model.generate(
            inputs_embeds=input_embeds,
            attention_mask=batch["tokens"]["attention_mask"],
            do_sample=do_sample,
            num_beams=num_beams,
            max_new_tokens=max_gen_length,
            min_new_tokens=min_gen_length,
            pad_token_id=self.llama_tokenizer.pad_token_id,
            eos_token_id=self.llama_tokenizer.eos_token_id,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            num_return_sequences=num_return_sequences
            )
        output_text = self.llama_tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        outputs = [text.strip() for text in output_text]
        return outputs

    def training_step(self, batch, batch_idx):
        out = self(batch)
        loss = out.loss

        self.scheduler.step(cur_step=self.trainer.global_step, cur_epoch=self.current_epoch, max_step=self.trainer.max_steps)

        if torch.isnan(loss):
            logger.info(f"batch index is {batch_idx}")
            logger.info(out.loss)
            logger.info(out.logits)
            logger.info("-------")

        self.log("loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("lr", self.scheduler.optimizer.param_groups[0]["lr"], on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("global_step_num", self.trainer.global_step, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        garbage_collection_cuda()
        return loss
    
    def on_train_start(self):
        # Move it manually to the device since Lightening only detects the llama model
        if self.temporal_projector_config:
            self.temporal_projector.to(self.device)
        if self.interval_infused_attention:
            self.interval_attention.to(self.device)
        if self.pretrained_rec_used:
            self.rec_model.to(self.device)
            self.item_embedder.to(self.device)
    
    def on_validation_epoch_start(self):
        self.val_content = {
            "generate" : [],
            "real" : [],
            "cans" : [],
        }

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        generate_output = self.generate(batch)
        output = []
        for i, generate in enumerate(generate_output):
            real = batch["correct_answer"][i]
            cans = batch["cans_name"][i]
            generate = generate.strip().split("\n")[0]
            output.append((generate, real, cans))
        return output

    def on_validation_batch_end(self, outputs, batch, batch_idx, dataloader_idx):
        for generate, real, cans in outputs:
            self.val_content["generate"].append(generate)
            self.val_content["real"].append(real)
            self.val_content["cans"].append(cans)

    def on_validation_epoch_end(self):
        df = DataFrame(self.val_content)
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        df.to_csv(op.join(self.output_dir, "valid.csv"))
        prediction_valid_ratio, hr = self.calculate_hr1(self.val_content)
        metric = hr
        self.log("val_prediction_valid", prediction_valid_ratio, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val_hr", hr, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("metric", metric, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

    def on_test_epoch_start(self):
        self.test_content = {
            "generate" : [],
            "real" : [],
            "cans" : [],
        }

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        generate_output = self.generate(batch)
        output = []
        for i,generate in enumerate(generate_output):
            real = batch["correct_answer"][i]
            cans = batch["cans_name"][i]
            generate = generate.strip().split("\n")[0]
            output.append((generate, real, cans))

        return output
    
    def on_test_batch_end(self, outputs, batch, batch_idx, dataloader_idx):
        for generate, real, cans in outputs:
            self.test_content["generate"].append(generate)
            real = real.cpu().item() if isinstance(real, torch.Tensor) else real
            cans = cans.cpu().item() if isinstance(cans, torch.Tensor) else cans

            self.test_content["real"].append(real)
            self.test_content["cans"].append(cans)
        torch.cuda.empty_cache()

    def on_test_epoch_end(self):
        df = DataFrame(self.test_content)
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        df.to_csv(op.join(self.output_dir, "test.csv"))
        prediction_valid_ratio, hr = self.calculate_hr1(self.test_content)
        metric = hr * prediction_valid_ratio
        self.log("test_prediction_valid", prediction_valid_ratio, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("test_hr", hr, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("metric", metric, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        logger.info(f"Test HR@1: {hr} | Test Prediction Valid Ratio: {prediction_valid_ratio} | Test Metric: {metric}")

    def configure_optimizers(self):
        lora_params = filter(lambda p: p.requires_grad, self.llama_model.parameters())

        optimizer = torch.optim.AdamW(
            list(lora_params) + list(self.temporal_projector.parameters() if self.temporal_projector_config else []) + list(self.interval_attention.parameters() if self.interval_infused_attention else []),
            lr=self.lr,
            weight_decay=self.weight_decay
        )

        max_step = self.trainer.max_steps
        warmup_steps = max_step // 20
        logger.info(f"max_step: {max_step} | warmup_steps: {warmup_steps}")
        if self.temporal_projector_config:
            logger.info(f"Additional trainable parameters of temporal projector: {sum(p.numel() for p in self.temporal_projector.parameters() if p.requires_grad)}")
        if self.interval_infused_attention:
            logger.info(f"Additional trainable parameters of interval infused attention: {sum(p.numel() for p in self.interval_attention.parameters() if p.requires_grad)}")
        if self.lr_scheduler == "cosine":
            self.scheduler = LinearWarmupCosineLRScheduler(optimizer,
                                                max_step=max_step,
                                                min_lr=self.lr_decay_min_lr,
                                                init_lr=self.lr,
                                                warmup_steps=warmup_steps,
                                                warmup_start_lr=self.lr_warmup_start_lr)
        else:
            raise NotImplementedError("Current only support cosine lr scheduler")

        return optimizer
        
    def on_save_checkpoint(self, checkpoint):
        if self.save == "part":
            checkpoint.pop("optimizer_states")
            to_be_removed = []
            for key, value in checkpoint["state_dict"].items():
                try:
                    if not self.get_parameter(key).requires_grad:
                        to_be_removed.append(key)
                except AttributeError:
                    to_be_removed.append(key)
            for key in to_be_removed:
                checkpoint["state_dict"].pop(key)
        elif self.save == "all":
            pass

    def load_llm(self, llm_path):
        self.llama_tokenizer = AutoTokenizer.from_pretrained(llm_path)
        self.llama_tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        self.llama_tokenizer.padding_side = "right"
        if self.temporal_projector_config:
            self.llama_tokenizer.add_special_tokens({"additional_special_tokens": ["[TSEmb]"]})
            logger.info(f"Add additional special tokens: {['[TSEmb]']}")
        if self.interval_infused_attention or self.pretrained_rec_used:
            self.llama_tokenizer.add_special_tokens({"additional_special_tokens": ["[ATT]"]})
            logger.info(f"Add additional special tokens: {['[ATT]']}")
            if self.pretrained_rec_used:
                self.llama_tokenizer.add_special_tokens({"additional_special_tokens": ["[CAND]"]})
                logger.info(f"Add additional special tokens: {['[CAND]']}")
        self.llama_model = LlamaForCausalLM.from_pretrained(llm_path, torch_dtype = torch.float32)
        self.llama_model.resize_token_embeddings(len(self.llama_tokenizer))
        if self.llm_tuning == "lora":
            peft_config = LoraConfig(task_type = TaskType.CAUSAL_LM,
                                        inference_mode = False,
                                        r = self.lora_r,
                                        lora_alpha = self.lora_alpha,
                                        lora_dropout = self.lora_dropout,
                                        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
            self.peft_config = peft_config
            self.llama_model = get_peft_model(self.llama_model, peft_config)
            self.llama_model.print_trainable_parameters()
        else:
            for param in self.llama_model.parameters(): 
                param.requires_grad = False
        logger.info("Loading LLAMA Done")

    def load_rec_model(self, pretrained_item_config):
        self.rec_model = SASRec(pretrained_item_config.user_num, pretrained_item_config.item_num, pretrained_item_config)
        _, checkpoint = torch.load(pretrained_item_config.rec_path, map_location="cpu")
        self.rec_model.load_state_dict(checkpoint)
        self.rec_model.eval()
        for name, param in self.rec_model.named_parameters():
            param.requires_grad = False

        self.item_embedder = ItemImbedder()
        logger.info("Loading Rec Model Done")

    @staticmethod
    def _make_bidirectional_exclude_padding_mask(
        input_ids_shape: torch.Size,
        attention_mask_2d: torch.Tensor,
        dtype: torch.dtype,
        device: torch.device,
    ):
        """
        Make causal mask used for bi-directional self-attention.
        [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        """
        bsz, tgt_len = input_ids_shape
        mask = torch.full((tgt_len, tgt_len), 0, device=device).to(dtype)      # [NOTE]: torch.finfo(dtype).min -> 0
        mask = mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len)

        expanded_mask = attention_mask_2d[:, None, None, :].expand(bsz, 1, tgt_len, tgt_len).to(dtype)
        inverted_mask = 1.0 - expanded_mask
        expanded_attn_mask = inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)
        expanded_attn_mask = mask.masked_fill(expanded_attn_mask.bool(), 1)
        return expanded_attn_mask

    @staticmethod
    def stack_with_padding(list_of_tensors, padding_value=-1, padding_side="right"):
        max_tokens = max(tensor.size(0) for tensor in list_of_tensors)
        padded_tensors, att_tensors = [], []
        for tensor in list_of_tensors:
            num_tokens = tensor.size(0)
            if len(tensor.size()) == 1:
                padding = torch.full(
                    (max_tokens - num_tokens,),
                    padding_value,
                    dtype=tensor.dtype,
                    device=tensor.device,
                )
                att = torch.full(
                    (num_tokens,),
                    1,
                    dtype=tensor.dtype,
                    device=tensor.device,
                )
            else:
                padding = torch.full(
                    (max_tokens - num_tokens, tensor.size(1)),
                    padding_value,
                    dtype=tensor.dtype,
                    device=tensor.device,
                )
                att = torch.full(
                    (num_tokens,),
                    1,
                    dtype=tensor.dtype,
                    device=tensor.device,
                )
            padded_tensor = (
                torch.cat((tensor, padding), dim=0)
                if padding_side == "right"
                else torch.cat((padding, tensor), dim=0)
            )
            att_padding = torch.full(
                    (max_tokens - num_tokens,),
                    0,
                    dtype=tensor.dtype,
                    device=tensor.device,
            )
            att_tensor = (
                torch.cat((att, att_padding), dim=0)
                if padding_side == "right"
                else torch.cat((att_padding, att), dim=0)
            ) 
            padded_tensors.append(padded_tensor)
            att_tensors.append(att_tensor)
        return torch.stack(padded_tensors), torch.stack(att_tensors)

    def replace_special_token_with_embeddings(self, batch, input_embeds, replaced_embeds, special_token, candidate=False):
        # Fetch token IDs for special tokens
        special_token_id = self.llama_tokenizer(special_token, return_tensors="pt", add_special_tokens=False).input_ids.item()

        # Find the indeces of the special tokens from batch["tokens"]["input_ids"]
        special_token_indices = torch.where(batch["tokens"]["input_ids"]==special_token_id)
        
        # Reverse the elements
        special_token_indices = (special_token_indices[0].flip(0), special_token_indices[1].flip(0))

        unique_row_indices = torch.unique(special_token_indices[0])
        current_row_idx = unique_row_indices[-1].item()

        # if candidate, always be the number of candidates since all of them are preserved
        if candidate:
            current_col_idx = 20 - 1
        else:
            # -1 for 0-index
            current_col_idx = batch["len_seq"][current_row_idx] - 1
        
        # need to truncate if exceeds self.MAX_ITEM_NUM
        if current_col_idx >= self.MAX_ITEM_NUM:
            current_col_idx = self.MAX_ITEM_NUM - 1

        for idx, (row_idx, col_idx) in enumerate(zip(special_token_indices[0], special_token_indices[1])):
            if row_idx == current_row_idx:
                # If the current row index is the same as the previous row index, replace input_embeds and decrement the column index
                assert current_col_idx >= 0
                # Replace the input embedding with the corresponding embedding
                input_embeds[row_idx, col_idx] = replaced_embeds[current_row_idx, current_col_idx]
                current_col_idx -= 1
            else:
                # If the current row index is different from the previous row index, update the current row index and column index
                current_row_idx = row_idx
                if candidate:
                    current_col_idx = 20 - 1
                else:
                    # -1 for 0-index
                    current_col_idx = batch["len_seq"][current_row_idx] - 1
                # need to truncate if exceeds self.MAX_ITEM_NUM
                if current_col_idx >= self.MAX_ITEM_NUM:
                    current_col_idx = self.MAX_ITEM_NUM - 1
                # Replace the input embedding with the corresponding embedding
                input_embeds[row_idx, col_idx] = replaced_embeds[current_row_idx, current_col_idx]
                current_col_idx -= 1

        return input_embeds
    
    def encode_items(self, seq_ids):
        seq_ids_embeds = self.rec_model.item_emb(seq_ids)
        seq_ids_embeds = self.item_embedder(seq_ids_embeds)
        return seq_ids_embeds

    def wrap_emb(self, batch):
        input_embeds = self.llama_model.get_input_embeddings()(batch["tokens"]["input_ids"])

        if self.temporal_projector_config:
            if batch["interval_seq"].size(1) > self.MAX_ITEM_NUM:
                logger.info(f"Truncating the interval sequence from {batch['interval_seq'].size(1)} to {self.MAX_ITEM_NUM}")
                batch["interval_seq"] = batch["interval_seq"][:, -self.MAX_ITEM_NUM:]

            # replace [TSEmb] wtih the corresponding temporal embeddings
            replaced_temporal_embeds = self.temporal_projector(batch["interval_seq"].unsqueeze(-1).float())
            input_embeds = self.replace_special_token_with_embeddings(batch, input_embeds, replaced_temporal_embeds, "[TSEmb]")

        # replace [ATT] and [CAND] wtih the corresponding item embeddings for LLaRA
        if self.pretrained_rec_used:
            history_item_embeds = self.encode_items(batch["seq"])
            input_embeds = self.replace_special_token_with_embeddings(batch, input_embeds, history_item_embeds, "[ATT]")

            candidate_item_embeds = self.encode_items(batch["cans"])
            input_embeds = self.replace_special_token_with_embeddings(batch, input_embeds, candidate_item_embeds, "[CAND]", candidate=True)

        # replace [ATT] wtih the corresponding interval infused attention
        if self.interval_infused_attention:
            # fetch item embeddings
            seq_name_embeds = []
            for i, sample in enumerate(batch["seq_name"]):
                # truncate to 1k if the numbers of item is too long
                if sample["input_ids"].size(0) > self.MAX_ITEM_NUM:
                    sample["input_ids"] = sample["input_ids"][-self.MAX_ITEM_NUM:, :]
                    sample["attention_mask"] = sample["attention_mask"][-self.MAX_ITEM_NUM:, :]
                seq_name_embed = self.llama_model.get_input_embeddings()(sample["input_ids"])
                positions_for_non_padding = sample["attention_mask"] * torch.ones(seq_name_embed.shape[:2], device=input_embeds.device)
                
                sum_embeddings = torch.sum(seq_name_embed*positions_for_non_padding.unsqueeze(-1), dim=1)
                num_of_none_padding_tokens = torch.sum(positions_for_non_padding, dim=-1).unsqueeze(-1)
                seq_name_embed = sum_embeddings / num_of_none_padding_tokens

                seq_name_embeds.append(seq_name_embed)

            seq_name_embeds, attention_mask_2d = self.stack_with_padding(seq_name_embeds)
            batch_size, sequence_len, _ = seq_name_embeds.shape
            causal_4d_mask = self._make_bidirectional_exclude_padding_mask((batch_size, sequence_len), attention_mask_2d, torch.float32, device=attention_mask_2d.device)
                
            replaced_attn_embeds, _ = self.interval_attention(replaced_temporal_embeds, seq_name_embeds, seq_name_embeds, causal_4d_mask)

            input_embeds = self.replace_special_token_with_embeddings(batch, input_embeds, replaced_attn_embeds, "[ATT]")

        return input_embeds
     
    def calculate_hr1(self, eval_content):
        correct_num, valid_num, total_num = 0, 0, len(eval_content["generate"])
        cans = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T"]
        for generate, real in zip(eval_content["generate"], eval_content["real"]):
            generate = generate.strip()

            if generate in cans:
                valid_num += 1
                if real == generate:
                    correct_num += 1

        valid_ratio = valid_num/total_num
        hr1 = correct_num / valid_num if valid_num > 0 else 0

        return valid_ratio, hr1