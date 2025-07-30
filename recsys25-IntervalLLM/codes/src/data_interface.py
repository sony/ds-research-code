import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from .data import Data

MAX_LENGTH = 768

class TrainCollater:
    def __init__(self,
                 prompt_list = None,
                 llm_tokenizer = None,
                 train = False,
                 input_prompt = "item_only",
                 interval_infused_attention = 0,
                 pretrained_rec_used = 0,
                 max_step = 1):
        self.prompt_list = prompt_list
        self.llm_tokenizer = llm_tokenizer
        self.train = train
        self.max_step = max_step
        self.input_prompt = input_prompt
        self.interval_infused_attention = interval_infused_attention
        self.pretrained_rec_used = pretrained_rec_used
        
        self.system_message = "<s>This user has purchased: "
        self.tokenized_system_message = self.llm_tokenizer(self.system_message, return_tensors = "pt", add_special_tokens = False)
        self.len_system_message = len(self.tokenized_system_message.input_ids[0])

    def customized_truncation(self, inputs_pair):
        max_content_length = MAX_LENGTH - self.len_system_message

        input_ids, attention_masks, token_type_ids = [], [], []
        for i, pair in enumerate(inputs_pair):
            pair_tokens = self.llm_tokenizer([pair], 
                                            return_tensors="pt", 
                                            add_special_tokens=False,
                                            return_attention_mask=True,
                                            return_token_type_ids=True)
            
            tokens = pair_tokens.input_ids[0]
            attention_mask = pair_tokens.attention_mask[0]
            token_type_id = pair_tokens.token_type_ids[0]
            if tokens.shape[0] > max_content_length:
                # Truncate the input (left truncation)
                truncated_seq = tokens[-max_content_length:]
                attention_mask = attention_mask[-max_content_length:]
                token_type_id = token_type_id[-max_content_length:]
                tokens = truncated_seq.clone()
            else:
                tokens = torch.cat((tokens, torch.full((max_content_length - tokens.shape[0],), self.llm_tokenizer.pad_token_id, dtype=torch.long)), dim=0)
                attention_mask = torch.cat((attention_mask, torch.zeros(max_content_length - attention_mask.shape[0], dtype=torch.long)), dim=0)
                token_type_id = torch.cat((token_type_id, torch.zeros(max_content_length - token_type_id.shape[0], dtype=torch.long)), dim=0)
            
            # concat truncated sequence with system message
            input_ids.append(torch.cat((self.tokenized_system_message.input_ids[0], tokens), dim=0))
            attention_masks.append(torch.cat((torch.ones(self.len_system_message), attention_mask), dim=0))
            token_type_ids.append(torch.cat((torch.zeros(self.len_system_message), token_type_id), dim=0))
        
        input_ids = torch.stack(input_ids, dim=0)
        attention_masks = torch.stack(attention_masks, dim=0)
        token_type_ids = torch.stack(token_type_ids, dim=0)
                
        return {"input_ids": input_ids, "attention_mask": attention_masks, "token_type_ids": token_type_ids}

    def __call__(self, batch):
        inputs_text = [self.prompt_list] * len(batch)
        can_id_name_dict = []

        if self.input_prompt == "item_only" or "item_interval":
            for i, sample in enumerate(batch):
                input_text = inputs_text[i]
                if "[HistoryHere]" in input_text:
                    insert_prompt = ""
                    for j, seq_title in enumerate(sample["seq_name"]):
                        if self.interval_infused_attention or self.pretrained_rec_used:
                            insert_prompt = insert_prompt + "[PH] " + seq_title + " [ATT] [/PH] "
                        else:
                            insert_prompt = insert_prompt + "[PH] " + seq_title + " [/PH] "
                        if j != (len(sample["seq_name"])-1) and self.input_prompt == "item_interval":
                            insert_prompt = insert_prompt + ", and after [TS] " + str(sample["diff_date"][j]) +  " [TSEmb] [/TS] days purchased "
                    input_text = input_text.replace("[HistoryHere]", insert_prompt)
                if "[CansHere]" in input_text:
                    can_prompt, can_dict = "", {}
                    option_id_list = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T"]
                    for id, can in zip(option_id_list, sample["cans_name"]):
                        if self.pretrained_rec_used:
                            can_prompt = can_prompt + id + ". " + can + "[CAND]" + "\n"
                        else:
                            can_prompt = can_prompt + id + ". " + can + "\n"
                        can_dict[can] = id
                    input_text = input_text.replace("[CansHere]", can_prompt) 
                    can_id_name_dict.append(can_dict)   
                inputs_text[i] = input_text
        else:
            raise NotImplementedError(f"Input prompt {self.input_prompt} not implemented")
                
        targets_text = [sample["correct_answer"] for sample in batch]

        if self.train:
            targets_id = []
            for i, target_text in enumerate(targets_text):
                targets_id.append(f"{can_id_name_dict[i][target_text]}</s>")
            inputs_pair = [[p, t] for p, t in zip(inputs_text, targets_id)]
            
            # customized truncation for inputs_pair
            batch_tokens = self.customized_truncation(inputs_pair)

            seq_name_tokens = []
            for sample in batch:
                seq_name = sample["seq_name"]
                seq_name_token = self.llm_tokenizer(
                    seq_name,
                    return_tensors="pt",
                    padding="max_length", # longest
                    truncation=True,
                    add_special_tokens=False,
                    return_attention_mask=True,
                    return_token_type_ids=False,
                    max_length=MAX_LENGTH)
                
                if torch.all(seq_name_token["input_ids"][0] == 32000):
                    print("[ALERT] Targets all 32000")
                seq_name_tokens.append(seq_name_token)
            
            stacked_diff_date = self.compute_temporal_difference(batch)
            
            new_batch = {"tokens": batch_tokens,
                        "seq_name": seq_name_tokens,
                        "cans": torch.stack([torch.tensor(sample["cans"]) for sample in batch], dim=0),
                        "len_seq": torch.stack([torch.tensor(sample["len_seq"]) for sample in batch], dim=0),
                        "interval_seq": stacked_diff_date,
                        "seq": self.stack_with_padding([torch.tensor(sample["seq"]) for sample in batch], padding_value=0),
                        }
        else:
            targets_id = []
            for i, target_text in enumerate(targets_text):
                targets_id.append(f"{can_id_name_dict[i][target_text]}")

            # customized truncation for inputs_pair
            batch_tokens = self.customized_truncation(inputs_text)

            seq_name_tokens = []
            for sample in batch:
                seq_name = sample["seq_name"]
                seq_name_token = self.llm_tokenizer(
                    seq_name,
                    return_tensors="pt",
                    padding="max_length", # longest
                    truncation=True,
                    add_special_tokens=False,
                    return_attention_mask=True,
                    return_token_type_ids=True,
                    max_length=MAX_LENGTH)
                seq_name_tokens.append(seq_name_token)

            stacked_diff_date = self.compute_temporal_difference(batch)
            cans_name = [sample["cans_name"] for sample in batch]

            new_batch = {"tokens": batch_tokens,
                        "seq_name": seq_name_tokens,
                        "cans": torch.stack([torch.tensor(sample["cans"]) for sample in batch], dim=0),
                        "len_seq": torch.stack([torch.tensor(sample["len_seq"]) for sample in batch], dim=0),
                        "interval_seq": stacked_diff_date,
                        "correct_answer": targets_id,
                        "cans_name": cans_name,
                        "seq": self.stack_with_padding([torch.tensor(sample["seq"]) for sample in batch], padding_value=0),
                        }
        return new_batch
    
    def compute_temporal_difference(self, batch):
        tensor_diff_date = []
        for sample in batch:
            # prepend 0 for interval-infused attention as the first item has no interval
            tensor_diff_date.append(torch.log1p(torch.tensor([0] + sample["diff_date"])))
        return self.stack_with_padding(tensor_diff_date)
    
    @staticmethod
    def stack_with_padding(list_of_tensors, padding_value=-1, padding_side="right"):
        """
        Stack a list of tensors with padding on one side
        Args:
            list_of_tensors (list[torch.Tensor]): List of tensors to stack
            padding_value (int, optional): Value to pad with. Defaults to 0.
            padding_side (str, optional): Side to pad on. Defaults to "right".
        Returns:
            torch.Tensor: Stacked tensors
        """
        max_tokens = max(tensor.size(0) for tensor in list_of_tensors)
        padded_tensors = []
        for tensor in list_of_tensors:
            num_tokens = tensor.size(0)
            if len(tensor.size()) == 1:
                padding = torch.full(
                    (max_tokens - num_tokens,),
                    padding_value,
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
            padded_tensor = (
                torch.cat((tensor, padding), dim=0)
                if padding_side == "right"
                else torch.cat((padding, tensor), dim=0)
            )
            padded_tensors.append(padded_tensor)
        return torch.stack(padded_tensors)

class DInterface(pl.LightningDataModule):
    def __init__(self, 
                 llm_tokenizer,
                 num_workers,
                 batch_size,
                 max_epochs,
                 prompt_path,
                 data_dir,
                 cans_num,
                 input_prompt,
                 padding_item_id,
                 interval_infused_attention,
                 pretrained_rec_used):
        super().__init__()
        self.llm_tokenizer = llm_tokenizer
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.prompt_path = prompt_path
        self.input_prompt = input_prompt
        self.interval_infused_attention = interval_infused_attention
        self.pretrained_rec_used = pretrained_rec_used
        self.load_prompt(self.prompt_path)

        self.trainset = Data(data_dir=data_dir, stage="train", cans_num=cans_num, padding_item_id=padding_item_id)
        self.valset = Data(data_dir=data_dir, stage="val", cans_num=cans_num, padding_item_id=padding_item_id)
        self.testset = Data(data_dir=data_dir, stage="test", cans_num=cans_num, padding_item_id=padding_item_id)

        self.max_steps = self.max_epochs * (len(self.trainset)//self.batch_size)

    def train_dataloader(self):
        return DataLoader(self.trainset,
                          batch_size=self.batch_size, 
                          num_workers=self.num_workers, 
                          shuffle=True,
                          drop_last=True,
                          collate_fn=TrainCollater(prompt_list=self.prompt_list, llm_tokenizer=self.llm_tokenizer, train=True, input_prompt=self.input_prompt, interval_infused_attention=self.interval_infused_attention, pretrained_rec_used=self.pretrained_rec_used, max_step=self.max_steps))

    def val_dataloader(self):
        return DataLoader(self.valset, 
                          batch_size=self.batch_size, 
                          num_workers=self.num_workers, 
                          shuffle=False,
                          collate_fn=TrainCollater(prompt_list=self.prompt_list, llm_tokenizer=self.llm_tokenizer, train=False, input_prompt=self.input_prompt, interval_infused_attention=self.interval_infused_attention, pretrained_rec_used=self.pretrained_rec_used))

    def test_dataloader(self):
        return DataLoader(self.testset, 
                          batch_size=self.batch_size, 
                          num_workers=self.num_workers, 
                          shuffle=False,
                          collate_fn=TrainCollater(prompt_list=self.prompt_list, llm_tokenizer=self.llm_tokenizer, train=False, input_prompt=self.input_prompt, interval_infused_attention=self.interval_infused_attention, pretrained_rec_used=self.pretrained_rec_used))
    
    def load_prompt(self, prompt_path):
        with open(prompt_path, "r") as f:
            self.prompt_list = f.read()