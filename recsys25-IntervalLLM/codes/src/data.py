import os.path as op
import torch.utils.data as data

import pandas as pd
import random
import ast
import logging

logger = logging.getLogger(__name__)


class Data(data.Dataset):
    def __init__(self, data_dir,
                 stage,
                 cans_num,
                 padding_item_id,
                 sep = ", "):
        self.data_dir = data_dir
        self.stage = stage
        self.cans_num = cans_num
        self.sep = sep
        self.padding_item_id = padding_item_id
        self.check_files()

        # load option file from previous predictions to align the results
        if stage == "test":
            ## Video Games
            test_option_file = pd.read_csv("./IntervalLLM/outputs/IntervalLLM-Video_Games/test.csv", index_col=0)
            ## CD
            # test_option_file = pd.read_csv("./IntervalLLM/outputs/IntervalLLM-CD/test.csv", index_col=0)
            ## Books
            # test_option_file = pd.read_csv("./IntervalLLM/outputs/IntervalLLM-Books/test.csv", index_col=0)
            test_option_file["cans"] = test_option_file["cans"].apply(ast.literal_eval)
            self.test_option_file = test_option_file["cans"].to_list()

    def __len__(self):
        return len(self.session_data["seq"])

    def __getitem__(self, i):
        temp = self.session_data.iloc[i]
        candidates = self.negative_sampling(temp["seq"], temp["next"], i)
        cans_name = [self.item_id2name[can] for can in candidates]
        sample = {
            "seq": temp["seq"],
            "seq_name": temp["seq_title"],
            "len_seq": temp["len_seq"],
            "seq_str": self.sep.join(temp["seq_title"]),
            "cans": candidates,
            "cans_name": cans_name,
            "cans_str": self.sep.join(cans_name),
            "item_id": temp["next"],
            "item_name": temp["next_item_name"],
            "correct_answer": temp["next_item_name"],
            "date": temp["date"],
            "diff_date": temp["diff_date"]
        }
        return sample
    
    def negative_sampling(self, seq, next_item, index):
        if self.stage != "test":
            canset = [i for i in list(self.item_id2name.keys()) if i not in seq and i != next_item]
            candidates = random.sample(canset, self.cans_num-1) + [int(next_item)]
            random.shuffle(candidates)
        else:
            candidates_name = self.test_option_file[index]
            candidates = []
            for k in candidates_name:
                candidate_name = k.strip()
                if candidate_name == "":
                    logger.info(f"Error in negative sampling for {index}: {candidates_name}")
                    logger.info(f"Next item: {next_item}")
                    candidate_name = "Video Games"
                candidates.append(int(self.item_name2id[candidate_name]))
        return candidates

    def check_files(self):
        self.item_id2name = self.get_id2name()
        data_path = op.join(self.data_dir, f"{self.stage}.df")
        self.session_data = self.session_data4frame(data_path)

        # need to have a name2id mapping for test
        if self.stage == "test":
            self.item_name2id = self.get_name2id()

    def get_name2id(self):
        name2id = dict()
        item_path = op.join(self.data_dir, "id2name.txt")
        with open(item_path, "r") as f:
            for l in f.readlines():
                ll = l.strip("\n").split("::")
                if len(ll[1]) == 0:
                    name2id["empty"] = int(ll[0])
                else:
                    name2id[ll[1].strip()] = int(ll[0])
        return name2id
        
    def get_id2name(self):
        id2name = dict()
        item_path = op.join(self.data_dir, "id2name.txt")
        with open(item_path, "r") as f:
            for l in f.readlines():
                ll = l.strip("\n").split("::")
                id2name[int(ll[0])] = ll[1].strip()
                # Use to check if there is any empty name
                if len(ll[1]) == 0:
                    logger.info(f"no name for {ll[0]}")
        return id2name
    
    def session_data4frame(self, datapath):
        train_data = pd.read_pickle(datapath)
        train_data = train_data[train_data["len_seq"] >= 2]

        def seq_to_name(x): 
            return [self.item_id2name[x_i] for x_i in x]
        train_data["seq_title"] = train_data["seq"].apply(seq_to_name)
        
        def next_item_name(x): 
            return self.item_id2name[x]
        train_data["next_item_name"] = train_data["next"].apply(next_item_name)
        
        def diff_date_to_int(x):
            return list(map(int, x))
        train_data["diff_date"] = train_data["diff_date"].apply(diff_date_to_int)

        return train_data