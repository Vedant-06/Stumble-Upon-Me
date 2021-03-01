import config
import torch


class XMLRDataset:
    def __init__(self, boilerplate, label = None):
        self.boilerplate = boilerplate
        self.label = label
        self.tokenizer = config.TOKENIZER
        self.max_len = config.MAX_LEN

    def __len__(self):
        return len(self.boilerplate)

    def __getitem__(self, item):
        boilerplate = str(self.boilerplate[item])
        boilerplate = " ".join(boilerplate.split())

        inputs = self.tokenizer.encode_plus(
            boilerplate,
            None,
            padding='max_length',
            add_special_tokens=True,
            max_length=self.max_len,
            truncation=True,
            return_token_type_ids=True
        )

        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]
        token_type_ids = inputs["token_type_ids"]

        return {
            "ids": torch.tensor(ids, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "targets": torch.tensor(self.label[item], dtype=torch.float),
        }
