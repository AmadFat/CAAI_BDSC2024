
import torch
import lightning
import torch.nn as nn
from skip_gram import predictor
from torch.utils import data as data
from torch.utils.data import Dataset, DataLoader
from raw2idx import get_dataset, get_cleaned_reviews, Vocab, get_idx_with_voc


class EmbeddingWorker(lightning.LightningModule):
    def __init__(self, backbone, train_set, test_set, half_size):
        super().__init__()
        self.backbone = backbone
        self.half_size = half_size
        self.loss_fn = nn.CrossEntropyLoss(reduction='mean')
        self.register_buffer('train_set', train_set)
        self.register_buffer('test_set', test_set)

        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        x = self.train_set[batch]
        y = []
        for idx in batch:
            a, b = torch.arange(idx-self.half_size, idx), torch.arange(idx+1, idx+self.half_size+1)
            # print(a.shape, b.shape)
            eager_idxs = torch.cat((self.train_set[a], self.train_set[b]), dim=0).tolist()
            # print(eager_idxs, eager_idxs.shape)
            y.append(eager_idxs)
        y = torch.tensor(y, dtype=torch.long).cuda().detach()
        y_hat = self.backbone(x)
        y = y.view(-1)
        y_hat = y_hat.view(-1, y_hat.shape[-1])
        loss = self.loss_fn(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=3e-4, weight_decay=1e-4)
        return optimizer


class REVIEW_DATASET(Dataset):
    def __init__(self, reviews):
        self.reviews = [torch.tensor(review) for review in reviews]

    def __getitem__(self, index):
        return self.reviews[index]

    def __len__(self):
        return len(self.reviews)


def main():
    seed = torch.Generator().manual_seed(3407)
    embedding_dim = 64
    half_window_size = 2
    valid_ratio = 0.2
    batch_size = 512
    min_freq = 10

    dataset_raw_labeled_train, dataset_raw_test = get_dataset()
    reviews_cleaned_train = get_cleaned_reviews(dataset_raw_labeled_train)
    reviews_cleaned_test = get_cleaned_reviews(dataset_raw_test)
    # sentiments_train = [sentiment for sentiment in dataset_raw_labeled_train.sentiment]
    voc = Vocab(reviews_cleaned_train + reviews_cleaned_test, min_freq=min_freq, reserved_tokens=['<pad>'])
    reviews_discrete_train = get_idx_with_voc(voc, reviews_cleaned_train)
    reviews_discrete_test = get_idx_with_voc(voc, reviews_cleaned_test)
    reviews_discrete_train = [item for review in reviews_discrete_train for item in [voc['<pad>']] * half_window_size +
                              review + [voc['<pad>']] * half_window_size]
    reviews_discrete_test = [item for review in reviews_discrete_test for item in [voc['<pad>']] * half_window_size +
                             review + [voc['<pad>']] * half_window_size]
    idxs_train = REVIEW_DATASET(torch.arange(half_window_size, len(reviews_discrete_train) + half_window_size))
    idxs_test = REVIEW_DATASET(torch.arange(half_window_size, len(reviews_discrete_test) + half_window_size))
    train_set_size = int(len(idxs_train) * (1 - valid_ratio))
    valid_set_size = len(idxs_train) - train_set_size
    train_set, valid_set = data.random_split(idxs_train, [train_set_size, valid_set_size], generator=seed)
    train_dataloader = DataLoader(train_set, shuffle=True, batch_size=batch_size, drop_last=True, pin_memory=True)
    valid_dataloader = DataLoader(valid_set, shuffle=False, batch_size=1, drop_last=False, pin_memory=True)

    backbone = predictor(voc_size=len(voc), embedding_dim=embedding_dim, half_window_size=half_window_size)
    model = EmbeddingWorker(backbone=backbone, train_set=torch.tensor(reviews_discrete_train),
                            test_set=torch.tensor(reviews_discrete_test), half_size=half_window_size)

    trainer = lightning.Trainer(default_root_dir='./ckpt/train', accelerator='gpu')
    trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders=valid_dataloader)


if __name__ == '__main__':
    main()

