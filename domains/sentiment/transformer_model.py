from transformers import BertTokenizer, BertForSequenceClassification
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup

from torch.nn import functional as F
import torch
import numpy as np

import argparse
from tqdm import tqdm

from sklearn.metrics import f1_score

import sys
sys.path.insert(0, "..")
from skweak.utils import docbin_reader

class SSTDataLoader():
    def __init__(self, datafile, num_examples=None):
        labels, examples = [], []
        for line in open(datafile):
            label, sent = line.strip().split("\t", 1)
            labels.append(int(label))
            examples.append(sent)
        if num_examples is not None:
            labels = labels[:num_examples]
            examples = examples[:num_examples]
        self.labels = np.array(labels)
        self.examples = np.array(examples)

    def get_batches(self, batch_size=32, shuffle=True):
        if shuffle:
            idxs = np.arange(len(self.labels))
            np.random.shuffle(idxs)
            labels = list(self.labels[idxs])
            examples = list(self.examples[idxs])
        else:
            labels = list(self.labels)
            examples = list(self.examples)
        num_batches = self.get_num_batches(batch_size)
        i = 0
        for batch in range(num_batches):
            blabels = torch.tensor(labels[i:i+batch_size])
            bexamples = examples[i:i+batch_size]
            i += batch_size
            yield (blabels, bexamples)

    def get_num_batches(self, batch_size=32):
        num_batches = len(self.labels) // batch_size
        if (len(self.labels) % batch_size) > 0:
            num_batches += 1
        return num_batches

class DocbinDataLoader():
    def __init__(self, datafile, num_examples=None, gold=False):
        labels, examples = [], []
        for doc in docbin_reader(datafile):
            examples.append(doc.text)
            if gold:
                labels.append(doc.user_data["gold"])
            else:
                labels.append(list(doc.user_data["agg_spans"]["hmm"].values())[0])
        if num_examples is not None:
            labels = labels[:num_examples]
            examples = examples[:num_examples]
        self.labels = np.array(labels)
        self.examples = np.array(examples)

    def get_batches(self, batch_size=32, shuffle=True):
        if shuffle:
            idxs = np.arange(len(self.labels))
            np.random.shuffle(idxs)
            labels = list(self.labels[idxs])
            examples = list(self.examples[idxs])
        else:
            labels = list(self.labels)
            examples = list(self.examples)
        num_batches = self.get_num_batches(batch_size)
        i = 0
        for batch in range(num_batches):
            blabels = torch.tensor(labels[i:i+batch_size])
            bexamples = examples[i:i+batch_size]
            i += batch_size
            yield (blabels, bexamples)

    def get_num_batches(self, batch_size=32):
        num_batches = len(self.labels) // batch_size
        if (len(self.labels) % batch_size) > 0:
            num_batches += 1
        return num_batches

def train(model, save_dir="../data/sentiment/models/norbert"):
    model.train()


    optimizer = AdamW(model.parameters(), lr=1e-5)

    num_train_steps = int(len(train_loader.examples) / args.train_batch_size) * args.num_train_epochs

    scheduler = get_linear_schedule_with_warmup(optimizer, args.warmup_steps, num_train_steps)

    best_dev_f1 = 0.0

    print("training for {} epochs...".format(args.num_train_epochs))

    for epoch_num, epoch in enumerate(range(args.num_train_epochs)):
        model.train()
        train_loss = 0
        num_batches = 0
        train_preds = []
        train_gold = []
        for b in tqdm(train_loader.get_batches(batch_size=args.train_batch_size), total=train_loader.get_num_batches(batch_size=args.train_batch_size)):
            labels, sents = b
            encoding = tokenizer(sents, return_tensors='pt', padding=True, truncation=True, max_length=150)

            outputs = model(**encoding)
            _, p = outputs.logits.max(1)
            train_preds.extend(p.tolist())
            train_gold.extend(labels.tolist())
            loss = F.cross_entropy(outputs.logits, labels)
            train_loss += loss.data
            num_batches += 1
            loss.backward()
            optimizer.step() #type: ignore
            scheduler.step() #type: ignore
            optimizer.zero_grad() #type: ignore
        print("Epoch {0}: Loss {1:.3f}".format(epoch_num + 1, train_loss / num_batches))
        print("Train F1: {0:.3f}".format(f1_score(train_gold, train_preds, average="macro")))


        model.eval()
        dev_loss = 0
        num_batches = 0
        dev_preds = []
        dev_gold = []
        for b in tqdm(dev_loader.get_batches(batch_size=args.eval_batch_size), total=dev_loader.get_num_batches(batch_size=args.eval_batch_size)):
            labels, sents = b
            encoding = tokenizer(sents, return_tensors='pt', padding=True, truncation=True, max_length=150)

            outputs = model(**encoding)
            _, p = outputs.logits.max(1)
            dev_preds.extend(p.tolist())
            dev_gold.extend(labels.tolist())
            loss = F.cross_entropy(outputs.logits, labels)
            dev_loss += loss.data
            num_batches += 1
        dev_f1 = f1_score(dev_gold, dev_preds, average="macro")
        print("Dev F1: {0:.3f}".format(dev_f1))

        if dev_f1 > best_dev_f1: #type: ignore
            best_dev_f1 = dev_f1
            print("Current best dev: {0:.3f}".format(best_dev_f1))
            print("Saving model")
            model.save_pretrained(save_dir)


def test(model):
    print("loading best model on dev data")
    model.eval()
    test_loss = 0
    num_batches = 0
    test_preds = []
    test_gold = []
    for b in tqdm(test_loader.get_batches(batch_size=args.eval_batch_size), total=test_loader.get_num_batches(batch_size=args.eval_batch_size)):
        labels, sents = b
        encoding = tokenizer(sents, return_tensors='pt', padding=True, truncation=True, max_length=150)

        outputs = model(**encoding)
        _, p = outputs.logits.max(1)
        test_preds.extend(p.tolist())
        test_gold.extend(labels.tolist())
        loss = F.cross_entropy(outputs.logits, labels)
        test_loss += loss.data
        num_batches += 1
    test_f1 = f1_score(test_gold, test_preds, average="macro")
    print("Test F1: {0:.3f}".format(test_f1))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_batch_size", default=16, type=int)
    parser.add_argument("--eval_batch_size", default=16, type=int)
    parser.add_argument("--num_train_epochs", default=20, type=int)
    parser.add_argument("--warmup_steps", default=50, type=int)
    parser.add_argument("--model",
                        default="../data/sentiment/models/norbert")
    parser.add_argument("--save_dir",
                        default="../data/sentiment/models/nobert")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--train_on_gold", action="store_true")


    args = parser.parse_args()

    print("loading data...")
    # train_loader = SSTDataLoader("../data/sentiment/sst/train.txt")
    # dev_loader = SSTDataLoader("../data/sentiment/sst/dev.txt")
    # test_loader = SSTDataLoader("../data/sentiment/sst/test.txt")
    train_loader = DocbinDataLoader("../data/sentiment/norec_sentence/train_pred.docbin", num_examples=500, gold=args.train_on_gold)
    dev_loader = DocbinDataLoader("../data/sentiment/norec_sentence/dev_pred.docbin", num_examples=500, gold=args.train_on_gold)
    test_loader = DocbinDataLoader("../data/sentiment/norec_sentence/test_pred.docbin", gold=True)

    print("loading model...")
    tokenizer = BertTokenizer.from_pretrained("ltgoslo/norbert")
    model = BertForSequenceClassification.from_pretrained(args.model, num_labels=3)

    if args.train:
        train(model, args.save_dir)

    # Test model
    if args.test:
        test(model)

