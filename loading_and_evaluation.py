import numpy as np
import torch
from collections import Counter
from sklearn.cluster import AgglomerativeClustering
from statistics import median, mean
from sklearn.metrics import classification_report
from torch.utils.data import Dataset
from transformers import AutoTokenizer

class NewsDataset(Dataset):
    def __init__(self, records, model_path, max_tokens, bert_embeds=None):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            do_lower_case=False,
            do_basic_tokenize=False,
            strip_accents=False
        )
        self.max_tokens = max_tokens
        self.records = records
        self.bert_embeds = bert_embeds
    
    def __len__(self):
        return len(self.records)
    
    def __getitem__(self, index):
        samples = self.records[index]
      
        if type(samples) is tuple:
            samples = [s["title"] + " [SEP] " + s["text"] for s in samples]
            samples = [self.tokenizer(
                s,
               add_special_tokens=True,
               max_length=self.max_tokens,
               padding="max_length",
               truncation=True,
               return_tensors='pt'
               ) for s in samples]
            samples = [{key: value.squeeze(0) for key, value in s.items()} for s in samples]
          
            if self.bert_embeds is not None:
                bert_embeds = self.bert_embeds[index]
                return samples[0], samples[1], samples[2], bert_embeds[0], bert_embeds[1], bert_embeds[2]
            return samples[0], samples[1], samples[2]

        else:
            samples = samples["title"] + " [SEP] " + samples["text"]
            samples = self.tokenizer(
                samples,
                add_special_tokens=True,
                max_length=self.max_tokens,
                padding="max_length",
                truncation=True,
                return_tensors='pt'
                ) 
            samples = {key: value.squeeze(0) for key, value in samples.items()}

            if self.bert_embeds is not None:
               bert_embeds = self.bert_embeds[index]
               return samples, bert_embeds
            return samples

def calc_metrics(markup, url2record, labels):
    not_found_count = 0
    for record in markup:
        first_url = record["first_url"]
        second_url = record["second_url"]
        not_found_in_labels = first_url not in labels or second_url not in labels
        not_found_in_records = first_url not in url2record or second_url not in url2record
        if not_found_in_labels or not_found_in_records:
            not_found_count += 1
            markup.remove(record)
    if not_found_count != 0:
        print("Not found {} pairs from markup".format(not_found_count))

    targets = []
    predictions = []
    errors = []
    for record in markup:
        first_url = record["first_url"]
        second_url = record["second_url"]
        target = int(record["quality"] == "OK")
        prediction = int(labels[first_url] == labels[second_url])
        first = url2record.get(first_url)
        second = url2record.get(second_url)
        targets.append(target)
        predictions.append(prediction)
        if target == prediction:
            continue
        errors.append({
            "target": target,
            "prediction": prediction,
            "first_url": first_url,
            "second_url": second_url,
            "first_title": first["title"],
            "second_title": second["title"],
            "first_text": first["text"],
            "second_text": second["text"]
        })

    metrics = classification_report(targets, predictions, output_dict=True)
    return metrics, errors


def gen_batch(records, batch_size):
    batch_start = 0
    while batch_start < len(records):
        batch_end = batch_start + batch_size
        batch = records[batch_start: batch_end]
        batch_start = batch_end
        yield batch

def records_to_embeds(records, model, tokenizer, batch_size, max_tokens_count):
    current_index = 0
    embeddings = np.zeros((len(records), 256))
    for batch in gen_batch(records, batch_size):
        samples = [r["title"] + " [SEP] " + r["text"] for r in batch]
        inputs = tokenizer(
            samples,
            add_special_tokens=True,
            max_length=max_tokens_count,
            padding="max_length",
            truncation=True,
            return_tensors='pt'
        )
        batch_input_ids = inputs["input_ids"].cuda()
        batch_mask = inputs["attention_mask"].cuda()
        
        assert len(batch_input_ids[0]) == len(batch_mask[0]) == max_tokens_count
        with torch.no_grad():
            try:
              batch_embeddings = model(batch_input_ids, batch_mask)
            except:
              batch_embeddings = model(batch_input_ids)
            batch_embeddings = batch_embeddings.cpu().numpy()
        embeddings[current_index:current_index+batch_size, :] = batch_embeddings
        current_index += batch_size
    return embeddings
    
def get_quality(markup, embeds, records, dist_threshold, print_result=False):
    clustering_model = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=dist_threshold,
        linkage="average",
        affinity="cosine"
    )

    clustering_model.fit(embeds)
    labels = clustering_model.labels_
    
    idx2url = dict()
    url2record = dict()
    for i, record in enumerate(records):
        idx2url[i] = record["url"]
        url2record[record["url"]] = record

    url2label = dict()
    for i, label in enumerate(labels):
        url2label[idx2url[i]] = label
        
    metrics = calc_metrics(markup, url2record, url2label)[0]
    if print_result:
        print()
        print("Accuracy: {:.1f}".format(metrics["accuracy"] * 100.0))
        print("Positives Recall: {:.1f}".format(metrics["1"]["recall"] * 100.0))
        print("Positives Precision: {:.1f}".format(metrics["1"]["precision"] * 100.0))
        print("Positives F1: {:.1f}".format(metrics["1"]["f1-score"] * 100.0))
        print("Distance: ", dist_threshold)
        sizes = list(Counter(labels).values())
        print("Max cluster size: ", max(sizes))
        print("Median cluster size: ", median(sizes))
        print("Avg cluster size: {:.2f}".format(mean(sizes)))
        return
    return metrics["1"]["f1-score"]