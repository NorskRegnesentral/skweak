import numpy as np
import pandas
import sklearn.metrics
from skweak import utils
from spacy.tokens import Span # type: ignore

def evaluate(docs, all_labels, target_sources):
    """Extracts the evaluation results for one or more sources, and add them to a pandas DataFrame."""
    
    if isinstance(target_sources, str):
        target_sources = [target_sources]

    records = []
    for source in target_sources:
        results = get_results(docs, all_labels, source)
        labels = set(results["label_weights"].keys())
        for name in sorted(labels) + ["micro", "weighted", "macro"]:
            if name in results:
                record = results[name]
                record["label"] = name
                record["model"] = source
                if name in labels:
                    record["proportion"] = results["label_weights"][name]          
                records.append(record)
    
    df = pandas.DataFrame.from_records(records)
    df["proportion"] = df.proportion.apply(lambda x: "%.1f %%"%(x*100) if not np.isnan(x) else "")
    df["tok_cee"] = df.tok_cee.apply(lambda x: str(x) if not np.isnan(x) else "")
    df["tok_acc"] = df.tok_acc.apply(lambda x: str(x) if not np.isnan(x) else "")
    df["coverage"] = df.coverage.apply(lambda x: str(x) if not np.isnan(x) else "")
    df = df.set_index(["label", "proportion", "model"]).sort_index()
    df = df[["tok_precision", "tok_recall", "tok_f1", "tok_cee", "tok_acc", "coverage",
             "ent_precision", "ent_recall", "ent_f1"]]
    return df



def get_results(docs, all_labels, target_source, conf_threshold=0.5):
    """Computes the usual metrics (precision, recall, F1, cross-entropy) on the dataset, using the spacy entities 
    in each document as gold standard, and the annotations of a given source as the predicted values"""


    all_numbers = compute_raw_numbers(docs, all_labels, target_source, conf_threshold)
    tok_tp, tok_fp, tok_fn, tok_logloss, tok_nb, tok_tp_tn, ent_tp, ent_fp, ent_fn, ent_support, tok_support = all_numbers

    # We then compute the metrics themselves
    results = {}
    for label in ent_support:
        ent_pred = ent_tp[label]+ent_fp[label] + 1E-10
        ent_true = ent_tp[label]+ent_fn[label] + 1E-10
        tok_pred = tok_tp[label]+tok_fp[label] + 1E-10
        tok_true = tok_tp[label]+tok_fn[label] + 1E-10
        results[label] = {}
        results[label]["ent_precision"] = round(ent_tp[label]/ent_pred, 3)
        results[label]["ent_recall"] = round(ent_tp[label]/ent_true, 3)
        results[label]["tok_precision"] = round(tok_tp[label]/tok_pred, 3)
        results[label]["tok_recall"] = round(tok_tp[label]/tok_true, 3)
        
        ent_f1_numerator = (results[label]["ent_precision"] * results[label]["ent_recall"])
        ent_f1_denominator = (results[label]["ent_precision"] +results[label]["ent_recall"]) + 1E-10
        results[label]["ent_f1"] = 2*round(ent_f1_numerator / ent_f1_denominator, 3)
            
        tok_f1_numerator = (results[label]["tok_precision"] * results[label]["tok_recall"])
        tok_f1_denominator = (results[label]["tok_precision"] +results[label]["tok_recall"]) + 1E-10
        results[label]["tok_f1"] = 2*round(tok_f1_numerator / tok_f1_denominator, 3)
    
    results["macro"] = {"ent_precision":round(np.mean([results[l]["ent_precision"] for l in ent_support]), 3), 
                       "ent_recall":round(np.mean([results[l]["ent_recall"] for l in ent_support]), 3), 
                       "tok_precision":round(np.mean([results[l]["tok_precision"] for l in ent_support]), 3), 
                       "tok_recall":round(np.mean([results[l]["tok_recall"] for l in ent_support]), 3)}
    
        
    label_weights = {l:ent_support[l]/sum(ent_support.values()) for l in ent_support}
    results["label_weights"] = label_weights
    results["weighted"] = {"ent_precision":round(np.sum([results[l]["ent_precision"]*label_weights[l] 
                                                            for l in ent_support]), 3), 
                           "ent_recall":round(np.sum([results[l]["ent_recall"]*label_weights[l] 
                                                         for l in ent_support]), 3), 
                           "tok_precision":round(np.sum([results[l]["tok_precision"]*label_weights[l] 
                                                           for l in ent_support]), 3), 
                           "tok_recall":round(np.sum([results[l]["tok_recall"]*label_weights[l] 
                                                        for l in ent_support]), 3)}
    
    ent_pred = sum([ent_tp[l] for l in ent_support]) + sum([ent_fp[l] for l in ent_support]) + 1E-10
    ent_true = sum([ent_tp[l] for l in ent_support]) + sum([ent_fn[l] for l in ent_support]) + 1E-10
    tok_pred = sum([tok_tp[l] for l in ent_support]) + sum([tok_fp[l] for l in ent_support]) + 1E-10
    tok_true = sum([tok_tp[l] for l in ent_support])  + sum([tok_fn[l] for l in ent_support]) + 1E-10
    results["micro"] = {"ent_precision":round(sum([ent_tp[l] for l in ent_support]) / ent_pred, 3), 
                        "ent_recall":round(sum([ent_tp[l] for l in ent_support]) / ent_true, 3), 
                        "tok_precision":round(sum([tok_tp[l] for l in ent_support]) /tok_pred, 3), 
                        "tok_recall":round(sum([tok_tp[l] for l in ent_support]) / tok_true, 3),
                        "tok_cee":round(tok_logloss/tok_nb, 3),
                        "tok_acc": round(tok_tp_tn/tok_nb, 3),
                        "coverage":round((sum(tok_tp.values()) +sum(tok_fp.values())) / sum(tok_support.values()), 3)}
    
    for metric in ["macro", "weighted", "micro"]:
        ent_f1_numerator = (results[metric]["ent_precision"] * results[metric]["ent_recall"])
        ent_f1_denominator = (results[metric]["ent_precision"] +results[metric]["ent_recall"]) + 1E-10
        results[metric]["ent_f1"] = 2*round(ent_f1_numerator / ent_f1_denominator, 3)
            
        tok_f1_numerator = (results[metric]["tok_precision"] * results[metric]["tok_recall"])
        tok_f1_denominator = (results[metric]["tok_precision"] +results[metric]["tok_recall"]) + 1E-10
        results[metric]["tok_f1"] = 2*round(tok_f1_numerator / tok_f1_denominator, 3)
        
    return results


def compute_raw_numbers(docs, all_labels, target_source, conf_threshold=0.5):
    """Computes the raw metrics (true positives, true negatives, ...) on the dataset, using the spacy entities 
    in each document as gold standard, and the annotations of a given source as the predicted values"""

    # We start by computing the TP, FP and FN values
    tok_tp = {}
    tok_fp = {}
    tok_fn ={}

    tok_logloss = 0
    tok_nb = 0
    tok_tp_tn = 0
    
    ent_tp ={}
    ent_fp = {}
    ent_fn = {}
    ent_support = {}
    tok_support = {}

    for doc in docs:
        if target_source in doc.spans:
            spans = utils.get_spans_with_probs(doc, target_source)
        else:
            spans = []
        spans = [span for (span, prob) in spans if prob >= conf_threshold]
            
        for label in all_labels:
            true_spans = {(ent.start, ent.end) for ent in doc.ents if ent.label_==label}
            pred_spans = {(span.start,span.end) for span in spans if span.label_==label}
        
            ent_tp[label] = ent_tp.get(label,0) + len(true_spans.intersection(pred_spans))
            ent_fp[label] = ent_fp.get(label,0) + len(pred_spans - true_spans)
            ent_fn[label] = ent_fn.get(label,0) +  len(true_spans - pred_spans)
            ent_support[label] = ent_support.get(label, 0) + len(true_spans)
            
            true_tok_labels = {i for start,end in true_spans for i in range(start, end)}
            pred_tok_labels = {i for start,end in pred_spans for i in range(start, end)}
            tok_tp[label] = tok_tp.get(label, 0) + len(true_tok_labels.intersection(pred_tok_labels))
            tok_fp[label] = tok_fp.get(label, 0) + len(pred_tok_labels - true_tok_labels)
            tok_fn[label] = tok_fn.get(label,0) + len(true_tok_labels - pred_tok_labels)
            tok_support[label] = tok_support.get(label, 0) + len(true_tok_labels)

        gold_probs, pred_probs = _get_probs(doc, all_labels, target_source)
        tok_logloss += sklearn.metrics.log_loss(gold_probs, pred_probs, normalize=False)
        tok_tp_tn += sum(gold_probs.argmax(axis=1) == pred_probs.argmax(axis=1))
        tok_nb += len(doc)

    return (tok_tp, tok_fp, tok_fn, tok_logloss, tok_nb, tok_tp_tn, ent_tp, 
            ent_fp, ent_fn, ent_support, tok_support)


def _get_probs(doc, all_labels, target_source):
    """Retrieves the gold and predicted probabilities (as matrices)"""
    
    out_label_indices = {"O":0}
    for label in all_labels:
        for prefix in "BI":
            out_label_indices["%s-%s" % (prefix, label)] = len(out_label_indices)
                            
    gold_probs = np.zeros((len(doc), len(out_label_indices)), dtype=np.int16)    
    for ent in doc.ents:
        gold_probs[ent.start, out_label_indices.get("B-%s" % ent.label_, 0)] = 1
        for i in range(ent.start+1, ent.end):
            gold_probs[i, out_label_indices.get("I-%s" % ent.label_, 0)] = 1
    
    pred_probs = np.zeros(gold_probs.shape)
    if target_source in doc.spans and "probs" in doc.spans[target_source].attrs:       
        for tok_pos, labels in doc.spans[target_source].attrs["probs"].items():
            for label, label_prob in labels.items():
                pred_probs[tok_pos, out_label_indices[label]] = label_prob
        pred_probs[:,0] = np.clip(1-pred_probs[:,1:].sum(axis=1), 0.0, 1.0)
    else:
        vector = utils.spans_to_array(doc, all_labels, [target_source])[:,0]
        pred_probs[np.arange(vector.size), vector] = True       
    
    return gold_probs, pred_probs


def show_errors(docs, all_labels, target_source, conf_threshold=0.5):
    """Utilities to display the errors/omissions of a given source"""
    
    for i, doc in enumerate(docs):
        
        spans = utils.get_spans_with_probs(doc, target_source)
            
        print("Doc %i:"%i, doc)
        true_spans = {(ent.start, ent.end):ent.label_ for ent in doc.ents}
        pred_spans = {(span.start,span.end):span.label_ for span, prob in spans if prob >=conf_threshold}

        for start,end in true_spans:
            if (start,end) not in pred_spans:
                print("Not found: %s [%i:%i] -> %s"%(doc[start:end], start, end, true_spans[(start,end)]))
            elif true_spans[(start,end)]!=pred_spans[(start,end)]:
                print("Wrong label: %s [%i:%i] -> %s but predicted as %s"%(doc[start:end], start, end, 
                true_spans[(start,end)], pred_spans[(start,end)]))

        for start,end in pred_spans:
            if (start,end) not in true_spans:
                print("Spurious: %s [%i:%i] -> %s"%(doc[start:end], start, end, pred_spans[(start,end)]))
