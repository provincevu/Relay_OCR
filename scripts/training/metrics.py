"""Loss and validation metrics for page-3 role/link training."""

import torch

from scripts.inference.page_n_prediction_filter import filter_prediction_links
from scripts.training.page3_layout_model import encode_page, predict_page, row_links


ROLES = [
    "KEY",
    "VALUE",
    "HEADER",
    "FIELD_TABLE",
    "NOTE",
    "PAGE_NUMBER",
    "FORM_NUMBER",
    "INLINE_KEY_VALUE",
    "OTHER",
]


def is_trainable_annotation(labels):
    """Return whether labels can be represented by the single-class head."""
    return "UNLABELED" not in labels and set(labels) != {"KEY", "VALUE"}


def masked_training_loss(role_logits, page):
    """Compute single-class role loss while masking unsupported labels."""
    region_ids = [region["id"] for region in page["regions"]]
    labels = [page["annotations"][region_id] for region_id in region_ids]
    target_indices = []
    known_values = []
    for labels_for_region in labels:
        labels_set = set(labels_for_region)
        known_values.append(is_trainable_annotation(labels_set))
        target_role = next(
            (role for role in ROLES if role in labels_set),
            "OTHER",
        )
        target_indices.append(ROLES.index(target_role))
    known = torch.tensor(
        known_values,
        dtype=torch.bool,
        device=role_logits.device,
    )
    role_targets = torch.tensor(
        target_indices,
        dtype=torch.long,
        device=role_logits.device,
    )
    if not known.any():
        return role_logits.sum() * 0
    return torch.nn.functional.cross_entropy(
        role_logits[known],
        role_targets[known],
    )


def _update_counts(counts, gold, predicted):
    counts[0] += len(gold & predicted)
    counts[1] += len(predicted - gold)
    counts[2] += len(gold - predicted)


def _f1_from_counts(counts):
    true_positive, false_positive, false_negative = counts
    precision = true_positive / (true_positive + false_positive) \
        if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) \
        if true_positive + false_negative else 0.0
    return 2 * precision * recall / (precision + recall) \
        if precision + recall else 0.0


def _normalize_prediction_labels(labels):
    labels = set(labels)
    if labels.intersection(set(ROLES) - {"OTHER"}):
        labels.discard("OTHER")
    return labels or {"OTHER"}


class ValidationMetrics:
    """Accumulate the unchanged KEY, VALUE, and row-link F1 definitions."""

    def __init__(self):
        self.role_counts = {role: [0, 0, 0] for role in ("KEY", "VALUE")}
        self.row_link_counts = [0, 0, 0]

    def add_page(self, page, predicted_regions, predicted_links, normalize=False):
        labels = page["annotations"]
        known_ids = {
            region["id"]
            for region in page["regions"]
            if is_trainable_annotation(labels[region["id"]])
        }
        predicted_labels = {
            region["id"]: (
                _normalize_prediction_labels(region["labels"])
                if normalize
                else set(region["labels"])
            )
            for region in predicted_regions
        }
        for role, counts in self.role_counts.items():
            gold = {
                region_id for region_id in known_ids
                if role in labels[region_id]
            }
            predicted = {
                region_id for region_id in known_ids
                if role in predicted_labels[region_id]
            }
            _update_counts(counts, gold, predicted)

        gold_links = {
            (key_id, value_id)
            for relation in page["relations"]
            for key_id in relation["key_region_ids"]
            for value_id in relation["value_region_ids"]
            if key_id in known_ids and value_id in known_ids
        }
        predicted_link_pairs = {
            (link["key_region_id"], link["value_region_id"])
            for link in predicted_links
            if link["key_region_id"] in known_ids
            and link["value_region_id"] in known_ids
        }
        _update_counts(self.row_link_counts, gold_links, predicted_link_pairs)

    def result(self):
        key_f1 = _f1_from_counts(self.role_counts["KEY"])
        value_f1 = _f1_from_counts(self.role_counts["VALUE"])
        row_link_f1 = _f1_from_counts(self.row_link_counts)
        role_f1 = (key_f1 + value_f1) / 2
        return {
            "key_f1": key_f1,
            "value_f1": value_f1,
            "role_f1": role_f1,
            "row_link_f1": row_link_f1,
            "val_f1": 0.5 * role_f1 + 0.5 * row_link_f1,
        }


def validation_metrics(pages, model, tokenizer, device, threshold):
    """Evaluate full-model role and row-link F1."""
    metrics = ValidationMetrics()
    model.eval()
    with torch.inference_mode():
        for page in pages:
            encoded = encode_page(page, tokenizer, None, "B", device)
            prediction = predict_page(
                model, encoded, page, ROLES, threshold, "row"
            )
            metrics.add_page(
                page,
                prediction["regions"],
                prediction["links"],
                normalize=True,
            )
    model.train()
    return metrics.result()


def cached_validation_metrics(pages, role_head, dropout, device, cached_logits):
    """Evaluate role and row-link F1 from cached region embeddings."""
    metrics = ValidationMetrics()
    role_head.eval()
    dropout.eval()
    with torch.inference_mode():
        for page in pages:
            role_logits, embeddings = cached_logits(
                role_head, dropout, page, device
            )
            role_scores = role_logits.softmax(dim=-1).cpu().tolist()
            predicted_regions = []
            for region, scores in zip(page["regions"], role_scores, strict=True):
                best_role_index = max(range(len(scores)), key=scores.__getitem__)
                predicted_regions.append({
                    "id": region["id"],
                    "text": region["text"],
                    "bbox": region["bbox"],
                    "labels": [ROLES[best_role_index]],
                })
            prediction = filter_prediction_links({
                "regions": predicted_regions,
                "links": row_links(
                    predicted_regions,
                    "KEY",
                    "VALUE",
                    "key_region_id",
                    "value_region_id",
                ),
            })
            metrics.add_page(
                page,
                predicted_regions,
                prediction["links"],
            )
            del embeddings
    role_head.train()
    dropout.train()
    return metrics.result()
