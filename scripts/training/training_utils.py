"""Shared optimizer, scheduler, early-stopping, and checkpoint helpers."""

from pathlib import Path

import torch
from transformers import get_linear_schedule_with_warmup

from scripts.training.metrics import ROLES


def optimizer_parameters(model, weight_decay):
    """Separate AdamW parameters that receive decay from bias/norm weights."""
    decay = []
    no_decay = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.ndim == 1 or name.endswith("bias"):
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def create_scheduler(optimizer, epochs, steps_per_epoch, warmup_ratio):
    """Create the same linear warmup/decay schedule used by both train paths."""
    total_steps = epochs * steps_per_epoch
    warmup_steps = int(total_steps * warmup_ratio)
    return get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )


class EarlyStopping:
    """Track best validation score, model states, and patience consistently."""

    def __init__(self, patience, min_delta):
        self.patience = patience
        self.min_delta = min_delta
        self.best_score = float("-inf")
        self.best_epoch = 0
        self.best_metrics = None
        self.best_states = None
        self.epochs_without_improvement = 0

    def update(self, epoch, metrics, modules):
        """Save module snapshots on improvement; return whether to stop."""
        score = metrics["val_f1"]
        if score > self.best_score + self.min_delta:
            self.best_score = score
            self.best_epoch = epoch
            self.best_metrics = dict(metrics)
            self.best_states = {
                name: {
                    key: value.detach().cpu().clone()
                    for key, value in module.state_dict().items()
                }
                for name, module in modules.items()
            }
            self.epochs_without_improvement = 0
            return False

        self.epochs_without_improvement += 1
        return self.epochs_without_improvement >= self.patience

    def restore_best(self, modules):
        """Restore each tracked module to its best validation state."""
        if self.best_states is None or self.best_metrics is None:
            raise RuntimeError("Không tạo được checkpoint tốt nhất theo validation F1.")
        for name, module in modules.items():
            module.load_state_dict(self.best_states[name])


def run_training_epochs(
    epochs,
    modules,
    train_epoch,
    validate,
    patience,
    min_delta,
):
    """Share epoch reporting, validation selection, and early stopping."""
    early_stopping = EarlyStopping(patience, min_delta)
    for epoch in range(epochs):
        average_loss = train_epoch()
        metrics = validate()
        print(
            f"Epoch {epoch + 1}/{epochs}: "
            f"loss={average_loss:.6f}, "
            f"val_f1={metrics['val_f1']:.4f}, "
            f"KEY={metrics['key_f1']:.4f}, "
            f"VALUE={metrics['value_f1']:.4f}, "
            f"row-link={metrics['row_link_f1']:.4f}",
            flush=True,
        )
        if early_stopping.update(epoch + 1, metrics, modules):
            print(
                f"Early stopping tại epoch {epoch + 1}; "
                f"best epoch={early_stopping.best_epoch}, "
                f"val_f1={early_stopping.best_score:.4f}",
                flush=True,
            )
            break
    return early_stopping


def save_checkpoint(output, tokenizer, model, epoch, metrics, head_dropout):
    """Save tokenizer and model state using the established checkpoint schema."""
    output = Path(output)
    if (output / "model.pt").exists():
        raise FileExistsError(
            f"Checkpoint đã tồn tại: {output / 'model.pt'}. "
            "Hãy chọn thư mục output mới cho model single class."
        )
    output.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(output)
    model_state = {
        name: parameter.detach().cpu()
        for name, parameter in model.state_dict().items()
    }
    torch.save(
        {
            "roles": ROLES,
            "role_mode": "single_class",
            "encoder_config": model.encoder.config.to_dict(),
            "model_state": model_state,
            "head_dropout": head_dropout,
            "link_mode": "row",
            "best_epoch": epoch,
            "best_val_metrics": metrics,
        },
        output / "model.pt",
    )
