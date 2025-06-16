from improve_mesh_segmentation.training.imcnn import SegImcnn

from geoconv_examples.mpi_faust.pytorch.model import custom_exp_scheduler
from torcheval.metrics.functional import multiclass_accuracy
from torch import nn

import torch
import sys


class CustomSegImcnn(SegImcnn):
    def train_loop(self,
                   dataset,
                   mesh_indices,
                   candidate_indices,
                   optimizer,
                   decay_rate=0.95,
                   decay_steps=500,
                   verbose=True,
                   epoch=None,
                   prev_steps=None,
                   use_lr_decay=False):
        self.train()
        epoch_accuracy = 0.
        epoch_loss = 0.
        mean_accuracy = 0.
        mean_loss = 0.

        step = 0
        loss_values = []
        for mesh_idx, ((signal, bc), gt) in zip(mesh_indices, dataset):
            pred = self([signal, bc])
            loss = nn.functional.cross_entropy(pred, gt, reduction="none")

            # Compute loss only for given candidates
            mesh_candidates = candidate_indices[mesh_idx == candidate_indices[:, 0]][:, 1]
            loss = loss[mesh_candidates]

            # Remember loss values
            loss_values.append(
                torch.cat(
                    [
                        torch.full_like(mesh_candidates, mesh_idx).unsqueeze(dim=-1),
                        mesh_candidates.unsqueeze(dim=-1),
                        loss.detach().unsqueeze(dim=-1)
                    ],
                    dim=-1
                )
            )

            loss = loss.mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if use_lr_decay:
                custom_exp_scheduler(optimizer, prev_steps + step, decay_rate=decay_rate, decay_steps=decay_steps)

            # Statistics
            epoch_accuracy = epoch_accuracy + multiclass_accuracy(pred, gt).detach()
            epoch_loss = epoch_loss + loss.detach()

            # I/O
            mean_accuracy = epoch_accuracy / (step + 1)
            mean_loss = epoch_loss / (step + 1)
            if verbose:
                sys.stdout.write(
                    f"\rEpoch: {epoch} - "
                    f"Training step: {step} - "
                    f"Loss {mean_loss:.4f} - "
                    f"Accuracy {mean_accuracy:.4f}"
                )
            step += 1

        return {"epoch_loss": mean_loss, "epoch_accuracy": mean_accuracy, "candidate_loss_values": loss_values}
