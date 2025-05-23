from torch import nn

from improve_mesh_segmentation.partnet_grasp.dataset import PartNetGraspDataset, processed_partnet_grasp_generator
from improve_mesh_segmentation.training.imcnn import SegImcnn
import torch
import torch.nn.functional as F
from torch_influence import BaseObjective
from torch_influence import AutogradInfluenceModule
import scipy as sp


data_path = "/home/iroberts/projects/VisMeshSegmentation/improve_mesh_segmentation/datasets/partnet_grasp.zip"
model_path = "/home/iroberts/projects/VisMeshSegmentation/run_through/logs/model.zip"
if torch.cuda.is_available():
    device = torch.device('cuda:0')
    print(f"CUDA is available, using {device}")
else:
    device = torch.device('cpu')
    print(f"CUDA is not available, using {device}")

    # === Define loss function ===
criterion = nn.CrossEntropyLoss(reduction="none")  # get per-point loss


    # === Conjugate Gradient Solver ===
def hvp(loss, model, v):
    grads = torch.autograd.grad(loss, model.parameters(), create_graph=True, retain_graph=True)
    flat_grads = torch.cat([g.contiguous().view(-1) for g in grads])
    hv = torch.autograd.grad(flat_grads, model.parameters(), grad_outputs=v, retain_graph=True)
    hv_flat = torch.cat([h.contiguous().view(-1) for h in hv])
    return hv_flat

def conjugate_gradient(loss, model, v, tol=1e-5, max_iter=100):
    x = torch.zeros_like(v)
    r = v.clone()
    p = r.clone()
    rsold = torch.dot(r, r)

    for _ in range(max_iter):
        Avp = hvp(loss, model, p)
        alpha = rsold / (torch.dot(p, Avp) + 1e-8)
        x += alpha * p
        r -= alpha * Avp
        rsnew = torch.dot(r, r)
        if torch.sqrt(rsnew) < tol:
            break
        p = r + (rsnew / rsold) * p
        rsold = rsnew
    return x


if __name__ == "__main__":

    model = SegImcnn(adapt_data=PartNetGraspDataset(data_path, set_type=0, only_signal=True))
    model.load_state_dict(torch.load(model_path))
    # model = model.to(device)
    print(f"Model parameters on CUDA: {next(model.parameters()).is_cuda}")

    dataset = processed_partnet_grasp_generator(data_path, set_type=0)

    all_influence_scores = []

    for mesh_idx, ((signal, bc), labels) in enumerate(dataset):
        # signal = signal.to(device)
        # bc = bc.to(device)
        # labels = labels.to(device)
        model.zero_grad()

        # 🔍 Sanity check: devices
        # assert signal.device == device, f"Signal is on {signal.device}, expected {device}"
        # assert bc.device == device, f"Barycentric coords are on {bc.device}, expected {device}"
        # assert labels.device == device, f"Labels are on {labels.device}, expected {device}"
        # for p in model.parameters():
        #     assert p.device == device, f"Model parameter not on {device}"

        pred = model([signal, bc])  # shape (N, C)
        loss_all = criterion(pred, labels)  # shape (N,)
        total_loss = loss_all.sum()

        # Compute total mesh gradient
        total_grad = torch.autograd.grad(total_loss, model.parameters(), create_graph=True)
        total_grad_flat = torch.cat([g.reshape(-1) for g in total_grad]).detach()

        # Estimate inverse Hessian-vector product
        s_m = conjugate_gradient(total_loss, model, total_grad_flat)

        mesh_influences = []

        for i in range(labels.shape[0]):
            point_loss = loss_all[i]
            point_grad = torch.autograd.grad(point_loss, model.parameters(), retain_graph=True)
            point_grad_flat = torch.cat([g.reshape(-1) for g in point_grad])
            influence_i = -torch.dot(s_m, point_grad_flat)
            mesh_influences.append(influence_i.detach().cpu().item())

        all_influence_scores.append({
            "mesh_idx": mesh_idx,
            "influence_per_vertex": mesh_influences,
        })

        print(f"[{mesh_idx}] Computed influence for {len(mesh_influences)} vertices.")
        # print(all_influence_scores)
        # break

    # (Optional) Save results
    import json

    with open("influence_scores.json", "w") as f:
        json.dump(all_influence_scores, f)




