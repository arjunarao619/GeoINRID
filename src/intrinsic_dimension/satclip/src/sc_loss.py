import torch
import torch.nn.functional as F
import torch.nn as nn

# The SatCLIP loss function is equivalent to the SatCLIP loss: https://github.com/microsoft/satclip/blob/main/satclip/loss.py 
class SatCLIPLoss(nn.Module):

    def __init__(
            self,
            local_loss=False,
            gather_with_grad=False,
            cache_labels=False,
            rank=0,
            world_size=1,
            use_horovod=False,
            type='contrastive'
    ):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.use_horovod = use_horovod
        self.type = type

        # cache state
        self.prev_num_logits = 0
        self.labels = {}

    def get_ground_truth(self, device, num_logits) -> torch.Tensor:
        # calculated ground-truth and cache if enabled
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            if self.world_size > 1 and self.local_loss:
                labels = labels + num_logits * self.rank
            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]
        return labels

    def forward(self, logits_per_image, logits_per_coord, output_dict=False):
        device = logits_per_image.device

        if self.type == 'contrastive':
            labels = self.get_ground_truth(device, logits_per_image.shape[0])
            total_loss = (
                F.cross_entropy(logits_per_image, labels) +
                F.cross_entropy(logits_per_coord, labels)
            ) / 2
            return {"contrastive_loss": total_loss} if output_dict else total_loss
        
        elif self.type == 'siglip':
            n = logits_per_image.size(0)
            # -1 for off-diagonals and 1 for diagonals
            labels = 2 * torch.eye(n, device=device) - 1
            # pairwise sigmoid loss
            total_loss = -torch.sum(F.logsigmoid(labels * logits_per_image)) / n
            return {"siglip_loss": total_loss} if output_dict else total_loss
