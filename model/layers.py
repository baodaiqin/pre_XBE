import torch
from torch import nn


class CrossStitch(nn.Module):

    def __init__(self, n_layers, input1_dim, input2_dim, **kwargs):
        super().__init__()
        # assert input1_dim == input2_dim, 'TODO'
        self.n_layers = n_layers


class Adapter(nn.Module):
    def __init__(self, input_dim, bottleneck_dim, output_dim):
        super().__init__()
        self.proj_in = nn.Linear(input_dim, bottleneck_dim)
        self.nonlin = nn.ReLU()
        self.proj_out = nn.Linear(bottleneck_dim, output_dim)

    def forward(self, x):
        res = x
        x = self.proj_in(x)
        x = self.nonlin(x)
        x = self.proj_out(x)
        return x + res


class NoSkipAdapter(Adapter):
    def forward(self, x):
        x = self.proj_in(x)
        x = self.nonlin(x)
        x = self.proj_out(x)
        return x


class GatedCrossStitch_Multihead(CrossStitch):

    def __init__(
            self,
            n_layers,
            input1_dim,
            input2_dim,
            adapter_bottleneck_dim=256,
            attn_dim=64,
            n_attn_heads=8,
            ):
        super().__init__(n_layers, input1_dim, input2_dim)
        self.scaling_factor = torch.sqrt(torch.tensor(input1_dim))
        self.gates12 = nn.ModuleList([
            nn.Linear(2 * input1_dim, 1)
            for _ in range(n_layers)])
        self.gates21 = nn.ModuleList([
            nn.Linear(2 * input2_dim, 1)
            for _ in range(n_layers)])

        if input1_dim == input2_dim:
            adapter_cls = Adapter
        else:
            adapter_cls = NoSkipAdapter
            self.proj12 = nn.Linear(input2_dim, input1_dim)
            self.proj21 = nn.Linear(input1_dim, input2_dim)

        self.adapters12 = nn.ModuleList([
            adapter_cls(input2_dim, adapter_bottleneck_dim, input1_dim)
            for _ in range(n_layers)])
        self.adapters21 = nn.ModuleList([
            adapter_cls(input1_dim, adapter_bottleneck_dim, input2_dim)
            for _ in range(n_layers)])

        if input1_dim != input2_dim:
            raise NotImplementedError()
        self.attns1 = nn.ModuleList([
            nn.MultiheadAttention(input1_dim, n_attn_heads, batch_first=True)
            for _ in range(n_layers)])

        self.attns2 = nn.ModuleList([
            nn.MultiheadAttention(input2_dim, n_attn_heads, batch_first=True)
            for _ in range(n_layers)])

    def forward(self, layer_key, input1, input2):
        if input1 is None or input2 is None:
            # no cross stitch, just pass through
            return input1, input2
        attn1_out, attn1_scores = self.attns1[layer_key](input1, input2, input2)
        attn2_out, attn2_scores = self.attns2[layer_key](input2, input1, input1)

        input12 = self.adapters12[layer_key](attn1_out)
        weight12 = self.gates12[layer_key](torch.cat([input1, input12], dim=-1)).sigmoid()
        output1 = input1 * weight12 + input12 * (1 - weight12)

        input21 = self.adapters21[layer_key](attn2_out)
        weight21 = self.gates21[layer_key](torch.cat([input2, input21], dim=-1)).sigmoid()
        output2 = input2 * weight21 + input21 * (1 - weight21)

        # self.i += 1
        # if not self.i % 100:
        #     print(weight12[0].min().item(), weight12[0].max().item(), weight21[0].min().item(), weight21[0].max().item())

        return output1, output2


class GatedCrossStitchOut():
    __slots__ = (
        'output1', 'output2', 'weight12', 'weight21', 'scores12', 'scores21')

    def __init__(
            self, *, output1, output2, weight12, weight21, scores12, scores21):
        self.output1 = output1
        self.output2 = output2
        self.weight12 = weight12
        self.weight21 = weight21
        self.scores12 = scores12
        self.scores21 = scores21


class GatedCrossStitch(CrossStitch):

    def __init__(self, n_layers, input1_dim, input2_dim, adapter_bottleneck_dim=256, attn_dim=64):
        super().__init__(n_layers, input1_dim, input2_dim)
        self.scaling_factor = torch.sqrt(torch.tensor(input1_dim))
        self.gates12 = nn.ModuleList([
            nn.Linear(2 * input1_dim, 1)
            for _ in range(n_layers)])
        self.gates21 = nn.ModuleList([
            nn.Linear(2 * input2_dim, 1)
            for _ in range(n_layers)])

        if input1_dim == input2_dim:
            adapter_cls = Adapter
            self.cross_attention_scores = self._cross_attention_scores
        else:
            adapter_cls = NoSkipAdapter
            self.proj12 = nn.Linear(input2_dim, input1_dim)
            self.proj21 = nn.Linear(input1_dim, input2_dim)
            self.cross_attention_scores = self._cross_attention_scores_with_proj

        self.adapters12 = nn.ModuleList([
            adapter_cls(input2_dim, adapter_bottleneck_dim, input1_dim)
            for _ in range(n_layers)])
        self.adapters21 = nn.ModuleList([
            adapter_cls(input1_dim, adapter_bottleneck_dim, input2_dim)
            for _ in range(n_layers)])

        if attn_dim > 0:
            self.scaling_factor = torch.sqrt(torch.tensor(input1_dim))
            self.query_proj1 = nn.Linear(input1_dim, attn_dim)
            self.key_proj1 = nn.Linear(input1_dim, attn_dim)
            if input1_dim == input2_dim:
                self.query_proj2 = self.query_proj1
                self.key_proj2 = self.key_proj1
            else:
                self.query_proj2 = nn.Linear(input2_dim, attn_dim)
                self.key_proj2 = nn.Linear(input2_dim, attn_dim)
            self.cross_attention_scores = self._cross_attention_scores_querykey

    def _cross_attention_scores_querykey(self, input1, input2):
        query1 = self.query_proj1(input1)
        key2 = self.key_proj2(input2)
        scores12 = torch.einsum('bid,bjd->bij', query1, key2) / self.scaling_factor

        query2 = self.query_proj2(input2)
        key1 = self.key_proj1(input1)
        scores21 = torch.einsum('bid,bjd->bij', query2, key1) / self.scaling_factor

        return scores12.softmax(dim=2), scores21.softmax(dim=2)


    def _cross_attention_scores(self, input1, input2):
        pairwise_scores = torch.einsum('bid,bjd->bij', input1, input2) / self.scaling_factor
        scores12 = pairwise_scores.softmax(dim=2)
        scores21 = pairwise_scores.transpose(1, 2).softmax(dim=2)
        return scores12, scores21

    def _cross_attention_scores_with_proj(self, input1, input2):
        input1_proj = self.proj21(input1)
        input2_proj = self.proj12(input2)

        pairwise_scores1 = torch.einsum('bid,bjd->bij', input1, input2_proj) / self.scaling_factor
        pairwise_scores2 = torch.einsum('bid,bjd->bij', input1_proj, input2) / self.scaling_factor

        scores12 = pairwise_scores1.softmax(dim=2)
        scores21 = pairwise_scores2.transpose(1, 2).softmax(dim=2)
        return scores12, scores21

    def forward(self, layer_key, input1, input2):
        if input1 is None or input2 is None:
            # no cross stitch
            print('pass through')
            return input1, input2
        scores12, scores21 = self.cross_attention_scores(input1, input2)

        input12 = self.adapters12[layer_key](torch.bmm(scores12, input2))
        weight12 = self.gates12[layer_key](torch.cat([input1, input12], dim=-1)).sigmoid()
        output1 = input1 * weight12 + input12 * (1 - weight12)

        input21 = self.adapters21[layer_key](torch.bmm(scores21, input1))
        weight21 = self.gates21[layer_key](torch.cat([input2, input21], dim=-1)).sigmoid()
        output2 = input2 * weight21 + input21 * (1 - weight21)
        
        # self.i += 1
        # if not self.i % 100:
        #     print(weight12[0].min().item(), weight12[0].max().item(), weight21[0].min().item(), weight21[0].max().item())

        return GatedCrossStitchOut(
            output1=output1,
            output2=output2,
            weight12=weight12,
            weight21=weight21,
            scores12=scores12,
            scores21=scores21,
            )


class ResiCrossStitch(CrossStitch):

    def __init__(self, n_layers, input1_dim, input2_dim, adapter_bottleneck_dim=256, attn_dim=64):
        super().__init__(n_layers, input1_dim, input2_dim)
        self.scaling_factor = torch.sqrt(torch.tensor(input1_dim))
        self.gates12 = nn.ModuleList([
            nn.Linear(2 * input1_dim, 1)
            for _ in range(n_layers)])
        self.gates21 = nn.ModuleList([
            nn.Linear(2 * input2_dim, 1)
            for _ in range(n_layers)])

        if input1_dim == input2_dim:
            adapter_cls = Adapter
            self.cross_attention_scores = self._cross_attention_scores
        else:
            adapter_cls = NoSkipAdapter
            self.proj12 = nn.Linear(input2_dim, input1_dim)
            self.proj21 = nn.Linear(input1_dim, input2_dim)
            self.cross_attention_scores = self._cross_attention_scores_with_proj

        self.adapters12 = nn.ModuleList([
            adapter_cls(input2_dim, adapter_bottleneck_dim, input1_dim)
            for _ in range(n_layers)])
        self.adapters21 = nn.ModuleList([
            adapter_cls(input1_dim, adapter_bottleneck_dim, input2_dim)
            for _ in range(n_layers)])

        if attn_dim > 0:
            self.scaling_factor = torch.sqrt(torch.tensor(input1_dim))
            self.query_proj1 = nn.Linear(input1_dim, attn_dim)
            self.key_proj1 = nn.Linear(input1_dim, attn_dim)
            if input1_dim == input2_dim:
                self.query_proj2 = self.query_proj1
                self.key_proj2 = self.key_proj1
            else:
                self.query_proj2 = nn.Linear(input2_dim, attn_dim)
                self.key_proj2 = nn.Linear(input2_dim, attn_dim)
            self.cross_attention_scores = self._cross_attention_scores_querykey

    def _cross_attention_scores_querykey(self, input1, input2):
        query1 = self.query_proj1(input1)
        key2 = self.key_proj2(input2)
        scores12 = torch.einsum('bid,bjd->bij', query1, key2) / self.scaling_factor

        query2 = self.query_proj2(input2)
        key1 = self.key_proj1(input1)
        scores21 = torch.einsum('bid,bjd->bij', query2, key1) / self.scaling_factor

        return scores12.softmax(dim=2), scores21.softmax(dim=2)


    def _cross_attention_scores(self, input1, input2):
        pairwise_scores = torch.einsum('bid,bjd->bij', input1, input2) / self.scaling_factor
        scores12 = pairwise_scores.softmax(dim=2)
        scores21 = pairwise_scores.transpose(1, 2).softmax(dim=2)
        return scores12, scores21

    def _cross_attention_scores_with_proj(self, input1, input2):
        input1_proj = self.proj21(input1)
        input2_proj = self.proj12(input2)

        pairwise_scores1 = torch.einsum('bid,bjd->bij', input1, input2_proj) / self.scaling_factor
        pairwise_scores2 = torch.einsum('bid,bjd->bij', input1_proj, input2) / self.scaling_factor

        scores12 = pairwise_scores1.softmax(dim=2)
        scores21 = pairwise_scores2.transpose(1, 2).softmax(dim=2)
        return scores12, scores21

    def forward(self, layer_key, input1, input2):
        if input1 is None or input2 is None:
            # no cross stitch
            print('pass through')
            return input1, input2
        scores12, scores21 = self.cross_attention_scores(input1, input2)

        input12 = self.adapters12[layer_key](torch.bmm(scores12, input2))
        output1 = input12 + input1

        input21 = self.adapters21[layer_key](torch.bmm(scores21, input1))
        output2 = input21 + input2
        
        # self.i += 1
        # if not self.i % 100:
        #     print(weight12[0].min().item(), weight12[0].max().item(), weight21[0].min().item(), weight21[0].max().item())

        return output1, output2
    

class AverageCrossStitch(CrossStitch):

    def __init__(self, n_layers, input1_dim, input2_dim):
        super().__init__(n_layers, input1_dim, input2_dim)
        self.scaling_factor = torch.sqrt(torch.tensor(input1_dim))
        self.gates12 = nn.ModuleList([
            nn.Linear(input1_dim + input2_dim, 1)
            for _ in range(n_layers)])
        self.gates21 = nn.ModuleList([
            nn.Linear(input1_dim + input2_dim, 1)
            for _ in range(n_layers)])

    def forward(self, layer_key, input1, input2):
        pairwise_scores = torch.einsum('bid,bjd->bij', input1, input2) / self.scaling_factor

        scores_12 = pairwise_scores.softmax(dim=2)
        input12 = torch.bmm(scores_12, input2)
        weight12 = 0.5
        output1 = input1 * weight12 + input12 * (1 - weight12)

        scores_21 = pairwise_scores.transpose(1, 2).softmax(dim=2)
        input21 = torch.bmm(scores_21, input1)
        weight21 = 0.5
        output2 = input2 * weight21 + input21 * (1 - weight21)

        return output1, output2


class NoCrossStitch(CrossStitch):
    def forward(self, layer_key, input1, input2):
        return input1, input2


class FirstTokenCrossStitch(CrossStitch):

    def __init__(self, n_layers, input1_dim, input2_dim):
        super().__init__(n_layers, input1_dim, input2_dim)
        self.scaling_factor = torch.sqrt(torch.tensor(input1_dim))
        self.gates12 = nn.ModuleList([
            nn.Linear(input1_dim + input2_dim, 1)
            for _ in range(n_layers)])
        self.gates21 = nn.ModuleList([
            nn.Linear(input1_dim + input2_dim, 1)
            for _ in range(n_layers)])

    def forward(self, layer_key, input1, input2):
        input12 = torch.zeros_like(input1)
        input12[:, 0] = input2[:, 0]
        weight12 = 0.5
        output1 = input1 * weight12 + input12 * (1 - weight12)

        input21 = torch.zeros_like(input2)
        input21[:, 0] = input1[:, 0]
        weight21 = 0.5
        output2 = input2 * weight21 + input21 * (1 - weight21)

        return output1, output2


class Alignment(nn.Module):
    def __init__(self, repr1_dim, repr2_dim, attn_dim):
        super().__init__()
        self.query_proj = nn.Linear(repr1_dim, attn_dim)
        self.key_proj = nn.Linear(repr2_dim, attn_dim)
        self.attn = nn.MultiheadAttention(attn_dim, 1, batch_first=True)
        self.crit = nn.MSELoss()

    def forward(self, batch, repr1, repr2, return_instances=False):
        query = self.query_proj(repr1)
        key = self.key_proj(repr2)
        dummy_value = key
        _, attn_scores = self.attn(query, key, dummy_value)
        target = batch['alignment']
        target_mask = target != -100
        # target = target / target.sum(dim=-1, keepdim=True)
        loss = self.crit(attn_scores[target_mask], target[target_mask].float())
        result = {'loss': loss}
        if return_instances:
            result['alignment_pred'] = attn_scores
            result['alignment_target'] = target
        return result
