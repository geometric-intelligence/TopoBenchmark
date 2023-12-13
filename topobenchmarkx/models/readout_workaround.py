import torch


class ReadOutWorkaround(torch.nn.Module):
    def __init__(self, backbone_outputs):
        super().__init__()
        self.backbone_outputs = backbone_outputs

    def __call__(self, outputs):
        results = {}
        assert len(self.backbone_outputs) == len(
            outputs
        ), f"{len(self.backbone_outputs)} names where passed to ReadOutWorkaround while model outputs are {len(outputs)}"
        for name, o in zip(self.backbone_outputs, outputs):
            results[name] = o
        return results
