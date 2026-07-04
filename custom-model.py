from typing import List, Optional, Tuple, Type

import torch
from torch import nn
from transformers import PreTrainedModel, PretrainedConfig


class ResnetConfig(PretrainedConfig):
    model_type = "custom_resnet"

    def __init__(
        self,
        block_type: str = "bottleneck",
        layers: Optional[List[int]] = None,
        num_classes: int = 1000,
        input_channels: int = 3,
        cardinality: int = 1,
        base_width: int = 64,
        stem_width: int = 64,
        stem_type: str = "",
        avg_down: bool = False,
        **kwargs,
    ):
        if block_type not in ["basic", "bottleneck"]:
            raise ValueError(f"`block_type` must be 'basic' or 'bottleneck', got {block_type}.")
        if stem_type not in ["", "deep", "deep-tiered"]:
            raise ValueError(f"`stem_type` must be '', 'deep' or 'deep-tiered', got {stem_type}.")

        self.block_type = block_type
        self.layers = layers or [3, 4, 6, 3]
        self.num_classes = num_classes
        self.input_channels = input_channels
        self.cardinality = cardinality
        self.base_width = base_width
        self.stem_width = stem_width
        self.stem_type = stem_type
        self.avg_down = avg_down
        super().__init__(**kwargs)


def conv3x3(in_channels: int, out_channels: int, stride: int = 1, groups: int = 1) -> nn.Conv2d:
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=1,
        groups=groups,
        bias=False,
    )


def conv1x1(in_channels: int, out_channels: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        cardinality: int = 1,
        base_width: int = 64,
    ):
        super().__init__()
        if cardinality != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports cardinality=1 and base_width=64.")

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        cardinality: int = 1,
        base_width: int = 64,
    ):
        super().__init__()
        width = int(planes * (base_width / 64.0)) * cardinality

        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = conv3x3(width, width, stride, groups=cardinality)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


BLOCK_MAPPING = {
    "basic": BasicBlock,
    "bottleneck": Bottleneck,
}


class ResNet(nn.Module):
    def __init__(
        self,
        block: Type[nn.Module],
        layers: List[int],
        num_classes: int = 1000,
        in_chans: int = 3,
        cardinality: int = 1,
        base_width: int = 64,
        stem_width: int = 64,
        stem_type: str = "",
        avg_down: bool = False,
    ):
        super().__init__()
        self.inplanes = stem_width
        self.cardinality = cardinality
        self.base_width = base_width
        self.avg_down = avg_down

        if stem_type:
            mid_width = stem_width // 2 if stem_type == "deep-tiered" else stem_width
            self.stem = nn.Sequential(
                conv3x3(in_chans, mid_width, stride=2),
                nn.BatchNorm2d(mid_width),
                nn.ReLU(inplace=True),
                conv3x3(mid_width, mid_width),
                nn.BatchNorm2d(mid_width),
                nn.ReLU(inplace=True),
                conv3x3(mid_width, stem_width),
                nn.BatchNorm2d(stem_width),
                nn.ReLU(inplace=True),
            )
        else:
            self.stem = nn.Sequential(
                nn.Conv2d(in_chans, stem_width, kernel_size=7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(stem_width),
                nn.ReLU(inplace=True),
            )

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        self._init_weights()

    def _make_downsample(self, inplanes: int, outplanes: int, stride: int) -> nn.Module:
        if self.avg_down and stride != 1:
            return nn.Sequential(
                nn.AvgPool2d(kernel_size=stride, stride=stride, ceil_mode=True, count_include_pad=False),
                conv1x1(inplanes, outplanes),
                nn.BatchNorm2d(outplanes),
            )
        return nn.Sequential(conv1x1(inplanes, outplanes, stride), nn.BatchNorm2d(outplanes))

    def _make_layer(
        self,
        block: Type[nn.Module],
        planes: int,
        blocks: int,
        stride: int = 1,
    ) -> nn.Sequential:
        outplanes = planes * block.expansion
        downsample = None
        if stride != 1 or self.inplanes != outplanes:
            downsample = self._make_downsample(self.inplanes, outplanes, stride)

        layers = [
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                cardinality=self.cardinality,
                base_width=self.base_width,
            )
        ]
        self.inplanes = outplanes
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    cardinality=self.cardinality,
                    base_width=self.base_width,
                )
            )
        return nn.Sequential(*layers)

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.01)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


class ResnetModelForImageClassification(PreTrainedModel):
    config_class = ResnetConfig
    base_model_prefix = "resnet"

    def __init__(self, config: ResnetConfig):
        super().__init__(config)
        block_layer = BLOCK_MAPPING[config.block_type]
        self.resnet = ResNet(
            block_layer,
            config.layers,
            num_classes=config.num_classes,
            in_chans=config.input_channels,
            cardinality=config.cardinality,
            base_width=config.base_width,
            stem_width=config.stem_width,
            stem_type=config.stem_type,
            avg_down=config.avg_down,
        )

    def forward(
        self,
        pixel_values: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> dict:
        logits = self.resnet(pixel_values)
        loss = None
        if labels is not None:
            loss = torch.nn.functional.cross_entropy(logits, labels)
        return {"loss": loss, "logits": logits}


if __name__ == "__main__":
    resnet50d_config = ResnetConfig(
        block_type="bottleneck",
        stem_width=32,
        stem_type="deep",
        avg_down=True,
    )
    resnet50d = ResnetModelForImageClassification(resnet50d_config)

    inputs = torch.randn(2, 3, 224, 224)
    labels = torch.tensor([0, 1])
    outputs = resnet50d(inputs, labels=labels)
    print("loss:", float(outputs["loss"]))
    print("logits shape:", tuple(outputs["logits"].shape))

    resnet50d.save_pretrained("custom-resnet")
    loaded_model = ResnetModelForImageClassification.from_pretrained("custom-resnet")
    print("loaded logits shape:", tuple(loaded_model(inputs)["logits"].shape))
