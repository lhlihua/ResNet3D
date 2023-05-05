import random

from torchvision.transforms import transforms
from torchvision.transforms import functional as F
from PIL import Image

# torchvision.transforms 主要用于一些常见的图片变换 如裁剪旋转等
class Compose(transforms.Compose):

    def randomize_parameters(self):
        for t in self.transforms:
            t.randomize_parameters()

# 占位 使用transforms的ToTensor 转为tensor并归一化为[0-1]
class ToTensor(transforms.ToTensor):

    def randomize_parameters(self):
        pass

# 占位 使用transforms的标准化
class Normalize(transforms.Normalize):

    def randomize_parameters(self):
        pass

# 设置标度值
class ScaleValue(object):

    def __init__(self, s):
        self.s = s

    def __call__(self, tensor):
        tensor *= self.s
        return tensor

    def randomize_parameters(self):
        pass

# 将图片设置为指定大小
class Resize(transforms.Resize):

    def randomize_parameters(self):
        pass

# 同Resize
class Scale(transforms.Scale):

    def randomize_parameters(self):
        pass

# 中心裁剪
class CenterCrop(transforms.CenterCrop):

    def randomize_parameters(self):
        pass

# 四角裁剪
class CornerCrop(object):

    def __init__(self,
                 size,
                 crop_position=None,
                 crop_positions=['c', 'tl', 'tr', 'bl', 'br']):
        self.size = size
        self.crop_position = crop_position
        self.crop_positions = crop_positions

        if crop_position is None:
            self.randomize = True
        else:
            self.randomize = False
        self.randomize_parameters()

    def __call__(self, img):
        image_width = img.size[0]
        image_height = img.size[1]

        h, w = (self.size, self.size)
        if self.crop_position == 'c':
            i = int(round((image_height - h) / 2.))
            j = int(round((image_width - w) / 2.))
        elif self.crop_position == 'tl':
            i = 0
            j = 0
        elif self.crop_position == 'tr':
            i = 0
            j = image_width - self.size
        elif self.crop_position == 'bl':
            i = image_height - self.size
            j = 0
        elif self.crop_position == 'br':
            i = image_height - self.size
            j = image_width - self.size

        img = F.crop(img, i, j, h, w)

        return img

    def randomize_parameters(self):
        if self.randomize:
            self.crop_position = self.crop_positions[random.randint(
                0,
                len(self.crop_positions) - 1)]

    def __repr__(self):
        return self.__class__.__name__ + '(size={0}, crop_position={1}, randomize={2})'.format(
            self.size, self.crop_position, self.randomize)

# 随机水平翻转
class RandomHorizontalFlip(transforms.RandomHorizontalFlip):

    def __init__(self, p=0.5):
        super().__init__(p)
        self.randomize_parameters()

    def __call__(self, img):
        """
        Args:
            img (PIL.Image): Image to be flipped.
        Returns:
            PIL.Image: Randomly flipped image.
        """
        if self.random_p < self.p:
            return F.hflip(img)
        return img

    def randomize_parameters(self):
        self.random_p = random.random()

# 多尺度四角裁剪(选定一个角)
class MultiScaleCornerCrop(object):

    def __init__(self,
                 size,
                 scales,
                 crop_positions=['c', 'tl', 'tr', 'bl', 'br'],
                 interpolation=Image.BILINEAR):
        self.size = size
        self.scales = scales
        self.interpolation = interpolation
        self.crop_positions = crop_positions

        self.randomize_parameters()

    def __call__(self, img):
        short_side = min(img.size[0], img.size[1])
        crop_size = int(short_side * self.scale)
        self.corner_crop.size = crop_size

        img = self.corner_crop(img)
        return img.resize((self.size, self.size), self.interpolation)

    def randomize_parameters(self):
        self.scale = self.scales[random.randint(0, len(self.scales) - 1)]
        crop_position = self.crop_positions[random.randint(
            0,
            len(self.crop_positions) - 1)]

        self.corner_crop = CornerCrop(None, crop_position)

    def __repr__(self):
        return self.__class__.__name__ + '(size={0}, scales={1}, interpolation={2})'.format(
            self.size, self.scales, self.interpolation)

# 随机长宽比裁剪 最终会调整到size大小
class RandomResizedCrop(transforms.RandomResizedCrop):

    def __init__(self,
                 size,
                 scale=(0.08, 1.0),
                 ratio=(3. / 4., 4. / 3.),
                 interpolation=Image.BILINEAR):
        super().__init__(size, scale, ratio, interpolation)
        self.randomize_parameters()

    def __call__(self, img):
        if self.randomize:
            self.random_crop = self.get_params(img, self.scale, self.ratio)
            self.randomize = False

        i, j, h, w = self.random_crop
        return F.resized_crop(img, i, j, h, w, self.size, self.interpolation)

    def randomize_parameters(self):
        self.randomize = True

# 转灰度图
class ColorJitter(transforms.ColorJitter):

    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        super().__init__(brightness, contrast, saturation, hue)
        self.randomize_parameters()

    def __call__(self, img):
        if self.randomize:
            self.transform = self.get_params(self.brightness, self.contrast,
                                             self.saturation, self.hue)
            self.randomize = False

        return self.transform(img)

    def randomize_parameters(self):
        self.randomize = True

# 选择第一个通道
class PickFirstChannels(object):

    def __init__(self, n):
        self.n = n

    def __call__(self, tensor):
        return tensor[:self.n, :, :]

    def randomize_parameters(self):
        pass