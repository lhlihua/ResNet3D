import random
import math


class Compose(object):

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, frame_indices):
        for i, t in enumerate(self.transforms):
            if isinstance(frame_indices[0], list):
                next_transforms = Compose(self.transforms[i:])
                dst_frame_indices = [
                    next_transforms(clip_frame_indices)
                    for clip_frame_indices in frame_indices
                ]

                return dst_frame_indices
            else:
                frame_indices = t(frame_indices)
        return frame_indices

# 进行补足batch_size的视频帧
class LoopPadding(object):

    def __init__(self, size):
        self.size = size

    def __call__(self, frame_indices):
        out = frame_indices

        for index in out:
            if len(out) >= self.size:
                break
            out.append(index)
        # print('*'*20)
        # print(len(out))
        return out

# 从视频帧的第一帧进行提取
class TemporalBeginCrop(object):

    def __init__(self, size):
        self.size = size

    def __call__(self, frame_indices):
        out = frame_indices[:self.size]

        for index in out:
            if len(out) >= self.size:
                break
            out.append(index)

        return out

# 从视频帧的中间帧进行提取
class TemporalCenterCrop(object):

    def __init__(self, size):
        self.size = size

    def __call__(self, frame_indices):

        center_index = len(frame_indices) // 2
        begin_index = max(0, center_index - (self.size // 2))
        end_index = min(begin_index + self.size, len(frame_indices))

        out = frame_indices[begin_index:end_index]

        for index in out:
            if len(out) >= self.size:
                break
            out.append(index)

        return out

# 从视频帧的中间某一帧进行提取
class TemporalRandomCrop(object):

    def __init__(self, size):
        self.size = size
        self.loop = LoopPadding(size)

    def __call__(self, frame_indices):

        rand_end = max(0, len(frame_indices) - self.size - 1)
        begin_index = random.randint(0, rand_end)
        end_index = min(begin_index + self.size, len(frame_indices))

        out = frame_indices[begin_index:end_index]

        if len(out) < self.size:
            out = self.loop(out)

        return out

# 事件裁剪视频帧
class TemporalEvenCrop(object):

    def __init__(self, size, n_samples=1):
        self.size = size
        self.n_samples = n_samples
        self.loop = LoopPadding(size)

    def __call__(self, frame_indices):
        n_frames = len(frame_indices)
        stride = max(
            1, math.ceil((n_frames - 1 - self.size) / (self.n_samples - 1)))

        out = []
        for begin_index in frame_indices[::stride]:
            if len(out) >= self.n_samples:
                break
            end_index = min(frame_indices[-1] + 1, begin_index + self.size)
            sample = list(range(begin_index, end_index))

            if len(sample) < self.size:
                out.append(self.loop(sample))
                break
            else:
                out.append(sample)

        return out

# 滑动窗口裁剪视频帧 提取每个视频的所有帧 长度要为size到的整数倍 长度不够则用loop补足
class SlidingWindow(object):

    def __init__(self, size, stride=0):
        self.size = size
        if stride == 0:
            self.stride = self.size
        else:
            self.stride = stride
        self.loop = LoopPadding(size)

    def __call__(self, frame_indices):
        out = []
        n = 1
        for begin_index in frame_indices[::self.stride]:
            end_index = min(frame_indices[-1] + 1, begin_index + self.size)
            sample = list(range(begin_index, end_index))

            if len(sample) < self.size:
                out.append(self.loop(sample))
                break
            else:
                out.append(sample)
                n += 1
                if n > 3:
                  break

        return out

# 提取视频样本 以某一步长为跨度进行提取 例如list[1,2,3,4,5,6,7,8] strode = 则提取的是 [1,3,5,7]
class TemporalSubsampling(object):

    def __init__(self, stride):
        self.stride = stride

    def __call__(self, frame_indices):
        return frame_indices[::self.stride]

# 打乱视频帧
class Shuffle(object):

    def __init__(self, block_size):
        self.block_size = block_size

    def __call__(self, frame_indices):
        frame_indices = [
            frame_indices[i:(i + self.block_size)]
            for i in range(0, len(frame_indices), self.block_size)
        ]
        random.shuffle(frame_indices)
        frame_indices = [t for block in frame_indices for t in block]
        return frame_indices