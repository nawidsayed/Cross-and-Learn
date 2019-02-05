# ===================================================================================================
# This module was taken from torchvision and is modified to handle datasets in a deterministic manner
# It can also returns all generated random variables as an additional tuple by changing the 'mode' keyword
# ===================================================================================================

import random
from PIL import Image, ImageOps, ImageFilter, ImageEnhance
import numpy as np
import numbers
import types
import torch
from copy import copy


class Compose(object):
    """Composes several transforms together.

    Args:
        transforms (List[Transform]): list of transforms to compose.

    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms, mode='silent'):
        self.transforms = transforms
        self.mode = mode

    def __call__(self, img, random=None):
        if random == None:
            random = np.random.RandomState()
        rand = ()
        for t in self.transforms:
            img = t(img, random)
            if isinstance(img, tuple):
                rand += img[1:]
                img = img[0]
        if self.mode == 'silent':
            return img
        return img, rand


class ToTensor(object):
    """Converts a PIL.Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    """

    def __call__(self, pic, random=None):
        if isinstance(pic, list):
            tensors = []
            for img in pic:
                tensors.append(self.__call__(img, random))
            return tensors
        if isinstance(pic, np.ndarray):
            # handle numpy array
            img = torch.from_numpy(pic.transpose((2, 0, 1)))
            # backard compability
            return img.float().div(255)
        # handle PIL Image
        if pic.mode == 'I':
            img = torch.from_numpy(np.array(pic, np.int32, copy=False))
        elif pic.mode == 'I;16':
            img = torch.from_numpy(np.array(pic, np.int16, copy=False))
        else:
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
        # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
        if pic.mode == 'YCbCr':
            nchannel = 3
        elif pic.mode == 'I;16':
            nchannel = 1
        else:
            nchannel = len(pic.mode)
        img = img.view(pic.size[1], pic.size[0], nchannel)
        # put it from HWC to CHW format
        # yikes, this transpose takes 80% of the loading time/CPU
        img = img.transpose(0, 1).transpose(0, 2).contiguous()
        if isinstance(img, torch.ByteTensor):
            return img.float().div(255)
        else:
            return img


class ToPILImage(object):
    """Converts a torch.*Tensor of shape C x H x W or a numpy ndarray of shape
    H x W x C to a PIL.Image while preserving value range.
    """

    def __call__(self, pic, random=None):
        npimg = pic
        mode = None
        if isinstance(pic, torch.FloatTensor):
            pic = pic.mul(255).byte()
        if torch.is_tensor(pic):
            npimg = np.transpose(pic.numpy(), (1, 2, 0))
        assert isinstance(npimg, np.ndarray), 'pic should be Tensor or ndarray'
        if npimg.shape[2] == 1:
            npimg = npimg[:, :, 0]

            if npimg.dtype == np.uint8:
                mode = 'L'
            if npimg.dtype == np.int16:
                mode = 'I;16'
            if npimg.dtype == np.int32:
                mode = 'I'
            elif npimg.dtype == np.float32:
                mode = 'F'
        else:
            if npimg.dtype == np.uint8:
                mode = 'RGB'
        assert mode is not None, '{} is not supported'.format(npimg.dtype)
        return Image.fromarray(npimg, mode=mode)


class Normalize(object):
    """Given mean: (R, G, B) and std: (R, G, B),
    will normalize each channel of the torch.*Tensor, i.e.
    channel = (channel - mean) / std
    """

    def __init__(self, mean, std, inverted=False):
        if not inverted:
            self.mean = mean
            self.std = std
        else:
            self.mean = list(- np.array(mean) / np.array(std))
            self.std = list(1 / np.array(std))

    def __call__(self, tensor, random=None):
        if isinstance(tensor, list):
            tensors = []
            for img in tensor:
                tensors.append(self.__call__(img, random))
            return tensors
        else:
            for t, m, s in zip(tensor, self.mean, self.std):
                t.sub_(m).div_(s)
            return tensor


class Scale(object):
    """Rescales the input PIL.Image to the given 'size'.
    'size' will be the size of the smaller edge.
    For example, if height > width, then image will be
    rescaled to (size * height / width, size)
    size: size of the smaller edge
    interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img, random=None):
        w, h = img.size
        if (w <= h and w == self.size) or (h <= w and h == self.size):
            return img
        if w < h:
            ow = self.size
            oh = int(self.size * h / w)
            return img.resize((ow, oh), self.interpolation)
        else:
            oh = self.size
            ow = int(self.size * w / h)
            return img.resize((ow, oh), self.interpolation)


class CenterCrop(object):
    """Crops the given PIL.Image at the center to have a region of
    the given size. size can be a tuple (target_height, target_width)
    or an integer, in which case the target will be of a square shape (size, size)
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img, random=None):
        w, h = img.size
        th, tw = self.size
        x1 = int(round((w - tw) / 2.))
        y1 = int(round((h - th) / 2.))
        return img.crop((x1, y1, x1 + tw, y1 + th))

class HorizontalFlip(object):
    """If initialized with True, horizontally flips the given PIL.Image 
    """
    def __init__(self, flip=True):
        self.flip = flip

    def __call__(self, img, random=None):
        if self.flip:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        return img

class Smoothen(object):
    """Smoothens the image with gaussian kernel
    """

    def __init__(self, kernel_size):
        self.kernel_size = kernel_size

    def __call__(self, img, random=None):
        if self.kernel_size == 0:
            return img
        elif isinstance(img, list):
            tensors = []
            for pic in img:
                tensors.append(self.__call__(pic, random))
            return tensors
        else:
            return img.filter(ImageFilter.GaussianBlur(self.kernel_size))


class SubMeanDisplacement(object):
    def __call__(self, img, random=None):
        if isinstance(img, list):
            imgs = []
            for image in img:
                imgs.append(self.__call__(image, random))
            return imgs
        else:
            if img.size()[0] == 1:
                mean = img.mean()
                img -= mean
            return img



class Pad(object):
    """Pads the given PIL.Image on all sides with the given "pad" value"""

    def __init__(self, padding, fill=0):
        assert isinstance(padding, numbers.Number)
        assert isinstance(fill, numbers.Number) or isinstance(fill, str) or isinstance(fill, tuple)
        self.padding = padding
        self.fill = fill

    def __call__(self, img, random=None):
        return ImageOps.expand(img, border=self.padding, fill=self.fill)


class Lambda(object):
    """Applies a lambda as a transform."""

    def __init__(self, lambd):
        assert isinstance(lambd, types.LambdaType)
        self.lambd = lambd

    def __call__(self, img, random=None):
        return self.lambd(img)

class RandomCrop(object):
    """Crops the given PIL.Image at a random location to have a region of
    the given size. size can be a tuple (target_height, target_width)
    or an integer, in which case the target will be of a square shape (size, size)
    """

    def __init__(self, size, padding=0, max_shift=0):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding
        self.max_shift = max_shift

    def __call__(self, img, random=None):
        if random == None:
            random = np.random.RandomState()
        if self.padding > 0:
            img = ImageOps.expand(img, border=self.padding, fill=0)

        w, h = img.size
        th, tw = self.size
        if w == tw and h == th:
            return img

        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)

        x1 = np.random.randint(np.max([0, x1-self.max_shift]), np.min([w-tw, x1+1+self.max_shift]))
        y1 = np.random.randint(np.max([0, y1-self.max_shift]), np.min([h-th, y1+1+self.max_shift]))
        return img.crop((x1, y1, x1 + tw, y1 + th)), x1, y1


class RandomHorizontalFlip(object):
    """Randomly horizontally flips the given PIL.Image with a probability of 0.5
    """

    def __call__(self, img, random=None):
        if random == None:
            random = np.random.RandomState()
        r = random.rand() 
        if r < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        return img, r

class SplitChannels(object):
    def __init__(self, use_rand=True, train=True):
        self.use_rand = use_rand  
        self.train = train

    def __call__(self, img, random=None):
        if random == None:
            random = np.random.RandomState()  
        img = np.array(img) 
        if img.shape[2] == 3:
            if self.train:
                rand = random.randint(0, 3)
                if not self.use_rand:
                    rand = np.random.randint(0, 3) 
                for channel in range(3):
                    img[:,:,channel] = img[:,:,rand]
            else:
                avg = np.mean(img, axis=2)
                for channel in range(3):
                    img[:,:,channel] = avg
                img = img.astype(np.uint8)
        img = Image.fromarray(img)
        return img

class RandomColorJitter(object):
    """Randomly change the brightness, contrast and saturation of an image.
    Args:
        brightness (float): How much to jitter brightness. brightness_factor
            is chosen uniformly from [max(0, 1 - brightness), 1 + brightness].
        contrast (float): How much to jitter contrast. contrast_factor
            is chosen uniformly from [max(0, 1 - contrast), 1 + contrast].
        saturation (float): How much to jitter saturation. saturation_factor
            is chosen uniformly from [max(0, 1 - saturation), 1 + saturation].
        hue(float): How much to jitter hue. hue_factor is chosen uniformly from
            [-hue, hue]. Should be >=0 and <= 0.5.
    """
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0, use_rand=True):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self.use_rand = use_rand

    @staticmethod
    def get_params(brightness, contrast, saturation, hue, random):
        """Get a randomized transform to be applied on image.
        Arguments are same as that of __init__.
        Returns:
            Transform which randomly adjusts brightness, contrast and
            saturation in a random order.
        """
        transforms = []
        if brightness > 0:
            brightness_factor = random.uniform(max(0, 1 - brightness), 1 + brightness)
            transforms.append(Lambda(lambda img: adjust_brightness(img, brightness_factor)))

        if contrast > 0:
            contrast_factor = random.uniform(max(0, 1 - contrast), 1 + contrast)
            transforms.append(Lambda(lambda img: adjust_contrast(img, contrast_factor)))

        if saturation > 0:
            saturation_factor = random.uniform(max(0, 1 - saturation), 1 + saturation)
            transforms.append(Lambda(lambda img: adjust_saturation(img, saturation_factor)))

        if hue > 0:
            hue_factor = random.uniform(-hue, hue)
            transforms.append(Lambda(lambda img: adjust_hue(img, hue_factor)))

        random.shuffle(transforms)
        transform = Compose(transforms)

        return transform

    def __call__(self, img, random=None):
        """
        Args:
            img (PIL Image): Input image.
        Returns:
            PIL Image: Color jittered image.
        """
        if random == None:
            random = np.random.RandomState()

        if img.mode == 'RGB':    
            transform = self.get_params(self.brightness, self.contrast,
                                        self.saturation, self.hue, random)
            if not self.use_rand:
                random2 = np.random.RandomState()
                transform = self.get_params(self.brightness, self.contrast,
                                            self.saturation, self.hue, random2)
            return transform(img)
        return img


def adjust_brightness(img, brightness_factor):
    """Adjust brightness of an Image.
    Args:
        img (PIL Image): PIL Image to be adjusted.
        brightness_factor (float):  How much to adjust the brightness. Can be
            any non negative number. 0 gives a black image, 1 gives the
            original image while 2 increases the brightness by a factor of 2.
    Returns:
        PIL Image: Brightness adjusted image.
    """

    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(brightness_factor)
    return img


def adjust_contrast(img, contrast_factor):
    """Adjust contrast of an Image.
    Args:
        img (PIL Image): PIL Image to be adjusted.
        contrast_factor (float): How much to adjust the contrast. Can be any
            non negative number. 0 gives a solid gray image, 1 gives the
            original image while 2 increases the contrast by a factor of 2.
    Returns:
        PIL Image: Contrast adjusted image.
    """

    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(contrast_factor)
    return img


def adjust_saturation(img, saturation_factor):
    """Adjust color saturation of an image.
    Args:
        img (PIL Image): PIL Image to be adjusted.
        saturation_factor (float):  How much to adjust the saturation. 0 will
            give a black and white image, 1 will give the original image while
            2 will enhance the saturation by a factor of 2.
    Returns:
        PIL Image: Saturation adjusted image.
    """

    enhancer = ImageEnhance.Color(img)
    img = enhancer.enhance(saturation_factor)
    return img


def adjust_hue(img, hue_factor):
    """Adjust hue of an image.
    The image hue is adjusted by converting the image to HSV and
    cyclically shifting the intensities in the hue channel (H).
    The image is then converted back to original image mode.
    `hue_factor` is the amount of shift in H channel and must be in the
    interval `[-0.5, 0.5]`.
    See https://en.wikipedia.org/wiki/Hue for more details on Hue.
    Args:
        img (PIL Image): PIL Image to be adjusted.
        hue_factor (float):  How much to shift the hue channel. Should be in
            [-0.5, 0.5]. 0.5 and -0.5 give complete reversal of hue channel in
            HSV space in positive and negative direction respectively.
            0 means no shift. Therefore, both -0.5 and 0.5 will give an image
            with complementary colors while 0 gives the original image.
    Returns:
        PIL Image: Hue adjusted image.
    """

    input_mode = img.mode
    if input_mode in {'L', '1', 'I', 'F'}:
        return img

    h, s, v = img.convert('HSV').split()

    np_h = np.array(h, dtype=np.uint8)
    # uint8 addition take cares of rotation across boundaries
    with np.errstate(over='ignore'):
        np_h += np.uint8(hue_factor * 255)
    h = Image.fromarray(np_h, 'L')

    img = Image.merge('HSV', (h, s, v)).convert(input_mode)
    return img




class TenCrop(object):
    """Returns tencrop of given PIL.Image for testtime data augmentation, random is not used here
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img, random=None):
        images = []
        img_f = img.transpose(Image.FLIP_LEFT_RIGHT)
        return self._five_crop(img) + self._five_crop(img_f)

    def _five_crop(self, img):
        w, h = img.size
        crop_h, crop_w = self.size
        tl = img.crop((0, 0, crop_w, crop_h))
        tr = img.crop((w - crop_w, 0, w, crop_h))
        bl = img.crop((0, h - crop_h, crop_w, h))
        br = img.crop((w - crop_w, h - crop_h, w, h))
        center = self._center_crop(img)
        return [tl, tr, bl, br, center]

    def _center_crop(self, img):
        w, h = img.size
        th, tw = self.size
        i = int(round((h - th) / 2.))
        j = int(round((w - tw) / 2.))
        return img.crop((j, i, j+tw, i+th))

class CCCrop(object):
    def __init__(self, size, mode=0):
        self.mode = mode
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img, random=None):
        w, h = img.size
        crop_h, crop_w = self.size
        if self.mode == 0:
            return self._center_crop(img)
        if self.mode == 1: #tl
            return img.crop((0, 0, crop_w, crop_h))
        if self.mode == 2: #tr
            return img.crop((w - crop_w, 0, w, crop_h))
        if self.mode == 3: #bl
            return img.crop((0, h - crop_h, crop_w, h))
        if self.mode == 4: #br
            return img.crop((w - crop_w, h - crop_h, w, h))        

    def _center_crop(self, img):
        w, h = img.size
        th, tw = self.size
        i = int(round((h - th) / 2.))
        j = int(round((w - tw) / 2.))
        return img.crop((j, i, j+tw, i+th))

# THIS METHOD MIGHT NOT WORK
class RandomSizedCrop(object):
    """Random crop the given PIL.Image to a random size of (0.08 to 1.0) of the original size
    and and a random aspect ratio of 3/4 to 4/3 of the original aspect ratio
    This is popularly used to train the Inception networks
    size: size of the smaller edge
    interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img, random=None):
        if random == None:
            random = np.random.RandomState()
        for attempt in range(10):
            area = img.size[0] * img.size[1]
            target_area = random.uniform(0.08, 1.0) * area
            aspect_ratio = random.uniform(3. / 4, 4. / 3)

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if random.rand() < 0.5:
                w, h = h, w

            if w <= img.size[0] and h <= img.size[1]:
                x1 = random.randint(0, img.size[0] - w)
                y1 = random.randint(0, img.size[1] - h)

                img = img.crop((x1, y1, x1 + w, y1 + h))
                assert(img.size == (w, h))

                return img.resize((self.size, self.size), self.interpolation)

        # Fallback
        scale = Scale(self.size, interpolation=self.interpolation)
        crop = CenterCrop(self.size)
        return crop(scale(img))



