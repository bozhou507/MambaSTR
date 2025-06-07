from PIL import Image as IM
from PIL.Image import Image
from typing import List, Literal, Tuple


def load_image(img_path: str) -> Image:
    return IM.open(img_path)


def save_image(img: Image, save_path: str):
    img.save(save_path)


def white_image(wh: Tuple[int,int]) -> Image:
    return IM.new('RGB', wh, (255, 255, 255))


def black_image(wh: Tuple[int,int]) -> Image:
    return IM.new('RGB', wh, (0, 0, 0))


def convert_to_rgb(img: Image) -> Image:
    return img.convert(mode='RGB')


def convert_to_gray(img: Image) -> Image:
    return img.convert(mode='L')


def copy(img: Image) -> Image:
    return img.copy()


def crop(img: Image, lu: Tuple[int,int], rb: Tuple[int,int]) -> Image:
    return img.crop((*lu, rb[0] + 1, rb[1] + 1))


def paste(src: Image, dst: Image, lu: Tuple[int,int]):
    """inplace paste src to dst
    
    Args:
        src (PIL.Image): source image
        dst (PIL.Image): target image
        lu (Tuple[int,int]): a point on dst where the upper-left src will be paste
    """
    dst.paste(src, lu)


NEAREST = IM.Resampling.NEAREST
BILINEAR = IM.Resampling.BILINEAR
BICUBIC = IM.Resampling.BICUBIC
def resize(img: Image, wh: Tuple[int,int],
        interplotion: Literal[NEAREST, BILINEAR, BICUBIC]=BILINEAR) -> Image:
    return img.resize(wh, interplotion)


def rotate(img: Image, degrees: float) -> Image:
    """rotated img with the given number of degrees counter clockwise around its centre"""
    return img.rotate(degrees, expand=True, fillcolor=(255, 255, 255))


def split_to_patches(img: Image, patch_wh: Tuple[int,int]) -> List[List[Image]]:
    """patch_size should be (width, height) of one patch"""
    w, h = img.size
    pw, ph = patch_wh
    assert w % pw == 0 and h % ph == 0, f"image size should be divisible by patch size"
    patches = []
    for y in range(0, h, ph):
        row = []
        for x in range(0, w, pw):
            row.append(img.crop([x, y, x + pw, y + ph]))
        patches.append(row)
    return patches


def arrange_patches(patches_list: List[List[Image]], gap_wh: Tuple[int,int]=(5, 5)) -> Image:
    """all patch size should be equal"""
    if isinstance(patches_list[0], Image):
        patches_list = [patches_list,]
    pw, ph = patches_list[0][0].size
    sw = pw + gap_wh[0]
    sh = ph + gap_wh[1]
    w = sw * (len(patches_list[0]) - 1) + pw
    h = sh * (len(patches_list) - 1) + ph
    img = IM.new('RGB', (w, h), (255, 255, 255))
    for y in range(0, h, sh):
        i = y // sh
        assert len(patches_list[i]) == len(patches_list[0]), 'num patches on each row should be equal'
        for x in range(0, w, sw):
            j = x // sw
            assert patches_list[i][j].size == (pw, ph), 'all patches must have same size'
            img.paste(patches_list[i][j], (x, y))
    return img


def pad_patches(patches_list: List[List[Image]]) -> List[List[Image]]:
    row_num = len(max(patches_list, key=lambda x: len(x)))
    image_to_pad = IM.new('RGB', patches_list[0][0].size, (0, 0, 0))
    for i in range(len(patches_list)):
        pad_num = row_num - len(patches_list[i])
        if pad_num > 0:
            patches_list[i].extend([image_to_pad.copy() for _ in range(pad_num)])
    return patches_list


if __name__ == '__main__':
    img_path = 'imgs/a.jpg'
    image = load_image(img_path)
    # image = white_image((100,100))
    # image = black_image((100,100))
    image = resize(image, (32, 32))
    patches = split_to_patches(image, (8, 8))
    patches[-1].pop()
    patches = pad_patches(patches)
    # patches = sum(patches, start=[])
    image = arrange_patches(patches)
    save_image(image, "t.png")
