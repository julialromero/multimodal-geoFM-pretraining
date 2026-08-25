"""Small sample-level transforms shared by evaluation datasets."""


class ImageSampleTransform:
    """Apply an image transform while retaining the rest of a sample mapping."""

    def __init__(self, transform):
        self.transform = transform

    def __call__(self, sample):
        sample["image"] = self.transform(sample["image"])
        return sample
