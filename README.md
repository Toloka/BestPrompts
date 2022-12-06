# Best Prompts for Text-to-Image Models  and How to Find Them

This repository contains code and data for [Best Prompts for Text-to-Image Models and How to Find Them](https://arxiv.org/abs/2209.11711) paper.

## Code

To run the prompt optimization, you need to create a class that gets image generation queries and returns images generated with Stable Diffusion. We use the following interface:
```python
class DiffusionApi:
    def generate(self, prompt, steps=50, scale=7.5, seed=0, height=512, width=512):
        pass
```

Here you pass the `prompt` string, the number of steps, guidance scale, seed number, and shape of the image, and the `generate` function returns a Pillow `Image` object.

To run a genetic optimization, you need to use `optimize.py` script that has the following arguments:
* `--toloka-token`. Toloka API token
* `--aws-access-key-id`. AWS secret key ID. Here we use it to store generated images in an AWS bucket to get direct links that will be embedded into annotation tasks
* `--aws-secret-access-key`. AWS secret access key
* `--endpoint-url`. Base URL at which your images will be stored. In other words, a URL to your AWS bucket
* `--bucket`. Name of the AWS bucket
* `--base-pool-id`. An ID of configured Toloka pool that will be cloned on every optimization iteration

## Data

* The `annotation.csv` contains a CSV file with the results of the pairwise comparisons. It has five columns: `prompt_id` is an ID of an image description, `left_uid` is a UID of four left images (more details below), `right_uid` is a UID of four right images, `worker` is the worker's ID, and `label` is a worker's preference (left or right)
* The `uid_to_keywords.csv` contains a mapping of image UID to keywords that it was obtained with
* `prompts.csv` contains image descriptions. Here index of a prompt is `prompt_id` in `annotation.csv`
* `keywords.csv` contains keywords and their occurrences in the Stable Diffusion Discord

To obtain generated images, you need to use `https://storage.yandexcloud.net/diffusion/` as a base URL and append `{UID}_{0-3}.png` to it. For example,
```
https://storage.yandexcloud.net/diffusion/0000298d546d4a6299774ca323fa7f34_0.png
```
