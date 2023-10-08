
# bedrock-sdxl

Simple script for using Stable Diffusion XL text-to-image on AWS Bedrock

I wrote this to mess around. Use at your own risk.

## Prequisites / Setup

- Get an AWS account
- Get model access to Stable Diffusion XL in AWS Bedrock
- Generate access keys with permission to use the Bedrock service and the SDXL model
- Edit these two files on your machine as follows:

~/.aws/profile

```
[profile bedrock-sdxl]
# or whatever region you're using
region=us-east-1
```

~/.aws/credentials

```
[bedrock-sdxl]
aws_access_key_id=PASTE_YOUR_KEY_HERE
aws_secret_access_key=PASTE_YOUR_KEY_HERE
```

## Install

```console
python -m venv venv
source ./venv/bin/activate
pip install -r requirements.txt
```

## Usage

Single text prompt. A default seed is calculated using your machine ID,
so the hash in your filename will be different. The image will open on
your computer if `xdg-open` is available.

```console
$ ./bedrock-sdxl.py -p "a swimming duck"
writing output/a_duck_eating_a_sandwich_888eceb0.json
writing output/a_duck_eating_a_sandwich_888eceb0.png
```

Text prompt with multiple weights

```console
$ ./bedrock-sdxl.py -p "a swimming duck" 1.0 "a swimming elephant" 0.7
```

Text prompt with multiple weights, including negative weight

```console
$ ./bedrock-sdxl.py -p "a swimming duck" 1.0 "a swimming elephant" 0.7 "orange sky" -1.0
```

Setting 'steps' body parameter to 75 (see
[these docs](https://platform.stability.ai/docs/api-reference#tag/v1generation/operation/textToImage)
for a complete list of parameters)

```console
$ ./bedrock-sdxl.py -p "a swimming duck" 1.0 "a swimming elephant" 0.7 "orange sky" -1.0 -b steps 75
```

To (re)generate an image from parameters stored in a .json file

```console
$ ./bedrock-sdxl.py -r output/a_duck_eating_a_sandwich_888eceb0.json
```

To see all the options

```console
$ ./bedrock-sdxl.py -h
```
