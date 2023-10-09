#!/usr/bin/env python

import argparse
import base64
import hashlib
import json
import os
import os.path
import pprint
import re
import shutil
import sys
from types import SimpleNamespace
from typing import Dict, List
import uuid

import boto3

MODEL_ID = "stability.stable-diffusion-xl"
SEED_MAX = 4294967295


def normalize_str(s: str) -> str:
    # strip punctuation and quotes
    s = re.sub(r"[,\.'\"]", "", s)
    # replace remaining non-alphanumerics with underscores
    s = re.sub("[^A-Za-z0-9]", "_", s)
    # replace multiple underscores with a single one
    s = re.sub("_{2,}", "_", s)
    # strip underscores at beg and end
    s = re.sub("^_*", "", s)
    s = re.sub("_*$", "", s)
    return s


def normalize_prompts(prompts: List) -> str:
    """
    normalized prompts input to a filename-appropriate str
    """
    return "_".join([normalize_str(prompt['text']) for prompt in prompts])


def pairs(_list: List) -> List[List]:
    """
    transforms a flat list into a list of lists of pairs
    """
    return list(zip(*(iter(_list),) * 2))


def get_stable_seed() -> int:
    """
    get a seed that's repeatable per machine
    """
    return uuid.getnode() % SEED_MAX


def get_body_defaults() -> Dict:
    return {
        "seed": get_stable_seed(),
    }


def create_runtime(profile_name):
    session = boto3.Session(profile_name=profile_name)
    return session.client(
        service_name="bedrock-runtime"
        )


def write_b64_str(s: str, filename: str):
    img_data = base64.b64decode(s)
    with open(filename, "wb") as f:
        f.write(img_data)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Stable Diffusion XL text-to-image on AWS Bedrock')
    parser.add_argument("-p", "--prompts",
                        help='a single string, or one or more pairs of string/weight values',
                        metavar='prompt',
                        nargs='+',
                        type=str,
                        )
    parser.add_argument('-b', "--body-params",
                        action='store',
                        help="pairs of body parameters and values",
                        metavar=('params'),
                        nargs="*"
                        )
    parser.add_argument('-o', "--output-dir",
                        action='store',
                        default='output',
                        help="output directory (default is output/)"
                        )
    parser.add_argument('-r', "--request",
                        action='store',
                        help="use parameters in .json file"
                        )
    parser.add_argument('-a', "--aws-profile",
                        action='store',
                        default='bedrock-sdxl',
                        help="aws profile to use from ~/.aws/config (defaults to bedrock-sdxl)"
                        )
    parser.add_argument('--debug',
                        action='store_true'
                        )
    args = parser.parse_args()

    if not (args.prompts or args.request):
        print("Error: you must use either -p or -r")
        parser.print_help()
        sys.exit(1)

    if args.prompts:
        if len(args.prompts) > 2 and len(args.prompts) % 2 != 0:
            print("Error: prompts must be a single string, or one or more pairs of string/weight values")
            parser.print_help()
            sys.exit(1)

    if args.body_params:
        if len(args.body_params) % 2 != 0:
            print("Error: body params must be in pairs")
            parser.print_help()
            sys.exit(1)

    return args


def parse_prompt(prompts: List) -> List[Dict]:
    """
    parse a list of prompts provided as input into dicts
    expected by API
    """
    _prompts = list(prompts)
    if len(_prompts) == 1:
        _prompts.append(1.0)
    return [{"text": pair[0], "weight": float(pair[1])} for pair in pairs(_prompts)]


def generate_image(args: SimpleNamespace):
    runtime = create_runtime(args.aws_profile)

    ####
    # sanitize inputs and create request body

    if args.request:
        with open(args.request) as f:
            body = json.loads(f.read())
    else:
        body_params = {}
        if args.body_params:
            body_params = dict(pairs(args.body_params))
            for k, v in body_params.items():
                if k in ['height', 'width', 'cfg_scale', 'samples', 'seed', 'steps']:
                    body_params[k] = int(v)

        prompts = parse_prompt(args.prompts)

        # https://platform.stability.ai/docs/api-reference#tag/v1generation/operation/textToImage

        body = {
            **get_body_defaults(),
            **body_params,
            "text_prompts": prompts,
            }

    if args.debug:
        pprint.pprint(body)

    ####
    # make the request

    body_serialized = json.dumps(body)

    response = runtime.invoke_model(
        body=body_serialized,
        modelId="stability.stable-diffusion-xl")

    response_body = json.loads(response.get("body").read())

    if response_body["result"] != 'success':
        print(response_body["result"])
        sys.exit(1)

    ####
    # write the file

    artifacts = response_body["artifacts"]
    # API says you can request up to 10 samples but apparently the Bedrock
    # version only allows a max of 1, so this loop is kind of moot
    for i, artifact in enumerate(artifacts):
        suffix = f"_{i}" if i > 0 else ""

        body_hash = hashlib.sha256(body_serialized.encode('utf-8')).hexdigest()[0:8]
        filename_base = f"{normalize_prompts(body['text_prompts'])}{suffix}_{body_hash}"
        filename_json = os.path.join(args.output_dir, f"{filename_base}.json")
        filename_img = os.path.join(args.output_dir, f"{filename_base}.png")

        if not args.request:
            print(f"writing {filename_json}")
            with open(filename_json, "w") as f:
                f.write(body_serialized)

        base_64_img_str = artifact.get("base64")

        print(f"writing {filename_img}")
        write_b64_str(base_64_img_str, filename_img)

        if shutil.which("xdg-open"):
            os.system(f"xdg-open {filename_img}")


if __name__ == '__main__':
    args = parse_args()
    generate_image(args)
