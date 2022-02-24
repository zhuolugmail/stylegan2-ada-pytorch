# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Generate images using onnx model."""

import os
import re
from typing import List, Optional

import click
import numpy as np
import PIL.Image
import torch

import onnx
import onnxruntime as ort
from google.protobuf.json_format import MessageToDict


#----------------------------------------------------------------------------

def num_range(s: str) -> List[int]:
    '''Accept either a comma separated list of numbers 'a,b,c' or a range 'a-c' and return as a list of ints.'''

    range_re = re.compile(r'^(\d+)-(\d+)$')
    m = range_re.match(s)
    if m:
        return list(range(int(m.group(1)), int(m.group(2))+1))
    vals = s.split(',')
    return [int(x) for x in vals]

#----------------------------------------------------------------------------

@click.command()
@click.pass_context
@click.option('--model', 'model_name', help='ONNX model', required=True)
@click.option('--seeds', type=num_range, help='List of random seeds')
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=1, show_default=True)
@click.option('--class', 'class_idx', type=int, help='Class label (unconditional if not specified)')
@click.option('--projected-w', help='Projection result file', type=str, metavar='FILE')
@click.option('--outdir', help='Where to save the output images', type=str, default="./", metavar='DIR')
def generate_images_from_onnx(
    ctx: click.Context,
    model_name: str,
    seeds: Optional[List[int]],
    truncation_psi: float,
    outdir: str,
    class_idx: Optional[int],
    projected_w: Optional[str]
):
    """Generate images using a onnx model.

    Examples:

    \b
    # Generate curated MetFaces images without truncation (Fig.10 left)
    python generate.py --outdir=out --trunc=1 --seeds=85,265,297,849 \\
        --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metfaces.pkl

    """

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    if not os.path.exists(model_name):
        raise Exception("[ ERROR ] The onnx model file does not exist")

    print('Loading model from "%s"...' % model_name)

    # Load the ONNX model
    model = onnx.load(model_name)

    # Check that the model is well formed
    onnx.checker.check_model(model)

    # Print a human readable representation of the graph
    # print(onnx.helper.printable_graph(model.graph))

    z_dim = 0
    c_dim = 0
    ws_dim = None
    print("Input shapes:")
    for _input in model.graph.input:
        print(MessageToDict(_input))
        name = _input.name
        dim = _input.type.tensor_type.shape.dim
        input_shape = [MessageToDict(d).get("dimValue") for d in dim]
        print("\t", name, input_shape)
        if (name == "random_z"):
            z_dim = int(input_shape[1])
        if (name == "label"):
            c_dim = int(input_shape[1])
        if (name == "ws"):
            ws_dim = (int(input_shape[1]), int(input_shape[2]))

    os.makedirs(outdir, exist_ok=True)

    if model_name:
        short_name = model_name.split('/')[-1].replace('.onnx', '')
    else:
        short_name = 'default'

    # Synthesize the result of a W projection.
    if projected_w is not None:
        assert ws_dim
        if seeds is not None:
            print ('warn: --seeds is ignored when using --projected-w')
        
        print(f'Generating images from projected W "{projected_w}"')
        ws = np.load(projected_w)['w']
        ws = torch.tensor(ws, dtype=torch.float32, device=device) # pylint: disable=not-callable
        assert ws.shape[1:] == ws_dim

        ort_session = ort.InferenceSession(model_name)
        for idx, w in enumerate(ws):
            outputs = ort_session.run(
                None,
                {"ws": w.unsqueeze(0).numpy()}
            )
            img = torch.tensor(outputs[0], dtype=torch.float32)
            img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(f'{outdir}/proj-{short_name}-{idx:010d}.png')
        return

    assert z_dim > 0

    if seeds is None:
        ctx.fail('--seeds option is required when not using --projected-w')

    # Labels.
    label = torch.zeros([1, c_dim], device=device)
    if c_dim != 0:
        if class_idx is None:
            ctx.fail('Must specify class label with --class when using a conditional network')
        label[:, class_idx] = 1
    else:
        if class_idx is not None:
            print ('warn: --class=lbl ignored when running on an unconditional network')
 
    # Generate images.
    ort_session = ort.InferenceSession(model_name)

    for seed_idx, seed in enumerate(seeds):
        print('Generating image for seed %d (%d/%d) ...' % (seed, seed_idx, len(seeds)))
        z = np.random.RandomState(seed).randn(1, z_dim).astype(np.float32)

        outputs = ort_session.run(
            None,
            {"random_z": z, "truncation": np.array([truncation_psi], dtype=np.float32)},
        )

        img = torch.tensor(outputs[0])
        img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        if c_dim != 0:
            PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB') \
                .save(f'{outdir}/{short_name}-c{class_idx}-{seed:010d}-{truncation_psi}.png')
        else:
            PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB') \
                .save(f'{outdir}/{short_name}-{seed:010d}-{truncation_psi}.png')




#----------------------------------------------------------------------------

if __name__ == "__main__":
    generate_images_from_onnx() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
