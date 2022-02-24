# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Generate images using pretrained network pickle."""

import os
import re
from typing import List, Optional

import click
import dnnlib
import numpy as np
import PIL.Image
import torch

import legacy
import functools
from training import networks
import onnx
import onnxruntime

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
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
# @click.option('--seeds', type=num_range, help='List of random seeds')
# @click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=1, show_default=True)
@click.option('--class', 'class_idx', type=int, help='Class label (unconditional if not specified)')
# @click.option('--noise-mode', help='Noise mode', type=click.Choice(['const', 'random', 'none']), default='const', show_default=True)
@click.option('--projected-w', help='Projection result file', type=str, metavar='FILE')
# @click.option('--outdir', help='Where to save the output images', type=str, required=True, metavar='DIR')
def generate_onnx_model(
    ctx: click.Context,
    network_pkl: str,
#    seeds: Optional[List[int]],
#    truncation_psi: float,
#    noise_mode: str,
#    outdir: str,
    class_idx: Optional[int],
    projected_w: Optional[str]
):
    """Generate images using pretrained network pickle.

    Examples:

    \b
    # Generate curated MetFaces images without truncation (Fig.10 left)
    python generate.py --outdir=out --trunc=1 --seeds=85,265,297,849 \\
        --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metfaces.pkl

    \b
    # Generate uncurated MetFaces images with truncation (Fig.12 upper left)
    python generate.py --outdir=out --trunc=0.7 --seeds=600-605 \\
        --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metfaces.pkl

    \b
    # Generate class conditional CIFAR-10 images (Fig.17 left, Car)
    python generate.py --outdir=out --seeds=0-35 --class=1 \\
        --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/cifar10.pkl

    \b
    # Render an image from projected W
    python generate.py --outdir=out --projected_w=projected_w.npz \\
        --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metfaces.pkl
    """

    print('Loading networks from "%s"...' % network_pkl)

    # Based on https://github.com/NVlabs/stylegan2-ada-pytorch/issues/54

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore

    if network_pkl:
        short_name = network_pkl.split('/')[-1].replace('.pkl', '')
    else:
        short_name = 'default'

    # Synthesize the result of a W projection.
    # if projected_w is not None:
    #     print(f'Generating images from projected W "{projected_w}"')
    #     ws = np.load(projected_w)['w']
    #     ws = torch.tensor(ws, device=device) # pylint: disable=not-callable
    #     assert ws.shape[1:] == (G.num_ws, G.w_dim)
    #     for idx, w in enumerate(ws):
    #         img = G.synthesis(w.unsqueeze(0), noise_mode=noise_mode)
    #         img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    #         img = PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(f'{outdir}/proj-{short_name}-{idx:010d}.png')
    #     return

    # Generate onnx model G_syn
    G.synthesis.forward = functools.partial(G.synthesis.forward, noise_mode="const", force_fp32=True)
    w = torch.from_numpy(np.random.RandomState(0).randn(1, G.num_ws, G.w_dim)).to(device).type(torch.float32)
    print("Input shapes for G_syn model", w.shape)
    torch.onnx.export(G.synthesis, w, short_name + "_G_syn.onnx", verbose=False,
                      input_names=["ws"], output_names=["outputs"])
    print("Generated model", short_name + "_G_syn.onnx")

    # Labels.
    label = torch.zeros([1, G.c_dim], device=device)
    if G.c_dim != 0:
        if class_idx is None:
            ctx.fail('Must specify class label with --class when using a conditional network')
        label[:, class_idx] = 1
    else:
        if class_idx is not None:
            print ('warn: --class=lbl ignored when running on an unconditional network')

    # Generate onnx model G.
    G.forward = functools.partial(G.forward, noise_mode="const", force_fp32=True)
    G_new = networks.Generator(G.z_dim, G.c_dim, G.w_dim, G.img_resolution, G.img_channels).eval().requires_grad_(False)
    G_new.load_state_dict(G.state_dict())

    z = torch.from_numpy(np.random.RandomState(0).randn(1, G.z_dim)).to(device).type(torch.float32)

    args = (z, label, torch.tensor(0.5), None)
    print("Input shapes G model", z.shape, label.shape)
    torch.onnx.export(G_new, args, short_name + "_G.onnx", verbose=False,
                      input_names=["random_z", "label", "truncation"], output_names=["outputs"])
    print("Generated model", short_name + "_G_syn.onnx")


#----------------------------------------------------------------------------

if __name__ == "__main__":
    generate_onnx_model() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
