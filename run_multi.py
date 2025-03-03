import os
# os.environ['ATTN_BACKEND'] = 'xformers'   # Can be 'flash-attn' or 'xformers', default is 'flash-attn'
os.environ['SPCONV_ALGO'] = 'native'        # Can be 'native' or 'auto', default is 'auto'.
                                            # 'auto' is faster but will do benchmarking at the beginning.
                                            # Recommended to set to 'native' if run only once.

import numpy as np
import imageio
from PIL import Image
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils import render_utils

# Load a pipeline from a model folder or a Hugging Face model hub.
#pipeline = TrellisImageTo3DPipeline.from_pretrained("JeffreyXiang/TRELLIS-image-large")
pipeline = TrellisImageTo3DPipeline.from_pretrained("/home/liu/code/TRELLIS/path/to/TRELLIS-image-large")
pipeline.cuda()

# Load an image
'''image_paths = [
    "assets/example_multi_image/yoimiya_1.png",
    "assets/example_multi_image/yoimiya_2.png",
    "assets/example_multi_image/yoimiya_3.png"
]'''
image_paths = [
    "assets/multi/mouse1.jpg",
    "assets/multi/mouse2.jpg",
    "assets/multi/mouse3.jpg"
]
images = [Image.open(img_path) for img_path in image_paths]

# Extract the common part of the image names (before the numbers)
image_name = os.path.commonprefix([os.path.splitext(os.path.basename(img_path))[0] for img_path in image_paths])

output_dir = os.path.join("Output_multi", image_name)
os.makedirs(output_dir, exist_ok=True)

# Run the pipeline
outputs = pipeline.run_multi_image(
    images,
    seed=1,
    # Optional parameters
    sparse_structure_sampler_params={
        "steps": 12,
        "cfg_strength": 7.5,
    },
    slat_sampler_params={
        "steps": 12,
        "cfg_strength": 3,
    },
)
# outputs is a dictionary containing generated 3D assets in different formats:
# - outputs['gaussian']: a list of 3D Gaussians
# - outputs['radiance_field']: a list of radiance fields
# - outputs['mesh']: a list of meshes

video_gs = render_utils.render_video(outputs['gaussian'][0])['color']
video_mesh = render_utils.render_video(outputs['mesh'][0])['normal']
video = [np.concatenate([frame_gs, frame_mesh], axis=1) for frame_gs, frame_mesh in zip(video_gs, video_mesh)]

# Save the video to the output folder
imageio.mimsave(os.path.join(output_dir, f"{image_name}_multi.mp4"), video, fps=30)
