import os
import imageio
import time
from PIL import Image
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils import render_utils, postprocessing_utils

# Start timing
start_time = time.time()

# Load a pipeline from a model folder or a Hugging Face model hub.
pipeline = TrellisImageTo3DPipeline.from_pretrained("/home/liu/code/TRELLIS/path/to/TRELLIS-image-large")
pipeline.cuda()

# Load an image
#image_path = "assets/example_image/building.png"
#image_path = "assets/furniture_photo/chair.jpg"
#image_path = "assets/test/cup.png"
#image_path = "assets/charater/yoimiya.png"
#image_path = "assets/people/kobe.png"
#image_path = "assets/furniture/old_bed_1.png"
#image_path = "assets/furniture_draw/multi_2.png"
#image_path = "assets/furniture_AI/sofa.png"
#image_path = "assets/furniture_photo/table_chair.png"
#image_path = "assets/car/car2.jpg"
#image_path = "assets/chair/chair1.png"
image_path = "assets/chair/chair2.png"
image = Image.open(image_path)

# Extract the input folder name
input_folder_name = os.path.basename(os.path.dirname(image_path))

# Create output directory based on the input folder name and the image name
image_name = os.path.splitext(os.path.basename(image_path))[0]
output_dir = os.path.join("Output", input_folder_name, image_name)
os.makedirs(output_dir, exist_ok=True)

# Output the output directory path
print(f"Output directory path: {output_dir}")

# Run the pipeline
outputs = pipeline.run(
    image,
    seed=1,
)

# Render the outputs and save videos
video = render_utils.render_video(outputs['gaussian'][0])['color']
imageio.mimsave(os.path.join(output_dir, f"{image_name}_gs.mp4"), video, fps=30)
video = render_utils.render_video(outputs['radiance_field'][0])['color']
imageio.mimsave(os.path.join(output_dir, f"{image_name}_rf.mp4"), video, fps=30)
video = render_utils.render_video(outputs['mesh'][0])['normal']
imageio.mimsave(os.path.join(output_dir, f"{image_name}_mesh.mp4"), video, fps=30)

# GLB files can be extracted from the outputs
glb = postprocessing_utils.to_glb(
    outputs['gaussian'][0],
    outputs['mesh'][0],
    simplify=0.95,
    texture_size=1024,
)
glb.export(os.path.join(output_dir, f"{image_name}.glb"))

# Save Gaussians as PLY files
outputs['gaussian'][0].save_ply(os.path.join(output_dir, f"{image_name}.ply"))

# End timing and print the total time taken
end_time = time.time()
total_time = end_time - start_time
print(f"Total time taken: {total_time:.2f} seconds")
