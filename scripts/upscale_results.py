from PIL import Image
from pathlib import Path

res_dir = Path("results/pool_camera")
for name in [
    "regen_half_wipe_strip",
    "regen_crater_strip",
    "regen_pepper_strip",
    "deq_discrete_seg",
    "deq_recon",
    "target",
]:
    p = res_dir / f"{name}.png"
    if p.is_file():
        im = Image.open(p)
        large = im.resize((im.width * 6, im.height * 6), Image.Resampling.NEAREST)
        large.save(res_dir / f"{name}_step23_large.png")
print("All items upscaled to _step23_large.png successfully!")

