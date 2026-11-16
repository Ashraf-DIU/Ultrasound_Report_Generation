"""Developer test only: runs kaggle/export_web_bundle.py with a RANDOM ViT-B-16
instead of BiomedCLIP, so the export can be tested without internet.
Use with make_fake_kaggle_inputs.py. The resulting bundle is meaningless."""
import open_clip, runpy, sys
def fake(name, **kw):
    m, _, pre = open_clip.create_model_and_transforms("ViT-B-16", pretrained=None)
    return m, pre
open_clip.create_model_from_pretrained = fake
runpy.run_path(sys.argv[1], run_name="__main__")
