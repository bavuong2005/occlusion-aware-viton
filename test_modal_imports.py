import modal
import sys

# We reuse the image definition from modal_app.py to test it
from modal_app import image

app = modal.App("test-imports")

@app.function(
    image=image,
    include_source=True,
)
def test_all():
    sys.path.insert(0, "/root")
    try:
        from pipeline import run_full_pipeline
        print("[TEST] pipeline imported successfully!")
        
        from stages.object_detection import ObjectDetectionStage
        print("[TEST] ObjectDetectionStage imported successfully!")
        
        from stages.object_segmentation import ObjectSegmentationStage
        print("[TEST] ObjectSegmentationStage imported successfully!")
        
        from stages.person_parsing import PersonParsingStage
        print("[TEST] PersonParsingStage imported successfully!")
        
        from stages.densepose import DensePoseStage
        print("[TEST] DensePoseStage imported successfully!")
        
        from stages.cloth_parsing import ClothParsingStage
        print("[TEST] ClothParsingStage imported successfully!")
        
        from stages.tryon_inference import TryOnInferenceStage
        print("[TEST] TryOnInferenceStage imported successfully!")
        
        print("ALL IMPORTS SUCCESSFUL!")
    except Exception as e:
        print(f"IMPORT FAILED: {type(e).__name__} - {e}")
        raise e

@app.local_entrypoint()
def main():
    test_all.remote()
