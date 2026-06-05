import modal
import inspect

app = modal.App("inspect-dino")
image = modal.Image.debian_slim().pip_install("transformers>=4.38,<4.45", "torch")

@app.function(image=image)
def inspect_dino():
    from transformers import GroundingDinoProcessor
    print(inspect.signature(GroundingDinoProcessor.post_process_grounded_object_detection))

@app.local_entrypoint()
def main():
    inspect_dino.remote()
