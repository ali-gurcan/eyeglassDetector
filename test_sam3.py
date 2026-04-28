import traceback
import torch

try:
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    
    print("Loading SAM 3 model...")
    model = build_sam3_image_model()
    processor = Sam3Processor(model)
    print("SAM 3 loaded successfully!")
except Exception as e:
    traceback.print_exc()
