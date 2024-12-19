from cog import BasePredictor # type: ignore
import numpy as np # type: ignore
import requests
import torch # type: ignore
from transformers import  AutoProcessor, AutoModel # type: ignore
from PIL import Image

class Predictor(BasePredictor):
    device = "cuda"
    model_id = "unum-cloud/uform-gen2-qwen-500m"

    def setup(self) -> None:
        self.model = AutoModel.from_pretrained(self.model_id, trust_remote_code=True).to(self.device)
        self.processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)

    def predict(
        self,
        image_url: str,
        prompt: str
    ) -> str:
        image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")

        inputs = self.processor(text=[prompt], images=[image], return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                do_sample=False,
                use_cache=True,
                max_new_tokens=256,
                eos_token_id=151645,
                pad_token_id=self.processor.tokenizer.pad_token_id
            )

        prompt_len = inputs["input_ids"].shape[1]
        decoded_text = self.processor.batch_decode(output[:, prompt_len:])[0]

        return decoded_text