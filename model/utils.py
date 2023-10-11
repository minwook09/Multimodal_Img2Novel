from PIL import Image
import requests
from transformers import (
    PreTrainedTokenizerFast,
    GPT2LMHeadModel,
)

def inference(device, url:str, max_length:int, feature_extractor, model, tokenizer, temperature, repetition_penalty) -> str:
    image = Image.open(requests.get(url, stream=True).raw)
    image.show()
    pixel_values = feature_extractor(image, return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values.to(device),num_beams=5,max_length=20)
    generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)    
    model2 = GPT2LMHeadModel.from_pretrained('skt/kogpt2-base-v2')
    tokenizer2 = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2",
      bos_token='</s>', eos_token='</s>', unk_token='<unk>',
      pad_token='<pad>', mask_token='<mask>')
    input_ids = tokenizer2.encode(generated_text[0], return_tensors='pt')
    gen_ids = model2.generate(input_ids, 
                              temperature=temperature,
                              max_length=max_length,
                              repetition_penalty=2.0,
                              pad_token_id=tokenizer.pad_token_id,
                              eos_token_id=tokenizer.eos_token_id,
                              bos_token_id=tokenizer.bos_token_id,
                              use_cache=True)
    generated = tokenizer.decode(gen_ids[0])
    return generated