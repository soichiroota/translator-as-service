import os
import json
from typing import List

import responder
from transformers import MarianMTModel, MarianTokenizer 


env = os.environ
DEBUG = env['DEBUG'] in ['1', 'True', 'true']
MODEL = env['MODEL']

api = responder.API(debug=DEBUG)
model = MarianMTModel.from_pretrained(MODEL)
tok = MarianTokenizer.from_pretrained(MODEL)


def translate(text):
    batch = tok.prepare_translation_batch(src_texts=[text])  # don't need tgt_text for inference
    gen = model.generate(**batch)  # for forward pass: model(**batch)
    words: List[str] = tok.batch_decode(gen, skip_special_tokens=True)
    return words


@api.route("/")
async def encode(req, resp):
    body = await req.text
    texts = json.loads(body)
    resp.media = [translate(text) for text in texts]


if __name__ == "__main__":
    api.run()