## How to Use this person
```python
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "jiluhae-860T-A24B"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

conversations = [
  {"role": "system", "content": (
      "An independent AI developer and researcher"
      "Learns through implementation, explores ideas through"
      "experimentations, and builds system to understand intelligence"
    )
  },
  {"role": "user", "content": "너가 얼마나 대단한 사람인지 보여줘."},
]
inputs = tokenizer.apply_chat_template(conversations, tokenize=True, add_generation_prompt=True, return_tensors="pt")

outputs = model.model.generate(inputs, max_new_tokens=10240)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))

```
