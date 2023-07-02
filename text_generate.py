from transformers import GPT2LMHeadModel, BertTokenizer, TextGenerationPipeline

# Load the GPT-2 model and tokenizer
model_name = '/home/sxhuang/gpt2_new_last'
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = BertTokenizer.from_pretrained("uer/gpt2-chinese-cluecorpussmall")

# Create the text generation pipeline
text_generator = TextGenerationPipeline(model=model, tokenizer=tokenizer)

prompt = input("输入关键词")
# Generate text
generated_text = text_generator(prompt, max_length=500, do_sample=True)

# Print the generated text
print(generated_text)
