import pandas as pd
from transformers import MarianMTModel, MarianTokenizer
from tqdm import tqdm

# Load translation model
model_name = "Helsinki-NLP/opus-mt-en-ar"
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

# Translation function
def translate_text(text):
    # Handle NaN
    if pd.isna(text):
        return ""

    # Tokenize and translate
    tokens = tokenizer(text, return_tensors="pt", truncation=True, padding="longest")
    translated = model.generate(**tokens)
    arabic_text = tokenizer.decode(translated[0], skip_special_tokens=True)

    return arabic_text


def translate_dataframe(df, output_name):
    tqdm.pandas()

    # Translate title and text columns
    df["title_ar"] = df["title"].progress_apply(translate_text)
    df["text_ar"] = df["text"].progress_apply(translate_text)

    df.to_csv(output_name, index=False, encoding="utf-8")
    print(f"Saved: {output_name}")


# Load your datasets
df_true = pd.read_csv("Dataset/true.csv")
df_fake = pd.read_csv("Dataset/fake.csv")

# Run translation
translate_dataframe(df_true, "true_ar.csv")
translate_dataframe(df_fake, "fake_ar.csv")
