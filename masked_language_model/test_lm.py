import transformers

if __name__ == "__main__":
    tokenizer_path = "./tokenizer.json"
    encoder_dir = "./encoder_dir"           # Should contain config.json and pytorch_model.bin

    tokenizer = transformers.RobertaTokenizer.from_pretrained(tokenizer_path)

    fill_mask = transformers.pipeline(
        task="fill-mask",
        model=encoder_dir,
        tokenizer=tokenizer
    )

    # Specify the string you want to fill
    input_string = f"t-shirty stan {fill_mask.tokenizer.mask_token}"

    prediction = fill_mask(
        input_string,
        top_k=3
    )
    print(prediction)
