import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, DynamicCache
from peft import get_peft_model

class L3LoraL3(nn.Module):
    def __init__(
        self,
        model_name,
        max_length,
        use_peft,
        lora_config,
        num_mem,
        device
    ):
        """
        Create the compression model: LLM-LoRA + LLM.

        Args:
            model_name (str): Backbone model.
            max_length (int): Max number of tokens to be compressed.
            use_peft(bool): Whether to enable Parameter-Efficient Fine-Tuning (PEFT).
            lora_config (LoraConfig): LoRA configurations.
            num_mem (int): Number of compressed tokens.
            device (torch.device): CPU or GPU.
        """
        super(L3LoraL3, self).__init__()
        # load the original base backbone model
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
        # add LoRA parameters to the LLM
        self.use_peft = use_peft

        if self.use_peft:
            # add LoRA parameters to the LLM
            self.model = get_peft_model(model, lora_config)
            # only LoRA parameters are trainable
            for name, param in self.model.named_parameters():
                param.requires_grad = False
                if 'lora' in name:
                    param.requires_grad = True
        else:
            self.model = model

        # Save the config
        self.config = self.model.config

        print(f"Total parameters of model: {sum(p.numel() for p in self.model.parameters())}")

        # load the model tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        print("model tokenizer loaded.")
        # set the padding token
        self.tokenizer.pad_token = self.tokenizer.eos_token
        # max number of tokens to be compressed
        self.max_length = max_length
        self.criterion = nn.CrossEntropyLoss()
        # number of compressed tokens
        self.num_mem = num_mem
        # compressed token
        self.memory_embeddings = nn.Parameter(torch.randn(1, num_mem, self.model.config.hidden_size, dtype=torch.bfloat16).to(device))
        self.memory_embeddings.requires_grad = True
        self.device = device

    def forward(self, input_ids, labels, attention_mask=None):
        ####################
        # Encoder - model+lora
        ####################
        # input text tokens to be compressed
        text_tokens = input_ids
        # target tokens: input text tokens + EOS token
        target_tokens = labels
        text_tok_embeddings = self.model.get_input_embeddings()(text_tokens).to(self.device)
        # compressed tokens
        memory_tok_embeddings = self.memory_embeddings.repeat(text_tok_embeddings.shape[0], 1, 1).to(self.device)
        # encoder input: text tokens + compressed tokens
        encoder_input_embeddings = torch.cat((text_tok_embeddings, memory_tok_embeddings), dim=1)
        encoder_output = self.model(inputs_embeds=encoder_input_embeddings)
        # get the K V values for the encoder output
        past_key_values = encoder_output.past_key_values
        # get the K V values for the compressed tokens
        trimmed_past_key_values = tuple(
            (layer_key[:, :, -self.num_mem:, :], layer_value[:, :, -self.num_mem:, :]) 
            for layer_key, layer_value in past_key_values
        )
        trimmed_past_key_values = DynamicCache(trimmed_past_key_values)
        ####################
        # Decoder - model
        ####################
        # BOS token
        prompt_tokens = [self.tokenizer.bos_token_id]
        prompt_tokens = torch.tensor(prompt_tokens, device=self.device)
        prompt_tok_embeddings = self.model.get_input_embeddings()(prompt_tokens)
        prompt_tok_embeddings = prompt_tok_embeddings.repeat(text_tok_embeddings.shape[0], 1, 1)

        # decoder input: BOS token + text tokens
        decoder_input_embeddings = torch.cat((prompt_tok_embeddings, text_tok_embeddings), dim=1)
        # use the original LLM without LoRA parameters
        if self.use_peft:
            with self.model.disable_adapter():
                decoder_output = self.model(inputs_embeds=decoder_input_embeddings, past_key_values=trimmed_past_key_values)
        else:
            decoder_output = self.model(inputs_embeds=decoder_input_embeddings, past_key_values=trimmed_past_key_values)
        
        # logits for the decoder output
        all_logits = decoder_output.logits

        # target output: text tokens + EOS token
        # calculate the cross entropy
        loss = self.criterion(all_logits.view(-1, all_logits.size(-1)), target_tokens.view(-1))

        return {'loss': loss, 'logits': all_logits}


