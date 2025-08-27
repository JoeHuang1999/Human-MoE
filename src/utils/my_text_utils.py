import torch
from transformers import DistilBertModel, DistilBertTokenizer, CLIPTokenizer, CLIPTextModel

def getModelSize(model):
	param_size = 0
	param_sum = 0
	for param in model.parameters():
		param_size += param.nelement() * param.element_size()
		param_sum += param.nelement()
	buffer_size = 0
	buffer_sum = 0
	for buffer in model.buffers():
		buffer_size += buffer.nelement() * buffer.element_size()
		buffer_sum += buffer.nelement()
	all_size = (param_size + buffer_size) / 1024 / 1024
	print('Model size: {:.3f}MB'.format(all_size))
	print(f'Sum of parameters: {param_sum}')
	return (param_size, param_sum, buffer_size, buffer_sum, all_size)

def get_tokenizer_and_model(model_type, device, eval_mode=True):
    assert model_type in ('bert', 'clip'), "Text model can only be one of clip or bert"
    if model_type == 'bert':
        text_tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        text_model = DistilBertModel.from_pretrained('distilbert-base-uncased').to(device)
    else:
        # text_tokenizer = CLIPTokenizer.from_pretrained('openai/clip-vit-base-patch16')
        # text_model = CLIPTextModel.from_pretrained('openai/clip-vit-base-patch16').to(device)
        # text_tokenizer = CLIPTokenizer.from_pretrained('openai/clip-vit-large-patch14')
        # text_model = CLIPTextModel.from_pretrained('openai/clip-vit-large-patch14').to(device)
        text_tokenizer = CLIPTokenizer.from_pretrained('laion/CLIP-ViT-H-14-laion2B-s32B-b79K')
        text_model = CLIPTextModel.from_pretrained('laion/CLIP-ViT-H-14-laion2B-s32B-b79K').to(device)
    if eval_mode:
        text_model.eval()
    getModelSize(text_model)
    return text_tokenizer, text_model
    

def get_text_representation(text, text_tokenizer, text_model, device,
                            truncation=True,
                            padding='max_length',
                            max_length=77):
    """
    token_output: {
        input_ids: [[49406, 49407, ...], ...]
        attention_mask: [[1, 1, 0, ...], ...]
    }
    indexed_tokens: [[49406, 49407, ...], ...]
    att_masks: [[1, 1, 0, ...], ...]
    """
    token_output = text_tokenizer(text,
                                  truncation=truncation,
                                  padding=padding,
                                  return_attention_mask=True,
                                  max_length=max_length)
    indexed_tokens = token_output['input_ids']
    att_masks = token_output['attention_mask']

    # tokens_tensor: tensor([[49406, 49407, ...], ...])
    tokens_tensor = torch.tensor(indexed_tokens).to(device)

    # mask_tensor: tensor([[1, 1, 0, ...], ...])
    mask_tensor = torch.tensor(att_masks).to(device)

    # text_embed.shape: torch.Size([16, 77, 512])
    text_embed = text_model(tokens_tensor, attention_mask=mask_tensor).last_hidden_state
    return text_embed
