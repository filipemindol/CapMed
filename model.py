import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel, GPT2LMHeadModel, GPT2Tokenizer, AutoTokenizer, AutoModelForCausalLM
import timm
from torch.utils.data import DataLoader
from torchvision import transforms
from dataloader import RadiologyDataset
from PIL import Image
import numpy as np



class ViTModel(nn.Module):
    """This is the model that will encoder the images as embedings and only select 
        the CLS token, so the embeding of each image is a single vector (1 dimension)
        This model is pretrained and will not be trained.
    Args:
        vit_model_name (str): corresponds to the image encoding model
        x (tensor): input batch of image (batch_size, 3,24,24)
    """
    def __init__(self, vit_model_name='vit_base_patch16_224'):
        super(ViTModel, self).__init__()
        self.vit = timm.create_model(vit_model_name, pretrained=True, num_classes=0)  # Disable classification head

    def forward(self, x):
        # x: (batch_size, 3,24,24)
        x = self.vit.forward_features(x)  # Output: (batch_size, num_patches, embeding_size)
        x = x[:, 0, :]  # CLS token output: (batch_size,  embeding_size)
        return x

class MedGPTModel(nn.Module):
    """This is a LLM that receives an input prefix as vector of token ids
        and creates a caption with max_length of tokens. The output is 
        a tensor of dimension (batch_size, max_length) with EOS as padding token
    Args:
        medgpt_model_name (str): corresponds to the text generation model
        input_ids (tensor): corresponds to a prefix associated with the image features by
                            mapping network and that corresponds to token ids 
                            of size  (batch_size, seq_len).
    """
    def __init__(self, medgpt_model_name="Sharathhebbar24/chat_gpt2_dpo"):
        super(MedGPTModel, self).__init__()
        self.model = AutoModelForCausalLM.from_pretrained(medgpt_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(medgpt_model_name)

    def forward(self, input_ids):
        # input_ids: (batch_size, seq_len) - assuming inputs are token IDs
        outputs = []
        for ids in input_ids:
            output = self.model.generate(ids.unsqueeze(0), max_length=80, pad_token_id=self.tokenizer.eos_token_id)
            # use as paddind EOS token
            outputs.append(output[0])  # Collect generated token IDs
        return torch.stack(outputs)  # (batch_size, max_length)




class MappingNetwork(nn.Module):
    """This is a network that will map the feature images to input tokens for LLM
        The network will create a prefix which is vector of tokens ids of length seq_len. 
    Args:
        input_size (integer): corresponds to the embeding size of image encoder CLS token
        hidden_size (integer): dimension of hidden size of network
        vocab_size (integer): this is dimention of vocaabulary embeding/token embeding 
                                of language model
        seq_len (integer): this is the length of tokens used as prefix to generate caption
        x (tensor): input image features size (batch_size, embeding_size)
    """
    
    def __init__(self, input_size, hidden_size, vocab_size=50257, seq_len=20):
        super(MappingNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)  # Adding Batch Normalization
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.fc3 = nn.Linear(hidden_size, vocab_size)
        self.seq_len = seq_len
        
    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        logits = self.fc3(x)
        
        # Temperature scaling
        temperature = 1.0
        logits /= temperature
        
        # Convert logits to probabilities
        probabilities = F.softmax(logits, dim=-1)
        
        # Sample tokens from the probabilities
        _, top_indices = torch.topk(probabilities, self.seq_len, dim=-1)
        
        return top_indices






class Model(nn.Module):
    """Build all the model. The idea is to fix the parameters of the 
        text decoder and image encoder and only train the mapping network.
    Args:
        vit_model_name (str): corresponds to the image encoding model
        medgpt_model_name (str): corresponds to the text generation model
        input_size (integer): corresponds to the embeding size of image encoder CLS token
        hidden_size (integer): dimension of hidden size of network
        vocab_size (integer): this is dimention of vocaabulary embeding/token embeding 
                                of language model
        seq_len (integer): this is the length of tokens used as prefix to generate caption
    """
    def __init__(self, vit_model_name='vit_base_patch16_224', medgpt_model_name='Sharathhebbar24/chat_gpt2_dpo',
                 input_size=768, hidden_size=1024, vocab_size=50257, seq_len=20): 
        super(Model, self).__init__()
        self.vit_model = ViTModel(vit_model_name)
        self.mapping = MappingNetwork(input_size, hidden_size, vocab_size, seq_len)
        self.medgpt_model = MedGPTModel(medgpt_model_name)
        
        for param in self.vit_model.parameters():
            param.requires_grad = False

        for param in self.medgpt_model.parameters():
            param.requires_grad = False
            
        for param in self.mapping.parameters():
            param.requires_grad = True


    def forward(self, images):
        image_features = self.vit_model(images)  # image_features: (batch_size, embeding_size)
        text_token_ids = self.mapping(image_features)  # prefix: (batch_size, seq_len)
        medgpt_output = self.medgpt_model(text_token_ids) # caption_token_id: (batch_size, max_length)
        
        return medgpt_output


















