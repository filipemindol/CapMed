import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel
from dataloader import RadiologyDataset
from model import Model 
import pickle
import json
from tqdm import tqdm
import wandb
import torch.nn.functional as F


def cosine_similarity_loss(pred_embeddings, target_embeddings):
    """Calculate loss as 1 - cosine similarity.
    Args:
        pred_embeddings (tensor): (batch_size, text_embedding_size)
        target_embeddings (tensor): (batch_size, text_embedding_size)
    """
    # Normalize the embeddings to unit vectors
    pred_embeddings = F.normalize(pred_embeddings, p=2, dim=1)
    target_embeddings = F.normalize(target_embeddings, p=2, dim=1)
    
    # Compute cosine similarity
    cosine_sim = F.cosine_similarity(pred_embeddings, target_embeddings)
    
    # Convert similarity to loss (1 - similarity)
    loss = 1 - cosine_sim.mean()
    
    return loss



def tokenize_captions(captions, tokenizer, max_length=80):
    """Tokenizes the ground truth captions with padding and EOS token until max_length.
    Args:
        captions (str): Ground truth captions.
        tokenizer (_type_): Tokenizer that corresponds to the same language decoder model.
        max_length (int): Length of tokens (after padding).
    """
    encodings = tokenizer(
        captions,
        padding='max_length',
        truncation=True,
        max_length=max_length,
        return_tensors='pt',
        return_attention_mask=True  # Return attention masks
    )
    
    # Return both input_ids and attention_mask
    return encodings['input_ids'], encodings['attention_mask']

    
    # Return only the input_ids and attention_mask
    return encodings['input_ids'], encodings['attention_mask']


def get_embeddings(input_ids, model):
    """This function is used to calculate the loss. 
    It does the text embedding of the generated and truth captions and only 
    retrieves the embeding of CLS token
    Args:
        input_ids (tensor): (batch_size, max_length) input token ids to embedd
        model (model) : model to perform text embeding and extract semantic meaning
    """
    
    outputs = model(input_ids).last_hidden_state  # Use last_hidden_state directly
    embeddings = outputs[:,0,:] # Only get embeding of CLS token (batch_size, text_embeding_size)
    
    return embeddings

def load_dataset(pickle_path):
    """function to load the dataset created either train, test or validation
    Args:
        pickle_path (str): path to .pkl dataset
    """
    with open(pickle_path, 'rb') as f:
        dataset = pickle.load(f)
    return dataset


def train(dataset_path, tokenizer, num_epochs=5):
    """Function to train the model in model.py.
    Args:
        dataset_path (str): Path to train dataset.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = load_dataset(dataset_path)  # Load dataset
    images = data['images']
    captions = data['captions']
    dataset = RadiologyDataset(images=images, captions=captions, transform=transform)  # Apply preprocessing of images
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0)  # Create a data loader
    model = Model().to(device)
    gpt_model = AutoModel.from_pretrained('gpt2').to(device)
    optimizer = optim.Adam(model.mapping.parameters(), lr=1e-4)  # Only train mapping network parameters
    loss_history = []
    criterion = nn.MSELoss()

    
    wandb.login(key='4391d9f95380fafdeba7f39c09289a6b5824a488')
    wandb.init(project='Capmed', entity='magda_amorim')

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        epoch_loss = []
        
        progress_bar = tqdm(dataloader, desc=f'Epoch {epoch + 1}/{num_epochs}', leave=False)
        
        for images, captions in progress_bar:
            images = images.to(device)
            target_caption_ids, target_attention_mask = tokenize_captions(captions, tokenizer)
            target_caption_ids = target_caption_ids.to(device)
            target_attention_mask = target_attention_mask.to(device)
            
            optimizer.zero_grad()
            
            try:
                pred_captions_ids = model(images)
                #pred_embeddings = get_embeddings(pred_captions_ids, gpt_model)
                #target_embeddings = get_embeddings(target_caption_ids, gpt_model)
                
                #loss = cosine_similarity_loss(pred_embeddings, target_embeddings)
                loss=criterion(pred_captions_ids,target_caption_ids)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                progress_bar.set_postfix({'loss': loss.item()})
                wandb.log({'loss': loss.item()})
                
            except RuntimeError as e:
                print(f"An error occurred: {e}")
                continue  # Skip this batch and proceed with the next one
            
        average_loss = total_loss / len(dataloader)
        tqdm.write(f'Epoch {epoch + 1}, Average Loss: {average_loss}')
        
        loss_history.append({'epoch': epoch + 1, 'loss': epoch_loss, 'average_loss': average_loss})
        
        tqdm.write(f'Epoch {epoch + 1}, Average Loss: {average_loss}')
        
        # Log average loss to W&B
        wandb.log({'average_loss': average_loss})
        with open('loss_history.json', 'w') as f:
            json.dump(loss_history, f, indent=4)
    
    torch.save(model.state_dict(), 'model_epoch.pth')
    wandb.save('model_epoch.pth')
    wandb.finish()
   
   
   
if __name__ == "__main__":
    # Initialize W&B with your project name and API key

    # Load GPT tokenizer and model for embeddings
    tokenizer = AutoTokenizer.from_pretrained('Sharathhebbar24/chat_gpt2_dpo')
    gpt_model = AutoModel.from_pretrained('gpt2')
    gpt_model.eval()  # Set GPT to evaluation mode
    tokenizer.pad_token = tokenizer.eos_token # add padding to tolenizer
    #train("/nas-ctm01/datasets/public/ROCO/files/train.pkl", tokenizer)
    train("/Users/magdaamorim/Desktop/Master Thesis/Datasets/ROCO/files/train.pkl", tokenizer)
