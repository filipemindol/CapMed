import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel
from dataloader import RadiologyDataset
from model import Model
import pickle
import json
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
import pandas as pd
from tqdm import tqdm  # Correct import statement for tqdm


def load_dataset(pickle_path):
    """Function to load the dataset created either train, test or validation
    Args:
        pickle_path (str): Path to .pkl dataset
    """
    with open(pickle_path, 'rb') as f:
        dataset = pickle.load(f)
    return dataset

def test(test_data_path, tokenizer):
    """Test the model in the test dataset
    Args:
        test_data_path (str): path to test.pkl
        tokenizer: medgpt tokenizer
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_dataset(test_data_path)  # Load dataset
    images = data['images']
    true_captions = data['captions']
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    dataset = RadiologyDataset(images=images, captions=true_captions, transform=transform)
    model = Model()
    #model_path = "/nas-ctm01/homes/mtamorim/CapMed/model_epoch.pth"  # Update with your model path
    model_path = "/Users/magdaamorim/Desktop/Master Thesis/model_epoch_1.pth"  # Update with your model path

    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()
    predictions = []
    with torch.no_grad(), tqdm(total=len(dataset)) as pbar:
        for image, true_caption in dataset:
            image = image.unsqueeze(0).to(device)  # Add batch dimension
            output = model(image)  # Forward pass
            output_ids = output.squeeze(0).tolist()  # Remove batch dimension and convert to list
            # Example: Decode output to caption using tokenizer
            predicted_caption = tokenizer.decode(output_ids, skip_special_tokens=True)
            print("true caption")
            print(true_caption)
            print("predicted caption")
            print(predicted_caption)
            predictions.append(predicted_caption)
            pbar.update(1)  # Update progress bar

    bleu_scores = []
    meteor_scores = []
    ground_truths = [{'image_id': i, 'caption': [caption]} for i, caption in enumerate(true_captions)]
    results = [{'image_id': i, 'caption': prediction} for i, prediction in enumerate(predictions)]

    for true_caption, pred_caption in zip(true_captions, predictions):
        # Compute BLEU score
        bleu_score = sentence_bleu([true_caption.split()], pred_caption.split())
        bleu_scores.append(bleu_score)
        
        # Compute METEOR score
        meteor = meteor_score([' '.join(true_caption.split())], ' '.join(pred_caption.split()))
        meteor_scores.append(meteor)

    avg_bleu_score = sum(bleu_scores) / len(bleu_scores)
    avg_meteor_score = sum(meteor_scores) / len(meteor_scores)

    print(f"Average BLEU Score: {avg_bleu_score}")
    print(f"Average METEOR Score: {avg_meteor_score}")

    df = pd.DataFrame({
        'True Caption': true_captions,
        'Predicted Caption': predictions
    })
    #save_path = '/nas-ctm01/homes/mtamorim/CapMed/predictions.csv'
    save_path='predictions.csv'
    df.to_csv(save_path, index=False)

    print(f"Predictions and Captions saved to {save_path}")

if __name__ == "__main__":
    # Load GPT tokenizer and model for embeddings
    tokenizer = AutoTokenizer.from_pretrained('Sharathhebbar24/chat_gpt2_dpo')
    gpt_model = AutoModel.from_pretrained('gpt2')
    gpt_model.eval()  # Set GPT to evaluation mode
    tokenizer.pad_token = tokenizer.eos_token  # Add padding to tokenizer
    #test("/nas-ctm01/datasets/public/ROCO/files/test.pkl", tokenizer)
    test("/Users/magdaamorim/Desktop/Master Thesis/Datasets/ROCO/files/test.pkl", tokenizer)