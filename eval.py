import argparse
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from preprocess.dataset import CustomDataset
from preprocess.dataload import create_model

def evaluate_model(model, dataloader, device):
    model.eval()
    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating", unit="batch"):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            total_predictions += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

    accuracy = correct_predictions / total_predictions
    print(f"Accuracy: {accuracy:.4f}")

def main(args):
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() and args.device == 'cuda' else "cpu")

    # Data transformations for validation dataset
    data_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Create validation dataset
    val_dataset = CustomDataset(root_dir=args.data_dir, transform=data_transform)

    # Create validation data loader
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # Load the model
    model = create_model(args.model_name, args.num_workers)  # Adjust this based on your model architecture
    model.load_state_dict(torch.load(args.model_path))
    model.to(device)

    # Evaluate the model
    evaluate_model(model, val_dataloader, device)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluation of RepVGG model")
    parser.add_argument("--data-dir", type=str, default="../../../data/test_set", help="Data directory path")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--num-workers", type=int, default=7, help="Number of workers for data loader")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Device to use (cuda or cpu)")
    parser.add_argument('-mp',"--model-path", type=str, required=True, help="Path to the trained model")
    parser.add_argument('-mn', '--model-name', type=str, required=True, choices=['Repvgg', 'VIT'],help='Chocie model [Repvgg, VIT]')
    args = parser.parse_args()
    main(args)
