# inference.py
import torch
from SoundModel import AudioClassifier
from SoundClassificationSplit import get_data_loaders

# Load validation data only
_, val_dl = get_data_loaders(batch_size=16)

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AudioClassifier()
model.load_state_dict(torch.load("trained_model.pth", map_location=device))
model.to(device)
model.eval()


# ----------------------------
# Inference
# ----------------------------
def inference(model, val_dl):
    correct_prediction = 0
    total_prediction = 0

    with torch.no_grad():
        for data in val_dl:
            inputs, labels = data[0].to(device), data[1].to(device)

            # Normalize inputs
            inputs_m, inputs_s = inputs.mean(), inputs.std()
            inputs = (inputs - inputs_m) / inputs_s

            outputs = model(inputs)
            _, prediction = torch.max(outputs, 1)
            correct_prediction += (prediction == labels).sum().item()
            total_prediction += prediction.shape[0]

    acc = correct_prediction / total_prediction
    print(f'Inference Accuracy: {acc:.2f} over {total_prediction} samples')


# Run inference
if __name__ == "__main__":
    inference(model, val_dl)
