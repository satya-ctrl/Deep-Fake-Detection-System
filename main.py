import torch
import torchvision.transforms as transforms
import torchvision.models as models
import cv2
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import accuracy_score, mean_squared_error

# Load pretrained model (ResNet18)
model = models.resnet18(pretrained=True)
model.fc = torch.nn.Linear(model.fc.in_features, 2)  # 2 classes: Real or Fake
model.eval()

# Frame classification
def classify_frame(frame):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    input_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(input_tensor)
        prediction = torch.argmax(output, 1).item()
    return prediction  # 0 = Real, 1 = Fake

# Process individual video
def process_video(video_path, max_frames=30):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    results = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame_count > max_frames:
            break
        label = classify_frame(frame)
        results.append(label)
        frame_count += 1
    cap.release()
    return results

# Batch process videos from a single folder (no labels required)
def process_all_videos(folder_path):
    video_results = []
    
    for file in os.listdir(folder_path):
        if file.endswith(".mp4") or file.endswith(".avi"):
            video_path = os.path.join(folder_path, file)
            print(f"Processing: {file}")
            predictions = process_video(video_path)
            real = predictions.count(0)
            fake = predictions.count(1)
            predicted_label = 1 if fake > real else 0  # 1 = Fake, 0 = Real

            video_results.append({
                "Video": file,
                "Predicted": predicted_label,
                "Real": real,
                "Fake": fake,
                "Total": real + fake
            })

    return pd.DataFrame(video_results)

# Accuracy + Metrics (optional simulation)
def calculate_metrics(results_df):
    results_df["Label"] = 0  # Dummy labels: assuming all are Real for demo
    results_df["Accuracy (%)"] = (results_df["Predicted"] == results_df["Label"]).astype(int) * 100
    results_df["Real (%)"] = (results_df["Real"] / results_df["Total"]) * 100
    results_df["Fake (%)"] = (results_df["Fake"] / results_df["Total"]) * 100

    ground_truth = results_df["Label"].values
    predictions = results_df["Predicted"].values

    mse = mean_squared_error(ground_truth, predictions)
    acc = accuracy_score(ground_truth, predictions)

    print(f"\n🔍 Overall Accuracy: {acc * 100:.2f}%")
    print(f"📉 Mean Squared Error (MSE): {mse:.4f}")

    return results_df

# Graphs
def generate_visualizations(df):
    df_sum = df[["Real", "Fake"]].sum()
    
    # Bar Chart
    plt.figure(figsize=(6,4))
    plt.bar(df_sum.index, df_sum.values, color=["green", "red"])
    plt.title("Overall Real vs Fake Count")
    plt.savefig("bar_chart.png")
    plt.show()

    # Pie Chart
    plt.figure(figsize=(5,5))
    plt.pie(df_sum, labels=df_sum.index, colors=["green", "red"], autopct="%1.1f%%", startangle=140)
    plt.title("Real vs Fake Distribution")
    plt.savefig("pie_chart.png")
    plt.show()

    # Heatmap
    plt.figure(figsize=(8,4))
    sns.heatmap(df[["Real", "Fake"]].corr(), annot=True, cmap="coolwarm")
    plt.title("Correlation between Real and Fake detections")
    plt.savefig("heatmap.png")
    plt.show()

    # Line Plot
    plt.figure(figsize=(10,6))
    for index, row in df.iterrows():
        plt.plot(["Real", "Fake"], [row["Real"], row["Fake"]], label=row["Video"])
    plt.title("Per Video Real vs Fake Trends")
    plt.legend(fontsize=8)
    plt.savefig("line_plot.png")
    plt.show()

    # Scatter Plot
    plt.figure(figsize=(6,6))
    sns.scatterplot(data=df, x="Real", y="Fake", hue="Predicted", palette="Set2")
    plt.title("Scatter Plot of Real vs Fake Counts")
    plt.savefig("scatter_plot.png")
    plt.show()

# Main
if __name__ == "__main__":
    video_folder = r"C:\Users\Satya\Downloads\DFD_manipulated_sequences" # Change to your path
    results_df = process_all_videos(video_folder)
    results_df = calculate_metrics(results_df)
    results_df.to_csv("video_classification_summary.csv", index=False)
    generate_visualizations(results_df)
