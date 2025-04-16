import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
import copy

class GameObjectMemory(Dataset):
    def __init__(self, max_size=1000):
        self.max_size = max_size
        self.images = []
        self.targets = []

    def is_ui_element(self, image_region, box):
        x1, y1, x2, y2 = map(int, box)
        region = image_region[y1:y2, x1:x2]
        if region.size == 0:
            return True
        height, width = image_region.shape[:2]
        edge_threshold = 50
        if (
            x1 < edge_threshold or x2 > width - edge_threshold or
            y1 < edge_threshold or y2 > height - edge_threshold
        ):
            return True
        if np.std(region) < 20:
            return True
        edges = cv2.Canny(region, 100, 200)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 30, minLineLength=20, maxLineGap=5)
        if lines is not None and len(lines) > 4:
            return True
        return False

    def add_detection(self, image, boxes, labels, scores):
        valid_indices = []
        for i, box in enumerate(boxes):
            if scores[i] >= 0.7 and not self.is_ui_element(image, box):
                valid_indices.append(i)
        if not valid_indices:
            print("[Info] No valid detections to store.")
            return
        valid_boxes = boxes[valid_indices]
        valid_labels = labels[valid_indices]
        target = {
            'boxes': torch.tensor(valid_boxes, dtype=torch.float32),
            'labels': torch.tensor(valid_labels, dtype=torch.int64)
        }
        if len(self.images) >= self.max_size:
            self.images.pop(0)
            self.targets.pop(0)
            print("[Memory] Oldest entry removed.")
        image_tensor = torch.tensor(image).permute(2, 0, 1).float() / 255.0
        self.images.append(image_tensor)
        self.targets.append(target)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.targets[idx]
      
def update_model(model, memory_dataset, device, num_epochs=1):
    if len(memory_dataset) < 2:
        print("[Train] Hmm, not enough samples to update model.")
        return model
    print("[Train] Temporarily stopping, updating model with memory dataset")
    training_model = copy.deepcopy(model)
    training_model.train()
    params = [p for p in training_model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    data_loader = DataLoader(
        memory_dataset,
        batch_size=2,
        shuffle=True,
        collate_fn=lambda x: tuple(zip(*x))
    )
    for epoch in range(num_epochs):
        print(f"[Epoch {epoch + 1}]")
        for i, (images, targets) in enumerate(data_loader):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            try:
                loss_dict = training_model(images, targets)
                total_loss = sum(loss for loss in loss_dict.values())
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                print(f"  Batch {i + 1}: Loss = {total_loss.item():.4f}")
            except RuntimeError as e:
                print(f"[Warning] Skipping batch due to error: {e}")
    print("[Train] Model update complete!\n")
    training_model.eval()
    return training_model
  
def detect_objects(model, frame, device):
    tensor_frame = torchvision.transforms.ToTensor()(frame).unsqueeze(0).to(device)
    with torch.no_grad():
        predictions = model(tensor_frame)
    return predictions[0]
  
def draw_boxes(frame, boxes, scores, labels, threshold=0.5):
    for i, box in enumerate(boxes):
        if scores[i] >= threshold:
            x1, y1, x2, y2 = map(int, box)
            label = f"Game Object {labels[i]}: {scores[i]:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return frame

model = fasterrcnn_resnet50_fpn(weights=True)
model.eval()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
memory_dataset = GameObjectMemory()
detection_count = 0
cap = cv2.VideoCapture(0) 
frame_history = []
print("[System] Starting game object detection...")

while True:
    ret, frame = cap.read()
    if not ret:
        print("[Error] Frame wasn't captured, make sure you have an input video source")
        break
    frame = cv2.resize(frame, (960, 540))
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_history.append(frame_gray)
    if len(frame_history) > 3:
        frame_history.pop(0)
        frame_diff = cv2.absdiff(frame_history[0], frame_history[2])
        _, thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)
        movement = np.sum(thresh) / 255
        if movement > 1000:
            print(f"[Motion] Movement detected ({int(movement)} pixels)")
            predictions = detect_objects(model, frame, device)
            boxes = predictions['boxes'].cpu().numpy()
            scores = predictions['scores'].cpu().numpy()
            labels = predictions['labels'].cpu().numpy()
            memory_dataset.add_detection(frame, boxes, labels, scores)
            detection_count += 1
            frame = draw_boxes(frame, boxes, scores, labels)
            if detection_count >= 20:
                print("[System] Enough detections collected. Updating model... this may take a while")
                model = update_model(model, memory_dataset, device)
                detection_count = 0
        cv2.putText(frame, f"Memory Size: {len(memory_dataset)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('GameObjectDetection (:', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
