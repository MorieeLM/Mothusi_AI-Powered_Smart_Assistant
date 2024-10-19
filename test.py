from ultralytics import YOLO

model = YOLO('yolo11n.pt')

source = 'https://ultralytics.com/images/bus.jpg'

results = model(source)

for result in results:
    boxes = result.boxes
    masks =result.masks
    keypoints = result.keypoints
    probs =result.probs
    obb = result.obb
    result.show()
    results.save(filename="/result.jpg")