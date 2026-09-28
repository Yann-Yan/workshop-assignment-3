import jetson.inference
import jetson.utils

# Load SSD-MobileNet-v2
net = jetson.inference.detectNet(
    "ssd-mobilenet-v2",
    threshold=0.5
)


def detect_image(input_path, output_path):
    print("\n==============================")
    print("Input image:", input_path)

    # Load image
    img = jetson.utils.loadImage(input_path)

    # Run object detection
    detections = net.Detect(img)

    print("Detected objects:", len(detections))

    # Print detection information
    for i, detection in enumerate(detections):
        print("\nDetection", i + 1)

        print("ClassID:", detection.ClassID)
        print("Class:", net.GetClassDesc(detection.ClassID))
        print("Confidence:", detection.Confidence)

        print("Left:", detection.Left)
        print("Top:", detection.Top)
        print("Right:", detection.Right)
        print("Bottom:", detection.Bottom)

        print("Width:", detection.Width)
        print("Height:", detection.Height)
        print("Area:", detection.Area)
        print("Center:", detection.Center)

    # Save image with detection boxes
    jetson.utils.saveImage(output_path, img)

    print("Saved image:", output_path)


# Image 1: bottle / cup
detect_image(
    "/home/nvidia/Desktop/bottle.jpeg",
    "/home/nvidia/Desktop/bottle_assignment.jpeg"
)

# Image 2: chair
detect_image(
    "/home/nvidia/Desktop/keyboard.jpeg",
    "/home/nvidia/Desktop/keyboard_assignment.jpeg"
)
