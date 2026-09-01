# EcoVision
EcoVision uses AI-powered YOLO computer vision to detect and classify waste from photos and videos. A simple Streamlit app then maps garbage locations and plans the fastest collection routes saving fuel, speeding up cleanup, and keeping streets cleaner.

# Problem Statement and Motivation

Most cities collect waste on fixed, schedule-based routes trucks visit every bin at set times, regardless of how full it actually is. This wastes fuel on empty stops while overflowing bins in busy areas go unnoticed until residents complain, leading to littering, higher costs, and unnecessary emissions.

EcoVision solves this using AI-powered computer vision (YOLO) to automatically detect and classify garbage from street images and video, flagging dirty areas by location and priority turning waste collection into a need-based data-driven process instead of a fixed schedule.

# Dataset Description & Sources

The dataset was built by combining multiple datasets sourced and annotated via Roboflow, merged into a single unified dataset (ecovision-quw28 project) and exported in YOLO format.

Size: 31,817 images containing 50,098 annotated objects 1.66 objects per image on average
Classes (5): Glass, General Waste, Paper, Plastic, Metal
Splits: divided into train, valid, and test sets 

Combining data from multiple sources helped increase the variety of scenes, lighting conditions, and object appearances the model was trained on, giving it a stronger foundation for detecting and classifying garbage in real-world outdoor street scenes.


# Model Architecture and Training Pipeline

The model is built on YOLO (yolo26n.pt) fine-tuned on the custom EcoVision waste dataset using the ultralytics library. YOLO was chosen because it detects and classifies multiple objects in a single pass, making it fast enough for both static images and live video

Training setup: image size 640×640, 100 epochs with early stopping (patience=15), batch size 16

Pipeline: download and inspect the dataset and run a baseline check with pretrained weights and fine-tune on the custom data and load the best checkpoint for evaluation and deployment

# Evaluation Results & Baseline Comparison

The model was evaluated on the held-out test split using ultralytics' built-in validator.

Precision: 0.8204 how often a flagged detection is actually garbage
Recall: 0.6505 how much of the actual garbage in an image gets detected
F1-score:  0.7256 balance between precision and recall
mAP@0.5: 0.7217 detection accuracy at a lenient overlap threshold
mAP@0.5:0.95: 0.5939 stricter, averaged detection accuracy (COCO-style)

Per-class performance: Glass performed best (mAP = 0.7691), while Plastics lagged behind (mAP = 0.4687) likely due to fewer training examples and smaller or overlapping object instances for that category.

A confusion matrix (generated in the notebook) further shows where the model confuses one waste type for another.

# Limitations & Ethical Considerations

Class imbalance: Metals and Plastics score lower (mAP = 0.47–0.48) than Glass and Paper (mAP = 0.75–0.77), likely due to fewer training examples and smaller/overlapping objects.

Dataset scope: trained mainly on outdoor street scenes — indoor or unusual lighting conditions aren't validated.

Inference speed: live video hasn't been benchmarked on lower-power hardware, which matters for real truck deployment.

Privacy: street images may capture people or license plates, with no anonymization applied.

Fairness: report priority is based only on object count per photo, not broader equity factors , could bias which areas get faster attention.

# Future Work

Deploy the model on real collection trucks edge devices, insha'allah.

Improve inference speed for real-time video processing.

Integrate live GPS-based route optimization for collection trucks.

Collect more targeted data for underrepresented classes like Metals and Plastics.

Fine-tune the model for small and overlapping objects.

Expand the waste taxonomy beyond the current 5 categories.

Add Arabic language support and further polish the dashboard UX.

# Data Link
https://app.roboflow.com/noora-hussain-s-workspace/ecovision-quw28/3 

# App Link 
https://garbage-detection-mkjddfjwfp3q436jug6u8h.streamlit.app/

# Video Link

 
