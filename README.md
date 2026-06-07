# Lane Detection With Semantic Segmentation

This repository explores lane detection using semantic segmentation on BDD100K driving-scene images.

The project generates lane labels from raw road images, trains a segmentation model, and visualizes predicted lane regions, overlays, and intermediate feature maps.

## Dataset

- **Dataset**: BDD100K
- **Task**: Lane-region segmentation from road-scene images
- **Label generation**: Canny edge detector-based preprocessing for training labels

## Workflow

1. Generate lane labels for training images.
2. Train a semantic-segmentation model on the labeled image set.
3. Evaluate the trained model on road-scene examples.
4. Visualize raw inputs, generated labels, model outputs, overlays, and feature maps.

## Raw Images

<img src="Car1.png" height="300" width="200"> <img src="Car2.png" height="300" width="200"> <img src="Car3.png" height="300" width="200"> <img src="Car4.png" height="300" width="200">

## Generated Labels

<img src="Label1.png" height="300" width="200"> <img src="Label2.png" height="300" width="200"> <img src="Label3.png" height="300" width="200"> <img src="Label4.png" height="300" width="200">

## Labels Overlaid On Images

<img src="LabelonImage1.png" height="300" width="200"> <img src="LabelonImage2.png" height="300" width="200"> <img src="LabelOnImage3.png" height="300" width="200"> <img src="LabelOnImage4.png" height="300" width="200">

## Model Outputs

<img src="Output1.png" height="300" width="200"> <img src="Output2.png" height="300" width="200"> <img src="Output3.png" height="300" width="200"> <img src="Output4.png" height="300" width="200">

## Final Output Overlays

<img src="FinalOutput1.png" height="300" width="200"> <img src="FinalOutput2.png" height="300" width="200"> <img src="FinalOutput3.png" height="300" width="200"> <img src="FinalOutput4.png" height="300" width="200">

## Feature Map Visualization

<img src="FeatureVisualisation1.png" height="300" width="200"> <img src="FeatureVisualisation2.png" height="300" width="200"> <img src="FeatureVisualisation3.png" height="300" width="200"> <img src="FeatureVisualisation4.png" height="300" width="200"> <img src="FeatureVisualisation5.png" height="300" width="200"> <img src="FeatureVisualisation6.png" height="300" width="200">
