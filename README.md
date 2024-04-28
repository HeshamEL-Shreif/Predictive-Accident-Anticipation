# Predictive-Accident-Anticipation

### Introduction

Innovation throughout history has been fueled by humanity's relentless pursuit of safety. Despite tremendous strides, traffic accidents persist as a formidable threat, annually claiming the lives of 1.35 million individuals globally, according to the World Health Organization (WHO). Deep learning emerges as a beacon of hope in our quest to enhance road safety. Its capacity to sift through vast and intricate datasets presents an unprecedented opportunity for anticipating and preventing accidents. However, the true efficacy of deep learning in averting accidents lies not solely in its analytical prowess but in its ability to mirror the nuanced vigilance of human attention. Human cognition excels in discerning subtle cues, recognizing hazards, and swiftly adapting to unforeseen circumstances. Thus, integrating insights from human cognitive faculties into deep learning models becomes imperative for achieving optimal results. This synergy between deep learning and human-like attention mechanisms signifies a paradigm shift in accident anticipation. It transcends mere data analysis to synthesize information with the finesse and intuition inherent in human cognition. The fusion of these technologies holds the promise of not only predicting but proactively preventing accidents before they occur, particularly in the context of autonomous driving. As we delve deeper into the realm of artificial intelligence and autonomous vehicles, the urgency to anticipate accidents intensifies. Reported collisions involving autonomous vehicles underscore he critical need for enhanced safety measures. Leveraging the intricate spatial and temporal data captured by dashboard cameras presents an opportunity to mitigate risks proactively, thus reducing reliance on human attention and bolstering road safety measures significantly. Ultimately, the goal extends beyond prediction to proactive accident prevention. By harnessing the power of computational insights and technological advancements, we aim to minimize the devastating toll of traffic accidents and establish a future where accidents are not just foreseen but actively mitigated. This transformation is not only about introducing innovative technologies but also redefining our approach to road safety, making our roads safer for all.
### Architecting Precision

the incorporation of YOLOv9 as the object detection model enriches the accident anticipation system with robust capabilities to identify multiple objects within each frame of dashcam videos. Renowned for its real-time object detection prowess and accuracy, YOLOv9 operates by leveraging a single convolutional neural network, enabling precise localization and class probability prediction directly from full images in a single inference. Its multi-scale feature extraction and adaptability to object scale variations make it an excellent fit for identifying objects, a fundamental step in the accident anticipation process.
Meanwhile, EfficientNetV2, a state-of-the-art convolutional neural network architecture, assumes the role of the feature extractor within the system. Comprising efficient building blocks and advanced architecture design, EfficientNetV2 excels in extracting deep and hierarchical features from dashcam video frames. Renowned for its simplicity, effectiveness, and scalability across different model sizes, this network progressively refines features, capturing intricate patterns and hierarchical representations. Leveraging pre-trained weights from the ImageNet dataset, EfficientNetV2 offers a robust foundation for extracting both frame-level and object-level features, crucial for subsequent stages within the accident anticipation network.
Now, turning our attention to the Dynamic Spatial-temporal Attention (DSTA) network, this complex and interconnected system comprises several key modules working in tandem to anticipate accidents. The process commences with YOLOv9 processing dashcam videos, accurately detecting and localizing multiple objects within each frame. The identified objects then pass through EfficientNetV2, a proficient extractor that meticulously discerns intricate details at both frame and object levels.
These extracted features, a fusion of frame and object specifics, undergo meticulous amalgamation and precise weighting before reaching the core component, the Gated Recurrent Unit (GRU). The GRU orchestrates a complex symphony by integrating current input features with historically weighted representations, generating a hidden representation pivotal in foreseeing potential accidents in future frames.
The Dynamic Temporal Attention (DTA) module meticulously analyzes historical representations, assigning attention weights to decode temporal sequences, thereby aiding in predicting potential accident occurrences. Simultaneously, the Dynamic Spatial Attention (DSA) module learns to allocate attention weights, ensuring the precise fusion of object-level features with utmost precision and relevance.
Moreover, the Temporal Self-Attention Aggregation (TSAA) auxiliary network plays a critical role in discerning and predicting video classes exclusively during the training phase. This intricate orchestration within the DSTA network establishes the backbone of its predictive capabilities, meticulously detailed in forthcoming sections

###  Dataset
<ul>
<li>We train the model using <a href="https://drive.google.com/drive/folders/1NUwC-bkka0-iPqhEhIgsXWtj0DA2MR-F?usp=share_link" >CCD(Car Crash Dataset)</a>
<li>To intiate dataloader you will use ccd_load_data function and give it the path of the dataset and the path of <a href="https://drive.google.com/file/d/13OgrD0-8cKG0X00MlA6JXr0G_JJmHYXg/view?usp=share_link">anotate file</a> 

</ul>

### Model Training & Testing
<ul>
<li> To start training from scratch run the train_DSTA.py File
<li>  To continue training from checkpoint run the cont_train_from_chechpoint.py file and change the path of the checkpoint file in the code
<li> To test the model run the test_model.py file and change the path of the checkpoint file in the code and the path of video to test on it or use it throw the user interface by running the index.html file
</ul>

#### YOLOv9 
<ul>
<li>YOLOv9 trained on a custom dataset manually anotated on roboflow  <br>
<a href="https://universe.roboflow.com/accident-anticipating/accident-anticipation">
    <img src="https://app.roboflow.com/images/download-dataset-badge.svg"></img>
</a>
<li> YOLOv9 Notebook included in the repository
<li> YOLOv9 weights included on the <a href="https://drive.google.com/file/d/153UJCTQEPKjp_sfvN3ygqQahHTXX60jc/view?usp=share_link">drive</a>
</ul>




