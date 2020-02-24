# face_key_points
I used CNN to build up a model to extract face key points(eyes, nose, mouse)

Training data file has columns showed below. 

left_eye_center_x	left_eye_center_y	right_eye_center_x	right_eye_center_y	left_eye_inner_corner_x	left_eye_inner_corner_y	left_eye_outer_corner_x	left_eye_outer_corner_y	right_eye_inner_corner_x	right_eye_inner_corner_y	right_eye_outer_corner_x	right_eye_outer_corner_y	left_eyebrow_inner_end_x	left_eyebrow_inner_end_y	left_eyebrow_outer_end_x	left_eyebrow_outer_end_y	right_eyebrow_inner_end_x	right_eyebrow_inner_end_y	right_eyebrow_outer_end_x	right_eyebrow_outer_end_y	nose_tip_x	nose_tip_y	mouth_left_corner_x	mouth_left_corner_y	mouth_right_corner_x	mouth_right_corner_y	mouth_center_top_lip_x	mouth_center_top_lip_y	mouth_center_bottom_lip_x	mouth_center_bottom_lip_y	Image

There are 15 keypoints we want to extract on a face. Each keypoint has a coordinate (x, y).
So we have 15*2 = 30 numbers to predict in our model.
The Image data can be reshape to an array with shape of (96, 96). There are 2140 images in the training data file.

For test data file, it contains Image data but no keypoints labels. There are 1873 images in the test data file.

Training data and test data can be downloaded from https://drive.google.com/open?id=1qKj9qQvEqENVTU4LsobX_ilAK76JpwpY

By running model_builder.py, we build and save the model. By running face_key_points.py, we visualize our model performance.
test_result01.png and test_result02.png show some of the testing results. We can see that this model captures the mouth, nose, eyes keypoints quite well regardless of whether there is eye glasses, or the mouse is open, or there is mustache, or the face is turning to other directin to some extend.

However, when some of the face key points were blocked (by sun glasses, hair, hat) or do not exist in the image, the model prediction becomes bad. When the face angle becomes large (~ 45 degree), the model does not well, neither. When the mouth opens big with teeths showing obviously, the model performance is not good. Please see worst16_in_validation.png which shows the worst 16 face key points extractions from validation data set.

To build and test the model, I used
Python 3.7
Keras 2.3.1
matplotlib 3.1.3
numpy 1.18.1
pandas 1.0.1
tensorflow 1.15.0
sklearn 0.0

With face key points extracted, we can do lots of insteresting things such as beautifying selfie, automatic face recognition, expression analysis, face reconstruction. 
