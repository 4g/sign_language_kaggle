features
0. limit number of points going into model
   1. lips, eyes
   2. pose
   3. hands

1. augmentations
    rotation
    flip
    per part augmentation / rotation etc. 
    head and hands have accurate 3d and can be rotated , find online method that works
    
2. Normalization - very important
    center
    range

3. manual features ? are these necessary and even helpful ? we need complex features without making the network too deep.
    distance
    angle

4. separate different parts and feed them separately to model. intuition is each part's 'pose' is captured. 
5. chose only 2d coordinates

model
1. part x part 
2. gcn
4. blazeblock

training
1. train individual parts embedding (use only parts to classify and then use part embeddings for final classifier)
2. order parts differently before giving to convolution

data
    1. how2sign mapping from openpose to mediapipe and use as pretraining
    2. can we create/get more data that looks like this ?


Results
- 12/04 
  - small transformer , with projection layer before, no augs, val in .9-1. nparams = 915,450 : 87 epochs : 0.55, 0.5
  - transformer with position + embedding + limited parts + 2d + kaggle guy normalization : 87 epochs : .6967, .6
  - do better normalization, per part maybe

- 13/04 PLAN 
    - separate normalization from augmentation. do preprocessing just before sending to network. preprocess will chose parts and do normalize etc. flag protect for render.
    - implement augmentations , for sure : hflip, rotate, reverse, reflect_pad
    - add features such as angles, distances. think of other features
    - can we further reduce points in hands, lips
    - think about right way to normalize. with simple things + good network people have .77. we should be able do better with analysis and changes
    - go through videos, write code to analyse misclassified videos
    - what are major things still remaining to be addressed . e.g. design/size of transformer.
    - can GCN help ? best resource for GCN : https://github.com/kennymckormick/pyskl
    - how to train transforfmers https://kikaben.com/transformers-training-details/

    - increasing emb dim makes it to .69 : .63
    - increasing video length makes it train very fast (or maybe its reduced keypoints)
    - large model with augs and normalization : 0.8, 0.67
    - lets make better normalization !!!!

14/04 PLAN
    - chose keypoints myself
    - go through failure videos
    - better normalization
    - more augmentation
    - do an end to end submission (separate keypoint extraction + normalization outside)

TODO
    - temporal center crop + 
    - AI4BHARAT OPEN HANDS GOLDEN PAPER (aug, norm, massive dataset): https://arxiv.org/pdf/2110.05877.pdf
    - normalize with some fixed body points like shoulders or neck to be at center of video + 
    - shear transform +
    - smaller set of keypoints + 

changing val makes accuracy .82 : .70 at 90 epochs.

Things to do with dates
1. Refactor code slightly - 4
2. write script to visualize errors and test model on live camera - 3
3. + Better Normalization - 0
4. + remove frames with both hands as NA ???
5. separate NA parts and give them separate embeddings - 1
6. better embedder for keypoints before sending them to attention layer
7. + chose keypoints - 2
8. experiment with larger network
9. external dataset

16/04 -- 
10. distance, angles etc features
11. determine misperforming classes
12. larger embedding layer
13. give separate embedding to NA

preclassify then only use good frames

1. simplify points and network and make network and make a submission
2. find a better way to chose frames

Problems 
1. labels are close to each other in meaning and looks
2. average pooling means averaging the class across video


Layers in model
1. normalization
2. keypoint extraction
3. choose non nan frames
4. angles etc maybe

augmentations before sending to model
1. flip, rotation, shift etc.
   
steps 
0. flipping is incorrect in pose , only hand and head points are flipped. correct the fucking augmentation. expected accuracy 0.75
1. rewrite network to have the required layers
2. train with new code, see transformer parameters of other people and if needed separate embeddings
3. make submission

best config :
1. 1536 ffdim
2. no z

things to do 
PLAN THE FUCKING DAY FIRST
1. capture hand pose : use tflite embedder in some way. maybe use it to train another embedder for hands only.
2. how to handle classes that are close to each other
3. implement normalization and frame choice in model
4. separate embedding for NAN