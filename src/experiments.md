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
    - increasing video length makes it train very fast
    - large model with augs and normalization : 0.8, 0.67
    - lets make better normalization !!!!

14/04 PLAN
    - chose keypoints myself
    - go through failure videos
    - better normalization
    - more augmentation
    - do an end to end submission (separate keypoint extraction + normalization outside)

TODO
    - temporal center crop
    - AI4BHARAT OPEN HANDS GOLDEN PAPER (aug, norm, massive dataset): https://arxiv.org/pdf/2110.05877.pdf
    - normalize with some fixed body points like shoulders or neck to be at center of video
    - shear transform
    - smaller set of keypoints