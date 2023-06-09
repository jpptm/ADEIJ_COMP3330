# ADEIJ_COMP3330
## The group project for COMP3330 SEM 1 2023


For this group project, we can split it into multiple parts:

1) Data preprocessing and loading
    This involves scrubbing through the data, cleaning it, splitting it and applying any augmentations for the model. Wil also involve the part where we have to load it     so that the neural network can process it easily. Quite important as trash data will always result to a trash model.
    
2) Network architecture
    This part of the project will require someone investigating the network structure that we can use to train our model. __Lectures for weeks 5-6__ offer some pretty
    good pathway to ensure that any model is optimally trained.
    
3) Learning process
    This part will require a member / members to write code for the training, validation and testing process. They will also be responsible for history logging for           losses and accuracies as well as any visualisations needed.
    
4) Inference
    Test the model on a dataset separate from the training set. This is how we'll know how bad or good we are doing
    

*NB: The people who will be doing network architecture will have to make sure that the code for the network class can be easily scaled and transformed with NO repercussions to training. We also want to train multiple models with it. We should be able to easily change our network if we wanted to and we won't have to rewrite the whole code base.*

**Based on my conversations with each member, here's the allocation of work:**

Adebayo/Montano - 1) Data preprocessing and loading -> Involved data augmentation and (Ismail was the only one that did this) labelling 3000 images from the unlabelled images in seg_pred/seg_pred

Adebayo/Bird/Herfel - 3) Learning process

Bird/Herfel/Konijn - 2) Network architecture -> Herfel and Bird did extra research on network architecture and training automation, Konijn wrote the mode and history exporter

Konijn/Montano - 4) Inference -> Konijn fixed bugs in the inference script and implemented torch.jit

All of us - REPORT MAKING
