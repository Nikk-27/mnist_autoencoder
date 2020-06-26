# mnist_autoencoder
This is a autoencoder formed on the mnist (handwritten digits dataset). To add some amount of noise I have added a gaussian layer in between the CNN layers. For initial training CNN layers are used. For tuning the model I have freezed the initial layers and then trained just the last two layers (flatten and softmax). Then again trained the model combining the initial layers and the last two layers. 

This model takes time while training if you are training on cpu so my advice would be to add checkpoints after every 5 epochs. I did it that way, after every 5 epochs weights were saved and simultaneously loss/accuracy graph was drawn to check the performance of model. This model slightly overfits but that won't affect the performance of model.
 
In the training.py file I have added code to save the weights only in the last and final training phase (combining both the initiall CNN layers and last two layers), but you can also add that code in first phase(CNN layers training) and second phase (flatten and softmaxe layers training). 
 
If still you don't understand how to do this mail me I'll help you out. I personally didn't train my whole model in one go I trained it in 4 days as I was training it on cpu. 
 
Callbacks are a way to see the model performance if its overfitting or underfitting and also to save weights after certain duration so that if somehow your training stops midway, you have your last weights to start the training again from the last checkpoint and not from the start again.

I have added a .ipynb file name View_Autoencoder_results.ipynb to only view your results, as I was training in my local system and matplotlib doesn't show results in terminal, so this file will help your to see your results in jupyter notebook.
