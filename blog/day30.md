# Work Log Day 30

*EDIT* I had flipped the images correctly and created duplicates which was why the model suddenly performed so much better. Check your data!

I have been struggling with the fact that despite having increased the number of images to approximately 2k that the validation loss had gone from 80 up to 130. I worried that the problem was not well suited to CNNs and that I was not going to end up with anything meaningful. Now I had started this project in part to see if that was the case, but with promising results early on I came enthusiastic for this to become a reality. Finally I have managed to move the validation loss back down!

I had been messing about with the [Random Flip](https://keras.io/api/layers/preprocessing_layers/image_preprocessing/random_flip/) layer in Keras as flips are the simplest transformation to the images that still provides reliable data. I found that it didn't seem to improve the model at all. Thinking about I realized that randomly flipping images, including the validation set, was probably not wise as it could easily confuse the optimizer and shift the validation error. Subsequently I generated the three combinations of flips and mirrors for all existing images and used those to train the model. Suddenly the MSE went from around 130 down to 80. While that is not nearly the performance I would like to have, it is a promising improvement.

Now that I have found flipping to be effective, I modified the capture command to flip the images automatically. I may also look at modifying the prediction code to predict based on the flips and provide the mean back as the value. Though I would hope that I can make the model robust against the orientation of the image, though that requires more investigation.

Thankfully I recently came into possession of a newer GPU that seems much more well suited to ML, which has sped up training a bit though with the flipped images the training is still about 3x slower than it was previously. If only the newer GPU didn't result in me fighting cuda all day. I have followed the instructions that [tensorflow](https://www.tensorflow.org/install/gpu) provides and every time cuda has ended up broken in a novel way. Its truly wonderful.

Hope to have some graphs of the new results soon, the longer wall clock time is going to slow things down considerably sadly.