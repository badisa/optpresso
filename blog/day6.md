# Work Log Day 6

Thanksgiving today, thankfully it is Covid so it wasn't a big affair. Which meant I could hide out making shots of coffee. I modified my approach to pull 5 shots at the same grind to ensure pretty good coverage of an individual grind setting. What I learnt was that the range of shots for the same grind is about 3 seconds. Likely due to inconsistency in my puck prep, though I intentionally didn't try to make that incredibly consistent. Because I want to see how the model works with 'real' world data rather than licked clean data. However that does reduce my expectations of the model.

At this point the model is within about 6 seconds, regardless of model structure. At this point I attribute that to a lack of data and the poor quality of the data. I intend to try another budget 'microscope' to see if I can collect better images, though perhaps CNNs are just [not very robust yet](https://stats.stackexchange.com/a/336077) for regression. And perhaps trying to classify the images as a time would be of benefit? Until I hit somewhere in the range of a 1000 images (at \~500 as of writing this) I will continue to see if superior accuracy is available.

I spent some time repartitioning the data in the Google drive folder, as I had a very sparse validation set. At some point I would like to do a [k-fold cross validation](https://machinelearningmastery.com/k-fold-cross-validation/) though seems premature with the lack of data. With the new validation data better spread across all of the bins, definitely seeing some signal, though not hugely compelling.

![Validation on model]({{site.url}}/optpresso/blog/img/eval-updated-model.png)
