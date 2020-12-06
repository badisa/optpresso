# Work Log Day 16

Since having created my [notebook](https://github.com/badisa/optpresso/blob/main/notebooks/coffee-data-notebook.ipynb) for determining where I needed to collect more data I have found my model performance going down. Early on with only about 100 images I found myself to be within about 5-6 of a correct guess, and now with a bit over 1200 images the range is closer to 8-9 seconds.

My intuition tells me that this is because of undrinkable range of pull times. I rarely pull shots that are less than 10 seconds to get 36g. Largely because it is pretty obvious by visual inspection and has a 'grainy' feel. And same for shots that are up around 50s, which is effectively powder.

In attempting to get more data in these ranges I found that the model I had didn't detect the differences, treating both ends of the spectrum as effectively 30s pulls. I decided to increase the sizes of my Convolution filters to detect larger features, hoping to pick up more distinctions in particle size. It seemed to improve things and with some inspirations from [here](https://github.com/udacity/self-driving-car/blob/master/steering-models/community-models/komanda/solution-komanda.ipynb) I removed two of the pooling layers that produced the best model to date. 

![Latest model performance]({{site.url}}/optpresso/blog/img/eval-no-pooling.png)

What was most promising about this model was the density of the values close to the actual time. While there are certainly some pretty extreme outliers, the model seems like it could perform well for sampling multiple images to get a more precise prediction. An approach that offers to produce more reliable results than a single image. Note that this is an evaluation against all of the data, which I know to be an imprecise method to measure performance, so it could well be over fitted to the data. It was trained on a 0.7/0.3 train/validation split.

It is important to note that I have moved away from a fixed number of epochs. In part as the amount of data has increased and so has the time for each epoch, and more importantly because more epochs don't translate into better performance. Instead I save the best model based on the validation loss, and only run up to 25 epochs without improvement.

Hoping to get some [exotic coffee](https://proudmarycoffee.com/collections/deluxe/products/limited-panama-lamastus-family-estates-luito-geisha-asd-natural-100gm-tin) from Proud Mary tomorrow that I can test the model with the type of coffee that I built this model for.