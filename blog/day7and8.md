# Work Log Day 7/8

As the quantity of data comes to seem like the biggest issue with the effectiveness of the model, the amount to discuss each day decreases. I have been reading `Learning From Data` by Yaser Abu-Mostafa as a way of learning some more theory though so far its just a probability refresher. 

I put together a [notebook](https://github.com/badisa/optpresso/blob/main/notebooks/coffee-data-notebook.ipynb) yesterday for looking at the data. Currently it is only to display a histogram of the data to determine where data is most severely lacking. Certainly seems like the mid twenties and the times greater than 30 are what need to be sampled most.

Recently I have been looking at how to improve the training process. Currently I rely on the number of Epochs, which doesn't have much relation to the effectiveness of the model. It looks like the `EarlyStopping` callback in Keras is the way to go, though I have also been considering [training ensembles](https://arxiv.org/pdf/1704.00109.pdf) to avoid the local minima problem.

As I return to work soon, these posts will probably die out until the Christmas vacation begins. Though I will continue to collect data and tinker with models, hoping to get a model that can predict fairly reliably within 3s of actual pull time (the time that seems reasonable with how finicky espresso can be). After which I want to create another model that allows other espresso makers to dial in their coffee. 