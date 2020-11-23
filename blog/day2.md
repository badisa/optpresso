# Work Log Day 2

The second day was oriented all around data collection. Having only 18 images, none of which were split out validation and testing, there wasn't much I could expect from my neural network. Armed with about 2 pounds of various coffees and a full water tank, I set to collecting data.

## Collection Methodology

A rough work flow is described in the ``README`` of the github repository that assumes you're using the model to determine how to adjust your coffee grind. As I had 40 cups of coffee to make from 7 different coffees the collection methodology was adjusted for speed and convenience.

1. Grind 18g of a coffee
2. Take a pinch of the coffee grounds (<0.1g) and set aside
3. Tap grounds flat twice
4. Use an OCD Distribution tool to flatten coffee and make the work flow more consistent.
5. Tamp coffee.
6. Pull shot of espresso until 36g of liquid are in the cup.
7. Record total time (from when the pump engages to when scale says 36g is in the cup)
8. Take coffee grounds and place on surface to photograph
9. Spread the coffee to be relatively flat (Personally found a small plastic box to be the most effective)
10. Recorded 3-4 images using ``optpresso capture`` 

Considerations had to be made to change the grind to try and hit a range of times. A "good" cup of coffee generally pulls in 30s, however to ensure the model is effective at predicting bad grinds, the grinder was adjusted to try and cover a larger range.

I adjusted my coffee setup to improve the work flow by placing my grinders next to my laptop rather than the espresso machine being next to the grinder.

![Coffee setup]({{site.url}}/optpresso/blog/img/day2-setup.jpg)

The testing ended up being pretty mess with the lower pull times, due to the messy stream of coffee. 

![Not what you want your coffee to look like]({{site.url}}/optpresso/blog/img/day2-cup.jpg)

## Beginning to tune the network

Once I had burned through all of the coffee I was willing to sacrifice and or drink, I returned to see how the model handled the new data. What I noticed initially was that the loss during training would bounce around between 50 and 200. Doing some quick googling I found [this](https://stats.stackexchange.com/questions/201129/training-loss-goes-down-and-up-again-what-is-happening) which indicated that my learning rate might be too great. 

Looking through the code it appeared that I have a pretty low learning rate of ``1e-4``, however I was using the wrong keyword argument (thanks copy and paste). Reducing the learning rate as well adding the callback ``ReduceLROnPlateau`` to adjust it dynamically if it starts to plateau. For training I ran 500 epochs, seemingly more than I will want in the future with the images being resized to 255x255. 

At this point I had a model that seemed to be about 3-6 seconds off for existing images, IE those that the model had trained on. I decided to test that prospectively on a [Panamanian Geisha](https://proudmarycoffee.com/collections/new-coffee/products/limited-panama-auromar-geisha-washed-100gm-tin) I had only 24g of. 

Grinding a gram of the coffee (about 5 beans) at the setting I had ground the most recent tests at and spreading a pinch of it under the microscope, I found the model predicted an average shot time of 27 seconds over 4. I probably will play around with standard deviation in the future, rather than relying on average. Aiming for 30 seconds, I adjusted the grinder a bit finer.

Upon pulling a shot with 18 grams the final time was 35 seconds. Indicating that the grinder had probably been about right the first time around. Overall was pretty satisfying to get so close after only two days in. Would like this model to be accurate to one second, as that is already a pretty tight margin for pulling espresso and grind is far from the only factor at play.

## What I Learnt and What I Need to Learn

### Learnt

* Learning rate plays an important role in training
* Hacking together things you find on the Internet for ML seems to work surprisingly well
* Need to have some test data that isn't apart of the training
* Takes about 3 hours to collect data from about 45 shots working continuously.


### To Learn

* More Machine Learning theory
  - Activation layers
  - Convolution layers
  - Drop out layers
  - Stochastic Gradient Descent
* How to reason about layers in a Neural Net