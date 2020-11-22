# Work Log Day 1

Its the height (or so we hope) of Covid and I have become trapped at home with an espresso machine. What better time to start my long time project to determine the pull time of a coffee using pictures of a small sample of the grinds?

I am big consumer of limited edition coffees, typically from [Proud Mary](https://proudmarycoffee.com/), which come in 100g quantities. For anyone who has tried to dial in espresso, that is a pretty small quantity and most can end up down the drain. As a software developer I am always on the look out software solutions to these sorts of questions. And thus optpresso (Optimize + espresso).

Before I start describing my image collection, I should mention the coffee equipment I am working with. Only because I love coffee and the related equipment.

The machine upon which I intend to collect all of the pull times (and delicious espresso) from is the La Marzocco Linea Mini. Had it for a number of years now and love it to death.

![Linea Mini]({{site.url}}/optpresso/blog/img/day1-machine.jpg)

For grinders I have the Conical Monolith and the Niche Zero. Both fantastic grinders. I predict (or perhaps hope) that the grinder is not a factor in the pull time compared to the machine (flow profile, pressure, etc), though only time will tell. Definitely plan to try some other grinders (blade grinders anyone?) to see if the model holds up.

![Grinders]({{site.url}}/optpresso/blog/img/day1-grinder.jpg)

To get started and to keep things cheap I got a [cheap digital "microscope"](https://www.amazon.com/Microscope-Digital-Carrying-Compatible-Portable/dp/B085XZVFGT/ref=sr_1_3) with which I could collect images of the grind samples. Hooking it up to an old laptop, I was able to start collecting images of my coffee grinds.

For each cup of coffee I made today I took 2 or 3 photos of a small sample of the grinds. Including some photos taken with a cell phone, to evaluate how much detail a image requires to be precise.

Initially I had been collecting data by using the Photo Booth app on the laptop. This required moving files about, renaming them to include the pull times and was generally fussy. Using OpenCV I was able to write a little CLI command ``optpresso capture`` that would allow me to take a photo and immediately place it where the information was stored. It's not a great interface, unclear from the OpenCV documentation on how to clean it up.

After having collected some images, and properly caffeinating myself, I took to tackling the ML aspect of the project. Initially I started [here](https://github.com/rsyamil/cnn-regression) as this doesn't seem like a classification problem which is what Convolutional Neural Networks (CNN) are most commonly used for. However the use of the MNIST dataset built into tensorflow quickly got in the way since I had images that weren't yet in numpy arrays. Using a [keras tutorial](https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html) I was able to find some utilities for loading images as datasets which made things much more straight forward, until the model for the regression fell apart on me.

The rudimentary behavior of Neural Nets are pretty straight forward, and I have a long way to go to understand exactly the distinction between different layers and how they transform the data. Having run into issues with the original model, I swapped to using the classification model in the second tutorial. This is very likely wrong and merits a lot of reading up on the different types of layers.
