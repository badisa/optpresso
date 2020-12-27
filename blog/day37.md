# Work Log Day 37

A few days ago a friend, who works on these sorts of problems for a living, provided me with some feedback on the progress that I had made with this project. Despite hoping they would provide me with the silver bullet, I expected a much duller interpretation of what I had to do. Sadly I was right.

The most significant resource to come out of the conversation was [this](https://karpathy.github.io/2019/04/25/recipe/). The approach is much more akin to what I would expect from a statistical analysis (which is basically what ML is) than anything else. I will summarize some of the most notable take aways.

* "Don’t be a hero" - Copy and paste a simple architecture
* "Use Adam with a learning rate of 3e-4."
* "Use a constant [learning rate]"
* "Get more data" - data is more valuable than poking the model
* "Always be very careful with learning rate decay"

These take aways have turned out to be very useful in the last few days. First I had to stop myself from being a hero.

I came into this project wanting to 'understand' how to built neural networks, however as I wade deeper into it the harder it is to pin down exactly how exactly to rationalize adding or removing a layer. There doesn't seem to be a formula and I should leave that to the PhDs who spend their days trying to figure out the best architecture. So I went back to all of the networks I had and decreased the learning rate to 3e-4 and so far I have managed to run all of the Nvidia models I had added previously. All of them managed to do a better job than the default model I had been using, with a difference of 10 by MSE. Still running the rest of the models to determine if there is a notably 'better' architecture to move forward with.

I had just finished up work on implementing the Snapshot Ensemble method, which hadn't provided much utility as I could tell. Which told me that I was getting ahead of myself. I have moved back to using a constant learning rate, hoping to try and use data to get a better sense of how well the model can perform before trying to eek out better performance.

Collect more data. Its obvious, I only have about 11k images as of typing this out. I have begun to focus on collecting data on the tail ends in hoping to avoid the models endless attempt to optimize for the mean. It seems to have had some impact on the low end, though the high end is still giving me trouble. Probably doesn't help that pull times end up all over the map when you get into the 50 second range. And while it is something I have been doing for awhile now, I take quite a few more images than I initially did. I take somewhere between 5 and 10 images per pull rather than the 3 or 4 early on. In part this provides a larger sample of the grounds and also helps me gather data quicker. 