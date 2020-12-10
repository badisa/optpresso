# Work Log Day 20

After having using the model for two days or so, almost in anger, I have found it to be less reliable than I would like it to be. Particularly for coffees that I have trained the model on. However there have been some successes where the actual pull is within a standard deviation of the mean of \~5 predictions. Which I believe is the best way to get meaningful results.

At some point it would be worth investigating the outliers in the data, as there is definitely inferior images within the training set. Should probably use the Laplacian to detect blurry images as that is most commonly the issue I have found in my casual pruning of the data. Also providing some way to use the `optpresso eval` command to display the outliers.

The actual progress that was made was the addition of a `--k-folds` flag to the training command to perform [k folds cross validation](https://magoosh.com/data-science/k-fold-cross-validation/) automagically. Which using a k of 10 produced an average model performance of 115 MSE. Certainly a lot higher than I would like, however it is reasonable with all of the issues I have identified in this projects. Before I go into those, here is a table of the 10 MSE scores from each run to give an idea of the range.

| Fold | MSE    |
| :----|:-------|
| 0    | 120.49 |
| 1    | 131.18 |
| 2    | 109.50 |
| 3    |  94.13 |
| 4    | 119.27 |
| 5    | 128.94 |
| 6    | 128.74 |
| 7    | 116.06 |
| 8    |  99.44 |
| 9    | 104.53 |

## Issues with Optpresso

As I have been working on this project I have identified some things that I think will raise some eyebrows with folks and I wanted to cover some of them.

### Data Collection

Concerns that might arise regarding the labeling of the data (ie how is the shot pulled)

* The coffee is weighted before the grinder and not in the portafilter!
* The OCD is bad, inconsistent, etc
* Not controlling for tamping pressure

I wanted this model to help me in my work flow and hopefully others. While I might get really particular about pulling some exotic coffee just right, I play it a bit faster and looser with my every day cup of coffee. And if this is to be useful to anyone else, it will need to handle their own quirks and methodologies. I don't gather data on shots that sputter or channel (if they are >10 seconds, sub 10 second shots are a mess at the best of time) and avoid any other gross inconsistencies with the general process. This might well harm the overall performance, but this has to be fun to keep devoting my time to it.

Concerns that might arise from the terrible "microscope".

* The microscope and images are junk!
* Spreading the coffee out doesn't give you "X" which is what you really want to measure!

I had the idea for this project for a long time, largely based on some [work](https://www.rxrx.ai/) I had heard about through work. In my mind I had thought about buying some sophisticated microscope and generating a bunch of images. However that didn't seem practical for a project I hope will be useful, or at least entertaining, to others. Plus if I had gotten zero results and killed the projects after a month I would have an expensive microscope to have to get rid of.

The images do vary quite a bit in their quality and I hope to prune them to be more representative of a good image. I also need to do a study of what time of images provide the best representation of the pull time.
