# OptPresso

Optpresso is a toy project to see if CNNs can be used to evaluate images
of espresso grinds to reliably predict the time it takes to pull. For simplicity
several assumptions are made regarding the espresso:

1. Shots use 18g of coffee
2. Pull time is how long it takes to end up with ~36g in the cup
3. Shots pulled using a linea mini
4. Grinder used is the Conical Monolith from Kafatek

These assumptions are made to simplify the number of varibles the CNN needs
to account.

## Data Collection Methodology

Collected data lives in Google Drive [here](https://drive.google.com/drive/folders/1MTZe69StPiZw1J9uAkJloxB7YduGlczp?usp=sharing). 

### Coffee Data

1. Grind a few coffee beans
2. Place grounds on piece of white paper
3. Flatten grounds
4. Capture image of grounds as closely as possible
5. Pull espresso shot, timing it from start of the pump
6. Stop shot at 36g in the cup
7. Associate shot time with image

### Images

Images are typically collected using the `optpresso capture` command along with
a Cainda Digital Microscope (i.e. a cheap digital "microscope"). Additional data is
supplemented using images captured with a phone or DSLR, input by hand. The variety
of inputs is intended to evaluate the effectiveness of different capture techniques.

## But I own a <insert espresso machine here>!

While this project is currently tightly coupled to the Linea Mini's flow/pressure profile, if not to my specific machine, that is not the long term goal! Once the underlying Neural Network has been trained to reliably determine the pull time from grind, an additional tool will be added to build your own model to go from the Neural Network's prediction to your own personal machine. The difficult part is taking an image and getting a number, once you have a number there are simpler methods of mapping one function (pull time on a Linea Mini) to another (pull time on your machine of choice).

## Installation

Developed/"tested" using python 3.7

```
$ pip install -e .
```


