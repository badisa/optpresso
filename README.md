# OptPresso

Optpresso is a toy project to see if CNNs can be used to evaluate images
of espresso grinds to reliably predict the time it takes to pull. For simplicity
several assumptions are made regarding the espresso:

1. Shots use 18g of coffee
2. Pull time is how long it takes to end up with ~36g in the cup
3. Shots pulled using a Linea mini and an [18g VST basket](https://store.vstapps.com/collections/vst-precision-filter-baskets/products/vst-precision-filter-baskets)
4. Grinder used is the Conical Monolith (MC3) from [Kafatek](https://www.kafatek.com/), the [Vario-W](https://baratza.com/grinder/vario-w/), or the [Niche Zero](https://www.nichecoffee.co.uk/)

These assumptions are made to simplify the number of variables the CNN needs
to account.

## [Blog](blog/index.md)

## Data Collection Methodology

Collected data lives in Google Drive [here](https://drive.google.com/drive/folders/1MTZe69StPiZw1J9uAkJloxB7YduGlczp?usp=sharing). 

### Coffee Data

1. Grind a few coffee beans
2. Place grounds on piece of white paper
3. Flatten grounds
4. Capture image of grounds (refer to reference image to see how close to focus)
5. Pull espresso shot, timing it from start of the pump
6. Stop shot at 36g in the cup
7. Associate shot time with image

### Images

Images are typically collected using the `optpresso capture` command along with
a Cainda Digital Microscope (i.e. a cheap digital "microscope"). Additional data is
supplemented using images captured with a phone or DSLR, input by hand. The variety
of inputs is intended to evaluate the effectiveness of different capture techniques.

### Collection Equipment

The following are 'microscopes' that have been used to collect data. There is a reference image within the Google drive [folder](https://drive.google.com/drive/folders/1MTZe69StPiZw1J9uAkJloxB7YduGlczp?usp=sharing) to show the approximate size of the frame. The tallest line in the image indicates 1/10th of an inch, making the width covered by the sensor 0.15 of an inch.

* [Cheap Microscope](https://www.amazon.com/gp/product/B085XZVFGT/)

## But I own a \<insert espresso machine here\>!

While this project is currently tightly coupled to the Linea Mini's flow/pressure profile, if not to my specific machine, that is not the long term goal! Once the underlying Neural Network has been trained to reliably determine the pull time from grind, an additional tool will be added to build your own model to go from the Neural Network's prediction to your own personal machine. The difficult part is taking an image and getting a number, once you have a number there are simpler methods of mapping one function (pull time on a Linea Mini) to another (pull time on your machine of choice).

## Installation and Initialization

Developed/"tested" using python 3.7, suggested to use the same setup if you want to use it.

```
$ conda create -n optpresso python=3.7
$ pip install -e .
```

To install requirements for testing, development and the notebooks, install the additional requirements.

```
$ pip install -r requirements.txt
```

Once you have installed `optpresso` you will want to download the model (named optpresso.h5) [here](https://drive.google.com/drive/folders/1MTZe69StPiZw1J9uAkJloxB7YduGlczp?usp=sharing). Once it is downloaded you can initialize your optpresso and get started.

```
$ optpresso init path/to/downloaded/model.h5
```

## Usage

It is possible to use Optpresso via two different means. Either using the CLI with the following commands:

```
$ optpresso predict --camera 0  # For predictions
$ optpresso capture 0  # For capturing images
```

Or you can use the Web UI which has more advanced features and generally easier to use.

```
$ optpresso serve /path/to/capture/to [--capture-split] [--seed=0] [--browser] [--port=888] [--split-ratio=7,2,1]
```

## TODO

* Improve secondary model's fitting data (its down right unusable right now)
* Make the model better on average than within 6 seconds (on average)


