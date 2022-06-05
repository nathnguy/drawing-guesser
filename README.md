# Bird Drawing Guesser
by Nathan Nguyen

![drawings](img/cover.png)

## Summary
Bird Drawing Guesser (BDG) was inspired by the recent influx of "Wordle" type games. In BDG, the player guesses the type of bird that a drawing is supposed to represent. If they guess incorrectly, they are shown a more complex/detailed drawing.

## Data
I used Google's [Quick, Draw!](https://quickdraw.withgoogle.com/data) dataset. Originally, I began to work with their binary files. These provided 256x256 images with data describing each stroke in the drawing. Because of the difficulty in working with the stroke data and larger images, I decided to switch to using their numpy bitmaps. These are 28x28 images with no additional information. You can still view the version of BDG using the binary files [here](https://github.com/nathnguy/drawing-guesser/tree/binary).

## Defining Complexity

## Adding GANs

## Features for the Future
