# main.py
# Binary file parsing: https://github.com/googlecreativelab/quickdraw-dataset/blob/master/examples/binary_file_parser.py

from parser import unpack_drawings
from graphics import *
from harris import num_corners

import numpy as np

# window size
WIDTH = 500
HEIGHT = 500

IMAGE_SIZE = 256

NUM_SAMPLES = 100

win = GraphWin("Generative Line Art", WIDTH, HEIGHT, autoflush=False)
objects = [] # all objects currently drawn
padding_x = (WIDTH - IMAGE_SIZE) / 2
padding_y = (HEIGHT - IMAGE_SIZE) / 2

# drawing: contains stroke information
def draw(drawing):
  for stroke in drawing['image']:
    points = []
    num_points = len(stroke[0])
    for i in range(num_points):
      pt = Point(stroke[0][i] + padding_x, stroke[1][i] + padding_y)
      points.append(pt)
    interpolate_points(points)

# connect given points with straight lines
def interpolate_points(points):
  for i in range(1, len(points)):
    line = Line(points[i], points[i - 1])
    line.setFill(color_rgb(0, 0, 0))
    line.setWidth(1)
    line.draw(win)
    objects.append(line)

# clear window
def clear():
  for obj in objects:
    obj.undraw()
  objects.clear()

def main():
  drawings = list(unpack_drawings('data/duck.bin'))

  level1 = []
  level2 = []
  level3 = []

  # take random sample of drawings
  drawings = np.random.choice(drawings, NUM_SAMPLES, replace=False)
  for drawing in drawings:
    # drawing['corners'] = num_corners(drawing['pixels'])
    corners = num_corners(drawing['pixels'])
    if corners < 50:
      level1.append(drawing)
    elif corners > 100 and corners < 125:
      level2.append(drawing)
    elif corners > 175 and corners < 200:
      level3.append(drawing)

  print(len(level1))
  print(len(level2))
  print(len(level3))

  while True:
    key = win.getKey()
    clear()
    if key == '1':
      draw(np.random.choice(level1))
    elif key == '2':
      draw(np.random.choice(level2))
    elif key == '3':
      draw(np.random.choice(level3))
    elif key == 'q':
      break
  
  win.close()

if __name__ == "__main__":
  main()