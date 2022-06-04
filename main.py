# main.py
# Binary file parsing: https://github.com/googlecreativelab/quickdraw-dataset/blob/master/examples/binary_file_parser.py

from graphics import *
from harris import num_corners

import numpy as np

# window size
WIDTH = 500
HEIGHT = 500

IMAGE_SIZE = 28

COORD_SIZE = 150

NUM_SAMPLES = 1000

win = GraphWin("Generative Line Art", WIDTH, HEIGHT, autoflush=False)
win.setCoords(0, COORD_SIZE, COORD_SIZE, 0)
objects = [] # all objects currently drawn

padding = (COORD_SIZE - IMAGE_SIZE) / 2

# drawing: contains stroke information
def draw(drawing):
  i = 0
  for row in range(IMAGE_SIZE):
    for col in range(IMAGE_SIZE):
      if drawing[i] > 0:
        color = 255 - drawing[i]
        rect = Rectangle(Point(col + padding, row + padding), 
                        Point(col + 1 + padding, row + 1 + padding))
        rect.setFill(color_rgb(color, color, color))
        rect.setOutline(color_rgb(color, color, color))
        rect.draw(win)
        objects.append(rect)
      i += 1

# clear window
def clear():
  for obj in objects:
    obj.undraw()
  objects.clear()

def label_corners(corners):
  for corner in corners:
    c = Circle(Point(corner[0] + padding, corner[1] + padding), 0.5)
    c.setOutline("red")
    c.setWidth(0.5)
    c.draw(win)
    objects.append(c)

def main():
  drawings = np.load('data/duck.npy')

  level1 = []
  level2 = []
  level3 = []

  # take random sample of drawings
  # drawings = np.random.choice(drawings, NUM_SAMPLES, replace=False)
  drawings = drawings[:NUM_SAMPLES]
  for drawing in drawings:
    # drawing['corners'] = num_corners(drawing['pixels'])
    corners, locs = num_corners(np.reshape(drawing, (-1, IMAGE_SIZE)))
    if corners < 150:
      level1.append((drawing, locs))
    elif corners > 250 and corners < 300:
      level2.append((drawing, locs))
    elif corners > 400 and corners < 450:
      level3.append((drawing, locs))

  print(len(level1))
  print(len(level2))
  print(len(level3))

  while True:
    key = win.getKey()
    clear()
    if key == '1':
      drawing = level1[np.random.randint(len(level1))]
    elif key == '2':
      drawing = level2[np.random.randint(len(level2))]
    elif key == '3':
      drawing = level3[np.random.randint(len(level3))]
    elif key == 'q':
      break

    draw(drawing[0])
    label_corners(drawing[1])
  
  win.close()

if __name__ == "__main__":
  main()