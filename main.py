# main.py

from parser import unpack_drawings
from graphics import *
from harris import num_corners

import random
import numpy as np

# window size
WIDTH = 500
HEIGHT = 500

IMAGE_SIZE = 256

NUM_SAMPLES = 100

win = GraphWin("Bird Drawing Guesser", WIDTH, HEIGHT, autoflush=False)
objects = [] # all objects currently drawn

padding_x = (WIDTH - IMAGE_SIZE) / 2
padding_y = (HEIGHT - IMAGE_SIZE) / 2

categories = ['duck', 'flamingo', 'owl', 'parrot', 'swan']

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

# gets data from all categories using corner detection method
def get_corner_data():
  result = {}
  for category in categories:
    print(category + ':')
    drawings = list(unpack_drawings('data/' + category + '.bin'))
    drawings = np.random.choice(drawings, size=NUM_SAMPLES, replace=False)
    # drawings = drawings[:NUM_SAMPLES]

    # from low to high complexity
    levels = [[], [], []]

    # take random sample of drawings
    for drawing in drawings:
      corners = num_corners(drawing['pixels'])
      if corners < 75:
        levels[0].append(drawing)
      elif corners > 100 and corners < 150:
        levels[1].append(drawing)
      elif corners > 200 and corners < 250:
        levels[2].append(drawing)

    print('Level 1: ' + str(len(levels[0])))
    print('Level 2: ' + str(len(levels[1])))
    print('Level 3: ' + str(len(levels[2])))
    result[category] = levels

  return result

def main():
  print('Loading data...')
  data = get_corner_data()

  # game loop
  print('Hi! Welcome to Bird Drawing Guesser. Press enter to continue.')
  input()

  while True:

    print('Types of birds: ' + str(categories))

    random.shuffle(categories)

    for category in categories:
      print('What kind of bird is this?')
      for level in range(3):
        print()
        print('Level ' + str(level + 1))
        clear()

        if data[category][level]:
          arr = data[category][level]
        else:
          arr = data[category][1]

        draw(np.random.choice(arr))

        val = input('Guess: ').lower().strip()

        if val == category:
          print('Correct!')
          break
        else:
          print('Incorrect.')
      print('This is a ' + category + '!')
      print('Press enter to continue.')
      input()

    val = input('Enter "q" to quit. Otherwise, continue playing! ').lower().strip()
    if val == 'q':
      break
  
  print('Thanks for playing!')
  
  win.close()

if __name__ == "__main__":
  main()