# main.py
# Binary file parsing: https://github.com/googlecreativelab/quickdraw-dataset/blob/master/examples/binary_file_parser.py

from graphics import *
from harris import num_corners

import numpy as np
import random

from generator import Generator
import matplotlib.pyplot as plt
import torch
from gan import batch_size


# window size
WIDTH = 500
HEIGHT = 500

IMAGE_SIZE = 28

COORD_SIZE = 125

NUM_SAMPLES = 2000

LABEL_CORNERS = True

win = GraphWin("Bird Drawing Guesser", WIDTH, HEIGHT, autoflush=False)
win.setCoords(0, COORD_SIZE, COORD_SIZE, 0)
objects = [] # all objects currently drawn

padding = (COORD_SIZE - IMAGE_SIZE) / 2

categories = ['duck', 'flamingo', 'owl', 'parrot', 'swan']

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

# gets data from all categories using corner detection method
def get_corner_data():
  result = {}
  for category in categories:
    print(category + ':')
    drawings = np.load('data/' + category + '.npy')
    # drawings = np.random.choice(drawings, size=NUM_SAMPLES, replace=False)
    drawings = drawings[:NUM_SAMPLES]

    # from low to high complexity
    levels = [[], [], []]

    # take random sample of drawings
    for drawing in drawings:
      corners, locs = num_corners(np.reshape(drawing, (-1, IMAGE_SIZE)))
      if corners < 150:
        levels[0].append((drawing, locs))
      elif corners > 250 and corners < 300:
        levels[1].append((drawing, locs))
      elif corners > 400 and corners < 450:
        levels[2].append((drawing, locs))

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
        
        drawing = arr[np.random.randint(len(arr))]
        draw(drawing[0])

        if LABEL_CORNERS:
          label_corners(drawing[1])

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

def test():
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
    elif corners > 300 and corners < 350:
      level2.append((drawing, locs))
    elif corners > 500 and corners < 550:
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
    # label_corners(drawing[1])
  
  win.close()

def test():
  device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
  generator = Generator()
  generator.load_state_dict(torch.load('model/generator.pth'))
  latent_space_samples = torch.randn(batch_size, 100).to(device=device)
  generated_samples = generator(latent_space_samples)

  generated_samples = generated_samples.cpu().detach()
  for i in range(16):
      ax = plt.subplot(4, 4, i + 1)
      plt.imshow(generated_samples[i].reshape(28, 28), cmap="gray_r")
      plt.xticks([])
      plt.yticks([])

  plt.show()

if __name__ == "__main__":
  # main()
  test()