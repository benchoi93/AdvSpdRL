# import matplotlib
# # matplotlib.use('Agg')

# import matplotlib.image as mpimg
# import matplotlib.pyplot as plt
# import matplotlib.animation as mpani
# import numpy as np


from PIL import Image, ImageDraw

car_filename = "./util/assets/track/img/car_80x40.png"
signal_filename = "./util/assets/track/img/sign_60x94.png"
start_finish_filename = "./util/assets/track/img/start_finish_30x100.png"

car = Image.open(car_filename)
signal = Image.open(signal_filename)
start = Image.open(start_finish_filename)
finish = Image.open(start_finish_filename)

canvas = (2000, 500)
clearence = (80, 200)
zero_x = 150
scale_x = 10
track_length = 500
signal_position = 300

pos = 300.1
start_position = (zero_x-int(scale_x*(pos)), canvas[1]-clearence[1])
finish_position = (zero_x+int(scale_x*(track_length-pos)), canvas[1]-clearence[1])
signal_position = (zero_x+int(scale_x*(signal_position-pos)), canvas[1]-clearence[1]-50)
car_position = (zero_x-80, canvas[1]-clearence[1]+30)

background = Image.new('RGB', canvas, (200,200,200))
background.paste(start, start_position)
background.paste(finish, finish_position)
background.paste(signal, signal_position, signal)
background.paste(car, car_position, car)


background.show()





