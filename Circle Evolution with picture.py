import numpy as np
import requests
from circle_evolution import evolution
from circle_evolution import helpers

# 網路抓取圖片資料
photo_url = "https://media.nownews.com/nn_media/thumbnail/2022/03/1647485747126-b81ba35597f8494e89d6b3be76ac9fcb-800x600.jpg?unShow=false&waterMark=false"
img_data = requests.get(photo_url).content

# 寫入
with open("cat.jpg", "wb") as img:
    img.write(img_data)

target = helpers.load_target_image("cat.jpg", size = (100, 100))
output = evolution.Evolution(target, genes = 256)
output.evolve(max_generation = 50000)