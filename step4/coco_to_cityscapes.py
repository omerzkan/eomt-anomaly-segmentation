

# Coco Dataset to Cityscapes Dataset Mapping


COCO_LABEL_DICT = {
    "0": "person",
    "1": "bicycle",
    "2": "car",
    "3": "motorcycle",
    "4": "airplane",
    "5": "bus",
    "6": "train",
    "7": "truck",
    "8": "boat",
    "9": "traffic light",
    "10": "fire hydrant",
    "11": "stop sign",
    "12": "parking meter",
    "13": "bench",
    "14": "bird",
    "15": "cat",
    "16": "dog",
    "17": "horse",
    "18": "sheep",
    "19": "cow",
    "20": "elephant",
    "21": "bear",
    "22": "zebra",
    "23": "giraffe",
    "24": "backpack",
    "25": "umbrella",
    "26": "handbag",
    "27": "tie",
    "28": "suitcase",
    "29": "frisbee",
    "30": "skis",
    "31": "snowboard",
    "32": "sports ball",
    "33": "kite",
    "34": "baseball bat",
    "35": "baseball glove",
    "36": "skateboard",
    "37": "surfboard",
    "38": "tennis racket",
    "39": "bottle",
    "40": "wine glass",
    "41": "cup",
    "42": "fork",
    "43": "knife",
    "44": "spoon",
    "45": "bowl",
    "46": "banana",
    "47": "apple",
    "48": "sandwich",
    "49": "orange",
    "50": "broccoli",
    "51": "carrot",
    "52": "hot dog",
    "53": "pizza",
    "54": "donut",
    "55": "cake",
    "56": "chair",
    "57": "couch",
    "58": "potted plant",
    "59": "bed",
    "60": "dining table",
    "61": "toilet",
    "62": "tv",
    "63": "laptop",
    "64": "mouse",
    "65": "remote",
    "66": "keyboard",
    "67": "cell phone",
    "68": "microwave",
    "69": "oven",
    "70": "toaster",
    "71": "sink",
    "72": "refrigerator",
    "73": "book",
    "74": "clock",
    "75": "vase",
    "76": "scissors",
    "77": "teddy bear",
    "78": "hair drier",
    "79": "toothbrush",
    "80": "banner",
    "81": "blanket",
    "82": "bridge",
    "83": "cardboard",
    "84": "counter",
    "85": "curtain",
    "86": "door-stuff",
    "87": "floor-wood",
    "88": "flower",
    "89": "fruit",
    "90": "gravel",
    "91": "house",
    "92": "light",
    "93": "mirror-stuff",
    "94": "net",
    "95": "pillow",
    "96": "platform",
    "97": "playingfield",
    "98": "railroad",
    "99": "river",
    "100": "road",
    "101": "roof",
    "102": "sand",
    "103": "sea",
    "104": "shelf",
    "105": "snow",
    "106": "stairs",
    "107": "tent",
    "108": "towel",
    "109": "wall-brick",
    "110": "wall-stone",
    "111": "wall-tile",
    "112": "wall-wood",
    "113": "water-other",
    "114": "window-blind",
    "115": "window-other",
    "116": "tree-merged",
    "117": "fence-merged",
    "118": "ceiling-merged",
    "119": "sky-other-merged",
    "120": "cabinet-merged",
    "121": "table-merged",
    "122": "floor-other-merged",
    "123": "pavement-merged",
    "124": "mountain-merged",
    "125": "grass-merged",
    "126": "dirt-merged",
    "127": "paper-merged",
    "128": "food-other-merged",
    "129": "building-other-merged",
    "130": "rock-merged",
    "131": "wall-other-merged",
    "132": "rug-merged"
}

CITYSCAPE_TRAIN_ID_DICT = {
    0: 'road',
    1: 'sidewalk',
    2: 'building',
    3: 'wall',
    4: 'fence',
    5: 'pole',
    6: 'traffic light',
    7: 'traffic sign',
    8: 'vegetation',
    9: 'terrain',
    10: 'sky',
    11: 'person',
    12: 'rider',
    13: 'car',
    14: 'truck',
    15: 'bus',
    16: 'train',
    17: 'motorcycle',
    18: 'bicycle',
}

"""
This is the trainid version of the cityscape labels, there is two type of ids, one is normal ids, and the other is trained ids. In the trained ids,
there are some ids with 255 and we need to filter them. Also, we need to use train ids to map. Because train ids use in the training
"""
SYNONYMS_COCO_TO_CITY = {
    "stop sign": "traffic sign",
    "pavement-merged": "sidewalk",
    "tree-merged": "vegetation",
    "grass-merged": "terrain",
    "dirt-merged": "terrain",
    "gravel": "terrain",
}

IGNORED = 255

coco_to_city_id_map = {}

for coco_key, coco_val in COCO_LABEL_DICT.items():
    
    flag = True
    coco_val_list = coco_val.split("-")    
    for city_key, city_val in CITYSCAPE_TRAIN_ID_DICT.items():
        
        if(city_val in coco_val_list):
            coco_to_city_id_map[int(coco_key)] = city_key
            flag = False
            break
        
        if(SYNONYMS_COCO_TO_CITY.get(coco_val, "")==city_val):
            coco_to_city_id_map[int(coco_key)] = city_key
            flag = False
            break
    
    if(flag):
        coco_to_city_id_map[int(coco_key)] = IGNORED

if __name__ == "__main__":
    
    n_mapped = sum(1 for v in coco_to_city_id_map.values() if v != IGNORED)
    print(f"Mapped {n_mapped}/133 COCO classes\n")
    
    for coco_id, city_id in sorted(coco_to_city_id_map.items()):
        
        if city_id != IGNORED:
            
            print(f"id:{coco_id}, label:{COCO_LABEL_DICT[str(coco_id)]} -> "
                  f"id:{city_id}, label:{CITYSCAPE_TRAIN_ID_DICT[city_id]}")
            