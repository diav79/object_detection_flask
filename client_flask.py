import requests
import json
import cv2

addr = 'http://localhost:5000'
# test_url = addr + '/det_obj/all_detection'
test_url = addr + '/det_obj/largest_detection'

# prepare headers for http request
content_type = 'image/jpeg'
headers = {'content-type': content_type}

img = cv2.imread('ng.jpg')
# encode image as jpeg
_, img_encoded = cv2.imencode('.jpg', img)
# send http request with image and receive response
response = requests.post(test_url, data=img_encoded.tostring(), headers=headers)
# decode response
print (json.loads(response.text))