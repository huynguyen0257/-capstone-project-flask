import cv2
from PIL import Image, ImageDraw
import io


def process_image(image):
    covert_height = 600
    height, width, channels = image.shape

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    #resize image to 1/4

    small_frame = cv2.resize(
        gray, (covert_height, int(covert_height * (600 / height))))

    # show image
    # cv2.imshow('sample image', small_frame)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    ret, jpeg = cv2.imencode('.jpg', image)

    return small_frame


def convertfromnptoimage(nparray):
    img_2 = Image.fromarray(nparray.astype('uint8'))
    file_object = io.BytesIO()
    img_2.save(file_object, 'PNG')
    file_object.seek(0)
    return file_object
