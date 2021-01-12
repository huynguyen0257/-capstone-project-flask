
from flask_cors import CORS, cross_origin
from flask import Flask
import sys
sys.path.append('services/')
sys.path.append('route/')
from route.face_route import face_api
from route.object_route import object_api
from route.mask_route import mask_api

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'


# Load object detection routes
app.register_blueprint(object_api)

# Load face detection routes
app.register_blueprint(face_api)

# Load mask detection routes
app.register_blueprint(mask_api)


@app.route('/')
def index():
    return "Hello, World!"


if __name__ == '__main__':
    app.run(use_reloader=False, debug=True, host='0.0.0.0')
