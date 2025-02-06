import os
import time
from flask import Flask, render_template, request
from google.cloud import vision
from werkzeug.utils import secure_filename

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "acs-course-new-449514-f773a0fb014a.json"

vision_client = vision.ImageAnnotatorClient()

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "image" not in request.files:
            return "No file part"

        file = request.files["image"]
        if file.filename == "":
            return "No selected file"

        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)

           
            start_time = time.time()

            with open(filepath, "rb") as image_file:
                content = image_file.read()

            image = vision.Image(content=content)
            response = vision_client.label_detection(image=image)
            labels = response.label_annotations

          
            response_time = round(time.time() - start_time, 3)  

            os.remove(filepath)

            return render_template("index.html", labels=labels, response_time=response_time)

    return render_template("index.html", labels=None, response_time=None)

if __name__ == "__main__":
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)
