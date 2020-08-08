import os
from flask import Flask,session,render_template,request,jsonify,redirect,url_for,flash
from werkzeug.utils import secure_filename
from flask import send_from_directory
from simplereco.src import maine
import argparse
app = Flask(__name__)
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0
app.secret_key = "kdfjldkfjdlkfjdlkfjdlfjl"

UPLOAD_FOLDER = "static"
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
parser = argparse.ArgumentParser()
parser.add_argument('--dump', help='dump output of NN to CSV file(s)', action='store_true')
args = parser.parse_args()
decoderType = maine.DecoderType.BestPath
model = maine.Model(open(maine.FilePaths.fnCharList).read(), decoderType, mustRestore=True, dump=args.dump)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/uploads/<filename>',methods=["GET","POST"])
def uploaded_file(filename):
    flash("Upload Successful")
    full_filename = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    return render_template("success.html",full_filename='static/tesst.png',message = None)

@app.route("/infer", methods=["GET", "POST"])
def index():
    a = maine.infer(model,maine.FilePaths.fnInfer)
    full_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'tesst.png')
    return render_template("success.html",message = a,full_filename='static/tesst.png')

@app.route("/", methods=["GET", "POST"])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file found')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            file.filename ="tesst.png"
            filename = file.filename
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('uploaded_file',filename=filename))

    return render_template("index.html")
app.run(debug = True)
