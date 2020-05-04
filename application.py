import os
from flask import Flask,session,render_template,request,jsonify,redirect,url_for
from werkzeug.utils import secure_filename
from flask import send_from_directory
from simplereco.src import maine
import argparse
app = Flask(__name__)

UPLOAD_FOLDER = r'uploaded_images'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
parser = argparse.ArgumentParser()
parser.add_argument('--dump', help='dump output of NN to CSV file(s)', action='store_true')
args = parser.parse_args()
decoderType = maine.DecoderType.BestPath
model = maine.Model(open(maine.FilePaths.fnCharList).read(), decoderType, mustRestore=True, dump=args.dump)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)
@app.route("/infer", methods=["GET", "POST"])
def index():
    a = maine.infer(model,maine.FilePaths.fnInfer)
    return a

@app.route("/", methods=["GET", "POST"])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            file.filename ="tesst.png"
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('uploaded_file',
                                    filename=filename))
    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    '''
app.run(debug = True)
