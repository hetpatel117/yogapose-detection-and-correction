from flask import Flask, render_template,Response
import cv2
import start
app = Flask(__name__)

class stop_camera():
    camera = cv2.VideoCapture(0)

@app.route('/')
def homepage():
    return render_template('index.html')

@app.route('/yoga_list')
def list():
    return render_template('yoga_list.html')

@app.route('/downdogyoga')
def yoga1():
    return render_template('downdog.html')

@app.route('/downdog')
def downdog():
    return Response(start.downdog(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/treeyoga')
def yoga2():
    return render_template('tree.html')

@app.route('/tree')
def tree():
    return Response(start.tree(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/warrior2yoga')
def yoga3():
    return render_template('warrior2.html')

@app.route('/warrior2')
def warrior2():
    return Response(start.warrior2(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/goddessyoga')
def yoga4():
    return render_template('goddess.html')

@app.route('/goddess')
def goddess():
    return Response(start.goddess(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/plankyoga')
def yoga5():
    return render_template('plank.html')

@app.route('/plank')
def plank():
    return Response(start.plank(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stop_video')
def stop_video():
    stop_camera.camera.release()
    return render_template("yoga_list.html")

if __name__ == "__main__":
    app.run(host='0.0.0.0')