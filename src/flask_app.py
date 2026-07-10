from flask import Flask, render_template, Response
from api import get_dashboard_data, get_event_log
from analytics import generate_graph

from video_stream import (
    generate_frames,
    set_video_mode
)

app = Flask(
    __name__,
    template_folder="../templates",
    static_folder="../static"
)

# ---------------------------------
# Home Page
# ---------------------------------
@app.route("/")
def home():

    return render_template("index.html")


# ---------------------------------
# Dashboard API
# ---------------------------------
@app.route("/api/dashboard")
def dashboard():

    generate_graph()

    return get_dashboard_data()


# ---------------------------------
# Event Log API
# ---------------------------------
@app.route("/api/events")
def events():

    return get_event_log()


# ---------------------------------
# Live Camera Feed
# ---------------------------------
@app.route("/video_feed")
def video_feed():

    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


# ---------------------------------
# Change Video Mode
# ---------------------------------
@app.route("/set_mode/<mode>")
def change_mode(mode):

    if mode == "camera":

        set_video_mode("camera")

    elif mode == "dataset":

        set_video_mode("dataset")

    return "OK"


# ---------------------------------
# Run Flask
# ---------------------------------
if __name__ == "__main__":

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False,
        threaded=True
)
    