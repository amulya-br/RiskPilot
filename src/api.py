from flask import jsonify
from live_data import dashboard_data
import csv
import os

LOG_FILE = "logs/risk_log.csv"


def get_dashboard_data():
    return jsonify(dashboard_data)


def get_event_log():

    events = []

    if os.path.exists(LOG_FILE):

        with open(LOG_FILE, "r", newline="") as file:

            reader = list(csv.reader(file))

            if len(reader) > 1:

                rows = reader[1:][-10:]

                for row in reversed(rows):

                    events.append({
                        "time": row[0],
                        "people": row[1],
                        "risk": row[6]
                    })

    return jsonify(events)