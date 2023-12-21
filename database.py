import mysql.connector
import base64
from face_recognition import face_encodings
import cv2
import numpy as np
from helper import logger, db_config


def get_passenger_data():
    data_from_db = []

    connection = mysql.connector.connect(**db_config)
    try:
        cursor = connection.cursor()
        cursor.execute("SELECT passengerName, personImage, passengerSeat FROM passengerdetails")
        results = cursor.fetchall()

        for passenger in results:
            try:
                load_image = base64.b64decode(passenger[1])
                image_np = np.frombuffer(load_image, dtype=np.uint8)
                image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
                face_encoding = face_encodings(image)[0]

                passenger_data = {
                    "passenger_name": passenger[0],
                    "passenger_assign_seat": passenger[2],
                    "passenger_image": load_image,
                    "passenger_encoding": face_encoding,
                }

                data_from_db.append(passenger_data)

            except Exception as e:
                logger.error("Error processing passenger %s: %s", passenger[0], e)
    except Exception as e:
        logger.error("Error fetching data from the database: %s", e)
    else:
        connection.close()

    return data_from_db
