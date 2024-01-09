import os
from collections import Counter, defaultdict
from io import BytesIO
from pprint import pprint
from timeit import default_timer as timer

import numpy as np
from PIL import Image, ImageTk

from database import get_passenger_data
from helper import Logger, Seat, face_img

logger = Logger(module="Test Module")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
start = timer()


# Example usage with the provided data
last_five_frames = {
    1: {
        "A1": [{"passenger_name": "Uma Shanker", "passenger_assign_seat": "A1", "passenger_match_distance": 0.5113961492903852}],
        "A2": [{"passenger_name": "Adarsh", "passenger_assign_seat": "A2", "passenger_match_distance": 0.5113961492903852}],
        "B1": [{"passenger_name": "Ravi Shanker Singh", "passenger_assign_seat": "B1", "passenger_match_distance": 0.5113961492903852}],
        "B2": [{"passenger_name": "Hari", "passenger_assign_seat": "B2", "passenger_match_distance": 0.5113961492903852}],
    },
    2: {
        "A1": [{"passenger_name": "Uma Shanker", "passenger_assign_seat": "B2", "passenger_match_distance": 0.5113961492903852}],
        "A2": [{"passenger_name": "Adarsh", "passenger_assign_seat": "A2", "passenger_match_distance": 0.5113961492903852}],
        "B1": [{"passenger_name": "Ravi Shanker Singh", "passenger_assign_seat": "B1", "passenger_match_distance": 0.5113961492903852}],
        "B2": [{"passenger_name": "Hari", "passenger_assign_seat": "B2", "passenger_match_distance": 0.5113961492903852}],
    },
    3: {
        "A1": [{"passenger_name": "Uma Shanker", "passenger_assign_seat": "B2", "passenger_match_distance": 0.5113961492903852}],
        "A2": [{"passenger_name": "Adarsh", "passenger_assign_seat": "A2", "passenger_match_distance": 0.5113961492903852}],
        "B1": [{"passenger_name": "Ravi Shanker Singh", "passenger_assign_seat": "B1", "passenger_match_distance": 0.5113961492903852}],
        "B2": [{"passenger_name": "Hari", "passenger_assign_seat": "B2", "passenger_match_distance": 0.5113961492903852}],
    },
    4: {
        "A1": [{"passenger_name": "Uma Shanker", "passenger_assign_seat": "B2", "passenger_match_distance": 0.5113961492903852}],
        "A2": [{"passenger_name": "Adarsh", "passenger_assign_seat": "A2", "passenger_match_distance": 0.5113961492903852}],
        "B1": [{"passenger_name": "Ravi Shanker Singh", "passenger_assign_seat": "B1", "passenger_match_distance": 0.5113961492903852}],
        "B2": [{"passenger_name": "Hari", "passenger_assign_seat": "A1", "passenger_match_distance": 0.5113961492903852}],
    },
    5: {
        "A1": [{"passenger_name": "Uma Shanker", "passenger_assign_seat": "B2", "passenger_match_distance": 0.5113961492903852}],
        "A2": [{"passenger_name": "Adarsh", "passenger_assign_seat": "A2", "passenger_match_distance": 0.5113961492903852}],
        "B1": [{"passenger_name": "Ravi Shanker Singh", "passenger_assign_seat": "B1", "passenger_match_distance": 0.5113961492903852}],
        "B2": [{"passenger_name": "Hari", "passenger_assign_seat": "A1", "passenger_match_distance": 0.5113961492903852}],
    },
}


last_five_frames2 = {
    1: {
        "A1": [],
        "A2": [{"passenger_name": "Adarsh", "passenger_assign_seat": "A2", "passenger_match_distance": 0.5113961492903852}],
        "B1": [{"passenger_name": "Ravi Shanker Singh", "passenger_assign_seat": "B1", "passenger_match_distance": 0.5113961492903852}],
        "B2": [{"passenger_name": "Hari", "passenger_assign_seat": "B2", "passenger_match_distance": 0.5113961492903852}],
    },
    2: {
        "A1": [],
        "A2": [],
        "B1": [{"passenger_name": "Ravi Shanker Singh", "passenger_assign_seat": "B1", "passenger_match_distance": 0.5113961492903852}],
        "B2": [{"passenger_name": "Hari", "passenger_assign_seat": "B2", "passenger_match_distance": 0.5113961492903852}],
    },
    3: {
        "A1": [],
        "A2": [{"passenger_name": "Adarsh", "passenger_assign_seat": "A2", "passenger_match_distance": 0.5113961492903852}],
        "B1": [{"passenger_name": "Ravi Shanker Singh", "passenger_assign_seat": "B1", "passenger_match_distance": 0.5113961492903852}],
        "B2": [{"passenger_name": "Hari", "passenger_assign_seat": "B2", "passenger_match_distance": 0.5113961492903852}],
    },
    4: {
        "A1": [],
        "A2": [{"passenger_name": "Adarsh", "passenger_assign_seat": "A2", "passenger_match_distance": 0.5113961492903852}],
        "B1": [{"passenger_name": "Ravi Shanker Singh", "passenger_assign_seat": "B1", "passenger_match_distance": 0.5113961492903852}],
        "B2": [{"passenger_name": "Hari", "passenger_assign_seat": "A1", "passenger_match_distance": 0.5113961492903852}],
    },
    5: {
        "A1": [],
        "A2": [{"passenger_name": "Adarsh", "passenger_assign_seat": "A2", "passenger_match_distance": 0.5113961492903852}],
        "B1": [{"passenger_name": "Ravi Shanker Singh", "passenger_assign_seat": "B1", "passenger_match_distance": 0.5113961492903852}],
        "B2": [{"passenger_name": "Hari", "passenger_assign_seat": "A1", "passenger_match_distance": 0.5113961492903852}],
    },
}

testCase_1 = {
    "A1": [("", "Empty", "white"), ("", "Empty", "white"), ("", "Empty", "white")],
    "A2": [("Adarsh", "Incorrect", "orange"), ("Adarsh", "Incorrect", "orange"), ("Adarsh", "Incorrect", "orange")],
    "B1": [("", "Empty", "white"), ("", "Empty", "white"), ("", "Empty", "white")],
    "B2": [
        ("Ravi Shanker Singh", "Incorrect", "orange"),
        ("Ravi Shanker Singh", "Incorrect", "orange"),
        ("Ravi Shanker Singh", "Incorrect", "orange"),
    ],
}
testCase_2 = {'A1': [('', 'Empty', 'white'),
                    ('', 'Empty', 'white'),
                    ('', 'Empty', 'white')],
             'A2': [('Unknown', 'Unauthorized', 'red'),
                    ('Unknown', 'Unauthorized', 'red'),
                    ('', 'Empty', 'white')],
             'B1': [('', 'Empty', 'white'),
                    ('', 'Empty', 'white'),
                    ('', 'Empty', 'white')],
             'B2': [('Ravi Shanker Singh', 'Incorrect', 'orange'),
                    ('Ravi Shanker Singh', 'Incorrect', 'orange'),
                    ('Ravi Shanker Singh', 'Incorrect', 'orange')]}

testCase_3 ={'A1': [('', 'Empty', 'white'),
                    ('', 'Empty', 'white'),
                    ('', 'Empty', 'white')],
             'A2': [('', 'Empty', 'white'),
                    ('', 'Empty', 'white'),
                    ('', 'Empty', 'white')],
             'B1': [('', 'Empty', 'white'),
                    ('', 'Empty', 'white'),
                    ('', 'Empty', 'white')],
             'B2': [('Ravi Shanker Singh', 'Incorrect', 'orange'),
                    ('Ravi Shanker Singh', 'Incorrect', 'orange'),
                    ('Ravi Shanker Singh', 'Incorrect', 'orange')]}

testCase_4 = {'A1': [('', 'Empty', 'white'),
                    ('', 'Empty', 'white'),
                    ('', 'Empty', 'white')],
             'A2': [('Adarsh', 'Incorrect', 'orange'),
                    ('Adarsh', 'Incorrect', 'orange'),
                    ('Adarsh', 'Incorrect', 'orange')],
             'B1': [('', 'Empty', 'white'),
                    ('', 'Empty', 'white'),
                    ('', 'Empty', 'white')],
             'B2': [('', 'Empty', 'white'),
                    ('', 'Empty', 'white'),
                    ('', 'Empty', 'white')]}

testCase_5 =  {'A1': [('', 'Empty', 'white'),
                    ('', 'Empty', 'white'),
                    ('', 'Empty', 'white')],
             'A2': [('', 'Empty', 'white'),
                    ('', 'Empty', 'white'),
                    ('', 'Empty', 'white')],
             'B1': [('', 'Empty', 'white'),
                    ('', 'Empty', 'white'),
                    ('', 'Empty', 'white')],
             'B2': [('', 'Empty', 'white'),
                    ('', 'Empty', 'white'),
                    ('', 'Empty', 'white')]}

def analyze_frames(data):
        """
        Analyze frame results.

        Args:
            data (dict): Dictionary with seat names as keys and values as lists of frame results.

        Returns:
            dict: Dictionary with seat names as keys and analyzed results as values.

        """
    
        # Dictionary comprehension to find the passenger with maximum occurrences in each seat
        result = {seat: max(Counter(values).items(), key=lambda x: x[1])[0] for seat, values in data.items()}
        return result



# Test cases
testCases = [testCase_1, testCase_2, testCase_3, testCase_4, testCase_5]
for i, data in enumerate(testCases, start=1):
    result = analyze_frames(data)
    print(f"Test Case {i}:")
    print(result)
    print("*" * 50)

end = timer()
logger.info("Elapsed time: " + str(end - start))
