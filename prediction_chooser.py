import numpy as np

letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

def get_highest_predictions(classifier, imgs):
    predictions = []

    for filename, i in imgs:
        result = classifier.predict(i)
        print(result)
        for idx, r in enumerate(result):
            max_ind = np.argmax(r)
            predictions.append((filename, letters[max_ind], r[max_ind]))

    return predictions

def predict_frame(classifier, frame):
        result = classifier.predict(frame)
        max_ind = np.argmax(result)
        return (letters[max_ind], result[0][max_ind])
