import os
from iosum import dbhandler
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from global_vars import MODEL_PATH, MODEL_NAME, TEST_DB_PATH, TEST_DB_NAME, TRAIN_DB_PATH, TRAIN_DB_NAME, \
    EVAL_TXT_PATH, FIGURE_EVAL_PATH


def prediction_figure(pred, true, classes, ramanshift, data):
    fig, axs = plt.subplots(1,2, figsize=(10,5))
    plt.bar(classes, pred * 100, label='True prediction: ' + classes[int(true)])
    plt.ylabel('Prediction (%)')
    plt.ylim(0,100)
    plt.xlabel('Class')
    plt.legend()

    axs[0].plot(ramanshift, data)
    axs[0].set_xlabel("Raman-shift (1/cm)")
    axs[0].set_ylabel("Intensity (counts)")

    plt.tight_layout()

    return 0

def evaluate_model(model, classes, x_test, y_test, ramanshift, file_path, file_name, figure_path):
    confidences = model.predict(x_test)
    predictions = np.empty(shape=(len(confidences)))
    for i in range(len(confidences)):
        predictions[i] = np.argmax(confidences[i])
    corr_pred = 0
    prob_pred = 0
    prob_pred_index = []
    conf_pred = 0
    conf_pred_index = []
    for i in range(len(predictions)):
        if predictions[i] == y_test[i]:
            corr_pred += 1
        if np.max(confidences[i]) > 0.9:
            conf_pred += 1
            conf_pred_index.append(i)
        if np.max(confidences[i]) <= 0.6:
            prob_pred += 1
            prob_pred_index.append(i)

    classes_predictions = []
    incorrect_indices = []
    for i in range(len(classes)):
        corr_pred_nr = 0
        incorr_pred_nr = 0
        for j, pred in enumerate(predictions):
            if y_test[j] == i:
                if pred == i:
                    corr_pred_nr += 1
                else:
                    incorr_pred_nr += 1
                    incorrect_indices.append(j)
        classes_predictions.append((corr_pred_nr, incorr_pred_nr))

    res_str = f"Classes: "
    for i, class_name in enumerate(classes):
        res_str += f"{i} - {class_name}\t"
    res_str += "\n"

    res_str += f"Correct Predictions: {corr_pred:.0f}/{len(y_test)}\n"

    for i, class_name in enumerate(classes):
        res_str += (f"Class {i} - Predicted:"
                    f" {classes_predictions[i][0]:.0f}/{classes_predictions[i][0]+classes_predictions[i][1]:.0f}\n")

    res_str += f"Confident predictions (confidence over 90%): {conf_pred:.0f}\n"
    res_str += f"Predictions with confidence under 50%: {prob_pred:.0f}\n"

    res_str += "Class\tPrediction\tConfidence\n"
    for i in range(len(y_test)):
        res_str += f"{y_test[i]:^5.0f}\t{predictions[i]:^10.0f}\t{np.max(confidences[i]):^10.3f}"
        res_str += "\n"

    print(res_str)
    #Saving
    #Create directories if they don't exist
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    if not os.path.exists(figure_path):
        os.makedirs(figure_path)
    if not os.path.exists(figure_path + "/confidence over 90"):
        os.makedirs(figure_path + "/confidence over 90")
    if not os.path.exists(figure_path + "/confidence under 50"):
        os.makedirs(figure_path + "/confidence under 50")
    if not os.path.exists(figure_path + "/incorrect predictions"):
        os.makedirs(figure_path + "/incorrect predictions")

    #Saving the notes
    with open(file_path+"/"+file_name+".txt", 'w', encoding="utf-8") as f:
        f.write(res_str)

    #Saving the figures
    #Confident predictions
    for i in conf_pred_index:
        prediction_figure(confidences[i], y_test[i], classes, ramanshift, x_test[i])
        plt.savefig(figure_path+'/confidence over 90/'+"fig_"+str(i+1))
        plt.close()

    #Inconfident predictions
    for i in prob_pred_index:
        prediction_figure(confidences[i], y_test[i], classes, ramanshift, x_test[i])
        plt.savefig(figure_path+'/confidence under 50/'+"fig_"+str(i+1))
        plt.close()

    #Wrong predictions
    for i in incorrect_indices:
        prediction_figure(confidences[i], y_test[i], classes, ramanshift, x_test[i])
        plt.savefig(figure_path+'/incorrect predictions/'+"fig_"+str(i+1))
        plt.close()

    return 0

def write_result(file_path):
    with open(file_path, 'w',encoding="utf-8") as f:
        pass
    return 0


#load model and test data
model = tf.keras.models.load_model(MODEL_PATH+"/"+MODEL_NAME+".keras")
model_summary = model.summary()

raman_shifts = dbhandler.convert_array(dbhandler.select_all(TRAIN_DB_PATH,TRAIN_DB_NAME,"spectra")[0][1])

res = dbhandler.select_all(TEST_DB_PATH,TEST_DB_NAME,'spectra')
index_test = np.zeros(len(res))
x_test = np.zeros(shape=(len(res),len(raman_shifts)))
y_test = np.zeros(len(res))

for i in range(len(res)):
    index_test[i] = res[i][0]
    x_test[i,:] = dbhandler.convert_array(res[i][1])
    y_test[i] = res[i][2]


#Make predictions
predictions = model.predict(x_test)
loss, acc = model.evaluate(x_test, y_test, verbose=0)

evaluate_model(model, ['under 24 ppm', 'in between', 'above 24 ppm'], x_test, y_test, raman_shifts,
               EVAL_TXT_PATH, MODEL_NAME, FIGURE_EVAL_PATH)

plt.show()


#teszt komment