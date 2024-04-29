from flask import Flask, jsonify
import pandas as pd
import pickle

app = Flask(__name__)

# Load your model
with open('model_saced.pkl', 'rb') as file:
    model_loaded = pickle.load(file)

def sum_str(lst):
    new_lst = []
    for i in lst:
        new_lst.append(float(i))
    return new_lst

def test_instance(data):
    processed_data = []
    intermediate_data = [[],[],[],[],[],[]]
    for i in range(len(data)):
        temp = data.iloc[i,0]
        temp = temp.split(',')
        for j in range(6):
            intermediate_data[j].append(temp[j])

    norm_inter_data = []
    for i in intermediate_data:
        temp = []
        for j in i:
            i = sum_str(i)
            temp.append((float(j)-min(i))/(max(i)-min(i))) # normalization
        norm_inter_data.append(temp)

    data_instance = []
    for i in norm_inter_data:
        data_instance += i

    to_test = pd.DataFrame(data_instance).T
    y_pred = model_loaded.predict(to_test)
    if y_pred == 1:
        return "The sample is sign language!"
    return "The sample is not sign language"

@app.route('/predict', methods=['GET'])
def predict():
    try:
        # Hardcoded Excel file path
        excel_file_path = 'Copy of Raw_test_file_sign.xlsx'

        # Read Excel file
        data = pd.read_excel(excel_file_path)
        data.columns = ['data']
        data = data.iloc[:139, :]

        # Perform prediction
        result = test_instance(data)
        return jsonify({'result': result})
    except Exception as e:
        return jsonify({'error': str(e)}), 400
if __name__ == '__main__':
    app.run()
