from flask import Flask, jsonify, request
import pandas as pd
import pickle

with open('trained_model.plk', 'rb') as file:
    model = pickle.load(file)


app = Flask(__name__)
train_columns = ['homeworld_Alderaan', 'homeworld_Aleen Minor', 'homeworld_Bestine IV',
        'homeworld_Cerea', 'homeworld_Champala', 'homeworld_Chandrila',
        'homeworld_Concord Dawn', 'homeworld_Corellia', 'homeworld_Dagobah',
        'homeworld_Dathomir', 'homeworld_Dorin', 'homeworld_Eriadu',
        'homeworld_Glee Anselm', 'homeworld_Haruun Kal', 'homeworld_Iktotch',
        'homeworld_Iridonia', 'homeworld_Kalee', 'homeworld_Kashyyyk',
        'homeworld_Malastare', 'homeworld_Mirial', 'homeworld_Mon Cala',
        'homeworld_Muunilinst', 'homeworld_Naboo', 'homeworld_Ojom',
        'homeworld_Quermia', 'homeworld_Rodia', 'homeworld_Ryloth',
        'homeworld_Serenno', 'homeworld_Shili', 'homeworld_Skako',
        'homeworld_Socorro', 'homeworld_Stewjon', 'homeworld_Sullust',
        'homeworld_Tatooine', 'homeworld_Tholoth', 'homeworld_Toydaria',
        'homeworld_Trandosha', 'homeworld_Troiken', 'homeworld_Tund',
        'homeworld_Umbara', 'homeworld_Vulpter', 'homeworld_Zolan',
        'unit_type_at-at', 'unit_type_at-st', 'unit_type_resistance_soldier',
        'unit_type_stormtrooper', 'unit_type_tie_fighter',
        'unit_type_tie_silencer', 'unit_type_unknown', 'unit_type_x-wing']

@app.route("/api/predict", methods=["POST"])
def get_prediction():
    data = request.get_json(force = True)

    if isinstance(data, dict):
        data = [data]
    
    X = pd.DataFrame(data)

    X_encoded = pd.get_dummies(X, columns=X.columns)
    X_encoded_new = X_encoded.reindex(columns=train_columns, fill_value=0)
    # print(X_encoded.columns[0])
    # print(X_encoded.columns[1])

    pred = model.predict(X_encoded_new)

    return jsonify(pred.tolist())


if __name__ == "__main__":
    app.run(port=5000, debug=True)