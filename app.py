from flask import Flask, request, jsonify
import algorithms

app = Flask(__name__)

@app.route('/genetic', methods=['POST'])
def genetic():
	dataset = request.get_json()
	print(dataset)
	result = algorithms.genetic_algorithm(dataset)
	return jsonify(result), 200


@app.route('/knn', methods=['POST'])
def knn():
	dataset = request.get_data('dataset')
	result = algorithms.knn_algorithm(dataset)

	return jsonify({
		'accuracy': result,
	}), 200


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
