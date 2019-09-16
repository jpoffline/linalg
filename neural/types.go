package neural

import (
	"encoding/json"

	linalg "github.com/jpoffline/linalg/linearalgebra"
)

// NeuralNet is the main neural net struct, containing
// all the required data.
type NeuralNet struct {
	numInputs, numHidden, numOutput int
	trainCount                      int
	weightsIH                       Weights
	weightsHO                       Weights
	biasIH                          Bias
	biasHO                          Bias
	learningRate                    linalg.Number
}

// Weights is the type to hold the weights connecting neurons
type Weights = *linalg.NumericMatrix

// Bias is the type to hold the biases
type Bias = *linalg.NumericMatrix

// Inputs is the type to hold the inputs to neurons
type Inputs = *linalg.NumericMatrix

// Outputs is the type to hold the outputs of the neural net
type Outputs = *linalg.NumericMatrix

type TrainingData struct {
	Inputs  []linalg.Number
	Targets []linalg.Number
}

func (m *NeuralNet) MarshalJSON() ([]byte, error) {
	return json.Marshal(map[string]interface{}{
		"learning_rate": m.learningRate,
		"train_loops":   m.trainCount,
		"weights_ih":    m.weightsIH,
		"weights_ho":    m.weightsHO,
		"bias_ih":       m.biasIH,
		"bias_ho":       m.biasHO,
	})
}
