package neural

import (
	"encoding/json"
	"io/ioutil"
	"path/filepath"
)

// Serialise will write the neural net to json file.
func (nn *NeuralNet) Serialise(filename string) {
	fn := filepath.Join(nn.OutputData.Loc, filename)
	file, _ := json.MarshalIndent(nn, "", " ")
	_ = ioutil.WriteFile(fn, file, 0644)
}

// MarshalJSON will convert a NeuralNet object into a json string.
func (nn *NeuralNet) MarshalJSON() ([]byte, error) {
	return json.Marshal(map[string]interface{}{
		"meta":       nn.meta,
		"weights_ih": nn.weightsIH,
		"weights_ho": nn.weightsHO,
		"bias_ih":    nn.biasIH,
		"bias_ho":    nn.biasHO,
	})
}

// MarshalJSON will convert a NeuralNet object into a json string.
func (m Meta) MarshalJSON() ([]byte, error) {
	return json.Marshal(map[string]interface{}{
		"numInputs":    m.numInputs,
		"numHidden":    m.numHidden,
		"numOutput":    m.numOutput,
		"trainCount":   m.trainCount,
		"learningRate": m.learningRate,
	})
}
