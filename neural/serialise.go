package neural

import (
	"encoding/json"
	"io/ioutil"
)

func (nn *NeuralNet) Serialise(filename string) {
	file, _ := json.MarshalIndent(nn, "", " ")
	_ = ioutil.WriteFile(filename, file, 0644)
}
