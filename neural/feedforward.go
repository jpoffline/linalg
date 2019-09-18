package neural

import (
	linalg "github.com/jpoffline/linalg/linearalgebra"
)

// Predict is the place to pass the provided inputs to the neural network.
func (nn *NeuralNet) Predict(inputs Vector) Vector {
	// prepare the input.
	im := linalg.NewNumericMatrixFromSlice(inputs)
	// generate the hidden outputs.
	nn.doLayerCalc(im, 0)
	// generate the final output.
	nn.doLayerCalc(nn.layers[0].activations, 1)
	// send back to caller as a vector.
	return nn.layers[1].activations.ToVector()
}

func (nn *NeuralNet) doLayerCalc(inputs Inputs, lyridx int) {
	// multiply W x I
	p1 := nn.layers[lyridx].weights.Mul(inputs)
	// now add in the bias
	p2 := p1.Add(nn.layers[lyridx].bias)
	// apply activation function
	nn.layers[lyridx].activations = p2.Map(func(num Number) Number { return linalg.Sigmoid(num) })
}
