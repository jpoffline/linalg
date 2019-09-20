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

	nlayers := len(nn.layers) - 1

	for l := 0; l < nlayers; l++ {

		// generate activations per layer.
		nn.doLayerCalc(nn.layers[l].activations, l+1)
	}

	// send back activations of the final (output) layer
	// to caller, as a vector.
	return nn.layers[nlayers].activations.ToVector()
}

func (nn *NeuralNet) doLayerCalc(inputs Inputs, lyridx int) {
	// multiply W x I
	p1 := nn.layers[lyridx].weights.Mul(inputs)
	// now add in the bias
	p2 := p1.Add(nn.layers[lyridx].bias)
	// apply activation function
	nn.layers[lyridx].activations = p2.Map(func(num Number) Number { return linalg.Sigmoid(num) })
}
