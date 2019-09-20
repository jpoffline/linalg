package neural

import linalg "github.com/jpoffline/linalg/linearalgebra"

func (nn *NeuralNet) build(meta Meta) {

	lyrs := []NeuralLayerMeta{
		NeuralLayerMeta{ID: 0, NumNeurons: meta.numInputs},
		NeuralLayerMeta{ID: 1, NumNeurons: meta.numHidden},
		NeuralLayerMeta{ID: 2, NumNeurons: meta.numOutput},
	}

	for idx := 0; idx < len(lyrs)-1; idx++ {

		thislayer := neurallayer{
			weights: linalg.NewRandomMatrix(lyrs[idx+1].NumNeurons, lyrs[idx].NumNeurons),
			bias:    linalg.NewRandomMatrix(lyrs[idx+1].NumNeurons, 1),
		}

		nn.layers = append(nn.layers, thislayer)

	}

}
