package neural

import (
	"math"
	"math/rand"

	linalg "github.com/jpoffline/linalg/linearalgebra"
	history "github.com/jpoffline/linalg/neural/history"
)

// Train will train the network on the provided training data
// for the given number of iterations.
func (nn *NeuralNet) Train(data []TrainingData, iters int) {
	ndata := len(data)
	hist := history.New(nn.OutputData.Loc)
	defer hist.Write()
	for i := 0; i < iters; i++ {
		data := data[rand.Intn(ndata)]
		score := nn.train(data.Inputs, data.Targets)
		if math.Mod(float64(i), float64(iters)/1000) == 0 {
			hist.Add(history.Item{T: linalg.Number(i) * nn.meta.learningRate, Score: score})
		}
	}

}

// train will train the net for the provided inputs and targets.
func (nn *NeuralNet) train(inputs, targets []linalg.Number) linalg.Number {
	nn.trainCount++
	// prepare inputs.
	im := linalg.NewNumericMatrixFromSlice(inputs)

	// generate the hidden outputs.

	ho1 := nn.layers[0].weights.Mul(im)
	hiddenOutputs := ho1.Add(nn.layers[0].bias)
	hiddenOutputs = hiddenOutputs.Map(func(num linalg.Number) linalg.Number { return linalg.Sigmoid(num) })

	// generate the final output.
	fo1 := nn.layers[1].weights.Mul(hiddenOutputs)
	outputs := fo1.Add(nn.layers[1].bias)
	outputs = outputs.Map(func(num linalg.Number) linalg.Number { return linalg.Sigmoid(num) })

	// calc output layer errors.
	targetsM := linalg.NewNumericMatrixFromSlice(targets)
	errorsOutput := calcErrorOutputLayer(targetsM, outputs)
	nn.calcNewWeightsBias(hiddenOutputs, outputs, errorsOutput, 1)

	// calc hidden layer errors
	errorsHidden := calcErrorHiddenLayer(nn.layers[1].weights, errorsOutput)
	nn.calcNewWeightsBias(im, hiddenOutputs, errorsHidden, 0)

	return errorsOutput.Mag()
}

func (nn *NeuralNet) calcNewWeightsBias(ip, o, e *linalg.NumericMatrix, lyridx int) {
	gradHidden := gradient(o, e)
	gradHidden = gradHidden.Map(func(elem linalg.Number) linalg.Number { return elem * nn.meta.learningRate })
	inputsT := ip.Transpose()
	weightsIHdel := gradHidden.Mul(inputsT)

	// adjust
	nn.layers[lyridx].weights = nn.layers[lyridx].weights.Add(weightsIHdel)
	nn.layers[lyridx].bias = nn.layers[lyridx].bias.Add(gradHidden)
}

func gradient(output, errors *linalg.NumericMatrix) *linalg.NumericMatrix {
	o2 := output.Map(func(elem linalg.Number) linalg.Number { return linalg.Dsigmoid2(elem) })
	return o2.ElemMul(errors)
}

func calcErrorOutputLayer(t, o *linalg.NumericMatrix) *linalg.NumericMatrix {
	return t.Subtract(o)
}

func calcErrorHiddenLayer(w, e *linalg.NumericMatrix) *linalg.NumericMatrix {
	weightsHOT := w.Transpose()
	return weightsHOT.Mul(e)
}
