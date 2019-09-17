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

	ho1 := nn.weightsIH.Mul(im)
	hiddenOutputs := ho1.Add(nn.biasIH)
	hiddenOutputs = hiddenOutputs.Map(func(num linalg.Number) linalg.Number { return linalg.Sigmoid(num) })

	// generate the final output.
	fo1 := nn.weightsHO.Mul(hiddenOutputs)
	outputs := fo1.Add(nn.biasHO)
	outputs = outputs.Map(func(num linalg.Number) linalg.Number { return linalg.Sigmoid(num) })

	// calc output layer errors.
	targetsM := linalg.NewNumericMatrixFromSlice(targets)
	errorsOutput := calcErrorOutputLayer(targetsM, outputs)
	nn.weightsHO, nn.biasHO = calcNewWeightsBias(nn.meta.learningRate, hiddenOutputs, outputs, errorsOutput, nn.weightsHO, nn.biasHO)

	// calc hidden layer errors
	errorsHidden := calcErrorHiddenLayer(nn.weightsHO, errorsOutput)
	nn.weightsIH, nn.biasIH = calcNewWeightsBias(nn.meta.learningRate, im, hiddenOutputs, errorsHidden, nn.weightsIH, nn.biasIH)

	return errorsOutput.Mag()
}

func calcNewWeightsBias(lr linalg.Number, ip, o, e, w, b *linalg.NumericMatrix) (*linalg.NumericMatrix, *linalg.NumericMatrix) {
	gradHidden := gradient(o, e)
	gradHidden = gradHidden.Map(func(elem linalg.Number) linalg.Number { return elem * lr })
	inputsT := ip.Transpose()
	weightsIHdel := gradHidden.Mul(inputsT)

	// adjust
	return w.Add(weightsIHdel), b.Add(gradHidden)
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
