package neural

import (
	linalg "github.com/jpoffline/linalg/linearalgebra"
)

// Train will train the net for the provided inputs and targets.
func (nn *NeuralNet) Train(inputs, targets *linalg.NumericVector) {

	// prepare inputs.
	im := linalg.NewNumericMatrixFromVector(*inputs)
	// generate the hidden outputs.

	ho1 := nn.weightsIH.Mul(im)
	hiddenOutputs := ho1.Add(nn.biasIH)
	hiddenOutputs = hiddenOutputs.Map(func(num linalg.Number) linalg.Number { return linalg.Sigmoid(num) })

	// generate the final output.
	fo1 := nn.weightsHO.Mul(hiddenOutputs)
	outputs := fo1.Add(nn.biasHO)
	outputs = outputs.Map(func(num linalg.Number) linalg.Number { return linalg.Sigmoid(num) })

	// prepare the targets
	targetsM := linalg.NewNumericMatrixFromVector(*targets)

	// calc output layer errors.
	errorsOutput := calcErrorOutputLayer(targetsM, outputs)

	// calc gradient
	gradOutput := gradient(nn.learningRate, outputs, errorsOutput)
	hoT := hiddenOutputs.Transpose()
	weightsHOdel := gradOutput.Mul(hoT)

	// adjust
	nn.weightsHO = nn.weightsHO.Add(weightsHOdel)
	nn.biasHO = nn.biasHO.Add(gradOutput)

	// calc hidden layer errors
	errorsHidden := calcErrorHiddenLayer(nn.weightsHO, errorsOutput)

	// calc gradient
	gradHidden := gradient(nn.learningRate, hiddenOutputs, errorsHidden)
	inputsT := im.Transpose()
	weightsIHdel := gradHidden.Mul(inputsT)

	// adjust
	nn.weightsIH = nn.weightsIH.Add(weightsIHdel)
	nn.biasIH = nn.biasIH.Add(gradHidden)
}

func gradient(lr linalg.Number, output, errors *linalg.NumericMatrix) *linalg.NumericMatrix {
	o2 := output.Map(func(elem linalg.Number) linalg.Number { return linalg.Dsigmoid2(elem) })
	eo := o2.ElemMul(errors)
	return eo.Map(func(elem linalg.Number) linalg.Number { return elem * lr })
}

func calcErrorOutputLayer(t, o *linalg.NumericMatrix) *linalg.NumericMatrix {
	return t.Subtract(o)
}

func calcErrorHiddenLayer(w, e *linalg.NumericMatrix) *linalg.NumericMatrix {
	weightsHOT := w.Transpose()
	return weightsHOT.Mul(e)
}
