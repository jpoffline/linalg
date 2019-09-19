package neural

import (
	"fmt"
	"math"
	"math/rand"

	linalg "github.com/jpoffline/linalg/linearalgebra"

	history "github.com/jpoffline/linalg/neural/history"
)

// Train will train the network on the provided training data
// for the given number of iterations.
func (nn *NeuralNet) Train(data []TrainingData, iters int) {
	ndata := len(data)
	fmt.Printf("* training neural net with %v peices of training data over %v iterations\n", ndata, iters)

	hist := history.New(nn.OutputData.Loc)
	defer hist.Write()
	for i := 0; i < iters; i++ {
		data := data[rand.Intn(ndata)]
		score := nn.train(data.Inputs, data.Targets)
		if math.Mod(float64(i), float64(iters)/1000) == 0 {
			hist.Add(history.Item{T: linalg.Number(i) * nn.meta.learningRate, Score: score})
		}
	}

	fmt.Printf("* training complete\n")
}

// train will train the net for the provided inputs and targets.
func (nn *NeuralNet) train(inputs, targets Vector) Number {
	nn.trainCount++
	// prepare inputs.
	im := linalg.NewNumericMatrixFromSlice(inputs)

	// generate the hidden activations.
	nn.doLayerCalc(im, 0)
	// generate the output activations.
	nn.doLayerCalc(nn.layers[0].activations, 1)

	// calc output layer errors.
	targetsM := linalg.NewNumericMatrixFromSlice(targets)
	errorsOutput := calcErrorOutputLayer(targetsM, nn.layers[1].activations)
	nn.calcNewWeightsBias(nn.layers[0].activations, errorsOutput, 1)

	// calc hidden layer errors
	errorsHidden := calcErrorHiddenLayer(nn.layers[1].weights, errorsOutput)
	nn.calcNewWeightsBias(im, errorsHidden, 0)

	return errorsOutput.Mag()
}

func (nn *NeuralNet) calcNewWeightsBias(ip, e Matrix, lyridx int) {
	gradHidden := gradient(nn.layers[lyridx].activations, e)
	gradHidden = gradHidden.Map(func(elem Number) Number {
		return elem * nn.meta.learningRate
	})
	inputsT := ip.Transpose()
	weightsIHdel := gradHidden.Mul(inputsT)

	// adjust
	nn.layers[lyridx].weights = nn.layers[lyridx].weights.Add(weightsIHdel)
	nn.layers[lyridx].bias = nn.layers[lyridx].bias.Add(gradHidden)
}

func gradient(output, errors Matrix) Matrix {
	o2 := output.Map(func(elem Number) Number {
		return linalg.Dsigmoid2(elem)
	})
	return o2.ElemMul(errors)
}

func calcErrorOutputLayer(t, o Matrix) Matrix {
	return t.Subtract(o)
}

func calcErrorHiddenLayer(w, e Matrix) Matrix {
	weightsHOT := w.Transpose()
	return weightsHOT.Mul(e)
}
