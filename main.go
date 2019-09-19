package main

import (
	"fmt"
	"math"
	"math/rand"

	"github.com/jpoffline/linalg/neural"
)

func NewTrainPointValue(x, y, p float64) neural.TrainingData {
	td := neural.TrainingData{
		Inputs:  neural.Vector{neural.Number(x), neural.Number(y)},
		Targets: neural.Vector{neural.Number(p)},
	}
	return td
}

func checkPredict(nn *neural.NeuralNet, i, t neural.Vector) {
	out := nn.Predict(i)
	fmt.Printf("Input: %v, output: %v, expected: %v\n", i, out, t)
}

func xorTrainingData() []neural.TrainingData {
	return []neural.TrainingData{
		NewTrainPointValue(1.0, 1.0, 0.0),
		NewTrainPointValue(1.0, 0.0, 1.0),
		NewTrainPointValue(0.0, 1.0, 1.0),
		NewTrainPointValue(0.0, 0.0, 0.0),
	}
}

func nnXOR() {

	netArch := neural.NetworkArchitecture(2, 2, 1)

	neuralnet := neural.New(netArch)
	neuralnet.InitDataStore("xor")
	neuralnet.Info()

	td := xorTrainingData()
	neuralnet.Train(td, 1000000)

	neuralnet.Serialise("nn.json")

	for _, d := range td {
		checkPredict(neuralnet, d.Inputs, d.Targets)
	}

	neuralnet.ClassifyDraw(1000000)

}

func circleIsIn(x, y float64) float64 {
	r := math.Sqrt(x*x + y*y)

	if r < 0.5 {
		return 1.0
	}

	return 0.0
}

func circleTrainingData(np int) []neural.TrainingData {

	data := []neural.TrainingData{}

	for p := 0; p < np; p++ {
		x := rand.Float64()*4 - 2
		y := rand.Float64()*4 - 2
		data = append(data, NewTrainPointValue(x, y, circleIsIn(x, y)))
	}

	return data
}

func circleClassify() {
	netArch := neural.NetworkArchitecture(2, 6, 1)

	neuralnet := neural.New(netArch)
	neuralnet.InitDataStore("nncircle")

	neuralnet.SetLearningRate(0.5)
	neuralnet.Info()
	td := circleTrainingData(20000)

	neuralnet.Train(td, 10)
	neuralnet.ClassifyDraw(10)

	neuralnet.Train(td, 100)
	neuralnet.ClassifyDraw(110)

	neuralnet.Train(td, 1000)
	neuralnet.ClassifyDraw(1110)

	neuralnet.Train(td, 10000)
	neuralnet.ClassifyDraw(11110)

	neuralnet.Train(td, 100000)
	neuralnet.ClassifyDraw(111110)

	neuralnet.Serialise("nn.json")

}

func main() {
	//nnXOR()
	circleClassify()

}
