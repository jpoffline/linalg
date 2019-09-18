package main

import (
	"bufio"
	"fmt"
	"image"
	"image/png"
	"math"
	"math/rand"

	"github.com/jpoffline/linalg/datastore"
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

	netArch := neural.NetworkArchitecture(2, 4, 1)

	neuralnet := neural.New(netArch)
	neuralnet.SetOutputLoc("output")

	td := xorTrainingData()
	neuralnet.Train(td, 100000)

	neuralnet.Serialise("nn.json")

	for _, d := range td {
		checkPredict(neuralnet, d.Inputs, d.Targets)
	}

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
	ds := datastore.New("nncircle")
	netArch := neural.NetworkArchitecture(2, 6, 1)

	neuralnet := neural.New(netArch)

	neuralnet.SetOutputLoc(ds.Root())
	neuralnet.SetLearningRate(0.5)
	neuralnet.Info()
	td := circleTrainingData(100)
	neuralnet.Train(td, 500000)

	myImage := image.NewRGBA(image.Rect(0, 0, 400, 400))
	outputFile, _ := ds.CreateFile("predictions.png")
	file, _ := ds.CreateFile("predictions.csv")
	defer file.Close()
	defer outputFile.Close()
	w := bufio.NewWriter(file)
	c := 0
	for i := -2.0; i <= 2; i += 0.01 {
		for j := -2.0; j <= 2; j += 0.01 {

			pred := neuralnet.Predict(neural.Vector{neural.Number(i), neural.Number(j)})[0]

			fmt.Fprintf(w, "%v, ", pred)
			myImage.Pix[c] = uint8(float64(pred) * 255)
			myImage.Pix[c+3] = 255
			c += 4
		}
		fmt.Fprintf(w, "\n")
	}

	png.Encode(outputFile, myImage)

}

func main() {
	//nnXOR()
	circleClassify()

}
