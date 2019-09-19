package neural

import (
	"image"
	"image/png"
	"strconv"
)

func (neuralnet *NeuralNet) ClassifyDraw(id int) {

	xmin := -1.0
	xmax := 1.0
	ymin := xmin
	ymax := xmax

	imgsize := 100

	dx := (xmax - xmin) / float64(imgsize)
	dy := (ymax - ymin) / float64(imgsize)

	myImage := image.NewRGBA(image.Rect(0, 0, imgsize, imgsize))
	outputFile, _ := neuralnet.DataStore.CreateFile("predictions-" + strconv.Itoa(id) + ".png")

	defer outputFile.Close()

	c := 0

	for i := xmin; i <= xmax; i += dx {
		for j := ymin; j <= ymax; j += dy {

			pred := float64(neuralnet.Predict(Vector{Number(i), Number(j)})[0])

			myImage.Pix[c] = uint8(pred * 255)
			myImage.Pix[c+3] = 255
			c += 4
		}

	}

	png.Encode(outputFile, myImage)

}
