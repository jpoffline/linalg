package linalg

type Number float64

type NumericVector struct {
	values []Number
	dim    int
}

type NumericMatrix struct {
	values     [][]Number
	dimx, dimy int
}
