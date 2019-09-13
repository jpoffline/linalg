package linearalgebra

// Number is the type used for numbers.
type Number float64

// NumericVector is the struct for
// holding a numeric vector.
type NumericVector struct {
	values []Number
	dim    int
}

// NumericMatrix is the struct for
// holding a numeric matrix.
type NumericMatrix struct {
	values     [][]Number
	dimx, dimy int
}
