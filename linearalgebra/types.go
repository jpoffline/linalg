package linearalgebra

import "encoding/json"

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

// MarshalJSON will convert a NumericMatrix object into a json string.
func (m *NumericMatrix) MarshalJSON() ([]byte, error) {
	return json.Marshal(map[string]interface{}{
		"dimx":   m.dimx,
		"dimy":   m.dimy,
		"values": m.values,
	})
}
