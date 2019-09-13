package linearalgebra

import "fmt"

// Mul will multiply the provided matrices together.
// We check for dimensional-compatibility.
func (mtx *NumericMatrix) Mul(mtx1 *NumericMatrix) *NumericMatrix {
	if mtx.dimy != mtx1.dimx {
		mtx.Print()
		mtx1.Print()
		panic("incompatible matrix multiplication attempted")
		return nil
	}

	res := NewNumericMatrix(mtx.dimx, mtx1.dimy)

	for i := 0; i < res.dimx; i++ {
		for j := 0; j < res.dimy; j++ {
			for k := 0; k < mtx1.dimx; k++ {
				res.values[i][j] += mtx.values[i][k] * mtx1.values[k][j]
			}
		}
	}

	return res
}

// ElemMul will perform element-wise multiplication.
func (mtx *NumericMatrix) ElemMul(mtx1 *NumericMatrix) *NumericMatrix {
	if mtx.dimx != mtx1.dimx {
		mtx.Print()
		mtx1.Print()
		panic("incompatible matrix element-multiplication attempted: dimx")
		return nil
	}

	if mtx.dimy != mtx1.dimy {
		mtx.Print()
		mtx1.Print()
		panic("incompatible matrix element-multiplication attempted: dimy")
		return nil
	}

	res := NewNumericMatrix(mtx.dimx, mtx1.dimy)

	for i := 0; i < res.dimx; i++ {
		for j := 0; j < res.dimy; j++ {
			res.values[i][j] = mtx.values[i][j] * mtx1.values[i][j]
		}
	}

	return res
}

// Add will add two matrices together and return a new matrix.
func (mtx *NumericMatrix) Add(mtx1 *NumericMatrix) *NumericMatrix {
	if mtx.dimx != mtx1.dimx {
		panic("incompatible matrix addition attempted: dimx")
		return nil
	}

	if mtx.dimy != mtx1.dimy {
		mtx.Print()
		mtx1.Print()
		panic("incompatible matrix addition attempted: dimy")
		return nil
	}

	res := NewNumericMatrix(mtx.dimx, mtx.dimy)
	for i := 0; i < mtx.dimx; i++ {
		for j := 0; j < mtx.dimy; j++ {
			res.values[i][j] = mtx.values[i][j] + mtx1.values[i][j]
		}
	}
	return res
}

// Subtract will subtract the provided matrix and return a new matrix.
func (mtx *NumericMatrix) Subtract(mtx1 *NumericMatrix) *NumericMatrix {
	if mtx.dimx != mtx1.dimx {
		panic("incompatible matrix subtraction attempted: dimx")
		return nil
	}

	if mtx.dimy != mtx1.dimy {
		mtx.Print()
		mtx1.Print()
		panic("incompatible matrix subtraction attempted: dimy")
		return nil
	}

	res := NewNumericMatrix(mtx.dimx, mtx.dimy)
	for i := 0; i < mtx.dimx; i++ {
		for j := 0; j < mtx.dimy; j++ {
			res.values[i][j] = mtx.values[i][j] - mtx1.values[i][j]
		}
	}
	return res
}

// Map will apply the provided function to every element of the matrix
// and return a new matrix.
func (mtx *NumericMatrix) Map(cb func(Number) Number) *NumericMatrix {
	res := NewNumericMatrix(mtx.dimy, mtx.dimx)
	for i := 0; i < mtx.dimx; i++ {
		for j := 0; j < mtx.dimy; j++ {
			res.values[i][j] = cb(mtx.values[i][j])
		}
	}
	return res
}

// Value will return the value of the matrix at the
// provided i,j values.
func (mtx *NumericMatrix) Value(i, j int) Number {
	return mtx.values[i][j]
}

// Set will set the value of the matrix at i,j to be
// the provided value.
func (mtx *NumericMatrix) Set(i, j int, val Number) {
	mtx.values[i][j] = val
}

// Print will crudely print out the matrix.
func (mtx *NumericMatrix) Print() {
	fmt.Printf("matrix has dim %v x %v\n", mtx.dimx, mtx.dimy)
	for _, rr := range mtx.values {
		fmt.Println(rr)
	}
}

// Len returns the dimx and dimy of the matrix.
func (mtx *NumericMatrix) Len() (int, int) {
	return mtx.dimx, mtx.dimy
}

// ToVector will convert a matrix into a vector by squashing.
func (mtx *NumericMatrix) ToVector() *NumericVector {
	vec := NewEmptyNumericVector()
	for i := 0; i < mtx.dimx; i++ {
		for j := 0; j < mtx.dimy; j++ {
			vec.Push(mtx.values[i][j])
		}
	}
	return vec
}

// Transpose returns the transpose of the matrix.
func (mtx *NumericMatrix) Transpose() *NumericMatrix {
	res := NewNumericMatrix(mtx.dimy, mtx.dimx)
	for i := 0; i < mtx.dimx; i++ {
		for j := 0; j < mtx.dimy; j++ {
			res.values[j][i] = mtx.values[i][j]
		}
	}
	return res
}
