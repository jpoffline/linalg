package linearalgebra

import "testing"

func TestNewIdentityMatrix(t *testing.T) {
	mtx := NewIdentityMatrix(2)
	dimx, dimy := mtx.Len()
	if dimx != 2 && dimy != 2 {
		t.Errorf("dimensions not as expected")
	}

	if mtx.Value(0, 0) != 1 && mtx.Value(1, 1) != 1 {
		t.Errorf("not unity on diagonal")
	}
}

func TestMatrixMultiplicationForIncompatibleSquareMatrices(t *testing.T) {
	mtx1 := NewIdentityMatrix(2)
	mtx2 := NewIdentityMatrix(3)

	defer func() {
		if r := recover(); r == nil {
			t.Errorf("The code did not panic")
		}
	}()
	mtx1.Mul(mtx2)

}

func TestMatrixMultiplicationForIncompatibleRectangularMatrices(t *testing.T) {
	mtx1 := NewNumericMatrix(3, 2)
	mtx2 := NewNumericMatrix(3, 2)

	defer func() {
		if r := recover(); r == nil {
			t.Errorf("The code did not panic")
		}
	}()

	mtx1.Mul(mtx2)

}

func TestMatrixMultiplicationForCompatibleSquareMatrices(t *testing.T) {
	mtx1 := NewIdentityMatrix(2)
	mtx2 := NewIdentityMatrix(2)

	mtx3 := mtx1.Mul(mtx2)

	if mtx3.Value(1, 1) != 1 {
		t.Errorf("didnt get expected value on 1,1")
	}
	if mtx3.Value(0, 0) != 1 {
		t.Errorf("didnt get expected value on 0,0")
	}
	if mtx3.Value(0, 1) != 0 {
		t.Errorf("didnt get expected value on 0,1")
	}
	if mtx3.Value(1, 0) != 0 {
		t.Errorf("didnt get expected value on 1,0")
	}

}

func TestMatrixMultiplicationForCompatibleRectangularMatrices(t *testing.T) {
	mtx1 := NewNumericMatrix(3, 2)
	mtx2 := NewNumericMatrix(2, 3)

	mtx3 := mtx1.Mul(mtx2)

	if mtx3.dimx != mtx3.dimy {
		t.Errorf("should be a square matrix result")
	}

	if mtx3.dimx != 3 {
		t.Errorf("should be 3x3")
	}
}

func TestToVector(t *testing.T) {
	mtx := NewNumericMatrix(3, 2)
	vec := mtx.ToVector()
	if len(vec) != mtx.dimx*mtx.dimy {
		t.Errorf("vector not of expected size")
	}
}

func TestMatrixElemMul(t *testing.T) {
	mtx1 := NewIdentityMatrix(2)
	mtx2 := NewIdentityMatrix(2)
	mtx3 := mtx1.ElemMul(mtx2)

	if mtx3.Value(1, 1) != 1 {
		t.Errorf("didnt get expected value on 1,1")
	}
	if mtx3.Value(0, 0) != 1 {
		t.Errorf("didnt get expected value on 0,0")
	}
	if mtx3.Value(0, 1) != 0 {
		t.Errorf("didnt get expected value on 0,1")
	}
	if mtx3.Value(1, 0) != 0 {
		t.Errorf("didnt get expected value on 1,0")
	}
}

func TestMatrixTransposeForSquareMatrixInput(t *testing.T) {
	mtx1 := NewIdentityMatrix(2)
	mtx3 := mtx1.Transpose()

	if mtx3.Value(1, 1) != 1 {
		t.Errorf("didnt get expected value on 1,1")
	}
	if mtx3.Value(0, 0) != 1 {
		t.Errorf("didnt get expected value on 0,0")
	}
	if mtx3.Value(0, 1) != 0 {
		t.Errorf("didnt get expected value on 0,1")
	}
	if mtx3.Value(1, 0) != 0 {
		t.Errorf("didnt get expected value on 1,0")
	}
}

func TestMatrixTransposeNonSquare(t *testing.T) {
	mtx1 := NewNumericMatrix(2, 3)
	mtx3 := mtx1.Transpose()

	if mtx3.dimx != mtx1.dimy {
		t.Errorf("transposed dimx should be input dimy")
	}

	if mtx3.dimy != mtx1.dimx {
		t.Errorf("transposed dimy should be input dimx")
	}
}

func TestMatrixmap(t *testing.T) {
	mtx1 := NewNumericMatrix(2, 3)
	mtx3 := mtx1.Map(func(n Number) Number { return 4 })

	if mtx3.dimx != mtx1.dimx {
		t.Errorf("Map result should have same dimension as map input: dimx")
	}

	if mtx3.dimy != mtx1.dimy {
		t.Errorf("Map result should have same dimension as map input: dimy")
	}

	if mtx3.Value(1, 1) != 4 {
		t.Errorf("value after map not ok: 2,2")
	}

	if mtx3.Value(1, 2) != 4 {
		t.Errorf("value after map not ok: 2,3")
	}
}

func TestMatrixAdd(t *testing.T) {
	mtx1 := NewNumericMatrix(2, 3)
	mtx2 := NewNumericMatrix(2, 3)
	mtx3 := mtx1.Add(mtx2)

	if mtx3.dimx != mtx1.dimx {
		t.Errorf("Map result should have same dimension as map input: dimx")
	}

	if mtx3.dimy != mtx1.dimy {
		t.Errorf("Map result should have same dimension as map input: dimy")
	}

}

func TestMatrixAddIncompatible(t *testing.T) {
	mtx1 := NewNumericMatrix(2, 3)
	mtx2 := NewNumericMatrix(3, 3)

	defer func() {
		if r := recover(); r == nil {
			t.Errorf("The code did not panic")
		}
	}()

	mtx1.Add(mtx2)

}
