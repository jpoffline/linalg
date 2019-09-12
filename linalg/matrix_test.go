package linalg

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

func TestMatrixMultiplicationForIncompatibleMatrices(t *testing.T) {
	mtx1 := NewIdentityMatrix(2)
	mtx2 := NewIdentityMatrix(3)

	_, err := mtx1.Mul(mtx2)
	if err.Error() != "incompatible matrix multiplication attempted" {
		t.Errorf("didnt get expected error string")
	}

}

func TestMatrixMultiplicationForCompatibleMatrices(t *testing.T) {
	mtx1 := NewIdentityMatrix(2)
	mtx2 := NewIdentityMatrix(2)

	mtx3, _ := mtx1.Mul(mtx2)

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
