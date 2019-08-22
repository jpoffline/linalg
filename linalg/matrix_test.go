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
