package history

import (
	"bufio"
	"fmt"
	"os"
	"path/filepath"

	linalg "github.com/jpoffline/linalg/linearalgebra"
)

type Item struct {
	T     linalg.Number
	Score linalg.Number
}

func (h *History) Add(i Item) {
	h.items = append(h.items, i)
}

type History struct {
	items  []Item
	oloc   string
	histfn string
}

func New(oloc string) (h *History) {
	hh := &History{oloc: oloc}
	hh.histfn = "trainhistory.csv"
	return hh
}

func (h *History) Write() error {
	os.MkdirAll(h.oloc, os.ModePerm)
	filename := filepath.Join(h.oloc, h.histfn)
	file, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer file.Close()

	w := bufio.NewWriter(file)
	for _, line := range h.items {
		fmt.Fprintf(w, "%v,%v\n", line.T, line.Score)
	}
	return w.Flush()
}
