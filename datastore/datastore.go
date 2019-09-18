package datastore

import (
	"os"
	"path/filepath"
)

const DSROOT = "output"

type DataStore struct {
	appname string
	root    string
}

func (ds *DataStore) PathTo(fn string) string {
	return filepath.Join(ds.root, fn)
}

func (ds *DataStore) Root() string {
	return ds.root
}

func New(app string) *DataStore {
	ds := &DataStore{appname: app}
	ds.root = filepath.Join(DSROOT, ds.appname)
	os.MkdirAll(ds.root, os.ModePerm)
	return ds
}

func (ds *DataStore) CreateFile(fn string) (*os.File, error) {
	return os.Create(ds.PathTo(fn))
}
