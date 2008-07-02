/* ---------------------------------------------------------------- */

#include "numpy/arrayobject.h"

#include "petsc4py/petsc4py_PETSc_npy.h"

/* ---------------------------------------------------------------- */

#include "petscvec.h"

static void Petsc_array_struct_del(void* cptr, void* descr)
{
  PyArrayInterface *inter = (PyArrayInterface *) cptr;
  PyObject *self = (PyObject *)descr;
  Py_XDECREF(self);
  if (inter != NULL) {
    PyMem_Del(inter->shape);
    Py_XDECREF(inter->descr);
    PyMem_Del(inter);
  }
}


static PyObject* Petsc_array_struct_new(PyObject* self, 
					void* array, PetscInt size,
					int NPY_PETSC_TYPE, int flags)
{
  npy_intp         *shape = NULL;
  PyArray_Descr    *descr = NULL;
  PyArrayInterface *inter = NULL;
  PyObject         *cobj  = NULL;
  Py_INCREF(self);
  inter = PyMem_New(PyArrayInterface, 1);
  if (inter == NULL) { PyErr_NoMemory(); goto fail; }
  shape = PyMem_New(npy_intp, 1);
  if (shape == NULL) { PyErr_NoMemory(); goto fail; }
  inter->shape = shape;
  descr = PyArray_DescrFromType(NPY_PETSC_TYPE);
  if (descr == NULL) { goto fail; }
  /* fill array interface struct */
  inter->two = 2;
  inter->data = (void *) array;
  inter->nd = 1;
  inter->shape[0] = (npy_intp)size;
  inter->strides = NULL;
  inter->descr = (PyObject *)descr;
  inter->typekind = descr->kind;
  inter->itemsize = descr->elsize;
  inter->flags  = NPY_C_CONTIGUOUS | NPY_F_CONTIGUOUS;
  inter->flags |= NPY_ALIGNED | NPY_NOTSWAPPED;
  inter->flags |= NPY_ARR_HAS_DESCR;
  inter->flags |= flags;
  /* create C Object holding array interface struct and data owner */
  cobj = PyCObject_FromVoidPtrAndDesc(inter, self, 
				      Petsc_array_struct_del);
  if (cobj == NULL) { 
    Petsc_array_struct_del((void* )inter, (void* )self); 
  }
  return cobj;

 fail:
  PyMem_Del(inter); 
  PyMem_Del(shape);
  Py_DECREF(self);
  return NULL;
}

static PyObject* PetscIS_array_struct(PyObject* self, IS is)
{
  PetscTruth valid  = PETSC_FALSE;
  PetscTruth stride = PETSC_FALSE;
  PetscTruth block  = PETSC_FALSE;
  PetscInt   size   = 0;
  PetscInt   *array = PETSC_NULL;
  PyObject   *iface = NULL;
  /* check index set handle */
  ISValid(is,&valid);
  if (!valid) {
    PyErr_SetString(PyExc_ValueError, "index set is not valid");
    return NULL;
  }
  ISStride(is, &stride); 
  ISBlock(is, &block);
  /* get index set indices and size*/
  ISGetIndices(is, &array); 
  ISGetLocalSize(is, &size);
  if (stride || block) { /* perhaps I should find a better way */
    npy_intp s = (npy_intp) size;
    PyObject* ary = PyArray_EMPTY(1, &s, NPY_PETSC_INT, 0);
    if (ary != NULL) {
      PetscMemcpy(PyArray_DATA(ary), array, size*sizeof(PetscInt));
      iface = Petsc_array_struct_new(ary, PyArray_DATA(ary), size,
				     NPY_PETSC_INT, NPY_WRITEABLE);
      Py_DECREF(ary);
    }

  } else {  
    iface = Petsc_array_struct_new(self, (void *)array, size,
				   NPY_PETSC_INT, 0);
  }
  ISRestoreIndices(is, &array);
  return iface;
}

static PyObject* PetscVec_array_struct(PyObject* self, Vec vec) 
{
  PetscTruth  valid  = PETSC_FALSE;
  PetscInt    size   = 0;
  PetscScalar *array = PETSC_NULL;
  PyObject    *iface = NULL;
  /* check vector handle */
  VecValid(vec,&valid);
  if (!valid) {
    PyErr_SetString(PyExc_ValueError, "vector is not valid");
    return NULL;
  }
  /* get vector array and size*/
  VecGetArray(vec, &array);
  VecGetLocalSize(vec, &size);
  if (!vec->petscnative) {
    npy_intp s = (npy_intp) size;
    PyObject* ary = PyArray_EMPTY(1, &s, NPY_PETSC_SCALAR, 0);
    if (ary != NULL) {
      PetscMemcpy(PyArray_DATA(ary), array, size*sizeof(PetscScalar));
      iface = Petsc_array_struct_new(ary, PyArray_DATA(ary), size,
				     NPY_PETSC_SCALAR, NPY_WRITEABLE);
      Py_DECREF(ary);
    }
  } else {
    iface = Petsc_array_struct_new(self, (void *)array, size,
				   NPY_PETSC_SCALAR, NPY_WRITEABLE);
  }
  VecRestoreArray(vec, &array);
  return iface;
}

/* ---------------------------------------------------------------- */
