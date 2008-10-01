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
    Petsc_array_struct_del((void *)inter, (void *)self);
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
  PetscErrorCode ierr;
  PetscTruth 	 valid  = PETSC_FALSE;
  PetscTruth 	 stride = PETSC_FALSE;
  PetscTruth 	 block  = PETSC_FALSE;
  PetscInt   	 size   = 0;
  const PetscInt *array = PETSC_NULL;
  PyObject   *iface = NULL;
  /* check index set handle */
  ierr = ISValid(is,&valid);
  if (!valid) {
    PyErr_SetString(PyExc_ValueError, "index set is not valid");
    return NULL;
  }
  /* check index set type */
  ierr = ISStride(is, &stride);
  ierr = ISBlock(is, &block);
  if (stride || block) {
    PyErr_SetString(PyExc_ValueError, "index set is not general");
    return NULL;
  }
  /* get index set size and array */
  ierr = ISGetLocalSize(is, &size);    /* XXX */
  ierr = ISGetIndices(is, &array);     /* XXX */
  iface = Petsc_array_struct_new(self, (void *)array, size,
				 NPY_PETSC_INT, 0);
  ierr = ISRestoreIndices(is, &array); /* XXX */
  return iface;
}

static PyObject* PetscVec_array_struct(PyObject* self, Vec vec)
{
  PetscErrorCode ierr;
  PetscTruth  valid  = PETSC_FALSE;
  PetscInt    size   = 0;
  PetscScalar *array = PETSC_NULL;
  PyObject    *iface = NULL;
  /* check vector handle */
  ierr = VecValid(vec, &valid);
  if (!valid) {
    PyErr_SetString(PyExc_ValueError, "vector is not valid");
    return NULL;
  }
  /* check vector is native */
  if (!vec->petscnative) {
    PyErr_SetString(PyExc_ValueError, "vector is not native");
    return NULL;
  }
  /* get vector size and array */
  ierr = VecGetLocalSize(vec, &size);  /* XXX */
  ierr = VecGetArray(vec, &array);     /* XXX */
  iface = Petsc_array_struct_new(self, (void *)array, size,
				 NPY_PETSC_SCALAR, NPY_WRITEABLE);
  ierr = VecRestoreArray(vec, &array); /* XXX */
  return iface;
}

/* ---------------------------------------------------------------- */
