/* ---------------------------------------------------------------- */

#include "numpy/arrayobject.h"

#include "petsc4py/petsc4py_PETSc_npy.h"

/* ---------------------------------------------------------------- */

#include "petscvec.h"

static void Petsc_array_struct_del(void* cptr, void* descr)
{
  PyArrayInterface *inter = (PyArrayInterface *) cptr;
  PyObject *self = (PyObject *)descr;
  Py_DecRef(self);
  if (inter != NULL) {
    PyMem_Del(inter->shape);
    Py_DecRef(inter->descr);
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
  Py_IncRef(self);
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
  Py_DecRef(self);
  return NULL;
}

static PyObject* PetscIS_array_struct(PyObject* self, IS is)
{
  PetscErrorCode ierr;
  PetscTruth     stride = PETSC_FALSE;
  PetscTruth     block  = PETSC_FALSE;
  PetscInt       size   = 0;
  const PetscInt *array = PETSC_NULL;
  PyObject       *iface = NULL;
  /* get index set size */
  ierr = ISGetLocalSize(is, &size); if (ierr) {goto fail;}
  /* check index set type */
  ierr = ISStride(is,&stride); if (ierr) {goto fail;}
  ierr = ISBlock(is,&block); if (ierr) {goto fail;}
  if (stride || block) {
    PyErr_SetString(PyExc_ValueError, "index set is not general");
    return NULL;
  }
  ierr = ISGetIndices(is, &array); if (ierr) {goto fail;}
  iface = Petsc_array_struct_new(self,(void *)array,size,
				 NPY_PETSC_INT,/*READONLY*/0);
  ierr = ISRestoreIndices(is, &array); if (ierr) {goto fail;}
  return iface;
 fail:
  Py_XDECREF(iface);
  {
    const char *text=0; char *specific=0;
    PetscErrorMessage(ierr,&text,&specific);
    PyErr_Format(PyExc_RuntimeError,
		 "PETSc error [code %d]:\n%s\n%s\n",
		 (int)ierr,text?text:"",specific?specific:"");
  }
  return NULL;
}

static PyObject* PetscVec_array_struct(PyObject* self, Vec vec)
{
  PetscErrorCode ierr;
  PetscInt    	 size   = 0;
  PetscScalar 	 *array = PETSC_NULL;
  PyObject    	 *iface = NULL;
  /* get vector size */
  ierr = VecGetLocalSize(vec, &size); if (ierr) goto fail;
  /* check vector is native */
  if (!vec->petscnative) {
    PyErr_SetString(PyExc_ValueError, "vector is not native");
    return NULL;
  }
  ierr = VecGetArray(vec, &array); if (ierr) goto fail;
  iface = Petsc_array_struct_new(self,(void *)array,size,
                                 NPY_PETSC_SCALAR, NPY_WRITEABLE);
  ierr = VecRestoreArray(vec, &array); if (ierr) goto fail;
  return iface;
 fail:
  Py_XDECREF(iface);
  {
    const char *text=0; char *specific=0;
    PetscErrorMessage(ierr,&text,&specific);
    PyErr_Format(PyExc_RuntimeError,
		 "PETSc error [code %d]:\n%s\n%s\n",
		 (int)ierr,text?text:"",specific?specific:"");
  }
  return NULL;
}

/* ---------------------------------------------------------------- */
