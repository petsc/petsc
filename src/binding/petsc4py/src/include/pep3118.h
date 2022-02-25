#ifndef PETSC4PY_PEP3118_H
#define PETSC4PY_PEP3118_H

#include "Python.h"
#include "petsc.h"

#if defined(PETSC_USE_64BIT_INDICES)
# define _PyPetsc_FMT_PETSC_INT     "q"
#else
# define _PyPetsc_FMT_PETSC_INT     "i"
#endif

#if   defined(PETSC_USE_REAL_SINGLE)
# define _PyPetsc_FMT_PETSC_REAL    "f"
# define _PyPetsc_FMT_PETSC_COMPLEX "Zf"
#elif defined(PETSC_USE_REAL_DOUBLE)
# define _PyPetsc_FMT_PETSC_REAL    "d"
# define _PyPetsc_FMT_PETSC_COMPLEX "Zd"
#elif defined(PETSC_USE_REAL_LONG_DOUBLE)
# define _PyPetsc_FMT_PETSC_REAL    "g"
# define _PyPetsc_FMT_PETSC_COMPLEX "Zg"
#elif defined(PETSC_USE_REAL___FLOAT128)
# define _PyPetsc_FMT_PETSC_REAL    "g"
# define _PyPetsc_FMT_PETSC_COMPLEX "Zg"
#else
# error "unsupported real precision"
#endif

#if   defined(PETSC_USE_COMPLEX)
# define _PyPetsc_FMT_PETSC_SCALAR  _PyPetsc_FMT_PETSC_COMPLEX
#else
# define _PyPetsc_FMT_PETSC_SCALAR  _PyPetsc_FMT_PETSC_REAL
#endif

static inline
int PyPetscBuffer_FillInfo(Py_buffer *view,
                           void *buf, PetscInt count, char typechar,
                           int readonly, int flags)
{
  if (view == NULL) return 0;
  if (((flags & PyBUF_WRITABLE) == PyBUF_WRITABLE) && (readonly == 1)) {
    PyErr_SetString(PyExc_BufferError, "Object is not writable.");
    return -1;
  }
  view->buf = buf;
  switch (typechar) {
  case 'i': view->itemsize = sizeof(PetscInt);    break;
  case 'r': view->itemsize = sizeof(PetscReal);   break;
  case 's': view->itemsize = sizeof(PetscScalar); break;
  case 'c': view->itemsize = sizeof(PetscReal)*2; break;
  default:  view->itemsize = 1;
  }
  view->len = count*view->itemsize;
  view->readonly = readonly;
  view->format = NULL;
  if ((flags & PyBUF_FORMAT) == PyBUF_FORMAT) {
    switch (typechar) {
    case 'i': view->format = (char *) _PyPetsc_FMT_PETSC_INT;     break;
    case 'r': view->format = (char *) _PyPetsc_FMT_PETSC_REAL;    break;
    case 's': view->format = (char *) _PyPetsc_FMT_PETSC_SCALAR;  break;
    case 'c': view->format = (char *) _PyPetsc_FMT_PETSC_COMPLEX; break;
    default:  view->format = (char *) "B";
    }
  }
  view->ndim = 0;
  view->shape = NULL;
  view->strides = NULL;
  view->suboffsets = NULL;
  view->internal = NULL;
  if ((flags & PyBUF_ND) == PyBUF_ND) {
    view->ndim = 1;
    view->internal = PyMem_Malloc(2*sizeof(Py_ssize_t));
    if (!view->internal) { PyErr_NoMemory(); return -1; }
    view->shape = (Py_ssize_t *) view->internal;
    view->shape[0] = view->len/view->itemsize;
    if ((flags & PyBUF_STRIDES) == PyBUF_STRIDES) {
      view->strides = view->shape + 1;
      view->strides[0] = view->itemsize;
    }
  }
  return 0;
}

static inline
void PyPetscBuffer_Release(Py_buffer *view)
{
  if (view == NULL) return;
  if (view->internal) PyMem_Free(view->internal);
  view->internal = NULL;
}

#undef _PyPetsc_FMT_PETSC_INT
#undef _PyPetsc_FMT_PETSC_REAL
#undef _PyPetsc_FMT_PETSC_SCALAR
#undef _PyPetsc_FMT_PETSC_COMPLEX

#endif/*!PETSC4PY_PEP3118_H*/
