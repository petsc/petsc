#ifndef PETSC4PY_NUMPY_H
#define PETSC4PY_NUMPY_H

#include "Python.h"
#include "numpy/arrayobject.h"
#include "petsc.h"

#if defined(PETSC_USE_64BIT_INDICES)
#  define NPY_PETSC_INT  NPY_LONGLONG
#else
#  define NPY_PETSC_INT  NPY_INT
#endif

#if   defined(PETSC_USE_REAL_SINGLE)
#  define NPY_PETSC_REAL    NPY_FLOAT
#  define NPY_PETSC_COMPLEX NPY_CFLOAT
#elif defined(PETSC_USE_REAL_DOUBLE)
#  define NPY_PETSC_REAL    NPY_DOUBLE
#  define NPY_PETSC_COMPLEX NPY_CDOUBLE
#elif defined(PETSC_USE_REAL_LONG_DOUBLE)
#  define NPY_PETSC_REAL    NPY_LONGDOUBLE
#  define NPY_PETSC_COMPLEX NPY_CLONGDOUBLE
#else
#  error "unsupported real precision"
#endif

#if   defined(PETSC_USE_SCALAR_COMPLEX)
#  define NPY_PETSC_SCALAR  NPY_PETSC_COMPLEX
#elif defined(PETSC_USE_SCALAR_REAL)
#  define NPY_PETSC_SCALAR  NPY_PETSC_REAL
#else
#  error "unsupported scalar type"
#endif

#endif /* !PETSC4PY_NUMPY_H */
