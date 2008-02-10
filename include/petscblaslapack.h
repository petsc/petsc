/*
  This file dispatches between various header files for blas/lapack distributions.
*/
#if !defined(_BLASLAPACK_H)
#define _BLASLAPACK_H
#include "petsc.h"

#if defined(PETSC_HAVE_FORTRAN_STDCALL)
#include "petscblaslapack_stdcall.h"
#elif defined(PETSC_HAVE_FORTRAN_UNDERSCORE) || defined(PETSC_BLASLAPACK_UNDERSCORE)
#include "petscblaslapack_uscore.h"
#elif defined(PETSC_HAVE_FORTRAN_CAPS)
#include "petscblaslapack_caps.h"
#else
#include "petscblaslapack_c.h"
#endif

#endif
