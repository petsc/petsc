/*
  This file dispatches between various header files for blas/lapack distributions.
*/
#if !defined(_BLASLAPACK_H)
#define _BLASLAPACK_H
#include "petsc.h"

#if defined(PETSC_BLASLAPACK_MKL64_ONLY)
# define PETSC_MISSING_LAPACK_GESVD
# define PETSC_MISSING_LAPACK_GEEV
# define PETSC_MISSING_LAPACK_SYGV
# define PETSC_MISSING_LAPACK_SYGVX
# define PETSC_MISSING_LAPACK_GETRF
# define PETSC_MISSING_LAPACK_POTRF
# define PETSC_MISSING_LAPACK_GETRS
# define PETSC_MISSING_LAPACK_POTRS
#elif defined(PETSC_BLASLAPACK_MKL_ONLY)
# define PETSC_MISSING_LAPACK_GESVD
# define PETSC_MISSING_LAPACK_GEEV
# define PETSC_MISSING_LAPACK_SYGV
# define PETSC_MISSING_LAPACK_SYGVX
#elif defined(PETSC_BLASLAPACK_CRAY_ONLY)
# define PETSC_MISSING_LAPACK_GESVD
#elif defined(PETSC_BLASLAPACK_ESSL_ONLY)
# define PETSC_MISSING_LAPACK_GESVD
# define PETSC_MISSING_LAPACK_GETRF
# define PETSC_MISSING_LAPACK_GETRS
# define PETSC_MISSING_LAPACK_POTRF
# define PETSC_MISSING_LAPACK_POTRS
#endif

#if defined(PETSC_USES_CPTOFCD)
#include "petscblaslapack_cptofcd.h"
#elif defined(PETSC_HAVE_FORTRAN_STDCALL)
#include "petscblaslapack_stdcall.h"
#elif defined(PETSC_HAVE_FORTRAN_UNDERSCORE) || defined(PETSC_BLASLAPACK_F2C)
#include "petscblaslapack_uscore.h"
#elif defined(PETSC_HAVE_FORTRAN_CAPS)
#include "petscblaslapack_caps.h"
#else
#include "petscblaslapack_c.h"
#endif

#endif
