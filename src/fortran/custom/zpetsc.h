
/* This file contains info for the use of PETSc Fortran interface stubs */

#include "petsc.h"

       int     PetscScalarAddressToFortran(Scalar*,Scalar*);
       Scalar* PetscScalarAddressFromFortran(Scalar*,int);
       int     PetscIntAddressToFortran(int*,int*);
       int    *PetscIntAddressFromFortran(int*,int); 
extern void   *PETSC_NULL_Fortran;
extern char   *PETSC_NULL_CHAR_Fortran;

/*
   On the Cray T3D using the University of Edinburgh MPI release
  the MPI_Comm is an integer. Hence we must not convert them with 
  the MPIR_XX() routines, thus we have seperate conversion routines
  for MPI_Comm objects.
*/
#ifdef HAVE_64BITS
#if defined(__cplusplus)
extern "C" {
#endif
extern void *MPIR_ToPointer(int);
extern int  MPIR_FromPointer(void*);
extern void MPIR_RmPointer(int);
#if defined(__cplusplus)
}
#endif

#if defined(PARCH_t3d) && defined(_T3DMPI_RELEASE_ID)
#define MPIR_ToPointer_Comm(a)        (a)
#define MPIR_FromPointer_Comm(a) (int)(a)
#else
#define MPIR_ToPointer_Comm(a)    MPIR_ToPointer(a)
#define MPIR_FromPointer_Comm(a)  MPIR_FromPointer(a)
#endif

#else
#define MPIR_ToPointer(a)        (a)
#define MPIR_FromPointer(a) (int)(a)
#define MPIR_RmPointer(a)
#define MPIR_ToPointer_Comm(a)        (a)
#define MPIR_FromPointer_Comm(a) (int)(a)
#endif

/*
    This defines the mappings from Fortran charactor strings 
  to C charactor strings on the Cray T3D.
  */
#if defined(PARCH_t3d)
#include <fortran.h>

#define CHAR _fcd
#define FIXCHAR(a,n,b) \
  b = _fcdtocp(a); \
  if (b == PETSC_NULL_CHAR_Fortran) {b = 0;} 
#define FREECHAR(a,b) 

#else

#define CHAR char*
#define FIXCHAR(a,n,b) \
  if (a == PETSC_NULL_CHAR_Fortran) { \
    b = a = 0; \
  } else if (a[n] != 0) { \
    b = (char *) PetscMalloc( (n+1)*sizeof(char)); \
    PetscStrncpy(b,a,n); \
    b[n] = 0; \
  } else b = a;
#define FREECHAR(a,b) if (a != b) PetscFree(b);

#endif

#define FORTRANNULL(a) (((void *) a) == PETSC_NULL_Fortran)

