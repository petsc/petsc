
/* This file contains info for the use of PETSc Fortran interface stubs */

#include "petsc.h"

       int     PetscScalarAddressToFortran(Scalar*,Scalar*);
       Scalar* PetscScalarAddressFromFortran(Scalar*,int);
       int     PetscIntAddressToFortran(int*,int*);
       int    *PetscIntAddressFromFortran(int*,int); 
extern void   *PETSC_NULL_Fortran;

#ifdef HAVE_64BITS
extern void *MPIR_ToPointer(int);
extern int MPIR_FromPointer(void*);
extern void MPIR_RmPointer(int);
#else
#define MPIR_ToPointer(a) (a)
#define MPIR_FromPointer(a) (int)(a)
#define MPIR_RmPointer(a)
#endif

/*
    This defines the mappings from Fortran charactor strings 
  to C charactor strings on the Cray T3D.
  */
#if defined(PARCH_t3d)
#include <fortran.h>
#endif
