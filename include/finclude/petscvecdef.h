!
!
!  Include file for Fortran use of the Vec package in PETSc
!
#if !defined (__PETSCVECDEF_H)
#define __PETSCVECDEF_H

#include "finclude/petscaodef.h"

#if !defined(PETSC_USE_FORTRAN_DATATYPES)
#define Vec PetscFortranAddr
#define VecScatter PetscFortranAddr
#endif

#define NormType PetscEnum
#define InsertMode PetscEnum
#define ScatterMode PetscEnum 
#define VecOption PetscEnum
#define VecType character*(80)
#define VecOperation PetscEnum

#define VECSEQ 'seq'
#define VECMPI 'mpi'
#define VECFETI 'feti'
#define VECSHARED 'shared'
#define VECESI 'esi'
#define VECPETSCESI 'petscesi'

#endif
