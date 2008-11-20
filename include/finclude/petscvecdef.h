!
!
!  Include file for Fortran use of the Vec package in PETSc
!
#if !defined (__PETSCVECDEF_H)
#define __PETSCVECDEF_H

#if defined(PETSC_USE_FORTRAN_MODULES)
#define VEC_HIDE type(Vec)
#define VECSCATTER_HIDE type(VecScatter)
#define USE_VEC_HIDE use petscvecdef
#else
#define VEC_HIDE Vec
#define VECSCATTER_HIDE VecScatter
#define USE_VEC_HIDE

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
