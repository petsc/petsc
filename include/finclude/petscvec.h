!
!
!  Include file for Fortran use of the Vec package in PETSc
!
#if !defined (__PETSCVEC_H)
#define __PETSCVEC_H

#define Vec PetscFortranAddr
#define VecScatter PetscFortranAddr
#define PetscMap PetscFortranAddr
#define NormType integer
#define InsertMode integer
#define ScatterMode integer 
#define VecOption integer
#define VecType character*(80)
#define VecOperation integer

#define VECSEQ 'seq'
#define VECMPI 'mpi'
#define VECFETI 'feti'
#define VECSHARED 'shared'
#define VECESI 'esi'
#define VECPETSCESI 'petscesi'

#endif

#if !defined (PETSC_AVOID_DECLARATIONS)
!
!
!  Types of vector and matrix norms
!
      integer NORM_1,NORM_2,NORM_FROBENIUS,NORM_INFINITY
      integer NORM_MAX,NORM_1_AND_2

      parameter (NORM_1=1,NORM_2=2,NORM_FROBENIUS=3,NORM_INFINITY=4)
      parameter (NORM_MAX=4,NORM_1_AND_2=5)
!
!  Flags for VecSetValues() and MatSetValues()
!
      integer NOT_SET_VALUES,INSERT_VALUES,ADD_VALUES,MAX_VALUES

      parameter (NOT_SET_VALUES=0,INSERT_VALUES=1,ADD_VALUES=2)
      parameter (MAX_VALUES=3)
!
!  Types of vector scatters
!
      integer SCATTER_FORWARD,SCATTER_REVERSE,SCATTER_FORWARD_LOCAL
      integer SCATTER_REVERSE_LOCAL,SCATTER_LOCAL

      parameter (SCATTER_FORWARD=0,SCATTER_REVERSE=1)
      parameter (SCATTER_FORWARD_LOCAL=2,SCATTER_REVERSE_LOCAL=3)
      parameter (SCATTER_LOCAL=2)
!
!  VecOption
!
      integer VEC_IGNORE_OFF_PROC_ENTRIES

      parameter (VEC_IGNORE_OFF_PROC_ENTRIES=0)
!
!  VecOperation
!
      integer VECOP_VIEW,VECOP_LOADINTOVECTOR

      parameter (VECOP_VIEW=33,VECOP_LOADINTOVECTOR=40)
!
!  End of Fortran include file for the Vec package in PETSc

#endif
