!
!  $Id: vec.h,v 1.21 1998/03/25 00:36:55 balay Exp balay $;
!
!  Include file for Fortran use of the Vec package in PETSc
!
#define Vec          PetscFortranAddr
#define VecScatter   PetscFortranAddr 
#define NormType     integer
#define InsertMode   integer
#define ScatterMode  integer 
#define VecType      integer
#define VecOption    integer
!
!  VecType
!     
      integer VECSAME, VECSEQ, VECMPI
      parameter (VECSAME=-1, VECSEQ=0, VECMPI=1)
!
!  Types of vector and matrix norms
!
      integer NORM_1,NORM_2,NORM_FROBENIUS,NORM_INFINITY,NORM_MAX,
     *        NORM_1_AND_2
      parameter(NORM_1=1,NORM_2=2,NORM_FROBENIUS=3,NORM_INFINITY=4,
     *          NORM_MAX=4,NORM_1_AND_2=5)
!
!  Flags for VecSetValues() and MatSetValues()
!
      integer NOT_SET_VALUES,INSERT_VALUES, ADD_VALUES
      parameter (NOT_SET_VALUES=0,INSERT_VALUES=1, ADD_VALUES=2)
!
!  Types of vector scatters
!
      integer SCATTER_FORWARD,SCATTER_REVERSE,SCATTER_FORWARD_LOCAL,
     *        SCATTER_REVERSE_LOCAL,SCATTER_LOCAL
      parameter (SCATTER_FORWARD=0,SCATTER_REVERSE=1,
     *           SCATTER_FORWARD_LOCAL=2,SCATTER_REVERSE_LOCAL=3,
     *           SCATTER_LOCAL=2)
!
!  VecOption
!
      integer VEC_IGNORE_OFF_PROC_ENTRIES
      parameter (VEC_IGNORE_OFF_PROC_ENTRIES=0)
!
!  End of Fortran include file for the Vec package in PETSc

