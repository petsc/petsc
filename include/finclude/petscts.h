!
!  Include file for Fortran use of the TS (timestepping) package in PETSc
!
#include "finclude/petsctsdef.h"

!
!  TSProblemType
!
      PetscEnum TS_LINEAR
      PetscEnum TS_NONLINEAR
      parameter (TS_LINEAR = 0,TS_NONLINEAR = 1)
!
!  TSSundialsType
!
      PetscEnum SUNDIALS_ADAMS
      PetscEnum SUNDIALS_BDF
      parameter (SUNDIALS_ADAMS=1,SUNDIALS_BDF=2)
!
!  TSSundialsGramSchmidtType
!
      PetscEnum SUNDIALS_MODIFIED_GS
      PetscEnum SUNDIALS_CLASSICAL_GS
      parameter (SUNDIALS_MODIFIED_GS=1,SUNDIALS_CLASSICAL_GS=2)
#define SUNDIALS_UNMODIFIED_GS SUNDIALS_CLASSICAL_GS
!
!  Some PETSc fortran functions that the user might pass as arguments
!
      external TSDEFAULTCOMPUTEJACOBIAN
      external TSDEFAULTCOMPUTEJACOBIANCOLOR

!  End of Fortran include file for the TS package in PETSc

