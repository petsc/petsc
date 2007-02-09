!
!  Include file for Fortran use of the TS (timestepping) package in PETSc
!
#if !defined (__PETSCTS_H)
#define __PETSCTS_H

#define TS PetscFortranAddr
#define TSType character*(80)
#define TSSundialsType PetscEnum
#define TSProblemType PetscEnum 
#define TSSundialsGramSchmitdType PetscEnum

#define TS_EULER 'euler'
#define TS_BEULER 'beuler'
#define TS_PSEUDO 'pseudo'
#define TS_SUNDIALS 'sundials'
#define TS_CRANK_NICHOLSON 'crank-nicholson'
#define TS_RUNGE_KUTTA 'runge-kutta'

#endif

#if !defined (PETSC_AVOID_DECLARATIONS)
!
!  TSProblemType
!
      PetscEnum TS_LINEAR,TS_NONLINEAR
      parameter (TS_LINEAR = 0,TS_NONLINEAR = 1)
!
!  TSSundialsType
!
      PetscEnum SUNDIALS_ADAMS,SUNDIALS_BDF
      parameter (SUNDIALS_ADAMS=1,SUNDIALS_BDF=2)
!
!  TSSundialsGramSchmidtType
!
      PetscEnum SUNDIALS_MODIFIED_GS,SUNDIALS_CLASSICAL_GS
      parameter (SUNDIALS_MODIFIED_GS=1,SUNDIALS_CLASSICAL_GS=2)
#define SUNDIALS_UNMODIFIED_GS SUNDIALS_CLASSICAL_GS
!
!  Some PETSc fortran functions that the user might pass as arguments
!
      external TSDEFAULTCOMPUTEJACOBIAN
      external TSDEFAULTCOMPUTEJACOBIANCOLOR

!PETSC_DEC_ATTRIBUTES(TSDEFAULTCOMPUTEJACOBIAN,'_TSDEFAULTCOMPUTEJACOBIAN')
!PETSC_DEC_ATTRIBUTES(TSDEFAULTCOMPUTEJACOBIANCOLOR,'_TSDEFAULTCOMPUTEJACOBIANCOLOR')
!
!  End of Fortran include file for the TS package in PETSc

#endif
