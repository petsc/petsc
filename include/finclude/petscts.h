!
!  $Id: petscts.h,v 1.20 2000/09/25 18:03:45 balay Exp $;
!
!  Include file for Fortran use of the TS (timestepping) package in PETSc
!
#if !defined (__PETSCTS_H)
#define __PETSCTS_H

#define TS PetscFortranAddr
#define TSType character*(80)
#define TSPVodeType integer
#define TSProblemType integer 
#define TSPVodeGramSchmitdType integer

#define TS_EULER 'euler'
#define TS_BEULER 'beuler'
#define TS_PSEUDO 'pseudo'
#define TS_PVODE 'pvode'
#define TS_CRANK_NICHOLSON 'crank-nicholson'

#endif

#if !defined (PETSC_AVOID_DECLARATIONS)
!
!  TSProblemType
!
      integer TS_LINEAR,TS_NONLINEAR
      parameter (TS_LINEAR = 0,TS_NONLINEAR = 1)
!
!  TSPvodeType
!
      integer PVODE_ADAMS,PVODE_BDF
      parameter (PVODE_ADAMS=0,PVODE_BDF=1)
!
!  TSPvodeGramSchmidtType
!
      integer PVODE_MODIFIED_GS,PVODE_CLASSICAL_GS,PVODE_UNMODIFIED_GS

      parameter (PVODE_MODIFIED_GS=0,PVODE_CLASSICAL_GS=1)
      parameter (PVODE_UNMODIFIED_GS=1)
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
