!
!  Include file for Fortran use of the TS (timestepping) package in PETSc
!
#include "petsc/finclude/petscts.h"

      type tTS
        PetscFortranAddr:: v
      end type tTS
      type tTSAdapt
        PetscFortranAddr:: v
      end type tTSAdapt
      type tTSTrajectory
        PetscFortranAddr:: v
      end type tTSTrajectory

      TS, parameter :: PETSC_NULL_TS = tTS(-1)
      TSAdapt, parameter :: PETSC_NULL_TSADAPT = tTSAdapt(-1)
      TSTrajectory, parameter :: PETSC_NULL_TSTrajectory                  &
     &                           = tTSTrajectory(-1)

!
!  Convergence flags
!
      PetscEnum TS_CONVERGED_ITERATING
      PetscEnum TS_CONVERGED_TIME
      PetscEnum TS_CONVERGED_ITS
      PetscEnum TS_DIVERGED_NONLINEAR_SOLVE
      PetscEnum TS_DIVERGED_STEP_REJECTED

      parameter (TS_CONVERGED_ITERATING      = 0)
      parameter (TS_CONVERGED_TIME           = 1)
      parameter (TS_CONVERGED_ITS            = 2)
      parameter (TS_DIVERGED_NONLINEAR_SOLVE = -1)
      parameter (TS_DIVERGED_STEP_REJECTED   = -2)

!
!  Equation type flags
!
      PetscEnum TS_EQ_UNSPECIFIED
      PetscEnum TS_EQ_EXPLICIT
      PetscEnum TS_EQ_ODE_EXPLICIT
      PetscEnum TS_EQ_DAE_SEMI_EXPLICIT_INDEX1
      PetscEnum TS_EQ_DAE_SEMI_EXPLICIT_INDEX2
      PetscEnum TS_EQ_DAE_SEMI_EXPLICIT_INDEX3
      PetscEnum TS_EQ_DAE_SEMI_EXPLICIT_INDEXHI
      PetscEnum TS_EQ_IMPLICIT
      PetscEnum TS_EQ_ODE_IMPLICIT
      PetscEnum TS_EQ_DAE_IMPLICIT_INDEX1
      PetscEnum TS_EQ_DAE_IMPLICIT_INDEX2
      PetscEnum TS_EQ_DAE_IMPLICIT_INDEX3
      PetscEnum TS_EQ_DAE_IMPLICIT_INDEXHI

      parameter (TS_EQ_UNSPECIFIED               = -1)
      parameter (TS_EQ_EXPLICIT                  = 0)
      parameter (TS_EQ_ODE_EXPLICIT              = 1)
      parameter (TS_EQ_DAE_SEMI_EXPLICIT_INDEX1  = 100)
      parameter (TS_EQ_DAE_SEMI_EXPLICIT_INDEX2  = 200)
      parameter (TS_EQ_DAE_SEMI_EXPLICIT_INDEX3  = 300)
      parameter (TS_EQ_DAE_SEMI_EXPLICIT_INDEXHI = 500)
      parameter (TS_EQ_IMPLICIT                  = 1000)
      parameter (TS_EQ_ODE_IMPLICIT              = 1001)
      parameter (TS_EQ_DAE_IMPLICIT_INDEX1       = 1100)
      parameter (TS_EQ_DAE_IMPLICIT_INDEX2       = 1200)
      parameter (TS_EQ_DAE_IMPLICIT_INDEX3       = 1300)
      parameter (TS_EQ_DAE_IMPLICIT_INDEXHI      = 1500)

!
!  TSExactFinalTime
!
      PetscEnum TS_EXACTFINALTIME_UNSPECIFIED
      PetscEnum TS_EXACTFINALTIME_STEPOVER
      PetscEnum TS_EXACTFINALTIME_INTERPOLATE
      PetscEnum TS_EXACTFINALTIME_MATCHSTEP

      parameter (TS_EXACTFINALTIME_UNSPECIFIED = 0)
      parameter (TS_EXACTFINALTIME_STEPOVER    = 1)
      parameter (TS_EXACTFINALTIME_INTERPOLATE = 2)
      parameter (TS_EXACTFINALTIME_MATCHSTEP   = 3)

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
      external TSCOMPUTERHSFUNCTIONLINEAR
      external TSCOMPUTERHSJACOBIANCONSTANT
      external TSCOMPUTEIFUNCTIONLINEAR
      external TSCOMPUTEIJACOBIANCONSTANT

!  End of Fortran include file for the TS package in PETSc

