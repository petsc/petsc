!
! Used by petsctsmod.F90 to create Fortran module file
!
#include "petsc/finclude/petscts.h"

      type, extends(tPetscObject) :: tTS
      end type tTS
      TS, parameter :: PETSC_NULL_TS = tTS(0)
#if defined(_WIN32) && defined(PETSC_USE_SHARED_LIBRARIES)
!DEC$ ATTRIBUTES DLLEXPORT::PETSC_NULL_TS
#endif

      type, extends(tPetscObject) :: tTSAdapt
      end type tTSAdapt
      TSAdapt, parameter :: PETSC_NULL_TS_ADAPT = tTSAdapt(0)
#if defined(_WIN32) && defined(PETSC_USE_SHARED_LIBRARIES)
!DEC$ ATTRIBUTES DLLEXPORT::PETSC_NULL_TS_ADAPT
#endif

      type, extends(tPetscObject) :: tTSTrajectory
      end type tTSTrajectory
      TSTrajectory, parameter :: PETSC_NULL_TS_TRAJECTORY = tTSTrajectory(0)
#if defined(_WIN32) && defined(PETSC_USE_SHARED_LIBRARIES)
!DEC$ ATTRIBUTES DLLEXPORT::PETSC_NULL_TS_TRAJECTORY
#endif

      type, extends(tPetscObject) :: tTSGLLEAdapt
      end type tTSGLLEAdapt
      TSGLLEAdapt, parameter :: PETSC_NULL_TS_GLLE_ADAPT = tTSGLLEAdapt(0)
#if defined(_WIN32) && defined(PETSC_USE_SHARED_LIBRARIES)
!DEC$ ATTRIBUTES DLLEXPORT::PETSC_NULL_TS_GLLE_ADAPT
#endif

!
!  Convergence flags
!
      PetscEnum, parameter :: TS_CONVERGED_ITERATING      = 0
      PetscEnum, parameter :: TS_CONVERGED_TIME           = 1
      PetscEnum, parameter :: TS_CONVERGED_ITS            = 2
      PetscEnum, parameter :: TS_DIVERGED_NONLINEAR_SOLVE = -1
      PetscEnum, parameter :: TS_DIVERGED_STEP_REJECTED   = -2
!
!  Equation type flags
!
      PetscEnum, parameter :: TS_EQ_UNSPECIFIED               = -1
      PetscEnum, parameter :: TS_EQ_EXPLICIT                  = 0
      PetscEnum, parameter :: TS_EQ_ODE_EXPLICIT              = 1
      PetscEnum, parameter :: TS_EQ_DAE_SEMI_EXPLICIT_INDEX1  = 100
      PetscEnum, parameter :: TS_EQ_DAE_SEMI_EXPLICIT_INDEX2  = 200
      PetscEnum, parameter :: TS_EQ_DAE_SEMI_EXPLICIT_INDEX3  = 300
      PetscEnum, parameter :: TS_EQ_DAE_SEMI_EXPLICIT_INDEXHI = 500
      PetscEnum, parameter :: TS_EQ_IMPLICIT                  = 1000
      PetscEnum, parameter :: TS_EQ_ODE_IMPLICIT              = 1001
      PetscEnum, parameter :: TS_EQ_DAE_IMPLICIT_INDEX1       = 1100
      PetscEnum, parameter :: TS_EQ_DAE_IMPLICIT_INDEX2       = 1200
      PetscEnum, parameter :: TS_EQ_DAE_IMPLICIT_INDEX3       = 1300
      PetscEnum, parameter :: TS_EQ_DAE_IMPLICIT_INDEXHI      = 1500
!
!  TSExactFinalTime
!
      PetscEnum, parameter :: TS_EXACTFINALTIME_UNSPECIFIED = 0
      PetscEnum, parameter :: TS_EXACTFINALTIME_STEPOVER    = 1
      PetscEnum, parameter :: TS_EXACTFINALTIME_INTERPOLATE = 2
      PetscEnum, parameter :: TS_EXACTFINALTIME_MATCHSTEP   = 3
!
!  TSProblemType
!
      PetscEnum, parameter :: TS_LINEAR    = 0
      PetscEnum, parameter :: TS_NONLINEAR = 1
!
!  TSSundialsType
!
      PetscEnum, parameter :: SUNDIALS_ADAMS = 1
      PetscEnum, parameter :: SUNDIALS_BDF   = 2
!
!  TSSundialsGramSchmidtType
!
      PetscEnum, parameter :: SUNDIALS_MODIFIED_GS  = 1
      PetscEnum, parameter :: SUNDIALS_CLASSICAL_GS = 2
#define SUNDIALS_UNMODIFIED_GS SUNDIALS_CLASSICAL_GS
!
!  Some PETSc fortran functions that the user might pass as arguments
!
      external TSCOMPUTERHSFUNCTIONLINEAR
      external TSCOMPUTERHSJACOBIANCONSTANT
      external TSCOMPUTEIFUNCTIONLINEAR
      external TSCOMPUTEIJACOBIANCONSTANT
