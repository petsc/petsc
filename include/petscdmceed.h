#pragma once

#include <petscdm.h>

/* MANSEC = DM */

#if defined(PETSC_HAVE_LIBCEED)
  #include <ceed.h>

  #if defined(PETSC_CLANG_STATIC_ANALYZER)
void PetscCallCEED(CeedErrorType);
  #else
    #define PetscCallCEED(...) \
      do { \
        CeedErrorType ierr_ceed_ = __VA_ARGS__; \
        PetscCheck(ierr_ceed_ == CEED_ERROR_SUCCESS, PETSC_COMM_SELF, PETSC_ERR_LIB, "libCEED error: %s", CeedErrorTypes[ierr_ceed_]); \
      } while (0)
  #endif /* PETSC_CLANG_STATIC_ANALYZER */
  #define CHKERRQ_CEED(...) PetscCallCEED(__VA_ARGS__)

PETSC_EXTERN PetscErrorCode DMGetCeed(DM, Ceed *);

PETSC_EXTERN PetscErrorCode VecGetCeedVector(Vec, Ceed, CeedVector *);
PETSC_EXTERN PetscErrorCode VecGetCeedVectorRead(Vec, Ceed, CeedVector *);
PETSC_EXTERN PetscErrorCode VecRestoreCeedVector(Vec, CeedVector *);
PETSC_EXTERN PetscErrorCode VecRestoreCeedVectorRead(Vec, CeedVector *);
PETSC_INTERN PetscErrorCode DMCeedCreate_Internal(DM, IS, PetscBool, CeedQFunctionUser, const char *, DMCeed *);
PETSC_EXTERN PetscErrorCode DMCeedCreate(DM, PetscBool, CeedQFunctionUser, const char *);
PETSC_EXTERN PetscErrorCode DMCeedCreateFVM(DM, PetscBool, CeedQFunctionUser, const char *, CeedQFunctionContext);

struct _PETSc_DMCEED {
  CeedBasis           basis;      // Basis for element function space
  CeedElemRestriction er;         // Map from PETSc local vector to FE element vectors
  CeedElemRestriction erL;        // Map from PETSc local vector to FV left cell vectors
  CeedElemRestriction erR;        // Map from PETSc local vector to FV right cell vectors
  CeedQFunctionUser   func;       // Plex Function for this operator
  char               *funcSource; // Plex Function source as text
  CeedQFunction       qf;         // QFunction expressing the operator action
  CeedOperator        op;         // Operator action for this object
  DMCeed              geom;       // Operator computing geometric data at quadrature points
  CeedElemRestriction erq;        // Map from PETSc local vector to quadrature points
  CeedVector          qd;         // Geometric data at quadrature points used in calculating the qfunction
  DMCeed              info;       // Mesh information at quadrature points
  CeedElemRestriction eri;        // Map from PETSc local vector to quadrature points
  CeedVector          qi;         // Mesh information at quadrature points
};

#else

struct _PETSc_DMCEED {
  PetscInt dummy;
};

#endif

PETSC_EXTERN PetscErrorCode DMCeedComputeGeometry(DM, DMCeed);
PETSC_EXTERN PetscErrorCode DMCeedComputeInfo(DM, DMCeed);
PETSC_EXTERN PetscErrorCode DMCeedDestroy(DMCeed *);
