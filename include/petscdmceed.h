#ifndef PETSCDMCEED_H
#define PETSCDMCEED_H

#include <petscdm.h>

#if defined(PETSC_HAVE_LIBCEED)
  #include <ceed.h>

  #if defined(PETSC_CLANG_STATIC_ANALYZER)
void PetscCallCEED(PetscErrorCode);
  #else
    #define PetscCallCEED(...) \
      do { \
        PetscErrorCode ierr_ceed_ = __VA_ARGS__; \
        PetscCheck(ierr_ceed_ == CEED_ERROR_SUCCESS, PETSC_COMM_SELF, PETSC_ERR_LIB, "libCEED error: %s", CeedErrorTypes[ierr_ceed_]); \
      } while (0)
  #endif /* PETSC_CLANG_STATIC_ANALYZER */
  #define CHKERRQ_CEED(...) PetscCallCEED(__VA_ARGS__)

PETSC_EXTERN PetscErrorCode DMGetCeed(DM, Ceed *);

#endif

#endif
