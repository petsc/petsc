#if !defined(PETSCDMCEED_H)
#define PETSCDMCEED_H

#include <petscdm.h>

#if defined(PETSC_HAVE_LIBCEED)
#include <ceed.h>

#if defined(PETSC_CLANG_STATIC_ANALYZER)
void CHKERRQ_CEED(PetscErrorCode);
#else
#define CHKERRQ_CEED(...)                                                                      \
  do {                                                                                         \
    PetscErrorCode ierr_ceed_ = __VA_ARGS__;                                                   \
    PetscCheck(ierr_ceed_ == CEED_ERROR_SUCCESS, PETSC_COMM_SELF, PETSC_ERR_LIB, "libCEED error: %s",CeedErrorTypes[ierr_ceed_]); \
  } while (0)
#endif /* PETSC_CLANG_STATIC_ANALYZER */

PETSC_EXTERN PetscErrorCode DMGetCeed(DM, Ceed *);

#endif

#endif
