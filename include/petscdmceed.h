#if !defined(PETSCDMCEED_H)
#define PETSCDMCEED_H

#include <petscdm.h>

#if defined(PETSC_HAVE_LIBCEED)
#include <ceed.h>

#define CHKERRQ_CEED(ierr)                                                     \
  do {                                                                         \
    PetscErrorCode ierr_ = (ierr);                                             \
    if (ierr_ != CEED_ERROR_SUCCESS)                                           \
      SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_LIB, "libCEED error: %s",             \
              CeedErrorTypes[ierr_]);                                          \
  } while (0)

PETSC_EXTERN PetscErrorCode DMGetCeed(DM, Ceed *);

#endif

#endif
