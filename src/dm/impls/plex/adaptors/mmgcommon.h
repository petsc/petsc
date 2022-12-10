#ifndef PETSC_DM_PLEX_MMGCOMMON_H
#define PETSC_DM_PLEX_MMGCOMMON_H

#include <petsc/private/dmpleximpl.h>

#define PetscCallMMG_Private(ret, ...) \
  do { \
    PetscStackPushExternal(PetscStringize(__VA_ARGS__)); \
    int PETSC_UNUSED mmg_ierr_ = __VA_ARGS__; \
    PetscStackPop; \
    /* PetscCheck(mmg_ierr_ == (ret),PETSC_COMM_SELF,PETSC_ERR_LIB,"Error calling %s: error code %d",PetscStringize(__VA_ARGS__),mmg_ierr_); */ \
  } while (0)

// MMG sometimes returns 1, sometimes 0 when an error has occurred
#define PetscCallMMG(...)             PetscCallMMG_Private(MMG5_SUCCESS, __VA_ARGS__)
#define PetscCallMMG_NONSTANDARD(...) PetscCallMMG_Private(1, __VA_ARGS__)
#endif // PETSC_DM_PLEX_MMGCOMMON_H
