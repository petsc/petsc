#pragma once

#include <petsc/private/dmpleximpl.h>

#define PetscCallMMG_Private(ret, name, ...) \
  do { \
    PetscCall(PetscFPTrapPush(PETSC_FP_TRAP_OFF)); \
    PetscStackPushExternal(PetscStringize(name)); \
    int PETSC_UNUSED mmg_ierr_ = name(__VA_ARGS__); \
    PetscStackPop; \
    PetscCall(PetscFPTrapPop()); \
    /* PetscCheck(mmg_ierr_ == (ret),PETSC_COMM_SELF,PETSC_ERR_LIB,"Error calling %s: error code %d",PetscStringize(__VA_ARGS__),mmg_ierr_); */ \
  } while (0)

// MMG sometimes returns 1, sometimes 0 when an error has occurred
#define PetscCallMMG(name, ...)             PetscCallMMG_Private(MMG5_SUCCESS, name, __VA_ARGS__)
#define PetscCallMMG_NONSTANDARD(name, ...) PetscCallMMG_Private(1, name, __VA_ARGS__)
