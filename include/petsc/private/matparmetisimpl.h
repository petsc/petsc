#pragma once

#define PetscCallParMETIS(func, ...) \
  do { \
    int _status; \
    PetscStackPushExternal(PetscStringize(func)); \
    _status = func(__VA_ARGS__); \
    PetscStackPop; \
    PetscCheck(_status != METIS_ERROR_INPUT, PETSC_COMM_SELF, PETSC_ERR_LIB, "ParMETIS error due to wrong inputs and/or options for %s", PetscStringize(func)); \
    PetscCheck(_status != METIS_ERROR_MEMORY, PETSC_COMM_SELF, PETSC_ERR_MEM, "ParMETIS error due to insufficient memory in %s", PetscStringize(func)); \
    PetscCheck(_status != METIS_ERROR, PETSC_COMM_SELF, PETSC_ERR_LIB, "ParMETIS general error in %s", PetscStringize(func)); \
  } while (0)
