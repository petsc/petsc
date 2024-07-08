/*
      Objects which encapsulate mesh adaptation operation
*/
#pragma once

#include <petscdm.h>
#include <petscsnestypes.h>
#include <petscdmadaptortypes.h>

PETSC_EXTERN PetscClassId DMADAPTOR_CLASSID;

/*J
   DMAdaptorType - String with the name of a PETSc DMAdaptor type

   Level: beginner

   Note:
   [](dm_adaptor_table) for a table of available matrix types

.seealso: [](dm_adaptor_table), [](ch_unstructured), `DMAdaptorCreate()`, `DMAdaptor`, `DMAdaptorRegister()`
J*/
typedef const char *DMAdaptorType;
#define DMADAPTORGRADIENT "gradient"
#define DMADAPTORFLUX     "flux"

PETSC_EXTERN PetscFunctionList DMAdaptorList;

PETSC_EXTERN PetscErrorCode DMAdaptorCreate(MPI_Comm, DMAdaptor *);
PETSC_EXTERN PetscErrorCode DMAdaptorSetType(DMAdaptor, DMAdaptorType);
PETSC_EXTERN PetscErrorCode DMAdaptorGetType(DMAdaptor, DMAdaptorType *);
PETSC_EXTERN PetscErrorCode DMAdaptorRegister(const char[], PetscErrorCode (*)(DMAdaptor));
PETSC_EXTERN PetscErrorCode DMAdaptorRegisterAll(void);
PETSC_EXTERN PetscErrorCode DMAdaptorRegisterDestroy(void);
PETSC_EXTERN PetscErrorCode DMAdaptorSetFromOptions(DMAdaptor);
PETSC_EXTERN PetscErrorCode DMAdaptorSetUp(DMAdaptor);
PETSC_EXTERN PetscErrorCode DMAdaptorView(DMAdaptor, PetscViewer);
PETSC_EXTERN PetscErrorCode DMAdaptorDestroy(DMAdaptor *);
PETSC_EXTERN PetscErrorCode DMAdaptorGetSolver(DMAdaptor, SNES *);
PETSC_EXTERN PetscErrorCode DMAdaptorSetSolver(DMAdaptor, SNES);
PETSC_EXTERN PetscErrorCode DMAdaptorGetSequenceLength(DMAdaptor, PetscInt *);
PETSC_EXTERN PetscErrorCode DMAdaptorSetSequenceLength(DMAdaptor, PetscInt);
PETSC_EXTERN PetscErrorCode DMAdaptorGetTransferFunction(DMAdaptor, PetscErrorCode (**)(DMAdaptor, DM, Vec, DM, Vec, void *));
PETSC_EXTERN PetscErrorCode DMAdaptorSetTransferFunction(DMAdaptor, PetscErrorCode (*)(DMAdaptor, DM, Vec, DM, Vec, void *));
PETSC_EXTERN PetscErrorCode DMAdaptorGetMixedSetupFunction(DMAdaptor, PetscErrorCode (**)(DMAdaptor, DM));
PETSC_EXTERN PetscErrorCode DMAdaptorSetMixedSetupFunction(DMAdaptor, PetscErrorCode (*)(DMAdaptor, DM));
PETSC_EXTERN PetscErrorCode DMAdaptorAdapt(DMAdaptor, Vec, DMAdaptationStrategy, DM *, Vec *);
