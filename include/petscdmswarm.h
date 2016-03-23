#if !defined(__PETSCDMSWARM_H)
#define __PETSCDMSWARM_H

#include <petscdm.h>

PETSC_EXTERN PetscErrorCode DMSwarmCreateGlobalVectorFromField(DM dm,const char fieldname[],Vec *vec);
PETSC_EXTERN PetscErrorCode DMSwarmDestroyGlobalVectorFromField(DM dm,const char fieldname[],Vec *vec);

PETSC_EXTERN PetscErrorCode DMSwarmInitializeFieldRegister(DM dm);
PETSC_EXTERN PetscErrorCode DMSwarmFinalizeFieldRegister(DM dm);
PETSC_EXTERN PetscErrorCode DMSwarmSetLocalSizes(DM dm,PetscInt nlocal,PetscInt buffer);
PETSC_EXTERN PetscErrorCode DMSwarmRegisterPetscDatatypeField(DM dm,const char fieldname[],PetscInt blocksize,PetscDataType type);
PETSC_EXTERN PetscErrorCode DMSwarmRegisterUserStructField(DM dm,const char fieldname[],size_t size);
PETSC_EXTERN PetscErrorCode DMSwarmRegisterUserDatatypeField(DM dm,const char fieldname[],size_t size);
PETSC_EXTERN PetscErrorCode DMSwarmGetField(DM dm,const char fieldname[],PetscInt *blocksize,PetscDataType *type,void **data);
PETSC_EXTERN PetscErrorCode DMSwarmRestoreField(DM dm,const char fieldname[],PetscInt *blocksize,PetscDataType *type,void **data);

PETSC_EXTERN PetscErrorCode DMSwarmVectorDefineField(DM dm,const char fieldname[]);

PETSC_EXTERN PetscErrorCode DMSwarmAddPoint(DM dm);
PETSC_EXTERN PetscErrorCode DMSwarmAddNPoints(DM dm,PetscInt npoints);
PETSC_EXTERN PetscErrorCode DMSwarmRemovePoint(DM dm);
PETSC_EXTERN PetscErrorCode DMSwarmRemovePointAtIndex(DM dm,PetscInt idx);

PETSC_EXTERN PetscErrorCode DMSwarmGetLocalSize(DM dm,PetscInt *nlocal);
PETSC_EXTERN PetscErrorCode DMSwarmGetSize(DM dm,PetscInt *n);
PETSC_EXTERN PetscErrorCode DMSwarmMigrate(DM dm,PetscBool remove_sent_points);

PETSC_EXTERN PetscErrorCode DMSwarmGlobalToLocalViewCreate(DM dm,InsertMode mode);
PETSC_EXTERN PetscErrorCode DMSwarmGlobalToLocalViewDestroy(DM dm,InsertMode mode);

#endif

