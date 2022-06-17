#if !defined(__DMSWARM_DATA_BUCKET_H__)
#define __DMSWARM_DATA_BUCKET_H__

#include <petsc/private/dmswarmimpl.h>    /*I   "petscdmswarm.h"   I*/

#define DMSWARM_DATA_BUCKET_BUFFER_DEFAULT -1
#define DMSWARM_DATAFIELD_POINT_ACCESS_GUARD

/* Logging flag */
#define DMSWARM_DATA_BUCKET_LOG

typedef enum { DATABUCKET_VIEW_STDOUT=0, DATABUCKET_VIEW_ASCII, DATABUCKET_VIEW_BINARY, DATABUCKET_VIEW_HDF5 } DMSwarmDataBucketViewType;

struct _p_DMSwarmDataField {
        char          *registration_function;
        PetscInt      L,bs;
        PetscBool     active;
        size_t        atomic_size;
        char          *name; /* what are they called */
        void          *data; /* the data - an array of structs */
  PetscDataType petsc_type;
};

struct _p_DMSwarmDataBucket {
        PetscInt  L;             /* number in use */
        PetscInt  buffer;        /* memory buffer used for re-allocation */
        PetscInt  allocated;     /* number allocated, this will equal datafield->L */
        PetscBool finalised;     /* DEPRECATED */
        PetscInt  nfields;       /* how many fields of this type */
        DMSwarmDataField *field; /* the data */
};

#define DMSWARM_DATAFIELD_point_access(data,index,atomic_size) (void*)((char*)(data) + (index)*(atomic_size))
#define DMSWARM_DATAFIELD_point_access_offset(data,index,atomic_size,offset) (void*)((char*)(data) + (index)*(atomic_size) + (offset))

PETSC_INTERN PetscErrorCode DMSwarmDataFieldStringInList(const char [],const PetscInt ,const DMSwarmDataField [],PetscBool *);
PETSC_INTERN PetscErrorCode DMSwarmDataFieldStringFindInList(const char [],const PetscInt,const DMSwarmDataField [],PetscInt *);

PETSC_INTERN PetscErrorCode DMSwarmDataFieldCreate(const char [],const char [],const size_t,const PetscInt,DMSwarmDataField *);
PETSC_INTERN PetscErrorCode DMSwarmDataFieldDestroy(DMSwarmDataField *);
PETSC_INTERN PetscErrorCode DMSwarmDataBucketCreate(DMSwarmDataBucket *);
PETSC_INTERN PetscErrorCode DMSwarmDataBucketDestroy(DMSwarmDataBucket *);
PETSC_INTERN PetscErrorCode DMSwarmDataBucketQueryForActiveFields(DMSwarmDataBucket,PetscBool *);
PETSC_INTERN PetscErrorCode DMSwarmDataBucketRegisterField(DMSwarmDataBucket,const char [],const char [],size_t,DMSwarmDataField *);

PETSC_INTERN PetscErrorCode DMSwarmDataFieldGetNumEntries(DMSwarmDataField,PetscInt *);
PETSC_INTERN PetscErrorCode DMSwarmDataFieldSetBlockSize(DMSwarmDataField,PetscInt);
PETSC_INTERN PetscErrorCode DMSwarmDataFieldSetSize(DMSwarmDataField,const PetscInt);
PETSC_INTERN PetscErrorCode DMSwarmDataFieldZeroBlock(DMSwarmDataField,const PetscInt,const PetscInt);
PETSC_INTERN PetscErrorCode DMSwarmDataFieldGetAccess(const DMSwarmDataField);
PETSC_INTERN PetscErrorCode DMSwarmDataFieldAccessPoint(const DMSwarmDataField,const PetscInt,void **);
PETSC_INTERN PetscErrorCode DMSwarmDataFieldAccessPointOffset(const DMSwarmDataField,const size_t,const PetscInt,void **);
PETSC_INTERN PetscErrorCode DMSwarmDataFieldRestoreAccess(DMSwarmDataField);
PETSC_INTERN PetscErrorCode DMSwarmDataFieldVerifyAccess(const DMSwarmDataField,const size_t);
PETSC_INTERN PetscErrorCode DMSwarmDataFieldGetAtomicSize(const DMSwarmDataField,size_t *);

PETSC_INTERN PetscErrorCode DMSwarmDataFieldGetEntries(const DMSwarmDataField,void **);
PETSC_INTERN PetscErrorCode DMSwarmDataFieldRestoreEntries(const DMSwarmDataField,void **);

PETSC_INTERN PetscErrorCode DMSwarmDataFieldInsertPoint(const DMSwarmDataField,const PetscInt,const void *);
PETSC_INTERN PetscErrorCode DMSwarmDataFieldCopyPoint(const PetscInt,const DMSwarmDataField,const PetscInt,const DMSwarmDataField);
PETSC_INTERN PetscErrorCode DMSwarmDataFieldZeroPoint(const DMSwarmDataField,const PetscInt);

PETSC_INTERN PetscErrorCode DMSwarmDataBucketGetDMSwarmDataFieldByName(DMSwarmDataBucket,const char [],DMSwarmDataField *);
PETSC_INTERN PetscErrorCode DMSwarmDataBucketGetDMSwarmDataFieldIdByName(DMSwarmDataBucket,const char [],PetscInt *);
PETSC_INTERN PetscErrorCode DMSwarmDataBucketQueryDMSwarmDataFieldByName(DMSwarmDataBucket,const char [],PetscBool *);
PETSC_INTERN PetscErrorCode DMSwarmDataBucketFinalize(DMSwarmDataBucket);
PETSC_INTERN PetscErrorCode DMSwarmDataBucketSetInitialSizes(DMSwarmDataBucket,const PetscInt,const PetscInt);
PETSC_INTERN PetscErrorCode DMSwarmDataBucketSetSizes(DMSwarmDataBucket,const PetscInt,const PetscInt);
PETSC_INTERN PetscErrorCode DMSwarmDataBucketGetSizes(DMSwarmDataBucket,PetscInt *,PetscInt *,PetscInt *);
PETSC_INTERN PetscErrorCode DMSwarmDataBucketGetGlobalSizes(MPI_Comm,DMSwarmDataBucket,PetscInt *,PetscInt *,PetscInt *);
PETSC_INTERN PetscErrorCode DMSwarmDataBucketGetDMSwarmDataFields(DMSwarmDataBucket,PetscInt *,DMSwarmDataField *[]);

PETSC_INTERN PetscErrorCode DMSwarmDataBucketCopyPoint(const DMSwarmDataBucket,const PetscInt,const DMSwarmDataBucket,const PetscInt);
PETSC_INTERN PetscErrorCode DMSwarmDataBucketCreateFromSubset(DMSwarmDataBucket,const PetscInt,const PetscInt [],DMSwarmDataBucket *);
PETSC_INTERN PetscErrorCode DMSwarmDataBucketZeroPoint(const DMSwarmDataBucket,const PetscInt);

PETSC_INTERN PetscErrorCode DMSwarmDataBucketView(MPI_Comm,DMSwarmDataBucket,const char [],DMSwarmDataBucketViewType);

PETSC_INTERN PetscErrorCode DMSwarmDataBucketAddPoint(DMSwarmDataBucket);
PETSC_INTERN PetscErrorCode DMSwarmDataBucketRemovePoint(DMSwarmDataBucket);
PETSC_INTERN PetscErrorCode DMSwarmDataBucketRemovePointAtIndex(const DMSwarmDataBucket,const PetscInt);

PETSC_INTERN PetscErrorCode DMSwarmDataBucketDuplicateFields(DMSwarmDataBucket,DMSwarmDataBucket *);
PETSC_INTERN PetscErrorCode DMSwarmDataBucketInsertValues(DMSwarmDataBucket,DMSwarmDataBucket);

/* helpers for parallel send/recv */
PETSC_INTERN PetscErrorCode DMSwarmDataBucketCreatePackedArray(DMSwarmDataBucket,size_t *,void **);
PETSC_INTERN PetscErrorCode DMSwarmDataBucketDestroyPackedArray(DMSwarmDataBucket,void **);
PETSC_INTERN PetscErrorCode DMSwarmDataBucketFillPackedArray(DMSwarmDataBucket,const PetscInt,void *);
PETSC_INTERN PetscErrorCode DMSwarmDataBucketInsertPackedArray(DMSwarmDataBucket,const PetscInt,void *);

#endif
