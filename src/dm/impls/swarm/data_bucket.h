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

PETSC_INTERN PetscErrorCode DMSwarmDataFieldStringInList(const char name[],const PetscInt N,const DMSwarmDataField gfield[],PetscBool *val);
PETSC_INTERN PetscErrorCode DMSwarmDataFieldStringFindInList(const char name[],const PetscInt N,const DMSwarmDataField gfield[],PetscInt *index);

PETSC_INTERN PetscErrorCode DMSwarmDataFieldCreate(const char registration_function[],const char name[],const size_t size,const PetscInt L,DMSwarmDataField *DF);
PETSC_INTERN PetscErrorCode DMSwarmDataFieldDestroy(DMSwarmDataField *DF);
PETSC_INTERN PetscErrorCode DMSwarmDataBucketCreate(DMSwarmDataBucket *DB);
PETSC_INTERN PetscErrorCode DMSwarmDataBucketDestroy(DMSwarmDataBucket *DB);
PETSC_INTERN PetscErrorCode DMSwarmDataBucketQueryForActiveFields(DMSwarmDataBucket db,PetscBool *any_active_fields);
PETSC_INTERN PetscErrorCode DMSwarmDataBucketRegisterField(DMSwarmDataBucket db,const char registration_function[],const char field_name[],size_t atomic_size,DMSwarmDataField *_gfield);

PETSC_INTERN PetscErrorCode DMSwarmDataFieldGetNumEntries(DMSwarmDataField df,PetscInt *sum);
PETSC_INTERN PetscErrorCode DMSwarmDataFieldSetBlockSize(DMSwarmDataField df,PetscInt blocksize);
PETSC_INTERN PetscErrorCode DMSwarmDataFieldSetSize(DMSwarmDataField df,const PetscInt new_L);
PETSC_INTERN PetscErrorCode DMSwarmDataFieldZeroBlock(DMSwarmDataField df,const PetscInt start,const PetscInt end);
PETSC_INTERN PetscErrorCode DMSwarmDataFieldGetAccess(const DMSwarmDataField gfield);
PETSC_INTERN PetscErrorCode DMSwarmDataFieldAccessPoint(const DMSwarmDataField gfield,const PetscInt pid,void **ctx_p);
PETSC_INTERN PetscErrorCode DMSwarmDataFieldAccessPointOffset(const DMSwarmDataField gfield,const size_t offset,const PetscInt pid,void **ctx_p);
PETSC_INTERN PetscErrorCode DMSwarmDataFieldRestoreAccess(DMSwarmDataField gfield);
PETSC_INTERN PetscErrorCode DMSwarmDataFieldVerifyAccess(const DMSwarmDataField gfield,const size_t size);
PETSC_INTERN PetscErrorCode DMSwarmDataFieldGetAtomicSize(const DMSwarmDataField gfield,size_t *size);

PETSC_INTERN PetscErrorCode DMSwarmDataFieldGetEntries(const DMSwarmDataField gfield,void **data);
PETSC_INTERN PetscErrorCode DMSwarmDataFieldRestoreEntries(const DMSwarmDataField gfield,void **data);

PETSC_INTERN PetscErrorCode DMSwarmDataFieldInsertPoint(const DMSwarmDataField field,const PetscInt index,const void *ctx);
PETSC_INTERN PetscErrorCode DMSwarmDataFieldCopyPoint(const PetscInt pid_x,const DMSwarmDataField field_x,const PetscInt pid_y,const DMSwarmDataField field_y);
PETSC_INTERN PetscErrorCode DMSwarmDataFieldZeroPoint(const DMSwarmDataField field,const PetscInt index);

PETSC_INTERN PetscErrorCode DMSwarmDataBucketGetDMSwarmDataFieldByName(DMSwarmDataBucket db,const char name[],DMSwarmDataField *gfield);
PETSC_INTERN PetscErrorCode DMSwarmDataBucketQueryDMSwarmDataFieldByName(DMSwarmDataBucket db,const char name[],PetscBool *found);
PETSC_INTERN PetscErrorCode DMSwarmDataBucketFinalize(DMSwarmDataBucket db);
PETSC_INTERN PetscErrorCode DMSwarmDataBucketSetInitialSizes(DMSwarmDataBucket db,const PetscInt L,const PetscInt buffer);
PETSC_INTERN PetscErrorCode DMSwarmDataBucketSetSizes(DMSwarmDataBucket db,const PetscInt L,const PetscInt buffer);
PETSC_INTERN PetscErrorCode DMSwarmDataBucketGetSizes(DMSwarmDataBucket db,PetscInt *L,PetscInt *buffer,PetscInt *allocated);
PETSC_INTERN PetscErrorCode DMSwarmDataBucketGetGlobalSizes(MPI_Comm comm,DMSwarmDataBucket db,PetscInt *L,PetscInt *buffer,PetscInt *allocated);
PETSC_INTERN PetscErrorCode DMSwarmDataBucketGetDMSwarmDataFields(DMSwarmDataBucket db,PetscInt *L,DMSwarmDataField *fields[]);

PETSC_INTERN PetscErrorCode DMSwarmDataBucketCopyPoint(const DMSwarmDataBucket xb,const PetscInt pid_x,const DMSwarmDataBucket yb,const PetscInt pid_y);
PETSC_INTERN PetscErrorCode DMSwarmDataBucketCreateFromSubset(DMSwarmDataBucket DBIn,const PetscInt N,const PetscInt list[],DMSwarmDataBucket *DB);
PETSC_INTERN PetscErrorCode DMSwarmDataBucketZeroPoint(const DMSwarmDataBucket db,const PetscInt index);

PETSC_INTERN PetscErrorCode DMSwarmDataBucketView(MPI_Comm comm,DMSwarmDataBucket db,const char filename[],DMSwarmDataBucketViewType type);

PETSC_INTERN PetscErrorCode DMSwarmDataBucketAddPoint(DMSwarmDataBucket db);
PETSC_INTERN PetscErrorCode DMSwarmDataBucketRemovePoint(DMSwarmDataBucket db);
PETSC_INTERN PetscErrorCode DMSwarmDataBucketRemovePointAtIndex(const DMSwarmDataBucket db,const PetscInt index);

PETSC_INTERN PetscErrorCode DMSwarmDataBucketDuplicateFields(DMSwarmDataBucket dbA,DMSwarmDataBucket *dbB);
PETSC_INTERN PetscErrorCode DMSwarmDataBucketInsertValues(DMSwarmDataBucket db1,DMSwarmDataBucket db2);

/* helpers for parallel send/recv */
PETSC_INTERN PetscErrorCode DMSwarmDataBucketCreatePackedArray(DMSwarmDataBucket db,size_t *bytes,void **buf);
PETSC_INTERN PetscErrorCode DMSwarmDataBucketDestroyPackedArray(DMSwarmDataBucket db,void **buf);
PETSC_INTERN PetscErrorCode DMSwarmDataBucketFillPackedArray(DMSwarmDataBucket db,const PetscInt index,void *buf);
PETSC_INTERN PetscErrorCode DMSwarmDataBucketInsertPackedArray(DMSwarmDataBucket db,const PetscInt idx,void *data);

#endif
