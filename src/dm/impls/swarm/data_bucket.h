#ifndef __DATA_BUCKET_H__
#define __DATA_BUCKET_H__

#include <petsc/private/dmswarmimpl.h>    /*I   "petscdmswarm.h"   I*/

#define DATA_BUCKET_BUFFER_DEFAULT -1
#define DATAFIELD_POINT_ACCESS_GUARD

/* Logging flag */
#define DATA_BUCKET_LOG


typedef enum { DATABUCKET_VIEW_STDOUT=0, DATABUCKET_VIEW_ASCII, DATABUCKET_VIEW_BINARY, DATABUCKET_VIEW_HDF5 } DataBucketViewType;



struct _p_DataField {
	char          *registeration_function;
	PetscInt      L,bs;
	PetscBool     active;
	size_t        atomic_size;
	char          *name; /* what are they called */
	void          *data; /* the data - an array of structs */
  PetscDataType petsc_type;
};

struct _p_DataBucket {
	PetscInt  L; /* number in use */
	PetscInt  buffer; /* memory buffer used for re-allocation */
	PetscInt  allocated;  /* number allocated, this will equal datafield->L */
	PetscBool finalised; /* DEPRECIATED */
	PetscInt  nfields; /* how many fields of this type */
	DataField *field; /* the data */
};

#define __DATATFIELD_point_access(data,index,atomic_size) (void*)((char*)(data) + (index)*(atomic_size))
#define __DATATFIELD_point_access_offset(data,index,atomic_size,offset) (void*)((char*)(data) + (index)*(atomic_size) + (offset))



PetscErrorCode StringInList(const char name[],const PetscInt N,const DataField gfield[],PetscBool *val);
PetscErrorCode StringFindInList(const char name[],const PetscInt N,const DataField gfield[],PetscInt *index);

PetscErrorCode DataFieldCreate(const char registeration_function[],const char name[],const size_t size,const PetscInt L,DataField *DF);
PetscErrorCode DataFieldDestroy(DataField *DF);
PetscErrorCode DataBucketCreate(DataBucket *DB);
PetscErrorCode DataBucketDestroy(DataBucket *DB);
PetscErrorCode DataBucketQueryForActiveFields(DataBucket db,PetscBool *any_active_fields);
PetscErrorCode DataBucketRegisterField(
															DataBucket db,
															const char registeration_function[],
															const char field_name[],
															size_t atomic_size,DataField *_gfield );


PetscErrorCode DataFieldGetNumEntries(DataField df,PetscInt *sum);
PetscErrorCode DataFieldSetBlockSize(DataField df,PetscInt blocksize);
PetscErrorCode DataFieldSetSize(DataField df,const PetscInt new_L);
PetscErrorCode DataFieldZeroBlock(DataField df,const PetscInt start,const PetscInt end);
PetscErrorCode DataFieldGetAccess(const DataField gfield);
PetscErrorCode DataFieldAccessPoint(const DataField gfield,const PetscInt pid,void **ctx_p);
PetscErrorCode DataFieldAccessPointOffset(const DataField gfield,const size_t offset,const PetscInt pid,void **ctx_p);
PetscErrorCode DataFieldRestoreAccess(DataField gfield);
PetscErrorCode DataFieldVerifyAccess(const DataField gfield,const size_t size);
PetscErrorCode DataFieldGetAtomicSize(const DataField gfield,size_t *size);

PetscErrorCode DataFieldGetEntries(const DataField gfield,void **data);
PetscErrorCode DataFieldRestoreEntries(const DataField gfield,void **data);

PetscErrorCode DataFieldInsertPoint(const DataField field,const PetscInt index,const void *ctx);
PetscErrorCode DataFieldCopyPoint(const PetscInt pid_x,const DataField field_x,const PetscInt pid_y,const DataField field_y);
PetscErrorCode DataFieldZeroPoint(const DataField field,const PetscInt index);

PetscErrorCode DataBucketGetDataFieldByName(DataBucket db,const char name[],DataField *gfield);
PetscErrorCode DataBucketQueryDataFieldByName(DataBucket db,const char name[],PetscBool *found);
PetscErrorCode DataBucketFinalize(DataBucket db);
PetscErrorCode DataBucketSetInitialSizes(DataBucket db,const PetscInt L,const PetscInt buffer);
PetscErrorCode DataBucketSetSizes(DataBucket db,const PetscInt L,const PetscInt buffer);
PetscErrorCode DataBucketGetSizes(DataBucket db,PetscInt *L,PetscInt *buffer,PetscInt *allocated);
PetscErrorCode DataBucketGetGlobalSizes(MPI_Comm comm,DataBucket db,PetscInt *L,PetscInt *buffer,PetscInt *allocated);
PetscErrorCode DataBucketGetDataFields(DataBucket db,PetscInt *L,DataField *fields[]);

PetscErrorCode DataBucketCopyPoint(const DataBucket xb,const PetscInt pid_x,const DataBucket yb,const PetscInt pid_y);
PetscErrorCode DataBucketCreateFromSubset(DataBucket DBIn,const PetscInt N,const PetscInt list[],DataBucket *DB);
PetscErrorCode DataBucketZeroPoint(const DataBucket db,const PetscInt index);

PetscErrorCode DataBucketView(MPI_Comm comm,DataBucket db,const char filename[],DataBucketViewType type);

PetscErrorCode DataBucketAddPoint(DataBucket db);
PetscErrorCode DataBucketRemovePoint(DataBucket db);
PetscErrorCode DataBucketRemovePointAtIndex(const DataBucket db,const PetscInt index);

PetscErrorCode DataBucketDuplicateFields(DataBucket dbA,DataBucket *dbB);
PetscErrorCode DataBucketInsertValues(DataBucket db1,DataBucket db2);

/* helpers for parallel send/recv */
PetscErrorCode DataBucketCreatePackedArray(DataBucket db,size_t *bytes,void **buf);
PetscErrorCode DataBucketDestroyPackedArray(DataBucket db,void **buf);
PetscErrorCode DataBucketFillPackedArray(DataBucket db,const PetscInt index,void *buf);
PetscErrorCode DataBucketInsertPackedArray(DataBucket db,const PetscInt idx,void *data);


#endif

