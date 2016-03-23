
#ifndef __DATA_BUCKET_H__
#define __DATA_BUCKET_H__


#include <petsc.h>


#define DEFAULT -32654789

#define DATAFIELD_POINT_ACCESS_GUARD 

/* Logging flag */
#define PTAT3D_LOG_DATA_BUCKET


typedef enum { DATABUCKET_VIEW_STDOUT=0, DATABUCKET_VIEW_ASCII, DATABUCKET_VIEW_BINARY, DATABUCKET_VIEW_HDF5 } DataBucketViewType;

typedef struct _p_DataField* DataField;
typedef struct _p_DataBucket* DataBucket;


struct _p_DataField {
	char   *registeration_function;
	int    L;
	PetscBool active;
	size_t atomic_size;
	char   *name; /* what are they called */
	void   *data; /* the data - an array of structs */
};

struct _p_DataBucket {
	int L; /* number in use */
	int buffer; /* memory buffer used for re-allocation */
	int allocated;  /* number allocated, this will equal datafield->L */
	PetscBool finalised; /* DEPRECIATED */
	int nfields; /* how many fields of this type */
	DataField *field; /* the data */
};

#define ERROR() {\
printf("ERROR: %s() from line %d in %s !!\n", __FUNCTION__, __LINE__, __FILE__);\
exit(EXIT_FAILURE);\
}

#define __DATATFIELD_point_access(data,index,atomic_size) (void*)((char*)(data) + (index)*(atomic_size))
#define __DATATFIELD_point_access_offset(data,index,atomic_size,offset) (void*)((char*)(data) + (index)*(atomic_size) + (offset))



void StringInList( const char name[], const int N, const DataField gfield[], PetscBool *val );
void StringFindInList( const char name[], const int N, const DataField gfield[], int *index );

void DataFieldCreate( const char registeration_function[], const char name[], const size_t size, const int L, DataField *DF );
void DataFieldDestroy( DataField *DF );
void DataBucketCreate( DataBucket *DB );
void DataBucketDestroy( DataBucket *DB );
void _DataBucketRegisterField(
															DataBucket db,
															const char registeration_function[],
															const char field_name[],
															size_t atomic_size, DataField *_gfield );


#define DataBucketRegisterField(db,name,size,k) {\
  char *location;\
  asprintf(&location,"Registered by %s() at line %d within file %s", __FUNCTION__, __LINE__, __FILE__);\
  _DataBucketRegisterField( (db), location, (name), (size), (k) );\
  free(location);\
}

void DataFieldGetNumEntries(DataField df, int *sum);
void DataFieldSetSize( DataField df, const int new_L );
void DataFieldZeroBlock( DataField df, const int start, const int end );
void DataFieldGetAccess( const DataField gfield );
void DataFieldAccessPoint( const DataField gfield, const int pid, void **ctx_p );
void DataFieldAccessPointOffset( const DataField gfield, const size_t offset, const int pid, void **ctx_p );
void DataFieldRestoreAccess( DataField gfield );
void DataFieldVerifyAccess( const DataField gfield, const size_t size);
void DataFieldGetAtomicSize(const DataField gfield,size_t *size);

void DataFieldGetEntries(const DataField gfield,void **data);
void DataFieldRestoreEntries(const DataField gfield,void **data);

void DataFieldInsertPoint( const DataField field, const int index, const void *ctx );
void DataFieldCopyPoint( const int pid_x, const DataField field_x,
												const int pid_y, const DataField field_y );
void DataFieldZeroPoint( const DataField field, const int index ); 

void DataBucketGetDataFieldByName(DataBucket db,const char name[],DataField *gfield);
void DataBucketQueryDataFieldByName(DataBucket db,const char name[],PetscBool *found);
void DataBucketFinalize(DataBucket db);
void DataBucketSetInitialSizes( DataBucket db, const int L, const int buffer );
void DataBucketSetSizes( DataBucket db, const int L, const int buffer );
void DataBucketGetSizes( DataBucket db, int *L, int *buffer, int *allocated );
void DataBucketGetGlobalSizes(MPI_Comm comm, DataBucket db, long int *L, long int *buffer, long int *allocated );
void DataBucketGetDataFields( DataBucket db, int *L, DataField *fields[] );

void DataBucketCopyPoint( const DataBucket xb, const int pid_x,
												 const DataBucket yb, const int pid_y );
void DataBucketCreateFromSubset( DataBucket DBIn, const int N, const int list[], DataBucket *DB );
void DataBucketZeroPoint( const DataBucket db, const int index );

//void DataBucketLoadFromFile(const char filename[], DataBucketViewType type, DataBucket *db);
void DataBucketLoadFromFile(MPI_Comm comm,const char filename[], DataBucketViewType type, DataBucket *db);
//void DataBucketView(DataBucket db,const char filename[],DataBucketViewType type);
void DataBucketView(MPI_Comm comm,DataBucket db,const char filename[],DataBucketViewType type);

void DataBucketAddPoint( DataBucket db );
void DataBucketRemovePoint( DataBucket db );
void DataBucketRemovePointAtIndex( const DataBucket db, const int index );

void DataBucketDuplicateFields(DataBucket dbA,DataBucket *dbB);
void DataBucketInsertValues(DataBucket db1,DataBucket db2);

/* helpers for parallel send/recv */
void DataBucketCreatePackedArray(DataBucket db,size_t *bytes,void **buf);
void DataBucketDestroyPackedArray(DataBucket db,void **buf);
void DataBucketFillPackedArray(DataBucket db,const int index,void *buf);
void DataBucketInsertPackedArray(DataBucket db,const int idx,void *data);


#endif

