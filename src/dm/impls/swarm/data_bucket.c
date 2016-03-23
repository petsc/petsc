
#include "data_bucket.h"

/* string helpers */
void StringInList( const char name[], const int N, const DataField gfield[], PetscBool *val )
{
	int i;
	
	*val = PETSC_FALSE;
	for( i=0; i<N; i++ ) {
		if( strcmp( name, gfield[i]->name ) == 0 ) {
			*val = PETSC_TRUE;
			return;
		}
	}
}

void StringFindInList( const char name[], const int N, const DataField gfield[], int *index )
{
	int i;
	
	*index = -1;
	for( i=0; i<N; i++ ) {
		if( strcmp( name, gfield[i]->name ) == 0 ) {
			*index = i;
			return;
		}
	}
}

void DataFieldCreate( const char registeration_function[], const char name[], const size_t size, const int L, DataField *DF )
{
	DataField df;
	
	df = malloc( sizeof(struct _p_DataField) );
	memset( df, 0, sizeof(struct _p_DataField) ); 
	
	
	asprintf( &df->registeration_function, "%s", registeration_function );
	asprintf( &df->name, "%s", name );
	df->atomic_size = size;
	df->L = L;
	
	df->data = malloc( size * L ); /* allocate something so we don't have to reallocate */
	memset( df->data, 0, size * L );
	
	*DF = df;
}

void DataFieldDestroy( DataField *DF )
{
	DataField df = *DF;
	
	free( df->registeration_function );
	free( df->name );
	free( df->data );
	free(df);
	
	*DF = NULL;
}

/* data bucket */
void DataBucketCreate( DataBucket *DB )
{
	DataBucket db;
	
	
	db = malloc( sizeof(struct _p_DataBucket) );
	memset( db, 0, sizeof(struct _p_DataBucket) );

	db->finalised = PETSC_FALSE;
	
	/* create empty spaces for fields */
	db->L         = 0;
	db->buffer    = 1;
	db->allocated = 1;

	db->nfields   = 0;
	db->field     = malloc(sizeof(DataField));
	
	*DB = db;
}

void DataBucketDestroy( DataBucket *DB )
{
	DataBucket db = *DB;
	int f;
	
	/* release fields */
	for( f=0; f<db->nfields; f++ ) {
		DataFieldDestroy(&db->field[f]);	
	}

	/* this will catch the initially allocated objects in the event that no fields are registered */
	if(db->field!=NULL) {
		free(db->field);
	}
	
	free(db);
	
	*DB = NULL;
}

void _DataBucketRegisterField(
						DataBucket db,
						const char registeration_function[],
						const char field_name[],
						size_t atomic_size, DataField *_gfield )
{
	PetscBool val;
	DataField *field,fp;
	
	/* check we haven't finalised the registration of fields */
	/*
	if(db->finalised==PETSC_TRUE) {
		printf("ERROR: DataBucketFinalize() has been called. Cannot register more fields\n");
		ERROR();
	}
	*/
	 
	/* check for repeated name */
	StringInList( field_name, db->nfields, (const DataField*)db->field, &val );
	if(val == PETSC_TRUE ) {
		printf("ERROR: Cannot add same field twice\n");
		ERROR();
	}

	/* create new space for data */
	field = realloc( db->field,     sizeof(DataField)*(db->nfields+1));
	db->field     = field;
	
	/* add field */
	DataFieldCreate( registeration_function, field_name, atomic_size, db->allocated, &fp );
	db->field[ db->nfields ] = fp;
	
	db->nfields++;
	
	if(_gfield!=NULL){
		*_gfield = fp;
	}
}

/*
#define DataBucketRegisterField(db,name,size,k) {\
  char *location;\
  asprintf(&location,"Registered by %s() at line %d within file %s", __FUNCTION__, __LINE__, __FILE__);\
  _DataBucketRegisterField( (db), location, (name), (size), (k) );\
  free(location);\
}
*/

void DataBucketGetDataFieldByName(DataBucket db,const char name[],DataField *gfield)
{
	int idx;
	PetscBool found;
	
	StringInList(name,db->nfields,(const DataField*)db->field,&found);
	if(found==PETSC_FALSE) {
		printf("ERROR: Cannot find DataField with name %s \n", name );
		ERROR();
	}
	StringFindInList(name,db->nfields,(const DataField*)db->field,&idx);
		
	*gfield = db->field[idx];
}

void DataBucketQueryDataFieldByName(DataBucket db,const char name[],PetscBool *found)
{
	*found = PETSC_FALSE;
	StringInList(name,db->nfields,(const DataField*)db->field,found);
}

void DataBucketFinalize(DataBucket db)
{
	db->finalised = PETSC_TRUE;
}

void DataFieldGetNumEntries(DataField df, int *sum)
{
	*sum = df->L;
}

void DataFieldSetSize( DataField df, const int new_L )
{
	void *tmp_data;
	
	if( new_L <= 0 ) {
		printf("ERROR: Cannot set size of DataField to be <= 0 \n");
		ERROR();
	}
	if( new_L == df->L ) return;
	
	if( new_L > df->L ) {
		
		tmp_data = realloc( df->data, df->atomic_size * (new_L) );
		df->data = tmp_data;
		
		/* init new contents */
		memset( ( ((char*)df->data)+df->L*df->atomic_size), 0, (new_L-df->L)*df->atomic_size );
		
	}
	else {
		/* reallocate pointer list, add +1 in case new_L = 0 */
		tmp_data = realloc( df->data, df->atomic_size * (new_L+1) );
		df->data = tmp_data;
	}
	
	df->L = new_L;
}

void DataFieldZeroBlock( DataField df, const int start, const int end )
{
	if( start > end ) {
		printf("ERROR: Cannot zero a block of entries if start(%d) > end(%d) \n",start,end);
		ERROR();
	}
	if( start < 0 ) {
		printf("ERROR: Cannot zero a block of entries if start(%d) < 0 \n",start);
		ERROR();
	}
	if( end > df->L ) {
		printf("ERROR: Cannot zero a block of entries if end(%d) >= array size(%d) \n",end,df->L);
		ERROR();
	}
	
	memset( ( ((char*)df->data)+start*df->atomic_size), 0, (end-start)*df->atomic_size );
}

/*
 A negative buffer value will simply be ignored and the old buffer value will be used.
 */
void DataBucketSetSizes( DataBucket db, const int L, const int buffer )
{
	int current_allocated,new_used,new_unused,new_buffer,new_allocated,f;
	
	
	if( db->finalised == PETSC_FALSE ) {
		printf("ERROR: You must call DataBucketFinalize() before DataBucketSetSizes() \n");
		ERROR();
	}
	 
	current_allocated = db->allocated;
	
	new_used   = L;
	new_unused = current_allocated - new_used;
	new_buffer = db->buffer;
	if( buffer >= 0 ) { /* update the buffer value */
		new_buffer = buffer;
	}
	new_allocated = new_used + new_buffer;
	
	/* action */
	if ( new_allocated > current_allocated ) {
		/* increase size to new_used + new_buffer */
		for( f=0; f<db->nfields; f++ ) {
			DataFieldSetSize( db->field[f], new_allocated );
		}
		
		db->L         = new_used;
		db->buffer    = new_buffer;
		db->allocated = new_used + new_buffer;
	}
	else {
		if( new_unused > 2 * new_buffer ) {
			
			/* shrink array to new_used + new_buffer */
			for( f=0; f<db->nfields; f++ ) {
				DataFieldSetSize( db->field[f], new_allocated );
			}
			
			db->L         = new_used;
			db->buffer    = new_buffer;
			db->allocated = new_used + new_buffer;
		}
		else {
			db->L      = new_used;
			db->buffer = new_buffer;
		}
	}
	
	/* zero all entries from db->L to db->allocated */
	for( f=0; f<db->nfields; f++ ) {
		DataField field = db->field[f];
		DataFieldZeroBlock(field, db->L,db->allocated);
	}
}

void DataBucketSetInitialSizes( DataBucket db, const int L, const int buffer )
{
	int f;
	DataBucketSetSizes(db,L,buffer);
	
	for( f=0; f<db->nfields; f++ ) {
		DataField field = db->field[f];
		DataFieldZeroBlock(field,0,db->allocated);
	}
}

void DataBucketGetSizes( DataBucket db, int *L, int *buffer, int *allocated )
{
	if (L) { *L = db->L; }
	if (buffer) { *buffer = db->buffer; }
	if (allocated) { *allocated = db->allocated; }
}

void DataBucketGetGlobalSizes(MPI_Comm comm, DataBucket db, long int *L, long int *buffer, long int *allocated )
{
	long int _L,_buffer,_allocated;
	int ierr;
	
	_L = (long int)db->L;
	_buffer = (long int)db->buffer;
	_allocated = (long int)db->allocated;
	
	if (L) {         ierr = MPI_Allreduce(&_L,L,1,MPI_LONG,MPI_SUM,comm); }
	if (buffer) {    ierr = MPI_Allreduce(&_buffer,buffer,1,MPI_LONG,MPI_SUM,comm); }
	if (allocated) { ierr = MPI_Allreduce(&_allocated,allocated,1,MPI_LONG,MPI_SUM,comm); }
}

void DataBucketGetDataFields( DataBucket db, int *L, DataField *fields[] )
{
	if(L){      *L      = db->nfields; }
	if(fields){ *fields = db->field; }
}

void DataFieldGetAccess( const DataField gfield )
{
	if(gfield->active==PETSC_TRUE) {
		printf("ERROR: Field \"%s\" is already active. You must call DataFieldRestoreAccess()\n", gfield->name );
		ERROR();
	}
	gfield->active = PETSC_TRUE;
}

void DataFieldAccessPoint( const DataField gfield, const int pid, void **ctx_p )
{
#ifdef DATAFIELD_POINT_ACCESS_GUARD
	/* debug mode */
	/* check point is valid */
	if( pid < 0 ){ printf("ERROR: index must be >= 0\n"); ERROR();  }
	if( pid >= gfield->L ){ printf("ERROR: index must be < %d\n",gfield->L); ERROR(); }

	if(gfield->active==PETSC_FALSE) {
		printf("ERROR: Field \"%s\" is not active. You must call DataFieldGetAccess() before point data can be retrivied\n",gfield->name);
		ERROR();
	}
#endif
	
	//*ctx_p  = (void*)( ((char*)gfield->data) + pid * gfield->atomic_size);
	*ctx_p = __DATATFIELD_point_access(gfield->data,pid,gfield->atomic_size);
}

void DataFieldAccessPointOffset( const DataField gfield, const size_t offset, const int pid, void **ctx_p )
{
#ifdef DATAFIELD_POINT_ACCESS_GUARD
	/* debug mode */
	
	/* check point is valid */
	/* if( offset < 0 ){ printf("ERROR: offset must be >= 0\n"); ERROR();  } *//* Note compiler realizes this can never happen with an unsigned int */
	if( offset >= gfield->atomic_size ){ printf("ERROR: offset must be < %zu\n",gfield->atomic_size); ERROR(); }
	
	/* check point is valid */
	if( pid < 0 ){ printf("ERROR: index must be >= 0\n"); ERROR();  }
	if( pid >= gfield->L ){ printf("ERROR: index must be < %d\n",gfield->L); ERROR(); }

	if(gfield->active==PETSC_FALSE) {
		printf("ERROR: Field \"%s\" is not active. You must call DataFieldGetAccess() before point data can be retrivied\n",gfield->name);
		ERROR();
	}
#endif
	
	*ctx_p = __DATATFIELD_point_access_offset(gfield->data,pid,gfield->atomic_size,offset);
}

void DataFieldRestoreAccess( DataField gfield )
{
	if(gfield->active==PETSC_FALSE) {
		printf("ERROR: Field \"%s\" is not active. You must call DataFieldGetAccess()\n", gfield->name );
		ERROR();
	}
	gfield->active = PETSC_FALSE;
}

void DataFieldVerifyAccess( const DataField gfield, const size_t size)
{
#ifdef DATAFIELD_POINT_ACCESS_GUARD
	if(gfield->atomic_size != size ) {
        printf("ERROR: Field \"%s\" must be mapped to %zu bytes, your intended structure is %zu bytes in length.\n",
               gfield->name, gfield->atomic_size, size );
		ERROR();
	}
#endif
}

void DataFieldGetAtomicSize(const DataField gfield,size_t *size)
{
    if (size) { *size = gfield->atomic_size; }
}

void DataFieldGetEntries(const DataField gfield,void **data)
{
    if (data) {
        *data = gfield->data;
    }
}

void DataFieldRestoreEntries(const DataField gfield,void **data)
{
    if (data) {
        *data = NULL;
    }
}

/* y = x */
void DataBucketCopyPoint( const DataBucket xb, const int pid_x,
												  const DataBucket yb, const int pid_y )
{
	int f;
	for( f=0; f<xb->nfields; f++ ) {
		void *dest;
		void *src;
		
		DataFieldGetAccess( xb->field[f] );
		if (xb!=yb) { DataFieldGetAccess( yb->field[f] ); }
		
		DataFieldAccessPoint( xb->field[f],pid_x, &src );
		DataFieldAccessPoint( yb->field[f],pid_y, &dest );
		
		memcpy( dest, src, xb->field[f]->atomic_size );
		
		DataFieldRestoreAccess( xb->field[f] );
		if (xb!=yb) { DataFieldRestoreAccess( yb->field[f] ); }
	}
	
}

void DataBucketCreateFromSubset( DataBucket DBIn, const int N, const int list[], DataBucket *DB )
{
	int nfields;
	DataField *fields;
	DataBucketCreate(DB);
	int f,L,buffer,allocated,p;
	
	/* copy contents of DBIn */
	DataBucketGetDataFields(DBIn,&nfields,&fields);
	DataBucketGetSizes(DBIn,&L,&buffer,&allocated);
	
	for(f=0;f<nfields;f++) {
		DataBucketRegisterField(*DB,fields[f]->name,fields[f]->atomic_size,NULL);
	}
	DataBucketFinalize(*DB);
	
	DataBucketSetSizes(*DB,L,buffer);
	
	/* now copy the desired guys from DBIn => DB */
	for( p=0; p<N; p++ ) {
		DataBucketCopyPoint(DBIn,list[p], *DB,p);
	}
	
}

// insert into an exisitng location
void DataFieldInsertPoint( const DataField field, const int index, const void *ctx ) 
{

#ifdef DATAFIELD_POINT_ACCESS_GUARD
	/* check point is valid */
	if( index < 0 ){ printf("ERROR: index must be >= 0\n"); ERROR();  }
	if( index >= field->L ){ printf("ERROR: index must be < %d\n",field->L); ERROR(); }
#endif
	
//	memcpy( (void*)((char*)field->data + index*field->atomic_size), ctx, field->atomic_size );
	memcpy( __DATATFIELD_point_access(field->data,index,field->atomic_size), ctx, field->atomic_size );
}

// remove data at index - replace with last point
void DataBucketRemovePointAtIndex( const DataBucket db, const int index )
{
	int f;
	
#ifdef DATAFIELD_POINT_ACCESS_GUARD
	/* check point is valid */
	if( index < 0 ){ printf("ERROR: index must be >= 0\n"); ERROR(); }
	if( index >= db->allocated ){ printf("ERROR: index must be < %d\n",db->L+db->buffer); ERROR(); }
#endif	
	
	if (index >= db->L) { /* this point is not in the list - no need to error, but I will anyway */
		printf("ERROR: You should not be trying to remove point at index=%d since it's < db->L = %d \n", index, db->L );
		ERROR();
	}
	
#if 0	
	if (index == db->L-1) { /* last point in list */
		for( f=0; f<db->nfields; f++ ) {
			DataField field = db->field[f];

			DataFieldZeroPoint(field,index);
		}
	}
	else {
		for( f=0; f<db->nfields; f++ ) {
			DataField field = db->field[f];

			/* copy then remove */
			DataFieldCopyPoint( db->L-1,field, index,field ); 
			
			DataFieldZeroPoint(field,index);
		}
	}
#endif

	if (index != db->L-1) { /* not last point in list */
		for( f=0; f<db->nfields; f++ ) {
			DataField field = db->field[f];
			
			/* copy then remove */
			DataFieldCopyPoint( db->L-1,field, index,field ); 
			
			//DataFieldZeroPoint(field,index);
		}
	}
	
	/* decrement size */
	/* this will zero out an crap at the end of the list */
	DataBucketRemovePoint(db);
	
}

/* copy x into y */
void DataFieldCopyPoint( const int pid_x, const DataField field_x,
												 const int pid_y, const DataField field_y ) 
{

#ifdef DATAFIELD_POINT_ACCESS_GUARD	
	/* check point is valid */
	if( pid_x < 0 ){ printf("ERROR: (IN) index must be >= 0\n"); ERROR(); }
	if( pid_x >= field_x->L ){ printf("ERROR: (IN) index must be < %d\n",field_x->L); ERROR(); }

	if( pid_y < 0 ){ printf("ERROR: (OUT) index must be >= 0\n"); ERROR(); }
	if( pid_y >= field_y->L ){ printf("ERROR: (OUT) index must be < %d\n",field_y->L); ERROR(); }

	if( field_y->atomic_size != field_x->atomic_size ) {
		printf("ERROR: atomic size must match \n"); ERROR();
	}
#endif	
	/*
	memcpy( (void*)((char*)field_y->data + pid_y*field_y->atomic_size), 
					(void*)((char*)field_x->data + pid_x*field_x->atomic_size), 
				  field_x->atomic_size );
	*/
	memcpy(		__DATATFIELD_point_access(field_y->data,pid_y,field_y->atomic_size),
						__DATATFIELD_point_access(field_x->data,pid_x,field_x->atomic_size),
						field_y->atomic_size );
	
}


// zero only the datafield at this point
void DataFieldZeroPoint( const DataField field, const int index ) 
{
#ifdef DATAFIELD_POINT_ACCESS_GUARD
	/* check point is valid */
	if( index < 0 ){ printf("ERROR: index must be >= 0\n"); ERROR(); }
	if( index >= field->L ){ printf("ERROR: index must be < %d\n",field->L); ERROR(); }
#endif	
	
//	memset( (void*)((char*)field->data + index*field->atomic_size), 0, field->atomic_size );
	memset( __DATATFIELD_point_access(field->data,index,field->atomic_size), 0, field->atomic_size );
}

// zero ALL data for this point
void DataBucketZeroPoint( const DataBucket db, const int index ) 
{
	int f;
	
	/* check point is valid */
	if( index < 0 ){ printf("ERROR: index must be >= 0\n"); ERROR(); }
	if( index >= db->allocated ){ printf("ERROR: index must be < %d\n",db->allocated); ERROR(); }
	
	for(f=0;f<db->nfields;f++){
		DataField field = db->field[f];
		
		DataFieldZeroPoint(field,index);
	}	
}

/* increment */
void DataBucketAddPoint( DataBucket db )
{
	DataBucketSetSizes( db, db->L+1, -1 );
}
/* decrement */
void DataBucketRemovePoint( DataBucket db )
{
	DataBucketSetSizes( db, db->L-1, -1 );
}

void _DataFieldViewBinary(DataField field, FILE *fp )
{
	fprintf(fp,"<DataField>\n");
	fprintf(fp,"%d\n", field->L);
	fprintf(fp,"%zu\n",field->atomic_size);
	fprintf(fp,"%s\n", field->registeration_function);
	fprintf(fp,"%s\n", field->name);
	
	fwrite(field->data, field->atomic_size, field->L, fp);
/*
	printf("  ** wrote %zu bytes for DataField \"%s\" \n", field->atomic_size * field->L, field->name );
*/
	fprintf(fp,"\n</DataField>\n");
}

void _DataBucketRegisterFieldFromFile( FILE *fp, DataBucket db )
{
	PetscBool val;
	DataField *field;

	DataField gfield;
	char dummy[100];
	char registeration_function[5000];
	char field_name[5000];
	int L;
	size_t atomic_size,strL;
	
	
	/* check we haven't finalised the registration of fields */
	/*
	if(db->finalised==PETSC_TRUE) {
		printf("ERROR: DataBucketFinalize() has been called. Cannot register more fields\n");
		ERROR();
	}
	*/
	
	
	/* read file contents */
	fgets(dummy,99,fp); //printf("read(header): %s", dummy );
	
	fscanf( fp, "%d\n",&L); //printf("read(L): %d\n", L);
	
	fscanf( fp, "%zu\n",&atomic_size); //printf("read(size): %zu\n",atomic_size);
	
	fgets(registeration_function,4999,fp); //printf("read(reg func): %s", registeration_function );
	strL = strlen(registeration_function);
	if(strL>1){ 
		registeration_function[strL-1] = 0;
	}
	
	fgets(field_name,4999,fp); //printf("read(name): %s", field_name );
	strL = strlen(field_name);
	if(strL>1){ 
		field_name[strL-1] = 0;
	}

#ifdef PTAT3D_LOG_DATA_BUCKET
	printf("  ** read L=%d; atomic_size=%zu; reg_func=\"%s\"; name=\"%s\" \n", L,atomic_size,registeration_function,field_name);
#endif
	
	
	/* check for repeated name */
	StringInList( field_name, db->nfields, (const DataField*)db->field, &val );
	if(val == PETSC_TRUE ) {
		printf("ERROR: Cannot add same field twice\n");
		ERROR();
	}
	
	/* create new space for data */
	field = realloc( db->field,     sizeof(DataField)*(db->nfields+1));
	db->field     = field;
	
	/* add field */
	DataFieldCreate( registeration_function, field_name, atomic_size, L, &gfield );

	/* copy contents of file */
	fread(gfield->data, gfield->atomic_size, gfield->L, fp);
#ifdef PTAT3D_LOG_DATA_BUCKET
	printf("  ** read %zu bytes for DataField \"%s\" \n", gfield->atomic_size * gfield->L, field_name );
#endif	
	/* finish reading meta data */
	fgets(dummy,99,fp); //printf("read(header): %s", dummy );
	fgets(dummy,99,fp); //printf("read(header): %s", dummy );
	
	db->field[ db->nfields ] = gfield;
	
	db->nfields++;
	
}

void _DataBucketViewAscii_HeaderWrite_v00(FILE *fp)
{
	fprintf(fp,"<DataBucketHeader>\n");
	fprintf(fp,"type=DataBucket\n");
	fprintf(fp,"format=ascii\n");
	fprintf(fp,"version=0.0\n");
	fprintf(fp,"options=\n");
	fprintf(fp,"</DataBucketHeader>\n");
}
void _DataBucketViewAscii_HeaderRead_v00(FILE *fp)
{	
	char dummy[100];
	size_t strL;

	// header open
	fgets(dummy,99,fp); //printf("read(header): %s", dummy );

	// type
	fgets(dummy,99,fp); //printf("read(header): %s", dummy );
	strL = strlen(dummy);
	if(strL>1) { dummy[strL-1] = 0; }
	if(strcmp(dummy,"type=DataBucket")!=0) {
		printf("ERROR: Data file doesn't contain a DataBucket type\n");
		ERROR();
	}

	// format
	fgets(dummy,99,fp); //printf("read(header): %s", dummy );

	// version
	fgets(dummy,99,fp); //printf("read(header): %s", dummy );
	strL = strlen(dummy);
	if(strL>1) { dummy[strL-1] = 0; }
	if(strcmp(dummy,"version=0.0")!=0) {
		printf("ERROR: DataBucket file must be parsed with version=0.0 : You tried %s \n", dummy);
		ERROR();
	}
	
	// options
	fgets(dummy,99,fp); //printf("read(header): %s", dummy );
	// header close
	fgets(dummy,99,fp); //printf("read(header): %s", dummy );
}


void _DataBucketLoadFromFileBinary_SEQ(const char filename[], DataBucket *_db)
{
	DataBucket db;
	FILE *fp;
	int L,buffer,f,nfields;
	
	
#ifdef PTAT3D_LOG_DATA_BUCKET
	printf("** DataBucketLoadFromFile **\n");
#endif
	
	/* open file */
	fp = fopen(filename,"rb");
	if(fp==NULL){
		printf("ERROR: Cannot open file with name %s \n", filename);
		ERROR();
	}

	/* read header */
	_DataBucketViewAscii_HeaderRead_v00(fp);
	
	fscanf(fp,"%d\n%d\n%d\n",&L,&buffer,&nfields);
	
	DataBucketCreate(&db);
	
	for( f=0; f<nfields; f++ ) {
		_DataBucketRegisterFieldFromFile(fp,db);
	}
	fclose(fp);
	
	DataBucketFinalize(db);

	
/*	
  DataBucketSetSizes(db,L,buffer);
*/
	db->L = L;
	db->buffer = buffer;
	db->allocated = L + buffer;
	
	*_db = db;
}

void DataBucketLoadFromFile(MPI_Comm comm,const char filename[], DataBucketViewType type, DataBucket *db)
{
	int nproc,rank;
	
	MPI_Comm_size(comm,&nproc);
	MPI_Comm_rank(comm,&rank);
		
#ifdef PTAT3D_LOG_DATA_BUCKET
	printf("** DataBucketLoadFromFile **\n");
#endif
	if(type==DATABUCKET_VIEW_STDOUT) {
		
	} else if(type==DATABUCKET_VIEW_ASCII) {
		printf("ERROR: Cannot be implemented as we don't know the underlying particle data structure\n");
		ERROR();
	} else if(type==DATABUCKET_VIEW_BINARY) {
		if (nproc==1) {
			_DataBucketLoadFromFileBinary_SEQ(filename,db);
		} else {
			char *name;
			
			asprintf(&name,"%s_p%1.5d",filename, rank );
			_DataBucketLoadFromFileBinary_SEQ(name,db);
			free(name);
		}
	} else {
		printf("ERROR: Not implemented\n");
		ERROR();
	}
}


void _DataBucketViewBinary(DataBucket db,const char filename[])
{
	FILE *fp = NULL;
	int f;

	fp = fopen(filename,"wb");
	if(fp==NULL){
		printf("ERROR: Cannot open file with name %s \n", filename);
		ERROR();
	}
	
	/* db header */
	_DataBucketViewAscii_HeaderWrite_v00(fp);
	
	/* meta-data */
	fprintf(fp,"%d\n%d\n%d\n", db->L,db->buffer,db->nfields);

	for( f=0; f<db->nfields; f++ ) {
			/* load datafields */
		_DataFieldViewBinary(db->field[f],fp);
	}
	
	fclose(fp);
}

void DataBucketView_SEQ(DataBucket db,const char filename[],DataBucketViewType type)
{
	switch (type) {
		case DATABUCKET_VIEW_STDOUT:
		{
			int f;
			double memory_usage_total = 0.0;
			
			printf("DataBucketView(SEQ): (\"%s\")\n",filename);
			printf("  L                  = %d \n", db->L );
			printf("  buffer             = %d \n", db->buffer );
			printf("  allocated          = %d \n", db->allocated );
			
			printf("  nfields registered = %d \n", db->nfields );
			for( f=0; f<db->nfields; f++ ) {
				double memory_usage_f = (double)(db->field[f]->atomic_size * db->allocated) * 1.0e-6;
				
				printf("    [%3d]: field name  ==>> %30s : Mem. usage = %1.2e (MB) \n", f, db->field[f]->name, memory_usage_f  );
				memory_usage_total += memory_usage_f;
			}
			printf("  Total mem. usage                                                      = %1.2e (MB) \n", memory_usage_total );
		}
			break;

		case DATABUCKET_VIEW_ASCII:
		{
			printf("ERROR: Cannot be implemented as we don't know the underlying particle data structure\n");
			ERROR();
		}
			break;
			
		case DATABUCKET_VIEW_BINARY:
		{
			_DataBucketViewBinary(db,filename);
		}
			break;
			
		case DATABUCKET_VIEW_HDF5:
		{
			printf("ERROR: Has not been implemented \n");
			ERROR();
		}
			break;

		default:
			printf("ERROR: Unknown method requested \n");
			ERROR();
			break;
	}
}

void DataBucketView_MPI(MPI_Comm comm,DataBucket db,const char filename[],DataBucketViewType type)
{
	switch (type) {
		case DATABUCKET_VIEW_STDOUT:
		{
			int f;
			long int L,buffer,allocated;
			double memory_usage_total,memory_usage_total_local = 0.0;
			int rank;
			int ierr;
			
			ierr = MPI_Comm_rank(comm,&rank);
			
			DataBucketGetGlobalSizes(comm,db,&L,&buffer,&allocated);
			
			for( f=0; f<db->nfields; f++ ) {
				double memory_usage_f = (double)(db->field[f]->atomic_size * db->allocated) * 1.0e-6;
				
				memory_usage_total_local += memory_usage_f;
			}
			MPI_Allreduce(&memory_usage_total_local,&memory_usage_total,1,MPI_DOUBLE,MPI_SUM,comm);

			if (rank==0) {
				printf("DataBucketView(MPI): (\"%s\")\n",filename);
				printf("  L                  = %ld \n", L );
				printf("  buffer (max)       = %ld \n", buffer );
				printf("  allocated          = %ld \n", allocated );
				
				printf("  nfields registered = %d \n", db->nfields );
				for( f=0; f<db->nfields; f++ ) {
					double memory_usage_f = (double)(db->field[f]->atomic_size * db->allocated) * 1.0e-6;
					
					printf("    [%3d]: field name  ==>> %30s : Mem. usage = %1.2e (MB) : rank0\n", f, db->field[f]->name, memory_usage_f  );
				}
				
				printf("  Total mem. usage                                                      = %1.2e (MB) : collective\n", memory_usage_total );
			}			
			
		}
			break;
			
		case DATABUCKET_VIEW_ASCII:
		{
			printf("ERROR: Cannot be implemented as we don't know the underlying particle data structure\n");
			ERROR();
		}
			break;
			
		case DATABUCKET_VIEW_BINARY:
		{
			char *name;
			int rank;
			
			/* create correct extension */
			MPI_Comm_rank(comm,&rank);
			asprintf(&name,"%s_p%1.5d",filename, rank );

			_DataBucketViewBinary(db,name);
			
			free(name);
		}
			break;
			
		case DATABUCKET_VIEW_HDF5:
		{
			printf("ERROR: Has not been implemented \n");
			ERROR();
		}
			break;
			
		default:
			printf("ERROR: Unknown method requested \n");
			ERROR();
			break;
	}
}


void DataBucketView(MPI_Comm comm,DataBucket db,const char filename[],DataBucketViewType type)
{
	int nproc;
	
	MPI_Comm_size(comm,&nproc);
	if (nproc==1) {
		DataBucketView_SEQ(db,filename,type);
	} else {
		DataBucketView_MPI(comm,db,filename,type);
	}
}

void DataBucketDuplicateFields(DataBucket dbA,DataBucket *dbB)
{
	DataBucket db2;
	int f;
	
	DataBucketCreate(&db2);
	
	/* copy contents from dbA into db2 */
	for (f=0; f<dbA->nfields; f++) {
		DataField field;
		size_t    atomic_size;
		char      *name;
		
		field = dbA->field[f];
		
		atomic_size = field->atomic_size;
		name        = field->name;
		
		DataBucketRegisterField(db2,name,atomic_size,NULL);
	}
	DataBucketFinalize(db2);
	DataBucketSetInitialSizes(db2,0,1000);
	
	/* set pointer */
	*dbB = db2;
}

/*
 Insert points from db2 into db1
 db1 <<== db2
 */
void DataBucketInsertValues(DataBucket db1,DataBucket db2)
{
	int n_mp_points1,n_mp_points2;
	int n_mp_points1_new,p;
	
	DataBucketGetSizes(db1,&n_mp_points1,0,0);
	DataBucketGetSizes(db2,&n_mp_points2,0,0);
	
	n_mp_points1_new = n_mp_points1 + n_mp_points2;
	DataBucketSetSizes(db1,n_mp_points1_new,-1);
	
	for (p=0; p<n_mp_points2; p++) {
		// db1 <<== db2 //
		DataBucketCopyPoint( db2,p, db1,(n_mp_points1 + p) );
	}
}

/* helpers for parallel send/recv */
void DataBucketCreatePackedArray(DataBucket db,size_t *bytes,void **buf)
{
    int       f;
    size_t    sizeof_marker_contents;
    void      *buffer;
    
    sizeof_marker_contents = 0;
    for (f=0; f<db->nfields; f++) {
        DataField df = db->field[f];
        
        sizeof_marker_contents += df->atomic_size;
    }
    
    buffer = malloc(sizeof_marker_contents);
    memset(buffer,0,sizeof_marker_contents);
    
    if (bytes) { *bytes = sizeof_marker_contents; }
    if (buf)   { *buf   = buffer; }
}

void DataBucketDestroyPackedArray(DataBucket db,void **buf)
{
    if (buf) {
        free(*buf);
        *buf = NULL;
    }
}

void DataBucketFillPackedArray(DataBucket db,const int index,void *buf)
{
    int    f;
    void   *data,*data_p;
    size_t asize,offset;
    
    offset = 0;
    for (f=0; f<db->nfields; f++) {
        DataField df = db->field[f];
        
        asize = df->atomic_size;
        
        data = (void*)( df->data );
        data_p = (void*)( (char*)data + index*asize );
        
        memcpy( (void*)((char*)buf + offset),  data_p,  asize);
        offset = offset + asize;
    }
}

void DataBucketInsertPackedArray(DataBucket db,const int idx,void *data)
{
    int f;
    void *data_p;
    size_t offset;
    
    offset = 0;
    for (f=0; f<db->nfields; f++) {
        DataField df = db->field[f];
        
        data_p = (void*)( (char*)data + offset );
        
        DataFieldInsertPoint(df, idx, (void*)data_p );
        offset = offset + df->atomic_size;
    }
}
