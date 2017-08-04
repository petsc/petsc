#include "data_bucket.h"

/* string helpers */
PetscErrorCode StringInList(const char name[],const PetscInt N,const DataField gfield[],PetscBool *val)
{
  PetscInt       i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  *val = PETSC_FALSE;
  for (i = 0; i < N; ++i) {
    PetscBool flg;
    ierr = PetscStrcmp(name, gfield[i]->name, &flg);CHKERRQ(ierr);
    if (flg) {
      *val = PETSC_TRUE;
      PetscFunctionReturn(0);
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode StringFindInList(const char name[],const PetscInt N,const DataField gfield[],PetscInt *index)
{
  PetscInt       i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  *index = -1;
  for (i = 0; i < N; ++i) {
    PetscBool flg;
    ierr = PetscStrcmp(name, gfield[i]->name, &flg);CHKERRQ(ierr);
    if (flg) {
      *index = i;
      PetscFunctionReturn(0);
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode DataFieldCreate(const char registeration_function[],const char name[],const size_t size,const PetscInt L,DataField *DF)
{
  DataField      df;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscMalloc(sizeof(struct _p_DataField), &df);CHKERRQ(ierr);
  ierr = PetscMemzero(df, sizeof(struct _p_DataField));CHKERRQ(ierr);
  ierr = PetscStrallocpy(registeration_function, &df->registeration_function);CHKERRQ(ierr);
  ierr = PetscStrallocpy(name, &df->name);CHKERRQ(ierr);
  df->atomic_size = size;
  df->L  = L;
  df->bs = 1;
  /* allocate something so we don't have to reallocate */
  ierr = PetscMalloc(size * L, &df->data);CHKERRQ(ierr);
  ierr = PetscMemzero(df->data, size * L);CHKERRQ(ierr);
  *DF = df;
  PetscFunctionReturn(0);
}

PetscErrorCode DataFieldDestroy(DataField *DF)
{
  DataField      df = *DF;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFree(df->registeration_function);CHKERRQ(ierr);
  ierr = PetscFree(df->name);CHKERRQ(ierr);
  ierr = PetscFree(df->data);CHKERRQ(ierr);
  ierr = PetscFree(df);CHKERRQ(ierr);
  *DF  = NULL;
  PetscFunctionReturn(0);
}

/* data bucket */
PetscErrorCode DataBucketCreate(DataBucket *DB)
{
  DataBucket     db;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscMalloc(sizeof(struct _p_DataBucket), &db);CHKERRQ(ierr);
  ierr = PetscMemzero(db, sizeof(struct _p_DataBucket));CHKERRQ(ierr);

  db->finalised = PETSC_FALSE;
  /* create empty spaces for fields */
  db->L         = -1;
  db->buffer    = 1;
  db->allocated = 1;
  db->nfields   = 0;
  ierr = PetscMalloc1(1, &db->field);CHKERRQ(ierr);
  *DB  = db;
  PetscFunctionReturn(0);
}

PetscErrorCode DataBucketDestroy(DataBucket *DB)
{
  DataBucket     db = *DB;
  PetscInt       f;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* release fields */
  for (f = 0; f < db->nfields; ++f) {
    ierr = DataFieldDestroy(&db->field[f]);CHKERRQ(ierr);
  }
  /* this will catch the initially allocated objects in the event that no fields are registered */
  if (db->field != NULL) {
    ierr = PetscFree(db->field);CHKERRQ(ierr);
  }
  ierr = PetscFree(db);CHKERRQ(ierr);
  *DB = NULL;
  PetscFunctionReturn(0);
}

PetscErrorCode DataBucketQueryForActiveFields(DataBucket db,PetscBool *any_active_fields)
{
  PetscInt f;

  PetscFunctionBegin;
  *any_active_fields = PETSC_FALSE;
  for (f = 0; f < db->nfields; ++f) {
    if (db->field[f]->active) {
      *any_active_fields = PETSC_TRUE;
      PetscFunctionReturn(0);
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode DataBucketRegisterField(
                              DataBucket db,
                              const char registeration_function[],
                              const char field_name[],
                              size_t atomic_size, DataField *_gfield)
{
  PetscBool val;
  DataField fp;
  PetscErrorCode ierr;

  PetscFunctionBegin;
	/* check we haven't finalised the registration of fields */
	/*
   if(db->finalised==PETSC_TRUE) {
   printf("ERROR: DataBucketFinalize() has been called. Cannot register more fields\n");
   ERROR();
   }
   */
  /* check for repeated name */
  ierr = StringInList(field_name, db->nfields, (const DataField*) db->field, &val);CHKERRQ(ierr);
  if (val == PETSC_TRUE) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_USER,"Field %s already exists. Cannot add same field twice",field_name);
  /* create new space for data */
  ierr = PetscRealloc(sizeof(DataField)*(db->nfields+1), &db->field);CHKERRQ(ierr);
  /* add field */
  ierr = DataFieldCreate(registeration_function, field_name, atomic_size, db->allocated, &fp);CHKERRQ(ierr);
  db->field[db->nfields] = fp;
  db->nfields++;
  if (_gfield != NULL) {
    *_gfield = fp;
  }
  PetscFunctionReturn(0);
}

/*
 #define DataBucketRegisterField(db,name,size,k) {\
 char *location;\
 asprintf(&location,"Registered by %s() at line %d within file %s", __FUNCTION__, __LINE__, __FILE__);\
 _DataBucketRegisterField( (db), location, (name), (size), (k) );\
 ierr = PetscFree(location);\
 }
 */

PetscErrorCode DataBucketGetDataFieldByName(DataBucket db,const char name[],DataField *gfield)
{
  PetscInt       idx;
  PetscBool      found;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = StringInList(name,db->nfields,(const DataField*)db->field,&found);CHKERRQ(ierr);
  if (!found) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_USER,"Cannot find DataField with name %s",name);
  ierr = StringFindInList(name,db->nfields,(const DataField*)db->field,&idx);CHKERRQ(ierr);
  *gfield = db->field[idx];
  PetscFunctionReturn(0);
}

PetscErrorCode DataBucketQueryDataFieldByName(DataBucket db,const char name[],PetscBool *found)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  *found = PETSC_FALSE;
  ierr = StringInList(name,db->nfields,(const DataField*)db->field,found);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode DataBucketFinalize(DataBucket db)
{
  PetscFunctionBegin;
  db->finalised = PETSC_TRUE;
  PetscFunctionReturn(0);
}

PetscErrorCode DataFieldGetNumEntries(DataField df,PetscInt *sum)
{
  PetscFunctionBegin;
  *sum = df->L;
  PetscFunctionReturn(0);
}

PetscErrorCode DataFieldSetBlockSize(DataField df,PetscInt blocksize)
{
  PetscFunctionBegin;
  df->bs = blocksize;
  PetscFunctionReturn(0);
}

PetscErrorCode DataFieldSetSize(DataField df,const PetscInt new_L)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (new_L < 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"Cannot set size of DataField to be < 0");
  if (new_L == df->L) PetscFunctionReturn(0);
  if (new_L > df->L) {
    ierr = PetscRealloc(df->atomic_size * (new_L), &df->data);CHKERRQ(ierr);
    /* init new contents */
    ierr = PetscMemzero(( ((char*)df->data)+df->L*df->atomic_size), (new_L-df->L)*df->atomic_size);CHKERRQ(ierr);
  } else {
    /* reallocate pointer list, add +1 in case new_L = 0 */
    ierr = PetscRealloc(df->atomic_size * (new_L+1), &df->data);CHKERRQ(ierr);
  }
  df->L = new_L;
  PetscFunctionReturn(0);
}

PetscErrorCode DataFieldZeroBlock(DataField df,const PetscInt start,const PetscInt end)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (start > end) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_USER,"Cannot zero a block of entries if start(%D) > end(%D)",start,end);
  if (start < 0) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_USER,"Cannot zero a block of entries if start(%D) < 0",start);
  if (end > df->L) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_USER,"Cannot zero a block of entries if end(%D) >= array size(%D)",end,df->L);
  ierr = PetscMemzero((((char*)df->data)+start*df->atomic_size), (end-start)*df->atomic_size);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
 A negative buffer value will simply be ignored and the old buffer value will be used.
 */
PetscErrorCode DataBucketSetSizes(DataBucket db,const PetscInt L,const PetscInt buffer)
{
  PetscInt       current_allocated,new_used,new_unused,new_buffer,new_allocated,f;
  PetscBool      any_active_fields;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (db->finalised == PETSC_FALSE) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"You must call DataBucketFinalize() before DataBucketSetSizes()");
  ierr = DataBucketQueryForActiveFields(db,&any_active_fields);CHKERRQ(ierr);
  if (any_active_fields) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"Cannot safely re-size as at least one DataField is currently being accessed");

  current_allocated = db->allocated;
  new_used   = L;
  new_unused = current_allocated - new_used;
  new_buffer = db->buffer;
  if (buffer >= 0) { /* update the buffer value */
    new_buffer = buffer;
  }
  new_allocated = new_used + new_buffer;
  /* action */
  if (new_allocated > current_allocated) {
    /* increase size to new_used + new_buffer */
    for (f=0; f<db->nfields; f++) {
      ierr = DataFieldSetSize(db->field[f], new_allocated);CHKERRQ(ierr);
    }
    db->L         = new_used;
    db->buffer    = new_buffer;
    db->allocated = new_used + new_buffer;
  } else {
    if (new_unused > 2 * new_buffer) {
      /* shrink array to new_used + new_buffer */
      for (f = 0; f < db->nfields; ++f) {
        ierr = DataFieldSetSize(db->field[f], new_allocated);CHKERRQ(ierr);
      }
      db->L         = new_used;
      db->buffer    = new_buffer;
      db->allocated = new_used + new_buffer;
    } else {
      db->L      = new_used;
      db->buffer = new_buffer;
    }
  }
  /* zero all entries from db->L to db->allocated */
  for (f = 0; f < db->nfields; ++f) {
    DataField field = db->field[f];
    ierr = DataFieldZeroBlock(field, db->L,db->allocated);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode DataBucketSetInitialSizes(DataBucket db,const PetscInt L,const PetscInt buffer)
{
  PetscInt       f;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DataBucketSetSizes(db,L,buffer);CHKERRQ(ierr);
  for (f = 0; f < db->nfields; ++f) {
    DataField field = db->field[f];
    ierr = DataFieldZeroBlock(field,0,db->allocated);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode DataBucketGetSizes(DataBucket db,PetscInt *L,PetscInt *buffer,PetscInt *allocated)
{
  PetscFunctionBegin;
  if (L) {*L = db->L;}
  if (buffer) {*buffer = db->buffer;}
  if (allocated) {*allocated = db->allocated;}
  PetscFunctionReturn(0);
}

PetscErrorCode DataBucketGetGlobalSizes(MPI_Comm comm,DataBucket db,PetscInt *L,PetscInt *buffer,PetscInt *allocated)
{
  PetscInt _L,_buffer,_allocated;
  PetscInt ierr;

  PetscFunctionBegin;
  _L = db->L;
  _buffer = db->buffer;
  _allocated = db->allocated;

  if (L) {         ierr = MPI_Allreduce(&_L,L,1,MPIU_INT,MPI_SUM,comm);CHKERRQ(ierr); }
  if (buffer) {    ierr = MPI_Allreduce(&_buffer,buffer,1,MPIU_INT,MPI_SUM,comm);CHKERRQ(ierr); }
  if (allocated) { ierr = MPI_Allreduce(&_allocated,allocated,1,MPIU_INT,MPI_SUM,comm);CHKERRQ(ierr); }
  PetscFunctionReturn(0);
}

PetscErrorCode DataBucketGetDataFields(DataBucket db,PetscInt *L,DataField *fields[])
{
  PetscFunctionBegin;
  if (L)      {*L      = db->nfields;}
  if (fields) {*fields = db->field;}
  PetscFunctionReturn(0);
}

PetscErrorCode DataFieldGetAccess(const DataField gfield)
{
  PetscFunctionBegin;
  if (gfield->active) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_USER,"Field \"%s\" is already active. You must call DataFieldRestoreAccess()",gfield->name);
  gfield->active = PETSC_TRUE;
  PetscFunctionReturn(0);
}

PetscErrorCode DataFieldAccessPoint(const DataField gfield,const PetscInt pid,void **ctx_p)
{
  PetscFunctionBegin;
  *ctx_p = NULL;
#ifdef DATAFIELD_POINT_ACCESS_GUARD
  /* debug mode */
  /* check point is valid */
  if (pid < 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"index must be >= 0");
  if (pid >= gfield->L) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_USER,"index must be < %D",gfield->L);
  if (gfield->active == PETSC_FALSE) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_USER,"Field \"%s\" is not active. You must call DataFieldGetAccess() before point data can be retrivied",gfield->name);
#endif
  *ctx_p = __DATATFIELD_point_access(gfield->data,pid,gfield->atomic_size);
  PetscFunctionReturn(0);
}

PetscErrorCode DataFieldAccessPointOffset(const DataField gfield,const size_t offset,const PetscInt pid,void **ctx_p)
{
  PetscFunctionBegin;
#ifdef DATAFIELD_POINT_ACCESS_GUARD
  /* debug mode */
  /* check point is valid */
  /* if( offset < 0 ) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"offset must be >= 0");*/
  /* Note compiler realizes this can never happen with an unsigned PetscInt */
  if (offset >= gfield->atomic_size) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_USER,"offset must be < %zu",gfield->atomic_size);
  /* check point is valid */
  if (pid < 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"index must be >= 0");
  if (pid >= gfield->L) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_USER,"index must be < %D",gfield->L);
  if (gfield->active == PETSC_FALSE) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_USER,"Field \"%s\" is not active. You must call DataFieldGetAccess() before point data can be retrivied",gfield->name);
#endif
  *ctx_p = __DATATFIELD_point_access_offset(gfield->data,pid,gfield->atomic_size,offset);
  PetscFunctionReturn(0);
}

PetscErrorCode DataFieldRestoreAccess(DataField gfield)
{
  PetscFunctionBegin;
  if (gfield->active == PETSC_FALSE) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_USER,"Field \"%s\" is not active. You must call DataFieldGetAccess()", gfield->name);
  gfield->active = PETSC_FALSE;
  PetscFunctionReturn(0);
}

PetscErrorCode DataFieldVerifyAccess(const DataField gfield,const size_t size)
{
  PetscFunctionBegin;
#ifdef DATAFIELD_POINT_ACCESS_GUARD
  if (gfield->atomic_size != size) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_USER,"Field \"%s\" must be mapped to %zu bytes, your intended structure is %zu bytes in length.",gfield->name, gfield->atomic_size, size );
#endif
  PetscFunctionReturn(0);
}

PetscErrorCode DataFieldGetAtomicSize(const DataField gfield,size_t *size)
{
  PetscFunctionBegin;
  if (size) {*size = gfield->atomic_size;}
  PetscFunctionReturn(0);
}

PetscErrorCode DataFieldGetEntries(const DataField gfield,void **data)
{
  PetscFunctionBegin;
  if (data) {*data = gfield->data;}
  PetscFunctionReturn(0);
}

PetscErrorCode DataFieldRestoreEntries(const DataField gfield,void **data)
{
  PetscFunctionBegin;
  if (data) {*data = NULL;}
  PetscFunctionReturn(0);
}

/* y = x */
PetscErrorCode DataBucketCopyPoint(const DataBucket xb,const PetscInt pid_x,
                         const DataBucket yb,const PetscInt pid_y)
{
  PetscInt f;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  for (f = 0; f < xb->nfields; ++f) {
    void *dest;
    void *src;

    ierr = DataFieldGetAccess(xb->field[f]);CHKERRQ(ierr);
    if (xb != yb) { ierr = DataFieldGetAccess( yb->field[f]);CHKERRQ(ierr); }
    ierr = DataFieldAccessPoint(xb->field[f],pid_x, &src);CHKERRQ(ierr);
    ierr = DataFieldAccessPoint(yb->field[f],pid_y, &dest);CHKERRQ(ierr);
    ierr = PetscMemcpy(dest, src, xb->field[f]->atomic_size);CHKERRQ(ierr);
    ierr = DataFieldRestoreAccess(xb->field[f]);CHKERRQ(ierr);
    if (xb != yb) {ierr = DataFieldRestoreAccess(yb->field[f]);CHKERRQ(ierr);}
  }
  PetscFunctionReturn(0);
}

PetscErrorCode DataBucketCreateFromSubset(DataBucket DBIn,const PetscInt N,const PetscInt list[],DataBucket *DB)
{
  PetscInt nfields;
  DataField *fields;
  PetscInt f,L,buffer,allocated,p;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DataBucketCreate(DB);CHKERRQ(ierr);
  /* copy contents of DBIn */
  ierr = DataBucketGetDataFields(DBIn,&nfields,&fields);CHKERRQ(ierr);
  ierr = DataBucketGetSizes(DBIn,&L,&buffer,&allocated);CHKERRQ(ierr);
  for (f = 0; f < nfields; ++f) {
    ierr = DataBucketRegisterField(*DB,"DataBucketCreateFromSubset",fields[f]->name,fields[f]->atomic_size,NULL);CHKERRQ(ierr);
  }
  ierr = DataBucketFinalize(*DB);CHKERRQ(ierr);
  ierr = DataBucketSetSizes(*DB,L,buffer);CHKERRQ(ierr);
  /* now copy the desired guys from DBIn => DB */
  for (p = 0; p < N; ++p) {
    ierr = DataBucketCopyPoint(DBIn,list[p], *DB,p);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/* insert into an exisitng location */
PetscErrorCode DataFieldInsertPoint(const DataField field,const PetscInt index,const void *ctx)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
#ifdef DATAFIELD_POINT_ACCESS_GUARD
  /* check point is valid */
  if (index < 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"index must be >= 0");
  if (index >= field->L) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_USER,"index must be < %D",field->L);
#endif
  ierr = PetscMemcpy(__DATATFIELD_point_access(field->data,index,field->atomic_size), ctx, field->atomic_size);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* remove data at index - replace with last point */
PetscErrorCode DataBucketRemovePointAtIndex(const DataBucket db,const PetscInt index)
{
  PetscInt       f;
  PetscBool      any_active_fields;
  PetscErrorCode ierr;

  PetscFunctionBegin;
#ifdef DATAFIELD_POINT_ACCESS_GUARD
  /* check point is valid */
  if (index < 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"index must be >= 0");
  if (index >= db->allocated) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_USER,"index must be < %D",db->L+db->buffer);
#endif
  ierr = DataBucketQueryForActiveFields(db,&any_active_fields);CHKERRQ(ierr);
  if (any_active_fields) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"Cannot safely remove point as at least one DataField is currently being accessed");
  if (index >= db->L) { /* this point is not in the list - no need to error, but I will anyway */
    SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_USER,"You should not be trying to remove point at index=%D since it's < db->L = %D", index, db->L);
  }
  if (index != db->L-1) { /* not last point in list */
    for (f = 0; f < db->nfields; ++f) {
      DataField field = db->field[f];

      /* copy then remove */
      ierr = DataFieldCopyPoint(db->L-1, field, index, field);CHKERRQ(ierr);
      /* DataFieldZeroPoint(field,index); */
    }
  }
  /* decrement size */
  /* this will zero out an crap at the end of the list */
  ierr = DataBucketRemovePoint(db);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* copy x into y */
PetscErrorCode DataFieldCopyPoint(const PetscInt pid_x,const DataField field_x,
                        const PetscInt pid_y,const DataField field_y )
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
#ifdef DATAFIELD_POINT_ACCESS_GUARD
  /* check point is valid */
  if (pid_x < 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"(IN) index must be >= 0");
  if (pid_x >= field_x->L) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_USER,"(IN) index must be < %D",field_x->L);
  if (pid_y < 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"(OUT) index must be >= 0");
  if (pid_y >= field_y->L) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_USER,"(OUT) index must be < %D",field_y->L);
  if( field_y->atomic_size != field_x->atomic_size ) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"atomic size must match");
#endif
  ierr = PetscMemcpy(__DATATFIELD_point_access(field_y->data,pid_y,field_y->atomic_size),
                     __DATATFIELD_point_access(field_x->data,pid_x,field_x->atomic_size),
                     field_y->atomic_size);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


/* zero only the datafield at this point */
PetscErrorCode DataFieldZeroPoint(const DataField field,const PetscInt index)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
#ifdef DATAFIELD_POINT_ACCESS_GUARD
  /* check point is valid */
  if (index < 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"index must be >= 0");
  if (index >= field->L) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_USER,"index must be < %D",field->L);
#endif
  ierr = PetscMemzero(__DATATFIELD_point_access(field->data,index,field->atomic_size), field->atomic_size);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* zero ALL data for this point */
PetscErrorCode DataBucketZeroPoint(const DataBucket db,const PetscInt index)
{
  PetscInt f;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* check point is valid */
  if (index < 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"index must be >= 0");
  if (index >= db->allocated) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_USER,"index must be < %D",db->allocated);
  for (f = 0; f < db->nfields; ++f) {
    DataField field = db->field[f];
    ierr = DataFieldZeroPoint(field,index);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/* increment */
PetscErrorCode DataBucketAddPoint(DataBucket db)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DataBucketSetSizes(db,db->L+1,DATA_BUCKET_BUFFER_DEFAULT);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* decrement */
PetscErrorCode DataBucketRemovePoint(DataBucket db)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DataBucketSetSizes(db,db->L-1,DATA_BUCKET_BUFFER_DEFAULT);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode DataBucketView_stdout(MPI_Comm comm,DataBucket db)
{
  PetscInt f;
  double memory_usage_total,memory_usage_total_local = 0.0;
  PetscErrorCode ierr;
  
  PetscFunctionBegin;
  ierr = PetscPrintf(comm,"DataBucketView: \n");CHKERRQ(ierr);
  ierr = PetscPrintf(comm,"  L                  = %D \n", db->L);CHKERRQ(ierr);
  ierr = PetscPrintf(comm,"  buffer             = %D \n", db->buffer);CHKERRQ(ierr);
  ierr = PetscPrintf(comm,"  allocated          = %D \n", db->allocated);CHKERRQ(ierr);
  ierr = PetscPrintf(comm,"  nfields registered = %D \n", db->nfields);CHKERRQ(ierr);
  
  for (f = 0; f < db->nfields; ++f) {
    double memory_usage_f = (double)(db->field[f]->atomic_size * db->allocated) * 1.0e-6;
    memory_usage_total_local += memory_usage_f;
  }
  ierr = MPI_Allreduce(&memory_usage_total_local,&memory_usage_total,1,MPI_DOUBLE,MPI_SUM,comm);CHKERRQ(ierr);
  
  for (f = 0; f < db->nfields; ++f) {
    double memory_usage_f = (double)(db->field[f]->atomic_size * db->allocated) * 1.0e-6;
    ierr = PetscPrintf(comm,"    [%3D] %15s : Mem. usage       = %1.2e (MB) [rank0]\n", f, db->field[f]->name, memory_usage_f );CHKERRQ(ierr);
    ierr = PetscPrintf(comm,"                            blocksize        = %D \n", db->field[f]->bs);CHKERRQ(ierr);
    if (db->field[f]->bs != 1) {
      ierr = PetscPrintf(comm,"                            atomic size      = %zu [full block, bs=%D]\n", db->field[f]->atomic_size,db->field[f]->bs);CHKERRQ(ierr);
      ierr = PetscPrintf(comm,"                            atomic size/item = %zu \n", db->field[f]->atomic_size/db->field[f]->bs);CHKERRQ(ierr);
    } else {
      ierr = PetscPrintf(comm,"                            atomic size      = %zu \n", db->field[f]->atomic_size);CHKERRQ(ierr);
    }
  }
  ierr = PetscPrintf(comm,"  Total mem. usage                           = %1.2e (MB) (collective)\n", memory_usage_total);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode DataBucketView_SEQ(MPI_Comm comm,DataBucket db,const char filename[],DataBucketViewType type)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  switch (type) {
  case DATABUCKET_VIEW_STDOUT:
    ierr = DataBucketView_stdout(PETSC_COMM_SELF,db);CHKERRQ(ierr);
    break;
  case DATABUCKET_VIEW_ASCII:
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"No support for ascii output");
    break;
  case DATABUCKET_VIEW_BINARY:
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"No support for binary output");
    break;
  case DATABUCKET_VIEW_HDF5:
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"No support for HDF5 output");
    break;
  default: SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Unknown viewer method requested");
  }
  PetscFunctionReturn(0);
}

PetscErrorCode DataBucketView_MPI(MPI_Comm comm,DataBucket db,const char filename[],DataBucketViewType type)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  switch (type) {
  case DATABUCKET_VIEW_STDOUT:
    ierr = DataBucketView_stdout(comm,db);CHKERRQ(ierr);
    break;
  case DATABUCKET_VIEW_ASCII:
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"No support for ascii output");
    break;
  case DATABUCKET_VIEW_BINARY:
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"No support for binary output");
    break;
  case DATABUCKET_VIEW_HDF5:
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"No support for HDF5 output");
    break;
  default: SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Unknown viewer method requested");
  }
  PetscFunctionReturn(0);
}

PetscErrorCode DataBucketView(MPI_Comm comm,DataBucket db,const char filename[],DataBucketViewType type)
{
  PetscMPIInt nproc;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(comm,&nproc);CHKERRQ(ierr);
  if (nproc == 1) {
    ierr = DataBucketView_SEQ(comm,db,filename,type);CHKERRQ(ierr);
  } else {
    ierr = DataBucketView_MPI(comm,db,filename,type);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode DataBucketDuplicateFields(DataBucket dbA,DataBucket *dbB)
{
  DataBucket db2;
  PetscInt f;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DataBucketCreate(&db2);CHKERRQ(ierr);
  /* copy contents from dbA into db2 */
  for (f = 0; f < dbA->nfields; ++f) {
    DataField field;
    size_t    atomic_size;
    char      *name;

    field = dbA->field[f];
    atomic_size = field->atomic_size;
    name        = field->name;
    ierr = DataBucketRegisterField(db2,"DataBucketDuplicateFields",name,atomic_size,NULL);CHKERRQ(ierr);
  }
  ierr = DataBucketFinalize(db2);CHKERRQ(ierr);
  ierr = DataBucketSetInitialSizes(db2,0,1000);CHKERRQ(ierr);
  *dbB = db2;
  PetscFunctionReturn(0);
}

/*
 Insert points from db2 into db1
 db1 <<== db2
 */
PetscErrorCode DataBucketInsertValues(DataBucket db1,DataBucket db2)
{
  PetscInt n_mp_points1,n_mp_points2;
  PetscInt n_mp_points1_new,p;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DataBucketGetSizes(db1,&n_mp_points1,0,0);CHKERRQ(ierr);
  ierr = DataBucketGetSizes(db2,&n_mp_points2,0,0);CHKERRQ(ierr);
  n_mp_points1_new = n_mp_points1 + n_mp_points2;
  ierr = DataBucketSetSizes(db1,n_mp_points1_new,DATA_BUCKET_BUFFER_DEFAULT);CHKERRQ(ierr);
  for (p = 0; p < n_mp_points2; ++p) {
    /* db1 <<== db2 */
    ierr = DataBucketCopyPoint(db2,p, db1,(n_mp_points1 + p));CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/* helpers for parallel send/recv */
PetscErrorCode DataBucketCreatePackedArray(DataBucket db,size_t *bytes,void **buf)
{
  PetscInt       f;
  size_t         sizeof_marker_contents;
  void          *buffer;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  sizeof_marker_contents = 0;
  for (f = 0; f < db->nfields; ++f) {
    DataField df = db->field[f];
    sizeof_marker_contents += df->atomic_size;
  }
  ierr = PetscMalloc(sizeof_marker_contents, &buffer);CHKERRQ(ierr);
  ierr = PetscMemzero(buffer, sizeof_marker_contents);CHKERRQ(ierr);
  if (bytes) {*bytes = sizeof_marker_contents;}
  if (buf)   {*buf   = buffer;}
  PetscFunctionReturn(0);
}

PetscErrorCode DataBucketDestroyPackedArray(DataBucket db,void **buf)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (buf) {
    ierr = PetscFree(*buf);CHKERRQ(ierr);
    *buf = NULL;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode DataBucketFillPackedArray(DataBucket db,const PetscInt index,void *buf)
{
  PetscInt       f;
  void          *data, *data_p;
  size_t         asize, offset;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  offset = 0;
  for (f = 0; f < db->nfields; ++f) {
    DataField df = db->field[f];

    asize = df->atomic_size;
    data = (void*)( df->data );
    data_p = (void*)( (char*)data + index*asize );
    ierr = PetscMemcpy((void*)((char*)buf + offset), data_p, asize);CHKERRQ(ierr);
    offset = offset + asize;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode DataBucketInsertPackedArray(DataBucket db,const PetscInt idx,void *data)
{
  PetscInt f;
  void *data_p;
  size_t offset;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  offset = 0;
  for (f = 0; f < db->nfields; ++f) {
    DataField df = db->field[f];

    data_p = (void*)( (char*)data + offset );
    ierr = DataFieldInsertPoint(df, idx, (void*)data_p);CHKERRQ(ierr);
    offset = offset + df->atomic_size;
  }
  PetscFunctionReturn(0);
}
