#include "../src/dm/impls/swarm/data_bucket.h"

/* string helpers */
PetscErrorCode DMSwarmDataFieldStringInList(const char name[],const PetscInt N,const DMSwarmDataField gfield[],PetscBool *val)
{
  PetscInt       i;

  PetscFunctionBegin;
  *val = PETSC_FALSE;
  for (i = 0; i < N; ++i) {
    PetscBool flg;
    CHKERRQ(PetscStrcmp(name, gfield[i]->name, &flg));
    if (flg) {
      *val = PETSC_TRUE;
      PetscFunctionReturn(0);
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode DMSwarmDataFieldStringFindInList(const char name[],const PetscInt N,const DMSwarmDataField gfield[],PetscInt *index)
{
  PetscInt       i;

  PetscFunctionBegin;
  *index = -1;
  for (i = 0; i < N; ++i) {
    PetscBool flg;
    CHKERRQ(PetscStrcmp(name, gfield[i]->name, &flg));
    if (flg) {
      *index = i;
      PetscFunctionReturn(0);
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode DMSwarmDataFieldCreate(const char registration_function[],const char name[],const size_t size,const PetscInt L,DMSwarmDataField *DF)
{
  DMSwarmDataField df;

  PetscFunctionBegin;
  CHKERRQ(PetscNew(&df));
  CHKERRQ(PetscStrallocpy(registration_function, &df->registration_function));
  CHKERRQ(PetscStrallocpy(name, &df->name));
  df->atomic_size = size;
  df->L  = L;
  df->bs = 1;
  /* allocate something so we don't have to reallocate */
  CHKERRQ(PetscMalloc(size * L, &df->data));
  CHKERRQ(PetscMemzero(df->data, size * L));
  *DF = df;
  PetscFunctionReturn(0);
}

PetscErrorCode DMSwarmDataFieldDestroy(DMSwarmDataField *DF)
{
  DMSwarmDataField df = *DF;

  PetscFunctionBegin;
  CHKERRQ(PetscFree(df->registration_function));
  CHKERRQ(PetscFree(df->name));
  CHKERRQ(PetscFree(df->data));
  CHKERRQ(PetscFree(df));
  *DF  = NULL;
  PetscFunctionReturn(0);
}

/* data bucket */
PetscErrorCode DMSwarmDataBucketCreate(DMSwarmDataBucket *DB)
{
  DMSwarmDataBucket db;

  PetscFunctionBegin;
  CHKERRQ(PetscNew(&db));

  db->finalised = PETSC_FALSE;
  /* create empty spaces for fields */
  db->L         = -1;
  db->buffer    = 1;
  db->allocated = 1;
  db->nfields   = 0;
  CHKERRQ(PetscMalloc1(1, &db->field));
  *DB  = db;
  PetscFunctionReturn(0);
}

PetscErrorCode DMSwarmDataBucketDestroy(DMSwarmDataBucket *DB)
{
  DMSwarmDataBucket db = *DB;
  PetscInt          f;

  PetscFunctionBegin;
  /* release fields */
  for (f = 0; f < db->nfields; ++f) {
    CHKERRQ(DMSwarmDataFieldDestroy(&db->field[f]));
  }
  /* this will catch the initially allocated objects in the event that no fields are registered */
  if (db->field != NULL) {
    CHKERRQ(PetscFree(db->field));
  }
  CHKERRQ(PetscFree(db));
  *DB = NULL;
  PetscFunctionReturn(0);
}

PetscErrorCode DMSwarmDataBucketQueryForActiveFields(DMSwarmDataBucket db,PetscBool *any_active_fields)
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

PetscErrorCode DMSwarmDataBucketRegisterField(DMSwarmDataBucket db,const char registration_function[],const char field_name[],size_t atomic_size, DMSwarmDataField *_gfield)
{
  PetscBool        val;
  DMSwarmDataField fp;

  PetscFunctionBegin;
  /* check we haven't finalised the registration of fields */
        /*
   if (db->finalised==PETSC_TRUE) {
   printf("ERROR: DMSwarmDataBucketFinalize() has been called. Cannot register more fields\n");
   ERROR();
   }
  */
  /* check for repeated name */
  CHKERRQ(DMSwarmDataFieldStringInList(field_name, db->nfields, (const DMSwarmDataField*) db->field, &val));
  PetscCheckFalse(val == PETSC_TRUE,PETSC_COMM_SELF,PETSC_ERR_USER,"Field %s already exists. Cannot add same field twice",field_name);
  /* create new space for data */
  CHKERRQ(PetscRealloc(sizeof(DMSwarmDataField)*(db->nfields+1), &db->field));
  /* add field */
  CHKERRQ(DMSwarmDataFieldCreate(registration_function, field_name, atomic_size, db->allocated, &fp));
  db->field[db->nfields] = fp;
  db->nfields++;
  if (_gfield != NULL) {
    *_gfield = fp;
  }
  PetscFunctionReturn(0);
}

/*
 #define DMSwarmDataBucketRegisterField(db,name,size,k) {\
 char *location;\
 asprintf(&location,"Registered by %s() at line %d within file %s", __FUNCTION__, __LINE__, __FILE__);\
 _DMSwarmDataBucketRegisterField( (db), location, (name), (size), (k));\
 ierr = PetscFree(location);\
 }
 */

PetscErrorCode DMSwarmDataBucketGetDMSwarmDataFieldByName(DMSwarmDataBucket db,const char name[],DMSwarmDataField *gfield)
{
  PetscInt       idx;
  PetscBool      found;

  PetscFunctionBegin;
  CHKERRQ(DMSwarmDataFieldStringInList(name,db->nfields,(const DMSwarmDataField*)db->field,&found));
  PetscCheck(found,PETSC_COMM_SELF,PETSC_ERR_USER,"Cannot find DMSwarmDataField with name %s",name);
  CHKERRQ(DMSwarmDataFieldStringFindInList(name,db->nfields,(const DMSwarmDataField*)db->field,&idx));
  *gfield = db->field[idx];
  PetscFunctionReturn(0);
}

PetscErrorCode DMSwarmDataBucketQueryDMSwarmDataFieldByName(DMSwarmDataBucket db,const char name[],PetscBool *found)
{
  PetscFunctionBegin;
  *found = PETSC_FALSE;
  CHKERRQ(DMSwarmDataFieldStringInList(name,db->nfields,(const DMSwarmDataField*)db->field,found));
  PetscFunctionReturn(0);
}

PetscErrorCode DMSwarmDataBucketFinalize(DMSwarmDataBucket db)
{
  PetscFunctionBegin;
  db->finalised = PETSC_TRUE;
  PetscFunctionReturn(0);
}

PetscErrorCode DMSwarmDataFieldGetNumEntries(DMSwarmDataField df,PetscInt *sum)
{
  PetscFunctionBegin;
  *sum = df->L;
  PetscFunctionReturn(0);
}

PetscErrorCode DMSwarmDataFieldSetBlockSize(DMSwarmDataField df,PetscInt blocksize)
{
  PetscFunctionBegin;
  df->bs = blocksize;
  PetscFunctionReturn(0);
}

PetscErrorCode DMSwarmDataFieldSetSize(DMSwarmDataField df,const PetscInt new_L)
{
  PetscFunctionBegin;
  PetscCheckFalse(new_L < 0,PETSC_COMM_SELF,PETSC_ERR_USER,"Cannot set size of DMSwarmDataField to be < 0");
  if (new_L == df->L) PetscFunctionReturn(0);
  if (new_L > df->L) {
    CHKERRQ(PetscRealloc(df->atomic_size * (new_L), &df->data));
    /* init new contents */
    CHKERRQ(PetscMemzero(( ((char*)df->data)+df->L*df->atomic_size), (new_L-df->L)*df->atomic_size));
  } else {
    /* reallocate pointer list, add +1 in case new_L = 0 */
    CHKERRQ(PetscRealloc(df->atomic_size * (new_L+1), &df->data));
  }
  df->L = new_L;
  PetscFunctionReturn(0);
}

PetscErrorCode DMSwarmDataFieldZeroBlock(DMSwarmDataField df,const PetscInt start,const PetscInt end)
{
  PetscFunctionBegin;
  PetscCheckFalse(start > end,PETSC_COMM_SELF,PETSC_ERR_USER,"Cannot zero a block of entries if start(%D) > end(%D)",start,end);
  PetscCheckFalse(start < 0,PETSC_COMM_SELF,PETSC_ERR_USER,"Cannot zero a block of entries if start(%D) < 0",start);
  PetscCheckFalse(end > df->L,PETSC_COMM_SELF,PETSC_ERR_USER,"Cannot zero a block of entries if end(%D) >= array size(%D)",end,df->L);
  CHKERRQ(PetscMemzero((((char*)df->data)+start*df->atomic_size), (end-start)*df->atomic_size));
  PetscFunctionReturn(0);
}

/*
 A negative buffer value will simply be ignored and the old buffer value will be used.
 */
PetscErrorCode DMSwarmDataBucketSetSizes(DMSwarmDataBucket db,const PetscInt L,const PetscInt buffer)
{
  PetscInt       current_allocated,new_used,new_unused,new_buffer,new_allocated,f;
  PetscBool      any_active_fields;

  PetscFunctionBegin;
  PetscCheckFalse(db->finalised == PETSC_FALSE,PETSC_COMM_SELF,PETSC_ERR_USER,"You must call DMSwarmDataBucketFinalize() before DMSwarmDataBucketSetSizes()");
  CHKERRQ(DMSwarmDataBucketQueryForActiveFields(db,&any_active_fields));
  PetscCheck(!any_active_fields,PETSC_COMM_SELF,PETSC_ERR_USER,"Cannot safely re-size as at least one DMSwarmDataField is currently being accessed");

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
      CHKERRQ(DMSwarmDataFieldSetSize(db->field[f], new_allocated));
    }
    db->L         = new_used;
    db->buffer    = new_buffer;
    db->allocated = new_used + new_buffer;
  } else {
    if (new_unused > 2 * new_buffer) {
      /* shrink array to new_used + new_buffer */
      for (f = 0; f < db->nfields; ++f) {
        CHKERRQ(DMSwarmDataFieldSetSize(db->field[f], new_allocated));
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
    DMSwarmDataField field = db->field[f];
    CHKERRQ(DMSwarmDataFieldZeroBlock(field, db->L,db->allocated));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode DMSwarmDataBucketSetInitialSizes(DMSwarmDataBucket db,const PetscInt L,const PetscInt buffer)
{
  PetscInt       f;

  PetscFunctionBegin;
  CHKERRQ(DMSwarmDataBucketSetSizes(db,L,buffer));
  for (f = 0; f < db->nfields; ++f) {
    DMSwarmDataField field = db->field[f];
    CHKERRQ(DMSwarmDataFieldZeroBlock(field,0,db->allocated));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode DMSwarmDataBucketGetSizes(DMSwarmDataBucket db,PetscInt *L,PetscInt *buffer,PetscInt *allocated)
{
  PetscFunctionBegin;
  if (L) {*L = db->L;}
  if (buffer) {*buffer = db->buffer;}
  if (allocated) {*allocated = db->allocated;}
  PetscFunctionReturn(0);
}

PetscErrorCode DMSwarmDataBucketGetGlobalSizes(MPI_Comm comm,DMSwarmDataBucket db,PetscInt *L,PetscInt *buffer,PetscInt *allocated)
{
  PetscFunctionBegin;
  if (L) CHKERRMPI(MPI_Allreduce(&db->L,L,1,MPIU_INT,MPI_SUM,comm));
  if (buffer) CHKERRMPI(MPI_Allreduce(&db->buffer,buffer,1,MPIU_INT,MPI_SUM,comm));
  if (allocated) CHKERRMPI(MPI_Allreduce(&db->allocated,allocated,1,MPIU_INT,MPI_SUM,comm));
  PetscFunctionReturn(0);
}

PetscErrorCode DMSwarmDataBucketGetDMSwarmDataFields(DMSwarmDataBucket db,PetscInt *L,DMSwarmDataField *fields[])
{
  PetscFunctionBegin;
  if (L)      {*L      = db->nfields;}
  if (fields) {*fields = db->field;}
  PetscFunctionReturn(0);
}

PetscErrorCode DMSwarmDataFieldGetAccess(const DMSwarmDataField gfield)
{
  PetscFunctionBegin;
  PetscCheck(!gfield->active,PETSC_COMM_SELF,PETSC_ERR_USER,"Field \"%s\" is already active. You must call DMSwarmDataFieldRestoreAccess()",gfield->name);
  gfield->active = PETSC_TRUE;
  PetscFunctionReturn(0);
}

PetscErrorCode DMSwarmDataFieldAccessPoint(const DMSwarmDataField gfield,const PetscInt pid,void **ctx_p)
{
  PetscFunctionBegin;
  *ctx_p = NULL;
#if defined(DMSWARM_DATAFIELD_POINT_ACCESS_GUARD)
  /* debug mode */
  /* check point is valid */
  PetscCheckFalse(pid < 0,PETSC_COMM_SELF,PETSC_ERR_USER,"index must be >= 0");
  PetscCheckFalse(pid >= gfield->L,PETSC_COMM_SELF,PETSC_ERR_USER,"index must be < %D",gfield->L);
  PetscCheckFalse(gfield->active == PETSC_FALSE,PETSC_COMM_SELF,PETSC_ERR_USER,"Field \"%s\" is not active. You must call DMSwarmDataFieldGetAccess() before point data can be retrivied",gfield->name);
#endif
  *ctx_p = DMSWARM_DATAFIELD_point_access(gfield->data,pid,gfield->atomic_size);
  PetscFunctionReturn(0);
}

PetscErrorCode DMSwarmDataFieldAccessPointOffset(const DMSwarmDataField gfield,const size_t offset,const PetscInt pid,void **ctx_p)
{
  PetscFunctionBegin;
#if defined(DMSWARM_DATAFIELD_POINT_ACCESS_GUARD)
  /* debug mode */
  /* check point is valid */
  /* PetscCheckFalse(offset < 0,PETSC_COMM_SELF,PETSC_ERR_USER,"offset must be >= 0");*/
  /* Note compiler realizes this can never happen with an unsigned PetscInt */
  PetscCheckFalse(offset >= gfield->atomic_size,PETSC_COMM_SELF,PETSC_ERR_USER,"offset must be < %zu",gfield->atomic_size);
  /* check point is valid */
  PetscCheckFalse(pid < 0,PETSC_COMM_SELF,PETSC_ERR_USER,"index must be >= 0");
  PetscCheckFalse(pid >= gfield->L,PETSC_COMM_SELF,PETSC_ERR_USER,"index must be < %D",gfield->L);
  PetscCheckFalse(gfield->active == PETSC_FALSE,PETSC_COMM_SELF,PETSC_ERR_USER,"Field \"%s\" is not active. You must call DMSwarmDataFieldGetAccess() before point data can be retrivied",gfield->name);
#endif
  *ctx_p = DMSWARM_DATAFIELD_point_access_offset(gfield->data,pid,gfield->atomic_size,offset);
  PetscFunctionReturn(0);
}

PetscErrorCode DMSwarmDataFieldRestoreAccess(DMSwarmDataField gfield)
{
  PetscFunctionBegin;
  PetscCheckFalse(gfield->active == PETSC_FALSE,PETSC_COMM_SELF,PETSC_ERR_USER,"Field \"%s\" is not active. You must call DMSwarmDataFieldGetAccess()", gfield->name);
  gfield->active = PETSC_FALSE;
  PetscFunctionReturn(0);
}

PetscErrorCode DMSwarmDataFieldVerifyAccess(const DMSwarmDataField gfield,const size_t size)
{
  PetscFunctionBegin;
#if defined(DMSWARM_DATAFIELD_POINT_ACCESS_GUARD)
  PetscCheckFalse(gfield->atomic_size != size,PETSC_COMM_SELF,PETSC_ERR_USER,"Field \"%s\" must be mapped to %zu bytes, your intended structure is %zu bytes in length.",gfield->name, gfield->atomic_size, size);
#endif
  PetscFunctionReturn(0);
}

PetscErrorCode DMSwarmDataFieldGetAtomicSize(const DMSwarmDataField gfield,size_t *size)
{
  PetscFunctionBegin;
  if (size) {*size = gfield->atomic_size;}
  PetscFunctionReturn(0);
}

PetscErrorCode DMSwarmDataFieldGetEntries(const DMSwarmDataField gfield,void **data)
{
  PetscFunctionBegin;
  if (data) {*data = gfield->data;}
  PetscFunctionReturn(0);
}

PetscErrorCode DMSwarmDataFieldRestoreEntries(const DMSwarmDataField gfield,void **data)
{
  PetscFunctionBegin;
  if (data) {*data = NULL;}
  PetscFunctionReturn(0);
}

/* y = x */
PetscErrorCode DMSwarmDataBucketCopyPoint(const DMSwarmDataBucket xb,const PetscInt pid_x,const DMSwarmDataBucket yb,const PetscInt pid_y)
{
  PetscInt       f;

  PetscFunctionBegin;
  for (f = 0; f < xb->nfields; ++f) {
    void *dest;
    void *src;

    CHKERRQ(DMSwarmDataFieldGetAccess(xb->field[f]));
    if (xb != yb) CHKERRQ(DMSwarmDataFieldGetAccess( yb->field[f]));
    CHKERRQ(DMSwarmDataFieldAccessPoint(xb->field[f],pid_x, &src));
    CHKERRQ(DMSwarmDataFieldAccessPoint(yb->field[f],pid_y, &dest));
    CHKERRQ(PetscMemcpy(dest, src, xb->field[f]->atomic_size));
    CHKERRQ(DMSwarmDataFieldRestoreAccess(xb->field[f]));
    if (xb != yb) CHKERRQ(DMSwarmDataFieldRestoreAccess(yb->field[f]));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode DMSwarmDataBucketCreateFromSubset(DMSwarmDataBucket DBIn,const PetscInt N,const PetscInt list[],DMSwarmDataBucket *DB)
{
  PetscInt         nfields;
  DMSwarmDataField *fields;
  PetscInt         f,L,buffer,allocated,p;

  PetscFunctionBegin;
  CHKERRQ(DMSwarmDataBucketCreate(DB));
  /* copy contents of DBIn */
  CHKERRQ(DMSwarmDataBucketGetDMSwarmDataFields(DBIn,&nfields,&fields));
  CHKERRQ(DMSwarmDataBucketGetSizes(DBIn,&L,&buffer,&allocated));
  for (f = 0; f < nfields; ++f) {
    CHKERRQ(DMSwarmDataBucketRegisterField(*DB,"DMSwarmDataBucketCreateFromSubset",fields[f]->name,fields[f]->atomic_size,NULL));
  }
  CHKERRQ(DMSwarmDataBucketFinalize(*DB));
  CHKERRQ(DMSwarmDataBucketSetSizes(*DB,L,buffer));
  /* now copy the desired guys from DBIn => DB */
  for (p = 0; p < N; ++p) {
    CHKERRQ(DMSwarmDataBucketCopyPoint(DBIn,list[p], *DB,list[p]));
  }
  PetscFunctionReturn(0);
}

/* insert into an exisitng location */
PetscErrorCode DMSwarmDataFieldInsertPoint(const DMSwarmDataField field,const PetscInt index,const void *ctx)
{
  PetscFunctionBegin;
#if defined(DMSWARM_DATAFIELD_POINT_ACCESS_GUARD)
  /* check point is valid */
  PetscCheckFalse(index < 0,PETSC_COMM_SELF,PETSC_ERR_USER,"index must be >= 0");
  PetscCheckFalse(index >= field->L,PETSC_COMM_SELF,PETSC_ERR_USER,"index must be < %D",field->L);
#endif
  CHKERRQ(PetscMemcpy(DMSWARM_DATAFIELD_point_access(field->data,index,field->atomic_size), ctx, field->atomic_size));
  PetscFunctionReturn(0);
}

/* remove data at index - replace with last point */
PetscErrorCode DMSwarmDataBucketRemovePointAtIndex(const DMSwarmDataBucket db,const PetscInt index)
{
  PetscInt       f;
  PetscBool      any_active_fields;

  PetscFunctionBegin;
#if defined(DMSWARM_DATAFIELD_POINT_ACCESS_GUARD)
  /* check point is valid */
  PetscCheckFalse(index < 0,PETSC_COMM_SELF,PETSC_ERR_USER,"index must be >= 0");
  PetscCheckFalse(index >= db->allocated,PETSC_COMM_SELF,PETSC_ERR_USER,"index must be < %D",db->L+db->buffer);
#endif
  CHKERRQ(DMSwarmDataBucketQueryForActiveFields(db,&any_active_fields));
  PetscCheck(!any_active_fields,PETSC_COMM_SELF,PETSC_ERR_USER,"Cannot safely remove point as at least one DMSwarmDataField is currently being accessed");
  if (index >= db->L) { /* this point is not in the list - no need to error, but I will anyway */
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"You should not be trying to remove point at index=%D since it's < db->L = %D", index, db->L);
  }
  if (index != db->L-1) { /* not last point in list */
    for (f = 0; f < db->nfields; ++f) {
      DMSwarmDataField field = db->field[f];

      /* copy then remove */
      CHKERRQ(DMSwarmDataFieldCopyPoint(db->L-1, field, index, field));
      /* DMSwarmDataFieldZeroPoint(field,index); */
    }
  }
  /* decrement size */
  /* this will zero out an crap at the end of the list */
  CHKERRQ(DMSwarmDataBucketRemovePoint(db));
  PetscFunctionReturn(0);
}

/* copy x into y */
PetscErrorCode DMSwarmDataFieldCopyPoint(const PetscInt pid_x,const DMSwarmDataField field_x,const PetscInt pid_y,const DMSwarmDataField field_y)
{
  PetscFunctionBegin;
#if defined(DMSWARM_DATAFIELD_POINT_ACCESS_GUARD)
  /* check point is valid */
  PetscCheckFalse(pid_x < 0,PETSC_COMM_SELF,PETSC_ERR_USER,"(IN) index must be >= 0");
  PetscCheckFalse(pid_x >= field_x->L,PETSC_COMM_SELF,PETSC_ERR_USER,"(IN) index must be < %D",field_x->L);
  PetscCheckFalse(pid_y < 0,PETSC_COMM_SELF,PETSC_ERR_USER,"(OUT) index must be >= 0");
  PetscCheckFalse(pid_y >= field_y->L,PETSC_COMM_SELF,PETSC_ERR_USER,"(OUT) index must be < %D",field_y->L);
  PetscCheckFalse(field_y->atomic_size != field_x->atomic_size,PETSC_COMM_SELF,PETSC_ERR_USER,"atomic size must match");
#endif
  CHKERRQ(PetscMemcpy(DMSWARM_DATAFIELD_point_access(field_y->data,pid_y,field_y->atomic_size),DMSWARM_DATAFIELD_point_access(field_x->data,pid_x,field_x->atomic_size),field_y->atomic_size));
  PetscFunctionReturn(0);
}

/* zero only the datafield at this point */
PetscErrorCode DMSwarmDataFieldZeroPoint(const DMSwarmDataField field,const PetscInt index)
{
  PetscFunctionBegin;
#if defined(DMSWARM_DATAFIELD_POINT_ACCESS_GUARD)
  /* check point is valid */
  PetscCheckFalse(index < 0,PETSC_COMM_SELF,PETSC_ERR_USER,"index must be >= 0");
  PetscCheckFalse(index >= field->L,PETSC_COMM_SELF,PETSC_ERR_USER,"index must be < %D",field->L);
#endif
  CHKERRQ(PetscMemzero(DMSWARM_DATAFIELD_point_access(field->data,index,field->atomic_size), field->atomic_size));
  PetscFunctionReturn(0);
}

/* zero ALL data for this point */
PetscErrorCode DMSwarmDataBucketZeroPoint(const DMSwarmDataBucket db,const PetscInt index)
{
  PetscInt       f;

  PetscFunctionBegin;
  /* check point is valid */
  PetscCheckFalse(index < 0,PETSC_COMM_SELF,PETSC_ERR_USER,"index must be >= 0");
  PetscCheckFalse(index >= db->allocated,PETSC_COMM_SELF,PETSC_ERR_USER,"index must be < %D",db->allocated);
  for (f = 0; f < db->nfields; ++f) {
    DMSwarmDataField field = db->field[f];
    CHKERRQ(DMSwarmDataFieldZeroPoint(field,index));
  }
  PetscFunctionReturn(0);
}

/* increment */
PetscErrorCode DMSwarmDataBucketAddPoint(DMSwarmDataBucket db)
{
  PetscFunctionBegin;
  CHKERRQ(DMSwarmDataBucketSetSizes(db,db->L+1,DMSWARM_DATA_BUCKET_BUFFER_DEFAULT));
  PetscFunctionReturn(0);
}

/* decrement */
PetscErrorCode DMSwarmDataBucketRemovePoint(DMSwarmDataBucket db)
{
  PetscFunctionBegin;
  CHKERRQ(DMSwarmDataBucketSetSizes(db,db->L-1,DMSWARM_DATA_BUCKET_BUFFER_DEFAULT));
  PetscFunctionReturn(0);
}

/*  Should be redone to user PetscViewer */
PetscErrorCode DMSwarmDataBucketView_stdout(MPI_Comm comm,DMSwarmDataBucket db)
{
  PetscInt       f;
  double         memory_usage_total,memory_usage_total_local = 0.0;

  PetscFunctionBegin;
  CHKERRQ(PetscPrintf(comm,"DMSwarmDataBucketView: \n"));
  CHKERRQ(PetscPrintf(comm,"  L                  = %D \n", db->L));
  CHKERRQ(PetscPrintf(comm,"  buffer             = %D \n", db->buffer));
  CHKERRQ(PetscPrintf(comm,"  allocated          = %D \n", db->allocated));
  CHKERRQ(PetscPrintf(comm,"  nfields registered = %D \n", db->nfields));

  for (f = 0; f < db->nfields; ++f) {
    double memory_usage_f = (double)(db->field[f]->atomic_size * db->allocated) * 1.0e-6;
    memory_usage_total_local += memory_usage_f;
  }
  CHKERRMPI(MPI_Allreduce(&memory_usage_total_local,&memory_usage_total,1,MPI_DOUBLE,MPI_SUM,comm));

  for (f = 0; f < db->nfields; ++f) {
    double memory_usage_f = (double)(db->field[f]->atomic_size * db->allocated) * 1.0e-6;
    CHKERRQ(PetscPrintf(comm,"    [%3D] %15s : Mem. usage       = %1.2e (MB) [rank0]\n", f, db->field[f]->name, memory_usage_f));
    CHKERRQ(PetscPrintf(comm,"                            blocksize        = %D \n", db->field[f]->bs));
    if (db->field[f]->bs != 1) {
      CHKERRQ(PetscPrintf(comm,"                            atomic size      = %zu [full block, bs=%D]\n", db->field[f]->atomic_size,db->field[f]->bs));
      CHKERRQ(PetscPrintf(comm,"                            atomic size/item = %zu \n", db->field[f]->atomic_size/db->field[f]->bs));
    } else {
      CHKERRQ(PetscPrintf(comm,"                            atomic size      = %zu \n", db->field[f]->atomic_size));
    }
  }
  CHKERRQ(PetscPrintf(comm,"  Total mem. usage                           = %1.2e (MB) (collective)\n", memory_usage_total));
  PetscFunctionReturn(0);
}

PetscErrorCode DMSwarmDataBucketView_Seq(MPI_Comm comm,DMSwarmDataBucket db,const char filename[],DMSwarmDataBucketViewType type)
{
  PetscFunctionBegin;
  switch (type) {
  case DATABUCKET_VIEW_STDOUT:
    CHKERRQ(DMSwarmDataBucketView_stdout(PETSC_COMM_SELF,db));
    break;
  case DATABUCKET_VIEW_ASCII:
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"No support for ascii output");
  case DATABUCKET_VIEW_BINARY:
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"No support for binary output");
  case DATABUCKET_VIEW_HDF5:
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"No support for HDF5 output");
  default: SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Unknown viewer method requested");
  }
  PetscFunctionReturn(0);
}

PetscErrorCode DMSwarmDataBucketView_MPI(MPI_Comm comm,DMSwarmDataBucket db,const char filename[],DMSwarmDataBucketViewType type)
{
  PetscFunctionBegin;
  switch (type) {
  case DATABUCKET_VIEW_STDOUT:
    CHKERRQ(DMSwarmDataBucketView_stdout(comm,db));
    break;
  case DATABUCKET_VIEW_ASCII:
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"No support for ascii output");
  case DATABUCKET_VIEW_BINARY:
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"No support for binary output");
  case DATABUCKET_VIEW_HDF5:
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"No support for HDF5 output");
  default: SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Unknown viewer method requested");
  }
  PetscFunctionReturn(0);
}

PetscErrorCode DMSwarmDataBucketView(MPI_Comm comm,DMSwarmDataBucket db,const char filename[],DMSwarmDataBucketViewType type)
{
  PetscMPIInt    size;

  PetscFunctionBegin;
  CHKERRMPI(MPI_Comm_size(comm,&size));
  if (size == 1) {
    CHKERRQ(DMSwarmDataBucketView_Seq(comm,db,filename,type));
  } else {
    CHKERRQ(DMSwarmDataBucketView_MPI(comm,db,filename,type));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode DMSwarmDataBucketDuplicateFields(DMSwarmDataBucket dbA,DMSwarmDataBucket *dbB)
{
  DMSwarmDataBucket db2;
  PetscInt          f;

  PetscFunctionBegin;
  CHKERRQ(DMSwarmDataBucketCreate(&db2));
  /* copy contents from dbA into db2 */
  for (f = 0; f < dbA->nfields; ++f) {
    DMSwarmDataField field;
    size_t           atomic_size;
    char             *name;

    field = dbA->field[f];
    atomic_size = field->atomic_size;
    name        = field->name;
    CHKERRQ(DMSwarmDataBucketRegisterField(db2,"DMSwarmDataBucketDuplicateFields",name,atomic_size,NULL));
  }
  CHKERRQ(DMSwarmDataBucketFinalize(db2));
  CHKERRQ(DMSwarmDataBucketSetInitialSizes(db2,0,1000));
  *dbB = db2;
  PetscFunctionReturn(0);
}

/*
 Insert points from db2 into db1
 db1 <<== db2
 */
PetscErrorCode DMSwarmDataBucketInsertValues(DMSwarmDataBucket db1,DMSwarmDataBucket db2)
{
  PetscInt       n_mp_points1,n_mp_points2;
  PetscInt       n_mp_points1_new,p;

  PetscFunctionBegin;
  CHKERRQ(DMSwarmDataBucketGetSizes(db1,&n_mp_points1,NULL,NULL));
  CHKERRQ(DMSwarmDataBucketGetSizes(db2,&n_mp_points2,NULL,NULL));
  n_mp_points1_new = n_mp_points1 + n_mp_points2;
  CHKERRQ(DMSwarmDataBucketSetSizes(db1,n_mp_points1_new,DMSWARM_DATA_BUCKET_BUFFER_DEFAULT));
  for (p = 0; p < n_mp_points2; ++p) {
    /* db1 <<== db2 */
    CHKERRQ(DMSwarmDataBucketCopyPoint(db2,p, db1,(n_mp_points1 + p)));
  }
  PetscFunctionReturn(0);
}

/* helpers for parallel send/recv */
PetscErrorCode DMSwarmDataBucketCreatePackedArray(DMSwarmDataBucket db,size_t *bytes,void **buf)
{
  PetscInt       f;
  size_t         sizeof_marker_contents;
  void          *buffer;

  PetscFunctionBegin;
  sizeof_marker_contents = 0;
  for (f = 0; f < db->nfields; ++f) {
    DMSwarmDataField df = db->field[f];
    sizeof_marker_contents += df->atomic_size;
  }
  CHKERRQ(PetscMalloc(sizeof_marker_contents, &buffer));
  CHKERRQ(PetscMemzero(buffer, sizeof_marker_contents));
  if (bytes) {*bytes = sizeof_marker_contents;}
  if (buf)   {*buf   = buffer;}
  PetscFunctionReturn(0);
}

PetscErrorCode DMSwarmDataBucketDestroyPackedArray(DMSwarmDataBucket db,void **buf)
{
  PetscFunctionBegin;
  if (buf) {
    CHKERRQ(PetscFree(*buf));
    *buf = NULL;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode DMSwarmDataBucketFillPackedArray(DMSwarmDataBucket db,const PetscInt index,void *buf)
{
  PetscInt       f;
  void          *data, *data_p;
  size_t         asize, offset;

  PetscFunctionBegin;
  offset = 0;
  for (f = 0; f < db->nfields; ++f) {
    DMSwarmDataField df = db->field[f];

    asize = df->atomic_size;
    data = (void*)( df->data);
    data_p = (void*)( (char*)data + index*asize);
    CHKERRQ(PetscMemcpy((void*)((char*)buf + offset), data_p, asize));
    offset = offset + asize;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode DMSwarmDataBucketInsertPackedArray(DMSwarmDataBucket db,const PetscInt idx,void *data)
{
  PetscInt       f;
  void           *data_p;
  size_t         offset;

  PetscFunctionBegin;
  offset = 0;
  for (f = 0; f < db->nfields; ++f) {
    DMSwarmDataField df = db->field[f];

    data_p = (void*)( (char*)data + offset);
    CHKERRQ(DMSwarmDataFieldInsertPoint(df, idx, (void*)data_p));
    offset = offset + df->atomic_size;
  }
  PetscFunctionReturn(0);
}
