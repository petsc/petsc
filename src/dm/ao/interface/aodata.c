
#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: aodata.c,v 1.12 1997/10/28 14:25:29 bsmith Exp bsmith $";
#endif
/*  
   Defines the abstract operations on AOData
*/
#include "src/ao/aoimpl.h"      /*I "ao.h" I*/

#undef __FUNC__  
#define __FUNC__ "AODataKeyFind_Private" 
/*
   AODataKeyFind_Private - Given a key finds the int key coordinates. Generates a flag if not found.

   Input Paramters:
.    keyname - string name of key

   Output Parameter:
.    flag - zero if found, -1 if no room for new one, 1 if found available slot
.     key - integer of keyname

*/
int AODataKeyFind_Private(AOData aodata,char *keyname, int *flag,int *key)
{
  int i;

  PetscFunctionBegin;
  for ( i=0; i<aodata->nkeys; i++ ) {
    if (PetscStrcmp(aodata->keys[i].name,keyname)) continue;
     /* found the key */
     *flag    = 0;
     *key     = i;
     PetscFunctionReturn(0);
  }
  /* did not find a key */
  if (aodata->nkeys == aodata->nkeys_max) {/* no room for a new key */
    *flag = -1; 
    PetscFunctionReturn(0);
  } 
  *flag    = 1;
  *key     = aodata->nkeys;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "AODataSegmentFind_Private" 
/*
   AODataSegmentFind_Private - Given a key and segment finds the int key, segment
     coordinates. Generates a flag if not found.

   Input Paramters:
.    keyname - string name of key
.    segname - string name of segment

   Output Parameter:
.    flag - zero if found, -1 if no room for new one, 1 if found available slot
     key - integer of keyname
     segment - integer of segment
*/
int AODataSegmentFind_Private(AOData aodata,char *keyname, char *segname, int *flag,int *key,int *segment)
{
  int i,j;

  PetscFunctionBegin;
  for ( i=0; i<aodata->nkeys; i++ ) {
    if (PetscStrcmp(aodata->keys[i].name,keyname)) continue;
     /* found the key */
     for ( j=0; j<aodata->keys[i].nsegments; j++ ) {
       if (PetscStrcmp(aodata->keys[i].segments[j].name,segname)) continue;
       /*  found the segment */
         *flag    = 0;
         *key     = i;
         *segment = j;
         PetscFunctionReturn(0);
     }
     /* found key, but not segment */
     if (aodata->keys[i].nsegments == aodata->keys[i].nsegments_max) {
       *flag = -1; /* no room for a new segment */
       PetscFunctionReturn(0);
     }
     *flag    = 1;
     *key     = i;
     *segment = aodata->keys[i].nsegments;
     PetscFunctionReturn(0);
  }
  /* did not find a key */
  if (aodata->nkeys == aodata->nkeys_max) {/* no room for a new key */
    *flag = -1; 
    PetscFunctionReturn(0);
  } 
  *flag    = 1;
  *key     = aodata->nkeys;
  *segment = 0;
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------------------------------*/
#undef __FUNC__  
#define __FUNC__ "AODataSegmentGet" 
/*@
   AODataSegmentGet - Get data from a particular segment of a database.

   Input Parameters:
.  aodata - the database
.  name - the name of the key
.  segment - the name of the segment
.  n - the number of data items needed by this processor
.  keys - the keys provided by this processor

   Output Parameters:
.  data - the actual data

.keywords: database transactions

.seealso: AODataCreateBasic(), AODataDestroy(), AODataKeyAdd(), AODataSegmentRestore(),
          AODataSegmentGetIS(), AODataSegmentRestoreIS(), AODataSegmentAdd(), 
          AODataKeyGetInfo(), AODataSegmentGetInfo(), AODataSegmentAdd()
@*/
int AODataSegmentGet(AOData aodata,char *name,char *segment,int n,int *keys,void **data)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(aodata,AODATA_COOKIE);
  ierr = (*aodata->ops.segmentget)(aodata,name,segment,n,keys,data); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "AODataSegmentRestore" 
/*@
   AODataSegmentRestore - Restores data from a particular segment of a database.

   Input Parameters:
.  aodata - the database
.  name - the name of the key
.  segment - the name of the segment
.  n - the number of data items needed by this processor
.  keys - the keys provided by this processor

   Output Parameters:
.  data - the actual data

.keywords: database transactions

.seealso: 
@*/
int AODataSegmentRestore(AOData aodata,char *name,char *segment,int n,int *keys,void **data)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(aodata,AODATA_COOKIE);
  ierr = (*aodata->ops.segmentrestore)(aodata,name,segment,n,keys,data); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "AODataSegmentGetIS" 
/*@
   AODataSegmentGetIS - Get data from a particular segment of a database.

   Input Parameters:
.  aodata - the database
.  name - the name of the key
.  segment - the name of the segment
.  is - the keys for data requested on this processor

   Output Parameters:
.  data - the actual data

.keywords: database transactions

.seealso:
@*/
int AODataSegmentGetIS(AOData aodata,char *name,char *segment,IS is,void **data)
{
  int ierr,n,*keys;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(aodata,AODATA_COOKIE);
  PetscValidHeaderSpecific(is,IS_COOKIE);

  ierr = ISGetSize(is,&n); CHKERRQ(ierr);
  ierr = ISGetIndices(is,&keys); CHKERRQ(ierr);
  ierr = (*aodata->ops.segmentget)(aodata,name,segment,n,keys,data); CHKERRQ(ierr);
  ierr = ISRestoreIndices(is,&keys); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "AODataSegmentRestoreIS" 
/*@
   AODataSegmentRestoreIS - Restores data from a particular segment of a database.

   Input Parameters:
.  aodata - the database
.  name - the name of the data key
.  segment - the name of the segment
.  is - the keys provided by this processor

   Output Parameters:
.  data - the actual data

.keywords: database transactions

.seealso:
@*/
int AODataSegmentRestoreIS(AOData aodata,char *name,char *segment,IS is,void **data)
{
  int ierr,n,*keys;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(is,IS_COOKIE);
  PetscValidHeaderSpecific(aodata,AODATA_COOKIE);

  ierr = ISGetSize(is,&n); CHKERRQ(ierr);
  ierr = ISGetIndices(is,&keys); CHKERRQ(ierr);

  ierr = (*aodata->ops.segmentrestore)(aodata,name,segment,n,keys,data);CHKERRQ(ierr);
  ierr = ISRestoreIndices(is,&keys); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------------------------------*/
#undef __FUNC__  
#define __FUNC__ "AODataSegmentGetLocal" 
/*@
   AODataSegmentGetLocal - Get data from a particular segment of a database.

   Input Parameters:
.  aodata - the database
.  name - the name of the key
.  segment - the name of the segment
.  n - the number of data items needed by this processor
.  keys - the keys provided by this processor

   Output Parameters:
.  data - the actual data

.keywords: database transactions

.seealso: AODataCreateBasic(), AODataDestroy(), AODataKeyAdd(), AODataSegmentRestore(),
          AODataSegmentGetIS(), AODataSegmentRestoreIS(), AODataSegmentAdd(), 
          AODataKeyGetInfo(), AODataSegmentGetInfo(), AODataSegmentAdd()
@*/
int AODataSegmentGetLocal(AOData aodata,char *name,char *segment,int n,int *keys,void **data)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(aodata,AODATA_COOKIE);
  ierr = (*aodata->ops.segmentgetlocal)(aodata,name,segment,n,keys,data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "AODataSegmentRestoreLocal" 
/*@
   AODataSegmentRestoreLocal - Restores data from a particular segment of a database.

   Input Parameters:
.  aodata - the database
.  name - the name of the key
.  segment - the name of the segment
.  n - the number of data items needed by this processor
.  keys - the keys provided by this processor

   Output Parameters:
.  data - the actual data

.keywords: database transactions

.seealso: 
@*/
int AODataSegmentRestoreLocal(AOData aodata,char *name,char *segment,int n,int *keys,void **data)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(aodata,AODATA_COOKIE);
  ierr = (*aodata->ops.segmentrestorelocal)(aodata,name,segment,n,keys,data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "AODataSegmentGetLocalIS" 
/*@
   AODataSegmentGetLocalIS - Get data from a particular segment of a database.

   Input Parameters:
.  aodata - the database
.  name - the name of the key
.  segment - the name of the segment
.  is - the keys for data requested on this processor

   Output Parameters:
.  data - the actual data

.keywords: database transactions

.seealso:
@*/
int AODataSegmentGetLocalIS(AOData aodata,char *name,char *segment,IS is,void **data)
{
  int ierr,n,*keys;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(aodata,AODATA_COOKIE);
  PetscValidHeaderSpecific(is,IS_COOKIE);

  ierr = ISGetSize(is,&n); CHKERRQ(ierr);
  ierr = ISGetIndices(is,&keys); CHKERRQ(ierr);
  ierr = (*aodata->ops.segmentgetlocal)(aodata,name,segment,n,keys,data); CHKERRQ(ierr);
  ierr = ISRestoreIndices(is,&keys); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "AODataSegmentRestoreLocalIS" 
/*@
   AODataSegmentRestoreLocalIS - Restores data from a particular segment of a database.

   Input Parameters:
.  aodata - the database
.  name - the name of the data key
.  segment - the name of the segment
.  is - the keys provided by this processor

   Output Parameters:
.  data - the actual data

.keywords: database transactions

.seealso:
@*/
int AODataSegmentRestoreLocalIS(AOData aodata,char *name,char *segment,IS is,void **data)
{
  int ierr,n,*keys;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(is,IS_COOKIE);
  PetscValidHeaderSpecific(aodata,AODATA_COOKIE);

  ierr = ISGetSize(is,&n); CHKERRQ(ierr);
  ierr = ISGetIndices(is,&keys); CHKERRQ(ierr);

  ierr = (*aodata->ops.segmentrestorelocal)(aodata,name,segment,n,keys,data);CHKERRQ(ierr);
  ierr = ISRestoreIndices(is,&keys); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------------------------------*/

#undef __FUNC__  
#define __FUNC__ "AODataKeyGetNeighbors" 
/*@
   AODataKeyGetNeighbors - Given a list of keys generates a new list containing
         those keys plus neighbors found in a neighbors list.

   Input Parameters:
.  aodata - the database
.  name - the name of the key
.  n - the number of data items needed by this processor
.  keys - the keys provided by this processor

   Output Parameters:
.  is - the indices retrieved

.keywords: database transactions

.seealso: AODataCreateBasic(), AODataDestroy(), AODataKeyAdd(), AODataSegmentRestore(),
          AODataSegmentGetIS(), AODataSegmentRestoreIS(), AODataSegmentAdd(), 
          AODataKeyGetInfo(), AODataSegmentGetInfo(), AODataSegmentAdd(), 
          AODataKeyGetNeighborsIS()
@*/
int AODataKeyGetNeighbors(AOData aodata,char *name,int n,int *keys,IS *is)
{
  int ierr;
  IS  reduced,input;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(aodata,AODATA_COOKIE);
 
  /* get the list of neighbors */
  ierr = AODataSegmentGetReduced(aodata,name,name,n,keys,&reduced);CHKERRQ(ierr);

  ierr = ISCreateGeneral(aodata->comm,n,keys,&input);CHKERRQ(ierr);
  ierr = ISSum(input,reduced,is);CHKERRQ(ierr);
  ierr = ISDestroy(input);CHKERRQ(ierr);
  ierr = ISDestroy(reduced);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "AODataKeyGetNeighborsIS" 
/*@
   AODataKeyGetNeighborsIS - Given a list of keys generates a new list containing
         those keys plus neighbors found in a neighbors list.

   Input Parameters:
.  aodata - the database
.  name - the name of the key
.  n - the number of data items needed by this processor
.  keys - the keys provided by this processor

   Output Parameters:
.  is - the indices retrieved

.keywords: database transactions

.seealso: AODataCreateBasic(), AODataDestroy(), AODataKeyAdd(), AODataSegmentRestore(),
          AODataSegmentGetIS(), AODataSegmentRestoreIS(), AODataSegmentAdd(), 
          AODataKeyGetInfo(), AODataSegmentGetInfo(), AODataSegmentAdd(), 
          AODataKeyGetNeighbors()
@*/
int AODataKeyGetNeighborsIS(AOData aodata,char *name,IS keys,IS *is)
{
  int ierr;
  IS  reduced;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(aodata,AODATA_COOKIE);
 
  /* get the list of neighbors */
  ierr = AODataSegmentGetReducedIS(aodata,name,name,keys,&reduced);CHKERRQ(ierr);
  ierr = ISSum(keys,reduced,is);CHKERRQ(ierr);
  ierr = ISDestroy(reduced);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "AODataSegmentGetReduced" 
/*@
   AODataSegmentGetReduced - Get data from a particular segment of a database.

   Input Parameters:
.  aodata - the database
.  name - the name of the key
.  segment - the name of the segment
.  n - the number of data items needed by this processor
.  keys - the keys provided by this processor

   Output Parameters:
.  is - the indices retrieved

.keywords: database transactions

.seealso: AODataCreateBasic(), AODataDestroy(), AODataKeyAdd(), AODataSegmentRestore(),
          AODataSegmentGetIS(), AODataSegmentRestoreIS(), AODataSegmentAdd(), 
          AODataKeyGetInfo(), AODataSegmentGetInfo(), AODataSegmentAdd()
@*/
int AODataSegmentGetReduced(AOData aodata,char *name,char *segment,int n,int *keys,IS *is)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(aodata,AODATA_COOKIE);
  ierr = (*aodata->ops.segmentgetreduced)(aodata,name,segment,n,keys,is); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "AODataSegmentGetReducedIS" 
/*@
   AODataSegmentGetReducedIS - Get data from a particular segment of a database.

   Input Parameters:
.  aodata - the database
.  name - the name of the key
.  segment - the name of the segment
.  is - the keys for data requested on this processor

   Output Parameters:
.  isout - the indices retreived

.keywords: database transactions

.seealso:
@*/
int AODataSegmentGetReducedIS(AOData aodata,char *name,char *segment,IS is,IS *isout)
{
  int ierr,n,*keys;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(aodata,AODATA_COOKIE);
  PetscValidHeaderSpecific(is,IS_COOKIE);

  ierr = ISGetSize(is,&n); CHKERRQ(ierr);
  ierr = ISGetIndices(is,&keys); CHKERRQ(ierr);
  ierr = (*aodata->ops.segmentgetreduced)(aodata,name,segment,n,keys,isout); CHKERRQ(ierr);
  ierr = ISRestoreIndices(is,&keys); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------------------------------*/

#undef __FUNC__  
#define __FUNC__ "AODataKeyAddLocalToGlobalMapping" 
/*@
   AODataKeyAddLocalToGlobalMapping - Add another data key to a AOData database.

   Input Parameters:
.  aodata - the database
.  name - the name of the key
.  N - the number of indices in the key
.  nlocal - number of indices to be associated with this processor
.  nsegments - the number of segments associated with the key

.keywords: database additions

.seealso:
@*/
int AODataKeyAddLocalToGlobalMapping(AOData aodata,char *name,ISLocalToGlobalMapping map)
{
  int       ierr,ikey,flag;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(aodata,AODATA_COOKIE);

  ierr = AODataKeyFind_Private(aodata,name,&flag,&ikey);CHKERRQ(ierr);
  if (flag)  SETERRQ(1,1,"Key does not exist");

  aodata->keys[ikey].ltog = map;
  PetscObjectReference((PetscObject) map);

  PetscFunctionReturn(0);

}

#undef __FUNC__  
#define __FUNC__ "AODataKeyAdd" 
/*@
   AODataKeyAdd - Add another data key to a AOData database.

   Input Parameters:
.  aodata - the database
.  name - the name of the key
.  N - the number of indices in the key
.  nlocal - number of indices to be associated with this processor
.  nsegments - the number of segments associated with the key

.keywords: database additions

.seealso:
@*/
int AODataKeyAdd(AOData aodata,char *name,int nlocal,int N,int nsegments)
{
  int       ierr,ikey,flag,Ntmp,size,rank,i,len;
  AODataKey *key;
  MPI_Comm  comm = aodata->comm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(aodata,AODATA_COOKIE);

  ierr = AODataKeyFind_Private(aodata,name,&flag,&ikey);CHKERRQ(ierr);
  if (flag == 0)  SETERRQ(1,1,"Key already exists with given name");
  if (flag == -1) SETERRQ(1,1,"Already full set of keys defined");
  if (nlocal == PETSC_DECIDE && N == PETSC_DECIDE) SETERRQ(1,1,"nlocal and N both PETSC_DECIDE");

  key                = aodata->keys + ikey;
  len                = PetscStrlen(name);
  key->name          = (char *) PetscMalloc((len+1)*sizeof(char));CHKPTRQ(key->name);
  PetscStrcpy(key->name,name);
  key->N             = N;
  key->nsegments_max = nsegments;
  key->nsegments     = 0;
  key->segments      = (AODataSegment*) PetscMalloc((nsegments+1)*sizeof(AODataSegment));
                       CHKPTRQ(key->segments);
  key->ltog          = 0;

  MPI_Comm_rank(comm,&rank);
  MPI_Comm_size(comm,&size);

  /*  Set nlocal and ownership ranges */
  if (N == PETSC_DECIDE) {
    ierr = MPI_Allreduce(&nlocal,&N,1,MPI_INT,MPI_SUM,comm);CHKERRQ(ierr);
  } else if (nlocal != PETSC_DECIDE) {
    ierr = MPI_Allreduce(&nlocal,&Ntmp,1,MPI_INT,MPI_SUM,comm);CHKERRQ(ierr);
    if (Ntmp != N) SETERRQ(1,1,"Sum of nlocal is not N");
  } else {
    nlocal = N/size + ((N % size) > rank);
  }
  key->rowners = (int *) PetscMalloc((size+1)*sizeof(int));CHKPTRQ(key->rowners);
  ierr = MPI_Allgather(&nlocal,1,MPI_INT,key->rowners+1,1,MPI_INT,comm);CHKERRQ(ierr);
  key->rowners[0] = 0;
  for (i=2; i<=size; i++ ) {
    key->rowners[i] += key->rowners[i-1];
  }
  key->rstart        = key->rowners[rank];
  key->rend          = key->rowners[rank+1];

  key->nlocal        = nlocal;

  aodata->nkeys++;

  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "AODataSegmentAdd" 
/*@
   AODataSegmentAdd - Add another data segment to a AOData database.

   Input Parameters:
.  aodata - the database
.  name - the name of the key
.  segment - the name of the data segment
.  bs - the fundamental blocksize of the data
.  n - the number of data items contributed by this processor
.  keys - the keys provided by this processor
.  data - the actual data
.  dtype - the data type, one of PETSC_INT, PETSC_DOUBLE, PETSC_SCALAR etc

.keywords: database additions

.seealso:
@*/
int AODataSegmentAdd(AOData aodata,char *name,char *segment,int bs,int n,int *keys,void *data,
                     PetscDataType dtype)
{
  int      ierr,i,flg1;
  MPI_Comm comm = aodata->comm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(aodata,AODATA_COOKIE);

  ierr = (*aodata->ops.segmentadd)(aodata,name,segment,bs,n,keys,data,dtype); CHKERRQ(ierr);

  /* Determine if all segments for all keys have been filled yet */
  if (aodata->nkeys < aodata->nkeys_max) PetscFunctionReturn(0);
  for ( i=0; i<aodata->nkeys; i++ ) {
    if (aodata->keys[i].nsegments < aodata->keys[i].nsegments_max) PetscFunctionReturn(0);
  }
  aodata->datacomplete = 1;
  ierr = OptionsHasName(PETSC_NULL,"-ao_data_view",&flg1); CHKERRQ(ierr);
  if (flg1 && aodata->datacomplete) {
    ierr = AODataView(aodata,VIEWER_STDOUT_(comm)); CHKERRQ(ierr);
  }
  ierr = OptionsHasName(PETSC_NULL,"-ao_data_view_info",&flg1); CHKERRQ(ierr);
  if (flg1 && aodata->datacomplete) {
    ierr = ViewerPushFormat(VIEWER_STDOUT_(comm),VIEWER_FORMAT_ASCII_INFO,0);CHKERRQ(ierr);
    ierr = AODataView(aodata,VIEWER_STDOUT_(comm)); CHKERRQ(ierr);
    ierr = ViewerPopFormat(VIEWER_STDOUT_(comm));CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "AODataSegmentAddIS" 
/*@
   AODataSegmentAddIS - Add another data segment to a AOData database.

   Input Parameters:
.  aodata - the database
.  name - the name of the key
.  segment - name of segment
.  bs - the fundamental blocksize of the data
.  is - the keys provided by this processor
.  data - the actual data
.  dtype - the data type, one of PETSC_INT, PETSC_DOUBLE, PETSC_SCALAR etc

.keywords: database additions

.seealso:
@*/
int AODataSegmentAddIS(AOData aodata,char *name,char *segment,int bs,IS is,void *data,
                       PetscDataType dtype)
{
  int n,*keys,ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(aodata,AODATA_COOKIE);
  PetscValidHeaderSpecific(is,IS_COOKIE);

  ierr = ISGetSize(is,&n); CHKERRQ(ierr);
  ierr = ISGetIndices(is,&keys); CHKERRQ(ierr);
  ierr = (*aodata->ops.segmentadd)(aodata,name,segment,bs,n,keys,data,dtype); CHKERRQ(ierr);
  ierr = ISRestoreIndices(is,&keys); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "AODataKeyGetInfoOwnership"
/*@
   AODataKeyGetInfoOwnership - Gets the ownership range to this key type.

   Input Parameters:
.  aodata - the database
.  name - the name of the key

   Output Parameters:
.  rstart - first key owned locally
.  rend - last key owned locally

.keywords: database accessing

.seealso:
@*/
int AODataKeyGetInfoOwnership(AOData aodata,char *name,int *rstart,int *rend)
{
  int key,ierr,flag;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(aodata,AODATA_COOKIE);

  ierr = AODataKeyFind_Private(aodata,name,&flag,&key); CHKERRQ(ierr);
  if (flag) SETERRQ(1,1,"Key never created");

  *rstart = aodata->keys[key].rstart;
  *rend   = aodata->keys[key].rend;

  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "AODataKeyGetInfo"
/*@
   AODataKeyGetInfo - Gets the global size, local size and number of segments in a key.

   Input Parameters:
.  aodata - the database
.  name - the name of the key

   Output Parameters:
.  nglobal - global number of keys
.  nlocal - local number of keys
.  nsegments - number of segments associated with key

.keywords: database accessing

.seealso:
@*/
int AODataKeyGetInfo(AOData aodata,char *name,int *nglobal,int *nlocal,int *nsegments)
{
  int key,ierr,flag;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(aodata,AODATA_COOKIE);

  ierr = AODataKeyFind_Private(aodata,name,&flag,&key); CHKERRQ(ierr);
  if (flag) SETERRQ(1,1,"Key never created");

  if (nglobal)   *nglobal   = aodata->keys[key].N;
  if (nlocal)    *nlocal    = aodata->keys[key].nlocal;
  if (nsegments) *nsegments = aodata->keys[key].nsegments;

  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "AODataSegmentGetInfo"
/*@
   AODataSegmentGetInfo - Gets the global size, local size, blocksize and type of a data segment

   Input Parameters:
.  aodata - the database
.  keyname - the name of the key
.  segname - the name of the segment

   Output Parameters:
.  nglobal - total number of keys
.  nlocal - local number of keys
.  bs - the blocksize
.  dtype - the datatype


.keywords: database accessing

.seealso:
@*/
int AODataSegmentGetInfo(AOData aodata,char *keyname,char *segname,int *nglobal,int *nlocal,
                         int *bs, PetscDataType *dtype)
{
  int key,ierr,flag,seg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(aodata,AODATA_COOKIE);

  ierr = AODataSegmentFind_Private(aodata,keyname,segname,&flag,&key,&seg); CHKERRQ(ierr);
  if (flag) SETERRQ(1,1,"Key or segment never created");
  if (nglobal)   *nglobal   = aodata->keys[key].N;
  if (nlocal)    *nlocal    = aodata->keys[key].nlocal;
  if (bs)        *bs        = aodata->keys[key].segments[seg].bs;
  if (dtype)     *dtype     = aodata->keys[key].segments[seg].datatype;

  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "AODataView" 
/*@
   AODataView - Displays an application ordering.

   Input Parameters:
.  aodata - the database
.  viewer - viewer used to display the set, for example VIEWER_STDOUT_SELF.

.keywords: database viewing

.seealso: ViewerFileOpenASCII()
@*/
int AODataView(AOData aodata, Viewer viewer)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(aodata,AODATA_COOKIE);
  ierr = (*aodata->view)((PetscObject)aodata,viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "AODataDestroy" 
/*@
   AODataDestroy - Destroys an application ordering set.

   Input Parameters:
.  aodata - the database

.keywords: destroy, database

.seealso: AODataCreateBasic()
@*/
int AODataDestroy(AOData aodata)
{
  int ierr;

  PetscFunctionBegin;

  if (!aodata) PetscFunctionReturn(0);
  PetscValidHeaderSpecific(aodata,AODATA_COOKIE);
  if (--aodata->refct > 0) PetscFunctionReturn(0);
  ierr = (*aodata->destroy)((PetscObject)aodata); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "AODataKeyRemap" 
/*@
   AODataKeyRemap - Remaps a key and all references to a key to a new numbering 
     scheme where each processor indicates its new nodes by listing them in the
     previous numbering scheme.

   Input Parameters:
.  aodata - the database
.  key  - the key to remap
.  ao - the old to new ordering

.keywords: database remapping

.seealso: 
@*/
int AODataKeyRemap(AOData aodata, char *key,AO ao)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(aodata,AODATA_COOKIE);
  PetscValidHeaderSpecific(ao,AO_COOKIE);
  ierr = (*aodata->ops.keyremap)(aodata,key,ao);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "AODataKeyGetAdjacency" 
/*@
   AODataKeyGetAdjacency - Gets the adjacency graph for a key.

   Input Parameters:
.  aodata - the database
.  key  - the key

   Output Parameter:
.  adj - the adjacency graph

.keywords: database, adjacency graph

.seealso: 
@*/
int AODataKeyGetAdjacency(AOData aodata, char *key,Mat *adj)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(aodata,AODATA_COOKIE);
  ierr = (*aodata->ops.keygetadjacency)(aodata,key,adj);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}



