
#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: aodata.c,v 1.7 1997/10/09 03:55:49 bsmith Exp bsmith $";
#endif
/*  
   Defines the abstract operations on AOData
*/
#include "src/ao/aoimpl.h"      /*I "ao.h" I*/

#undef __FUNC__  
#define __FUNC__ "AODataFindKey_Private" 
/*
   AODataFindKey_Private - Given a key finds the int key coordinates. Generates a flag if not found.

   Input Paramters:
.    keyname - string name of key

   Output Parameter:
.    flag - zero if found, -1 if no room for new one, 1 if found available slot
.     key - integer of keyname

*/
int AODataFindKey_Private(AOData aodata,char *keyname, int *flag,int *key)
{
  int i;

  for ( i=0; i<aodata->nkeys; i++ ) {
    if (PetscStrcmp(aodata->keys[i].name,keyname)) continue;
     /* found the key */
     *flag    = 0;
     *key     = i;
     return 0;
  }
  /* did not find a key */
  if (aodata->nkeys == aodata->nkeys_max) {/* no room for a new key */
    *flag = -1; return 0;
  } 
  *flag    = 1;
  *key     = aodata->nkeys;
  return 0;
}


#undef __FUNC__  
#define __FUNC__ "AODataFindSegment_Private" 
/*
   AODataFindSegment_Private - Given a key and segment finds the int key, segment
     coordinates. Generates a flag if not found.

   Input Paramters:
.    keyname - string name of key
.    segname - string name of segment

   Output Parameter:
.    flag - zero if found, -1 if no room for new one, 1 if found available slot
     key - integer of keyname
     segment - integer of segment
*/
int AODataFindSegment_Private(AOData aodata,char *keyname, char *segname, int *flag,int *key,int *segment)
{
  int i,j;

  for ( i=0; i<aodata->nkeys; i++ ) {
    if (PetscStrcmp(aodata->keys[i].name,keyname)) continue;
     /* found the key */
     for ( j=0; j<aodata->keys[i].nsegments; j++ ) {
       if (PetscStrcmp(aodata->keys[i].segments[j].name,segname)) continue;
       /*  found the segment */
         *flag    = 0;
         *key     = i;
         *segment = j;
         return 0;
     }
     /* found key, but not segment */
     if (aodata->keys[i].nsegments == aodata->keys[i].nsegments_max) {
       *flag = -1; /* no room for a new segment */
       return 0;
     }
     *flag    = 1;
     *key     = i;
     *segment = aodata->keys[i].nsegments;
     return 0;
  }
  /* did not find a key */
  if (aodata->nkeys == aodata->nkeys_max) {/* no room for a new key */
    *flag = -1; 
    return 0;
  } 
  *flag    = 1;
  *key     = aodata->nkeys;
  *segment = 0;
  return 0;
}

/* ------------------------------------------------------------------------------------*/
#undef __FUNC__  
#define __FUNC__ "AODataGetSegment" 
/*@
   AODataGetSegment - Get data from a particular segment of a database.

   Input Parameters:
.  aodata - the database
.  name - the name of the key
.  segment - the name of the segment
.  n - the number of data items needed by this processor
.  keys - the keys provided by this processor

   Output Parameters:
.  data - the actual data

.keywords: database transactions

.seealso: AODataCreateBasic(), AODataDestroy(), AODataAddKey(), AODataRestoreSegment(),
          AODataGetSegmentIS(), AODataRestoreSegmentIS(), AODataAddSegment(), 
          AODataGetInfoKey(), AODataGetInfoSegment(), AODataAddSegment()
@*/
int AODataGetSegment(AOData aodata,char *name,char *segment,int n,int *keys,void **data)
{
  PetscValidHeaderSpecific(aodata,AODATA_COOKIE);
  return (*aodata->ops.getsegment)(aodata,name,segment,n,keys,data);
}

#undef __FUNC__  
#define __FUNC__ "AODataRestoreSegment" 
/*@
   AODataRestoreSegment - Restores data from a particular segment of a database.

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
int AODataRestoreSegment(AOData aodata,char *name,char *segment,int n,int *keys,void **data)
{
  PetscValidHeaderSpecific(aodata,AODATA_COOKIE);
  return (*aodata->ops.restoresegment)(aodata,name,segment,n,keys,data);
}

#undef __FUNC__  
#define __FUNC__ "AODataGetSegmentIS" 
/*@
   AODataGetSegmentIS - Get data from a particular segment of a database.

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
int AODataGetSegmentIS(AOData aodata,char *name,char *segment,IS is,void **data)
{
  int ierr,n,*keys;
  PetscValidHeaderSpecific(aodata,AODATA_COOKIE);
  PetscValidHeaderSpecific(is,IS_COOKIE);

  ierr = ISGetSize(is,&n); CHKERRQ(ierr);
  ierr = ISGetIndices(is,&keys); CHKERRQ(ierr);
  ierr = (*aodata->ops.getsegment)(aodata,name,segment,n,keys,data); CHKERRQ(ierr);
  ierr = ISRestoreIndices(is,&keys); CHKERRQ(ierr);
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "AODataRestoreSegmentIS" 
/*@
   AODataRestoreSegmentIS - Restores data from a particular segment of a database.

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
int AODataRestoreSegmentIS(AOData aodata,char *name,char *segment,IS is,void **data)
{
  int ierr,n,*keys;
  PetscValidHeaderSpecific(is,IS_COOKIE);
  PetscValidHeaderSpecific(aodata,AODATA_COOKIE);

  ierr = ISGetSize(is,&n); CHKERRQ(ierr);
  ierr = ISGetIndices(is,&keys); CHKERRQ(ierr);

  ierr = (*aodata->ops.restoresegment)(aodata,name,segment,n,keys,data);CHKERRQ(ierr);
  ierr = ISRestoreIndices(is,&keys); CHKERRQ(ierr);
  return 0;
}

/* ------------------------------------------------------------------------------------*/
#undef __FUNC__  
#define __FUNC__ "AODataGetLocalSegment" 
/*@
   AODataGetLocalSegment - Get data from a particular segment of a database.

   Input Parameters:
.  aodata - the database
.  name - the name of the key
.  segment - the name of the segment
.  n - the number of data items needed by this processor
.  keys - the keys provided by this processor

   Output Parameters:
.  data - the actual data

.keywords: database transactions

.seealso: AODataCreateBasic(), AODataDestroy(), AODataAddKey(), AODataRestoreSegment(),
          AODataGetSegmentIS(), AODataRestoreSegmentIS(), AODataAddSegment(), 
          AODataGetInfoKey(), AODataGetInfoSegment(), AODataAddSegment()
@*/
int AODataGetLocalSegment(AOData aodata,char *name,char *segment,int n,int *keys,void **data)
{
  PetscValidHeaderSpecific(aodata,AODATA_COOKIE);
  return (*aodata->ops.getlocalsegment)(aodata,name,segment,n,keys,data);
}

#undef __FUNC__  
#define __FUNC__ "AODataRestoreLocalSegment" 
/*@
   AODataRestoreLocalSegment - Restores data from a particular segment of a database.

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
int AODataRestoreLocalSegment(AOData aodata,char *name,char *segment,int n,int *keys,void **data)
{
  PetscValidHeaderSpecific(aodata,AODATA_COOKIE);
  return (*aodata->ops.restorelocalsegment)(aodata,name,segment,n,keys,data);
}

#undef __FUNC__  
#define __FUNC__ "AODataGetLocalSegmentIS" 
/*@
   AODataGetLocalSegmentIS - Get data from a particular segment of a database.

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
int AODataGetLocalSegmentIS(AOData aodata,char *name,char *segment,IS is,void **data)
{
  int ierr,n,*keys;
  PetscValidHeaderSpecific(aodata,AODATA_COOKIE);
  PetscValidHeaderSpecific(is,IS_COOKIE);

  ierr = ISGetSize(is,&n); CHKERRQ(ierr);
  ierr = ISGetIndices(is,&keys); CHKERRQ(ierr);
  ierr = (*aodata->ops.getlocalsegment)(aodata,name,segment,n,keys,data); CHKERRQ(ierr);
  ierr = ISRestoreIndices(is,&keys); CHKERRQ(ierr);
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "AODataRestoreLocalSegmentIS" 
/*@
   AODataRestoreLocalSegmentIS - Restores data from a particular segment of a database.

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
int AODataRestoreLocalSegmentIS(AOData aodata,char *name,char *segment,IS is,void **data)
{
  int ierr,n,*keys;
  PetscValidHeaderSpecific(is,IS_COOKIE);
  PetscValidHeaderSpecific(aodata,AODATA_COOKIE);

  ierr = ISGetSize(is,&n); CHKERRQ(ierr);
  ierr = ISGetIndices(is,&keys); CHKERRQ(ierr);

  ierr = (*aodata->ops.restorelocalsegment)(aodata,name,segment,n,keys,data);CHKERRQ(ierr);
  ierr = ISRestoreIndices(is,&keys); CHKERRQ(ierr);
  return 0;
}

/* ------------------------------------------------------------------------------------*/

#undef __FUNC__  
#define __FUNC__ "AODataGetReducedSegment" 
/*@
   AODataGetReducedSegment - Get data from a particular segment of a database.

   Input Parameters:
.  aodata - the database
.  name - the name of the key
.  segment - the name of the segment
.  n - the number of data items needed by this processor
.  keys - the keys provided by this processor

   Output Parameters:
.  is - the indices retrieved

.keywords: database transactions

.seealso: AODataCreateBasic(), AODataDestroy(), AODataAddKey(), AODataRestoreSegment(),
          AODataGetSegmentIS(), AODataRestoreSegmentIS(), AODataAddSegment(), 
          AODataGetInfoKey(), AODataGetInfoSegment(), AODataAddSegment()
@*/
int AODataGetReducedSegment(AOData aodata,char *name,char *segment,int n,int *keys,IS *is)
{
  PetscValidHeaderSpecific(aodata,AODATA_COOKIE);
  return (*aodata->ops.getreducedsegment)(aodata,name,segment,n,keys,is);
}

#undef __FUNC__  
#define __FUNC__ "AODataGetReducedSegmentIS" 
/*@
   AODataGetReducedSegmentIS - Get data from a particular segment of a database.

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
int AODataGetReducedSegmentIS(AOData aodata,char *name,char *segment,IS is,IS *isout)
{
  int ierr,n,*keys;
  PetscValidHeaderSpecific(aodata,AODATA_COOKIE);
  PetscValidHeaderSpecific(is,IS_COOKIE);

  ierr = ISGetSize(is,&n); CHKERRQ(ierr);
  ierr = ISGetIndices(is,&keys); CHKERRQ(ierr);
  ierr = (*aodata->ops.getreducedsegment)(aodata,name,segment,n,keys,isout); CHKERRQ(ierr);
  ierr = ISRestoreIndices(is,&keys); CHKERRQ(ierr);
  return 0;
}

/* ------------------------------------------------------------------------------------*/

#undef __FUNC__  
#define __FUNC__ "AODataAddKeyLocalToGlobalMapping" 
/*@
   AODataAddKeyLocalToGlobalMapping - Add another data key to a AOData database.

   Input Parameters:
.  aodata - the database
.  name - the name of the key
.  N - the number of indices in the key
.  nlocal - number of indices to be associated with this processor
.  nsegments - the number of segments associated with the key

.keywords: database additions

.seealso:
@*/
int AODataAddKeyLocalToGlobalMapping(AOData aodata,char *name,ISLocalToGlobalMapping map)
{
  int       ierr,ikey,flag;
  PetscValidHeaderSpecific(aodata,AODATA_COOKIE);

  ierr = AODataFindKey_Private(aodata,name,&flag,&ikey);CHKERRQ(ierr);
  if (flag)  SETERRQ(1,1,"Key does not exist");

  aodata->keys[ikey].ltog = map;
  PetscObjectReference((PetscObject) map);

  return 0;

}

#undef __FUNC__  
#define __FUNC__ "AODataAddKey" 
/*@
   AODataAddKey - Add another data key to a AOData database.

   Input Parameters:
.  aodata - the database
.  name - the name of the key
.  N - the number of indices in the key
.  nlocal - number of indices to be associated with this processor
.  nsegments - the number of segments associated with the key

.keywords: database additions

.seealso:
@*/
int AODataAddKey(AOData aodata,char *name,int nlocal,int N,int nsegments)
{
  int       ierr,ikey,flag,Ntmp,size,rank,i,len;
  AODataKey *key;
  MPI_Comm  comm = aodata->comm;
  PetscValidHeaderSpecific(aodata,AODATA_COOKIE);

  ierr = AODataFindKey_Private(aodata,name,&flag,&ikey);CHKERRQ(ierr);
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
    MPI_Allreduce(&nlocal,&N,1,MPI_INT,MPI_SUM,comm);
  } else if (nlocal != PETSC_DECIDE) {
    MPI_Allreduce(&nlocal,&Ntmp,1,MPI_INT,MPI_SUM,comm);
    if (Ntmp != N) SETERRQ(1,1,"Sum of nlocal is not N");
  } else {
    nlocal = N/size + ((N % size) > rank);
  }
  key->rowners = (int *) PetscMalloc((size+1)*sizeof(int));CHKPTRQ(key->rowners);
  MPI_Allgather(&nlocal,1,MPI_INT,key->rowners+1,1,MPI_INT,comm);
  key->rowners[0] = 0;
  for (i=2; i<=size; i++ ) {
    key->rowners[i] += key->rowners[i-1];
  }
  key->rstart        = key->rowners[rank];
  key->rend          = key->rowners[rank+1];

  key->nlocal        = nlocal;

  aodata->nkeys++;

  return 0;
}

#undef __FUNC__  
#define __FUNC__ "AODataAddSegment" 
/*@
   AODataAddSegment - Add another data segment to a AOData database.

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
int AODataAddSegment(AOData aodata,char *name,char *segment,int bs,int n,int *keys,void *data,
                     PetscDataType dtype)
{
  int      ierr,i,flg1;
  MPI_Comm comm = aodata->comm;
  PetscValidHeaderSpecific(aodata,AODATA_COOKIE);

  ierr = (*aodata->ops.addsegment)(aodata,name,segment,bs,n,keys,data,dtype); CHKERRQ(ierr);

  /* Determine if all segments for all keys have been filled yet */
  if (aodata->nkeys < aodata->nkeys_max) return 0;
  for ( i=0; i<aodata->nkeys; i++ ) {
    if (aodata->keys[i].nsegments < aodata->keys[i].nsegments_max) return 0;
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
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "AODataAddSegmentIS" 
/*@
   AODataAddSegmentIS - Add another data segment to a AOData database.

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
int AODataAddSegmentIS(AOData aodata,char *name,char *segment,int bs,IS is,void *data,
                       PetscDataType dtype)
{
  int n,*keys,ierr;
  PetscValidHeaderSpecific(aodata,AODATA_COOKIE);
  PetscValidHeaderSpecific(is,IS_COOKIE);

  ierr = ISGetSize(is,&n); CHKERRQ(ierr);
  ierr = ISGetIndices(is,&keys); CHKERRQ(ierr);
  ierr = (*aodata->ops.addsegment)(aodata,name,segment,bs,n,keys,data,dtype); CHKERRQ(ierr);
  ierr = ISRestoreIndices(is,&keys); CHKERRQ(ierr);
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "AODataGetInfoKeyOwnership"
/*@
   AODataGetInfoKeyOwnership - Gets the ownership range to this key type.

   Input Parameters:
.  aodata - the database
.  name - the name of the key

   Output Parameters:
.  rstart - first key owned locally
.  rend - last key owned locally

.keywords: database accessing

.seealso:
@*/
int AODataGetInfoKeyOwnership(AOData aodata,char *name,int *rstart,int *rend)
{
  int key,ierr,flag;
  PetscValidHeaderSpecific(aodata,AODATA_COOKIE);

  ierr = AODataFindKey_Private(aodata,name,&flag,&key); CHKERRQ(ierr);
  if (flag) SETERRQ(1,1,"Key never created");

  *rstart = aodata->keys[key].rstart;
  *rend   = aodata->keys[key].rend;

  return 0;
}

#undef __FUNC__  
#define __FUNC__ "AODataGetInfoKey"
/*@
   AODataGetInfoKey - Gets the global size, local size and number of segments in a key.

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
int AODataGetInfoKey(AOData aodata,char *name,int *nglobal,int *nlocal,int *nsegments)
{
  int key,ierr,flag;
  PetscValidHeaderSpecific(aodata,AODATA_COOKIE);

  ierr = AODataFindKey_Private(aodata,name,&flag,&key); CHKERRQ(ierr);
  if (flag) SETERRQ(1,1,"Key never created");

  if (nglobal)   *nglobal   = aodata->keys[key].N;
  if (nlocal)    *nlocal    = aodata->keys[key].nlocal;
  if (nsegments) *nsegments = aodata->keys[key].nsegments;

  return 0;
}

#undef __FUNC__  
#define __FUNC__ "AODataGetInfoSegment"
/*@
   AODataGetInfoSegment - Gets the global size, local size, blocksize and type of a data segment

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
int AODataGetInfoSegment(AOData aodata,char *keyname,char *segname,int *nglobal,int *nlocal,
                         int *bs, PetscDataType *dtype)
{
  int key,ierr,flag,seg;
  PetscValidHeaderSpecific(aodata,AODATA_COOKIE);

  ierr = AODataFindSegment_Private(aodata,keyname,segname,&flag,&key,&seg); CHKERRQ(ierr);
  if (flag) SETERRQ(1,1,"Key or segment never created");
  if (nglobal)   *nglobal   = aodata->keys[key].N;
  if (nlocal)    *nlocal    = aodata->keys[key].nlocal;
  if (bs)        *bs        = aodata->keys[key].segments[seg].bs;
  if (dtype)     *dtype     = aodata->keys[key].segments[seg].datatype;

  return 0;
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
  PetscValidHeaderSpecific(aodata,AODATA_COOKIE);
  return (*aodata->view)((PetscObject)aodata,viewer);
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
  if (!aodata) return 0;
  PetscValidHeaderSpecific(aodata,AODATA_COOKIE);
  if (--aodata->refct > 0) return 0;
  return (*aodata->destroy)((PetscObject)aodata);
}





