
#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: aodata.c,v 1.4 1997/10/01 22:47:36 bsmith Exp bsmith $";
#endif
/*  
   Defines the abstract operations on AOData
*/
#include "src/ao/aoimpl.h"      /*I "ao.h" I*/

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
  return (*aodata->ops.getsegment)(aodata,name,n,keys,data);
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
  return (*aodata->ops.restore)(aodata,name,segment,n,keys,data);
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
#define __FUNC__ "AODataRestoreSegment" 
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

  ierr = (*aodata->ops.restore)(aodata,name,n,keys,data); CHKERRQ(ierr);
  ierr = ISRestoreIndices(is,&keys); CHKERRQ(ierr);
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
  
  int i;
  PetscValidHeaderSpecific(aodata,AODATA_COOKIE);
  if (aodata->nc == aodata->nsegments) SETERRQ(1,1,"Data object already full");

  for ( i=0; i<aodata->nsegments; i++ ) {
    if (!PetscStrcmp(aodata->segments[i].name,name)) {
      SETERRQ(1,1,"Segment name already used");
    }
  }
  return (*aodata->ops.addsegment)(aodata,name,bs,n,keys,data,dtype);
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
  ierr = (*aodata->ops.addsegment)(aodata,name,bs,n,keys,data,dtype); CHKERRQ(ierr);
  ierr = ISRestoreIndices(is,&keys); CHKERRQ(ierr);
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "AODataViewGetSize" 
/*@
   AODataGetInfo - Gets the global size, blocksize and type of a data segment

   Input Parameters:
.  aodata - the database
.  name - the name of the segment

   Output Parameters:
.  bs - the blocksize
.  n  - the number of blocks
.  dtype - the datatype


.keywords: database accessing

.seealso:
@*/
int AODataGetInfo(AOData aodata,char *name,int *bs, int *n, PetscDataType *dtype)
{
  int i;
  PetscValidHeaderSpecific(aodata,AODATA_COOKIE);

  for ( i=0; i<aodata->nsegments; i++ ) {
    if (!PetscStrcmp(name,aodata->segments[i].name)) break;
  }
  if (i == aodata->nsegments) SETERRQ(1,1,"Unknown segment name");
  if (bs)    *bs    = aodata->segments[i].bs;
  if (n)     *n     = aodata->segments[i].N;
  if (dtype) *dtype = aodata->segments[i].datatype;
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





