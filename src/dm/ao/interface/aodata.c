#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: aodata.c,v 1.34 1999/04/19 22:16:56 bsmith Exp balay $";
#endif
/*  
   Defines the abstract operations on AOData
*/
#include "src/dm/ao/aoimpl.h"      /*I "ao.h" I*/

#undef __FUNC__  
#define __FUNC__ "AODataGetInfo" 
/*@C
    AODataGetInfo - Gets the number of keys and their names in a database.

    Not collective

    Input Parameter:
.   ao - the AOData database

    Output Parameters:
+   nkeys - the number of keys
-   keys - the names of the keys (or PETSC_NULL)

   Level: advanced

.keywords: application ordering

.seealso:  AODataSegmentGetInfo()
@*/ 
int AODataGetInfo(AOData ao,int *nkeys,char ***keys)
{
  int       n,i;
  AODataKey *key = ao->keys;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ao,AODATA_COOKIE);

  *nkeys = n = ao->nkeys;
  if (keys) {
    *keys = (char **) PetscMalloc((n+1)*sizeof(char *));CHKPTRQ(keys);
    for ( i=0; i<n; i++ ) {
      if (!key) SETERRQ(PETSC_ERR_COR,1,"Less keys in database then indicated");
      (*keys)[i] = key->name;
      key        = key->next;
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "AODataKeyFind_Private" 
/*
   AODataKeyFind_Private - Given a keyname  finds the key. Generates a flag if not found.

   Not collective

   Input Parameters:
.  keyname - string name of key

   Output Parameter:
+  flag - 1 if found, 0 if not found
-  key - the associated key

   Level: advanced

*/
int AODataKeyFind_Private(AOData aodata,char *keyname, int *flag,AODataKey **key)
{
  PetscFunctionBegin;
  *key   = aodata->keys;
  *flag  = 0;
  while (*key) {
    if (!PetscStrcmp((*key)->name,keyname)) {
       /* found the key */
       *flag  = 1;
       PetscFunctionReturn(0);
    }
    if (!(*key)->next) {
       *flag  = 0;
       PetscFunctionReturn(0);
    }
    *key = (*key)->next;
  } 
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "AODataKeyExists" 
/*@C
   AODataKeyExists - Determines if a key exists in the database.

   Not collective

   Input Parameters:
.  keyname - string name of key

   Output Parameter:
.  flag - PETSC_TRUE if found, otherwise PETSC_FALSE

   Level: advanced

@*/
int AODataKeyExists(AOData aodata,char *keyname, PetscTruth *flag)
{
  int       ierr,iflag;
  AODataKey *ikey;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(aodata,AODATA_COOKIE);
  ierr = AODataKeyFind_Private(aodata,keyname,&iflag,&ikey);CHKERRQ(ierr);
  if (iflag) *flag = PETSC_TRUE;
  else       *flag = PETSC_FALSE;
  PetscFunctionReturn(0);
}


#undef __FUNC__  
#define __FUNC__ "AODataSegmentFind_Private" 
/*
   AODataSegmentFind_Private - Given a key and segment finds the int key, segment
   coordinates. Generates a flag if not found.

   Not collective

   Input Parameters:
+  keyname - string name of key
-  segname - string name of segment

   Output Parameter:
+  flag - 1 if found, 0 if key but no segment, -1 if no key no segment
.  key - integer of keyname
-  segment - integer of segment

   Level: advanced

*/
int AODataSegmentFind_Private(AOData aodata,char *keyname, char *segname, int *flag,AODataKey **key,
                              AODataSegment **seg)
{
  int  ierr,keyflag;

  PetscFunctionBegin;
  ierr = AODataKeyFind_Private(aodata,keyname,&keyflag,key);CHKERRQ(ierr);
  if (keyflag) { /* found key now look for flag */
    *seg = (*key)->segments;
    while(*seg) {
      if (!PetscStrcmp((*seg)->name,segname)) {
        /* found the segment */
        *flag  = 1;
        PetscFunctionReturn(0);
      }
      if (!(*seg)->next) {
         *flag  = 0;
         PetscFunctionReturn(0);
      }
      *seg = (*seg)->next;
    }
    *flag = 0;
    PetscFunctionReturn(0);
  } 
  *flag = -1;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "AODataSegmentExists" 
/*@C
   AODataSegmentExists - Determines if a key  and segment exists in the database.

   Not collective

   Input Parameters:
+  keyname - string name of key
-  segname - string name of segment

   Output Parameter:
.  flag - PETSC_TRUE if found, else PETSC_FALSE

   Level: advanced

@*/
int AODataSegmentExists(AOData aodata,char *keyname, char *segname,PetscTruth *flag)
{
  int           ierr,iflag;
  AODataKey     *ikey;
  AODataSegment *iseg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(aodata,AODATA_COOKIE);
  ierr = AODataSegmentFind_Private(aodata,keyname,segname,&iflag,&ikey,&iseg);CHKERRQ(ierr);
  if (iflag == 1) *flag = PETSC_TRUE;
  else            *flag = PETSC_FALSE;
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------------------------------*/

#undef __FUNC__  
#define __FUNC__ "AODataKeyGetActive" 
/*@C
   AODataKeyGetActive - Get a sublist of key indices that have a logical flag on.

   Collective on AOData

   Input Parameters:
+  aodata - the database
.  name - the name of the key
.  segment - the name of the segment
.  n - the number of key indices provided by this processor
.  keys - the keys provided by this processor
-  wl - which logical key in the block (for block size 1 this is always 0)

   Output Parameters:
.  IS - the list of key indices

   Level: advanced

.keywords: database transactions

.seealso: AODataCreateBasic(), AODataDestroy(), AODataKeyAdd(), AODataSegmentRestore(),
          AODataSegmentGetIS(), AODataSegmentRestoreIS(), AODataSegmentAdd(), 
          AODataKeyGetInfo(), AODataSegmentGetInfo(), AODataSegmentAdd()
@*/
int AODataKeyGetActive(AOData aodata,char *name,char *segment,int n,int *keys,int wl,IS *is)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(aodata,AODATA_COOKIE);
  ierr = (*aodata->ops->keygetactive)(aodata,name,segment,n,keys,wl,is);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "AODataKeyGetActiveIS" 
/*@C
   AODataKeyGetActiveIS - Get a sublist of key indices that have a logical flag on.

   Collective on AOData

   Input Parameters:
+  aodata - the database
.  name - the name of the key
.  segment - the name of the segment
.  in - the key indices we are checking
-  wl - which logical key in the block (for block size 1 this is always 0)

   Output Parameters:
.  IS - the list of key indices

   Level: advanced

.keywords: database transactions

.seealso: AODataCreateBasic(), AODataDestroy(), AODataKeyAdd(), AODataSegmentRestore(),
          AODataSegmentGetIS(), AODataSegmentRestoreIS(), AODataSegmentAdd(), 
          AODataKeyGetInfo(), AODataSegmentGetInfo(), AODataSegmentAdd()
@*/
int AODataKeyGetActiveIS(AOData aodata,char *name,char *segname,IS in,int wl,IS *is)
{
  int ierr,n,*keys;

  PetscFunctionBegin;
  ierr = ISGetSize(in,&n);CHKERRQ(ierr);
  ierr = ISGetIndices(in,&keys);CHKERRQ(ierr);
  ierr = AODataKeyGetActive(aodata,name,segname,n,keys,wl,is);CHKERRQ(ierr);
  ierr = ISRestoreIndices(in,&keys);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "AODataKeyGetActiveLocal" 
/*@C
   AODataKeyGetActiveLocal - Get a sublist of key indices that have a logical flag on.

   Collective on AOData

   Input Parameters:
+  aodata - the database
.  name - the name of the key
.  segment - the name of the segment
.  n - the number of key indices provided by this processor
.  keys - the keys provided by this processor
-  wl - which logical key in the block (for block size 1 this is always 0)

   Output Parameters:
.  IS - the list of key indices

   Level: advanced

.keywords: database transactions

.seealso: AODataCreateBasic(), AODataDestroy(), AODataKeyAdd(), AODataSegmentRestore(),
          AODataSegmentGetIS(), AODataSegmentRestoreIS(), AODataSegmentAdd(), 
          AODataKeyGetInfo(), AODataSegmentGetInfo(), AODataSegmentAdd()
@*/
int AODataKeyGetActiveLocal(AOData aodata,char *name,char *segment,int n,int *keys,int wl,IS *is)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(aodata,AODATA_COOKIE);
  ierr = (*aodata->ops->keygetactivelocal)(aodata,name,segment,n,keys,wl,is);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "AODataKeyGetActiveLocalIS" 
/*@C
   AODataKeyGetActiveLocalIS - Get a sublist of key indices that have a logical flag on.

   Collective on AOData

   Input Parameters:
+  aodata - the database
.  name - the name of the key
.  segment - the name of the segment
.  in - the key indices we are checking
-  wl - which logical key in the block (for block size 1 this is always 0)

   Output Parameters:
.  IS - the list of key indices

   Level: advanced

.keywords: database transactions

.seealso: AODataCreateBasic(), AODataDestroy(), AODataKeyAdd(), AODataSegmentRestore(),
          AODataSegmentGetIS(), AODataSegmentRestoreIS(), AODataSegmentAdd(), 
          AODataKeyGetInfo(), AODataSegmentGetInfo(), AODataSegmentAdd()
@*/
int AODataKeyGetActiveLocalIS(AOData aodata,char *name,char *segname,IS in,int wl,IS *is)
{
  int ierr,n,*keys;

  PetscFunctionBegin;
  ierr = ISGetSize(in,&n);CHKERRQ(ierr);
  ierr = ISGetIndices(in,&keys);CHKERRQ(ierr);
  ierr = AODataKeyGetActiveLocal(aodata,name,segname,n,keys,wl,is);CHKERRQ(ierr);
  ierr = ISRestoreIndices(in,&keys);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------------------------------*/

#undef __FUNC__  
#define __FUNC__ "AODataSegmentGet" 
/*@C
   AODataSegmentGet - Get data from a particular segment of a database.

   Collective on AOData

   Input Parameters:
+  aodata - the database
.  name - the name of the key
.  segment - the name of the segment
.  n - the number of data items needed by this processor
-  keys - the keys provided by this processor

   Output Parameters:
.  data - the actual data

   Level: advanced

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
  ierr = (*aodata->ops->segmentget)(aodata,name,segment,n,keys,data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "AODataSegmentRestore" 
/*@C
   AODataSegmentRestore - Restores data from a particular segment of a database.

   Collective on AOData

   Input Parameters:
+  aodata - the database
.  name - the name of the key
.  segment - the name of the segment
.  n - the number of data items needed by this processor
-  keys - the keys provided by this processor

   Output Parameters:
.  data - the actual data

   Level: advanced

.keywords: database transactions

.seealso: AODataSegmentRestoreIS()
@*/
int AODataSegmentRestore(AOData aodata,char *name,char *segment,int n,int *keys,void **data)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(aodata,AODATA_COOKIE);
  ierr = (*aodata->ops->segmentrestore)(aodata,name,segment,n,keys,data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "AODataSegmentGetIS" 
/*@C
   AODataSegmentGetIS - Get data from a particular segment of a database.

   Collective on AOData and IS

   Input Parameters:
+  aodata - the database
.  name - the name of the key
.  segment - the name of the segment
-  is - the keys for data requested on this processor

   Output Parameters:
.  data - the actual data

   Level: advanced

.keywords: database transactions

@*/
int AODataSegmentGetIS(AOData aodata,char *name,char *segment,IS is,void **data)
{
  int ierr,n,*keys;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(aodata,AODATA_COOKIE);
  PetscValidHeaderSpecific(is,IS_COOKIE);

  ierr = ISGetSize(is,&n);CHKERRQ(ierr);
  ierr = ISGetIndices(is,&keys);CHKERRQ(ierr);
  ierr = (*aodata->ops->segmentget)(aodata,name,segment,n,keys,data);CHKERRQ(ierr);
  ierr = ISRestoreIndices(is,&keys);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "AODataSegmentRestoreIS" 
/*@C
   AODataSegmentRestoreIS - Restores data from a particular segment of a database.

   Collective on AOData and IS

   Input Parameters:
+  aodata - the database
.  name - the name of the data key
.  segment - the name of the segment
-  is - the keys provided by this processor

   Output Parameters:
.  data - the actual data

   Level: advanced

.keywords: database transactions

.seealso: AODataSegmentRestore()
@*/
int AODataSegmentRestoreIS(AOData aodata,char *name,char *segment,IS is,void **data)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(aodata,AODATA_COOKIE);

  ierr = (*aodata->ops->segmentrestore)(aodata,name,segment,0,0,data);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------------------------------*/
#undef __FUNC__  
#define __FUNC__ "AODataSegmentGetLocal" 
/*@C
   AODataSegmentGetLocal - Get data from a particular segment of a database. Returns the 
   values in the local numbering; valid only for integer segments.

   Collective on AOData

   Input Parameters:
+  aodata - the database
.  name - the name of the key
.  segment - the name of the segment
.  n - the number of data items needed by this processor
-  keys - the keys provided by this processor in local numbering

   Output Parameters:
.  data - the actual data

   Level: advanced

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
  ierr = (*aodata->ops->segmentgetlocal)(aodata,name,segment,n,keys,data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "AODataSegmentRestoreLocal" 
/*@C
   AODataSegmentRestoreLocal - Restores data from a particular segment of a database.

   Collective on AOData

   Input Parameters:
+  aodata - the database
.  name - the name of the key
.  segment - the name of the segment
.  n - the number of data items needed by this processor
-  keys - the keys provided by this processor

   Output Parameters:
.  data - the actual data

   Level: advanced

.keywords: database transactions

@*/
int AODataSegmentRestoreLocal(AOData aodata,char *name,char *segment,int n,int *keys,void **data)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(aodata,AODATA_COOKIE);
  ierr = (*aodata->ops->segmentrestorelocal)(aodata,name,segment,n,keys,data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "AODataSegmentGetLocalIS" 
/*@C
   AODataSegmentGetLocalIS - Get data from a particular segment of a database. Returns the 
   values in the local numbering; valid only for integer segments.

   Collective on AOData and IS

   Input Parameters:
+  aodata - the database
.  name - the name of the key
.  segment - the name of the segment
-  is - the keys for data requested on this processor

   Output Parameters:
.  data - the actual data

   Level: advanced

.keywords: database transactions

.seealso: AODataSegmentRestoreLocalIS()
@*/
int AODataSegmentGetLocalIS(AOData aodata,char *name,char *segment,IS is,void **data)
{
  int ierr,n,*keys;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(aodata,AODATA_COOKIE);
  PetscValidHeaderSpecific(is,IS_COOKIE);

  ierr = ISGetSize(is,&n);CHKERRQ(ierr);
  ierr = ISGetIndices(is,&keys);CHKERRQ(ierr);
  ierr = (*aodata->ops->segmentgetlocal)(aodata,name,segment,n,keys,data);CHKERRQ(ierr);
  ierr = ISRestoreIndices(is,&keys);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "AODataSegmentRestoreLocalIS" 
/*@C
   AODataSegmentRestoreLocalIS - Restores data from a particular segment of a database.

   Collective on AOData and IS

   Input Parameters:
+  aodata - the database
.  name - the name of the data key
.  segment - the name of the segment
-  is - the keys provided by this processor

   Output Parameters:
.  data - the actual data

   Level: advanced

.keywords: database transactions

.seealso: AODataSegmentGetLocalIS()
@*/
int AODataSegmentRestoreLocalIS(AOData aodata,char *name,char *segment,IS is,void **data)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(aodata,AODATA_COOKIE);
  ierr = (*aodata->ops->segmentrestorelocal)(aodata,name,segment,0,0,data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------------------------------*/

#undef __FUNC__  
#define __FUNC__ "AODataKeyGetNeighbors" 
/*@C
   AODataKeyGetNeighbors - Given a list of keys generates a new list containing
   those keys plus neighbors found in a neighbors list.

   Collective on AOData

   Input Parameters:
+  aodata - the database
.  name - the name of the key
.  n - the number of data items needed by this processor
-  keys - the keys provided by this processor

   Output Parameters:
.  is - the indices retrieved

   Level: advanced

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
/*@C
   AODataKeyGetNeighborsIS - Given a list of keys generates a new list containing
   those keys plus neighbors found in a neighbors list.

   Collective on AOData and IS

   Input Parameters:
+  aodata - the database
.  name - the name of the key
.  n - the number of data items needed by this processor
-  keys - the keys provided by this processor

   Output Parameters:
.  is - the indices retrieved

   Level: advanced

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
/*@C
   AODataSegmentGetReduced - Gets the unique list of segment values, by removing 
   duplicates.

   Collective on AOData and IS

   Input Parameters:
+  aodata - the database
.  name - the name of the key
.  segment - the name of the segment
.  n - the number of data items needed by this processor
-  keys - the keys provided by this processor

   Output Parameters:
.  is - the indices retrieved

   Level: advanced

   Example:
.vb
                      keys    ->      0  1  2  3  4   5  6  7
      if the segment contains ->      1  2  1  3  1   4  2  0
   and you request keys 0 1 2 5 7 it will return 1 2 4 0
.ve

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
  ierr = (*aodata->ops->segmentgetreduced)(aodata,name,segment,n,keys,is);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "AODataSegmentGetExtrema" 
/*@C
   AODataSegmentGetExtrema - Gets the largest and smallest values for each entry in the block

   Collective on AOData

   Input Parameters:
+  aodata - the database
.  name - the name of the key
-  segment - the name of the segment

   Output Parameters:
+  vmax - the maximum values (user must provide enough space)
-  vmin - the minimum values (user must provide enough space)

   Level: advanced

.keywords: database transactions

.seealso: AODataCreateBasic(), AODataDestroy(), AODataKeyAdd(), AODataSegmentRestore(),
          AODataSegmentGetIS(), AODataSegmentRestoreIS(), AODataSegmentAdd(), 
          AODataKeyGetInfo(), AODataSegmentGetInfo(), AODataSegmentAdd()
@*/
int AODataSegmentGetExtrema(AOData aodata,char *name,char *segment,void *vmax,void *vmin)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(aodata,AODATA_COOKIE);
  ierr = (*aodata->ops->segmentgetextrema)(aodata,name,segment,vmax,vmin);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "AODataSegmentGetReducedIS" 
/*@C
   AODataSegmentGetReducedIS -  Gets the unique list of segment values, by removing 
   duplicates.

   Collective on AOData and IS

   Input Parameters:
+  aodata - the database
.  name - the name of the key
.  segment - the name of the segment
-  is - the keys for data requested on this processor

   Output Parameters:
.  isout - the indices retreived

   Level: advanced

   Example:
.vb
                      keys    ->      0  1  2  3  4   5  6  7
      if the segment contains ->      1  2  1  3  1   4  2  0

  and you request keys 0 1 2 5 7, AODataSegmentGetReducedIS() will return 1 2 4 0
.ve

.keywords: database transactions

.seealso:
@*/
int AODataSegmentGetReducedIS(AOData aodata,char *name,char *segment,IS is,IS *isout)
{
  int ierr,n,*keys;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(aodata,AODATA_COOKIE);
  PetscValidHeaderSpecific(is,IS_COOKIE);

  ierr = ISGetSize(is,&n);CHKERRQ(ierr);
  ierr = ISGetIndices(is,&keys);CHKERRQ(ierr);
  ierr = (*aodata->ops->segmentgetreduced)(aodata,name,segment,n,keys,isout);CHKERRQ(ierr);
  ierr = ISRestoreIndices(is,&keys);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------------------------------*/

#undef __FUNC__  
#define __FUNC__ "AODataKeySetLocalToGlobalMapping" 
/*@C
   AODataKeySetLocalToGlobalMapping - Add another data key to a AOData database.

   Not collective

   Input Parameters:
+  aodata - the database
.   name - the name of the key
-  map - local to global mapping

   Level: advanced

.keywords: database additions

.seealso:
@*/
int AODataKeySetLocalToGlobalMapping(AOData aodata,char *name,ISLocalToGlobalMapping map)
{
  int       ierr,flag;
  AODataKey *ikey;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(aodata,AODATA_COOKIE);

  ierr = AODataKeyFind_Private(aodata,name,&flag,&ikey);CHKERRQ(ierr);
  if (!flag)  SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,1,"Key does not exist");

  ikey->ltog = map;
  PetscObjectReference((PetscObject) map);

  PetscFunctionReturn(0);

}

#undef __FUNC__  
#define __FUNC__ "AODataKeyGetLocalToGlobalMapping" 
/*@C
   AODataKeyGetLocalToGlobalMapping - Add another data key to a AOData database.

   Not collective

   Input Parameters:
+  aodata - the database
-  name - the name of the key

   Output Parameters:
.  map - local to global mapping

   Level: advanced

.keywords: database additions

.seealso:
@*/
int AODataKeyGetLocalToGlobalMapping(AOData aodata,char *name,ISLocalToGlobalMapping *map)
{
  int       ierr,flag;
  AODataKey *ikey;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(aodata,AODATA_COOKIE);

  ierr = AODataKeyFind_Private(aodata,name,&flag,&ikey);CHKERRQ(ierr);
  if (!flag)  SETERRQ1(PETSC_ERR_ARG_OUTOFRANGE,1,"Key does not exist: %s",name);

  *map = ikey->ltog;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "AODataKeyRemove" 
/*@C
   AODataKeyRemove - Remove a data key from a AOData database.

   Collective on AOData

   Input Parameters:
+  aodata - the database
-  name - the name of the key

   Level: advanced

.keywords: database removal

.seealso:
@*/
int AODataKeyRemove(AOData aodata,char *name)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(aodata,AODATA_COOKIE);
  ierr = (*aodata->ops->keyremove)(aodata,name);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "AODataSegmentRemove" 
/*@C
   AODataSegmentRemove - Remove a data segment from a AOData database.

   Collective on AOData

   Input Parameters:
+  aodata - the database
.  name - the name of the key
-  segname - name of the segment

   Level: advanced

.keywords: database removal

.seealso:
@*/
int AODataSegmentRemove(AOData aodata,char *name,char *segname)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(aodata,AODATA_COOKIE);
  ierr = (*aodata->ops->segmentremove)(aodata,name,segname);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "AODataKeyAdd" 
/*@C
   AODataKeyAdd - Add another data key to a AOData database.

   Collective on AOData

   Input Parameters:
+  aodata - the database
.  name - the name of the key
.  N - the number of indices in the key
-  nlocal - number of indices to be associated with this processor

   Level: advanced

.keywords: database additions

.seealso:
@*/
int AODataKeyAdd(AOData aodata,char *name,int nlocal,int N)
{
  int       ierr,flag,size,rank,i,len;
  AODataKey *key,*oldkey;
  MPI_Comm  comm = aodata->comm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(aodata,AODATA_COOKIE);

  ierr = AODataKeyFind_Private(aodata,name,&flag,&oldkey);CHKERRQ(ierr);
  if (flag == 1)  SETERRQ1(PETSC_ERR_ARG_OUTOFRANGE,1,"Key already exists with given name: %s",name);

  key                = PetscNew(AODataKey);CHKPTRQ(key);
  if (oldkey) { oldkey->next = key;} 
  else        { aodata->keys = key;} 
  len            = PetscStrlen(name);
  key->name      = (char *) PetscMalloc((len+1)*sizeof(char));CHKPTRQ(key->name);
  ierr           = PetscStrcpy(key->name,name);CHKERRQ(ierr);
  key->N         = N;
  key->nsegments = 0;
  key->segments  = 0;
  key->ltog      = 0;
  key->next      = 0;

  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);

  /*  Set nlocal and ownership ranges */
  ierr         = PetscSplitOwnership(comm,&nlocal,&N);CHKERRQ(ierr);
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
/*@C
   AODataSegmentAdd - Adds another data segment to a AOData database.

   Collective on AOData

   Input Parameters:
+  aodata  - the database
.  name    - the name of the key
.  segment - the name of the data segment
.  bs      - the fundamental blocksize of the data
.  n       - the number of data items contributed by this processor
.  keys    - the keys provided by this processor
.  data    - the actual data
-  dtype   - the data type (one of PETSC_INT, PETSC_DOUBLE, PETSC_SCALAR, etc.)

   Level: advanced

.keywords: database additions

.seealso: AODataSegmentAddIS()
@*/
int AODataSegmentAdd(AOData aodata,char *name,char *segment,int bs,int n,int *keys,void *data,
                     PetscDataType dtype)
{
  int      ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(aodata,AODATA_COOKIE);

  ierr = (*aodata->ops->segmentadd)(aodata,name,segment,bs,n,keys,data,dtype);CHKERRQ(ierr);

  /*
  ierr = OptionsHasName(PETSC_NULL,"-ao_data_view",&flg1);CHKERRQ(ierr);
  if (flg1) {
    ierr = AODataView(aodata,VIEWER_STDOUT_(comm));CHKERRQ(ierr);
  }
  ierr = OptionsHasName(PETSC_NULL,"-ao_data_view_info",&flg1);CHKERRQ(ierr);
  if (flg1) {
    ierr = ViewerPushFormat(VIEWER_STDOUT_(comm),VIEWER_FORMAT_ASCII_INFO,0);CHKERRQ(ierr);
    ierr = AODataView(aodata,VIEWER_STDOUT_(comm));CHKERRQ(ierr);
    ierr = ViewerPopFormat(VIEWER_STDOUT_(comm));CHKERRQ(ierr);
  }
  */
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "AODataSegmentAddIS" 
/*@C
   AODataSegmentAddIS - Add another data segment to a AOData database.

   Collective on AOData and IS

   Input Parameters:
+  aodata - the database
.  name - the name of the key
.  segment - name of segment
.  bs - the fundamental blocksize of the data
.  is - the keys provided by this processor
.  data - the actual data
-  dtype - the data type, one of PETSC_INT, PETSC_DOUBLE, PETSC_SCALAR, etc.

   Level: advanced

.keywords: database additions

.seealso: AODataSegmentAdd()
@*/
int AODataSegmentAddIS(AOData aodata,char *name,char *segment,int bs,IS is,void *data,
                       PetscDataType dtype)
{
  int n,*keys,ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(aodata,AODATA_COOKIE);
  PetscValidHeaderSpecific(is,IS_COOKIE);

  ierr = ISGetSize(is,&n);CHKERRQ(ierr);
  ierr = ISGetIndices(is,&keys);CHKERRQ(ierr);
  ierr = (*aodata->ops->segmentadd)(aodata,name,segment,bs,n,keys,data,dtype);CHKERRQ(ierr);
  ierr = ISRestoreIndices(is,&keys);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "AODataKeyGetOwnershipRange"
/*@C
   AODataKeyGetOwnershipRange - Gets the ownership range to this key type.

   Not collective

   Input Parameters:
+  aodata - the database
-  name - the name of the key

   Output Parameters:
+  rstart - first key owned locally
-  rend - last key owned locally

   Level: advanced

.keywords: database accessing

.seealso: AODataKeyGetInfo()
@*/
int AODataKeyGetOwnershipRange(AOData aodata,char *name,int *rstart,int *rend)
{
  int       ierr,flag;
  AODataKey *key;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(aodata,AODATA_COOKIE);

  ierr = AODataKeyFind_Private(aodata,name,&flag,&key);CHKERRQ(ierr);
  if (!flag) SETERRQ1(PETSC_ERR_ARG_OUTOFRANGE,1,"Key never created: %s",name);

  *rstart = key->rstart;
  *rend   = key->rend;

  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "AODataKeyGetInfo"
/*@C
   AODataKeyGetInfo - Gets the global size, local size and number of segments in a key.

   Not collective

   Input Parameters:
+  aodata - the database
-  name - the name of the key

   Output Parameters:
+  nglobal - global number of keys
.  nlocal - local number of keys
.  nsegments - number of segments associated with key
-  segnames - names of the segments or PETSC_NULL

   Level: advanced

.keywords: database accessing

.seealso: AODataKeyGetOwnershipRange()
@*/
int AODataKeyGetInfo(AOData aodata,char *name,int *nglobal,int *nlocal,int *nsegments,char ***segnames)
{
  int           ierr,flag,i,n=0;
  AODataKey     *key;
  AODataSegment *seg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(aodata,AODATA_COOKIE);

  ierr = AODataKeyFind_Private(aodata,name,&flag,&key);CHKERRQ(ierr);
  if (!flag) SETERRQ1(PETSC_ERR_ARG_OUTOFRANGE,1,"Key never created: %s",name);

  if (nglobal)   *nglobal   = key->N;
  if (nlocal)    *nlocal    = key->nlocal;
  if (nsegments) *nsegments = n = key->nsegments;
  if (nsegments && segnames) {
    *segnames = (char **) PetscMalloc((n+1)*sizeof(char *));CHKPTRQ(segnames);
    seg       = key->segments;
    for ( i=0; i<n; i++ ) {
      if (!seg) SETERRQ(PETSC_ERR_COR,1,"Less segments in database then indicated");
      (*segnames)[i] = seg->name;
      seg            = seg->next;
    }
  }

  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "AODataSegmentGetInfo"
/*@C
   AODataSegmentGetInfo - Gets the blocksize and type of a data segment

   Not collective

   Input Parameters:
+  aodata - the database
.  keyname - the name of the key
-  segname - the name of the segment

   Output Parameters:
+  bs - the blocksize
-  dtype - the datatype

   Level: advanced

.keywords: database accessing

.seealso:  AODataGetInfo()
@*/
int AODataSegmentGetInfo(AOData aodata,char *keyname,char *segname,int *bs, PetscDataType *dtype)
{
  int           ierr,flag;
  AODataKey     *key;
  AODataSegment *seg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(aodata,AODATA_COOKIE);

  ierr = AODataSegmentFind_Private(aodata,keyname,segname,&flag,&key,&seg);CHKERRQ(ierr);
  if (flag == 0) SETERRQ1(PETSC_ERR_ARG_OUTOFRANGE,1,"Segment never created: %s",segname);
  if (flag == -1) SETERRQ1(PETSC_ERR_ARG_OUTOFRANGE,1,"Key never created: %s",keyname);
  if (bs)        *bs        = seg->bs;
  if (dtype)     *dtype     = seg->datatype;

  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "AODataView" 
/*@C
   AODataView - Displays an application ordering.

   Collective on AOData and Viewer

   Input Parameters:
+  aodata - the database
-  viewer - viewer used for display

   Level: intermediate

   The available visualization contexts include
+     VIEWER_STDOUT_SELF - standard output (default)
-     VIEWER_STDOUT_WORLD - synchronized standard
         output where only the first processor opens
         the file.  All other processors send their 
         data to the first processor to print. 

   The user can open an alternative visualization context with
   ViewerASCIIOpen() - output to a specified file.


.keywords: database viewing

.seealso: ViewerASCIIOpen()
@*/
int AODataView(AOData aodata, Viewer viewer)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(aodata,AODATA_COOKIE);
  ierr = (*aodata->ops->view)(aodata,viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "AODataDestroy" 
/*@C
   AODataDestroy - Destroys an application ordering set.

   Collective on AOData

   Input Parameters:
.  aodata - the database

   Level: intermediate

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
  ierr = (*aodata->ops->destroy)(aodata);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "AODataKeyRemap" 
/*@C
   AODataKeyRemap - Remaps a key and all references to a key to a new numbering 
   scheme where each processor indicates its new nodes by listing them in the
   previous numbering scheme.

   Collective on AOData and AO

   Input Parameters:
+  aodata - the database
.  key  - the key to remap
-  ao - the old to new ordering

   Level: advanced

.keywords: database remapping

.seealso: AODataKeyGetAdjacency()
@*/
int AODataKeyRemap(AOData aodata, char *key,AO ao)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(aodata,AODATA_COOKIE);
  PetscValidHeaderSpecific(ao,AO_COOKIE);
  ierr = (*aodata->ops->keyremap)(aodata,key,ao);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "AODataKeyGetAdjacency" 
/*@C
   AODataKeyGetAdjacency - Gets the adjacency graph for a key.

   Collective on AOData

   Input Parameters:
+  aodata - the database
-  key  - the key

   Output Parameter:
.  adj - the adjacency graph

   Level: advanced

.keywords: database, adjacency graph

.seealso: AODataKeyRemap()
@*/
int AODataKeyGetAdjacency(AOData aodata, char *key,Mat *adj)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(aodata,AODATA_COOKIE);
  ierr = (*aodata->ops->keygetadjacency)(aodata,key,adj);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__
#define __FUNC__ "AODataSegmentPartition"
/*@C
    AODataSegmentPartition - Partitions a segment type across processors 
    relative to a key that is partitioned. This will try to keep as
    many elements of the segment on the same processor as corresponding
    neighboring key elements are.

    Collective on AOData

    Input Parameters:
+   aodata - the database
-   key - the key to be partitioned and renumbered

   Level: advanced

.seealso: AODataKeyPartition()
@*/
int AODataSegmentPartition(AOData aodata,char *key,char *seg)
{
  int             ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(aodata,AODATA_COOKIE);
  ierr = (*aodata->ops->segmentpartition)(aodata,key,seg);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


