#define PETSCDM_DLL

/*  
   Defines the abstract operations on AOData
*/
#include "src/dm/ao/aoimpl.h"      /*I "petscao.h" I*/


#undef __FUNCT__  
#define __FUNCT__ "AODataGetInfo" 
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
PetscErrorCode PETSCDM_DLLEXPORT AODataGetInfo(AOData ao,PetscInt *nkeys,char ***keys)
{
  PetscErrorCode ierr;
  PetscInt       n,i;
  AODataKey      *key = ao->keys;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ao,AODATA_COOKIE,1);

  *nkeys = n = ao->nkeys;
  if (keys) {
    ierr = PetscMalloc((n+1)*sizeof(char *),&keys);CHKERRQ(ierr);
    for (i=0; i<n; i++) {
      if (!key) SETERRQ(PETSC_ERR_COR,"Less keys in database then indicated");
      (*keys)[i] = key->name;
      key        = key->next;
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "AODataKeyFind_Private" 
/*
   AODataKeyFind_Private - Given a keyname  finds the key. Generates a flag if not found.

   Not collective

   Input Parameters:
.  keyname - string name of key

   Output Parameter:
+  flag - PETSC_TRUE if found, PETSC_FALSE if not found
-  key - the associated key

   Level: advanced

*/
PetscErrorCode AODataKeyFind_Private(AOData aodata,const char keyname[],PetscTruth *flag,AODataKey **key)
{
  PetscTruth     match;
  PetscErrorCode ierr;
  AODataAlias    *t = aodata->aliases;
  const char     *name = keyname;
  AODataKey      *nkey;

  PetscFunctionBegin;
  *key   = PETSC_NULL;
  *flag  = PETSC_FALSE;
  while (name) {
    nkey  = aodata->keys;
    while (nkey) {
      ierr = PetscStrcmp(nkey->name,name,&match);CHKERRQ(ierr);
      if (match) {
        /* found the key */
        *key   = nkey;
        *flag  = PETSC_TRUE;
        PetscFunctionReturn(0);
      }
      *key = nkey;
      nkey = nkey->next;
    }
    name = 0;
    while (t) {
      ierr = PetscStrcmp(keyname,t->alias,&match);CHKERRQ(ierr);
      if (match) {
        name = t->name;
        t    = t->next;
        break;
      }
      t = t->next;
    }
  } 
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "AODataKeyExists" 
/*@C
   AODataKeyExists - Determines if a key exists in the database.

   Not collective

   Input Parameters:
.  keyname - string name of key

   Output Parameter:
.  flag - PETSC_TRUE if found, otherwise PETSC_FALSE

   Level: advanced

@*/
PetscErrorCode PETSCDM_DLLEXPORT AODataKeyExists(AOData aodata,const char keyname[],PetscTruth *flag)
{
  PetscErrorCode ierr;
  PetscTruth     iflag;
  AODataKey      *ikey;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(aodata,AODATA_COOKIE,1);
  ierr = AODataKeyFind_Private(aodata,keyname,&iflag,&ikey);CHKERRQ(ierr);
  if (iflag) *flag = PETSC_TRUE;
  else       *flag = PETSC_FALSE;
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "AODataSegmentFind_Private" 
/*
   AODataSegmentFind_Private - Given a key and segment finds the int key, segment
   coordinates. Generates a flag if not found.

   Not collective

   Input Parameters:
+  keyname - string name of key
-  segname - string name of segment

   Output Parameter:
+  flag - PETSC_TRUE if found, PETSC_FALSE if not
.  key - integer of keyname
-  segment - integer of segment

   If it doesn't find it, it returns the last seg in the key (if the key exists)

   Level: advanced

*/
PetscErrorCode AODataSegmentFind_Private(AOData aodata,const char keyname[],const char segname[],PetscTruth *flag,AODataKey **key,AODataSegment **seg)
{
  PetscErrorCode ierr;
  PetscTruth     keyflag,match;
  AODataAlias    *t = aodata->aliases;
  const char     *name;
  AODataSegment  *nseg;

  PetscFunctionBegin;
  *seg  = PETSC_NULL;
  *flag = PETSC_FALSE;
  ierr  = AODataKeyFind_Private(aodata,keyname,&keyflag,key);CHKERRQ(ierr);
  if (keyflag) { /* found key now look for segment */
    name = segname;
    while (name) {
      nseg = (*key)->segments;
      while (nseg) {
        ierr = PetscStrcmp(nseg->name,name,&match);CHKERRQ(ierr);
        if (match) {
          /* found the segment */
          *seg   = nseg;
          *flag  = PETSC_TRUE;
          PetscFunctionReturn(0);
        }
        *seg = nseg;
        nseg = nseg->next;
      }
      name = 0;
      while (t) {
        ierr = PetscStrcmp(segname,t->alias,&match);CHKERRQ(ierr);
        if (match) {
          name = t->name;
          t    = t->next;
          break;
        }
        t = t->next;
      }
    }
  } 
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "AODataSegmentExists" 
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
PetscErrorCode PETSCDM_DLLEXPORT AODataSegmentExists(AOData aodata,const char keyname[],const char segname[],PetscTruth *flag)
{
  PetscErrorCode ierr;
  PetscTruth    iflag;
  AODataKey     *ikey;
  AODataSegment *iseg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(aodata,AODATA_COOKIE,1);
  ierr = AODataSegmentFind_Private(aodata,keyname,segname,&iflag,&ikey,&iseg);CHKERRQ(ierr);
  if (iflag) *flag = PETSC_TRUE;
  else       *flag = PETSC_FALSE;
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------------------------------*/

#undef __FUNCT__  
#define __FUNCT__ "AODataKeyGetActive" 
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
.  is - the list of key indices

   Level: advanced

.keywords: database transactions

.seealso: AODataCreateBasic(), AODataDestroy(), AODataKeyAdd(), AODataSegmentRestore(),
          AODataSegmentGetIS(), AODataSegmentRestoreIS(), AODataSegmentAdd(), 
          AODataKeyGetInfo(), AODataSegmentGetInfo(), AODataSegmentAdd()
@*/
PetscErrorCode PETSCDM_DLLEXPORT AODataKeyGetActive(AOData aodata,const char name[],const char segment[],PetscInt n,PetscInt *keys,PetscInt wl,IS *is)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(aodata,AODATA_COOKIE,1);
  ierr = (*aodata->ops->keygetactive)(aodata,name,segment,n,keys,wl,is);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "AODataKeyGetActiveIS" 
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
PetscErrorCode PETSCDM_DLLEXPORT AODataKeyGetActiveIS(AOData aodata,const char name[],const char segname[],IS in,PetscInt wl,IS *is)
{
  PetscErrorCode ierr;
  PetscInt       n,*keys;

  PetscFunctionBegin;
  ierr = ISGetLocalSize(in,&n);CHKERRQ(ierr);
  ierr = ISGetIndices(in,&keys);CHKERRQ(ierr);
  ierr = AODataKeyGetActive(aodata,name,segname,n,keys,wl,is);CHKERRQ(ierr);
  ierr = ISRestoreIndices(in,&keys);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "AODataKeyGetActiveLocal" 
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
PetscErrorCode PETSCDM_DLLEXPORT AODataKeyGetActiveLocal(AOData aodata,const char name[],const char segment[],PetscInt n,PetscInt *keys,PetscInt wl,IS *is)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(aodata,AODATA_COOKIE,1);
  ierr = (*aodata->ops->keygetactivelocal)(aodata,name,segment,n,keys,wl,is);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "AODataKeyGetActiveLocalIS" 
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
PetscErrorCode PETSCDM_DLLEXPORT AODataKeyGetActiveLocalIS(AOData aodata,const char name[],const char segname[],IS in,PetscInt wl,IS *is)
{
  PetscErrorCode ierr;
  PetscInt       n,*keys;

  PetscFunctionBegin;
  ierr = ISGetLocalSize(in,&n);CHKERRQ(ierr);
  ierr = ISGetIndices(in,&keys);CHKERRQ(ierr);
  ierr = AODataKeyGetActiveLocal(aodata,name,segname,n,keys,wl,is);CHKERRQ(ierr);
  ierr = ISRestoreIndices(in,&keys);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------------------------------*/

#undef __FUNCT__  
#define __FUNCT__ "AODataSegmentGet" 
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
PetscErrorCode PETSCDM_DLLEXPORT AODataSegmentGet(AOData aodata,const char name[],const char segment[],PetscInt n,PetscInt *keys,void **data)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(aodata,AODATA_COOKIE,1);
  ierr = (*aodata->ops->segmentget)(aodata,name,segment,n,keys,data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "AODataSegmentRestore" 
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
PetscErrorCode PETSCDM_DLLEXPORT AODataSegmentRestore(AOData aodata,const char name[],const char segment[],PetscInt n,PetscInt *keys,void **data)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(aodata,AODATA_COOKIE,1);
  ierr = (*aodata->ops->segmentrestore)(aodata,name,segment,n,keys,data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "AODataSegmentGetIS" 
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
PetscErrorCode PETSCDM_DLLEXPORT AODataSegmentGetIS(AOData aodata,const char name[],const char segment[],IS is,void **data)
{
  PetscErrorCode ierr;
  PetscInt       n,*keys;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(aodata,AODATA_COOKIE,1);
  PetscValidHeaderSpecific(is,IS_COOKIE,4);

  ierr = ISGetLocalSize(is,&n);CHKERRQ(ierr);
  ierr = ISGetIndices(is,&keys);CHKERRQ(ierr);
  ierr = (*aodata->ops->segmentget)(aodata,name,segment,n,keys,data);CHKERRQ(ierr);
  ierr = ISRestoreIndices(is,&keys);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "AODataSegmentRestoreIS" 
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
PetscErrorCode PETSCDM_DLLEXPORT AODataSegmentRestoreIS(AOData aodata,const char name[],const char segment[],IS is,void **data)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(aodata,AODATA_COOKIE,1);

  ierr = (*aodata->ops->segmentrestore)(aodata,name,segment,0,0,data);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------------------------------*/
#undef __FUNCT__  
#define __FUNCT__ "AODataSegmentGetLocal" 
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
PetscErrorCode PETSCDM_DLLEXPORT AODataSegmentGetLocal(AOData aodata,const char name[],const char segment[],PetscInt n,PetscInt *keys,void **data)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(aodata,AODATA_COOKIE,1);
  ierr = (*aodata->ops->segmentgetlocal)(aodata,name,segment,n,keys,data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "AODataSegmentRestoreLocal" 
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
PetscErrorCode PETSCDM_DLLEXPORT AODataSegmentRestoreLocal(AOData aodata,const char name[],const char segment[],PetscInt n,PetscInt *keys,void **data)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(aodata,AODATA_COOKIE,1);
  ierr = (*aodata->ops->segmentrestorelocal)(aodata,name,segment,n,keys,data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "AODataSegmentGetLocalIS" 
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
PetscErrorCode PETSCDM_DLLEXPORT AODataSegmentGetLocalIS(AOData aodata,const char name[],const char segment[],IS is,void **data)
{
  PetscErrorCode ierr;
  PetscInt       n,*keys;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(aodata,AODATA_COOKIE,1);
  PetscValidHeaderSpecific(is,IS_COOKIE,4);

  ierr = ISGetLocalSize(is,&n);CHKERRQ(ierr);
  ierr = ISGetIndices(is,&keys);CHKERRQ(ierr);
  ierr = (*aodata->ops->segmentgetlocal)(aodata,name,segment,n,keys,data);CHKERRQ(ierr);
  ierr = ISRestoreIndices(is,&keys);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "AODataSegmentRestoreLocalIS" 
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
PetscErrorCode PETSCDM_DLLEXPORT AODataSegmentRestoreLocalIS(AOData aodata,const char name[],const char segment[],IS is,void **data)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(aodata,AODATA_COOKIE,1);
  PetscValidHeaderSpecific(is,IS_COOKIE,4);
  ierr = (*aodata->ops->segmentrestorelocal)(aodata,name,segment,0,0,data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------------------------------*/

#undef __FUNCT__  
#define __FUNCT__ "AODataKeyGetNeighbors" 
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
PetscErrorCode PETSCDM_DLLEXPORT AODataKeyGetNeighbors(AOData aodata,const char name[],PetscInt n,PetscInt *keys,IS *is)
{
  PetscErrorCode ierr;
  IS             reduced,input;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(aodata,AODATA_COOKIE,1);
 
  /* get the list of neighbors */
  ierr = AODataSegmentGetReduced(aodata,name,name,n,keys,&reduced);CHKERRQ(ierr);

  ierr = ISCreateGeneral(aodata->comm,n,keys,&input);CHKERRQ(ierr);
  ierr = ISExpand(input,reduced,is);CHKERRQ(ierr);
  ierr = ISDestroy(input);CHKERRQ(ierr);
  ierr = ISDestroy(reduced);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "AODataKeyGetNeighborsIS" 
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
PetscErrorCode PETSCDM_DLLEXPORT AODataKeyGetNeighborsIS(AOData aodata,const char name[],IS keys,IS *is)
{
  PetscErrorCode ierr;
  IS             reduced;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(aodata,AODATA_COOKIE,1);
 
  /* get the list of neighbors */
  ierr = AODataSegmentGetReducedIS(aodata,name,name,keys,&reduced);CHKERRQ(ierr);
  /* combine keys and reduced is */
  ierr = ISExpand(keys,reduced,is);CHKERRQ(ierr);
  ierr = ISDestroy(reduced);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "AODataSegmentGetReduced" 
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
PetscErrorCode PETSCDM_DLLEXPORT AODataSegmentGetReduced(AOData aodata,const char name[],const char segment[],PetscInt n,PetscInt *keys,IS *is)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(aodata,AODATA_COOKIE,1);
  ierr = (*aodata->ops->segmentgetreduced)(aodata,name,segment,n,keys,is);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "AODataSegmentGetExtrema" 
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
PetscErrorCode PETSCDM_DLLEXPORT AODataSegmentGetExtrema(AOData aodata,const char name[],const char segment[],void *vmax,void *vmin)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(aodata,AODATA_COOKIE,1);
  ierr = (*aodata->ops->segmentgetextrema)(aodata,name,segment,vmax,vmin);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "AODataSegmentGetReducedIS" 
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
PetscErrorCode PETSCDM_DLLEXPORT AODataSegmentGetReducedIS(AOData aodata,const char name[],const char segment[],IS is,IS *isout)
{
  PetscErrorCode ierr;
  PetscInt       n,*keys;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(aodata,AODATA_COOKIE,1);
  PetscValidHeaderSpecific(is,IS_COOKIE,4);

  ierr = ISGetLocalSize(is,&n);CHKERRQ(ierr);
  ierr = ISGetIndices(is,&keys);CHKERRQ(ierr);
  ierr = (*aodata->ops->segmentgetreduced)(aodata,name,segment,n,keys,isout);CHKERRQ(ierr);
  ierr = ISRestoreIndices(is,&keys);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------------------------------*/

#undef __FUNCT__  
#define __FUNCT__ "AODataKeySetLocalTolGobalMapping" 
/*@C
   AODataKeySetLocalToGlobalMapping - Add a local to global mapping for a key in the 
     in the database

   Not collective

   Input Parameters:
+  aodata - the database
.   name - the name of the key
-  map - local to global mapping

   Level: advanced

.keywords: database additions

.seealso: AODataKeyGetLocalToGlobalMapping()
@*/
PetscErrorCode PETSCDM_DLLEXPORT AODataKeySetLocalToGlobalMapping(AOData aodata,const char name[],ISLocalToGlobalMapping map)
{
  PetscErrorCode ierr;
  PetscTruth     flag;
  AODataKey      *ikey;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(aodata,AODATA_COOKIE,1);

  ierr = AODataKeyFind_Private(aodata,name,&flag,&ikey);CHKERRQ(ierr);
  if (!flag)  SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Key does not exist");

  if (ikey->ltog) {
    SETERRQ1(PETSC_ERR_ARG_WRONGSTATE,"Database key %s already has local to global mapping",name);
  }

  ikey->ltog = map;
  ierr = PetscObjectReference((PetscObject)map);CHKERRQ(ierr);

  PetscFunctionReturn(0);

}

#undef __FUNCT__  
#define __FUNCT__ "AODataKeyGetLocalToGlobalMapping" 
/*@C
   AODataKeyGetLocalToGlobalMapping - gets a local to global mapping from a database

   Not collective

   Input Parameters:
+  aodata - the database
-  name - the name of the key

   Output Parameters:
.  map - local to global mapping

   Level: advanced

.keywords: database additions

.seealso: AODataKeySetLocalToGlobalMapping()
@*/
PetscErrorCode PETSCDM_DLLEXPORT AODataKeyGetLocalToGlobalMapping(AOData aodata,const char name[],ISLocalToGlobalMapping *map)
{
  PetscErrorCode ierr;
  PetscTruth     flag;
  AODataKey      *ikey;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(aodata,AODATA_COOKIE,1);

  ierr = AODataKeyFind_Private(aodata,name,&flag,&ikey);CHKERRQ(ierr);
  if (!flag)  SETERRQ1(PETSC_ERR_ARG_OUTOFRANGE,"Key does not exist: %s",name);

  *map = ikey->ltog;
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "AODataKeyGetOwnershipRange"
/*@C
   AODataKeyGetOwnershipRange - Gets the ownership range to this key type.

   Not collective

   Input Parameters:
+  aodata - the database
-  name - the name of the key

   Output Parameters:
+  rstart - first key owned locally (or PETSC_NULL if not needed) 
-  rend - last key owned locally + 1 (or PETSC_NULL if not needed)

   Level: advanced

.keywords: database accessing

.seealso: AODataKeyGetInfo()
@*/
PetscErrorCode PETSCDM_DLLEXPORT AODataKeyGetOwnershipRange(AOData aodata,const char name[],PetscInt *rstart,PetscInt *rend)
{
  PetscErrorCode ierr;
  PetscTruth     flag;
  AODataKey      *key;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(aodata,AODATA_COOKIE,1);

  ierr = AODataKeyFind_Private(aodata,name,&flag,&key);CHKERRQ(ierr);
  if (!flag) SETERRQ1(PETSC_ERR_ARG_OUTOFRANGE,"Key never created: %s",name);

  if (rstart) *rstart = key->rstart;
  if (rend)   *rend   = key->rend;

  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "AODataKeyGetInfo"
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
PetscErrorCode PETSCDM_DLLEXPORT AODataKeyGetInfo(AOData aodata,const char name[],PetscInt *nglobal,PetscInt *nlocal,PetscInt *nsegments,char ***segnames)
{
  PetscErrorCode ierr;
  PetscInt       i,n=0;
  AODataKey     *key;
  AODataSegment *seg;
  PetscTruth    flag;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(aodata,AODATA_COOKIE,1);

  ierr = AODataKeyFind_Private(aodata,name,&flag,&key);CHKERRQ(ierr);
  if (!flag) SETERRQ1(PETSC_ERR_ARG_OUTOFRANGE,"Key never created: %s",name);

  if (nglobal)   *nglobal   = key->N;
  if (nlocal)    *nlocal    = key->nlocal;
  if (nsegments) *nsegments = n = key->nsegments;
  if (nsegments && segnames) {
    ierr = PetscMalloc((n+1)*sizeof(char *),&segnames);CHKERRQ(ierr);
    seg  = key->segments;
    for (i=0; i<n; i++) {
      if (!seg) SETERRQ(PETSC_ERR_COR,"Less segments in database then indicated");
      (*segnames)[i] = seg->name;
      seg            = seg->next;
    }
  }

  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "AODataSegmentGetInfo"
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
PetscErrorCode PETSCDM_DLLEXPORT AODataSegmentGetInfo(AOData aodata,const char keyname[],const char segname[],PetscInt *bs,PetscDataType *dtype)
{
  PetscErrorCode ierr;
  PetscTruth    flag;
  AODataKey     *key;
  AODataSegment *seg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(aodata,AODATA_COOKIE,1);

  ierr = AODataSegmentFind_Private(aodata,keyname,segname,&flag,&key,&seg);CHKERRQ(ierr);
  if (!flag) SETERRQ1(PETSC_ERR_ARG_OUTOFRANGE,"Segment never created: %s",segname);
  if (bs)        *bs        = seg->bs;
  if (dtype)     *dtype     = seg->datatype;

  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "AODataView" 
/*@C
   AODataView - Displays an application ordering.

   Collective on AOData and PetscViewer

   Input Parameters:
+  aodata - the database
-  viewer - viewer used for display

   Level: intermediate

   The available visualization contexts include
+     PETSC_VIEWER_STDOUT_SELF - standard output (default)
-     PETSC_VIEWER_STDOUT_WORLD - synchronized standard
         output where only the first processor opens
         the file.  All other processors send their 
         data to the first processor to print. 

   The user can open an alternative visualization context with
   PetscViewerASCIIOpen() - output to a specified file.


.keywords: database viewing

.seealso: PetscViewerASCIIOpen()
@*/
PetscErrorCode PETSCDM_DLLEXPORT AODataView(AOData aodata,PetscViewer viewer)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(aodata,AODATA_COOKIE,1);
  if (!viewer) viewer = PETSC_VIEWER_STDOUT_(aodata->comm);
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_COOKIE,2);
  ierr = (*aodata->ops->view)(aodata,viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "AODataAliasDestroy_Private" 
static PetscErrorCode AODataAliasDestroy_Private(AODataAlias *aliases)
{
  AODataAlias    *t = aliases;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (t) {
    t = aliases->next;
    ierr = PetscFree(aliases->name);CHKERRQ(ierr);
    ierr = PetscFree(aliases->alias);CHKERRQ(ierr);
    ierr = PetscFree(aliases);CHKERRQ(ierr);
    while (t) {
      aliases = t;
      t       = t->next;
      ierr = PetscFree(aliases->name);CHKERRQ(ierr);
      ierr = PetscFree(aliases->alias);CHKERRQ(ierr);
      ierr    = PetscFree(aliases);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "AODataAliasAdd" 
/*@C
  AODataAliasAdd - Man page needed.
@*/
PetscErrorCode PETSCDM_DLLEXPORT AODataAliasAdd(AOData aodata,const char alias[],const char name[])
{
  AODataAlias *t = aodata->aliases;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (t) {
    while (t->next) t = t->next;
    ierr = PetscNew(AODataAlias,&t->next);CHKERRQ(ierr);
    t    = t->next;
  } else {
    ierr            = PetscNew(AODataAlias,&t);CHKERRQ(ierr);
    aodata->aliases = t;
  }
  ierr    = PetscStrallocpy(alias,&t->alias);CHKERRQ(ierr);
  ierr    = PetscStrallocpy(name,&t->name);CHKERRQ(ierr);
  t->next = 0;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "AODataDestroy" 
/*@C
   AODataDestroy - Destroys an application ordering set.

   Collective on AOData

   Input Parameters:
.  aodata - the database

   Level: intermediate

.keywords: destroy, database

.seealso: AODataCreateBasic()
@*/
PetscErrorCode PETSCDM_DLLEXPORT AODataDestroy(AOData aodata)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;

  if (!aodata) PetscFunctionReturn(0);
  PetscValidHeaderSpecific(aodata,AODATA_COOKIE,1);
  if (--aodata->refct > 0) PetscFunctionReturn(0);
  
  ierr = AODataAliasDestroy_Private(aodata->aliases);CHKERRQ(ierr);
  ierr = (*aodata->ops->destroy)(aodata);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "AODataKeyRemap" 
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
PetscErrorCode PETSCDM_DLLEXPORT AODataKeyRemap(AOData aodata,const char key[],AO ao)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(aodata,AODATA_COOKIE,1);
  PetscValidHeaderSpecific(ao,AO_COOKIE,3);
  ierr = (*aodata->ops->keyremap)(aodata,key,ao);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "AODataKeyGetAdjacency" 
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
PetscErrorCode PETSCDM_DLLEXPORT AODataKeyGetAdjacency(AOData aodata,const char key[],Mat *adj)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(aodata,AODATA_COOKIE,1);
  ierr = (*aodata->ops->keygetadjacency)(aodata,key,adj);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "AODataSegmentPartition"
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

.seealso: AODataKeyPartition(), AODataPartitionAndSetupLocal()

@*/
PetscErrorCode PETSCDM_DLLEXPORT AODataSegmentPartition(AOData aodata,const char key[],const char seg[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(aodata,AODATA_COOKIE,1);
  ierr = (*aodata->ops->segmentpartition)(aodata,key,seg);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "AODataPublish_Petsc"
PetscErrorCode AODataPublish_Petsc(PetscObject obj)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "AODataKeyRemove" 
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
PetscErrorCode PETSCDM_DLLEXPORT AODataKeyRemove(AOData aodata,const char name[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(aodata,AODATA_COOKIE,1);
  ierr = (*aodata->ops->keyremove)(aodata,name);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "AODataSegmentRemove" 
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
PetscErrorCode PETSCDM_DLLEXPORT AODataSegmentRemove(AOData aodata,const char name[],const char segname[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(aodata,AODATA_COOKIE,1);
  ierr = (*aodata->ops->segmentremove)(aodata,name,segname);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "AODataKeyAdd" 
/*@C
   AODataKeyAdd - Add another data key to a AOData database.

   Collective on AOData

   Input Parameters:
+  aodata - the database
.  name - the name of the key
.  nlocal - number of indices to be associated with this processor
-  N - the number of indices in the key

   Level: advanced

.keywords: database additions

.seealso:
@*/
PetscErrorCode PETSCDM_DLLEXPORT AODataKeyAdd(AOData aodata,const char name[],PetscInt nlocal,PetscInt N)
{
  PetscErrorCode ierr;
  PetscMPIInt    size,rank;
  PetscInt       i;
  size_t         len;
  AODataKey      *key,*oldkey;
  MPI_Comm       comm = aodata->comm;
  PetscTruth     flag;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(aodata,AODATA_COOKIE,1);

  ierr = AODataKeyFind_Private(aodata,name,&flag,&oldkey);CHKERRQ(ierr);
  if (flag)  SETERRQ1(PETSC_ERR_ARG_OUTOFRANGE,"Key already exists with given name: %s",name);

  ierr = PetscNew(AODataKey,&key);CHKERRQ(ierr);
  if (oldkey) { oldkey->next = key;} 
  else        { aodata->keys = key;} 
  ierr           = PetscStrlen(name,&len);CHKERRQ(ierr);
  ierr           = PetscMalloc((len+1)*sizeof(char),&key->name);CHKERRQ(ierr);
  ierr           = PetscStrcpy(key->name,name);CHKERRQ(ierr);
  key->N         = N;
  key->nsegments = 0;
  key->segments  = 0;
  key->ltog      = 0;
  key->next      = 0;

  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);

  /*  Set nlocal and ownership ranges */
  ierr            = PetscSplitOwnership(comm,&nlocal,&N);CHKERRQ(ierr);
  ierr            = PetscMalloc((size+1)*sizeof(PetscInt),&key->rowners);CHKERRQ(ierr);
  ierr            = MPI_Allgather(&nlocal,1,MPIU_INT,key->rowners+1,1,MPIU_INT,comm);CHKERRQ(ierr);
  key->rowners[0] = 0;
  for (i=2; i<=size; i++) {
    key->rowners[i] += key->rowners[i-1];
  }
  key->rstart        = key->rowners[rank];
  key->rend          = key->rowners[rank+1];

  key->nlocal        = nlocal;

  aodata->nkeys++;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "AODataSegmentAdd" 
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
PetscErrorCode PETSCDM_DLLEXPORT AODataSegmentAdd(AOData aodata,const char name[],const char segment[],PetscInt bs,PetscInt n,PetscInt *keys,void *data,PetscDataType dtype)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(aodata,AODATA_COOKIE,1);

  ierr = (*aodata->ops->segmentadd)(aodata,name,segment,bs,n,keys,data,dtype);CHKERRQ(ierr);

  /*
  ierr = PetscOptionsHasName(PETSC_NULL,"-ao_data_view",&flg1);CHKERRQ(ierr);
  if (flg1) {
    ierr = AODataView(aodata,PETSC_VIEWER_STDOUT_(comm));CHKERRQ(ierr);
  }
  ierr = PetscOptionsHasName(PETSC_NULL,"-ao_data_view_info",&flg1);CHKERRQ(ierr);
  if (flg1) {
    ierr = PetscViewerPushFormat(PETSC_VIEWER_STDOUT_(comm),PETSC_VIEWER_ASCII_INFO);CHKERRQ(ierr);
    ierr = AODataView(aodata,PETSC_VIEWER_STDOUT_(comm));CHKERRQ(ierr);
    ierr = PetscViewerPopFormat(PETSC_VIEWER_STDOUT_(comm));CHKERRQ(ierr);
  }
  */
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "AODataSegmentAddIS" 
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
PetscErrorCode PETSCDM_DLLEXPORT AODataSegmentAddIS(AOData aodata,const char name[],const char segment[],PetscInt bs,IS is,void *data,PetscDataType dtype)
{
  PetscErrorCode ierr;
  PetscInt       n,*keys;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(aodata,AODATA_COOKIE,1);
  PetscValidHeaderSpecific(is,IS_COOKIE,5);

  ierr = ISGetLocalSize(is,&n);CHKERRQ(ierr);
  ierr = ISGetIndices(is,&keys);CHKERRQ(ierr);
  ierr = (*aodata->ops->segmentadd)(aodata,name,segment,bs,n,keys,data,dtype);CHKERRQ(ierr);
  ierr = ISRestoreIndices(is,&keys);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}






