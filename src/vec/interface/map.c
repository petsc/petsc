/*$Id: map.c,v 1.12 2000/10/24 20:25:11 bsmith Exp bsmith $*/
/*
     Provides the interface functions for all map operations.
   These are the map functions the user calls.
*/
#include "src/vec/vecimpl.h"    /*I "petscvec.h" I*/

#undef __FUNC__  
#define __FUNC__ "MapGetLocalSize"
/*@C
   MapGetLocalSize - Gets the number of elements associated with this processor.

   Not Collective

   Input Parameter:
.  m - the map object

   Output Parameter:
.  n - the local size

   Level: developer

.seealso: MapGetSize(), MapGetLocalRange(), MapGetGlobalRange()

   Concepts: Map^local size

@*/
int MapGetLocalSize(Map m,int *n)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(m,MAP_COOKIE); 
  ierr = (*m->ops->getlocalsize)(m,n);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MapGetSize"
/*@C
   MapGetSize - Gets the total number of elements associated with this map.

   Not Collective

   Input Parameter:
.  m - the map object

   Output Parameter:
.  N - the global size

   Level: developer

.seealso: MapGetLocalSize(), MapGetLocalRange(), MapGetGlobalRange()

   Concepts: Map^size
@*/
int MapGetSize(Map m,int *N)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(m,MAP_COOKIE); 
  ierr = (*m->ops->getglobalsize)(m,N);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MapGetLocalRange"
/*@C
   MapGetLocalRange - Gets the local ownership range for this procesor.

   Not Collective

   Input Parameter:
.  m - the map object

   Output Parameter:
+  rstart - the first local index
-  rend   - the last local index + 1

   Level: developer

.seealso: MapGetLocalSize(), MapGetGlobalRange()

@*/
int MapGetLocalRange(Map m,int *rstart,int *rend)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(m,MAP_COOKIE); 
  ierr = (*m->ops->getlocalrange)(m,rstart,rend);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MapGetGlobalRange"
/*@C
   MapGetGlobalRange - Gets the ownership ranges for all processors.

   Not Collective

   Input Parameter:
.  m - the map object

   Output Parameter:
.  range - array of size + 1 where size is the size of the communicator 
           associated with the map. range[rank], range[rank+1] is the 
           range for processor 

   Level: developer

.seealso: MapGetSize(), MapGetLocalRange()

@*/
int MapGetGlobalRange(Map m,int *range[])
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(m,MAP_COOKIE); 
  ierr = (*m->ops->getglobalrange)(m,range);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MapDestroy"
/*@C
   MapDestroy - Destroys a map object.

   Not Collective

   Input Parameter:
.  m - the map object

   Level: developer

.seealso: MapCreateMPI()

@*/
int MapDestroy(Map m)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(m,MAP_COOKIE); 
  ierr = (*m->ops->destroy)(m);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

