/*$Id: map.c,v 1.15 2001/07/20 21:18:10 bsmith Exp $*/
/*
     Provides the interface functions for all map operations.
   These are the map functions the user calls.
*/
#include "src/vec/vecimpl.h"    /*I "petscvec.h" I*/

/* Logging support */
int MAP_COOKIE;

#undef __FUNCT__  
#define __FUNCT__ "PetscMapGetLocalSize"
/*@C
   PetscMapGetLocalSize - Gets the number of elements associated with this processor.

   Not Collective

   Input Parameter:
.  m - the map object

   Output Parameter:
.  n - the local size

   Level: developer

.seealso: PetscMapGetSize(), PetscMapGetLocalRange(), PetscMapGetGlobalRange()

   Concepts: PetscMap^local size

@*/
int PetscMapGetLocalSize(PetscMap m,int *n)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(m,MAP_COOKIE); 
  ierr = (*m->ops->getlocalsize)(m,n);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscMapGetSize"
/*@C
   PetscMapGetSize - Gets the total number of elements associated with this map.

   Not Collective

   Input Parameter:
.  m - the map object

   Output Parameter:
.  N - the global size

   Level: developer

.seealso: PetscMapGetLocalSize(), PetscMapGetLocalRange(), PetscMapGetGlobalRange()

   Concepts: PetscMap^size
@*/
int PetscMapGetSize(PetscMap m,int *N)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(m,MAP_COOKIE); 
  ierr = (*m->ops->getglobalsize)(m,N);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscMapGetLocalRange"
/*@C
   PetscMapGetLocalRange - Gets the local ownership range for this procesor.

   Not Collective

   Input Parameter:
.  m - the map object

   Output Parameter:
+  rstart - the first local index
-  rend   - the last local index + 1

   Level: developer

.seealso: PetscMapGetLocalSize(), PetscMapGetGlobalRange()

@*/
int PetscMapGetLocalRange(PetscMap m,int *rstart,int *rend)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(m,MAP_COOKIE); 
  ierr = (*m->ops->getlocalrange)(m,rstart,rend);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscMapGetGlobalRange"
/*@C
   PetscMapGetGlobalRange - Gets the ownership ranges for all processors.

   Not Collective

   Input Parameter:
.  m - the map object

   Output Parameter:
.  range - array of size + 1 where size is the size of the communicator 
           associated with the map. range[rank], range[rank+1] is the 
           range for processor 

   Level: developer

.seealso: PetscMapGetSize(), PetscMapGetLocalRange()

@*/
int PetscMapGetGlobalRange(PetscMap m,int *range[])
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(m,MAP_COOKIE); 
  ierr = (*m->ops->getglobalrange)(m,range);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscMapDestroy"
/*@C
   PetscMapDestroy - Destroys a map object.

   Not Collective

   Input Parameter:
.  m - the map object

   Level: developer

.seealso: PetscMapCreateMPI()

@*/
int PetscMapDestroy(PetscMap m)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(m,MAP_COOKIE); 
  ierr = (*m->ops->destroy)(m);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

