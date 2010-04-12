#define PETSCVEC_DLL
/*
   This file contains routines for basic map object implementation.
*/

#include "private/vecimpl.h"   /*I  "petscvec.h"   I*/
#undef __FUNCT__  
#define __FUNCT__ "PetscLayoutCreate"
/*@C
     PetscLayoutCreate - Allocates PetscLayout space and sets the map contents to the default.

    Collective on MPI_Comm

   Input Parameters:
+    comm - the MPI communicator
-    map - pointer to the map

   Level: developer

    Notes: Typical calling sequence
       PetscLayoutCreate(MPI_Comm,PetscLayout *);
       PetscLayoutSetBlockSize(PetscLayout,1);
       PetscLayoutSetSize(PetscLayout,n) or PetscLayoutSetLocalSize(PetscLayout,N);
       PetscLayoutSetUp(PetscLayout);
       PetscLayoutGetSize(PetscLayout,PetscInt *);
       PetscLayoutDestroy(PetscLayout);

      The PetscLayout object and methods are intended to be used in the PETSc Vec and Mat implementions; it is 
      recommended they not be used in user codes unless you really gain something in their use.

    Fortran Notes: 
      Not available from Fortran

.seealso: PetscLayoutSetLocalSize(), PetscLayoutSetSize(), PetscLayoutGetSize(), PetscLayoutGetLocalSize(), PetscLayout, PetscLayoutDestroy(),
          PetscLayoutGetRange(), PetscLayoutGetRanges(), PetscLayoutSetBlockSize(), PetscLayoutGetBlockSize(), PetscLayoutSetUp()

@*/
PetscErrorCode PETSCVEC_DLLEXPORT PetscLayoutCreate(MPI_Comm comm,PetscLayout *map)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscNew(struct _p_PetscLayout,map);CHKERRQ(ierr);
  (*map)->comm   = comm;
  (*map)->bs     = -1;
  (*map)->n      = -1;
  (*map)->N      = -1;
  (*map)->range  = 0;
  (*map)->rstart = 0;
  (*map)->rend   = 0;
  PetscFunctionReturn(0);
}

/*@C
     PetscLayoutDestroy - Frees a map object and frees its range if that exists. 

    Collective on MPI_Comm

   Input Parameters:
.    map - the PetscLayout

   Level: developer

      The PetscLayout object and methods are intended to be used in the PETSc Vec and Mat implementions; it is 
      recommended they not be used in user codes unless you really gain something in their use.

    Fortran Notes: 
      Not available from Fortran

.seealso: PetscLayoutSetLocalSize(), PetscLayoutSetSize(), PetscLayoutGetSize(), PetscLayoutGetLocalSize(), PetscLayout, PetscLayoutCreate(),
          PetscLayoutGetRange(), PetscLayoutGetRanges(), PetscLayoutSetBlockSize(), PetscLayoutGetBlockSize(), PetscLayoutSetUp()

@*/
#undef __FUNCT__  
#define __FUNCT__ "PetscLayoutDestroy"
PetscErrorCode PETSCVEC_DLLEXPORT PetscLayoutDestroy(PetscLayout map)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!map->refcnt--) {
    if (map->range) {ierr = PetscFree(map->range);CHKERRQ(ierr);}
    ierr = PetscFree(map);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*@C
     PetscLayoutSetUp - given a map where you have set either the global or local
           size sets up the map so that it may be used.

    Collective on MPI_Comm

   Input Parameters:
.    map - pointer to the map

   Level: developer

    Notes: Typical calling sequence
       PetscLayoutCreate(MPI_Comm,PetscLayout *);
       PetscLayoutSetBlockSize(PetscLayout,1);
       PetscLayoutSetSize(PetscLayout,n) or PetscLayoutSetLocalSize(PetscLayout,N); or both
       PetscLayoutSetUp(PetscLayout);
       PetscLayoutGetSize(PetscLayout,PetscInt *);


       If the local size, global size are already set and range exists then this does nothing.

    Fortran Notes: 
      Not available from Fortran

.seealso: PetscLayoutSetLocalSize(), PetscLayoutSetSize(), PetscLayoutGetSize(), PetscLayoutGetLocalSize(), PetscLayout, PetscLayoutDestroy(),
          PetscLayoutGetRange(), PetscLayoutGetRanges(), PetscLayoutSetBlockSize(), PetscLayoutGetBlockSize(), PetscLayoutCreate()

@*/
#undef __FUNCT__  
#define __FUNCT__ "PetscLayoutSetUp"
PetscErrorCode PETSCVEC_DLLEXPORT PetscLayoutSetUp(PetscLayout map)
{
  PetscMPIInt    rank,size;
  PetscInt       p;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (map->bs <=0) {SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"BlockSize not yet set");}
  if ((map->n >= 0) && (map->N >= 0) && (map->range)) PetscFunctionReturn(0);

  ierr = MPI_Comm_size(map->comm, &size);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(map->comm, &rank);CHKERRQ(ierr); 
  if (map->n > 0) map->n = map->n/map->bs;
  if (map->N > 0) map->N = map->N/map->bs;
  ierr = PetscSplitOwnership(map->comm,&map->n,&map->N);CHKERRQ(ierr);
  map->n = map->n*map->bs;
  map->N = map->N*map->bs;
  if (!map->range) {
    ierr = PetscMalloc((size+1)*sizeof(PetscInt), &map->range);CHKERRQ(ierr);
  }
  ierr = MPI_Allgather(&map->n, 1, MPIU_INT, map->range+1, 1, MPIU_INT, map->comm);CHKERRQ(ierr);

  map->range[0] = 0;
  for(p = 2; p <= size; p++) {
    map->range[p] += map->range[p-1];
  }

  map->rstart = map->range[rank];
  map->rend   = map->range[rank+1];
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscLayoutCopy"
/*@C

    PetscLayoutCopy - creates a new PetscLayout with the same information as a given one. If the PetscLayout already exists it is destroyed first.

     Collective on PetscLayout

    Input Parameter:
.     in - input PetscLayout to be copied

    Output Parameter:
.     out - the copy

   Level: developer

    Notes: PetscLayoutSetUp() does not need to be called on the resulting PetscLayout

    Developer Note: Unlike all other copy routines this destroys any input object and makes a new one. This routine should be fixed to have a PetscLayoutDuplicate() 
      that ONLY creates a new one and a PetscLayoutCopy() that truely copies the data and does not delete the old object.

.seealso: PetscLayoutCreate(), PetscLayoutDestroy(), PetscLayoutSetUp()

@*/
PetscErrorCode PETSCVEC_DLLEXPORT PetscLayoutCopy(PetscLayout in,PetscLayout *out)
{
  PetscMPIInt    size;
  PetscErrorCode ierr;
  MPI_Comm       comm = in->comm;

  PetscFunctionBegin;
  if (*out) {ierr = PetscLayoutDestroy(*out);CHKERRQ(ierr);}
  ierr = PetscLayoutCreate(comm,out);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  ierr = PetscMemcpy(*out,in,sizeof(struct _p_PetscLayout));CHKERRQ(ierr);
  ierr = PetscMalloc((size+1)*sizeof(PetscInt),&(*out)->range);CHKERRQ(ierr);
  ierr = PetscMemcpy((*out)->range,in->range,(size+1)*sizeof(PetscInt));CHKERRQ(ierr);
  (*out)->refcnt = 0;
  PetscFunctionReturn(0);
}

/*@C
     PetscLayoutSetLocalSize - Sets the local size for a PetscLayout object.

    Collective on PetscLayout

   Input Parameters:
+    map - pointer to the map
-    n - the local size

   Level: developer

    Notes:
       Call this after the call to PetscLayoutCreate()

    Fortran Notes: 
      Not available from Fortran

.seealso: PetscLayoutCreate(), PetscLayoutSetSize(), PetscLayoutGetSize(), PetscLayoutGetLocalSize(), PetscLayoutSetUp()
          PetscLayoutGetRange(), PetscLayoutGetRanges(), PetscLayoutSetBlockSize(), PetscLayoutGetBlockSize()

@*/
#undef __FUNCT__  
#define __FUNCT__ "PetscLayoutSetLocalSize"
PetscErrorCode PETSCVEC_DLLEXPORT PetscLayoutSetLocalSize(PetscLayout map,PetscInt n)
{
  PetscFunctionBegin;
  map->n = n;
  PetscFunctionReturn(0);
}

/*@C
     PetscLayoutGetLocalSize - Gets the local size for a PetscLayout object.

    Not Collective

   Input Parameters:
.    map - pointer to the map

   Output Parameters:
.    n - the local size

   Level: developer

    Notes:
       Call this after the call to PetscLayoutSetUp()

    Fortran Notes: 
      Not available from Fortran

.seealso: PetscLayoutCreate(), PetscLayoutSetSize(), PetscLayoutGetSize(), PetscLayoutGetLocalSize(), PetscLayoutSetUp()
          PetscLayoutGetRange(), PetscLayoutGetRanges(), PetscLayoutSetBlockSize(), PetscLayoutGetBlockSize()

@*/
#undef __FUNCT__  
#define __FUNCT__ "PetscLayoutGetLocalSize"
PetscErrorCode PETSCVEC_DLLEXPORT PetscLayoutGetLocalSize(PetscLayout map,PetscInt *n)
{
  PetscFunctionBegin;
  *n = map->n;
  PetscFunctionReturn(0);
}

/*@C
     PetscLayoutSetSize - Sets the global size for a PetscLayout object.

    Collective on PetscLayout

   Input Parameters:
+    map - pointer to the map
-    n - the global size

   Level: developer

    Notes:
       Call this after the call to PetscLayoutCreate()

    Fortran Notes: 
      Not available from Fortran

.seealso: PetscLayoutCreate(), PetscLayoutSetLocalSize(), PetscLayoutGetLocalSize(), PetscLayoutGetSize(), PetscLayoutSetUp()
          PetscLayoutGetRange(), PetscLayoutGetRanges(), PetscLayoutSetBlockSize(), PetscLayoutGetBlockSize()

@*/
#undef __FUNCT__  
#define __FUNCT__ "PetscLayoutSetSize"
PetscErrorCode PETSCVEC_DLLEXPORT PetscLayoutSetSize(PetscLayout map,PetscInt n)
{
  PetscFunctionBegin;
  map->N = n;
  PetscFunctionReturn(0);
}

/*@C
     PetscLayoutGetSize - Gets the global size for a PetscLayout object.

    Not Collective

   Input Parameters:
.    map - pointer to the map

   Output Parameters:
.    n - the global size

   Level: developer

    Notes:
       Call this after the call to PetscLayoutSetUp()

    Fortran Notes: 
      Not available from Fortran

.seealso: PetscLayoutCreate(), PetscLayoutSetLocalSize(), PetscLayoutGetLocalSize(), PetscLayoutSetSize(), PetscLayoutSetUp()
          PetscLayoutGetRange(), PetscLayoutGetRanges(), PetscLayoutSetBlockSize(), PetscLayoutGetBlockSize()

@*/
#undef __FUNCT__  
#define __FUNCT__ "PetscLayoutGetSize"
PetscErrorCode PETSCVEC_DLLEXPORT PetscLayoutGetSize(PetscLayout map,PetscInt *n)
{
  PetscFunctionBegin;
  *n = map->N;
  PetscFunctionReturn(0);
}

/*@C
     PetscLayoutSetBlockSize - Sets the block size for a PetscLayout object.

    Collective on PetscLayout

   Input Parameters:
+    map - pointer to the map
-    bs - the size

   Level: developer

    Notes:
       Call this after the call to PetscLayoutCreate()

    Fortran Notes: 
      Not available from Fortran

.seealso: PetscLayoutCreate(), PetscLayoutSetLocalSize(), PetscLayoutGetLocalSize(), PetscLayoutGetBlockSize(),
          PetscLayoutGetRange(), PetscLayoutGetRanges(), PetscLayoutSetSize(), PetscLayoutGetSize(), PetscLayoutSetUp()

@*/
#undef __FUNCT__  
#define __FUNCT__ "PetscLayoutSetBlockSize"
PetscErrorCode PETSCVEC_DLLEXPORT PetscLayoutSetBlockSize(PetscLayout map,PetscInt bs)
{
  PetscFunctionBegin;
  map->bs = bs;
  PetscFunctionReturn(0);
}

/*@C
     PetscLayoutGetBlockSize - Gets the block size for a PetscLayout object.

    Not Collective

   Input Parameters:
.    map - pointer to the map

   Output Parameters:
.    bs - the size

   Level: developer

    Notes:
       Call this after the call to PetscLayoutSetUp()

    Fortran Notes: 
      Not available from Fortran

.seealso: PetscLayoutCreate(), PetscLayoutSetLocalSize(), PetscLayoutGetLocalSize(), PetscLayoutSetSize(), PetscLayoutSetUp()
          PetscLayoutGetRange(), PetscLayoutGetRanges(), PetscLayoutSetBlockSize(), PetscLayoutGetSize()

@*/
#undef __FUNCT__  
#define __FUNCT__ "PetscLayoutGetBlockSize"
PetscErrorCode PETSCVEC_DLLEXPORT PetscLayoutGetBlockSize(PetscLayout map,PetscInt *bs)
{
  PetscFunctionBegin;
  *bs = map->bs;
  PetscFunctionReturn(0);
}


/*@C
     PetscLayoutGetRange - gets the range of values owned by this process

    Not Collective

   Input Parameters:
.    map - pointer to the map

   Output Parameters:
+    rstart - first index owned by this process
-    rend - one more than the last index owned by this process

   Level: developer

    Notes:
       Call this after the call to PetscLayoutSetUp()

    Fortran Notes: 
      Not available from Fortran

.seealso: PetscLayoutCreate(), PetscLayoutSetLocalSize(), PetscLayoutGetLocalSize(), PetscLayoutSetSize(),
          PetscLayoutGetSize(), PetscLayoutGetRanges(), PetscLayoutSetBlockSize(), PetscLayoutGetSize(), PetscLayoutSetUp()

@*/
#undef __FUNCT__  
#define __FUNCT__ "PetscLayoutGetRange"
PetscErrorCode PETSCVEC_DLLEXPORT PetscLayoutGetRange(PetscLayout map,PetscInt *rstart,PetscInt *rend)
{
  PetscFunctionBegin;
  if (rstart) *rstart = map->rstart;
  if (rend)   *rend   = map->rend;
  PetscFunctionReturn(0);
}

/*@C
     PetscLayoutGetRanges - gets the range of values owned by all processes

    Not Collective

   Input Parameters:
.    map - pointer to the map

   Output Parameters:
.    range - start of each processors range of indices (the final entry is one more then the
             last index on the last process)

   Level: developer

    Notes:
       Call this after the call to PetscLayoutSetUp()

    Fortran Notes: 
      Not available from Fortran

.seealso: PetscLayoutCreate(), PetscLayoutSetLocalSize(), PetscLayoutGetLocalSize(), PetscLayoutSetSize(),
          PetscLayoutGetSize(), PetscLayoutGetRange(), PetscLayoutSetBlockSize(), PetscLayoutGetSize(), PetscLayoutSetUp()

@*/
#undef __FUNCT__  
#define __FUNCT__ "PetscLayoutGetRanges"
PetscErrorCode PETSCVEC_DLLEXPORT PetscLayoutGetRanges(PetscLayout map,const PetscInt *range[])
{
  PetscFunctionBegin;
  *range = map->range;
  PetscFunctionReturn(0);
}
