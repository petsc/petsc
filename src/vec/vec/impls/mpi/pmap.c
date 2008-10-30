#define PETSCVEC_DLL
/*
   This file contains routines for basic map object implementation.
*/

#include "private/vecimpl.h"   /*I  "petscvec.h"   I*/
/*@C
     PetscMapInitialize - Sets the map contents to the default.

    Collective on MPI_Comm

   Input Parameters:
+    comm - the MPI communicator
-    map - pointer to the map

   Level: developer

    Notes: Typical calling sequence
       PetscMapInitialize(MPI_Comm,PetscMap *);
       PetscMapSetBlockSize(PetscMap*,1);
       PetscMapSetSize(PetscMap*,n) or PetscMapSetLocalSize(PetscMap*,N);
       PetscMapSetUp(PetscMap*);
       PetscMapGetSize(PetscMap*,PetscInt *);

       Unlike regular PETSc objects you work with a pointer to the object instead of 
     the object directly.

      The PetscMap object and methods are intended to be used in the PETSc Vec and Mat implementions; it is 
      recommended they not be used in user codes unless you really gain something in their use.

    Fortran Notes: 
      Not available from Fortran

.seealso: PetscMapSetLocalSize(), PetscMapSetSize(), PetscMapGetSize(), PetscMapGetLocalSize(), PetscMap,
          PetscMapGetRange(), PetscMapGetRanges(), PetscMapSetBlockSize(), PetscMapGetBlockSize(), PetscMapSetUp()

@*/
#undef __FUNCT__  
#define __FUNCT__ "PetscMapInitialize"
PetscErrorCode PETSCVEC_DLLEXPORT PetscMapInitialize(MPI_Comm comm,PetscMap *map)
{
  PetscFunctionBegin;
  map->comm   = comm;
  map->bs     = -1;
  map->n      = -1;
  map->N      = -1;
  map->range  = 0;
  map->rstart = 0;
  map->rend   = 0;
  PetscFunctionReturn(0);
}

/*@C
     PetscMapDestroy - Frees a map object and frees its range if that exists. 

    Collective on MPI_Comm

   Input Parameters:
.    map - the PetscMap

   Level: developer

       Unlike regular PETSc objects you work with a pointer to the object instead of 
     the object directly.

      The PetscMap object and methods are intended to be used in the PETSc Vec and Mat implementions; it is 
      recommended they not be used in user codes unless you really gain something in their use.

    Fortran Notes: 
      Not available from Fortran

.seealso: PetscMapSetLocalSize(), PetscMapSetSize(), PetscMapGetSize(), PetscMapGetLocalSize(), PetscMap, PetscMapInitialize(),
          PetscMapGetRange(), PetscMapGetRanges(), PetscMapSetBlockSize(), PetscMapGetBlockSize(), PetscMapSetUp()

@*/
#undef __FUNCT__  
#define __FUNCT__ "PetscMapDestroy"
PetscErrorCode PETSCVEC_DLLEXPORT PetscMapDestroy(PetscMap *map)
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
     PetscMapSetUp - given a map where you have set either the global or local
           size sets up the map so that it may be used.

    Collective on MPI_Comm

   Input Parameters:
.    map - pointer to the map

   Level: developer

    Notes: Typical calling sequence
       PetscMapInitialize(MPI_Comm,PetscMap *);
       PetscMapSetBlockSize(PetscMap*,1);
       PetscMapSetSize(PetscMap*,n) or PetscMapSetLocalSize(PetscMap*,N); or both
       PetscMapSetUp(PetscMap*);
       PetscMapGetSize(PetscMap*,PetscInt *);

       Unlike regular PETSc objects you work with a pointer to the object instead of 
     the object directly.

       If the local size, global size are already set and range exists then this does nothing.

    Fortran Notes: 
      Not available from Fortran

.seealso: PetscMapSetLocalSize(), PetscMapSetSize(), PetscMapGetSize(), PetscMapGetLocalSize(), PetscMap,
          PetscMapGetRange(), PetscMapGetRanges(), PetscMapSetBlockSize(), PetscMapGetBlockSize(), PetscMapInitialize()

@*/
#undef __FUNCT__  
#define __FUNCT__ "PetscMapSetUp"
PetscErrorCode PETSCVEC_DLLEXPORT PetscMapSetUp(PetscMap *map)
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
#define __FUNCT__ "PetscMapCopy"
PetscErrorCode PETSCVEC_DLLEXPORT PetscMapCopy(MPI_Comm comm,PetscMap *in,PetscMap *out)
{
  PetscMPIInt    size;
  PetscErrorCode ierr;
  PetscInt       *range = out->range; /* keep copy of this since PetscMemcpy() below will cover it */

  PetscFunctionBegin;
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  ierr = PetscMemcpy(out,in,sizeof(PetscMap));CHKERRQ(ierr);
  if (!range) {
    ierr = PetscMalloc((size+1)*sizeof(PetscInt),&out->range);CHKERRQ(ierr);
  } else {
    out->range = range;
  }
  out->refcnt = 0;
  ierr = PetscMemcpy(out->range,in->range,(size+1)*sizeof(PetscInt));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
     PetscMapSetLocalSize - Sets the local size for a PetscMap object.

    Collective on PetscMap

   Input Parameters:
+    map - pointer to the map
-    n - the local size

   Level: developer

    Notes:
       Call this after the call to PetscMapInitialize()

       Unlike regular PETSc objects you work with a pointer to the object instead of 
     the object directly.

    Fortran Notes: 
      Not available from Fortran

.seealso: PetscMapInitialize(), PetscMapSetSize(), PetscMapGetSize(), PetscMapGetLocalSize(), PetscMapSetUp()
          PetscMapGetRange(), PetscMapGetRanges(), PetscMapSetBlockSize(), PetscMapGetBlockSize()

@*/
#undef __FUNCT__  
#define __FUNCT__ "PetscMapSetLocalSize"
PetscErrorCode PETSCVEC_DLLEXPORT PetscMapSetLocalSize(PetscMap *map,PetscInt n)
{
  PetscFunctionBegin;
  map->n = n;
  PetscFunctionReturn(0);
}

/*@C
     PetscMapGetLocalSize - Gets the local size for a PetscMap object.

    Not Collective

   Input Parameters:
.    map - pointer to the map

   Output Parameters:
.    n - the local size

   Level: developer

    Notes:
       Call this after the call to PetscMapSetUp()

       Unlike regular PETSc objects you work with a pointer to the object instead of 
     the object directly.

    Fortran Notes: 
      Not available from Fortran

.seealso: PetscMapInitialize(), PetscMapSetSize(), PetscMapGetSize(), PetscMapGetLocalSize(), PetscMapSetUp()
          PetscMapGetRange(), PetscMapGetRanges(), PetscMapSetBlockSize(), PetscMapGetBlockSize()

@*/
#undef __FUNCT__  
#define __FUNCT__ "PetscMapGetLocalSize"
PetscErrorCode PETSCVEC_DLLEXPORT PetscMapGetLocalSize(PetscMap *map,PetscInt *n)
{
  PetscFunctionBegin;
  *n = map->n;
  PetscFunctionReturn(0);
}

/*@C
     PetscMapSetSize - Sets the global size for a PetscMap object.

    Collective on PetscMap

   Input Parameters:
+    map - pointer to the map
-    n - the global size

   Level: developer

    Notes:
       Call this after the call to PetscMapInitialize()

       Unlike regular PETSc objects you work with a pointer to the object instead of 
     the object directly.

    Fortran Notes: 
      Not available from Fortran

.seealso: PetscMapInitialize(), PetscMapSetLocalSize(), PetscMapGetLocalSize(), PetscMapGetSize(), PetscMapSetUp()
          PetscMapGetRange(), PetscMapGetRanges(), PetscMapSetBlockSize(), PetscMapGetBlockSize()

@*/
#undef __FUNCT__  
#define __FUNCT__ "PetscMapSetSize"
PetscErrorCode PETSCVEC_DLLEXPORT PetscMapSetSize(PetscMap *map,PetscInt n)
{
  PetscFunctionBegin;
  map->N = n;
  PetscFunctionReturn(0);
}

/*@C
     PetscMapGetSize - Gets the global size for a PetscMap object.

    Not Collective

   Input Parameters:
.    map - pointer to the map

   Output Parameters:
.    n - the global size

   Level: developer

    Notes:
       Call this after the call to PetscMapSetUp()

       Unlike regular PETSc objects you work with a pointer to the object instead of 
     the object directly.

    Fortran Notes: 
      Not available from Fortran

.seealso: PetscMapInitialize(), PetscMapSetLocalSize(), PetscMapGetLocalSize(), PetscMapSetSize(), PetscMapSetUp()
          PetscMapGetRange(), PetscMapGetRanges(), PetscMapSetBlockSize(), PetscMapGetBlockSize()

@*/
#undef __FUNCT__  
#define __FUNCT__ "PetscMapGetSize"
PetscErrorCode PETSCVEC_DLLEXPORT PetscMapGetSize(PetscMap *map,PetscInt *n)
{
  PetscFunctionBegin;
  *n = map->N;
  PetscFunctionReturn(0);
}

/*@C
     PetscMapSetBlockSize - Sets the block size for a PetscMap object.

    Collective on PetscMap

   Input Parameters:
+    map - pointer to the map
-    bs - the size

   Level: developer

    Notes:
       Call this after the call to PetscMapInitialize()

       Unlike regular PETSc objects you work with a pointer to the object instead of 
     the object directly.

    Fortran Notes: 
      Not available from Fortran

.seealso: PetscMapInitialize(), PetscMapSetLocalSize(), PetscMapGetLocalSize(), PetscMapGetBlockSize(),
          PetscMapGetRange(), PetscMapGetRanges(), PetscMapSetSize(), PetscMapGetSize(), PetscMapSetUp()

@*/
#undef __FUNCT__  
#define __FUNCT__ "PetscMapSetBlockSize"
PetscErrorCode PETSCVEC_DLLEXPORT PetscMapSetBlockSize(PetscMap *map,PetscInt bs)
{
  PetscFunctionBegin;
  map->bs = bs;
  PetscFunctionReturn(0);
}

/*@C
     PetscMapGetBlockSize - Gets the block size for a PetscMap object.

    Not Collective

   Input Parameters:
.    map - pointer to the map

   Output Parameters:
.    bs - the size

   Level: developer

    Notes:
       Call this after the call to PetscMapSetUp()

       Unlike regular PETSc objects you work with a pointer to the object instead of 
     the object directly.

    Fortran Notes: 
      Not available from Fortran

.seealso: PetscMapInitialize(), PetscMapSetLocalSize(), PetscMapGetLocalSize(), PetscMapSetSize(), PetscMapSetUp()
          PetscMapGetRange(), PetscMapGetRanges(), PetscMapSetBlockSize(), PetscMapGetSize()

@*/
#undef __FUNCT__  
#define __FUNCT__ "PetscMapGetBlockSize"
PetscErrorCode PETSCVEC_DLLEXPORT PetscMapGetBlockSize(PetscMap *map,PetscInt *bs)
{
  PetscFunctionBegin;
  *bs = map->bs;
  PetscFunctionReturn(0);
}


/*@C
     PetscMapGetRange - gets the range of values owned by this process

    Not Collective

   Input Parameters:
.    map - pointer to the map

   Output Parameters:
+    rstart - first index owned by this process
-    rend - one more than the last index owned by this process

   Level: developer

    Notes:
       Call this after the call to PetscMapSetUp()

       Unlike regular PETSc objects you work with a pointer to the object instead of 
     the object directly.

    Fortran Notes: 
      Not available from Fortran

.seealso: PetscMapInitialize(), PetscMapSetLocalSize(), PetscMapGetLocalSize(), PetscMapSetSize(),
          PetscMapGetSize(), PetscMapGetRanges(), PetscMapSetBlockSize(), PetscMapGetSize(), PetscMapSetUp()

@*/
#undef __FUNCT__  
#define __FUNCT__ "PetscMapGetRange"
PetscErrorCode PETSCVEC_DLLEXPORT PetscMapGetRange(PetscMap *map,PetscInt *rstart,PetscInt *rend)
{
  PetscFunctionBegin;
  if (rstart) *rstart = map->rstart;
  if (rend)   *rend   = map->rend;
  PetscFunctionReturn(0);
}

/*@C
     PetscMapGetRanges - gets the range of values owned by all processes

    Not Collective

   Input Parameters:
.    map - pointer to the map

   Output Parameters:
.    range - start of each processors range of indices (the final entry is one more then the
             last index on the last process)

   Level: developer

    Notes:
       Call this after the call to PetscMapSetUp()

       Unlike regular PETSc objects you work with a pointer to the object instead of 
     the object directly.

    Fortran Notes: 
      Not available from Fortran

.seealso: PetscMapInitialize(), PetscMapSetLocalSize(), PetscMapGetLocalSize(), PetscMapSetSize(),
          PetscMapGetSize(), PetscMapGetRange(), PetscMapSetBlockSize(), PetscMapGetSize(), PetscMapSetUp()

@*/
#undef __FUNCT__  
#define __FUNCT__ "PetscMapGetRanges"
PetscErrorCode PETSCVEC_DLLEXPORT PetscMapGetRanges(PetscMap *map,const PetscInt *range[])
{
  PetscFunctionBegin;
  *range = map->range;
  PetscFunctionReturn(0);
}
