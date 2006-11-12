#define PETSCVEC_DLL
/*
   This file contains routines for basic map object implementation.
*/

#include "private/vecimpl.h"   /*I  "petscvec.h"   I*/
/*@C
     PetscMapInitialize - given a map where you have set either the global or local
           size sets up the map so that it may be used.

    Collective on MPI_Comm

   Input Parameters:
+    comm - the MPI communicator
-    map - pointer to the map

   Level: intermediate

    Notes:
       You must call PetscMapSetBlockSize() and either PetscMapSetSize() or PetscMapSetLocalSize()
       before calling this routine.

       Unlike regular PETSc objects you work with a pointer to the object instead of 
     the object directly.

    Fortran Notes: 
      Not available from Fortran

.seealso: PetscMapSetLocalSize(), PetscMapSetSize(), PetscMapGetSize(), PetscMapGetLocalSize(), PetscMap,
          PetscMapGetLocalRange(), PetscMapGetGlobalRange(), PetscMapSetBlockSize(), PetscMapGetBlockSize()

@*/
#undef __FUNCT__  
#define __FUNCT__ "PetscMapInitialize"
PetscErrorCode PETSCVEC_DLLEXPORT PetscMapInitialize(MPI_Comm comm,PetscMap *map)
{
  PetscMPIInt    rank,size;
  PetscInt       p;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(comm, &size);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr); 
  if (map->bs <=0) {SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"BlockSize not yet set");}
  if (map->n > 0) map->n = map->n/map->bs;
  if (map->N > 0) map->N = map->N/map->bs;
  ierr = PetscSplitOwnership(comm,&map->n,&map->N);CHKERRQ(ierr);
  map->n = map->n*map->bs;
  map->N = map->N*map->bs;
  if (!map->range) {
    ierr = PetscMalloc((size+1)*sizeof(PetscInt), &map->range);CHKERRQ(ierr);
  }
  ierr = MPI_Allgather(&map->n, 1, MPIU_INT, map->range+1, 1, MPIU_INT, comm);CHKERRQ(ierr);

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
  PetscInt       *range = out->range;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  ierr = PetscMemcpy(out,in,sizeof(PetscMap));CHKERRQ(ierr);
  if (!range) {
    ierr = PetscMalloc((size+1)*sizeof(PetscInt),&out->range);CHKERRQ(ierr);
  } else {
    out->range = range;
  }
  ierr = PetscMemcpy(out->range,in->range,(size+1)*sizeof(PetscInt));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
     PetscMapSetLocalSize - Sets the local size for a PetscMap object.

    Collective on PetscMap

   Input Parameters:
+    map - pointer to the map
-    n - the local size

   Level: intermediate

    Notes:
       Call this after the call to PetscMapInitialize()

       Unlike regular PETSc objects you work with a pointer to the object instead of 
     the object directly.

    Fortran Notes: 
      Not available from Fortran

.seealso: PetscMapInitialize(), PetscMapSetSize(), PetscMapGetSize(), PetscMapGetLocalSize(),
          PetscMapGetLocalRange(), PetscMapGetGlobalRange(), PetscMapSetBlockSize(), PetscMapGetBlockSize()

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

   Level: intermediate

    Notes:
       Call this after the call to PetscMapInitialize()

       Unlike regular PETSc objects you work with a pointer to the object instead of 
     the object directly.

    Fortran Notes: 
      Not available from Fortran

.seealso: PetscMapInitialize(), PetscMapSetSize(), PetscMapGetSize(), PetscMapGetLocalSize(),
          PetscMapGetLocalRange(), PetscMapGetGlobalRange(), PetscMapSetBlockSize(), PetscMapGetBlockSize()

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

   Level: intermediate

    Notes:
       Call this after the call to PetscMapInitialize()

       Unlike regular PETSc objects you work with a pointer to the object instead of 
     the object directly.

    Fortran Notes: 
      Not available from Fortran

.seealso: PetscMapInitialize(), PetscMapSetLocalSize(), PetscMapGetLocalSize(), PetscMapGetSize(),
          PetscMapGetLocalRange(), PetscMapGetGlobalRange(), PetscMapSetBlockSize(), PetscMapGetBlockSize()

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

   Level: intermediate

    Notes:
       Call this after the call to PetscMapInitialize()

       Unlike regular PETSc objects you work with a pointer to the object instead of 
     the object directly.

    Fortran Notes: 
      Not available from Fortran

.seealso: PetscMapInitialize(), PetscMapSetLocalSize(), PetscMapGetLocalSize(), PetscMapSetSize(),
          PetscMapGetLocalRange(), PetscMapGetGlobalRange(), PetscMapSetBlockSize(), PetscMapGetBlockSize()

@*/
#undef __FUNCT__  
#define __FUNCT__ "PetscMapGetSize"
PetscErrorCode PETSCVEC_DLLEXPORT PetscMapGetSize(PetscMap *map,PetscInt *n)
{
  PetscFunctionBegin;
  *n = map->n;
  PetscFunctionReturn(0);
}

/*@C
     PetscMapSetBlockSize - Sets the block size for a PetscMap object.

    Collective on PetscMap

   Input Parameters:
+    map - pointer to the map
-    bs - the size

   Level: intermediate

    Notes:
       Call this after the call to PetscMapInitialize()

       Unlike regular PETSc objects you work with a pointer to the object instead of 
     the object directly.

    Fortran Notes: 
      Not available from Fortran

.seealso: PetscMapInitialize(), PetscMapSetLocalSize(), PetscMapGetLocalSize(), PetscMapGetBlockSize(),
          PetscMapGetLocalRange(), PetscMapGetGlobalRange(), PetscMapSetSize(), PetscMapGetSize()

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

   Level: intermediate

    Notes:
       Call this after the call to PetscMapInitialize()

       Unlike regular PETSc objects you work with a pointer to the object instead of 
     the object directly.

    Fortran Notes: 
      Not available from Fortran

.seealso: PetscMapInitialize(), PetscMapSetLocalSize(), PetscMapGetLocalSize(), PetscMapSetSize(),
          PetscMapGetLocalRange(), PetscMapGetGlobalRange(), PetscMapSetBlockSize(), PetscMapGetSize()

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
     PetscMapGetLocalRange - gets the range of values owned by this process

    Not Collective

   Input Parameters:
.    map - pointer to the map

   Output Parameters:
+    rstart - first index owned by this process
-    rend - one more than the last index owned by this process

   Level: intermediate

    Notes:
       Call this after the call to PetscMapInitialize()

       Unlike regular PETSc objects you work with a pointer to the object instead of 
     the object directly.

    Fortran Notes: 
      Not available from Fortran

.seealso: PetscMapInitialize(), PetscMapSetLocalSize(), PetscMapGetLocalSize(), PetscMapSetSize(),
          PetscMapGetSize(), PetscMapGetGlobalRange(), PetscMapSetBlockSize(), PetscMapGetSize()

@*/
#undef __FUNCT__  
#define __FUNCT__ "PetscMapGetLocalRange"
PetscErrorCode PETSCVEC_DLLEXPORT PetscMapGetLocalRange(PetscMap *map,PetscInt *rstart,PetscInt *rend)
{
  PetscFunctionBegin;
  if (rstart) *rstart = map->rstart;
  if (rend)   *rend   = map->rend;
  PetscFunctionReturn(0);
}

/*@C
     PetscMapGetGlobalRange - gets the range of values owned by all processes

    Not Collective

   Input Parameters:
.    map - pointer to the map

   Output Parameters:
.    range - start of each processors range of indices (the final entry is one more then the
             last index on the last process)

   Level: intermediate

    Notes:
       Call this after the call to PetscMapInitialize()

       Unlike regular PETSc objects you work with a pointer to the object instead of 
     the object directly.

    Fortran Notes: 
      Not available from Fortran

.seealso: PetscMapInitialize(), PetscMapSetLocalSize(), PetscMapGetLocalSize(), PetscMapSetSize(),
          PetscMapGetSize(), PetscMapGetLocalRange(), PetscMapSetBlockSize(), PetscMapGetSize()

@*/
#undef __FUNCT__  
#define __FUNCT__ "PetscMapGetGlobalRange"
PetscErrorCode PETSCVEC_DLLEXPORT PetscMapGetGlobalRange(PetscMap *map,const PetscInt *range[])
{
  PetscFunctionBegin;
  *range = map->range;
  PetscFunctionReturn(0);
}
