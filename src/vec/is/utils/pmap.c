
/*
   This file contains routines for basic map object implementation.
*/

#include <petsc/private/isimpl.h> /*I "petscis.h" I*/

/*@
  PetscLayoutCreate - Allocates PetscLayout space and sets the PetscLayout contents to the default.

  Collective

  Input Parameters:
. comm - the MPI communicator

  Output Parameters:
. map - the new PetscLayout

  Level: advanced

  Notes:
  Typical calling sequence
.vb
       PetscLayoutCreate(MPI_Comm,PetscLayout *);
       PetscLayoutSetBlockSize(PetscLayout,bs);
       PetscLayoutSetSize(PetscLayout,N); // or PetscLayoutSetLocalSize(PetscLayout,n);
       PetscLayoutSetUp(PetscLayout);
.ve
  Alternatively,
.vb
      PetscLayoutCreateFromSizes(comm,n,N,bs,&layout);
.ve

  Optionally use any of the following
.vb
  PetscLayoutGetSize(PetscLayout,PetscInt *);
  PetscLayoutGetLocalSize(PetscLayout,PetscInt *);
  PetscLayoutGetRange(PetscLayout,PetscInt *rstart,PetscInt *rend);
  PetscLayoutGetRanges(PetscLayout,const PetscInt *range[]);
  PetscLayoutDestroy(PetscLayout*);
.ve

  The PetscLayout object and methods are intended to be used in the PETSc Vec and Mat implementations; it is often not needed in
  user codes unless you really gain something in their use.

.seealso: PetscLayoutSetLocalSize(), PetscLayoutSetSize(), PetscLayoutGetSize(), PetscLayoutGetLocalSize(), PetscLayout, PetscLayoutDestroy(),
          PetscLayoutGetRange(), PetscLayoutGetRanges(), PetscLayoutSetBlockSize(), PetscLayoutGetBlockSize(), PetscLayoutSetUp(),
          PetscLayoutCreateFromSizes()

@*/
PetscErrorCode PetscLayoutCreate(MPI_Comm comm,PetscLayout *map)
{
  PetscFunctionBegin;
  PetscCall(PetscNew(map));
  PetscCallMPI(MPI_Comm_size(comm, &(*map)->size));
  (*map)->comm        = comm;
  (*map)->bs          = -1;
  (*map)->n           = -1;
  (*map)->N           = -1;
  (*map)->range       = NULL;
  (*map)->range_alloc = PETSC_TRUE;
  (*map)->rstart      = 0;
  (*map)->rend        = 0;
  (*map)->setupcalled = PETSC_FALSE;
  (*map)->oldn        = -1;
  (*map)->oldN        = -1;
  (*map)->oldbs       = -1;
  PetscFunctionReturn(0);
}

/*@
  PetscLayoutCreateFromSizes - Allocates PetscLayout space, sets the layout sizes, and sets the layout up.

  Collective

  Input Parameters:
+ comm  - the MPI communicator
. n     - the local size (or PETSC_DECIDE)
. N     - the global size (or PETSC_DECIDE)
- bs    - the block size (or PETSC_DECIDE)

  Output Parameters:
. map - the new PetscLayout

  Level: advanced

  Notes:
$ PetscLayoutCreateFromSizes(comm,n,N,bs,&layout);
  is a shorthand for
.vb
  PetscLayoutCreate(comm,&layout);
  PetscLayoutSetLocalSize(layout,n);
  PetscLayoutSetSize(layout,N);
  PetscLayoutSetBlockSize(layout,bs);
  PetscLayoutSetUp(layout);
.ve

.seealso: PetscLayoutCreate(), PetscLayoutSetLocalSize(), PetscLayoutSetSize(), PetscLayoutGetSize(), PetscLayoutGetLocalSize(), PetscLayout, PetscLayoutDestroy(),
          PetscLayoutGetRange(), PetscLayoutGetRanges(), PetscLayoutSetBlockSize(), PetscLayoutGetBlockSize(), PetscLayoutSetUp(), PetscLayoutCreateFromRanges()

@*/
PetscErrorCode PetscLayoutCreateFromSizes(MPI_Comm comm,PetscInt n,PetscInt N,PetscInt bs,PetscLayout *map)
{
  PetscFunctionBegin;
  PetscCall(PetscLayoutCreate(comm, map));
  PetscCall(PetscLayoutSetLocalSize(*map, n));
  PetscCall(PetscLayoutSetSize(*map, N));
  PetscCall(PetscLayoutSetBlockSize(*map, bs));
  PetscCall(PetscLayoutSetUp(*map));
  PetscFunctionReturn(0);
}

/*@
  PetscLayoutDestroy - Frees a map object and frees its range if that exists.

  Collective

  Input Parameters:
. map - the PetscLayout

  Level: developer

  Note:
  The PetscLayout object and methods are intended to be used in the PETSc Vec and Mat implementations; it is
  recommended they not be used in user codes unless you really gain something in their use.

.seealso: PetscLayoutSetLocalSize(), PetscLayoutSetSize(), PetscLayoutGetSize(), PetscLayoutGetLocalSize(), PetscLayout, PetscLayoutCreate(),
          PetscLayoutGetRange(), PetscLayoutGetRanges(), PetscLayoutSetBlockSize(), PetscLayoutGetBlockSize(), PetscLayoutSetUp()

@*/
PetscErrorCode PetscLayoutDestroy(PetscLayout *map)
{
  PetscFunctionBegin;
  if (!*map) PetscFunctionReturn(0);
  if (!(*map)->refcnt--) {
    if ((*map)->range_alloc) PetscCall(PetscFree((*map)->range));
    PetscCall(ISLocalToGlobalMappingDestroy(&(*map)->mapping));
    PetscCall(PetscFree((*map)));
  }
  *map = NULL;
  PetscFunctionReturn(0);
}

/*@
  PetscLayoutCreateFromRanges - Creates a new PetscLayout with the given ownership ranges and sets it up.

  Collective

  Input Parameters:
+ comm  - the MPI communicator
. range - the array of ownership ranges for each rank with length commsize+1
. mode  - the copy mode for range
- bs    - the block size (or PETSC_DECIDE)

  Output Parameters:
. newmap - the new PetscLayout

  Level: developer

.seealso: PetscLayoutCreate(), PetscLayoutSetLocalSize(), PetscLayoutSetSize(), PetscLayoutGetSize(), PetscLayoutGetLocalSize(), PetscLayout, PetscLayoutDestroy(),
          PetscLayoutGetRange(), PetscLayoutGetRanges(), PetscLayoutSetBlockSize(), PetscLayoutGetBlockSize(), PetscLayoutSetUp(), PetscLayoutCreateFromSizes()

@*/
PetscErrorCode PetscLayoutCreateFromRanges(MPI_Comm comm,const PetscInt range[],PetscCopyMode mode,PetscInt bs,PetscLayout *newmap)
{
  PetscLayout    map;
  PetscMPIInt    rank;

  PetscFunctionBegin;
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  PetscCall(PetscLayoutCreate(comm, &map));
  PetscCall(PetscLayoutSetBlockSize(map, bs));
  switch (mode) {
    case PETSC_COPY_VALUES:
      PetscCall(PetscMalloc1(map->size+1, &map->range));
      PetscCall(PetscArraycpy(map->range, range, map->size+1));
      break;
    case PETSC_USE_POINTER:
      map->range_alloc = PETSC_FALSE;
    default:
      map->range = (PetscInt*) range;
      break;
  }
  map->rstart = map->range[rank];
  map->rend   = map->range[rank+1];
  map->n      = map->rend - map->rstart;
  map->N      = map->range[map->size];
  if (PetscDefined(USE_DEBUG)) {  /* just check that n, N and bs are consistent */
    PetscInt tmp;
    PetscCall(MPIU_Allreduce(&map->n,&tmp,1,MPIU_INT,MPI_SUM,map->comm));
    PetscCheckFalse(tmp != map->N,PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Sum of local lengths %" PetscInt_FMT " does not equal global length %" PetscInt_FMT ", my local length %" PetscInt_FMT ".\nThe provided PetscLayout is wrong.",tmp,map->N,map->n);
    if (map->bs > 1) {
      PetscCheckFalse(map->n % map->bs,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Local size %" PetscInt_FMT " must be divisible by blocksize %" PetscInt_FMT,map->n,map->bs);
    }
    if (map->bs > 1) {
      PetscCheckFalse(map->N % map->bs,map->comm,PETSC_ERR_PLIB,"Global size %" PetscInt_FMT " must be divisible by blocksize %" PetscInt_FMT,map->N,map->bs);
    }
  }
  /* lock the layout */
  map->setupcalled = PETSC_TRUE;
  map->oldn = map->n;
  map->oldN = map->N;
  map->oldbs = map->bs;
  *newmap = map;
  PetscFunctionReturn(0);
}

/*@
  PetscLayoutSetUp - given a map where you have set either the global or local
                     size sets up the map so that it may be used.

  Collective

  Input Parameters:
. map - pointer to the map

  Level: developer

  Notes:
    Typical calling sequence
$ PetscLayoutCreate(MPI_Comm,PetscLayout *);
$ PetscLayoutSetBlockSize(PetscLayout,1);
$ PetscLayoutSetSize(PetscLayout,n) or PetscLayoutSetLocalSize(PetscLayout,N); or both
$ PetscLayoutSetUp(PetscLayout);
$ PetscLayoutGetSize(PetscLayout,PetscInt *);

  If range exists, and local size is not set, everything gets computed from the range.

  If the local size, global size are already set and range exists then this does nothing.

.seealso: PetscLayoutSetLocalSize(), PetscLayoutSetSize(), PetscLayoutGetSize(), PetscLayoutGetLocalSize(), PetscLayout, PetscLayoutDestroy(),
          PetscLayoutGetRange(), PetscLayoutGetRanges(), PetscLayoutSetBlockSize(), PetscLayoutGetBlockSize(), PetscLayoutCreate()
@*/
PetscErrorCode PetscLayoutSetUp(PetscLayout map)
{
  PetscMPIInt    rank;
  PetscInt       p;

  PetscFunctionBegin;
  PetscCheckFalse(map->setupcalled && (map->n != map->oldn || map->N != map->oldN),map->comm,PETSC_ERR_ARG_WRONGSTATE,"Layout is already setup with (local=%" PetscInt_FMT ",global=%" PetscInt_FMT "), cannot call setup again with (local=%" PetscInt_FMT ",global=%" PetscInt_FMT ")", map->oldn, map->oldN, map->n, map->N);
  if (map->setupcalled) PetscFunctionReturn(0);

  if (map->n > 0 && map->bs > 1) {
    PetscCheckFalse(map->n % map->bs,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Local size %" PetscInt_FMT " must be divisible by blocksize %" PetscInt_FMT,map->n,map->bs);
  }
  if (map->N > 0 && map->bs > 1) {
    PetscCheckFalse(map->N % map->bs,map->comm,PETSC_ERR_PLIB,"Global size %" PetscInt_FMT " must be divisible by blocksize %" PetscInt_FMT,map->N,map->bs);
  }

  PetscCallMPI(MPI_Comm_rank(map->comm, &rank));
  if (map->n > 0) map->n = map->n/PetscAbs(map->bs);
  if (map->N > 0) map->N = map->N/PetscAbs(map->bs);
  PetscCall(PetscSplitOwnership(map->comm,&map->n,&map->N));
  map->n = map->n*PetscAbs(map->bs);
  map->N = map->N*PetscAbs(map->bs);
  if (!map->range) {
    PetscCall(PetscMalloc1(map->size+1, &map->range));
  }
  PetscCallMPI(MPI_Allgather(&map->n, 1, MPIU_INT, map->range+1, 1, MPIU_INT, map->comm));

  map->range[0] = 0;
  for (p = 2; p <= map->size; p++) map->range[p] += map->range[p-1];

  map->rstart = map->range[rank];
  map->rend   = map->range[rank+1];

  /* lock the layout */
  map->setupcalled = PETSC_TRUE;
  map->oldn = map->n;
  map->oldN = map->N;
  map->oldbs = map->bs;
  PetscFunctionReturn(0);
}

/*@
  PetscLayoutDuplicate - creates a new PetscLayout with the same information as a given one. If the PetscLayout already exists it is destroyed first.

  Collective on PetscLayout

  Input Parameter:
. in - input PetscLayout to be duplicated

  Output Parameter:
. out - the copy

  Level: developer

  Notes:
    PetscLayoutSetUp() does not need to be called on the resulting PetscLayout

.seealso: PetscLayoutCreate(), PetscLayoutDestroy(), PetscLayoutSetUp(), PetscLayoutReference()
@*/
PetscErrorCode PetscLayoutDuplicate(PetscLayout in,PetscLayout *out)
{
  MPI_Comm       comm = in->comm;

  PetscFunctionBegin;
  PetscCall(PetscLayoutDestroy(out));
  PetscCall(PetscLayoutCreate(comm,out));
  PetscCall(PetscMemcpy(*out,in,sizeof(struct _n_PetscLayout)));
  if (in->range) {
    PetscCall(PetscMalloc1((*out)->size+1,&(*out)->range));
    PetscCall(PetscArraycpy((*out)->range,in->range,(*out)->size+1));
  }
  (*out)->refcnt = 0;
  PetscFunctionReturn(0);
}

/*@
  PetscLayoutReference - Causes a PETSc Vec or Mat to share a PetscLayout with one that already exists. Used by Vec/MatDuplicate_XXX()

  Collective on PetscLayout

  Input Parameter:
. in - input PetscLayout to be copied

  Output Parameter:
. out - the reference location

  Level: developer

  Notes:
    PetscLayoutSetUp() does not need to be called on the resulting PetscLayout

  If the out location already contains a PetscLayout it is destroyed

.seealso: PetscLayoutCreate(), PetscLayoutDestroy(), PetscLayoutSetUp(), PetscLayoutDuplicate()
@*/
PetscErrorCode PetscLayoutReference(PetscLayout in,PetscLayout *out)
{
  PetscFunctionBegin;
  in->refcnt++;
  PetscCall(PetscLayoutDestroy(out));
  *out = in;
  PetscFunctionReturn(0);
}

/*@
  PetscLayoutSetISLocalToGlobalMapping - sets a ISLocalGlobalMapping into a PetscLayout

  Collective on PetscLayout

  Input Parameters:
+ in - input PetscLayout
- ltog - the local to global mapping

  Level: developer

  Notes:
    PetscLayoutSetUp() does not need to be called on the resulting PetscLayout

  If the ltog location already contains a PetscLayout it is destroyed

.seealso: PetscLayoutCreate(), PetscLayoutDestroy(), PetscLayoutSetUp(), PetscLayoutDuplicate()
@*/
PetscErrorCode PetscLayoutSetISLocalToGlobalMapping(PetscLayout in,ISLocalToGlobalMapping ltog)
{
  PetscFunctionBegin;
  if (ltog) {
    PetscInt bs;

    PetscCall(ISLocalToGlobalMappingGetBlockSize(ltog,&bs));
    PetscCheckFalse(in->bs > 0 && (bs != 1) && in->bs != bs,in->comm,PETSC_ERR_PLIB,"Blocksize of layout %" PetscInt_FMT " must match that of mapping %" PetscInt_FMT " (or the latter must be 1)",in->bs,bs);
    PetscCall(PetscObjectReference((PetscObject)ltog));
  }
  PetscCall(ISLocalToGlobalMappingDestroy(&in->mapping));
  in->mapping = ltog;
  PetscFunctionReturn(0);
}

/*@
  PetscLayoutSetLocalSize - Sets the local size for a PetscLayout object.

  Collective on PetscLayout

  Input Parameters:
+ map - pointer to the map
- n - the local size

  Level: developer

  Notes:
  Call this after the call to PetscLayoutCreate()

.seealso: PetscLayoutCreate(), PetscLayoutSetSize(), PetscLayoutGetSize(), PetscLayoutGetLocalSize(), PetscLayoutSetUp()
          PetscLayoutGetRange(), PetscLayoutGetRanges(), PetscLayoutSetBlockSize(), PetscLayoutGetBlockSize()
@*/
PetscErrorCode PetscLayoutSetLocalSize(PetscLayout map,PetscInt n)
{
  PetscFunctionBegin;
  PetscCheckFalse(map->bs > 1 && n % map->bs,map->comm,PETSC_ERR_ARG_INCOMP,"Local size %" PetscInt_FMT " not compatible with block size %" PetscInt_FMT,n,map->bs);
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
PetscErrorCode  PetscLayoutGetLocalSize(PetscLayout map,PetscInt *n)
{
  PetscFunctionBegin;
  *n = map->n;
  PetscFunctionReturn(0);
}

/*@
  PetscLayoutSetSize - Sets the global size for a PetscLayout object.

  Logically Collective on PetscLayout

  Input Parameters:
+ map - pointer to the map
- n - the global size

  Level: developer

  Notes:
  Call this after the call to PetscLayoutCreate()

.seealso: PetscLayoutCreate(), PetscLayoutSetLocalSize(), PetscLayoutGetLocalSize(), PetscLayoutGetSize(), PetscLayoutSetUp()
          PetscLayoutGetRange(), PetscLayoutGetRanges(), PetscLayoutSetBlockSize(), PetscLayoutGetBlockSize()
@*/
PetscErrorCode PetscLayoutSetSize(PetscLayout map,PetscInt n)
{
  PetscFunctionBegin;
  map->N = n;
  PetscFunctionReturn(0);
}

/*@
  PetscLayoutGetSize - Gets the global size for a PetscLayout object.

  Not Collective

  Input Parameters:
. map - pointer to the map

  Output Parameters:
. n - the global size

  Level: developer

  Notes:
  Call this after the call to PetscLayoutSetUp()

.seealso: PetscLayoutCreate(), PetscLayoutSetLocalSize(), PetscLayoutGetLocalSize(), PetscLayoutSetSize(), PetscLayoutSetUp()
          PetscLayoutGetRange(), PetscLayoutGetRanges(), PetscLayoutSetBlockSize(), PetscLayoutGetBlockSize()
@*/
PetscErrorCode PetscLayoutGetSize(PetscLayout map,PetscInt *n)
{
  PetscFunctionBegin;
  *n = map->N;
  PetscFunctionReturn(0);
}

/*@
  PetscLayoutSetBlockSize - Sets the block size for a PetscLayout object.

  Logically Collective on PetscLayout

  Input Parameters:
+ map - pointer to the map
- bs - the size

  Level: developer

  Notes:
  Call this after the call to PetscLayoutCreate()

.seealso: PetscLayoutCreate(), PetscLayoutSetLocalSize(), PetscLayoutGetLocalSize(), PetscLayoutGetBlockSize(),
          PetscLayoutGetRange(), PetscLayoutGetRanges(), PetscLayoutSetSize(), PetscLayoutGetSize(), PetscLayoutSetUp()
@*/
PetscErrorCode PetscLayoutSetBlockSize(PetscLayout map,PetscInt bs)
{
  PetscFunctionBegin;
  if (bs < 0) PetscFunctionReturn(0);
  PetscCheckFalse(map->n > 0 && map->n % bs,PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Local size %" PetscInt_FMT " not compatible with block size %" PetscInt_FMT,map->n,bs);
  if (map->mapping) {
    PetscInt       obs;

    PetscCall(ISLocalToGlobalMappingGetBlockSize(map->mapping,&obs));
    if (obs > 1) {
      PetscCall(ISLocalToGlobalMappingSetBlockSize(map->mapping,bs));
    }
  }
  map->bs = bs;
  PetscFunctionReturn(0);
}

/*@
  PetscLayoutGetBlockSize - Gets the block size for a PetscLayout object.

  Not Collective

  Input Parameters:
. map - pointer to the map

  Output Parameters:
. bs - the size

  Level: developer

  Notes:
  Call this after the call to PetscLayoutSetUp()

.seealso: PetscLayoutCreate(), PetscLayoutSetLocalSize(), PetscLayoutGetLocalSize(), PetscLayoutSetSize(), PetscLayoutSetUp()
          PetscLayoutGetRange(), PetscLayoutGetRanges(), PetscLayoutSetBlockSize(), PetscLayoutGetSize()
@*/
PetscErrorCode PetscLayoutGetBlockSize(PetscLayout map,PetscInt *bs)
{
  PetscFunctionBegin;
  *bs = PetscAbs(map->bs);
  PetscFunctionReturn(0);
}

/*@
  PetscLayoutGetRange - gets the range of values owned by this process

  Not Collective

  Input Parameter:
. map - pointer to the map

  Output Parameters:
+ rstart - first index owned by this process
- rend   - one more than the last index owned by this process

  Level: developer

  Notes:
  Call this after the call to PetscLayoutSetUp()

.seealso: PetscLayoutCreate(), PetscLayoutSetLocalSize(), PetscLayoutGetLocalSize(), PetscLayoutSetSize(),
          PetscLayoutGetSize(), PetscLayoutGetRanges(), PetscLayoutSetBlockSize(), PetscLayoutGetSize(), PetscLayoutSetUp()
@*/
PetscErrorCode PetscLayoutGetRange(PetscLayout map,PetscInt *rstart,PetscInt *rend)
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
PetscErrorCode  PetscLayoutGetRanges(PetscLayout map,const PetscInt *range[])
{
  PetscFunctionBegin;
  *range = map->range;
  PetscFunctionReturn(0);
}

/*@
  PetscLayoutCompare - Compares two layouts

  Not Collective

  Input Parameters:
+ mapa - pointer to the first map
- mapb - pointer to the second map

  Output Parameters:
. congruent - PETSC_TRUE if the two layouts are congruent, PETSC_FALSE otherwise

  Level: beginner

  Notes:

.seealso: PetscLayoutCreate(), PetscLayoutSetLocalSize(), PetscLayoutGetLocalSize(), PetscLayoutGetBlockSize(),
          PetscLayoutGetRange(), PetscLayoutGetRanges(), PetscLayoutSetSize(), PetscLayoutGetSize(), PetscLayoutSetUp()
@*/
PetscErrorCode PetscLayoutCompare(PetscLayout mapa,PetscLayout mapb,PetscBool *congruent)
{
  PetscFunctionBegin;
  *congruent = PETSC_FALSE;
  if (mapa->N == mapb->N && mapa->range && mapb->range && mapa->size == mapb->size) {
    PetscCall(PetscArraycmp(mapa->range,mapb->range,mapa->size+1,congruent));
  }
  PetscFunctionReturn(0);
}
