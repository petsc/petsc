
/*
   This file contains routines for basic map object implementation.
*/

#include <petscis.h> /*I "petscis.h" I*/
#include <petscsf.h>
#include <petsc/private/vecimpl.h>

/*@
  PetscLayoutCreate - Allocates PetscLayout space and sets the map contents to the default.

  Collective on MPI_Comm

  Input Parameters:
+ comm - the MPI communicator
- map - pointer to the map

  Level: advanced

  Notes:
  Typical calling sequence
.vb
       PetscLayoutCreate(MPI_Comm,PetscLayout *);
       PetscLayoutSetBlockSize(PetscLayout,1);
       PetscLayoutSetSize(PetscLayout,N) // or PetscLayoutSetLocalSize(PetscLayout,n);
       PetscLayoutSetUp(PetscLayout);
.ve
  Optionally use any of the following:

+ PetscLayoutGetSize(PetscLayout,PetscInt *);
. PetscLayoutGetLocalSize(PetscLayout,PetscInt *);
. PetscLayoutGetRange(PetscLayout,PetscInt *rstart,PetscInt *rend);
. PetscLayoutGetRanges(PetscLayout,const PetscInt *range[]);
- PetscLayoutDestroy(PetscLayout*);

  The PetscLayout object and methods are intended to be used in the PETSc Vec and Mat implementions; it is often not needed in
  user codes unless you really gain something in their use.

.seealso: PetscLayoutSetLocalSize(), PetscLayoutSetSize(), PetscLayoutGetSize(), PetscLayoutGetLocalSize(), PetscLayout, PetscLayoutDestroy(),
          PetscLayoutGetRange(), PetscLayoutGetRanges(), PetscLayoutSetBlockSize(), PetscLayoutGetBlockSize(), PetscLayoutSetUp()

@*/
PetscErrorCode PetscLayoutCreate(MPI_Comm comm,PetscLayout *map)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscNew(map);CHKERRQ(ierr);

  (*map)->comm   = comm;
  (*map)->bs     = -1;
  (*map)->n      = -1;
  (*map)->N      = -1;
  (*map)->range  = NULL;
  (*map)->rstart = 0;
  (*map)->rend   = 0;
  PetscFunctionReturn(0);
}

/*@
  PetscLayoutDestroy - Frees a map object and frees its range if that exists.

  Collective on MPI_Comm

  Input Parameters:
. map - the PetscLayout

  Level: developer

  Note:
  The PetscLayout object and methods are intended to be used in the PETSc Vec and Mat implementions; it is
  recommended they not be used in user codes unless you really gain something in their use.

.seealso: PetscLayoutSetLocalSize(), PetscLayoutSetSize(), PetscLayoutGetSize(), PetscLayoutGetLocalSize(), PetscLayout, PetscLayoutCreate(),
          PetscLayoutGetRange(), PetscLayoutGetRanges(), PetscLayoutSetBlockSize(), PetscLayoutGetBlockSize(), PetscLayoutSetUp()

@*/
PetscErrorCode PetscLayoutDestroy(PetscLayout *map)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!*map) PetscFunctionReturn(0);
  if (!(*map)->refcnt--) {
    ierr = PetscFree((*map)->range);CHKERRQ(ierr);
    ierr = ISLocalToGlobalMappingDestroy(&(*map)->mapping);CHKERRQ(ierr);
    ierr = PetscFree((*map));CHKERRQ(ierr);
  }
  *map = NULL;
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscLayoutSetUp_SizesFromRanges_Private(PetscLayout map)
{
  PetscMPIInt    rank,size;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(map->comm, &size);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(map->comm, &rank);CHKERRQ(ierr);
  map->rstart = map->range[rank];
  map->rend   = map->range[rank+1];
  map->n      = map->rend - map->rstart;
  map->N      = map->range[size];
#if defined(PETSC_USE_DEBUG)
  /* just check that n, N and bs are consistent */
  {
    PetscInt tmp;
    ierr = MPIU_Allreduce(&map->n,&tmp,1,MPIU_INT,MPI_SUM,map->comm);CHKERRQ(ierr);
    if (tmp != map->N) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Sum of local lengths %D does not equal global length %D, my local length %D.\nThe provided PetscLayout is wrong.",tmp,map->N,map->n);
  }
  if (map->bs > 1) {
    if (map->n % map->bs) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Local size %D must be divisible by blocksize %D",map->n,map->bs);
  }
  if (map->bs > 1) {
    if (map->N % map->bs) SETERRQ2(map->comm,PETSC_ERR_PLIB,"Global size %D must be divisible by blocksize %D",map->N,map->bs);
  }
#endif
  PetscFunctionReturn(0);
}

/*@
  PetscLayoutSetUp - given a map where you have set either the global or local
                     size sets up the map so that it may be used.

  Collective on MPI_Comm

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
  PetscMPIInt    rank,size;
  PetscInt       p;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if ((map->n >= 0) && (map->N >= 0) && (map->range)) PetscFunctionReturn(0);
  if (map->range && map->n < 0) {
    ierr = PetscLayoutSetUp_SizesFromRanges_Private(map);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }

  if (map->n > 0 && map->bs > 1) {
    if (map->n % map->bs) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Local size %D must be divisible by blocksize %D",map->n,map->bs);
  }
  if (map->N > 0 && map->bs > 1) {
    if (map->N % map->bs) SETERRQ2(map->comm,PETSC_ERR_PLIB,"Global size %D must be divisible by blocksize %D",map->N,map->bs);
  }

  ierr = MPI_Comm_size(map->comm, &size);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(map->comm, &rank);CHKERRQ(ierr);
  if (map->n > 0) map->n = map->n/PetscAbs(map->bs);
  if (map->N > 0) map->N = map->N/PetscAbs(map->bs);
  ierr = PetscSplitOwnership(map->comm,&map->n,&map->N);CHKERRQ(ierr);
  map->n = map->n*PetscAbs(map->bs);
  map->N = map->N*PetscAbs(map->bs);
  if (!map->range) {
    ierr = PetscMalloc1(size+1, &map->range);CHKERRQ(ierr);
  }
  ierr = MPI_Allgather(&map->n, 1, MPIU_INT, map->range+1, 1, MPIU_INT, map->comm);CHKERRQ(ierr);

  map->range[0] = 0;
  for (p = 2; p <= size; p++) map->range[p] += map->range[p-1];

  map->rstart = map->range[rank];
  map->rend   = map->range[rank+1];
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
  PetscMPIInt    size;
  PetscErrorCode ierr;
  MPI_Comm       comm = in->comm;

  PetscFunctionBegin;
  ierr = PetscLayoutDestroy(out);CHKERRQ(ierr);
  ierr = PetscLayoutCreate(comm,out);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  ierr = PetscMemcpy(*out,in,sizeof(struct _n_PetscLayout));CHKERRQ(ierr);
  ierr = PetscMalloc1(size+1,&(*out)->range);CHKERRQ(ierr);
  ierr = PetscMemcpy((*out)->range,in->range,(size+1)*sizeof(PetscInt));CHKERRQ(ierr);

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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  in->refcnt++;
  ierr = PetscLayoutDestroy(out);CHKERRQ(ierr);
  *out = in;
  PetscFunctionReturn(0);
}

/*@
  PetscLayoutSetISLocalToGlobalMapping - sets a ISLocalGlobalMapping into a PetscLayout

  Collective on PetscLayout

  Input Parameter:
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
  PetscErrorCode ierr;
  PetscInt       bs;

  PetscFunctionBegin;
  ierr = ISLocalToGlobalMappingGetBlockSize(ltog,&bs);CHKERRQ(ierr);
  if (in->bs > 0 && (bs != 1) && in->bs != bs) SETERRQ2(in->comm,PETSC_ERR_PLIB,"Blocksize of layout %D must match that of mapping %D (or the latter must be 1)",in->bs,bs);
  ierr = PetscObjectReference((PetscObject)ltog);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingDestroy(&in->mapping);CHKERRQ(ierr);
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
  if (map->bs > 1 && n % map->bs) SETERRQ2(map->comm,PETSC_ERR_ARG_INCOMP,"Local size %D not compatible with block size %D",n,map->bs);
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
  if (map->n > 0 && map->n % bs) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Local size %D not compatible with block size %D",map->n,bs);
  if (map->mapping) {
    PetscInt       obs;
    PetscErrorCode ierr;

    ierr = ISLocalToGlobalMappingGetBlockSize(map->mapping,&obs);CHKERRQ(ierr);
    if (obs > 1) {
      ierr = ISLocalToGlobalMappingSetBlockSize(map->mapping,bs);CHKERRQ(ierr);
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

  Input Parameters:
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

/*@C
   PetscSFSetGraphLayout - Set a parallel star forest via global indices and a PetscLayout

   Collective

   Input Arguments:
+  sf - star forest
.  layout - PetscLayout defining the global space
.  nleaves - number of leaf vertices on the current process, each of these references a root on any process
.  ilocal - locations of leaves in leafdata buffers, pass NULL for contiguous storage
.  localmode - copy mode for ilocal
-  iremote - remote locations of root vertices for each leaf on the current process

   Level: intermediate

   Developers Note: Local indices which are the identity permutation in the range [0,nleaves) are discarded as they
   encode contiguous storage. In such case, if localmode is PETSC_OWN_POINTER, the memory is deallocated as it is not
   needed

.seealso: PetscSFCreate(), PetscSFView(), PetscSFSetGraph(), PetscSFGetGraph()
@*/
PetscErrorCode PetscSFSetGraphLayout(PetscSF sf,PetscLayout layout,PetscInt nleaves,const PetscInt *ilocal,PetscCopyMode localmode,const PetscInt *iremote)
{
  PetscErrorCode ierr;
  PetscInt       i,nroots;
  PetscSFNode    *remote;

  PetscFunctionBegin;
  ierr = PetscLayoutGetLocalSize(layout,&nroots);CHKERRQ(ierr);
  ierr = PetscMalloc1(nleaves,&remote);CHKERRQ(ierr);
  for (i=0; i<nleaves; i++) {
    PetscInt owner = -1;
    ierr = PetscLayoutFindOwner(layout,iremote[i],&owner);CHKERRQ(ierr);
    remote[i].rank  = owner;
    remote[i].index = iremote[i] - layout->range[owner];
  }
  ierr = PetscSFSetGraph(sf,nroots,nleaves,ilocal,localmode,remote,PETSC_OWN_POINTER);CHKERRQ(ierr);
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
  PetscErrorCode ierr;
  PetscMPIInt    sizea,sizeb;

  PetscFunctionBegin;
  *congruent = PETSC_FALSE;
  ierr = MPI_Comm_size(mapa->comm,&sizea);CHKERRQ(ierr);
  ierr = MPI_Comm_size(mapb->comm,&sizeb);CHKERRQ(ierr);
  if (mapa->N == mapb->N && mapa->range && mapb->range && sizea == sizeb) {
    ierr = PetscMemcmp(mapa->range,mapb->range,(sizea+1)*sizeof(PetscInt),congruent);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/* TODO: handle nooffprocentries like MatZeroRowsMapLocal_Private, since this code is the same */
PetscErrorCode PetscLayoutMapLocal(PetscLayout map,PetscInt N,const PetscInt idxs[], PetscInt *on,PetscInt **oidxs,PetscInt **ogidxs)
{
  PetscInt      *owners = map->range;
  PetscInt       n      = map->n;
  PetscSF        sf;
  PetscInt      *lidxs,*work = NULL;
  PetscSFNode   *ridxs;
  PetscMPIInt    rank;
  PetscInt       r, p = 0, len = 0;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (on) *on = 0;              /* squelch -Wmaybe-uninitialized */
  /* Create SF where leaves are input idxs and roots are owned idxs */
  ierr = MPI_Comm_rank(map->comm,&rank);CHKERRQ(ierr);
  ierr = PetscMalloc1(n,&lidxs);CHKERRQ(ierr);
  for (r = 0; r < n; ++r) lidxs[r] = -1;
  ierr = PetscMalloc1(N,&ridxs);CHKERRQ(ierr);
  for (r = 0; r < N; ++r) {
    const PetscInt idx = idxs[r];
    if (idx < 0 || map->N <= idx) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Index %D out of range [0,%D)",idx,map->N);
    if (idx < owners[p] || owners[p+1] <= idx) { /* short-circuit the search if the last p owns this idx too */
      ierr = PetscLayoutFindOwner(map,idx,&p);CHKERRQ(ierr);
    }
    ridxs[r].rank = p;
    ridxs[r].index = idxs[r] - owners[p];
  }
  ierr = PetscSFCreate(map->comm,&sf);CHKERRQ(ierr);
  ierr = PetscSFSetGraph(sf,n,N,NULL,PETSC_OWN_POINTER,ridxs,PETSC_OWN_POINTER);CHKERRQ(ierr);
  ierr = PetscSFReduceBegin(sf,MPIU_INT,(PetscInt*)idxs,lidxs,MPI_LOR);CHKERRQ(ierr);
  ierr = PetscSFReduceEnd(sf,MPIU_INT,(PetscInt*)idxs,lidxs,MPI_LOR);CHKERRQ(ierr);
  if (ogidxs) { /* communicate global idxs */
    PetscInt cum = 0,start,*work2;

    ierr = PetscMalloc1(n,&work);CHKERRQ(ierr);
    ierr = PetscCalloc1(N,&work2);CHKERRQ(ierr);
    for (r = 0; r < N; ++r) if (idxs[r] >=0) cum++;
    ierr = MPI_Scan(&cum,&start,1,MPIU_INT,MPI_SUM,map->comm);CHKERRQ(ierr);
    start -= cum;
    cum = 0;
    for (r = 0; r < N; ++r) if (idxs[r] >=0) work2[r] = start+cum++;
    ierr = PetscSFReduceBegin(sf,MPIU_INT,work2,work,MPIU_REPLACE);CHKERRQ(ierr);
    ierr = PetscSFReduceEnd(sf,MPIU_INT,work2,work,MPIU_REPLACE);CHKERRQ(ierr);
    ierr = PetscFree(work2);CHKERRQ(ierr);
  }
  ierr = PetscSFDestroy(&sf);CHKERRQ(ierr);
  /* Compress and put in indices */
  for (r = 0; r < n; ++r)
    if (lidxs[r] >= 0) {
      if (work) work[len] = work[r];
      lidxs[len++] = r;
    }
  if (on) *on = len;
  if (oidxs) *oidxs = lidxs;
  if (ogidxs) *ogidxs = work;
  PetscFunctionReturn(0);
}
