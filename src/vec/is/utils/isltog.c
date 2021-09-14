
#include <petsc/private/isimpl.h>    /*I "petscis.h"  I*/
#include <petsc/private/hashmapi.h>
#include <petscsf.h>
#include <petscviewer.h>

PetscClassId IS_LTOGM_CLASSID;
static PetscErrorCode  ISLocalToGlobalMappingGetBlockInfo_Private(ISLocalToGlobalMapping,PetscInt*,PetscInt**,PetscInt**,PetscInt***);

typedef struct {
  PetscInt *globals;
} ISLocalToGlobalMapping_Basic;

typedef struct {
  PetscHMapI globalht;
} ISLocalToGlobalMapping_Hash;

/*@C
  ISGetPointRange - Returns a description of the points in an IS suitable for traversal

  Not collective

  Input Parameter:
. pointIS - The IS object

  Output Parameters:
+ pStart - The first index, see notes
. pEnd   - One past the last index, see notes
- points - The indices, see notes

  Notes:
  If the IS contains contiguous indices in an ISSTRIDE, then the indices are contained in [pStart, pEnd) and points = NULL. Otherwise, pStart = 0, pEnd = numIndices, and points is an array of the indices. This supports the following pattern
$ ISGetPointRange(is, &pStart, &pEnd, &points);
$ for (p = pStart; p < pEnd; ++p) {
$   const PetscInt point = points ? points[p] : p;
$ }
$ ISRestorePointRange(is, &pstart, &pEnd, &points);

  Level: intermediate

.seealso: ISRestorePointRange(), ISGetPointSubrange(), ISGetIndices(), ISCreateStride()
@*/
PetscErrorCode ISGetPointRange(IS pointIS, PetscInt *pStart, PetscInt *pEnd, const PetscInt **points)
{
  PetscInt       numCells, step = 1;
  PetscBool      isStride;
  PetscErrorCode ierr;

  PetscFunctionBeginHot;
  *pStart = 0;
  *points = NULL;
  ierr = ISGetLocalSize(pointIS, &numCells);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject) pointIS, ISSTRIDE, &isStride);CHKERRQ(ierr);
  if (isStride) {ierr = ISStrideGetInfo(pointIS, pStart, &step);CHKERRQ(ierr);}
  *pEnd   = *pStart + numCells;
  if (!isStride || step != 1) {ierr = ISGetIndices(pointIS, points);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

/*@C
  ISRestorePointRange - Destroys the traversal description

  Not collective

  Input Parameters:
+ pointIS - The IS object
. pStart  - The first index, from ISGetPointRange()
. pEnd    - One past the last index, from ISGetPointRange()
- points  - The indices, from ISGetPointRange()

  Notes:
  If the IS contains contiguous indices in an ISSTRIDE, then the indices are contained in [pStart, pEnd) and points = NULL. Otherwise, pStart = 0, pEnd = numIndices, and points is an array of the indices. This supports the following pattern
$ ISGetPointRange(is, &pStart, &pEnd, &points);
$ for (p = pStart; p < pEnd; ++p) {
$   const PetscInt point = points ? points[p] : p;
$ }
$ ISRestorePointRange(is, &pstart, &pEnd, &points);

  Level: intermediate

.seealso: ISGetPointRange(), ISGetPointSubrange(), ISGetIndices(), ISCreateStride()
@*/
PetscErrorCode ISRestorePointRange(IS pointIS, PetscInt *pStart, PetscInt *pEnd, const PetscInt **points)
{
  PetscInt       step = 1;
  PetscBool      isStride;
  PetscErrorCode ierr;

  PetscFunctionBeginHot;
  ierr = PetscObjectTypeCompare((PetscObject) pointIS, ISSTRIDE, &isStride);CHKERRQ(ierr);
  if (isStride) {ierr = ISStrideGetInfo(pointIS, pStart, &step);CHKERRQ(ierr);}
  if (!isStride || step != 1) {ierr = ISGetIndices(pointIS, points);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

/*@C
  ISGetPointSubrange - Configures the input IS to be a subrange for the traversal information given

  Not collective

  Input Parameters:
+ subpointIS - The IS object to be configured
. pStar   t  - The first index of the subrange
. pEnd       - One past the last index for the subrange
- points     - The indices for the entire range, from ISGetPointRange()

  Output Parameters:
. subpointIS - The IS object now configured to be a subrange

  Notes:
  The input IS will now respond properly to calls to ISGetPointRange() and return the subrange.

  Level: intermediate

.seealso: ISGetPointRange(), ISRestorePointRange(), ISGetIndices(), ISCreateStride()
@*/
PetscErrorCode ISGetPointSubrange(IS subpointIS, PetscInt pStart, PetscInt pEnd, const PetscInt *points)
{
  PetscErrorCode ierr;

  PetscFunctionBeginHot;
  if (points) {
    ierr = ISSetType(subpointIS, ISGENERAL);CHKERRQ(ierr);
    ierr = ISGeneralSetIndices(subpointIS, pEnd-pStart, &points[pStart], PETSC_USE_POINTER);CHKERRQ(ierr);
  } else {
    ierr = ISSetType(subpointIS, ISSTRIDE);CHKERRQ(ierr);
    ierr = ISStrideSetStride(subpointIS, pEnd-pStart, pStart, 1);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/* -----------------------------------------------------------------------------------------*/

/*
    Creates the global mapping information in the ISLocalToGlobalMapping structure

    If the user has not selected how to handle the global to local mapping then use HASH for "large" problems
*/
static PetscErrorCode ISGlobalToLocalMappingSetUp(ISLocalToGlobalMapping mapping)
{
  PetscInt       i,*idx = mapping->indices,n = mapping->n,end,start;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (mapping->data) PetscFunctionReturn(0);
  end   = 0;
  start = PETSC_MAX_INT;

  for (i=0; i<n; i++) {
    if (idx[i] < 0) continue;
    if (idx[i] < start) start = idx[i];
    if (idx[i] > end)   end   = idx[i];
  }
  if (start > end) {start = 0; end = -1;}
  mapping->globalstart = start;
  mapping->globalend   = end;
  if (!((PetscObject)mapping)->type_name) {
    if ((end - start) > PetscMax(4*n,1000000)) {
      ierr = ISLocalToGlobalMappingSetType(mapping,ISLOCALTOGLOBALMAPPINGHASH);CHKERRQ(ierr);
    } else {
      ierr = ISLocalToGlobalMappingSetType(mapping,ISLOCALTOGLOBALMAPPINGBASIC);CHKERRQ(ierr);
    }
  }
  ierr = (*mapping->ops->globaltolocalmappingsetup)(mapping);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode ISGlobalToLocalMappingSetUp_Basic(ISLocalToGlobalMapping mapping)
{
  PetscErrorCode              ierr;
  PetscInt                    i,*idx = mapping->indices,n = mapping->n,end,start,*globals;
  ISLocalToGlobalMapping_Basic *map;

  PetscFunctionBegin;
  start            = mapping->globalstart;
  end              = mapping->globalend;
  ierr             = PetscNew(&map);CHKERRQ(ierr);
  ierr             = PetscMalloc1(end-start+2,&globals);CHKERRQ(ierr);
  map->globals     = globals;
  for (i=0; i<end-start+1; i++) globals[i] = -1;
  for (i=0; i<n; i++) {
    if (idx[i] < 0) continue;
    globals[idx[i] - start] = i;
  }
  mapping->data = (void*)map;
  ierr = PetscLogObjectMemory((PetscObject)mapping,(end-start+1)*sizeof(PetscInt));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode ISGlobalToLocalMappingSetUp_Hash(ISLocalToGlobalMapping mapping)
{
  PetscErrorCode              ierr;
  PetscInt                    i,*idx = mapping->indices,n = mapping->n;
  ISLocalToGlobalMapping_Hash *map;

  PetscFunctionBegin;
  ierr = PetscNew(&map);CHKERRQ(ierr);
  ierr = PetscHMapICreate(&map->globalht);CHKERRQ(ierr);
  for (i=0; i<n; i++) {
    if (idx[i] < 0) continue;
    ierr = PetscHMapISet(map->globalht,idx[i],i);CHKERRQ(ierr);
  }
  mapping->data = (void*)map;
  ierr = PetscLogObjectMemory((PetscObject)mapping,2*n*sizeof(PetscInt));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode ISLocalToGlobalMappingDestroy_Basic(ISLocalToGlobalMapping mapping)
{
  PetscErrorCode              ierr;
  ISLocalToGlobalMapping_Basic *map  = (ISLocalToGlobalMapping_Basic *)mapping->data;

  PetscFunctionBegin;
  if (!map) PetscFunctionReturn(0);
  ierr = PetscFree(map->globals);CHKERRQ(ierr);
  ierr = PetscFree(mapping->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode ISLocalToGlobalMappingDestroy_Hash(ISLocalToGlobalMapping mapping)
{
  PetscErrorCode              ierr;
  ISLocalToGlobalMapping_Hash *map  = (ISLocalToGlobalMapping_Hash*)mapping->data;

  PetscFunctionBegin;
  if (!map) PetscFunctionReturn(0);
  ierr = PetscHMapIDestroy(&map->globalht);CHKERRQ(ierr);
  ierr = PetscFree(mapping->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#define GTOLTYPE _Basic
#define GTOLNAME _Basic
#define GTOLBS mapping->bs
#define GTOL(g, local) do {                  \
    local = map->globals[g/bs - start];      \
    if (local >= 0) local = bs*local + (g % bs); \
  } while (0)

#include <../src/vec/is/utils/isltog.h>

#define GTOLTYPE _Basic
#define GTOLNAME Block_Basic
#define GTOLBS 1
#define GTOL(g, local) do {                  \
    local = map->globals[g - start];         \
  } while (0)
#include <../src/vec/is/utils/isltog.h>

#define GTOLTYPE _Hash
#define GTOLNAME _Hash
#define GTOLBS mapping->bs
#define GTOL(g, local) do {                         \
    (void)PetscHMapIGet(map->globalht,g/bs,&local); \
    if (local >= 0) local = bs*local + (g % bs);   \
   } while (0)
#include <../src/vec/is/utils/isltog.h>

#define GTOLTYPE _Hash
#define GTOLNAME Block_Hash
#define GTOLBS 1
#define GTOL(g, local) do {                         \
    (void)PetscHMapIGet(map->globalht,g,&local);    \
  } while (0)
#include <../src/vec/is/utils/isltog.h>

/*@
    ISLocalToGlobalMappingDuplicate - Duplicates the local to global mapping object

    Not Collective

    Input Parameter:
.   ltog - local to global mapping

    Output Parameter:
.   nltog - the duplicated local to global mapping

    Level: advanced

.seealso: ISLocalToGlobalMappingDestroy(), ISLocalToGlobalMappingCreate()
@*/
PetscErrorCode  ISLocalToGlobalMappingDuplicate(ISLocalToGlobalMapping ltog,ISLocalToGlobalMapping* nltog)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ltog,IS_LTOGM_CLASSID,1);
  ierr = ISLocalToGlobalMappingCreate(PetscObjectComm((PetscObject)ltog),ltog->bs,ltog->n,ltog->indices,PETSC_COPY_VALUES,nltog);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
    ISLocalToGlobalMappingGetSize - Gets the local size of a local to global mapping

    Not Collective

    Input Parameter:
.   ltog - local to global mapping

    Output Parameter:
.   n - the number of entries in the local mapping, ISLocalToGlobalMappingGetIndices() returns an array of this length

    Level: advanced

.seealso: ISLocalToGlobalMappingDestroy(), ISLocalToGlobalMappingCreate()
@*/
PetscErrorCode  ISLocalToGlobalMappingGetSize(ISLocalToGlobalMapping mapping,PetscInt *n)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mapping,IS_LTOGM_CLASSID,1);
  PetscValidIntPointer(n,2);
  *n = mapping->bs*mapping->n;
  PetscFunctionReturn(0);
}

/*@C
   ISLocalToGlobalMappingViewFromOptions - View from Options

   Collective on ISLocalToGlobalMapping

   Input Parameters:
+  A - the local to global mapping object
.  obj - Optional object
-  name - command line option

   Level: intermediate
.seealso:  ISLocalToGlobalMapping, ISLocalToGlobalMappingView, PetscObjectViewFromOptions(), ISLocalToGlobalMappingCreate()
@*/
PetscErrorCode  ISLocalToGlobalMappingViewFromOptions(ISLocalToGlobalMapping A,PetscObject obj,const char name[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,IS_LTOGM_CLASSID,1);
  ierr = PetscObjectViewFromOptions((PetscObject)A,obj,name);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
    ISLocalToGlobalMappingView - View a local to global mapping

    Not Collective

    Input Parameters:
+   ltog - local to global mapping
-   viewer - viewer

    Level: advanced

.seealso: ISLocalToGlobalMappingDestroy(), ISLocalToGlobalMappingCreate()
@*/
PetscErrorCode  ISLocalToGlobalMappingView(ISLocalToGlobalMapping mapping,PetscViewer viewer)
{
  PetscInt       i;
  PetscMPIInt    rank;
  PetscBool      iascii;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mapping,IS_LTOGM_CLASSID,1);
  if (!viewer) {
    ierr = PetscViewerASCIIGetStdout(PetscObjectComm((PetscObject)mapping),&viewer);CHKERRQ(ierr);
  }
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,2);

  ierr = MPI_Comm_rank(PetscObjectComm((PetscObject)mapping),&rank);CHKERRMPI(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    ierr = PetscObjectPrintClassNamePrefixType((PetscObject)mapping,viewer);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPushSynchronized(viewer);CHKERRQ(ierr);
    for (i=0; i<mapping->n; i++) {
      ierr = PetscViewerASCIISynchronizedPrintf(viewer,"[%d] %D %D\n",rank,i,mapping->indices[i]);CHKERRQ(ierr);
    }
    ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPopSynchronized(viewer);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*@
    ISLocalToGlobalMappingCreateIS - Creates a mapping between a local (0 to n)
    ordering and a global parallel ordering.

    Not collective

    Input Parameter:
.   is - index set containing the global numbers for each local number

    Output Parameter:
.   mapping - new mapping data structure

    Notes:
    the block size of the IS determines the block size of the mapping
    Level: advanced

.seealso: ISLocalToGlobalMappingDestroy(), ISLocalToGlobalMappingCreate(), ISLocalToGlobalMappingSetFromOptions()
@*/
PetscErrorCode  ISLocalToGlobalMappingCreateIS(IS is,ISLocalToGlobalMapping *mapping)
{
  PetscErrorCode ierr;
  PetscInt       n,bs;
  const PetscInt *indices;
  MPI_Comm       comm;
  PetscBool      isblock;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(is,IS_CLASSID,1);
  PetscValidPointer(mapping,2);

  ierr = PetscObjectGetComm((PetscObject)is,&comm);CHKERRQ(ierr);
  ierr = ISGetLocalSize(is,&n);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)is,ISBLOCK,&isblock);CHKERRQ(ierr);
  if (!isblock) {
    ierr = ISGetIndices(is,&indices);CHKERRQ(ierr);
    ierr = ISLocalToGlobalMappingCreate(comm,1,n,indices,PETSC_COPY_VALUES,mapping);CHKERRQ(ierr);
    ierr = ISRestoreIndices(is,&indices);CHKERRQ(ierr);
  } else {
    ierr = ISGetBlockSize(is,&bs);CHKERRQ(ierr);
    ierr = ISBlockGetIndices(is,&indices);CHKERRQ(ierr);
    ierr = ISLocalToGlobalMappingCreate(comm,bs,n/bs,indices,PETSC_COPY_VALUES,mapping);CHKERRQ(ierr);
    ierr = ISBlockRestoreIndices(is,&indices);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*@C
    ISLocalToGlobalMappingCreateSF - Creates a mapping between a local (0 to n)
    ordering and a global parallel ordering.

    Collective

    Input Parameters:
+   sf - star forest mapping contiguous local indices to (rank, offset)
-   start - first global index on this process

    Output Parameter:
.   mapping - new mapping data structure

    Level: advanced

.seealso: ISLocalToGlobalMappingDestroy(), ISLocalToGlobalMappingCreate(), ISLocalToGlobalMappingCreateIS(), ISLocalToGlobalMappingSetFromOptions()
@*/
PetscErrorCode ISLocalToGlobalMappingCreateSF(PetscSF sf,PetscInt start,ISLocalToGlobalMapping *mapping)
{
  PetscErrorCode ierr;
  PetscInt       i,maxlocal,nroots,nleaves,*globals,*ltog;
  const PetscInt *ilocal;
  MPI_Comm       comm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sf,PETSCSF_CLASSID,1);
  PetscValidPointer(mapping,3);

  ierr = PetscObjectGetComm((PetscObject)sf,&comm);CHKERRQ(ierr);
  ierr = PetscSFGetGraph(sf,&nroots,&nleaves,&ilocal,NULL);CHKERRQ(ierr);
  if (ilocal) {
    for (i=0,maxlocal=0; i<nleaves; i++) maxlocal = PetscMax(maxlocal,ilocal[i]+1);
  }
  else maxlocal = nleaves;
  ierr = PetscMalloc1(nroots,&globals);CHKERRQ(ierr);
  ierr = PetscMalloc1(maxlocal,&ltog);CHKERRQ(ierr);
  for (i=0; i<nroots; i++) globals[i] = start + i;
  for (i=0; i<maxlocal; i++) ltog[i] = -1;
  ierr = PetscSFBcastBegin(sf,MPIU_INT,globals,ltog,MPI_REPLACE);CHKERRQ(ierr);
  ierr = PetscSFBcastEnd(sf,MPIU_INT,globals,ltog,MPI_REPLACE);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingCreate(comm,1,maxlocal,ltog,PETSC_OWN_POINTER,mapping);CHKERRQ(ierr);
  ierr = PetscFree(globals);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
    ISLocalToGlobalMappingSetBlockSize - Sets the blocksize of the mapping

    Not collective

    Input Parameters:
+   mapping - mapping data structure
-   bs - the blocksize

    Level: advanced

.seealso: ISLocalToGlobalMappingDestroy(), ISLocalToGlobalMappingCreateIS()
@*/
PetscErrorCode  ISLocalToGlobalMappingSetBlockSize(ISLocalToGlobalMapping mapping,PetscInt bs)
{
  PetscInt       *nid;
  const PetscInt *oid;
  PetscInt       i,cn,on,obs,nn;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mapping,IS_LTOGM_CLASSID,1);
  if (bs < 1) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Invalid block size %D",bs);
  if (bs == mapping->bs) PetscFunctionReturn(0);
  on  = mapping->n;
  obs = mapping->bs;
  oid = mapping->indices;
  nn  = (on*obs)/bs;
  if ((on*obs)%bs) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Block size %D is inconsistent with block size %D and number of block indices %D",bs,obs,on);

  ierr = PetscMalloc1(nn,&nid);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingGetIndices(mapping,&oid);CHKERRQ(ierr);
  for (i=0;i<nn;i++) {
    PetscInt j;
    for (j=0,cn=0;j<bs-1;j++) {
      if (oid[i*bs+j] < 0) { cn++; continue; }
      if (oid[i*bs+j] != oid[i*bs+j+1]-1) SETERRQ4(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Block sizes %D and %D are incompatible with the block indices: non consecutive indices %D %D",bs,obs,oid[i*bs+j],oid[i*bs+j+1]);
    }
    if (oid[i*bs+j] < 0) cn++;
    if (cn) {
      if (cn != bs) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Block sizes %D and %D are incompatible with the block indices: invalid number of negative entries in block %D",bs,obs,cn);
      nid[i] = -1;
    } else {
      nid[i] = oid[i*bs]/bs;
    }
  }
  ierr = ISLocalToGlobalMappingRestoreIndices(mapping,&oid);CHKERRQ(ierr);

  mapping->n           = nn;
  mapping->bs          = bs;
  ierr                 = PetscFree(mapping->indices);CHKERRQ(ierr);
  mapping->indices     = nid;
  mapping->globalstart = 0;
  mapping->globalend   = 0;

  /* reset the cached information */
  ierr = PetscFree(mapping->info_procs);CHKERRQ(ierr);
  ierr = PetscFree(mapping->info_numprocs);CHKERRQ(ierr);
  if (mapping->info_indices) {
    PetscInt i;

    ierr = PetscFree((mapping->info_indices)[0]);CHKERRQ(ierr);
    for (i=1; i<mapping->info_nproc; i++) {
      ierr = PetscFree(mapping->info_indices[i]);CHKERRQ(ierr);
    }
    ierr = PetscFree(mapping->info_indices);CHKERRQ(ierr);
  }
  mapping->info_cached = PETSC_FALSE;

  if (mapping->ops->destroy) {
    ierr = (*mapping->ops->destroy)(mapping);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*@
    ISLocalToGlobalMappingGetBlockSize - Gets the blocksize of the mapping
    ordering and a global parallel ordering.

    Not Collective

    Input Parameters:
.   mapping - mapping data structure

    Output Parameter:
.   bs - the blocksize

    Level: advanced

.seealso: ISLocalToGlobalMappingDestroy(), ISLocalToGlobalMappingCreateIS()
@*/
PetscErrorCode  ISLocalToGlobalMappingGetBlockSize(ISLocalToGlobalMapping mapping,PetscInt *bs)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mapping,IS_LTOGM_CLASSID,1);
  *bs = mapping->bs;
  PetscFunctionReturn(0);
}

/*@
    ISLocalToGlobalMappingCreate - Creates a mapping between a local (0 to n)
    ordering and a global parallel ordering.

    Not Collective, but communicator may have more than one process

    Input Parameters:
+   comm - MPI communicator
.   bs - the block size
.   n - the number of local elements divided by the block size, or equivalently the number of block indices
.   indices - the global index for each local element, these do not need to be in increasing order (sorted), these values should not be scaled (i.e. multiplied) by the blocksize bs
-   mode - see PetscCopyMode

    Output Parameter:
.   mapping - new mapping data structure

    Notes:
    There is one integer value in indices per block and it represents the actual indices bs*idx + j, where j=0,..,bs-1

    For "small" problems when using ISGlobalToLocalMappingApply() and ISGlobalToLocalMappingApplyBlock(), the ISLocalToGlobalMappingType of ISLOCALTOGLOBALMAPPINGBASIC will be used;
    this uses more memory but is faster; this approach is not scalable for extremely large mappings. For large problems ISLOCALTOGLOBALMAPPINGHASH is used, this is scalable.
    Use ISLocalToGlobalMappingSetType() or call ISLocalToGlobalMappingSetFromOptions() with the option -islocaltoglobalmapping_type <basic,hash> to control which is used.

    Level: advanced

.seealso: ISLocalToGlobalMappingDestroy(), ISLocalToGlobalMappingCreateIS(), ISLocalToGlobalMappingSetFromOptions(), ISLOCALTOGLOBALMAPPINGBASIC, ISLOCALTOGLOBALMAPPINGHASH
          ISLocalToGlobalMappingSetType(), ISLocalToGlobalMappingType
@*/
PetscErrorCode  ISLocalToGlobalMappingCreate(MPI_Comm comm,PetscInt bs,PetscInt n,const PetscInt indices[],PetscCopyMode mode,ISLocalToGlobalMapping *mapping)
{
  PetscErrorCode ierr;
  PetscInt       *in;

  PetscFunctionBegin;
  if (n) PetscValidIntPointer(indices,4);
  PetscValidPointer(mapping,6);

  *mapping = NULL;
  ierr = ISInitializePackage();CHKERRQ(ierr);

  ierr = PetscHeaderCreate(*mapping,IS_LTOGM_CLASSID,"ISLocalToGlobalMapping","Local to global mapping","IS",
                           comm,ISLocalToGlobalMappingDestroy,ISLocalToGlobalMappingView);CHKERRQ(ierr);
  (*mapping)->n             = n;
  (*mapping)->bs            = bs;
  (*mapping)->info_cached   = PETSC_FALSE;
  (*mapping)->info_free     = PETSC_FALSE;
  (*mapping)->info_procs    = NULL;
  (*mapping)->info_numprocs = NULL;
  (*mapping)->info_indices  = NULL;
  (*mapping)->info_nodec    = NULL;
  (*mapping)->info_nodei    = NULL;

  (*mapping)->ops->globaltolocalmappingapply      = NULL;
  (*mapping)->ops->globaltolocalmappingapplyblock = NULL;
  (*mapping)->ops->destroy                        = NULL;
  if (mode == PETSC_COPY_VALUES) {
    ierr = PetscMalloc1(n,&in);CHKERRQ(ierr);
    ierr = PetscArraycpy(in,indices,n);CHKERRQ(ierr);
    (*mapping)->indices = in;
    ierr = PetscLogObjectMemory((PetscObject)*mapping,n*sizeof(PetscInt));CHKERRQ(ierr);
  } else if (mode == PETSC_OWN_POINTER) {
    (*mapping)->indices = (PetscInt*)indices;
    ierr = PetscLogObjectMemory((PetscObject)*mapping,n*sizeof(PetscInt));CHKERRQ(ierr);
  }
  else SETERRQ(comm,PETSC_ERR_SUP,"Cannot currently use PETSC_USE_POINTER");
  PetscFunctionReturn(0);
}

PetscFunctionList ISLocalToGlobalMappingList = NULL;

/*@
   ISLocalToGlobalMappingSetFromOptions - Set mapping options from the options database.

   Not collective

   Input Parameters:
.  mapping - mapping data structure

   Level: advanced

@*/
PetscErrorCode ISLocalToGlobalMappingSetFromOptions(ISLocalToGlobalMapping mapping)
{
  PetscErrorCode             ierr;
  char                       type[256];
  ISLocalToGlobalMappingType defaulttype = "Not set";
  PetscBool                  flg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mapping,IS_LTOGM_CLASSID,1);
  ierr = ISLocalToGlobalMappingRegisterAll();CHKERRQ(ierr);
  ierr = PetscObjectOptionsBegin((PetscObject)mapping);CHKERRQ(ierr);
  ierr = PetscOptionsFList("-islocaltoglobalmapping_type","ISLocalToGlobalMapping method","ISLocalToGlobalMappingSetType",ISLocalToGlobalMappingList,(char*)(((PetscObject)mapping)->type_name) ? ((PetscObject)mapping)->type_name : defaulttype,type,256,&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = ISLocalToGlobalMappingSetType(mapping,type);CHKERRQ(ierr);
  }
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
   ISLocalToGlobalMappingDestroy - Destroys a mapping between a local (0 to n)
   ordering and a global parallel ordering.

   Note Collective

   Input Parameters:
.  mapping - mapping data structure

   Level: advanced

.seealso: ISLocalToGlobalMappingCreate()
@*/
PetscErrorCode  ISLocalToGlobalMappingDestroy(ISLocalToGlobalMapping *mapping)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!*mapping) PetscFunctionReturn(0);
  PetscValidHeaderSpecific((*mapping),IS_LTOGM_CLASSID,1);
  if (--((PetscObject)(*mapping))->refct > 0) {*mapping = NULL;PetscFunctionReturn(0);}
  ierr = PetscFree((*mapping)->indices);CHKERRQ(ierr);
  ierr = PetscFree((*mapping)->info_procs);CHKERRQ(ierr);
  ierr = PetscFree((*mapping)->info_numprocs);CHKERRQ(ierr);
  if ((*mapping)->info_indices) {
    PetscInt i;

    ierr = PetscFree(((*mapping)->info_indices)[0]);CHKERRQ(ierr);
    for (i=1; i<(*mapping)->info_nproc; i++) {
      ierr = PetscFree(((*mapping)->info_indices)[i]);CHKERRQ(ierr);
    }
    ierr = PetscFree((*mapping)->info_indices);CHKERRQ(ierr);
  }
  if ((*mapping)->info_nodei) {
    ierr = PetscFree(((*mapping)->info_nodei)[0]);CHKERRQ(ierr);
  }
  ierr = PetscFree2((*mapping)->info_nodec,(*mapping)->info_nodei);CHKERRQ(ierr);
  if ((*mapping)->ops->destroy) {
    ierr = (*(*mapping)->ops->destroy)(*mapping);CHKERRQ(ierr);
  }
  ierr     = PetscHeaderDestroy(mapping);CHKERRQ(ierr);
  *mapping = NULL;
  PetscFunctionReturn(0);
}

/*@
    ISLocalToGlobalMappingApplyIS - Creates from an IS in the local numbering
    a new index set using the global numbering defined in an ISLocalToGlobalMapping
    context.

    Collective on is

    Input Parameters:
+   mapping - mapping between local and global numbering
-   is - index set in local numbering

    Output Parameters:
.   newis - index set in global numbering

    Notes:
    The output IS will have the same communicator of the input IS.

    Level: advanced

.seealso: ISLocalToGlobalMappingApply(), ISLocalToGlobalMappingCreate(),
          ISLocalToGlobalMappingDestroy(), ISGlobalToLocalMappingApply()
@*/
PetscErrorCode  ISLocalToGlobalMappingApplyIS(ISLocalToGlobalMapping mapping,IS is,IS *newis)
{
  PetscErrorCode ierr;
  PetscInt       n,*idxout;
  const PetscInt *idxin;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mapping,IS_LTOGM_CLASSID,1);
  PetscValidHeaderSpecific(is,IS_CLASSID,2);
  PetscValidPointer(newis,3);

  ierr = ISGetLocalSize(is,&n);CHKERRQ(ierr);
  ierr = ISGetIndices(is,&idxin);CHKERRQ(ierr);
  ierr = PetscMalloc1(n,&idxout);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingApply(mapping,n,idxin,idxout);CHKERRQ(ierr);
  ierr = ISRestoreIndices(is,&idxin);CHKERRQ(ierr);
  ierr = ISCreateGeneral(PetscObjectComm((PetscObject)is),n,idxout,PETSC_OWN_POINTER,newis);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
   ISLocalToGlobalMappingApply - Takes a list of integers in a local numbering
   and converts them to the global numbering.

   Not collective

   Input Parameters:
+  mapping - the local to global mapping context
.  N - number of integers
-  in - input indices in local numbering

   Output Parameter:
.  out - indices in global numbering

   Notes:
   The in and out array parameters may be identical.

   Level: advanced

.seealso: ISLocalToGlobalMappingApplyBlock(), ISLocalToGlobalMappingCreate(),ISLocalToGlobalMappingDestroy(),
          ISLocalToGlobalMappingApplyIS(),AOCreateBasic(),AOApplicationToPetsc(),
          AOPetscToApplication(), ISGlobalToLocalMappingApply()

@*/
PetscErrorCode ISLocalToGlobalMappingApply(ISLocalToGlobalMapping mapping,PetscInt N,const PetscInt in[],PetscInt out[])
{
  PetscInt i,bs,Nmax;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mapping,IS_LTOGM_CLASSID,1);
  bs   = mapping->bs;
  Nmax = bs*mapping->n;
  if (bs == 1) {
    const PetscInt *idx = mapping->indices;
    for (i=0; i<N; i++) {
      if (in[i] < 0) {
        out[i] = in[i];
        continue;
      }
      if (in[i] >= Nmax) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Local index %D too large %D (max) at %D",in[i],Nmax-1,i);
      out[i] = idx[in[i]];
    }
  } else {
    const PetscInt *idx = mapping->indices;
    for (i=0; i<N; i++) {
      if (in[i] < 0) {
        out[i] = in[i];
        continue;
      }
      if (in[i] >= Nmax) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Local index %D too large %D (max) at %D",in[i],Nmax-1,i);
      out[i] = idx[in[i]/bs]*bs + (in[i] % bs);
    }
  }
  PetscFunctionReturn(0);
}

/*@
   ISLocalToGlobalMappingApplyBlock - Takes a list of integers in a local block numbering and converts them to the global block numbering

   Not collective

   Input Parameters:
+  mapping - the local to global mapping context
.  N - number of integers
-  in - input indices in local block numbering

   Output Parameter:
.  out - indices in global block numbering

   Notes:
   The in and out array parameters may be identical.

   Example:
     If the index values are {0,1,6,7} set with a call to ISLocalToGlobalMappingCreate(PETSC_COMM_SELF,2,2,{0,3}) then the mapping applied to 0
     (the first block) would produce 0 and the mapping applied to 1 (the second block) would produce 3.

   Level: advanced

.seealso: ISLocalToGlobalMappingApply(), ISLocalToGlobalMappingCreate(),ISLocalToGlobalMappingDestroy(),
          ISLocalToGlobalMappingApplyIS(),AOCreateBasic(),AOApplicationToPetsc(),
          AOPetscToApplication(), ISGlobalToLocalMappingApply()

@*/
PetscErrorCode ISLocalToGlobalMappingApplyBlock(ISLocalToGlobalMapping mapping,PetscInt N,const PetscInt in[],PetscInt out[])
{

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mapping,IS_LTOGM_CLASSID,1);
  {
    PetscInt i,Nmax = mapping->n;
    const PetscInt *idx = mapping->indices;

    for (i=0; i<N; i++) {
      if (in[i] < 0) {
        out[i] = in[i];
        continue;
      }
      if (in[i] >= Nmax) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Local block index %D too large %D (max) at %D",in[i],Nmax-1,i);
      out[i] = idx[in[i]];
    }
  }
  PetscFunctionReturn(0);
}

/*@
    ISGlobalToLocalMappingApply - Provides the local numbering for a list of integers
    specified with a global numbering.

    Not collective

    Input Parameters:
+   mapping - mapping between local and global numbering
.   type - IS_GTOLM_MASK - maps global indices with no local value to -1 in the output list (i.e., mask them)
           IS_GTOLM_DROP - drops the indices with no local value from the output list
.   n - number of global indices to map
-   idx - global indices to map

    Output Parameters:
+   nout - number of indices in output array (if type == IS_GTOLM_MASK then nout = n)
-   idxout - local index of each global index, one must pass in an array long enough
             to hold all the indices. You can call ISGlobalToLocalMappingApply() with
             idxout == NULL to determine the required length (returned in nout)
             and then allocate the required space and call ISGlobalToLocalMappingApply()
             a second time to set the values.

    Notes:
    Either nout or idxout may be NULL. idx and idxout may be identical.

    For "small" problems when using ISGlobalToLocalMappingApply() and ISGlobalToLocalMappingApplyBlock(), the ISLocalToGlobalMappingType of ISLOCALTOGLOBALMAPPINGBASIC will be used;
    this uses more memory but is faster; this approach is not scalable for extremely large mappings. For large problems ISLOCALTOGLOBALMAPPINGHASH is used, this is scalable.
    Use ISLocalToGlobalMappingSetType() or call ISLocalToGlobalMappingSetFromOptions() with the option -islocaltoglobalmapping_type <basic,hash> to control which is used.

    Level: advanced

    Developer Note: The manual page states that idx and idxout may be identical but the calling
       sequence declares idx as const so it cannot be the same as idxout.

.seealso: ISLocalToGlobalMappingApply(), ISGlobalToLocalMappingApplyBlock(), ISLocalToGlobalMappingCreate(),
          ISLocalToGlobalMappingDestroy()
@*/
PetscErrorCode  ISGlobalToLocalMappingApply(ISLocalToGlobalMapping mapping,ISGlobalToLocalMappingMode type,PetscInt n,const PetscInt idx[],PetscInt *nout,PetscInt idxout[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mapping,IS_LTOGM_CLASSID,1);
  if (!mapping->data) {
    ierr = ISGlobalToLocalMappingSetUp(mapping);CHKERRQ(ierr);
  }
  ierr = (*mapping->ops->globaltolocalmappingapply)(mapping,type,n,idx,nout,idxout);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
    ISGlobalToLocalMappingApplyIS - Creates from an IS in the global numbering
    a new index set using the local numbering defined in an ISLocalToGlobalMapping
    context.

    Not collective

    Input Parameters:
+   mapping - mapping between local and global numbering
.   type - IS_GTOLM_MASK - maps global indices with no local value to -1 in the output list (i.e., mask them)
           IS_GTOLM_DROP - drops the indices with no local value from the output list
-   is - index set in global numbering

    Output Parameters:
.   newis - index set in local numbering

    Notes:
    The output IS will be sequential, as it encodes a purely local operation

    Level: advanced

.seealso: ISGlobalToLocalMappingApply(), ISLocalToGlobalMappingCreate(),
          ISLocalToGlobalMappingDestroy()
@*/
PetscErrorCode  ISGlobalToLocalMappingApplyIS(ISLocalToGlobalMapping mapping,ISGlobalToLocalMappingMode type,IS is,IS *newis)
{
  PetscErrorCode ierr;
  PetscInt       n,nout,*idxout;
  const PetscInt *idxin;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mapping,IS_LTOGM_CLASSID,1);
  PetscValidHeaderSpecific(is,IS_CLASSID,3);
  PetscValidPointer(newis,4);

  ierr = ISGetLocalSize(is,&n);CHKERRQ(ierr);
  ierr = ISGetIndices(is,&idxin);CHKERRQ(ierr);
  if (type == IS_GTOLM_MASK) {
    ierr = PetscMalloc1(n,&idxout);CHKERRQ(ierr);
  } else {
    ierr = ISGlobalToLocalMappingApply(mapping,type,n,idxin,&nout,NULL);CHKERRQ(ierr);
    ierr = PetscMalloc1(nout,&idxout);CHKERRQ(ierr);
  }
  ierr = ISGlobalToLocalMappingApply(mapping,type,n,idxin,&nout,idxout);CHKERRQ(ierr);
  ierr = ISRestoreIndices(is,&idxin);CHKERRQ(ierr);
  ierr = ISCreateGeneral(PETSC_COMM_SELF,nout,idxout,PETSC_OWN_POINTER,newis);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
    ISGlobalToLocalMappingApplyBlock - Provides the local block numbering for a list of integers
    specified with a block global numbering.

    Not collective

    Input Parameters:
+   mapping - mapping between local and global numbering
.   type - IS_GTOLM_MASK - maps global indices with no local value to -1 in the output list (i.e., mask them)
           IS_GTOLM_DROP - drops the indices with no local value from the output list
.   n - number of global indices to map
-   idx - global indices to map

    Output Parameters:
+   nout - number of indices in output array (if type == IS_GTOLM_MASK then nout = n)
-   idxout - local index of each global index, one must pass in an array long enough
             to hold all the indices. You can call ISGlobalToLocalMappingApplyBlock() with
             idxout == NULL to determine the required length (returned in nout)
             and then allocate the required space and call ISGlobalToLocalMappingApplyBlock()
             a second time to set the values.

    Notes:
    Either nout or idxout may be NULL. idx and idxout may be identical.

    For "small" problems when using ISGlobalToLocalMappingApply() and ISGlobalToLocalMappingApplyBlock(), the ISLocalToGlobalMappingType of ISLOCALTOGLOBALMAPPINGBASIC will be used;
    this uses more memory but is faster; this approach is not scalable for extremely large mappings. For large problems ISLOCALTOGLOBALMAPPINGHASH is used, this is scalable.
    Use ISLocalToGlobalMappingSetType() or call ISLocalToGlobalMappingSetFromOptions() with the option -islocaltoglobalmapping_type <basic,hash> to control which is used.

    Level: advanced

    Developer Note: The manual page states that idx and idxout may be identical but the calling
       sequence declares idx as const so it cannot be the same as idxout.

.seealso: ISLocalToGlobalMappingApply(), ISGlobalToLocalMappingApply(), ISLocalToGlobalMappingCreate(),
          ISLocalToGlobalMappingDestroy()
@*/
PetscErrorCode  ISGlobalToLocalMappingApplyBlock(ISLocalToGlobalMapping mapping,ISGlobalToLocalMappingMode type,
                                                 PetscInt n,const PetscInt idx[],PetscInt *nout,PetscInt idxout[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mapping,IS_LTOGM_CLASSID,1);
  if (!mapping->data) {
    ierr = ISGlobalToLocalMappingSetUp(mapping);CHKERRQ(ierr);
  }
  ierr = (*mapping->ops->globaltolocalmappingapplyblock)(mapping,type,n,idx,nout,idxout);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
    ISLocalToGlobalMappingGetBlockInfo - Gets the neighbor information for each processor and
     each index shared by more than one processor

    Collective on ISLocalToGlobalMapping

    Input Parameter:
.   mapping - the mapping from local to global indexing

    Output Parameters:
+   nproc - number of processors that are connected to this one
.   proc - neighboring processors
.   numproc - number of indices for each subdomain (processor)
-   indices - indices of nodes (in local numbering) shared with neighbors (sorted by global numbering)

    Level: advanced

    Fortran Usage:
$        ISLocalToGlobalMpngGetInfoSize(ISLocalToGlobalMapping,PetscInt nproc,PetscInt numprocmax,ierr) followed by
$        ISLocalToGlobalMappingGetInfo(ISLocalToGlobalMapping,PetscInt nproc, PetscInt procs[nproc],PetscInt numprocs[nproc],
          PetscInt indices[nproc][numprocmax],ierr)
        There is no ISLocalToGlobalMappingRestoreInfo() in Fortran. You must make sure that procs[], numprocs[] and
        indices[][] are large enough arrays, either by allocating them dynamically or defining static ones large enough.

.seealso: ISLocalToGlobalMappingDestroy(), ISLocalToGlobalMappingCreateIS(), ISLocalToGlobalMappingCreate(),
          ISLocalToGlobalMappingRestoreInfo()
@*/
PetscErrorCode  ISLocalToGlobalMappingGetBlockInfo(ISLocalToGlobalMapping mapping,PetscInt *nproc,PetscInt *procs[],PetscInt *numprocs[],PetscInt **indices[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mapping,IS_LTOGM_CLASSID,1);
  if (mapping->info_cached) {
    *nproc    = mapping->info_nproc;
    *procs    = mapping->info_procs;
    *numprocs = mapping->info_numprocs;
    *indices  = mapping->info_indices;
  } else {
    ierr = ISLocalToGlobalMappingGetBlockInfo_Private(mapping,nproc,procs,numprocs,indices);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode  ISLocalToGlobalMappingGetBlockInfo_Private(ISLocalToGlobalMapping mapping,PetscInt *nproc,PetscInt *procs[],PetscInt *numprocs[],PetscInt **indices[])
{
  PetscErrorCode ierr;
  PetscMPIInt    size,rank,tag1,tag2,tag3,*len,*source,imdex;
  PetscInt       i,n = mapping->n,Ng,ng,max = 0,*lindices = mapping->indices;
  PetscInt       *nprocs,*owner,nsends,*sends,j,*starts,nmax,nrecvs,*recvs,proc;
  PetscInt       cnt,scale,*ownedsenders,*nownedsenders,rstart,nowned;
  PetscInt       node,nownedm,nt,*sends2,nsends2,*starts2,*lens2,*dest,nrecvs2,*starts3,*recvs2,k,*bprocs,*tmp;
  PetscInt       first_procs,first_numprocs,*first_indices;
  MPI_Request    *recv_waits,*send_waits;
  MPI_Status     recv_status,*send_status,*recv_statuses;
  MPI_Comm       comm;
  PetscBool      debug = PETSC_FALSE;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)mapping,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm,&size);CHKERRMPI(ierr);
  ierr = MPI_Comm_rank(comm,&rank);CHKERRMPI(ierr);
  if (size == 1) {
    *nproc         = 0;
    *procs         = NULL;
    ierr           = PetscNew(numprocs);CHKERRQ(ierr);
    (*numprocs)[0] = 0;
    ierr           = PetscNew(indices);CHKERRQ(ierr);
    (*indices)[0]  = NULL;
    /* save info for reuse */
    mapping->info_nproc = *nproc;
    mapping->info_procs = *procs;
    mapping->info_numprocs = *numprocs;
    mapping->info_indices = *indices;
    mapping->info_cached = PETSC_TRUE;
    PetscFunctionReturn(0);
  }

  ierr = PetscOptionsGetBool(((PetscObject)mapping)->options,NULL,"-islocaltoglobalmappinggetinfo_debug",&debug,NULL);CHKERRQ(ierr);

  /*
    Notes on ISLocalToGlobalMappingGetBlockInfo

    globally owned node - the nodes that have been assigned to this processor in global
           numbering, just for this routine.

    nontrivial globally owned node - node assigned to this processor that is on a subdomain
           boundary (i.e. is has more than one local owner)

    locally owned node - node that exists on this processors subdomain

    nontrivial locally owned node - node that is not in the interior (i.e. has more than one
           local subdomain
  */
  ierr = PetscObjectGetNewTag((PetscObject)mapping,&tag1);CHKERRQ(ierr);
  ierr = PetscObjectGetNewTag((PetscObject)mapping,&tag2);CHKERRQ(ierr);
  ierr = PetscObjectGetNewTag((PetscObject)mapping,&tag3);CHKERRQ(ierr);

  for (i=0; i<n; i++) {
    if (lindices[i] > max) max = lindices[i];
  }
  ierr   = MPIU_Allreduce(&max,&Ng,1,MPIU_INT,MPI_MAX,comm);CHKERRMPI(ierr);
  Ng++;
  ierr   = MPI_Comm_size(comm,&size);CHKERRMPI(ierr);
  ierr   = MPI_Comm_rank(comm,&rank);CHKERRMPI(ierr);
  scale  = Ng/size + 1;
  ng     = scale; if (rank == size-1) ng = Ng - scale*(size-1); ng = PetscMax(1,ng);
  rstart = scale*rank;

  /* determine ownership ranges of global indices */
  ierr = PetscMalloc1(2*size,&nprocs);CHKERRQ(ierr);
  ierr = PetscArrayzero(nprocs,2*size);CHKERRQ(ierr);

  /* determine owners of each local node  */
  ierr = PetscMalloc1(n,&owner);CHKERRQ(ierr);
  for (i=0; i<n; i++) {
    proc             = lindices[i]/scale; /* processor that globally owns this index */
    nprocs[2*proc+1] = 1;                 /* processor globally owns at least one of ours */
    owner[i]         = proc;
    nprocs[2*proc]++;                     /* count of how many that processor globally owns of ours */
  }
  nsends = 0; for (i=0; i<size; i++) nsends += nprocs[2*i+1];
  ierr = PetscInfo1(mapping,"Number of global owners for my local data %D\n",nsends);CHKERRQ(ierr);

  /* inform other processors of number of messages and max length*/
  ierr = PetscMaxSum(comm,nprocs,&nmax,&nrecvs);CHKERRQ(ierr);
  ierr = PetscInfo1(mapping,"Number of local owners for my global data %D\n",nrecvs);CHKERRQ(ierr);

  /* post receives for owned rows */
  ierr = PetscMalloc1((2*nrecvs+1)*(nmax+1),&recvs);CHKERRQ(ierr);
  ierr = PetscMalloc1(nrecvs+1,&recv_waits);CHKERRQ(ierr);
  for (i=0; i<nrecvs; i++) {
    ierr = MPI_Irecv(recvs+2*nmax*i,2*nmax,MPIU_INT,MPI_ANY_SOURCE,tag1,comm,recv_waits+i);CHKERRMPI(ierr);
  }

  /* pack messages containing lists of local nodes to owners */
  ierr      = PetscMalloc1(2*n+1,&sends);CHKERRQ(ierr);
  ierr      = PetscMalloc1(size+1,&starts);CHKERRQ(ierr);
  starts[0] = 0;
  for (i=1; i<size; i++) starts[i] = starts[i-1] + 2*nprocs[2*i-2];
  for (i=0; i<n; i++) {
    sends[starts[owner[i]]++] = lindices[i];
    sends[starts[owner[i]]++] = i;
  }
  ierr = PetscFree(owner);CHKERRQ(ierr);
  starts[0] = 0;
  for (i=1; i<size; i++) starts[i] = starts[i-1] + 2*nprocs[2*i-2];

  /* send the messages */
  ierr = PetscMalloc1(nsends+1,&send_waits);CHKERRQ(ierr);
  ierr = PetscMalloc1(nsends+1,&dest);CHKERRQ(ierr);
  cnt = 0;
  for (i=0; i<size; i++) {
    if (nprocs[2*i]) {
      ierr      = MPI_Isend(sends+starts[i],2*nprocs[2*i],MPIU_INT,i,tag1,comm,send_waits+cnt);CHKERRMPI(ierr);
      dest[cnt] = i;
      cnt++;
    }
  }
  ierr = PetscFree(starts);CHKERRQ(ierr);

  /* wait on receives */
  ierr = PetscMalloc1(nrecvs+1,&source);CHKERRQ(ierr);
  ierr = PetscMalloc1(nrecvs+1,&len);CHKERRQ(ierr);
  cnt  = nrecvs;
  ierr = PetscCalloc1(ng+1,&nownedsenders);CHKERRQ(ierr);
  while (cnt) {
    ierr = MPI_Waitany(nrecvs,recv_waits,&imdex,&recv_status);CHKERRMPI(ierr);
    /* unpack receives into our local space */
    ierr          = MPI_Get_count(&recv_status,MPIU_INT,&len[imdex]);CHKERRMPI(ierr);
    source[imdex] = recv_status.MPI_SOURCE;
    len[imdex]    = len[imdex]/2;
    /* count how many local owners for each of my global owned indices */
    for (i=0; i<len[imdex]; i++) nownedsenders[recvs[2*imdex*nmax+2*i]-rstart]++;
    cnt--;
  }
  ierr = PetscFree(recv_waits);CHKERRQ(ierr);

  /* count how many globally owned indices are on an edge multiplied by how many processors own them. */
  nowned  = 0;
  nownedm = 0;
  for (i=0; i<ng; i++) {
    if (nownedsenders[i] > 1) {nownedm += nownedsenders[i]; nowned++;}
  }

  /* create single array to contain rank of all local owners of each globally owned index */
  ierr      = PetscMalloc1(nownedm+1,&ownedsenders);CHKERRQ(ierr);
  ierr      = PetscMalloc1(ng+1,&starts);CHKERRQ(ierr);
  starts[0] = 0;
  for (i=1; i<ng; i++) {
    if (nownedsenders[i-1] > 1) starts[i] = starts[i-1] + nownedsenders[i-1];
    else starts[i] = starts[i-1];
  }

  /* for each nontrival globally owned node list all arriving processors */
  for (i=0; i<nrecvs; i++) {
    for (j=0; j<len[i]; j++) {
      node = recvs[2*i*nmax+2*j]-rstart;
      if (nownedsenders[node] > 1) ownedsenders[starts[node]++] = source[i];
    }
  }

  if (debug) { /* -----------------------------------  */
    starts[0] = 0;
    for (i=1; i<ng; i++) {
      if (nownedsenders[i-1] > 1) starts[i] = starts[i-1] + nownedsenders[i-1];
      else starts[i] = starts[i-1];
    }
    for (i=0; i<ng; i++) {
      if (nownedsenders[i] > 1) {
        ierr = PetscSynchronizedPrintf(comm,"[%d] global node %D local owner processors: ",rank,i+rstart);CHKERRQ(ierr);
        for (j=0; j<nownedsenders[i]; j++) {
          ierr = PetscSynchronizedPrintf(comm,"%D ",ownedsenders[starts[i]+j]);CHKERRQ(ierr);
        }
        ierr = PetscSynchronizedPrintf(comm,"\n");CHKERRQ(ierr);
      }
    }
    ierr = PetscSynchronizedFlush(comm,PETSC_STDOUT);CHKERRQ(ierr);
  } /* -----------------------------------  */

  /* wait on original sends */
  if (nsends) {
    ierr = PetscMalloc1(nsends,&send_status);CHKERRQ(ierr);
    ierr = MPI_Waitall(nsends,send_waits,send_status);CHKERRMPI(ierr);
    ierr = PetscFree(send_status);CHKERRQ(ierr);
  }
  ierr = PetscFree(send_waits);CHKERRQ(ierr);
  ierr = PetscFree(sends);CHKERRQ(ierr);
  ierr = PetscFree(nprocs);CHKERRQ(ierr);

  /* pack messages to send back to local owners */
  starts[0] = 0;
  for (i=1; i<ng; i++) {
    if (nownedsenders[i-1] > 1) starts[i] = starts[i-1] + nownedsenders[i-1];
    else starts[i] = starts[i-1];
  }
  nsends2 = nrecvs;
  ierr    = PetscMalloc1(nsends2+1,&nprocs);CHKERRQ(ierr); /* length of each message */
  for (i=0; i<nrecvs; i++) {
    nprocs[i] = 1;
    for (j=0; j<len[i]; j++) {
      node = recvs[2*i*nmax+2*j]-rstart;
      if (nownedsenders[node] > 1) nprocs[i] += 2 + nownedsenders[node];
    }
  }
  nt = 0;
  for (i=0; i<nsends2; i++) nt += nprocs[i];

  ierr = PetscMalloc1(nt+1,&sends2);CHKERRQ(ierr);
  ierr = PetscMalloc1(nsends2+1,&starts2);CHKERRQ(ierr);

  starts2[0] = 0;
  for (i=1; i<nsends2; i++) starts2[i] = starts2[i-1] + nprocs[i-1];
  /*
     Each message is 1 + nprocs[i] long, and consists of
       (0) the number of nodes being sent back
       (1) the local node number,
       (2) the number of processors sharing it,
       (3) the processors sharing it
  */
  for (i=0; i<nsends2; i++) {
    cnt = 1;
    sends2[starts2[i]] = 0;
    for (j=0; j<len[i]; j++) {
      node = recvs[2*i*nmax+2*j]-rstart;
      if (nownedsenders[node] > 1) {
        sends2[starts2[i]]++;
        sends2[starts2[i]+cnt++] = recvs[2*i*nmax+2*j+1];
        sends2[starts2[i]+cnt++] = nownedsenders[node];
        ierr = PetscArraycpy(&sends2[starts2[i]+cnt],&ownedsenders[starts[node]],nownedsenders[node]);CHKERRQ(ierr);
        cnt += nownedsenders[node];
      }
    }
  }

  /* receive the message lengths */
  nrecvs2 = nsends;
  ierr    = PetscMalloc1(nrecvs2+1,&lens2);CHKERRQ(ierr);
  ierr    = PetscMalloc1(nrecvs2+1,&starts3);CHKERRQ(ierr);
  ierr    = PetscMalloc1(nrecvs2+1,&recv_waits);CHKERRQ(ierr);
  for (i=0; i<nrecvs2; i++) {
    ierr = MPI_Irecv(&lens2[i],1,MPIU_INT,dest[i],tag2,comm,recv_waits+i);CHKERRMPI(ierr);
  }

  /* send the message lengths */
  for (i=0; i<nsends2; i++) {
    ierr = MPI_Send(&nprocs[i],1,MPIU_INT,source[i],tag2,comm);CHKERRMPI(ierr);
  }

  /* wait on receives of lens */
  if (nrecvs2) {
    ierr = PetscMalloc1(nrecvs2,&recv_statuses);CHKERRQ(ierr);
    ierr = MPI_Waitall(nrecvs2,recv_waits,recv_statuses);CHKERRMPI(ierr);
    ierr = PetscFree(recv_statuses);CHKERRQ(ierr);
  }
  ierr = PetscFree(recv_waits);CHKERRQ(ierr);

  starts3[0] = 0;
  nt         = 0;
  for (i=0; i<nrecvs2-1; i++) {
    starts3[i+1] = starts3[i] + lens2[i];
    nt          += lens2[i];
  }
  if (nrecvs2) nt += lens2[nrecvs2-1];

  ierr = PetscMalloc1(nt+1,&recvs2);CHKERRQ(ierr);
  ierr = PetscMalloc1(nrecvs2+1,&recv_waits);CHKERRQ(ierr);
  for (i=0; i<nrecvs2; i++) {
    ierr = MPI_Irecv(recvs2+starts3[i],lens2[i],MPIU_INT,dest[i],tag3,comm,recv_waits+i);CHKERRMPI(ierr);
  }

  /* send the messages */
  ierr = PetscMalloc1(nsends2+1,&send_waits);CHKERRQ(ierr);
  for (i=0; i<nsends2; i++) {
    ierr = MPI_Isend(sends2+starts2[i],nprocs[i],MPIU_INT,source[i],tag3,comm,send_waits+i);CHKERRMPI(ierr);
  }

  /* wait on receives */
  if (nrecvs2) {
    ierr = PetscMalloc1(nrecvs2,&recv_statuses);CHKERRQ(ierr);
    ierr = MPI_Waitall(nrecvs2,recv_waits,recv_statuses);CHKERRMPI(ierr);
    ierr = PetscFree(recv_statuses);CHKERRQ(ierr);
  }
  ierr = PetscFree(recv_waits);CHKERRQ(ierr);
  ierr = PetscFree(nprocs);CHKERRQ(ierr);

  if (debug) { /* -----------------------------------  */
    cnt = 0;
    for (i=0; i<nrecvs2; i++) {
      nt = recvs2[cnt++];
      for (j=0; j<nt; j++) {
        ierr = PetscSynchronizedPrintf(comm,"[%d] local node %D number of subdomains %D: ",rank,recvs2[cnt],recvs2[cnt+1]);CHKERRQ(ierr);
        for (k=0; k<recvs2[cnt+1]; k++) {
          ierr = PetscSynchronizedPrintf(comm,"%D ",recvs2[cnt+2+k]);CHKERRQ(ierr);
        }
        cnt += 2 + recvs2[cnt+1];
        ierr = PetscSynchronizedPrintf(comm,"\n");CHKERRQ(ierr);
      }
    }
    ierr = PetscSynchronizedFlush(comm,PETSC_STDOUT);CHKERRQ(ierr);
  } /* -----------------------------------  */

  /* count number subdomains for each local node */
  ierr = PetscCalloc1(size,&nprocs);CHKERRQ(ierr);
  cnt  = 0;
  for (i=0; i<nrecvs2; i++) {
    nt = recvs2[cnt++];
    for (j=0; j<nt; j++) {
      for (k=0; k<recvs2[cnt+1]; k++) nprocs[recvs2[cnt+2+k]]++;
      cnt += 2 + recvs2[cnt+1];
    }
  }
  nt = 0; for (i=0; i<size; i++) nt += (nprocs[i] > 0);
  *nproc    = nt;
  ierr = PetscMalloc1(nt+1,procs);CHKERRQ(ierr);
  ierr = PetscMalloc1(nt+1,numprocs);CHKERRQ(ierr);
  ierr = PetscMalloc1(nt+1,indices);CHKERRQ(ierr);
  for (i=0;i<nt+1;i++) (*indices)[i]=NULL;
  ierr = PetscMalloc1(size,&bprocs);CHKERRQ(ierr);
  cnt  = 0;
  for (i=0; i<size; i++) {
    if (nprocs[i] > 0) {
      bprocs[i]        = cnt;
      (*procs)[cnt]    = i;
      (*numprocs)[cnt] = nprocs[i];
      ierr             = PetscMalloc1(nprocs[i],&(*indices)[cnt]);CHKERRQ(ierr);
      cnt++;
    }
  }

  /* make the list of subdomains for each nontrivial local node */
  ierr = PetscArrayzero(*numprocs,nt);CHKERRQ(ierr);
  cnt  = 0;
  for (i=0; i<nrecvs2; i++) {
    nt = recvs2[cnt++];
    for (j=0; j<nt; j++) {
      for (k=0; k<recvs2[cnt+1]; k++) (*indices)[bprocs[recvs2[cnt+2+k]]][(*numprocs)[bprocs[recvs2[cnt+2+k]]]++] = recvs2[cnt];
      cnt += 2 + recvs2[cnt+1];
    }
  }
  ierr = PetscFree(bprocs);CHKERRQ(ierr);
  ierr = PetscFree(recvs2);CHKERRQ(ierr);

  /* sort the node indexing by their global numbers */
  nt = *nproc;
  for (i=0; i<nt; i++) {
    ierr = PetscMalloc1((*numprocs)[i],&tmp);CHKERRQ(ierr);
    for (j=0; j<(*numprocs)[i]; j++) tmp[j] = lindices[(*indices)[i][j]];
    ierr = PetscSortIntWithArray((*numprocs)[i],tmp,(*indices)[i]);CHKERRQ(ierr);
    ierr = PetscFree(tmp);CHKERRQ(ierr);
  }

  if (debug) { /* -----------------------------------  */
    nt = *nproc;
    for (i=0; i<nt; i++) {
      ierr = PetscSynchronizedPrintf(comm,"[%d] subdomain %D number of indices %D: ",rank,(*procs)[i],(*numprocs)[i]);CHKERRQ(ierr);
      for (j=0; j<(*numprocs)[i]; j++) {
        ierr = PetscSynchronizedPrintf(comm,"%D ",(*indices)[i][j]);CHKERRQ(ierr);
      }
      ierr = PetscSynchronizedPrintf(comm,"\n");CHKERRQ(ierr);
    }
    ierr = PetscSynchronizedFlush(comm,PETSC_STDOUT);CHKERRQ(ierr);
  } /* -----------------------------------  */

  /* wait on sends */
  if (nsends2) {
    ierr = PetscMalloc1(nsends2,&send_status);CHKERRQ(ierr);
    ierr = MPI_Waitall(nsends2,send_waits,send_status);CHKERRMPI(ierr);
    ierr = PetscFree(send_status);CHKERRQ(ierr);
  }

  ierr = PetscFree(starts3);CHKERRQ(ierr);
  ierr = PetscFree(dest);CHKERRQ(ierr);
  ierr = PetscFree(send_waits);CHKERRQ(ierr);

  ierr = PetscFree(nownedsenders);CHKERRQ(ierr);
  ierr = PetscFree(ownedsenders);CHKERRQ(ierr);
  ierr = PetscFree(starts);CHKERRQ(ierr);
  ierr = PetscFree(starts2);CHKERRQ(ierr);
  ierr = PetscFree(lens2);CHKERRQ(ierr);

  ierr = PetscFree(source);CHKERRQ(ierr);
  ierr = PetscFree(len);CHKERRQ(ierr);
  ierr = PetscFree(recvs);CHKERRQ(ierr);
  ierr = PetscFree(nprocs);CHKERRQ(ierr);
  ierr = PetscFree(sends2);CHKERRQ(ierr);

  /* put the information about myself as the first entry in the list */
  first_procs    = (*procs)[0];
  first_numprocs = (*numprocs)[0];
  first_indices  = (*indices)[0];
  for (i=0; i<*nproc; i++) {
    if ((*procs)[i] == rank) {
      (*procs)[0]    = (*procs)[i];
      (*numprocs)[0] = (*numprocs)[i];
      (*indices)[0]  = (*indices)[i];
      (*procs)[i]    = first_procs;
      (*numprocs)[i] = first_numprocs;
      (*indices)[i]  = first_indices;
      break;
    }
  }

  /* save info for reuse */
  mapping->info_nproc = *nproc;
  mapping->info_procs = *procs;
  mapping->info_numprocs = *numprocs;
  mapping->info_indices = *indices;
  mapping->info_cached = PETSC_TRUE;
  PetscFunctionReturn(0);
}

/*@C
    ISLocalToGlobalMappingRestoreBlockInfo - Frees the memory allocated by ISLocalToGlobalMappingGetBlockInfo()

    Collective on ISLocalToGlobalMapping

    Input Parameter:
.   mapping - the mapping from local to global indexing

    Output Parameters:
+   nproc - number of processors that are connected to this one
.   proc - neighboring processors
.   numproc - number of indices for each processor
-   indices - indices of local nodes shared with neighbor (sorted by global numbering)

    Level: advanced

.seealso: ISLocalToGlobalMappingDestroy(), ISLocalToGlobalMappingCreateIS(), ISLocalToGlobalMappingCreate(),
          ISLocalToGlobalMappingGetInfo()
@*/
PetscErrorCode  ISLocalToGlobalMappingRestoreBlockInfo(ISLocalToGlobalMapping mapping,PetscInt *nproc,PetscInt *procs[],PetscInt *numprocs[],PetscInt **indices[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mapping,IS_LTOGM_CLASSID,1);
  if (mapping->info_free) {
    ierr = PetscFree(*numprocs);CHKERRQ(ierr);
    if (*indices) {
      PetscInt i;

      ierr = PetscFree((*indices)[0]);CHKERRQ(ierr);
      for (i=1; i<*nproc; i++) {
        ierr = PetscFree((*indices)[i]);CHKERRQ(ierr);
      }
      ierr = PetscFree(*indices);CHKERRQ(ierr);
    }
  }
  *nproc    = 0;
  *procs    = NULL;
  *numprocs = NULL;
  *indices  = NULL;
  PetscFunctionReturn(0);
}

/*@C
    ISLocalToGlobalMappingGetInfo - Gets the neighbor information for each processor and
     each index shared by more than one processor

    Collective on ISLocalToGlobalMapping

    Input Parameter:
.   mapping - the mapping from local to global indexing

    Output Parameters:
+   nproc - number of processors that are connected to this one
.   proc - neighboring processors
.   numproc - number of indices for each subdomain (processor)
-   indices - indices of nodes (in local numbering) shared with neighbors (sorted by global numbering)

    Level: advanced

    Notes: The user needs to call ISLocalToGlobalMappingRestoreInfo when the data is no longer needed.

    Fortran Usage:
$        ISLocalToGlobalMpngGetInfoSize(ISLocalToGlobalMapping,PetscInt nproc,PetscInt numprocmax,ierr) followed by
$        ISLocalToGlobalMappingGetInfo(ISLocalToGlobalMapping,PetscInt nproc, PetscInt procs[nproc],PetscInt numprocs[nproc],
          PetscInt indices[nproc][numprocmax],ierr)
        There is no ISLocalToGlobalMappingRestoreInfo() in Fortran. You must make sure that procs[], numprocs[] and
        indices[][] are large enough arrays, either by allocating them dynamically or defining static ones large enough.

.seealso: ISLocalToGlobalMappingDestroy(), ISLocalToGlobalMappingCreateIS(), ISLocalToGlobalMappingCreate(),
          ISLocalToGlobalMappingRestoreInfo()
@*/
PetscErrorCode  ISLocalToGlobalMappingGetInfo(ISLocalToGlobalMapping mapping,PetscInt *nproc,PetscInt *procs[],PetscInt *numprocs[],PetscInt **indices[])
{
  PetscErrorCode ierr;
  PetscInt       **bindices = NULL,*bnumprocs = NULL,bs = mapping->bs,i,j,k;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mapping,IS_LTOGM_CLASSID,1);
  ierr = ISLocalToGlobalMappingGetBlockInfo(mapping,nproc,procs,&bnumprocs,&bindices);CHKERRQ(ierr);
  if (bs > 1) { /* we need to expand the cached info */
    ierr = PetscCalloc1(*nproc,&*indices);CHKERRQ(ierr);
    ierr = PetscCalloc1(*nproc,&*numprocs);CHKERRQ(ierr);
    for (i=0; i<*nproc; i++) {
      ierr = PetscMalloc1(bs*bnumprocs[i],&(*indices)[i]);CHKERRQ(ierr);
      for (j=0; j<bnumprocs[i]; j++) {
        for (k=0; k<bs; k++) {
          (*indices)[i][j*bs+k] = bs*bindices[i][j] + k;
        }
      }
      (*numprocs)[i] = bnumprocs[i]*bs;
    }
    mapping->info_free = PETSC_TRUE;
  } else {
    *numprocs = bnumprocs;
    *indices  = bindices;
  }
  PetscFunctionReturn(0);
}

/*@C
    ISLocalToGlobalMappingRestoreInfo - Frees the memory allocated by ISLocalToGlobalMappingGetInfo()

    Collective on ISLocalToGlobalMapping

    Input Parameter:
.   mapping - the mapping from local to global indexing

    Output Parameters:
+   nproc - number of processors that are connected to this one
.   proc - neighboring processors
.   numproc - number of indices for each processor
-   indices - indices of local nodes shared with neighbor (sorted by global numbering)

    Level: advanced

.seealso: ISLocalToGlobalMappingDestroy(), ISLocalToGlobalMappingCreateIS(), ISLocalToGlobalMappingCreate(),
          ISLocalToGlobalMappingGetInfo()
@*/
PetscErrorCode  ISLocalToGlobalMappingRestoreInfo(ISLocalToGlobalMapping mapping,PetscInt *nproc,PetscInt *procs[],PetscInt *numprocs[],PetscInt **indices[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = ISLocalToGlobalMappingRestoreBlockInfo(mapping,nproc,procs,numprocs,indices);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
    ISLocalToGlobalMappingGetNodeInfo - Gets the neighbor information for each node

    Collective on ISLocalToGlobalMapping

    Input Parameter:
.   mapping - the mapping from local to global indexing

    Output Parameters:
+   nnodes - number of local nodes (same ISLocalToGlobalMappingGetSize())
.   count - number of neighboring processors per node
-   indices - indices of processes sharing the node (sorted)

    Level: advanced

    Notes: The user needs to call ISLocalToGlobalMappingRestoreInfo when the data is no longer needed.

.seealso: ISLocalToGlobalMappingDestroy(), ISLocalToGlobalMappingCreateIS(), ISLocalToGlobalMappingCreate(),
          ISLocalToGlobalMappingGetInfo(), ISLocalToGlobalMappingRestoreNodeInfo()
@*/
PetscErrorCode  ISLocalToGlobalMappingGetNodeInfo(ISLocalToGlobalMapping mapping,PetscInt *nnodes,PetscInt *count[],PetscInt **indices[])
{
  PetscInt       n;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mapping,IS_LTOGM_CLASSID,1);
  ierr = ISLocalToGlobalMappingGetSize(mapping,&n);CHKERRQ(ierr);
  if (!mapping->info_nodec) {
    PetscInt i,m,n_neigh,*neigh,*n_shared,**shared;

    ierr = PetscMalloc2(n+1,&mapping->info_nodec,n,&mapping->info_nodei);CHKERRQ(ierr);
    ierr = ISLocalToGlobalMappingGetInfo(mapping,&n_neigh,&neigh,&n_shared,&shared);CHKERRQ(ierr);
    for (i=0;i<n;i++) { mapping->info_nodec[i] = 1;}
    m = n;
    mapping->info_nodec[n] = 0;
    for (i=1;i<n_neigh;i++) {
      PetscInt j;

      m += n_shared[i];
      for (j=0;j<n_shared[i];j++) mapping->info_nodec[shared[i][j]] += 1;
    }
    if (n) { ierr = PetscMalloc1(m,&mapping->info_nodei[0]);CHKERRQ(ierr); }
    for (i=1;i<n;i++) mapping->info_nodei[i] = mapping->info_nodei[i-1] + mapping->info_nodec[i-1];
    ierr = PetscArrayzero(mapping->info_nodec,n);CHKERRQ(ierr);
    for (i=0;i<n;i++) { mapping->info_nodec[i] = 1; mapping->info_nodei[i][0] = neigh[0]; }
    for (i=1;i<n_neigh;i++) {
      PetscInt j;

      for (j=0;j<n_shared[i];j++) {
        PetscInt k = shared[i][j];

        mapping->info_nodei[k][mapping->info_nodec[k]] = neigh[i];
        mapping->info_nodec[k] += 1;
      }
    }
    for (i=0;i<n;i++) { ierr = PetscSortRemoveDupsInt(&mapping->info_nodec[i],mapping->info_nodei[i]);CHKERRQ(ierr); }
    ierr = ISLocalToGlobalMappingRestoreInfo(mapping,&n_neigh,&neigh,&n_shared,&shared);CHKERRQ(ierr);
  }
  if (nnodes)  *nnodes  = n;
  if (count)   *count   = mapping->info_nodec;
  if (indices) *indices = mapping->info_nodei;
  PetscFunctionReturn(0);
}

/*@C
    ISLocalToGlobalMappingRestoreNodeInfo - Frees the memory allocated by ISLocalToGlobalMappingGetNodeInfo()

    Collective on ISLocalToGlobalMapping

    Input Parameter:
.   mapping - the mapping from local to global indexing

    Output Parameters:
+   nnodes - number of local nodes
.   count - number of neighboring processors per node
-   indices - indices of processes sharing the node (sorted)

    Level: advanced

.seealso: ISLocalToGlobalMappingDestroy(), ISLocalToGlobalMappingCreateIS(), ISLocalToGlobalMappingCreate(),
          ISLocalToGlobalMappingGetInfo()
@*/
PetscErrorCode  ISLocalToGlobalMappingRestoreNodeInfo(ISLocalToGlobalMapping mapping,PetscInt *nnodes,PetscInt *count[],PetscInt **indices[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mapping,IS_LTOGM_CLASSID,1);
  if (nnodes)  *nnodes  = 0;
  if (count)   *count   = NULL;
  if (indices) *indices = NULL;
  PetscFunctionReturn(0);
}

/*@C
   ISLocalToGlobalMappingGetIndices - Get global indices for every local point that is mapped

   Not Collective

   Input Parameter:
. ltog - local to global mapping

   Output Parameter:
. array - array of indices, the length of this array may be obtained with ISLocalToGlobalMappingGetSize()

   Level: advanced

   Notes:
    ISLocalToGlobalMappingGetSize() returns the length the this array

.seealso: ISLocalToGlobalMappingCreate(), ISLocalToGlobalMappingApply(), ISLocalToGlobalMappingRestoreIndices(), ISLocalToGlobalMappingGetBlockIndices(), ISLocalToGlobalMappingRestoreBlockIndices()
@*/
PetscErrorCode  ISLocalToGlobalMappingGetIndices(ISLocalToGlobalMapping ltog,const PetscInt **array)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ltog,IS_LTOGM_CLASSID,1);
  PetscValidPointer(array,2);
  if (ltog->bs == 1) {
    *array = ltog->indices;
  } else {
    PetscInt       *jj,k,i,j,n = ltog->n, bs = ltog->bs;
    const PetscInt *ii;
    PetscErrorCode ierr;

    ierr = PetscMalloc1(bs*n,&jj);CHKERRQ(ierr);
    *array = jj;
    k    = 0;
    ii   = ltog->indices;
    for (i=0; i<n; i++)
      for (j=0; j<bs; j++)
        jj[k++] = bs*ii[i] + j;
  }
  PetscFunctionReturn(0);
}

/*@C
   ISLocalToGlobalMappingRestoreIndices - Restore indices obtained with ISLocalToGlobalMappingGetIndices()

   Not Collective

   Input Parameters:
+ ltog - local to global mapping
- array - array of indices

   Level: advanced

.seealso: ISLocalToGlobalMappingCreate(), ISLocalToGlobalMappingApply(), ISLocalToGlobalMappingGetIndices()
@*/
PetscErrorCode  ISLocalToGlobalMappingRestoreIndices(ISLocalToGlobalMapping ltog,const PetscInt **array)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ltog,IS_LTOGM_CLASSID,1);
  PetscValidPointer(array,2);
  if (ltog->bs == 1 && *array != ltog->indices) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_BADPTR,"Trying to return mismatched pointer");

  if (ltog->bs > 1) {
    PetscErrorCode ierr;
    ierr = PetscFree(*(void**)array);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*@C
   ISLocalToGlobalMappingGetBlockIndices - Get global indices for every local block

   Not Collective

   Input Parameter:
. ltog - local to global mapping

   Output Parameter:
. array - array of indices

   Level: advanced

.seealso: ISLocalToGlobalMappingCreate(), ISLocalToGlobalMappingApply(), ISLocalToGlobalMappingRestoreBlockIndices()
@*/
PetscErrorCode  ISLocalToGlobalMappingGetBlockIndices(ISLocalToGlobalMapping ltog,const PetscInt **array)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ltog,IS_LTOGM_CLASSID,1);
  PetscValidPointer(array,2);
  *array = ltog->indices;
  PetscFunctionReturn(0);
}

/*@C
   ISLocalToGlobalMappingRestoreBlockIndices - Restore indices obtained with ISLocalToGlobalMappingGetBlockIndices()

   Not Collective

   Input Parameters:
+ ltog - local to global mapping
- array - array of indices

   Level: advanced

.seealso: ISLocalToGlobalMappingCreate(), ISLocalToGlobalMappingApply(), ISLocalToGlobalMappingGetIndices()
@*/
PetscErrorCode  ISLocalToGlobalMappingRestoreBlockIndices(ISLocalToGlobalMapping ltog,const PetscInt **array)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ltog,IS_LTOGM_CLASSID,1);
  PetscValidPointer(array,2);
  if (*array != ltog->indices) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_BADPTR,"Trying to return mismatched pointer");
  *array = NULL;
  PetscFunctionReturn(0);
}

/*@C
   ISLocalToGlobalMappingConcatenate - Create a new mapping that concatenates a list of mappings

   Not Collective

   Input Parameters:
+ comm - communicator for the new mapping, must contain the communicator of every mapping to concatenate
. n - number of mappings to concatenate
- ltogs - local to global mappings

   Output Parameter:
. ltogcat - new mapping

   Note: this currently always returns a mapping with block size of 1

   Developer Note: If all the input mapping have the same block size we could easily handle that as a special case

   Level: advanced

.seealso: ISLocalToGlobalMappingCreate()
@*/
PetscErrorCode ISLocalToGlobalMappingConcatenate(MPI_Comm comm,PetscInt n,const ISLocalToGlobalMapping ltogs[],ISLocalToGlobalMapping *ltogcat)
{
  PetscInt       i,cnt,m,*idx;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (n < 0) SETERRQ1(comm,PETSC_ERR_ARG_OUTOFRANGE,"Must have a non-negative number of mappings, given %D",n);
  if (n > 0) PetscValidPointer(ltogs,3);
  for (i=0; i<n; i++) PetscValidHeaderSpecific(ltogs[i],IS_LTOGM_CLASSID,3);
  PetscValidPointer(ltogcat,4);
  for (cnt=0,i=0; i<n; i++) {
    ierr = ISLocalToGlobalMappingGetSize(ltogs[i],&m);CHKERRQ(ierr);
    cnt += m;
  }
  ierr = PetscMalloc1(cnt,&idx);CHKERRQ(ierr);
  for (cnt=0,i=0; i<n; i++) {
    const PetscInt *subidx;
    ierr = ISLocalToGlobalMappingGetSize(ltogs[i],&m);CHKERRQ(ierr);
    ierr = ISLocalToGlobalMappingGetIndices(ltogs[i],&subidx);CHKERRQ(ierr);
    ierr = PetscArraycpy(&idx[cnt],subidx,m);CHKERRQ(ierr);
    ierr = ISLocalToGlobalMappingRestoreIndices(ltogs[i],&subidx);CHKERRQ(ierr);
    cnt += m;
  }
  ierr = ISLocalToGlobalMappingCreate(comm,1,cnt,idx,PETSC_OWN_POINTER,ltogcat);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*MC
      ISLOCALTOGLOBALMAPPINGBASIC - basic implementation of the ISLocalToGlobalMapping object. When ISGlobalToLocalMappingApply() is
                                    used this is good for only small and moderate size problems.

   Options Database Keys:
.   -islocaltoglobalmapping_type basic - select this method

   Level: beginner

.seealso:  ISLocalToGlobalMappingCreate(), ISLocalToGlobalMappingSetType(), ISLOCALTOGLOBALMAPPINGHASH
M*/
PETSC_EXTERN PetscErrorCode ISLocalToGlobalMappingCreate_Basic(ISLocalToGlobalMapping ltog)
{
  PetscFunctionBegin;
  ltog->ops->globaltolocalmappingapply      = ISGlobalToLocalMappingApply_Basic;
  ltog->ops->globaltolocalmappingsetup      = ISGlobalToLocalMappingSetUp_Basic;
  ltog->ops->globaltolocalmappingapplyblock = ISGlobalToLocalMappingApplyBlock_Basic;
  ltog->ops->destroy                        = ISLocalToGlobalMappingDestroy_Basic;
  PetscFunctionReturn(0);
}

/*MC
      ISLOCALTOGLOBALMAPPINGHASH - hash implementation of the ISLocalToGlobalMapping object. When ISGlobalToLocalMappingApply() is
                                    used this is good for large memory problems.

   Options Database Keys:
.   -islocaltoglobalmapping_type hash - select this method

   Notes:
    This is selected automatically for large problems if the user does not set the type.

   Level: beginner

.seealso:  ISLocalToGlobalMappingCreate(), ISLocalToGlobalMappingSetType(), ISLOCALTOGLOBALMAPPINGHASH
M*/
PETSC_EXTERN PetscErrorCode ISLocalToGlobalMappingCreate_Hash(ISLocalToGlobalMapping ltog)
{
  PetscFunctionBegin;
  ltog->ops->globaltolocalmappingapply      = ISGlobalToLocalMappingApply_Hash;
  ltog->ops->globaltolocalmappingsetup      = ISGlobalToLocalMappingSetUp_Hash;
  ltog->ops->globaltolocalmappingapplyblock = ISGlobalToLocalMappingApplyBlock_Hash;
  ltog->ops->destroy                        = ISLocalToGlobalMappingDestroy_Hash;
  PetscFunctionReturn(0);
}

/*@C
    ISLocalToGlobalMappingRegister -  Adds a method for applying a global to local mapping with an ISLocalToGlobalMapping

   Not Collective

   Input Parameters:
+  sname - name of a new method
-  routine_create - routine to create method context

   Notes:
   ISLocalToGlobalMappingRegister() may be called multiple times to add several user-defined mappings.

   Sample usage:
.vb
   ISLocalToGlobalMappingRegister("my_mapper",MyCreate);
.ve

   Then, your mapping can be chosen with the procedural interface via
$     ISLocalToGlobalMappingSetType(ltog,"my_mapper")
   or at runtime via the option
$     -islocaltoglobalmapping_type my_mapper

   Level: advanced

.seealso: ISLocalToGlobalMappingRegisterAll(), ISLocalToGlobalMappingRegisterDestroy(), ISLOCALTOGLOBALMAPPINGBASIC, ISLOCALTOGLOBALMAPPINGHASH

@*/
PetscErrorCode  ISLocalToGlobalMappingRegister(const char sname[],PetscErrorCode (*function)(ISLocalToGlobalMapping))
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = ISInitializePackage();CHKERRQ(ierr);
  ierr = PetscFunctionListAdd(&ISLocalToGlobalMappingList,sname,function);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   ISLocalToGlobalMappingSetType - Builds ISLocalToGlobalMapping for a particular global to local mapping approach.

   Logically Collective on ISLocalToGlobalMapping

   Input Parameters:
+  ltog - the ISLocalToGlobalMapping object
-  type - a known method

   Options Database Key:
.  -islocaltoglobalmapping_type  <method> - Sets the method; use -help for a list
    of available methods (for instance, basic or hash)

   Notes:
   See "petsc/include/petscis.h" for available methods

  Normally, it is best to use the ISLocalToGlobalMappingSetFromOptions() command and
  then set the ISLocalToGlobalMapping type from the options database rather than by using
  this routine.

  Level: intermediate

  Developer Note: ISLocalToGlobalMappingRegister() is used to add new types to ISLocalToGlobalMappingList from which they
  are accessed by ISLocalToGlobalMappingSetType().

.seealso: PCSetType(), ISLocalToGlobalMappingType, ISLocalToGlobalMappingRegister(), ISLocalToGlobalMappingCreate()

@*/
PetscErrorCode  ISLocalToGlobalMappingSetType(ISLocalToGlobalMapping ltog, ISLocalToGlobalMappingType type)
{
  PetscErrorCode ierr,(*r)(ISLocalToGlobalMapping);
  PetscBool      match;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ltog,IS_LTOGM_CLASSID,1);
  PetscValidCharPointer(type,2);

  ierr = PetscObjectTypeCompare((PetscObject)ltog,type,&match);CHKERRQ(ierr);
  if (match) PetscFunctionReturn(0);

  ierr =  PetscFunctionListFind(ISLocalToGlobalMappingList,type,&r);CHKERRQ(ierr);
  if (!r) SETERRQ1(PetscObjectComm((PetscObject)ltog),PETSC_ERR_ARG_UNKNOWN_TYPE,"Unable to find requested ISLocalToGlobalMapping type %s",type);
  /* Destroy the previous private LTOG context */
  if (ltog->ops->destroy) {
    ierr              = (*ltog->ops->destroy)(ltog);CHKERRQ(ierr);
    ltog->ops->destroy = NULL;
  }
  ierr = PetscObjectChangeTypeName((PetscObject)ltog,type);CHKERRQ(ierr);
  ierr = (*r)(ltog);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscBool ISLocalToGlobalMappingRegisterAllCalled = PETSC_FALSE;

/*@C
  ISLocalToGlobalMappingRegisterAll - Registers all of the local to global mapping components in the IS package.

  Not Collective

  Level: advanced

.seealso:  ISRegister(),  ISLocalToGlobalRegister()
@*/
PetscErrorCode  ISLocalToGlobalMappingRegisterAll(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (ISLocalToGlobalMappingRegisterAllCalled) PetscFunctionReturn(0);
  ISLocalToGlobalMappingRegisterAllCalled = PETSC_TRUE;

  ierr = ISLocalToGlobalMappingRegister(ISLOCALTOGLOBALMAPPINGBASIC, ISLocalToGlobalMappingCreate_Basic);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingRegister(ISLOCALTOGLOBALMAPPINGHASH, ISLocalToGlobalMappingCreate_Hash);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
