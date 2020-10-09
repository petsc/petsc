#include <petsc/private/sfimpl.h> /*I "petscsf.h" I*/
#include <petsc/private/hashseti.h>
#include <petscctable.h>

#if defined(PETSC_HAVE_CUDA)
  #include <cuda_runtime.h>
#endif

#if defined(PETSC_HAVE_HIP)
  #include <hip/hip_runtime.h>
#endif

#if defined(PETSC_USE_DEBUG)
#  define PetscSFCheckGraphSet(sf,arg) do {                          \
    if (PetscUnlikely(!(sf)->graphset))                              \
      SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Must call PetscSFSetGraph() or PetscSFSetGraphWithPattern() on argument %D \"%s\" before %s()",(arg),#sf,PETSC_FUNCTION_NAME); \
  } while (0)
#else
#  define PetscSFCheckGraphSet(sf,arg) do {} while (0)
#endif

const char *const PetscSFDuplicateOptions[] = {"CONFONLY","RANKS","GRAPH","PetscSFDuplicateOption","PETSCSF_DUPLICATE_",NULL};

PETSC_STATIC_INLINE PetscErrorCode PetscGetMemType(const void *data,PetscMemType *type)
{
  PetscFunctionBegin;
  PetscValidPointer(type,2);
  *type = PETSC_MEMTYPE_HOST;
#if defined(PETSC_HAVE_CUDA)
  if (PetscCUDAInitialized && data) {
    cudaError_t                  cerr;
    struct cudaPointerAttributes attr;
    enum cudaMemoryType          mtype;
    cerr = cudaPointerGetAttributes(&attr,data); /* Do not check error since before CUDA 11.0, passing a host pointer returns cudaErrorInvalidValue */
    cudaGetLastError(); /* Reset the last error */
    #if (CUDART_VERSION < 10000)
      mtype = attr.memoryType;
    #else
      mtype = attr.type;
    #endif
    if (cerr == cudaSuccess && mtype == cudaMemoryTypeDevice) *type = PETSC_MEMTYPE_DEVICE;
  }
#endif

#if defined(PETSC_HAVE_HIP)
  if (PetscHIPInitialized && data) {
    hipError_t                   cerr;
    struct hipPointerAttribute_t attr;
    enum hipMemoryType           mtype;
    cerr = hipPointerGetAttributes(&attr,data); /* Do not check error since before CUDA 11.0, passing a host pointer returns cudaErrorInvalidValue */
    hipGetLastError(); /* Reset the last error */
    mtype = attr.memoryType;
    if (cerr == hipSuccess && mtype == hipMemoryTypeDevice) *type = PETSC_MEMTYPE_DEVICE;
  }
#endif
  PetscFunctionReturn(0);
}

/*@
   PetscSFCreate - create a star forest communication context

   Collective

   Input Arguments:
.  comm - communicator on which the star forest will operate

   Output Arguments:
.  sf - new star forest context

   Options Database Keys:
+  -sf_type basic     -Use MPI persistent Isend/Irecv for communication (Default)
.  -sf_type window    -Use MPI-3 one-sided window for communication
-  -sf_type neighbor  -Use MPI-3 neighborhood collectives for communication

   Level: intermediate

   Notes:
   When one knows the communication graph is one of the predefined graph, such as MPI_Alltoall, MPI_Allgatherv,
   MPI_Gatherv, one can create a PetscSF and then set its graph with PetscSFSetGraphWithPattern(). These special
   SFs are optimized and they have better performance than general SFs.

.seealso: PetscSFSetGraph(), PetscSFSetGraphWithPattern(), PetscSFDestroy()
@*/
PetscErrorCode PetscSFCreate(MPI_Comm comm,PetscSF *sf)
{
  PetscErrorCode ierr;
  PetscSF        b;

  PetscFunctionBegin;
  PetscValidPointer(sf,2);
  ierr = PetscSFInitializePackage();CHKERRQ(ierr);

  ierr = PetscHeaderCreate(b,PETSCSF_CLASSID,"PetscSF","Star Forest","PetscSF",comm,PetscSFDestroy,PetscSFView);CHKERRQ(ierr);

  b->nroots    = -1;
  b->nleaves   = -1;
  b->minleaf   = PETSC_MAX_INT;
  b->maxleaf   = PETSC_MIN_INT;
  b->nranks    = -1;
  b->rankorder = PETSC_TRUE;
  b->ingroup   = MPI_GROUP_NULL;
  b->outgroup  = MPI_GROUP_NULL;
  b->graphset  = PETSC_FALSE;
#if defined(PETSC_HAVE_DEVICE)
  b->use_gpu_aware_mpi    = use_gpu_aware_mpi;
  b->use_stream_aware_mpi = PETSC_FALSE;
  b->use_default_stream   = PETSC_TRUE; /* The assumption is true for PETSc internal use of SF */
  #if defined(PETSC_HAVE_KOKKOS) /* Prefer kokkos over cuda*/
    b->backend = PETSCSF_BACKEND_KOKKOS;
  #elif defined(PETSC_HAVE_CUDA)
    b->backend = PETSCSF_BACKEND_CUDA;
  #endif
#endif
  *sf = b;
  PetscFunctionReturn(0);
}

/*@
   PetscSFReset - Reset a star forest so that different sizes or neighbors can be used

   Collective

   Input Arguments:
.  sf - star forest

   Level: advanced

.seealso: PetscSFCreate(), PetscSFSetGraph(), PetscSFDestroy()
@*/
PetscErrorCode PetscSFReset(PetscSF sf)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sf,PETSCSF_CLASSID,1);
  if (sf->ops->Reset) {ierr = (*sf->ops->Reset)(sf);CHKERRQ(ierr);}
  sf->nroots   = -1;
  sf->nleaves  = -1;
  sf->minleaf  = PETSC_MAX_INT;
  sf->maxleaf  = PETSC_MIN_INT;
  sf->mine     = NULL;
  sf->remote   = NULL;
  sf->graphset = PETSC_FALSE;
  ierr = PetscFree(sf->mine_alloc);CHKERRQ(ierr);
  ierr = PetscFree(sf->remote_alloc);CHKERRQ(ierr);
  sf->nranks = -1;
  ierr = PetscFree4(sf->ranks,sf->roffset,sf->rmine,sf->rremote);CHKERRQ(ierr);
#if defined(PETSC_HAVE_DEVICE)
  for (PetscInt i=0; i<2; i++) {ierr = PetscSFFree(sf,PETSC_MEMTYPE_DEVICE,sf->rmine_d[i]);CHKERRQ(ierr);}
#endif
  sf->degreeknown = PETSC_FALSE;
  ierr = PetscFree(sf->degree);CHKERRQ(ierr);
  if (sf->ingroup  != MPI_GROUP_NULL) {ierr = MPI_Group_free(&sf->ingroup);CHKERRQ(ierr);}
  if (sf->outgroup != MPI_GROUP_NULL) {ierr = MPI_Group_free(&sf->outgroup);CHKERRQ(ierr);}
  ierr = PetscSFDestroy(&sf->multi);CHKERRQ(ierr);
  ierr = PetscLayoutDestroy(&sf->map);CHKERRQ(ierr);
  sf->setupcalled = PETSC_FALSE;
  PetscFunctionReturn(0);
}

/*@C
   PetscSFSetType - Set the PetscSF communication implementation

   Collective on PetscSF

   Input Parameters:
+  sf - the PetscSF context
-  type - a known method

   Options Database Key:
.  -sf_type <type> - Sets the method; use -help for a list
   of available methods (for instance, window, basic, neighbor)

   Notes:
   See "include/petscsf.h" for available methods (for instance)
+    PETSCSFWINDOW - MPI-2/3 one-sided
-    PETSCSFBASIC - basic implementation using MPI-1 two-sided

  Level: intermediate

.seealso: PetscSFType, PetscSFCreate()
@*/
PetscErrorCode PetscSFSetType(PetscSF sf,PetscSFType type)
{
  PetscErrorCode ierr,(*r)(PetscSF);
  PetscBool      match;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sf,PETSCSF_CLASSID,1);
  PetscValidCharPointer(type,2);

  ierr = PetscObjectTypeCompare((PetscObject)sf,type,&match);CHKERRQ(ierr);
  if (match) PetscFunctionReturn(0);

  ierr = PetscFunctionListFind(PetscSFList,type,&r);CHKERRQ(ierr);
  if (!r) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_UNKNOWN_TYPE,"Unable to find requested PetscSF type %s",type);
  /* Destroy the previous PetscSF implementation context */
  if (sf->ops->Destroy) {ierr = (*(sf)->ops->Destroy)(sf);CHKERRQ(ierr);}
  ierr = PetscMemzero(sf->ops,sizeof(*sf->ops));CHKERRQ(ierr);
  ierr = PetscObjectChangeTypeName((PetscObject)sf,type);CHKERRQ(ierr);
  ierr = (*r)(sf);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  PetscSFGetType - Get the PetscSF communication implementation

  Not Collective

  Input Parameter:
. sf  - the PetscSF context

  Output Parameter:
. type - the PetscSF type name

  Level: intermediate

.seealso: PetscSFSetType(), PetscSFCreate()
@*/
PetscErrorCode PetscSFGetType(PetscSF sf, PetscSFType *type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(sf, PETSCSF_CLASSID,1);
  PetscValidPointer(type,2);
  *type = ((PetscObject)sf)->type_name;
  PetscFunctionReturn(0);
}

/*@
   PetscSFDestroy - destroy star forest

   Collective

   Input Arguments:
.  sf - address of star forest

   Level: intermediate

.seealso: PetscSFCreate(), PetscSFReset()
@*/
PetscErrorCode PetscSFDestroy(PetscSF *sf)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!*sf) PetscFunctionReturn(0);
  PetscValidHeaderSpecific((*sf),PETSCSF_CLASSID,1);
  if (--((PetscObject)(*sf))->refct > 0) {*sf = NULL; PetscFunctionReturn(0);}
  ierr = PetscSFReset(*sf);CHKERRQ(ierr);
  if ((*sf)->ops->Destroy) {ierr = (*(*sf)->ops->Destroy)(*sf);CHKERRQ(ierr);}
  ierr = PetscHeaderDestroy(sf);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSFCheckGraphValid_Private(PetscSF sf)
{
  PetscInt           i, nleaves;
  PetscMPIInt        size;
  const PetscInt    *ilocal;
  const PetscSFNode *iremote;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  if (!sf->graphset || !PetscDefined(USE_DEBUG)) PetscFunctionReturn(0);
  ierr = PetscSFGetGraph(sf,NULL,&nleaves,&ilocal,&iremote);CHKERRQ(ierr);
  ierr = MPI_Comm_size(PetscObjectComm((PetscObject)sf),&size);CHKERRQ(ierr);
  for (i = 0; i < nleaves; i++) {
    const PetscInt rank = iremote[i].rank;
    const PetscInt remote = iremote[i].index;
    const PetscInt leaf = ilocal ? ilocal[i] : i;
    if (rank < 0 || rank >= size) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Provided rank (%D) for remote %D is invalid, should be in [0, %d)",rank,i,size);
    if (remote < 0) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Provided index (%D) for remote %D is invalid, should be >= 0",remote,i);
    if (leaf < 0) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Provided location (%D) for leaf %D is invalid, should be >= 0",leaf,i);
  }
  PetscFunctionReturn(0);
}




/*@
   PetscSFSetUp - set up communication structures

   Collective

   Input Arguments:
.  sf - star forest communication object

   Level: beginner

.seealso: PetscSFSetFromOptions(), PetscSFSetType()
@*/
PetscErrorCode PetscSFSetUp(PetscSF sf)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sf,PETSCSF_CLASSID,1);
  PetscSFCheckGraphSet(sf,1);
  if (sf->setupcalled) PetscFunctionReturn(0);
  ierr = PetscLogEventBegin(PETSCSF_SetUp,sf,0,0,0);CHKERRQ(ierr);
  ierr = PetscSFCheckGraphValid_Private(sf);CHKERRQ(ierr);
  if (!((PetscObject)sf)->type_name) {ierr = PetscSFSetType(sf,PETSCSFBASIC);CHKERRQ(ierr);} /* Zero all sf->ops */
  if (sf->ops->SetUp) {ierr = (*sf->ops->SetUp)(sf);CHKERRQ(ierr);}
#if defined(PETSC_HAVE_CUDA)
  if (sf->backend == PETSCSF_BACKEND_CUDA) {
    sf->ops->Malloc = PetscSFMalloc_Cuda;
    sf->ops->Free   = PetscSFFree_Cuda;
  }
#endif

#if defined(PETSC_HAVE_KOKKOS)
  if (sf->backend == PETSCSF_BACKEND_KOKKOS) {
    sf->ops->Malloc = PetscSFMalloc_Kokkos;
    sf->ops->Free   = PetscSFFree_Kokkos;
  }
#endif
  ierr = PetscLogEventEnd(PETSCSF_SetUp,sf,0,0,0);CHKERRQ(ierr);
  sf->setupcalled = PETSC_TRUE;
  PetscFunctionReturn(0);
}

/*@
   PetscSFSetFromOptions - set PetscSF options using the options database

   Logically Collective

   Input Arguments:
.  sf - star forest

   Options Database Keys:
+  -sf_type               - implementation type, see PetscSFSetType()
.  -sf_rank_order         - sort composite points for gathers and scatters in rank order, gathers are non-deterministic otherwise
.  -sf_use_default_stream - Assume callers of SF computed the input root/leafdata with the default cuda stream. SF will also
                            use the default stream to process data. Therefore, no stream synchronization is needed between SF and its caller (default: true).
                            If true, this option only works with -use_gpu_aware_mpi 1.
.  -sf_use_stream_aware_mpi  - Assume the underlying MPI is cuda-stream aware and SF won't sync streams for send/recv buffers passed to MPI (default: false).
                               If true, this option only works with -use_gpu_aware_mpi 1.

-  -sf_backend cuda | kokkos -Select the device backend SF uses. Currently SF has two backends: cuda and Kokkos.
                              On CUDA devices, one can choose cuda or kokkos with the default being cuda. On other devices,
                              the only available is kokkos.

   Level: intermediate
@*/
PetscErrorCode PetscSFSetFromOptions(PetscSF sf)
{
  PetscSFType    deft;
  char           type[256];
  PetscErrorCode ierr;
  PetscBool      flg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sf,PETSCSF_CLASSID,1);
  ierr = PetscObjectOptionsBegin((PetscObject)sf);CHKERRQ(ierr);
  deft = ((PetscObject)sf)->type_name ? ((PetscObject)sf)->type_name : PETSCSFBASIC;
  ierr = PetscOptionsFList("-sf_type","PetscSF implementation type","PetscSFSetType",PetscSFList,deft,type,sizeof(type),&flg);CHKERRQ(ierr);
  ierr = PetscSFSetType(sf,flg ? type : deft);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-sf_rank_order","sort composite points for gathers and scatters in rank order, gathers are non-deterministic otherwise","PetscSFSetRankOrder",sf->rankorder,&sf->rankorder,NULL);CHKERRQ(ierr);
#if defined(PETSC_HAVE_DEVICE)
  {
    char        backendstr[32] = {0};
    PetscBool   isCuda = PETSC_FALSE,isKokkos = PETSC_FALSE,set;
    /* Change the defaults set in PetscSFCreate() with command line options */
    ierr = PetscOptionsBool("-sf_use_default_stream","Assume SF's input and output root/leafdata is computed on the default stream","PetscSFSetFromOptions",sf->use_default_stream,&sf->use_default_stream,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsBool("-sf_use_stream_aware_mpi","Assume the underlying MPI is cuda-stream aware","PetscSFSetFromOptions",sf->use_stream_aware_mpi,&sf->use_stream_aware_mpi,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsString("-sf_backend","Select the device backend SF uses","PetscSFSetFromOptions",NULL,backendstr,sizeof(backendstr),&set);CHKERRQ(ierr);
    ierr = PetscStrcasecmp("cuda",backendstr,&isCuda);CHKERRQ(ierr);
    ierr = PetscStrcasecmp("kokkos",backendstr,&isKokkos);CHKERRQ(ierr);
  #if defined(PETSC_HAVE_CUDA)
    if (isCuda) sf->backend = PETSCSF_BACKEND_CUDA;
    else if (isKokkos) sf->backend = PETSCSF_BACKEND_KOKKOS;
    else if (set) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"-sf_backend %s is not supported on cuda devices. You may choose cuda or kokkos (if installed)", backendstr);
  #elif defined(PETSC_HAVE_KOKKOS)
    if (set && !isKokkos) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"-sf_backend %s is not supported. You can only choose kokkos", backendstr);
  #endif
  }
#endif
  if (sf->ops->SetFromOptions) {ierr = (*sf->ops->SetFromOptions)(PetscOptionsObject,sf);CHKERRQ(ierr);}
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
   PetscSFSetRankOrder - sort multi-points for gathers and scatters by rank order

   Logically Collective

   Input Arguments:
+  sf - star forest
-  flg - PETSC_TRUE to sort, PETSC_FALSE to skip sorting (lower setup cost, but non-deterministic)

   Level: advanced

.seealso: PetscSFGatherBegin(), PetscSFScatterBegin()
@*/
PetscErrorCode PetscSFSetRankOrder(PetscSF sf,PetscBool flg)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(sf,PETSCSF_CLASSID,1);
  PetscValidLogicalCollectiveBool(sf,flg,2);
  if (sf->multi) SETERRQ(PetscObjectComm((PetscObject)sf),PETSC_ERR_ARG_WRONGSTATE,"Rank ordering must be set before first call to PetscSFGatherBegin() or PetscSFScatterBegin()");
  sf->rankorder = flg;
  PetscFunctionReturn(0);
}

/*@
   PetscSFSetGraph - Set a parallel star forest

   Collective

   Input Arguments:
+  sf - star forest
.  nroots - number of root vertices on the current process (these are possible targets for other process to attach leaves)
.  nleaves - number of leaf vertices on the current process, each of these references a root on any process
.  ilocal - locations of leaves in leafdata buffers, pass NULL for contiguous storage (locations must be >= 0, enforced
during setup in debug mode)
.  localmode - copy mode for ilocal
.  iremote - remote locations of root vertices for each leaf on the current process (locations must be >= 0, enforced
during setup in debug mode)
-  remotemode - copy mode for iremote

   Level: intermediate

   Notes:
    In Fortran you must use PETSC_COPY_VALUES for localmode and remotemode

   Developers Note: Local indices which are the identity permutation in the range [0,nleaves) are discarded as they
   encode contiguous storage. In such case, if localmode is PETSC_OWN_POINTER, the memory is deallocated as it is not
   needed

   Developers Note: This object does not necessarily encode a true star forest in the graph theoretic sense, since leaf
   indices are not required to be unique. Some functions, however, rely on unique leaf indices (checked in debug mode).

.seealso: PetscSFCreate(), PetscSFView(), PetscSFGetGraph()
@*/
PetscErrorCode PetscSFSetGraph(PetscSF sf,PetscInt nroots,PetscInt nleaves,const PetscInt *ilocal,PetscCopyMode localmode,const PetscSFNode *iremote,PetscCopyMode remotemode)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sf,PETSCSF_CLASSID,1);
  if (nleaves > 0 && ilocal) PetscValidIntPointer(ilocal,4);
  if (nleaves > 0) PetscValidPointer(iremote,6);
  if (nroots  < 0) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"nroots %D, cannot be negative",nroots);
  if (nleaves < 0) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"nleaves %D, cannot be negative",nleaves);

  if (sf->nroots >= 0) { /* Reset only if graph already set */
    ierr = PetscSFReset(sf);CHKERRQ(ierr);
  }

  ierr = PetscLogEventBegin(PETSCSF_SetGraph,sf,0,0,0);CHKERRQ(ierr);

  sf->nroots  = nroots;
  sf->nleaves = nleaves;

  if (nleaves && ilocal) {
    PetscInt i;
    PetscInt minleaf = PETSC_MAX_INT;
    PetscInt maxleaf = PETSC_MIN_INT;
    int      contiguous = 1;
    for (i=0; i<nleaves; i++) {
      minleaf = PetscMin(minleaf,ilocal[i]);
      maxleaf = PetscMax(maxleaf,ilocal[i]);
      contiguous &= (ilocal[i] == i);
    }
    sf->minleaf = minleaf;
    sf->maxleaf = maxleaf;
    if (contiguous) {
      if (localmode == PETSC_OWN_POINTER) {
        ierr = PetscFree(ilocal);CHKERRQ(ierr);
      }
      ilocal = NULL;
    }
  } else {
    sf->minleaf = 0;
    sf->maxleaf = nleaves - 1;
  }

  if (ilocal) {
    switch (localmode) {
    case PETSC_COPY_VALUES:
      ierr = PetscMalloc1(nleaves,&sf->mine_alloc);CHKERRQ(ierr);
      ierr = PetscArraycpy(sf->mine_alloc,ilocal,nleaves);CHKERRQ(ierr);
      sf->mine = sf->mine_alloc;
      break;
    case PETSC_OWN_POINTER:
      sf->mine_alloc = (PetscInt*)ilocal;
      sf->mine       = sf->mine_alloc;
      break;
    case PETSC_USE_POINTER:
      sf->mine_alloc = NULL;
      sf->mine       = (PetscInt*)ilocal;
      break;
    default: SETERRQ(PetscObjectComm((PetscObject)sf),PETSC_ERR_ARG_OUTOFRANGE,"Unknown localmode");
    }
  }

  switch (remotemode) {
  case PETSC_COPY_VALUES:
    ierr = PetscMalloc1(nleaves,&sf->remote_alloc);CHKERRQ(ierr);
    ierr = PetscArraycpy(sf->remote_alloc,iremote,nleaves);CHKERRQ(ierr);
    sf->remote = sf->remote_alloc;
    break;
  case PETSC_OWN_POINTER:
    sf->remote_alloc = (PetscSFNode*)iremote;
    sf->remote       = sf->remote_alloc;
    break;
  case PETSC_USE_POINTER:
    sf->remote_alloc = NULL;
    sf->remote       = (PetscSFNode*)iremote;
    break;
  default: SETERRQ(PetscObjectComm((PetscObject)sf),PETSC_ERR_ARG_OUTOFRANGE,"Unknown remotemode");
  }

  ierr = PetscLogEventEnd(PETSCSF_SetGraph,sf,0,0,0);CHKERRQ(ierr);
  sf->graphset = PETSC_TRUE;
  PetscFunctionReturn(0);
}

/*@
  PetscSFSetGraphWithPattern - Sets the graph of an SF with a specific pattern

  Collective

  Input Parameters:
+ sf      - The PetscSF
. map     - Layout of roots over all processes (insignificant when pattern is PETSCSF_PATTERN_ALLTOALL)
- pattern - One of PETSCSF_PATTERN_ALLGATHER, PETSCSF_PATTERN_GATHER, PETSCSF_PATTERN_ALLTOALL

  Notes:
  It is easier to explain PetscSFPattern using vectors. Suppose we have an MPI vector x and its layout is map.
  n and N are local and global sizes of x respectively.

  With PETSCSF_PATTERN_ALLGATHER, the routine creates a graph that if one does Bcast on it, it will copy x to
  sequential vectors y on all ranks.

  With PETSCSF_PATTERN_GATHER, the routine creates a graph that if one does Bcast on it, it will copy x to a
  sequential vector y on rank 0.

  In above cases, entries of x are roots and entries of y are leaves.

  With PETSCSF_PATTERN_ALLTOALL, map is insignificant. Suppose NP is size of sf's communicator. The routine
  creates a graph that every rank has NP leaves and NP roots. On rank i, its leaf j is connected to root i
  of rank j. Here 0 <=i,j<NP. It is a kind of MPI_Alltoall with sendcount/recvcount being 1. Note that it does
  not mean one can not send multiple items. One just needs to create a new MPI datatype for the mulptiple data
  items with MPI_Type_contiguous() and use that as the <unit> argument in SF routines.

  In this case, roots and leaves are symmetric.

  Level: intermediate
 @*/
PetscErrorCode PetscSFSetGraphWithPattern(PetscSF sf,PetscLayout map,PetscSFPattern pattern)
{
  MPI_Comm       comm;
  PetscInt       n,N,res[2];
  PetscMPIInt    rank,size;
  PetscSFType    type;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)sf, &comm);CHKERRQ(ierr);
  if (pattern < PETSCSF_PATTERN_ALLGATHER || pattern > PETSCSF_PATTERN_ALLTOALL) SETERRQ1(comm,PETSC_ERR_ARG_OUTOFRANGE,"Unsupported PetscSFPattern %D\n",pattern);
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);

  if (pattern == PETSCSF_PATTERN_ALLTOALL) {
    type = PETSCSFALLTOALL;
    ierr = PetscLayoutCreate(comm,&sf->map);CHKERRQ(ierr);
    ierr = PetscLayoutSetLocalSize(sf->map,size);CHKERRQ(ierr);
    ierr = PetscLayoutSetSize(sf->map,((PetscInt)size)*size);CHKERRQ(ierr);
    ierr = PetscLayoutSetUp(sf->map);CHKERRQ(ierr);
  } else {
    ierr   = PetscLayoutGetLocalSize(map,&n);CHKERRQ(ierr);
    ierr   = PetscLayoutGetSize(map,&N);CHKERRQ(ierr);
    res[0] = n;
    res[1] = -n;
    /* Check if n are same over all ranks so that we can optimize it */
    ierr   = MPIU_Allreduce(MPI_IN_PLACE,res,2,MPIU_INT,MPI_MAX,comm);CHKERRQ(ierr);
    if (res[0] == -res[1]) { /* same n */
      type = (pattern == PETSCSF_PATTERN_ALLGATHER) ? PETSCSFALLGATHER  : PETSCSFGATHER;
    } else {
      type = (pattern == PETSCSF_PATTERN_ALLGATHER) ? PETSCSFALLGATHERV : PETSCSFGATHERV;
    }
    ierr = PetscLayoutReference(map,&sf->map);CHKERRQ(ierr);
  }
  ierr = PetscSFSetType(sf,type);CHKERRQ(ierr);

  sf->pattern = pattern;
  sf->mine    = NULL; /* Contiguous */

  /* Set nleaves, nroots here in case user calls PetscSFGetGraph, which is legal to call even before PetscSFSetUp is called.
     Also set other easy stuff.
   */
  if (pattern == PETSCSF_PATTERN_ALLGATHER) {
    sf->nleaves      = N;
    sf->nroots       = n;
    sf->nranks       = size;
    sf->minleaf      = 0;
    sf->maxleaf      = N - 1;
  } else if (pattern == PETSCSF_PATTERN_GATHER) {
    sf->nleaves      = rank ? 0 : N;
    sf->nroots       = n;
    sf->nranks       = rank ? 0 : size;
    sf->minleaf      = 0;
    sf->maxleaf      = rank ? -1 : N - 1;
  } else if (pattern == PETSCSF_PATTERN_ALLTOALL) {
    sf->nleaves      = size;
    sf->nroots       = size;
    sf->nranks       = size;
    sf->minleaf      = 0;
    sf->maxleaf      = size - 1;
  }
  sf->ndranks  = 0; /* We do not need to separate out distinguished ranks for patterned graphs to improve communication performance */
  sf->graphset = PETSC_TRUE;
  PetscFunctionReturn(0);
}

/*@
   PetscSFCreateInverseSF - given a PetscSF in which all vertices have degree 1, creates the inverse map

   Collective

   Input Arguments:

.  sf - star forest to invert

   Output Arguments:
.  isf - inverse of sf
   Level: advanced

   Notes:
   All roots must have degree 1.

   The local space may be a permutation, but cannot be sparse.

.seealso: PetscSFSetGraph()
@*/
PetscErrorCode PetscSFCreateInverseSF(PetscSF sf,PetscSF *isf)
{
  PetscErrorCode ierr;
  PetscMPIInt    rank;
  PetscInt       i,nroots,nleaves,maxlocal,count,*newilocal;
  const PetscInt *ilocal;
  PetscSFNode    *roots,*leaves;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sf,PETSCSF_CLASSID,1);
  PetscSFCheckGraphSet(sf,1);
  PetscValidPointer(isf,2);

  ierr = PetscSFGetGraph(sf,&nroots,&nleaves,&ilocal,NULL);CHKERRQ(ierr);
  maxlocal = sf->maxleaf+1; /* TODO: We should use PetscSFGetLeafRange() */

  ierr = MPI_Comm_rank(PetscObjectComm((PetscObject)sf),&rank);CHKERRQ(ierr);
  ierr = PetscMalloc2(nroots,&roots,maxlocal,&leaves);CHKERRQ(ierr);
  for (i=0; i<maxlocal; i++) {
    leaves[i].rank  = rank;
    leaves[i].index = i;
  }
  for (i=0; i <nroots; i++) {
    roots[i].rank  = -1;
    roots[i].index = -1;
  }
  ierr = PetscSFReduceBegin(sf,MPIU_2INT,leaves,roots,MPIU_REPLACE);CHKERRQ(ierr);
  ierr = PetscSFReduceEnd(sf,MPIU_2INT,leaves,roots,MPIU_REPLACE);CHKERRQ(ierr);

  /* Check whether our leaves are sparse */
  for (i=0,count=0; i<nroots; i++) if (roots[i].rank >= 0) count++;
  if (count == nroots) newilocal = NULL;
  else {                        /* Index for sparse leaves and compact "roots" array (which is to become our leaves). */
    ierr = PetscMalloc1(count,&newilocal);CHKERRQ(ierr);
    for (i=0,count=0; i<nroots; i++) {
      if (roots[i].rank >= 0) {
        newilocal[count]   = i;
        roots[count].rank  = roots[i].rank;
        roots[count].index = roots[i].index;
        count++;
      }
    }
  }

  ierr = PetscSFDuplicate(sf,PETSCSF_DUPLICATE_CONFONLY,isf);CHKERRQ(ierr);
  ierr = PetscSFSetGraph(*isf,maxlocal,count,newilocal,PETSC_OWN_POINTER,roots,PETSC_COPY_VALUES);CHKERRQ(ierr);
  ierr = PetscFree2(roots,leaves);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
   PetscSFDuplicate - duplicate a PetscSF, optionally preserving rank connectivity and graph

   Collective

   Input Arguments:
+  sf - communication object to duplicate
-  opt - PETSCSF_DUPLICATE_CONFONLY, PETSCSF_DUPLICATE_RANKS, or PETSCSF_DUPLICATE_GRAPH (see PetscSFDuplicateOption)

   Output Arguments:
.  newsf - new communication object

   Level: beginner

.seealso: PetscSFCreate(), PetscSFSetType(), PetscSFSetGraph()
@*/
PetscErrorCode PetscSFDuplicate(PetscSF sf,PetscSFDuplicateOption opt,PetscSF *newsf)
{
  PetscSFType    type;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sf,PETSCSF_CLASSID,1);
  PetscValidLogicalCollectiveEnum(sf,opt,2);
  PetscValidPointer(newsf,3);
  ierr = PetscSFCreate(PetscObjectComm((PetscObject)sf),newsf);CHKERRQ(ierr);
  ierr = PetscSFGetType(sf,&type);CHKERRQ(ierr);
  if (type) {ierr = PetscSFSetType(*newsf,type);CHKERRQ(ierr);}
  if (opt == PETSCSF_DUPLICATE_GRAPH) {
    PetscSFCheckGraphSet(sf,1);
    if (sf->pattern == PETSCSF_PATTERN_GENERAL) {
      PetscInt          nroots,nleaves;
      const PetscInt    *ilocal;
      const PetscSFNode *iremote;
      ierr = PetscSFGetGraph(sf,&nroots,&nleaves,&ilocal,&iremote);CHKERRQ(ierr);
      ierr = PetscSFSetGraph(*newsf,nroots,nleaves,ilocal,PETSC_COPY_VALUES,iremote,PETSC_COPY_VALUES);CHKERRQ(ierr);
    } else {
      ierr = PetscSFSetGraphWithPattern(*newsf,sf->map,sf->pattern);CHKERRQ(ierr);
    }
  }
#if defined(PETSC_HAVE_DEVICE)
  (*newsf)->backend              = sf->backend;
  (*newsf)->use_default_stream   = sf->use_default_stream;
  (*newsf)->use_gpu_aware_mpi    = sf->use_gpu_aware_mpi;
  (*newsf)->use_stream_aware_mpi = sf->use_stream_aware_mpi;
#endif
  if (sf->ops->Duplicate) {ierr = (*sf->ops->Duplicate)(sf,opt,*newsf);CHKERRQ(ierr);}
  /* Don't do PetscSFSetUp() since the new sf's graph might have not been set. */
  PetscFunctionReturn(0);
}

/*@C
   PetscSFGetGraph - Get the graph specifying a parallel star forest

   Not Collective

   Input Arguments:
.  sf - star forest

   Output Arguments:
+  nroots - number of root vertices on the current process (these are possible targets for other process to attach leaves)
.  nleaves - number of leaf vertices on the current process, each of these references a root on any process
.  ilocal - locations of leaves in leafdata buffers
-  iremote - remote locations of root vertices for each leaf on the current process

   Notes:
   We are not currently requiring that the graph is set, thus returning nroots=-1 if it has not been set yet

   When called from Fortran, the returned iremote array is a copy and must be deallocated after use. Consequently, if you
   want to update the graph, you must call PetscSFSetGraph after modifying the iremote array.

   Level: intermediate

.seealso: PetscSFCreate(), PetscSFView(), PetscSFSetGraph()
@*/
PetscErrorCode PetscSFGetGraph(PetscSF sf,PetscInt *nroots,PetscInt *nleaves,const PetscInt **ilocal,const PetscSFNode **iremote)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sf,PETSCSF_CLASSID,1);
  if (sf->ops->GetGraph) {
    ierr = (sf->ops->GetGraph)(sf,nroots,nleaves,ilocal,iremote);CHKERRQ(ierr);
  } else {
    if (nroots) *nroots = sf->nroots;
    if (nleaves) *nleaves = sf->nleaves;
    if (ilocal) *ilocal = sf->mine;
    if (iremote) *iremote = sf->remote;
  }
  PetscFunctionReturn(0);
}

/*@
   PetscSFGetLeafRange - Get the active leaf ranges

   Not Collective

   Input Arguments:
.  sf - star forest

   Output Arguments:
+  minleaf - minimum active leaf on this process. Return 0 if there are no leaves.
-  maxleaf - maximum active leaf on this process. Return -1 if there are no leaves.

   Level: developer

.seealso: PetscSFCreate(), PetscSFView(), PetscSFSetGraph(), PetscSFGetGraph()
@*/
PetscErrorCode PetscSFGetLeafRange(PetscSF sf,PetscInt *minleaf,PetscInt *maxleaf)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(sf,PETSCSF_CLASSID,1);
  PetscSFCheckGraphSet(sf,1);
  if (minleaf) *minleaf = sf->minleaf;
  if (maxleaf) *maxleaf = sf->maxleaf;
  PetscFunctionReturn(0);
}

/*@C
   PetscSFViewFromOptions - View from Options

   Collective on PetscSF

   Input Parameters:
+  A - the star forest
.  obj - Optional object
-  name - command line option

   Level: intermediate
.seealso:  PetscSF, PetscSFView, PetscObjectViewFromOptions(), PetscSFCreate()
@*/
PetscErrorCode  PetscSFViewFromOptions(PetscSF A,PetscObject obj,const char name[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,PETSCSF_CLASSID,1);
  ierr = PetscObjectViewFromOptions((PetscObject)A,obj,name);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   PetscSFView - view a star forest

   Collective

   Input Arguments:
+  sf - star forest
-  viewer - viewer to display graph, for example PETSC_VIEWER_STDOUT_WORLD

   Level: beginner

.seealso: PetscSFCreate(), PetscSFSetGraph()
@*/
PetscErrorCode PetscSFView(PetscSF sf,PetscViewer viewer)
{
  PetscErrorCode    ierr;
  PetscBool         iascii;
  PetscViewerFormat format;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sf,PETSCSF_CLASSID,1);
  if (!viewer) {ierr = PetscViewerASCIIGetStdout(PetscObjectComm((PetscObject)sf),&viewer);CHKERRQ(ierr);}
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,2);
  PetscCheckSameComm(sf,1,viewer,2);
  if (sf->graphset) {ierr = PetscSFSetUp(sf);CHKERRQ(ierr);}
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    PetscMPIInt rank;
    PetscInt    ii,i,j;

    ierr = PetscObjectPrintClassNamePrefixType((PetscObject)sf,viewer);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
    if (sf->ops->View) {ierr = (*sf->ops->View)(sf,viewer);CHKERRQ(ierr);}
    if (sf->pattern == PETSCSF_PATTERN_GENERAL) {
      if (!sf->graphset) {
        ierr = PetscViewerASCIIPrintf(viewer,"PetscSFSetGraph() has not been called yet\n");CHKERRQ(ierr);
        ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
        PetscFunctionReturn(0);
      }
      ierr = MPI_Comm_rank(PetscObjectComm((PetscObject)sf),&rank);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPushSynchronized(viewer);CHKERRQ(ierr);
      ierr = PetscViewerASCIISynchronizedPrintf(viewer,"[%d] Number of roots=%D, leaves=%D, remote ranks=%D\n",rank,sf->nroots,sf->nleaves,sf->nranks);CHKERRQ(ierr);
      for (i=0; i<sf->nleaves; i++) {
        ierr = PetscViewerASCIISynchronizedPrintf(viewer,"[%d] %D <- (%D,%D)\n",rank,sf->mine ? sf->mine[i] : i,sf->remote[i].rank,sf->remote[i].index);CHKERRQ(ierr);
      }
      ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
      ierr = PetscViewerGetFormat(viewer,&format);CHKERRQ(ierr);
      if (format == PETSC_VIEWER_ASCII_INFO_DETAIL) {
        PetscMPIInt *tmpranks,*perm;
        ierr = PetscMalloc2(sf->nranks,&tmpranks,sf->nranks,&perm);CHKERRQ(ierr);
        ierr = PetscArraycpy(tmpranks,sf->ranks,sf->nranks);CHKERRQ(ierr);
        for (i=0; i<sf->nranks; i++) perm[i] = i;
        ierr = PetscSortMPIIntWithArray(sf->nranks,tmpranks,perm);CHKERRQ(ierr);
        ierr = PetscViewerASCIISynchronizedPrintf(viewer,"[%d] Roots referenced by my leaves, by rank\n",rank);CHKERRQ(ierr);
        for (ii=0; ii<sf->nranks; ii++) {
          i = perm[ii];
          ierr = PetscViewerASCIISynchronizedPrintf(viewer,"[%d] %d: %D edges\n",rank,sf->ranks[i],sf->roffset[i+1]-sf->roffset[i]);CHKERRQ(ierr);
          for (j=sf->roffset[i]; j<sf->roffset[i+1]; j++) {
            ierr = PetscViewerASCIISynchronizedPrintf(viewer,"[%d]    %D <- %D\n",rank,sf->rmine[j],sf->rremote[j]);CHKERRQ(ierr);
          }
        }
        ierr = PetscFree2(tmpranks,perm);CHKERRQ(ierr);
      }
      ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPopSynchronized(viewer);CHKERRQ(ierr);
    }
    ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*@C
   PetscSFGetRootRanks - Get root ranks and number of vertices referenced by leaves on this process

   Not Collective

   Input Arguments:
.  sf - star forest

   Output Arguments:
+  nranks - number of ranks referenced by local part
.  ranks - array of ranks
.  roffset - offset in rmine/rremote for each rank (length nranks+1)
.  rmine - concatenated array holding local indices referencing each remote rank
-  rremote - concatenated array holding remote indices referenced for each remote rank

   Level: developer

.seealso: PetscSFGetLeafRanks()
@*/
PetscErrorCode PetscSFGetRootRanks(PetscSF sf,PetscInt *nranks,const PetscMPIInt **ranks,const PetscInt **roffset,const PetscInt **rmine,const PetscInt **rremote)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sf,PETSCSF_CLASSID,1);
  if (!sf->setupcalled) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Must call PetscSFSetUp() before obtaining ranks");
  if (sf->ops->GetRootRanks) {
    ierr = (sf->ops->GetRootRanks)(sf,nranks,ranks,roffset,rmine,rremote);CHKERRQ(ierr);
  } else {
    /* The generic implementation */
    if (nranks)  *nranks  = sf->nranks;
    if (ranks)   *ranks   = sf->ranks;
    if (roffset) *roffset = sf->roffset;
    if (rmine)   *rmine   = sf->rmine;
    if (rremote) *rremote = sf->rremote;
  }
  PetscFunctionReturn(0);
}

/*@C
   PetscSFGetLeafRanks - Get leaf ranks referencing roots on this process

   Not Collective

   Input Arguments:
.  sf - star forest

   Output Arguments:
+  niranks - number of leaf ranks referencing roots on this process
.  iranks - array of ranks
.  ioffset - offset in irootloc for each rank (length niranks+1)
-  irootloc - concatenated array holding local indices of roots referenced by each leaf rank

   Level: developer

.seealso: PetscSFGetRootRanks()
@*/
PetscErrorCode PetscSFGetLeafRanks(PetscSF sf,PetscInt *niranks,const PetscMPIInt **iranks,const PetscInt **ioffset,const PetscInt **irootloc)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sf,PETSCSF_CLASSID,1);
  if (!sf->setupcalled) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Must call PetscSFSetUp() before obtaining ranks");
  if (sf->ops->GetLeafRanks) {
    ierr = (sf->ops->GetLeafRanks)(sf,niranks,iranks,ioffset,irootloc);CHKERRQ(ierr);
  } else {
    PetscSFType type;
    ierr = PetscSFGetType(sf,&type);CHKERRQ(ierr);
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_SUP,"PetscSFGetLeafRanks() is not supported on this StarForest type: %s", type);
  }
  PetscFunctionReturn(0);
}

static PetscBool InList(PetscMPIInt needle,PetscMPIInt n,const PetscMPIInt *list) {
  PetscInt i;
  for (i=0; i<n; i++) {
    if (needle == list[i]) return PETSC_TRUE;
  }
  return PETSC_FALSE;
}

/*@C
   PetscSFSetUpRanks - Set up data structures associated with ranks; this is for internal use by PetscSF implementations.

   Collective

   Input Arguments:
+  sf - PetscSF to set up; PetscSFSetGraph() must have been called
-  dgroup - MPI_Group of ranks to be distinguished (e.g., for self or shared memory exchange)

   Level: developer

.seealso: PetscSFGetRootRanks()
@*/
PetscErrorCode PetscSFSetUpRanks(PetscSF sf,MPI_Group dgroup)
{
  PetscErrorCode     ierr;
  PetscTable         table;
  PetscTablePosition pos;
  PetscMPIInt        size,groupsize,*groupranks;
  PetscInt           *rcount,*ranks;
  PetscInt           i, irank = -1,orank = -1;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sf,PETSCSF_CLASSID,1);
  PetscSFCheckGraphSet(sf,1);
  ierr = MPI_Comm_size(PetscObjectComm((PetscObject)sf),&size);CHKERRQ(ierr);
  ierr = PetscTableCreate(10,size,&table);CHKERRQ(ierr);
  for (i=0; i<sf->nleaves; i++) {
    /* Log 1-based rank */
    ierr = PetscTableAdd(table,sf->remote[i].rank+1,1,ADD_VALUES);CHKERRQ(ierr);
  }
  ierr = PetscTableGetCount(table,&sf->nranks);CHKERRQ(ierr);
  ierr = PetscMalloc4(sf->nranks,&sf->ranks,sf->nranks+1,&sf->roffset,sf->nleaves,&sf->rmine,sf->nleaves,&sf->rremote);CHKERRQ(ierr);
  ierr = PetscMalloc2(sf->nranks,&rcount,sf->nranks,&ranks);CHKERRQ(ierr);
  ierr = PetscTableGetHeadPosition(table,&pos);CHKERRQ(ierr);
  for (i=0; i<sf->nranks; i++) {
    ierr = PetscTableGetNext(table,&pos,&ranks[i],&rcount[i]);CHKERRQ(ierr);
    ranks[i]--;             /* Convert back to 0-based */
  }
  ierr = PetscTableDestroy(&table);CHKERRQ(ierr);

  /* We expect that dgroup is reliably "small" while nranks could be large */
  {
    MPI_Group group = MPI_GROUP_NULL;
    PetscMPIInt *dgroupranks;
    ierr = MPI_Comm_group(PetscObjectComm((PetscObject)sf),&group);CHKERRQ(ierr);
    ierr = MPI_Group_size(dgroup,&groupsize);CHKERRQ(ierr);
    ierr = PetscMalloc1(groupsize,&dgroupranks);CHKERRQ(ierr);
    ierr = PetscMalloc1(groupsize,&groupranks);CHKERRQ(ierr);
    for (i=0; i<groupsize; i++) dgroupranks[i] = i;
    if (groupsize) {ierr = MPI_Group_translate_ranks(dgroup,groupsize,dgroupranks,group,groupranks);CHKERRQ(ierr);}
    ierr = MPI_Group_free(&group);CHKERRQ(ierr);
    ierr = PetscFree(dgroupranks);CHKERRQ(ierr);
  }

  /* Partition ranks[] into distinguished (first sf->ndranks) followed by non-distinguished */
  for (sf->ndranks=0,i=sf->nranks; sf->ndranks<i;) {
    for (i--; sf->ndranks<i; i--) { /* Scan i backward looking for distinguished rank */
      if (InList(ranks[i],groupsize,groupranks)) break;
    }
    for (; sf->ndranks<=i; sf->ndranks++) { /* Scan sf->ndranks forward looking for non-distinguished rank */
      if (!InList(ranks[sf->ndranks],groupsize,groupranks)) break;
    }
    if (sf->ndranks < i) {                         /* Swap ranks[sf->ndranks] with ranks[i] */
      PetscInt tmprank,tmpcount;

      tmprank             = ranks[i];
      tmpcount            = rcount[i];
      ranks[i]            = ranks[sf->ndranks];
      rcount[i]           = rcount[sf->ndranks];
      ranks[sf->ndranks]  = tmprank;
      rcount[sf->ndranks] = tmpcount;
      sf->ndranks++;
    }
  }
  ierr = PetscFree(groupranks);CHKERRQ(ierr);
  ierr = PetscSortIntWithArray(sf->ndranks,ranks,rcount);CHKERRQ(ierr);
  ierr = PetscSortIntWithArray(sf->nranks-sf->ndranks,ranks+sf->ndranks,rcount+sf->ndranks);CHKERRQ(ierr);
  sf->roffset[0] = 0;
  for (i=0; i<sf->nranks; i++) {
    ierr = PetscMPIIntCast(ranks[i],sf->ranks+i);CHKERRQ(ierr);
    sf->roffset[i+1] = sf->roffset[i] + rcount[i];
    rcount[i]        = 0;
  }
  for (i=0, irank = -1, orank = -1; i<sf->nleaves; i++) {
    /* short circuit */
    if (orank != sf->remote[i].rank) {
      /* Search for index of iremote[i].rank in sf->ranks */
      ierr = PetscFindMPIInt(sf->remote[i].rank,sf->ndranks,sf->ranks,&irank);CHKERRQ(ierr);
      if (irank < 0) {
        ierr = PetscFindMPIInt(sf->remote[i].rank,sf->nranks-sf->ndranks,sf->ranks+sf->ndranks,&irank);CHKERRQ(ierr);
        if (irank >= 0) irank += sf->ndranks;
      }
      orank = sf->remote[i].rank;
    }
    if (irank < 0) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Could not find rank %D in array",sf->remote[i].rank);
    sf->rmine[sf->roffset[irank] + rcount[irank]]   = sf->mine ? sf->mine[i] : i;
    sf->rremote[sf->roffset[irank] + rcount[irank]] = sf->remote[i].index;
    rcount[irank]++;
  }
  ierr = PetscFree2(rcount,ranks);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   PetscSFGetGroups - gets incoming and outgoing process groups

   Collective

   Input Argument:
.  sf - star forest

   Output Arguments:
+  incoming - group of origin processes for incoming edges (leaves that reference my roots)
-  outgoing - group of destination processes for outgoing edges (roots that I reference)

   Level: developer

.seealso: PetscSFGetWindow(), PetscSFRestoreWindow()
@*/
PetscErrorCode PetscSFGetGroups(PetscSF sf,MPI_Group *incoming,MPI_Group *outgoing)
{
  PetscErrorCode ierr;
  MPI_Group      group = MPI_GROUP_NULL;

  PetscFunctionBegin;
  if (sf->nranks < 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Must call PetscSFSetUpRanks() before obtaining groups");
  if (sf->ingroup == MPI_GROUP_NULL) {
    PetscInt       i;
    const PetscInt *indegree;
    PetscMPIInt    rank,*outranks,*inranks;
    PetscSFNode    *remote;
    PetscSF        bgcount;

    /* Compute the number of incoming ranks */
    ierr = PetscMalloc1(sf->nranks,&remote);CHKERRQ(ierr);
    for (i=0; i<sf->nranks; i++) {
      remote[i].rank  = sf->ranks[i];
      remote[i].index = 0;
    }
    ierr = PetscSFDuplicate(sf,PETSCSF_DUPLICATE_CONFONLY,&bgcount);CHKERRQ(ierr);
    ierr = PetscSFSetGraph(bgcount,1,sf->nranks,NULL,PETSC_COPY_VALUES,remote,PETSC_OWN_POINTER);CHKERRQ(ierr);
    ierr = PetscSFComputeDegreeBegin(bgcount,&indegree);CHKERRQ(ierr);
    ierr = PetscSFComputeDegreeEnd(bgcount,&indegree);CHKERRQ(ierr);
    /* Enumerate the incoming ranks */
    ierr = PetscMalloc2(indegree[0],&inranks,sf->nranks,&outranks);CHKERRQ(ierr);
    ierr = MPI_Comm_rank(PetscObjectComm((PetscObject)sf),&rank);CHKERRQ(ierr);
    for (i=0; i<sf->nranks; i++) outranks[i] = rank;
    ierr = PetscSFGatherBegin(bgcount,MPI_INT,outranks,inranks);CHKERRQ(ierr);
    ierr = PetscSFGatherEnd(bgcount,MPI_INT,outranks,inranks);CHKERRQ(ierr);
    ierr = MPI_Comm_group(PetscObjectComm((PetscObject)sf),&group);CHKERRQ(ierr);
    ierr = MPI_Group_incl(group,indegree[0],inranks,&sf->ingroup);CHKERRQ(ierr);
    ierr = MPI_Group_free(&group);CHKERRQ(ierr);
    ierr = PetscFree2(inranks,outranks);CHKERRQ(ierr);
    ierr = PetscSFDestroy(&bgcount);CHKERRQ(ierr);
  }
  *incoming = sf->ingroup;

  if (sf->outgroup == MPI_GROUP_NULL) {
    ierr = MPI_Comm_group(PetscObjectComm((PetscObject)sf),&group);CHKERRQ(ierr);
    ierr = MPI_Group_incl(group,sf->nranks,sf->ranks,&sf->outgroup);CHKERRQ(ierr);
    ierr = MPI_Group_free(&group);CHKERRQ(ierr);
  }
  *outgoing = sf->outgroup;
  PetscFunctionReturn(0);
}

/*@
   PetscSFGetMultiSF - gets the inner SF implemeting gathers and scatters

   Collective

   Input Argument:
.  sf - star forest that may contain roots with 0 or with more than 1 vertex

   Output Arguments:
.  multi - star forest with split roots, such that each root has degree exactly 1

   Level: developer

   Notes:

   In most cases, users should use PetscSFGatherBegin() and PetscSFScatterBegin() instead of manipulating multi
   directly. Since multi satisfies the stronger condition that each entry in the global space has exactly one incoming
   edge, it is a candidate for future optimization that might involve its removal.

.seealso: PetscSFSetGraph(), PetscSFGatherBegin(), PetscSFScatterBegin(), PetscSFComputeMultiRootOriginalNumbering()
@*/
PetscErrorCode PetscSFGetMultiSF(PetscSF sf,PetscSF *multi)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sf,PETSCSF_CLASSID,1);
  PetscValidPointer(multi,2);
  if (sf->nroots < 0) {         /* Graph has not been set yet; why do we need this? */
    ierr   = PetscSFDuplicate(sf,PETSCSF_DUPLICATE_RANKS,&sf->multi);CHKERRQ(ierr);
    *multi = sf->multi;
    PetscFunctionReturn(0);
  }
  if (!sf->multi) {
    const PetscInt *indegree;
    PetscInt       i,*inoffset,*outones,*outoffset,maxlocal;
    PetscSFNode    *remote;
    maxlocal = sf->maxleaf+1; /* TODO: We should use PetscSFGetLeafRange() */
    ierr = PetscSFComputeDegreeBegin(sf,&indegree);CHKERRQ(ierr);
    ierr = PetscSFComputeDegreeEnd(sf,&indegree);CHKERRQ(ierr);
    ierr = PetscMalloc3(sf->nroots+1,&inoffset,maxlocal,&outones,maxlocal,&outoffset);CHKERRQ(ierr);
    inoffset[0] = 0;
    for (i=0; i<sf->nroots; i++) inoffset[i+1] = inoffset[i] + indegree[i];
    for (i=0; i<maxlocal; i++) outones[i] = 1;
    ierr = PetscSFFetchAndOpBegin(sf,MPIU_INT,inoffset,outones,outoffset,MPI_SUM);CHKERRQ(ierr);
    ierr = PetscSFFetchAndOpEnd(sf,MPIU_INT,inoffset,outones,outoffset,MPI_SUM);CHKERRQ(ierr);
    for (i=0; i<sf->nroots; i++) inoffset[i] -= indegree[i]; /* Undo the increment */
    if (PetscDefined(USE_DEBUG)) { /* Check that the expected number of increments occurred */
      for (i=0; i<sf->nroots; i++) {
        if (inoffset[i] + indegree[i] != inoffset[i+1]) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Incorrect result after PetscSFFetchAndOp");
      }
    }
    ierr = PetscMalloc1(sf->nleaves,&remote);CHKERRQ(ierr);
    for (i=0; i<sf->nleaves; i++) {
      remote[i].rank  = sf->remote[i].rank;
      remote[i].index = outoffset[sf->mine ? sf->mine[i] : i];
    }
    ierr = PetscSFDuplicate(sf,PETSCSF_DUPLICATE_RANKS,&sf->multi);CHKERRQ(ierr);
    ierr = PetscSFSetGraph(sf->multi,inoffset[sf->nroots],sf->nleaves,sf->mine,PETSC_COPY_VALUES,remote,PETSC_OWN_POINTER);CHKERRQ(ierr);
    if (sf->rankorder) {        /* Sort the ranks */
      PetscMPIInt rank;
      PetscInt    *inranks,*newoffset,*outranks,*newoutoffset,*tmpoffset,maxdegree;
      PetscSFNode *newremote;
      ierr = MPI_Comm_rank(PetscObjectComm((PetscObject)sf),&rank);CHKERRQ(ierr);
      for (i=0,maxdegree=0; i<sf->nroots; i++) maxdegree = PetscMax(maxdegree,indegree[i]);
      ierr = PetscMalloc5(sf->multi->nroots,&inranks,sf->multi->nroots,&newoffset,maxlocal,&outranks,maxlocal,&newoutoffset,maxdegree,&tmpoffset);CHKERRQ(ierr);
      for (i=0; i<maxlocal; i++) outranks[i] = rank;
      ierr = PetscSFReduceBegin(sf->multi,MPIU_INT,outranks,inranks,MPIU_REPLACE);CHKERRQ(ierr);
      ierr = PetscSFReduceEnd(sf->multi,MPIU_INT,outranks,inranks,MPIU_REPLACE);CHKERRQ(ierr);
      /* Sort the incoming ranks at each vertex, build the inverse map */
      for (i=0; i<sf->nroots; i++) {
        PetscInt j;
        for (j=0; j<indegree[i]; j++) tmpoffset[j] = j;
        ierr = PetscSortIntWithArray(indegree[i],inranks+inoffset[i],tmpoffset);CHKERRQ(ierr);
        for (j=0; j<indegree[i]; j++) newoffset[inoffset[i] + tmpoffset[j]] = inoffset[i] + j;
      }
      ierr = PetscSFBcastBegin(sf->multi,MPIU_INT,newoffset,newoutoffset);CHKERRQ(ierr);
      ierr = PetscSFBcastEnd(sf->multi,MPIU_INT,newoffset,newoutoffset);CHKERRQ(ierr);
      ierr = PetscMalloc1(sf->nleaves,&newremote);CHKERRQ(ierr);
      for (i=0; i<sf->nleaves; i++) {
        newremote[i].rank  = sf->remote[i].rank;
        newremote[i].index = newoutoffset[sf->mine ? sf->mine[i] : i];
      }
      ierr = PetscSFSetGraph(sf->multi,inoffset[sf->nroots],sf->nleaves,sf->mine,PETSC_COPY_VALUES,newremote,PETSC_OWN_POINTER);CHKERRQ(ierr);
      ierr = PetscFree5(inranks,newoffset,outranks,newoutoffset,tmpoffset);CHKERRQ(ierr);
    }
    ierr = PetscFree3(inoffset,outones,outoffset);CHKERRQ(ierr);
  }
  *multi = sf->multi;
  PetscFunctionReturn(0);
}

/*@C
   PetscSFCreateEmbeddedSF - removes edges from all but the selected roots, does not remap indices

   Collective

   Input Arguments:
+  sf - original star forest
.  nselected  - number of selected roots on this process
-  selected   - indices of the selected roots on this process

   Output Arguments:
.  esf - new star forest

   Level: advanced

   Note:
   To use the new PetscSF, it may be necessary to know the indices of the leaves that are still participating. This can
   be done by calling PetscSFGetGraph().

.seealso: PetscSFSetGraph(), PetscSFGetGraph()
@*/
PetscErrorCode PetscSFCreateEmbeddedSF(PetscSF sf,PetscInt nselected,const PetscInt *selected,PetscSF *esf)
{
  PetscInt          i,j,n,nroots,nleaves,esf_nleaves,*new_ilocal,minleaf,maxleaf,maxlocal;
  const PetscInt    *ilocal;
  signed char       *rootdata,*leafdata,*leafmem;
  const PetscSFNode *iremote;
  PetscSFNode       *new_iremote;
  MPI_Comm          comm;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sf,PETSCSF_CLASSID,1);
  PetscSFCheckGraphSet(sf,1);
  if (nselected) PetscValidPointer(selected,3);
  PetscValidPointer(esf,4);

  ierr = PetscSFSetUp(sf);CHKERRQ(ierr);
  ierr = PetscLogEventBegin(PETSCSF_EmbedSF,sf,0,0,0);CHKERRQ(ierr);
  ierr = PetscObjectGetComm((PetscObject)sf,&comm);CHKERRQ(ierr);
  ierr = PetscSFGetGraph(sf,&nroots,&nleaves,&ilocal,&iremote);CHKERRQ(ierr);

  if (PetscDefined(USE_DEBUG)) {  /* Error out if selected[] has dups or  out of range indices */
    PetscBool dups;
    ierr = PetscCheckDupsInt(nselected,selected,&dups);CHKERRQ(ierr);
    if (dups) SETERRQ(comm,PETSC_ERR_ARG_WRONG,"selected[] has dups");
    for (i=0; i<nselected; i++)
      if (selected[i] < 0 || selected[i] >= nroots) SETERRQ2(comm,PETSC_ERR_ARG_OUTOFRANGE,"selected root indice %D is out of [0,%D)",selected[i],nroots);
  }

  if (sf->ops->CreateEmbeddedSF) {
    ierr = (*sf->ops->CreateEmbeddedSF)(sf,nselected,selected,esf);CHKERRQ(ierr);
  } else {
    /* A generic version of creating embedded sf */
    ierr = PetscSFGetLeafRange(sf,&minleaf,&maxleaf);CHKERRQ(ierr);
    maxlocal = maxleaf - minleaf + 1;
    ierr = PetscCalloc2(nroots,&rootdata,maxlocal,&leafmem);CHKERRQ(ierr);
    leafdata = leafmem - minleaf;
    /* Tag selected roots and bcast to leaves */
    for (i=0; i<nselected; i++) rootdata[selected[i]] = 1;
    ierr = PetscSFBcastBegin(sf,MPI_SIGNED_CHAR,rootdata,leafdata);CHKERRQ(ierr);
    ierr = PetscSFBcastEnd(sf,MPI_SIGNED_CHAR,rootdata,leafdata);CHKERRQ(ierr);

    /* Build esf with leaves that are still connected */
    esf_nleaves = 0;
    for (i=0; i<nleaves; i++) {
      j = ilocal ? ilocal[i] : i;
      /* esf_nleaves += leafdata[j] should work in theory, but failed with SFWindow bugs
         with PetscSFBcast. See https://gitlab.com/petsc/petsc/issues/555
      */
      esf_nleaves += (leafdata[j] ? 1 : 0);
    }
    ierr = PetscMalloc1(esf_nleaves,&new_ilocal);CHKERRQ(ierr);
    ierr = PetscMalloc1(esf_nleaves,&new_iremote);CHKERRQ(ierr);
    for (i=n=0; i<nleaves; i++) {
      j = ilocal ? ilocal[i] : i;
      if (leafdata[j]) {
        new_ilocal[n]        = j;
        new_iremote[n].rank  = iremote[i].rank;
        new_iremote[n].index = iremote[i].index;
        ++n;
      }
    }
    ierr = PetscSFCreate(comm,esf);CHKERRQ(ierr);
    ierr = PetscSFSetFromOptions(*esf);CHKERRQ(ierr);
    ierr = PetscSFSetGraph(*esf,nroots,esf_nleaves,new_ilocal,PETSC_OWN_POINTER,new_iremote,PETSC_OWN_POINTER);CHKERRQ(ierr);
    ierr = PetscFree2(rootdata,leafmem);CHKERRQ(ierr);
  }
  ierr = PetscLogEventEnd(PETSCSF_EmbedSF,sf,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  PetscSFCreateEmbeddedLeafSF - removes edges from all but the selected leaves, does not remap indices

  Collective

  Input Arguments:
+ sf - original star forest
. nselected  - number of selected leaves on this process
- selected   - indices of the selected leaves on this process

  Output Arguments:
.  newsf - new star forest

  Level: advanced

.seealso: PetscSFCreateEmbeddedSF(), PetscSFSetGraph(), PetscSFGetGraph()
@*/
PetscErrorCode PetscSFCreateEmbeddedLeafSF(PetscSF sf,PetscInt nselected,const PetscInt *selected,PetscSF *newsf)
{
  const PetscSFNode *iremote;
  PetscSFNode       *new_iremote;
  const PetscInt    *ilocal;
  PetscInt          i,nroots,*leaves,*new_ilocal;
  MPI_Comm          comm;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sf,PETSCSF_CLASSID,1);
  PetscSFCheckGraphSet(sf,1);
  if (nselected) PetscValidPointer(selected,3);
  PetscValidPointer(newsf,4);

  /* Uniq selected[] and put results in leaves[] */
  ierr = PetscObjectGetComm((PetscObject)sf,&comm);CHKERRQ(ierr);
  ierr = PetscMalloc1(nselected,&leaves);CHKERRQ(ierr);
  ierr = PetscArraycpy(leaves,selected,nselected);CHKERRQ(ierr);
  ierr = PetscSortedRemoveDupsInt(&nselected,leaves);CHKERRQ(ierr);
  if (nselected && (leaves[0] < 0 || leaves[nselected-1] >= sf->nleaves)) SETERRQ3(comm,PETSC_ERR_ARG_OUTOFRANGE,"Min/Max leaf indices %D/%D are not in [0,%D)",leaves[0],leaves[nselected-1],sf->nleaves);

  /* Optimize the routine only when sf is setup and hence we can reuse sf's communication pattern */
  if (sf->setupcalled && sf->ops->CreateEmbeddedLeafSF) {
    ierr = (*sf->ops->CreateEmbeddedLeafSF)(sf,nselected,leaves,newsf);CHKERRQ(ierr);
  } else {
    ierr = PetscSFGetGraph(sf,&nroots,NULL,&ilocal,&iremote);CHKERRQ(ierr);
    ierr = PetscMalloc1(nselected,&new_ilocal);CHKERRQ(ierr);
    ierr = PetscMalloc1(nselected,&new_iremote);CHKERRQ(ierr);
    for (i=0; i<nselected; ++i) {
      const PetscInt l     = leaves[i];
      new_ilocal[i]        = ilocal ? ilocal[l] : l;
      new_iremote[i].rank  = iremote[l].rank;
      new_iremote[i].index = iremote[l].index;
    }
    ierr = PetscSFDuplicate(sf,PETSCSF_DUPLICATE_CONFONLY,newsf);CHKERRQ(ierr);
    ierr = PetscSFSetGraph(*newsf,nroots,nselected,new_ilocal,PETSC_OWN_POINTER,new_iremote,PETSC_OWN_POINTER);CHKERRQ(ierr);
  }
  ierr = PetscFree(leaves);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   PetscSFBcastAndOpBegin - begin pointwise broadcast with root value being reduced to leaf value, to be concluded with call to PetscSFBcastAndOpEnd()

   Collective on PetscSF

   Input Arguments:
+  sf - star forest on which to communicate
.  unit - data type associated with each node
.  rootdata - buffer to broadcast
-  op - operation to use for reduction

   Output Arguments:
.  leafdata - buffer to be reduced with values from each leaf's respective root

   Level: intermediate

.seealso: PetscSFBcastAndOpEnd(), PetscSFBcastBegin(), PetscSFBcastEnd()
@*/
PetscErrorCode PetscSFBcastAndOpBegin(PetscSF sf,MPI_Datatype unit,const void *rootdata,void *leafdata,MPI_Op op)
{
  PetscErrorCode ierr;
  PetscMemType   rootmtype,leafmtype;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sf,PETSCSF_CLASSID,1);
  ierr = PetscSFSetUp(sf);CHKERRQ(ierr);
  ierr = PetscLogEventBegin(PETSCSF_BcastAndOpBegin,sf,0,0,0);CHKERRQ(ierr);
  ierr = PetscGetMemType(rootdata,&rootmtype);CHKERRQ(ierr);
  ierr = PetscGetMemType(leafdata,&leafmtype);CHKERRQ(ierr);
  ierr = (*sf->ops->BcastAndOpBegin)(sf,unit,rootmtype,rootdata,leafmtype,leafdata,op);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(PETSCSF_BcastAndOpBegin,sf,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   PetscSFBcastAndOpEnd - end a broadcast & reduce operation started with PetscSFBcastAndOpBegin()

   Collective

   Input Arguments:
+  sf - star forest
.  unit - data type
.  rootdata - buffer to broadcast
-  op - operation to use for reduction

   Output Arguments:
.  leafdata - buffer to be reduced with values from each leaf's respective root

   Level: intermediate

.seealso: PetscSFSetGraph(), PetscSFReduceEnd()
@*/
PetscErrorCode PetscSFBcastAndOpEnd(PetscSF sf,MPI_Datatype unit,const void *rootdata,void *leafdata,MPI_Op op)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sf,PETSCSF_CLASSID,1);
  ierr = PetscLogEventBegin(PETSCSF_BcastAndOpEnd,sf,0,0,0);CHKERRQ(ierr);
  ierr = (*sf->ops->BcastAndOpEnd)(sf,unit,rootdata,leafdata,op);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(PETSCSF_BcastAndOpEnd,sf,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   PetscSFReduceBegin - begin reduction of leafdata into rootdata, to be completed with call to PetscSFReduceEnd()

   Collective

   Input Arguments:
+  sf - star forest
.  unit - data type
.  leafdata - values to reduce
-  op - reduction operation

   Output Arguments:
.  rootdata - result of reduction of values from all leaves of each root

   Level: intermediate

.seealso: PetscSFBcastBegin()
@*/
PetscErrorCode PetscSFReduceBegin(PetscSF sf,MPI_Datatype unit,const void *leafdata,void *rootdata,MPI_Op op)
{
  PetscErrorCode ierr;
  PetscMemType   rootmtype,leafmtype;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sf,PETSCSF_CLASSID,1);
  ierr = PetscSFSetUp(sf);CHKERRQ(ierr);
  ierr = PetscLogEventBegin(PETSCSF_ReduceBegin,sf,0,0,0);CHKERRQ(ierr);
  ierr = PetscGetMemType(rootdata,&rootmtype);CHKERRQ(ierr);
  ierr = PetscGetMemType(leafdata,&leafmtype);CHKERRQ(ierr);
  ierr = (sf->ops->ReduceBegin)(sf,unit,leafmtype,leafdata,rootmtype,rootdata,op);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(PETSCSF_ReduceBegin,sf,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   PetscSFReduceEnd - end a reduction operation started with PetscSFReduceBegin()

   Collective

   Input Arguments:
+  sf - star forest
.  unit - data type
.  leafdata - values to reduce
-  op - reduction operation

   Output Arguments:
.  rootdata - result of reduction of values from all leaves of each root

   Level: intermediate

.seealso: PetscSFSetGraph(), PetscSFBcastEnd()
@*/
PetscErrorCode PetscSFReduceEnd(PetscSF sf,MPI_Datatype unit,const void *leafdata,void *rootdata,MPI_Op op)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sf,PETSCSF_CLASSID,1);
  ierr = PetscLogEventBegin(PETSCSF_ReduceEnd,sf,0,0,0);CHKERRQ(ierr);
  ierr = (*sf->ops->ReduceEnd)(sf,unit,leafdata,rootdata,op);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(PETSCSF_ReduceEnd,sf,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   PetscSFFetchAndOpBegin - begin operation that fetches values from root and updates atomically by applying operation using my leaf value, to be completed with PetscSFFetchAndOpEnd()

   Collective

   Input Arguments:
+  sf - star forest
.  unit - data type
.  leafdata - leaf values to use in reduction
-  op - operation to use for reduction

   Output Arguments:
+  rootdata - root values to be updated, input state is seen by first process to perform an update
-  leafupdate - state at each leaf's respective root immediately prior to my atomic update

   Level: advanced

   Note:
   The update is only atomic at the granularity provided by the hardware. Different roots referenced by the same process
   might be updated in a different order. Furthermore, if a composite type is used for the unit datatype, atomicity is
   not guaranteed across the whole vertex. Therefore, this function is mostly only used with primitive types such as
   integers.

.seealso: PetscSFComputeDegreeBegin(), PetscSFReduceBegin(), PetscSFSetGraph()
@*/
PetscErrorCode PetscSFFetchAndOpBegin(PetscSF sf,MPI_Datatype unit,void *rootdata,const void *leafdata,void *leafupdate,MPI_Op op)
{
  PetscErrorCode ierr;
  PetscMemType   rootmtype,leafmtype,leafupdatemtype;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sf,PETSCSF_CLASSID,1);
  ierr = PetscSFSetUp(sf);CHKERRQ(ierr);
  ierr = PetscLogEventBegin(PETSCSF_FetchAndOpBegin,sf,0,0,0);CHKERRQ(ierr);
  ierr = PetscGetMemType(rootdata,&rootmtype);CHKERRQ(ierr);
  ierr = PetscGetMemType(leafdata,&leafmtype);CHKERRQ(ierr);
  ierr = PetscGetMemType(leafupdate,&leafupdatemtype);CHKERRQ(ierr);
  if (leafmtype != leafupdatemtype) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"No support for leafdata and leafupdate in different memory types");
  ierr = (*sf->ops->FetchAndOpBegin)(sf,unit,rootmtype,rootdata,leafmtype,leafdata,leafupdate,op);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(PETSCSF_FetchAndOpBegin,sf,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   PetscSFFetchAndOpEnd - end operation started in matching call to PetscSFFetchAndOpBegin() to fetch values from roots and update atomically by applying operation using my leaf value

   Collective

   Input Arguments:
+  sf - star forest
.  unit - data type
.  leafdata - leaf values to use in reduction
-  op - operation to use for reduction

   Output Arguments:
+  rootdata - root values to be updated, input state is seen by first process to perform an update
-  leafupdate - state at each leaf's respective root immediately prior to my atomic update

   Level: advanced

.seealso: PetscSFComputeDegreeEnd(), PetscSFReduceEnd(), PetscSFSetGraph()
@*/
PetscErrorCode PetscSFFetchAndOpEnd(PetscSF sf,MPI_Datatype unit,void *rootdata,const void *leafdata,void *leafupdate,MPI_Op op)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sf,PETSCSF_CLASSID,1);
  ierr = PetscLogEventBegin(PETSCSF_FetchAndOpEnd,sf,0,0,0);CHKERRQ(ierr);
  ierr = (*sf->ops->FetchAndOpEnd)(sf,unit,rootdata,leafdata,leafupdate,op);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(PETSCSF_FetchAndOpEnd,sf,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   PetscSFComputeDegreeBegin - begin computation of degree for each root vertex, to be completed with PetscSFComputeDegreeEnd()

   Collective

   Input Arguments:
.  sf - star forest

   Output Arguments:
.  degree - degree of each root vertex

   Level: advanced

   Notes:
   The returned array is owned by PetscSF and automatically freed by PetscSFDestroy(). Hence no need to call PetscFree() on it.

.seealso: PetscSFGatherBegin()
@*/
PetscErrorCode PetscSFComputeDegreeBegin(PetscSF sf,const PetscInt **degree)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sf,PETSCSF_CLASSID,1);
  PetscSFCheckGraphSet(sf,1);
  PetscValidPointer(degree,2);
  if (!sf->degreeknown) {
    PetscInt i, nroots = sf->nroots, maxlocal;
    if (sf->degree) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Calls to PetscSFComputeDegreeBegin() cannot be nested.");
    maxlocal = sf->maxleaf-sf->minleaf+1;
    ierr = PetscMalloc1(nroots,&sf->degree);CHKERRQ(ierr);
    ierr = PetscMalloc1(PetscMax(maxlocal,1),&sf->degreetmp);CHKERRQ(ierr); /* allocate at least one entry, see check in PetscSFComputeDegreeEnd() */
    for (i=0; i<nroots; i++) sf->degree[i] = 0;
    for (i=0; i<maxlocal; i++) sf->degreetmp[i] = 1;
    ierr = PetscSFReduceBegin(sf,MPIU_INT,sf->degreetmp-sf->minleaf,sf->degree,MPI_SUM);CHKERRQ(ierr);
  }
  *degree = NULL;
  PetscFunctionReturn(0);
}

/*@C
   PetscSFComputeDegreeEnd - complete computation of degree for each root vertex, started with PetscSFComputeDegreeBegin()

   Collective

   Input Arguments:
.  sf - star forest

   Output Arguments:
.  degree - degree of each root vertex

   Level: developer

   Notes:
   The returned array is owned by PetscSF and automatically freed by PetscSFDestroy(). Hence no need to call PetscFree() on it.

.seealso:
@*/
PetscErrorCode PetscSFComputeDegreeEnd(PetscSF sf,const PetscInt **degree)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sf,PETSCSF_CLASSID,1);
  PetscSFCheckGraphSet(sf,1);
  PetscValidPointer(degree,2);
  if (!sf->degreeknown) {
    if (!sf->degreetmp) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Must call PetscSFComputeDegreeBegin() before PetscSFComputeDegreeEnd()");
    ierr = PetscSFReduceEnd(sf,MPIU_INT,sf->degreetmp-sf->minleaf,sf->degree,MPI_SUM);CHKERRQ(ierr);
    ierr = PetscFree(sf->degreetmp);CHKERRQ(ierr);
    sf->degreeknown = PETSC_TRUE;
  }
  *degree = sf->degree;
  PetscFunctionReturn(0);
}


/*@C
   PetscSFComputeMultiRootOriginalNumbering - Returns original numbering of multi-roots (roots of multi-SF returned by PetscSFGetMultiSF()).
   Each multi-root is assigned index of the corresponding original root.

   Collective

   Input Arguments:
+  sf - star forest
-  degree - degree of each root vertex, computed with PetscSFComputeDegreeBegin()/PetscSFComputeDegreeEnd()

   Output Arguments:
+  nMultiRoots - (optional) number of multi-roots (roots of multi-SF)
-  multiRootsOrigNumbering - original indices of multi-roots; length of this array is nMultiRoots

   Level: developer

   Notes:
   The returned array multiRootsOrigNumbering is newly allocated and should be destroyed with PetscFree() when no longer needed.

.seealso: PetscSFComputeDegreeBegin(), PetscSFComputeDegreeEnd(), PetscSFGetMultiSF()
@*/
PetscErrorCode PetscSFComputeMultiRootOriginalNumbering(PetscSF sf, const PetscInt degree[], PetscInt *nMultiRoots, PetscInt *multiRootsOrigNumbering[])
{
  PetscSF             msf;
  PetscInt            i, j, k, nroots, nmroots;
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sf,PETSCSF_CLASSID,1);
  ierr = PetscSFGetGraph(sf, &nroots, NULL, NULL, NULL);CHKERRQ(ierr);
  if (nroots) PetscValidIntPointer(degree,2);
  if (nMultiRoots) PetscValidIntPointer(nMultiRoots,3);
  PetscValidPointer(multiRootsOrigNumbering,4);
  ierr = PetscSFGetMultiSF(sf,&msf);CHKERRQ(ierr);
  ierr = PetscSFGetGraph(msf, &nmroots, NULL, NULL, NULL);CHKERRQ(ierr);
  ierr = PetscMalloc1(nmroots, multiRootsOrigNumbering);CHKERRQ(ierr);
  for (i=0,j=0,k=0; i<nroots; i++) {
    if (!degree[i]) continue;
    for (j=0; j<degree[i]; j++,k++) {
      (*multiRootsOrigNumbering)[k] = i;
    }
  }
  if (PetscUnlikely(k != nmroots)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"sanity check fail");
  if (nMultiRoots) *nMultiRoots = nmroots;
  PetscFunctionReturn(0);
}

/*@C
   PetscSFGatherBegin - begin pointwise gather of all leaves into multi-roots, to be completed with PetscSFGatherEnd()

   Collective

   Input Arguments:
+  sf - star forest
.  unit - data type
-  leafdata - leaf data to gather to roots

   Output Argument:
.  multirootdata - root buffer to gather into, amount of space per root is equal to its degree

   Level: intermediate

.seealso: PetscSFComputeDegreeBegin(), PetscSFScatterBegin()
@*/
PetscErrorCode PetscSFGatherBegin(PetscSF sf,MPI_Datatype unit,const void *leafdata,void *multirootdata)
{
  PetscErrorCode ierr;
  PetscSF        multi = NULL;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sf,PETSCSF_CLASSID,1);
  ierr = PetscSFSetUp(sf);CHKERRQ(ierr);
  ierr = PetscSFGetMultiSF(sf,&multi);CHKERRQ(ierr);
  ierr = PetscSFReduceBegin(multi,unit,leafdata,multirootdata,MPIU_REPLACE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   PetscSFGatherEnd - ends pointwise gather operation that was started with PetscSFGatherBegin()

   Collective

   Input Arguments:
+  sf - star forest
.  unit - data type
-  leafdata - leaf data to gather to roots

   Output Argument:
.  multirootdata - root buffer to gather into, amount of space per root is equal to its degree

   Level: intermediate

.seealso: PetscSFComputeDegreeEnd(), PetscSFScatterEnd()
@*/
PetscErrorCode PetscSFGatherEnd(PetscSF sf,MPI_Datatype unit,const void *leafdata,void *multirootdata)
{
  PetscErrorCode ierr;
  PetscSF        multi = NULL;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sf,PETSCSF_CLASSID,1);
  ierr = PetscSFGetMultiSF(sf,&multi);CHKERRQ(ierr);
  ierr = PetscSFReduceEnd(multi,unit,leafdata,multirootdata,MPIU_REPLACE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   PetscSFScatterBegin - begin pointwise scatter operation from multi-roots to leaves, to be completed with PetscSFScatterEnd()

   Collective

   Input Arguments:
+  sf - star forest
.  unit - data type
-  multirootdata - root buffer to send to each leaf, one unit of data per leaf

   Output Argument:
.  leafdata - leaf data to be update with personal data from each respective root

   Level: intermediate

.seealso: PetscSFComputeDegreeBegin(), PetscSFScatterBegin()
@*/
PetscErrorCode PetscSFScatterBegin(PetscSF sf,MPI_Datatype unit,const void *multirootdata,void *leafdata)
{
  PetscErrorCode ierr;
  PetscSF        multi = NULL;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sf,PETSCSF_CLASSID,1);
  ierr = PetscSFSetUp(sf);CHKERRQ(ierr);
  ierr = PetscSFGetMultiSF(sf,&multi);CHKERRQ(ierr);
  ierr = PetscSFBcastBegin(multi,unit,multirootdata,leafdata);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   PetscSFScatterEnd - ends pointwise scatter operation that was started with PetscSFScatterBegin()

   Collective

   Input Arguments:
+  sf - star forest
.  unit - data type
-  multirootdata - root buffer to send to each leaf, one unit of data per leaf

   Output Argument:
.  leafdata - leaf data to be update with personal data from each respective root

   Level: intermediate

.seealso: PetscSFComputeDegreeEnd(), PetscSFScatterEnd()
@*/
PetscErrorCode PetscSFScatterEnd(PetscSF sf,MPI_Datatype unit,const void *multirootdata,void *leafdata)
{
  PetscErrorCode ierr;
  PetscSF        multi = NULL;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sf,PETSCSF_CLASSID,1);
  ierr = PetscSFGetMultiSF(sf,&multi);CHKERRQ(ierr);
  ierr = PetscSFBcastEnd(multi,unit,multirootdata,leafdata);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSFCheckLeavesUnique_Private(PetscSF sf)
{
  PetscInt        i, n, nleaves;
  const PetscInt *ilocal = NULL;
  PetscHSetI      seen;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  if (!PetscDefined(USE_DEBUG)) PetscFunctionReturn(0);
  ierr = PetscSFGetGraph(sf,NULL,&nleaves,&ilocal,NULL);CHKERRQ(ierr);
  ierr = PetscHSetICreate(&seen);CHKERRQ(ierr);
  for (i = 0; i < nleaves; i++) {
    const PetscInt leaf = ilocal ? ilocal[i] : i;
    ierr = PetscHSetIAdd(seen,leaf);CHKERRQ(ierr);
  }
  ierr = PetscHSetIGetSize(seen,&n);CHKERRQ(ierr);
  if (n != nleaves) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Provided leaves have repeated values: all leaves must be unique");
  ierr = PetscHSetIDestroy(&seen);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  PetscSFCompose - Compose a new PetscSF by putting the second SF under the first one in a top (roots) down (leaves) view

  Input Parameters:
+ sfA - The first PetscSF
- sfB - The second PetscSF

  Output Parameters:
. sfBA - The composite SF

  Level: developer

  Notes:
  Currently, the two SFs must be defined on congruent communicators and they must be true star
  forests, i.e. the same leaf is not connected with different roots.

  sfA's leaf space and sfB's root space might be partially overlapped. The composition builds
  a graph with sfA's roots and sfB's leaves only when there is a path between them. Unconnected
  nodes (roots or leaves) are not in sfBA. Doing a Bcast on the new SF is equivalent to doing a
  Bcast on sfA, then a Bcast on sfB, on connected nodes.

.seealso: PetscSF, PetscSFComposeInverse(), PetscSFGetGraph(), PetscSFSetGraph()
@*/
PetscErrorCode PetscSFCompose(PetscSF sfA,PetscSF sfB,PetscSF *sfBA)
{
  PetscErrorCode    ierr;
  const PetscSFNode *remotePointsA,*remotePointsB;
  PetscSFNode       *remotePointsBA=NULL,*reorderedRemotePointsA = NULL,*leafdataB;
  const PetscInt    *localPointsA,*localPointsB;
  PetscInt          *localPointsBA;
  PetscInt          i,numRootsA,numLeavesA,numRootsB,numLeavesB,minleaf,maxleaf,numLeavesBA;
  PetscBool         denseB;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sfA,PETSCSF_CLASSID,1);
  PetscSFCheckGraphSet(sfA,1);
  PetscValidHeaderSpecific(sfB,PETSCSF_CLASSID,2);
  PetscSFCheckGraphSet(sfB,2);
  PetscCheckSameComm(sfA,1,sfB,2);
  PetscValidPointer(sfBA,3);
  ierr = PetscSFCheckLeavesUnique_Private(sfA);CHKERRQ(ierr);
  ierr = PetscSFCheckLeavesUnique_Private(sfB);CHKERRQ(ierr);

  ierr = PetscSFGetGraph(sfA,&numRootsA,&numLeavesA,&localPointsA,&remotePointsA);CHKERRQ(ierr);
  ierr = PetscSFGetGraph(sfB,&numRootsB,&numLeavesB,&localPointsB,&remotePointsB);CHKERRQ(ierr);
  if (localPointsA) {
    ierr = PetscMalloc1(numRootsB,&reorderedRemotePointsA);CHKERRQ(ierr);
    for (i=0; i<numRootsB; i++) {
      reorderedRemotePointsA[i].rank = -1;
      reorderedRemotePointsA[i].index = -1;
    }
    for (i=0; i<numLeavesA; i++) {
      if (localPointsA[i] >= numRootsB) continue;
      reorderedRemotePointsA[localPointsA[i]] = remotePointsA[i];
    }
    remotePointsA = reorderedRemotePointsA;
  }
  ierr = PetscSFGetLeafRange(sfB,&minleaf,&maxleaf);CHKERRQ(ierr);
  ierr = PetscMalloc1(maxleaf-minleaf+1,&leafdataB);CHKERRQ(ierr);
  ierr = PetscSFBcastBegin(sfB,MPIU_2INT,remotePointsA,leafdataB-minleaf);CHKERRQ(ierr);
  ierr = PetscSFBcastEnd(sfB,MPIU_2INT,remotePointsA,leafdataB-minleaf);CHKERRQ(ierr);
  ierr = PetscFree(reorderedRemotePointsA);CHKERRQ(ierr);

  denseB = (PetscBool)!localPointsB;
  for (i=0,numLeavesBA=0; i<numLeavesB; i++) {
    if (leafdataB[localPointsB ? localPointsB[i]-minleaf : i].rank == -1) denseB = PETSC_FALSE;
    else numLeavesBA++;
  }
  if (denseB) {
    localPointsBA  = NULL;
    remotePointsBA = leafdataB;
  } else {
    ierr = PetscMalloc1(numLeavesBA,&localPointsBA);CHKERRQ(ierr);
    ierr = PetscMalloc1(numLeavesBA,&remotePointsBA);CHKERRQ(ierr);
    for (i=0,numLeavesBA=0; i<numLeavesB; i++) {
      const PetscInt l = localPointsB ? localPointsB[i] : i;

      if (leafdataB[l-minleaf].rank == -1) continue;
      remotePointsBA[numLeavesBA] = leafdataB[l-minleaf];
      localPointsBA[numLeavesBA] = l;
      numLeavesBA++;
    }
    ierr = PetscFree(leafdataB);CHKERRQ(ierr);
  }
  ierr = PetscSFCreate(PetscObjectComm((PetscObject)sfA),sfBA);CHKERRQ(ierr);
  ierr = PetscSFSetFromOptions(*sfBA);CHKERRQ(ierr);
  ierr = PetscSFSetGraph(*sfBA,numRootsA,numLeavesBA,localPointsBA,PETSC_OWN_POINTER,remotePointsBA,PETSC_OWN_POINTER);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  PetscSFComposeInverse - Compose a new PetscSF by putting the inverse of the second SF under the first one

  Input Parameters:
+ sfA - The first PetscSF
- sfB - The second PetscSF

  Output Parameters:
. sfBA - The composite SF.

  Level: developer

  Notes:
  Currently, the two SFs must be defined on congruent communicators and they must be true star
  forests, i.e. the same leaf is not connected with different roots. Even more, all roots of the
  second SF must have a degree of 1, i.e., no roots have more than one leaf connected.

  sfA's leaf space and sfB's leaf space might be partially overlapped. The composition builds
  a graph with sfA's roots and sfB's roots only when there is a path between them. Unconnected
  roots are not in sfBA. Doing a Bcast on the new SF is equivalent to doing a Bcast on sfA, then
  a Reduce on sfB, on connected roots.

.seealso: PetscSF, PetscSFCompose(), PetscSFGetGraph(), PetscSFSetGraph(), PetscSFCreateInverseSF()
@*/
PetscErrorCode PetscSFComposeInverse(PetscSF sfA,PetscSF sfB,PetscSF *sfBA)
{
  PetscErrorCode    ierr;
  const PetscSFNode *remotePointsA,*remotePointsB;
  PetscSFNode       *remotePointsBA;
  const PetscInt    *localPointsA,*localPointsB;
  PetscSFNode       *reorderedRemotePointsA = NULL;
  PetscInt          i,numRootsA,numLeavesA,numLeavesBA,numRootsB,numLeavesB,minleaf,maxleaf,*localPointsBA;
  MPI_Op            op;
#if defined(PETSC_USE_64BIT_INDICES)
  PetscBool         iswin;
#endif

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sfA,PETSCSF_CLASSID,1);
  PetscSFCheckGraphSet(sfA,1);
  PetscValidHeaderSpecific(sfB,PETSCSF_CLASSID,2);
  PetscSFCheckGraphSet(sfB,2);
  PetscCheckSameComm(sfA,1,sfB,2);
  PetscValidPointer(sfBA,3);
  ierr = PetscSFCheckLeavesUnique_Private(sfA);CHKERRQ(ierr);
  ierr = PetscSFCheckLeavesUnique_Private(sfB);CHKERRQ(ierr);

  ierr = PetscSFGetGraph(sfA, &numRootsA, &numLeavesA, &localPointsA, &remotePointsA);CHKERRQ(ierr);
  ierr = PetscSFGetGraph(sfB, &numRootsB, &numLeavesB, &localPointsB, &remotePointsB);CHKERRQ(ierr);

  /* TODO: Check roots of sfB have degree of 1 */
  /* Once we implement it, we can replace the MPI_MAXLOC
     with MPIU_REPLACE. In that case, MPI_MAXLOC and MPIU_REPLACE have the same effect.
     We use MPI_MAXLOC only to have a deterministic output from this routine if
     the root condition is not meet.
   */
  op = MPI_MAXLOC;
#if defined(PETSC_USE_64BIT_INDICES)
  /* we accept a non-deterministic output (if any) with PETSCSFWINDOW, since MPI_MAXLOC cannot operate on MPIU_2INT with MPI_Accumulate */
  ierr = PetscObjectTypeCompare((PetscObject)sfB,PETSCSFWINDOW,&iswin);CHKERRQ(ierr);
  if (iswin) op = MPIU_REPLACE;
#endif

  ierr = PetscSFGetLeafRange(sfB, &minleaf, &maxleaf);CHKERRQ(ierr);
  ierr = PetscMalloc1(maxleaf - minleaf + 1,&reorderedRemotePointsA);CHKERRQ(ierr);
  for (i=0; i<maxleaf - minleaf + 1; i++) {
    reorderedRemotePointsA[i].rank = -1;
    reorderedRemotePointsA[i].index = -1;
  }
  if (localPointsA) {
    for (i=0; i<numLeavesA; i++) {
      if (localPointsA[i] > maxleaf || localPointsA[i] < minleaf) continue;
      reorderedRemotePointsA[localPointsA[i] - minleaf] = remotePointsA[i];
    }
  } else {
    for (i=0; i<numLeavesA; i++) {
      if (i > maxleaf || i < minleaf) continue;
      reorderedRemotePointsA[i - minleaf] = remotePointsA[i];
    }
  }

  ierr = PetscMalloc1(numRootsB,&localPointsBA);CHKERRQ(ierr);
  ierr = PetscMalloc1(numRootsB,&remotePointsBA);CHKERRQ(ierr);
  for (i=0; i<numRootsB; i++) {
    remotePointsBA[i].rank = -1;
    remotePointsBA[i].index = -1;
  }

  ierr = PetscSFReduceBegin(sfB,MPIU_2INT,reorderedRemotePointsA-minleaf,remotePointsBA,op);CHKERRQ(ierr);
  ierr = PetscSFReduceEnd(sfB,MPIU_2INT,reorderedRemotePointsA-minleaf,remotePointsBA,op);CHKERRQ(ierr);
  ierr = PetscFree(reorderedRemotePointsA);CHKERRQ(ierr);
  for (i=0,numLeavesBA=0; i<numRootsB; i++) {
    if (remotePointsBA[i].rank == -1) continue;
    remotePointsBA[numLeavesBA].rank = remotePointsBA[i].rank;
    remotePointsBA[numLeavesBA].index = remotePointsBA[i].index;
    localPointsBA[numLeavesBA] = i;
    numLeavesBA++;
  }
  ierr = PetscSFCreate(PetscObjectComm((PetscObject)sfA),sfBA);CHKERRQ(ierr);
  ierr = PetscSFSetFromOptions(*sfBA);CHKERRQ(ierr);
  ierr = PetscSFSetGraph(*sfBA,numRootsA,numLeavesBA,localPointsBA,PETSC_OWN_POINTER,remotePointsBA,PETSC_OWN_POINTER);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
  PetscSFCreateLocalSF_Private - Creates a local PetscSF that only has intra-process edges of the global PetscSF

  Input Parameters:
. sf - The global PetscSF

  Output Parameters:
. out - The local PetscSF
 */
PetscErrorCode PetscSFCreateLocalSF_Private(PetscSF sf,PetscSF *out)
{
  MPI_Comm           comm;
  PetscMPIInt        myrank;
  const PetscInt     *ilocal;
  const PetscSFNode  *iremote;
  PetscInt           i,j,nroots,nleaves,lnleaves,*lilocal;
  PetscSFNode        *liremote;
  PetscSF            lsf;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sf,PETSCSF_CLASSID,1);
  if (sf->ops->CreateLocalSF) {
    ierr = (*sf->ops->CreateLocalSF)(sf,out);CHKERRQ(ierr);
  } else {
    /* Could use PetscSFCreateEmbeddedLeafSF, but since we know the comm is PETSC_COMM_SELF, we can make it fast */
    ierr = PetscObjectGetComm((PetscObject)sf,&comm);CHKERRQ(ierr);
    ierr = MPI_Comm_rank(comm,&myrank);CHKERRQ(ierr);

    /* Find out local edges and build a local SF */
    ierr = PetscSFGetGraph(sf,&nroots,&nleaves,&ilocal,&iremote);CHKERRQ(ierr);
    for (i=lnleaves=0; i<nleaves; i++) {if (iremote[i].rank == (PetscInt)myrank) lnleaves++;}
    ierr = PetscMalloc1(lnleaves,&lilocal);CHKERRQ(ierr);
    ierr = PetscMalloc1(lnleaves,&liremote);CHKERRQ(ierr);

    for (i=j=0; i<nleaves; i++) {
      if (iremote[i].rank == (PetscInt)myrank) {
        lilocal[j]        = ilocal? ilocal[i] : i; /* ilocal=NULL for contiguous storage */
        liremote[j].rank  = 0; /* rank in PETSC_COMM_SELF */
        liremote[j].index = iremote[i].index;
        j++;
      }
    }
    ierr = PetscSFCreate(PETSC_COMM_SELF,&lsf);CHKERRQ(ierr);
    ierr = PetscSFSetFromOptions(lsf);CHKERRQ(ierr);
    ierr = PetscSFSetGraph(lsf,nroots,lnleaves,lilocal,PETSC_OWN_POINTER,liremote,PETSC_OWN_POINTER);CHKERRQ(ierr);
    ierr = PetscSFSetUp(lsf);CHKERRQ(ierr);
    *out = lsf;
  }
  PetscFunctionReturn(0);
}

/* Similar to PetscSFBcast, but only Bcast to leaves on rank 0 */
PetscErrorCode PetscSFBcastToZero_Private(PetscSF sf,MPI_Datatype unit,const void *rootdata,void *leafdata)
{
  PetscErrorCode     ierr;
  PetscMemType       rootmtype,leafmtype;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sf,PETSCSF_CLASSID,1);
  ierr = PetscSFSetUp(sf);CHKERRQ(ierr);
  ierr = PetscLogEventBegin(PETSCSF_BcastAndOpBegin,sf,0,0,0);CHKERRQ(ierr);
  ierr = PetscGetMemType(rootdata,&rootmtype);CHKERRQ(ierr);
  ierr = PetscGetMemType(leafdata,&leafmtype);CHKERRQ(ierr);
  if (sf->ops->BcastToZero) {
    ierr = (*sf->ops->BcastToZero)(sf,unit,rootmtype,rootdata,leafmtype,leafdata);CHKERRQ(ierr);
  } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"PetscSFBcastToZero_Private is not supported on this SF type");
  ierr = PetscLogEventEnd(PETSCSF_BcastAndOpBegin,sf,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

