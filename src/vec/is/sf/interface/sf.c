#include <petsc/private/sfimpl.h> /*I "petscsf.h" I*/
#include <petsc/private/hashseti.h>
#include <petsc/private/viewerimpl.h>
#include <petsc/private/hashmapi.h>

#if defined(PETSC_HAVE_CUDA)
  #include <cuda_runtime.h>
  #include <petscdevice_cuda.h>
#endif

#if defined(PETSC_HAVE_HIP)
  #include <hip/hip_runtime.h>
#endif

#if defined(PETSC_CLANG_STATIC_ANALYZER)
extern void PetscSFCheckGraphSet(PetscSF, int);
#else
  #if defined(PETSC_USE_DEBUG)
    #define PetscSFCheckGraphSet(sf, arg) PetscCheck((sf)->graphset, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Must call PetscSFSetGraph() or PetscSFSetGraphWithPattern() on argument %d \"%s\" before %s()", (arg), #sf, PETSC_FUNCTION_NAME)
  #else
    #define PetscSFCheckGraphSet(sf, arg) \
      do { \
      } while (0)
  #endif
#endif

const char *const PetscSFDuplicateOptions[]     = {"CONFONLY", "RANKS", "GRAPH", "PetscSFDuplicateOption", "PETSCSF_DUPLICATE_", NULL};
const char *const PetscSFConcatenateRootModes[] = {"local", "shared", "global", "PetscSFConcatenateRootMode", "PETSCSF_CONCATENATE_ROOTMODE_", NULL};

/*@
  PetscSFCreate - create a star forest communication context

  Collective

  Input Parameter:
. comm - communicator on which the star forest will operate

  Output Parameter:
. sf - new star forest context

  Options Database Key:
+ -sf_type basic                 - Use MPI persistent Isend/Irecv for communication (Default)
. -sf_type window                - Use MPI-3 one-sided window for communication
. -sf_type neighbor              - Use MPI-3 neighborhood collectives for communication
- -sf_neighbor_persistent <bool> - If true, use MPI-4 persistent neighborhood collectives for communication (used along with -sf_type neighbor)

  Level: intermediate

  Note:
  When one knows the communication graph is one of the predefined graph, such as `MPI_Alltoall()`, `MPI_Allgatherv()`,
  `MPI_Gatherv()`, one can create a `PetscSF` and then set its graph with `PetscSFSetGraphWithPattern()`. These special
  `SF`s are optimized and they have better performance than the general `SF`s.

.seealso: `PetscSF`, `PetscSFSetType`, `PetscSFSetGraph()`, `PetscSFSetGraphWithPattern()`, `PetscSFDestroy()`
@*/
PetscErrorCode PetscSFCreate(MPI_Comm comm, PetscSF *sf)
{
  PetscSF b;

  PetscFunctionBegin;
  PetscAssertPointer(sf, 2);
  PetscCall(PetscSFInitializePackage());

  PetscCall(PetscHeaderCreate(b, PETSCSF_CLASSID, "PetscSF", "Star Forest", "PetscSF", comm, PetscSFDestroy, PetscSFView));
  b->nroots    = -1;
  b->nleaves   = -1;
  b->minleaf   = PETSC_INT_MAX;
  b->maxleaf   = PETSC_INT_MIN;
  b->nranks    = -1;
  b->rankorder = PETSC_TRUE;
  b->ingroup   = MPI_GROUP_NULL;
  b->outgroup  = MPI_GROUP_NULL;
  b->graphset  = PETSC_FALSE;
#if defined(PETSC_HAVE_DEVICE)
  b->use_gpu_aware_mpi    = use_gpu_aware_mpi;
  b->use_stream_aware_mpi = PETSC_FALSE;
  b->unknown_input_stream = PETSC_FALSE;
  #if defined(PETSC_HAVE_KOKKOS) /* Prefer kokkos over cuda*/
  b->backend = PETSCSF_BACKEND_KOKKOS;
  #elif defined(PETSC_HAVE_CUDA)
  b->backend = PETSCSF_BACKEND_CUDA;
  #elif defined(PETSC_HAVE_HIP)
  b->backend = PETSCSF_BACKEND_HIP;
  #endif

  #if defined(PETSC_HAVE_NVSHMEM)
  b->use_nvshmem     = PETSC_FALSE; /* Default is not to try NVSHMEM */
  b->use_nvshmem_get = PETSC_FALSE; /* Default is to use nvshmem_put based protocol */
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-use_nvshmem", &b->use_nvshmem, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-use_nvshmem_get", &b->use_nvshmem_get, NULL));
  #endif
#endif
  b->vscat.from_n = -1;
  b->vscat.to_n   = -1;
  b->vscat.unit   = MPIU_SCALAR;
  *sf             = b;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscSFReset - Reset a star forest so that different sizes or neighbors can be used

  Collective

  Input Parameter:
. sf - star forest

  Level: advanced

.seealso: `PetscSF`, `PetscSFCreate()`, `PetscSFSetGraph()`, `PetscSFDestroy()`
@*/
PetscErrorCode PetscSFReset(PetscSF sf)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(sf, PETSCSF_CLASSID, 1);
  PetscTryTypeMethod(sf, Reset);
  PetscCall(PetscSFDestroy(&sf->rankssf));

  sf->nroots   = -1;
  sf->nleaves  = -1;
  sf->minleaf  = PETSC_INT_MAX;
  sf->maxleaf  = PETSC_INT_MIN;
  sf->mine     = NULL;
  sf->remote   = NULL;
  sf->graphset = PETSC_FALSE;
  PetscCall(PetscFree(sf->mine_alloc));
  PetscCall(PetscFree(sf->remote_alloc));
  sf->nranks = -1;
  PetscCall(PetscFree4(sf->ranks, sf->roffset, sf->rmine, sf->rremote));
  sf->degreeknown = PETSC_FALSE;
  PetscCall(PetscFree(sf->degree));
  if (sf->ingroup != MPI_GROUP_NULL) PetscCallMPI(MPI_Group_free(&sf->ingroup));
  if (sf->outgroup != MPI_GROUP_NULL) PetscCallMPI(MPI_Group_free(&sf->outgroup));

  if (sf->multi) sf->multi->multi = NULL;
  PetscCall(PetscSFDestroy(&sf->multi));

  PetscCall(PetscLayoutDestroy(&sf->map));

#if defined(PETSC_HAVE_DEVICE)
  for (PetscInt i = 0; i < 2; i++) PetscCall(PetscSFFree(sf, PETSC_MEMTYPE_DEVICE, sf->rmine_d[i]));
#endif

  sf->setupcalled = PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscSFSetType - Set the `PetscSF` communication implementation

  Collective

  Input Parameters:
+ sf   - the `PetscSF` context
- type - a known method
.vb
    PETSCSFWINDOW - MPI-2/3 one-sided
    PETSCSFBASIC - basic implementation using MPI-1 two-sided
.ve

  Options Database Key:
. -sf_type <type> - Sets the method; for example `basic` or `window` use -help for a list of available methods

  Level: intermediate

  Notes:
  See `PetscSFType` for possible values

.seealso: `PetscSF`, `PetscSFType`, `PetscSFCreate()`
@*/
PetscErrorCode PetscSFSetType(PetscSF sf, PetscSFType type)
{
  PetscBool match;
  PetscErrorCode (*r)(PetscSF);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sf, PETSCSF_CLASSID, 1);
  PetscAssertPointer(type, 2);

  PetscCall(PetscObjectTypeCompare((PetscObject)sf, type, &match));
  if (match) PetscFunctionReturn(PETSC_SUCCESS);

  PetscCall(PetscFunctionListFind(PetscSFList, type, &r));
  PetscCheck(r, PetscObjectComm((PetscObject)sf), PETSC_ERR_ARG_UNKNOWN_TYPE, "Unable to find requested PetscSF type %s", type);
  /* Destroy the previous PetscSF implementation context */
  PetscTryTypeMethod(sf, Destroy);
  PetscCall(PetscMemzero(sf->ops, sizeof(*sf->ops)));
  PetscCall(PetscObjectChangeTypeName((PetscObject)sf, type));
  PetscCall((*r)(sf));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscSFGetType - Get the `PetscSF` communication implementation

  Not Collective

  Input Parameter:
. sf - the `PetscSF` context

  Output Parameter:
. type - the `PetscSF` type name

  Level: intermediate

.seealso: `PetscSF`, `PetscSFType`, `PetscSFSetType()`, `PetscSFCreate()`
@*/
PetscErrorCode PetscSFGetType(PetscSF sf, PetscSFType *type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(sf, PETSCSF_CLASSID, 1);
  PetscAssertPointer(type, 2);
  *type = ((PetscObject)sf)->type_name;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscSFDestroy - destroy a star forest

  Collective

  Input Parameter:
. sf - address of star forest

  Level: intermediate

.seealso: `PetscSF`, `PetscSFType`, `PetscSFCreate()`, `PetscSFReset()`
@*/
PetscErrorCode PetscSFDestroy(PetscSF *sf)
{
  PetscFunctionBegin;
  if (!*sf) PetscFunctionReturn(PETSC_SUCCESS);
  PetscValidHeaderSpecific(*sf, PETSCSF_CLASSID, 1);
  if (--((PetscObject)*sf)->refct > 0) {
    *sf = NULL;
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  PetscCall(PetscSFReset(*sf));
  PetscTryTypeMethod(*sf, Destroy);
  PetscCall(PetscSFDestroy(&(*sf)->vscat.lsf));
  if ((*sf)->vscat.bs > 1) PetscCallMPI(MPI_Type_free(&(*sf)->vscat.unit));
#if defined(PETSC_HAVE_CUDA) && defined(PETSC_HAVE_MPIX_STREAM)
  if ((*sf)->use_stream_aware_mpi) {
    PetscCallMPI(MPIX_Stream_free(&(*sf)->mpi_stream));
    PetscCallMPI(MPI_Comm_free(&(*sf)->stream_comm));
  }
#endif
  PetscCall(PetscHeaderDestroy(sf));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscSFCheckGraphValid_Private(PetscSF sf)
{
  PetscInt           i, nleaves;
  PetscMPIInt        size;
  const PetscInt    *ilocal;
  const PetscSFNode *iremote;

  PetscFunctionBegin;
  if (!sf->graphset || !PetscDefined(USE_DEBUG)) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(PetscSFGetGraph(sf, NULL, &nleaves, &ilocal, &iremote));
  PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)sf), &size));
  for (i = 0; i < nleaves; i++) {
    const PetscInt rank   = iremote[i].rank;
    const PetscInt remote = iremote[i].index;
    const PetscInt leaf   = ilocal ? ilocal[i] : i;
    PetscCheck(rank >= 0 && rank < size, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Provided rank (%" PetscInt_FMT ") for remote %" PetscInt_FMT " is invalid, should be in [0, %d)", rank, i, size);
    PetscCheck(remote >= 0, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Provided index (%" PetscInt_FMT ") for remote %" PetscInt_FMT " is invalid, should be >= 0", remote, i);
    PetscCheck(leaf >= 0, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Provided location (%" PetscInt_FMT ") for leaf %" PetscInt_FMT " is invalid, should be >= 0", leaf, i);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscSFSetUp - set up communication structures for a `PetscSF`, after this is done it may be used to perform communication

  Collective

  Input Parameter:
. sf - star forest communication object

  Level: beginner

.seealso: `PetscSF`, `PetscSFType`, `PetscSFSetFromOptions()`, `PetscSFSetType()`
@*/
PetscErrorCode PetscSFSetUp(PetscSF sf)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(sf, PETSCSF_CLASSID, 1);
  PetscSFCheckGraphSet(sf, 1);
  if (sf->setupcalled) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(PetscLogEventBegin(PETSCSF_SetUp, sf, 0, 0, 0));
  PetscCall(PetscSFCheckGraphValid_Private(sf));
  if (!((PetscObject)sf)->type_name) PetscCall(PetscSFSetType(sf, PETSCSFBASIC)); /* Zero all sf->ops */
  PetscTryTypeMethod(sf, SetUp);
#if defined(PETSC_HAVE_CUDA)
  if (sf->backend == PETSCSF_BACKEND_CUDA) {
    sf->ops->Malloc = PetscSFMalloc_CUDA;
    sf->ops->Free   = PetscSFFree_CUDA;
  }
#endif
#if defined(PETSC_HAVE_HIP)
  if (sf->backend == PETSCSF_BACKEND_HIP) {
    sf->ops->Malloc = PetscSFMalloc_HIP;
    sf->ops->Free   = PetscSFFree_HIP;
  }
#endif

#if defined(PETSC_HAVE_KOKKOS)
  if (sf->backend == PETSCSF_BACKEND_KOKKOS) {
    sf->ops->Malloc = PetscSFMalloc_Kokkos;
    sf->ops->Free   = PetscSFFree_Kokkos;
  }
#endif
  PetscCall(PetscLogEventEnd(PETSCSF_SetUp, sf, 0, 0, 0));
  sf->setupcalled = PETSC_TRUE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscSFSetFromOptions - set `PetscSF` options using the options database

  Logically Collective

  Input Parameter:
. sf - star forest

  Options Database Keys:
+ -sf_type                      - implementation type, see `PetscSFSetType()`
. -sf_rank_order                - sort composite points for gathers and scatters in rank order, gathers are non-deterministic otherwise
. -sf_use_default_stream        - Assume callers of `PetscSF` computed the input root/leafdata with the default CUDA stream. `PetscSF` will also
                                  use the default stream to process data. Therefore, no stream synchronization is needed between `PetscSF` and its caller (default: true).
                                  If true, this option only works with `-use_gpu_aware_mpi 1`.
. -sf_use_stream_aware_mpi      - Assume the underlying MPI is CUDA-stream aware and `PetscSF` won't sync streams for send/recv buffers passed to MPI (default: false).
                                  If true, this option only works with `-use_gpu_aware_mpi 1`.

- -sf_backend <cuda,hip,kokkos> - Select the device backend`PetscSF` uses. Currently `PetscSF` has these backends: cuda - hip and Kokkos.
                                  On CUDA (HIP) devices, one can choose cuda (hip) or kokkos with the default being kokkos. On other devices,
                                  the only available is kokkos.

  Level: intermediate

.seealso: `PetscSF`, `PetscSFCreate()`, `PetscSFSetType()`
@*/
PetscErrorCode PetscSFSetFromOptions(PetscSF sf)
{
  PetscSFType deft;
  char        type[256];
  PetscBool   flg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sf, PETSCSF_CLASSID, 1);
  PetscObjectOptionsBegin((PetscObject)sf);
  deft = ((PetscObject)sf)->type_name ? ((PetscObject)sf)->type_name : PETSCSFBASIC;
  PetscCall(PetscOptionsFList("-sf_type", "PetscSF implementation type", "PetscSFSetType", PetscSFList, deft, type, sizeof(type), &flg));
  PetscCall(PetscSFSetType(sf, flg ? type : deft));
  PetscCall(PetscOptionsBool("-sf_rank_order", "sort composite points for gathers and scatters in rank order, gathers are non-deterministic otherwise", "PetscSFSetRankOrder", sf->rankorder, &sf->rankorder, NULL));
#if defined(PETSC_HAVE_DEVICE)
  {
    char      backendstr[32] = {0};
    PetscBool isCuda = PETSC_FALSE, isHip = PETSC_FALSE, isKokkos = PETSC_FALSE, set;
    /* Change the defaults set in PetscSFCreate() with command line options */
    PetscCall(PetscOptionsBool("-sf_unknown_input_stream", "SF root/leafdata is computed on arbitrary streams unknown to SF", "PetscSFSetFromOptions", sf->unknown_input_stream, &sf->unknown_input_stream, NULL));
    PetscCall(PetscOptionsBool("-sf_use_stream_aware_mpi", "Assume the underlying MPI is cuda-stream aware", "PetscSFSetFromOptions", sf->use_stream_aware_mpi, &sf->use_stream_aware_mpi, NULL));
    PetscCall(PetscOptionsString("-sf_backend", "Select the device backend SF uses", "PetscSFSetFromOptions", NULL, backendstr, sizeof(backendstr), &set));
    PetscCall(PetscStrcasecmp("cuda", backendstr, &isCuda));
    PetscCall(PetscStrcasecmp("kokkos", backendstr, &isKokkos));
    PetscCall(PetscStrcasecmp("hip", backendstr, &isHip));
  #if defined(PETSC_HAVE_CUDA) || defined(PETSC_HAVE_HIP)
    if (isCuda) sf->backend = PETSCSF_BACKEND_CUDA;
    else if (isKokkos) sf->backend = PETSCSF_BACKEND_KOKKOS;
    else if (isHip) sf->backend = PETSCSF_BACKEND_HIP;
    else PetscCheck(!set, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "-sf_backend %s is not supported. You may choose cuda, hip or kokkos (if installed)", backendstr);
  #elif defined(PETSC_HAVE_KOKKOS)
    PetscCheck(!set || isKokkos, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "-sf_backend %s is not supported. You can only choose kokkos", backendstr);
  #endif

  #if defined(PETSC_HAVE_CUDA) && defined(PETSC_HAVE_MPIX_STREAM)
    if (sf->use_stream_aware_mpi) {
      MPI_Info info;

      PetscCallMPI(MPI_Info_create(&info));
      PetscCallMPI(MPI_Info_set(info, "type", "cudaStream_t"));
      PetscCallMPI(MPIX_Info_set_hex(info, "value", &PetscDefaultCudaStream, sizeof(PetscDefaultCudaStream)));
      PetscCallMPI(MPIX_Stream_create(info, &sf->mpi_stream));
      PetscCallMPI(MPI_Info_free(&info));
      PetscCallMPI(MPIX_Stream_comm_create(PetscObjectComm((PetscObject)sf), sf->mpi_stream, &sf->stream_comm));
    }
  #endif
  }
#endif
  PetscTryTypeMethod(sf, SetFromOptions, PetscOptionsObject);
  PetscOptionsEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscSFSetRankOrder - sort multi-points for gathers and scatters by rank order

  Logically Collective

  Input Parameters:
+ sf  - star forest
- flg - `PETSC_TRUE` to sort, `PETSC_FALSE` to skip sorting (lower setup cost, but non-deterministic)

  Level: advanced

.seealso: `PetscSF`, `PetscSFType`, `PetscSFGatherBegin()`, `PetscSFScatterBegin()`
@*/
PetscErrorCode PetscSFSetRankOrder(PetscSF sf, PetscBool flg)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(sf, PETSCSF_CLASSID, 1);
  PetscValidLogicalCollectiveBool(sf, flg, 2);
  PetscCheck(!sf->multi, PetscObjectComm((PetscObject)sf), PETSC_ERR_ARG_WRONGSTATE, "Rank ordering must be set before first call to PetscSFGatherBegin() or PetscSFScatterBegin()");
  sf->rankorder = flg;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscSFSetGraph - Set a parallel star forest

  Collective

  Input Parameters:
+ sf         - star forest
. nroots     - number of root vertices on the current process (these are possible targets for other process to attach leaves)
. nleaves    - number of leaf vertices on the current process, each of these references a root on any process
. ilocal     - locations of leaves in leafdata buffers, pass `NULL` for contiguous storage (locations must be >= 0, enforced
during setup in debug mode)
. localmode  - copy mode for `ilocal`
. iremote    - remote locations of root vertices for each leaf on the current process (locations must be >= 0, enforced
during setup in debug mode)
- remotemode - copy mode for `iremote`

  Level: intermediate

  Notes:
  Leaf indices in `ilocal` must be unique, otherwise an error occurs.

  Input arrays `ilocal` and `iremote` follow the `PetscCopyMode` semantics.
  In particular, if `localmode` or `remotemode` is `PETSC_OWN_POINTER` or `PETSC_USE_POINTER`,
  PETSc might modify the respective array;
  if `PETSC_USE_POINTER`, the user must delete the array after `PetscSFDestroy()`.
  Only if `PETSC_COPY_VALUES` is used, the respective array is guaranteed to stay intact and a const array can be passed (but a cast to non-const is needed).

  Fortran Notes:
  In Fortran you must use `PETSC_COPY_VALUES` for `localmode` and `remotemode`.

  Developer Notes:
  We sort leaves to check for duplicates and contiguousness and to find minleaf/maxleaf.
  This also allows to compare leaf sets of two `PetscSF`s easily.

.seealso: `PetscSF`, `PetscSFType`, `PetscSFCreate()`, `PetscSFView()`, `PetscSFGetGraph()`
@*/
PetscErrorCode PetscSFSetGraph(PetscSF sf, PetscInt nroots, PetscInt nleaves, PetscInt *ilocal, PetscCopyMode localmode, PetscSFNode *iremote, PetscCopyMode remotemode)
{
  PetscBool unique, contiguous;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sf, PETSCSF_CLASSID, 1);
  if (nleaves > 0 && ilocal) PetscAssertPointer(ilocal, 4);
  if (nleaves > 0) PetscAssertPointer(iremote, 6);
  PetscCheck(nroots >= 0, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "nroots %" PetscInt_FMT ", cannot be negative", nroots);
  PetscCheck(nleaves >= 0, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "nleaves %" PetscInt_FMT ", cannot be negative", nleaves);
  /* enums may be handled as unsigned by some compilers, NVHPC for example, the int cast
   * below is to prevent NVHPC from warning about meaningless comparison of unsigned with zero */
  PetscCheck((int)localmode >= PETSC_COPY_VALUES && localmode <= PETSC_USE_POINTER, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Wrong localmode %d", localmode);
  PetscCheck((int)remotemode >= PETSC_COPY_VALUES && remotemode <= PETSC_USE_POINTER, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Wrong remotemode %d", remotemode);

  if (sf->nroots >= 0) { /* Reset only if graph already set */
    PetscCall(PetscSFReset(sf));
  }

  PetscCall(PetscLogEventBegin(PETSCSF_SetGraph, sf, 0, 0, 0));
  if (PetscDefined(USE_DEBUG)) {
    PetscMPIInt size;

    PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)sf), &size));
    for (PetscInt i = 0; i < nleaves; i++) { PetscCheck(iremote[i].rank >= -1 && iremote[i].rank < size, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "iremote contains incorrect rank values"); }
  }

  sf->nroots  = nroots;
  sf->nleaves = nleaves;

  if (localmode == PETSC_COPY_VALUES && ilocal) {
    PetscInt *tlocal = NULL;

    PetscCall(PetscMalloc1(nleaves, &tlocal));
    PetscCall(PetscArraycpy(tlocal, ilocal, nleaves));
    ilocal = tlocal;
  }
  if (remotemode == PETSC_COPY_VALUES) {
    PetscSFNode *tremote = NULL;

    PetscCall(PetscMalloc1(nleaves, &tremote));
    PetscCall(PetscArraycpy(tremote, iremote, nleaves));
    iremote = tremote;
  }

  if (nleaves && ilocal) {
    PetscSFNode work;

    PetscCall(PetscSortIntWithDataArray(nleaves, ilocal, iremote, sizeof(PetscSFNode), &work));
    PetscCall(PetscSortedCheckDupsInt(nleaves, ilocal, &unique));
    unique = PetscNot(unique);
    PetscCheck(sf->allow_multi_leaves || unique, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Input ilocal has duplicate entries which is not allowed for this PetscSF");
    sf->minleaf = ilocal[0];
    sf->maxleaf = ilocal[nleaves - 1];
    contiguous  = (PetscBool)(unique && ilocal[0] == 0 && ilocal[nleaves - 1] == nleaves - 1);
  } else {
    sf->minleaf = 0;
    sf->maxleaf = nleaves - 1;
    unique      = PETSC_TRUE;
    contiguous  = PETSC_TRUE;
  }

  if (contiguous) {
    if (localmode == PETSC_USE_POINTER) {
      ilocal = NULL;
    } else {
      PetscCall(PetscFree(ilocal));
    }
  }
  sf->mine = ilocal;
  if (localmode == PETSC_USE_POINTER) {
    sf->mine_alloc = NULL;
  } else {
    sf->mine_alloc = ilocal;
  }
  if (PetscDefined(USE_DEBUG)) {
    PetscMPIInt size;

    PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)sf), &size));
    for (PetscInt i = 0; i < nleaves; i++) { PetscCheck(iremote[i].rank >= -1 && iremote[i].rank < size, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "iremote contains incorrect rank values"); }
  }
  sf->remote = iremote;
  if (remotemode == PETSC_USE_POINTER) {
    sf->remote_alloc = NULL;
  } else {
    sf->remote_alloc = iremote;
  }
  PetscCall(PetscLogEventEnd(PETSCSF_SetGraph, sf, 0, 0, 0));
  sf->graphset = PETSC_TRUE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscSFSetGraphWithPattern - Sets the graph of a `PetscSF` with a specific pattern

  Collective

  Input Parameters:
+ sf      - The `PetscSF`
. map     - Layout of roots over all processes (insignificant when pattern is `PETSCSF_PATTERN_ALLTOALL`)
- pattern - One of `PETSCSF_PATTERN_ALLGATHER`, `PETSCSF_PATTERN_GATHER`, `PETSCSF_PATTERN_ALLTOALL`

  Level: intermediate

  Notes:
  It is easier to explain `PetscSFPattern` using vectors. Suppose we have an MPI vector `x` and its `PetscLayout` is `map`.
  `n` and `N` are the local and global sizes of `x` respectively.

  With `PETSCSF_PATTERN_ALLGATHER`, the routine creates a graph that if one does `PetscSFBcastBegin()`/`PetscSFBcastEnd()` on it, it will copy `x` to
  sequential vectors `y` on all MPI processes.

  With `PETSCSF_PATTERN_GATHER`, the routine creates a graph that if one does `PetscSFBcastBegin()`/`PetscSFBcastEnd()` on it, it will copy `x` to a
  sequential vector `y` on rank 0.

  In above cases, entries of `x` are roots and entries of `y` are leaves.

  With `PETSCSF_PATTERN_ALLTOALL`, map is insignificant. Suppose NP is size of `sf`'s communicator. The routine
  creates a graph that every rank has NP leaves and NP roots. On rank i, its leaf j is connected to root i
  of rank j. Here 0 <=i,j<NP. It is a kind of `MPI_Alltoall()` with sendcount/recvcount being 1. Note that it does
  not mean one can not send multiple items. One just needs to create a new MPI datatype for the mulptiple data
  items with `MPI_Type_contiguous` and use that as the <unit> argument in SF routines.

  In this case, roots and leaves are symmetric.

.seealso: `PetscSF`, `PetscSFCreate()`, `PetscSFView()`, `PetscSFGetGraph()`
 @*/
PetscErrorCode PetscSFSetGraphWithPattern(PetscSF sf, PetscLayout map, PetscSFPattern pattern)
{
  MPI_Comm    comm;
  PetscInt    n, N, res[2];
  PetscMPIInt rank, size;
  PetscSFType type;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sf, PETSCSF_CLASSID, 1);
  if (pattern != PETSCSF_PATTERN_ALLTOALL) PetscAssertPointer(map, 2);
  PetscCall(PetscObjectGetComm((PetscObject)sf, &comm));
  PetscCheck(pattern >= PETSCSF_PATTERN_ALLGATHER && pattern <= PETSCSF_PATTERN_ALLTOALL, comm, PETSC_ERR_ARG_OUTOFRANGE, "Unsupported PetscSFPattern %d", pattern);
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  PetscCallMPI(MPI_Comm_size(comm, &size));

  if (pattern == PETSCSF_PATTERN_ALLTOALL) {
    type = PETSCSFALLTOALL;
    PetscCall(PetscLayoutCreate(comm, &sf->map));
    PetscCall(PetscLayoutSetLocalSize(sf->map, size));
    PetscCall(PetscLayoutSetSize(sf->map, (PetscInt)size * size));
    PetscCall(PetscLayoutSetUp(sf->map));
  } else {
    PetscCall(PetscLayoutGetLocalSize(map, &n));
    PetscCall(PetscLayoutGetSize(map, &N));
    res[0] = n;
    res[1] = -n;
    /* Check if n are same over all ranks so that we can optimize it */
    PetscCallMPI(MPIU_Allreduce(MPI_IN_PLACE, res, 2, MPIU_INT, MPI_MAX, comm));
    if (res[0] == -res[1]) { /* same n */
      type = (pattern == PETSCSF_PATTERN_ALLGATHER) ? PETSCSFALLGATHER : PETSCSFGATHER;
    } else {
      type = (pattern == PETSCSF_PATTERN_ALLGATHER) ? PETSCSFALLGATHERV : PETSCSFGATHERV;
    }
    PetscCall(PetscLayoutReference(map, &sf->map));
  }
  PetscCall(PetscSFSetType(sf, type));

  sf->pattern = pattern;
  sf->mine    = NULL; /* Contiguous */

  /* Set nleaves, nroots here in case user calls PetscSFGetGraph, which is legal to call even before PetscSFSetUp is called.
     Also set other easy stuff.
   */
  if (pattern == PETSCSF_PATTERN_ALLGATHER) {
    sf->nleaves = N;
    sf->nroots  = n;
    sf->nranks  = size;
    sf->minleaf = 0;
    sf->maxleaf = N - 1;
  } else if (pattern == PETSCSF_PATTERN_GATHER) {
    sf->nleaves = rank ? 0 : N;
    sf->nroots  = n;
    sf->nranks  = rank ? 0 : size;
    sf->minleaf = 0;
    sf->maxleaf = rank ? -1 : N - 1;
  } else if (pattern == PETSCSF_PATTERN_ALLTOALL) {
    sf->nleaves = size;
    sf->nroots  = size;
    sf->nranks  = size;
    sf->minleaf = 0;
    sf->maxleaf = size - 1;
  }
  sf->ndranks  = 0; /* We do not need to separate out distinguished ranks for patterned graphs to improve communication performance */
  sf->graphset = PETSC_TRUE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscSFCreateInverseSF - given a `PetscSF` in which all vertices have degree 1, creates the inverse map

  Collective

  Input Parameter:
. sf - star forest to invert

  Output Parameter:
. isf - inverse of `sf`

  Level: advanced

  Notes:
  All roots must have degree 1.

  The local space may be a permutation, but cannot be sparse.

.seealso: `PetscSF`, `PetscSFType`, `PetscSFSetGraph()`
@*/
PetscErrorCode PetscSFCreateInverseSF(PetscSF sf, PetscSF *isf)
{
  PetscMPIInt     rank;
  PetscInt        i, nroots, nleaves, maxlocal, count, *newilocal;
  const PetscInt *ilocal;
  PetscSFNode    *roots, *leaves;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sf, PETSCSF_CLASSID, 1);
  PetscSFCheckGraphSet(sf, 1);
  PetscAssertPointer(isf, 2);

  PetscCall(PetscSFGetGraph(sf, &nroots, &nleaves, &ilocal, NULL));
  maxlocal = sf->maxleaf + 1; /* TODO: We should use PetscSFGetLeafRange() */

  PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)sf), &rank));
  PetscCall(PetscMalloc2(nroots, &roots, maxlocal, &leaves));
  for (i = 0; i < maxlocal; i++) {
    leaves[i].rank  = rank;
    leaves[i].index = i;
  }
  for (i = 0; i < nroots; i++) {
    roots[i].rank  = -1;
    roots[i].index = -1;
  }
  PetscCall(PetscSFReduceBegin(sf, MPIU_SF_NODE, leaves, roots, MPI_REPLACE));
  PetscCall(PetscSFReduceEnd(sf, MPIU_SF_NODE, leaves, roots, MPI_REPLACE));

  /* Check whether our leaves are sparse */
  for (i = 0, count = 0; i < nroots; i++)
    if (roots[i].rank >= 0) count++;
  if (count == nroots) newilocal = NULL;
  else { /* Index for sparse leaves and compact "roots" array (which is to become our leaves). */ PetscCall(PetscMalloc1(count, &newilocal));
    for (i = 0, count = 0; i < nroots; i++) {
      if (roots[i].rank >= 0) {
        newilocal[count]   = i;
        roots[count].rank  = roots[i].rank;
        roots[count].index = roots[i].index;
        count++;
      }
    }
  }

  PetscCall(PetscSFDuplicate(sf, PETSCSF_DUPLICATE_CONFONLY, isf));
  PetscCall(PetscSFSetGraph(*isf, maxlocal, count, newilocal, PETSC_OWN_POINTER, roots, PETSC_COPY_VALUES));
  PetscCall(PetscFree2(roots, leaves));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscSFDuplicate - duplicate a `PetscSF`, optionally preserving rank connectivity and graph

  Collective

  Input Parameters:
+ sf  - communication object to duplicate
- opt - `PETSCSF_DUPLICATE_CONFONLY`, `PETSCSF_DUPLICATE_RANKS`, or `PETSCSF_DUPLICATE_GRAPH` (see `PetscSFDuplicateOption`)

  Output Parameter:
. newsf - new communication object

  Level: beginner

.seealso: `PetscSF`, `PetscSFType`, `PetscSFCreate()`, `PetscSFSetType()`, `PetscSFSetGraph()`
@*/
PetscErrorCode PetscSFDuplicate(PetscSF sf, PetscSFDuplicateOption opt, PetscSF *newsf)
{
  PetscSFType  type;
  MPI_Datatype dtype = MPIU_SCALAR;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sf, PETSCSF_CLASSID, 1);
  PetscValidLogicalCollectiveEnum(sf, opt, 2);
  PetscAssertPointer(newsf, 3);
  PetscCall(PetscSFCreate(PetscObjectComm((PetscObject)sf), newsf));
  PetscCall(PetscSFGetType(sf, &type));
  if (type) PetscCall(PetscSFSetType(*newsf, type));
  (*newsf)->allow_multi_leaves = sf->allow_multi_leaves; /* Dup this flag earlier since PetscSFSetGraph() below checks on this flag */
  if (opt == PETSCSF_DUPLICATE_GRAPH) {
    PetscSFCheckGraphSet(sf, 1);
    if (sf->pattern == PETSCSF_PATTERN_GENERAL) {
      PetscInt           nroots, nleaves;
      const PetscInt    *ilocal;
      const PetscSFNode *iremote;
      PetscCall(PetscSFGetGraph(sf, &nroots, &nleaves, &ilocal, &iremote));
      PetscCall(PetscSFSetGraph(*newsf, nroots, nleaves, (PetscInt *)ilocal, PETSC_COPY_VALUES, (PetscSFNode *)iremote, PETSC_COPY_VALUES));
    } else {
      PetscCall(PetscSFSetGraphWithPattern(*newsf, sf->map, sf->pattern));
    }
  }
  /* Since oldtype is committed, so is newtype, according to MPI */
  if (sf->vscat.bs > 1) PetscCallMPI(MPI_Type_dup(sf->vscat.unit, &dtype));
  (*newsf)->vscat.bs     = sf->vscat.bs;
  (*newsf)->vscat.unit   = dtype;
  (*newsf)->vscat.to_n   = sf->vscat.to_n;
  (*newsf)->vscat.from_n = sf->vscat.from_n;
  /* Do not copy lsf. Build it on demand since it is rarely used */

#if defined(PETSC_HAVE_DEVICE)
  (*newsf)->backend              = sf->backend;
  (*newsf)->unknown_input_stream = sf->unknown_input_stream;
  (*newsf)->use_gpu_aware_mpi    = sf->use_gpu_aware_mpi;
  (*newsf)->use_stream_aware_mpi = sf->use_stream_aware_mpi;
#endif
  PetscTryTypeMethod(sf, Duplicate, opt, *newsf);
  /* Don't do PetscSFSetUp() since the new sf's graph might have not been set. */
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscSFGetGraph - Get the graph specifying a parallel star forest

  Not Collective

  Input Parameter:
. sf - star forest

  Output Parameters:
+ nroots  - number of root vertices on the current process (these are possible targets for other process to attach leaves)
. nleaves - number of leaf vertices on the current process, each of these references a root on any process
. ilocal  - locations of leaves in leafdata buffers (if returned value is `NULL`, it means leaves are in contiguous storage)
- iremote - remote locations of root vertices for each leaf on the current process

  Level: intermediate

  Notes:
  We are not currently requiring that the graph is set, thus returning `nroots` = -1 if it has not been set yet

  The returned `ilocal` and `iremote` might contain values in different order than the input ones in `PetscSFSetGraph()`

  Fortran Notes:
  The returned `iremote` array is a copy and must be deallocated after use. Consequently, if you
  want to update the graph, you must call `PetscSFSetGraph()` after modifying the `iremote` array.

  To check for a `NULL` `ilocal` use
$      if (loc(ilocal) == loc(PETSC_NULL_INTEGER)) then

.seealso: `PetscSF`, `PetscSFType`, `PetscSFCreate()`, `PetscSFView()`, `PetscSFSetGraph()`
@*/
PetscErrorCode PetscSFGetGraph(PetscSF sf, PetscInt *nroots, PetscInt *nleaves, const PetscInt **ilocal, const PetscSFNode **iremote)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(sf, PETSCSF_CLASSID, 1);
  if (sf->ops->GetGraph) {
    PetscCall(sf->ops->GetGraph(sf, nroots, nleaves, ilocal, iremote));
  } else {
    if (nroots) *nroots = sf->nroots;
    if (nleaves) *nleaves = sf->nleaves;
    if (ilocal) *ilocal = sf->mine;
    if (iremote) *iremote = sf->remote;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscSFGetLeafRange - Get the active leaf ranges

  Not Collective

  Input Parameter:
. sf - star forest

  Output Parameters:
+ minleaf - minimum active leaf on this process. Returns 0 if there are no leaves.
- maxleaf - maximum active leaf on this process. Returns -1 if there are no leaves.

  Level: developer

.seealso: `PetscSF`, `PetscSFType`, `PetscSFCreate()`, `PetscSFView()`, `PetscSFSetGraph()`, `PetscSFGetGraph()`
@*/
PetscErrorCode PetscSFGetLeafRange(PetscSF sf, PetscInt *minleaf, PetscInt *maxleaf)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(sf, PETSCSF_CLASSID, 1);
  PetscSFCheckGraphSet(sf, 1);
  if (minleaf) *minleaf = sf->minleaf;
  if (maxleaf) *maxleaf = sf->maxleaf;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscSFViewFromOptions - View a `PetscSF` based on arguments in the options database

  Collective

  Input Parameters:
+ A    - the star forest
. obj  - Optional object that provides the prefix for the option names
- name - command line option

  Level: intermediate

  Note:
  See `PetscObjectViewFromOptions()` for possible `PetscViewer` and `PetscViewerFormat`

.seealso: `PetscSF`, `PetscSFView`, `PetscObjectViewFromOptions()`, `PetscSFCreate()`
@*/
PetscErrorCode PetscSFViewFromOptions(PetscSF A, PetscObject obj, const char name[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(A, PETSCSF_CLASSID, 1);
  PetscCall(PetscObjectViewFromOptions((PetscObject)A, obj, name));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscSFView - view a star forest

  Collective

  Input Parameters:
+ sf     - star forest
- viewer - viewer to display graph, for example `PETSC_VIEWER_STDOUT_WORLD`

  Level: beginner

.seealso: `PetscSF`, `PetscViewer`, `PetscSFCreate()`, `PetscSFSetGraph()`
@*/
PetscErrorCode PetscSFView(PetscSF sf, PetscViewer viewer)
{
  PetscBool         iascii;
  PetscViewerFormat format;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sf, PETSCSF_CLASSID, 1);
  if (!viewer) PetscCall(PetscViewerASCIIGetStdout(PetscObjectComm((PetscObject)sf), &viewer));
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 2);
  PetscCheckSameComm(sf, 1, viewer, 2);
  if (sf->graphset) PetscCall(PetscSFSetUp(sf));
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &iascii));
  if (iascii && viewer->format != PETSC_VIEWER_ASCII_MATLAB) {
    PetscMPIInt rank;
    PetscInt    j;

    PetscCall(PetscObjectPrintClassNamePrefixType((PetscObject)sf, viewer));
    PetscCall(PetscViewerASCIIPushTab(viewer));
    if (sf->pattern == PETSCSF_PATTERN_GENERAL) {
      if (!sf->graphset) {
        PetscCall(PetscViewerASCIIPrintf(viewer, "PetscSFSetGraph() has not been called yet\n"));
        PetscCall(PetscViewerASCIIPopTab(viewer));
        PetscFunctionReturn(PETSC_SUCCESS);
      }
      PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)sf), &rank));
      PetscCall(PetscViewerASCIIPushSynchronized(viewer));
      PetscCall(PetscViewerASCIISynchronizedPrintf(viewer, "[%d] Number of roots=%" PetscInt_FMT ", leaves=%" PetscInt_FMT ", remote ranks=%d\n", rank, sf->nroots, sf->nleaves, sf->nranks));
      for (PetscInt i = 0; i < sf->nleaves; i++) PetscCall(PetscViewerASCIISynchronizedPrintf(viewer, "[%d] %" PetscInt_FMT " <- (%d,%" PetscInt_FMT ")\n", rank, sf->mine ? sf->mine[i] : i, (PetscMPIInt)sf->remote[i].rank, sf->remote[i].index));
      PetscCall(PetscViewerFlush(viewer));
      PetscCall(PetscViewerGetFormat(viewer, &format));
      if (format == PETSC_VIEWER_ASCII_INFO_DETAIL) {
        PetscMPIInt *tmpranks, *perm;

        PetscCall(PetscMalloc2(sf->nranks, &tmpranks, sf->nranks, &perm));
        PetscCall(PetscArraycpy(tmpranks, sf->ranks, sf->nranks));
        for (PetscMPIInt i = 0; i < sf->nranks; i++) perm[i] = i;
        PetscCall(PetscSortMPIIntWithArray(sf->nranks, tmpranks, perm));
        PetscCall(PetscViewerASCIISynchronizedPrintf(viewer, "[%d] Roots referenced by my leaves, by rank\n", rank));
        for (PetscMPIInt ii = 0; ii < sf->nranks; ii++) {
          PetscMPIInt i = perm[ii];

          PetscCall(PetscViewerASCIISynchronizedPrintf(viewer, "[%d] %d: %" PetscInt_FMT " edges\n", rank, sf->ranks[i], sf->roffset[i + 1] - sf->roffset[i]));
          for (j = sf->roffset[i]; j < sf->roffset[i + 1]; j++) PetscCall(PetscViewerASCIISynchronizedPrintf(viewer, "[%d]    %" PetscInt_FMT " <- %" PetscInt_FMT "\n", rank, sf->rmine[j], sf->rremote[j]));
        }
        PetscCall(PetscFree2(tmpranks, perm));
      }
      PetscCall(PetscViewerFlush(viewer));
      PetscCall(PetscViewerASCIIPopSynchronized(viewer));
    }
    PetscCall(PetscViewerASCIIPopTab(viewer));
  }
  PetscTryTypeMethod(sf, View, viewer);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscSFGetRootRanks - Get root ranks and number of vertices referenced by leaves on this process

  Not Collective

  Input Parameter:
. sf - star forest

  Output Parameters:
+ nranks  - number of ranks referenced by local part
. ranks   - [`nranks`] array of ranks
. roffset - [`nranks`+1] offset in `rmine`/`rremote` for each rank
. rmine   - [`roffset`[`nranks`]] concatenated array holding local indices referencing each remote rank, or `NULL`
- rremote - [`roffset`[`nranks`]] concatenated array holding remote indices referenced for each remote rank, or `NULL`

  Level: developer

.seealso: `PetscSF`, `PetscSFGetLeafRanks()`
@*/
PetscErrorCode PetscSFGetRootRanks(PetscSF sf, PetscMPIInt *nranks, const PetscMPIInt **ranks, const PetscInt **roffset, const PetscInt **rmine, const PetscInt **rremote)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(sf, PETSCSF_CLASSID, 1);
  PetscCheck(sf->setupcalled, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Must call PetscSFSetUp() before obtaining ranks");
  if (sf->ops->GetRootRanks) {
    PetscUseTypeMethod(sf, GetRootRanks, nranks, ranks, roffset, rmine, rremote);
  } else {
    /* The generic implementation */
    if (nranks) *nranks = sf->nranks;
    if (ranks) *ranks = sf->ranks;
    if (roffset) *roffset = sf->roffset;
    if (rmine) *rmine = sf->rmine;
    if (rremote) *rremote = sf->rremote;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscSFGetLeafRanks - Get leaf ranks referencing roots on this process

  Not Collective

  Input Parameter:
. sf - star forest

  Output Parameters:
+ niranks  - number of leaf ranks referencing roots on this process
. iranks   - [`niranks`] array of ranks
. ioffset  - [`niranks`+1] offset in `irootloc` for each rank
- irootloc - [`ioffset`[`niranks`]] concatenated array holding local indices of roots referenced by each leaf rank

  Level: developer

.seealso: `PetscSF`, `PetscSFGetRootRanks()`
@*/
PetscErrorCode PetscSFGetLeafRanks(PetscSF sf, PetscMPIInt *niranks, const PetscMPIInt **iranks, const PetscInt **ioffset, const PetscInt **irootloc)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(sf, PETSCSF_CLASSID, 1);
  PetscCheck(sf->setupcalled, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Must call PetscSFSetUp() before obtaining ranks");
  if (sf->ops->GetLeafRanks) {
    PetscUseTypeMethod(sf, GetLeafRanks, niranks, iranks, ioffset, irootloc);
  } else {
    PetscSFType type;
    PetscCall(PetscSFGetType(sf, &type));
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "PetscSFGetLeafRanks() is not supported on this StarForest type: %s", type);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscBool InList(PetscMPIInt needle, PetscMPIInt n, const PetscMPIInt *list)
{
  PetscInt i;
  for (i = 0; i < n; i++) {
    if (needle == list[i]) return PETSC_TRUE;
  }
  return PETSC_FALSE;
}

/*@C
  PetscSFSetUpRanks - Set up data structures associated with ranks; this is for internal use by `PetscSF` implementations.

  Collective

  Input Parameters:
+ sf     - `PetscSF` to set up; `PetscSFSetGraph()` must have been called
- dgroup - `MPI_Group` of ranks to be distinguished (e.g., for self or shared memory exchange)

  Level: developer

.seealso: `PetscSF`, `PetscSFGetRootRanks()`
@*/
PetscErrorCode PetscSFSetUpRanks(PetscSF sf, MPI_Group dgroup)
{
  PetscHMapI    table;
  PetscHashIter pos;
  PetscMPIInt   size, groupsize, *groupranks, *ranks;
  PetscInt     *rcount;
  PetscInt      irank, sfnrank, ranksi;
  PetscMPIInt   i, orank = -1;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sf, PETSCSF_CLASSID, 1);
  PetscSFCheckGraphSet(sf, 1);
  PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)sf), &size));
  PetscCall(PetscHMapICreateWithSize(10, &table));
  for (i = 0; i < sf->nleaves; i++) {
    /* Log 1-based rank */
    PetscCall(PetscHMapISetWithMode(table, sf->remote[i].rank + 1, 1, ADD_VALUES));
  }
  PetscCall(PetscHMapIGetSize(table, &sfnrank));
  PetscCall(PetscMPIIntCast(sfnrank, &sf->nranks));
  PetscCall(PetscMalloc4(sf->nranks, &sf->ranks, sf->nranks + 1, &sf->roffset, sf->nleaves, &sf->rmine, sf->nleaves, &sf->rremote));
  PetscCall(PetscMalloc2(sf->nranks, &rcount, sf->nranks, &ranks));
  PetscHashIterBegin(table, pos);
  for (i = 0; i < sf->nranks; i++) {
    PetscHashIterGetKey(table, pos, ranksi);
    PetscCall(PetscMPIIntCast(ranksi, &ranks[i]));
    PetscHashIterGetVal(table, pos, rcount[i]);
    PetscHashIterNext(table, pos);
    ranks[i]--; /* Convert back to 0-based */
  }
  PetscCall(PetscHMapIDestroy(&table));

  /* We expect that dgroup is reliably "small" while nranks could be large */
  {
    MPI_Group    group = MPI_GROUP_NULL;
    PetscMPIInt *dgroupranks;

    PetscCallMPI(MPI_Comm_group(PetscObjectComm((PetscObject)sf), &group));
    PetscCallMPI(MPI_Group_size(dgroup, &groupsize));
    PetscCall(PetscMalloc1(groupsize, &dgroupranks));
    PetscCall(PetscMalloc1(groupsize, &groupranks));
    for (i = 0; i < groupsize; i++) dgroupranks[i] = i;
    if (groupsize) PetscCallMPI(MPI_Group_translate_ranks(dgroup, groupsize, dgroupranks, group, groupranks));
    PetscCallMPI(MPI_Group_free(&group));
    PetscCall(PetscFree(dgroupranks));
  }

  /* Partition ranks[] into distinguished (first sf->ndranks) followed by non-distinguished */
  for (sf->ndranks = 0, i = sf->nranks; sf->ndranks < i;) {
    for (i--; sf->ndranks < i; i--) { /* Scan i backward looking for distinguished rank */
      if (InList(ranks[i], groupsize, groupranks)) break;
    }
    for (; sf->ndranks <= i; sf->ndranks++) { /* Scan sf->ndranks forward looking for non-distinguished rank */
      if (!InList(ranks[sf->ndranks], groupsize, groupranks)) break;
    }
    if (sf->ndranks < i) { /* Swap ranks[sf->ndranks] with ranks[i] */
      PetscMPIInt tmprank;
      PetscInt    tmpcount;

      tmprank             = ranks[i];
      tmpcount            = rcount[i];
      ranks[i]            = ranks[sf->ndranks];
      rcount[i]           = rcount[sf->ndranks];
      ranks[sf->ndranks]  = tmprank;
      rcount[sf->ndranks] = tmpcount;
      sf->ndranks++;
    }
  }
  PetscCall(PetscFree(groupranks));
  PetscCall(PetscSortMPIIntWithIntArray(sf->ndranks, ranks, rcount));
  if (rcount) PetscCall(PetscSortMPIIntWithIntArray(sf->nranks - sf->ndranks, ranks + sf->ndranks, rcount + sf->ndranks));
  sf->roffset[0] = 0;
  for (i = 0; i < sf->nranks; i++) {
    PetscCall(PetscMPIIntCast(ranks[i], sf->ranks + i));
    sf->roffset[i + 1] = sf->roffset[i] + rcount[i];
    rcount[i]          = 0;
  }
  for (i = 0, irank = -1, orank = -1; i < sf->nleaves; i++) {
    /* short circuit */
    if (orank != sf->remote[i].rank) {
      /* Search for index of iremote[i].rank in sf->ranks */
      PetscCall(PetscFindMPIInt((PetscMPIInt)sf->remote[i].rank, sf->ndranks, sf->ranks, &irank));
      if (irank < 0) {
        PetscCall(PetscFindMPIInt((PetscMPIInt)sf->remote[i].rank, sf->nranks - sf->ndranks, sf->ranks + sf->ndranks, &irank));
        if (irank >= 0) irank += sf->ndranks;
      }
      orank = (PetscMPIInt)sf->remote[i].rank;
    }
    PetscCheck(irank >= 0, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Could not find rank %d in array", (PetscMPIInt)sf->remote[i].rank);
    sf->rmine[sf->roffset[irank] + rcount[irank]]   = sf->mine ? sf->mine[i] : i;
    sf->rremote[sf->roffset[irank] + rcount[irank]] = sf->remote[i].index;
    rcount[irank]++;
  }
  PetscCall(PetscFree2(rcount, ranks));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscSFGetGroups - gets incoming and outgoing process groups

  Collective

  Input Parameter:
. sf - star forest

  Output Parameters:
+ incoming - group of origin processes for incoming edges (leaves that reference my roots)
- outgoing - group of destination processes for outgoing edges (roots that I reference)

  Level: developer

.seealso: `PetscSF`, `PetscSFGetWindow()`, `PetscSFRestoreWindow()`
@*/
PetscErrorCode PetscSFGetGroups(PetscSF sf, MPI_Group *incoming, MPI_Group *outgoing)
{
  MPI_Group group = MPI_GROUP_NULL;

  PetscFunctionBegin;
  PetscCheck(sf->nranks >= 0, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Must call PetscSFSetUpRanks() before obtaining groups");
  if (sf->ingroup == MPI_GROUP_NULL) {
    PetscInt        i;
    const PetscInt *indegree;
    PetscMPIInt     rank, *outranks, *inranks, indegree0;
    PetscSFNode    *remote;
    PetscSF         bgcount;

    /* Compute the number of incoming ranks */
    PetscCall(PetscMalloc1(sf->nranks, &remote));
    for (i = 0; i < sf->nranks; i++) {
      remote[i].rank  = sf->ranks[i];
      remote[i].index = 0;
    }
    PetscCall(PetscSFDuplicate(sf, PETSCSF_DUPLICATE_CONFONLY, &bgcount));
    PetscCall(PetscSFSetGraph(bgcount, 1, sf->nranks, NULL, PETSC_COPY_VALUES, remote, PETSC_OWN_POINTER));
    PetscCall(PetscSFComputeDegreeBegin(bgcount, &indegree));
    PetscCall(PetscSFComputeDegreeEnd(bgcount, &indegree));
    /* Enumerate the incoming ranks */
    PetscCall(PetscMalloc2(indegree[0], &inranks, sf->nranks, &outranks));
    PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)sf), &rank));
    for (i = 0; i < sf->nranks; i++) outranks[i] = rank;
    PetscCall(PetscSFGatherBegin(bgcount, MPI_INT, outranks, inranks));
    PetscCall(PetscSFGatherEnd(bgcount, MPI_INT, outranks, inranks));
    PetscCallMPI(MPI_Comm_group(PetscObjectComm((PetscObject)sf), &group));
    PetscCall(PetscMPIIntCast(indegree[0], &indegree0));
    PetscCallMPI(MPI_Group_incl(group, indegree0, inranks, &sf->ingroup));
    PetscCallMPI(MPI_Group_free(&group));
    PetscCall(PetscFree2(inranks, outranks));
    PetscCall(PetscSFDestroy(&bgcount));
  }
  *incoming = sf->ingroup;

  if (sf->outgroup == MPI_GROUP_NULL) {
    PetscCallMPI(MPI_Comm_group(PetscObjectComm((PetscObject)sf), &group));
    PetscCallMPI(MPI_Group_incl(group, sf->nranks, sf->ranks, &sf->outgroup));
    PetscCallMPI(MPI_Group_free(&group));
  }
  *outgoing = sf->outgroup;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscSFGetRanksSF - gets the `PetscSF` to perform communications with root ranks

  Collective

  Input Parameter:
. sf - star forest

  Output Parameter:
. rsf - the star forest with a single root per process to perform communications

  Level: developer

.seealso: `PetscSF`, `PetscSFSetGraph()`, `PetscSFGetRootRanks()`
@*/
PetscErrorCode PetscSFGetRanksSF(PetscSF sf, PetscSF *rsf)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(sf, PETSCSF_CLASSID, 1);
  PetscAssertPointer(rsf, 2);
  if (!sf->rankssf) {
    PetscSFNode       *rremotes;
    const PetscMPIInt *ranks;
    PetscMPIInt        nranks;

    PetscCall(PetscSFGetRootRanks(sf, &nranks, &ranks, NULL, NULL, NULL));
    PetscCall(PetscMalloc1(nranks, &rremotes));
    for (PetscInt i = 0; i < nranks; i++) {
      rremotes[i].rank  = ranks[i];
      rremotes[i].index = 0;
    }
    PetscCall(PetscSFDuplicate(sf, PETSCSF_DUPLICATE_CONFONLY, &sf->rankssf));
    PetscCall(PetscSFSetGraph(sf->rankssf, 1, nranks, NULL, PETSC_OWN_POINTER, rremotes, PETSC_OWN_POINTER));
  }
  *rsf = sf->rankssf;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscSFGetMultiSF - gets the inner `PetscSF` implementing gathers and scatters

  Collective

  Input Parameter:
. sf - star forest that may contain roots with 0 or with more than 1 vertex

  Output Parameter:
. multi - star forest with split roots, such that each root has degree exactly 1

  Level: developer

  Note:
  In most cases, users should use `PetscSFGatherBegin()` and `PetscSFScatterBegin()` instead of manipulating multi
  directly. Since multi satisfies the stronger condition that each entry in the global space has exactly one incoming
  edge, it is a candidate for future optimization that might involve its removal.

.seealso: `PetscSF`, `PetscSFSetGraph()`, `PetscSFGatherBegin()`, `PetscSFScatterBegin()`, `PetscSFComputeMultiRootOriginalNumbering()`
@*/
PetscErrorCode PetscSFGetMultiSF(PetscSF sf, PetscSF *multi)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(sf, PETSCSF_CLASSID, 1);
  PetscAssertPointer(multi, 2);
  if (sf->nroots < 0) { /* Graph has not been set yet; why do we need this? */
    PetscCall(PetscSFDuplicate(sf, PETSCSF_DUPLICATE_RANKS, &sf->multi));
    *multi           = sf->multi;
    sf->multi->multi = sf->multi;
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  if (!sf->multi) {
    const PetscInt *indegree;
    PetscInt        i, *inoffset, *outones, *outoffset, maxlocal;
    PetscSFNode    *remote;
    maxlocal = sf->maxleaf + 1; /* TODO: We should use PetscSFGetLeafRange() */
    PetscCall(PetscSFComputeDegreeBegin(sf, &indegree));
    PetscCall(PetscSFComputeDegreeEnd(sf, &indegree));
    PetscCall(PetscMalloc3(sf->nroots + 1, &inoffset, maxlocal, &outones, maxlocal, &outoffset));
    inoffset[0] = 0;
    for (i = 0; i < sf->nroots; i++) inoffset[i + 1] = inoffset[i] + indegree[i];
    for (i = 0; i < maxlocal; i++) outones[i] = 1;
    PetscCall(PetscSFFetchAndOpBegin(sf, MPIU_INT, inoffset, outones, outoffset, MPI_SUM));
    PetscCall(PetscSFFetchAndOpEnd(sf, MPIU_INT, inoffset, outones, outoffset, MPI_SUM));
    for (i = 0; i < sf->nroots; i++) inoffset[i] -= indegree[i]; /* Undo the increment */
    if (PetscDefined(USE_DEBUG)) {                               /* Check that the expected number of increments occurred */
      for (i = 0; i < sf->nroots; i++) PetscCheck(inoffset[i] + indegree[i] == inoffset[i + 1], PETSC_COMM_SELF, PETSC_ERR_PLIB, "Incorrect result after PetscSFFetchAndOp");
    }
    PetscCall(PetscMalloc1(sf->nleaves, &remote));
    for (i = 0; i < sf->nleaves; i++) {
      remote[i].rank  = sf->remote[i].rank;
      remote[i].index = outoffset[sf->mine ? sf->mine[i] : i];
    }
    PetscCall(PetscSFDuplicate(sf, PETSCSF_DUPLICATE_RANKS, &sf->multi));
    sf->multi->multi = sf->multi;
    PetscCall(PetscSFSetGraph(sf->multi, inoffset[sf->nroots], sf->nleaves, sf->mine, PETSC_COPY_VALUES, remote, PETSC_OWN_POINTER));
    if (sf->rankorder) { /* Sort the ranks */
      PetscMPIInt  rank;
      PetscInt    *inranks, *newoffset, *outranks, *newoutoffset, *tmpoffset, maxdegree;
      PetscSFNode *newremote;
      PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)sf), &rank));
      for (i = 0, maxdegree = 0; i < sf->nroots; i++) maxdegree = PetscMax(maxdegree, indegree[i]);
      PetscCall(PetscMalloc5(sf->multi->nroots, &inranks, sf->multi->nroots, &newoffset, maxlocal, &outranks, maxlocal, &newoutoffset, maxdegree, &tmpoffset));
      for (i = 0; i < maxlocal; i++) outranks[i] = rank;
      PetscCall(PetscSFReduceBegin(sf->multi, MPIU_INT, outranks, inranks, MPI_REPLACE));
      PetscCall(PetscSFReduceEnd(sf->multi, MPIU_INT, outranks, inranks, MPI_REPLACE));
      /* Sort the incoming ranks at each vertex, build the inverse map */
      for (i = 0; i < sf->nroots; i++) {
        PetscInt j;
        for (j = 0; j < indegree[i]; j++) tmpoffset[j] = j;
        PetscCall(PetscSortIntWithArray(indegree[i], PetscSafePointerPlusOffset(inranks, inoffset[i]), tmpoffset));
        for (j = 0; j < indegree[i]; j++) newoffset[inoffset[i] + tmpoffset[j]] = inoffset[i] + j;
      }
      PetscCall(PetscSFBcastBegin(sf->multi, MPIU_INT, newoffset, newoutoffset, MPI_REPLACE));
      PetscCall(PetscSFBcastEnd(sf->multi, MPIU_INT, newoffset, newoutoffset, MPI_REPLACE));
      PetscCall(PetscMalloc1(sf->nleaves, &newremote));
      for (i = 0; i < sf->nleaves; i++) {
        newremote[i].rank  = sf->remote[i].rank;
        newremote[i].index = newoutoffset[sf->mine ? sf->mine[i] : i];
      }
      PetscCall(PetscSFSetGraph(sf->multi, inoffset[sf->nroots], sf->nleaves, sf->mine, PETSC_COPY_VALUES, newremote, PETSC_OWN_POINTER));
      PetscCall(PetscFree5(inranks, newoffset, outranks, newoutoffset, tmpoffset));
    }
    PetscCall(PetscFree3(inoffset, outones, outoffset));
  }
  *multi = sf->multi;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscSFCreateEmbeddedRootSF - removes edges from all but the selected roots of a `PetscSF`, does not remap indices

  Collective

  Input Parameters:
+ sf        - original star forest
. nselected - number of selected roots on this process
- selected  - indices of the selected roots on this process

  Output Parameter:
. esf - new star forest

  Level: advanced

  Note:
  To use the new `PetscSF`, it may be necessary to know the indices of the leaves that are still participating. This can
  be done by calling PetscSFGetGraph().

.seealso: `PetscSF`, `PetscSFSetGraph()`, `PetscSFGetGraph()`
@*/
PetscErrorCode PetscSFCreateEmbeddedRootSF(PetscSF sf, PetscInt nselected, const PetscInt *selected, PetscSF *esf)
{
  PetscInt           i, j, n, nroots, nleaves, esf_nleaves, *new_ilocal, minleaf, maxleaf, maxlocal;
  const PetscInt    *ilocal;
  signed char       *rootdata, *leafdata, *leafmem;
  const PetscSFNode *iremote;
  PetscSFNode       *new_iremote;
  MPI_Comm           comm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sf, PETSCSF_CLASSID, 1);
  PetscSFCheckGraphSet(sf, 1);
  if (nselected) PetscAssertPointer(selected, 3);
  PetscAssertPointer(esf, 4);

  PetscCall(PetscSFSetUp(sf));
  PetscCall(PetscLogEventBegin(PETSCSF_EmbedSF, sf, 0, 0, 0));
  PetscCall(PetscObjectGetComm((PetscObject)sf, &comm));
  PetscCall(PetscSFGetGraph(sf, &nroots, &nleaves, &ilocal, &iremote));

  if (PetscDefined(USE_DEBUG)) { /* Error out if selected[] has dups or out of range indices */
    PetscBool dups;
    PetscCall(PetscCheckDupsInt(nselected, selected, &dups));
    PetscCheck(!dups, comm, PETSC_ERR_ARG_WRONG, "selected[] has dups");
    for (i = 0; i < nselected; i++) PetscCheck(selected[i] >= 0 && selected[i] < nroots, comm, PETSC_ERR_ARG_OUTOFRANGE, "selected root index %" PetscInt_FMT " is out of [0,%" PetscInt_FMT ")", selected[i], nroots);
  }

  if (sf->ops->CreateEmbeddedRootSF) PetscUseTypeMethod(sf, CreateEmbeddedRootSF, nselected, selected, esf);
  else {
    /* A generic version of creating embedded sf */
    PetscCall(PetscSFGetLeafRange(sf, &minleaf, &maxleaf));
    maxlocal = maxleaf - minleaf + 1;
    PetscCall(PetscCalloc2(nroots, &rootdata, maxlocal, &leafmem));
    leafdata = PetscSafePointerPlusOffset(leafmem, -minleaf);
    /* Tag selected roots and bcast to leaves */
    for (i = 0; i < nselected; i++) rootdata[selected[i]] = 1;
    PetscCall(PetscSFBcastBegin(sf, MPI_SIGNED_CHAR, rootdata, leafdata, MPI_REPLACE));
    PetscCall(PetscSFBcastEnd(sf, MPI_SIGNED_CHAR, rootdata, leafdata, MPI_REPLACE));

    /* Build esf with leaves that are still connected */
    esf_nleaves = 0;
    for (i = 0; i < nleaves; i++) {
      j = ilocal ? ilocal[i] : i;
      /* esf_nleaves += leafdata[j] should work in theory, but failed with SFWindow bugs
         with PetscSFBcast. See https://gitlab.com/petsc/petsc/issues/555
      */
      esf_nleaves += (leafdata[j] ? 1 : 0);
    }
    PetscCall(PetscMalloc1(esf_nleaves, &new_ilocal));
    PetscCall(PetscMalloc1(esf_nleaves, &new_iremote));
    for (i = n = 0; i < nleaves; i++) {
      j = ilocal ? ilocal[i] : i;
      if (leafdata[j]) {
        new_ilocal[n]        = j;
        new_iremote[n].rank  = iremote[i].rank;
        new_iremote[n].index = iremote[i].index;
        ++n;
      }
    }
    PetscCall(PetscSFCreate(comm, esf));
    PetscCall(PetscSFSetFromOptions(*esf));
    PetscCall(PetscSFSetGraph(*esf, nroots, esf_nleaves, new_ilocal, PETSC_OWN_POINTER, new_iremote, PETSC_OWN_POINTER));
    PetscCall(PetscFree2(rootdata, leafmem));
  }
  PetscCall(PetscLogEventEnd(PETSCSF_EmbedSF, sf, 0, 0, 0));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscSFCreateEmbeddedLeafSF - removes edges from all but the selected leaves of a `PetscSF`, does not remap indices

  Collective

  Input Parameters:
+ sf        - original star forest
. nselected - number of selected leaves on this process
- selected  - indices of the selected leaves on this process

  Output Parameter:
. newsf - new star forest

  Level: advanced

.seealso: `PetscSF`, `PetscSFCreateEmbeddedRootSF()`, `PetscSFSetGraph()`, `PetscSFGetGraph()`
@*/
PetscErrorCode PetscSFCreateEmbeddedLeafSF(PetscSF sf, PetscInt nselected, const PetscInt *selected, PetscSF *newsf)
{
  const PetscSFNode *iremote;
  PetscSFNode       *new_iremote;
  const PetscInt    *ilocal;
  PetscInt           i, nroots, *leaves, *new_ilocal;
  MPI_Comm           comm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sf, PETSCSF_CLASSID, 1);
  PetscSFCheckGraphSet(sf, 1);
  if (nselected) PetscAssertPointer(selected, 3);
  PetscAssertPointer(newsf, 4);

  /* Uniq selected[] and put results in leaves[] */
  PetscCall(PetscObjectGetComm((PetscObject)sf, &comm));
  PetscCall(PetscMalloc1(nselected, &leaves));
  PetscCall(PetscArraycpy(leaves, selected, nselected));
  PetscCall(PetscSortedRemoveDupsInt(&nselected, leaves));
  PetscCheck(!nselected || !(leaves[0] < 0 || leaves[nselected - 1] >= sf->nleaves), comm, PETSC_ERR_ARG_OUTOFRANGE, "Min/Max leaf indices %" PetscInt_FMT "/%" PetscInt_FMT " are not in [0,%" PetscInt_FMT ")", leaves[0], leaves[nselected - 1], sf->nleaves);

  /* Optimize the routine only when sf is setup and hence we can reuse sf's communication pattern */
  if (sf->setupcalled && sf->ops->CreateEmbeddedLeafSF) PetscUseTypeMethod(sf, CreateEmbeddedLeafSF, nselected, leaves, newsf);
  else {
    PetscCall(PetscSFGetGraph(sf, &nroots, NULL, &ilocal, &iremote));
    PetscCall(PetscMalloc1(nselected, &new_ilocal));
    PetscCall(PetscMalloc1(nselected, &new_iremote));
    for (i = 0; i < nselected; ++i) {
      const PetscInt l     = leaves[i];
      new_ilocal[i]        = ilocal ? ilocal[l] : l;
      new_iremote[i].rank  = iremote[l].rank;
      new_iremote[i].index = iremote[l].index;
    }
    PetscCall(PetscSFDuplicate(sf, PETSCSF_DUPLICATE_CONFONLY, newsf));
    PetscCall(PetscSFSetGraph(*newsf, nroots, nselected, new_ilocal, PETSC_OWN_POINTER, new_iremote, PETSC_OWN_POINTER));
  }
  PetscCall(PetscFree(leaves));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscSFBcastBegin - begin pointwise broadcast with root value being reduced to leaf value, to be concluded with call to `PetscSFBcastEnd()`

  Collective

  Input Parameters:
+ sf       - star forest on which to communicate
. unit     - data type associated with each node
. rootdata - buffer to broadcast
- op       - operation to use for reduction

  Output Parameter:
. leafdata - buffer to be reduced with values from each leaf's respective root

  Level: intermediate

  Note:
  When PETSc is configured with device support, it will use its own mechanism to figure out whether the given data pointers
  are host pointers or device pointers, which may incur a noticeable cost. If you already knew the info, you should
  use `PetscSFBcastWithMemTypeBegin()` instead.

.seealso: `PetscSF`, `PetscSFBcastEnd()`, `PetscSFBcastWithMemTypeBegin()`
@*/
PetscErrorCode PetscSFBcastBegin(PetscSF sf, MPI_Datatype unit, const void *rootdata, void *leafdata, MPI_Op op)
{
  PetscMemType rootmtype, leafmtype;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sf, PETSCSF_CLASSID, 1);
  PetscCall(PetscSFSetUp(sf));
  if (!sf->vscat.logging) PetscCall(PetscLogEventBegin(PETSCSF_BcastBegin, sf, 0, 0, 0));
  PetscCall(PetscGetMemType(rootdata, &rootmtype));
  PetscCall(PetscGetMemType(leafdata, &leafmtype));
  PetscUseTypeMethod(sf, BcastBegin, unit, rootmtype, rootdata, leafmtype, leafdata, op);
  if (!sf->vscat.logging) PetscCall(PetscLogEventEnd(PETSCSF_BcastBegin, sf, 0, 0, 0));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscSFBcastWithMemTypeBegin - begin pointwise broadcast with root value being reduced to leaf value with explicit memory types, to be concluded with call
  to `PetscSFBcastEnd()`

  Collective

  Input Parameters:
+ sf        - star forest on which to communicate
. unit      - data type associated with each node
. rootmtype - memory type of rootdata
. rootdata  - buffer to broadcast
. leafmtype - memory type of leafdata
- op        - operation to use for reduction

  Output Parameter:
. leafdata - buffer to be reduced with values from each leaf's respective root

  Level: intermediate

.seealso: `PetscSF`, `PetscSFBcastEnd()`, `PetscSFBcastBegin()`
@*/
PetscErrorCode PetscSFBcastWithMemTypeBegin(PetscSF sf, MPI_Datatype unit, PetscMemType rootmtype, const void *rootdata, PetscMemType leafmtype, void *leafdata, MPI_Op op)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(sf, PETSCSF_CLASSID, 1);
  PetscCall(PetscSFSetUp(sf));
  if (!sf->vscat.logging) PetscCall(PetscLogEventBegin(PETSCSF_BcastBegin, sf, 0, 0, 0));
  PetscUseTypeMethod(sf, BcastBegin, unit, rootmtype, rootdata, leafmtype, leafdata, op);
  if (!sf->vscat.logging) PetscCall(PetscLogEventEnd(PETSCSF_BcastBegin, sf, 0, 0, 0));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscSFBcastEnd - end a broadcast and reduce operation started with `PetscSFBcastBegin()` or `PetscSFBcastWithMemTypeBegin()`

  Collective

  Input Parameters:
+ sf       - star forest
. unit     - data type
. rootdata - buffer to broadcast
- op       - operation to use for reduction

  Output Parameter:
. leafdata - buffer to be reduced with values from each leaf's respective root

  Level: intermediate

.seealso: `PetscSF`, `PetscSFSetGraph()`, `PetscSFReduceEnd()`
@*/
PetscErrorCode PetscSFBcastEnd(PetscSF sf, MPI_Datatype unit, const void *rootdata, void *leafdata, MPI_Op op)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(sf, PETSCSF_CLASSID, 1);
  if (!sf->vscat.logging) PetscCall(PetscLogEventBegin(PETSCSF_BcastEnd, sf, 0, 0, 0));
  PetscUseTypeMethod(sf, BcastEnd, unit, rootdata, leafdata, op);
  if (!sf->vscat.logging) PetscCall(PetscLogEventEnd(PETSCSF_BcastEnd, sf, 0, 0, 0));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscSFReduceBegin - begin reduction of leafdata into rootdata, to be completed with call to `PetscSFReduceEnd()`

  Collective

  Input Parameters:
+ sf       - star forest
. unit     - data type
. leafdata - values to reduce
- op       - reduction operation

  Output Parameter:
. rootdata - result of reduction of values from all leaves of each root

  Level: intermediate

  Note:
  When PETSc is configured with device support, it will use its own mechanism to figure out whether the given data pointers
  are host pointers or device pointers, which may incur a noticeable cost. If you already knew the info, you should
  use `PetscSFReduceWithMemTypeBegin()` instead.

.seealso: `PetscSF`, `PetscSFBcastBegin()`, `PetscSFReduceWithMemTypeBegin()`, `PetscSFReduceEnd()`
@*/
PetscErrorCode PetscSFReduceBegin(PetscSF sf, MPI_Datatype unit, const void *leafdata, void *rootdata, MPI_Op op)
{
  PetscMemType rootmtype, leafmtype;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sf, PETSCSF_CLASSID, 1);
  PetscCall(PetscSFSetUp(sf));
  if (!sf->vscat.logging) PetscCall(PetscLogEventBegin(PETSCSF_ReduceBegin, sf, 0, 0, 0));
  PetscCall(PetscGetMemType(rootdata, &rootmtype));
  PetscCall(PetscGetMemType(leafdata, &leafmtype));
  PetscCall(sf->ops->ReduceBegin(sf, unit, leafmtype, leafdata, rootmtype, rootdata, op));
  if (!sf->vscat.logging) PetscCall(PetscLogEventEnd(PETSCSF_ReduceBegin, sf, 0, 0, 0));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscSFReduceWithMemTypeBegin - begin reduction of leafdata into rootdata with explicit memory types, to be completed with call to `PetscSFReduceEnd()`

  Collective

  Input Parameters:
+ sf        - star forest
. unit      - data type
. leafmtype - memory type of leafdata
. leafdata  - values to reduce
. rootmtype - memory type of rootdata
- op        - reduction operation

  Output Parameter:
. rootdata - result of reduction of values from all leaves of each root

  Level: intermediate

.seealso: `PetscSF`, `PetscSFBcastBegin()`, `PetscSFReduceBegin()`, `PetscSFReduceEnd()`
@*/
PetscErrorCode PetscSFReduceWithMemTypeBegin(PetscSF sf, MPI_Datatype unit, PetscMemType leafmtype, const void *leafdata, PetscMemType rootmtype, void *rootdata, MPI_Op op)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(sf, PETSCSF_CLASSID, 1);
  PetscCall(PetscSFSetUp(sf));
  if (!sf->vscat.logging) PetscCall(PetscLogEventBegin(PETSCSF_ReduceBegin, sf, 0, 0, 0));
  PetscCall(sf->ops->ReduceBegin(sf, unit, leafmtype, leafdata, rootmtype, rootdata, op));
  if (!sf->vscat.logging) PetscCall(PetscLogEventEnd(PETSCSF_ReduceBegin, sf, 0, 0, 0));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscSFReduceEnd - end a reduction operation started with `PetscSFReduceBegin()` or `PetscSFReduceWithMemTypeBegin()`

  Collective

  Input Parameters:
+ sf       - star forest
. unit     - data type
. leafdata - values to reduce
- op       - reduction operation

  Output Parameter:
. rootdata - result of reduction of values from all leaves of each root

  Level: intermediate

.seealso: `PetscSF`, `PetscSFSetGraph()`, `PetscSFBcastEnd()`, `PetscSFReduceBegin()`, `PetscSFReduceWithMemTypeBegin()`
@*/
PetscErrorCode PetscSFReduceEnd(PetscSF sf, MPI_Datatype unit, const void *leafdata, void *rootdata, MPI_Op op)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(sf, PETSCSF_CLASSID, 1);
  if (!sf->vscat.logging) PetscCall(PetscLogEventBegin(PETSCSF_ReduceEnd, sf, 0, 0, 0));
  PetscUseTypeMethod(sf, ReduceEnd, unit, leafdata, rootdata, op);
  if (!sf->vscat.logging) PetscCall(PetscLogEventEnd(PETSCSF_ReduceEnd, sf, 0, 0, 0));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscSFFetchAndOpBegin - begin operation that fetches values from root and updates atomically by applying operation using my leaf value,
  to be completed with `PetscSFFetchAndOpEnd()`

  Collective

  Input Parameters:
+ sf       - star forest
. unit     - data type
. leafdata - leaf values to use in reduction
- op       - operation to use for reduction

  Output Parameters:
+ rootdata   - root values to be updated, input state is seen by first process to perform an update
- leafupdate - state at each leaf's respective root immediately prior to my atomic update

  Level: advanced

  Note:
  The update is only atomic at the granularity provided by the hardware. Different roots referenced by the same process
  might be updated in a different order. Furthermore, if a composite type is used for the unit datatype, atomicity is
  not guaranteed across the whole vertex. Therefore, this function is mostly only used with primitive types such as
  integers.

.seealso: `PetscSF`, `PetscSFComputeDegreeBegin()`, `PetscSFReduceBegin()`, `PetscSFSetGraph()`
@*/
PetscErrorCode PetscSFFetchAndOpBegin(PetscSF sf, MPI_Datatype unit, void *rootdata, const void *leafdata, void *leafupdate, MPI_Op op)
{
  PetscMemType rootmtype, leafmtype, leafupdatemtype;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sf, PETSCSF_CLASSID, 1);
  PetscCall(PetscSFSetUp(sf));
  PetscCall(PetscLogEventBegin(PETSCSF_FetchAndOpBegin, sf, 0, 0, 0));
  PetscCall(PetscGetMemType(rootdata, &rootmtype));
  PetscCall(PetscGetMemType(leafdata, &leafmtype));
  PetscCall(PetscGetMemType(leafupdate, &leafupdatemtype));
  PetscCheck(leafmtype == leafupdatemtype, PETSC_COMM_SELF, PETSC_ERR_SUP, "No support for leafdata and leafupdate in different memory types");
  PetscUseTypeMethod(sf, FetchAndOpBegin, unit, rootmtype, rootdata, leafmtype, leafdata, leafupdate, op);
  PetscCall(PetscLogEventEnd(PETSCSF_FetchAndOpBegin, sf, 0, 0, 0));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscSFFetchAndOpWithMemTypeBegin - begin operation with explicit memory types that fetches values from root and updates atomically by
  applying operation using my leaf value, to be completed with `PetscSFFetchAndOpEnd()`

  Collective

  Input Parameters:
+ sf              - star forest
. unit            - data type
. rootmtype       - memory type of rootdata
. leafmtype       - memory type of leafdata
. leafdata        - leaf values to use in reduction
. leafupdatemtype - memory type of leafupdate
- op              - operation to use for reduction

  Output Parameters:
+ rootdata   - root values to be updated, input state is seen by first process to perform an update
- leafupdate - state at each leaf's respective root immediately prior to my atomic update

  Level: advanced

  Note:
  See `PetscSFFetchAndOpBegin()` for more details.

.seealso: `PetscSF`, `PetscSFFetchAndOpBegin()`, `PetscSFComputeDegreeBegin()`, `PetscSFReduceBegin()`, `PetscSFSetGraph()`, `PetscSFFetchAndOpEnd()`
@*/
PetscErrorCode PetscSFFetchAndOpWithMemTypeBegin(PetscSF sf, MPI_Datatype unit, PetscMemType rootmtype, void *rootdata, PetscMemType leafmtype, const void *leafdata, PetscMemType leafupdatemtype, void *leafupdate, MPI_Op op)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(sf, PETSCSF_CLASSID, 1);
  PetscCall(PetscSFSetUp(sf));
  PetscCall(PetscLogEventBegin(PETSCSF_FetchAndOpBegin, sf, 0, 0, 0));
  PetscCheck(leafmtype == leafupdatemtype, PETSC_COMM_SELF, PETSC_ERR_SUP, "No support for leafdata and leafupdate in different memory types");
  PetscUseTypeMethod(sf, FetchAndOpBegin, unit, rootmtype, rootdata, leafmtype, leafdata, leafupdate, op);
  PetscCall(PetscLogEventEnd(PETSCSF_FetchAndOpBegin, sf, 0, 0, 0));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscSFFetchAndOpEnd - end operation started in matching call to `PetscSFFetchAndOpBegin()` or `PetscSFFetchAndOpWithMemTypeBegin()`
  to fetch values from roots and update atomically by applying operation using my leaf value

  Collective

  Input Parameters:
+ sf       - star forest
. unit     - data type
. leafdata - leaf values to use in reduction
- op       - operation to use for reduction

  Output Parameters:
+ rootdata   - root values to be updated, input state is seen by first process to perform an update
- leafupdate - state at each leaf's respective root immediately prior to my atomic update

  Level: advanced

.seealso: `PetscSF`, `PetscSFComputeDegreeEnd()`, `PetscSFReduceEnd()`, `PetscSFSetGraph()`, `PetscSFFetchAndOpBegin()`, `PetscSFFetchAndOpWithMemTypeBegin()`
@*/
PetscErrorCode PetscSFFetchAndOpEnd(PetscSF sf, MPI_Datatype unit, void *rootdata, const void *leafdata, void *leafupdate, MPI_Op op)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(sf, PETSCSF_CLASSID, 1);
  PetscCall(PetscLogEventBegin(PETSCSF_FetchAndOpEnd, sf, 0, 0, 0));
  PetscUseTypeMethod(sf, FetchAndOpEnd, unit, rootdata, leafdata, leafupdate, op);
  PetscCall(PetscLogEventEnd(PETSCSF_FetchAndOpEnd, sf, 0, 0, 0));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscSFComputeDegreeBegin - begin computation of degree for each root vertex, to be completed with `PetscSFComputeDegreeEnd()`

  Collective

  Input Parameter:
. sf - star forest

  Output Parameter:
. degree - degree of each root vertex

  Level: advanced

  Note:
  The returned array is owned by `PetscSF` and automatically freed by `PetscSFDestroy()`. Hence there is no need to call `PetscFree()` on it.

.seealso: `PetscSF`, `PetscSFGatherBegin()`, `PetscSFComputeDegreeEnd()`
@*/
PetscErrorCode PetscSFComputeDegreeBegin(PetscSF sf, const PetscInt *degree[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(sf, PETSCSF_CLASSID, 1);
  PetscSFCheckGraphSet(sf, 1);
  PetscAssertPointer(degree, 2);
  if (!sf->degreeknown) {
    PetscInt i, nroots = sf->nroots, maxlocal;
    PetscCheck(!sf->degree, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Calls to PetscSFComputeDegreeBegin() cannot be nested.");
    maxlocal = sf->maxleaf - sf->minleaf + 1;
    PetscCall(PetscMalloc1(nroots, &sf->degree));
    PetscCall(PetscMalloc1(PetscMax(maxlocal, 1), &sf->degreetmp)); /* allocate at least one entry, see check in PetscSFComputeDegreeEnd() */
    for (i = 0; i < nroots; i++) sf->degree[i] = 0;
    for (i = 0; i < maxlocal; i++) sf->degreetmp[i] = 1;
    PetscCall(PetscSFReduceBegin(sf, MPIU_INT, sf->degreetmp - sf->minleaf, sf->degree, MPI_SUM));
  }
  *degree = NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscSFComputeDegreeEnd - complete computation of degree for each root vertex, started with `PetscSFComputeDegreeBegin()`

  Collective

  Input Parameter:
. sf - star forest

  Output Parameter:
. degree - degree of each root vertex

  Level: developer

  Note:
  The returned array is owned by `PetscSF` and automatically freed by `PetscSFDestroy()`. Hence there is no need to call `PetscFree()` on it.

.seealso: `PetscSF`, `PetscSFGatherBegin()`, `PetscSFComputeDegreeBegin()`
@*/
PetscErrorCode PetscSFComputeDegreeEnd(PetscSF sf, const PetscInt **degree)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(sf, PETSCSF_CLASSID, 1);
  PetscSFCheckGraphSet(sf, 1);
  PetscAssertPointer(degree, 2);
  if (!sf->degreeknown) {
    PetscCheck(sf->degreetmp, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Must call PetscSFComputeDegreeBegin() before PetscSFComputeDegreeEnd()");
    PetscCall(PetscSFReduceEnd(sf, MPIU_INT, sf->degreetmp - sf->minleaf, sf->degree, MPI_SUM));
    PetscCall(PetscFree(sf->degreetmp));
    sf->degreeknown = PETSC_TRUE;
  }
  *degree = sf->degree;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscSFComputeMultiRootOriginalNumbering - Returns original numbering of multi-roots (roots of multi-`PetscSF` returned by `PetscSFGetMultiSF()`).
  Each multi-root is assigned index of the corresponding original root.

  Collective

  Input Parameters:
+ sf     - star forest
- degree - degree of each root vertex, computed with `PetscSFComputeDegreeBegin()`/`PetscSFComputeDegreeEnd()`

  Output Parameters:
+ nMultiRoots             - (optional) number of multi-roots (roots of multi-`PetscSF`)
- multiRootsOrigNumbering - original indices of multi-roots; length of this array is `nMultiRoots`

  Level: developer

  Note:
  The returned array `multiRootsOrigNumbering` is newly allocated and should be destroyed with `PetscFree()` when no longer needed.

.seealso: `PetscSF`, `PetscSFComputeDegreeBegin()`, `PetscSFComputeDegreeEnd()`, `PetscSFGetMultiSF()`
@*/
PetscErrorCode PetscSFComputeMultiRootOriginalNumbering(PetscSF sf, const PetscInt degree[], PetscInt *nMultiRoots, PetscInt *multiRootsOrigNumbering[])
{
  PetscSF  msf;
  PetscInt k = 0, nroots, nmroots;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sf, PETSCSF_CLASSID, 1);
  PetscCall(PetscSFGetGraph(sf, &nroots, NULL, NULL, NULL));
  if (nroots) PetscAssertPointer(degree, 2);
  if (nMultiRoots) PetscAssertPointer(nMultiRoots, 3);
  PetscAssertPointer(multiRootsOrigNumbering, 4);
  PetscCall(PetscSFGetMultiSF(sf, &msf));
  PetscCall(PetscSFGetGraph(msf, &nmroots, NULL, NULL, NULL));
  PetscCall(PetscMalloc1(nmroots, multiRootsOrigNumbering));
  for (PetscInt i = 0; i < nroots; i++) {
    if (!degree[i]) continue;
    for (PetscInt j = 0; j < degree[i]; j++, k++) (*multiRootsOrigNumbering)[k] = i;
  }
  PetscCheck(k == nmroots, PETSC_COMM_SELF, PETSC_ERR_PLIB, "sanity check fail");
  if (nMultiRoots) *nMultiRoots = nmroots;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscSFGatherBegin - begin pointwise gather of all leaves into multi-roots, to be completed with `PetscSFGatherEnd()`

  Collective

  Input Parameters:
+ sf       - star forest
. unit     - data type
- leafdata - leaf data to gather to roots

  Output Parameter:
. multirootdata - root buffer to gather into, amount of space per root is equal to its degree

  Level: intermediate

.seealso: `PetscSF`, `PetscSFComputeDegreeBegin()`, `PetscSFScatterBegin()`
@*/
PetscErrorCode PetscSFGatherBegin(PetscSF sf, MPI_Datatype unit, const void *leafdata, void *multirootdata)
{
  PetscSF multi = NULL;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sf, PETSCSF_CLASSID, 1);
  PetscCall(PetscSFSetUp(sf));
  PetscCall(PetscSFGetMultiSF(sf, &multi));
  PetscCall(PetscSFReduceBegin(multi, unit, leafdata, multirootdata, MPI_REPLACE));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscSFGatherEnd - ends pointwise gather operation that was started with `PetscSFGatherBegin()`

  Collective

  Input Parameters:
+ sf       - star forest
. unit     - data type
- leafdata - leaf data to gather to roots

  Output Parameter:
. multirootdata - root buffer to gather into, amount of space per root is equal to its degree

  Level: intermediate

.seealso: `PetscSF`, `PetscSFComputeDegreeEnd()`, `PetscSFScatterEnd()`
@*/
PetscErrorCode PetscSFGatherEnd(PetscSF sf, MPI_Datatype unit, const void *leafdata, void *multirootdata)
{
  PetscSF multi = NULL;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sf, PETSCSF_CLASSID, 1);
  PetscCall(PetscSFGetMultiSF(sf, &multi));
  PetscCall(PetscSFReduceEnd(multi, unit, leafdata, multirootdata, MPI_REPLACE));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscSFScatterBegin - begin pointwise scatter operation from multi-roots to leaves, to be completed with `PetscSFScatterEnd()`

  Collective

  Input Parameters:
+ sf            - star forest
. unit          - data type
- multirootdata - root buffer to send to each leaf, one unit of data per leaf

  Output Parameter:
. leafdata - leaf data to be update with personal data from each respective root

  Level: intermediate

.seealso: `PetscSF`, `PetscSFComputeDegreeBegin()`, `PetscSFScatterEnd()`
@*/
PetscErrorCode PetscSFScatterBegin(PetscSF sf, MPI_Datatype unit, const void *multirootdata, void *leafdata)
{
  PetscSF multi = NULL;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sf, PETSCSF_CLASSID, 1);
  PetscCall(PetscSFSetUp(sf));
  PetscCall(PetscSFGetMultiSF(sf, &multi));
  PetscCall(PetscSFBcastBegin(multi, unit, multirootdata, leafdata, MPI_REPLACE));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscSFScatterEnd - ends pointwise scatter operation that was started with `PetscSFScatterBegin()`

  Collective

  Input Parameters:
+ sf            - star forest
. unit          - data type
- multirootdata - root buffer to send to each leaf, one unit of data per leaf

  Output Parameter:
. leafdata - leaf data to be update with personal data from each respective root

  Level: intermediate

.seealso: `PetscSF`, `PetscSFComputeDegreeEnd()`, `PetscSFScatterBegin()`
@*/
PetscErrorCode PetscSFScatterEnd(PetscSF sf, MPI_Datatype unit, const void *multirootdata, void *leafdata)
{
  PetscSF multi = NULL;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sf, PETSCSF_CLASSID, 1);
  PetscCall(PetscSFGetMultiSF(sf, &multi));
  PetscCall(PetscSFBcastEnd(multi, unit, multirootdata, leafdata, MPI_REPLACE));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscSFCheckLeavesUnique_Private(PetscSF sf)
{
  PetscInt        i, n, nleaves;
  const PetscInt *ilocal = NULL;
  PetscHSetI      seen;

  PetscFunctionBegin;
  if (PetscDefined(USE_DEBUG)) {
    PetscCall(PetscSFGetGraph(sf, NULL, &nleaves, &ilocal, NULL));
    PetscCall(PetscHSetICreate(&seen));
    for (i = 0; i < nleaves; i++) {
      const PetscInt leaf = ilocal ? ilocal[i] : i;
      PetscCall(PetscHSetIAdd(seen, leaf));
    }
    PetscCall(PetscHSetIGetSize(seen, &n));
    PetscCheck(n == nleaves, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Provided leaves have repeated values: all leaves must be unique");
    PetscCall(PetscHSetIDestroy(&seen));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscSFCompose - Compose a new `PetscSF` by putting the second `PetscSF` under the first one in a top (roots) down (leaves) view

  Input Parameters:
+ sfA - The first `PetscSF`
- sfB - The second `PetscSF`

  Output Parameter:
. sfBA - The composite `PetscSF`

  Level: developer

  Notes:
  Currently, the two `PetscSF`s must be defined on congruent communicators and they must be true star
  forests, i.e. the same leaf is not connected with different roots.

  `sfA`'s leaf space and `sfB`'s root space might be partially overlapped. The composition builds
  a graph with `sfA`'s roots and `sfB`'s leaves only when there is a path between them. Unconnected
  nodes (roots or leaves) are not in `sfBA`. Doing a `PetscSFBcastBegin()`/`PetscSFBcastEnd()` on the new `PetscSF` is equivalent to doing a
  `PetscSFBcastBegin()`/`PetscSFBcastEnd()` on `sfA`, then a `PetscSFBcastBegin()`/`PetscSFBcastEnd()` on `sfB`, on connected nodes.

.seealso: `PetscSF`, `PetscSFComposeInverse()`, `PetscSFGetGraph()`, `PetscSFSetGraph()`
@*/
PetscErrorCode PetscSFCompose(PetscSF sfA, PetscSF sfB, PetscSF *sfBA)
{
  const PetscSFNode *remotePointsA, *remotePointsB;
  PetscSFNode       *remotePointsBA = NULL, *reorderedRemotePointsA = NULL, *leafdataB;
  const PetscInt    *localPointsA, *localPointsB;
  PetscInt          *localPointsBA;
  PetscInt           i, numRootsA, numLeavesA, numRootsB, numLeavesB, minleaf, maxleaf, numLeavesBA;
  PetscBool          denseB;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sfA, PETSCSF_CLASSID, 1);
  PetscSFCheckGraphSet(sfA, 1);
  PetscValidHeaderSpecific(sfB, PETSCSF_CLASSID, 2);
  PetscSFCheckGraphSet(sfB, 2);
  PetscCheckSameComm(sfA, 1, sfB, 2);
  PetscAssertPointer(sfBA, 3);
  PetscCall(PetscSFCheckLeavesUnique_Private(sfA));
  PetscCall(PetscSFCheckLeavesUnique_Private(sfB));

  PetscCall(PetscSFGetGraph(sfA, &numRootsA, &numLeavesA, &localPointsA, &remotePointsA));
  PetscCall(PetscSFGetGraph(sfB, &numRootsB, &numLeavesB, &localPointsB, &remotePointsB));
  /* Make sure that PetscSFBcast{Begin, End}(sfB, ...) works with root data of size
     numRootsB; otherwise, garbage will be broadcasted.
     Example (comm size = 1):
     sfA: 0 <- (0, 0)
     sfB: 100 <- (0, 0)
          101 <- (0, 1)
     Here, we have remotePointsA = [(0, 0)], but for remotePointsA to be a valid tartget
     of sfB, it has to be recasted as [(0, 0), (-1, -1)] so that points 100 and 101 would
     receive (0, 0) and (-1, -1), respectively, when PetscSFBcast(sfB, ...) is called on
     remotePointsA; if not recasted, point 101 would receive a garbage value.             */
  PetscCall(PetscMalloc1(numRootsB, &reorderedRemotePointsA));
  for (i = 0; i < numRootsB; i++) {
    reorderedRemotePointsA[i].rank  = -1;
    reorderedRemotePointsA[i].index = -1;
  }
  for (i = 0; i < numLeavesA; i++) {
    PetscInt localp = localPointsA ? localPointsA[i] : i;

    if (localp >= numRootsB) continue;
    reorderedRemotePointsA[localp] = remotePointsA[i];
  }
  remotePointsA = reorderedRemotePointsA;
  PetscCall(PetscSFGetLeafRange(sfB, &minleaf, &maxleaf));
  PetscCall(PetscMalloc1(maxleaf - minleaf + 1, &leafdataB));
  for (i = 0; i < maxleaf - minleaf + 1; i++) {
    leafdataB[i].rank  = -1;
    leafdataB[i].index = -1;
  }
  PetscCall(PetscSFBcastBegin(sfB, MPIU_SF_NODE, remotePointsA, PetscSafePointerPlusOffset(leafdataB, -minleaf), MPI_REPLACE));
  PetscCall(PetscSFBcastEnd(sfB, MPIU_SF_NODE, remotePointsA, PetscSafePointerPlusOffset(leafdataB, -minleaf), MPI_REPLACE));
  PetscCall(PetscFree(reorderedRemotePointsA));

  denseB = (PetscBool)!localPointsB;
  for (i = 0, numLeavesBA = 0; i < numLeavesB; i++) {
    if (leafdataB[localPointsB ? localPointsB[i] - minleaf : i].rank == -1) denseB = PETSC_FALSE;
    else numLeavesBA++;
  }
  if (denseB) {
    localPointsBA  = NULL;
    remotePointsBA = leafdataB;
  } else {
    PetscCall(PetscMalloc1(numLeavesBA, &localPointsBA));
    PetscCall(PetscMalloc1(numLeavesBA, &remotePointsBA));
    for (i = 0, numLeavesBA = 0; i < numLeavesB; i++) {
      const PetscInt l = localPointsB ? localPointsB[i] : i;

      if (leafdataB[l - minleaf].rank == -1) continue;
      remotePointsBA[numLeavesBA] = leafdataB[l - minleaf];
      localPointsBA[numLeavesBA]  = l;
      numLeavesBA++;
    }
    PetscCall(PetscFree(leafdataB));
  }
  PetscCall(PetscSFCreate(PetscObjectComm((PetscObject)sfA), sfBA));
  PetscCall(PetscSFSetFromOptions(*sfBA));
  PetscCall(PetscSFSetGraph(*sfBA, numRootsA, numLeavesBA, localPointsBA, PETSC_OWN_POINTER, remotePointsBA, PETSC_OWN_POINTER));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscSFComposeInverse - Compose a new `PetscSF` by putting the inverse of the second `PetscSF` under the first one

  Input Parameters:
+ sfA - The first `PetscSF`
- sfB - The second `PetscSF`

  Output Parameter:
. sfBA - The composite `PetscSF`.

  Level: developer

  Notes:
  Currently, the two `PetscSF`s must be defined on congruent communicators and they must be true star
  forests, i.e. the same leaf is not connected with different roots. Even more, all roots of the
  second `PetscSF` must have a degree of 1, i.e., no roots have more than one leaf connected.

  `sfA`'s leaf space and `sfB`'s leaf space might be partially overlapped. The composition builds
  a graph with `sfA`'s roots and `sfB`'s roots only when there is a path between them. Unconnected
  roots are not in `sfBA`. Doing a `PetscSFBcastBegin()`/`PetscSFBcastEnd()` on the new `PetscSF` is equivalent to doing a `PetscSFBcastBegin()`/`PetscSFBcastEnd()`
  on `sfA`, then
  a `PetscSFReduceBegin()`/`PetscSFReduceEnd()` on `sfB`, on connected roots.

.seealso: `PetscSF`, `PetscSFCompose()`, `PetscSFGetGraph()`, `PetscSFSetGraph()`, `PetscSFCreateInverseSF()`
@*/
PetscErrorCode PetscSFComposeInverse(PetscSF sfA, PetscSF sfB, PetscSF *sfBA)
{
  const PetscSFNode *remotePointsA, *remotePointsB;
  PetscSFNode       *remotePointsBA;
  const PetscInt    *localPointsA, *localPointsB;
  PetscSFNode       *reorderedRemotePointsA = NULL;
  PetscInt           i, numRootsA, numLeavesA, numLeavesBA, numRootsB, numLeavesB, minleaf, maxleaf, *localPointsBA;
  MPI_Op             op;
#if defined(PETSC_USE_64BIT_INDICES)
  PetscBool iswin;
#endif

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sfA, PETSCSF_CLASSID, 1);
  PetscSFCheckGraphSet(sfA, 1);
  PetscValidHeaderSpecific(sfB, PETSCSF_CLASSID, 2);
  PetscSFCheckGraphSet(sfB, 2);
  PetscCheckSameComm(sfA, 1, sfB, 2);
  PetscAssertPointer(sfBA, 3);
  PetscCall(PetscSFCheckLeavesUnique_Private(sfA));
  PetscCall(PetscSFCheckLeavesUnique_Private(sfB));

  PetscCall(PetscSFGetGraph(sfA, &numRootsA, &numLeavesA, &localPointsA, &remotePointsA));
  PetscCall(PetscSFGetGraph(sfB, &numRootsB, &numLeavesB, &localPointsB, &remotePointsB));

  /* TODO: Check roots of sfB have degree of 1 */
  /* Once we implement it, we can replace the MPI_MAXLOC
     with MPI_REPLACE. In that case, MPI_MAXLOC and MPI_REPLACE have the same effect.
     We use MPI_MAXLOC only to have a deterministic output from this routine if
     the root condition is not meet.
   */
  op = MPI_MAXLOC;
#if defined(PETSC_USE_64BIT_INDICES)
  /* we accept a non-deterministic output (if any) with PETSCSFWINDOW, since MPI_MAXLOC cannot operate on MPIU_2INT with MPI_Accumulate */
  PetscCall(PetscObjectTypeCompare((PetscObject)sfB, PETSCSFWINDOW, &iswin));
  if (iswin) op = MPI_REPLACE;
#endif

  PetscCall(PetscSFGetLeafRange(sfB, &minleaf, &maxleaf));
  PetscCall(PetscMalloc1(maxleaf - minleaf + 1, &reorderedRemotePointsA));
  for (i = 0; i < maxleaf - minleaf + 1; i++) {
    reorderedRemotePointsA[i].rank  = -1;
    reorderedRemotePointsA[i].index = -1;
  }
  if (localPointsA) {
    for (i = 0; i < numLeavesA; i++) {
      if (localPointsA[i] > maxleaf || localPointsA[i] < minleaf) continue;
      reorderedRemotePointsA[localPointsA[i] - minleaf] = remotePointsA[i];
    }
  } else {
    for (i = 0; i < numLeavesA; i++) {
      if (i > maxleaf || i < minleaf) continue;
      reorderedRemotePointsA[i - minleaf] = remotePointsA[i];
    }
  }

  PetscCall(PetscMalloc1(numRootsB, &localPointsBA));
  PetscCall(PetscMalloc1(numRootsB, &remotePointsBA));
  for (i = 0; i < numRootsB; i++) {
    remotePointsBA[i].rank  = -1;
    remotePointsBA[i].index = -1;
  }

  PetscCall(PetscSFReduceBegin(sfB, MPIU_SF_NODE, PetscSafePointerPlusOffset(reorderedRemotePointsA, -minleaf), remotePointsBA, op));
  PetscCall(PetscSFReduceEnd(sfB, MPIU_SF_NODE, PetscSafePointerPlusOffset(reorderedRemotePointsA, -minleaf), remotePointsBA, op));
  PetscCall(PetscFree(reorderedRemotePointsA));
  for (i = 0, numLeavesBA = 0; i < numRootsB; i++) {
    if (remotePointsBA[i].rank == -1) continue;
    remotePointsBA[numLeavesBA].rank  = remotePointsBA[i].rank;
    remotePointsBA[numLeavesBA].index = remotePointsBA[i].index;
    localPointsBA[numLeavesBA]        = i;
    numLeavesBA++;
  }
  PetscCall(PetscSFCreate(PetscObjectComm((PetscObject)sfA), sfBA));
  PetscCall(PetscSFSetFromOptions(*sfBA));
  PetscCall(PetscSFSetGraph(*sfBA, numRootsA, numLeavesBA, localPointsBA, PETSC_OWN_POINTER, remotePointsBA, PETSC_OWN_POINTER));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  PetscSFCreateLocalSF_Private - Creates a local `PetscSF` that only has intra-process edges of the global `PetscSF`

  Input Parameter:
. sf - The global `PetscSF`

  Output Parameter:
. out - The local `PetscSF`

.seealso: `PetscSF`, `PetscSFCreate()`
 */
PetscErrorCode PetscSFCreateLocalSF_Private(PetscSF sf, PetscSF *out)
{
  MPI_Comm           comm;
  PetscMPIInt        myrank;
  const PetscInt    *ilocal;
  const PetscSFNode *iremote;
  PetscInt           i, j, nroots, nleaves, lnleaves, *lilocal;
  PetscSFNode       *liremote;
  PetscSF            lsf;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sf, PETSCSF_CLASSID, 1);
  if (sf->ops->CreateLocalSF) PetscUseTypeMethod(sf, CreateLocalSF, out);
  else {
    /* Could use PetscSFCreateEmbeddedLeafSF, but since we know the comm is PETSC_COMM_SELF, we can make it fast */
    PetscCall(PetscObjectGetComm((PetscObject)sf, &comm));
    PetscCallMPI(MPI_Comm_rank(comm, &myrank));

    /* Find out local edges and build a local SF */
    PetscCall(PetscSFGetGraph(sf, &nroots, &nleaves, &ilocal, &iremote));
    for (i = lnleaves = 0; i < nleaves; i++) {
      if (iremote[i].rank == (PetscInt)myrank) lnleaves++;
    }
    PetscCall(PetscMalloc1(lnleaves, &lilocal));
    PetscCall(PetscMalloc1(lnleaves, &liremote));

    for (i = j = 0; i < nleaves; i++) {
      if (iremote[i].rank == (PetscInt)myrank) {
        lilocal[j]        = ilocal ? ilocal[i] : i; /* ilocal=NULL for contiguous storage */
        liremote[j].rank  = 0;                      /* rank in PETSC_COMM_SELF */
        liremote[j].index = iremote[i].index;
        j++;
      }
    }
    PetscCall(PetscSFCreate(PETSC_COMM_SELF, &lsf));
    PetscCall(PetscSFSetFromOptions(lsf));
    PetscCall(PetscSFSetGraph(lsf, nroots, lnleaves, lilocal, PETSC_OWN_POINTER, liremote, PETSC_OWN_POINTER));
    PetscCall(PetscSFSetUp(lsf));
    *out = lsf;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Similar to PetscSFBcast, but only Bcast to leaves on rank 0 */
PetscErrorCode PetscSFBcastToZero_Private(PetscSF sf, MPI_Datatype unit, const void *rootdata, void *leafdata)
{
  PetscMemType rootmtype, leafmtype;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sf, PETSCSF_CLASSID, 1);
  PetscCall(PetscSFSetUp(sf));
  PetscCall(PetscLogEventBegin(PETSCSF_BcastBegin, sf, 0, 0, 0));
  PetscCall(PetscGetMemType(rootdata, &rootmtype));
  PetscCall(PetscGetMemType(leafdata, &leafmtype));
  PetscUseTypeMethod(sf, BcastToZero, unit, rootmtype, rootdata, leafmtype, leafdata);
  PetscCall(PetscLogEventEnd(PETSCSF_BcastBegin, sf, 0, 0, 0));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscSFConcatenate - concatenate multiple `PetscSF` into one

  Input Parameters:
+ comm        - the communicator
. nsfs        - the number of input `PetscSF`
. sfs         - the array of input `PetscSF`
. rootMode    - the root mode specifying how roots are handled
- leafOffsets - the array of local leaf offsets, one for each input `PetscSF`, or `NULL` for contiguous storage

  Output Parameter:
. newsf - The resulting `PetscSF`

  Level: advanced

  Notes:
  The communicator of all `PetscSF`s in `sfs` must be comm.

  Leaves are always concatenated locally, keeping them ordered by the input `PetscSF` index and original local order.

  The offsets in `leafOffsets` are added to the original leaf indices.

  If all input SFs use contiguous leaf storage (`ilocal` = `NULL`), `leafOffsets` can be passed as `NULL` as well.
  In this case, `NULL` is also passed as `ilocal` to the resulting `PetscSF`.

  If any input `PetscSF` has non-null `ilocal`, `leafOffsets` is needed to distinguish leaves from different input `PetscSF`s.
  In this case, user is responsible to provide correct offsets so that the resulting leaves are unique (otherwise an error occurs).

  All root modes retain the essential connectivity condition.
  If two leaves of the same input `PetscSF` are connected (sharing the same root), they are also connected in the output `PetscSF`.
  Parameter `rootMode` controls how the input root spaces are combined.
  For `PETSCSF_CONCATENATE_ROOTMODE_SHARED`, the root space is considered the same for each input `PetscSF` (checked in debug mode)
  and is also the same in the output `PetscSF`.
  For `PETSCSF_CONCATENATE_ROOTMODE_LOCAL` and `PETSCSF_CONCATENATE_ROOTMODE_GLOBAL`, the input root spaces are taken as separate and joined.
  `PETSCSF_CONCATENATE_ROOTMODE_LOCAL` joins the root spaces locally;
  roots of sfs[0], sfs[1], sfs[2], ... are joined on each rank separately, ordered by input `PetscSF` and original local index, and renumbered contiguously.
  `PETSCSF_CONCATENATE_ROOTMODE_GLOBAL` joins the root spaces globally;
  roots of sfs[0], sfs[1], sfs[2], ... are joined globally, ordered by input `PetscSF` index and original global index, and renumbered contiguously;
  the original root ranks are ignored.
  For both `PETSCSF_CONCATENATE_ROOTMODE_LOCAL` and `PETSCSF_CONCATENATE_ROOTMODE_GLOBAL`,
  the output `PetscSF`'s root layout is such that the local number of roots is a sum of the input `PetscSF`'s local numbers of roots on each rank
  to keep the load balancing.
  However, for `PETSCSF_CONCATENATE_ROOTMODE_GLOBAL`, roots can move to different ranks.

  Example:
  We can use src/vec/is/sf/tests/ex18.c to compare the root modes. By running
.vb
  make -C $PETSC_DIR/src/vec/is/sf/tests ex18
  for m in {local,global,shared}; do
    mpirun -n 2 $PETSC_DIR/src/vec/is/sf/tests/ex18 -nsfs 2 -n 2 -root_mode $m -sf_view
  done
.ve
  we generate two identical `PetscSF`s sf_0 and sf_1,
.vb
  PetscSF Object: sf_0 2 MPI processes
    type: basic
    rank #leaves #roots
    [ 0]       4      2
    [ 1]       4      2
    leaves      roots       roots in global numbering
    ( 0,  0) <- ( 0,  0)  =   0
    ( 0,  1) <- ( 0,  1)  =   1
    ( 0,  2) <- ( 1,  0)  =   2
    ( 0,  3) <- ( 1,  1)  =   3
    ( 1,  0) <- ( 0,  0)  =   0
    ( 1,  1) <- ( 0,  1)  =   1
    ( 1,  2) <- ( 1,  0)  =   2
    ( 1,  3) <- ( 1,  1)  =   3
.ve
  and pass them to `PetscSFConcatenate()` along with different choices of `rootMode`, yielding different result_sf\:
.vb
  rootMode = local:
  PetscSF Object: result_sf 2 MPI processes
    type: basic
    rank #leaves #roots
    [ 0]       8      4
    [ 1]       8      4
    leaves      roots       roots in global numbering
    ( 0,  0) <- ( 0,  0)  =   0
    ( 0,  1) <- ( 0,  1)  =   1
    ( 0,  2) <- ( 1,  0)  =   4
    ( 0,  3) <- ( 1,  1)  =   5
    ( 0,  4) <- ( 0,  2)  =   2
    ( 0,  5) <- ( 0,  3)  =   3
    ( 0,  6) <- ( 1,  2)  =   6
    ( 0,  7) <- ( 1,  3)  =   7
    ( 1,  0) <- ( 0,  0)  =   0
    ( 1,  1) <- ( 0,  1)  =   1
    ( 1,  2) <- ( 1,  0)  =   4
    ( 1,  3) <- ( 1,  1)  =   5
    ( 1,  4) <- ( 0,  2)  =   2
    ( 1,  5) <- ( 0,  3)  =   3
    ( 1,  6) <- ( 1,  2)  =   6
    ( 1,  7) <- ( 1,  3)  =   7

  rootMode = global:
  PetscSF Object: result_sf 2 MPI processes
    type: basic
    rank #leaves #roots
    [ 0]       8      4
    [ 1]       8      4
    leaves      roots       roots in global numbering
    ( 0,  0) <- ( 0,  0)  =   0
    ( 0,  1) <- ( 0,  1)  =   1
    ( 0,  2) <- ( 0,  2)  =   2
    ( 0,  3) <- ( 0,  3)  =   3
    ( 0,  4) <- ( 1,  0)  =   4
    ( 0,  5) <- ( 1,  1)  =   5
    ( 0,  6) <- ( 1,  2)  =   6
    ( 0,  7) <- ( 1,  3)  =   7
    ( 1,  0) <- ( 0,  0)  =   0
    ( 1,  1) <- ( 0,  1)  =   1
    ( 1,  2) <- ( 0,  2)  =   2
    ( 1,  3) <- ( 0,  3)  =   3
    ( 1,  4) <- ( 1,  0)  =   4
    ( 1,  5) <- ( 1,  1)  =   5
    ( 1,  6) <- ( 1,  2)  =   6
    ( 1,  7) <- ( 1,  3)  =   7

  rootMode = shared:
  PetscSF Object: result_sf 2 MPI processes
    type: basic
    rank #leaves #roots
    [ 0]       8      2
    [ 1]       8      2
    leaves      roots       roots in global numbering
    ( 0,  0) <- ( 0,  0)  =   0
    ( 0,  1) <- ( 0,  1)  =   1
    ( 0,  2) <- ( 1,  0)  =   2
    ( 0,  3) <- ( 1,  1)  =   3
    ( 0,  4) <- ( 0,  0)  =   0
    ( 0,  5) <- ( 0,  1)  =   1
    ( 0,  6) <- ( 1,  0)  =   2
    ( 0,  7) <- ( 1,  1)  =   3
    ( 1,  0) <- ( 0,  0)  =   0
    ( 1,  1) <- ( 0,  1)  =   1
    ( 1,  2) <- ( 1,  0)  =   2
    ( 1,  3) <- ( 1,  1)  =   3
    ( 1,  4) <- ( 0,  0)  =   0
    ( 1,  5) <- ( 0,  1)  =   1
    ( 1,  6) <- ( 1,  0)  =   2
    ( 1,  7) <- ( 1,  1)  =   3
.ve

.seealso: `PetscSF`, `PetscSFCompose()`, `PetscSFGetGraph()`, `PetscSFSetGraph()`, `PetscSFConcatenateRootMode`
@*/
PetscErrorCode PetscSFConcatenate(MPI_Comm comm, PetscInt nsfs, PetscSF sfs[], PetscSFConcatenateRootMode rootMode, PetscInt leafOffsets[], PetscSF *newsf)
{
  PetscInt     i, s, nLeaves, nRoots;
  PetscInt    *leafArrayOffsets;
  PetscInt    *ilocal_new;
  PetscSFNode *iremote_new;
  PetscBool    all_ilocal_null = PETSC_FALSE;
  PetscLayout  glayout         = NULL;
  PetscInt    *gremote         = NULL;
  PetscMPIInt  rank, size;

  PetscFunctionBegin;
  if (PetscDefined(USE_DEBUG)) {
    PetscSF dummy; /* just to have a PetscObject on comm for input validation */

    PetscCall(PetscSFCreate(comm, &dummy));
    PetscValidLogicalCollectiveInt(dummy, nsfs, 2);
    PetscAssertPointer(sfs, 3);
    for (i = 0; i < nsfs; i++) {
      PetscValidHeaderSpecific(sfs[i], PETSCSF_CLASSID, 3);
      PetscCheckSameComm(dummy, 1, sfs[i], 3);
    }
    PetscValidLogicalCollectiveEnum(dummy, rootMode, 4);
    if (leafOffsets) PetscAssertPointer(leafOffsets, 5);
    PetscAssertPointer(newsf, 6);
    PetscCall(PetscSFDestroy(&dummy));
  }
  if (!nsfs) {
    PetscCall(PetscSFCreate(comm, newsf));
    PetscCall(PetscSFSetGraph(*newsf, 0, 0, NULL, PETSC_OWN_POINTER, NULL, PETSC_OWN_POINTER));
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  PetscCallMPI(MPI_Comm_size(comm, &size));

  /* Calculate leaf array offsets */
  PetscCall(PetscMalloc1(nsfs + 1, &leafArrayOffsets));
  leafArrayOffsets[0] = 0;
  for (s = 0; s < nsfs; s++) {
    PetscInt nl;

    PetscCall(PetscSFGetGraph(sfs[s], NULL, &nl, NULL, NULL));
    leafArrayOffsets[s + 1] = leafArrayOffsets[s] + nl;
  }
  nLeaves = leafArrayOffsets[nsfs];

  /* Calculate number of roots */
  switch (rootMode) {
  case PETSCSF_CONCATENATE_ROOTMODE_SHARED: {
    PetscCall(PetscSFGetGraph(sfs[0], &nRoots, NULL, NULL, NULL));
    if (PetscDefined(USE_DEBUG)) {
      for (s = 1; s < nsfs; s++) {
        PetscInt nr;

        PetscCall(PetscSFGetGraph(sfs[s], &nr, NULL, NULL, NULL));
        PetscCheck(nr == nRoots, comm, PETSC_ERR_ARG_SIZ, "rootMode = %s but sfs[%" PetscInt_FMT "] has a different number of roots (%" PetscInt_FMT ") than sfs[0] (%" PetscInt_FMT ")", PetscSFConcatenateRootModes[rootMode], s, nr, nRoots);
      }
    }
  } break;
  case PETSCSF_CONCATENATE_ROOTMODE_GLOBAL: {
    /* Calculate also global layout in this case */
    PetscInt    *nls;
    PetscLayout *lts;
    PetscInt   **inds;
    PetscInt     j;
    PetscInt     rootOffset = 0;

    PetscCall(PetscCalloc3(nsfs, &lts, nsfs, &nls, nsfs, &inds));
    PetscCall(PetscLayoutCreate(comm, &glayout));
    glayout->bs = 1;
    glayout->n  = 0;
    glayout->N  = 0;
    for (s = 0; s < nsfs; s++) {
      PetscCall(PetscSFGetGraphLayout(sfs[s], &lts[s], &nls[s], NULL, &inds[s]));
      glayout->n += lts[s]->n;
      glayout->N += lts[s]->N;
    }
    PetscCall(PetscLayoutSetUp(glayout));
    PetscCall(PetscMalloc1(nLeaves, &gremote));
    for (s = 0, j = 0; s < nsfs; s++) {
      for (i = 0; i < nls[s]; i++, j++) gremote[j] = inds[s][i] + rootOffset;
      rootOffset += lts[s]->N;
      PetscCall(PetscLayoutDestroy(&lts[s]));
      PetscCall(PetscFree(inds[s]));
    }
    PetscCall(PetscFree3(lts, nls, inds));
    nRoots = glayout->N;
  } break;
  case PETSCSF_CONCATENATE_ROOTMODE_LOCAL:
    /* nRoots calculated later in this case */
    break;
  default:
    SETERRQ(comm, PETSC_ERR_ARG_WRONG, "Invalid PetscSFConcatenateRootMode %d", rootMode);
  }

  if (!leafOffsets) {
    all_ilocal_null = PETSC_TRUE;
    for (s = 0; s < nsfs; s++) {
      const PetscInt *ilocal;

      PetscCall(PetscSFGetGraph(sfs[s], NULL, NULL, &ilocal, NULL));
      if (ilocal) {
        all_ilocal_null = PETSC_FALSE;
        break;
      }
    }
    PetscCheck(all_ilocal_null, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "leafOffsets can be passed as NULL only if all SFs have ilocal = NULL");
  }

  /* Renumber and concatenate local leaves */
  ilocal_new = NULL;
  if (!all_ilocal_null) {
    PetscCall(PetscMalloc1(nLeaves, &ilocal_new));
    for (i = 0; i < nLeaves; i++) ilocal_new[i] = -1;
    for (s = 0; s < nsfs; s++) {
      const PetscInt *ilocal;
      PetscInt       *ilocal_l = PetscSafePointerPlusOffset(ilocal_new, leafArrayOffsets[s]);
      PetscInt        i, nleaves_l;

      PetscCall(PetscSFGetGraph(sfs[s], NULL, &nleaves_l, &ilocal, NULL));
      for (i = 0; i < nleaves_l; i++) ilocal_l[i] = (ilocal ? ilocal[i] : i) + leafOffsets[s];
    }
  }

  /* Renumber and concatenate remote roots */
  if (rootMode == PETSCSF_CONCATENATE_ROOTMODE_LOCAL || rootMode == PETSCSF_CONCATENATE_ROOTMODE_SHARED) {
    PetscInt rootOffset = 0;

    PetscCall(PetscMalloc1(nLeaves, &iremote_new));
    for (i = 0; i < nLeaves; i++) {
      iremote_new[i].rank  = -1;
      iremote_new[i].index = -1;
    }
    for (s = 0; s < nsfs; s++) {
      PetscInt           i, nl, nr;
      PetscSF            tmp_sf;
      const PetscSFNode *iremote;
      PetscSFNode       *tmp_rootdata;
      PetscSFNode       *tmp_leafdata = PetscSafePointerPlusOffset(iremote_new, leafArrayOffsets[s]);

      PetscCall(PetscSFGetGraph(sfs[s], &nr, &nl, NULL, &iremote));
      PetscCall(PetscSFCreate(comm, &tmp_sf));
      /* create helper SF with contiguous leaves */
      PetscCall(PetscSFSetGraph(tmp_sf, nr, nl, NULL, PETSC_USE_POINTER, (PetscSFNode *)iremote, PETSC_COPY_VALUES));
      PetscCall(PetscSFSetUp(tmp_sf));
      PetscCall(PetscMalloc1(nr, &tmp_rootdata));
      if (rootMode == PETSCSF_CONCATENATE_ROOTMODE_LOCAL) {
        for (i = 0; i < nr; i++) {
          tmp_rootdata[i].index = i + rootOffset;
          tmp_rootdata[i].rank  = rank;
        }
        rootOffset += nr;
      } else {
        for (i = 0; i < nr; i++) {
          tmp_rootdata[i].index = i;
          tmp_rootdata[i].rank  = rank;
        }
      }
      PetscCall(PetscSFBcastBegin(tmp_sf, MPIU_SF_NODE, tmp_rootdata, tmp_leafdata, MPI_REPLACE));
      PetscCall(PetscSFBcastEnd(tmp_sf, MPIU_SF_NODE, tmp_rootdata, tmp_leafdata, MPI_REPLACE));
      PetscCall(PetscSFDestroy(&tmp_sf));
      PetscCall(PetscFree(tmp_rootdata));
    }
    if (rootMode == PETSCSF_CONCATENATE_ROOTMODE_LOCAL) nRoots = rootOffset; // else nRoots already calculated above

    /* Build the new SF */
    PetscCall(PetscSFCreate(comm, newsf));
    PetscCall(PetscSFSetGraph(*newsf, nRoots, nLeaves, ilocal_new, PETSC_OWN_POINTER, iremote_new, PETSC_OWN_POINTER));
  } else {
    /* Build the new SF */
    PetscCall(PetscSFCreate(comm, newsf));
    PetscCall(PetscSFSetGraphLayout(*newsf, glayout, nLeaves, ilocal_new, PETSC_OWN_POINTER, gremote));
  }
  PetscCall(PetscSFSetUp(*newsf));
  PetscCall(PetscSFViewFromOptions(*newsf, NULL, "-sf_concat_view"));
  PetscCall(PetscLayoutDestroy(&glayout));
  PetscCall(PetscFree(gremote));
  PetscCall(PetscFree(leafArrayOffsets));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscSFRegisterPersistent - Register root and leaf data as memory regions that will be used for repeated PetscSF communications.

  Collective

  Input Parameters:
+ sf       - star forest
. unit     - the data type contained within the root and leaf data
. rootdata - root data that will be used for multiple PetscSF communications
- leafdata - leaf data that will be used for multiple PetscSF communications

  Level: advanced

  Notes:
  Implementations of `PetscSF` can make optimizations
  for repeated communication using the same memory regions, but these optimizations
  can be unsound if `rootdata` or `leafdata` is deallocated and the `PetscSF` is not informed.
  The intended pattern is

.vb
  PetscMalloc2(nroots, &rootdata, nleaves, &leafdata);

  PetscSFRegisterPersistent(sf, unit, rootdata, leafdata);
  // repeated use of rootdata and leafdata will now be optimized

  PetscSFBcastBegin(sf, unit, rootdata, leafdata, MPI_REPLACE);
  PetscSFBcastEnd(sf, unit, rootdata, leafdata, MPI_REPLACE);
  // ...
  PetscSFReduceBegin(sf, unit, leafdata, rootdata, MPI_SUM);
  PetscSFReduceEnd(sf, unit, leafdata, rootdata, MPI_SUM);
  // ... (other communications)

  // rootdata and leafdata must be deregistered before freeing
  // skipping this can lead to undefined behavior including
  // deadlocks
  PetscSFDeregisterPersistent(sf, unit, rootdata, leafdata);

  // it is now safe to free rootdata and leafdata
  PetscFree2(rootdata, leafdata);
.ve

  If you do not register `rootdata` and `leafdata` it will not cause an error,
  but optimizations that reduce the setup time for each communication cannot be
  made.  Currently, the only implementation of `PetscSF` that benefits from
  `PetscSFRegisterPersistent()` is `PETSCSFWINDOW`.  For the default
  `PETSCSFBASIC` there is no benefit to using `PetscSFRegisterPersistent()`.

.seealso: `PetscSF`, `PETSCSFWINDOW`, `PetscSFDeregisterPersistent()`
@*/
PetscErrorCode PetscSFRegisterPersistent(PetscSF sf, MPI_Datatype unit, const void *rootdata, const void *leafdata)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(sf, PETSCSF_CLASSID, 1);
  PetscTryMethod(sf, "PetscSFRegisterPersistent_C", (PetscSF, MPI_Datatype, const void *, const void *), (sf, unit, rootdata, leafdata));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscSFDeregisterPersistent - Signal that repeated usage of root and leaf data for PetscSF communication has concluded.

  Collective

  Input Parameters:
+ sf       - star forest
. unit     - the data type contained within the root and leaf data
. rootdata - root data that was previously registered with `PetscSFRegisterPersistent()`
- leafdata - leaf data that was previously registered with `PetscSFRegisterPersistent()`

  Level: advanced

  Note:
  See `PetscSFRegisterPersistent()` for when/how to use this function.

.seealso: `PetscSF`, `PETSCSFWINDOW`, `PetscSFRegisterPersistent()`
@*/
PetscErrorCode PetscSFDeregisterPersistent(PetscSF sf, MPI_Datatype unit, const void *rootdata, const void *leafdata)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(sf, PETSCSF_CLASSID, 1);
  PetscTryMethod(sf, "PetscSFDeregisterPersistent_C", (PetscSF, MPI_Datatype, const void *, const void *), (sf, unit, rootdata, leafdata));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode PetscSFGetDatatypeSize_Internal(MPI_Comm comm, MPI_Datatype unit, MPI_Aint *size)
{
  MPI_Aint lb, lb_true, bytes, bytes_true;

  PetscFunctionBegin;
  PetscCallMPI(MPI_Type_get_extent(unit, &lb, &bytes));
  PetscCallMPI(MPI_Type_get_true_extent(unit, &lb_true, &bytes_true));
  PetscCheck(lb == 0 && lb_true == 0, comm, PETSC_ERR_SUP, "No support for unit type with nonzero lower bound, write petsc-maint@mcs.anl.gov if you want this feature");
  *size = bytes;
  PetscFunctionReturn(PETSC_SUCCESS);
}
