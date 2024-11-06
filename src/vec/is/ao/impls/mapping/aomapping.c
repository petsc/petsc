/*
  These AO application ordering routines do not require that the input
  be a permutation, but merely a 1-1 mapping. This implementation still
  keeps the entire ordering on each processor.
*/

#include <../src/vec/is/ao/aoimpl.h> /*I  "petscao.h" I*/

typedef struct {
  PetscInt  N;
  PetscInt *app; /* app[i] is the partner for petsc[appPerm[i]] */
  PetscInt *appPerm;
  PetscInt *petsc; /* petsc[j] is the partner for app[petscPerm[j]] */
  PetscInt *petscPerm;
} AO_Mapping;

static PetscErrorCode AODestroy_Mapping(AO ao)
{
  AO_Mapping *aomap = (AO_Mapping *)ao->data;

  PetscFunctionBegin;
  PetscCall(PetscFree4(aomap->app, aomap->appPerm, aomap->petsc, aomap->petscPerm));
  PetscCall(PetscFree(aomap));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode AOView_Mapping(AO ao, PetscViewer viewer)
{
  AO_Mapping *aomap = (AO_Mapping *)ao->data;
  PetscMPIInt rank;
  PetscInt    i;
  PetscBool   iascii;

  PetscFunctionBegin;
  PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)ao), &rank));
  if (rank) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &iascii));
  if (iascii) {
    PetscCall(PetscViewerASCIIPrintf(viewer, "Number of elements in ordering %" PetscInt_FMT "\n", aomap->N));
    PetscCall(PetscViewerASCIIPrintf(viewer, "   App.   PETSc\n"));
    for (i = 0; i < aomap->N; i++) PetscCall(PetscViewerASCIIPrintf(viewer, "%" PetscInt_FMT "   %" PetscInt_FMT "    %" PetscInt_FMT "\n", i, aomap->app[i], aomap->petsc[aomap->appPerm[i]]));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode AOPetscToApplication_Mapping(AO ao, PetscInt n, PetscInt *ia)
{
  AO_Mapping *aomap = (AO_Mapping *)ao->data;
  PetscInt   *app   = aomap->app;
  PetscInt   *petsc = aomap->petsc;
  PetscInt   *perm  = aomap->petscPerm;
  PetscInt    N     = aomap->N;
  PetscInt    low, high, mid = 0;
  PetscInt    idex;
  PetscInt    i;

  /* It would be possible to use a single bisection search, which
     recursively divided the indices to be converted, and searched
     partitions which contained an index. This would result in
     better running times if indices are clustered.
  */
  PetscFunctionBegin;
  for (i = 0; i < n; i++) {
    idex = ia[i];
    if (idex < 0) continue;
    /* Use bisection since the array is sorted */
    low  = 0;
    high = N - 1;
    while (low <= high) {
      mid = (low + high) / 2;
      if (idex == petsc[mid]) break;
      else if (idex < petsc[mid]) high = mid - 1;
      else low = mid + 1;
    }
    if (low > high) ia[i] = -1;
    else ia[i] = app[perm[mid]];
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode AOApplicationToPetsc_Mapping(AO ao, PetscInt n, PetscInt *ia)
{
  AO_Mapping *aomap = (AO_Mapping *)ao->data;
  PetscInt   *app   = aomap->app;
  PetscInt   *petsc = aomap->petsc;
  PetscInt   *perm  = aomap->appPerm;
  PetscInt    N     = aomap->N;
  PetscInt    low, high, mid = 0;
  PetscInt    idex;
  PetscInt    i;

  /* It would be possible to use a single bisection search, which
     recursively divided the indices to be converted, and searched
     partitions which contained an index. This would result in
     better running times if indices are clustered.
  */
  PetscFunctionBegin;
  for (i = 0; i < n; i++) {
    idex = ia[i];
    if (idex < 0) continue;
    /* Use bisection since the array is sorted */
    low  = 0;
    high = N - 1;
    while (low <= high) {
      mid = (low + high) / 2;
      if (idex == app[mid]) break;
      else if (idex < app[mid]) high = mid - 1;
      else low = mid + 1;
    }
    if (low > high) ia[i] = -1;
    else ia[i] = petsc[perm[mid]];
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static const struct _AOOps AOps = {
  PetscDesignatedInitializer(view, AOView_Mapping),
  PetscDesignatedInitializer(destroy, AODestroy_Mapping),
  PetscDesignatedInitializer(petsctoapplication, AOPetscToApplication_Mapping),
  PetscDesignatedInitializer(applicationtopetsc, AOApplicationToPetsc_Mapping),
  PetscDesignatedInitializer(petsctoapplicationpermuteint, NULL),
  PetscDesignatedInitializer(applicationtopetscpermuteint, NULL),
  PetscDesignatedInitializer(petsctoapplicationpermutereal, NULL),
  PetscDesignatedInitializer(applicationtopetscpermutereal, NULL),
};

/*@
  AOMappingHasApplicationIndex - Checks if an `AO` has a requested application index.

  Not Collective

  Input Parameters:
+ ao   - The `AO`
- idex - The application index

  Output Parameter:
. hasIndex - Flag is `PETSC_TRUE` if the index exists

  Level: intermediate

  Developer Notes:
  The name of the function is wrong, it should be `AOHasApplicationIndex`

.seealso: [](sec_ao), `AOMappingHasPetscIndex()`, `AOCreateMapping()`, `AO`
@*/
PetscErrorCode AOMappingHasApplicationIndex(AO ao, PetscInt idex, PetscBool *hasIndex)
{
  AO_Mapping *aomap;
  PetscInt   *app;
  PetscInt    low, high, mid;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ao, AO_CLASSID, 1);
  PetscAssertPointer(hasIndex, 3);
  aomap = (AO_Mapping *)ao->data;
  app   = aomap->app;
  /* Use bisection since the array is sorted */
  low  = 0;
  high = aomap->N - 1;
  while (low <= high) {
    mid = (low + high) / 2;
    if (idex == app[mid]) break;
    else if (idex < app[mid]) high = mid - 1;
    else low = mid + 1;
  }
  if (low > high) *hasIndex = PETSC_FALSE;
  else *hasIndex = PETSC_TRUE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  AOMappingHasPetscIndex - checks if an `AO` has a requested petsc index.

  Not Collective

  Input Parameters:
+ ao   - The `AO`
- idex - The petsc index

  Output Parameter:
. hasIndex - Flag is `PETSC_TRUE` if the index exists

  Level: intermediate

  Developer Notes:
  The name of the function is wrong, it should be `AOHasPetscIndex`

.seealso: [](sec_ao), `AOMappingHasApplicationIndex()`, `AOCreateMapping()`
@*/
PetscErrorCode AOMappingHasPetscIndex(AO ao, PetscInt idex, PetscBool *hasIndex)
{
  AO_Mapping *aomap;
  PetscInt   *petsc;
  PetscInt    low, high, mid;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ao, AO_CLASSID, 1);
  PetscAssertPointer(hasIndex, 3);
  aomap = (AO_Mapping *)ao->data;
  petsc = aomap->petsc;
  /* Use bisection since the array is sorted */
  low  = 0;
  high = aomap->N - 1;
  while (low <= high) {
    mid = (low + high) / 2;
    if (idex == petsc[mid]) break;
    else if (idex < petsc[mid]) high = mid - 1;
    else low = mid + 1;
  }
  if (low > high) *hasIndex = PETSC_FALSE;
  else *hasIndex = PETSC_TRUE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  AOCreateMapping - Creates an application mapping using two integer arrays.

  Input Parameters:
+ comm    - MPI communicator that is to share the `AO`
. napp    - size of integer arrays
. myapp   - integer array that defines an ordering
- mypetsc - integer array that defines another ordering (may be `NULL` to indicate the identity ordering)

  Output Parameter:
. aoout - the new application mapping

  Options Database Key:
. -ao_view - call `AOView()` at the conclusion of `AOCreateMapping()`

  Level: beginner

  Note:
  The arrays `myapp` and `mypetsc` need NOT contain the all the integers 0 to `napp`-1, that is there CAN be "holes"  in the indices.
  Use `AOCreateBasic()` or `AOCreateBasicIS()` if they do not have holes for better performance.

.seealso: [](sec_ao), `AOCreateBasic()`, `AOCreateMappingIS()`, `AODestroy()`
@*/
PetscErrorCode AOCreateMapping(MPI_Comm comm, PetscInt napp, const PetscInt myapp[], const PetscInt mypetsc[], AO *aoout)
{
  AO          ao;
  AO_Mapping *aomap;
  PetscInt   *allpetsc, *allapp;
  PetscInt   *petscPerm, *appPerm;
  PetscInt   *petsc;
  PetscMPIInt size, rank, *lens, *disp, nnapp;
  PetscCount  N;
  PetscInt    start;

  PetscFunctionBegin;
  PetscAssertPointer(aoout, 5);
  *aoout = NULL;
  PetscCall(AOInitializePackage());

  PetscCall(PetscHeaderCreate(ao, AO_CLASSID, "AO", "Application Ordering", "AO", comm, AODestroy, AOView));
  PetscCall(PetscNew(&aomap));
  ao->ops[0] = AOps;
  ao->data   = (void *)aomap;

  /* transmit all lengths to all processors */
  PetscCallMPI(MPI_Comm_size(comm, &size));
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  PetscCall(PetscMalloc2(size, &lens, size, &disp));
  PetscCall(PetscMPIIntCast(napp, &nnapp));
  PetscCallMPI(MPI_Allgather(&nnapp, 1, MPI_INT, lens, 1, MPI_INT, comm));
  N = 0;
  for (PetscMPIInt i = 0; i < size; i++) {
    PetscCall(PetscMPIIntCast(N, &disp[i]));
    N += lens[i];
  }
  PetscCall(PetscIntCast(N, &aomap->N));
  ao->N = (PetscInt)N;
  ao->n = (PetscInt)N;

  /* If mypetsc is 0 then use "natural" numbering */
  if (!mypetsc) {
    start = disp[rank];
    PetscCall(PetscMalloc1(napp + 1, &petsc));
    for (PetscInt i = 0; i < napp; i++) petsc[i] = start + i;
  } else {
    petsc = (PetscInt *)mypetsc;
  }

  /* get all indices on all processors */
  PetscCall(PetscMalloc4(N, &allapp, N, &appPerm, N, &allpetsc, N, &petscPerm));
  PetscCallMPI(MPI_Allgatherv((void *)myapp, nnapp, MPIU_INT, allapp, lens, disp, MPIU_INT, comm));
  PetscCallMPI(MPI_Allgatherv((void *)petsc, nnapp, MPIU_INT, allpetsc, lens, disp, MPIU_INT, comm));
  PetscCall(PetscFree2(lens, disp));

  /* generate a list of application and PETSc node numbers */
  PetscCall(PetscMalloc4(N, &aomap->app, N, &aomap->appPerm, N, &aomap->petsc, N, &aomap->petscPerm));
  for (PetscInt i = 0; i < N; i++) {
    appPerm[i]   = i;
    petscPerm[i] = i;
  }
  PetscCall(PetscSortIntWithPermutation((PetscInt)N, allpetsc, petscPerm));
  PetscCall(PetscSortIntWithPermutation((PetscInt)N, allapp, appPerm));
  /* Form sorted arrays of indices */
  for (PetscInt i = 0; i < N; i++) {
    aomap->app[i]   = allapp[appPerm[i]];
    aomap->petsc[i] = allpetsc[petscPerm[i]];
  }
  /* Invert petscPerm[] into aomap->petscPerm[] */
  for (PetscInt i = 0; i < N; i++) aomap->petscPerm[petscPerm[i]] = i;

  /* Form map between aomap->app[] and aomap->petsc[] */
  for (PetscInt i = 0; i < N; i++) aomap->appPerm[i] = aomap->petscPerm[appPerm[i]];

  /* Invert appPerm[] into allapp[] */
  for (PetscInt i = 0; i < N; i++) allapp[appPerm[i]] = i;

  /* Form map between aomap->petsc[] and aomap->app[] */
  for (PetscInt i = 0; i < N; i++) aomap->petscPerm[i] = allapp[petscPerm[i]];

  if (PetscDefined(USE_DEBUG)) {
    /* Check that the permutations are complementary */
    for (PetscInt i = 0; i < N; i++) PetscCheck(i == aomap->appPerm[aomap->petscPerm[i]], PETSC_COMM_SELF, PETSC_ERR_PLIB, "Invalid ordering");
  }
  /* Cleanup */
  if (!mypetsc) PetscCall(PetscFree(petsc));
  PetscCall(PetscFree4(allapp, appPerm, allpetsc, petscPerm));
  PetscCall(AOViewFromOptions(ao, NULL, "-ao_view"));
  *aoout = ao;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  AOCreateMappingIS - Creates an application mapping using two index sets.

  Input Parameters:
+ isapp   - index set that defines an ordering
- ispetsc - index set that defines another ordering, maybe `NULL` for identity `IS`

  Output Parameter:
. aoout - the new application ordering

  Options Database Key:
. -ao_view - call `AOView()` at the conclusion of `AOCreateMappingIS()`

  Level: beginner

  Note:
  The index sets `isapp` and `ispetsc` need NOT contain the all the integers 0 to N-1, that is there CAN be "holes"  in the indices.
  Use `AOCreateBasic()` or `AOCreateBasicIS()` if they do not have holes for better performance.

.seealso: [](sec_ao), [](sec_scatter), `AOCreateBasic()`, `AOCreateMapping()`, `AODestroy()`
@*/
PetscErrorCode AOCreateMappingIS(IS isapp, IS ispetsc, AO *aoout)
{
  MPI_Comm        comm;
  const PetscInt *mypetsc, *myapp;
  PetscInt        napp, npetsc;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)isapp, &comm));
  PetscCall(ISGetLocalSize(isapp, &napp));
  if (ispetsc) {
    PetscCall(ISGetLocalSize(ispetsc, &npetsc));
    PetscCheck(napp == npetsc, PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Local IS lengths must match");
    PetscCall(ISGetIndices(ispetsc, &mypetsc));
  } else {
    mypetsc = NULL;
  }
  PetscCall(ISGetIndices(isapp, &myapp));

  PetscCall(AOCreateMapping(comm, napp, myapp, mypetsc, aoout));

  PetscCall(ISRestoreIndices(isapp, &myapp));
  if (ispetsc) PetscCall(ISRestoreIndices(ispetsc, &mypetsc));
  PetscFunctionReturn(PETSC_SUCCESS);
}
