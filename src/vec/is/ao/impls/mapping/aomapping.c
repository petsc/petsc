
/*
  These AO application ordering routines do not require that the input
  be a permutation, but merely a 1-1 mapping. This implementation still
  keeps the entire ordering on each processor.
*/

#include <../src/vec/is/ao/aoimpl.h>          /*I  "petscao.h" I*/

typedef struct {
  PetscInt N;
  PetscInt *app;       /* app[i] is the partner for petsc[appPerm[i]] */
  PetscInt *appPerm;
  PetscInt *petsc;     /* petsc[j] is the partner for app[petscPerm[j]] */
  PetscInt *petscPerm;
} AO_Mapping;

PetscErrorCode AODestroy_Mapping(AO ao)
{
  AO_Mapping     *aomap = (AO_Mapping*) ao->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFree4(aomap->app,aomap->appPerm,aomap->petsc,aomap->petscPerm);CHKERRQ(ierr);
  ierr = PetscFree(aomap);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode AOView_Mapping(AO ao, PetscViewer viewer)
{
  AO_Mapping     *aomap = (AO_Mapping*) ao->data;
  PetscMPIInt    rank;
  PetscInt       i;
  PetscBool      iascii;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(PetscObjectComm((PetscObject)ao), &rank);CHKERRMPI(ierr);
  if (rank) PetscFunctionReturn(0);
  ierr = PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERASCII, &iascii);CHKERRQ(ierr);
  if (iascii) {
    PetscViewerASCIIPrintf(viewer, "Number of elements in ordering %" PetscInt_FMT "\n", aomap->N);
    PetscViewerASCIIPrintf(viewer, "   App.   PETSc\n");
    for (i = 0; i < aomap->N; i++) {
      PetscViewerASCIIPrintf(viewer, "%" PetscInt_FMT "   %" PetscInt_FMT "    %" PetscInt_FMT "\n", i, aomap->app[i], aomap->petsc[aomap->appPerm[i]]);
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode AOPetscToApplication_Mapping(AO ao, PetscInt n, PetscInt *ia)
{
  AO_Mapping *aomap = (AO_Mapping*) ao->data;
  PetscInt   *app   = aomap->app;
  PetscInt   *petsc = aomap->petsc;
  PetscInt   *perm  = aomap->petscPerm;
  PetscInt   N      = aomap->N;
  PetscInt   low, high, mid=0;
  PetscInt   idex;
  PetscInt   i;

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
      mid = (low + high)/2;
      if (idex == petsc[mid]) break;
      else if (idex < petsc[mid]) high = mid - 1;
      else low = mid + 1;
    }
    if (low > high) ia[i] = -1;
    else            ia[i] = app[perm[mid]];
  }
  PetscFunctionReturn(0);
}

PetscErrorCode AOApplicationToPetsc_Mapping(AO ao, PetscInt n, PetscInt *ia)
{
  AO_Mapping *aomap = (AO_Mapping*) ao->data;
  PetscInt   *app   = aomap->app;
  PetscInt   *petsc = aomap->petsc;
  PetscInt   *perm  = aomap->appPerm;
  PetscInt   N      = aomap->N;
  PetscInt   low, high, mid=0;
  PetscInt   idex;
  PetscInt   i;

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
      mid = (low + high)/2;
      if (idex == app[mid]) break;
      else if (idex < app[mid]) high = mid - 1;
      else low = mid + 1;
    }
    if (low > high) ia[i] = -1;
    else            ia[i] = petsc[perm[mid]];
  }
  PetscFunctionReturn(0);
}

static struct _AOOps AOps = {
  PetscDesignatedInitializer(view,AOView_Mapping),
  PetscDesignatedInitializer(destroy,AODestroy_Mapping),
  PetscDesignatedInitializer(petsctoapplication,AOPetscToApplication_Mapping),
  PetscDesignatedInitializer(applicationtopetsc,AOApplicationToPetsc_Mapping),
};

/*@C
  AOMappingHasApplicationIndex - Searches for the supplied application index.

  Input Parameters:
+ ao       - The AOMapping
- index    - The application index

  Output Parameter:
. hasIndex - Flag is PETSC_TRUE if the index exists

  Level: intermediate

.seealso: AOMappingHasPetscIndex(), AOCreateMapping()
@*/
PetscErrorCode  AOMappingHasApplicationIndex(AO ao, PetscInt idex, PetscBool  *hasIndex)
{
  AO_Mapping *aomap;
  PetscInt   *app;
  PetscInt   low, high, mid;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ao, AO_CLASSID,1);
  PetscValidPointer(hasIndex,3);
  aomap = (AO_Mapping*) ao->data;
  app   = aomap->app;
  /* Use bisection since the array is sorted */
  low  = 0;
  high = aomap->N - 1;
  while (low <= high) {
    mid = (low + high)/2;
    if (idex == app[mid]) break;
    else if (idex < app[mid]) high = mid - 1;
    else low = mid + 1;
  }
  if (low > high) *hasIndex = PETSC_FALSE;
  else *hasIndex = PETSC_TRUE;
  PetscFunctionReturn(0);
}

/*@
  AOMappingHasPetscIndex - Searches for the supplied petsc index.

  Input Parameters:
+ ao       - The AOMapping
- index    - The petsc index

  Output Parameter:
. hasIndex - Flag is PETSC_TRUE if the index exists

  Level: intermediate

.seealso: AOMappingHasApplicationIndex(), AOCreateMapping()
@*/
PetscErrorCode  AOMappingHasPetscIndex(AO ao, PetscInt idex, PetscBool  *hasIndex)
{
  AO_Mapping *aomap;
  PetscInt   *petsc;
  PetscInt   low, high, mid;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ao, AO_CLASSID,1);
  PetscValidPointer(hasIndex,3);
  aomap = (AO_Mapping*) ao->data;
  petsc = aomap->petsc;
  /* Use bisection since the array is sorted */
  low  = 0;
  high = aomap->N - 1;
  while (low <= high) {
    mid = (low + high)/2;
    if (idex == petsc[mid]) break;
    else if (idex < petsc[mid]) high = mid - 1;
    else low = mid + 1;
  }
  if (low > high) *hasIndex = PETSC_FALSE;
  else *hasIndex = PETSC_TRUE;
  PetscFunctionReturn(0);
}

/*@C
  AOCreateMapping - Creates a basic application mapping using two integer arrays.

  Input Parameters:
+ comm    - MPI communicator that is to share AO
. napp    - size of integer arrays
. myapp   - integer array that defines an ordering
- mypetsc - integer array that defines another ordering (may be NULL to indicate the identity ordering)

  Output Parameter:
. aoout   - the new application mapping

  Options Database Key:
. -ao_view : call AOView() at the conclusion of AOCreateMapping()

  Level: beginner

    Notes:
    the arrays myapp and mypetsc need NOT contain the all the integers 0 to napp-1, that is there CAN be "holes"  in the indices.
       Use AOCreateBasic() or AOCreateBasicIS() if they do not have holes for better performance.

.seealso: AOCreateBasic(), AOCreateBasic(), AOCreateMappingIS(), AODestroy()
@*/
PetscErrorCode  AOCreateMapping(MPI_Comm comm,PetscInt napp,const PetscInt myapp[],const PetscInt mypetsc[],AO *aoout)
{
  AO             ao;
  AO_Mapping     *aomap;
  PetscInt       *allpetsc,  *allapp;
  PetscInt       *petscPerm, *appPerm;
  PetscInt       *petsc;
  PetscMPIInt    size, rank,*lens, *disp,nnapp;
  PetscInt       N, start;
  PetscInt       i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidPointer(aoout,5);
  *aoout = NULL;
  ierr = AOInitializePackage();CHKERRQ(ierr);

  ierr     = PetscHeaderCreate(ao, AO_CLASSID, "AO", "Application Ordering", "AO", comm, AODestroy, AOView);CHKERRQ(ierr);
  ierr     = PetscNewLog(ao,&aomap);CHKERRQ(ierr);
  ierr     = PetscMemcpy(ao->ops, &AOps, sizeof(AOps));CHKERRQ(ierr);
  ao->data = (void*) aomap;

  /* transmit all lengths to all processors */
  ierr  = MPI_Comm_size(comm, &size);CHKERRMPI(ierr);
  ierr  = MPI_Comm_rank(comm, &rank);CHKERRMPI(ierr);
  ierr  = PetscMalloc2(size, &lens,size,&disp);CHKERRQ(ierr);
  nnapp = napp;
  ierr  = MPI_Allgather(&nnapp, 1, MPI_INT, lens, 1, MPI_INT, comm);CHKERRMPI(ierr);
  N     = 0;
  for (i = 0; i < size; i++) {
    disp[i] = N;
    N      += lens[i];
  }
  aomap->N = N;
  ao->N    = N;
  ao->n    = N;

  /* If mypetsc is 0 then use "natural" numbering */
  if (!mypetsc) {
    start = disp[rank];
    ierr  = PetscMalloc1(napp+1, &petsc);CHKERRQ(ierr);
    for (i = 0; i < napp; i++) petsc[i] = start + i;
  } else {
    petsc = (PetscInt*)mypetsc;
  }

  /* get all indices on all processors */
  ierr = PetscMalloc4(N, &allapp,N,&appPerm,N,&allpetsc,N,&petscPerm);CHKERRQ(ierr);
  ierr = MPI_Allgatherv((void*)myapp, napp, MPIU_INT, allapp,   lens, disp, MPIU_INT, comm);CHKERRMPI(ierr);
  ierr = MPI_Allgatherv((void*)petsc, napp, MPIU_INT, allpetsc, lens, disp, MPIU_INT, comm);CHKERRMPI(ierr);
  ierr = PetscFree2(lens,disp);CHKERRQ(ierr);

  /* generate a list of application and PETSc node numbers */
  ierr = PetscMalloc4(N, &aomap->app,N,&aomap->appPerm,N,&aomap->petsc,N,&aomap->petscPerm);CHKERRQ(ierr);
  ierr = PetscLogObjectMemory((PetscObject)ao, 4*N * sizeof(PetscInt));CHKERRQ(ierr);
  for (i = 0; i < N; i++) {
    appPerm[i]   = i;
    petscPerm[i] = i;
  }
  ierr = PetscSortIntWithPermutation(N, allpetsc, petscPerm);CHKERRQ(ierr);
  ierr = PetscSortIntWithPermutation(N, allapp,   appPerm);CHKERRQ(ierr);
  /* Form sorted arrays of indices */
  for (i = 0; i < N; i++) {
    aomap->app[i]   = allapp[appPerm[i]];
    aomap->petsc[i] = allpetsc[petscPerm[i]];
  }
  /* Invert petscPerm[] into aomap->petscPerm[] */
  for (i = 0; i < N; i++) aomap->petscPerm[petscPerm[i]] = i;

  /* Form map between aomap->app[] and aomap->petsc[] */
  for (i = 0; i < N; i++) aomap->appPerm[i] = aomap->petscPerm[appPerm[i]];

  /* Invert appPerm[] into allapp[] */
  for (i = 0; i < N; i++) allapp[appPerm[i]] = i;

  /* Form map between aomap->petsc[] and aomap->app[] */
  for (i = 0; i < N; i++) aomap->petscPerm[i] = allapp[petscPerm[i]];

  if (PetscDefined(USE_DEBUG)) {
    /* Check that the permutations are complementary */
    for (i = 0; i < N; i++) {
      PetscCheckFalse(i != aomap->appPerm[aomap->petscPerm[i]],PETSC_COMM_SELF,PETSC_ERR_PLIB, "Invalid ordering");
    }
  }
  /* Cleanup */
  if (!mypetsc) {
    ierr = PetscFree(petsc);CHKERRQ(ierr);
  }
  ierr = PetscFree4(allapp,appPerm,allpetsc,petscPerm);CHKERRQ(ierr);

  ierr = AOViewFromOptions(ao,NULL,"-ao_view");CHKERRQ(ierr);

  *aoout = ao;
  PetscFunctionReturn(0);
}

/*@
  AOCreateMappingIS - Creates a basic application ordering using two index sets.

  Input Parameters:
+ comm    - MPI communicator that is to share AO
. isapp   - index set that defines an ordering
- ispetsc - index set that defines another ordering, maybe NULL for identity IS

  Output Parameter:
. aoout   - the new application ordering

  Options Database Key:
. -ao_view : call AOView() at the conclusion of AOCreateMappingIS()

  Level: beginner

    Notes:
    the index sets isapp and ispetsc need NOT contain the all the integers 0 to N-1, that is there CAN be "holes"  in the indices.
       Use AOCreateBasic() or AOCreateBasicIS() if they do not have holes for better performance.

.seealso: AOCreateBasic(), AOCreateMapping(), AODestroy()
@*/
PetscErrorCode  AOCreateMappingIS(IS isapp, IS ispetsc, AO *aoout)
{
  MPI_Comm       comm;
  const PetscInt *mypetsc, *myapp;
  PetscInt       napp, npetsc;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject) isapp, &comm);CHKERRQ(ierr);
  ierr = ISGetLocalSize(isapp, &napp);CHKERRQ(ierr);
  if (ispetsc) {
    ierr = ISGetLocalSize(ispetsc, &npetsc);CHKERRQ(ierr);
    PetscCheckFalse(napp != npetsc,PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ, "Local IS lengths must match");
    ierr = ISGetIndices(ispetsc, &mypetsc);CHKERRQ(ierr);
  } else {
    mypetsc = NULL;
  }
  ierr = ISGetIndices(isapp, &myapp);CHKERRQ(ierr);

  ierr = AOCreateMapping(comm, napp, myapp, mypetsc, aoout);CHKERRQ(ierr);

  ierr = ISRestoreIndices(isapp, &myapp);CHKERRQ(ierr);
  if (ispetsc) {
    ierr = ISRestoreIndices(ispetsc, &mypetsc);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
