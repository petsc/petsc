#define PETSCDM_DLL

/*
  These AO application ordering routines do not require that the input
  be a permutation, but merely a 1-1 mapping. This implementation still
  keeps the entire ordering on each processor.
*/

#include "src/dm/ao/aoimpl.h"
#include "petscsys.h"

typedef struct {
  PetscInt N;
  PetscInt *app;       /* app[i] is the partner for petsc[appPerm[i]] */
  PetscInt *appPerm;
  PetscInt *petsc;     /* petsc[j] is the partner for app[petscPerm[j]] */
  PetscInt *petscPerm;
} AO_Mapping;

#undef __FUNCT__  
#define __FUNCT__ "AODestroy_Mapping"
PetscErrorCode AODestroy_Mapping(AO ao)
{
  AO_Mapping     *aomap = (AO_Mapping *) ao->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFree(aomap->app);CHKERRQ(ierr);
  ierr = PetscFree(ao->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "AOView_Mapping"
PetscErrorCode AOView_Mapping(AO ao, PetscViewer viewer)
{
  AO_Mapping     *aomap = (AO_Mapping *) ao->data;
  PetscMPIInt    rank;
  PetscInt       i;
  PetscTruth     iascii;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(ao->comm, &rank);CHKERRQ(ierr);
  if (rank) PetscFunctionReturn(0);

  if (!viewer) {
    viewer = PETSC_VIEWER_STDOUT_SELF; 
  }

  ierr = PetscTypeCompare((PetscObject) viewer, PETSC_VIEWER_ASCII, &iascii);CHKERRQ(ierr);
  if (iascii) {
    PetscViewerASCIIPrintf(viewer, "Number of elements in ordering %D\n", aomap->N);
    PetscViewerASCIIPrintf(viewer, "   App.   PETSc\n");
    for(i = 0; i < aomap->N; i++) {
      PetscViewerASCIIPrintf(viewer, "%D   %D    %D\n", i, aomap->app[i], aomap->petsc[aomap->appPerm[i]]);
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "AOPetscToApplication_Mapping"
PetscErrorCode AOPetscToApplication_Mapping(AO ao, PetscInt n, PetscInt *ia)
{
  AO_Mapping *aomap = (AO_Mapping *) ao->data;
  PetscInt   *app   = aomap->app;
  PetscInt   *petsc = aomap->petsc;
  PetscInt   *perm  = aomap->petscPerm;
  PetscInt   N     = aomap->N;
  PetscInt   low, high, mid=0;
  PetscInt   idex;
  PetscInt   i;

  /* It would be possible to use a single bisection search, which
     recursively divided the indices to be converted, and searched
     partitions which contained an index. This would result in
     better running times if indices are clustered.
  */
  PetscFunctionBegin;
  for(i = 0; i < n; i++) {
    idex = ia[i];
    if (idex < 0) continue;
    /* Use bisection since the array is sorted */
    low  = 0;
    high = N - 1;
    while (low <= high) {
      mid = (low + high)/2;
      if (idex == petsc[mid]) {
        break;
      } else if (idex < petsc[mid]) {
        high = mid - 1;
      } else {
        low  = mid + 1;
      }
    }
    if (low > high) SETERRQ1(PETSC_ERR_ARG_OUTOFRANGE, "Invalid input index %D", idex);
    ia[i] = app[perm[mid]];
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "AOApplicationToPetsc_Mapping"
PetscErrorCode AOApplicationToPetsc_Mapping(AO ao, PetscInt n, PetscInt *ia)
{
  AO_Mapping *aomap = (AO_Mapping *) ao->data;
  PetscInt   *app   = aomap->app;
  PetscInt   *petsc = aomap->petsc;
  PetscInt   *perm  = aomap->appPerm;
  PetscInt   N     = aomap->N;
  PetscInt   low, high, mid=0;
  PetscInt   idex;
  PetscInt   i;

  /* It would be possible to use a single bisection search, which
     recursively divided the indices to be converted, and searched
     partitions which contained an index. This would result in
     better running times if indices are clustered.
  */
  PetscFunctionBegin;
  for(i = 0; i < n; i++) {
    idex = ia[i];
    if (idex < 0) continue;
    /* Use bisection since the array is sorted */
    low  = 0;
    high = N - 1;
    while (low <= high) {
      mid = (low + high)/2;
      if (idex == app[mid]) {
        break;
      } else if (idex < app[mid]) {
        high = mid - 1;
      } else {
        low  = mid + 1;
      }
    }
    if (low > high) SETERRQ1(PETSC_ERR_ARG_OUTOFRANGE, "Invalid input index %D", idex);
    ia[i] = petsc[perm[mid]];
  }
  PetscFunctionReturn(0);
}

static struct _AOOps AOps = {AOView_Mapping,
                             AODestroy_Mapping,
                             AOPetscToApplication_Mapping,
                             AOApplicationToPetsc_Mapping,
                             PETSC_NULL,
                             PETSC_NULL,
                             PETSC_NULL,
                             PETSC_NULL};

#undef __FUNCT__  
#define __FUNCT__ "AOMappingHasApplicationIndex"
/*@C
  AOMappingHasApplicationIndex - Searches for the supplied application index.

  Input Parameters:
+ ao       - The AOMapping
- index    - The application index

  Output Parameter:
. hasIndex - Flag is PETSC_TRUE if the index exists

  Level: intermediate

.keywords: AO, index
.seealso: AOMappingHasPetscIndex(), AOCreateMapping()
@*/
PetscErrorCode PETSCDM_DLLEXPORT AOMappingHasApplicationIndex(AO ao, PetscInt idex, PetscTruth *hasIndex)
{
  AO_Mapping *aomap;
  PetscInt   *app;
  PetscInt   low, high, mid;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ao, AO_COOKIE,1);
  PetscValidPointer(hasIndex,3);
  aomap = (AO_Mapping *) ao->data;
  app   = aomap->app;
  /* Use bisection since the array is sorted */
  low  = 0;
  high = aomap->N - 1;
  while (low <= high) {
    mid = (low + high)/2;
    if (idex == app[mid]) {
      break;
    } else if (idex < app[mid]) {
      high = mid - 1;
    } else {
      low  = mid + 1;
    }
  }
  if (low > high) {
    *hasIndex = PETSC_FALSE;
  } else {
    *hasIndex = PETSC_TRUE;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "AOMappingHasPetscIndex"
/*@C
  AOMappingHasPetscIndex - Searches for the supplied petsc index.

  Input Parameters:
+ ao       - The AOMapping
- index    - The petsc index

  Output Parameter:
. hasIndex - Flag is PETSC_TRUE if the index exists

  Level: intermediate

.keywords: AO, index
.seealso: AOMappingHasApplicationIndex(), AOCreateMapping()
@*/
PetscErrorCode PETSCDM_DLLEXPORT AOMappingHasPetscIndex(AO ao, PetscInt idex, PetscTruth *hasIndex)
{
  AO_Mapping *aomap;
  PetscInt   *petsc;
  PetscInt   low, high, mid;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ao, AO_COOKIE,1);
  PetscValidPointer(hasIndex,3);
  aomap = (AO_Mapping *) ao->data;
  petsc = aomap->petsc;
  /* Use bisection since the array is sorted */
  low  = 0;
  high = aomap->N - 1;
  while (low <= high) {
    mid = (low + high)/2;
    if (idex == petsc[mid]) {
      break;
    } else if (idex < petsc[mid]) {
      high = mid - 1;
    } else {
      low  = mid + 1;
    }
  }
  if (low > high) {
    *hasIndex = PETSC_FALSE;
  } else {
    *hasIndex = PETSC_TRUE;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "AOCreateMapping"
/*@C
  AOCreateMapping - Creates a basic application mapping using two integer arrays.

  Input Parameters:
+ comm    - MPI communicator that is to share AO
. napp    - size of integer arrays
. myapp   - integer array that defines an ordering
- mypetsc - integer array that defines another ordering

  Output Parameter:
. aoout   - the new application mapping

  Options Database Key:
$ -ao_view : call AOView() at the conclusion of AOCreateMapping()

  Level: beginner

.keywords: AO, create
.seealso: AOCreateDebug(), AOCreateBasic(), AOCreateMappingIS(), AODestroy()
@*/
PetscErrorCode PETSCDM_DLLEXPORT AOCreateMapping(MPI_Comm comm,PetscInt napp,const PetscInt myapp[],const PetscInt mypetsc[],AO *aoout)
{
  AO             ao;
  AO_Mapping     *aomap;
  PetscInt       *allpetsc,  *allapp;
  PetscInt       *petscPerm, *appPerm;
  PetscInt       *petsc;
  PetscMPIInt    size, rank,*lens, *disp,nnapp;
  PetscInt       N, start;
  PetscInt       i;
  PetscTruth     opt;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidPointer(aoout,5);
  *aoout = 0;
#ifndef PETSC_USE_DYNAMIC_LIBRARIES
  ierr = DMInitializePackage(PETSC_NULL);CHKERRQ(ierr);
#endif

  ierr = PetscHeaderCreate(ao, _p_AO, struct _AOOps, AO_COOKIE, AO_MAPPING, "AO", comm, AODestroy, AOView);CHKERRQ(ierr);
  ierr = PetscNew(AO_Mapping, &aomap);CHKERRQ(ierr);
  ierr = PetscLogObjectMemory(ao, sizeof(struct _p_AO) + sizeof(AO_Mapping));CHKERRQ(ierr);
  ierr = PetscMemcpy(ao->ops, &AOps, sizeof(AOps));CHKERRQ(ierr);
  ao->data = (void*) aomap;

  /* transmit all lengths to all processors */
  ierr  = MPI_Comm_size(comm, &size);CHKERRQ(ierr);
  ierr  = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
  ierr  = PetscMalloc(2*size * sizeof(PetscMPIInt), &lens);CHKERRQ(ierr);
  disp  = lens + size;
  nnapp = napp;
  ierr  = MPI_Allgather(&nnapp, 1, MPI_INT, lens, 1, MPI_INT, comm);CHKERRQ(ierr);
  N    = 0;
  for(i = 0; i < size; i++) {
    disp[i] = N;
    N += lens[i];
  }
  aomap->N = N;
  ao->N    = N;
  ao->n    = N;

  /* If mypetsc is 0 then use "natural" numbering */
  if (!mypetsc) {
    start = disp[rank];
    ierr  = PetscMalloc((napp+1) * sizeof(PetscInt), &petsc);CHKERRQ(ierr);
    for(i = 0; i < napp; i++) {
      petsc[i] = start + i;
    }
  } else {
    petsc = (PetscInt*)mypetsc;
  }

  /* get all indices on all processors */
  ierr = PetscMalloc(N*4 * sizeof(PetscInt), &allapp);CHKERRQ(ierr);
  appPerm   = allapp   + N;
  allpetsc  = appPerm  + N;
  petscPerm = allpetsc + N;
  ierr = MPI_Allgatherv((void*)myapp, napp, MPIU_INT, allapp,   lens, disp, MPIU_INT, comm);CHKERRQ(ierr);
  ierr = MPI_Allgatherv((void*)petsc, napp, MPIU_INT, allpetsc, lens, disp, MPIU_INT, comm);CHKERRQ(ierr);
  ierr = PetscFree(lens);CHKERRQ(ierr);

  /* generate a list of application and PETSc node numbers */
  ierr = PetscMalloc(N*4 * sizeof(PetscInt), &aomap->app);CHKERRQ(ierr);
  ierr = PetscLogObjectMemory(ao, 4*N * sizeof(PetscInt));CHKERRQ(ierr);
  aomap->appPerm   = aomap->app     + N;
  aomap->petsc     = aomap->appPerm + N;
  aomap->petscPerm = aomap->petsc   + N;
  for(i = 0; i < N; i++) {
    appPerm[i]   = i;
    petscPerm[i] = i;
  }
  ierr = PetscSortIntWithPermutation(N, allpetsc, petscPerm);CHKERRQ(ierr);
  ierr = PetscSortIntWithPermutation(N, allapp,   appPerm);CHKERRQ(ierr);
  /* Form sorted arrays of indices */
  for(i = 0; i < N; i++) {
    aomap->app[i]   = allapp[appPerm[i]];
    aomap->petsc[i] = allpetsc[petscPerm[i]];
  }
  /* Invert petscPerm[] into aomap->petscPerm[] */
  for(i = 0; i < N; i++) {
    aomap->petscPerm[petscPerm[i]] = i;
  }
  /* Form map between aomap->app[] and aomap->petsc[] */
  for(i = 0; i < N; i++) {
    aomap->appPerm[i] = aomap->petscPerm[appPerm[i]];
  }
  /* Invert appPerm[] into allapp[] */
  for(i = 0; i < N; i++) {
    allapp[appPerm[i]] = i;
  }
  /* Form map between aomap->petsc[] and aomap->app[] */
  for(i = 0; i < N; i++) {
    aomap->petscPerm[i] = allapp[petscPerm[i]];
  }
#ifdef PETSC_USE_DEBUG
  /* Check that the permutations are complementary */
  for(i = 0; i < N; i++) {
    if (i != aomap->appPerm[aomap->petscPerm[i]])
      SETERRQ(PETSC_ERR_PLIB, "Invalid ordering");
  }
#endif
  /* Cleanup */
  if (!mypetsc) {
    ierr = PetscFree(petsc);CHKERRQ(ierr);
  }
  ierr = PetscFree(allapp);CHKERRQ(ierr);

  ierr = PetscOptionsHasName(PETSC_NULL, "-ao_view", &opt);CHKERRQ(ierr);
  if (opt) {
    ierr = AOView(ao, PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);
  }

  *aoout = ao;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "AOCreateMappingIS"
/*@C
  AOCreateMappingIS - Creates a basic application ordering using two index sets.

  Input Parameters:
+ comm    - MPI communicator that is to share AO
. isapp   - index set that defines an ordering
- ispetsc - index set that defines another ordering

  Output Parameter:
. aoout   - the new application ordering

  Options Database Key:
$ -ao_view : call AOView() at the conclusion of AOCreateMappingIS()

  Level: beginner

.keywords: AO, create
.seealso: AOCreateBasic(), AOCreateMapping(), AODestroy()
@*/
PetscErrorCode PETSCDM_DLLEXPORT AOCreateMappingIS(IS isapp, IS ispetsc, AO *aoout)
{
  MPI_Comm       comm;
  PetscInt       *mypetsc, *myapp;
  PetscInt       napp, npetsc;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject) isapp, &comm);CHKERRQ(ierr);
  ierr = ISGetSize(isapp, &napp);CHKERRQ(ierr);
  if (ispetsc) {
    ierr = ISGetSize(ispetsc, &npetsc);CHKERRQ(ierr);
    if (napp != npetsc) SETERRQ(PETSC_ERR_ARG_SIZ, "Local IS lengths must match");
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
