
/*
    The most basic AO application ordering routines. These store the
  entire orderings on each processor.
*/

#include <../src/vec/is/ao/aoimpl.h> /*I  "petscao.h"   I*/

typedef struct {
  PetscInt *app;   /* app[i] is the partner for the ith PETSc slot */
  PetscInt *petsc; /* petsc[j] is the partner for the jth app slot */
} AO_Basic;

/*
       All processors have the same data so processor 1 prints it
*/
PetscErrorCode AOView_Basic(AO ao, PetscViewer viewer)
{
  PetscMPIInt rank;
  PetscInt    i;
  AO_Basic   *aobasic = (AO_Basic *)ao->data;
  PetscBool   iascii;

  PetscFunctionBegin;
  PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)ao), &rank));
  if (rank == 0) {
    PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &iascii));
    if (iascii) {
      PetscCall(PetscViewerASCIIPrintf(viewer, "Number of elements in ordering %" PetscInt_FMT "\n", ao->N));
      PetscCall(PetscViewerASCIIPrintf(viewer, "PETSc->App  App->PETSc\n"));
      for (i = 0; i < ao->N; i++) PetscCall(PetscViewerASCIIPrintf(viewer, "%3" PetscInt_FMT "  %3" PetscInt_FMT "    %3" PetscInt_FMT "  %3" PetscInt_FMT "\n", i, aobasic->app[i], i, aobasic->petsc[i]));
    }
  }
  PetscCall(PetscViewerFlush(viewer));
  PetscFunctionReturn(0);
}

PetscErrorCode AODestroy_Basic(AO ao)
{
  AO_Basic *aobasic = (AO_Basic *)ao->data;

  PetscFunctionBegin;
  PetscCall(PetscFree2(aobasic->app, aobasic->petsc));
  PetscCall(PetscFree(aobasic));
  PetscFunctionReturn(0);
}

PetscErrorCode AOBasicGetIndices_Private(AO ao, PetscInt **app, PetscInt **petsc)
{
  AO_Basic *basic = (AO_Basic *)ao->data;

  PetscFunctionBegin;
  if (app) *app = basic->app;
  if (petsc) *petsc = basic->petsc;
  PetscFunctionReturn(0);
}

PetscErrorCode AOPetscToApplication_Basic(AO ao, PetscInt n, PetscInt *ia)
{
  PetscInt  i, N = ao->N;
  AO_Basic *aobasic = (AO_Basic *)ao->data;

  PetscFunctionBegin;
  for (i = 0; i < n; i++) {
    if (ia[i] >= 0 && ia[i] < N) {
      ia[i] = aobasic->app[ia[i]];
    } else {
      ia[i] = -1;
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode AOApplicationToPetsc_Basic(AO ao, PetscInt n, PetscInt *ia)
{
  PetscInt  i, N = ao->N;
  AO_Basic *aobasic = (AO_Basic *)ao->data;

  PetscFunctionBegin;
  for (i = 0; i < n; i++) {
    if (ia[i] >= 0 && ia[i] < N) {
      ia[i] = aobasic->petsc[ia[i]];
    } else {
      ia[i] = -1;
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode AOPetscToApplicationPermuteInt_Basic(AO ao, PetscInt block, PetscInt *array)
{
  AO_Basic *aobasic = (AO_Basic *)ao->data;
  PetscInt *temp;
  PetscInt  i, j;

  PetscFunctionBegin;
  PetscCall(PetscMalloc1(ao->N * block, &temp));
  for (i = 0; i < ao->N; i++) {
    for (j = 0; j < block; j++) temp[i * block + j] = array[aobasic->petsc[i] * block + j];
  }
  PetscCall(PetscArraycpy(array, temp, ao->N * block));
  PetscCall(PetscFree(temp));
  PetscFunctionReturn(0);
}

PetscErrorCode AOApplicationToPetscPermuteInt_Basic(AO ao, PetscInt block, PetscInt *array)
{
  AO_Basic *aobasic = (AO_Basic *)ao->data;
  PetscInt *temp;
  PetscInt  i, j;

  PetscFunctionBegin;
  PetscCall(PetscMalloc1(ao->N * block, &temp));
  for (i = 0; i < ao->N; i++) {
    for (j = 0; j < block; j++) temp[i * block + j] = array[aobasic->app[i] * block + j];
  }
  PetscCall(PetscArraycpy(array, temp, ao->N * block));
  PetscCall(PetscFree(temp));
  PetscFunctionReturn(0);
}

PetscErrorCode AOPetscToApplicationPermuteReal_Basic(AO ao, PetscInt block, PetscReal *array)
{
  AO_Basic  *aobasic = (AO_Basic *)ao->data;
  PetscReal *temp;
  PetscInt   i, j;

  PetscFunctionBegin;
  PetscCall(PetscMalloc1(ao->N * block, &temp));
  for (i = 0; i < ao->N; i++) {
    for (j = 0; j < block; j++) temp[i * block + j] = array[aobasic->petsc[i] * block + j];
  }
  PetscCall(PetscArraycpy(array, temp, ao->N * block));
  PetscCall(PetscFree(temp));
  PetscFunctionReturn(0);
}

PetscErrorCode AOApplicationToPetscPermuteReal_Basic(AO ao, PetscInt block, PetscReal *array)
{
  AO_Basic  *aobasic = (AO_Basic *)ao->data;
  PetscReal *temp;
  PetscInt   i, j;

  PetscFunctionBegin;
  PetscCall(PetscMalloc1(ao->N * block, &temp));
  for (i = 0; i < ao->N; i++) {
    for (j = 0; j < block; j++) temp[i * block + j] = array[aobasic->app[i] * block + j];
  }
  PetscCall(PetscArraycpy(array, temp, ao->N * block));
  PetscCall(PetscFree(temp));
  PetscFunctionReturn(0);
}

static struct _AOOps AOOps_Basic = {
  PetscDesignatedInitializer(view, AOView_Basic),
  PetscDesignatedInitializer(destroy, AODestroy_Basic),
  PetscDesignatedInitializer(petsctoapplication, AOPetscToApplication_Basic),
  PetscDesignatedInitializer(applicationtopetsc, AOApplicationToPetsc_Basic),
  PetscDesignatedInitializer(petsctoapplicationpermuteint, AOPetscToApplicationPermuteInt_Basic),
  PetscDesignatedInitializer(applicationtopetscpermuteint, AOApplicationToPetscPermuteInt_Basic),
  PetscDesignatedInitializer(petsctoapplicationpermutereal, AOPetscToApplicationPermuteReal_Basic),
  PetscDesignatedInitializer(applicationtopetscpermutereal, AOApplicationToPetscPermuteReal_Basic),
};

PETSC_EXTERN PetscErrorCode AOCreate_Basic(AO ao)
{
  AO_Basic       *aobasic;
  PetscMPIInt     size, rank, count, *lens, *disp;
  PetscInt        napp, *allpetsc, *allapp, ip, ia, N, i, *petsc = NULL, start;
  IS              isapp = ao->isapp, ispetsc = ao->ispetsc;
  MPI_Comm        comm;
  const PetscInt *myapp, *mypetsc = NULL;

  PetscFunctionBegin;
  /* create special struct aobasic */
  PetscCall(PetscNew(&aobasic));
  ao->data = (void *)aobasic;
  PetscCall(PetscMemcpy(ao->ops, &AOOps_Basic, sizeof(struct _AOOps)));
  PetscCall(PetscObjectChangeTypeName((PetscObject)ao, AOBASIC));

  PetscCall(ISGetLocalSize(isapp, &napp));
  PetscCall(ISGetIndices(isapp, &myapp));

  PetscCall(PetscMPIIntCast(napp, &count));

  /* transmit all lengths to all processors */
  PetscCall(PetscObjectGetComm((PetscObject)isapp, &comm));
  PetscCallMPI(MPI_Comm_size(comm, &size));
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  PetscCall(PetscMalloc2(size, &lens, size, &disp));
  PetscCallMPI(MPI_Allgather(&count, 1, MPI_INT, lens, 1, MPI_INT, comm));
  N = 0;
  for (i = 0; i < size; i++) {
    PetscCall(PetscMPIIntCast(N, disp + i)); /* = sum(lens[j]), j< i */
    N += lens[i];
  }
  ao->N = N;
  ao->n = N;

  /* If mypetsc is 0 then use "natural" numbering */
  if (napp) {
    if (!ispetsc) {
      start = disp[rank];
      PetscCall(PetscMalloc1(napp + 1, &petsc));
      for (i = 0; i < napp; i++) petsc[i] = start + i;
    } else {
      PetscCall(ISGetIndices(ispetsc, &mypetsc));
      petsc = (PetscInt *)mypetsc;
    }
  }

  /* get all indices on all processors */
  PetscCall(PetscMalloc2(N, &allpetsc, N, &allapp));
  PetscCallMPI(MPI_Allgatherv(petsc, count, MPIU_INT, allpetsc, lens, disp, MPIU_INT, comm));
  PetscCallMPI(MPI_Allgatherv((void *)myapp, count, MPIU_INT, allapp, lens, disp, MPIU_INT, comm));
  PetscCall(PetscFree2(lens, disp));

  if (PetscDefined(USE_DEBUG)) {
    PetscInt *sorted;
    PetscCall(PetscMalloc1(N, &sorted));

    PetscCall(PetscArraycpy(sorted, allpetsc, N));
    PetscCall(PetscSortInt(N, sorted));
    for (i = 0; i < N; i++) PetscCheck(sorted[i] == i, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "PETSc ordering requires a permutation of numbers 0 to N-1\n it is missing %" PetscInt_FMT " has %" PetscInt_FMT, i, sorted[i]);

    PetscCall(PetscArraycpy(sorted, allapp, N));
    PetscCall(PetscSortInt(N, sorted));
    for (i = 0; i < N; i++) PetscCheck(sorted[i] == i, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Application ordering requires a permutation of numbers 0 to N-1\n it is missing %" PetscInt_FMT " has %" PetscInt_FMT, i, sorted[i]);

    PetscCall(PetscFree(sorted));
  }

  /* generate a list of application and PETSc node numbers */
  PetscCall(PetscCalloc2(N, &aobasic->app, N, &aobasic->petsc));
  for (i = 0; i < N; i++) {
    ip = allpetsc[i];
    ia = allapp[i];
    /* check there are no duplicates */
    PetscCheck(!aobasic->app[ip], PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Duplicate in PETSc ordering at position %" PetscInt_FMT ". Already mapped to %" PetscInt_FMT ", not %" PetscInt_FMT ".", i, aobasic->app[ip] - 1, ia);
    aobasic->app[ip] = ia + 1;
    PetscCheck(!aobasic->petsc[ia], PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Duplicate in Application ordering at position %" PetscInt_FMT ". Already mapped to %" PetscInt_FMT ", not %" PetscInt_FMT ".", i, aobasic->petsc[ia] - 1, ip);
    aobasic->petsc[ia] = ip + 1;
  }
  if (napp && !mypetsc) PetscCall(PetscFree(petsc));
  PetscCall(PetscFree2(allpetsc, allapp));
  /* shift indices down by one */
  for (i = 0; i < N; i++) {
    aobasic->app[i]--;
    aobasic->petsc[i]--;
  }

  PetscCall(ISRestoreIndices(isapp, &myapp));
  if (napp) {
    if (ispetsc) {
      PetscCall(ISRestoreIndices(ispetsc, &mypetsc));
    } else {
      PetscCall(PetscFree(petsc));
    }
  }
  PetscFunctionReturn(0);
}

/*@C
   AOCreateBasic - Creates a basic application ordering using two integer arrays.

   Collective

   Input Parameters:
+  comm - MPI communicator that is to share `AO`
.  napp - size of integer arrays
.  myapp - integer array that defines an ordering
-  mypetsc - integer array that defines another ordering (may be NULL to
             indicate the natural ordering, that is 0,1,2,3,...)

   Output Parameter:
.  aoout - the new application ordering

   Level: beginner

   Note:
   The arrays myapp and mypetsc must contain the all the integers 0 to napp-1 with no duplicates; that is there cannot be any "holes"
   in the indices. Use `AOCreateMapping()` or `AOCreateMappingIS()` if you wish to have "holes" in the indices.

.seealso: [](sec_ao), [](sec_scatter), `AO`, `AOCreateBasicIS()`, `AODestroy()`, `AOPetscToApplication()`, `AOApplicationToPetsc()`
@*/
PetscErrorCode AOCreateBasic(MPI_Comm comm, PetscInt napp, const PetscInt myapp[], const PetscInt mypetsc[], AO *aoout)
{
  IS              isapp, ispetsc;
  const PetscInt *app = myapp, *petsc = mypetsc;

  PetscFunctionBegin;
  PetscCall(ISCreateGeneral(comm, napp, app, PETSC_USE_POINTER, &isapp));
  if (mypetsc) {
    PetscCall(ISCreateGeneral(comm, napp, petsc, PETSC_USE_POINTER, &ispetsc));
  } else {
    ispetsc = NULL;
  }
  PetscCall(AOCreateBasicIS(isapp, ispetsc, aoout));
  PetscCall(ISDestroy(&isapp));
  if (mypetsc) PetscCall(ISDestroy(&ispetsc));
  PetscFunctionReturn(0);
}

/*@C
   AOCreateBasicIS - Creates a basic application ordering using two `IS` index sets.

   Collective on isapp

   Input Parameters:
+  isapp - index set that defines an ordering
-  ispetsc - index set that defines another ordering (may be NULL to use the
             natural ordering)

   Output Parameter:
.  aoout - the new application ordering

   Level: beginner

    Note:
    The index sets isapp and ispetsc must contain the all the integers 0 to napp-1 (where napp is the length of the index sets) with no duplicates;
    that is there cannot be any "holes"

.seealso: [](sec_ao), [](sec_scatter), `IS`, `AO`, `AOCreateBasic()`, `AODestroy()`
@*/
PetscErrorCode AOCreateBasicIS(IS isapp, IS ispetsc, AO *aoout)
{
  MPI_Comm comm;
  AO       ao;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)isapp, &comm));
  PetscCall(AOCreate(comm, &ao));
  PetscCall(AOSetIS(ao, isapp, ispetsc));
  PetscCall(AOSetType(ao, AOBASIC));
  PetscCall(AOViewFromOptions(ao, NULL, "-ao_view"));
  *aoout = ao;
  PetscFunctionReturn(0);
}
