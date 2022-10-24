
/*
    The memory scalable AO application ordering routines. These store the
  orderings on each processor for that processor's range of values
*/

#include <../src/vec/is/ao/aoimpl.h> /*I  "petscao.h"   I*/

typedef struct {
  PetscInt   *app_loc;   /* app_loc[i] is the partner for the ith local PETSc slot */
  PetscInt   *petsc_loc; /* petsc_loc[j] is the partner for the jth local app slot */
  PetscLayout map;       /* determines the local sizes of ao */
} AO_MemoryScalable;

/*
       All processors ship the data to process 0 to be printed; note that this is not scalable because
       process 0 allocates space for all the orderings entry across all the processes
*/
PetscErrorCode AOView_MemoryScalable(AO ao, PetscViewer viewer)
{
  PetscMPIInt        rank, size;
  AO_MemoryScalable *aomems = (AO_MemoryScalable *)ao->data;
  PetscBool          iascii;
  PetscMPIInt        tag_app, tag_petsc;
  PetscLayout        map = aomems->map;
  PetscInt          *app, *app_loc, *petsc, *petsc_loc, len, i, j;
  MPI_Status         status;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &iascii));
  PetscCheck(iascii, PetscObjectComm((PetscObject)viewer), PETSC_ERR_SUP, "Viewer type %s not supported for AO MemoryScalable", ((PetscObject)viewer)->type_name);

  PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)ao), &rank));
  PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)ao), &size));

  PetscCall(PetscObjectGetNewTag((PetscObject)ao, &tag_app));
  PetscCall(PetscObjectGetNewTag((PetscObject)ao, &tag_petsc));

  if (rank == 0) {
    PetscCall(PetscViewerASCIIPrintf(viewer, "Number of elements in ordering %" PetscInt_FMT "\n", ao->N));
    PetscCall(PetscViewerASCIIPrintf(viewer, "PETSc->App  App->PETSc\n"));

    PetscCall(PetscMalloc2(map->N, &app, map->N, &petsc));
    len = map->n;
    /* print local AO */
    PetscCall(PetscViewerASCIIPrintf(viewer, "Process [%d]\n", rank));
    for (i = 0; i < len; i++) PetscCall(PetscViewerASCIIPrintf(viewer, "%3" PetscInt_FMT "  %3" PetscInt_FMT "    %3" PetscInt_FMT "  %3" PetscInt_FMT "\n", i, aomems->app_loc[i], i, aomems->petsc_loc[i]));

    /* recv and print off-processor's AO */
    for (i = 1; i < size; i++) {
      len       = map->range[i + 1] - map->range[i];
      app_loc   = app + map->range[i];
      petsc_loc = petsc + map->range[i];
      PetscCallMPI(MPI_Recv(app_loc, (PetscMPIInt)len, MPIU_INT, i, tag_app, PetscObjectComm((PetscObject)ao), &status));
      PetscCallMPI(MPI_Recv(petsc_loc, (PetscMPIInt)len, MPIU_INT, i, tag_petsc, PetscObjectComm((PetscObject)ao), &status));
      PetscCall(PetscViewerASCIIPrintf(viewer, "Process [%" PetscInt_FMT "]\n", i));
      for (j = 0; j < len; j++) PetscCall(PetscViewerASCIIPrintf(viewer, "%3" PetscInt_FMT "  %3" PetscInt_FMT "    %3" PetscInt_FMT "  %3" PetscInt_FMT "\n", map->range[i] + j, app_loc[j], map->range[i] + j, petsc_loc[j]));
    }
    PetscCall(PetscFree2(app, petsc));

  } else {
    /* send values */
    PetscCallMPI(MPI_Send((void *)aomems->app_loc, map->n, MPIU_INT, 0, tag_app, PetscObjectComm((PetscObject)ao)));
    PetscCallMPI(MPI_Send((void *)aomems->petsc_loc, map->n, MPIU_INT, 0, tag_petsc, PetscObjectComm((PetscObject)ao)));
  }
  PetscCall(PetscViewerFlush(viewer));
  PetscFunctionReturn(0);
}

PetscErrorCode AODestroy_MemoryScalable(AO ao)
{
  AO_MemoryScalable *aomems = (AO_MemoryScalable *)ao->data;

  PetscFunctionBegin;
  PetscCall(PetscFree2(aomems->app_loc, aomems->petsc_loc));
  PetscCall(PetscLayoutDestroy(&aomems->map));
  PetscCall(PetscFree(aomems));
  PetscFunctionReturn(0);
}

/*
   Input Parameters:
+   ao - the application ordering context
.   n  - the number of integers in ia[]
.   ia - the integers; these are replaced with their mapped value
-   maploc - app_loc or petsc_loc in struct "AO_MemoryScalable"

   Output Parameter:
.   ia - the mapped interges
 */
PetscErrorCode AOMap_MemoryScalable_private(AO ao, PetscInt n, PetscInt *ia, const PetscInt *maploc)
{
  AO_MemoryScalable *aomems = (AO_MemoryScalable *)ao->data;
  MPI_Comm           comm;
  PetscMPIInt        rank, size, tag1, tag2;
  PetscInt          *owner, *start, *sizes, nsends, nreceives;
  PetscInt           nmax, count, *sindices, *rindices, i, j, idx, lastidx, *sindices2, *rindices2;
  const PetscInt    *owners = aomems->map->range;
  MPI_Request       *send_waits, *recv_waits, *send_waits2, *recv_waits2;
  MPI_Status         recv_status;
  PetscMPIInt        nindices, source, widx;
  PetscInt          *rbuf, *sbuf;
  MPI_Status        *send_status, *send_status2;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)ao, &comm));
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  PetscCallMPI(MPI_Comm_size(comm, &size));

  /*  first count number of contributors to each processor */
  PetscCall(PetscMalloc1(size, &start));
  PetscCall(PetscCalloc2(2 * size, &sizes, n, &owner));

  j       = 0;
  lastidx = -1;
  for (i = 0; i < n; i++) {
    if (ia[i] < 0) owner[i] = -1;      /* mark negative entries (which are not to be mapped) with a special negative value */
    if (ia[i] >= ao->N) owner[i] = -2; /* mark out of range entries with special negative value */
    else {
      /* if indices are NOT locally sorted, need to start search at the beginning */
      if (lastidx > (idx = ia[i])) j = 0;
      lastidx = idx;
      for (; j < size; j++) {
        if (idx >= owners[j] && idx < owners[j + 1]) {
          sizes[2 * j]++;       /* num of indices to be sent */
          sizes[2 * j + 1] = 1; /* send to proc[j] */
          owner[i]         = j;
          break;
        }
      }
    }
  }
  sizes[2 * rank] = sizes[2 * rank + 1] = 0; /* do not receive from self! */
  nsends                                = 0;
  for (i = 0; i < size; i++) nsends += sizes[2 * i + 1];

  /* inform other processors of number of messages and max length*/
  PetscCall(PetscMaxSum(comm, sizes, &nmax, &nreceives));

  /* allocate arrays */
  PetscCall(PetscObjectGetNewTag((PetscObject)ao, &tag1));
  PetscCall(PetscObjectGetNewTag((PetscObject)ao, &tag2));

  PetscCall(PetscMalloc2(nreceives * nmax, &rindices, nreceives, &recv_waits));
  PetscCall(PetscMalloc2(nsends * nmax, &rindices2, nsends, &recv_waits2));

  PetscCall(PetscMalloc3(n, &sindices, nsends, &send_waits, nsends, &send_status));
  PetscCall(PetscMalloc3(n, &sindices2, nreceives, &send_waits2, nreceives, &send_status2));

  /* post 1st receives: receive others requests
     since we don't know how long each individual message is we
     allocate the largest needed buffer for each receive. Potentially
     this is a lot of wasted space.
  */
  for (i = 0, count = 0; i < nreceives; i++) PetscCallMPI(MPI_Irecv(rindices + nmax * i, nmax, MPIU_INT, MPI_ANY_SOURCE, tag1, comm, recv_waits + count++));

  /* do 1st sends:
      1) starts[i] gives the starting index in svalues for stuff going to
         the ith processor
  */
  start[0] = 0;
  for (i = 1; i < size; i++) start[i] = start[i - 1] + sizes[2 * i - 2];
  for (i = 0; i < n; i++) {
    j = owner[i];
    if (j == -1) continue; /* do not remap negative entries in ia[] */
    else if (j == -2) { /* out of range entries get mapped to -1 */ ia[i] = -1;
      continue;
    } else if (j != rank) {
      sindices[start[j]++] = ia[i];
    } else { /* compute my own map */
      ia[i] = maploc[ia[i] - owners[rank]];
    }
  }

  start[0] = 0;
  for (i = 1; i < size; i++) start[i] = start[i - 1] + sizes[2 * i - 2];
  for (i = 0, count = 0; i < size; i++) {
    if (sizes[2 * i + 1]) {
      /* send my request to others */
      PetscCallMPI(MPI_Isend(sindices + start[i], sizes[2 * i], MPIU_INT, i, tag1, comm, send_waits + count));
      /* post receive for the answer of my request */
      PetscCallMPI(MPI_Irecv(sindices2 + start[i], sizes[2 * i], MPIU_INT, i, tag2, comm, recv_waits2 + count));
      count++;
    }
  }
  PetscCheck(nsends == count, comm, PETSC_ERR_SUP, "nsends %" PetscInt_FMT " != count %" PetscInt_FMT, nsends, count);

  /* wait on 1st sends */
  if (nsends) PetscCallMPI(MPI_Waitall(nsends, send_waits, send_status));

  /* 1st recvs: other's requests */
  for (j = 0; j < nreceives; j++) {
    PetscCallMPI(MPI_Waitany(nreceives, recv_waits, &widx, &recv_status)); /* idx: index of handle for operation that completed */
    PetscCallMPI(MPI_Get_count(&recv_status, MPIU_INT, &nindices));
    rbuf   = rindices + nmax * widx; /* global index */
    source = recv_status.MPI_SOURCE;

    /* compute mapping */
    sbuf = rbuf;
    for (i = 0; i < nindices; i++) sbuf[i] = maploc[rbuf[i] - owners[rank]];

    /* send mapping back to the sender */
    PetscCallMPI(MPI_Isend(sbuf, nindices, MPIU_INT, source, tag2, comm, send_waits2 + widx));
  }

  /* wait on 2nd sends */
  if (nreceives) PetscCallMPI(MPI_Waitall(nreceives, send_waits2, send_status2));

  /* 2nd recvs: for the answer of my request */
  for (j = 0; j < nsends; j++) {
    PetscCallMPI(MPI_Waitany(nsends, recv_waits2, &widx, &recv_status));
    PetscCallMPI(MPI_Get_count(&recv_status, MPIU_INT, &nindices));
    source = recv_status.MPI_SOURCE;
    /* pack output ia[] */
    rbuf  = sindices2 + start[source];
    count = 0;
    for (i = 0; i < n; i++) {
      if (source == owner[i]) ia[i] = rbuf[count++];
    }
  }

  /* free arrays */
  PetscCall(PetscFree(start));
  PetscCall(PetscFree2(sizes, owner));
  PetscCall(PetscFree2(rindices, recv_waits));
  PetscCall(PetscFree2(rindices2, recv_waits2));
  PetscCall(PetscFree3(sindices, send_waits, send_status));
  PetscCall(PetscFree3(sindices2, send_waits2, send_status2));
  PetscFunctionReturn(0);
}

PetscErrorCode AOPetscToApplication_MemoryScalable(AO ao, PetscInt n, PetscInt *ia)
{
  AO_MemoryScalable *aomems  = (AO_MemoryScalable *)ao->data;
  PetscInt          *app_loc = aomems->app_loc;

  PetscFunctionBegin;
  PetscCall(AOMap_MemoryScalable_private(ao, n, ia, app_loc));
  PetscFunctionReturn(0);
}

PetscErrorCode AOApplicationToPetsc_MemoryScalable(AO ao, PetscInt n, PetscInt *ia)
{
  AO_MemoryScalable *aomems    = (AO_MemoryScalable *)ao->data;
  PetscInt          *petsc_loc = aomems->petsc_loc;

  PetscFunctionBegin;
  PetscCall(AOMap_MemoryScalable_private(ao, n, ia, petsc_loc));
  PetscFunctionReturn(0);
}

static struct _AOOps AOOps_MemoryScalable = {
  PetscDesignatedInitializer(view, AOView_MemoryScalable),
  PetscDesignatedInitializer(destroy, AODestroy_MemoryScalable),
  PetscDesignatedInitializer(petsctoapplication, AOPetscToApplication_MemoryScalable),
  PetscDesignatedInitializer(applicationtopetsc, AOApplicationToPetsc_MemoryScalable),
};

PetscErrorCode AOCreateMemoryScalable_private(MPI_Comm comm, PetscInt napp, const PetscInt from_array[], const PetscInt to_array[], AO ao, PetscInt *aomap_loc)
{
  AO_MemoryScalable *aomems  = (AO_MemoryScalable *)ao->data;
  PetscLayout        map     = aomems->map;
  PetscInt           n_local = map->n, i, j;
  PetscMPIInt        rank, size, tag;
  PetscInt          *owner, *start, *sizes, nsends, nreceives;
  PetscInt           nmax, count, *sindices, *rindices, idx, lastidx;
  PetscInt          *owners = aomems->map->range;
  MPI_Request       *send_waits, *recv_waits;
  MPI_Status         recv_status;
  PetscMPIInt        nindices, widx;
  PetscInt          *rbuf;
  PetscInt           n = napp, ip, ia;
  MPI_Status        *send_status;

  PetscFunctionBegin;
  PetscCall(PetscArrayzero(aomap_loc, n_local));

  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  PetscCallMPI(MPI_Comm_size(comm, &size));

  /*  first count number of contributors (of from_array[]) to each processor */
  PetscCall(PetscCalloc1(2 * size, &sizes));
  PetscCall(PetscMalloc1(n, &owner));

  j       = 0;
  lastidx = -1;
  for (i = 0; i < n; i++) {
    /* if indices are NOT locally sorted, need to start search at the beginning */
    if (lastidx > (idx = from_array[i])) j = 0;
    lastidx = idx;
    for (; j < size; j++) {
      if (idx >= owners[j] && idx < owners[j + 1]) {
        sizes[2 * j] += 2;    /* num of indices to be sent - in pairs (ip,ia) */
        sizes[2 * j + 1] = 1; /* send to proc[j] */
        owner[i]         = j;
        break;
      }
    }
  }
  sizes[2 * rank] = sizes[2 * rank + 1] = 0; /* do not receive from self! */
  nsends                                = 0;
  for (i = 0; i < size; i++) nsends += sizes[2 * i + 1];

  /* inform other processors of number of messages and max length*/
  PetscCall(PetscMaxSum(comm, sizes, &nmax, &nreceives));

  /* allocate arrays */
  PetscCall(PetscObjectGetNewTag((PetscObject)ao, &tag));
  PetscCall(PetscMalloc2(nreceives * nmax, &rindices, nreceives, &recv_waits));
  PetscCall(PetscMalloc3(2 * n, &sindices, nsends, &send_waits, nsends, &send_status));
  PetscCall(PetscMalloc1(size, &start));

  /* post receives: */
  for (i = 0; i < nreceives; i++) PetscCallMPI(MPI_Irecv(rindices + nmax * i, nmax, MPIU_INT, MPI_ANY_SOURCE, tag, comm, recv_waits + i));

  /* do sends:
      1) starts[i] gives the starting index in svalues for stuff going to
         the ith processor
  */
  start[0] = 0;
  for (i = 1; i < size; i++) start[i] = start[i - 1] + sizes[2 * i - 2];
  for (i = 0; i < n; i++) {
    j = owner[i];
    if (j != rank) {
      ip                   = from_array[i];
      ia                   = to_array[i];
      sindices[start[j]++] = ip;
      sindices[start[j]++] = ia;
    } else { /* compute my own map */
      ip            = from_array[i] - owners[rank];
      ia            = to_array[i];
      aomap_loc[ip] = ia;
    }
  }

  start[0] = 0;
  for (i = 1; i < size; i++) start[i] = start[i - 1] + sizes[2 * i - 2];
  for (i = 0, count = 0; i < size; i++) {
    if (sizes[2 * i + 1]) {
      PetscCallMPI(MPI_Isend(sindices + start[i], sizes[2 * i], MPIU_INT, i, tag, comm, send_waits + count));
      count++;
    }
  }
  PetscCheck(nsends == count, comm, PETSC_ERR_SUP, "nsends %" PetscInt_FMT " != count %" PetscInt_FMT, nsends, count);

  /* wait on sends */
  if (nsends) PetscCallMPI(MPI_Waitall(nsends, send_waits, send_status));

  /* recvs */
  count = 0;
  for (j = nreceives; j > 0; j--) {
    PetscCallMPI(MPI_Waitany(nreceives, recv_waits, &widx, &recv_status));
    PetscCallMPI(MPI_Get_count(&recv_status, MPIU_INT, &nindices));
    rbuf = rindices + nmax * widx; /* global index */

    /* compute local mapping */
    for (i = 0; i < nindices; i += 2) {       /* pack aomap_loc */
      ip            = rbuf[i] - owners[rank]; /* local index */
      ia            = rbuf[i + 1];
      aomap_loc[ip] = ia;
    }
    count++;
  }

  PetscCall(PetscFree(start));
  PetscCall(PetscFree3(sindices, send_waits, send_status));
  PetscCall(PetscFree2(rindices, recv_waits));
  PetscCall(PetscFree(owner));
  PetscCall(PetscFree(sizes));
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode AOCreate_MemoryScalable(AO ao)
{
  IS                 isapp = ao->isapp, ispetsc = ao->ispetsc;
  const PetscInt    *mypetsc, *myapp;
  PetscInt           napp, n_local, N, i, start, *petsc, *lens, *disp;
  MPI_Comm           comm;
  AO_MemoryScalable *aomems;
  PetscLayout        map;
  PetscMPIInt        size, rank;

  PetscFunctionBegin;
  PetscCheck(isapp, PetscObjectComm((PetscObject)ao), PETSC_ERR_ARG_WRONGSTATE, "AOSetIS() must be called before AOSetType()");
  /* create special struct aomems */
  PetscCall(PetscNew(&aomems));
  ao->data = (void *)aomems;
  PetscCall(PetscMemcpy(ao->ops, &AOOps_MemoryScalable, sizeof(struct _AOOps)));
  PetscCall(PetscObjectChangeTypeName((PetscObject)ao, AOMEMORYSCALABLE));

  /* transmit all local lengths of isapp to all processors */
  PetscCall(PetscObjectGetComm((PetscObject)isapp, &comm));
  PetscCallMPI(MPI_Comm_size(comm, &size));
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  PetscCall(PetscMalloc2(size, &lens, size, &disp));
  PetscCall(ISGetLocalSize(isapp, &napp));
  PetscCallMPI(MPI_Allgather(&napp, 1, MPIU_INT, lens, 1, MPIU_INT, comm));

  N = 0;
  for (i = 0; i < size; i++) {
    disp[i] = N;
    N += lens[i];
  }

  /* If ispetsc is 0 then use "natural" numbering */
  if (napp) {
    if (!ispetsc) {
      start = disp[rank];
      PetscCall(PetscMalloc1(napp + 1, &petsc));
      for (i = 0; i < napp; i++) petsc[i] = start + i;
    } else {
      PetscCall(ISGetIndices(ispetsc, &mypetsc));
      petsc = (PetscInt *)mypetsc;
    }
  } else {
    petsc = NULL;
  }

  /* create a map with global size N - used to determine the local sizes of ao - shall we use local napp instead of N? */
  PetscCall(PetscLayoutCreate(comm, &map));
  map->bs = 1;
  map->N  = N;
  PetscCall(PetscLayoutSetUp(map));

  ao->N       = N;
  ao->n       = map->n;
  aomems->map = map;

  /* create distributed indices app_loc: petsc->app and petsc_loc: app->petsc */
  n_local = map->n;
  PetscCall(PetscCalloc2(n_local, &aomems->app_loc, n_local, &aomems->petsc_loc));
  PetscCall(ISGetIndices(isapp, &myapp));

  PetscCall(AOCreateMemoryScalable_private(comm, napp, petsc, myapp, ao, aomems->app_loc));
  PetscCall(AOCreateMemoryScalable_private(comm, napp, myapp, petsc, ao, aomems->petsc_loc));

  PetscCall(ISRestoreIndices(isapp, &myapp));
  if (napp) {
    if (ispetsc) {
      PetscCall(ISRestoreIndices(ispetsc, &mypetsc));
    } else {
      PetscCall(PetscFree(petsc));
    }
  }
  PetscCall(PetscFree2(lens, disp));
  PetscFunctionReturn(0);
}

/*@C
   AOCreateMemoryScalable - Creates a memory scalable application ordering using two integer arrays.

   Collective

   Input Parameters:
+  comm - MPI communicator that is to share the `AO`
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
    Comparing with `AOCreateBasic()`, this routine trades memory with message communication.

.seealso: [](sec_ao), [](sec_scatter), `AO`, `AOCreateMemoryScalableIS()`, `AODestroy()`, `AOPetscToApplication()`, `AOApplicationToPetsc()`
@*/
PetscErrorCode AOCreateMemoryScalable(MPI_Comm comm, PetscInt napp, const PetscInt myapp[], const PetscInt mypetsc[], AO *aoout)
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
  PetscCall(AOCreateMemoryScalableIS(isapp, ispetsc, aoout));
  PetscCall(ISDestroy(&isapp));
  if (mypetsc) PetscCall(ISDestroy(&ispetsc));
  PetscFunctionReturn(0);
}

/*@C
   AOCreateMemoryScalableIS - Creates a memory scalable application ordering using two index sets.

   Collective on isapp

   Input Parameters:
+  isapp - index set that defines an ordering
-  ispetsc - index set that defines another ordering (may be NULL to use the
             natural ordering)

   Output Parameter:
.  aoout - the new application ordering

   Level: beginner

    Notes:
    The index sets isapp and ispetsc must contain the all the integers 0 to napp-1 (where napp is the length of the index sets) with no duplicates;
    that is there cannot be any "holes".

    Comparing with `AOCreateBasicIS()`, this routine trades memory with message communication.

.seealso: [](sec_ao), [](sec_scatter), `AO`, `AOCreateBasicIS()`, `AOCreateMemoryScalable()`, `AODestroy()`
@*/
PetscErrorCode AOCreateMemoryScalableIS(IS isapp, IS ispetsc, AO *aoout)
{
  MPI_Comm comm;
  AO       ao;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)isapp, &comm));
  PetscCall(AOCreate(comm, &ao));
  PetscCall(AOSetIS(ao, isapp, ispetsc));
  PetscCall(AOSetType(ao, AOMEMORYSCALABLE));
  PetscCall(AOViewFromOptions(ao, NULL, "-ao_view"));
  *aoout = ao;
  PetscFunctionReturn(0);
}
