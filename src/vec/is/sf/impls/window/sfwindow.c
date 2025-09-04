#include <petsc/private/sfimpl.h> /*I "petscsf.h" I*/

typedef struct _n_PetscSFDataLink *PetscSFDataLink;
typedef struct _n_PetscSFWinLink  *PetscSFWinLink;

typedef struct {
  PetscSFWindowSyncType   sync;   /* FENCE, LOCK, or ACTIVE synchronization */
  PetscSFDataLink         link;   /* List of MPI data types, lazily constructed for each data type */
  PetscSFWinLink          wins;   /* List of active windows */
  PetscSFWindowFlavorType flavor; /* Current PETSCSF_WINDOW_FLAVOR_ */
  PetscSF                 dynsf;
  MPI_Info                info;
  MPI_Comm                window_comm;
  PetscBool               is_empty;
  PetscMPIInt            *wcommranks;
} PetscSF_Window;

struct _n_PetscSFDataLink {
  MPI_Datatype    unit;
  MPI_Datatype   *mine;
  MPI_Datatype   *remote;
  PetscSFDataLink next;
};

struct _n_PetscSFWinLink {
  PetscBool               inuse;
  MPI_Aint                bytes;
  void                   *addr;
  void                   *rootdata;
  void                   *leafdata;
  MPI_Win                 win;
  MPI_Request            *reqs;
  PetscSFWindowFlavorType flavor;
  MPI_Aint               *dyn_target_addr;
  PetscBool               epoch;
  PetscBool               persistent;
  PetscSFWinLink          next;
};

const char *const PetscSFWindowSyncTypes[]   = {"FENCE", "LOCK", "ACTIVE", "PetscSFWindowSyncType", "PETSCSF_WINDOW_SYNC_", NULL};
const char *const PetscSFWindowFlavorTypes[] = {"CREATE", "DYNAMIC", "ALLOCATE", "SHARED", "PetscSFWindowFlavorType", "PETSCSF_WINDOW_FLAVOR_", NULL};

/* Built-in MPI_Ops act elementwise inside MPI_Accumulate, but cannot be used with composite types inside collectives (MPI_Allreduce) */
static PetscErrorCode PetscSFWindowOpTranslate(MPI_Op *op)
{
  PetscFunctionBegin;
  if (*op == MPIU_SUM) *op = MPI_SUM;
  else if (*op == MPIU_MAX) *op = MPI_MAX;
  else if (*op == MPIU_MIN) *op = MPI_MIN;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
   PetscSFWindowGetDataTypes - gets composite local and remote data types for each rank

   Not Collective

   Input Parameters:
+  sf - star forest of type `PETSCSFWINDOW`
-  unit - data type for each node

   Output Parameters:
+  localtypes - types describing part of local leaf buffer referencing each remote rank
-  remotetypes - types describing part of remote root buffer referenced for each remote rank

   Level: developer

.seealso: `PetscSF`, `PETSCSFWINDOW`, `PetscSFSetGraph()`, `PetscSFView()`
@*/
static PetscErrorCode PetscSFWindowGetDataTypes(PetscSF sf, MPI_Datatype unit, const MPI_Datatype **localtypes, const MPI_Datatype **remotetypes)
{
  PetscSF_Window *w = (PetscSF_Window *)sf->data;
  PetscSFDataLink link;
  PetscMPIInt     nranks;
  const PetscInt *roffset;

  PetscFunctionBegin;
  /* Look for types in cache */
  for (link = w->link; link; link = link->next) {
    PetscBool match;

    PetscCall(MPIPetsc_Type_compare(unit, link->unit, &match));
    if (match) {
      *localtypes  = link->mine;
      *remotetypes = link->remote;
      PetscFunctionReturn(PETSC_SUCCESS);
    }
  }

  /* Create new composite types for each send rank */
  PetscCall(PetscSFGetRootRanks(sf, &nranks, NULL, &roffset, NULL, NULL));
  PetscCall(PetscNew(&link));
  PetscCallMPI(MPI_Type_dup(unit, &link->unit));
  PetscCall(PetscMalloc2(nranks, &link->mine, nranks, &link->remote));
  for (PetscMPIInt i = 0; i < nranks; i++) {
    PetscMPIInt  rcount;
    PetscMPIInt *rmine, *rremote;

    PetscCall(PetscMPIIntCast(roffset[i + 1] - roffset[i], &rcount));
#if !defined(PETSC_USE_64BIT_INDICES)
    rmine   = sf->rmine + sf->roffset[i];
    rremote = sf->rremote + sf->roffset[i];
#else
    PetscCall(PetscMalloc2(rcount, &rmine, rcount, &rremote));
    for (PetscInt j = 0; j < rcount; j++) {
      PetscCall(PetscMPIIntCast(sf->rmine[sf->roffset[i] + j], &rmine[j]));
      PetscCall(PetscMPIIntCast(sf->rremote[sf->roffset[i] + j], &rremote[j]));
    }
#endif

    PetscCallMPI(MPI_Type_create_indexed_block(rcount, 1, rmine, link->unit, &link->mine[i]));
    PetscCallMPI(MPI_Type_create_indexed_block(rcount, 1, rremote, link->unit, &link->remote[i]));
#if defined(PETSC_USE_64BIT_INDICES)
    PetscCall(PetscFree2(rmine, rremote));
#endif
    PetscCallMPI(MPI_Type_commit(&link->mine[i]));
    PetscCallMPI(MPI_Type_commit(&link->remote[i]));
  }
  link->next = w->link;
  w->link    = link;

  *localtypes  = link->mine;
  *remotetypes = link->remote;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscSFWindowSetFlavorType - Set flavor type for `MPI_Win` creation

  Logically Collective

  Input Parameters:
+ sf     - star forest for communication of type `PETSCSFWINDOW`
- flavor - flavor type

  Options Database Key:
. -sf_window_flavor <flavor> - sets the flavor type CREATE, DYNAMIC, ALLOCATE or SHARED (see `PetscSFWindowFlavorType`)

  Level: advanced

  Notes:
  Windows reuse follows these rules\:
.vb
     PETSCSF_WINDOW_FLAVOR_CREATE: creates a new window every time, uses MPI_Win_create

     PETSCSF_WINDOW_FLAVOR_DYNAMIC: uses MPI_Win_create_dynamic/MPI_Win_attach and tries to reuse windows by comparing the root array. Intended to be used on repeated applications of the same SF, e.g.
       PetscSFRegisterPersistent(sf,rootdata1,leafdata);
       for i=1 to K
         PetscSFOperationBegin(sf,rootdata1,leafdata);
         PetscSFOperationEnd(sf,rootdata1,leafdata);
         ...
         PetscSFOperationBegin(sf,rootdata1,leafdata);
         PetscSFOperationEnd(sf,rootdata1,leafdata);
       endfor
       PetscSFDeregisterPersistent(sf,rootdata1,leafdata);

       The following pattern will instead raise an error
         PetscSFOperationBegin(sf,rootdata1,leafdata);
         PetscSFOperationEnd(sf,rootdata1,leafdata);
         PetscSFOperationBegin(sf,rank ? rootdata1 : rootdata2,leafdata);
         PetscSFOperationEnd(sf,rank ? rootdata1 : rootdata2,leafdata);

     PETSCSF_WINDOW_FLAVOR_ALLOCATE: uses MPI_Win_allocate, reuses any pre-existing window which fits the data and it is not in use

     PETSCSF_WINDOW_FLAVOR_SHARED: uses MPI_Win_allocate_shared, reusage policy as for PETSCSF_WINDOW_FLAVOR_ALLOCATE
.ve

.seealso: `PetscSF`, `PETSCSFWINDOW`, `PetscSFSetFromOptions()`, `PetscSFWindowGetFlavorType()`
@*/
PetscErrorCode PetscSFWindowSetFlavorType(PetscSF sf, PetscSFWindowFlavorType flavor)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(sf, PETSCSF_CLASSID, 1);
  PetscValidLogicalCollectiveEnum(sf, flavor, 2);
  PetscTryMethod(sf, "PetscSFWindowSetFlavorType_C", (PetscSF, PetscSFWindowFlavorType), (sf, flavor));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscSFWindowSetFlavorType_Window(PetscSF sf, PetscSFWindowFlavorType flavor)
{
  PetscSF_Window *w = (PetscSF_Window *)sf->data;

  PetscFunctionBegin;
  w->flavor = flavor;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscSFWindowGetFlavorType - Get  `PETSCSFWINDOW` flavor type for `PetscSF` communication

  Logically Collective

  Input Parameter:
. sf - star forest for communication of type `PETSCSFWINDOW`

  Output Parameter:
. flavor - flavor type

  Level: advanced

.seealso: `PetscSF`, `PETSCSFWINDOW`, `PetscSFSetFromOptions()`, `PetscSFWindowSetFlavorType()`
@*/
PetscErrorCode PetscSFWindowGetFlavorType(PetscSF sf, PetscSFWindowFlavorType *flavor)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(sf, PETSCSF_CLASSID, 1);
  PetscAssertPointer(flavor, 2);
  PetscUseMethod(sf, "PetscSFWindowGetFlavorType_C", (PetscSF, PetscSFWindowFlavorType *), (sf, flavor));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscSFWindowGetFlavorType_Window(PetscSF sf, PetscSFWindowFlavorType *flavor)
{
  PetscSF_Window *w = (PetscSF_Window *)sf->data;

  PetscFunctionBegin;
  *flavor = w->flavor;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscSFWindowSetSyncType - Set synchronization type for `PetscSF` communication of type  `PETSCSFWINDOW`

  Logically Collective

  Input Parameters:
+ sf   - star forest for communication
- sync - synchronization type

  Options Database Key:
. -sf_window_sync <sync> - sets the synchronization type FENCE, LOCK, or ACTIVE (see `PetscSFWindowSyncType`)

  Level: advanced

.seealso: `PetscSF`, `PETSCSFWINDOW`, `PetscSFSetFromOptions()`, `PetscSFWindowGetSyncType()`, `PetscSFWindowSyncType`
@*/
PetscErrorCode PetscSFWindowSetSyncType(PetscSF sf, PetscSFWindowSyncType sync)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(sf, PETSCSF_CLASSID, 1);
  PetscValidLogicalCollectiveEnum(sf, sync, 2);
  PetscTryMethod(sf, "PetscSFWindowSetSyncType_C", (PetscSF, PetscSFWindowSyncType), (sf, sync));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscSFWindowSetSyncType_Window(PetscSF sf, PetscSFWindowSyncType sync)
{
  PetscSF_Window *w = (PetscSF_Window *)sf->data;

  PetscFunctionBegin;
  w->sync = sync;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscSFWindowGetSyncType - Get synchronization type for `PetscSF` communication of type `PETSCSFWINDOW`

  Logically Collective

  Input Parameter:
. sf - star forest for communication

  Output Parameter:
. sync - synchronization type

  Level: advanced

.seealso: `PetscSF`, `PETSCSFWINDOW`, `PetscSFSetFromOptions()`, `PetscSFWindowSetSyncType()`, `PetscSFWindowSyncType`
@*/
PetscErrorCode PetscSFWindowGetSyncType(PetscSF sf, PetscSFWindowSyncType *sync)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(sf, PETSCSF_CLASSID, 1);
  PetscAssertPointer(sync, 2);
  PetscUseMethod(sf, "PetscSFWindowGetSyncType_C", (PetscSF, PetscSFWindowSyncType *), (sf, sync));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscSFWindowGetSyncType_Window(PetscSF sf, PetscSFWindowSyncType *sync)
{
  PetscSF_Window *w = (PetscSF_Window *)sf->data;

  PetscFunctionBegin;
  *sync = w->sync;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscSFWindowSetInfo - Set the `MPI_Info` handle that will be used for subsequent windows allocation

  Logically Collective

  Input Parameters:
+ sf   - star forest for communication
- info - `MPI_Info` handle

  Level: advanced

  Note:
  The info handle is duplicated with a call to `MPI_Info_dup()` unless info = `MPI_INFO_NULL`.

.seealso: `PetscSF`, `PETSCSFWINDOW`, `PetscSFSetFromOptions()`, `PetscSFWindowGetInfo()`
@*/
PetscErrorCode PetscSFWindowSetInfo(PetscSF sf, MPI_Info info)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(sf, PETSCSF_CLASSID, 1);
  PetscTryMethod(sf, "PetscSFWindowSetInfo_C", (PetscSF, MPI_Info), (sf, info));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscSFWindowSetInfo_Window(PetscSF sf, MPI_Info info)
{
  PetscSF_Window *w = (PetscSF_Window *)sf->data;

  PetscFunctionBegin;
  if (w->info != MPI_INFO_NULL) PetscCallMPI(MPI_Info_free(&w->info));
  if (info != MPI_INFO_NULL) PetscCallMPI(MPI_Info_dup(info, &w->info));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscSFWindowGetInfo - Get the `MPI_Info` handle used for windows allocation

  Logically Collective

  Input Parameter:
. sf - star forest for communication

  Output Parameter:
. info - `MPI_Info` handle

  Level: advanced

  Note:
  If `PetscSFWindowSetInfo()` has not be called, this returns `MPI_INFO_NULL`

.seealso: `PetscSF`, `PETSCSFWINDOW`, `PetscSFSetFromOptions()`, `PetscSFWindowSetInfo()`
@*/
PetscErrorCode PetscSFWindowGetInfo(PetscSF sf, MPI_Info *info)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(sf, PETSCSF_CLASSID, 1);
  PetscAssertPointer(info, 2);
  PetscUseMethod(sf, "PetscSFWindowGetInfo_C", (PetscSF, MPI_Info *), (sf, info));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscSFWindowGetInfo_Window(PetscSF sf, MPI_Info *info)
{
  PetscSF_Window *w = (PetscSF_Window *)sf->data;

  PetscFunctionBegin;
  *info = w->info;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscSFWindowCreateDynamicSF(PetscSF sf, PetscSF *dynsf)
{
  PetscSFNode *remotes;

  PetscFunctionBegin;
  PetscCall(PetscMalloc1(sf->nranks, &remotes));
  for (PetscInt i = 0; i < sf->nranks; i++) {
    remotes[i].rank  = sf->ranks[i];
    remotes[i].index = 0;
  }
  PetscCall(PetscSFDuplicate(sf, PETSCSF_DUPLICATE_RANKS, dynsf));
  PetscCall(PetscSFSetType(*dynsf, PETSCSFBASIC)); /* break recursion */
  PetscCall(PetscSFSetGraph(*dynsf, 1, sf->nranks, NULL, PETSC_OWN_POINTER, remotes, PETSC_OWN_POINTER));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscSFWindowAttach(PetscSF sf, PetscSFWinLink link, void *rootdata, size_t wsize)
{
  PetscFunctionBegin;
#if defined(PETSC_HAVE_MPI_FEATURE_DYNAMIC_WINDOW)
  {
    PetscSF_Window *w = (PetscSF_Window *)sf->data;
    MPI_Comm        wcomm;
    MPI_Aint        winaddr;
    void           *addr = rootdata;
    PetscMPIInt     nranks;
    // some Open MPI versions do not support MPI_Win_attach(win,NULL,0);
    wcomm = w->window_comm;
    if (addr != NULL) PetscCallMPI(MPI_Win_attach(link->win, addr, wsize));
    link->addr = addr;
    PetscCheck(w->dynsf, wcomm, PETSC_ERR_ORDER, "Must call PetscSFSetUp()");
    PetscCall(PetscSFGetRootRanks(w->dynsf, &nranks, NULL, NULL, NULL, NULL));
    PetscCallMPI(MPI_Get_address(addr, &winaddr));
    if (!link->dyn_target_addr) PetscCall(PetscMalloc1(nranks, &link->dyn_target_addr));
    PetscCall(PetscSFBcastBegin(w->dynsf, MPI_AINT, &winaddr, link->dyn_target_addr, MPI_REPLACE));
    PetscCall(PetscSFBcastEnd(w->dynsf, MPI_AINT, &winaddr, link->dyn_target_addr, MPI_REPLACE));
  }
#else
  SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "dynamic windows not supported");
#endif
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
   PetscSFGetWindow - Get a window for use with a given data type

   Collective

   Input Parameters:
+  sf - star forest
.  unit - data type
.  rootdata - array to be sent
.  leafdata - only used to help uniquely identify windows
.  sync - type of synchronization `PetscSFWindowSyncType`
.  epoch - `PETSC_TRUE` to acquire the window and start an epoch, `PETSC_FALSE` to just acquire the window
.  fenceassert - assert parameter for call to `MPI_Win_fence()`, if sync == `PETSCSF_WINDOW_SYNC_FENCE`
.  postassert - assert parameter for call to `MPI_Win_post()`, if sync == `PETSCSF_WINDOW_SYNC_ACTIVE`
-  startassert - assert parameter for call to `MPI_Win_start()`, if sync == `PETSCSF_WINDOW_SYNC_ACTIVE`

   Output Parameters:
+  target_disp - target_disp argument for RMA calls (significative for `PETSCSF_WINDOW_FLAVOR_DYNAMIC` only)
+  reqs - array of requests (significative for sync == `PETSCSF_WINDOW_SYNC_LOCK` only)
-  win - window

   Level: developer

.seealso: `PetscSF`, `PETSCSFWINDOW`, `PetscSFGetRootRanks()`, `PetscSFWindowGetDataTypes()`
*/

static PetscErrorCode PetscSFGetWindow(PetscSF sf, MPI_Datatype unit, void *rootdata, void *leafdata, PetscSFWindowSyncType sync, PetscBool epoch, PetscMPIInt fenceassert, PetscMPIInt postassert, PetscMPIInt startassert, const MPI_Aint **target_disp, MPI_Request **reqs, MPI_Win *win)
{
  PetscSF_Window *w = (PetscSF_Window *)sf->data;
  MPI_Aint        bytes;
  PetscSFWinLink  link;
  PetscBool       reuse = PETSC_FALSE, update = PETSC_FALSE;
  MPI_Aint        wsize;
  MPI_Comm        wcomm;
  PetscBool       is_empty;

  PetscFunctionBegin;
  PetscCall(PetscSFGetDatatypeSize_Internal(PetscObjectComm((PetscObject)sf), unit, &bytes));
  wsize    = (MPI_Aint)(bytes * sf->nroots);
  wcomm    = w->window_comm;
  is_empty = w->is_empty;
  if (is_empty) {
    if (target_disp) *target_disp = NULL;
    if (reqs) *reqs = NULL;
    *win = MPI_WIN_NULL;
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  if (w->flavor != PETSCSF_WINDOW_FLAVOR_CREATE) reuse = PETSC_TRUE;
  if (PetscDefined(HAVE_MPI_FEATURE_DYNAMIC_WINDOW) && w->flavor == PETSCSF_WINDOW_FLAVOR_DYNAMIC) {
    // first search for a persistent window
    for (link = w->wins; reuse && link; link = link->next) {
      PetscBool match;

      if (!link->persistent) continue;
      match = (link->flavor == w->flavor && link->rootdata == rootdata && link->leafdata == leafdata) ? PETSC_TRUE : PETSC_FALSE;
      if (PetscDefined(USE_DEBUG)) {
        PetscInt matches[2];
        PetscInt all_matches[2];

        matches[0] = match ? 1 : 0;
        matches[1] = match ? -1 : 0;
        PetscCallMPI(MPIU_Allreduce(matches, all_matches, 2, MPIU_INT, MPI_MAX, wcomm));
        all_matches[1] = -all_matches[1];
        PetscCheck(all_matches[0] == all_matches[1], wcomm, PETSC_ERR_ARG_INCOMP,
                   "Inconsistent use across MPI processes of persistent leaf and root data registered with PetscSFRegisterPersistent().\n"
                   "Either the persistent data was changed on a subset of processes (which is not allowed),\n"
                   "or persistent data was not deregistered with PetscSFDeregisterPersistent() before being deallocated");
      }
      if (match) {
        PetscCheck(!link->inuse, wcomm, PETSC_ERR_ARG_WRONGSTATE, "Communication already in progress on persistent root and leaf data");
        PetscCheck(!epoch || !link->epoch, wcomm, PETSC_ERR_ARG_WRONGSTATE, "Communication epoch already open for window");
        PetscCheck(bytes == link->bytes, wcomm, PETSC_ERR_ARG_WRONGSTATE, "Wrong data type for persistent root and leaf data");
        *win = link->win;
        goto found;
      }
    }
  }
  for (link = w->wins; reuse && link; link = link->next) {
    if (w->flavor != link->flavor) continue;
    /* an existing window can be used (1) if it is not in use, (2) if we are
       not asking to start an epoch or it does not have an already started
       epoch, and (3) if it is the right size */
    if (!link->inuse && (!epoch || !link->epoch) && bytes == (MPI_Aint)link->bytes) {
      if (w->flavor == PETSCSF_WINDOW_FLAVOR_DYNAMIC) {
        PetscCall(PetscSFWindowAttach(sf, link, rootdata, wsize));
      } else {
        update = PETSC_TRUE;
      }
      link->rootdata = rootdata;
      link->leafdata = leafdata;
      PetscCall(PetscInfo(sf, "Reusing window %" PETSC_INTPTR_T_FMT " of flavor %d for comm %" PETSC_INTPTR_T_FMT "\n", (PETSC_INTPTR_T)link->win, link->flavor, (PETSC_INTPTR_T)wcomm));
      *win = link->win;
      goto found;
    }
  }

  PetscCall(PetscNew(&link));
  link->bytes           = bytes;
  link->next            = w->wins;
  link->flavor          = w->flavor;
  link->dyn_target_addr = NULL;
  link->reqs            = NULL;
  w->wins               = link;
  link->rootdata        = rootdata;
  link->leafdata        = leafdata;
  if (sync == PETSCSF_WINDOW_SYNC_LOCK) {
    PetscCall(PetscMalloc1(sf->nranks, &link->reqs));
    for (PetscMPIInt i = 0; i < sf->nranks; i++) link->reqs[i] = MPI_REQUEST_NULL;
  }
  switch (w->flavor) {
  case PETSCSF_WINDOW_FLAVOR_CREATE:
    PetscCallMPI(MPI_Win_create(rootdata, wsize, (PetscMPIInt)bytes, w->info, wcomm, &link->win));
    link->addr = rootdata;
    break;
#if defined(PETSC_HAVE_MPI_FEATURE_DYNAMIC_WINDOW)
  case PETSCSF_WINDOW_FLAVOR_DYNAMIC:
    PetscCallMPI(MPI_Win_create_dynamic(w->info, wcomm, &link->win));
    PetscCall(PetscSFWindowAttach(sf, link, rootdata, wsize));
    break;
  case PETSCSF_WINDOW_FLAVOR_ALLOCATE:
    PetscCallMPI(MPI_Win_allocate(wsize, (PetscMPIInt)bytes, w->info, wcomm, &link->addr, &link->win));
    update = PETSC_TRUE;
    break;
#endif
#if defined(PETSC_HAVE_MPI_PROCESS_SHARED_MEMORY)
  case PETSCSF_WINDOW_FLAVOR_SHARED:
    PetscCallMPI(MPI_Win_allocate_shared(wsize, (PetscMPIInt)bytes, w->info, wcomm, &link->addr, &link->win));
    update = PETSC_TRUE;
    break;
#endif
  default:
    SETERRQ(wcomm, PETSC_ERR_SUP, "No support for flavor %s", PetscSFWindowFlavorTypes[w->flavor]);
  }
  PetscCall(PetscInfo(sf, "New window %" PETSC_INTPTR_T_FMT " of flavor %d for comm %" PETSC_INTPTR_T_FMT "\n", (PETSC_INTPTR_T)link->win, link->flavor, (PETSC_INTPTR_T)wcomm));
  *win = link->win;

found:

  if (target_disp) *target_disp = link->dyn_target_addr;
  if (reqs) *reqs = link->reqs;
  if (update) { /* locks are needed for the "separate" memory model only, the fence guarantees memory-synchronization */
    PetscMPIInt rank;

    PetscCallMPI(MPI_Comm_rank(wcomm, &rank));
    if (sync == PETSCSF_WINDOW_SYNC_LOCK) PetscCallMPI(MPI_Win_lock(MPI_LOCK_EXCLUSIVE, rank, MPI_MODE_NOCHECK, *win));
    PetscCall(PetscMemcpy(link->addr, rootdata, sf->nroots * bytes));
    if (sync == PETSCSF_WINDOW_SYNC_LOCK) {
      PetscCallMPI(MPI_Win_unlock(rank, *win));
      PetscCallMPI(MPI_Win_fence(0, *win));
    }
  }
  link->inuse = PETSC_TRUE;
  link->epoch = epoch;
  if (epoch) {
    switch (sync) {
    case PETSCSF_WINDOW_SYNC_FENCE:
      PetscCallMPI(MPI_Win_fence(fenceassert, *win));
      break;
    case PETSCSF_WINDOW_SYNC_LOCK: /* Handled outside */
      break;
    case PETSCSF_WINDOW_SYNC_ACTIVE: {
      MPI_Group   ingroup, outgroup;
      PetscMPIInt isize, osize;

      /* Open MPI 4.0.2 with btl=vader does not like calling
         - MPI_Win_complete when ogroup is empty
         - MPI_Win_wait when igroup is empty
         So, we do not even issue the corresponding start and post calls
         The MPI standard (Sec. 11.5.2 of MPI 3.1) only requires that
         start(outgroup) has a matching post(ingroup)
         and this is guaranteed by PetscSF
      */
      PetscCall(PetscSFGetGroups(sf, &ingroup, &outgroup));
      PetscCallMPI(MPI_Group_size(ingroup, &isize));
      PetscCallMPI(MPI_Group_size(outgroup, &osize));
      if (isize) PetscCallMPI(MPI_Win_post(ingroup, postassert, *win));
      if (osize) PetscCallMPI(MPI_Win_start(outgroup, startassert, *win));
    } break;
    default:
      SETERRQ(wcomm, PETSC_ERR_PLIB, "Unknown synchronization type");
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
   PetscSFFindWindow - Finds a window that is already in use

   Not Collective

   Input Parameters:
+  sf - star forest
.  unit - data type
.  rootdata - array with which the window is associated
-  leafdata - only used to help uniquely identify windows

   Output Parameters:
+  win - window
-  reqs - outstanding requests associated to the window

   Level: developer

.seealso: `PetscSF`, `PETSCSFWINDOW`, `PetscSFGetWindow()`, `PetscSFRestoreWindow()`
*/
static PetscErrorCode PetscSFFindWindow(PetscSF sf, MPI_Datatype unit, const void *rootdata, const void *leafdata, MPI_Win *win, MPI_Request **reqs)
{
  PetscSF_Window *w = (PetscSF_Window *)sf->data;
  PetscSFWinLink  link;
  PetscBool       is_empty;
  MPI_Aint        bytes;

  PetscFunctionBegin;
  PetscCall(PetscSFGetDatatypeSize_Internal(PetscObjectComm((PetscObject)sf), unit, &bytes));
  *win     = MPI_WIN_NULL;
  is_empty = w->is_empty;
  if (is_empty) {
    *reqs = NULL;
    *win  = MPI_WIN_NULL;
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  for (link = w->wins; link; link = link->next) {
    if (rootdata == link->rootdata && leafdata == link->leafdata && bytes == link->bytes) {
      PetscCall(PetscInfo(sf, "Window %" PETSC_INTPTR_T_FMT " of flavor %d for comm %" PETSC_INTPTR_T_FMT "\n", (PETSC_INTPTR_T)link->win, link->flavor, (PETSC_INTPTR_T)w->window_comm));
      *win  = link->win;
      *reqs = link->reqs;
      PetscFunctionReturn(PETSC_SUCCESS);
    }
  }
  SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, "Requested window not in use");
}

/*
   PetscSFRestoreWindow - Restores a window obtained with `PetscSFGetWindow()`

   Collective

   Input Parameters:
+  sf - star forest
.  unit - data type
.  array - array associated with window
.  sync - type of synchronization `PetscSFWindowSyncType`
.  epoch - close an epoch, must match argument to `PetscSFGetWindow()`
.  update - if we have to update the local window array
-  win - window

   Level: developer

.seealso: `PetscSF`, `PETSCSFWINDOW`, `PetscSFFindWindow()`
*/
static PetscErrorCode PetscSFRestoreWindow(PetscSF sf, MPI_Datatype unit, void *array, PetscSFWindowSyncType sync, PetscBool epoch, PetscMPIInt fenceassert, PetscBool update, MPI_Win *win)
{
  PetscSF_Window         *w = (PetscSF_Window *)sf->data;
  PetscSFWinLink         *p, link;
  PetscBool               reuse = PETSC_FALSE;
  PetscSFWindowFlavorType flavor;
  void                   *laddr;
  MPI_Aint                bytes;
  MPI_Comm                wcomm;

  PetscFunctionBegin;
  if (*win == MPI_WIN_NULL) PetscFunctionReturn(PETSC_SUCCESS);
  wcomm = w->window_comm;
  for (p = &w->wins; *p; p = &(*p)->next) {
    link = *p;
    if (*win == link->win) {
      PetscCheck(array == link->rootdata, PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, "Matched window, but not array");
      if (epoch != link->epoch) {
        PetscCheck(!epoch, PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, "No epoch to end");
        SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, "Restoring window without ending epoch");
      }
      laddr  = link->addr;
      flavor = link->flavor;
      bytes  = link->bytes;
      if (flavor != PETSCSF_WINDOW_FLAVOR_CREATE) reuse = PETSC_TRUE;
      else {
        *p     = link->next;
        update = PETSC_FALSE;
      } /* remove from list */
      goto found;
    }
  }
  SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, "Requested window not in use");

found:
  PetscCall(PetscInfo(sf, "Window %" PETSC_INTPTR_T_FMT " of flavor %d for comm %" PETSC_INTPTR_T_FMT "\n", (PETSC_INTPTR_T)link->win, link->flavor, (PETSC_INTPTR_T)wcomm));
  if (epoch) {
    switch (sync) {
    case PETSCSF_WINDOW_SYNC_FENCE:
      PetscCallMPI(MPI_Win_fence(fenceassert, *win));
      break;
    case PETSCSF_WINDOW_SYNC_LOCK: /* Handled outside */
      break;
    case PETSCSF_WINDOW_SYNC_ACTIVE: {
      MPI_Group   ingroup, outgroup;
      PetscMPIInt isize, osize;

      /* Open MPI 4.0.2 with btl=wader does not like calling
         - MPI_Win_complete when ogroup is empty
         - MPI_Win_wait when igroup is empty
         The MPI standard (Sec. 11.5.2 of MPI 3.1) only requires that
         - each process who issues a call to MPI_Win_start issues a call to MPI_Win_Complete
         - each process who issues a call to MPI_Win_post issues a call to MPI_Win_Wait
      */
      PetscCall(PetscSFGetGroups(sf, &ingroup, &outgroup));
      PetscCallMPI(MPI_Group_size(ingroup, &isize));
      PetscCallMPI(MPI_Group_size(outgroup, &osize));
      if (osize) PetscCallMPI(MPI_Win_complete(*win));
      if (isize) PetscCallMPI(MPI_Win_wait(*win));
    } break;
    default:
      SETERRQ(wcomm, PETSC_ERR_PLIB, "Unknown synchronization type");
    }
  }
#if defined(PETSC_HAVE_MPI_FEATURE_DYNAMIC_WINDOW)
  if (link->flavor == PETSCSF_WINDOW_FLAVOR_DYNAMIC && !link->persistent) {
    if (link->addr != NULL) PetscCallMPI(MPI_Win_detach(link->win, link->addr));
    link->addr = NULL;
  }
#endif
  if (update) {
    if (sync == PETSCSF_WINDOW_SYNC_LOCK) PetscCallMPI(MPI_Win_fence(MPI_MODE_NOPUT | MPI_MODE_NOSUCCEED, *win));
    PetscCall(PetscMemcpy(array, laddr, sf->nroots * bytes));
  }
  link->epoch = PETSC_FALSE;
  link->inuse = PETSC_FALSE;
  if (!link->persistent) {
    link->rootdata = NULL;
    link->leafdata = NULL;
  }
  if (!reuse) {
    PetscCall(PetscFree(link->dyn_target_addr));
    PetscCall(PetscFree(link->reqs));
    PetscCallMPI(MPI_Win_free(&link->win));
    PetscCall(PetscFree(link));
    *win = MPI_WIN_NULL;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscSFSetUp_Window(PetscSF sf)
{
  PetscSF_Window *w = (PetscSF_Window *)sf->data;
  MPI_Group       ingroup, outgroup;
  MPI_Comm        comm;

  PetscFunctionBegin;
  PetscCall(PetscSFSetUpRanks(sf, MPI_GROUP_EMPTY));
  PetscCall(PetscObjectGetComm((PetscObject)sf, &comm));
  if (w->window_comm == MPI_COMM_NULL) {
    PetscInt    nroots, nleaves, nranks;
    PetscBool   has_empty;
    PetscMPIInt wcommrank;
    PetscSF     dynsf_full = NULL;

    if (w->flavor == PETSCSF_WINDOW_FLAVOR_DYNAMIC) PetscCall(PetscSFWindowCreateDynamicSF(sf, &dynsf_full));

    PetscCall(PetscSFGetGraph(sf, &nroots, &nleaves, NULL, NULL));
    has_empty = (nroots == 0 && nleaves == 0) ? PETSC_TRUE : PETSC_FALSE;
    nranks    = sf->nranks;
    PetscCall(PetscMalloc1(nranks, &w->wcommranks));
    w->is_empty = has_empty;
    PetscCallMPI(MPIU_Allreduce(MPI_IN_PLACE, &has_empty, 1, MPI_C_BOOL, MPI_LOR, comm));
    if (has_empty) {
      PetscMPIInt  rank;
      MPI_Comm     raw_comm;
      PetscSFNode *remotes;

      PetscCallMPI(MPI_Comm_rank(comm, &rank));
      PetscCallMPI(MPI_Comm_split(comm, w->is_empty ? 1 : 0, rank, &raw_comm));
      PetscCall(PetscCommDuplicate(raw_comm, &w->window_comm, NULL));
      PetscCallMPI(MPI_Comm_free(&raw_comm));

      PetscCallMPI(MPI_Comm_rank(w->window_comm, &wcommrank));
      if (!dynsf_full) PetscCall(PetscSFWindowCreateDynamicSF(sf, &dynsf_full));
      PetscCall(PetscSFBcastBegin(dynsf_full, MPI_INT, &wcommrank, w->wcommranks, MPI_REPLACE));
      PetscCall(PetscSFBcastEnd(dynsf_full, MPI_INT, &wcommrank, w->wcommranks, MPI_REPLACE));

      if (w->flavor == PETSCSF_WINDOW_FLAVOR_DYNAMIC) {
        PetscCall(PetscSFCreate(w->window_comm, &w->dynsf));
        PetscCall(PetscSFSetType(w->dynsf, PETSCSFBASIC)); /* break recursion */
        PetscCall(PetscMalloc1(sf->nranks, &remotes));
        for (PetscInt i = 0; i < sf->nranks; i++) {
          remotes[i].rank  = w->wcommranks[i];
          remotes[i].index = 0;
        }
        PetscCall(PetscSFSetGraph(w->dynsf, 1, sf->nranks, NULL, PETSC_OWN_POINTER, remotes, PETSC_OWN_POINTER));
      }
    } else {
      PetscCall(PetscCommDuplicate(PetscObjectComm((PetscObject)sf), &w->window_comm, NULL));
      PetscCall(PetscArraycpy(w->wcommranks, sf->ranks, nranks));
      PetscCall(PetscObjectReference((PetscObject)dynsf_full));
      w->dynsf = dynsf_full;
    }
    if (w->dynsf) PetscCall(PetscSFSetUp(w->dynsf));
    PetscCall(PetscSFDestroy(&dynsf_full));
  }
  switch (w->sync) {
  case PETSCSF_WINDOW_SYNC_ACTIVE:
    PetscCall(PetscSFGetGroups(sf, &ingroup, &outgroup));
  default:
    break;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscSFSetFromOptions_Window(PetscSF sf, PetscOptionItems PetscOptionsObject)
{
  PetscSF_Window         *w      = (PetscSF_Window *)sf->data;
  PetscSFWindowFlavorType flavor = w->flavor;

  PetscFunctionBegin;
  PetscOptionsHeadBegin(PetscOptionsObject, "PetscSF Window options");
  PetscCall(PetscOptionsEnum("-sf_window_sync", "synchronization type to use for PetscSF Window communication", "PetscSFWindowSetSyncType", PetscSFWindowSyncTypes, (PetscEnum)w->sync, (PetscEnum *)&w->sync, NULL));
  PetscCall(PetscOptionsEnum("-sf_window_flavor", "flavor to use for PetscSF Window creation", "PetscSFWindowSetFlavorType", PetscSFWindowFlavorTypes, (PetscEnum)flavor, (PetscEnum *)&flavor, NULL));
  PetscCall(PetscSFWindowSetFlavorType(sf, flavor));
  PetscOptionsHeadEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscSFReset_Window(PetscSF sf)
{
  PetscSF_Window *w = (PetscSF_Window *)sf->data;
  PetscSFDataLink link, next;
  PetscSFWinLink  wlink, wnext;
  PetscInt        i;
  MPI_Comm        wcomm;
  PetscBool       is_empty;

  PetscFunctionBegin;
  for (link = w->link; link; link = next) {
    next = link->next;
    PetscCallMPI(MPI_Type_free(&link->unit));
    for (i = 0; i < sf->nranks; i++) {
      PetscCallMPI(MPI_Type_free(&link->mine[i]));
      PetscCallMPI(MPI_Type_free(&link->remote[i]));
    }
    PetscCall(PetscFree2(link->mine, link->remote));
    PetscCall(PetscFree(link));
  }
  w->link  = NULL;
  wcomm    = w->window_comm;
  is_empty = w->is_empty;
  if (!is_empty) {
    for (wlink = w->wins; wlink; wlink = wnext) {
      wnext = wlink->next;
      PetscCheck(!wlink->inuse, wcomm, PETSC_ERR_ARG_WRONGSTATE, "Window still in use with address %p", (void *)wlink->addr);
      PetscCall(PetscFree(wlink->dyn_target_addr));
      PetscCall(PetscFree(wlink->reqs));
      PetscCallMPI(MPI_Win_free(&wlink->win));
      PetscCall(PetscFree(wlink));
    }
  }
  w->wins = NULL;
  PetscCall(PetscSFDestroy(&w->dynsf));
  if (w->info != MPI_INFO_NULL) PetscCallMPI(MPI_Info_free(&w->info));
  PetscCall(PetscCommDestroy(&w->window_comm));
  PetscCall(PetscFree(w->wcommranks));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscSFRegisterPersistent_Window(PetscSF sf, MPI_Datatype unit, const void *rootdata, const void *leafdata)
{
  PetscSF_Window *w = (PetscSF_Window *)sf->data;
  MPI_Aint        bytes, wsize;
  PetscBool       is_empty;
  PetscSFWinLink  link;

  PetscFunctionBegin;
  PetscCall(PetscSFSetUp(sf));
  if (w->flavor != PETSCSF_WINDOW_FLAVOR_DYNAMIC) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(PetscSFGetDatatypeSize_Internal(PetscObjectComm((PetscObject)sf), unit, &bytes));
  wsize    = (MPI_Aint)(bytes * sf->nroots);
  is_empty = w->is_empty;
  if (is_empty) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(PetscNew(&link));
  link->flavor = w->flavor;
  link->next   = w->wins;
#if defined(PETSC_HAVE_MPI_FEATURE_DYNAMIC_WINDOW)
  {
    MPI_Comm wcomm = w->window_comm;
    PetscCallMPI(MPI_Win_create_dynamic(w->info, wcomm, &link->win));
  }
#endif
  PetscCall(PetscSFWindowAttach(sf, link, (void *)rootdata, wsize));
  link->rootdata   = (void *)rootdata;
  link->leafdata   = (void *)leafdata;
  link->bytes      = bytes;
  link->epoch      = PETSC_FALSE;
  link->inuse      = PETSC_FALSE;
  link->persistent = PETSC_TRUE;
  w->wins          = link;
  if (w->sync == PETSCSF_WINDOW_SYNC_LOCK) {
    PetscInt i;

    PetscCall(PetscMalloc1(sf->nranks, &link->reqs));
    for (i = 0; i < sf->nranks; i++) link->reqs[i] = MPI_REQUEST_NULL;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscSFDeregisterPersistent_Window(PetscSF sf, MPI_Datatype unit, const void *rootdata, const void *leafdata)
{
  PetscSF_Window *w = (PetscSF_Window *)sf->data;
  MPI_Aint        bytes;
  MPI_Comm        wcomm;
  PetscBool       is_empty;
  PetscSFWinLink *p;

  PetscFunctionBegin;
  PetscCall(PetscSFSetUp(sf));
  if (w->flavor != PETSCSF_WINDOW_FLAVOR_DYNAMIC) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(PetscSFGetDatatypeSize_Internal(PetscObjectComm((PetscObject)sf), unit, &bytes));
  wcomm    = w->window_comm;
  is_empty = w->is_empty;
  if (is_empty) PetscFunctionReturn(PETSC_SUCCESS);
  for (p = &w->wins; *p; p = &(*p)->next) {
    PetscSFWinLink link = *p;
    if (link->flavor == w->flavor && link->persistent && link->rootdata == rootdata && link->leafdata == leafdata && link->bytes == bytes) {
      PetscCheck(!link->inuse, wcomm, PETSC_ERR_ARG_WRONGSTATE, "Deregistering a window when communication is still in progress");
      PetscCheck(!link->epoch, wcomm, PETSC_ERR_ARG_WRONGSTATE, "Deregistering a window with an unconcluded epoch");
#if defined(PETSC_HAVE_MPI_FEATURE_DYNAMIC_WINDOW)
      PetscCallMPI(MPI_Win_detach(link->win, link->addr));
      link->addr = NULL;
#endif
      PetscCall(PetscFree(link->dyn_target_addr));
      PetscCall(PetscFree(link->reqs));
      PetscCallMPI(MPI_Win_free(&link->win));
      *p = link->next;
      PetscCall(PetscFree(link));
      break;
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscSFDestroy_Window(PetscSF sf)
{
  PetscFunctionBegin;
  PetscCall(PetscSFReset_Window(sf));
  PetscCall(PetscFree(sf->data));
  PetscCall(PetscObjectComposeFunction((PetscObject)sf, "PetscSFWindowSetSyncType_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)sf, "PetscSFWindowGetSyncType_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)sf, "PetscSFWindowSetFlavorType_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)sf, "PetscSFWindowGetFlavorType_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)sf, "PetscSFWindowSetInfo_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)sf, "PetscSFWindowGetInfo_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)sf, "PetscSFRegisterPersistent_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)sf, "PetscSFDeregisterPersistent_C", NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscSFView_Window(PetscSF sf, PetscViewer viewer)
{
  PetscSF_Window   *w = (PetscSF_Window *)sf->data;
  PetscBool         isascii;
  PetscViewerFormat format;

  PetscFunctionBegin;
  PetscCall(PetscViewerGetFormat(viewer, &format));
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &isascii));
  if (isascii) {
    PetscCall(PetscViewerASCIIPrintf(viewer, "  current flavor=%s synchronization=%s MultiSF sort=%s\n", PetscSFWindowFlavorTypes[w->flavor], PetscSFWindowSyncTypes[w->sync], sf->rankorder ? "rank-order" : "unordered"));
    if (format == PETSC_VIEWER_ASCII_INFO_DETAIL) {
      if (w->info != MPI_INFO_NULL) {
        PetscMPIInt k, nkeys;
        char        key[MPI_MAX_INFO_KEY], value[MPI_MAX_INFO_VAL];

        PetscCallMPI(MPI_Info_get_nkeys(w->info, &nkeys));
        PetscCall(PetscViewerASCIIPrintf(viewer, "    current info with %d keys. Ordered key-value pairs follow:\n", nkeys));
        for (k = 0; k < nkeys; k++) {
          PetscMPIInt flag;

          PetscCallMPI(MPI_Info_get_nthkey(w->info, k, key));
          PetscCallMPI(MPI_Info_get(w->info, key, MPI_MAX_INFO_VAL, value, &flag));
          PetscCheck(flag, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Missing key %s", key);
          PetscCall(PetscViewerASCIIPrintf(viewer, "      %s = %s\n", key, value));
        }
      } else {
        PetscCall(PetscViewerASCIIPrintf(viewer, "    current info=MPI_INFO_NULL\n"));
      }
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscSFDuplicate_Window(PetscSF sf, PetscSFDuplicateOption opt, PetscSF newsf)
{
  PetscSF_Window       *w = (PetscSF_Window *)sf->data;
  PetscSFWindowSyncType synctype;

  PetscFunctionBegin;
  synctype = w->sync;
  /* HACK: Must use FENCE or LOCK when called from PetscSFGetGroups() because ACTIVE here would cause recursion. */
  if (!sf->setupcalled) synctype = PETSCSF_WINDOW_SYNC_LOCK;
  PetscCall(PetscSFWindowSetSyncType(newsf, synctype));
  PetscCall(PetscSFWindowSetFlavorType(newsf, w->flavor));
  PetscCall(PetscSFWindowSetInfo(newsf, w->info));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscSFBcastBegin_Window(PetscSF sf, MPI_Datatype unit, PetscMemType rootmtype, const void *rootdata, PetscMemType leafmtype, void *leafdata, MPI_Op op)
{
  PetscSF_Window     *w = (PetscSF_Window *)sf->data;
  PetscMPIInt         nranks;
  const PetscMPIInt  *ranks;
  const MPI_Aint     *target_disp;
  const MPI_Datatype *mine, *remote;
  MPI_Request        *reqs;
  MPI_Win             win;

  PetscFunctionBegin;
  PetscCheck(op == MPI_REPLACE, PetscObjectComm((PetscObject)sf), PETSC_ERR_SUP, "PetscSFBcastBegin_Window with op!=MPI_REPLACE has not been implemented");
  PetscCall(PetscSFGetRootRanks(sf, &nranks, NULL, NULL, NULL, NULL));
  PetscCall(PetscSFWindowGetDataTypes(sf, unit, &mine, &remote));
  PetscCall(PetscSFGetWindow(sf, unit, (void *)rootdata, leafdata, w->sync, PETSC_TRUE, MPI_MODE_NOPUT | MPI_MODE_NOPRECEDE, MPI_MODE_NOPUT, 0, &target_disp, &reqs, &win));
  ranks = w->wcommranks;
  for (PetscMPIInt i = 0; i < nranks; i++) {
    MPI_Aint tdp = target_disp ? target_disp[i] : 0;
    if (w->sync == PETSCSF_WINDOW_SYNC_LOCK) {
      PetscCallMPI(MPI_Win_lock(MPI_LOCK_SHARED, ranks[i], MPI_MODE_NOCHECK, win));
#if defined(PETSC_HAVE_MPI_RGET)
      PetscCallMPI(MPI_Rget(leafdata, 1, mine[i], ranks[i], tdp, 1, remote[i], win, &reqs[i]));
#else
      PetscCallMPI(MPI_Get(leafdata, 1, mine[i], ranks[i], tdp, 1, remote[i], win));
#endif
    } else {
      CHKMEMQ;
      PetscCallMPI(MPI_Get(leafdata, 1, mine[i], ranks[i], tdp, 1, remote[i], win));
      CHKMEMQ;
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscSFBcastEnd_Window(PetscSF sf, MPI_Datatype unit, const void *rootdata, void *leafdata, MPI_Op op)
{
  PetscSF_Window *w = (PetscSF_Window *)sf->data;
  MPI_Win         win;
  MPI_Request    *reqs = NULL;

  PetscFunctionBegin;
  PetscCall(PetscSFFindWindow(sf, unit, rootdata, leafdata, &win, &reqs));
  if (reqs) PetscCallMPI(MPI_Waitall(sf->nranks, reqs, MPI_STATUSES_IGNORE));
  if (w->sync == PETSCSF_WINDOW_SYNC_LOCK) {
    PetscMPIInt        nranks;
    const PetscMPIInt *ranks;

    PetscCall(PetscSFGetRootRanks(sf, &nranks, NULL, NULL, NULL, NULL));
    ranks = w->wcommranks;
    for (PetscMPIInt i = 0; i < nranks; i++) PetscCallMPI(MPI_Win_unlock(ranks[i], win));
  }
  PetscCall(PetscSFRestoreWindow(sf, unit, (void *)rootdata, w->sync, PETSC_TRUE, MPI_MODE_NOSTORE | MPI_MODE_NOSUCCEED, PETSC_FALSE, &win));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscSFReduceBegin_Window(PetscSF sf, MPI_Datatype unit, PetscMemType leafmtype, const void *leafdata, PetscMemType rootmtype, void *rootdata, MPI_Op op)
{
  PetscSF_Window     *w = (PetscSF_Window *)sf->data;
  PetscMPIInt         nranks;
  const PetscMPIInt  *ranks;
  const MPI_Aint     *target_disp;
  const MPI_Datatype *mine, *remote;
  MPI_Win             win;

  PetscFunctionBegin;
  PetscCall(PetscSFGetRootRanks(sf, &nranks, NULL, NULL, NULL, NULL));
  PetscCall(PetscSFWindowGetDataTypes(sf, unit, &mine, &remote));
  PetscCall(PetscSFWindowOpTranslate(&op));
  PetscCall(PetscSFGetWindow(sf, unit, rootdata, (void *)leafdata, w->sync, PETSC_TRUE, MPI_MODE_NOPRECEDE, 0, 0, &target_disp, NULL, &win));
  ranks = w->wcommranks;
  for (PetscMPIInt i = 0; i < nranks; i++) {
    MPI_Aint tdp = target_disp ? target_disp[i] : 0;

    if (w->sync == PETSCSF_WINDOW_SYNC_LOCK) PetscCallMPI(MPI_Win_lock(MPI_LOCK_SHARED, ranks[i], MPI_MODE_NOCHECK, win));
    PetscCallMPI(MPI_Accumulate((void *)leafdata, 1, mine[i], ranks[i], tdp, 1, remote[i], op, win));
    if (w->sync == PETSCSF_WINDOW_SYNC_LOCK) PetscCallMPI(MPI_Win_unlock(ranks[i], win));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscSFReduceEnd_Window(PetscSF sf, MPI_Datatype unit, const void *leafdata, void *rootdata, MPI_Op op)
{
  PetscSF_Window *w = (PetscSF_Window *)sf->data;
  MPI_Win         win;
  MPI_Request    *reqs = NULL;

  PetscFunctionBegin;
  PetscCall(PetscSFFindWindow(sf, unit, rootdata, leafdata, &win, &reqs));
  if (reqs) PetscCallMPI(MPI_Waitall(sf->nranks, reqs, MPI_STATUSES_IGNORE));
  PetscCall(PetscSFRestoreWindow(sf, unit, rootdata, w->sync, PETSC_TRUE, MPI_MODE_NOSUCCEED, PETSC_TRUE, &win));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscSFFetchAndOpBegin_Window(PetscSF sf, MPI_Datatype unit, PetscMemType rootmtype, void *rootdata, PetscMemType leafmtype, const void *leafdata, void *leafupdate, MPI_Op op)
{
  PetscMPIInt         nranks;
  const PetscMPIInt  *ranks;
  const MPI_Datatype *mine, *remote;
  const MPI_Aint     *target_disp;
  MPI_Win             win;
  PetscSF_Window     *w = (PetscSF_Window *)sf->data;
#if !defined(PETSC_HAVE_MPI_GET_ACCUMULATE)
  PetscSFWindowFlavorType oldf;
#endif

  PetscFunctionBegin;
  PetscCall(PetscSFGetRootRanks(sf, &nranks, NULL, NULL, NULL, NULL));
  PetscCall(PetscSFWindowGetDataTypes(sf, unit, &mine, &remote));
  PetscCall(PetscSFWindowOpTranslate(&op));
#if !defined(PETSC_HAVE_MPI_GET_ACCUMULATE)
  /* FetchAndOp without MPI_Get_Accumulate requires locking.
     we create a new window every time to not interfere with user-defined MPI_Info which may have used "no_locks"="true" */
  oldf      = w->flavor;
  w->flavor = PETSCSF_WINDOW_FLAVOR_CREATE;
  PetscCall(PetscSFGetWindow(sf, unit, rootdata, (void *)leafdata, PETSCSF_WINDOW_SYNC_LOCK, PETSC_FALSE, 0, 0, 0, &target_disp, NULL, &win));
#else
  PetscCall(PetscSFGetWindow(sf, unit, rootdata, (void *)leafdata, w->sync, PETSC_TRUE, MPI_MODE_NOPRECEDE, 0, 0, &target_disp, NULL, &win));
#endif
  ranks = w->wcommranks;
  for (PetscMPIInt i = 0; i < nranks; i++) {
    MPI_Aint tdp = target_disp ? target_disp[i] : 0;

#if !defined(PETSC_HAVE_MPI_GET_ACCUMULATE)
    PetscCallMPI(MPI_Win_lock(MPI_LOCK_EXCLUSIVE, ranks[i], 0, win));
    PetscCallMPI(MPI_Get(leafupdate, 1, mine[i], ranks[i], tdp, 1, remote[i], win));
    PetscCallMPI(MPI_Accumulate((void *)leafdata, 1, mine[i], ranks[i], tdp, 1, remote[i], op, win));
    PetscCallMPI(MPI_Win_unlock(ranks[i], win));
#else
    if (w->sync == PETSCSF_WINDOW_SYNC_LOCK) PetscCallMPI(MPI_Win_lock(MPI_LOCK_SHARED, ranks[i], 0, win));
    PetscCallMPI(MPI_Get_accumulate((void *)leafdata, 1, mine[i], leafupdate, 1, mine[i], ranks[i], tdp, 1, remote[i], op, win));
    if (w->sync == PETSCSF_WINDOW_SYNC_LOCK) PetscCallMPI(MPI_Win_unlock(ranks[i], win));
#endif
  }
#if !defined(PETSC_HAVE_MPI_GET_ACCUMULATE)
  w->flavor = oldf;
#endif
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscSFFetchAndOpEnd_Window(PetscSF sf, MPI_Datatype unit, void *rootdata, const void *leafdata, void *leafupdate, MPI_Op op)
{
  MPI_Win win;
#if defined(PETSC_HAVE_MPI_GET_ACCUMULATE)
  PetscSF_Window *w = (PetscSF_Window *)sf->data;
#endif
  MPI_Request *reqs = NULL;

  PetscFunctionBegin;
  PetscCall(PetscSFFindWindow(sf, unit, rootdata, leafdata, &win, &reqs));
  if (reqs) PetscCallMPI(MPI_Waitall(sf->nranks, reqs, MPI_STATUSES_IGNORE));
#if defined(PETSC_HAVE_MPI_GET_ACCUMULATE)
  PetscCall(PetscSFRestoreWindow(sf, unit, rootdata, w->sync, PETSC_TRUE, MPI_MODE_NOSUCCEED, PETSC_TRUE, &win));
#else
  PetscCall(PetscSFRestoreWindow(sf, unit, rootdata, PETSCSF_WINDOW_SYNC_LOCK, PETSC_FALSE, 0, PETSC_TRUE, &win));
#endif
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode PetscSFCreate_Window(PetscSF sf)
{
  PetscSF_Window *w = (PetscSF_Window *)sf->data;

  PetscFunctionBegin;
  sf->ops->SetUp           = PetscSFSetUp_Window;
  sf->ops->SetFromOptions  = PetscSFSetFromOptions_Window;
  sf->ops->Reset           = PetscSFReset_Window;
  sf->ops->Destroy         = PetscSFDestroy_Window;
  sf->ops->View            = PetscSFView_Window;
  sf->ops->Duplicate       = PetscSFDuplicate_Window;
  sf->ops->BcastBegin      = PetscSFBcastBegin_Window;
  sf->ops->BcastEnd        = PetscSFBcastEnd_Window;
  sf->ops->ReduceBegin     = PetscSFReduceBegin_Window;
  sf->ops->ReduceEnd       = PetscSFReduceEnd_Window;
  sf->ops->FetchAndOpBegin = PetscSFFetchAndOpBegin_Window;
  sf->ops->FetchAndOpEnd   = PetscSFFetchAndOpEnd_Window;

  PetscCall(PetscNew(&w));
  sf->data       = (void *)w;
  w->sync        = PETSCSF_WINDOW_SYNC_FENCE;
  w->flavor      = PETSCSF_WINDOW_FLAVOR_CREATE;
  w->info        = MPI_INFO_NULL;
  w->window_comm = MPI_COMM_NULL;

  PetscCall(PetscObjectComposeFunction((PetscObject)sf, "PetscSFWindowSetSyncType_C", PetscSFWindowSetSyncType_Window));
  PetscCall(PetscObjectComposeFunction((PetscObject)sf, "PetscSFWindowGetSyncType_C", PetscSFWindowGetSyncType_Window));
  PetscCall(PetscObjectComposeFunction((PetscObject)sf, "PetscSFWindowSetFlavorType_C", PetscSFWindowSetFlavorType_Window));
  PetscCall(PetscObjectComposeFunction((PetscObject)sf, "PetscSFWindowGetFlavorType_C", PetscSFWindowGetFlavorType_Window));
  PetscCall(PetscObjectComposeFunction((PetscObject)sf, "PetscSFWindowSetInfo_C", PetscSFWindowSetInfo_Window));
  PetscCall(PetscObjectComposeFunction((PetscObject)sf, "PetscSFWindowGetInfo_C", PetscSFWindowGetInfo_Window));
  PetscCall(PetscObjectComposeFunction((PetscObject)sf, "PetscSFRegisterPersistent_C", PetscSFRegisterPersistent_Window));
  PetscCall(PetscObjectComposeFunction((PetscObject)sf, "PetscSFDeregisterPersistent_C", PetscSFDeregisterPersistent_Window));

#if defined(PETSC_HAVE_OPENMPI)
  #if PETSC_PKG_OPENMPI_VERSION_LE(1, 6, 0)
  {
    PetscBool ackbug = PETSC_FALSE;

    PetscCall(PetscOptionsGetBool(NULL, NULL, "-acknowledge_ompi_onesided_bug", &ackbug, NULL));
    PetscCheck(ackbug, PetscObjectComm((PetscObject)sf), PETSC_ERR_LIB, "Open MPI is known to be buggy (https://svn.open-mpi.org/trac/ompi/ticket/1905 and 2656), use -acknowledge_ompi_onesided_bug to proceed");
    PetscCall(PetscInfo(sf, "Acknowledged Open MPI bug, proceeding anyway. Expect memory corruption.\n"));
  }
  #endif
#endif
  PetscFunctionReturn(PETSC_SUCCESS);
}
