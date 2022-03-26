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
} PetscSF_Window;

struct _n_PetscSFDataLink {
  MPI_Datatype    unit;
  MPI_Datatype    *mine;
  MPI_Datatype    *remote;
  PetscSFDataLink next;
};

struct _n_PetscSFWinLink {
  PetscBool               inuse;
  size_t                  bytes;
  void                    *addr;
  void                    *paddr;
  MPI_Win                 win;
  MPI_Request             *reqs;
  PetscSFWindowFlavorType flavor;
  MPI_Aint                *dyn_target_addr;
  PetscBool               epoch;
  PetscSFWinLink          next;
};

const char *const PetscSFWindowSyncTypes[] = {"FENCE","LOCK","ACTIVE","PetscSFWindowSyncType","PETSCSF_WINDOW_SYNC_",NULL};
const char *const PetscSFWindowFlavorTypes[] = {"CREATE","DYNAMIC","ALLOCATE","SHARED","PetscSFWindowFlavorType","PETSCSF_WINDOW_FLAVOR_",NULL};

/* Built-in MPI_Ops act elementwise inside MPI_Accumulate, but cannot be used with composite types inside collectives (MPI_Allreduce) */
static PetscErrorCode PetscSFWindowOpTranslate(MPI_Op *op)
{
  PetscFunctionBegin;
  if (*op == MPIU_SUM) *op = MPI_SUM;
  else if (*op == MPIU_MAX) *op = MPI_MAX;
  else if (*op == MPIU_MIN) *op = MPI_MIN;
  PetscFunctionReturn(0);
}

/*@C
   PetscSFWindowGetDataTypes - gets composite local and remote data types for each rank

   Not Collective

   Input Parameters:
+  sf - star forest
-  unit - data type for each node

   Output Parameters:
+  localtypes - types describing part of local leaf buffer referencing each remote rank
-  remotetypes - types describing part of remote root buffer referenced for each remote rank

   Level: developer

.seealso: PetscSFSetGraph(), PetscSFView()
@*/
static PetscErrorCode PetscSFWindowGetDataTypes(PetscSF sf,MPI_Datatype unit,const MPI_Datatype **localtypes,const MPI_Datatype **remotetypes)
{
  PetscSF_Window    *w = (PetscSF_Window*)sf->data;
  PetscSFDataLink   link;
  PetscInt          i,nranks;
  const PetscInt    *roffset,*rmine,*rremote;
  const PetscMPIInt *ranks;

  PetscFunctionBegin;
  /* Look for types in cache */
  for (link=w->link; link; link=link->next) {
    PetscBool match;
    PetscCall(MPIPetsc_Type_compare(unit,link->unit,&match));
    if (match) {
      *localtypes  = link->mine;
      *remotetypes = link->remote;
      PetscFunctionReturn(0);
    }
  }

  /* Create new composite types for each send rank */
  PetscCall(PetscSFGetRootRanks(sf,&nranks,&ranks,&roffset,&rmine,&rremote));
  PetscCall(PetscNew(&link));
  PetscCallMPI(MPI_Type_dup(unit,&link->unit));
  PetscCall(PetscMalloc2(nranks,&link->mine,nranks,&link->remote));
  for (i=0; i<nranks; i++) {
    PetscInt    rcount = roffset[i+1] - roffset[i];
    PetscMPIInt *rmine,*rremote;
#if !defined(PETSC_USE_64BIT_INDICES)
    rmine   = sf->rmine + sf->roffset[i];
    rremote = sf->rremote + sf->roffset[i];
#else
    PetscInt j;
    PetscCall(PetscMalloc2(rcount,&rmine,rcount,&rremote));
    for (j=0; j<rcount; j++) {
      PetscCall(PetscMPIIntCast(sf->rmine[sf->roffset[i]+j],rmine+j));
      PetscCall(PetscMPIIntCast(sf->rremote[sf->roffset[i]+j],rremote+j));
    }
#endif

    PetscCallMPI(MPI_Type_create_indexed_block(rcount,1,rmine,link->unit,&link->mine[i]));
    PetscCallMPI(MPI_Type_create_indexed_block(rcount,1,rremote,link->unit,&link->remote[i]));
#if defined(PETSC_USE_64BIT_INDICES)
    PetscCall(PetscFree2(rmine,rremote));
#endif
    PetscCallMPI(MPI_Type_commit(&link->mine[i]));
    PetscCallMPI(MPI_Type_commit(&link->remote[i]));
  }
  link->next = w->link;
  w->link    = link;

  *localtypes  = link->mine;
  *remotetypes = link->remote;
  PetscFunctionReturn(0);
}

/*@C
   PetscSFWindowSetFlavorType - Set flavor type for MPI_Win creation

   Logically Collective

   Input Parameters:
+  sf - star forest for communication
-  flavor - flavor type

   Options Database Key:
.  -sf_window_flavor <flavor> - sets the flavor type CREATE, DYNAMIC, ALLOCATE or SHARED (see PetscSFWindowFlavorType)

   Level: advanced

   Notes: Windows reusage follow this rules:

     PETSCSF_WINDOW_FLAVOR_CREATE: creates a new window every time, uses MPI_Win_create

     PETSCSF_WINDOW_FLAVOR_DYNAMIC: uses MPI_Win_create_dynamic/MPI_Win_attach and tries to reuse windows by comparing the root array. Intended to be used on repeated applications of the same SF, e.g.
       for i=1 to K
         PetscSFOperationBegin(rootdata1,leafdata_whatever);
         PetscSFOperationEnd(rootdata1,leafdata_whatever);
         ...
         PetscSFOperationBegin(rootdataN,leafdata_whatever);
         PetscSFOperationEnd(rootdataN,leafdata_whatever);
       endfor
       The following pattern will instead raise an error
         PetscSFOperationBegin(rootdata1,leafdata_whatever);
         PetscSFOperationEnd(rootdata1,leafdata_whatever);
         PetscSFOperationBegin(rank ? rootdata1 : rootdata2,leafdata_whatever);
         PetscSFOperationEnd(rank ? rootdata1 : rootdata2,leafdata_whatever);

     PETSCSF_WINDOW_FLAVOR_ALLOCATE: uses MPI_Win_allocate, reuses any pre-existing window which fits the data and it is not in use

     PETSCSF_WINDOW_FLAVOR_SHARED: uses MPI_Win_allocate_shared, reusage policy as for PETSCSF_WINDOW_FLAVOR_ALLOCATE

.seealso: PetscSFSetFromOptions(), PetscSFWindowGetFlavorType()
@*/
PetscErrorCode PetscSFWindowSetFlavorType(PetscSF sf,PetscSFWindowFlavorType flavor)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(sf,PETSCSF_CLASSID,1);
  PetscValidLogicalCollectiveEnum(sf,flavor,2);
  PetscCall(PetscTryMethod(sf,"PetscSFWindowSetFlavorType_C",(PetscSF,PetscSFWindowFlavorType),(sf,flavor)));
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSFWindowSetFlavorType_Window(PetscSF sf,PetscSFWindowFlavorType flavor)
{
  PetscSF_Window *w = (PetscSF_Window*)sf->data;

  PetscFunctionBegin;
  w->flavor = flavor;
  PetscFunctionReturn(0);
}

/*@C
   PetscSFWindowGetFlavorType - Get flavor type for PetscSF communication

   Logically Collective

   Input Parameter:
.  sf - star forest for communication

   Output Parameter:
.  flavor - flavor type

   Level: advanced

.seealso: PetscSFSetFromOptions(), PetscSFWindowSetFlavorType()
@*/
PetscErrorCode PetscSFWindowGetFlavorType(PetscSF sf,PetscSFWindowFlavorType *flavor)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(sf,PETSCSF_CLASSID,1);
  PetscValidPointer(flavor,2);
  PetscCall(PetscUseMethod(sf,"PetscSFWindowGetFlavorType_C",(PetscSF,PetscSFWindowFlavorType*),(sf,flavor)));
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSFWindowGetFlavorType_Window(PetscSF sf,PetscSFWindowFlavorType *flavor)
{
  PetscSF_Window *w = (PetscSF_Window*)sf->data;

  PetscFunctionBegin;
  *flavor = w->flavor;
  PetscFunctionReturn(0);
}

/*@C
   PetscSFWindowSetSyncType - Set synchronization type for PetscSF communication

   Logically Collective

   Input Parameters:
+  sf - star forest for communication
-  sync - synchronization type

   Options Database Key:
.  -sf_window_sync <sync> - sets the synchronization type FENCE, LOCK, or ACTIVE (see PetscSFWindowSyncType)

   Level: advanced

.seealso: PetscSFSetFromOptions(), PetscSFWindowGetSyncType()
@*/
PetscErrorCode PetscSFWindowSetSyncType(PetscSF sf,PetscSFWindowSyncType sync)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(sf,PETSCSF_CLASSID,1);
  PetscValidLogicalCollectiveEnum(sf,sync,2);
  PetscCall(PetscTryMethod(sf,"PetscSFWindowSetSyncType_C",(PetscSF,PetscSFWindowSyncType),(sf,sync)));
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSFWindowSetSyncType_Window(PetscSF sf,PetscSFWindowSyncType sync)
{
  PetscSF_Window *w = (PetscSF_Window*)sf->data;

  PetscFunctionBegin;
  w->sync = sync;
  PetscFunctionReturn(0);
}

/*@C
   PetscSFWindowGetSyncType - Get synchronization type for PetscSF communication

   Logically Collective

   Input Parameter:
.  sf - star forest for communication

   Output Parameter:
.  sync - synchronization type

   Level: advanced

.seealso: PetscSFSetFromOptions(), PetscSFWindowSetSyncType()
@*/
PetscErrorCode PetscSFWindowGetSyncType(PetscSF sf,PetscSFWindowSyncType *sync)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(sf,PETSCSF_CLASSID,1);
  PetscValidPointer(sync,2);
  PetscCall(PetscUseMethod(sf,"PetscSFWindowGetSyncType_C",(PetscSF,PetscSFWindowSyncType*),(sf,sync)));
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSFWindowGetSyncType_Window(PetscSF sf,PetscSFWindowSyncType *sync)
{
  PetscSF_Window *w = (PetscSF_Window*)sf->data;

  PetscFunctionBegin;
  *sync = w->sync;
  PetscFunctionReturn(0);
}

/*@C
   PetscSFWindowSetInfo - Set the MPI_Info handle that will be used for subsequent windows allocation

   Logically Collective

   Input Parameters:
+  sf - star forest for communication
-  info - MPI_Info handle

   Level: advanced

   Notes: the info handle is duplicated with a call to MPI_Info_dup unless info = MPI_INFO_NULL.

.seealso: PetscSFSetFromOptions(), PetscSFWindowGetInfo()
@*/
PetscErrorCode PetscSFWindowSetInfo(PetscSF sf,MPI_Info info)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(sf,PETSCSF_CLASSID,1);
  PetscCall(PetscTryMethod(sf,"PetscSFWindowSetInfo_C",(PetscSF,MPI_Info),(sf,info)));
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSFWindowSetInfo_Window(PetscSF sf,MPI_Info info)
{
  PetscSF_Window *w = (PetscSF_Window*)sf->data;

  PetscFunctionBegin;
  if (w->info != MPI_INFO_NULL) {
    PetscCallMPI(MPI_Info_free(&w->info));
  }
  if (info != MPI_INFO_NULL) {
    PetscCallMPI(MPI_Info_dup(info,&w->info));
  }
  PetscFunctionReturn(0);
}

/*@C
   PetscSFWindowGetInfo - Get the MPI_Info handle used for windows allocation

   Logically Collective

   Input Parameter:
.  sf - star forest for communication

   Output Parameter:
.  info - MPI_Info handle

   Level: advanced

   Notes: if PetscSFWindowSetInfo() has not be called, this returns MPI_INFO_NULL

.seealso: PetscSFSetFromOptions(), PetscSFWindowSetInfo()
@*/
PetscErrorCode PetscSFWindowGetInfo(PetscSF sf,MPI_Info *info)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(sf,PETSCSF_CLASSID,1);
  PetscValidPointer(info,2);
  PetscCall(PetscUseMethod(sf,"PetscSFWindowGetInfo_C",(PetscSF,MPI_Info*),(sf,info)));
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSFWindowGetInfo_Window(PetscSF sf,MPI_Info *info)
{
  PetscSF_Window *w = (PetscSF_Window*)sf->data;

  PetscFunctionBegin;
  *info = w->info;
  PetscFunctionReturn(0);
}

/*
   PetscSFGetWindow - Get a window for use with a given data type

   Collective on PetscSF

   Input Parameters:
+  sf - star forest
.  unit - data type
.  array - array to be sent
.  sync - type of synchronization PetscSFWindowSyncType
.  epoch - PETSC_TRUE to acquire the window and start an epoch, PETSC_FALSE to just acquire the window
.  fenceassert - assert parameter for call to MPI_Win_fence(), if sync == PETSCSF_WINDOW_SYNC_FENCE
.  postassert - assert parameter for call to MPI_Win_post(), if sync == PETSCSF_WINDOW_SYNC_ACTIVE
-  startassert - assert parameter for call to MPI_Win_start(), if sync == PETSCSF_WINDOW_SYNC_ACTIVE

   Output Parameters:
+  target_disp - target_disp argument for RMA calls (significative for PETSCSF_WINDOW_FLAVOR_DYNAMIC only)
+  reqs - array of requests (significative for sync == PETSCSF_WINDOW_SYNC_LOCK only)
-  win - window

   Level: developer
.seealso: PetscSFGetRootRanks(), PetscSFWindowGetDataTypes()
*/
static PetscErrorCode PetscSFGetWindow(PetscSF sf,MPI_Datatype unit,void *array,PetscSFWindowSyncType sync,PetscBool epoch,PetscMPIInt fenceassert,PetscMPIInt postassert,PetscMPIInt startassert,const MPI_Aint **target_disp, MPI_Request **reqs, MPI_Win *win)
{
  PetscSF_Window *w = (PetscSF_Window*)sf->data;
  MPI_Aint       lb,lb_true,bytes,bytes_true;
  PetscSFWinLink link;
#if defined(PETSC_HAVE_MPI_FEATURE_DYNAMIC_WINDOW)
  MPI_Aint       winaddr;
  PetscInt       nranks;
#endif
  PetscBool      reuse = PETSC_FALSE, update = PETSC_FALSE;
  PetscBool      dummy[2];
  MPI_Aint       wsize;

  PetscFunctionBegin;
  PetscCallMPI(MPI_Type_get_extent(unit,&lb,&bytes));
  PetscCallMPI(MPI_Type_get_true_extent(unit,&lb_true,&bytes_true));
  PetscCheckFalse(lb != 0 || lb_true != 0,PetscObjectComm((PetscObject)sf),PETSC_ERR_SUP,"No support for unit type with nonzero lower bound, write petsc-maint@mcs.anl.gov if you want this feature");
  PetscCheckFalse(bytes != bytes_true,PetscObjectComm((PetscObject)sf),PETSC_ERR_SUP,"No support for unit type with modified extent, write petsc-maint@mcs.anl.gov if you want this feature");
  if (w->flavor != PETSCSF_WINDOW_FLAVOR_CREATE) reuse = PETSC_TRUE;
  for (link=w->wins; reuse && link; link=link->next) {
    PetscBool winok = PETSC_FALSE;
    if (w->flavor != link->flavor) continue;
    switch (w->flavor) {
    case PETSCSF_WINDOW_FLAVOR_DYNAMIC: /* check available matching array, error if in use (we additionally check that the matching condition is the same across processes) */
      if (array == link->addr) {
        if (PetscDefined(USE_DEBUG)) {
          dummy[0] = PETSC_TRUE;
          dummy[1] = PETSC_TRUE;
          PetscCallMPI(MPI_Allreduce(MPI_IN_PLACE,dummy,1,MPIU_BOOL,MPI_LAND,PetscObjectComm((PetscObject)sf)));
          PetscCallMPI(MPI_Allreduce(MPI_IN_PLACE,dummy+1,1,MPIU_BOOL,MPI_LOR ,PetscObjectComm((PetscObject)sf)));
          PetscCheckFalse(dummy[0] != dummy[1],PetscObjectComm((PetscObject)sf),PETSC_ERR_SUP,"PETSCSF_WINDOW_FLAVOR_DYNAMIC requires root pointers to be consistently used across the comm. Use PETSCSF_WINDOW_FLAVOR_CREATE or PETSCSF_WINDOW_FLAVOR_ALLOCATE instead");
        }
        PetscCheck(!link->inuse,PetscObjectComm((PetscObject)sf),PETSC_ERR_PLIB,"Window in use");
        PetscCheckFalse(epoch && link->epoch,PetscObjectComm((PetscObject)sf),PETSC_ERR_PLIB,"Window epoch not finished");
        winok = PETSC_TRUE;
        link->paddr = array;
      } else if (PetscDefined(USE_DEBUG)) {
        dummy[0] = PETSC_FALSE;
        dummy[1] = PETSC_FALSE;
        PetscCallMPI(MPI_Allreduce(MPI_IN_PLACE,dummy  ,1,MPIU_BOOL,MPI_LAND,PetscObjectComm((PetscObject)sf)));
        PetscCallMPI(MPI_Allreduce(MPI_IN_PLACE,dummy+1,1,MPIU_BOOL,MPI_LOR ,PetscObjectComm((PetscObject)sf)));
        PetscCheckFalse(dummy[0] != dummy[1],PetscObjectComm((PetscObject)sf),PETSC_ERR_SUP,"PETSCSF_WINDOW_FLAVOR_DYNAMIC requires root pointers to be consistently used across the comm. Use PETSCSF_WINDOW_FLAVOR_CREATE or PETSCSF_WINDOW_FLAVOR_ALLOCATE instead");
      }
      break;
    case PETSCSF_WINDOW_FLAVOR_ALLOCATE: /* check available by matching size, allocate if in use */
    case PETSCSF_WINDOW_FLAVOR_SHARED:
      if (!link->inuse && bytes == (MPI_Aint)link->bytes) {
        update = PETSC_TRUE;
        link->paddr = array;
        winok = PETSC_TRUE;
      }
      break;
    default: SETERRQ(PetscObjectComm((PetscObject)sf),PETSC_ERR_SUP,"No support for flavor %s",PetscSFWindowFlavorTypes[w->flavor]);
    }
    if (winok) {
      *win = link->win;
      PetscCall(PetscInfo(sf,"Reusing window %" PETSC_MPI_WIN_FMT " of flavor %d for comm %" PETSC_MPI_COMM_FMT "\n",link->win,link->flavor,PetscObjectComm((PetscObject)sf)));
      goto found;
    }
  }

  wsize = (MPI_Aint)bytes*sf->nroots;
  PetscCall(PetscNew(&link));
  link->bytes           = bytes;
  link->next            = w->wins;
  link->flavor          = w->flavor;
  link->dyn_target_addr = NULL;
  link->reqs            = NULL;
  w->wins               = link;
  if (sync == PETSCSF_WINDOW_SYNC_LOCK) {
    PetscInt i;

    PetscCall(PetscMalloc1(sf->nranks,&link->reqs));
    for (i = 0; i < sf->nranks; i++) link->reqs[i] = MPI_REQUEST_NULL;
  }
  switch (w->flavor) {
  case PETSCSF_WINDOW_FLAVOR_CREATE:
    PetscCallMPI(MPI_Win_create(array,wsize,(PetscMPIInt)bytes,w->info,PetscObjectComm((PetscObject)sf),&link->win));
    link->addr  = array;
    link->paddr = array;
    break;
#if defined(PETSC_HAVE_MPI_FEATURE_DYNAMIC_WINDOW)
  case PETSCSF_WINDOW_FLAVOR_DYNAMIC:
    PetscCallMPI(MPI_Win_create_dynamic(w->info,PetscObjectComm((PetscObject)sf),&link->win));
#if defined(PETSC_HAVE_OMPI_MAJOR_VERSION) /* some OpenMPI versions do not support MPI_Win_attach(win,NULL,0); */
    PetscCallMPI(MPI_Win_attach(link->win,wsize ? array : (void*)dummy,wsize));
#else
    PetscCallMPI(MPI_Win_attach(link->win,array,wsize));
#endif
    link->addr  = array;
    link->paddr = array;
    PetscCheck(w->dynsf,PetscObjectComm((PetscObject)sf),PETSC_ERR_ORDER,"Must call PetscSFSetUp()");
    PetscCall(PetscSFSetUp(w->dynsf));
    PetscCall(PetscSFGetRootRanks(w->dynsf,&nranks,NULL,NULL,NULL,NULL));
    PetscCall(PetscMalloc1(nranks,&link->dyn_target_addr));
    PetscCallMPI(MPI_Get_address(array,&winaddr));
    PetscCall(PetscSFBcastBegin(w->dynsf,MPI_AINT,&winaddr,link->dyn_target_addr,MPI_REPLACE));
    PetscCall(PetscSFBcastEnd(w->dynsf,MPI_AINT,&winaddr,link->dyn_target_addr,MPI_REPLACE));
    break;
  case PETSCSF_WINDOW_FLAVOR_ALLOCATE:
    PetscCallMPI(MPI_Win_allocate(wsize,(PetscMPIInt)bytes,w->info,PetscObjectComm((PetscObject)sf),&link->addr,&link->win));
    update = PETSC_TRUE;
    link->paddr = array;
    break;
#endif
#if defined(PETSC_HAVE_MPI_PROCESS_SHARED_MEMORY)
  case PETSCSF_WINDOW_FLAVOR_SHARED:
    PetscCallMPI(MPI_Win_allocate_shared(wsize,(PetscMPIInt)bytes,w->info,PetscObjectComm((PetscObject)sf),&link->addr,&link->win));
    update = PETSC_TRUE;
    link->paddr = array;
    break;
#endif
  default: SETERRQ(PetscObjectComm((PetscObject)sf),PETSC_ERR_SUP,"No support for flavor %s",PetscSFWindowFlavorTypes[w->flavor]);
  }
  PetscCall(PetscInfo(sf,"New window %" PETSC_MPI_WIN_FMT " of flavor %d for comm %" PETSC_MPI_COMM_FMT "\n",link->win,link->flavor,PetscObjectComm((PetscObject)sf)));
  *win = link->win;

found:

  if (target_disp) *target_disp = link->dyn_target_addr;
  if (reqs) *reqs = link->reqs;
  if (update) { /* locks are needed for the "separate" memory model only, the fence guaranties memory-synchronization */
    PetscMPIInt rank;

    PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)sf),&rank));
    if (sync == PETSCSF_WINDOW_SYNC_LOCK) PetscCallMPI(MPI_Win_lock(MPI_LOCK_EXCLUSIVE,rank,MPI_MODE_NOCHECK,*win));
    PetscCall(PetscMemcpy(link->addr,array,sf->nroots*bytes));
    if (sync == PETSCSF_WINDOW_SYNC_LOCK) {
      PetscCallMPI(MPI_Win_unlock(rank,*win));
      PetscCallMPI(MPI_Win_fence(0,*win));
    }
  }
  link->inuse = PETSC_TRUE;
  link->epoch = epoch;
  if (epoch) {
    switch (sync) {
    case PETSCSF_WINDOW_SYNC_FENCE:
      PetscCallMPI(MPI_Win_fence(fenceassert,*win));
      break;
    case PETSCSF_WINDOW_SYNC_LOCK: /* Handled outside */
      break;
    case PETSCSF_WINDOW_SYNC_ACTIVE: {
      MPI_Group   ingroup,outgroup;
      PetscMPIInt isize,osize;

      /* OpenMPI 4.0.2 with btl=vader does not like calling
         - MPI_Win_complete when ogroup is empty
         - MPI_Win_wait when igroup is empty
         So, we do not even issue the corresponding start and post calls
         The MPI standard (Sec. 11.5.2 of MPI 3.1) only requires that
         start(outgroup) has a matching post(ingroup)
         and this is guaranteed by PetscSF
      */
      PetscCall(PetscSFGetGroups(sf,&ingroup,&outgroup));
      PetscCallMPI(MPI_Group_size(ingroup,&isize));
      PetscCallMPI(MPI_Group_size(outgroup,&osize));
      if (isize) PetscCallMPI(MPI_Win_post(ingroup,postassert,*win));
      if (osize) PetscCallMPI(MPI_Win_start(outgroup,startassert,*win));
    } break;
    default: SETERRQ(PetscObjectComm((PetscObject)sf),PETSC_ERR_PLIB,"Unknown synchronization type");
    }
  }
  PetscFunctionReturn(0);
}

/*
   PetscSFFindWindow - Finds a window that is already in use

   Not Collective

   Input Parameters:
+  sf - star forest
.  unit - data type
-  array - array with which the window is associated

   Output Parameters:
+  win - window
-  reqs - outstanding requests associated to the window

   Level: developer

.seealso: PetscSFGetWindow(), PetscSFRestoreWindow()
*/
static PetscErrorCode PetscSFFindWindow(PetscSF sf,MPI_Datatype unit,const void *array,MPI_Win *win,MPI_Request **reqs)
{
  PetscSF_Window *w = (PetscSF_Window*)sf->data;
  PetscSFWinLink link;

  PetscFunctionBegin;
  *win = MPI_WIN_NULL;
  for (link=w->wins; link; link=link->next) {
    if (array == link->paddr) {

      PetscCall(PetscInfo(sf,"Window %" PETSC_MPI_WIN_FMT " of flavor %d for comm %" PETSC_MPI_COMM_FMT "\n",link->win,link->flavor,PetscObjectComm((PetscObject)sf)));
      *win = link->win;
      *reqs = link->reqs;
      PetscFunctionReturn(0);
    }
  }
  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Requested window not in use");
}

/*
   PetscSFRestoreWindow - Restores a window obtained with PetscSFGetWindow()

   Collective

   Input Parameters:
+  sf - star forest
.  unit - data type
.  array - array associated with window
.  sync - type of synchronization PetscSFWindowSyncType
.  epoch - close an epoch, must match argument to PetscSFGetWindow()
.  update - if we have to update the local window array
-  win - window

   Level: developer

.seealso: PetscSFFindWindow()
*/
static PetscErrorCode PetscSFRestoreWindow(PetscSF sf,MPI_Datatype unit,void *array,PetscSFWindowSyncType sync,PetscBool epoch,PetscMPIInt fenceassert,PetscBool update,MPI_Win *win)
{
  PetscSF_Window          *w = (PetscSF_Window*)sf->data;
  PetscSFWinLink          *p,link;
  PetscBool               reuse = PETSC_FALSE;
  PetscSFWindowFlavorType flavor;
  void*                   laddr;
  size_t                  bytes;

  PetscFunctionBegin;
  for (p=&w->wins; *p; p=&(*p)->next) {
    link = *p;
    if (*win == link->win) {
      PetscCheckFalse(array != link->paddr,PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Matched window, but not array");
      if (epoch != link->epoch) {
        PetscCheck(!epoch,PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"No epoch to end");
        else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Restoring window without ending epoch");
      }
      laddr = link->addr;
      flavor = link->flavor;
      bytes = link->bytes;
      if (flavor != PETSCSF_WINDOW_FLAVOR_CREATE) reuse = PETSC_TRUE;
      else { *p = link->next; update = PETSC_FALSE; } /* remove from list */
      goto found;
    }
  }
  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Requested window not in use");

found:
  PetscCall(PetscInfo(sf,"Window %" PETSC_MPI_WIN_FMT " of flavor %d for comm %" PETSC_MPI_COMM_FMT "\n",link->win,link->flavor,PetscObjectComm((PetscObject)sf)));
  if (epoch) {
    switch (sync) {
    case PETSCSF_WINDOW_SYNC_FENCE:
      PetscCallMPI(MPI_Win_fence(fenceassert,*win));
      break;
    case PETSCSF_WINDOW_SYNC_LOCK: /* Handled outside */
      break;
    case PETSCSF_WINDOW_SYNC_ACTIVE: {
      MPI_Group   ingroup,outgroup;
      PetscMPIInt isize,osize;

      /* OpenMPI 4.0.2 with btl=wader does not like calling
         - MPI_Win_complete when ogroup is empty
         - MPI_Win_wait when igroup is empty
         The MPI standard (Sec. 11.5.2 of MPI 3.1) only requires that
         - each process who issues a call to MPI_Win_start issues a call to MPI_Win_Complete
         - each process who issues a call to MPI_Win_post issues a call to MPI_Win_Wait
      */
      PetscCall(PetscSFGetGroups(sf,&ingroup,&outgroup));
      PetscCallMPI(MPI_Group_size(ingroup,&isize));
      PetscCallMPI(MPI_Group_size(outgroup,&osize));
      if (osize) PetscCallMPI(MPI_Win_complete(*win));
      if (isize) PetscCallMPI(MPI_Win_wait(*win));
    } break;
    default: SETERRQ(PetscObjectComm((PetscObject)sf),PETSC_ERR_PLIB,"Unknown synchronization type");
    }
  }
  if (update) {
    if (sync == PETSCSF_WINDOW_SYNC_LOCK) {
      PetscCallMPI(MPI_Win_fence(MPI_MODE_NOPUT|MPI_MODE_NOSUCCEED,*win));
    }
    PetscCall(PetscMemcpy(array,laddr,sf->nroots*bytes));
  }
  link->epoch = PETSC_FALSE;
  link->inuse = PETSC_FALSE;
  link->paddr = NULL;
  if (!reuse) {
    PetscCall(PetscFree(link->dyn_target_addr));
    PetscCall(PetscFree(link->reqs));
    PetscCallMPI(MPI_Win_free(&link->win));
    PetscCall(PetscFree(link));
    *win = MPI_WIN_NULL;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSFSetUp_Window(PetscSF sf)
{
  PetscSF_Window *w = (PetscSF_Window*)sf->data;
  MPI_Group      ingroup,outgroup;

  PetscFunctionBegin;
  PetscCall(PetscSFSetUpRanks(sf,MPI_GROUP_EMPTY));
  if (!w->dynsf) {
    PetscInt    i;
    PetscSFNode *remotes;

    PetscCall(PetscMalloc1(sf->nranks,&remotes));
    for (i=0;i<sf->nranks;i++) {
      remotes[i].rank  = sf->ranks[i];
      remotes[i].index = 0;
    }
    PetscCall(PetscSFDuplicate(sf,PETSCSF_DUPLICATE_RANKS,&w->dynsf));
    PetscCall(PetscSFWindowSetFlavorType(w->dynsf,PETSCSF_WINDOW_FLAVOR_CREATE)); /* break recursion */
    PetscCall(PetscSFSetGraph(w->dynsf,1,sf->nranks,NULL,PETSC_OWN_POINTER,remotes,PETSC_OWN_POINTER));
    PetscCall(PetscLogObjectParent((PetscObject)sf,(PetscObject)w->dynsf));
  }
  switch (w->sync) {
  case PETSCSF_WINDOW_SYNC_ACTIVE:
    PetscCall(PetscSFGetGroups(sf,&ingroup,&outgroup));
  default:
    break;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSFSetFromOptions_Window(PetscOptionItems *PetscOptionsObject,PetscSF sf)
{
  PetscSF_Window          *w = (PetscSF_Window*)sf->data;
  PetscSFWindowFlavorType flavor = w->flavor;

  PetscFunctionBegin;
  PetscCall(PetscOptionsHead(PetscOptionsObject,"PetscSF Window options"));
  PetscCall(PetscOptionsEnum("-sf_window_sync","synchronization type to use for PetscSF Window communication","PetscSFWindowSetSyncType",PetscSFWindowSyncTypes,(PetscEnum)w->sync,(PetscEnum*)&w->sync,NULL));
  PetscCall(PetscOptionsEnum("-sf_window_flavor","flavor to use for PetscSF Window creation","PetscSFWindowSetFlavorType",PetscSFWindowFlavorTypes,(PetscEnum)flavor,(PetscEnum*)&flavor,NULL));
  PetscCall(PetscSFWindowSetFlavorType(sf,flavor));
  PetscCall(PetscOptionsTail());
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSFReset_Window(PetscSF sf)
{
  PetscSF_Window  *w = (PetscSF_Window*)sf->data;
  PetscSFDataLink link,next;
  PetscSFWinLink  wlink,wnext;
  PetscInt        i;

  PetscFunctionBegin;
  for (link=w->link; link; link=next) {
    next = link->next;
    PetscCallMPI(MPI_Type_free(&link->unit));
    for (i=0; i<sf->nranks; i++) {
      PetscCallMPI(MPI_Type_free(&link->mine[i]));
      PetscCallMPI(MPI_Type_free(&link->remote[i]));
    }
    PetscCall(PetscFree2(link->mine,link->remote));
    PetscCall(PetscFree(link));
  }
  w->link = NULL;
  for (wlink=w->wins; wlink; wlink=wnext) {
    wnext = wlink->next;
    PetscCheck(!wlink->inuse,PetscObjectComm((PetscObject)sf),PETSC_ERR_ARG_WRONGSTATE,"Window still in use with address %p",(void*)wlink->addr);
    PetscCall(PetscFree(wlink->dyn_target_addr));
    PetscCall(PetscFree(wlink->reqs));
    PetscCallMPI(MPI_Win_free(&wlink->win));
    PetscCall(PetscFree(wlink));
  }
  w->wins = NULL;
  PetscCall(PetscSFDestroy(&w->dynsf));
  if (w->info != MPI_INFO_NULL) {
    PetscCallMPI(MPI_Info_free(&w->info));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSFDestroy_Window(PetscSF sf)
{
  PetscFunctionBegin;
  PetscCall(PetscSFReset_Window(sf));
  PetscCall(PetscFree(sf->data));
  PetscCall(PetscObjectComposeFunction((PetscObject)sf,"PetscSFWindowSetSyncType_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)sf,"PetscSFWindowGetSyncType_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)sf,"PetscSFWindowSetFlavorType_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)sf,"PetscSFWindowGetFlavorType_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)sf,"PetscSFWindowSetInfo_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)sf,"PetscSFWindowGetInfo_C",NULL));
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSFView_Window(PetscSF sf,PetscViewer viewer)
{
  PetscSF_Window    *w = (PetscSF_Window*)sf->data;
  PetscBool         iascii;
  PetscViewerFormat format;

  PetscFunctionBegin;
  PetscCall(PetscViewerGetFormat(viewer,&format));
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii));
  if (iascii) {
    PetscCall(PetscViewerASCIIPrintf(viewer,"  current flavor=%s synchronization=%s MultiSF sort=%s\n",PetscSFWindowFlavorTypes[w->flavor],PetscSFWindowSyncTypes[w->sync],sf->rankorder ? "rank-order" : "unordered"));
    if (format == PETSC_VIEWER_ASCII_INFO_DETAIL) {
      if (w->info != MPI_INFO_NULL) {
        PetscMPIInt k,nkeys;
        char        key[MPI_MAX_INFO_KEY], value[MPI_MAX_INFO_VAL];

        PetscCallMPI(MPI_Info_get_nkeys(w->info,&nkeys));
        PetscCall(PetscViewerASCIIPrintf(viewer,"    current info with %d keys. Ordered key-value pairs follow:\n",nkeys));
        for (k = 0; k < nkeys; k++) {
          PetscMPIInt flag;

          PetscCallMPI(MPI_Info_get_nthkey(w->info,k,key));
          PetscCallMPI(MPI_Info_get(w->info,key,MPI_MAX_INFO_VAL,value,&flag));
          PetscCheck(flag,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Missing key %s",key);
          PetscCall(PetscViewerASCIIPrintf(viewer,"      %s = %s\n",key,value));
        }
      } else {
        PetscCall(PetscViewerASCIIPrintf(viewer,"    current info=MPI_INFO_NULL\n"));
      }
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSFDuplicate_Window(PetscSF sf,PetscSFDuplicateOption opt,PetscSF newsf)
{
  PetscSF_Window        *w = (PetscSF_Window*)sf->data;
  PetscSFWindowSyncType synctype;

  PetscFunctionBegin;
  synctype = w->sync;
  /* HACK: Must use FENCE or LOCK when called from PetscSFGetGroups() because ACTIVE here would cause recursion. */
  if (!sf->setupcalled) synctype = PETSCSF_WINDOW_SYNC_LOCK;
  PetscCall(PetscSFWindowSetSyncType(newsf,synctype));
  PetscCall(PetscSFWindowSetFlavorType(newsf,w->flavor));
  PetscCall(PetscSFWindowSetInfo(newsf,w->info));
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSFBcastBegin_Window(PetscSF sf,MPI_Datatype unit,PetscMemType rootmtype,const void *rootdata,PetscMemType leafmtype,void *leafdata,MPI_Op op)
{
  PetscSF_Window     *w = (PetscSF_Window*)sf->data;
  PetscInt           i,nranks;
  const PetscMPIInt  *ranks;
  const MPI_Aint     *target_disp;
  const MPI_Datatype *mine,*remote;
  MPI_Request        *reqs;
  MPI_Win            win;

  PetscFunctionBegin;
  PetscCheckFalse(op != MPI_REPLACE,PetscObjectComm((PetscObject)sf), PETSC_ERR_SUP, "PetscSFBcastBegin_Window with op!=MPI_REPLACE has not been implemented");
  PetscCall(PetscSFGetRootRanks(sf,&nranks,&ranks,NULL,NULL,NULL));
  PetscCall(PetscSFWindowGetDataTypes(sf,unit,&mine,&remote));
  PetscCall(PetscSFGetWindow(sf,unit,(void*)rootdata,w->sync,PETSC_TRUE,MPI_MODE_NOPUT|MPI_MODE_NOPRECEDE,MPI_MODE_NOPUT,0,&target_disp,&reqs,&win));
  for (i=0; i<nranks; i++) {
    MPI_Aint tdp = target_disp ? target_disp[i] : 0;

    if (w->sync == PETSCSF_WINDOW_SYNC_LOCK) {
      PetscCallMPI(MPI_Win_lock(MPI_LOCK_SHARED,ranks[i],MPI_MODE_NOCHECK,win));
#if defined(PETSC_HAVE_MPI_RGET)
      PetscCallMPI(MPI_Rget(leafdata,1,mine[i],ranks[i],tdp,1,remote[i],win,&reqs[i]));
#else
      PetscCallMPI(MPI_Get(leafdata,1,mine[i],ranks[i],tdp,1,remote[i],win));
#endif
    } else {
      PetscCallMPI(MPI_Get(leafdata,1,mine[i],ranks[i],tdp,1,remote[i],win));
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PetscSFBcastEnd_Window(PetscSF sf,MPI_Datatype unit,const void *rootdata,void *leafdata,MPI_Op op)
{
  PetscSF_Window *w = (PetscSF_Window*)sf->data;
  MPI_Win        win;
  MPI_Request    *reqs = NULL;

  PetscFunctionBegin;
  PetscCall(PetscSFFindWindow(sf,unit,rootdata,&win,&reqs));
  if (reqs) PetscCallMPI(MPI_Waitall(sf->nranks,reqs,MPI_STATUSES_IGNORE));
  if (w->sync == PETSCSF_WINDOW_SYNC_LOCK) {
    PetscInt           i,nranks;
    const PetscMPIInt  *ranks;

    PetscCall(PetscSFGetRootRanks(sf,&nranks,&ranks,NULL,NULL,NULL));
    for (i=0; i<nranks; i++) {
      PetscCallMPI(MPI_Win_unlock(ranks[i],win));
    }
  }
  PetscCall(PetscSFRestoreWindow(sf,unit,(void*)rootdata,w->sync,PETSC_TRUE,MPI_MODE_NOSTORE|MPI_MODE_NOSUCCEED,PETSC_FALSE,&win));
  PetscFunctionReturn(0);
}

PetscErrorCode PetscSFReduceBegin_Window(PetscSF sf,MPI_Datatype unit,PetscMemType leafmtype,const void *leafdata,PetscMemType rootmtype,void *rootdata,MPI_Op op)
{
  PetscSF_Window     *w = (PetscSF_Window*)sf->data;
  PetscInt           i,nranks;
  const PetscMPIInt  *ranks;
  const MPI_Aint     *target_disp;
  const MPI_Datatype *mine,*remote;
  MPI_Win            win;

  PetscFunctionBegin;
  PetscCall(PetscSFGetRootRanks(sf,&nranks,&ranks,NULL,NULL,NULL));
  PetscCall(PetscSFWindowGetDataTypes(sf,unit,&mine,&remote));
  PetscCall(PetscSFWindowOpTranslate(&op));
  PetscCall(PetscSFGetWindow(sf,unit,rootdata,w->sync,PETSC_TRUE,MPI_MODE_NOPRECEDE,0,0,&target_disp,NULL,&win));
  for (i=0; i<nranks; i++) {
    MPI_Aint tdp = target_disp ? target_disp[i] : 0;

    if (w->sync == PETSCSF_WINDOW_SYNC_LOCK) PetscCallMPI(MPI_Win_lock(MPI_LOCK_SHARED,ranks[i],MPI_MODE_NOCHECK,win));
    PetscCallMPI(MPI_Accumulate((void*)leafdata,1,mine[i],ranks[i],tdp,1,remote[i],op,win));
    if (w->sync == PETSCSF_WINDOW_SYNC_LOCK) PetscCallMPI(MPI_Win_unlock(ranks[i],win));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSFReduceEnd_Window(PetscSF sf,MPI_Datatype unit,const void *leafdata,void *rootdata,MPI_Op op)
{
  PetscSF_Window *w = (PetscSF_Window*)sf->data;
  MPI_Win        win;
  MPI_Request    *reqs = NULL;

  PetscFunctionBegin;
  PetscCall(PetscSFFindWindow(sf,unit,rootdata,&win,&reqs));
  if (reqs) PetscCallMPI(MPI_Waitall(sf->nranks,reqs,MPI_STATUSES_IGNORE));
  PetscCall(PetscSFRestoreWindow(sf,unit,rootdata,w->sync,PETSC_TRUE,MPI_MODE_NOSUCCEED,PETSC_TRUE,&win));
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSFFetchAndOpBegin_Window(PetscSF sf,MPI_Datatype unit,PetscMemType rootmtype,void *rootdata,PetscMemType leafmtype,const void *leafdata,void *leafupdate,MPI_Op op)
{
  PetscInt           i,nranks;
  const PetscMPIInt  *ranks;
  const MPI_Datatype *mine,*remote;
  const MPI_Aint     *target_disp;
  MPI_Win            win;
  PetscSF_Window     *w = (PetscSF_Window*)sf->data;
#if !defined(PETSC_HAVE_MPI_GET_ACCUMULATE)
  PetscSFWindowFlavorType oldf;
#endif

  PetscFunctionBegin;
  PetscCall(PetscSFGetRootRanks(sf,&nranks,&ranks,NULL,NULL,NULL));
  PetscCall(PetscSFWindowGetDataTypes(sf,unit,&mine,&remote));
  PetscCall(PetscSFWindowOpTranslate(&op));
#if !defined(PETSC_HAVE_MPI_GET_ACCUMULATE)
  /* FetchAndOp without MPI_Get_Accumulate requires locking.
     we create a new window every time to not interfere with user-defined MPI_Info which may have used "no_locks"="true" */
  oldf = w->flavor;
  w->flavor = PETSCSF_WINDOW_FLAVOR_CREATE;
  PetscCall(PetscSFGetWindow(sf,unit,rootdata,PETSCSF_WINDOW_SYNC_LOCK,PETSC_FALSE,0,0,0,&target_disp,NULL,&win));
#else
  PetscCall(PetscSFGetWindow(sf,unit,rootdata,w->sync,PETSC_TRUE,MPI_MODE_NOPRECEDE,0,0,&target_disp,NULL,&win));
#endif
  for (i=0; i<nranks; i++) {
    MPI_Aint tdp = target_disp ? target_disp[i] : 0;

#if !defined(PETSC_HAVE_MPI_GET_ACCUMULATE)
    PetscCallMPI(MPI_Win_lock(MPI_LOCK_EXCLUSIVE,ranks[i],0,win));
    PetscCallMPI(MPI_Get(leafupdate,1,mine[i],ranks[i],tdp,1,remote[i],win));
    PetscCallMPI(MPI_Accumulate((void*)leafdata,1,mine[i],ranks[i],tdp,1,remote[i],op,win));
    PetscCallMPI(MPI_Win_unlock(ranks[i],win));
#else
    if (w->sync == PETSCSF_WINDOW_SYNC_LOCK) PetscCallMPI(MPI_Win_lock(MPI_LOCK_SHARED,ranks[i],0,win));
    PetscCallMPI(MPI_Get_accumulate((void*)leafdata,1,mine[i],leafupdate,1,mine[i],ranks[i],tdp,1,remote[i],op,win));
    if (w->sync == PETSCSF_WINDOW_SYNC_LOCK) PetscCallMPI(MPI_Win_unlock(ranks[i],win));
#endif
  }
#if !defined(PETSC_HAVE_MPI_GET_ACCUMULATE)
  w->flavor = oldf;
#endif
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSFFetchAndOpEnd_Window(PetscSF sf,MPI_Datatype unit,void *rootdata,const void *leafdata,void *leafupdate,MPI_Op op)
{
  MPI_Win        win;
#if defined(PETSC_HAVE_MPI_GET_ACCUMULATE)
  PetscSF_Window *w = (PetscSF_Window*)sf->data;
#endif
  MPI_Request    *reqs = NULL;

  PetscFunctionBegin;
  PetscCall(PetscSFFindWindow(sf,unit,rootdata,&win,&reqs));
  if (reqs) PetscCallMPI(MPI_Waitall(sf->nranks,reqs,MPI_STATUSES_IGNORE));
#if defined(PETSC_HAVE_MPI_GET_ACCUMULATE)
  PetscCall(PetscSFRestoreWindow(sf,unit,rootdata,w->sync,PETSC_TRUE,MPI_MODE_NOSUCCEED,PETSC_TRUE,&win));
#else
  PetscCall(PetscSFRestoreWindow(sf,unit,rootdata,PETSCSF_WINDOW_SYNC_LOCK,PETSC_FALSE,0,PETSC_TRUE,&win));
#endif
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode PetscSFCreate_Window(PetscSF sf)
{
  PetscSF_Window *w = (PetscSF_Window*)sf->data;

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

  PetscCall(PetscNewLog(sf,&w));
  sf->data  = (void*)w;
  w->sync   = PETSCSF_WINDOW_SYNC_FENCE;
  w->flavor = PETSCSF_WINDOW_FLAVOR_CREATE;
  w->info   = MPI_INFO_NULL;

  PetscCall(PetscObjectComposeFunction((PetscObject)sf,"PetscSFWindowSetSyncType_C",PetscSFWindowSetSyncType_Window));
  PetscCall(PetscObjectComposeFunction((PetscObject)sf,"PetscSFWindowGetSyncType_C",PetscSFWindowGetSyncType_Window));
  PetscCall(PetscObjectComposeFunction((PetscObject)sf,"PetscSFWindowSetFlavorType_C",PetscSFWindowSetFlavorType_Window));
  PetscCall(PetscObjectComposeFunction((PetscObject)sf,"PetscSFWindowGetFlavorType_C",PetscSFWindowGetFlavorType_Window));
  PetscCall(PetscObjectComposeFunction((PetscObject)sf,"PetscSFWindowSetInfo_C",PetscSFWindowSetInfo_Window));
  PetscCall(PetscObjectComposeFunction((PetscObject)sf,"PetscSFWindowGetInfo_C",PetscSFWindowGetInfo_Window));

#if defined(OMPI_MAJOR_VERSION) && (OMPI_MAJOR_VERSION < 1 || (OMPI_MAJOR_VERSION == 1 && OMPI_MINOR_VERSION <= 6))
  {
    PetscBool ackbug = PETSC_FALSE;
    PetscCall(PetscOptionsGetBool(NULL,NULL,"-acknowledge_ompi_onesided_bug",&ackbug,NULL));
    if (ackbug) {
      PetscCall(PetscInfo(sf,"Acknowledged Open MPI bug, proceeding anyway. Expect memory corruption.\n"));
    } else SETERRQ(PetscObjectComm((PetscObject)sf),PETSC_ERR_LIB,"Open MPI is known to be buggy (https://svn.open-mpi.org/trac/ompi/ticket/1905 and 2656), use -acknowledge_ompi_onesided_bug to proceed");
  }
#endif
  PetscFunctionReturn(0);
}
