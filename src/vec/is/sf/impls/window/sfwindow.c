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
  PetscSFWindowFlavorType flavor;
  MPI_Aint                *dyn_target_addr;
  PetscBool               epoch;
  PetscSFWinLink          next;
};

const char *const PetscSFWindowSyncTypes[] = {"FENCE","LOCK","ACTIVE","PetscSFWindowSyncType","PETSCSF_WINDOW_SYNC_",0};
const char *const PetscSFWindowFlavorTypes[] = {"CREATE","DYNAMIC","ALLOCATE","SHARED","PetscSFWindowFlavorType","PETSCSF_WINDOW_FLAVOR_",0};

/* Built-in MPI_Ops act elementwise inside MPI_Accumulate, but cannot be used with composite types inside collectives (MPIU_Allreduce) */
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

   Input Arguments:
+  sf - star forest
-  unit - data type for each node

   Output Arguments:
+  localtypes - types describing part of local leaf buffer referencing each remote rank
-  remotetypes - types describing part of remote root buffer referenced for each remote rank

   Level: developer

.seealso: PetscSFSetGraph(), PetscSFView()
@*/
static PetscErrorCode PetscSFWindowGetDataTypes(PetscSF sf,MPI_Datatype unit,const MPI_Datatype **localtypes,const MPI_Datatype **remotetypes)
{
  PetscSF_Window    *w = (PetscSF_Window*)sf->data;
  PetscErrorCode    ierr;
  PetscSFDataLink   link;
  PetscInt          i,nranks;
  const PetscInt    *roffset,*rmine,*rremote;
  const PetscMPIInt *ranks;

  PetscFunctionBegin;
  /* Look for types in cache */
  for (link=w->link; link; link=link->next) {
    PetscBool match;
    ierr = MPIPetsc_Type_compare(unit,link->unit,&match);CHKERRQ(ierr);
    if (match) {
      *localtypes  = link->mine;
      *remotetypes = link->remote;
      PetscFunctionReturn(0);
    }
  }

  /* Create new composite types for each send rank */
  ierr = PetscSFGetRootRanks(sf,&nranks,&ranks,&roffset,&rmine,&rremote);CHKERRQ(ierr);
  ierr = PetscNew(&link);CHKERRQ(ierr);
  ierr = MPI_Type_dup(unit,&link->unit);CHKERRQ(ierr);
  ierr = PetscMalloc2(nranks,&link->mine,nranks,&link->remote);CHKERRQ(ierr);
  for (i=0; i<nranks; i++) {
    PetscInt    rcount = roffset[i+1] - roffset[i];
    PetscMPIInt *rmine,*rremote;
#if !defined(PETSC_USE_64BIT_INDICES)
    rmine   = sf->rmine + sf->roffset[i];
    rremote = sf->rremote + sf->roffset[i];
#else
    PetscInt j;
    ierr = PetscMalloc2(rcount,&rmine,rcount,&rremote);CHKERRQ(ierr);
    for (j=0; j<rcount; j++) {
      ierr = PetscMPIIntCast(sf->rmine[sf->roffset[i]+j],rmine+j);CHKERRQ(ierr);
      ierr = PetscMPIIntCast(sf->rremote[sf->roffset[i]+j],rremote+j);CHKERRQ(ierr);
    }
#endif

    ierr = MPI_Type_create_indexed_block(rcount,1,rmine,link->unit,&link->mine[i]);CHKERRQ(ierr);
    ierr = MPI_Type_create_indexed_block(rcount,1,rremote,link->unit,&link->remote[i]);CHKERRQ(ierr);
#if defined(PETSC_USE_64BIT_INDICES)
    ierr = PetscFree2(rmine,rremote);CHKERRQ(ierr);
#endif
    ierr = MPI_Type_commit(&link->mine[i]);CHKERRQ(ierr);
    ierr = MPI_Type_commit(&link->remote[i]);CHKERRQ(ierr);
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

   Input Arguments:
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sf,PETSCSF_CLASSID,1);
  PetscValidLogicalCollectiveEnum(sf,flavor,2);
  ierr = PetscTryMethod(sf,"PetscSFWindowSetFlavorType_C",(PetscSF,PetscSFWindowFlavorType),(sf,flavor));CHKERRQ(ierr);
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

   Input Argument:
.  sf - star forest for communication

   Output Argument:
.  flavor - flavor type

   Level: advanced

.seealso: PetscSFSetFromOptions(), PetscSFWindowSetFlavorType()
@*/
PetscErrorCode PetscSFWindowGetFlavorType(PetscSF sf,PetscSFWindowFlavorType *flavor)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sf,PETSCSF_CLASSID,1);
  PetscValidPointer(flavor,2);
  ierr = PetscUseMethod(sf,"PetscSFWindowGetFlavorType_C",(PetscSF,PetscSFWindowFlavorType*),(sf,flavor));CHKERRQ(ierr);
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

   Input Arguments:
+  sf - star forest for communication
-  sync - synchronization type

   Options Database Key:
.  -sf_window_sync <sync> - sets the synchronization type FENCE, LOCK, or ACTIVE (see PetscSFWindowSyncType)

   Level: advanced

.seealso: PetscSFSetFromOptions(), PetscSFWindowGetSyncType()
@*/
PetscErrorCode PetscSFWindowSetSyncType(PetscSF sf,PetscSFWindowSyncType sync)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sf,PETSCSF_CLASSID,1);
  PetscValidLogicalCollectiveEnum(sf,sync,2);
  ierr = PetscTryMethod(sf,"PetscSFWindowSetSyncType_C",(PetscSF,PetscSFWindowSyncType),(sf,sync));CHKERRQ(ierr);
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

   Input Argument:
.  sf - star forest for communication

   Output Argument:
.  sync - synchronization type

   Level: advanced

.seealso: PetscSFSetFromOptions(), PetscSFWindowSetSyncType()
@*/
PetscErrorCode PetscSFWindowGetSyncType(PetscSF sf,PetscSFWindowSyncType *sync)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sf,PETSCSF_CLASSID,1);
  PetscValidPointer(sync,2);
  ierr = PetscUseMethod(sf,"PetscSFWindowGetSyncType_C",(PetscSF,PetscSFWindowSyncType*),(sf,sync));CHKERRQ(ierr);
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

   Input Argument:
+  sf - star forest for communication
-  info - MPI_Info handle

   Level: advanced

   Notes: the info handle is duplicated with a call to MPI_Info_dup unless info = MPI_INFO_NULL.

.seealso: PetscSFSetFromOptions(), PetscSFWindowGetInfo()
@*/
PetscErrorCode PetscSFWindowSetInfo(PetscSF sf,MPI_Info info)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sf,PETSCSF_CLASSID,1);
  ierr = PetscTryMethod(sf,"PetscSFWindowSetInfo_C",(PetscSF,MPI_Info),(sf,info));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSFWindowSetInfo_Window(PetscSF sf,MPI_Info info)
{
  PetscSF_Window *w = (PetscSF_Window*)sf->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (w->info != MPI_INFO_NULL) {
    ierr = MPI_Info_free(&w->info);CHKERRQ(ierr);
  }
  if (info != MPI_INFO_NULL) {
    ierr = MPI_Info_dup(info,&w->info);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*@C
   PetscSFWindowGetInfo - Get the MPI_Info handle used for windows allocation

   Logically Collective

   Input Argument:
.  sf - star forest for communication

   Output Argument:
.  info - MPI_Info handle

   Level: advanced

   Notes: if PetscSFWindowSetInfo() has not be called, this returns MPI_INFO_NULL

.seealso: PetscSFSetFromOptions(), PetscSFWindowSetInfo()
@*/
PetscErrorCode PetscSFWindowGetInfo(PetscSF sf,MPI_Info *info)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sf,PETSCSF_CLASSID,1);
  PetscValidPointer(info,2);
  ierr = PetscUseMethod(sf,"PetscSFWindowGetInfo_C",(PetscSF,MPI_Info*),(sf,info));CHKERRQ(ierr);
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

   Input Arguments:
+  sf - star forest
.  unit - data type
.  array - array to be sent
.  sync - type of synchronization PetscSFWindowSyncType
.  epoch - PETSC_TRUE to acquire the window and start an epoch, PETSC_FALSE to just acquire the window
.  fenceassert - assert parameter for call to MPI_Win_fence(), if sync == PETSCSF_WINDOW_SYNC_FENCE
.  postassert - assert parameter for call to MPI_Win_post(), if sync == PETSCSF_WINDOW_SYNC_ACTIVE
.  startassert - assert parameter for call to MPI_Win_start(), if sync == PETSCSF_WINDOW_SYNC_ACTIVE
-  target_disp - target_disp argument to RMA calls (significative for PETSCSF_WINDOW_FLAVOR_DYNAMIC flavor only)

   Output Arguments:
.  win - window

   Level: developer
.seealso: PetscSFGetRootRanks(), PetscSFWindowGetDataTypes()
*/
static PetscErrorCode PetscSFGetWindow(PetscSF sf,MPI_Datatype unit,void *array,PetscSFWindowSyncType sync,PetscBool epoch,PetscMPIInt fenceassert,PetscMPIInt postassert,PetscMPIInt startassert,const MPI_Aint **target_disp, MPI_Win *win)
{
  PetscSF_Window *w = (PetscSF_Window*)sf->data;
  PetscErrorCode ierr;
  MPI_Aint       lb,lb_true,bytes,bytes_true;
  PetscSFWinLink link;
  MPI_Aint       winaddr;
  PetscInt       nranks;
  PetscBool      reuse = PETSC_FALSE, update = PETSC_FALSE;
#if defined(PETSC_USE_DEBUG)
  PetscBool      dummy[2];
#endif
  MPI_Aint       wsize;

  PetscFunctionBegin;
  ierr = MPI_Type_get_extent(unit,&lb,&bytes);CHKERRQ(ierr);
  ierr = MPI_Type_get_true_extent(unit,&lb_true,&bytes_true);CHKERRQ(ierr);
  if (lb != 0 || lb_true != 0) SETERRQ(PetscObjectComm((PetscObject)sf),PETSC_ERR_SUP,"No support for unit type with nonzero lower bound, write petsc-maint@mcs.anl.gov if you want this feature");
  if (bytes != bytes_true) SETERRQ(PetscObjectComm((PetscObject)sf),PETSC_ERR_SUP,"No support for unit type with modified extent, write petsc-maint@mcs.anl.gov if you want this feature");
  if (w->flavor != PETSCSF_WINDOW_FLAVOR_CREATE) reuse = PETSC_TRUE;
  for (link=w->wins; reuse && link; link=link->next) {
    PetscBool winok = PETSC_FALSE;
    if (w->flavor != link->flavor) continue;
    switch (w->flavor) {
    case PETSCSF_WINDOW_FLAVOR_DYNAMIC: /* check available matching array, error if in use (we additionally check that the matching condition is the same across processes) */
      if (array == link->addr) {
#if defined(PETSC_USE_DEBUG)
        dummy[0] = PETSC_TRUE;
        dummy[1] = PETSC_TRUE;
        ierr = MPI_Allreduce(MPI_IN_PLACE,dummy  ,1,MPIU_BOOL,MPI_LAND,PetscObjectComm((PetscObject)sf));CHKERRQ(ierr);
        ierr = MPI_Allreduce(MPI_IN_PLACE,dummy+1,1,MPIU_BOOL,MPI_LOR ,PetscObjectComm((PetscObject)sf));CHKERRQ(ierr);
        if (dummy[0] != dummy[1]) SETERRQ(PetscObjectComm((PetscObject)sf),PETSC_ERR_SUP,"PETSCSF_WINDOW_FLAVOR_DYNAMIC requires root pointers to be consistently used across the comm. Use PETSCSF_WINDOW_FLAVOR_CREATE or PETSCSF_WINDOW_FLAVOR_ALLOCATE instead");
#endif
        if (link->inuse) SETERRQ(PetscObjectComm((PetscObject)sf),PETSC_ERR_PLIB,"Window in use");
        if (epoch && link->epoch) SETERRQ(PetscObjectComm((PetscObject)sf),PETSC_ERR_PLIB,"Window epoch not finished");
        winok = PETSC_TRUE;
        link->paddr = array;
#if defined(PETSC_USE_DEBUG)
      } else {
        dummy[0] = PETSC_FALSE;
        dummy[1] = PETSC_FALSE;
        ierr = MPI_Allreduce(MPI_IN_PLACE,dummy  ,1,MPIU_BOOL,MPI_LAND,PetscObjectComm((PetscObject)sf));CHKERRQ(ierr);
        ierr = MPI_Allreduce(MPI_IN_PLACE,dummy+1,1,MPIU_BOOL,MPI_LOR ,PetscObjectComm((PetscObject)sf));CHKERRQ(ierr);
        if (dummy[0] != dummy[1]) SETERRQ(PetscObjectComm((PetscObject)sf),PETSC_ERR_SUP,"PETSCSF_WINDOW_FLAVOR_DYNAMIC requires root pointers to be consistently used across the comm. Use PETSCSF_WINDOW_FLAVOR_CREATE or PETSCSF_WINDOW_FLAVOR_ALLOCATE instead");
#endif
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
    default: SETERRQ1(PetscObjectComm((PetscObject)sf),PETSC_ERR_SUP,"No support for flavor %s",PetscSFWindowFlavorTypes[w->flavor]);
    }
    if (winok) {
      *win = link->win;
      ierr = PetscInfo3(sf,"Reusing window %d of flavor %d for comm %d\n",link->win,link->flavor,PetscObjectComm((PetscObject)sf));CHKERRQ(ierr);
      goto found;
    }
  }

  wsize = (MPI_Aint)bytes*sf->nroots;
  ierr = PetscNew(&link);CHKERRQ(ierr);
  link->bytes           = bytes;
  link->next            = w->wins;
  link->flavor          = w->flavor;
  link->dyn_target_addr = NULL;
  w->wins               = link;
  switch (w->flavor) {
  case PETSCSF_WINDOW_FLAVOR_CREATE:
    ierr = MPI_Win_create(array,wsize,(PetscMPIInt)bytes,w->info,PetscObjectComm((PetscObject)sf),&link->win);CHKERRQ(ierr);
    link->addr  = array;
    link->paddr = array;
    break;
  case PETSCSF_WINDOW_FLAVOR_DYNAMIC:
    ierr = MPI_Win_create_dynamic(w->info,PetscObjectComm((PetscObject)sf),&link->win);CHKERRQ(ierr);
#if defined(PETSC_HAVE_OMPI_MAJOR_VERSION) /* some OpenMPI versions do not support MPI_Win_attach(win,NULL,0); */
    ierr = MPI_Win_attach(link->win,wsize ? array : &ierr,wsize);CHKERRQ(ierr);
#else
    ierr = MPI_Win_attach(link->win,array,wsize);CHKERRQ(ierr);
#endif
    link->addr  = array;
    link->paddr = array;
    if (!w->dynsf) SETERRQ(PetscObjectComm((PetscObject)sf),PETSC_ERR_ORDER,"Must call PetscSFSetUp()");
    ierr = PetscSFSetUp(w->dynsf);CHKERRQ(ierr);
    ierr = PetscSFGetRootRanks(w->dynsf,&nranks,NULL,NULL,NULL,NULL);CHKERRQ(ierr);
    ierr = PetscMalloc1(nranks,&link->dyn_target_addr);CHKERRQ(ierr);
    ierr = MPI_Get_address(array,&winaddr);CHKERRQ(ierr);
    ierr = PetscSFBcastBegin(w->dynsf,MPI_AINT,&winaddr,link->dyn_target_addr);CHKERRQ(ierr);
    ierr = PetscSFBcastEnd(w->dynsf,MPI_AINT,&winaddr,link->dyn_target_addr);CHKERRQ(ierr);
    break;
  case PETSCSF_WINDOW_FLAVOR_ALLOCATE:
    ierr = MPI_Win_allocate(wsize,(PetscMPIInt)bytes,w->info,PetscObjectComm((PetscObject)sf),&link->addr,&link->win);CHKERRQ(ierr);
    update = PETSC_TRUE;
    link->paddr = array;
    break;
#if defined(PETSC_HAVE_MPI_PROCESS_SHARED_MEMORY)
  case PETSCSF_WINDOW_FLAVOR_SHARED:
    ierr = MPI_Win_allocate_shared(wsize,(PetscMPIInt)bytes,w->info,PetscObjectComm((PetscObject)sf),&link->addr,&link->win);CHKERRQ(ierr);
    update = PETSC_TRUE;
    link->paddr = array;
    break;
#endif
  default: SETERRQ1(PetscObjectComm((PetscObject)sf),PETSC_ERR_SUP,"No support for flavor %s",PetscSFWindowFlavorTypes[w->flavor]);
  }
  ierr = PetscInfo3(sf,"New window %d of flavor %d for comm %d\n",link->win,link->flavor,PetscObjectComm((PetscObject)sf));CHKERRQ(ierr);
  *win = link->win;

found:

  if (update) {
    ierr = PetscMemcpy(link->addr,array,sf->nroots*bytes);CHKERRQ(ierr);
    if (sync == PETSCSF_WINDOW_SYNC_LOCK) {
      ierr = MPI_Win_fence(0,*win);CHKERRQ(ierr);
    }
  }
  link->inuse = PETSC_TRUE;
  link->epoch = epoch;
  *target_disp = link->dyn_target_addr;
  if (epoch) {
    switch (sync) {
    case PETSCSF_WINDOW_SYNC_FENCE:
      ierr = MPI_Win_fence(fenceassert,*win);CHKERRQ(ierr);
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
      ierr = PetscSFGetGroups(sf,&ingroup,&outgroup);CHKERRQ(ierr);
      ierr = MPI_Group_size(ingroup,&isize);CHKERRQ(ierr);
      ierr = MPI_Group_size(outgroup,&osize);CHKERRQ(ierr);
      if (isize) { ierr = MPI_Win_post(ingroup,postassert,*win);CHKERRQ(ierr); }
      if (osize) { ierr = MPI_Win_start(outgroup,startassert,*win);CHKERRQ(ierr); }
    } break;
    default: SETERRQ(PetscObjectComm((PetscObject)sf),PETSC_ERR_PLIB,"Unknown synchronization type");
    }
  }
  PetscFunctionReturn(0);
}

/*
   PetscSFFindWindow - Finds a window that is already in use

   Not Collective

   Input Arguments:
+  sf - star forest
.  unit - data type
-  array - array with which the window is associated

   Output Arguments:
.  win - window

   Level: developer

.seealso: PetscSFGetWindow(), PetscSFRestoreWindow()
*/
static PetscErrorCode PetscSFFindWindow(PetscSF sf,MPI_Datatype unit,const void *array,MPI_Win *win)
{
  PetscSF_Window *w = (PetscSF_Window*)sf->data;
  PetscSFWinLink link;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  *win = MPI_WIN_NULL;
  for (link=w->wins; link; link=link->next) {
    if (array == link->paddr) {
      ierr = PetscInfo3(sf,"Window %d of flavor %d for comm %d\n",link->win,link->flavor,PetscObjectComm((PetscObject)sf));CHKERRQ(ierr);
      *win = link->win;
      PetscFunctionReturn(0);
    }
  }
  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Requested window not in use");
  PetscFunctionReturn(0);
}

/*
   PetscSFRestoreWindow - Restores a window obtained with PetscSFGetWindow()

   Collective

   Input Arguments:
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
  PetscErrorCode          ierr;
  PetscSFWinLink          *p,link;
  PetscBool               reuse = PETSC_FALSE;
  PetscSFWindowFlavorType flavor;
  void*                   laddr;
  size_t                  bytes;

  PetscFunctionBegin;
  for (p=&w->wins; *p; p=&(*p)->next) {
    link = *p;
    if (*win == link->win) {
      if (array != link->paddr) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Matched window, but not array");
      if (epoch != link->epoch) {
        if (epoch) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"No epoch to end");
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
  ierr = PetscInfo3(sf,"Window %d of flavor %d for comm %d\n",link->win,link->flavor,PetscObjectComm((PetscObject)sf));CHKERRQ(ierr);
  if (epoch) {
    switch (sync) {
    case PETSCSF_WINDOW_SYNC_FENCE:
      ierr = MPI_Win_fence(fenceassert,*win);CHKERRQ(ierr);
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
      ierr = PetscSFGetGroups(sf,&ingroup,&outgroup);CHKERRQ(ierr);
      ierr = MPI_Group_size(ingroup,&isize);CHKERRQ(ierr);
      ierr = MPI_Group_size(outgroup,&osize);CHKERRQ(ierr);
      if (osize) { ierr = MPI_Win_complete(*win);CHKERRQ(ierr); }
      if (isize) { ierr = MPI_Win_wait(*win);CHKERRQ(ierr); }
    } break;
    default: SETERRQ(PetscObjectComm((PetscObject)sf),PETSC_ERR_PLIB,"Unknown synchronization type");
    }
  }
  if (update) {
    if (sync == PETSCSF_WINDOW_SYNC_LOCK) {
      ierr = MPI_Win_fence(MPI_MODE_NOPUT|MPI_MODE_NOSUCCEED,*win);CHKERRQ(ierr);
    }
    ierr = PetscMemcpy(array,laddr,sf->nroots*bytes);CHKERRQ(ierr);
  }
  link->epoch = PETSC_FALSE;
  link->inuse = PETSC_FALSE;
  link->paddr = NULL;
  if (!reuse) {
    ierr = MPI_Win_free(&link->win);CHKERRQ(ierr);
    ierr = PetscFree(link);CHKERRQ(ierr);
    *win = MPI_WIN_NULL;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSFSetUp_Window(PetscSF sf)
{
  PetscSF_Window *w = (PetscSF_Window*)sf->data;
  PetscErrorCode ierr;
  MPI_Group      ingroup,outgroup;

  PetscFunctionBegin;
  ierr = PetscSFSetUpRanks(sf,MPI_GROUP_EMPTY);CHKERRQ(ierr);
  if (!w->dynsf) {
    PetscInt    i;
    PetscSFNode *remotes;

    ierr = PetscMalloc1(sf->nranks,&remotes);CHKERRQ(ierr);
    for (i=0;i<sf->nranks;i++) {
      remotes[i].rank  = sf->ranks[i];
      remotes[i].index = 0;
    }
    ierr = PetscSFDuplicate(sf,PETSCSF_DUPLICATE_RANKS,&w->dynsf);CHKERRQ(ierr);
    ierr = PetscSFWindowSetFlavorType(w->dynsf,PETSCSF_WINDOW_FLAVOR_CREATE);CHKERRQ(ierr); /* break recursion */
    ierr = PetscSFSetGraph(w->dynsf,1,sf->nranks,NULL,PETSC_OWN_POINTER,remotes,PETSC_OWN_POINTER);CHKERRQ(ierr);
    ierr = PetscLogObjectParent((PetscObject)sf,(PetscObject)w->dynsf);CHKERRQ(ierr);
  }
  switch (w->sync) {
  case PETSCSF_WINDOW_SYNC_ACTIVE:
    ierr = PetscSFGetGroups(sf,&ingroup,&outgroup);CHKERRQ(ierr);
  default:
    break;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSFSetFromOptions_Window(PetscOptionItems *PetscOptionsObject,PetscSF sf)
{
  PetscSF_Window          *w = (PetscSF_Window*)sf->data;
  PetscErrorCode          ierr;
  PetscSFWindowFlavorType flavor = w->flavor;

  PetscFunctionBegin;
  ierr = PetscOptionsHead(PetscOptionsObject,"PetscSF Window options");CHKERRQ(ierr);
  ierr = PetscOptionsEnum("-sf_window_sync","synchronization type to use for PetscSF Window communication","PetscSFWindowSetSyncType",PetscSFWindowSyncTypes,(PetscEnum)w->sync,(PetscEnum*)&w->sync,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnum("-sf_window_flavor","flavor to use for PetscSF Window creation","PetscSFWindowSetFlavorType",PetscSFWindowFlavorTypes,(PetscEnum)flavor,(PetscEnum*)&flavor,NULL);CHKERRQ(ierr);
  ierr = PetscSFWindowSetFlavorType(sf,flavor);CHKERRQ(ierr);
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSFReset_Window(PetscSF sf)
{
  PetscSF_Window  *w = (PetscSF_Window*)sf->data;
  PetscErrorCode  ierr;
  PetscSFDataLink link,next;
  PetscSFWinLink  wlink,wnext;
  PetscInt        i;

  PetscFunctionBegin;
  for (link=w->link; link; link=next) {
    next = link->next;
    ierr = MPI_Type_free(&link->unit);CHKERRQ(ierr);
    for (i=0; i<sf->nranks; i++) {
      ierr = MPI_Type_free(&link->mine[i]);CHKERRQ(ierr);
      ierr = MPI_Type_free(&link->remote[i]);CHKERRQ(ierr);
    }
    ierr = PetscFree2(link->mine,link->remote);CHKERRQ(ierr);
    ierr = PetscFree(link);CHKERRQ(ierr);
  }
  w->link = NULL;
  for (wlink=w->wins; wlink; wlink=wnext) {
    wnext = wlink->next;
    if (wlink->inuse) SETERRQ1(PetscObjectComm((PetscObject)sf),PETSC_ERR_ARG_WRONGSTATE,"Window still in use with address %p",(void*)wlink->addr);
    ierr = PetscFree(wlink->dyn_target_addr);CHKERRQ(ierr);
    ierr = MPI_Win_free(&wlink->win);CHKERRQ(ierr);
    ierr = PetscFree(wlink);CHKERRQ(ierr);
  }
  w->wins = NULL;
  ierr = PetscSFDestroy(&w->dynsf);CHKERRQ(ierr);
  if (w->info != MPI_INFO_NULL) {
    ierr = MPI_Info_free(&w->info);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSFDestroy_Window(PetscSF sf)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscSFReset_Window(sf);CHKERRQ(ierr);
  ierr = PetscFree(sf->data);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)sf,"PetscSFWindowSetSyncType_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)sf,"PetscSFWindowGetSyncType_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)sf,"PetscSFWindowSetFlavorType_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)sf,"PetscSFWindowGetFlavorType_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)sf,"PetscSFWindowSetInfo_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)sf,"PetscSFWindowGetInfo_C",NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSFView_Window(PetscSF sf,PetscViewer viewer)
{
  PetscSF_Window    *w = (PetscSF_Window*)sf->data;
  PetscErrorCode    ierr;
  PetscBool         iascii;
  PetscViewerFormat format;

  PetscFunctionBegin;
  ierr = PetscViewerGetFormat(viewer,&format);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    ierr = PetscViewerASCIIPrintf(viewer,"  current flavor=%s synchronization=%s sort=%s\n",PetscSFWindowFlavorTypes[w->flavor],PetscSFWindowSyncTypes[w->sync],sf->rankorder ? "rank-order" : "unordered");CHKERRQ(ierr);
    if (format == PETSC_VIEWER_ASCII_INFO_DETAIL) {
      if (w->info != MPI_INFO_NULL) {
        PetscMPIInt k,nkeys;
        char        key[MPI_MAX_INFO_KEY], value[MPI_MAX_INFO_VAL];

        ierr = MPI_Info_get_nkeys(w->info,&nkeys);CHKERRQ(ierr);
        ierr = PetscViewerASCIIPrintf(viewer,"    current info with %d keys. Ordered key-value pairs follow:\n",nkeys);CHKERRQ(ierr);
        for (k = 0; k < nkeys; k++) {
          PetscMPIInt flag;

          ierr = MPI_Info_get_nthkey(w->info,k,key);CHKERRQ(ierr);
          ierr = MPI_Info_get(w->info,key,MPI_MAX_INFO_VAL,value,&flag);CHKERRQ(ierr);
          if (!flag) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Missing key %s",key);
          ierr = PetscViewerASCIIPrintf(viewer,"      %s = %s\n",key,value);CHKERRQ(ierr);
        }
      } else {
        ierr = PetscViewerASCIIPrintf(viewer,"    current info=MPI_INFO_NULL\n");CHKERRQ(ierr);
      }
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSFDuplicate_Window(PetscSF sf,PetscSFDuplicateOption opt,PetscSF newsf)
{
  PetscSF_Window        *w = (PetscSF_Window*)sf->data;
  PetscErrorCode        ierr;
  PetscSFWindowSyncType synctype;

  PetscFunctionBegin;
  synctype = w->sync;
  /* HACK: Must use FENCE or LOCK when called from PetscSFGetGroups() because ACTIVE here would cause recursion. */
  if (!sf->setupcalled) synctype = PETSCSF_WINDOW_SYNC_LOCK;
  ierr = PetscSFWindowSetSyncType(newsf,synctype);CHKERRQ(ierr);
  ierr = PetscSFWindowSetFlavorType(newsf,w->flavor);CHKERRQ(ierr);
  ierr = PetscSFWindowSetInfo(newsf,w->info);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSFBcastAndOpBegin_Window(PetscSF sf,MPI_Datatype unit,PetscMemType rootmtype,const void *rootdata,PetscMemType leafmtype,void *leafdata,MPI_Op op)
{
  PetscSF_Window     *w = (PetscSF_Window*)sf->data;
  PetscErrorCode     ierr;
  PetscInt           i,nranks;
  const PetscMPIInt  *ranks;
  const MPI_Aint     *target_disp;
  const MPI_Datatype *mine,*remote;
  MPI_Win            win;

  PetscFunctionBegin;
  if (op != MPI_REPLACE || op != MPIU_REPLACE) SETERRQ(PetscObjectComm((PetscObject)sf), PETSC_ERR_SUP, "PetscSFBcastAndOpBegin_Window with op!=MPI_REPLACE has not been implemented");
  ierr = PetscSFGetRootRanks(sf,&nranks,&ranks,NULL,NULL,NULL);CHKERRQ(ierr);
  ierr = PetscSFWindowGetDataTypes(sf,unit,&mine,&remote);CHKERRQ(ierr);
  ierr = PetscSFGetWindow(sf,unit,(void*)rootdata,w->sync,PETSC_TRUE,MPI_MODE_NOPUT|MPI_MODE_NOPRECEDE,MPI_MODE_NOPUT,0,&target_disp,&win);CHKERRQ(ierr);
  for (i=0; i<nranks; i++) {
    MPI_Aint tdp = target_disp ? target_disp[i] : 0;

    if (w->sync == PETSCSF_WINDOW_SYNC_LOCK) {ierr = MPI_Win_lock(MPI_LOCK_SHARED,ranks[i],MPI_MODE_NOCHECK,win);CHKERRQ(ierr);}
    ierr = MPI_Get(leafdata,1,mine[i],ranks[i],tdp,1,remote[i],win);CHKERRQ(ierr);
    if (w->sync == PETSCSF_WINDOW_SYNC_LOCK) {ierr = MPI_Win_unlock(ranks[i],win);CHKERRQ(ierr);}
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PetscSFBcastAndOpEnd_Window(PetscSF sf,MPI_Datatype unit,PetscMemType rootmtype,const void *rootdata,PetscMemType leafmtype,void *leafdata,MPI_Op op)
{
  PetscSF_Window *w = (PetscSF_Window*)sf->data;
  PetscErrorCode ierr;
  MPI_Win        win;

  PetscFunctionBegin;
  ierr = PetscSFFindWindow(sf,unit,rootdata,&win);CHKERRQ(ierr);
  ierr = PetscSFRestoreWindow(sf,unit,(void*)rootdata,w->sync,PETSC_TRUE,MPI_MODE_NOSTORE|MPI_MODE_NOSUCCEED,PETSC_FALSE,&win);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscSFReduceBegin_Window(PetscSF sf,MPI_Datatype unit,PetscMemType leafmtype,const void *leafdata,PetscMemType rootmtype,void *rootdata,MPI_Op op)
{
  PetscSF_Window     *w = (PetscSF_Window*)sf->data;
  PetscErrorCode     ierr;
  PetscInt           i,nranks;
  const PetscMPIInt  *ranks;
  const MPI_Aint     *target_disp;
  const MPI_Datatype *mine,*remote;
  MPI_Win            win;

  PetscFunctionBegin;
  ierr = PetscSFGetRootRanks(sf,&nranks,&ranks,NULL,NULL,NULL);CHKERRQ(ierr);
  ierr = PetscSFWindowGetDataTypes(sf,unit,&mine,&remote);CHKERRQ(ierr);
  ierr = PetscSFWindowOpTranslate(&op);CHKERRQ(ierr);
  ierr = PetscSFGetWindow(sf,unit,rootdata,w->sync,PETSC_TRUE,MPI_MODE_NOPRECEDE,0,0,&target_disp,&win);CHKERRQ(ierr);
  for (i=0; i<nranks; i++) {
    MPI_Aint tdp = target_disp ? target_disp[i] : 0;

    if (w->sync == PETSCSF_WINDOW_SYNC_LOCK) {ierr = MPI_Win_lock(MPI_LOCK_SHARED,ranks[i],MPI_MODE_NOCHECK,win);CHKERRQ(ierr);}
    ierr = MPI_Accumulate((void*)leafdata,1,mine[i],ranks[i],tdp,1,remote[i],op,win);
    if (ierr) { /* intercept the MPI error since the combination of unit and op is not supported */
      PetscMPIInt len;
      char        errstring[MPI_MAX_ERROR_STRING];

      MPI_Error_string(ierr,errstring,&len);
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_SUP,"Internal error in MPI: %s",errstring);
    }
    if (w->sync == PETSCSF_WINDOW_SYNC_LOCK) {ierr = MPI_Win_unlock(ranks[i],win);CHKERRQ(ierr);}
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSFReduceEnd_Window(PetscSF sf,MPI_Datatype unit,PetscMemType leafmtype,const void *leafdata,PetscMemType rootmtype,void *rootdata,MPI_Op op)
{
  PetscSF_Window *w = (PetscSF_Window*)sf->data;
  PetscErrorCode ierr;
  MPI_Win        win;

  PetscFunctionBegin;
  if (!w->wins) PetscFunctionReturn(0);
  ierr = PetscSFFindWindow(sf,unit,rootdata,&win);CHKERRQ(ierr);
  ierr = PetscSFRestoreWindow(sf,unit,rootdata,w->sync,PETSC_TRUE,MPI_MODE_NOSUCCEED,PETSC_TRUE,&win);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSFFetchAndOpBegin_Window(PetscSF sf,MPI_Datatype unit,PetscMemType rootmtype,void *rootdata,PetscMemType leafmtype,const void *leafdata,void *leafupdate,MPI_Op op)
{
  PetscErrorCode     ierr;
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
  ierr = PetscSFGetRootRanks(sf,&nranks,&ranks,NULL,NULL,NULL);CHKERRQ(ierr);
  ierr = PetscSFWindowGetDataTypes(sf,unit,&mine,&remote);CHKERRQ(ierr);
  ierr = PetscSFWindowOpTranslate(&op);CHKERRQ(ierr);
#if !defined(PETSC_HAVE_MPI_GET_ACCUMULATE)
  /* FetchAndOp without MPI_Get_Accumulate requires locking.
     we create a new window every time to not interfere with user-defined MPI_Info which may have used "no_locks"="true" */
  oldf = w->flavor;
  w->flavor = PETSCSF_WINDOW_FLAVOR_CREATE;
  ierr = PetscSFGetWindow(sf,unit,rootdata,PETSCSF_WINDOW_SYNC_LOCK,PETSC_FALSE,0,0,0,&target_disp,&win);CHKERRQ(ierr);
#else
  ierr = PetscSFGetWindow(sf,unit,rootdata,w->sync,PETSC_TRUE,MPI_MODE_NOPRECEDE,0,0,&target_disp,&win);CHKERRQ(ierr);
#endif
  for (i=0; i<nranks; i++) {
    MPI_Aint tdp = target_disp ? target_disp[i] : 0;

#if !defined(PETSC_HAVE_MPI_GET_ACCUMULATE)
    ierr = MPI_Win_lock(MPI_LOCK_EXCLUSIVE,ranks[i],0,win);CHKERRQ(ierr);
    ierr = MPI_Get(leafupdate,1,mine[i],ranks[i],tdp,1,remote[i],win);CHKERRQ(ierr);
    ierr = MPI_Accumulate((void*)leafdata,1,mine[i],ranks[i],tdp,1,remote[i],op,win);
    if (ierr) { /* intercept the MPI error since the combination of unit and op is not supported */
      PetscMPIInt len;
      char        errstring[MPI_MAX_ERROR_STRING];

      MPI_Error_string(ierr,errstring,&len);
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_SUP,"Internal error in MPI: %s",errstring);
    }
    ierr = MPI_Win_unlock(ranks[i],win);CHKERRQ(ierr);
#else
    if (w->sync == PETSCSF_WINDOW_SYNC_LOCK) { ierr = MPI_Win_lock(MPI_LOCK_SHARED,ranks[i],0,win);CHKERRQ(ierr); }
    ierr = MPI_Get_accumulate((void*)leafdata,1,mine[i],leafupdate,1,mine[i],ranks[i],tdp,1,remote[i],op,win);
    if (ierr) { /* intercept the MPI error since the combination of unit and op is not supported */
      PetscMPIInt len;
      char        errstring[MPI_MAX_ERROR_STRING];

      MPI_Error_string(ierr,errstring,&len);
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_SUP,"Internal error in MPI: %s",errstring);
    }
    if (w->sync == PETSCSF_WINDOW_SYNC_LOCK) { ierr = MPI_Win_unlock(ranks[i],win);CHKERRQ(ierr); }
#endif
  }
#if !defined(PETSC_HAVE_MPI_GET_ACCUMULATE)
  w->flavor = oldf;
#endif
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSFFetchAndOpEnd_Window(PetscSF sf,MPI_Datatype unit,PetscMemType rootmtype,void *rootdata,PetscMemType leafmtype,const void *leafdata,void *leafupdate,MPI_Op op)
{
  PetscErrorCode ierr;
  MPI_Win        win;
#if defined(PETSC_HAVE_MPI_GET_ACCUMULATE)
  PetscSF_Window *w = (PetscSF_Window*)sf->data;
#endif

  PetscFunctionBegin;
  ierr = PetscSFFindWindow(sf,unit,rootdata,&win);CHKERRQ(ierr);
#if defined(PETSC_HAVE_MPI_GET_ACCUMULATE)
  ierr = PetscSFRestoreWindow(sf,unit,rootdata,w->sync,PETSC_TRUE,MPI_MODE_NOSUCCEED,PETSC_TRUE,&win);CHKERRQ(ierr);
#else
  ierr = PetscSFRestoreWindow(sf,unit,rootdata,PETSCSF_WINDOW_SYNC_LOCK,PETSC_FALSE,0,PETSC_TRUE,&win);CHKERRQ(ierr);
#endif
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode PetscSFCreate_Window(PetscSF sf)
{
  PetscSF_Window *w = (PetscSF_Window*)sf->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  sf->ops->SetUp           = PetscSFSetUp_Window;
  sf->ops->SetFromOptions  = PetscSFSetFromOptions_Window;
  sf->ops->Reset           = PetscSFReset_Window;
  sf->ops->Destroy         = PetscSFDestroy_Window;
  sf->ops->View            = PetscSFView_Window;
  sf->ops->Duplicate       = PetscSFDuplicate_Window;
  sf->ops->BcastAndOpBegin = PetscSFBcastAndOpBegin_Window;
  sf->ops->BcastAndOpEnd   = PetscSFBcastAndOpEnd_Window;
  sf->ops->ReduceBegin     = PetscSFReduceBegin_Window;
  sf->ops->ReduceEnd       = PetscSFReduceEnd_Window;
  sf->ops->FetchAndOpBegin = PetscSFFetchAndOpBegin_Window;
  sf->ops->FetchAndOpEnd   = PetscSFFetchAndOpEnd_Window;

  ierr = PetscNewLog(sf,&w);CHKERRQ(ierr);
  sf->data  = (void*)w;
  w->sync   = PETSCSF_WINDOW_SYNC_FENCE;
  w->flavor = PETSCSF_WINDOW_FLAVOR_CREATE;
  w->info   = MPI_INFO_NULL;

  ierr = PetscObjectComposeFunction((PetscObject)sf,"PetscSFWindowSetSyncType_C",PetscSFWindowSetSyncType_Window);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)sf,"PetscSFWindowGetSyncType_C",PetscSFWindowGetSyncType_Window);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)sf,"PetscSFWindowSetFlavorType_C",PetscSFWindowSetFlavorType_Window);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)sf,"PetscSFWindowGetFlavorType_C",PetscSFWindowGetFlavorType_Window);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)sf,"PetscSFWindowSetInfo_C",PetscSFWindowSetInfo_Window);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)sf,"PetscSFWindowGetInfo_C",PetscSFWindowGetInfo_Window);CHKERRQ(ierr);

#if defined(OMPI_MAJOR_VERSION) && (OMPI_MAJOR_VERSION < 1 || (OMPI_MAJOR_VERSION == 1 && OMPI_MINOR_VERSION <= 6))
  {
    PetscBool ackbug = PETSC_FALSE;
    ierr = PetscOptionsGetBool(NULL,NULL,"-acknowledge_ompi_onesided_bug",&ackbug,NULL);CHKERRQ(ierr);
    if (ackbug) {
      ierr = PetscInfo(sf,"Acknowledged Open MPI bug, proceeding anyway. Expect memory corruption.\n");CHKERRQ(ierr);
    } else SETERRQ(PetscObjectComm((PetscObject)sf),PETSC_ERR_LIB,"Open MPI is known to be buggy (https://svn.open-mpi.org/trac/ompi/ticket/1905 and 2656), use -acknowledge_ompi_onesided_bug to proceed");
  }
#endif
  PetscFunctionReturn(0);
}
