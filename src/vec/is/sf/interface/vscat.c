#include "petscsf.h"
#include "petscsystypes.h"
#include "petscvec.h"
#include <petsc/private/sfimpl.h> /*I "petscsf.h" I*/
#include <../src/vec/is/sf/impls/basic/sfbasic.h> /* for VecScatterRemap_Internal */
#include <../src/vec/is/sf/impls/basic/sfpack.h>
#include <petsc/private/vecimpl.h>

typedef enum {IS_INVALID, IS_GENERAL, IS_BLOCK, IS_STRIDE} ISTypeID;

static inline PetscErrorCode ISGetTypeID_Private(IS is,ISTypeID *id)
{
  PetscBool      same;

  PetscFunctionBegin;
  *id  = IS_INVALID;
  CHKERRQ(PetscObjectTypeCompare((PetscObject)is,ISGENERAL,&same));
  if (same) {*id = IS_GENERAL; goto functionend;}
  CHKERRQ(PetscObjectTypeCompare((PetscObject)is,ISBLOCK,&same));
  if (same) {*id = IS_BLOCK; goto functionend;}
  CHKERRQ(PetscObjectTypeCompare((PetscObject)is,ISSTRIDE,&same));
  if (same) {*id = IS_STRIDE; goto functionend;}
functionend:
  PetscFunctionReturn(0);
}

static PetscErrorCode VecScatterBegin_Internal(VecScatter sf,Vec x,Vec y,InsertMode addv,ScatterMode mode)
{
  PetscSF        wsf=NULL; /* either sf or its local part */
  MPI_Op         mop=MPI_OP_NULL;
  PetscMPIInt    size;
  PetscMemType   xmtype=PETSC_MEMTYPE_HOST,ymtype=PETSC_MEMTYPE_HOST;

  PetscFunctionBegin;
  if (x != y) CHKERRQ(VecLockReadPush(x));
  CHKERRQ(VecGetArrayReadAndMemType(x,&sf->vscat.xdata,&xmtype));
  CHKERRQ(VecGetArrayAndMemType(y,&sf->vscat.ydata,&ymtype));
  CHKERRQ(VecLockWriteSet_Private(y,PETSC_TRUE));

  /* SCATTER_LOCAL indicates ignoring inter-process communication */
  CHKERRMPI(MPI_Comm_size(PetscObjectComm((PetscObject)sf),&size));
  if ((mode & SCATTER_LOCAL) && size > 1) { /* Lazy creation of sf->vscat.lsf since SCATTER_LOCAL is uncommon */
    if (!sf->vscat.lsf) CHKERRQ(PetscSFCreateLocalSF_Private(sf,&sf->vscat.lsf));
    wsf = sf->vscat.lsf;
  } else {
    wsf = sf;
  }

  /* Note xdata/ydata is always recorded on sf (not lsf) above */
  if (addv == INSERT_VALUES)   mop = MPI_REPLACE;
  else if (addv == ADD_VALUES) mop = MPIU_SUM; /* Petsc defines its own MPI datatype and SUM operation for __float128 etc. */
  else if (addv == MAX_VALUES) mop = MPIU_MAX;
  else if (addv == MIN_VALUES) mop = MPIU_MIN;
  else SETERRQ(PetscObjectComm((PetscObject)sf),PETSC_ERR_SUP,"Unsupported InsertMode %d in VecScatterBegin/End",addv);

  if (mode & SCATTER_REVERSE) { /* REVERSE indicates leaves to root scatter. Note that x and y are swapped in input */
    CHKERRQ(PetscSFReduceWithMemTypeBegin(wsf,sf->vscat.unit,xmtype,sf->vscat.xdata,ymtype,sf->vscat.ydata,mop));
  } else { /* FORWARD indicates x to y scatter, where x is root and y is leaf */
    CHKERRQ(PetscSFBcastWithMemTypeBegin(wsf,sf->vscat.unit,xmtype,sf->vscat.xdata,ymtype,sf->vscat.ydata,mop));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode VecScatterEnd_Internal(VecScatter sf,Vec x,Vec y,InsertMode addv,ScatterMode mode)
{
  PetscSF        wsf=NULL;
  MPI_Op         mop=MPI_OP_NULL;
  PetscMPIInt    size;

  PetscFunctionBegin;
  /* SCATTER_LOCAL indicates ignoring inter-process communication */
  CHKERRMPI(MPI_Comm_size(PetscObjectComm((PetscObject)sf),&size));
  wsf  = ((mode & SCATTER_LOCAL) && size > 1) ? sf->vscat.lsf : sf;

  if (addv == INSERT_VALUES)   mop = MPI_REPLACE;
  else if (addv == ADD_VALUES) mop = MPIU_SUM;
  else if (addv == MAX_VALUES) mop = MPIU_MAX;
  else if (addv == MIN_VALUES) mop = MPIU_MIN;
  else SETERRQ(PetscObjectComm((PetscObject)sf),PETSC_ERR_SUP,"Unsupported InsertMode %d in VecScatterBegin/End",addv);

  if (mode & SCATTER_REVERSE) { /* reverse scatter sends leaves to roots. Note that x and y are swapped in input */
    CHKERRQ(PetscSFReduceEnd(wsf,sf->vscat.unit,sf->vscat.xdata,sf->vscat.ydata,mop));
  } else { /* forward scatter sends roots to leaves, i.e., x to y */
    CHKERRQ(PetscSFBcastEnd(wsf,sf->vscat.unit,sf->vscat.xdata,sf->vscat.ydata,mop));
  }

  CHKERRQ(VecRestoreArrayReadAndMemType(x,&sf->vscat.xdata));
  if (x != y) CHKERRQ(VecLockReadPop(x));
  CHKERRQ(VecRestoreArrayAndMemType(y,&sf->vscat.ydata));
  CHKERRQ(VecLockWriteSet_Private(y,PETSC_FALSE));
  PetscFunctionReturn(0);
}

/* VecScatterRemap provides a light way to slightly modify a VecScatter. Suppose the input sf scatters
   x[i] to y[j], tomap gives a plan to change vscat to scatter x[tomap[i]] to y[j]. Note that in SF,
   x is roots. That means we need to change incoming stuffs such as bas->irootloc[].
 */
static PetscErrorCode VecScatterRemap_Internal(VecScatter sf,const PetscInt *tomap,const PetscInt *frommap)
{
  PetscInt       i,bs = sf->vscat.bs;
  PetscMPIInt    size;
  PetscBool      ident = PETSC_TRUE,isbasic,isneighbor;
  PetscSFType    type;
  PetscSF_Basic  *bas = NULL;

  PetscFunctionBegin;
  /* check if it is an identity map. If it is, do nothing */
  if (tomap) {
    for (i=0; i<sf->nroots*bs; i++) {if (i != tomap[i]) {ident = PETSC_FALSE; break; } }
    if (ident) PetscFunctionReturn(0);
  }
  PetscCheckFalse(frommap,PETSC_COMM_SELF,PETSC_ERR_SUP,"Unable to remap the FROM in scatters yet");
  if (!tomap) PetscFunctionReturn(0);

  CHKERRMPI(MPI_Comm_size(PetscObjectComm((PetscObject)sf),&size));

  /* Since the indices changed, we must also update the local SF. But we do not do it since
     lsf is rarely used. We just destroy lsf and rebuild it on demand from updated sf.
  */
  if (sf->vscat.lsf) CHKERRQ(PetscSFDestroy(&sf->vscat.lsf));

  CHKERRQ(PetscSFGetType(sf,&type));
  CHKERRQ(PetscObjectTypeCompare((PetscObject)sf,PETSCSFBASIC,&isbasic));
  CHKERRQ(PetscObjectTypeCompare((PetscObject)sf,PETSCSFNEIGHBOR,&isneighbor));
  PetscCheckFalse(!isbasic && !isneighbor,PetscObjectComm((PetscObject)sf),PETSC_ERR_SUP,"VecScatterRemap on SF type %s is not supported",type);

  CHKERRQ(PetscSFSetUp(sf)); /* to bulid sf->irootloc if SetUp is not yet called */

  /* Root indices are going to be remapped. This is tricky for SF. Root indices are used in sf->rremote,
    sf->remote and bas->irootloc. The latter one is cheap to remap, but the former two are not.
    To remap them, we have to do a bcast from roots to leaves, to let leaves know their updated roots.
    Since VecScatterRemap is supposed to be a cheap routine to adapt a vecscatter by only changing where
    x[] data is taken, we do not remap sf->rremote, sf->remote. The consequence is that operations
    accessing them (such as PetscSFCompose) may get stale info. Considering VecScatter does not need
    that complicated SF operations, we do not remap sf->rremote, sf->remote, instead we destroy them
    so that code accessing them (if any) will crash (instead of get silent errors). Note that BcastAndOp/Reduce,
    which are used by VecScatter and only rely on bas->irootloc, are updated and correct.
  */
  sf->remote = NULL;
  CHKERRQ(PetscFree(sf->remote_alloc));
  /* Not easy to free sf->rremote since it was allocated with PetscMalloc4(), so just give it crazy values */
  for (i=0; i<sf->roffset[sf->nranks]; i++) sf->rremote[i] = PETSC_MIN_INT;

  /* Indices in tomap[] are for each indivisual vector entry. But indices in sf are for each
     block in the vector. So before the remapping, we have to expand indices in sf by bs, and
     after the remapping, we have to shrink them back.
   */
  bas = (PetscSF_Basic*)sf->data;
  for (i=0; i<bas->ioffset[bas->niranks]; i++) bas->irootloc[i] = tomap[bas->irootloc[i]*bs]/bs;
#if defined(PETSC_HAVE_DEVICE)
  /* Free the irootloc copy on device. We allocate a new copy and get the updated value on demand. See PetscSFLinkGetRootPackOptAndIndices() */
  for (i=0; i<2; i++) CHKERRQ(PetscSFFree(sf,PETSC_MEMTYPE_DEVICE,bas->irootloc_d[i]));
#endif
  /* Destroy and then rebuild root packing optimizations since indices are changed */
  CHKERRQ(PetscSFResetPackFields(sf));
  CHKERRQ(PetscSFSetUpPackFields(sf));
  PetscFunctionReturn(0);
}

/* Given a parallel VecScatter context, return number of procs and vector entries involved in remote (i.e., off-process) communication

  Input Parameters:
+ sf   - the context (must be a parallel vecscatter)
- send  - true to select the send info (i.e., todata), otherwise to select the recv info (i.e., fromdata)

  Output parameters:
+ num_procs   - number of remote processors
- num_entries - number of vector entries to send or recv

  .seealso: VecScatterGetRemote_Private(), VecScatterGetRemoteOrdered_Private()

  Notes:
  Sometimes PETSc internally needs to use the matrix-vector-multiply vecscatter context for other purposes. The client code
  usually only uses MPI_Send/Recv. This group of subroutines provides info needed for such uses.
 */
PetscErrorCode VecScatterGetRemoteCount_Private(VecScatter sf,PetscBool send,PetscInt *num_procs,PetscInt *num_entries)
{
  PetscInt          nranks,remote_start;
  PetscMPIInt       rank;
  const PetscInt    *offset;
  const PetscMPIInt *ranks;

  PetscFunctionBegin;
  CHKERRQ(PetscSFSetUp(sf));
  CHKERRMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)sf),&rank));

  /* This routine is mainly used for MatMult's Mvctx. In Mvctx, we scatter an MPI vector x to a sequential vector lvec.
     Remember x is roots and lvec is leaves. 'send' means roots to leaves communication. If 'send' is true, we need to
     get info about which ranks this processor needs to send to. In other words, we need to call PetscSFGetLeafRanks().
     If send is false, we do the opposite, calling PetscSFGetRootRanks().
  */
  if (send) CHKERRQ(PetscSFGetLeafRanks(sf,&nranks,&ranks,&offset,NULL));
  else CHKERRQ(PetscSFGetRootRanks(sf,&nranks,&ranks,&offset,NULL,NULL));
  if (nranks) {
    remote_start = (rank == ranks[0])? 1 : 0;
    if (num_procs)   *num_procs   = nranks - remote_start;
    if (num_entries) *num_entries = offset[nranks] - offset[remote_start];
  } else {
    if (num_procs)   *num_procs   = 0;
    if (num_entries) *num_entries = 0;
  }
  PetscFunctionReturn(0);
}

/* Given a parallel VecScatter context, return a plan that represents the remote communication.
   Any output parameter can be NULL.

  Input Parameters:
+ sf   - the context
- send  - true to select the send info (i.e., todata), otherwise to select the recv info (i.e., fromdata)

  Output parameters:
+ n        - number of remote processors
. starts   - starting point in indices for each proc. ATTENTION: starts[0] is not necessarily zero.
             Therefore, expressions like starts[i+1]-starts[i] and indices[starts[i]+j] work as
             expected for a CSR structure but buf[starts[i]+j] may be out of range if buf was allocated
             with length starts[n]-starts[0]. One should use buf[starts[i]-starts[0]+j] instead.
. indices  - indices of entries to send/recv
. procs    - ranks of remote processors
- bs       - block size

  .seealso: VecScatterRestoreRemote_Private(), VecScatterGetRemoteOrdered_Private()
 */
PetscErrorCode VecScatterGetRemote_Private(VecScatter sf,PetscBool send,PetscInt *n,const PetscInt **starts,const PetscInt **indices,const PetscMPIInt **procs,PetscInt *bs)
{
  PetscInt          nranks,remote_start;
  PetscMPIInt       rank;
  const PetscInt    *offset,*location;
  const PetscMPIInt *ranks;

  PetscFunctionBegin;
  CHKERRQ(PetscSFSetUp(sf));
  CHKERRMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)sf),&rank));

  if (send) CHKERRQ(PetscSFGetLeafRanks(sf,&nranks,&ranks,&offset,&location));
  else CHKERRQ(PetscSFGetRootRanks(sf,&nranks,&ranks,&offset,&location,NULL));

  if (nranks) {
    remote_start = (rank == ranks[0])? 1 : 0;
    if (n)       *n       = nranks - remote_start;
    if (starts)  *starts  = &offset[remote_start];
    if (indices) *indices = location; /* not &location[offset[remote_start]]. Starts[0] may point to the middle of indices[] */
    if (procs)   *procs   = &ranks[remote_start];
  } else {
    if (n)       *n       = 0;
    if (starts)  *starts  = NULL;
    if (indices) *indices = NULL;
    if (procs)   *procs   = NULL;
  }

  if (bs) *bs = 1;
  PetscFunctionReturn(0);
}

/* Given a parallel VecScatter context, return a plan that represents the remote communication. Ranks of remote
   processors returned in procs must be sorted in ascending order. Any output parameter can be NULL.

  Input Parameters:
+ sf   - the context
- send  - true to select the send info (i.e., todata), otherwise to select the recv info (i.e., fromdata)

  Output parameters:
+ n        - number of remote processors
. starts   - starting point in indices for each proc. ATTENTION: starts[0] is not necessarily zero.
             Therefore, expressions like starts[i+1]-starts[i] and indices[starts[i]+j] work as
             expected for a CSR structure but buf[starts[i]+j] may be out of range if buf was allocated
             with length starts[n]-starts[0]. One should use buf[starts[i]-starts[0]+j] instead.
. indices  - indices of entries to send/recv
. procs    - ranks of remote processors
- bs       - block size

  .seealso: VecScatterRestoreRemoteOrdered_Private(), VecScatterGetRemote_Private()

  Notes:
  Output parameters like starts, indices must also be adapted according to the sorted ranks.
 */
PetscErrorCode VecScatterGetRemoteOrdered_Private(VecScatter sf,PetscBool send,PetscInt *n,const PetscInt **starts,const PetscInt **indices,const PetscMPIInt **procs,PetscInt *bs)
{
  PetscFunctionBegin;
  CHKERRQ(VecScatterGetRemote_Private(sf,send,n,starts,indices,procs,bs));
  if (PetscUnlikelyDebug(n && procs)) {
    PetscInt i;
    /* from back to front to also handle cases *n=0 */
    for (i=*n-1; i>0; i--) { PetscCheckFalse((*procs)[i-1] > (*procs)[i],PETSC_COMM_SELF,PETSC_ERR_PLIB,"procs[] are not ordered"); }
  }
  PetscFunctionReturn(0);
}

/* Given a parallel VecScatter context, restore the plan returned by VecScatterGetRemote_Private. This gives a chance for
   an implementation to free memory allocated in the VecScatterGetRemote_Private call.

  Input Parameters:
+ sf   - the context
- send  - true to select the send info (i.e., todata), otherwise to select the recv info (i.e., fromdata)

  Output parameters:
+ n        - number of remote processors
. starts   - starting point in indices for each proc
. indices  - indices of entries to send/recv
. procs    - ranks of remote processors
- bs       - block size

  .seealso: VecScatterGetRemote_Private()
 */
PetscErrorCode VecScatterRestoreRemote_Private(VecScatter sf,PetscBool send,PetscInt *n,const PetscInt **starts,const PetscInt **indices,const PetscMPIInt **procs,PetscInt *bs)
{
  PetscFunctionBegin;
  if (starts)   *starts  = NULL;
  if (indices)  *indices = NULL;
  if (procs)    *procs   = NULL;
  PetscFunctionReturn(0);
}

/* Given a parallel VecScatter context, restore the plan returned by VecScatterGetRemoteOrdered_Private. This gives a chance for
   an implementation to free memory allocated in the VecScatterGetRemoteOrdered_Private call.

  Input Parameters:
+ sf   - the context
- send  - true to select the send info (i.e., todata), otherwise to select the recv info (i.e., fromdata)

  Output parameters:
+ n        - number of remote processors
. starts   - starting point in indices for each proc
. indices  - indices of entries to send/recv
. procs    - ranks of remote processors
- bs       - block size

  .seealso: VecScatterGetRemoteOrdered_Private()
 */
PetscErrorCode VecScatterRestoreRemoteOrdered_Private(VecScatter sf,PetscBool send,PetscInt *n,const PetscInt **starts,const PetscInt **indices,const PetscMPIInt **procs,PetscInt *bs)
{
  PetscFunctionBegin;
  CHKERRQ(VecScatterRestoreRemote_Private(sf,send,n,starts,indices,procs,bs));
  PetscFunctionReturn(0);
}

/*@
   VecScatterSetUp - Sets up the VecScatter to be able to actually scatter information between vectors

   Collective on VecScatter

   Input Parameter:
.  sf - the scatter context

   Level: intermediate

.seealso: VecScatterCreate(), VecScatterCopy()
@*/
PetscErrorCode VecScatterSetUp(VecScatter sf)
{
  PetscFunctionBegin;
  CHKERRQ(PetscSFSetUp(sf));
  PetscFunctionReturn(0);
}

/*@C
  VecScatterSetType - Builds a vector scatter, for a particular vector scatter implementation.

  Collective on VecScatter

  Input Parameters:
+ sf - The VecScatter (SF) object
- type - The name of the vector scatter type

  Options Database Key:
. -sf_type <type> - Sets the VecScatter (SF) type

  Notes:
  Use VecScatterDuplicate() to form additional vectors scatter of the same type as an existing vector scatter.

  Level: intermediate

.seealso: VecScatterGetType(), VecScatterCreate()
@*/
PetscErrorCode VecScatterSetType(VecScatter sf, VecScatterType type)
{
  PetscFunctionBegin;
  CHKERRQ(PetscSFSetType(sf,type));
  PetscFunctionReturn(0);
}

/*@C
  VecScatterGetType - Gets the vector scatter type name (as a string) from the VecScatter.

  Not Collective

  Input Parameter:
. sf  - The vector scatter (SF)

  Output Parameter:
. type - The vector scatter type name

  Level: intermediate

.seealso: VecScatterSetType(), VecScatterCreate()
@*/
PetscErrorCode VecScatterGetType(VecScatter sf, VecScatterType *type)
{
  PetscFunctionBegin;
  CHKERRQ(PetscSFGetType(sf,type));
  PetscFunctionReturn(0);
}

/*@C
  VecScatterRegister -  Adds a new vector scatter component implementation

  Not Collective

  Input Parameters:
+ name        - The name of a new user-defined creation routine
- create_func - The creation routine itself

  Level: advanced

.seealso: VecRegister()
@*/
PetscErrorCode VecScatterRegister(const char sname[], PetscErrorCode (*function)(VecScatter))
{
  PetscFunctionBegin;
  CHKERRQ(PetscSFRegister(sname,function));
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------------*/
/*@
   VecScatterGetMerged - Returns true if the scatter is completed in the VecScatterBegin()
      and the VecScatterEnd() does nothing

   Not Collective

   Input Parameter:
.   sf - scatter context created with VecScatterCreate()

   Output Parameter:
.   flg - PETSC_TRUE if the VecScatterBegin/End() are all done during the VecScatterBegin()

   Level: developer

.seealso: VecScatterCreate(), VecScatterEnd(), VecScatterBegin()
@*/
PetscErrorCode  VecScatterGetMerged(VecScatter sf,PetscBool *flg)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(sf,PETSCSF_CLASSID,1);
  if (flg) *flg = sf->vscat.beginandendtogether;
  PetscFunctionReturn(0);
}
/*@C
   VecScatterDestroy - Destroys a scatter context created by VecScatterCreate()

   Collective on VecScatter

   Input Parameter:
.  sf - the scatter context

   Level: intermediate

.seealso: VecScatterCreate(), VecScatterCopy()
@*/
PetscErrorCode VecScatterDestroy(VecScatter *sf)
{
  PetscFunctionBegin;
  CHKERRQ(PetscSFDestroy(sf));
  PetscFunctionReturn(0);
}

/*@
   VecScatterCopy - Makes a copy of a scatter context.

   Collective on VecScatter

   Input Parameter:
.  sf - the scatter context

   Output Parameter:
.  newsf - the context copy

   Level: advanced

.seealso: VecScatterCreate(), VecScatterDestroy()
@*/
PetscErrorCode  VecScatterCopy(VecScatter sf,VecScatter *newsf)
{
  PetscFunctionBegin;
  PetscValidPointer(newsf,2);
  CHKERRQ(PetscSFDuplicate(sf,PETSCSF_DUPLICATE_GRAPH,newsf));
  CHKERRQ(PetscSFSetUp(*newsf));
  PetscFunctionReturn(0);
}

/*@C
   VecScatterViewFromOptions - View from Options

   Collective on VecScatter

   Input Parameters:
+  sf - the scatter context
.  obj - Optional object
-  name - command line option

   Level: intermediate
.seealso:  VecScatter, VecScatterView, PetscObjectViewFromOptions(), VecScatterCreate()
@*/
PetscErrorCode  VecScatterViewFromOptions(VecScatter sf,PetscObject obj,const char name[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(sf,PETSCSF_CLASSID,1);
  CHKERRQ(PetscObjectViewFromOptions((PetscObject)sf,obj,name));
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------------*/
/*@C
   VecScatterView - Views a vector scatter context.

   Collective on VecScatter

   Input Parameters:
+  sf - the scatter context
-  viewer - the viewer for displaying the context

   Level: intermediate

@*/
PetscErrorCode  VecScatterView(VecScatter sf,PetscViewer viewer)
{
  PetscFunctionBegin;
  CHKERRQ(PetscSFView(sf,viewer));
  PetscFunctionReturn(0);
}

/*@C
   VecScatterRemap - Remaps the "from" and "to" indices in a
   vector scatter context. FOR EXPERTS ONLY!

   Collective on VecScatter

   Input Parameters:
+  sf    - vector scatter context
.  tomap   - remapping plan for "to" indices (may be NULL).
-  frommap - remapping plan for "from" indices (may be NULL)

   Level: developer

   Notes:
     In the parallel case the todata contains indices from where the data is taken
     (and then sent to others)! The fromdata contains indices from where the received
     data is finally put locally.

     In the sequential case the todata contains indices from where the data is put
     and the fromdata contains indices from where the data is taken from.
     This is backwards from the paralllel case!

@*/
PetscErrorCode  VecScatterRemap(VecScatter sf,PetscInt tomap[],PetscInt frommap[])
{
  PetscFunctionBegin;
  if (tomap)   PetscValidIntPointer(tomap,2);
  if (frommap) PetscValidIntPointer(frommap,3);
  CHKERRQ(VecScatterRemap_Internal(sf,tomap,frommap));
  PetscCheckFalse(frommap,PETSC_COMM_SELF,PETSC_ERR_SUP,"Unable to remap the FROM in scatters yet");
  /* Mark then vector lengths as unknown because we do not know the lengths of the remapped vectors */
  sf->vscat.from_n = -1;
  sf->vscat.to_n   = -1;
  PetscFunctionReturn(0);
}

/*@
  VecScatterSetFromOptions - Configures the vector scatter from the options database.

  Collective on VecScatter

  Input Parameter:
. sf - The vector scatter

  Notes:
    To see all options, run your program with the -help option, or consult the users manual.
          Must be called before VecScatterSetUp() but before the vector scatter is used.

  Level: beginner

.seealso: VecScatterCreate(), VecScatterDestroy(), VecScatterSetUp()
@*/
PetscErrorCode VecScatterSetFromOptions(VecScatter sf)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sf,PETSCSF_CLASSID,1);
  ierr = PetscObjectOptionsBegin((PetscObject)sf);CHKERRQ(ierr);

  sf->vscat.beginandendtogether = PETSC_FALSE;
  CHKERRQ(PetscOptionsBool("-vecscatter_merge","Use combined (merged) vector scatter begin and end","VecScatterCreate",sf->vscat.beginandendtogether,&sf->vscat.beginandendtogether,NULL));
  if (sf->vscat.beginandendtogether) CHKERRQ(PetscInfo(sf,"Using combined (merged) vector scatter begin and end\n"));
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------- */
/*@
   VecScatterCreate - Creates a vector scatter context.

   Collective on Vec

   Input Parameters:
+  xin - a vector that defines the shape (parallel data layout of the vector)
         of vectors from which we scatter
.  yin - a vector that defines the shape (parallel data layout of the vector)
         of vectors to which we scatter
.  ix - the indices of xin to scatter (if NULL scatters all values)
-  iy - the indices of yin to hold results (if NULL fills entire vector yin)

   Output Parameter:
.  newsf - location to store the new scatter (SF) context

   Options Database Keys:
+  -vecscatter_view         - Prints detail of communications
.  -vecscatter_view ::ascii_info    - Print less details about communication
-  -vecscatter_merge        - VecScatterBegin() handles all of the communication, VecScatterEnd() is a nop
                              eliminates the chance for overlap of computation and communication

  Level: intermediate

  Notes:
   If both xin and yin are parallel, their communicator must be on the same
   set of processes, but their process order can be different.
   In calls to VecScatter() you can use different vectors than the xin and
   yin you used above; BUT they must have the same parallel data layout, for example,
   they could be obtained from VecDuplicate().
   A VecScatter context CANNOT be used in two or more simultaneous scatters;
   that is you cannot call a second VecScatterBegin() with the same scatter
   context until the VecScatterEnd() has been called on the first VecScatterBegin().
   In this case a separate VecScatter is needed for each concurrent scatter.

   Currently the MPI_Send() use PERSISTENT versions.
   (this unfortunately requires that the same in and out arrays be used for each use, this
    is why  we always need to pack the input into the work array before sending
    and unpack upon receiving instead of using MPI datatypes to avoid the packing/unpacking).

   Both ix and iy cannot be NULL at the same time.

   Use VecScatterCreateToAll() to create a vecscatter that copies an MPI vector to sequential vectors on all MPI ranks.
   Use VecScatterCreateToZero() to create a vecscatter that copies an MPI vector to a sequential vector on MPI rank 0.
   These special vecscatters have better performance than general ones.

.seealso: VecScatterDestroy(), VecScatterCreateToAll(), VecScatterCreateToZero(), PetscSFCreate()
@*/
PetscErrorCode VecScatterCreate(Vec x,IS ix,Vec y,IS iy,VecScatter *newsf)
{
  MPI_Comm       xcomm,ycomm,bigcomm;
  Vec            xx,yy;
  IS             ix_old=ix,iy_old=iy,ixx,iyy;
  PetscMPIInt    xcommsize,ycommsize,rank,result;
  PetscInt       i,n,N,nroots,nleaves,*ilocal,xstart,ystart,ixsize,iysize,xlen,ylen;
  const PetscInt *xindices,*yindices;
  PetscSFNode    *iremote;
  PetscLayout    xlayout,ylayout;
  ISTypeID       ixid,iyid;
  PetscInt       bs,bsx,bsy,min,max,m[2],ixfirst,ixstep,iyfirst,iystep;
  PetscBool      can_do_block_opt=PETSC_FALSE;
  PetscSF        sf;

  PetscFunctionBegin;
  PetscValidPointer(newsf,5);
  PetscCheckFalse(!ix && !iy,PetscObjectComm((PetscObject)x),PETSC_ERR_SUP,"Cannot pass default in for both input and output indices");

  /* Get comm from x and y */
  CHKERRQ(PetscObjectGetComm((PetscObject)x,&xcomm));
  CHKERRMPI(MPI_Comm_size(xcomm,&xcommsize));
  CHKERRQ(PetscObjectGetComm((PetscObject)y,&ycomm));
  CHKERRMPI(MPI_Comm_size(ycomm,&ycommsize));
  if (xcommsize > 1 && ycommsize > 1) {
    CHKERRMPI(MPI_Comm_compare(xcomm,ycomm,&result));
    PetscCheckFalse(result == MPI_UNEQUAL,PETSC_COMM_SELF,PETSC_ERR_ARG_NOTSAMECOMM,"VecScatterCreate: parallel vectors x and y must have identical/congruent/similar communicators");
  }
  bs = 1; /* default, no blocking */

  /*
   Let P and S stand for parallel and sequential vectors respectively. There are four combinations of vecscatters: PtoP, PtoS,
   StoP and StoS. The assumption of VecScatterCreate(Vec x,IS ix,Vec y,IS iy,VecScatter *newctx) is: if x is parallel, then ix
   contains global indices of x. If x is sequential, ix contains local indices of x. Similarily for y and iy.

   SF builds around concepts of local leaves and remote roots. We treat source vector x as roots and destination vector y as
   leaves. A PtoS scatter can be naturally mapped to SF. We transform PtoP and StoP to PtoS, and treat StoS as trivial PtoS.
  */

  /* NULL ix or iy in VecScatterCreate(x,ix,y,iy,newctx) has special meaning. Recover them for these cases */
  if (!ix) {
    if (xcommsize > 1 && ycommsize == 1) { /* PtoS: null ix means the whole x will be scattered to each seq y */
      CHKERRQ(VecGetSize(x,&N));
      CHKERRQ(ISCreateStride(PETSC_COMM_SELF,N,0,1,&ix));
    } else { /* PtoP, StoP or StoS: null ix means the whole local part of x will be scattered */
      CHKERRQ(VecGetLocalSize(x,&n));
      CHKERRQ(VecGetOwnershipRange(x,&xstart,NULL));
      CHKERRQ(ISCreateStride(PETSC_COMM_SELF,n,xstart,1,&ix));
    }
  }

  if (!iy) {
    if (xcommsize == 1 && ycommsize > 1) { /* StoP: null iy means the whole y will be scattered to from each seq x */
      CHKERRQ(VecGetSize(y,&N));
      CHKERRQ(ISCreateStride(PETSC_COMM_SELF,N,0,1,&iy));
    } else { /* PtoP, StoP or StoS: null iy means the whole local part of y will be scattered to */
      CHKERRQ(VecGetLocalSize(y,&n));
      CHKERRQ(VecGetOwnershipRange(y,&ystart,NULL));
      CHKERRQ(ISCreateStride(PETSC_COMM_SELF,n,ystart,1,&iy));
    }
  }

  /* Do error checking immediately after we have non-empty ix, iy */
  CHKERRQ(ISGetLocalSize(ix,&ixsize));
  CHKERRQ(ISGetLocalSize(iy,&iysize));
  CHKERRQ(VecGetSize(x,&xlen));
  CHKERRQ(VecGetSize(y,&ylen));
  PetscCheckFalse(ixsize != iysize,PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Scatter sizes of ix and iy don't match locally ix=%" PetscInt_FMT " iy=%" PetscInt_FMT,ixsize,iysize);
  CHKERRQ(ISGetMinMax(ix,&min,&max));
  PetscCheckFalse(min < 0 || max >= xlen,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Scatter indices in ix are out of range");
  CHKERRQ(ISGetMinMax(iy,&min,&max));
  PetscCheckFalse(min < 0 || max >= ylen,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Scatter indices in iy are out of range");

  /* Extract info about ix, iy for further test */
  CHKERRQ(ISGetTypeID_Private(ix,&ixid));
  CHKERRQ(ISGetTypeID_Private(iy,&iyid));
  if (ixid == IS_BLOCK)       CHKERRQ(ISGetBlockSize(ix,&bsx));
  else if (ixid == IS_STRIDE) CHKERRQ(ISStrideGetInfo(ix,&ixfirst,&ixstep));

  if (iyid == IS_BLOCK)      CHKERRQ(ISGetBlockSize(iy,&bsy));
  else if (iyid == IS_STRIDE) CHKERRQ(ISStrideGetInfo(iy,&iyfirst,&iystep));

  /* Check if a PtoS is special ToAll/ToZero scatters, which can be results of VecScatterCreateToAll/Zero.
     ToAll means a whole MPI vector is copied to a seq vector on every process. ToZero means a whole MPI
     vector is copied to a seq vector on rank 0 and other processes do nothing(i.e.,they input empty ix,iy).

     We can optimize these scatters with MPI collectives. We can also avoid costly analysis used for general scatters.
  */
  if (xcommsize > 1 && ycommsize == 1) { /* Ranks do not diverge at this if-test */
    PetscInt    pattern[2] = {0, 0}; /* A boolean array with pattern[0] for allgather-like (ToAll) and pattern[1] for gather-like (ToZero) */
    PetscLayout map;

    CHKERRMPI(MPI_Comm_rank(xcomm,&rank));
    CHKERRQ(VecGetLayout(x,&map));
    if (rank == 0) {
      if (ixid == IS_STRIDE && iyid == IS_STRIDE && ixsize == xlen && ixfirst == 0 && ixstep == 1 && iyfirst == 0 && iystep == 1) {
        /* Rank 0 scatters the whole mpi x to seq y, so it is either a ToAll or a ToZero candidate in its view */
        pattern[0] = pattern[1] = 1;
      }
    } else {
      if (ixid == IS_STRIDE && iyid == IS_STRIDE && ixsize == xlen && ixfirst == 0 && ixstep == 1 && iyfirst == 0 && iystep == 1) {
        /* Other ranks also scatter the whole mpi x to seq y, so it is a ToAll candidate in their view */
        pattern[0] = 1;
      } else if (ixsize == 0) {
        /* Other ranks do nothing, so it is a ToZero candiate */
        pattern[1] = 1;
      }
    }

    /* One stone (the expensive allreduce) two birds: pattern[] tells if it is ToAll or ToZero */
    CHKERRMPI(MPIU_Allreduce(MPI_IN_PLACE,pattern,2,MPIU_INT,MPI_LAND,xcomm));

    if (pattern[0] || pattern[1]) {
      CHKERRQ(PetscSFCreate(xcomm,&sf));
      CHKERRQ(PetscSFSetFromOptions(sf));
      CHKERRQ(PetscSFSetGraphWithPattern(sf,map,pattern[0] ? PETSCSF_PATTERN_ALLGATHER : PETSCSF_PATTERN_GATHER));
      goto functionend; /* No further analysis needed. What a big win! */
    }
  }

  /* Continue ...
     Do block optimization by taking advantage of high level info available in ix, iy.
     The block optimization is valid when all of the following conditions are met:
     1) ix, iy are blocked or can be blocked (i.e., strided with step=1);
     2) ix, iy have the same block size;
     3) all processors agree on one block size;
     4) no blocks span more than one process;
   */
  bigcomm = (xcommsize == 1) ? ycomm : xcomm;

  /* Processors could go through different path in this if-else test */
  m[0] = m[1] = PETSC_MPI_INT_MIN;
  if (ixid == IS_BLOCK && iyid == IS_BLOCK) {
    m[0] = PetscMax(bsx,bsy);
    m[1] = -PetscMin(bsx,bsy);
  } else if (ixid == IS_BLOCK  && iyid == IS_STRIDE && iystep==1 && iyfirst%bsx==0) {
    m[0] = bsx;
    m[1] = -bsx;
  } else if (ixid == IS_STRIDE && iyid == IS_BLOCK  && ixstep==1 && ixfirst%bsy==0) {
    m[0] = bsy;
    m[1] = -bsy;
  }
  /* Get max and min of bsx,bsy over all processes in one allreduce */
  CHKERRMPI(MPIU_Allreduce(MPI_IN_PLACE,m,2,MPIU_INT,MPI_MAX,bigcomm));
  max = m[0]; min = -m[1];

  /* Since we used allreduce above, all ranks will have the same min and max. min==max
     implies all ranks have the same bs. Do further test to see if local vectors are dividable
     by bs on ALL ranks. If they are, we are ensured that no blocks span more than one processor.
   */
  if (min == max && min > 1) {
    CHKERRQ(VecGetLocalSize(x,&xlen));
    CHKERRQ(VecGetLocalSize(y,&ylen));
    m[0] = xlen%min;
    m[1] = ylen%min;
    CHKERRMPI(MPIU_Allreduce(MPI_IN_PLACE,m,2,MPIU_INT,MPI_LOR,bigcomm));
    if (!m[0] && !m[1]) can_do_block_opt = PETSC_TRUE;
  }

  /* If can_do_block_opt, then shrink x, y, ix and iy by bs to get xx, yy, ixx and iyy, whose indices
     and layout are actually used in building SF. Suppose blocked ix representing {0,1,2,6,7,8} has
     indices {0,2} and bs=3, then ixx = {0,2}; suppose strided iy={3,4,5,6,7,8}, then iyy={1,2}.

     xx is a little special. If x is seq, then xx is the concatenation of seq x's on ycomm. In this way,
     we can treat PtoP and StoP uniformly as PtoS.
   */
  if (can_do_block_opt) {
    const PetscInt *indices;

    /* Shrink x and ix */
    bs   = min;
    CHKERRQ(VecCreateMPIWithArray(bigcomm,1,xlen/bs,PETSC_DECIDE,NULL,&xx)); /* We only care xx's layout */
    if (ixid == IS_BLOCK) {
      CHKERRQ(ISBlockGetIndices(ix,&indices));
      CHKERRQ(ISBlockGetLocalSize(ix,&ixsize));
      CHKERRQ(ISCreateGeneral(PETSC_COMM_SELF,ixsize,indices,PETSC_COPY_VALUES,&ixx));
      CHKERRQ(ISBlockRestoreIndices(ix,&indices));
    } else { /* ixid == IS_STRIDE */
      CHKERRQ(ISGetLocalSize(ix,&ixsize));
      CHKERRQ(ISCreateStride(PETSC_COMM_SELF,ixsize/bs,ixfirst/bs,1,&ixx));
    }

    /* Shrink y and iy */
    CHKERRQ(VecCreateMPIWithArray(ycomm,1,ylen/bs,PETSC_DECIDE,NULL,&yy));
    if (iyid == IS_BLOCK) {
      CHKERRQ(ISBlockGetIndices(iy,&indices));
      CHKERRQ(ISBlockGetLocalSize(iy,&iysize));
      CHKERRQ(ISCreateGeneral(PETSC_COMM_SELF,iysize,indices,PETSC_COPY_VALUES,&iyy));
      CHKERRQ(ISBlockRestoreIndices(iy,&indices));
    } else { /* iyid == IS_STRIDE */
      CHKERRQ(ISGetLocalSize(iy,&iysize));
      CHKERRQ(ISCreateStride(PETSC_COMM_SELF,iysize/bs,iyfirst/bs,1,&iyy));
    }
  } else {
    ixx = ix;
    iyy = iy;
    yy  = y;
    if (xcommsize == 1) CHKERRQ(VecCreateMPIWithArray(bigcomm,1,xlen,PETSC_DECIDE,NULL,&xx)); else xx = x;
  }

  /* Now it is ready to build SF with preprocessed (xx, yy) and (ixx, iyy) */
  CHKERRQ(ISGetIndices(ixx,&xindices));
  CHKERRQ(ISGetIndices(iyy,&yindices));
  CHKERRQ(VecGetLayout(xx,&xlayout));

  if (ycommsize > 1) {
    /* PtoP or StoP */

    /* Below is a piece of complex code with a very simple goal: move global index pairs (xindices[i], yindices[i]),
       to owner process of yindices[i] according to ylayout, i = 0..n.

       I did it through a temp sf, but later I thought the old design was inefficient and also distorted log view.
       We want to mape one VecScatterCreate() call to one PetscSFCreate() call. The old design mapped to three
       PetscSFCreate() calls. This code is on critical path of VecScatterSetUp and is used by every VecScatterCreate.
       So I commented it out and did another optimized implementation. The commented code is left here for reference.
     */
#if 0
    const PetscInt *degree;
    PetscSF        tmpsf;
    PetscInt       inedges=0,*leafdata,*rootdata;

    CHKERRQ(VecGetOwnershipRange(xx,&xstart,NULL));
    CHKERRQ(VecGetLayout(yy,&ylayout));
    CHKERRQ(VecGetOwnershipRange(yy,&ystart,NULL));

    CHKERRQ(VecGetLocalSize(yy,&nroots));
    CHKERRQ(ISGetLocalSize(iyy,&nleaves));
    CHKERRQ(PetscMalloc2(nleaves,&iremote,nleaves*2,&leafdata));

    for (i=0; i<nleaves; i++) {
      CHKERRQ(PetscLayoutFindOwnerIndex(ylayout,yindices[i],&iremote[i].rank,&iremote[i].index));
      leafdata[2*i]   = yindices[i];
      leafdata[2*i+1] = (xcommsize > 1)? xindices[i] : xindices[i] + xstart;
    }

    CHKERRQ(PetscSFCreate(ycomm,&tmpsf));
    CHKERRQ(PetscSFSetGraph(tmpsf,nroots,nleaves,NULL,PETSC_USE_POINTER,iremote,PETSC_USE_POINTER));

    CHKERRQ(PetscSFComputeDegreeBegin(tmpsf,&degree));
    CHKERRQ(PetscSFComputeDegreeEnd(tmpsf,&degree));

    for (i=0; i<nroots; i++) inedges += degree[i];
    CHKERRQ(PetscMalloc1(inedges*2,&rootdata));
    CHKERRQ(PetscSFGatherBegin(tmpsf,MPIU_2INT,leafdata,rootdata));
    CHKERRQ(PetscSFGatherEnd(tmpsf,MPIU_2INT,leafdata,rootdata));

    CHKERRQ(PetscFree2(iremote,leafdata));
    CHKERRQ(PetscSFDestroy(&tmpsf));

    /* rootdata contains global index pairs (i, j). j's are owned by the current process, but i's can point to anywhere.
       We convert j to local, and convert i to (rank, index). In the end, we get an PtoS suitable for building SF.
     */
    nleaves = inedges;
    CHKERRQ(VecGetLocalSize(xx,&nroots));
    CHKERRQ(PetscMalloc1(nleaves,&ilocal));
    CHKERRQ(PetscMalloc1(nleaves,&iremote));

    for (i=0; i<inedges; i++) {
      ilocal[i] = rootdata[2*i] - ystart; /* covert y's global index to local index */
      CHKERRQ(PetscLayoutFindOwnerIndex(xlayout,rootdata[2*i+1],&iremote[i].rank,&iremote[i].index)); /* convert x's global index to (rank, index) */
    }
    CHKERRQ(PetscFree(rootdata));
#else
    PetscInt       j,k,n,disp,rlentotal,*sstart,*xindices_sorted,*yindices_sorted;
    const PetscInt *yrange;
    PetscMPIInt    nsend,nrecv,nreq,yrank,*sendto,*recvfrom,tag1,tag2;
    PetscInt       *slens,*rlens,count;
    PetscInt       *rxindices,*ryindices;
    MPI_Request    *reqs,*sreqs,*rreqs;

    /* Sorting makes code simpler, faster and also helps getting rid of many O(P) arrays, which hurt scalability at large scale
       yindices_sorted - sorted yindices
       xindices_sorted - xindices sorted along with yindces
     */
    CHKERRQ(ISGetLocalSize(ixx,&n)); /*ixx, iyy have the same local size */
    CHKERRQ(PetscMalloc2(n,&xindices_sorted,n,&yindices_sorted));
    CHKERRQ(PetscArraycpy(xindices_sorted,xindices,n));
    CHKERRQ(PetscArraycpy(yindices_sorted,yindices,n));
    CHKERRQ(PetscSortIntWithArray(n,yindices_sorted,xindices_sorted));
    CHKERRQ(VecGetOwnershipRange(xx,&xstart,NULL));
    if (xcommsize == 1) {for (i=0; i<n; i++) xindices_sorted[i] += xstart;} /* Convert to global indices */

    /*=============================================================================
             Calculate info about messages I need to send
      =============================================================================*/
    /* nsend    - number of non-empty messages to send
       sendto   - [nsend] ranks I will send messages to
       sstart   - [nsend+1] sstart[i] is the start index in xsindices_sorted[] I send to rank sendto[i]
       slens    - [ycommsize] I want to send slens[i] entries to rank i.
     */
    CHKERRQ(VecGetLayout(yy,&ylayout));
    CHKERRQ(PetscLayoutGetRanges(ylayout,&yrange));
    CHKERRQ(PetscCalloc1(ycommsize,&slens)); /* The only O(P) array in this algorithm */

    i = j = nsend = 0;
    while (i < n) {
      if (yindices_sorted[i] >= yrange[j+1]) { /* If i-th index is out of rank j's bound */
        do {j++;} while (yindices_sorted[i] >= yrange[j+1] && j < ycommsize); /* Increase j until i-th index falls in rank j's bound */
        PetscCheckFalse(j == ycommsize,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Index %" PetscInt_FMT " not owned by any process, upper bound %" PetscInt_FMT,yindices_sorted[i],yrange[ycommsize]);
      }
      i++;
      if (!slens[j]++) nsend++;
    }

    CHKERRQ(PetscMalloc2(nsend+1,&sstart,nsend,&sendto));

    sstart[0] = 0;
    for (i=j=0; i<ycommsize; i++) {
      if (slens[i]) {
        sendto[j]   = (PetscMPIInt)i;
        sstart[j+1] = sstart[j] + slens[i];
        j++;
      }
    }

    /*=============================================================================
      Calculate the reverse info about messages I will recv
      =============================================================================*/
    /* nrecv     - number of messages I will recv
       recvfrom  - [nrecv] ranks I recv from
       rlens     - [nrecv] I will recv rlens[i] entries from rank recvfrom[i]
       rlentotal - sum of rlens[]
       rxindices - [rlentotal] recv buffer for xindices_sorted
       ryindices - [rlentotal] recv buffer for yindices_sorted
     */
    CHKERRQ(PetscGatherNumberOfMessages_Private(ycomm,NULL,slens,&nrecv));
    CHKERRQ(PetscGatherMessageLengths_Private(ycomm,nsend,nrecv,slens,&recvfrom,&rlens));
    CHKERRQ(PetscFree(slens)); /* Free the O(P) array ASAP */
    rlentotal = 0; for (i=0; i<nrecv; i++) rlentotal += rlens[i];

    /*=============================================================================
      Communicate with processors in recvfrom[] to populate rxindices and ryindices
      ============================================================================*/
    CHKERRQ(PetscCommGetNewTag(ycomm,&tag1));
    CHKERRQ(PetscCommGetNewTag(ycomm,&tag2));
    CHKERRQ(PetscMalloc2(rlentotal,&rxindices,rlentotal,&ryindices));
    CHKERRQ(PetscMPIIntCast((nsend+nrecv)*2,&nreq));
    CHKERRQ(PetscMalloc1(nreq,&reqs));
    sreqs = reqs;
    rreqs = reqs + nsend*2;

    for (i=disp=0; i<nrecv; i++) {
      count = rlens[i];
      CHKERRMPI(MPIU_Irecv(rxindices+disp,count,MPIU_INT,recvfrom[i],tag1,ycomm,rreqs+i));
      CHKERRMPI(MPIU_Irecv(ryindices+disp,count,MPIU_INT,recvfrom[i],tag2,ycomm,rreqs+nrecv+i));
      disp += rlens[i];
    }

    for (i=0; i<nsend; i++) {
      count = sstart[i+1]-sstart[i];
      CHKERRMPI(MPIU_Isend(xindices_sorted+sstart[i],count,MPIU_INT,sendto[i],tag1,ycomm,sreqs+i));
      CHKERRMPI(MPIU_Isend(yindices_sorted+sstart[i],count,MPIU_INT,sendto[i],tag2,ycomm,sreqs+nsend+i));
    }
    CHKERRMPI(MPI_Waitall(nreq,reqs,MPI_STATUS_IGNORE));

    /* Transform VecScatter into SF */
    nleaves = rlentotal;
    CHKERRQ(PetscMalloc1(nleaves,&ilocal));
    CHKERRQ(PetscMalloc1(nleaves,&iremote));
    CHKERRMPI(MPI_Comm_rank(ycomm,&yrank));
    for (i=disp=0; i<nrecv; i++) {
      for (j=0; j<rlens[i]; j++) {
        k               = disp + j; /* k-th index pair */
        ilocal[k]       = ryindices[k] - yrange[yrank]; /* Convert y's global index to local index */
        CHKERRQ(PetscLayoutFindOwnerIndex(xlayout,rxindices[k],&rank,&iremote[k].index)); /* Convert x's global index to (rank, index) */
        iremote[k].rank = rank;
      }
      disp += rlens[i];
    }

    CHKERRQ(PetscFree2(sstart,sendto));
    CHKERRQ(PetscFree(rlens));
    CHKERRQ(PetscFree(recvfrom));
    CHKERRQ(PetscFree(reqs));
    CHKERRQ(PetscFree2(rxindices,ryindices));
    CHKERRQ(PetscFree2(xindices_sorted,yindices_sorted));
#endif
  } else {
    /* PtoS or StoS */
    CHKERRQ(ISGetLocalSize(iyy,&nleaves));
    CHKERRQ(PetscMalloc1(nleaves,&ilocal));
    CHKERRQ(PetscMalloc1(nleaves,&iremote));
    CHKERRQ(PetscArraycpy(ilocal,yindices,nleaves));
    for (i=0; i<nleaves; i++) {
      CHKERRQ(PetscLayoutFindOwnerIndex(xlayout,xindices[i],&rank,&iremote[i].index));
      iremote[i].rank = rank;
    }
  }

  /* MUST build SF on xx's comm, which is not necessarily identical to yy's comm.
     In SF's view, xx contains the roots (i.e., the remote) and iremote[].rank are ranks in xx's comm.
     yy contains leaves, which are local and can be thought as part of PETSC_COMM_SELF. */
  CHKERRQ(PetscSFCreate(PetscObjectComm((PetscObject)xx),&sf));
  sf->allow_multi_leaves = PETSC_TRUE;
  CHKERRQ(PetscSFSetFromOptions(sf));
  CHKERRQ(VecGetLocalSize(xx,&nroots));
  CHKERRQ(PetscSFSetGraph(sf,nroots,nleaves,ilocal,PETSC_OWN_POINTER,iremote,PETSC_OWN_POINTER)); /* Give ilocal/iremote to petsc and no need to free them here */

  /* Free memory no longer needed */
  CHKERRQ(ISRestoreIndices(ixx,&xindices));
  CHKERRQ(ISRestoreIndices(iyy,&yindices));
  if (can_do_block_opt) {
    CHKERRQ(VecDestroy(&xx));
    CHKERRQ(VecDestroy(&yy));
    CHKERRQ(ISDestroy(&ixx));
    CHKERRQ(ISDestroy(&iyy));
  } else if (xcommsize == 1) {
    CHKERRQ(VecDestroy(&xx));
  }

functionend:
  sf->vscat.bs = bs;
  if (sf->vscat.bs > 1) {
    CHKERRMPI(MPI_Type_contiguous(sf->vscat.bs,MPIU_SCALAR,&sf->vscat.unit));
    CHKERRMPI(MPI_Type_commit(&sf->vscat.unit));
  } else {
    sf->vscat.unit = MPIU_SCALAR;
  }
  CHKERRQ(VecGetLocalSize(x,&sf->vscat.from_n));
  CHKERRQ(VecGetLocalSize(y,&sf->vscat.to_n));
  if (!ix_old) CHKERRQ(ISDestroy(&ix)); /* We created helper ix, iy. Free them */
  if (!iy_old) CHKERRQ(ISDestroy(&iy));

  /* Set default */
  CHKERRQ(VecScatterSetFromOptions(sf));

  *newsf = sf;
  PetscFunctionReturn(0);
}

/*@C
      VecScatterCreateToAll - Creates a vector and a scatter context that copies all
          vector values to each processor

  Collective on Vec

  Input Parameter:
.  vin  - input MPIVEC

  Output Parameters:
+  ctx - scatter context
-  vout - output SEQVEC that is large enough to scatter into

  Level: intermediate

   Note: vout may be NULL [PETSC_NULL_VEC from fortran] if you do not
   need to have it created

   Usage:
$        VecScatterCreateToAll(vin,&ctx,&vout);
$
$        // scatter as many times as you need
$        VecScatterBegin(ctx,vin,vout,INSERT_VALUES,SCATTER_FORWARD);
$        VecScatterEnd(ctx,vin,vout,INSERT_VALUES,SCATTER_FORWARD);
$
$        // destroy scatter context and local vector when no longer needed
$        VecScatterDestroy(&ctx);
$        VecDestroy(&vout);

    Do NOT create a vector and then pass it in as the final argument vout! vout is created by this routine
  automatically (unless you pass NULL in for that argument if you do not need it).

.seealso VecScatterCreate(), VecScatterCreateToZero(), VecScatterBegin(), VecScatterEnd()

@*/
PetscErrorCode  VecScatterCreateToAll(Vec vin,VecScatter *ctx,Vec *vout)
{
  PetscInt       N;
  IS             is;
  Vec            tmp;
  Vec            *tmpv;
  PetscBool      tmpvout = PETSC_FALSE;
  VecType        roottype;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(vin,VEC_CLASSID,1);
  PetscValidType(vin,1);
  PetscValidPointer(ctx,2);
  if (vout) {
    PetscValidPointer(vout,3);
    tmpv = vout;
  } else {
    tmpvout = PETSC_TRUE;
    tmpv    = &tmp;
  }

  /* Create seq vec on each proc, with the same size of the original vec */
  CHKERRQ(VecGetSize(vin,&N));
  CHKERRQ(VecGetRootType_Private(vin,&roottype));
  CHKERRQ(VecCreate(PETSC_COMM_SELF,tmpv));
  CHKERRQ(VecSetSizes(*tmpv,N,PETSC_DECIDE));
  CHKERRQ(VecSetType(*tmpv,roottype));
  /* Create the VecScatter ctx with the communication info */
  CHKERRQ(ISCreateStride(PETSC_COMM_SELF,N,0,1,&is));
  CHKERRQ(VecScatterCreate(vin,is,*tmpv,is,ctx));
  CHKERRQ(ISDestroy(&is));
  if (tmpvout) CHKERRQ(VecDestroy(tmpv));
  PetscFunctionReturn(0);
}

/*@C
      VecScatterCreateToZero - Creates an output vector and a scatter context used to
              copy all vector values into the output vector on the zeroth processor

  Collective on Vec

  Input Parameter:
.  vin  - input MPIVEC

  Output Parameters:
+  ctx - scatter context
-  vout - output SEQVEC that is large enough to scatter into on processor 0 and
          of length zero on all other processors

  Level: intermediate

   Note: vout may be NULL [PETSC_NULL_VEC from fortran] if you do not
   need to have it created

   Usage:
$        VecScatterCreateToZero(vin,&ctx,&vout);
$
$        // scatter as many times as you need
$        VecScatterBegin(ctx,vin,vout,INSERT_VALUES,SCATTER_FORWARD);
$        VecScatterEnd(ctx,vin,vout,INSERT_VALUES,SCATTER_FORWARD);
$
$        // destroy scatter context and local vector when no longer needed
$        VecScatterDestroy(&ctx);
$        VecDestroy(&vout);

.seealso VecScatterCreate(), VecScatterCreateToAll(), VecScatterBegin(), VecScatterEnd()

    Do NOT create a vector and then pass it in as the final argument vout! vout is created by this routine
  automatically (unless you pass NULL in for that argument if you do not need it).

@*/
PetscErrorCode  VecScatterCreateToZero(Vec vin,VecScatter *ctx,Vec *vout)
{

  PetscInt       N;
  PetscMPIInt    rank;
  IS             is;
  Vec            tmp;
  Vec            *tmpv;
  PetscBool      tmpvout = PETSC_FALSE;
  VecType        roottype;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(vin,VEC_CLASSID,1);
  PetscValidType(vin,1);
  PetscValidPointer(ctx,2);
  if (vout) {
    PetscValidPointer(vout,3);
    tmpv = vout;
  } else {
    tmpvout = PETSC_TRUE;
    tmpv    = &tmp;
  }

  /* Create vec on each proc, with the same size of the original vec all on process 0 */
  CHKERRQ(VecGetSize(vin,&N));
  CHKERRMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)vin),&rank));
  if (rank) N = 0;
  CHKERRQ(VecGetRootType_Private(vin,&roottype));
  CHKERRQ(VecCreate(PETSC_COMM_SELF,tmpv));
  CHKERRQ(VecSetSizes(*tmpv,N,PETSC_DECIDE));
  CHKERRQ(VecSetType(*tmpv,roottype));
  /* Create the VecScatter ctx with the communication info */
  CHKERRQ(ISCreateStride(PETSC_COMM_SELF,N,0,1,&is));
  CHKERRQ(VecScatterCreate(vin,is,*tmpv,is,ctx));
  CHKERRQ(ISDestroy(&is));
  if (tmpvout) CHKERRQ(VecDestroy(tmpv));
  PetscFunctionReturn(0);
}

/*@
   VecScatterBegin - Begins a generalized scatter from one vector to
   another. Complete the scattering phase with VecScatterEnd().

   Neighbor-wise Collective on VecScatter

   Input Parameters:
+  sf - scatter context generated by VecScatterCreate()
.  x - the vector from which we scatter
.  y - the vector to which we scatter
.  addv - either ADD_VALUES, MAX_VALUES, MIN_VALUES or INSERT_VALUES, with INSERT_VALUES mode any location
          not scattered to retains its old value; i.e. the vector is NOT first zeroed.
-  mode - the scattering mode, usually SCATTER_FORWARD.  The available modes are:
    SCATTER_FORWARD or SCATTER_REVERSE

   Level: intermediate

   Options Database: See VecScatterCreate()

   Notes:
   The vectors x and y need not be the same vectors used in the call
   to VecScatterCreate(), but x must have the same parallel data layout
   as that passed in as the x to VecScatterCreate(), similarly for the y.
   Most likely they have been obtained from VecDuplicate().

   You cannot change the values in the input vector between the calls to VecScatterBegin()
   and VecScatterEnd().

   If you use SCATTER_REVERSE the two arguments x and y should be reversed, from
   the SCATTER_FORWARD.

   y[iy[i]] = x[ix[i]], for i=0,...,ni-1

   This scatter is far more general than the conventional
   scatter, since it can be a gather or a scatter or a combination,
   depending on the indices ix and iy.  If x is a parallel vector and y
   is sequential, VecScatterBegin() can serve to gather values to a
   single processor.  Similarly, if y is parallel and x sequential, the
   routine can scatter from one processor to many processors.

.seealso: VecScatterCreate(), VecScatterEnd()
@*/
PetscErrorCode  VecScatterBegin(VecScatter sf,Vec x,Vec y,InsertMode addv,ScatterMode mode)
{
  PetscInt       to_n,from_n;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sf,PETSCSF_CLASSID,1);
  PetscValidHeaderSpecific(x,VEC_CLASSID,2);
  PetscValidHeaderSpecific(y,VEC_CLASSID,3);
  if (PetscDefined(USE_DEBUG)) {
    /*
     Error checking to make sure these vectors match the vectors used
     to create the vector scatter context. -1 in the from_n and to_n indicate the
     vector lengths are unknown (for example with mapped scatters) and thus
     no error checking is performed.
     */
    if (sf->vscat.from_n >= 0 && sf->vscat.to_n >= 0) {
      CHKERRQ(VecGetLocalSize(x,&from_n));
      CHKERRQ(VecGetLocalSize(y,&to_n));
      if (mode & SCATTER_REVERSE) {
        PetscCheckFalse(to_n != sf->vscat.from_n,PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Vector wrong size %" PetscInt_FMT " for scatter %" PetscInt_FMT " (scatter reverse and vector to != sf from size)",to_n,sf->vscat.from_n);
        PetscCheckFalse(from_n != sf->vscat.to_n,PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Vector wrong size %" PetscInt_FMT " for scatter %" PetscInt_FMT " (scatter reverse and vector from != sf to size)",from_n,sf->vscat.to_n);
      } else {
        PetscCheckFalse(to_n != sf->vscat.to_n,PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Vector wrong size %" PetscInt_FMT " for scatter %" PetscInt_FMT " (scatter forward and vector to != sf to size)",to_n,sf->vscat.to_n);
        PetscCheckFalse(from_n != sf->vscat.from_n,PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Vector wrong size %" PetscInt_FMT " for scatter %" PetscInt_FMT " (scatter forward and vector from != sf from size)",from_n,sf->vscat.from_n);
      }
    }
  }

  sf->vscat.logging = PETSC_TRUE;
  CHKERRQ(PetscLogEventBegin(VEC_ScatterBegin,sf,x,y,0));
  CHKERRQ(VecScatterBegin_Internal(sf,x,y,addv,mode));
  if (sf->vscat.beginandendtogether) {
    CHKERRQ(VecScatterEnd_Internal(sf,x,y,addv,mode));
  }
  CHKERRQ(PetscLogEventEnd(VEC_ScatterBegin,sf,x,y,0));
  sf->vscat.logging = PETSC_FALSE;
  PetscFunctionReturn(0);
}

/*@
   VecScatterEnd - Ends a generalized scatter from one vector to another.  Call
   after first calling VecScatterBegin().

   Neighbor-wise Collective on VecScatter

   Input Parameters:
+  sf - scatter context generated by VecScatterCreate()
.  x - the vector from which we scatter
.  y - the vector to which we scatter
.  addv - one of ADD_VALUES, MAX_VALUES, MIN_VALUES or INSERT_VALUES
-  mode - the scattering mode, usually SCATTER_FORWARD.  The available modes are:
     SCATTER_FORWARD, SCATTER_REVERSE

   Level: intermediate

   Notes:
   If you use SCATTER_REVERSE the arguments x and y should be reversed, from the SCATTER_FORWARD.

   y[iy[i]] = x[ix[i]], for i=0,...,ni-1

.seealso: VecScatterBegin(), VecScatterCreate()
@*/
PetscErrorCode  VecScatterEnd(VecScatter sf,Vec x,Vec y,InsertMode addv,ScatterMode mode)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(sf,PETSCSF_CLASSID,1);
  PetscValidHeaderSpecific(x,VEC_CLASSID,2);
  PetscValidHeaderSpecific(y,VEC_CLASSID,3);
  if (!sf->vscat.beginandendtogether) {
    sf->vscat.logging = PETSC_TRUE;
    CHKERRQ(PetscLogEventBegin(VEC_ScatterEnd,sf,x,y,0));
    CHKERRQ(VecScatterEnd_Internal(sf,x,y,addv,mode));
    CHKERRQ(PetscLogEventEnd(VEC_ScatterEnd,sf,x,y,0));
    sf->vscat.logging = PETSC_FALSE;
  }
  PetscFunctionReturn(0);
}
