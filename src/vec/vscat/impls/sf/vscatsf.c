#include <petsc/private/vecscatterimpl.h>    /*I   "petscvec.h"    I*/
#include <petsc/private/sfimpl.h> /*I "petscsf.h" I*/
#include <../src/vec/is/sf/impls/basic/sfbasic.h> /* for VecScatterRemap_SF */
#include <../src/vec/is/sf/impls/basic/sfpack.h>

#if defined(PETSC_HAVE_CUDA)
#include <petsc/private/cudavecimpl.h>
#endif

typedef struct {
  PetscSF           sf;     /* the whole scatter, including local and remote */
  PetscSF           lsf;    /* the local part of the scatter, used for SCATTER_LOCAL */
  PetscInt          bs;     /* block size */
  MPI_Datatype      unit;   /* one unit = bs PetscScalars */
} VecScatter_SF;

static PetscErrorCode VecScatterBegin_SF(VecScatter vscat,Vec x,Vec y,InsertMode addv,ScatterMode mode)
{
  VecScatter_SF  *data=(VecScatter_SF*)vscat->data;
  PetscSF        sf=data->sf;
  MPI_Op         mop=MPI_OP_NULL;
  PetscMPIInt    size;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (x != y) {ierr = VecLockReadPush(x);CHKERRQ(ierr);}

  if (sf->use_gpu_aware_mpi || vscat->packongpu) {
    ierr = VecGetArrayReadInPlace(x,&vscat->xdata);CHKERRQ(ierr);
  } else {
#if defined(PETSC_HAVE_CUDA)
    PetscBool is_cudatype = PETSC_FALSE;
    ierr = PetscObjectTypeCompareAny((PetscObject)x,&is_cudatype,VECSEQCUDA,VECMPICUDA,VECCUDA,"");CHKERRQ(ierr);
    if (is_cudatype) {
      VecCUDAAllocateCheckHost(x);
      if (x->offloadmask == PETSC_OFFLOAD_GPU) {
        if (x->spptr && vscat->spptr) {ierr = VecCUDACopyFromGPUSome_Public(x,(PetscCUDAIndices)vscat->spptr,mode);CHKERRQ(ierr);}
        else {ierr = VecCUDACopyFromGPU(x);CHKERRQ(ierr);}
      }
      vscat->xdata = *((PetscScalar**)x->data);
    } else
#endif
    {ierr = VecGetArrayRead(x,&vscat->xdata);CHKERRQ(ierr);}
  }

  if (x != y) {
    if (sf->use_gpu_aware_mpi || vscat->packongpu) {ierr = VecGetArrayInPlace(y,&vscat->ydata);CHKERRQ(ierr);}
    else {ierr = VecGetArray(y,&vscat->ydata);CHKERRQ(ierr);}
  } else vscat->ydata = (PetscScalar *)vscat->xdata;
  ierr = VecLockWriteSet_Private(y,PETSC_TRUE);CHKERRQ(ierr);

  /* SCATTER_LOCAL indicates ignoring inter-process communication */
  ierr = MPI_Comm_size(PetscObjectComm((PetscObject)data->sf),&size);CHKERRQ(ierr);
  if ((mode & SCATTER_LOCAL) && size > 1) { /* Lazy creation of data->lsf since SCATTER_LOCAL is uncommon */
    if (!data->lsf) {ierr = PetscSFCreateLocalSF_Private(data->sf,&data->lsf);CHKERRQ(ierr);}
    sf = data->lsf;
  } else {
    sf = data->sf;
  }

  if (addv == INSERT_VALUES)   mop = MPIU_REPLACE;
  else if (addv == ADD_VALUES) mop = MPIU_SUM; /* Petsc defines its own MPI datatype and SUM operation for __float128 etc. */
  else if (addv == MAX_VALUES) mop = MPIU_MAX;
  else if (addv == MIN_VALUES) mop = MPIU_MIN;
  else SETERRQ1(PetscObjectComm((PetscObject)sf),PETSC_ERR_SUP,"Unsupported InsertMode %D in VecScatterBegin/End",addv);

  if (mode & SCATTER_REVERSE) { /* REVERSE indicates leaves to root scatter. Note that x and y are swapped in input */
    ierr = PetscSFReduceBegin(sf,data->unit,vscat->xdata,vscat->ydata,mop);CHKERRQ(ierr);
  } else { /* FORWARD indicates x to y scatter, where x is root and y is leaf */
    ierr = PetscSFBcastAndOpBegin(sf,data->unit,vscat->xdata,vscat->ydata,mop);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode VecScatterEnd_SF(VecScatter vscat,Vec x,Vec y,InsertMode addv,ScatterMode mode)
{
  VecScatter_SF  *data=(VecScatter_SF*)vscat->data;
  PetscSF        sf=data->sf;
  MPI_Op         mop=MPI_OP_NULL;
  PetscMPIInt    size;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* SCATTER_LOCAL indicates ignoring inter-process communication */
  ierr = MPI_Comm_size(PetscObjectComm((PetscObject)data->sf),&size);CHKERRQ(ierr);
  sf = ((mode & SCATTER_LOCAL) && size > 1) ? data->lsf : data->sf;

  if (addv == INSERT_VALUES)   mop = MPIU_REPLACE;
  else if (addv == ADD_VALUES) mop = MPIU_SUM;
  else if (addv == MAX_VALUES) mop = MPIU_MAX;
  else if (addv == MIN_VALUES) mop = MPIU_MIN;
  else SETERRQ1(PetscObjectComm((PetscObject)sf),PETSC_ERR_SUP,"Unsupported InsertMode %D in VecScatterBegin/End",addv);

  if (mode & SCATTER_REVERSE) { /* reverse scatter sends leaves to roots. Note that x and y are swapped in input */
    ierr = PetscSFReduceEnd(sf,data->unit,vscat->xdata,vscat->ydata,mop);CHKERRQ(ierr);
  } else { /* forward scatter sends roots to leaves, i.e., x to y */
    ierr = PetscSFBcastAndOpEnd(sf,data->unit,vscat->xdata,vscat->ydata,mop);CHKERRQ(ierr);
  }

  if (x != y) {
    if (sf->use_gpu_aware_mpi || vscat->packongpu) {ierr = VecRestoreArrayReadInPlace(x,&vscat->xdata);CHKERRQ(ierr);}
    else {ierr = VecRestoreArrayRead(x,&vscat->xdata);CHKERRQ(ierr);}
    ierr = VecLockReadPop(x);CHKERRQ(ierr);
  }

  if (sf->use_gpu_aware_mpi || vscat->packongpu) {ierr = VecRestoreArrayInPlace(y,&vscat->ydata);CHKERRQ(ierr);}
  else {ierr = VecRestoreArray(y,&vscat->ydata);CHKERRQ(ierr);}
  ierr = VecLockWriteSet_Private(y,PETSC_FALSE);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

static PetscErrorCode VecScatterCopy_SF(VecScatter vscat,VecScatter ctx)
{
  VecScatter_SF  *data=(VecScatter_SF*)vscat->data,*out;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscMemcpy(ctx->ops,vscat->ops,sizeof(vscat->ops));CHKERRQ(ierr);
  ierr = PetscNewLog(ctx,&out);CHKERRQ(ierr);
  ierr = PetscSFDuplicate(data->sf,PETSCSF_DUPLICATE_GRAPH,&out->sf);CHKERRQ(ierr);
  ierr = PetscSFSetUp(out->sf);CHKERRQ(ierr);
  /* Do not copy lsf. Build it on demand since it is rarely used */

  out->bs = data->bs;
  if (out->bs > 1) {
    ierr = MPI_Type_dup(data->unit,&out->unit);CHKERRQ(ierr); /* Since oldtype is committed, so is newtype, according to MPI */
  } else {
    out->unit = MPIU_SCALAR;
  }
  ctx->data = (void*)out;
  PetscFunctionReturn(0);
}

static PetscErrorCode VecScatterDestroy_SF(VecScatter vscat)
{
  VecScatter_SF *data = (VecScatter_SF *)vscat->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscSFDestroy(&data->sf);CHKERRQ(ierr);
  ierr = PetscSFDestroy(&data->lsf);CHKERRQ(ierr);
  if (data->bs > 1) {ierr = MPI_Type_free(&data->unit);CHKERRQ(ierr);}
  ierr = PetscFree(vscat->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode VecScatterView_SF(VecScatter vscat,PetscViewer viewer)
{
  VecScatter_SF *data = (VecScatter_SF *)vscat->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscSFView(data->sf,viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* VecScatterRemap provides a light way to slightly modify a VecScatter. Suppose the input vscat scatters
   x[i] to y[j], tomap gives a plan to change vscat to scatter x[tomap[i]] to y[j]. Note that in SF,
   x is roots. That means we need to change incoming stuffs such as bas->irootloc[].
 */
static PetscErrorCode VecScatterRemap_SF(VecScatter vscat,const PetscInt *tomap,const PetscInt *frommap)
{
  VecScatter_SF  *data = (VecScatter_SF *)vscat->data;
  PetscSF        sf = data->sf;
  PetscInt       i,bs = data->bs;
  PetscMPIInt    size;
  PetscBool      ident = PETSC_TRUE,isbasic,isneighbor;
  PetscSFType    type;
  PetscSF_Basic  *bas = NULL;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* check if it is an identity map. If it is, do nothing */
  if (tomap) {
    for (i=0; i<sf->nroots*bs; i++) {if (i != tomap[i]) {ident = PETSC_FALSE; break; } }
    if (ident) PetscFunctionReturn(0);
  }
  if (frommap) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Unable to remap the FROM in scatters yet");
  if (!tomap) PetscFunctionReturn(0);

  ierr = MPI_Comm_size(PetscObjectComm((PetscObject)data->sf),&size);CHKERRQ(ierr);

  /* Since the indices changed, we must also update the local SF. But we do not do it since
     lsf is rarely used. We just destroy lsf and rebuild it on demand from updated data->sf.
  */
  if (data->lsf) {ierr = PetscSFDestroy(&data->lsf);CHKERRQ(ierr);}

  ierr = PetscSFGetType(sf,&type);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)sf,PETSCSFBASIC,&isbasic);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)sf,PETSCSFNEIGHBOR,&isneighbor);CHKERRQ(ierr);
  if (!isbasic && !isneighbor) SETERRQ1(PetscObjectComm((PetscObject)sf),PETSC_ERR_SUP,"VecScatterRemap on SF type %s is not supported",type);CHKERRQ(ierr);

  ierr = PetscSFSetUp(sf);CHKERRQ(ierr); /* to bulid sf->irootloc if SetUp is not yet called */

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
  ierr       = PetscFree(sf->remote_alloc);CHKERRQ(ierr);
  /* Not easy to free sf->rremote since it was allocated with PetscMalloc4(), so just give it crazy values */
  for (i=0; i<sf->roffset[sf->nranks]; i++) sf->rremote[i] = PETSC_MIN_INT;

  /* Indices in tomap[] are for each indivisual vector entry. But indices in sf are for each
     block in the vector. So before the remapping, we have to expand indices in sf by bs, and
     after the remapping, we have to shrink them back.
   */
  bas = (PetscSF_Basic*)sf->data;
  for (i=0; i<bas->ioffset[bas->niranks]; i++) bas->irootloc[i] = tomap[bas->irootloc[i]*bs]/bs;
#if defined(PETSC_HAVE_CUDA)
  /* Free the irootloc copy on device. We allocate a new copy and get the updated value on demand. See PetscSFLinkGetRootPackOptAndIndices() */
  for (i=0; i<2; i++) {if (bas->irootloc_d[i]) {cudaError_t err = cudaFree(bas->irootloc_d[i]);CHKERRCUDA(err);bas->irootloc_d[i]=NULL;}}
#endif
  /* Destroy and then rebuild root packing optimizations since indices are changed */
  ierr = PetscSFResetPackFields(sf);CHKERRQ(ierr);
  ierr = PetscSFSetUpPackFields(sf);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode VecScatterGetRemoteCount_SF(VecScatter vscat,PetscBool send,PetscInt *num_procs,PetscInt *num_entries)
{
  PetscErrorCode    ierr;
  VecScatter_SF     *data = (VecScatter_SF *)vscat->data;
  PetscSF           sf = data->sf;
  PetscInt          nranks,remote_start;
  PetscMPIInt       rank;
  const PetscInt    *offset;
  const PetscMPIInt *ranks;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(PetscObjectComm((PetscObject)sf),&rank);CHKERRQ(ierr);

  /* This routine is mainly used for MatMult's Mvctx. In Mvctx, we scatter an MPI vector x to a sequential vector lvec.
     Remember x is roots and lvec is leaves. 'send' means roots to leaves communication. If 'send' is true, we need to
     get info about which ranks this processor needs to send to. In other words, we need to call PetscSFGetLeafRanks().
     If send is false, we do the opposite, calling PetscSFGetRootRanks().
  */
  if (send) {ierr = PetscSFGetLeafRanks(sf,&nranks,&ranks,&offset,NULL);CHKERRQ(ierr);}
  else {ierr = PetscSFGetRootRanks(sf,&nranks,&ranks,&offset,NULL,NULL);CHKERRQ(ierr);}
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

static PetscErrorCode VecScatterGetRemote_SF(VecScatter vscat,PetscBool send,PetscInt *n,const PetscInt **starts,const PetscInt **indices,const PetscMPIInt **procs,PetscInt *bs)
{
  PetscErrorCode    ierr;
  VecScatter_SF     *data = (VecScatter_SF *)vscat->data;
  PetscSF           sf = data->sf;
  PetscInt          nranks,remote_start;
  PetscMPIInt       rank;
  const PetscInt    *offset,*location;
  const PetscMPIInt *ranks;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(PetscObjectComm((PetscObject)sf),&rank);CHKERRQ(ierr);

  if (send) {ierr = PetscSFGetLeafRanks(sf,&nranks,&ranks,&offset,&location);CHKERRQ(ierr);}
  else {ierr = PetscSFGetRootRanks(sf,&nranks,&ranks,&offset,&location,NULL);CHKERRQ(ierr);}

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

static PetscErrorCode VecScatterGetRemoteOrdered_SF(VecScatter vscat,PetscBool send,PetscInt *n,const PetscInt **starts,const PetscInt **indices,const PetscMPIInt **procs,PetscInt *bs)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecScatterGetRemote_SF(vscat,send,n,starts,indices,procs,bs);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode VecScatterRestoreRemote_SF(VecScatter vscat,PetscBool send,PetscInt *n,const PetscInt **starts,const PetscInt **indices,const PetscMPIInt **procs,PetscInt *bs)
{
  PetscFunctionBegin;
  if (starts)   *starts  = NULL;
  if (indices)  *indices = NULL;
  if (procs)    *procs   = NULL;
  PetscFunctionReturn(0);
}

static PetscErrorCode VecScatterRestoreRemoteOrdered_SF(VecScatter vscat,PetscBool send,PetscInt *n,const PetscInt **starts,const PetscInt **indices,const PetscMPIInt **procs,PetscInt *bs)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = VecScatterRestoreRemote_SF(vscat,send,n,starts,indices,procs,bs);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

typedef enum {IS_INVALID, IS_GENERAL, IS_BLOCK, IS_STRIDE} ISTypeID;

PETSC_STATIC_INLINE PetscErrorCode ISGetTypeID_Private(IS is,ISTypeID *id)
{
  PetscErrorCode ierr;
  PetscBool      same;

  PetscFunctionBegin;
  *id  = IS_INVALID;
  ierr = PetscObjectTypeCompare((PetscObject)is,ISGENERAL,&same);CHKERRQ(ierr);
  if (same) {*id = IS_GENERAL; goto functionend;}
  ierr = PetscObjectTypeCompare((PetscObject)is,ISBLOCK,&same);CHKERRQ(ierr);
  if (same) {*id = IS_BLOCK; goto functionend;}
  ierr = PetscObjectTypeCompare((PetscObject)is,ISSTRIDE,&same);CHKERRQ(ierr);
  if (same) {*id = IS_STRIDE; goto functionend;}
functionend:
  PetscFunctionReturn(0);
}

static PetscErrorCode VecScatterSetUp_SF(VecScatter vscat)
{
  PetscErrorCode ierr;
  VecScatter_SF  *data;
  MPI_Comm       xcomm,ycomm,bigcomm;
  Vec            x=vscat->from_v,y=vscat->to_v,xx,yy;
  IS             ix=vscat->from_is,iy=vscat->to_is,ixx,iyy;
  PetscMPIInt    xcommsize,ycommsize,rank;
  PetscInt       i,n,N,nroots,nleaves,*ilocal,xstart,ystart,ixsize,iysize,xlen,ylen;
  const PetscInt *xindices,*yindices;
  PetscSFNode    *iremote;
  PetscLayout    xlayout,ylayout;
  ISTypeID       ixid,iyid;
  PetscInt       bs,bsx,bsy,min,max,m[2],ixfirst,ixstep,iyfirst,iystep;
  PetscBool      can_do_block_opt=PETSC_FALSE;

  PetscFunctionBegin;
  ierr = PetscNewLog(vscat,&data);CHKERRQ(ierr);
  data->bs   = 1; /* default, no blocking */
  data->unit = MPIU_SCALAR;

  /*
   Let P and S stand for parallel and sequential vectors respectively. There are four combinations of vecscatters: PtoP, PtoS,
   StoP and StoS. The assumption of VecScatterCreate(Vec x,IS ix,Vec y,IS iy,VecScatter *newctx) is: if x is parallel, then ix
   contains global indices of x. If x is sequential, ix contains local indices of x. Similarily for y and iy.

   SF builds around concepts of local leaves and remote roots. We treat source vector x as roots and destination vector y as
   leaves. A PtoS scatter can be naturally mapped to SF. We transform PtoP and StoP to PtoS, and treat StoS as trivial PtoS.
  */
  ierr = PetscObjectGetComm((PetscObject)x,&xcomm);CHKERRQ(ierr);
  ierr = PetscObjectGetComm((PetscObject)y,&ycomm);CHKERRQ(ierr);
  ierr = MPI_Comm_size(xcomm,&xcommsize);CHKERRQ(ierr);
  ierr = MPI_Comm_size(ycomm,&ycommsize);CHKERRQ(ierr);

  /* NULL ix or iy in VecScatterCreate(x,ix,y,iy,newctx) has special meaning. Recover them for these cases */
  if (!ix) {
    if (xcommsize > 1 && ycommsize == 1) { /* PtoS: null ix means the whole x will be scattered to each seq y */
      ierr = VecGetSize(x,&N);CHKERRQ(ierr);
      ierr = ISCreateStride(PETSC_COMM_SELF,N,0,1,&ix);CHKERRQ(ierr);
    } else { /* PtoP, StoP or StoS: null ix means the whole local part of x will be scattered */
      ierr = VecGetLocalSize(x,&n);CHKERRQ(ierr);
      ierr = VecGetOwnershipRange(x,&xstart,NULL);CHKERRQ(ierr);
      ierr = ISCreateStride(PETSC_COMM_SELF,n,xstart,1,&ix);CHKERRQ(ierr);
    }
  }

  if (!iy) {
    if (xcommsize == 1 && ycommsize > 1) { /* StoP: null iy means the whole y will be scattered to from each seq x */
      ierr = VecGetSize(y,&N);CHKERRQ(ierr);
      ierr = ISCreateStride(PETSC_COMM_SELF,N,0,1,&iy);CHKERRQ(ierr);
    } else { /* PtoP, StoP or StoS: null iy means the whole local part of y will be scattered to */
      ierr = VecGetLocalSize(y,&n);CHKERRQ(ierr);
      ierr = VecGetOwnershipRange(y,&ystart,NULL);CHKERRQ(ierr);
      ierr = ISCreateStride(PETSC_COMM_SELF,n,ystart,1,&iy);CHKERRQ(ierr);
    }
  }

  /* Do error checking immediately after we have non-empty ix, iy */
  ierr = ISGetLocalSize(ix,&ixsize);CHKERRQ(ierr);
  ierr = ISGetLocalSize(iy,&iysize);CHKERRQ(ierr);
  ierr = VecGetSize(x,&xlen);CHKERRQ(ierr);
  ierr = VecGetSize(y,&ylen);CHKERRQ(ierr);
  if (ixsize != iysize) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Scatter sizes of ix and iy don't match locally");
  ierr = ISGetMinMax(ix,&min,&max);CHKERRQ(ierr);
  if (min < 0 || max >= xlen) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Scatter indices in ix are out of range");
  ierr = ISGetMinMax(iy,&min,&max);CHKERRQ(ierr);
  if (min < 0 || max >= ylen) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Scatter indices in iy are out of range");

  /* Extract info about ix, iy for further test */
  ierr = ISGetTypeID_Private(ix,&ixid);CHKERRQ(ierr);
  ierr = ISGetTypeID_Private(iy,&iyid);CHKERRQ(ierr);
  if (ixid == IS_BLOCK)       {ierr = ISGetBlockSize(ix,&bsx);CHKERRQ(ierr);}
  else if (ixid == IS_STRIDE) {ierr = ISStrideGetInfo(ix,&ixfirst,&ixstep);CHKERRQ(ierr);}

  if ( iyid == IS_BLOCK)      {ierr = ISGetBlockSize(iy,&bsy);CHKERRQ(ierr);}
  else if (iyid == IS_STRIDE) {ierr = ISStrideGetInfo(iy,&iyfirst,&iystep);CHKERRQ(ierr);}

  /* Check if a PtoS is special ToAll/ToZero scatters, which can be results of VecScatterCreateToAll/Zero.
     ToAll means a whole MPI vector is copied to a seq vector on every process. ToZero means a whole MPI
     vector is copied to a seq vector on rank 0 and other processes do nothing(i.e.,they input empty ix,iy).

     We can optimize these scatters with MPI collectives. We can also avoid costly analysis used for general scatters.
  */
  if (xcommsize > 1 && ycommsize == 1) { /* Ranks do not diverge at this if-test */
    PetscInt    pattern[2] = {0, 0}; /* A boolean array with pattern[0] for allgather-like (ToAll) and pattern[1] for gather-like (ToZero) */
    PetscLayout map;

    ierr = MPI_Comm_rank(xcomm,&rank);CHKERRQ(ierr);
    ierr = VecGetLayout(x,&map);CHKERRQ(ierr);
    if (!rank) {
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
    ierr   = MPIU_Allreduce(MPI_IN_PLACE,pattern,2,MPIU_INT,MPI_LAND,xcomm);CHKERRQ(ierr);

    if (pattern[0] || pattern[1]) {
      ierr = PetscSFCreate(xcomm,&data->sf);CHKERRQ(ierr);
      ierr = PetscSFSetGraphWithPattern(data->sf,map,pattern[0] ? PETSCSF_PATTERN_ALLGATHER : PETSCSF_PATTERN_GATHER);CHKERRQ(ierr);
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
  ierr = MPIU_Allreduce(MPI_IN_PLACE,m,2,MPIU_INT,MPI_MAX,bigcomm);CHKERRQ(ierr);
  max = m[0]; min = -m[1];

  /* Since we used allreduce above, all ranks will have the same min and max. min==max
     implies all ranks have the same bs. Do further test to see if local vectors are dividable
     by bs on ALL ranks. If they are, we are ensured that no blocks span more than one processor.
   */
  if (min == max && min > 1) {
    ierr = VecGetLocalSize(x,&xlen);CHKERRQ(ierr);
    ierr = VecGetLocalSize(y,&ylen);CHKERRQ(ierr);
    m[0] = xlen%min;
    m[1] = ylen%min;
    ierr = MPIU_Allreduce(MPI_IN_PLACE,m,2,MPIU_INT,MPI_LOR,bigcomm);CHKERRQ(ierr);
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

    data->bs = bs = min;
    ierr     = MPI_Type_contiguous(bs,MPIU_SCALAR,&data->unit);CHKERRQ(ierr);
    ierr     = MPI_Type_commit(&data->unit);CHKERRQ(ierr);

    /* Shrink x and ix */
    ierr = VecCreateMPIWithArray(bigcomm,1,xlen/bs,PETSC_DECIDE,NULL,&xx);CHKERRQ(ierr); /* We only care xx's layout */
    if (ixid == IS_BLOCK) {
      ierr = ISBlockGetIndices(ix,&indices);CHKERRQ(ierr);
      ierr = ISBlockGetLocalSize(ix,&ixsize);CHKERRQ(ierr);
      ierr = ISCreateGeneral(PETSC_COMM_SELF,ixsize,indices,PETSC_COPY_VALUES,&ixx);CHKERRQ(ierr);
      ierr = ISBlockRestoreIndices(ix,&indices);CHKERRQ(ierr);
    } else { /* ixid == IS_STRIDE */
      ierr = ISGetLocalSize(ix,&ixsize);CHKERRQ(ierr);
      ierr = ISCreateStride(PETSC_COMM_SELF,ixsize/bs,ixfirst/bs,1,&ixx);CHKERRQ(ierr);
    }

    /* Shrink y and iy */
    ierr = VecCreateMPIWithArray(ycomm,1,ylen/bs,PETSC_DECIDE,NULL,&yy);CHKERRQ(ierr);
    if (iyid == IS_BLOCK) {
      ierr = ISBlockGetIndices(iy,&indices);CHKERRQ(ierr);
      ierr = ISBlockGetLocalSize(iy,&iysize);CHKERRQ(ierr);
      ierr = ISCreateGeneral(PETSC_COMM_SELF,iysize,indices,PETSC_COPY_VALUES,&iyy);CHKERRQ(ierr);
      ierr = ISBlockRestoreIndices(iy,&indices);CHKERRQ(ierr);
    } else { /* iyid == IS_STRIDE */
      ierr = ISGetLocalSize(iy,&iysize);CHKERRQ(ierr);
      ierr = ISCreateStride(PETSC_COMM_SELF,iysize/bs,iyfirst/bs,1,&iyy);CHKERRQ(ierr);
    }
  } else {
    ixx = ix;
    iyy = iy;
    yy  = y;
    if (xcommsize == 1) {ierr = VecCreateMPIWithArray(bigcomm,1,xlen,PETSC_DECIDE,NULL,&xx);CHKERRQ(ierr);} else xx = x;
  }

  /* Now it is ready to build SF with preprocessed (xx, yy) and (ixx, iyy) */
  ierr = ISGetIndices(ixx,&xindices);CHKERRQ(ierr);
  ierr = ISGetIndices(iyy,&yindices);CHKERRQ(ierr);
  ierr = VecGetLayout(xx,&xlayout);CHKERRQ(ierr);

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

    ierr = VecGetOwnershipRange(xx,&xstart,NULL);CHKERRQ(ierr);
    ierr = VecGetLayout(yy,&ylayout);CHKERRQ(ierr);
    ierr = VecGetOwnershipRange(yy,&ystart,NULL);CHKERRQ(ierr);

    ierr = VecGetLocalSize(yy,&nroots);CHKERRQ(ierr);
    ierr = ISGetLocalSize(iyy,&nleaves);CHKERRQ(ierr);
    ierr = PetscMalloc2(nleaves,&iremote,nleaves*2,&leafdata);CHKERRQ(ierr);

    for (i=0; i<nleaves; i++) {
      ierr            = PetscLayoutFindOwnerIndex(ylayout,yindices[i],&iremote[i].rank,&iremote[i].index);CHKERRQ(ierr);
      leafdata[2*i]   = yindices[i];
      leafdata[2*i+1] = (xcommsize > 1)? xindices[i] : xindices[i] + xstart;
    }

    ierr = PetscSFCreate(ycomm,&tmpsf);CHKERRQ(ierr);
    ierr = PetscSFSetGraph(tmpsf,nroots,nleaves,NULL,PETSC_USE_POINTER,iremote,PETSC_USE_POINTER);CHKERRQ(ierr);

    ierr = PetscSFComputeDegreeBegin(tmpsf,&degree);CHKERRQ(ierr);
    ierr = PetscSFComputeDegreeEnd(tmpsf,&degree);CHKERRQ(ierr);

    for (i=0; i<nroots; i++) inedges += degree[i];
    ierr = PetscMalloc1(inedges*2,&rootdata);CHKERRQ(ierr);
    ierr = PetscSFGatherBegin(tmpsf,MPIU_2INT,leafdata,rootdata);CHKERRQ(ierr);
    ierr = PetscSFGatherEnd(tmpsf,MPIU_2INT,leafdata,rootdata);CHKERRQ(ierr);

    ierr = PetscFree2(iremote,leafdata);CHKERRQ(ierr);
    ierr = PetscSFDestroy(&tmpsf);CHKERRQ(ierr);

    /* rootdata contains global index pairs (i, j). j's are owned by the current process, but i's can point to anywhere.
       We convert j to local, and convert i to (rank, index). In the end, we get an PtoS suitable for building SF.
     */
    nleaves = inedges;
    ierr    = VecGetLocalSize(xx,&nroots);CHKERRQ(ierr);
    ierr    = PetscMalloc1(nleaves,&ilocal);CHKERRQ(ierr);
    ierr    = PetscMalloc1(nleaves,&iremote);CHKERRQ(ierr);

    for (i=0; i<inedges; i++) {
      ilocal[i] = rootdata[2*i] - ystart; /* covert y's global index to local index */
      ierr      = PetscLayoutFindOwnerIndex(xlayout,rootdata[2*i+1],&iremote[i].rank,&iremote[i].index);CHKERRQ(ierr); /* convert x's global index to (rank, index) */
    }
    ierr = PetscFree(rootdata);CHKERRQ(ierr);
#else
    PetscInt       j,k,n,disp,rlentotal,*sstart,*xindices_sorted,*yindices_sorted;
    const PetscInt *yrange;
    PetscMPIInt    nsend,nrecv,nreq,count,yrank,*slens,*rlens,*sendto,*recvfrom,tag1,tag2;
    PetscInt       *rxindices,*ryindices;
    MPI_Request    *reqs,*sreqs,*rreqs;

    /* Sorting makes code simpler, faster and also helps getting rid of many O(P) arrays, which hurt scalability at large scale
       yindices_sorted - sorted yindices
       xindices_sorted - xindices sorted along with yindces
     */
    ierr = ISGetLocalSize(ixx,&n);CHKERRQ(ierr); /*ixx, iyy have the same local size */
    ierr = PetscMalloc2(n,&xindices_sorted,n,&yindices_sorted);CHKERRQ(ierr);
    ierr = PetscArraycpy(xindices_sorted,xindices,n);CHKERRQ(ierr);
    ierr = PetscArraycpy(yindices_sorted,yindices,n);CHKERRQ(ierr);
    ierr = PetscSortIntWithArray(n,yindices_sorted,xindices_sorted);CHKERRQ(ierr);
    ierr = VecGetOwnershipRange(xx,&xstart,NULL);CHKERRQ(ierr);
    if (xcommsize == 1) {for (i=0; i<n; i++) xindices_sorted[i] += xstart;} /* Convert to global indices */

    /*=============================================================================
             Calculate info about messages I need to send
      =============================================================================*/
    /* nsend    - number of non-empty messages to send
       sendto   - [nsend] ranks I will send messages to
       sstart   - [nsend+1] sstart[i] is the start index in xsindices_sorted[] I send to rank sendto[i]
       slens    - [ycommsize] I want to send slens[i] entries to rank i.
     */
    ierr = VecGetLayout(yy,&ylayout);CHKERRQ(ierr);
    ierr = PetscLayoutGetRanges(ylayout,&yrange);CHKERRQ(ierr);
    ierr = PetscCalloc1(ycommsize,&slens);CHKERRQ(ierr); /* The only O(P) array in this algorithm */

    i = j = nsend = 0;
    while (i < n) {
      if (yindices_sorted[i] >= yrange[j+1]) { /* If i-th index is out of rank j's bound */
        do {j++;} while (yindices_sorted[i] >= yrange[j+1] && j < ycommsize); /* Increase j until i-th index falls in rank j's bound */
        if (j == ycommsize) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Index %D not owned by any process, upper bound %D",yindices_sorted[i],yrange[ycommsize]);
      }
      i++;
      if (!slens[j]++) nsend++;
    }

    ierr = PetscMalloc2(nsend+1,&sstart,nsend,&sendto);CHKERRQ(ierr);

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
    ierr = PetscGatherNumberOfMessages(ycomm,NULL,slens,&nrecv);CHKERRQ(ierr);
    ierr = PetscGatherMessageLengths(ycomm,nsend,nrecv,slens,&recvfrom,&rlens);CHKERRQ(ierr);
    ierr = PetscFree(slens);CHKERRQ(ierr); /* Free the O(P) array ASAP */
    rlentotal = 0; for (i=0; i<nrecv; i++) rlentotal += rlens[i];

    /*=============================================================================
      Communicate with processors in recvfrom[] to populate rxindices and ryindices
      ============================================================================*/
    ierr  = PetscCommGetNewTag(ycomm,&tag1);CHKERRQ(ierr);
    ierr  = PetscCommGetNewTag(ycomm,&tag2);CHKERRQ(ierr);
    ierr  = PetscMalloc2(rlentotal,&rxindices,rlentotal,&ryindices);CHKERRQ(ierr);
    ierr  = PetscMPIIntCast((nsend+nrecv)*2,&nreq);CHKERRQ(ierr);
    ierr  = PetscMalloc1(nreq,&reqs);CHKERRQ(ierr);
    sreqs = reqs;
    rreqs = reqs + nsend*2;

    for (i=disp=0; i<nrecv; i++) {
      count = rlens[i];
      ierr  = MPI_Irecv(rxindices+disp,count,MPIU_INT,recvfrom[i],tag1,ycomm,rreqs+i);CHKERRQ(ierr);
      ierr  = MPI_Irecv(ryindices+disp,count,MPIU_INT,recvfrom[i],tag2,ycomm,rreqs+nrecv+i);CHKERRQ(ierr);
      disp += rlens[i];
    }

    for (i=0; i<nsend; i++) {
      ierr  = PetscMPIIntCast(sstart[i+1]-sstart[i],&count);CHKERRQ(ierr);
      ierr  = MPI_Isend(xindices_sorted+sstart[i],count,MPIU_INT,sendto[i],tag1,ycomm,sreqs+i);CHKERRQ(ierr);
      ierr  = MPI_Isend(yindices_sorted+sstart[i],count,MPIU_INT,sendto[i],tag2,ycomm,sreqs+nsend+i);CHKERRQ(ierr);
    }
    ierr = MPI_Waitall(nreq,reqs,MPI_STATUS_IGNORE);CHKERRQ(ierr);

    /* Transform VecScatter into SF */
    nleaves = rlentotal;
    ierr    = PetscMalloc1(nleaves,&ilocal);CHKERRQ(ierr);
    ierr    = PetscMalloc1(nleaves,&iremote);CHKERRQ(ierr);
    ierr    = MPI_Comm_rank(ycomm,&yrank);CHKERRQ(ierr);
    for (i=disp=0; i<nrecv; i++) {
      for (j=0; j<rlens[i]; j++) {
        k               = disp + j; /* k-th index pair */
        ilocal[k]       = ryindices[k] - yrange[yrank]; /* Convert y's global index to local index */
        ierr            = PetscLayoutFindOwnerIndex(xlayout,rxindices[k],&rank,&iremote[k].index);CHKERRQ(ierr); /* Convert x's global index to (rank, index) */
        iremote[k].rank = rank;
      }
      disp += rlens[i];
    }

    ierr = PetscFree2(sstart,sendto);CHKERRQ(ierr);
    ierr = PetscFree(slens);CHKERRQ(ierr);
    ierr = PetscFree(rlens);CHKERRQ(ierr);
    ierr = PetscFree(recvfrom);CHKERRQ(ierr);
    ierr = PetscFree(reqs);CHKERRQ(ierr);
    ierr = PetscFree2(rxindices,ryindices);CHKERRQ(ierr);
    ierr = PetscFree2(xindices_sorted,yindices_sorted);CHKERRQ(ierr);
#endif
  } else {
    /* PtoS or StoS */
    ierr = ISGetLocalSize(iyy,&nleaves);CHKERRQ(ierr);
    ierr = PetscMalloc1(nleaves,&ilocal);CHKERRQ(ierr);
    ierr = PetscMalloc1(nleaves,&iremote);CHKERRQ(ierr);
    ierr = PetscArraycpy(ilocal,yindices,nleaves);CHKERRQ(ierr);
    for (i=0; i<nleaves; i++) {
      ierr = PetscLayoutFindOwnerIndex(xlayout,xindices[i],&rank,&iremote[i].index);CHKERRQ(ierr);
      iremote[i].rank = rank;
    }
  }

  /* MUST build SF on xx's comm, which is not necessarily identical to yy's comm.
     In SF's view, xx contains the roots (i.e., the remote) and iremote[].rank are ranks in xx's comm.
     yy contains leaves, which are local and can be thought as part of PETSC_COMM_SELF. */
  ierr = PetscSFCreate(PetscObjectComm((PetscObject)xx),&data->sf);CHKERRQ(ierr);
  ierr = PetscSFSetFromOptions(data->sf);CHKERRQ(ierr);
  ierr = VecGetLocalSize(xx,&nroots);CHKERRQ(ierr);
  ierr = PetscSFSetGraph(data->sf,nroots,nleaves,ilocal,PETSC_OWN_POINTER,iremote,PETSC_OWN_POINTER);CHKERRQ(ierr); /* Give ilocal/iremote to petsc and no need to free them here */

  /* Free memory no longer needed */
  ierr = ISRestoreIndices(ixx,&xindices);CHKERRQ(ierr);
  ierr = ISRestoreIndices(iyy,&yindices);CHKERRQ(ierr);
  if (can_do_block_opt) {
    ierr = VecDestroy(&xx);CHKERRQ(ierr);
    ierr = VecDestroy(&yy);CHKERRQ(ierr);
    ierr = ISDestroy(&ixx);CHKERRQ(ierr);
    ierr = ISDestroy(&iyy);CHKERRQ(ierr);
  } else if (xcommsize == 1) {
    ierr = VecDestroy(&xx);CHKERRQ(ierr);
  }

functionend:
  if (!vscat->from_is) {ierr = ISDestroy(&ix);CHKERRQ(ierr);}
  if (!vscat->to_is  ) {ierr = ISDestroy(&iy);CHKERRQ(ierr);}

  /* vecscatter uses eager setup */
  ierr = PetscSFSetUp(data->sf);CHKERRQ(ierr);

  vscat->data                      = (void*)data;
  vscat->ops->begin                = VecScatterBegin_SF;
  vscat->ops->end                  = VecScatterEnd_SF;
  vscat->ops->remap                = VecScatterRemap_SF;
  vscat->ops->copy                 = VecScatterCopy_SF;
  vscat->ops->destroy              = VecScatterDestroy_SF;
  vscat->ops->view                 = VecScatterView_SF;
  vscat->ops->getremotecount       = VecScatterGetRemoteCount_SF;
  vscat->ops->getremote            = VecScatterGetRemote_SF;
  vscat->ops->getremoteordered     = VecScatterGetRemoteOrdered_SF;
  vscat->ops->restoreremote        = VecScatterRestoreRemote_SF;
  vscat->ops->restoreremoteordered = VecScatterRestoreRemoteOrdered_SF;
  PetscFunctionReturn(0);
}

PetscErrorCode VecScatterCreate_SF(VecScatter ctx)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ctx->ops->setup = VecScatterSetUp_SF;
  ierr = PetscObjectChangeTypeName((PetscObject)ctx,VECSCATTERSF);CHKERRQ(ierr);
  ierr = PetscInfo(ctx,"Using StarForest for vector scatter\n");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
