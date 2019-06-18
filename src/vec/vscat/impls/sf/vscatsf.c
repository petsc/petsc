#include <petsc/private/vecscatterimpl.h>    /*I   "petscvec.h"    I*/
#include <petsc/private/sfimpl.h> /*I "petscsf.h" I*/

typedef struct {
  PetscSF           sf;     /* the whole scatter, including local and remote */
  PetscSF           lsf;    /* the local part of the scatter, used for SCATTER_LOCAL */
  PetscInt          bs;     /* block size */
  MPI_Datatype      unit;   /* one unit = bs PetscScalars */
} VecScatter_SF;

static PetscErrorCode VecScatterBegin_SF(VecScatter vscat,Vec x,Vec y,InsertMode addv,ScatterMode mode)
{
  VecScatter_SF  *data=(VecScatter_SF*)vscat->data;
  PetscSF        sf;
  MPI_Op         mop=MPI_OP_NULL;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (x != y) {ierr = VecLockReadPush(x);CHKERRQ(ierr);}

  {
#if defined(PETSC_HAVE_CUDA)
    PetscBool is_cudatype = PETSC_FALSE;
    ierr = PetscObjectTypeCompareAny((PetscObject)x,&is_cudatype,VECSEQCUDA,VECMPICUDA,VECCUDA,"");CHKERRQ(ierr);
    if (is_cudatype) {
      VecCUDAAllocateCheckHost(x);
      if (x->valid_GPU_array == PETSC_OFFLOAD_GPU) {
        if (x->spptr && vscat->spptr) {ierr = VecCUDACopyFromGPUSome_Public(x,(PetscCUDAIndices)vscat->spptr,mode);CHKERRQ(ierr);}
        else {ierr = VecCUDACopyFromGPU(x);CHKERRQ(ierr);}
      }
      vscat->xdata = *((PetscScalar**)x->data);
    } else
#endif
    {
      ierr = VecGetArrayRead(x,&vscat->xdata);CHKERRQ(ierr);
    }
  }

  if (x != y) {ierr = VecGetArray(y,&vscat->ydata);CHKERRQ(ierr);}
  else vscat->ydata = (PetscScalar *)vscat->xdata;
  ierr = VecLockWriteSet_Private(y,PETSC_TRUE);CHKERRQ(ierr);

  /* SCATTER_LOCAL indicates ignoring inter-process communication */
  sf = (mode & SCATTER_LOCAL) ? data->lsf : data->sf;

  if (addv == INSERT_VALUES)   mop = MPI_REPLACE;
  else if (addv == ADD_VALUES) mop = MPI_SUM;
  else if (addv == MAX_VALUES) mop = MPI_MAX;
  else SETERRQ1(PetscObjectComm((PetscObject)sf),PETSC_ERR_SUP,"Unsupported InsertMode %D in VecScatterBegin/End",addv);

  if (mode & SCATTER_REVERSE) { /* reverse scatter sends root to leaf. Note that x and y are swapped in input */
    ierr = PetscSFBcastAndOpBegin(sf,data->unit,vscat->xdata,vscat->ydata,mop);CHKERRQ(ierr);
  } else { /* forward scatter sends leaf to root, i.e., x to y */
    ierr = PetscSFReduceBegin(sf,data->unit,vscat->xdata,vscat->ydata,mop);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode VecScatterEnd_SF(VecScatter vscat,Vec x,Vec y,InsertMode addv,ScatterMode mode)
{
  VecScatter_SF  *data=(VecScatter_SF*)vscat->data;
  PetscSF        sf;
  MPI_Op         mop=MPI_OP_NULL;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* SCATTER_LOCAL indicates ignoring inter-process communication */
  sf = (mode & SCATTER_LOCAL) ? data->lsf : data->sf;

  if (addv == INSERT_VALUES)   mop = MPI_REPLACE;
  else if (addv == ADD_VALUES) mop = MPI_SUM;
  else if (addv == MAX_VALUES) mop = MPI_MAX;
  else SETERRQ1(PetscObjectComm((PetscObject)sf),PETSC_ERR_SUP,"Unsupported InsertMode %D in VecScatterBegin/End",addv);

  if (mode & SCATTER_REVERSE) {/* reverse scatter sends root to leaf. Note that x and y are swapped in input */
    ierr = PetscSFBcastAndOpEnd(sf,data->unit,vscat->xdata,vscat->ydata,mop);CHKERRQ(ierr);
  } else { /* forward scatter sends leaf to root, i.e., x to y */
    ierr = PetscSFReduceEnd(sf,data->unit,vscat->xdata,vscat->ydata,mop);CHKERRQ(ierr);
  }

  if (x != y) {
    ierr = VecRestoreArrayRead(x,&vscat->xdata);CHKERRQ(ierr);
    ierr = VecLockReadPop(x);CHKERRQ(ierr);
  }
  ierr = VecRestoreArray(y,&vscat->ydata);CHKERRQ(ierr);
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
  ierr = PetscSFDuplicate(data->lsf,PETSCSF_DUPLICATE_GRAPH,&out->lsf);CHKERRQ(ierr);
  ierr = PetscSFSetUp(out->sf);CHKERRQ(ierr);
  ierr = PetscSFSetUp(out->lsf);CHKERRQ(ierr);

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
   x[i] to y[j], tomap gives a plan to change vscat to scatter x[tomap[i]] to y[j].
 */
static PetscErrorCode VecScatterRemap_SF(VecScatter vscat,const PetscInt *tomap,const PetscInt *frommap)
{
  VecScatter_SF  *data = (VecScatter_SF *)vscat->data;
  PetscSF        sfs[2],sf;
  PetscInt       i,j;
  PetscBool      ident;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  sfs[0] = data->sf;
  sfs[1] = data->lsf;

  if (tomap) {
    /* check if it is an identity map. If it is, do nothing */
    ident = PETSC_TRUE;
    for (i=0; i<data->sf->nleaves; i++) {if (i != tomap[i]) {ident = PETSC_FALSE; break; } }
    if (ident) PetscFunctionReturn(0);

    for (j=0; j<2; j++) {
      sf   = sfs[j];
      ierr = PetscSFSetUp(sf);CHKERRQ(ierr); /* to bulid sf->rmine if SetUp is not yet called */
      if (!sf->mine) { /* the old SF uses contiguous ilocal. After the remapping, it may not be true */
        ierr = PetscMalloc1(sf->nleaves,&sf->mine);CHKERRQ(ierr);
        ierr = PetscArraycpy(sf->mine,tomap,sf->nleaves);CHKERRQ(ierr);
        sf->mine_alloc = sf->mine;
      } else {
        for (i=0; i<sf->nleaves; i++)             sf->mine[i]   = tomap[sf->mine[i]];
      }
      for (i=0; i<sf->roffset[sf->nranks]; i++)   sf->rmine[i]  = tomap[sf->rmine[i]];
    }
  }

  if (frommap) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Unable to remap the FROM in scatters yet");
  PetscFunctionReturn(0);
}

static PetscErrorCode VecScatterGetRemoteCount_SF(VecScatter vscat,PetscBool send,PetscInt *num_procs,PetscInt *num_entries)
{
  VecScatter_SF     *data = (VecScatter_SF *)vscat->data;
  PetscSF           sf = data->sf;
  PetscInt          nranks,remote_start;
  PetscMPIInt       myrank;
  const PetscInt    *offset;
  const PetscMPIInt *ranks;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(PetscObjectComm((PetscObject)sf),&myrank);CHKERRQ(ierr);

  if (send) { ierr = PetscSFGetRootRanks(sf,&nranks,&ranks,&offset,NULL,NULL);CHKERRQ(ierr); }
  else { ierr = PetscSFGetLeafRanks(sf,&nranks,&ranks,&offset,NULL);CHKERRQ(ierr); }

  if (nranks) {
    remote_start = (myrank == ranks[0])? 1 : 0;
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
  VecScatter_SF     *data = (VecScatter_SF *)vscat->data;
  PetscSF           sf = data->sf;
  PetscInt          nranks,remote_start;
  PetscMPIInt       myrank;
  const PetscInt    *offset,*location;
  const PetscMPIInt *ranks;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(PetscObjectComm((PetscObject)sf),&myrank);CHKERRQ(ierr);

  if (send) { ierr = PetscSFGetRootRanks(sf,&nranks,&ranks,&offset,&location,NULL);CHKERRQ(ierr); }
  else { ierr = PetscSFGetLeafRanks(sf,&nranks,&ranks,&offset,&location);CHKERRQ(ierr); }

  if (nranks) {
    remote_start = (myrank == ranks[0])? 1 : 0;
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
  VecScatter_SF  *data;
  MPI_Comm       comm,xcomm,ycomm,bigcomm;
  Vec            x=vscat->from_v,y=vscat->to_v,xx,yy;
  IS             ix=vscat->from_is,iy=vscat->to_is,ixx,iyy;
  PetscMPIInt    size,xcommsize,ycommsize,myrank;
  PetscInt       i,j,n,N,nroots,nleaves,inedges=0,*leafdata,*rootdata,*ilocal,*lilocal,xstart,ystart,lnleaves,ixsize,iysize,xlen,ylen;
  const PetscInt *xindices,*yindices,*degree;
  PetscSFNode    *iremote,*liremote;
  PetscLayout    xlayout,ylayout;
  PetscSF        tmpsf;
  ISTypeID       ixid,iyid;
  PetscInt       bs,bsx,bsy,min=PETSC_MIN_INT,max=PETSC_MAX_INT,ixfirst,ixstep,iyfirst,iystep;
  PetscBool      can_do_block_opt=PETSC_FALSE;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscNewLog(vscat,&data);CHKERRQ(ierr);

  /* Let P and S stand for parallel and sequential vectors respectively, there are four combinations of vecscatters: PtoP, PtoS, StoP and StoS.
    The assumption of VecScatterCreate(Vec x,IS ix,Vec y,IS iy,VecScatter *newctx) is: if x is parallel, then ix contains global
    indices of x. If x is sequential, ix contains local indices of x. Similarily for y and iy.

    SF builds around concepts of local leaves and remote roots, which correspond to an StoP scatter. We transform PtoP and PtoS to StoP, and
    treat StoS as a trivial StoP.
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

  /* Do block optimization by taking advantage of high level info available in ix, iy.
     The block optimization is valid when all of the following conditions are met:
     1) ix, iy are blocked or can be blocked (i.e., strided with step=1);
     2) ix, iy have the same block size;
     3) all processors agree on one block size;
     4) no blocks span more than one process;
   */
  data->bs   = 1; /* default, no blocking */
  data->unit = MPIU_SCALAR;
  ierr       = ISGetTypeID_Private(ix,&ixid);CHKERRQ(ierr);
  ierr       = ISGetTypeID_Private(iy,&iyid);CHKERRQ(ierr);
  bigcomm    = (ycommsize == 1) ? xcomm : ycomm;

  if (ixid == IS_BLOCK)       {ierr = ISGetBlockSize(ix,&bsx);CHKERRQ(ierr);}
  else if (ixid == IS_STRIDE) {ierr = ISStrideGetInfo(ix,&ixfirst,&ixstep);CHKERRQ(ierr);}

  if ( iyid == IS_BLOCK)      {ierr = ISGetBlockSize(iy,&bsy);CHKERRQ(ierr);}
  else if (iyid == IS_STRIDE) {ierr = ISStrideGetInfo(iy,&iyfirst,&iystep);CHKERRQ(ierr);}

  /* Processors could go through different path in this if-else test */
  if (ixid == IS_BLOCK && iyid == IS_BLOCK) {
    min = PetscMin(bsx,bsy);
    max = PetscMax(bsx,bsy);
  } else if (ixid == IS_BLOCK  && iyid == IS_STRIDE && iystep==1 && iyfirst%bsx==0) {
    min = max = bsx;
  } else if (ixid == IS_STRIDE && iyid == IS_BLOCK  && ixstep==1 && ixfirst%bsy==0) {
    min = max = bsy;
  }
  ierr = MPIU_Allreduce(MPI_IN_PLACE,&min,1,MPIU_INT,MPI_MIN,bigcomm);CHKERRQ(ierr);
  ierr = MPIU_Allreduce(MPI_IN_PLACE,&max,1,MPIU_INT,MPI_MAX,bigcomm);CHKERRQ(ierr);

  /* Since we used allreduce above, all ranks will have the same min and max. min==max
     implies all ranks have the same bs. Do further test to see if local vectors are dividable
     by bs on ALL ranks. If they are, we are ensured that no blocks span more than one processor.
   */
  if (min == max && min > 1) {
    PetscInt m[2];
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

     yy is a little special. If y is seq, then yy is the concatenation of seq y's on xcomm. In this way,
     we can treat PtoP and PtoS uniformly as PtoP.
   */
  if (can_do_block_opt) {
    const PetscInt *indices;

    data->bs = bs = min;
    ierr     = MPI_Type_contiguous(bs,MPIU_SCALAR,&data->unit);CHKERRQ(ierr);
    ierr     = MPI_Type_commit(&data->unit);CHKERRQ(ierr);

    /* Shrink x and ix */
    ierr = VecCreateMPIWithArray(xcomm,1,xlen/bs,PETSC_DECIDE,NULL,&xx);CHKERRQ(ierr); /* We only care xx's layout */
    if (ixid == IS_BLOCK) {
      ierr = ISBlockGetIndices(ix,&indices);CHKERRQ(ierr);
      ierr = ISBlockGetLocalSize(ix,&ixsize);CHKERRQ(ierr);
      ierr = ISCreateGeneral(PETSC_COMM_SELF,ixsize,indices,PETSC_COPY_VALUES,&ixx);CHKERRQ(ierr);
      ierr = ISBlockRestoreIndices(ix,&indices);CHKERRQ(ierr);
    } else if (ixid == IS_STRIDE) {
      ierr = ISGetLocalSize(ix,&ixsize);CHKERRQ(ierr);
      ierr = ISCreateStride(PETSC_COMM_SELF,ixsize/bs,ixfirst/bs,1,&ixx);CHKERRQ(ierr);
    }

    /* Shrink y and iy */
    ierr = VecCreateMPIWithArray(bigcomm,1,ylen/bs,PETSC_DECIDE,NULL,&yy);CHKERRQ(ierr);
    if (iyid == IS_BLOCK) {
      ierr = ISBlockGetIndices(iy,&indices);CHKERRQ(ierr);
      ierr = ISBlockGetLocalSize(iy,&iysize);CHKERRQ(ierr);
      ierr = ISCreateGeneral(PETSC_COMM_SELF,iysize,indices,PETSC_COPY_VALUES,&iyy);CHKERRQ(ierr);
      ierr = ISBlockRestoreIndices(iy,&indices);CHKERRQ(ierr);
    } else if (iyid == IS_STRIDE) {
      ierr = ISGetLocalSize(iy,&iysize);CHKERRQ(ierr);
      ierr = ISCreateStride(PETSC_COMM_SELF,iysize/bs,iyfirst/bs,1,&iyy);CHKERRQ(ierr);
    }
  } else {
    ixx = ix;
    iyy = iy;
    xx  = x;
    if (ycommsize == 1) {ierr = VecCreateMPIWithArray(bigcomm,1,ylen,PETSC_DECIDE,NULL,&yy);CHKERRQ(ierr);} else yy = y;
  }

  /* Now it is ready to build SF with preprocessed (xx, yy) and (ixx, iyy) */
  ierr = ISGetIndices(ixx,&xindices);CHKERRQ(ierr);
  ierr = ISGetIndices(iyy,&yindices);CHKERRQ(ierr);

  if (xcommsize > 1) {
    /* PtoP or PtoS */
    ierr = VecGetLayout(xx,&xlayout);CHKERRQ(ierr);
    ierr = VecGetOwnershipRange(xx,&xstart,NULL);CHKERRQ(ierr);
    ierr = VecGetLayout(yy,&ylayout);CHKERRQ(ierr);
    ierr = VecGetOwnershipRange(yy,&ystart,NULL);CHKERRQ(ierr);

    /* Each process has a set of global index pairs (i, j) to scatter xx[i] to yy[j]. We first shift (i, j) to owner process of i through a tmp SF */
    ierr = VecGetLocalSize(xx,&nroots);CHKERRQ(ierr);
    ierr = ISGetLocalSize(ixx,&nleaves);CHKERRQ(ierr);
    ierr = PetscMalloc2(nleaves,&iremote,nleaves*2,&leafdata);CHKERRQ(ierr);

    for (i=0; i<nleaves; i++) {
      ierr            = PetscLayoutFindOwnerIndex(xlayout,xindices[i],&iremote[i].rank,&iremote[i].index);CHKERRQ(ierr);
      leafdata[2*i]   = xindices[i];
      leafdata[2*i+1] = (ycommsize > 1)? yindices[i] : yindices[i] + ystart;
    }

    ierr = PetscSFCreate(xcomm,&tmpsf);CHKERRQ(ierr);
    ierr = PetscSFSetGraph(tmpsf,nroots,nleaves,NULL,PETSC_USE_POINTER,iremote,PETSC_USE_POINTER);CHKERRQ(ierr);

    ierr = PetscSFComputeDegreeBegin(tmpsf,&degree);CHKERRQ(ierr);
    ierr = PetscSFComputeDegreeEnd(tmpsf,&degree);CHKERRQ(ierr);

    for (i=0; i<nroots; i++) inedges += degree[i];
    ierr = PetscMalloc1(inedges*2,&rootdata);CHKERRQ(ierr);
    ierr = PetscSFGatherBegin(tmpsf,MPIU_2INT,leafdata,rootdata);CHKERRQ(ierr);
    ierr = PetscSFGatherEnd(tmpsf,MPIU_2INT,leafdata,rootdata);CHKERRQ(ierr);

    ierr = PetscFree2(iremote,leafdata);CHKERRQ(ierr);
    ierr = PetscSFDestroy(&tmpsf);CHKERRQ(ierr);

    /* rootdata contains global index pairs (i, j). i's are owned by the current process, but j's can point to anywhere.
       We convert i to local, and convert j to (rank, index). In the end, we get an StoP suitable for building SF.
     */
    nleaves = inedges;
    ierr    = VecGetLocalSize(yy,&nroots);CHKERRQ(ierr);
    ierr    = PetscMalloc1(nleaves,&ilocal);CHKERRQ(ierr);
    ierr    = PetscMalloc1(nleaves,&iremote);CHKERRQ(ierr);

    for (i=0; i<inedges; i++) {
      ilocal[i] = rootdata[2*i] - xstart; /* covert x's global index to local index */
      ierr      = PetscLayoutFindOwnerIndex(ylayout,rootdata[2*i+1],&iremote[i].rank,&iremote[i].index);CHKERRQ(ierr); /* convert y's global index to (rank, index) */
    }

    /* MUST build SF on yy's comm, which is not necessarily identical to xx's comm.
       In SF's view, yy contains the roots (i.e., the remote) and iremote[].rank are ranks in yy's comm.
       xx contains leaves, which are local and can be thought as part of PETSC_COMM_SELF. */
    ierr = PetscSFCreate(PetscObjectComm((PetscObject)yy),&data->sf);CHKERRQ(ierr);
    ierr = PetscSFSetGraph(data->sf,nroots,nleaves,ilocal,PETSC_OWN_POINTER,iremote,PETSC_OWN_POINTER);CHKERRQ(ierr);
    ierr = PetscFree(rootdata);CHKERRQ(ierr);
  } else {
    /* StoP or StoS */
    ierr = VecGetLayout(yy,&ylayout);CHKERRQ(ierr);
    ierr = ISGetLocalSize(ixx,&nleaves);CHKERRQ(ierr);
    ierr = VecGetLocalSize(yy,&nroots);CHKERRQ(ierr);
    ierr = PetscMalloc1(nleaves,&ilocal);CHKERRQ(ierr);
    ierr = PetscMalloc1(nleaves,&iremote);CHKERRQ(ierr);
    ierr = PetscArraycpy(ilocal,xindices,nleaves);CHKERRQ(ierr);
    for (i=0; i<nleaves; i++) {ierr = PetscLayoutFindOwnerIndex(ylayout,yindices[i],&iremote[i].rank,&iremote[i].index);CHKERRQ(ierr);}
    ierr = PetscSFCreate(PetscObjectComm((PetscObject)yy),&data->sf);CHKERRQ(ierr);
    ierr = PetscSFSetGraph(data->sf,nroots,nleaves,ilocal,PETSC_OWN_POINTER,iremote,PETSC_OWN_POINTER);CHKERRQ(ierr);
  }

  /* Free memory no longer needed */
  ierr = ISRestoreIndices(ixx,&xindices);CHKERRQ(ierr);
  ierr = ISRestoreIndices(iyy,&yindices);CHKERRQ(ierr);
  if (can_do_block_opt) {
    ierr = VecDestroy(&xx);CHKERRQ(ierr);
    ierr = VecDestroy(&yy);CHKERRQ(ierr);
    ierr = ISDestroy(&ixx);CHKERRQ(ierr);
    ierr = ISDestroy(&iyy);CHKERRQ(ierr);
  } else if (ycommsize == 1) {
    ierr = VecDestroy(&yy);CHKERRQ(ierr);
  }
  if (!vscat->from_is) {ierr = ISDestroy(&ix);CHKERRQ(ierr);}
  if (!vscat->to_is  ) {ierr = ISDestroy(&iy);CHKERRQ(ierr);}

  /* Create lsf, the local scatter. Could use PetscSFCreateEmbeddedLeafSF, but since we know the comm is PETSC_COMM_SELF, we can make it fast */
  ierr = PetscObjectGetComm((PetscObject)data->sf,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm,&myrank);CHKERRQ(ierr);

  /* Find out local edges and build a local SF */
  {
    const PetscInt    *ilocal;
    const PetscSFNode *iremote;
    ierr = PetscSFGetGraph(data->sf,&nroots,&nleaves,&ilocal,&iremote);CHKERRQ(ierr);
    for (i=lnleaves=0; i<nleaves; i++) {if (iremote[i].rank == (PetscInt)myrank) lnleaves++;}
    ierr = PetscMalloc1(lnleaves,&lilocal);CHKERRQ(ierr);
    ierr = PetscMalloc1(lnleaves,&liremote);CHKERRQ(ierr);

    for (i=j=0; i<nleaves; i++) {
      if (iremote[i].rank == (PetscInt)myrank) {
        lilocal[j]        = ilocal? ilocal[i] : i; /* ilocal=NULL for contiguous storage */
        liremote[j].rank  = 0; /* rank in PETSC_COMM_SELF */
        liremote[j].index = iremote[i].index;
        j++;
      }
    }
    ierr = PetscSFCreate(PETSC_COMM_SELF,&data->lsf);CHKERRQ(ierr);
    ierr = PetscSFSetGraph(data->lsf,nroots,lnleaves,lilocal,PETSC_OWN_POINTER,liremote,PETSC_OWN_POINTER);CHKERRQ(ierr);
  }

  /* vecscatter uses eager setup */
  ierr = PetscSFSetUp(data->sf);CHKERRQ(ierr);
  ierr = PetscSFSetUp(data->lsf);CHKERRQ(ierr);

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
