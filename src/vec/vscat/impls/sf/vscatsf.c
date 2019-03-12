#include <petsc/private/vecscatterimpl.h>    /*I   "petscvec.h"    I*/
#include <petsc/private/sfimpl.h> /*I "petscsf.h" I*/

typedef struct {
  PetscSF           sf;     /* the whole scatter, including local and remote */
  PetscSF           lsf;    /* the local part of the scatter, used for SCATTER_LOCAL */
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
    ierr = PetscSFBcastAndOpBegin(sf,MPIU_SCALAR,vscat->xdata,vscat->ydata,mop);CHKERRQ(ierr);
  } else { /* forward scatter sends leaf to root, i.e., x to y */
    ierr = PetscSFReduceBegin(sf,MPIU_SCALAR,vscat->xdata,vscat->ydata,mop);CHKERRQ(ierr);
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
    ierr = PetscSFBcastAndOpEnd(sf,MPIU_SCALAR,vscat->xdata,vscat->ydata,mop);CHKERRQ(ierr);
  } else { /* forward scatter sends leaf to root, i.e., x to y */
    ierr = PetscSFReduceEnd(sf,MPIU_SCALAR,vscat->xdata,vscat->ydata,mop);CHKERRQ(ierr);
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
        ierr = PetscMemcpy(sf->mine,tomap,sizeof(PetscInt)*sf->nleaves);CHKERRQ(ierr);
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

  if (send) { ierr = PetscSFGetRanks(sf,&nranks,&ranks,&offset,NULL,NULL);CHKERRQ(ierr); }
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

  if (send) { ierr = PetscSFGetRanks(sf,&nranks,&ranks,&offset,&location,NULL);CHKERRQ(ierr); }
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

static PetscErrorCode VecScatterSetUp_SF(VecScatter vscat)
{
  VecScatter_SF  *data;
  MPI_Comm       comm,xcomm,ycomm;
  Vec            x=vscat->from_v,y=vscat->to_v,yy;
  IS             ix=vscat->from_is,iy=vscat->to_is;
  PetscMPIInt    size,xsize,ysize,myrank;
  PetscInt       i,j,n,N,nroots,nleaves,inedges=0,*leafdata,*rootdata,*ilocal,*lilocal,xstart,ystart,lnleaves;
  const PetscInt *xindices,*yindices,*degree;
  PetscSFNode    *iremote,*liremote;
  PetscLayout    xlayout,ylayout;
  PetscSF        tmpsf;
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
  ierr = MPI_Comm_size(xcomm,&xsize);CHKERRQ(ierr);
  ierr = MPI_Comm_size(ycomm,&ysize);CHKERRQ(ierr);

  /* NULL ix or iy in VecScatterCreate(x,ix,y,iy,newctx) has special meaning. Recover them for these cases */
  if (!ix) {
    if (xsize > 1 && ysize == 1) { /* PtoS: null ix means the whole x will be scattered to each seq y */
      ierr = VecGetSize(x,&N);CHKERRQ(ierr);
      ierr = ISCreateStride(PETSC_COMM_SELF,N,0,1,&ix);CHKERRQ(ierr);
    } else { /* PtoP, StoP or StoS: null ix means the whole local part of x will be scattered */
      ierr = VecGetLocalSize(x,&n);CHKERRQ(ierr);
      ierr = VecGetOwnershipRange(x,&xstart,NULL);CHKERRQ(ierr);
      ierr = ISCreateStride(PETSC_COMM_SELF,n,xstart,1,&ix);CHKERRQ(ierr);
    }
  }

  if (!iy) {
    if (xsize == 1 && ysize > 1) { /* StoP: null iy means the whole y will be scattered to from each seq x */
      ierr = VecGetSize(y,&N);CHKERRQ(ierr);
      ierr = ISCreateStride(PETSC_COMM_SELF,N,0,1,&iy);CHKERRQ(ierr);
    } else { /* PtoP, StoP or StoS: null iy means the whole local part of y will be scattered to */
      ierr = VecGetLocalSize(y,&n);CHKERRQ(ierr);
      ierr = VecGetOwnershipRange(y,&ystart,NULL);CHKERRQ(ierr);
      ierr = ISCreateStride(PETSC_COMM_SELF,n,ystart,1,&iy);CHKERRQ(ierr);
    }
  }

  ierr = ISGetIndices(ix,&xindices);CHKERRQ(ierr);
  ierr = ISGetIndices(iy,&yindices);CHKERRQ(ierr);

  if (xsize > 1) {
    /* PtoP or PtoS */
    /* If y is seq, make up a parallel yy by concat'ing seq y's so we can treat PtoP and PtoS uniformly as PtoP */
    if (ysize == 1) {
      ierr = VecGetLocalSize(y,&n);CHKERRQ(ierr);
      ierr = VecCreateMPIWithArray(xcomm,1,n,PETSC_DECIDE,NULL,&yy);CHKERRQ(ierr); /* Attention: use the bigger xcomm instead of seq ycomm */
    } else {
      yy = y;
    }

    ierr = VecGetLayout(x,&xlayout);CHKERRQ(ierr);
    ierr = VecGetOwnershipRange(x,&xstart,NULL);CHKERRQ(ierr);
    ierr = VecGetLayout(yy,&ylayout);CHKERRQ(ierr);
    ierr = VecGetOwnershipRange(yy,&ystart,NULL);CHKERRQ(ierr);

    /* Each process has a set of global index pairs (i, j) to scatter x[i] to yy[j]. We first shift (i, j) to owner process of i through a tmp SF */
    ierr = VecGetLocalSize(x,&nroots);CHKERRQ(ierr);
    ierr = ISGetLocalSize(ix,&nleaves);CHKERRQ(ierr); /* ix and iy should have the same local size */
    ierr = PetscMalloc2(nleaves,&iremote,nleaves*2,&leafdata);CHKERRQ(ierr);

    for (i=0; i<nleaves; i++) {
      ierr            = PetscLayoutFindOwnerIndex(xlayout,xindices[i],&iremote[i].rank,&iremote[i].index);CHKERRQ(ierr);
      leafdata[2*i]   = xindices[i];
      leafdata[2*i+1] = (ysize > 1)? yindices[i] : yindices[i] + ystart;
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
    ierr    = VecGetLocalSize(y,&nroots);CHKERRQ(ierr);
    ierr    = PetscMalloc1(nleaves,&ilocal);CHKERRQ(ierr);
    ierr    = PetscMalloc1(nleaves,&iremote);CHKERRQ(ierr);

    for (i=0; i<inedges; i++) {
      ilocal[i] = rootdata[2*i] - xstart; /* covert x's global index to local index */
      ierr      = PetscLayoutFindOwnerIndex(ylayout,rootdata[2*i+1],&iremote[i].rank,&iremote[i].index);CHKERRQ(ierr); /* convert y's global index to (rank, index) */
    }

    /* MUST build SF on yy's comm, which is not necessarily identical to x's comm.
       In SF's view, yy contains the roots (i.e., the remote) and iremote[].rank are ranks in yy's comm.
       x contains leaves, which are local and can be thought as part of PETSC_COMM_SELF. */
    ierr = PetscSFCreate(PetscObjectComm((PetscObject)yy),&data->sf);CHKERRQ(ierr);
    ierr = PetscSFSetGraph(data->sf,nroots,nleaves,ilocal,PETSC_OWN_POINTER,iremote,PETSC_OWN_POINTER);CHKERRQ(ierr);
    ierr = PetscFree(rootdata);CHKERRQ(ierr);
    if (ysize == 1) { ierr = VecDestroy(&yy);CHKERRQ(ierr); }
  } else {
    /* StoP or StoS */
    ierr = VecGetLayout(y,&ylayout);CHKERRQ(ierr);
    ierr = ISGetLocalSize(ix,&nleaves);CHKERRQ(ierr);
    ierr = VecGetLocalSize(y,&nroots);CHKERRQ(ierr);
    ierr = PetscMalloc1(nleaves,&ilocal);CHKERRQ(ierr);
    ierr = PetscMalloc1(nleaves,&iremote);CHKERRQ(ierr);
    ierr = PetscMemcpy(ilocal,xindices,nleaves*sizeof(PetscInt));CHKERRQ(ierr);
    for (i=0; i<nleaves; i++) { ierr = PetscLayoutFindOwnerIndex(ylayout,yindices[i],&iremote[i].rank,&iremote[i].index);CHKERRQ(ierr); }
    ierr = PetscSFCreate(ycomm,&data->sf);CHKERRQ(ierr); /* Attention: the SF is built on ycomm, which is equal or bigger than xcomm */
    ierr = PetscSFSetGraph(data->sf,nroots,nleaves,ilocal,PETSC_OWN_POINTER,iremote,PETSC_OWN_POINTER);CHKERRQ(ierr);
  }

  ierr = ISRestoreIndices(ix,&xindices);CHKERRQ(ierr);
  ierr = ISRestoreIndices(iy,&yindices);CHKERRQ(ierr);
  if (!vscat->from_is) { ierr = ISDestroy(&ix);CHKERRQ(ierr); }
  if (!vscat->to_is)   { ierr = ISDestroy(&iy);CHKERRQ(ierr); }

  /* Create lsf, the local scatter. Could use PetscSFCreateEmbeddedLeafSF, but since we know the comm is PETSC_COMM_SELF, we can make it fast */
  ierr = PetscObjectGetComm((PetscObject)data->sf,&comm);CHKERRQ(ierr);  /* sf's comm is either xcomm or ycomm, whichever is bigger */
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm,&myrank);CHKERRQ(ierr);

  /* Find out local edges and build a local SF */
  {
    const PetscInt    *ilocal;
    const PetscSFNode *iremote;
    ierr = PetscSFGetGraph(data->sf,&nroots,&nleaves,&ilocal,&iremote);CHKERRQ(ierr);
    for (i=lnleaves=0; i<nleaves; i++) { if (iremote[i].rank == (PetscInt)myrank) lnleaves++; }
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
