
/*
     Defines the methods VecScatterBegin/End_1,2,......
     This is included by vpscat.c with different values for BS

     This is a terrible way of doing "templates" in C.
*/
#define PETSCMAP1_a(a,b)  a ## _ ## b
#define PETSCMAP1_b(a,b)  PETSCMAP1_a(a,b)
#define PETSCMAP1(a)      PETSCMAP1_b(a,BS)

PetscErrorCode PETSCMAP1(VecScatterBegin)(VecScatter ctx,Vec xin,Vec yin,InsertMode addv,ScatterMode mode)
{
  VecScatter_MPI_General *to,*from;
  PetscScalar            *xv,*yv,*svalues;
  MPI_Request            *rwaits,*swaits;
  PetscErrorCode         ierr;
  PetscInt               i,*indices,*sstarts,nrecvs,nsends,bs;
#if defined(PETSC_HAVE_CUDA)
  PetscBool              is_cudatype = PETSC_FALSE;
#endif

  PetscFunctionBegin;
  if (mode & SCATTER_REVERSE) {
    to     = (VecScatter_MPI_General*)ctx->fromdata;
    from   = (VecScatter_MPI_General*)ctx->todata;
    rwaits = from->rev_requests;
    swaits = to->rev_requests;
  } else {
    to     = (VecScatter_MPI_General*)ctx->todata;
    from   = (VecScatter_MPI_General*)ctx->fromdata;
    rwaits = from->requests;
    swaits = to->requests;
  }
  bs      = to->bs;
  svalues = to->values;
  nrecvs  = from->n;
  nsends  = to->n;
  indices = to->indices;
  sstarts = to->starts;
#if defined(PETSC_HAVE_CUDA)
  ierr = PetscObjectTypeCompareAny((PetscObject)xin,&is_cudatype,VECSEQCUDA,VECMPICUDA,VECCUDA,"");CHKERRQ(ierr);
  if (is_cudatype) {
    VecCUDAAllocateCheckHost(xin);
    if (xin->offloadmask == PETSC_OFFLOAD_GPU) {
      if (xin->spptr && ctx->spptr) {
        ierr = VecCUDACopyFromGPUSome_Public(xin,(PetscCUDAIndices)ctx->spptr,mode);CHKERRQ(ierr);
      } else {
        ierr = VecCUDACopyFromGPU(xin);CHKERRQ(ierr);
      }
    }
    xv = *((PetscScalar**)xin->data);
  } else
#endif
  {
    ierr = VecGetArrayRead(xin,(const PetscScalar**)&xv);CHKERRQ(ierr);
  }

  if (xin != yin) {ierr = VecGetArray(yin,&yv);CHKERRQ(ierr);}
  else yv = xv;

  if (!(mode & SCATTER_LOCAL)) {
    /* post receives since they were not previously posted    */
    ierr = MPI_Startall_irecv(from->starts[nrecvs]*bs,MPIU_SCALAR,nrecvs,rwaits);CHKERRQ(ierr);

    if (to->sharedspace) {
      /* Pack the send data into my shared memory buffer  --- this is the normal forward scatter */
      PETSCMAP1(Pack)(to->sharedcnt,to->sharedspaceindices,xv,to->sharedspace,bs);
    } else {
      /* Pack the send data into receivers shared memory buffer -- this is the normal backward scatter */
      for (i=0; i<to->msize; i++) {
        if (to->sharedspacesoffset && to->sharedspacesoffset[i] > -1) {
          PETSCMAP1(Pack)(to->sharedspacestarts[i+1] - to->sharedspacestarts[i],to->sharedspaceindices + to->sharedspacestarts[i],xv,&to->sharedspaces[i][bs*to->sharedspacesoffset[i]],bs);
        }
      }
    }
    /* this version packs and sends one at a time */
    for (i=0; i<nsends; i++) {
      PETSCMAP1(Pack)(sstarts[i+1]-sstarts[i],indices + sstarts[i],xv,svalues + bs*sstarts[i],bs);
      ierr = MPI_Start_isend((sstarts[i+1]-sstarts[i])*bs,MPIU_SCALAR,swaits+i);CHKERRQ(ierr);
    }
  }

  /* take care of local scatters */
  if (to->local.n) {
    if (to->local.memcpy_plan.optimized[0] && addv == INSERT_VALUES) {
      /* do copy when it is not a self-to-self copy */
      if (!(yv == xv && to->local.memcpy_plan.same_copy_starts)) {
        for (i=to->local.memcpy_plan.copy_offsets[0]; i<to->local.memcpy_plan.copy_offsets[1]; i++) {
          /* Do we need to take care of overlaps? We could but overlaps sound more like a bug than a requirement,
             so I just leave it and let PetscMemcpy detect this bug.
           */
          ierr = PetscArraycpy(yv + from->local.memcpy_plan.copy_starts[i],xv + to->local.memcpy_plan.copy_starts[i],to->local.memcpy_plan.copy_lengths[i]);CHKERRQ(ierr);
        }
      }
    } else {
      if (xv == yv && addv == INSERT_VALUES && to->local.nonmatching_computed) {
        /* only copy entries that do not share identical memory locations */
        ierr = PETSCMAP1(Scatter)(to->local.n_nonmatching,to->local.slots_nonmatching,xv,from->local.slots_nonmatching,yv,addv,bs);CHKERRQ(ierr);
      } else {
        ierr = PETSCMAP1(Scatter)(to->local.n,to->local.vslots,xv,from->local.vslots,yv,addv,bs);CHKERRQ(ierr);
      }
    }
  }
  ierr = VecRestoreArrayRead(xin,(const PetscScalar**)&xv);CHKERRQ(ierr);
  if (xin != yin) {ierr = VecRestoreArray(yin,&yv);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

/* --------------------------------------------------------------------------------------*/

PetscErrorCode PETSCMAP1(VecScatterEnd)(VecScatter ctx,Vec xin,Vec yin,InsertMode addv,ScatterMode mode)
{
  VecScatter_MPI_General *to,*from;
  PetscScalar            *rvalues,*yv;
  PetscErrorCode         ierr;
  PetscInt               nrecvs,nsends,*indices,count,*rstarts,bs;
  PetscMPIInt            imdex;
  MPI_Request            *rwaits,*swaits;
  MPI_Status             xrstatus,*sstatus;
  PetscMPIInt            i;

  PetscFunctionBegin;
  if (mode & SCATTER_LOCAL) PetscFunctionReturn(0);
  ierr = VecGetArray(yin,&yv);CHKERRQ(ierr);

  to      = (VecScatter_MPI_General*)ctx->todata;
  from    = (VecScatter_MPI_General*)ctx->fromdata;
  rwaits  = from->requests;
  swaits  = to->requests;
  sstatus = to->sstatus;    /* sstatus and rstatus are always stored in to */
  if (mode & SCATTER_REVERSE) {
    to     = (VecScatter_MPI_General*)ctx->fromdata;
    from   = (VecScatter_MPI_General*)ctx->todata;
    rwaits = from->rev_requests;
    swaits = to->rev_requests;
  }
  bs      = from->bs;
  rvalues = from->values;
  nrecvs  = from->n;
  nsends  = to->n;
  indices = from->indices;
  rstarts = from->starts;


  ierr = MPI_Barrier(PetscObjectComm((PetscObject)ctx));CHKERRQ(ierr);

  /* unpack one at a time */
  count = nrecvs;
  while (count) {
    ierr = MPI_Waitany(nrecvs,rwaits,&imdex,&xrstatus);CHKERRQ(ierr);
    /* unpack receives into our local space */
    ierr = PETSCMAP1(UnPack)(rstarts[imdex+1] - rstarts[imdex],rvalues + bs*rstarts[imdex],indices + rstarts[imdex],yv,addv,bs);CHKERRQ(ierr);
    count--;
  }
  /* handle processes that share the same shared memory communicator */
  if (from->sharedspace) {
    /* unpack the data from my shared memory buffer  --- this is the normal backward scatter */
    PETSCMAP1(UnPack)(from->sharedcnt,from->sharedspace,from->sharedspaceindices,yv,addv,bs);
  } else {
    /* unpack the data from each of my sending partners shared memory buffers --- this is the normal forward scatter */
    for (i=0; i<from->msize; i++) {
      if (from->sharedspacesoffset && from->sharedspacesoffset[i] > -1) {
        ierr = PETSCMAP1(UnPack)(from->sharedspacestarts[i+1] - from->sharedspacestarts[i],&from->sharedspaces[i][bs*from->sharedspacesoffset[i]],from->sharedspaceindices + from->sharedspacestarts[i],yv,addv,bs);CHKERRQ(ierr);
      }
    }
  }
  ierr = MPI_Barrier(PetscObjectComm((PetscObject)ctx));CHKERRQ(ierr);

  /* wait on sends */
  if (nsends) {ierr = MPI_Waitall(nsends,swaits,sstatus);CHKERRQ(ierr);}
  ierr = VecRestoreArray(yin,&yv);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------- */
#include <../src/vec/vec/impls/node/vecnodeimpl.h>

PetscErrorCode PETSCMAP1(VecScatterBeginMPI3Node)(VecScatter ctx,Vec xin,Vec yin,InsertMode addv,ScatterMode mode)
{
  VecScatter_MPI_General *to,*from;
  PetscScalar            *xv,*yv,*svalues;
  MPI_Request            *rwaits,*swaits;
  PetscErrorCode         ierr;
  PetscInt               i,*indices,*sstarts,nrecvs,nsends,bs;

  PetscFunctionBegin;
  if (mode & SCATTER_REVERSE) {
    to     = (VecScatter_MPI_General*)ctx->fromdata;
    from   = (VecScatter_MPI_General*)ctx->todata;
    rwaits = from->rev_requests;
    swaits = to->rev_requests;
  } else {
    to     = (VecScatter_MPI_General*)ctx->todata;
    from   = (VecScatter_MPI_General*)ctx->fromdata;
    rwaits = from->requests;
    swaits = to->requests;
  }
  bs      = to->bs;
  svalues = to->values;
  nrecvs  = from->n;
  nsends  = to->n;
  indices = to->indices;
  sstarts = to->starts;

  ierr = VecGetArrayRead(xin,(const PetscScalar**)&xv);CHKERRQ(ierr);

  if (xin != yin) {ierr = VecGetArray(yin,&yv);CHKERRQ(ierr);}
  else yv = xv;

  if (!(mode & SCATTER_LOCAL)) {
    /* post receives since they were not previously posted    */
    ierr = MPI_Startall_irecv(from->starts[nrecvs]*bs,MPIU_SCALAR,nrecvs,rwaits);CHKERRQ(ierr);

    /* this version packs and sends one at a time */
    for (i=0; i<nsends; i++) {
      PETSCMAP1(Pack)(sstarts[i+1]-sstarts[i],indices + sstarts[i],xv,svalues + bs*sstarts[i],bs);
      ierr = MPI_Start_isend((sstarts[i+1]-sstarts[i])*bs,MPIU_SCALAR,swaits+i);CHKERRQ(ierr);
    }
  }

  /* take care of local scatters */
  if (to->local.n) {
   if (to->local.memcpy_plan.optimized[0] && addv == INSERT_VALUES) {
      /* do copy when it is not a self-to-self copy */
      if (!(yv == xv && to->local.memcpy_plan.same_copy_starts)) {
        for (i=to->local.memcpy_plan.copy_offsets[0]; i<to->local.memcpy_plan.copy_offsets[1]; i++) {
          /* Do we need to take care of overlaps? We could but overlaps sound more like a bug than a requirement,
             so I just leave it and let PetscMemcpy detect this bug.
           */
          ierr = PetscArraycpy(yv + from->local.memcpy_plan.copy_starts[i],xv + to->local.memcpy_plan.copy_starts[i],to->local.memcpy_plan.copy_lengths[i]);CHKERRQ(ierr);
        }
      }
    } else {
      if (xv == yv && addv == INSERT_VALUES && to->local.nonmatching_computed) {
        /* only copy entries that do not share identical memory locations */
        ierr = PETSCMAP1(Scatter)(to->local.n_nonmatching,to->local.slots_nonmatching,xv,from->local.slots_nonmatching,yv,addv,bs);CHKERRQ(ierr);
      } else {
        ierr = PETSCMAP1(Scatter)(to->local.n,to->local.vslots,xv,from->local.vslots,yv,addv,bs);CHKERRQ(ierr);
      }
    }
  }
  ierr = VecRestoreArrayRead(xin,(const PetscScalar**)&xv);CHKERRQ(ierr);
  if (xin != yin) {ierr = VecRestoreArray(yin,&yv);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

#define PETSC_MEMSHARE_SAFE
PetscErrorCode PETSCMAP1(VecScatterEndMPI3Node)(VecScatter ctx,Vec xin,Vec yin,InsertMode addv,ScatterMode mode)
{
  VecScatter_MPI_General *to,*from;
  PetscScalar            *rvalues,*yv;
  const PetscScalar      *xv;
  PetscErrorCode         ierr;
  PetscInt               nrecvs,nsends,*indices,count,*rstarts,bs;
  PetscMPIInt            imdex;
  MPI_Request            *rwaits,*swaits;
  MPI_Status             xrstatus,*sstatus;
  Vec_Node               *vnode;
  PetscInt               cnt,*idx,*idy;
  MPI_Comm               comm,mscomm,veccomm;
  PetscShmComm           scomm;
  PetscMPIInt            i,xsize;
  PetscInt               k,k1;
  PetscScalar            *sharedspace;

  PetscFunctionBegin;
  if (mode & SCATTER_LOCAL) PetscFunctionReturn(0);

  ierr = PetscObjectGetComm((PetscObject)ctx,&comm);CHKERRQ(ierr);
  ierr = PetscShmCommGet(comm,&scomm);CHKERRQ(ierr);
  ierr = PetscShmCommGetMpiShmComm(scomm,&mscomm);CHKERRQ(ierr);

  ierr = VecGetArray(yin,&yv);CHKERRQ(ierr);

  to      = (VecScatter_MPI_General*)ctx->todata;
  from    = (VecScatter_MPI_General*)ctx->fromdata;
  rwaits  = from->requests;
  swaits  = to->requests;
  sstatus = to->sstatus;    /* sstatus and rstatus are always stored in to */
  if (mode & SCATTER_REVERSE) {
    to     = (VecScatter_MPI_General*)ctx->fromdata;
    from   = (VecScatter_MPI_General*)ctx->todata;
    rwaits = from->rev_requests;
    swaits = to->rev_requests;
  }
  bs      = from->bs;
  rvalues = from->values;
  nrecvs  = from->n;
  nsends  = to->n;
  indices = from->indices;
  rstarts = from->starts;


  /* unpack one at a time */
  count = nrecvs;
  while (count) {
    ierr = MPI_Waitany(nrecvs,rwaits,&imdex,&xrstatus);CHKERRQ(ierr);
    /* unpack receives into our local space */
    ierr = PETSCMAP1(UnPack)(rstarts[imdex+1] - rstarts[imdex],rvalues + bs*rstarts[imdex],indices + rstarts[imdex],yv,addv,bs);CHKERRQ(ierr);
    count--;
  }

  /* handle processes that share the same shared memory communicator */
#if defined(PETSC_MEMSHARE_SAFE)
  ierr = MPI_Barrier(mscomm);CHKERRQ(ierr);
#endif

  /* check if xin is sequential */
  ierr = PetscObjectGetComm((PetscObject)xin,&veccomm);CHKERRQ(ierr);
  ierr = MPI_Comm_size(veccomm,&xsize);CHKERRQ(ierr);

  if (xsize == 1 || from->sharedspace) { /* 'from->sharedspace' indicates this core's shared memory will be written */
    /* StoP: read sequential local xvalues, then write to shared yvalues */
    PetscInt notdone = to->notdone;
    vnode = (Vec_Node*)yin->data;
    if (!vnode->win) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_NULL,"vector y must have type VECNODE with shared memory");
    if (ctx->is_duplicate) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Duplicate index is not supported");
    ierr  = VecGetArrayRead(xin,&xv);CHKERRQ(ierr);

    i = 0;
    while (notdone) {
      while (i < to->msize) {
        if (to->sharedspacesoffset && to->sharedspacesoffset[i] > -1) {
          cnt = to->sharedspacestarts[i+1] - to->sharedspacestarts[i];
          idx = to->sharedspaceindices + to->sharedspacestarts[i];
          idy = idx + to->sharedcnt;

          sharedspace = vnode->winarray[i];

          if (sharedspace[-1] != yv[-1]) {
            if (PetscRealPart(sharedspace[-1] - yv[-1]) > 0.0) {
              PetscMPIInt msrank;
              ierr = MPI_Comm_rank(mscomm,&msrank);CHKERRQ(ierr);
              SETERRQ4(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"[%d] statecnt %g > [%d] my_statecnt %g",i,PetscRealPart(sharedspace[-1]),msrank,PetscRealPart(yv[-1]));
            }
            /* i-the core has not reached the current object statecnt yet, wait ... */
            continue;
          }

          if (addv == ADD_VALUES) {
            for (k= 0; k<cnt; k++) {
              for (k1=0; k1<bs; k1++) sharedspace[idy[k]+k1] += xv[idx[k]+k1];
            }
          } else if (addv == INSERT_VALUES) {
            for (k= 0; k<cnt; k++) {
              for (k1=0; k1<bs; k1++) sharedspace[idy[k]+k1] = xv[idx[k]+k1];
            }
          } else SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Cannot handle insert mode %D", addv);
          notdone--;
        }
        i++;
      }
    }
    ierr = VecRestoreArrayRead(xin,&xv);CHKERRQ(ierr);
  } else {
    /* PtoS: read shared xvalues, then write to sequential local yvalues */
    PetscInt notdone = from->notdone;

    vnode = (Vec_Node*)xin->data;
    if (!vnode->win && notdone) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_NULL,"vector x must have type VECNODE with shared memory");
    ierr  = VecGetArrayRead(xin,&xv);CHKERRQ(ierr);

    i = 0;
    while (notdone) {
      while (i < from->msize) {
        if (from->sharedspacesoffset && from->sharedspacesoffset[i] > -1) {
          cnt = from->sharedspacestarts[i+1] - from->sharedspacestarts[i];
          idy = from->sharedspaceindices + from->sharedspacestarts[i]; /* recv local y indices */
          idx = idy + from->sharedcnt;

          sharedspace = vnode->winarray[i];

          if (sharedspace[-1] != xv[-1]) {
            if (PetscRealPart(sharedspace[-1] - xv[-1]) > 0.0) {
              PetscMPIInt msrank;
              ierr = MPI_Comm_rank(mscomm,&msrank);CHKERRQ(ierr);
              SETERRQ4(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"[%d] statecnt %g > [%d] my_statecnt %g",i,PetscRealPart(sharedspace[-1]),msrank,PetscRealPart(xv[-1]));
            }
            /* i-the core has not reached the current object state cnt yet, wait ... */
            continue;
          }

          if (addv==ADD_VALUES) {
            for (k=0; k<cnt; k++) {
              for (k1=0; k1<bs; k1++) yv[idy[k]+k1] += sharedspace[idx[k]+k1]; /* read x shared values */
            }
          } else if (addv==INSERT_VALUES){
            for (k=0; k<cnt; k++) {
              for (k1=0; k1<bs; k1++) yv[idy[k]+k1] = sharedspace[idx[k]+k1]; /* read x shared values */
            }
          } else SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Cannot handle insert mode %D", addv);

          notdone--;
        }
        i++;
      }
    }
    ierr = VecRestoreArrayRead(xin,&xv);CHKERRQ(ierr);
  }
  /* output y is parallel, ensure it is done -- would lose performance */
  ierr = MPI_Barrier(mscomm);CHKERRQ(ierr);

  /* wait on sends */
  if (nsends) {ierr = MPI_Waitall(nsends,swaits,sstatus);CHKERRQ(ierr);}
  ierr = VecRestoreArray(yin,&yv);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef PETSCMAP1_a
#undef PETSCMAP1_b
#undef PETSCMAP1
#undef BS
