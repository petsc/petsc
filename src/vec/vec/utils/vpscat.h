
/*
     Defines the methods VecScatterBegin/End_1,2,......
     This is included by vpscat.c with different values for BS

     This is a terrible way of doing "templates" in C.
*/
#define PETSCMAP1_a(a,b)  a ## _ ## b
#define PETSCMAP1_b(a,b)  PETSCMAP1_a(a,b)
#define PETSCMAP1(a)      PETSCMAP1_b(a,BS)

#undef __FUNCT__  
#define __FUNCT__ "VecScatterBegin_" PetscStringize(BS)
PetscErrorCode PETSCMAP1(VecScatterBegin)(VecScatter ctx,Vec xin,Vec yin,InsertMode addv,ScatterMode mode)
{
  VecScatter_MPI_General *to,*from;
  PetscScalar            *xv,*yv,*svalues;
  MPI_Request            *rwaits,*swaits;
  PetscErrorCode         ierr;
  PetscInt               i,*indices,*sstarts,nrecvs,nsends,bs;

  PetscFunctionBegin;
  if (mode & SCATTER_REVERSE) {
    to   = (VecScatter_MPI_General*)ctx->fromdata;
    from = (VecScatter_MPI_General*)ctx->todata;
    rwaits   = from->rev_requests;
    swaits   = to->rev_requests;
  } else {
    to   = (VecScatter_MPI_General*)ctx->todata;
    from = (VecScatter_MPI_General*)ctx->fromdata;
    rwaits   = from->requests;
    swaits   = to->requests;
  }
  bs       = to->bs;
  svalues  = to->values;
  nrecvs   = from->n;
  nsends   = to->n;
  indices  = to->indices;
  sstarts  = to->starts;
#if defined(PETSC_HAVE_CUSP)

#if defined(PETSC_HAVE_TXPETSCGPU)

#if 0
  /* This branch messages the entire vector */
  ierr = VecGetArrayRead(xin,(const PetscScalar**)&xv);CHKERRQ(ierr);
#else
  /*
   This branch messages only the parts that are necessary.
   ... this seems to perform about the same due to the necessity of calling
       a separate kernel before the SpMV for gathering data into
       a contiguous buffer. We leaves both branches in for the time being.
       I expect that ultimately this branch will be the right choice, however
       the just is still out.
   */
  if (xin->valid_GPU_array == PETSC_CUSP_GPU) {
    if (xin->spptr && ctx->spptr) 
      ierr = VecCUSPCopyFromGPUSome_Public(xin,(PetscCUSPIndices)ctx->spptr);CHKERRQ(ierr);
  }
#endif
  xv   = *(PetscScalar**)xin->data;

#else
  if (!xin->map->n || ((xin->map->n > 10000) && (sstarts[nsends]*bs < 0.05*xin->map->n) && (xin->valid_GPU_array == PETSC_CUSP_GPU) && !(to->local.n))) {
    if (!ctx->spptr) {
      PetscInt k,*tindices,n = sstarts[nsends],*sindices;
      ierr = PetscMalloc(n*sizeof(PetscInt),&tindices);CHKERRQ(ierr);
      ierr = PetscMemcpy(tindices,to->indices,n*sizeof(PetscInt));CHKERRQ(ierr);
      ierr = PetscSortRemoveDupsInt(&n,tindices);CHKERRQ(ierr);
      ierr = PetscMalloc(bs*n*sizeof(PetscInt),&sindices);CHKERRQ(ierr);
      for (i=0; i<n; i++) {
        for (k=0; k<bs; k++) {
          sindices[i*bs+k] = tindices[i]+k;
        }
      }
      ierr = PetscFree(tindices);CHKERRQ(ierr);
      ierr = PetscCUSPIndicesCreate(n*bs,sindices,n*bs,sindices,(PetscCUSPIndices*)&ctx->spptr);CHKERRQ(ierr);
      ierr = PetscFree(sindices);CHKERRQ(ierr);
    }
    ierr = VecCUSPCopyFromGPUSome_Public(xin,(PetscCUSPIndices)ctx->spptr);CHKERRQ(ierr);
    xv   = *(PetscScalar**)xin->data;
    } else {
    ierr = VecGetArrayRead(xin,(const PetscScalar**)&xv);CHKERRQ(ierr);
  }
#endif

#else
  ierr = VecGetArrayRead(xin,(const PetscScalar**)&xv);CHKERRQ(ierr);
#endif

  if (xin != yin) {ierr = VecGetArray(yin,&yv);CHKERRQ(ierr);} else {yv = xv;}

  if (!(mode & SCATTER_LOCAL)) {
    if (!from->use_readyreceiver && !to->sendfirst && !to->use_alltoallv  & !to->use_window) {  
      /* post receives since they were not previously posted    */
      if (nrecvs) {ierr = MPI_Startall_irecv(from->starts[nrecvs]*bs,nrecvs,rwaits);CHKERRQ(ierr);}
    }

#if defined(PETSC_HAVE_MPI_ALLTOALLW)  && !defined(PETSC_USE_64BIT_INDICES)
    if (to->use_alltoallw && addv == INSERT_VALUES) {
      ierr = MPI_Alltoallw(xv,to->wcounts,to->wdispls,to->types,yv,from->wcounts,from->wdispls,from->types,((PetscObject)ctx)->comm);CHKERRQ(ierr);
    } else
#endif
    if (ctx->packtogether || to->use_alltoallv || to->use_window) {
      /* this version packs all the messages together and sends, when -vecscatter_packtogether used */
      PETSCMAP1(Pack)(sstarts[nsends],indices,xv,svalues);
      if (to->use_alltoallv) {
        ierr = MPI_Alltoallv(to->values,to->counts,to->displs,MPIU_SCALAR,from->values,from->counts,from->displs,MPIU_SCALAR,((PetscObject)ctx)->comm);CHKERRQ(ierr);
#if defined(PETSC_HAVE_MPI_WIN_CREATE)
      } else if (to->use_window) {
        PetscInt cnt;

        ierr = MPI_Win_fence(0,from->window);CHKERRQ(ierr);
        for (i=0; i<nsends; i++) {
          cnt  = bs*(to->starts[i+1]-to->starts[i]);
          ierr = MPI_Put(to->values+bs*to->starts[i],cnt,MPIU_SCALAR,to->procs[i],bs*to->winstarts[i],cnt,MPIU_SCALAR,from->window);CHKERRQ(ierr);
        }
#endif
      } else if (nsends) {
        ierr = MPI_Startall_isend(to->starts[to->n],nsends,swaits);CHKERRQ(ierr);
      }
    } else {
      /* this version packs and sends one at a time */
      for (i=0; i<nsends; i++) {
        PETSCMAP1(Pack)(sstarts[i+1]-sstarts[i],indices + sstarts[i],xv,svalues + bs*sstarts[i]);
        ierr = MPI_Start_isend(sstarts[i+1]-sstarts[i],swaits+i);CHKERRQ(ierr);
      }
    }

    if (!from->use_readyreceiver && to->sendfirst && !to->use_alltoallv && !to->use_window) {  
      /* post receives since they were not previously posted   */
      if (nrecvs) {ierr = MPI_Startall_irecv(from->starts[nrecvs]*bs,nrecvs,rwaits);CHKERRQ(ierr);}
    }
  }

  /* take care of local scatters */
  if (to->local.n) {
    if (to->local.is_copy && addv == INSERT_VALUES) {
      ierr = PetscMemcpy(yv + from->local.copy_start,xv + to->local.copy_start,to->local.copy_length);CHKERRQ(ierr);
    } else {
      PETSCMAP1(Scatter)(to->local.n,to->local.vslots,xv,from->local.vslots,yv,addv);
    }
  }
#if defined(PETSC_HAVE_CUSP)
  if (xin->valid_GPU_array != PETSC_CUSP_GPU) {
    ierr = VecRestoreArrayRead(xin,(const PetscScalar**)&xv);CHKERRQ(ierr);
  } else {
    ierr = VecRestoreArrayRead(xin,(const PetscScalar**)&xv);CHKERRQ(ierr);
  }
#else
  ierr = VecRestoreArrayRead(xin,(const PetscScalar**)&xv);CHKERRQ(ierr);
#endif
  if (xin != yin) {ierr = VecRestoreArray(yin,&yv);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

/* --------------------------------------------------------------------------------------*/

#undef __FUNCT__  
#define __FUNCT__ "VecScatterEnd_" PetscStringize(BS)
PetscErrorCode PETSCMAP1(VecScatterEnd)(VecScatter ctx,Vec xin,Vec yin,InsertMode addv,ScatterMode mode)
{
  VecScatter_MPI_General *to,*from;
  PetscScalar            *rvalues,*yv;
  PetscErrorCode         ierr;
  PetscInt               nrecvs,nsends,*indices,count,*rstarts,bs;
  PetscMPIInt            imdex;
  MPI_Request            *rwaits,*swaits;
  MPI_Status             xrstatus,*rstatus,*sstatus;

  PetscFunctionBegin;
  if (mode & SCATTER_LOCAL) PetscFunctionReturn(0);
  ierr = VecGetArray(yin,&yv);CHKERRQ(ierr);

  to       = (VecScatter_MPI_General*)ctx->todata;
  from     = (VecScatter_MPI_General*)ctx->fromdata;
  rwaits   = from->requests;
  swaits   = to->requests;
  sstatus  = to->sstatus;   /* sstatus and rstatus are always stored in to */
  rstatus  = to->rstatus;
  if (mode & SCATTER_REVERSE) {
    to       = (VecScatter_MPI_General*)ctx->fromdata;
    from     = (VecScatter_MPI_General*)ctx->todata;
    rwaits   = from->rev_requests;
    swaits   = to->rev_requests;
  }
  bs       = from->bs;
  rvalues  = from->values;
  nrecvs   = from->n;
  nsends   = to->n;
  indices  = from->indices;
  rstarts  = from->starts;

  if (ctx->packtogether || (to->use_alltoallw && (addv != INSERT_VALUES)) || (to->use_alltoallv && !to->use_alltoallw) || to->use_window) {
#if defined(PETSC_HAVE_MPI_WIN_CREATE)
    if (to->use_window) {ierr = MPI_Win_fence(0,from->window);CHKERRQ(ierr);}
    else
#endif
    if (nrecvs && !to->use_alltoallv) {ierr = MPI_Waitall(nrecvs,rwaits,rstatus);CHKERRQ(ierr);}
    PETSCMAP1(UnPack)(from->starts[from->n],from->values,indices,yv,addv);
  } else if (!to->use_alltoallw) {
    /* unpack one at a time */
    count = nrecvs;
    while (count) {
      if (ctx->reproduce) {
	imdex = count - 1;
	ierr = MPI_Wait(rwaits+imdex,&xrstatus);CHKERRQ(ierr);
      } else {
	ierr = MPI_Waitany(nrecvs,rwaits,&imdex,&xrstatus);CHKERRQ(ierr);
      }
      /* unpack receives into our local space */
      PETSCMAP1(UnPack)(rstarts[imdex+1] - rstarts[imdex],rvalues + bs*rstarts[imdex],indices + rstarts[imdex],yv,addv);
      count--;
    }
  }
  if (from->use_readyreceiver) {  
    if (nrecvs) {ierr = MPI_Startall_irecv(from->starts[nrecvs]*bs,nrecvs,rwaits);CHKERRQ(ierr);}
    ierr = MPI_Barrier(((PetscObject)ctx)->comm);CHKERRQ(ierr);
  }

  /* wait on sends */
  if (nsends  && !to->use_alltoallv  && !to->use_window) {ierr = MPI_Waitall(nsends,swaits,sstatus);CHKERRQ(ierr);}
  ierr = VecRestoreArray(yin,&yv);CHKERRQ(ierr);
#if defined(PETSC_HAVE_TXPETSCGPU)
  if (yin->valid_GPU_array == PETSC_CUSP_CPU) {
    if (yin->spptr && ctx->spptr) 
      ierr = VecCUSPCopyToGPUSome_Public(yin,(PetscCUSPIndices)ctx->spptr);CHKERRQ(ierr);
  }
#endif
  PetscFunctionReturn(0);
}

#undef PETSCMAP1_a
#undef PETSCMAP1_b
#undef PETSCMAP1
#undef BS

