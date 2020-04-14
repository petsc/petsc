/*
    Defines parallel vector scatters.
*/

#include <petsc/private/vecscatterimpl.h>


PetscErrorCode VecScatterView_MPI(VecScatter ctx,PetscViewer viewer)
{
  VecScatter_MPI_General *to  =(VecScatter_MPI_General*)ctx->todata;
  VecScatter_MPI_General *from=(VecScatter_MPI_General*)ctx->fromdata;
  PetscErrorCode         ierr;
  PetscInt               i;
  PetscMPIInt            rank;
  PetscViewerFormat      format;
  PetscBool              iascii;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    ierr = MPI_Comm_rank(PetscObjectComm((PetscObject)ctx),&rank);CHKERRQ(ierr);
    ierr = PetscViewerGetFormat(viewer,&format);CHKERRQ(ierr);
    if (format ==  PETSC_VIEWER_ASCII_INFO) {
      PetscInt nsend_max,nrecv_max,lensend_max,lenrecv_max,alldata,itmp;

      ierr = MPI_Reduce(&to->n,&nsend_max,1,MPIU_INT,MPI_MAX,0,PetscObjectComm((PetscObject)ctx));CHKERRQ(ierr);
      ierr = MPI_Reduce(&from->n,&nrecv_max,1,MPIU_INT,MPI_MAX,0,PetscObjectComm((PetscObject)ctx));CHKERRQ(ierr);
      itmp = to->starts[to->n+1];
      ierr = MPI_Reduce(&itmp,&lensend_max,1,MPIU_INT,MPI_MAX,0,PetscObjectComm((PetscObject)ctx));CHKERRQ(ierr);
      itmp = from->starts[from->n+1];
      ierr = MPI_Reduce(&itmp,&lenrecv_max,1,MPIU_INT,MPI_MAX,0,PetscObjectComm((PetscObject)ctx));CHKERRQ(ierr);
      ierr = MPI_Reduce(&itmp,&alldata,1,MPIU_INT,MPI_SUM,0,PetscObjectComm((PetscObject)ctx));CHKERRQ(ierr);

      ierr = PetscViewerASCIIPrintf(viewer,"VecScatter statistics\n");CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"  Blocksize %D\n",to->bs);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"  Maximum number sends %D\n",nsend_max);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"  Maximum number receives %D\n",nrecv_max);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"  Maximum data sent %D\n",(int)(lensend_max*to->bs*sizeof(PetscScalar)));CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"  Maximum data received %D\n",(int)(lenrecv_max*to->bs*sizeof(PetscScalar)));CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"  Total data sent %D\n",(int)(alldata*to->bs*sizeof(PetscScalar)));CHKERRQ(ierr);
    } else {
      ierr = PetscViewerASCIIPrintf(viewer,"  VecScatter Blocksize %D\n",to->bs);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPushSynchronized(viewer);CHKERRQ(ierr);
      ierr = PetscViewerASCIISynchronizedPrintf(viewer,"[%d] Number sends = %D; Number to self = %D\n",rank,to->n,to->local.n);CHKERRQ(ierr);
      if (to->n) {
        for (i=0; i<to->n; i++) {
          ierr = PetscViewerASCIISynchronizedPrintf(viewer,"[%d]   %D length = %D to whom %d\n",rank,i,to->starts[i+1]-to->starts[i],to->procs[i]);CHKERRQ(ierr);
        }
        ierr = PetscViewerASCIISynchronizedPrintf(viewer,"[%d] Now the indices for all remote sends (in order by process sent to)\n",rank);CHKERRQ(ierr);
        for (i=0; i<to->starts[to->n]; i++) {
          ierr = PetscViewerASCIISynchronizedPrintf(viewer,"[%d] %D \n",rank,to->indices[i]);CHKERRQ(ierr);
        }
      }

      ierr = PetscViewerASCIISynchronizedPrintf(viewer,"[%d] Number receives = %D; Number from self = %D\n",rank,from->n,from->local.n);CHKERRQ(ierr);
      if (from->n) {
        for (i=0; i<from->n; i++) {
          ierr = PetscViewerASCIISynchronizedPrintf(viewer,"[%d] %D length %D from whom %d\n",rank,i,from->starts[i+1]-from->starts[i],from->procs[i]);CHKERRQ(ierr);
        }

        ierr = PetscViewerASCIISynchronizedPrintf(viewer,"[%d] Now the indices for all remote receives (in order by process received from)\n",rank);CHKERRQ(ierr);
        for (i=0; i<from->starts[from->n]; i++) {
          ierr = PetscViewerASCIISynchronizedPrintf(viewer,"[%d] %D \n",rank,from->indices[i]);CHKERRQ(ierr);
        }
      }
      if (to->local.n) {
        ierr = PetscViewerASCIISynchronizedPrintf(viewer,"[%d] Indices for local part of scatter\n",rank);CHKERRQ(ierr);
        for (i=0; i<to->local.n; i++) {  /* the to and from have the opposite meaning from what you would expect */
          ierr = PetscViewerASCIISynchronizedPrintf(viewer,"[%d] From %D to %D \n",rank,to->local.vslots[i],from->local.vslots[i]);CHKERRQ(ierr);
        }
      }

     for (i=0; i<to->msize; i++) {
       if (to->sharedspacestarts[i+1] > to->sharedspacestarts[i]) {
          ierr = PetscViewerASCIISynchronizedPrintf(viewer,"[%d] Via shared memory to local memory partner %d count %d\n",rank,i,to->sharedspacestarts[i+1]-to->sharedspacestarts[i]);CHKERRQ(ierr);
        }
      }
     for (i=0; i<from->msize; i++) {
       if (from->sharedspacestarts[i+1] > from->sharedspacestarts[i]) {
          ierr = PetscViewerASCIISynchronizedPrintf(viewer,"[%d] Via shared memory from local memory partner %d count %d\n",rank,i,from->sharedspacestarts[i+1]-from->sharedspacestarts[i]);CHKERRQ(ierr);
        }
      }

      ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPopSynchronized(viewer);CHKERRQ(ierr);

      ierr = PetscViewerASCIIPrintf(viewer,"Method used to implement the VecScatter: ");CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

/* -----------------------------------------------------------------------------------*/
/*
      The next routine determines what part of  the local part of the scatter is an
  exact copy of values into their current location. We check this here and
  then know that we need not perform that portion of the scatter when the vector is
  scattering to itself with INSERT_VALUES.

     This is currently not used but would speed up, for example DMLocalToLocalBegin/End()

*/
PetscErrorCode VecScatterLocalOptimize_Private(VecScatter scatter,VecScatter_Seq_General *to,VecScatter_Seq_General *from)
{
  PetscInt       n = to->n,n_nonmatching = 0,i,*to_slots = to->vslots,*from_slots = from->vslots;
  PetscErrorCode ierr;
  PetscInt       *nto_slots,*nfrom_slots,j = 0;

  PetscFunctionBegin;
  for (i=0; i<n; i++) {
    if (to_slots[i] != from_slots[i]) n_nonmatching++;
  }

  if (!n_nonmatching) {
    to->nonmatching_computed = PETSC_TRUE;
    to->n_nonmatching        = from->n_nonmatching = 0;
    ierr = PetscInfo1(scatter,"Reduced %D to 0\n", n);CHKERRQ(ierr);
  } else if (n_nonmatching == n) {
    to->nonmatching_computed = PETSC_FALSE;
    ierr = PetscInfo(scatter,"All values non-matching\n");CHKERRQ(ierr);
  } else {
    to->nonmatching_computed= PETSC_TRUE;
    to->n_nonmatching       = from->n_nonmatching = n_nonmatching;

    ierr = PetscMalloc1(n_nonmatching,&nto_slots);CHKERRQ(ierr);
    ierr = PetscMalloc1(n_nonmatching,&nfrom_slots);CHKERRQ(ierr);

    to->slots_nonmatching   = nto_slots;
    from->slots_nonmatching = nfrom_slots;
    for (i=0; i<n; i++) {
      if (to_slots[i] != from_slots[i]) {
        nto_slots[j]   = to_slots[i];
        nfrom_slots[j] = from_slots[i];
        j++;
      }
    }
    ierr = PetscInfo2(scatter,"Reduced %D to %D\n",n,n_nonmatching);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/* --------------------------------------------------------------------------------------*/

/* -------------------------------------------------------------------------------------*/
PetscErrorCode VecScatterDestroy_PtoP_MPI3(VecScatter ctx)
{
  VecScatter_MPI_General *to   = (VecScatter_MPI_General*)ctx->todata;
  VecScatter_MPI_General *from = (VecScatter_MPI_General*)ctx->fromdata;
  PetscErrorCode         ierr;
  PetscInt               i;

  PetscFunctionBegin;
  /* release MPI resources obtained with MPI_Send_init() and MPI_Recv_init() */
  if (to->requests) {
    for (i=0; i<to->n; i++) {
      ierr = MPI_Request_free(to->requests + i);CHKERRQ(ierr);
    }
  }
  if (to->rev_requests) {
    for (i=0; i<to->n; i++) {
      ierr = MPI_Request_free(to->rev_requests + i);CHKERRQ(ierr);
    }
  }
  if (from->requests) {
    for (i=0; i<from->n; i++) {
      ierr = MPI_Request_free(from->requests + i);CHKERRQ(ierr);
    }
  }
  if (from->rev_requests) {
    for (i=0; i<from->n; i++) {
      ierr = MPI_Request_free(from->rev_requests + i);CHKERRQ(ierr);
    }
  }
  if (to->sharedwin != MPI_WIN_NULL) {ierr = MPI_Win_free(&to->sharedwin);CHKERRQ(ierr);}
  if (from->sharedwin != MPI_WIN_NULL) {ierr = MPI_Win_free(&from->sharedwin);CHKERRQ(ierr);}
  ierr = PetscFree(to->sharedspaces);CHKERRQ(ierr);
  ierr = PetscFree(to->sharedspacesoffset);CHKERRQ(ierr);
  ierr = PetscFree(to->sharedspaceindices);CHKERRQ(ierr);
  ierr = PetscFree(to->sharedspacestarts);CHKERRQ(ierr);

  ierr = PetscFree(from->sharedspaceindices);CHKERRQ(ierr);
  ierr = PetscFree(from->sharedspaces);CHKERRQ(ierr);
  ierr = PetscFree(from->sharedspacesoffset);CHKERRQ(ierr);
  ierr = PetscFree(from->sharedspacestarts);CHKERRQ(ierr);

  ierr = PetscFree(to->local.vslots);CHKERRQ(ierr);
  ierr = PetscFree(from->local.vslots);CHKERRQ(ierr);
  ierr = PetscFree(to->local.slots_nonmatching);CHKERRQ(ierr);
  ierr = PetscFree(from->local.slots_nonmatching);CHKERRQ(ierr);
  ierr = PetscFree(to->rev_requests);CHKERRQ(ierr);
  ierr = PetscFree(from->rev_requests);CHKERRQ(ierr);
  ierr = PetscFree(to->requests);CHKERRQ(ierr);
  ierr = PetscFree(from->requests);CHKERRQ(ierr);
  ierr = PetscFree4(to->values,to->indices,to->starts,to->procs);CHKERRQ(ierr);
  ierr = PetscFree2(to->sstatus,to->rstatus);CHKERRQ(ierr);
  ierr = PetscFree4(from->values,from->indices,from->starts,from->procs);CHKERRQ(ierr);
  ierr = VecScatterMemcpyPlanDestroy_PtoP(to,from);CHKERRQ(ierr);
  ierr = PetscFree(from);CHKERRQ(ierr);
  ierr = PetscFree(to);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* --------------------------------------------------------------------------------------*/

PetscErrorCode VecScatterCopy_PtoP_X(VecScatter in,VecScatter out)
{
  VecScatter_MPI_General *in_to   = (VecScatter_MPI_General*)in->todata;
  VecScatter_MPI_General *in_from = (VecScatter_MPI_General*)in->fromdata,*out_to,*out_from;
  PetscErrorCode         ierr;
  PetscInt               ny,bs = in_from->bs,jj;
  PetscShmComm           scomm;
  MPI_Comm               mscomm;
  MPI_Info               info;

  PetscFunctionBegin;
  out->ops->begin   = in->ops->begin;
  out->ops->end     = in->ops->end;
  out->ops->copy    = in->ops->copy;
  out->ops->destroy = in->ops->destroy;
  out->ops->view    = in->ops->view;

  /* allocate entire send scatter context */
  ierr = PetscNewLog(out,&out_to);CHKERRQ(ierr);
  out_to->sharedwin       = MPI_WIN_NULL;
  ierr = PetscNewLog(out,&out_from);CHKERRQ(ierr);
  out_from->sharedwin       = MPI_WIN_NULL;

  ny                = in_to->starts[in_to->n];
  out_to->n         = in_to->n;
  out_to->format    = in_to->format;

  ierr = PetscMalloc1(out_to->n,&out_to->requests);CHKERRQ(ierr);
  ierr = PetscMalloc4(bs*ny,&out_to->values,ny,&out_to->indices,out_to->n+1,&out_to->starts,out_to->n,&out_to->procs);CHKERRQ(ierr);
  ierr = PetscMalloc2(PetscMax(in_to->n,in_from->n),&out_to->sstatus,PetscMax(in_to->n,in_from->n),&out_to->rstatus);CHKERRQ(ierr);
  ierr = PetscArraycpy(out_to->indices,in_to->indices,ny);CHKERRQ(ierr);
  ierr = PetscArraycpy(out_to->starts,in_to->starts,out_to->n+1);CHKERRQ(ierr);
  ierr = PetscArraycpy(out_to->procs,in_to->procs,out_to->n);CHKERRQ(ierr);

  out->todata                        = (void*)out_to;
  out_to->local.n                    = in_to->local.n;
  out_to->local.nonmatching_computed = PETSC_FALSE;
  out_to->local.n_nonmatching        = 0;
  out_to->local.slots_nonmatching    = 0;
  if (in_to->local.n) {
    ierr = PetscMalloc1(in_to->local.n,&out_to->local.vslots);CHKERRQ(ierr);
    ierr = PetscMalloc1(in_from->local.n,&out_from->local.vslots);CHKERRQ(ierr);
    ierr = PetscArraycpy(out_to->local.vslots,in_to->local.vslots,in_to->local.n);CHKERRQ(ierr);
    ierr = PetscArraycpy(out_from->local.vslots,in_from->local.vslots,in_from->local.n);CHKERRQ(ierr);
  } else {
    out_to->local.vslots   = 0;
    out_from->local.vslots = 0;
  }

  /* allocate entire receive context */
  ierr = VecScatterMemcpyPlanCopy(&in_to->local.memcpy_plan,&out_to->local.memcpy_plan);CHKERRQ(ierr);
  ierr = VecScatterMemcpyPlanCopy(&in_from->local.memcpy_plan,&out_from->local.memcpy_plan);CHKERRQ(ierr);
  ierr = VecScatterMemcpyPlanCopy(&in_to->memcpy_plan,&out_to->memcpy_plan);CHKERRQ(ierr);
  ierr = VecScatterMemcpyPlanCopy(&in_from->memcpy_plan,&out_from->memcpy_plan);CHKERRQ(ierr);

  out_from->format    = in_from->format;
  ny                  = in_from->starts[in_from->n];
  out_from->n         = in_from->n;

  ierr = PetscMalloc1(out_from->n,&out_from->requests);CHKERRQ(ierr);
  ierr = PetscMalloc4(ny*bs,&out_from->values,ny,&out_from->indices,out_from->n+1,&out_from->starts,out_from->n,&out_from->procs);CHKERRQ(ierr);
  ierr = PetscArraycpy(out_from->indices,in_from->indices,ny);CHKERRQ(ierr);
  ierr = PetscArraycpy(out_from->starts,in_from->starts,out_from->n+1);CHKERRQ(ierr);
  ierr = PetscArraycpy(out_from->procs,in_from->procs,out_from->n);CHKERRQ(ierr);

  out->fromdata                        = (void*)out_from;
  out_from->local.n                    = in_from->local.n;
  out_from->local.nonmatching_computed = PETSC_FALSE;
  out_from->local.n_nonmatching        = 0;
  out_from->local.slots_nonmatching    = 0;

  /*
      set up the request arrays for use with isend_init() and irecv_init()
  */
  {
    PetscMPIInt tag;
    MPI_Comm    comm;
    PetscInt    *sstarts = out_to->starts,  *rstarts = out_from->starts;
    PetscMPIInt *sprocs  = out_to->procs,   *rprocs  = out_from->procs;
    PetscInt    i;
    MPI_Request *swaits   = out_to->requests,*rwaits  = out_from->requests;
    MPI_Request *rev_swaits,*rev_rwaits;
    PetscScalar *Ssvalues = out_to->values, *Srvalues = out_from->values;

    ierr = PetscMalloc1(in_to->n,&out_to->rev_requests);CHKERRQ(ierr);
    ierr = PetscMalloc1(in_from->n,&out_from->rev_requests);CHKERRQ(ierr);

    rev_rwaits = out_to->rev_requests;
    rev_swaits = out_from->rev_requests;

    out_from->bs = out_to->bs = bs;
    tag          = ((PetscObject)out)->tag;
    ierr         = PetscObjectGetComm((PetscObject)out,&comm);CHKERRQ(ierr);

    /* Register the receives that you will use later (sends for scatter reverse) */
    for (i=0; i<out_from->n; i++) {
      ierr = MPI_Recv_init(Srvalues+bs*rstarts[i],bs*rstarts[i+1]-bs*rstarts[i],MPIU_SCALAR,rprocs[i],tag,comm,rwaits+i);CHKERRQ(ierr);
      ierr = MPI_Send_init(Srvalues+bs*rstarts[i],bs*rstarts[i+1]-bs*rstarts[i],MPIU_SCALAR,rprocs[i],tag,comm,rev_swaits+i);CHKERRQ(ierr);
    }

    for (i=0; i<out_to->n; i++) {
      ierr = MPI_Send_init(Ssvalues+bs*sstarts[i],bs*sstarts[i+1]-bs*sstarts[i],MPIU_SCALAR,sprocs[i],tag,comm,swaits+i);CHKERRQ(ierr);
    }

    /* Register receives for scatter reverse */
    for (i=0; i<out_to->n; i++) {
      ierr = MPI_Recv_init(Ssvalues+bs*sstarts[i],bs*sstarts[i+1]-bs*sstarts[i],MPIU_SCALAR,sprocs[i],tag,comm,rev_rwaits+i);CHKERRQ(ierr);
    }
  }

    /* since the to and from data structures are not symmetric for shared memory copies we insure they always listed in "standard" form */
  if (!in_to->sharedwin) {
    VecScatter_MPI_General *totmp = in_to,*out_totmp = out_to;
    in_to       = in_from;
    in_from     = totmp;
    out_to      = out_from;
    out_from    = out_totmp;
  }

  /* copy the to parts for the shared memory copies between processes */
  out_to->sharedcnt = in_to->sharedcnt;
  out_to->msize     = in_to->msize;
  ierr = PetscMalloc1(out_to->msize+1,&out_to->sharedspacestarts);CHKERRQ(ierr);
  ierr = PetscArraycpy(out_to->sharedspacestarts,in_to->sharedspacestarts,out_to->msize+1);CHKERRQ(ierr);
  ierr = PetscMalloc1(out_to->sharedcnt,&out_to->sharedspaceindices);CHKERRQ(ierr);
  ierr = PetscArraycpy(out_to->sharedspaceindices,in_to->sharedspaceindices,out_to->sharedcnt);CHKERRQ(ierr);

  ierr = PetscShmCommGet(PetscObjectComm((PetscObject)in),&scomm);CHKERRQ(ierr);
  ierr = PetscShmCommGetMpiShmComm(scomm,&mscomm);CHKERRQ(ierr);
  ierr = MPI_Info_create(&info);CHKERRQ(ierr);
  ierr = MPI_Info_set(info, "alloc_shared_noncontig", "true");CHKERRQ(ierr);
  ierr = MPIU_Win_allocate_shared(bs*out_to->sharedcnt*sizeof(PetscScalar),sizeof(PetscScalar),info,mscomm,&out_to->sharedspace,&out_to->sharedwin);CHKERRQ(ierr);
  ierr = MPI_Info_free(&info);CHKERRQ(ierr);

  /* copy the to parts for the shared memory copies between processes */
  out_from->sharedcnt = in_from->sharedcnt;
  out_from->msize     = in_from->msize;
  ierr = PetscMalloc1(out_from->msize+1,&out_from->sharedspacestarts);CHKERRQ(ierr);
  ierr = PetscArraycpy(out_from->sharedspacestarts,in_from->sharedspacestarts,out_from->msize+1);CHKERRQ(ierr);
  ierr = PetscMalloc1(out_from->sharedcnt,&out_from->sharedspaceindices);CHKERRQ(ierr);
  ierr = PetscArraycpy(out_from->sharedspaceindices,in_from->sharedspaceindices,out_from->sharedcnt);CHKERRQ(ierr);
  ierr = PetscMalloc1(out_from->msize,&out_from->sharedspacesoffset);CHKERRQ(ierr);
  ierr = PetscArraycpy(out_from->sharedspacesoffset,in_from->sharedspacesoffset,out_from->msize);CHKERRQ(ierr);
  ierr = PetscCalloc1(out_from->msize,&out_from->sharedspaces);CHKERRQ(ierr);
  for (jj=0; jj<out_from->msize; jj++) {
    MPI_Aint    isize;
    PetscMPIInt disp_unit;
    ierr = MPIU_Win_shared_query(out_to->sharedwin,jj,&isize,&disp_unit,&out_from->sharedspaces[jj]);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode VecScatterCopy_PtoP_AllToAll(VecScatter in,VecScatter out)
{
  VecScatter_MPI_General *in_to   = (VecScatter_MPI_General*)in->todata;
  VecScatter_MPI_General *in_from = (VecScatter_MPI_General*)in->fromdata,*out_to,*out_from;
  PetscErrorCode         ierr;
  PetscInt               ny,bs = in_from->bs;
  PetscMPIInt            size;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(PetscObjectComm((PetscObject)in),&size);CHKERRQ(ierr);

  out->ops->begin     = in->ops->begin;
  out->ops->end       = in->ops->end;
  out->ops->copy      = in->ops->copy;
  out->ops->destroy   = in->ops->destroy;
  out->ops->view      = in->ops->view;

  /* allocate entire send scatter context */
  ierr = PetscNewLog(out,&out_to);CHKERRQ(ierr);
  out_to->sharedwin       = MPI_WIN_NULL;
  ierr = PetscNewLog(out,&out_from);CHKERRQ(ierr);
  out_from->sharedwin       = MPI_WIN_NULL;

  ny                = in_to->starts[in_to->n];
  out_to->n         = in_to->n;
  out_to->format    = in_to->format;

  ierr = PetscMalloc1(out_to->n,&out_to->requests);CHKERRQ(ierr);
  ierr = PetscMalloc4(bs*ny,&out_to->values,ny,&out_to->indices,out_to->n+1,&out_to->starts,out_to->n,&out_to->procs);CHKERRQ(ierr);
  ierr = PetscMalloc2(PetscMax(in_to->n,in_from->n),&out_to->sstatus,PetscMax(in_to->n,in_from->n),&out_to->rstatus);CHKERRQ(ierr);
  ierr = PetscArraycpy(out_to->indices,in_to->indices,ny);CHKERRQ(ierr);
  ierr = PetscArraycpy(out_to->starts,in_to->starts,out_to->n+1);CHKERRQ(ierr);
  ierr = PetscArraycpy(out_to->procs,in_to->procs,out_to->n);CHKERRQ(ierr);

  out->todata                        = (void*)out_to;
  out_to->local.n                    = in_to->local.n;
  out_to->local.nonmatching_computed = PETSC_FALSE;
  out_to->local.n_nonmatching        = 0;
  out_to->local.slots_nonmatching    = 0;
  if (in_to->local.n) {
    ierr = PetscMalloc1(in_to->local.n,&out_to->local.vslots);CHKERRQ(ierr);
    ierr = PetscMalloc1(in_from->local.n,&out_from->local.vslots);CHKERRQ(ierr);
    ierr = PetscArraycpy(out_to->local.vslots,in_to->local.vslots,in_to->local.n);CHKERRQ(ierr);
    ierr = PetscArraycpy(out_from->local.vslots,in_from->local.vslots,in_from->local.n);CHKERRQ(ierr);
  } else {
    out_to->local.vslots   = 0;
    out_from->local.vslots = 0;
  }

  /* allocate entire receive context */
  ierr = VecScatterMemcpyPlanCopy_PtoP(in_to,in_from,out_to,out_from);CHKERRQ(ierr);

  out_from->format    = in_from->format;
  ny                  = in_from->starts[in_from->n];
  out_from->n         = in_from->n;

  ierr = PetscMalloc1(out_from->n,&out_from->requests);CHKERRQ(ierr);
  ierr = PetscMalloc4(ny*bs,&out_from->values,ny,&out_from->indices,out_from->n+1,&out_from->starts,out_from->n,&out_from->procs);CHKERRQ(ierr);
  ierr = PetscArraycpy(out_from->indices,in_from->indices,ny);CHKERRQ(ierr);
  ierr = PetscArraycpy(out_from->starts,in_from->starts,out_from->n+1);CHKERRQ(ierr);
  ierr = PetscArraycpy(out_from->procs,in_from->procs,out_from->n);CHKERRQ(ierr);

  out->fromdata                        = (void*)out_from;
  out_from->local.n                    = in_from->local.n;
  out_from->local.nonmatching_computed = PETSC_FALSE;
  out_from->local.n_nonmatching        = 0;
  out_from->local.slots_nonmatching    = 0;

  PetscFunctionReturn(0);
}
/* --------------------------------------------------------------------------------------------------
    Packs and unpacks the message data into send or from receive buffers.

    These could be generated automatically.

    Fortran kernels etc. could be used.
*/
PETSC_STATIC_INLINE void Pack_1(PetscInt n,const PetscInt *indicesx,const PetscScalar *x,PetscScalar *y,PetscInt bs)
{
  PetscInt i;
  for (i=0; i<n; i++) y[i] = x[indicesx[i]];
}

PETSC_STATIC_INLINE PetscErrorCode UnPack_1(PetscInt n,const PetscScalar *x,const PetscInt *indicesy,PetscScalar *y,InsertMode addv,PetscInt bs)
{
  PetscInt i;

  PetscFunctionBegin;
  switch (addv) {
  case INSERT_VALUES:
  case INSERT_ALL_VALUES:
    for (i=0; i<n; i++) y[indicesy[i]] = x[i];
    break;
  case ADD_VALUES:
  case ADD_ALL_VALUES:
    for (i=0; i<n; i++) y[indicesy[i]] += x[i];
    break;
#if !defined(PETSC_USE_COMPLEX)
  case MAX_VALUES:
    for (i=0; i<n; i++) y[indicesy[i]] = PetscMax(y[indicesy[i]],x[i]);
#else
  case MAX_VALUES:
#endif
  case NOT_SET_VALUES:
    break;
  default:
    SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Cannot handle insert mode %d", addv);
  }
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode Scatter_1(PetscInt n,const PetscInt *indicesx,const PetscScalar *x,const PetscInt *indicesy,PetscScalar *y,InsertMode addv,PetscInt bs)
{
  PetscInt i;

  PetscFunctionBegin;
  switch (addv) {
  case INSERT_VALUES:
  case INSERT_ALL_VALUES:
    for (i=0; i<n; i++) y[indicesy[i]] = x[indicesx[i]];
    break;
  case ADD_VALUES:
  case ADD_ALL_VALUES:
    for (i=0; i<n; i++) y[indicesy[i]] += x[indicesx[i]];
    break;
#if !defined(PETSC_USE_COMPLEX)
  case MAX_VALUES:
    for (i=0; i<n; i++) y[indicesy[i]] = PetscMax(y[indicesy[i]],x[indicesx[i]]);
#else
  case MAX_VALUES:
#endif
  case NOT_SET_VALUES:
    break;
  default:
    SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Cannot handle insert mode %d", addv);
  }
  PetscFunctionReturn(0);
}

/* ----------------------------------------------------------------------------------------------- */
PETSC_STATIC_INLINE void Pack_2(PetscInt n,const PetscInt *indicesx,const PetscScalar *x,PetscScalar *y,PetscInt bs)
{
  PetscInt i,idx;

  for (i=0; i<n; i++) {
    idx  = *indicesx++;
    y[0] = x[idx];
    y[1] = x[idx+1];
    y   += 2;
  }
}

PETSC_STATIC_INLINE PetscErrorCode UnPack_2(PetscInt n,const PetscScalar *x,const PetscInt *indicesy,PetscScalar *y,InsertMode addv,PetscInt bs)
{
  PetscInt i,idy;

  PetscFunctionBegin;
  switch (addv) {
  case INSERT_VALUES:
  case INSERT_ALL_VALUES:
    for (i=0; i<n; i++) {
      idy      = *indicesy++;
      y[idy]   = x[0];
      y[idy+1] = x[1];
      x       += 2;
    }
    break;
  case ADD_VALUES:
  case ADD_ALL_VALUES:
    for (i=0; i<n; i++) {
      idy       = *indicesy++;
      y[idy]   += x[0];
      y[idy+1] += x[1];
      x        += 2;
    }
    break;
#if !defined(PETSC_USE_COMPLEX)
  case MAX_VALUES:
    for (i=0; i<n; i++) {
      idy      = *indicesy++;
      y[idy]   = PetscMax(y[idy],x[0]);
      y[idy+1] = PetscMax(y[idy+1],x[1]);
      x       += 2;
    }
#else
  case MAX_VALUES:
#endif
  case NOT_SET_VALUES:
    break;
  default:
    SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Cannot handle insert mode %d", addv);
  }
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode Scatter_2(PetscInt n,const PetscInt *indicesx,const PetscScalar *x,const PetscInt *indicesy,PetscScalar *y,InsertMode addv,PetscInt bs)
{
  PetscInt i,idx,idy;

  PetscFunctionBegin;
  switch (addv) {
  case INSERT_VALUES:
  case INSERT_ALL_VALUES:
    for (i=0; i<n; i++) {
      idx      = *indicesx++;
      idy      = *indicesy++;
      y[idy]   = x[idx];
      y[idy+1] = x[idx+1];
    }
    break;
  case ADD_VALUES:
  case ADD_ALL_VALUES:
    for (i=0; i<n; i++) {
      idx       = *indicesx++;
      idy       = *indicesy++;
      y[idy]   += x[idx];
      y[idy+1] += x[idx+1];
    }
    break;
#if !defined(PETSC_USE_COMPLEX)
  case MAX_VALUES:
    for (i=0; i<n; i++) {
      idx      = *indicesx++;
      idy      = *indicesy++;
      y[idy]   = PetscMax(y[idy],x[idx]);
      y[idy+1] = PetscMax(y[idy+1],x[idx+1]);
    }
#else
  case MAX_VALUES:
#endif
  case NOT_SET_VALUES:
    break;
  default:
    SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Cannot handle insert mode %d", addv);
  }
  PetscFunctionReturn(0);
}
/* ----------------------------------------------------------------------------------------------- */
PETSC_STATIC_INLINE void Pack_3(PetscInt n,const PetscInt *indicesx,const PetscScalar *x,PetscScalar *y,PetscInt bs)
{
  PetscInt i,idx;

  for (i=0; i<n; i++) {
    idx  = *indicesx++;
    y[0] = x[idx];
    y[1] = x[idx+1];
    y[2] = x[idx+2];
    y   += 3;
  }
}
PETSC_STATIC_INLINE PetscErrorCode UnPack_3(PetscInt n,const PetscScalar *x,const PetscInt *indicesy,PetscScalar *y,InsertMode addv,PetscInt bs)
{
  PetscInt i,idy;

  PetscFunctionBegin;
  switch (addv) {
  case INSERT_VALUES:
  case INSERT_ALL_VALUES:
    for (i=0; i<n; i++) {
      idy      = *indicesy++;
      y[idy]   = x[0];
      y[idy+1] = x[1];
      y[idy+2] = x[2];
      x       += 3;
    }
    break;
  case ADD_VALUES:
  case ADD_ALL_VALUES:
    for (i=0; i<n; i++) {
      idy       = *indicesy++;
      y[idy]   += x[0];
      y[idy+1] += x[1];
      y[idy+2] += x[2];
      x        += 3;
    }
    break;
#if !defined(PETSC_USE_COMPLEX)
  case MAX_VALUES:
    for (i=0; i<n; i++) {
      idy      = *indicesy++;
      y[idy]   = PetscMax(y[idy],x[0]);
      y[idy+1] = PetscMax(y[idy+1],x[1]);
      y[idy+2] = PetscMax(y[idy+2],x[2]);
      x       += 3;
    }
#else
  case MAX_VALUES:
#endif
  case NOT_SET_VALUES:
    break;
  default:
    SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Cannot handle insert mode %d", addv);
  }
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode Scatter_3(PetscInt n,const PetscInt *indicesx,const PetscScalar *x,const PetscInt *indicesy,PetscScalar *y,InsertMode addv,PetscInt bs)
{
  PetscInt i,idx,idy;

  PetscFunctionBegin;
  switch (addv) {
  case INSERT_VALUES:
  case INSERT_ALL_VALUES:
    for (i=0; i<n; i++) {
      idx      = *indicesx++;
      idy      = *indicesy++;
      y[idy]   = x[idx];
      y[idy+1] = x[idx+1];
      y[idy+2] = x[idx+2];
    }
    break;
  case ADD_VALUES:
  case ADD_ALL_VALUES:
    for (i=0; i<n; i++) {
      idx       = *indicesx++;
      idy       = *indicesy++;
      y[idy]   += x[idx];
      y[idy+1] += x[idx+1];
      y[idy+2] += x[idx+2];
    }
    break;
#if !defined(PETSC_USE_COMPLEX)
  case MAX_VALUES:
    for (i=0; i<n; i++) {
      idx      = *indicesx++;
      idy      = *indicesy++;
      y[idy]   = PetscMax(y[idy],x[idx]);
      y[idy+1] = PetscMax(y[idy+1],x[idx+1]);
      y[idy+2] = PetscMax(y[idy+2],x[idx+2]);
    }
#else
  case MAX_VALUES:
#endif
  case NOT_SET_VALUES:
    break;
  default:
    SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Cannot handle insert mode %d", addv);
  }
  PetscFunctionReturn(0);
}
/* ----------------------------------------------------------------------------------------------- */
PETSC_STATIC_INLINE void Pack_4(PetscInt n,const PetscInt *indicesx,const PetscScalar *x,PetscScalar *y,PetscInt bs)
{
  PetscInt i,idx;

  for (i=0; i<n; i++) {
    idx  = *indicesx++;
    y[0] = x[idx];
    y[1] = x[idx+1];
    y[2] = x[idx+2];
    y[3] = x[idx+3];
    y   += 4;
  }
}
PETSC_STATIC_INLINE PetscErrorCode UnPack_4(PetscInt n,const PetscScalar *x,const PetscInt *indicesy,PetscScalar *y,InsertMode addv,PetscInt bs)
{
  PetscInt i,idy;

  PetscFunctionBegin;
  switch (addv) {
  case INSERT_VALUES:
  case INSERT_ALL_VALUES:
    for (i=0; i<n; i++) {
      idy      = *indicesy++;
      y[idy]   = x[0];
      y[idy+1] = x[1];
      y[idy+2] = x[2];
      y[idy+3] = x[3];
      x       += 4;
    }
    break;
  case ADD_VALUES:
  case ADD_ALL_VALUES:
    for (i=0; i<n; i++) {
      idy       = *indicesy++;
      y[idy]   += x[0];
      y[idy+1] += x[1];
      y[idy+2] += x[2];
      y[idy+3] += x[3];
      x        += 4;
    }
    break;
#if !defined(PETSC_USE_COMPLEX)
  case MAX_VALUES:
    for (i=0; i<n; i++) {
      idy      = *indicesy++;
      y[idy]   = PetscMax(y[idy],x[0]);
      y[idy+1] = PetscMax(y[idy+1],x[1]);
      y[idy+2] = PetscMax(y[idy+2],x[2]);
      y[idy+3] = PetscMax(y[idy+3],x[3]);
      x       += 4;
    }
#else
  case MAX_VALUES:
#endif
  case NOT_SET_VALUES:
    break;
  default:
    SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Cannot handle insert mode %d", addv);
  }
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode Scatter_4(PetscInt n,const PetscInt *indicesx,const PetscScalar *x,const PetscInt *indicesy,PetscScalar *y,InsertMode addv,PetscInt bs)
{
  PetscInt i,idx,idy;

  PetscFunctionBegin;
  switch (addv) {
  case INSERT_VALUES:
  case INSERT_ALL_VALUES:
    for (i=0; i<n; i++) {
      idx      = *indicesx++;
      idy      = *indicesy++;
      y[idy]   = x[idx];
      y[idy+1] = x[idx+1];
      y[idy+2] = x[idx+2];
      y[idy+3] = x[idx+3];
    }
    break;
  case ADD_VALUES:
  case ADD_ALL_VALUES:
    for (i=0; i<n; i++) {
      idx       = *indicesx++;
      idy       = *indicesy++;
      y[idy]   += x[idx];
      y[idy+1] += x[idx+1];
      y[idy+2] += x[idx+2];
      y[idy+3] += x[idx+3];
    }
    break;
#if !defined(PETSC_USE_COMPLEX)
  case MAX_VALUES:
    for (i=0; i<n; i++) {
      idx      = *indicesx++;
      idy      = *indicesy++;
      y[idy]   = PetscMax(y[idy],x[idx]);
      y[idy+1] = PetscMax(y[idy+1],x[idx+1]);
      y[idy+2] = PetscMax(y[idy+2],x[idx+2]);
      y[idy+3] = PetscMax(y[idy+3],x[idx+3]);
    }
#else
  case MAX_VALUES:
#endif
  case NOT_SET_VALUES:
    break;
  default:
    SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Cannot handle insert mode %d", addv);
  }
  PetscFunctionReturn(0);
}
/* ----------------------------------------------------------------------------------------------- */
PETSC_STATIC_INLINE void Pack_5(PetscInt n,const PetscInt *indicesx,const PetscScalar *x,PetscScalar *y,PetscInt bs)
{
  PetscInt i,idx;

  for (i=0; i<n; i++) {
    idx  = *indicesx++;
    y[0] = x[idx];
    y[1] = x[idx+1];
    y[2] = x[idx+2];
    y[3] = x[idx+3];
    y[4] = x[idx+4];
    y   += 5;
  }
}

PETSC_STATIC_INLINE PetscErrorCode UnPack_5(PetscInt n,const PetscScalar *x,const PetscInt *indicesy,PetscScalar *y,InsertMode addv,PetscInt bs)
{
  PetscInt i,idy;

  PetscFunctionBegin;
  switch (addv) {
  case INSERT_VALUES:
  case INSERT_ALL_VALUES:
    for (i=0; i<n; i++) {
      idy      = *indicesy++;
      y[idy]   = x[0];
      y[idy+1] = x[1];
      y[idy+2] = x[2];
      y[idy+3] = x[3];
      y[idy+4] = x[4];
      x       += 5;
    }
    break;
  case ADD_VALUES:
  case ADD_ALL_VALUES:
    for (i=0; i<n; i++) {
      idy       = *indicesy++;
      y[idy]   += x[0];
      y[idy+1] += x[1];
      y[idy+2] += x[2];
      y[idy+3] += x[3];
      y[idy+4] += x[4];
      x        += 5;
    }
    break;
#if !defined(PETSC_USE_COMPLEX)
  case MAX_VALUES:
    for (i=0; i<n; i++) {
      idy      = *indicesy++;
      y[idy]   = PetscMax(y[idy],x[0]);
      y[idy+1] = PetscMax(y[idy+1],x[1]);
      y[idy+2] = PetscMax(y[idy+2],x[2]);
      y[idy+3] = PetscMax(y[idy+3],x[3]);
      y[idy+4] = PetscMax(y[idy+4],x[4]);
      x       += 5;
    }
#else
  case MAX_VALUES:
#endif
  case NOT_SET_VALUES:
    break;
  default:
    SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Cannot handle insert mode %d", addv);
  }
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode Scatter_5(PetscInt n,const PetscInt *indicesx,const PetscScalar *x,const PetscInt *indicesy,PetscScalar *y,InsertMode addv,PetscInt bs)
{
  PetscInt i,idx,idy;

  PetscFunctionBegin;
  switch (addv) {
  case INSERT_VALUES:
  case INSERT_ALL_VALUES:
    for (i=0; i<n; i++) {
      idx      = *indicesx++;
      idy      = *indicesy++;
      y[idy]   = x[idx];
      y[idy+1] = x[idx+1];
      y[idy+2] = x[idx+2];
      y[idy+3] = x[idx+3];
      y[idy+4] = x[idx+4];
    }
    break;
  case ADD_VALUES:
  case ADD_ALL_VALUES:
    for (i=0; i<n; i++) {
      idx       = *indicesx++;
      idy       = *indicesy++;
      y[idy]   += x[idx];
      y[idy+1] += x[idx+1];
      y[idy+2] += x[idx+2];
      y[idy+3] += x[idx+3];
      y[idy+4] += x[idx+4];
    }
    break;
#if !defined(PETSC_USE_COMPLEX)
  case MAX_VALUES:
    for (i=0; i<n; i++) {
      idx      = *indicesx++;
      idy      = *indicesy++;
      y[idy]   = PetscMax(y[idy],x[idx]);
      y[idy+1] = PetscMax(y[idy+1],x[idx+1]);
      y[idy+2] = PetscMax(y[idy+2],x[idx+2]);
      y[idy+3] = PetscMax(y[idy+3],x[idx+3]);
      y[idy+4] = PetscMax(y[idy+4],x[idx+4]);
    }
#else
  case MAX_VALUES:
#endif
  case NOT_SET_VALUES:
    break;
  default:
    SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Cannot handle insert mode %d", addv);
  }
  PetscFunctionReturn(0);
}
/* ----------------------------------------------------------------------------------------------- */
PETSC_STATIC_INLINE void Pack_6(PetscInt n,const PetscInt *indicesx,const PetscScalar *x,PetscScalar *y,PetscInt bs)
{
  PetscInt i,idx;

  for (i=0; i<n; i++) {
    idx  = *indicesx++;
    y[0] = x[idx];
    y[1] = x[idx+1];
    y[2] = x[idx+2];
    y[3] = x[idx+3];
    y[4] = x[idx+4];
    y[5] = x[idx+5];
    y   += 6;
  }
}

PETSC_STATIC_INLINE PetscErrorCode UnPack_6(PetscInt n,const PetscScalar *x,const PetscInt *indicesy,PetscScalar *y,InsertMode addv,PetscInt bs)
{
  PetscInt i,idy;

  PetscFunctionBegin;
  switch (addv) {
  case INSERT_VALUES:
  case INSERT_ALL_VALUES:
    for (i=0; i<n; i++) {
      idy      = *indicesy++;
      y[idy]   = x[0];
      y[idy+1] = x[1];
      y[idy+2] = x[2];
      y[idy+3] = x[3];
      y[idy+4] = x[4];
      y[idy+5] = x[5];
      x       += 6;
    }
    break;
  case ADD_VALUES:
  case ADD_ALL_VALUES:
    for (i=0; i<n; i++) {
      idy       = *indicesy++;
      y[idy]   += x[0];
      y[idy+1] += x[1];
      y[idy+2] += x[2];
      y[idy+3] += x[3];
      y[idy+4] += x[4];
      y[idy+5] += x[5];
      x        += 6;
    }
    break;
#if !defined(PETSC_USE_COMPLEX)
  case MAX_VALUES:
    for (i=0; i<n; i++) {
      idy      = *indicesy++;
      y[idy]   = PetscMax(y[idy],x[0]);
      y[idy+1] = PetscMax(y[idy+1],x[1]);
      y[idy+2] = PetscMax(y[idy+2],x[2]);
      y[idy+3] = PetscMax(y[idy+3],x[3]);
      y[idy+4] = PetscMax(y[idy+4],x[4]);
      y[idy+5] = PetscMax(y[idy+5],x[5]);
      x       += 6;
    }
#else
  case MAX_VALUES:
#endif
  case NOT_SET_VALUES:
    break;
  default:
    SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Cannot handle insert mode %d", addv);
  }
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode Scatter_6(PetscInt n,const PetscInt *indicesx,const PetscScalar *x,const PetscInt *indicesy,PetscScalar *y,InsertMode addv,PetscInt bs)
{
  PetscInt i,idx,idy;

  PetscFunctionBegin;
  switch (addv) {
  case INSERT_VALUES:
  case INSERT_ALL_VALUES:
    for (i=0; i<n; i++) {
      idx      = *indicesx++;
      idy      = *indicesy++;
      y[idy]   = x[idx];
      y[idy+1] = x[idx+1];
      y[idy+2] = x[idx+2];
      y[idy+3] = x[idx+3];
      y[idy+4] = x[idx+4];
      y[idy+5] = x[idx+5];
    }
    break;
  case ADD_VALUES:
  case ADD_ALL_VALUES:
    for (i=0; i<n; i++) {
      idx       = *indicesx++;
      idy       = *indicesy++;
      y[idy]   += x[idx];
      y[idy+1] += x[idx+1];
      y[idy+2] += x[idx+2];
      y[idy+3] += x[idx+3];
      y[idy+4] += x[idx+4];
      y[idy+5] += x[idx+5];
    }
    break;
#if !defined(PETSC_USE_COMPLEX)
  case MAX_VALUES:
    for (i=0; i<n; i++) {
      idx      = *indicesx++;
      idy      = *indicesy++;
      y[idy]   = PetscMax(y[idy],x[idx]);
      y[idy+1] = PetscMax(y[idy+1],x[idx+1]);
      y[idy+2] = PetscMax(y[idy+2],x[idx+2]);
      y[idy+3] = PetscMax(y[idy+3],x[idx+3]);
      y[idy+4] = PetscMax(y[idy+4],x[idx+4]);
      y[idy+5] = PetscMax(y[idy+5],x[idx+5]);
    }
#else
  case MAX_VALUES:
#endif
  case NOT_SET_VALUES:
    break;
  default:
    SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Cannot handle insert mode %d", addv);
  }
  PetscFunctionReturn(0);
}
/* ----------------------------------------------------------------------------------------------- */
PETSC_STATIC_INLINE void Pack_7(PetscInt n,const PetscInt *indicesx,const PetscScalar *x,PetscScalar *y,PetscInt bs)
{
  PetscInt i,idx;

  for (i=0; i<n; i++) {
    idx  = *indicesx++;
    y[0] = x[idx];
    y[1] = x[idx+1];
    y[2] = x[idx+2];
    y[3] = x[idx+3];
    y[4] = x[idx+4];
    y[5] = x[idx+5];
    y[6] = x[idx+6];
    y   += 7;
  }
}

PETSC_STATIC_INLINE PetscErrorCode UnPack_7(PetscInt n,const PetscScalar *x,const PetscInt *indicesy,PetscScalar *y,InsertMode addv,PetscInt bs)
{
  PetscInt i,idy;

  PetscFunctionBegin;
  switch (addv) {
  case INSERT_VALUES:
  case INSERT_ALL_VALUES:
    for (i=0; i<n; i++) {
      idy      = *indicesy++;
      y[idy]   = x[0];
      y[idy+1] = x[1];
      y[idy+2] = x[2];
      y[idy+3] = x[3];
      y[idy+4] = x[4];
      y[idy+5] = x[5];
      y[idy+6] = x[6];
      x       += 7;
    }
    break;
  case ADD_VALUES:
  case ADD_ALL_VALUES:
    for (i=0; i<n; i++) {
      idy       = *indicesy++;
      y[idy]   += x[0];
      y[idy+1] += x[1];
      y[idy+2] += x[2];
      y[idy+3] += x[3];
      y[idy+4] += x[4];
      y[idy+5] += x[5];
      y[idy+6] += x[6];
      x        += 7;
    }
    break;
#if !defined(PETSC_USE_COMPLEX)
  case MAX_VALUES:
    for (i=0; i<n; i++) {
      idy      = *indicesy++;
      y[idy]   = PetscMax(y[idy],x[0]);
      y[idy+1] = PetscMax(y[idy+1],x[1]);
      y[idy+2] = PetscMax(y[idy+2],x[2]);
      y[idy+3] = PetscMax(y[idy+3],x[3]);
      y[idy+4] = PetscMax(y[idy+4],x[4]);
      y[idy+5] = PetscMax(y[idy+5],x[5]);
      y[idy+6] = PetscMax(y[idy+6],x[6]);
      x       += 7;
    }
#else
  case MAX_VALUES:
#endif
  case NOT_SET_VALUES:
    break;
  default:
    SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Cannot handle insert mode %d", addv);
  }
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode Scatter_7(PetscInt n,const PetscInt *indicesx,const PetscScalar *x,const PetscInt *indicesy,PetscScalar *y,InsertMode addv,PetscInt bs)
{
  PetscInt i,idx,idy;

  PetscFunctionBegin;
  switch (addv) {
  case INSERT_VALUES:
  case INSERT_ALL_VALUES:
    for (i=0; i<n; i++) {
      idx      = *indicesx++;
      idy      = *indicesy++;
      y[idy]   = x[idx];
      y[idy+1] = x[idx+1];
      y[idy+2] = x[idx+2];
      y[idy+3] = x[idx+3];
      y[idy+4] = x[idx+4];
      y[idy+5] = x[idx+5];
      y[idy+6] = x[idx+6];
    }
    break;
  case ADD_VALUES:
  case ADD_ALL_VALUES:
    for (i=0; i<n; i++) {
      idx       = *indicesx++;
      idy       = *indicesy++;
      y[idy]   += x[idx];
      y[idy+1] += x[idx+1];
      y[idy+2] += x[idx+2];
      y[idy+3] += x[idx+3];
      y[idy+4] += x[idx+4];
      y[idy+5] += x[idx+5];
      y[idy+6] += x[idx+6];
    }
    break;
#if !defined(PETSC_USE_COMPLEX)
  case MAX_VALUES:
    for (i=0; i<n; i++) {
      idx      = *indicesx++;
      idy      = *indicesy++;
      y[idy]   = PetscMax(y[idy],x[idx]);
      y[idy+1] = PetscMax(y[idy+1],x[idx+1]);
      y[idy+2] = PetscMax(y[idy+2],x[idx+2]);
      y[idy+3] = PetscMax(y[idy+3],x[idx+3]);
      y[idy+4] = PetscMax(y[idy+4],x[idx+4]);
      y[idy+5] = PetscMax(y[idy+5],x[idx+5]);
      y[idy+6] = PetscMax(y[idy+6],x[idx+6]);
    }
#else
  case MAX_VALUES:
#endif
  case NOT_SET_VALUES:
    break;
  default:
    SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Cannot handle insert mode %d", addv);
  }
  PetscFunctionReturn(0);
}
/* ----------------------------------------------------------------------------------------------- */
PETSC_STATIC_INLINE void Pack_8(PetscInt n,const PetscInt *indicesx,const PetscScalar *x,PetscScalar *y,PetscInt bs)
{
  PetscInt i,idx;

  for (i=0; i<n; i++) {
    idx  = *indicesx++;
    y[0] = x[idx];
    y[1] = x[idx+1];
    y[2] = x[idx+2];
    y[3] = x[idx+3];
    y[4] = x[idx+4];
    y[5] = x[idx+5];
    y[6] = x[idx+6];
    y[7] = x[idx+7];
    y   += 8;
  }
}

PETSC_STATIC_INLINE PetscErrorCode UnPack_8(PetscInt n,const PetscScalar *x,const PetscInt *indicesy,PetscScalar *y,InsertMode addv,PetscInt bs)
{
  PetscInt i,idy;

  PetscFunctionBegin;
  switch (addv) {
  case INSERT_VALUES:
  case INSERT_ALL_VALUES:
    for (i=0; i<n; i++) {
      idy      = *indicesy++;
      y[idy]   = x[0];
      y[idy+1] = x[1];
      y[idy+2] = x[2];
      y[idy+3] = x[3];
      y[idy+4] = x[4];
      y[idy+5] = x[5];
      y[idy+6] = x[6];
      y[idy+7] = x[7];
      x       += 8;
    }
    break;
  case ADD_VALUES:
  case ADD_ALL_VALUES:
    for (i=0; i<n; i++) {
      idy       = *indicesy++;
      y[idy]   += x[0];
      y[idy+1] += x[1];
      y[idy+2] += x[2];
      y[idy+3] += x[3];
      y[idy+4] += x[4];
      y[idy+5] += x[5];
      y[idy+6] += x[6];
      y[idy+7] += x[7];
      x        += 8;
    }
    break;
#if !defined(PETSC_USE_COMPLEX)
  case MAX_VALUES:
    for (i=0; i<n; i++) {
      idy      = *indicesy++;
      y[idy]   = PetscMax(y[idy],x[0]);
      y[idy+1] = PetscMax(y[idy+1],x[1]);
      y[idy+2] = PetscMax(y[idy+2],x[2]);
      y[idy+3] = PetscMax(y[idy+3],x[3]);
      y[idy+4] = PetscMax(y[idy+4],x[4]);
      y[idy+5] = PetscMax(y[idy+5],x[5]);
      y[idy+6] = PetscMax(y[idy+6],x[6]);
      y[idy+7] = PetscMax(y[idy+7],x[7]);
      x       += 8;
    }
#else
  case MAX_VALUES:
#endif
  case NOT_SET_VALUES:
    break;
  default:
    SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Cannot handle insert mode %d", addv);
  }
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode Scatter_8(PetscInt n,const PetscInt *indicesx,const PetscScalar *x,const PetscInt *indicesy,PetscScalar *y,InsertMode addv,PetscInt bs)
{
  PetscInt i,idx,idy;

  PetscFunctionBegin;
  switch (addv) {
  case INSERT_VALUES:
  case INSERT_ALL_VALUES:
    for (i=0; i<n; i++) {
      idx      = *indicesx++;
      idy      = *indicesy++;
      y[idy]   = x[idx];
      y[idy+1] = x[idx+1];
      y[idy+2] = x[idx+2];
      y[idy+3] = x[idx+3];
      y[idy+4] = x[idx+4];
      y[idy+5] = x[idx+5];
      y[idy+6] = x[idx+6];
      y[idy+7] = x[idx+7];
    }
    break;
  case ADD_VALUES:
  case ADD_ALL_VALUES:
    for (i=0; i<n; i++) {
      idx       = *indicesx++;
      idy       = *indicesy++;
      y[idy]   += x[idx];
      y[idy+1] += x[idx+1];
      y[idy+2] += x[idx+2];
      y[idy+3] += x[idx+3];
      y[idy+4] += x[idx+4];
      y[idy+5] += x[idx+5];
      y[idy+6] += x[idx+6];
      y[idy+7] += x[idx+7];
    }
    break;
#if !defined(PETSC_USE_COMPLEX)
  case MAX_VALUES:
    for (i=0; i<n; i++) {
      idx      = *indicesx++;
      idy      = *indicesy++;
      y[idy]   = PetscMax(y[idy],x[idx]);
      y[idy+1] = PetscMax(y[idy+1],x[idx+1]);
      y[idy+2] = PetscMax(y[idy+2],x[idx+2]);
      y[idy+3] = PetscMax(y[idy+3],x[idx+3]);
      y[idy+4] = PetscMax(y[idy+4],x[idx+4]);
      y[idy+5] = PetscMax(y[idy+5],x[idx+5]);
      y[idy+6] = PetscMax(y[idy+6],x[idx+6]);
      y[idy+7] = PetscMax(y[idy+7],x[idx+7]);
    }
#else
  case MAX_VALUES:
#endif
  case NOT_SET_VALUES:
    break;
  default:
    SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Cannot handle insert mode %d", addv);
  }
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE void Pack_9(PetscInt n,const PetscInt *indicesx,const PetscScalar *x,PetscScalar *y,PetscInt bs)
{
  PetscInt i,idx;

  for (i=0; i<n; i++) {
    idx   = *indicesx++;
    y[0]  = x[idx];
    y[1]  = x[idx+1];
    y[2]  = x[idx+2];
    y[3]  = x[idx+3];
    y[4]  = x[idx+4];
    y[5]  = x[idx+5];
    y[6]  = x[idx+6];
    y[7]  = x[idx+7];
    y[8]  = x[idx+8];
    y    += 9;
  }
}

PETSC_STATIC_INLINE PetscErrorCode UnPack_9(PetscInt n,const PetscScalar *x,const PetscInt *indicesy,PetscScalar *y,InsertMode addv,PetscInt bs)
{
  PetscInt i,idy;

  PetscFunctionBegin;
  switch (addv) {
  case INSERT_VALUES:
  case INSERT_ALL_VALUES:
    for (i=0; i<n; i++) {
      idy       = *indicesy++;
      y[idy]    = x[0];
      y[idy+1]  = x[1];
      y[idy+2]  = x[2];
      y[idy+3]  = x[3];
      y[idy+4]  = x[4];
      y[idy+5]  = x[5];
      y[idy+6]  = x[6];
      y[idy+7]  = x[7];
      y[idy+8]  = x[8];
      x        += 9;
    }
    break;
  case ADD_VALUES:
  case ADD_ALL_VALUES:
    for (i=0; i<n; i++) {
      idy        = *indicesy++;
      y[idy]    += x[0];
      y[idy+1]  += x[1];
      y[idy+2]  += x[2];
      y[idy+3]  += x[3];
      y[idy+4]  += x[4];
      y[idy+5]  += x[5];
      y[idy+6]  += x[6];
      y[idy+7]  += x[7];
      y[idy+8]  += x[8];
      x         += 9;
    }
    break;
#if !defined(PETSC_USE_COMPLEX)
  case MAX_VALUES:
    for (i=0; i<n; i++) {
      idy       = *indicesy++;
      y[idy]    = PetscMax(y[idy],x[0]);
      y[idy+1]  = PetscMax(y[idy+1],x[1]);
      y[idy+2]  = PetscMax(y[idy+2],x[2]);
      y[idy+3]  = PetscMax(y[idy+3],x[3]);
      y[idy+4]  = PetscMax(y[idy+4],x[4]);
      y[idy+5]  = PetscMax(y[idy+5],x[5]);
      y[idy+6]  = PetscMax(y[idy+6],x[6]);
      y[idy+7]  = PetscMax(y[idy+7],x[7]);
      y[idy+8]  = PetscMax(y[idy+8],x[8]);
      x        += 9;
    }
#else
  case MAX_VALUES:
#endif
  case NOT_SET_VALUES:
    break;
  default:
    SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Cannot handle insert mode %d", addv);
  }
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode Scatter_9(PetscInt n,const PetscInt *indicesx,const PetscScalar *x,const PetscInt *indicesy,PetscScalar *y,InsertMode addv,PetscInt bs)
{
  PetscInt i,idx,idy;

  PetscFunctionBegin;
  switch (addv) {
  case INSERT_VALUES:
  case INSERT_ALL_VALUES:
    for (i=0; i<n; i++) {
      idx       = *indicesx++;
      idy       = *indicesy++;
      y[idy]    = x[idx];
      y[idy+1]  = x[idx+1];
      y[idy+2]  = x[idx+2];
      y[idy+3]  = x[idx+3];
      y[idy+4]  = x[idx+4];
      y[idy+5]  = x[idx+5];
      y[idy+6]  = x[idx+6];
      y[idy+7]  = x[idx+7];
      y[idy+8]  = x[idx+8];
    }
    break;
  case ADD_VALUES:
  case ADD_ALL_VALUES:
    for (i=0; i<n; i++) {
      idx        = *indicesx++;
      idy        = *indicesy++;
      y[idy]    += x[idx];
      y[idy+1]  += x[idx+1];
      y[idy+2]  += x[idx+2];
      y[idy+3]  += x[idx+3];
      y[idy+4]  += x[idx+4];
      y[idy+5]  += x[idx+5];
      y[idy+6]  += x[idx+6];
      y[idy+7]  += x[idx+7];
      y[idy+8]  += x[idx+8];
    }
    break;
#if !defined(PETSC_USE_COMPLEX)
  case MAX_VALUES:
    for (i=0; i<n; i++) {
      idx       = *indicesx++;
      idy       = *indicesy++;
      y[idy]    = PetscMax(y[idy],x[idx]);
      y[idy+1]  = PetscMax(y[idy+1],x[idx+1]);
      y[idy+2]  = PetscMax(y[idy+2],x[idx+2]);
      y[idy+3]  = PetscMax(y[idy+3],x[idx+3]);
      y[idy+4]  = PetscMax(y[idy+4],x[idx+4]);
      y[idy+5]  = PetscMax(y[idy+5],x[idx+5]);
      y[idy+6]  = PetscMax(y[idy+6],x[idx+6]);
      y[idy+7]  = PetscMax(y[idy+7],x[idx+7]);
      y[idy+8]  = PetscMax(y[idy+8],x[idx+8]);
    }
#else
  case MAX_VALUES:
#endif
  case NOT_SET_VALUES:
    break;
  default:
    SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Cannot handle insert mode %d", addv);
  }
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE void Pack_10(PetscInt n,const PetscInt *indicesx,const PetscScalar *x,PetscScalar *y,PetscInt bs)
{
  PetscInt i,idx;

  for (i=0; i<n; i++) {
    idx   = *indicesx++;
    y[0]  = x[idx];
    y[1]  = x[idx+1];
    y[2]  = x[idx+2];
    y[3]  = x[idx+3];
    y[4]  = x[idx+4];
    y[5]  = x[idx+5];
    y[6]  = x[idx+6];
    y[7]  = x[idx+7];
    y[8]  = x[idx+8];
    y[9]  = x[idx+9];
    y    += 10;
  }
}

PETSC_STATIC_INLINE PetscErrorCode UnPack_10(PetscInt n,const PetscScalar *x,const PetscInt *indicesy,PetscScalar *y,InsertMode addv,PetscInt bs)
{
  PetscInt i,idy;

  PetscFunctionBegin;
  switch (addv) {
  case INSERT_VALUES:
  case INSERT_ALL_VALUES:
    for (i=0; i<n; i++) {
      idy       = *indicesy++;
      y[idy]    = x[0];
      y[idy+1]  = x[1];
      y[idy+2]  = x[2];
      y[idy+3]  = x[3];
      y[idy+4]  = x[4];
      y[idy+5]  = x[5];
      y[idy+6]  = x[6];
      y[idy+7]  = x[7];
      y[idy+8]  = x[8];
      y[idy+9]  = x[9];
      x        += 10;
    }
    break;
  case ADD_VALUES:
  case ADD_ALL_VALUES:
    for (i=0; i<n; i++) {
      idy        = *indicesy++;
      y[idy]    += x[0];
      y[idy+1]  += x[1];
      y[idy+2]  += x[2];
      y[idy+3]  += x[3];
      y[idy+4]  += x[4];
      y[idy+5]  += x[5];
      y[idy+6]  += x[6];
      y[idy+7]  += x[7];
      y[idy+8]  += x[8];
      y[idy+9]  += x[9];
      x         += 10;
    }
    break;
#if !defined(PETSC_USE_COMPLEX)
  case MAX_VALUES:
    for (i=0; i<n; i++) {
      idy       = *indicesy++;
      y[idy]    = PetscMax(y[idy],x[0]);
      y[idy+1]  = PetscMax(y[idy+1],x[1]);
      y[idy+2]  = PetscMax(y[idy+2],x[2]);
      y[idy+3]  = PetscMax(y[idy+3],x[3]);
      y[idy+4]  = PetscMax(y[idy+4],x[4]);
      y[idy+5]  = PetscMax(y[idy+5],x[5]);
      y[idy+6]  = PetscMax(y[idy+6],x[6]);
      y[idy+7]  = PetscMax(y[idy+7],x[7]);
      y[idy+8]  = PetscMax(y[idy+8],x[8]);
      y[idy+9]  = PetscMax(y[idy+9],x[9]);
      x        += 10;
    }
#else
  case MAX_VALUES:
#endif
  case NOT_SET_VALUES:
    break;
  default:
    SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Cannot handle insert mode %d", addv);
  }
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode Scatter_10(PetscInt n,const PetscInt *indicesx,const PetscScalar *x,const PetscInt *indicesy,PetscScalar *y,InsertMode addv,PetscInt bs)
{
  PetscInt i,idx,idy;

  PetscFunctionBegin;
  switch (addv) {
  case INSERT_VALUES:
  case INSERT_ALL_VALUES:
    for (i=0; i<n; i++) {
      idx       = *indicesx++;
      idy       = *indicesy++;
      y[idy]    = x[idx];
      y[idy+1]  = x[idx+1];
      y[idy+2]  = x[idx+2];
      y[idy+3]  = x[idx+3];
      y[idy+4]  = x[idx+4];
      y[idy+5]  = x[idx+5];
      y[idy+6]  = x[idx+6];
      y[idy+7]  = x[idx+7];
      y[idy+8]  = x[idx+8];
      y[idy+9]  = x[idx+9];
    }
    break;
  case ADD_VALUES:
  case ADD_ALL_VALUES:
    for (i=0; i<n; i++) {
      idx        = *indicesx++;
      idy        = *indicesy++;
      y[idy]    += x[idx];
      y[idy+1]  += x[idx+1];
      y[idy+2]  += x[idx+2];
      y[idy+3]  += x[idx+3];
      y[idy+4]  += x[idx+4];
      y[idy+5]  += x[idx+5];
      y[idy+6]  += x[idx+6];
      y[idy+7]  += x[idx+7];
      y[idy+8]  += x[idx+8];
      y[idy+9]  += x[idx+9];
    }
    break;
#if !defined(PETSC_USE_COMPLEX)
  case MAX_VALUES:
    for (i=0; i<n; i++) {
      idx       = *indicesx++;
      idy       = *indicesy++;
      y[idy]    = PetscMax(y[idy],x[idx]);
      y[idy+1]  = PetscMax(y[idy+1],x[idx+1]);
      y[idy+2]  = PetscMax(y[idy+2],x[idx+2]);
      y[idy+3]  = PetscMax(y[idy+3],x[idx+3]);
      y[idy+4]  = PetscMax(y[idy+4],x[idx+4]);
      y[idy+5]  = PetscMax(y[idy+5],x[idx+5]);
      y[idy+6]  = PetscMax(y[idy+6],x[idx+6]);
      y[idy+7]  = PetscMax(y[idy+7],x[idx+7]);
      y[idy+8]  = PetscMax(y[idy+8],x[idx+8]);
      y[idy+9]  = PetscMax(y[idy+9],x[idx+9]);
    }
#else
  case MAX_VALUES:
#endif
  case NOT_SET_VALUES:
    break;
  default:
    SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Cannot handle insert mode %d", addv);
  }
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE void Pack_11(PetscInt n,const PetscInt *indicesx,const PetscScalar *x,PetscScalar *y,PetscInt bs)
{
  PetscInt i,idx;

  for (i=0; i<n; i++) {
    idx   = *indicesx++;
    y[0]  = x[idx];
    y[1]  = x[idx+1];
    y[2]  = x[idx+2];
    y[3]  = x[idx+3];
    y[4]  = x[idx+4];
    y[5]  = x[idx+5];
    y[6]  = x[idx+6];
    y[7]  = x[idx+7];
    y[8]  = x[idx+8];
    y[9]  = x[idx+9];
    y[10] = x[idx+10];
    y    += 11;
  }
}

PETSC_STATIC_INLINE PetscErrorCode UnPack_11(PetscInt n,const PetscScalar *x,const PetscInt *indicesy,PetscScalar *y,InsertMode addv,PetscInt bs)
{
  PetscInt i,idy;

  PetscFunctionBegin;
  switch (addv) {
  case INSERT_VALUES:
  case INSERT_ALL_VALUES:
    for (i=0; i<n; i++) {
      idy       = *indicesy++;
      y[idy]    = x[0];
      y[idy+1]  = x[1];
      y[idy+2]  = x[2];
      y[idy+3]  = x[3];
      y[idy+4]  = x[4];
      y[idy+5]  = x[5];
      y[idy+6]  = x[6];
      y[idy+7]  = x[7];
      y[idy+8]  = x[8];
      y[idy+9]  = x[9];
      y[idy+10] = x[10];
      x        += 11;
    }
    break;
  case ADD_VALUES:
  case ADD_ALL_VALUES:
    for (i=0; i<n; i++) {
      idy        = *indicesy++;
      y[idy]    += x[0];
      y[idy+1]  += x[1];
      y[idy+2]  += x[2];
      y[idy+3]  += x[3];
      y[idy+4]  += x[4];
      y[idy+5]  += x[5];
      y[idy+6]  += x[6];
      y[idy+7]  += x[7];
      y[idy+8]  += x[8];
      y[idy+9]  += x[9];
      y[idy+10] += x[10];
      x         += 11;
    }
    break;
#if !defined(PETSC_USE_COMPLEX)
  case MAX_VALUES:
    for (i=0; i<n; i++) {
      idy       = *indicesy++;
      y[idy]    = PetscMax(y[idy],x[0]);
      y[idy+1]  = PetscMax(y[idy+1],x[1]);
      y[idy+2]  = PetscMax(y[idy+2],x[2]);
      y[idy+3]  = PetscMax(y[idy+3],x[3]);
      y[idy+4]  = PetscMax(y[idy+4],x[4]);
      y[idy+5]  = PetscMax(y[idy+5],x[5]);
      y[idy+6]  = PetscMax(y[idy+6],x[6]);
      y[idy+7]  = PetscMax(y[idy+7],x[7]);
      y[idy+8]  = PetscMax(y[idy+8],x[8]);
      y[idy+9]  = PetscMax(y[idy+9],x[9]);
      y[idy+10] = PetscMax(y[idy+10],x[10]);
      x        += 11;
    }
#else
  case MAX_VALUES:
#endif
  case NOT_SET_VALUES:
    break;
  default:
    SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Cannot handle insert mode %d", addv);
  }
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode Scatter_11(PetscInt n,const PetscInt *indicesx,const PetscScalar *x,const PetscInt *indicesy,PetscScalar *y,InsertMode addv,PetscInt bs)
{
  PetscInt i,idx,idy;

  PetscFunctionBegin;
  switch (addv) {
  case INSERT_VALUES:
  case INSERT_ALL_VALUES:
    for (i=0; i<n; i++) {
      idx       = *indicesx++;
      idy       = *indicesy++;
      y[idy]    = x[idx];
      y[idy+1]  = x[idx+1];
      y[idy+2]  = x[idx+2];
      y[idy+3]  = x[idx+3];
      y[idy+4]  = x[idx+4];
      y[idy+5]  = x[idx+5];
      y[idy+6]  = x[idx+6];
      y[idy+7]  = x[idx+7];
      y[idy+8]  = x[idx+8];
      y[idy+9]  = x[idx+9];
      y[idy+10] = x[idx+10];
    }
    break;
  case ADD_VALUES:
  case ADD_ALL_VALUES:
    for (i=0; i<n; i++) {
      idx        = *indicesx++;
      idy        = *indicesy++;
      y[idy]    += x[idx];
      y[idy+1]  += x[idx+1];
      y[idy+2]  += x[idx+2];
      y[idy+3]  += x[idx+3];
      y[idy+4]  += x[idx+4];
      y[idy+5]  += x[idx+5];
      y[idy+6]  += x[idx+6];
      y[idy+7]  += x[idx+7];
      y[idy+8]  += x[idx+8];
      y[idy+9]  += x[idx+9];
      y[idy+10] += x[idx+10];
    }
    break;
#if !defined(PETSC_USE_COMPLEX)
  case MAX_VALUES:
    for (i=0; i<n; i++) {
      idx       = *indicesx++;
      idy       = *indicesy++;
      y[idy]    = PetscMax(y[idy],x[idx]);
      y[idy+1]  = PetscMax(y[idy+1],x[idx+1]);
      y[idy+2]  = PetscMax(y[idy+2],x[idx+2]);
      y[idy+3]  = PetscMax(y[idy+3],x[idx+3]);
      y[idy+4]  = PetscMax(y[idy+4],x[idx+4]);
      y[idy+5]  = PetscMax(y[idy+5],x[idx+5]);
      y[idy+6]  = PetscMax(y[idy+6],x[idx+6]);
      y[idy+7]  = PetscMax(y[idy+7],x[idx+7]);
      y[idy+8]  = PetscMax(y[idy+8],x[idx+8]);
      y[idy+9]  = PetscMax(y[idy+9],x[idx+9]);
      y[idy+10] = PetscMax(y[idy+10],x[idx+10]);
    }
#else
  case MAX_VALUES:
#endif
  case NOT_SET_VALUES:
    break;
  default:
    SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Cannot handle insert mode %d", addv);
  }
  PetscFunctionReturn(0);
}

/* ----------------------------------------------------------------------------------------------- */
PETSC_STATIC_INLINE void Pack_12(PetscInt n,const PetscInt *indicesx,const PetscScalar *x,PetscScalar *y,PetscInt bs)
{
  PetscInt i,idx;

  for (i=0; i<n; i++) {
    idx   = *indicesx++;
    y[0]  = x[idx];
    y[1]  = x[idx+1];
    y[2]  = x[idx+2];
    y[3]  = x[idx+3];
    y[4]  = x[idx+4];
    y[5]  = x[idx+5];
    y[6]  = x[idx+6];
    y[7]  = x[idx+7];
    y[8]  = x[idx+8];
    y[9]  = x[idx+9];
    y[10] = x[idx+10];
    y[11] = x[idx+11];
    y    += 12;
  }
}

PETSC_STATIC_INLINE PetscErrorCode UnPack_12(PetscInt n,const PetscScalar *x,const PetscInt *indicesy,PetscScalar *y,InsertMode addv,PetscInt bs)
{
  PetscInt i,idy;

  PetscFunctionBegin;
  switch (addv) {
  case INSERT_VALUES:
  case INSERT_ALL_VALUES:
    for (i=0; i<n; i++) {
      idy       = *indicesy++;
      y[idy]    = x[0];
      y[idy+1]  = x[1];
      y[idy+2]  = x[2];
      y[idy+3]  = x[3];
      y[idy+4]  = x[4];
      y[idy+5]  = x[5];
      y[idy+6]  = x[6];
      y[idy+7]  = x[7];
      y[idy+8]  = x[8];
      y[idy+9]  = x[9];
      y[idy+10] = x[10];
      y[idy+11] = x[11];
      x        += 12;
    }
    break;
  case ADD_VALUES:
  case ADD_ALL_VALUES:
    for (i=0; i<n; i++) {
      idy        = *indicesy++;
      y[idy]    += x[0];
      y[idy+1]  += x[1];
      y[idy+2]  += x[2];
      y[idy+3]  += x[3];
      y[idy+4]  += x[4];
      y[idy+5]  += x[5];
      y[idy+6]  += x[6];
      y[idy+7]  += x[7];
      y[idy+8]  += x[8];
      y[idy+9]  += x[9];
      y[idy+10] += x[10];
      y[idy+11] += x[11];
      x         += 12;
    }
    break;
#if !defined(PETSC_USE_COMPLEX)
  case MAX_VALUES:
    for (i=0; i<n; i++) {
      idy       = *indicesy++;
      y[idy]    = PetscMax(y[idy],x[0]);
      y[idy+1]  = PetscMax(y[idy+1],x[1]);
      y[idy+2]  = PetscMax(y[idy+2],x[2]);
      y[idy+3]  = PetscMax(y[idy+3],x[3]);
      y[idy+4]  = PetscMax(y[idy+4],x[4]);
      y[idy+5]  = PetscMax(y[idy+5],x[5]);
      y[idy+6]  = PetscMax(y[idy+6],x[6]);
      y[idy+7]  = PetscMax(y[idy+7],x[7]);
      y[idy+8]  = PetscMax(y[idy+8],x[8]);
      y[idy+9]  = PetscMax(y[idy+9],x[9]);
      y[idy+10] = PetscMax(y[idy+10],x[10]);
      y[idy+11] = PetscMax(y[idy+11],x[11]);
      x        += 12;
    }
#else
  case MAX_VALUES:
#endif
  case NOT_SET_VALUES:
    break;
  default:
    SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Cannot handle insert mode %d", addv);
  }
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode Scatter_12(PetscInt n,const PetscInt *indicesx,const PetscScalar *x,const PetscInt *indicesy,PetscScalar *y,InsertMode addv,PetscInt bs)
{
  PetscInt i,idx,idy;

  PetscFunctionBegin;
  switch (addv) {
  case INSERT_VALUES:
  case INSERT_ALL_VALUES:
    for (i=0; i<n; i++) {
      idx       = *indicesx++;
      idy       = *indicesy++;
      y[idy]    = x[idx];
      y[idy+1]  = x[idx+1];
      y[idy+2]  = x[idx+2];
      y[idy+3]  = x[idx+3];
      y[idy+4]  = x[idx+4];
      y[idy+5]  = x[idx+5];
      y[idy+6]  = x[idx+6];
      y[idy+7]  = x[idx+7];
      y[idy+8]  = x[idx+8];
      y[idy+9]  = x[idx+9];
      y[idy+10] = x[idx+10];
      y[idy+11] = x[idx+11];
    }
    break;
  case ADD_VALUES:
  case ADD_ALL_VALUES:
    for (i=0; i<n; i++) {
      idx        = *indicesx++;
      idy        = *indicesy++;
      y[idy]    += x[idx];
      y[idy+1]  += x[idx+1];
      y[idy+2]  += x[idx+2];
      y[idy+3]  += x[idx+3];
      y[idy+4]  += x[idx+4];
      y[idy+5]  += x[idx+5];
      y[idy+6]  += x[idx+6];
      y[idy+7]  += x[idx+7];
      y[idy+8]  += x[idx+8];
      y[idy+9]  += x[idx+9];
      y[idy+10] += x[idx+10];
      y[idy+11] += x[idx+11];
    }
    break;
#if !defined(PETSC_USE_COMPLEX)
  case MAX_VALUES:
    for (i=0; i<n; i++) {
      idx       = *indicesx++;
      idy       = *indicesy++;
      y[idy]    = PetscMax(y[idy],x[idx]);
      y[idy+1]  = PetscMax(y[idy+1],x[idx+1]);
      y[idy+2]  = PetscMax(y[idy+2],x[idx+2]);
      y[idy+3]  = PetscMax(y[idy+3],x[idx+3]);
      y[idy+4]  = PetscMax(y[idy+4],x[idx+4]);
      y[idy+5]  = PetscMax(y[idy+5],x[idx+5]);
      y[idy+6]  = PetscMax(y[idy+6],x[idx+6]);
      y[idy+7]  = PetscMax(y[idy+7],x[idx+7]);
      y[idy+8]  = PetscMax(y[idy+8],x[idx+8]);
      y[idy+9]  = PetscMax(y[idy+9],x[idx+9]);
      y[idy+10] = PetscMax(y[idy+10],x[idx+10]);
      y[idy+11] = PetscMax(y[idy+11],x[idx+11]);
    }
#else
  case MAX_VALUES:
#endif
  case NOT_SET_VALUES:
    break;
  default:
    SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Cannot handle insert mode %d", addv);
  }
  PetscFunctionReturn(0);
}

/* ----------------------------------------------------------------------------------------------- */
PETSC_STATIC_INLINE void Pack_bs(PetscInt n,const PetscInt *indicesx,const PetscScalar *x,PetscScalar *y,PetscInt bs)
{
  PetscInt       i,idx;
  PetscErrorCode ierr;

  for (i=0; i<n; i++) {
    idx   = *indicesx++;
    ierr = PetscArraycpy(y,x + idx,bs);CHKERRV(ierr);
    y    += bs;
  }
}

PETSC_STATIC_INLINE PetscErrorCode UnPack_bs(PetscInt n,const PetscScalar *x,const PetscInt *indicesy,PetscScalar *y,InsertMode addv,PetscInt bs)
{
  PetscInt i,idy,j;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  switch (addv) {
  case INSERT_VALUES:
  case INSERT_ALL_VALUES:
    for (i=0; i<n; i++) {
      idy       = *indicesy++;
      ierr = PetscArraycpy(y + idy,x,bs);CHKERRQ(ierr);
      x        += bs;
    }
    break;
  case ADD_VALUES:
  case ADD_ALL_VALUES:
    for (i=0; i<n; i++) {
      idy        = *indicesy++;
      for (j=0; j<bs; j++) y[idy+j] += x[j];
      x         += bs;
    }
    break;
#if !defined(PETSC_USE_COMPLEX)
  case MAX_VALUES:
    for (i=0; i<n; i++) {
      idy = *indicesy++;
      for (j=0; j<bs; j++) y[idy+j] = PetscMax(y[idy+j],x[j]);
      x  += bs;
    }
#else
  case MAX_VALUES:
#endif
  case NOT_SET_VALUES:
    break;
  default:
    SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Cannot handle insert mode %d", addv);
  }
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode Scatter_bs(PetscInt n,const PetscInt *indicesx,const PetscScalar *x,const PetscInt *indicesy,PetscScalar *y,InsertMode addv,PetscInt bs)
{
  PetscInt i,idx,idy,j;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  switch (addv) {
  case INSERT_VALUES:
  case INSERT_ALL_VALUES:
    for (i=0; i<n; i++) {
      idx       = *indicesx++;
      idy       = *indicesy++;
      ierr = PetscArraycpy(y + idy, x + idx,bs);CHKERRQ(ierr);
    }
    break;
  case ADD_VALUES:
  case ADD_ALL_VALUES:
    for (i=0; i<n; i++) {
      idx        = *indicesx++;
      idy        = *indicesy++;
      for (j=0; j<bs; j++ )  y[idy+j] += x[idx+j];
    }
    break;
#if !defined(PETSC_USE_COMPLEX)
  case MAX_VALUES:
    for (i=0; i<n; i++) {
      idx       = *indicesx++;
      idy       = *indicesy++;
      for (j=0; j<bs; j++ )  y[idy+j] = PetscMax(y[idy+j],x[idx+j]);
    }
#else
  case MAX_VALUES:
#endif
  case NOT_SET_VALUES:
    break;
  default:
    SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Cannot handle insert mode %d", addv);
  }
  PetscFunctionReturn(0);
}

/* Create the VecScatterBegin/End_P for our chosen block sizes */
#define BS 1
#include <../src/vec/vscat/impls/mpi3/vpscat.h>
#define BS 2
#include <../src/vec/vscat/impls/mpi3/vpscat.h>
#define BS 3
#include <../src/vec/vscat/impls/mpi3/vpscat.h>
#define BS 4
#include <../src/vec/vscat/impls/mpi3/vpscat.h>
#define BS 5
#include <../src/vec/vscat/impls/mpi3/vpscat.h>
#define BS 6
#include <../src/vec/vscat/impls/mpi3/vpscat.h>
#define BS 7
#include <../src/vec/vscat/impls/mpi3/vpscat.h>
#define BS 8
#include <../src/vec/vscat/impls/mpi3/vpscat.h>
#define BS 9
#include <../src/vec/vscat/impls/mpi3/vpscat.h>
#define BS 10
#include <../src/vec/vscat/impls/mpi3/vpscat.h>
#define BS 11
#include <../src/vec/vscat/impls/mpi3/vpscat.h>
#define BS 12
#include <../src/vec/vscat/impls/mpi3/vpscat.h>
#define BS bs
#include <../src/vec/vscat/impls/mpi3/vpscat.h>

/* ==========================================================================================*/
/*              create parallel to sequential scatter context                                */

extern PetscErrorCode VecScatterCreateCommon_PtoS_MPI3(VecScatter_MPI_General*,VecScatter_MPI_General*,VecScatter);

/*
   bs indicates how many elements there are in each block. Normally this would be 1.

   contains check that PetscMPIInt can handle the sizes needed
*/
#include <petscbt.h>
PetscErrorCode VecScatterCreateLocal_PtoS_MPI3(PetscInt nx,const PetscInt *inidx,PetscInt ny,const PetscInt *inidy,Vec xin,Vec yin,PetscInt bs,VecScatter ctx)
{
  VecScatter_MPI_General *from,*to;
  PetscMPIInt            size,rank,mrank,imdex,tag,n;
  PetscInt               *source = NULL,*owners = NULL,nxr;
  PetscInt               *lowner = NULL,*start = NULL,*lsharedowner = NULL,*sharedstart = NULL,lengthy,lengthx;
  PetscMPIInt            *nprocs = NULL,nrecvs,nrecvshared;
  PetscInt               i,j,idx = 0,nsends;
  PetscMPIInt            *owner = NULL;
  PetscInt               *starts = NULL,count,slen,slenshared;
  PetscInt               *rvalues,*svalues,base,*values,nprocslocal,recvtotal,*rsvalues;
  PetscMPIInt            *onodes1,*olengths1;
  MPI_Comm               comm;
  MPI_Request            *send_waits = NULL,*recv_waits = NULL;
  MPI_Status             recv_status,*send_status;
  PetscErrorCode         ierr;
  PetscShmComm           scomm;
  PetscMPIInt            jj;
  MPI_Info               info;
  MPI_Comm               mscomm;
  MPI_Win                sharedoffsetwin;  /* Window that owns sharedspaceoffset */
  PetscInt               *sharedspaceoffset;
  VecScatterType         type;
  PetscBool              mpi3node;
  PetscFunctionBegin;
  ierr   = PetscObjectGetNewTag((PetscObject)ctx,&tag);CHKERRQ(ierr);
  ierr   = PetscObjectGetComm((PetscObject)xin,&comm);CHKERRQ(ierr);
  ierr   = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
  ierr   = MPI_Comm_size(comm,&size);CHKERRQ(ierr);

  ierr = VecScatterGetType(ctx,&type);CHKERRQ(ierr);
  ierr = PetscStrcmp(type,"mpi3node",&mpi3node);CHKERRQ(ierr);

  ierr = PetscShmCommGet(comm,&scomm);CHKERRQ(ierr);
  ierr = PetscShmCommGetMpiShmComm(scomm,&mscomm);CHKERRQ(ierr);

  ierr = MPI_Info_create(&info);CHKERRQ(ierr);
  ierr = MPI_Info_set(info, "alloc_shared_noncontig", "true");CHKERRQ(ierr);

  if (mpi3node) {
    /* Check if (parallel) inidx has duplicate indices.
     Current VecScatterEndMPI3Node() (case StoP) writes the sequential vector to the parallel vector.
     Writing to the same shared location without variable locking
     leads to incorrect scattering. See src/vec/vscat/examples/runex2_5 and runex3_5 */

    PetscInt    *mem,**optr;
    MPI_Win     swin;
    PetscMPIInt msize;

    ierr = MPI_Comm_size(mscomm,&msize);CHKERRQ(ierr);
    ierr = MPI_Comm_rank(mscomm,&mrank);CHKERRQ(ierr);

    ierr = PetscMalloc1(msize,&optr);CHKERRQ(ierr);
    ierr = MPIU_Win_allocate_shared((nx+1)*sizeof(PetscInt),sizeof(PetscInt),MPI_INFO_NULL,mscomm,&mem,&swin);CHKERRQ(ierr);

    /* Write local nx and inidx into mem */
    mem[0] = nx;
    for (i=1; i<=nx; i++) mem[i] = inidx[i-1];
    ierr = MPI_Barrier(mscomm);CHKERRQ(ierr); /* sync shared memory */

    if (!mrank) {
      PetscBT        table;
      /* sz and dsp_unit are not used. Replacing them with NULL would cause MPI_Win_shared_query() crash */
      MPI_Aint       sz;
      PetscMPIInt    dsp_unit;
      PetscInt       N = xin->map->N;

      ierr = PetscBTCreate(N,&table);CHKERRQ(ierr); /* may not be scalable */
      ierr = PetscBTMemzero(N,table);CHKERRQ(ierr);

      jj = 0;
      while (jj<msize && !ctx->is_duplicate) {
        if (jj == mrank) {jj++; continue;}
        ierr = MPIU_Win_shared_query(swin,jj,&sz,&dsp_unit,&optr[jj]);CHKERRQ(ierr);
        for (j=1; j<=optr[jj][0]; j++) { /* optr[jj][0] = nx */
          if (PetscBTLookupSet(table,optr[jj][j])) {
            ctx->is_duplicate = PETSC_TRUE; /* only mrank=0 has this info., will be checked in VecScatterEndMPI3Node() */
            break;
          }
        }
        jj++;
      }
      ierr = PetscBTDestroy(&table);CHKERRQ(ierr);
    }

    if (swin != MPI_WIN_NULL) {ierr = MPI_Win_free(&swin);CHKERRQ(ierr);}
    ierr = PetscFree(optr);CHKERRQ(ierr);
  }

  owners = xin->map->range;
  ierr   = VecGetSize(yin,&lengthy);CHKERRQ(ierr);
  ierr   = VecGetSize(xin,&lengthx);CHKERRQ(ierr);

  /*  first count number of contributors to each processor */
  ierr = PetscMalloc2(size,&nprocs,nx,&owner);CHKERRQ(ierr);
  ierr = PetscArrayzero(nprocs,size);CHKERRQ(ierr);

  j      = 0;
  nsends = 0;
  for (i=0; i<nx; i++) {
    ierr = PetscIntMultError(bs,inidx[i],&idx);CHKERRQ(ierr);
    if (idx < owners[j]) j = 0;
    for (; j<size; j++) {
      if (idx < owners[j+1]) {
        if (!nprocs[j]++) nsends++;
        owner[i] = j;
        break;
      }
    }
    if (j == size) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_PLIB,"ith %D block entry %D not owned by any process, upper bound %D",i,idx,owners[size]);
  }

  nprocslocal  = nprocs[rank];
  nprocs[rank] = 0;
  if (nprocslocal) nsends--;
  /* inform other processors of number of messages and max length*/
  ierr = PetscGatherNumberOfMessages(comm,NULL,nprocs,&nrecvs);CHKERRQ(ierr);
  ierr = PetscGatherMessageLengths(comm,nsends,nrecvs,nprocs,&onodes1,&olengths1);CHKERRQ(ierr);
  ierr = PetscSortMPIIntWithArray(nrecvs,onodes1,olengths1);CHKERRQ(ierr);
  recvtotal = 0;
  for (i=0; i<nrecvs; i++) {
    ierr = PetscIntSumError(recvtotal,olengths1[i],&recvtotal);CHKERRQ(ierr);
  }

  /* post receives:   */
  ierr  = PetscMalloc3(recvtotal,&rvalues,nrecvs,&source,nrecvs,&recv_waits);CHKERRQ(ierr);
  count = 0;
  for (i=0; i<nrecvs; i++) {
    ierr   = MPI_Irecv((rvalues+count),olengths1[i],MPIU_INT,onodes1[i],tag,comm,recv_waits+i);CHKERRQ(ierr);
    count += olengths1[i];
  }

  /* do sends:
     1) starts[i] gives the starting index in svalues for stuff going to
     the ith processor
  */
  nxr = 0;
  for (i=0; i<nx; i++) {
    if (owner[i] != rank) nxr++;
  }
  ierr = PetscMalloc3(nxr,&svalues,nsends,&send_waits,size+1,&starts);CHKERRQ(ierr);

  starts[0]  = 0;
  for (i=1; i<size; i++) starts[i] = starts[i-1] + nprocs[i-1];
  for (i=0; i<nx; i++) {
    if (owner[i] != rank) svalues[starts[owner[i]]++] = bs*inidx[i];
  }
  starts[0] = 0;
  for (i=1; i<size+1; i++) starts[i] = starts[i-1] + nprocs[i-1];
  count = 0;
  for (i=0; i<size; i++) {
    if (nprocs[i]) {
      ierr = MPI_Isend(svalues+starts[i],nprocs[i],MPIU_INT,i,tag,comm,send_waits+count++);CHKERRQ(ierr);
    }
  }

  /*  wait on receives; this is everyone who we need to deliver data to */
  count = nrecvs;
  slen  = 0;
  nrecvshared = 0;
  slenshared  = 0;
  while (count) {
    ierr = MPI_Waitany(nrecvs,recv_waits,&imdex,&recv_status);CHKERRQ(ierr);
    /* unpack receives into our local space */
    ierr  = MPI_Get_count(&recv_status,MPIU_INT,&n);CHKERRQ(ierr);
    ierr = PetscShmCommGlobalToLocal(scomm,onodes1[imdex],&jj);CHKERRQ(ierr);
    if (jj > -1) {
      ierr = PetscInfo3(NULL,"[%d] Sending values to shared memory partner %d,global rank %d\n",rank,jj,onodes1[imdex]);CHKERRQ(ierr);
      nrecvshared++;
      slenshared += n;
    }
    slen += n;
    count--;
  }

  if (slen != recvtotal) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Total message lengths %D not expected %D",slen,recvtotal);

  /* allocate entire send scatter context */
  ierr          = PetscNewLog(ctx,&to);CHKERRQ(ierr);
  to->sharedwin = MPI_WIN_NULL;
  to->n         = nrecvs-nrecvshared;

  ierr = PetscMalloc1(nrecvs-nrecvshared,&to->requests);CHKERRQ(ierr);
  ierr = PetscMalloc4(bs*(slen-slenshared),&to->values,slen-slenshared,&to->indices,nrecvs-nrecvshared+1,&to->starts,nrecvs-nrecvshared,&to->procs);CHKERRQ(ierr);
  ierr = PetscMalloc2(PetscMax(to->n,nsends),&to->sstatus,PetscMax(to->n,nsends),&to->rstatus);CHKERRQ(ierr);

  ierr = MPI_Comm_size(mscomm,&to->msize);CHKERRQ(ierr);
  ierr = PetscMalloc1(slenshared,&to->sharedspaceindices);CHKERRQ(ierr);
  ierr = PetscCalloc1(to->msize+1,&to->sharedspacestarts);CHKERRQ(ierr);

  ctx->todata              = (void*)to;
  to->starts[0]            = 0;
  to->sharedspacestarts[0] = 0;

  /* Allocate shared memory space for shared memory partner communication */
  ierr = MPIU_Win_allocate_shared(to->msize*sizeof(PetscInt),sizeof(PetscInt),info,mscomm,&sharedspaceoffset,&sharedoffsetwin);CHKERRQ(ierr);
  for (i=0; i<to->msize; i++) sharedspaceoffset[i] = -1; /* mark with -1 for later error checking */
  if (nrecvs) {
    /* move the data into the send scatter */
    base     = owners[rank];
    rsvalues = rvalues;
    to->n    = 0;
    for (i=0; i<nrecvs; i++) {
      values = rsvalues;
      ierr = PetscShmCommGlobalToLocal(scomm,(PetscMPIInt)onodes1[i],&jj);CHKERRQ(ierr);
      if (jj > -1) {
        to->sharedspacestarts[jj]   = to->sharedcnt;
        to->sharedspacestarts[jj+1] = to->sharedcnt + olengths1[i];
        for (j=0; j<olengths1[i]; j++) to->sharedspaceindices[to->sharedcnt + j] = values[j] - base;
        sharedspaceoffset[jj]      = to->sharedcnt;
        to->sharedcnt             += olengths1[i];
        ierr = PetscInfo4(NULL,"[%d] Sending %d values to shared memory partner %d,global rank %d\n",rank,olengths1[i],jj,onodes1[i]);CHKERRQ(ierr);
      } else {
        to->starts[to->n+1] = to->starts[to->n] + olengths1[i];
        to->procs[to->n]    = onodes1[i];
        for (j=0; j<olengths1[i]; j++) to->indices[to->starts[to->n] + j] = values[j] - base;
        to->n++;
      }
      rsvalues += olengths1[i];
    }
  }
  ierr = PetscFree(olengths1);CHKERRQ(ierr);
  ierr = PetscFree(onodes1);CHKERRQ(ierr);
  ierr = PetscFree3(rvalues,source,recv_waits);CHKERRQ(ierr);

  if (mpi3node) {
    /* to->sharedspace is used only as a flag */
    i = 0;
    if (to->sharedcnt) i = 1;
    ierr = MPIU_Win_allocate_shared(i*sizeof(PetscScalar),sizeof(PetscScalar),info,mscomm,&to->sharedspace,&to->sharedwin);CHKERRQ(ierr);
  } else { /* mpi3 */
    ierr = MPIU_Win_allocate_shared(bs*to->sharedcnt*sizeof(PetscScalar),sizeof(PetscScalar),info,mscomm,&to->sharedspace,&to->sharedwin);CHKERRQ(ierr);
  }
  if (to->sharedwin == MPI_WIN_NULL) SETERRQ(PETSC_COMM_SELF,100,"what the");
  ierr = MPI_Info_free(&info);CHKERRQ(ierr);

  /* allocate entire receive scatter context */
  ierr = PetscNewLog(ctx,&from);CHKERRQ(ierr);
  from->sharedwin       = MPI_WIN_NULL;
  from->msize           = to->msize;
  /* we don't actually know the correct value for from->n at this point so we use the upper bound */
  from->n = nsends;

  ierr = PetscMalloc1(nsends,&from->requests);CHKERRQ(ierr);
  ierr = PetscMalloc4((ny-nprocslocal)*bs,&from->values,ny-nprocslocal,&from->indices,nsends+1,&from->starts,from->n,&from->procs);CHKERRQ(ierr);
  ctx->fromdata = (void*)from;

  ierr  = PetscCalloc1(to->msize+1,&from->sharedspacestarts);CHKERRQ(ierr);
  /* move data into receive scatter */
  ierr = PetscMalloc2(size,&lowner,nsends+1,&start);CHKERRQ(ierr);
  ierr = PetscMalloc2(size,&lsharedowner,to->msize+1,&sharedstart);CHKERRQ(ierr);
  count = 0; from->starts[0] = start[0] = 0;
  from->sharedcnt = 0;
  for (i=0; i<size; i++) {
    lsharedowner[i] = -1;
    lowner[i]       = -1;
    if (nprocs[i]) {
      ierr = PetscShmCommGlobalToLocal(scomm,i,&jj);CHKERRQ(ierr);
      if (jj > -1) {
        from->sharedspacestarts[jj] = sharedstart[jj] = from->sharedcnt;
        from->sharedspacestarts[jj+1] = sharedstart[jj+1] = from->sharedspacestarts[jj] + nprocs[i];
        from->sharedcnt += nprocs[i];
        lsharedowner[i] = jj;
      } else {
        lowner[i]            = count++;
        from->procs[count-1]  = i;
        from->starts[count] = start[count] = start[count-1] + nprocs[i];
      }
    }
  }
  from->n = count;

  for (i=0; i<nx; i++) {
    if (owner[i] != rank && lowner[owner[i]] > -1) {
      from->indices[start[lowner[owner[i]]]++] = bs*inidy[i];
      if (bs*inidy[i] >= lengthy) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Scattering past end of TO vector");
    }
  }

  /* copy over appropriate parts of inidy[] into from->sharedspaceindices */
  if (mpi3node) {
    /* in addition, copy over appropriate parts of inidx[] into 2nd part of from->sharedspaceindices */
    PetscInt *sharedspaceindicesx;
    ierr = PetscMalloc1(2*from->sharedcnt,&from->sharedspaceindices);CHKERRQ(ierr);
    sharedspaceindicesx = from->sharedspaceindices + from->sharedcnt;
    for (i=0; i<nx; i++) {
      if (owner[i] != rank && lsharedowner[owner[i]] > -1) {
        if (sharedstart[lsharedowner[owner[i]]] > from->sharedcnt) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_MEM,"won");

        from->sharedspaceindices[sharedstart[lsharedowner[owner[i]]]] = bs*inidy[i];
        sharedspaceindicesx[sharedstart[lsharedowner[owner[i]]]] = bs*inidx[i]-owners[owner[i]]; /* local inidx */
        sharedstart[lsharedowner[owner[i]]]++;
      }
    }
  } else { /* mpi3 */
    ierr = PetscMalloc1(from->sharedcnt,&from->sharedspaceindices);CHKERRQ(ierr);
    for (i=0; i<nx; i++) {
      if (owner[i] != rank && lsharedowner[owner[i]] > -1) {
        if (sharedstart[lsharedowner[owner[i]]] > from->sharedcnt) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_MEM,"won");
        from->sharedspaceindices[sharedstart[lsharedowner[owner[i]]]++] = bs*inidy[i];
      }
    }
  }

  ierr = PetscFree2(lowner,start);CHKERRQ(ierr);
  ierr = PetscFree2(lsharedowner,sharedstart);CHKERRQ(ierr);
  ierr = PetscFree2(nprocs,owner);CHKERRQ(ierr);

  /* wait on sends */
  if (nsends) {
    ierr = PetscMalloc1(nsends,&send_status);CHKERRQ(ierr);
    ierr = MPI_Waitall(nsends,send_waits,send_status);CHKERRQ(ierr);
    ierr = PetscFree(send_status);CHKERRQ(ierr);
  }
  ierr = PetscFree3(svalues,send_waits,starts);CHKERRQ(ierr);

  if (nprocslocal) {
    PetscInt nt = from->local.n = to->local.n = nprocslocal;
    /* we have a scatter to ourselves */
    ierr = PetscMalloc1(nt,&to->local.vslots);CHKERRQ(ierr);
    ierr = PetscMalloc1(nt,&from->local.vslots);CHKERRQ(ierr);
    nt   = 0;
    for (i=0; i<nx; i++) {
      idx = bs*inidx[i];
      if (idx >= owners[rank] && idx < owners[rank+1]) {
        to->local.vslots[nt]     = idx - owners[rank];
        from->local.vslots[nt++] = bs*inidy[i];
        if (bs*inidy[i] >= lengthy) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Scattering past end of TO vector");
      }
    }
    ierr = PetscLogObjectMemory((PetscObject)ctx,2*nt*sizeof(PetscInt));CHKERRQ(ierr);
  } else {
    from->local.n      = 0;
    from->local.vslots = 0;
    to->local.n        = 0;
    to->local.vslots   = 0;
  }

  /* Get the shared memory address for all processes we will be copying data from */
  ierr = PetscCalloc1(to->msize,&from->sharedspaces);CHKERRQ(ierr);
  ierr = PetscCalloc1(to->msize,&from->sharedspacesoffset);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(mscomm,&mrank);CHKERRQ(ierr);
  for (jj=0; jj<to->msize; jj++) {
    MPI_Aint    isize;
    PetscMPIInt disp_unit;
    PetscInt    *ptr;
    ierr = MPIU_Win_shared_query(to->sharedwin,jj,&isize,&disp_unit,&from->sharedspaces[jj]);CHKERRQ(ierr);
    ierr = MPIU_Win_shared_query(sharedoffsetwin,jj,&isize,&disp_unit,&ptr);CHKERRQ(ierr);
    from->sharedspacesoffset[jj] = ptr[mrank];
  }
  ierr = MPI_Win_free(&sharedoffsetwin);CHKERRQ(ierr);

  if (mpi3node && !from->sharedspace) {
    /* comput from->notdone to be used by VecScatterEndMPI3Node() */
    PetscInt notdone = 0;
    for (i=0; i<from->msize; i++) {
      if (from->sharedspacesoffset && from->sharedspacesoffset[i] > -1) {
        notdone++;
      }
    }
    from->notdone = notdone;
  }

  from->local.nonmatching_computed = PETSC_FALSE;
  from->local.n_nonmatching        = 0;
  from->local.slots_nonmatching    = 0;
  to->local.nonmatching_computed   = PETSC_FALSE;
  to->local.n_nonmatching          = 0;
  to->local.slots_nonmatching      = 0;

  from->format = VEC_SCATTER_MPI_GENERAL;
  to->format   = VEC_SCATTER_MPI_GENERAL;
  from->bs     = bs;
  to->bs       = bs;

  ierr = VecScatterCreateCommon_PtoS_MPI3(from,to,ctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
   bs indicates how many elements there are in each block. Normally this would be 1.
*/
PetscErrorCode VecScatterCreateCommon_PtoS_MPI3(VecScatter_MPI_General *from,VecScatter_MPI_General *to,VecScatter ctx)
{
  MPI_Comm       comm;
  PetscMPIInt    tag  = ((PetscObject)ctx)->tag, tagr;
  PetscInt       bs   = to->bs;
  PetscMPIInt    size;
  PetscInt       i, n;
  PetscErrorCode ierr;
  VecScatterType type;
  PetscBool      mpi3node;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)ctx,&comm);CHKERRQ(ierr);
  ierr = PetscObjectGetNewTag((PetscObject)ctx,&tagr);CHKERRQ(ierr);
  ctx->ops->destroy = VecScatterDestroy_PtoP_MPI3;

  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  /* check if the receives are ALL going into contiguous locations; if so can skip indexing */
  to->contiq = PETSC_FALSE;
  n = from->starts[from->n];
  from->contiq = PETSC_TRUE;
  for (i=1; i<n; i++) {
    if (from->indices[i] != from->indices[i-1] + bs) {
      from->contiq = PETSC_FALSE;
      break;
    }
  }

  ctx->ops->copy      = VecScatterCopy_PtoP_AllToAll;
  {
    PetscInt    *sstarts  = to->starts,  *rstarts = from->starts;
    PetscMPIInt *sprocs   = to->procs,   *rprocs  = from->procs;
    MPI_Request *swaits   = to->requests,*rwaits  = from->requests;
    MPI_Request *rev_swaits,*rev_rwaits;
    PetscScalar *Ssvalues = to->values, *Srvalues = from->values;

    /* allocate additional wait variables for the "reverse" scatter */
    ierr = PetscMalloc1(to->n,&rev_rwaits);CHKERRQ(ierr);
    ierr = PetscMalloc1(from->n,&rev_swaits);CHKERRQ(ierr);
    to->rev_requests   = rev_rwaits;
    from->rev_requests = rev_swaits;

    for (i=0; i<from->n; i++) {
      ierr = MPI_Send_init(Srvalues+bs*rstarts[i],bs*rstarts[i+1]-bs*rstarts[i],MPIU_SCALAR,rprocs[i],tagr,comm,rev_swaits+i);CHKERRQ(ierr);
    }

    for (i=0; i<to->n; i++) {
      ierr = MPI_Send_init(Ssvalues+bs*sstarts[i],bs*sstarts[i+1]-bs*sstarts[i],MPIU_SCALAR,sprocs[i],tag,comm,swaits+i);CHKERRQ(ierr);
    }
    /* Register receives for scatter and reverse */
    for (i=0; i<from->n; i++) {
      ierr = MPI_Recv_init(Srvalues+bs*rstarts[i],bs*rstarts[i+1]-bs*rstarts[i],MPIU_SCALAR,rprocs[i],tag,comm,rwaits+i);CHKERRQ(ierr);
    }
    for (i=0; i<to->n; i++) {
      ierr = MPI_Recv_init(Ssvalues+bs*sstarts[i],bs*sstarts[i+1]-bs*sstarts[i],MPIU_SCALAR,sprocs[i],tagr,comm,rev_rwaits+i);CHKERRQ(ierr);
    }
    ctx->ops->copy = VecScatterCopy_PtoP_X;
  }
  ierr = PetscInfo1(ctx,"Using blocksize %D scatter\n",bs);CHKERRQ(ierr);

  if (PetscDefined(USE_DEBUG)) {
    ierr = MPIU_Allreduce(&bs,&i,1,MPIU_INT,MPI_MIN,PetscObjectComm((PetscObject)ctx));CHKERRQ(ierr);
    ierr = MPIU_Allreduce(&bs,&n,1,MPIU_INT,MPI_MAX,PetscObjectComm((PetscObject)ctx));CHKERRQ(ierr);
    if (bs!=i || bs!=n) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Blocks size %D != %D or %D",bs,i,n);
  }

  ierr = VecScatterGetType(ctx,&type);CHKERRQ(ierr);
  ierr = PetscStrcmp(type,"mpi3node",&mpi3node);CHKERRQ(ierr);

  if (mpi3node) {
    switch (bs) {
    case 12:
      ctx->ops->begin = VecScatterBeginMPI3Node_12;
      ctx->ops->end   = VecScatterEndMPI3Node_12;
      break;
    case 11:
      ctx->ops->begin = VecScatterBeginMPI3Node_11;
      ctx->ops->end   = VecScatterEndMPI3Node_11;
      break;
    case 10:
      ctx->ops->begin = VecScatterBeginMPI3Node_10;
      ctx->ops->end   = VecScatterEndMPI3Node_10;
      break;
    case 9:
      ctx->ops->begin = VecScatterBeginMPI3Node_9;
      ctx->ops->end   = VecScatterEndMPI3Node_9;
      break;
    case 8:
      ctx->ops->begin = VecScatterBeginMPI3Node_8;
      ctx->ops->end   = VecScatterEndMPI3Node_8;
      break;
    case 7:
      ctx->ops->begin = VecScatterBeginMPI3Node_7;
      ctx->ops->end   = VecScatterEndMPI3Node_7;
      break;
    case 6:
      ctx->ops->begin = VecScatterBeginMPI3Node_6;
      ctx->ops->end   = VecScatterEndMPI3Node_6;
      break;
    case 5:
      ctx->ops->begin = VecScatterBeginMPI3Node_5;
      ctx->ops->end   = VecScatterEndMPI3Node_5;
      break;
    case 4:
      ctx->ops->begin = VecScatterBeginMPI3Node_4;
      ctx->ops->end   = VecScatterEndMPI3Node_4;
      break;
    case 3:
      ctx->ops->begin = VecScatterBeginMPI3Node_3;
      ctx->ops->end   = VecScatterEndMPI3Node_3;
      break;
    case 2:
      ctx->ops->begin = VecScatterBeginMPI3Node_2;
      ctx->ops->end   = VecScatterEndMPI3Node_2;
      break;
    case 1:
      ctx->ops->begin = VecScatterBeginMPI3Node_1;
      ctx->ops->end   = VecScatterEndMPI3Node_1;
      break;
    default:
      ctx->ops->begin = VecScatterBegin_bs;
      ctx->ops->end   = VecScatterEnd_bs;
    }
  } else { /* !mpi3node */

    switch (bs) {
    case 12:
      ctx->ops->begin = VecScatterBegin_12;
      ctx->ops->end   = VecScatterEnd_12;
      break;
    case 11:
      ctx->ops->begin = VecScatterBegin_11;
      ctx->ops->end   = VecScatterEnd_11;
      break;
    case 10:
      ctx->ops->begin = VecScatterBegin_10;
      ctx->ops->end   = VecScatterEnd_10;
      break;
    case 9:
      ctx->ops->begin = VecScatterBegin_9;
      ctx->ops->end   = VecScatterEnd_9;
      break;
    case 8:
      ctx->ops->begin = VecScatterBegin_8;
      ctx->ops->end   = VecScatterEnd_8;
      break;
    case 7:
      ctx->ops->begin = VecScatterBegin_7;
      ctx->ops->end   = VecScatterEnd_7;
      break;
    case 6:
      ctx->ops->begin = VecScatterBegin_6;
      ctx->ops->end   = VecScatterEnd_6;
      break;
    case 5:
      ctx->ops->begin = VecScatterBegin_5;
      ctx->ops->end   = VecScatterEnd_5;
      break;
    case 4:
      ctx->ops->begin = VecScatterBegin_4;
      ctx->ops->end   = VecScatterEnd_4;
      break;
    case 3:
      ctx->ops->begin = VecScatterBegin_3;
      ctx->ops->end   = VecScatterEnd_3;
      break;
    case 2:
      ctx->ops->begin = VecScatterBegin_2;
      ctx->ops->end   = VecScatterEnd_2;
      break;
    case 1:
      if (mpi3node) {
        ctx->ops->begin = VecScatterBeginMPI3Node_1;
        ctx->ops->end   = VecScatterEndMPI3Node_1;
      } else {
        ctx->ops->begin = VecScatterBegin_1;
        ctx->ops->end   = VecScatterEnd_1;
      }
      break;
    default:
      ctx->ops->begin = VecScatterBegin_bs;
      ctx->ops->end   = VecScatterEnd_bs;
    }
  }

  ctx->ops->view = VecScatterView_MPI;

  /* try to optimize PtoP vecscatter with memcpy's */
  ierr = VecScatterMemcpyPlanCreate_PtoP(to,from);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------------------------------*/
/*
         Scatter from local Seq vectors to a parallel vector.
         Reverses the order of the arguments, calls VecScatterCreateLocal_PtoS() then
         reverses the result.
*/
PetscErrorCode VecScatterCreateLocal_StoP_MPI3(PetscInt nx,const PetscInt *inidx,PetscInt ny,const PetscInt *inidy,Vec xin,Vec yin,PetscInt bs,VecScatter ctx)
{
  PetscErrorCode         ierr;
  MPI_Request            *waits;
  VecScatter_MPI_General *to,*from;

  PetscFunctionBegin;
  ierr          = VecScatterCreateLocal_PtoS_MPI3(ny,inidy,nx,inidx,yin,xin,bs,ctx);CHKERRQ(ierr);
  to            = (VecScatter_MPI_General*)ctx->fromdata;
  from          = (VecScatter_MPI_General*)ctx->todata;
  ctx->todata   = (void*)to;
  ctx->fromdata = (void*)from;
  /* these two are special, they are ALWAYS stored in to struct */
  to->sstatus   = from->sstatus;
  to->rstatus   = from->rstatus;

  from->sstatus = 0;
  from->rstatus = 0;
  waits              = from->rev_requests;
  from->rev_requests = from->requests;
  from->requests     = waits;
  waits              = to->rev_requests;
  to->rev_requests   = to->requests;
  to->requests       = waits;
  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------------------------*/
PetscErrorCode VecScatterCreateLocal_PtoP_MPI3(PetscInt nx,const PetscInt *inidx,PetscInt ny,const PetscInt *inidy,Vec xin,Vec yin,PetscInt bs,VecScatter ctx)
{
  PetscErrorCode ierr;
  PetscMPIInt    size,rank,tag,imdex,n;
  PetscInt       *owners = xin->map->range;
  PetscMPIInt    *nprocs = NULL;
  PetscInt       i,j,idx,nsends,*local_inidx = NULL,*local_inidy = NULL;
  PetscMPIInt    *owner   = NULL;
  PetscInt       *starts  = NULL,count,slen;
  PetscInt       *rvalues = NULL,*svalues = NULL,base,*values = NULL,*rsvalues,recvtotal,lastidx;
  PetscMPIInt    *onodes1,*olengths1,nrecvs;
  MPI_Comm       comm;
  MPI_Request    *send_waits = NULL,*recv_waits = NULL;
  MPI_Status     recv_status,*send_status = NULL;
  PetscBool      duplicate = PETSC_FALSE;
  PetscBool      found = PETSC_FALSE;

  PetscFunctionBegin;
  ierr = PetscObjectGetNewTag((PetscObject)ctx,&tag);CHKERRQ(ierr);
  ierr = PetscObjectGetComm((PetscObject)xin,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
  if (size == 1) {
    ierr = VecScatterCreateLocal_StoP_MPI3(nx,inidx,ny,inidy,xin,yin,bs,ctx);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }

  /*
     Each processor ships off its inidx[j] and inidy[j] to the appropriate processor
     They then call the StoPScatterCreate()
  */
  /*  first count number of contributors to each processor */
  ierr = PetscMalloc3(size,&nprocs,nx,&owner,(size+1),&starts);CHKERRQ(ierr);
  ierr = PetscArrayzero(nprocs,size);CHKERRQ(ierr);

  lastidx = -1;
  j       = 0;
  for (i=0; i<nx; i++) {
    /* if indices are NOT locally sorted, need to start search at the beginning */
    if (lastidx > (idx = bs*inidx[i])) j = 0;
    lastidx = idx;
    for (; j<size; j++) {
      if (idx >= owners[j] && idx < owners[j+1]) {
        nprocs[j]++;
        owner[i] = j;
        found = PETSC_TRUE;
        break;
      }
    }
    if (PetscDefined(USE_DEBUG) && !found) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Index %D out of range",idx);
    found = PETSC_FALSE;
  }
  nsends = 0;
  for (i=0; i<size; i++) nsends += (nprocs[i] > 0);

  /* inform other processors of number of messages and max length*/
  ierr = PetscGatherNumberOfMessages(comm,NULL,nprocs,&nrecvs);CHKERRQ(ierr);
  ierr = PetscGatherMessageLengths(comm,nsends,nrecvs,nprocs,&onodes1,&olengths1);CHKERRQ(ierr);
  ierr = PetscSortMPIIntWithArray(nrecvs,onodes1,olengths1);CHKERRQ(ierr);
  recvtotal = 0;
  for (i=0; i<nrecvs; i++) {
    ierr = PetscIntSumError(recvtotal,olengths1[i],&recvtotal);CHKERRQ(ierr);
  }

  /* post receives:   */
  ierr = PetscMalloc5(2*recvtotal,&rvalues,2*nx,&svalues,nrecvs,&recv_waits,nsends,&send_waits,nsends,&send_status);CHKERRQ(ierr);

  count = 0;
  for (i=0; i<nrecvs; i++) {
    ierr = MPI_Irecv((rvalues+2*count),2*olengths1[i],MPIU_INT,onodes1[i],tag,comm,recv_waits+i);CHKERRQ(ierr);
    ierr = PetscIntSumError(count,olengths1[i],&count);CHKERRQ(ierr);
  }
  ierr = PetscFree(onodes1);CHKERRQ(ierr);

  /* do sends:
      1) starts[i] gives the starting index in svalues for stuff going to
         the ith processor
  */
  starts[0]= 0;
  for (i=1; i<size; i++) starts[i] = starts[i-1] + nprocs[i-1];
  for (i=0; i<nx; i++) {
    svalues[2*starts[owner[i]]]       = bs*inidx[i];
    svalues[1 + 2*starts[owner[i]]++] = bs*inidy[i];
  }

  starts[0] = 0;
  for (i=1; i<size+1; i++) starts[i] = starts[i-1] + nprocs[i-1];
  count = 0;
  for (i=0; i<size; i++) {
    if (nprocs[i]) {
      ierr = MPI_Isend(svalues+2*starts[i],2*nprocs[i],MPIU_INT,i,tag,comm,send_waits+count);CHKERRQ(ierr);
      count++;
    }
  }
  ierr = PetscFree3(nprocs,owner,starts);CHKERRQ(ierr);

  /*  wait on receives */
  count = nrecvs;
  slen  = 0;
  while (count) {
    ierr = MPI_Waitany(nrecvs,recv_waits,&imdex,&recv_status);CHKERRQ(ierr);
    /* unpack receives into our local space */
    ierr  = MPI_Get_count(&recv_status,MPIU_INT,&n);CHKERRQ(ierr);
    slen += n/2;
    count--;
  }
  if (slen != recvtotal) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Total message lengths %D not as expected %D",slen,recvtotal);

  ierr     = PetscMalloc2(slen,&local_inidx,slen,&local_inidy);CHKERRQ(ierr);
  base     = owners[rank];
  count    = 0;
  rsvalues = rvalues;
  for (i=0; i<nrecvs; i++) {
    values    = rsvalues;
    rsvalues += 2*olengths1[i];
    for (j=0; j<olengths1[i]; j++) {
      local_inidx[count]   = values[2*j] - base;
      local_inidy[count++] = values[2*j+1];
    }
  }
  ierr = PetscFree(olengths1);CHKERRQ(ierr);

  /* wait on sends */
  if (nsends) {ierr = MPI_Waitall(nsends,send_waits,send_status);CHKERRQ(ierr);}
  ierr = PetscFree5(rvalues,svalues,recv_waits,send_waits,send_status);CHKERRQ(ierr);

  /*
     should sort and remove duplicates from local_inidx,local_inidy
  */
#if defined(do_it_slow)
  /* sort on the from index */
  ierr  = PetscSortIntWithArray(slen,local_inidx,local_inidy);CHKERRQ(ierr);
  start = 0;
  while (start < slen) {
    count = start+1;
    last  = local_inidx[start];
    while (count < slen && last == local_inidx[count]) count++;
    if (count > start + 1) { /* found 2 or more same local_inidx[] in a row */
      /* sort on to index */
      ierr = PetscSortInt(count-start,local_inidy+start);CHKERRQ(ierr);
    }
    /* remove duplicates; not most efficient way, but probably good enough */
    i = start;
    while (i < count-1) {
      if (local_inidy[i] != local_inidy[i+1]) i++;
      else { /* found a duplicate */
        duplicate = PETSC_TRUE;
        for (j=i; j<slen-1; j++) {
          local_inidx[j] = local_inidx[j+1];
          local_inidy[j] = local_inidy[j+1];
        }
        slen--;
        count--;
      }
    }
    start = count;
  }
#endif
  if (duplicate) {
    ierr = PetscInfo(ctx,"Duplicate from to indices passed in VecScatterCreate(), they are ignored\n");CHKERRQ(ierr);
  }
  ierr = VecScatterCreateLocal_StoP_MPI3(slen,local_inidx,slen,local_inidy,xin,yin,bs,ctx);CHKERRQ(ierr);
  ierr = PetscFree2(local_inidx,local_inidy);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode VecScatterSetUp_MPI3(VecScatter ctx)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecScatterSetUp_vectype_private(ctx,VecScatterCreateLocal_PtoS_MPI3,VecScatterCreateLocal_StoP_MPI3,VecScatterCreateLocal_PtoP_MPI3);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode VecScatterCreate_MPI3(VecScatter ctx)
{
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ctx->ops->setup = VecScatterSetUp_MPI3;
  ierr = PetscObjectChangeTypeName((PetscObject)ctx,VECSCATTERMPI3);CHKERRQ(ierr);
  ierr = PetscInfo(ctx,"Using MPI3 for vector scatter\n");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode VecScatterCreate_MPI3Node(VecScatter ctx)
{
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ctx->ops->setup = VecScatterSetUp_MPI3;
  ierr = PetscObjectChangeTypeName((PetscObject)ctx,VECSCATTERMPI3NODE);CHKERRQ(ierr);
  ierr = PetscInfo(ctx,"Using MPI3NODE for vector scatter\n");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
