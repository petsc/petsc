#define PETSCVEC_DLL

/*
    Defines parallel vector scatters.
*/

#include "private/isimpl.h"
#include "private/vecimpl.h"         /*I "petscvec.h" I*/
#include "../src/vec/vec/impls/dvecimpl.h"
#include "../src/vec/vec/impls/mpi/pvecimpl.h"

#undef __FUNCT__  
#define __FUNCT__ "VecScatterView_MPI"
PetscErrorCode VecScatterView_MPI(VecScatter ctx,PetscViewer viewer)
{
  VecScatter_MPI_General *to=(VecScatter_MPI_General*)ctx->todata;
  VecScatter_MPI_General *from=(VecScatter_MPI_General*)ctx->fromdata;
  PetscErrorCode         ierr;
  PetscInt               i;
  PetscMPIInt            rank;
  PetscViewerFormat      format;
  PetscTruth             iascii;

  PetscFunctionBegin;
  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_ASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    ierr = MPI_Comm_rank(((PetscObject)ctx)->comm,&rank);CHKERRQ(ierr);
    ierr = PetscViewerGetFormat(viewer,&format);CHKERRQ(ierr);
    if (format ==  PETSC_VIEWER_ASCII_INFO) {
      PetscInt nsend_max,nrecv_max,lensend_max,lenrecv_max,alldata,itmp;

      ierr = MPI_Reduce(&to->n,&nsend_max,1,MPIU_INT,MPI_MAX,0,((PetscObject)ctx)->comm);CHKERRQ(ierr);
      ierr = MPI_Reduce(&from->n,&nrecv_max,1,MPIU_INT,MPI_MAX,0,((PetscObject)ctx)->comm);CHKERRQ(ierr);
      itmp = to->starts[to->n+1];
      ierr = MPI_Reduce(&itmp,&lensend_max,1,MPIU_INT,MPI_MAX,0,((PetscObject)ctx)->comm);CHKERRQ(ierr);
      itmp = from->starts[from->n+1];
      ierr = MPI_Reduce(&itmp,&lenrecv_max,1,MPIU_INT,MPI_MAX,0,((PetscObject)ctx)->comm);CHKERRQ(ierr);
      ierr = MPI_Reduce(&itmp,&alldata,1,MPIU_INT,MPI_SUM,0,((PetscObject)ctx)->comm);CHKERRQ(ierr);

      ierr = PetscViewerASCIIPrintf(viewer,"VecScatter statistics\n");CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"  Maximum number sends %D\n",nsend_max);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"  Maximum number receives %D\n",nrecv_max);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"  Maximum data sent %D\n",(int)(lensend_max*to->bs*sizeof(PetscScalar)));CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"  Maximum data received %D\n",(int)(lenrecv_max*to->bs*sizeof(PetscScalar)));CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"  Total data sent %D\n",(int)(alldata*to->bs*sizeof(PetscScalar)));CHKERRQ(ierr);

    } else { 
      ierr = PetscViewerASCIISynchronizedPrintf(viewer,"[%d] Number sends = %D; Number to self = %D\n",rank,to->n,to->local.n);CHKERRQ(ierr);
      if (to->n) {
        for (i=0; i<to->n; i++){
          ierr = PetscViewerASCIISynchronizedPrintf(viewer,"[%d]   %D length = %D to whom %D\n",rank,i,to->starts[i+1]-to->starts[i],to->procs[i]);CHKERRQ(ierr);
        }
        ierr = PetscViewerASCIISynchronizedPrintf(viewer,"Now the indices for all remote sends (in order by process sent to)\n");CHKERRQ(ierr);
        for (i=0; i<to->starts[to->n]; i++){
          ierr = PetscViewerASCIISynchronizedPrintf(viewer,"[%d]%D \n",rank,to->indices[i]);CHKERRQ(ierr);
        }
      }

      ierr = PetscViewerASCIISynchronizedPrintf(viewer,"[%d]Number receives = %D; Number from self = %D\n",rank,from->n,from->local.n);CHKERRQ(ierr);
      if (from->n) {
	for (i=0; i<from->n; i++){
	  ierr = PetscViewerASCIISynchronizedPrintf(viewer,"[%d] %D length %D from whom %D\n",rank,i,from->starts[i+1]-from->starts[i],from->procs[i]);CHKERRQ(ierr);
	}

	ierr = PetscViewerASCIISynchronizedPrintf(viewer,"Now the indices for all remote receives (in order by process received from)\n");CHKERRQ(ierr);
	for (i=0; i<from->starts[from->n]; i++){
	  ierr = PetscViewerASCIISynchronizedPrintf(viewer,"[%d]%D \n",rank,from->indices[i]);CHKERRQ(ierr);
	}
      }
      if (to->local.n) {
        ierr = PetscViewerASCIISynchronizedPrintf(viewer,"[%d]Indices for local part of scatter\n",rank);CHKERRQ(ierr);
        for (i=0; i<to->local.n; i++){
          ierr = PetscViewerASCIISynchronizedPrintf(viewer,"[%d]From %D to %D \n",rank,from->local.vslots[i],to->local.vslots[i]);CHKERRQ(ierr);
        }
      }

      ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
    }
  } else {
    SETERRQ1(PETSC_ERR_SUP,"Viewer type %s not supported for this scatter",((PetscObject)viewer)->type_name);
  }
  PetscFunctionReturn(0);
}  

/* -----------------------------------------------------------------------------------*/
/*
      The next routine determines what part of  the local part of the scatter is an
  exact copy of values into their current location. We check this here and
  then know that we need not perform that portion of the scatter when the vector is
  scattering to itself with INSERT_VALUES.

     This is currently not used but would speed up, for example DALocalToLocalBegin/End()

*/
#undef __FUNCT__  
#define __FUNCT__ "VecScatterLocalOptimize_Private"
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
    ierr = PetscMalloc(n_nonmatching*sizeof(PetscInt),&nto_slots);CHKERRQ(ierr);
    ierr = PetscMalloc(n_nonmatching*sizeof(PetscInt),&nfrom_slots);CHKERRQ(ierr);
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
#undef __FUNCT__  
#define __FUNCT__ "VecScatterDestroy_PtoP"
PetscErrorCode VecScatterDestroy_PtoP(VecScatter ctx)
{
  VecScatter_MPI_General *to   = (VecScatter_MPI_General*)ctx->todata;
  VecScatter_MPI_General *from = (VecScatter_MPI_General*)ctx->fromdata;
  PetscErrorCode         ierr;
  PetscInt               i;

  PetscFunctionBegin;
  if (to->use_readyreceiver) {
    /*
       Since we have already posted sends we must cancel them before freeing 
       the requests
    */
    for (i=0; i<from->n; i++) {
      ierr = MPI_Cancel(from->requests+i);CHKERRQ(ierr);
    }
    for (i=0; i<to->n; i++) {
      ierr = MPI_Cancel(to->rev_requests+i);CHKERRQ(ierr);
    }
    ierr = MPI_Waitall(from->n,from->requests,to->rstatus);CHKERRQ(ierr);
    ierr = MPI_Waitall(to->n,to->rev_requests,to->rstatus);CHKERRQ(ierr);
  }

#if defined(PETSC_HAVE_MPI_ALLTOALLW) && !defined(PETSC_USE_64BIT_INDICES)
  if (to->use_alltoallw) {
    ierr = PetscFree3(to->wcounts,to->wdispls,to->types);CHKERRQ(ierr);
    ierr = PetscFree3(from->wcounts,from->wdispls,from->types);CHKERRQ(ierr);
  }
#endif

#if defined(PETSC_HAVE_MPI_WIN_CREATE)
  if (to->use_window) {
    ierr = MPI_Win_free(&from->window);CHKERRQ(ierr);
    ierr = MPI_Win_free(&to->window);CHKERRQ(ierr);
    ierr = PetscFree(from->winstarts);CHKERRQ(ierr);
    ierr = PetscFree(to->winstarts);CHKERRQ(ierr);
  }
#endif

  if (to->use_alltoallv) {
    ierr = PetscFree2(to->counts,to->displs);CHKERRQ(ierr);
    ierr = PetscFree2(from->counts,from->displs);CHKERRQ(ierr);
  }

  /* release MPI resources obtained with MPI_Send_init() and MPI_Recv_init() */
  /* 
     IBM's PE version of MPI has a bug where freeing these guys will screw up later
     message passing.
  */
#if !defined(PETSC_HAVE_BROKEN_REQUEST_FREE)
  if (!to->use_alltoallv && !to->use_window) {   /* currently the to->requests etc are ALWAYS allocated even if not used */
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
  }
  /*
      MPICH could not properly cancel requests thus with ready receiver mode we
    cannot free the requests. It may be fixed now, if not then put the following 
    code inside a if !to->use_readyreceiver) {
  */
  if (!to->use_alltoallv && !to->use_window) {    /* currently the from->requests etc are ALWAYS allocated even if not used */
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
  }
#endif

  ierr = PetscFree(to->local.vslots);CHKERRQ(ierr);
  ierr = PetscFree(from->local.vslots);CHKERRQ(ierr);
  ierr = PetscFree2(to->counts,to->displs);CHKERRQ(ierr);
  ierr = PetscFree2(from->counts,from->displs);CHKERRQ(ierr);
  ierr = PetscFree(to->local.slots_nonmatching);CHKERRQ(ierr);
  ierr = PetscFree(from->local.slots_nonmatching);CHKERRQ(ierr);
  ierr = PetscFree(to->rev_requests);CHKERRQ(ierr);
  ierr = PetscFree(from->rev_requests);CHKERRQ(ierr);
  ierr = PetscFree(to->requests);
  ierr = PetscFree(from->requests);
  ierr = PetscFree4(to->values,to->indices,to->starts,to->procs);CHKERRQ(ierr);
  ierr = PetscFree2(to->sstatus,to->rstatus);CHKERRQ(ierr);
  ierr = PetscFree4(from->values,from->indices,from->starts,from->procs);CHKERRQ(ierr);
  ierr = PetscFree(from);CHKERRQ(ierr);
  ierr = PetscFree(to);CHKERRQ(ierr);
  ierr = PetscHeaderDestroy(ctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}



/* --------------------------------------------------------------------------------------*/
/*
    Special optimization to see if the local part of the scatter is actually 
    a copy. The scatter routines call PetscMemcpy() instead.
 
*/
#undef __FUNCT__  
#define __FUNCT__ "VecScatterLocalOptimizeCopy_Private"
PetscErrorCode VecScatterLocalOptimizeCopy_Private(VecScatter scatter,VecScatter_Seq_General *to,VecScatter_Seq_General *from,PetscInt bs)
{
  PetscInt       n = to->n,i,*to_slots = to->vslots,*from_slots = from->vslots;
  PetscInt       to_start,from_start;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  to_start   = to_slots[0];
  from_start = from_slots[0];

  for (i=1; i<n; i++) {
    to_start   += bs;
    from_start += bs;
    if (to_slots[i]   != to_start)   PetscFunctionReturn(0);
    if (from_slots[i] != from_start) PetscFunctionReturn(0);
  }
  to->is_copy       = PETSC_TRUE; 
  to->copy_start    = to_slots[0]; 
  to->copy_length   = bs*sizeof(PetscScalar)*n;
  from->is_copy     = PETSC_TRUE;
  from->copy_start  = from_slots[0];
  from->copy_length = bs*sizeof(PetscScalar)*n;
  ierr = PetscInfo(scatter,"Local scatter is a copy, optimizing for it\n");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* --------------------------------------------------------------------------------------*/

#undef __FUNCT__  
#define __FUNCT__ "VecScatterCopy_PtoP_X"
PetscErrorCode VecScatterCopy_PtoP_X(VecScatter in,VecScatter out)
{
  VecScatter_MPI_General *in_to   = (VecScatter_MPI_General*)in->todata;
  VecScatter_MPI_General *in_from = (VecScatter_MPI_General*)in->fromdata,*out_to,*out_from;
  PetscErrorCode         ierr;
  PetscInt               ny,bs = in_from->bs;

  PetscFunctionBegin;
  out->begin     = in->begin;
  out->end       = in->end;
  out->copy      = in->copy;
  out->destroy   = in->destroy;
  out->view      = in->view;

  /* allocate entire send scatter context */
  ierr = PetscNewLog(out,VecScatter_MPI_General,&out_to);CHKERRQ(ierr);
  ierr = PetscNewLog(out,VecScatter_MPI_General,&out_from);CHKERRQ(ierr);

  ny                = in_to->starts[in_to->n];
  out_to->n         = in_to->n; 
  out_to->type      = in_to->type;
  out_to->sendfirst = in_to->sendfirst;

  ierr = PetscMalloc(out_to->n*sizeof(MPI_Request),&out_to->requests);CHKERRQ(ierr);
  ierr = PetscMalloc4(bs*ny,PetscScalar,&out_to->values,ny,PetscInt,&out_to->indices,out_to->n+1,PetscInt,&out_to->starts,out_to->n,PetscMPIInt,&out_to->procs);CHKERRQ(ierr);
  ierr = PetscMalloc2(PetscMax(in_to->n,in_from->n),MPI_Status,&out_to->sstatus,PetscMax(in_to->n,in_from->n),MPI_Status,&out_to->rstatus);CHKERRQ(ierr);
  ierr = PetscMemcpy(out_to->indices,in_to->indices,ny*sizeof(PetscInt));CHKERRQ(ierr);
  ierr = PetscMemcpy(out_to->starts,in_to->starts,(out_to->n+1)*sizeof(PetscInt));CHKERRQ(ierr);
  ierr = PetscMemcpy(out_to->procs,in_to->procs,(out_to->n)*sizeof(PetscMPIInt));CHKERRQ(ierr);
     
  out->todata       = (void*)out_to;
  out_to->local.n   = in_to->local.n;
  out_to->local.nonmatching_computed = PETSC_FALSE;
  out_to->local.n_nonmatching        = 0;
  out_to->local.slots_nonmatching    = 0;
  if (in_to->local.n) {
    ierr = PetscMalloc(in_to->local.n*sizeof(PetscInt),&out_to->local.vslots);CHKERRQ(ierr);
    ierr = PetscMalloc(in_from->local.n*sizeof(PetscInt),&out_from->local.vslots);CHKERRQ(ierr);
    ierr = PetscMemcpy(out_to->local.vslots,in_to->local.vslots,in_to->local.n*sizeof(PetscInt));CHKERRQ(ierr);
    ierr = PetscMemcpy(out_from->local.vslots,in_from->local.vslots,in_from->local.n*sizeof(PetscInt));CHKERRQ(ierr);
  } else {
    out_to->local.vslots   = 0;
    out_from->local.vslots = 0;
  }

  /* allocate entire receive context */
  out_from->type      = in_from->type;
  ny                  = in_from->starts[in_from->n];
  out_from->n         = in_from->n; 
  out_from->sendfirst = in_from->sendfirst;

  ierr = PetscMalloc(out_from->n*sizeof(MPI_Request),&out_from->requests);CHKERRQ(ierr);
  ierr = PetscMalloc4(ny*bs,PetscScalar,&out_from->values,ny,PetscInt,&out_from->indices,out_from->n+1,PetscInt,&out_from->starts,out_from->n,PetscMPIInt,&out_from->procs);CHKERRQ(ierr);
  ierr = PetscMemcpy(out_from->indices,in_from->indices,ny*sizeof(PetscInt));CHKERRQ(ierr);
  ierr = PetscMemcpy(out_from->starts,in_from->starts,(out_from->n+1)*sizeof(PetscInt));CHKERRQ(ierr);
  ierr = PetscMemcpy(out_from->procs,in_from->procs,(out_from->n)*sizeof(PetscMPIInt));CHKERRQ(ierr);
  out->fromdata       = (void*)out_from;
  out_from->local.n   = in_from->local.n;
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
    PetscTruth  flg;
    MPI_Request *swaits  = out_to->requests,*rwaits  = out_from->requests;
    MPI_Request *rev_swaits,*rev_rwaits;
    PetscScalar *Ssvalues = out_to->values, *Srvalues = out_from->values;

    ierr = PetscMalloc(in_to->n*sizeof(MPI_Request),&out_to->rev_requests);CHKERRQ(ierr);
    ierr = PetscMalloc(in_from->n*sizeof(MPI_Request),&out_from->rev_requests);CHKERRQ(ierr);

    rev_rwaits = out_to->rev_requests;
    rev_swaits = out_from->rev_requests;

    out_from->bs = out_to->bs = bs; 
    tag     = ((PetscObject)out)->tag;
    comm    = ((PetscObject)out)->comm;

    /* Register the receives that you will use later (sends for scatter reverse) */
    for (i=0; i<out_from->n; i++) {
      ierr = MPI_Recv_init(Srvalues+bs*rstarts[i],bs*rstarts[i+1]-bs*rstarts[i],MPIU_SCALAR,rprocs[i],tag,comm,rwaits+i);CHKERRQ(ierr);
      ierr = MPI_Send_init(Srvalues+bs*rstarts[i],bs*rstarts[i+1]-bs*rstarts[i],MPIU_SCALAR,rprocs[i],tag,comm,rev_swaits+i);CHKERRQ(ierr);
    }

    flg  = PETSC_FALSE;
    ierr = PetscOptionsGetTruth(PETSC_NULL,"-vecscatter_rsend",&flg,PETSC_NULL);CHKERRQ(ierr);
    if (flg) {
      out_to->use_readyreceiver    = PETSC_TRUE;
      out_from->use_readyreceiver  = PETSC_TRUE;
      for (i=0; i<out_to->n; i++) {
        ierr = MPI_Rsend_init(Ssvalues+bs*sstarts[i],bs*sstarts[i+1]-bs*sstarts[i],MPIU_SCALAR,sprocs[i],tag,comm,swaits+i);CHKERRQ(ierr);
      } 
      if (out_from->n) {ierr = MPI_Startall_irecv(out_from->starts[out_from->n]*out_from->bs,out_from->n,out_from->requests);CHKERRQ(ierr);}
      ierr = MPI_Barrier(comm);CHKERRQ(ierr);
      ierr = PetscInfo(in,"Using VecScatter ready receiver mode\n");CHKERRQ(ierr);
    } else {
      out_to->use_readyreceiver    = PETSC_FALSE;
      out_from->use_readyreceiver  = PETSC_FALSE;
      flg                          = PETSC_FALSE;
      ierr                         = PetscOptionsGetTruth(PETSC_NULL,"-vecscatter_ssend",&flg,PETSC_NULL);CHKERRQ(ierr);
      if (flg) {
        ierr = PetscInfo(in,"Using VecScatter Ssend mode\n");CHKERRQ(ierr);
      }
      for (i=0; i<out_to->n; i++) {
        if (!flg) {
          ierr = MPI_Send_init(Ssvalues+bs*sstarts[i],bs*sstarts[i+1]-bs*sstarts[i],MPIU_SCALAR,sprocs[i],tag,comm,swaits+i);CHKERRQ(ierr);
        } else {
          ierr = MPI_Ssend_init(Ssvalues+bs*sstarts[i],bs*sstarts[i+1]-bs*sstarts[i],MPIU_SCALAR,sprocs[i],tag,comm,swaits+i);CHKERRQ(ierr);
        }
      } 
    }
    /* Register receives for scatter reverse */
    for (i=0; i<out_to->n; i++) {
      ierr = MPI_Recv_init(Ssvalues+bs*sstarts[i],bs*sstarts[i+1]-bs*sstarts[i],MPIU_SCALAR,sprocs[i],tag,comm,rev_rwaits+i);CHKERRQ(ierr);
    }
  } 

  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecScatterCopy_PtoP_AllToAll"
PetscErrorCode VecScatterCopy_PtoP_AllToAll(VecScatter in,VecScatter out)
{
  VecScatter_MPI_General *in_to   = (VecScatter_MPI_General*)in->todata;
  VecScatter_MPI_General *in_from = (VecScatter_MPI_General*)in->fromdata,*out_to,*out_from;
  PetscErrorCode         ierr;
  PetscInt               ny,bs = in_from->bs;
  PetscMPIInt            size;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(((PetscObject)in)->comm,&size);CHKERRQ(ierr);
  out->begin     = in->begin;
  out->end       = in->end;
  out->copy      = in->copy;
  out->destroy   = in->destroy;
  out->view      = in->view;

  /* allocate entire send scatter context */
  ierr = PetscNewLog(out,VecScatter_MPI_General,&out_to);CHKERRQ(ierr);
  ierr = PetscNewLog(out,VecScatter_MPI_General,&out_from);CHKERRQ(ierr);

  ny                = in_to->starts[in_to->n];
  out_to->n         = in_to->n; 
  out_to->type      = in_to->type;
  out_to->sendfirst = in_to->sendfirst;

  ierr = PetscMalloc(out_to->n*sizeof(MPI_Request),&out_to->requests);CHKERRQ(ierr);
  ierr = PetscMalloc4(bs*ny,PetscScalar,&out_to->values,ny,PetscInt,&out_to->indices,out_to->n+1,PetscInt,&out_to->starts,out_to->n,PetscMPIInt,&out_to->procs);CHKERRQ(ierr);
  ierr = PetscMalloc2(PetscMax(in_to->n,in_from->n),MPI_Status,&out_to->sstatus,PetscMax(in_to->n,in_from->n),MPI_Status,&out_to->rstatus);CHKERRQ(ierr);
  ierr = PetscMemcpy(out_to->indices,in_to->indices,ny*sizeof(PetscInt));CHKERRQ(ierr);
  ierr = PetscMemcpy(out_to->starts,in_to->starts,(out_to->n+1)*sizeof(PetscInt));CHKERRQ(ierr);
  ierr = PetscMemcpy(out_to->procs,in_to->procs,(out_to->n)*sizeof(PetscMPIInt));CHKERRQ(ierr);
     
  out->todata       = (void*)out_to;
  out_to->local.n   = in_to->local.n;
  out_to->local.nonmatching_computed = PETSC_FALSE;
  out_to->local.n_nonmatching        = 0;
  out_to->local.slots_nonmatching    = 0;
  if (in_to->local.n) {
    ierr = PetscMalloc(in_to->local.n*sizeof(PetscInt),&out_to->local.vslots);CHKERRQ(ierr);
    ierr = PetscMalloc(in_from->local.n*sizeof(PetscInt),&out_from->local.vslots);CHKERRQ(ierr);
    ierr = PetscMemcpy(out_to->local.vslots,in_to->local.vslots,in_to->local.n*sizeof(PetscInt));CHKERRQ(ierr);
    ierr = PetscMemcpy(out_from->local.vslots,in_from->local.vslots,in_from->local.n*sizeof(PetscInt));CHKERRQ(ierr);
  } else {
    out_to->local.vslots   = 0;
    out_from->local.vslots = 0;
  }

  /* allocate entire receive context */
  out_from->type      = in_from->type;
  ny                  = in_from->starts[in_from->n];
  out_from->n         = in_from->n; 
  out_from->sendfirst = in_from->sendfirst;

  ierr = PetscMalloc(out_from->n*sizeof(MPI_Request),&out_from->requests);CHKERRQ(ierr);
  ierr = PetscMalloc4(ny*bs,PetscScalar,&out_from->values,ny,PetscInt,&out_from->indices,out_from->n+1,PetscInt,&out_from->starts,out_from->n,PetscMPIInt,&out_from->procs);CHKERRQ(ierr);
  ierr = PetscMemcpy(out_from->indices,in_from->indices,ny*sizeof(PetscInt));CHKERRQ(ierr);
  ierr = PetscMemcpy(out_from->starts,in_from->starts,(out_from->n+1)*sizeof(PetscInt));CHKERRQ(ierr);
  ierr = PetscMemcpy(out_from->procs,in_from->procs,(out_from->n)*sizeof(PetscMPIInt));CHKERRQ(ierr);
  out->fromdata       = (void*)out_from;
  out_from->local.n   = in_from->local.n;
  out_from->local.nonmatching_computed = PETSC_FALSE;
  out_from->local.n_nonmatching        = 0;
  out_from->local.slots_nonmatching    = 0;

  out_to->use_alltoallv = out_from->use_alltoallv = PETSC_TRUE;

  ierr = PetscMalloc2(size,PetscMPIInt,&out_to->counts,size,PetscMPIInt,&out_to->displs);CHKERRQ(ierr);
  ierr = PetscMemcpy(out_to->counts,in_to->counts,size*sizeof(PetscMPIInt));CHKERRQ(ierr);
  ierr = PetscMemcpy(out_to->displs,in_to->displs,size*sizeof(PetscMPIInt));CHKERRQ(ierr);

  ierr = PetscMalloc2(size,PetscMPIInt,&out_from->counts,size,PetscMPIInt,&out_from->displs);CHKERRQ(ierr);
  ierr = PetscMemcpy(out_from->counts,in_from->counts,size*sizeof(PetscMPIInt));CHKERRQ(ierr);
  ierr = PetscMemcpy(out_from->displs,in_from->displs,size*sizeof(PetscMPIInt));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
/* --------------------------------------------------------------------------------------------------
    Packs and unpacks the message data into send or from receive buffers. 

    These could be generated automatically. 

    Fortran kernels etc. could be used.
*/
PETSC_STATIC_INLINE void Pack_1(PetscInt n,const PetscInt *indicesx,const PetscScalar *x,PetscScalar *y)
{
  PetscInt i;
  for (i=0; i<n; i++) {
    y[i]  = x[indicesx[i]];
  }
}
PETSC_STATIC_INLINE void UnPack_1(PetscInt n,const PetscScalar *x,const PetscInt *indicesy,PetscScalar *y,InsertMode addv)
{
  PetscInt i;
  switch (addv) {
  case INSERT_VALUES: 
    for (i=0; i<n; i++) {
      y[indicesy[i]] = x[i];
    }
    break;
  case ADD_VALUES: 
    for (i=0; i<n; i++) {
      y[indicesy[i]] += x[i];
    }
    break;
#if !defined(PETSC_USE_COMPLEX)
  case MAX_VALUES: 
    for (i=0; i<n; i++) {
      y[indicesy[i]] = PetscMax(y[indicesy[i]],x[i]);
    }
#else
  case MAX_VALUES: 
#endif
  case NOT_SET_VALUES:
    break;
  }
}

PETSC_STATIC_INLINE void Scatter_1(PetscInt n,const PetscInt *indicesx,const PetscScalar *x,const PetscInt *indicesy,PetscScalar *y,InsertMode addv)
{
  PetscInt i;
  switch (addv) {
  case INSERT_VALUES: 
    for (i=0; i<n; i++) {
      y[indicesy[i]] = x[indicesx[i]];
    }
    break;
  case ADD_VALUES: 
    for (i=0; i<n; i++) {
      y[indicesy[i]] += x[indicesx[i]];
    }
    break;
#if !defined(PETSC_USE_COMPLEX)
  case MAX_VALUES: 
    for (i=0; i<n; i++) {
      y[indicesy[i]] = PetscMax(y[indicesy[i]],x[indicesx[i]]);
    }
#else
  case MAX_VALUES: 
#endif
  case NOT_SET_VALUES:
    break;
  }
}

  /* ----------------------------------------------------------------------------------------------- */
PETSC_STATIC_INLINE void Pack_2(PetscInt n,const PetscInt *indicesx,const PetscScalar *x,PetscScalar *y)
{
  PetscInt i,idx;

  for (i=0; i<n; i++) {
    idx   = *indicesx++;
    y[0]  = x[idx];
    y[1]  = x[idx+1];
    y    += 2;
  }
}
PETSC_STATIC_INLINE void UnPack_2(PetscInt n,const PetscScalar *x,const PetscInt *indicesy,PetscScalar *y,InsertMode addv)
{
  PetscInt i,idy;

  switch (addv) {
  case INSERT_VALUES: 
    for (i=0; i<n; i++) {
      idy       = *indicesy++;
      y[idy]    = x[0];
      y[idy+1]  = x[1];
      x    += 2;
    }
    break;
  case ADD_VALUES:
    for (i=0; i<n; i++) {
      idy       = *indicesy++;
      y[idy]    += x[0];
      y[idy+1]  += x[1];
      x    += 2;
    }
    break;
#if !defined(PETSC_USE_COMPLEX)
  case MAX_VALUES:
    for (i=0; i<n; i++) {
      idy       = *indicesy++;
      y[idy]    = PetscMax(y[idy],x[0]);
      y[idy+1]  = PetscMax(y[idy+1],x[1]);
      x    += 2;
    }
#else
  case MAX_VALUES: 
#endif
  case NOT_SET_VALUES:
    break;
  }
}

PETSC_STATIC_INLINE void Scatter_2(PetscInt n,const PetscInt *indicesx,const PetscScalar *x,const PetscInt *indicesy,PetscScalar *y,InsertMode addv)
{
  PetscInt i,idx,idy;

  switch (addv) {
  case INSERT_VALUES: 
    for (i=0; i<n; i++) {
      idx       = *indicesx++;
      idy       = *indicesy++;
      y[idy]    = x[idx];
      y[idy+1]  = x[idx+1];
    }
    break;
  case ADD_VALUES:
    for (i=0; i<n; i++) {
      idx       = *indicesx++;
      idy       = *indicesy++;
      y[idy]    += x[idx];
      y[idy+1]  += x[idx+1];
    }
    break;
#if !defined(PETSC_USE_COMPLEX)
  case MAX_VALUES:
    for (i=0; i<n; i++) {
      idx       = *indicesx++;
      idy       = *indicesy++;
      y[idy]    = PetscMax(y[idy],x[idx]);
      y[idy+1]  = PetscMax(y[idy+1],x[idx+1]);
    }
#else
  case MAX_VALUES: 
#endif
  case NOT_SET_VALUES:
    break;
  }
}
  /* ----------------------------------------------------------------------------------------------- */
PETSC_STATIC_INLINE void Pack_3(PetscInt n,const PetscInt *indicesx,const PetscScalar *x,PetscScalar *y)
{
  PetscInt i,idx;

  for (i=0; i<n; i++) {
    idx   = *indicesx++;
    y[0]  = x[idx];
    y[1]  = x[idx+1];
    y[2]  = x[idx+2];
    y    += 3;
  }
}
PETSC_STATIC_INLINE void UnPack_3(PetscInt n,const PetscScalar *x,const PetscInt *indicesy,PetscScalar *y,InsertMode addv)
{
  PetscInt i,idy;

  switch (addv) {
  case INSERT_VALUES: 
    for (i=0; i<n; i++) {
      idy       = *indicesy++;
      y[idy]    = x[0];
      y[idy+1]  = x[1];
      y[idy+2]  = x[2];
      x    += 3;
    }
    break;
  case ADD_VALUES:
    for (i=0; i<n; i++) {
      idy       = *indicesy++;
      y[idy]    += x[0];
      y[idy+1]  += x[1];
      y[idy+2]  += x[2];
      x    += 3;
    }
    break;
#if !defined(PETSC_USE_COMPLEX)
  case MAX_VALUES:
    for (i=0; i<n; i++) {
      idy       = *indicesy++;
      y[idy]    = PetscMax(y[idy],x[0]);
      y[idy+1]  = PetscMax(y[idy+1],x[1]);
      y[idy+2]  = PetscMax(y[idy+2],x[2]);
      x    += 3;
    }
#else
  case MAX_VALUES: 
#endif
  case NOT_SET_VALUES:
    break;
  }
}

PETSC_STATIC_INLINE void Scatter_3(PetscInt n,const PetscInt *indicesx,const PetscScalar *x,const PetscInt *indicesy,PetscScalar *y,InsertMode addv)
{
  PetscInt i,idx,idy;

  switch (addv) {
  case INSERT_VALUES: 
    for (i=0; i<n; i++) {
      idx       = *indicesx++;
      idy       = *indicesy++;
      y[idy]    = x[idx];
      y[idy+1]  = x[idx+1];
      y[idy+2]  = x[idx+2];
    }
    break;
  case ADD_VALUES:
    for (i=0; i<n; i++) {
      idx       = *indicesx++;
      idy       = *indicesy++;
      y[idy]    += x[idx];
      y[idy+1]  += x[idx+1];
      y[idy+2]  += x[idx+2];
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
    }
#else
  case MAX_VALUES: 
#endif
  case NOT_SET_VALUES:
    break;
  }
}
  /* ----------------------------------------------------------------------------------------------- */
PETSC_STATIC_INLINE void Pack_4(PetscInt n,const PetscInt *indicesx,const PetscScalar *x,PetscScalar *y)
{
  PetscInt i,idx;

  for (i=0; i<n; i++) {
    idx   = *indicesx++;
    y[0]  = x[idx];
    y[1]  = x[idx+1];
    y[2]  = x[idx+2];
    y[3]  = x[idx+3];
    y    += 4;
  }
}
PETSC_STATIC_INLINE void UnPack_4(PetscInt n,const PetscScalar *x,const PetscInt *indicesy,PetscScalar *y,InsertMode addv)
{
  PetscInt i,idy;

  switch (addv) {
  case INSERT_VALUES: 
    for (i=0; i<n; i++) {
      idy       = *indicesy++;
      y[idy]    = x[0];
      y[idy+1]  = x[1];
      y[idy+2]  = x[2];
      y[idy+3]  = x[3];
      x    += 4;
    }
    break;
  case ADD_VALUES:
    for (i=0; i<n; i++) {
      idy       = *indicesy++;
      y[idy]    += x[0];
      y[idy+1]  += x[1];
      y[idy+2]  += x[2];
      y[idy+3]  += x[3];
      x    += 4;
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
      x    += 4;
    }
#else
  case MAX_VALUES: 
#endif
  case NOT_SET_VALUES:
    break;
  }
}

PETSC_STATIC_INLINE void Scatter_4(PetscInt n,const PetscInt *indicesx,const PetscScalar *x,const PetscInt *indicesy,PetscScalar *y,InsertMode addv)
{
  PetscInt i,idx,idy;

  switch (addv) {
  case INSERT_VALUES: 
    for (i=0; i<n; i++) {
      idx       = *indicesx++;
      idy       = *indicesy++;
      y[idy]    = x[idx];
      y[idy+1]  = x[idx+1];
      y[idy+2]  = x[idx+2];
      y[idy+3]  = x[idx+3];
    }
    break;
  case ADD_VALUES:
    for (i=0; i<n; i++) {
      idx       = *indicesx++;
      idy       = *indicesy++;
      y[idy]    += x[idx];
      y[idy+1]  += x[idx+1];
      y[idy+2]  += x[idx+2];
      y[idy+3]  += x[idx+3];
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
    }
#else
  case MAX_VALUES: 
#endif
  case NOT_SET_VALUES:
    break;
  }
}
  /* ----------------------------------------------------------------------------------------------- */
PETSC_STATIC_INLINE void Pack_5(PetscInt n,const PetscInt *indicesx,const PetscScalar *x,PetscScalar *y)
{
  PetscInt i,idx;

  for (i=0; i<n; i++) {
    idx   = *indicesx++;
    y[0]  = x[idx];
    y[1]  = x[idx+1];
    y[2]  = x[idx+2];
    y[3]  = x[idx+3];
    y[4]  = x[idx+4];
    y    += 5;
  }
}
PETSC_STATIC_INLINE void UnPack_5(PetscInt n,const PetscScalar *x,const PetscInt *indicesy,PetscScalar *y,InsertMode addv)
{
  PetscInt i,idy;

  switch (addv) {
  case INSERT_VALUES: 
    for (i=0; i<n; i++) {
      idy       = *indicesy++;
      y[idy]    = x[0];
      y[idy+1]  = x[1];
      y[idy+2]  = x[2];
      y[idy+3]  = x[3];
      y[idy+4]  = x[4];
      x    += 5;
    }
    break;
  case ADD_VALUES:
    for (i=0; i<n; i++) {
      idy       = *indicesy++;
      y[idy]    += x[0];
      y[idy+1]  += x[1];
      y[idy+2]  += x[2];
      y[idy+3]  += x[3];
      y[idy+4]  += x[4];
      x    += 5;
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
      x    += 5;
    }
#else
  case MAX_VALUES: 
#endif
  case NOT_SET_VALUES:
    break;
  }
}

PETSC_STATIC_INLINE void Scatter_5(PetscInt n,const PetscInt *indicesx,const PetscScalar *x,const PetscInt *indicesy,PetscScalar *y,InsertMode addv)
{
  PetscInt i,idx,idy;

  switch (addv) {
  case INSERT_VALUES: 
    for (i=0; i<n; i++) {
      idx       = *indicesx++;
      idy       = *indicesy++;
      y[idy]    = x[idx];
      y[idy+1]  = x[idx+1];
      y[idy+2]  = x[idx+2];
      y[idy+3]  = x[idx+3];
      y[idy+4]  = x[idx+4];
    }
    break;
  case ADD_VALUES:
    for (i=0; i<n; i++) {
      idx       = *indicesx++;
      idy       = *indicesy++;
      y[idy]    += x[idx];
      y[idy+1]  += x[idx+1];
      y[idy+2]  += x[idx+2];
      y[idy+3]  += x[idx+3];
      y[idy+4]  += x[idx+4];
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
    }
#else
  case MAX_VALUES: 
#endif
  case NOT_SET_VALUES:
    break;
  }
}
  /* ----------------------------------------------------------------------------------------------- */
PETSC_STATIC_INLINE void Pack_6(PetscInt n,const PetscInt *indicesx,const PetscScalar *x,PetscScalar *y)
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
    y    += 6;
  }
}
PETSC_STATIC_INLINE void UnPack_6(PetscInt n,const PetscScalar *x,const PetscInt *indicesy,PetscScalar *y,InsertMode addv)
{
  PetscInt i,idy;

  switch (addv) {
  case INSERT_VALUES: 
    for (i=0; i<n; i++) {
      idy       = *indicesy++;
      y[idy]    = x[0];
      y[idy+1]  = x[1];
      y[idy+2]  = x[2];
      y[idy+3]  = x[3];
      y[idy+4]  = x[4];
      y[idy+5]  = x[5];
      x    += 6;
    }
    break;
  case ADD_VALUES:
    for (i=0; i<n; i++) {
      idy       = *indicesy++;
      y[idy]    += x[0];
      y[idy+1]  += x[1];
      y[idy+2]  += x[2];
      y[idy+3]  += x[3];
      y[idy+4]  += x[4];
      y[idy+5]  += x[5];
      x    += 6;
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
      x    += 6;
    }
#else
  case MAX_VALUES: 
#endif
  case NOT_SET_VALUES:
    break;
  }
}

PETSC_STATIC_INLINE void Scatter_6(PetscInt n,const PetscInt *indicesx,const PetscScalar *x,const PetscInt *indicesy,PetscScalar *y,InsertMode addv)
{
  PetscInt i,idx,idy;

  switch (addv) {
  case INSERT_VALUES: 
    for (i=0; i<n; i++) {
      idx       = *indicesx++;
      idy       = *indicesy++;
      y[idy]    = x[idx];
      y[idy+1]  = x[idx+1];
      y[idy+2]  = x[idx+2];
      y[idy+3]  = x[idx+3];
      y[idy+4]  = x[idx+4];
      y[idy+5]  = x[idx+5];
    }
    break;
  case ADD_VALUES:
    for (i=0; i<n; i++) {
      idx       = *indicesx++;
      idy       = *indicesy++;
      y[idy]    += x[idx];
      y[idy+1]  += x[idx+1];
      y[idy+2]  += x[idx+2];
      y[idy+3]  += x[idx+3];
      y[idy+4]  += x[idx+4];
      y[idy+5]  += x[idx+5];
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
    }
#else
  case MAX_VALUES: 
#endif
  case NOT_SET_VALUES:
    break;
  }
}
  /* ----------------------------------------------------------------------------------------------- */
PETSC_STATIC_INLINE void Pack_7(PetscInt n,const PetscInt *indicesx,const PetscScalar *x,PetscScalar *y)
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
    y    += 7;
  }
}
PETSC_STATIC_INLINE void UnPack_7(PetscInt n,const PetscScalar *x,const PetscInt *indicesy,PetscScalar *y,InsertMode addv)
{
  PetscInt i,idy;

  switch (addv) {
  case INSERT_VALUES: 
    for (i=0; i<n; i++) {
      idy       = *indicesy++;
      y[idy]    = x[0];
      y[idy+1]  = x[1];
      y[idy+2]  = x[2];
      y[idy+3]  = x[3];
      y[idy+4]  = x[4];
      y[idy+5]  = x[5];
      y[idy+6]  = x[6];
      x    += 7;
    }
    break;
  case ADD_VALUES:
    for (i=0; i<n; i++) {
      idy       = *indicesy++;
      y[idy]    += x[0];
      y[idy+1]  += x[1];
      y[idy+2]  += x[2];
      y[idy+3]  += x[3];
      y[idy+4]  += x[4];
      y[idy+5]  += x[5];
      y[idy+6]  += x[6];
      x    += 7;
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
      x    += 7;
    }
#else
  case MAX_VALUES: 
#endif
  case NOT_SET_VALUES:
    break;
  }
}

PETSC_STATIC_INLINE void Scatter_7(PetscInt n,const PetscInt *indicesx,const PetscScalar *x,const PetscInt *indicesy,PetscScalar *y,InsertMode addv)
{
  PetscInt i,idx,idy;

  switch (addv) {
  case INSERT_VALUES: 
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
    }
    break;
  case ADD_VALUES:
    for (i=0; i<n; i++) {
      idx       = *indicesx++;
      idy       = *indicesy++;
      y[idy]    += x[idx];
      y[idy+1]  += x[idx+1];
      y[idy+2]  += x[idx+2];
      y[idy+3]  += x[idx+3];
      y[idy+4]  += x[idx+4];
      y[idy+5]  += x[idx+5];
      y[idy+6]  += x[idx+6];
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
    }
#else
  case MAX_VALUES: 
#endif
  case NOT_SET_VALUES:
    break;
  }
}
  /* ----------------------------------------------------------------------------------------------- */
PETSC_STATIC_INLINE void Pack_8(PetscInt n,const PetscInt *indicesx,const PetscScalar *x,PetscScalar *y)
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
    y    += 8;
  }
}
PETSC_STATIC_INLINE void UnPack_8(PetscInt n,const PetscScalar *x,const PetscInt *indicesy,PetscScalar *y,InsertMode addv)
{
  PetscInt i,idy;

  switch (addv) {
  case INSERT_VALUES: 
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
      x    += 8;
    }
    break;
  case ADD_VALUES:
    for (i=0; i<n; i++) {
      idy       = *indicesy++;
      y[idy]    += x[0];
      y[idy+1]  += x[1];
      y[idy+2]  += x[2];
      y[idy+3]  += x[3];
      y[idy+4]  += x[4];
      y[idy+5]  += x[5];
      y[idy+6]  += x[6];
      y[idy+7]  += x[7];
      x    += 8;
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
      x    += 8;
    }
#else
  case MAX_VALUES: 
#endif
  case NOT_SET_VALUES:
    break;
  }
}

PETSC_STATIC_INLINE void Scatter_8(PetscInt n,const PetscInt *indicesx,const PetscScalar *x,const PetscInt *indicesy,PetscScalar *y,InsertMode addv)
{
  PetscInt i,idx,idy;

  switch (addv) {
  case INSERT_VALUES: 
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
    }
    break;
  case ADD_VALUES:
    for (i=0; i<n; i++) {
      idx       = *indicesx++;
      idy       = *indicesy++;
      y[idy]    += x[idx];
      y[idy+1]  += x[idx+1];
      y[idy+2]  += x[idx+2];
      y[idy+3]  += x[idx+3];
      y[idy+4]  += x[idx+4];
      y[idy+5]  += x[idx+5];
      y[idy+6]  += x[idx+6];
      y[idy+7]  += x[idx+7];
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
    }
#else
  case MAX_VALUES: 
#endif
  case NOT_SET_VALUES:
    break;
  }
}

  /* ----------------------------------------------------------------------------------------------- */
PETSC_STATIC_INLINE void Pack_12(PetscInt n,const PetscInt *indicesx,const PetscScalar *x,PetscScalar *y)
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
PETSC_STATIC_INLINE void UnPack_12(PetscInt n,const PetscScalar *x,const PetscInt *indicesy,PetscScalar *y,InsertMode addv)
{
  PetscInt i,idy;

  switch (addv) {
  case INSERT_VALUES: 
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
      x    += 12;
    }
    break;
  case ADD_VALUES:
    for (i=0; i<n; i++) {
      idy       = *indicesy++;
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
      x    += 12;
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
      x    += 12;
    }
#else
  case MAX_VALUES: 
#endif
  case NOT_SET_VALUES:
    break;
  }
}

PETSC_STATIC_INLINE void Scatter_12(PetscInt n,const PetscInt *indicesx,const PetscScalar *x,const PetscInt *indicesy,PetscScalar *y,InsertMode addv)
{
  PetscInt i,idx,idy;

  switch (addv) {
  case INSERT_VALUES: 
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
    for (i=0; i<n; i++) {
      idx       = *indicesx++;
      idy       = *indicesy++;
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
  }
}

/* Create the VecScatterBegin/End_P for our chosen block sizes */
#define BS 1
#include "../src/vec/vec/utils/vpscat.h"
#define BS 2
#include "../src/vec/vec/utils/vpscat.h"
#define BS 3
#include "../src/vec/vec/utils/vpscat.h"
#define BS 4
#include "../src/vec/vec/utils/vpscat.h"
#define BS 5
#include "../src/vec/vec/utils/vpscat.h"
#define BS 6
#include "../src/vec/vec/utils/vpscat.h"
#define BS 7
#include "../src/vec/vec/utils/vpscat.h"
#define BS 8
#include "../src/vec/vec/utils/vpscat.h"
#define BS 12
#include "../src/vec/vec/utils/vpscat.h"

/* ==========================================================================================*/

/*              create parallel to sequential scatter context                           */

PetscErrorCode VecScatterCreateCommon_PtoS(VecScatter_MPI_General *,VecScatter_MPI_General *,VecScatter);

#undef __FUNCT__  
#define __FUNCT__ "VecScatterCreateLocal"
/*@
     VecScatterCreateLocal - Creates a VecScatter from a list of messages it must send and receive.

     Collective on VecScatter

   Input Parameters:
+     VecScatter - obtained with VecScatterCreateEmpty()
.     nsends -
.     sendSizes -
.     sendProcs -
.     sendIdx - indices where the sent entries are obtained from (in local, on process numbering), this is one long array of size \sum_{i=0,i<nsends} sendSizes[i]
.     nrecvs - number of receives to expect
.     recvSizes - 
.     recvProcs - processes who are sending to me
.     recvIdx - indices of where received entries are to be put, (in local, on process numbering), this is one long array of size \sum_{i=0,i<nrecvs} recvSizes[i]
-     bs - size of block

     Notes:  sendSizes[] and recvSizes[] cannot have any 0 entries. If you want to support having 0 entries you need
      to change the code below to "compress out" the sendProcs[] and recvProcs[] entries that have 0 entries.

       Probably does not handle sends to self properly. It should remove those from the counts that are used
      in allocating space inside of the from struct

  Level: intermediate

@*/
PetscErrorCode VecScatterCreateLocal(VecScatter ctx,PetscInt nsends,const PetscInt sendSizes[],const PetscInt sendProcs[],const PetscInt sendIdx[],PetscInt nrecvs,const PetscInt recvSizes[],const PetscInt recvProcs[],const PetscInt recvIdx[],PetscInt bs)
{
  VecScatter_MPI_General *from, *to;
  PetscInt               sendSize, recvSize;
  PetscInt               n, i;
  PetscErrorCode         ierr;

  /* allocate entire send scatter context */
  ierr = PetscNewLog(ctx,VecScatter_MPI_General,&to);CHKERRQ(ierr);
  to->n = nsends; 
  for(n = 0, sendSize = 0; n < to->n; n++) {sendSize += sendSizes[n];}
  ierr = PetscMalloc(to->n*sizeof(MPI_Request),&to->requests);CHKERRQ(ierr);
  ierr = PetscMalloc4(bs*sendSize,PetscScalar,&to->values,sendSize,PetscInt,&to->indices,to->n+1,PetscInt,&to->starts,to->n,PetscMPIInt,&to->procs);CHKERRQ(ierr);
  ierr = PetscMalloc2(PetscMax(to->n,nrecvs),MPI_Status,&to->sstatus,PetscMax(to->n,nrecvs),MPI_Status,&to->rstatus);CHKERRQ(ierr);
  to->starts[0] = 0;
  for(n = 0; n < to->n; n++) {
    if (sendSizes[n] <=0 ) SETERRQ2(PETSC_ERR_ARG_OUTOFRANGE,"sendSizes[n=%D] = %D cannot be less than 1",n,sendSizes[n]);
    to->starts[n+1] = to->starts[n] + sendSizes[n];
    to->procs[n] = sendProcs[n];
    for(i = to->starts[n]; i < to->starts[n]+sendSizes[n]; i++) {
      to->indices[i] = sendIdx[i];
    }
  }
  ctx->todata = (void *) to;

  /* allocate entire receive scatter context */
  ierr = PetscNewLog(ctx,VecScatter_MPI_General,&from);CHKERRQ(ierr);
  from->n = nrecvs;
  for(n = 0, recvSize = 0; n < from->n; n++) {recvSize += recvSizes[n];}
  ierr = PetscMalloc(from->n*sizeof(MPI_Request),&from->requests);CHKERRQ(ierr);
  ierr = PetscMalloc4(bs*recvSize,PetscScalar,&from->values,recvSize,PetscInt,&from->indices,from->n+1,PetscInt,&from->starts,from->n,PetscMPIInt,&from->procs);CHKERRQ(ierr);
  from->starts[0] = 0;
  for(n = 0; n < from->n; n++) {
    if (recvSizes[n] <=0 ) SETERRQ2(PETSC_ERR_ARG_OUTOFRANGE,"recvSizes[n=%D] = %D cannot be less than 1",n,recvSizes[n]);
    from->starts[n+1] = from->starts[n] + recvSizes[n];
    from->procs[n] = recvProcs[n];
    for(i = from->starts[n]; i < from->starts[n]+recvSizes[n]; i++) {
      from->indices[i] = recvIdx[i];
    }
  }
  ctx->fromdata = (void *)from;

  /* No local scatter optimization */
  from->local.n      = 0;
  from->local.vslots = 0;
  to->local.n        = 0; 
  to->local.vslots   = 0;
  from->local.nonmatching_computed = PETSC_FALSE;
  from->local.n_nonmatching        = 0;
  from->local.slots_nonmatching    = 0;
  to->local.nonmatching_computed   = PETSC_FALSE;
  to->local.n_nonmatching          = 0;
  to->local.slots_nonmatching      = 0;

  from->type = VEC_SCATTER_MPI_GENERAL;
  to->type   = VEC_SCATTER_MPI_GENERAL; 
  from->bs = bs;
  to->bs   = bs;
  ierr = VecScatterCreateCommon_PtoS(from, to, ctx); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
   bs indicates how many elements there are in each block. Normally this would be 1.

   contains check that PetscMPIInt can handle the sizes needed 
*/
#undef __FUNCT__  
#define __FUNCT__ "VecScatterCreate_PtoS"
PetscErrorCode VecScatterCreate_PtoS(PetscInt nx,const PetscInt *inidx,PetscInt ny,const PetscInt *inidy,Vec xin,Vec yin,PetscInt bs,VecScatter ctx)
{
  VecScatter_MPI_General *from,*to;
  PetscMPIInt            size,rank,imdex,tag,n;
  PetscInt               *source = PETSC_NULL,*owners = PETSC_NULL;
  PetscInt               *lowner = PETSC_NULL,*start = PETSC_NULL,lengthy,lengthx;
  PetscMPIInt            *nprocs = PETSC_NULL,nrecvs;
  PetscInt               i,j,idx,nsends;
  PetscInt               *owner = PETSC_NULL,*starts = PETSC_NULL,count,slen;
  PetscInt               *rvalues,*svalues,base,*values,nprocslocal,recvtotal,*rsvalues;
  PetscMPIInt            *onodes1,*olengths1;
  MPI_Comm               comm;
  MPI_Request            *send_waits = PETSC_NULL,*recv_waits = PETSC_NULL;
  MPI_Status             recv_status,*send_status;
  PetscErrorCode         ierr;

  PetscFunctionBegin;
  ierr = PetscObjectGetNewTag((PetscObject)ctx,&tag);CHKERRQ(ierr);
  ierr = PetscObjectGetComm((PetscObject)xin,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  owners = xin->map->range;
  ierr = VecGetSize(yin,&lengthy);CHKERRQ(ierr);
  ierr = VecGetSize(xin,&lengthx);CHKERRQ(ierr);

  /*  first count number of contributors to each processor */
  ierr = PetscMalloc2(size,PetscMPIInt,&nprocs,nx,PetscInt,&owner);CHKERRQ(ierr);
  ierr = PetscMemzero(nprocs,size*sizeof(PetscMPIInt));CHKERRQ(ierr);
  j      = 0;
  nsends = 0;
  for (i=0; i<nx; i++) {
    idx = inidx[i];
    if (idx < owners[j]) j = 0;
    for (; j<size; j++) {
      if (idx < owners[j+1]) {
        if (!nprocs[j]++) nsends++;
        owner[i] = j; 
        break;
      }
    }
  }
  nprocslocal  = nprocs[rank]; 
  nprocs[rank] = 0;
  if (nprocslocal) nsends--;
  /* inform other processors of number of messages and max length*/
  ierr = PetscGatherNumberOfMessages(comm,PETSC_NULL,nprocs,&nrecvs);CHKERRQ(ierr);
  ierr = PetscGatherMessageLengths(comm,nsends,nrecvs,nprocs,&onodes1,&olengths1);CHKERRQ(ierr);
  ierr = PetscSortMPIIntWithArray(nrecvs,onodes1,olengths1);CHKERRQ(ierr);
  recvtotal = 0; for (i=0; i<nrecvs; i++) recvtotal += olengths1[i];

  /* post receives:   */
  ierr = PetscMalloc3(recvtotal,PetscInt,&rvalues,nrecvs,PetscInt,&source,nrecvs,MPI_Request,&recv_waits);CHKERRQ(ierr);
  count  = 0;
  for (i=0; i<nrecvs; i++) {
    ierr  = MPI_Irecv((rvalues+count),olengths1[i],MPIU_INT,onodes1[i],tag,comm,recv_waits+i);CHKERRQ(ierr);
    count += olengths1[i];
  }

  /* do sends:
     1) starts[i] gives the starting index in svalues for stuff going to 
     the ith processor
  */
  ierr = PetscMalloc3(nx,PetscInt,&svalues,nsends,MPI_Request,&send_waits,size+1,PetscInt,&starts);CHKERRQ(ierr);
  starts[0]  = 0; 
  for (i=1; i<size; i++) { starts[i] = starts[i-1] + nprocs[i-1];} 
  for (i=0; i<nx; i++) {
    if (owner[i] != rank) {
      svalues[starts[owner[i]]++] = inidx[i];
    }
  }

  starts[0] = 0;
  for (i=1; i<size+1; i++) { starts[i] = starts[i-1] + nprocs[i-1];} 
  count = 0;
  for (i=0; i<size; i++) {
    if (nprocs[i]) {
      ierr = MPI_Isend(svalues+starts[i],nprocs[i],MPIU_INT,i,tag,comm,send_waits+count++);CHKERRQ(ierr);
    }
  }

  /*  wait on receives */
  count  = nrecvs; 
  slen   = 0;
  while (count) {
    ierr = MPI_Waitany(nrecvs,recv_waits,&imdex,&recv_status);CHKERRQ(ierr);
    /* unpack receives into our local space */
    ierr = MPI_Get_count(&recv_status,MPIU_INT,&n);CHKERRQ(ierr);
    slen += n;
    count--;
  }

  if (slen != recvtotal) SETERRQ2(PETSC_ERR_PLIB,"Total message lengths %D not expected %D",slen,recvtotal);
 
  /* allocate entire send scatter context */
  ierr  = PetscNewLog(ctx,VecScatter_MPI_General,&to);CHKERRQ(ierr);
  to->n = nrecvs; 
  ierr  = PetscMalloc(nrecvs*sizeof(MPI_Request),&to->requests);CHKERRQ(ierr);
  ierr  = PetscMalloc4(bs*slen,PetscScalar,&to->values,slen,PetscInt,&to->indices,nrecvs+1,PetscInt,&to->starts,nrecvs,PetscMPIInt,&to->procs);CHKERRQ(ierr);
  ierr  = PetscMalloc2(PetscMax(to->n,nsends),MPI_Status,&to->sstatus,PetscMax(to->n,nsends),MPI_Status,&to->rstatus);CHKERRQ(ierr);
  ctx->todata   = (void*)to;
  to->starts[0] = 0;

  if (nrecvs) {

    /* move the data into the send scatter */
    base     = owners[rank];
    rsvalues = rvalues;
    for (i=0; i<nrecvs; i++) {
      to->starts[i+1] = to->starts[i] + olengths1[i];
      to->procs[i]    = onodes1[i];
      values = rsvalues;
      rsvalues += olengths1[i];
      for (j=0; j<olengths1[i]; j++) {
        to->indices[to->starts[i] + j] = values[j] - base;
      }
    }
  }
  ierr = PetscFree(olengths1);CHKERRQ(ierr);
  ierr = PetscFree(onodes1);CHKERRQ(ierr);
  ierr = PetscFree3(rvalues,source,recv_waits);CHKERRQ(ierr);

  /* allocate entire receive scatter context */
  ierr = PetscNewLog(ctx,VecScatter_MPI_General,&from);CHKERRQ(ierr);
  from->n = nsends;

  ierr = PetscMalloc(nsends*sizeof(MPI_Request),&from->requests);CHKERRQ(ierr);
  ierr = PetscMalloc4((ny-nprocslocal)*bs,PetscScalar,&from->values,ny-nprocslocal,PetscInt,&from->indices,nsends+1,PetscInt,&from->starts,from->n,PetscMPIInt,&from->procs);CHKERRQ(ierr);
  ctx->fromdata  = (void*)from;

  /* move data into receive scatter */
  ierr = PetscMalloc2(size,PetscInt,&lowner,nsends+1,PetscInt,&start);CHKERRQ(ierr);
  count = 0; from->starts[0] = start[0] = 0;
  for (i=0; i<size; i++) {
    if (nprocs[i]) {
      lowner[i]            = count;
      from->procs[count++] = i;
      from->starts[count]  = start[count] = start[count-1] + nprocs[i];
    }
  }

  for (i=0; i<nx; i++) {
    if (owner[i] != rank) {
      from->indices[start[lowner[owner[i]]]++] = inidy[i];
      if (inidy[i] >= lengthy) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Scattering past end of TO vector");
    }
  }
  ierr = PetscFree2(lowner,start);CHKERRQ(ierr);
  ierr = PetscFree2(nprocs,owner);CHKERRQ(ierr);
    
  /* wait on sends */
  if (nsends) {
    ierr = PetscMalloc(nsends*sizeof(MPI_Status),&send_status);CHKERRQ(ierr);
    ierr = MPI_Waitall(nsends,send_waits,send_status);CHKERRQ(ierr);
    ierr = PetscFree(send_status);CHKERRQ(ierr);
  }
  ierr = PetscFree3(svalues,send_waits,starts);CHKERRQ(ierr);

  if (nprocslocal) {
    PetscInt nt = from->local.n = to->local.n = nprocslocal;
    /* we have a scatter to ourselves */
    ierr = PetscMalloc(nt*sizeof(PetscInt),&to->local.vslots);CHKERRQ(ierr);
    ierr = PetscMalloc(nt*sizeof(PetscInt),&from->local.vslots);CHKERRQ(ierr);
    nt   = 0;
    for (i=0; i<nx; i++) {
      idx = inidx[i];
      if (idx >= owners[rank] && idx < owners[rank+1]) {
        to->local.vslots[nt]     = idx - owners[rank];        
        from->local.vslots[nt++] = inidy[i];        
        if (inidy[i] >= lengthy) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Scattering past end of TO vector");
      }
    }
  } else { 
    from->local.n      = 0;
    from->local.vslots = 0;
    to->local.n        = 0; 
    to->local.vslots   = 0;
  } 

  from->local.nonmatching_computed = PETSC_FALSE;
  from->local.n_nonmatching        = 0;
  from->local.slots_nonmatching    = 0;
  to->local.nonmatching_computed   = PETSC_FALSE;
  to->local.n_nonmatching          = 0;
  to->local.slots_nonmatching      = 0;

  from->type = VEC_SCATTER_MPI_GENERAL;
  to->type   = VEC_SCATTER_MPI_GENERAL; 
  from->bs = bs;
  to->bs   = bs;
  ierr = VecScatterCreateCommon_PtoS(from,to,ctx); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
   bs indicates how many elements there are in each block. Normally this would be 1.
*/
#undef __FUNCT__  
#define __FUNCT__ "VecScatterCreateCommon_PtoS"
PetscErrorCode VecScatterCreateCommon_PtoS(VecScatter_MPI_General *from,VecScatter_MPI_General *to,VecScatter ctx)
{
  MPI_Comm       comm = ((PetscObject)ctx)->comm;
  PetscMPIInt    tag  = ((PetscObject)ctx)->tag, tagr;
  PetscInt       bs   = to->bs;
  PetscMPIInt    size;
  PetscInt       i, n;
  PetscErrorCode ierr;
  
  PetscFunctionBegin;
  ierr = PetscObjectGetNewTag((PetscObject)ctx,&tagr);CHKERRQ(ierr);
  ctx->destroy = VecScatterDestroy_PtoP;

  ctx->reproduce = PETSC_FALSE;
  to->sendfirst  = PETSC_FALSE;
  ierr = PetscOptionsGetTruth(PETSC_NULL,"-vecscatter_reproduce",&ctx->reproduce,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetTruth(PETSC_NULL,"-vecscatter_sendfirst",&to->sendfirst,PETSC_NULL);CHKERRQ(ierr);
  from->sendfirst = to->sendfirst;

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

  to->use_alltoallv = PETSC_FALSE;
  ierr = PetscOptionsGetTruth(PETSC_NULL,"-vecscatter_alltoall",&to->use_alltoallv,PETSC_NULL);CHKERRQ(ierr);
  from->use_alltoallv = to->use_alltoallv;
  if (from->use_alltoallv) PetscInfo(ctx,"Using MPI_Alltoallv() for scatter\n");
#if defined(PETSC_HAVE_MPI_ALLTOALLW)  && !defined(PETSC_USE_64BIT_INDICES)
  if (to->use_alltoallv) {
    to->use_alltoallw = PETSC_FALSE;
    ierr = PetscOptionsGetTruth(PETSC_NULL,"-vecscatter_nopack",&to->use_alltoallw,PETSC_NULL);CHKERRQ(ierr);
  }
  from->use_alltoallw = to->use_alltoallw;
  if (from->use_alltoallw) PetscInfo(ctx,"Using MPI_Alltoallw() for scatter\n");
#endif

#if defined(PETSC_HAVE_MPI_WIN_CREATE)
  to->use_window = PETSC_FALSE;
  ierr = PetscOptionsGetTruth(PETSC_NULL,"-vecscatter_window",&to->use_window,PETSC_NULL);CHKERRQ(ierr);
  from->use_window = to->use_window;
#endif

  if (to->use_alltoallv) {

    ierr       = PetscMalloc2(size,PetscMPIInt,&to->counts,size,PetscMPIInt,&to->displs);CHKERRQ(ierr);
    ierr       = PetscMemzero(to->counts,size*sizeof(PetscMPIInt));
    for (i=0; i<to->n; i++) {
      to->counts[to->procs[i]] = bs*(to->starts[i+1] - to->starts[i]);
    }
    to->displs[0] = 0;
    for (i=1; i<size; i++) {
      to->displs[i] = to->displs[i-1] + to->counts[i-1]; 
    }

    ierr       = PetscMalloc2(size,PetscMPIInt,&from->counts,size,PetscMPIInt,&from->displs);CHKERRQ(ierr);
    ierr       = PetscMemzero(from->counts,size*sizeof(PetscMPIInt));
    for (i=0; i<from->n; i++) {
      from->counts[from->procs[i]] = bs*(from->starts[i+1] - from->starts[i]);
    }
    from->displs[0] = 0;
    for (i=1; i<size; i++) {
      from->displs[i] = from->displs[i-1] + from->counts[i-1]; 
    }
#if defined(PETSC_HAVE_MPI_ALLTOALLW) && !defined(PETSC_USE_64BIT_INDICES)
    if (to->use_alltoallw) {
      PetscMPIInt mpibs = PetscMPIIntCast(bs), mpilen;
      ctx->packtogether = PETSC_FALSE;
      ierr       = PetscMalloc3(size,PetscMPIInt,&to->wcounts,size,PetscMPIInt,&to->wdispls,size,MPI_Datatype,&to->types);CHKERRQ(ierr);
      ierr       = PetscMemzero(to->wcounts,size*sizeof(PetscMPIInt));CHKERRQ(ierr);
      ierr       = PetscMemzero(to->wdispls,size*sizeof(PetscMPIInt));CHKERRQ(ierr);
      for (i=0; i<size; i++) {
        to->types[i] = MPIU_SCALAR;
      }

      for (i=0; i<to->n; i++) {
        to->wcounts[to->procs[i]] = 1;
        mpilen = PetscMPIIntCast(to->starts[i+1]-to->starts[i]);
        ierr = MPI_Type_create_indexed_block(mpilen,mpibs,to->indices+to->starts[i],MPIU_SCALAR,to->types+to->procs[i]);CHKERRQ(ierr);
        ierr = MPI_Type_commit(to->types+to->procs[i]);CHKERRQ(ierr);
      }
      ierr       = PetscMalloc3(size,PetscMPIInt,&from->wcounts,size,PetscMPIInt,&from->wdispls,size,MPI_Datatype,&from->types);CHKERRQ(ierr);
      ierr       = PetscMemzero(from->wcounts,size*sizeof(PetscMPIInt));CHKERRQ(ierr);
      ierr       = PetscMemzero(from->wdispls,size*sizeof(PetscMPIInt));CHKERRQ(ierr);
      for (i=0; i<size; i++) {
        from->types[i] = MPIU_SCALAR;
      }
      if (from->contiq) {
        PetscInfo(ctx,"Scattered vector entries are stored contiquously, taking advantage of this with -vecscatter_alltoall\n");
	for (i=0; i<from->n; i++) {
	  from->wcounts[from->procs[i]] = bs*(from->starts[i+1] - from->starts[i]);
        }
        if (from->n) from->wdispls[from->procs[0]] = sizeof(PetscScalar)*from->indices[0];
	for (i=1; i<from->n; i++) {
          from->wdispls[from->procs[i]] = from->wdispls[from->procs[i-1]] + sizeof(PetscScalar)*from->wcounts[from->procs[i-1]]; 
        }
      } else {
	for (i=0; i<from->n; i++) {
	  from->wcounts[from->procs[i]] = 1;
          mpilen = PetscMPIIntCast(from->starts[i+1]-from->starts[i]);
	  ierr = MPI_Type_create_indexed_block(mpilen,mpibs,from->indices+from->starts[i],MPIU_SCALAR,from->types+from->procs[i]);CHKERRQ(ierr);
	  ierr = MPI_Type_commit(from->types+from->procs[i]);CHKERRQ(ierr);
        }
      }
    } else {
      ctx->copy = VecScatterCopy_PtoP_AllToAll;
    }
#else 
    to->use_alltoallw   = PETSC_FALSE;
    from->use_alltoallw = PETSC_FALSE;
    ctx->copy           = VecScatterCopy_PtoP_AllToAll;
#endif
#if defined(PETSC_HAVE_MPI_WIN_CREATE)
  } else if (to->use_window) {
    PetscMPIInt temptag,winsize;
    MPI_Request *request;
    MPI_Status  *status;
    
    ierr = PetscObjectGetNewTag((PetscObject)ctx,&temptag);CHKERRQ(ierr);
    winsize = (to->n ? to->starts[to->n] : 0)*bs*sizeof(PetscScalar);
    ierr = MPI_Win_create(to->values ? to->values : MPI_BOTTOM,winsize,sizeof(PetscScalar),MPI_INFO_NULL,comm,&to->window);CHKERRQ(ierr); 
    ierr = PetscMalloc(to->n,&to->winstarts);CHKERRQ(ierr);
    ierr = PetscMalloc2(to->n,MPI_Request,&request,to->n,MPI_Status,&status);CHKERRQ(ierr);
    for (i=0; i<to->n; i++) {
      ierr = MPI_Irecv(to->winstarts+i,1,MPIU_INT,to->procs[i],temptag,comm,request+i);CHKERRQ(ierr);
    }
    for (i=0; i<from->n; i++) {
      ierr = MPI_Send(from->starts+i,1,MPIU_INT,from->procs[i],temptag,comm);CHKERRQ(ierr);
    }
    ierr = MPI_Waitall(to->n,request,status);CHKERRQ(ierr);
    ierr = PetscFree2(request,status);CHKERRQ(ierr);

    winsize = (from->n ? from->starts[from->n] : 0)*bs*sizeof(PetscScalar);
    ierr = MPI_Win_create(from->values ? from->values : MPI_BOTTOM,winsize,sizeof(PetscScalar),MPI_INFO_NULL,comm,&from->window);CHKERRQ(ierr); 
    ierr = PetscMalloc(from->n,&from->winstarts);CHKERRQ(ierr);
    ierr = PetscMalloc2(from->n,MPI_Request,&request,from->n,MPI_Status,&status);CHKERRQ(ierr);
    for (i=0; i<from->n; i++) {
      ierr = MPI_Irecv(from->winstarts+i,1,MPIU_INT,from->procs[i],temptag,comm,request+i);CHKERRQ(ierr);
    }
    for (i=0; i<to->n; i++) {
      ierr = MPI_Send(to->starts+i,1,MPIU_INT,to->procs[i],temptag,comm);CHKERRQ(ierr);
    }
    ierr = MPI_Waitall(from->n,request,status);CHKERRQ(ierr);
    ierr = PetscFree2(request,status);CHKERRQ(ierr);
#endif
  } else {
    PetscTruth  use_rsend = PETSC_FALSE, use_ssend = PETSC_FALSE;
    PetscInt    *sstarts = to->starts,  *rstarts = from->starts;
    PetscMPIInt *sprocs  = to->procs,   *rprocs  = from->procs;
    MPI_Request *swaits  = to->requests,*rwaits  = from->requests;
    MPI_Request *rev_swaits,*rev_rwaits;
    PetscScalar *Ssvalues = to->values, *Srvalues = from->values;

    /* allocate additional wait variables for the "reverse" scatter */
    ierr = PetscMalloc(to->n*sizeof(MPI_Request),&rev_rwaits);CHKERRQ(ierr);
    ierr = PetscMalloc(from->n*sizeof(MPI_Request),&rev_swaits);CHKERRQ(ierr);
    to->rev_requests   = rev_rwaits;
    from->rev_requests = rev_swaits;

    /* Register the receives that you will use later (sends for scatter reverse) */
    ierr = PetscOptionsGetTruth(PETSC_NULL,"-vecscatter_rsend",&use_rsend,PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsGetTruth(PETSC_NULL,"-vecscatter_ssend",&use_ssend,PETSC_NULL);CHKERRQ(ierr);
    if (use_rsend) {
      ierr = PetscInfo(ctx,"Using VecScatter ready receiver mode\n");CHKERRQ(ierr);
      to->use_readyreceiver    = PETSC_TRUE;
      from->use_readyreceiver  = PETSC_TRUE;
    } else {
      to->use_readyreceiver    = PETSC_FALSE;
      from->use_readyreceiver  = PETSC_FALSE;
    }
    if (use_ssend) {
      ierr = PetscInfo(ctx,"Using VecScatter Ssend mode\n");CHKERRQ(ierr);
    }

    for (i=0; i<from->n; i++) {
      if (use_rsend) {
        ierr = MPI_Rsend_init(Srvalues+bs*rstarts[i],bs*rstarts[i+1]-bs*rstarts[i],MPIU_SCALAR,rprocs[i],tagr,comm,rev_swaits+i);CHKERRQ(ierr);
      } else if (use_ssend) {
        ierr = MPI_Ssend_init(Srvalues+bs*rstarts[i],bs*rstarts[i+1]-bs*rstarts[i],MPIU_SCALAR,rprocs[i],tagr,comm,rev_swaits+i);CHKERRQ(ierr);
      } else {
        ierr = MPI_Send_init(Srvalues+bs*rstarts[i],bs*rstarts[i+1]-bs*rstarts[i],MPIU_SCALAR,rprocs[i],tagr,comm,rev_swaits+i);CHKERRQ(ierr);
      }
    }

    for (i=0; i<to->n; i++) {
      if (use_rsend) {
	ierr = MPI_Rsend_init(Ssvalues+bs*sstarts[i],bs*sstarts[i+1]-bs*sstarts[i],MPIU_SCALAR,sprocs[i],tag,comm,swaits+i);CHKERRQ(ierr);
      } else if (use_ssend) {
	ierr = MPI_Ssend_init(Ssvalues+bs*sstarts[i],bs*sstarts[i+1]-bs*sstarts[i],MPIU_SCALAR,sprocs[i],tag,comm,swaits+i);CHKERRQ(ierr);
      } else {
	ierr = MPI_Send_init(Ssvalues+bs*sstarts[i],bs*sstarts[i+1]-bs*sstarts[i],MPIU_SCALAR,sprocs[i],tag,comm,swaits+i);CHKERRQ(ierr);
      }
    }
    /* Register receives for scatter and reverse */
    for (i=0; i<from->n; i++) {
      ierr = MPI_Recv_init(Srvalues+bs*rstarts[i],bs*rstarts[i+1]-bs*rstarts[i],MPIU_SCALAR,rprocs[i],tag,comm,rwaits+i);CHKERRQ(ierr);
    }
    for (i=0; i<to->n; i++) {
      ierr = MPI_Recv_init(Ssvalues+bs*sstarts[i],bs*sstarts[i+1]-bs*sstarts[i],MPIU_SCALAR,sprocs[i],tagr,comm,rev_rwaits+i);CHKERRQ(ierr);
    } 
    if (use_rsend) {
      if (to->n)   {ierr = MPI_Startall_irecv(to->starts[to->n]*to->bs,to->n,to->rev_requests);CHKERRQ(ierr);}
      if (from->n) {ierr = MPI_Startall_irecv(from->starts[from->n]*from->bs,from->n,from->requests);CHKERRQ(ierr);}
      ierr = MPI_Barrier(comm);CHKERRQ(ierr);
    }

    ctx->copy      = VecScatterCopy_PtoP_X;
  }
  ierr = PetscInfo1(ctx,"Using blocksize %D scatter\n",bs);CHKERRQ(ierr);
  switch (bs) {
  case 12: 
    ctx->begin     = VecScatterBegin_12;
    ctx->end       = VecScatterEnd_12; 
    break;
  case 8: 
    ctx->begin     = VecScatterBegin_8;
    ctx->end       = VecScatterEnd_8; 
    break;
  case 7: 
    ctx->begin     = VecScatterBegin_7;
    ctx->end       = VecScatterEnd_7; 
    break;
  case 6: 
    ctx->begin     = VecScatterBegin_6;
    ctx->end       = VecScatterEnd_6; 
    break;
  case 5: 
    ctx->begin     = VecScatterBegin_5;
    ctx->end       = VecScatterEnd_5; 
    break;
  case 4: 
    ctx->begin     = VecScatterBegin_4;
    ctx->end       = VecScatterEnd_4; 
    break;
  case 3: 
    ctx->begin     = VecScatterBegin_3;
    ctx->end       = VecScatterEnd_3; 
    break;
  case 2: 
    ctx->begin     = VecScatterBegin_2;
    ctx->end       = VecScatterEnd_2; 
    break;
  case 1: 
    ctx->begin     = VecScatterBegin_1;
    ctx->end       = VecScatterEnd_1; 
    break;
  default:
    SETERRQ(PETSC_ERR_SUP,"Blocksize not supported");
  }
  ctx->view      = VecScatterView_MPI;
  /* Check if the local scatter is actually a copy; important special case */
  if (to->local.n) { 
    ierr = VecScatterLocalOptimizeCopy_Private(ctx,&to->local,&from->local,bs);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}



/* ------------------------------------------------------------------------------------*/
/*
         Scatter from local Seq vectors to a parallel vector. 
         Reverses the order of the arguments, calls VecScatterCreate_PtoS() then
         reverses the result.
*/
#undef __FUNCT__  
#define __FUNCT__ "VecScatterCreate_StoP"
PetscErrorCode VecScatterCreate_StoP(PetscInt nx,const PetscInt *inidx,PetscInt ny,const PetscInt *inidy,Vec xin,Vec yin,PetscInt bs,VecScatter ctx)
{
  PetscErrorCode         ierr;
  MPI_Request            *waits;
  VecScatter_MPI_General *to,*from;

  PetscFunctionBegin;
  ierr          = VecScatterCreate_PtoS(ny,inidy,nx,inidx,yin,xin,bs,ctx);CHKERRQ(ierr);
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
#undef __FUNCT__  
#define __FUNCT__ "VecScatterCreate_PtoP"
PetscErrorCode VecScatterCreate_PtoP(PetscInt nx,const PetscInt *inidx,PetscInt ny,const PetscInt *inidy,Vec xin,Vec yin,VecScatter ctx)
{
  PetscErrorCode ierr;
  PetscMPIInt    size,rank,tag,imdex,n;
  PetscInt       *owners = xin->map->range;
  PetscMPIInt    *nprocs = PETSC_NULL;
  PetscInt       i,j,idx,nsends,*local_inidx = PETSC_NULL,*local_inidy = PETSC_NULL;
  PetscInt       *owner = PETSC_NULL,*starts = PETSC_NULL,count,slen;
  PetscInt       *rvalues = PETSC_NULL,*svalues = PETSC_NULL,base,*values = PETSC_NULL,*rsvalues,recvtotal,lastidx;
  PetscMPIInt    *onodes1,*olengths1,nrecvs;
  MPI_Comm       comm;
  MPI_Request    *send_waits = PETSC_NULL,*recv_waits = PETSC_NULL;
  MPI_Status     recv_status,*send_status = PETSC_NULL;
  PetscTruth     duplicate = PETSC_FALSE;
#if defined(PETSC_USE_DEBUG)
  PetscTruth     found = PETSC_FALSE;
#endif

  PetscFunctionBegin;
  ierr = PetscObjectGetNewTag((PetscObject)ctx,&tag);CHKERRQ(ierr);
  ierr = PetscObjectGetComm((PetscObject)xin,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
  if (size == 1) {
    ierr = VecScatterCreate_StoP(nx,inidx,ny,inidy,xin,yin,1,ctx);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }

  /*
     Each processor ships off its inidx[j] and inidy[j] to the appropriate processor
     They then call the StoPScatterCreate()
  */
  /*  first count number of contributors to each processor */
  ierr  = PetscMalloc3(size,PetscMPIInt,&nprocs,nx,PetscInt,&owner,(size+1),PetscInt,&starts);CHKERRQ(ierr);
  ierr  = PetscMemzero(nprocs,size*sizeof(PetscMPIInt));CHKERRQ(ierr);
  lastidx = -1;
  j       = 0;
  for (i=0; i<nx; i++) {
    /* if indices are NOT locally sorted, need to start search at the beginning */
    if (lastidx > (idx = inidx[i])) j = 0;
    lastidx = idx;
    for (; j<size; j++) {
      if (idx >= owners[j] && idx < owners[j+1]) {
        nprocs[j]++; 
        owner[i] = j; 
#if defined(PETSC_USE_DEBUG)
        found = PETSC_TRUE; 
#endif
        break;
      }
    }
#if defined(PETSC_USE_DEBUG)
    if (!found) SETERRQ1(PETSC_ERR_ARG_OUTOFRANGE,"Index %D out of range",idx);
    found = PETSC_FALSE;
#endif
  }
  nsends = 0;  for (i=0; i<size; i++) { nsends += (nprocs[i] > 0);} 

  /* inform other processors of number of messages and max length*/
  ierr = PetscGatherNumberOfMessages(comm,PETSC_NULL,nprocs,&nrecvs);CHKERRQ(ierr);
  ierr = PetscGatherMessageLengths(comm,nsends,nrecvs,nprocs,&onodes1,&olengths1);CHKERRQ(ierr);
  ierr = PetscSortMPIIntWithArray(nrecvs,onodes1,olengths1);CHKERRQ(ierr);
  recvtotal = 0; for (i=0; i<nrecvs; i++) recvtotal += olengths1[i];

  /* post receives:   */
  ierr = PetscMalloc5(2*recvtotal,PetscInt,&rvalues,2*nx,PetscInt,&svalues,nrecvs,MPI_Request,&recv_waits,nsends,MPI_Request,&send_waits,nsends,MPI_Status,&send_status);CHKERRQ(ierr);

  count = 0;
  for (i=0; i<nrecvs; i++) {
    ierr = MPI_Irecv((rvalues+2*count),2*olengths1[i],MPIU_INT,onodes1[i],tag,comm,recv_waits+i);CHKERRQ(ierr);
    count += olengths1[i];
  }
  ierr = PetscFree(onodes1);CHKERRQ(ierr);

  /* do sends:
      1) starts[i] gives the starting index in svalues for stuff going to 
         the ith processor
  */
  starts[0]= 0; 
  for (i=1; i<size; i++) { starts[i] = starts[i-1] + nprocs[i-1];} 
  for (i=0; i<nx; i++) {
    svalues[2*starts[owner[i]]]       = inidx[i];
    svalues[1 + 2*starts[owner[i]]++] = inidy[i];
  }

  starts[0] = 0;
  for (i=1; i<size+1; i++) { starts[i] = starts[i-1] + nprocs[i-1];} 
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
    ierr = MPI_Get_count(&recv_status,MPIU_INT,&n);CHKERRQ(ierr);
    slen += n/2;
    count--;
  }
  if (slen != recvtotal) SETERRQ2(PETSC_ERR_PLIB,"Total message lengths %D not as expected %D",slen,recvtotal);
  
  ierr  = PetscMalloc2(slen,PetscInt,&local_inidx,slen,PetscInt,&local_inidy);CHKERRQ(ierr);
  base  = owners[rank];
  count = 0;
  rsvalues = rvalues;
  for (i=0; i<nrecvs; i++) {
    values = rsvalues;
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
  ierr = PetscSortIntWithArray(slen,local_inidx,local_inidy);CHKERRQ(ierr);
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
      if (local_inidy[i] != local_inidy[i+1]) {
        i++;
      } else { /* found a duplicate */
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
  ierr = VecScatterCreate_StoP(slen,local_inidx,slen,local_inidy,xin,yin,1,ctx);CHKERRQ(ierr);
  ierr = PetscFree2(local_inidx,local_inidy);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
