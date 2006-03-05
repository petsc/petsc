#define PETSCVEC_DLL

/*
    Defines parallel vector scatters.
*/

#include "src/vec/is/isimpl.h"
#include "private/vecimpl.h"                     /*I "petscvec.h" I*/
#include "src/vec/vec/impls/dvecimpl.h"
#include "src/vec/vec/impls/mpi/pvecimpl.h"
#include "petscsys.h"

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
    ierr = MPI_Comm_rank(ctx->comm,&rank);CHKERRQ(ierr);
    ierr = PetscViewerGetFormat(viewer,&format);CHKERRQ(ierr);
    if (format ==  PETSC_VIEWER_ASCII_INFO) {
      PetscInt nsend_max,nrecv_max,lensend_max,lenrecv_max,alldata,itmp;

      ierr = MPI_Reduce(&to->n,&nsend_max,1,MPIU_INT,MPI_MAX,0,ctx->comm);CHKERRQ(ierr);
      ierr = MPI_Reduce(&from->n,&nrecv_max,1,MPIU_INT,MPI_MAX,0,ctx->comm);CHKERRQ(ierr);
      itmp = to->starts[to->n+1];
      ierr = MPI_Reduce(&itmp,&lensend_max,1,MPIU_INT,MPI_MAX,0,ctx->comm);CHKERRQ(ierr);
      itmp = from->starts[from->n+1];
      ierr = MPI_Reduce(&itmp,&lenrecv_max,1,MPIU_INT,MPI_MAX,0,ctx->comm);CHKERRQ(ierr);
      ierr = MPI_Reduce(&itmp,&alldata,1,MPIU_INT,MPI_SUM,0,ctx->comm);CHKERRQ(ierr);

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
  then know that we need not perform that portion of the scatter.
*/
#undef __FUNCT__  
#define __FUNCT__ "VecScatterLocalOptimize_Private"
PetscErrorCode VecScatterLocalOptimize_Private(VecScatter_Seq_General *to,VecScatter_Seq_General *from)
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
    ierr = PetscInfo1(0,"Reduced %D to 0\n", n);CHKERRQ(ierr);
  } else if (n_nonmatching == n) {
    to->nonmatching_computed = PETSC_FALSE;
    ierr = PetscInfo(0,"All values non-matching\n");CHKERRQ(ierr);
  } else {
    to->nonmatching_computed= PETSC_TRUE;
    to->n_nonmatching       = from->n_nonmatching = n_nonmatching;
    ierr = PetscMalloc2(n_nonmatching,PetscInt,&nto_slots,n_nonmatching,PetscInt,&nfrom_slots);CHKERRQ(ierr);
    to->slots_nonmatching   = nto_slots;
    from->slots_nonmatching = nfrom_slots;
    for (i=0; i<n; i++) {
      if (to_slots[i] != from_slots[i]) {
        nto_slots[j]   = to_slots[i];
        nfrom_slots[j] = from_slots[i];
        j++;
      }
    }
    ierr = PetscInfo2(0,"Reduced %D to %D\n",n,n_nonmatching);CHKERRQ(ierr);
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

  PetscFunctionBegin;
  CHKMEMQ;
  ierr = PetscFree2(to->local.vslots,from->local.vslots);CHKERRQ(ierr);
  ierr = PetscFree2(to->counts,to->displs);CHKERRQ(ierr);
  ierr = PetscFree2(from->counts,from->displs);CHKERRQ(ierr);
  ierr = PetscFree2(to->local.slots_nonmatching,from->local.slots_nonmatching);CHKERRQ(ierr);
  ierr = PetscFree2(to->rev_requests,from->rev_requests);CHKERRQ(ierr);
  ierr = PetscFree7(to->values,to->requests,to->indices,to->starts,to->procs,to->sstatus,to->rstatus);CHKERRQ(ierr);
  ierr = PetscFree5(from->values,from->requests,from->indices,from->starts,from->procs);CHKERRQ(ierr);
  ierr = PetscFree(from);CHKERRQ(ierr);
  ierr = PetscFree(to);CHKERRQ(ierr);
  ierr = PetscHeaderDestroy(ctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* --------------------------------------------------------------------------------------*/
/*
     Even though the next routines are written with parallel 
  vectors, either xin or yin (but not both) may be Seq
  vectors, one for each processor.
  
     from indices indicate where arriving stuff is stashed
     to   indices indicate where departing stuff came from. 
     the naming can be VERY confusing.

*/
#undef __FUNCT__  
#define __FUNCT__ "VecScatterBegin_PtoP"
PetscErrorCode VecScatterBegin_PtoP(Vec xin,Vec yin,InsertMode addv,ScatterMode mode,VecScatter ctx)
{
  VecScatter_MPI_General *to,*from;
  MPI_Comm               comm = ctx->comm;
  PetscScalar            *xv,*yv,*val,*rvalues,*svalues;
  MPI_Request            *rwaits,*swaits;
  PetscInt               i,j,*indices,*rstarts,*sstarts;
  PetscMPIInt            tag = ctx->tag,*rprocs,*sprocs;
  PetscErrorCode         ierr;
  PetscInt               nrecvs,nsends,iend;

  PetscFunctionBegin;
  CHKMEMQ;
  ierr = VecGetArray(xin,&xv);CHKERRQ(ierr);
  if (xin != yin) {ierr = VecGetArray(yin,&yv);CHKERRQ(ierr);} else {yv = xv;}
  if (mode & SCATTER_REVERSE){
    to      = (VecScatter_MPI_General*)ctx->fromdata;
    from    = (VecScatter_MPI_General*)ctx->todata;
    rwaits  = from->rev_requests;
    swaits  = to->rev_requests;
  } else {
    to   = (VecScatter_MPI_General*)ctx->todata;
    from = (VecScatter_MPI_General*)ctx->fromdata;
    rwaits  = from->requests;
    swaits  = to->requests;
  }
  rvalues  = from->values;
  svalues  = to->values;
  nrecvs   = from->n;
  nsends   = to->n;
  indices  = to->indices;
  rstarts  = from->starts;
  sstarts  = to->starts;
  rprocs   = from->procs;
  sprocs   = to->procs;

  if (!(mode & SCATTER_LOCAL)) {  

    if (to->sendfirst) {
      /* do sends:  */
      for (i=0; i<nsends; i++) {
        val  = svalues + sstarts[i];
        iend = sstarts[i+1]-sstarts[i];
        /* pack the message */
        for (j=0; j<iend; j++) {
          val[j] = xv[*indices++];
        } 
        ierr = MPI_Isend(val,iend,MPIU_SCALAR,sprocs[i],tag,comm,swaits+i);CHKERRQ(ierr);
      }
    } 
 
    /* post receives:   */
    for (i=0; i<nrecvs; i++) {
      ierr = MPI_Irecv(rvalues+rstarts[i],rstarts[i+1]-rstarts[i],MPIU_SCALAR,rprocs[i],tag,comm,rwaits+i);CHKERRQ(ierr);
    }

    if (!to->sendfirst) {
      /* do sends:  */
      for (i=0; i<nsends; i++) {
        val  = svalues + sstarts[i];
        iend = sstarts[i+1]-sstarts[i];
        /* pack the message */
        for (j=0; j<iend; j++) {
          val[j] = xv[*indices++];
        } 
        ierr = MPI_Isend(val,iend,MPIU_SCALAR,sprocs[i],tag,comm,swaits+i);CHKERRQ(ierr);
      }
    } 
  }

  /* take care of local scatters */
  if (to->local.n && addv == INSERT_VALUES) {
    if (yv == xv && !to->local.nonmatching_computed) {
      ierr = VecScatterLocalOptimize_Private(&to->local,&from->local);CHKERRQ(ierr);
    }
    if (to->local.is_copy) {
      ierr = PetscMemcpy(yv + from->local.copy_start,xv + to->local.copy_start,to->local.copy_length);CHKERRQ(ierr);
    } else if (yv != xv || !to->local.nonmatching_computed) {
      PetscInt *tslots = to->local.vslots,*fslots = from->local.vslots;
      PetscInt n       = to->local.n;
      for (i=0; i<n; i++) {yv[fslots[i]] = xv[tslots[i]];}
    } else {
      /* 
        In this case, it is copying the values into their old locations, thus we can skip those  
      */
      PetscInt *tslots = to->local.slots_nonmatching,*fslots = from->local.slots_nonmatching;
      PetscInt n       = to->local.n_nonmatching;
      for (i=0; i<n; i++) {yv[fslots[i]] = xv[tslots[i]];}
    } 
  } else if (to->local.n) {
    PetscInt *tslots = to->local.vslots,*fslots = from->local.vslots;
    PetscInt n = to->local.n;
    if (addv == ADD_VALUES) {
      for (i=0; i<n; i++) {yv[fslots[i]] += xv[tslots[i]];}
#if !defined(PETSC_USE_COMPLEX)
    } else if (addv == MAX_VALUES) {
      for (i=0; i<n; i++) {yv[fslots[i]] = PetscMax(yv[fslots[i]],xv[tslots[i]]);}
#endif
    } else {SETERRQ(PETSC_ERR_ARG_UNKNOWN_TYPE,"Wrong insert option");}
  }

  ierr = VecRestoreArray(xin,&xv);CHKERRQ(ierr);
  if (xin != yin) {ierr = VecRestoreArray(yin,&yv);CHKERRQ(ierr);}
  CHKMEMQ;
  PetscFunctionReturn(0);
}

/* --------------------------------------------------------------------------------------*/
#undef __FUNCT__  
#define __FUNCT__ "VecScatterEnd_PtoP"
PetscErrorCode VecScatterEnd_PtoP(Vec xin,Vec yin,InsertMode addv,ScatterMode mode,VecScatter ctx)
{
  VecScatter_MPI_General *to,*from;
  PetscScalar            *rvalues,*yv,*val;
  PetscErrorCode         ierr;
  PetscInt               nrecvs,nsends,i,*indices,count,n,*rstarts,*lindices;
  PetscMPIInt            imdex;
  MPI_Request            *rwaits,*swaits;
  MPI_Status             rstatus,*sstatus;

  PetscFunctionBegin;
  CHKMEMQ;
  if (mode & SCATTER_LOCAL) PetscFunctionReturn(0);
  ierr = VecGetArray(yin,&yv);CHKERRQ(ierr);

  to      = (VecScatter_MPI_General*)ctx->todata;
  from    = (VecScatter_MPI_General*)ctx->fromdata;
  sstatus = to->sstatus;
  rwaits  = from->requests;
  swaits  = to->requests;
  if (mode & SCATTER_REVERSE){
    to   = (VecScatter_MPI_General*)ctx->fromdata;
    from = (VecScatter_MPI_General*)ctx->todata;
    rwaits   = from->rev_requests;
    swaits   = to->rev_requests;
  }
  rvalues  = from->values;
  nrecvs   = from->n;
  nsends   = to->n;
  indices  = from->indices;
  rstarts  = from->starts;

  /*  wait on receives */
  count = nrecvs;
  while (count) {
    ierr = MPI_Waitany(nrecvs,rwaits,&imdex,&rstatus);CHKERRQ(ierr);
    /* unpack receives into our local space */
    val      = rvalues + rstarts[imdex];
    n        = rstarts[imdex+1]-rstarts[imdex];
    lindices = indices + rstarts[imdex];
    if (addv == INSERT_VALUES) {
      for (i=0; i<n; i++) {
        yv[lindices[i]] = *val++;
      }
    } else if (addv == ADD_VALUES) {
      for (i=0; i<n; i++) {
        yv[lindices[i]] += *val++;
      }
#if !defined(PETSC_USE_COMPLEX)
    } else if (addv == MAX_VALUES) {
      for (i=0; i<n; i++) {
        yv[lindices[i]] = PetscMax(yv[lindices[i]],*val); val++;
      }
#endif
    }  else {SETERRQ(PETSC_ERR_ARG_UNKNOWN_TYPE,"Wrong insert option");}
    count--;
  }
  if (from->use_readyreceiver) {  
    if (from->n) {ierr = MPI_Startall_irecv(from->starts[from->n]*from->bs,from->n,from->requests);CHKERRQ(ierr);}
    ierr = MPI_Barrier(ctx->comm);CHKERRQ(ierr);
  }

  /* wait on sends */
  if (nsends) {ierr = MPI_Waitall(nsends,swaits,sstatus);CHKERRQ(ierr);}
  ierr = VecRestoreArray(yin,&yv);CHKERRQ(ierr);
  CHKMEMQ;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecScatterBegin_PtoP_Alltoallv_1"
PetscErrorCode VecScatterBegin_PtoP_Alltoallv_1(Vec xin,Vec yin,InsertMode addv,ScatterMode mode,VecScatter ctx)
{
  VecScatter_MPI_General *to,*from;
  PetscInt               i,*indices,len;
  PetscErrorCode         ierr;
  PetscScalar            *xv,*yv,*val;

  PetscFunctionBegin;
  CHKMEMQ;
  ierr = VecGetArray(xin,&xv);CHKERRQ(ierr);
  if (xin != yin) {ierr = VecGetArray(yin,&yv);CHKERRQ(ierr);} else {yv = xv;}
  if (mode & SCATTER_REVERSE){
    to   = (VecScatter_MPI_General*)ctx->fromdata;
    from = (VecScatter_MPI_General*)ctx->todata;
  } else {
    to   = (VecScatter_MPI_General*)ctx->todata;
    from = (VecScatter_MPI_General*)ctx->fromdata;
  }
  if (!(mode & SCATTER_LOCAL)) {  
    /* pack messages */
    indices = to->indices;
    val     = to->values;
    len     = to->starts[to->n];
    for (i=0; i<len; i++) {
      val[i] = xv[*indices++];
    }
    ierr = MPI_Alltoallv(val,to->counts,to->displs,MPIU_SCALAR,from->values,from->counts,from->displs,MPIU_SCALAR,ctx->comm);CHKERRQ(ierr);
    /* unpack messages */
    indices = from->indices;
    val     = from->values;
    len     = from->starts[from->n];
    if (addv == INSERT_VALUES) {
      for (i=0; i<len; i++) {
        yv[*indices++] = val[i];
      }
    } else if (addv == ADD_VALUES) {
      for (i=0; i<len; i++) {
        yv[*indices++] += val[i];
      }
#if !defined(PETSC_USE_COMPLEX)
    } else if (addv == MAX_VALUES) {
      for (i=0; i<len; i++) {
        yv[*indices] = PetscMax(yv[*indices],val[i]); indices++;
      }
#endif
    }  else {SETERRQ(PETSC_ERR_ARG_UNKNOWN_TYPE,"Wrong insert option");}
  }

  /* take care of local scatters */
  if (to->local.n && addv == INSERT_VALUES) {
    if (yv == xv && !to->local.nonmatching_computed) {
      ierr = VecScatterLocalOptimize_Private(&to->local,&from->local);CHKERRQ(ierr);
    }
    if (to->local.is_copy) {
      ierr = PetscMemcpy(yv + from->local.copy_start,xv + to->local.copy_start,to->local.copy_length);CHKERRQ(ierr);
    } else if (yv != xv || !to->local.nonmatching_computed) {
      PetscInt *tslots = to->local.vslots,*fslots = from->local.vslots;
      PetscInt n = to->local.n;
      for (i=0; i<n; i++) {yv[fslots[i]] = xv[tslots[i]];}
    } else {
      /* 
        In this case, it is copying the values into their old locations, thus we can skip those  
      */
      PetscInt *tslots = to->local.slots_nonmatching,*fslots = from->local.slots_nonmatching;
      PetscInt n = to->local.n_nonmatching;
      for (i=0; i<n; i++) {yv[fslots[i]] = xv[tslots[i]];}
    } 
  } else if (to->local.n) {
    PetscInt *tslots = to->local.vslots,*fslots = from->local.vslots;
    PetscInt n = to->local.n;
    if (addv == ADD_VALUES) {
      for (i=0; i<n; i++) {yv[fslots[i]] += xv[tslots[i]];}
#if !defined(PETSC_USE_COMPLEX)
    } else if (addv == MAX_VALUES) {
      for (i=0; i<n; i++) {yv[fslots[i]] = PetscMax(yv[fslots[i]],xv[tslots[i]]);}
#endif
    } else {SETERRQ(PETSC_ERR_ARG_UNKNOWN_TYPE,"Wrong insert option");}
  }

  ierr = VecRestoreArray(xin,&xv);CHKERRQ(ierr);
  if (xin != yin) {ierr = VecRestoreArray(yin,&yv);CHKERRQ(ierr);}
  CHKMEMQ;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecScatterBegin_PtoP_Alltoallv_2"
PetscErrorCode VecScatterBegin_PtoP_Alltoallv_2(Vec xin,Vec yin,InsertMode addv,ScatterMode mode,VecScatter ctx)
{
  VecScatter_MPI_General *to,*from;
  PetscInt               i,*indices,len,idx;
  PetscErrorCode         ierr;
  PetscScalar            *xv,*yv,*val;

  PetscFunctionBegin;
  CHKMEMQ;
  ierr = VecGetArray(xin,&xv);CHKERRQ(ierr);
  if (xin != yin) {ierr = VecGetArray(yin,&yv);CHKERRQ(ierr);} else {yv = xv;}
  if (mode & SCATTER_REVERSE){
    to   = (VecScatter_MPI_General*)ctx->fromdata;
    from = (VecScatter_MPI_General*)ctx->todata;
  } else {
    to   = (VecScatter_MPI_General*)ctx->todata;
    from = (VecScatter_MPI_General*)ctx->fromdata;
  }
  if (!(mode & SCATTER_LOCAL)) {  
    indices = to->indices;
    val     = to->values;
    len     = to->starts[to->n];
    for (i=0; i<len; i++) {
      idx    = *indices++;
      val[0] = xv[idx];
      val[1] = xv[idx+1];
      val    += 2;
    }
    ierr = MPI_Alltoallv(to->values,to->counts,to->displs,MPIU_SCALAR,from->values,from->counts,from->displs,MPIU_SCALAR,ctx->comm);CHKERRQ(ierr);
    len      = from->starts[from->n];
    val      = from->values;
    indices  = from->indices;
    if (addv == INSERT_VALUES) {
      for (i=0; i<len; i++) {
	idx       = indices[i];
	yv[idx]   = val[0];
	yv[idx+1] = val[1];
	val      += 2;
      }
    } else if (addv == ADD_VALUES) {
      for (i=0; i<len; i++) {
	idx       = indices[i];
	yv[idx]   += val[0];
	yv[idx+1] += val[1];
	val       += 2;
      }
#if !defined(PETSC_USE_COMPLEX)
    } else if (addv == MAX_VALUES) {
      for (i=0; i<len; i++) {
	idx       = indices[i];
	yv[idx]   = PetscMax(yv[idx],val[0]);
	yv[idx+1] = PetscMax(yv[idx+1],val[1]);
	val       += 2;
      }
#endif
    }  else {SETERRQ(PETSC_ERR_ARG_UNKNOWN_TYPE,"Wrong insert option");}
  }

  /* take care of local scatters */
  if (to->local.n) {
    PetscInt *tslots = to->local.vslots,*fslots = from->local.vslots;
    PetscInt n       = to->local.n,il,ir;
    if (addv == INSERT_VALUES) {
      if (to->local.is_copy) {
        ierr = PetscMemcpy(yv + from->local.copy_start,xv + to->local.copy_start,to->local.copy_length);CHKERRQ(ierr);
      } else {
        for (i=0; i<n; i++) {
          il = fslots[i]; ir = tslots[i];
          yv[il]   = xv[ir];
          yv[il+1] = xv[ir+1];
        }
      }
    }  else if (addv == ADD_VALUES) {
      for (i=0; i<n; i++) {
        il = fslots[i]; ir = tslots[i];
        yv[il]   += xv[ir];
        yv[il+1] += xv[ir+1];
      }
#if !defined(PETSC_USE_COMPLEX)
    }  else if (addv == MAX_VALUES) {
      for (i=0; i<n; i++) {
        il = fslots[i]; ir = tslots[i];
        yv[il]   = PetscMax(yv[il],xv[ir]);
        yv[il+1] = PetscMax(yv[il+1],xv[ir+1]);
      }
#endif
    }  else {SETERRQ(PETSC_ERR_ARG_UNKNOWN_TYPE,"Wrong insert option");}
  }

  ierr = VecRestoreArray(xin,&xv);CHKERRQ(ierr);
  if (xin != yin) {ierr = VecRestoreArray(yin,&yv);CHKERRQ(ierr);}
  CHKMEMQ;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecScatterBegin_PtoP_Alltoallv_3"
PetscErrorCode VecScatterBegin_PtoP_Alltoallv_3(Vec xin,Vec yin,InsertMode addv,ScatterMode mode,VecScatter ctx)
{
  VecScatter_MPI_General *to,*from;
  PetscInt               i,*indices,len,idx;
  PetscErrorCode         ierr;
  PetscScalar            *xv,*yv,*val;

  PetscFunctionBegin;
  CHKMEMQ;
  ierr = VecGetArray(xin,&xv);CHKERRQ(ierr);
  if (xin != yin) {ierr = VecGetArray(yin,&yv);CHKERRQ(ierr);} else {yv = xv;}
  if (mode & SCATTER_REVERSE){
    to   = (VecScatter_MPI_General*)ctx->fromdata;
    from = (VecScatter_MPI_General*)ctx->todata;
  } else {
    to   = (VecScatter_MPI_General*)ctx->todata;
    from = (VecScatter_MPI_General*)ctx->fromdata;
  }
  if (!(mode & SCATTER_LOCAL)) {  
    indices = to->indices;
    val     = to->values;
    len     = to->starts[to->n];
    for (i=0; i<len; i++) {
      idx    = *indices++;
      val[0] = xv[idx];
      val[1] = xv[idx+1];
      val[2] = xv[idx+2];
      val    += 3;
    }
    ierr = MPI_Alltoallv(to->values,to->counts,to->displs,MPIU_SCALAR,from->values,from->counts,from->displs,MPIU_SCALAR,ctx->comm);CHKERRQ(ierr);
    len      = from->starts[from->n];
    val      = from->values;
    indices  = from->indices;
    if (addv == INSERT_VALUES) {
      for (i=0; i<len; i++) {
	idx       = indices[i];
	yv[idx]   = val[0];
	yv[idx+1] = val[1];
	yv[idx+2] = val[2];
	val      += 3;
      }
    } else if (addv == ADD_VALUES) {
      for (i=0; i<len; i++) {
	idx       = indices[i];
	yv[idx]   += val[0];
	yv[idx+1] += val[1];
	yv[idx+2] += val[2];
	val       += 3;
      }
#if !defined(PETSC_USE_COMPLEX)
    } else if (addv == MAX_VALUES) {
      for (i=0; i<len; i++) {
	idx       = indices[i];
	yv[idx]   = PetscMax(yv[idx],val[0]);
	yv[idx+1] = PetscMax(yv[idx+1],val[1]);
	yv[idx+2] = PetscMax(yv[idx+2],val[2]);
	val       += 3;
      }
#endif
    }  else {SETERRQ(PETSC_ERR_ARG_UNKNOWN_TYPE,"Wrong insert option");}
  }

  /* take care of local scatters */
  if (to->local.n) {
    PetscInt *tslots = to->local.vslots,*fslots = from->local.vslots;
    PetscInt n       = to->local.n,il,ir;
    if (addv == INSERT_VALUES) {
      if (to->local.is_copy) {
        ierr = PetscMemcpy(yv + from->local.copy_start,xv + to->local.copy_start,to->local.copy_length);CHKERRQ(ierr);
      } else {
        for (i=0; i<n; i++) {
          il = fslots[i]; ir = tslots[i];
          yv[il]   = xv[ir];
          yv[il+1] = xv[ir+1];
          yv[il+2] = xv[ir+2];
        }
      }
    }  else if (addv == ADD_VALUES) {
      for (i=0; i<n; i++) {
        il = fslots[i]; ir = tslots[i];
        yv[il]   += xv[ir];
        yv[il+1] += xv[ir+1];
        yv[il+2] += xv[ir+2];
      }
#if !defined(PETSC_USE_COMPLEX)
    }  else if (addv == MAX_VALUES) {
      for (i=0; i<n; i++) {
        il = fslots[i]; ir = tslots[i];
        yv[il]   = PetscMax(yv[il],xv[ir]);
        yv[il+1] = PetscMax(yv[il+1],xv[ir+1]);
        yv[il+2] = PetscMax(yv[il+2],xv[ir+2]);
      }
#endif
    }  else {SETERRQ(PETSC_ERR_ARG_UNKNOWN_TYPE,"Wrong insert option");}
  }

  ierr = VecRestoreArray(xin,&xv);CHKERRQ(ierr);
  if (xin != yin) {ierr = VecRestoreArray(yin,&yv);CHKERRQ(ierr);}
  CHKMEMQ;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecScatterBegin_PtoP_Alltoallv_4"
PetscErrorCode VecScatterBegin_PtoP_Alltoallv_4(Vec xin,Vec yin,InsertMode addv,ScatterMode mode,VecScatter ctx)
{
  VecScatter_MPI_General *to,*from;
  PetscInt               i,*indices,len,idx;
  PetscErrorCode         ierr;
  PetscScalar            *xv,*yv,*val;

  PetscFunctionBegin;
  CHKMEMQ;
  ierr = VecGetArray(xin,&xv);CHKERRQ(ierr);
  if (xin != yin) {ierr = VecGetArray(yin,&yv);CHKERRQ(ierr);} else {yv = xv;}
  if (mode & SCATTER_REVERSE){
    to   = (VecScatter_MPI_General*)ctx->fromdata;
    from = (VecScatter_MPI_General*)ctx->todata;
  } else {
    to   = (VecScatter_MPI_General*)ctx->todata;
    from = (VecScatter_MPI_General*)ctx->fromdata;
  }
  if (!(mode & SCATTER_LOCAL)) {  
    indices = to->indices;
    val     = to->values;
    len     = to->starts[to->n];
    for (i=0; i<len; i++) {
      idx    = *indices++;
      val[0] = xv[idx];
      val[1] = xv[idx+1];
      val[2] = xv[idx+2];
      val[3] = xv[idx+3];
      val    += 4;
    }
    ierr = MPI_Alltoallv(to->values,to->counts,to->displs,MPIU_SCALAR,from->values,from->counts,from->displs,MPIU_SCALAR,ctx->comm);CHKERRQ(ierr);
    len      = from->starts[from->n];
    val      = from->values;
    indices  = from->indices;
    if (addv == INSERT_VALUES) {
      for (i=0; i<len; i++) {
	idx       = indices[i];
	yv[idx]   = val[0];
	yv[idx+1] = val[1];
	yv[idx+2] = val[2];
	yv[idx+3] = val[3];
	val      += 4;
      }
    } else if (addv == ADD_VALUES) {
      for (i=0; i<len; i++) {
	idx       = indices[i];
	yv[idx]   += val[0];
	yv[idx+1] += val[1];
	yv[idx+2] += val[2];
	yv[idx+3] += val[3];
	val       += 4;
      }
#if !defined(PETSC_USE_COMPLEX)
    } else if (addv == MAX_VALUES) {
      for (i=0; i<len; i++) {
	idx       = indices[i];
	yv[idx]   = PetscMax(yv[idx],val[0]);
	yv[idx+1] = PetscMax(yv[idx+1],val[1]);
	yv[idx+2] = PetscMax(yv[idx+2],val[2]);
	yv[idx+3] = PetscMax(yv[idx+3],val[3]);
	val       += 4;
      }
#endif
    }  else {SETERRQ(PETSC_ERR_ARG_UNKNOWN_TYPE,"Wrong insert option");}
  }

  /* take care of local scatters */
  if (to->local.n) {
    PetscInt *tslots = to->local.vslots,*fslots = from->local.vslots;
    PetscInt n       = to->local.n,il,ir;
    if (addv == INSERT_VALUES) {
      if (to->local.is_copy) {
        ierr = PetscMemcpy(yv + from->local.copy_start,xv + to->local.copy_start,to->local.copy_length);CHKERRQ(ierr);
      } else {
        for (i=0; i<n; i++) {
          il = fslots[i]; ir = tslots[i];
          yv[il]   = xv[ir];
          yv[il+1] = xv[ir+1];
          yv[il+2] = xv[ir+2];
          yv[il+3] = xv[ir+3];
        }
      }
    }  else if (addv == ADD_VALUES) {
      for (i=0; i<n; i++) {
        il = fslots[i]; ir = tslots[i];
        yv[il]   += xv[ir];
        yv[il+1] += xv[ir+1];
        yv[il+2] += xv[ir+2];
        yv[il+3] += xv[ir+3];
      }
#if !defined(PETSC_USE_COMPLEX)
    }  else if (addv == MAX_VALUES) {
      for (i=0; i<n; i++) {
        il = fslots[i]; ir = tslots[i];
        yv[il]   = PetscMax(yv[il],xv[ir]);
        yv[il+1] = PetscMax(yv[il+1],xv[ir+1]);
        yv[il+2] = PetscMax(yv[il+2],xv[ir+2]);
        yv[il+3] = PetscMax(yv[il+3],xv[ir+3]);
      }
#endif
    }  else {SETERRQ(PETSC_ERR_ARG_UNKNOWN_TYPE,"Wrong insert option");}
  }

  ierr = VecRestoreArray(xin,&xv);CHKERRQ(ierr);
  if (xin != yin) {ierr = VecRestoreArray(yin,&yv);CHKERRQ(ierr);}
  CHKMEMQ;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecScatterBegin_PtoP_Alltoallv_5"
PetscErrorCode VecScatterBegin_PtoP_Alltoallv_5(Vec xin,Vec yin,InsertMode addv,ScatterMode mode,VecScatter ctx)
{
  VecScatter_MPI_General *to,*from;
  PetscInt               i,*indices,len,idx;
  PetscErrorCode         ierr;
  PetscScalar            *xv,*yv,*val;

  PetscFunctionBegin;
  CHKMEMQ;
  ierr = VecGetArray(xin,&xv);CHKERRQ(ierr);
  if (xin != yin) {ierr = VecGetArray(yin,&yv);CHKERRQ(ierr);} else {yv = xv;}
  if (mode & SCATTER_REVERSE){
    to   = (VecScatter_MPI_General*)ctx->fromdata;
    from = (VecScatter_MPI_General*)ctx->todata;
  } else {
    to   = (VecScatter_MPI_General*)ctx->todata;
    from = (VecScatter_MPI_General*)ctx->fromdata;
  }
  if (!(mode & SCATTER_LOCAL)) {  
    indices = to->indices;
    val     = to->values;
    len     = to->starts[to->n];
    for (i=0; i<len; i++) {
      idx    = *indices++;
      val[0] = xv[idx];
      val[1] = xv[idx+1];
      val[2] = xv[idx+2];
      val[3] = xv[idx+3];
      val[4] = xv[idx+4];
      val    += 5;
    }
    ierr = MPI_Alltoallv(to->values,to->counts,to->displs,MPIU_SCALAR,from->values,from->counts,from->displs,MPIU_SCALAR,ctx->comm);CHKERRQ(ierr);
    len      = from->starts[from->n];
    val      = from->values;
    indices  = from->indices;
    if (addv == INSERT_VALUES) {
      for (i=0; i<len; i++) {
	idx       = indices[i];
	yv[idx]   = val[0];
	yv[idx+1] = val[1];
	yv[idx+2] = val[2];
	yv[idx+3] = val[3];
	yv[idx+4] = val[4];
	val      += 5;
      }
    } else if (addv == ADD_VALUES) {
      for (i=0; i<len; i++) {
	idx       = indices[i];
	yv[idx]   += val[0];
	yv[idx+1] += val[1];
	yv[idx+2] += val[2];
	yv[idx+3] += val[3];
	yv[idx+4] += val[4];
	val       += 5;
      }
#if !defined(PETSC_USE_COMPLEX)
    } else if (addv == MAX_VALUES) {
      for (i=0; i<len; i++) {
	idx       = indices[i];
	yv[idx]   = PetscMax(yv[idx],val[0]);
	yv[idx+1] = PetscMax(yv[idx+1],val[1]);
	yv[idx+2] = PetscMax(yv[idx+2],val[2]);
	yv[idx+3] = PetscMax(yv[idx+3],val[3]);
	yv[idx+4] = PetscMax(yv[idx+4],val[4]);
	val       += 5;
      }
#endif
    }  else {SETERRQ(PETSC_ERR_ARG_UNKNOWN_TYPE,"Wrong insert option");}
  }

  /* take care of local scatters */
  if (to->local.n) {
    PetscInt *tslots = to->local.vslots,*fslots = from->local.vslots;
    PetscInt n       = to->local.n,il,ir;
    if (addv == INSERT_VALUES) {
      if (to->local.is_copy) {
        ierr = PetscMemcpy(yv + from->local.copy_start,xv + to->local.copy_start,to->local.copy_length);CHKERRQ(ierr);
      } else {
        for (i=0; i<n; i++) {
          il = fslots[i]; ir = tslots[i];
          yv[il]   = xv[ir];
          yv[il+1] = xv[ir+1];
          yv[il+2] = xv[ir+2];
          yv[il+3] = xv[ir+3];
          yv[il+4] = xv[ir+4];
        }
      }
    }  else if (addv == ADD_VALUES) {
      for (i=0; i<n; i++) {
        il = fslots[i]; ir = tslots[i];
        yv[il]   += xv[ir];
        yv[il+1] += xv[ir+1];
        yv[il+2] += xv[ir+2];
        yv[il+3] += xv[ir+3];
        yv[il+4] += xv[ir+4];
      }
#if !defined(PETSC_USE_COMPLEX)
    }  else if (addv == MAX_VALUES) {
      for (i=0; i<n; i++) {
        il = fslots[i]; ir = tslots[i];
        yv[il]   = PetscMax(yv[il],xv[ir]);
        yv[il+1] = PetscMax(yv[il+1],xv[ir+1]);
        yv[il+2] = PetscMax(yv[il+2],xv[ir+2]);
        yv[il+3] = PetscMax(yv[il+3],xv[ir+3]);
        yv[il+4] = PetscMax(yv[il+4],xv[ir+4]);
      }
#endif
    }  else {SETERRQ(PETSC_ERR_ARG_UNKNOWN_TYPE,"Wrong insert option");}
  }

  ierr = VecRestoreArray(xin,&xv);CHKERRQ(ierr);
  if (xin != yin) {ierr = VecRestoreArray(yin,&yv);CHKERRQ(ierr);}
  CHKMEMQ;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecScatterBegin_PtoP_Alltoallv_6"
PetscErrorCode VecScatterBegin_PtoP_Alltoallv_6(Vec xin,Vec yin,InsertMode addv,ScatterMode mode,VecScatter ctx)
{
  VecScatter_MPI_General *to,*from;
  PetscInt               i,*indices,len,idx;
  PetscErrorCode         ierr;
  PetscScalar            *xv,*yv,*val;

  PetscFunctionBegin;
  CHKMEMQ;
  ierr = VecGetArray(xin,&xv);CHKERRQ(ierr);
  if (xin != yin) {ierr = VecGetArray(yin,&yv);CHKERRQ(ierr);} else {yv = xv;}
  if (mode & SCATTER_REVERSE){
    to   = (VecScatter_MPI_General*)ctx->fromdata;
    from = (VecScatter_MPI_General*)ctx->todata;
  } else {
    to   = (VecScatter_MPI_General*)ctx->todata;
    from = (VecScatter_MPI_General*)ctx->fromdata;
  }
  if (!(mode & SCATTER_LOCAL)) {  
    indices = to->indices;
    val     = to->values;
    len     = to->starts[to->n];
    for (i=0; i<len; i++) {
      idx    = *indices++;
      val[0] = xv[idx];
      val[1] = xv[idx+1];
      val[2] = xv[idx+2];
      val[3] = xv[idx+3];
      val[4] = xv[idx+4];
      val[5] = xv[idx+5];
      val    += 6;
    }
    ierr = MPI_Alltoallv(to->values,to->counts,to->displs,MPIU_SCALAR,from->values,from->counts,from->displs,MPIU_SCALAR,ctx->comm);CHKERRQ(ierr);
    len      = from->starts[from->n];
    val      = from->values;
    indices  = from->indices;
    if (addv == INSERT_VALUES) {
      for (i=0; i<len; i++) {
	idx       = indices[i];
	yv[idx]   = val[0];
	yv[idx+1] = val[1];
	yv[idx+2] = val[2];
	yv[idx+3] = val[3];
	yv[idx+4] = val[4];
	yv[idx+5] = val[5];
	val      += 6;
      }
    } else if (addv == ADD_VALUES) {
      for (i=0; i<len; i++) {
	idx       = indices[i];
	yv[idx]   += val[0];
	yv[idx+1] += val[1];
	yv[idx+2] += val[2];
	yv[idx+3] += val[3];
	yv[idx+4] += val[4];
	yv[idx+5] += val[5];
	val       += 6;
      }
#if !defined(PETSC_USE_COMPLEX)
    } else if (addv == MAX_VALUES) {
      for (i=0; i<len; i++) {
	idx       = indices[i];
	yv[idx]   = PetscMax(yv[idx],val[0]);
	yv[idx+1] = PetscMax(yv[idx+1],val[1]);
	yv[idx+2] = PetscMax(yv[idx+2],val[2]);
	yv[idx+3] = PetscMax(yv[idx+3],val[3]);
	yv[idx+4] = PetscMax(yv[idx+4],val[4]);
	yv[idx+5] = PetscMax(yv[idx+5],val[5]);
	val       += 6;
      }
#endif
    }  else {SETERRQ(PETSC_ERR_ARG_UNKNOWN_TYPE,"Wrong insert option");}
  }

  /* take care of local scatters */
  if (to->local.n) {
    PetscInt *tslots = to->local.vslots,*fslots = from->local.vslots;
    PetscInt n       = to->local.n,il,ir;
    if (addv == INSERT_VALUES) {
      if (to->local.is_copy) {
        ierr = PetscMemcpy(yv + from->local.copy_start,xv + to->local.copy_start,to->local.copy_length);CHKERRQ(ierr);
      } else {
        for (i=0; i<n; i++) {
          il = fslots[i]; ir = tslots[i];
          yv[il]   = xv[ir];
          yv[il+1] = xv[ir+1];
          yv[il+2] = xv[ir+2];
          yv[il+3] = xv[ir+3];
          yv[il+4] = xv[ir+4];
          yv[il+5] = xv[ir+5];
        }
      }
    }  else if (addv == ADD_VALUES) {
      for (i=0; i<n; i++) {
        il = fslots[i]; ir = tslots[i];
        yv[il]   += xv[ir];
        yv[il+1] += xv[ir+1];
        yv[il+2] += xv[ir+2];
        yv[il+3] += xv[ir+3];
        yv[il+4] += xv[ir+4];
        yv[il+5] += xv[ir+5];
      }
#if !defined(PETSC_USE_COMPLEX)
    }  else if (addv == MAX_VALUES) {
      for (i=0; i<n; i++) {
        il = fslots[i]; ir = tslots[i];
        yv[il]   = PetscMax(yv[il],xv[ir]);
        yv[il+1] = PetscMax(yv[il+1],xv[ir+1]);
        yv[il+2] = PetscMax(yv[il+2],xv[ir+2]);
        yv[il+3] = PetscMax(yv[il+3],xv[ir+3]);
        yv[il+4] = PetscMax(yv[il+4],xv[ir+4]);
        yv[il+5] = PetscMax(yv[il+5],xv[ir+5]);
      }
#endif
    }  else {SETERRQ(PETSC_ERR_ARG_UNKNOWN_TYPE,"Wrong insert option");}
  }

  ierr = VecRestoreArray(xin,&xv);CHKERRQ(ierr);
  if (xin != yin) {ierr = VecRestoreArray(yin,&yv);CHKERRQ(ierr);}
  CHKMEMQ;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecScatterBegin_PtoP_Alltoallv_7"
PetscErrorCode VecScatterBegin_PtoP_Alltoallv_7(Vec xin,Vec yin,InsertMode addv,ScatterMode mode,VecScatter ctx)
{
  VecScatter_MPI_General *to,*from;
  PetscInt               i,*indices,len,idx;
  PetscErrorCode         ierr;
  PetscScalar            *xv,*yv,*val;

  PetscFunctionBegin;
  CHKMEMQ;
  ierr = VecGetArray(xin,&xv);CHKERRQ(ierr);
  if (xin != yin) {ierr = VecGetArray(yin,&yv);CHKERRQ(ierr);} else {yv = xv;}
  if (mode & SCATTER_REVERSE){
    to   = (VecScatter_MPI_General*)ctx->fromdata;
    from = (VecScatter_MPI_General*)ctx->todata;
  } else {
    to   = (VecScatter_MPI_General*)ctx->todata;
    from = (VecScatter_MPI_General*)ctx->fromdata;
  }
  if (!(mode & SCATTER_LOCAL)) {  
    indices = to->indices;
    val     = to->values;
    len     = to->starts[to->n];
    for (i=0; i<len; i++) {
      idx    = *indices++;
      val[0] = xv[idx];
      val[1] = xv[idx+1];
      val[2] = xv[idx+2];
      val[3] = xv[idx+3];
      val[4] = xv[idx+4];
      val[5] = xv[idx+5];
      val[6] = xv[idx+6];
      val    += 7;
    }
    ierr = MPI_Alltoallv(to->values,to->counts,to->displs,MPIU_SCALAR,from->values,from->counts,from->displs,MPIU_SCALAR,ctx->comm);CHKERRQ(ierr);
    len      = from->starts[from->n];
    val      = from->values;
    indices  = from->indices;
    if (addv == INSERT_VALUES) {
      for (i=0; i<len; i++) {
	idx       = indices[i];
	yv[idx]   = val[0];
	yv[idx+1] = val[1];
	yv[idx+2] = val[2];
	yv[idx+3] = val[3];
	yv[idx+4] = val[4];
	yv[idx+5] = val[5];
	yv[idx+6] = val[6];
	val      += 7;
      }
    } else if (addv == ADD_VALUES) {
      for (i=0; i<len; i++) {
	idx       = indices[i];
	yv[idx]   += val[0];
	yv[idx+1] += val[1];
	yv[idx+2] += val[2];
	yv[idx+3] += val[3];
	yv[idx+4] += val[4];
	yv[idx+5] += val[5];
	yv[idx+6] += val[6];
	val       += 7;
      }
#if !defined(PETSC_USE_COMPLEX)
    } else if (addv == MAX_VALUES) {
      for (i=0; i<len; i++) {
	idx       = indices[i];
	yv[idx]   = PetscMax(yv[idx],val[0]);
	yv[idx+1] = PetscMax(yv[idx+1],val[1]);
	yv[idx+2] = PetscMax(yv[idx+2],val[2]);
	yv[idx+3] = PetscMax(yv[idx+3],val[3]);
	yv[idx+4] = PetscMax(yv[idx+4],val[4]);
	yv[idx+5] = PetscMax(yv[idx+5],val[5]);
	yv[idx+6] = PetscMax(yv[idx+6],val[6]);
	val       += 7;
      }
#endif
    }  else {SETERRQ(PETSC_ERR_ARG_UNKNOWN_TYPE,"Wrong insert option");}
  }

  /* take care of local scatters */
  if (to->local.n) {
    PetscInt *tslots = to->local.vslots,*fslots = from->local.vslots;
    PetscInt n       = to->local.n,il,ir;
    if (addv == INSERT_VALUES) {
      if (to->local.is_copy) {
        ierr = PetscMemcpy(yv + from->local.copy_start,xv + to->local.copy_start,to->local.copy_length);CHKERRQ(ierr);
      } else {
        for (i=0; i<n; i++) {
          il = fslots[i]; ir = tslots[i];
          yv[il]   = xv[ir];
          yv[il+1] = xv[ir+1];
          yv[il+2] = xv[ir+2];
          yv[il+3] = xv[ir+3];
          yv[il+4] = xv[ir+4];
          yv[il+5] = xv[ir+5];
          yv[il+6] = xv[ir+6];
        }
      }
    }  else if (addv == ADD_VALUES) {
      for (i=0; i<n; i++) {
        il = fslots[i]; ir = tslots[i];
        yv[il]   += xv[ir];
        yv[il+1] += xv[ir+1];
        yv[il+2] += xv[ir+2];
        yv[il+3] += xv[ir+3];
        yv[il+4] += xv[ir+4];
        yv[il+5] += xv[ir+5];
        yv[il+6] += xv[ir+6];
      }
#if !defined(PETSC_USE_COMPLEX)
    }  else if (addv == MAX_VALUES) {
      for (i=0; i<n; i++) {
        il = fslots[i]; ir = tslots[i];
        yv[il]   = PetscMax(yv[il],xv[ir]);
        yv[il+1] = PetscMax(yv[il+1],xv[ir+1]);
        yv[il+2] = PetscMax(yv[il+2],xv[ir+2]);
        yv[il+3] = PetscMax(yv[il+3],xv[ir+3]);
        yv[il+4] = PetscMax(yv[il+4],xv[ir+4]);
        yv[il+5] = PetscMax(yv[il+5],xv[ir+5]);
        yv[il+6] = PetscMax(yv[il+6],xv[ir+6]);
      }
#endif
    }  else {SETERRQ(PETSC_ERR_ARG_UNKNOWN_TYPE,"Wrong insert option");}
  }

  ierr = VecRestoreArray(xin,&xv);CHKERRQ(ierr);
  if (xin != yin) {ierr = VecRestoreArray(yin,&yv);CHKERRQ(ierr);}
  CHKMEMQ;
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "VecScatterBegin_PtoP_Alltoallv_8"
PetscErrorCode VecScatterBegin_PtoP_Alltoallv_8(Vec xin,Vec yin,InsertMode addv,ScatterMode mode,VecScatter ctx)
{
  VecScatter_MPI_General *to,*from;
  PetscInt               i,*indices,len,idx;
  PetscErrorCode         ierr;
  PetscScalar            *xv,*yv,*val;

  PetscFunctionBegin;
  CHKMEMQ;
  ierr = VecGetArray(xin,&xv);CHKERRQ(ierr);
  if (xin != yin) {ierr = VecGetArray(yin,&yv);CHKERRQ(ierr);} else {yv = xv;}
  if (mode & SCATTER_REVERSE){
    to   = (VecScatter_MPI_General*)ctx->fromdata;
    from = (VecScatter_MPI_General*)ctx->todata;
  } else {
    to   = (VecScatter_MPI_General*)ctx->todata;
    from = (VecScatter_MPI_General*)ctx->fromdata;
  }
  if (!(mode & SCATTER_LOCAL)) {  
    indices = to->indices;
    val     = to->values;
    len     = to->starts[to->n];
    for (i=0; i<len; i++) {
      idx    = *indices++;
      val[0] = xv[idx];
      val[1] = xv[idx+1];
      val[2] = xv[idx+2];
      val[3] = xv[idx+3];
      val[4] = xv[idx+4];
      val[5] = xv[idx+5];
      val[6] = xv[idx+6];
      val[7] = xv[idx+7];
      val    += 8;
    }
    ierr = MPI_Alltoallv(to->values,to->counts,to->displs,MPIU_SCALAR,from->values,from->counts,from->displs,MPIU_SCALAR,ctx->comm);CHKERRQ(ierr);
    len      = from->starts[from->n];
    val      = from->values;
    indices  = from->indices;
    if (addv == INSERT_VALUES) {
      for (i=0; i<len; i++) {
	idx       = indices[i];
	yv[idx]   = val[0];
	yv[idx+1] = val[1];
	yv[idx+2] = val[2];
	yv[idx+3] = val[3];
	yv[idx+4] = val[4];
	yv[idx+5] = val[5];
	yv[idx+6] = val[6];
	yv[idx+7] = val[7];
	val      += 8;
      }
    } else if (addv == ADD_VALUES) {
      for (i=0; i<len; i++) {
	idx       = indices[i];
	yv[idx]   += val[0];
	yv[idx+1] += val[1];
	yv[idx+2] += val[2];
	yv[idx+3] += val[3];
	yv[idx+4] += val[4];
	yv[idx+5] += val[5];
	yv[idx+6] += val[6];
	yv[idx+7] += val[7];
	val       += 8;
      }
#if !defined(PETSC_USE_COMPLEX)
    } else if (addv == MAX_VALUES) {
      for (i=0; i<len; i++) {
	idx       = indices[i];
	yv[idx]   = PetscMax(yv[idx],val[0]);
	yv[idx+1] = PetscMax(yv[idx+1],val[1]);
	yv[idx+2] = PetscMax(yv[idx+2],val[2]);
	yv[idx+3] = PetscMax(yv[idx+3],val[3]);
	yv[idx+4] = PetscMax(yv[idx+4],val[4]);
	yv[idx+5] = PetscMax(yv[idx+5],val[5]);
	yv[idx+6] = PetscMax(yv[idx+6],val[6]);
	yv[idx+7] = PetscMax(yv[idx+7],val[7]);
	val       += 8;
      }
#endif
    }  else {SETERRQ(PETSC_ERR_ARG_UNKNOWN_TYPE,"Wrong insert option");}
  }

  /* take care of local scatters */
  if (to->local.n) {
    PetscInt *tslots = to->local.vslots,*fslots = from->local.vslots;
    PetscInt n       = to->local.n,il,ir;
    if (addv == INSERT_VALUES) {
      if (to->local.is_copy) {
        ierr = PetscMemcpy(yv + from->local.copy_start,xv + to->local.copy_start,to->local.copy_length);CHKERRQ(ierr);
      } else {
        for (i=0; i<n; i++) {
          il = fslots[i]; ir = tslots[i];
          yv[il]   = xv[ir];
          yv[il+1] = xv[ir+1];
          yv[il+2] = xv[ir+2];
          yv[il+3] = xv[ir+3];
          yv[il+4] = xv[ir+4];
          yv[il+5] = xv[ir+5];
          yv[il+6] = xv[ir+6];
          yv[il+7] = xv[ir+7];
        }
      }
    }  else if (addv == ADD_VALUES) {
      for (i=0; i<n; i++) {
        il = fslots[i]; ir = tslots[i];
        yv[il]   += xv[ir];
        yv[il+1] += xv[ir+1];
        yv[il+2] += xv[ir+2];
        yv[il+3] += xv[ir+3];
        yv[il+4] += xv[ir+4];
        yv[il+5] += xv[ir+5];
        yv[il+6] += xv[ir+6];
        yv[il+7] += xv[ir+7];
      }
#if !defined(PETSC_USE_COMPLEX)
    }  else if (addv == MAX_VALUES) {
      for (i=0; i<n; i++) {
        il = fslots[i]; ir = tslots[i];
        yv[il]   = PetscMax(yv[il],xv[ir]);
        yv[il+1] = PetscMax(yv[il+1],xv[ir+1]);
        yv[il+2] = PetscMax(yv[il+2],xv[ir+2]);
        yv[il+3] = PetscMax(yv[il+3],xv[ir+3]);
        yv[il+4] = PetscMax(yv[il+4],xv[ir+4]);
        yv[il+5] = PetscMax(yv[il+5],xv[ir+5]);
        yv[il+6] = PetscMax(yv[il+6],xv[ir+6]);
        yv[il+7] = PetscMax(yv[il+7],xv[ir+7]);
      }
#endif
    }  else {SETERRQ(PETSC_ERR_ARG_UNKNOWN_TYPE,"Wrong insert option");}
  }

  ierr = VecRestoreArray(xin,&xv);CHKERRQ(ierr);
  if (xin != yin) {ierr = VecRestoreArray(yin,&yv);CHKERRQ(ierr);}
  CHKMEMQ;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecScatterBegin_PtoP_Alltoallv_12"
PetscErrorCode VecScatterBegin_PtoP_Alltoallv_12(Vec xin,Vec yin,InsertMode addv,ScatterMode mode,VecScatter ctx)
{
  VecScatter_MPI_General *to,*from;
  PetscInt               i,*indices,len,idx;
  PetscErrorCode         ierr;
  PetscScalar            *xv,*yv,*val;

  PetscFunctionBegin;
  CHKMEMQ;
  ierr = VecGetArray(xin,&xv);CHKERRQ(ierr);
  if (xin != yin) {ierr = VecGetArray(yin,&yv);CHKERRQ(ierr);} else {yv = xv;}
  if (mode & SCATTER_REVERSE){
    to   = (VecScatter_MPI_General*)ctx->fromdata;
    from = (VecScatter_MPI_General*)ctx->todata;
  } else {
    to   = (VecScatter_MPI_General*)ctx->todata;
    from = (VecScatter_MPI_General*)ctx->fromdata;
  }
  if (!(mode & SCATTER_LOCAL)) {  
    indices = to->indices;
    val     = to->values;
    len     = to->starts[to->n];
    for (i=0; i<len; i++) {
      idx    = *indices++;
      val[0] = xv[idx];
      val[1] = xv[idx+1];
      val[2] = xv[idx+2];
      val[3] = xv[idx+3];
      val[4] = xv[idx+4];
      val[5] = xv[idx+5];
      val[6] = xv[idx+6];
      val[7] = xv[idx+7];
      val[8] = xv[idx+8];
      val[9] = xv[idx+9];
      val[10] = xv[idx+10];
      val[11] = xv[idx+11];
      val    += 12;
    }
    ierr = MPI_Alltoallv(to->values,to->counts,to->displs,MPIU_SCALAR,from->values,from->counts,from->displs,MPIU_SCALAR,ctx->comm);CHKERRQ(ierr);
    len      = from->starts[from->n];
    val      = from->values;
    indices  = from->indices;
    if (addv == INSERT_VALUES) {
      for (i=0; i<len; i++) {
	idx       = indices[i];
	yv[idx]   = val[0];
	yv[idx+1] = val[1];
	yv[idx+2] = val[2];
	yv[idx+3] = val[3];
	yv[idx+4] = val[4];
	yv[idx+5] = val[5];
	yv[idx+6] = val[6];
	yv[idx+7] = val[7];
	yv[idx+8] = val[8];
	yv[idx+9] = val[9];
	yv[idx+10] = val[10];
	yv[idx+11] = val[11];
	val      += 12;
      }
    } else if (addv == ADD_VALUES) {
      for (i=0; i<len; i++) {
	idx       = indices[i];
	yv[idx]   += val[0];
	yv[idx+1] += val[1];
	yv[idx+2] += val[2];
	yv[idx+3] += val[3];
	yv[idx+4] += val[4];
	yv[idx+5] += val[5];
	yv[idx+6] += val[6];
	yv[idx+7] += val[7];
	yv[idx+8] += val[8];
	yv[idx+9] += val[9];
	yv[idx+10] += val[10];
	yv[idx+11] += val[11];
	val       += 12;
      }
#if !defined(PETSC_USE_COMPLEX)
    } else if (addv == MAX_VALUES) {
      for (i=0; i<len; i++) {
	idx       = indices[i];
	yv[idx]   = PetscMax(yv[idx],val[0]);
	yv[idx+1] = PetscMax(yv[idx+1],val[1]);
	yv[idx+2] = PetscMax(yv[idx+2],val[2]);
	yv[idx+3] = PetscMax(yv[idx+3],val[3]);
	yv[idx+4] = PetscMax(yv[idx+4],val[4]);
	yv[idx+5] = PetscMax(yv[idx+5],val[5]);
	yv[idx+6] = PetscMax(yv[idx+6],val[6]);
	yv[idx+7] = PetscMax(yv[idx+7],val[7]);
	yv[idx+8] = PetscMax(yv[idx+8],val[8]);
	yv[idx+9] = PetscMax(yv[idx+9],val[9]);
	yv[idx+10] = PetscMax(yv[idx+10],val[10]);
	yv[idx+11] = PetscMax(yv[idx+11],val[11]);
	val       += 12;
      }
#endif
    }  else {SETERRQ(PETSC_ERR_ARG_UNKNOWN_TYPE,"Wrong insert option");}
  }

  /* take care of local scatters */
  if (to->local.n) {
    PetscInt *tslots = to->local.vslots,*fslots = from->local.vslots;
    PetscInt n       = to->local.n,il,ir;
    if (addv == INSERT_VALUES) {
      if (to->local.is_copy) {
        ierr = PetscMemcpy(yv + from->local.copy_start,xv + to->local.copy_start,to->local.copy_length);CHKERRQ(ierr);
      } else {
        for (i=0; i<n; i++) {
          il = fslots[i]; ir = tslots[i];
          yv[il]   = xv[ir];
          yv[il+1] = xv[ir+1];
          yv[il+2] = xv[ir+2];
          yv[il+3] = xv[ir+3];
          yv[il+4] = xv[ir+4];
          yv[il+5] = xv[ir+5];
          yv[il+6] = xv[ir+6];
          yv[il+7] = xv[ir+7];
          yv[il+8] = xv[ir+8];
          yv[il+9] = xv[ir+9];
          yv[il+10] = xv[ir+10];
          yv[il+11] = xv[ir+11];
        }
      }
    }  else if (addv == ADD_VALUES) {
      for (i=0; i<n; i++) {
        il = fslots[i]; ir = tslots[i];
        yv[il]   += xv[ir];
        yv[il+1] += xv[ir+1];
        yv[il+2] += xv[ir+2];
        yv[il+3] += xv[ir+3];
        yv[il+4] += xv[ir+4];
        yv[il+5] += xv[ir+5];
        yv[il+6] += xv[ir+6];
        yv[il+7] += xv[ir+7];
        yv[il+8] += xv[ir+8];
        yv[il+9] += xv[ir+9];
        yv[il+10] += xv[ir+10];
        yv[il+11] += xv[ir+11];
      }
#if !defined(PETSC_USE_COMPLEX)
    }  else if (addv == MAX_VALUES) {
      for (i=0; i<n; i++) {
        il = fslots[i]; ir = tslots[i];
        yv[il]   = PetscMax(yv[il],xv[ir]);
        yv[il+1] = PetscMax(yv[il+1],xv[ir+1]);
        yv[il+2] = PetscMax(yv[il+2],xv[ir+2]);
        yv[il+3] = PetscMax(yv[il+3],xv[ir+3]);
        yv[il+4] = PetscMax(yv[il+4],xv[ir+4]);
        yv[il+5] = PetscMax(yv[il+5],xv[ir+5]);
        yv[il+6] = PetscMax(yv[il+6],xv[ir+6]);
        yv[il+7] = PetscMax(yv[il+7],xv[ir+7]);
        yv[il+8] = PetscMax(yv[il+8],xv[ir+8]);
        yv[il+9] = PetscMax(yv[il+9],xv[ir+9]);
        yv[il+10] = PetscMax(yv[il+10],xv[ir+10]);
        yv[il+11] = PetscMax(yv[il+11],xv[ir+11]);
      }
#endif
    }  else {SETERRQ(PETSC_ERR_ARG_UNKNOWN_TYPE,"Wrong insert option");}
  }

  ierr = VecRestoreArray(xin,&xv);CHKERRQ(ierr);
  if (xin != yin) {ierr = VecRestoreArray(yin,&yv);CHKERRQ(ierr);}
  CHKMEMQ;
  PetscFunctionReturn(0);
}

/* ==========================================================================================*/
/*
    Special scatters for fixed block sizes. These provide better performance
    because the local copying and packing and unpacking are done with loop 
    unrolling to the size of the block.

    Also uses MPI persistent sends and receives, these (at least in theory)
    allow MPI to optimize repeated sends and receives of the same type.
*/

/* --------------------------------------------------------------------------------------*/
/*
    Special optimization to see if the local part of the scatter is actually 
    a copy. The scatter routines call PetscMemcpy() instead.
 
*/
#undef __FUNCT__  
#define __FUNCT__ "VecScatterLocalOptimizeCopy_Private"
PetscErrorCode VecScatterLocalOptimizeCopy_Private(VecScatter_Seq_General *to,VecScatter_Seq_General *from,PetscInt bs)
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
  ierr = PetscInfo(0,"Local scatter is a copy, optimizing for it\n");CHKERRQ(ierr);
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
  ierr = PetscNew(VecScatter_MPI_General,&out_to);CHKERRQ(ierr);
  ierr = PetscNew(VecScatter_MPI_General,&out_from);CHKERRQ(ierr);

  ny                = in_to->starts[in_to->n];
  out_to->n         = in_to->n; 
  out_to->type      = in_to->type;
  out_to->sendfirst = in_to->sendfirst;

  ierr = PetscMalloc7(bs*ny,PetscScalar,&out_to->values,out_to->n,MPI_Request,&out_to->requests,ny,PetscInt,&out_to->indices,out_to->n+1,PetscInt,&out_to->starts,
                      out_to->n,PetscMPIInt,&out_to->procs,PetscMax(in_to->n,in_from->n),MPI_Status,&out_to->sstatus,PetscMax(in_to->n,in_from->n),MPI_Status,
                      &out_to->rstatus);CHKERRQ(ierr);
  ierr = PetscMemcpy(out_to->indices,in_to->indices,ny*sizeof(PetscInt));CHKERRQ(ierr);
  ierr = PetscMemcpy(out_to->starts,in_to->starts,(out_to->n+1)*sizeof(PetscInt));CHKERRQ(ierr);
  ierr = PetscMemcpy(out_to->procs,in_to->procs,(out_to->n)*sizeof(PetscMPIInt));CHKERRQ(ierr);
     
  out->todata       = (void*)out_to;
  out_to->local.n   = in_to->local.n;
  out_to->local.nonmatching_computed = PETSC_FALSE;
  out_to->local.n_nonmatching        = 0;
  out_to->local.slots_nonmatching    = 0;
  if (in_to->local.n) {
    ierr = PetscMalloc2(in_to->local.n,PetscInt,&out_to->local.vslots,in_from->local.n,PetscInt,&out_from->local.vslots);CHKERRQ(ierr);
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

  ierr = PetscMalloc5(ny*bs,PetscScalar,&out_from->values,out_from->n,MPI_Request,&out_from->requests,ny,PetscInt,&out_from->indices,
                      out_from->n+1,PetscInt,&out_from->starts,out_from->n,PetscMPIInt,&out_from->procs);CHKERRQ(ierr);
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

    ierr = PetscMalloc2(in_to->n,MPI_Request,&out_to->rev_requests,in_from->n,MPI_Request,&out_from->rev_requests);CHKERRQ(ierr);

    rev_rwaits = out_to->rev_requests;
    rev_swaits = out_from->rev_requests;

    out_from->bs = out_to->bs = bs; 
    tag     = out->tag;
    comm    = out->comm;

    /* Register the receives that you will use later (sends for scatter reverse) */
    for (i=0; i<out_from->n; i++) {
      ierr = MPI_Recv_init(Srvalues+bs*rstarts[i],bs*rstarts[i+1]-bs*rstarts[i],MPIU_SCALAR,rprocs[i],tag,comm,rwaits+i);CHKERRQ(ierr);
      ierr = MPI_Send_init(Srvalues+bs*rstarts[i],bs*rstarts[i+1]-bs*rstarts[i],MPIU_SCALAR,rprocs[i],tag,comm,rev_swaits+i);CHKERRQ(ierr);
    }

    ierr = PetscOptionsHasName(PETSC_NULL,"-vecscatter_rr",&flg);CHKERRQ(ierr);
    if (flg) {
      out_to->use_readyreceiver    = PETSC_TRUE;
      out_from->use_readyreceiver  = PETSC_TRUE;
      for (i=0; i<out_to->n; i++) {
        ierr = MPI_Rsend_init(Ssvalues+bs*sstarts[i],bs*sstarts[i+1]-bs*sstarts[i],MPIU_SCALAR,sprocs[i],tag,comm,swaits+i);CHKERRQ(ierr);
      } 
      if (out_from->n) {ierr = MPI_Startall_irecv(out_from->starts[out_from->n]*out_from->bs,out_from->n,out_from->requests);CHKERRQ(ierr);}
      ierr = MPI_Barrier(comm);CHKERRQ(ierr);
      ierr = PetscInfo(0,"Using VecScatter ready receiver mode\n");CHKERRQ(ierr);
    } else {
      out_to->use_readyreceiver    = PETSC_FALSE;
      out_from->use_readyreceiver  = PETSC_FALSE;
      flg                          = PETSC_FALSE;
      ierr                         = PetscOptionsHasName(PETSC_NULL,"-vecscatter_ssend",&flg);CHKERRQ(ierr);
      if (flg) {
        ierr = PetscInfo(0,"Using VecScatter Ssend mode\n");CHKERRQ(ierr);
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

/* --------------------------------------------------------------------------------------*/

#undef __FUNCT__  
#define __FUNCT__ "VecScatterBegin_PtoP_12"
PetscErrorCode VecScatterBegin_PtoP_12(Vec xin,Vec yin,InsertMode addv,ScatterMode mode,VecScatter ctx)
{
  VecScatter_MPI_General *to,*from;
  PetscScalar            *xv,*yv,*val,*svalues;
  MPI_Request            *rwaits,*swaits;
  PetscInt               *indices,*sstarts,iend,i,j,nrecvs,nsends,idx,len;
  PetscErrorCode         ierr;

  PetscFunctionBegin;
  ierr = VecGetArray(xin,&xv);CHKERRQ(ierr);
  if (xin != yin) {ierr = VecGetArray(yin,&yv);CHKERRQ(ierr);} else {yv = xv;}

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
  svalues  = to->values;
  nrecvs   = from->n;
  nsends   = to->n;
  indices  = to->indices;
  sstarts  = to->starts;

  if (!(mode & SCATTER_LOCAL)) {

    if (!from->use_readyreceiver && !to->sendfirst) {  
      /* post receives since they were not previously posted    */
      if (nrecvs) {ierr = MPI_Startall_irecv(from->starts[nrecvs]*from->bs,nrecvs,rwaits);CHKERRQ(ierr);}
    }
    if (ctx->packtogether) {
      /* this version packs all the messages together and sends, when -vecscatter_packtogether used */
      len  = sstarts[nsends];
      val  = svalues;
      for (i=0; i<len; i++) {
        idx     = *indices++;
        val[0]  = xv[idx];
        val[1]  = xv[idx+1];
        val[2]  = xv[idx+2];
        val[3]  = xv[idx+3];
	val[4]  = xv[idx+4];
	val[5]  = xv[idx+5];
	val[6]  = xv[idx+6];
	val[7]  = xv[idx+7];
	val[8]  = xv[idx+8];
	val[9]  = xv[idx+9];
	val[10] = xv[idx+10];
	val[11] = xv[idx+11];
        val    += 12;
      }
      if (nsends) {ierr = MPI_Startall_isend(len,nsends,swaits);CHKERRQ(ierr);}
    } else {
      /* this version packs and sends one at a time */
      val  = svalues;
      for (i=0; i<nsends; i++) {
        iend = sstarts[i+1]-sstarts[i];

        for (j=0; j<iend; j++) {
          idx     = *indices++;
	  val[0]  = xv[idx];
	  val[1]  = xv[idx+1];
	  val[2]  = xv[idx+2];
	  val[3]  = xv[idx+3];
	  val[4]  = xv[idx+4];
	  val[5]  = xv[idx+5];
	  val[6]  = xv[idx+6];
	  val[7]  = xv[idx+7];
	  val[8]  = xv[idx+8];
	  val[9]  = xv[idx+9];
	  val[10] = xv[idx+10];
	  val[11] = xv[idx+11];
          val    += 12;
        } 
        ierr = MPI_Start_isend(12*iend,swaits+i);CHKERRQ(ierr);
      } 
    }

    if (!from->use_readyreceiver && to->sendfirst) {  
      /* post receives since they were not previously posted   */
      if (nrecvs) {ierr = MPI_Startall_irecv(from->starts[nrecvs]*from->bs,nrecvs,rwaits);CHKERRQ(ierr);}
    }
  }

  /* take care of local scatters */
  if (to->local.n) {
    PetscInt *tslots = to->local.vslots,*fslots = from->local.vslots;
    PetscInt n       = to->local.n,il,ir;
    if (addv == INSERT_VALUES) {
      if (to->local.is_copy) {
        ierr = PetscMemcpy(yv+from->local.copy_start,xv+to->local.copy_start,to->local.copy_length);CHKERRQ(ierr);
      } else {
        for (i=0; i<n; i++) {
          il = fslots[i]; ir = tslots[i];
          yv[il]    = xv[ir];
          yv[il+1]  = xv[ir+1];
          yv[il+2]  = xv[ir+2];
          yv[il+3]  = xv[ir+3];
          yv[il+4]  = xv[ir+4];
          yv[il+5]  = xv[ir+5];
          yv[il+6]  = xv[ir+6];
          yv[il+7]  = xv[ir+7];
          yv[il+8]  = xv[ir+8];
          yv[il+9]  = xv[ir+9];
          yv[il+10] = xv[ir+10];
          yv[il+11] = xv[ir+11];
        }
      }
    }  else if (addv == ADD_VALUES) {
      for (i=0; i<n; i++) {
        il = fslots[i]; ir = tslots[i];
        yv[il]    += xv[ir];
        yv[il+1]  += xv[ir+1];
        yv[il+2]  += xv[ir+2];
        yv[il+3]  += xv[ir+3];
        yv[il+4]  += xv[ir+4];
        yv[il+5]  += xv[ir+5];
        yv[il+6]  += xv[ir+6];
        yv[il+7]  += xv[ir+7];
        yv[il+8]  += xv[ir+8];
        yv[il+9]  += xv[ir+9];
        yv[il+10] += xv[ir+10];
        yv[il+11] += xv[ir+11];
      }
#if !defined(PETSC_USE_COMPLEX)
    }  else if (addv == MAX_VALUES) {
      for (i=0; i<n; i++) {
        il = fslots[i]; ir = tslots[i];
        yv[il]    = PetscMax(yv[il],xv[ir]);
        yv[il+1]  = PetscMax(yv[il+1],xv[ir+1]);
        yv[il+2]  = PetscMax(yv[il+2],xv[ir+2]);
        yv[il+3]  = PetscMax(yv[il+3],xv[ir+3]);
        yv[il+4]  = PetscMax(yv[il+4],xv[ir+4]);
        yv[il+5]  = PetscMax(yv[il+5],xv[ir+5]);
        yv[il+6]  = PetscMax(yv[il+6],xv[ir+6]);
        yv[il+7]  = PetscMax(yv[il+7],xv[ir+7]);
        yv[il+8]  = PetscMax(yv[il+8],xv[ir+8]);
        yv[il+9]  = PetscMax(yv[il+9],xv[ir+9]);
        yv[il+10] = PetscMax(yv[il+10],xv[ir+10]);
        yv[il+11] = PetscMax(yv[il+11],xv[ir+11]);
      }
#endif
    } else {SETERRQ(PETSC_ERR_ARG_UNKNOWN_TYPE,"Wrong insert option");}
  }  
  ierr = VecRestoreArray(xin,&xv);CHKERRQ(ierr);
  if (xin != yin) {ierr = VecRestoreArray(yin,&yv);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

/* --------------------------------------------------------------------------------------*/

#undef __FUNCT__  
#define __FUNCT__ "VecScatterEnd_PtoP_12"
PetscErrorCode VecScatterEnd_PtoP_12(Vec xin,Vec yin,InsertMode addv,ScatterMode mode,VecScatter ctx)
{
  VecScatter_MPI_General *to,*from;
  PetscScalar            *rvalues,*yv,*val;
  PetscErrorCode         ierr;
  PetscMPIInt            imdex;
  PetscInt               nrecvs,nsends,i,*indices,count,n,*rstarts,idx,*lindices;
  MPI_Request            *rwaits,*swaits;
  MPI_Status             *rstatus,*sstatus,xrstatus;

  PetscFunctionBegin;
  if (mode & SCATTER_LOCAL) PetscFunctionReturn(0);
  ierr = VecGetArray(yin,&yv);CHKERRQ(ierr);

  to   = (VecScatter_MPI_General*)ctx->todata;
  from = (VecScatter_MPI_General*)ctx->fromdata;
  sstatus  = to->sstatus;
  rstatus  = to->rstatus;
  rwaits   = from->requests;
  swaits   = to->requests;
  if (mode & SCATTER_REVERSE) {
    to       = (VecScatter_MPI_General*)ctx->fromdata;
    from     = (VecScatter_MPI_General*)ctx->todata;
    rwaits   = from->rev_requests;
    swaits   = to->rev_requests;
  }
  rvalues  = from->values;
  nrecvs   = from->n;
  nsends   = to->n;
  indices  = from->indices;
  rstarts  = from->starts;

  /*  wait on receives */
  count = nrecvs;
  if (ctx->packtogether) { /* receive all messages, then unpack all, when -vecscatter_packtogether used */
    if (nrecvs) {ierr = MPI_Waitall(nrecvs,rwaits,rstatus);CHKERRQ(ierr);}
    n        = rstarts[count];
    val      = rvalues;
    if (addv == INSERT_VALUES) {
      for (i=0; i<n; i++) {
	idx        = indices[i];
	yv[idx]    = val[0];
	yv[idx+1]  = val[1];
	yv[idx+2]  = val[2];
	yv[idx+3]  = val[3];
	yv[idx+4]  = val[4];
	yv[idx+5]  = val[5];
	yv[idx+6]  = val[6];
	yv[idx+7]  = val[7];
	yv[idx+8]  = val[8];
	yv[idx+9]  = val[9];
	yv[idx+10] = val[10];
	yv[idx+11] = val[11];
	val       += 12;
      }
    } else if (addv == ADD_VALUES) {
      for (i=0; i<n; i++) {
	idx         = indices[i];
	yv[idx]    += val[0];
	yv[idx+1]  += val[1];
	yv[idx+2]  += val[2];
	yv[idx+3]  += val[3];
        yv[idx+4]  += val[4];
        yv[idx+5]  += val[5];
        yv[idx+6]  += val[6];
        yv[idx+7]  += val[7];
        yv[idx+8]  += val[8];
        yv[idx+9]  += val[9];
        yv[idx+10] += val[10];
        yv[idx+11] += val[11];
	val        += 12;
      }
#if !defined(PETSC_USE_COMPLEX)
    } else if (addv == MAX_VALUES) {
      for (i=0; i<n; i++) {
	idx        = indices[i];
	yv[idx]    = PetscMax(yv[idx],val[0]);
	yv[idx+1]  = PetscMax(yv[idx+1],val[1]);
	yv[idx+2]  = PetscMax(yv[idx+2],val[2]);
	yv[idx+3]  = PetscMax(yv[idx+3],val[3]);
        yv[idx+4]  = PetscMax(yv[idx+4],val[4]);
        yv[idx+5]  = PetscMax(yv[idx+5],val[5]);
        yv[idx+6]  = PetscMax(yv[idx+6],val[6]);
        yv[idx+7]  = PetscMax(yv[idx+7],val[7]);
        yv[idx+8]  = PetscMax(yv[idx+8],val[8]);
        yv[idx+9]  = PetscMax(yv[idx+9],val[9]);
        yv[idx+10] = PetscMax(yv[idx+10],val[10]);
        yv[idx+11] = PetscMax(yv[idx+11],val[11]);
	val       += 12;
      }
#endif
    }  else {SETERRQ(PETSC_ERR_ARG_UNKNOWN_TYPE,"Wrong insert option");}
  } else { /* unpack each message as it arrives, default version */
    while (count) {
      ierr = MPI_Waitany(nrecvs,rwaits,&imdex,&xrstatus);CHKERRQ(ierr);
      /* unpack receives into our local space */
      val      = rvalues + 12*rstarts[imdex];
      lindices = indices + rstarts[imdex];
      n        = rstarts[imdex+1] - rstarts[imdex];
      if (addv == INSERT_VALUES) {
	for (i=0; i<n; i++) {
	  idx        = lindices[i];
	  yv[idx]    = val[0];
	  yv[idx+1]  = val[1];
	  yv[idx+2]  = val[2];
	  yv[idx+3]  = val[3];
	  yv[idx+4]  = val[4];
	  yv[idx+5]  = val[5];
	  yv[idx+6]  = val[6];
	  yv[idx+7]  = val[7];
	  yv[idx+8]  = val[8];
	  yv[idx+9]  = val[9];
	  yv[idx+10] = val[10];
	  yv[idx+11] = val[11];
	  val       += 12;
      }
    } else if (addv == ADD_VALUES) {
      for (i=0; i<n; i++) {
        idx        = lindices[i];
        yv[idx]    += val[0];
        yv[idx+1]  += val[1];
        yv[idx+2]  += val[2];
        yv[idx+3]  += val[3];
        yv[idx+4]  += val[4];
        yv[idx+5]  += val[5];
        yv[idx+6]  += val[6];
        yv[idx+7]  += val[7];
        yv[idx+8]  += val[8];
        yv[idx+9]  += val[9];
        yv[idx+10] += val[10];
        yv[idx+11] += val[11];
        val        += 12;
      }
#if !defined(PETSC_USE_COMPLEX)
    } else if (addv == MAX_VALUES) {
      for (i=0; i<n; i++) {
        idx        = lindices[i];
        yv[idx]    = PetscMax(yv[idx],val[0]);
        yv[idx+1]  = PetscMax(yv[idx+1],val[1]);
        yv[idx+2]  = PetscMax(yv[idx+2],val[2]);
        yv[idx+3]  = PetscMax(yv[idx+3],val[3]);
        yv[idx+4]  = PetscMax(yv[idx+4],val[4]);
        yv[idx+5]  = PetscMax(yv[idx+5],val[5]);
        yv[idx+6]  = PetscMax(yv[idx+6],val[6]);
        yv[idx+7]  = PetscMax(yv[idx+7],val[7]);
        yv[idx+8]  = PetscMax(yv[idx+8],val[8]);
        yv[idx+9]  = PetscMax(yv[idx+9],val[9]);
        yv[idx+10] = PetscMax(yv[idx+10],val[10]);
        yv[idx+11] = PetscMax(yv[idx+11],val[11]);
        val        += 12;
      }
#endif
    }  else {SETERRQ(PETSC_ERR_ARG_UNKNOWN_TYPE,"Wrong insert option");}
    count--;
    }
  }
  if (from->use_readyreceiver) {  
    if (from->n) {ierr = MPI_Startall_irecv(from->starts[from->n]*from->bs,from->n,from->requests);CHKERRQ(ierr);}
    ierr = MPI_Barrier(ctx->comm);CHKERRQ(ierr);
  }

  /* wait on sends */
  if (nsends) {ierr = MPI_Waitall(nsends,swaits,sstatus);CHKERRQ(ierr);}
  ierr = VecRestoreArray(yin,&yv);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* --------------------------------------------------------------------------------------*/

#undef __FUNCT__  
#define __FUNCT__ "VecScatterBegin_PtoP_8"
PetscErrorCode VecScatterBegin_PtoP_8(Vec xin,Vec yin,InsertMode addv,ScatterMode mode,VecScatter ctx)
{
  VecScatter_MPI_General *to,*from;
  PetscScalar            *xv,*yv,*val,*svalues;
  MPI_Request            *rwaits,*swaits;
  PetscErrorCode         ierr;
  PetscInt               i,*indices,*sstarts,iend,j,nrecvs,nsends,idx,len;

  PetscFunctionBegin;
  ierr = VecGetArray(xin,&xv);CHKERRQ(ierr);
  if (xin != yin) {ierr = VecGetArray(yin,&yv);CHKERRQ(ierr);} else {yv = xv;}
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
  svalues  = to->values;
  nrecvs   = from->n;
  nsends   = to->n;
  indices  = to->indices;
  sstarts  = to->starts;

  if (!(mode & SCATTER_LOCAL)) {

    if (!from->use_readyreceiver && !to->sendfirst) {  
      /* post receives since they were not previously posted    */
      if (nrecvs) {ierr = MPI_Startall_irecv(from->starts[nrecvs]*from->bs,nrecvs,rwaits);CHKERRQ(ierr);}
    }

    if (ctx->packtogether) {
      /* this version packs all the messages together and sends, when -vecscatter_packtogether used */
      len  = sstarts[nsends];
      val  = svalues;
      for (i=0; i<len; i++) {
        idx     = *indices++;
        val[0]  = xv[idx];
        val[1]  = xv[idx+1];
        val[2]  = xv[idx+2];
        val[3]  = xv[idx+3];
	val[4]  = xv[idx+4];
	val[5]  = xv[idx+5];
	val[6]  = xv[idx+6];
	val[7]  = xv[idx+7];
        val    += 8;
      }
      if (nsends) {ierr = MPI_Startall_isend(len,nsends,swaits);CHKERRQ(ierr);}
    } else {
      /* this version packs and sends one at a time */
      val  = svalues;
      for (i=0; i<nsends; i++) {
        iend = sstarts[i+1]-sstarts[i];

        for (j=0; j<iend; j++) {
          idx     = *indices++;
          val[0] = xv[idx];
          val[1] = xv[idx+1];
          val[2] = xv[idx+2];
          val[3] = xv[idx+3];
          val[4] = xv[idx+4];
          val[5] = xv[idx+5];
          val[6] = xv[idx+6];
          val[7] = xv[idx+7];
          val    += 8;
        } 
        ierr = MPI_Start_isend(8*iend,swaits+i);CHKERRQ(ierr);
      }
    }

    if (!from->use_readyreceiver && to->sendfirst) {  
      /* post receives since they were not previously posted   */
      if (nrecvs) {ierr = MPI_Startall_irecv(from->starts[nrecvs]*from->bs,nrecvs,rwaits);CHKERRQ(ierr);}
    }
  }

  /* take care of local scatters */
  if (to->local.n) {
    PetscInt *tslots = to->local.vslots,*fslots = from->local.vslots;
    PetscInt n       = to->local.n,il,ir;
    if (addv == INSERT_VALUES) {
      if (to->local.is_copy) {
        ierr = PetscMemcpy(yv + from->local.copy_start,xv + to->local.copy_start,to->local.copy_length);CHKERRQ(ierr);
      } else {
        for (i=0; i<n; i++) {
          il = fslots[i]; ir = tslots[i];
          yv[il]   = xv[ir];
          yv[il+1] = xv[ir+1];
          yv[il+2] = xv[ir+2];
          yv[il+3] = xv[ir+3];
          yv[il+4] = xv[ir+4];
          yv[il+5] = xv[ir+5];
          yv[il+6] = xv[ir+6];
          yv[il+7] = xv[ir+7];
        }
      }
    }  else if (addv == ADD_VALUES) {
      for (i=0; i<n; i++) {
        il = fslots[i]; ir = tslots[i];
        yv[il]   += xv[ir];
        yv[il+1] += xv[ir+1];
        yv[il+2] += xv[ir+2];
        yv[il+3] += xv[ir+3];
        yv[il+4] += xv[ir+4];
        yv[il+5] += xv[ir+5];
        yv[il+6] += xv[ir+6];
        yv[il+7] += xv[ir+7];
      }
#if !defined(PETSC_USE_COMPLEX)
    }  else if (addv == MAX_VALUES) {
      for (i=0; i<n; i++) {
        il = fslots[i]; ir = tslots[i];
        yv[il]   = PetscMax(yv[il],xv[ir]);
        yv[il+1] = PetscMax(yv[il+1],xv[ir+1]);
        yv[il+2] = PetscMax(yv[il+2],xv[ir+2]);
        yv[il+3] = PetscMax(yv[il+3],xv[ir+3]);
        yv[il+4] = PetscMax(yv[il+4],xv[ir+4]);
        yv[il+5] = PetscMax(yv[il+5],xv[ir+5]);
        yv[il+6] = PetscMax(yv[il+6],xv[ir+6]);
        yv[il+7] = PetscMax(yv[il+7],xv[ir+7]);
      }
#endif
    }  else {SETERRQ(PETSC_ERR_ARG_UNKNOWN_TYPE,"Wrong insert option");}
  }
  ierr = VecRestoreArray(xin,&xv);CHKERRQ(ierr);
  if (xin != yin) {ierr = VecRestoreArray(yin,&yv);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

/* --------------------------------------------------------------------------------------*/

#undef __FUNCT__  
#define __FUNCT__ "VecScatterEnd_PtoP_8"
PetscErrorCode VecScatterEnd_PtoP_8(Vec xin,Vec yin,InsertMode addv,ScatterMode mode,VecScatter ctx)
{
  VecScatter_MPI_General *to,*from;
  PetscScalar            *rvalues,*yv,*val;
  PetscErrorCode         ierr;
  PetscInt               nrecvs,nsends,i,*indices,count,n,*rstarts,*lindices,idx;
  PetscMPIInt            imdex;
  MPI_Request            *rwaits,*swaits;
  MPI_Status             xrstatus,*rstatus,*sstatus;

  PetscFunctionBegin;
  if (mode & SCATTER_LOCAL) PetscFunctionReturn(0);
  ierr = VecGetArray(yin,&yv);CHKERRQ(ierr);

  to   = (VecScatter_MPI_General*)ctx->todata;
  from = (VecScatter_MPI_General*)ctx->fromdata;
  sstatus  = to->sstatus;
  rstatus  = to->rstatus;
  rwaits   = from->requests;
  swaits   = to->requests;
  if (mode & SCATTER_REVERSE) {
    to       = (VecScatter_MPI_General*)ctx->fromdata;
    from     = (VecScatter_MPI_General*)ctx->todata;
    rwaits   = from->rev_requests;
    swaits   = to->rev_requests;
  }
  rvalues  = from->values;
  nrecvs   = from->n;
  nsends   = to->n;
  indices  = from->indices;
  rstarts  = from->starts;

  /*  wait on receives */
  if (ctx->packtogether) { /* receive all messages, then unpack all, when -vecscatter_packtogether used */
    if (nrecvs) {ierr = MPI_Waitall(nrecvs,rwaits,rstatus);CHKERRQ(ierr);}
    n        = rstarts[count];
    val      = rvalues;
    if (addv == INSERT_VALUES) {
      for (i=0; i<n; i++) {
	idx        = indices[i];
	yv[idx]    = val[0];
	yv[idx+1]  = val[1];
	yv[idx+2]  = val[2];
	yv[idx+3]  = val[3];
	yv[idx+4]  = val[4];
	yv[idx+5]  = val[5];
	yv[idx+6]  = val[6];
	yv[idx+7]  = val[7];
	val       += 8;
      }
    } else if (addv == ADD_VALUES) {
      for (i=0; i<n; i++) {
	idx         = indices[i];
	yv[idx]    += val[0];
	yv[idx+1]  += val[1];
	yv[idx+2]  += val[2];
	yv[idx+3]  += val[3];
        yv[idx+4]  += val[4];
        yv[idx+5]  += val[5];
        yv[idx+6]  += val[6];
        yv[idx+7]  += val[7];
	val        += 8;
      }
#if !defined(PETSC_USE_COMPLEX)
    } else if (addv == MAX_VALUES) {
      for (i=0; i<n; i++) {
	idx        = indices[i];
	yv[idx]    = PetscMax(yv[idx],val[0]);
	yv[idx+1]  = PetscMax(yv[idx+1],val[1]);
	yv[idx+2]  = PetscMax(yv[idx+2],val[2]);
	yv[idx+3]  = PetscMax(yv[idx+3],val[3]);
        yv[idx+4]  = PetscMax(yv[idx+4],val[4]);
        yv[idx+5]  = PetscMax(yv[idx+5],val[5]);
        yv[idx+6]  = PetscMax(yv[idx+6],val[6]);
        yv[idx+7]  = PetscMax(yv[idx+7],val[7]);
	val       += 8;
      }
#endif
    }  else {SETERRQ(PETSC_ERR_ARG_UNKNOWN_TYPE,"Wrong insert option");}
  } else { /* unpack each message as it arrives, default version */
    count = nrecvs;
    while (count) {
      ierr = MPI_Waitany(nrecvs,rwaits,&imdex,&xrstatus);CHKERRQ(ierr);
      /* unpack receives into our local space */
      val      = rvalues + 8*rstarts[imdex];
      lindices = indices + rstarts[imdex];
      n        = rstarts[imdex+1] - rstarts[imdex];
      if (addv == INSERT_VALUES) {
        for (i=0; i<n; i++) {
          idx       = lindices[i];
	  yv[idx]   = val[0];
	  yv[idx+1] = val[1];
	  yv[idx+2] = val[2];
	  yv[idx+3] = val[3];
	  yv[idx+4] = val[4];
	  yv[idx+5] = val[5];
	  yv[idx+6] = val[6];
	  yv[idx+7] = val[7];
	  val      += 8;
	}
      } else if (addv == ADD_VALUES) {
	for (i=0; i<n; i++) {
	  idx       = lindices[i];
	  yv[idx]   += val[0];
	  yv[idx+1] += val[1];
	  yv[idx+2] += val[2];
	  yv[idx+3] += val[3];
	  yv[idx+4] += val[4];
	  yv[idx+5] += val[5];
	  yv[idx+6] += val[6];
	  yv[idx+7] += val[7];
	  val       += 8;
	}
#if !defined(PETSC_USE_COMPLEX)
      } else if (addv == MAX_VALUES) {
	for (i=0; i<n; i++) {
	  idx       = lindices[i];
	  yv[idx]   = PetscMax(yv[idx],val[0]);
	  yv[idx+1] = PetscMax(yv[idx+1],val[1]);
	  yv[idx+2] = PetscMax(yv[idx+2],val[2]);
	  yv[idx+3] = PetscMax(yv[idx+3],val[3]);
	  yv[idx+4] = PetscMax(yv[idx+4],val[4]);
	  yv[idx+5] = PetscMax(yv[idx+5],val[5]);
	  yv[idx+6] = PetscMax(yv[idx+6],val[6]);
	  yv[idx+7] = PetscMax(yv[idx+7],val[7]);
	  val       += 8;
	}
#endif
      }  else {SETERRQ(PETSC_ERR_ARG_UNKNOWN_TYPE,"Wrong insert option");}
      count--;
    }
  }

  if (from->use_readyreceiver) {  
    if (from->n) {ierr = MPI_Startall_irecv(from->starts[from->n]*from->bs,from->n,from->requests);CHKERRQ(ierr);}
    ierr = MPI_Barrier(ctx->comm);CHKERRQ(ierr);
  }

  /* wait on sends */
  if (nsends) {ierr = MPI_Waitall(nsends,swaits,sstatus);CHKERRQ(ierr);}
  ierr = VecRestoreArray(yin,&yv);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
/* --------------------------------------------------------------------------------------*/

#undef __FUNCT__  
#define __FUNCT__ "VecScatterBegin_PtoP_7"
PetscErrorCode VecScatterBegin_PtoP_7(Vec xin,Vec yin,InsertMode addv,ScatterMode mode,VecScatter ctx)
{
  VecScatter_MPI_General *to,*from;
  PetscScalar            *xv,*yv,*val,*svalues;
  MPI_Request            *rwaits,*swaits;
  PetscErrorCode         ierr;
  PetscInt               i,*indices,*sstarts,iend,j,nrecvs,nsends,idx,len;

  PetscFunctionBegin;
  ierr = VecGetArray(xin,&xv);CHKERRQ(ierr);
  if (xin != yin) {ierr = VecGetArray(yin,&yv);CHKERRQ(ierr);} else {yv = xv;}
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
  svalues  = to->values;
  nrecvs   = from->n;
  nsends   = to->n;
  indices  = to->indices;
  sstarts  = to->starts;

  if (!(mode & SCATTER_LOCAL)) {

    if (!from->use_readyreceiver && !to->sendfirst) {  
      /* post receives since they were not previously posted    */
      if (nrecvs) {ierr = MPI_Startall_irecv(from->starts[nrecvs]*from->bs,nrecvs,rwaits);CHKERRQ(ierr);}
    }

    if (ctx->packtogether) {
      /* this version packs all the messages together and sends, when -vecscatter_packtogether used */
      len  = sstarts[nsends];
      val  = svalues;
      for (i=0; i<len; i++) {
        idx     = *indices++;
        val[0]  = xv[idx];
        val[1]  = xv[idx+1];
        val[2]  = xv[idx+2];
        val[3]  = xv[idx+3];
	val[4]  = xv[idx+4];
	val[5]  = xv[idx+5];
	val[6]  = xv[idx+6];
        val    += 7;
      }
      if (nsends) {ierr = MPI_Startall_isend(len,nsends,swaits);CHKERRQ(ierr);}
    } else {
      /* this version packs and sends one at a time */
      val  = svalues;
      for (i=0; i<nsends; i++) {
        iend = sstarts[i+1]-sstarts[i];

        for (j=0; j<iend; j++) {
          idx     = *indices++;
          val[0] = xv[idx];
          val[1] = xv[idx+1];
          val[2] = xv[idx+2];
          val[3] = xv[idx+3];
          val[4] = xv[idx+4];
          val[5] = xv[idx+5];
          val[6] = xv[idx+6];
          val    += 7;
        } 
        ierr = MPI_Start_isend(7*iend,swaits+i);CHKERRQ(ierr);
      }
    }

    if (!from->use_readyreceiver && to->sendfirst) {  
      /* post receives since they were not previously posted   */
      if (nrecvs) {ierr = MPI_Startall_irecv(from->starts[nrecvs]*from->bs,nrecvs,rwaits);CHKERRQ(ierr);}
    }
  }

  /* take care of local scatters */
  if (to->local.n) {
    PetscInt *tslots = to->local.vslots,*fslots = from->local.vslots;
    PetscInt n       = to->local.n,il,ir;
    if (addv == INSERT_VALUES) {
      if (to->local.is_copy) {
        ierr = PetscMemcpy(yv + from->local.copy_start,xv + to->local.copy_start,to->local.copy_length);CHKERRQ(ierr);
      } else {
        for (i=0; i<n; i++) {
          il = fslots[i]; ir = tslots[i];
          yv[il]   = xv[ir];
          yv[il+1] = xv[ir+1];
          yv[il+2] = xv[ir+2];
          yv[il+3] = xv[ir+3];
          yv[il+4] = xv[ir+4];
          yv[il+5] = xv[ir+5];
          yv[il+6] = xv[ir+6];
        }
      }
    }  else if (addv == ADD_VALUES) {
      for (i=0; i<n; i++) {
        il = fslots[i]; ir = tslots[i];
        yv[il]   += xv[ir];
        yv[il+1] += xv[ir+1];
        yv[il+2] += xv[ir+2];
        yv[il+3] += xv[ir+3];
        yv[il+4] += xv[ir+4];
        yv[il+5] += xv[ir+5];
        yv[il+6] += xv[ir+6];
      }
#if !defined(PETSC_USE_COMPLEX)
    }  else if (addv == MAX_VALUES) {
      for (i=0; i<n; i++) {
        il = fslots[i]; ir = tslots[i];
        yv[il]   = PetscMax(yv[il],xv[ir]);
        yv[il+1] = PetscMax(yv[il+1],xv[ir+1]);
        yv[il+2] = PetscMax(yv[il+2],xv[ir+2]);
        yv[il+3] = PetscMax(yv[il+3],xv[ir+3]);
        yv[il+4] = PetscMax(yv[il+4],xv[ir+4]);
        yv[il+5] = PetscMax(yv[il+5],xv[ir+5]);
        yv[il+6] = PetscMax(yv[il+6],xv[ir+6]);
      }
#endif
    }  else {SETERRQ(PETSC_ERR_ARG_UNKNOWN_TYPE,"Wrong insert option");}
  }
  ierr = VecRestoreArray(xin,&xv);CHKERRQ(ierr);
  if (xin != yin) {ierr = VecRestoreArray(yin,&yv);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

/* --------------------------------------------------------------------------------------*/

#undef __FUNCT__  
#define __FUNCT__ "VecScatterEnd_PtoP_7"
PetscErrorCode VecScatterEnd_PtoP_7(Vec xin,Vec yin,InsertMode addv,ScatterMode mode,VecScatter ctx)
{
  VecScatter_MPI_General *to,*from;
  PetscScalar            *rvalues,*yv,*val;
  PetscErrorCode         ierr;
  PetscInt               nrecvs,nsends,i,*indices,count,n,*rstarts,*lindices,idx;
  PetscMPIInt            imdex;
  MPI_Request            *rwaits,*swaits;
  MPI_Status             xrstatus,*rstatus,*sstatus;

  PetscFunctionBegin;
  if (mode & SCATTER_LOCAL) PetscFunctionReturn(0);
  ierr = VecGetArray(yin,&yv);CHKERRQ(ierr);

  to       = (VecScatter_MPI_General*)ctx->todata;
  from     = (VecScatter_MPI_General*)ctx->fromdata;
  sstatus  = to->sstatus;
  rstatus  = to->rstatus;
  rwaits   = from->requests;
  swaits   = to->requests;
  if (mode & SCATTER_REVERSE) {
    to       = (VecScatter_MPI_General*)ctx->fromdata;
    from     = (VecScatter_MPI_General*)ctx->todata;
    rwaits   = from->rev_requests;
    swaits   = to->rev_requests;
  } 
  rvalues  = from->values;
  nrecvs   = from->n;
  nsends   = to->n;
  indices  = from->indices;
  rstarts  = from->starts;

  /*  wait on receives */
  if (ctx->packtogether) { /* receive all messages, then unpack all, when -vecscatter_packtogether used */
    if (nrecvs) {ierr = MPI_Waitall(nrecvs,rwaits,rstatus);CHKERRQ(ierr);}
    n        = rstarts[count];
    val      = rvalues;
    if (addv == INSERT_VALUES) {
      for (i=0; i<n; i++) {
	idx        = indices[i];
	yv[idx]    = val[0];
	yv[idx+1]  = val[1];
	yv[idx+2]  = val[2];
	yv[idx+3]  = val[3];
	yv[idx+4]  = val[4];
	yv[idx+5]  = val[5];
	yv[idx+6]  = val[6];
	val       += 7;
      }
    } else if (addv == ADD_VALUES) {
      for (i=0; i<n; i++) {
	idx         = indices[i];
	yv[idx]    += val[0];
	yv[idx+1]  += val[1];
	yv[idx+2]  += val[2];
	yv[idx+3]  += val[3];
        yv[idx+4]  += val[4];
        yv[idx+5]  += val[5];
        yv[idx+6]  += val[6];
	val        += 7;
      }
#if !defined(PETSC_USE_COMPLEX)
    } else if (addv == MAX_VALUES) {
      for (i=0; i<n; i++) {
	idx        = indices[i];
	yv[idx]    = PetscMax(yv[idx],val[0]);
	yv[idx+1]  = PetscMax(yv[idx+1],val[1]);
	yv[idx+2]  = PetscMax(yv[idx+2],val[2]);
	yv[idx+3]  = PetscMax(yv[idx+3],val[3]);
        yv[idx+4]  = PetscMax(yv[idx+4],val[4]);
        yv[idx+5]  = PetscMax(yv[idx+5],val[5]);
        yv[idx+6]  = PetscMax(yv[idx+6],val[6]);
	val       += 7;
      }
#endif
    }  else {SETERRQ(PETSC_ERR_ARG_UNKNOWN_TYPE,"Wrong insert option");}
  } else { /* unpack each message as it arrives, default version */
    count = nrecvs;
    while (count) {
      ierr = MPI_Waitany(nrecvs,rwaits,&imdex,&xrstatus);CHKERRQ(ierr);
      /* unpack receives into our local space */
      val      = rvalues + 7*rstarts[imdex];
      lindices = indices + rstarts[imdex];
      n        = rstarts[imdex+1] - rstarts[imdex];
      if (addv == INSERT_VALUES) {
        for (i=0; i<n; i++) {
          idx       = lindices[i];
          yv[idx]   = val[0];
          yv[idx+1] = val[1];
          yv[idx+2] = val[2];
          yv[idx+3] = val[3];
          yv[idx+4] = val[4];
          yv[idx+5] = val[5];
          yv[idx+6] = val[6];
          val      += 7;
        }
      } else if (addv == ADD_VALUES) {
        for (i=0; i<n; i++) {
          idx       = lindices[i];
          yv[idx]   += val[0];
          yv[idx+1] += val[1];
          yv[idx+2] += val[2];
          yv[idx+3] += val[3];
          yv[idx+4] += val[4];
          yv[idx+5] += val[5];
          yv[idx+6] += val[6];
          val       += 7;
        }
#if !defined(PETSC_USE_COMPLEX)
      } else if (addv == MAX_VALUES) {
        for (i=0; i<n; i++) {
          idx       = lindices[i];
          yv[idx]   = PetscMax(yv[idx],val[0]);
          yv[idx+1] = PetscMax(yv[idx+1],val[1]);
          yv[idx+2] = PetscMax(yv[idx+2],val[2]);
          yv[idx+3] = PetscMax(yv[idx+3],val[3]);
          yv[idx+4] = PetscMax(yv[idx+4],val[4]);
          yv[idx+5] = PetscMax(yv[idx+5],val[5]);
          yv[idx+6] = PetscMax(yv[idx+6],val[6]);
          val       += 7;
        }
#endif
      }  else {SETERRQ(PETSC_ERR_ARG_UNKNOWN_TYPE,"Wrong insert option");}
      count--;
    }
  }
  if (from->use_readyreceiver) {  
    if (from->n) {ierr = MPI_Startall_irecv(from->starts[from->n]*from->bs,from->n,from->requests);CHKERRQ(ierr);}
    ierr = MPI_Barrier(ctx->comm);CHKERRQ(ierr);
  }

  /* wait on sends */
  if (nsends) {ierr = MPI_Waitall(nsends,swaits,sstatus);CHKERRQ(ierr);}
  ierr = VecRestoreArray(yin,&yv);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* --------------------------------------------------------------------------------------*/

#undef __FUNCT__  
#define __FUNCT__ "VecScatterBegin_PtoP_6"
PetscErrorCode VecScatterBegin_PtoP_6(Vec xin,Vec yin,InsertMode addv,ScatterMode mode,VecScatter ctx)
{
  VecScatter_MPI_General *to,*from;
  PetscScalar            *xv,*yv,*val,*svalues;
  MPI_Request            *rwaits,*swaits;
  PetscErrorCode         ierr;
  PetscInt               i,*indices,*sstarts,iend,j,nrecvs,nsends,idx,len;

  PetscFunctionBegin;
  ierr = VecGetArray(xin,&xv);CHKERRQ(ierr);
  if (xin != yin) {ierr = VecGetArray(yin,&yv);CHKERRQ(ierr);} else {yv = xv;}
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
  svalues  = to->values;
  nrecvs   = from->n;
  nsends   = to->n;
  indices  = to->indices;
  sstarts  = to->starts;

  if (!(mode & SCATTER_LOCAL)) {

    if (!from->use_readyreceiver && !to->sendfirst) {  
      /* post receives since they were not previously posted    */
      if (nrecvs) {ierr = MPI_Startall_irecv(from->starts[nrecvs]*from->bs,nrecvs,rwaits);CHKERRQ(ierr);}
    }

    if (ctx->packtogether) {
      /* this version packs all the messages together and sends, when -vecscatter_packtogether used */
      len  = sstarts[nsends];
      val  = svalues;
      for (i=0; i<len; i++) {
        idx     = *indices++;
        val[0]  = xv[idx];
        val[1]  = xv[idx+1];
        val[2]  = xv[idx+2];
        val[3]  = xv[idx+3];
	val[4]  = xv[idx+4];
	val[5]  = xv[idx+5];
        val    += 6;
      }
      if (nsends) {ierr = MPI_Startall_isend(len,nsends,swaits);CHKERRQ(ierr);}
    } else {
      /* this version packs and sends one at a time */
      val  = svalues;
      for (i=0; i<nsends; i++) {
        iend = sstarts[i+1]-sstarts[i];

        for (j=0; j<iend; j++) {
          idx     = *indices++;
          val[0] = xv[idx];
          val[1] = xv[idx+1];
          val[2] = xv[idx+2];
          val[3] = xv[idx+3];
          val[4] = xv[idx+4];
          val[5] = xv[idx+5];
          val    += 6;
        } 
        ierr = MPI_Start_isend(6*iend,swaits+i);CHKERRQ(ierr);
      }
    }

    if (!from->use_readyreceiver && to->sendfirst) {  
      /* post receives since they were not previously posted   */
      if (nrecvs) {ierr = MPI_Startall_irecv(from->starts[nrecvs]*from->bs,nrecvs,rwaits);CHKERRQ(ierr);}
    }
  }

  /* take care of local scatters */
  if (to->local.n) {
    PetscInt *tslots = to->local.vslots,*fslots = from->local.vslots;
    PetscInt n       = to->local.n,il,ir;
    if (addv == INSERT_VALUES) {
      if (to->local.is_copy) {
        ierr = PetscMemcpy(yv + from->local.copy_start,xv + to->local.copy_start,to->local.copy_length);CHKERRQ(ierr);
      } else {
        for (i=0; i<n; i++) {
          il = fslots[i]; ir = tslots[i];
          yv[il]   = xv[ir];
          yv[il+1] = xv[ir+1];
          yv[il+2] = xv[ir+2];
          yv[il+3] = xv[ir+3];
          yv[il+4] = xv[ir+4];
          yv[il+5] = xv[ir+5];
        }
      }
    }  else if (addv == ADD_VALUES) {
      for (i=0; i<n; i++) {
        il = fslots[i]; ir = tslots[i];
        yv[il]   += xv[ir];
        yv[il+1] += xv[ir+1];
        yv[il+2] += xv[ir+2];
        yv[il+3] += xv[ir+3];
        yv[il+4] += xv[ir+4];
        yv[il+5] += xv[ir+5];
      }
#if !defined(PETSC_USE_COMPLEX)
    }  else if (addv == MAX_VALUES) {
      for (i=0; i<n; i++) {
        il = fslots[i]; ir = tslots[i];
        yv[il]   = PetscMax(yv[il],xv[ir]);
        yv[il+1] = PetscMax(yv[il+1],xv[ir+1]);
        yv[il+2] = PetscMax(yv[il+2],xv[ir+2]);
        yv[il+3] = PetscMax(yv[il+3],xv[ir+3]);
        yv[il+4] = PetscMax(yv[il+4],xv[ir+4]);
        yv[il+5] = PetscMax(yv[il+5],xv[ir+5]);
      }
#endif
    }  else {SETERRQ(PETSC_ERR_ARG_UNKNOWN_TYPE,"Wrong insert option");}
  }
  ierr = VecRestoreArray(xin,&xv);CHKERRQ(ierr);
  if (xin != yin) {ierr = VecRestoreArray(yin,&yv);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

/* --------------------------------------------------------------------------------------*/

#undef __FUNCT__  
#define __FUNCT__ "VecScatterEnd_PtoP_6"
PetscErrorCode VecScatterEnd_PtoP_6(Vec xin,Vec yin,InsertMode addv,ScatterMode mode,VecScatter ctx)
{
  VecScatter_MPI_General *to,*from;
  PetscScalar            *rvalues,*yv,*val;
  PetscErrorCode         ierr;
  PetscInt               nrecvs,nsends,i,*indices,count,n,*rstarts,*lindices,idx;
  PetscMPIInt            imdex;
  MPI_Request            *rwaits,*swaits;
  MPI_Status             xrstatus,*rstatus,*sstatus;

  PetscFunctionBegin;
  if (mode & SCATTER_LOCAL) PetscFunctionReturn(0);
  ierr = VecGetArray(yin,&yv);CHKERRQ(ierr);

  to       = (VecScatter_MPI_General*)ctx->todata;
  from     = (VecScatter_MPI_General*)ctx->fromdata;
  sstatus  = to->sstatus;
  rstatus  = to->rstatus;
  rwaits   = from->requests;
  swaits   = to->requests;
  if (mode & SCATTER_REVERSE) {
    to     = (VecScatter_MPI_General*)ctx->fromdata;
    from   = (VecScatter_MPI_General*)ctx->todata;
    rwaits = from->rev_requests;
    swaits = to->rev_requests;
  }
  rvalues  = from->values;
  nrecvs   = from->n;
  nsends   = to->n;
  indices  = from->indices;
  rstarts  = from->starts;

  /*  wait on receives */
  if (ctx->packtogether) { /* receive all messages, then unpack all, when -vecscatter_packtogether used */
    if (nrecvs) {ierr = MPI_Waitall(nrecvs,rwaits,rstatus);CHKERRQ(ierr);}
    n        = rstarts[count];
    val      = rvalues;
    if (addv == INSERT_VALUES) {
      for (i=0; i<n; i++) {
	idx        = indices[i];
	yv[idx]    = val[0];
	yv[idx+1]  = val[1];
	yv[idx+2]  = val[2];
	yv[idx+3]  = val[3];
	yv[idx+4]  = val[4];
	yv[idx+5]  = val[5];
	val       += 6;
      }
    } else if (addv == ADD_VALUES) {
      for (i=0; i<n; i++) {
	idx         = indices[i];
	yv[idx]    += val[0];
	yv[idx+1]  += val[1];
	yv[idx+2]  += val[2];
	yv[idx+3]  += val[3];
        yv[idx+4]  += val[4];
        yv[idx+5]  += val[5];
	val        += 6;
      }
#if !defined(PETSC_USE_COMPLEX)
    } else if (addv == MAX_VALUES) {
      for (i=0; i<n; i++) {
	idx        = indices[i];
	yv[idx]    = PetscMax(yv[idx],val[0]);
	yv[idx+1]  = PetscMax(yv[idx+1],val[1]);
	yv[idx+2]  = PetscMax(yv[idx+2],val[2]);
	yv[idx+3]  = PetscMax(yv[idx+3],val[3]);
        yv[idx+4]  = PetscMax(yv[idx+4],val[4]);
        yv[idx+5]  = PetscMax(yv[idx+5],val[5]);
	val       += 6;
      }
#endif
    }  else {SETERRQ(PETSC_ERR_ARG_UNKNOWN_TYPE,"Wrong insert option");}
  } else { /* unpack each message as it arrives, default version */
    count = nrecvs;
    while (count) {
      ierr = MPI_Waitany(nrecvs,rwaits,&imdex,&xrstatus);CHKERRQ(ierr);
      /* unpack receives into our local space */
      val      = rvalues + 6*rstarts[imdex];
      lindices = indices + rstarts[imdex];
      n        = rstarts[imdex+1] - rstarts[imdex];
      if (addv == INSERT_VALUES) {
        for (i=0; i<n; i++) {
          idx       = lindices[i];
          yv[idx]   = val[0];
          yv[idx+1] = val[1];
          yv[idx+2] = val[2];
          yv[idx+3] = val[3];
          yv[idx+4] = val[4];
          yv[idx+5] = val[5];
          val      += 6;
        }
      } else if (addv == ADD_VALUES) {
        for (i=0; i<n; i++) {
          idx       = lindices[i];
          yv[idx]   += val[0];
          yv[idx+1] += val[1];
          yv[idx+2] += val[2];
          yv[idx+3] += val[3];
          yv[idx+4] += val[4];
          yv[idx+5] += val[5];
          val       += 6;
        }
#if !defined(PETSC_USE_COMPLEX)
      } else if (addv == MAX_VALUES) {
        for (i=0; i<n; i++) {
          idx       = lindices[i];
          yv[idx]   = PetscMax(yv[idx],val[0]);
          yv[idx+1] = PetscMax(yv[idx+1],val[1]);
          yv[idx+2] = PetscMax(yv[idx+2],val[2]);
          yv[idx+3] = PetscMax(yv[idx+3],val[3]);
          yv[idx+4] = PetscMax(yv[idx+4],val[4]);
          yv[idx+5] = PetscMax(yv[idx+5],val[5]);
          val       += 6;
        }
#endif
      }  else {SETERRQ(PETSC_ERR_ARG_UNKNOWN_TYPE,"Wrong insert option");}
      count--;
    }
  }
  if (from->use_readyreceiver) {  
    if (from->n) {ierr = MPI_Startall_irecv(from->starts[from->n]*from->bs,from->n,from->requests);CHKERRQ(ierr);}
    ierr = MPI_Barrier(ctx->comm);CHKERRQ(ierr);
  }

  /* wait on sends */
  if (nsends) {ierr = MPI_Waitall(nsends,swaits,sstatus);CHKERRQ(ierr);}
  ierr = VecRestoreArray(yin,&yv);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* --------------------------------------------------------------------------------------*/

#undef __FUNCT__  
#define __FUNCT__ "VecScatterBegin_PtoP_5"
PetscErrorCode VecScatterBegin_PtoP_5(Vec xin,Vec yin,InsertMode addv,ScatterMode mode,VecScatter ctx)
{
  VecScatter_MPI_General *to,*from;
  PetscScalar            *xv,*yv,*val,*svalues;
  MPI_Request            *rwaits,*swaits;
  PetscErrorCode         ierr;
  PetscInt               i,*indices,*sstarts,iend,j,nrecvs,nsends,idx,len;

  PetscFunctionBegin;
  ierr = VecGetArray(xin,&xv);CHKERRQ(ierr);
  if (xin != yin) {ierr = VecGetArray(yin,&yv);CHKERRQ(ierr);} else {yv = xv;}
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
  svalues  = to->values;
  nrecvs   = from->n;
  nsends   = to->n;
  indices  = to->indices;
  sstarts  = to->starts;

  if (!(mode & SCATTER_LOCAL)) {

    if (!from->use_readyreceiver && !to->sendfirst) {  
      /* post receives since they were not previously posted    */
      if (nrecvs) {ierr = MPI_Startall_irecv(from->starts[nrecvs]*from->bs,nrecvs,rwaits);CHKERRQ(ierr);}
    }

    if (ctx->packtogether) {
      /* this version packs all the messages together and sends, when -vecscatter_packtogether used */
      len  = sstarts[nsends];
      val  = svalues;
      for (i=0; i<len; i++) {
        idx     = *indices++;
        val[0]  = xv[idx];
        val[1]  = xv[idx+1];
        val[2]  = xv[idx+2];
        val[3]  = xv[idx+3];
	val[4]  = xv[idx+4];
	val[5]  = xv[idx+5];
        val    += 5;
      }
      if (nsends) {ierr = MPI_Startall_isend(len,nsends,swaits);CHKERRQ(ierr);}
    } else {
      /* this version packs and sends one at a time */
      val  = svalues;
      for (i=0; i<nsends; i++) {
        iend = sstarts[i+1]-sstarts[i];

        for (j=0; j<iend; j++) {
          idx     = *indices++;
          val[0] = xv[idx];
          val[1] = xv[idx+1];
          val[2] = xv[idx+2];
          val[3] = xv[idx+3];
          val[4] = xv[idx+4];
          val    += 5;
        } 
        ierr = MPI_Start_isend(5*iend,swaits+i);CHKERRQ(ierr);
      }
    }

    if (!from->use_readyreceiver && to->sendfirst) {  
      /* post receives since they were not previously posted   */
      if (nrecvs) {ierr = MPI_Startall_irecv(from->starts[nrecvs]*from->bs,nrecvs,rwaits);CHKERRQ(ierr);}
    }
  }

  /* take care of local scatters */
  if (to->local.n) {
    PetscInt *tslots = to->local.vslots,*fslots = from->local.vslots;
    PetscInt n       = to->local.n,il,ir;
    if (addv == INSERT_VALUES) {
      if (to->local.is_copy) {
        ierr = PetscMemcpy(yv + from->local.copy_start,xv + to->local.copy_start,to->local.copy_length);CHKERRQ(ierr);
      } else {
        for (i=0; i<n; i++) {
          il = fslots[i]; ir = tslots[i];
          yv[il]   = xv[ir];
          yv[il+1] = xv[ir+1];
          yv[il+2] = xv[ir+2];
          yv[il+3] = xv[ir+3];
          yv[il+4] = xv[ir+4];
        }
      }
    }  else if (addv == ADD_VALUES) {
      for (i=0; i<n; i++) {
        il = fslots[i]; ir = tslots[i];
        yv[il]   += xv[ir];
        yv[il+1] += xv[ir+1];
        yv[il+2] += xv[ir+2];
        yv[il+3] += xv[ir+3];
        yv[il+4] += xv[ir+4];
      }
#if !defined(PETSC_USE_COMPLEX)
    }  else if (addv == MAX_VALUES) {
      for (i=0; i<n; i++) {
        il = fslots[i]; ir = tslots[i];
        yv[il]   = PetscMax(yv[il],xv[ir]);
        yv[il+1] = PetscMax(yv[il+1],xv[ir+1]);
        yv[il+2] = PetscMax(yv[il+2],xv[ir+2]);
        yv[il+3] = PetscMax(yv[il+3],xv[ir+3]);
        yv[il+4] = PetscMax(yv[il+4],xv[ir+4]);
      }
#endif
    }  else {SETERRQ(PETSC_ERR_ARG_UNKNOWN_TYPE,"Wrong insert option");}
  }
  ierr = VecRestoreArray(xin,&xv);CHKERRQ(ierr);
  if (xin != yin) {ierr = VecRestoreArray(yin,&yv);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

/* --------------------------------------------------------------------------------------*/

#undef __FUNCT__  
#define __FUNCT__ "VecScatterEnd_PtoP_5"
PetscErrorCode VecScatterEnd_PtoP_5(Vec xin,Vec yin,InsertMode addv,ScatterMode mode,VecScatter ctx)
{
  VecScatter_MPI_General *to,*from;
  PetscScalar            *rvalues,*yv,*val;
  PetscErrorCode         ierr;
  PetscInt               nrecvs,nsends,i,*indices,count,n,*rstarts,*lindices,idx;
  PetscMPIInt            imdex;
  MPI_Request            *rwaits,*swaits;
  MPI_Status             xrstatus,*rstatus,*sstatus;

  PetscFunctionBegin;
  if (mode & SCATTER_LOCAL) PetscFunctionReturn(0);
  ierr = VecGetArray(yin,&yv);CHKERRQ(ierr);

  to       = (VecScatter_MPI_General*)ctx->todata;
  from     = (VecScatter_MPI_General*)ctx->fromdata;
  sstatus  = to->sstatus;
  rstatus  = to->rstatus;
  rwaits   = from->requests;
  swaits   = to->requests;
  if (mode & SCATTER_REVERSE) {
    to      = (VecScatter_MPI_General*)ctx->fromdata;
    from    = (VecScatter_MPI_General*)ctx->todata;
    rwaits  = from->rev_requests;
    swaits  = to->rev_requests;
  }
  rvalues  = from->values;
  nrecvs   = from->n;
  nsends   = to->n;
  indices  = from->indices;
  rstarts  = from->starts;

  /*  wait on receives */
  if (ctx->packtogether) { /* receive all messages, then unpack all, when -vecscatter_packtogether used */
    if (nrecvs) {ierr = MPI_Waitall(nrecvs,rwaits,rstatus);CHKERRQ(ierr);}
    n        = rstarts[count];
    val      = rvalues;
    if (addv == INSERT_VALUES) {
      for (i=0; i<n; i++) {
	idx        = indices[i];
	yv[idx]    = val[0];
	yv[idx+1]  = val[1];
	yv[idx+2]  = val[2];
	yv[idx+3]  = val[3];
	yv[idx+4]  = val[4];
	val       += 5;
      }
    } else if (addv == ADD_VALUES) {
      for (i=0; i<n; i++) {
	idx         = indices[i];
	yv[idx]    += val[0];
	yv[idx+1]  += val[1];
	yv[idx+2]  += val[2];
	yv[idx+3]  += val[3];
        yv[idx+4]  += val[4];
	val        += 5;
      }
#if !defined(PETSC_USE_COMPLEX)
    } else if (addv == MAX_VALUES) {
      for (i=0; i<n; i++) {
	idx        = indices[i];
	yv[idx]    = PetscMax(yv[idx],val[0]);
	yv[idx+1]  = PetscMax(yv[idx+1],val[1]);
	yv[idx+2]  = PetscMax(yv[idx+2],val[2]);
	yv[idx+3]  = PetscMax(yv[idx+3],val[3]);
        yv[idx+4]  = PetscMax(yv[idx+4],val[4]);
	val       += 5;
      }
#endif
    }  else {SETERRQ(PETSC_ERR_ARG_UNKNOWN_TYPE,"Wrong insert option");}
  } else { /* unpack each message as it arrives, default version */
    count = nrecvs;
    while (count) {
      ierr = MPI_Waitany(nrecvs,rwaits,&imdex,&xrstatus);CHKERRQ(ierr);
      /* unpack receives into our local space */
      val      = rvalues + 5*rstarts[imdex];
      lindices = indices + rstarts[imdex];
      n        = rstarts[imdex+1] - rstarts[imdex];
      if (addv == INSERT_VALUES) {
	for (i=0; i<n; i++) {
	  idx       = lindices[i];
	  yv[idx]   = val[0];
	  yv[idx+1] = val[1];
	  yv[idx+2] = val[2];
	  yv[idx+3] = val[3];
	  yv[idx+4] = val[4];
	  val      += 5;
	}
      } else if (addv == ADD_VALUES) {
	for (i=0; i<n; i++) {
	  idx       = lindices[i];
	  yv[idx]   += val[0];
	  yv[idx+1] += val[1];
	  yv[idx+2] += val[2];
	  yv[idx+3] += val[3];
	  yv[idx+4] += val[4];
	  val       += 5;
	}
#if !defined(PETSC_USE_COMPLEX)
      } else if (addv == MAX_VALUES) {
	for (i=0; i<n; i++) {
	  idx       = lindices[i];
	  yv[idx]   = PetscMax(yv[idx],val[0]);
	  yv[idx+1] = PetscMax(yv[idx+1],val[1]);
	  yv[idx+2] = PetscMax(yv[idx+2],val[2]);
	  yv[idx+3] = PetscMax(yv[idx+3],val[3]);
	  yv[idx+4] = PetscMax(yv[idx+4],val[4]);
	  val       += 5;
	}
#endif
      }  else {SETERRQ(PETSC_ERR_ARG_UNKNOWN_TYPE,"Wrong insert option");}
      count--;
    }
  }
  if (from->use_readyreceiver) {  
    if (from->n) {ierr = MPI_Startall_irecv(from->starts[from->n]*from->bs,from->n,from->requests);CHKERRQ(ierr);}
    ierr = MPI_Barrier(ctx->comm);CHKERRQ(ierr);
  }

  /* wait on sends */
  if (nsends) {ierr = MPI_Waitall(nsends,swaits,sstatus);CHKERRQ(ierr);}
  ierr = VecRestoreArray(yin,&yv);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* --------------------------------------------------------------------------------------*/

#undef __FUNCT__  
#define __FUNCT__ "VecScatterBegin_PtoP_4"
PetscErrorCode VecScatterBegin_PtoP_4(Vec xin,Vec yin,InsertMode addv,ScatterMode mode,VecScatter ctx)
{
  VecScatter_MPI_General *to,*from;
  PetscScalar            *xv,*yv,*val,*svalues;
  MPI_Request            *rwaits,*swaits;
  PetscInt               *indices,*sstarts,iend,i,j,nrecvs,nsends,idx,len;
  PetscErrorCode         ierr;

  PetscFunctionBegin;
  ierr = VecGetArray(xin,&xv);CHKERRQ(ierr);
  if (xin != yin) {ierr = VecGetArray(yin,&yv);CHKERRQ(ierr);} else {yv = xv;}

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
  svalues  = to->values;
  nrecvs   = from->n;
  nsends   = to->n;
  indices  = to->indices;
  sstarts  = to->starts;

  if (!(mode & SCATTER_LOCAL)) {

    if (!from->use_readyreceiver && !to->sendfirst) {  
      /* post receives since they were not previously posted   */
      if (nrecvs) {ierr = MPI_Startall_irecv(from->starts[nrecvs]*from->bs,nrecvs,rwaits);CHKERRQ(ierr);}
    }

    if (ctx->packtogether) {
      /* this version packs all the messages together and sends, when -vecscatter_packtogether used */
      len  = sstarts[nsends];
      val  = svalues;
      for (i=0; i<len; i++) {
        idx    = *indices++;
        val[0] = xv[idx];
        val[1] = xv[idx+1];
        val[2] = xv[idx+2];
        val[3] = xv[idx+3];
        val    += 4;
      }
      if (nsends) {ierr = MPI_Startall_isend(len,nsends,swaits);CHKERRQ(ierr);}
    } else {
      /* this version packs and sends one at a time, default */
      val  = svalues;
      for (i=0; i<nsends; i++) {
        iend = sstarts[i+1]-sstarts[i];

        for (j=0; j<iend; j++) {
          idx     = *indices++;
          val[0] = xv[idx];
          val[1] = xv[idx+1];
          val[2] = xv[idx+2];
          val[3] = xv[idx+3];
          val    += 4;
        } 
        ierr = MPI_Start_isend(4*iend,swaits+i);CHKERRQ(ierr);
      }
    }

    if (!from->use_readyreceiver && to->sendfirst) {  
      /* post receives since they were not previously posted   */
      if (nrecvs) {ierr = MPI_Startall_irecv(from->starts[nrecvs]*from->bs,nrecvs,rwaits);CHKERRQ(ierr);}
    }
  }

  /* take care of local scatters */
  if (to->local.n) {
    PetscInt *tslots = to->local.vslots,*fslots = from->local.vslots;
    PetscInt n       = to->local.n,il,ir;
    if (addv == INSERT_VALUES) {
      if (to->local.is_copy) {
        ierr = PetscMemcpy(yv+from->local.copy_start,xv+to->local.copy_start,to->local.copy_length);CHKERRQ(ierr);
      } else {
        for (i=0; i<n; i++) {
          il = fslots[i]; ir = tslots[i];
          yv[il]   = xv[ir];
          yv[il+1] = xv[ir+1];
          yv[il+2] = xv[ir+2];
          yv[il+3] = xv[ir+3];
        }
      }
    }  else if (addv == ADD_VALUES) {
      for (i=0; i<n; i++) {
        il = fslots[i]; ir = tslots[i];
        yv[il]   += xv[ir];
        yv[il+1] += xv[ir+1];
        yv[il+2] += xv[ir+2];
        yv[il+3] += xv[ir+3];
      }
#if !defined(PETSC_USE_COMPLEX)
    }  else if (addv == MAX_VALUES) {
      for (i=0; i<n; i++) {
        il = fslots[i]; ir = tslots[i];
        yv[il]   = PetscMax(yv[il],xv[ir]);
        yv[il+1] = PetscMax(yv[il+1],xv[ir+1]);
        yv[il+2] = PetscMax(yv[il+2],xv[ir+2]);
        yv[il+3] = PetscMax(yv[il+3],xv[ir+3]);
      }
#endif
    }  else {SETERRQ(PETSC_ERR_ARG_UNKNOWN_TYPE,"Wrong insert option");}
  }
  ierr = VecRestoreArray(xin,&xv);CHKERRQ(ierr);
  if (xin != yin) {ierr = VecRestoreArray(yin,&yv);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

/* --------------------------------------------------------------------------------------*/

#undef __FUNCT__  
#define __FUNCT__ "VecScatterEnd_PtoP_4"
PetscErrorCode VecScatterEnd_PtoP_4(Vec xin,Vec yin,InsertMode addv,ScatterMode mode,VecScatter ctx)
{
  VecScatter_MPI_General *to,*from;
  PetscScalar            *rvalues,*yv,*val;
  PetscErrorCode         ierr;
  PetscInt               nrecvs,nsends,i,*indices,count,n,*rstarts,*lindices,idx;
  PetscMPIInt            imdex;
  MPI_Request            *rwaits,*swaits;
  MPI_Status             *rstatus,*sstatus,xrstatus;

  PetscFunctionBegin;
  if (mode & SCATTER_LOCAL) PetscFunctionReturn(0);
  ierr = VecGetArray(yin,&yv);CHKERRQ(ierr);

  to       = (VecScatter_MPI_General*)ctx->todata;
  from     = (VecScatter_MPI_General*)ctx->fromdata;
  sstatus  = to->sstatus;
  rstatus  = to->rstatus;
  rwaits   = from->requests;
  swaits   = to->requests;
  if (mode & SCATTER_REVERSE) {
    to       = (VecScatter_MPI_General*)ctx->fromdata;
    from     = (VecScatter_MPI_General*)ctx->todata;
    rwaits   = from->rev_requests;
    swaits   = to->rev_requests;
  } 
  rvalues  = from->values;
  nrecvs   = from->n;
  nsends   = to->n;
  indices  = from->indices;
  rstarts  = from->starts;

  if (ctx->packtogether) { /* receive all messages, then unpack all, when -vecscatter_packtogether used */
    if (nrecvs) {ierr = MPI_Waitall(nrecvs,rwaits,rstatus);CHKERRQ(ierr);}
    n        = rstarts[count];
    val      = rvalues;
    lindices = indices;
    if (addv == INSERT_VALUES) {
      for (i=0; i<n; i++) {
	idx       = lindices[i];
	yv[idx]   = val[0];
	yv[idx+1] = val[1];
	yv[idx+2] = val[2];
	yv[idx+3] = val[3];
	val      += 4;
      }
    } else if (addv == ADD_VALUES) {
      for (i=0; i<n; i++) {
	idx       = lindices[i];
	yv[idx]   += val[0];
	yv[idx+1] += val[1];
	yv[idx+2] += val[2];
	yv[idx+3] += val[3];
	val       += 4;
      }
#if !defined(PETSC_USE_COMPLEX)
    } else if (addv == MAX_VALUES) {
      for (i=0; i<n; i++) {
	idx       = lindices[i];
	yv[idx]   = PetscMax(yv[idx],val[0]);
	yv[idx+1] = PetscMax(yv[idx+1],val[1]);
	yv[idx+2] = PetscMax(yv[idx+2],val[2]);
	yv[idx+3] = PetscMax(yv[idx+3],val[3]);
	val       += 4;
      }
#endif
    }  else {SETERRQ(PETSC_ERR_ARG_UNKNOWN_TYPE,"Wrong insert option");}
  } else { /* unpack each message as it arrives, default version */
    count = nrecvs;
    while (count) {
      ierr = MPI_Waitany(nrecvs,rwaits,&imdex,&xrstatus);CHKERRQ(ierr);
      /* unpack receives into our local space */
      val      = rvalues + 4*rstarts[imdex];
      lindices = indices + rstarts[imdex];
      n        = rstarts[imdex+1] - rstarts[imdex];
      if (addv == INSERT_VALUES) {
        for (i=0; i<n; i++) {
          idx       = lindices[i];
          yv[idx]   = val[0];
          yv[idx+1] = val[1];
          yv[idx+2] = val[2];
          yv[idx+3] = val[3];
          val      += 4;
        }
      } else if (addv == ADD_VALUES) {
	for (i=0; i<n; i++) {
	  idx       = lindices[i];
	  yv[idx]   += val[0];
	  yv[idx+1] += val[1];
	  yv[idx+2] += val[2];
	  yv[idx+3] += val[3];
	  val       += 4;
	}
#if !defined(PETSC_USE_COMPLEX)
      } else if (addv == MAX_VALUES) {
	for (i=0; i<n; i++) {
	  idx       = lindices[i];
	  yv[idx]   = PetscMax(yv[idx],val[0]);
	  yv[idx+1] = PetscMax(yv[idx+1],val[1]);
	  yv[idx+2] = PetscMax(yv[idx+2],val[2]);
	  yv[idx+3] = PetscMax(yv[idx+3],val[3]);
	  val       += 4;
	}
#endif
      }  else {SETERRQ(PETSC_ERR_ARG_UNKNOWN_TYPE,"Wrong insert option");}
      count--;
    }
  }
  if (from->use_readyreceiver) {  
    if (from->n) {ierr = MPI_Startall_irecv(from->starts[from->n]*from->bs,from->n,from->requests);CHKERRQ(ierr);}
    ierr = MPI_Barrier(ctx->comm);CHKERRQ(ierr);
  }

  /* wait on sends */
  if (nsends) {ierr = MPI_Waitall(nsends,swaits,sstatus);CHKERRQ(ierr);}
  ierr = VecRestoreArray(yin,&yv);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* --------------------------------------------------------------------------------------*/

#undef __FUNCT__  
#define __FUNCT__ "VecScatterBegin_PtoP_3"
PetscErrorCode VecScatterBegin_PtoP_3(Vec xin,Vec yin,InsertMode addv,ScatterMode mode,VecScatter ctx)
{
  VecScatter_MPI_General *to,*from;
  PetscScalar            *xv,*yv,*val,*svalues;
  MPI_Request            *rwaits,*swaits;
  PetscErrorCode         ierr;
  PetscInt               i,*indices,*sstarts,iend,j,nrecvs,nsends,idx,len;

  PetscFunctionBegin;
  ierr = VecGetArray(xin,&xv);CHKERRQ(ierr);
  if (xin != yin) {ierr = VecGetArray(yin,&yv);CHKERRQ(ierr);} else {yv = xv;}

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
  svalues  = to->values;
  nrecvs   = from->n;
  nsends   = to->n;
  indices  = to->indices;
  sstarts  = to->starts;

  if (!(mode & SCATTER_LOCAL)) {

    if (!from->use_readyreceiver && !to->sendfirst) {  
      /* post receives since they were not previously posted    */
      if (nrecvs) {ierr = MPI_Startall_irecv(from->starts[nrecvs]*from->bs,nrecvs,rwaits);CHKERRQ(ierr);}
    }

    if (ctx->packtogether) {
      /* this version packs all the messages together and sends, when -vecscatter_packtogether used */
      len  = sstarts[nsends];
      val  = svalues;
      for (i=0; i<len; i++) {
        idx    = *indices++;
        val[0] = xv[idx];
        val[1] = xv[idx+1];
        val[2] = xv[idx+2];
        val    += 3;
      }
      if (nsends) {ierr = MPI_Startall_isend(len,nsends,swaits);CHKERRQ(ierr);}
    } else {
      /* this version packs and sends one at a time */
      val  = svalues;
      for (i=0; i<nsends; i++) {
        iend = sstarts[i+1]-sstarts[i];

        for (j=0; j<iend; j++) {
          idx     = *indices++;
          val[0] = xv[idx];
          val[1] = xv[idx+1];
          val[2] = xv[idx+2];
          val    += 3;
        } 
        ierr = MPI_Start_isend(3*iend,swaits+i);CHKERRQ(ierr);
      }
    }

    if (!from->use_readyreceiver && to->sendfirst) {  
      /* post receives since they were not previously posted   */
      if (nrecvs) {ierr = MPI_Startall_irecv(from->starts[nrecvs]*from->bs,nrecvs,rwaits);CHKERRQ(ierr);}
    }
  }

  /* take care of local scatters */
  if (to->local.n) {
    PetscInt *tslots = to->local.vslots,*fslots = from->local.vslots;
    PetscInt n       = to->local.n,il,ir;
    if (addv == INSERT_VALUES) {
      if (to->local.is_copy) {
        ierr = PetscMemcpy(yv + from->local.copy_start,xv + to->local.copy_start,to->local.copy_length);CHKERRQ(ierr);
      } else {
        for (i=0; i<n; i++) {
          il = fslots[i]; ir = tslots[i];
          yv[il]   = xv[ir];
          yv[il+1] = xv[ir+1];
          yv[il+2] = xv[ir+2];
        }
      }
    }  else if (addv == ADD_VALUES) {
      for (i=0; i<n; i++) {
        il = fslots[i]; ir = tslots[i];
        yv[il]   += xv[ir];
        yv[il+1] += xv[ir+1];
        yv[il+2] += xv[ir+2];
      }
#if !defined(PETSC_USE_COMPLEX)
    }  else if (addv == MAX_VALUES) {
      for (i=0; i<n; i++) {
        il = fslots[i]; ir = tslots[i];
        yv[il]   = PetscMax(yv[il],xv[ir]);
        yv[il+1] = PetscMax(yv[il+1],xv[ir+1]);
        yv[il+2] = PetscMax(yv[il+2],xv[ir+2]);
      }
#endif
    }  else {SETERRQ(PETSC_ERR_ARG_UNKNOWN_TYPE,"Wrong insert option");}
  }
  ierr = VecRestoreArray(xin,&xv);CHKERRQ(ierr);
  if (xin != yin) {ierr = VecRestoreArray(yin,&yv);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

/* --------------------------------------------------------------------------------------*/

#undef __FUNCT__  
#define __FUNCT__ "VecScatterEnd_PtoP_3"
PetscErrorCode VecScatterEnd_PtoP_3(Vec xin,Vec yin,InsertMode addv,ScatterMode mode,VecScatter ctx)
{
  VecScatter_MPI_General *to,*from;
  PetscScalar            *rvalues,*yv,*val;
  PetscErrorCode         ierr;
  PetscInt               nrecvs,nsends,i,*indices,count,n,*rstarts,*lindices,idx;
  PetscMPIInt            imdex;
  MPI_Request            *rwaits,*swaits;
  MPI_Status             xrstatus,*rstatus,*sstatus;

  PetscFunctionBegin;
  if (mode & SCATTER_LOCAL) PetscFunctionReturn(0);
  ierr = VecGetArray(yin,&yv);CHKERRQ(ierr);

  to       = (VecScatter_MPI_General*)ctx->todata;
  from     = (VecScatter_MPI_General*)ctx->fromdata;
  sstatus  = to->sstatus;
  rstatus  = to->rstatus;
  rwaits   = from->requests;
  swaits   = to->requests;
  if (mode & SCATTER_REVERSE) {
    to     = (VecScatter_MPI_General*)ctx->fromdata;
    from   = (VecScatter_MPI_General*)ctx->todata;
    rwaits = from->rev_requests;
    swaits = to->rev_requests;
  } 
  rvalues  = from->values;
  nrecvs   = from->n;
  nsends   = to->n;
  indices  = from->indices;
  rstarts  = from->starts;

  /*  wait on receives */
  if (ctx->packtogether) { /* receive all messages, then unpack all, when -vecscatter_packtogether used */
    if (nrecvs) {ierr = MPI_Waitall(nrecvs,rwaits,rstatus);CHKERRQ(ierr);}
    n        = rstarts[count];
    val      = rvalues;
    lindices = indices;
    if (addv == INSERT_VALUES) {
      for (i=0; i<n; i++) {
	idx       = lindices[i];
	yv[idx]   = val[0];
	yv[idx+1] = val[1];
	yv[idx+2] = val[2];
	val      += 3;
      }
    } else if (addv == ADD_VALUES) {
      for (i=0; i<n; i++) {
	idx       = lindices[i];
	yv[idx]   += val[0];
	yv[idx+1] += val[1];
	yv[idx+2] += val[2];
	val       += 3;
      }
#if !defined(PETSC_USE_COMPLEX)
    } else if (addv == MAX_VALUES) {
      for (i=0; i<n; i++) {
	idx       = lindices[i];
	yv[idx]   = PetscMax(yv[idx],val[0]);
	yv[idx+1] = PetscMax(yv[idx+1],val[1]);
	yv[idx+2] = PetscMax(yv[idx+2],val[2]);
	val       += 3;
      }
#endif
    }  else {SETERRQ(PETSC_ERR_ARG_UNKNOWN_TYPE,"Wrong insert option");}
  } else { /* unpack each message as it arrives, default version */
    count = nrecvs;
    while (count) {
      ierr = MPI_Waitany(nrecvs,rwaits,&imdex,&xrstatus);CHKERRQ(ierr);
      /* unpack receives into our local space */
      val      = rvalues + 3*rstarts[imdex];
      lindices = indices + rstarts[imdex];
      n        = rstarts[imdex+1] - rstarts[imdex];
      if (addv == INSERT_VALUES) {
	for (i=0; i<n; i++) {
	  idx       = lindices[i];
	  yv[idx]   = val[0];
	  yv[idx+1] = val[1];
	  yv[idx+2] = val[2];
	  val      += 3;
	}
      } else if (addv == ADD_VALUES) {
	for (i=0; i<n; i++) {
	  idx       = lindices[i];
	  yv[idx]   += val[0];
	  yv[idx+1] += val[1];
	  yv[idx+2] += val[2];
	  val       += 3;
	}
#if !defined(PETSC_USE_COMPLEX)
      } else if (addv == MAX_VALUES) {
	for (i=0; i<n; i++) {
	  idx       = lindices[i];
	  yv[idx]   = PetscMax(yv[idx],val[0]);
	  yv[idx+1] = PetscMax(yv[idx+1],val[1]);
	  yv[idx+2] = PetscMax(yv[idx+2],val[2]);
	  val       += 3;
	}
#endif
      }  else {SETERRQ(PETSC_ERR_ARG_UNKNOWN_TYPE,"Wrong insert option");}
      count--;
    }
  }
  if (from->use_readyreceiver) {  
    if (from->n) {ierr = MPI_Startall_irecv(from->starts[from->n]*from->bs,from->n,from->requests);CHKERRQ(ierr);}
    ierr = MPI_Barrier(ctx->comm);CHKERRQ(ierr);
  }

  /* wait on sends */
  if (nsends) {ierr = MPI_Waitall(nsends,swaits,sstatus);CHKERRQ(ierr);}
  ierr = VecRestoreArray(yin,&yv);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* --------------------------------------------------------------------------------------*/

#undef __FUNCT__  
#define __FUNCT__ "VecScatterBegin_PtoP_2"
PetscErrorCode VecScatterBegin_PtoP_2(Vec xin,Vec yin,InsertMode addv,ScatterMode mode,VecScatter ctx)
{
  VecScatter_MPI_General *to,*from;
  PetscScalar            *xv,*yv,*val,*svalues;
  MPI_Request            *rwaits,*swaits;
  PetscErrorCode         ierr;
  PetscInt               i,*indices,*sstarts,iend,j,nrecvs,nsends,idx,len;

  PetscFunctionBegin;
  ierr = VecGetArray(xin,&xv);CHKERRQ(ierr);
  if (xin != yin) {ierr = VecGetArray(yin,&yv);CHKERRQ(ierr);} else {yv = xv;}
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
  svalues  = to->values;
  nrecvs   = from->n;
  nsends   = to->n;
  indices  = to->indices;
  sstarts  = to->starts;

  if (!(mode & SCATTER_LOCAL)) {

    if (!from->use_readyreceiver && !to->sendfirst) {  
      /* post receives since they were not previously posted    */
      if (nrecvs) {ierr = MPI_Startall_irecv(from->starts[nrecvs]*from->bs,nrecvs,rwaits);CHKERRQ(ierr);}
    }

    if (ctx->packtogether) {
      /* this version packs all the messages together and sends, when -vecscatter_packtogether used */
      len  = sstarts[nsends];
      val  = svalues;
      for (i=0; i<len; i++) {
        idx     = *indices++;
        val[0]  = xv[idx];
        val[1]  = xv[idx+1];
        val    += 2;
      }
      if (nsends) {ierr = MPI_Startall_isend(len,nsends,swaits);CHKERRQ(ierr);}
    } else {
      /* this version packs and sends one at a time */
      val  = svalues;
      for (i=0; i<nsends; i++) {
        iend = sstarts[i+1]-sstarts[i];

        for (j=0; j<iend; j++) {
          idx     = *indices++;
          val[0] = xv[idx];
          val[1] = xv[idx+1];
          val    += 2;
        } 
        ierr = MPI_Start_isend(2*iend,swaits+i);CHKERRQ(ierr);
      }
    }

    if (!from->use_readyreceiver && to->sendfirst) {  
      /* post receives since they were not previously posted   */
      if (nrecvs) {ierr = MPI_Startall_irecv(from->starts[nrecvs]*from->bs,nrecvs,rwaits);CHKERRQ(ierr);}
    }
  }

  /* take care of local scatters */
  if (to->local.n) {
    PetscInt *tslots = to->local.vslots,*fslots = from->local.vslots;
    PetscInt n       = to->local.n,il,ir;
    if (addv == INSERT_VALUES) {
      if (to->local.is_copy) {
        ierr = PetscMemcpy(yv + from->local.copy_start,xv + to->local.copy_start,to->local.copy_length);CHKERRQ(ierr);
      } else {
        for (i=0; i<n; i++) {
          il = fslots[i]; ir = tslots[i];
          yv[il]   = xv[ir];
          yv[il+1] = xv[ir+1];
        }
      }
    }  else if (addv == ADD_VALUES) {
      for (i=0; i<n; i++) {
        il = fslots[i]; ir = tslots[i];
        yv[il]   += xv[ir];
        yv[il+1] += xv[ir+1];
      }
#if !defined(PETSC_USE_COMPLEX)
    }  else if (addv == MAX_VALUES) {
      for (i=0; i<n; i++) {
        il = fslots[i]; ir = tslots[i];
        yv[il]   = PetscMax(yv[il],xv[ir]);
        yv[il+1] = PetscMax(yv[il+1],xv[ir+1]);
      }
#endif
    }  else {SETERRQ(PETSC_ERR_ARG_UNKNOWN_TYPE,"Wrong insert option");}
  }
  ierr = VecRestoreArray(xin,&xv);CHKERRQ(ierr);
  if (xin != yin) {ierr = VecRestoreArray(yin,&yv);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

/* --------------------------------------------------------------------------------------*/

#undef __FUNCT__  
#define __FUNCT__ "VecScatterEnd_PtoP_2"
PetscErrorCode VecScatterEnd_PtoP_2(Vec xin,Vec yin,InsertMode addv,ScatterMode mode,VecScatter ctx)
{
  VecScatter_MPI_General *to,*from;
  PetscScalar            *rvalues,*yv,*val;
  PetscErrorCode         ierr;
  PetscInt               nrecvs,nsends,i,*indices,count,n,*rstarts,*lindices,idx;
  PetscMPIInt            imdex;
  MPI_Request            *rwaits,*swaits;
  MPI_Status             xrstatus,*rstatus,*sstatus;

  PetscFunctionBegin;
  if (mode & SCATTER_LOCAL) PetscFunctionReturn(0);
  ierr = VecGetArray(yin,&yv);CHKERRQ(ierr);

  to       = (VecScatter_MPI_General*)ctx->todata;
  from     = (VecScatter_MPI_General*)ctx->fromdata;
  sstatus  = to->sstatus;
  rstatus  = to->rstatus;
  rwaits   = from->requests;
  swaits   = to->requests;
  if (mode & SCATTER_REVERSE) {
    to       = (VecScatter_MPI_General*)ctx->fromdata;
    from     = (VecScatter_MPI_General*)ctx->todata;
    rwaits   = from->rev_requests;
    swaits   = to->rev_requests;
  }
  rvalues  = from->values;
  nrecvs   = from->n;
  nsends   = to->n;
  indices  = from->indices;
  rstarts  = from->starts;

  /*  wait on receives */
  if (ctx->packtogether) { /* receive all messages, then unpack all, when -vecscatter_packtogether used */
    if (nrecvs) {ierr = MPI_Waitall(nrecvs,rwaits,rstatus);CHKERRQ(ierr);}
    n        = rstarts[count];
    val      = rvalues;
    lindices = indices;
    if (addv == INSERT_VALUES) {
      for (i=0; i<n; i++) {
	idx       = lindices[i];
	yv[idx]   = val[0];
	yv[idx+1] = val[1];
	val      += 2;
      }
    } else if (addv == ADD_VALUES) {
      for (i=0; i<n; i++) {
	idx       = lindices[i];
	yv[idx]   += val[0];
	yv[idx+1] += val[1];
	val       += 2;
      }
#if !defined(PETSC_USE_COMPLEX)
    } else if (addv == MAX_VALUES) {
      for (i=0; i<n; i++) {
	idx       = lindices[i];
	yv[idx]   = PetscMax(yv[idx],val[0]);
	yv[idx+1] = PetscMax(yv[idx+1],val[1]);
	val       += 2;
      }
#endif
    }  else {SETERRQ(PETSC_ERR_ARG_UNKNOWN_TYPE,"Wrong insert option");}
  } else { /* unpack each message as it arrives, default version */
    count = nrecvs;
    while (count) {
      ierr = MPI_Waitany(nrecvs,rwaits,&imdex,&xrstatus);CHKERRQ(ierr);
      /* unpack receives into our local space */
      val      = rvalues + 2*rstarts[imdex];
      lindices = indices + rstarts[imdex];
      n        = rstarts[imdex+1] - rstarts[imdex];
      if (addv == INSERT_VALUES) {
	for (i=0; i<n; i++) {
	  idx       = lindices[i];
	  yv[idx]   = val[0];
	  yv[idx+1] = val[1];
	  val      += 2;
	}
      } else if (addv == ADD_VALUES) {
	for (i=0; i<n; i++) {
	  idx       = lindices[i];
	  yv[idx]   += val[0];
	  yv[idx+1] += val[1];
	  val       += 2;
	}
#if !defined(PETSC_USE_COMPLEX)
      } else if (addv == MAX_VALUES) {
	for (i=0; i<n; i++) {
	  idx       = lindices[i];
	  yv[idx]   = PetscMax(yv[idx],val[0]);
	  yv[idx+1] = PetscMax(yv[idx+1],val[1]);
	  val       += 2;
	}
#endif
      }  else {SETERRQ(PETSC_ERR_ARG_UNKNOWN_TYPE,"Wrong insert option");}
      count--;
    }
  }
  if (from->use_readyreceiver) {  
    if (from->n) {ierr = MPI_Startall_irecv(from->starts[from->n]*from->bs,from->n,from->requests);CHKERRQ(ierr);}
    ierr = MPI_Barrier(ctx->comm);CHKERRQ(ierr);
  }

  /* wait on sends */
  if (nsends) {ierr = MPI_Waitall(nsends,swaits,sstatus);CHKERRQ(ierr);}
  ierr = VecRestoreArray(yin,&yv);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecScatterBegin_PtoP_1"
PetscErrorCode VecScatterBegin_PtoP_1(Vec xin,Vec yin,InsertMode addv,ScatterMode mode,VecScatter ctx)
{
  VecScatter_MPI_General *to,*from;
  PetscScalar            *xv,*yv,*val,*svalues;
  MPI_Request            *rwaits,*swaits;
  PetscErrorCode         ierr;
  PetscInt               i,*indices,*sstarts,iend,j,nrecvs,nsends,len;

  PetscFunctionBegin;
  ierr = VecGetArray(xin,&xv);CHKERRQ(ierr);
  if (xin != yin) {ierr = VecGetArray(yin,&yv);CHKERRQ(ierr);} else {yv = xv;}
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
  svalues  = to->values;
  nrecvs   = from->n;
  nsends   = to->n;
  indices  = to->indices;
  sstarts  = to->starts;

  if (!(mode & SCATTER_LOCAL)) {

    if (!from->use_readyreceiver && !to->sendfirst) {  
      /* post receives since they were not previously posted    */
      if (nrecvs) {ierr = MPI_Startall_irecv(from->starts[nrecvs]*from->bs,nrecvs,rwaits);CHKERRQ(ierr);}
    }

    if (ctx->packtogether) {
      /* this version packs all the messages together and sends, when -vecscatter_packtogether used */
      len  = sstarts[nsends];
      val  = svalues;
      for (i=0; i<len; i++) {
        *val++  = xv[*indices++];
      }
      if (nsends) {ierr = MPI_Startall_isend(len,nsends,swaits);CHKERRQ(ierr);}
    } else {
      /* this version packs and sends one at a time */
      val  = svalues;
      for (i=0; i<nsends; i++) {
        iend = sstarts[i+1]-sstarts[i];
        for (j=0; j<iend; j++) {
          *val++ = xv[*indices++];
        } 
        ierr = MPI_Start_isend(iend,swaits+i);CHKERRQ(ierr);
      }
    }

    if (!from->use_readyreceiver && to->sendfirst) {  
      /* post receives since they were not previously posted   */
      if (nrecvs) {ierr = MPI_Startall_irecv(from->starts[nrecvs]*from->bs,nrecvs,rwaits);CHKERRQ(ierr);}
    }
  }

  /* take care of local scatters */
  if (to->local.n) {
    PetscInt *tslots = to->local.vslots,*fslots = from->local.vslots;
    PetscInt n = to->local.n;
    if (addv == INSERT_VALUES) {
      if (to->local.is_copy) {
        ierr = PetscMemcpy(yv + from->local.copy_start,xv + to->local.copy_start,to->local.copy_length);CHKERRQ(ierr);
      } else {
        for (i=0; i<n; i++) {
          yv[fslots[i]]   = xv[tslots[i]];
        }
      }
    }  else if (addv == ADD_VALUES) {
      for (i=0; i<n; i++) {
        yv[fslots[i]]  += xv[tslots[i]];
      }
#if !defined(PETSC_USE_COMPLEX)
    }  else if (addv == MAX_VALUES) {
      for (i=0; i<n; i++) {
        yv[fslots[i]]  = PetscMax(xv[tslots[i]],yv[fslots[i]]);
      }
#endif
    }  else {SETERRQ(PETSC_ERR_ARG_UNKNOWN_TYPE,"Wrong insert option");}
  }
  ierr = VecRestoreArray(xin,&xv);CHKERRQ(ierr);
  if (xin != yin) {ierr = VecRestoreArray(yin,&yv);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

/* --------------------------------------------------------------------------------------*/

#undef __FUNCT__  
#define __FUNCT__ "VecScatterEnd_PtoP_1"
PetscErrorCode VecScatterEnd_PtoP_1(Vec xin,Vec yin,InsertMode addv,ScatterMode mode,VecScatter ctx)
{
  VecScatter_MPI_General *to,*from;
  PetscScalar            *rvalues,*yv,*val;
  PetscErrorCode         ierr;
  PetscInt               nrecvs,nsends,i,*indices,count,n,*rstarts,*lindices;
  PetscMPIInt            imdex;
  MPI_Request            *rwaits,*swaits;
  MPI_Status             xrstatus,*rstatus,*sstatus;

  PetscFunctionBegin;
  if (mode & SCATTER_LOCAL) PetscFunctionReturn(0);
  ierr = VecGetArray(yin,&yv);CHKERRQ(ierr);

  to       = (VecScatter_MPI_General*)ctx->todata;
  from     = (VecScatter_MPI_General*)ctx->fromdata;
  sstatus  = to->sstatus;
  rstatus  = to->rstatus;
  rwaits   = from->requests;
  swaits   = to->requests;
  if (mode & SCATTER_REVERSE) {
    to       = (VecScatter_MPI_General*)ctx->fromdata;
    from     = (VecScatter_MPI_General*)ctx->todata;
    rwaits   = from->rev_requests;
    swaits   = to->rev_requests;
  }
  rvalues  = from->values;
  nrecvs   = from->n;
  nsends   = to->n;
  indices  = from->indices;
  rstarts  = from->starts;

  /*  wait on receives */
  count = nrecvs;
  while (count) {
    ierr = MPI_Waitany(nrecvs,rwaits,&imdex,&xrstatus);CHKERRQ(ierr);
    /* unpack receives into our local space */
    val      = rvalues + rstarts[imdex];
    lindices = indices + rstarts[imdex];
    n        = rstarts[imdex+1] - rstarts[imdex];
    if (addv == INSERT_VALUES) {
      for (i=0; i<n; i++) {
        yv[lindices[i]]  = *val++;
      }
    } else if (addv == ADD_VALUES) {
      for (i=0; i<n; i++) {
        yv[lindices[i]]  += *val++;
      }
#if !defined(PETSC_USE_COMPLEX)
    } else if (addv == MAX_VALUES) {
      for (i=0; i<n; i++) {
        yv[lindices[i]] = PetscMax(yv[lindices[i]],*val);val++;
      }
#endif
    }  else {SETERRQ(PETSC_ERR_ARG_UNKNOWN_TYPE,"Wrong insert option");}
    count--;
  }
  if (from->use_readyreceiver) {  
    if (from->n) {ierr = MPI_Startall_irecv(from->starts[from->n]*from->bs,from->n,from->requests);CHKERRQ(ierr);}
    ierr = MPI_Barrier(ctx->comm);CHKERRQ(ierr);
  }

  /* wait on sends */
  if (nsends) {ierr = MPI_Waitall(nsends,swaits,sstatus);CHKERRQ(ierr);}
  ierr = VecRestoreArray(yin,&yv);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------------------------*/

#undef __FUNCT__  
#define __FUNCT__ "VecScatterDestroy_PtoP_X"
PetscErrorCode VecScatterDestroy_PtoP_X(VecScatter ctx)
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
  }

  /* release MPI resources obtained with MPI_Send_init() and MPI_Recv_init() */
  /* 
     IBM's PE version of MPI has a bug where freeing these guys will screw up later
     message passing.
  */
#if !defined(PETSC_HAVE_BROKEN_REQUEST_FREE)
  for (i=0; i<to->n; i++) {
    ierr = MPI_Request_free(to->requests + i);CHKERRQ(ierr);
    ierr = MPI_Request_free(to->rev_requests + i);CHKERRQ(ierr);
  }

  /*
      MPICH could not properly cancel requests thus with ready receiver mode we
    cannot free the requests. It may be fixed now, if not then put the following 
    code inside a if !to->use_readyreceiver) {
  */
  for (i=0; i<from->n; i++) {
    ierr = MPI_Request_free(from->requests + i);CHKERRQ(ierr);
    ierr = MPI_Request_free(from->rev_requests + i);CHKERRQ(ierr);
  }  
#endif
 
  ierr = VecScatterDestroy_PtoP(ctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* ==========================================================================================*/

/*              create parallel to sequential scatter context                           */
/*
   bs indicates how many elements there are in each block. Normally this would be 1.
*/
#undef __FUNCT__  
#define __FUNCT__ "VecScatterCreate_PtoS"
PetscErrorCode VecScatterCreate_PtoS(PetscInt nx,PetscInt *inidx,PetscInt ny,PetscInt *inidy,Vec xin,Vec yin,PetscInt bs,VecScatter ctx)
{
  VecScatter_MPI_General *from,*to;
  PetscErrorCode         ierr;
  PetscMPIInt            size,rank,imdex,tag,n;
  PetscInt               *source,*lens,*owners;
  PetscInt               *lowner,*start,lengthy;
  PetscInt               *nprocs,i,j,idx,nsends,nrecvs;
  PetscInt               *owner,*starts,count,slen;
  PetscInt               *rvalues,*svalues,base,nmax,*values,*indx,nprocslocal,lastidx;
  PetscTruth             use_alltoallv;
  MPI_Comm               comm;
  MPI_Request            *send_waits,*recv_waits;
  MPI_Status             recv_status,*send_status;
#if defined(PETSC_USE_DEBUG)
  PetscTruth             found = PETSC_FALSE;
#endif
  
  PetscFunctionBegin;
  ierr = PetscObjectGetNewTag((PetscObject)ctx,&tag);CHKERRQ(ierr);
  ierr = PetscObjectGetComm((PetscObject)xin,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  owners = xin->map.range;
  ierr = VecGetSize(yin,&lengthy);CHKERRQ(ierr);

  /*  first count number of contributors to each processor */
  ierr = PetscMalloc2(2*size,PetscInt,&nprocs,nx,PetscInt,&owner);CHKERRQ(ierr);
  ierr = PetscMemzero(nprocs,2*size*sizeof(PetscInt));CHKERRQ(ierr);
  j       = 0;
  lastidx = -1;
  for (i=0; i<nx; i++) {
    if (lastidx > (idx = inidx[i])) j = 0;
    lastidx = idx;
    for (; j<size; j++) {
      if (idx >= owners[j] && idx < owners[j+1]) {
        nprocs[2*j]++; 
        nprocs[2*j+1] = 1; 
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
  nprocslocal    = nprocs[2*rank]; 
  nprocs[2*rank] = nprocs[2*rank+1] = 0; 
  nsends         = 0;  for (i=0; i<size; i++) { nsends += nprocs[2*i+1];} 

  /* inform other processors of number of messages and max length*/
  ierr = PetscMaxSum(comm,nprocs,&nmax,&nrecvs);CHKERRQ(ierr);

  /* post receives:   */
  ierr = PetscMalloc4(nrecvs*nmax,PetscInt,&rvalues,nrecvs,PetscInt,&lens,nrecvs,PetscInt,&source,nrecvs,MPI_Request,&recv_waits);CHKERRQ(ierr);
  for (i=0; i<nrecvs; i++) {
    ierr = MPI_Irecv((rvalues+nmax*i),nmax,MPIU_INT,MPI_ANY_SOURCE,tag,comm,recv_waits+i);CHKERRQ(ierr);
  }

  /* do sends:
      1) starts[i] gives the starting index in svalues for stuff going to 
         the ith processor
  */
  ierr = PetscMalloc3(nx,PetscInt,&svalues,nsends,MPI_Request,&send_waits,size+1,PetscInt,&starts);CHKERRQ(ierr);
  starts[0]  = 0; 
  for (i=1; i<size; i++) { starts[i] = starts[i-1] + nprocs[2*i-2];} 
  for (i=0; i<nx; i++) {
    if (owner[i] != rank) {
      svalues[starts[owner[i]]++] = inidx[i];
    }
  }

  starts[0] = 0;
  for (i=1; i<size+1; i++) { starts[i] = starts[i-1] + nprocs[2*i-2];} 
  count = 0;
  for (i=0; i<size; i++) {
    if (nprocs[2*i+1]) {
      ierr = MPI_Isend(svalues+starts[i],nprocs[2*i],MPIU_INT,i,tag,comm,send_waits+count++);CHKERRQ(ierr);
    }
  }

  /*  wait on receives */
  count  = nrecvs; 
  slen   = 0;
  while (count) {
    ierr = MPI_Waitany(nrecvs,recv_waits,&imdex,&recv_status);CHKERRQ(ierr);
    /* unpack receives into our local space */
    ierr = MPI_Get_count(&recv_status,MPIU_INT,&n);CHKERRQ(ierr);
    source[imdex]  = recv_status.MPI_SOURCE;
    lens[imdex]    = n;
    slen          += n;
    count--;
  }
  
  /* allocate entire send scatter context */
  ierr  = PetscNew(VecScatter_MPI_General,&to);CHKERRQ(ierr);
  ierr  = PetscOptionsHasName(PETSC_NULL,"-vecscatter_sendfirst",&to->sendfirst);CHKERRQ(ierr);
  to->n = nrecvs; 
  ierr  = PetscMalloc7(bs*slen,PetscScalar,&to->values,nrecvs,MPI_Request,&to->requests,slen,PetscInt,&to->indices,nrecvs+1,PetscInt,&to->starts,
                       nrecvs,PetscMPIInt,&to->procs,PetscMax(to->n,nsends),MPI_Status,&to->sstatus,PetscMax(to->n,nsends),MPI_Status,
                      &to->rstatus);CHKERRQ(ierr);
  ctx->todata   = (void*)to;
  to->starts[0] = 0;

  if (nrecvs) {
    ierr = PetscMalloc(nrecvs*sizeof(PetscInt),&indx);CHKERRQ(ierr);
    for (i=0; i<nrecvs; i++) indx[i] = i;
    ierr = PetscSortIntWithPermutation(nrecvs,source,indx);CHKERRQ(ierr);

    /* move the data into the send scatter */
    base = owners[rank];
    for (i=0; i<nrecvs; i++) {
      to->starts[i+1] = to->starts[i] + lens[indx[i]];
      to->procs[i]    = source[indx[i]];
      values = rvalues + indx[i]*nmax;
      for (j=0; j<lens[indx[i]]; j++) {
        to->indices[to->starts[i] + j] = values[j] - base;
      }
    }
    ierr = PetscFree(indx);CHKERRQ(ierr);
  }
  ierr = PetscFree4(rvalues,lens,source,recv_waits);CHKERRQ(ierr);
 
  /* allocate entire receive scatter context */
  ierr = PetscNew(VecScatter_MPI_General,&from);CHKERRQ(ierr);
  ierr = PetscOptionsHasName(PETSC_NULL,"-vecscatter_sendfirst",&from->sendfirst);CHKERRQ(ierr);
  from->n        = nsends;

  ierr = PetscMalloc5(ny*bs,PetscScalar,&from->values,nsends,MPI_Request,&from->requests,ny,PetscInt,&from->indices,
                      nsends+1,PetscInt,&from->starts,from->n,PetscMPIInt,&from->procs);CHKERRQ(ierr);
  ctx->fromdata  = (void*)from;

  /* move data into receive scatter */
  ierr = PetscMalloc2(size,PetscInt,&lowner,nsends+1,PetscInt,&start);CHKERRQ(ierr);
  count = 0; from->starts[0] = start[0] = 0;
  for (i=0; i<size; i++) {
    if (nprocs[2*i+1]) {
      lowner[i]            = count;
      from->procs[count++] = i;
      from->starts[count]  = start[count] = start[count-1] + nprocs[2*i];
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
    ierr = PetscMalloc2(nt,PetscInt,&to->local.vslots,nt,PetscInt,&from->local.vslots);CHKERRQ(ierr);
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

  to->type   = VEC_SCATTER_MPI_GENERAL; 
  from->type = VEC_SCATTER_MPI_GENERAL;

  from->bs = bs;
  to->bs   = bs;
  ierr     = PetscOptionsHasName(PETSC_NULL,"-vecscatter_alltoallv",&use_alltoallv);CHKERRQ(ierr);
  if (use_alltoallv) {
    ctx->destroy = VecScatterDestroy_PtoP;
    switch (bs) {
    case 12: 
      ctx->begin     = VecScatterBegin_PtoP_Alltoallv_12;
      break;
    case 8: 
      ctx->begin     = VecScatterBegin_PtoP_Alltoallv_8;
      break;
    case 7: 
      ctx->begin     = VecScatterBegin_PtoP_Alltoallv_7;
      break;
    case 6: 
      ctx->begin     = VecScatterBegin_PtoP_Alltoallv_6;
      break;
    case 5: 
      ctx->begin     = VecScatterBegin_PtoP_Alltoallv_5;
      break;
    case 4: 
      ctx->begin     = VecScatterBegin_PtoP_Alltoallv_4;
      break;
    case 3: 
      ctx->begin     = VecScatterBegin_PtoP_Alltoallv_3;
      break;
    case 2: 
      ctx->begin     = VecScatterBegin_PtoP_Alltoallv_2;
      break;
    case 1: 
      ctx->begin     = VecScatterBegin_PtoP_Alltoallv_1;
      break;
    default:
      SETERRQ(PETSC_ERR_SUP,"Blocksize not supported");
    }

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

  } else {
    PetscTruth  use_rr = PETSC_FALSE, use_ssend = PETSC_FALSE;
    PetscInt    *sstarts = to->starts,  *rstarts = from->starts;
    PetscMPIInt *sprocs  = to->procs,   *rprocs  = from->procs;
    MPI_Request *swaits  = to->requests,*rwaits  = from->requests;
    MPI_Request *rev_swaits,*rev_rwaits;
    PetscScalar *Ssvalues = to->values, *Srvalues = from->values;

    tag      = ctx->tag;
    comm     = ctx->comm;

    /* allocate additional wait variables for the "reverse" scatter */
    ierr = PetscMalloc2(nrecvs,MPI_Request,&rev_rwaits,nsends,MPI_Request,&rev_swaits);CHKERRQ(ierr);
    to->rev_requests   = rev_rwaits;
    from->rev_requests = rev_swaits;

    /* Register the receives that you will use later (sends for scatter reverse) */
    ierr = PetscOptionsHasName(PETSC_NULL,"-vecscatter_rr",&use_rr);CHKERRQ(ierr);
    ierr = PetscOptionsHasName(PETSC_NULL,"-vecscatter_ssend",&use_ssend);CHKERRQ(ierr);
    if (use_rr) {
      ierr = PetscInfo(0,"Using VecScatter ready receiver mode\n");CHKERRQ(ierr);
    }
    if (use_ssend) {
      ierr = PetscInfo(0,"Using VecScatter Ssend mode\n");CHKERRQ(ierr);
    }

    for (i=0; i<from->n; i++) {
      ierr = MPI_Recv_init(Srvalues+bs*rstarts[i],bs*rstarts[i+1]-bs*rstarts[i],MPIU_SCALAR,rprocs[i],tag,comm,rwaits+i);CHKERRQ(ierr);
      if (!use_ssend) {
        ierr = MPI_Send_init(Srvalues+bs*rstarts[i],bs*rstarts[i+1]-bs*rstarts[i],MPIU_SCALAR,rprocs[i],tag,comm,rev_swaits+i);CHKERRQ(ierr);
      } else {
        ierr = MPI_Ssend_init(Srvalues+bs*rstarts[i],bs*rstarts[i+1]-bs*rstarts[i],MPIU_SCALAR,rprocs[i],tag,comm,rev_swaits+i);CHKERRQ(ierr);
      }
    }

    if (use_rr) {
      to->use_readyreceiver    = PETSC_TRUE;
      from->use_readyreceiver  = PETSC_TRUE;
      for (i=0; i<to->n; i++) {
        ierr = MPI_Rsend_init(Ssvalues+bs*sstarts[i],bs*sstarts[i+1]-bs*sstarts[i],MPIU_SCALAR,sprocs[i],tag,comm,swaits+i);CHKERRQ(ierr);
      } 
      if (from->n) {ierr = MPI_Startall_irecv(from->starts[from->n]*from->bs,from->n,from->requests);CHKERRQ(ierr);}
      ierr = MPI_Barrier(comm);CHKERRQ(ierr);
    } else {
      to->use_readyreceiver    = PETSC_FALSE;
      from->use_readyreceiver  = PETSC_FALSE;
      for (i=0; i<to->n; i++) {
        if (!use_ssend) {
          ierr = MPI_Send_init(Ssvalues+bs*sstarts[i],bs*sstarts[i+1]-bs*sstarts[i],MPIU_SCALAR,sprocs[i],tag,comm,swaits+i);CHKERRQ(ierr);
	} else {
          ierr = MPI_Ssend_init(Ssvalues+bs*sstarts[i],bs*sstarts[i+1]-bs*sstarts[i],MPIU_SCALAR,sprocs[i],tag,comm,swaits+i);CHKERRQ(ierr);
        }
      } 
    }
    /* Register receives for scatter reverse */
    for (i=0; i<to->n; i++) {
      ierr = MPI_Recv_init(Ssvalues+bs*sstarts[i],bs*sstarts[i+1]-bs*sstarts[i],MPIU_SCALAR,sprocs[i],tag,comm,rev_rwaits+i);CHKERRQ(ierr);
    } 

    ierr = PetscInfo1(0,"Using blocksize %D scatter\n",bs);CHKERRQ(ierr);
    ctx->destroy   = VecScatterDestroy_PtoP_X;
    ctx->copy      = VecScatterCopy_PtoP_X;
    switch (bs) {
    case 12: 
      ctx->begin     = VecScatterBegin_PtoP_12;
      ctx->end       = VecScatterEnd_PtoP_12; 
      break;
    case 8: 
      ctx->begin     = VecScatterBegin_PtoP_8;
      ctx->end       = VecScatterEnd_PtoP_8; 
      break;
    case 7: 
      ctx->begin     = VecScatterBegin_PtoP_7;
      ctx->end       = VecScatterEnd_PtoP_7; 
      break;
    case 6: 
      ctx->begin     = VecScatterBegin_PtoP_6;
      ctx->end       = VecScatterEnd_PtoP_6; 
      break;
    case 5: 
      ctx->begin     = VecScatterBegin_PtoP_5;
      ctx->end       = VecScatterEnd_PtoP_5; 
      break;
    case 4: 
      ctx->begin     = VecScatterBegin_PtoP_4;
      ctx->end       = VecScatterEnd_PtoP_4; 
      break;
    case 3: 
      ctx->begin     = VecScatterBegin_PtoP_3;
      ctx->end       = VecScatterEnd_PtoP_3; 
      break;
    case 2: 
      ctx->begin     = VecScatterBegin_PtoP_2;
      ctx->end       = VecScatterEnd_PtoP_2; 
      break;
    case 1: 
      ctx->begin     = VecScatterBegin_PtoP_1;
      ctx->end       = VecScatterEnd_PtoP_1; 
      break;
    default:
      SETERRQ(PETSC_ERR_SUP,"Blocksize not supported");
    }
  }
  ctx->view      = VecScatterView_MPI;

  /* Check if the local scatter is actually a copy; important special case */
  if (nprocslocal) { 
    ierr = VecScatterLocalOptimizeCopy_Private(&to->local,&from->local,bs);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
#if defined(foo)

/* ------------------------------------------------------------------------------------*/
/*
         Scatter from local Seq vectors to a parallel vector. 
*/
#undef __FUNCT__  
#define __FUNCT__ "VecScatterCreate_StoP"
PetscErrorCode VecScatterCreate_StoP(PetscInt nx,PetscInt *inidx,PetscInt ny,PetscInt *inidy,Vec xin,Vec yin,PetscInt bs,VecScatter ctx)
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
  /*  
  waits              = from->rev_requests;
  from->rev_requests = to->requests;
  to->requests       = waits;
  waits              = to->rev_requests;
  to->rev_requests   = from->requests;
  from->requests     = waits;*/
  PetscFunctionReturn(0);
}
#endif


#undef __FUNCT__  
#define __FUNCT__ "VecScatterCopy_PtoP"
PetscErrorCode VecScatterCopy_PtoP(VecScatter in,VecScatter out)
{
  VecScatter_MPI_General *in_to   = (VecScatter_MPI_General*)in->todata;
  VecScatter_MPI_General *in_from = (VecScatter_MPI_General*)in->fromdata,*out_to,*out_from;
  PetscErrorCode         ierr;
  PetscInt               ny;

  PetscFunctionBegin;
  out->begin     = in->begin;
  out->end       = in->end;
  out->copy      = in->copy;
  out->destroy   = in->destroy;
  out->view      = in->view;

  /* allocate entire send scatter context */
  ierr = PetscNew(VecScatter_MPI_General,&out_to);CHKERRQ(ierr);
  ierr = PetscNew(VecScatter_MPI_General,&out_from);CHKERRQ(ierr);

  ny                = in_to->starts[in_to->n];
  out_to->n         = in_to->n; 
  out_to->type      = in_to->type;
  out_to->sendfirst = in_to->sendfirst;
  ierr = PetscMalloc7(ny,PetscScalar,&out_to->values,out_to->n,MPI_Request,&out_to->requests,ny,PetscInt,&out_to->indices,out_to->n+1,PetscInt,&out_to->starts,
                      out_to->n,PetscMPIInt,&out_to->procs,PetscMax(in_to->n,in_from->n),MPI_Status,&out_to->sstatus,PetscMax(in_to->n,in_from->n),MPI_Status,
                      &out_to->rstatus);CHKERRQ(ierr);
  ierr = PetscMemcpy(out_to->indices,in_to->indices,ny*sizeof(PetscInt));CHKERRQ(ierr);
  ierr = PetscMemcpy(out_to->starts,in_to->starts,(out_to->n+1)*sizeof(PetscInt));CHKERRQ(ierr);
  ierr = PetscMemcpy(out_to->procs,in_to->procs,(out_to->n)*sizeof(PetscMPIInt));CHKERRQ(ierr);

  out->todata      = (void*)out_to;
  out_to->local.n  = in_to->local.n;
  out_to->local.nonmatching_computed = PETSC_FALSE;
  out_to->local.n_nonmatching        = 0;
  out_to->local.slots_nonmatching    = 0;
  if (in_to->local.n) {
    ierr = PetscMalloc2(in_to->local.n,PetscInt,&out_to->local.vslots,in_from->local.n,PetscInt,&out_from->local.vslots);CHKERRQ(ierr);
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

  ierr = PetscMalloc5(ny,PetscScalar,&out_from->values,out_from->n,MPI_Request,&out_from->requests,ny,PetscInt,&out_from->indices,
                      out_from->n+1,PetscInt,&out_from->starts,out_from->n,PetscMPIInt,&out_from->procs);CHKERRQ(ierr);

  ierr = PetscMemcpy(out_from->indices,in_from->indices,ny*sizeof(PetscInt));CHKERRQ(ierr);
  ierr = PetscMemcpy(out_from->starts,in_from->starts,(out_from->n+1)*sizeof(PetscInt));CHKERRQ(ierr);
  ierr = PetscMemcpy(out_from->procs,in_from->procs,(out_from->n)*sizeof(PetscMPIInt));CHKERRQ(ierr);
  out->fromdata       = (void*)out_from;
  out_from->local.n   = in_from->local.n;
  out_from->local.nonmatching_computed = PETSC_FALSE;
  out_from->local.n_nonmatching        = 0;
  out_from->local.slots_nonmatching    = 0;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecScatterCreate_StoP"
PetscErrorCode VecScatterCreate_StoP(PetscInt nx,PetscInt *inidx,PetscInt ny,PetscInt *inidy,Vec xin,Vec yin,PetscInt bs,VecScatter ctx)
{
  VecScatter_MPI_General *from,*to;
  PetscInt               *source,nprocslocal,*lens,*owners = yin->map.range;
  PetscMPIInt            rank = yin->stash.rank,size = yin->stash.size,tag,imdex,n;
  PetscErrorCode         ierr;
  PetscInt               *lowner,*start;
  PetscInt               *nprocs,i,j,idx,nsends,nrecvs;
  PetscInt               *owner,*starts,count,slen;
  PetscInt               *rvalues,*svalues,base,nmax,*values,lastidx;
  MPI_Comm               comm = yin->comm;
  MPI_Request            *send_waits,*recv_waits;
  MPI_Status             recv_status,*send_status;
#if defined(PETSC_USE_DEBUG)
  PetscTruth             found = PETSC_FALSE;
#endif

  PetscFunctionBegin;
  ierr = PetscObjectGetNewTag((PetscObject)ctx,&tag);CHKERRQ(ierr);
  ierr = PetscMalloc5(2*size,PetscInt,&nprocs,nx,PetscInt,&owner,size,PetscInt,&lowner,size,PetscInt,&start,size+1,PetscInt,&starts);CHKERRQ(ierr);

  /*  count number of contributors to each processor */
  ierr    = PetscMemzero(nprocs,2*size*sizeof(PetscInt));CHKERRQ(ierr);
  j       = 0;
  lastidx = -1;
  for (i=0; i<nx; i++) {
    /* if indices are NOT locally sorted, need to start search at the beginning */
    if (lastidx > (idx = inidy[i])) j = 0;
    lastidx = idx;
    for (; j<size; j++) {
      if (idx >= owners[j] && idx < owners[j+1]) {
        nprocs[2*j]++; 
        nprocs[2*j+1] = 1; 
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
  nprocslocal    = nprocs[2*rank];
  nprocs[2*rank] = nprocs[2*rank+1] = 0; 
  nsends = 0;  for (i=0; i<size; i++) { nsends += nprocs[2*i+1];} 

  /* inform other processors of number of messages and max length*/
  ierr = PetscMaxSum(comm,nprocs,&nmax,&nrecvs);CHKERRQ(ierr);

  /* post receives:   */
  ierr = PetscMalloc6(nrecvs*nmax,PetscInt,&rvalues,nrecvs,MPI_Request,&recv_waits,nx,PetscInt,&svalues,nsends,MPI_Request,&send_waits,nrecvs,PetscInt,&lens,nrecvs,PetscInt,&source);CHKERRQ(ierr);

  for (i=0; i<nrecvs; i++) {
    ierr = MPI_Irecv(rvalues+nmax*i,nmax,MPIU_INT,MPI_ANY_SOURCE,tag,comm,recv_waits+i);CHKERRQ(ierr);
  }

  /* do sends:
      1) starts[i] gives the starting index in svalues for stuff going to 
         the ith processor
  */

  starts[0]  = 0; 
  for (i=1; i<size; i++) { starts[i] = starts[i-1] + nprocs[2*i-2];} 
  for (i=0; i<nx; i++) {
    if (owner[i] != rank) {
      svalues[starts[owner[i]]++] = inidy[i];
    }
  }

  /* reset starts because it is destroyed above */
  starts[0]  = 0; 
  for (i=1; i<size; i++) { starts[i] = starts[i-1] + nprocs[2*i-2];} 
  count = 0;
  for (i=0; i<size; i++) {
    if (nprocs[2*i+1]) {
      ierr = MPI_Isend(svalues+starts[i],nprocs[2*i],MPIU_INT,i,tag,comm,send_waits+count);CHKERRQ(ierr);
      count++;
    }
  }

  /* allocate entire send scatter context */
  ierr = PetscNew(VecScatter_MPI_General,&to);CHKERRQ(ierr);
  ctx->todata   = (void*)to;
  to->sendfirst = PETSC_FALSE;
  ierr          = PetscOptionsHasName(PETSC_NULL,"-vecscatter_sendfirst",&to->sendfirst);CHKERRQ(ierr);
  to->n         = nsends; 
  
  ierr = PetscMalloc7(ny,PetscScalar,&to->values,nsends,MPI_Request,&to->requests,ny,PetscInt,&to->indices,nsends+1,PetscInt,&to->starts,
                      nsends,PetscMPIInt,&to->procs,PetscMax(to->n,nrecvs),MPI_Status,&to->sstatus,PetscMax(to->n,nrecvs),MPI_Status,
                      &to->rstatus);CHKERRQ(ierr);

  /* move data into send scatter context */
  count         = 0;
  to->starts[0] = start[0] = 0;
  for (i=0; i<size; i++) {
    if (nprocs[2*i+1]) {
      lowner[i]          = count;
      to->procs[count++] = i;
      to->starts[count]  = start[count] = start[count-1] + nprocs[2*i];
    }
  }
  for (i=0; i<nx; i++) {
    if (owner[i] != rank) {
      to->indices[start[lowner[owner[i]]]++] = inidx[i];
    }
  }
  ierr = PetscFree5(nprocs,owner,lowner,start,starts);CHKERRQ(ierr);

  /*  wait on receives */
  count = nrecvs;
  slen  = 0;
  while (count) {
    ierr = MPI_Waitany(nrecvs,recv_waits,&imdex,&recv_status);CHKERRQ(ierr);
    /* unpack receives into our local space */
    ierr = MPI_Get_count(&recv_status,MPIU_INT,&n);CHKERRQ(ierr);
    source[imdex]  = recv_status.MPI_SOURCE;
    lens[imdex]    = n;
    slen          += n;
    count--;
  }
 
  /* allocate entire receive scatter context */
  ierr = PetscNew(VecScatter_MPI_General,&from);CHKERRQ(ierr);
  ierr = PetscOptionsHasName(PETSC_NULL,"-vecscatter_sendfirst",&from->sendfirst);CHKERRQ(ierr);
  from->n        = nrecvs; 

  ierr = PetscMalloc5(slen,PetscScalar,&from->values,nrecvs,MPI_Request,&from->requests,slen,PetscInt,&from->indices,
                      nrecvs+1,PetscInt,&from->starts,from->n,PetscMPIInt,&from->procs);CHKERRQ(ierr);
  ctx->fromdata  = (void*)from;

  /* move the data into the receive scatter context*/
  base            = owners[rank];
  from->starts[0] = 0;
  for (i=0; i<nrecvs; i++) {
    from->starts[i+1] = from->starts[i] + lens[i];
    from->procs[i]    = source[i];
    values            = rvalues + i*nmax;
    for (j=0; j<lens[i]; j++) {
      from->indices[from->starts[i] + j] = values[j] - base;
    }
  }
    
  /* wait on sends */
  if (nsends) {
    ierr = PetscMalloc(nsends*sizeof(MPI_Status),&send_status);CHKERRQ(ierr);
    ierr = MPI_Waitall(nsends,send_waits,send_status);CHKERRQ(ierr);
    ierr = PetscFree(send_status);CHKERRQ(ierr);
  }
  ierr = PetscFree6(rvalues,recv_waits,svalues,send_waits,lens,source);CHKERRQ(ierr);

  if (nprocslocal) {
    /* we have a scatter to ourselves */
    PetscInt nt = from->local.n = to->local.n = nprocslocal;    
    ierr = PetscMalloc2(nt,PetscInt,&to->local.vslots,nt,PetscInt,&from->local.vslots);CHKERRQ(ierr);
    nt   = 0;
    for (i=0; i<ny; i++) {
      idx = inidy[i];
      if (idx >= owners[rank] && idx < owners[rank+1]) {
        from->local.vslots[nt] = idx - owners[rank];        
        to->local.vslots[nt++] = inidx[i];        
      }
    }
  } else {
    from->local.n     = 0; 
    from->local.vslots = 0;
    to->local.n       = 0;
    to->local.vslots   = 0;

  }
  from->local.nonmatching_computed = PETSC_FALSE;
  from->local.n_nonmatching        = 0;
  from->local.slots_nonmatching    = 0;
  to->local.nonmatching_computed   = PETSC_FALSE;
  to->local.n_nonmatching          = 0;
  to->local.slots_nonmatching      = 0;

  to->type   = VEC_SCATTER_MPI_GENERAL; 
  from->type = VEC_SCATTER_MPI_GENERAL;

  if (bs > 1) {
    ierr = PetscInfo1(0,"Using blocksize %D scatter\n",bs);CHKERRQ(ierr);
    ctx->copy        = VecScatterCopy_PtoP_X;
    switch (bs) {
    case 12: 
      ctx->begin     = VecScatterBegin_PtoP_12;
      ctx->end       = VecScatterEnd_PtoP_12; 
      break;
    case 8: 
      ctx->begin     = VecScatterBegin_PtoP_8;
      ctx->end       = VecScatterEnd_PtoP_8; 
      break;
    case 7: 
      ctx->begin     = VecScatterBegin_PtoP_7;
      ctx->end       = VecScatterEnd_PtoP_7; 
      break;
    case 6: 
      ctx->begin     = VecScatterBegin_PtoP_6;
      ctx->end       = VecScatterEnd_PtoP_6; 
      break;
    case 5: 
      ctx->begin     = VecScatterBegin_PtoP_5;
      ctx->end       = VecScatterEnd_PtoP_5; 
      break;
    case 4: 
      ctx->begin     = VecScatterBegin_PtoP_4;
      ctx->end       = VecScatterEnd_PtoP_4; 
      break;
    case 3: 
      ctx->begin     = VecScatterBegin_PtoP_3;
      ctx->end       = VecScatterEnd_PtoP_3; 
      break;
    case 2: 
      ctx->begin     = VecScatterBegin_PtoP_2;
      ctx->end       = VecScatterEnd_PtoP_2; 
      break;
    default:
      SETERRQ(PETSC_ERR_SUP,"Blocksize not supported");
    }
  } else {
    ierr = PetscInfo(0,"Using nonblocked scatter\n");CHKERRQ(ierr);
    ctx->begin     = VecScatterBegin_PtoP;
    ctx->end       = VecScatterEnd_PtoP; 
    ctx->copy      = VecScatterCopy_PtoP;
  }
  ctx->destroy   = VecScatterDestroy_PtoP;
  ctx->view      = VecScatterView_MPI;

  to->bs   = bs;
  from->bs = bs;

  /* Check if the local scatter is actually a copy; important special case */
  if (nprocslocal) { 
    ierr = VecScatterLocalOptimizeCopy_Private(&to->local,&from->local,bs);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------------------------*/
#undef __FUNCT__  
#define __FUNCT__ "VecScatterCreate_PtoP"
PetscErrorCode VecScatterCreate_PtoP(PetscInt nx,PetscInt *inidx,PetscInt ny,PetscInt *inidy,Vec xin,Vec yin,VecScatter ctx)
{
  PetscErrorCode ierr;
  PetscMPIInt    size,rank,tag,imdex,n;
  PetscInt       *lens,*owners = xin->map.range;
  PetscInt       *nprocs,i,j,idx,nsends,nrecvs,*local_inidx,*local_inidy;
  PetscInt       *owner,*starts,count,slen;
  PetscInt       *rvalues,*svalues,base,nmax,*values,lastidx;
  MPI_Comm       comm;
  MPI_Request    *send_waits,*recv_waits;
  MPI_Status     recv_status,*send_status;
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
  ierr  = PetscMalloc3(2*size,PetscInt,&nprocs,nx,PetscInt,&owner,(size+1),PetscInt,&starts);CHKERRQ(ierr);
  ierr  = PetscMemzero(nprocs,2*size*sizeof(PetscInt));CHKERRQ(ierr);
  lastidx = -1;
  j       = 0;
  for (i=0; i<nx; i++) {
    /* if indices are NOT locally sorted, need to start search at the beginning */
    if (lastidx > (idx = inidx[i])) j = 0;
    lastidx = idx;
    for (; j<size; j++) {
      if (idx >= owners[j] && idx < owners[j+1]) {
        nprocs[2*j]++; 
        nprocs[2*j+1] = 1; 
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
  nsends = 0;  for (i=0; i<size; i++) { nsends += nprocs[2*i+1];} 

  /* inform other processors of number of messages and max length*/
  ierr = PetscMaxSum(comm,nprocs,&nmax,&nrecvs);CHKERRQ(ierr);

  /* post receives:   */
  ierr = PetscMalloc6(2*nrecvs*nmax,PetscInt,&rvalues,2*nx,PetscInt,&svalues,2*nrecvs,PetscInt,&lens,nrecvs,MPI_Request,&recv_waits,nsends,MPI_Request,&send_waits,nsends,MPI_Status,&send_status);CHKERRQ(ierr);

  for (i=0; i<nrecvs; i++) {
    ierr = MPI_Irecv(rvalues+2*nmax*i,2*nmax,MPIU_INT,MPI_ANY_SOURCE,tag,comm,recv_waits+i);CHKERRQ(ierr);
  }

  /* do sends:
      1) starts[i] gives the starting index in svalues for stuff going to 
         the ith processor
  */
  starts[0]= 0; 
  for (i=1; i<size; i++) { starts[i] = starts[i-1] + nprocs[2*i-2];} 
  for (i=0; i<nx; i++) {
    svalues[2*starts[owner[i]]]       = inidx[i];
    svalues[1 + 2*starts[owner[i]]++] = inidy[i];
  }

  starts[0] = 0;
  for (i=1; i<size+1; i++) { starts[i] = starts[i-1] + nprocs[2*i-2];} 
  count = 0;
  for (i=0; i<size; i++) {
    if (nprocs[2*i+1]) {
      ierr = MPI_Isend(svalues+2*starts[i],2*nprocs[2*i],MPIU_INT,i,tag,comm,send_waits+count);CHKERRQ(ierr);
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
    lens[imdex]  =  n/2;
    slen         += n/2;
    count--;
  }
  
  ierr  = PetscMalloc2(slen,PetscInt,&local_inidx,slen,PetscInt,&local_inidy);CHKERRQ(ierr);
  base  = owners[rank];
  count = 0;
  for (i=0; i<nrecvs; i++) {
    values = rvalues + 2*i*nmax;
    for (j=0; j<lens[i]; j++) {
      local_inidx[count]   = values[2*j] - base;
      local_inidy[count++] = values[2*j+1];
    }
  }

  /* wait on sends */
  if (nsends) {ierr = MPI_Waitall(nsends,send_waits,send_status);CHKERRQ(ierr);}
  ierr = PetscFree6(rvalues,svalues,lens,recv_waits,send_waits,send_status);CHKERRQ(ierr);

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
    ierr = PetscInfo(0,"Duplicate from to indices passed in VecScatterCreate(), they are ignored\n");CHKERRQ(ierr);
  }
  ierr = VecScatterCreate_StoP(slen,local_inidx,slen,local_inidy,xin,yin,1,ctx);CHKERRQ(ierr);
  ierr = PetscFree2(local_inidx,local_inidy);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
