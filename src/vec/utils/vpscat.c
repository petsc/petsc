/*$Id: vpscat.c,v 1.165 2001/09/08 13:16:48 bsmith Exp $*/
/*
    Defines parallel vector scatters.
*/

#include "petscsys.h"
#include "src/vec/is/isimpl.h"
#include "src/vec/vecimpl.h"                     /*I "petscvec.h" I*/
#include "src/vec/impls/dvecimpl.h"
#include "src/vec/impls/mpi/pvecimpl.h"

#undef __FUNCT__  
#define __FUNCT__ "VecScatterView_MPI"
int VecScatterView_MPI(VecScatter ctx,PetscViewer viewer)
{
  VecScatter_MPI_General *to=(VecScatter_MPI_General*)ctx->todata;
  VecScatter_MPI_General *from=(VecScatter_MPI_General*)ctx->fromdata;
  int                    i,rank,ierr;
  PetscViewerFormat      format;
  PetscTruth             isascii;

  PetscFunctionBegin;
  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_ASCII,&isascii);CHKERRQ(ierr);
  if (isascii) {
    ierr = MPI_Comm_rank(ctx->comm,&rank);CHKERRQ(ierr);
    ierr = PetscViewerGetFormat(viewer,&format);CHKERRQ(ierr);
    if (format ==  PETSC_VIEWER_ASCII_INFO) {
      int nsend_max,nrecv_max,lensend_max,lenrecv_max,alldata,itmp;

      ierr = MPI_Reduce(&to->n,&nsend_max,1,MPI_INT,MPI_MAX,0,ctx->comm);CHKERRQ(ierr);
      ierr = MPI_Reduce(&from->n,&nrecv_max,1,MPI_INT,MPI_MAX,0,ctx->comm);CHKERRQ(ierr);
      itmp = to->starts[to->n+1];
      ierr = MPI_Reduce(&itmp,&lensend_max,1,MPI_INT,MPI_MAX,0,ctx->comm);CHKERRQ(ierr);
      itmp = from->starts[from->n+1];
      ierr = MPI_Reduce(&itmp,&lenrecv_max,1,MPI_INT,MPI_MAX,0,ctx->comm);CHKERRQ(ierr);
      ierr = MPI_Reduce(&itmp,&alldata,1,MPI_INT,MPI_SUM,0,ctx->comm);CHKERRQ(ierr);

      ierr = PetscViewerASCIIPrintf(viewer,"VecScatter statistics\n");CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"  Maximum number sends %d\n",nsend_max);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"  Maximum number receives %d\n",nrecv_max);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"  Maximum data sent %d\n",lensend_max*to->bs*sizeof(PetscScalar));CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"  Maximum data received %d\n",lenrecv_max*to->bs*sizeof(PetscScalar));CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"  Total data sent %d\n",alldata*to->bs*sizeof(PetscScalar));CHKERRQ(ierr);

    } else { 
      ierr = PetscViewerASCIISynchronizedPrintf(viewer,"[%d] Number sends = %d; Number to self = %d\n",rank,to->n,to->local.n);CHKERRQ(ierr);
      if (to->n) {
        for (i=0; i<to->n; i++){
          ierr = PetscViewerASCIISynchronizedPrintf(viewer,"[%d]   %d length = %d to whom %d\n",rank,i,to->starts[i+1]-to->starts[i],to->procs[i]);CHKERRQ(ierr);
        }
        ierr = PetscViewerASCIISynchronizedPrintf(viewer,"Now the indices for all remote sends (in order by process sent to)\n");CHKERRQ(ierr);
        for (i=0; i<to->starts[to->n]; i++){
          ierr = PetscViewerASCIISynchronizedPrintf(viewer,"[%d]%d \n",rank,to->indices[i]);CHKERRQ(ierr);
        }
      }

      ierr = PetscViewerASCIISynchronizedPrintf(viewer,"[%d]Number receives = %d; Number from self = %d\n",rank,from->n,from->local.n);CHKERRQ(ierr);
      if (from->n) {
	for (i=0; i<from->n; i++){
	  ierr = PetscViewerASCIISynchronizedPrintf(viewer,"[%d] %d length %d from whom %d\n",rank,i,from->starts[i+1]-from->starts[i],from->procs[i]);CHKERRQ(ierr);
	}

	ierr = PetscViewerASCIISynchronizedPrintf(viewer,"Now the indices for all remote receives (in order by process received from)\n");CHKERRQ(ierr);
	for (i=0; i<from->starts[from->n]; i++){
	  ierr = PetscViewerASCIISynchronizedPrintf(viewer,"[%d]%d \n",rank,from->indices[i]);CHKERRQ(ierr);
	}
      }
      if (to->local.n) {
        ierr = PetscViewerASCIISynchronizedPrintf(viewer,"[%d]Indices for local part of scatter\n",rank);CHKERRQ(ierr);
        for (i=0; i<to->local.n; i++){
          ierr = PetscViewerASCIISynchronizedPrintf(viewer,"[%d]From %d to %d \n",rank,from->local.slots[i],to->local.slots[i]);CHKERRQ(ierr);
        }
      }

      ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
    }
  } else {
    SETERRQ1(1,"Viewer type %s not supported for this scatter",((PetscObject)viewer)->type_name);
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
int VecScatterLocalOptimize_Private(VecScatter_Seq_General *gen_to,VecScatter_Seq_General *gen_from)
{
  int n = gen_to->n,n_nonmatching = 0,i,*to_slots = gen_to->slots,*from_slots = gen_from->slots;
  int *nto_slots,*nfrom_slots,j = 0,ierr;
  
  PetscFunctionBegin;
  for (i=0; i<n; i++) {
    if (to_slots[i] != from_slots[i]) n_nonmatching++;
  }

  if (!n_nonmatching) {
    gen_to->nonmatching_computed = PETSC_TRUE;
    gen_to->n_nonmatching        = gen_from->n_nonmatching = 0;
    PetscLogInfo(0,"VecScatterLocalOptimize_Private:Reduced %d to 0\n");
  } else if (n_nonmatching == n) {
    gen_to->nonmatching_computed = PETSC_FALSE;
    PetscLogInfo(0,"VecScatterLocalOptimize_Private:All values non-matching\n");
  } else {
    gen_to->nonmatching_computed= PETSC_TRUE;
    gen_to->n_nonmatching       = gen_from->n_nonmatching = n_nonmatching;
    ierr = PetscMalloc(n_nonmatching*sizeof(int),&nto_slots);CHKERRQ(ierr);
    gen_to->slots_nonmatching   = nto_slots;
    ierr = PetscMalloc(n_nonmatching*sizeof(int),&nfrom_slots);CHKERRQ(ierr);
    gen_from->slots_nonmatching = nfrom_slots;
    for (i=0; i<n; i++) {
      if (to_slots[i] != from_slots[i]) {
        nto_slots[j]   = to_slots[i];
        nfrom_slots[j] = from_slots[i];
        j++;
      }
    }
    PetscLogInfo(0,"VecScatterLocalOptimize_Private:Reduced %d to %d\n",n,n_nonmatching);
  } 
  PetscFunctionReturn(0);
}

/* --------------------------------------------------------------------------------------*/
#undef __FUNCT__  
#define __FUNCT__ "VecScatterCopy_PtoP"
int VecScatterCopy_PtoP(VecScatter in,VecScatter out)
{
  VecScatter_MPI_General *in_to   = (VecScatter_MPI_General*)in->todata;
  VecScatter_MPI_General *in_from = (VecScatter_MPI_General*)in->fromdata,*out_to,*out_from;
  int                    len,ny,ierr;

  PetscFunctionBegin;
  out->postrecvs = in->postrecvs;
  out->begin     = in->begin;
  out->end       = in->end;
  out->copy      = in->copy;
  out->destroy   = in->destroy;
  out->view      = in->view;

  /* allocate entire send scatter context */
  ierr = PetscNew(VecScatter_MPI_General,&out_to);CHKERRQ(ierr);
  ierr = PetscMemzero(out_to,sizeof(VecScatter_MPI_General));CHKERRQ(ierr);
  PetscLogObjectMemory(out,sizeof(VecScatter_MPI_General));
  ny                = in_to->starts[in_to->n];
  len               = ny*(sizeof(int) + sizeof(PetscScalar)) + (in_to->n+1)*sizeof(int) +
                     (in_to->n)*(sizeof(int) + sizeof(MPI_Request));
  out_to->n         = in_to->n; 
  out_to->type      = in_to->type;
  out_to->sendfirst = in_to->sendfirst;
  ierr = PetscMalloc(len,&out_to->values);CHKERRQ(ierr);
  PetscLogObjectMemory(out,len); 
  out_to->requests  = (MPI_Request*)(out_to->values + ny);
  out_to->indices   = (int*)(out_to->requests + out_to->n); 
  out_to->starts    = (int*)(out_to->indices + ny);
  out_to->procs     = (int*)(out_to->starts + out_to->n + 1);
  ierr = PetscMemcpy(out_to->indices,in_to->indices,ny*sizeof(int));CHKERRQ(ierr);
  ierr = PetscMemcpy(out_to->starts,in_to->starts,(out_to->n+1)*sizeof(int));CHKERRQ(ierr);
  ierr = PetscMemcpy(out_to->procs,in_to->procs,(out_to->n)*sizeof(int));CHKERRQ(ierr);
  ierr = PetscMalloc(2*(PetscMax(in_to->n,in_from->n)+1)*sizeof(MPI_Status),&out_to->sstatus);CHKERRQ(ierr);
  out_to->rstatus = out_to->rstatus + PetscMax(in_to->n,in_from->n) + 1;
  PetscLogObjectMemory(out,2*(PetscMax(in_to->n,in_from->n) + 1)*sizeof(MPI_Status));
  out->todata      = (void*)out_to;
  out_to->local.n  = in_to->local.n;
  out_to->local.nonmatching_computed = PETSC_FALSE;
  out_to->local.n_nonmatching        = 0;
  out_to->local.slots_nonmatching    = 0;
  if (in_to->local.n) {
    ierr = PetscMalloc(in_to->local.n*sizeof(int),&out_to->local.slots);CHKERRQ(ierr);
    ierr = PetscMemcpy(out_to->local.slots,in_to->local.slots,in_to->local.n*sizeof(int));CHKERRQ(ierr);
    PetscLogObjectMemory(out,in_to->local.n*sizeof(int));
  } else {
    out_to->local.slots = 0;
  }

  /* allocate entire receive context */
  ierr = PetscNew(VecScatter_MPI_General,&out_from);CHKERRQ(ierr);
  ierr = PetscMemzero(out_from,sizeof(VecScatter_MPI_General));CHKERRQ(ierr);
  out_from->type      = in_from->type;
  PetscLogObjectMemory(out,sizeof(VecScatter_MPI_General));
  ny                  = in_from->starts[in_from->n];
  len                 = ny*(sizeof(int) + sizeof(PetscScalar)) + (in_from->n+1)*sizeof(int) +
                       (in_from->n)*(sizeof(int) + sizeof(MPI_Request));
  out_from->n         = in_from->n; 
  out_from->sendfirst = in_from->sendfirst;
  ierr = PetscMalloc(len,&out_from->values);CHKERRQ(ierr); 
  PetscLogObjectMemory(out,len);
  out_from->requests  = (MPI_Request*)(out_from->values + ny);
  out_from->indices   = (int*)(out_from->requests + out_from->n); 
  out_from->starts    = (int*)(out_from->indices + ny);
  out_from->procs     = (int*)(out_from->starts + out_from->n + 1);
  ierr = PetscMemcpy(out_from->indices,in_from->indices,ny*sizeof(int));CHKERRQ(ierr);
  ierr = PetscMemcpy(out_from->starts,in_from->starts,(out_from->n+1)*sizeof(int));CHKERRQ(ierr);
  ierr = PetscMemcpy(out_from->procs,in_from->procs,(out_from->n)*sizeof(int));CHKERRQ(ierr);
  out->fromdata       = (void*)out_from;
  out_from->local.n   = in_from->local.n;
  out_from->local.nonmatching_computed = PETSC_FALSE;
  out_from->local.n_nonmatching        = 0;
  out_from->local.slots_nonmatching    = 0;
  if (in_from->local.n) {
    ierr = PetscMalloc(in_from->local.n*sizeof(int),&out_from->local.slots);CHKERRQ(ierr);
    PetscLogObjectMemory(out,in_from->local.n*sizeof(int));CHKERRQ(ierr);
    ierr = PetscMemcpy(out_from->local.slots,in_from->local.slots,in_from->local.n*sizeof(int));CHKERRQ(ierr);
  } else {
    out_from->local.slots = 0;
  }
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------------------*/
#undef __FUNCT__  
#define __FUNCT__ "VecScatterDestroy_PtoP"
int VecScatterDestroy_PtoP(VecScatter ctx)
{
  VecScatter_MPI_General *gen_to   = (VecScatter_MPI_General*)ctx->todata;
  VecScatter_MPI_General *gen_from = (VecScatter_MPI_General*)ctx->fromdata;
  int ierr;

  PetscFunctionBegin;
  if (gen_to->local.slots) {ierr = PetscFree(gen_to->local.slots);CHKERRQ(ierr);}
  if (gen_from->local.slots) {ierr = PetscFree(gen_from->local.slots);CHKERRQ(ierr);}
  if (gen_to->local.slots_nonmatching) {ierr = PetscFree(gen_to->local.slots_nonmatching);CHKERRQ(ierr);}
  if (gen_from->local.slots_nonmatching) {ierr = PetscFree(gen_from->local.slots_nonmatching);CHKERRQ(ierr);}
  ierr = PetscFree(gen_to->sstatus);CHKERRQ(ierr);
  ierr = PetscFree(gen_to->values);CHKERRQ(ierr);
  ierr = PetscFree(gen_to);CHKERRQ(ierr);
  ierr = PetscFree(gen_from->values);CHKERRQ(ierr);
  ierr = PetscFree(gen_from);CHKERRQ(ierr);
  PetscLogObjectDestroy(ctx);
  PetscHeaderDestroy(ctx);
  PetscFunctionReturn(0);
}

/* --------------------------------------------------------------------------------------*/
/*
     Even though the next routines are written with parallel 
  vectors, either xin or yin (but not both) may be Seq
  vectors, one for each processor.
  
     gen_from indices indicate where arriving stuff is stashed
     gen_to   indices indicate where departing stuff came from. 
     the naming can be VERY confusing.

*/
#undef __FUNCT__  
#define __FUNCT__ "VecScatterBegin_PtoP"
int VecScatterBegin_PtoP(Vec xin,Vec yin,InsertMode addv,ScatterMode mode,VecScatter ctx)
{
  VecScatter_MPI_General *gen_to,*gen_from;
  MPI_Comm               comm = ctx->comm;
  PetscScalar            *xv,*yv,*val,*rvalues,*svalues;
  MPI_Request            *rwaits,*swaits;
  int                    tag = ctx->tag,i,j,*indices,*rstarts,*sstarts,*rprocs,*sprocs;
  int                    nrecvs,nsends,iend,ierr;

  PetscFunctionBegin;
  ierr = VecGetArray(xin,&xv);CHKERRQ(ierr);
  if (xin != yin) {ierr = VecGetArray(yin,&yv);CHKERRQ(ierr);} else {yv = xv;}
  if (mode & SCATTER_REVERSE){
    gen_to   = (VecScatter_MPI_General*)ctx->fromdata;
    gen_from = (VecScatter_MPI_General*)ctx->todata;
  } else {
    gen_to   = (VecScatter_MPI_General*)ctx->todata;
    gen_from = (VecScatter_MPI_General*)ctx->fromdata;
  }
  rvalues  = gen_from->values;
  svalues  = gen_to->values;
  nrecvs   = gen_from->n;
  nsends   = gen_to->n;
  rwaits   = gen_from->requests;
  swaits   = gen_to->requests;
  indices  = gen_to->indices;
  rstarts  = gen_from->starts;
  sstarts  = gen_to->starts;
  rprocs   = gen_from->procs;
  sprocs   = gen_to->procs;

  if (!(mode & SCATTER_LOCAL)) {  

    if (gen_to->sendfirst) {
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

    if (!gen_to->sendfirst) {
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
  if (gen_to->local.n && addv == INSERT_VALUES) {
    if (yv == xv && !gen_to->local.nonmatching_computed) {
      ierr = VecScatterLocalOptimize_Private(&gen_to->local,&gen_from->local);CHKERRQ(ierr);
    }
    if (gen_to->local.is_copy) {
      ierr = PetscMemcpy(yv + gen_from->local.copy_start,xv + gen_to->local.copy_start,gen_to->local.copy_length);CHKERRQ(ierr);
    } else if (yv != xv || gen_to->local.nonmatching_computed == -1) {
      int *tslots = gen_to->local.slots,*fslots = gen_from->local.slots;
      int n       = gen_to->local.n;
      for (i=0; i<n; i++) {yv[fslots[i]] = xv[tslots[i]];}
    } else {
      /* 
        In this case, it is copying the values into their old locations, thus we can skip those  
      */
      int *tslots = gen_to->local.slots_nonmatching,*fslots = gen_from->local.slots_nonmatching;
      int n       = gen_to->local.n_nonmatching;
      for (i=0; i<n; i++) {yv[fslots[i]] = xv[tslots[i]];}
    } 
  } else if (gen_to->local.n) {
    int *tslots = gen_to->local.slots,*fslots = gen_from->local.slots;
    int n = gen_to->local.n;
    if (addv == ADD_VALUES) {
      for (i=0; i<n; i++) {yv[fslots[i]] += xv[tslots[i]];}
#if !defined(PETSC_USE_COMPLEX)
    } else if (addv == MAX_VALUES) {
      for (i=0; i<n; i++) {yv[fslots[i]] = PetscMax(yv[fslots[i]],xv[tslots[i]]);}
#endif
    } else {SETERRQ(1,"Wrong insert option");}
  }

  ierr = VecRestoreArray(xin,&xv);CHKERRQ(ierr);
  if (xin != yin) {ierr = VecRestoreArray(yin,&yv);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

/* --------------------------------------------------------------------------------------*/
#undef __FUNCT__  
#define __FUNCT__ "VecScatterEnd_PtoP"
int VecScatterEnd_PtoP(Vec xin,Vec yin,InsertMode addv,ScatterMode mode,VecScatter ctx)
{
  VecScatter_MPI_General *gen_to,*gen_from;
  PetscScalar            *rvalues,*yv,*val;
  int                    ierr,nrecvs,nsends,i,*indices,count,imdex,n,*rstarts,*lindices;
  MPI_Request            *rwaits,*swaits;
  MPI_Status             rstatus,*sstatus;

  PetscFunctionBegin;
  if (mode & SCATTER_LOCAL) PetscFunctionReturn(0);
  ierr = VecGetArray(yin,&yv);CHKERRQ(ierr);

  if (mode & SCATTER_REVERSE){
    gen_to   = (VecScatter_MPI_General*)ctx->fromdata;
    gen_from = (VecScatter_MPI_General*)ctx->todata;
    sstatus  = gen_from->sstatus;
  } else {
    gen_to   = (VecScatter_MPI_General*)ctx->todata;
    gen_from = (VecScatter_MPI_General*)ctx->fromdata;
    sstatus  = gen_to->sstatus;
  }
  rvalues  = gen_from->values;
  nrecvs   = gen_from->n;
  nsends   = gen_to->n;
  rwaits   = gen_from->requests;
  swaits   = gen_to->requests;
  indices  = gen_from->indices;
  rstarts  = gen_from->starts;

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
    }  else {SETERRQ(1,"Wrong insert option");}
    count--;
  }

  /* wait on sends */
  if (nsends) {
    ierr = MPI_Waitall(nsends,swaits,sstatus);CHKERRQ(ierr);
  }
  ierr = VecRestoreArray(yin,&yv);CHKERRQ(ierr);
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

/*
    This is for use with the "ready-receiver" mode. In theory on some
    machines it could lead to better performance. In practice we've never
    seen it give better performance. Accessed with the -vecscatter_rr flag.
*/
#undef __FUNCT__  
#define __FUNCT__ "VecScatterPostRecvs_PtoP_X"
int VecScatterPostRecvs_PtoP_X(Vec xin,Vec yin,InsertMode addv,ScatterMode mode,VecScatter ctx)
{
  VecScatter_MPI_General *gen_from = (VecScatter_MPI_General*)ctx->fromdata;

  PetscFunctionBegin;
  MPI_Startall_irecv(gen_from->starts[gen_from->n]*gen_from->bs,gen_from->n,gen_from->requests); 
  PetscFunctionReturn(0);
}

/* --------------------------------------------------------------------------------------*/
/*
    Special optimization to see if the local part of the scatter is actually 
    a copy. The scatter routines call PetscMemcpy() instead.
 
*/
#undef __FUNCT__  
#define __FUNCT__ "VecScatterLocalOptimizeCopy_Private"
int VecScatterLocalOptimizeCopy_Private(VecScatter_Seq_General *gen_to,VecScatter_Seq_General *gen_from,int bs)
{
  int n = gen_to->n,i,*to_slots = gen_to->slots,*from_slots = gen_from->slots;
  int to_start,from_start;
  
  PetscFunctionBegin;
  to_start   = to_slots[0];
  from_start = from_slots[0];

  for (i=1; i<n; i++) {
    to_start   += bs;
    from_start += bs;
    if (to_slots[i]   != to_start)   PetscFunctionReturn(0);
    if (from_slots[i] != from_start) PetscFunctionReturn(0);
  }
  gen_to->is_copy       = PETSC_TRUE; 
  gen_to->copy_start    = to_slots[0]; 
  gen_to->copy_length   = bs*sizeof(PetscScalar)*n;
  gen_from->is_copy     = PETSC_TRUE;
  gen_from->copy_start  = from_slots[0];
  gen_from->copy_length = bs*sizeof(PetscScalar)*n;

  PetscLogInfo(0,"VecScatterLocalOptimizeCopy_Private:Local scatter is a copy, optimizing for it\n");

  PetscFunctionReturn(0);
}

/* --------------------------------------------------------------------------------------*/

#undef __FUNCT__  
#define __FUNCT__ "VecScatterCopy_PtoP_X"
int VecScatterCopy_PtoP_X(VecScatter in,VecScatter out)
{
  VecScatter_MPI_General *in_to   = (VecScatter_MPI_General*)in->todata;
  VecScatter_MPI_General *in_from = (VecScatter_MPI_General*)in->fromdata,*out_to,*out_from;
  int                    len,ny,bs = in_from->bs,ierr;

  PetscFunctionBegin;
  out->postrecvs = in->postrecvs;
  out->begin     = in->begin;
  out->end       = in->end;
  out->copy      = in->copy;
  out->destroy   = in->destroy;
  out->view      = in->view;

  /* allocate entire send scatter context */
  ierr = PetscNew(VecScatter_MPI_General,&out_to);CHKERRQ(ierr);
  ierr = PetscMemzero(out_to,sizeof(VecScatter_MPI_General));CHKERRQ(ierr);
  PetscLogObjectMemory(out,sizeof(VecScatter_MPI_General));
  ny                = in_to->starts[in_to->n];
  len               = ny*(sizeof(int) + bs*sizeof(PetscScalar)) + (in_to->n+1)*sizeof(int) +
                     (in_to->n)*(sizeof(int) + sizeof(MPI_Request));
  out_to->n         = in_to->n; 
  out_to->type      = in_to->type;
  out_to->sendfirst = in_to->sendfirst;

  ierr = PetscMalloc(len,&out_to->values);CHKERRQ(ierr);
  PetscLogObjectMemory(out,len); 
  out_to->requests  = (MPI_Request*)(out_to->values + bs*ny);
  out_to->indices   = (int*)(out_to->requests + out_to->n); 
  out_to->starts    = (int*)(out_to->indices + ny);
  out_to->procs     = (int*)(out_to->starts + out_to->n + 1);
  ierr = PetscMemcpy(out_to->indices,in_to->indices,ny*sizeof(int));CHKERRQ(ierr);
  ierr = PetscMemcpy(out_to->starts,in_to->starts,(out_to->n+1)*sizeof(int));CHKERRQ(ierr);
  ierr = PetscMemcpy(out_to->procs,in_to->procs,(out_to->n)*sizeof(int));CHKERRQ(ierr);
  ierr = PetscMalloc(2*(PetscMax(in_to->n,in_from->n)+1)*sizeof(MPI_Status),&out_to->sstatus);CHKERRQ(ierr);
  out_to->rstatus   =  out_to->sstatus + PetscMax(in_to->n,in_from->n) + 1;
     
  PetscLogObjectMemory(out,2*(PetscMax(in_to->n,in_from->n)+1)*sizeof(MPI_Status));
  out->todata       = (void*)out_to;
  out_to->local.n   = in_to->local.n;
  out_to->local.nonmatching_computed = PETSC_FALSE;
  out_to->local.n_nonmatching        = 0;
  out_to->local.slots_nonmatching    = 0;
  if (in_to->local.n) {
    ierr = PetscMalloc(in_to->local.n*sizeof(int),out_to->local.slots);CHKERRQ(ierr);
    ierr = PetscMemcpy(out_to->local.slots,in_to->local.slots,in_to->local.n*sizeof(int));CHKERRQ(ierr);
    PetscLogObjectMemory(out,in_to->local.n*sizeof(int));
  } else {
    out_to->local.slots = 0;
  }

  /* allocate entire receive context */
  ierr = PetscNew(VecScatter_MPI_General,&out_from);CHKERRQ(ierr);
  ierr = PetscMemzero(out_from,sizeof(VecScatter_MPI_General));CHKERRQ(ierr);
  out_from->type      = in_from->type;
  PetscLogObjectMemory(out,sizeof(VecScatter_MPI_General));
  ny                  = in_from->starts[in_from->n];
  len                 = ny*(sizeof(int) + bs*sizeof(PetscScalar)) + (in_from->n+1)*sizeof(int) +
                       (in_from->n)*(sizeof(int) + sizeof(MPI_Request));
  out_from->n         = in_from->n; 
  out_from->sendfirst = in_from->sendfirst;
  ierr = PetscMalloc(len,&out_from->values);CHKERRQ(ierr); 
  PetscLogObjectMemory(out,len);
  out_from->requests  = (MPI_Request*)(out_from->values + bs*ny);
  out_from->indices   = (int*)(out_from->requests + out_from->n); 
  out_from->starts    = (int*)(out_from->indices + ny);
  out_from->procs     = (int*)(out_from->starts + out_from->n + 1);
  ierr = PetscMemcpy(out_from->indices,in_from->indices,ny*sizeof(int));CHKERRQ(ierr);
  ierr = PetscMemcpy(out_from->starts,in_from->starts,(out_from->n+1)*sizeof(int));CHKERRQ(ierr);
  ierr = PetscMemcpy(out_from->procs,in_from->procs,(out_from->n)*sizeof(int));CHKERRQ(ierr);
  out->fromdata       = (void*)out_from;
  out_from->local.n   = in_from->local.n;
  out_from->local.nonmatching_computed = PETSC_FALSE;
  out_from->local.n_nonmatching        = 0;
  out_from->local.slots_nonmatching    = 0;
  if (in_from->local.n) {
    ierr = PetscMalloc(in_from->local.n*sizeof(int),&out_from->local.slots);CHKERRQ(ierr);
    PetscLogObjectMemory(out,in_from->local.n*sizeof(int));CHKERRQ(ierr);
    ierr = PetscMemcpy(out_from->local.slots,in_from->local.slots,in_from->local.n*sizeof(int));CHKERRQ(ierr);
  } else {
    out_from->local.slots = 0;
  }

  /* 
      set up the request arrays for use with isend_init() and irecv_init()
  */
  {
    MPI_Comm    comm;
    int         *sstarts = out_to->starts,  *rstarts = out_from->starts;
    int         *sprocs  = out_to->procs,   *rprocs  = out_from->procs;
    int         tag,i;
    PetscTruth  flg;
    MPI_Request *swaits  = out_to->requests,*rwaits  = out_from->requests;
    MPI_Request *rev_swaits,*rev_rwaits;
    PetscScalar *Ssvalues = out_to->values, *Srvalues = out_from->values;

    ierr = PetscMalloc((in_to->n+1)*sizeof(MPI_Request),&out_to->rev_requests);CHKERRQ(ierr);
    ierr = PetscMalloc((in_from->n+1)*sizeof(MPI_Request),&out_from->rev_requests);CHKERRQ(ierr);
    PetscLogObjectMemory(out,(in_to->n+in_from->n+2)*sizeof(MPI_Request));

    rev_rwaits = out_to->rev_requests;
    rev_swaits = out_from->rev_requests;

    out_from->bs = out_to->bs = bs; 
    tag     = out->tag;
    comm    = out->comm;

    /* Register the receives that you will use later (sends for scatter reverse) */
    for (i=0; i<out_from->n; i++) {
      ierr = MPI_Recv_init(Srvalues+bs*rstarts[i],bs*rstarts[i+1]-bs*rstarts[i],MPIU_SCALAR,rprocs[i],tag,
                           comm,rwaits+i);CHKERRQ(ierr);
      ierr = MPI_Send_init(Srvalues+bs*rstarts[i],bs*rstarts[i+1]-bs*rstarts[i],MPIU_SCALAR,rprocs[i],tag,
                           comm,rev_swaits+i);CHKERRQ(ierr);
    }

    ierr = PetscOptionsHasName(PETSC_NULL,"-vecscatter_rr",&flg);CHKERRQ(ierr);
    if (flg) {
      out->postrecvs               = VecScatterPostRecvs_PtoP_X;
      out_to->use_readyreceiver    = PETSC_TRUE;
      out_from->use_readyreceiver  = PETSC_TRUE;
      for (i=0; i<out_to->n; i++) {
        ierr = MPI_Rsend_init(Ssvalues+bs*sstarts[i],bs*sstarts[i+1]-bs*sstarts[i],MPIU_SCALAR,sprocs[i],tag,
                              comm,swaits+i);CHKERRQ(ierr);
      } 
      PetscLogInfo(0,"VecScatterCopy_PtoP_X:Using VecScatter ready receiver mode\n");
    } else {
      out->postrecvs               = 0;
      out_to->use_readyreceiver    = PETSC_FALSE;
      out_from->use_readyreceiver  = PETSC_FALSE;
      flg                          = PETSC_FALSE;
      ierr                         = PetscOptionsHasName(PETSC_NULL,"-vecscatter_ssend",&flg);CHKERRQ(ierr);
      if (flg) {
        PetscLogInfo(0,"VecScatterCopy_PtoP_X:Using VecScatter Ssend mode\n");
      }
      for (i=0; i<out_to->n; i++) {
        if (!flg) {
          ierr = MPI_Send_init(Ssvalues+bs*sstarts[i],bs*sstarts[i+1]-bs*sstarts[i],MPIU_SCALAR,sprocs[i],tag,
                               comm,swaits+i);CHKERRQ(ierr);
        } else {
          ierr = MPI_Ssend_init(Ssvalues+bs*sstarts[i],bs*sstarts[i+1]-bs*sstarts[i],MPIU_SCALAR,sprocs[i],tag,
                               comm,swaits+i);CHKERRQ(ierr);
        }
      } 
    }
    /* Register receives for scatter reverse */
    for (i=0; i<out_to->n; i++) {
      ierr = MPI_Recv_init(Ssvalues+bs*sstarts[i],bs*sstarts[i+1]-bs*sstarts[i],MPIU_SCALAR,sprocs[i],tag,
                           comm,rev_rwaits+i);CHKERRQ(ierr);
    }
  } 

  PetscFunctionReturn(0);
}

/* --------------------------------------------------------------------------------------*/

#undef __FUNCT__  
#define __FUNCT__ "VecScatterBegin_PtoP_12"
int VecScatterBegin_PtoP_12(Vec xin,Vec yin,InsertMode addv,ScatterMode mode,VecScatter ctx)
{
  VecScatter_MPI_General *gen_to,*gen_from;
  PetscScalar            *xv,*yv,*val,*svalues;
  MPI_Request            *rwaits,*swaits;
  int                    *indices,*sstarts,iend,i,j,nrecvs,nsends,idx,ierr,len;

  PetscFunctionBegin;
  ierr = VecGetArray(xin,&xv);CHKERRQ(ierr);
  if (xin != yin) {ierr = VecGetArray(yin,&yv);CHKERRQ(ierr);} else {yv = xv;}

  if (mode & SCATTER_REVERSE) {
    gen_to   = (VecScatter_MPI_General*)ctx->fromdata;
    gen_from = (VecScatter_MPI_General*)ctx->todata;
    rwaits   = gen_from->rev_requests;
    swaits   = gen_to->rev_requests;
  } else {
    gen_to   = (VecScatter_MPI_General*)ctx->todata;
    gen_from = (VecScatter_MPI_General*)ctx->fromdata;
    rwaits   = gen_from->requests;
    swaits   = gen_to->requests;
  }
  svalues  = gen_to->values;
  nrecvs   = gen_from->n;
  nsends   = gen_to->n;
  indices  = gen_to->indices;
  sstarts  = gen_to->starts;

  if (!(mode & SCATTER_LOCAL)) {

    if (!gen_from->use_readyreceiver && !gen_to->sendfirst) {  
      /* post receives since they were not posted in VecScatterPostRecvs()   */
      ierr = MPI_Startall_irecv(gen_from->starts[nrecvs]*gen_from->bs,nrecvs,rwaits);CHKERRQ(ierr);
    }
    if (ctx->packtogether) {
      /* this version packs all the messages together and sends, when -vecscatter_packtogether used */
      len  = 12*sstarts[nsends];
      val  = svalues;
      for (i=0; i<len; i += 12) {
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
      ierr = MPI_Startall_isend(len,nsends,swaits);CHKERRQ(ierr);
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

    if (!gen_from->use_readyreceiver && gen_to->sendfirst) {  
      /* post receives since they were not posted in VecScatterPostRecvs()   */
      ierr = MPI_Startall_irecv(gen_from->starts[nrecvs]*gen_from->bs,nrecvs,rwaits);CHKERRQ(ierr);
    }
  }

  /* take care of local scatters */
  if (gen_to->local.n) {
    int *tslots = gen_to->local.slots,*fslots = gen_from->local.slots;
    int n       = gen_to->local.n,il,ir;
    if (addv == INSERT_VALUES) {
      if (gen_to->local.is_copy) {
        ierr = PetscMemcpy(yv+gen_from->local.copy_start,xv+gen_to->local.copy_start,gen_to->local.copy_length);CHKERRQ(ierr);
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
    } else {SETERRQ(1,"Wrong insert option");}
  }  
  ierr = VecRestoreArray(xin,&xv);CHKERRQ(ierr);
  if (xin != yin) {ierr = VecRestoreArray(yin,&yv);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

/* --------------------------------------------------------------------------------------*/

#undef __FUNCT__  
#define __FUNCT__ "VecScatterEnd_PtoP_12"
int VecScatterEnd_PtoP_12(Vec xin,Vec yin,InsertMode addv,ScatterMode mode,VecScatter ctx)
{
  VecScatter_MPI_General *gen_to,*gen_from;
  PetscScalar            *rvalues,*yv,*val;
  int                    ierr,nrecvs,nsends,i,*indices,count,imdex,n,*rstarts,*lindices,idx;
  MPI_Request            *rwaits,*swaits;
  MPI_Status             *rstatus,*sstatus;

  PetscFunctionBegin;
  if (mode & SCATTER_LOCAL) PetscFunctionReturn(0);
  ierr = VecGetArray(yin,&yv);CHKERRQ(ierr);

  if (mode & SCATTER_REVERSE) {
    gen_to   = (VecScatter_MPI_General*)ctx->fromdata;
    gen_from = (VecScatter_MPI_General*)ctx->todata;
    rwaits   = gen_from->rev_requests;
    swaits   = gen_to->rev_requests;
    sstatus  = gen_from->sstatus;
    rstatus  = gen_from->rstatus;
  } else {
    gen_to   = (VecScatter_MPI_General*)ctx->todata;
    gen_from = (VecScatter_MPI_General*)ctx->fromdata;
    rwaits   = gen_from->requests;
    swaits   = gen_to->requests;
    sstatus  = gen_to->sstatus;
    rstatus  = gen_to->rstatus;
  }
  rvalues  = gen_from->values;
  nrecvs   = gen_from->n;
  nsends   = gen_to->n;
  indices  = gen_from->indices;
  rstarts  = gen_from->starts;

  /*  wait on receives */
  count = nrecvs;
  if (ctx->packtogether) { /* receive all messages, then unpack all, when -vecscatter_packtogether used */
    ierr     = MPI_Waitall(nrecvs,rwaits,rstatus);CHKERRQ(ierr);
    n        = rstarts[count];
    val      = rvalues;
    lindices = indices;
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
	idx         = lindices[i];
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
	val       += 12;
      }
#endif
    }  else {SETERRQ(1,"Wrong insert option");}
  } else { /* unpack each message as it arrives, default version */
    while (count) {
      ierr = MPI_Waitany(nrecvs,rwaits,&imdex,&rstatus[0]);CHKERRQ(ierr);
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
    }  else {SETERRQ(1,"Wrong insert option");}
    count--;
    }
  }
  /* wait on sends */
  if (nsends) {
    ierr = MPI_Waitall(nsends,swaits,sstatus);CHKERRQ(ierr);
  }
  ierr = VecRestoreArray(yin,&yv);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* --------------------------------------------------------------------------------------*/

#undef __FUNCT__  
#define __FUNCT__ "VecScatterBegin_PtoP_5"
int VecScatterBegin_PtoP_5(Vec xin,Vec yin,InsertMode addv,ScatterMode mode,VecScatter ctx)
{
  VecScatter_MPI_General *gen_to,*gen_from;
  PetscScalar            *xv,*yv,*val,*svalues;
  MPI_Request            *rwaits,*swaits;
  int                    ierr,i,*indices,*sstarts,iend,j,nrecvs,nsends,idx;

  PetscFunctionBegin;
  ierr = VecGetArray(xin,&xv);CHKERRQ(ierr);
  if (xin != yin) {ierr = VecGetArray(yin,&yv);CHKERRQ(ierr);} else {yv = xv;}
  if (mode & SCATTER_REVERSE) {
    gen_to   = (VecScatter_MPI_General*)ctx->fromdata;
    gen_from = (VecScatter_MPI_General*)ctx->todata;
    rwaits   = gen_from->rev_requests;
    swaits   = gen_to->rev_requests;
  } else {
    gen_to   = (VecScatter_MPI_General*)ctx->todata;
    gen_from = (VecScatter_MPI_General*)ctx->fromdata;
    rwaits   = gen_from->requests;
    swaits   = gen_to->requests;
  }
  svalues  = gen_to->values;
  nrecvs   = gen_from->n;
  nsends   = gen_to->n;
  indices  = gen_to->indices;
  sstarts  = gen_to->starts;

  if (!(mode & SCATTER_LOCAL)) {

    if (gen_to->sendfirst) {
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

    if (!gen_from->use_readyreceiver) {  
      /* post receives since they were not posted in VecScatterPostRecvs()   */
      ierr = MPI_Startall_irecv(gen_from->starts[nrecvs]*gen_from->bs,nrecvs,rwaits);CHKERRQ(ierr);
    }

    if (!gen_to->sendfirst) {
      /* this version packs all the messages together and sends */
      /*
      len  = 5*sstarts[nsends];
      val  = svalues;
      for (i=0; i<len; i += 5) {
        idx     = *indices++;
        val[0] = xv[idx];
        val[1] = xv[idx+1];
        val[2] = xv[idx+2];
        val[3] = xv[idx+3];
        val[4] = xv[idx+4];
        val      += 5;
      }
      ierr = MPI_Startall_isend(len,nsends,swaits);CHKERRQ(ierr);
      */

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
  }

  /* take care of local scatters */
  if (gen_to->local.n) {
    int *tslots = gen_to->local.slots,*fslots = gen_from->local.slots;
    int n       = gen_to->local.n,il,ir;
    if (addv == INSERT_VALUES) {
      if (gen_to->local.is_copy) {
        ierr = PetscMemcpy(yv + gen_from->local.copy_start,xv + gen_to->local.copy_start,gen_to->local.copy_length);CHKERRQ(ierr);
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
    }  else {SETERRQ(1,"Wrong insert option");}
  }
  ierr = VecRestoreArray(xin,&xv);CHKERRQ(ierr);
  if (xin != yin) {ierr = VecRestoreArray(yin,&yv);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

/* --------------------------------------------------------------------------------------*/

#undef __FUNCT__  
#define __FUNCT__ "VecScatterEnd_PtoP_5"
int VecScatterEnd_PtoP_5(Vec xin,Vec yin,InsertMode addv,ScatterMode mode,VecScatter ctx)
{
  VecScatter_MPI_General *gen_to,*gen_from;
  PetscScalar            *rvalues,*yv,*val;
  int                    ierr,nrecvs,nsends,i,*indices,count,imdex,n,*rstarts,*lindices,idx;
  MPI_Request            *rwaits,*swaits;
  MPI_Status             rstatus,*sstatus;

  PetscFunctionBegin;
  if (mode & SCATTER_LOCAL) PetscFunctionReturn(0);
  ierr = VecGetArray(yin,&yv);CHKERRQ(ierr);

  if (mode & SCATTER_REVERSE) {
    gen_to   = (VecScatter_MPI_General*)ctx->fromdata;
    gen_from = (VecScatter_MPI_General*)ctx->todata;
    rwaits   = gen_from->rev_requests;
    swaits   = gen_to->rev_requests;
    sstatus  = gen_from->sstatus;
  } else {
    gen_to   = (VecScatter_MPI_General*)ctx->todata;
    gen_from = (VecScatter_MPI_General*)ctx->fromdata;
    rwaits   = gen_from->requests;
    swaits   = gen_to->requests;
    sstatus  = gen_to->sstatus;
  }
  rvalues  = gen_from->values;
  nrecvs   = gen_from->n;
  nsends   = gen_to->n;
  indices  = gen_from->indices;
  rstarts  = gen_from->starts;

  /*  wait on receives */
  count = nrecvs;
  while (count) {
    ierr = MPI_Waitany(nrecvs,rwaits,&imdex,&rstatus);CHKERRQ(ierr);
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
    }  else {SETERRQ(1,"Wrong insert option");}
    count--;
  }
  /* wait on sends */
  if (nsends) {
    ierr = MPI_Waitall(nsends,swaits,sstatus);CHKERRQ(ierr);
  }
  ierr = VecRestoreArray(yin,&yv);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* --------------------------------------------------------------------------------------*/

#undef __FUNCT__  
#define __FUNCT__ "VecScatterBegin_PtoP_4"
int VecScatterBegin_PtoP_4(Vec xin,Vec yin,InsertMode addv,ScatterMode mode,VecScatter ctx)
{
  VecScatter_MPI_General *gen_to,*gen_from;
  PetscScalar            *xv,*yv,*val,*svalues;
  MPI_Request            *rwaits,*swaits;
  int                    *indices,*sstarts,iend,i,j,nrecvs,nsends,idx,ierr,len;

  PetscFunctionBegin;
  ierr = VecGetArray(xin,&xv);CHKERRQ(ierr);
  if (xin != yin) {ierr = VecGetArray(yin,&yv);CHKERRQ(ierr);} else {yv = xv;}

  if (mode & SCATTER_REVERSE) {
    gen_to   = (VecScatter_MPI_General*)ctx->fromdata;
    gen_from = (VecScatter_MPI_General*)ctx->todata;
    rwaits   = gen_from->rev_requests;
    swaits   = gen_to->rev_requests;
  } else {
    gen_to   = (VecScatter_MPI_General*)ctx->todata;
    gen_from = (VecScatter_MPI_General*)ctx->fromdata;
    rwaits   = gen_from->requests;
    swaits   = gen_to->requests;
  }
  svalues  = gen_to->values;
  nrecvs   = gen_from->n;
  nsends   = gen_to->n;
  indices  = gen_to->indices;
  sstarts  = gen_to->starts;

  if (!(mode & SCATTER_LOCAL)) {

    if (!gen_from->use_readyreceiver && !gen_to->sendfirst) {  
      /* post receives since they were not posted in VecScatterPostRecvs()   */
      ierr = MPI_Startall_irecv(gen_from->starts[nrecvs]*gen_from->bs,nrecvs,rwaits);CHKERRQ(ierr);
    }

    if (ctx->packtogether) {
      /* this version packs all the messages together and sends, when -vecscatter_packtogether used */
      len  = 4*sstarts[nsends];
      val  = svalues;
      for (i=0; i<len; i += 4) {
        idx    = *indices++;
        val[0] = xv[idx];
        val[1] = xv[idx+1];
        val[2] = xv[idx+2];
        val[3] = xv[idx+3];
        val    += 4;
      }
      ierr = MPI_Startall_isend(len,nsends,swaits);CHKERRQ(ierr);
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

    if (!gen_from->use_readyreceiver && gen_to->sendfirst) {  
      /* post receives since they were not posted in VecScatterPostRecvs()   */
      ierr = MPI_Startall_irecv(gen_from->starts[nrecvs]*gen_from->bs,nrecvs,rwaits);CHKERRQ(ierr);
    }
  }

  /* take care of local scatters */
  if (gen_to->local.n) {
    int *tslots = gen_to->local.slots,*fslots = gen_from->local.slots;
    int n       = gen_to->local.n,il,ir;
    if (addv == INSERT_VALUES) {
      if (gen_to->local.is_copy) {
        ierr = PetscMemcpy(yv+gen_from->local.copy_start,xv+gen_to->local.copy_start,gen_to->local.copy_length);CHKERRQ(ierr);
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
    }  else {SETERRQ(1,"Wrong insert option");}
  }
  ierr = VecRestoreArray(xin,&xv);CHKERRQ(ierr);
  if (xin != yin) {ierr = VecRestoreArray(yin,&yv);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

/* --------------------------------------------------------------------------------------*/

#undef __FUNCT__  
#define __FUNCT__ "VecScatterEnd_PtoP_4"
int VecScatterEnd_PtoP_4(Vec xin,Vec yin,InsertMode addv,ScatterMode mode,VecScatter ctx)
{
  VecScatter_MPI_General *gen_to,*gen_from;
  PetscScalar            *rvalues,*yv,*val;
  int                    ierr,nrecvs,nsends,i,*indices,count,imdex,n,*rstarts,*lindices,idx;
  MPI_Request            *rwaits,*swaits;
  MPI_Status             *rstatus,*sstatus;

  PetscFunctionBegin;
  if (mode & SCATTER_LOCAL) PetscFunctionReturn(0);
  ierr = VecGetArray(yin,&yv);CHKERRQ(ierr);

  if (mode & SCATTER_REVERSE) {
    gen_to   = (VecScatter_MPI_General*)ctx->fromdata;
    gen_from = (VecScatter_MPI_General*)ctx->todata;
    rwaits   = gen_from->rev_requests;
    swaits   = gen_to->rev_requests;
    sstatus  = gen_from->sstatus;
    rstatus  = gen_from->rstatus;
  } else {
    gen_to   = (VecScatter_MPI_General*)ctx->todata;
    gen_from = (VecScatter_MPI_General*)ctx->fromdata;
    rwaits   = gen_from->requests;
    swaits   = gen_to->requests;
    sstatus  = gen_to->sstatus;
    rstatus  = gen_to->rstatus;
  }
  rvalues  = gen_from->values;
  nrecvs   = gen_from->n;
  nsends   = gen_to->n;
  indices  = gen_from->indices;
  rstarts  = gen_from->starts;

  /*  wait on receives */
  count = nrecvs;
  if (ctx->packtogether) { /* receive all messages, then unpack all, when -vecscatter_packtogether used */
    ierr     = MPI_Waitall(nrecvs,rwaits,rstatus);CHKERRQ(ierr);
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
    }  else {SETERRQ(1,"Wrong insert option");}
  } else { /* unpack each message as it arrives, default version */
    while (count) {
      ierr = MPI_Waitany(nrecvs,rwaits,&imdex,&rstatus[0]);CHKERRQ(ierr);
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
      }  else {SETERRQ(1,"Wrong insert option");}
      count--;
    }
  }

  /* wait on sends */
  if (nsends) {
    ierr = MPI_Waitall(nsends,swaits,sstatus);CHKERRQ(ierr);
  }
  ierr = VecRestoreArray(yin,&yv);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* --------------------------------------------------------------------------------------*/

#undef __FUNCT__  
#define __FUNCT__ "VecScatterBegin_PtoP_3"
int VecScatterBegin_PtoP_3(Vec xin,Vec yin,InsertMode addv,ScatterMode mode,VecScatter ctx)
{
  VecScatter_MPI_General *gen_to,*gen_from;
  PetscScalar            *xv,*yv,*val,*svalues;
  MPI_Request            *rwaits,*swaits;
  int                    ierr,i,*indices,*sstarts,iend,j,nrecvs,nsends,idx;

  PetscFunctionBegin;
  ierr = VecGetArray(xin,&xv);CHKERRQ(ierr);
  if (xin != yin) {ierr = VecGetArray(yin,&yv);CHKERRQ(ierr);} else {yv = xv;}

  if (mode & SCATTER_REVERSE) {
    gen_to   = (VecScatter_MPI_General*)ctx->fromdata;
    gen_from = (VecScatter_MPI_General*)ctx->todata;
    rwaits   = gen_from->rev_requests;
    swaits   = gen_to->rev_requests;
  } else {
    gen_to   = (VecScatter_MPI_General*)ctx->todata;
    gen_from = (VecScatter_MPI_General*)ctx->fromdata;
    rwaits   = gen_from->requests;
    swaits   = gen_to->requests;
  }
  svalues  = gen_to->values;
  nrecvs   = gen_from->n;
  nsends   = gen_to->n;
  indices  = gen_to->indices;
  sstarts  = gen_to->starts;

  if (!(mode & SCATTER_LOCAL)) {

    if (gen_to->sendfirst) {
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

    if (!gen_from->use_readyreceiver) {  
      /* post receives since they were not posted in VecScatterPostRecvs()   */
      ierr = MPI_Startall_irecv(gen_from->starts[nrecvs]*gen_from->bs,nrecvs,rwaits);CHKERRQ(ierr);
    }

    if (!gen_to->sendfirst) {
      /* this version packs all the messages together and sends */
      /*
      len  = 3*sstarts[nsends];
      val  = svalues;
      for (i=0; i<len; i += 3) {
        idx     = *indices++;
        val[0] = xv[idx];
        val[1] = xv[idx+1];
        val[2] = xv[idx+2];
        val      += 3;
      }
      ierr = MPI_Startall_isend(len,nsends,swaits);CHKERRQ(ierr);
      */

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
  }

  /* take care of local scatters */
  if (gen_to->local.n) {
    int *tslots = gen_to->local.slots,*fslots = gen_from->local.slots;
    int n       = gen_to->local.n,il,ir;
    if (addv == INSERT_VALUES) {
      if (gen_to->local.is_copy) {
        ierr = PetscMemcpy(yv + gen_from->local.copy_start,xv + gen_to->local.copy_start,gen_to->local.copy_length);CHKERRQ(ierr);
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
    }  else {SETERRQ(1,"Wrong insert option");}
  }
  ierr = VecRestoreArray(xin,&xv);CHKERRQ(ierr);
  if (xin != yin) {ierr = VecRestoreArray(yin,&yv);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

/* --------------------------------------------------------------------------------------*/

#undef __FUNCT__  
#define __FUNCT__ "VecScatterEnd_PtoP_3"
int VecScatterEnd_PtoP_3(Vec xin,Vec yin,InsertMode addv,ScatterMode mode,VecScatter ctx)
{
  VecScatter_MPI_General *gen_to,*gen_from;
  PetscScalar            *rvalues,*yv,*val;
  int                    ierr,nrecvs,nsends,i,*indices,count,imdex,n,*rstarts,*lindices,idx;
  MPI_Request            *rwaits,*swaits;
  MPI_Status             rstatus,*sstatus;

  PetscFunctionBegin;
  if (mode & SCATTER_LOCAL) PetscFunctionReturn(0);
  ierr = VecGetArray(yin,&yv);CHKERRQ(ierr);

  if (mode & SCATTER_REVERSE) {
    gen_to   = (VecScatter_MPI_General*)ctx->fromdata;
    gen_from = (VecScatter_MPI_General*)ctx->todata;
    rwaits   = gen_from->rev_requests;
    swaits   = gen_to->rev_requests;
    sstatus  = gen_from->sstatus;
  } else {
    gen_to   = (VecScatter_MPI_General*)ctx->todata;
    gen_from = (VecScatter_MPI_General*)ctx->fromdata;
    rwaits   = gen_from->requests;
    swaits   = gen_to->requests;
    sstatus  = gen_to->sstatus;
  }
  rvalues  = gen_from->values;
  nrecvs   = gen_from->n;
  nsends   = gen_to->n;
  indices  = gen_from->indices;
  rstarts  = gen_from->starts;

  /*  wait on receives */
  count = nrecvs;
  while (count) {
    ierr = MPI_Waitany(nrecvs,rwaits,&imdex,&rstatus);CHKERRQ(ierr);
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
    }  else {SETERRQ(1,"Wrong insert option");}
    count--;
  }
  /* wait on sends */
  if (nsends) {
    ierr = MPI_Waitall(nsends,swaits,sstatus);CHKERRQ(ierr);
  }
  ierr = VecRestoreArray(yin,&yv);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* --------------------------------------------------------------------------------------*/

#undef __FUNCT__  
#define __FUNCT__ "VecScatterBegin_PtoP_2"
int VecScatterBegin_PtoP_2(Vec xin,Vec yin,InsertMode addv,ScatterMode mode,VecScatter ctx)
{
  VecScatter_MPI_General *gen_to,*gen_from;
  PetscScalar            *xv,*yv,*val,*svalues;
  MPI_Request            *rwaits,*swaits;
  int                    ierr,i,*indices,*sstarts,iend,j,nrecvs,nsends,idx;

  PetscFunctionBegin;
  ierr = VecGetArray(xin,&xv);CHKERRQ(ierr);
  if (xin != yin) {ierr = VecGetArray(yin,&yv);CHKERRQ(ierr);} else {yv = xv;}
  if (mode & SCATTER_REVERSE) {
    gen_to   = (VecScatter_MPI_General*)ctx->fromdata;
    gen_from = (VecScatter_MPI_General*)ctx->todata;
    rwaits   = gen_from->rev_requests;
    swaits   = gen_to->rev_requests;
  } else {
    gen_to   = (VecScatter_MPI_General*)ctx->todata;
    gen_from = (VecScatter_MPI_General*)ctx->fromdata;
    rwaits   = gen_from->requests;
    swaits   = gen_to->requests;
  }
  svalues  = gen_to->values;
  nrecvs   = gen_from->n;
  nsends   = gen_to->n;
  indices  = gen_to->indices;
  sstarts  = gen_to->starts;

  if (!(mode & SCATTER_LOCAL)) {

    if (gen_to->sendfirst) {
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

    if (!gen_from->use_readyreceiver) {  
      /* post receives since they were not posted in VecScatterPostRecvs()   */
      ierr = MPI_Startall_irecv(gen_from->starts[nrecvs]*gen_from->bs,nrecvs,rwaits);CHKERRQ(ierr);
    }

    if (!gen_to->sendfirst) {
      /* this version packs all the messages together and sends */
      /*
      len  = 2*sstarts[nsends];
      val  = svalues;
      for (i=0; i<len; i += 2) {
        idx     = *indices++;
        val[0] = xv[idx];
        val[1] = xv[idx+1];
        val      += 2;
      }
      ierr = MPI_Startall_isend(len,nsends,swaits);CHKERRQ(ierr);
      */

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
  }

  /* take care of local scatters */
  if (gen_to->local.n) {
    int *tslots = gen_to->local.slots,*fslots = gen_from->local.slots;
    int n       = gen_to->local.n,il,ir;
    if (addv == INSERT_VALUES) {
      if (gen_to->local.is_copy) {
        ierr = PetscMemcpy(yv + gen_from->local.copy_start,xv + gen_to->local.copy_start,gen_to->local.copy_length);CHKERRQ(ierr);
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
    }  else {SETERRQ(1,"Wrong insert option");}
  }
  ierr = VecRestoreArray(xin,&xv);CHKERRQ(ierr);
  if (xin != yin) {ierr = VecRestoreArray(yin,&yv);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

/* --------------------------------------------------------------------------------------*/

#undef __FUNCT__  
#define __FUNCT__ "VecScatterEnd_PtoP_2"
int VecScatterEnd_PtoP_2(Vec xin,Vec yin,InsertMode addv,ScatterMode mode,VecScatter ctx)
{
  VecScatter_MPI_General *gen_to,*gen_from;
  PetscScalar            *rvalues,*yv,*val;
  int                    ierr,nrecvs,nsends,i,*indices,count,imdex,n,*rstarts,*lindices,idx;
  MPI_Request            *rwaits,*swaits;
  MPI_Status             rstatus,*sstatus;

  PetscFunctionBegin;
  if (mode & SCATTER_LOCAL) PetscFunctionReturn(0);
  ierr = VecGetArray(yin,&yv);CHKERRQ(ierr);

  if (mode & SCATTER_REVERSE) {
    gen_to   = (VecScatter_MPI_General*)ctx->fromdata;
    gen_from = (VecScatter_MPI_General*)ctx->todata;
    rwaits   = gen_from->rev_requests;
    swaits   = gen_to->rev_requests;
    sstatus  = gen_from->sstatus;
  } else {
    gen_to   = (VecScatter_MPI_General*)ctx->todata;
    gen_from = (VecScatter_MPI_General*)ctx->fromdata;
    rwaits   = gen_from->requests;
    swaits   = gen_to->requests;
    sstatus  = gen_to->sstatus;
  }
  rvalues  = gen_from->values;
  nrecvs   = gen_from->n;
  nsends   = gen_to->n;
  indices  = gen_from->indices;
  rstarts  = gen_from->starts;

  /*  wait on receives */
  count = nrecvs;
  while (count) {
    ierr = MPI_Waitany(nrecvs,rwaits,&imdex,&rstatus);CHKERRQ(ierr);
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
    }  else {SETERRQ(1,"Wrong insert option");}
    count--;
  }
  /* wait on sends */
  if (nsends) {
    ierr = MPI_Waitall(nsends,swaits,sstatus);CHKERRQ(ierr);
  }
  ierr = VecRestoreArray(yin,&yv);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------------------------*/

#undef __FUNCT__  
#define __FUNCT__ "VecScatterDestroy_PtoP_X"
int VecScatterDestroy_PtoP_X(VecScatter ctx)
{
  VecScatter_MPI_General *gen_to   = (VecScatter_MPI_General*)ctx->todata;
  VecScatter_MPI_General *gen_from = (VecScatter_MPI_General*)ctx->fromdata;
  int                    i,ierr;

  PetscFunctionBegin;
  if (gen_to->use_readyreceiver) {
    /*
       Since we have already posted sends we must cancel them before freeing 
       the requests
    */
    for (i=0; i<gen_from->n; i++) {
      ierr = MPI_Cancel(gen_from->requests+i);CHKERRQ(ierr);
    }
  }

  if (gen_to->local.slots) {ierr = PetscFree(gen_to->local.slots);CHKERRQ(ierr);}
  if (gen_from->local.slots) {ierr = PetscFree(gen_from->local.slots);CHKERRQ(ierr);}
  if (gen_to->local.slots_nonmatching) {ierr = PetscFree(gen_to->local.slots_nonmatching);CHKERRQ(ierr);}
  if (gen_from->local.slots_nonmatching) {ierr = PetscFree(gen_from->local.slots_nonmatching);CHKERRQ(ierr);}

  /* release MPI resources obtained with MPI_Send_init() and MPI_Recv_init() */
  /* 
     IBM's PE version of MPI has a bug where freeing these guys will screw up later
     message passing.
  */
#if !defined(HAVE_BROKEN_REQUEST_FREE)
  for (i=0; i<gen_to->n; i++) {
    ierr = MPI_Request_free(gen_to->requests + i);CHKERRQ(ierr);
    ierr = MPI_Request_free(gen_to->rev_requests + i);CHKERRQ(ierr);
  }

  /*
      MPICH could not properly cancel requests thus with ready receiver mode we
    cannot free the requests. It may be fixed now, if not then put the following 
    code inside a if !gen_to->use_readyreceiver) {
  */
  for (i=0; i<gen_from->n; i++) {
    ierr = MPI_Request_free(gen_from->requests + i);CHKERRQ(ierr);
    ierr = MPI_Request_free(gen_from->rev_requests + i);CHKERRQ(ierr);
  }  
#endif
 
  ierr = PetscFree(gen_to->sstatus);CHKERRQ(ierr);
  ierr = PetscFree(gen_to->values);CHKERRQ(ierr);
  ierr = PetscFree(gen_to->rev_requests);CHKERRQ(ierr);
  ierr = PetscFree(gen_to);CHKERRQ(ierr);
  ierr = PetscFree(gen_from->values);CHKERRQ(ierr);
  ierr = PetscFree(gen_from->rev_requests);CHKERRQ(ierr);
  ierr = PetscFree(gen_from);CHKERRQ(ierr);
  PetscLogObjectDestroy(ctx);
  PetscHeaderDestroy(ctx);
  PetscFunctionReturn(0);
}

/* ==========================================================================================*/

/*              create parallel to sequential scatter context                           */
/*
   bs indicates how many elements there are in each block. Normally
   this would be 1.
*/
#undef __FUNCT__  
#define __FUNCT__ "VecScatterCreate_PtoS"
int VecScatterCreate_PtoS(int nx,int *inidx,int ny,int *inidy,Vec xin,Vec yin,int bs,VecScatter ctx)
{
  VecScatter_MPI_General *from,*to;
  int                    *source,*lens,rank,*owners;
  int                    size,*lowner,*start,lengthy;
  int                    *nprocs,i,j,n,idx,*procs,nsends,nrecvs,*work;
  int                    *owner,*starts,count,tag,slen,ierr;
  int                    *rvalues,*svalues,base,imdex,nmax,*values,len,*indx,nprocslocal;
  MPI_Comm               comm;
  MPI_Request            *send_waits,*recv_waits;
  MPI_Status             recv_status,*send_status;
  PetscMap               map;
  PetscTruth             found;
  
  PetscFunctionBegin;
  ierr = PetscObjectGetNewTag((PetscObject)ctx,&tag);CHKERRQ(ierr);
  ierr = PetscObjectGetComm((PetscObject)xin,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  ierr = VecGetPetscMap(xin,&map);CHKERRQ(ierr);
  ierr = PetscMapGetGlobalRange(map,&owners);CHKERRQ(ierr);

  ierr = VecGetSize(yin,&lengthy);CHKERRQ(ierr);

  /*  first count number of contributors to each processor */
  ierr = PetscMalloc(2*size*sizeof(int),&nprocs);CHKERRQ(ierr);
  ierr   = PetscMemzero(nprocs,2*size*sizeof(int));CHKERRQ(ierr);
  procs  = nprocs + size;
  ierr = PetscMalloc((nx+1)*sizeof(int),&owner);CHKERRQ(ierr);
  for (i=0; i<nx; i++) {
    idx = inidx[i];
    found = PETSC_FALSE;
    for (j=0; j<size; j++) {
      if (idx >= owners[j] && idx < owners[j+1]) {
        nprocs[j]++; procs[j] = 1; owner[i] = j; found = PETSC_TRUE; break;
      }
    }
    if (!found) SETERRQ1(PETSC_ERR_ARG_OUTOFRANGE,"Index %d out of range",idx);
  }
  nprocslocal  = nprocs[rank]; 
  nprocs[rank] = procs[rank] = 0; 
  nsends       = 0;  for (i=0; i<size; i++) { nsends += procs[i];} 

  /* inform other processors of number of messages and max length*/
  ierr = PetscMalloc(2*size*sizeof(int),&work);CHKERRQ(ierr);
  ierr   = MPI_Allreduce(nprocs,work,2*size,MPI_INT,PetscMaxSum_Op,comm);CHKERRQ(ierr);
  nmax   = work[rank];
  nrecvs = work[size+rank]; 
  ierr   = PetscFree(work);CHKERRQ(ierr);

  /* post receives:   */
  ierr = PetscMalloc((nrecvs+1)*(nmax+1)*sizeof(int),&rvalues);CHKERRQ(ierr);
  ierr = PetscMalloc((nrecvs+1)*sizeof(MPI_Request),&recv_waits);CHKERRQ(ierr);
  for (i=0; i<nrecvs; i++) {
    ierr = MPI_Irecv((rvalues+nmax*i),nmax,MPI_INT,MPI_ANY_SOURCE,tag,comm,recv_waits+i);CHKERRQ(ierr);
  }

  /* do sends:
      1) starts[i] gives the starting index in svalues for stuff going to 
         the ith processor
  */
  ierr = PetscMalloc((nx+1)*sizeof(int),&svalues);CHKERRQ(ierr);
  ierr = PetscMalloc((nsends+1)*sizeof(MPI_Request),&send_waits);CHKERRQ(ierr);
  ierr = PetscMalloc((size+1)*sizeof(int),&starts);CHKERRQ(ierr);
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
    if (procs[i]) {
      ierr = MPI_Isend(svalues+starts[i],nprocs[i],MPI_INT,i,tag,comm,send_waits+count++);CHKERRQ(ierr);
    }
  }
  ierr = PetscFree(starts);CHKERRQ(ierr);

  base = owners[rank];

  /*  wait on receives */
  ierr   = PetscMalloc(2*(nrecvs+1)*sizeof(int),&lens);CHKERRQ(ierr);
  source = lens + nrecvs;
  count  = nrecvs; 
  slen   = 0;
  while (count) {
    ierr = MPI_Waitany(nrecvs,recv_waits,&imdex,&recv_status);CHKERRQ(ierr);
    /* unpack receives into our local space */
    ierr = MPI_Get_count(&recv_status,MPI_INT,&n);CHKERRQ(ierr);
    source[imdex]  = recv_status.MPI_SOURCE;
    lens[imdex]    = n;
    slen          += n;
    count--;
  }
  ierr = PetscFree(recv_waits);CHKERRQ(ierr);
  
  /* allocate entire send scatter context */
  ierr = PetscNew(VecScatter_MPI_General,&to);CHKERRQ(ierr);
  ierr = PetscOptionsHasName(PETSC_NULL,"-vecscatter_sendfirst",&to->sendfirst);CHKERRQ(ierr);
  ierr = PetscMemzero(to,sizeof(VecScatter_MPI_General));CHKERRQ(ierr);
  PetscLogObjectMemory(ctx,sizeof(VecScatter_MPI_General));
  len = slen*sizeof(int) + bs*slen*sizeof(PetscScalar) + (nrecvs+1)*sizeof(int) +
        nrecvs*(sizeof(int) + sizeof(MPI_Request));
  to->n         = nrecvs; 
  ierr = PetscMalloc(len,&to->values);CHKERRQ(ierr);
  PetscLogObjectMemory(ctx,len);
  to->requests  = (MPI_Request*)(to->values + bs*slen);
  to->indices   = (int*)(to->requests + nrecvs); 
  to->starts    = (int*)(to->indices + slen);
  to->procs     = (int*)(to->starts + nrecvs + 1);
  ierr          = PetscMalloc(2*(PetscMax(nrecvs,nsends)+1)*sizeof(MPI_Status),&to->sstatus);CHKERRQ(ierr);
  to->rstatus   = to->sstatus + PetscMax(nrecvs,nsends) + 1;
  PetscLogObjectMemory(ctx,2*(PetscMax(nrecvs,nsends)+1)*sizeof(MPI_Status));
  ctx->todata   = (void*)to;
  to->starts[0] = 0;

  if (nrecvs) {
    ierr = PetscMalloc(nrecvs*sizeof(int),&indx);CHKERRQ(ierr);
    for (i=0; i<nrecvs; i++) indx[i] = i;
    ierr = PetscSortIntWithPermutation(nrecvs,source,indx);CHKERRQ(ierr);

    /* move the data into the send scatter */
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
  ierr = PetscFree(rvalues);CHKERRQ(ierr);
  ierr = PetscFree(lens);CHKERRQ(ierr);
 
  /* allocate entire receive scatter context */
  ierr = PetscNew(VecScatter_MPI_General,&from);CHKERRQ(ierr);
  ierr = PetscOptionsHasName(PETSC_NULL,"-vecscatter_sendfirst",&from->sendfirst);CHKERRQ(ierr);
  ierr = PetscMemzero(from,sizeof(VecScatter_MPI_General));CHKERRQ(ierr);
  PetscLogObjectMemory(ctx,sizeof(VecScatter_MPI_General));
  len = ny*sizeof(int) + ny*bs*sizeof(PetscScalar) + (nsends+1)*sizeof(int) +
        nsends*(sizeof(int) + sizeof(MPI_Request));
  from->n        = nsends;
  ierr = PetscMalloc(len,&from->values);CHKERRQ(ierr);
  PetscLogObjectMemory(ctx,len);
  from->requests = (MPI_Request*)(from->values + bs*ny);
  from->indices  = (int*)(from->requests + nsends); 
  from->starts   = (int*)(from->indices + ny);
  from->procs    = (int*)(from->starts + nsends + 1);
  ctx->fromdata  = (void*)from;

  /* move data into receive scatter */
  ierr = PetscMalloc((size+nsends+1)*sizeof(int),&lowner);CHKERRQ(ierr);
  start = lowner + size;
  count = 0; from->starts[0] = start[0] = 0;
  for (i=0; i<size; i++) {
    if (procs[i]) {
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
  ierr = PetscFree(lowner);CHKERRQ(ierr);
  ierr = PetscFree(owner);CHKERRQ(ierr);
  ierr = PetscFree(nprocs);CHKERRQ(ierr);
    
  /* wait on sends */
  if (nsends) {
    ierr = PetscMalloc(nsends*sizeof(MPI_Status),&send_status);CHKERRQ(ierr);
    ierr = MPI_Waitall(nsends,send_waits,send_status);CHKERRQ(ierr);
    ierr = PetscFree(send_status);CHKERRQ(ierr);
  }
  ierr = PetscFree(send_waits);CHKERRQ(ierr);
  ierr = PetscFree(svalues);CHKERRQ(ierr);

  if (nprocslocal) {
    int nt;
    /* we have a scatter to ourselves */
    from->local.n     = to->local.n = nt = nprocslocal;    
    ierr = PetscMalloc(nt*sizeof(int),&from->local.slots);CHKERRQ(ierr);
    ierr = PetscMalloc(nt*sizeof(int),&to->local.slots);CHKERRQ(ierr);
    PetscLogObjectMemory(ctx,2*nt*sizeof(int));
    nt = 0;
    for (i=0; i<nx; i++) {
      idx = inidx[i];
      if (idx >= owners[rank] && idx < owners[rank+1]) {
        to->local.slots[nt]     = idx - owners[rank];        
        from->local.slots[nt++] = inidy[i];        
        if (inidy[i] >= lengthy) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Scattering past end of TO vector");
      }
    }
  } else { 
    from->local.n     = 0;
    from->local.slots = 0;
    to->local.n       = 0; 
    to->local.slots   = 0;
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
  if (bs > 1) {
    PetscTruth  flg,flgs = PETSC_FALSE;
    int         *sstarts = to->starts,  *rstarts = from->starts;
    int         *sprocs  = to->procs,   *rprocs  = from->procs;
    MPI_Request *swaits  = to->requests,*rwaits  = from->requests;
    MPI_Request *rev_swaits,*rev_rwaits;
    PetscScalar *Ssvalues = to->values, *Srvalues = from->values;

    ctx->destroy   = VecScatterDestroy_PtoP_X;
    ctx->copy      = VecScatterCopy_PtoP_X;
    ctx->view      = VecScatterView_MPI;
  
    tag      = ctx->tag;
    comm     = ctx->comm;

    /* allocate additional wait variables for the "reverse" scatter */
    ierr = PetscMalloc((nrecvs+1)*sizeof(MPI_Request),&rev_rwaits);CHKERRQ(ierr);
    to->rev_requests   = rev_rwaits;
    ierr = PetscMalloc((nsends+1)*sizeof(MPI_Request),&rev_swaits);CHKERRQ(ierr);
    from->rev_requests = rev_swaits;
    PetscLogObjectMemory(ctx,(nsends+nrecvs+2)*sizeof(MPI_Request));

    /* Register the receives that you will use later (sends for scatter reverse) */
    ierr = PetscOptionsHasName(PETSC_NULL,"-vecscatter_ssend",&flgs);CHKERRQ(ierr);
    if (flgs) {
      PetscLogInfo(0,"VecScatterCreate_PtoS:Using VecScatter Ssend mode\n");
    }
    for (i=0; i<from->n; i++) {
      ierr = MPI_Recv_init(Srvalues+bs*rstarts[i],bs*rstarts[i+1]-bs*rstarts[i],MPIU_SCALAR,rprocs[i],tag,
                           comm,rwaits+i);CHKERRQ(ierr);
      if (!flgs) {
        ierr = MPI_Send_init(Srvalues+bs*rstarts[i],bs*rstarts[i+1]-bs*rstarts[i],MPIU_SCALAR,rprocs[i],tag,
                             comm,rev_swaits+i);CHKERRQ(ierr);
      } else {
        ierr = MPI_Ssend_init(Srvalues+bs*rstarts[i],bs*rstarts[i+1]-bs*rstarts[i],MPIU_SCALAR,rprocs[i],tag,
                              comm,rev_swaits+i);CHKERRQ(ierr);
      }
    }

    ierr = PetscOptionsHasName(PETSC_NULL,"-vecscatter_rr",&flg);CHKERRQ(ierr);
    if (flg) {
      ctx->postrecvs           = VecScatterPostRecvs_PtoP_X;
      to->use_readyreceiver    = PETSC_TRUE;
      from->use_readyreceiver  = PETSC_TRUE;
      for (i=0; i<to->n; i++) {
        ierr = MPI_Rsend_init(Ssvalues+bs*sstarts[i],bs*sstarts[i+1]-bs*sstarts[i],MPIU_SCALAR,sprocs[i],tag,
                              comm,swaits+i);CHKERRQ(ierr);
      } 
      PetscLogInfo(0,"VecScatterCreate_PtoS:Using VecScatter ready receiver mode\n");
    } else {
      ctx->postrecvs           = 0;
      to->use_readyreceiver    = PETSC_FALSE;
      from->use_readyreceiver  = PETSC_FALSE;
      for (i=0; i<to->n; i++) {
        if (!flgs) {
          ierr = MPI_Send_init(Ssvalues+bs*sstarts[i],bs*sstarts[i+1]-bs*sstarts[i],MPIU_SCALAR,sprocs[i],tag,
                               comm,swaits+i);CHKERRQ(ierr);
	} else {
          ierr = MPI_Ssend_init(Ssvalues+bs*sstarts[i],bs*sstarts[i+1]-bs*sstarts[i],MPIU_SCALAR,sprocs[i],tag,
                               comm,swaits+i);CHKERRQ(ierr);
        }
      } 
    }
    /* Register receives for scatter reverse */
    for (i=0; i<to->n; i++) {
      ierr = MPI_Recv_init(Ssvalues+bs*sstarts[i],bs*sstarts[i+1]-bs*sstarts[i],MPIU_SCALAR,sprocs[i],tag,
                           comm,rev_rwaits+i);CHKERRQ(ierr);
    } 

    PetscLogInfo(0,"VecScatterCreate_PtoS:Using blocksize %d scatter\n",bs);
    switch (bs) {
    case 12: 
      ctx->begin     = VecScatterBegin_PtoP_12;
      ctx->end       = VecScatterEnd_PtoP_12; 
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
    ctx->postrecvs = 0;
    ctx->destroy   = VecScatterDestroy_PtoP;
    ctx->begin     = VecScatterBegin_PtoP;
    ctx->end       = VecScatterEnd_PtoP; 
    ctx->copy      = VecScatterCopy_PtoP;
    ctx->view      = VecScatterView_MPI;
  }

  /* Check if the local scatter is actually a copy; important special case */
  if (nprocslocal) { 
    ierr = VecScatterLocalOptimizeCopy_Private(&to->local,&from->local,bs);CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------------------------------*/
/*
         Scatter from local Seq vectors to a parallel vector. 
*/
#undef __FUNCT__  
#define __FUNCT__ "VecScatterCreate_StoP"
int VecScatterCreate_StoP(int nx,int *inidx,int ny,int *inidy,Vec yin,VecScatter ctx)
{
  VecScatter_MPI_General *from,*to;
  int                    *source,nprocslocal,*lens,rank = yin->stash.rank,*owners = yin->map->range;
  int                    ierr,size = yin->stash.size,*lowner,*start;
  int                    *nprocs,i,j,n,idx,*procs,nsends,nrecvs,*work;
  int                    *owner,*starts,count,tag,slen;
  int                    *rvalues,*svalues,base,imdex,nmax,*values,len;
  PetscTruth             found;
  MPI_Comm               comm = yin->comm;
  MPI_Request            *send_waits,*recv_waits;
  MPI_Status             recv_status,*send_status;

  PetscFunctionBegin;
  ierr = PetscObjectGetNewTag((PetscObject)ctx,&tag);CHKERRQ(ierr);
  /*  first count number of contributors to each processor */
  ierr = PetscMalloc(2*size*sizeof(int),&nprocs);CHKERRQ(ierr);
  ierr   = PetscMemzero(nprocs,2*size*sizeof(int));CHKERRQ(ierr);
  procs  = nprocs + size; 
  ierr = PetscMalloc((nx+1)*sizeof(int),&owner);CHKERRQ(ierr); 
  for (i=0; i<nx; i++) {
    idx = inidy[i];
    found = PETSC_FALSE;
    for (j=0; j<size; j++) {
      if (idx >= owners[j] && idx < owners[j+1]) {
        nprocs[j]++; procs[j] = 1; owner[i] = j; found = PETSC_TRUE; break;
      }
    }
    if (!found) SETERRQ1(PETSC_ERR_ARG_OUTOFRANGE,"Index %d out of range",idx);
  }
  nprocslocal  = nprocs[rank];
  nprocs[rank] = procs[rank] = 0; 
  nsends = 0;  for (i=0; i<size; i++) { nsends += procs[i];} 

  /* inform other processors of number of messages and max length*/
  ierr   = PetscMalloc(2*size*sizeof(int),&work);CHKERRQ(ierr);
  ierr   = MPI_Allreduce(nprocs,work,2*size,MPI_INT,PetscMaxSum_Op,comm);CHKERRQ(ierr);
  nmax   = work[rank];
  nrecvs = work[size+rank]; 
  ierr   = PetscFree(work);CHKERRQ(ierr);

  /* post receives:   */
  ierr = PetscMalloc((nrecvs+1)*(nmax+1)*sizeof(int),&rvalues);CHKERRQ(ierr);
  ierr = PetscMalloc((nrecvs+1)*sizeof(MPI_Request),&recv_waits);CHKERRQ(ierr);
  for (i=0; i<nrecvs; i++) {
    ierr = MPI_Irecv(rvalues+nmax*i,nmax,MPI_INT,MPI_ANY_SOURCE,tag,comm,recv_waits+i);CHKERRQ(ierr);
  }

  /* do sends:
      1) starts[i] gives the starting index in svalues for stuff going to 
         the ith processor
  */
  ierr = PetscMalloc((nx+1)*sizeof(int),&svalues);CHKERRQ(ierr);
  ierr = PetscMalloc((nsends+1)*sizeof(MPI_Request),&send_waits);CHKERRQ(ierr);
  ierr = PetscMalloc((size+1)*sizeof(int),&starts);CHKERRQ(ierr);
  starts[0]  = 0; 
  for (i=1; i<size; i++) { starts[i] = starts[i-1] + nprocs[i-1];} 
  for (i=0; i<nx; i++) {
    if (owner[i] != rank) {
      svalues[starts[owner[i]]++] = inidy[i];
    }
  }

  starts[0] = 0;
  for (i=1; i<size+1; i++) { starts[i] = starts[i-1] + nprocs[i-1];} 
  count = 0;
  for (i=0; i<size; i++) {
    if (procs[i]) {
      ierr = MPI_Isend(svalues+starts[i],nprocs[i],MPI_INT,i,tag,comm,send_waits+count);CHKERRQ(ierr);
      count++;
    }
  }
  ierr = PetscFree(starts);CHKERRQ(ierr);

  /* allocate entire send scatter context */
  ierr = PetscNew(VecScatter_MPI_General,&to);CHKERRQ(ierr);
  ierr = PetscMemzero(to,sizeof(VecScatter_MPI_General));CHKERRQ(ierr);
  PetscLogObjectMemory(ctx,sizeof(VecScatter_MPI_General));
  ierr = PetscOptionsHasName(PETSC_NULL,"-vecscatter_sendfirst",&to->sendfirst);CHKERRQ(ierr);
  len  = ny*(sizeof(int) + sizeof(PetscScalar)) + (nsends+1)*sizeof(int) +
        nsends*(sizeof(int) + sizeof(MPI_Request));
  to->n        = nsends; 
  ierr = PetscMalloc(len,&to->values);CHKERRQ(ierr); 
  PetscLogObjectMemory(ctx,len);
  to->requests = (MPI_Request*)(to->values + ny);
  to->indices  = (int*)(to->requests + nsends); 
  to->starts   = (int*)(to->indices + ny);
  to->procs    = (int*)(to->starts + nsends + 1);
  ierr         = PetscMalloc((PetscMax(nsends,nrecvs) + 1)*sizeof(MPI_Status),&to->sstatus);CHKERRQ(ierr);
  to->rstatus  = to->sstatus + PetscMax(nsends,nrecvs) + 1;
  PetscLogObjectMemory(ctx,2*(PetscMax(nsends,nrecvs) + 1)*sizeof(MPI_Status));
  ctx->todata  = (void*)to;

  /* move data into send scatter context */
  ierr = PetscMalloc((size+nsends+1)*sizeof(int),&lowner);CHKERRQ(ierr);
  start         = lowner + size;
  count         = 0;
  to->starts[0] = start[0] = 0;
  for (i=0; i<size; i++) {
    if (procs[i]) {
      lowner[i]          = count;
      to->procs[count++] = i;
      to->starts[count]  = start[count] = start[count-1] + nprocs[i];
    }
  }
  for (i=0; i<nx; i++) {
    if (owner[i] != rank) {
      to->indices[start[lowner[owner[i]]]++] = inidx[i];
    }
  }
  ierr = PetscFree(lowner);CHKERRQ(ierr);
  ierr = PetscFree(owner);CHKERRQ(ierr);
  ierr = PetscFree(nprocs);CHKERRQ(ierr);

  base = owners[rank];

  /*  wait on receives */
  ierr   = PetscMalloc(2*(nrecvs+1)*sizeof(int),&lens);CHKERRQ(ierr);
  source = lens + nrecvs;
  count  = nrecvs; slen = 0;
  while (count) {
    ierr = MPI_Waitany(nrecvs,recv_waits,&imdex,&recv_status);CHKERRQ(ierr);
    /* unpack receives into our local space */
    ierr = MPI_Get_count(&recv_status,MPI_INT,&n);CHKERRQ(ierr);
    source[imdex]  = recv_status.MPI_SOURCE;
    lens[imdex]    = n;
    slen          += n;
    count--;
  }
  ierr = PetscFree(recv_waits);CHKERRQ(ierr);
 
  /* allocate entire receive scatter context */
  ierr = PetscNew(VecScatter_MPI_General,&from);CHKERRQ(ierr);
  ierr = PetscMemzero(from,sizeof(VecScatter_MPI_General));CHKERRQ(ierr);
  PetscLogObjectMemory(ctx,sizeof(VecScatter_MPI_General));
  ierr = PetscOptionsHasName(PETSC_NULL,"-vecscatter_sendfirst",&from->sendfirst);CHKERRQ(ierr);
  len  = slen*(sizeof(int) + sizeof(PetscScalar)) + (nrecvs+1)*sizeof(int) +
          nrecvs*(sizeof(int) + sizeof(MPI_Request));
  from->n        = nrecvs; 
  ierr = PetscMalloc(len,&from->values);CHKERRQ(ierr);
  PetscLogObjectMemory(ctx,len);
  from->requests = (MPI_Request*)(from->values + slen);
  from->indices  = (int*)(from->requests + nrecvs); 
  from->starts   = (int*)(from->indices + slen);
  from->procs    = (int*)(from->starts + nrecvs + 1);
  ctx->fromdata  = (void*)from;

  /* move the data into the receive scatter context*/
  from->starts[0] = 0;
  for (i=0; i<nrecvs; i++) {
    from->starts[i+1] = from->starts[i] + lens[i];
    from->procs[i]    = source[i];
    values            = rvalues + i*nmax;
    for (j=0; j<lens[i]; j++) {
      from->indices[from->starts[i] + j] = values[j] - base;
    }
  }
  ierr = PetscFree(rvalues);CHKERRQ(ierr);
  ierr = PetscFree(lens);CHKERRQ(ierr);
    
  /* wait on sends */
  if (nsends) {
    ierr = PetscMalloc(nsends*sizeof(MPI_Status),&send_status);CHKERRQ(ierr);
    ierr = MPI_Waitall(nsends,send_waits,send_status);CHKERRQ(ierr);
    ierr = PetscFree(send_status);CHKERRQ(ierr);
  }
  ierr = PetscFree(send_waits);CHKERRQ(ierr);
  ierr = PetscFree(svalues);CHKERRQ(ierr);

  if (nprocslocal) {
    int nt;
    /* we have a scatter to ourselves */
    from->local.n     = to->local.n = nt = nprocslocal;    
    ierr = PetscMalloc(nt*sizeof(int),&from->local.slots);CHKERRQ(ierr);
    ierr = PetscMalloc(nt*sizeof(int),&to->local.slots);CHKERRQ(ierr);
    PetscLogObjectMemory(ctx,2*nt*sizeof(int));
    nt = 0;
    for (i=0; i<ny; i++) {
      idx = inidy[i];
      if (idx >= owners[rank] && idx < owners[rank+1]) {
        from->local.slots[nt] = idx - owners[rank];        
        to->local.slots[nt++] = inidx[i];        
      }
    }
  } else {
    from->local.n     = 0; 
    from->local.slots = 0;
    to->local.n       = 0;
    to->local.slots   = 0;

  }
  from->local.nonmatching_computed = PETSC_FALSE;
  from->local.n_nonmatching        = 0;
  from->local.slots_nonmatching    = 0;
  to->local.nonmatching_computed   = PETSC_FALSE;
  to->local.n_nonmatching          = 0;
  to->local.slots_nonmatching      = 0;

  to->type   = VEC_SCATTER_MPI_GENERAL; 
  from->type = VEC_SCATTER_MPI_GENERAL;

  ctx->destroy    = VecScatterDestroy_PtoP;
  ctx->postrecvs  = 0;
  ctx->begin      = VecScatterBegin_PtoP;
  ctx->end        = VecScatterEnd_PtoP; 
  ctx->copy       = 0;
  ctx->view       = VecScatterView_MPI;

  to->bs   = 1;
  from->bs = 1;
  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------------------------*/
#undef __FUNCT__  
#define __FUNCT__ "VecScatterCreate_PtoP"
int VecScatterCreate_PtoP(int nx,int *inidx,int ny,int *inidy,Vec xin,Vec yin,VecScatter ctx)
{
  int         *lens,rank,*owners = xin->map->range,size;
  int         *nprocs,i,j,n,idx,*procs,nsends,nrecvs,*work,*local_inidx,*local_inidy;
  int         *owner,*starts,count,tag,slen,ierr;
  int         *rvalues,*svalues,base,imdex,nmax,*values;
  MPI_Comm    comm;
  MPI_Request *send_waits,*recv_waits;
  MPI_Status  recv_status;
  PetscTruth  duplicate = PETSC_FALSE,found;

  PetscFunctionBegin;
  ierr = PetscObjectGetNewTag((PetscObject)ctx,&tag);CHKERRQ(ierr);
  ierr = PetscObjectGetComm((PetscObject)xin,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
  if (size == 1) {
    ierr = VecScatterCreate_StoP(nx,inidx,ny,inidy,yin,ctx);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }

  /*
     Each processor ships off its inidx[j] and inidy[j] to the appropriate processor
     They then call the StoPScatterCreate()
  */
  /*  first count number of contributors to each processor */
  ierr  = PetscMalloc(2*size*sizeof(int),&nprocs);CHKERRQ(ierr);
  procs = nprocs + size;
  ierr  = PetscMemzero(nprocs,2*size*sizeof(int));CHKERRQ(ierr);
  ierr  = PetscMalloc((nx+1)*sizeof(int),&owner);CHKERRQ(ierr);
  for (i=0; i<nx; i++) {
    idx   = inidx[i];
    found = PETSC_FALSE;
    for (j=0; j<size; j++) {
      if (idx >= owners[j] && idx < owners[j+1]) {
        nprocs[j]++; procs[j] = 1; owner[i] = j; found = PETSC_TRUE; break;
      }
    }
    if (!found) SETERRQ1(PETSC_ERR_ARG_OUTOFRANGE,"Index %d out of range",idx);
  }
  nsends = 0;  for (i=0; i<size; i++) { nsends += procs[i];} 

  /* inform other processors of number of messages and max length*/
  ierr = PetscMalloc(2*size*sizeof(int),&work);CHKERRQ(ierr);
  ierr   = MPI_Allreduce(nprocs,work,2*size,MPI_INT,PetscMaxSum_Op,comm);CHKERRQ(ierr);
  nmax   = work[rank];
  nrecvs = work[size+rank]; 
  ierr   = PetscFree(work);CHKERRQ(ierr);

  /* post receives:   */
  ierr = PetscMalloc(2*(nrecvs+1)*(nmax+1)*sizeof(int),&rvalues);CHKERRQ(ierr);
  ierr = PetscMalloc((nrecvs+1)*sizeof(MPI_Request),&recv_waits);CHKERRQ(ierr);
  for (i=0; i<nrecvs; i++) {
    ierr = MPI_Irecv(rvalues+2*nmax*i,2*nmax,MPI_INT,MPI_ANY_SOURCE,tag,comm,recv_waits+i);CHKERRQ(ierr);
  }

  /* do sends:
      1) starts[i] gives the starting index in svalues for stuff going to 
         the ith processor
  */
  ierr = PetscMalloc(2*(nx+1)*sizeof(int),&svalues);CHKERRQ(ierr);
  ierr = PetscMalloc((nsends+1)*sizeof(MPI_Request),&send_waits);CHKERRQ(ierr);
  ierr = PetscMalloc((size+1)*sizeof(int),&starts);CHKERRQ(ierr);
  starts[0]  = 0; 
  for (i=1; i<size; i++) { starts[i] = starts[i-1] + nprocs[i-1];} 
  for (i=0; i<nx; i++) {
    svalues[2*starts[owner[i]]]       = inidx[i];
    svalues[1 + 2*starts[owner[i]]++] = inidy[i];
  }
  ierr = PetscFree(owner);CHKERRQ(ierr);

  starts[0] = 0;
  for (i=1; i<size+1; i++) { starts[i] = starts[i-1] + nprocs[i-1];} 
  count = 0;
  for (i=0; i<size; i++) {
    if (procs[i]) {
      ierr = MPI_Isend(svalues+2*starts[i],2*nprocs[i],MPI_INT,i,tag,comm,send_waits+count);CHKERRQ(ierr);
      count++;
    }
  }
  ierr = PetscFree(starts);CHKERRQ(ierr);
  ierr = PetscFree(nprocs);CHKERRQ(ierr);

  base = owners[rank];

  /*  wait on receives */
  ierr = PetscMalloc(2*(nrecvs+1)*sizeof(int),&lens);CHKERRQ(ierr);
  count = nrecvs; 
  slen  = 0;
  while (count) {
    ierr = MPI_Waitany(nrecvs,recv_waits,&imdex,&recv_status);CHKERRQ(ierr);
    /* unpack receives into our local space */
    ierr = MPI_Get_count(&recv_status,MPI_INT,&n);CHKERRQ(ierr);
    lens[imdex]  =  n/2;
    slen         += n/2;
    count--;
  }
  ierr = PetscFree(recv_waits);CHKERRQ(ierr);
  
  ierr = PetscMalloc(2*(slen+1)*sizeof(int),&local_inidx);CHKERRQ(ierr);
  local_inidy = local_inidx + slen;

  count = 0;
  for (i=0; i<nrecvs; i++) {
    values = rvalues + 2*i*nmax;
    for (j=0; j<lens[i]; j++) {
      local_inidx[count]   = values[2*j] - base;
      local_inidy[count++] = values[2*j+1];
    }
  }
  ierr = PetscFree(rvalues);CHKERRQ(ierr);
  ierr = PetscFree(lens);CHKERRQ(ierr);

  /* wait on sends */
  if (nsends) {
    MPI_Status *send_status;
    ierr = PetscMalloc(nsends*sizeof(MPI_Status),&send_status);CHKERRQ(ierr);
    ierr = MPI_Waitall(nsends,send_waits,send_status);CHKERRQ(ierr);
    ierr = PetscFree(send_status);CHKERRQ(ierr);
  }
  ierr = PetscFree(send_waits);CHKERRQ(ierr);
  ierr = PetscFree(svalues);CHKERRQ(ierr);

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
        /* printf("found dup %d %d\n",local_inidx[i],local_inidy[i]);*/
      }
    }
    start = count;
  }
#endif
  if (duplicate) {
    PetscLogInfo(0,"VecScatterCreate_PtoP:Duplicate to from indices passed in VecScatterCreate(), they are ignored\n");
  }
  ierr = VecScatterCreate_StoP(slen,local_inidx,slen,local_inidy,yin,ctx);CHKERRQ(ierr);
  ierr = PetscFree(local_inidx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}





