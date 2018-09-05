
/*
     Code for creating scatters between vectors. This file
  includes the code for scattering between sequential vectors and
  some special cases for parallel scatters.
*/

#include <petsc/private/vecscatterimpl.h>    /*I   "petscvec.h"    I*/

#if defined(PETSC_HAVE_VECCUDA)
#include <../src/vec/vec/impls/seq/seqcuda/cudavecimpl.h>
#endif

#if defined(PETSC_USE_DEBUG)
/*
     Checks if any indices are less than zero and generates an error
*/
PetscErrorCode VecScatterCheckIndices_Private(PetscInt nmax,PetscInt n,const PetscInt *idx)
{
  PetscInt i;

  PetscFunctionBegin;
  for (i=0; i<n; i++) {
    if (idx[i] < 0)     SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Negative index %D at %D location",idx[i],i);
    if (idx[i] >= nmax) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Index %D at %D location greater than max %D",idx[i],i,nmax);
  }
  PetscFunctionReturn(0);
}
#endif

/*
      This is special scatter code for when the entire parallel vector is copied to each processor.

   This code was written by Cameron Cooper, Occidental College, Fall 1995,
   will working at ANL as a SERS student.
*/
PetscErrorCode VecScatterBegin_MPI_ToAll(VecScatter ctx,Vec x,Vec y,InsertMode addv,ScatterMode mode)
{
  PetscErrorCode ierr;
  PetscInt       yy_n,xx_n;
  PetscScalar    *xv,*yv;

  PetscFunctionBegin;
  ierr = VecGetLocalSize(y,&yy_n);CHKERRQ(ierr);
  ierr = VecGetLocalSize(x,&xx_n);CHKERRQ(ierr);
  ierr = VecGetArrayPair(x,y,&xv,&yv);CHKERRQ(ierr);

  if (mode & SCATTER_REVERSE) {
    PetscScalar          *xvt,*xvt2;
    VecScatter_MPI_ToAll *scat = (VecScatter_MPI_ToAll*)ctx->todata;
    PetscInt             i;
    PetscMPIInt          *disply = scat->displx;

    if (addv == INSERT_VALUES) {
      PetscInt rstart,rend;
      /*
         copy the correct part of the local vector into the local storage of
         the MPI one.  Note: this operation only makes sense if all the local
         vectors have the same values
      */
      ierr = VecGetOwnershipRange(y,&rstart,&rend);CHKERRQ(ierr);
      ierr = PetscMemcpy(yv,xv+rstart,yy_n*sizeof(PetscScalar));CHKERRQ(ierr);
    } else {
      MPI_Comm    comm;
      PetscMPIInt rank;
      ierr = PetscObjectGetComm((PetscObject)y,&comm);CHKERRQ(ierr);
      ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
      if (scat->work1) xvt = scat->work1;
      else {
        ierr        = PetscMalloc1(xx_n,&xvt);CHKERRQ(ierr);
        scat->work1 = xvt;
      }
      if (!rank) { /* I am the zeroth processor, values are accumulated here */
        if   (scat->work2) xvt2 = scat->work2;
        else {
          ierr        = PetscMalloc1(xx_n,&xvt2);CHKERRQ(ierr);
          scat->work2 = xvt2;
        }
        ierr = MPI_Gatherv(yv,yy_n,MPIU_SCALAR,xvt2,scat->count,disply,MPIU_SCALAR,0,PetscObjectComm((PetscObject)ctx));CHKERRQ(ierr);
        ierr = MPI_Reduce(xv,xvt,xx_n,MPIU_SCALAR,MPIU_SUM,0,PetscObjectComm((PetscObject)ctx));CHKERRQ(ierr);
        if (addv == ADD_VALUES) {
          for (i=0; i<xx_n; i++) xvt[i] += xvt2[i];
#if !defined(PETSC_USE_COMPLEX)
        } else if (addv == MAX_VALUES) {
          for (i=0; i<xx_n; i++) xvt[i] = PetscMax(xvt[i],xvt2[i]);
#endif
        } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_UNKNOWN_TYPE,"Wrong insert option");
        ierr = MPI_Scatterv(xvt,scat->count,disply,MPIU_SCALAR,yv,yy_n,MPIU_SCALAR,0,PetscObjectComm((PetscObject)ctx));CHKERRQ(ierr);
      } else {
        ierr = MPI_Gatherv(yv,yy_n,MPIU_SCALAR,0, 0,0,MPIU_SCALAR,0,PetscObjectComm((PetscObject)ctx));CHKERRQ(ierr);
        ierr = MPI_Reduce(xv,xvt,xx_n,MPIU_SCALAR,MPIU_SUM,0,PetscObjectComm((PetscObject)ctx));CHKERRQ(ierr);
        ierr = MPI_Scatterv(0,scat->count,disply,MPIU_SCALAR,yv,yy_n,MPIU_SCALAR,0,PetscObjectComm((PetscObject)ctx));CHKERRQ(ierr);
      }
    }
  } else {
    PetscScalar          *yvt;
    VecScatter_MPI_ToAll *scat = (VecScatter_MPI_ToAll*)ctx->todata;
    PetscInt             i;
    PetscMPIInt          *displx = scat->displx;

    if (addv == INSERT_VALUES) {
      ierr = MPI_Allgatherv(xv,xx_n,MPIU_SCALAR,yv,scat->count,displx,MPIU_SCALAR,PetscObjectComm((PetscObject)ctx));CHKERRQ(ierr);
    } else {
      if (scat->work1) yvt = scat->work1;
      else {
        ierr        = PetscMalloc1(yy_n,&yvt);CHKERRQ(ierr);
        scat->work1 = yvt;
      }
      ierr = MPI_Allgatherv(xv,xx_n,MPIU_SCALAR,yvt,scat->count,displx,MPIU_SCALAR,PetscObjectComm((PetscObject)ctx));CHKERRQ(ierr);
      if (addv == ADD_VALUES) {
        for (i=0; i<yy_n; i++) yv[i] += yvt[i];
#if !defined(PETSC_USE_COMPLEX)
      } else if (addv == MAX_VALUES) {
        for (i=0; i<yy_n; i++) yv[i] = PetscMax(yv[i],yvt[i]);
#endif
      } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_UNKNOWN_TYPE,"Wrong insert option");
    }
  }
  ierr = VecRestoreArrayPair(x,y,&xv,&yv);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode VecScatterView_MPI_ToAll(VecScatter in,PetscViewer viewer)
{
  PetscErrorCode ierr;
  PetscBool      isascii;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&isascii);CHKERRQ(ierr);
  if (isascii) {
    ierr = PetscViewerASCIIPrintf(viewer,"Entire parallel vector is copied to each process\n");CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*
      This is special scatter code for when the entire parallel vector is  copied to processor 0.

*/
PetscErrorCode VecScatterBegin_MPI_ToOne(VecScatter ctx,Vec x,Vec y,InsertMode addv,ScatterMode mode)
{
  PetscErrorCode    ierr;
  PetscMPIInt       rank;
  PetscInt          yy_n,xx_n;
  PetscScalar       *yv;
  const PetscScalar *xv;
  MPI_Comm          comm;

  PetscFunctionBegin;
  ierr = VecGetLocalSize(y,&yy_n);CHKERRQ(ierr);
  ierr = VecGetLocalSize(x,&xx_n);CHKERRQ(ierr);
  ierr = VecGetArrayRead(x,&xv);CHKERRQ(ierr);
  ierr = VecGetArray(y,&yv);CHKERRQ(ierr);

  ierr = PetscObjectGetComm((PetscObject)x,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);

  /* --------  Reverse scatter; spread from processor 0 to other processors */
  if (mode & SCATTER_REVERSE) {
    PetscScalar          *yvt;
    VecScatter_MPI_ToAll *scat = (VecScatter_MPI_ToAll*)ctx->todata;
    PetscInt             i;
    PetscMPIInt          *disply = scat->displx;

    if (addv == INSERT_VALUES) {
      ierr = MPI_Scatterv((PetscScalar*)xv,scat->count,disply,MPIU_SCALAR,yv,yy_n,MPIU_SCALAR,0,PetscObjectComm((PetscObject)ctx));CHKERRQ(ierr);
    } else {
      if (scat->work2) yvt = scat->work2;
      else {
        PetscInt xx_nt;
        ierr = MPI_Allreduce(&xx_n,&xx_nt,1,MPIU_INT,MPI_MAX,PetscObjectComm((PetscObject)y));CHKERRQ(ierr);
        ierr        = PetscMalloc1(xx_nt,&yvt);CHKERRQ(ierr);
        scat->work2 = yvt;
      }
      ierr = MPI_Scatterv((PetscScalar*)xv,scat->count,disply,MPIU_SCALAR,yvt,yy_n,MPIU_SCALAR,0,PetscObjectComm((PetscObject)ctx));CHKERRQ(ierr);
      if (addv == ADD_VALUES) {
        for (i=0; i<yy_n; i++) yv[i] += yvt[i];
#if !defined(PETSC_USE_COMPLEX)
      } else if (addv == MAX_VALUES) {
        for (i=0; i<yy_n; i++) yv[i] = PetscMax(yv[i],yvt[i]);
#endif
      } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_UNKNOWN_TYPE,"Wrong insert option");
    }
  /* ---------  Forward scatter; gather all values onto processor 0 */
  } else {
    PetscScalar          *yvt  = 0;
    VecScatter_MPI_ToAll *scat = (VecScatter_MPI_ToAll*)ctx->todata;
    PetscInt             i;
    PetscMPIInt          *displx = scat->displx;

    if (addv == INSERT_VALUES) {
      ierr = MPI_Gatherv((PetscScalar*)xv,xx_n,MPIU_SCALAR,yv,scat->count,displx,MPIU_SCALAR,0,PetscObjectComm((PetscObject)ctx));CHKERRQ(ierr);
    } else {
      if (!rank) {
        if (scat->work1) yvt = scat->work1;
        else {
          ierr        = PetscMalloc1(yy_n,&yvt);CHKERRQ(ierr);
          scat->work1 = yvt;
        }
      }
      ierr = MPI_Gatherv((PetscScalar*)xv,xx_n,MPIU_SCALAR,yvt,scat->count,displx,MPIU_SCALAR,0,PetscObjectComm((PetscObject)ctx));CHKERRQ(ierr);
      if (!rank) {
        if (addv == ADD_VALUES) {
          for (i=0; i<yy_n; i++) yv[i] += yvt[i];
#if !defined(PETSC_USE_COMPLEX)
        } else if (addv == MAX_VALUES) {
          for (i=0; i<yy_n; i++) yv[i] = PetscMax(yv[i],yvt[i]);
#endif
        }  else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_UNKNOWN_TYPE,"Wrong insert option");
      }
    }
  }
  ierr = VecRestoreArrayRead(x,&xv);CHKERRQ(ierr);
  ierr = VecRestoreArray(y,&yv);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
       The follow to are used for both VecScatterBegin_MPI_ToAll() and VecScatterBegin_MPI_ToOne()
*/
PetscErrorCode VecScatterDestroy_MPI_ToAll(VecScatter ctx)
{
  VecScatter_MPI_ToAll *scat = (VecScatter_MPI_ToAll*)ctx->todata;
  PetscErrorCode       ierr;

  PetscFunctionBegin;
  ierr = PetscFree(scat->work1);CHKERRQ(ierr);
  ierr = PetscFree(scat->work2);CHKERRQ(ierr);
  ierr = PetscFree3(ctx->todata,scat->count,scat->displx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode VecScatterDestroy_SGToSG(VecScatter ctx)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFree2(((VecScatter_Seq_General*)ctx->todata)->vslots,((VecScatter_Seq_General*)ctx->fromdata)->vslots);CHKERRQ(ierr);
  ierr = VecScatterMemcpyPlanDestroy(&((VecScatter_Seq_General*)ctx->fromdata)->memcpy_plan);CHKERRQ(ierr);;
  ierr = VecScatterMemcpyPlanDestroy(&((VecScatter_Seq_General*)ctx->todata)->memcpy_plan);CHKERRQ(ierr);;
  ierr = PetscFree2(ctx->todata,ctx->fromdata);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode VecScatterDestroy_SGToSS(VecScatter ctx)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFree(((VecScatter_Seq_General*)ctx->fromdata)->vslots);CHKERRQ(ierr);
  ierr = VecScatterMemcpyPlanDestroy(&((VecScatter_Seq_General*)ctx->fromdata)->memcpy_plan);CHKERRQ(ierr);;
  ierr = PetscFree2(ctx->todata,ctx->fromdata);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode VecScatterDestroy_SSToSG(VecScatter ctx)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFree(((VecScatter_Seq_General*)ctx->todata)->vslots);CHKERRQ(ierr);
  ierr = VecScatterMemcpyPlanDestroy(&((VecScatter_Seq_General*)ctx->todata)->memcpy_plan);CHKERRQ(ierr);;
  ierr = PetscFree2(ctx->todata,ctx->fromdata);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode VecScatterDestroy_SSToSS(VecScatter ctx)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFree2(ctx->todata,ctx->fromdata);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------*/
PetscErrorCode VecScatterCopy_MPI_ToAll(VecScatter in,VecScatter out)
{
  VecScatter_MPI_ToAll *in_to = (VecScatter_MPI_ToAll*)in->todata,*sto;
  PetscErrorCode       ierr;
  PetscMPIInt          size,*count,*displx;
  PetscInt             i;

  PetscFunctionBegin;
  out->ops->begin   = in->ops->begin;
  out->ops->end     = in->ops->end;
  out->ops->copy    = in->ops->copy;
  out->ops->destroy = in->ops->destroy;
  out->ops->view    = in->ops->view;

  ierr        = MPI_Comm_size(PetscObjectComm((PetscObject)out),&size);CHKERRQ(ierr);
  ierr        = PetscMalloc3(1,&sto,size,&count,size,&displx);CHKERRQ(ierr);
  sto->format = in_to->format;
  sto->count  = count;
  sto->displx = displx;
  for (i=0; i<size; i++) {
    sto->count[i]  = in_to->count[i];
    sto->displx[i] = in_to->displx[i];
  }
  sto->work1    = 0;
  sto->work2    = 0;
  out->todata   = (void*)sto;
  out->fromdata = (void*)0;
  PetscFunctionReturn(0);
}

/* --------------------------------------------------------------------------------------*/
/*
        Scatter: sequential general to sequential general
*/
PetscErrorCode VecScatterBegin_SGToSG(VecScatter ctx,Vec x,Vec y,InsertMode addv,ScatterMode mode)
{
  VecScatter_Seq_General *gen_to   = (VecScatter_Seq_General*)ctx->todata;
  VecScatter_Seq_General *gen_from = (VecScatter_Seq_General*)ctx->fromdata;
  PetscErrorCode         ierr;
  PetscInt               i,n = gen_from->n,*fslots,*tslots;
  PetscScalar            *xv,*yv;
#if defined(PETSC_HAVE_VECCUDA)
  PetscBool              is_veccuda;
#endif

  PetscFunctionBegin;
#if defined(PETSC_HAVE_VECCUDA)
  ierr = PetscObjectTypeCompareAny((PetscObject)x,&is_veccuda,VECSEQCUDA,VECMPICUDA,VECCUDA,"");CHKERRQ(ierr);
  if (is_veccuda && x->valid_GPU_array == PETSC_OFFLOAD_GPU) {
    /* create the scatter indices if not done already */
    if (!ctx->spptr) {
      PetscInt tofirst = 0,tostep = 0,fromfirst = 0,fromstep = 0;
      fslots = gen_from->vslots;
      tslots = gen_to->vslots;
      ierr = VecScatterCUDAIndicesCreate_StoS(n,tofirst,fromfirst,tostep,fromstep,tslots,fslots,(PetscCUDAIndices*)&(ctx->spptr));CHKERRQ(ierr);
    }
    /* next do the scatter */
    ierr = VecScatterCUDA_StoS(x,y,(PetscCUDAIndices)ctx->spptr,addv,mode);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
#endif

  ierr = VecGetArrayPair(x,y,&xv,&yv);CHKERRQ(ierr);
  if (mode & SCATTER_REVERSE) {
    gen_to   = (VecScatter_Seq_General*)ctx->fromdata;
    gen_from = (VecScatter_Seq_General*)ctx->todata;
  }
  fslots = gen_from->vslots;
  tslots = gen_to->vslots;

  if (gen_from->memcpy_plan.optimized[0]) { ierr = VecScatterMemcpyPlanExecute_Scatter(0,xv,&gen_from->memcpy_plan,yv,&gen_to->memcpy_plan,addv);CHKERRQ(ierr); }
  else if (addv == INSERT_VALUES) { for (i=0; i<n; i++) yv[tslots[i]]  = xv[fslots[i]]; }
  else if (addv == ADD_VALUES)    { for (i=0; i<n; i++) yv[tslots[i]] += xv[fslots[i]]; }
#if !defined(PETSC_USE_COMPLEX)
  else if (addv == MAX_VALUES)    { for (i=0; i<n; i++) yv[tslots[i]]  = PetscMax(yv[tslots[i]],xv[fslots[i]]); }
#endif
  else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_UNKNOWN_TYPE,"Wrong insert option");
  ierr = VecRestoreArrayPair(x,y,&xv,&yv);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
    Scatter: sequential general to sequential stride 1
*/
PetscErrorCode VecScatterBegin_SGToSS_Stride1(VecScatter ctx,Vec x,Vec y,InsertMode addv,ScatterMode mode)
{
  VecScatter_Seq_Stride  *gen_to   = (VecScatter_Seq_Stride*)ctx->todata;
  VecScatter_Seq_General *gen_from = (VecScatter_Seq_General*)ctx->fromdata;
  PetscInt               i,n = gen_from->n,*fslots = gen_from->vslots;
  PetscErrorCode         ierr;
  PetscInt               first = gen_to->first;
  PetscScalar            *xv,*yv;
#if defined(PETSC_HAVE_VECCUDA)
  PetscBool              is_veccuda;
#endif

  PetscFunctionBegin;
#if defined(PETSC_HAVE_VECCUDA)
  ierr = PetscObjectTypeCompareAny((PetscObject)x,&is_veccuda,VECSEQCUDA,VECMPICUDA,VECCUDA,"");CHKERRQ(ierr);
  if (is_veccuda && x->valid_GPU_array == PETSC_OFFLOAD_GPU) {
    /* create the scatter indices if not done already */
    if (!ctx->spptr) {
      PetscInt tofirst = first,tostep = 1,fromfirst = 0,fromstep = 0;
      PetscInt *tslots = 0;
      ierr = VecScatterCUDAIndicesCreate_StoS(n,tofirst,fromfirst,tostep,fromstep,tslots,fslots,(PetscCUDAIndices*)&(ctx->spptr));CHKERRQ(ierr);
    }
    /* next do the scatter */
    ierr = VecScatterCUDA_StoS(x,y,(PetscCUDAIndices)ctx->spptr,addv,mode);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
#endif

  ierr = VecGetArrayPair(x,y,&xv,&yv);CHKERRQ(ierr);
  if (mode & SCATTER_REVERSE) {
    PetscScalar *xxv = xv + first;
    if (gen_from->memcpy_plan.optimized[0]) { ierr = VecScatterMemcpyPlanExecute_Unpack(0,xxv,yv,&gen_from->memcpy_plan,addv,1);CHKERRQ(ierr); }
    else if (addv == INSERT_VALUES) { for (i=0; i<n; i++) yv[fslots[i]]  = xxv[i]; }
    else if (addv == ADD_VALUES)    { for (i=0; i<n; i++) yv[fslots[i]] += xxv[i]; }
#if !defined(PETSC_USE_COMPLEX)
    else if (addv == MAX_VALUES)    { for (i=0; i<n; i++) yv[fslots[i]]  = PetscMax(yv[fslots[i]],xxv[i]); }
#endif
    else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_UNKNOWN_TYPE,"Wrong insert option");
  } else {
    PetscScalar *yyv = yv + first;
    if (gen_from->memcpy_plan.optimized[0]) { ierr = VecScatterMemcpyPlanExecute_Pack(0,xv,&gen_from->memcpy_plan,yyv,addv,1);CHKERRQ(ierr); }
    else if (addv == INSERT_VALUES) { for (i=0; i<n; i++) yyv[i]  = xv[fslots[i]]; }
    else if (addv == ADD_VALUES)    { for (i=0; i<n; i++) yyv[i] += xv[fslots[i]]; }
#if !defined(PETSC_USE_COMPLEX)
    else if (addv == MAX_VALUES)    { for (i=0; i<n; i++) yyv[i]  = PetscMax(yyv[i],xv[fslots[i]]); }
#endif
    else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_UNKNOWN_TYPE,"Wrong insert option");
  }
  ierr = VecRestoreArrayPair(x,y,&xv,&yv);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
   Scatter: sequential general to sequential stride
*/
PetscErrorCode VecScatterBegin_SGToSS(VecScatter ctx,Vec x,Vec y,InsertMode addv,ScatterMode mode)
{
  VecScatter_Seq_Stride  *gen_to   = (VecScatter_Seq_Stride*)ctx->todata;
  VecScatter_Seq_General *gen_from = (VecScatter_Seq_General*)ctx->fromdata;
  PetscInt               i,n = gen_from->n,*fslots = gen_from->vslots;
  PetscErrorCode         ierr;
  PetscInt               first = gen_to->first,step = gen_to->step;
  PetscScalar            *xv,*yv;
#if defined(PETSC_HAVE_VECCUDA)
  PetscBool              is_veccuda;
#endif

  PetscFunctionBegin;
#if defined(PETSC_HAVE_VECCUDA)
  ierr = PetscObjectTypeCompareAny((PetscObject)x,&is_veccuda,VECSEQCUDA,VECMPICUDA,VECCUDA,"");CHKERRQ(ierr);
  if (is_veccuda && x->valid_GPU_array == PETSC_OFFLOAD_GPU) {
    /* create the scatter indices if not done already */
    if (!ctx->spptr) {
      PetscInt tofirst = first,tostep = step,fromfirst = 0,fromstep = 0;
      PetscInt * tslots = 0;
      ierr = VecScatterCUDAIndicesCreate_StoS(n,tofirst,fromfirst,tostep,fromstep,tslots,fslots,(PetscCUDAIndices*)&(ctx->spptr));CHKERRQ(ierr);
    }
    /* next do the scatter */
    ierr = VecScatterCUDA_StoS(x,y,(PetscCUDAIndices)ctx->spptr,addv,mode);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
#endif

  ierr = VecGetArrayPair(x,y,&xv,&yv);CHKERRQ(ierr);
  if (mode & SCATTER_REVERSE) {
    if (addv == INSERT_VALUES) {
      for (i=0; i<n; i++) yv[fslots[i]] = xv[first + i*step];
    } else if (addv == ADD_VALUES) {
      for (i=0; i<n; i++) yv[fslots[i]] += xv[first + i*step];
#if !defined(PETSC_USE_COMPLEX)
    } else if (addv == MAX_VALUES) {
      for (i=0; i<n; i++) yv[fslots[i]] = PetscMax(yv[fslots[i]],xv[first + i*step]);
#endif
    } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_UNKNOWN_TYPE,"Wrong insert option");
  } else {
    if (addv == INSERT_VALUES) {
      for (i=0; i<n; i++) yv[first + i*step] = xv[fslots[i]];
    } else if (addv == ADD_VALUES) {
      for (i=0; i<n; i++) yv[first + i*step] += xv[fslots[i]];
#if !defined(PETSC_USE_COMPLEX)
    } else if (addv == MAX_VALUES) {
      for (i=0; i<n; i++) yv[first + i*step] = PetscMax(yv[first + i*step],xv[fslots[i]]);
#endif
    } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_UNKNOWN_TYPE,"Wrong insert option");
  }
  ierr = VecRestoreArrayPair(x,y,&xv,&yv);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
    Scatter: sequential stride 1 to sequential general
*/
PetscErrorCode VecScatterBegin_SSToSG_Stride1(VecScatter ctx,Vec x,Vec y,InsertMode addv,ScatterMode mode)
{
  VecScatter_Seq_Stride  *gen_from = (VecScatter_Seq_Stride*)ctx->fromdata;
  VecScatter_Seq_General *gen_to   = (VecScatter_Seq_General*)ctx->todata;
  PetscInt               i,n = gen_from->n,*tslots = gen_to->vslots;
  PetscErrorCode         ierr;
  PetscInt               first = gen_from->first;
  PetscScalar            *xv,*yv;
#if defined(PETSC_HAVE_VECCUDA)
  PetscBool              is_veccuda;
#endif

  PetscFunctionBegin;
#if defined(PETSC_HAVE_VECCUDA)
  ierr = PetscObjectTypeCompareAny((PetscObject)x,&is_veccuda,VECSEQCUDA,VECMPICUDA,VECCUDA,"");CHKERRQ(ierr);
  if (is_veccuda && x->valid_GPU_array == PETSC_OFFLOAD_GPU) {
    /* create the scatter indices if not done already */
    if (!ctx->spptr) {
      PetscInt tofirst = 0,tostep = 0,fromfirst = first,fromstep = 1;
      PetscInt *fslots = 0;
      ierr = VecScatterCUDAIndicesCreate_StoS(n,tofirst,fromfirst,tostep,fromstep,tslots,fslots,(PetscCUDAIndices*)&(ctx->spptr));CHKERRQ(ierr);
    }
    /* next do the scatter */
    ierr = VecScatterCUDA_StoS(x,y,(PetscCUDAIndices)ctx->spptr,addv,mode);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
#endif

  ierr = VecGetArrayPair(x,y,&xv,&yv);CHKERRQ(ierr);
  if (mode & SCATTER_REVERSE) {
    PetscScalar *yyv = yv + first;
    if (gen_to->memcpy_plan.optimized[0]) { ierr = VecScatterMemcpyPlanExecute_Pack(0,xv,&gen_to->memcpy_plan,yyv,addv,1);CHKERRQ(ierr); }
    else if (addv == INSERT_VALUES) { for (i=0; i<n; i++) yyv[i]  = xv[tslots[i]]; }
    else if (addv == ADD_VALUES)    { for (i=0; i<n; i++) yyv[i] += xv[tslots[i]]; }
#if !defined(PETSC_USE_COMPLEX)
    else if (addv == MAX_VALUES)    { for (i=0; i<n; i++) yyv[i]  = PetscMax(yyv[i],xv[tslots[i]]); }
#endif
    else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_UNKNOWN_TYPE,"Wrong insert option");
  } else {
    PetscScalar *xxv = xv + first;
    if (gen_to->memcpy_plan.optimized[0]) { ierr = VecScatterMemcpyPlanExecute_Unpack(0,xxv,yv,&gen_to->memcpy_plan,addv,1);CHKERRQ(ierr); }
    else if (addv == INSERT_VALUES) { for (i=0; i<n; i++) yv[tslots[i]]  = xxv[i]; }
    else if (addv == ADD_VALUES)    { for (i=0; i<n; i++) yv[tslots[i]] += xxv[i]; }
#if !defined(PETSC_USE_COMPLEX)
    else if (addv == MAX_VALUES)    { for (i=0; i<n; i++) yv[tslots[i]]  = PetscMax(yv[tslots[i]],xxv[i]); }
#endif
    else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_UNKNOWN_TYPE,"Wrong insert option");
  }
  ierr = VecRestoreArrayPair(x,y,&xv,&yv);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
   Scatter: sequential stride to sequential general
*/
PetscErrorCode VecScatterBegin_SSToSG(VecScatter ctx,Vec x,Vec y,InsertMode addv,ScatterMode mode)
{
  VecScatter_Seq_Stride  *gen_from = (VecScatter_Seq_Stride*)ctx->fromdata;
  VecScatter_Seq_General *gen_to   = (VecScatter_Seq_General*)ctx->todata;
  PetscInt               i,n = gen_from->n,*tslots = gen_to->vslots;
  PetscErrorCode         ierr;
  PetscInt               first = gen_from->first,step = gen_from->step;
  PetscScalar            *xv,*yv;
#if defined(PETSC_HAVE_VECCUDA)
  PetscBool              is_veccuda;
#endif

  PetscFunctionBegin;
#if defined(PETSC_HAVE_VECCUDA)
  ierr = PetscObjectTypeCompareAny((PetscObject)x,&is_veccuda,VECSEQCUDA,VECMPICUDA,VECCUDA,"");CHKERRQ(ierr);
  if (is_veccuda && x->valid_GPU_array == PETSC_OFFLOAD_GPU) {
    /* create the scatter indices if not done already */
    if (!ctx->spptr) {
      PetscInt tofirst = 0,tostep = 0,fromfirst = first,fromstep = step;
      PetscInt *fslots = 0;
      ierr = VecScatterCUDAIndicesCreate_StoS(n,tofirst,fromfirst,tostep,fromstep,tslots,fslots,(PetscCUDAIndices*)&(ctx->spptr));CHKERRQ(ierr);
    }
    /* next do the scatter */
    ierr = VecScatterCUDA_StoS(x,y,(PetscCUDAIndices)ctx->spptr,addv,mode);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
#endif

  ierr = VecGetArrayPair(x,y,&xv,&yv);CHKERRQ(ierr);
  if (mode & SCATTER_REVERSE) {
    if (addv == INSERT_VALUES) {
      for (i=0; i<n; i++) yv[first + i*step] = xv[tslots[i]];
    } else if (addv == ADD_VALUES) {
      for (i=0; i<n; i++) yv[first + i*step] += xv[tslots[i]];
#if !defined(PETSC_USE_COMPLEX)
    } else if (addv == MAX_VALUES) {
      for (i=0; i<n; i++) yv[first + i*step] = PetscMax(yv[first + i*step],xv[tslots[i]]);
#endif
    } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_UNKNOWN_TYPE,"Wrong insert option");
  } else {
    if (addv == INSERT_VALUES) {
      for (i=0; i<n; i++) yv[tslots[i]] = xv[first + i*step];
    } else if (addv == ADD_VALUES) {
      for (i=0; i<n; i++) yv[tslots[i]] += xv[first + i*step];
#if !defined(PETSC_USE_COMPLEX)
    } else if (addv == MAX_VALUES) {
      for (i=0; i<n; i++) yv[tslots[i]] = PetscMax(yv[tslots[i]],xv[first + i*step]);
#endif
    } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_UNKNOWN_TYPE,"Wrong insert option");
  }
  ierr = VecRestoreArrayPair(x,y,&xv,&yv);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode VecScatterView_SSToSG(VecScatter in,PetscViewer viewer)
{
  PetscErrorCode         ierr;
  VecScatter_Seq_Stride  *in_from = (VecScatter_Seq_Stride*)in->fromdata;
  VecScatter_Seq_General *in_to   = (VecScatter_Seq_General*)in->todata;
  PetscInt               i;
  PetscBool              isascii;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&isascii);CHKERRQ(ierr);
  if (isascii) {
    ierr = PetscViewerASCIIPrintf(viewer,"Sequential stride to general scatter\n");CHKERRQ(ierr);
    for (i=0; i<in_to->n; i++) {
      ierr = PetscViewerASCIIPrintf(viewer,"%D to %D\n",in_from->first + in_from->step*i,in_to->vslots[i]);CHKERRQ(ierr);
    }
    if (in_to->memcpy_plan.optimized[0]) {
      ierr = PetscViewerASCIIPrintf(viewer,"This stride1 to general scatter is made of %D copies\n",in_to->memcpy_plan.copy_offsets[1]);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}
/*
     Scatter: sequential stride to sequential stride
*/
PetscErrorCode VecScatterBegin_SSToSS(VecScatter ctx,Vec x,Vec y,InsertMode addv,ScatterMode mode)
{
  VecScatter_Seq_Stride *gen_to   = (VecScatter_Seq_Stride*)ctx->todata;
  VecScatter_Seq_Stride *gen_from = (VecScatter_Seq_Stride*)ctx->fromdata;
  PetscInt              i,n = gen_from->n,to_first = gen_to->first,to_step = gen_to->step;
  PetscErrorCode        ierr;
  PetscInt              from_first = gen_from->first,from_step = gen_from->step;
  PetscScalar           *xv,*yv;
#if defined(PETSC_HAVE_VECCUDA)
  PetscBool              is_veccuda;
#endif

  PetscFunctionBegin;
#if defined(PETSC_HAVE_VECCUDA)
  ierr = PetscObjectTypeCompareAny((PetscObject)x,&is_veccuda,VECSEQCUDA,VECMPICUDA,VECCUDA,"");CHKERRQ(ierr);
  if (is_veccuda && x->valid_GPU_array == PETSC_OFFLOAD_GPU) {
    /* create the scatter indices if not done already */
    if (!ctx->spptr) {
      PetscInt *tslots = 0,*fslots = 0;
      ierr = VecScatterCUDAIndicesCreate_StoS(n,to_first,from_first,to_step,from_step,tslots,fslots,(PetscCUDAIndices*)&(ctx->spptr));CHKERRQ(ierr);
    }
    /* next do the scatter */
    ierr = VecScatterCUDA_StoS(x,y,(PetscCUDAIndices)ctx->spptr,addv,mode);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
#endif

  ierr = VecGetArrayPair(x,y,&xv,&yv);CHKERRQ(ierr);
  if (mode & SCATTER_REVERSE) {
    from_first = gen_to->first;
    to_first   = gen_from->first;
    from_step  = gen_to->step;
    to_step    = gen_from->step;
  }

  if (addv == INSERT_VALUES) {
    if (to_step == 1 && from_step == 1) {
      ierr = PetscMemcpy(yv+to_first,xv+from_first,n*sizeof(PetscScalar));CHKERRQ(ierr);
    } else  {
      for (i=0; i<n; i++) yv[to_first + i*to_step] = xv[from_first+i*from_step];
    }
  } else if (addv == ADD_VALUES) {
    if (to_step == 1 && from_step == 1) {
      PetscScalar *yyv = yv + to_first, *xxv = xv + from_first;
      for (i=0; i<n; i++) yyv[i] += xxv[i];
    } else {
      for (i=0; i<n; i++) yv[to_first + i*to_step] += xv[from_first+i*from_step];
    }
#if !defined(PETSC_USE_COMPLEX)
  } else if (addv == MAX_VALUES) {
    if (to_step == 1 && from_step == 1) {
      PetscScalar *yyv = yv + to_first, *xxv = xv + from_first;
      for (i=0; i<n; i++) yyv[i] = PetscMax(yyv[i],xxv[i]);
    } else {
      for (i=0; i<n; i++) yv[to_first + i*to_step] = PetscMax(yv[to_first + i*to_step],xv[from_first+i*from_step]);
    }
#endif
  } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_UNKNOWN_TYPE,"Wrong insert option");
  ierr = VecRestoreArrayPair(x,y,&xv,&yv);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* --------------------------------------------------------------------------*/


PetscErrorCode VecScatterCopy_SGToSG(VecScatter in,VecScatter out)
{
  PetscErrorCode         ierr;
  VecScatter_Seq_General *in_to   = (VecScatter_Seq_General*)in->todata,*out_to = NULL;
  VecScatter_Seq_General *in_from = (VecScatter_Seq_General*)in->fromdata,*out_from = NULL;

  PetscFunctionBegin;
  out->ops->begin   = in->ops->begin;
  out->ops->end     = in->ops->end;
  out->ops->copy    = in->ops->copy;
  out->ops->destroy = in->ops->destroy;
  out->ops->view    = in->ops->view;

  ierr                         = PetscMalloc2(1,&out_to,1,&out_from);CHKERRQ(ierr);
  ierr                         = PetscMalloc2(in_to->n,&out_to->vslots,in_from->n,&out_from->vslots);CHKERRQ(ierr);
  out_to->n                    = in_to->n;
  out_to->format               = in_to->format;
  out_to->nonmatching_computed = PETSC_FALSE;
  out_to->slots_nonmatching    = 0;
  ierr = PetscMemcpy(out_to->vslots,in_to->vslots,(out_to->n)*sizeof(PetscInt));CHKERRQ(ierr);
  ierr = VecScatterMemcpyPlanCopy(&in_to->memcpy_plan,&out_to->memcpy_plan);CHKERRQ(ierr);

  out_from->n                    = in_from->n;
  out_from->format               = in_from->format;
  out_from->nonmatching_computed = PETSC_FALSE;
  out_from->slots_nonmatching    = 0;
  ierr = PetscMemcpy(out_from->vslots,in_from->vslots,(out_from->n)*sizeof(PetscInt));CHKERRQ(ierr);
  ierr = VecScatterMemcpyPlanCopy(&in_from->memcpy_plan,&out_from->memcpy_plan);CHKERRQ(ierr);

  out->todata   = (void*)out_to;
  out->fromdata = (void*)out_from;
  PetscFunctionReturn(0);
}

PetscErrorCode VecScatterView_SGToSG(VecScatter in,PetscViewer viewer)
{
  PetscErrorCode         ierr;
  VecScatter_Seq_General *in_to   = (VecScatter_Seq_General*)in->todata;
  VecScatter_Seq_General *in_from = (VecScatter_Seq_General*)in->fromdata;
  PetscInt               i;
  PetscBool              isascii;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&isascii);CHKERRQ(ierr);
  if (isascii) {
    ierr = PetscViewerASCIIPrintf(viewer,"Sequential general scatter\n");CHKERRQ(ierr);
    for (i=0; i<in_to->n; i++) {
      ierr = PetscViewerASCIIPrintf(viewer,"%D to %D\n",in_from->vslots[i],in_to->vslots[i]);CHKERRQ(ierr);
    }
    if (in_from->memcpy_plan.optimized[0]) {
      ierr = PetscViewerASCIIPrintf(viewer,"This general to general scatter is made of %D copies\n",in_from->memcpy_plan.copy_offsets[1]);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}


PetscErrorCode VecScatterCopy_SGToSS(VecScatter in,VecScatter out)
{
  PetscErrorCode         ierr;
  VecScatter_Seq_Stride  *in_to   = (VecScatter_Seq_Stride*)in->todata,*out_to = NULL;
  VecScatter_Seq_General *in_from = (VecScatter_Seq_General*)in->fromdata,*out_from = NULL;

  PetscFunctionBegin;
  out->ops->begin   = in->ops->begin;
  out->ops->end     = in->ops->end;
  out->ops->copy    = in->ops->copy;
  out->ops->destroy = in->ops->destroy;
  out->ops->view    = in->ops->view;

  ierr           = PetscMalloc2(1,&out_to,1,&out_from);CHKERRQ(ierr);
  ierr           = PetscMalloc1(in_from->n,&out_from->vslots);CHKERRQ(ierr);
  out_to->n      = in_to->n;
  out_to->format = in_to->format;
  out_to->first  = in_to->first;
  out_to->step   = in_to->step;
  out_to->format = in_to->format;

  out_from->n                    = in_from->n;
  out_from->format               = in_from->format;
  out_from->nonmatching_computed = PETSC_FALSE;
  out_from->slots_nonmatching    = 0;
  ierr = PetscMemcpy(out_from->vslots,in_from->vslots,(out_from->n)*sizeof(PetscInt));CHKERRQ(ierr);
  ierr = VecScatterMemcpyPlanCopy(&in_from->memcpy_plan,&out_from->memcpy_plan);CHKERRQ(ierr);

  out->todata   = (void*)out_to;
  out->fromdata = (void*)out_from;
  PetscFunctionReturn(0);
}

PetscErrorCode VecScatterView_SGToSS(VecScatter in,PetscViewer viewer)
{
  PetscErrorCode         ierr;
  VecScatter_Seq_Stride  *in_to   = (VecScatter_Seq_Stride*)in->todata;
  VecScatter_Seq_General *in_from = (VecScatter_Seq_General*)in->fromdata;
  PetscInt               i;
  PetscBool              isascii;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&isascii);CHKERRQ(ierr);
  if (isascii) {
    ierr = PetscViewerASCIIPrintf(viewer,"Sequential general scatter to stride\n");CHKERRQ(ierr);
    for (i=0; i<in_to->n; i++) {
      ierr = PetscViewerASCIIPrintf(viewer,"%D to %D\n",in_from->vslots[i],in_to->first + in_to->step*i);CHKERRQ(ierr);
    }
    if (in_from->memcpy_plan.optimized[0]) {
      ierr = PetscViewerASCIIPrintf(viewer,"This general to stride1 scatter is made of %D copies\n",in_from->memcpy_plan.copy_offsets[1]);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

/* --------------------------------------------------------------------------*/
/*
    Scatter: parallel to sequential vector, sequential strides for both.
*/
PetscErrorCode VecScatterCopy_SSToSS(VecScatter in,VecScatter out)
{
  VecScatter_Seq_Stride *in_to   = (VecScatter_Seq_Stride*)in->todata,*out_to = NULL;
  VecScatter_Seq_Stride *in_from = (VecScatter_Seq_Stride*)in->fromdata,*out_from = NULL;
  PetscErrorCode        ierr;

  PetscFunctionBegin;
  out->ops->begin   = in->ops->begin;
  out->ops->end     = in->ops->end;
  out->ops->copy    = in->ops->copy;
  out->ops->destroy = in->ops->destroy;
  out->ops->view    = in->ops->view;

  ierr            = PetscMalloc2(1,&out_to,1,&out_from);CHKERRQ(ierr);
  out_to->n       = in_to->n;
  out_to->format  = in_to->format;
  out_to->first   = in_to->first;
  out_to->step    = in_to->step;
  out_to->format  = in_to->format;
  out_from->n     = in_from->n;
  out_from->format= in_from->format;
  out_from->first = in_from->first;
  out_from->step  = in_from->step;
  out_from->format= in_from->format;
  out->todata     = (void*)out_to;
  out->fromdata   = (void*)out_from;
  PetscFunctionReturn(0);
}

PetscErrorCode VecScatterView_SSToSS(VecScatter in,PetscViewer viewer)
{
  VecScatter_Seq_Stride *in_to   = (VecScatter_Seq_Stride*)in->todata;
  VecScatter_Seq_Stride *in_from = (VecScatter_Seq_Stride*)in->fromdata;
  PetscErrorCode        ierr;
  PetscBool             isascii;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&isascii);CHKERRQ(ierr);
  if (isascii) {
    ierr = PetscViewerASCIIPrintf(viewer,"Sequential stride count %D start %D step to start %D stride %D\n",in_to->n,in_to->first,in_to->step,in_from->first,in_from->step);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*
  Create memcpy optimization plans based on indices of vector entries we want to scatter

   Input Parameters:
  +  n       - number of target processors
  .  starts  - [n+1] for the i-th processor, its associated indices are indices[starts[i], starts[i+1])
  .  indices - [] array storing indices. Its length is starts[n+1]
  -  bs      - block size

   Output Parameters:
  +  plan    - the memcpy plan
*/
PetscErrorCode VecScatterMemcpyPlanCreate_Index(PetscInt n,const PetscInt *starts,const PetscInt *indices,PetscInt bs,VecScatterMemcpyPlan *plan)
{
  PetscErrorCode ierr;
  PetscInt       i,j,k,my_copies,n_copies=0,step;
  PetscBool      strided,has_strided;

  PetscFunctionBegin;
  ierr    = PetscMemzero(plan,sizeof(VecScatterMemcpyPlan));CHKERRQ(ierr);
  plan->n = n;
  ierr    = PetscMalloc2(n,&plan->optimized,n+1,&plan->copy_offsets);CHKERRQ(ierr);

  /* check if each remote part of the scatter is made of copies, and count total_copies */
  for (i=0; i<n; i++) { /* for each target processor procs[i] */
    my_copies = 1; /* count num. of copies for procs[i] */
    for (j=starts[i]; j<starts[i+1]-1; j++) { /* go through indices from the first to the second to last */
      if (indices[j]+bs != indices[j+1]) my_copies++;
    }
    if (bs*(starts[i+1]-starts[i])*sizeof(PetscScalar)/my_copies >= 256) { /* worth using memcpy? */
      plan->optimized[i] = PETSC_TRUE;
      n_copies += my_copies;
    } else {
      plan->optimized[i] = PETSC_FALSE;
    }
  }

  /* do malloc with the recently known n_copies */
  ierr = PetscMalloc2(n_copies,&plan->copy_starts,n_copies,&plan->copy_lengths);CHKERRQ(ierr);

  /* analyze the total_copies one by one */
  k                     = 0; /* k-th copy */
  plan->copy_offsets[0] = 0;
  for (i=0; i<n; i++) { /* for each target processor procs[i] */
    if (plan->optimized[i]) {
      my_copies            = 1;
      plan->copy_starts[k] = indices[starts[i]];
      for (j=starts[i]; j<starts[i+1]-1; j++) {
        if (indices[j]+bs != indices[j+1]) { /* meet end of a copy (and next copy must exist) */
          my_copies++;
          plan->copy_starts[k+1] = indices[j+1];
          plan->copy_lengths[k]  = sizeof(PetscScalar)*(indices[j]+bs-plan->copy_starts[k]);
          k++;
        }
      }
      /* set copy length of the last copy for this remote proc */
      plan->copy_lengths[k] = sizeof(PetscScalar)*(indices[j]+bs-plan->copy_starts[k]);
      k++;
    }

    /* set offset for next proc. When optimized[i] is false, copy_offsets[i] = copy_offsets[i+1] */
    plan->copy_offsets[i+1] = k;
  }

  /* try the last chance to optimize. If a scatter is not memory copies, then is it strided? */
  has_strided = PETSC_FALSE;
  ierr = PetscMalloc3(n,&plan->stride_first,n,&plan->stride_step,n,&plan->stride_n);CHKERRQ(ierr);
  for (i=0; i<n; i++) { /* for each target processor procs[i] */
    if (!plan->optimized[i] && starts[i+1] - starts[i] >= 16) { /* few indices (<16) are not worth striding */
      strided = PETSC_TRUE;
      step    = indices[starts[i]+1] - indices[starts[i]];
      for (j=starts[i]; j<starts[i+1]-1; j++) {
        if (indices[j]+step != indices[j+1]) { strided = PETSC_FALSE; break; }
      }
      if (strided) {
        plan->optimized[i]    = PETSC_TRUE;
        plan->stride_first[i] = indices[starts[i]];
        plan->stride_step[i]  = step;
        plan->stride_n[i]     = starts[i+1] - starts[i];
        has_strided           = PETSC_TRUE;
      }
    }
  }
  /* if none is strided, free the arrays to save memory here and also in plan copying */
  if (!has_strided) { ierr = PetscFree3(plan->stride_first,plan->stride_step,plan->stride_n);CHKERRQ(ierr); }
  PetscFunctionReturn(0);
}

/* --------------------------------------------------------------------------------------*/
/* Create a memcpy plan for a sequential general (SG) to SG scatter */
PetscErrorCode VecScatterMemcpyPlanCreate_SGToSG(PetscInt bs,VecScatter_Seq_General *to,VecScatter_Seq_General *from)
{
  PetscInt       n = to->n,i,*to_slots = to->vslots,*from_slots = from->vslots;
  PetscInt       j,n_copies;
  PetscErrorCode ierr;
  PetscBool      same_copy_starts;

  PetscFunctionBegin;
  ierr                = PetscMemzero(&to->memcpy_plan,sizeof(VecScatterMemcpyPlan));CHKERRQ(ierr);
  ierr                = PetscMemzero(&from->memcpy_plan,sizeof(VecScatterMemcpyPlan));CHKERRQ(ierr);
  to->memcpy_plan.n   = 1;
  from->memcpy_plan.n = 1;

  /* malloc and init the two fields to false and zero */
  ierr = PetscCalloc2(1,&to->memcpy_plan.optimized,2,&to->memcpy_plan.copy_offsets);CHKERRQ(ierr);
  ierr = PetscCalloc2(1,&from->memcpy_plan.optimized,2,&from->memcpy_plan.copy_offsets);CHKERRQ(ierr);

  /* count number of copies, which runs from 1 to n */
  n_copies = 1;
  for (i=0; i<n-1; i++) {
    if (to_slots[i]+bs != to_slots[i+1] || from_slots[i]+bs != from_slots[i+1]) n_copies++;
  }

  /* if average copy size >= 256 bytes, use memcpy instead of load/store */
  if (bs*n*sizeof(PetscScalar)/n_copies >= 256) {
    ierr = PetscMalloc2(n_copies,&to->memcpy_plan.copy_starts,n_copies,&to->memcpy_plan.copy_lengths);CHKERRQ(ierr);
    ierr = PetscMalloc2(n_copies,&from->memcpy_plan.copy_starts,n_copies,&from->memcpy_plan.copy_lengths);CHKERRQ(ierr);

    /* set up copy_starts[] & copy_lenghts[] of to and from */
    to->memcpy_plan.copy_starts[0]   = to_slots[0];
    from->memcpy_plan.copy_starts[0] = from_slots[0];

    if (n_copies != 1) { /* one copy is trival and we can save some work */
      j = 0;  /* j-th copy */
      for (i=0; i<n-1; i++) {
        if (to_slots[i]+bs != to_slots[i+1] || from_slots[i]+bs != from_slots[i+1]) {
          to->memcpy_plan.copy_lengths[j]    = sizeof(PetscScalar)*(to_slots[i]+bs-to->memcpy_plan.copy_starts[j]);
          from->memcpy_plan.copy_lengths[j]  = sizeof(PetscScalar)*(from_slots[i]+bs-from->memcpy_plan.copy_starts[j]);
          to->memcpy_plan.copy_starts[j+1]   = to_slots[i+1];
          from->memcpy_plan.copy_starts[j+1] = from_slots[i+1];
          j++;
        }
      }
    }

    /* set up copy_lengths[] of the last copy */
    to->memcpy_plan.copy_lengths[n_copies-1]   = sizeof(PetscScalar)*(to_slots[n-1]+bs-to->memcpy_plan.copy_starts[n_copies-1]);
    from->memcpy_plan.copy_lengths[n_copies-1] = sizeof(PetscScalar)*(from_slots[n-1]+bs-from->memcpy_plan.copy_starts[n_copies-1]);

    /* check if to and from have the same copy_starts[] values */
    same_copy_starts = PETSC_TRUE;
    for (i=0; i<n_copies; i++) {
      if (to->memcpy_plan.copy_starts[i] != from->memcpy_plan.copy_starts[i]) { same_copy_starts = PETSC_FALSE; break; }
    }

    to->memcpy_plan.optimized[0]        = PETSC_TRUE;
    from->memcpy_plan.optimized[0]      = PETSC_TRUE;
    to->memcpy_plan.copy_offsets[1]     = n_copies;
    from->memcpy_plan.copy_offsets[1]   = n_copies;
    to->memcpy_plan.same_copy_starts    = same_copy_starts;
    from->memcpy_plan.same_copy_starts  = same_copy_starts;
  }

  /* we do not do stride optimzation for this kind of scatter since the chance is rare. All related fields are zeroed out */
  PetscFunctionReturn(0);
}

/* Copy the memcpy plan from in to out */
PetscErrorCode VecScatterMemcpyPlanCopy(const VecScatterMemcpyPlan *in,VecScatterMemcpyPlan *out)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr   = PetscMemzero(out,sizeof(VecScatterMemcpyPlan));CHKERRQ(ierr);
  out->n = in->n;
  ierr   = PetscMalloc2(in->n,&out->optimized,in->n+1,&out->copy_offsets);CHKERRQ(ierr);
  ierr   = PetscMalloc2(in->copy_offsets[in->n],&out->copy_starts,in->copy_offsets[in->n],&out->copy_lengths);CHKERRQ(ierr);
  ierr   = PetscMemcpy(out->optimized,in->optimized,sizeof(PetscBool)*in->n);CHKERRQ(ierr);
  ierr   = PetscMemcpy(out->copy_offsets,in->copy_offsets,sizeof(PetscInt)*(in->n+1));CHKERRQ(ierr);
  ierr   = PetscMemcpy(out->copy_starts,in->copy_starts,sizeof(PetscInt)*in->copy_offsets[in->n]);CHKERRQ(ierr);
  ierr   = PetscMemcpy(out->copy_lengths,in->copy_lengths,sizeof(PetscInt)*in->copy_offsets[in->n]);CHKERRQ(ierr);
  if (in->stride_first) {
    ierr = PetscMalloc3(in->n,&out->stride_first,in->n,&out->stride_step,in->n,&out->stride_n);CHKERRQ(ierr);
    ierr = PetscMemcpy(out->stride_first,in->stride_first,sizeof(PetscInt)*in->n);CHKERRQ(ierr);
    ierr = PetscMemcpy(out->stride_step,in->stride_step,sizeof(PetscInt)*in->n);CHKERRQ(ierr);
    ierr = PetscMemcpy(out->stride_n,in->stride_n,sizeof(PetscInt)*in->n);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/* Destroy the vecscatter memcpy plan */
PetscErrorCode VecScatterMemcpyPlanDestroy(VecScatterMemcpyPlan *plan)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFree2(plan->optimized,plan->copy_offsets);CHKERRQ(ierr);
  ierr = PetscFree2(plan->copy_starts,plan->copy_lengths);CHKERRQ(ierr);
  ierr = PetscFree3(plan->stride_first,plan->stride_step,plan->stride_n);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#if defined(PETSC_HAVE_MPI_WIN_CREATE_FEATURE)
extern PetscErrorCode VecScatterCreateLocal_PtoS_MPI3(PetscInt,const PetscInt*,PetscInt,const PetscInt*,Vec,Vec,PetscInt,VecScatter);
extern PetscErrorCode VecScatterCreateLocal_PtoP_MPI3(PetscInt,const PetscInt*,PetscInt,const PetscInt*,Vec,Vec,PetscInt,VecScatter);
extern PetscErrorCode VecScatterCreateLocal_StoP_MPI3(PetscInt,const PetscInt*,PetscInt,const PetscInt*,Vec,Vec,PetscInt,VecScatter);
#endif

extern PetscErrorCode VecScatterCreateLocal_PtoS_MPI1(PetscInt,const PetscInt*,PetscInt,const PetscInt*,Vec,Vec,PetscInt,VecScatter);
extern PetscErrorCode VecScatterCreateLocal_PtoP_MPI1(PetscInt,const PetscInt*,PetscInt,const PetscInt*,Vec,Vec,PetscInt,VecScatter);
extern PetscErrorCode VecScatterCreateLocal_StoP_MPI1(PetscInt,const PetscInt*,PetscInt,const PetscInt*,Vec,Vec,PetscInt,VecScatter);

static PetscErrorCode GetInputISType_private(VecScatter,PetscInt,PetscInt,PetscInt*,IS*,PetscInt*,IS*);

/* =======================================================================*/
#define VEC_SEQ_ID 0
#define VEC_MPI_ID 1
#define IS_GENERAL_ID 0
#define IS_STRIDE_ID  1
#define IS_BLOCK_ID   2

/*
   Blocksizes we have optimized scatters for
*/

#define VecScatterOptimizedBS(mbs) (2 <= mbs)


PetscErrorCode  VecScatterCreateEmpty(MPI_Comm comm,VecScatter *newctx)
{
  VecScatter     ctx;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscHeaderCreate(ctx,VEC_SCATTER_CLASSID,"VecScatter","VecScatter","Vec",comm,VecScatterDestroy,VecScatterView);CHKERRQ(ierr);
  ctx->inuse               = PETSC_FALSE;
  ctx->beginandendtogether = PETSC_FALSE;
  ierr = PetscOptionsGetBool(NULL,NULL,"-vecscatter_merge",&ctx->beginandendtogether,NULL);CHKERRQ(ierr);
  if (ctx->beginandendtogether) {
    ierr = PetscInfo(ctx,"Using combined (merged) vector scatter begin and end\n");CHKERRQ(ierr);
  }
  *newctx = ctx;
  PetscFunctionReturn(0);
}

/* -------------------------------------------------- */
PetscErrorCode VecScatterCreate_Seq(VecScatter ctx)
{
  PetscErrorCode    ierr;
  PetscInt          ix_type=-1,iy_type=-1;
  IS                tix=NULL,tiy=NULL,ix=ctx->from_is,iy=ctx->to_is;
  Vec               xin=ctx->from_v;

  PetscFunctionBegin;
  ierr = PetscObjectChangeTypeName((PetscObject)ctx,VECSCATTERSEQ);CHKERRQ(ierr);
  ierr = GetInputISType_private(ctx,VEC_SEQ_ID,VEC_SEQ_ID,&ix_type,&tix,&iy_type,&tiy);CHKERRQ(ierr);
  if (tix) ix = tix;
  if (tiy) iy = tiy;

  if (ix_type == IS_GENERAL_ID && iy_type == IS_GENERAL_ID) {
    PetscInt               nx,ny;
    const PetscInt         *idx,*idy;
    VecScatter_Seq_General *to = NULL,*from = NULL;

    ierr = ISGetLocalSize(ix,&nx);CHKERRQ(ierr);
    ierr = ISGetLocalSize(iy,&ny);CHKERRQ(ierr);
    if (nx != ny) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Local scatter sizes don't match");
    ierr  = ISGetIndices(ix,&idx);CHKERRQ(ierr);
    ierr  = ISGetIndices(iy,&idy);CHKERRQ(ierr);
    ierr  = PetscMalloc2(1,&to,1,&from);CHKERRQ(ierr);
    ierr  = PetscMalloc2(nx,&to->vslots,nx,&from->vslots);CHKERRQ(ierr);
    to->n = nx;
#if defined(PETSC_USE_DEBUG)
    ierr = VecScatterCheckIndices_Private(ctx->to_n,ny,idy);CHKERRQ(ierr);
#endif
    ierr    = PetscMemcpy(to->vslots,idy,nx*sizeof(PetscInt));CHKERRQ(ierr);
    from->n = nx;
#if defined(PETSC_USE_DEBUG)
    ierr = VecScatterCheckIndices_Private(ctx->from_n,nx,idx);CHKERRQ(ierr);
#endif
    ierr              =  PetscMemcpy(from->vslots,idx,nx*sizeof(PetscInt));CHKERRQ(ierr);
    to->format          = VEC_SCATTER_SEQ_GENERAL;
    from->format        = VEC_SCATTER_SEQ_GENERAL;
    ctx->todata       = (void*)to;
    ctx->fromdata     = (void*)from;
    ierr              = VecScatterMemcpyPlanCreate_SGToSG(1,to,from);CHKERRQ(ierr);
    ctx->ops->begin   = VecScatterBegin_SGToSG;
    ctx->ops->end     = 0;
    ctx->ops->destroy = VecScatterDestroy_SGToSG;
    ctx->ops->copy    = VecScatterCopy_SGToSG;
    ctx->ops->view    = VecScatterView_SGToSG;
    ierr              = PetscInfo(xin,"Special case: sequential vector general scatter\n");CHKERRQ(ierr);
    goto functionend;
  } else if (ix_type == IS_STRIDE_ID &&  iy_type == IS_STRIDE_ID) {
    PetscInt              nx,ny,to_first,to_step,from_first,from_step;
    VecScatter_Seq_Stride *from8 = NULL,*to8 = NULL;

    ierr = ISGetLocalSize(ix,&nx);CHKERRQ(ierr);
    ierr = ISGetLocalSize(iy,&ny);CHKERRQ(ierr);
    if (nx != ny) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Local scatter sizes don't match");
    ierr              = ISStrideGetInfo(iy,&to_first,&to_step);CHKERRQ(ierr);
    ierr              = ISStrideGetInfo(ix,&from_first,&from_step);CHKERRQ(ierr);
    ierr              = PetscMalloc2(1,&to8,1,&from8);CHKERRQ(ierr);
    to8->n            = nx;
    to8->first        = to_first;
    to8->step         = to_step;
    from8->n          = nx;
    from8->first      = from_first;
    from8->step       = from_step;
    to8->format         = VEC_SCATTER_SEQ_STRIDE;
    from8->format       = VEC_SCATTER_SEQ_STRIDE;
    ctx->todata       = (void*)to8;
    ctx->fromdata     = (void*)from8;
    ctx->ops->begin   = VecScatterBegin_SSToSS;
    ctx->ops->end     = 0;
    ctx->ops->destroy = VecScatterDestroy_SSToSS;
    ctx->ops->copy    = VecScatterCopy_SSToSS;
    ctx->ops->view    = VecScatterView_SSToSS;
    ierr          = PetscInfo(xin,"Special case: sequential vector stride to stride\n");CHKERRQ(ierr);
    goto functionend;
  } else if (ix_type == IS_GENERAL_ID && iy_type == IS_STRIDE_ID) {
    PetscInt               nx,ny,first,step;
    const PetscInt         *idx;
    VecScatter_Seq_General *from9 = NULL;
    VecScatter_Seq_Stride  *to9   = NULL;

    ierr = ISGetLocalSize(ix,&nx);CHKERRQ(ierr);
    ierr = ISGetIndices(ix,&idx);CHKERRQ(ierr);
    ierr = ISGetLocalSize(iy,&ny);CHKERRQ(ierr);
    ierr = ISStrideGetInfo(iy,&first,&step);CHKERRQ(ierr);
    if (nx != ny) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Local scatter sizes don't match");
    ierr       = PetscMalloc2(1,&to9,1,&from9);CHKERRQ(ierr);
    ierr       = PetscMemzero(&from9->memcpy_plan,sizeof(VecScatterMemcpyPlan));CHKERRQ(ierr);
    ierr       = PetscMalloc1(nx,&from9->vslots);CHKERRQ(ierr);
    to9->n     = nx;
    to9->first = first;
    to9->step  = step;
    from9->n   = nx;
#if defined(PETSC_USE_DEBUG)
    ierr           = VecScatterCheckIndices_Private(ctx->from_n,nx,idx);CHKERRQ(ierr);
#endif
    ierr           = PetscMemcpy(from9->vslots,idx,nx*sizeof(PetscInt));CHKERRQ(ierr);
    ctx->todata    = (void*)to9; ctx->fromdata = (void*)from9;
    if (step == 1) {
      PetscInt tmp[2];
      tmp[0] = 0; tmp[1] = nx;
      ierr = VecScatterMemcpyPlanCreate_Index(1,tmp,from9->vslots,1,&from9->memcpy_plan);CHKERRQ(ierr);
      ctx->ops->begin = VecScatterBegin_SGToSS_Stride1;
    } else {
      ctx->ops->begin = VecScatterBegin_SGToSS;
    }
    ctx->ops->destroy   = VecScatterDestroy_SGToSS;
    ctx->ops->end       = 0;
    ctx->ops->copy      = VecScatterCopy_SGToSS;
    ctx->ops->view      = VecScatterView_SGToSS;
    to9->format      = VEC_SCATTER_SEQ_STRIDE;
    from9->format    = VEC_SCATTER_SEQ_GENERAL;
    ierr = PetscInfo(xin,"Special case: sequential vector general to stride\n");CHKERRQ(ierr);
    goto functionend;
  } else if (ix_type == IS_STRIDE_ID && iy_type == IS_GENERAL_ID) {
    PetscInt               nx,ny,first,step;
    const PetscInt         *idy;
    VecScatter_Seq_General *to10 = NULL;
    VecScatter_Seq_Stride  *from10 = NULL;

    ierr = ISGetLocalSize(ix,&nx);CHKERRQ(ierr);
    ierr = ISGetIndices(iy,&idy);CHKERRQ(ierr);
    ierr = ISGetLocalSize(iy,&ny);CHKERRQ(ierr);
    ierr = ISStrideGetInfo(ix,&first,&step);CHKERRQ(ierr);
    if (nx != ny) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Local scatter sizes don't match");
    ierr = PetscMalloc2(1,&to10,1,&from10);CHKERRQ(ierr);
    ierr = PetscMemzero(&to10->memcpy_plan,sizeof(VecScatterMemcpyPlan));CHKERRQ(ierr);
    ierr = PetscMalloc1(nx,&to10->vslots);CHKERRQ(ierr);
    from10->n     = nx;
    from10->first = first;
    from10->step  = step;
    to10->n       = nx;
#if defined(PETSC_USE_DEBUG)
    ierr = VecScatterCheckIndices_Private(ctx->to_n,ny,idy);CHKERRQ(ierr);
#endif
    ierr = PetscMemcpy(to10->vslots,idy,nx*sizeof(PetscInt));CHKERRQ(ierr);
    ctx->todata   = (void*)to10;
    ctx->fromdata = (void*)from10;
    if (step == 1) {
      PetscInt tmp[2];
      tmp[0] = 0; tmp[1] = nx;
      ierr = VecScatterMemcpyPlanCreate_Index(1,tmp,to10->vslots,1,&to10->memcpy_plan);CHKERRQ(ierr);
      ctx->ops->begin = VecScatterBegin_SSToSG_Stride1;
    } else {
      ctx->ops->begin = VecScatterBegin_SSToSG;
    }
    ctx->ops->destroy = VecScatterDestroy_SSToSG;
    ctx->ops->end     = 0;
    ctx->ops->copy    = 0;
    ctx->ops->view    = VecScatterView_SSToSG;
    to10->format   = VEC_SCATTER_SEQ_GENERAL;
    from10->format = VEC_SCATTER_SEQ_STRIDE;
    ierr = PetscInfo(xin,"Special case: sequential vector stride to general\n");CHKERRQ(ierr);
    goto functionend;
  } else {
    PetscInt               nx,ny;
    const PetscInt         *idx,*idy;
    VecScatter_Seq_General *to11 = NULL,*from11 = NULL;
    PetscBool              idnx,idny;

    ierr = ISGetLocalSize(ix,&nx);CHKERRQ(ierr);
    ierr = ISGetLocalSize(iy,&ny);CHKERRQ(ierr);
    if (nx != ny) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Local scatter sizes don't match, in %D out %D",nx,ny);

    ierr = ISIdentity(ix,&idnx);CHKERRQ(ierr);
    ierr = ISIdentity(iy,&idny);CHKERRQ(ierr);
    if (idnx && idny) {
      VecScatter_Seq_Stride *to13 = NULL,*from13 = NULL;
      ierr          = PetscMalloc2(1,&to13,1,&from13);CHKERRQ(ierr);
      to13->n       = nx;
      to13->first   = 0;
      to13->step    = 1;
      from13->n     = nx;
      from13->first = 0;
      from13->step  = 1;
      to13->format    = VEC_SCATTER_SEQ_STRIDE;
      from13->format  = VEC_SCATTER_SEQ_STRIDE;
      ctx->todata   = (void*)to13;
      ctx->fromdata = (void*)from13;
      ctx->ops->begin    = VecScatterBegin_SSToSS;
      ctx->ops->end      = 0;
      ctx->ops->destroy  = VecScatterDestroy_SSToSS;
      ctx->ops->copy     = VecScatterCopy_SSToSS;
      ctx->ops->view     = VecScatterView_SSToSS;
      ierr = PetscInfo(xin,"Special case: sequential copy\n");CHKERRQ(ierr);
      goto functionend;
    }

    ierr = ISGetIndices(iy,&idy);CHKERRQ(ierr);
    ierr = ISGetIndices(ix,&idx);CHKERRQ(ierr);
    ierr = PetscMalloc2(1,&to11,1,&from11);CHKERRQ(ierr);
    ierr = PetscMalloc2(nx,&to11->vslots,nx,&from11->vslots);CHKERRQ(ierr);
    to11->n = nx;
#if defined(PETSC_USE_DEBUG)
    ierr = VecScatterCheckIndices_Private(ctx->to_n,ny,idy);CHKERRQ(ierr);
#endif
    ierr = PetscMemcpy(to11->vslots,idy,nx*sizeof(PetscInt));CHKERRQ(ierr);
    from11->n = nx;
#if defined(PETSC_USE_DEBUG)
    ierr = VecScatterCheckIndices_Private(ctx->from_n,nx,idx);CHKERRQ(ierr);
#endif
    ierr = PetscMemcpy(from11->vslots,idx,nx*sizeof(PetscInt));CHKERRQ(ierr);
    to11->format        = VEC_SCATTER_SEQ_GENERAL;
    from11->format      = VEC_SCATTER_SEQ_GENERAL;
    ctx->todata       = (void*)to11;
    ctx->fromdata     = (void*)from11;
    ctx->ops->begin   = VecScatterBegin_SGToSG;
    ctx->ops->end     = 0;
    ctx->ops->destroy = VecScatterDestroy_SGToSG;
    ctx->ops->copy    = VecScatterCopy_SGToSG;
    ctx->ops->view    = VecScatterView_SGToSG;
    ierr = VecScatterMemcpyPlanCreate_SGToSG(1,to11,from11);CHKERRQ(ierr);
    ierr = ISRestoreIndices(ix,&idx);CHKERRQ(ierr);
    ierr = ISRestoreIndices(iy,&idy);CHKERRQ(ierr);
    ierr = PetscInfo(xin,"Sequential vector scatter with block indices\n");CHKERRQ(ierr);
    goto functionend;
  }
  functionend:
  ierr = ISDestroy(&tix);CHKERRQ(ierr);
  ierr = ISDestroy(&tiy);CHKERRQ(ierr);
  ierr = VecScatterViewFromOptions(ctx,NULL,"-vecscatter_view");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode VecScatterCreate_PtoS(VecScatter ctx)
{
  PetscErrorCode    ierr;
  PetscMPIInt       size;
  PetscInt          ix_type=-1,iy_type=-1,*range;
  MPI_Comm          comm;
  IS                tix=NULL,tiy=NULL,ix=ctx->from_is,iy=ctx->to_is;
  Vec               xin=ctx->from_v,yin=ctx->to_v;
  VecScatterType    type;
  PetscBool         vec_mpi1_flg;
  PetscBool         totalv,ixblock,iyblock,iystride,islocal,cando;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)ctx,&comm);CHKERRQ(ierr);
  ierr = GetInputISType_private(ctx,VEC_MPI_ID,VEC_SEQ_ID,&ix_type,&tix,&iy_type,&tiy);CHKERRQ(ierr);
  if (tix) ix = tix;
  if (tiy) iy = tiy;

  ierr = VecScatterGetType(ctx,&type);CHKERRQ(ierr);
  ierr = PetscStrcmp(type,"mpi1",&vec_mpi1_flg);CHKERRQ(ierr);

  islocal = PETSC_FALSE;
  /* special case extracting (subset of) local portion */
  if (ix_type == IS_STRIDE_ID && iy_type == IS_STRIDE_ID) {
    /* Case (2-a) */
    PetscInt              nx,ny,to_first,to_step,from_first,from_step;
    PetscInt              start,end,min,max;
    VecScatter_Seq_Stride *from12 = NULL,*to12 = NULL;

    ierr = VecGetOwnershipRange(xin,&start,&end);CHKERRQ(ierr);
    ierr = ISGetLocalSize(ix,&nx);CHKERRQ(ierr);
    ierr = ISStrideGetInfo(ix,&from_first,&from_step);CHKERRQ(ierr);
    ierr = ISGetLocalSize(iy,&ny);CHKERRQ(ierr);
    ierr = ISStrideGetInfo(iy,&to_first,&to_step);CHKERRQ(ierr);
    if (nx != ny) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Local scatter sizes don't match");
    ierr = ISGetMinMax(ix,&min,&max);CHKERRQ(ierr);
    if (min >= start && max < end) islocal = PETSC_TRUE;
    else islocal = PETSC_FALSE;
    /* cannot use MPIU_Allreduce() since this call matches with the MPI_Allreduce() in the else statement below */
    ierr = MPI_Allreduce(&islocal,&cando,1,MPIU_BOOL,MPI_LAND,PetscObjectComm((PetscObject)xin));CHKERRQ(ierr);
    if (cando) {
      ierr               = PetscMalloc2(1,&to12,1,&from12);CHKERRQ(ierr);
      to12->n            = nx;
      to12->first        = to_first;
      to12->step         = to_step;
      from12->n          = nx;
      from12->first      = from_first-start;
      from12->step       = from_step;
      to12->format         = VEC_SCATTER_SEQ_STRIDE;
      from12->format       = VEC_SCATTER_SEQ_STRIDE;
      ctx->todata        = (void*)to12;
      ctx->fromdata      = (void*)from12;
      ctx->ops->begin    = VecScatterBegin_SSToSS;
      ctx->ops->end      = 0;
      ctx->ops->destroy  = VecScatterDestroy_SSToSS;
      ctx->ops->copy     = VecScatterCopy_SSToSS;
      ctx->ops->view     = VecScatterView_SSToSS;
      ierr = PetscInfo(xin,"Special case: processors only getting local values\n");CHKERRQ(ierr);
      goto functionend;
    }
  } else {
    ierr = MPI_Allreduce(&islocal,&cando,1,MPIU_BOOL,MPI_LAND,PetscObjectComm((PetscObject)xin));CHKERRQ(ierr);
  }

  /* test for special case of all processors getting entire vector */
  /* contains check that PetscMPIInt can handle the sizes needed */
  totalv = PETSC_FALSE;
  if (ix_type == IS_STRIDE_ID && iy_type == IS_STRIDE_ID) {
    /* Case (2-b) */
    PetscInt             i,nx,ny,to_first,to_step,from_first,from_step,N;
    PetscMPIInt          *count = NULL,*displx;
    VecScatter_MPI_ToAll *sto   = NULL;

    ierr = ISGetLocalSize(ix,&nx);CHKERRQ(ierr);
    ierr = ISStrideGetInfo(ix,&from_first,&from_step);CHKERRQ(ierr);
    ierr = ISGetLocalSize(iy,&ny);CHKERRQ(ierr);
    ierr = ISStrideGetInfo(iy,&to_first,&to_step);CHKERRQ(ierr);
    if (nx != ny) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Local scatter sizes don't match");
    ierr = VecGetSize(xin,&N);CHKERRQ(ierr);
    if (nx != N) totalv = PETSC_FALSE;
    else if (from_first == 0 && from_step == 1 && from_first == to_first && from_step == to_step) totalv = PETSC_TRUE;
    else totalv = PETSC_FALSE;
    /* cannot use MPIU_Allreduce() since this call matches with the MPI_Allreduce() in the else statement below */
    ierr = MPI_Allreduce(&totalv,&cando,1,MPIU_BOOL,MPI_LAND,PetscObjectComm((PetscObject)xin));CHKERRQ(ierr);

#if defined(PETSC_USE_64BIT_INDICES)
    if (cando && (yin->map->N < PETSC_MPI_INT_MAX)) {
#else
    if (cando) {
#endif
      ierr  = MPI_Comm_size(PetscObjectComm((PetscObject)ctx),&size);CHKERRQ(ierr);
      ierr  = PetscMalloc3(1,&sto,size,&count,size,&displx);CHKERRQ(ierr);
      range = xin->map->range;
      for (i=0; i<size; i++) {
        ierr = PetscMPIIntCast(range[i+1] - range[i],count+i);CHKERRQ(ierr);
        ierr = PetscMPIIntCast(range[i],displx+i);CHKERRQ(ierr);
      }
      sto->count        = count;
      sto->displx       = displx;
      sto->work1        = 0;
      sto->work2        = 0;
      sto->format         = VEC_SCATTER_MPI_TOALL;
      ctx->todata       = (void*)sto;
      ctx->fromdata     = 0;
      ctx->ops->begin   = VecScatterBegin_MPI_ToAll;
      ctx->ops->end     = 0;
      ctx->ops->destroy = VecScatterDestroy_MPI_ToAll;
      ctx->ops->copy    = VecScatterCopy_MPI_ToAll;
      ctx->ops->view    = VecScatterView_MPI_ToAll;
      ierr = PetscInfo(xin,"Special case: all processors get entire parallel vector\n");CHKERRQ(ierr);
      goto functionend;
    }
  } else {
    ierr = MPI_Allreduce(&totalv,&cando,1,MPIU_BOOL,MPI_LAND,PetscObjectComm((PetscObject)xin));CHKERRQ(ierr);
  }

  /* test for special case of processor 0 getting entire vector */
  /* contains check that PetscMPIInt can handle the sizes needed */
  totalv = PETSC_FALSE;
  if (ix_type == IS_STRIDE_ID && iy_type == IS_STRIDE_ID) {
    /* Case (2-c) */
    PetscInt             i,nx,ny,to_first,to_step,from_first,from_step,N;
    PetscMPIInt          rank,*count = NULL,*displx;
    VecScatter_MPI_ToAll *sto = NULL;

    ierr = PetscObjectGetComm((PetscObject)xin,&comm);CHKERRQ(ierr);
    ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
    ierr = ISGetLocalSize(ix,&nx);CHKERRQ(ierr);
    ierr = ISStrideGetInfo(ix,&from_first,&from_step);CHKERRQ(ierr);
    ierr = ISGetLocalSize(iy,&ny);CHKERRQ(ierr);
    ierr = ISStrideGetInfo(iy,&to_first,&to_step);CHKERRQ(ierr);
    if (nx != ny) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Local scatter sizes don't match");
    if (!rank) {
      ierr = VecGetSize(xin,&N);CHKERRQ(ierr);
      if (nx != N) totalv = PETSC_FALSE;
      else if (from_first == 0        && from_step == 1 &&
               from_first == to_first && from_step == to_step) totalv = PETSC_TRUE;
      else totalv = PETSC_FALSE;
    } else {
      if (!nx) totalv = PETSC_TRUE;
      else     totalv = PETSC_FALSE;
    }
    /* cannot use MPIU_Allreduce() since this call matches with the MPI_Allreduce() in the else statement below */
    ierr = MPI_Allreduce(&totalv,&cando,1,MPIU_BOOL,MPI_LAND,PetscObjectComm((PetscObject)xin));CHKERRQ(ierr);

#if defined(PETSC_USE_64BIT_INDICES)
    if (cando && (yin->map->N < PETSC_MPI_INT_MAX)) {
#else
    if (cando) {
#endif
      ierr  = MPI_Comm_size(PetscObjectComm((PetscObject)ctx),&size);CHKERRQ(ierr);
      ierr  = PetscMalloc3(1,&sto,size,&count,size,&displx);CHKERRQ(ierr);
      range = xin->map->range;
      for (i=0; i<size; i++) {
        ierr = PetscMPIIntCast(range[i+1] - range[i],count+i);CHKERRQ(ierr);
        ierr = PetscMPIIntCast(range[i],displx+i);CHKERRQ(ierr);
      }
      sto->count        = count;
      sto->displx       = displx;
      sto->work1        = 0;
      sto->work2        = 0;
      sto->format         = VEC_SCATTER_MPI_TOONE;
      ctx->todata       = (void*)sto;
      ctx->fromdata     = 0;
      ctx->ops->begin   = VecScatterBegin_MPI_ToOne;
      ctx->ops->end     = 0;
      ctx->ops->destroy = VecScatterDestroy_MPI_ToAll;
      ctx->ops->copy    = VecScatterCopy_MPI_ToAll;
      ctx->ops->view    = VecScatterView_MPI_ToAll;
      ierr = PetscInfo(xin,"Special case: processor zero gets entire parallel vector, rest get none\n");CHKERRQ(ierr);
      goto functionend;
    }
  } else {
    ierr = MPI_Allreduce(&totalv,&cando,1,MPIU_BOOL,MPI_LAND,PetscObjectComm((PetscObject)xin));CHKERRQ(ierr);
  }

  /* Case 2-d */
  ierr = PetscObjectTypeCompare((PetscObject)ix,ISBLOCK,&ixblock);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)iy,ISBLOCK,&iyblock);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)iy,ISSTRIDE,&iystride);CHKERRQ(ierr);
  if (ixblock) {
    /* special case block to block */
    if (iyblock) {
      PetscInt       nx,ny,bsx,bsy;
      const PetscInt *idx,*idy;
      ierr = ISGetBlockSize(iy,&bsy);CHKERRQ(ierr);
      ierr = ISGetBlockSize(ix,&bsx);CHKERRQ(ierr);
      if (bsx == bsy && VecScatterOptimizedBS(bsx)) {
        ierr = ISBlockGetLocalSize(ix,&nx);CHKERRQ(ierr);
        ierr = ISBlockGetIndices(ix,&idx);CHKERRQ(ierr);
        ierr = ISBlockGetLocalSize(iy,&ny);CHKERRQ(ierr);
        ierr = ISBlockGetIndices(iy,&idy);CHKERRQ(ierr);
        if (nx != ny) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Local scatter sizes don't match");
#if defined(PETSC_HAVE_MPI_WIN_CREATE_FEATURE)
        if (vec_mpi1_flg) {
          ierr = VecScatterCreateLocal_PtoS_MPI1(nx,idx,ny,idy,xin,yin,bsx,ctx);CHKERRQ(ierr);
        } else {
          ierr = VecScatterCreateLocal_PtoS_MPI3(nx,idx,ny,idy,xin,yin,bsx,ctx);CHKERRQ(ierr);
        }
#else
        ierr = VecScatterCreateLocal_PtoS_MPI1(nx,idx,ny,idy,xin,yin,bsx,ctx);CHKERRQ(ierr);
#endif
        ierr = ISBlockRestoreIndices(ix,&idx);CHKERRQ(ierr);
        ierr = ISBlockRestoreIndices(iy,&idy);CHKERRQ(ierr);
        ierr = PetscInfo(xin,"Special case: blocked indices\n");CHKERRQ(ierr);
        goto functionend;
      }
      /* special case block to stride */
    } else if (iystride) {
      /* Case (2-e) */
      PetscInt ystart,ystride,ysize,bsx;
      ierr = ISStrideGetInfo(iy,&ystart,&ystride);CHKERRQ(ierr);
      ierr = ISGetLocalSize(iy,&ysize);CHKERRQ(ierr);
      ierr = ISGetBlockSize(ix,&bsx);CHKERRQ(ierr);
      /* see if stride index set is equivalent to block index set */
      if (VecScatterOptimizedBS(bsx) && ((ystart % bsx) == 0) && (ystride == 1) && ((ysize % bsx) == 0)) {
        PetscInt       nx,il,*idy;
        const PetscInt *idx;
        ierr = ISBlockGetLocalSize(ix,&nx);CHKERRQ(ierr);
        ierr = ISBlockGetIndices(ix,&idx);CHKERRQ(ierr);
        if (ysize != bsx*nx) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Local scatter sizes don't match");
        ierr = PetscMalloc1(nx,&idy);CHKERRQ(ierr);
        if (nx) {
          idy[0] = ystart/bsx;
          for (il=1; il<nx; il++) idy[il] = idy[il-1] + 1;
        }
#if defined(PETSC_HAVE_MPI_WIN_CREATE_FEATURE)
        if (vec_mpi1_flg) {
          ierr = VecScatterCreateLocal_PtoS_MPI1(nx,idx,nx,idy,xin,yin,bsx,ctx);CHKERRQ(ierr);
        } else {
          ierr = VecScatterCreateLocal_PtoS_MPI3(nx,idx,nx,idy,xin,yin,bsx,ctx);CHKERRQ(ierr);
        }
#else
        ierr = VecScatterCreateLocal_PtoS_MPI1(nx,idx,nx,idy,xin,yin,bsx,ctx);CHKERRQ(ierr);
#endif
        ierr = PetscFree(idy);CHKERRQ(ierr);
        ierr = ISBlockRestoreIndices(ix,&idx);CHKERRQ(ierr);
        ierr = PetscInfo(xin,"Special case: blocked indices to stride\n");CHKERRQ(ierr);
        goto functionend;
      }
    }
  }

  /* left over general case (2-f) */
  {
    PetscInt       nx,ny;
    const PetscInt *idx,*idy;
    ierr = ISGetLocalSize(ix,&nx);CHKERRQ(ierr);
    ierr = ISGetIndices(ix,&idx);CHKERRQ(ierr);
    ierr = ISGetLocalSize(iy,&ny);CHKERRQ(ierr);
    ierr = ISGetIndices(iy,&idy);CHKERRQ(ierr);
    if (nx != ny) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Local scatter sizes don't match (%D %D)",nx,ny);
#if defined(PETSC_HAVE_MPI_WIN_CREATE_FEATURE)
    if (vec_mpi1_flg) {
      ierr = VecScatterCreateLocal_PtoS_MPI1(nx,idx,ny,idy,xin,yin,1,ctx);CHKERRQ(ierr);
    } else {
      ierr = VecScatterCreateLocal_PtoS_MPI3(nx,idx,ny,idy,xin,yin,1,ctx);CHKERRQ(ierr);
    }
#else
    ierr = VecScatterCreateLocal_PtoS_MPI1(nx,idx,ny,idy,xin,yin,1,ctx);CHKERRQ(ierr);
#endif
    ierr = ISRestoreIndices(ix,&idx);CHKERRQ(ierr);
    ierr = ISRestoreIndices(iy,&idy);CHKERRQ(ierr);
    ierr = PetscInfo(xin,"General case: MPI to Seq\n");CHKERRQ(ierr);
    goto functionend;
  }
  functionend:
  ierr = ISDestroy(&tix);CHKERRQ(ierr);
  ierr = ISDestroy(&tiy);CHKERRQ(ierr);
  ierr = VecScatterViewFromOptions(ctx,NULL,"-vecscatter_view");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode VecScatterCreate_StoP(VecScatter ctx)
{
  PetscErrorCode    ierr;
  PetscInt          ix_type=-1,iy_type=-1;
  IS                tix=NULL,tiy=NULL,ix=ctx->from_is,iy=ctx->to_is;
  Vec               xin=ctx->from_v,yin=ctx->to_v;
  VecScatterType    type;
  PetscBool         vscat_mpi1;
  PetscBool         islocal,cando;

  PetscFunctionBegin;
  ierr = GetInputISType_private(ctx,VEC_MPI_ID,VEC_SEQ_ID,&ix_type,&tix,&iy_type,&tiy);CHKERRQ(ierr);
  if (tix) ix = tix;
  if (tiy) iy = tiy;

  ierr = VecScatterGetType(ctx,&type);CHKERRQ(ierr);
  ierr = PetscStrcmp(type,"mpi1",&vscat_mpi1);CHKERRQ(ierr);

  /* special case local copy portion */
  islocal = PETSC_FALSE;
  if (ix_type == IS_STRIDE_ID && iy_type == IS_STRIDE_ID) {
    PetscInt              nx,ny,to_first,to_step,from_step,start,end,from_first,min,max;
    VecScatter_Seq_Stride *from = NULL,*to = NULL;

    ierr = VecGetOwnershipRange(yin,&start,&end);CHKERRQ(ierr);
    ierr = ISGetLocalSize(ix,&nx);CHKERRQ(ierr);
    ierr = ISStrideGetInfo(ix,&from_first,&from_step);CHKERRQ(ierr);
    ierr = ISGetLocalSize(iy,&ny);CHKERRQ(ierr);
    ierr = ISStrideGetInfo(iy,&to_first,&to_step);CHKERRQ(ierr);
    if (nx != ny) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Local scatter sizes don't match");
    ierr = ISGetMinMax(iy,&min,&max);CHKERRQ(ierr);
    if (min >= start && max < end) islocal = PETSC_TRUE;
    else islocal = PETSC_FALSE;
    /* cannot use MPIU_Allreduce() since this call matches with the MPI_Allreduce() in the else statement below */
    ierr = MPI_Allreduce(&islocal,&cando,1,MPIU_BOOL,MPI_LAND,PetscObjectComm((PetscObject)yin));CHKERRQ(ierr);
    if (cando) {
      ierr              = PetscMalloc2(1,&to,1,&from);CHKERRQ(ierr);
      to->n             = nx;
      to->first         = to_first-start;
      to->step          = to_step;
      from->n           = nx;
      from->first       = from_first;
      from->step        = from_step;
      to->format          = VEC_SCATTER_SEQ_STRIDE;
      from->format        = VEC_SCATTER_SEQ_STRIDE;
      ctx->todata       = (void*)to;
      ctx->fromdata     = (void*)from;
      ctx->ops->begin   = VecScatterBegin_SSToSS;
      ctx->ops->end     = 0;
      ctx->ops->destroy = VecScatterDestroy_SSToSS;
      ctx->ops->copy    = VecScatterCopy_SSToSS;
      ctx->ops->view    = VecScatterView_SSToSS;
      ierr          = PetscInfo(xin,"Special case: sequential stride to MPI stride\n");CHKERRQ(ierr);
      goto functionend;
    }
  } else {
    ierr = MPI_Allreduce(&islocal,&cando,1,MPIU_BOOL,MPI_LAND,PetscObjectComm((PetscObject)yin));CHKERRQ(ierr);
  }
  /* special case block to stride */
  if (ix_type == IS_BLOCK_ID && iy_type == IS_STRIDE_ID) {
    PetscInt ystart,ystride,ysize,bsx;
    ierr = ISStrideGetInfo(iy,&ystart,&ystride);CHKERRQ(ierr);
    ierr = ISGetLocalSize(iy,&ysize);CHKERRQ(ierr);
    ierr = ISGetBlockSize(ix,&bsx);CHKERRQ(ierr);
    /* see if stride index set is equivalent to block index set */
    if (VecScatterOptimizedBS(bsx) && ((ystart % bsx) == 0) && (ystride == 1) && ((ysize % bsx) == 0)) {
      PetscInt       nx,il,*idy;
      const PetscInt *idx;
      ierr = ISBlockGetLocalSize(ix,&nx);CHKERRQ(ierr);
      ierr = ISBlockGetIndices(ix,&idx);CHKERRQ(ierr);
      if (ysize != bsx*nx) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Local scatter sizes don't match");
      ierr = PetscMalloc1(nx,&idy);CHKERRQ(ierr);
      if (nx) {
        idy[0] = ystart/bsx;
        for (il=1; il<nx; il++) idy[il] = idy[il-1] + 1;
      }
      if (vscat_mpi1) {
        ierr = VecScatterCreateLocal_StoP_MPI1(nx,idx,nx,idy,xin,yin,bsx,ctx);CHKERRQ(ierr);
      }
#if defined(PETSC_HAVE_MPI_WIN_CREATE_FEATURE)
      else {
        ierr = VecScatterCreateLocal_StoP_MPI3(nx,idx,nx,idy,xin,yin,bsx,ctx);CHKERRQ(ierr);
      }
#endif
      ierr = PetscFree(idy);CHKERRQ(ierr);
      ierr = ISBlockRestoreIndices(ix,&idx);CHKERRQ(ierr);
      ierr = PetscInfo(xin,"Special case: Blocked indices to stride\n");CHKERRQ(ierr);
      goto functionend;
    }
  }

  /* general case */
  {
    PetscInt       nx,ny;
    const PetscInt *idx,*idy;
    ierr = ISGetLocalSize(ix,&nx);CHKERRQ(ierr);
    ierr = ISGetIndices(ix,&idx);CHKERRQ(ierr);
    ierr = ISGetLocalSize(iy,&ny);CHKERRQ(ierr);
    ierr = ISGetIndices(iy,&idy);CHKERRQ(ierr);
    if (nx != ny) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Local scatter sizes don't match");
    if (vscat_mpi1) {
      ierr = VecScatterCreateLocal_StoP_MPI1(nx,idx,ny,idy,xin,yin,1,ctx);CHKERRQ(ierr);
    }
#if defined(PETSC_HAVE_MPI_WIN_CREATE_FEATURE)
    else {
      ierr = VecScatterCreateLocal_StoP_MPI3(nx,idx,ny,idy,xin,yin,1,ctx);CHKERRQ(ierr);
    }
#endif
    ierr = ISRestoreIndices(ix,&idx);CHKERRQ(ierr);
    ierr = ISRestoreIndices(iy,&idy);CHKERRQ(ierr);
    ierr = PetscInfo(xin,"General case: Seq to MPI\n");CHKERRQ(ierr);
    goto functionend;
  }
  functionend:
  ierr = ISDestroy(&tix);CHKERRQ(ierr);
  ierr = ISDestroy(&tiy);CHKERRQ(ierr);
  ierr = VecScatterViewFromOptions(ctx,NULL,"-vecscatter_view");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode VecScatterCreate_PtoP(VecScatter ctx)
{
  PetscErrorCode    ierr;
  PetscInt          ix_type=-1,iy_type=-1;
  IS                tix=NULL,tiy=NULL,ix=ctx->from_is,iy=ctx->to_is;
  Vec               xin=ctx->from_v,yin=ctx->to_v;
  VecScatterType    type;
  PetscBool         vscat_mpi1;
  PetscInt          nx,ny;
  const PetscInt    *idx,*idy;

  PetscFunctionBegin;
  ierr = GetInputISType_private(ctx,VEC_MPI_ID,VEC_MPI_ID,&ix_type,&tix,&iy_type,&tiy);CHKERRQ(ierr);
  if (tix) ix = tix;
  if (tiy) iy = tiy;

  ierr = VecScatterGetType(ctx,&type);CHKERRQ(ierr);
  ierr = PetscStrcmp(type,"mpi1",&vscat_mpi1);CHKERRQ(ierr);

  /* no special cases for now */
  ierr = ISGetLocalSize(ix,&nx);CHKERRQ(ierr);
  ierr = ISGetIndices(ix,&idx);CHKERRQ(ierr);
  ierr = ISGetLocalSize(iy,&ny);CHKERRQ(ierr);
  ierr = ISGetIndices(iy,&idy);CHKERRQ(ierr);
  if (nx != ny) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Local scatter sizes don't match");
  if (vscat_mpi1) {
    ierr = VecScatterCreateLocal_PtoP_MPI1(nx,idx,ny,idy,xin,yin,1,ctx);CHKERRQ(ierr);
  }
#if defined(PETSC_HAVE_MPI_WIN_CREATE_FEATURE)
  else {
    ierr = VecScatterCreateLocal_PtoP_MPI3(nx,idx,ny,idy,xin,yin,1,ctx);CHKERRQ(ierr);
  }
#endif
  ierr = ISRestoreIndices(ix,&idx);CHKERRQ(ierr);
  ierr = ISRestoreIndices(iy,&idy);CHKERRQ(ierr);
  ierr = PetscInfo(xin,"General case: MPI to MPI\n");CHKERRQ(ierr);

  ierr = ISDestroy(&tix);CHKERRQ(ierr);
  ierr = ISDestroy(&tiy);CHKERRQ(ierr);
  ierr = VecScatterViewFromOptions(ctx,NULL,"-vecscatter_view");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode VecScatterGetInputVecType_private(VecScatter ctx,PetscInt *xin_type1,PetscInt *yin_type1)
{
  PetscErrorCode    ierr;
  MPI_Comm          comm,ycomm;
  PetscMPIInt       size;
  Vec               xin = ctx->from_v,yin = ctx->to_v;
  PetscInt          xin_type,yin_type;

  PetscFunctionBegin;
  /*
      Determine if the vectors are "parallel", ie. it shares a comm with other processors, or
      sequential (it does not share a comm). The difference is that parallel vectors treat the
      index set as providing indices in the global parallel numbering of the vector, with
      sequential vectors treat the index set as providing indices in the local sequential
      numbering
  */
  ierr = PetscObjectGetComm((PetscObject)xin,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  if (size > 1) {
    xin_type = VEC_MPI_ID;
  } else xin_type = VEC_SEQ_ID;
  *xin_type1 = xin_type;

  ierr = PetscObjectGetComm((PetscObject)yin,&ycomm);CHKERRQ(ierr);
  ierr = MPI_Comm_size(ycomm,&size);CHKERRQ(ierr);
  if (size > 1) {
    yin_type = VEC_MPI_ID;
  } else yin_type = VEC_SEQ_ID;
  *yin_type1 = yin_type;
  PetscFunctionReturn(0);
}

static PetscErrorCode GetInputISType_private(VecScatter ctx,PetscInt xin_type,PetscInt yin_type,PetscInt *ix_type1,IS *tix1,PetscInt *iy_type1,IS *tiy1)
{
  PetscErrorCode    ierr;
  MPI_Comm          comm;
  Vec               xin = ctx->from_v,yin = ctx->to_v;
  IS                tix = 0,tiy = 0,ix = ctx->from_is, iy = ctx->to_is;
  PetscInt          ix_type  = IS_GENERAL_ID,iy_type = IS_GENERAL_ID;
  PetscBool         flag;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)ctx,&comm);CHKERRQ(ierr);

  /* if ix or iy is not included; assume just grabbing entire vector */
  if (!ix && xin_type == VEC_SEQ_ID) {
    ierr = ISCreateStride(comm,ctx->from_n,0,1,&ix);CHKERRQ(ierr);
    tix  = ix;
  } else if (!ix && xin_type == VEC_MPI_ID) {
    if (yin_type == VEC_MPI_ID) {
      PetscInt ntmp, low;
      ierr = VecGetLocalSize(xin,&ntmp);CHKERRQ(ierr);
      ierr = VecGetOwnershipRange(xin,&low,NULL);CHKERRQ(ierr);
      ierr = ISCreateStride(comm,ntmp,low,1,&ix);CHKERRQ(ierr);
    } else {
      PetscInt Ntmp;
      ierr = VecGetSize(xin,&Ntmp);CHKERRQ(ierr);
      ierr = ISCreateStride(comm,Ntmp,0,1,&ix);CHKERRQ(ierr);
    }
    tix = ix;
  } else if (!ix) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"ix not given, but not Seq or MPI vector");

  if (!iy && yin_type == VEC_SEQ_ID) {
    ierr = ISCreateStride(comm,ctx->to_n,0,1,&iy);CHKERRQ(ierr);
    tiy  = iy;
  } else if (!iy && yin_type == VEC_MPI_ID) {
    if (xin_type == VEC_MPI_ID) {
      PetscInt ntmp, low;
      ierr = VecGetLocalSize(yin,&ntmp);CHKERRQ(ierr);
      ierr = VecGetOwnershipRange(yin,&low,NULL);CHKERRQ(ierr);
      ierr = ISCreateStride(comm,ntmp,low,1,&iy);CHKERRQ(ierr);
    } else {
      PetscInt Ntmp;
      ierr = VecGetSize(yin,&Ntmp);CHKERRQ(ierr);
      ierr = ISCreateStride(comm,Ntmp,0,1,&iy);CHKERRQ(ierr);
    }
    tiy = iy;
  } else if (!iy) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"iy not given, but not Seq or MPI vector");

  /* Determine types of index sets */
  ierr = PetscObjectTypeCompare((PetscObject)ix,ISBLOCK,&flag);CHKERRQ(ierr);
  if (flag) ix_type = IS_BLOCK_ID;
  ierr = PetscObjectTypeCompare((PetscObject)iy,ISBLOCK,&flag);CHKERRQ(ierr);
  if (flag) iy_type = IS_BLOCK_ID;
  ierr = PetscObjectTypeCompare((PetscObject)ix,ISSTRIDE,&flag);CHKERRQ(ierr);
  if (flag) ix_type = IS_STRIDE_ID;
  ierr = PetscObjectTypeCompare((PetscObject)iy,ISSTRIDE,&flag);CHKERRQ(ierr);
  if (flag) iy_type = IS_STRIDE_ID;

  if (ix_type1) *ix_type1 = ix_type;
  if (iy_type1) *iy_type1 = iy_type;
  if (tix1)     *tix1     = tix;
  if (tiy1)     *tiy1     = tiy;
  PetscFunctionReturn(0);
}

static PetscErrorCode VecScatterCreate_vectype_private(VecScatter ctx)
{
  PetscErrorCode    ierr;
  PetscInt          xin_type=-1,yin_type=-1;

  PetscFunctionBegin;
  ierr = VecScatterGetInputVecType_private(ctx,&xin_type,&yin_type);CHKERRQ(ierr);
  if (xin_type == VEC_MPI_ID && yin_type == VEC_SEQ_ID) {
    ierr = VecScatterCreate_PtoS(ctx);CHKERRQ(ierr);
  } else if (xin_type == VEC_SEQ_ID && yin_type == VEC_MPI_ID) {
    ierr = VecScatterCreate_StoP(ctx);CHKERRQ(ierr);
  } else if (xin_type == VEC_MPI_ID && yin_type == VEC_MPI_ID) {
    ierr = VecScatterCreate_PtoP(ctx);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode VecScatterCreate_MPI1(VecScatter ctx)
{
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  /* subroutines called in VecScatterCreate_vectype_private() need scatter_type as an input */
  ierr = PetscObjectChangeTypeName((PetscObject)ctx,VECSCATTERMPI1);CHKERRQ(ierr);
  ierr = PetscInfo(ctx,"Using MPI1 for vector scatter\n");CHKERRQ(ierr);
  ierr = VecScatterCreate_vectype_private(ctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode VecScatterCreate_MPI3(VecScatter ctx)
{
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  /* subroutines called in VecScatterCreate_vectype_private() need scatter_type as an input */
  ierr = PetscObjectChangeTypeName((PetscObject)ctx,VECSCATTERMPI3);CHKERRQ(ierr);
  ierr = PetscInfo(ctx,"Using MPI3 for vector scatter\n");CHKERRQ(ierr);
  ierr = VecScatterCreate_vectype_private(ctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode VecScatterCreate_MPI3Node(VecScatter ctx)
{
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  /* subroutines called in VecScatterCreate_vectype_private() need scatter_type as an input */
  ierr = PetscObjectChangeTypeName((PetscObject)ctx,VECSCATTERMPI3NODE);CHKERRQ(ierr);
  ierr = PetscInfo(ctx,"Using MPI3NODE for vector scatter\n");CHKERRQ(ierr);
  ierr = VecScatterCreate_vectype_private(ctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

