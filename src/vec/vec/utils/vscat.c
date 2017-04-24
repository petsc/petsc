
/*
     Code for creating scatters between vectors. This file
  includes the code for scattering between sequential vectors and
  some special cases for parallel scatters.
*/

#include <petsc/private/isimpl.h>
#include <petsc/private/vecimpl.h>    /*I   "petscvec.h"    I*/

#if defined(PETSC_HAVE_CUSP)
PETSC_INTERN PetscErrorCode VecScatterCUSPIndicesCreate_PtoP(PetscInt,PetscInt*,PetscInt,PetscInt*,PetscCUSPIndices*);
PETSC_INTERN PetscErrorCode VecScatterCUSPIndicesCreate_StoS(PetscInt,PetscInt,PetscInt,PetscInt,PetscInt,PetscInt*,PetscInt*,PetscCUSPIndices*);
PETSC_INTERN PetscErrorCode VecScatterCUSPIndicesDestroy(PetscCUSPIndices*);
PETSC_INTERN PetscErrorCode VecScatterCUSP_StoS(Vec,Vec,PetscCUSPIndices,InsertMode,ScatterMode);
#elif defined(PETSC_HAVE_VECCUDA)
PETSC_INTERN PetscErrorCode VecScatterCUDAIndicesCreate_PtoP(PetscInt,PetscInt*,PetscInt,PetscInt*,PetscCUDAIndices*);
PETSC_INTERN PetscErrorCode VecScatterCUDAIndicesCreate_StoS(PetscInt,PetscInt,PetscInt,PetscInt,PetscInt,PetscInt*,PetscInt*,PetscCUDAIndices*);
PETSC_INTERN PetscErrorCode VecScatterCUDAIndicesDestroy(PetscCUDAIndices*);
PETSC_INTERN PetscErrorCode VecScatterCUDA_StoS(Vec,Vec,PetscCUDAIndices,InsertMode,ScatterMode);
#endif

/* Logging support */
PetscClassId VEC_SCATTER_CLASSID;

#if defined(PETSC_USE_DEBUG)
/*
     Checks if any indices are less than zero and generates an error
*/
static PetscErrorCode VecScatterCheckIndices_Private(PetscInt nmax,PetscInt n,const PetscInt *idx)
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
        ierr        = PetscMalloc1(xx_n,&yvt);CHKERRQ(ierr);
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
  ierr = PetscFree2(ctx->todata,ctx->fromdata);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode VecScatterDestroy_SGToSS(VecScatter ctx)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFree(((VecScatter_Seq_General*)ctx->fromdata)->vslots);CHKERRQ(ierr);
  ierr = PetscFree2(ctx->todata,ctx->fromdata);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode VecScatterDestroy_SSToSG(VecScatter ctx)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFree(((VecScatter_Seq_General*)ctx->todata)->vslots);CHKERRQ(ierr);
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
  sto->type   = in_to->type;
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

  PetscFunctionBegin;
#if defined(PETSC_HAVE_CUSP)
  if (x->valid_GPU_array == PETSC_CUSP_GPU) {
    /* create the scatter indices if not done already */
    if (!ctx->spptr) {
      PetscInt tofirst = 0,tostep = 0,fromfirst = 0,fromstep = 0;
      fslots = gen_from->vslots;
      tslots = gen_to->vslots;
      ierr = VecScatterCUSPIndicesCreate_StoS(n,tofirst,fromfirst,tostep,fromstep,tslots,fslots,(PetscCUSPIndices*)&(ctx->spptr));CHKERRQ(ierr);
    }
    /* next do the scatter */
    ierr = VecScatterCUSP_StoS(x,y,(PetscCUSPIndices)ctx->spptr,addv,mode);
    PetscFunctionReturn(0);
  }
#elif defined(PETSC_HAVE_VECCUDA)
  if (x->valid_GPU_array == PETSC_CUDA_GPU) {
    /* create the scatter indices if not done already */
    if (!ctx->spptr) {
      PetscInt tofirst = 0,tostep = 0,fromfirst = 0,fromstep = 0;
      fslots = gen_from->vslots;
      tslots = gen_to->vslots;
      ierr = VecScatterCUDAIndicesCreate_StoS(n,tofirst,fromfirst,tostep,fromstep,tslots,fslots,(PetscCUDAIndices*)&(ctx->spptr));CHKERRQ(ierr);
    }
    /* next do the scatter */
    ierr = VecScatterCUDA_StoS(x,y,(PetscCUDAIndices)ctx->spptr,addv,mode);
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

  if (addv == INSERT_VALUES) {
    for (i=0; i<n; i++) yv[tslots[i]]  = xv[fslots[i]];
  } else if (addv == ADD_VALUES) {
    for (i=0; i<n; i++) yv[tslots[i]] += xv[fslots[i]];
#if !defined(PETSC_USE_COMPLEX)
  } else if (addv == MAX_VALUES) {
    for (i=0; i<n; i++) yv[tslots[i]] = PetscMax(yv[tslots[i]],xv[fslots[i]]);
#endif
  } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_UNKNOWN_TYPE,"Wrong insert option");
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

  PetscFunctionBegin;
#if defined(PETSC_HAVE_CUSP)
  if (x->valid_GPU_array == PETSC_CUSP_GPU) {
    /* create the scatter indices if not done already */
    if (!ctx->spptr) {
      PetscInt tofirst = first,tostep = 1,fromfirst = 0,fromstep = 0;
      PetscInt *tslots = 0;
      ierr = VecScatterCUSPIndicesCreate_StoS(n,tofirst,fromfirst,tostep,fromstep,tslots,fslots,(PetscCUSPIndices*)&(ctx->spptr));CHKERRQ(ierr);
    }
    /* next do the scatter */
    ierr = VecScatterCUSP_StoS(x,y,(PetscCUSPIndices)ctx->spptr,addv,mode);
    PetscFunctionReturn(0);
  }
#elif defined(PETSC_HAVE_VECCUDA)
  if (x->valid_GPU_array == PETSC_CUDA_GPU) {
    /* create the scatter indices if not done already */
    if (!ctx->spptr) {
      PetscInt tofirst = first,tostep = 1,fromfirst = 0,fromstep = 0;
      PetscInt *tslots = 0;
      ierr = VecScatterCUDAIndicesCreate_StoS(n,tofirst,fromfirst,tostep,fromstep,tslots,fslots,(PetscCUDAIndices*)&(ctx->spptr));CHKERRQ(ierr);
    }
    /* next do the scatter */
    ierr = VecScatterCUDA_StoS(x,y,(PetscCUDAIndices)ctx->spptr,addv,mode);
    PetscFunctionReturn(0);
  }
#endif

  ierr = VecGetArrayPair(x,y,&xv,&yv);CHKERRQ(ierr);
  if (mode & SCATTER_REVERSE) {
    xv += first;
    if (addv == INSERT_VALUES) {
      for (i=0; i<n; i++) yv[fslots[i]] = xv[i];
    } else if (addv == ADD_VALUES) {
      for (i=0; i<n; i++) yv[fslots[i]] += xv[i];
#if !defined(PETSC_USE_COMPLEX)
    } else if (addv == MAX_VALUES) {
      for (i=0; i<n; i++) yv[fslots[i]] = PetscMax(yv[fslots[i]],xv[i]);
#endif
    } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_UNKNOWN_TYPE,"Wrong insert option");
  } else {
    yv += first;
    if (addv == INSERT_VALUES) {
      for (i=0; i<n; i++) yv[i] = xv[fslots[i]];
    } else if (addv == ADD_VALUES) {
      for (i=0; i<n; i++) yv[i] += xv[fslots[i]];
#if !defined(PETSC_USE_COMPLEX)
    } else if (addv == MAX_VALUES) {
      for (i=0; i<n; i++) yv[i] = PetscMax(yv[i],xv[fslots[i]]);
#endif
    } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_UNKNOWN_TYPE,"Wrong insert option");
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

  PetscFunctionBegin;
#if defined(PETSC_HAVE_CUSP)
  if (x->valid_GPU_array == PETSC_CUSP_GPU) {
    /* create the scatter indices if not done already */
    if (!ctx->spptr) {
      PetscInt tofirst = first,tostep = step,fromfirst = 0,fromstep = 0;
      PetscInt * tslots = 0;
      ierr = VecScatterCUSPIndicesCreate_StoS(n,tofirst,fromfirst,tostep,fromstep,tslots,fslots,(PetscCUSPIndices*)&(ctx->spptr));CHKERRQ(ierr);
    }
    /* next do the scatter */
    ierr = VecScatterCUSP_StoS(x,y,(PetscCUSPIndices)ctx->spptr,addv,mode);
    PetscFunctionReturn(0);
  }
#elif defined(PETSC_HAVE_VECCUDA)
  if (x->valid_GPU_array == PETSC_CUDA_GPU) {
    /* create the scatter indices if not done already */
    if (!ctx->spptr) {
      PetscInt tofirst = first,tostep = step,fromfirst = 0,fromstep = 0;
      PetscInt * tslots = 0;
      ierr = VecScatterCUDAIndicesCreate_StoS(n,tofirst,fromfirst,tostep,fromstep,tslots,fslots,(PetscCUDAIndices*)&(ctx->spptr));CHKERRQ(ierr);
    }
    /* next do the scatter */
    ierr = VecScatterCUDA_StoS(x,y,(PetscCUDAIndices)ctx->spptr,addv,mode);
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

  PetscFunctionBegin;
#if defined(PETSC_HAVE_CUSP)
  if (x->valid_GPU_array == PETSC_CUSP_GPU) {
    /* create the scatter indices if not done already */
    if (!ctx->spptr) {
      PetscInt tofirst = 0,tostep = 0,fromfirst = first,fromstep = 1;
      PetscInt *fslots = 0;
      ierr = VecScatterCUSPIndicesCreate_StoS(n,tofirst,fromfirst,tostep,fromstep,tslots,fslots,(PetscCUSPIndices*)&(ctx->spptr));CHKERRQ(ierr);
    }
    /* next do the scatter */
    ierr = VecScatterCUSP_StoS(x,y,(PetscCUSPIndices)ctx->spptr,addv,mode);
    PetscFunctionReturn(0);
  }
#elif defined(PETSC_HAVE_VECCUDA)
  if (x->valid_GPU_array == PETSC_CUDA_GPU) {
    /* create the scatter indices if not done already */
    if (!ctx->spptr) {
      PetscInt tofirst = 0,tostep = 0,fromfirst = first,fromstep = 1;
      PetscInt *fslots = 0;
      ierr = VecScatterCUDAIndicesCreate_StoS(n,tofirst,fromfirst,tostep,fromstep,tslots,fslots,(PetscCUDAIndices*)&(ctx->spptr));CHKERRQ(ierr);
    }
    /* next do the scatter */
    ierr = VecScatterCUDA_StoS(x,y,(PetscCUDAIndices)ctx->spptr,addv,mode);
    PetscFunctionReturn(0);
  }
#endif

  ierr = VecGetArrayPair(x,y,&xv,&yv);CHKERRQ(ierr);
  if (mode & SCATTER_REVERSE) {
    yv += first;
    if (addv == INSERT_VALUES) {
      for (i=0; i<n; i++) yv[i] = xv[tslots[i]];
    } else  if (addv == ADD_VALUES) {
      for (i=0; i<n; i++) yv[i] += xv[tslots[i]];
#if !defined(PETSC_USE_COMPLEX)
    } else  if (addv == MAX_VALUES) {
      for (i=0; i<n; i++) yv[i] = PetscMax(yv[i],xv[tslots[i]]);
#endif
    } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_UNKNOWN_TYPE,"Wrong insert option");
  } else {
    xv += first;
    if (addv == INSERT_VALUES) {
      for (i=0; i<n; i++) yv[tslots[i]] = xv[i];
    } else  if (addv == ADD_VALUES) {
      for (i=0; i<n; i++) yv[tslots[i]] += xv[i];
#if !defined(PETSC_USE_COMPLEX)
    } else  if (addv == MAX_VALUES) {
      for (i=0; i<n; i++) yv[tslots[i]] = PetscMax(yv[tslots[i]],xv[i]);
#endif
    } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_UNKNOWN_TYPE,"Wrong insert option");
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

  PetscFunctionBegin;
#if defined(PETSC_HAVE_CUSP)
  if (x->valid_GPU_array == PETSC_CUSP_GPU) {
    /* create the scatter indices if not done already */
    if (!ctx->spptr) {
      PetscInt tofirst = 0,tostep = 0,fromfirst = first,fromstep = step;
      PetscInt *fslots = 0;
      ierr = VecScatterCUSPIndicesCreate_StoS(n,tofirst,fromfirst,tostep,fromstep,tslots,fslots,(PetscCUSPIndices*)&(ctx->spptr));CHKERRQ(ierr);
    }
    /* next do the scatter */
    ierr = VecScatterCUSP_StoS(x,y,(PetscCUSPIndices)ctx->spptr,addv,mode);
    PetscFunctionReturn(0);
  }
#elif defined(PETSC_HAVE_VECCUDA)
  if (x->valid_GPU_array == PETSC_CUDA_GPU) {
    /* create the scatter indices if not done already */
    if (!ctx->spptr) {
      PetscInt tofirst = 0,tostep = 0,fromfirst = first,fromstep = step;
      PetscInt *fslots = 0;
      ierr = VecScatterCUDAIndicesCreate_StoS(n,tofirst,fromfirst,tostep,fromstep,tslots,fslots,(PetscCUDAIndices*)&(ctx->spptr));CHKERRQ(ierr);
    }
    /* next do the scatter */
    ierr = VecScatterCUDA_StoS(x,y,(PetscCUDAIndices)ctx->spptr,addv,mode);
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

  PetscFunctionBegin;
#if defined(PETSC_HAVE_CUSP)
  if (x->valid_GPU_array == PETSC_CUSP_GPU) {
    /* create the scatter indices if not done already */
    if (!ctx->spptr) {
      PetscInt *tslots = 0,*fslots = 0;
      ierr = VecScatterCUSPIndicesCreate_StoS(n,to_first,from_first,to_step,from_step,tslots,fslots,(PetscCUSPIndices*)&(ctx->spptr));CHKERRQ(ierr);
    }
    /* next do the scatter */
    ierr = VecScatterCUSP_StoS(x,y,(PetscCUSPIndices)ctx->spptr,addv,mode);
    PetscFunctionReturn(0);
  }
#elif defined(PETSC_HAVE_VECCUDA)
  if (x->valid_GPU_array == PETSC_CUDA_GPU) {
    /* create the scatter indices if not done already */
    if (!ctx->spptr) {
      PetscInt *tslots = 0,*fslots = 0;
      ierr = VecScatterCUDAIndicesCreate_StoS(n,to_first,from_first,to_step,from_step,tslots,fslots,(PetscCUDAIndices*)&(ctx->spptr));CHKERRQ(ierr);
    }
    /* next do the scatter */
    ierr = VecScatterCUDA_StoS(x,y,(PetscCUDAIndices)ctx->spptr,addv,mode);
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
      yv += to_first; xv += from_first;
      for (i=0; i<n; i++) yv[i] += xv[i];
    } else {
      for (i=0; i<n; i++) yv[to_first + i*to_step] += xv[from_first+i*from_step];
    }
#if !defined(PETSC_USE_COMPLEX)
  } else if (addv == MAX_VALUES) {
    if (to_step == 1 && from_step == 1) {
      yv += to_first; xv += from_first;
      for (i=0; i<n; i++) yv[i] = PetscMax(yv[i],xv[i]);
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
  out_to->type                 = in_to->type;
  out_to->nonmatching_computed = PETSC_FALSE;
  out_to->slots_nonmatching    = 0;
  out_to->is_copy              = PETSC_FALSE;
  ierr = PetscMemcpy(out_to->vslots,in_to->vslots,(out_to->n)*sizeof(PetscInt));CHKERRQ(ierr);

  out_from->n                    = in_from->n;
  out_from->type                 = in_from->type;
  out_from->nonmatching_computed = PETSC_FALSE;
  out_from->slots_nonmatching    = 0;
  out_from->is_copy              = PETSC_FALSE;
  ierr = PetscMemcpy(out_from->vslots,in_from->vslots,(out_from->n)*sizeof(PetscInt));CHKERRQ(ierr);

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

  ierr          = PetscMalloc2(1,&out_to,1,&out_from);CHKERRQ(ierr);
  ierr          = PetscMalloc1(in_from->n,&out_from->vslots);CHKERRQ(ierr);
  out_to->n     = in_to->n;
  out_to->type  = in_to->type;
  out_to->first = in_to->first;
  out_to->step  = in_to->step;
  out_to->type  = in_to->type;

  out_from->n                    = in_from->n;
  out_from->type                 = in_from->type;
  out_from->nonmatching_computed = PETSC_FALSE;
  out_from->slots_nonmatching    = 0;
  out_from->is_copy              = PETSC_FALSE;
  ierr = PetscMemcpy(out_from->vslots,in_from->vslots,(out_from->n)*sizeof(PetscInt));CHKERRQ(ierr);

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
  out_to->type    = in_to->type;
  out_to->first   = in_to->first;
  out_to->step    = in_to->step;
  out_to->type    = in_to->type;
  out_from->n     = in_from->n;
  out_from->type  = in_from->type;
  out_from->first = in_from->first;
  out_from->step  = in_from->step;
  out_from->type  = in_from->type;
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


extern PetscErrorCode VecScatterCreate_PtoS(PetscInt,const PetscInt*,PetscInt,const PetscInt*,Vec,Vec,PetscInt,VecScatter);
extern PetscErrorCode VecScatterCreate_PtoP(PetscInt,const PetscInt*,PetscInt,const PetscInt*,Vec,Vec,PetscInt,VecScatter);
extern PetscErrorCode VecScatterCreate_StoP(PetscInt,const PetscInt*,PetscInt,const PetscInt*,Vec,Vec,PetscInt,VecScatter);

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
  ctx->packtogether = PETSC_FALSE;
  ierr = PetscOptionsGetBool(NULL,NULL,"-vecscatter_packtogether",&ctx->packtogether,NULL);CHKERRQ(ierr);
  if (ctx->packtogether) {
    ierr = PetscInfo(ctx,"Pack all messages before sending\n");CHKERRQ(ierr);
  }
  *newctx = ctx;
  PetscFunctionReturn(0);
}

/*@C
   VecScatterCreate - Creates a vector scatter context.

   Collective on Vec

   Input Parameters:
+  xin - a vector that defines the shape (parallel data layout of the vector)
         of vectors from which we scatter
.  yin - a vector that defines the shape (parallel data layout of the vector)
         of vectors to which we scatter
.  ix - the indices of xin to scatter (if NULL scatters all values)
-  iy - the indices of yin to hold results (if NULL fills entire vector yin)

   Output Parameter:
.  newctx - location to store the new scatter context

   Options Database Keys: (uses regular MPI_Sends by default)
+  -vecscatter_view         - Prints detail of communications
.  -vecscatter_view ::ascii_info    - Print less details about communication
.  -vecscatter_ssend        - Uses MPI_Ssend_init() instead of MPI_Send_init()
.  -vecscatter_rsend           - use ready receiver mode for MPI sends
.  -vecscatter_merge        - VecScatterBegin() handles all of the communication, VecScatterEnd() is a nop
                              eliminates the chance for overlap of computation and communication
.  -vecscatter_sendfirst    - Posts sends before receives
.  -vecscatter_packtogether - Pack all messages before sending, receive all messages before unpacking
.  -vecscatter_alltoall     - Uses MPI all to all communication for scatter
.  -vecscatter_window       - Use MPI 2 window operations to move data
.  -vecscatter_nopack       - Avoid packing to work vector when possible (if used with -vecscatter_alltoall then will use MPI_Alltoallw()
-  -vecscatter_reproduce    - insure that the order of the communications are done the same for each scatter, this under certain circumstances
                              will make the results of scatters deterministic when otherwise they are not (it may be slower also).

$
$                                                                                    --When packing is used--
$                               MPI Datatypes (no packing)  sendfirst   merge        packtogether  persistent*
$                                _nopack                   _sendfirst    _merge      _packtogether                -vecscatter_
$ ----------------------------------------------------------------------------------------------------------------------------
$    Message passing    Send       p                           X            X           X         always
$                      Ssend       p                           X            X           X         always          _ssend
$                      Rsend       p                        nonsense        X           X         always          _rsend
$    AlltoAll  v or w              X                        nonsense     always         X         nonsense        _alltoall
$    MPI_Win                       p                        nonsense        p           p         nonsense        _window
$
$   Since persistent sends and receives require a constant memory address they can only be used when data is packed into the work vector
$   because the in and out array may be different for each call to VecScatterBegin/End().
$
$    p indicates possible, but not implemented. X indicates implemented
$

    Level: intermediate

  Notes:
   In calls to VecScatter() you can use different vectors than the xin and
   yin you used above; BUT they must have the same parallel data layout, for example,
   they could be obtained from VecDuplicate().
   A VecScatter context CANNOT be used in two or more simultaneous scatters;
   that is you cannot call a second VecScatterBegin() with the same scatter
   context until the VecScatterEnd() has been called on the first VecScatterBegin().
   In this case a separate VecScatter is needed for each concurrent scatter.

   Currently the MPI_Send(), MPI_Ssend() and MPI_Rsend() all use PERSISTENT versions.
   (this unfortunately requires that the same in and out arrays be used for each use, this
    is why when not using MPI_alltoallw() we always need to pack the input into the work array before sending
    and unpack upon receeving instead of using MPI datatypes to avoid the packing/unpacking).

   Both ix and iy cannot be NULL at the same time.

   Concepts: scatter^between vectors
   Concepts: gather^between vectors

.seealso: VecScatterDestroy(), VecScatterCreateToAll(), VecScatterCreateToZero()
@*/
PetscErrorCode  VecScatterCreate(Vec xin,IS ix,Vec yin,IS iy,VecScatter *newctx)
{
  VecScatter        ctx;
  PetscErrorCode    ierr;
  PetscMPIInt       size;
  PetscInt          xin_type = VEC_SEQ_ID,yin_type = VEC_SEQ_ID,*range;
  PetscInt          ix_type  = IS_GENERAL_ID,iy_type = IS_GENERAL_ID;
  MPI_Comm          comm,ycomm;
  PetscBool         totalv,ixblock,iyblock,iystride,islocal,cando,flag;
  IS                tix = 0,tiy = 0;

  PetscFunctionBegin;
  if (!ix && !iy) SETERRQ(PetscObjectComm((PetscObject)xin),PETSC_ERR_SUP,"Cannot pass default in for both input and output indices");

  /*
      Determine if the vectors are "parallel", ie. it shares a comm with other processors, or
      sequential (it does not share a comm). The difference is that parallel vectors treat the
      index set as providing indices in the global parallel numbering of the vector, with
      sequential vectors treat the index set as providing indices in the local sequential
      numbering
  */
  ierr = PetscObjectGetComm((PetscObject)xin,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  if (size > 1) xin_type = VEC_MPI_ID;

  ierr = PetscObjectGetComm((PetscObject)yin,&ycomm);CHKERRQ(ierr);
  ierr = MPI_Comm_size(ycomm,&size);CHKERRQ(ierr);
  if (size > 1) {comm = ycomm; yin_type = VEC_MPI_ID;}

  /* generate the Scatter context */
  ierr = PetscHeaderCreate(ctx,VEC_SCATTER_CLASSID,"VecScatter","VecScatter","Vec",comm,VecScatterDestroy,VecScatterView);CHKERRQ(ierr);
  ctx->inuse               = PETSC_FALSE;

  ctx->beginandendtogether = PETSC_FALSE;
  ierr = PetscOptionsGetBool(NULL,NULL,"-vecscatter_merge",&ctx->beginandendtogether,NULL);CHKERRQ(ierr);
  if (ctx->beginandendtogether) {
    ierr = PetscInfo(ctx,"Using combined (merged) vector scatter begin and end\n");CHKERRQ(ierr);
  }
  ctx->packtogether = PETSC_FALSE;
  ierr = PetscOptionsGetBool(NULL,NULL,"-vecscatter_packtogether",&ctx->packtogether,NULL);CHKERRQ(ierr);
  if (ctx->packtogether) {
    ierr = PetscInfo(ctx,"Pack all messages before sending\n");CHKERRQ(ierr);
  }

  ierr = VecGetLocalSize(xin,&ctx->from_n);CHKERRQ(ierr);
  ierr = VecGetLocalSize(yin,&ctx->to_n);CHKERRQ(ierr);

  /*
      if ix or iy is not included; assume just grabbing entire vector
  */
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

  /*
     Determine types of index sets
  */
  ierr = PetscObjectTypeCompare((PetscObject)ix,ISBLOCK,&flag);CHKERRQ(ierr);
  if (flag) ix_type = IS_BLOCK_ID;
  ierr = PetscObjectTypeCompare((PetscObject)iy,ISBLOCK,&flag);CHKERRQ(ierr);
  if (flag) iy_type = IS_BLOCK_ID;
  ierr = PetscObjectTypeCompare((PetscObject)ix,ISSTRIDE,&flag);CHKERRQ(ierr);
  if (flag) ix_type = IS_STRIDE_ID;
  ierr = PetscObjectTypeCompare((PetscObject)iy,ISSTRIDE,&flag);CHKERRQ(ierr);
  if (flag) iy_type = IS_STRIDE_ID;

  /* ===========================================================================================================
        Check for special cases
     ==========================================================================================================*/
  /* ---------------------------------------------------------------------------*/
  if (xin_type == VEC_SEQ_ID && yin_type == VEC_SEQ_ID) {
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
      to->type          = VEC_SCATTER_SEQ_GENERAL;
      from->type        = VEC_SCATTER_SEQ_GENERAL;
      ctx->todata       = (void*)to;
      ctx->fromdata     = (void*)from;
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
      to8->type         = VEC_SCATTER_SEQ_STRIDE;
      from8->type       = VEC_SCATTER_SEQ_STRIDE;
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
      if (step == 1)  ctx->ops->begin = VecScatterBegin_SGToSS_Stride1;
      else            ctx->ops->begin = VecScatterBegin_SGToSS;
      ctx->ops->destroy   = VecScatterDestroy_SGToSS;
      ctx->ops->end       = 0;
      ctx->ops->copy      = VecScatterCopy_SGToSS;
      ctx->ops->view      = VecScatterView_SGToSS;
      to9->type      = VEC_SCATTER_SEQ_STRIDE;
      from9->type    = VEC_SCATTER_SEQ_GENERAL;
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
      if (step == 1) ctx->ops->begin = VecScatterBegin_SSToSG_Stride1;
      else           ctx->ops->begin = VecScatterBegin_SSToSG;
      ctx->ops->destroy = VecScatterDestroy_SSToSG;
      ctx->ops->end     = 0;
      ctx->ops->copy    = 0;
      ctx->ops->view    = VecScatterView_SSToSG;
      to10->type   = VEC_SCATTER_SEQ_GENERAL;
      from10->type = VEC_SCATTER_SEQ_STRIDE;
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
        to13->type    = VEC_SCATTER_SEQ_STRIDE;
        from13->type  = VEC_SCATTER_SEQ_STRIDE;
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
      to11->type        = VEC_SCATTER_SEQ_GENERAL;
      from11->type      = VEC_SCATTER_SEQ_GENERAL;
      ctx->todata       = (void*)to11;
      ctx->fromdata     = (void*)from11;
      ctx->ops->begin   = VecScatterBegin_SGToSG;
      ctx->ops->end     = 0;
      ctx->ops->destroy = VecScatterDestroy_SGToSG;
      ctx->ops->copy    = VecScatterCopy_SGToSG;
      ctx->ops->view    = VecScatterView_SGToSG;
      ierr = ISRestoreIndices(ix,&idx);CHKERRQ(ierr);
      ierr = ISRestoreIndices(iy,&idy);CHKERRQ(ierr);
      ierr = PetscInfo(xin,"Sequential vector scatter with block indices\n");CHKERRQ(ierr);
      goto functionend;
    }
  }
  /* ---------------------------------------------------------------------------*/
  if (xin_type == VEC_MPI_ID && yin_type == VEC_SEQ_ID) {

    /* ===========================================================================================================
          Check for special cases
       ==========================================================================================================*/
    islocal = PETSC_FALSE;
    /* special case extracting (subset of) local portion */
    if (ix_type == IS_STRIDE_ID && iy_type == IS_STRIDE_ID) {
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
        to12->type         = VEC_SCATTER_SEQ_STRIDE;
        from12->type       = VEC_SCATTER_SEQ_STRIDE;
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
        sto->type         = VEC_SCATTER_MPI_TOALL;
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
        sto->type         = VEC_SCATTER_MPI_TOONE;
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
          ierr = VecScatterCreate_PtoS(nx,idx,ny,idy,xin,yin,bsx,ctx);CHKERRQ(ierr);
          ierr = ISBlockRestoreIndices(ix,&idx);CHKERRQ(ierr);
          ierr = ISBlockRestoreIndices(iy,&idy);CHKERRQ(ierr);
          ierr = PetscInfo(xin,"Special case: blocked indices\n");CHKERRQ(ierr);
          goto functionend;
        }
        /* special case block to stride */
      } else if (iystride) {
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
          ierr = VecScatterCreate_PtoS(nx,idx,nx,idy,xin,yin,bsx,ctx);CHKERRQ(ierr);
          ierr = PetscFree(idy);CHKERRQ(ierr);
          ierr = ISBlockRestoreIndices(ix,&idx);CHKERRQ(ierr);
          ierr = PetscInfo(xin,"Special case: blocked indices to stride\n");CHKERRQ(ierr);
          goto functionend;
        }
      }
    }
    /* left over general case */
    {
      PetscInt       nx,ny;
      const PetscInt *idx,*idy;
      ierr = ISGetLocalSize(ix,&nx);CHKERRQ(ierr);
      ierr = ISGetIndices(ix,&idx);CHKERRQ(ierr);
      ierr = ISGetLocalSize(iy,&ny);CHKERRQ(ierr);
      ierr = ISGetIndices(iy,&idy);CHKERRQ(ierr);
      if (nx != ny) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Local scatter sizes don't match (%d %d)",nx,ny);
      ierr = VecScatterCreate_PtoS(nx,idx,ny,idy,xin,yin,1,ctx);CHKERRQ(ierr);
      ierr = ISRestoreIndices(ix,&idx);CHKERRQ(ierr);
      ierr = ISRestoreIndices(iy,&idy);CHKERRQ(ierr);
      ierr = PetscInfo(xin,"General case: MPI to Seq\n");CHKERRQ(ierr);
      goto functionend;
    }
  }
  /* ---------------------------------------------------------------------------*/
  if (xin_type == VEC_SEQ_ID && yin_type == VEC_MPI_ID) {
    /* ===========================================================================================================
          Check for special cases
       ==========================================================================================================*/
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
        to->type          = VEC_SCATTER_SEQ_STRIDE;
        from->type        = VEC_SCATTER_SEQ_STRIDE;
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
        ierr = VecScatterCreate_StoP(nx,idx,nx,idy,xin,yin,bsx,ctx);CHKERRQ(ierr);
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
      ierr = VecScatterCreate_StoP(nx,idx,ny,idy,xin,yin,1,ctx);CHKERRQ(ierr);
      ierr = ISRestoreIndices(ix,&idx);CHKERRQ(ierr);
      ierr = ISRestoreIndices(iy,&idy);CHKERRQ(ierr);
      ierr = PetscInfo(xin,"General case: Seq to MPI\n");CHKERRQ(ierr);
      goto functionend;
    }
  }
  /* ---------------------------------------------------------------------------*/
  if (xin_type == VEC_MPI_ID && yin_type == VEC_MPI_ID) {
    /* no special cases for now */
    PetscInt       nx,ny;
    const PetscInt *idx,*idy;
    ierr = ISGetLocalSize(ix,&nx);CHKERRQ(ierr);
    ierr = ISGetIndices(ix,&idx);CHKERRQ(ierr);
    ierr = ISGetLocalSize(iy,&ny);CHKERRQ(ierr);
    ierr = ISGetIndices(iy,&idy);CHKERRQ(ierr);
    if (nx != ny) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Local scatter sizes don't match");
    ierr = VecScatterCreate_PtoP(nx,idx,ny,idy,xin,yin,1,ctx);CHKERRQ(ierr);
    ierr = ISRestoreIndices(ix,&idx);CHKERRQ(ierr);
    ierr = ISRestoreIndices(iy,&idy);CHKERRQ(ierr);
    ierr = PetscInfo(xin,"General case: MPI to MPI\n");CHKERRQ(ierr);
    goto functionend;
  }

  functionend:
  *newctx = ctx;
  ierr = ISDestroy(&tix);CHKERRQ(ierr);
  ierr = ISDestroy(&tiy);CHKERRQ(ierr);
  ierr = VecScatterViewFromOptions(ctx,NULL,"-vecscatter_view");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------------*/
/*@
   VecScatterGetMerged - Returns true if the scatter is completed in the VecScatterBegin()
      and the VecScatterEnd() does nothing

   Not Collective

   Input Parameter:
.   ctx - scatter context created with VecScatterCreate()

   Output Parameter:
.   flg - PETSC_TRUE if the VecScatterBegin/End() are all done during the VecScatterBegin()

   Level: developer

.seealso: VecScatterCreate(), VecScatterEnd(), VecScatterBegin()
@*/
PetscErrorCode  VecScatterGetMerged(VecScatter ctx,PetscBool  *flg)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ctx,VEC_SCATTER_CLASSID,1);
  *flg = ctx->beginandendtogether;
  PetscFunctionReturn(0);
}

/*@
   VecScatterBegin - Begins a generalized scatter from one vector to
   another. Complete the scattering phase with VecScatterEnd().

   Neighbor-wise Collective on VecScatter and Vec

   Input Parameters:
+  inctx - scatter context generated by VecScatterCreate()
.  x - the vector from which we scatter
.  y - the vector to which we scatter
.  addv - either ADD_VALUES or INSERT_VALUES, with INSERT_VALUES mode any location
          not scattered to retains its old value; i.e. the vector is NOT first zeroed.
-  mode - the scattering mode, usually SCATTER_FORWARD.  The available modes are:
    SCATTER_FORWARD or SCATTER_REVERSE


   Level: intermediate

   Options Database: See VecScatterCreate()

   Notes:
   The vectors x and y need not be the same vectors used in the call
   to VecScatterCreate(), but x must have the same parallel data layout
   as that passed in as the x to VecScatterCreate(), similarly for the y.
   Most likely they have been obtained from VecDuplicate().

   You cannot change the values in the input vector between the calls to VecScatterBegin()
   and VecScatterEnd().

   If you use SCATTER_REVERSE the two arguments x and y should be reversed, from
   the SCATTER_FORWARD.

   y[iy[i]] = x[ix[i]], for i=0,...,ni-1

   This scatter is far more general than the conventional
   scatter, since it can be a gather or a scatter or a combination,
   depending on the indices ix and iy.  If x is a parallel vector and y
   is sequential, VecScatterBegin() can serve to gather values to a
   single processor.  Similarly, if y is parallel and x sequential, the
   routine can scatter from one processor to many processors.

   Concepts: scatter^between vectors
   Concepts: gather^between vectors

.seealso: VecScatterCreate(), VecScatterEnd()
@*/
PetscErrorCode  VecScatterBegin(VecScatter inctx,Vec x,Vec y,InsertMode addv,ScatterMode mode)
{
  PetscErrorCode ierr;
#if defined(PETSC_USE_DEBUG)
  PetscInt       to_n,from_n;
#endif
  PetscFunctionBegin;
  PetscValidHeaderSpecific(inctx,VEC_SCATTER_CLASSID,1);
  PetscValidHeaderSpecific(x,VEC_CLASSID,2);
  PetscValidHeaderSpecific(y,VEC_CLASSID,3);
  if (inctx->inuse) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE," Scatter ctx already in use");

#if defined(PETSC_USE_DEBUG)
  /*
     Error checking to make sure these vectors match the vectors used
   to create the vector scatter context. -1 in the from_n and to_n indicate the
   vector lengths are unknown (for example with mapped scatters) and thus
   no error checking is performed.
  */
  if (inctx->from_n >= 0 && inctx->to_n >= 0) {
    ierr = VecGetLocalSize(x,&from_n);CHKERRQ(ierr);
    ierr = VecGetLocalSize(y,&to_n);CHKERRQ(ierr);
    if (mode & SCATTER_REVERSE) {
      if (to_n != inctx->from_n) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Vector wrong size %D for scatter %D (scatter reverse and vector to != ctx from size)",to_n,inctx->from_n);
      if (from_n != inctx->to_n) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Vector wrong size %D for scatter %D (scatter reverse and vector from != ctx to size)",from_n,inctx->to_n);
    } else {
      if (to_n != inctx->to_n)     SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Vector wrong size %D for scatter %D (scatter forward and vector to != ctx to size)",to_n,inctx->to_n);
      if (from_n != inctx->from_n) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Vector wrong size %D for scatter %D (scatter forward and vector from != ctx from size)",from_n,inctx->from_n);
    }
  }
#endif

  inctx->inuse = PETSC_TRUE;
  ierr = PetscLogEventBarrierBegin(VEC_ScatterBarrier,0,0,0,0,PetscObjectComm((PetscObject)inctx));CHKERRQ(ierr);
  ierr = (*inctx->ops->begin)(inctx,x,y,addv,mode);CHKERRQ(ierr);
  if (inctx->beginandendtogether && inctx->ops->end) {
    inctx->inuse = PETSC_FALSE;
    ierr = (*inctx->ops->end)(inctx,x,y,addv,mode);CHKERRQ(ierr);
  }
  ierr = PetscLogEventBarrierEnd(VEC_ScatterBarrier,0,0,0,0,PetscObjectComm((PetscObject)inctx));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* --------------------------------------------------------------------*/
/*@
   VecScatterEnd - Ends a generalized scatter from one vector to another.  Call
   after first calling VecScatterBegin().

   Neighbor-wise Collective on VecScatter and Vec

   Input Parameters:
+  ctx - scatter context generated by VecScatterCreate()
.  x - the vector from which we scatter
.  y - the vector to which we scatter
.  addv - either ADD_VALUES or INSERT_VALUES.
-  mode - the scattering mode, usually SCATTER_FORWARD.  The available modes are:
     SCATTER_FORWARD, SCATTER_REVERSE

   Level: intermediate

   Notes:
   If you use SCATTER_REVERSE the arguments x and y should be reversed, from the SCATTER_FORWARD.

   y[iy[i]] = x[ix[i]], for i=0,...,ni-1

.seealso: VecScatterBegin(), VecScatterCreate()
@*/
PetscErrorCode  VecScatterEnd(VecScatter ctx,Vec x,Vec y,InsertMode addv,ScatterMode mode)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ctx,VEC_SCATTER_CLASSID,1);
  PetscValidHeaderSpecific(x,VEC_CLASSID,2);
  PetscValidHeaderSpecific(y,VEC_CLASSID,3);
  ctx->inuse = PETSC_FALSE;
  if (!ctx->ops->end) PetscFunctionReturn(0);
  if (!ctx->beginandendtogether) {
    ierr = PetscLogEventBegin(VEC_ScatterEnd,ctx,x,y,0);CHKERRQ(ierr);
    ierr = (*(ctx)->ops->end)(ctx,x,y,addv,mode);CHKERRQ(ierr);
    ierr = PetscLogEventEnd(VEC_ScatterEnd,ctx,x,y,0);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*@C
   VecScatterDestroy - Destroys a scatter context created by
   VecScatterCreate().

   Collective on VecScatter

   Input Parameter:
.  ctx - the scatter context

   Level: intermediate

.seealso: VecScatterCreate(), VecScatterCopy()
@*/
PetscErrorCode  VecScatterDestroy(VecScatter *ctx)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!*ctx) PetscFunctionReturn(0);
  PetscValidHeaderSpecific(*ctx,VEC_SCATTER_CLASSID,1);
  if ((*ctx)->inuse) SETERRQ(((PetscObject)(*ctx))->comm,PETSC_ERR_ARG_WRONGSTATE,"Scatter context is in use");
  if (--((PetscObject)(*ctx))->refct > 0) {*ctx = 0; PetscFunctionReturn(0);}

  /* if memory was published with SAWs then destroy it */
  ierr = PetscObjectSAWsViewOff((PetscObject)(*ctx));CHKERRQ(ierr);
  ierr = (*(*ctx)->ops->destroy)(*ctx);CHKERRQ(ierr);
#if defined(PETSC_HAVE_CUSP)
  ierr = VecScatterCUSPIndicesDestroy((PetscCUSPIndices*)&((*ctx)->spptr));CHKERRQ(ierr);
#elif defined(PETSC_HAVE_VECCUDA)
  ierr = VecScatterCUDAIndicesDestroy((PetscCUDAIndices*)&((*ctx)->spptr));CHKERRQ(ierr);
#endif
  ierr = PetscHeaderDestroy(ctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
   VecScatterCopy - Makes a copy of a scatter context.

   Collective on VecScatter

   Input Parameter:
.  sctx - the scatter context

   Output Parameter:
.  ctx - the context copy

   Level: advanced

.seealso: VecScatterCreate(), VecScatterDestroy()
@*/
PetscErrorCode  VecScatterCopy(VecScatter sctx,VecScatter *ctx)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sctx,VEC_SCATTER_CLASSID,1);
  PetscValidPointer(ctx,2);
  if (!sctx->ops->copy) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Cannot copy this type");
  ierr = PetscHeaderCreate(*ctx,VEC_SCATTER_CLASSID,"VecScatter","VecScatter","Vec",PetscObjectComm((PetscObject)sctx),VecScatterDestroy,VecScatterView);CHKERRQ(ierr);
  (*ctx)->to_n   = sctx->to_n;
  (*ctx)->from_n = sctx->from_n;
  ierr = (*sctx->ops->copy)(sctx,*ctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


/* ------------------------------------------------------------------*/
/*@C
   VecScatterView - Views a vector scatter context.

   Collective on VecScatter

   Input Parameters:
+  ctx - the scatter context
-  viewer - the viewer for displaying the context

   Level: intermediate

C@*/
PetscErrorCode  VecScatterView(VecScatter ctx,PetscViewer viewer)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ctx,VEC_SCATTER_CLASSID,1);
  if (!viewer) {
    ierr = PetscViewerASCIIGetStdout(PetscObjectComm((PetscObject)ctx),&viewer);CHKERRQ(ierr);
  }
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,2);
  if (ctx->ops->view) {
    ierr = (*ctx->ops->view)(ctx,viewer);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*@C
   VecScatterRemap - Remaps the "from" and "to" indices in a
   vector scatter context. FOR EXPERTS ONLY!

   Collective on VecScatter

   Input Parameters:
+  scat - vector scatter context
.  from - remapping for "from" indices (may be NULL)
-  to   - remapping for "to" indices (may be NULL)

   Level: developer

   Notes: In the parallel case the todata is actually the indices
          from which the data is TAKEN! The from stuff is where the
          data is finally put. This is VERY VERY confusing!

          In the sequential case the todata is the indices where the
          data is put and the fromdata is where it is taken from.
          This is backwards from the paralllel case! CRY! CRY! CRY!

@*/
PetscErrorCode  VecScatterRemap(VecScatter scat,PetscInt *rto,PetscInt *rfrom)
{
  VecScatter_Seq_General *to,*from;
  VecScatter_MPI_General *mto;
  PetscInt               i;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(scat,VEC_SCATTER_CLASSID,1);
  if (rto)   PetscValidIntPointer(rto,2);
  if (rfrom) PetscValidIntPointer(rfrom,3);

  from = (VecScatter_Seq_General*)scat->fromdata;
  mto  = (VecScatter_MPI_General*)scat->todata;

  if (mto->type == VEC_SCATTER_MPI_TOALL) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Not for to all scatter");

  if (rto) {
    if (mto->type == VEC_SCATTER_MPI_GENERAL) {
      /* handle off processor parts */
      for (i=0; i<mto->starts[mto->n]; i++) mto->indices[i] = rto[mto->indices[i]];

      /* handle local part */
      to = &mto->local;
      for (i=0; i<to->n; i++) to->vslots[i] = rto[to->vslots[i]];
    } else if (from->type == VEC_SCATTER_SEQ_GENERAL) {
      for (i=0; i<from->n; i++) from->vslots[i] = rto[from->vslots[i]];
    } else if (from->type == VEC_SCATTER_SEQ_STRIDE) {
      VecScatter_Seq_Stride *sto = (VecScatter_Seq_Stride*)from;

      /* if the remapping is the identity and stride is identity then skip remap */
      if (sto->step == 1 && sto->first == 0) {
        for (i=0; i<sto->n; i++) {
          if (rto[i] != i) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Unable to remap such scatters");
        }
      } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Unable to remap such scatters");
    } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Unable to remap such scatters");
  }

  if (rfrom) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Unable to remap the FROM in scatters yet");

  /*
     Mark then vector lengths as unknown because we do not know the
   lengths of the remapped vectors
  */
  scat->from_n = -1;
  scat->to_n   = -1;
  PetscFunctionReturn(0);
}

/*
 VecScatterGetTypes_Private - Returns the scatter types.

 scatter - The scatter.
 from    - Upon exit this contains the type of the from scatter.
 to      - Upon exit this contains the type of the to scatter.
*/
PetscErrorCode VecScatterGetTypes_Private(VecScatter scatter,VecScatterType *from,VecScatterType *to)
{
  VecScatter_Common* fromdata = (VecScatter_Common*)scatter->fromdata;
  VecScatter_Common* todata   = (VecScatter_Common*)scatter->todata;

  PetscFunctionBegin;
  *from = fromdata->type;
  *to = todata->type;
  PetscFunctionReturn(0);
}


/*
  VecScatterIsSequential_Private - Returns true if the scatter is sequential.

  scatter - The scatter.
  flag    - Upon exit flag is true if the scatter is of type VecScatter_Seq_General 
            or VecScatter_Seq_Stride; otherwise flag is false.
*/
PetscErrorCode VecScatterIsSequential_Private(VecScatter_Common *scatter,PetscBool *flag)
{
  VecScatterType scatterType = scatter->type;

  PetscFunctionBegin;
  if (scatterType == VEC_SCATTER_SEQ_GENERAL || scatterType == VEC_SCATTER_SEQ_STRIDE) {
    *flag = PETSC_TRUE;
  } else {
    *flag = PETSC_FALSE;
  }
  PetscFunctionReturn(0);
}

#if defined(PETSC_HAVE_CUSP) || defined(PETSC_HAVE_VECCUDA)

/*@C
   VecScatterInitializeForGPU - Initializes a generalized scatter from one vector
   to another for GPU based computation.

   Input Parameters:
+  inctx - scatter context generated by VecScatterCreate()
.  x - the vector from which we scatter
-  mode - the scattering mode, usually SCATTER_FORWARD.  The available modes are:
    SCATTER_FORWARD or SCATTER_REVERSE

  Level: intermediate

  Notes:
   Effectively, this function creates all the necessary indexing buffers and work
   vectors needed to move data only those data points in a vector which need to
   be communicated across ranks. This is done at the first time this function is
   called. Currently, this only used in the context of the parallel SpMV call in
   MatMult_MPIAIJCUSP or MatMult_MPIAIJCUSPARSE.

   This function is executed before the call to MatMult. This enables the memory
   transfers to be overlapped with the MatMult SpMV kernel call.

.seealso: VecScatterFinalizeForGPU(), VecScatterCreate(), VecScatterEnd()
@*/
PETSC_EXTERN PetscErrorCode VecScatterInitializeForGPU(VecScatter inctx,Vec x,ScatterMode mode)
{
  PetscFunctionBegin;
  VecScatter_MPI_General *to,*from;
  PetscErrorCode         ierr;
  PetscInt               i,*indices,*sstartsSends,*sstartsRecvs,nrecvs,nsends,bs;
  PetscBool              isSeq1,isSeq2;

  PetscFunctionBegin;
  ierr = VecScatterIsSequential_Private((VecScatter_Common*)inctx->fromdata,&isSeq1);CHKERRQ(ierr);
  ierr = VecScatterIsSequential_Private((VecScatter_Common*)inctx->todata,&isSeq2);CHKERRQ(ierr);
  if (isSeq1 || isSeq2) {
    PetscFunctionReturn(0);
  }
  if (mode & SCATTER_REVERSE) {
    to     = (VecScatter_MPI_General*)inctx->fromdata;
    from   = (VecScatter_MPI_General*)inctx->todata;
  } else {
    to     = (VecScatter_MPI_General*)inctx->todata;
    from   = (VecScatter_MPI_General*)inctx->fromdata;
  }
  bs           = to->bs;
  nrecvs       = from->n;
  nsends       = to->n;
  indices      = to->indices;
  sstartsSends = to->starts;
  sstartsRecvs = from->starts;
#if defined(PETSC_HAVE_CUSP)
  if (x->valid_GPU_array != PETSC_CUSP_UNALLOCATED && (nsends>0 || nrecvs>0)) {
#else
  if (x->valid_GPU_array != PETSC_CUDA_UNALLOCATED && (nsends>0 || nrecvs>0)) {
#endif
    if (!inctx->spptr) {
      PetscInt k,*tindicesSends,*sindicesSends,*tindicesRecvs,*sindicesRecvs;
      PetscInt ns = sstartsSends[nsends],nr = sstartsRecvs[nrecvs];
      /* Here we create indices for both the senders and receivers. */
      ierr = PetscMalloc1(ns,&tindicesSends);CHKERRQ(ierr);
      ierr = PetscMalloc1(nr,&tindicesRecvs);CHKERRQ(ierr);

      ierr = PetscMemcpy(tindicesSends,indices,ns*sizeof(PetscInt));CHKERRQ(ierr);
      ierr = PetscMemcpy(tindicesRecvs,from->indices,nr*sizeof(PetscInt));CHKERRQ(ierr);

      ierr = PetscSortRemoveDupsInt(&ns,tindicesSends);CHKERRQ(ierr);
      ierr = PetscSortRemoveDupsInt(&nr,tindicesRecvs);CHKERRQ(ierr);

      ierr = PetscMalloc1(bs*ns,&sindicesSends);CHKERRQ(ierr);
      ierr = PetscMalloc1(from->bs*nr,&sindicesRecvs);CHKERRQ(ierr);

      /* sender indices */
      for (i=0; i<ns; i++) {
        for (k=0; k<bs; k++) sindicesSends[i*bs+k] = tindicesSends[i]+k;
      }
      ierr = PetscFree(tindicesSends);CHKERRQ(ierr);

      /* receiver indices */
      for (i=0; i<nr; i++) {
        for (k=0; k<from->bs; k++) sindicesRecvs[i*from->bs+k] = tindicesRecvs[i]+k;
      }
      ierr = PetscFree(tindicesRecvs);CHKERRQ(ierr);

      /* create GPU indices, work vectors, ... */
#if defined(PETSC_HAVE_CUSP)
      ierr = VecScatterCUSPIndicesCreate_PtoP(ns*bs,sindicesSends,nr*from->bs,sindicesRecvs,(PetscCUSPIndices*)&inctx->spptr);CHKERRQ(ierr);
#else
      ierr = VecScatterCUDAIndicesCreate_PtoP(ns*bs,sindicesSends,nr*from->bs,sindicesRecvs,(PetscCUDAIndices*)&inctx->spptr);CHKERRQ(ierr);
#endif
      ierr = PetscFree(sindicesSends);CHKERRQ(ierr);
      ierr = PetscFree(sindicesRecvs);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

/*@C
   VecScatterFinalizeForGPU - Finalizes a generalized scatter from one vector to
   another for GPU based computation.

   Input Parameter:
+  inctx - scatter context generated by VecScatterCreate()

  Level: intermediate

  Notes:
   Effectively, this function resets the temporary buffer flags. Currently, this
   only used in the context of the parallel SpMV call in in MatMult_MPIAIJCUDA
   or MatMult_MPIAIJCUDAARSE. Once the MatMultAdd is finished, the GPU temporary
   buffers used for messaging are no longer valid.

.seealso: VecScatterInitializeForGPU(), VecScatterCreate(), VecScatterEnd()
@*/
PETSC_EXTERN PetscErrorCode VecScatterFinalizeForGPU(VecScatter inctx)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

#endif

