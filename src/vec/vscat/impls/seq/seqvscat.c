
/*
     Code for creating scatters between vectors. This file
  includes the code for scattering between sequential vectors and
  some special cases for parallel scatters.
*/

#include <petsc/private/vecscatterimpl.h>    /*I   "petscvec.h"    I*/

#if defined(PETSC_HAVE_CUDA)
#include <../src/vec/vec/impls/seq/seqcuda/cudavecimpl.h>
#endif

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
#if defined(PETSC_HAVE_CUDA)
  PetscBool              is_veccuda,isy_veccuda;
#endif

  PetscFunctionBegin;
#if defined(PETSC_HAVE_CUDA)
  ierr = PetscObjectTypeCompareAny((PetscObject)x,&is_veccuda,VECSEQCUDA,VECMPICUDA,VECCUDA,"");CHKERRQ(ierr);
  ierr = PetscObjectTypeCompareAny((PetscObject)y,&isy_veccuda,VECSEQCUDA,VECMPICUDA,VECCUDA,"");CHKERRQ(ierr);
  if (is_veccuda && isy_veccuda && x->valid_GPU_array == PETSC_OFFLOAD_GPU) {
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
#if defined(PETSC_HAVE_CUDA)
  PetscBool              is_veccuda,isy_veccuda;
#endif

  PetscFunctionBegin;
#if defined(PETSC_HAVE_CUDA)
  ierr = PetscObjectTypeCompareAny((PetscObject)x,&is_veccuda,VECSEQCUDA,VECMPICUDA,VECCUDA,"");CHKERRQ(ierr);
  ierr = PetscObjectTypeCompareAny((PetscObject)y,&isy_veccuda,VECSEQCUDA,VECMPICUDA,VECCUDA,"");CHKERRQ(ierr);
  if (is_veccuda && isy_veccuda && x->valid_GPU_array == PETSC_OFFLOAD_GPU) {
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
#if defined(PETSC_HAVE_CUDA)
  PetscBool              is_veccuda,isy_veccuda;
#endif

  PetscFunctionBegin;
#if defined(PETSC_HAVE_CUDA)
  ierr = PetscObjectTypeCompareAny((PetscObject)x,&is_veccuda,VECSEQCUDA,VECMPICUDA,VECCUDA,"");CHKERRQ(ierr);
  ierr = PetscObjectTypeCompareAny((PetscObject)y,&isy_veccuda,VECSEQCUDA,VECMPICUDA,VECCUDA,"");CHKERRQ(ierr);
  if (is_veccuda && isy_veccuda && x->valid_GPU_array == PETSC_OFFLOAD_GPU) {
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
#if defined(PETSC_HAVE_CUDA)
  PetscBool              is_veccuda,isy_veccuda;
#endif

  PetscFunctionBegin;
#if defined(PETSC_HAVE_CUDA)
  ierr = PetscObjectTypeCompareAny((PetscObject)x,&is_veccuda,VECSEQCUDA,VECMPICUDA,VECCUDA,"");CHKERRQ(ierr);
  ierr = PetscObjectTypeCompareAny((PetscObject)y,&isy_veccuda,VECSEQCUDA,VECMPICUDA,VECCUDA,"");CHKERRQ(ierr);
  if (is_veccuda && isy_veccuda && x->valid_GPU_array == PETSC_OFFLOAD_GPU) {
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
#if defined(PETSC_HAVE_CUDA)
  PetscBool              is_veccuda,isy_veccuda;
#endif

  PetscFunctionBegin;
#if defined(PETSC_HAVE_CUDA)
  ierr = PetscObjectTypeCompareAny((PetscObject)x,&is_veccuda,VECSEQCUDA,VECMPICUDA,VECCUDA,"");CHKERRQ(ierr);
  ierr = PetscObjectTypeCompareAny((PetscObject)y,&isy_veccuda,VECSEQCUDA,VECMPICUDA,VECCUDA,"");CHKERRQ(ierr);
  if (is_veccuda && isy_veccuda && x->valid_GPU_array == PETSC_OFFLOAD_GPU) {
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
#if defined(PETSC_HAVE_CUDA)
  PetscBool              is_veccuda,isy_veccuda;
#endif

  PetscFunctionBegin;
#if defined(PETSC_HAVE_CUDA)
  ierr = PetscObjectTypeCompareAny((PetscObject)x,&is_veccuda,VECSEQCUDA,VECMPICUDA,VECCUDA,"");CHKERRQ(ierr);
  ierr = PetscObjectTypeCompareAny((PetscObject)y,&isy_veccuda,VECSEQCUDA,VECMPICUDA,VECCUDA,"");CHKERRQ(ierr);
  if (is_veccuda && isy_veccuda && x->valid_GPU_array == PETSC_OFFLOAD_GPU) {
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
      ierr = PetscArraycpy(yv+to_first,xv+from_first,n);CHKERRQ(ierr);
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
  ierr = PetscArraycpy(out_to->vslots,in_to->vslots,out_to->n);CHKERRQ(ierr);
  ierr = VecScatterMemcpyPlanCopy(&in_to->memcpy_plan,&out_to->memcpy_plan);CHKERRQ(ierr);

  out_from->n                    = in_from->n;
  out_from->format               = in_from->format;
  out_from->nonmatching_computed = PETSC_FALSE;
  out_from->slots_nonmatching    = 0;
  ierr = PetscArraycpy(out_from->vslots,in_from->vslots,out_from->n);CHKERRQ(ierr);
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
  ierr = PetscArraycpy(out_from->vslots,in_from->vslots,out_from->n);CHKERRQ(ierr);
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
          to->memcpy_plan.copy_lengths[j]    = to_slots[i]+bs-to->memcpy_plan.copy_starts[j];
          from->memcpy_plan.copy_lengths[j]  = from_slots[i]+bs-from->memcpy_plan.copy_starts[j];
          to->memcpy_plan.copy_starts[j+1]   = to_slots[i+1];
          from->memcpy_plan.copy_starts[j+1] = from_slots[i+1];
          j++;
        }
      }
    }

    /* set up copy_lengths[] of the last copy */
    to->memcpy_plan.copy_lengths[n_copies-1]   = to_slots[n-1]+bs-to->memcpy_plan.copy_starts[n_copies-1];
    from->memcpy_plan.copy_lengths[n_copies-1] = from_slots[n-1]+bs-from->memcpy_plan.copy_starts[n_copies-1];

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

/* -------------------------------------------------- */
PetscErrorCode VecScatterSetUp_Seq(VecScatter ctx)
{
  PetscErrorCode    ierr;
  PetscInt          ix_type=-1,iy_type=-1;
  IS                tix = NULL,tiy = NULL,ix=ctx->from_is,iy=ctx->to_is;
  Vec               xin=ctx->from_v;

  PetscFunctionBegin;
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
    ierr    = PetscArraycpy(to->vslots,idy,nx);CHKERRQ(ierr);
    from->n = nx;
#if defined(PETSC_USE_DEBUG)
    ierr = VecScatterCheckIndices_Private(ctx->from_n,nx,idx);CHKERRQ(ierr);
#endif
    ierr              = PetscArraycpy(from->vslots,idx,nx);CHKERRQ(ierr);
    to->format        = VEC_SCATTER_SEQ_GENERAL;
    from->format      = VEC_SCATTER_SEQ_GENERAL;
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
    ierr           = PetscArraycpy(from9->vslots,idx,nx);CHKERRQ(ierr);
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
    ierr = PetscArraycpy(to10->vslots,idy,nx);CHKERRQ(ierr);
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
    ierr = PetscArraycpy(to11->vslots,idy,nx);CHKERRQ(ierr);
    from11->n = nx;
#if defined(PETSC_USE_DEBUG)
    ierr = VecScatterCheckIndices_Private(ctx->from_n,nx,idx);CHKERRQ(ierr);
#endif
    ierr = PetscArraycpy(from11->vslots,idx,nx);CHKERRQ(ierr);
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

PetscErrorCode VecScatterCreate_Seq(VecScatter ctx)
{
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ctx->ops->setup = VecScatterSetUp_Seq;
  ierr = PetscObjectChangeTypeName((PetscObject)ctx,VECSCATTERSEQ);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

