
#ifndef lint
static char vcid[] = "$Id: vscat.c,v 1.92 1997/06/05 12:51:01 bsmith Exp bsmith $";
#endif

/*
     Code for creating scatters between vectors. This file 
  includes the code for scattering between sequential vectors and
  some special cases for parallel scatters.
*/

#include "src/is/isimpl.h"
#include "src/vec/vecimpl.h"                     /*I "vec.h" I*/
#include "src/vec/impls/dvecimpl.h"
#include "src/vec/impls/mpi/pvecimpl.h"

/*
      This is special scatter code for when the entire parallel vector is 
   copied to each processor.

   This code was written by Cameron Cooper, Occidental College, Fall 1995,
   will working at ANL as a SERS student.
*/
#undef __FUNC__  
#define __FUNC__ "VecScatterBegin_MPI_ToAll"
int VecScatterBegin_MPI_ToAll(Vec x,Vec y,InsertMode addv,ScatterMode mode,VecScatter ctx)
{ 
  if (mode & SCATTER_REVERSE) {
    Vec_MPI              *yy = (Vec_MPI *) y->data;
    Vec_Seq              *xx = (Vec_Seq *) x->data;
    Scalar               *xv = xx->array, *yv = yy->array, *xvt, *xvt2;
    VecScatter_MPI_ToAll *scat = (VecScatter_MPI_ToAll *) ctx->todata;
    int                  i, size = xx->n;

    if (addv == INSERT_VALUES) {
      /* 
         copy the correct part of the local vector into the local storage of 
         the MPI one  Note: this operation only makes sense if all the local 
         vectors have the same values
      */
      PetscMemcpy(yv,xv+yy->ownership[yy->rank],yy->n*sizeof(Scalar));
    } else {
      if (scat->work1) xvt = scat->work1; 
      else {
        scat->work1 = xvt = (Scalar *) PetscMalloc(size*sizeof(Scalar));CHKPTRQ(xvt);
        PLogObjectMemory(ctx,size*sizeof(Scalar));
      }
      if (!yy->rank) { /* I am the zeroth processor, values are accumulated here */
        if   (scat->work2) xvt2 = scat->work2; 
        else {
          scat->work2 = xvt2 = (Scalar *) PetscMalloc(size*sizeof(Scalar));CHKPTRQ(xvt2);
          PLogObjectMemory(ctx,size*sizeof(Scalar));
        }
        MPI_Gatherv(yv,yy->n,MPIU_SCALAR,xvt2,scat->count,yy->ownership,MPIU_SCALAR,0,ctx->comm);
#if defined(PETSC_COMPLEX)
        MPI_Reduce(xv, xvt, 2*size, MPI_DOUBLE, MPI_SUM, 0, ctx->comm);
#else
        MPI_Reduce(xv, xvt, size, MPIU_SCALAR, MPI_SUM, 0, ctx->comm);
#endif
	for ( i=0; i<size; i++ ) {
	  xvt[i] += xvt2[i];
	}
        MPI_Scatterv(xvt,scat->count,yy->ownership,MPIU_SCALAR,yv,yy->n,MPIU_SCALAR,0,ctx->comm);
      }
      else {
        MPI_Gatherv(yv, yy->n, MPIU_SCALAR, 0,  0, 0, MPIU_SCALAR, 0, ctx->comm);
#if defined(PETSC_COMPLEX)
        MPI_Reduce(xv, xvt, 2*size, MPI_DOUBLE, MPI_SUM, 0, ctx->comm);
#else
        MPI_Reduce(xv, xvt, size, MPIU_SCALAR, MPI_SUM, 0, ctx->comm);
#endif
        MPI_Scatterv(0,scat->count,yy->ownership,MPIU_SCALAR,yv,yy->n,MPIU_SCALAR,0,ctx->comm);
      }
    }
  }
  else {
    Vec_MPI              *xx = (Vec_MPI *) x->data;
    Vec_Seq              *yy = (Vec_Seq *) y->data;
    Scalar               *xv = xx->array, *yv = yy->array, *yvt;
    VecScatter_MPI_ToAll *scat = (VecScatter_MPI_ToAll *) ctx->todata;
    int                  i, size = yy->n;

    if (addv == INSERT_VALUES) {
      MPI_Allgatherv(xv,xx->n,MPIU_SCALAR,yv,scat->count,xx->ownership,MPIU_SCALAR,ctx->comm);
    }
    else {
      if (scat->work1) yvt = scat->work1; 
      else {
        scat->work1 = yvt = (Scalar *) PetscMalloc(size*sizeof(Scalar));CHKPTRQ(yvt);
        PLogObjectMemory(ctx,size*sizeof(Scalar));
      }
      MPI_Allgatherv(xv,xx->n,MPIU_SCALAR,yvt,scat->count,xx->ownership,MPIU_SCALAR,ctx->comm);
      for ( i=0; i<size; i++ ) {
	yv[i] += yvt[i];
      }
    }
  }
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "VecScatterDestroy_MPI_ToAll" /* ADIC Ignore */
int VecScatterDestroy_MPI_ToAll(PetscObject obj)
{
  VecScatter           ctx = (VecScatter) obj;
  VecScatter_MPI_ToAll *scat = (VecScatter_MPI_ToAll *) ctx->todata;

  PetscFree(scat->count);
  if (scat->work1) PetscFree(scat->work1);
  if (scat->work2) PetscFree(scat->work2);
  PetscFree(ctx->todata); 
  PLogObjectDestroy(ctx);
  PetscHeaderDestroy(ctx);
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "VecScatterCopy_MPI_ToAll"
int VecScatterCopy_MPI_ToAll(VecScatter in,VecScatter out)
{
  VecScatter_MPI_ToAll *in_to = (VecScatter_MPI_ToAll *) in->todata, *sto;
  int                  size, i;

  out->postrecvs = 0;
  out->begin     = in->begin;
  out->end       = in->end;
  out->copy      = in->copy;
  out->destroy   = in->destroy;
  out->view      = in->view;

  sto       = PetscNew(VecScatter_MPI_ToAll); CHKPTRQ(sto);
  sto->type = VEC_SCATTER_MPI_TOALL;

  MPI_Comm_size(out->comm,&size);
  sto->count = (int *) PetscMalloc(size*sizeof(int)); CHKPTRQ(sto->count);
  for ( i=0; i<size; i++ ) {
    sto->count[i] = in_to->count[i];
  }
  sto->work1         = 0;
  sto->work2         = 0;
  PLogObjectMemory(out,sizeof(VecScatter_MPI_ToAll)+size*sizeof(int));
  out->todata        = (void *) sto; 
  out->fromdata      = (void *) 0;
  return 0;
}

/* --------------------------------------------------------------------------------------*/
/* 
        Scatter: sequential general to sequential general 
*/
#undef __FUNC__  
#define __FUNC__ "VecScatterBegin_SGtoSG"
int VecScatterBegin_SGtoSG(Vec x,Vec y,InsertMode addv,ScatterMode mode,VecScatter ctx)
{
  VecScatter_Seq_General *gen_to = (VecScatter_Seq_General *) ctx->todata;
  VecScatter_Seq_General *gen_from = (VecScatter_Seq_General *) ctx->fromdata;
  int                    i, n = gen_from->n, *fslots, *tslots;
  Vec_Seq                *xx = (Vec_Seq *) x->data,*yy = (Vec_Seq *) y->data;
  Scalar                 *xv = xx->array, *yv = yy->array;
  
  if (mode & SCATTER_REVERSE ){
    gen_to   = (VecScatter_Seq_General *) ctx->fromdata;
    gen_from = (VecScatter_Seq_General *) ctx->todata;
  }
  fslots = gen_from->slots;
  tslots = gen_to->slots;

  if (addv == INSERT_VALUES) {
    for ( i=0; i<n; i++ ) {yv[tslots[i]] = xv[fslots[i]];}
  }
  else {
    for ( i=0; i<n; i++ ) {yv[tslots[i]] += xv[fslots[i]];}
  }
  return 0;
}

/* 
    Scatter: sequential general to sequential stride 1 
*/
#undef __FUNC__  
#define __FUNC__ "VecScatterBegin_SGtoSS_Stride1"
int VecScatterBegin_SGtoSS_Stride1(Vec x,Vec y,InsertMode addv,ScatterMode mode,VecScatter ctx)
{
  VecScatter_Seq_Stride  *gen_to   = (VecScatter_Seq_Stride *) ctx->todata;
  VecScatter_Seq_General *gen_from = (VecScatter_Seq_General *) ctx->fromdata;
  int                    i, n = gen_from->n, *fslots = gen_from->slots;
  int                    first = gen_to->first;
  Vec_Seq                *xx = (Vec_Seq *) x->data,*yy = (Vec_Seq *) y->data;
  Scalar                 *xv = xx->array, *yv = yy->array;
  
  if (mode & SCATTER_REVERSE ){
    xv += first;
    if (addv == INSERT_VALUES) {
      for ( i=0; i<n; i++ ) {yv[fslots[i]] = xv[i];}
    }
    else {
      for ( i=0; i<n; i++ ) {yv[fslots[i]] += xv[i];}
    }
  } else {
    yv += first;
    if (addv == INSERT_VALUES) {
      for ( i=0; i<n; i++ ) {yv[i] = xv[fslots[i]];}
    }
    else {
      for ( i=0; i<n; i++ ) {yv[i] += xv[fslots[i]];}
    }
  }
  return 0;
}

/* 
   Scatter: sequential general to sequential stride 
*/
#undef __FUNC__  
#define __FUNC__ "VecScatterBegin_SGtoSS"
int VecScatterBegin_SGtoSS(Vec x,Vec y,InsertMode addv,ScatterMode mode,VecScatter ctx)
{
  VecScatter_Seq_Stride  *gen_to   = (VecScatter_Seq_Stride *) ctx->todata;
  VecScatter_Seq_General *gen_from = (VecScatter_Seq_General *) ctx->fromdata;
  int                    i, n = gen_from->n, *fslots = gen_from->slots;
  int                    first = gen_to->first,step = gen_to->step;
  Vec_Seq                *xx = (Vec_Seq *) x->data,*yy = (Vec_Seq *) y->data;
  Scalar                 *xv = xx->array, *yv = yy->array;
  
  if (mode & SCATTER_REVERSE ){
    if (addv == INSERT_VALUES) {
      for ( i=0; i<n; i++ ) {yv[fslots[i]] = xv[first + i*step];}
    }
    else {
      for ( i=0; i<n; i++ ) {yv[fslots[i]] += xv[first + i*step];}
    }
  } else {
    if (addv == INSERT_VALUES) {
      for ( i=0; i<n; i++ ) {yv[first + i*step] = xv[fslots[i]];}
    }
    else {
      for ( i=0; i<n; i++ ) {yv[first + i*step] += xv[fslots[i]];}
    }
  }
  return 0;
}

/* 
    Scatter: sequential stride 1 to sequential general 
*/
#undef __FUNC__  
#define __FUNC__ "VecScatterBegin_SStoSG_Stride1"
int VecScatterBegin_SStoSG_Stride1(Vec x,Vec y,InsertMode addv,ScatterMode mode,VecScatter ctx)
{
  VecScatter_Seq_Stride  *gen_from = (VecScatter_Seq_Stride *) ctx->fromdata;
  VecScatter_Seq_General *gen_to   = (VecScatter_Seq_General *) ctx->todata;
  int                    i, n = gen_from->n, *fslots = gen_to->slots;
  int                    first = gen_from->first;
  Vec_Seq                *xx = (Vec_Seq *) x->data,*yy = (Vec_Seq *) y->data;
  Scalar                 *xv = xx->array, *yv = yy->array;
  
  if (mode & SCATTER_REVERSE ){
    yv += first;
    if (addv == INSERT_VALUES) {
      for ( i=0; i<n; i++ ) {yv[i] = xv[fslots[i]];}
    }
    else {
      for ( i=0; i<n; i++ ) {yv[i] += xv[fslots[i]];}
    }
  } else {
    xv += first;
    if (addv == INSERT_VALUES) {
      for ( i=0; i<n; i++ ) {yv[fslots[i]] = xv[i];}
    }
    else {
      for ( i=0; i<n; i++ ) {yv[fslots[i]] += xv[i];}
    }
  }
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "VecScatterBegin_SStoSG"
/* 
   Scatter: sequential stride to sequential general 
*/
int VecScatterBegin_SStoSG(Vec x,Vec y,InsertMode addv,ScatterMode mode,VecScatter ctx)
{
  VecScatter_Seq_Stride  *gen_from = (VecScatter_Seq_Stride *) ctx->fromdata;
  VecScatter_Seq_General *gen_to   = (VecScatter_Seq_General *) ctx->todata;
  int                    i, n = gen_from->n, *fslots = gen_to->slots;
  int                    first = gen_from->first,step = gen_from->step;
  Vec_Seq                *xx = (Vec_Seq *) x->data,*yy = (Vec_Seq *) y->data;
  Scalar                 *xv = xx->array, *yv = yy->array;
  
  if (mode & SCATTER_REVERSE ){
    if (addv == INSERT_VALUES) {
      for ( i=0; i<n; i++ ) {yv[first + i*step] = xv[fslots[i]];}
    }
    else {
      for ( i=0; i<n; i++ ) {yv[first + i*step] += xv[fslots[i]];}
    }
  } else {
    if (addv == INSERT_VALUES) {
      for ( i=0; i<n; i++ ) {yv[fslots[i]] = xv[first + i*step];}
    }
    else {
      for ( i=0; i<n; i++ ) {yv[fslots[i]] += xv[first + i*step];}
    }
  }
  return 0;
}

/* 
     Scatter: sequential stride to sequential stride 
*/
#undef __FUNC__  
#define __FUNC__ "VecScatterBegin_SStoSS"
int VecScatterBegin_SStoSS(Vec x,Vec y,InsertMode addv,ScatterMode mode,VecScatter ctx)
{
  VecScatter_Seq_Stride *gen_to   = (VecScatter_Seq_Stride *) ctx->todata;
  VecScatter_Seq_Stride *gen_from = (VecScatter_Seq_Stride *) ctx->fromdata;
  int                   i, n = gen_from->n, to_first = gen_to->first,to_step = gen_to->step;
  int                   from_first = gen_from->first,from_step = gen_from->step;
  Vec_Seq               *xx = (Vec_Seq *) x->data,*yy = (Vec_Seq *) y->data;
  Scalar                *xv = xx->array, *yv = yy->array;
  
  if (mode & SCATTER_REVERSE ){
    from_first = gen_to->first; 
    to_first   = gen_from->first;
    from_step  = gen_to->step; 
    to_step    = gen_from->step;
  }

  if (addv == INSERT_VALUES) {
    if (to_step == 1 && from_step == 1) {
      PetscMemcpy(yv+to_first,xv+from_first,n*sizeof(Scalar));
    }
    else {
      for ( i=0; i<n; i++ ) {
        yv[to_first + i*to_step] = xv[from_first+i*from_step];
      }
    }
  }
  else {
    if (to_step == 1 && from_step == 1) {
      yv += to_first; xv += from_first;
      for ( i=0; i<n; i++ ) {
        yv[i] += xv[i];
      }
    }
    else {
      for ( i=0; i<n; i++ ) {
        yv[to_first + i*to_step] += xv[from_first+i*from_step];
      }
    }
  }
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "VecScatterDestroy_SGtoSG"
int VecScatterDestroy_SGtoSG(PetscObject obj)
{
  VecScatter ctx = (VecScatter) obj;
  PetscFree(ctx->todata); PetscFree(ctx->fromdata); 
  PLogObjectDestroy(ctx);
  PetscHeaderDestroy(ctx);
  return 0;
}

/* --------------------------------------------------------------------------*/
/* 
    Scatter: parallel to sequential vector, sequential strides for both. 
*/
#undef __FUNC__  
#define __FUNC__ "VecScatterCopy_PStoSS"
int VecScatterCopy_PStoSS(VecScatter in,VecScatter out)
{
  VecScatter_Seq_Stride *in_to   = (VecScatter_Seq_Stride *) in->todata, *out_to;
  VecScatter_Seq_Stride *in_from = (VecScatter_Seq_Stride *) in->fromdata, *out_from;

  out->postrecvs  = 0;
  out->begin      = in->begin;
  out->end        = in->end;
  out->copy       = in->copy;
  out->destroy    = in->destroy;
  out->view       = in->view;

  out_to          = PetscNew(VecScatter_Seq_Stride); CHKPTRQ(out_to);
  out_to->n       = in_to->n; 
  out_to->first   = in_to->first; 
  out_to->step    = in_to->step;
  out_to->type    = in_to->type;
  out_from        = PetscNew(VecScatter_Seq_Stride); CHKPTRQ(out_from);
  PLogObjectMemory(out,2*sizeof(VecScatter_Seq_Stride));
  out_from->n     = in_from->n; 
  out_from->first = in_from->first; 
  out_from->step  = in_from->step;
  out_from->type  = in_from->type;
  out->todata     = (void *) out_to; 
  out->fromdata   = (void *) out_from;
  return 0;
}

int VecScatterCreate_PtoS(int,int *,int,int *,Vec,int,VecScatter);
int VecScatterCreate_PtoP(int,int *,int,int *,Vec,Vec,VecScatter);
int VecScatterCreate_StoP(int,int *,int,int *,Vec,VecScatter);

/* --------------------------------------------------------------*/
#undef __FUNC__  
#define __FUNC__ "VecScatterCreate"
/*@C
   VecScatterCreate - Creates a vector scatter context.

   Input Parameters:
.  xin - the vector from which we scatter
.  yin - the vector to which we scatter
.  ix - the indices of xin to scatter
.  iy - the indices of yin to hold results

   Output Parameter:
.  newctx - location to store the new scatter context

   Notes:
   A VecScatter context CANNOT be used in two or more simultaneous scatters.
   In this case a separate VecScatter is needed for each concurrent scatter.

.keywords: vector, scatter, context, create

.seealso: VecScatterDestroy()
@*/
int VecScatterCreate(Vec xin,IS ix,Vec yin,IS iy,VecScatter *newctx)
{
  VecScatter ctx;
  int        len,size,cando,islocal,totalv,ierr; 
  MPI_Comm   comm = xin->comm;
  PetscTruth ixblock,iyblock,iystride;

  /* next 2 lines insure that we use parallel comm if it exists */
  MPI_Comm_size(yin->comm,&size);
  if (size > 1) comm = yin->comm; 

  /* generate the Scatter context */
  PetscHeaderCreate(ctx,_p_VecScatter,VEC_SCATTER_COOKIE,0,comm);
  PLogObjectCreate(ctx);
  PLogObjectMemory(ctx,sizeof(struct _p_VecScatter));
  ctx->inuse = 0;

  VecGetLocalSize_Fast(xin,ctx->to_n);
  VecGetLocalSize_Fast(yin,ctx->from_n);

  /*
        Check for special cases
  */
  /* ---------------------------------------------------------------------------*/
  if (xin->type == VECSEQ && yin->type == VECSEQ) {
    if (ix->type == IS_GENERAL && iy->type == IS_GENERAL){
      int                    nx,ny,*idx,*idy;
      VecScatter_Seq_General *to,*from;

      ISGetSize(ix,&nx); ISGetIndices(ix,&idx);
      ISGetSize(iy,&ny); ISGetIndices(iy,&idy);
      if (nx != ny) SETERRQ(1,0,"Local scatter sizes don't match");
      len               = sizeof(VecScatter_Seq_General) + nx*sizeof(int);
      to                = (VecScatter_Seq_General *) PetscMalloc(len); CHKPTRQ(to)
      PLogObjectMemory(ctx,2*len);
      to->slots         = (int *) (to + 1); 
      to->n             = nx; 
      PetscMemcpy(to->slots,idy,nx*sizeof(int));
      from              = (VecScatter_Seq_General *) PetscMalloc(len); CHKPTRQ(from);
      from->slots       = (int *) (from + 1);
      from->n           = nx; 
      PetscMemcpy(from->slots,idx,nx*sizeof(int));
      to->type          = VEC_SCATTER_SEQ_GENERAL; 
      from->type        = VEC_SCATTER_SEQ_GENERAL; 
      ctx->todata       = (void *) to; 
      ctx->fromdata     = (void *) from;
      ctx->postrecvs    = 0;
      ctx->begin        = VecScatterBegin_SGtoSG; 
      ctx->end          = 0; 
      ctx->destroy      = VecScatterDestroy_SGtoSG;
      ctx->copy         = 0;
      *newctx           = ctx;
      return 0;
    }
    else if (ix->type == IS_STRIDE &&  iy->type == IS_STRIDE){
      int                    nx,ny,to_first,to_step,from_first,from_step;
      VecScatter_Seq_Stride  *from,*to;

      ISGetSize(ix,&nx); ISStrideGetInfo(ix,&from_first,&from_step);
      ISGetSize(iy,&ny); ISStrideGetInfo(iy,&to_first,&to_step);
      if (nx != ny) SETERRQ(1,0,"Local scatter sizes don't match");
      to                = PetscNew(VecScatter_Seq_Stride); CHKPTRQ(to);
      to->n             = nx; 
      to->first         = to_first; 
      to->step          = to_step;
      from              = PetscNew(VecScatter_Seq_Stride);CHKPTRQ(from);
      PLogObjectMemory(ctx,2*sizeof(VecScatter_Seq_Stride));
      from->n           = nx;
      from->first       = from_first; 
      from->step        = from_step;
      to->type          = VEC_SCATTER_SEQ_STRIDE; 
      from->type        = VEC_SCATTER_SEQ_STRIDE; 
      ctx->todata       = (void *) to; 
      ctx->fromdata     = (void *) from;
      ctx->postrecvs    = 0;
      ctx->begin        = VecScatterBegin_SStoSS; 
      ctx->end          = 0; 
      ctx->destroy      = VecScatterDestroy_SGtoSG;
      ctx->copy         = 0;
      *newctx           = ctx;
      return 0; 
    }
    else if (ix->type == IS_GENERAL && iy->type == IS_STRIDE){
      int                    nx,ny,*idx,first,step;
      VecScatter_Seq_General *from;
      VecScatter_Seq_Stride  *to;

      ISGetSize(ix,&nx); ISGetIndices(ix,&idx);
      ISGetSize(iy,&ny); ISStrideGetInfo(iy,&first,&step);
      if (nx != ny) SETERRQ(PETSC_ERR_ARG_SIZ,0,"Local scatter sizes don't match");
      to              = PetscNew(VecScatter_Seq_Stride); CHKPTRQ(to);
      to->n           = nx; 
      to->first       = first; 
      to->step        = step;
      len             = sizeof(VecScatter_Seq_General) + nx*sizeof(int);
      from            = (VecScatter_Seq_General *) PetscMalloc(len); CHKPTRQ(from);
      PLogObjectMemory(ctx,len + sizeof(VecScatter_Seq_Stride));
      from->slots     = (int *) (from + 1); 
      from->n         = nx; 
      PetscMemcpy(from->slots,idx,nx*sizeof(int));
      ctx->todata     = (void *) to; ctx->fromdata = (void *) from;
      ctx->postrecvs  = 0;
      if (step == 1)  ctx->begin = VecScatterBegin_SGtoSS_Stride1;
      else            ctx->begin = VecScatterBegin_SGtoSS;
      ctx->destroy    = VecScatterDestroy_SGtoSG;
      ctx->end        = 0; 
      ctx->copy       = 0;
      to->type        = VEC_SCATTER_SEQ_STRIDE; 
      from->type      = VEC_SCATTER_SEQ_GENERAL;
      *newctx         = ctx;
      return 0;
    }
    else if (ix->type == IS_STRIDE && iy->type == IS_GENERAL){
      int                    nx,ny,*idx,first,step;
      VecScatter_Seq_General *to;
      VecScatter_Seq_Stride  *from;

      ISGetSize(ix,&nx); ISGetIndices(iy,&idx);
      ISGetSize(iy,&ny); ISStrideGetInfo(ix,&first,&step);
      if (nx != ny) SETERRQ(PETSC_ERR_ARG_SIZ,0,"Local scatter sizes don't match");
      from            = PetscNew(VecScatter_Seq_Stride);CHKPTRQ(from);
      from->n         = nx; 
      from->first     = first; 
      from->step      = step;
      len             = sizeof(VecScatter_Seq_General) + nx*sizeof(int);
      to              = (VecScatter_Seq_General *) PetscMalloc(len); CHKPTRQ(to);
      PLogObjectMemory(ctx,len + sizeof(VecScatter_Seq_Stride));
      to->slots       = (int *) (to + 1); 
      to->n           = nx; 
      PetscMemcpy(to->slots,idx,nx*sizeof(int));
      ctx->todata     = (void *) to; 
      ctx->fromdata   = (void *) from;
      ctx->postrecvs  = 0;
      if (step == 1) ctx->begin = VecScatterBegin_SStoSG_Stride1; 
      else           ctx->begin = VecScatterBegin_SStoSG; 
      ctx->destroy    = VecScatterDestroy_SGtoSG;
      ctx->end        = 0; 
      ctx->copy       = 0;
      to->type        = VEC_SCATTER_SEQ_GENERAL; 
      from->type      = VEC_SCATTER_SEQ_STRIDE; 
      *newctx         = ctx;
      return 0;
    }
    else {
      SETERRQ(PETSC_ERR_SUP,0,"Cannot generate such a scatter context yet");
    }
  }
  /* ---------------------------------------------------------------------------*/
  if (xin->type == VECMPI && yin->type == VECSEQ) {
    islocal = 0;
    /* special case extracting (subset of) local portion */ 
    if (ix->type == IS_STRIDE && iy->type == IS_STRIDE){
      Vec_MPI               *x = (Vec_MPI *)xin->data;
      int                   nx,ny,to_first,to_step,from_first,from_step;
      int                   start = x->ownership[x->rank], end = x->ownership[x->rank+1];
      VecScatter_Seq_Stride *from,*to;

      ISGetSize(ix,&nx); ISStrideGetInfo(ix,&from_first,&from_step);
      ISGetSize(iy,&ny); ISStrideGetInfo(iy,&to_first,&to_step);
      if (nx != ny) SETERRQ(PETSC_ERR_ARG_SIZ,0,"Local scatter sizes don't match");
      if (ix->min >= start && ix->max < end ) islocal = 1; else islocal = 0;
      MPI_Allreduce( &islocal, &cando,1,MPI_INT,MPI_LAND,xin->comm);
      if (cando) {
        to                = PetscNew(VecScatter_Seq_Stride);CHKPTRQ(to);
        to->n             = nx; 
        to->first         = to_first;
        to->step          = to_step;
        from              = PetscNew(VecScatter_Seq_Stride);CHKPTRQ(from);
        PLogObjectMemory(ctx,2*sizeof(VecScatter_Seq_Stride));
        from->n           = nx; 
        from->first       = from_first-start; 
        from->step        = from_step;
        to->type          = VEC_SCATTER_SEQ_STRIDE; 
        from->type        = VEC_SCATTER_SEQ_STRIDE; 
        ctx->todata       = (void *) to; 
        ctx->fromdata     = (void *) from;
        ctx->postrecvs    = 0;
        ctx->begin        = VecScatterBegin_SStoSS; 
        ctx->end          = 0; 
        ctx->destroy      = VecScatterDestroy_SGtoSG;
        ctx->copy         = VecScatterCopy_PStoSS;
        *newctx           = ctx;
        return 0;
      }
    }
    else {
      MPI_Allreduce( &islocal, &cando,1,MPI_INT,MPI_LAND,xin->comm);
    }
    /* test for special case of all processors getting entire vector */
    totalv = 0;
    if (ix->type == IS_STRIDE && iy->type == IS_STRIDE){
      Vec_MPI              *x = (Vec_MPI *)xin->data;
      int                  i,nx,ny,to_first,to_step,from_first,from_step,*count;
      VecScatter_MPI_ToAll *sto;

      ISGetSize(ix,&nx); ISStrideGetInfo(ix,&from_first,&from_step);
      ISGetSize(iy,&ny); ISStrideGetInfo(iy,&to_first,&to_step);
      if (nx != ny) SETERRQ(PETSC_ERR_ARG_SIZ,0,"Local scatter sizes don't match");
      if (nx != x->N) {
        totalv = 0;
      } else if (from_first == 0        && from_step == 1 && 
                 from_first == to_first && from_step == to_step){
        totalv = 1; 
      } else totalv = 0;
      MPI_Allreduce(&totalv,&cando,1,MPI_INT,MPI_LAND,xin->comm);

      if (cando) {
        MPI_Comm_size(ctx->comm,&size);
        sto   = PetscNew(VecScatter_MPI_ToAll);CHKPTRQ(sto);
        count = (int *) PetscMalloc(size*sizeof(int)); CHKPTRQ(count);
        for ( i=0; i<size; i++ ) {
	  count[i] = x->ownership[i+1]-x->ownership[i];
        }
        sto->count        = count;
        sto->work1        = 0;
        sto->work2        = 0;
        sto->type         = VEC_SCATTER_MPI_TOALL;
        PLogObjectMemory(ctx,sizeof(VecScatter_MPI_ToAll)+size*sizeof(int));
        ctx->todata       = (void *) sto;
        ctx->fromdata     = 0;
        ctx->postrecvs    = 0;
        ctx->begin        = VecScatterBegin_MPI_ToAll;   
        ctx->end          = 0;
        ctx->destroy      = VecScatterDestroy_MPI_ToAll;
        ctx->copy         = VecScatterCopy_MPI_ToAll;
        *newctx           = ctx;
        return 0;
      }
    }
    else {
      MPI_Allreduce( &totalv, &cando,1,MPI_INT,MPI_LAND,xin->comm);
    }
    ierr = ISBlock(ix,&ixblock); CHKERRQ(ierr);
    ierr = ISBlock(iy,&iyblock); CHKERRQ(ierr);
    ierr = ISStride(iy,&iystride); CHKERRQ(ierr);
    /* test for blocked indices */
    if (ixblock) {
      if (iyblock) {
        int nx, ny, *idx, *idy, bsx, bsy;
        ierr = ISBlockGetBlockSize(iy,&bsy); CHKERRQ(ierr);
        ierr = ISBlockGetBlockSize(ix,&bsx); CHKERRQ(ierr);
        if (bsx == bsy && (bsx == 12 || bsx == 5 || bsx == 4 || bsx == 3 || bsx == 2)) {
          ISBlockGetSize(ix,&nx); ISBlockGetIndices(ix,&idx);
          ISBlockGetSize(iy,&ny); ISBlockGetIndices(iy,&idy);
          if (nx != ny) SETERRQ(1,0,"Local scatter sizes don't match");
          ierr = VecScatterCreate_PtoS(nx,idx,ny,idy,xin,bsx,ctx); CHKERRQ(ierr);
          ISBlockRestoreIndices(ix,&idx);
          ISBlockRestoreIndices(iy,&idy);
          *newctx = ctx;
          return 0;
        }
      } else if (iystride) {
        int ystart,ystride,ysize,bsx;
        ierr = ISStrideGetInfo(iy,&ystart,&ystride);CHKERRQ(ierr);
        ierr = ISGetSize(iy,&ysize); CHKERRQ(ierr);
        ierr = ISBlockGetBlockSize(ix,&bsx); CHKERRQ(ierr);
        /* see if stride index set is equivalent to block index set */
        if (((bsx == 2) || (bsx == 3) || (bsx == 4) || (bsx == 5) || (bsx == 12)) && 
            ((ystart % bsx) == 0) && (ystride == 1) && ((ysize % bsx) == 0)) {
          int nx, *idx, *idy,il;
          ISBlockGetSize(ix,&nx); ISBlockGetIndices(ix,&idx);
          if (ysize != bsx*nx) SETERRQ(1,0,"Local scatter sizes don't match");
          idy    = (int *) PetscMalloc( nx*sizeof(int) );CHKPTRQ(idy);
          idy[0] = ystart;
          for ( il=1; il<nx; il++ ) idy[il] = idy[il-1] + bsx; 
          ierr = VecScatterCreate_PtoS(nx,idx,nx,idy,xin,bsx,ctx); CHKERRQ(ierr);
          PetscFree(idy);
          ISBlockRestoreIndices(ix,&idx);
          *newctx = ctx;
          return 0;
        }
      }
    }
    /* left over general case */
    {
      int nx,ny,*idx,*idy;
      ISGetSize(ix,&nx); ISGetIndices(ix,&idx);
      ISGetSize(iy,&ny); ISGetIndices(iy,&idy);
      if (nx != ny) SETERRQ(PETSC_ERR_ARG_SIZ,0,"Local scatter sizes don't match");
      ierr = VecScatterCreate_PtoS(nx,idx,ny,idy,xin,1,ctx); CHKERRQ(ierr);
      ISRestoreIndices(ix,&idx);
      ISRestoreIndices(iy,&idy);
      *newctx = ctx;
      return 0;
    }
  }
  /* ---------------------------------------------------------------------------*/
  if (xin->type == VECSEQ && yin->type == VECMPI) {
    /* special case local copy portion */ 
    islocal = 0;
    if (ix->type == IS_STRIDE && iy->type == IS_STRIDE){
      Vec_MPI               *y = (Vec_MPI *)yin->data;
      int                   nx,ny,to_first,to_step,from_step,start=y->ownership[y->rank];
      int                   end = y->ownership[y->rank+1],from_first;
      VecScatter_Seq_Stride *from,*to;

      ISGetSize(ix,&nx); ISStrideGetInfo(ix,&from_first,&from_step);
      ISGetSize(iy,&ny); ISStrideGetInfo(iy,&to_first,&to_step);
      if (nx != ny) SETERRQ(PETSC_ERR_ARG_SIZ,0,"Local scatter sizes don't match");
      if (iy->min >= start && iy->max < end ) islocal = 1; else islocal = 0;
      MPI_Allreduce( &islocal, &cando,1,MPI_INT,MPI_LAND,yin->comm);
      if (cando) {
        to                = PetscNew(VecScatter_Seq_Stride);CHKPTRQ(to);
        to->n             = nx; 
        to->first         = to_first-start; 
        to->step          = to_step;
        from              = PetscNew(VecScatter_Seq_Stride);CHKPTRQ(from);
        PLogObjectMemory(ctx,2*sizeof(VecScatter_Seq_Stride));
        from->n           = nx; 
        from->first       = from_first; 
        from->step        = from_step;
        to->type          = VEC_SCATTER_SEQ_STRIDE; 
        from->type        = VEC_SCATTER_SEQ_STRIDE;
        ctx->todata       = (void *) to;
        ctx->fromdata     = (void *) from;
        ctx->postrecvs    = 0;
        ctx->begin        = VecScatterBegin_SStoSS; 
        ctx->end          = 0;  
        ctx->destroy      = VecScatterDestroy_SGtoSG;
        ctx->copy         = 0;
        *newctx           = ctx;
        return 0;
      }
    }
    else {
      MPI_Allreduce( &islocal, &cando,1,MPI_INT,MPI_LAND,yin->comm);
    }
    /* general case */
    {
      int nx,ny,*idx,*idy;
      ISGetSize(ix,&nx); ISGetIndices(ix,&idx);
      ISGetSize(iy,&ny); ISGetIndices(iy,&idy);
      if (nx != ny) SETERRQ(1,0,"Local scatter sizes don't match");
      ierr = VecScatterCreate_StoP(nx,idx,ny,idy,yin,ctx); CHKERRQ(ierr);
      ISRestoreIndices(ix,&idx); ISRestoreIndices(iy,&idy);
      *newctx = ctx;
      return 0;
    }
  }
  /* ---------------------------------------------------------------------------*/
  if (xin->type == VECMPI && yin->type == VECMPI) {
    /* no special cases for now */
    int nx,ny,*idx,*idy;
    ISGetSize(ix,&nx); ISGetIndices(ix,&idx);
    ISGetSize(iy,&ny); ISGetIndices(iy,&idy);
    if (nx != ny) SETERRQ(PETSC_ERR_ARG_SIZ,0,"Local scatter sizes don't match");
    ierr    = VecScatterCreate_PtoP(nx,idx,ny,idy,xin,yin,ctx); CHKERRQ(ierr);
    ISRestoreIndices(ix,&idx); 
    ISRestoreIndices(iy,&idy);
    *newctx = ctx;
    return 0;
  }
  /* One of the above 4 cases is true, and the next istruction should never be executed */
  return 0;
}

/* ------------------------------------------------------------------*/
#undef __FUNC__  
#define __FUNC__ "VecScatterPostRecvs"
/*@
   VecScatterPostRecvs - Posts the receives required for the ready-receiver
       version of the VecScatter routines.

   Input Parameters:
.  x - the vector from which we scatter (not needed, can be null)
.  y - the vector to which we scatter
.  addv - either ADD_VALUES or INSERT_VALUES
.  mode - the scattering mode, usually SCATTER_FORWARD.  The available modes are:
$    SCATTER_FORWARD, SCATTER_REVERSE
.  inctx - scatter context generated by VecScatterCreate()

   Output Parameter:
.  y - the vector to which we scatter

   Notes:
   If you use SCATTER_REVERSE the first two arguments should be reversed, from 
   the SCATTER_FORWARD.
   The vectors x and y cannot be the same. y[iy[i]] = x[ix[i]], for i=0,...,ni-1

   This scatter is far more general than the conventional
   scatter, since it can be a gather or a scatter or a combination,
   depending on the indices ix and iy.  If x is a parallel vector and y
   is sequential, VecScatterBegin() can serve to gather values to a
   single processor.  Similarly, if y is parallel and x sequential, the
   routine can scatter from one processor to many processors.

.keywords: vector, scatter, gather, begin

.seealso: VecScatterCreate(), VecScatterEnd(), VecScatterBegin()
@*/
int VecScatterPostRecvs(Vec x,Vec y,InsertMode addv,ScatterMode mode,VecScatter inctx)
{
  int ierr;
  PetscValidHeaderSpecific(y,VEC_COOKIE);
  PetscValidHeaderSpecific(inctx,VEC_SCATTER_COOKIE);

  if (inctx->postrecvs) {
    ierr = (*inctx->postrecvs)(x,y,addv,mode,inctx); CHKERRQ(ierr);
  }
  return 0;
}

/* ------------------------------------------------------------------*/
#undef __FUNC__  
#define __FUNC__ "VecScatterBegin"
/*@
   VecScatterBegin - Begins a generalized scatter from one vector to
   another. Complete the scattering phase with VecScatterEnd().

   Input Parameters:
.  x - the vector from which we scatter
.  y - the vector to which we scatter
.  addv - either ADD_VALUES or INSERT_VALUES
.  mode - the scattering mode, usually SCATTER_FORWARD.  The available modes are:
$    SCATTER_FORWARD, SCATTER_REVERSE
.  inctx - scatter context generated by VecScatterCreate()

   Output Parameter:
.  y - the vector to which we scatter

   Notes:
   If you use SCATTER_REVERSE the first two arguments should be reversed, from 
   the SCATTER_FORWARD.
   
   y[iy[i]] = x[ix[i]], for i=0,...,ni-1

   This scatter is far more general than the conventional
   scatter, since it can be a gather or a scatter or a combination,
   depending on the indices ix and iy.  If x is a parallel vector and y
   is sequential, VecScatterBegin() can serve to gather values to a
   single processor.  Similarly, if y is parallel and x sequential, the
   routine can scatter from one processor to many processors.

.keywords: vector, scatter, gather, begin

.seealso: VecScatterCreate(), VecScatterEnd()
@*/
int VecScatterBegin(Vec x,Vec y,InsertMode addv,ScatterMode mode,VecScatter inctx)
{
  int ierr;
#if defined(PETSC_BOPT_g)
  int to_n,from_n;
#endif
  PetscValidHeaderSpecific(x,VEC_COOKIE); PetscValidHeaderSpecific(y,VEC_COOKIE);
  PetscValidHeaderSpecific(inctx,VEC_SCATTER_COOKIE);
  if (inctx->inuse) SETERRQ(1,0," Scatter ctx already in use");
#if defined(PETSC_BOPT_g)
  /*
     Error checking to make sure these vectors match the vectors used
   to create the vector scatter context. -1 in the from_n and to_n indicate the
   vector lengths are unknown (for example with mapped scatters) and thus 
   no error checking is performed.
  */
  if (inctx->from_n >= 0 && inctx->to_n >= 0) {
    VecGetLocalSize_Fast(x,to_n);
    VecGetLocalSize_Fast(y,from_n);
    if (mode & SCATTER_REVERSE) {
      if (to_n != inctx->from_n) SETERRQ(PETSC_ERR_ARG_SIZ,0,"Vector wrong size for scatter");
      if (from_n != inctx->to_n) SETERRQ(PETSC_ERR_ARG_SIZ,0,"Vector wrong size for scatter");
    } else {
      if (to_n != inctx->to_n)     SETERRQ(PETSC_ERR_ARG_SIZ,0,"Vector wrong size for scatter");
      if (from_n != inctx->from_n) SETERRQ(PETSC_ERR_ARG_SIZ,0,"Vector wrong size for scatter");
    }
  }
#endif

  inctx->inuse = 1;
  PLogEventBegin(VEC_ScatterBegin,inctx,x,y,0);
  ierr = (*inctx->begin)(x,y,addv,mode,inctx); CHKERRQ(ierr);
  PLogEventEnd(VEC_ScatterBegin,inctx,x,y,0);
  return 0;
}

/* --------------------------------------------------------------------*/
#undef __FUNC__  
#define __FUNC__ "VecScatterEnd"
/*@
   VecScatterEnd - Ends a generalized scatter from one vector to another.  Call
   after first calling VecScatterBegin().

   Input Parameters:
.  x - the vector from which we scatter
.  y - the vector to which we scatter
.  addv - either ADD_VALUES or INSERT_VALUES.
.  mode - the scattering mode, usually SCATTER_FORWARD.  The available modes are:
$    SCATTER_FORWARD, SCATTER_REVERSE
.  ctx - scatter context generated by VecScatterCreate()

   Output Parameter:
.  y - the vector to which we scatter

   Notes:
   If you use SCATTER_REVERSE the first two arguments should be reversed, from 
   the SCATTER_FORWARD.
   y[iy[i]] = x[ix[i]], for i=0,...,ni-1

.keywords: vector, scatter, gather, end

.seealso: VecScatterBegin(), VecScatterCreate()
@*/
int VecScatterEnd(Vec x,Vec y,InsertMode addv,ScatterMode mode, VecScatter ctx)
{
  int ierr;
  PetscValidHeaderSpecific(x,VEC_COOKIE); PetscValidHeaderSpecific(y,VEC_COOKIE);
  PetscValidHeaderSpecific(ctx,VEC_SCATTER_COOKIE);
  ctx->inuse = 0;
  if (!ctx->end) return 0;
  PLogEventBegin(VEC_ScatterEnd,ctx,x,y,0);
  ierr = (*(ctx)->end)(x,y,addv,mode,ctx); CHKERRQ(ierr);
  PLogEventEnd(VEC_ScatterEnd,ctx,x,y,0);
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "VecScatterDestroy"
/*@C
   VecScatterDestroy - Destroys a scatter context created by 
   VecScatterCreate().

   Input Parameter:
.  ctx - the scatter context

.keywords: vector, scatter, context, destroy

.seealso: VecScatterCreate(), VecScatterCopy()
@*/
int VecScatterDestroy( VecScatter ctx )
{
  PetscValidHeaderSpecific(ctx,VEC_SCATTER_COOKIE);
  return (*ctx->destroy)((PetscObject)ctx);
}

#undef __FUNC__  
#define __FUNC__ "VecScatterCopy"
/*@C
   VecScatterCopy - Makes a copy of a scatter context.

   Input Parameter:
.  sctx - the scatter context

   Output Parameter:
.  ctx - the context copy

.keywords: vector, scatter, copy, context

.seealso: VecScatterCreate(), VecScatterDestroy()
@*/
int VecScatterCopy( VecScatter sctx,VecScatter *ctx )
{
  PetscValidHeaderSpecific(sctx,VEC_SCATTER_COOKIE);
  PetscValidPointer(ctx);
  if (!sctx->copy) SETERRQ(PETSC_ERR_SUP,0,"Cannot copy this type");
  PetscHeaderCreate(*ctx,_p_VecScatter,VEC_SCATTER_COOKIE,0,sctx->comm);
  PLogObjectCreate(*ctx);
  PLogObjectMemory(*ctx,sizeof(struct _p_VecScatter));
  (*ctx)->to_n   = sctx->to_n;
  (*ctx)->from_n = sctx->from_n;
  return (*sctx->copy)(sctx,*ctx);
}


/* ------------------------------------------------------------------*/
#undef __FUNC__  
#define __FUNC__ "VecScatterView" /* ADIC Ignore */
/*@
   VecScatterView - Views a vector scatter context.

   Input Parameters:
.  ctx - the scatter context
.  viewer - the viewer for displaying the context

.keywords: vector, scatter, view
@*/
int VecScatterView(VecScatter ctx, Viewer viewer)
{
  PetscValidHeaderSpecific(ctx,VEC_SCATTER_COOKIE);
  if (!viewer) { viewer = VIEWER_STDOUT_SELF;}
  else {PetscValidHeaderSpecific(viewer,VIEWER_COOKIE);}
  if (ctx->view) return (*ctx->view)((PetscObject)ctx,viewer);
  else SETERRQ(PETSC_ERR_SUP,0,"Cannot view this type of scatter context yet");
}

#undef __FUNC__  
#define __FUNC__ "VecScatterRemap" /* ADIC Ignore */
/*@
   VecScatterRemap - Remaps the "from" and "to" indices in a 
   vector scatter context. FOR EXPERTS ONLY!

  Input Parameters:
.   scat - vector scatter context
.   from - remapping for "from" indices (may be PETSC_NULL)
.   to   - remapping for "to" indices (may be PETSC_NULL)

.keywords: Vec, scatter, remap
@*/
int VecScatterRemap(VecScatter scat,int *rto,int *rfrom)
{
  VecScatter_Seq_General *to;
  VecScatter_MPI_General *mto;
  int                    i;

  PetscValidHeaderSpecific(scat,VEC_SCATTER_COOKIE);
  if (rto)   {PetscValidIntPointer(rto);}
  if (rfrom) {PetscValidIntPointer(rfrom);}

  to   = (VecScatter_Seq_General *)scat->todata;
  mto  = (VecScatter_MPI_General *)scat->todata;

  if (to->type == VEC_SCATTER_MPI_TOALL) SETERRQ(PETSC_ERR_ARG_SIZ,0,"Not for to all scatter");

  if (rto) {
    if (to->type == VEC_SCATTER_SEQ_GENERAL) {
      for ( i=0; i<to->n; i++ ) {
        to->slots[i] = rto[to->slots[i]];
      }
    } else if (to->type == VEC_SCATTER_MPI_GENERAL) {
      /* handle off processor parts */
      for ( i=0; i<mto->starts[mto->n]; i++ ) {
        mto->indices[i] = rto[mto->indices[i]];
      }
      /* handle local part */
      to = &mto->local;
      for ( i=0; i<to->n; i++ ) {
        to->slots[i] = rto[to->slots[i]];
      }
    } else SETERRQ(PETSC_ERR_ARG_SIZ,0,"Unable to remap such scatters");
  }
  if (rfrom) {
    SETERRQ(PETSC_ERR_SUP,0,"Unable to remap the FROM in scatters yet");
  }

  /*
     Mark then vector lengths as unknown because we do not know the 
   lengths of the remapped vectors
  */
  scat->from_n = -1;
  scat->to_n   = -1;

  return 0;
}
