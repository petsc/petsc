/*$Id: vscat.c,v 1.173 2001/08/10 03:29:59 bsmith Exp $*/

/*
     Code for creating scatters between vectors. This file 
  includes the code for scattering between sequential vectors and
  some special cases for parallel scatters.
*/

#include "src/vec/is/isimpl.h"
#include "src/vec/vecimpl.h"                     /*I "petscvec.h" I*/

/* Logging support */
int VEC_SCATTER_COOKIE = 0;

/*
     Checks if any indices are less than zero and generates an error
*/
#undef __FUNCT__  
#define __FUNCT__ "VecScatterCheckIndicates_Private"
static int VecScatterCheckIndices_Private(int nmax,int n,int *idx)
{
  int i;

  PetscFunctionBegin;
  for (i=0; i<n; i++) {
    if (idx[i] < 0)     SETERRQ2(1,"Negative index %d at %d location",idx[i],i);
    if (idx[i] >= nmax) SETERRQ3(1,"Index %d at %d location greater than max %d",idx[i],i,nmax);
  }
  PetscFunctionReturn(0);
}

/*
      This is special scatter code for when the entire parallel vector is 
   copied to each processor.

   This code was written by Cameron Cooper, Occidental College, Fall 1995,
   will working at ANL as a SERS student.
*/
#undef __FUNCT__  
#define __FUNCT__ "VecScatterBegin_MPI_ToAll"
int VecScatterBegin_MPI_ToAll(Vec x,Vec y,InsertMode addv,ScatterMode mode,VecScatter ctx)
{ 
  int          ierr,yy_n,xx_n,*range;
  PetscScalar  *xv,*yv;
  PetscMap     map;

  PetscFunctionBegin;
  ierr = VecGetLocalSize(y,&yy_n);CHKERRQ(ierr);
  ierr = VecGetLocalSize(x,&xx_n);CHKERRQ(ierr);
  ierr = VecGetArray(y,&yv);CHKERRQ(ierr);
  if (x != y) {ierr = VecGetArray(x,&xv);CHKERRQ(ierr);}

  if (mode & SCATTER_REVERSE) {
    PetscScalar          *xvt,*xvt2;
    VecScatter_MPI_ToAll *scat = (VecScatter_MPI_ToAll*)ctx->todata;
    int                  i;

    if (addv == INSERT_VALUES) {
      int rstart,rend;
      /* 
         copy the correct part of the local vector into the local storage of 
         the MPI one.  Note: this operation only makes sense if all the local 
         vectors have the same values
      */
      ierr = VecGetOwnershipRange(y,&rstart,&rend);CHKERRQ(ierr);
      ierr = PetscMemcpy(yv,xv+rstart,yy_n*sizeof(PetscScalar));CHKERRQ(ierr);
    } else {
      MPI_Comm comm;
      int      rank;
      ierr = PetscObjectGetComm((PetscObject)y,&comm);CHKERRQ(ierr);
      ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
      if (scat->work1) xvt = scat->work1; 
      else {
        ierr        = PetscMalloc((xx_n+1)*sizeof(PetscScalar),&xvt);CHKERRQ(ierr);
        scat->work1 = xvt;
        PetscLogObjectMemory(ctx,xx_n*sizeof(PetscScalar));
      }
      if (!rank) { /* I am the zeroth processor, values are accumulated here */
        if   (scat->work2) xvt2 = scat->work2; 
        else {
          ierr        = PetscMalloc((xx_n+1)*sizeof(PetscScalar),& xvt2);CHKERRQ(ierr);
          scat->work2 = xvt2;
          PetscLogObjectMemory(ctx,xx_n*sizeof(PetscScalar));
        }
        ierr = VecGetPetscMap(y,&map);CHKERRQ(ierr);
        ierr = PetscMapGetGlobalRange(map,&range);CHKERRQ(ierr);
        ierr = MPI_Gatherv(yv,yy_n,MPIU_SCALAR,xvt2,scat->count,range,MPIU_SCALAR,0,ctx->comm);CHKERRQ(ierr);
#if defined(PETSC_USE_COMPLEX)
        ierr = MPI_Reduce(xv,xvt,2*xx_n,MPIU_REAL,MPI_SUM,0,ctx->comm);CHKERRQ(ierr);
#else
        ierr = MPI_Reduce(xv,xvt,xx_n,MPIU_SCALAR,MPI_SUM,0,ctx->comm);CHKERRQ(ierr);
#endif
        if (addv == ADD_VALUES) {
          for (i=0; i<xx_n; i++) {
	    xvt[i] += xvt2[i];
	  }
#if !defined(PETSC_USE_COMPLEX)
        } else if (addv == MAX_VALUES) {
          for (i=0; i<xx_n; i++) {
	    xvt[i] = PetscMax(xvt[i],xvt2[i]);
	  }
#endif
        } else {SETERRQ(1,"Wrong insert option");}
        ierr = MPI_Scatterv(xvt,scat->count,map->range,MPIU_SCALAR,yv,yy_n,MPIU_SCALAR,0,ctx->comm);CHKERRQ(ierr);
      } else {
        ierr = VecGetPetscMap(y,&map);CHKERRQ(ierr);
        ierr = PetscMapGetGlobalRange(map,&range);CHKERRQ(ierr);
        ierr = MPI_Gatherv(yv,yy_n,MPIU_SCALAR,0, 0,0,MPIU_SCALAR,0,ctx->comm);CHKERRQ(ierr);
#if defined(PETSC_USE_COMPLEX)
        ierr = MPI_Reduce(xv,xvt,2*xx_n,MPIU_REAL,MPI_SUM,0,ctx->comm);CHKERRQ(ierr);
#else
        ierr = MPI_Reduce(xv,xvt,xx_n,MPIU_SCALAR,MPI_SUM,0,ctx->comm);CHKERRQ(ierr);
#endif
        ierr = MPI_Scatterv(0,scat->count,range,MPIU_SCALAR,yv,yy_n,MPIU_SCALAR,0,ctx->comm);CHKERRQ(ierr);
      }
    }
  } else {
    PetscScalar          *yvt;
    VecScatter_MPI_ToAll *scat = (VecScatter_MPI_ToAll*)ctx->todata;
    int                  i;

    ierr = VecGetPetscMap(x,&map);CHKERRQ(ierr);
    ierr = PetscMapGetGlobalRange(map,&range);CHKERRQ(ierr);
    if (addv == INSERT_VALUES) {
      ierr = MPI_Allgatherv(xv,xx_n,MPIU_SCALAR,yv,scat->count,range,MPIU_SCALAR,ctx->comm);CHKERRQ(ierr);
    } else {
      if (scat->work1) yvt = scat->work1; 
      else {
        ierr        = PetscMalloc((yy_n+1)*sizeof(PetscScalar),&yvt);CHKERRQ(ierr);
        scat->work1 = yvt;
        PetscLogObjectMemory(ctx,yy_n*sizeof(PetscScalar));
      }
      ierr = MPI_Allgatherv(xv,xx_n,MPIU_SCALAR,yvt,scat->count,map->range,MPIU_SCALAR,ctx->comm);CHKERRQ(ierr);
      if (addv == ADD_VALUES){
        for (i=0; i<yy_n; i++) {
	  yv[i] += yvt[i];
        }
#if !defined(PETSC_USE_COMPLEX)
      } else if (addv == MAX_VALUES) {
        for (i=0; i<yy_n; i++) {
          yv[i] = PetscMax(yv[i],yvt[i]);
	}
#endif
      } else {SETERRQ(1,"Wrong insert option");}
    }
  }
  ierr = VecRestoreArray(y,&yv);CHKERRQ(ierr);
  if (x != y) {ierr = VecRestoreArray(x,&xv);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

/*
      This is special scatter code for when the entire parallel vector is 
   copied to processor 0.

*/
#undef __FUNCT__  
#define __FUNCT__ "VecScatterBegin_MPI_ToOne"
int VecScatterBegin_MPI_ToOne(Vec x,Vec y,InsertMode addv,ScatterMode mode,VecScatter ctx)
{ 
  int          rank,ierr,yy_n,xx_n,*range;
  PetscScalar  *xv,*yv;
  MPI_Comm     comm;
  PetscMap     map;

  PetscFunctionBegin;
  ierr = VecGetLocalSize(y,&yy_n);CHKERRQ(ierr);
  ierr = VecGetLocalSize(x,&xx_n);CHKERRQ(ierr);
  ierr = VecGetArray(x,&xv);CHKERRQ(ierr);
  ierr = VecGetArray(y,&yv);CHKERRQ(ierr);

  ierr = PetscObjectGetComm((PetscObject)x,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);

  /* --------  Reverse scatter; spread from processor 0 to other processors */
  if (mode & SCATTER_REVERSE) {
    PetscScalar          *yvt;
    VecScatter_MPI_ToAll *scat = (VecScatter_MPI_ToAll*)ctx->todata;
    int                  i;

    ierr = VecGetPetscMap(y,&map);CHKERRQ(ierr);
    ierr = PetscMapGetGlobalRange(map,&range);CHKERRQ(ierr);
    if (addv == INSERT_VALUES) {
      ierr = MPI_Scatterv(xv,scat->count,range,MPIU_SCALAR,yv,yy_n,MPIU_SCALAR,0,ctx->comm);CHKERRQ(ierr);
    } else {
      if (scat->work2) yvt = scat->work2; 
      else {
        ierr        = PetscMalloc((xx_n+1)*sizeof(PetscScalar),&yvt);CHKERRQ(ierr);
        scat->work2 = yvt;
        PetscLogObjectMemory(ctx,xx_n*sizeof(PetscScalar));
      }
      ierr = MPI_Scatterv(xv,scat->count,range,MPIU_SCALAR,yvt,yy_n,MPIU_SCALAR,0,ctx->comm);CHKERRQ(ierr);
      if (addv == ADD_VALUES) {
        for (i=0; i<yy_n; i++) {
	  yv[i] += yvt[i];
        }
#if !defined(PETSC_USE_COMPLEX)
      } else if (addv == MAX_VALUES) {
        for (i=0; i<yy_n; i++) {
          yv[i] = PetscMax(yv[i],yvt[i]);
	}
#endif
      } else {SETERRQ(1,"Wrong insert option");}
    }
  /* ---------  Forward scatter; gather all values onto processor 0 */
  } else { 
    PetscScalar          *yvt  = 0;
    VecScatter_MPI_ToAll *scat = (VecScatter_MPI_ToAll*)ctx->todata;
    int                  i;

    ierr = VecGetPetscMap(x,&map);CHKERRQ(ierr);
    ierr = PetscMapGetGlobalRange(map,&range);CHKERRQ(ierr);
    if (addv == INSERT_VALUES) {
      ierr = MPI_Gatherv(xv,xx_n,MPIU_SCALAR,yv,scat->count,range,MPIU_SCALAR,0,ctx->comm);CHKERRQ(ierr);
    } else {
      if (!rank) {
        if (scat->work1) yvt = scat->work1; 
        else {
          ierr        = PetscMalloc((yy_n+1)*sizeof(PetscScalar),&yvt);CHKERRQ(ierr);
          scat->work1 = yvt;
          PetscLogObjectMemory(ctx,yy_n*sizeof(PetscScalar));
        }
      }
      ierr = MPI_Gatherv(xv,xx_n,MPIU_SCALAR,yvt,scat->count,range,MPIU_SCALAR,0,ctx->comm);CHKERRQ(ierr);
      if (!rank) {
        if (addv == ADD_VALUES) {
          for (i=0; i<yy_n; i++) {
	    yv[i] += yvt[i];
          }
#if !defined(PETSC_USE_COMPLEX)
        } else if (addv == MAX_VALUES) {
          for (i=0; i<yy_n; i++) {
            yv[i] = PetscMax(yv[i],yvt[i]);
          }
#endif
        }  else {SETERRQ(1,"Wrong insert option");}
      }
    }
  }
  ierr = VecRestoreArray(x,&xv);CHKERRQ(ierr);
  ierr = VecRestoreArray(y,&yv);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
       The follow to are used for both VecScatterBegin_MPI_ToAll() and VecScatterBegin_MPI_ToOne()
*/
#undef __FUNCT__  
#define __FUNCT__ "VecScatterDestroy_MPI_ToAll"
int VecScatterDestroy_MPI_ToAll(VecScatter ctx)
{
  VecScatter_MPI_ToAll *scat = (VecScatter_MPI_ToAll*)ctx->todata;
  int ierr;

  PetscFunctionBegin;
  ierr = PetscFree(scat->count);CHKERRQ(ierr);
  if (scat->work1) {ierr = PetscFree(scat->work1);CHKERRQ(ierr);}
  if (scat->work2) {ierr = PetscFree(scat->work2);CHKERRQ(ierr);}
  ierr = PetscFree(ctx->todata);CHKERRQ(ierr);
  PetscLogObjectDestroy(ctx);
  PetscHeaderDestroy(ctx);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecScatterCopy_MPI_ToAll"
int VecScatterCopy_MPI_ToAll(VecScatter in,VecScatter out)
{
  VecScatter_MPI_ToAll *in_to = (VecScatter_MPI_ToAll*)in->todata,*sto;
  int                  size,i,ierr;

  PetscFunctionBegin;
  out->postrecvs      = 0;
  out->begin          = in->begin;
  out->end            = in->end;
  out->copy           = in->copy;
  out->destroy        = in->destroy;
  out->view           = in->view;

  ierr      = PetscNew(VecScatter_MPI_ToAll,&sto);CHKERRQ(ierr);
  sto->type = in_to->type;

  ierr = MPI_Comm_size(out->comm,&size);CHKERRQ(ierr);
  ierr = PetscMalloc(size*sizeof(int),&sto->count);CHKERRQ(ierr);
  for (i=0; i<size; i++) {
    sto->count[i] = in_to->count[i];
  }
  sto->work1         = 0;
  sto->work2         = 0;
  PetscLogObjectMemory(out,sizeof(VecScatter_MPI_ToAll)+size*sizeof(int));
  out->todata        = (void*)sto; 
  out->fromdata      = (void*)0;
  PetscFunctionReturn(0);
}

/* --------------------------------------------------------------------------------------*/
/* 
        Scatter: sequential general to sequential general 
*/
#undef __FUNCT__  
#define __FUNCT__ "VecScatterBegin_SGtoSG"
int VecScatterBegin_SGtoSG(Vec x,Vec y,InsertMode addv,ScatterMode mode,VecScatter ctx)
{
  VecScatter_Seq_General *gen_to = (VecScatter_Seq_General*)ctx->todata;
  VecScatter_Seq_General *gen_from = (VecScatter_Seq_General*)ctx->fromdata;
  int                    i,n = gen_from->n,*fslots,*tslots,ierr;
  PetscScalar            *xv,*yv;
  
  PetscFunctionBegin;
  ierr = VecGetArray(x,&xv);CHKERRQ(ierr);
  if (x != y) {ierr = VecGetArray(y,&yv);CHKERRQ(ierr);} else {yv = xv;}
  if (mode & SCATTER_REVERSE){
    gen_to   = (VecScatter_Seq_General*)ctx->fromdata;
    gen_from = (VecScatter_Seq_General*)ctx->todata;
  }
  fslots = gen_from->slots;
  tslots = gen_to->slots;

  if (addv == INSERT_VALUES) {
    for (i=0; i<n; i++) {yv[tslots[i]] = xv[fslots[i]];}
  } else if (addv == ADD_VALUES) {
    for (i=0; i<n; i++) {yv[tslots[i]] += xv[fslots[i]];}
#if !defined(PETSC_USE_COMPLEX)
  } else  if (addv == MAX_VALUES) {
    for (i=0; i<n; i++) {yv[tslots[i]] = PetscMax(yv[tslots[i]],xv[fslots[i]]);}
#endif
  } else {SETERRQ(1,"Wrong insert option");}
  ierr = VecRestoreArray(x,&xv);CHKERRQ(ierr);
  if (x != y) {ierr = VecRestoreArray(y,&yv);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

/* 
    Scatter: sequential general to sequential stride 1 
*/
#undef __FUNCT__  
#define __FUNCT__ "VecScatterBegin_SGtoSS_Stride1"
int VecScatterBegin_SGtoSS_Stride1(Vec x,Vec y,InsertMode addv,ScatterMode mode,VecScatter ctx)
{
  VecScatter_Seq_Stride  *gen_to   = (VecScatter_Seq_Stride*)ctx->todata;
  VecScatter_Seq_General *gen_from = (VecScatter_Seq_General*)ctx->fromdata;
  int                    i,n = gen_from->n,*fslots = gen_from->slots;
  int                    first = gen_to->first,ierr;
  PetscScalar            *xv,*yv;
  
  PetscFunctionBegin;
  ierr = VecGetArray(x,&xv);CHKERRQ(ierr);
  if (x != y) {ierr = VecGetArray(y,&yv);CHKERRQ(ierr);} else {yv = xv;}
  if (mode & SCATTER_REVERSE){
    xv += first;
    if (addv == INSERT_VALUES) {
      for (i=0; i<n; i++) {yv[fslots[i]] = xv[i];}
    } else  if (addv == ADD_VALUES) {
      for (i=0; i<n; i++) {yv[fslots[i]] += xv[i];}
#if !defined(PETSC_USE_COMPLEX)
    } else  if (addv == MAX_VALUES) {
      for (i=0; i<n; i++) {yv[fslots[i]] = PetscMax(yv[fslots[i]],xv[i]);}
#endif
    } else {SETERRQ(1,"Wrong insert option");}
  } else {
    yv += first;
    if (addv == INSERT_VALUES) {
      for (i=0; i<n; i++) {yv[i] = xv[fslots[i]];}
    } else  if (addv == ADD_VALUES) {
      for (i=0; i<n; i++) {yv[i] += xv[fslots[i]];}
#if !defined(PETSC_USE_COMPLEX)
    } else if (addv == MAX_VALUES) {
      for (i=0; i<n; i++) {yv[i] = PetscMax(yv[i],xv[fslots[i]]);}
#endif
    } else {SETERRQ(1,"Wrong insert option");}
  }
  ierr = VecRestoreArray(x,&xv);CHKERRQ(ierr);
  if (x != y) {ierr = VecRestoreArray(y,&yv);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

/* 
   Scatter: sequential general to sequential stride 
*/
#undef __FUNCT__  
#define __FUNCT__ "VecScatterBegin_SGtoSS"
int VecScatterBegin_SGtoSS(Vec x,Vec y,InsertMode addv,ScatterMode mode,VecScatter ctx)
{
  VecScatter_Seq_Stride  *gen_to   = (VecScatter_Seq_Stride*)ctx->todata;
  VecScatter_Seq_General *gen_from = (VecScatter_Seq_General*)ctx->fromdata;
  int                    i,n = gen_from->n,*fslots = gen_from->slots;
  int                    first = gen_to->first,step = gen_to->step,ierr;
  PetscScalar            *xv,*yv;
  
  PetscFunctionBegin;
  ierr = VecGetArray(x,&xv);CHKERRQ(ierr);
  if (x != y) {ierr = VecGetArray(y,&yv);CHKERRQ(ierr);} else {yv = xv;} 

  if (mode & SCATTER_REVERSE){
    if (addv == INSERT_VALUES) {
      for (i=0; i<n; i++) {yv[fslots[i]] = xv[first + i*step];}
    } else if (addv == ADD_VALUES) {
      for (i=0; i<n; i++) {yv[fslots[i]] += xv[first + i*step];}
#if !defined(PETSC_USE_COMPLEX)
    } else if (addv == MAX_VALUES) {
      for (i=0; i<n; i++) {yv[fslots[i]] = PetscMax(yv[fslots[i]],xv[first + i*step]);}
#endif
    } else {SETERRQ(1,"Wrong insert option");}
  } else {
    if (addv == INSERT_VALUES) {
      for (i=0; i<n; i++) {yv[first + i*step] = xv[fslots[i]];}
    } else if (addv == ADD_VALUES) {
      for (i=0; i<n; i++) {yv[first + i*step] += xv[fslots[i]];}
#if !defined(PETSC_USE_COMPLEX)
    } else if (addv == MAX_VALUES) {
      for (i=0; i<n; i++) {yv[first + i*step] = PetscMax(yv[first + i*step],xv[fslots[i]]);}
#endif
    } else {SETERRQ(1,"Wrong insert option");}
  }
  ierr = VecRestoreArray(x,&xv);CHKERRQ(ierr);
  if (x != y) {ierr = VecRestoreArray(y,&yv);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

/* 
    Scatter: sequential stride 1 to sequential general 
*/
#undef __FUNCT__  
#define __FUNCT__ "VecScatterBegin_SStoSG_Stride1"
int VecScatterBegin_SStoSG_Stride1(Vec x,Vec y,InsertMode addv,ScatterMode mode,VecScatter ctx)
{
  VecScatter_Seq_Stride  *gen_from = (VecScatter_Seq_Stride*)ctx->fromdata;
  VecScatter_Seq_General *gen_to   = (VecScatter_Seq_General*)ctx->todata;
  int                    i,n = gen_from->n,*fslots = gen_to->slots;
  int                    first = gen_from->first,ierr;
  PetscScalar            *xv,*yv;
  
  PetscFunctionBegin;
  ierr = VecGetArray(x,&xv);CHKERRQ(ierr);
  if (x != y) {ierr = VecGetArray(y,&yv);CHKERRQ(ierr);} else {yv = xv;}

  if (mode & SCATTER_REVERSE){
    yv += first;
    if (addv == INSERT_VALUES) {
      for (i=0; i<n; i++) {yv[i] = xv[fslots[i]];}
    } else  if (addv == ADD_VALUES) {
      for (i=0; i<n; i++) {yv[i] += xv[fslots[i]];}
#if !defined(PETSC_USE_COMPLEX)
    } else  if (addv == MAX_VALUES) {
      for (i=0; i<n; i++) {yv[i] = PetscMax(yv[i],xv[fslots[i]]);}
#endif
    } else {SETERRQ(1,"Wrong insert option");}
  } else {
    xv += first;
    if (addv == INSERT_VALUES) {
      for (i=0; i<n; i++) {yv[fslots[i]] = xv[i];}
    } else  if (addv == ADD_VALUES) {
      for (i=0; i<n; i++) {yv[fslots[i]] += xv[i];}
#if !defined(PETSC_USE_COMPLEX)
    } else  if (addv == MAX_VALUES) {
      for (i=0; i<n; i++) {yv[fslots[i]] = PetscMax(yv[fslots[i]],xv[i]);}
#endif
    } else {SETERRQ(1,"Wrong insert option");}
  } 
  ierr = VecRestoreArray(x,&xv);CHKERRQ(ierr);
  if (x != y) {ierr = VecRestoreArray(y,&yv);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecScatterBegin_SStoSG"
/* 
   Scatter: sequential stride to sequential general 
*/
int VecScatterBegin_SStoSG(Vec x,Vec y,InsertMode addv,ScatterMode mode,VecScatter ctx)
{
  VecScatter_Seq_Stride  *gen_from = (VecScatter_Seq_Stride*)ctx->fromdata;
  VecScatter_Seq_General *gen_to   = (VecScatter_Seq_General*)ctx->todata;
  int                    i,n = gen_from->n,*fslots = gen_to->slots;
  int                    first = gen_from->first,step = gen_from->step,ierr;
  PetscScalar            *xv,*yv;
  
  PetscFunctionBegin;
  ierr = VecGetArray(x,&xv);CHKERRQ(ierr);
  if (x != y) {ierr = VecGetArray(y,&yv);CHKERRQ(ierr);} else {yv = xv;}

  if (mode & SCATTER_REVERSE){
    if (addv == INSERT_VALUES) {
      for (i=0; i<n; i++) {yv[first + i*step] = xv[fslots[i]];}
    } else  if (addv == ADD_VALUES) {
      for (i=0; i<n; i++) {yv[first + i*step] += xv[fslots[i]];}
#if !defined(PETSC_USE_COMPLEX)
    } else  if (addv == MAX_VALUES) {
      for (i=0; i<n; i++) {yv[first + i*step] = PetscMax(yv[first + i*step],xv[fslots[i]]);}
#endif
    } else {SETERRQ(1,"Wrong insert option");}
  } else {
    if (addv == INSERT_VALUES) {
      for (i=0; i<n; i++) {yv[fslots[i]] = xv[first + i*step];}
    } else  if (addv == ADD_VALUES) {
      for (i=0; i<n; i++) {yv[fslots[i]] += xv[first + i*step];}
#if !defined(PETSC_USE_COMPLEX)
    } else  if (addv == MAX_VALUES) {
      for (i=0; i<n; i++) {yv[fslots[i]] = PetscMax(yv[fslots[i]],xv[first + i*step]);}
#endif
    } else {SETERRQ(1,"Wrong insert option");}
  }
  ierr = VecRestoreArray(x,&xv);CHKERRQ(ierr);
  if (x != y) {ierr = VecRestoreArray(y,&yv);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

/* 
     Scatter: sequential stride to sequential stride 
*/
#undef __FUNCT__  
#define __FUNCT__ "VecScatterBegin_SStoSS"
int VecScatterBegin_SStoSS(Vec x,Vec y,InsertMode addv,ScatterMode mode,VecScatter ctx)
{
  VecScatter_Seq_Stride *gen_to   = (VecScatter_Seq_Stride*)ctx->todata;
  VecScatter_Seq_Stride *gen_from = (VecScatter_Seq_Stride*)ctx->fromdata;
  int                   i,n = gen_from->n,to_first = gen_to->first,to_step = gen_to->step;
  int                   from_first = gen_from->first,from_step = gen_from->step,ierr;
  PetscScalar           *xv,*yv;
  
  PetscFunctionBegin;
  ierr = VecGetArray(x,&xv);CHKERRQ(ierr);
  if (x != y) {ierr = VecGetArray(y,&yv);CHKERRQ(ierr);} else {yv = xv;}

  if (mode & SCATTER_REVERSE){
    from_first = gen_to->first; 
    to_first   = gen_from->first;
    from_step  = gen_to->step; 
    to_step    = gen_from->step;
  }

  if (addv == INSERT_VALUES) {
    if (to_step == 1 && from_step == 1) {
      ierr = PetscMemcpy(yv+to_first,xv+from_first,n*sizeof(PetscScalar));CHKERRQ(ierr);
    } else  {
      for (i=0; i<n; i++) {
        yv[to_first + i*to_step] = xv[from_first+i*from_step];
      }
    }
  } else if (addv == ADD_VALUES) {
    if (to_step == 1 && from_step == 1) {
      yv += to_first; xv += from_first;
      for (i=0; i<n; i++) {
        yv[i] += xv[i];
      }
    } else {
      for (i=0; i<n; i++) {
        yv[to_first + i*to_step] += xv[from_first+i*from_step];
      }
    }
#if !defined(PETSC_USE_COMPLEX)
  } else if (addv == MAX_VALUES) {
    if (to_step == 1 && from_step == 1) {
      yv += to_first; xv += from_first;
      for (i=0; i<n; i++) {
        yv[i] = PetscMax(yv[i],xv[i]);
      }
    } else {
      for (i=0; i<n; i++) {
        yv[to_first + i*to_step] = PetscMax(yv[to_first + i*to_step],xv[from_first+i*from_step]);
      }
    }
#endif
  } else {SETERRQ(1,"Wrong insert option");}
  ierr = VecRestoreArray(x,&xv);CHKERRQ(ierr);
  if (x != y) {ierr = VecRestoreArray(y,&yv);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

/* --------------------------------------------------------------------------*/


#undef __FUNCT__  
#define __FUNCT__ "VecScatterCopy_SGToSG"
int VecScatterCopy_SGToSG(VecScatter in,VecScatter out)
{
  int                    ierr;
  VecScatter_Seq_General *in_to   = (VecScatter_Seq_General*)in->todata,*out_to;
  VecScatter_Seq_General *in_from = (VecScatter_Seq_General*)in->fromdata,*out_from;
  
  PetscFunctionBegin;
  out->postrecvs     = 0;
  out->begin         = in->begin;
  out->end           = in->end;
  out->copy          = in->copy;
  out->destroy       = in->destroy;
  out->view          = in->view;

  ierr                           = PetscMalloc(in_to->n*sizeof(int)+sizeof(VecScatter_Seq_General),&out_to);CHKERRQ(ierr);
  out_to->n                      = in_to->n; 
  out_to->type                   = in_to->type;
  out_to->nonmatching_computed   = PETSC_FALSE;
  out_to->slots_nonmatching      = 0;
  out_to->is_copy                = PETSC_FALSE;
  out_to->slots                  = (int*)(out_to + 1);
  ierr = PetscMemcpy(out_to->slots,in_to->slots,(out_to->n)*sizeof(int));CHKERRQ(ierr);

  ierr                           = PetscMalloc(in_from->n*sizeof(int)+sizeof(VecScatter_Seq_General),&out_from);CHKERRQ(ierr);
  out_from->n                    = in_from->n; 
  out_from->type                 = in_from->type;
  out_from->nonmatching_computed = PETSC_FALSE;
  out_from->slots_nonmatching    = 0;
  out_from->is_copy              = PETSC_FALSE;
  out_from->slots                = (int*)(out_from + 1);
  ierr = PetscMemcpy(out_from->slots,in_from->slots,(out_from->n)*sizeof(int));CHKERRQ(ierr);

  PetscLogObjectMemory(out,2*sizeof(VecScatter_Seq_General)+(out_from->n+out_to->n)*sizeof(int));
  out->todata     = (void*)out_to; 
  out->fromdata   = (void*)out_from;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecScatterDestroy_SGtoSG"
int VecScatterDestroy_SGtoSG(VecScatter ctx)
{
  int ierr;

  PetscFunctionBegin;
  ierr = PetscFree(ctx->todata);CHKERRQ(ierr);
  ierr = PetscFree(ctx->fromdata);CHKERRQ(ierr);
  PetscLogObjectDestroy(ctx);
  PetscHeaderDestroy(ctx);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecScatterCopy_SGToStride"
int VecScatterCopy_SGToStride(VecScatter in,VecScatter out)
{
  int ierr;
  VecScatter_Seq_Stride  *in_to   = (VecScatter_Seq_Stride*)in->todata,*out_to;
  VecScatter_Seq_General *in_from = (VecScatter_Seq_General*)in->fromdata,*out_from;
  
  PetscFunctionBegin;
  out->postrecvs     = 0;
  out->begin         = in->begin;
  out->end           = in->end;
  out->copy          = in->copy;
  out->destroy       = in->destroy;
  out->view          = in->view;

  ierr            = PetscNew(VecScatter_Seq_Stride,&out_to);CHKERRQ(ierr);
  out_to->n       = in_to->n; 
  out_to->type    = in_to->type;
  out_to->first   = in_to->first; 
  out_to->step    = in_to->step;
  out_to->type    = in_to->type;

  ierr                           = PetscMalloc(in_from->n*sizeof(int)+sizeof(VecScatter_Seq_General),&out_from);CHKERRQ(ierr);
  out_from->n                    = in_from->n; 
  out_from->type                 = in_from->type;
  out_from->nonmatching_computed = PETSC_FALSE;
  out_from->slots_nonmatching    = 0;
  out_from->is_copy              = PETSC_FALSE;
  out_from->slots                = (int*)(out_from + 1);
  ierr = PetscMemcpy(out_from->slots,in_from->slots,(out_from->n)*sizeof(int));CHKERRQ(ierr);

  PetscLogObjectMemory(out,sizeof(VecScatter_Seq_General)+sizeof(VecScatter_Seq_Stride)+in_from->n*sizeof(int));
  out->todata     = (void*)out_to; 
  out->fromdata   = (void*)out_from;
  PetscFunctionReturn(0);
}

/* --------------------------------------------------------------------------*/
/* 
    Scatter: parallel to sequential vector, sequential strides for both. 
*/
#undef __FUNCT__  
#define __FUNCT__ "VecScatterCopy_PStoSS"
int VecScatterCopy_PStoSS(VecScatter in,VecScatter out)
{
  VecScatter_Seq_Stride *in_to   = (VecScatter_Seq_Stride*)in->todata,*out_to;
  VecScatter_Seq_Stride *in_from = (VecScatter_Seq_Stride*)in->fromdata,*out_from;
  int                   ierr;

  PetscFunctionBegin;
  out->postrecvs  = 0;
  out->begin      = in->begin;
  out->end        = in->end;
  out->copy       = in->copy;
  out->destroy    = in->destroy;
  out->view       = in->view;

  ierr            = PetscNew(VecScatter_Seq_Stride,&out_to);CHKERRQ(ierr);
  out_to->n       = in_to->n; 
  out_to->type    = in_to->type;
  out_to->first   = in_to->first; 
  out_to->step    = in_to->step;
  out_to->type    = in_to->type;
  ierr            = PetscNew(VecScatter_Seq_Stride,&out_from);CHKERRQ(ierr);
  PetscLogObjectMemory(out,2*sizeof(VecScatter_Seq_Stride));
  out_from->n     = in_from->n; 
  out_from->type  = in_from->type;
  out_from->first = in_from->first; 
  out_from->step  = in_from->step;
  out_from->type  = in_from->type;
  out->todata     = (void*)out_to; 
  out->fromdata   = (void*)out_from;
  PetscFunctionReturn(0);
}

EXTERN int VecScatterCreate_PtoS(int,int *,int,int *,Vec,Vec,int,VecScatter);
EXTERN int VecScatterCreate_PtoP(int,int *,int,int *,Vec,Vec,VecScatter);
EXTERN int VecScatterCreate_StoP(int,int *,int,int *,Vec,VecScatter);

/* =======================================================================*/
#define VEC_SEQ_ID 0
#define VEC_MPI_ID 1

#undef __FUNCT__  
#define __FUNCT__ "VecScatterCreate"
/*@C
   VecScatterCreate - Creates a vector scatter context.

   Collective on Vec

   Input Parameters:
+  xin - a vector that defines the shape (parallel data layout of the vector)
         of vectors from which we scatter
.  yin - a vector that defines the shape (parallel data layout of the vector)
         of vectors to which we scatter
.  ix - the indices of xin to scatter
-  iy - the indices of yin to hold results

   Output Parameter:
.  newctx - location to store the new scatter context

   Options Database Keys:
+  -vecscatter_merge     - Merges scatter send and receive (may offer better performance with some MPIs)
.  -vecscatter_ssend     - Uses MPI_Ssend_init() instead of MPI_Send_init() (may offer better performance with some MPIs)
.  -vecscatter_sendfirst - Posts sends before receives (may offer better performance with some MPIs)
.  -vecscatter_rr        - use ready receiver mode for MPI sends in scatters (rarely used)
-  -vecscatter_packtogether - Pack all messages before sending, receive all messages before unpacking
    Level: intermediate

  Notes:
   In calls to VecScatter() you can use different vectors than the xin and 
   yin you used above; BUT they must have the same parallel data layout, for example,
   they could be obtained from VecDuplicate().
   A VecScatter context CANNOT be used in two or more simultaneous scatters;
   that is you cannot call a second VecScatterBegin() with the same scatter
   context until the VecScatterEnd() has been called on the first VecScatterBegin().
   In this case a separate VecScatter is needed for each concurrent scatter.

   Concepts: scatter^between vectors
   Concepts: gather^between vectors

.seealso: VecScatterDestroy()
@*/
int VecScatterCreate(Vec xin,IS ix,Vec yin,IS iy,VecScatter *newctx)
{
  VecScatter ctx;
  int        len,size,cando,totalv,ierr,*range,xin_type = VEC_SEQ_ID,yin_type = VEC_SEQ_ID; 
  PetscTruth flag;
  MPI_Comm   comm,ycomm;
  PetscTruth ixblock,iyblock,iystride,islocal;
  IS         tix = 0,tiy = 0;

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
  if (size > 1) {xin_type = VEC_MPI_ID;}

  ierr = PetscObjectGetComm((PetscObject)yin,&ycomm);CHKERRQ(ierr);
  ierr = MPI_Comm_size(ycomm,&size);CHKERRQ(ierr);
  if (size > 1) {comm = ycomm; yin_type = VEC_MPI_ID;}
  
  /* generate the Scatter context */
  PetscHeaderCreate(ctx,_p_VecScatter,int,VEC_SCATTER_COOKIE,0,"VecScatter",comm,VecScatterDestroy,VecScatterView);
  PetscLogObjectCreate(ctx);
  PetscLogObjectMemory(ctx,sizeof(struct _p_VecScatter));
  ctx->inuse               = PETSC_FALSE;

  ctx->beginandendtogether = PETSC_FALSE;
  ierr = PetscOptionsHasName(PETSC_NULL,"-vecscatter_merge",&ctx->beginandendtogether);CHKERRQ(ierr);
  if (ctx->beginandendtogether) {
    PetscLogInfo(ctx,"VecScatterCreate:Using combined (merged) vector scatter begin and end\n");
  }
  ierr = PetscOptionsHasName(PETSC_NULL,"-vecscatter_packtogether",&ctx->packtogether);CHKERRQ(ierr);
  if (ctx->packtogether) {
    PetscLogInfo(ctx,"VecScatterCreate:Pack all messages before sending\n");
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
    int bign;
    ierr = VecGetSize(xin,&bign);CHKERRQ(ierr);
    ierr = ISCreateStride(comm,bign,0,1,&ix);CHKERRQ(ierr);
    tix  = ix;
  } else if (!ix) {
    SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"iy not given, but not Seq or MPI vector");
  }

  if (!iy && yin_type == VEC_SEQ_ID) {
    ierr = ISCreateStride(comm,ctx->to_n,0,1,&iy);CHKERRQ(ierr);
    tiy  = iy;
  } else if (!iy && yin_type == VEC_MPI_ID) {
    int bign;
    ierr = VecGetSize(yin,&bign);CHKERRQ(ierr);
    ierr = ISCreateStride(comm,bign,0,1,&iy);CHKERRQ(ierr);
    tiy  = iy;
  } else if (!iy) {
    SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"iy not given, but not Seq or MPI vector");
  }

  /* ===========================================================================================================
        Check for special cases
     ==========================================================================================================*/
  /* ---------------------------------------------------------------------------*/
  if (xin_type == VEC_SEQ_ID && yin_type == VEC_SEQ_ID) {
    if (ix->type == IS_GENERAL && iy->type == IS_GENERAL){
      int                    nx,ny,*idx,*idy;
      VecScatter_Seq_General *to,*from;

      ierr = ISGetLocalSize(ix,&nx);CHKERRQ(ierr);
      ierr = ISGetLocalSize(iy,&ny);CHKERRQ(ierr);
      if (nx != ny) SETERRQ(PETSC_ERR_ARG_SIZ,"Local scatter sizes don't match");
      ierr = ISGetIndices(ix,&idx);CHKERRQ(ierr);
      ierr = ISGetIndices(iy,&idy);CHKERRQ(ierr);
      len  = sizeof(VecScatter_Seq_General) + nx*sizeof(int);
      ierr = PetscMalloc(len,&to);CHKERRQ(ierr);
      PetscLogObjectMemory(ctx,2*len);
      to->slots         = (int*)(to + 1); 
      to->n             = nx; 
      ierr = VecScatterCheckIndices_Private(ctx->to_n,ny,idy);CHKERRQ(ierr);
      ierr = PetscMemcpy(to->slots,idy,nx*sizeof(int));CHKERRQ(ierr);
      ierr = PetscMalloc(len,&from);CHKERRQ(ierr);
      from->slots       = (int*)(from + 1);
      from->n           = nx; 
      ierr = VecScatterCheckIndices_Private(ctx->from_n,nx,idx);CHKERRQ(ierr);
      ierr =  PetscMemcpy(from->slots,idx,nx*sizeof(int));CHKERRQ(ierr);
      to->type          = VEC_SCATTER_SEQ_GENERAL; 
      from->type        = VEC_SCATTER_SEQ_GENERAL; 
      ctx->todata       = (void*)to; 
      ctx->fromdata     = (void*)from;
      ctx->postrecvs    = 0;
      ctx->begin        = VecScatterBegin_SGtoSG; 
      ctx->end          = 0; 
      ctx->destroy      = VecScatterDestroy_SGtoSG;
      ctx->copy         = VecScatterCopy_SGToSG;
      PetscLogInfo(xin,"VecScatterCreate:Special case: sequential vector general scatter\n");
      goto functionend;
    } else if (ix->type == IS_STRIDE &&  iy->type == IS_STRIDE){
      int                    nx,ny,to_first,to_step,from_first,from_step;
      VecScatter_Seq_Stride  *from8,*to8;

      ierr = ISGetLocalSize(ix,&nx);CHKERRQ(ierr); 
      ierr = ISGetLocalSize(iy,&ny);CHKERRQ(ierr); 
      if (nx != ny) SETERRQ(PETSC_ERR_ARG_SIZ,"Local scatter sizes don't match");
      ierr = ISStrideGetInfo(iy,&to_first,&to_step);CHKERRQ(ierr);
      ierr = ISStrideGetInfo(ix,&from_first,&from_step);CHKERRQ(ierr);
      ierr               = PetscNew(VecScatter_Seq_Stride,&to8);CHKERRQ(ierr);
      to8->n             = nx; 
      to8->first         = to_first; 
      to8->step          = to_step;
      ierr               = PetscNew(VecScatter_Seq_Stride,&from8);CHKERRQ(ierr);
      PetscLogObjectMemory(ctx,2*sizeof(VecScatter_Seq_Stride));
      from8->n           = nx;
      from8->first       = from_first; 
      from8->step        = from_step;
      to8->type          = VEC_SCATTER_SEQ_STRIDE; 
      from8->type        = VEC_SCATTER_SEQ_STRIDE; 
      ctx->todata       = (void*)to8; 
      ctx->fromdata     = (void*)from8;
      ctx->postrecvs    = 0;
      ctx->begin        = VecScatterBegin_SStoSS; 
      ctx->end          = 0; 
      ctx->destroy      = VecScatterDestroy_SGtoSG;
      ctx->copy         = 0;
      PetscLogInfo(xin,"VecScatterCreate:Special case: sequential vector stride to stride\n");
      goto functionend; 
    } else if (ix->type == IS_GENERAL && iy->type == IS_STRIDE){
      int                    nx,ny,*idx,first,step;
      VecScatter_Seq_General *from9;
      VecScatter_Seq_Stride  *to9;

      ierr = ISGetLocalSize(ix,&nx);CHKERRQ(ierr);
      ierr = ISGetIndices(ix,&idx);CHKERRQ(ierr);
      ierr = ISGetLocalSize(iy,&ny);CHKERRQ(ierr);
      ierr = ISStrideGetInfo(iy,&first,&step);CHKERRQ(ierr);
      if (nx != ny) SETERRQ(PETSC_ERR_ARG_SIZ,"Local scatter sizes don't match");
      ierr           = PetscNew(VecScatter_Seq_Stride,&to9);CHKERRQ(ierr);
      to9->n         = nx; 
      to9->first     = first; 
      to9->step      = step;
      len            = sizeof(VecScatter_Seq_General) + nx*sizeof(int);
      ierr = PetscMalloc(len,&from9);CHKERRQ(ierr);
      PetscLogObjectMemory(ctx,len + sizeof(VecScatter_Seq_Stride));
      from9->slots   = (int*)(from9 + 1); 
      from9->n       = nx; 
      ierr           = VecScatterCheckIndices_Private(ctx->from_n,nx,idx);CHKERRQ(ierr);
      ierr           = PetscMemcpy(from9->slots,idx,nx*sizeof(int));CHKERRQ(ierr);
      ctx->todata    = (void*)to9; ctx->fromdata = (void*)from9;
      ctx->postrecvs = 0;
      if (step == 1)  ctx->begin = VecScatterBegin_SGtoSS_Stride1;
      else            ctx->begin = VecScatterBegin_SGtoSS;
      ctx->destroy = VecScatterDestroy_SGtoSG;
      ctx->end     = 0; 
      ctx->copy    = VecScatterCopy_SGToStride;
      to9->type    = VEC_SCATTER_SEQ_STRIDE; 
      from9->type  = VEC_SCATTER_SEQ_GENERAL;
      PetscLogInfo(xin,"VecScatterCreate:Special case: sequential vector general to stride\n");
      goto functionend;
    } else if (ix->type == IS_STRIDE && iy->type == IS_GENERAL){
      int                    nx,ny,*idy,first,step;
      VecScatter_Seq_General *to10;
      VecScatter_Seq_Stride  *from10;

      ierr = ISGetLocalSize(ix,&nx);CHKERRQ(ierr); 
      ierr = ISGetIndices(iy,&idy);CHKERRQ(ierr);
      ierr = ISGetLocalSize(iy,&ny);CHKERRQ(ierr); 
      ierr = ISStrideGetInfo(ix,&first,&step);CHKERRQ(ierr);
      if (nx != ny) SETERRQ(PETSC_ERR_ARG_SIZ,"Local scatter sizes don't match");
      ierr              = PetscNew(VecScatter_Seq_Stride,&from10);CHKERRQ(ierr);
      from10->n         = nx; 
      from10->first     = first; 
      from10->step      = step;
      len               = sizeof(VecScatter_Seq_General) + nx*sizeof(int);
      ierr = PetscMalloc(len,&to10);CHKERRQ(ierr);
      PetscLogObjectMemory(ctx,len + sizeof(VecScatter_Seq_Stride));
      to10->slots       = (int*)(to10 + 1); 
      to10->n           = nx; 
      ierr = VecScatterCheckIndices_Private(ctx->to_n,ny,idy);CHKERRQ(ierr);
      ierr = PetscMemcpy(to10->slots,idy,nx*sizeof(int));CHKERRQ(ierr);
      ctx->todata     = (void*)to10; 
      ctx->fromdata   = (void*)from10;
      ctx->postrecvs  = 0;
      if (step == 1) ctx->begin = VecScatterBegin_SStoSG_Stride1; 
      else           ctx->begin = VecScatterBegin_SStoSG; 
      ctx->destroy    = VecScatterDestroy_SGtoSG;
      ctx->end        = 0; 
      ctx->copy       = 0;
      to10->type      = VEC_SCATTER_SEQ_GENERAL; 
      from10->type    = VEC_SCATTER_SEQ_STRIDE; 
      PetscLogInfo(xin,"VecScatterCreate:Special case: sequential vector stride to general\n");
      goto functionend;
    } else {
      int                    nx,ny,*idx,*idy;
      VecScatter_Seq_General *to11,*from11;
      PetscTruth             idnx,idny;

      ierr = ISGetLocalSize(ix,&nx);CHKERRQ(ierr);
      ierr = ISGetLocalSize(iy,&ny);CHKERRQ(ierr);
      if (nx != ny) SETERRQ(PETSC_ERR_ARG_SIZ,"Local scatter sizes don't match");

      ierr = ISIdentity(ix,&idnx);CHKERRQ(ierr);
      ierr = ISIdentity(iy,&idny);CHKERRQ(ierr);
      if (idnx && idny) {
        VecScatter_Seq_Stride *to13,*from13;
        ierr              = PetscNew(VecScatter_Seq_Stride,&to13);CHKERRQ(ierr);
        to13->n           = nx; 
        to13->first       = 0;
        to13->step        = 1;
        ierr              = PetscNew(VecScatter_Seq_Stride,&from13);CHKERRQ(ierr);
        PetscLogObjectMemory(ctx,2*sizeof(VecScatter_Seq_Stride));
        from13->n         = nx; 
        from13->first     = 0;
        from13->step      = 1;
        to13->type        = VEC_SCATTER_SEQ_STRIDE; 
        from13->type      = VEC_SCATTER_SEQ_STRIDE;
        ctx->todata       = (void*)to13;
        ctx->fromdata     = (void*)from13;
        ctx->postrecvs    = 0;
        ctx->begin        = VecScatterBegin_SStoSS; 
        ctx->end          = 0;  
        ctx->destroy      = VecScatterDestroy_SGtoSG;
        ctx->copy         = 0;
        PetscLogInfo(xin,"VecScatterCreate:Special case: sequential copy\n");
        goto functionend;
      }

      ierr = ISGetIndices(iy,&idy);CHKERRQ(ierr);
      ierr = ISGetIndices(ix,&idx);CHKERRQ(ierr);
      len               = sizeof(VecScatter_Seq_General) + nx*sizeof(int);
      ierr = PetscMalloc(len,&to11);CHKERRQ(ierr);
      PetscLogObjectMemory(ctx,2*len);
      to11->slots       = (int*)(to11 + 1); 
      to11->n           = nx; 
      ierr = VecScatterCheckIndices_Private(ctx->to_n,ny,idy);CHKERRQ(ierr);
      ierr =  PetscMemcpy(to11->slots,idy,nx*sizeof(int));CHKERRQ(ierr);
      ierr = PetscMalloc(len,&from11);CHKERRQ(ierr);
      from11->slots     = (int*)(from11 + 1);
      from11->n         = nx; 
      ierr = VecScatterCheckIndices_Private(ctx->from_n,nx,idx);CHKERRQ(ierr);
      ierr = PetscMemcpy(from11->slots,idx,nx*sizeof(int));CHKERRQ(ierr);
      to11->type        = VEC_SCATTER_SEQ_GENERAL; 
      from11->type      = VEC_SCATTER_SEQ_GENERAL; 
      ctx->todata       = (void*)to11; 
      ctx->fromdata     = (void*)from11;
      ctx->postrecvs    = 0;
      ctx->begin        = VecScatterBegin_SGtoSG; 
      ctx->end          = 0; 
      ctx->destroy      = VecScatterDestroy_SGtoSG;
      ctx->copy         = VecScatterCopy_SGToSG;
      ierr = ISRestoreIndices(ix,&idx);CHKERRQ(ierr);
      ierr = ISRestoreIndices(iy,&idy);CHKERRQ(ierr);
      PetscLogInfo(xin,"VecScatterCreate:Sequential vector scatter with block indices\n");
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
    if (ix->type == IS_STRIDE && iy->type == IS_STRIDE){
      int                   nx,ny,to_first,to_step,from_first,from_step;
      int                   start,end;
      VecScatter_Seq_Stride *from12,*to12;

      ierr = VecGetOwnershipRange(xin,&start,&end);CHKERRQ(ierr);
      ierr = ISGetLocalSize(ix,&nx);CHKERRQ(ierr);
      ierr = ISStrideGetInfo(ix,&from_first,&from_step);CHKERRQ(ierr);
      ierr = ISGetLocalSize(iy,&ny);CHKERRQ(ierr); 
      ierr = ISStrideGetInfo(iy,&to_first,&to_step);CHKERRQ(ierr);
      if (nx != ny) SETERRQ(PETSC_ERR_ARG_SIZ,"Local scatter sizes don't match");
      if (ix->min >= start && ix->max < end) islocal = PETSC_TRUE; else islocal = PETSC_FALSE;
      ierr = MPI_Allreduce(&islocal,&cando,1,MPI_INT,MPI_LAND,xin->comm);CHKERRQ(ierr);
      if (cando) {
        ierr                = PetscNew(VecScatter_Seq_Stride,&to12);CHKERRQ(ierr);
        to12->n             = nx; 
        to12->first         = to_first;
        to12->step          = to_step;
        ierr                = PetscNew(VecScatter_Seq_Stride,&from12);CHKERRQ(ierr);
        PetscLogObjectMemory(ctx,2*sizeof(VecScatter_Seq_Stride));
        from12->n           = nx; 
        from12->first       = from_first-start; 
        from12->step        = from_step;
        to12->type          = VEC_SCATTER_SEQ_STRIDE; 
        from12->type        = VEC_SCATTER_SEQ_STRIDE; 
        ctx->todata       = (void*)to12; 
        ctx->fromdata     = (void*)from12;
        ctx->postrecvs    = 0;
        ctx->begin        = VecScatterBegin_SStoSS; 
        ctx->end          = 0; 
        ctx->destroy      = VecScatterDestroy_SGtoSG;
        ctx->copy         = VecScatterCopy_PStoSS;
        PetscLogInfo(xin,"VecScatterCreate:Special case: processors only getting local values\n");
        goto functionend;
      }
    } else {
      ierr = MPI_Allreduce(&islocal,&cando,1,MPI_INT,MPI_LAND,xin->comm);CHKERRQ(ierr);
    }

    /* test for special case of all processors getting entire vector */
    totalv = 0;
    if (ix->type == IS_STRIDE && iy->type == IS_STRIDE){
      int                  i,nx,ny,to_first,to_step,from_first,from_step,*count,N;
      VecScatter_MPI_ToAll *sto;

      ierr = ISGetLocalSize(ix,&nx);CHKERRQ(ierr);
      ierr = ISStrideGetInfo(ix,&from_first,&from_step);CHKERRQ(ierr);
      ierr = ISGetLocalSize(iy,&ny);CHKERRQ(ierr); 
      ierr = ISStrideGetInfo(iy,&to_first,&to_step);CHKERRQ(ierr);
      if (nx != ny) SETERRQ(PETSC_ERR_ARG_SIZ,"Local scatter sizes don't match");
      ierr = VecGetSize(xin,&N);CHKERRQ(ierr);
      if (nx != N) {
        totalv = 0;
      } else if (from_first == 0        && from_step == 1 && 
                 from_first == to_first && from_step == to_step){
        totalv = 1; 
      } else totalv = 0;
      ierr = MPI_Allreduce(&totalv,&cando,1,MPI_INT,MPI_LAND,xin->comm);CHKERRQ(ierr);

      if (cando) {
        PetscMap map;

        ierr  = MPI_Comm_size(ctx->comm,&size);CHKERRQ(ierr);
        ierr  = PetscNew(VecScatter_MPI_ToAll,&sto);CHKERRQ(ierr);
	ierr  = PetscMalloc(size*sizeof(int),&count);CHKERRQ(ierr);
        ierr  = VecGetPetscMap(xin,&map);CHKERRQ(ierr);
        ierr  = PetscMapGetGlobalRange(map,&range);CHKERRQ(ierr);
        for (i=0; i<size; i++) {
	  count[i] = range[i+1] - range[i];
        }
        sto->count        = count;
        sto->work1        = 0;
        sto->work2        = 0;
        sto->type         = VEC_SCATTER_MPI_TOALL;
        PetscLogObjectMemory(ctx,sizeof(VecScatter_MPI_ToAll)+size*sizeof(int));
        ctx->todata       = (void*)sto;
        ctx->fromdata     = 0;
        ctx->postrecvs    = 0;
        ctx->begin        = VecScatterBegin_MPI_ToAll;   
        ctx->end          = 0;
        ctx->destroy      = VecScatterDestroy_MPI_ToAll;
        ctx->copy         = VecScatterCopy_MPI_ToAll;
        PetscLogInfo(xin,"VecScatterCreate:Special case: all processors get entire parallel vector\n");
        goto functionend;
      }
    } else {
      ierr = MPI_Allreduce(&totalv,&cando,1,MPI_INT,MPI_LAND,xin->comm);CHKERRQ(ierr);
    }

    /* test for special case of processor 0 getting entire vector */
    totalv = 0;
    if (ix->type == IS_STRIDE && iy->type == IS_STRIDE){
      int                  i,nx,ny,to_first,to_step,from_first,from_step,*count,rank,N;
      VecScatter_MPI_ToAll *sto;

      ierr = PetscObjectGetComm((PetscObject)xin,&comm);CHKERRQ(ierr);
      ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
      ierr = ISGetLocalSize(ix,&nx);CHKERRQ(ierr);
      ierr = ISStrideGetInfo(ix,&from_first,&from_step);CHKERRQ(ierr);
      ierr = ISGetLocalSize(iy,&ny);CHKERRQ(ierr);
      ierr = ISStrideGetInfo(iy,&to_first,&to_step);CHKERRQ(ierr);
      if (nx != ny) SETERRQ(PETSC_ERR_ARG_SIZ,"Local scatter sizes don't match");
      if (!rank) {
        ierr = VecGetSize(xin,&N);CHKERRQ(ierr);
        if (nx != N) {
          totalv = 0;
        } else if (from_first == 0        && from_step == 1 && 
                   from_first == to_first && from_step == to_step){
          totalv = 1; 
        } else totalv = 0;
      } else {
        if (!nx) totalv = 1;
        else     totalv = 0;
      }
      ierr = MPI_Allreduce(&totalv,&cando,1,MPI_INT,MPI_LAND,xin->comm);CHKERRQ(ierr);

      if (cando) {
        PetscMap map;

        ierr  = MPI_Comm_size(ctx->comm,&size);CHKERRQ(ierr);
        ierr  = PetscNew(VecScatter_MPI_ToAll,&sto);CHKERRQ(ierr);
	ierr  = PetscMalloc(size*sizeof(int),&count);CHKERRQ(ierr);
        ierr  = VecGetPetscMap(xin,&map);CHKERRQ(ierr);
        ierr  = PetscMapGetGlobalRange(map,&range);CHKERRQ(ierr);
        for (i=0; i<size; i++) {
	  count[i] = range[i+1] - range[i];
        }
        sto->count        = count;
        sto->work1        = 0;
        sto->work2        = 0;
        sto->type         = VEC_SCATTER_MPI_TOONE;
        PetscLogObjectMemory(ctx,sizeof(VecScatter_MPI_ToAll)+size*sizeof(int));
        ctx->todata       = (void*)sto;
        ctx->fromdata     = 0;
        ctx->postrecvs    = 0;
        ctx->begin        = VecScatterBegin_MPI_ToOne;   
        ctx->end          = 0;
        ctx->destroy      = VecScatterDestroy_MPI_ToAll;
        ctx->copy         = VecScatterCopy_MPI_ToAll;
        PetscLogInfo(xin,"VecScatterCreate:Special case: processor zero gets entire parallel vector, rest get none\n");
        goto functionend;
      }
    } else {
      ierr = MPI_Allreduce(&totalv,&cando,1,MPI_INT,MPI_LAND,xin->comm);CHKERRQ(ierr);
    }

    ierr = ISBlock(ix,&ixblock);CHKERRQ(ierr);
    ierr = ISBlock(iy,&iyblock);CHKERRQ(ierr);
    ierr = ISStride(iy,&iystride);CHKERRQ(ierr);
    if (ixblock) {
      /* special case block to block */
      if (iyblock) {
        int nx,ny,*idx,*idy,bsx,bsy;
        ierr = ISBlockGetBlockSize(iy,&bsy);CHKERRQ(ierr);
        ierr = ISBlockGetBlockSize(ix,&bsx);CHKERRQ(ierr);
        if (bsx == bsy && (bsx == 12 || bsx == 5 || bsx == 4 || bsx == 3 || bsx == 2)) {
          ierr = ISBlockGetSize(ix,&nx);CHKERRQ(ierr);
          ierr = ISBlockGetIndices(ix,&idx);CHKERRQ(ierr);
          ierr = ISBlockGetSize(iy,&ny);CHKERRQ(ierr);
          ierr = ISBlockGetIndices(iy,&idy);CHKERRQ(ierr);
          if (nx != ny) SETERRQ(PETSC_ERR_ARG_SIZ,"Local scatter sizes don't match");
          ierr = VecScatterCreate_PtoS(nx,idx,ny,idy,xin,yin,bsx,ctx);CHKERRQ(ierr);
          ierr = ISBlockRestoreIndices(ix,&idx);CHKERRQ(ierr);
          ierr = ISBlockRestoreIndices(iy,&idy);CHKERRQ(ierr);
          PetscLogInfo(xin,"VecScatterCreate:Special case: blocked indices\n");
          goto functionend;
        }
      /* special case block to stride */
      } else if (iystride) {
        int ystart,ystride,ysize,bsx;
        ierr = ISStrideGetInfo(iy,&ystart,&ystride);CHKERRQ(ierr);
        ierr = ISGetLocalSize(iy,&ysize);CHKERRQ(ierr);
        ierr = ISBlockGetBlockSize(ix,&bsx);CHKERRQ(ierr);
        /* see if stride index set is equivalent to block index set */
        if (((bsx == 2) || (bsx == 3) || (bsx == 4) || (bsx == 5) || (bsx == 12)) && 
            ((ystart % bsx) == 0) && (ystride == 1) && ((ysize % bsx) == 0)) {
          int nx,*idx,*idy,il;
          ierr = ISBlockGetSize(ix,&nx); ISBlockGetIndices(ix,&idx);CHKERRQ(ierr);
          if (ysize != bsx*nx) SETERRQ(PETSC_ERR_ARG_SIZ,"Local scatter sizes don't match");
	  ierr = PetscMalloc((nx+1)*sizeof(int),&idy);CHKERRQ(ierr);
          idy[0] = ystart;
          for (il=1; il<nx; il++) idy[il] = idy[il-1] + bsx; 
          ierr = VecScatterCreate_PtoS(nx,idx,nx,idy,xin,yin,bsx,ctx);CHKERRQ(ierr);
          ierr = PetscFree(idy);CHKERRQ(ierr);
          ierr = ISBlockRestoreIndices(ix,&idx);CHKERRQ(ierr);
          PetscLogInfo(xin,"VecScatterCreate:Special case: blocked indices to stride\n");
          goto functionend;
        }
      }
    }
    /* left over general case */
    {
      int nx,ny,*idx,*idy;
      ierr = ISGetLocalSize(ix,&nx);CHKERRQ(ierr); 
      ierr = ISGetIndices(ix,&idx);CHKERRQ(ierr);
      ierr = ISGetLocalSize(iy,&ny);CHKERRQ(ierr);
      ierr = ISGetIndices(iy,&idy);CHKERRQ(ierr);
      if (nx != ny) SETERRQ(PETSC_ERR_ARG_SIZ,"Local scatter sizes don't match");
      ierr = VecScatterCreate_PtoS(nx,idx,ny,idy,xin,yin,1,ctx);CHKERRQ(ierr);
      ierr = ISRestoreIndices(ix,&idx);CHKERRQ(ierr);
      ierr = ISRestoreIndices(iy,&idy);CHKERRQ(ierr);
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
    if (ix->type == IS_STRIDE && iy->type == IS_STRIDE){
      int                   nx,ny,to_first,to_step,from_step,start,end,from_first;
      VecScatter_Seq_Stride *from,*to;

      ierr = VecGetOwnershipRange(yin,&start,&end);CHKERRQ(ierr);
      ierr = ISGetLocalSize(ix,&nx);CHKERRQ(ierr);
      ierr = ISStrideGetInfo(ix,&from_first,&from_step);CHKERRQ(ierr);
      ierr = ISGetLocalSize(iy,&ny);CHKERRQ(ierr); 
      ierr = ISStrideGetInfo(iy,&to_first,&to_step);CHKERRQ(ierr);
      if (nx != ny) SETERRQ(PETSC_ERR_ARG_SIZ,"Local scatter sizes don't match");
      if (iy->min >= start && iy->max < end) islocal = PETSC_TRUE; else islocal = PETSC_FALSE;
      ierr = MPI_Allreduce(&islocal,&cando,1,MPI_INT,MPI_LAND,yin->comm);CHKERRQ(ierr);
      if (cando) {
        ierr              = PetscNew(VecScatter_Seq_Stride,&to);CHKERRQ(ierr);
        to->n             = nx; 
        to->first         = to_first-start; 
        to->step          = to_step;
        ierr              = PetscNew(VecScatter_Seq_Stride,&from);CHKERRQ(ierr);
        PetscLogObjectMemory(ctx,2*sizeof(VecScatter_Seq_Stride));
        from->n           = nx; 
        from->first       = from_first; 
        from->step        = from_step;
        to->type          = VEC_SCATTER_SEQ_STRIDE; 
        from->type        = VEC_SCATTER_SEQ_STRIDE;
        ctx->todata       = (void*)to;
        ctx->fromdata     = (void*)from;
        ctx->postrecvs    = 0;
        ctx->begin        = VecScatterBegin_SStoSS; 
        ctx->end          = 0;  
        ctx->destroy      = VecScatterDestroy_SGtoSG;
        ctx->copy         = VecScatterCopy_PStoSS;
        PetscLogInfo(xin,"VecScatterCreate:Special case: sequential stride to stride\n");
        goto functionend;
      }
    } else {
      ierr = MPI_Allreduce(&islocal,&cando,1,MPI_INT,MPI_LAND,yin->comm);CHKERRQ(ierr);
    }
    /* general case */
    {
      int nx,ny,*idx,*idy;
      ierr = ISGetLocalSize(ix,&nx);CHKERRQ(ierr);
      ierr = ISGetIndices(ix,&idx);CHKERRQ(ierr);
      ierr = ISGetLocalSize(iy,&ny);CHKERRQ(ierr);
      ierr = ISGetIndices(iy,&idy);CHKERRQ(ierr);
      if (nx != ny) SETERRQ(PETSC_ERR_ARG_SIZ,"Local scatter sizes don't match");
      ierr = VecScatterCreate_StoP(nx,idx,ny,idy,yin,ctx);CHKERRQ(ierr);
      ierr = ISRestoreIndices(ix,&idx);CHKERRQ(ierr);
      ierr = ISRestoreIndices(iy,&idy);CHKERRQ(ierr);
      goto functionend;
    }
  }
  /* ---------------------------------------------------------------------------*/
  if (xin_type == VEC_MPI_ID && yin_type == VEC_MPI_ID) {
    /* no special cases for now */
    int nx,ny,*idx,*idy;
    ierr    = ISGetLocalSize(ix,&nx);CHKERRQ(ierr); 
    ierr    = ISGetIndices(ix,&idx);CHKERRQ(ierr);
    ierr    = ISGetLocalSize(iy,&ny);CHKERRQ(ierr); 
    ierr    = ISGetIndices(iy,&idy);CHKERRQ(ierr);
    if (nx != ny) SETERRQ(PETSC_ERR_ARG_SIZ,"Local scatter sizes don't match");
    ierr    = VecScatterCreate_PtoP(nx,idx,ny,idy,xin,yin,ctx);CHKERRQ(ierr);
    ierr    = ISRestoreIndices(ix,&idx);CHKERRQ(ierr); 
    ierr    = ISRestoreIndices(iy,&idy);CHKERRQ(ierr);
    ctx->copy = VecScatterCopy_MPI_ToAll;
    goto functionend;
  }

  functionend:
  *newctx = ctx;
  if (tix) {ierr = ISDestroy(tix);CHKERRQ(ierr);}
  if (tiy) {ierr = ISDestroy(tiy);CHKERRQ(ierr);}
  ierr = PetscOptionsHasName(PETSC_NULL,"-vecscatter_view_info",&flag);CHKERRQ(ierr);
  if (flag) {
    ierr = PetscViewerPushFormat(PETSC_VIEWER_STDOUT_(comm),PETSC_VIEWER_ASCII_INFO);CHKERRQ(ierr);
    ierr = VecScatterView(ctx,PETSC_VIEWER_STDOUT_(comm));CHKERRQ(ierr);
    ierr = PetscViewerPopFormat(PETSC_VIEWER_STDOUT_(comm));CHKERRQ(ierr);
  }
  ierr = PetscOptionsHasName(PETSC_NULL,"-vecscatter_view",&flag);CHKERRQ(ierr);
  if (flag) {
    ierr = VecScatterView(ctx,PETSC_VIEWER_STDOUT_(comm));CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------------*/
#undef __FUNCT__  
#define __FUNCT__ "VecScatterPostRecvs"
/*@
   VecScatterPostRecvs - Posts the receives required for the ready-receiver
   version of the VecScatter routines.

   Collective on VecScatter

   Input Parameters:
+  x - the vector from which we scatter (not needed,can be null)
.  y - the vector to which we scatter
.  addv - either ADD_VALUES or INSERT_VALUES
.  mode - the scattering mode, usually SCATTER_FORWARD.  The available modes are:
    SCATTER_FORWARD, SCATTER_REVERSE
-  inctx - scatter context generated by VecScatterCreate()

   Output Parameter:
.  y - the vector to which we scatter

   Level: advanced

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

.seealso: VecScatterCreate(), VecScatterEnd(), VecScatterBegin()
@*/
int VecScatterPostRecvs(Vec x,Vec y,InsertMode addv,ScatterMode mode,VecScatter inctx)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(y,VEC_COOKIE);
  PetscValidHeaderSpecific(inctx,VEC_SCATTER_COOKIE);

  if (inctx->postrecvs) {
    ierr = (*inctx->postrecvs)(x,y,addv,mode,inctx);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------------*/
#undef __FUNCT__  
#define __FUNCT__ "VecScatterBegin"
/*@
   VecScatterBegin - Begins a generalized scatter from one vector to
   another. Complete the scattering phase with VecScatterEnd().

   Collective on VecScatter and Vec

   Input Parameters:
+  x - the vector from which we scatter
.  y - the vector to which we scatter
.  addv - either ADD_VALUES or INSERT_VALUES
.  mode - the scattering mode, usually SCATTER_FORWARD.  The available modes are:
    SCATTER_FORWARD or SCATTER_REVERSE
-  inctx - scatter context generated by VecScatterCreate()

   Level: intermediate

   Notes:
   The vectors x and y need not be the same vectors used in the call 
   to VecScatterCreate(), but x must have the same parallel data layout
   as that passed in as the x to VecScatterCreate(), similarly for the y.
   Most likely they have been obtained from VecDuplicate().

   You cannot change the values in the input vector between the calls to VecScatterBegin()
   and VecScatterEnd().

   If you use SCATTER_REVERSE the first two arguments should be reversed, from 
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
int VecScatterBegin(Vec x,Vec y,InsertMode addv,ScatterMode mode,VecScatter inctx)
{
  int ierr;
#if defined(PETSC_USE_BOPT_g)
  int to_n,from_n;
#endif

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x,VEC_COOKIE);
  PetscValidHeaderSpecific(y,VEC_COOKIE);
  PetscValidHeaderSpecific(inctx,VEC_SCATTER_COOKIE);
  if (inctx->inuse) SETERRQ(PETSC_ERR_ARG_WRONGSTATE," Scatter ctx already in use");
#if defined(PETSC_USE_BOPT_g)
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
      if (to_n != inctx->from_n) SETERRQ(PETSC_ERR_ARG_SIZ,"Vector wrong size for scatter (scatter reverse and vector to != ctx from size)");
      if (from_n != inctx->to_n) SETERRQ(PETSC_ERR_ARG_SIZ,"Vector wrong size for scatter (scatter reverse and vector from != ctx to size)");
    } else {
      if (to_n != inctx->to_n)     SETERRQ(PETSC_ERR_ARG_SIZ,"Vector wrong size for scatter (scatter forward and vector to != ctx to size)");
      if (from_n != inctx->from_n) SETERRQ(PETSC_ERR_ARG_SIZ,"Vector wrong size for scatter (scatter forward and vector from != ctx from size)");
    }
  }
#endif

  inctx->inuse = PETSC_TRUE;
  ierr = PetscLogEventBarrierBegin(VEC_ScatterBarrier,0,0,0,0,inctx->comm);CHKERRQ(ierr);
  ierr = (*inctx->begin)(x,y,addv,mode,inctx);CHKERRQ(ierr);
  if (inctx->beginandendtogether && inctx->end) {
    inctx->inuse = PETSC_FALSE;
    ierr = (*inctx->end)(x,y,addv,mode,inctx);CHKERRQ(ierr);
  }
  ierr = PetscLogEventBarrierEnd(VEC_ScatterBarrier,0,0,0,0,inctx->comm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* --------------------------------------------------------------------*/
#undef __FUNCT__  
#define __FUNCT__ "VecScatterEnd"
/*@
   VecScatterEnd - Ends a generalized scatter from one vector to another.  Call
   after first calling VecScatterBegin().

   Collective on VecScatter and Vec

   Input Parameters:
+  x - the vector from which we scatter
.  y - the vector to which we scatter
.  addv - either ADD_VALUES or INSERT_VALUES.
.  mode - the scattering mode, usually SCATTER_FORWARD.  The available modes are:
     SCATTER_FORWARD, SCATTER_REVERSE
-  ctx - scatter context generated by VecScatterCreate()

   Level: intermediate

   Notes:
   If you use SCATTER_REVERSE the first two arguments should be reversed, from 
   the SCATTER_FORWARD.
   y[iy[i]] = x[ix[i]], for i=0,...,ni-1

.seealso: VecScatterBegin(), VecScatterCreate()
@*/
int VecScatterEnd(Vec x,Vec y,InsertMode addv,ScatterMode mode,VecScatter ctx)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x,VEC_COOKIE);
  PetscValidHeaderSpecific(y,VEC_COOKIE);
  PetscValidHeaderSpecific(ctx,VEC_SCATTER_COOKIE);
  ctx->inuse = PETSC_FALSE;
  if (!ctx->end) PetscFunctionReturn(0);
  if (!ctx->beginandendtogether) {
    ierr = PetscLogEventBegin(VEC_ScatterEnd,ctx,x,y,0);CHKERRQ(ierr);
    ierr = (*(ctx)->end)(x,y,addv,mode,ctx);CHKERRQ(ierr);
    ierr = PetscLogEventEnd(VEC_ScatterEnd,ctx,x,y,0);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecScatterDestroy"
/*@C
   VecScatterDestroy - Destroys a scatter context created by 
   VecScatterCreate().

   Collective on VecScatter

   Input Parameter:
.  ctx - the scatter context

   Level: intermediate

.seealso: VecScatterCreate(), VecScatterCopy()
@*/
int VecScatterDestroy(VecScatter ctx)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ctx,VEC_SCATTER_COOKIE);
  if (--ctx->refct > 0) PetscFunctionReturn(0);

  /* if memory was published with AMS then destroy it */
  ierr = PetscObjectDepublish(ctx);CHKERRQ(ierr);

  ierr = (*ctx->destroy)(ctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecScatterCopy"
/*@C
   VecScatterCopy - Makes a copy of a scatter context.

   Collective on VecScatter

   Input Parameter:
.  sctx - the scatter context

   Output Parameter:
.  ctx - the context copy

   Level: advanced

.seealso: VecScatterCreate(), VecScatterDestroy()
@*/
int VecScatterCopy(VecScatter sctx,VecScatter *ctx)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sctx,VEC_SCATTER_COOKIE);
  PetscValidPointer(ctx);
  if (!sctx->copy) SETERRQ(PETSC_ERR_SUP,"Cannot copy this type");
  PetscHeaderCreate(*ctx,_p_VecScatter,int,VEC_SCATTER_COOKIE,0,"VecScatter",sctx->comm,VecScatterDestroy,VecScatterView);
  PetscLogObjectCreate(*ctx);
  PetscLogObjectMemory(*ctx,sizeof(struct _p_VecScatter));
  (*ctx)->to_n   = sctx->to_n;
  (*ctx)->from_n = sctx->from_n;
  ierr = (*sctx->copy)(sctx,*ctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


/* ------------------------------------------------------------------*/
#undef __FUNCT__  
#define __FUNCT__ "VecScatterView"
/*@
   VecScatterView - Views a vector scatter context.

   Collective on VecScatter

   Input Parameters:
+  ctx - the scatter context
-  viewer - the viewer for displaying the context

   Level: intermediate

@*/
int VecScatterView(VecScatter ctx,PetscViewer viewer)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ctx,VEC_SCATTER_COOKIE);
  if (!viewer) viewer = PETSC_VIEWER_STDOUT_(ctx->comm);
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_COOKIE);
  if (!ctx->view) SETERRQ(PETSC_ERR_SUP,"Cannot view this type of scatter context yet");

  ierr = (*ctx->view)(ctx,viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecScatterRemap"
/*@
   VecScatterRemap - Remaps the "from" and "to" indices in a 
   vector scatter context. FOR EXPERTS ONLY!

   Collective on VecScatter

   Input Parameters:
+  scat - vector scatter context
.  from - remapping for "from" indices (may be PETSC_NULL)
-  to   - remapping for "to" indices (may be PETSC_NULL)

   Level: developer

   Notes: In the parallel case the todata is actually the indices
          from which the data is TAKEN! The from stuff is where the 
          data is finally put. This is VERY VERY confusing!

          In the sequential case the todata is the indices where the 
          data is put and the fromdata is where it is taken from.
          This is backwards from the paralllel case! CRY! CRY! CRY!

@*/
int VecScatterRemap(VecScatter scat,int *rto,int *rfrom)
{
  VecScatter_Seq_General *to,*from;
  VecScatter_MPI_General *mto;
  int                    i;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(scat,VEC_SCATTER_COOKIE);
  if (rto)   {PetscValidIntPointer(rto);}
  if (rfrom) {PetscValidIntPointer(rfrom);}

  from = (VecScatter_Seq_General *)scat->fromdata;
  mto  = (VecScatter_MPI_General *)scat->todata;

  if (mto->type == VEC_SCATTER_MPI_TOALL) SETERRQ(PETSC_ERR_ARG_SIZ,"Not for to all scatter");

  if (rto) {
    if (mto->type == VEC_SCATTER_MPI_GENERAL) {
      /* handle off processor parts */
      for (i=0; i<mto->starts[mto->n]; i++) {
        mto->indices[i] = rto[mto->indices[i]];
      }
      /* handle local part */
      to = &mto->local;
      for (i=0; i<to->n; i++) {
        to->slots[i] = rto[to->slots[i]];
      }
    } else if (from->type == VEC_SCATTER_SEQ_GENERAL) {
      for (i=0; i<from->n; i++) {
        from->slots[i] = rto[from->slots[i]];
      }
    } else if (from->type == VEC_SCATTER_SEQ_STRIDE) {
      VecScatter_Seq_Stride *sto = (VecScatter_Seq_Stride*)from;
      
      /* if the remapping is the identity and stride is identity then skip remap */
      if (sto->step == 1 && sto->first == 0) {
        for (i=0; i<sto->n; i++) {
          if (rto[i] != i) {
            SETERRQ(PETSC_ERR_ARG_SIZ,"Unable to remap such scatters");
          }
        }
      } else SETERRQ(PETSC_ERR_ARG_SIZ,"Unable to remap such scatters");
    } else SETERRQ(PETSC_ERR_ARG_SIZ,"Unable to remap such scatters");
  }

  if (rfrom) {
    SETERRQ(PETSC_ERR_SUP,"Unable to remap the FROM in scatters yet");
  }

  /*
     Mark then vector lengths as unknown because we do not know the 
   lengths of the remapped vectors
  */
  scat->from_n = -1;
  scat->to_n   = -1;

  PetscFunctionReturn(0);
}
