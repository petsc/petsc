
#include <stdlib.h>

#include <petsc.h>
#include <petscvec.h>
#include <private/vecimpl.h>

#include "vecblockimpl.h"
#include "vecblockprivate.h"

/* check all blocks are filled */
#undef __FUNCT__  
#define __FUNCT__ "VecAssemblyBegin_Block"
PetscErrorCode VecAssemblyBegin_Block(Vec v)
{
  Vec_Block      *vs = (Vec_Block*)v->data;
  PetscInt       i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  for (i=0;i<vs->nb;i++) {
    if (!vs->v[i]) {
      SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Block vector cannot contain NULL blocks");
    }
    ierr = VecAssemblyBegin(vs->v[i]);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecAssemblyEnd_Block"
PetscErrorCode VecAssemblyEnd_Block(Vec v)
{
  Vec_Block      *vs = (Vec_Block*)v->data;
  PetscInt       i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  for (i=0;i<vs->nb;i++) {
    ierr = VecAssemblyEnd(vs->v[i]);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecDestroy_Block"
PetscErrorCode VecDestroy_Block(Vec v)
{
  Vec_Block      *vs = (Vec_Block*)v->data;
  PetscInt       i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!vs->v) {
    for (i=0; i<vs->nb; i++) {
      if (!vs->v[i]) {
        ierr = VecDestroy(vs->v[i]);CHKERRQ(ierr);
        vs->v[i] = PETSC_NULL;
      }
    }
    ierr = PetscFree(vs->v);CHKERRQ(ierr);
  }
  ierr = PetscFree(vs);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecSetUp_Block"
PetscErrorCode VecSetUp_Block(Vec V)
{
  Vec_Block      *ctx = (Vec_Block*)V->data;
  PetscInt       i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (ctx->setup_called) PetscFunctionReturn(0);

  ctx->nb = V->map->N;
  V->map->n = V->map->N;

  if (ctx->nb < 0) {
    SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_WRONG,"Cannot create VEC_BLOCK with < 0 blocks.");
  }

  /* Create space */
  ierr = PetscMalloc(ctx->nb*sizeof(Vec),&ctx->v);CHKERRQ(ierr);
  for (i=0; i<ctx->nb; i++) {
    ctx->v[i] = PETSC_NULL;
    /* Do not allocate memory for internal sub blocks */
  }

  ctx->setup_called = PETSC_TRUE;
  PetscFunctionReturn(0);
}

/* supports nested blocks */
#undef __FUNCT__  
#define __FUNCT__ "VecCopy_Block"
PetscErrorCode VecCopy_Block(Vec x,Vec y)
{
  Vec_Block      *bx = (Vec_Block*)x->data;
  Vec_Block      *by = (Vec_Block*)y->data;
  PetscInt       i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PETSc_VecBlock_Check2(x,y);CHKERRQ(ierr);
  for (i=0; i<bx->nb; i++) {
    ierr = VecCopy(bx->v[i],by->v[i]);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/* supports nested blocks */
#undef __FUNCT__  
#define __FUNCT__ "VecDuplicate_Block"
PetscErrorCode VecDuplicate_Block(Vec x,Vec *y)
{
  Vec_Block      *bx = (Vec_Block*)x->data;
  Vec_Block      *by;
  PetscInt       i;
  Vec            _y,Y;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecCreate(((PetscObject)x)->comm,&_y);CHKERRQ(ierr);
  ierr = VecSetSizes(_y,bx->nb,bx->nb);CHKERRQ(ierr);
  ierr = VecSetType(_y,VECBLOCK);CHKERRQ(ierr);

  by = (Vec_Block*)_y->data;
  for (i=0; i<bx->nb; i++) {
    ierr = VecDuplicate(bx->v[i],&Y);CHKERRQ(ierr);
    ierr = VecBlockSetSubVec(_y,i,Y);CHKERRQ(ierr);
    ierr = VecDestroy(Y);CHKERRQ(ierr); /* Hand over control of Y to the block vector _y */
  }

  *y = _y;
  PetscFunctionReturn(0);
}

/* supports nested blocks */
#undef __FUNCT__  
#define __FUNCT__ "VecDot_Block"
PetscErrorCode VecDot_Block(Vec x,Vec y,PetscScalar *val)
{
  Vec_Block      *bx = (Vec_Block*)x->data;
  Vec_Block      *by = (Vec_Block*)y->data;
  PetscInt       i,nr;
  PetscScalar    x_dot_y,_val;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  nr = bx->nb;
  _val = 0.0;
  for (i=0; i<nr; i++) {
    ierr = VecDot(bx->v[i],by->v[i],&x_dot_y);CHKERRQ(ierr);
    _val = _val + x_dot_y;
  }
  *val = _val;
  PetscFunctionReturn(0);
}

/* supports nested blocks */
#undef __FUNCT__  
#define __FUNCT__ "VecTDot_Block"
PetscErrorCode VecTDot_Block(Vec x,Vec y,PetscScalar *val)
{
  Vec_Block      *bx = (Vec_Block*)x->data;
  Vec_Block      *by = (Vec_Block*)y->data;
  PetscInt       i,nr;
  PetscScalar    x_dot_y,_val;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  nr = bx->nb;
  _val = 0.0;
  for (i=0; i<nr; i++) {
    ierr = VecTDot(bx->v[i],by->v[i],&x_dot_y);CHKERRQ(ierr);
    _val = _val + x_dot_y;
  }
  *val = _val;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecAXPY_Block"
PetscErrorCode VecAXPY_Block(Vec y,PetscScalar alpha,Vec x)
{
  Vec_Block      *bx = (Vec_Block*)x->data;
  Vec_Block      *by = (Vec_Block*)y->data;
  PetscInt       i,nr;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  nr = bx->nb;
  for (i=0; i<nr; i++) {
    ierr = VecAXPY(by->v[i],alpha,bx->v[i]);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecAYPX_Block"
PetscErrorCode VecAYPX_Block(Vec y,PetscScalar alpha,Vec x)
{
  Vec_Block      *bx = (Vec_Block*)x->data;
  Vec_Block      *by = (Vec_Block*)y->data;
  PetscInt       i,nr;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  nr = bx->nb;
  for (i=0; i<nr; i++) {
    ierr = VecAYPX(by->v[i],alpha,bx->v[i]);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecAXPBY_Block"
PetscErrorCode VecAXPBY_Block(Vec y,PetscScalar alpha,PetscScalar beta,Vec x)
{
  Vec_Block      *bx = (Vec_Block*)x->data;
  Vec_Block      *by = (Vec_Block*)y->data;
  PetscInt       i,nr;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  nr = bx->nb;
  for (i=0; i<nr; i++) {
    ierr = VecAXPBY(by->v[i],alpha,beta,bx->v[i]);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecScale_Block"
PetscErrorCode VecScale_Block(Vec x,PetscScalar alpha)
{
  Vec_Block      *bx = (Vec_Block*)x->data;
  PetscInt       i,nr;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  nr = bx->nb;
  for (i=0; i<nr; i++) {
    ierr = VecScale(bx->v[i],alpha);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecPointwiseMult_Block"
PetscErrorCode VecPointwiseMult_Block(Vec w,Vec x,Vec y)
{
  Vec_Block      *bx = (Vec_Block*)x->data;
  Vec_Block      *by = (Vec_Block*)y->data;
  Vec_Block      *bw = (Vec_Block*)w->data;
  PetscInt       i,nr;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PETSc_VecBlock_Check3(w,x,y);CHKERRQ(ierr);
  nr = bx->nb;
  for (i=0; i<nr; i++) {
    ierr = VecPointwiseMult(bw->v[i],bx->v[i],by->v[i]);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecPointwiseDivide_Block"
PetscErrorCode VecPointwiseDivide_Block(Vec w,Vec x,Vec y)
{
  Vec_Block      *bx = (Vec_Block*)x->data;
  Vec_Block      *by = (Vec_Block*)y->data;
  Vec_Block      *bw = (Vec_Block*)w->data;
  PetscInt       i,nr;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PETSc_VecBlock_Check3(w,x,y);CHKERRQ(ierr);

  nr = bx->nb;
  for (i=0; i<nr; i++) {
    ierr = VecPointwiseDivide(bw->v[i],bx->v[i],by->v[i]);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecReciprocal_Block"
PetscErrorCode VecReciprocal_Block(Vec x)
{
  Vec_Block      *bx = (Vec_Block*)x->data;
  PetscInt       i,nr;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  nr = bx->nb;
  for (i=0; i<nr; i++) {
    ierr = VecReciprocal(bx->v[i]);CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecNorm_Block"
PetscErrorCode VecNorm_Block(Vec xin,NormType type,PetscReal* z)
{
  Vec_Block      *bx = (Vec_Block*)xin->data;
  PetscInt       i,nr;
  PetscReal      z_i;
  PetscReal      _z;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  nr = bx->nb;
  _z = 0.0;

  if (type == NORM_2) {
    PetscScalar dot;
#ifdef PETSC_USE_COMPLEX
    PetscReal im,re;
#endif
    ierr = VecDot(xin,xin,&dot);CHKERRQ(ierr);
#ifdef PETSC_USE_COMPLEX
    re = PetscRealPart(dot);
    im = PetscImaginaryPart(dot);
    _z = sqrt(re - im);
#else
    _z = sqrt(dot);
#endif
  } else if (type == NORM_1) {
    for (i=0; i<nr; i++) {
      ierr = VecNorm(bx->v[i],type,&z_i);CHKERRQ(ierr);
      _z = _z + z_i;
    }
  } else if (type == NORM_INFINITY) {
    for (i=0; i<nr; i++) {
      ierr = VecNorm(bx->v[i],type,&z_i);CHKERRQ(ierr);
      if (z_i > _z) {
        _z = z_i;
      }
    }
  }

  *z = _z;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecMAXPY_Block"
PetscErrorCode VecMAXPY_Block(Vec y,PetscInt nv,const PetscScalar alpha[],Vec *x)
{
  PetscInt       v;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  for (v=0; v<nv; v++) {
    /* Do axpy on each vector,v */
    ierr = VecAXPY(y,alpha[v],x[v]);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "VecMDot_Block"
PetscErrorCode VecMDot_Block(Vec x,PetscInt nv,const Vec y[],PetscScalar *val)
{
  PetscInt       j;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  for (j=0; j<nv; j++) {
    ierr = VecDot(x,y[j],&val[j]);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "VecMTDot_Block"
PetscErrorCode VecMTDot_Block(Vec x,PetscInt nv,const Vec y[],PetscScalar *val)
{
  PetscInt       j;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  for (j=0; j<nv; j++) {
    ierr = VecTDot(x,y[j],&val[j]);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "VecSet_Block"
PetscErrorCode VecSet_Block(Vec x,PetscScalar alpha)
{
  Vec_Block      *bx = (Vec_Block*)x->data;
  PetscInt       i,nr;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  nr = bx->nb;
  for (i=0; i<nr; i++) {
    ierr = VecSet(bx->v[i],alpha);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "VecConjugate_Block"
PetscErrorCode VecConjugate_Block(Vec x)
{
  Vec_Block      *bx = (Vec_Block*)x->data;
  PetscInt       j,nr;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  nr = bx->nb;
  for (j=0; j<nr; j++) {
    ierr = VecConjugate(bx->v[j]);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "VecSwap_Block"
PetscErrorCode VecSwap_Block(Vec x,Vec y)
{
  Vec_Block      *bx = (Vec_Block*)x->data;
  Vec_Block      *by = (Vec_Block*)y->data;
  PetscInt       i,nr;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PETSc_VecBlock_Check2(x,y);CHKERRQ(ierr);
  nr = bx->nb;
  for (i=0; i<nr; i++) {
    ierr = VecSwap(bx->v[i],by->v[i]);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "VecWAXPY_Block"
PetscErrorCode VecWAXPY_Block(Vec w,PetscScalar alpha,Vec x,Vec y)
{
  Vec_Block      *bx = (Vec_Block*)x->data;
  Vec_Block      *by = (Vec_Block*)y->data;
  Vec_Block      *bw = (Vec_Block*)w->data;
  PetscInt       i,nr;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PETSc_VecBlock_Check3(w,x,y);CHKERRQ(ierr);

  nr = bx->nb;
  for (i=0; i<nr; i++) {
    ierr = VecWAXPY(bw->v[i],alpha,bx->v[i],by->v[i]);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "__vec_max_block"
PetscErrorCode __vec_max_block(Vec x,PetscInt *cnt,PetscInt *p,PetscReal *max)
{
  Vec_Block *bx = (Vec_Block*)x->data;

  PetscInt     i,nr;
  PetscBool    isblock;
  PetscInt     L;
  PetscInt     _entry_loc;
  PetscReal    _entry_val;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscTypeCompare((PetscObject)x,VECBLOCK,&isblock);CHKERRQ(ierr);
  if (!isblock) {
    /* Not block */
    ierr = VecMax(x,&_entry_loc,&_entry_val);CHKERRQ(ierr);
    if (_entry_val > *max) {
      *max = _entry_val;
      *p = _entry_loc + *cnt;
    }
    ierr = VecGetSize(x,&L);CHKERRQ(ierr);
    *cnt = *cnt + L;
    PetscFunctionReturn(0);
  }

  /* Otherwise we have a block */
  bx = (Vec_Block*)x->data;
  nr = bx->nb;

  /* now descend recursively */
  for (i=0; i<nr; i++) {
    ierr = __vec_max_block(bx->v[i],cnt,p,max);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/* supports nested blocks */
#undef __FUNCT__
#define __FUNCT__ "VecMax_Block"
PetscErrorCode VecMax_Block(Vec x,PetscInt *p,PetscReal *max)
{
  PetscInt       cnt;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  cnt = 0;
  *p = 0;
  *max = 0.0;
  ierr = __vec_max_block(x,&cnt,p,max);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "__vec_min_block"
PetscErrorCode __vec_min_block(Vec x,PetscInt *cnt,PetscInt *p,PetscReal *min)
{
  Vec_Block      *bx = (Vec_Block*)x->data;
  PetscInt       i,nr,L,_entry_loc;
  PetscBool      isblock;
  PetscReal      _entry_val;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscTypeCompare((PetscObject)x,VECBLOCK,&isblock);CHKERRQ(ierr);
  if (!isblock) {
    /* Not block */
    ierr = VecMin(x,&_entry_loc,&_entry_val);CHKERRQ(ierr);
    if (_entry_val < *min) {
      *min = _entry_val;
      *p = _entry_loc + *cnt;
    }
    ierr = VecGetSize(x,&L);CHKERRQ(ierr);
    *cnt = *cnt + L;
    PetscFunctionReturn(0);
  }

  /* Otherwise we have a block */
  bx = (Vec_Block*)x->data;
  nr = bx->nb;

  /* now descend recursively */
  for (i=0; i<nr; i++) {
    ierr = __vec_min_block(bx->v[i],cnt,p,min);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "VecMin_Block"
PetscErrorCode VecMin_Block(Vec x,PetscInt *p,PetscReal *min)
{
  PetscInt       cnt;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  cnt = 0;
  *p = 0;
  *min = 1.0e308;
  ierr = __vec_min_block(x,&cnt,p,min);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* supports nested blocks */
#undef __FUNCT__
#define __FUNCT__ "VecView_Block"
PetscErrorCode VecView_Block(Vec x,PetscViewer viewer)
{
  Vec_Block      *bx = (Vec_Block*)x->data;
  PetscBool      isascii;
  PetscInt       i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&isascii);CHKERRQ(ierr);
  if (isascii) {
    ierr = PetscViewerASCIIPrintf(viewer,"Vector Object:\n");CHKERRQ(ierr);
    ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);         /* push0 */
    ierr = PetscViewerASCIIPrintf(viewer,"type=block, rows=%d \n",bx->nb);CHKERRQ(ierr);

    ierr = PetscViewerASCIIPrintf(viewer,"VecBlock structure: \n");CHKERRQ(ierr);
    for (i=0; i<bx->nb; i++) {
      const VecType type;
      const char *name;
      PetscInt NR;

      ierr = VecGetSize(bx->v[i],&NR);CHKERRQ(ierr);
      ierr = VecGetType(bx->v[i],&type);CHKERRQ(ierr);
      name = ((PetscObject)bx->v[i])->prefix;

      ierr = PetscViewerASCIIPrintf(viewer,"(%D) : name=\"%s\", type=%s, rows=%D \n",i,name,type,NR);CHKERRQ(ierr);

      ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);             /* push1 */
      ierr = VecView(bx->v[i],viewer);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);              /* pop1 */
    }
    ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);                /* pop0 */
  }
  PetscFunctionReturn(0);
}

/* Returns the number of blocks in size */
#undef __FUNCT__
#define __FUNCT__ "VecGetSize_Block"
PetscErrorCode VecGetSize_Block(Vec x,PetscInt *size)
{
  Vec_Block *bx = (Vec_Block*)x->data;

  PetscFunctionBegin;
  *size = bx->nb;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "VecMaxPointwiseDivide_Block"
PetscErrorCode VecMaxPointwiseDivide_Block(Vec x,Vec y,PetscReal *max)
{
  Vec_Block      *bx = (Vec_Block*)x->data;
  Vec_Block      *by = (Vec_Block*)y->data;
  PetscInt       i,nr;
  PetscReal      local_max,m;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PETSc_VecBlock_Check2(x,y);CHKERRQ(ierr);
  nr = bx->nb;
  m = 0.0;
  for (i=0; i<nr; i++) {
    ierr = VecMaxPointwiseDivide(bx->v[i],by->v[i],&local_max);CHKERRQ(ierr);
    if (local_max > m) {
      m = local_max;
    }
  }
  *max = m;
  PetscFunctionReturn(0);
}
