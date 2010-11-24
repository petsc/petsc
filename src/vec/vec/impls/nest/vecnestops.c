#include <private/vecimpl.h>

#include "vecnestimpl.h"

/* check all blocks are filled */
#undef __FUNCT__  
#define __FUNCT__ "VecAssemblyBegin_Nest"
PetscErrorCode VecAssemblyBegin_Nest(Vec v)
{
  Vec_Nest       *vs = (Vec_Nest*)v->data;
  PetscInt       i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  for (i=0;i<vs->nb;i++) {
    if (!vs->v[i]) {
      SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Nest  vector cannot contain NULL blocks");
    }
    ierr = VecAssemblyBegin(vs->v[i]);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecAssemblyEnd_Nest"
PetscErrorCode VecAssemblyEnd_Nest(Vec v)
{
  Vec_Nest       *vs = (Vec_Nest*)v->data;
  PetscInt       i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  for (i=0;i<vs->nb;i++) {
    ierr = VecAssemblyEnd(vs->v[i]);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecDestroy_Nest"
PetscErrorCode VecDestroy_Nest(Vec v)
{
  Vec_Nest       *vs = (Vec_Nest*)v->data;
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
#define __FUNCT__ "VecSetUp_Nest"
PetscErrorCode VecSetUp_Nest(Vec V)
{
  Vec_Nest       *ctx = (Vec_Nest*)V->data;
  PetscInt       i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (ctx->setup_called) PetscFunctionReturn(0);

  ctx->nb = V->map->N;
  V->map->n = V->map->N;

  if (ctx->nb < 0) {
    SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_WRONG,"Cannot create VECNEST with < 0 blocks.");
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
#define __FUNCT__ "VecCopy_Nest"
PetscErrorCode VecCopy_Nest(Vec x,Vec y)
{
  Vec_Nest       *bx = (Vec_Nest*)x->data;
  Vec_Nest       *by = (Vec_Nest*)y->data;
  PetscInt       i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  VecNestCheckCompatible2(x,1,y,2);
  for (i=0; i<bx->nb; i++) {
    ierr = VecCopy(bx->v[i],by->v[i]);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/* supports nested blocks */
#undef __FUNCT__  
#define __FUNCT__ "VecDuplicate_Nest"
PetscErrorCode VecDuplicate_Nest(Vec x,Vec *y)
{
  Vec_Nest       *bx = (Vec_Nest*)x->data;
  Vec_Nest       *by;
  PetscInt       i;
  Vec            _y,Y;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecCreate(((PetscObject)x)->comm,&_y);CHKERRQ(ierr);
  ierr = VecSetSizes(_y,bx->nb,bx->nb);CHKERRQ(ierr);
  ierr = VecSetType(_y,VECNEST);CHKERRQ(ierr);

  by = (Vec_Nest*)_y->data;
  for (i=0; i<bx->nb; i++) {
    ierr = VecDuplicate(bx->v[i],&Y);CHKERRQ(ierr);
    ierr = VecNestSetSubVec(_y,i,Y);CHKERRQ(ierr);
    ierr = VecDestroy(Y);CHKERRQ(ierr); /* Hand over control of Y to the nested vector _y */
  }

  *y = _y;
  PetscFunctionReturn(0);
}

/* supports nested blocks */
#undef __FUNCT__  
#define __FUNCT__ "VecDot_Nest"
PetscErrorCode VecDot_Nest(Vec x,Vec y,PetscScalar *val)
{
  Vec_Nest       *bx = (Vec_Nest*)x->data;
  Vec_Nest       *by = (Vec_Nest*)y->data;
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
#define __FUNCT__ "VecTDot_Nest"
PetscErrorCode VecTDot_Nest(Vec x,Vec y,PetscScalar *val)
{
  Vec_Nest       *bx = (Vec_Nest*)x->data;
  Vec_Nest       *by = (Vec_Nest*)y->data;
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
#define __FUNCT__ "VecAXPY_Nest"
PetscErrorCode VecAXPY_Nest(Vec y,PetscScalar alpha,Vec x)
{
  Vec_Nest       *bx = (Vec_Nest*)x->data;
  Vec_Nest       *by = (Vec_Nest*)y->data;
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
#define __FUNCT__ "VecAYPX_Nest"
PetscErrorCode VecAYPX_Nest(Vec y,PetscScalar alpha,Vec x)
{
  Vec_Nest       *bx = (Vec_Nest*)x->data;
  Vec_Nest       *by = (Vec_Nest*)y->data;
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
#define __FUNCT__ "VecAXPBY_Nest"
PetscErrorCode VecAXPBY_Nest(Vec y,PetscScalar alpha,PetscScalar beta,Vec x)
{
  Vec_Nest       *bx = (Vec_Nest*)x->data;
  Vec_Nest       *by = (Vec_Nest*)y->data;
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
#define __FUNCT__ "VecScale_Nest"
PetscErrorCode VecScale_Nest(Vec x,PetscScalar alpha)
{
  Vec_Nest       *bx = (Vec_Nest*)x->data;
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
#define __FUNCT__ "VecPointwiseMult_Nest"
PetscErrorCode VecPointwiseMult_Nest(Vec w,Vec x,Vec y)
{
  Vec_Nest       *bx = (Vec_Nest*)x->data;
  Vec_Nest       *by = (Vec_Nest*)y->data;
  Vec_Nest       *bw = (Vec_Nest*)w->data;
  PetscInt       i,nr;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  VecNestCheckCompatible3(w,1,x,3,y,4);
  nr = bx->nb;
  for (i=0; i<nr; i++) {
    ierr = VecPointwiseMult(bw->v[i],bx->v[i],by->v[i]);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecPointwiseDivide_Nest"
PetscErrorCode VecPointwiseDivide_Nest(Vec w,Vec x,Vec y)
{
  Vec_Nest       *bx = (Vec_Nest*)x->data;
  Vec_Nest       *by = (Vec_Nest*)y->data;
  Vec_Nest       *bw = (Vec_Nest*)w->data;
  PetscInt       i,nr;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  VecNestCheckCompatible3(w,1,x,2,y,3);

  nr = bx->nb;
  for (i=0; i<nr; i++) {
    ierr = VecPointwiseDivide(bw->v[i],bx->v[i],by->v[i]);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecReciprocal_Nest"
PetscErrorCode VecReciprocal_Nest(Vec x)
{
  Vec_Nest       *bx = (Vec_Nest*)x->data;
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
#define __FUNCT__ "VecNorm_Nest"
PetscErrorCode VecNorm_Nest(Vec xin,NormType type,PetscReal* z)
{
  Vec_Nest       *bx = (Vec_Nest*)xin->data;
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
#define __FUNCT__ "VecMAXPY_Nest"
PetscErrorCode VecMAXPY_Nest(Vec y,PetscInt nv,const PetscScalar alpha[],Vec *x)
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
#define __FUNCT__ "VecMDot_Nest"
PetscErrorCode VecMDot_Nest(Vec x,PetscInt nv,const Vec y[],PetscScalar *val)
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
#define __FUNCT__ "VecMTDot_Nest"
PetscErrorCode VecMTDot_Nest(Vec x,PetscInt nv,const Vec y[],PetscScalar *val)
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
#define __FUNCT__ "VecSet_Nest"
PetscErrorCode VecSet_Nest(Vec x,PetscScalar alpha)
{
  Vec_Nest       *bx = (Vec_Nest*)x->data;
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
#define __FUNCT__ "VecConjugate_Nest"
PetscErrorCode VecConjugate_Nest(Vec x)
{
  Vec_Nest       *bx = (Vec_Nest*)x->data;
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
#define __FUNCT__ "VecSwap_Nest"
PetscErrorCode VecSwap_Nest(Vec x,Vec y)
{
  Vec_Nest       *bx = (Vec_Nest*)x->data;
  Vec_Nest       *by = (Vec_Nest*)y->data;
  PetscInt       i,nr;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  VecNestCheckCompatible2(x,1,y,2);
  nr = bx->nb;
  for (i=0; i<nr; i++) {
    ierr = VecSwap(bx->v[i],by->v[i]);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "VecWAXPY_Nest"
PetscErrorCode VecWAXPY_Nest(Vec w,PetscScalar alpha,Vec x,Vec y)
{
  Vec_Nest       *bx = (Vec_Nest*)x->data;
  Vec_Nest       *by = (Vec_Nest*)y->data;
  Vec_Nest       *bw = (Vec_Nest*)w->data;
  PetscInt       i,nr;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  VecNestCheckCompatible3(w,1,x,3,y,4);

  nr = bx->nb;
  for (i=0; i<nr; i++) {
    ierr = VecWAXPY(bw->v[i],alpha,bx->v[i],by->v[i]);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "VecMax_Nest_Recursive"
PetscErrorCode VecMax_Nest_Recursive(Vec x,PetscInt *cnt,PetscInt *p,PetscReal *max)
{
  Vec_Nest  *bx = (Vec_Nest*)x->data;

  PetscInt     i,nr;
  PetscBool    isnest;
  PetscInt     L;
  PetscInt     _entry_loc;
  PetscReal    _entry_val;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscTypeCompare((PetscObject)x,VECNEST,&isnest);CHKERRQ(ierr);
  if (!isnest) {
    /* Not nest */
    ierr = VecMax(x,&_entry_loc,&_entry_val);CHKERRQ(ierr);
    if (_entry_val > *max) {
      *max = _entry_val;
      *p = _entry_loc + *cnt;
    }
    ierr = VecGetSize(x,&L);CHKERRQ(ierr);
    *cnt = *cnt + L;
    PetscFunctionReturn(0);
  }

  /* Otherwise we have a nest */
  bx = (Vec_Nest*)x->data;
  nr = bx->nb;

  /* now descend recursively */
  for (i=0; i<nr; i++) {
    ierr = VecMax_Nest_Recursive(bx->v[i],cnt,p,max);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/* supports nested blocks */
#undef __FUNCT__
#define __FUNCT__ "VecMax_Nest"
PetscErrorCode VecMax_Nest(Vec x,PetscInt *p,PetscReal *max)
{
  PetscInt       cnt;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  cnt = 0;
  *p = 0;
  *max = 0.0;
  ierr = VecMax_Nest_Recursive(x,&cnt,p,max);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "VecMin_Nest_Recursive"
PetscErrorCode VecMin_Nest_Recursive(Vec x,PetscInt *cnt,PetscInt *p,PetscReal *min)
{
  Vec_Nest       *bx = (Vec_Nest*)x->data;
  PetscInt       i,nr,L,_entry_loc;
  PetscBool      isnest;
  PetscReal      _entry_val;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscTypeCompare((PetscObject)x,VECNEST,&isnest);CHKERRQ(ierr);
  if (!isnest) {
    /* Not nest */
    ierr = VecMin(x,&_entry_loc,&_entry_val);CHKERRQ(ierr);
    if (_entry_val < *min) {
      *min = _entry_val;
      *p = _entry_loc + *cnt;
    }
    ierr = VecGetSize(x,&L);CHKERRQ(ierr);
    *cnt = *cnt + L;
    PetscFunctionReturn(0);
  }

  /* Otherwise we have a nest */
  bx = (Vec_Nest*)x->data;
  nr = bx->nb;

  /* now descend recursively */
  for (i=0; i<nr; i++) {
    ierr = VecMin_Nest_Recursive(bx->v[i],cnt,p,min);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "VecMin_Nest"
PetscErrorCode VecMin_Nest(Vec x,PetscInt *p,PetscReal *min)
{
  PetscInt       cnt;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  cnt = 0;
  *p = 0;
  *min = 1.0e308;
  ierr = VecMin_Nest_Recursive(x,&cnt,p,min);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* supports nested blocks */
#undef __FUNCT__
#define __FUNCT__ "VecView_Nest"
PetscErrorCode VecView_Nest(Vec x,PetscViewer viewer)
{
  Vec_Nest       *bx = (Vec_Nest*)x->data;
  PetscBool      isascii;
  PetscInt       i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&isascii);CHKERRQ(ierr);
  if (isascii) {
    ierr = PetscViewerASCIIPrintf(viewer,"Vector Object:\n");CHKERRQ(ierr);
    ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);         /* push0 */
    ierr = PetscViewerASCIIPrintf(viewer,"type=nest, rows=%d \n",bx->nb);CHKERRQ(ierr);

    ierr = PetscViewerASCIIPrintf(viewer,"VecNest  structure: \n");CHKERRQ(ierr);
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
#define __FUNCT__ "VecGetSize_Nest"
PetscErrorCode VecGetSize_Nest(Vec x,PetscInt *size)
{
  Vec_Nest  *bx = (Vec_Nest*)x->data;

  PetscFunctionBegin;
  *size = bx->nb;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "VecMaxPointwiseDivide_Nest"
PetscErrorCode VecMaxPointwiseDivide_Nest(Vec x,Vec y,PetscReal *max)
{
  Vec_Nest       *bx = (Vec_Nest*)x->data;
  Vec_Nest       *by = (Vec_Nest*)y->data;
  PetscInt       i,nr;
  PetscReal      local_max,m;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  VecNestCheckCompatible2(x,1,y,2);
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
