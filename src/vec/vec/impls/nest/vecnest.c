
#include <../src/vec/vec/impls/nest/vecnestimpl.h>   /*I  "petscvec.h"   I*/

/* check all blocks are filled */
static PetscErrorCode VecAssemblyBegin_Nest(Vec v)
{
  Vec_Nest       *vs = (Vec_Nest*)v->data;
  PetscInt       i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  for (i=0; i<vs->nb; i++) {
    if (!vs->v[i]) SETERRQ(PetscObjectComm((PetscObject)v),PETSC_ERR_SUP,"Nest  vector cannot contain NULL blocks");
    ierr = VecAssemblyBegin(vs->v[i]);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode VecAssemblyEnd_Nest(Vec v)
{
  Vec_Nest       *vs = (Vec_Nest*)v->data;
  PetscInt       i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  for (i=0; i<vs->nb; i++) {
    ierr = VecAssemblyEnd(vs->v[i]);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode VecDestroy_Nest(Vec v)
{
  Vec_Nest       *vs = (Vec_Nest*)v->data;
  PetscInt       i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (vs->v) {
    for (i=0; i<vs->nb; i++) {
      ierr = VecDestroy(&vs->v[i]);CHKERRQ(ierr);
    }
    ierr = PetscFree(vs->v);CHKERRQ(ierr);
  }
  for (i=0; i<vs->nb; i++) {
    ierr = ISDestroy(&vs->is[i]);CHKERRQ(ierr);
  }
  ierr = PetscFree(vs->is);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)v,"",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)v,"",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)v,"",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)v,"",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)v,"",NULL);CHKERRQ(ierr);

  ierr = PetscFree(vs);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* supports nested blocks */
static PetscErrorCode VecCopy_Nest(Vec x,Vec y)
{
  Vec_Nest       *bx = (Vec_Nest*)x->data;
  Vec_Nest       *by = (Vec_Nest*)y->data;
  PetscInt       i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscCheckTypeName(y,VECNEST);
  VecNestCheckCompatible2(x,1,y,2);
  for (i=0; i<bx->nb; i++) {
    ierr = VecCopy(bx->v[i],by->v[i]);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/* supports nested blocks */
static PetscErrorCode VecDuplicate_Nest(Vec x,Vec *y)
{
  Vec_Nest       *bx = (Vec_Nest*)x->data;
  Vec            Y;
  Vec            *sub;
  PetscInt       i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscMalloc1(bx->nb,&sub);CHKERRQ(ierr);
  for (i=0; i<bx->nb; i++) {
    ierr = VecDuplicate(bx->v[i],&sub[i]);CHKERRQ(ierr);
  }
  ierr = VecCreateNest(PetscObjectComm((PetscObject)x),bx->nb,bx->is,sub,&Y);CHKERRQ(ierr);
  for (i=0; i<bx->nb; i++) {
    ierr = VecDestroy(&sub[i]);CHKERRQ(ierr);
  }
  ierr = PetscFree(sub);CHKERRQ(ierr);
  *y   = Y;
  PetscFunctionReturn(0);
}

/* supports nested blocks */
static PetscErrorCode VecDot_Nest(Vec x,Vec y,PetscScalar *val)
{
  Vec_Nest       *bx = (Vec_Nest*)x->data;
  Vec_Nest       *by = (Vec_Nest*)y->data;
  PetscInt       i,nr;
  PetscScalar    x_dot_y,_val;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  nr   = bx->nb;
  _val = 0.0;
  for (i=0; i<nr; i++) {
    ierr = VecDot(bx->v[i],by->v[i],&x_dot_y);CHKERRQ(ierr);
    _val = _val + x_dot_y;
  }
  *val = _val;
  PetscFunctionReturn(0);
}

/* supports nested blocks */
static PetscErrorCode VecTDot_Nest(Vec x,Vec y,PetscScalar *val)
{
  Vec_Nest       *bx = (Vec_Nest*)x->data;
  Vec_Nest       *by = (Vec_Nest*)y->data;
  PetscInt       i,nr;
  PetscScalar    x_dot_y,_val;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  nr   = bx->nb;
  _val = 0.0;
  for (i=0; i<nr; i++) {
    ierr = VecTDot(bx->v[i],by->v[i],&x_dot_y);CHKERRQ(ierr);
    _val = _val + x_dot_y;
  }
  *val = _val;
  PetscFunctionReturn(0);
}

static PetscErrorCode VecDotNorm2_Nest(Vec x,Vec y,PetscScalar *dp, PetscScalar *nm)
{
  Vec_Nest       *bx = (Vec_Nest*)x->data;
  Vec_Nest       *by = (Vec_Nest*)y->data;
  PetscInt       i,nr;
  PetscScalar    x_dot_y,_dp,_nm;
  PetscReal      norm2_y;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  nr  = bx->nb;
  _dp = 0.0;
  _nm = 0.0;
  for (i=0; i<nr; i++) {
    ierr = VecDotNorm2(bx->v[i],by->v[i],&x_dot_y,&norm2_y);CHKERRQ(ierr);
    _dp += x_dot_y;
    _nm += norm2_y;
  }
  *dp = _dp;
  *nm = _nm;
  PetscFunctionReturn(0);
}

static PetscErrorCode VecAXPY_Nest(Vec y,PetscScalar alpha,Vec x)
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

static PetscErrorCode VecAYPX_Nest(Vec y,PetscScalar alpha,Vec x)
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

static PetscErrorCode VecAXPBY_Nest(Vec y,PetscScalar alpha,PetscScalar beta,Vec x)
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

static PetscErrorCode VecAXPBYPCZ_Nest(Vec z,PetscScalar alpha,PetscScalar beta,PetscScalar gamma,Vec x,Vec y)
{
  Vec_Nest       *bx = (Vec_Nest*)x->data;
  Vec_Nest       *by = (Vec_Nest*)y->data;
  Vec_Nest       *bz = (Vec_Nest*)z->data;
  PetscInt       i,nr;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  nr = bx->nb;
  for (i=0; i<nr; i++) {
    ierr = VecAXPBYPCZ(bz->v[i],alpha,beta,gamma,bx->v[i],by->v[i]);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode VecScale_Nest(Vec x,PetscScalar alpha)
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

static PetscErrorCode VecPointwiseMult_Nest(Vec w,Vec x,Vec y)
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
    ierr = VecPointwiseMult(bw->v[i],bx->v[i],by->v[i]);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode VecPointwiseDivide_Nest(Vec w,Vec x,Vec y)
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

static PetscErrorCode VecReciprocal_Nest(Vec x)
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

static PetscErrorCode VecNorm_Nest(Vec xin,NormType type,PetscReal *z)
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
    ierr = VecDot(xin,xin,&dot);CHKERRQ(ierr);
    _z = PetscAbsScalar(PetscSqrtScalar(dot));
  } else if (type == NORM_1) {
    for (i=0; i<nr; i++) {
      ierr = VecNorm(bx->v[i],type,&z_i);CHKERRQ(ierr);
      _z = _z + z_i;
    }
  } else if (type == NORM_INFINITY) {
    for (i=0; i<nr; i++) {
      ierr = VecNorm(bx->v[i],type,&z_i);CHKERRQ(ierr);
      if (z_i > _z) _z = z_i;
    }
  }

  *z = _z;
  PetscFunctionReturn(0);
}

static PetscErrorCode VecMAXPY_Nest(Vec y,PetscInt nv,const PetscScalar alpha[],Vec *x)
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

static PetscErrorCode VecMDot_Nest(Vec x,PetscInt nv,const Vec y[],PetscScalar *val)
{
  PetscInt       j;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  for (j=0; j<nv; j++) {
    ierr = VecDot(x,y[j],&val[j]);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode VecMTDot_Nest(Vec x,PetscInt nv,const Vec y[],PetscScalar *val)
{
  PetscInt       j;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  for (j=0; j<nv; j++) {
    ierr = VecTDot(x,y[j],&val[j]);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode VecSet_Nest(Vec x,PetscScalar alpha)
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

static PetscErrorCode VecConjugate_Nest(Vec x)
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

static PetscErrorCode VecSwap_Nest(Vec x,Vec y)
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

static PetscErrorCode VecWAXPY_Nest(Vec w,PetscScalar alpha,Vec x,Vec y)
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

static PetscErrorCode VecMax_Nest_Recursive(Vec x,PetscInt *cnt,PetscInt *p,PetscReal *max)
{
  Vec_Nest       *bx;
  PetscInt       i,nr;
  PetscBool      isnest;
  PetscInt       L;
  PetscInt       _entry_loc;
  PetscReal      _entry_val;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)x,VECNEST,&isnest);CHKERRQ(ierr);
  if (!isnest) {
    /* Not nest */
    ierr = VecMax(x,&_entry_loc,&_entry_val);CHKERRQ(ierr);
    if (_entry_val > *max) {
      *max = _entry_val;
      if (p) *p = _entry_loc + *cnt;
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
static PetscErrorCode VecMax_Nest(Vec x,PetscInt *p,PetscReal *max)
{
  PetscInt       cnt;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  cnt  = 0;
  if (p) *p = 0;
  *max = PETSC_MIN_REAL;
  ierr = VecMax_Nest_Recursive(x,&cnt,p,max);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode VecMin_Nest_Recursive(Vec x,PetscInt *cnt,PetscInt *p,PetscReal *min)
{
  Vec_Nest       *bx = (Vec_Nest*)x->data;
  PetscInt       i,nr,L,_entry_loc;
  PetscBool      isnest;
  PetscReal      _entry_val;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)x,VECNEST,&isnest);CHKERRQ(ierr);
  if (!isnest) {
    /* Not nest */
    ierr = VecMin(x,&_entry_loc,&_entry_val);CHKERRQ(ierr);
    if (_entry_val < *min) {
      *min = _entry_val;
      if (p) *p = _entry_loc + *cnt;
    }
    ierr = VecGetSize(x,&L);CHKERRQ(ierr);
    *cnt = *cnt + L;
    PetscFunctionReturn(0);
  }

  /* Otherwise we have a nest */
  nr = bx->nb;

  /* now descend recursively */
  for (i=0; i<nr; i++) {
    ierr = VecMin_Nest_Recursive(bx->v[i],cnt,p,min);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode VecMin_Nest(Vec x,PetscInt *p,PetscReal *min)
{
  PetscInt       cnt;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  cnt  = 0;
  if (p) *p = 0;
  *min = PETSC_MAX_REAL;
  ierr = VecMin_Nest_Recursive(x,&cnt,p,min);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* supports nested blocks */
static PetscErrorCode VecView_Nest(Vec x,PetscViewer viewer)
{
  Vec_Nest       *bx = (Vec_Nest*)x->data;
  PetscBool      isascii;
  PetscInt       i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&isascii);CHKERRQ(ierr);
  if (isascii) {
    ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"VecNest, rows=%" PetscInt_FMT ",  structure: \n",bx->nb);CHKERRQ(ierr);
    for (i=0; i<bx->nb; i++) {
      VecType  type;
      char     name[256] = "",prefix[256] = "";
      PetscInt NR;

      ierr = VecGetSize(bx->v[i],&NR);CHKERRQ(ierr);
      ierr = VecGetType(bx->v[i],&type);CHKERRQ(ierr);
      if (((PetscObject)bx->v[i])->name) {ierr = PetscSNPrintf(name,sizeof(name),"name=\"%s\", ",((PetscObject)bx->v[i])->name);CHKERRQ(ierr);}
      if (((PetscObject)bx->v[i])->prefix) {ierr = PetscSNPrintf(prefix,sizeof(prefix),"prefix=\"%s\", ",((PetscObject)bx->v[i])->prefix);CHKERRQ(ierr);}

      ierr = PetscViewerASCIIPrintf(viewer,"(%" PetscInt_FMT ") : %s%stype=%s, rows=%" PetscInt_FMT " \n",i,name,prefix,type,NR);CHKERRQ(ierr);

      ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);             /* push1 */
      ierr = VecView(bx->v[i],viewer);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);              /* pop1 */
    }
    ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);                /* pop0 */
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode VecSize_Nest_Recursive(Vec x,PetscBool globalsize,PetscInt *L)
{
  Vec_Nest       *bx;
  PetscInt       size,i,nr;
  PetscBool      isnest;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)x,VECNEST,&isnest);CHKERRQ(ierr);
  if (!isnest) {
    /* Not nest */
    if (globalsize) { ierr = VecGetSize(x,&size);CHKERRQ(ierr); }
    else {            ierr = VecGetLocalSize(x,&size);CHKERRQ(ierr); }
    *L = *L + size;
    PetscFunctionReturn(0);
  }

  /* Otherwise we have a nest */
  bx = (Vec_Nest*)x->data;
  nr = bx->nb;

  /* now descend recursively */
  for (i=0; i<nr; i++) {
    ierr = VecSize_Nest_Recursive(bx->v[i],globalsize,L);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/* Returns the sum of the global size of all the consituent vectors in the nest */
static PetscErrorCode VecGetSize_Nest(Vec x,PetscInt *N)
{
  PetscFunctionBegin;
  *N = x->map->N;
  PetscFunctionReturn(0);
}

/* Returns the sum of the local size of all the consituent vectors in the nest */
static PetscErrorCode VecGetLocalSize_Nest(Vec x,PetscInt *n)
{
  PetscFunctionBegin;
  *n = x->map->n;
  PetscFunctionReturn(0);
}

static PetscErrorCode VecMaxPointwiseDivide_Nest(Vec x,Vec y,PetscReal *max)
{
  Vec_Nest       *bx = (Vec_Nest*)x->data;
  Vec_Nest       *by = (Vec_Nest*)y->data;
  PetscInt       i,nr;
  PetscReal      local_max,m;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  VecNestCheckCompatible2(x,1,y,2);
  nr = bx->nb;
  m  = 0.0;
  for (i=0; i<nr; i++) {
    ierr = VecMaxPointwiseDivide(bx->v[i],by->v[i],&local_max);CHKERRQ(ierr);
    if (local_max > m) m = local_max;
  }
  *max = m;
  PetscFunctionReturn(0);
}

static PetscErrorCode  VecGetSubVector_Nest(Vec X,IS is,Vec *x)
{
  Vec_Nest       *bx = (Vec_Nest*)X->data;
  PetscInt       i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  *x = NULL;
  for (i=0; i<bx->nb; i++) {
    PetscBool issame = PETSC_FALSE;
    ierr = ISEqual(is,bx->is[i],&issame);CHKERRQ(ierr);
    if (issame) {
      *x   = bx->v[i];
      ierr = PetscObjectReference((PetscObject)(*x));CHKERRQ(ierr);
      break;
    }
  }
  if (!*x) SETERRQ(PetscObjectComm((PetscObject)is),PETSC_ERR_ARG_OUTOFRANGE,"Index set not found in nested Vec");
  PetscFunctionReturn(0);
}

static PetscErrorCode  VecRestoreSubVector_Nest(Vec X,IS is,Vec *x)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecDestroy(x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode VecGetArray_Nest(Vec X,PetscScalar **x)
{
  Vec_Nest       *bx = (Vec_Nest*)X->data;
  PetscInt       i,m,rstart,rend;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecGetOwnershipRange(X,&rstart,&rend);CHKERRQ(ierr);
  ierr = VecGetLocalSize(X,&m);CHKERRQ(ierr);
  ierr = PetscMalloc1(m,x);CHKERRQ(ierr);
  for (i=0; i<bx->nb; i++) {
    Vec               subvec = bx->v[i];
    IS                isy    = bx->is[i];
    PetscInt          j,sm;
    const PetscInt    *ixy;
    const PetscScalar *y;
    ierr = VecGetLocalSize(subvec,&sm);CHKERRQ(ierr);
    ierr = VecGetArrayRead(subvec,&y);CHKERRQ(ierr);
    ierr = ISGetIndices(isy,&ixy);CHKERRQ(ierr);
    for (j=0; j<sm; j++) {
      PetscInt ix = ixy[j];
      if (ix < rstart || rend <= ix) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"No support for getting array from nonlocal subvec");
      (*x)[ix-rstart] = y[j];
    }
    ierr = ISRestoreIndices(isy,&ixy);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(subvec,&y);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode VecRestoreArray_Nest(Vec X,PetscScalar **x)
{
  Vec_Nest       *bx = (Vec_Nest*)X->data;
  PetscInt       i,m,rstart,rend;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecGetOwnershipRange(X,&rstart,&rend);CHKERRQ(ierr);
  ierr = VecGetLocalSize(X,&m);CHKERRQ(ierr);
  for (i=0; i<bx->nb; i++) {
    Vec            subvec = bx->v[i];
    IS             isy    = bx->is[i];
    PetscInt       j,sm;
    const PetscInt *ixy;
    PetscScalar    *y;
    ierr = VecGetLocalSize(subvec,&sm);CHKERRQ(ierr);
    ierr = VecGetArray(subvec,&y);CHKERRQ(ierr);
    ierr = ISGetIndices(isy,&ixy);CHKERRQ(ierr);
    for (j=0; j<sm; j++) {
      PetscInt ix = ixy[j];
      if (ix < rstart || rend <= ix) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"No support for getting array from nonlocal subvec");
      y[j] = (*x)[ix-rstart];
    }
    ierr = ISRestoreIndices(isy,&ixy);CHKERRQ(ierr);
    ierr = VecRestoreArray(subvec,&y);CHKERRQ(ierr);
  }
  ierr = PetscFree(*x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode VecRestoreArrayRead_Nest(Vec X,const PetscScalar **x)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFree(*x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode VecConcatenate_Nest(PetscInt nx, const Vec X[], Vec *Y, IS *x_is[])
{
  PetscFunctionBegin;
  if (nx > 0) SETERRQ(PetscObjectComm((PetscObject)(*X)), PETSC_ERR_SUP, "VecConcatenate() is not supported for VecNest");
  PetscFunctionReturn(0);
}

static PetscErrorCode VecNestSetOps_Private(struct _VecOps *ops)
{
  PetscFunctionBegin;
  ops->duplicate               = VecDuplicate_Nest;
  ops->duplicatevecs           = VecDuplicateVecs_Default;
  ops->destroyvecs             = VecDestroyVecs_Default;
  ops->dot                     = VecDot_Nest;
  ops->mdot                    = VecMDot_Nest;
  ops->norm                    = VecNorm_Nest;
  ops->tdot                    = VecTDot_Nest;
  ops->mtdot                   = VecMTDot_Nest;
  ops->scale                   = VecScale_Nest;
  ops->copy                    = VecCopy_Nest;
  ops->set                     = VecSet_Nest;
  ops->swap                    = VecSwap_Nest;
  ops->axpy                    = VecAXPY_Nest;
  ops->axpby                   = VecAXPBY_Nest;
  ops->maxpy                   = VecMAXPY_Nest;
  ops->aypx                    = VecAYPX_Nest;
  ops->waxpy                   = VecWAXPY_Nest;
  ops->axpbypcz                = NULL;
  ops->pointwisemult           = VecPointwiseMult_Nest;
  ops->pointwisedivide         = VecPointwiseDivide_Nest;
  ops->setvalues               = NULL;
  ops->assemblybegin           = VecAssemblyBegin_Nest;
  ops->assemblyend             = VecAssemblyEnd_Nest;
  ops->getarray                = VecGetArray_Nest;
  ops->getsize                 = VecGetSize_Nest;
  ops->getlocalsize            = VecGetLocalSize_Nest;
  ops->restorearray            = VecRestoreArray_Nest;
  ops->restorearrayread        = VecRestoreArrayRead_Nest;
  ops->max                     = VecMax_Nest;
  ops->min                     = VecMin_Nest;
  ops->setrandom               = NULL;
  ops->setoption               = NULL;
  ops->setvaluesblocked        = NULL;
  ops->destroy                 = VecDestroy_Nest;
  ops->view                    = VecView_Nest;
  ops->placearray              = NULL;
  ops->replacearray            = NULL;
  ops->dot_local               = VecDot_Nest;
  ops->tdot_local              = VecTDot_Nest;
  ops->norm_local              = VecNorm_Nest;
  ops->mdot_local              = VecMDot_Nest;
  ops->mtdot_local             = VecMTDot_Nest;
  ops->load                    = NULL;
  ops->reciprocal              = VecReciprocal_Nest;
  ops->conjugate               = VecConjugate_Nest;
  ops->setlocaltoglobalmapping = NULL;
  ops->setvalueslocal          = NULL;
  ops->resetarray              = NULL;
  ops->setfromoptions          = NULL;
  ops->maxpointwisedivide      = VecMaxPointwiseDivide_Nest;
  ops->load                    = NULL;
  ops->pointwisemax            = NULL;
  ops->pointwisemaxabs         = NULL;
  ops->pointwisemin            = NULL;
  ops->getvalues               = NULL;
  ops->sqrt                    = NULL;
  ops->abs                     = NULL;
  ops->exp                     = NULL;
  ops->shift                   = NULL;
  ops->create                  = NULL;
  ops->stridegather            = NULL;
  ops->stridescatter           = NULL;
  ops->dotnorm2                = VecDotNorm2_Nest;
  ops->getsubvector            = VecGetSubVector_Nest;
  ops->restoresubvector        = VecRestoreSubVector_Nest;
  ops->axpbypcz                = VecAXPBYPCZ_Nest;
  ops->concatenate             = VecConcatenate_Nest;
  PetscFunctionReturn(0);
}

static PetscErrorCode VecNestGetSubVecs_Private(Vec x,PetscInt m,const PetscInt idxm[],Vec vec[])
{
  Vec_Nest *b = (Vec_Nest*)x->data;
  PetscInt i;
  PetscInt row;

  PetscFunctionBegin;
  if (!m) PetscFunctionReturn(0);
  for (i=0; i<m; i++) {
    row = idxm[i];
    if (row >= b->nb) SETERRQ2(PetscObjectComm((PetscObject)x),PETSC_ERR_ARG_OUTOFRANGE,"Row too large: row %" PetscInt_FMT " max %" PetscInt_FMT "",row,b->nb-1);
    vec[i] = b->v[row];
  }
  PetscFunctionReturn(0);
}

PetscErrorCode  VecNestGetSubVec_Nest(Vec X,PetscInt idxm,Vec *sx)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecNestGetSubVecs_Private(X,1,&idxm,sx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
 VecNestGetSubVec - Returns a single, sub-vector from a nest vector.

 Not collective

 Input Parameters:
+  X  - nest vector
-  idxm - index of the vector within the nest

 Output Parameter:
.  sx - vector at index idxm within the nest

 Notes:

 Level: developer

.seealso: VecNestGetSize(), VecNestGetSubVecs()
@*/
PetscErrorCode  VecNestGetSubVec(Vec X,PetscInt idxm,Vec *sx)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscUseMethod(X,"VecNestGetSubVec_C",(Vec,PetscInt,Vec*),(X,idxm,sx));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode  VecNestGetSubVecs_Nest(Vec X,PetscInt *N,Vec **sx)
{
  Vec_Nest *b = (Vec_Nest*)X->data;

  PetscFunctionBegin;
  if (N)  *N  = b->nb;
  if (sx) *sx = b->v;
  PetscFunctionReturn(0);
}

/*@C
 VecNestGetSubVecs - Returns the entire array of vectors defining a nest vector.

 Not collective

 Input Parameter:
.  X  - nest vector

 Output Parameters:
+  N - number of nested vecs
-  sx - array of vectors

 Notes:
 The user should not free the array sx.

 Fortran Notes:
 The caller must allocate the array to hold the subvectors.

 Level: developer

.seealso: VecNestGetSize(), VecNestGetSubVec()
@*/
PetscErrorCode  VecNestGetSubVecs(Vec X,PetscInt *N,Vec **sx)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscUseMethod(X,"VecNestGetSubVecs_C",(Vec,PetscInt*,Vec**),(X,N,sx));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode  VecNestSetSubVec_Private(Vec X,PetscInt idxm,Vec x)
{
  Vec_Nest       *bx = (Vec_Nest*)X->data;
  PetscInt       i,offset=0,n=0,bs;
  IS             is;
  PetscErrorCode ierr;
  PetscBool      issame = PETSC_FALSE;
  PetscInt       N=0;

  /* check if idxm < bx->nb */
  if (idxm >= bx->nb) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Out of range index value %" PetscInt_FMT " maximum %" PetscInt_FMT "",idxm,bx->nb);

  PetscFunctionBegin;
  ierr = VecDestroy(&bx->v[idxm]);CHKERRQ(ierr);       /* destroy the existing vector */
  ierr = VecDuplicate(x,&bx->v[idxm]);CHKERRQ(ierr);   /* duplicate the layout of given vector */
  ierr = VecCopy(x,bx->v[idxm]);CHKERRQ(ierr);         /* copy the contents of the given vector */

  /* check if we need to update the IS for the block */
  offset = X->map->rstart;
  for (i=0; i<idxm; i++) {
    n=0;
    ierr = VecGetLocalSize(bx->v[i],&n);CHKERRQ(ierr);
    offset += n;
  }

  /* get the local size and block size */
  ierr = VecGetLocalSize(x,&n);CHKERRQ(ierr);
  ierr = VecGetBlockSize(x,&bs);CHKERRQ(ierr);

  /* create the new IS */
  ierr = ISCreateStride(PetscObjectComm((PetscObject)x),n,offset,1,&is);CHKERRQ(ierr);
  ierr = ISSetBlockSize(is,bs);CHKERRQ(ierr);

  /* check if they are equal */
  ierr = ISEqual(is,bx->is[idxm],&issame);CHKERRQ(ierr);

  if (!issame) {
    /* The IS of given vector has a different layout compared to the existing block vector.
     Destroy the existing reference and update the IS. */
    ierr = ISDestroy(&bx->is[idxm]);CHKERRQ(ierr);
    ierr = ISDuplicate(is,&bx->is[idxm]);CHKERRQ(ierr);
    ierr = ISCopy(is,bx->is[idxm]);CHKERRQ(ierr);

    offset += n;
    /* Since the current IS[idxm] changed, we need to update all the subsequent IS */
    for (i=idxm+1; i<bx->nb; i++) {
      /* get the local size and block size */
      ierr = VecGetLocalSize(bx->v[i],&n);CHKERRQ(ierr);
      ierr = VecGetBlockSize(bx->v[i],&bs);CHKERRQ(ierr);

      /* destroy the old and create the new IS */
      ierr = ISDestroy(&bx->is[i]);CHKERRQ(ierr);
      ierr = ISCreateStride(((PetscObject)bx->v[i])->comm,n,offset,1,&bx->is[i]);CHKERRQ(ierr);
      ierr = ISSetBlockSize(bx->is[i],bs);CHKERRQ(ierr);

      offset += n;
    }

    n=0;
    ierr = VecSize_Nest_Recursive(X,PETSC_TRUE,&N);CHKERRQ(ierr);
    ierr = VecSize_Nest_Recursive(X,PETSC_FALSE,&n);CHKERRQ(ierr);
    ierr = PetscLayoutSetSize(X->map,N);CHKERRQ(ierr);
    ierr = PetscLayoutSetLocalSize(X->map,n);CHKERRQ(ierr);
  }

  ierr = ISDestroy(&is);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode  VecNestSetSubVec_Nest(Vec X,PetscInt idxm,Vec sx)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecNestSetSubVec_Private(X,idxm,sx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
   VecNestSetSubVec - Set a single component vector in a nest vector at specified index.

   Not collective

   Input Parameters:
+  X  - nest vector
.  idxm - index of the vector within the nest vector
-  sx - vector at index idxm within the nest vector

   Notes:
   The new vector sx does not have to be of same size as X[idxm]. Arbitrary vector layouts are allowed.

   Level: developer

.seealso: VecNestSetSubVecs(), VecNestGetSubVec()
@*/
PetscErrorCode  VecNestSetSubVec(Vec X,PetscInt idxm,Vec sx)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscUseMethod(X,"VecNestSetSubVec_C",(Vec,PetscInt,Vec),(X,idxm,sx));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode  VecNestSetSubVecs_Nest(Vec X,PetscInt N,PetscInt *idxm,Vec *sx)
{
  PetscInt       i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  for (i=0; i<N; i++) {
    ierr = VecNestSetSubVec_Private(X,idxm[i],sx[i]);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*@C
   VecNestSetSubVecs - Sets the component vectors at the specified indices in a nest vector.

   Not collective

   Input Parameters:
+  X  - nest vector
.  N - number of component vecs in sx
.  idxm - indices of component vecs that are to be replaced
-  sx - array of vectors

   Notes:
   The components in the vector array sx do not have to be of the same size as corresponding
   components in X. The user can also free the array "sx" after the call.

   Level: developer

.seealso: VecNestGetSize(), VecNestGetSubVec()
@*/
PetscErrorCode  VecNestSetSubVecs(Vec X,PetscInt N,PetscInt *idxm,Vec *sx)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscUseMethod(X,"VecNestSetSubVecs_C",(Vec,PetscInt,PetscInt*,Vec*),(X,N,idxm,sx));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode  VecNestGetSize_Nest(Vec X,PetscInt *N)
{
  Vec_Nest *b = (Vec_Nest*)X->data;

  PetscFunctionBegin;
  *N = b->nb;
  PetscFunctionReturn(0);
}

/*@
 VecNestGetSize - Returns the size of the nest vector.

 Not collective

 Input Parameter:
.  X  - nest vector

 Output Parameter:
.  N - number of nested vecs

 Notes:

 Level: developer

.seealso: VecNestGetSubVec(), VecNestGetSubVecs()
@*/
PetscErrorCode  VecNestGetSize(Vec X,PetscInt *N)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(X,VEC_CLASSID,1);
  PetscValidIntPointer(N,2);
  ierr = PetscUseMethod(X,"VecNestGetSize_C",(Vec,PetscInt*),(X,N));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode VecSetUp_Nest_Private(Vec V,PetscInt nb,Vec x[])
{
  Vec_Nest       *ctx = (Vec_Nest*)V->data;
  PetscInt       i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (ctx->setup_called) PetscFunctionReturn(0);

  ctx->nb = nb;
  if (ctx->nb < 0) SETERRQ(PetscObjectComm((PetscObject)V),PETSC_ERR_ARG_WRONG,"Cannot create VECNEST with < 0 blocks.");

  /* Create space */
  ierr = PetscMalloc1(ctx->nb,&ctx->v);CHKERRQ(ierr);
  for (i=0; i<ctx->nb; i++) {
    ctx->v[i] = x[i];
    ierr = PetscObjectReference((PetscObject)x[i]);CHKERRQ(ierr);
    /* Do not allocate memory for internal sub blocks */
  }

  ierr = PetscMalloc1(ctx->nb,&ctx->is);CHKERRQ(ierr);

  ctx->setup_called = PETSC_TRUE;
  PetscFunctionReturn(0);
}

static PetscErrorCode VecSetUp_NestIS_Private(Vec V,PetscInt nb,IS is[])
{
  Vec_Nest       *ctx = (Vec_Nest*)V->data;
  PetscInt       i,offset,m,n,M,N;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (is) {                     /* Do some consistency checks and reference the is */
    offset = V->map->rstart;
    for (i=0; i<ctx->nb; i++) {
      ierr = ISGetSize(is[i],&M);CHKERRQ(ierr);
      ierr = VecGetSize(ctx->v[i],&N);CHKERRQ(ierr);
      if (M != N) SETERRQ3(PetscObjectComm((PetscObject)V),PETSC_ERR_ARG_INCOMP,"In slot %" PetscInt_FMT ", IS of size %" PetscInt_FMT " is not compatible with Vec of size %" PetscInt_FMT "",i,M,N);
      ierr = ISGetLocalSize(is[i],&m);CHKERRQ(ierr);
      ierr = VecGetLocalSize(ctx->v[i],&n);CHKERRQ(ierr);
      if (m != n) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"In slot %" PetscInt_FMT ", IS of local size %" PetscInt_FMT " is not compatible with Vec of local size %" PetscInt_FMT "",i,m,n);
      if (PetscDefined(USE_DEBUG)) { /* This test can be expensive */
        PetscInt  start;
        PetscBool contiguous;
        ierr = ISContiguousLocal(is[i],offset,offset+n,&start,&contiguous);CHKERRQ(ierr);
        if (!contiguous) SETERRQ1(PetscObjectComm((PetscObject)V),PETSC_ERR_SUP,"Index set %" PetscInt_FMT " is not contiguous with layout of matching vector",i);
        if (start != 0) SETERRQ1(PetscObjectComm((PetscObject)V),PETSC_ERR_SUP,"Index set %" PetscInt_FMT " introduces overlap or a hole",i);
      }
      ierr = PetscObjectReference((PetscObject)is[i]);CHKERRQ(ierr);
      ctx->is[i] = is[i];
      offset += n;
    }
  } else {                      /* Create a contiguous ISStride for each entry */
    offset = V->map->rstart;
    for (i=0; i<ctx->nb; i++) {
      PetscInt bs;
      ierr = VecGetLocalSize(ctx->v[i],&n);CHKERRQ(ierr);
      ierr = VecGetBlockSize(ctx->v[i],&bs);CHKERRQ(ierr);
      ierr = ISCreateStride(((PetscObject)ctx->v[i])->comm,n,offset,1,&ctx->is[i]);CHKERRQ(ierr);
      ierr = ISSetBlockSize(ctx->is[i],bs);CHKERRQ(ierr);
      offset += n;
    }
  }
  PetscFunctionReturn(0);
}

/*@C
   VecCreateNest - Creates a new vector containing several nested subvectors, each stored separately

   Collective on Vec

   Input Parameters:
+  comm - Communicator for the new Vec
.  nb - number of nested blocks
.  is - array of nb index sets describing each nested block, or NULL to pack subvectors contiguously
-  x - array of nb sub-vectors

   Output Parameter:
.  Y - new vector

   Level: advanced

.seealso: VecCreate(), MatCreateNest(), DMSetVecType(), VECNEST
@*/
PetscErrorCode  VecCreateNest(MPI_Comm comm,PetscInt nb,IS is[],Vec x[],Vec *Y)
{
  Vec            V;
  Vec_Nest       *s;
  PetscInt       n,N;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecCreate(comm,&V);CHKERRQ(ierr);
  ierr = PetscObjectChangeTypeName((PetscObject)V,VECNEST);CHKERRQ(ierr);

  /* allocate and set pointer for implememtation data */
  ierr = PetscNew(&s);CHKERRQ(ierr);
  V->data          = (void*)s;
  s->setup_called  = PETSC_FALSE;
  s->nb            = -1;
  s->v             = NULL;

  ierr = VecSetUp_Nest_Private(V,nb,x);CHKERRQ(ierr);

  n = N = 0;
  ierr = VecSize_Nest_Recursive(V,PETSC_TRUE,&N);CHKERRQ(ierr);
  ierr = VecSize_Nest_Recursive(V,PETSC_FALSE,&n);CHKERRQ(ierr);
  ierr = PetscLayoutSetSize(V->map,N);CHKERRQ(ierr);
  ierr = PetscLayoutSetLocalSize(V->map,n);CHKERRQ(ierr);
  ierr = PetscLayoutSetBlockSize(V->map,1);CHKERRQ(ierr);
  ierr = PetscLayoutSetUp(V->map);CHKERRQ(ierr);

  ierr = VecSetUp_NestIS_Private(V,nb,is);CHKERRQ(ierr);

  ierr = VecNestSetOps_Private(V->ops);CHKERRQ(ierr);
  V->petscnative = PETSC_FALSE;

  /* expose block api's */
  ierr = PetscObjectComposeFunction((PetscObject)V,"VecNestGetSubVec_C",VecNestGetSubVec_Nest);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)V,"VecNestGetSubVecs_C",VecNestGetSubVecs_Nest);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)V,"VecNestSetSubVec_C",VecNestSetSubVec_Nest);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)V,"VecNestSetSubVecs_C",VecNestSetSubVecs_Nest);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)V,"VecNestGetSize_C",VecNestGetSize_Nest);CHKERRQ(ierr);

  *Y = V;
  PetscFunctionReturn(0);
}

/*MC
  VECNEST - VECNEST = "nest" - Vector type consisting of nested subvectors, each stored separately.

  Level: intermediate

  Notes:
  This vector type reduces the number of copies for certain solvers applied to multi-physics problems.
  It is usually used with MATNEST and DMComposite via DMSetVecType().

.seealso: VecCreate(), VecType, VecCreateNest(), MatCreateNest()
M*/
