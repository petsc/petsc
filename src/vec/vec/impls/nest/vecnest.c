
#include <../src/vec/vec/impls/nest/vecnestimpl.h> /*I  "petscvec.h"   I*/

/* check all blocks are filled */
static PetscErrorCode VecAssemblyBegin_Nest(Vec v)
{
  Vec_Nest *vs = (Vec_Nest *)v->data;
  PetscInt  i;

  PetscFunctionBegin;
  for (i = 0; i < vs->nb; i++) {
    PetscCheck(vs->v[i], PetscObjectComm((PetscObject)v), PETSC_ERR_SUP, "Nest  vector cannot contain NULL blocks");
    PetscCall(VecAssemblyBegin(vs->v[i]));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode VecAssemblyEnd_Nest(Vec v)
{
  Vec_Nest *vs = (Vec_Nest *)v->data;
  PetscInt  i;

  PetscFunctionBegin;
  for (i = 0; i < vs->nb; i++) PetscCall(VecAssemblyEnd(vs->v[i]));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode VecDestroy_Nest(Vec v)
{
  Vec_Nest *vs = (Vec_Nest *)v->data;
  PetscInt  i;

  PetscFunctionBegin;
  if (vs->v) {
    for (i = 0; i < vs->nb; i++) PetscCall(VecDestroy(&vs->v[i]));
    PetscCall(PetscFree(vs->v));
  }
  for (i = 0; i < vs->nb; i++) PetscCall(ISDestroy(&vs->is[i]));
  PetscCall(PetscFree(vs->is));
  PetscCall(PetscObjectComposeFunction((PetscObject)v, "VecNestGetSubVec_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)v, "VecNestGetSubVecs_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)v, "VecNestSetSubVec_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)v, "VecNestSetSubVecs_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)v, "VecNestGetSize_C", NULL));

  PetscCall(PetscFree(vs));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode VecCopy_Nest(Vec x, Vec y)
{
  Vec_Nest *bx = (Vec_Nest *)x->data;
  Vec_Nest *by = (Vec_Nest *)y->data;
  PetscInt  i;

  PetscFunctionBegin;
  VecNestCheckCompatible2(x, 1, y, 2);
  for (i = 0; i < bx->nb; i++) PetscCall(VecCopy(bx->v[i], by->v[i]));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode VecDuplicate_Nest(Vec x, Vec *y)
{
  Vec_Nest *bx = (Vec_Nest *)x->data;
  Vec       Y;
  Vec      *sub;
  PetscInt  i;

  PetscFunctionBegin;
  PetscCall(PetscMalloc1(bx->nb, &sub));
  for (i = 0; i < bx->nb; i++) PetscCall(VecDuplicate(bx->v[i], &sub[i]));
  PetscCall(VecCreateNest(PetscObjectComm((PetscObject)x), bx->nb, bx->is, sub, &Y));
  for (i = 0; i < bx->nb; i++) PetscCall(VecDestroy(&sub[i]));
  PetscCall(PetscFree(sub));
  *y = Y;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode VecDot_Nest(Vec x, Vec y, PetscScalar *val)
{
  Vec_Nest   *bx = (Vec_Nest *)x->data;
  Vec_Nest   *by = (Vec_Nest *)y->data;
  PetscInt    i, nr;
  PetscScalar x_dot_y, _val;

  PetscFunctionBegin;
  VecNestCheckCompatible2(x, 1, y, 2);
  nr   = bx->nb;
  _val = 0.0;
  for (i = 0; i < nr; i++) {
    PetscCall(VecDot(bx->v[i], by->v[i], &x_dot_y));
    _val = _val + x_dot_y;
  }
  *val = _val;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode VecTDot_Nest(Vec x, Vec y, PetscScalar *val)
{
  Vec_Nest   *bx = (Vec_Nest *)x->data;
  Vec_Nest   *by = (Vec_Nest *)y->data;
  PetscInt    i, nr;
  PetscScalar x_dot_y, _val;

  PetscFunctionBegin;
  VecNestCheckCompatible2(x, 1, y, 2);
  nr   = bx->nb;
  _val = 0.0;
  for (i = 0; i < nr; i++) {
    PetscCall(VecTDot(bx->v[i], by->v[i], &x_dot_y));
    _val = _val + x_dot_y;
  }
  *val = _val;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode VecDotNorm2_Nest(Vec x, Vec y, PetscScalar *dp, PetscScalar *nm)
{
  Vec_Nest   *bx = (Vec_Nest *)x->data;
  Vec_Nest   *by = (Vec_Nest *)y->data;
  PetscInt    i, nr;
  PetscScalar x_dot_y, _dp, _nm;
  PetscReal   norm2_y;

  PetscFunctionBegin;
  VecNestCheckCompatible2(x, 1, y, 2);
  nr  = bx->nb;
  _dp = 0.0;
  _nm = 0.0;
  for (i = 0; i < nr; i++) {
    PetscCall(VecDotNorm2(bx->v[i], by->v[i], &x_dot_y, &norm2_y));
    _dp += x_dot_y;
    _nm += norm2_y;
  }
  *dp = _dp;
  *nm = _nm;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode VecAXPY_Nest(Vec y, PetscScalar alpha, Vec x)
{
  Vec_Nest *bx = (Vec_Nest *)x->data;
  Vec_Nest *by = (Vec_Nest *)y->data;
  PetscInt  i, nr;

  PetscFunctionBegin;
  VecNestCheckCompatible2(y, 1, x, 3);
  nr = bx->nb;
  for (i = 0; i < nr; i++) PetscCall(VecAXPY(by->v[i], alpha, bx->v[i]));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode VecAYPX_Nest(Vec y, PetscScalar alpha, Vec x)
{
  Vec_Nest *bx = (Vec_Nest *)x->data;
  Vec_Nest *by = (Vec_Nest *)y->data;
  PetscInt  i, nr;

  PetscFunctionBegin;
  VecNestCheckCompatible2(y, 1, x, 3);
  nr = bx->nb;
  for (i = 0; i < nr; i++) PetscCall(VecAYPX(by->v[i], alpha, bx->v[i]));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode VecAXPBY_Nest(Vec y, PetscScalar alpha, PetscScalar beta, Vec x)
{
  Vec_Nest *bx = (Vec_Nest *)x->data;
  Vec_Nest *by = (Vec_Nest *)y->data;
  PetscInt  i, nr;

  PetscFunctionBegin;
  VecNestCheckCompatible2(y, 1, x, 4);
  nr = bx->nb;
  for (i = 0; i < nr; i++) PetscCall(VecAXPBY(by->v[i], alpha, beta, bx->v[i]));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode VecAXPBYPCZ_Nest(Vec z, PetscScalar alpha, PetscScalar beta, PetscScalar gamma, Vec x, Vec y)
{
  Vec_Nest *bx = (Vec_Nest *)x->data;
  Vec_Nest *by = (Vec_Nest *)y->data;
  Vec_Nest *bz = (Vec_Nest *)z->data;
  PetscInt  i, nr;

  PetscFunctionBegin;
  VecNestCheckCompatible3(z, 1, x, 5, y, 6);
  nr = bx->nb;
  for (i = 0; i < nr; i++) PetscCall(VecAXPBYPCZ(bz->v[i], alpha, beta, gamma, bx->v[i], by->v[i]));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode VecScale_Nest(Vec x, PetscScalar alpha)
{
  Vec_Nest *bx = (Vec_Nest *)x->data;
  PetscInt  i, nr;

  PetscFunctionBegin;
  nr = bx->nb;
  for (i = 0; i < nr; i++) PetscCall(VecScale(bx->v[i], alpha));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode VecPointwiseMult_Nest(Vec w, Vec x, Vec y)
{
  Vec_Nest *bx = (Vec_Nest *)x->data;
  Vec_Nest *by = (Vec_Nest *)y->data;
  Vec_Nest *bw = (Vec_Nest *)w->data;
  PetscInt  i, nr;

  PetscFunctionBegin;
  VecNestCheckCompatible3(w, 1, x, 2, y, 3);
  nr = bx->nb;
  for (i = 0; i < nr; i++) PetscCall(VecPointwiseMult(bw->v[i], bx->v[i], by->v[i]));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode VecPointwiseDivide_Nest(Vec w, Vec x, Vec y)
{
  Vec_Nest *bx = (Vec_Nest *)x->data;
  Vec_Nest *by = (Vec_Nest *)y->data;
  Vec_Nest *bw = (Vec_Nest *)w->data;
  PetscInt  i, nr;

  PetscFunctionBegin;
  VecNestCheckCompatible3(w, 1, x, 2, y, 3);
  nr = bx->nb;
  for (i = 0; i < nr; i++) PetscCall(VecPointwiseDivide(bw->v[i], bx->v[i], by->v[i]));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode VecReciprocal_Nest(Vec x)
{
  Vec_Nest *bx = (Vec_Nest *)x->data;
  PetscInt  i, nr;

  PetscFunctionBegin;
  nr = bx->nb;
  for (i = 0; i < nr; i++) PetscCall(VecReciprocal(bx->v[i]));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode VecNorm_Nest(Vec xin, NormType type, PetscReal *z)
{
  Vec_Nest *bx = (Vec_Nest *)xin->data;
  PetscInt  i, nr;
  PetscReal z_i;
  PetscReal _z;

  PetscFunctionBegin;
  nr = bx->nb;
  _z = 0.0;

  if (type == NORM_2) {
    PetscScalar dot;
    PetscCall(VecDot(xin, xin, &dot));
    _z = PetscAbsScalar(PetscSqrtScalar(dot));
  } else if (type == NORM_1) {
    for (i = 0; i < nr; i++) {
      PetscCall(VecNorm(bx->v[i], type, &z_i));
      _z = _z + z_i;
    }
  } else if (type == NORM_INFINITY) {
    for (i = 0; i < nr; i++) {
      PetscCall(VecNorm(bx->v[i], type, &z_i));
      if (z_i > _z) _z = z_i;
    }
  }

  *z = _z;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode VecMAXPY_Nest(Vec y, PetscInt nv, const PetscScalar alpha[], Vec *x)
{
  PetscFunctionBegin;
  /* TODO: implement proper MAXPY. For now, do axpy on each vector */
  for (PetscInt v = 0; v < nv; v++) PetscCall(VecAXPY(y, alpha[v], x[v]));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode VecMDot_Nest(Vec x, PetscInt nv, const Vec y[], PetscScalar *val)
{
  PetscFunctionBegin;
  /* TODO: implement proper MDOT. For now, do dot on each vector */
  for (PetscInt j = 0; j < nv; j++) PetscCall(VecDot(x, y[j], &val[j]));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode VecMTDot_Nest(Vec x, PetscInt nv, const Vec y[], PetscScalar *val)
{
  PetscFunctionBegin;
  /* TODO: implement proper MTDOT. For now, do tdot on each vector */
  for (PetscInt j = 0; j < nv; j++) PetscCall(VecTDot(x, y[j], &val[j]));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode VecSet_Nest(Vec x, PetscScalar alpha)
{
  Vec_Nest *bx = (Vec_Nest *)x->data;
  PetscInt  i, nr;

  PetscFunctionBegin;
  nr = bx->nb;
  for (i = 0; i < nr; i++) PetscCall(VecSet(bx->v[i], alpha));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode VecConjugate_Nest(Vec x)
{
  Vec_Nest *bx = (Vec_Nest *)x->data;
  PetscInt  j, nr;

  PetscFunctionBegin;
  nr = bx->nb;
  for (j = 0; j < nr; j++) PetscCall(VecConjugate(bx->v[j]));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode VecSwap_Nest(Vec x, Vec y)
{
  Vec_Nest *bx = (Vec_Nest *)x->data;
  Vec_Nest *by = (Vec_Nest *)y->data;
  PetscInt  i, nr;

  PetscFunctionBegin;
  VecNestCheckCompatible2(x, 1, y, 2);
  nr = bx->nb;
  for (i = 0; i < nr; i++) PetscCall(VecSwap(bx->v[i], by->v[i]));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode VecWAXPY_Nest(Vec w, PetscScalar alpha, Vec x, Vec y)
{
  Vec_Nest *bx = (Vec_Nest *)x->data;
  Vec_Nest *by = (Vec_Nest *)y->data;
  Vec_Nest *bw = (Vec_Nest *)w->data;
  PetscInt  i, nr;

  PetscFunctionBegin;
  VecNestCheckCompatible3(w, 1, x, 3, y, 4);
  nr = bx->nb;
  for (i = 0; i < nr; i++) PetscCall(VecWAXPY(bw->v[i], alpha, bx->v[i], by->v[i]));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode VecMin_Nest(Vec x, PetscInt *p, PetscReal *v)
{
  PetscInt  i, lp = -1, lw = -1;
  PetscReal lv;
  Vec_Nest *bx = (Vec_Nest *)x->data;

  PetscFunctionBegin;
  *v = PETSC_MAX_REAL;
  for (i = 0; i < bx->nb; i++) {
    PetscInt tp;
    PetscCall(VecMin(bx->v[i], &tp, &lv));
    if (lv < *v) {
      *v = lv;
      lw = i;
      lp = tp;
    }
  }
  if (p && lw > -1) {
    PetscInt        st, en;
    const PetscInt *idxs;

    *p = -1;
    PetscCall(VecGetOwnershipRange(bx->v[lw], &st, &en));
    if (lp >= st && lp < en) {
      PetscCall(ISGetIndices(bx->is[lw], &idxs));
      *p = idxs[lp - st];
      PetscCall(ISRestoreIndices(bx->is[lw], &idxs));
    }
    PetscCall(MPIU_Allreduce(MPI_IN_PLACE, p, 1, MPIU_INT, MPI_MAX, PetscObjectComm((PetscObject)x)));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode VecMax_Nest(Vec x, PetscInt *p, PetscReal *v)
{
  PetscInt  i, lp = -1, lw = -1;
  PetscReal lv;
  Vec_Nest *bx = (Vec_Nest *)x->data;

  PetscFunctionBegin;
  *v = PETSC_MIN_REAL;
  for (i = 0; i < bx->nb; i++) {
    PetscInt tp;
    PetscCall(VecMax(bx->v[i], &tp, &lv));
    if (lv > *v) {
      *v = lv;
      lw = i;
      lp = tp;
    }
  }
  if (p && lw > -1) {
    PetscInt        st, en;
    const PetscInt *idxs;

    *p = -1;
    PetscCall(VecGetOwnershipRange(bx->v[lw], &st, &en));
    if (lp >= st && lp < en) {
      PetscCall(ISGetIndices(bx->is[lw], &idxs));
      *p = idxs[lp - st];
      PetscCall(ISRestoreIndices(bx->is[lw], &idxs));
    }
    PetscCall(MPIU_Allreduce(MPI_IN_PLACE, p, 1, MPIU_INT, MPI_MAX, PetscObjectComm((PetscObject)x)));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode VecView_Nest(Vec x, PetscViewer viewer)
{
  Vec_Nest *bx = (Vec_Nest *)x->data;
  PetscBool isascii;
  PetscInt  i;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &isascii));
  if (isascii) {
    PetscCall(PetscViewerASCIIPushTab(viewer));
    PetscCall(PetscViewerASCIIPrintf(viewer, "VecNest, rows=%" PetscInt_FMT ",  structure: \n", bx->nb));
    for (i = 0; i < bx->nb; i++) {
      VecType  type;
      char     name[256] = "", prefix[256] = "";
      PetscInt NR;

      PetscCall(VecGetSize(bx->v[i], &NR));
      PetscCall(VecGetType(bx->v[i], &type));
      if (((PetscObject)bx->v[i])->name) PetscCall(PetscSNPrintf(name, sizeof(name), "name=\"%s\", ", ((PetscObject)bx->v[i])->name));
      if (((PetscObject)bx->v[i])->prefix) PetscCall(PetscSNPrintf(prefix, sizeof(prefix), "prefix=\"%s\", ", ((PetscObject)bx->v[i])->prefix));

      PetscCall(PetscViewerASCIIPrintf(viewer, "(%" PetscInt_FMT ") : %s%stype=%s, rows=%" PetscInt_FMT " \n", i, name, prefix, type, NR));

      PetscCall(PetscViewerASCIIPushTab(viewer)); /* push1 */
      PetscCall(VecView(bx->v[i], viewer));
      PetscCall(PetscViewerASCIIPopTab(viewer)); /* pop1 */
    }
    PetscCall(PetscViewerASCIIPopTab(viewer)); /* pop0 */
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode VecSize_Nest_Recursive(Vec x, PetscBool globalsize, PetscInt *L)
{
  Vec_Nest *bx;
  PetscInt  size, i, nr;
  PetscBool isnest;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)x, VECNEST, &isnest));
  if (!isnest) {
    /* Not nest */
    if (globalsize) PetscCall(VecGetSize(x, &size));
    else PetscCall(VecGetLocalSize(x, &size));
    *L = *L + size;
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  /* Otherwise we have a nest */
  bx = (Vec_Nest *)x->data;
  nr = bx->nb;

  /* now descend recursively */
  for (i = 0; i < nr; i++) PetscCall(VecSize_Nest_Recursive(bx->v[i], globalsize, L));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Returns the sum of the global size of all the constituent vectors in the nest */
static PetscErrorCode VecGetSize_Nest(Vec x, PetscInt *N)
{
  PetscFunctionBegin;
  *N = x->map->N;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Returns the sum of the local size of all the constituent vectors in the nest */
static PetscErrorCode VecGetLocalSize_Nest(Vec x, PetscInt *n)
{
  PetscFunctionBegin;
  *n = x->map->n;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode VecMaxPointwiseDivide_Nest(Vec x, Vec y, PetscReal *max)
{
  Vec_Nest *bx = (Vec_Nest *)x->data;
  Vec_Nest *by = (Vec_Nest *)y->data;
  PetscInt  i, nr;
  PetscReal local_max, m;

  PetscFunctionBegin;
  VecNestCheckCompatible2(x, 1, y, 2);
  nr = bx->nb;
  m  = 0.0;
  for (i = 0; i < nr; i++) {
    PetscCall(VecMaxPointwiseDivide(bx->v[i], by->v[i], &local_max));
    if (local_max > m) m = local_max;
  }
  *max = m;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode VecGetSubVector_Nest(Vec X, IS is, Vec *x)
{
  Vec_Nest *bx = (Vec_Nest *)X->data;
  PetscInt  i;

  PetscFunctionBegin;
  *x = NULL;
  for (i = 0; i < bx->nb; i++) {
    PetscBool issame = PETSC_FALSE;
    PetscCall(ISEqual(is, bx->is[i], &issame));
    if (issame) {
      *x = bx->v[i];
      PetscCall(PetscObjectReference((PetscObject)(*x)));
      break;
    }
  }
  PetscCheck(*x, PetscObjectComm((PetscObject)is), PETSC_ERR_ARG_OUTOFRANGE, "Index set not found in nested Vec");
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode VecRestoreSubVector_Nest(Vec X, IS is, Vec *x)
{
  PetscFunctionBegin;
  PetscCall(PetscObjectStateIncrease((PetscObject)X));
  PetscCall(VecDestroy(x));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode VecGetArray_Nest(Vec X, PetscScalar **x)
{
  Vec_Nest *bx = (Vec_Nest *)X->data;
  PetscInt  i, m, rstart, rend;

  PetscFunctionBegin;
  PetscCall(VecGetOwnershipRange(X, &rstart, &rend));
  PetscCall(VecGetLocalSize(X, &m));
  PetscCall(PetscMalloc1(m, x));
  for (i = 0; i < bx->nb; i++) {
    Vec                subvec = bx->v[i];
    IS                 isy    = bx->is[i];
    PetscInt           j, sm;
    const PetscInt    *ixy;
    const PetscScalar *y;
    PetscCall(VecGetLocalSize(subvec, &sm));
    PetscCall(VecGetArrayRead(subvec, &y));
    PetscCall(ISGetIndices(isy, &ixy));
    for (j = 0; j < sm; j++) {
      PetscInt ix = ixy[j];
      PetscCheck(ix >= rstart && rend > ix, PETSC_COMM_SELF, PETSC_ERR_SUP, "No support for getting array from nonlocal subvec");
      (*x)[ix - rstart] = y[j];
    }
    PetscCall(ISRestoreIndices(isy, &ixy));
    PetscCall(VecRestoreArrayRead(subvec, &y));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode VecRestoreArray_Nest(Vec X, PetscScalar **x)
{
  Vec_Nest *bx = (Vec_Nest *)X->data;
  PetscInt  i, m, rstart, rend;

  PetscFunctionBegin;
  PetscCall(VecGetOwnershipRange(X, &rstart, &rend));
  PetscCall(VecGetLocalSize(X, &m));
  for (i = 0; i < bx->nb; i++) {
    Vec             subvec = bx->v[i];
    IS              isy    = bx->is[i];
    PetscInt        j, sm;
    const PetscInt *ixy;
    PetscScalar    *y;
    PetscCall(VecGetLocalSize(subvec, &sm));
    PetscCall(VecGetArray(subvec, &y));
    PetscCall(ISGetIndices(isy, &ixy));
    for (j = 0; j < sm; j++) {
      PetscInt ix = ixy[j];
      PetscCheck(ix >= rstart && rend > ix, PETSC_COMM_SELF, PETSC_ERR_SUP, "No support for getting array from nonlocal subvec");
      y[j] = (*x)[ix - rstart];
    }
    PetscCall(ISRestoreIndices(isy, &ixy));
    PetscCall(VecRestoreArray(subvec, &y));
  }
  PetscCall(PetscFree(*x));
  PetscCall(PetscObjectStateIncrease((PetscObject)X));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode VecRestoreArrayRead_Nest(Vec X, const PetscScalar **x)
{
  PetscFunctionBegin;
  PetscCall(PetscFree(*x));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode VecConcatenate_Nest(PetscInt nx, const Vec X[], Vec *Y, IS *x_is[])
{
  PetscFunctionBegin;
  PetscCheck(nx <= 0, PetscObjectComm((PetscObject)(*X)), PETSC_ERR_SUP, "VecConcatenate() is not supported for VecNest");
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode VecCreateLocalVector_Nest(Vec v, Vec *w)
{
  Vec      *ww;
  IS       *wis;
  Vec_Nest *bv = (Vec_Nest *)v->data;
  PetscInt  i;

  PetscFunctionBegin;
  PetscCall(PetscMalloc2(bv->nb, &ww, bv->nb, &wis));
  for (i = 0; i < bv->nb; i++) PetscCall(VecCreateLocalVector(bv->v[i], &ww[i]));
  for (i = 0; i < bv->nb; i++) {
    PetscInt off;

    PetscCall(VecGetOwnershipRange(v, &off, NULL));
    PetscCall(ISOnComm(bv->is[i], PetscObjectComm((PetscObject)ww[i]), PETSC_COPY_VALUES, &wis[i]));
    PetscCall(ISShift(wis[i], -off, wis[i]));
  }
  PetscCall(VecCreateNest(PETSC_COMM_SELF, bv->nb, wis, ww, w));
  for (i = 0; i < bv->nb; i++) {
    PetscCall(VecDestroy(&ww[i]));
    PetscCall(ISDestroy(&wis[i]));
  }
  PetscCall(PetscFree2(ww, wis));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode VecGetLocalVector_Nest(Vec v, Vec w)
{
  Vec_Nest *bv = (Vec_Nest *)v->data;
  Vec_Nest *bw = (Vec_Nest *)w->data;
  PetscInt  i;

  PetscFunctionBegin;
  PetscCheckSameType(v, 1, w, 2);
  PetscCheck(bv->nb = bw->nb, PetscObjectComm((PetscObject)w), PETSC_ERR_ARG_WRONG, "Invalid local vector");
  for (i = 0; i < bv->nb; i++) PetscCall(VecGetLocalVector(bv->v[i], bw->v[i]));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode VecRestoreLocalVector_Nest(Vec v, Vec w)
{
  Vec_Nest *bv = (Vec_Nest *)v->data;
  Vec_Nest *bw = (Vec_Nest *)w->data;
  PetscInt  i;

  PetscFunctionBegin;
  PetscCheckSameType(v, 1, w, 2);
  PetscCheck(bv->nb = bw->nb, PetscObjectComm((PetscObject)w), PETSC_ERR_ARG_WRONG, "Invalid local vector");
  for (i = 0; i < bv->nb; i++) PetscCall(VecRestoreLocalVector(bv->v[i], bw->v[i]));
  PetscCall(PetscObjectStateIncrease((PetscObject)v));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode VecGetLocalVectorRead_Nest(Vec v, Vec w)
{
  Vec_Nest *bv = (Vec_Nest *)v->data;
  Vec_Nest *bw = (Vec_Nest *)w->data;
  PetscInt  i;

  PetscFunctionBegin;
  PetscCheckSameType(v, 1, w, 2);
  PetscCheck(bv->nb = bw->nb, PetscObjectComm((PetscObject)w), PETSC_ERR_ARG_WRONG, "Invalid local vector");
  for (i = 0; i < bv->nb; i++) PetscCall(VecGetLocalVectorRead(bv->v[i], bw->v[i]));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode VecRestoreLocalVectorRead_Nest(Vec v, Vec w)
{
  Vec_Nest *bv = (Vec_Nest *)v->data;
  Vec_Nest *bw = (Vec_Nest *)w->data;
  PetscInt  i;

  PetscFunctionBegin;
  PetscCheckSameType(v, 1, w, 2);
  PetscCheck(bv->nb = bw->nb, PetscObjectComm((PetscObject)w), PETSC_ERR_ARG_WRONG, "Invalid local vector");
  for (i = 0; i < bv->nb; i++) PetscCall(VecRestoreLocalVectorRead(bv->v[i], bw->v[i]));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode VecSetRandom_Nest(Vec v, PetscRandom r)
{
  Vec_Nest *bv = (Vec_Nest *)v->data;
  PetscInt  i;

  PetscFunctionBegin;
  for (i = 0; i < bv->nb; i++) PetscCall(VecSetRandom(bv->v[i], r));
  PetscFunctionReturn(PETSC_SUCCESS);
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
  ops->createlocalvector       = VecCreateLocalVector_Nest;
  ops->getlocalvector          = VecGetLocalVector_Nest;
  ops->getlocalvectorread      = VecGetLocalVectorRead_Nest;
  ops->restorelocalvector      = VecRestoreLocalVector_Nest;
  ops->restorelocalvectorread  = VecRestoreLocalVectorRead_Nest;
  ops->setrandom               = VecSetRandom_Nest;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode VecNestGetSubVecs_Private(Vec x, PetscInt m, const PetscInt idxm[], Vec vec[])
{
  Vec_Nest *b = (Vec_Nest *)x->data;
  PetscInt  i;
  PetscInt  row;

  PetscFunctionBegin;
  if (!m) PetscFunctionReturn(PETSC_SUCCESS);
  for (i = 0; i < m; i++) {
    row = idxm[i];
    PetscCheck(row < b->nb, PetscObjectComm((PetscObject)x), PETSC_ERR_ARG_OUTOFRANGE, "Row too large: row %" PetscInt_FMT " max %" PetscInt_FMT, row, b->nb - 1);
    vec[i] = b->v[row];
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode VecNestGetSubVec_Nest(Vec X, PetscInt idxm, Vec *sx)
{
  PetscFunctionBegin;
  PetscCall(PetscObjectStateIncrease((PetscObject)X));
  PetscCall(VecNestGetSubVecs_Private(X, 1, &idxm, sx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
 VecNestGetSubVec - Returns a single, sub-vector from a nest vector.

 Not Collective

 Input Parameters:
+  X  - nest vector
-  idxm - index of the vector within the nest

 Output Parameter:
.  sx - vector at index `idxm` within the nest

 Level: developer

.seealso: `VECNEST`,  [](chapter_vectors), `Vec`, `VecType`, `VecNestGetSize()`, `VecNestGetSubVecs()`
@*/
PetscErrorCode VecNestGetSubVec(Vec X, PetscInt idxm, Vec *sx)
{
  PetscFunctionBegin;
  PetscUseMethod(X, "VecNestGetSubVec_C", (Vec, PetscInt, Vec *), (X, idxm, sx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode VecNestGetSubVecs_Nest(Vec X, PetscInt *N, Vec **sx)
{
  Vec_Nest *b = (Vec_Nest *)X->data;

  PetscFunctionBegin;
  PetscCall(PetscObjectStateIncrease((PetscObject)X));
  if (N) *N = b->nb;
  if (sx) *sx = b->v;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
 VecNestGetSubVecs - Returns the entire array of vectors defining a nest vector.

 Not Collective

 Input Parameter:
.  X  - nest vector

 Output Parameters:
+  N - number of nested vecs
-  sx - array of vectors

 Level: developer

 Note:
 The user should not free the array `sx`.

 Fortran Note:
 The caller must allocate the array to hold the subvectors.

.seealso: `VECNEST`,  [](chapter_vectors), `Vec`, `VecType`, `VecNestGetSize()`, `VecNestGetSubVec()`
@*/
PetscErrorCode VecNestGetSubVecs(Vec X, PetscInt *N, Vec **sx)
{
  PetscFunctionBegin;
  PetscUseMethod(X, "VecNestGetSubVecs_C", (Vec, PetscInt *, Vec **), (X, N, sx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode VecNestSetSubVec_Private(Vec X, PetscInt idxm, Vec x)
{
  Vec_Nest *bx = (Vec_Nest *)X->data;
  PetscInt  i, offset = 0, n = 0, bs;
  IS        is;
  PetscBool issame = PETSC_FALSE;
  PetscInt  N      = 0;

  PetscFunctionBegin;
  /* check if idxm < bx->nb */
  PetscCheck(idxm < bx->nb, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Out of range index value %" PetscInt_FMT " maximum %" PetscInt_FMT, idxm, bx->nb);

  PetscCall(VecDestroy(&bx->v[idxm]));      /* destroy the existing vector */
  PetscCall(VecDuplicate(x, &bx->v[idxm])); /* duplicate the layout of given vector */
  PetscCall(VecCopy(x, bx->v[idxm]));       /* copy the contents of the given vector */

  /* check if we need to update the IS for the block */
  offset = X->map->rstart;
  for (i = 0; i < idxm; i++) {
    n = 0;
    PetscCall(VecGetLocalSize(bx->v[i], &n));
    offset += n;
  }

  /* get the local size and block size */
  PetscCall(VecGetLocalSize(x, &n));
  PetscCall(VecGetBlockSize(x, &bs));

  /* create the new IS */
  PetscCall(ISCreateStride(PetscObjectComm((PetscObject)x), n, offset, 1, &is));
  PetscCall(ISSetBlockSize(is, bs));

  /* check if they are equal */
  PetscCall(ISEqual(is, bx->is[idxm], &issame));

  if (!issame) {
    /* The IS of given vector has a different layout compared to the existing block vector.
     Destroy the existing reference and update the IS. */
    PetscCall(ISDestroy(&bx->is[idxm]));
    PetscCall(ISDuplicate(is, &bx->is[idxm]));
    PetscCall(ISCopy(is, bx->is[idxm]));

    offset += n;
    /* Since the current IS[idxm] changed, we need to update all the subsequent IS */
    for (i = idxm + 1; i < bx->nb; i++) {
      /* get the local size and block size */
      PetscCall(VecGetLocalSize(bx->v[i], &n));
      PetscCall(VecGetBlockSize(bx->v[i], &bs));

      /* destroy the old and create the new IS */
      PetscCall(ISDestroy(&bx->is[i]));
      PetscCall(ISCreateStride(((PetscObject)bx->v[i])->comm, n, offset, 1, &bx->is[i]));
      PetscCall(ISSetBlockSize(bx->is[i], bs));

      offset += n;
    }

    n = 0;
    PetscCall(VecSize_Nest_Recursive(X, PETSC_TRUE, &N));
    PetscCall(VecSize_Nest_Recursive(X, PETSC_FALSE, &n));
    PetscCall(PetscLayoutSetSize(X->map, N));
    PetscCall(PetscLayoutSetLocalSize(X->map, n));
  }

  PetscCall(ISDestroy(&is));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode VecNestSetSubVec_Nest(Vec X, PetscInt idxm, Vec sx)
{
  PetscFunctionBegin;
  PetscCall(PetscObjectStateIncrease((PetscObject)X));
  PetscCall(VecNestSetSubVec_Private(X, idxm, sx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
   VecNestSetSubVec - Set a single component vector in a nest vector at specified index.

   Not Collective

   Input Parameters:
+  X  - nest vector
.  idxm - index of the vector within the nest vector
-  sx - vector at index `idxm` within the nest vector

   Level: developer

   Note:
   The new vector `sx` does not have to be of same size as X[idxm]. Arbitrary vector layouts are allowed.

.seealso: `VECNEST`,  [](chapter_vectors), `Vec`, `VecType`, `VecNestSetSubVecs()`, `VecNestGetSubVec()`
@*/
PetscErrorCode VecNestSetSubVec(Vec X, PetscInt idxm, Vec sx)
{
  PetscFunctionBegin;
  PetscUseMethod(X, "VecNestSetSubVec_C", (Vec, PetscInt, Vec), (X, idxm, sx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode VecNestSetSubVecs_Nest(Vec X, PetscInt N, PetscInt *idxm, Vec *sx)
{
  PetscInt i;

  PetscFunctionBegin;
  PetscCall(PetscObjectStateIncrease((PetscObject)X));
  for (i = 0; i < N; i++) PetscCall(VecNestSetSubVec_Private(X, idxm[i], sx[i]));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
   VecNestSetSubVecs - Sets the component vectors at the specified indices in a nest vector.

   Not Collective

   Input Parameters:
+  X  - nest vector
.  N - number of component vecs in `sx`
.  idxm - indices of component vectors that are to be replaced
-  sx - array of vectors

   Level: developer

   Note:
   The components in the vector array `sx` do not have to be of the same size as corresponding
   components in `X`. The user can also free the array `sx` after the call.

.seealso: `VECNEST`,  [](chapter_vectors), `Vec`, `VecType`, `VecNestGetSize()`, `VecNestGetSubVec()`
@*/
PetscErrorCode VecNestSetSubVecs(Vec X, PetscInt N, PetscInt *idxm, Vec *sx)
{
  PetscFunctionBegin;
  PetscUseMethod(X, "VecNestSetSubVecs_C", (Vec, PetscInt, PetscInt *, Vec *), (X, N, idxm, sx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode VecNestGetSize_Nest(Vec X, PetscInt *N)
{
  Vec_Nest *b = (Vec_Nest *)X->data;

  PetscFunctionBegin;
  *N = b->nb;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
 VecNestGetSize - Returns the size of the nest vector.

 Not Collective

 Input Parameter:
.  X  - nest vector

 Output Parameter:
.  N - number of nested vecs

 Level: developer

.seealso: `VECNEST`,  [](chapter_vectors), `Vec`, `VecType`, `VecNestGetSubVec()`, `VecNestGetSubVecs()`
@*/
PetscErrorCode VecNestGetSize(Vec X, PetscInt *N)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(X, VEC_CLASSID, 1);
  PetscValidIntPointer(N, 2);
  PetscUseMethod(X, "VecNestGetSize_C", (Vec, PetscInt *), (X, N));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode VecSetUp_Nest_Private(Vec V, PetscInt nb, Vec x[])
{
  Vec_Nest *ctx = (Vec_Nest *)V->data;
  PetscInt  i;

  PetscFunctionBegin;
  if (ctx->setup_called) PetscFunctionReturn(PETSC_SUCCESS);

  ctx->nb = nb;
  PetscCheck(ctx->nb >= 0, PetscObjectComm((PetscObject)V), PETSC_ERR_ARG_WRONG, "Cannot create VECNEST with < 0 blocks.");

  /* Create space */
  PetscCall(PetscMalloc1(ctx->nb, &ctx->v));
  for (i = 0; i < ctx->nb; i++) {
    ctx->v[i] = x[i];
    PetscCall(PetscObjectReference((PetscObject)x[i]));
    /* Do not allocate memory for internal sub blocks */
  }

  PetscCall(PetscMalloc1(ctx->nb, &ctx->is));

  ctx->setup_called = PETSC_TRUE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode VecSetUp_NestIS_Private(Vec V, PetscInt nb, IS is[])
{
  Vec_Nest *ctx = (Vec_Nest *)V->data;
  PetscInt  i, offset, m, n, M, N;

  PetscFunctionBegin;
  if (is) { /* Do some consistency checks and reference the is */
    offset = V->map->rstart;
    for (i = 0; i < ctx->nb; i++) {
      PetscCall(ISGetSize(is[i], &M));
      PetscCall(VecGetSize(ctx->v[i], &N));
      PetscCheck(M == N, PetscObjectComm((PetscObject)V), PETSC_ERR_ARG_INCOMP, "In slot %" PetscInt_FMT ", IS of size %" PetscInt_FMT " is not compatible with Vec of size %" PetscInt_FMT, i, M, N);
      PetscCall(ISGetLocalSize(is[i], &m));
      PetscCall(VecGetLocalSize(ctx->v[i], &n));
      PetscCheck(m == n, PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, "In slot %" PetscInt_FMT ", IS of local size %" PetscInt_FMT " is not compatible with Vec of local size %" PetscInt_FMT, i, m, n);
      PetscCall(PetscObjectReference((PetscObject)is[i]));
      ctx->is[i] = is[i];
      offset += n;
    }
  } else { /* Create a contiguous ISStride for each entry */
    offset = V->map->rstart;
    for (i = 0; i < ctx->nb; i++) {
      PetscInt bs;
      PetscCall(VecGetLocalSize(ctx->v[i], &n));
      PetscCall(VecGetBlockSize(ctx->v[i], &bs));
      PetscCall(ISCreateStride(((PetscObject)ctx->v[i])->comm, n, offset, 1, &ctx->is[i]));
      PetscCall(ISSetBlockSize(ctx->is[i], bs));
      offset += n;
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
  VECNEST - VECNEST = "nest" - Vector type consisting of nested subvectors, each stored separately.

  Level: intermediate

  Notes:
  This vector type reduces the number of copies for certain solvers applied to multi-physics problems.
  It is usually used with `MATNEST` and `DMCOMPOSITE` via `DMSetVecType()`.

.seealso: [](chapter_vectors), `Vec`, `VecType`, `VecCreate()`, `VecType`, `VecCreateNest()`, `MatCreateNest()`
M*/

/*@C
   VecCreateNest - Creates a new vector containing several nested subvectors, each stored separately

   Collective

   Input Parameters:
+  comm - Communicator for the new `Vec`
.  nb - number of nested blocks
.  is - array of `nb` index sets describing each nested block, or `NULL` to pack subvectors contiguously
-  x - array of `nb` sub-vectors

   Output Parameter:
.  Y - new vector

   Level: advanced

.seealso: `VECNEST`,  [](chapter_vectors), `Vec`, `VecType`, `VecCreate()`, `MatCreateNest()`, `DMSetVecType()`, `VECNEST`
@*/
PetscErrorCode VecCreateNest(MPI_Comm comm, PetscInt nb, IS is[], Vec x[], Vec *Y)
{
  Vec       V;
  Vec_Nest *s;
  PetscInt  n, N;

  PetscFunctionBegin;
  PetscCall(VecCreate(comm, &V));
  PetscCall(PetscObjectChangeTypeName((PetscObject)V, VECNEST));

  /* allocate and set pointer for implementation data */
  PetscCall(PetscNew(&s));
  V->data         = (void *)s;
  s->setup_called = PETSC_FALSE;
  s->nb           = -1;
  s->v            = NULL;

  PetscCall(VecSetUp_Nest_Private(V, nb, x));

  n = N = 0;
  PetscCall(VecSize_Nest_Recursive(V, PETSC_TRUE, &N));
  PetscCall(VecSize_Nest_Recursive(V, PETSC_FALSE, &n));
  PetscCall(PetscLayoutSetSize(V->map, N));
  PetscCall(PetscLayoutSetLocalSize(V->map, n));
  PetscCall(PetscLayoutSetBlockSize(V->map, 1));
  PetscCall(PetscLayoutSetUp(V->map));

  PetscCall(VecSetUp_NestIS_Private(V, nb, is));

  PetscCall(VecNestSetOps_Private(V->ops));
  V->petscnative = PETSC_FALSE;

  /* expose block api's */
  PetscCall(PetscObjectComposeFunction((PetscObject)V, "VecNestGetSubVec_C", VecNestGetSubVec_Nest));
  PetscCall(PetscObjectComposeFunction((PetscObject)V, "VecNestGetSubVecs_C", VecNestGetSubVecs_Nest));
  PetscCall(PetscObjectComposeFunction((PetscObject)V, "VecNestSetSubVec_C", VecNestSetSubVec_Nest));
  PetscCall(PetscObjectComposeFunction((PetscObject)V, "VecNestSetSubVecs_C", VecNestSetSubVecs_Nest));
  PetscCall(PetscObjectComposeFunction((PetscObject)V, "VecNestGetSize_C", VecNestGetSize_Nest));

  *Y = V;
  PetscFunctionReturn(PETSC_SUCCESS);
}
