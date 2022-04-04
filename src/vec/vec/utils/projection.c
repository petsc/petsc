#include <petsc/private/vecimpl.h>    /*I   "petscvec.h"  I*/

/*@
  VecWhichEqual - Creates an index set containing the indices
             where the vectors Vec1 and Vec2 have identical elements.

  Collective on Vec

  Input Parameters:
. Vec1, Vec2 - the two vectors to compare

  OutputParameter:
. S - The index set containing the indices i where vec1[i] == vec2[i]

  Notes:
    the two vectors must have the same parallel layout

  Level: advanced
@*/
PetscErrorCode VecWhichEqual(Vec Vec1, Vec Vec2, IS *S)
{
  PetscInt          i,n_same=0;
  PetscInt          n,low,high;
  PetscInt          *same=NULL;
  const PetscScalar *v1,*v2;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(Vec1,VEC_CLASSID,1);
  PetscValidHeaderSpecific(Vec2,VEC_CLASSID,2);
  PetscCheckSameComm(Vec1,1,Vec2,2);
  VecCheckSameSize(Vec1,1,Vec2,2);

  PetscCall(VecGetOwnershipRange(Vec1,&low,&high));
  PetscCall(VecGetLocalSize(Vec1,&n));
  if (n>0) {
    if (Vec1 == Vec2) {
      PetscCall(VecGetArrayRead(Vec1,&v1));
      v2=v1;
    } else {
      PetscCall(VecGetArrayRead(Vec1,&v1));
      PetscCall(VecGetArrayRead(Vec2,&v2));
    }

    PetscCall(PetscMalloc1(n,&same));

    for (i=0; i<n; ++i) {
      if (v1[i] == v2[i]) {same[n_same]=low+i; ++n_same;}
    }

    if (Vec1 == Vec2) {
      PetscCall(VecRestoreArrayRead(Vec1,&v1));
    } else {
      PetscCall(VecRestoreArrayRead(Vec1,&v1));
      PetscCall(VecRestoreArrayRead(Vec2,&v2));
    }
  }
  PetscCall(ISCreateGeneral(PetscObjectComm((PetscObject)Vec1),n_same,same,PETSC_OWN_POINTER,S));
  PetscFunctionReturn(0);
}

/*@
  VecWhichLessThan - Creates an index set containing the indices
  where the vectors Vec1 < Vec2

  Collective on S

  Input Parameters:
. Vec1, Vec2 - the two vectors to compare

  OutputParameter:
. S - The index set containing the indices i where vec1[i] < vec2[i]

  Notes:
  The two vectors must have the same parallel layout

  For complex numbers this only compares the real part

  Level: advanced
@*/
PetscErrorCode VecWhichLessThan(Vec Vec1, Vec Vec2, IS *S)
{
  PetscInt          i,n_lt=0;
  PetscInt          n,low,high;
  PetscInt          *lt=NULL;
  const PetscScalar *v1,*v2;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(Vec1,VEC_CLASSID,1);
  PetscValidHeaderSpecific(Vec2,VEC_CLASSID,2);
  PetscCheckSameComm(Vec1,1,Vec2,2);
  VecCheckSameSize(Vec1,1,Vec2,2);

  PetscCall(VecGetOwnershipRange(Vec1,&low,&high));
  PetscCall(VecGetLocalSize(Vec1,&n));
  if (n>0) {
    if (Vec1 == Vec2) {
      PetscCall(VecGetArrayRead(Vec1,&v1));
      v2=v1;
    } else {
      PetscCall(VecGetArrayRead(Vec1,&v1));
      PetscCall(VecGetArrayRead(Vec2,&v2));
    }

    PetscCall(PetscMalloc1(n,&lt));

    for (i=0; i<n; ++i) {
      if (PetscRealPart(v1[i]) < PetscRealPart(v2[i])) {lt[n_lt]=low+i; ++n_lt;}
    }

    if (Vec1 == Vec2) {
      PetscCall(VecRestoreArrayRead(Vec1,&v1));
    } else {
      PetscCall(VecRestoreArrayRead(Vec1,&v1));
      PetscCall(VecRestoreArrayRead(Vec2,&v2));
    }
  }
  PetscCall(ISCreateGeneral(PetscObjectComm((PetscObject)Vec1),n_lt,lt,PETSC_OWN_POINTER,S));
  PetscFunctionReturn(0);
}

/*@
  VecWhichGreaterThan - Creates an index set containing the indices
  where the vectors Vec1 > Vec2

  Collective on S

  Input Parameters:
. Vec1, Vec2 - the two vectors to compare

  OutputParameter:
. S - The index set containing the indices i where vec1[i] > vec2[i]

  Notes:
  The two vectors must have the same parallel layout

  For complex numbers this only compares the real part

  Level: advanced
@*/
PetscErrorCode VecWhichGreaterThan(Vec Vec1, Vec Vec2, IS *S)
{
  PetscInt          i,n_gt=0;
  PetscInt          n,low,high;
  PetscInt          *gt=NULL;
  const PetscScalar *v1,*v2;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(Vec1,VEC_CLASSID,1);
  PetscValidHeaderSpecific(Vec2,VEC_CLASSID,2);
  PetscCheckSameComm(Vec1,1,Vec2,2);
  VecCheckSameSize(Vec1,1,Vec2,2);

  PetscCall(VecGetOwnershipRange(Vec1,&low,&high));
  PetscCall(VecGetLocalSize(Vec1,&n));
  if (n>0) {
    if (Vec1 == Vec2) {
      PetscCall(VecGetArrayRead(Vec1,&v1));
      v2=v1;
    } else {
      PetscCall(VecGetArrayRead(Vec1,&v1));
      PetscCall(VecGetArrayRead(Vec2,&v2));
    }

    PetscCall(PetscMalloc1(n,&gt));

    for (i=0; i<n; ++i) {
      if (PetscRealPart(v1[i]) > PetscRealPart(v2[i])) {gt[n_gt]=low+i; ++n_gt;}
    }

    if (Vec1 == Vec2) {
      PetscCall(VecRestoreArrayRead(Vec1,&v1));
    } else {
      PetscCall(VecRestoreArrayRead(Vec1,&v1));
      PetscCall(VecRestoreArrayRead(Vec2,&v2));
    }
  }
  PetscCall(ISCreateGeneral(PetscObjectComm((PetscObject)Vec1),n_gt,gt,PETSC_OWN_POINTER,S));
  PetscFunctionReturn(0);
}

/*@
  VecWhichBetween - Creates an index set containing the indices
               where  VecLow < V < VecHigh

  Collective on S

  Input Parameters:
+ VecLow - lower bound
. V - Vector to compare
- VecHigh - higher bound

  OutputParameter:
. S - The index set containing the indices i where veclow[i] < v[i] < vechigh[i]

  Notes:
  The vectors must have the same parallel layout

  For complex numbers this only compares the real part

  Level: advanced
@*/
PetscErrorCode VecWhichBetween(Vec VecLow, Vec V, Vec VecHigh, IS *S)
{

  PetscInt          i,n_vm=0;
  PetscInt          n,low,high;
  PetscInt          *vm=NULL;
  const PetscScalar *v1,*v2,*vmiddle;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(VecLow,VEC_CLASSID,1);
  PetscValidHeaderSpecific(V,VEC_CLASSID,2);
  PetscValidHeaderSpecific(VecHigh,VEC_CLASSID,3);
  PetscCheckSameComm(V,2,VecLow,1);
  PetscCheckSameComm(V,2,VecHigh,3);
  VecCheckSameSize(V,2,VecLow,1);
  VecCheckSameSize(V,2,VecHigh,3);

  PetscCall(VecGetOwnershipRange(VecLow,&low,&high));
  PetscCall(VecGetLocalSize(VecLow,&n));
  if (n>0) {
    PetscCall(VecGetArrayRead(VecLow,&v1));
    if (VecLow != VecHigh) {
      PetscCall(VecGetArrayRead(VecHigh,&v2));
    } else {
      v2=v1;
    }
    if (V != VecLow && V != VecHigh) {
      PetscCall(VecGetArrayRead(V,&vmiddle));
    } else if (V==VecLow) {
      vmiddle=v1;
    } else {
      vmiddle=v2;
    }

    PetscCall(PetscMalloc1(n,&vm));

    for (i=0; i<n; ++i) {
      if (PetscRealPart(v1[i]) < PetscRealPart(vmiddle[i]) && PetscRealPart(vmiddle[i]) < PetscRealPart(v2[i])) {vm[n_vm]=low+i; ++n_vm;}
    }

    PetscCall(VecRestoreArrayRead(VecLow,&v1));
    if (VecLow != VecHigh) {
      PetscCall(VecRestoreArrayRead(VecHigh,&v2));
    }
    if (V != VecLow && V != VecHigh) {
      PetscCall(VecRestoreArrayRead(V,&vmiddle));
    }
  }
  PetscCall(ISCreateGeneral(PetscObjectComm((PetscObject)V),n_vm,vm,PETSC_OWN_POINTER,S));
  PetscFunctionReturn(0);
}

/*@
  VecWhichBetweenOrEqual - Creates an index set containing the indices
  where  VecLow <= V <= VecHigh

  Collective on S

  Input Parameters:
+ VecLow - lower bound
. V - Vector to compare
- VecHigh - higher bound

  OutputParameter:
. S - The index set containing the indices i where veclow[i] <= v[i] <= vechigh[i]

  Level: advanced
@*/

PetscErrorCode VecWhichBetweenOrEqual(Vec VecLow, Vec V, Vec VecHigh, IS * S)
{
  PetscInt          i,n_vm=0;
  PetscInt          n,low,high;
  PetscInt          *vm=NULL;
  const PetscScalar *v1,*v2,*vmiddle;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(VecLow,VEC_CLASSID,1);
  PetscValidHeaderSpecific(V,VEC_CLASSID,2);
  PetscValidHeaderSpecific(VecHigh,VEC_CLASSID,3);
  PetscCheckSameComm(V,2,VecLow,1);
  PetscCheckSameComm(V,2,VecHigh,3);
  VecCheckSameSize(V,2,VecLow,1);
  VecCheckSameSize(V,2,VecHigh,3);

  PetscCall(VecGetOwnershipRange(VecLow,&low,&high));
  PetscCall(VecGetLocalSize(VecLow,&n));
  if (n>0) {
    PetscCall(VecGetArrayRead(VecLow,&v1));
    if (VecLow != VecHigh) {
      PetscCall(VecGetArrayRead(VecHigh,&v2));
    } else {
      v2=v1;
    }
    if (V != VecLow && V != VecHigh) {
      PetscCall(VecGetArrayRead(V,&vmiddle));
    } else if (V==VecLow) {
      vmiddle=v1;
    } else {
      vmiddle =v2;
    }

    PetscCall(PetscMalloc1(n,&vm));

    for (i=0; i<n; ++i) {
      if (PetscRealPart(v1[i]) <= PetscRealPart(vmiddle[i]) && PetscRealPart(vmiddle[i]) <= PetscRealPart(v2[i])) {vm[n_vm]=low+i; ++n_vm;}
    }

    PetscCall(VecRestoreArrayRead(VecLow,&v1));
    if (VecLow != VecHigh) {
      PetscCall(VecRestoreArrayRead(VecHigh,&v2));
    }
    if (V != VecLow && V != VecHigh) {
      PetscCall(VecRestoreArrayRead(V,&vmiddle));
    }
  }
  PetscCall(ISCreateGeneral(PetscObjectComm((PetscObject)V),n_vm,vm,PETSC_OWN_POINTER,S));
  PetscFunctionReturn(0);
}

/*@
   VecWhichInactive - Creates an index set containing the indices
  where one of the following holds:
    a) VecLow(i)  < V(i) < VecHigh(i)
    b) VecLow(i)  = V(i) and D(i) <= 0 (< 0 when Strong is true)
    c) VecHigh(i) = V(i) and D(i) >= 0 (> 0 when Strong is true)

  Collective on S

  Input Parameters:
+ VecLow - lower bound
. V - Vector to compare
. D - Direction to compare
. VecHigh - higher bound
- Strong - indicator for applying strongly inactive test

  OutputParameter:
. S - The index set containing the indices i where the bound is inactive

  Level: advanced
@*/

PetscErrorCode VecWhichInactive(Vec VecLow, Vec V, Vec D, Vec VecHigh, PetscBool Strong, IS * S)
{
  PetscInt          i,n_vm=0;
  PetscInt          n,low,high;
  PetscInt          *vm=NULL;
  const PetscScalar *v1,*v2,*v,*d;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(VecLow,VEC_CLASSID,1);
  PetscValidHeaderSpecific(V,VEC_CLASSID,2);
  PetscValidHeaderSpecific(D,VEC_CLASSID,3);
  PetscValidHeaderSpecific(VecHigh,VEC_CLASSID,4);
  PetscCheckSameComm(V,2,VecLow,1);
  PetscCheckSameComm(V,2,D,3);
  PetscCheckSameComm(V,2,VecHigh,4);
  VecCheckSameSize(V,2,VecLow,1);
  VecCheckSameSize(V,2,D,3);
  VecCheckSameSize(V,2,VecHigh,4);

  PetscCall(VecGetOwnershipRange(VecLow,&low,&high));
  PetscCall(VecGetLocalSize(VecLow,&n));
  if (n>0) {
    PetscCall(VecGetArrayRead(VecLow,&v1));
    if (VecLow != VecHigh) {
      PetscCall(VecGetArrayRead(VecHigh,&v2));
    } else {
      v2=v1;
    }
    if (V != VecLow && V != VecHigh) {
      PetscCall(VecGetArrayRead(V,&v));
    } else if (V==VecLow) {
      v = v1;
    } else {
      v = v2;
    }
    if (D != VecLow && D != VecHigh && D != V) {
      PetscCall(VecGetArrayRead(D,&d));
    } else if (D==VecLow) {
      d = v1;
    } else if (D==VecHigh) {
      d = v2;
    } else {
      d = v;
    }

    PetscCall(PetscMalloc1(n,&vm));

    if (Strong) {
      for (i=0; i<n; ++i) {
        if (PetscRealPart(v1[i]) < PetscRealPart(v[i]) && PetscRealPart(v[i]) < PetscRealPart(v2[i])) {
          vm[n_vm]=low+i; ++n_vm;
        } else if (PetscRealPart(v1[i]) == PetscRealPart(v[i]) && PetscRealPart(d[i]) < 0) {
          vm[n_vm]=low+i; ++n_vm;
        } else if (PetscRealPart(v2[i]) == PetscRealPart(v[i]) && PetscRealPart(d[i]) > 0) {
          vm[n_vm]=low+i; ++n_vm;
        }
      }
    } else {
      for (i=0; i<n; ++i) {
        if (PetscRealPart(v1[i]) < PetscRealPart(v[i]) && PetscRealPart(v[i]) < PetscRealPart(v2[i])) {
          vm[n_vm]=low+i; ++n_vm;
        } else if (PetscRealPart(v1[i]) == PetscRealPart(v[i]) && PetscRealPart(d[i]) <= 0) {
          vm[n_vm]=low+i; ++n_vm;
        } else if (PetscRealPart(v2[i]) == PetscRealPart(v[i]) && PetscRealPart(d[i]) >= 0) {
          vm[n_vm]=low+i; ++n_vm;
        }
      }
    }

    PetscCall(VecRestoreArrayRead(VecLow,&v1));
    if (VecLow != VecHigh) {
      PetscCall(VecRestoreArrayRead(VecHigh,&v2));
    }
    if (V != VecLow && V != VecHigh) {
      PetscCall(VecRestoreArrayRead(V,&v));
    }
    if (D != VecLow && D != VecHigh && D != V) {
      PetscCall(VecRestoreArrayRead(D,&d));
    }
  }
  PetscCall(ISCreateGeneral(PetscObjectComm((PetscObject)V),n_vm,vm,PETSC_OWN_POINTER,S));
  PetscFunctionReturn(0);
}

/*@
  VecISAXPY - Adds a reduced vector to the appropriate elements of a full-space vector.
                  vfull[is[i]] += alpha*vreduced[i]

  Input Parameters:
+ vfull    - the full-space vector
. is       - the index set for the reduced space
. alpha    - the scalar coefficient
- vreduced - the reduced-space vector

  Output Parameters:
. vfull    - the sum of the full-space vector and reduced-space vector

  Notes:
    The index set identifies entries in the global vector.
    Negative indices are skipped; indices outside the ownership range of vfull will raise an error.

  Level: advanced

.seealso:  VecAXPY(), VecGetOwnershipRange()
@*/
PetscErrorCode VecISAXPY(Vec vfull, IS is, PetscScalar alpha, Vec vreduced)
{
  PetscInt       nfull,nreduced;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(vfull,VEC_CLASSID,1);
  PetscValidHeaderSpecific(is,IS_CLASSID,2);
  PetscValidHeaderSpecific(vreduced,VEC_CLASSID,4);
  PetscCall(VecGetSize(vfull,&nfull));
  PetscCall(VecGetSize(vreduced,&nreduced));

  if (nfull == nreduced) { /* Also takes care of masked vectors */
    PetscCall(VecAXPY(vfull,alpha,vreduced));
  } else {
    PetscScalar      *y;
    const PetscScalar *x;
    PetscInt          i,n,m,rstart,rend;
    const PetscInt    *id;

    PetscCall(VecGetArray(vfull,&y));
    PetscCall(VecGetArrayRead(vreduced,&x));
    PetscCall(ISGetIndices(is,&id));
    PetscCall(ISGetLocalSize(is,&n));
    PetscCall(VecGetLocalSize(vreduced,&m));
    PetscCheck(m == n,PETSC_COMM_SELF,PETSC_ERR_SUP,"IS local length not equal to Vec local length");
    PetscCall(VecGetOwnershipRange(vfull,&rstart,&rend));
    y   -= rstart;
    if (alpha == 1.0) {
      for (i=0; i<n; ++i) {
        if (id[i] < 0) continue;
        PetscCheckFalse(id[i] < rstart || id[i] >= rend,PETSC_COMM_SELF, PETSC_ERR_SUP, "Only owned values supported");
        y[id[i]] += x[i];
      }
    } else {
      for (i=0; i<n; ++i) {
        if (id[i] < 0) continue;
        PetscCheckFalse(id[i] < rstart || id[i] >= rend,PETSC_COMM_SELF, PETSC_ERR_SUP, "Only owned values supported");
        y[id[i]] += alpha*x[i];
      }
    }
    y += rstart;
    PetscCall(ISRestoreIndices(is,&id));
    PetscCall(VecRestoreArray(vfull,&y));
    PetscCall(VecRestoreArrayRead(vreduced,&x));
  }
  PetscFunctionReturn(0);
}

/*@
  VecISCopy - Copies between a reduced vector and the appropriate elements of a full-space vector.

  Input Parameters:
+ vfull    - the full-space vector
. is       - the index set for the reduced space
. mode     - the direction of copying, SCATTER_FORWARD or SCATTER_REVERSE
- vreduced - the reduced-space vector

  Output Parameters:
. vfull    - the sum of the full-space vector and reduced-space vector

  Notes:
    The index set identifies entries in the global vector.
    Negative indices are skipped; indices outside the ownership range of vfull will raise an error.

    mode == SCATTER_FORWARD: vfull[is[i]] = vreduced[i]
    mode == SCATTER_REVERSE: vreduced[i] = vfull[is[i]]

  Level: advanced

.seealso:  VecAXPY(), VecGetOwnershipRange()
@*/
PetscErrorCode VecISCopy(Vec vfull, IS is, ScatterMode mode, Vec vreduced)
{
  PetscInt       nfull, nreduced;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(vfull,VEC_CLASSID,1);
  PetscValidHeaderSpecific(is,IS_CLASSID,2);
  PetscValidHeaderSpecific(vreduced,VEC_CLASSID,4);
  PetscCall(VecGetSize(vfull, &nfull));
  PetscCall(VecGetSize(vreduced, &nreduced));

  if (nfull == nreduced) { /* Also takes care of masked vectors */
    if (mode == SCATTER_FORWARD) {
      PetscCall(VecCopy(vreduced, vfull));
    } else {
      PetscCall(VecCopy(vfull, vreduced));
    }
  } else {
    const PetscInt *id;
    PetscInt        i, n, m, rstart, rend;

    PetscCall(ISGetIndices(is, &id));
    PetscCall(ISGetLocalSize(is, &n));
    PetscCall(VecGetLocalSize(vreduced, &m));
    PetscCall(VecGetOwnershipRange(vfull, &rstart, &rend));
    PetscCheck(m == n,PETSC_COMM_SELF, PETSC_ERR_SUP, "IS local length %" PetscInt_FMT " not equal to Vec local length %" PetscInt_FMT, n, m);
    if (mode == SCATTER_FORWARD) {
      PetscScalar       *y;
      const PetscScalar *x;

      PetscCall(VecGetArray(vfull, &y));
      PetscCall(VecGetArrayRead(vreduced, &x));
      y   -= rstart;
      for (i = 0; i < n; ++i) {
        if (id[i] < 0) continue;
        PetscCheckFalse(id[i] < rstart || id[i] >= rend,PETSC_COMM_SELF, PETSC_ERR_SUP, "Only owned values supported");
        y[id[i]] = x[i];
      }
      y   += rstart;
      PetscCall(VecRestoreArrayRead(vreduced, &x));
      PetscCall(VecRestoreArray(vfull, &y));
    } else if (mode == SCATTER_REVERSE) {
      PetscScalar       *x;
      const PetscScalar *y;

      PetscCall(VecGetArrayRead(vfull, &y));
      PetscCall(VecGetArray(vreduced, &x));
      for (i = 0; i < n; ++i) {
        if (id[i] < 0) continue;
        PetscCheckFalse(id[i] < rstart || id[i] >= rend,PETSC_COMM_SELF, PETSC_ERR_SUP, "Only owned values supported");
        x[i] = y[id[i]-rstart];
      }
      PetscCall(VecRestoreArray(vreduced, &x));
      PetscCall(VecRestoreArrayRead(vfull, &y));
    } else SETERRQ(PetscObjectComm((PetscObject) vfull), PETSC_ERR_ARG_WRONG, "Only forward or reverse modes are legal");
    PetscCall(ISRestoreIndices(is, &id));
  }
  PetscFunctionReturn(0);
}

/*@
   ISComplementVec - Creates the complement of the index set relative to a layout defined by a Vec

   Collective on IS

   Input Parameters:
+  S -  a PETSc IS
-  V - the reference vector space

   Output Parameter:
.  T -  the complement of S

   Level: advanced

.seealso: ISCreateGeneral()
@*/
PetscErrorCode ISComplementVec(IS S, Vec V, IS *T)
{
  PetscInt       start, end;

  PetscFunctionBegin;
  PetscCall(VecGetOwnershipRange(V,&start,&end));
  PetscCall(ISComplement(S,start,end,T));
  PetscFunctionReturn(0);
}

/*@
   VecISSet - Sets the elements of a vector, specified by an index set, to a constant

   Input Parameters:
+  V - the vector
.  S - index set for the locations in the vector
-  c - the constant

  Notes:
    The index set identifies entries in the global vector.
    Negative indices are skipped; indices outside the ownership range of V will raise an error.

   Level: advanced

.seealso: VecSet(), VecGetOwnershipRange()
@*/
PetscErrorCode VecISSet(Vec V,IS S, PetscScalar c)
{
  PetscInt       nloc,low,high,i;
  const PetscInt *s;
  PetscScalar    *v;

  PetscFunctionBegin;
  if (!S) PetscFunctionReturn(0); /* simply return with no-op if the index set is NULL */
  PetscValidHeaderSpecific(V,VEC_CLASSID,1);
  PetscValidHeaderSpecific(S,IS_CLASSID,2);
  PetscValidType(V,1);

  PetscCall(VecGetOwnershipRange(V,&low,&high));
  PetscCall(ISGetLocalSize(S,&nloc));
  PetscCall(ISGetIndices(S,&s));
  PetscCall(VecGetArray(V,&v));
  for (i=0; i<nloc; ++i) {
    if (s[i] < 0) continue;
    PetscCheckFalse(s[i] < low || s[i] >= high,PETSC_COMM_SELF, PETSC_ERR_SUP, "Only owned values supported");
    v[s[i]-low] = c;
  }
  PetscCall(ISRestoreIndices(S,&s));
  PetscCall(VecRestoreArray(V,&v));
  PetscFunctionReturn(0);
}

#if !defined(PETSC_USE_COMPLEX)
/*@C
  VecBoundGradientProjection - Projects  vector according to this definition.
  If XL[i] < X[i] < XU[i], then GP[i] = G[i];
  If X[i] <= XL[i], then GP[i] = min(G[i],0);
  If X[i] >= XU[i], then GP[i] = max(G[i],0);

  Input Parameters:
+ G - current gradient vector
. X - current solution vector with XL[i] <= X[i] <= XU[i]
. XL - lower bounds
- XU - upper bounds

  Output Parameter:
. GP - gradient projection vector

  Notes:
    GP may be the same vector as G

  Level: advanced
@*/
PetscErrorCode VecBoundGradientProjection(Vec G, Vec X, Vec XL, Vec XU, Vec GP)
{

  PetscInt        n,i;
  const PetscReal *xptr,*xlptr,*xuptr;
  PetscReal       *gptr,*gpptr;
  PetscReal       xval,gpval;

  /* Project variables at the lower and upper bound */
  PetscFunctionBegin;
  PetscValidHeaderSpecific(G,VEC_CLASSID,1);
  PetscValidHeaderSpecific(X,VEC_CLASSID,2);
  PetscValidHeaderSpecific(XL,VEC_CLASSID,3);
  PetscValidHeaderSpecific(XU,VEC_CLASSID,4);
  PetscValidHeaderSpecific(GP,VEC_CLASSID,5);

  PetscCall(VecGetLocalSize(X,&n));

  PetscCall(VecGetArrayRead(X,&xptr));
  PetscCall(VecGetArrayRead(XL,&xlptr));
  PetscCall(VecGetArrayRead(XU,&xuptr));
  PetscCall(VecGetArrayPair(G,GP,&gptr,&gpptr));

  for (i=0; i<n; ++i) {
    gpval = gptr[i]; xval = xptr[i];
    if (gpval>0.0 && xval<=xlptr[i]) {
      gpval = 0.0;
    } else if (gpval<0.0 && xval>=xuptr[i]) {
      gpval = 0.0;
    }
    gpptr[i] = gpval;
  }

  PetscCall(VecRestoreArrayRead(X,&xptr));
  PetscCall(VecRestoreArrayRead(XL,&xlptr));
  PetscCall(VecRestoreArrayRead(XU,&xuptr));
  PetscCall(VecRestoreArrayPair(G,GP,&gptr,&gpptr));
  PetscFunctionReturn(0);
}
#endif

/*@
     VecStepMaxBounded - See below

     Collective on Vec

     Input Parameters:
+      X  - vector with no negative entries
.      XL - lower bounds
.      XU - upper bounds
-      DX  - step direction, can have negative, positive or zero entries

     Output Parameter:
.     stepmax -   minimum value so that X[i] + stepmax*DX[i] <= XL[i]  or  XU[i] <= X[i] + stepmax*DX[i]

  Level: intermediate

@*/
PetscErrorCode VecStepMaxBounded(Vec X, Vec DX, Vec XL, Vec XU, PetscReal *stepmax)
{
  PetscInt          i,nn;
  const PetscScalar *xx,*dx,*xl,*xu;
  PetscReal         localmax=0;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(X,VEC_CLASSID,1);
  PetscValidHeaderSpecific(DX,VEC_CLASSID,2);
  PetscValidHeaderSpecific(XL,VEC_CLASSID,3);
  PetscValidHeaderSpecific(XU,VEC_CLASSID,4);

  PetscCall(VecGetArrayRead(X,&xx));
  PetscCall(VecGetArrayRead(XL,&xl));
  PetscCall(VecGetArrayRead(XU,&xu));
  PetscCall(VecGetArrayRead(DX,&dx));
  PetscCall(VecGetLocalSize(X,&nn));
  for (i=0;i<nn;i++) {
    if (PetscRealPart(dx[i]) > 0) {
      localmax=PetscMax(localmax,PetscRealPart((xu[i]-xx[i])/dx[i]));
    } else if (PetscRealPart(dx[i])<0) {
      localmax=PetscMax(localmax,PetscRealPart((xl[i]-xx[i])/dx[i]));
    }
  }
  PetscCall(VecRestoreArrayRead(X,&xx));
  PetscCall(VecRestoreArrayRead(XL,&xl));
  PetscCall(VecRestoreArrayRead(XU,&xu));
  PetscCall(VecRestoreArrayRead(DX,&dx));
  PetscCall(MPIU_Allreduce(&localmax,stepmax,1,MPIU_REAL,MPIU_MAX,PetscObjectComm((PetscObject)X)));
  PetscFunctionReturn(0);
}

/*@
     VecStepBoundInfo - See below

     Collective on Vec

     Input Parameters:
+      X  - vector with no negative entries
.      XL - lower bounds
.      XU - upper bounds
-      DX  - step direction, can have negative, positive or zero entries

     Output Parameters:
+     boundmin -  (may be NULL this it is not computed) maximum value so that   XL[i] <= X[i] + boundmax*DX[i] <= XU[i]
.     wolfemin -  (may be NULL this it is not computed)
-     boundmax -   (may be NULL this it is not computed) minimum value so that X[i] + boundmax*DX[i] <= XL[i]  or  XU[i] <= X[i] + boundmax*DX[i]

     Notes:
    For complex numbers only compares the real part

  Level: advanced
@*/
PetscErrorCode VecStepBoundInfo(Vec X, Vec DX, Vec XL, Vec XU, PetscReal *boundmin, PetscReal *wolfemin, PetscReal *boundmax)
{
  PetscInt          n,i;
  const PetscScalar *x,*xl,*xu,*dx;
  PetscReal         t;
  PetscReal         localmin=PETSC_INFINITY,localwolfemin=PETSC_INFINITY,localmax=-1;
  MPI_Comm          comm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(X,VEC_CLASSID,1);
  PetscValidHeaderSpecific(XL,VEC_CLASSID,3);
  PetscValidHeaderSpecific(XU,VEC_CLASSID,4);
  PetscValidHeaderSpecific(DX,VEC_CLASSID,2);

  PetscCall(VecGetArrayRead(X,&x));
  PetscCall(VecGetArrayRead(XL,&xl));
  PetscCall(VecGetArrayRead(XU,&xu));
  PetscCall(VecGetArrayRead(DX,&dx));
  PetscCall(VecGetLocalSize(X,&n));
  for (i=0; i<n; ++i) {
    if (PetscRealPart(dx[i])>0 && PetscRealPart(xu[i]) < PETSC_INFINITY) {
      t=PetscRealPart((xu[i]-x[i])/dx[i]);
      localmin=PetscMin(t,localmin);
      if (localmin>0) {
        localwolfemin = PetscMin(t,localwolfemin);
      }
      localmax = PetscMax(t,localmax);
    } else if (PetscRealPart(dx[i])<0 && PetscRealPart(xl[i]) > PETSC_NINFINITY) {
      t=PetscRealPart((xl[i]-x[i])/dx[i]);
      localmin = PetscMin(t,localmin);
      if (localmin>0) {
        localwolfemin = PetscMin(t,localwolfemin);
      }
      localmax = PetscMax(t,localmax);
    }
  }

  PetscCall(VecRestoreArrayRead(X,&x));
  PetscCall(VecRestoreArrayRead(XL,&xl));
  PetscCall(VecRestoreArrayRead(XU,&xu));
  PetscCall(VecRestoreArrayRead(DX,&dx));
  PetscCall(PetscObjectGetComm((PetscObject)X,&comm));

  if (boundmin) {
    PetscCall(MPIU_Allreduce(&localmin,boundmin,1,MPIU_REAL,MPIU_MIN,comm));
    PetscCall(PetscInfo(X,"Step Bound Info: Closest Bound: %20.19e\n",(double)*boundmin));
  }
  if (wolfemin) {
    PetscCall(MPIU_Allreduce(&localwolfemin,wolfemin,1,MPIU_REAL,MPIU_MIN,comm));
    PetscCall(PetscInfo(X,"Step Bound Info: Wolfe: %20.19e\n",(double)*wolfemin));
  }
  if (boundmax) {
    PetscCall(MPIU_Allreduce(&localmax,boundmax,1,MPIU_REAL,MPIU_MAX,comm));
    if (*boundmax < 0) *boundmax=PETSC_INFINITY;
    PetscCall(PetscInfo(X,"Step Bound Info: Max: %20.19e\n",(double)*boundmax));
  }
  PetscFunctionReturn(0);
}

/*@
     VecStepMax - Returns the largest value so that x[i] + step*DX[i] >= 0 for all i

     Collective on Vec

     Input Parameters:
+      X  - vector with no negative entries
-      DX  - a step direction, can have negative, positive or zero entries

     Output Parameter:
.    step - largest value such that x[i] + step*DX[i] >= 0 for all i

     Notes:
    For complex numbers only compares the real part

  Level: advanced
 @*/
PetscErrorCode VecStepMax(Vec X, Vec DX, PetscReal *step)
{
  PetscInt          i,nn;
  PetscReal         stepmax=PETSC_INFINITY;
  const PetscScalar *xx,*dx;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(X,VEC_CLASSID,1);
  PetscValidHeaderSpecific(DX,VEC_CLASSID,2);

  PetscCall(VecGetLocalSize(X,&nn));
  PetscCall(VecGetArrayRead(X,&xx));
  PetscCall(VecGetArrayRead(DX,&dx));
  for (i=0;i<nn;++i) {
    PetscCheck(PetscRealPart(xx[i]) >= 0,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Vector must be positive");
    else if (PetscRealPart(dx[i])<0) stepmax=PetscMin(stepmax,PetscRealPart(-xx[i]/dx[i]));
  }
  PetscCall(VecRestoreArrayRead(X,&xx));
  PetscCall(VecRestoreArrayRead(DX,&dx));
  PetscCall(MPIU_Allreduce(&stepmax,step,1,MPIU_REAL,MPIU_MIN,PetscObjectComm((PetscObject)X)));
  PetscFunctionReturn(0);
}

/*@
  VecPow - Replaces each component of a vector by x_i^p

  Logically Collective on v

  Input Parameters:
+ v - the vector
- p - the exponent to use on each element

  Level: intermediate

@*/
PetscErrorCode VecPow(Vec v, PetscScalar p)
{
  PetscInt       n,i;
  PetscScalar    *v1;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(v,VEC_CLASSID,1);

  PetscCall(VecGetArray(v,&v1));
  PetscCall(VecGetLocalSize(v,&n));

  if (1.0 == p) {
  } else if (-1.0 == p) {
    for (i = 0; i < n; ++i) {
      v1[i] = 1.0 / v1[i];
    }
  } else if (0.0 == p) {
    for (i = 0; i < n; ++i) {
      /*  Not-a-number left alone
          Infinity set to one  */
      if (v1[i] == v1[i]) {
        v1[i] = 1.0;
      }
    }
  } else if (0.5 == p) {
    for (i = 0; i < n; ++i) {
      if (PetscRealPart(v1[i]) >= 0) {
        v1[i] = PetscSqrtScalar(v1[i]);
      } else {
        v1[i] = PETSC_INFINITY;
      }
    }
  } else if (-0.5 == p) {
    for (i = 0; i < n; ++i) {
      if (PetscRealPart(v1[i]) >= 0) {
        v1[i] = 1.0 / PetscSqrtScalar(v1[i]);
      } else {
        v1[i] = PETSC_INFINITY;
      }
    }
  } else if (2.0 == p) {
    for (i = 0; i < n; ++i) {
      v1[i] *= v1[i];
    }
  } else if (-2.0 == p) {
    for (i = 0; i < n; ++i) {
      v1[i] = 1.0 / (v1[i] * v1[i]);
    }
  } else {
    for (i = 0; i < n; ++i) {
      if (PetscRealPart(v1[i]) >= 0) {
        v1[i] = PetscPowScalar(v1[i],p);
      } else {
        v1[i] = PETSC_INFINITY;
      }
    }
  }
  PetscCall(VecRestoreArray(v,&v1));
  PetscFunctionReturn(0);
}

/*@
  VecMedian - Computes the componentwise median of three vectors
  and stores the result in this vector.  Used primarily for projecting
  a vector within upper and lower bounds.

  Logically Collective

  Input Parameters:
+ Vec1 - The first vector
. Vec2 - The second vector
- Vec3 - The third vector

  Output Parameter:
. VMedian - The median vector (this can be any one of the input vectors)

  Level: advanced
@*/
PetscErrorCode VecMedian(Vec Vec1, Vec Vec2, Vec Vec3, Vec VMedian)
{
  PetscInt          i,n,low1,high1;
  const PetscScalar *v1,*v2,*v3;
  PetscScalar       *vmed;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(Vec1,VEC_CLASSID,1);
  PetscValidHeaderSpecific(Vec2,VEC_CLASSID,2);
  PetscValidHeaderSpecific(Vec3,VEC_CLASSID,3);
  PetscValidHeaderSpecific(VMedian,VEC_CLASSID,4);

  if (Vec1==Vec2 || Vec1==Vec3) {
    PetscCall(VecCopy(Vec1,VMedian));
    PetscFunctionReturn(0);
  }
  if (Vec2==Vec3) {
    PetscCall(VecCopy(Vec2,VMedian));
    PetscFunctionReturn(0);
  }

  /* Assert that Vec1 != Vec2 and Vec2 != Vec3 */
  PetscValidType(Vec1,1);
  PetscValidType(Vec2,2);
  PetscValidType(Vec3,3);
  PetscValidType(VMedian,4);
  PetscCheckSameType(Vec1,1,Vec2,2);
  PetscCheckSameType(Vec1,1,Vec3,3);
  PetscCheckSameType(Vec1,1,VMedian,4);
  PetscCheckSameComm(Vec1,1,Vec2,2);
  PetscCheckSameComm(Vec1,1,Vec3,3);
  PetscCheckSameComm(Vec1,1,VMedian,4);
  VecCheckSameSize(Vec1,1,Vec2,2);
  VecCheckSameSize(Vec1,1,Vec3,3);
  VecCheckSameSize(Vec1,1,VMedian,4);

  PetscCall(VecGetOwnershipRange(Vec1,&low1,&high1));
  PetscCall(VecGetLocalSize(Vec1,&n));
  if (n>0) {
    PetscCall(VecGetArray(VMedian,&vmed));
    if (Vec1 != VMedian) {
      PetscCall(VecGetArrayRead(Vec1,&v1));
    } else {
      v1=vmed;
    }
    if (Vec2 != VMedian) {
      PetscCall(VecGetArrayRead(Vec2,&v2));
    } else {
      v2=vmed;
    }
    if (Vec3 != VMedian) {
      PetscCall(VecGetArrayRead(Vec3,&v3));
    } else {
      v3=vmed;
    }

    for (i=0;i<n;++i) {
      vmed[i]=PetscMax(PetscMax(PetscMin(PetscRealPart(v1[i]),PetscRealPart(v2[i])),PetscMin(PetscRealPart(v1[i]),PetscRealPart(v3[i]))),PetscMin(PetscRealPart(v2[i]),PetscRealPart(v3[i])));
    }

    PetscCall(VecRestoreArray(VMedian,&vmed));
    if (VMedian != Vec1) {
      PetscCall(VecRestoreArrayRead(Vec1,&v1));
    }
    if (VMedian != Vec2) {
      PetscCall(VecRestoreArrayRead(Vec2,&v2));
    }
    if (VMedian != Vec3) {
      PetscCall(VecRestoreArrayRead(Vec3,&v3));
    }
  }
  PetscFunctionReturn(0);
}
