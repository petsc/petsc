
#include <petsc/private/matimpl.h> /*I   "petscmat.h"  I*/

static PetscErrorCode MatMultEqual_Private(Mat A, Mat B, PetscInt n, PetscBool *flg, PetscInt t, PetscBool add)
{
  Vec         Ax = NULL, Bx = NULL, s1 = NULL, s2 = NULL, Ay = NULL, By = NULL;
  PetscRandom rctx;
  PetscReal   r1, r2, tol = PETSC_SQRT_MACHINE_EPSILON;
  PetscInt    am, an, bm, bn, k;
  PetscScalar none = -1.0;
#if defined(PETSC_USE_INFO)
  const char *sops[] = {"MatMult", "MatMultAdd", "MatMultTranspose", "MatMultTransposeAdd", "MatMultHermitianTranspose", "MatMultHermitianTransposeAdd"};
  const char *sop;
#endif

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A, MAT_CLASSID, 1);
  PetscValidHeaderSpecific(B, MAT_CLASSID, 2);
  PetscCheckSameComm(A, 1, B, 2);
  PetscValidLogicalCollectiveInt(A, n, 3);
  PetscValidBoolPointer(flg, 4);
  PetscValidLogicalCollectiveInt(A, t, 5);
  PetscValidLogicalCollectiveBool(A, add, 6);
  PetscCall(MatGetLocalSize(A, &am, &an));
  PetscCall(MatGetLocalSize(B, &bm, &bn));
  PetscCheck(am == bm && an == bn, PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Mat A,Mat B: local dim %" PetscInt_FMT " %" PetscInt_FMT " %" PetscInt_FMT " %" PetscInt_FMT, am, bm, an, bn);
#if defined(PETSC_USE_INFO)
  sop = sops[(add ? 1 : 0) + 2 * t]; /* t = 0 => no transpose, t = 1 => transpose, t = 2 => Hermitian transpose */
#endif
  PetscCall(PetscRandomCreate(PetscObjectComm((PetscObject)A), &rctx));
  PetscCall(PetscRandomSetFromOptions(rctx));
  if (t) {
    PetscCall(MatCreateVecs(A, &s1, &Ax));
    PetscCall(MatCreateVecs(B, &s2, &Bx));
  } else {
    PetscCall(MatCreateVecs(A, &Ax, &s1));
    PetscCall(MatCreateVecs(B, &Bx, &s2));
  }
  if (add) {
    PetscCall(VecDuplicate(s1, &Ay));
    PetscCall(VecDuplicate(s2, &By));
  }

  *flg = PETSC_TRUE;
  for (k = 0; k < n; k++) {
    PetscCall(VecSetRandom(Ax, rctx));
    PetscCall(VecCopy(Ax, Bx));
    if (add) {
      PetscCall(VecSetRandom(Ay, rctx));
      PetscCall(VecCopy(Ay, By));
    }
    if (t == 1) {
      if (add) {
        PetscCall(MatMultTransposeAdd(A, Ax, Ay, s1));
        PetscCall(MatMultTransposeAdd(B, Bx, By, s2));
      } else {
        PetscCall(MatMultTranspose(A, Ax, s1));
        PetscCall(MatMultTranspose(B, Bx, s2));
      }
    } else if (t == 2) {
      if (add) {
        PetscCall(MatMultHermitianTransposeAdd(A, Ax, Ay, s1));
        PetscCall(MatMultHermitianTransposeAdd(B, Bx, By, s2));
      } else {
        PetscCall(MatMultHermitianTranspose(A, Ax, s1));
        PetscCall(MatMultHermitianTranspose(B, Bx, s2));
      }
    } else {
      if (add) {
        PetscCall(MatMultAdd(A, Ax, Ay, s1));
        PetscCall(MatMultAdd(B, Bx, By, s2));
      } else {
        PetscCall(MatMult(A, Ax, s1));
        PetscCall(MatMult(B, Bx, s2));
      }
    }
    PetscCall(VecNorm(s2, NORM_INFINITY, &r2));
    if (r2 < tol) {
      PetscCall(VecNorm(s1, NORM_INFINITY, &r1));
    } else {
      PetscCall(VecAXPY(s2, none, s1));
      PetscCall(VecNorm(s2, NORM_INFINITY, &r1));
      r1 /= r2;
    }
    if (r1 > tol) {
      *flg = PETSC_FALSE;
      PetscCall(PetscInfo(A, "Error: %" PetscInt_FMT "-th %s() %g\n", k, sop, (double)r1));
      break;
    }
  }
  PetscCall(PetscRandomDestroy(&rctx));
  PetscCall(VecDestroy(&Ax));
  PetscCall(VecDestroy(&Bx));
  PetscCall(VecDestroy(&Ay));
  PetscCall(VecDestroy(&By));
  PetscCall(VecDestroy(&s1));
  PetscCall(VecDestroy(&s2));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatMatMultEqual_Private(Mat A, Mat B, Mat C, PetscInt n, PetscBool *flg, PetscBool At, PetscBool Bt)
{
  Vec         Ax, Bx, Cx, s1, s2, s3;
  PetscRandom rctx;
  PetscReal   r1, r2, tol = PETSC_SQRT_MACHINE_EPSILON;
  PetscInt    am, an, bm, bn, cm, cn, k;
  PetscScalar none = -1.0;
#if defined(PETSC_USE_INFO)
  const char *sops[] = {"MatMatMult", "MatTransposeMatMult", "MatMatTransposeMult", "MatTransposeMatTransposeMult"};
  const char *sop;
#endif

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A, MAT_CLASSID, 1);
  PetscValidHeaderSpecific(B, MAT_CLASSID, 2);
  PetscCheckSameComm(A, 1, B, 2);
  PetscValidHeaderSpecific(C, MAT_CLASSID, 3);
  PetscCheckSameComm(A, 1, C, 3);
  PetscValidLogicalCollectiveInt(A, n, 4);
  PetscValidBoolPointer(flg, 5);
  PetscValidLogicalCollectiveBool(A, At, 6);
  PetscValidLogicalCollectiveBool(B, Bt, 7);
  PetscCall(MatGetLocalSize(A, &am, &an));
  PetscCall(MatGetLocalSize(B, &bm, &bn));
  PetscCall(MatGetLocalSize(C, &cm, &cn));
  if (At) {
    PetscInt tt = an;
    an          = am;
    am          = tt;
  };
  if (Bt) {
    PetscInt tt = bn;
    bn          = bm;
    bm          = tt;
  };
  PetscCheck(an == bm && am == cm && bn == cn, PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Mat A, B, C local dim %" PetscInt_FMT " %" PetscInt_FMT " %" PetscInt_FMT " %" PetscInt_FMT " %" PetscInt_FMT " %" PetscInt_FMT, am, an, bm, bn, cm, cn);

#if defined(PETSC_USE_INFO)
  sop = sops[(At ? 1 : 0) + 2 * (Bt ? 1 : 0)];
#endif
  PetscCall(PetscRandomCreate(PetscObjectComm((PetscObject)C), &rctx));
  PetscCall(PetscRandomSetFromOptions(rctx));
  if (Bt) {
    PetscCall(MatCreateVecs(B, &s1, &Bx));
  } else {
    PetscCall(MatCreateVecs(B, &Bx, &s1));
  }
  if (At) {
    PetscCall(MatCreateVecs(A, &s2, &Ax));
  } else {
    PetscCall(MatCreateVecs(A, &Ax, &s2));
  }
  PetscCall(MatCreateVecs(C, &Cx, &s3));

  *flg = PETSC_TRUE;
  for (k = 0; k < n; k++) {
    PetscCall(VecSetRandom(Bx, rctx));
    if (Bt) {
      PetscCall(MatMultTranspose(B, Bx, s1));
    } else {
      PetscCall(MatMult(B, Bx, s1));
    }
    PetscCall(VecCopy(s1, Ax));
    if (At) {
      PetscCall(MatMultTranspose(A, Ax, s2));
    } else {
      PetscCall(MatMult(A, Ax, s2));
    }
    PetscCall(VecCopy(Bx, Cx));
    PetscCall(MatMult(C, Cx, s3));

    PetscCall(VecNorm(s2, NORM_INFINITY, &r2));
    if (r2 < tol) {
      PetscCall(VecNorm(s3, NORM_INFINITY, &r1));
    } else {
      PetscCall(VecAXPY(s2, none, s3));
      PetscCall(VecNorm(s2, NORM_INFINITY, &r1));
      r1 /= r2;
    }
    if (r1 > tol) {
      *flg = PETSC_FALSE;
      PetscCall(PetscInfo(A, "Error: %" PetscInt_FMT "-th %s %g\n", k, sop, (double)r1));
      break;
    }
  }
  PetscCall(PetscRandomDestroy(&rctx));
  PetscCall(VecDestroy(&Ax));
  PetscCall(VecDestroy(&Bx));
  PetscCall(VecDestroy(&Cx));
  PetscCall(VecDestroy(&s1));
  PetscCall(VecDestroy(&s2));
  PetscCall(VecDestroy(&s3));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
   MatMultEqual - Compares matrix-vector products of two matrices.

   Collective

   Input Parameters:
+  A - the first matrix
.  B - the second matrix
-  n - number of random vectors to be tested

   Output Parameter:
.  flg - `PETSC_TRUE` if the products are equal; `PETSC_FALSE` otherwise.

   Level: intermediate

.seealso: `Mat`, `MatMultAddEqual()`, `MatMultTransposeEqual()`, `MatMultTransposeAddEqual()`, `MatIsLinear()`
@*/
PetscErrorCode MatMultEqual(Mat A, Mat B, PetscInt n, PetscBool *flg)
{
  PetscFunctionBegin;
  PetscCall(MatMultEqual_Private(A, B, n, flg, 0, PETSC_FALSE));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
   MatMultAddEqual - Compares matrix-vector product plus vector add of two matrices.

   Collective

   Input Parameters:
+  A - the first matrix
.  B - the second matrix
-  n - number of random vectors to be tested

   Output Parameter:
.  flg - `PETSC_TRUE` if the products are equal; `PETSC_FALSE` otherwise.

   Level: intermediate

.seealso: `Mat`, `MatMultEqual()`, `MatMultTransposeEqual()`, `MatMultTransposeAddEqual()`
@*/
PetscErrorCode MatMultAddEqual(Mat A, Mat B, PetscInt n, PetscBool *flg)
{
  PetscFunctionBegin;
  PetscCall(MatMultEqual_Private(A, B, n, flg, 0, PETSC_TRUE));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
   MatMultTransposeEqual - Compares matrix-vector products of two matrices.

   Collective

   Input Parameters:
+  A - the first matrix
.  B - the second matrix
-  n - number of random vectors to be tested

   Output Parameter:
.  flg - `PETSC_TRUE` if the products are equal; `PETSC_FALSE` otherwise.

   Level: intermediate

.seealso: `Mat`, `MatMultEqual()`, `MatMultAddEqual()`, `MatMultTransposeAddEqual()`
@*/
PetscErrorCode MatMultTransposeEqual(Mat A, Mat B, PetscInt n, PetscBool *flg)
{
  PetscFunctionBegin;
  PetscCall(MatMultEqual_Private(A, B, n, flg, 1, PETSC_FALSE));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
   MatMultTransposeAddEqual - Compares matrix-vector products of two matrices.

   Collective

   Input Parameters:
+  A - the first matrix
.  B - the second matrix
-  n - number of random vectors to be tested

   Output Parameter:
.  flg - `PETSC_TRUE` if the products are equal; `PETSC_FALSE` otherwise.

   Level: intermediate

.seealso: `Mat`, `MatMultEqual()`, `MatMultAddEqual()`, `MatMultTransposeEqual()`
@*/
PetscErrorCode MatMultTransposeAddEqual(Mat A, Mat B, PetscInt n, PetscBool *flg)
{
  PetscFunctionBegin;
  PetscCall(MatMultEqual_Private(A, B, n, flg, 1, PETSC_TRUE));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
   MatMultHermitianTransposeEqual - Compares matrix-vector products of two matrices.

   Collective

   Input Parameters:
+  A - the first matrix
.  B - the second matrix
-  n - number of random vectors to be tested

   Output Parameter:
.  flg - `PETSC_TRUE` if the products are equal; `PETSC_FALSE` otherwise.

   Level: intermediate

.seealso: `Mat`, `MatMatMultAddEqual()`, `MatMultEqual()`, `MatMultAddEqual()`, `MatMultTransposeEqual()`
@*/
PetscErrorCode MatMultHermitianTransposeEqual(Mat A, Mat B, PetscInt n, PetscBool *flg)
{
  PetscFunctionBegin;
  PetscCall(MatMultEqual_Private(A, B, n, flg, 2, PETSC_FALSE));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
   MatMultHermitianTransposeAddEqual - Compares matrix-vector products of two matrices.

   Collective

   Input Parameters:
+  A - the first matrix
.  B - the second matrix
-  n - number of random vectors to be tested

   Output Parameter:
.  flg - `PETSC_TRUE` if the products are equal; `PETSC_FALSE` otherwise.

   Level: intermediate

.seealso: `Mat`, `MatMatMultAddEqual()`, `MatMultEqual()`, `MatMultAddEqual()`, `MatMultTransposeEqual()`
@*/
PetscErrorCode MatMultHermitianTransposeAddEqual(Mat A, Mat B, PetscInt n, PetscBool *flg)
{
  PetscFunctionBegin;
  PetscCall(MatMultEqual_Private(A, B, n, flg, 2, PETSC_TRUE));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
   MatMatMultEqual - Test A*B*x = C*x for n random vector x

   Collective

   Input Parameters:
+  A - the first matrix
.  B - the second matrix
.  C - the third matrix
-  n - number of random vectors to be tested

   Output Parameter:
.  flg - `PETSC_TRUE` if the products are equal; `PETSC_FALSE` otherwise.

   Level: intermediate

.seealso: `Mat`, `MatMatMultAddEqual()`, `MatMultEqual()`, `MatMultAddEqual()`, `MatMultTransposeEqual()`
@*/
PetscErrorCode MatMatMultEqual(Mat A, Mat B, Mat C, PetscInt n, PetscBool *flg)
{
  PetscFunctionBegin;
  PetscCall(MatMatMultEqual_Private(A, B, C, n, flg, PETSC_FALSE, PETSC_FALSE));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
   MatTransposeMatMultEqual - Test A^T*B*x = C*x for n random vector x

   Collective

   Input Parameters:
+  A - the first matrix
.  B - the second matrix
.  C - the third matrix
-  n - number of random vectors to be tested

   Output Parameter:
.  flg - `PETSC_TRUE` if the products are equal; `PETSC_FALSE` otherwise.

   Level: intermediate

.seealso: `Mat`, `MatMatMultAddEqual()`, `MatMultEqual()`, `MatMultAddEqual()`, `MatMultTransposeEqual()`
@*/
PetscErrorCode MatTransposeMatMultEqual(Mat A, Mat B, Mat C, PetscInt n, PetscBool *flg)
{
  PetscFunctionBegin;
  PetscCall(MatMatMultEqual_Private(A, B, C, n, flg, PETSC_TRUE, PETSC_FALSE));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
   MatMatTransposeMultEqual - Test A*B^T*x = C*x for n random vector x

   Collective

   Input Parameters:
+  A - the first matrix
.  B - the second matrix
.  C - the third matrix
-  n - number of random vectors to be tested

   Output Parameter:
.  flg - `PETSC_TRUE` if the products are equal; `PETSC_FALSE` otherwise.

   Level: intermediate

.seealso: `Mat`, `MatMatMultAddEqual()`, `MatMultEqual()`, `MatMultAddEqual()`, `MatMultTransposeEqual()`
@*/
PetscErrorCode MatMatTransposeMultEqual(Mat A, Mat B, Mat C, PetscInt n, PetscBool *flg)
{
  PetscFunctionBegin;
  PetscCall(MatMatMultEqual_Private(A, B, C, n, flg, PETSC_FALSE, PETSC_TRUE));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatProjMultEqual_Private(Mat A, Mat B, Mat C, PetscInt n, PetscBool rart, PetscBool *flg)
{
  Vec         x, v1, v2, v3, v4, Cx, Bx;
  PetscReal   norm_abs, norm_rel, tol = PETSC_SQRT_MACHINE_EPSILON;
  PetscInt    i, am, an, bm, bn, cm, cn;
  PetscRandom rdm;
  PetscScalar none = -1.0;

  PetscFunctionBegin;
  PetscCall(MatGetLocalSize(A, &am, &an));
  PetscCall(MatGetLocalSize(B, &bm, &bn));
  if (rart) {
    PetscInt t = bm;
    bm         = bn;
    bn         = t;
  }
  PetscCall(MatGetLocalSize(C, &cm, &cn));
  PetscCheck(an == bm && bn == cm && bn == cn, PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Mat A, B, C local dim %" PetscInt_FMT " %" PetscInt_FMT " %" PetscInt_FMT " %" PetscInt_FMT " %" PetscInt_FMT " %" PetscInt_FMT, am, an, bm, bn, cm, cn);

  /* Create left vector of A: v2 */
  PetscCall(MatCreateVecs(A, &Bx, &v2));

  /* Create right vectors of B: x, v3, v4 */
  if (rart) {
    PetscCall(MatCreateVecs(B, &v1, &x));
  } else {
    PetscCall(MatCreateVecs(B, &x, &v1));
  }
  PetscCall(VecDuplicate(x, &v3));

  PetscCall(MatCreateVecs(C, &Cx, &v4));
  PetscCall(PetscRandomCreate(PETSC_COMM_WORLD, &rdm));
  PetscCall(PetscRandomSetFromOptions(rdm));

  *flg = PETSC_TRUE;
  for (i = 0; i < n; i++) {
    PetscCall(VecSetRandom(x, rdm));
    PetscCall(VecCopy(x, Cx));
    PetscCall(MatMult(C, Cx, v4)); /* v4 = C*x   */
    if (rart) {
      PetscCall(MatMultTranspose(B, x, v1));
    } else {
      PetscCall(MatMult(B, x, v1));
    }
    PetscCall(VecCopy(v1, Bx));
    PetscCall(MatMult(A, Bx, v2)); /* v2 = A*B*x */
    PetscCall(VecCopy(v2, v1));
    if (rart) {
      PetscCall(MatMult(B, v1, v3)); /* v3 = R*A*R^t*x */
    } else {
      PetscCall(MatMultTranspose(B, v1, v3)); /* v3 = Bt*A*B*x */
    }
    PetscCall(VecNorm(v4, NORM_2, &norm_abs));
    PetscCall(VecAXPY(v4, none, v3));
    PetscCall(VecNorm(v4, NORM_2, &norm_rel));

    if (norm_abs > tol) norm_rel /= norm_abs;
    if (norm_rel > tol) {
      *flg = PETSC_FALSE;
      PetscCall(PetscInfo(A, "Error: %" PetscInt_FMT "-th Mat%sMult() %g\n", i, rart ? "RARt" : "PtAP", (double)norm_rel));
      break;
    }
  }

  PetscCall(PetscRandomDestroy(&rdm));
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&Bx));
  PetscCall(VecDestroy(&Cx));
  PetscCall(VecDestroy(&v1));
  PetscCall(VecDestroy(&v2));
  PetscCall(VecDestroy(&v3));
  PetscCall(VecDestroy(&v4));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
   MatPtAPMultEqual - Compares matrix-vector products of C = Bt*A*B

   Collective

   Input Parameters:
+  A - the first matrix
.  B - the second matrix
.  C - the third matrix
-  n - number of random vectors to be tested

   Output Parameter:
.  flg - `PETSC_TRUE` if the products are equal; `PETSC_FALSE` otherwise.

   Level: intermediate

.seealso: `Mat`, `MatMatMultAddEqual()`, `MatMultEqual()`, `MatMultAddEqual()`, `MatMultTransposeEqual()`
@*/
PetscErrorCode MatPtAPMultEqual(Mat A, Mat B, Mat C, PetscInt n, PetscBool *flg)
{
  PetscFunctionBegin;
  PetscCall(MatProjMultEqual_Private(A, B, C, n, PETSC_FALSE, flg));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
   MatRARtMultEqual - Compares matrix-vector products of C = B*A*B^t

   Collective

   Input Parameters:
+  A - the first matrix
.  B - the second matrix
.  C - the third matrix
-  n - number of random vectors to be tested

   Output Parameter:
.  flg - `PETSC_TRUE` if the products are equal; `PETSC_FALSE` otherwise.

   Level: intermediate

.seealso: `Mat`, `MatMatMultAddEqual()`, `MatMultEqual()`, `MatMultAddEqual()`, `MatMultTransposeEqual()`
@*/
PetscErrorCode MatRARtMultEqual(Mat A, Mat B, Mat C, PetscInt n, PetscBool *flg)
{
  PetscFunctionBegin;
  PetscCall(MatProjMultEqual_Private(A, B, C, n, PETSC_TRUE, flg));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
   MatIsLinear - Check if a shell matrix `A` is a linear operator.

   Collective

   Input Parameters:
+  A - the shell matrix
-  n - number of random vectors to be tested

   Output Parameter:
.  flg - `PETSC_TRUE` if the shell matrix is linear; `PETSC_FALSE` otherwise.

   Level: intermediate

.seealso: `Mat`, `MatMatMultAddEqual()`, `MatMultEqual()`, `MatMultAddEqual()`, `MatMultTransposeEqual()`
@*/
PetscErrorCode MatIsLinear(Mat A, PetscInt n, PetscBool *flg)
{
  Vec         x, y, s1, s2;
  PetscRandom rctx;
  PetscScalar a;
  PetscInt    k;
  PetscReal   norm, normA;
  MPI_Comm    comm;
  PetscMPIInt rank;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A, MAT_CLASSID, 1);
  PetscCall(PetscObjectGetComm((PetscObject)A, &comm));
  PetscCallMPI(MPI_Comm_rank(comm, &rank));

  PetscCall(PetscRandomCreate(comm, &rctx));
  PetscCall(PetscRandomSetFromOptions(rctx));
  PetscCall(MatCreateVecs(A, &x, &s1));
  PetscCall(VecDuplicate(x, &y));
  PetscCall(VecDuplicate(s1, &s2));

  *flg = PETSC_TRUE;
  for (k = 0; k < n; k++) {
    PetscCall(VecSetRandom(x, rctx));
    PetscCall(VecSetRandom(y, rctx));
    if (rank == 0) PetscCall(PetscRandomGetValue(rctx, &a));
    PetscCallMPI(MPI_Bcast(&a, 1, MPIU_SCALAR, 0, comm));

    /* s2 = a*A*x + A*y */
    PetscCall(MatMult(A, y, s2));  /* s2 = A*y */
    PetscCall(MatMult(A, x, s1));  /* s1 = A*x */
    PetscCall(VecAXPY(s2, a, s1)); /* s2 = a s1 + s2 */

    /* s1 = A * (a x + y) */
    PetscCall(VecAXPY(y, a, x)); /* y = a x + y */
    PetscCall(MatMult(A, y, s1));
    PetscCall(VecNorm(s1, NORM_INFINITY, &normA));

    PetscCall(VecAXPY(s2, -1.0, s1)); /* s2 = - s1 + s2 */
    PetscCall(VecNorm(s2, NORM_INFINITY, &norm));
    if (norm / normA > 100. * PETSC_MACHINE_EPSILON) {
      *flg = PETSC_FALSE;
      PetscCall(PetscInfo(A, "Error: %" PetscInt_FMT "-th |A*(ax+y) - (a*A*x+A*y)|/|A(ax+y)| %g > tol %g\n", k, (double)(norm / normA), (double)(100. * PETSC_MACHINE_EPSILON)));
      break;
    }
  }
  PetscCall(PetscRandomDestroy(&rctx));
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&y));
  PetscCall(VecDestroy(&s1));
  PetscCall(VecDestroy(&s2));
  PetscFunctionReturn(PETSC_SUCCESS);
}
