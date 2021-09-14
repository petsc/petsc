#include <petsc/private/petscimpl.h>
#include <petsctao.h>      /*I "petsctao.h" I*/
#include <petscsys.h>

PETSC_STATIC_INLINE PetscReal Fischer(PetscReal a, PetscReal b)
{
  /* Method suggested by Bob Vanderbei */
   if (a + b <= 0) {
     return PetscSqrtReal(a*a + b*b) - (a + b);
   }
   return -2.0*a*b / (PetscSqrtReal(a*a + b*b) + (a + b));
}

/*@
   VecFischer - Evaluates the Fischer-Burmeister function for complementarity
   problems.

   Logically Collective on vectors

   Input Parameters:
+  X - current point
.  F - function evaluated at x
.  L - lower bounds
-  U - upper bounds

   Output Parameter:
.  FB - The Fischer-Burmeister function vector

   Notes:
   The Fischer-Burmeister function is defined as
$        phi(a,b) := sqrt(a*a + b*b) - a - b
   and is used reformulate a complementarity problem as a semismooth
   system of equations.

   The result of this function is done by cases:
+  l[i] == -infinity, u[i] == infinity  -- fb[i] = -f[i]
.  l[i] == -infinity, u[i] finite       -- fb[i] = phi(u[i]-x[i], -f[i])
.  l[i] finite,       u[i] == infinity  -- fb[i] = phi(x[i]-l[i],  f[i])
.  l[i] finite < u[i] finite -- fb[i] = phi(x[i]-l[i], phi(u[i]-x[i], -f[u]))
-  otherwise l[i] == u[i] -- fb[i] = l[i] - x[i]

   Level: developer

@*/
PetscErrorCode VecFischer(Vec X, Vec F, Vec L, Vec U, Vec FB)
{
  const PetscScalar *x, *f, *l, *u;
  PetscScalar       *fb;
  PetscReal         xval, fval, lval, uval;
  PetscErrorCode    ierr;
  PetscInt          low[5], high[5], n, i;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(X, VEC_CLASSID,1);
  PetscValidHeaderSpecific(F, VEC_CLASSID,2);
  PetscValidHeaderSpecific(L, VEC_CLASSID,3);
  PetscValidHeaderSpecific(U, VEC_CLASSID,4);
  PetscValidHeaderSpecific(FB, VEC_CLASSID,5);

  ierr = VecGetOwnershipRange(X, low, high);CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(F, low + 1, high + 1);CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(L, low + 2, high + 2);CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(U, low + 3, high + 3);CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(FB, low + 4, high + 4);CHKERRQ(ierr);

  for (i = 1; i < 4; ++i) {
    if (low[0] != low[i] || high[0] != high[i]) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Vectors must be identically loaded over processors");
  }

  ierr = VecGetArrayRead(X, &x);CHKERRQ(ierr);
  ierr = VecGetArrayRead(F, &f);CHKERRQ(ierr);
  ierr = VecGetArrayRead(L, &l);CHKERRQ(ierr);
  ierr = VecGetArrayRead(U, &u);CHKERRQ(ierr);
  ierr = VecGetArray(FB, &fb);CHKERRQ(ierr);

  ierr = VecGetLocalSize(X, &n);CHKERRQ(ierr);

  for (i = 0; i < n; ++i) {
    xval = PetscRealPart(x[i]); fval = PetscRealPart(f[i]);
    lval = PetscRealPart(l[i]); uval = PetscRealPart(u[i]);

    if ((lval <= -PETSC_INFINITY) && (uval >= PETSC_INFINITY)) {
      fb[i] = -fval;
    } else if (lval <= -PETSC_INFINITY) {
      fb[i] = -Fischer(uval - xval, -fval);
    } else if (uval >=  PETSC_INFINITY) {
      fb[i] =  Fischer(xval - lval,  fval);
    } else if (lval == uval) {
      fb[i] = lval - xval;
    } else {
      fval  =  Fischer(uval - xval, -fval);
      fb[i] =  Fischer(xval - lval,  fval);
    }
  }

  ierr = VecRestoreArrayRead(X, &x);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(F, &f);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(L, &l);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(U, &u);CHKERRQ(ierr);
  ierr = VecRestoreArray(FB, &fb);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscReal SFischer(PetscReal a, PetscReal b, PetscReal c)
{
  /* Method suggested by Bob Vanderbei */
   if (a + b <= 0) {
     return PetscSqrtReal(a*a + b*b + 2.0*c*c) - (a + b);
   }
   return 2.0*(c*c - a*b) / (PetscSqrtReal(a*a + b*b + 2.0*c*c) + (a + b));
}

/*@
   VecSFischer - Evaluates the Smoothed Fischer-Burmeister function for
   complementarity problems.

   Logically Collective on vectors

   Input Parameters:
+  X - current point
.  F - function evaluated at x
.  L - lower bounds
.  U - upper bounds
-  mu - smoothing parameter

   Output Parameter:
.  FB - The Smoothed Fischer-Burmeister function vector

   Notes:
   The Smoothed Fischer-Burmeister function is defined as
$        phi(a,b) := sqrt(a*a + b*b + 2*mu*mu) - a - b
   and is used reformulate a complementarity problem as a semismooth
   system of equations.

   The result of this function is done by cases:
+  l[i] == -infinity, u[i] == infinity  -- fb[i] = -f[i] - 2*mu*x[i]
.  l[i] == -infinity, u[i] finite       -- fb[i] = phi(u[i]-x[i], -f[i], mu)
.  l[i] finite,       u[i] == infinity  -- fb[i] = phi(x[i]-l[i],  f[i], mu)
.  l[i] finite < u[i] finite -- fb[i] = phi(x[i]-l[i], phi(u[i]-x[i], -f[u], mu), mu)
-  otherwise l[i] == u[i] -- fb[i] = l[i] - x[i]

   Level: developer

.seealso  VecFischer()
@*/
PetscErrorCode VecSFischer(Vec X, Vec F, Vec L, Vec U, PetscReal mu, Vec FB)
{
  const PetscScalar *x, *f, *l, *u;
  PetscScalar       *fb;
  PetscReal         xval, fval, lval, uval;
  PetscErrorCode    ierr;
  PetscInt          low[5], high[5], n, i;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(X, VEC_CLASSID,1);
  PetscValidHeaderSpecific(F, VEC_CLASSID,2);
  PetscValidHeaderSpecific(L, VEC_CLASSID,3);
  PetscValidHeaderSpecific(U, VEC_CLASSID,4);
  PetscValidHeaderSpecific(FB, VEC_CLASSID,6);

  ierr = VecGetOwnershipRange(X, low, high);CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(F, low + 1, high + 1);CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(L, low + 2, high + 2);CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(U, low + 3, high + 3);CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(FB, low + 4, high + 4);CHKERRQ(ierr);

  for (i = 1; i < 4; ++i) {
    if (low[0] != low[i] || high[0] != high[i]) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Vectors must be identically loaded over processors");
  }

  ierr = VecGetArrayRead(X, &x);CHKERRQ(ierr);
  ierr = VecGetArrayRead(F, &f);CHKERRQ(ierr);
  ierr = VecGetArrayRead(L, &l);CHKERRQ(ierr);
  ierr = VecGetArrayRead(U, &u);CHKERRQ(ierr);
  ierr = VecGetArray(FB, &fb);CHKERRQ(ierr);

  ierr = VecGetLocalSize(X, &n);CHKERRQ(ierr);

  for (i = 0; i < n; ++i) {
    xval = PetscRealPart(*x++); fval = PetscRealPart(*f++);
    lval = PetscRealPart(*l++); uval = PetscRealPart(*u++);

    if ((lval <= -PETSC_INFINITY) && (uval >= PETSC_INFINITY)) {
      (*fb++) = -fval - mu*xval;
    } else if (lval <= -PETSC_INFINITY) {
      (*fb++) = -SFischer(uval - xval, -fval, mu);
    } else if (uval >=  PETSC_INFINITY) {
      (*fb++) =  SFischer(xval - lval,  fval, mu);
    } else if (lval == uval) {
      (*fb++) = lval - xval;
    } else {
      fval    =  SFischer(uval - xval, -fval, mu);
      (*fb++) =  SFischer(xval - lval,  fval, mu);
    }
  }
  x -= n; f -= n; l -=n; u -= n; fb -= n;

  ierr = VecRestoreArrayRead(X, &x);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(F, &f);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(L, &l);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(U, &u);CHKERRQ(ierr);
  ierr = VecRestoreArray(FB, &fb);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscReal fischnorm(PetscReal a, PetscReal b)
{
  return PetscSqrtReal(a*a + b*b);
}

PETSC_STATIC_INLINE PetscReal fischsnorm(PetscReal a, PetscReal b, PetscReal c)
{
  return PetscSqrtReal(a*a + b*b + 2.0*c*c);
}

/*@
   MatDFischer - Calculates an element of the B-subdifferential of the
   Fischer-Burmeister function for complementarity problems.

   Collective on jac

   Input Parameters:
+  jac - the jacobian of f at X
.  X - current point
.  Con - constraints function evaluated at X
.  XL - lower bounds
.  XU - upper bounds
.  t1 - work vector
-  t2 - work vector

   Output Parameters:
+  Da - diagonal perturbation component of the result
-  Db - row scaling component of the result

   Level: developer

.seealso: VecFischer()
@*/
PetscErrorCode MatDFischer(Mat jac, Vec X, Vec Con, Vec XL, Vec XU, Vec T1, Vec T2, Vec Da, Vec Db)
{
  PetscErrorCode    ierr;
  PetscInt          i,nn;
  const PetscScalar *x,*f,*l,*u,*t2;
  PetscScalar       *da,*db,*t1;
  PetscReal          ai,bi,ci,di,ei;

  PetscFunctionBegin;
  ierr = VecGetLocalSize(X,&nn);CHKERRQ(ierr);
  ierr = VecGetArrayRead(X,&x);CHKERRQ(ierr);
  ierr = VecGetArrayRead(Con,&f);CHKERRQ(ierr);
  ierr = VecGetArrayRead(XL,&l);CHKERRQ(ierr);
  ierr = VecGetArrayRead(XU,&u);CHKERRQ(ierr);
  ierr = VecGetArray(Da,&da);CHKERRQ(ierr);
  ierr = VecGetArray(Db,&db);CHKERRQ(ierr);
  ierr = VecGetArray(T1,&t1);CHKERRQ(ierr);
  ierr = VecGetArrayRead(T2,&t2);CHKERRQ(ierr);

  for (i = 0; i < nn; i++) {
    da[i] = 0.0;
    db[i] = 0.0;
    t1[i] = 0.0;

    if (PetscAbsScalar(f[i]) <= PETSC_MACHINE_EPSILON) {
      if (PetscRealPart(l[i]) > PETSC_NINFINITY && PetscAbsScalar(x[i] - l[i]) <= PETSC_MACHINE_EPSILON) {
        t1[i] = 1.0;
        da[i] = 1.0;
      }

      if (PetscRealPart(u[i]) <  PETSC_INFINITY && PetscAbsScalar(u[i] - x[i]) <= PETSC_MACHINE_EPSILON) {
        t1[i] = 1.0;
        db[i] = 1.0;
      }
    }
  }

  ierr = VecRestoreArray(T1,&t1);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(T2,&t2);CHKERRQ(ierr);
  ierr = MatMult(jac,T1,T2);CHKERRQ(ierr);
  ierr = VecGetArrayRead(T2,&t2);CHKERRQ(ierr);

  for (i = 0; i < nn; i++) {
    if ((PetscRealPart(l[i]) <= PETSC_NINFINITY) && (PetscRealPart(u[i]) >= PETSC_INFINITY)) {
      da[i] = 0.0;
      db[i] = -1.0;
    } else if (PetscRealPart(l[i]) <= PETSC_NINFINITY) {
      if (PetscRealPart(db[i]) >= 1) {
        ai = fischnorm(1.0, PetscRealPart(t2[i]));

        da[i] = -1.0 / ai - 1.0;
        db[i] = -t2[i] / ai - 1.0;
      } else {
        bi = PetscRealPart(u[i]) - PetscRealPart(x[i]);
        ai = fischnorm(bi, PetscRealPart(f[i]));
        ai = PetscMax(PETSC_MACHINE_EPSILON, ai);

        da[i] = bi / ai - 1.0;
        db[i] = -f[i] / ai - 1.0;
      }
    } else if (PetscRealPart(u[i]) >=  PETSC_INFINITY) {
      if (PetscRealPart(da[i]) >= 1) {
        ai = fischnorm(1.0, PetscRealPart(t2[i]));

        da[i] = 1.0 / ai - 1.0;
        db[i] = t2[i] / ai - 1.0;
      } else {
        bi = PetscRealPart(x[i]) - PetscRealPart(l[i]);
        ai = fischnorm(bi, PetscRealPart(f[i]));
        ai = PetscMax(PETSC_MACHINE_EPSILON, ai);

        da[i] = bi / ai - 1.0;
        db[i] = f[i] / ai - 1.0;
      }
    } else if (PetscRealPart(l[i]) == PetscRealPart(u[i])) {
      da[i] = -1.0;
      db[i] = 0.0;
    } else {
      if (PetscRealPart(db[i]) >= 1) {
        ai = fischnorm(1.0, PetscRealPart(t2[i]));

        ci = 1.0 / ai + 1.0;
        di = PetscRealPart(t2[i]) / ai + 1.0;
      } else {
        bi = PetscRealPart(x[i]) - PetscRealPart(u[i]);
        ai = fischnorm(bi, PetscRealPart(f[i]));
        ai = PetscMax(PETSC_MACHINE_EPSILON, ai);

        ci = bi / ai + 1.0;
        di = PetscRealPart(f[i]) / ai + 1.0;
      }

      if (PetscRealPart(da[i]) >= 1) {
        bi = ci + di*PetscRealPart(t2[i]);
        ai = fischnorm(1.0, bi);

        bi = bi / ai - 1.0;
        ai = 1.0 / ai - 1.0;
      } else {
        ei = Fischer(PetscRealPart(u[i]) - PetscRealPart(x[i]), -PetscRealPart(f[i]));
        ai = fischnorm(PetscRealPart(x[i]) - PetscRealPart(l[i]), ei);
        ai = PetscMax(PETSC_MACHINE_EPSILON, ai);

        bi = ei / ai - 1.0;
        ai = (PetscRealPart(x[i]) - PetscRealPart(l[i])) / ai - 1.0;
      }

      da[i] = ai + bi*ci;
      db[i] = bi*di;
    }
  }

  ierr = VecRestoreArray(Da,&da);CHKERRQ(ierr);
  ierr = VecRestoreArray(Db,&db);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(X,&x);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(Con,&f);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(XL,&l);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(XU,&u);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(T2,&t2);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
   MatDSFischer - Calculates an element of the B-subdifferential of the
   smoothed Fischer-Burmeister function for complementarity problems.

   Collective on jac

   Input Parameters:
+  jac - the jacobian of f at X
.  X - current point
.  F - constraint function evaluated at X
.  XL - lower bounds
.  XU - upper bounds
.  mu - smoothing parameter
.  T1 - work vector
-  T2 - work vector

   Output Parameters:
+  Da - diagonal perturbation component of the result
.  Db - row scaling component of the result
-  Dm - derivative with respect to scaling parameter

   Level: developer

.seealso MatDFischer()
@*/
PetscErrorCode MatDSFischer(Mat jac, Vec X, Vec Con,Vec XL, Vec XU, PetscReal mu,Vec T1, Vec T2,Vec Da, Vec Db, Vec Dm)
{
  PetscErrorCode    ierr;
  PetscInt          i,nn;
  const PetscScalar *x, *f, *l, *u;
  PetscScalar       *da, *db, *dm;
  PetscReal         ai, bi, ci, di, ei, fi;

  PetscFunctionBegin;
  if (PetscAbsReal(mu) <= PETSC_MACHINE_EPSILON) {
    ierr = VecZeroEntries(Dm);CHKERRQ(ierr);
    ierr = MatDFischer(jac, X, Con, XL, XU, T1, T2, Da, Db);CHKERRQ(ierr);
  } else {
    ierr = VecGetLocalSize(X,&nn);CHKERRQ(ierr);
    ierr = VecGetArrayRead(X,&x);CHKERRQ(ierr);
    ierr = VecGetArrayRead(Con,&f);CHKERRQ(ierr);
    ierr = VecGetArrayRead(XL,&l);CHKERRQ(ierr);
    ierr = VecGetArrayRead(XU,&u);CHKERRQ(ierr);
    ierr = VecGetArray(Da,&da);CHKERRQ(ierr);
    ierr = VecGetArray(Db,&db);CHKERRQ(ierr);
    ierr = VecGetArray(Dm,&dm);CHKERRQ(ierr);

    for (i = 0; i < nn; ++i) {
      if ((PetscRealPart(l[i]) <= PETSC_NINFINITY) && (PetscRealPart(u[i]) >= PETSC_INFINITY)) {
        da[i] = -mu;
        db[i] = -1.0;
        dm[i] = -x[i];
      } else if (PetscRealPart(l[i]) <= PETSC_NINFINITY) {
        bi = PetscRealPart(u[i]) - PetscRealPart(x[i]);
        ai = fischsnorm(bi, PetscRealPart(f[i]), mu);
        ai = PetscMax(PETSC_MACHINE_EPSILON, ai);

        da[i] = bi / ai - 1.0;
        db[i] = -PetscRealPart(f[i]) / ai - 1.0;
        dm[i] = 2.0 * mu / ai;
      } else if (PetscRealPart(u[i]) >=  PETSC_INFINITY) {
        bi = PetscRealPart(x[i]) - PetscRealPart(l[i]);
        ai = fischsnorm(bi, PetscRealPart(f[i]), mu);
        ai = PetscMax(PETSC_MACHINE_EPSILON, ai);

        da[i] = bi / ai - 1.0;
        db[i] = PetscRealPart(f[i]) / ai - 1.0;
        dm[i] = 2.0 * mu / ai;
      } else if (PetscRealPart(l[i]) == PetscRealPart(u[i])) {
        da[i] = -1.0;
        db[i] = 0.0;
        dm[i] = 0.0;
      } else {
        bi = PetscRealPart(x[i]) - PetscRealPart(u[i]);
        ai = fischsnorm(bi, PetscRealPart(f[i]), mu);
        ai = PetscMax(PETSC_MACHINE_EPSILON, ai);

        ci = bi / ai + 1.0;
        di = PetscRealPart(f[i]) / ai + 1.0;
        fi = 2.0 * mu / ai;

        ei = SFischer(PetscRealPart(u[i]) - PetscRealPart(x[i]), -PetscRealPart(f[i]), mu);
        ai = fischsnorm(PetscRealPart(x[i]) - PetscRealPart(l[i]), ei, mu);
        ai = PetscMax(PETSC_MACHINE_EPSILON, ai);

        bi = ei / ai - 1.0;
        ei = 2.0 * mu / ei;
        ai = (PetscRealPart(x[i]) - PetscRealPart(l[i])) / ai - 1.0;

        da[i] = ai + bi*ci;
        db[i] = bi*di;
        dm[i] = ei + bi*fi;
      }
    }

    ierr = VecRestoreArrayRead(X,&x);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(Con,&f);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(XL,&l);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(XU,&u);CHKERRQ(ierr);
    ierr = VecRestoreArray(Da,&da);CHKERRQ(ierr);
    ierr = VecRestoreArray(Db,&db);CHKERRQ(ierr);
    ierr = VecRestoreArray(Dm,&dm);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscReal ST_InternalPN(PetscScalar in, PetscReal lb, PetscReal ub)
{
  return PetscMax(0,(PetscReal)PetscRealPart(in)-ub) - PetscMax(0,-(PetscReal)PetscRealPart(in)-PetscAbsReal(lb));
}

PETSC_STATIC_INLINE PetscReal ST_InternalNN(PetscScalar in, PetscReal lb, PetscReal ub)
{
  return PetscMax(0,(PetscReal)PetscRealPart(in) + PetscAbsReal(ub)) - PetscMax(0,-(PetscReal)PetscRealPart(in) - PetscAbsReal(lb));
}

PETSC_STATIC_INLINE PetscReal ST_InternalPP(PetscScalar in, PetscReal lb, PetscReal ub)
{
  return PetscMax(0, (PetscReal)PetscRealPart(in)-ub) + PetscMin(0, (PetscReal)PetscRealPart(in) - lb);
}

/*@
   TaoSoftThreshold - Calculates soft thresholding routine with input vector
   and given lower and upper bound and returns it to output vector.

   Input Parameters:
+  in - input vector to be thresholded
.  lb - lower bound
-  ub - upper bound

   Output Parameter:
.  out - Soft thresholded output vector

   Notes:
   Soft thresholding is defined as
   \[ S(input,lb,ub) =
     \begin{cases}
    input - ub  \text{input > ub} \\
    0           \text{lb =< input <= ub} \\
    input + lb  \text{input < lb} \\
   \]

   Level: developer

@*/
PetscErrorCode TaoSoftThreshold(Vec in, PetscReal lb, PetscReal ub, Vec out)
{
  PetscErrorCode ierr;
  PetscInt       i, nlocal, mlocal;
  PetscScalar   *inarray, *outarray;

  PetscFunctionBegin;
  ierr = VecGetArrayPair(in, out, &inarray, &outarray);CHKERRQ(ierr);
  ierr = VecGetLocalSize(in, &nlocal);CHKERRQ(ierr);
  ierr = VecGetLocalSize(in, &mlocal);CHKERRQ(ierr);

  if (nlocal != mlocal) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, "Input and output vectors need to be of same size.");
  if (lb == ub) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, "Lower bound and upper bound need to be different.");
  if (lb > ub) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, "Lower bound needs to be lower than upper bound.");

  if (ub >= 0 && lb < 0) {
    for (i=0; i<nlocal; i++) outarray[i] = ST_InternalPN(inarray[i], lb, ub);
  } else if (ub < 0 && lb < 0) {
    for (i=0; i<nlocal; i++) outarray[i] = ST_InternalNN(inarray[i], lb, ub);
  } else {
    for (i=0; i<nlocal; i++) outarray[i] = ST_InternalPP(inarray[i], lb, ub);
  }

  ierr = VecRestoreArrayPair(in, out, &inarray, &outarray);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
