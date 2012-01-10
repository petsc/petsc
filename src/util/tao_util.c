#include "tao.h" /*I "tao.h" I*/
#include "tao_util.h" /*I "tao_util.h" I*/

#undef __FUNCT__  
#define __FUNCT__ "VecPow"
/*@
  VecPow - Replaces each component of a vector by x_i^p

  Logically Collective on v

  Input Parameter:
+ v - the vector
- p - the exponent to use on each element

  Output Parameter:
. v - the vector

  Level: intermediate

@*/
PetscErrorCode VecPow(Vec v, PetscReal p)
{
  PetscErrorCode ierr;
  PetscInt n,i;
  PetscReal *v1;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(v, VEC_CLASSID, 1); 

  ierr = VecGetArray(v, &v1); CHKERRQ(ierr);
  ierr = VecGetLocalSize(v, &n); CHKERRQ(ierr);

  if (1.0 == p) {
  }
  else if (-1.0 == p) {
    for (i = 0; i < n; ++i){
      v1[i] = 1.0 / v1[i];
    }
  }
  else if (0.0 == p) {
    for (i = 0; i < n; ++i){
      /*  Not-a-number left alone
	  Infinity set to one  */
      if (v1[i] == v1[i]) {
        v1[i] = 1.0;
      }
    }
  }
  else if (0.5 == p) {
    for (i = 0; i < n; ++i) {
      if (v1[i] >= 0) {
        v1[i] = PetscSqrtScalar(v1[i]);
      }
      else {
        v1[i] = TAO_INFINITY;
      }
    }
  }
  else if (-0.5 == p) {
    for (i = 0; i < n; ++i) {
      if (v1[i] >= 0) {
        v1[i] = 1.0 / PetscSqrtScalar(v1[i]);
      }
      else {
        v1[i] = TAO_INFINITY;
      }
    }
  }
  else if (2.0 == p) {
    for (i = 0; i < n; ++i){
      v1[i] *= v1[i];
    }
  }
  else if (-2.0 == p) {
    for (i = 0; i < n; ++i){
      v1[i] = 1.0 / (v1[i] * v1[i]);
    }
  }
  else {
    for (i = 0; i < n; ++i) {
      if (v1[i] >= 0) {
        v1[i] = PetscPowScalar(v1[i], p);
      }
      else {
        v1[i] = TAO_INFINITY;
      }
    }
  }

  ierr = VecRestoreArray(v,&v1); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "VecMedian"
/*@
  VecMedian - Computes the componentwise median of three vectors
  and stores the result in this vector.  Used primarily for projecting
  a vector within upper and lower bounds. 

  Logically Collective 

  Input Parameters:
. Vec1, Vec2, Vec3 - The three vectors

  Output Parameter:
. VMedian - The median vector

  Level: advanced
@*/
PetscErrorCode VecMedian(Vec Vec1, Vec Vec2, Vec Vec3, Vec VMedian)
{
  PetscErrorCode ierr;
  PetscInt i,n,low1,low2,low3,low4,high1,high2,high3,high4;
  PetscReal *v1,*v2,*v3,*vmed;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(Vec1,VEC_CLASSID,1); 
  PetscValidHeaderSpecific(Vec2,VEC_CLASSID,2); 
  PetscValidHeaderSpecific(Vec3,VEC_CLASSID,3); 
  PetscValidHeaderSpecific(VMedian,VEC_CLASSID,4); 

  if (Vec1==Vec2 || Vec1==Vec3){
    ierr=VecCopy(Vec1,VMedian); CHKERRQ(ierr); 
    PetscFunctionReturn(0);
  }
  if (Vec2==Vec3){
    ierr=VecCopy(Vec2,VMedian); CHKERRQ(ierr); 
    PetscFunctionReturn(0);
  }

  PetscValidType(Vec1,1);
  PetscValidType(Vec2,2);
  PetscValidType(VMedian,4);
  PetscCheckSameType(Vec1,1,Vec2,2); PetscCheckSameType(Vec1,1,VMedian,4);
  PetscCheckSameComm(Vec1,1,Vec2,2); PetscCheckSameComm(Vec1,1,VMedian,4);

  ierr = VecGetOwnershipRange(Vec1, &low1, &high1); CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(Vec2, &low2, &high2); CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(Vec3, &low3, &high3); CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(VMedian, &low4, &high4); CHKERRQ(ierr);
  if ( low1!= low2 || low1!= low3 || low1!= low4 ||
       high1!= high2 || high1!= high3 || high1!= high4)
    SETERRQ(PETSC_COMM_SELF,1,"InCompatible vector local lengths");

  ierr = VecGetArray(Vec1,&v1); CHKERRQ(ierr);
  ierr = VecGetArray(Vec2,&v2); CHKERRQ(ierr);
  ierr = VecGetArray(Vec3,&v3); CHKERRQ(ierr);

  if ( VMedian != Vec1 && VMedian != Vec2 && VMedian != Vec3){
    ierr = VecGetArray(VMedian,&vmed); CHKERRQ(ierr);
  } else if ( VMedian==Vec1 ){
    vmed=v1;
  } else if ( VMedian==Vec2 ){
    vmed=v2;
  } else {
    vmed=v3;
  }

  ierr=VecGetLocalSize(Vec1,&n); CHKERRQ(ierr);

  for (i=0;i<n;i++){
    vmed[i]=PetscMax(PetscMax(PetscMin(v1[i],v2[i]),PetscMin(v1[i],v3[i])),PetscMin(v2[i],v3[i]));
  }

  ierr = VecRestoreArray(Vec1,&v1); CHKERRQ(ierr);
  ierr = VecRestoreArray(Vec2,&v2); CHKERRQ(ierr);
  ierr = VecRestoreArray(Vec3,&v2); CHKERRQ(ierr);
  
  if (VMedian!=Vec1 && VMedian != Vec2 && VMedian != Vec3){
    ierr = VecRestoreArray(VMedian,&vmed); CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscReal Fischer(PetscReal a, PetscReal b)
{
  /* Method suggested by Bob Vanderbei */
   if (a + b <= 0) {
     return PetscSqrtScalar(a*a + b*b) - (a + b);
   }
   return -2.0*a*b / (PetscSqrtScalar(a*a + b*b) + (a + b));
}

#undef __FUNCT__  
#define __FUNCT__ "VecFischer"
/*@
   VecFischer - Evaluates the Fischer-Burmeister function for complementarity 
   problems.

   Logically Collective on vectors

   Input Parameters:
+  X - current point
.  F - function evaluated at x
.  L - lower bounds 
-  U - upper bounds

   Output Parameters:
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
  PetscReal *x, *f, *l, *u, *fb;
  PetscReal xval, fval, lval, uval;
  PetscErrorCode ierr;
  PetscInt low[5], high[5], n, i;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(X, VEC_CLASSID,1); 
  PetscValidHeaderSpecific(F, VEC_CLASSID,2); 
  PetscValidHeaderSpecific(L, VEC_CLASSID,3); 
  PetscValidHeaderSpecific(U, VEC_CLASSID,4); 
  PetscValidHeaderSpecific(FB, VEC_CLASSID,4); 

  ierr = VecGetOwnershipRange(X, low, high); CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(F, low + 1, high + 1); CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(L, low + 2, high + 2); CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(U, low + 3, high + 3); CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(FB, low + 4, high + 4); CHKERRQ(ierr);

  for (i = 1; i < 4; ++i) {
    if (low[0] != low[i] || high[0] != high[i])
      SETERRQ(PETSC_COMM_SELF,1,"Vectors must be identically loaded over processors");
  }

  ierr = VecGetArray(X, &x); CHKERRQ(ierr);
  ierr = VecGetArray(F, &f); CHKERRQ(ierr);
  ierr = VecGetArray(L, &l); CHKERRQ(ierr);
  ierr = VecGetArray(U, &u); CHKERRQ(ierr);
  ierr = VecGetArray(FB, &fb); CHKERRQ(ierr);

  ierr = VecGetLocalSize(X, &n); CHKERRQ(ierr);

  for (i = 0; i < n; ++i) {
    xval = x[i]; fval = f[i];
    lval = l[i]; uval = u[i];

    if ((lval <= -TAO_INFINITY) && (uval >= TAO_INFINITY)) {
      fb[i] = -fval;
    } 
    else if (lval <= -TAO_INFINITY) {
      fb[i] = -Fischer(uval - xval, -fval);
    } 
    else if (uval >=  TAO_INFINITY) {
      fb[i] =  Fischer(xval - lval,  fval);
    } 
    else if (lval == uval) {
      fb[i] = lval - xval;
    }
    else {
      fval  =  Fischer(uval - xval, -fval);
      fb[i] =  Fischer(xval - lval,  fval);
    }
  }
  
  ierr = VecRestoreArray(X, &x); CHKERRQ(ierr);
  ierr = VecRestoreArray(F, &f); CHKERRQ(ierr);
  ierr = VecRestoreArray(L, &l); CHKERRQ(ierr);
  ierr = VecRestoreArray(U, &u); CHKERRQ(ierr);
  ierr = VecRestoreArray(FB, &fb); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscReal SFischer(PetscReal a, PetscReal b, PetscReal c)
{
  /* Method suggested by Bob Vanderbei */
   if (a + b <= 0) {
     return PetscSqrtScalar(a*a + b*b + 2.0*c*c) - (a + b);
   }
   return 2.0*(c*c - a*b) / (PetscSqrtScalar(a*a + b*b + 2.0*c*c) + (a + b));
}

#undef __FUNCT__
#define __FUNCT__ "VecSFischer"
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

   Output Parameters:
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
  PetscReal *x, *f, *l, *u, *fb;
  PetscReal xval, fval, lval, uval;
  PetscErrorCode ierr;
  PetscInt low[5], high[5], n, i;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(X, VEC_CLASSID,1);
  PetscValidHeaderSpecific(F, VEC_CLASSID,2);
  PetscValidHeaderSpecific(L, VEC_CLASSID,3);
  PetscValidHeaderSpecific(U, VEC_CLASSID,4);
  PetscValidHeaderSpecific(FB, VEC_CLASSID,6);

  ierr = VecGetOwnershipRange(X, low, high); CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(F, low + 1, high + 1); CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(L, low + 2, high + 2); CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(U, low + 3, high + 3); CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(FB, low + 4, high + 4); CHKERRQ(ierr);

  for (i = 1; i < 4; ++i) {
    if (low[0] != low[i] || high[0] != high[i])
      SETERRQ(PETSC_COMM_SELF,1,"Vectors must be identically loaded over processors");
  }

  ierr = VecGetArray(X, &x); CHKERRQ(ierr);
  ierr = VecGetArray(F, &f); CHKERRQ(ierr);
  ierr = VecGetArray(L, &l); CHKERRQ(ierr);
  ierr = VecGetArray(U, &u); CHKERRQ(ierr);
  ierr = VecGetArray(FB, &fb); CHKERRQ(ierr);

  ierr = VecGetLocalSize(X, &n); CHKERRQ(ierr);
  
  for (i = 0; i < n; ++i) {
    xval = (*x++); fval = (*f++);
    lval = (*l++); uval = (*u++);

    if ((lval <= -TAO_INFINITY) && (uval >= TAO_INFINITY)) {
      (*fb++) = -fval - mu*xval;
    } 
    else if (lval <= -TAO_INFINITY) {
      (*fb++) = -SFischer(uval - xval, -fval, mu);
    } 
    else if (uval >=  TAO_INFINITY) {
      (*fb++) =  SFischer(xval - lval,  fval, mu);
    } 
    else if (lval == uval) {
      (*fb++) = lval - xval;
    } 
    else {
      fval    =  SFischer(uval - xval, -fval, mu);
      (*fb++) =  SFischer(xval - lval,  fval, mu);
    }
  }
  x -= n; f -= n; l -=n; u -= n; fb -= n;

  ierr = VecRestoreArray(X, &x); CHKERRQ(ierr);
  ierr = VecRestoreArray(F, &f); CHKERRQ(ierr);
  ierr = VecRestoreArray(L, &l); CHKERRQ(ierr);
  ierr = VecRestoreArray(U, &u); CHKERRQ(ierr);
  ierr = VecRestoreArray(FB, &fb); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscReal fischnorm(PetscReal a, PetscReal b)
{
  return PetscSqrtScalar(a*a + b*b);
}

PETSC_STATIC_INLINE PetscReal fischsnorm(PetscReal a, PetscReal b, PetscReal c)
{
  return PetscSqrtScalar(a*a + b*b + 2.0*c*c);
}

#undef __FUNCT__
#define __FUNCT__ "D_Fischer"
/*@
   D_Fischer - Calculates an element of the B-subdifferential of the 
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
PetscErrorCode D_Fischer(Mat jac, Vec X, Vec Con, Vec XL, Vec XU, 
		      Vec T1, Vec T2, Vec Da, Vec Db)
{
  PetscErrorCode ierr;
  PetscInt i,nn;
  PetscReal *x,*f,*l,*u,*da,*db,*t1,*t2;
  PetscReal ai,bi,ci,di,ei;

  PetscFunctionBegin;

  ierr = VecGetLocalSize(X,&nn); CHKERRQ(ierr);
  
  ierr = VecGetArray(X,&x);CHKERRQ(ierr);
  ierr = VecGetArray(Con,&f);CHKERRQ(ierr);
  ierr = VecGetArray(XL,&l);CHKERRQ(ierr);
  ierr = VecGetArray(XU,&u);CHKERRQ(ierr);
  ierr = VecGetArray(Da,&da);CHKERRQ(ierr);
  ierr = VecGetArray(Db,&db);CHKERRQ(ierr);
  ierr = VecGetArray(T1,&t1);CHKERRQ(ierr);
  ierr = VecGetArray(T2,&t2);CHKERRQ(ierr);

  for (i = 0; i < nn; i++) {
    da[i] = 0;
    db[i] = 0;
    t1[i] = 0;

    if (PetscAbsReal(f[i]) <= PETSC_MACHINE_EPSILON) {
      if (l[i] > TAO_NINFINITY && PetscAbsReal(x[i] - l[i]) <= PETSC_MACHINE_EPSILON) {
        t1[i] = 1;
        da[i] = 1;
      }

      if (u[i] <  TAO_INFINITY && PetscAbsReal(u[i] - x[i]) <= PETSC_MACHINE_EPSILON) {
        t1[i] = 1;
        db[i] = 1;
      }
    }
  }

  ierr = VecRestoreArray(T1,&t1); CHKERRQ(ierr);
  ierr = VecRestoreArray(T2,&t2); CHKERRQ(ierr);
  ierr = MatMult(jac,T1,T2); CHKERRQ(ierr);
  ierr = VecGetArray(T2,&t2); CHKERRQ(ierr);

  for (i = 0; i < nn; i++) {
    if ((l[i] <= TAO_NINFINITY) && (u[i] >= TAO_INFINITY)) {
      da[i] = 0;
      db[i] = -1;
    } 
    else if (l[i] <= TAO_NINFINITY) {
      if (db[i] >= 1) {
        ai = fischnorm(1, t2[i]);

        da[i] = -1/ai - 1;
        db[i] = -t2[i]/ai - 1;
      } 
      else {
        bi = u[i] - x[i];
        ai = fischnorm(bi, f[i]);
        ai = PetscMax(PETSC_MACHINE_EPSILON, ai);

        da[i] = bi / ai - 1;
        db[i] = -f[i] / ai - 1;
      }
    } 
    else if (u[i] >=  TAO_INFINITY) {
      if (da[i] >= 1) {
        ai = fischnorm(1, t2[i]);

        da[i] = 1 / ai - 1;
        db[i] = t2[i] / ai - 1;
      } 
      else {
        bi = x[i] - l[i];
        ai = fischnorm(bi, f[i]);
        ai = PetscMax(PETSC_MACHINE_EPSILON, ai);

        da[i] = bi / ai - 1;
        db[i] = f[i] / ai - 1;
      }
    } 
    else if (l[i] == u[i]) {
      da[i] = -1;
      db[i] = 0;
    } 
    else {
      if (db[i] >= 1) {
        ai = fischnorm(1, t2[i]);

        ci = 1 / ai + 1;
        di = t2[i] / ai + 1;
      } 
      else {
        bi = x[i] - u[i];
        ai = fischnorm(bi, f[i]);
        ai = PetscMax(PETSC_MACHINE_EPSILON, ai);

        ci = bi / ai + 1;
        di = f[i] / ai + 1;
      }

      if (da[i] >= 1) {
        bi = ci + di*t2[i];
        ai = fischnorm(1, bi);

        bi = bi / ai - 1;
        ai = 1 / ai - 1;
      } 
      else {
        ei = Fischer(u[i] - x[i], -f[i]);
        ai = fischnorm(x[i] - l[i], ei);
        ai = PetscMax(PETSC_MACHINE_EPSILON, ai);

        bi = ei / ai - 1;
        ai = (x[i] - l[i]) / ai - 1;
      }

      da[i] = ai + bi*ci;
      db[i] = bi*di;
    }
  }

  ierr = VecRestoreArray(Da,&da); CHKERRQ(ierr);
  ierr = VecRestoreArray(Db,&db); CHKERRQ(ierr);
  ierr = VecRestoreArray(X,&x); CHKERRQ(ierr);
  ierr = VecRestoreArray(Con,&f); CHKERRQ(ierr);
  ierr = VecRestoreArray(XL,&l); CHKERRQ(ierr);
  ierr = VecRestoreArray(XU,&u); CHKERRQ(ierr);
  ierr = VecRestoreArray(T2,&t2); CHKERRQ(ierr);
  PetscFunctionReturn(0);
};

#undef __FUNCT__
#define __FUNCT__ "D_SFischer"
/*@
   D_SFischer - Calculates an element of the B-subdifferential of the
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

   Output Parameter: 
+  Da - diagonal perturbation component of the result
.  Db - row scaling component of the result
-  Dm - derivative with respect to scaling parameter

   Level: developer

.seealso D_Fischer()
@*/
PetscErrorCode D_SFischer(Mat jac, Vec X, Vec Con, 
                       Vec XL, Vec XU, PetscReal mu, 
                       Vec T1, Vec T2, 
                       Vec Da, Vec Db, Vec Dm)
{
  PetscErrorCode ierr;
  PetscInt i,nn;
  PetscReal *x, *f, *l, *u, *da, *db, *dm;
  PetscReal ai, bi, ci, di, ei, fi;

  PetscFunctionBegin;

  if (PetscAbsReal(mu) <= PETSC_MACHINE_EPSILON) {
    ierr = VecZeroEntries(Dm); CHKERRQ(ierr);
    ierr = D_Fischer(jac, X, Con, XL, XU, T1, T2, Da, Db); CHKERRQ(ierr);
  } 
  else {
    ierr = VecGetLocalSize(X,&nn); CHKERRQ(ierr);
    ierr = VecGetArray(X,&x); CHKERRQ(ierr);
    ierr = VecGetArray(Con,&f); CHKERRQ(ierr);
    ierr = VecGetArray(XL,&l); CHKERRQ(ierr);
    ierr = VecGetArray(XU,&u); CHKERRQ(ierr);
    ierr = VecGetArray(Da,&da); CHKERRQ(ierr);
    ierr = VecGetArray(Db,&db); CHKERRQ(ierr);
    ierr = VecGetArray(Dm,&dm); CHKERRQ(ierr);

    for (i = 0; i < nn; ++i) {
      if ((l[i] <= TAO_NINFINITY) && (u[i] >= TAO_INFINITY)) {
        da[i] = -mu;
        db[i] = -1;
        dm[i] = -x[i];
      } 
      else if (l[i] <= TAO_NINFINITY) {
        bi = u[i] - x[i];
        ai = fischsnorm(bi, f[i], mu);
        ai = PetscMax(PETSC_MACHINE_EPSILON, ai);

        da[i] = bi / ai - 1;
        db[i] = -f[i] / ai - 1;
        dm[i] = 2.0 * mu / ai;
      } 
      else if (u[i] >=  TAO_INFINITY) {
        bi = x[i] - l[i];
        ai = fischsnorm(bi, f[i], mu);
        ai = PetscMax(PETSC_MACHINE_EPSILON, ai);

        da[i] = bi / ai - 1;
        db[i] = f[i] / ai - 1;
        dm[i] = 2.0 * mu / ai;
      } 
      else if (l[i] == u[i]) {
        da[i] = -1;
        db[i] = 0;
        dm[i] = 0;
      } 
      else {
        bi = x[i] - u[i];
        ai = fischsnorm(bi, f[i], mu);
        ai = PetscMax(PETSC_MACHINE_EPSILON, ai);
  
        ci = bi / ai + 1;
        di = f[i] / ai + 1;
        fi = 2.0 * mu / ai;

        ei = SFischer(u[i] - x[i], -f[i], mu);
        ai = fischsnorm(x[i] - l[i], ei, mu);
        ai = PetscMax(PETSC_MACHINE_EPSILON, ai);
  
        bi = ei / ai - 1;
        ei = 2.0 * mu / ei;
        ai = (x[i] - l[i]) / ai - 1;
  
        da[i] = ai + bi*ci;
        db[i] = bi*di;
        dm[i] = ei + bi*fi;
      }
    }
    
    ierr = VecRestoreArray(X,&x); CHKERRQ(ierr);
    ierr = VecRestoreArray(Con,&f); CHKERRQ(ierr);
    ierr = VecRestoreArray(XL,&l); CHKERRQ(ierr);
    ierr = VecRestoreArray(XU,&u); CHKERRQ(ierr);
    ierr = VecRestoreArray(Da,&da); CHKERRQ(ierr);
    ierr = VecRestoreArray(Db,&db); CHKERRQ(ierr);
    ierr = VecRestoreArray(Dm,&dm); CHKERRQ(ierr);

  }
  PetscFunctionReturn(0);
}

