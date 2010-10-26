#include "tao.h"
#include "tao_util.h"

#undef __FUNCT__  
#define __FUNCT__ "VecPow"
PetscErrorCode VecPow(Vec Vec1, PetscScalar p)
{
  PetscErrorCode ierr;
  PetscInt n,i;
  PetscReal *v1;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(Vec1, VEC_CLASSID, 1); 

  ierr = VecGetArray(Vec1, &v1); CHKERRQ(ierr);
  ierr = VecGetLocalSize(Vec1, &n); CHKERRQ(ierr);

  if (1.0 == p) {
  }
  else if (-1.0 == p) {
    for (i = 0; i < n; ++i){
      v1[i] = 1.0 / v1[i];
    }
  }
  else if (0.0 == p) {
    for (i = 0; i < n; ++i){
      // Not-a-number left alone
      // Infinity set to one 
      if (v1[i] == v1[i]) {
        v1[i] = 1.0;
      }
    }
  }
  else if (0.5 == p) {
    for (i = 0; i < n; ++i) {
      if (v1[i] >= 0) {
        v1[i] = sqrt(v1[i]);
      }
      else {
        v1[i] = TAO_INFINITY;
      }
    }
  }
  else if (-0.5 == p) {
    for (i = 0; i < n; ++i) {
      if (v1[i] >= 0) {
        v1[i] = 1.0 / sqrt(v1[i]);
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
        v1[i] = pow(v1[i], p);
      }
      else {
        v1[i] = TAO_INFINITY;
      }
    }
  }

  ierr = VecRestoreArray(Vec1,&v1); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------- */
#undef __FUNCT__  
#define __FUNCT__ "VecMedian"
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


#undef __FUNCT__  
#define __FUNCT__ "VecCompare"
PetscErrorCode TAOSOLVER_DLLEXPORT VecCompare(Vec V1,Vec V2, PetscBool *flg){
  PetscErrorCode ierr;
  PetscInt n1,n2,N1,N2;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(V1,VEC_CLASSID,1); 
  PetscValidHeaderSpecific(V2,VEC_CLASSID,2); 
  ierr = VecGetSize(V1,&N1);CHKERRQ(ierr);
  ierr = VecGetSize(V2,&N2);CHKERRQ(ierr);
  ierr = VecGetLocalSize(V1,&n1);CHKERRQ(ierr);
  ierr = VecGetLocalSize(V2,&n2);CHKERRQ(ierr);
  if (N1==N2 && n1==n2) 
    *flg=PETSC_TRUE;
  else
    *flg=PETSC_FALSE;

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "Fischer"
inline static PetscScalar Fischer(PetscScalar a, PetscScalar b)
{
   // Method suggested by Bob Vanderbei
   if (a + b <= 0) {
     return sqrt(a*a + b*b) - (a + b);
   }
   return -2.0*a*b / (sqrt(a*a + b*b) + (a + b));
}

#undef __FUNCT__  
#define __FUNCT__ "VecFischer"
PetscErrorCode TAOSOLVER_DLLEXPORT VecFischer(Vec X, Vec F, Vec L, Vec U, Vec FF)
{
  PetscScalar *x, *f, *l, *u, *ff;
  PetscScalar xval, fval, lval, uval;
  PetscErrorCode ierr;
  PetscInt low[5], high[5], n, i;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(X, VEC_CLASSID,1); 
  PetscValidHeaderSpecific(F, VEC_CLASSID,2); 
  PetscValidHeaderSpecific(L, VEC_CLASSID,3); 
  PetscValidHeaderSpecific(U, VEC_CLASSID,4); 
  PetscValidHeaderSpecific(FF, VEC_CLASSID,4); 

  ierr = VecGetOwnershipRange(X, low, high); CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(F, low + 1, high + 1); CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(L, low + 2, high + 2); CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(U, low + 3, high + 3); CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(FF, low + 4, high + 4); CHKERRQ(ierr);

  for (i = 1; i < 4; ++i) {
    if (low[0] != low[i] || high[0] != high[i])
      SETERRQ(PETSC_COMM_SELF,1,"Vectors must be identically loaded over processors");
  }

  ierr = VecGetArray(X, &x); CHKERRQ(ierr);
  ierr = VecGetArray(F, &f); CHKERRQ(ierr);
  ierr = VecGetArray(L, &l); CHKERRQ(ierr);
  ierr = VecGetArray(U, &u); CHKERRQ(ierr);
  ierr = VecGetArray(FF, &ff); CHKERRQ(ierr);

  ierr = VecGetLocalSize(X, &n); CHKERRQ(ierr);

  for (i = 0; i < n; ++i) {
    xval = x[i]; fval = f[i];
    lval = l[i]; uval = u[i];

    if ((lval <= -TAO_INFINITY) && (uval >= TAO_INFINITY)) {
      ff[i] = -fval;
    } 
    else if (lval <= -TAO_INFINITY) {
      ff[i] = -Fischer(uval - xval, -fval);
    } 
    else if (uval >=  TAO_INFINITY) {
      ff[i] =  Fischer(xval - lval,  fval);
    } 
    else if (lval == uval) {
      ff[i] = lval - xval;
    }
    else {
      fval  =  Fischer(uval - xval, -fval);
      ff[i] =  Fischer(xval - lval,  fval);
    }
  }
  
  ierr = VecRestoreArray(X, &x); CHKERRQ(ierr);
  ierr = VecRestoreArray(F, &f); CHKERRQ(ierr);
  ierr = VecRestoreArray(L, &l); CHKERRQ(ierr);
  ierr = VecRestoreArray(U, &u); CHKERRQ(ierr);
  ierr = VecRestoreArray(FF, &ff); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SFischer"
inline static PetscScalar SFischer(PetscScalar a, PetscScalar b, PetscScalar c)
{
   // Method suggested by Bob Vanderbei
   if (a + b <= 0) {
     return sqrt(a*a + b*b + 2.0*c*c) - (a + b);
   }
   return 2.0*(c*c - a*b) / (sqrt(a*a + b*b + 2.0*c*c) + (a + b));
}

#undef __FUNCT__
#define __FUNCT__ "VecSFischer"
PetscErrorCode TAOSOLVER_DLLEXPORT VecSFischer(Vec X, Vec F, Vec L, Vec U, PetscScalar mu, Vec FF)
{
  PetscScalar *x, *f, *l, *u, *ff;
  PetscScalar xval, fval, lval, uval;
  PetscErrorCode ierr;
  PetscInt low[5], high[5], n, i;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(X, VEC_CLASSID,1);
  PetscValidHeaderSpecific(F, VEC_CLASSID,2);
  PetscValidHeaderSpecific(L, VEC_CLASSID,3);
  PetscValidHeaderSpecific(U, VEC_CLASSID,4);
  PetscValidHeaderSpecific(FF, VEC_CLASSID,6);

  ierr = VecGetOwnershipRange(X, low, high); CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(F, low + 1, high + 1); CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(L, low + 2, high + 2); CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(U, low + 3, high + 3); CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(FF, low + 4, high + 4); CHKERRQ(ierr);

  for (i = 1; i < 4; ++i) {
    if (low[0] != low[i] || high[0] != high[i])
      SETERRQ(PETSC_COMM_SELF,1,"Vectors must be identically loaded over processors");
  }

  ierr = VecGetArray(X, &x); CHKERRQ(ierr);
  ierr = VecGetArray(F, &f); CHKERRQ(ierr);
  ierr = VecGetArray(L, &l); CHKERRQ(ierr);
  ierr = VecGetArray(U, &u); CHKERRQ(ierr);
  ierr = VecGetArray(FF, &ff); CHKERRQ(ierr);

  ierr = VecGetLocalSize(X, &n); CHKERRQ(ierr);

  for (i = 0; i < n; ++i) {
    xval = (*x++); fval = (*f++);
    lval = (*l++); uval = (*u++);

    if ((lval <= -TAO_INFINITY) && (uval >= TAO_INFINITY)) {
      (*ff++) = -fval - mu*xval;
    } 
    else if (lval <= -TAO_INFINITY) {
      (*ff++) = -SFischer(uval - xval, -fval, mu);
    } 
    else if (uval >=  TAO_INFINITY) {
      (*ff++) =  SFischer(xval - lval,  fval, mu);
    } 
    else if (lval == uval) {
      (*ff++) = lval - xval;
    } 
    else {
      fval    =  SFischer(uval - xval, -fval, mu);
      (*ff++) =  SFischer(xval - lval,  fval, mu);
    }
  }
  x -= n; f -= n; l -=n; u -= n; ff -= n;

  ierr = VecRestoreArray(X, &x); CHKERRQ(ierr);
  ierr = VecRestoreArray(F, &f); CHKERRQ(ierr);
  ierr = VecRestoreArray(L, &l); CHKERRQ(ierr);
  ierr = VecRestoreArray(U, &u); CHKERRQ(ierr);
  ierr = VecRestoreArray(FF, &ff); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

