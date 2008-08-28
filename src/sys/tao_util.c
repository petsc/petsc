#include "tao.h"
#include "tao_util.h"

#undef __FUNCT__  
#define __FUNCT__ "VecPow"
PetscErrorCode VecPow(Vec Vec1, PetscScalar p)
{
  PetscErrorCode ierr;
  PetscInt n,i;
  PetscScalar *v1;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(Vec1, VEC_COOKIE, 1); 

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
int VecMedian(Vec Vec1, Vec Vec2, Vec Vec3, Vec VMedian)
{
  int ierr;
  PetscInt i,n,low1,low2,low3,low4,high1,high2,high3,high4;
  PetscScalar *v1,*v2,*v3,*vmed;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(Vec1,VEC_COOKIE,1); 
  PetscValidHeaderSpecific(Vec2,VEC_COOKIE,2); 
  PetscValidHeaderSpecific(Vec3,VEC_COOKIE,3); 
  PetscValidHeaderSpecific(VMedian,VEC_COOKIE,4); 

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
    SETERRQ(1,"InCompatible vector local lengths");

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
