#include "taosolver.h"
#include "petscmat.h"
#include "src/matrix/lmvmmat.h"
#include "numbers.h"
PetscErrorCode viewme(Vec v); 
PetscErrorCode initializevecs(Vec**,PetscInt,PetscInt);

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc, char *argv[])
{
    PetscErrorCode ierr;
    Vec *v,y;
    PetscInt localsize,i;
    Mat lmvm_mat;

    PetscInitialize(&argc, &argv, 0, 0);
    TaoInitialize(&argc, &argv, 0, 0);
    ierr = initializevecs(&v,50,10); CHKERRQ(ierr);
    ierr = VecDuplicate(v[0], &y); CHKERRQ(ierr);
    ierr = VecGetLocalSize(v[0],&localsize); CHKERRQ(ierr);
    ierr = MatCreateLMVM(PETSC_COMM_WORLD,localsize,10,&lmvm_mat); CHKERRQ(ierr);
    for (i=0;i<4;i++) {
      ierr = MatLMVMUpdate(lmvm_mat,v[i],v[20+i]); CHKERRQ(ierr);
    }
    ierr = MatLMVMSolve(lmvm_mat,v[41],y); CHKERRQ(ierr);

    ierr = VecView(y, PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);
//    ierr = MatView(lmvm_mat,PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);
    ierr = MatDestroy(&lmvm_mat); CHKERRQ(ierr);
    
    for (i=0;i<50;i++) {
      ierr = VecDestroy(&v[i]); CHKERRQ(ierr);
    }

    ierr = VecDestroy(&y); CHKERRQ(ierr);

    TaoFinalize();
    PetscFinalize();
    return 0;
}

#undef __FUNCT__
#define __FUNCT__ "viewme"
PetscErrorCode viewme(Vec v) 
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(v,VEC_CLASSID,1);
  ierr = VecView(v,PETSC_VIEWER_STDOUT_SELF); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "initializevecs"
PetscErrorCode initializevecs(Vec **vv, PetscInt numvecs, PetscInt size) 
{
    PetscErrorCode ierr;
    PetscInt l,h;
    int i,j;
    Vec *v;
    PetscReal *x;
    Vec tmp;
    PetscFunctionBegin;
    if (numvecs*size > length) {
	SETERRQ(PETSC_COMM_SELF,1,"data set not large enough.\n");
    }

    ierr = VecCreate(PETSC_COMM_WORLD,&tmp); CHKERRQ(ierr);
    ierr = VecSetSizes(tmp,PETSC_DECIDE,size); CHKERRQ(ierr);
    ierr = VecSetFromOptions(tmp); CHKERRQ(ierr);
    ierr = VecSet(tmp,0.0); CHKERRQ(ierr);
    ierr = VecDuplicateVecs(tmp,numvecs,&v); CHKERRQ(ierr);
    for (i=0;i<numvecs;i++) {
    
	ierr = VecGetArray(v[i],&x);CHKERRQ(ierr);
	ierr = VecGetOwnershipRange(v[i],&l,&h); CHKERRQ(ierr);
	for (j=0; j<h-l; j++) {
	    x[j] = numbers[size*i+j+l];
	}
	ierr = VecRestoreArray(v[i],&x); CHKERRQ(ierr);
    }

    *vv = v;
    ierr = VecDestroy(&tmp); CHKERRQ(ierr);
    PetscFunctionReturn(0);
}



