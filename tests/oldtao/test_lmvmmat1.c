#include "tao.h"
#include "petscmat.h"
#include "src/matrix/lmvmmat.h"
#include "src/petsctao/vector/taovec_petsc.h"
#include "../numbers.h"
PetscErrorCode initializevecs(Vec**,PetscInt,PetscInt);

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc, char *argv[])
{
    PetscErrorCode ierr;
    Vec *v;
    PetscInt localsize,i;
    TaoTruth success;
    TaoLMVMMat *lmvm_mat;
    TaoVecPetsc *tv[50];
    TaoVec *y;
    TAO_SOLVER tao;
    TAO_APPLICATION app;

    PetscInitialize(&argc, &argv, 0, 0);
    TaoInitialize(&argc, &argv, 0,0);
    /* Create TAO solver with desired solution method */
    ierr = TaoCreate(PETSC_COMM_SELF,"tao_lmvm",&tao); CHKERRQ(ierr);
    ierr = TaoApplicationCreate(PETSC_COMM_SELF,&app); CHKERRQ(ierr);
    ierr = initializevecs(&v,50,10); CHKERRQ(ierr);
    for (i=0;i<50;i++) {
	tv[i] = new TaoVecPetsc(v[i]);
    }

    ierr = tv[0]->Clone(&y); CHKERRQ(ierr);
    ierr = VecGetLocalSize(v[0],&localsize); CHKERRQ(ierr);

    lmvm_mat = new TaoLMVMMat(tv[0]); CHKERRQ(ierr);
    for (i=0;i<20;i++) {
      ierr = lmvm_mat->Update(tv[i],tv[20+i]); CHKERRQ(ierr);
    }
    ierr = lmvm_mat->Solve(tv[41],y,&success); CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Y\n");  CHKERRQ(ierr);

    ierr = y->View(); CHKERRQ(ierr);


    ierr = VecDestroyVecs(v,50); CHKERRQ(ierr);
    delete lmvm_mat;
    for (i=0;i<50;i++) {
	delete tv[i];
    }
    delete y;
//    ierr = PetscFree(&v);  CHKERRQ(ierr);
    TaoFinalize();
    PetscFinalize();
    return 0;
}


#undef __FUNCT__
#define __FUNCT__ "initalizevecs"
PetscErrorCode initializevecs(Vec **vv, PetscInt numvecs, PetscInt size) 
{
    PetscErrorCode ierr;
    PetscInt l,h;
    int i,j;
    Vec *v;
    PetscReal **xarr,*x;
    Vec tmp;
    PetscFunctionBegin;
    if (numvecs*size > length) {
	SETERRQ(PETSC_COMM_SELF,1,"data set not large enough.\n");
    }
    ierr = PetscMalloc(sizeof(Vec)*numvecs,&v); CHKERRQ(ierr);

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
    PetscFunctionReturn(0);
}



