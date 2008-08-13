#include "tao.h"
#include "petscmat.h"
#include "src/matrix/lmvmmat.h"
int main(int argc, char *argv[])
{
    PetscErrorCode info;
    Mat lmvm_mat;

    PetscInitialize(&argc, &argv, 0, 0);
    info = MatCreateLMVM(PETSC_COMM_SELF,10,10,&lmvm_mat); CHKERRQ(info);
    info = MatView(lmvm_mat,PETSC_VIEWER_STDOUT_SELF); CHKERRQ(info);
    info = MatDestroy(lmvm_mat); CHKERRQ(info);
    PetscFinalize();
    return 0;
}
