
static char help[] = "Tests late MatSetBlockSizes.\n\n";

#include <petscmat.h>

int main(int argc,char **args)
{
  Mat                    A;
  Vec                    x[4];
  IS                     is;
  ISLocalToGlobalMapping rmap,cmap;
  PetscInt               bs[4],l2gbs[4],rbs,cbs,l2grbs,l2gcbs,i;
  PetscErrorCode         ierr;

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
  ierr = MatSetSizes(A,12,12,PETSC_DECIDE,PETSC_DECIDE);CHKERRQ(ierr);
  ierr = MatSetType(A,MATAIJ);CHKERRQ(ierr);
  ierr = ISCreateStride(PETSC_COMM_WORLD,12,0,1,&is);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingCreateIS(is,&rmap);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingSetBlockSize(rmap,2);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingCreateIS(is,&cmap);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingSetBlockSize(cmap,2);CHKERRQ(ierr);

  ierr = MatSetLocalToGlobalMapping(A,rmap,cmap);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingDestroy(&rmap);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingDestroy(&cmap);CHKERRQ(ierr);
  ierr = ISDestroy(&is);CHKERRQ(ierr);
  ierr = MatSetUp(A);CHKERRQ(ierr);

  ierr = MatCreateVecs(A,&x[1],&x[0]);CHKERRQ(ierr);
  ierr = MatSetBlockSizes(A,6,3);CHKERRQ(ierr);
  ierr = MatCreateVecs(A,&x[3],&x[2]);CHKERRQ(ierr);
  for (i=0;i<4;i++) {
    ISLocalToGlobalMapping l2g;

    ierr = VecGetBlockSize(x[i],&bs[i]);CHKERRQ(ierr);
    ierr = VecGetLocalToGlobalMapping(x[i],&l2g);CHKERRQ(ierr);
    ierr = ISLocalToGlobalMappingGetBlockSize(l2g,&l2gbs[i]);CHKERRQ(ierr);
    ierr = VecDestroy(&x[i]);CHKERRQ(ierr);
  }
  ierr = MatGetBlockSizes(A,&rbs,&cbs);CHKERRQ(ierr);
  ierr = MatGetLocalToGlobalMapping(A,&rmap,&cmap);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingGetBlockSize(rmap,&l2grbs);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingGetBlockSize(cmap,&l2gcbs);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Mat Block sizes: %D %D (l2g %D %D)\n",rbs,cbs,l2grbs,l2gcbs);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Vec Block sizes: %D %D (l2g %D %D)\n",bs[0],bs[1],l2gbs[0],l2gbs[1]);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Vec Block sizes: %D %D (l2g %D %D)\n",bs[2],bs[3],l2gbs[2],l2gbs[3]);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}



/*TEST

   test:
      nsize: 2

TEST*/
