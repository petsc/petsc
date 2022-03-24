
static char help[] = "Tests late MatSetBlockSizes.\n\n";

#include <petscmat.h>

int main(int argc,char **args)
{
  Mat                    A;
  Vec                    x[4];
  IS                     is;
  ISLocalToGlobalMapping rmap,cmap;
  PetscInt               bs[4],l2gbs[4],rbs,cbs,l2grbs,l2gcbs,i;

  CHKERRQ(PetscInitialize(&argc,&args,(char*)0,help));
  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&A));
  CHKERRQ(MatSetSizes(A,12,12,PETSC_DECIDE,PETSC_DECIDE));
  CHKERRQ(MatSetType(A,MATAIJ));
  CHKERRQ(ISCreateStride(PETSC_COMM_WORLD,12,0,1,&is));
  CHKERRQ(ISLocalToGlobalMappingCreateIS(is,&rmap));
  CHKERRQ(ISLocalToGlobalMappingSetBlockSize(rmap,2));
  CHKERRQ(ISLocalToGlobalMappingCreateIS(is,&cmap));
  CHKERRQ(ISLocalToGlobalMappingSetBlockSize(cmap,2));

  CHKERRQ(MatSetLocalToGlobalMapping(A,rmap,cmap));
  CHKERRQ(ISLocalToGlobalMappingDestroy(&rmap));
  CHKERRQ(ISLocalToGlobalMappingDestroy(&cmap));
  CHKERRQ(ISDestroy(&is));
  CHKERRQ(MatSetUp(A));

  CHKERRQ(MatCreateVecs(A,&x[1],&x[0]));
  CHKERRQ(MatSetBlockSizes(A,6,3));
  CHKERRQ(MatCreateVecs(A,&x[3],&x[2]));
  for (i=0;i<4;i++) {
    ISLocalToGlobalMapping l2g;

    CHKERRQ(VecGetBlockSize(x[i],&bs[i]));
    CHKERRQ(VecGetLocalToGlobalMapping(x[i],&l2g));
    CHKERRQ(ISLocalToGlobalMappingGetBlockSize(l2g,&l2gbs[i]));
    CHKERRQ(VecDestroy(&x[i]));
  }
  CHKERRQ(MatGetBlockSizes(A,&rbs,&cbs));
  CHKERRQ(MatGetLocalToGlobalMapping(A,&rmap,&cmap));
  CHKERRQ(ISLocalToGlobalMappingGetBlockSize(rmap,&l2grbs));
  CHKERRQ(ISLocalToGlobalMappingGetBlockSize(cmap,&l2gcbs));
  CHKERRQ(MatDestroy(&A));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Mat Block sizes: %" PetscInt_FMT " %" PetscInt_FMT " (l2g %" PetscInt_FMT " %" PetscInt_FMT ")\n",rbs,cbs,l2grbs,l2gcbs));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Vec Block sizes: %" PetscInt_FMT " %" PetscInt_FMT " (l2g %" PetscInt_FMT " %" PetscInt_FMT ")\n",bs[0],bs[1],l2gbs[0],l2gbs[1]));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Vec Block sizes: %" PetscInt_FMT " %" PetscInt_FMT " (l2g %" PetscInt_FMT " %" PetscInt_FMT ")\n",bs[2],bs[3],l2gbs[2],l2gbs[3]));
  CHKERRQ(PetscFinalize());
  return 0;
}

/*TEST

   test:
      nsize: 2

TEST*/
