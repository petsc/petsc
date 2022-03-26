
static char help[] = "Tests late MatSetBlockSizes.\n\n";

#include <petscmat.h>

int main(int argc,char **args)
{
  Mat                    A;
  Vec                    x[4];
  IS                     is;
  ISLocalToGlobalMapping rmap,cmap;
  PetscInt               bs[4],l2gbs[4],rbs,cbs,l2grbs,l2gcbs,i;

  PetscCall(PetscInitialize(&argc,&args,(char*)0,help));
  PetscCall(MatCreate(PETSC_COMM_WORLD,&A));
  PetscCall(MatSetSizes(A,12,12,PETSC_DECIDE,PETSC_DECIDE));
  PetscCall(MatSetType(A,MATAIJ));
  PetscCall(ISCreateStride(PETSC_COMM_WORLD,12,0,1,&is));
  PetscCall(ISLocalToGlobalMappingCreateIS(is,&rmap));
  PetscCall(ISLocalToGlobalMappingSetBlockSize(rmap,2));
  PetscCall(ISLocalToGlobalMappingCreateIS(is,&cmap));
  PetscCall(ISLocalToGlobalMappingSetBlockSize(cmap,2));

  PetscCall(MatSetLocalToGlobalMapping(A,rmap,cmap));
  PetscCall(ISLocalToGlobalMappingDestroy(&rmap));
  PetscCall(ISLocalToGlobalMappingDestroy(&cmap));
  PetscCall(ISDestroy(&is));
  PetscCall(MatSetUp(A));

  PetscCall(MatCreateVecs(A,&x[1],&x[0]));
  PetscCall(MatSetBlockSizes(A,6,3));
  PetscCall(MatCreateVecs(A,&x[3],&x[2]));
  for (i=0;i<4;i++) {
    ISLocalToGlobalMapping l2g;

    PetscCall(VecGetBlockSize(x[i],&bs[i]));
    PetscCall(VecGetLocalToGlobalMapping(x[i],&l2g));
    PetscCall(ISLocalToGlobalMappingGetBlockSize(l2g,&l2gbs[i]));
    PetscCall(VecDestroy(&x[i]));
  }
  PetscCall(MatGetBlockSizes(A,&rbs,&cbs));
  PetscCall(MatGetLocalToGlobalMapping(A,&rmap,&cmap));
  PetscCall(ISLocalToGlobalMappingGetBlockSize(rmap,&l2grbs));
  PetscCall(ISLocalToGlobalMappingGetBlockSize(cmap,&l2gcbs));
  PetscCall(MatDestroy(&A));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Mat Block sizes: %" PetscInt_FMT " %" PetscInt_FMT " (l2g %" PetscInt_FMT " %" PetscInt_FMT ")\n",rbs,cbs,l2grbs,l2gcbs));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Vec Block sizes: %" PetscInt_FMT " %" PetscInt_FMT " (l2g %" PetscInt_FMT " %" PetscInt_FMT ")\n",bs[0],bs[1],l2gbs[0],l2gbs[1]));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Vec Block sizes: %" PetscInt_FMT " %" PetscInt_FMT " (l2g %" PetscInt_FMT " %" PetscInt_FMT ")\n",bs[2],bs[3],l2gbs[2],l2gbs[3]));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:
      nsize: 2

TEST*/
