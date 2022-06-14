
static char help[] = "Tests solving linear system with KSPFGMRES + PCSOR (omega != 1) on a matrix obtained from MatTransposeMatMult.\n\n";

#include <petscksp.h>

int main(int argc,char **args)
{
  Mat            A,Ad,B;
  PetscInt       N = 10, M = 3;
  PetscBool      no_inodes=PETSC_TRUE,flg;
  KSP            ksp;
  PC             pc;
  Vec            x,y;
  char           mtype[256];

  PetscCall(PetscInitialize(&argc,&args,(char*)0,help));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-N",&N,NULL));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-M",&M,NULL));
  PetscCall(PetscOptionsGetString(NULL,NULL,"-mtype",mtype,sizeof(mtype),&flg));
  PetscCall(PetscOptionsGetBool(NULL,NULL,"-no_inodes",&no_inodes,NULL));
  PetscCall(MatCreateDense(PETSC_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,M,N,NULL,&Ad));
  PetscCall(MatSetRandom(Ad,NULL));
  PetscCall(MatConvert(Ad,flg ? mtype : MATAIJ,MAT_INITIAL_MATRIX,&A));
  PetscCall(MatProductCreate(A,A,NULL,&B));
  PetscCall(MatProductSetType(B,MATPRODUCT_AtB));
  PetscCall(MatProductSetAlgorithm(B,"default"));
  PetscCall(MatProductSetFill(B,PETSC_DEFAULT));
  PetscCall(MatProductSetFromOptions(B));
  PetscCall(MatProductSymbolic(B));
  if (no_inodes) PetscCall(MatSetOption(B,MAT_USE_INODES,PETSC_FALSE));
  PetscCall(MatProductNumeric(B));
  PetscCall(MatTransposeMatMultEqual(A,A,B,10,&flg));
  if (!flg) {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Wrong MatTransposeMat"));
  }
  PetscCall(KSPCreate(PETSC_COMM_WORLD,&ksp));
  PetscCall(KSPSetOperators(ksp,B,B));
  PetscCall(KSPGetPC(ksp,&pc));
  PetscCall(PCSetType(pc,PCSOR));
  PetscCall(PCSORSetOmega(pc,1.1));
  PetscCall(KSPSetUp(ksp));
  PetscCall(KSPView(ksp,NULL));
  PetscCall(KSPGetPC(ksp,&pc));
  PetscCall(MatCreateVecs(B,&y,&x));
  PetscCall(VecSetRandom(x,NULL));
  PetscCall(PCApply(pc,x,y));
  PetscCall(KSPDestroy(&ksp));
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&y));
  PetscCall(MatDestroy(&B));
  PetscCall(MatDestroy(&Ad));
  PetscCall(MatDestroy(&A));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

    test:
      nsize: 1
      suffix: 1

    test:
      nsize: 1
      suffix: 1_mpiaij
      args: -mtype mpiaij

    test:
      nsize: 3
      suffix: 2

TEST*/
