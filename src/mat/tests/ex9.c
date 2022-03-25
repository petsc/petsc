
static char help[] = "Tests MPI parallel matrix creation. Test MatCreateRedundantMatrix() \n\n";

#include <petscmat.h>

int main(int argc,char **args)
{
  Mat            C,Credundant;
  MatInfo        info;
  PetscMPIInt    rank,size,subsize;
  PetscInt       i,j,m = 3,n = 2,low,high,iglobal;
  PetscInt       Ii,J,ldim,nsubcomms;
  PetscBool      flg_info,flg_mat;
  PetscScalar    v,one = 1.0;
  Vec            x,y;
  MPI_Comm       subcomm;

  PetscCall(PetscInitialize(&argc,&args,(char*)0,help));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-m",&m,NULL));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  n    = 2*size;

  PetscCall(MatCreate(PETSC_COMM_WORLD,&C));
  PetscCall(MatSetSizes(C,PETSC_DECIDE,PETSC_DECIDE,m*n,m*n));
  PetscCall(MatSetFromOptions(C));
  PetscCall(MatSetUp(C));

  /* Create the matrix for the five point stencil, YET AGAIN */
  for (i=0; i<m; i++) {
    for (j=2*rank; j<2*rank+2; j++) {
      v = -1.0;  Ii = j + n*i;
      if (i>0)   {J = Ii - n; PetscCall(MatSetValues(C,1,&Ii,1,&J,&v,INSERT_VALUES));}
      if (i<m-1) {J = Ii + n; PetscCall(MatSetValues(C,1,&Ii,1,&J,&v,INSERT_VALUES));}
      if (j>0)   {J = Ii - 1; PetscCall(MatSetValues(C,1,&Ii,1,&J,&v,INSERT_VALUES));}
      if (j<n-1) {J = Ii + 1; PetscCall(MatSetValues(C,1,&Ii,1,&J,&v,INSERT_VALUES));}
      v = 4.0; PetscCall(MatSetValues(C,1,&Ii,1,&Ii,&v,INSERT_VALUES));
    }
  }

  /* Add extra elements (to illustrate variants of MatGetInfo) */
  Ii   = n; J = n-2; v = 100.0;
  PetscCall(MatSetValues(C,1,&Ii,1,&J,&v,INSERT_VALUES));
  Ii   = n-2; J = n; v = 100.0;
  PetscCall(MatSetValues(C,1,&Ii,1,&J,&v,INSERT_VALUES));

  PetscCall(MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY));

  /* Form vectors */
  PetscCall(MatCreateVecs(C,&x,&y));
  PetscCall(VecGetLocalSize(x,&ldim));
  PetscCall(VecGetOwnershipRange(x,&low,&high));
  for (i=0; i<ldim; i++) {
    iglobal = i + low;
    v       = one*((PetscReal)i) + 100.0*rank;
    PetscCall(VecSetValues(x,1,&iglobal,&v,INSERT_VALUES));
  }
  PetscCall(VecAssemblyBegin(x));
  PetscCall(VecAssemblyEnd(x));

  PetscCall(MatMult(C,x,y));

  PetscCall(PetscOptionsHasName(NULL,NULL,"-view_info",&flg_info));
  if (flg_info)  {
    PetscCall(PetscViewerPushFormat(PETSC_VIEWER_STDOUT_WORLD,PETSC_VIEWER_ASCII_INFO));
    PetscCall(MatView(C,PETSC_VIEWER_STDOUT_WORLD));

    PetscCall(MatGetInfo(C,MAT_GLOBAL_SUM,&info));
    PetscCall(PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_WORLD,"matrix information (global sums):\nnonzeros = %" PetscInt_FMT ", allocated nonzeros = %" PetscInt_FMT "\n",(PetscInt)info.nz_used,(PetscInt)info.nz_allocated));
    PetscCall(MatGetInfo (C,MAT_GLOBAL_MAX,&info));
    PetscCall(PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_WORLD,"matrix information (global max):\nnonzeros = %" PetscInt_FMT ", allocated nonzeros = %" PetscInt_FMT "\n",(PetscInt)info.nz_used,(PetscInt)info.nz_allocated));
  }

  PetscCall(PetscOptionsHasName(NULL,NULL,"-view_mat",&flg_mat));
  if (flg_mat) {
    PetscCall(MatView(C,PETSC_VIEWER_STDOUT_WORLD));
  }

  /* Test MatCreateRedundantMatrix() */
  nsubcomms = size;
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-nsubcomms",&nsubcomms,NULL));
  PetscCall(MatCreateRedundantMatrix(C,nsubcomms,MPI_COMM_NULL,MAT_INITIAL_MATRIX,&Credundant));
  PetscCall(MatCreateRedundantMatrix(C,nsubcomms,MPI_COMM_NULL,MAT_REUSE_MATRIX,&Credundant));

  PetscCall(PetscObjectGetComm((PetscObject)Credundant,&subcomm));
  PetscCallMPI(MPI_Comm_size(subcomm,&subsize));

  if (subsize==2 && flg_mat) {
    PetscCall(PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_(subcomm),"\n[%d] Credundant:\n",rank));
    PetscCall(MatView(Credundant,PETSC_VIEWER_STDOUT_(subcomm)));
  }
  PetscCall(MatDestroy(&Credundant));

  /* Test MatCreateRedundantMatrix() with user-provided subcomm */
  {
    PetscSubcomm psubcomm;

    PetscCall(PetscSubcommCreate(PETSC_COMM_WORLD,&psubcomm));
    PetscCall(PetscSubcommSetNumber(psubcomm,nsubcomms));
    PetscCall(PetscSubcommSetType(psubcomm,PETSC_SUBCOMM_CONTIGUOUS));
    /* enable runtime switch of psubcomm type, e.g., '-psubcomm_type interlaced */
    PetscCall(PetscSubcommSetFromOptions(psubcomm));

    PetscCall(MatCreateRedundantMatrix(C,nsubcomms,PetscSubcommChild(psubcomm),MAT_INITIAL_MATRIX,&Credundant));
    PetscCall(MatCreateRedundantMatrix(C,nsubcomms,PetscSubcommChild(psubcomm),MAT_REUSE_MATRIX,&Credundant));

    PetscCall(PetscSubcommDestroy(&psubcomm));
    PetscCall(MatDestroy(&Credundant));
  }

  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&y));
  PetscCall(MatDestroy(&C));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:
      nsize: 3
      args: -view_info

   test:
      suffix: 2
      nsize: 3
      args: -nsubcomms 2 -view_mat -psubcomm_type interlaced

   test:
      suffix: 3
      nsize: 3
      args: -nsubcomms 2 -view_mat -psubcomm_type contiguous

   test:
      suffix: 3_baij
      nsize: 3
      args: -mat_type baij -nsubcomms 2 -view_mat

   test:
      suffix: 3_sbaij
      nsize: 3
      args: -mat_type sbaij -nsubcomms 2 -view_mat

   test:
      suffix: 3_dense
      nsize: 3
      args: -mat_type dense -nsubcomms 2 -view_mat

   test:
      suffix: 4_baij
      nsize: 3
      args: -mat_type baij -nsubcomms 2 -view_mat -psubcomm_type interlaced

   test:
      suffix: 4_sbaij
      nsize: 3
      args: -mat_type sbaij -nsubcomms 2 -view_mat -psubcomm_type interlaced

   test:
      suffix: 4_dense
      nsize: 3
      args: -mat_type dense -nsubcomms 2 -view_mat -psubcomm_type interlaced

TEST*/
