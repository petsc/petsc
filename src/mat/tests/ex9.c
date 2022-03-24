
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

  CHKERRQ(PetscInitialize(&argc,&args,(char*)0,help));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-m",&m,NULL));
  CHKERRMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  CHKERRMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  n    = 2*size;

  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&C));
  CHKERRQ(MatSetSizes(C,PETSC_DECIDE,PETSC_DECIDE,m*n,m*n));
  CHKERRQ(MatSetFromOptions(C));
  CHKERRQ(MatSetUp(C));

  /* Create the matrix for the five point stencil, YET AGAIN */
  for (i=0; i<m; i++) {
    for (j=2*rank; j<2*rank+2; j++) {
      v = -1.0;  Ii = j + n*i;
      if (i>0)   {J = Ii - n; CHKERRQ(MatSetValues(C,1,&Ii,1,&J,&v,INSERT_VALUES));}
      if (i<m-1) {J = Ii + n; CHKERRQ(MatSetValues(C,1,&Ii,1,&J,&v,INSERT_VALUES));}
      if (j>0)   {J = Ii - 1; CHKERRQ(MatSetValues(C,1,&Ii,1,&J,&v,INSERT_VALUES));}
      if (j<n-1) {J = Ii + 1; CHKERRQ(MatSetValues(C,1,&Ii,1,&J,&v,INSERT_VALUES));}
      v = 4.0; CHKERRQ(MatSetValues(C,1,&Ii,1,&Ii,&v,INSERT_VALUES));
    }
  }

  /* Add extra elements (to illustrate variants of MatGetInfo) */
  Ii   = n; J = n-2; v = 100.0;
  CHKERRQ(MatSetValues(C,1,&Ii,1,&J,&v,INSERT_VALUES));
  Ii   = n-2; J = n; v = 100.0;
  CHKERRQ(MatSetValues(C,1,&Ii,1,&J,&v,INSERT_VALUES));

  CHKERRQ(MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY));

  /* Form vectors */
  CHKERRQ(MatCreateVecs(C,&x,&y));
  CHKERRQ(VecGetLocalSize(x,&ldim));
  CHKERRQ(VecGetOwnershipRange(x,&low,&high));
  for (i=0; i<ldim; i++) {
    iglobal = i + low;
    v       = one*((PetscReal)i) + 100.0*rank;
    CHKERRQ(VecSetValues(x,1,&iglobal,&v,INSERT_VALUES));
  }
  CHKERRQ(VecAssemblyBegin(x));
  CHKERRQ(VecAssemblyEnd(x));

  CHKERRQ(MatMult(C,x,y));

  CHKERRQ(PetscOptionsHasName(NULL,NULL,"-view_info",&flg_info));
  if (flg_info)  {
    CHKERRQ(PetscViewerPushFormat(PETSC_VIEWER_STDOUT_WORLD,PETSC_VIEWER_ASCII_INFO));
    CHKERRQ(MatView(C,PETSC_VIEWER_STDOUT_WORLD));

    CHKERRQ(MatGetInfo(C,MAT_GLOBAL_SUM,&info));
    CHKERRQ(PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_WORLD,"matrix information (global sums):\nnonzeros = %" PetscInt_FMT ", allocated nonzeros = %" PetscInt_FMT "\n",(PetscInt)info.nz_used,(PetscInt)info.nz_allocated));
    CHKERRQ(MatGetInfo (C,MAT_GLOBAL_MAX,&info));
    CHKERRQ(PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_WORLD,"matrix information (global max):\nnonzeros = %" PetscInt_FMT ", allocated nonzeros = %" PetscInt_FMT "\n",(PetscInt)info.nz_used,(PetscInt)info.nz_allocated));
  }

  CHKERRQ(PetscOptionsHasName(NULL,NULL,"-view_mat",&flg_mat));
  if (flg_mat) {
    CHKERRQ(MatView(C,PETSC_VIEWER_STDOUT_WORLD));
  }

  /* Test MatCreateRedundantMatrix() */
  nsubcomms = size;
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-nsubcomms",&nsubcomms,NULL));
  CHKERRQ(MatCreateRedundantMatrix(C,nsubcomms,MPI_COMM_NULL,MAT_INITIAL_MATRIX,&Credundant));
  CHKERRQ(MatCreateRedundantMatrix(C,nsubcomms,MPI_COMM_NULL,MAT_REUSE_MATRIX,&Credundant));

  CHKERRQ(PetscObjectGetComm((PetscObject)Credundant,&subcomm));
  CHKERRMPI(MPI_Comm_size(subcomm,&subsize));

  if (subsize==2 && flg_mat) {
    CHKERRQ(PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_(subcomm),"\n[%d] Credundant:\n",rank));
    CHKERRQ(MatView(Credundant,PETSC_VIEWER_STDOUT_(subcomm)));
  }
  CHKERRQ(MatDestroy(&Credundant));

  /* Test MatCreateRedundantMatrix() with user-provided subcomm */
  {
    PetscSubcomm psubcomm;

    CHKERRQ(PetscSubcommCreate(PETSC_COMM_WORLD,&psubcomm));
    CHKERRQ(PetscSubcommSetNumber(psubcomm,nsubcomms));
    CHKERRQ(PetscSubcommSetType(psubcomm,PETSC_SUBCOMM_CONTIGUOUS));
    /* enable runtime switch of psubcomm type, e.g., '-psubcomm_type interlaced */
    CHKERRQ(PetscSubcommSetFromOptions(psubcomm));

    CHKERRQ(MatCreateRedundantMatrix(C,nsubcomms,PetscSubcommChild(psubcomm),MAT_INITIAL_MATRIX,&Credundant));
    CHKERRQ(MatCreateRedundantMatrix(C,nsubcomms,PetscSubcommChild(psubcomm),MAT_REUSE_MATRIX,&Credundant));

    CHKERRQ(PetscSubcommDestroy(&psubcomm));
    CHKERRQ(MatDestroy(&Credundant));
  }

  CHKERRQ(VecDestroy(&x));
  CHKERRQ(VecDestroy(&y));
  CHKERRQ(MatDestroy(&C));
  CHKERRQ(PetscFinalize());
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
