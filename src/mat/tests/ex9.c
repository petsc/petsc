
static char help[] = "Tests MPI parallel matrix creation. Test MatCreateRedundantMatrix() \n\n";

#include <petscmat.h>

int main(int argc,char **args)
{
  Mat            C,Credundant;
  MatInfo        info;
  PetscMPIInt    rank,size,subsize;
  PetscInt       i,j,m = 3,n = 2,low,high,iglobal;
  PetscInt       Ii,J,ldim,nsubcomms;
  PetscErrorCode ierr;
  PetscBool      flg_info,flg_mat;
  PetscScalar    v,one = 1.0;
  Vec            x,y;
  MPI_Comm       subcomm;

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  ierr = PetscOptionsGetInt(NULL,NULL,"-m",&m,NULL);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRMPI(ierr);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRMPI(ierr);
  n    = 2*size;

  ierr = MatCreate(PETSC_COMM_WORLD,&C);CHKERRQ(ierr);
  ierr = MatSetSizes(C,PETSC_DECIDE,PETSC_DECIDE,m*n,m*n);CHKERRQ(ierr);
  ierr = MatSetFromOptions(C);CHKERRQ(ierr);
  ierr = MatSetUp(C);CHKERRQ(ierr);

  /* Create the matrix for the five point stencil, YET AGAIN */
  for (i=0; i<m; i++) {
    for (j=2*rank; j<2*rank+2; j++) {
      v = -1.0;  Ii = j + n*i;
      if (i>0)   {J = Ii - n; ierr = MatSetValues(C,1,&Ii,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);}
      if (i<m-1) {J = Ii + n; ierr = MatSetValues(C,1,&Ii,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);}
      if (j>0)   {J = Ii - 1; ierr = MatSetValues(C,1,&Ii,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);}
      if (j<n-1) {J = Ii + 1; ierr = MatSetValues(C,1,&Ii,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);}
      v = 4.0; ierr = MatSetValues(C,1,&Ii,1,&Ii,&v,INSERT_VALUES);CHKERRQ(ierr);
    }
  }

  /* Add extra elements (to illustrate variants of MatGetInfo) */
  Ii   = n; J = n-2; v = 100.0;
  ierr = MatSetValues(C,1,&Ii,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);
  Ii   = n-2; J = n; v = 100.0;
  ierr = MatSetValues(C,1,&Ii,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);

  ierr = MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  /* Form vectors */
  ierr = MatCreateVecs(C,&x,&y);CHKERRQ(ierr);
  ierr = VecGetLocalSize(x,&ldim);CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(x,&low,&high);CHKERRQ(ierr);
  for (i=0; i<ldim; i++) {
    iglobal = i + low;
    v       = one*((PetscReal)i) + 100.0*rank;
    ierr    = VecSetValues(x,1,&iglobal,&v,INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = VecAssemblyBegin(x);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(x);CHKERRQ(ierr);

  ierr = MatMult(C,x,y);CHKERRQ(ierr);

  ierr = PetscOptionsHasName(NULL,NULL,"-view_info",&flg_info);CHKERRQ(ierr);
  if (flg_info)  {
    ierr = PetscViewerPushFormat(PETSC_VIEWER_STDOUT_WORLD,PETSC_VIEWER_ASCII_INFO);CHKERRQ(ierr);
    ierr = MatView(C,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

    ierr = MatGetInfo(C,MAT_GLOBAL_SUM,&info);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_WORLD,"matrix information (global sums):\nnonzeros = %D, allocated nonzeros = %D\n",(PetscInt)info.nz_used,(PetscInt)info.nz_allocated);CHKERRQ(ierr);
    ierr = MatGetInfo (C,MAT_GLOBAL_MAX,&info);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_WORLD,"matrix information (global max):\nnonzeros = %D, allocated nonzeros = %D\n",(PetscInt)info.nz_used,(PetscInt)info.nz_allocated);CHKERRQ(ierr);
  }

  ierr = PetscOptionsHasName(NULL,NULL,"-view_mat",&flg_mat);CHKERRQ(ierr);
  if (flg_mat) {
    ierr = MatView(C,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  }

  /* Test MatCreateRedundantMatrix() */
  nsubcomms = size;
  ierr = PetscOptionsGetInt(NULL,NULL,"-nsubcomms",&nsubcomms,NULL);CHKERRQ(ierr);
  ierr = MatCreateRedundantMatrix(C,nsubcomms,MPI_COMM_NULL,MAT_INITIAL_MATRIX,&Credundant);CHKERRQ(ierr);
  ierr = MatCreateRedundantMatrix(C,nsubcomms,MPI_COMM_NULL,MAT_REUSE_MATRIX,&Credundant);CHKERRQ(ierr);

  ierr = PetscObjectGetComm((PetscObject)Credundant,&subcomm);CHKERRQ(ierr);
  ierr = MPI_Comm_size(subcomm,&subsize);CHKERRMPI(ierr);

  if (subsize==2 && flg_mat) {
    ierr = PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_(subcomm),"\n[%d] Credundant:\n",rank);CHKERRQ(ierr);
    ierr = MatView(Credundant,PETSC_VIEWER_STDOUT_(subcomm));CHKERRQ(ierr);
  }
  ierr = MatDestroy(&Credundant);CHKERRQ(ierr);

  /* Test MatCreateRedundantMatrix() with user-provided subcomm */
  {
    PetscSubcomm psubcomm;

    ierr = PetscSubcommCreate(PETSC_COMM_WORLD,&psubcomm);CHKERRQ(ierr);
    ierr = PetscSubcommSetNumber(psubcomm,nsubcomms);CHKERRQ(ierr);
    ierr = PetscSubcommSetType(psubcomm,PETSC_SUBCOMM_CONTIGUOUS);CHKERRQ(ierr);
    /* enable runtime switch of psubcomm type, e.g., '-psubcomm_type interlaced */
    ierr = PetscSubcommSetFromOptions(psubcomm);CHKERRQ(ierr);

    ierr = MatCreateRedundantMatrix(C,nsubcomms,PetscSubcommChild(psubcomm),MAT_INITIAL_MATRIX,&Credundant);CHKERRQ(ierr);
    ierr = MatCreateRedundantMatrix(C,nsubcomms,PetscSubcommChild(psubcomm),MAT_REUSE_MATRIX,&Credundant);CHKERRQ(ierr);

    ierr = PetscSubcommDestroy(&psubcomm);CHKERRQ(ierr);
    ierr = MatDestroy(&Credundant);CHKERRQ(ierr);
  }

  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&y);CHKERRQ(ierr);
  ierr = MatDestroy(&C);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
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
