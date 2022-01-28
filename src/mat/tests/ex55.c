
static char help[] = "Tests converting a matrix to another format with MatConvert().\n\n";

#include <petscmat.h>
/* Usage: mpiexec -n <np> ex55 -verbose <0 or 1> */

int main(int argc,char **args)
{
  Mat            C,A,B,D;
  PetscErrorCode ierr;
  PetscInt       i,j,ntypes,bs,mbs,m,block,d_nz=6, o_nz=3,col[3],row,verbose=0;
  PetscMPIInt    size,rank;
  MatType        type[9];
  char           file[PETSC_MAX_PATH_LEN];
  PetscViewer    fd;
  PetscBool      equal,flg_loadmat,flg,issymmetric;
  PetscScalar    value[3];

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  ierr = PetscOptionsGetInt(NULL,NULL,"-verbose",&verbose,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetString(NULL,NULL,"-f",file,sizeof(file),&flg_loadmat);CHKERRQ(ierr);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRMPI(ierr);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRMPI(ierr);

  ierr = PetscOptionsHasName(NULL,NULL,"-testseqaij",&flg);CHKERRQ(ierr);
  if (flg) {
    if (size == 1) {
      type[0] = MATSEQAIJ;
    } else {
      type[0] = MATMPIAIJ;
    }
  } else {
    type[0] = MATAIJ;
  }
  if (size == 1) {
    ntypes  = 3;
    type[1] = MATSEQBAIJ;
    type[2] = MATSEQSBAIJ;
  } else {
    ntypes  = 3;
    type[1] = MATMPIBAIJ;
    type[2] = MATMPISBAIJ;
  }

  /* input matrix C */
  if (flg_loadmat) {
    /* Open binary file. */
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,file,FILE_MODE_READ,&fd);CHKERRQ(ierr);

    /* Load a baij matrix, then destroy the viewer. */
    ierr = MatCreate(PETSC_COMM_WORLD,&C);CHKERRQ(ierr);
    if (size == 1) {
      ierr = MatSetType(C,MATSEQBAIJ);CHKERRQ(ierr);
    } else {
      ierr = MatSetType(C,MATMPIBAIJ);CHKERRQ(ierr);
    }
    ierr = MatSetFromOptions(C);CHKERRQ(ierr);
    ierr = MatLoad(C,fd);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&fd);CHKERRQ(ierr);
  } else { /* Create a baij mat with bs>1  */
    bs   = 2; mbs=8;
    ierr = PetscOptionsGetInt(NULL,NULL,"-mbs",&mbs,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsGetInt(NULL,NULL,"-bs",&bs,NULL);CHKERRQ(ierr);
    PetscAssertFalse(bs <= 1,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG," bs must be >1 in this case");
    m    = mbs*bs;
    ierr = MatCreateBAIJ(PETSC_COMM_WORLD,bs,PETSC_DECIDE,PETSC_DECIDE,m,m,d_nz,NULL,o_nz,NULL,&C);CHKERRQ(ierr);
    for (block=0; block<mbs; block++) {
      /* diagonal blocks */
      value[0] = -1.0; value[1] = 4.0; value[2] = -1.0;
      for (i=1+block*bs; i<bs-1+block*bs; i++) {
        col[0] = i-1; col[1] = i; col[2] = i+1;
        ierr   = MatSetValues(C,1,&i,3,col,value,INSERT_VALUES);CHKERRQ(ierr);
      }
      i       = bs - 1+block*bs; col[0] = bs - 2+block*bs; col[1] = bs - 1+block*bs;
      value[0]=-1.0; value[1]=4.0;
      ierr    = MatSetValues(C,1,&i,2,col,value,INSERT_VALUES);CHKERRQ(ierr);

      i       = 0+block*bs; col[0] = 0+block*bs; col[1] = 1+block*bs;
      value[0]=4.0; value[1] = -1.0;
      ierr    = MatSetValues(C,1,&i,2,col,value,INSERT_VALUES);CHKERRQ(ierr);
    }
    /* off-diagonal blocks */
    value[0]=-1.0;
    for (i=0; i<(mbs-1)*bs; i++) {
      col[0]=i+bs;
      ierr  = MatSetValues(C,1,&i,1,col,value,INSERT_VALUES);CHKERRQ(ierr);
      col[0]=i; row=i+bs;
      ierr  = MatSetValues(C,1,&row,1,col,value,INSERT_VALUES);CHKERRQ(ierr);
    }
    ierr = MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }
  {
    /* Check the symmetry of C because it will be converted to a sbaij matrix */
    Mat Ctrans;
    ierr = MatTranspose(C,MAT_INITIAL_MATRIX,&Ctrans);CHKERRQ(ierr);
    ierr = MatEqual(C,Ctrans,&flg);CHKERRQ(ierr);
/*
    {
      ierr = MatAXPY(C,1.,Ctrans,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
      flg  = PETSC_TRUE;
    }
*/
    ierr = MatSetOption(C,MAT_SYMMETRIC,flg);CHKERRQ(ierr);
    ierr = MatDestroy(&Ctrans);CHKERRQ(ierr);
  }
  ierr = MatIsSymmetric(C,0.0,&issymmetric);CHKERRQ(ierr);
  ierr = MatViewFromOptions(C,NULL,"-view_mat");CHKERRQ(ierr);

  /* convert C to other formats */
  for (i=0; i<ntypes; i++) {
    PetscBool ismpisbaij,isseqsbaij;

    ierr = PetscStrcmp(type[i],MATMPISBAIJ,&ismpisbaij);CHKERRQ(ierr);
    ierr = PetscStrcmp(type[i],MATMPISBAIJ,&isseqsbaij);CHKERRQ(ierr);
    if (!issymmetric && (ismpisbaij || isseqsbaij)) continue;
    ierr = MatConvert(C,type[i],MAT_INITIAL_MATRIX,&A);CHKERRQ(ierr);
    ierr = MatMultEqual(A,C,10,&equal);CHKERRQ(ierr);
    PetscAssertFalse(!equal,PETSC_COMM_SELF,PETSC_ERR_ARG_NOTSAMETYPE,"Error in conversion from BAIJ to %s",type[i]);
    for (j=i+1; j<ntypes; j++) {
      ierr = PetscStrcmp(type[j],MATMPISBAIJ,&ismpisbaij);CHKERRQ(ierr);
      ierr = PetscStrcmp(type[j],MATMPISBAIJ,&isseqsbaij);CHKERRQ(ierr);
      if (!issymmetric && (ismpisbaij || isseqsbaij)) continue;
      if (verbose>0) {
        ierr = PetscPrintf(PETSC_COMM_WORLD," \n[%d] test conversion between %s and %s\n",rank,type[i],type[j]);CHKERRQ(ierr);
      }

      if (rank == 0 && verbose) printf("Convert %s A to %s B\n",type[i],type[j]);
      ierr = MatConvert(A,type[j],MAT_INITIAL_MATRIX,&B);CHKERRQ(ierr);
      /*
      if (j == 2) {
        ierr = PetscPrintf(PETSC_COMM_SELF," A: %s\n",type[i]);CHKERRQ(ierr);
        ierr = MatView(A,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
        ierr = PetscPrintf(PETSC_COMM_SELF," B: %s\n",type[j]);CHKERRQ(ierr);
        ierr = MatView(B,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
      }
       */
      ierr = MatMultEqual(A,B,10,&equal);CHKERRQ(ierr);
      PetscAssertFalse(!equal,PETSC_COMM_SELF,PETSC_ERR_ARG_NOTSAMETYPE,"Error in conversion from %s to %s",type[i],type[j]);

      if (size == 1 || j != 2) { /* Matconvert from mpisbaij mat to other formats are not supported */
        if (rank == 0 && verbose) printf("Convert %s B to %s D\n",type[j],type[i]);
        ierr = MatConvert(B,type[i],MAT_INITIAL_MATRIX,&D);CHKERRQ(ierr);
        ierr = MatMultEqual(B,D,10,&equal);CHKERRQ(ierr);
        PetscAssertFalse(!equal,PETSC_COMM_SELF,PETSC_ERR_ARG_NOTSAMETYPE,"Error in conversion from %s to %s",type[j],type[i]);

        ierr = MatDestroy(&D);CHKERRQ(ierr);
      }
      ierr = MatDestroy(&B);CHKERRQ(ierr);
      ierr = MatDestroy(&D);CHKERRQ(ierr);
    }

    /* Test in-place convert */
    if (size == 1) { /* size > 1 is not working yet! */
      j = (i+1)%ntypes;
      ierr = PetscStrcmp(type[j],MATMPISBAIJ,&ismpisbaij);CHKERRQ(ierr);
      ierr = PetscStrcmp(type[j],MATMPISBAIJ,&isseqsbaij);CHKERRQ(ierr);
      if (!issymmetric && (ismpisbaij || isseqsbaij)) continue;
      /* printf("[%d] i: %d, j: %d\n",rank,i,j); */
      ierr = MatConvert(A,type[j],MAT_INPLACE_MATRIX,&A);CHKERRQ(ierr);
    }

    ierr = MatDestroy(&A);CHKERRQ(ierr);
  }

  /* test BAIJ to MATIS */
  if (size > 1) {
    MatType ctype;

    ierr = MatGetType(C,&ctype);CHKERRQ(ierr);
    ierr = MatConvert(C,MATIS,MAT_INITIAL_MATRIX,&A);CHKERRQ(ierr);
    ierr = MatMultEqual(A,C,10,&equal);CHKERRQ(ierr);
    ierr = MatViewFromOptions(A,NULL,"-view_conv");CHKERRQ(ierr);
    if (!equal) {
      Mat C2;

      ierr = MatConvert(A,ctype,MAT_INITIAL_MATRIX,&C2);CHKERRQ(ierr);
      ierr = MatViewFromOptions(C2,NULL,"-view_conv_assembled");CHKERRQ(ierr);
      ierr = MatAXPY(C2,-1.,C,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
      ierr = MatChop(C2,PETSC_SMALL);CHKERRQ(ierr);
      ierr = MatViewFromOptions(C2,NULL,"-view_err");CHKERRQ(ierr);
      ierr = MatDestroy(&C2);CHKERRQ(ierr);
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Error in conversion from BAIJ to MATIS");
    }
    ierr = MatConvert(C,MATIS,MAT_REUSE_MATRIX,&A);CHKERRQ(ierr);
    ierr = MatMultEqual(A,C,10,&equal);CHKERRQ(ierr);
    ierr = MatViewFromOptions(A,NULL,"-view_conv");CHKERRQ(ierr);
    if (!equal) {
      Mat C2;

      ierr = MatConvert(A,ctype,MAT_INITIAL_MATRIX,&C2);CHKERRQ(ierr);
      ierr = MatViewFromOptions(C2,NULL,"-view_conv_assembled");CHKERRQ(ierr);
      ierr = MatAXPY(C2,-1.,C,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
      ierr = MatChop(C2,PETSC_SMALL);CHKERRQ(ierr);
      ierr = MatViewFromOptions(C2,NULL,"-view_err");CHKERRQ(ierr);
      ierr = MatDestroy(&C2);CHKERRQ(ierr);
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Error in conversion from BAIJ to MATIS");
    }
    ierr = MatDestroy(&A);CHKERRQ(ierr);
    ierr = MatDuplicate(C,MAT_COPY_VALUES,&A);CHKERRQ(ierr);
    ierr = MatConvert(A,MATIS,MAT_INPLACE_MATRIX,&A);CHKERRQ(ierr);
    ierr = MatViewFromOptions(A,NULL,"-view_conv");CHKERRQ(ierr);
    ierr = MatMultEqual(A,C,10,&equal);CHKERRQ(ierr);
    if (!equal) {
      Mat C2;

      ierr = MatViewFromOptions(A,NULL,"-view_conv");CHKERRQ(ierr);
      ierr = MatConvert(A,ctype,MAT_INITIAL_MATRIX,&C2);CHKERRQ(ierr);
      ierr = MatViewFromOptions(C2,NULL,"-view_conv_assembled");CHKERRQ(ierr);
      ierr = MatAXPY(C2,-1.,C,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
      ierr = MatChop(C2,PETSC_SMALL);CHKERRQ(ierr);
      ierr = MatViewFromOptions(C2,NULL,"-view_err");CHKERRQ(ierr);
      ierr = MatDestroy(&C2);CHKERRQ(ierr);
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Error in conversion from BAIJ to MATIS");
    }
    ierr = MatDestroy(&A);CHKERRQ(ierr);
  }
  ierr = MatDestroy(&C);CHKERRQ(ierr);

  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:

   test:
      suffix: 2
      nsize: 3

   testset:
      requires: parmetis
      output_file: output/ex55_1.out
      nsize: 3
      args: -mat_is_disassemble_l2g_type nd -mat_partitioning_type parmetis
      test:
        suffix: matis_baij_parmetis_nd
      test:
        suffix: matis_aij_parmetis_nd
        args: -testseqaij
      test:
        requires: datafilespath !complex double !defined(PETSC_USE_64BIT_INDICES)
        suffix: matis_poisson1_parmetis_nd
        args: -f ${DATAFILESPATH}/matrices/poisson1

   testset:
      requires: ptscotch defined(PETSC_HAVE_SCOTCH_PARMETIS_V3_NODEND)
      output_file: output/ex55_1.out
      nsize: 4
      args: -mat_is_disassemble_l2g_type nd -mat_partitioning_type ptscotch
      test:
        suffix: matis_baij_ptscotch_nd
      test:
        suffix: matis_aij_ptscotch_nd
        args: -testseqaij
      test:
        requires: datafilespath !complex double !defined(PETSC_USE_64BIT_INDICES)
        suffix: matis_poisson1_ptscotch_nd
        args: -f ${DATAFILESPATH}/matrices/poisson1

TEST*/
