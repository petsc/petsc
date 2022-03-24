
static char help[] = "Tests the use of interface functions for MATIS matrices.\n\
This example tests: MatZeroRows(), MatZeroRowsLocal(), MatView(), MatDuplicate(),\n\
MatCopy(), MatCreateSubMatrix(), MatGetLocalSubMatrix(), MatAXPY(), MatShift()\n\
MatDiagonalSet(), MatTranspose() and MatPtAP(). It also tests some\n\
conversion routines.\n";

#include <petscmat.h>

PetscErrorCode TestMatZeroRows(Mat,Mat,PetscBool,IS,PetscScalar);
PetscErrorCode CheckMat(Mat,Mat,PetscBool,const char*);

int main(int argc,char **args)
{
  Mat                    A,B,A2,B2,T;
  Mat                    Aee,Aeo,Aoe,Aoo;
  Mat                    *mats;
  Vec                    x,y;
  MatInfo                info;
  ISLocalToGlobalMapping cmap,rmap;
  IS                     is,is2,reven,rodd,ceven,codd;
  IS                     *rows,*cols;
  MatType                lmtype;
  PetscScalar            diag = 2.;
  PetscInt               n,m,i,lm,ln;
  PetscInt               rst,ren,cst,cen,nr,nc;
  PetscMPIInt            rank,size;
  PetscBool              testT,squaretest,isaij;
  PetscBool              permute = PETSC_FALSE, negmap = PETSC_FALSE, repmap = PETSC_FALSE;
  PetscBool              diffmap = PETSC_TRUE, symmetric = PETSC_FALSE, issymmetric;
  PetscErrorCode         ierr;

  CHKERRQ(PetscInitialize(&argc,&args,(char*)0,help));
  CHKERRMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  CHKERRMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  m = n = 2*size;
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-symmetric",&symmetric,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-m",&m,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-negmap",&negmap,NULL));
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-repmap",&repmap,NULL));
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-permmap",&permute,NULL));
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-diffmap",&diffmap,NULL));
  PetscCheckFalse(size > 1 && m < 4,PETSC_COMM_WORLD,PETSC_ERR_ARG_WRONG,"Number of rows should be larger or equal 4 for parallel runs");
  PetscCheckFalse(size == 1 && m < 2,PETSC_COMM_WORLD,PETSC_ERR_ARG_WRONG,"Number of rows should be larger or equal 2 for uniprocessor runs");
  PetscCheckFalse(n < 2,PETSC_COMM_WORLD,PETSC_ERR_ARG_WRONG,"Number of cols should be larger or equal 2");

  /* create a MATIS matrix */
  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&A));
  CHKERRQ(MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,m,n));
  CHKERRQ(MatSetType(A,MATIS));
  CHKERRQ(MatSetFromOptions(A));
  if (!negmap && !repmap) {
    /* This is not the proper setting for MATIS for finite elements, it is just used to test the routines
       Here we use a one-to-one correspondence between local row/column spaces and global row/column spaces
       Equivalent to passing NULL for the mapping */
    CHKERRQ(ISCreateStride(PETSC_COMM_WORLD,n,0,1,&is));
  } else if (negmap && !repmap) { /* non repeated but with negative indices */
    CHKERRQ(ISCreateStride(PETSC_COMM_WORLD,n+2,-2,1,&is));
  } else if (!negmap && repmap) { /* non negative but repeated indices */
    IS isl[2];

    CHKERRQ(ISCreateStride(PETSC_COMM_WORLD,n,0,1,&isl[0]));
    CHKERRQ(ISCreateStride(PETSC_COMM_WORLD,n,n-1,-1,&isl[1]));
    CHKERRQ(ISConcatenate(PETSC_COMM_WORLD,2,isl,&is));
    CHKERRQ(ISDestroy(&isl[0]));
    CHKERRQ(ISDestroy(&isl[1]));
  } else { /* negative and repeated indices */
    IS isl[2];

    CHKERRQ(ISCreateStride(PETSC_COMM_WORLD,n+1,-1,1,&isl[0]));
    CHKERRQ(ISCreateStride(PETSC_COMM_WORLD,n+1,n-1,-1,&isl[1]));
    CHKERRQ(ISConcatenate(PETSC_COMM_WORLD,2,isl,&is));
    CHKERRQ(ISDestroy(&isl[0]));
    CHKERRQ(ISDestroy(&isl[1]));
  }
  CHKERRQ(ISLocalToGlobalMappingCreateIS(is,&cmap));
  CHKERRQ(ISDestroy(&is));

  if (m != n || diffmap) {
    CHKERRQ(ISCreateStride(PETSC_COMM_WORLD,m,permute ? m-1 : 0,permute ? -1 : 1,&is));
    CHKERRQ(ISLocalToGlobalMappingCreateIS(is,&rmap));
    CHKERRQ(ISDestroy(&is));
  } else {
    CHKERRQ(PetscObjectReference((PetscObject)cmap));
    rmap = cmap;
  }

  CHKERRQ(MatSetLocalToGlobalMapping(A,rmap,cmap));
  CHKERRQ(MatISStoreL2L(A,PETSC_FALSE));
  CHKERRQ(MatISSetPreallocation(A,3,NULL,3,NULL));
  CHKERRQ(MatSetOption(A,MAT_NEW_NONZERO_ALLOCATION_ERR,(PetscBool)!(repmap || negmap))); /* I do not want to precompute the pattern */
  CHKERRQ(ISLocalToGlobalMappingGetSize(rmap,&lm));
  CHKERRQ(ISLocalToGlobalMappingGetSize(cmap,&ln));
  for (i=0; i<lm; i++) {
    PetscScalar v[3];
    PetscInt    cols[3];

    cols[0] = (i-1+n)%n;
    cols[1] = i%n;
    cols[2] = (i+1)%n;
    v[0] = -1.*(symmetric ? PetscMin(i+1,cols[0]+1) : i+1);
    v[1] =  2.*(symmetric ? PetscMin(i+1,cols[1]+1) : i+1);
    v[2] = -1.*(symmetric ? PetscMin(i+1,cols[2]+1) : i+1);
    CHKERRQ(ISGlobalToLocalMappingApply(cmap,IS_GTOLM_MASK,3,cols,NULL,cols));
    CHKERRQ(MatSetValuesLocal(A,1,&i,3,cols,v,ADD_VALUES));
  }
  CHKERRQ(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));

  /* activate tests for square matrices with same maps only */
  CHKERRQ(MatHasCongruentLayouts(A,&squaretest));
  if (squaretest && rmap != cmap) {
    PetscInt nr, nc;

    CHKERRQ(ISLocalToGlobalMappingGetSize(rmap,&nr));
    CHKERRQ(ISLocalToGlobalMappingGetSize(cmap,&nc));
    if (nr != nc) squaretest = PETSC_FALSE;
    else {
      const PetscInt *idxs1,*idxs2;

      CHKERRQ(ISLocalToGlobalMappingGetIndices(rmap,&idxs1));
      CHKERRQ(ISLocalToGlobalMappingGetIndices(cmap,&idxs2));
      CHKERRQ(PetscArraycmp(idxs1,idxs2,nr,&squaretest));
      CHKERRQ(ISLocalToGlobalMappingRestoreIndices(rmap,&idxs1));
      CHKERRQ(ISLocalToGlobalMappingRestoreIndices(cmap,&idxs2));
    }
    CHKERRMPI(MPIU_Allreduce(MPI_IN_PLACE,&squaretest,1,MPIU_BOOL,MPI_LAND,PetscObjectComm((PetscObject)A)));
  }

  /* test MatISGetLocalMat */
  CHKERRQ(MatISGetLocalMat(A,&B));
  CHKERRQ(MatGetType(B,&lmtype));

  /* test MatGetInfo */
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Test MatGetInfo\n"));
  CHKERRQ(MatGetInfo(A,MAT_LOCAL,&info));
  CHKERRQ(PetscViewerASCIIPushSynchronized(PETSC_VIEWER_STDOUT_WORLD));
  ierr = PetscViewerASCIISynchronizedPrintf(PETSC_VIEWER_STDOUT_WORLD,"Process  %2d: %" PetscInt_FMT " %" PetscInt_FMT " %" PetscInt_FMT " %" PetscInt_FMT " %" PetscInt_FMT "\n",PetscGlobalRank,(PetscInt)info.nz_used,
                                            (PetscInt)info.nz_allocated,(PetscInt)info.nz_unneeded,(PetscInt)info.assemblies,(PetscInt)info.mallocs);CHKERRQ(ierr);
  CHKERRQ(PetscViewerFlush(PETSC_VIEWER_STDOUT_WORLD));
  CHKERRQ(MatGetInfo(A,MAT_GLOBAL_MAX,&info));
  ierr = PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_WORLD,"GlobalMax  : %" PetscInt_FMT " %" PetscInt_FMT " %" PetscInt_FMT " %" PetscInt_FMT " %" PetscInt_FMT "\n",(PetscInt)info.nz_used,
                                (PetscInt)info.nz_allocated,(PetscInt)info.nz_unneeded,(PetscInt)info.assemblies,(PetscInt)info.mallocs);CHKERRQ(ierr);
  CHKERRQ(MatGetInfo(A,MAT_GLOBAL_SUM,&info));
  ierr = PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_WORLD,"GlobalSum  : %" PetscInt_FMT " %" PetscInt_FMT " %" PetscInt_FMT " %" PetscInt_FMT " %" PetscInt_FMT "\n",(PetscInt)info.nz_used,
                                (PetscInt)info.nz_allocated,(PetscInt)info.nz_unneeded,(PetscInt)info.assemblies,(PetscInt)info.mallocs);CHKERRQ(ierr);

  /* test MatIsSymmetric */
  CHKERRQ(MatIsSymmetric(A,0.0,&issymmetric));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Test MatIsSymmetric: %d\n",issymmetric));

  /* Create a MPIAIJ matrix, same as A */
  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&B));
  CHKERRQ(MatSetSizes(B,PETSC_DECIDE,PETSC_DECIDE,m,n));
  CHKERRQ(MatSetType(B,MATAIJ));
  CHKERRQ(MatSetFromOptions(B));
  CHKERRQ(MatSetUp(B));
  CHKERRQ(MatSetLocalToGlobalMapping(B,rmap,cmap));
  CHKERRQ(MatMPIAIJSetPreallocation(B,3,NULL,3,NULL));
  CHKERRQ(MatMPIBAIJSetPreallocation(B,1,3,NULL,3,NULL));
#if defined(PETSC_HAVE_HYPRE)
  CHKERRQ(MatHYPRESetPreallocation(B,3,NULL,3,NULL));
#endif
  CHKERRQ(MatISSetPreallocation(B,3,NULL,3,NULL));
  CHKERRQ(MatSetOption(B,MAT_NEW_NONZERO_ALLOCATION_ERR,(PetscBool)!(repmap || negmap))); /* I do not want to precompute the pattern */
  for (i=0; i<lm; i++) {
    PetscScalar v[3];
    PetscInt    cols[3];

    cols[0] = (i-1+n)%n;
    cols[1] = i%n;
    cols[2] = (i+1)%n;
    v[0] = -1.*(symmetric ? PetscMin(i+1,cols[0]+1) : i+1);
    v[1] =  2.*(symmetric ? PetscMin(i+1,cols[1]+1) : i+1);
    v[2] = -1.*(symmetric ? PetscMin(i+1,cols[2]+1) : i+1);
    CHKERRQ(ISGlobalToLocalMappingApply(cmap,IS_GTOLM_MASK,3,cols,NULL,cols));
    CHKERRQ(MatSetValuesLocal(B,1,&i,3,cols,v,ADD_VALUES));
  }
  CHKERRQ(MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY));

  /* test MatView */
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Test MatView\n"));
  CHKERRQ(MatView(A,NULL));
  CHKERRQ(MatView(B,NULL));

  /* test CheckMat */
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Test CheckMat\n"));
  CHKERRQ(CheckMat(A,B,PETSC_FALSE,"CheckMat"));

  /* test MatDuplicate and MatAXPY */
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Test MatDuplicate and MatAXPY\n"));
  CHKERRQ(MatDuplicate(A,MAT_COPY_VALUES,&A2));
  CHKERRQ(CheckMat(A,A2,PETSC_FALSE,"MatDuplicate and MatAXPY"));

  /* test MatConvert */
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Test MatConvert_IS_XAIJ\n"));
  CHKERRQ(MatConvert(A2,MATAIJ,MAT_INITIAL_MATRIX,&B2));
  CHKERRQ(CheckMat(B,B2,PETSC_TRUE,"MatConvert_IS_XAIJ MAT_INITIAL_MATRIX"));
  CHKERRQ(MatConvert(A2,MATAIJ,MAT_REUSE_MATRIX,&B2));
  CHKERRQ(CheckMat(B,B2,PETSC_TRUE,"MatConvert_IS_XAIJ MAT_REUSE_MATRIX"));
  CHKERRQ(MatConvert(A2,MATAIJ,MAT_INPLACE_MATRIX,&A2));
  CHKERRQ(CheckMat(B,A2,PETSC_TRUE,"MatConvert_IS_XAIJ MAT_INPLACE_MATRIX"));
  CHKERRQ(MatDestroy(&A2));
  CHKERRQ(MatDestroy(&B2));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Test MatConvert_XAIJ_IS\n"));
  CHKERRQ(MatDuplicate(B,MAT_COPY_VALUES,&B2));
  CHKERRQ(MatConvert(B2,MATIS,MAT_INITIAL_MATRIX,&A2));
  CHKERRQ(CheckMat(A,A2,PETSC_TRUE,"MatConvert_XAIJ_IS MAT_INITIAL_MATRIX"));
  CHKERRQ(MatConvert(B2,MATIS,MAT_REUSE_MATRIX,&A2));
  CHKERRQ(CheckMat(A,A2,PETSC_TRUE,"MatConvert_XAIJ_IS MAT_REUSE_MATRIX"));
  CHKERRQ(MatConvert(B2,MATIS,MAT_INPLACE_MATRIX,&B2));
  CHKERRQ(CheckMat(A,B2,PETSC_TRUE,"MatConvert_XAIJ_IS MAT_INPLACE_MATRIX"));
  CHKERRQ(MatDestroy(&A2));
  CHKERRQ(MatDestroy(&B2));
  CHKERRQ(PetscStrcmp(lmtype,MATSEQAIJ,&isaij));
  if (size == 1 && isaij) { /* tests special code paths in MatConvert_IS_XAIJ */
    PetscInt               ri, ci, rr[3] = {0,1,0}, cr[4] = {1,2,0,1}, rk[3] = {0,2,1}, ck[4] = {1,0,3,2};
    ISLocalToGlobalMapping tcmap,trmap;

    for (ri = 0; ri < 2; ri++) {
      PetscInt *r;

      r = (PetscInt*)(ri == 0 ? rr : rk);
      for (ci = 0; ci < 2; ci++) {
        PetscInt *c,rb,cb;

        c = (PetscInt*)(ci == 0 ? cr : ck);
        for (rb = 1; rb < 4; rb++) {
          CHKERRQ(ISCreateBlock(PETSC_COMM_SELF,rb,3,r,PETSC_COPY_VALUES,&is));
          CHKERRQ(ISLocalToGlobalMappingCreateIS(is,&trmap));
          CHKERRQ(ISDestroy(&is));
          for (cb = 1; cb < 4; cb++) {
            Mat  T,lT,T2;
            char testname[256];

            CHKERRQ(PetscSNPrintf(testname,sizeof(testname),"MatConvert_IS_XAIJ special case (%" PetscInt_FMT " %" PetscInt_FMT ", bs %" PetscInt_FMT " %" PetscInt_FMT ")",ri,ci,rb,cb));
            CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Test %s\n",testname));

            CHKERRQ(ISCreateBlock(PETSC_COMM_SELF,cb,4,c,PETSC_COPY_VALUES,&is));
            CHKERRQ(ISLocalToGlobalMappingCreateIS(is,&tcmap));
            CHKERRQ(ISDestroy(&is));

            CHKERRQ(MatCreate(PETSC_COMM_SELF,&T));
            CHKERRQ(MatSetSizes(T,PETSC_DECIDE,PETSC_DECIDE,rb*3,cb*4));
            CHKERRQ(MatSetType(T,MATIS));
            CHKERRQ(MatSetLocalToGlobalMapping(T,trmap,tcmap));
            CHKERRQ(ISLocalToGlobalMappingDestroy(&tcmap));
            CHKERRQ(MatISGetLocalMat(T,&lT));
            CHKERRQ(MatSetType(lT,MATSEQAIJ));
            CHKERRQ(MatSeqAIJSetPreallocation(lT,cb*4,NULL));
            CHKERRQ(MatSetRandom(lT,NULL));
            CHKERRQ(MatConvert(lT,lmtype,MAT_INPLACE_MATRIX,&lT));
            CHKERRQ(MatISRestoreLocalMat(T,&lT));
            CHKERRQ(MatAssemblyBegin(T,MAT_FINAL_ASSEMBLY));
            CHKERRQ(MatAssemblyEnd(T,MAT_FINAL_ASSEMBLY));

            CHKERRQ(MatConvert(T,MATAIJ,MAT_INITIAL_MATRIX,&T2));
            CHKERRQ(CheckMat(T,T2,PETSC_TRUE,"MAT_INITIAL_MATRIX"));
            CHKERRQ(MatConvert(T,MATAIJ,MAT_REUSE_MATRIX,&T2));
            CHKERRQ(CheckMat(T,T2,PETSC_TRUE,"MAT_REUSE_MATRIX"));
            CHKERRQ(MatDestroy(&T2));
            CHKERRQ(MatDuplicate(T,MAT_COPY_VALUES,&T2));
            CHKERRQ(MatConvert(T2,MATAIJ,MAT_INPLACE_MATRIX,&T2));
            CHKERRQ(CheckMat(T,T2,PETSC_TRUE,"MAT_INPLACE_MATRIX"));
            CHKERRQ(MatDestroy(&T));
            CHKERRQ(MatDestroy(&T2));
          }
          CHKERRQ(ISLocalToGlobalMappingDestroy(&trmap));
        }
      }
    }
  }

  /* test MatDiagonalScale */
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Test MatDiagonalScale\n"));
  CHKERRQ(MatDuplicate(A,MAT_COPY_VALUES,&A2));
  CHKERRQ(MatDuplicate(B,MAT_COPY_VALUES,&B2));
  CHKERRQ(MatCreateVecs(A,&x,&y));
  CHKERRQ(VecSetRandom(x,NULL));
  if (issymmetric) {
    CHKERRQ(VecCopy(x,y));
  } else {
    CHKERRQ(VecSetRandom(y,NULL));
    CHKERRQ(VecScale(y,8.));
  }
  CHKERRQ(MatDiagonalScale(A2,y,x));
  CHKERRQ(MatDiagonalScale(B2,y,x));
  CHKERRQ(CheckMat(A2,B2,PETSC_FALSE,"MatDiagonalScale"));
  CHKERRQ(MatDestroy(&A2));
  CHKERRQ(MatDestroy(&B2));
  CHKERRQ(VecDestroy(&x));
  CHKERRQ(VecDestroy(&y));

  /* test MatPtAP (A IS and B AIJ) */
  if (isaij && m == n) {
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Test MatPtAP\n"));
    CHKERRQ(MatISStoreL2L(A,PETSC_TRUE));
    CHKERRQ(MatPtAP(A,B,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&A2));
    CHKERRQ(MatPtAP(B,B,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&B2));
    CHKERRQ(CheckMat(A2,B2,PETSC_FALSE,"MatPtAP MAT_INITIAL_MATRIX"));
    CHKERRQ(MatPtAP(A,B,MAT_REUSE_MATRIX,PETSC_DEFAULT,&A2));
    CHKERRQ(CheckMat(A2,B2,PETSC_FALSE,"MatPtAP MAT_REUSE_MATRIX"));
    CHKERRQ(MatDestroy(&A2));
    CHKERRQ(MatDestroy(&B2));
  }

  /* test MatGetLocalSubMatrix */
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Test MatGetLocalSubMatrix\n"));
  CHKERRQ(MatDuplicate(A,MAT_DO_NOT_COPY_VALUES,&A2));
  CHKERRQ(ISCreateStride(PETSC_COMM_SELF,lm/2+lm%2,0,2,&reven));
  CHKERRQ(ISComplement(reven,0,lm,&rodd));
  CHKERRQ(ISCreateStride(PETSC_COMM_SELF,ln/2+ln%2,0,2,&ceven));
  CHKERRQ(ISComplement(ceven,0,ln,&codd));
  CHKERRQ(MatGetLocalSubMatrix(A2,reven,ceven,&Aee));
  CHKERRQ(MatGetLocalSubMatrix(A2,reven,codd,&Aeo));
  CHKERRQ(MatGetLocalSubMatrix(A2,rodd,ceven,&Aoe));
  CHKERRQ(MatGetLocalSubMatrix(A2,rodd,codd,&Aoo));
  for (i=0; i<lm; i++) {
    PetscInt    j,je,jo,colse[3], colso[3];
    PetscScalar ve[3], vo[3];
    PetscScalar v[3];
    PetscInt    cols[3];
    PetscInt    row = i/2;

    cols[0] = (i-1+n)%n;
    cols[1] = i%n;
    cols[2] = (i+1)%n;
    v[0] = -1.*(symmetric ? PetscMin(i+1,cols[0]+1) : i+1);
    v[1] =  2.*(symmetric ? PetscMin(i+1,cols[1]+1) : i+1);
    v[2] = -1.*(symmetric ? PetscMin(i+1,cols[2]+1) : i+1);
    CHKERRQ(ISGlobalToLocalMappingApply(cmap,IS_GTOLM_MASK,3,cols,NULL,cols));
    for (j=0,je=0,jo=0;j<3;j++) {
      if (cols[j]%2) {
        vo[jo] = v[j];
        colso[jo++] = cols[j]/2;
      } else {
        ve[je] = v[j];
        colse[je++] = cols[j]/2;
      }
    }
    if (i%2) {
      CHKERRQ(MatSetValuesLocal(Aoe,1,&row,je,colse,ve,ADD_VALUES));
      CHKERRQ(MatSetValuesLocal(Aoo,1,&row,jo,colso,vo,ADD_VALUES));
    } else {
      CHKERRQ(MatSetValuesLocal(Aee,1,&row,je,colse,ve,ADD_VALUES));
      CHKERRQ(MatSetValuesLocal(Aeo,1,&row,jo,colso,vo,ADD_VALUES));
    }
  }
  CHKERRQ(MatRestoreLocalSubMatrix(A2,reven,ceven,&Aee));
  CHKERRQ(MatRestoreLocalSubMatrix(A2,reven,codd,&Aeo));
  CHKERRQ(MatRestoreLocalSubMatrix(A2,rodd,ceven,&Aoe));
  CHKERRQ(MatRestoreLocalSubMatrix(A2,rodd,codd,&Aoo));
  CHKERRQ(ISDestroy(&reven));
  CHKERRQ(ISDestroy(&ceven));
  CHKERRQ(ISDestroy(&rodd));
  CHKERRQ(ISDestroy(&codd));
  CHKERRQ(MatAssemblyBegin(A2,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(A2,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAXPY(A2,-1.,A,SAME_NONZERO_PATTERN));
  CHKERRQ(CheckMat(A2,NULL,PETSC_FALSE,"MatGetLocalSubMatrix"));
  CHKERRQ(MatDestroy(&A2));

  /* test MatConvert_Nest_IS */
  testT = PETSC_FALSE;
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-test_trans",&testT,NULL));

  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Test MatConvert_Nest_IS\n"));
  nr   = 2;
  nc   = 2;
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-nr",&nr,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-nc",&nc,NULL));
  if (testT) {
    CHKERRQ(MatGetOwnershipRange(A,&cst,&cen));
    CHKERRQ(MatGetOwnershipRangeColumn(A,&rst,&ren));
  } else {
    CHKERRQ(MatGetOwnershipRange(A,&rst,&ren));
    CHKERRQ(MatGetOwnershipRangeColumn(A,&cst,&cen));
  }
  CHKERRQ(PetscMalloc3(nr,&rows,nc,&cols,2*nr*nc,&mats));
  for (i=0;i<nr*nc;i++) {
    if (testT) {
      CHKERRQ(MatCreateTranspose(A,&mats[i]));
      CHKERRQ(MatTranspose(B,MAT_INITIAL_MATRIX,&mats[i+nr*nc]));
    } else {
      CHKERRQ(MatDuplicate(A,MAT_COPY_VALUES,&mats[i]));
      CHKERRQ(MatDuplicate(B,MAT_COPY_VALUES,&mats[i+nr*nc]));
    }
  }
  for (i=0;i<nr;i++) {
    CHKERRQ(ISCreateStride(PETSC_COMM_WORLD,ren-rst,i+rst,nr,&rows[i]));
  }
  for (i=0;i<nc;i++) {
    CHKERRQ(ISCreateStride(PETSC_COMM_WORLD,cen-cst,i+cst,nc,&cols[i]));
  }
  CHKERRQ(MatCreateNest(PETSC_COMM_WORLD,nr,rows,nc,cols,mats,&A2));
  CHKERRQ(MatCreateNest(PETSC_COMM_WORLD,nr,rows,nc,cols,mats+nr*nc,&B2));
  for (i=0;i<nr;i++) {
    CHKERRQ(ISDestroy(&rows[i]));
  }
  for (i=0;i<nc;i++) {
    CHKERRQ(ISDestroy(&cols[i]));
  }
  for (i=0;i<2*nr*nc;i++) {
    CHKERRQ(MatDestroy(&mats[i]));
  }
  CHKERRQ(PetscFree3(rows,cols,mats));
  CHKERRQ(MatConvert(B2,MATAIJ,MAT_INITIAL_MATRIX,&T));
  CHKERRQ(MatDestroy(&B2));
  CHKERRQ(MatConvert(A2,MATIS,MAT_INITIAL_MATRIX,&B2));
  CHKERRQ(CheckMat(B2,T,PETSC_TRUE,"MatConvert_Nest_IS MAT_INITIAL_MATRIX"));
  CHKERRQ(MatConvert(A2,MATIS,MAT_REUSE_MATRIX,&B2));
  CHKERRQ(CheckMat(B2,T,PETSC_TRUE,"MatConvert_Nest_IS MAT_REUSE_MATRIX"));
  CHKERRQ(MatDestroy(&B2));
  CHKERRQ(MatConvert(A2,MATIS,MAT_INPLACE_MATRIX,&A2));
  CHKERRQ(CheckMat(A2,T,PETSC_TRUE,"MatConvert_Nest_IS MAT_INPLACE_MATRIX"));
  CHKERRQ(MatDestroy(&T));
  CHKERRQ(MatDestroy(&A2));

  /* test MatCreateSubMatrix */
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Test MatCreateSubMatrix\n"));
  if (rank == 0) {
    CHKERRQ(ISCreateStride(PETSC_COMM_WORLD,1,1,1,&is));
    CHKERRQ(ISCreateStride(PETSC_COMM_WORLD,2,0,1,&is2));
  } else if (rank == 1) {
    CHKERRQ(ISCreateStride(PETSC_COMM_WORLD,1,0,1,&is));
    if (n > 3) {
      CHKERRQ(ISCreateStride(PETSC_COMM_WORLD,1,3,1,&is2));
    } else {
      CHKERRQ(ISCreateStride(PETSC_COMM_WORLD,0,0,1,&is2));
    }
  } else if (rank == 2 && n > 4) {
    CHKERRQ(ISCreateStride(PETSC_COMM_WORLD,0,0,1,&is));
    CHKERRQ(ISCreateStride(PETSC_COMM_WORLD,n-4,4,1,&is2));
  } else {
    CHKERRQ(ISCreateStride(PETSC_COMM_WORLD,0,0,1,&is));
    CHKERRQ(ISCreateStride(PETSC_COMM_WORLD,0,0,1,&is2));
  }
  CHKERRQ(MatCreateSubMatrix(A,is,is,MAT_INITIAL_MATRIX,&A2));
  CHKERRQ(MatCreateSubMatrix(B,is,is,MAT_INITIAL_MATRIX,&B2));
  CHKERRQ(CheckMat(A2,B2,PETSC_TRUE,"first MatCreateSubMatrix"));

  CHKERRQ(MatCreateSubMatrix(A,is,is,MAT_REUSE_MATRIX,&A2));
  CHKERRQ(MatCreateSubMatrix(B,is,is,MAT_REUSE_MATRIX,&B2));
  CHKERRQ(CheckMat(A2,B2,PETSC_FALSE,"reuse MatCreateSubMatrix"));
  CHKERRQ(MatDestroy(&A2));
  CHKERRQ(MatDestroy(&B2));

  if (!issymmetric) {
    CHKERRQ(MatCreateSubMatrix(A,is,is2,MAT_INITIAL_MATRIX,&A2));
    CHKERRQ(MatCreateSubMatrix(B,is,is2,MAT_INITIAL_MATRIX,&B2));
    CHKERRQ(MatCreateSubMatrix(A,is,is2,MAT_REUSE_MATRIX,&A2));
    CHKERRQ(MatCreateSubMatrix(B,is,is2,MAT_REUSE_MATRIX,&B2));
    CHKERRQ(CheckMat(A2,B2,PETSC_FALSE,"second MatCreateSubMatrix"));
  }

  CHKERRQ(MatDestroy(&A2));
  CHKERRQ(MatDestroy(&B2));
  CHKERRQ(ISDestroy(&is));
  CHKERRQ(ISDestroy(&is2));

  /* Create an IS required by MatZeroRows(): just rank zero provides the rows to be eliminated */
  if (size > 1) {
    if (rank == 0) {
      PetscInt st,len;

      st   = (m+1)/2;
      len  = PetscMin(m/2,PetscMax(m-(m+1)/2-1,0));
      CHKERRQ(ISCreateStride(PETSC_COMM_WORLD,len,st,1,&is));
    } else {
      CHKERRQ(ISCreateStride(PETSC_COMM_WORLD,0,0,1,&is));
    }
  } else {
    CHKERRQ(ISCreateStride(PETSC_COMM_WORLD,1,0,1,&is));
  }

  if (squaretest) { /* tests for square matrices only, with same maps for rows and columns */
    /* test MatDiagonalSet */
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Test MatDiagonalSet\n"));
    CHKERRQ(MatDuplicate(A,MAT_COPY_VALUES,&A2));
    CHKERRQ(MatDuplicate(B,MAT_COPY_VALUES,&B2));
    CHKERRQ(MatCreateVecs(A,NULL,&x));
    CHKERRQ(VecSetRandom(x,NULL));
    CHKERRQ(MatDiagonalSet(A2,x,INSERT_VALUES));
    CHKERRQ(MatDiagonalSet(B2,x,INSERT_VALUES));
    CHKERRQ(CheckMat(A2,B2,PETSC_FALSE,"MatDiagonalSet"));
    CHKERRQ(VecDestroy(&x));
    CHKERRQ(MatDestroy(&A2));
    CHKERRQ(MatDestroy(&B2));

    /* test MatShift (MatShift_IS internally uses MatDiagonalSet_IS with ADD_VALUES) */
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Test MatShift\n"));
    CHKERRQ(MatDuplicate(A,MAT_COPY_VALUES,&A2));
    CHKERRQ(MatDuplicate(B,MAT_COPY_VALUES,&B2));
    CHKERRQ(MatShift(A2,2.0));
    CHKERRQ(MatShift(B2,2.0));
    CHKERRQ(CheckMat(A2,B2,PETSC_FALSE,"MatShift"));
    CHKERRQ(MatDestroy(&A2));
    CHKERRQ(MatDestroy(&B2));

    /* nonzero diag value is supported for square matrices only */
    CHKERRQ(TestMatZeroRows(A,B,PETSC_TRUE,is,diag));
  }
  CHKERRQ(TestMatZeroRows(A,B,squaretest,is,0.0));
  CHKERRQ(ISDestroy(&is));

  /* test MatTranspose */
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Test MatTranspose\n"));
  CHKERRQ(MatTranspose(A,MAT_INITIAL_MATRIX,&A2));
  CHKERRQ(MatTranspose(B,MAT_INITIAL_MATRIX,&B2));
  CHKERRQ(CheckMat(A2,B2,PETSC_FALSE,"initial matrix MatTranspose"));

  CHKERRQ(MatTranspose(A,MAT_REUSE_MATRIX,&A2));
  CHKERRQ(CheckMat(A2,B2,PETSC_FALSE,"reuse matrix (not in place) MatTranspose"));
  CHKERRQ(MatDestroy(&A2));

  CHKERRQ(MatDuplicate(A,MAT_COPY_VALUES,&A2));
  CHKERRQ(MatTranspose(A2,MAT_INPLACE_MATRIX,&A2));
  CHKERRQ(CheckMat(A2,B2,PETSC_FALSE,"reuse matrix (in place) MatTranspose"));
  CHKERRQ(MatDestroy(&A2));

  CHKERRQ(MatTranspose(A,MAT_INITIAL_MATRIX,&A2));
  CHKERRQ(CheckMat(A2,B2,PETSC_FALSE,"reuse matrix (different type) MatTranspose"));
  CHKERRQ(MatDestroy(&A2));
  CHKERRQ(MatDestroy(&B2));

  /* test MatISFixLocalEmpty */
  if (isaij) {
    PetscInt r[2];

    r[0] = 0;
    r[1] = PetscMin(m,n)-1;
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Test MatISFixLocalEmpty\n"));
    CHKERRQ(MatDuplicate(A,MAT_COPY_VALUES,&A2));

    CHKERRQ(MatISFixLocalEmpty(A2,PETSC_TRUE));
    CHKERRQ(MatAssemblyBegin(A2,MAT_FINAL_ASSEMBLY));
    CHKERRQ(MatAssemblyEnd(A2,MAT_FINAL_ASSEMBLY));
    CHKERRQ(CheckMat(A2,B,PETSC_FALSE,"MatISFixLocalEmpty (null)"));

    CHKERRQ(MatZeroRows(A2,2,r,0.0,NULL,NULL));
    CHKERRQ(MatViewFromOptions(A2,NULL,"-fixempty_view"));
    CHKERRQ(MatDuplicate(B,MAT_COPY_VALUES,&B2));
    CHKERRQ(MatZeroRows(B2,2,r,0.0,NULL,NULL));
    CHKERRQ(MatISFixLocalEmpty(A2,PETSC_TRUE));
    CHKERRQ(MatAssemblyBegin(A2,MAT_FINAL_ASSEMBLY));
    CHKERRQ(MatAssemblyEnd(A2,MAT_FINAL_ASSEMBLY));
    CHKERRQ(MatViewFromOptions(A2,NULL,"-fixempty_view"));
    CHKERRQ(CheckMat(A2,B2,PETSC_FALSE,"MatISFixLocalEmpty (rows)"));
    CHKERRQ(MatDestroy(&A2));

    CHKERRQ(MatDuplicate(A,MAT_COPY_VALUES,&A2));
    CHKERRQ(MatZeroRows(A2,2,r,0.0,NULL,NULL));
    CHKERRQ(MatTranspose(A2,MAT_INPLACE_MATRIX,&A2));
    CHKERRQ(MatTranspose(B2,MAT_INPLACE_MATRIX,&B2));
    CHKERRQ(MatViewFromOptions(A2,NULL,"-fixempty_view"));
    CHKERRQ(MatISFixLocalEmpty(A2,PETSC_TRUE));
    CHKERRQ(MatAssemblyBegin(A2,MAT_FINAL_ASSEMBLY));
    CHKERRQ(MatAssemblyEnd(A2,MAT_FINAL_ASSEMBLY));
    CHKERRQ(MatViewFromOptions(A2,NULL,"-fixempty_view"));
    CHKERRQ(CheckMat(A2,B2,PETSC_FALSE,"MatISFixLocalEmpty (cols)"));

    CHKERRQ(MatDestroy(&A2));
    CHKERRQ(MatDestroy(&B2));

    if (squaretest) {
      CHKERRQ(MatDuplicate(A,MAT_COPY_VALUES,&A2));
      CHKERRQ(MatDuplicate(B,MAT_COPY_VALUES,&B2));
      CHKERRQ(MatZeroRowsColumns(A2,2,r,0.0,NULL,NULL));
      CHKERRQ(MatZeroRowsColumns(B2,2,r,0.0,NULL,NULL));
      CHKERRQ(MatViewFromOptions(A2,NULL,"-fixempty_view"));
      CHKERRQ(MatISFixLocalEmpty(A2,PETSC_TRUE));
      CHKERRQ(MatAssemblyBegin(A2,MAT_FINAL_ASSEMBLY));
      CHKERRQ(MatAssemblyEnd(A2,MAT_FINAL_ASSEMBLY));
      CHKERRQ(MatViewFromOptions(A2,NULL,"-fixempty_view"));
      CHKERRQ(CheckMat(A2,B2,PETSC_FALSE,"MatISFixLocalEmpty (rows+cols)"));
      CHKERRQ(MatDestroy(&A2));
      CHKERRQ(MatDestroy(&B2));
    }

  }

  /* test MatInvertBlockDiagonal
       special cases for block-diagonal matrices */
  if (m == n) {
    ISLocalToGlobalMapping map;
    Mat                    Abd,Bbd;
    IS                     is,bis;
    const PetscScalar      *isbd,*aijbd;
    PetscScalar            *vals;
    const PetscInt         *sts,*idxs;
    PetscInt               *idxs2,diff,perm,nl,bs,st,en,in;
    PetscBool              ok;

    for (diff = 0; diff < 3; diff++) {
      for (perm = 0; perm < 3; perm++) {
        for (bs = 1; bs < 4; bs++) {
          CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Test MatInvertBlockDiagonal blockdiag %" PetscInt_FMT " %" PetscInt_FMT " %" PetscInt_FMT " %" PetscInt_FMT "\n",n,diff,perm,bs));
          CHKERRQ(PetscMalloc1(bs*bs,&vals));
          CHKERRQ(MatGetOwnershipRanges(A,&sts));
          switch (diff) {
          case 1: /* inverted layout by processes */
            in = 1;
            st = sts[size - rank - 1];
            en = sts[size - rank];
            nl = en - st;
            break;
          case 2: /* round-robin layout */
            in = size;
            st = rank;
            nl = n/size;
            if (rank < n%size) nl++;
            break;
          default: /* same layout */
            in = 1;
            st = sts[rank];
            en = sts[rank + 1];
            nl = en - st;
            break;
          }
          CHKERRQ(ISCreateStride(PETSC_COMM_WORLD,nl,st,in,&is));
          CHKERRQ(ISGetLocalSize(is,&nl));
          CHKERRQ(ISGetIndices(is,&idxs));
          CHKERRQ(PetscMalloc1(nl,&idxs2));
          for (i=0;i<nl;i++) {
            switch (perm) { /* invert some of the indices */
            case 2:
              idxs2[i] = rank%2 ? idxs[i] : idxs[nl-i-1];
              break;
            case 1:
              idxs2[i] = rank%2 ? idxs[nl-i-1] : idxs[i];
              break;
            default:
              idxs2[i] = idxs[i];
              break;
            }
          }
          CHKERRQ(ISRestoreIndices(is,&idxs));
          CHKERRQ(ISCreateBlock(PETSC_COMM_WORLD,bs,nl,idxs2,PETSC_OWN_POINTER,&bis));
          CHKERRQ(ISLocalToGlobalMappingCreateIS(bis,&map));
          CHKERRQ(MatCreateIS(PETSC_COMM_WORLD,bs,PETSC_DECIDE,PETSC_DECIDE,bs*n,bs*n,map,map,&Abd));
          CHKERRQ(ISLocalToGlobalMappingDestroy(&map));
          CHKERRQ(MatISSetPreallocation(Abd,bs,NULL,0,NULL));
          for (i=0;i<nl;i++) {
            PetscInt b1,b2;

            for (b1=0;b1<bs;b1++) {
              for (b2=0;b2<bs;b2++) {
                vals[b1*bs + b2] = i*bs*bs + b1*bs + b2 + 1 + (b1 == b2 ? 1.0 : 0);
              }
            }
            CHKERRQ(MatSetValuesBlockedLocal(Abd,1,&i,1,&i,vals,INSERT_VALUES));
          }
          CHKERRQ(MatAssemblyBegin(Abd,MAT_FINAL_ASSEMBLY));
          CHKERRQ(MatAssemblyEnd(Abd,MAT_FINAL_ASSEMBLY));
          CHKERRQ(MatConvert(Abd,MATAIJ,MAT_INITIAL_MATRIX,&Bbd));
          CHKERRQ(MatInvertBlockDiagonal(Abd,&isbd));
          CHKERRQ(MatInvertBlockDiagonal(Bbd,&aijbd));
          CHKERRQ(MatGetLocalSize(Bbd,&nl,NULL));
          ok   = PETSC_TRUE;
          for (i=0;i<nl/bs;i++) {
            PetscInt b1,b2;

            for (b1=0;b1<bs;b1++) {
              for (b2=0;b2<bs;b2++) {
                if (PetscAbsScalar(isbd[i*bs*bs+b1*bs + b2]-aijbd[i*bs*bs+b1*bs + b2]) > PETSC_SMALL) ok = PETSC_FALSE;
                if (!ok) {
                  CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"[%d] ERROR block %" PetscInt_FMT ", entry %" PetscInt_FMT " %" PetscInt_FMT ": %g %g\n",rank,i,b1,b2,(double)PetscAbsScalar(isbd[i*bs*bs+b1*bs + b2]),(double)PetscAbsScalar(aijbd[i*bs*bs+b1*bs + b2])));
                  break;
                }
              }
              if (!ok) break;
            }
            if (!ok) break;
          }
          CHKERRQ(MatDestroy(&Abd));
          CHKERRQ(MatDestroy(&Bbd));
          CHKERRQ(PetscFree(vals));
          CHKERRQ(ISDestroy(&is));
          CHKERRQ(ISDestroy(&bis));
        }
      }
    }
  }
  /* free testing matrices */
  CHKERRQ(ISLocalToGlobalMappingDestroy(&cmap));
  CHKERRQ(ISLocalToGlobalMappingDestroy(&rmap));
  CHKERRQ(MatDestroy(&A));
  CHKERRQ(MatDestroy(&B));
  CHKERRQ(PetscFinalize());
  return 0;
}

PetscErrorCode CheckMat(Mat A, Mat B, PetscBool usemult, const char* func)
{
  Mat            Bcheck;
  PetscReal      error;

  PetscFunctionBeginUser;
  if (!usemult && B) {
    PetscBool hasnorm;

    CHKERRQ(MatHasOperation(B,MATOP_NORM,&hasnorm));
    if (!hasnorm) usemult = PETSC_TRUE;
  }
  if (!usemult) {
    if (B) {
      MatType Btype;

      CHKERRQ(MatGetType(B,&Btype));
      CHKERRQ(MatConvert(A,Btype,MAT_INITIAL_MATRIX,&Bcheck));
    } else {
      CHKERRQ(MatConvert(A,MATAIJ,MAT_INITIAL_MATRIX,&Bcheck));
    }
    if (B) { /* if B is present, subtract it */
      CHKERRQ(MatAXPY(Bcheck,-1.,B,DIFFERENT_NONZERO_PATTERN));
    }
    CHKERRQ(MatNorm(Bcheck,NORM_INFINITY,&error));
    if (error > PETSC_SQRT_MACHINE_EPSILON) {
      ISLocalToGlobalMapping rl2g,cl2g;

      CHKERRQ(PetscObjectSetName((PetscObject)Bcheck,"Assembled Bcheck"));
      CHKERRQ(MatView(Bcheck,NULL));
      if (B) {
        CHKERRQ(PetscObjectSetName((PetscObject)B,"Assembled AIJ"));
        CHKERRQ(MatView(B,NULL));
        CHKERRQ(MatDestroy(&Bcheck));
        CHKERRQ(MatConvert(A,MATAIJ,MAT_INITIAL_MATRIX,&Bcheck));
        CHKERRQ(PetscObjectSetName((PetscObject)Bcheck,"Assembled IS"));
        CHKERRQ(MatView(Bcheck,NULL));
      }
      CHKERRQ(MatDestroy(&Bcheck));
      CHKERRQ(PetscObjectSetName((PetscObject)A,"MatIS"));
      CHKERRQ(MatView(A,NULL));
      CHKERRQ(MatGetLocalToGlobalMapping(A,&rl2g,&cl2g));
      CHKERRQ(ISLocalToGlobalMappingView(rl2g,NULL));
      CHKERRQ(ISLocalToGlobalMappingView(cl2g,NULL));
      SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"ERROR ON %s: %g",func,(double)error);
    }
    CHKERRQ(MatDestroy(&Bcheck));
  } else {
    PetscBool ok,okt;

    CHKERRQ(MatMultEqual(A,B,3,&ok));
    CHKERRQ(MatMultTransposeEqual(A,B,3,&okt));
    PetscCheckFalse(!ok || !okt,PETSC_COMM_WORLD,PETSC_ERR_PLIB,"ERROR ON %s: mult ok ?  %d, multtranspose ok ? %d",func,ok,okt);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode TestMatZeroRows(Mat A, Mat Afull, PetscBool squaretest, IS is, PetscScalar diag)
{
  Mat                    B,Bcheck,B2 = NULL,lB;
  Vec                    x = NULL, b = NULL, b2 = NULL;
  ISLocalToGlobalMapping l2gr,l2gc;
  PetscReal              error;
  char                   diagstr[16];
  const PetscInt         *idxs;
  PetscInt               rst,ren,i,n,N,d;
  PetscMPIInt            rank;
  PetscBool              miss,haszerorows;

  PetscFunctionBeginUser;
  if (diag == 0.) {
    CHKERRQ(PetscStrcpy(diagstr,"zero"));
  } else {
    CHKERRQ(PetscStrcpy(diagstr,"nonzero"));
  }
  CHKERRQ(ISView(is,NULL));
  CHKERRQ(MatGetLocalToGlobalMapping(A,&l2gr,&l2gc));
  /* tests MatDuplicate and MatCopy */
  if (diag == 0.) {
    CHKERRQ(MatDuplicate(A,MAT_COPY_VALUES,&B));
  } else {
    CHKERRQ(MatDuplicate(A,MAT_DO_NOT_COPY_VALUES,&B));
    CHKERRQ(MatCopy(A,B,SAME_NONZERO_PATTERN));
  }
  CHKERRQ(MatISGetLocalMat(B,&lB));
  CHKERRQ(MatHasOperation(lB,MATOP_ZERO_ROWS,&haszerorows));
  if (squaretest && haszerorows) {

    CHKERRQ(MatCreateVecs(B,&x,&b));
    CHKERRQ(MatDuplicate(B,MAT_COPY_VALUES,&B2));
    CHKERRQ(VecSetLocalToGlobalMapping(b,l2gr));
    CHKERRQ(VecSetLocalToGlobalMapping(x,l2gc));
    CHKERRQ(VecSetRandom(x,NULL));
    CHKERRQ(VecSetRandom(b,NULL));
    /* mimic b[is] = x[is] */
    CHKERRQ(VecDuplicate(b,&b2));
    CHKERRQ(VecSetLocalToGlobalMapping(b2,l2gr));
    CHKERRQ(VecCopy(b,b2));
    CHKERRQ(ISGetLocalSize(is,&n));
    CHKERRQ(ISGetIndices(is,&idxs));
    CHKERRQ(VecGetSize(x,&N));
    for (i=0;i<n;i++) {
      if (0 <= idxs[i] && idxs[i] < N) {
        CHKERRQ(VecSetValue(b2,idxs[i],diag,INSERT_VALUES));
        CHKERRQ(VecSetValue(x,idxs[i],1.,INSERT_VALUES));
      }
    }
    CHKERRQ(VecAssemblyBegin(b2));
    CHKERRQ(VecAssemblyEnd(b2));
    CHKERRQ(VecAssemblyBegin(x));
    CHKERRQ(VecAssemblyEnd(x));
    CHKERRQ(ISRestoreIndices(is,&idxs));
    /*  test ZeroRows on MATIS */
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Test MatZeroRows (diag %s)\n",diagstr));
    CHKERRQ(MatZeroRowsIS(B,is,diag,x,b));
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Test MatZeroRowsColumns (diag %s)\n",diagstr));
    CHKERRQ(MatZeroRowsColumnsIS(B2,is,diag,NULL,NULL));
  } else if (haszerorows) {
    /*  test ZeroRows on MATIS */
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Test MatZeroRows (diag %s)\n",diagstr));
    CHKERRQ(MatZeroRowsIS(B,is,diag,NULL,NULL));
    b = b2 = x = NULL;
  } else {
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Skipping MatZeroRows (diag %s)\n",diagstr));
    b = b2 = x = NULL;
  }

  if (squaretest && haszerorows) {
    CHKERRQ(VecAXPY(b2,-1.,b));
    CHKERRQ(VecNorm(b2,NORM_INFINITY,&error));
    PetscCheckFalse(error > PETSC_SQRT_MACHINE_EPSILON,PETSC_COMM_WORLD,PETSC_ERR_PLIB,"ERROR IN ZEROROWS ON B %g (diag %s)",(double)error,diagstr);
  }

  /* test MatMissingDiagonal */
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Test MatMissingDiagonal\n"));
  CHKERRMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  CHKERRQ(MatMissingDiagonal(B,&miss,&d));
  CHKERRQ(MatGetOwnershipRange(B,&rst,&ren));
  CHKERRQ(PetscViewerASCIIPushSynchronized(PETSC_VIEWER_STDOUT_WORLD));
  CHKERRQ(PetscViewerASCIISynchronizedPrintf(PETSC_VIEWER_STDOUT_WORLD, "[%d] [%" PetscInt_FMT ",%" PetscInt_FMT ") Missing %d, row %" PetscInt_FMT " (diag %s)\n",rank,rst,ren,(int)miss,d,diagstr));
  CHKERRQ(PetscViewerFlush(PETSC_VIEWER_STDOUT_WORLD));
  CHKERRQ(PetscViewerASCIIPopSynchronized(PETSC_VIEWER_STDOUT_WORLD));

  CHKERRQ(VecDestroy(&x));
  CHKERRQ(VecDestroy(&b));
  CHKERRQ(VecDestroy(&b2));

  /* check the result of ZeroRows with that from MPIAIJ routines
     assuming that MatConvert_IS_XAIJ and MatZeroRows_MPIAIJ work fine */
  if (haszerorows) {
    CHKERRQ(MatDuplicate(Afull,MAT_COPY_VALUES,&Bcheck));
    CHKERRQ(MatSetOption(Bcheck,MAT_NEW_NONZERO_ALLOCATION_ERR,PETSC_FALSE));
    CHKERRQ(MatZeroRowsIS(Bcheck,is,diag,NULL,NULL));
    CHKERRQ(CheckMat(B,Bcheck,PETSC_FALSE,"Zerorows"));
    CHKERRQ(MatDestroy(&Bcheck));
  }
  CHKERRQ(MatDestroy(&B));

  if (B2) { /* test MatZeroRowsColumns */
    CHKERRQ(MatDuplicate(Afull,MAT_COPY_VALUES,&B));
    CHKERRQ(MatSetOption(B,MAT_NEW_NONZERO_ALLOCATION_ERR,PETSC_FALSE));
    CHKERRQ(MatZeroRowsColumnsIS(B,is,diag,NULL,NULL));
    CHKERRQ(CheckMat(B2,B,PETSC_FALSE,"MatZeroRowsColumns"));
    CHKERRQ(MatDestroy(&B));
    CHKERRQ(MatDestroy(&B2));
  }
  PetscFunctionReturn(0);
}

/*TEST

   test:
      args: -test_trans

   test:
      suffix: 2
      nsize: 4
      args: -matis_convert_local_nest -nr 3 -nc 4

   test:
      suffix: 3
      nsize: 5
      args: -m 11 -n 10 -matis_convert_local_nest -nr 2 -nc 1

   test:
      suffix: 4
      nsize: 6
      args: -m 9 -n 12 -test_trans -nr 2 -nc 7

   test:
      suffix: 5
      nsize: 6
      args: -m 12 -n 12 -test_trans -nr 3 -nc 1

   test:
      suffix: 6
      args: -m 12 -n 12 -test_trans -nr 2 -nc 3 -diffmap

   test:
      suffix: 7
      args: -m 12 -n 12 -test_trans -nr 2 -nc 3 -diffmap -permmap

   test:
      suffix: 8
      args: -m 12 -n 17 -test_trans -nr 2 -nc 3 -permmap

   test:
      suffix: 9
      nsize: 5
      args: -m 12 -n 12 -test_trans -nr 2 -nc 3 -diffmap

   test:
      suffix: 10
      nsize: 5
      args: -m 12 -n 12 -test_trans -nr 2 -nc 3 -diffmap -permmap

   test:
      suffix: vscat_default
      nsize: 5
      args: -m 12 -n 17 -test_trans -nr 2 -nc 3 -permmap
      output_file: output/ex23_11.out

   test:
      suffix: 12
      nsize: 3
      args: -m 12 -n 12 -symmetric -matis_localmat_type sbaij -test_trans -nr 2 -nc 3

   testset:
      output_file: output/ex23_13.out
      nsize: 3
      args: -m 12 -n 17 -test_trans -nr 2 -nc 3 -diffmap -permmap
      filter: grep -v "type:"
      test:
        suffix: baij
        args: -matis_localmat_type baij
      test:
        requires: viennacl
        suffix: viennacl
        args: -matis_localmat_type aijviennacl
      test:
        requires: cuda
        suffix: cusparse
        args: -matis_localmat_type aijcusparse

   test:
      suffix: negrep
      nsize: {{1 3}separate output}
      args: -m {{5 7}separate output} -n {{5 7}separate output} -test_trans -nr 2 -nc 3 -negmap {{0 1}separate output} -repmap {{0 1}separate output} -permmap -diffmap {{0 1}separate output}

TEST*/
