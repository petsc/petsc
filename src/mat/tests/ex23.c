
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
  PetscInt               n,m,i;
  PetscInt               rst,ren,cst,cen,nr,nc;
  PetscMPIInt            rank,size;
  PetscBool              testT,squaretest,isaij;
  PetscBool              permute = PETSC_FALSE;
  PetscBool              diffmap = PETSC_TRUE, symmetric = PETSC_FALSE;
  PetscErrorCode         ierr;

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRMPI(ierr);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRMPI(ierr);
  m = n = 2*size;
  ierr = PetscOptionsGetBool(NULL,NULL,"-symmetric",&symmetric,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-m",&m,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL);CHKERRQ(ierr);
  PetscCheckFalse(size > 1 && m < 4,PETSC_COMM_WORLD,PETSC_ERR_ARG_WRONG,"Number of rows should be larger or equal 4 for parallel runs");
  PetscCheckFalse(size == 1 && m < 2,PETSC_COMM_WORLD,PETSC_ERR_ARG_WRONG,"Number of rows should be larger or equal 2 for uniprocessor runs");
  PetscCheckFalse(n < 2,PETSC_COMM_WORLD,PETSC_ERR_ARG_WRONG,"Number of cols should be larger or equal 2");
  if (symmetric) m = n = PetscMax(m,n);

  /* create a MATIS matrix */
  ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
  ierr = MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,m,n);CHKERRQ(ierr);
  ierr = MatSetType(A,MATIS);CHKERRQ(ierr);
  ierr = MatSetFromOptions(A);CHKERRQ(ierr);
  /* This is not the proper setting for MATIS for finite elements, it is just used to test the routines
     Here we use a one-to-one correspondence between local row/column spaces and global row/column spaces
     Equivalent to passing NULL for the mapping */
  ierr = ISCreateStride(PETSC_COMM_WORLD,n,0,1,&is);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingCreateIS(is,&cmap);CHKERRQ(ierr);
  ierr = ISDestroy(&is);CHKERRQ(ierr);

  ierr = PetscOptionsGetBool(NULL,NULL,"-permmap",&permute,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL,"-diffmap",&diffmap,NULL);CHKERRQ(ierr);
  if (!symmetric && (diffmap || m != n)) {

    ierr = ISCreateStride(PETSC_COMM_WORLD,m,permute ? m -1 : 0,permute ? -1 : 1,&is);CHKERRQ(ierr);
    ierr = ISLocalToGlobalMappingCreateIS(is,&rmap);CHKERRQ(ierr);
    ierr = ISDestroy(&is);CHKERRQ(ierr);
    if (m==n && !permute) squaretest = PETSC_TRUE;
    else squaretest = PETSC_FALSE;
  } else {
    ierr = PetscObjectReference((PetscObject)cmap);CHKERRQ(ierr);
    rmap = cmap;
    squaretest = PETSC_TRUE;
  }
  ierr = MatSetLocalToGlobalMapping(A,rmap,cmap);CHKERRQ(ierr);
  ierr = MatISStoreL2L(A,PETSC_FALSE);CHKERRQ(ierr);
  ierr = MatISSetPreallocation(A,3,NULL,0,NULL);CHKERRQ(ierr);
  for (i=0; i<m; i++) {
    PetscScalar v[3];
    PetscInt    cols[3];

    cols[0] = (i-1+n)%n;
    cols[1] = i%n;
    cols[2] = (i+1)%n;
    v[0]    = -1.*(symmetric ? PetscMin(i+1,cols[0]+1) : i+1);
    v[1]    =  2.*(symmetric ? PetscMin(i+1,cols[1]+1) : i+1);
    v[2]    = -1.*(symmetric ? PetscMin(i+1,cols[2]+1) : i+1);
    ierr    = MatSetValuesLocal(A,1,&i,3,cols,v,ADD_VALUES);CHKERRQ(ierr);
  }
  if (symmetric) {
    ierr = MatSetOption(A,MAT_SYMMETRIC,PETSC_TRUE);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  ierr = MatISGetLocalMat(A,&B);CHKERRQ(ierr);
  ierr = MatGetType(B,&lmtype);CHKERRQ(ierr);

  /* test MatGetInfo */
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Test MatGetInfo\n");CHKERRQ(ierr);
  if (!PetscGlobalRank) {
    ierr = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }
  ierr = MatGetInfo(A,MAT_LOCAL,&info);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPushSynchronized(PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = PetscViewerASCIISynchronizedPrintf(PETSC_VIEWER_STDOUT_WORLD,"Process  %2d: %" PetscInt_FMT " %" PetscInt_FMT " %" PetscInt_FMT " %" PetscInt_FMT " %" PetscInt_FMT "\n",PetscGlobalRank,(PetscInt)info.nz_used,
                                            (PetscInt)info.nz_allocated,(PetscInt)info.nz_unneeded,(PetscInt)info.assemblies,(PetscInt)info.mallocs);CHKERRQ(ierr);
  ierr = PetscViewerFlush(PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = MatGetInfo(A,MAT_GLOBAL_MAX,&info);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_WORLD,"GlobalMax  : %" PetscInt_FMT " %" PetscInt_FMT " %" PetscInt_FMT " %" PetscInt_FMT " %" PetscInt_FMT "\n",(PetscInt)info.nz_used,
                                (PetscInt)info.nz_allocated,(PetscInt)info.nz_unneeded,(PetscInt)info.assemblies,(PetscInt)info.mallocs);CHKERRQ(ierr);
  ierr = MatGetInfo(A,MAT_GLOBAL_SUM,&info);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_WORLD,"GlobalSum  : %" PetscInt_FMT " %" PetscInt_FMT " %" PetscInt_FMT " %" PetscInt_FMT " %" PetscInt_FMT "\n",(PetscInt)info.nz_used,
                                (PetscInt)info.nz_allocated,(PetscInt)info.nz_unneeded,(PetscInt)info.assemblies,(PetscInt)info.mallocs);CHKERRQ(ierr);

  /* test MatView */
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Test MatView\n");CHKERRQ(ierr);
  ierr = MatView(A,NULL);CHKERRQ(ierr);

  /* Create a MPIAIJ matrix, same as A */
  ierr = MatCreate(PETSC_COMM_WORLD,&B);CHKERRQ(ierr);
  ierr = MatSetSizes(B,PETSC_DECIDE,PETSC_DECIDE,m,n);CHKERRQ(ierr);
  ierr = MatSetType(B,MATAIJ);CHKERRQ(ierr);
  ierr = MatSetFromOptions(B);CHKERRQ(ierr);
  ierr = MatSetUp(B);CHKERRQ(ierr);
  ierr = MatSetLocalToGlobalMapping(B,rmap,cmap);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(B,3,NULL,3,NULL);CHKERRQ(ierr);
  ierr = MatMPIBAIJSetPreallocation(B,1,3,NULL,3,NULL);CHKERRQ(ierr);
#if defined(PETSC_HAVE_HYPRE)
  ierr = MatHYPRESetPreallocation(B,3,NULL,3,NULL);CHKERRQ(ierr);
#endif
  ierr = MatISSetPreallocation(B,3,NULL,3,NULL);CHKERRQ(ierr);
  for (i=0; i<m; i++) {
    PetscScalar v[3];
    PetscInt    cols[3];

    cols[0] = (i-1+n)%n;
    cols[1] = i%n;
    cols[2] = (i+1)%n;
    v[0]    = -1.*(symmetric ? PetscMin(i+1,cols[0]+1) : i+1);
    v[1]    =  2.*(symmetric ? PetscMin(i+1,cols[1]+1) : i+1);
    v[2]    = -1.*(symmetric ? PetscMin(i+1,cols[2]+1) : i+1);
    ierr    = MatSetValuesLocal(B,1,&i,3,cols,v,ADD_VALUES);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingDestroy(&cmap);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingDestroy(&rmap);CHKERRQ(ierr);

  /* test CheckMat */
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Test CheckMat\n");CHKERRQ(ierr);
  ierr = CheckMat(A,B,PETSC_FALSE,"CheckMat");CHKERRQ(ierr);

  /* test MatDuplicate and MatAXPY */
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Test MatDuplicate and MatAXPY\n");CHKERRQ(ierr);
  ierr = MatDuplicate(A,MAT_COPY_VALUES,&A2);CHKERRQ(ierr);
  ierr = CheckMat(A,A2,PETSC_FALSE,"MatDuplicate and MatAXPY");CHKERRQ(ierr);

  /* test MatConvert */
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Test MatConvert_IS_XAIJ\n");CHKERRQ(ierr);
  ierr = MatConvert(A2,MATAIJ,MAT_INITIAL_MATRIX,&B2);CHKERRQ(ierr);
  ierr = CheckMat(B,B2,PETSC_TRUE,"MatConvert_IS_XAIJ MAT_INITIAL_MATRIX");CHKERRQ(ierr);
  ierr = MatConvert(A2,MATAIJ,MAT_REUSE_MATRIX,&B2);CHKERRQ(ierr);
  ierr = CheckMat(B,B2,PETSC_TRUE,"MatConvert_IS_XAIJ MAT_REUSE_MATRIX");CHKERRQ(ierr);
  ierr = MatConvert(A2,MATAIJ,MAT_INPLACE_MATRIX,&A2);CHKERRQ(ierr);
  ierr = CheckMat(B,A2,PETSC_TRUE,"MatConvert_IS_XAIJ MAT_INPLACE_MATRIX");CHKERRQ(ierr);
  ierr = MatDestroy(&A2);CHKERRQ(ierr);
  ierr = MatDestroy(&B2);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Test MatConvert_XAIJ_IS\n");CHKERRQ(ierr);
  ierr = MatDuplicate(B,MAT_COPY_VALUES,&B2);CHKERRQ(ierr);
  ierr = MatConvert(B2,MATIS,MAT_INITIAL_MATRIX,&A2);CHKERRQ(ierr);
  ierr = CheckMat(A,A2,PETSC_TRUE,"MatConvert_XAIJ_IS MAT_INITIAL_MATRIX");CHKERRQ(ierr);
  ierr = MatConvert(B2,MATIS,MAT_REUSE_MATRIX,&A2);CHKERRQ(ierr);
  ierr = CheckMat(A,A2,PETSC_TRUE,"MatConvert_XAIJ_IS MAT_REUSE_MATRIX");CHKERRQ(ierr);
  ierr = MatConvert(B2,MATIS,MAT_INPLACE_MATRIX,&B2);CHKERRQ(ierr);
  ierr = CheckMat(A,B2,PETSC_TRUE,"MatConvert_XAIJ_IS MAT_INPLACE_MATRIX");CHKERRQ(ierr);
  ierr = MatDestroy(&A2);CHKERRQ(ierr);
  ierr = MatDestroy(&B2);CHKERRQ(ierr);
  ierr = PetscStrcmp(lmtype,MATSEQAIJ,&isaij);CHKERRQ(ierr);
  if (size == 1 && isaij) { /* tests special code paths in MatConvert_IS_XAIJ */
    PetscInt ri, ci, rr[3] = {0,1,0}, cr[4] = {1,2,0,1}, rk[3] = {0,2,1}, ck[4] = {1,0,3,2};

    for (ri = 0; ri < 2; ri++) {
      PetscInt *r;

      r = (PetscInt*)(ri == 0 ? rr : rk);
      for (ci = 0; ci < 2; ci++) {
        PetscInt *c,rb,cb;

        c = (PetscInt*)(ci == 0 ? cr : ck);
        for (rb = 1; rb < 4; rb++) {
          ierr = ISCreateBlock(PETSC_COMM_SELF,rb,3,r,PETSC_COPY_VALUES,&is);CHKERRQ(ierr);
          ierr = ISLocalToGlobalMappingCreateIS(is,&rmap);CHKERRQ(ierr);
          ierr = ISDestroy(&is);CHKERRQ(ierr);
          for (cb = 1; cb < 4; cb++) {
            Mat  T,lT,T2;
            char testname[256];

            ierr = PetscSNPrintf(testname,sizeof(testname),"MatConvert_IS_XAIJ special case (%" PetscInt_FMT " %" PetscInt_FMT ", bs %" PetscInt_FMT " %" PetscInt_FMT ")",ri,ci,rb,cb);CHKERRQ(ierr);
            ierr = PetscPrintf(PETSC_COMM_WORLD,"Test %s\n",testname);CHKERRQ(ierr);

            ierr = ISCreateBlock(PETSC_COMM_SELF,cb,4,c,PETSC_COPY_VALUES,&is);CHKERRQ(ierr);
            ierr = ISLocalToGlobalMappingCreateIS(is,&cmap);CHKERRQ(ierr);
            ierr = ISDestroy(&is);CHKERRQ(ierr);

            ierr = MatCreate(PETSC_COMM_SELF,&T);CHKERRQ(ierr);
            ierr = MatSetSizes(T,PETSC_DECIDE,PETSC_DECIDE,rb*3,cb*4);CHKERRQ(ierr);
            ierr = MatSetType(T,MATIS);CHKERRQ(ierr);
            ierr = MatSetLocalToGlobalMapping(T,rmap,cmap);CHKERRQ(ierr);
            ierr = ISLocalToGlobalMappingDestroy(&cmap);CHKERRQ(ierr);
            ierr = MatISGetLocalMat(T,&lT);CHKERRQ(ierr);
            ierr = MatSetType(lT,MATSEQAIJ);CHKERRQ(ierr);
            ierr = MatSeqAIJSetPreallocation(lT,cb*4,NULL);CHKERRQ(ierr);
            ierr = MatSetRandom(lT,NULL);CHKERRQ(ierr);
            ierr = MatConvert(lT,lmtype,MAT_INPLACE_MATRIX,&lT);CHKERRQ(ierr);
            ierr = MatISRestoreLocalMat(T,&lT);CHKERRQ(ierr);
            ierr = MatAssemblyBegin(T,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
            ierr = MatAssemblyEnd(T,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

            ierr = MatConvert(T,MATAIJ,MAT_INITIAL_MATRIX,&T2);CHKERRQ(ierr);
            ierr = CheckMat(T,T2,PETSC_TRUE,"MAT_INITIAL_MATRIX");CHKERRQ(ierr);
            ierr = MatConvert(T,MATAIJ,MAT_REUSE_MATRIX,&T2);CHKERRQ(ierr);
            ierr = CheckMat(T,T2,PETSC_TRUE,"MAT_REUSE_MATRIX");CHKERRQ(ierr);
            ierr = MatDestroy(&T2);CHKERRQ(ierr);
            ierr = MatDuplicate(T,MAT_COPY_VALUES,&T2);CHKERRQ(ierr);
            ierr = MatConvert(T2,MATAIJ,MAT_INPLACE_MATRIX,&T2);CHKERRQ(ierr);
            ierr = CheckMat(T,T2,PETSC_TRUE,"MAT_INPLACE_MATRIX");CHKERRQ(ierr);
            ierr = MatDestroy(&T);CHKERRQ(ierr);
            ierr = MatDestroy(&T2);CHKERRQ(ierr);
          }
          ierr = ISLocalToGlobalMappingDestroy(&rmap);CHKERRQ(ierr);
        }
      }
    }
  }

  /* test MatDiagonalScale */
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Test MatDiagonalScale\n");CHKERRQ(ierr);
  ierr = MatDuplicate(A,MAT_COPY_VALUES,&A2);CHKERRQ(ierr);
  ierr = MatDuplicate(B,MAT_COPY_VALUES,&B2);CHKERRQ(ierr);
  ierr = MatCreateVecs(A,&x,&y);CHKERRQ(ierr);
  ierr = VecSetRandom(x,NULL);CHKERRQ(ierr);
  if (symmetric) {
    ierr = VecCopy(x,y);CHKERRQ(ierr);
  } else {
    ierr = VecSetRandom(y,NULL);CHKERRQ(ierr);
    ierr = VecScale(y,8.);CHKERRQ(ierr);
  }
  ierr = MatDiagonalScale(A2,y,x);CHKERRQ(ierr);
  ierr = MatDiagonalScale(B2,y,x);CHKERRQ(ierr);
  ierr = CheckMat(A2,B2,PETSC_FALSE,"MatDiagonalScale");CHKERRQ(ierr);
  ierr = MatDestroy(&A2);CHKERRQ(ierr);
  ierr = MatDestroy(&B2);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&y);CHKERRQ(ierr);

  /* test MatPtAP (A IS and B AIJ) */
  if (isaij && m == n) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Test MatPtAP\n");CHKERRQ(ierr);
    ierr = MatISStoreL2L(A,PETSC_TRUE);CHKERRQ(ierr);
    ierr = MatPtAP(A,B,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&A2);CHKERRQ(ierr);
    ierr = MatPtAP(B,B,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&B2);CHKERRQ(ierr);
    ierr = CheckMat(A2,B2,PETSC_FALSE,"MatPtAP MAT_INITIAL_MATRIX");CHKERRQ(ierr);
    ierr = MatPtAP(A,B,MAT_REUSE_MATRIX,PETSC_DEFAULT,&A2);CHKERRQ(ierr);
    ierr = CheckMat(A2,B2,PETSC_FALSE,"MatPtAP MAT_REUSE_MATRIX");CHKERRQ(ierr);
    ierr = MatDestroy(&A2);CHKERRQ(ierr);
    ierr = MatDestroy(&B2);CHKERRQ(ierr);
  }

  /* test MatGetLocalSubMatrix */
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Test MatGetLocalSubMatrix\n");CHKERRQ(ierr);
  ierr = MatDuplicate(A,MAT_DO_NOT_COPY_VALUES,&A2);CHKERRQ(ierr);
  ierr = ISCreateStride(PETSC_COMM_WORLD,m/2+m%2,0,2,&reven);CHKERRQ(ierr);
  ierr = ISCreateStride(PETSC_COMM_WORLD,m/2,1,2,&rodd);CHKERRQ(ierr);
  ierr = ISCreateStride(PETSC_COMM_WORLD,n/2+n%2,0,2,&ceven);CHKERRQ(ierr);
  ierr = ISCreateStride(PETSC_COMM_WORLD,n/2,1,2,&codd);CHKERRQ(ierr);
  ierr = MatGetLocalSubMatrix(A2,reven,ceven,&Aee);CHKERRQ(ierr);
  ierr = MatGetLocalSubMatrix(A2,reven,codd,&Aeo);CHKERRQ(ierr);
  ierr = MatGetLocalSubMatrix(A2,rodd,ceven,&Aoe);CHKERRQ(ierr);
  ierr = MatGetLocalSubMatrix(A2,rodd,codd,&Aoo);CHKERRQ(ierr);
  for (i=0; i<m; i++) {
    PetscInt    j,je,jo,colse[3], colso[3];
    PetscScalar ve[3], vo[3];
    PetscScalar v[3];
    PetscInt    cols[3];

    cols[0] = (i-1+n)%n;
    cols[1] = i%n;
    cols[2] = (i+1)%n;
    v[0]    = -1.*(symmetric ? PetscMin(i+1,cols[0]+1) : i+1);
    v[1]    =  2.*(symmetric ? PetscMin(i+1,cols[1]+1) : i+1);
    v[2]    = -1.*(symmetric ? PetscMin(i+1,cols[2]+1) : i+1);
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
      PetscInt row = i/2;
      ierr = MatSetValuesLocal(Aoe,1,&row,je,colse,ve,ADD_VALUES);CHKERRQ(ierr);
      ierr = MatSetValuesLocal(Aoo,1,&row,jo,colso,vo,ADD_VALUES);CHKERRQ(ierr);
    } else {
      PetscInt row = i/2;
      ierr = MatSetValuesLocal(Aee,1,&row,je,colse,ve,ADD_VALUES);CHKERRQ(ierr);
      ierr = MatSetValuesLocal(Aeo,1,&row,jo,colso,vo,ADD_VALUES);CHKERRQ(ierr);
    }
  }
  ierr = MatRestoreLocalSubMatrix(A2,reven,ceven,&Aee);CHKERRQ(ierr);
  ierr = MatRestoreLocalSubMatrix(A2,reven,codd,&Aeo);CHKERRQ(ierr);
  ierr = MatRestoreLocalSubMatrix(A2,rodd,ceven,&Aoe);CHKERRQ(ierr);
  ierr = MatRestoreLocalSubMatrix(A2,rodd,codd,&Aoo);CHKERRQ(ierr);
  ierr = ISDestroy(&reven);CHKERRQ(ierr);
  ierr = ISDestroy(&ceven);CHKERRQ(ierr);
  ierr = ISDestroy(&rodd);CHKERRQ(ierr);
  ierr = ISDestroy(&codd);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(A2,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A2,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAXPY(A2,-1.,A,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
  ierr = CheckMat(A2,NULL,PETSC_FALSE,"MatGetLocalSubMatrix");CHKERRQ(ierr);
  ierr = MatDestroy(&A2);CHKERRQ(ierr);

  /* test MatConvert_Nest_IS */
  testT = PETSC_FALSE;
  ierr  = PetscOptionsGetBool(NULL,NULL,"-test_trans",&testT,NULL);CHKERRQ(ierr);

  ierr = PetscPrintf(PETSC_COMM_WORLD,"Test MatConvert_Nest_IS\n");CHKERRQ(ierr);
  nr   = 2;
  nc   = 2;
  ierr = PetscOptionsGetInt(NULL,NULL,"-nr",&nr,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-nc",&nc,NULL);CHKERRQ(ierr);
  if (testT) {
    ierr = MatGetOwnershipRange(A,&cst,&cen);CHKERRQ(ierr);
    ierr = MatGetOwnershipRangeColumn(A,&rst,&ren);CHKERRQ(ierr);
  } else {
    ierr = MatGetOwnershipRange(A,&rst,&ren);CHKERRQ(ierr);
    ierr = MatGetOwnershipRangeColumn(A,&cst,&cen);CHKERRQ(ierr);
  }
  ierr = PetscMalloc3(nr,&rows,nc,&cols,2*nr*nc,&mats);CHKERRQ(ierr);
  for (i=0;i<nr*nc;i++) {
    if (testT) {
      ierr = MatCreateTranspose(A,&mats[i]);CHKERRQ(ierr);
      ierr = MatTranspose(B,MAT_INITIAL_MATRIX,&mats[i+nr*nc]);CHKERRQ(ierr);
    } else {
      ierr = MatDuplicate(A,MAT_COPY_VALUES,&mats[i]);CHKERRQ(ierr);
      ierr = MatDuplicate(B,MAT_COPY_VALUES,&mats[i+nr*nc]);CHKERRQ(ierr);
    }
  }
  for (i=0;i<nr;i++) {
    ierr = ISCreateStride(PETSC_COMM_WORLD,ren-rst,i+rst,nr,&rows[i]);CHKERRQ(ierr);
  }
  for (i=0;i<nc;i++) {
    ierr = ISCreateStride(PETSC_COMM_WORLD,cen-cst,i+cst,nc,&cols[i]);CHKERRQ(ierr);
  }
  ierr = MatCreateNest(PETSC_COMM_WORLD,nr,rows,nc,cols,mats,&A2);CHKERRQ(ierr);
  ierr = MatCreateNest(PETSC_COMM_WORLD,nr,rows,nc,cols,mats+nr*nc,&B2);CHKERRQ(ierr);
  for (i=0;i<nr;i++) {
    ierr = ISDestroy(&rows[i]);CHKERRQ(ierr);
  }
  for (i=0;i<nc;i++) {
    ierr = ISDestroy(&cols[i]);CHKERRQ(ierr);
  }
  for (i=0;i<2*nr*nc;i++) {
    ierr = MatDestroy(&mats[i]);CHKERRQ(ierr);
  }
  ierr = PetscFree3(rows,cols,mats);CHKERRQ(ierr);
  ierr = MatConvert(B2,MATAIJ,MAT_INITIAL_MATRIX,&T);CHKERRQ(ierr);
  ierr = MatDestroy(&B2);CHKERRQ(ierr);
  ierr = MatConvert(A2,MATIS,MAT_INITIAL_MATRIX,&B2);CHKERRQ(ierr);
  ierr = CheckMat(B2,T,PETSC_TRUE,"MatConvert_Nest_IS MAT_INITIAL_MATRIX");CHKERRQ(ierr);
  ierr = MatConvert(A2,MATIS,MAT_REUSE_MATRIX,&B2);CHKERRQ(ierr);
  ierr = CheckMat(B2,T,PETSC_TRUE,"MatConvert_Nest_IS MAT_REUSE_MATRIX");CHKERRQ(ierr);
  ierr = MatDestroy(&B2);CHKERRQ(ierr);
  ierr = MatConvert(A2,MATIS,MAT_INPLACE_MATRIX,&A2);CHKERRQ(ierr);
  ierr = CheckMat(A2,T,PETSC_TRUE,"MatConvert_Nest_IS MAT_INPLACE_MATRIX");CHKERRQ(ierr);
  ierr = MatDestroy(&T);CHKERRQ(ierr);
  ierr = MatDestroy(&A2);CHKERRQ(ierr);

  /* test MatCreateSubMatrix */
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Test MatCreateSubMatrix\n");CHKERRQ(ierr);
  if (rank == 0) {
    ierr = ISCreateStride(PETSC_COMM_WORLD,1,1,1,&is);CHKERRQ(ierr);
    ierr = ISCreateStride(PETSC_COMM_WORLD,2,0,1,&is2);CHKERRQ(ierr);
  } else if (rank == 1) {
    ierr = ISCreateStride(PETSC_COMM_WORLD,1,0,1,&is);CHKERRQ(ierr);
    if (n > 3) {
      ierr = ISCreateStride(PETSC_COMM_WORLD,1,3,1,&is2);CHKERRQ(ierr);
    } else {
      ierr = ISCreateStride(PETSC_COMM_WORLD,0,0,1,&is2);CHKERRQ(ierr);
    }
  } else if (rank == 2 && n > 4) {
    ierr = ISCreateStride(PETSC_COMM_WORLD,0,0,1,&is);CHKERRQ(ierr);
    ierr = ISCreateStride(PETSC_COMM_WORLD,n-4,4,1,&is2);CHKERRQ(ierr);
  } else {
    ierr = ISCreateStride(PETSC_COMM_WORLD,0,0,1,&is);CHKERRQ(ierr);
    ierr = ISCreateStride(PETSC_COMM_WORLD,0,0,1,&is2);CHKERRQ(ierr);
  }
  ierr = MatCreateSubMatrix(A,is,is,MAT_INITIAL_MATRIX,&A2);CHKERRQ(ierr);
  ierr = MatCreateSubMatrix(B,is,is,MAT_INITIAL_MATRIX,&B2);CHKERRQ(ierr);
  ierr = CheckMat(A2,B2,PETSC_TRUE,"first MatCreateSubMatrix");CHKERRQ(ierr);

  ierr = MatCreateSubMatrix(A,is,is,MAT_REUSE_MATRIX,&A2);CHKERRQ(ierr);
  ierr = MatCreateSubMatrix(B,is,is,MAT_REUSE_MATRIX,&B2);CHKERRQ(ierr);
  ierr = CheckMat(A2,B2,PETSC_FALSE,"reuse MatCreateSubMatrix");CHKERRQ(ierr);
  ierr = MatDestroy(&A2);CHKERRQ(ierr);
  ierr = MatDestroy(&B2);CHKERRQ(ierr);

  if (!symmetric) {
    ierr = MatCreateSubMatrix(A,is,is2,MAT_INITIAL_MATRIX,&A2);CHKERRQ(ierr);
    ierr = MatCreateSubMatrix(B,is,is2,MAT_INITIAL_MATRIX,&B2);CHKERRQ(ierr);
    ierr = MatCreateSubMatrix(A,is,is2,MAT_REUSE_MATRIX,&A2);CHKERRQ(ierr);
    ierr = MatCreateSubMatrix(B,is,is2,MAT_REUSE_MATRIX,&B2);CHKERRQ(ierr);
    ierr = CheckMat(A2,B2,PETSC_FALSE,"second MatCreateSubMatrix");CHKERRQ(ierr);
  }

  ierr = MatDestroy(&A2);CHKERRQ(ierr);
  ierr = MatDestroy(&B2);CHKERRQ(ierr);
  ierr = ISDestroy(&is);CHKERRQ(ierr);
  ierr = ISDestroy(&is2);CHKERRQ(ierr);

  /* Create an IS required by MatZeroRows(): just rank zero provides the rows to be eliminated */
  if (size > 1) {
    if (rank == 0) {
      PetscInt st,len;

      st   = (m+1)/2;
      len  = PetscMin(m/2,PetscMax(m-(m+1)/2-1,0));
      ierr = ISCreateStride(PETSC_COMM_WORLD,len,st,1,&is);CHKERRQ(ierr);
    } else {
      ierr = ISCreateStride(PETSC_COMM_WORLD,0,0,1,&is);CHKERRQ(ierr);
    }
  } else {
    ierr = ISCreateStride(PETSC_COMM_WORLD,1,0,1,&is);CHKERRQ(ierr);
  }

  if (squaretest) { /* tests for square matrices only, with same maps for rows and columns */
    /* test MatDiagonalSet */
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Test MatDiagonalSet\n");CHKERRQ(ierr);
    ierr = MatDuplicate(A,MAT_COPY_VALUES,&A2);CHKERRQ(ierr);
    ierr = MatDuplicate(B,MAT_COPY_VALUES,&B2);CHKERRQ(ierr);
    ierr = MatCreateVecs(A,NULL,&x);CHKERRQ(ierr);
    ierr = VecSetRandom(x,NULL);CHKERRQ(ierr);
    ierr = MatDiagonalSet(A2,x,INSERT_VALUES);CHKERRQ(ierr);
    ierr = MatDiagonalSet(B2,x,INSERT_VALUES);CHKERRQ(ierr);
    ierr = CheckMat(A2,B2,PETSC_FALSE,"MatDiagonalSet");CHKERRQ(ierr);
    ierr = VecDestroy(&x);CHKERRQ(ierr);
    ierr = MatDestroy(&A2);CHKERRQ(ierr);
    ierr = MatDestroy(&B2);CHKERRQ(ierr);

    /* test MatShift (MatShift_IS internally uses MatDiagonalSet_IS with ADD_VALUES) */
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Test MatShift\n");CHKERRQ(ierr);
    ierr = MatDuplicate(A,MAT_COPY_VALUES,&A2);CHKERRQ(ierr);
    ierr = MatDuplicate(B,MAT_COPY_VALUES,&B2);CHKERRQ(ierr);
    ierr = MatShift(A2,2.0);CHKERRQ(ierr);
    ierr = MatShift(B2,2.0);CHKERRQ(ierr);
    ierr = CheckMat(A2,B2,PETSC_FALSE,"MatShift");CHKERRQ(ierr);
    ierr = MatDestroy(&A2);CHKERRQ(ierr);
    ierr = MatDestroy(&B2);CHKERRQ(ierr);

    /* nonzero diag value is supported for square matrices only */
    ierr = TestMatZeroRows(A,B,PETSC_TRUE,is,diag);CHKERRQ(ierr);
  }
  ierr = TestMatZeroRows(A,B,squaretest,is,0.0);CHKERRQ(ierr);
  ierr = ISDestroy(&is);CHKERRQ(ierr);

  /* test MatTranspose */
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Test MatTranspose\n");CHKERRQ(ierr);
  ierr = MatTranspose(A,MAT_INITIAL_MATRIX,&A2);CHKERRQ(ierr);
  ierr = MatTranspose(B,MAT_INITIAL_MATRIX,&B2);CHKERRQ(ierr);
  ierr = CheckMat(A2,B2,PETSC_FALSE,"initial matrix MatTranspose");CHKERRQ(ierr);

  ierr = MatTranspose(A,MAT_REUSE_MATRIX,&A2);CHKERRQ(ierr);
  ierr = CheckMat(A2,B2,PETSC_FALSE,"reuse matrix (not in place) MatTranspose");CHKERRQ(ierr);
  ierr = MatDestroy(&A2);CHKERRQ(ierr);

  ierr = MatDuplicate(A,MAT_COPY_VALUES,&A2);CHKERRQ(ierr);
  ierr = MatTranspose(A2,MAT_INPLACE_MATRIX,&A2);CHKERRQ(ierr);
  ierr = CheckMat(A2,B2,PETSC_FALSE,"reuse matrix (in place) MatTranspose");CHKERRQ(ierr);
  ierr = MatDestroy(&A2);CHKERRQ(ierr);

  ierr = MatTranspose(A,MAT_INITIAL_MATRIX,&A2);CHKERRQ(ierr);
  ierr = CheckMat(A2,B2,PETSC_FALSE,"reuse matrix (different type) MatTranspose");CHKERRQ(ierr);
  ierr = MatDestroy(&A2);CHKERRQ(ierr);
  ierr = MatDestroy(&B2);CHKERRQ(ierr);

  /* test MatISFixLocalEmpty */
  if (isaij) {
    PetscInt r[2];

    r[0] = 0;
    r[1] = PetscMin(m,n)-1;
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Test MatISFixLocalEmpty\n");CHKERRQ(ierr);
    ierr = MatDuplicate(A,MAT_COPY_VALUES,&A2);CHKERRQ(ierr);
    ierr = MatISFixLocalEmpty(A2,PETSC_TRUE);CHKERRQ(ierr);
    ierr = MatAssemblyBegin(A2,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(A2,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = CheckMat(A2,B,PETSC_FALSE,"MatISFixLocalEmpty (null)");CHKERRQ(ierr);

    ierr = MatZeroRows(A2,2,r,0.0,NULL,NULL);CHKERRQ(ierr);
    ierr = MatViewFromOptions(A2,NULL,"-fixempty_view");CHKERRQ(ierr);
    ierr = MatDuplicate(B,MAT_COPY_VALUES,&B2);CHKERRQ(ierr);
    ierr = MatZeroRows(B2,2,r,0.0,NULL,NULL);CHKERRQ(ierr);
    ierr = MatISFixLocalEmpty(A2,PETSC_TRUE);CHKERRQ(ierr);
    ierr = MatAssemblyBegin(A2,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(A2,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatViewFromOptions(A2,NULL,"-fixempty_view");CHKERRQ(ierr);
    ierr = CheckMat(A2,B2,PETSC_FALSE,"MatISFixLocalEmpty (rows)");CHKERRQ(ierr);
    ierr = MatDestroy(&A2);CHKERRQ(ierr);

    ierr = MatDuplicate(A,MAT_COPY_VALUES,&A2);CHKERRQ(ierr);
    ierr = MatZeroRows(A2,2,r,0.0,NULL,NULL);CHKERRQ(ierr);
    ierr = MatTranspose(A2,MAT_INPLACE_MATRIX,&A2);CHKERRQ(ierr);
    ierr = MatTranspose(B2,MAT_INPLACE_MATRIX,&B2);CHKERRQ(ierr);
    ierr = MatViewFromOptions(A2,NULL,"-fixempty_view");CHKERRQ(ierr);
    ierr = MatISFixLocalEmpty(A2,PETSC_TRUE);CHKERRQ(ierr);
    ierr = MatAssemblyBegin(A2,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(A2,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatViewFromOptions(A2,NULL,"-fixempty_view");CHKERRQ(ierr);
    ierr = CheckMat(A2,B2,PETSC_FALSE,"MatISFixLocalEmpty (cols)");CHKERRQ(ierr);

    ierr = MatDestroy(&A2);CHKERRQ(ierr);
    ierr = MatDestroy(&B2);CHKERRQ(ierr);

    if (squaretest) {
      ierr = MatDuplicate(A,MAT_COPY_VALUES,&A2);CHKERRQ(ierr);
      ierr = MatDuplicate(B,MAT_COPY_VALUES,&B2);CHKERRQ(ierr);
      ierr = MatZeroRowsColumns(A2,2,r,0.0,NULL,NULL);CHKERRQ(ierr);
      ierr = MatZeroRowsColumns(B2,2,r,0.0,NULL,NULL);CHKERRQ(ierr);
      ierr = MatViewFromOptions(A2,NULL,"-fixempty_view");CHKERRQ(ierr);
      ierr = MatISFixLocalEmpty(A2,PETSC_TRUE);CHKERRQ(ierr);
      ierr = MatAssemblyBegin(A2,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
      ierr = MatAssemblyEnd(A2,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
      ierr = MatViewFromOptions(A2,NULL,"-fixempty_view");CHKERRQ(ierr);
      ierr = CheckMat(A2,B2,PETSC_FALSE,"MatISFixLocalEmpty (rows+cols)");CHKERRQ(ierr);
      ierr = MatDestroy(&A2);CHKERRQ(ierr);
      ierr = MatDestroy(&B2);CHKERRQ(ierr);
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
          ierr = PetscPrintf(PETSC_COMM_WORLD,"Test MatInvertBlockDiagonal blockdiag %" PetscInt_FMT " %" PetscInt_FMT " %" PetscInt_FMT " %" PetscInt_FMT "\n",n,diff,perm,bs);CHKERRQ(ierr);
          ierr = PetscMalloc1(bs*bs,&vals);CHKERRQ(ierr);
          ierr = MatGetOwnershipRanges(A,&sts);CHKERRQ(ierr);
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
          ierr = ISCreateStride(PETSC_COMM_WORLD,nl,st,in,&is);CHKERRQ(ierr);
          ierr = ISGetLocalSize(is,&nl);CHKERRQ(ierr);
          ierr = ISGetIndices(is,&idxs);CHKERRQ(ierr);
          ierr = PetscMalloc1(nl,&idxs2);CHKERRQ(ierr);
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
          ierr = ISRestoreIndices(is,&idxs);CHKERRQ(ierr);
          ierr = ISCreateBlock(PETSC_COMM_WORLD,bs,nl,idxs2,PETSC_OWN_POINTER,&bis);CHKERRQ(ierr);
          ierr = ISLocalToGlobalMappingCreateIS(bis,&map);CHKERRQ(ierr);
          ierr = MatCreateIS(PETSC_COMM_WORLD,bs,PETSC_DECIDE,PETSC_DECIDE,bs*n,bs*n,map,map,&Abd);CHKERRQ(ierr);
          ierr = ISLocalToGlobalMappingDestroy(&map);CHKERRQ(ierr);
          ierr = MatISSetPreallocation(Abd,bs,NULL,0,NULL);CHKERRQ(ierr);
          for (i=0;i<nl;i++) {
            PetscInt b1,b2;

            for (b1=0;b1<bs;b1++) {
              for (b2=0;b2<bs;b2++) {
                vals[b1*bs + b2] = i*bs*bs + b1*bs + b2 + 1 + (b1 == b2 ? 1.0 : 0);
              }
            }
            ierr = MatSetValuesBlockedLocal(Abd,1,&i,1,&i,vals,INSERT_VALUES);CHKERRQ(ierr);
          }
          ierr = MatAssemblyBegin(Abd,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
          ierr = MatAssemblyEnd(Abd,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
          ierr = MatConvert(Abd,MATAIJ,MAT_INITIAL_MATRIX,&Bbd);CHKERRQ(ierr);
          ierr = MatInvertBlockDiagonal(Abd,&isbd);CHKERRQ(ierr);
          ierr = MatInvertBlockDiagonal(Bbd,&aijbd);CHKERRQ(ierr);
          ierr = MatGetLocalSize(Bbd,&nl,NULL);CHKERRQ(ierr);
          ok   = PETSC_TRUE;
          for (i=0;i<nl/bs;i++) {
            PetscInt b1,b2;

            for (b1=0;b1<bs;b1++) {
              for (b2=0;b2<bs;b2++) {
                if (PetscAbsScalar(isbd[i*bs*bs+b1*bs + b2]-aijbd[i*bs*bs+b1*bs + b2]) > PETSC_SMALL) ok = PETSC_FALSE;
                if (!ok) {
                  ierr = PetscPrintf(PETSC_COMM_SELF,"[%d] ERROR block %" PetscInt_FMT ", entry %" PetscInt_FMT " %" PetscInt_FMT ": %g %g\n",rank,i,b1,b2,(double)PetscAbsScalar(isbd[i*bs*bs+b1*bs + b2]),(double)PetscAbsScalar(aijbd[i*bs*bs+b1*bs + b2]));CHKERRQ(ierr);
                  break;
                }
              }
              if (!ok) break;
            }
            if (!ok) break;
          }
          ierr = MatDestroy(&Abd);CHKERRQ(ierr);
          ierr = MatDestroy(&Bbd);CHKERRQ(ierr);
          ierr = PetscFree(vals);CHKERRQ(ierr);
          ierr = ISDestroy(&is);CHKERRQ(ierr);
          ierr = ISDestroy(&bis);CHKERRQ(ierr);
        }
      }
    }
  }
  /* free testing matrices */
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = MatDestroy(&B);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

PetscErrorCode CheckMat(Mat A, Mat B, PetscBool usemult, const char* func)
{
  Mat            Bcheck;
  PetscReal      error;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  if (!usemult && B) {
    PetscBool hasnorm;

    ierr = MatHasOperation(B,MATOP_NORM,&hasnorm);CHKERRQ(ierr);
    if (!hasnorm) usemult = PETSC_TRUE;
  }
  if (!usemult) {
    if (B) {
      MatType Btype;

      ierr = MatGetType(B,&Btype);CHKERRQ(ierr);
      ierr = MatConvert(A,Btype,MAT_INITIAL_MATRIX,&Bcheck);CHKERRQ(ierr);
    } else {
      ierr = MatConvert(A,MATAIJ,MAT_INITIAL_MATRIX,&Bcheck);CHKERRQ(ierr);
    }
    if (B) { /* if B is present, subtract it */
      ierr = MatAXPY(Bcheck,-1.,B,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
    }
    ierr = MatNorm(Bcheck,NORM_INFINITY,&error);CHKERRQ(ierr);
    if (error > PETSC_SQRT_MACHINE_EPSILON) {
      ISLocalToGlobalMapping rl2g,cl2g;

      ierr = PetscObjectSetName((PetscObject)Bcheck,"Assembled Bcheck");CHKERRQ(ierr);
      ierr = MatView(Bcheck,NULL);CHKERRQ(ierr);
      if (B) {
        ierr = PetscObjectSetName((PetscObject)B,"Assembled AIJ");CHKERRQ(ierr);
        ierr = MatView(B,NULL);CHKERRQ(ierr);
        ierr = MatDestroy(&Bcheck);CHKERRQ(ierr);
        ierr = MatConvert(A,MATAIJ,MAT_INITIAL_MATRIX,&Bcheck);CHKERRQ(ierr);
        ierr = PetscObjectSetName((PetscObject)Bcheck,"Assembled IS");CHKERRQ(ierr);
        ierr = MatView(Bcheck,NULL);CHKERRQ(ierr);
      }
      ierr = MatDestroy(&Bcheck);CHKERRQ(ierr);
      ierr = PetscObjectSetName((PetscObject)A,"MatIS");CHKERRQ(ierr);
      ierr = MatView(A,NULL);CHKERRQ(ierr);
      ierr = MatGetLocalToGlobalMapping(A,&rl2g,&cl2g);CHKERRQ(ierr);
      ierr = ISLocalToGlobalMappingView(rl2g,NULL);CHKERRQ(ierr);
      ierr = ISLocalToGlobalMappingView(cl2g,NULL);CHKERRQ(ierr);
      SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"ERROR ON %s: %g",func,(double)error);
    }
    ierr = MatDestroy(&Bcheck);CHKERRQ(ierr);
  } else {
    PetscBool ok,okt;

    ierr = MatMultEqual(A,B,3,&ok);CHKERRQ(ierr);
    ierr = MatMultTransposeEqual(A,B,3,&okt);CHKERRQ(ierr);
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
  PetscErrorCode         ierr;

  PetscFunctionBeginUser;
  if (diag == 0.) {
    ierr = PetscStrcpy(diagstr,"zero");CHKERRQ(ierr);
  } else {
    ierr = PetscStrcpy(diagstr,"nonzero");CHKERRQ(ierr);
  }
  ierr = ISView(is,NULL);CHKERRQ(ierr);
  ierr = MatGetLocalToGlobalMapping(A,&l2gr,&l2gc);CHKERRQ(ierr);
  /* tests MatDuplicate and MatCopy */
  if (diag == 0.) {
    ierr = MatDuplicate(A,MAT_COPY_VALUES,&B);CHKERRQ(ierr);
  } else {
    ierr = MatDuplicate(A,MAT_DO_NOT_COPY_VALUES,&B);CHKERRQ(ierr);
    ierr = MatCopy(A,B,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
  }
  ierr = MatISGetLocalMat(B,&lB);CHKERRQ(ierr);
  ierr = MatHasOperation(lB,MATOP_ZERO_ROWS,&haszerorows);CHKERRQ(ierr);
  if (squaretest && haszerorows) {

    ierr = MatCreateVecs(B,&x,&b);CHKERRQ(ierr);
    ierr = MatDuplicate(B,MAT_COPY_VALUES,&B2);CHKERRQ(ierr);
    ierr = VecSetLocalToGlobalMapping(b,l2gr);CHKERRQ(ierr);
    ierr = VecSetLocalToGlobalMapping(x,l2gc);CHKERRQ(ierr);
    ierr = VecSetRandom(x,NULL);CHKERRQ(ierr);
    ierr = VecSetRandom(b,NULL);CHKERRQ(ierr);
    /* mimic b[is] = x[is] */
    ierr = VecDuplicate(b,&b2);CHKERRQ(ierr);
    ierr = VecSetLocalToGlobalMapping(b2,l2gr);CHKERRQ(ierr);
    ierr = VecCopy(b,b2);CHKERRQ(ierr);
    ierr = ISGetLocalSize(is,&n);CHKERRQ(ierr);
    ierr = ISGetIndices(is,&idxs);CHKERRQ(ierr);
    ierr = VecGetSize(x,&N);CHKERRQ(ierr);
    for (i=0;i<n;i++) {
      if (0 <= idxs[i] && idxs[i] < N) {
        ierr = VecSetValue(b2,idxs[i],diag,INSERT_VALUES);CHKERRQ(ierr);
        ierr = VecSetValue(x,idxs[i],1.,INSERT_VALUES);CHKERRQ(ierr);
      }
    }
    ierr = VecAssemblyBegin(b2);CHKERRQ(ierr);
    ierr = VecAssemblyEnd(b2);CHKERRQ(ierr);
    ierr = VecAssemblyBegin(x);CHKERRQ(ierr);
    ierr = VecAssemblyEnd(x);CHKERRQ(ierr);
    ierr = ISRestoreIndices(is,&idxs);CHKERRQ(ierr);
    /*  test ZeroRows on MATIS */
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Test MatZeroRows (diag %s)\n",diagstr);CHKERRQ(ierr);
    ierr = MatZeroRowsIS(B,is,diag,x,b);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Test MatZeroRowsColumns (diag %s)\n",diagstr);CHKERRQ(ierr);
    ierr = MatZeroRowsColumnsIS(B2,is,diag,NULL,NULL);CHKERRQ(ierr);
  } else if (haszerorows) {
    /*  test ZeroRows on MATIS */
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Test MatZeroRows (diag %s)\n",diagstr);CHKERRQ(ierr);
    ierr = MatZeroRowsIS(B,is,diag,NULL,NULL);CHKERRQ(ierr);
    b = b2 = x = NULL;
  } else {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Skipping MatZeroRows (diag %s)\n",diagstr);CHKERRQ(ierr);
    b = b2 = x = NULL;
  }

  if (squaretest && haszerorows) {
    ierr = VecAXPY(b2,-1.,b);CHKERRQ(ierr);
    ierr = VecNorm(b2,NORM_INFINITY,&error);CHKERRQ(ierr);
    PetscCheckFalse(error > PETSC_SQRT_MACHINE_EPSILON,PETSC_COMM_WORLD,PETSC_ERR_PLIB,"ERROR IN ZEROROWS ON B %g (diag %s)",(double)error,diagstr);
  }

  /* test MatMissingDiagonal */
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Test MatMissingDiagonal\n");CHKERRQ(ierr);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRMPI(ierr);
  ierr = MatMissingDiagonal(B,&miss,&d);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(B,&rst,&ren);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPushSynchronized(PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = PetscViewerASCIISynchronizedPrintf(PETSC_VIEWER_STDOUT_WORLD, "[%d] [%" PetscInt_FMT ",%" PetscInt_FMT ") Missing %d, row %" PetscInt_FMT " (diag %s)\n",rank,rst,ren,(int)miss,d,diagstr);CHKERRQ(ierr);
  ierr = PetscViewerFlush(PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPopSynchronized(PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&b);CHKERRQ(ierr);
  ierr = VecDestroy(&b2);CHKERRQ(ierr);

  /* check the result of ZeroRows with that from MPIAIJ routines
     assuming that MatConvert_IS_XAIJ and MatZeroRows_MPIAIJ work fine */
  if (haszerorows) {
    ierr = MatDuplicate(Afull,MAT_COPY_VALUES,&Bcheck);CHKERRQ(ierr);
    ierr = MatSetOption(Bcheck,MAT_NEW_NONZERO_ALLOCATION_ERR,PETSC_FALSE);CHKERRQ(ierr);
    ierr = MatZeroRowsIS(Bcheck,is,diag,NULL,NULL);CHKERRQ(ierr);
    ierr = CheckMat(B,Bcheck,PETSC_FALSE,"Zerorows");CHKERRQ(ierr);
    ierr = MatDestroy(&Bcheck);CHKERRQ(ierr);
  }
  ierr = MatDestroy(&B);CHKERRQ(ierr);

  if (B2) { /* test MatZeroRowsColumns */
    ierr = MatDuplicate(Afull,MAT_COPY_VALUES,&B);CHKERRQ(ierr);
    ierr = MatSetOption(B,MAT_NEW_NONZERO_ALLOCATION_ERR,PETSC_FALSE);CHKERRQ(ierr);
    ierr = MatZeroRowsColumnsIS(B,is,diag,NULL,NULL);CHKERRQ(ierr);
    ierr = CheckMat(B2,B,PETSC_FALSE,"MatZeroRowsColumns");CHKERRQ(ierr);
    ierr = MatDestroy(&B);CHKERRQ(ierr);
    ierr = MatDestroy(&B2);CHKERRQ(ierr);
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

TEST*/
