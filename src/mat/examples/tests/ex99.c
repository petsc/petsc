static char help[] = "Test LAPACK routine DSYGV() or DSYGVX(). \n\
Reads PETSc matrix A and B (or create B=I), \n\
then computes selected eigenvalues, and optionally, eigenvectors of \n\
a real generalized symmetric-definite eigenproblem \n\
 A*x = lambda*B*x \n\
Input parameters include\n\
  -f0 <input_file> : first file to load (small system)\n\
  -fA <input_file> -fB <input_file>: second files to load (larger system) \n\
e.g. ./ex99 -f0 $D/small -fA $D/Eigdftb/dftb_bin/diamond_xxs_A -fB $D/Eigdftb/dftb_bin/diamond_xxs_B -mat_getrow_uppertriangular,\n\
     where $D = /home/petsc/datafiles/matrices/Eigdftb/dftb_bin\n\n";

#include <petscmat.h>
#include <../src/mat/impls/sbaij/seq/sbaij.h>
#include <petscblaslapack.h>

extern PetscErrorCode CkEigenSolutions(PetscInt*,Mat*,PetscReal*,Vec*,PetscInt*,PetscInt*,PetscReal*);

#undef __FUNCT__
#define __FUNCT__ "main"
PetscInt main(PetscInt argc,char **args)
{
  Mat            A,B,A_dense,B_dense,mats[2],A_sp;    
  Vec            *evecs;
  PetscViewer    fd;                /* viewer */
  char           file[3][PETSC_MAX_PATH_LEN];     /* input file name */
  PetscBool      flg,flgA=PETSC_FALSE,flgB=PETSC_FALSE,TestSYGVX=PETSC_TRUE;
  PetscErrorCode ierr;
  PetscBool      preload=PETSC_TRUE,isSymmetric;
  PetscScalar    sigma,one=1.0,*arrayA,*arrayB,*evecs_array,*work,*evals;
  PetscMPIInt    size;
  PetscInt       m,n,i,j,nevs,il,iu;
  PetscLogStage  stages[2];
  PetscReal      vl,vu,abstol=1.e-8; 
  PetscBLASInt   *iwork,*ifail,lone=1,lwork,lierr,bn;
  PetscInt       ievbd_loc[2],offset=0,cklvl=2;
  PetscReal      tols[2];
  Mat_SeqSBAIJ   *sbaij;
  PetscScalar    *aa;
  PetscInt       *ai,*aj;
  PetscInt       nzeros[2],nz;
  PetscReal      ratio;
  
  PetscInitialize(&argc,&args,(char *)0,help);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
  if (size != 1) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"This is a uniprocessor example only!");
  ierr = PetscLogStageRegister("EigSolve",&stages[0]);
  ierr = PetscLogStageRegister("EigCheck",&stages[1]);

  /* Determine files from which we read the two matrices */
  ierr = PetscOptionsGetString(PETSC_NULL,"-f0",file[0],PETSC_MAX_PATH_LEN,&flg);CHKERRQ(ierr);
  if (!flg) {
    ierr = PetscOptionsGetString(PETSC_NULL,"-fA",file[0],PETSC_MAX_PATH_LEN,&flgA);CHKERRQ(ierr);
    if (!flgA) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"Must indicate binary file with the -fA or -fB options");
    ierr = PetscOptionsGetString(PETSC_NULL,"-fB",file[1],PETSC_MAX_PATH_LEN,&flgB);CHKERRQ(ierr);
    preload = PETSC_FALSE;
  } else {
    ierr = PetscOptionsGetString(PETSC_NULL,"-fA",file[1],PETSC_MAX_PATH_LEN,&flgA);CHKERRQ(ierr);
    if (!flgA) {preload = PETSC_FALSE;} /* don't bother with second system */
    ierr = PetscOptionsGetString(PETSC_NULL,"-fB",file[2],PETSC_MAX_PATH_LEN,&flgB);CHKERRQ(ierr);
  }

  PetscPreLoadBegin(preload,"Load system");
    /* Load matrices */
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,file[PetscPreLoadIt],FILE_MODE_READ,&fd);CHKERRQ(ierr);
    ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
    ierr = MatSetType(A,MATSBAIJ);CHKERRQ(ierr);
    ierr = MatLoad(A,fd);CHKERRQ(ierr); 
    ierr = PetscViewerDestroy(&fd);CHKERRQ(ierr); 
    ierr = MatGetSize(A,&m,&n);CHKERRQ(ierr);
    if ((flgB && PetscPreLoadIt) || (flgB && !preload)){
      ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,file[PetscPreLoadIt+1],FILE_MODE_READ,&fd);CHKERRQ(ierr);
      ierr = MatCreate(PETSC_COMM_WORLD,&B);CHKERRQ(ierr);
      ierr = MatSetType(B,MATSBAIJ);CHKERRQ(ierr);
      ierr = MatLoad(B,fd);CHKERRQ(ierr);
      ierr = PetscViewerDestroy(&fd);CHKERRQ(ierr);
    } else { /* create B=I */
      ierr = MatCreate(PETSC_COMM_WORLD,&B);CHKERRQ(ierr);
      ierr = MatSetSizes(B,PETSC_DECIDE,PETSC_DECIDE,m,n);CHKERRQ(ierr);
      ierr = MatSetType(B,MATSEQSBAIJ);CHKERRQ(ierr);
      ierr = MatSetFromOptions(B);CHKERRQ(ierr);
      for (i=0; i<m; i++) {
        ierr = MatSetValues(B,1,&i,1,&i,&one,INSERT_VALUES);CHKERRQ(ierr);
      }
      ierr = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
      ierr = MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    }
    
    /* Add a shift to A */
    ierr = PetscOptionsGetScalar(PETSC_NULL,"-mat_sigma",&sigma,&flg);CHKERRQ(ierr);
    if(flg) {
      ierr = MatAXPY(A,sigma,B,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr); /* A <- sigma*B + A */  
    }

    /* Check whether A is symmetric */
    ierr = PetscOptionsHasName(PETSC_NULL, "-check_symmetry", &flg);CHKERRQ(ierr);
    if (flg) {
      Mat Trans;
      ierr = MatTranspose(A,MAT_INITIAL_MATRIX, &Trans);
      ierr = MatEqual(A, Trans, &isSymmetric);
      if (!isSymmetric) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"A must be symmetric");
      ierr = MatDestroy(&Trans);CHKERRQ(ierr);
      if (flgB && PetscPreLoadIt){
        ierr = MatTranspose(B,MAT_INITIAL_MATRIX, &Trans);
        ierr = MatEqual(B, Trans, &isSymmetric);
        if (!isSymmetric) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"B must be symmetric");
        ierr = MatDestroy(&Trans);CHKERRQ(ierr);
      }
    }

    /* View small entries of A */
    ierr = PetscOptionsHasName(PETSC_NULL, "-Asp_view", &flg);CHKERRQ(ierr);
    if (flg){
      ierr = MatCreate(PETSC_COMM_SELF,&A_sp);CHKERRQ(ierr);
      ierr = MatSetSizes(A_sp,PETSC_DECIDE,PETSC_DECIDE,m,n);CHKERRQ(ierr);
      ierr = MatSetType(A_sp,MATSEQSBAIJ);CHKERRQ(ierr);

      tols[0] = 1.e-6, tols[1] = 1.e-9;
      sbaij = (Mat_SeqSBAIJ*)A->data;
      ai    = sbaij->i; 
      aj    = sbaij->j;
      aa    = sbaij->a;
      nzeros[0] = nzeros[1] = 0; 
      for (i=0; i<m; i++) {
        nz = ai[i+1] - ai[i];
        for (j=0; j<nz; j++){
          if (PetscAbsScalar(*aa)<tols[0]) {
            ierr = MatSetValues(A_sp,1,&i,1,aj,aa,INSERT_VALUES);CHKERRQ(ierr);
            nzeros[0]++;
          }
          if (PetscAbsScalar(*aa)<tols[1]) nzeros[1]++;
          aa++; aj++;
        }
      }
      ierr = MatAssemblyBegin(A_sp,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
      ierr = MatAssemblyEnd(A_sp,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr); 

      ierr = MatDestroy(&A_sp);CHKERRQ(ierr);

      ratio = (PetscReal)nzeros[0]/sbaij->nz;
      ierr = PetscPrintf(PETSC_COMM_SELF," %d matrix entries < %e, ratio %G of %d nonzeros\n",nzeros[0],tols[0],ratio,sbaij->nz);CHKERRQ(ierr);
      ierr = PetscPrintf(PETSC_COMM_SELF," %d matrix entries < %e\n",nzeros[1],tols[1]);CHKERRQ(ierr);
    }

    /* Convert aij matrix to MatSeqDense for LAPACK */
    ierr = PetscObjectTypeCompare((PetscObject)A,MATSEQDENSE,&flg);CHKERRQ(ierr);
    if (!flg) {
      ierr = MatConvert(A,MATSEQDENSE,MAT_INITIAL_MATRIX,&A_dense);CHKERRQ(ierr); 
    }
    ierr = PetscObjectTypeCompare((PetscObject)B,MATSEQDENSE,&flg);CHKERRQ(ierr);
    if (!flg) {ierr = MatConvert(B,MATSEQDENSE,MAT_INITIAL_MATRIX,&B_dense);CHKERRQ(ierr);}

    /* Solve eigenvalue problem: A*x = lambda*B*x */
    /*============================================*/
    lwork = PetscBLASIntCast(8*n);
    bn    = PetscBLASIntCast(n);
    ierr = PetscMalloc(n*sizeof(PetscScalar),&evals);CHKERRQ(ierr);
    ierr = PetscMalloc(lwork*sizeof(PetscScalar),&work);CHKERRQ(ierr);
    ierr = MatGetArray(A_dense,&arrayA);CHKERRQ(ierr);
    ierr = MatGetArray(B_dense,&arrayB);CHKERRQ(ierr);

    if (!TestSYGVX){ /* test sygv()  */
      evecs_array = arrayA;
      LAPACKsygv_(&lone,"V","U",&bn,arrayA,&bn,arrayB,&bn,evals,work,&lwork,&lierr); 
      nevs = m;
      il=1; 
    } else { /* test sygvx()  */
      il = 1; iu=PetscBLASIntCast(.6*m); /* request 1 to 60%m evalues */
      ierr = PetscMalloc((m*n+1)*sizeof(PetscScalar),&evecs_array);CHKERRQ(ierr);
      ierr = PetscMalloc((6*n+1)*sizeof(PetscBLASInt),&iwork);CHKERRQ(ierr);
      ifail = iwork + 5*n;
      if(PetscPreLoadIt){ierr = PetscLogStagePush(stages[0]);CHKERRQ(ierr);}
      /* in the case "I", vl and vu are not referenced */
      LAPACKsygvx_(&lone,"V","I","U",&bn,arrayA,&bn,arrayB,&bn,&vl,&vu,&il,&iu,&abstol,&nevs,evals,evecs_array,&n,work,&lwork,iwork,ifail,&lierr);
      if(PetscPreLoadIt){ierr = PetscLogStagePop();}
      ierr = PetscFree(iwork);CHKERRQ(ierr);
    }
    ierr = MatRestoreArray(A,&arrayA);CHKERRQ(ierr);
    ierr = MatRestoreArray(B,&arrayB);CHKERRQ(ierr);

    if (nevs <= 0 ) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_CONV_FAILED, "nev=%d, no eigensolution has found", nevs);
    /* View evals */
    ierr = PetscOptionsHasName(PETSC_NULL, "-eig_view", &flg);CHKERRQ(ierr);
    if (flg){
      printf(" %d evals: \n",nevs);
      for (i=0; i<nevs; i++) printf("%d  %G\n",i+il,evals[i]); 
    }

    /* Check residuals and orthogonality */
    if(PetscPreLoadIt){
      mats[0] = A; mats[1] = B;
      one = (PetscInt)one;
      ierr = PetscMalloc((nevs+1)*sizeof(Vec),&evecs);CHKERRQ(ierr);
      for (i=0; i<nevs; i++){
        ierr = VecCreate(PETSC_COMM_SELF,&evecs[i]);CHKERRQ(ierr);
        ierr = VecSetSizes(evecs[i],PETSC_DECIDE,n);CHKERRQ(ierr);
        ierr = VecSetFromOptions(evecs[i]);CHKERRQ(ierr);
        ierr = VecPlaceArray(evecs[i],evecs_array+i*n);CHKERRQ(ierr);
      }
    
      ievbd_loc[0] = 0; ievbd_loc[1] = nevs-1;
      tols[0] = 1.e-8;  tols[1] = 1.e-8;
      ierr = PetscLogStagePush(stages[1]);CHKERRQ(ierr);
      ierr = CkEigenSolutions(&cklvl,mats,evals,evecs,ievbd_loc,&offset,tols);CHKERRQ(ierr);
      ierr = PetscLogStagePop();CHKERRQ(ierr);
      for (i=0; i<nevs; i++){ ierr = VecDestroy(&evecs[i]);CHKERRQ(ierr);}
      ierr = PetscFree(evecs);CHKERRQ(ierr);
    }
    
    /* Free work space. */
    if (TestSYGVX){ierr = PetscFree(evecs_array);CHKERRQ(ierr);}
    
    ierr = PetscFree(evals);CHKERRQ(ierr);
    ierr = PetscFree(work);CHKERRQ(ierr);

    ierr = MatDestroy(&A_dense);CHKERRQ(ierr); 
    ierr = MatDestroy(&B_dense);CHKERRQ(ierr); 
    ierr = MatDestroy(&B);CHKERRQ(ierr);
    ierr = MatDestroy(&A);CHKERRQ(ierr);

  PetscPreLoadEnd();
  ierr = PetscFinalize();
  return 0;
}
/*------------------------------------------------
  Check the accuracy of the eigen solution
  ----------------------------------------------- */
/*
  input: 
     cklvl      - check level: 
                    1: check residual
                    2: 1 and check B-orthogonality locally 
     mats       - matrix pencil
     eval, evec - eigenvalues and eigenvectors stored in this process
     ievbd_loc  - local eigenvalue bounds, see eigc()
     offset     - see eigc()
     tols[0]    - reporting tol_res: || A evec[i] - eval[i] B evec[i]||
     tols[1]    - reporting tol_orth: evec[i] B evec[j] - delta_ij
*/
#undef DEBUG_CkEigenSolutions
#undef __FUNCT__
#define __FUNCT__ "CkEigenSolutions"
PetscErrorCode CkEigenSolutions(PetscInt *fcklvl,Mat *mats,
                   PetscReal *eval,Vec *evec,PetscInt *ievbd_loc,PetscInt *offset, 
                   PetscReal *tols)
{
  PetscInt     ierr,cklvl=*fcklvl,nev_loc,i,j;
  Mat          A=mats[0], B=mats[1];
  Vec          vt1,vt2; /* tmp vectors */
  PetscReal    norm,tmp,dot,norm_max,dot_max;  

  PetscFunctionBegin;
  nev_loc = ievbd_loc[1] - ievbd_loc[0];
  if (nev_loc == 0) PetscFunctionReturn(0);

  nev_loc += (*offset);
  ierr = VecDuplicate(evec[*offset],&vt1);
  ierr = VecDuplicate(evec[*offset],&vt2);

  switch (cklvl){
  case 2:  
    dot_max = 0.0;
    for (i = *offset; i<nev_loc; i++){
      ierr = MatMult(B, evec[i], vt1);
      for (j=i; j<nev_loc; j++){ 
        ierr = VecDot(evec[j],vt1,&dot);
        if (j == i){
          dot = PetscAbsScalar(dot - 1.0);
        } else {
          dot = PetscAbsScalar(dot);
        }
        if (dot > dot_max) dot_max = dot;
#ifdef DEBUG_CkEigenSolutions
        if (dot > tols[1] ) {
          ierr = VecNorm(evec[i],NORM_INFINITY,&norm);
          ierr = PetscPrintf(PETSC_COMM_SELF,"|delta(%d,%d)|: %G, norm: %G\n",i,j,dot,norm);
        } 
#endif
      } /* for (j=i; j<nev_loc; j++) */
    } 
    ierr = PetscPrintf(PETSC_COMM_SELF,"    max|(x_j*B*x_i) - delta_ji|: %G\n",dot_max);

  case 1: 
    norm_max = 0.0;
    for (i = *offset; i< nev_loc; i++){
      ierr = MatMult(A, evec[i], vt1);
      ierr = MatMult(B, evec[i], vt2);
      tmp  = -eval[i];
      ierr = VecAXPY(vt1,tmp,vt2);
      ierr = VecNorm(vt1, NORM_INFINITY, &norm);
      norm = PetscAbsScalar(norm); 
      if (norm > norm_max) norm_max = norm;
#ifdef DEBUG_CkEigenSolutions
      /* sniff, and bark if necessary */
      if (norm > tols[0]){
        printf( "  residual violation: %d, resi: %g\n",i, norm);
      }
#endif
    }
    
      ierr = PetscPrintf(PETSC_COMM_SELF,"    max_resi:                    %G\n", norm_max);
    
   break;
  default:
    ierr = PetscPrintf(PETSC_COMM_SELF,"Error: cklvl=%d is not supported \n",cklvl);
  }
  ierr = VecDestroy(&vt2); 
  ierr = VecDestroy(&vt1);
  PetscFunctionReturn(0);
}
