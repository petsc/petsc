/*$Id: ex1.c,v 1.79 2000/01/11 21:02:20 bsmith Exp $*/

/* Program usage:  mpirun ex1 [-help] [all PETSc options] */

static char help[] = "Solves a tridiagonal linear system with SLES.\n\n";

/*T
   Concepts: SLES^Solving a system of linear equations (basic uniprocessor example);
   Routines: SLESCreate(); SLESSetOperators(); SLESSetFromOptions();
   Routines: SLESSolve(); SLESView(); SLESGetKSP(); SLESGetPC();
   Routines: KSPSetTolerances(); PCSetType();
   Processors: 1
T*/

/* 
  Include "sles.h" so that we can use SLES solvers.  Note that this file
  automatically includes:
     petsc.h  - base PETSc routines   vec.h - vectors
     sys.h    - system routines       mat.h - matrices
     is.h     - index sets            ksp.h - Krylov subspace methods
     viewer.h - viewers               pc.h  - preconditioners
*/
#include "petscsles.h"
#include "sbaij.h"

#undef __FUNC__
#define __FUNC__ "main"
int main(int argc,char **args)
{
  Vec     x,b,u,Ab;      /* approx solution, RHS, exact solution */
  Mat     A,C;         /* linear system matrix */ 
  Mat     sA,sC;           /* symmetric part of A */ 
  SLES    sles;         /* linear solver context */
  PC      pc;           /* preconditioner context */
  KSP     ksp;          /* Krylov subspace method context */
  double  norm;         /* norm of solution error */
  int     n = 20, prob=1;
  Scalar  neg_one = -1.0,one = 1.0,four=4.0,value[3],alpha=0.1; 
  int     ierr,i, col[3],its,size,bs, nz, block, row,I,J,j,n1,*ip_ptr;
  IS      ip, isrow, iscol;
  PetscTruth flg;
  Mat_SeqSBAIJ     *c; 
  MatInfo    minfo;
  MatILUInfo info;
  
  int        lf = 0; /* level of fill for ilu */
  Scalar   *vr;
  int      ncols, *cols,rstart,rend,nrows,mbs;
  double   normf,norm1,normi;
  PetscRandom r; 

  PetscInitialize(&argc,&args,(char *)0,help);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRA(ierr);
  if (size != 1) SETERRA(1,0,"This is a uniprocessor example only!");
  ierr = OptionsGetInt(PETSC_NULL,"-n",&n,PETSC_NULL);CHKERRA(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
         Compute the matrix and right-hand-side vector that define
         the linear system, Ax = b.
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /* 
     Create matrix.  When using MatCreate(), the matrix format can
     be specified at runtime.

     Performance tuning note:  For problems of substantial size,
     preallocation of matrix memory is crucial for attaining good 
     performance.  Since preallocation is not possible via the generic
     matrix creation routine MatCreate(), we recommend for practical 
     problems instead to use the creation routine for a particular matrix
     format, e.g.,
         MatCreateSeqAIJ() - sequential AIJ (compressed sparse row)
         MatCreateSeqBAIJ() - block AIJ
     See the matrix chapter of the users manual for details.
  */
  bs=1; nz=3; mbs = n/bs;
  ierr=MatCreateSeqBAIJ(PETSC_COMM_WORLD,bs,n,n,nz,PETSC_NULL, &A); CHKERRA(ierr);
  ierr=MatCreateSeqSBAIJ(PETSC_COMM_WORLD,bs,n,n,nz,PETSC_NULL, &sA); CHKERRA(ierr);
  /*
  ierr = MatGetOwnershipRange(sA,&i,&j);CHKERRA(ierr);
  printf("rstart=%d, rend=%d\n", i,j);
  */

  /* 
     Assemble matrix
  */
  if (bs == 1){
    if (prob == 1){ /* tridiagonal matrix */
      value[0] = -1.0; value[1] = 2.0; value[2] = -1.0;
      for (i=1; i<n-1; i++) {
        col[0] = i-1; col[1] = i; col[2] = i+1;
        PetscTrValid(0,0,0,0);
        ierr = MatSetValues_SeqBAIJ(A,1,&i,3,col,value,INSERT_VALUES);
        CHKERRA(ierr);
        ierr = MatSetValues_SeqSBAIJ(sA,1,&i,3,col,value,INSERT_VALUES);
        CHKERRA(ierr);
      }
      i = n - 1; col[0]=0; col[1] = n - 2; col[2] = n - 1;
      value[0]= 0.1; value[1]=-1; value[2]=2;
      ierr = MatSetValues_SeqBAIJ(A,1,&i,3,col,value,INSERT_VALUES);
      CHKERRA(ierr);
      ierr = MatSetValues(sA,1,&i,3,col,value,INSERT_VALUES);
      CHKERRA(ierr);

      i = 0; col[0] = 0; col[1] = 1; col[2]=n-1;
      value[0] = 2.0; value[1] = -1.0; value[2]=0.1;
      ierr = MatSetValues(A,1,&i,3,col,value,INSERT_VALUES);
      CHKERRA(ierr);
      ierr = MatSetValues_SeqSBAIJ(sA,1,&i,3,col,value,INSERT_VALUES);
      CHKERRA(ierr);
    }
    else if (prob ==2){ /* matrix for the five point stencil */
      n1 = sqrt(n); printf("n1 = %d\n",n1);
      for (i=0; i<n1; i++) {
        for (j=0; j<n1; j++) {
          I = j + n1*i;
          if (i>0)   {J = I - n1; ierr = MatSetValues(sA,1,&I,1,&J,&neg_one,INSERT_VALUES);CHKERRA(ierr);}
          if (i<n1-1) {J = I + n1; ierr = MatSetValues(sA,1,&I,1,&J,&neg_one,INSERT_VALUES);CHKERRA(ierr);}
          if (j>0)   {J = I - 1; ierr = MatSetValues(sA,1,&I,1,&J,&neg_one,INSERT_VALUES);CHKERRA(ierr);}
          if (j<n1-1) {J = I + 1; ierr = MatSetValues(sA,1,&I,1,&J,&neg_one,INSERT_VALUES);CHKERRA(ierr);}
          ierr = MatSetValues(sA,1,&I,1,&I,&four,INSERT_VALUES);CHKERRA(ierr);
        }
      }                   
    }
  } /* end of if (bs == 1) */
  else {
  for (block=0; block<n/bs; block++){
    /* diagonal blocks */
    value[0] = -1.0; value[1] = 4.0; value[2] = -1.0;
    for (i=1+block*bs; i<bs-1+block*bs; i++) {
      col[0] = i-1; col[1] = i; col[2] = i+1;
      ierr = MatSetValues(A,1,&i,3,col,value,INSERT_VALUES); CHKERRA(ierr);
      ierr = MatSetValues(sA,1,&i,3,col,value,INSERT_VALUES); CHKERRA(ierr);    
    }
    i = bs - 1+block*bs; col[0] = bs - 2+block*bs; col[1] = bs - 1+block*bs;
    value[0]=-1.0; value[1]=4.0;  
    ierr = MatSetValues(A,1,&i,2,col,value,INSERT_VALUES); CHKERRA(ierr);
    ierr = MatSetValues(sA,1,&i,2,col,value,INSERT_VALUES); CHKERRA(ierr); 

    i = 0+block*bs; col[0] = 0+block*bs; col[1] = 1+block*bs; 
    value[0]=4.0; value[1] = -1.0; 
    ierr = MatSetValues(A,1,&i,2,col,value,INSERT_VALUES); CHKERRA(ierr);
    ierr = MatSetValues(sA,1,&i,2,col,value,INSERT_VALUES); CHKERRA(ierr);  
  }
  /* off-diagonal blocks */
  value[0]=-1.0;
  for (i=0; i<(n/bs-1)*bs; i++){
    col[0]=i+bs;
    ierr = MatSetValues(A,1,&i,1,col,value,INSERT_VALUES); CHKERRA(ierr);
    ierr = MatSetValues(sA,1,&i,1,col,value,INSERT_VALUES); CHKERRA(ierr);
    col[0]=i; row=i+bs;
    ierr = MatSetValues(A,1,&row,1,col,value,INSERT_VALUES); CHKERRA(ierr);
    ierr = MatSetValues(sA,1,&row,1,col,value,INSERT_VALUES); CHKERRA(ierr);
  }
  }
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRA(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRA(ierr);
  /* printf("\n The Matrix: \n");
  MatView(A, VIEWER_DRAW_WORLD);
  MatView(A, VIEWER_STDOUT_WORLD); */ 

  ierr = MatAssemblyBegin(sA,MAT_FINAL_ASSEMBLY); CHKERRA(ierr);
  ierr = MatAssemblyEnd(sA,MAT_FINAL_ASSEMBLY); CHKERRA(ierr);  
  /*
  printf("\n Symmetric Part of Matrix: \n");
  MatView(sA, VIEWER_DRAW_WORLD); 
  MatView(sA, VIEWER_STDOUT_WORLD); 
  */
#ifdef MatNorm
  ierr = MatNorm(A,NORM_FROBENIUS,&normf);CHKERRA(ierr); 
  /* ierr = MatNorm(A,NORM_1,&norm1);        CHKERRA(ierr); */
  ierr = MatNorm(A,NORM_INFINITY,&normi); CHKERRA(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"A: Frob. norm = %g, infinity norm = %g\n", normf,normi);CHKERRA(ierr); 

  ierr = MatNorm(sA,NORM_FROBENIUS,&normf);CHKERRA(ierr); 
  /* ierr = MatNorm(sA,NORM_1,&norm1);     CHKERRA(ierr); */
  ierr = MatNorm(sA,NORM_INFINITY,&normi); CHKERRA(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"sA: Frob. norm = %g, infinity norm = %g\n", normf,normi);CHKERRA(ierr);
#endif
  /*
  ierr = MatGetInfo_SeqSBAIJ(sA,MAT_LOCAL,&minfo);CHKERRA(ierr);
  printf("matrix nonzeros = %d, allocated nonzeros= %d\n", (int)minfo.nz_used,(int)minfo.nz_allocated); 
  ierr = MatGetSize(A,&i,&j);CHKERRA(ierr); 
  printf("size of sA = %d, %d\n",i,j);
  */

  /*
  ierr = MatGetBlockSize(sA, &bs); CHKERRA(ierr);
  printf("bs= %d\n", bs);
  */

  /*
  row = 3*bs-1; 
  ierr = MatGetRow(sA,row,&ncols,&cols,&vr); CHKERRA(ierr); 
  printf("row=%d\n", row);
  for (i=0; i<ncols; i++) printf("cols[%d] = %d, %g\n",i,*cols++,*vr++);
  cols -= ncols; vr -= ncols; 
  ierr = MatRestoreRow(sA,row,&ncols,&cols,&vr); CHKERRA(ierr); 
  */

  /*
  printf("Diagonal of sA:\n");
  ierr = MatGetDiagonal(sA, Ab); CHKERRA(ierr);
  ierr = VecView(Ab, VIEWER_STDOUT_WORLD); CHKERRA(ierr);
  */
  /*
  printf("sA =\n"); MatView(sA, VIEWER_STDOUT_WORLD);
  ierr = MatScale(&alpha,sA);CHKERRA(ierr); 
  printf("After MatScale by alpha, sA =\n"); MatView(sA, VIEWER_STDOUT_WORLD);
  */

  /*
  ierr = MatGetOwnershipRange(sA,&rstart,&rend);CHKERRA(ierr);
  printf("rstart = %d, rend = %d\n",rstart,rend);
  */
#ifdef GetSubMatrix
  /*Get a submatrix consisting of every next block row and column of the original matrix*/

  /* list all the rows we want on THIS processor. For symm. matrix, iscol=isrow. */
  ip_ptr = (int*)PetscMalloc(n*sizeof(int)); CHKERRA(ierr);
  j = 0;
  for (n1=0; n1<mbs; n1 += 2){ /* n1: block row */
    for (i=0; i<bs; i++) ip_ptr[j++] = n1*bs + i;  
  }
  ierr = ISCreateGeneral(PETSC_COMM_SELF, j, ip_ptr, &isrow); CHKERRA(ierr);
  ierr = PetscFree(ip_ptr); CHKERRA(ierr); 
  ISView(isrow, VIEWER_STDOUT_SELF); CHKERRA(ierr);
  
  ip_ptr = (int*)PetscMalloc(n*sizeof(int)); CHKERRA(ierr);
  j = 0;
  for (n1=1; n1<mbs; n1 += 2){ 
    for (i=0; i<bs; i++) ip_ptr[j++] = n1*bs + i;  
  }
  ierr = ISCreateGeneral(PETSC_COMM_SELF, j, ip_ptr, &iscol); CHKERRA(ierr);
  ierr = PetscFree(ip_ptr); CHKERRA(ierr); 
  /* ISView(iscol, VIEWER_STDOUT_SELF); CHKERRA(ierr); */
  
  ierr = MatGetSubMatrix(sA,isrow,isrow,PETSC_DECIDE,MAT_INITIAL_MATRIX,&sC);
  CHKERRA(ierr);
  ierr = ISDestroy(isrow);CHKERRA(ierr);
  ierr = ISDestroy(iscol);CHKERRA(ierr);
  printf("sA =\n");
  ierr = MatView(sA,VIEWER_STDOUT_WORLD);CHKERRA(ierr);
  printf("sC =\n");
  ierr = MatView(sC,VIEWER_STDOUT_WORLD);CHKERRA(ierr);  
#endif
  /* 
     Create vectors.  Note that we form 1 vector from scratch and
     then duplicate as needed.
  */
  ierr = VecCreate(PETSC_COMM_WORLD,PETSC_DECIDE,n,&x);CHKERRA(ierr);
  ierr = VecSetFromOptions(x);CHKERRA(ierr);
  ierr = VecDuplicate(x,&b);CHKERRA(ierr);
  ierr = VecDuplicate(x,&u);CHKERRA(ierr);
  ierr = VecDuplicate(x,&Ab);CHKERRA(ierr); 

  /* 
     Set exact solution u; then compute right-hand-side vector b.
  */
  ierr = PetscRandomCreate(PETSC_COMM_SELF,RANDOM_DEFAULT,&r);CHKERRA(ierr);
  ierr = VecSetRandom(r,u);CHKERRA(ierr);
  ierr = PetscRandomDestroy(r); CHKERRA(ierr);

  ierr = VecSet(&one,x);CHKERRA(ierr);

  /* Test MatMult, MatMultAdd, MatDiagonalScale */
 
#ifdef MatDiagonalScale
  ierr = MatDiagonalScale(A,u,u);CHKERRA(ierr);
  ierr = MatDiagonalScale(sA,u,u);CHKERRA(ierr); 
  /* MatView(A, VIEWER_STDOUT_WORLD); */
#endif
  
#ifndef MatMult
  ierr = MatMult(A,u,Ab);CHKERRA(ierr);
  /* printf("Ab=A*x:\n");
  ierr = VecView(Ab, VIEWER_STDOUT_WORLD); CHKERRA(ierr); */
  ierr = MatMult(sA,u,b);CHKERRA(ierr);
  /* printf("Symm. b=sA*u:\n");
  ierr = VecView(b, VIEWER_STDOUT_WORLD); CHKERRA(ierr); */
#endif

#ifdef MatMultAdd 
  ierr = MatMultAdd(A,u,x,Ab); CHKERRA(ierr);   /* Ab = x +  A*u */
  ierr = MatMultAdd(sA,u,x,b); CHKERRA(ierr);   /* b  = x + sA*u */
#endif

  ierr = VecAXPY(&neg_one,b,Ab);CHKERRA(ierr);
  ierr  = VecNorm(Ab,NORM_2,&norm);CHKERRA(ierr);
  if (norm > n*1.e-15){ 
    printf("MatMult, MatMultAdd, or MatDiagonalScale failed, norm of error: %e \n", norm);
  } 

  ierr = MatGetOrdering(A,MATORDERING_NATURAL,&isrow,&iscol);CHKERRA(ierr); 
  ip = isrow;

#ifdef ReOrder
  ISGetIndices(ip,&ip_ptr);
  i = ip_ptr[1]; ip_ptr[1] = ip_ptr[n-2]; ip_ptr[n-2] = i; 
  i = ip_ptr[0]; ip_ptr[0] = ip_ptr[n-1]; ip_ptr[n-1] = i; 
  ISRestoreIndices(ip,&ip_ptr);
  ierr = MatReordering_SeqSBAIJ(sA, ip); CHKERRA(ierr);  
  /* ierr = ISView(ip, VIEWER_STDOUT_SELF); CHKERRA(ierr); */
#endif

#ifdef Factor
  /* compute factorization */
  /*
  ierr = MatLUFactorSymbolic(A,ip,ip,2.0,&C);CHKERRA(ierr);
  ierr = MatLUFactorNumeric(A,&C);CHKERRA(ierr); 
  ierr = MatSolve(C,b,x);CHKERRA(ierr);    
  */
  /* ierr = MatLUFactorSymbolic(sA,ip,ip,1.5,&sC);CHKERRA(ierr); */

  ierr = OptionsHasName(PETSC_NULL,"-lu",&flg);CHKERRA(ierr);
  if (flg){ 
    printf("flg");
    ierr = MatLUFactorSymbolic(sA,ip,ip,1.0,&sC);CHKERRA(ierr);
  } else {
  info.levels        = lf;
  info.fill          = 1.0;
  info.diagonal_fill = 0;
  ierr = MatILUFactorSymbolic(sA,ip,ip,&info,&sC);CHKERRA(ierr);
  }
  ierr = MatLUFactorNumeric_SeqSBAIJ_1(sA,&sC);CHKERRA(ierr);
  /* MatView(sC, VIEWER_STDOUT_WORLD); */
  MatView(sC, VIEWER_DRAW_WORLD); 
  /* ierr = MatSolve_SeqSBAIJ_1_NaturalOrdering(sC,b,x);CHKERRA(ierr); */
  ierr = MatSolve_SeqSBAIJ_1(sC,b,x);CHKERRA(ierr);
  /*
  printf("C->m = %d\n",sC->m);  
  c = sC->data; 
  printf("c->a = \n");
  for (i=0; i<n; i++) printf("ca[%d]=%f\n", i, c->a[i]); 
  */
  /*
  printf("x= \n");
  ierr = VecView(x,VIEWER_STDOUT_WORLD); CHKERRA(ierr); 
  */
#endif

#ifndef SLES
  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
                Create the linear solver and set various options
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  /* 
     Create linear solver context
  */
  ierr = SLESCreate(PETSC_COMM_WORLD,&sles);CHKERRA(ierr);

  /* 
     Set operators. Here the matrix that defines the linear system
     also serves as the preconditioning matrix.
  */
  ierr = SLESSetOperators(sles,sA,sA,DIFFERENT_NONZERO_PATTERN);CHKERRA(ierr); 

  /* 
     Set linear solver defaults for this problem (optional).
     - By extracting the KSP and PC contexts from the SLES context,
       we can then directly call any KSP and PC routines to set
       various options.
     - The following four statements are optional; all of these
       parameters could alternatively be specified at runtime via
       SLESSetFromOptions();
  */
  ierr = SLESGetKSP(sles,&ksp);CHKERRA(ierr);
  ierr = SLESGetPC(sles,&pc);CHKERRA(ierr);
  ierr = PCSetType(pc,PCJACOBI);CHKERRA(ierr);
  ierr = KSPSetTolerances(ksp,1.e-7,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT);CHKERRA(ierr);

  /* 
    Set runtime options, e.g.,
        -ksp_type <type> -pc_type <type> -ksp_monitor -ksp_rtol <rtol>
    These options will override those specified above as long as
    SLESSetFromOptions() is called _after_ any other customization
    routines.
  */
  ierr = SLESSetFromOptions(sles);CHKERRA(ierr);
 
  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
                      Solve the linear system
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  /* 
     Solve linear system
  */
  ierr = SLESSolve(sles,b,x,&its);CHKERRA(ierr); 

  /* 
     View solver info; we could instead use the option -sles_view to
     print this info to the screen at the conclusion of SLESSolve().
  */
  ierr = SLESView(sles,VIEWER_STDOUT_WORLD);CHKERRA(ierr);

#endif
  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
                      Check solution and clean up
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  /* 
     Check the error
  */
  ierr = VecAXPY(&neg_one,u,x);CHKERRA(ierr);
  ierr  = VecNorm(x,NORM_2,&norm);CHKERRA(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Norm of error %A, Iterations %d\n",norm,its);CHKERRA(ierr);
  /* 
     Free work space.  All PETSc objects should be destroyed when they
     are no longer needed.
  */
  ierr = VecDestroy(x);CHKERRA(ierr); ierr = VecDestroy(u);CHKERRA(ierr);
  ierr = VecDestroy(b);CHKERRA(ierr); ierr = MatDestroy(A);CHKERRA(ierr);
  /* ierr = SLESDestroy(sles);CHKERRA(ierr); */

  /*
     Always call PetscFinalize() before exiting a program.  This routine
       - finalizes the PETSc libraries as well as MPI
       - provides summary and diagnostic information if certain runtime
         options are chosen (e.g., -log_summary).
  */


  PetscFinalize();
  return 0;
}
