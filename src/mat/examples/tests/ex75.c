/*$Id: ex75.c,v 1.4 2000/07/10 21:51:15 hzhang Exp hzhang $*/

/* Program usage:  mpirun -np <procs> ex75 [-help] [all PETSc options] */ 

static char help[] = "Tests the vatious routines in MatMPISBAIJ format.\n";

#include "petscmat.h"

#undef __FUNC__
#define __FUNC__ "main"
int main(int argc,char **args)
{
  Vec         x,b,u;    /* approx solution, RHS, exact solution */
  Mat         A,sA;     /* linear system matrix */
  PetscRandom rctx;     /* random number generator context */
  double      norm;     /* norm of solution error */
  int         i,j,i1,i2,j1,j2,I,J,Istart,Iend,ierr,its,m,m1;

  PetscTruth  flg;
  Scalar      v, one=1.0, neg_one=-1.0, value[3], four=4.0,alpha=0.1,*diag;
  int         bs=1, d_nz=3, o_nz=3, n = 16, prob=1;
  int         rank,size,col[3],n1,mbs,block,row;
  int         flg_A = 0, flg_sA = 1;
  int         ncols,*cols,*ip_ptr,rstart,rend,N;
  Scalar      *vr;
  IS          isrow;

  PetscInitialize(&argc,&args,(char *)0,help);
  ierr = OptionsGetInt(PETSC_NULL,"-n",&n,PETSC_NULL);CHKERRA(ierr);
  ierr = OptionsGetInt(PETSC_NULL,"-bs",&bs,PETSC_NULL);CHKERRA(ierr);
  
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRA(ierr);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRA(ierr);
  
  mbs = n/bs;
  if (mbs*bs != n) SETERRQ(PETSC_ERR_ARG_SIZ,0,"n/bs must be an integer.");

  /* Assemble MPISBAIJ matrix sA */
  ierr = MatCreateMPISBAIJ(PETSC_COMM_WORLD,bs,PETSC_DECIDE,PETSC_DECIDE,n,n,d_nz,PETSC_NULL,o_nz,PETSC_NULL,&sA);CHKERRA(ierr);

  if (bs == 1){
    if (prob == 1){ /* tridiagonal matrix */
      value[0] = -1.0; value[1] = 2.0; value[2] = -1.0;
      for (i=1; i<n-1; i++) {
        col[0] = i-1; col[1] = i; col[2] = i+1;
        /* PetscTrValid(0,0,0,0); */
        ierr = MatSetValues(sA,1,&i,3,col,value,INSERT_VALUES); CHKERRA(ierr);
      }
      i = n - 1; col[0]=0; col[1] = n - 2; col[2] = n - 1;
      value[0]= 0.1; value[1]=-1; value[2]=2;
      ierr = MatSetValues(sA,1,&i,3,col,value,INSERT_VALUES); CHKERRA(ierr);

      i = 0; col[0] = 0; col[1] = 1; col[2]=n-1;
      value[0] = 2.0; value[1] = -1.0; value[2]=0.1;
      ierr = MatSetValues(sA,1,&i,3,col,value,INSERT_VALUES); CHKERRA(ierr);
    }
    else if (prob ==2){ /* matrix for the five point stencil */
      n1 = sqrt(n); 
      PetscSynchronizedPrintf(PETSC_COMM_WORLD,"[%d],for sA, n1=%d\n",rank,n1); 
  PetscSynchronizedFlush(PETSC_COMM_WORLD);
      if (n1*n1 != n){
        SETERRQ(PETSC_ERR_ARG_SIZ,0,"n must be a perfect square of n1");
      }
        
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
  else {  /* bs > 1 */
  for (block=0; block<n/bs; block++){
    /* diagonal blocks */
    value[0] = -1.0; value[1] = 4.0; value[2] = -1.0;
    for (i=1+block*bs; i<bs-1+block*bs; i++) {
      col[0] = i-1; col[1] = i; col[2] = i+1;
      ierr = MatSetValues(sA,1,&i,3,col,value,INSERT_VALUES); CHKERRA(ierr);    
    }
    i = bs - 1+block*bs; col[0] = bs - 2+block*bs; col[1] = bs - 1+block*bs;
    value[0]=-1.0; value[1]=4.0;  
    ierr = MatSetValues(sA,1,&i,2,col,value,INSERT_VALUES); CHKERRA(ierr); 

    i = 0+block*bs; col[0] = 0+block*bs; col[1] = 1+block*bs; 
    value[0]=4.0; value[1] = -1.0; 
    ierr = MatSetValues(sA,1,&i,2,col,value,INSERT_VALUES); CHKERRA(ierr);  
  }
  /* off-diagonal blocks */
  value[0]=-1.0;
  for (i=0; i<(n/bs-1)*bs; i++){
    col[0]=i+bs;
    ierr = MatSetValues(sA,1,&i,1,col,value,INSERT_VALUES); CHKERRA(ierr);
    col[0]=i; row=i+bs;
    ierr = MatSetValues(sA,1,&row,1,col,value,INSERT_VALUES); CHKERRA(ierr);
  }
  }
  ierr = MatAssemblyBegin(sA,MAT_FINAL_ASSEMBLY);CHKERRA(ierr);
  ierr = MatAssemblyEnd(sA,MAT_FINAL_ASSEMBLY);CHKERRA(ierr);
 
  /* Assemble MPIBAIJ matrix A */
  ierr = MatCreateMPIBAIJ(PETSC_COMM_WORLD,bs,PETSC_DECIDE,PETSC_DECIDE,n,n,d_nz,PETSC_NULL,o_nz,PETSC_NULL,&A);CHKERRA(ierr);

  if (bs == 1){
    if (prob == 1){ /* tridiagonal matrix */
      value[0] = -1.0; value[1] = 2.0; value[2] = -1.0;
      for (i=1; i<n-1; i++) {
        col[0] = i-1; col[1] = i; col[2] = i+1;
        /* PetscTrValid(0,0,0,0); */
        ierr = MatSetValues(A,1,&i,3,col,value,INSERT_VALUES); CHKERRA(ierr);
      }
      i = n - 1; col[0]=0; col[1] = n - 2; col[2] = n - 1;
      value[0]= 0.1; value[1]=-1; value[2]=2;
      ierr = MatSetValues(A,1,&i,3,col,value,INSERT_VALUES); CHKERRA(ierr);

      i = 0; col[0] = 0; col[1] = 1; col[2]=n-1;
      value[0] = 2.0; value[1] = -1.0; value[2]=0.1;
      ierr = MatSetValues(A,1,&i,3,col,value,INSERT_VALUES); CHKERRA(ierr);
    }
    else if (prob ==2){ /* matrix for the five point stencil */
      n1 = sqrt(n); printf("n1 = %d\n",n1);
      for (i=0; i<n1; i++) {
        for (j=0; j<n1; j++) {
          I = j + n1*i;
          if (i>0)   {J = I - n1; ierr = MatSetValues(A,1,&I,1,&J,&neg_one,INSERT_VALUES);CHKERRA(ierr);}
          if (i<n1-1) {J = I + n1; ierr = MatSetValues(A,1,&I,1,&J,&neg_one,INSERT_VALUES);CHKERRA(ierr);}
          if (j>0)   {J = I - 1; ierr = MatSetValues(A,1,&I,1,&J,&neg_one,INSERT_VALUES);CHKERRA(ierr);}
          if (j<n1-1) {J = I + 1; ierr = MatSetValues(A,1,&I,1,&J,&neg_one,INSERT_VALUES);CHKERRA(ierr);}
          ierr = MatSetValues(A,1,&I,1,&I,&four,INSERT_VALUES);CHKERRA(ierr);
        }
      }                   
    }
  } /* end of if (bs == 1) */
  else {  /* bs > 1 */
  for (block=0; block<n/bs; block++){
    /* diagonal blocks */
    value[0] = -1.0; value[1] = 4.0; value[2] = -1.0;
    for (i=1+block*bs; i<bs-1+block*bs; i++) {
      col[0] = i-1; col[1] = i; col[2] = i+1;
      ierr = MatSetValues(A,1,&i,3,col,value,INSERT_VALUES); CHKERRA(ierr);    
    }
    i = bs - 1+block*bs; col[0] = bs - 2+block*bs; col[1] = bs - 1+block*bs;
    value[0]=-1.0; value[1]=4.0;  
    ierr = MatSetValues(A,1,&i,2,col,value,INSERT_VALUES); CHKERRA(ierr); 

    i = 0+block*bs; col[0] = 0+block*bs; col[1] = 1+block*bs; 
    value[0]=4.0; value[1] = -1.0; 
    ierr = MatSetValues(A,1,&i,2,col,value,INSERT_VALUES); CHKERRA(ierr);  
  }
  /* off-diagonal blocks */
  value[0]=-1.0;
  for (i=0; i<(n/bs-1)*bs; i++){
    col[0]=i+bs;
    ierr = MatSetValues(A,1,&i,1,col,value,INSERT_VALUES); CHKERRA(ierr);
    col[0]=i; row=i+bs;
    ierr = MatSetValues(A,1,&row,1,col,value,INSERT_VALUES); CHKERRA(ierr);
  }
  }
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRA(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRA(ierr);
  

  /* Test MatGetSize(), MatGetLocalSize() */
  /* ierr = MatScale(&alpha,sA);CHKERRA(ierr); MatView(sA, VIEWER_STDOUT_WORLD); */
  ierr = MatGetSize(sA, &i1,&j1); ierr = MatGetSize(A, &i2,&j2);
  i1 -= i2; j1 -= j2;
  if (i1 || j1) {
    PetscSynchronizedPrintf(PETSC_COMM_WORLD,"[%d], Error: MatGetSize()\n",rank);
    PetscSynchronizedFlush(PETSC_COMM_WORLD);
  }
    
  ierr = MatGetLocalSize(sA, &i1,&j1); ierr = MatGetLocalSize(A, &i2,&j2);
  i1 -= i2; j1 -= j2;
  if (i1 || j1) {
    PetscSynchronizedPrintf(PETSC_COMM_WORLD,"[%d], Error: MatGetLocalSize()\n",rank);
    PetscSynchronizedFlush(PETSC_COMM_WORLD);
  }
  
  /* Test MatGetOwnershipRange() */ 
  ierr = MatGetOwnershipRange(sA,&i1,&j1);CHKERRA(ierr);
  ierr = MatGetOwnershipRange(A,&i2,&j2);CHKERRA(ierr);
  i2 -= i1; j2 -= j1;
  if (i2 || j2) {
    PetscSynchronizedPrintf(PETSC_COMM_WORLD,"[%d], Error: MaGetOwnershipRange()\n",rank);
    PetscSynchronizedFlush(PETSC_COMM_WORLD);
  }

  /* Test MatGetRow(): can only obtain rows associated with the given processor */
  for (i=i1; i<i1+1; i++) {
    ierr = MatGetRow(sA,i,&ncols,&cols,&vr);CHKERRA(ierr);
    ierr = PetscSynchronizedFPrintf(PETSC_COMM_WORLD,stdout,"[%d] get row %d: ",rank,i);CHKERRA(ierr);
    for (j=0; j<ncols; j++) {
      ierr = PetscSynchronizedFPrintf(PETSC_COMM_WORLD,stdout,"%d %g  ",cols[j],vr[j]);CHKERRA(ierr);
    }
    ierr = PetscSynchronizedFPrintf(PETSC_COMM_WORLD,stdout,"\n");CHKERRA(ierr);
    ierr = MatRestoreRow(sA,i,&ncols,&cols,&vr);CHKERRA(ierr);
  }
  ierr = PetscSynchronizedFlush(PETSC_COMM_WORLD);CHKERRA(ierr);CHKERRA(ierr);      
 

#ifndef MatZeroRows
  /* list all the rows we want on THIS processor. For symm. matrix, iscol=isrow. */
  /*
  ip_ptr = (int*)PetscMalloc(n*sizeof(int)); CHKERRA(ierr);
  j = 0;
  for (n1=0; n1<mbs; n1 += 2){ /* n1: block row */
    for (i=0; i<bs; i++) ip_ptr[j++] = n1*bs + i;  
  }
  PetscSynchronizedPrintf(PETSC_COMM_WORLD,"in ex75, [%d], j=%d\n",rank,j);
  PetscSynchronizedFlush(PETSC_COMM_WORLD);
  ierr = ISCreateGeneral(PETSC_COMM_WORLD, j, ip_ptr, &isrow); CHKERRA(ierr);
  ierr = PetscFree(ip_ptr); CHKERRA(ierr); 
*/

  ierr = ISGetSize(isrow,&N);CHKERRQ(ierr);
  PetscSynchronizedPrintf(PETSC_COMM_WORLD,"in ex75, [%d], N=%d\n",rank,N);
  PetscSynchronizedFlush(PETSC_COMM_WORLD);

  ISView(isrow, VIEWER_STDOUT_WORLD); CHKERRA(ierr);

  ierr = MatZeroRows(sA,isrow,PETSC_NULL); CHKERRA(ierr);
  MatView(sA, VIEWER_STDOUT_WORLD);
  ierr = ISDestroy(isrow);CHKERRA(ierr);
#endif

  /* vectors */
  /*--------------------*/
  /* ierr = VecCreate(PETSC_COMM_WORLD,PETSC_DECIDE,n,&u);CHKERRA(ierr);*/
  ierr = MatGetLocalSize(sA,&m,&m1);CHKERRA(ierr);
  ierr = VecCreateMPI(PETSC_COMM_WORLD,m,n,&u); CHKERRA(ierr);
  ierr = VecSetFromOptions(u);CHKERRA(ierr);
  ierr = VecDuplicate(u,&b);CHKERRA(ierr); 
  ierr = VecDuplicate(b,&x);CHKERRA(ierr);

  /* 
     Set exact solution; then compute right-hand-side vector.
     By default we use an exact solution of a vector with all
     elements of 1.0;  Alternatively, using the runtime option
     -random_sol forms a solution vector with random components.
  */
  ierr = OptionsHasName(PETSC_NULL,"-random_exact_sol",&flg);CHKERRA(ierr);
  if (flg) {
    ierr = PetscRandomCreate(PETSC_COMM_WORLD,RANDOM_DEFAULT,&rctx);CHKERRA(ierr);
    ierr = VecSetRandom(rctx,u);CHKERRA(ierr);
    ierr = PetscRandomDestroy(rctx);CHKERRA(ierr);
  } else {
    ierr = VecSet(&one,u);CHKERRA(ierr);
  }
  ierr = VecSet(&one,x);CHKERRA(ierr);

#ifdef MatDiagonalScale
  /* ierr = MatDiagonalScale(A,u,u);CHKERRA(ierr); */
  /* MatView(A, VIEWER_STDOUT_WORLD); */
  ierr = MatDiagonalScale(sA,u,u);CHKERRA(ierr); 
  /* MatView(sA, VIEWER_STDOUT_WORLD); */
#endif

#ifdef MatNorm
  ierr = MatNorm(A,NORM_FROBENIUS,&norm);CHKERRA(ierr); 
  /* ierr = MatNorm(A,NORM_1,&norm1);        CHKERRA(ierr); */
  /* ierr = MatNorm(A,NORM_INFINITY,&normi); CHKERRA(ierr); */
  ierr = PetscPrintf(PETSC_COMM_WORLD,"A: Frob. norm = %g\n", norm);CHKERRA(ierr); 

  ierr = MatNorm(sA,NORM_FROBENIUS,&norm);CHKERRA(ierr); 
  /* ierr = MatNorm(sA,NORM_1,&norm1);     CHKERRA(ierr); 
  ierr = MatNorm(sA,NORM_INFINITY,&normi); CHKERRA(ierr); */
  ierr = PetscPrintf(PETSC_COMM_WORLD,"sA: Frob. norm = %g\n", norm);CHKERRA(ierr);
#endif

  ierr = MatMult(sA,u,b);CHKERRA(ierr); 
  /* ierr = MatMult(A,u,b);CHKERRA(ierr);     */    
  /* ierr = MatMultAdd(sA,u,x,b);                 */
  /* ierr = MatGetDiagonal(sA, b); CHKERRA(ierr); */

  /*
     View the exact solution vector if desired
  */
  ierr = OptionsHasName(PETSC_NULL,"-view_exact_sol",&flg);CHKERRA(ierr);
  if (flg) {ierr = VecView(u,VIEWER_STDOUT_WORLD);CHKERRA(ierr);}

  ierr = OptionsHasName(PETSC_NULL,"-view_b",&flg);CHKERRA(ierr);
  if (flg) {ierr = VecView(b,VIEWER_STDOUT_WORLD);CHKERRA(ierr);}
#ifdef CONT
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
     - The following two statements are optional; all of these
       parameters could alternatively be specified at runtime via
       SLESSetFromOptions().  All of these defaults can be
       overridden at runtime, as indicated below.
  */

  ierr = SLESGetKSP(sles,&ksp);CHKERRA(ierr);
  ierr = SLESGetPC(sles,&pc);CHKERRA(ierr);
  ierr = PCSetType(pc,PCJACOBI);CHKERRA(ierr);
  ierr = KSPSetTolerances(ksp,1.e-2/n,1.e-50,PETSC_DEFAULT,PETSC_DEFAULT);CHKERRA(ierr);

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

  ierr = SLESSolve(sles,b,x,&its);CHKERRA(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
                      Check solution and clean up
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /* 
     Check the error
  */
  ierr = VecAXPY(&neg_one,u,x);CHKERRA(ierr);
  ierr = VecNorm(x,NORM_2,&norm);CHKERRA(ierr);

  /* Scale the norm */
  /*  norm *= sqrt(1.0/n); */

  /*
     Print convergence information.  PetscPrintf() produces a single 
     print statement from all processes that share a communicator.
     An alternative is PetscFPrintf(), which prints to a file.
  */
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Norm of error %A iterations %d\n",norm/n,its);CHKERRA(ierr);

  /*
     Free work space.  All PETSc objects should be destroyed when they
     are no longer needed.
  */
  ierr = SLESDestroy(sles);CHKERRA(ierr);
#endif
  ierr = VecDestroy(u);CHKERRA(ierr);  ierr = VecDestroy(x);CHKERRA(ierr);
  ierr = VecDestroy(b);CHKERRA(ierr);  ierr = MatDestroy(sA);CHKERRA(ierr);

  /*
     Always call PetscFinalize() before exiting a program.  This routine
       - finalizes the PETSc libraries as well as MPI
       - provides summary and diagnostic information if certain runtime
         options are chosen (e.g., -log_summary). 
  */

  PetscFinalize();
  return 0;
}
