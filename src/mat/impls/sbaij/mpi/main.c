/*$Id: modified from ex2.c,v 1.84 2000/01/11 21:02:20 bsmith Exp $*/

/* Program usage:  mpirun -np <procs> ex2 [-help] [all PETSc options] */ 

static char help[] = "Solves a linear system in parallel with SLES.\n\
Input parameters include:\n\
  -random_exact_sol : use a random exact solution vector\n\
  -view_exact_sol   : write exact solution vector to stdout\n\
  -m <mesh_x>       : number of mesh points in x-direction\n\
  -n <mesh_n>       : number of mesh points in y-direction\n\n";

/*T
   Concepts: SLES^Solving a system of linear equations (basic parallel example);
   Concepts: SLES^Laplacian, 2d
   Concepts: Laplacian, 2d
   Routines: SLESCreate(); SLESSetOperators(); SLESSetFromOptions();
   Routines: SLESSolve(); SLESGetKSP(); SLESGetPC();
   Routines: KSPSetTolerances(); PCSetType();
   Routines: PetscRandomCreate(); PetscRandomDestroy(); VecSetRandom();
   Processors: n
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
#include "mpisbaij.h"

#undef __FUNC__
#define __FUNC__ "main"
int main(int argc,char **args)
{
  Vec         x,b,u;    /* approx solution, RHS, exact solution */
  Mat         A,sA;     /* linear system matrix */
  SLES        sles;     /* linear solver context */
  PC          pc;       /* preconditioner context */
  PetscRandom rctx;     /* random number generator context */
  double      norm;     /* norm of solution error */
  int         i,j,I,J,Istart,Iend,ierr,its;
  PetscTruth  flg;
  Scalar      v, one=1.0, neg_one=-1.0, value[3], four=4.0,alpha=0.1,*diag;
  KSP         ksp;
  int         bs=2, d_nz=3, o_nz=3, n = 16, prob=2;
  int         rank,col[3],n1,mbs,block,row;
  int         flg_A = 0, flg_sA = 1;
  int         ncols, *cols,*ip_ptr;
  Scalar      *vr;
  IS          isrow;

  PetscInitialize(&argc,&args,(char *)0,help);
  ierr = OptionsGetInt(PETSC_NULL,"-n",&n,PETSC_NULL);CHKERRA(ierr);
  ierr = OptionsGetInt(PETSC_NULL,"-bs",&bs,PETSC_NULL);CHKERRA(ierr);
  
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRA(ierr);
  
  /* Assemble matrix */

  mbs = n/bs;
  if (flg_sA ){
  ierr = MatCreateMPISBAIJ(PETSC_COMM_WORLD,bs,PETSC_DECIDE,PETSC_DECIDE,n,n,d_nz,PETSC_NULL,o_nz,PETSC_NULL,&sA);CHKERRA(ierr);

  ierr = MatGetOwnershipRange(sA,&Istart,&Iend);CHKERRA(ierr); 
  PetscSynchronizedPrintf(PETSC_COMM_WORLD,"[%d],for sA, Istart=%d, Iend=%d\n",rank,Istart,Iend); 
  PetscSynchronizedFlush(PETSC_COMM_WORLD); 

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
  }
  /* ------ A -----------*/
  if (flg_A){
  ierr = MatCreateMPIBAIJ(PETSC_COMM_WORLD,bs,PETSC_DECIDE,PETSC_DECIDE,n,n,d_nz,PETSC_NULL,o_nz,PETSC_NULL,&A);CHKERRA(ierr);

  ierr = MatGetOwnershipRange(A,&Istart,&Iend);CHKERRA(ierr);  
  PetscSynchronizedPrintf(PETSC_COMM_WORLD,"[%d],for A, Istart=%d, Iend=%d\n",rank,Istart,Iend); 
  PetscSynchronizedFlush(PETSC_COMM_WORLD); 

  /* Assemble matrix */
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
  }

#ifdef Test
  /* ierr = MatScale(&alpha,sA);CHKERRA(ierr); MatView(sA, VIEWER_STDOUT_WORLD); */
  ierr = MatGetSize(sA, &i,&j);
  MatGetLocalSize(sA, &i,&j);
  /* ierr = MatGetOwnershipRange(sA,&i,&j);CHKERRA(ierr); */

  PetscSynchronizedPrintf(PETSC_COMM_WORLD,"[%d],for sA, i=%d, j=%d\n",rank,i,j); 
  PetscSynchronizedFlush(PETSC_COMM_WORLD); 
#endif

#ifdef MatGetRow
  row = 3*bs-1; 
  ierr = MatGetRow(sA,row,&ncols,&cols,&vr); CHKERRA(ierr); 
  PetscPrintf(PETSC_COMM_WORLD,"[%d], row=%d\n", rank,row);
  for (i=0; i<ncols; i++) {
    PetscSynchronizedPrintf(PETSC_COMM_WORLD,"[%d], cols[%d] = %d, %g\n",rank,i,*cols++,*vr++);
  }

  PetscSynchronizedFlush(PETSC_COMM_WORLD);
  cols -= ncols; vr -= ncols; 
  ierr = MatRestoreRow(sA,row,&ncols,&cols,&vr); CHKERRA(ierr); 
#endif

#ifdef MatZeroRows
  /* list all the rows we want on THIS processor. For symm. matrix, iscol=isrow. */
  ip_ptr = (int*)PetscMalloc(n*sizeof(int)); CHKERRA(ierr);
  j = 0;
  for (n1=0; n1<mbs; n1 += 2){ /* n1: block row */
    for (i=0; i<bs; i++) ip_ptr[j++] = n1*bs + i;  
  }
  ierr = ISCreateGeneral(PETSC_COMM_WORLD, j, ip_ptr, &isrow); CHKERRA(ierr);
  ierr = PetscFree(ip_ptr); CHKERRA(ierr); 
  ISView(isrow, VIEWER_STDOUT_WORLD); CHKERRA(ierr);

  ierr = MatZeroRows(sA,isrow,diag); CHKERRA(ierr);
  MatView(sA, VIEWER_STDOUT_WORLD);
  ierr = ISDestroy(isrow);CHKERRA(ierr);
#endif
  /* vectors */
  /*--------------------*/
  /* ierr = VecCreate(PETSC_COMM_WORLD,PETSC_DECIDE,n,&u);CHKERRA(ierr);*/
  ierr = VecCreateMPI(PETSC_COMM_WORLD,sA->m,n,&u); CHKERRA(ierr);
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
#ifndef CONT
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
