
static char help[] = 
"This example tests MPI parallel linear solves with SLES.  The code\n\
illustrates repeated solution of linear systems with the same preconditioner\n\
method but different matrices (having the same nonzero structure).\n";

#include "vec.h"
#include "mat.h"
#include "options.h"
#include  <stdio.h>
#include "sles.h"

extern int KSPMonitor_MPIRowbs(KSP,int,double,void *);

int main(int argc,char **args)
{
  Mat    C; 
  Scalar v, one = 1.0, none = -1.0;
  int    I, J, ldim, ierr, low, high, iglobal;
  int    i, j, m = 3, n = 2, mytid, numtids, its;
  Vec    x, u, b;
  SLES   sles;
  KSP    ksp;
  double norm;

  PetscInitialize(&argc,&args,0,0);
  if (OptionsHasName(0,0,"-help")) fprintf(stderr,"%s",help);
  OptionsGetInt(0,0,"-m",&m);
  MPI_Comm_rank(MPI_COMM_WORLD,&mytid);
  MPI_Comm_size(MPI_COMM_WORLD,&numtids);
  n = 2*numtids;

  if (OptionsHasName(0,0,"-row_mat"))
    ierr = MatCreateMPIRow(MPI_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,
                           m*n,m*n,5,0,5,0,&C); 
#if defined(HAVE_BLOCKSOLVE) && !defined(PETSC_COMPLEX)
  else if (OptionsHasName(0,0,"-rowbs_mat"))
    ierr = MatCreateMPIRowbs(MPI_COMM_WORLD,PETSC_DECIDE,m*n,5,0,0,&C); 
#endif
  else
    ierr = MatCreateMPIAIJ(MPI_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,
                           m*n,m*n,5,0,5,0,&C); 
  CHKERRA(ierr);

  /* Generate matrix */
  for ( i=0; i<m; i++ ) { 
    for ( j=2*mytid; j<2*mytid+2; j++ ) {
      v = -1.0;  I = j + n*i;
      if ( i>0 )   {J = I - n; MatSetValues(C,1,&I,1,&J,&v,InsertValues);}
      if ( i<m-1 ) {J = I + n; MatSetValues(C,1,&I,1,&J,&v,InsertValues);}
      if ( j>0 )   {J = I - 1; MatSetValues(C,1,&I,1,&J,&v,InsertValues);}
      if ( j<n-1 ) {J = I + 1; MatSetValues(C,1,&I,1,&J,&v,InsertValues);}
      v = 4.0; MatSetValues(C,1,&I,1,&I,&v,InsertValues);
    }
  }
  ierr = MatBeginAssembly(C,FINAL_ASSEMBLY); CHKERRA(ierr);
  ierr = MatEndAssembly(C,FINAL_ASSEMBLY); CHKERRA(ierr);

  /* Generate vectors */
  ierr = VecCreateMPI(MPI_COMM_WORLD,PETSC_DECIDE,m*n,&u); CHKERRA(ierr);
  ierr = VecCreate(u,&b); CHKERRA(ierr);
  ierr = VecCreate(b,&x); CHKERRA(ierr);
  ierr = VecGetLocalSize(x,&ldim); CHKERR(ierr);
  ierr = VecGetOwnershipRange(x,&low,&high); CHKERR(ierr);
  for (i=0; i<ldim; i++) {
    iglobal = i + low;
    v = one*i + 100*mytid;
    ierr = VecSetValues(u,1,&iglobal,&v,InsertValues); CHKERR(ierr);
  }
  ierr = VecBeginAssembly(u); CHKERR(ierr);
  ierr = VecEndAssembly(u); CHKERR(ierr);
  
  /* Compute right-hand-side */
  ierr = MatMult(C,u,b); CHKERRA(ierr);
  
  /* Solve linear system */
  if ((ierr = SLESCreate(MPI_COMM_WORLD,&sles))) SETERRA(ierr,0);
  if ((ierr = SLESSetOperators(sles,C,C,MAT_SAME_NONZERO_PATTERN)))
    SETERRA(ierr,0);
  if ((ierr = SLESSetFromOptions(sles))) SETERRA(ierr,0);

#if defined(HAVE_BLOCKSOLVE) && !defined(PETSC_COMPLEX)
  if (OptionsHasName(0,0,"-rowbs_mat")) {
    ierr = SLESGetKSP(sles,&ksp); CHKERR(ierr);
    ierr = KSPSetMonitor(ksp,KSPMonitor_MPIRowbs,(void *)C); CHKERR(ierr);
  }
#endif

  if ((ierr = SLESSolve(sles,b,x,&its))) SETERRA(ierr,0);
 
  /* Check error */
  if ((ierr = VecAXPY(&none,u,x))) SETERRA(ierr,0);
  if ((ierr = VecNorm(x,&norm))) SETERRA(ierr,0);
  MPE_printf(MPI_COMM_WORLD,"Norm of error %g, Number of iterations %d\n",norm,its);

  /* Change matrix (keeping same nonzero structure) and solve again */
  MatSetOption(C,NO_NEW_NONZERO_LOCATIONS);
  MatZeroEntries(C);
  /* Fill matrix again */
  for ( i=0; i<m; i++ ) { 
    for ( j=2*mytid; j<2*mytid+2; j++ ) {
      v = -1.0;  I = j + n*i;
      if ( i>0 )   {J = I - n; MatSetValues(C,1,&I,1,&J,&v,InsertValues);}
      if ( i<m-1 ) {J = I + n; MatSetValues(C,1,&I,1,&J,&v,InsertValues);}
      if ( j>0 )   {J = I - 1; MatSetValues(C,1,&I,1,&J,&v,InsertValues);}
      if ( j<n-1 ) {J = I + 1; MatSetValues(C,1,&I,1,&J,&v,InsertValues);}
      v = 6.0; MatSetValues(C,1,&I,1,&I,&v,InsertValues);
    }
  } 
  ierr = MatBeginAssembly(C,FINAL_ASSEMBLY); CHKERRA(ierr);
  ierr = MatEndAssembly(C,FINAL_ASSEMBLY); CHKERRA(ierr); 

  /* Compute another right-hand-side; then solve */
  ierr = MatMult(C,u,b); CHKERRA(ierr);
  if ((ierr = SLESSetOperators(sles,C,C,MAT_SAME_NONZERO_PATTERN)))
    SETERRA(ierr,0);
  if ((ierr = SLESSolve(sles,b,x,&its))) SETERRA(ierr,0);

  /* Check error */
  if ((ierr = VecAXPY(&none,u,x))) SETERRA(ierr,0);
  if ((ierr = VecNorm(x,&norm))) SETERRA(ierr,0);
  MPE_printf(MPI_COMM_WORLD,"Norm of error %g, Number of iterations %d\n",norm,its);

  /* Free work space */
  ierr = SLESDestroy(sles); CHKERRA(ierr);
  ierr = VecDestroy(u); CHKERRA(ierr);
  ierr = VecDestroy(x); CHKERRA(ierr);
  ierr = VecDestroy(b); CHKERRA(ierr);
  ierr = MatDestroy(C); CHKERRA(ierr);

  PetscFinalize();
  return 0;
}


