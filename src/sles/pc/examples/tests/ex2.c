
static char help[] = "Tests PC and KSP on tridiagonal matrix\n";

#include "vec.h"
#include "mat.h"
#include "ksp.h"
#include "pc.h"
#include "stdio.h"
#include "options.h"

int main(int argc,char **args)
{
  Mat       mat;
  Vec       b,ustar,u;
  PC        pc;
  KSP       ksp;
  int       ierr, n = 10, i, its, col[3];
  Scalar    value[3], mone = -1.0, norm, one = 1.0, zero = 0.0;
  KSPMETHOD kspmethod;
  PCMETHOD  pcmethod;
  char      *kspname, *pcname;

  PetscInitialize(&argc,&args,0,0);
  if (OptionsHasName(0,0,"-help")) fprintf(stderr,"%s",help);

  ierr = VecCreateSequential(n,&b);     CHKERRA(ierr);
  ierr = VecCreateSequential(n,&ustar); CHKERRA(ierr);
  ierr = VecCreateSequential(n,&u);     CHKERRA(ierr);
  ierr = VecSet(&one,ustar);            CHKERRA(ierr);
  ierr = VecSet(&zero,u);               CHKERRA(ierr);

  ierr = MatCreateSequentialAIJ(n,n,3,0,&mat); CHKERRA(ierr);
  value[0] = -1.0; value[1] = 2.0; value[2] = -1.0;
  for (i=1; i<n-1; i++ ) {
    col[0] = i-1; col[1] = i; col[2] = i+1;
    ierr = MatSetValues(mat,1,&i,3,col,value,InsertValues); CHKERRA(ierr);
  }
  i = n - 1; col[0] = n - 2; col[1] = n - 1;
  ierr = MatSetValues(mat,1,&i,2,col,value,InsertValues); CHKERRA(ierr);
  i = 0; col[0] = 0; col[1] = 1; value[0] = 2.0; value[1] = -1.0;
  ierr = MatSetValues(mat,1,&i,2,col,value,InsertValues); CHKERRA(ierr);
  ierr = MatBeginAssembly(mat,FINAL_ASSEMBLY); CHKERRA(ierr);
  ierr = MatEndAssembly(mat,FINAL_ASSEMBLY); CHKERRA(ierr);


  ierr = MatMult(mat,ustar,b); CHKERRA(ierr);

  ierr = PCCreate(&pc); CHKERRA(ierr);
  ierr = PCSetMethod(pc,PCNONE); CHKERRA(ierr);
  PCSetFromOptions(pc);
  ierr = PCSetOperators(pc,mat,mat,0); CHKERRA(ierr);
  ierr = PCSetVector(pc,u);   CHKERRA(ierr);
  ierr = PCSetUp(pc); CHKERRA(ierr);

  ierr = KSPCreate(&ksp); CHKERRA(ierr);
  ierr = KSPSetMethod(ksp,KSPRICHARDSON); CHKERRA(ierr);
  KSPSetFromOptions(ksp);
  ierr = KSPSetSolution(ksp,u); CHKERRA(ierr);
  ierr = KSPSetRhs(ksp,b); CHKERRA(ierr);
  ierr = PCSetOperators(pc,mat,mat,0); CHKERRA(ierr);
  ierr = KSPSetBinv(ksp,pc); CHKERRA(ierr);
  ierr = KSPSetUp(ksp); CHKERRA(ierr);

  KSPGetMethodFromContext(ksp,&kspmethod);
  KSPGetMethodName(kspmethod,&kspname);
  PCGetMethodFromContext(pc,&pcmethod);
  PCGetMethodName(pcmethod,&pcname);
  
  printf("Running %s with %s preconditioning\n",kspname,pcname);
  ierr = KSPSolve(ksp,&its); CHKERRA(ierr);
  ierr = VecAXPY(&mone,ustar,u); CHKERRA(ierr);
  ierr = VecNorm(u,&norm);
  fprintf(stdout,"Number of iterations %d 2 norm error %g\n",its,norm);

  ierr = KSPDestroy(ksp); CHKERRA(ierr);
  ierr = VecDestroy(u); CHKERRA(ierr);
  ierr = VecDestroy(ustar); CHKERRA(ierr);
  ierr = VecDestroy(b); CHKERRA(ierr);
  ierr = MatDestroy(mat); CHKERRA(ierr);
  ierr = PCDestroy(pc); CHKERRA(ierr);

  PetscFinalize();
  return 0;
}
    


