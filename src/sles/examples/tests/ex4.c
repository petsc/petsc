
static char help[] = "Tests AIJ  solve with SLES.\n";

#include "vec.h"
#include "mat.h"
#include "options.h"
#include  <stdio.h>
#include "sles.h"


int FormElementStiffness(double H,Scalar *Ke)
{
  Ke[0]= H/6.0;Ke[1]= -.125*H;      Ke[2] = H/12.0;      Ke[3] = -.125*H;
  Ke[4]= -.125*H;      Ke[5]= H/6.0;Ke[6] = -.125*H;      Ke[7] = H/12.0;
  Ke[8]= H/12.0;      Ke[9]= -.125*H;   Ke[10]=H/6.0;Ke[11] = -.125*H;
  Ke[12]= -.125*H;     Ke[13]= H/12.0;     Ke[14] = -.125*H;  Ke[15] = H/6.0;
  return 0;
}
int FormElementRhs(double x, double y, double H,Scalar *r)
{
  r[0] = 0.; r[1] = 0.; r[2] = 0.; r[3] = 0.0; 
  return 0;
}

int main(int argc,char **args)
{
  Mat         C; 
  int         i, m = 2,  N,M,its;
  Scalar      val, zero = 0.0,norm, one = 1.0, none = -1.0,Ke[16],r[4];
  double      x,y,h;
  int         ierr,idx[4],count,*rows;
  Vec         u,ustar,b;
  SLES        sles;
  KSP         ksp;
  IS          is;

  PetscInitialize(&argc,&args,0,0);
  if (OptionsHasName(0,0,"-help")) fprintf(stderr,"%s",help);
  OptionsGetInt(0,0,"-m",&m);
  N = (m+1)*(m+1); /* dimension of matrix */
  M = m*m; /* number of elements */
  h = 1.0/m;       /* mesh width */

  /* create stiffness matrix */
  ierr = MatCreateSequentialAIJ(N,N,9,0,&C); 
  CHKERR(ierr);

  /* forms the element stiffness for the Laplacian */
  ierr = FormElementStiffness(h*h,Ke);
  for ( i=0; i<M; i++ ) {
     /* location of lower left corner of element */
     x = h*(i % m); y = h*(i/m); 
     /* node numbers for the four corners of element */
     idx[0] = (m+1)*(i/m) + ( i % m);
     idx[1] = idx[0]+1; idx[2] = idx[1] + m + 1; idx[3] = idx[2] - 1;
     MatSetValues(C,4,idx,4,idx,Ke,AddValues); 
  }
  ierr = MatBeginAssembly(C); CHKERR(ierr);
  ierr = MatEndAssembly(C); CHKERR(ierr);

  /* create right hand side and solution */

  ierr = VecCreateSequential(N,&u); CHKERR(ierr); 
  ierr = VecCreate(u,&b); CHKERR(ierr);
  ierr = VecCreate(b,&ustar); CHKERR(ierr);
  VecSet(&zero,u); VecSet(&zero,b);

  for ( i=0; i<M; i++ ) {
     /* location of lower left corner of element */
     x = h*(i % m); y = h*(i/m); 
     /* node numbers for the four corners of element */
     idx[0] = (m+1)*(i/m) + ( i % m);
     idx[1] = idx[0]+1; idx[2] = idx[1] + m + 1; idx[3] = idx[2] - 1;
     FormElementRhs(x,y,h*h,r);
     VecSetValues(b,4,idx,r,AddValues);
  }
  ierr = VecBeginAssembly(b);
  ierr = VecEndAssembly(b);

  /* modify matrix and rhs for Dirichlet boundary conditions */
  rows = (int *) MALLOC( 4*m*sizeof(int) ); CHKPTR(rows);
  for ( i=0; i<m+1; i++ ) {
    rows[i] = i; /* bottom */
    rows[3*m - 1 +i] = m*(m+1) + i; /* top */
  }
  count = m+1; /* left side */
  for ( i=m+1; i<m*(m+1); i+= m+1 ) {
    rows[count++] = i;
  }
  count = 2*m; /* left side */
  for ( i=2*m+1; i<m*(m+1); i+= m+1 ) {
    rows[count++] = i;
  }
  ierr = ISCreateSequential(4*m,rows,&is); CHKERR(ierr);
  for ( i=0; i<4*m; i++ ) {
     x = h*(rows[i] % (m+1)); y = h*(rows[i]/(m+1)); 
     val = y;
     VecSetValues(u,1,&rows[i],&val,InsertValues); 
     VecSetValues(b,1,&rows[i],&val,InsertValues); 
  }    
  FREE(rows);
  VecBeginAssembly(u); VecEndAssembly(u);
  VecBeginAssembly(b); VecEndAssembly(b);

  ierr = MatZeroRows(C,is,&one); CHKERR(ierr);
  ISDestroy(is);

  /* solve linear system */
  if ((ierr = SLESCreate(&sles))) SETERR(ierr,0);
  if ((ierr = SLESSetOperators(sles,C,C,0))) SETERR(ierr,0);
  if ((ierr = SLESSetFromOptions(sles))) SETERR(ierr,0);
  SLESGetKSP(sles,&ksp);
  KSPSetInitialGuessNonZero(ksp);
  if ((ierr = SLESSolve(sles,b,u,&its))) SETERR(ierr,0);

  /* check error */
  for ( i=0; i<N; i++ ) {
     x = h*(i % (m+1)); y = h*(i/(m+1)); 
     val = y;
     VecSetValues(ustar,1,&i,&val,InsertValues); 
  }
  VecBeginAssembly(ustar); VecEndAssembly(ustar);
/*VecView(u,0); */
/*VecView(ustar,0); */
  if ((ierr = VecAXPY(&none,ustar,u))) SETERR(ierr,0);
  if ((ierr = VecNorm(u,&norm))) SETERR(ierr,0);
  printf("Norm of error %g Number iterations %d\n",norm*h,its);

  sleep(2);
  ierr = SLESDestroy(sles); CHKERR(ierr);
  ierr = VecDestroy(ustar); CHKERR(ierr);
  ierr = VecDestroy(u); CHKERR(ierr);
  ierr = VecDestroy(b); CHKERR(ierr);
  ierr = MatDestroy(C); CHKERR(ierr);
  PetscFinalize();
  return 0;
}
