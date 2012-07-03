
static char help[] = "Solves a linear system with KSP.  The matrix uses simple\n\
bilinear elements on the unit square. Input arguments are:\n\
  -m <size> : problem size\n\n";

#include <petscksp.h>

#undef __FUNCT__
#define __FUNCT__ "FormatElementStiffness"
int FormElementStiffness(PetscReal H,PetscScalar *Ke)
{
  Ke[0]  = H/6.0;    Ke[1]  = -.125*H; Ke[2]  = H/12.0;   Ke[3]  = -.125*H;
  Ke[4]  = -.125*H;  Ke[5]  = H/6.0;   Ke[6]  = -.125*H;  Ke[7]  = H/12.0;
  Ke[8]  = H/12.0;   Ke[9]  = -.125*H; Ke[10] = H/6.0;    Ke[11] = -.125*H;
  Ke[12] = -.125*H;  Ke[13] = H/12.0;  Ke[14] = -.125*H;  Ke[15] = H/6.0;
  return 0;
}
#undef __FUNCT__
#define __FUNCT__ "FormElementRhs"
int FormElementRhs(PetscReal x,PetscReal y,PetscReal H,PetscScalar *r)
{
  r[0] = 0.; r[1] = 0.; r[2] = 0.; r[3] = 0.0; 
  return 0;
}

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **args)
{
  Mat            C; 
  PetscErrorCode ierr;
  PetscInt       i,m = 2,N,M,its,idx[4],count,*rows;
  PetscScalar    val,Ke[16],r[4];
  PetscReal      x,y,h,norm,tol=1.e-14;
  Vec            u,ustar,b;
  KSP            ksp;

  PetscInitialize(&argc,&args,(char *)0,help);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-m",&m,PETSC_NULL);CHKERRQ(ierr);
  N = (m+1)*(m+1); /* dimension of matrix */
  M = m*m; /* number of elements */
  h = 1.0/m;       /* mesh width */

  /* create stiffness matrix */
  ierr = MatCreateSeqAIJ(PETSC_COMM_SELF,N,N,9,PETSC_NULL,&C);CHKERRQ(ierr);
  ierr = MatSetUp(C);CHKERRQ(ierr);

  /* forms the element stiffness for the Laplacian */
  ierr = FormElementStiffness(h*h,Ke);CHKERRQ(ierr);
  for (i=0; i<M; i++) {
     /* location of lower left corner of element */
     x = h*(i % m); y = h*(i/m); 
     /* node numbers for the four corners of element */
     idx[0] = (m+1)*(i/m) + (i % m);
     idx[1] = idx[0]+1; idx[2] = idx[1] + m + 1; idx[3] = idx[2] - 1;
     ierr = MatSetValues(C,4,idx,4,idx,Ke,ADD_VALUES);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  /* create right hand side and solution */

  ierr = VecCreateSeq(PETSC_COMM_SELF,N,&u);CHKERRQ(ierr); 
  ierr = VecDuplicate(u,&b);CHKERRQ(ierr);
  ierr = VecDuplicate(b,&ustar);CHKERRQ(ierr);
  ierr = VecSet(u,0.0);CHKERRQ(ierr);
  ierr = VecSet(b,0.0);CHKERRQ(ierr);

  for (i=0; i<M; i++) {
     /* location of lower left corner of element */
     x = h*(i % m); y = h*(i/m); 
     /* node numbers for the four corners of element */
     idx[0] = (m+1)*(i/m) + (i % m);
     idx[1] = idx[0]+1; idx[2] = idx[1] + m + 1; idx[3] = idx[2] - 1;
     ierr = FormElementRhs(x,y,h*h,r);CHKERRQ(ierr);
     ierr = VecSetValues(b,4,idx,r,ADD_VALUES);CHKERRQ(ierr);
  }
  ierr = VecAssemblyBegin(b);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(b);CHKERRQ(ierr);

  /* modify matrix and rhs for Dirichlet boundary conditions */
  ierr = PetscMalloc((4*m+1)*sizeof(PetscInt),&rows);CHKERRQ(ierr);
  for (i=0; i<m+1; i++) {
    rows[i] = i; /* bottom */
    rows[3*m - 1 +i] = m*(m+1) + i; /* top */
  }
  count = m+1; /* left side */
  for (i=m+1; i<m*(m+1); i+= m+1) {
    rows[count++] = i;
  }
  count = 2*m; /* left side */
  for (i=2*m+1; i<m*(m+1); i+= m+1) {
    rows[count++] = i;
  }
  for (i=0; i<4*m; i++) {
     x = h*(rows[i] % (m+1)); y = h*(rows[i]/(m+1)); 
     val = y;
     ierr = VecSetValues(u,1,&rows[i],&val,INSERT_VALUES);CHKERRQ(ierr);
     ierr = VecSetValues(b,1,&rows[i],&val,INSERT_VALUES);CHKERRQ(ierr);
  }    
  ierr = MatZeroRows(C,4*m,rows,1.0,0,0);CHKERRQ(ierr);

  ierr = PetscFree(rows);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(u);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(u);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(b);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(b);CHKERRQ(ierr);

  /* solve linear system */
  ierr = KSPCreate(PETSC_COMM_WORLD,&ksp);CHKERRQ(ierr);
  ierr = KSPSetOperators(ksp,C,C,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
  ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);
  ierr = KSPSetInitialGuessNonzero(ksp,PETSC_TRUE);CHKERRQ(ierr);
  ierr = KSPSolve(ksp,b,u);CHKERRQ(ierr);

  /* check error */
  for (i=0; i<N; i++) {
     x = h*(i % (m+1)); y = h*(i/(m+1)); 
     val = y;
     ierr = VecSetValues(ustar,1,&i,&val,INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = VecAssemblyBegin(ustar);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(ustar);CHKERRQ(ierr);

  ierr = VecAXPY(u,-1.0,ustar);CHKERRQ(ierr);
  ierr = VecNorm(u,NORM_2,&norm);CHKERRQ(ierr);
  ierr = KSPGetIterationNumber(ksp,&its);CHKERRQ(ierr);
  if (norm > tol){
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Norm of error %G Iterations %D\n",norm*h,its);CHKERRQ(ierr);
  }

  ierr = KSPDestroy(&ksp);CHKERRQ(ierr);
  ierr = VecDestroy(&ustar);CHKERRQ(ierr);
  ierr = VecDestroy(&u);CHKERRQ(ierr);
  ierr = VecDestroy(&b);CHKERRQ(ierr);
  ierr = MatDestroy(&C);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return 0;
}
