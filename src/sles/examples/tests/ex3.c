/*$Id: ex3.c,v 1.64 2001/01/15 21:47:31 bsmith Exp balay $*/

static char help[] = 
"This example solves a linear system in parallel with SLES.  The matrix\n\
uses simple bilinear elements on the unit square.  To test the parallel\n\
matrix assembly, the matrix is intentionally laid out across processors\n\
differently from the way it is assembled.  Input arguments are:\n\
  -m <size> : problem size\n\n";

#include "petscsles.h"

#undef __FUNC__
#define __FUNC__ "FormElementStiffness"
int FormElementStiffness(double H,Scalar *Ke)
{
  PetscFunctionBegin;
  Ke[0]  = H/6.0;    Ke[1]  = -.125*H; Ke[2]  = H/12.0;   Ke[3]  = -.125*H;
  Ke[4]  = -.125*H;  Ke[5]  = H/6.0;   Ke[6]  = -.125*H;  Ke[7]  = H/12.0;
  Ke[8]  = H/12.0;   Ke[9]  = -.125*H; Ke[10] = H/6.0;    Ke[11] = -.125*H;
  Ke[12] = -.125*H;  Ke[13] = H/12.0;  Ke[14] = -.125*H;  Ke[15] = H/6.0;
  PetscFunctionReturn(0);
}
#undef __FUNC__
#define __FUNC__ "FormElementRhs"
int FormElementRhs(double x,double y,double H,Scalar *r)
{
  PetscFunctionBegin;
  r[0] = 0.; r[1] = 0.; r[2] = 0.; r[3] = 0.0; 
  PetscFunctionReturn(0);
}

#undef __FUNC__
#define __FUNC__ "main"
int main(int argc,char **args)
{
  Mat         C; 
  int         i,m = 5,rank,size,N,start,end,M,its;
  Scalar      val,zero = 0.0,one = 1.0,none = -1.0,Ke[16],r[4];
  double      x,y,h,norm;
  int         ierr,idx[4],count,*rows;
  Vec         u,ustar,b;
  SLES        sles;
  KSP         ksp;
  IS          is;

  PetscInitialize(&argc,&args,(char *)0,help);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-m",&m,PETSC_NULL);CHKERRA(ierr);
  N = (m+1)*(m+1); /* dimension of matrix */
  M = m*m; /* number of elements */
  h = 1.0/m;       /* mesh width */
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRA(ierr);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRA(ierr);

  /* Create stiffness matrix */
  ierr = MatCreate(PETSC_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,N,N,&C);CHKERRA(ierr);
  ierr = MatSetFromOptions(C);CHKERRA(ierr);
  start = rank*(M/size) + ((M%size) < rank ? (M%size) : rank);
  end   = start + M/size + ((M%size) > rank); 

  /* Assemble matrix */
  ierr = FormElementStiffness(h*h,Ke);   /* element stiffness for Laplacian */
  for (i=start; i<end; i++) {
     /* location of lower left corner of element */
     x = h*(i % m); y = h*(i/m); 
     /* node numbers for the four corners of element */
     idx[0] = (m+1)*(i/m) + (i % m);
     idx[1] = idx[0]+1; idx[2] = idx[1] + m + 1; idx[3] = idx[2] - 1;
     ierr = MatSetValues(C,4,idx,4,idx,Ke,ADD_VALUES);CHKERRA(ierr);
  }
  ierr = MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY);CHKERRA(ierr);
  ierr = MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY);CHKERRA(ierr);

  /* Create right-hand-side and solution vectors */
  ierr = VecCreate(PETSC_COMM_WORLD,PETSC_DECIDE,N,&u);CHKERRA(ierr); 
  ierr = VecSetFromOptions(u);CHKERRA(ierr);
  ierr = PetscObjectSetName((PetscObject)u,"Approx. Solution");CHKERRA(ierr);
  ierr = VecDuplicate(u,&b);CHKERRA(ierr);
  ierr = PetscObjectSetName((PetscObject)b,"Right hand side");CHKERRA(ierr);
  ierr = VecDuplicate(b,&ustar);CHKERRA(ierr);
  ierr = VecSet(&zero,u);CHKERRA(ierr);
  ierr = VecSet(&zero,b);CHKERRA(ierr);

  /* Assemble right-hand-side vector */
  for (i=start; i<end; i++) {
     /* location of lower left corner of element */
     x = h*(i % m); y = h*(i/m); 
     /* node numbers for the four corners of element */
     idx[0] = (m+1)*(i/m) + (i % m);
     idx[1] = idx[0]+1; idx[2] = idx[1] + m + 1; idx[3] = idx[2] - 1;
     ierr = FormElementRhs(x,y,h*h,r);CHKERRA(ierr);
     ierr = VecSetValues(b,4,idx,r,ADD_VALUES);CHKERRA(ierr);
  }
  ierr = VecAssemblyBegin(b);CHKERRA(ierr);
  ierr = VecAssemblyEnd(b);CHKERRA(ierr);

  /* Modify matrix and right-hand-side for Dirichlet boundary conditions */
  ierr = PetscMalloc(4*m*sizeof(int),&rows);CHKERRQ(ierr);
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
  ierr = ISCreateGeneral(PETSC_COMM_SELF,4*m,rows,&is);CHKERRA(ierr);
  for (i=0; i<4*m; i++) {
     x = h*(rows[i] % (m+1)); y = h*(rows[i]/(m+1)); 
     val = y;
     ierr = VecSetValues(u,1,&rows[i],&val,INSERT_VALUES);CHKERRA(ierr);
     ierr = VecSetValues(b,1,&rows[i],&val,INSERT_VALUES);CHKERRA(ierr);
  }    
  ierr = PetscFree(rows);CHKERRA(ierr);
  ierr = VecAssemblyBegin(u);CHKERRA(ierr);
  ierr = VecAssemblyEnd(u);CHKERRA(ierr);
  ierr = VecAssemblyBegin(b);CHKERRA(ierr); 
  ierr = VecAssemblyEnd(b);CHKERRA(ierr);

  ierr = MatZeroRows(C,is,&one);CHKERRA(ierr);
  ierr = ISDestroy(is);CHKERRA(ierr);


  { Mat A;
  ierr = MatConvert(C,MATSAME,&A);CHKERRA(ierr);
  ierr = MatDestroy(C);CHKERRA(ierr);
  ierr = MatConvert(A,MATSAME,&C);CHKERRA(ierr);
  ierr = MatDestroy(A);CHKERRA(ierr);
  }

  /* Solve linear system */
  ierr = SLESCreate(PETSC_COMM_WORLD,&sles);CHKERRA(ierr);
  ierr = SLESSetOperators(sles,C,C,DIFFERENT_NONZERO_PATTERN);CHKERRA(ierr);
  ierr = SLESSetFromOptions(sles);CHKERRA(ierr);
  ierr = SLESGetKSP(sles,&ksp);CHKERRA(ierr);
  ierr = KSPSetInitialGuessNonzero(ksp);CHKERRA(ierr);
  ierr = SLESSolve(sles,b,u,&its);CHKERRA(ierr);

  /* Check error */
  ierr = VecGetOwnershipRange(ustar,&start,&end);CHKERRA(ierr);
  for (i=start; i<end; i++) {
     x = h*(i % (m+1)); y = h*(i/(m+1)); 
     val = y;
     ierr = VecSetValues(ustar,1,&i,&val,INSERT_VALUES);CHKERRA(ierr);
  }
  ierr = VecAssemblyBegin(ustar);CHKERRA(ierr);
  ierr = VecAssemblyEnd(ustar);CHKERRA(ierr);
  ierr = VecAXPY(&none,ustar,u);CHKERRA(ierr);
  ierr = VecNorm(u,NORM_2,&norm);CHKERRA(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Norm of error %A Iterations %d\n",norm*h,its);CHKERRA(ierr);

  /* Free work space */
  ierr = SLESDestroy(sles);CHKERRA(ierr);
  ierr = VecDestroy(ustar);CHKERRA(ierr);
  ierr = VecDestroy(u);CHKERRA(ierr);
  ierr = VecDestroy(b);CHKERRA(ierr);
  ierr = MatDestroy(C);CHKERRA(ierr);
  PetscFinalize();
  return 0;
}


