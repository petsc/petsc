#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: ex17.c,v 1.28 1999/05/12 03:32:04 bsmith Exp balay $";
#endif

static char help[] = "Solves a linear system with SLES.  This problem is\n\
intended to test the complex numbers version of various solvers.\n\n";

#include "sles.h"

typedef enum {TEST_1, TEST_2, TEST_3, HELMHOLTZ_1, HELMHOLTZ_2} TestType;
extern int FormTestMatrix(Mat,int,TestType);

#undef __FUNC__
#define __FUNC__ "main"
int main(int argc,char **args)
{
  Vec         x, b, u;      /* approx solution, RHS, exact solution */
  Mat         A;            /* linear system matrix */
  SLES        sles;         /* SLES context */
  int         ierr, n = 10, its, flg, dim, p = 1, use_random;
  Scalar      none = -1.0, pfive = 0.5;
  double      norm;
  PetscRandom rctx;
  TestType    type;


  PetscInitialize(&argc,&args,(char *)0,help);
  ierr = OptionsGetInt(PETSC_NULL,"-n",&n,&flg);CHKERRA(ierr);
  ierr = OptionsGetInt(PETSC_NULL,"-p",&p,&flg);CHKERRA(ierr);
  switch (p) {
    case 1:  type = TEST_1;      dim = n;   break;
    case 2:  type = TEST_2;      dim = n;   break;
    case 3:  type = TEST_3;      dim = n;   break;
    case 4:  type = HELMHOLTZ_1; dim = n*n; break;
    case 5:  type = HELMHOLTZ_2; dim = n*n; break;
    default: type = TEST_1;      dim = n;
  }

  /* Create vectors */
  ierr = VecCreate(PETSC_COMM_WORLD,PETSC_DECIDE,dim,&x);CHKERRA(ierr);
  ierr = VecSetFromOptions(x);CHKERRA(ierr);
  ierr = VecDuplicate(x,&b);CHKERRA(ierr);
  ierr = VecDuplicate(x,&u);CHKERRA(ierr);

  use_random = 1;
  ierr = OptionsHasName(PETSC_NULL,"-norandom",&flg);CHKERRA(ierr);
  if (flg) {
    use_random = 0;
    ierr = VecSet(&pfive,u);CHKERRA(ierr);
  } else {
    ierr = PetscRandomCreate(PETSC_COMM_WORLD,RANDOM_DEFAULT,&rctx);CHKERRA(ierr);
    ierr = VecSetRandom(rctx,u);CHKERRA(ierr);
  }

  /* Create and assemble matrix */
  ierr = MatCreate(PETSC_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,dim,dim,&A);CHKERRA(ierr);
  ierr = FormTestMatrix(A,n,type);CHKERRQ(ierr);
  ierr = MatMult(A,u,b);CHKERRA(ierr);
  ierr = OptionsHasName(PETSC_NULL,"-printout",&flg);CHKERRA(ierr);
  if (flg) {
    ierr = MatView(A,VIEWER_STDOUT_WORLD);CHKERRA(ierr);
    ierr = VecView(u,VIEWER_STDOUT_WORLD);CHKERRA(ierr);
    ierr = VecView(b,VIEWER_STDOUT_WORLD);CHKERRA(ierr);
  }

  /* Create SLES context; set operators and options; solve linear system */
  ierr = SLESCreate(PETSC_COMM_WORLD,&sles);CHKERRA(ierr);
  ierr = SLESSetOperators(sles,A,A,DIFFERENT_NONZERO_PATTERN);
 CHKERRA(ierr);
  ierr = SLESSetFromOptions(sles);CHKERRA(ierr);
  ierr = SLESSolve(sles,b,x,&its);CHKERRA(ierr);
  ierr = SLESView(sles,VIEWER_STDOUT_WORLD);CHKERRA(ierr);

  /* Check error */
  ierr = VecAXPY(&none,u,x);CHKERRA(ierr);
  ierr  = VecNorm(x,NORM_2,&norm);CHKERRA(ierr);
  if (norm > 1.e-12) 
    PetscPrintf(PETSC_COMM_WORLD,"Norm of error %g, Iterations %d\n",norm,its);
  else 
    PetscPrintf(PETSC_COMM_WORLD,"Norm of error < 1.e-12, Iterations %d\n",its);

  /* Free work space */
  ierr = VecDestroy(x);CHKERRA(ierr); ierr = VecDestroy(u);CHKERRA(ierr);
  ierr = VecDestroy(b);CHKERRA(ierr); ierr = MatDestroy(A);CHKERRA(ierr);
  if (use_random) {ierr = PetscRandomDestroy(rctx);CHKERRQ(ierr);}
  ierr = SLESDestroy(sles);CHKERRA(ierr);
  PetscFinalize();
  return 0;
}

#undef __FUNC__
#define __FUNC__ "FormTestMatrix"
int FormTestMatrix(Mat A,int n,TestType type)
{
#if !defined(PETSC_USE_COMPLEX)
  SETERRQ(1,0,"FormTestMatrix: These problems require complex numbers.");
#else

  Scalar val[5], h;
  int    flg, i, j, I, J, ierr, col[5], Istart, Iend;

  ierr = MatGetOwnershipRange(A,&Istart,&Iend);CHKERRA(ierr);
  if (type == TEST_1) {
    val[0] = 1.0; val[1] = 4.0; val[2] = -2.0;
    for (i=1; i<n-1; i++ ) {
      col[0] = i-1; col[1] = i; col[2] = i+1;
      ierr = MatSetValues(A,1,&i,3,col,val,INSERT_VALUES);CHKERRQ(ierr);
    }
    i = n-1; col[0] = n-2; col[1] = n-1;
    ierr = MatSetValues(A,1,&i,2,col,val,INSERT_VALUES);CHKERRQ(ierr);
    i = 0; col[0] = 0; col[1] = 1; val[0] = 4.0; val[1] = -2.0;
    ierr = MatSetValues(A,1,&i,2,col,val,INSERT_VALUES);CHKERRQ(ierr);
  } 
  else if (type == TEST_2) {
    val[0] = 1.0; val[1] = 0.0; val[2] = 2.0; val[3] = 1.0;
    for (i=2; i<n-1; i++ ) {
      col[0] = i-2; col[1] = i-1; col[2] = i; col[3] = i+1;
      ierr = MatSetValues(A,1,&i,4,col,val,INSERT_VALUES);CHKERRQ(ierr);
    }
    i = n-1; col[0] = n-3; col[1] = n-2; col[2] = n-1;
    ierr = MatSetValues(A,1,&i,3,col,val,INSERT_VALUES);CHKERRQ(ierr);
    i = 1; col[0] = 0; col[1] = 1; col[2] = 2;
    ierr = MatSetValues(A,1,&i,3,col,&val[1],INSERT_VALUES);CHKERRQ(ierr);
    i = 0;
    ierr = MatSetValues(A,1,&i,2,col,&val[2],INSERT_VALUES);CHKERRQ(ierr);
  } 
  else if (type == TEST_3) {
    val[0] = PETSC_i * 2.0;
    val[1] = 4.0; val[2] = 0.0; val[3] = 1.0; val[4] = 0.7;
    for (i=1; i<n-3; i++ ) {
      col[0] = i-1; col[1] = i; col[2] = i+1; col[3] = i+2; col[4] = i+3;
      ierr = MatSetValues(A,1,&i,5,col,val,INSERT_VALUES);CHKERRQ(ierr);
    }
    i = n-3; col[0] = n-4; col[1] = n-3; col[2] = n-2; col[3] = n-1;
    ierr = MatSetValues(A,1,&i,4,col,val,INSERT_VALUES);CHKERRQ(ierr);
    i = n-2; col[0] = n-3; col[1] = n-2; col[2] = n-1;
    ierr = MatSetValues(A,1,&i,3,col,val,INSERT_VALUES);CHKERRQ(ierr);
    i = n-1; col[0] = n-2; col[1] = n-1;
    ierr = MatSetValues(A,1,&i,2,col,val,INSERT_VALUES);CHKERRQ(ierr);
    i = 0; col[0] = 0; col[1] = 1; col[2] = 2; col[3] = 3;
    ierr = MatSetValues(A,1,&i,4,col,&val[1],INSERT_VALUES);CHKERRQ(ierr);
  } 
  else if (type == HELMHOLTZ_1) {
    /* Problem domain: unit square: (0,1) x (0,1)
       Solve Helmholtz equation:
          -delta u - sigma1*u + i*sigma2*u = f, 
           where delta = Laplace operator
       Dirichlet b.c.'s on all sides
     */
    PetscRandom rctx;
    double      h2, sigma1 = 5.0;
    Scalar      sigma2;
    ierr = OptionsGetDouble(PETSC_NULL,"-sigma1",&sigma1,&flg);CHKERRA(ierr);
    ierr = PetscRandomCreate(PETSC_COMM_WORLD,RANDOM_DEFAULT_IMAGINARY,&rctx);CHKERRQ(ierr);
    h2 = 1.0/((n+1)*(n+1));
    for ( I=Istart; I<Iend; I++ ) { 
      *val = -1.0; i = I/n; j = I - i*n;  
      if ( i>0 ) {
        J = I-n; ierr = MatSetValues(A,1,&I,1,&J,val,ADD_VALUES);CHKERRQ(ierr);}
      if ( i<n-1 ) {
        J = I+n; ierr = MatSetValues(A,1,&I,1,&J,val,ADD_VALUES);CHKERRQ(ierr);}
      if ( j>0 ) {
        J = I-1; ierr = MatSetValues(A,1,&I,1,&J,val,ADD_VALUES);CHKERRQ(ierr);}
      if ( j<n-1 ) {
        J = I+1; ierr = MatSetValues(A,1,&I,1,&J,val,ADD_VALUES);CHKERRQ(ierr);}
      ierr = PetscRandomGetValue(rctx,&sigma2);CHKERRQ(ierr);
      *val = 4.0 - sigma1*h2 + sigma2*h2;
      ierr = MatSetValues(A,1,&I,1,&I,val,ADD_VALUES);CHKERRQ(ierr);
    }
    ierr = PetscRandomDestroy(rctx);CHKERRQ(ierr);
  }
  else if (type == HELMHOLTZ_2) {
    /* Problem domain: unit square: (0,1) x (0,1)
       Solve Helmholtz equation:
          -delta u - sigma1*u = f, 
           where delta = Laplace operator
       Dirichlet b.c.'s on 3 sides
       du/dn = i*alpha*u on (1,y), 0<y<1
     */
    double  h2, sigma1 = 200.0;
    Scalar alpha_h;
    ierr = OptionsGetDouble(PETSC_NULL,"-sigma1",&sigma1,&flg);CHKERRA(ierr);
    h2 = 1.0/((n+1)*(n+1));
    alpha_h = (PETSC_i * 10.0) / (double)(n+1);  /* alpha_h = alpha * h */
    for ( I=Istart; I<Iend; I++ ) { 
      *val = -1.0; i = I/n; j = I - i*n;  
      if ( i>0 ) {
        J = I-n; ierr = MatSetValues(A,1,&I,1,&J,val,ADD_VALUES);CHKERRQ(ierr);}
      if ( i<n-1 ) {
        J = I+n; ierr = MatSetValues(A,1,&I,1,&J,val,ADD_VALUES);CHKERRQ(ierr);}
      if ( j>0 ) {
        J = I-1; ierr = MatSetValues(A,1,&I,1,&J,val,ADD_VALUES);CHKERRQ(ierr);}
      if ( j<n-1 ) {
        J = I+1; ierr = MatSetValues(A,1,&I,1,&J,val,ADD_VALUES);CHKERRQ(ierr);}
      *val = 4.0 - sigma1*h2;
      if (!((I+1)%n)) *val += alpha_h;
      ierr = MatSetValues(A,1,&I,1,&I,val,ADD_VALUES);CHKERRQ(ierr);
    }
  }
  else SETERRQ(1,0,"FormTestMatrix: unknown test matrix type");

  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
#endif

  return 0;
}
