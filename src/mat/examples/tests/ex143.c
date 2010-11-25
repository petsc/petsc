
#include <stdlib.h>

#include <petsc.h>
#include <petscvec.h>
#include <petscmat.h>


static char help[] = "Newton's method to solve a two-variable system, sequentially.\n"\
  "The same problem is solved twice - i) fully assembled system + ii) block system\n\n";

/*T
Concepts: SNES^basic uniprocessor example, block objects
Processors: 1
T*/

/* 
Include "petscsnes.h" so that we can use SNES solvers.  Note that this
file automatically includes:
petsc.h       - base PETSc routines   petscvec.h - vectors
petscsys.h    - system routines       petscmat.h - matrices
petscis.h     - index sets            petscksp.h - Krylov subspace methods
petscviewer.h - viewers               petscpc.h  - preconditioners
petscksp.h   - linear solvers
petscvec-block.h - block definition for vectors
petscmat-block.h - block definition for matrics
*/

/*
This example is block version of the test found at 
  ${PETSC_DIR}/src/snes/examples/tutorials/ex1.c
In this test we replace the Jacobian systems
  [J]{x} = {F} 
where

[J] = ( j_00, j_01 ),  {x} = ( x_0, x_1 )^T,   {F} = ( f_0, f_1 )^T
      ( j_10, j_11 )      
where [J] \in \mathbb^{2 \times 2}, {x},{F} \in \mathbb^{2 \times 1},

with a block system in which each block is of length 1.
i.e. The block system is thus

[J] = ( [j00], [j01] ),  {x} = ( {x0}, {x1} )^T, {F} = ( {f0}, {f1} )^T
      ( [j10], [j11] ) 
where 
[j00], [j01], [j10], [j11] \in \mathbb^{1 \times 1}
{x0}, {x1}, {f0}, {f1} \in \mathbb^{1 \times 1}

In practise we would not bother defing blocks of size one, and would instead assemble the 
full system. This is just a simple test to illustrate how to manipulate the blocks and
to confirm the implementation is correct.
*/



#include "petscsnes.h"

/* 
User-defined routines
*/
extern PetscErrorCode FormJacobian1(SNES,Vec,Mat*,Mat*,MatStructure*,void*);
extern PetscErrorCode FormFunction1(SNES,Vec,Vec,void*);
extern PetscErrorCode FormJacobian2(SNES,Vec,Mat*,Mat*,MatStructure*,void*);
extern PetscErrorCode FormFunction2(SNES,Vec,Vec,void*);

extern PetscErrorCode FormJacobian1_block(SNES,Vec,Mat*,Mat*,MatStructure*,void*);
extern PetscErrorCode FormFunction1_block(SNES,Vec,Vec,void*);
extern PetscErrorCode FormJacobian2_block(SNES,Vec,Mat*,Mat*,MatStructure*,void*);
extern PetscErrorCode FormFunction2_block(SNES,Vec,Vec,void*);


#undef __FUNCT__
#define __FUNCT__ "assembled_system"
int assembled_system(int argc,char **argv)
{
  SNES           snes;         /* nonlinear solver context */
  KSP            ksp;         /* linear solver context */
  PC             pc;           /* preconditioner context */
  Vec            x,r;         /* solution, residual vectors */
  Mat            J;            /* Jacobian matrix */
  PetscErrorCode ierr;
  PetscInt       its;
  PetscScalar    pfive = .5,*xx;
  PetscBool      flg;
  
  
  PetscPrintf( PETSC_COMM_WORLD, "\n\n========================= Assembled system =========================\n\n" );
  
  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  Create nonlinear solver context
  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  
  ierr = SNESCreate(PETSC_COMM_WORLD,&snes);CHKERRQ(ierr);
  
  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  Create matrix and vector data structures; set corresponding routines
  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  
  /*
  Create vectors for solution and nonlinear function
  */
  ierr = VecCreateSeq(PETSC_COMM_SELF,2,&x);CHKERRQ(ierr);
  ierr = VecDuplicate(x,&r);CHKERRQ(ierr);
  
  /*
  Create Jacobian matrix data structure
  */
  ierr = MatCreate(PETSC_COMM_SELF,&J);CHKERRQ(ierr);
  ierr = MatSetSizes(J,PETSC_DECIDE,PETSC_DECIDE,2,2);CHKERRQ(ierr);
  ierr = MatSetFromOptions(J);CHKERRQ(ierr);
  
  ierr = PetscOptionsHasName(PETSC_NULL,"-hard",&flg);CHKERRQ(ierr);
  if (!flg) {
    /* 
    Set function evaluation routine and vector.
    */
    ierr = SNESSetFunction(snes,r,FormFunction1,PETSC_NULL);CHKERRQ(ierr);
    
    /* 
    Set Jacobian matrix data structure and Jacobian evaluation routine
    */
    ierr = SNESSetJacobian(snes,J,J,FormJacobian1,PETSC_NULL);CHKERRQ(ierr);
  } else {
    ierr = SNESSetFunction(snes,r,FormFunction2,PETSC_NULL);CHKERRQ(ierr);
    ierr = SNESSetJacobian(snes,J,J,FormJacobian2,PETSC_NULL);CHKERRQ(ierr);
  }
  
  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  Customize nonlinear solver; set runtime options
  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  
  /* 
  Set linear solver defaults for this problem. By extracting the
  KSP, KSP, and PC contexts from the SNES context, we can then
  directly call any KSP, KSP, and PC routines to set various options.
  */
  ierr = SNESGetKSP(snes,&ksp);CHKERRQ(ierr);
  ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
  ierr = PCSetType(pc,PCNONE);CHKERRQ(ierr);
  ierr = KSPSetTolerances(ksp,1.e-4,PETSC_DEFAULT,PETSC_DEFAULT,20);CHKERRQ(ierr);
  
  /* 
  Set SNES/KSP/KSP/PC runtime options, e.g.,
  -snes_view -snes_monitor -ksp_type <ksp> -pc_type <pc>
  These options will override those specified above as long as
  SNESSetFromOptions() is called _after_ any other customization
  routines.
  */
  ierr = SNESSetFromOptions(snes);CHKERRQ(ierr);
  
  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  Evaluate initial guess; then solve nonlinear system
  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  if (!flg) {
    ierr = VecSet(x,pfive);CHKERRQ(ierr);
  } else {
    ierr = VecGetArray(x,&xx);CHKERRQ(ierr);
    xx[0] = 2.0; xx[1] = 3.0;
    ierr = VecRestoreArray(x,&xx);CHKERRQ(ierr);
  }
  /*
  Note: The user should initialize the vector, x, with the initial guess
  for the nonlinear solver prior to calling SNESSolve().  In particular,
  to employ an initial guess of zero, the user should explicitly set
  this vector to zero by calling VecSet().
  */
  
  ierr = SNESSolve(snes,PETSC_NULL,x);CHKERRQ(ierr);
  ierr = SNESGetIterationNumber(snes,&its);CHKERRQ(ierr);
  if (flg) {
    Vec f;
    ierr = VecView(x,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    ierr = SNESGetFunction(snes,&f,0,0);CHKERRQ(ierr);
    ierr = VecView(r,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  }
  
  ierr = PetscPrintf(PETSC_COMM_SELF,"number of Newton iterations = %D\n\n",its);CHKERRQ(ierr);
  
  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  Free work space.  All PETSc objects should be destroyed when they
  are no longer needed.
  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  
  ierr = VecDestroy(x);CHKERRQ(ierr); ierr = VecDestroy(r);CHKERRQ(ierr);
  ierr = MatDestroy(J);CHKERRQ(ierr); ierr = SNESDestroy(snes);CHKERRQ(ierr);
  
  return 0;
}
/* ------------------------------------------------------------------- */
#undef __FUNCT__
#define __FUNCT__ "FormFunction1"
/* 
FormFunction1 - Evaluates nonlinear function, F(x).

Input Parameters:
.  snes - the SNES context
.  x - input vector
.  dummy - optional user-defined context (not used here)

Output Parameter:
.  f - function vector
*/
PetscErrorCode FormFunction1(SNES snes,Vec x,Vec f,void *dummy)
{
  PetscErrorCode ierr;
  PetscScalar    *xx,*ff;
  
  /*
  Get pointers to vector data.
  - For default PETSc vectors, VecGetArray() returns a pointer to
  the data array.  Otherwise, the routine is implementation dependent.
  - You MUST call VecRestoreArray() when you no longer need access to
  the array.
  */
  ierr = VecGetArray(x,&xx);CHKERRQ(ierr);
  ierr = VecGetArray(f,&ff);CHKERRQ(ierr);
  
  /*
  Compute function
  */
  ff[0] = xx[0]*xx[0] + xx[0]*xx[1] - 3.0;
  ff[1] = xx[0]*xx[1] + xx[1]*xx[1] - 6.0;
  
  
  /*
  Restore vectors
  */
  ierr = VecRestoreArray(x,&xx);CHKERRQ(ierr);
  ierr = VecRestoreArray(f,&ff);CHKERRQ(ierr); 
  
  return 0;
}
/* ------------------------------------------------------------------- */
#undef __FUNCT__
#define __FUNCT__ "FormJacobian1"
/*
FormJacobian1 - Evaluates Jacobian matrix.

Input Parameters:
.  snes - the SNES context
.  x - input vector
.  dummy - optional user-defined context (not used here)

Output Parameters:
.  jac - Jacobian matrix
.  B - optionally different preconditioning matrix
.  flag - flag indicating matrix structure
*/
PetscErrorCode FormJacobian1(SNES snes,Vec x,Mat *jac,Mat *B,MatStructure *flag,void *dummy)
{
  PetscScalar    *xx,A[4];
  PetscErrorCode ierr;
  PetscInt       idx[2] = {0,1};
  
  /*
  Get pointer to vector data
  */
  ierr = VecGetArray(x,&xx);CHKERRQ(ierr);
  
  /*
  Compute Jacobian entries and insert into matrix.
  - Since this is such a small problem, we set all entries for
  the matrix at once.
  */
  A[0] = 2.0*xx[0] + xx[1]; A[1] = xx[0];
  A[2] = xx[1]; A[3] = xx[0] + 2.0*xx[1];
  ierr = MatSetValues(*jac,2,idx,2,idx,A,INSERT_VALUES);CHKERRQ(ierr);
  *flag = SAME_NONZERO_PATTERN;
  
  /*
  Restore vector
  */
  ierr = VecRestoreArray(x,&xx);CHKERRQ(ierr);
  
  /*
  Assemble matrix
  */
  ierr = MatAssemblyBegin(*jac,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*jac,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  
  return 0;
}


/* ------------------------------------------------------------------- */
#undef __FUNCT__
#define __FUNCT__ "FormFunction2"
PetscErrorCode FormFunction2(SNES snes,Vec x,Vec f,void *dummy)
{
  PetscErrorCode ierr;
  PetscScalar    *xx,*ff;
  
  /*
  Get pointers to vector data.
  - For default PETSc vectors, VecGetArray() returns a pointer to
  the data array.  Otherwise, the routine is implementation dependent.
  - You MUST call VecRestoreArray() when you no longer need access to
  the array.
  */
  ierr = VecGetArray(x,&xx);CHKERRQ(ierr);
  ierr = VecGetArray(f,&ff);CHKERRQ(ierr);
  
  /*
  Compute function
  */
  ff[0] = PetscSinScalar(3.0*xx[0]) + xx[0];
  ff[1] = xx[1];
  
  /*
  Restore vectors
  */
  ierr = VecRestoreArray(x,&xx);CHKERRQ(ierr);
  ierr = VecRestoreArray(f,&ff);CHKERRQ(ierr); 
  
  return 0;
}
/* ------------------------------------------------------------------- */
#undef __FUNCT__
#define __FUNCT__ "FormJacobian2"
PetscErrorCode FormJacobian2(SNES snes,Vec x,Mat *jac,Mat *B,MatStructure *flag,void *dummy)
{
  PetscScalar    *xx,A[4];
  PetscErrorCode ierr;
  PetscInt       idx[2] = {0,1};
  
  /*
  Get pointer to vector data
  */
  ierr = VecGetArray(x,&xx);CHKERRQ(ierr);
  
  /*
  Compute Jacobian entries and insert into matrix.
  - Since this is such a small problem, we set all entries for
  the matrix at once.
  */
  A[0] = 3.0*PetscCosScalar(3.0*xx[0]) + 1.0; A[1] = 0.0;
  A[2] = 0.0;                     A[3] = 1.0;
  ierr = MatSetValues(*jac,2,idx,2,idx,A,INSERT_VALUES);CHKERRQ(ierr);
  *flag = SAME_NONZERO_PATTERN;
  
  /*
  Restore vector
  */
  ierr = VecRestoreArray(x,&xx);CHKERRQ(ierr);
  
  /*
  Assemble matrix
  */
  ierr = MatAssemblyBegin(*jac,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*jac,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  
  return 0;
}

#undef __FUNCT__
#define __FUNCT__ "block_system"
int block_system(int argc,char **argv)
{
  SNES           snes;         /* nonlinear solver context */
  KSP            ksp;         /* linear solver context */
  PC             pc;           /* preconditioner context */
  Vec            x,r;         /* solution, residual vectors */
  Mat            J;            /* Jacobian matrix */
  PetscErrorCode ierr;
  PetscInt       its;
  PetscScalar    pfive = .5;
  PetscBool      flg;
  
  Mat j11, j12, j21, j22;
  Vec x1, x2, r1, r2;
  Vec bv;
  Vec bx[2];
  Mat *bA[2],_bA[2][2];
  
  PetscPrintf( PETSC_COMM_WORLD, "\n\n========================= Block system =========================\n\n" );
  
  ierr = SNESCreate(PETSC_COMM_WORLD,&snes);CHKERRQ(ierr);
  
  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  Create matrix and vector data structures; set corresponding routines
  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  
  /*
  Create sub vectors for solution and nonlinear function
  */
  ierr = VecCreateSeq(PETSC_COMM_SELF,1,&x1);CHKERRQ(ierr);
  ierr = VecDuplicate(x1,&r1);CHKERRQ(ierr);
  
  ierr = VecCreateSeq(PETSC_COMM_SELF,1,&x2);CHKERRQ(ierr);
  ierr = VecDuplicate(x2,&r2);CHKERRQ(ierr);
  
  /*
  Create the block vectors
  */
  bx[0] = x1;
  bx[1] = x2;
  ierr = VecCreateNest(PETSC_COMM_WORLD,2,PETSC_NULL,bx,&x);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(x);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(x);CHKERRQ(ierr);
  ierr = VecDestroy(x1);CHKERRQ(ierr);
  ierr = VecDestroy(x2);CHKERRQ(ierr);

  bx[0] = r1;
  bx[1] = r2;
  ierr = VecCreateNest(PETSC_COMM_WORLD,2,PETSC_NULL,bx,&r);CHKERRQ(ierr);
  ierr = VecDestroy(r1);CHKERRQ(ierr);
  ierr = VecDestroy(r2);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(r);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(r);CHKERRQ(ierr);

  /*
  Create sub Jacobian matrix data structure
  */
  ierr = MatCreate( PETSC_COMM_WORLD, &j11 );CHKERRQ(ierr);
  ierr = MatSetSizes( j11, 1, 1, 1, 1 );CHKERRQ(ierr);
  ierr = MatSetType( j11, MATSEQAIJ );CHKERRQ(ierr);
  
  ierr = MatCreate( PETSC_COMM_WORLD, &j12 );CHKERRQ(ierr);
  ierr = MatSetSizes( j12, 1, 1, 1, 1 );CHKERRQ(ierr);
  ierr = MatSetType( j12, MATSEQAIJ );CHKERRQ(ierr);
  
  ierr = MatCreate( PETSC_COMM_WORLD, &j21 );CHKERRQ(ierr);
  ierr = MatSetSizes( j21, 1, 1, 1, 1 );CHKERRQ(ierr);
  ierr = MatSetType( j21, MATSEQAIJ );CHKERRQ(ierr);
  
  ierr = MatCreate( PETSC_COMM_WORLD, &j22 );CHKERRQ(ierr);
  ierr = MatSetSizes( j22, PETSC_DECIDE, PETSC_DECIDE, 1, 1 );CHKERRQ(ierr);
  ierr = MatSetType( j22, MATSEQAIJ );CHKERRQ(ierr);
  
  /*
  Create block Jacobian matrix data structure
  */
  bA[0] = _bA[0];
  bA[1] = _bA[1];
  bA[0][0] = j11;
  bA[0][1] = j12;
  bA[1][0] = j21;
  bA[1][1] = j22;
  ierr = MatCreateNest(PETSC_COMM_WORLD,2,2,PETSC_NULL,PETSC_NULL,bA,&J);CHKERRQ(ierr);
  ierr = MatDestroy(j11);CHKERRQ(ierr);
  ierr = MatDestroy(j12);CHKERRQ(ierr);
  ierr = MatDestroy(j21);CHKERRQ(ierr);
  ierr = MatDestroy(j22);CHKERRQ(ierr);

  ierr = MatAssemblyBegin(J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  
  ierr = PetscOptionsHasName(PETSC_NULL,"-hard",&flg);CHKERRQ(ierr);
  if (!flg) {
    /* 
    Set function evaluation routine and vector.
    */
    ierr = SNESSetFunction(snes,r,FormFunction1_block,PETSC_NULL);CHKERRQ(ierr);
    
    /* 
    Set Jacobian matrix data structure and Jacobian evaluation routine
    */
    ierr = SNESSetJacobian(snes,J,J,FormJacobian1_block,PETSC_NULL);CHKERRQ(ierr);
  } else {
    ierr = SNESSetFunction(snes,r,FormFunction2_block,PETSC_NULL);CHKERRQ(ierr);
    ierr = SNESSetJacobian(snes,J,J,FormJacobian2_block,PETSC_NULL);CHKERRQ(ierr);
  }
  
  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  Customize nonlinear solver; set runtime options
  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  
  /* 
  Set linear solver defaults for this problem. By extracting the
  KSP, KSP, and PC contexts from the SNES context, we can then
  directly call any KSP, KSP, and PC routines to set various options.
  */
  ierr = SNESGetKSP(snes,&ksp);CHKERRQ(ierr);
  ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
  ierr = PCSetType(pc,PCNONE);CHKERRQ(ierr);
  ierr = KSPSetTolerances(ksp,1.e-4,PETSC_DEFAULT,PETSC_DEFAULT,20);CHKERRQ(ierr);
  
  /* 
  Set SNES/KSP/KSP/PC runtime options, e.g.,
  -snes_view -snes_monitor -ksp_type <ksp> -pc_type <pc>
  These options will override those specified above as long as
  SNESSetFromOptions() is called _after_ any other customization
  routines.
  */
  ierr = SNESSetFromOptions(snes);CHKERRQ(ierr);
  
  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  Evaluate initial guess; then solve nonlinear system
  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  if (!flg) {
    ierr = VecSet(x,pfive);CHKERRQ(ierr);
  } else {
    Vec *vecs;
    ierr = VecNestGetSubVecs( x, PETSC_NULL, &vecs );CHKERRQ(ierr);
    bv = vecs[0];
//    ierr = VecBlockGetSubVec( x, 0, &bv );CHKERRQ(ierr);
    ierr = VecSetValue( bv, 0, 2.0, INSERT_VALUES );CHKERRQ(ierr);  /* xx[0] = 2.0; */
    ierr = VecAssemblyBegin(bv);CHKERRQ(ierr);
    ierr = VecAssemblyEnd(bv);CHKERRQ(ierr);
    
//    ierr = VecBlockGetSubVec( x, 1, &bv );CHKERRQ(ierr);
    bv = vecs[1];
    ierr = VecSetValue( bv, 0, 3.0, INSERT_VALUES );CHKERRQ(ierr);  /* xx[1] = 3.0; */
    ierr = VecAssemblyBegin(bv);CHKERRQ(ierr);  
    ierr = VecAssemblyEnd(bv);CHKERRQ(ierr);
  }
  /*
  Note: The user should initialize the vector, x, with the initial guess
  for the nonlinear solver prior to calling SNESSolve().  In particular,
  to employ an initial guess of zero, the user should explicitly set
  this vector to zero by calling VecSet().
  */
  ierr = SNESSolve(snes,PETSC_NULL,x);CHKERRQ(ierr);
  ierr = SNESGetIterationNumber(snes,&its);CHKERRQ(ierr);
  if (flg) {
    Vec f;
    ierr = VecView(x,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    ierr = SNESGetFunction(snes,&f,0,0);CHKERRQ(ierr);
    ierr = VecView(r,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  }
  
  ierr = PetscPrintf(PETSC_COMM_SELF,"number of Newton iterations = %D\n\n",its);CHKERRQ(ierr);
  
  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  Free work space.  All PETSc objects should be destroyed when they
  are no longer needed.
  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = VecDestroy(x);CHKERRQ(ierr); ierr = VecDestroy(r);CHKERRQ(ierr);
  ierr = MatDestroy(J);CHKERRQ(ierr); ierr = SNESDestroy(snes);CHKERRQ(ierr);
  
  return 0;
}
/* ------------------------------------------------------------------- */
#undef __FUNCT__
#define __FUNCT__ "FormFunction1_block"
PetscErrorCode FormFunction1_block(SNES snes,Vec x,Vec f,void *dummy)
{
  Vec *xx, *ff, x1,x2, f1,f2;
  PetscScalar ff_0, ff_1;
  PetscScalar xx_0, xx_1;
  PetscInt index,nb;
  PetscErrorCode ierr;
  
  /* get blocks for function */
  ierr = VecNestGetSubVecs( f, &nb, &ff );CHKERRQ(ierr);
  f1 = ff[0];  f2 = ff[1];
  
  /* get blocks for solution */
  ierr = VecNestGetSubVecs( x, &nb, &xx );CHKERRQ(ierr);
  x1 = xx[0];  x2 = xx[1];
  
  /* get solution values */
  index = 0;
  ierr = VecGetValues( x1,1, &index, &xx_0 );CHKERRQ(ierr);
  ierr = VecGetValues( x2,1, &index, &xx_1 );CHKERRQ(ierr);
  
  /* Compute function */
  ff_0 = xx_0*xx_0 + xx_0*xx_1 - 3.0;
  ff_1 = xx_0*xx_1 + xx_1*xx_1 - 6.0;
  
  /* set function values */
  ierr = VecSetValue( f1, index, ff_0, INSERT_VALUES );CHKERRQ(ierr);
  
  ierr = VecSetValue( f2, index, ff_1, INSERT_VALUES );CHKERRQ(ierr);

  ierr = VecAssemblyBegin(f);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(f);CHKERRQ(ierr);

  return 0;
}
/* ------------------------------------------------------------------- */
#undef __FUNCT__
#define __FUNCT__ "FormJacobian1_block"
PetscErrorCode FormJacobian1_block(SNES snes,Vec x,Mat *jac,Mat *B,MatStructure *flag,void *dummy)
{
  Vec *xx, x1,x2;
  PetscScalar xx_0, xx_1;
  PetscInt index,nb;
  PetscScalar A_00, A_01, A_10, A_11;
  Mat j11, j12, j21, j22;
  Mat **mats;
  PetscErrorCode ierr;

  /* get blocks for solution */
  ierr = VecNestGetSubVecs( x, &nb, &xx );CHKERRQ(ierr);
  x1 = xx[0];  x2 = xx[1];
  
  /* get solution values */
  index = 0;
  ierr = VecGetValues( x1,1, &index, &xx_0 );CHKERRQ(ierr);
  ierr = VecGetValues( x2,1, &index, &xx_1 );CHKERRQ(ierr);
  
  /* get block matrices */
  ierr = MatNestGetSubMats(*jac,PETSC_NULL,PETSC_NULL,&mats);CHKERRQ(ierr);
  j11 = mats[0][0];
  j12 = mats[0][1];
  j21 = mats[1][0];
  j22 = mats[1][1];
  
  /* compute jacobian entries */
  A_00 = 2.0*xx_0 + xx_1;
  A_01 = xx_0;
  A_10 = xx_1;
  A_11 = xx_0 + 2.0*xx_1;
  
  /* set jacobian values */
  ierr = MatSetValue( j11, 0,0, A_00, INSERT_VALUES);CHKERRQ(ierr);
  ierr = MatSetValue( j12, 0,0, A_01, INSERT_VALUES);CHKERRQ(ierr);
  ierr = MatSetValue( j21, 0,0, A_10, INSERT_VALUES);CHKERRQ(ierr);
  ierr = MatSetValue( j22, 0,0, A_11, INSERT_VALUES);CHKERRQ(ierr);
  
  
  
  *flag = SAME_NONZERO_PATTERN;
  
  /* Assemble sub matrix */
  ierr = MatAssemblyBegin(*jac,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*jac,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  return 0;
}


/* ------------------------------------------------------------------- */
#undef __FUNCT__
#define __FUNCT__ "FormFunction2_block"
PetscErrorCode FormFunction2_block(SNES snes,Vec x,Vec f,void *dummy)
{
  PetscErrorCode ierr;
  PetscScalar    *xx,*ff;
  
  /*
  Get pointers to vector data.
  - For default PETSc vectors, VecGetArray() returns a pointer to
  the data array.  Otherwise, the routine is implementation dependent.
  - You MUST call VecRestoreArray() when you no longer need access to
  the array.
  */
  ierr = VecGetArray(x,&xx);CHKERRQ(ierr);
  ierr = VecGetArray(f,&ff);CHKERRQ(ierr);
  
  /*
  Compute function
  */
  ff[0] = PetscSinScalar(3.0*xx[0]) + xx[0];
  ff[1] = xx[1];
  
  /*
  Restore vectors
  */
  ierr = VecRestoreArray(x,&xx);CHKERRQ(ierr);
  ierr = VecRestoreArray(f,&ff);CHKERRQ(ierr); 
  
  return 0;
}
/* ------------------------------------------------------------------- */
#undef __FUNCT__
#define __FUNCT__ "FormJacobian2_block"
PetscErrorCode FormJacobian2_block(SNES snes,Vec x,Mat *jac,Mat *B,MatStructure *flag,void *dummy)
{
  PetscScalar    *xx,A[4];
  PetscErrorCode ierr;
  PetscInt       idx[2] = {0,1};
  
  /*
  Get pointer to vector data
  */
  ierr = VecGetArray(x,&xx);CHKERRQ(ierr);
  
  /*
  Compute Jacobian entries and insert into matrix.
  - Since this is such a small problem, we set all entries for
  the matrix at once.
  */
  A[0] = 3.0*PetscCosScalar(3.0*xx[0]) + 1.0; A[1] = 0.0;
  A[2] = 0.0;                     A[3] = 1.0;
  ierr = MatSetValues(*jac,2,idx,2,idx,A,INSERT_VALUES);CHKERRQ(ierr);
  *flag = SAME_NONZERO_PATTERN;
  
  /*
  Restore vector
  */
  ierr = VecRestoreArray(x,&xx);CHKERRQ(ierr);
  
  /*
  Assemble matrix
  */
  ierr = MatAssemblyBegin(*jac,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*jac,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  
  return 0;
}

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  PetscMPIInt    size;
  PetscErrorCode ierr;
  
  PetscInitialize(&argc,&argv,(char *)0,help);
  
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
  if (size != 1) SETERRQ(PETSC_COMM_WORLD, 1,"This is a uniprocessor example only!");
  
  /* force snes monitor to be on */
  PetscOptionsInsertString( "-snes_monitor" );
  
  assembled_system(argc,argv);
  
  block_system(argc,argv);
  
  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}
