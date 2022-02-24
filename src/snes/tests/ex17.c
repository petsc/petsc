static const char help[] = "Newton's method to solve a two-variable system, sequentially.\n"
                           "The same problem is solved twice - i) fully assembled system + ii) block system\n\n";

/*T
Concepts: SNES^basic uniprocessor example, block objects
Processors: 1
T*/

/*
Include "petscsnes.h" so that we can use SNES solvers.  Note that this
file automatically includes:
petscsys.h       - base PETSc routines   petscvec.h - vectors
petscsys.h    - system routines       petscmat.h - matrices
petscis.h     - index sets            petscksp.h - Krylov subspace methods
petscviewer.h - viewers               petscpc.h  - preconditioners
petscksp.h   - linear solvers
*/
#include <petscsnes.h>

/*
This example is block version of the test found at
  ${PETSC_DIR}/src/snes/tutorials/ex1.c
In this test we replace the Jacobian systems
  [J]{x} = {F}
where

[J] = (j_00, j_01),  {x} = (x_0, x_1)^T,   {F} = (f_0, f_1)^T
      (j_10, j_11)
where [J] \in \mathbb^{2 \times 2}, {x},{F} \in \mathbb^{2 \times 1},

with a block system in which each block is of length 1.
i.e. The block system is thus

[J] = ([j00], [j01]),  {x} = ({x0}, {x1})^T, {F} = ({f0}, {f1})^T
      ([j10], [j11])
where
[j00], [j01], [j10], [j11] \in \mathbb^{1 \times 1}
{x0}, {x1}, {f0}, {f1} \in \mathbb^{1 \times 1}

In practice we would not bother defing blocks of size one, and would instead assemble the
full system. This is just a simple test to illustrate how to manipulate the blocks and
to confirm the implementation is correct.
*/

/*
User-defined routines
*/
static PetscErrorCode FormJacobian1(SNES,Vec,Mat,Mat,void*);
static PetscErrorCode FormFunction1(SNES,Vec,Vec,void*);
static PetscErrorCode FormJacobian2(SNES,Vec,Mat,Mat,void*);
static PetscErrorCode FormFunction2(SNES,Vec,Vec,void*);
static PetscErrorCode FormJacobian1_block(SNES,Vec,Mat,Mat,void*);
static PetscErrorCode FormFunction1_block(SNES,Vec,Vec,void*);
static PetscErrorCode FormJacobian2_block(SNES,Vec,Mat,Mat,void*);
static PetscErrorCode FormFunction2_block(SNES,Vec,Vec,void*);

static PetscErrorCode assembled_system(void)
{
  SNES           snes;         /* nonlinear solver context */
  KSP            ksp;         /* linear solver context */
  PC             pc;           /* preconditioner context */
  Vec            x,r;         /* solution, residual vectors */
  Mat            J;            /* Jacobian matrix */
  PetscInt       its;
  PetscScalar    pfive = .5,*xx;
  PetscBool      flg;

  PetscFunctionBeginUser;
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD, "\n\n========================= Assembled system =========================\n\n"));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  Create nonlinear solver context
  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  CHKERRQ(SNESCreate(PETSC_COMM_WORLD,&snes));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  Create matrix and vector data structures; set corresponding routines
  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /*
  Create vectors for solution and nonlinear function
  */
  CHKERRQ(VecCreateSeq(PETSC_COMM_SELF,2,&x));
  CHKERRQ(VecDuplicate(x,&r));

  /*
  Create Jacobian matrix data structure
  */
  CHKERRQ(MatCreate(PETSC_COMM_SELF,&J));
  CHKERRQ(MatSetSizes(J,PETSC_DECIDE,PETSC_DECIDE,2,2));
  CHKERRQ(MatSetFromOptions(J));
  CHKERRQ(MatSetUp(J));

  CHKERRQ(PetscOptionsHasName(NULL,NULL,"-hard",&flg));
  if (!flg) {
    /*
    Set function evaluation routine and vector.
    */
    CHKERRQ(SNESSetFunction(snes,r,FormFunction1,NULL));

    /*
    Set Jacobian matrix data structure and Jacobian evaluation routine
    */
    CHKERRQ(SNESSetJacobian(snes,J,J,FormJacobian1,NULL));
  } else {
    CHKERRQ(SNESSetFunction(snes,r,FormFunction2,NULL));
    CHKERRQ(SNESSetJacobian(snes,J,J,FormJacobian2,NULL));
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  Customize nonlinear solver; set runtime options
  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /*
  Set linear solver defaults for this problem. By extracting the
  KSP, KSP, and PC contexts from the SNES context, we can then
  directly call any KSP, KSP, and PC routines to set various options.
  */
  CHKERRQ(SNESGetKSP(snes,&ksp));
  CHKERRQ(KSPGetPC(ksp,&pc));
  CHKERRQ(PCSetType(pc,PCNONE));
  CHKERRQ(KSPSetTolerances(ksp,1.e-4,PETSC_DEFAULT,PETSC_DEFAULT,20));

  /*
  Set SNES/KSP/KSP/PC runtime options, e.g.,
  -snes_view -snes_monitor -ksp_type <ksp> -pc_type <pc>
  These options will override those specified above as long as
  SNESSetFromOptions() is called _after_ any other customization
  routines.
  */
  CHKERRQ(SNESSetFromOptions(snes));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  Evaluate initial guess; then solve nonlinear system
  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  if (!flg) {
    CHKERRQ(VecSet(x,pfive));
  } else {
    CHKERRQ(VecGetArray(x,&xx));
    xx[0] = 2.0; xx[1] = 3.0;
    CHKERRQ(VecRestoreArray(x,&xx));
  }
  /*
  Note: The user should initialize the vector, x, with the initial guess
  for the nonlinear solver prior to calling SNESSolve().  In particular,
  to employ an initial guess of zero, the user should explicitly set
  this vector to zero by calling VecSet().
  */

  CHKERRQ(SNESSolve(snes,NULL,x));
  CHKERRQ(SNESGetIterationNumber(snes,&its));
  if (flg) {
    Vec f;
    CHKERRQ(VecView(x,PETSC_VIEWER_STDOUT_WORLD));
    CHKERRQ(SNESGetFunction(snes,&f,0,0));
    CHKERRQ(VecView(r,PETSC_VIEWER_STDOUT_WORLD));
  }
  CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"number of SNES iterations = %D\n\n",its));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  Free work space.  All PETSc objects should be destroyed when they
  are no longer needed.
  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  CHKERRQ(VecDestroy(&x)); CHKERRQ(VecDestroy(&r));
  CHKERRQ(MatDestroy(&J)); CHKERRQ(SNESDestroy(&snes));
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------------- */
/*
FormFunction1 - Evaluates nonlinear function, F(x).

Input Parameters:
.  snes - the SNES context
.  x - input vector
.  dummy - optional user-defined context (not used here)

Output Parameter:
.  f - function vector
*/
static PetscErrorCode FormFunction1(SNES snes,Vec x,Vec f,void *dummy)
{
  const PetscScalar *xx;
  PetscScalar       *ff;

  PetscFunctionBeginUser;
  /*
  Get pointers to vector data.
  - For default PETSc vectors, VecGetArray() returns a pointer to
  the data array.  Otherwise, the routine is implementation dependent.
  - You MUST call VecRestoreArray() when you no longer need access to
  the array.
  */
  CHKERRQ(VecGetArrayRead(x,&xx));
  CHKERRQ(VecGetArray(f,&ff));

  /*
  Compute function
  */
  ff[0] = xx[0]*xx[0] + xx[0]*xx[1] - 3.0;
  ff[1] = xx[0]*xx[1] + xx[1]*xx[1] - 6.0;

  /*
  Restore vectors
  */
  CHKERRQ(VecRestoreArrayRead(x,&xx));
  CHKERRQ(VecRestoreArray(f,&ff));
  PetscFunctionReturn(0);
}
/* ------------------------------------------------------------------- */
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
static PetscErrorCode FormJacobian1(SNES snes,Vec x,Mat jac,Mat B,void *dummy)
{
  const PetscScalar *xx;
  PetscScalar       A[4];
  PetscInt          idx[2] = {0,1};

  PetscFunctionBeginUser;
  /*
  Get pointer to vector data
  */
  CHKERRQ(VecGetArrayRead(x,&xx));

  /*
  Compute Jacobian entries and insert into matrix.
  - Since this is such a small problem, we set all entries for
  the matrix at once.
  */
  A[0]  = 2.0*xx[0] + xx[1]; A[1] = xx[0];
  A[2]  = xx[1]; A[3] = xx[0] + 2.0*xx[1];
  CHKERRQ(MatSetValues(jac,2,idx,2,idx,A,INSERT_VALUES));

  /*
  Restore vector
  */
  CHKERRQ(VecRestoreArrayRead(x,&xx));

  /*
  Assemble matrix
  */
  CHKERRQ(MatAssemblyBegin(jac,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(jac,MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------------- */
static PetscErrorCode FormFunction2(SNES snes,Vec x,Vec f,void *dummy)
{
  const PetscScalar *xx;
  PetscScalar       *ff;

  PetscFunctionBeginUser;
  /*
  Get pointers to vector data.
  - For default PETSc vectors, VecGetArray() returns a pointer to
  the data array.  Otherwise, the routine is implementation dependent.
  - You MUST call VecRestoreArray() when you no longer need access to
  the array.
  */
  CHKERRQ(VecGetArrayRead(x,&xx));
  CHKERRQ(VecGetArray(f,&ff));

  /*
  Compute function
  */
  ff[0] = PetscSinScalar(3.0*xx[0]) + xx[0];
  ff[1] = xx[1];

  /*
  Restore vectors
  */
  CHKERRQ(VecRestoreArrayRead(x,&xx));
  CHKERRQ(VecRestoreArray(f,&ff));
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------------- */
static PetscErrorCode FormJacobian2(SNES snes,Vec x,Mat jac,Mat B,void *dummy)
{
  const PetscScalar *xx;
  PetscScalar       A[4];
  PetscInt          idx[2] = {0,1};

  PetscFunctionBeginUser;
  /*
  Get pointer to vector data
  */
  CHKERRQ(VecGetArrayRead(x,&xx));

  /*
  Compute Jacobian entries and insert into matrix.
  - Since this is such a small problem, we set all entries for
  the matrix at once.
  */
  A[0]  = 3.0*PetscCosScalar(3.0*xx[0]) + 1.0; A[1] = 0.0;
  A[2]  = 0.0;                     A[3] = 1.0;
  CHKERRQ(MatSetValues(jac,2,idx,2,idx,A,INSERT_VALUES));

  /*
  Restore vector
  */
  CHKERRQ(VecRestoreArrayRead(x,&xx));

  /*
  Assemble matrix
  */
  CHKERRQ(MatAssemblyBegin(jac,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(jac,MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(0);
}

static PetscErrorCode block_system(void)
{
  SNES           snes;         /* nonlinear solver context */
  KSP            ksp;         /* linear solver context */
  PC             pc;           /* preconditioner context */
  Vec            x,r;         /* solution, residual vectors */
  Mat            J;            /* Jacobian matrix */
  PetscInt       its;
  PetscScalar    pfive = .5;
  PetscBool      flg;

  Mat            j11, j12, j21, j22;
  Vec            x1, x2, r1, r2;
  Vec            bv;
  Vec            bx[2];
  Mat            bA[2][2];

  PetscFunctionBeginUser;
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD, "\n\n========================= Block system =========================\n\n"));

  CHKERRQ(SNESCreate(PETSC_COMM_WORLD,&snes));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  Create matrix and vector data structures; set corresponding routines
  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /*
  Create sub vectors for solution and nonlinear function
  */
  CHKERRQ(VecCreateSeq(PETSC_COMM_SELF,1,&x1));
  CHKERRQ(VecDuplicate(x1,&r1));

  CHKERRQ(VecCreateSeq(PETSC_COMM_SELF,1,&x2));
  CHKERRQ(VecDuplicate(x2,&r2));

  /*
  Create the block vectors
  */
  bx[0] = x1;
  bx[1] = x2;
  CHKERRQ(VecCreateNest(PETSC_COMM_WORLD,2,NULL,bx,&x));
  CHKERRQ(VecAssemblyBegin(x));
  CHKERRQ(VecAssemblyEnd(x));
  CHKERRQ(VecDestroy(&x1));
  CHKERRQ(VecDestroy(&x2));

  bx[0] = r1;
  bx[1] = r2;
  CHKERRQ(VecCreateNest(PETSC_COMM_WORLD,2,NULL,bx,&r));
  CHKERRQ(VecDestroy(&r1));
  CHKERRQ(VecDestroy(&r2));
  CHKERRQ(VecAssemblyBegin(r));
  CHKERRQ(VecAssemblyEnd(r));

  /*
  Create sub Jacobian matrix data structure
  */
  CHKERRQ(MatCreate(PETSC_COMM_WORLD, &j11));
  CHKERRQ(MatSetSizes(j11, 1, 1, 1, 1));
  CHKERRQ(MatSetType(j11, MATSEQAIJ));
  CHKERRQ(MatSetUp(j11));

  CHKERRQ(MatCreate(PETSC_COMM_WORLD, &j12));
  CHKERRQ(MatSetSizes(j12, 1, 1, 1, 1));
  CHKERRQ(MatSetType(j12, MATSEQAIJ));
  CHKERRQ(MatSetUp(j12));

  CHKERRQ(MatCreate(PETSC_COMM_WORLD, &j21));
  CHKERRQ(MatSetSizes(j21, 1, 1, 1, 1));
  CHKERRQ(MatSetType(j21, MATSEQAIJ));
  CHKERRQ(MatSetUp(j21));

  CHKERRQ(MatCreate(PETSC_COMM_WORLD, &j22));
  CHKERRQ(MatSetSizes(j22, PETSC_DECIDE, PETSC_DECIDE, 1, 1));
  CHKERRQ(MatSetType(j22, MATSEQAIJ));
  CHKERRQ(MatSetUp(j22));
  /*
  Create block Jacobian matrix data structure
  */
  bA[0][0] = j11;
  bA[0][1] = j12;
  bA[1][0] = j21;
  bA[1][1] = j22;

  CHKERRQ(MatCreateNest(PETSC_COMM_WORLD,2,NULL,2,NULL,&bA[0][0],&J));
  CHKERRQ(MatSetUp(J));
  CHKERRQ(MatNestSetVecType(J,VECNEST));
  CHKERRQ(MatDestroy(&j11));
  CHKERRQ(MatDestroy(&j12));
  CHKERRQ(MatDestroy(&j21));
  CHKERRQ(MatDestroy(&j22));

  CHKERRQ(MatAssemblyBegin(J,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(J,MAT_FINAL_ASSEMBLY));

  CHKERRQ(PetscOptionsHasName(NULL,NULL,"-hard",&flg));
  if (!flg) {
    /*
    Set function evaluation routine and vector.
    */
    CHKERRQ(SNESSetFunction(snes,r,FormFunction1_block,NULL));

    /*
    Set Jacobian matrix data structure and Jacobian evaluation routine
    */
    CHKERRQ(SNESSetJacobian(snes,J,J,FormJacobian1_block,NULL));
  } else {
    CHKERRQ(SNESSetFunction(snes,r,FormFunction2_block,NULL));
    CHKERRQ(SNESSetJacobian(snes,J,J,FormJacobian2_block,NULL));
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  Customize nonlinear solver; set runtime options
  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /*
  Set linear solver defaults for this problem. By extracting the
  KSP, KSP, and PC contexts from the SNES context, we can then
  directly call any KSP, KSP, and PC routines to set various options.
  */
  CHKERRQ(SNESGetKSP(snes,&ksp));
  CHKERRQ(KSPGetPC(ksp,&pc));
  CHKERRQ(PCSetType(pc,PCNONE));
  CHKERRQ(KSPSetTolerances(ksp,1.e-4,PETSC_DEFAULT,PETSC_DEFAULT,20));

  /*
  Set SNES/KSP/KSP/PC runtime options, e.g.,
  -snes_view -snes_monitor -ksp_type <ksp> -pc_type <pc>
  These options will override those specified above as long as
  SNESSetFromOptions() is called _after_ any other customization
  routines.
  */
  CHKERRQ(SNESSetFromOptions(snes));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  Evaluate initial guess; then solve nonlinear system
  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  if (!flg) {
    CHKERRQ(VecSet(x,pfive));
  } else {
    Vec *vecs;
    CHKERRQ(VecNestGetSubVecs(x, NULL, &vecs));
    bv   = vecs[0];
/*    CHKERRQ(VecBlockGetSubVec(x, 0, &bv)); */
    CHKERRQ(VecSetValue(bv, 0, 2.0, INSERT_VALUES));  /* xx[0] = 2.0; */
    CHKERRQ(VecAssemblyBegin(bv));
    CHKERRQ(VecAssemblyEnd(bv));

/*    CHKERRQ(VecBlockGetSubVec(x, 1, &bv)); */
    bv   = vecs[1];
    CHKERRQ(VecSetValue(bv, 0, 3.0, INSERT_VALUES));  /* xx[1] = 3.0; */
    CHKERRQ(VecAssemblyBegin(bv));
    CHKERRQ(VecAssemblyEnd(bv));
  }
  /*
  Note: The user should initialize the vector, x, with the initial guess
  for the nonlinear solver prior to calling SNESSolve().  In particular,
  to employ an initial guess of zero, the user should explicitly set
  this vector to zero by calling VecSet().
  */
  CHKERRQ(SNESSolve(snes,NULL,x));
  CHKERRQ(SNESGetIterationNumber(snes,&its));
  if (flg) {
    Vec f;
    CHKERRQ(VecView(x,PETSC_VIEWER_STDOUT_WORLD));
    CHKERRQ(SNESGetFunction(snes,&f,0,0));
    CHKERRQ(VecView(r,PETSC_VIEWER_STDOUT_WORLD));
  }

  CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"number of SNES iterations = %D\n\n",its));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  Free work space.  All PETSc objects should be destroyed when they
  are no longer needed.
  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(VecDestroy(&x)); CHKERRQ(VecDestroy(&r));
  CHKERRQ(MatDestroy(&J)); CHKERRQ(SNESDestroy(&snes));
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------------- */
static PetscErrorCode FormFunction1_block(SNES snes,Vec x,Vec f,void *dummy)
{
  Vec            *xx, *ff, x1,x2, f1,f2;
  PetscScalar    ff_0, ff_1;
  PetscScalar    xx_0, xx_1;
  PetscInt       index,nb;

  PetscFunctionBeginUser;
  /* get blocks for function */
  CHKERRQ(VecNestGetSubVecs(f, &nb, &ff));
  f1   = ff[0];  f2 = ff[1];

  /* get blocks for solution */
  CHKERRQ(VecNestGetSubVecs(x, &nb, &xx));
  x1   = xx[0];  x2 = xx[1];

  /* get solution values */
  index = 0;
  CHKERRQ(VecGetValues(x1,1, &index, &xx_0));
  CHKERRQ(VecGetValues(x2,1, &index, &xx_1));

  /* Compute function */
  ff_0 = xx_0*xx_0 + xx_0*xx_1 - 3.0;
  ff_1 = xx_0*xx_1 + xx_1*xx_1 - 6.0;

  /* set function values */
  CHKERRQ(VecSetValue(f1, index, ff_0, INSERT_VALUES));

  CHKERRQ(VecSetValue(f2, index, ff_1, INSERT_VALUES));

  CHKERRQ(VecAssemblyBegin(f));
  CHKERRQ(VecAssemblyEnd(f));
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------------- */
static PetscErrorCode FormJacobian1_block(SNES snes,Vec x,Mat jac,Mat B,void *dummy)
{
  Vec            *xx, x1,x2;
  PetscScalar    xx_0, xx_1;
  PetscInt       index,nb;
  PetscScalar    A_00, A_01, A_10, A_11;
  Mat            j11, j12, j21, j22;
  Mat            **mats;

  PetscFunctionBeginUser;
  /* get blocks for solution */
  CHKERRQ(VecNestGetSubVecs(x, &nb, &xx));
  x1   = xx[0];  x2 = xx[1];

  /* get solution values */
  index = 0;
  CHKERRQ(VecGetValues(x1,1, &index, &xx_0));
  CHKERRQ(VecGetValues(x2,1, &index, &xx_1));

  /* get block matrices */
  CHKERRQ(MatNestGetSubMats(jac,NULL,NULL,&mats));
  j11  = mats[0][0];
  j12  = mats[0][1];
  j21  = mats[1][0];
  j22  = mats[1][1];

  /* compute jacobian entries */
  A_00 = 2.0*xx_0 + xx_1;
  A_01 = xx_0;
  A_10 = xx_1;
  A_11 = xx_0 + 2.0*xx_1;

  /* set jacobian values */
  CHKERRQ(MatSetValue(j11, 0,0, A_00, INSERT_VALUES));
  CHKERRQ(MatSetValue(j12, 0,0, A_01, INSERT_VALUES));
  CHKERRQ(MatSetValue(j21, 0,0, A_10, INSERT_VALUES));
  CHKERRQ(MatSetValue(j22, 0,0, A_11, INSERT_VALUES));

  /* Assemble sub matrix */
  CHKERRQ(MatAssemblyBegin(jac,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(jac,MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------------- */
static PetscErrorCode FormFunction2_block(SNES snes,Vec x,Vec f,void *dummy)
{
  PetscScalar       *ff;
  const PetscScalar *xx;

  PetscFunctionBeginUser;
  /*
  Get pointers to vector data.
  - For default PETSc vectors, VecGetArray() returns a pointer to
  the data array.  Otherwise, the routine is implementation dependent.
  - You MUST call VecRestoreArray() when you no longer need access to
  the array.
  */
  CHKERRQ(VecGetArrayRead(x,&xx));
  CHKERRQ(VecGetArray(f,&ff));

  /*
  Compute function
  */
  ff[0] = PetscSinScalar(3.0*xx[0]) + xx[0];
  ff[1] = xx[1];

  /*
  Restore vectors
  */
  CHKERRQ(VecRestoreArrayRead(x,&xx));
  CHKERRQ(VecRestoreArray(f,&ff));
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------------- */
static PetscErrorCode FormJacobian2_block(SNES snes,Vec x,Mat jac,Mat B,void *dummy)
{
  const PetscScalar *xx;
  PetscScalar       A[4];
  PetscInt          idx[2] = {0,1};

  PetscFunctionBeginUser;
  /*
  Get pointer to vector data
  */
  CHKERRQ(VecGetArrayRead(x,&xx));

  /*
  Compute Jacobian entries and insert into matrix.
  - Since this is such a small problem, we set all entries for
  the matrix at once.
  */
  A[0]  = 3.0*PetscCosScalar(3.0*xx[0]) + 1.0; A[1] = 0.0;
  A[2]  = 0.0;                     A[3] = 1.0;
  CHKERRQ(MatSetValues(jac,2,idx,2,idx,A,INSERT_VALUES));

  /*
  Restore vector
  */
  CHKERRQ(VecRestoreArrayRead(x,&xx));

  /*
  Assemble matrix
  */
  CHKERRQ(MatAssemblyBegin(jac,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(jac,MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(0);
}

int main(int argc,char **argv)
{
  PetscMPIInt    size;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  CHKERRMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  PetscCheckFalse(size != 1,PETSC_COMM_WORLD, PETSC_ERR_WRONG_MPI_SIZE,"This is a uniprocessor example only!");
  CHKERRQ(assembled_system());
  CHKERRQ(block_system());
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:
      args: -snes_monitor_short
      requires: !single

TEST*/
