static char help[] = "Poiseuille flow problem. Viscous, laminar flow in a 2D channel with parabolic velocity\n\
                      profile and linear pressure drop, exact solution of the 2D Stokes\n";

/*
     M A R I T I M E  R E S E A R C H  I N S T I T U T E  N E T H E R L A N D S
   author : Christiaan M. Klaij

   Poiseuille flow problem.

   Viscous, laminar flow in a 2D channel with parabolic velocity
   profile and linear pressure drop, exact solution of the 2D Stokes
   equations.

   Discretized with the cell-centered finite-volume method on a
   Cartesian grid with co-located variables. Variables ordered as
   [u1...uN v1...vN p1...pN]^T. Matrix [A00 A01; A10, A11] solved with
   PCFIELDSPLIT. Lower factorization is used to mimic the Semi-Implicit
   Method for Pressure Linked Equations (SIMPLE) used as preconditioner
   instead of solver.

   Disclaimer: does not contain the pressure-weighed interpolation
   method needed to suppress spurious pressure modes in real-life
   problems.

   Usage:
     mpiexec -n 2 ./ex70 -nx 32 -ny 48 -ksp_type fgmres -pc_type fieldsplit -pc_fieldsplit_type schur -pc_fieldsplit_schur_fact_type lower -fieldsplit_1_pc_type none

     Runs with PCFIELDSPLIT on 32x48 grid, no PC for the Schur
     complement because A11 is zero. FGMRES is needed because
     PCFIELDSPLIT is a variable preconditioner.

     mpiexec -n 2 ./ex70 -nx 32 -ny 48 -ksp_type fgmres -pc_type fieldsplit -pc_fieldsplit_type schur -pc_fieldsplit_schur_fact_type lower -user_pc

     Same as above but with user defined PC for the true Schur
     complement. PC based on the SIMPLE-type approximation (inverse of
     A00 approximated by inverse of its diagonal).

     mpiexec -n 2 ./ex70 -nx 32 -ny 48 -ksp_type fgmres -pc_type fieldsplit -pc_fieldsplit_type schur -pc_fieldsplit_schur_fact_type lower -user_ksp

     Replace the true Schur complement with a user defined Schur
     complement based on the SIMPLE-type approximation. Same matrix is
     used as PC.

     mpiexec -n 2 ./ex70 -nx 32 -ny 48 -ksp_type fgmres -pc_type fieldsplit -pc_fieldsplit_type schur -pc_fieldsplit_schur_fact_type lower -fieldsplit_0_ksp_type gmres -fieldsplit_0_pc_type bjacobi -fieldsplit_1_pc_type jacobi -fieldsplit_1_inner_ksp_type preonly -fieldsplit_1_inner_pc_type jacobi -fieldsplit_1_upper_ksp_type preonly -fieldsplit_1_upper_pc_type jacobi

     Out-of-the-box SIMPLE-type preconditioning. The major advantage
     is that the user neither needs to provide the approximation of
     the Schur complement, nor the corresponding preconditioner.
*/

#include <petscksp.h>

typedef struct {
  PetscBool userPC, userKSP, matsymmetric; /* user defined preconditioner and matrix for the Schur complement */
  PetscInt  nx, ny;  /* nb of cells in x- and y-direction */
  PetscReal hx, hy;  /* mesh size in x- and y-direction */
  Mat       A;       /* block matrix */
  Mat       subA[4]; /* the four blocks */
  Mat       myS;     /* the approximation of the Schur complement */
  Vec       x, b, y; /* solution, rhs and temporary vector */
  IS        isg[2];  /* index sets of split "0" and "1" */
} Stokes;

PetscErrorCode StokesSetupMatBlock00(Stokes*);  /* setup the block Q */
PetscErrorCode StokesSetupMatBlock01(Stokes*);  /* setup the block G */
PetscErrorCode StokesSetupMatBlock10(Stokes*);  /* setup the block D (equal to the transpose of G) */
PetscErrorCode StokesSetupMatBlock11(Stokes*);  /* setup the block C (equal to zero) */

PetscErrorCode StokesGetPosition(Stokes*, PetscInt, PetscInt*, PetscInt*); /* row number j*nx+i corresponds to position (i,j) in grid */

PetscErrorCode StokesStencilLaplacian(Stokes*, PetscInt, PetscInt, PetscInt*, PetscInt*, PetscScalar*);  /* stencil of the Laplacian operator */
PetscErrorCode StokesStencilGradientX(Stokes*, PetscInt, PetscInt, PetscInt*, PetscInt*, PetscScalar*);  /* stencil of the Gradient operator (x-component) */
PetscErrorCode StokesStencilGradientY(Stokes*, PetscInt, PetscInt, PetscInt*, PetscInt*, PetscScalar*);  /* stencil of the Gradient operator (y-component) */

PetscErrorCode StokesRhs(Stokes*);                                         /* rhs vector */
PetscErrorCode StokesRhsMomX(Stokes*, PetscInt, PetscInt, PetscScalar*);   /* right hand side of velocity (x-component) */
PetscErrorCode StokesRhsMomY(Stokes*, PetscInt, PetscInt, PetscScalar*);   /* right hand side of velocity (y-component) */
PetscErrorCode StokesRhsMass(Stokes*, PetscInt, PetscInt, PetscScalar*);   /* right hand side of pressure */

PetscErrorCode StokesSetupApproxSchur(Stokes*);  /* approximation of the Schur complement */

PetscErrorCode StokesExactSolution(Stokes*); /* exact solution vector */
PetscErrorCode StokesWriteSolution(Stokes*); /* write solution to file */

/* exact solution for the velocity (x-component, y-component is zero) */
PetscScalar StokesExactVelocityX(const PetscScalar y)
{
  return 4.0*y*(1.0-y);
}

/* exact solution for the pressure */
PetscScalar StokesExactPressure(const PetscScalar x)
{
  return 8.0*(2.0-x);
}

PetscErrorCode StokesSetupPC(Stokes *s, KSP ksp)
{
  KSP            *subksp;
  PC             pc;
  PetscInt       n = 1;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = KSPGetPC(ksp, &pc);CHKERRQ(ierr);
  ierr = PCFieldSplitSetIS(pc, "0", s->isg[0]);CHKERRQ(ierr);
  ierr = PCFieldSplitSetIS(pc, "1", s->isg[1]);CHKERRQ(ierr);
  if (s->userPC) {
    ierr = PCFieldSplitSetSchurPre(pc, PC_FIELDSPLIT_SCHUR_PRE_USER, s->myS);CHKERRQ(ierr);
  }
  if (s->userKSP) {
    ierr = PCSetUp(pc);CHKERRQ(ierr);
    ierr = PCFieldSplitGetSubKSP(pc, &n, &subksp);CHKERRQ(ierr);
    ierr = KSPSetOperators(subksp[1], s->myS, s->myS);CHKERRQ(ierr);
    ierr = PetscFree(subksp);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode StokesWriteSolution(Stokes *s)
{
  PetscMPIInt       size;
  PetscInt          n,i,j;
  const PetscScalar *array;
  PetscErrorCode    ierr;

  PetscFunctionBeginUser;
  /* write data (*warning* only works sequential) */
  ierr = MPI_Comm_size(MPI_COMM_WORLD,&size);CHKERRMPI(ierr);
  if (size == 1) {
    PetscViewer viewer;
    ierr = VecGetArrayRead(s->x, &array);CHKERRQ(ierr);
    ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD, "solution.dat", &viewer);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer, "# x, y, u, v, p\n");CHKERRQ(ierr);
    for (j = 0; j < s->ny; j++) {
      for (i = 0; i < s->nx; i++) {
        n    = j*s->nx+i;
        ierr = PetscViewerASCIIPrintf(viewer, "%.12g %.12g %.12g %.12g %.12g\n", (double)(i*s->hx+s->hx/2),(double)(j*s->hy+s->hy/2), (double)PetscRealPart(array[n]), (double)PetscRealPart(array[n+s->nx*s->ny]),(double)PetscRealPart(array[n+2*s->nx*s->ny]));CHKERRQ(ierr);
      }
    }
    ierr = VecRestoreArrayRead(s->x, &array);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode StokesSetupIndexSets(Stokes *s)
{
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  /* the two index sets */
  ierr = MatNestGetISs(s->A, s->isg, NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode StokesSetupVectors(Stokes *s)
{
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  /* solution vector x */
  ierr = VecCreate(PETSC_COMM_WORLD, &s->x);CHKERRQ(ierr);
  ierr = VecSetSizes(s->x, PETSC_DECIDE, 3*s->nx*s->ny);CHKERRQ(ierr);
  ierr = VecSetType(s->x, VECMPI);CHKERRQ(ierr);

  /* exact solution y */
  ierr = VecDuplicate(s->x, &s->y);CHKERRQ(ierr);
  ierr = StokesExactSolution(s);CHKERRQ(ierr);

  /* rhs vector b */
  ierr = VecDuplicate(s->x, &s->b);CHKERRQ(ierr);
  ierr = StokesRhs(s);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode StokesGetPosition(Stokes *s, PetscInt row, PetscInt *i, PetscInt *j)
{
  PetscInt n;

  PetscFunctionBeginUser;
  /* cell number n=j*nx+i has position (i,j) in grid */
  n  = row%(s->nx*s->ny);
  *i = n%s->nx;
  *j = (n-(*i))/s->nx;
  PetscFunctionReturn(0);
}

PetscErrorCode StokesExactSolution(Stokes *s)
{
  PetscInt       row, start, end, i, j;
  PetscScalar    val;
  Vec            y0,y1;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  /* velocity part */
  ierr = VecGetSubVector(s->y, s->isg[0], &y0);CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(y0, &start, &end);CHKERRQ(ierr);
  for (row = start; row < end; row++) {
    ierr = StokesGetPosition(s, row,&i,&j);CHKERRQ(ierr);
    if (row < s->nx*s->ny) {
      val = StokesExactVelocityX(j*s->hy+s->hy/2);
    } else {
      val = 0;
    }
    ierr = VecSetValue(y0, row, val, INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = VecRestoreSubVector(s->y, s->isg[0], &y0);CHKERRQ(ierr);

  /* pressure part */
  ierr = VecGetSubVector(s->y, s->isg[1], &y1);CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(y1, &start, &end);CHKERRQ(ierr);
  for (row = start; row < end; row++) {
    ierr = StokesGetPosition(s, row, &i, &j);CHKERRQ(ierr);
    val  = StokesExactPressure(i*s->hx+s->hx/2);
    ierr = VecSetValue(y1, row, val, INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = VecRestoreSubVector(s->y, s->isg[1], &y1);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode StokesRhs(Stokes *s)
{
  PetscInt       row, start, end, i, j;
  PetscScalar    val;
  Vec            b0,b1;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  /* velocity part */
  ierr = VecGetSubVector(s->b, s->isg[0], &b0);CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(b0, &start, &end);CHKERRQ(ierr);
  for (row = start; row < end; row++) {
    ierr = StokesGetPosition(s, row, &i, &j);CHKERRQ(ierr);
    if (row < s->nx*s->ny) {
      ierr = StokesRhsMomX(s, i, j, &val);CHKERRQ(ierr);
    } else {
      ierr = StokesRhsMomY(s, i, j, &val);CHKERRQ(ierr);
    }
    ierr = VecSetValue(b0, row, val, INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = VecRestoreSubVector(s->b, s->isg[0], &b0);CHKERRQ(ierr);

  /* pressure part */
  ierr = VecGetSubVector(s->b, s->isg[1], &b1);CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(b1, &start, &end);CHKERRQ(ierr);
  for (row = start; row < end; row++) {
    ierr = StokesGetPosition(s, row, &i, &j);CHKERRQ(ierr);
    ierr = StokesRhsMass(s, i, j, &val);CHKERRQ(ierr);
    if (s->matsymmetric) {
      val = -1.0*val;
    }
    ierr = VecSetValue(b1, row, val, INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = VecRestoreSubVector(s->b, s->isg[1], &b1);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode StokesSetupMatBlock00(Stokes *s)
{
  PetscInt       row, start, end, sz, i, j;
  PetscInt       cols[5];
  PetscScalar    vals[5];
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  /* A[0] is 2N-by-2N */
  ierr = MatCreate(PETSC_COMM_WORLD,&s->subA[0]);CHKERRQ(ierr);
  ierr = MatSetOptionsPrefix(s->subA[0],"a00_");CHKERRQ(ierr);
  ierr = MatSetSizes(s->subA[0],PETSC_DECIDE,PETSC_DECIDE,2*s->nx*s->ny,2*s->nx*s->ny);CHKERRQ(ierr);
  ierr = MatSetType(s->subA[0],MATMPIAIJ);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(s->subA[0],5,NULL,5,NULL);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(s->subA[0], &start, &end);CHKERRQ(ierr);

  for (row = start; row < end; row++) {
    ierr = StokesGetPosition(s, row, &i, &j);CHKERRQ(ierr);
    /* first part: rows 0 to (nx*ny-1) */
    ierr = StokesStencilLaplacian(s, i, j, &sz, cols, vals);CHKERRQ(ierr);
    /* second part: rows (nx*ny) to (2*nx*ny-1) */
    if (row >= s->nx*s->ny) {
      for (i = 0; i < sz; i++) cols[i] += s->nx*s->ny;
    }
    for (i = 0; i < sz; i++) vals[i] = -1.0*vals[i]; /* dynamic viscosity coef mu=-1 */
    ierr = MatSetValues(s->subA[0], 1, &row, sz, cols, vals, INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(s->subA[0], MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(s->subA[0], MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode StokesSetupMatBlock01(Stokes *s)
{
  PetscInt       row, start, end, sz, i, j;
  PetscInt       cols[5];
  PetscScalar    vals[5];
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  /* A[1] is 2N-by-N */
  ierr = MatCreate(PETSC_COMM_WORLD, &s->subA[1]);CHKERRQ(ierr);
  ierr = MatSetOptionsPrefix(s->subA[1],"a01_");CHKERRQ(ierr);
  ierr = MatSetSizes(s->subA[1],PETSC_DECIDE,PETSC_DECIDE,2*s->nx*s->ny,s->nx*s->ny);CHKERRQ(ierr);
  ierr = MatSetType(s->subA[1],MATMPIAIJ);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(s->subA[1],5,NULL,5,NULL);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(s->subA[1],&start,&end);CHKERRQ(ierr);

  ierr = MatSetOption(s->subA[1],MAT_IGNORE_ZERO_ENTRIES,PETSC_TRUE);CHKERRQ(ierr);

  for (row = start; row < end; row++) {
    ierr = StokesGetPosition(s, row, &i, &j);CHKERRQ(ierr);
    /* first part: rows 0 to (nx*ny-1) */
    if (row < s->nx*s->ny) {
      ierr = StokesStencilGradientX(s, i, j, &sz, cols, vals);CHKERRQ(ierr);
    } else {    /* second part: rows (nx*ny) to (2*nx*ny-1) */
      ierr = StokesStencilGradientY(s, i, j, &sz, cols, vals);CHKERRQ(ierr);
    }
    ierr = MatSetValues(s->subA[1], 1, &row, sz, cols, vals, INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(s->subA[1], MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(s->subA[1], MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode StokesSetupMatBlock10(Stokes *s)
{
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  /* A[2] is minus transpose of A[1] */
  ierr = MatTranspose(s->subA[1], MAT_INITIAL_MATRIX, &s->subA[2]);CHKERRQ(ierr);
  if (!s->matsymmetric) {
    ierr = MatScale(s->subA[2], -1.0);CHKERRQ(ierr);
  }
  ierr = MatSetOptionsPrefix(s->subA[2], "a10_");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode StokesSetupMatBlock11(Stokes *s)
{
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  /* A[3] is N-by-N null matrix */
  ierr = MatCreate(PETSC_COMM_WORLD, &s->subA[3]);CHKERRQ(ierr);
  ierr = MatSetOptionsPrefix(s->subA[3], "a11_");CHKERRQ(ierr);
  ierr = MatSetSizes(s->subA[3], PETSC_DECIDE, PETSC_DECIDE, s->nx*s->ny, s->nx*s->ny);CHKERRQ(ierr);
  ierr = MatSetType(s->subA[3], MATMPIAIJ);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(s->subA[3], 0, NULL, 0, NULL);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(s->subA[3], MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(s->subA[3], MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode StokesSetupApproxSchur(Stokes *s)
{
  Vec            diag;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  /* Schur complement approximation: myS = A11 - A10 inv(DIAGFORM(A00)) A01 */
  /* note: A11 is zero */
  /* note: in real life this matrix would be build directly, */
  /* i.e. without MatMatMult */

  /* inverse of diagonal of A00 */
  ierr = VecCreate(PETSC_COMM_WORLD,&diag);CHKERRQ(ierr);
  ierr = VecSetSizes(diag,PETSC_DECIDE,2*s->nx*s->ny);CHKERRQ(ierr);
  ierr = VecSetType(diag,VECMPI);CHKERRQ(ierr);
  ierr = MatGetDiagonal(s->subA[0],diag);CHKERRQ(ierr);
  ierr = VecReciprocal(diag);CHKERRQ(ierr);

  /* compute: - A10 inv(DIAGFORM(A00)) A01 */
  ierr = MatDiagonalScale(s->subA[1],diag,NULL);CHKERRQ(ierr); /* (*warning* overwrites subA[1]) */
  ierr = MatMatMult(s->subA[2],s->subA[1],MAT_INITIAL_MATRIX,PETSC_DEFAULT,&s->myS);CHKERRQ(ierr);
  ierr = MatScale(s->myS,-1.0);CHKERRQ(ierr);

  /* restore A10 */
  ierr = MatGetDiagonal(s->subA[0],diag);CHKERRQ(ierr);
  ierr = MatDiagonalScale(s->subA[1],diag,NULL);CHKERRQ(ierr);
  ierr = VecDestroy(&diag);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode StokesSetupMatrix(Stokes *s)
{
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = StokesSetupMatBlock00(s);CHKERRQ(ierr);
  ierr = StokesSetupMatBlock01(s);CHKERRQ(ierr);
  ierr = StokesSetupMatBlock10(s);CHKERRQ(ierr);
  ierr = StokesSetupMatBlock11(s);CHKERRQ(ierr);
  ierr = MatCreateNest(PETSC_COMM_WORLD, 2, NULL, 2, NULL, s->subA, &s->A);CHKERRQ(ierr);
  ierr = StokesSetupApproxSchur(s);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode StokesStencilLaplacian(Stokes *s, PetscInt i, PetscInt j, PetscInt *sz, PetscInt *cols, PetscScalar *vals)
{
  PetscInt    p =j*s->nx+i, w=p-1, e=p+1, s2=p-s->nx, n=p+s->nx;
  PetscScalar ae=s->hy/s->hx, aeb=0;
  PetscScalar aw=s->hy/s->hx, awb=s->hy/(s->hx/2);
  PetscScalar as=s->hx/s->hy, asb=s->hx/(s->hy/2);
  PetscScalar an=s->hx/s->hy, anb=s->hx/(s->hy/2);

  PetscFunctionBeginUser;
  if (i==0 && j==0) { /* south-west corner */
    *sz  =3;
    cols[0]=p; vals[0]=-(ae+awb+asb+an);
    cols[1]=e; vals[1]=ae;
    cols[2]=n; vals[2]=an;
  } else if (i==0 && j==s->ny-1) { /* north-west corner */
    *sz  =3;
    cols[0]=s2; vals[0]=as;
    cols[1]=p; vals[1]=-(ae+awb+as+anb);
    cols[2]=e; vals[2]=ae;
  } else if (i==s->nx-1 && j==0) { /* south-east corner */
    *sz  =3;
    cols[0]=w; vals[0]=aw;
    cols[1]=p; vals[1]=-(aeb+aw+asb+an);
    cols[2]=n; vals[2]=an;
  } else if (i==s->nx-1 && j==s->ny-1) { /* north-east corner */
    *sz  =3;
    cols[0]=s2; vals[0]=as;
    cols[1]=w; vals[1]=aw;
    cols[2]=p; vals[2]=-(aeb+aw+as+anb);
  } else if (i==0) { /* west boundary */
    *sz  =4;
    cols[0]=s2; vals[0]=as;
    cols[1]=p; vals[1]=-(ae+awb+as+an);
    cols[2]=e; vals[2]=ae;
    cols[3]=n; vals[3]=an;
  } else if (i==s->nx-1) { /* east boundary */
    *sz  =4;
    cols[0]=s2; vals[0]=as;
    cols[1]=w; vals[1]=aw;
    cols[2]=p; vals[2]=-(aeb+aw+as+an);
    cols[3]=n; vals[3]=an;
  } else if (j==0) { /* south boundary */
    *sz  =4;
    cols[0]=w; vals[0]=aw;
    cols[1]=p; vals[1]=-(ae+aw+asb+an);
    cols[2]=e; vals[2]=ae;
    cols[3]=n; vals[3]=an;
  } else if (j==s->ny-1) { /* north boundary */
    *sz  =4;
    cols[0]=s2; vals[0]=as;
    cols[1]=w; vals[1]=aw;
    cols[2]=p; vals[2]=-(ae+aw+as+anb);
    cols[3]=e; vals[3]=ae;
  } else { /* interior */
    *sz  =5;
    cols[0]=s2; vals[0]=as;
    cols[1]=w; vals[1]=aw;
    cols[2]=p; vals[2]=-(ae+aw+as+an);
    cols[3]=e; vals[3]=ae;
    cols[4]=n; vals[4]=an;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode StokesStencilGradientX(Stokes *s, PetscInt i, PetscInt j, PetscInt *sz, PetscInt *cols, PetscScalar *vals)
{
  PetscInt    p =j*s->nx+i, w=p-1, e=p+1;
  PetscScalar ae= s->hy/2, aeb=s->hy;
  PetscScalar aw=-s->hy/2, awb=0;

  PetscFunctionBeginUser;
  if (i==0 && j==0) { /* south-west corner */
    *sz  =2;
    cols[0]=p; vals[0]=-(ae+awb);
    cols[1]=e; vals[1]=ae;
  } else if (i==0 && j==s->ny-1) { /* north-west corner */
    *sz  =2;
    cols[0]=p; vals[0]=-(ae+awb);
    cols[1]=e; vals[1]=ae;
  } else if (i==s->nx-1 && j==0) { /* south-east corner */
    *sz  =2;
    cols[0]=w; vals[0]=aw;
    cols[1]=p; vals[1]=-(aeb+aw);
  } else if (i==s->nx-1 && j==s->ny-1) { /* north-east corner */
    *sz  =2;
    cols[0]=w; vals[0]=aw;
    cols[1]=p; vals[1]=-(aeb+aw);
  } else if (i==0) { /* west boundary */
    *sz  =2;
    cols[0]=p; vals[0]=-(ae+awb);
    cols[1]=e; vals[1]=ae;
  } else if (i==s->nx-1) { /* east boundary */
    *sz  =2;
    cols[0]=w; vals[0]=aw;
    cols[1]=p; vals[1]=-(aeb+aw);
  } else if (j==0) { /* south boundary */
    *sz  =3;
    cols[0]=w; vals[0]=aw;
    cols[1]=p; vals[1]=-(ae+aw);
    cols[2]=e; vals[2]=ae;
  } else if (j==s->ny-1) { /* north boundary */
    *sz  =3;
    cols[0]=w; vals[0]=aw;
    cols[1]=p; vals[1]=-(ae+aw);
    cols[2]=e; vals[2]=ae;
  } else { /* interior */
    *sz  =3;
    cols[0]=w; vals[0]=aw;
    cols[1]=p; vals[1]=-(ae+aw);
    cols[2]=e; vals[2]=ae;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode StokesStencilGradientY(Stokes *s, PetscInt i, PetscInt j, PetscInt *sz, PetscInt *cols, PetscScalar *vals)
{
  PetscInt    p =j*s->nx+i, s2=p-s->nx, n=p+s->nx;
  PetscScalar as=-s->hx/2, asb=0;
  PetscScalar an= s->hx/2, anb=0;

  PetscFunctionBeginUser;
  if (i==0 && j==0) { /* south-west corner */
    *sz  =2;
    cols[0]=p; vals[0]=-(asb+an);
    cols[1]=n; vals[1]=an;
  } else if (i==0 && j==s->ny-1) { /* north-west corner */
    *sz  =2;
    cols[0]=s2; vals[0]=as;
    cols[1]=p; vals[1]=-(as+anb);
  } else if (i==s->nx-1 && j==0) { /* south-east corner */
    *sz  =2;
    cols[0]=p; vals[0]=-(asb+an);
    cols[1]=n; vals[1]=an;
  } else if (i==s->nx-1 && j==s->ny-1) { /* north-east corner */
    *sz  =2;
    cols[0]=s2; vals[0]=as;
    cols[1]=p; vals[1]=-(as+anb);
  } else if (i==0) { /* west boundary */
    *sz  =3;
    cols[0]=s2; vals[0]=as;
    cols[1]=p; vals[1]=-(as+an);
    cols[2]=n; vals[2]=an;
  } else if (i==s->nx-1) { /* east boundary */
    *sz  =3;
    cols[0]=s2; vals[0]=as;
    cols[1]=p; vals[1]=-(as+an);
    cols[2]=n; vals[2]=an;
  } else if (j==0) { /* south boundary */
    *sz  =2;
    cols[0]=p; vals[0]=-(asb+an);
    cols[1]=n; vals[1]=an;
  } else if (j==s->ny-1) { /* north boundary */
    *sz  =2;
    cols[0]=s2; vals[0]=as;
    cols[1]=p; vals[1]=-(as+anb);
  } else { /* interior */
    *sz  =3;
    cols[0]=s2; vals[0]=as;
    cols[1]=p; vals[1]=-(as+an);
    cols[2]=n; vals[2]=an;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode StokesRhsMomX(Stokes *s, PetscInt i, PetscInt j, PetscScalar *val)
{
  PetscScalar y   = j*s->hy+s->hy/2;
  PetscScalar awb = s->hy/(s->hx/2);

  PetscFunctionBeginUser;
  if (i == 0) { /* west boundary */
    *val = awb*StokesExactVelocityX(y);
  } else {
    *val = 0.0;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode StokesRhsMomY(Stokes *s, PetscInt i, PetscInt j, PetscScalar *val)
{
  PetscFunctionBeginUser;
  *val = 0.0;
  PetscFunctionReturn(0);
}

PetscErrorCode StokesRhsMass(Stokes *s, PetscInt i, PetscInt j, PetscScalar *val)
{
  PetscScalar y   = j*s->hy+s->hy/2;
  PetscScalar aeb = s->hy;

  PetscFunctionBeginUser;
  if (i == 0) { /* west boundary */
    *val = aeb*StokesExactVelocityX(y);
  } else {
    *val = 0.0;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode StokesCalcResidual(Stokes *s)
{
  PetscReal      val;
  Vec            b0, b1;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  /* residual Ax-b (*warning* overwrites b) */
  ierr = VecScale(s->b, -1.0);CHKERRQ(ierr);
  ierr = MatMultAdd(s->A, s->x, s->b, s->b);CHKERRQ(ierr);

  /* residual velocity */
  ierr = VecGetSubVector(s->b, s->isg[0], &b0);CHKERRQ(ierr);
  ierr = VecNorm(b0, NORM_2, &val);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD," residual u = %g\n",(double)val);CHKERRQ(ierr);
  ierr = VecRestoreSubVector(s->b, s->isg[0], &b0);CHKERRQ(ierr);

  /* residual pressure */
  ierr = VecGetSubVector(s->b, s->isg[1], &b1);CHKERRQ(ierr);
  ierr = VecNorm(b1, NORM_2, &val);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD," residual p = %g\n",(double)val);CHKERRQ(ierr);
  ierr = VecRestoreSubVector(s->b, s->isg[1], &b1);CHKERRQ(ierr);

  /* total residual */
  ierr = VecNorm(s->b, NORM_2, &val);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD," residual [u,p] = %g\n", (double)val);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode StokesCalcError(Stokes *s)
{
  PetscScalar    scale = PetscSqrtReal((double)s->nx*s->ny);
  PetscReal      val;
  Vec            y0, y1;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  /* error y-x */
  ierr = VecAXPY(s->y, -1.0, s->x);CHKERRQ(ierr);

  /* error in velocity */
  ierr = VecGetSubVector(s->y, s->isg[0], &y0);CHKERRQ(ierr);
  ierr = VecNorm(y0, NORM_2, &val);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD," discretization error u = %g\n",(double)(PetscRealPart(val/scale)));CHKERRQ(ierr);
  ierr = VecRestoreSubVector(s->y, s->isg[0], &y0);CHKERRQ(ierr);

  /* error in pressure */
  ierr = VecGetSubVector(s->y, s->isg[1], &y1);CHKERRQ(ierr);
  ierr = VecNorm(y1, NORM_2, &val);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD," discretization error p = %g\n",(double)(PetscRealPart(val/scale)));CHKERRQ(ierr);
  ierr = VecRestoreSubVector(s->y, s->isg[1], &y1);CHKERRQ(ierr);

  /* total error */
  ierr = VecNorm(s->y, NORM_2, &val);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD," discretization error [u,p] = %g\n", (double)PetscRealPart((val/scale)));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  Stokes         s;
  KSP            ksp;
  PetscErrorCode ierr;

  ierr           = PetscInitialize(&argc, &argv, NULL,help);if (ierr) return ierr;
  s.nx           = 4;
  s.ny           = 6;
  ierr           = PetscOptionsGetInt(NULL,NULL, "-nx", &s.nx, NULL);CHKERRQ(ierr);
  ierr           = PetscOptionsGetInt(NULL,NULL, "-ny", &s.ny, NULL);CHKERRQ(ierr);
  s.hx           = 2.0/s.nx;
  s.hy           = 1.0/s.ny;
  s.matsymmetric = PETSC_FALSE;
  ierr           = PetscOptionsGetBool(NULL,NULL, "-mat_set_symmetric", &s.matsymmetric,NULL);CHKERRQ(ierr);
  s.userPC       = s.userKSP = PETSC_FALSE;
  ierr           = PetscOptionsHasName(NULL,NULL, "-user_pc", &s.userPC);CHKERRQ(ierr);
  ierr           = PetscOptionsHasName(NULL,NULL, "-user_ksp", &s.userKSP);CHKERRQ(ierr);

  ierr = StokesSetupMatrix(&s);CHKERRQ(ierr);
  ierr = StokesSetupIndexSets(&s);CHKERRQ(ierr);
  ierr = StokesSetupVectors(&s);CHKERRQ(ierr);

  ierr = KSPCreate(PETSC_COMM_WORLD, &ksp);CHKERRQ(ierr);
  ierr = KSPSetOperators(ksp, s.A, s.A);CHKERRQ(ierr);
  ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);
  ierr = StokesSetupPC(&s, ksp);CHKERRQ(ierr);
  ierr = KSPSolve(ksp, s.b, s.x);CHKERRQ(ierr);

  /* don't trust, verify! */
  ierr = StokesCalcResidual(&s);CHKERRQ(ierr);
  ierr = StokesCalcError(&s);CHKERRQ(ierr);
  ierr = StokesWriteSolution(&s);CHKERRQ(ierr);

  ierr = KSPDestroy(&ksp);CHKERRQ(ierr);
  ierr = MatDestroy(&s.subA[0]);CHKERRQ(ierr);
  ierr = MatDestroy(&s.subA[1]);CHKERRQ(ierr);
  ierr = MatDestroy(&s.subA[2]);CHKERRQ(ierr);
  ierr = MatDestroy(&s.subA[3]);CHKERRQ(ierr);
  ierr = MatDestroy(&s.A);CHKERRQ(ierr);
  ierr = VecDestroy(&s.x);CHKERRQ(ierr);
  ierr = VecDestroy(&s.b);CHKERRQ(ierr);
  ierr = VecDestroy(&s.y);CHKERRQ(ierr);
  ierr = MatDestroy(&s.myS);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:
      nsize: 2
      args: -nx 16 -ny 24 -ksp_type fgmres -pc_type fieldsplit -pc_fieldsplit_type schur -pc_fieldsplit_schur_fact_type lower -fieldsplit_1_pc_type none

   test:
      suffix: 2
      nsize: 2
      args: -nx 16 -ny 24 -ksp_type fgmres -pc_type fieldsplit -pc_fieldsplit_type schur -pc_fieldsplit_schur_fact_type lower -user_pc

   test:
      suffix: 3
      nsize: 2
      args: -nx 16 -ny 24 -ksp_type fgmres -pc_type fieldsplit -pc_fieldsplit_type schur -pc_fieldsplit_schur_fact_type lower -user_pc

   test:
      suffix: 4
      nsize: 2
      args: -nx 16 -ny 24 -ksp_type fgmres -pc_type fieldsplit -pc_fieldsplit_type schur -pc_fieldsplit_schur_fact_type lower -fieldsplit_0_ksp_type gmres -fieldsplit_0_pc_type bjacobi -fieldsplit_1_pc_type jacobi -fieldsplit_1_inner_ksp_type preonly -fieldsplit_1_inner_pc_type jacobi -fieldsplit_1_upper_ksp_type preonly -fieldsplit_1_upper_pc_type jacobi

   test:
      suffix: 4_pcksp
      nsize: 2
      args: -nx 16 -ny 24 -ksp_type fgmres -pc_type fieldsplit -pc_fieldsplit_type schur -pc_fieldsplit_schur_fact_type lower -fieldsplit_0_ksp_type gmres -fieldsplit_0_pc_type bjacobi -fieldsplit_1_pc_type jacobi -fieldsplit_1_inner_ksp_type preonly -fieldsplit_1_upper_ksp_type preonly -fieldsplit_1_upper_pc_type jacobi

   test:
      suffix: 5
      nsize: 2
      args: -nx 4 -ny 8 -mat_set_symmetric -ksp_type fgmres -pc_type fieldsplit -pc_fieldsplit_type gkb -fieldsplit_0_ksp_type cg -fieldsplit_0_pc_type jacobi -fieldsplit_0_ksp_rtol 1e-10

   test:
      suffix: 6
      nsize: 2
      args: -nx 4 -ny 8 -mat_set_symmetric -ksp_type preonly -pc_type fieldsplit -pc_fieldsplit_type gkb -fieldsplit_0_ksp_type cg -fieldsplit_0_pc_type jacobi -fieldsplit_0_ksp_rtol 1e-10

   test:
      suffix: 7
      nsize: 2
      args: -nx 4 -ny 8 -mat_set_symmetric -ksp_type preonly -pc_type fieldsplit -pc_fieldsplit_type gkb -pc_fieldsplit_gkb_tol 1e-4 -pc_fieldsplit_gkb_nu 5 -fieldsplit_0_ksp_type cg -fieldsplit_0_pc_type jacobi -fieldsplit_0_ksp_rtol 1e-6

TEST*/
