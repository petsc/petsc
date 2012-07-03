static char help[] = "Test MatSetValuesBatch: setting batches of elements using the GPU.\n\
This works with SeqAIJCUSP and MPIAIJCUSP matrices.\n\n";
#include <petscdmda.h>
#include <petscksp.h>

/* We will use a structured mesh for this assembly test. Each square will be divided into two triangles:
  C       D
   _______
  |\      | The matrix for 0 and 1 is /   1  -0.5 -0.5 \
  | \   1 |                           | -0.5  0.5  0.0 |
  |  \    |                           \ -0.5  0.0  0.5 /
  |   \   |
  |    \  |
  |  0  \ |
  |      \|
  ---------
  A       B

TO ADD:
  DONE 1) Build and run on baconost
    - Gather data for CPU/GPU up to da_grid_x 1300
      - Looks 6x faster than CPU
    - Make plot

  DONE 2) Solve the Neumann Poisson problem

  3) Multi-GPU Assembly
    - MPIAIJCUSP: Just have two SEQAIJCUSP matrices, nothing else special
    a) Filter rows to be sent to other procs (normally stashed)
    b) send/recv rows, might as well do with a VecScatter
    c) Potential to overlap this computation w/ GPU (talk to Nathan)
    c') Just shove these rows in after the local
    d) Have implicit rep of COO from repeated/tiled_range
    e) Do a filtered copy, decrementing rows and remapping columns, which splits into two sets
    f) Make two COO matrices and do separate aggregation on each one

  4) Solve the Neumann Poisson problem in parallel
    - Try it on GPU machine at Brown (They need another GNU install)

  5) GPU FEM integration
    - Move launch code to PETSc   or   - Try again now that assembly is in PETSc
    - Move build code to PETSc

  6) Try out CUSP PCs
*/

#undef __FUNCT__
#define __FUNCT__ "IntegrateCells"
PetscErrorCode IntegrateCells(DM dm, PetscInt *Ne, PetscInt *Nl, PetscInt **elemRows, PetscScalar **elemMats) {
  DMDALocalInfo  info;
  PetscInt      *er;
  PetscScalar   *em;
  PetscInt       X, Y, dof;
  PetscInt       nl, nxe, nye, ne;
  PetscInt       k  = 0, m  = 0;
  PetscInt       i, j;
  PetscLogEvent  integrationEvent;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscLogEventRegister("ElemIntegration", DM_CLASSID, &integrationEvent);CHKERRQ(ierr);
  ierr = PetscLogEventBegin(integrationEvent,0,0,0,0);CHKERRQ(ierr);
  ierr = DMDAGetInfo(dm, 0, &X, &Y,0,0,0,0, &dof,0,0,0,0,0);CHKERRQ(ierr);
  ierr = DMDAGetLocalInfo(dm, &info);CHKERRQ(ierr);
  nl   = dof*3;
  nxe  = info.xm; if (info.xs+info.xm == X) nxe--;
  nye  = info.ym; if (info.ys+info.ym == Y) nye--;
  ne   = 2 * nxe * nye;
  *Ne  = ne;
  *Nl  = nl;
  ierr = PetscMalloc2(ne*nl, PetscInt, elemRows, ne*nl*nl, PetscScalar, elemMats);CHKERRQ(ierr);
  er   = *elemRows;
  em   = *elemMats;
  // Proc 0        Proc 1
  // xs: 0  xm: 3  xs: 0 xm: 3
  // ys: 0  ym: 2  ys: 2 ym: 1
  // 8 elements x 3 vertices = 24 element matrix rows and 72 entries
  //   6 offproc rows containing 18 element matrix entries
  //  18  onproc rows containing 54 element matrix entries
  //   3 offproc columns in 8 element matrix entries
  //   so we should have 46 diagonal matrix entries
  for(j = info.ys; j < info.ys+nye; ++j) {
    for(i = info.xs; i < info.xs+nxe; ++i) {
      PetscInt rowA = j*X     + i, rowB = j*X     + i+1;
      PetscInt rowC = (j+1)*X + i, rowD = (j+1)*X + i+1;

      /* Lower triangle */
      er[k+0] = rowA; em[m+0*nl+0] =  1.0; em[m+0*nl+1] = -0.5; em[m+0*nl+2] = -0.5;
      er[k+1] = rowB; em[m+1*nl+0] = -0.5; em[m+1*nl+1] =  0.5; em[m+1*nl+2] =  0.0;
      er[k+2] = rowC; em[m+2*nl+0] = -0.5; em[m+2*nl+1] =  0.0; em[m+2*nl+2] =  0.5;
      k += nl; m += nl*nl;
      /* Upper triangle */
      er[k+0] = rowD; em[m+0*nl+0] =  1.0; em[m+0*nl+1] = -0.5; em[m+0*nl+2] = -0.5;
      er[k+1] = rowC; em[m+1*nl+0] = -0.5; em[m+1*nl+1] =  0.5; em[m+1*nl+2] =  0.0;
      er[k+2] = rowB; em[m+2*nl+0] = -0.5; em[m+2*nl+1] =  0.0; em[m+2*nl+2] =  0.5;
      k += nl; m += nl*nl;
    }
  }
  ierr = PetscLogEventEnd(integrationEvent,0,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc, char **argv)
{
  KSP            ksp;
  MatNullSpace   nullsp;
  DM             dm;
  Mat            A;
  Vec            x, b;
  PetscViewer    viewer;
  PetscInt       Nl, Ne;
  PetscInt      *elemRows;
  PetscScalar   *elemMats;
  PetscBool      doGPU = PETSC_TRUE, doCPU = PETSC_TRUE, doSolve = PETSC_FALSE, doView = PETSC_TRUE;
  PetscLogStage  gpuStage, cpuStage;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc, &argv, 0, help);CHKERRQ(ierr);
  ierr = DMDACreate2d(PETSC_COMM_WORLD, DMDA_BOUNDARY_NONE, DMDA_BOUNDARY_NONE, DMDA_STENCIL_BOX, -3, -3, PETSC_DECIDE, PETSC_DECIDE, 1, 1, PETSC_NULL, PETSC_NULL, &dm);CHKERRQ(ierr);
  ierr = IntegrateCells(dm, &Ne, &Nl, &elemRows, &elemMats);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(PETSC_NULL, "-view", &doView, PETSC_NULL);CHKERRQ(ierr);
  /* Construct matrix using GPU */
  ierr = PetscOptionsGetBool(PETSC_NULL, "-gpu", &doGPU, PETSC_NULL);CHKERRQ(ierr);
  if (doGPU) {
    ierr = PetscLogStageRegister("GPU Stage", &gpuStage);CHKERRQ(ierr);
    ierr = PetscLogStagePush(gpuStage);CHKERRQ(ierr);
    ierr = DMCreateMatrix(dm, MATAIJ, &A);CHKERRQ(ierr);
    ierr = MatSetType(A, MATAIJCUSP);CHKERRQ(ierr);
    ierr = MatSeqAIJSetPreallocation(A, 0, PETSC_NULL);CHKERRQ(ierr);
    ierr = MatMPIAIJSetPreallocation(A, 0, PETSC_NULL, 0, PETSC_NULL);CHKERRQ(ierr);
    ierr = MatSetValuesBatch(A, Ne, Nl, elemRows, elemMats);CHKERRQ(ierr);
    ierr = MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    if (doView) {
      ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD, PETSC_NULL, &viewer);CHKERRQ(ierr);
      if (Ne > 500) {ierr = PetscViewerPushFormat(viewer, PETSC_VIEWER_ASCII_INFO);CHKERRQ(ierr);}
      ierr = MatView(A, viewer);CHKERRQ(ierr);
      ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
    }
    ierr = PetscLogStagePop();CHKERRQ(ierr);
    ierr = MatDestroy(&A);CHKERRQ(ierr);
  }
  /* Construct matrix using CPU */
  ierr = PetscOptionsGetBool(PETSC_NULL, "-cpu", &doCPU, PETSC_NULL);CHKERRQ(ierr);
  if (doCPU) {
    ierr = PetscLogStageRegister("CPU Stage", &cpuStage);CHKERRQ(ierr);
    ierr = PetscLogStagePush(cpuStage);CHKERRQ(ierr);
    ierr = DMCreateMatrix(dm, MATAIJ, &A);CHKERRQ(ierr);
    ierr = MatZeroEntries(A);CHKERRQ(ierr);
    ierr = MatSetValuesBatch(A, Ne, Nl, elemRows, elemMats);CHKERRQ(ierr);
    ierr = MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    if (doView) {
      ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD, PETSC_NULL, &viewer);CHKERRQ(ierr);
      if (Ne > 500) {ierr = PetscViewerPushFormat(viewer, PETSC_VIEWER_ASCII_INFO);CHKERRQ(ierr);}
      ierr = MatView(A, viewer);CHKERRQ(ierr);
      ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
    }
    ierr = PetscLogStagePop();CHKERRQ(ierr);
  }
  /* Solve simple system with random rhs */
  ierr = PetscOptionsGetBool(PETSC_NULL, "-solve", &doSolve, PETSC_NULL);CHKERRQ(ierr);
  if (doSolve) {
    ierr = MatGetVecs(A, &x, &b);CHKERRQ(ierr);
    ierr = VecSetRandom(b, PETSC_NULL);CHKERRQ(ierr);
    ierr = KSPCreate(PETSC_COMM_WORLD, &ksp);CHKERRQ(ierr);
    ierr = KSPSetOperators(ksp, A, A, DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
    ierr = MatNullSpaceCreate(PETSC_COMM_WORLD, PETSC_TRUE, 0, PETSC_NULL, &nullsp);CHKERRQ(ierr);
    ierr = KSPSetNullSpace(ksp, nullsp);CHKERRQ(ierr);
    ierr = MatNullSpaceDestroy(&nullsp);CHKERRQ(ierr);
    ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);
    ierr = KSPSolve(ksp, b, x);CHKERRQ(ierr);
    ierr = VecDestroy(&x);CHKERRQ(ierr);
    ierr = VecDestroy(&b);CHKERRQ(ierr);
    /* Solve physical system:

         -\Delta u = -6 (x + y - 1)

       where u = x^3 - 3/2 x^2 + y^3 - 3/2y^2 + 1/2,
       so \Delta u = 6 x - 3 + 6 y - 3,
       and \frac{\partial u}{\partial n} = {3x (x - 1), 3y (y - 1)} \cdot n
                                         = \pm 3x (x - 1) at x=0,1 = 0
                                         = \pm 3y (y - 1) at y=0,1 = 0
    */
  }
  /* Cleanup */
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = PetscFree2(elemRows, elemMats);CHKERRQ(ierr);
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return 0;
}
