
static char help[] = "Creates a matrix from quadrilateral finite elements in 2D, Laplacian \n\
  -ne <size>       : problem size in number of elements (eg, -ne 31 gives 32^2 grid)\n\
  -alpha <v>      : scaling of material coeficient in embedded circle\n\n";

#include <petscksp.h>

int main(int argc,char **args)
{
  Mat            Amat,Pmat;
  PetscErrorCode ierr;
  PetscInt       i,m,M,its,Istart,Iend,j,Ii,ix,ne=4;
  PetscReal      x,y,h;
  Vec            xx,bb;
  KSP            ksp;
  PetscReal      soft_alpha = 1.e-3;
  MPI_Comm       comm;
  PetscMPIInt    npe,mype;
  PetscScalar    DD[4][4],DD2[4][4];
#if defined(PETSC_USE_LOG)
  PetscLogStage stage;
#endif
#define DIAG_S 0.0
  PetscScalar DD1[4][4] = { {5.0+DIAG_S, -2.0, -1.0, -2.0},
                            {-2.0, 5.0+DIAG_S, -2.0, -1.0},
                            {-1.0, -2.0, 5.0+DIAG_S, -2.0},
                            {-2.0, -1.0, -2.0, 5.0+DIAG_S} };

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  comm = PETSC_COMM_WORLD;
  ierr  = MPI_Comm_rank(comm, &mype);CHKERRQ(ierr);
  ierr  = MPI_Comm_size(comm, &npe);CHKERRQ(ierr);
  ierr  = PetscOptionsGetInt(NULL,NULL,"-ne",&ne,NULL);CHKERRQ(ierr);
  h     = 1./ne;
  /* ne*ne; number of global elements */
  ierr = PetscOptionsGetReal(NULL,NULL,"-alpha",&soft_alpha,NULL);CHKERRQ(ierr);
  M    = (ne+1)*(ne+1); /* global number of nodes */

  /* create stiffness matrix (2) */
  ierr = MatCreate(comm,&Amat);CHKERRQ(ierr);
  ierr = MatSetSizes(Amat,PETSC_DECIDE,PETSC_DECIDE,M,M);CHKERRQ(ierr);
  ierr = MatSetType(Amat,MATAIJ);CHKERRQ(ierr);
  ierr = MatSetOption(Amat,MAT_SPD,PETSC_TRUE);CHKERRQ(ierr);
  ierr = MatSetFromOptions(Amat);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(Amat,81,NULL);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(Amat,81,NULL,57,NULL);CHKERRQ(ierr);

  ierr = MatCreate(comm,&Pmat);CHKERRQ(ierr);
  ierr = MatSetSizes(Pmat,PETSC_DECIDE,PETSC_DECIDE,M,M);CHKERRQ(ierr);
  ierr = MatSetType(Pmat,MATMPIAIJ);CHKERRQ(ierr);
  ierr = MatSetFromOptions(Pmat);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(Pmat,81,NULL);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(Pmat,81,NULL,57,NULL);CHKERRQ(ierr);

  /* vectors */
  ierr = MatCreateVecs(Amat,&bb,&xx);CHKERRQ(ierr);
  ierr = VecSet(bb,.0);CHKERRQ(ierr);
  /* generate element matrices -- see ex56.c on how to use different data set */
  {
    DD1[0][0] =  0.66666666666666663;
    DD1[0][1] = -0.16666666666666669;
    DD1[0][2] = -0.33333333333333343;
    DD1[0][3] = -0.16666666666666666;
    DD1[1][0] = -0.16666666666666669;
    DD1[1][1] =  0.66666666666666663;
    DD1[1][2] = -0.16666666666666666;
    DD1[1][3] = -0.33333333333333343;
    DD1[2][0] = -0.33333333333333343;
    DD1[2][1] = -0.16666666666666666;
    DD1[2][2] =  0.66666666666666663;
    DD1[2][3] = -0.16666666666666663;
    DD1[3][0] = -0.16666666666666666;
    DD1[3][1] = -0.33333333333333343;
    DD1[3][2] = -0.16666666666666663;
    DD1[3][3] =  0.66666666666666663;

    /* BC version of element */
    for (i=0;i<4;i++) {
      for (j=0;j<4;j++) {
        if (i<2 || j < 2) {
          if (i==j) DD2[i][j] = .1*DD1[i][j];
          else DD2[i][j] = 0.0;
        } else DD2[i][j] = DD1[i][j];
      }
    }
  }
  {
    PetscReal *coords;
    PC             pc;
    /* forms the element stiffness for the Laplacian and coordinates */
    ierr = MatGetOwnershipRange(Amat,&Istart,&Iend);CHKERRQ(ierr);
    m    = Iend-Istart;
    ierr = PetscMalloc1(2*m,&coords);CHKERRQ(ierr);
    for (Ii=Istart,ix=0; Ii<Iend; Ii++,ix++) {
      j = Ii/(ne+1); i = Ii%(ne+1);
      /* coords */
      x            = h*(Ii % (ne+1)); y = h*(Ii/(ne+1));
      coords[2*ix] = x; coords[2*ix+1] = y;
      if (i<ne && j<ne) {
        PetscInt jj,ii,idx[4];
        /* radius */
        PetscReal radius = PetscSqrtReal((x-.5+h/2)*(x-.5+h/2) + (y-.5+h/2)*(y-.5+h/2));
        PetscReal alpha  = 1.0;
        idx[0] = Ii; idx[1] = Ii+1; idx[2] = Ii + (ne+1) + 1; idx[3] =  Ii + (ne+1);
        if (radius < 0.25) alpha = soft_alpha;
        for (ii=0; ii<4; ii++) {
          for (jj=0; jj<4; jj++) DD[ii][jj] = alpha*DD1[ii][jj];
        }
        ierr = MatSetValues(Pmat,4,idx,4,idx,(const PetscScalar*)DD,ADD_VALUES);CHKERRQ(ierr);
        if (j>0) {
          ierr = MatSetValues(Amat,4,idx,4,idx,(const PetscScalar*)DD,ADD_VALUES);CHKERRQ(ierr);
        } else {
          /* a BC */
          for (ii=0;ii<4;ii++) {
            for (jj=0;jj<4;jj++) DD[ii][jj] = alpha*DD2[ii][jj];
          }
          ierr = MatSetValues(Amat,4,idx,4,idx,(const PetscScalar*)DD,ADD_VALUES);CHKERRQ(ierr);
        }
      }
      if (j>0) {
        PetscScalar v  = h*h;
        PetscInt    jj = Ii;
        ierr = VecSetValues(bb,1,&jj,&v,INSERT_VALUES);CHKERRQ(ierr);
      }
    }
    ierr = MatAssemblyBegin(Amat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(Amat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyBegin(Pmat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(Pmat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = VecAssemblyBegin(bb);CHKERRQ(ierr);
    ierr = VecAssemblyEnd(bb);CHKERRQ(ierr);

    /* Setup solver */
    ierr = KSPCreate(PETSC_COMM_WORLD,&ksp);CHKERRQ(ierr);
    ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);

    /* finish KSP/PC setup */
    ierr = KSPSetOperators(ksp, Amat, Amat);CHKERRQ(ierr);

    ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
    ierr = PCSetCoordinates(pc, 2, m, coords);CHKERRQ(ierr);
    ierr = PetscFree(coords);CHKERRQ(ierr);
  }

  if (!PETSC_TRUE) {
    PetscViewer viewer;
    ierr = PetscViewerASCIIOpen(comm, "Amat.m", &viewer);CHKERRQ(ierr);
    ierr = PetscViewerPushFormat(viewer, PETSC_VIEWER_ASCII_MATLAB);CHKERRQ(ierr);
    ierr = MatView(Amat,viewer);CHKERRQ(ierr);
    ierr = PetscViewerPopFormat(viewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer);
  }

  /* solve */
#if defined(PETSC_USE_LOG)
  ierr = PetscLogStageRegister("Solve", &stage);CHKERRQ(ierr);
  ierr = PetscLogStagePush(stage);CHKERRQ(ierr);
#endif
  ierr = VecSet(xx,.0);CHKERRQ(ierr);

  ierr = KSPSetUp(ksp);CHKERRQ(ierr);

  ierr = KSPSolve(ksp,bb,xx);CHKERRQ(ierr);

#if defined(PETSC_USE_LOG)
  ierr = PetscLogStagePop();CHKERRQ(ierr);
#endif

  ierr = KSPGetIterationNumber(ksp,&its);CHKERRQ(ierr);

  if (!PETSC_TRUE) {
    PetscReal   norm,norm2;
    PetscViewer viewer;
    Vec         res;
    ierr = PetscViewerASCIIOpen(comm, "rhs.m", &viewer);CHKERRQ(ierr);
    ierr = PetscViewerPushFormat(viewer, PETSC_VIEWER_ASCII_MATLAB);CHKERRQ(ierr);
    ierr = VecView(bb,viewer);CHKERRQ(ierr);
    ierr = PetscViewerPopFormat(viewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer);
    ierr = VecNorm(bb, NORM_2, &norm2);CHKERRQ(ierr);

    ierr = PetscViewerASCIIOpen(comm, "solution.m", &viewer);CHKERRQ(ierr);
    ierr = PetscViewerPushFormat(viewer, PETSC_VIEWER_ASCII_MATLAB);CHKERRQ(ierr);
    ierr = VecView(xx,viewer);CHKERRQ(ierr);
    ierr = PetscViewerPopFormat(viewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer);

    ierr = VecDuplicate(xx, &res);CHKERRQ(ierr);
    ierr = MatMult(Amat, xx, res);CHKERRQ(ierr);
    ierr = VecAXPY(bb, -1.0, res);CHKERRQ(ierr);
    ierr = VecDestroy(&res);CHKERRQ(ierr);
    ierr = VecNorm(bb,NORM_2,&norm);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"[%d]%s |b-Ax|/|b|=%e, |b|=%e\n",0,PETSC_FUNCTION_NAME,norm/norm2,norm2);CHKERRQ(ierr);

    ierr = PetscViewerASCIIOpen(comm, "residual.m", &viewer);CHKERRQ(ierr);
    ierr = PetscViewerPushFormat(viewer, PETSC_VIEWER_ASCII_MATLAB);CHKERRQ(ierr);
    ierr = VecView(bb,viewer);CHKERRQ(ierr);
    ierr = PetscViewerPopFormat(viewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer);
  }

  /* Free work space */
  ierr = KSPDestroy(&ksp);CHKERRQ(ierr);
  ierr = VecDestroy(&xx);CHKERRQ(ierr);
  ierr = VecDestroy(&bb);CHKERRQ(ierr);
  ierr = MatDestroy(&Amat);CHKERRQ(ierr);
  ierr = MatDestroy(&Pmat);CHKERRQ(ierr);

  ierr = PetscFinalize();
  return ierr;
}



/*TEST

   build:
      requires: !complex

   test:
      nsize: 4
      args: -ne 19 -alpha 1.e-3 -pc_type gamg -pc_gamg_agg_nsmooths 1  -mg_levels_ksp_max_it 3 -ksp_monitor -ksp_converged_reason -ksp_type cg

   test:
      suffix: seqaijmkl
      nsize: 4
      requires: mkl_sparse
      args: -ne 19 -alpha 1.e-3 -pc_type gamg -pc_gamg_agg_nsmooths 1  -mg_levels_ksp_max_it 3 -ksp_monitor -ksp_converged_reason -ksp_type cg -mat_seqaij_type seqaijmkl

   test:
      suffix: Classical
      args: -ne 49 -alpha 1.e-3 -ksp_type cg -pc_type gamg -pc_gamg_type classical -mg_levels_ksp_chebyshev_esteig 0,0.05,0,1.05 -ksp_converged_reason
      output_file: output/ex54_classical.out

   test:
      suffix: geo
      nsize: 4
      args: -ne 49 -alpha 1.e-3 -ksp_type cg -pc_type gamg -pc_gamg_type geo -pc_gamg_coarse_eq_limit 200 -mg_levels_pc_type jacobi -mg_levels_ksp_chebyshev_esteig 0,0.05,0,1.05 -ksp_monitor_short -mg_levels_ksp_max_it 3
      requires: triangle
      output_file: output/ex54_0.out

TEST*/
