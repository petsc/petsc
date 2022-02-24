
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
  CHKERRMPI(MPI_Comm_rank(comm, &mype));
  CHKERRMPI(MPI_Comm_size(comm, &npe));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-ne",&ne,NULL));
  h     = 1./ne;
  /* ne*ne; number of global elements */
  CHKERRQ(PetscOptionsGetReal(NULL,NULL,"-alpha",&soft_alpha,NULL));
  M    = (ne+1)*(ne+1); /* global number of nodes */

  /* create stiffness matrix (2) */
  CHKERRQ(MatCreate(comm,&Amat));
  CHKERRQ(MatSetSizes(Amat,PETSC_DECIDE,PETSC_DECIDE,M,M));
  CHKERRQ(MatSetType(Amat,MATAIJ));
  CHKERRQ(MatSetOption(Amat,MAT_SPD,PETSC_TRUE));
  CHKERRQ(MatSetFromOptions(Amat));
  CHKERRQ(MatSeqAIJSetPreallocation(Amat,81,NULL));
  CHKERRQ(MatMPIAIJSetPreallocation(Amat,81,NULL,57,NULL));

  CHKERRQ(MatCreate(comm,&Pmat));
  CHKERRQ(MatSetSizes(Pmat,PETSC_DECIDE,PETSC_DECIDE,M,M));
  CHKERRQ(MatSetType(Pmat,MATMPIAIJ));
  CHKERRQ(MatSetFromOptions(Pmat));
  CHKERRQ(MatSeqAIJSetPreallocation(Pmat,81,NULL));
  CHKERRQ(MatMPIAIJSetPreallocation(Pmat,81,NULL,57,NULL));

  /* vectors */
  CHKERRQ(MatCreateVecs(Amat,&bb,&xx));
  CHKERRQ(VecSet(bb,.0));
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
    CHKERRQ(MatGetOwnershipRange(Amat,&Istart,&Iend));
    m    = Iend-Istart;
    CHKERRQ(PetscMalloc1(2*m,&coords));
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
        CHKERRQ(MatSetValues(Pmat,4,idx,4,idx,(const PetscScalar*)DD,ADD_VALUES));
        if (j>0) {
          CHKERRQ(MatSetValues(Amat,4,idx,4,idx,(const PetscScalar*)DD,ADD_VALUES));
        } else {
          /* a BC */
          for (ii=0;ii<4;ii++) {
            for (jj=0;jj<4;jj++) DD[ii][jj] = alpha*DD2[ii][jj];
          }
          CHKERRQ(MatSetValues(Amat,4,idx,4,idx,(const PetscScalar*)DD,ADD_VALUES));
        }
      }
      if (j>0) {
        PetscScalar v  = h*h;
        PetscInt    jj = Ii;
        CHKERRQ(VecSetValues(bb,1,&jj,&v,INSERT_VALUES));
      }
    }
    CHKERRQ(MatAssemblyBegin(Amat,MAT_FINAL_ASSEMBLY));
    CHKERRQ(MatAssemblyEnd(Amat,MAT_FINAL_ASSEMBLY));
    CHKERRQ(MatAssemblyBegin(Pmat,MAT_FINAL_ASSEMBLY));
    CHKERRQ(MatAssemblyEnd(Pmat,MAT_FINAL_ASSEMBLY));
    CHKERRQ(VecAssemblyBegin(bb));
    CHKERRQ(VecAssemblyEnd(bb));

    /* Setup solver */
    CHKERRQ(KSPCreate(PETSC_COMM_WORLD,&ksp));
    CHKERRQ(KSPSetFromOptions(ksp));

    /* finish KSP/PC setup */
    CHKERRQ(KSPSetOperators(ksp, Amat, Amat));

    CHKERRQ(KSPGetPC(ksp,&pc));
    CHKERRQ(PCSetCoordinates(pc, 2, m, coords));
    CHKERRQ(PetscFree(coords));
  }

  if (!PETSC_TRUE) {
    PetscViewer viewer;
    CHKERRQ(PetscViewerASCIIOpen(comm, "Amat.m", &viewer));
    CHKERRQ(PetscViewerPushFormat(viewer, PETSC_VIEWER_ASCII_MATLAB));
    CHKERRQ(MatView(Amat,viewer));
    CHKERRQ(PetscViewerPopFormat(viewer));
    CHKERRQ(PetscViewerDestroy(&viewer));
  }

  /* solve */
#if defined(PETSC_USE_LOG)
  CHKERRQ(PetscLogStageRegister("Solve", &stage));
  CHKERRQ(PetscLogStagePush(stage));
#endif
  CHKERRQ(VecSet(xx,.0));

  CHKERRQ(KSPSetUp(ksp));

  CHKERRQ(KSPSolve(ksp,bb,xx));

#if defined(PETSC_USE_LOG)
  CHKERRQ(PetscLogStagePop());
#endif

  CHKERRQ(KSPGetIterationNumber(ksp,&its));

  if (!PETSC_TRUE) {
    PetscReal   norm,norm2;
    PetscViewer viewer;
    Vec         res;
    CHKERRQ(PetscViewerASCIIOpen(comm, "rhs.m", &viewer));
    CHKERRQ(PetscViewerPushFormat(viewer, PETSC_VIEWER_ASCII_MATLAB));
    CHKERRQ(VecView(bb,viewer));
    CHKERRQ(PetscViewerPopFormat(viewer));
    CHKERRQ(PetscViewerDestroy(&viewer));
    CHKERRQ(VecNorm(bb, NORM_2, &norm2));

    CHKERRQ(PetscViewerASCIIOpen(comm, "solution.m", &viewer));
    CHKERRQ(PetscViewerPushFormat(viewer, PETSC_VIEWER_ASCII_MATLAB));
    CHKERRQ(VecView(xx,viewer));
    CHKERRQ(PetscViewerPopFormat(viewer));
    CHKERRQ(PetscViewerDestroy(&viewer));

    CHKERRQ(VecDuplicate(xx, &res));
    CHKERRQ(MatMult(Amat, xx, res));
    CHKERRQ(VecAXPY(bb, -1.0, res));
    CHKERRQ(VecDestroy(&res));
    CHKERRQ(VecNorm(bb,NORM_2,&norm));
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"[%d]%s |b-Ax|/|b|=%e, |b|=%e\n",0,PETSC_FUNCTION_NAME,norm/norm2,norm2));

    CHKERRQ(PetscViewerASCIIOpen(comm, "residual.m", &viewer));
    CHKERRQ(PetscViewerPushFormat(viewer, PETSC_VIEWER_ASCII_MATLAB));
    CHKERRQ(VecView(bb,viewer));
    CHKERRQ(PetscViewerPopFormat(viewer));
    CHKERRQ(PetscViewerDestroy(&viewer));
  }

  /* Free work space */
  CHKERRQ(KSPDestroy(&ksp));
  CHKERRQ(VecDestroy(&xx));
  CHKERRQ(VecDestroy(&bb));
  CHKERRQ(MatDestroy(&Amat));
  CHKERRQ(MatDestroy(&Pmat));

  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   build:
      requires: !complex

   test:
      nsize: 4
      args: -ne 19 -alpha 1.e-3 -ksp_type cg -pc_type gamg -mg_levels_ksp_max_it 2 -ksp_monitor -ksp_converged_reason -pc_gamg_esteig_ksp_max_it 5 -pc_gamg_esteig_ksp_type cg -mg_levels_ksp_chebyshev_esteig 0,0.25,0,1.1

   test:
      suffix: seqaijmkl
      nsize: 4
      requires: mkl_sparse
      args: -ne 19 -alpha 1.e-3 -ksp_type cg -pc_type gamg -mg_levels_ksp_max_it 2 -ksp_monitor -ksp_converged_reason -pc_gamg_esteig_ksp_max_it 5 -pc_gamg_esteig_ksp_type cg -mg_levels_ksp_chebyshev_esteig 0,0.25,0,1.1 -mat_seqaij_type seqaijmkl

   test:
      suffix: Classical
      args: -ne 49 -alpha 1.e-3 -ksp_type cg -pc_type gamg -mg_levels_ksp_max_it 2 -pc_gamg_type classical -ksp_monitor -ksp_converged_reason -mg_levels_esteig_ksp_type cg -mg_levels_ksp_chebyshev_esteig 0,0.25,0,1.1
      output_file: output/ex54_classical.out

   test:
      suffix: geo
      nsize: 4
      args: -ne 49 -alpha 1.e-3 -ksp_type cg -pc_type gamg -mg_levels_ksp_max_it 4 -pc_gamg_type geo -pc_gamg_coarse_eq_limit 200 -mg_levels_esteig_ksp_type cg -mg_levels_esteig_ksp_max_it 10 -mg_levels_ksp_chebyshev_esteig 0,0.1,0,1.05 -ksp_monitor_short -ksp_converged_reason -ksp_rtol 1e-3 -ksp_norm_type unpreconditioned
      requires: triangle
      output_file: output/ex54_0.out

TEST*/
