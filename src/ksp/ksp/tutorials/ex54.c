
static char help[] = "Creates a matrix from quadrilateral finite elements in 2D, Laplacian \n\
  -ne <size>       : problem size in number of elements (eg, -ne 31 gives 32^2 grid)\n\
  -alpha <v>      : scaling of material coefficient in embedded circle\n\n";

#include <petscksp.h>

int main(int argc,char **args)
{
  Mat            Amat,Pmat;
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

  PetscCall(PetscInitialize(&argc,&args,(char*)0,help));
  comm = PETSC_COMM_WORLD;
  PetscCallMPI(MPI_Comm_rank(comm, &mype));
  PetscCallMPI(MPI_Comm_size(comm, &npe));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-ne",&ne,NULL));
  h     = 1./ne;
  /* ne*ne; number of global elements */
  PetscCall(PetscOptionsGetReal(NULL,NULL,"-alpha",&soft_alpha,NULL));
  M    = (ne+1)*(ne+1); /* global number of nodes */

  /* create stiffness matrix (2) */
  PetscCall(MatCreate(comm,&Amat));
  PetscCall(MatSetSizes(Amat,PETSC_DECIDE,PETSC_DECIDE,M,M));
  PetscCall(MatSetType(Amat,MATAIJ));
  PetscCall(MatSetOption(Amat,MAT_SPD,PETSC_TRUE));
  PetscCall(MatSetOption(Amat,MAT_SPD_ETERNAL,PETSC_TRUE));
  PetscCall(MatSetFromOptions(Amat));
  PetscCall(MatSeqAIJSetPreallocation(Amat,81,NULL));
  PetscCall(MatMPIAIJSetPreallocation(Amat,81,NULL,57,NULL));

  PetscCall(MatCreate(comm,&Pmat));
  PetscCall(MatSetSizes(Pmat,PETSC_DECIDE,PETSC_DECIDE,M,M));
  PetscCall(MatSetType(Pmat,MATMPIAIJ));
  PetscCall(MatSetFromOptions(Pmat));
  PetscCall(MatSeqAIJSetPreallocation(Pmat,81,NULL));
  PetscCall(MatMPIAIJSetPreallocation(Pmat,81,NULL,57,NULL));

  /* vectors */
  PetscCall(MatCreateVecs(Amat,&bb,&xx));
  PetscCall(VecSet(bb,.0));
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
    PetscCall(MatGetOwnershipRange(Amat,&Istart,&Iend));
    m    = Iend-Istart;
    PetscCall(PetscMalloc1(2*m,&coords));
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
        PetscCall(MatSetValues(Pmat,4,idx,4,idx,(const PetscScalar*)DD,ADD_VALUES));
        if (j>0) {
          PetscCall(MatSetValues(Amat,4,idx,4,idx,(const PetscScalar*)DD,ADD_VALUES));
        } else {
          /* a BC */
          for (ii=0;ii<4;ii++) {
            for (jj=0;jj<4;jj++) DD[ii][jj] = alpha*DD2[ii][jj];
          }
          PetscCall(MatSetValues(Amat,4,idx,4,idx,(const PetscScalar*)DD,ADD_VALUES));
        }
      }
      if (j>0) {
        PetscScalar v  = h*h;
        PetscInt    jj = Ii;
        PetscCall(VecSetValues(bb,1,&jj,&v,INSERT_VALUES));
      }
    }
    PetscCall(MatAssemblyBegin(Amat,MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(Amat,MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyBegin(Pmat,MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(Pmat,MAT_FINAL_ASSEMBLY));
    PetscCall(VecAssemblyBegin(bb));
    PetscCall(VecAssemblyEnd(bb));

    /* Setup solver */
    PetscCall(KSPCreate(PETSC_COMM_WORLD,&ksp));
    PetscCall(KSPSetFromOptions(ksp));

    /* finish KSP/PC setup */
    PetscCall(KSPSetOperators(ksp, Amat, Amat));

    PetscCall(KSPGetPC(ksp,&pc));
    PetscCall(PCSetCoordinates(pc, 2, m, coords));
    PetscCall(PetscFree(coords));
  }

  if (!PETSC_TRUE) {
    PetscViewer viewer;
    PetscCall(PetscViewerASCIIOpen(comm, "Amat.m", &viewer));
    PetscCall(PetscViewerPushFormat(viewer, PETSC_VIEWER_ASCII_MATLAB));
    PetscCall(MatView(Amat,viewer));
    PetscCall(PetscViewerPopFormat(viewer));
    PetscCall(PetscViewerDestroy(&viewer));
  }

  /* solve */
#if defined(PETSC_USE_LOG)
  PetscCall(PetscLogStageRegister("Solve", &stage));
  PetscCall(PetscLogStagePush(stage));
#endif
  PetscCall(VecSet(xx,.0));

  PetscCall(KSPSetUp(ksp));

  PetscCall(KSPSolve(ksp,bb,xx));

#if defined(PETSC_USE_LOG)
  PetscCall(PetscLogStagePop());
#endif

  PetscCall(KSPGetIterationNumber(ksp,&its));

  if (!PETSC_TRUE) {
    PetscReal   norm,norm2;
    PetscViewer viewer;
    Vec         res;
    PetscCall(PetscViewerASCIIOpen(comm, "rhs.m", &viewer));
    PetscCall(PetscViewerPushFormat(viewer, PETSC_VIEWER_ASCII_MATLAB));
    PetscCall(VecView(bb,viewer));
    PetscCall(PetscViewerPopFormat(viewer));
    PetscCall(PetscViewerDestroy(&viewer));
    PetscCall(VecNorm(bb, NORM_2, &norm2));

    PetscCall(PetscViewerASCIIOpen(comm, "solution.m", &viewer));
    PetscCall(PetscViewerPushFormat(viewer, PETSC_VIEWER_ASCII_MATLAB));
    PetscCall(VecView(xx,viewer));
    PetscCall(PetscViewerPopFormat(viewer));
    PetscCall(PetscViewerDestroy(&viewer));

    PetscCall(VecDuplicate(xx, &res));
    PetscCall(MatMult(Amat, xx, res));
    PetscCall(VecAXPY(bb, -1.0, res));
    PetscCall(VecDestroy(&res));
    PetscCall(VecNorm(bb,NORM_2,&norm));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"[%d]%s |b-Ax|/|b|=%e, |b|=%e\n",0,PETSC_FUNCTION_NAME,(double)(norm/norm2),(double)norm2));

    PetscCall(PetscViewerASCIIOpen(comm, "residual.m", &viewer));
    PetscCall(PetscViewerPushFormat(viewer, PETSC_VIEWER_ASCII_MATLAB));
    PetscCall(VecView(bb,viewer));
    PetscCall(PetscViewerPopFormat(viewer));
    PetscCall(PetscViewerDestroy(&viewer));
  }

  /* Free work space */
  PetscCall(KSPDestroy(&ksp));
  PetscCall(VecDestroy(&xx));
  PetscCall(VecDestroy(&bb));
  PetscCall(MatDestroy(&Amat));
  PetscCall(MatDestroy(&Pmat));

  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   build:
      requires: !complex

   test:
      nsize: 4
      args: -ne 19 -alpha 1.e-3 -ksp_type cg -pc_type gamg -mg_levels_ksp_max_it 2 -ksp_monitor -ksp_converged_reason -pc_gamg_esteig_ksp_max_it 5 -pc_gamg_esteig_ksp_type cg -mg_levels_ksp_chebyshev_esteig 0,0.25,0,1.1 -pc_gamg_aggressive_coarsening 0

   test:
      suffix: seqaijmkl
      nsize: 4
      requires: mkl_sparse
      args: -ne 19 -alpha 1.e-3 -ksp_type cg -pc_type gamg -mg_levels_ksp_max_it 2 -ksp_monitor -ksp_converged_reason -pc_gamg_esteig_ksp_max_it 5 -pc_gamg_esteig_ksp_type cg -mg_levels_ksp_chebyshev_esteig 0,0.25,0,1.1 -mat_seqaij_type seqaijmkl -pc_gamg_aggressive_coarsening 0

   test:
      suffix: Classical
      args: -ne 49 -alpha 1.e-3 -ksp_type cg -pc_type gamg -mg_levels_ksp_max_it 2 -pc_gamg_type classical -ksp_monitor -ksp_converged_reason -mg_levels_esteig_ksp_type cg -mg_levels_ksp_chebyshev_esteig 0,0.25,0,1.1 -mat_coarsen_type mis
      output_file: output/ex54_classical.out

   test:
      suffix: geo
      nsize: 4
      args: -ne 49 -alpha 1.e-3 -ksp_type cg -pc_type gamg -mg_levels_ksp_max_it 4 -pc_gamg_type geo -pc_gamg_coarse_eq_limit 200 -mg_levels_esteig_ksp_type cg -mg_levels_esteig_ksp_max_it 10 -mg_levels_ksp_chebyshev_esteig 0,0.1,0,1.05 -ksp_monitor_short -ksp_converged_reason -ksp_rtol 1e-3 -ksp_norm_type unpreconditioned
      requires: triangle
      output_file: output/ex54_0.out

TEST*/
