static char help[] = "2D, bi-linear quadrilateral (Q1), displacement finite element formulation\n\
of plain strain linear elasticity.  E=1.0, nu=0.25.\n\
Unit square domain with Dirichelet boundary condition on the y=0 side only.\n\
Load of 1.0 in x direction on all nodes (not a true uniform load).\n\
  -ne <size>      : number of (square) quadrilateral elements in each dimension\n\
  -alpha <v>      : scaling of material coefficient in embedded circle\n\n";

#include <petscksp.h>

int main(int argc,char **args)
{
  Mat            Amat;
  PetscInt       i,m,M,its,Istart,Iend,j,Ii,ix,ne=4;
  PetscReal      x,y,h;
  Vec            xx,bb;
  KSP            ksp;
  PetscReal      soft_alpha = 1.e-3;
  MPI_Comm       comm;
  PetscBool      use_coords = PETSC_FALSE;
  PetscMPIInt    npe,mype;
  PetscScalar    DD[8][8],DD2[8][8];
#if defined(PETSC_USE_LOG)
  PetscLogStage stage[2];
#endif
  PetscScalar DD1[8][8] = {  {5.333333333333333E-01,  2.0000E-01, -3.333333333333333E-01,  0.0000E+00, -2.666666666666667E-01, -2.0000E-01, 6.666666666666667E-02, 0.0000E-00 },
                             {2.0000E-01,  5.333333333333333E-01,  0.0000E-00,  6.666666666666667E-02, -2.0000E-01, -2.666666666666667E-01, 0.0000E-00, -3.333333333333333E-01 },
                             {-3.333333333333333E-01,  0.0000E-00,  5.333333333333333E-01, -2.0000E-01,  6.666666666666667E-02, 0.0000E-00, -2.666666666666667E-01,  2.0000E-01 },
                             {0.0000E+00,  6.666666666666667E-02, -2.0000E-01,  5.333333333333333E-01,  0.0000E-00, -3.333333333333333E-01, 2.0000E-01, -2.666666666666667E-01 },
                             {-2.666666666666667E-01, -2.0000E-01,  6.666666666666667E-02,  0.0000E-00,  5.333333333333333E-01,  2.0000E-01, -3.333333333333333E-01,  0.0000E+00 },
                             {-2.0000E-01, -2.666666666666667E-01, 0.0000E-00, -3.333333333333333E-01,  2.0000E-01,  5.333333333333333E-01, 0.0000E-00,  6.666666666666667E-02 },
                             {6.666666666666667E-02, 0.0000E-00, -2.666666666666667E-01,  2.0000E-01, -3.333333333333333E-01,  0.0000E-00, 5.333333333333333E-01, -2.0000E-01 },
                             {0.0000E-00, -3.333333333333333E-01,  2.0000E-01, -2.666666666666667E-01, 0.0000E-00,  6.666666666666667E-02, -2.0000E-01,  5.333333333333333E-01 } };

  PetscCall(PetscInitialize(&argc,&args,(char*)0,help));
  comm = PETSC_COMM_WORLD;
  PetscCallMPI(MPI_Comm_rank(comm, &mype));
  PetscCallMPI(MPI_Comm_size(comm, &npe));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-ne",&ne,NULL));
  h    = 1./ne;
  /* ne*ne; number of global elements */
  PetscCall(PetscOptionsGetReal(NULL,NULL,"-alpha",&soft_alpha,NULL));
  PetscCall(PetscOptionsGetBool(NULL,NULL,"-use_coordinates",&use_coords,NULL));
  M    = 2*(ne+1)*(ne+1); /* global number of equations */
  m    = (ne+1)*(ne+1)/npe;
  if (mype==npe-1) m = (ne+1)*(ne+1) - (npe-1)*m;
  m *= 2;
  /* create stiffness matrix */
  PetscCall(MatCreate(comm,&Amat));
  PetscCall(MatSetSizes(Amat,m,m,M,M));
  PetscCall(MatSetType(Amat,MATAIJ));
  PetscCall(MatSetOption(Amat,MAT_SPD,PETSC_TRUE));
  PetscCall(MatSetFromOptions(Amat));
  PetscCall(MatSetBlockSize(Amat,2));
  PetscCall(MatSeqAIJSetPreallocation(Amat,18,NULL));
  PetscCall(MatMPIAIJSetPreallocation(Amat,18,NULL,18,NULL));
#if defined(PETSC_HAVE_HYPRE)
  PetscCall(MatHYPRESetPreallocation(Amat,18,NULL,18,NULL));
#endif

  PetscCall(MatGetOwnershipRange(Amat,&Istart,&Iend));
  PetscCheck(m == Iend - Istart,PETSC_COMM_SELF,PETSC_ERR_PLIB,"m %" PetscInt_FMT " does not equal Iend %" PetscInt_FMT " - Istart %" PetscInt_FMT,m,Iend,Istart);
  /* Generate vectors */
  PetscCall(MatCreateVecs(Amat,&xx,&bb));
  PetscCall(VecSet(bb,.0));
  /* generate element matrices -- see ex56.c on how to use different data set */
  {
      DD[0][0] =  0.53333333333333321;
      DD[0][1] =  0.20000000000000001;
      DD[0][2] = -0.33333333333333331;
      DD[0][3] =   0.0000000000000000;
      DD[0][4] = -0.26666666666666666;
      DD[0][5] = -0.20000000000000001;
      DD[0][6] =  6.66666666666666796E-002;
      DD[0][7] =  6.93889390390722838E-018;
      DD[1][0] =  0.20000000000000001;
      DD[1][1] =  0.53333333333333333;
      DD[1][2] =  7.80625564189563192E-018;
      DD[1][3] =  6.66666666666666935E-002;
      DD[1][4] = -0.20000000000000001;
      DD[1][5] = -0.26666666666666666;
      DD[1][6] = -3.46944695195361419E-018;
      DD[1][7] = -0.33333333333333331;
      DD[2][0] = -0.33333333333333331;
      DD[2][1] =  1.12757025938492461E-017;
      DD[2][2] =  0.53333333333333333;
      DD[2][3] = -0.20000000000000001;
      DD[2][4] =  6.66666666666666935E-002;
      DD[2][5] = -6.93889390390722838E-018;
      DD[2][6] = -0.26666666666666666;
      DD[2][7] =  0.19999999999999998;
      DD[3][0] =   0.0000000000000000;
      DD[3][1] =  6.66666666666666935E-002;
      DD[3][2] = -0.20000000000000001;
      DD[3][3] =  0.53333333333333333;
      DD[3][4] =  4.33680868994201774E-018;
      DD[3][5] = -0.33333333333333331;
      DD[3][6] =  0.20000000000000001;
      DD[3][7] = -0.26666666666666666;
      DD[4][0] = -0.26666666666666666;
      DD[4][1] = -0.20000000000000001;
      DD[4][2] =  6.66666666666666935E-002;
      DD[4][3] =  8.67361737988403547E-019;
      DD[4][4] =  0.53333333333333333;
      DD[4][5] =  0.19999999999999998;
      DD[4][6] = -0.33333333333333331;
      DD[4][7] = -3.46944695195361419E-018;
      DD[5][0] = -0.20000000000000001;
      DD[5][1] = -0.26666666666666666;
      DD[5][2] = -1.04083408558608426E-017;
      DD[5][3] = -0.33333333333333331;
      DD[5][4] =  0.19999999999999998;
      DD[5][5] =  0.53333333333333333;
      DD[5][6] =  6.93889390390722838E-018;
      DD[5][7] =  6.66666666666666519E-002;
      DD[6][0] =  6.66666666666666796E-002;
      DD[6][1] = -6.93889390390722838E-018;
      DD[6][2] = -0.26666666666666666;
      DD[6][3] =  0.19999999999999998;
      DD[6][4] = -0.33333333333333331;
      DD[6][5] =  6.93889390390722838E-018;
      DD[6][6] =  0.53333333333333321;
      DD[6][7] = -0.20000000000000001;
      DD[7][0] =  6.93889390390722838E-018;
      DD[7][1] = -0.33333333333333331;
      DD[7][2] =  0.19999999999999998;
      DD[7][3] = -0.26666666666666666;
      DD[7][4] =   0.0000000000000000;
      DD[7][5] =  6.66666666666666519E-002;
      DD[7][6] = -0.20000000000000001;
      DD[7][7] =  0.53333333333333321;

    /* BC version of element */
    for (i=0; i<8; i++) {
      for (j=0; j<8; j++) {
        if (i<4 || j < 4) {
          if (i==j) DD2[i][j] = .1*DD1[i][j];
          else DD2[i][j] = 0.0;
        } else DD2[i][j] = DD1[i][j];
      }
    }
  }
  {
    PetscReal *coords;
    PetscCall(PetscMalloc1(m,&coords));
    /* forms the element stiffness and coordinates */
    for (Ii = Istart/2, ix = 0; Ii < Iend/2; Ii++, ix++) {
      j = Ii/(ne+1); i = Ii%(ne+1);
      /* coords */
      x            = h*(Ii % (ne+1)); y = h*(Ii/(ne+1));
      coords[2*ix] = x; coords[2*ix+1] = y;
      if (i<ne && j<ne) {
        PetscInt jj,ii,idx[4];
        /* radius */
        PetscReal radius = PetscSqrtReal((x-.5+h/2)*(x-.5+h/2) + (y-.5+h/2)*(y-.5+h/2));
        PetscReal alpha  = 1.0;
        if (radius < 0.25) alpha = soft_alpha;

        idx[0] = Ii; idx[1] = Ii+1; idx[2] = Ii + (ne+1) + 1;  idx[3] = Ii + (ne+1);
        for (ii=0; ii<8; ii++) {
          for (jj=0;jj<8;jj++) DD[ii][jj] = alpha*DD1[ii][jj];
        }
        if (j>0) {
          PetscCall(MatSetValuesBlocked(Amat,4,idx,4,idx,(const PetscScalar*)DD,ADD_VALUES));
        } else {
          /* a BC */
          for (ii=0; ii<8; ii++) {
            for (jj=0;jj<8;jj++) DD[ii][jj] = alpha*DD2[ii][jj];
          }
          PetscCall(MatSetValuesBlocked(Amat,4,idx,4,idx,(const PetscScalar*)DD,ADD_VALUES));
        }
      }
      if (j>0) {
        PetscScalar v  = h*h;
        PetscInt    jj = 2*Ii; /* load in x direction */
        PetscCall(VecSetValues(bb,1,&jj,&v,INSERT_VALUES));
      }
    }
    PetscCall(MatAssemblyBegin(Amat,MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(Amat,MAT_FINAL_ASSEMBLY));
    PetscCall(VecAssemblyBegin(bb));
    PetscCall(VecAssemblyEnd(bb));

    /* Setup solver */
    PetscCall(KSPCreate(PETSC_COMM_WORLD,&ksp));
    PetscCall(KSPSetFromOptions(ksp));

    /* finish KSP/PC setup */
    PetscCall(KSPSetOperators(ksp, Amat, Amat));
    if (use_coords) {
      PC pc;

      PetscCall(KSPGetPC(ksp, &pc));
      PetscCall(PCSetCoordinates(pc, 2, m/2, coords));
    }
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
  PetscCall(PetscLogStageRegister("Setup", &stage[0]));
  PetscCall(PetscLogStageRegister("Solve", &stage[1]));
  PetscCall(PetscLogStagePush(stage[0]));
#endif
  PetscCall(KSPSetUp(ksp));
#if defined(PETSC_USE_LOG)
  PetscCall(PetscLogStagePop());
#endif

  PetscCall(VecSet(xx,.0));

#if defined(PETSC_USE_LOG)
  PetscCall(PetscLogStagePush(stage[1]));
#endif
  PetscCall(KSPSolve(ksp, bb, xx));
#if defined(PETSC_USE_LOG)
  PetscCall(PetscLogStagePop());
#endif

  PetscCall(KSPGetIterationNumber(ksp,&its));

  if (0) {
    PetscReal   norm,norm2;
    PetscViewer viewer;
    Vec         res;

    PetscCall(PetscObjectGetComm((PetscObject)bb,&comm));
    PetscCall(VecNorm(bb, NORM_2, &norm2));

    PetscCall(VecDuplicate(xx, &res));
    PetscCall(MatMult(Amat, xx, res));
    PetscCall(VecAXPY(bb, -1.0, res));
    PetscCall(VecDestroy(&res));
    PetscCall(VecNorm(bb, NORM_2, &norm));
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

  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:
      suffix: 1
      nsize: 4
      args: -ne 29 -alpha 1.e-3 -ksp_type cg -pc_type gamg -pc_gamg_type agg -pc_gamg_agg_nsmooths 1 -use_coordinates -ksp_converged_reason -pc_gamg_esteig_ksp_max_it 5 -ksp_rtol 1.e-3 -ksp_monitor_short -mg_levels_ksp_chebyshev_esteig 0,0.05,0,1.2
      output_file: output/ex55_sa.out

   test:
      suffix: Classical
      nsize: 4
      args: -ne 29 -alpha 1.e-3 -ksp_type cg -pc_type gamg -pc_gamg_type classical -mg_levels_ksp_max_it 5 -ksp_converged_reason
      output_file: output/ex55_classical.out

   test:
      suffix: NC
      nsize: 4
      args: -ne 29 -alpha 1.e-3 -ksp_type cg -pc_type gamg -pc_gamg_type agg -pc_gamg_agg_nsmooths 1 -ksp_converged_reason -pc_gamg_esteig_ksp_max_it 10 -mg_levels_ksp_chebyshev_esteig 0,0.05,0,1.2

   test:
      suffix: geo
      nsize: 4
      args: -ne 29 -alpha 1.e-3 -ksp_type cg -pc_type gamg -pc_gamg_type geo -use_coordinates -ksp_monitor_short -ksp_type cg -ksp_norm_type unpreconditioned  -mg_levels_ksp_max_it 3
      output_file: output/ex55_0.out
      requires: triangle

   test:
      suffix: hypre
      nsize: 4
      requires: hypre !complex !defined(PETSC_HAVE_HYPRE_DEVICE)
      args: -ne 29 -alpha 1.e-3 -ksp_type cg -pc_type hypre -pc_hypre_type boomeramg -ksp_monitor_short

   # command line options match GPU defaults
   test:
      suffix: hypre_device
      nsize: 4
      requires: hypre !complex
      args: -mat_type hypre -ksp_view -ne 29 -alpha 1.e-3 -ksp_type cg -pc_type hypre -pc_hypre_type boomeramg -ksp_monitor_short -pc_hypre_boomeramg_relax_type_all l1scaled-Jacobi -pc_hypre_boomeramg_interp_type ext+i -pc_hypre_boomeramg_coarsen_type PMIS -pc_hypre_boomeramg_no_CF -pc_mg_galerkin_mat_product_algorithm hypre

TEST*/
