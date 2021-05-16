static char help[] = "2D, bi-linear quadrilateral (Q1), displacement finite element formulation\n\
of plain strain linear elasticity.  E=1.0, nu=0.25.\n\
Unit square domain with Dirichelet boundary condition on the y=0 side only.\n\
Load of 1.0 in x direction on all nodes (not a true uniform load).\n\
  -ne <size>      : number of (square) quadrilateral elements in each dimension\n\
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

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  comm = PETSC_COMM_WORLD;
  ierr  = MPI_Comm_rank(comm, &mype);CHKERRMPI(ierr);
  ierr  = MPI_Comm_size(comm, &npe);CHKERRMPI(ierr);
  ierr  = PetscOptionsGetInt(NULL,NULL,"-ne",&ne,NULL);CHKERRQ(ierr);
  h     = 1./ne;
  /* ne*ne; number of global elements */
  ierr = PetscOptionsGetReal(NULL,NULL,"-alpha",&soft_alpha,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL,"-use_coordinates",&use_coords,NULL);CHKERRQ(ierr);
  M    = 2*(ne+1)*(ne+1); /* global number of equations */
  m    = (ne+1)*(ne+1)/npe;
  if (mype==npe-1) m = (ne+1)*(ne+1) - (npe-1)*m;
  m *= 2;
  /* create stiffness matrix */
  ierr = MatCreate(comm,&Amat);CHKERRQ(ierr);
  ierr = MatSetSizes(Amat,m,m,M,M);CHKERRQ(ierr);
  ierr = MatSetBlockSize(Amat,2);CHKERRQ(ierr);
  ierr = MatSetType(Amat,MATAIJ);CHKERRQ(ierr);
  ierr = MatSetOption(Amat,MAT_SPD,PETSC_TRUE);CHKERRQ(ierr);
  ierr = MatSetFromOptions(Amat);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(Amat,18,NULL);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(Amat,18,NULL,18,NULL);CHKERRQ(ierr);

  ierr = MatCreate(comm,&Pmat);CHKERRQ(ierr);
  ierr = MatSetSizes(Pmat,m,m,M,M);CHKERRQ(ierr);
  ierr = MatSetBlockSize(Pmat,2);CHKERRQ(ierr);
  ierr = MatSetType(Pmat,MATAIJ);CHKERRQ(ierr);
  ierr = MatSetFromOptions(Pmat);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(Pmat,18,NULL);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(Pmat,18,NULL,12,NULL);CHKERRQ(ierr);

  ierr = MatGetOwnershipRange(Amat,&Istart,&Iend);CHKERRQ(ierr);
  if (m != Iend - Istart) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_PLIB,"m %D does not equal Iend %D - Istart %D",m,Iend,Istart);
  /* Generate vectors */
  ierr = VecCreate(comm,&xx);CHKERRQ(ierr);
  ierr = VecSetSizes(xx,m,M);CHKERRQ(ierr);
  ierr = VecSetFromOptions(xx);CHKERRQ(ierr);
  ierr = VecDuplicate(xx,&bb);CHKERRQ(ierr);
  ierr = VecSet(bb,.0);CHKERRQ(ierr);
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
    ierr = PetscMalloc1(m,&coords);CHKERRQ(ierr);
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
        ierr = MatSetValuesBlocked(Pmat,4,idx,4,idx,(const PetscScalar*)DD,ADD_VALUES);CHKERRQ(ierr);
        if (j>0) {
          ierr = MatSetValuesBlocked(Amat,4,idx,4,idx,(const PetscScalar*)DD,ADD_VALUES);CHKERRQ(ierr);
        } else {
          /* a BC */
          for (ii=0; ii<8; ii++) {
            for (jj=0;jj<8;jj++) DD[ii][jj] = alpha*DD2[ii][jj];
          }
          ierr = MatSetValuesBlocked(Amat,4,idx,4,idx,(const PetscScalar*)DD,ADD_VALUES);CHKERRQ(ierr);
        }
      }
      if (j>0) {
        PetscScalar v  = h*h;
        PetscInt    jj = 2*Ii; /* load in x direction */
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
    if (use_coords) {
      PC             pc;
      ierr = KSPGetPC(ksp, &pc);CHKERRQ(ierr);
      ierr = PCSetCoordinates(pc, 2, m/2, coords);CHKERRQ(ierr);
    }
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
  ierr = PetscLogStageRegister("Setup", &stage[0]);CHKERRQ(ierr);
  ierr = PetscLogStageRegister("Solve", &stage[1]);CHKERRQ(ierr);
  ierr = PetscLogStagePush(stage[0]);CHKERRQ(ierr);
#endif
  ierr = KSPSetUp(ksp);CHKERRQ(ierr);
#if defined(PETSC_USE_LOG)
  ierr = PetscLogStagePop();CHKERRQ(ierr);
#endif

  ierr = VecSet(xx,.0);CHKERRQ(ierr);

#if defined(PETSC_USE_LOG)
  ierr = PetscLogStagePush(stage[1]);CHKERRQ(ierr);
#endif
  ierr = KSPSolve(ksp, bb, xx);CHKERRQ(ierr);
#if defined(PETSC_USE_LOG)
  ierr = PetscLogStagePop();CHKERRQ(ierr);
#endif

  ierr = KSPGetIterationNumber(ksp,&its);CHKERRQ(ierr);

  if (0) {
    PetscReal   norm,norm2;
    PetscViewer viewer;
    Vec         res;

    ierr = PetscObjectGetComm((PetscObject)bb,&comm);CHKERRQ(ierr);
    ierr = VecNorm(bb, NORM_2, &norm2);CHKERRQ(ierr);

    ierr = VecDuplicate(xx, &res);CHKERRQ(ierr);
    ierr = MatMult(Amat, xx, res);CHKERRQ(ierr);
    ierr = VecAXPY(bb, -1.0, res);CHKERRQ(ierr);
    ierr = VecDestroy(&res);CHKERRQ(ierr);
    ierr = VecNorm(bb, NORM_2, &norm);CHKERRQ(ierr);
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

   test:
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
      requires: hypre !complex
      args: -ne 29 -alpha 1.e-3 -ksp_type cg -pc_type hypre -pc_hypre_type boomeramg -ksp_monitor_short

TEST*/
