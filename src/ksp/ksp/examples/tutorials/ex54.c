
static char help[] = "Creates a matrix using simple quadrilateral finite elements, and uses it to test GAMG\n\
  -ne <size>       : problem size\n                                      \
  -alpha <v>      : scaling of material coeficient in embedded circle\n\n";

#include <petscksp.h>

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **args)
{
  Mat            Amat,Pmat;
  PetscErrorCode ierr;
  PetscInt       i,m,M,its,Istart,Iend,j,Ii,bs,ix,ne=4;
  PetscReal      x,y,h;
  Vec            xx,bb;
  KSP            ksp;
  PetscReal      soft_alpha = 1.e-3;
  MPI_Comm       wcomm;
  PetscMPIInt    npe,mype;
  PC pc;
  PetscScalar DD[4][4],DD2[4][4];
#if defined(PETSC_USE_LOG)
  PetscLogStage  stage;
#endif
#define DIAG_S 0.0
  PetscScalar DD1[4][4] = { {5.0+DIAG_S, -2.0, -1.0, -2.0},
                            {-2.0, 5.0+DIAG_S, -2.0, -1.0},
                            {-1.0, -2.0, 5.0+DIAG_S, -2.0},
                            {-2.0, -1.0, -2.0, 5.0+DIAG_S} };

  PetscInitialize(&argc,&args,(char *)0,help);
  wcomm = PETSC_COMM_WORLD;
  ierr = MPI_Comm_rank( wcomm, &mype );   CHKERRQ(ierr);
  ierr = MPI_Comm_size( wcomm, &npe );    CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-ne",&ne,PETSC_NULL); CHKERRQ(ierr);
  h = 1./ne;
  /* ne*ne; number of global elements */
  ierr = PetscOptionsGetReal(PETSC_NULL,"-alpha",&soft_alpha,PETSC_NULL); CHKERRQ(ierr);
  M = (ne+1)*(ne+1); /* global number of nodes */
  /* create stiffness matrix */
  ierr = MatCreateAIJ(wcomm,PETSC_DECIDE,PETSC_DECIDE,M,M,
                         18,PETSC_NULL,6,PETSC_NULL,&Amat);CHKERRQ(ierr);
  ierr = MatCreateAIJ(wcomm,PETSC_DECIDE,PETSC_DECIDE,M,M,
                         18,PETSC_NULL,6,PETSC_NULL,&Pmat);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(Amat,&Istart,&Iend);CHKERRQ(ierr);
  m = Iend-Istart;
  bs = 1;
  /* Generate vectors */
  ierr = VecCreate(wcomm,&xx);   CHKERRQ(ierr);
  ierr = VecSetSizes(xx,m,M);    CHKERRQ(ierr);
  ierr = VecSetFromOptions(xx);  CHKERRQ(ierr);
  ierr = VecDuplicate(xx,&bb);   CHKERRQ(ierr);
  ierr = VecSet(bb,.0);         CHKERRQ(ierr);
  /* generate element matrices */
  {
    FILE *file;
    char fname[] = "data/elem_2d_therm.txt";
    file = fopen(fname, "r");
    if (file == 0) {
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
    }
    else {
      for (i=0;i<4;i++)
        for (j=0;j<4;j++)
          fscanf(file, "%le", &DD1[i][j]);
    }
    /* BC version of element */
    for (i=0;i<4;i++)
      for (j=0;j<4;j++)
        if (i<2 || j < 2)
          if (i==j) DD2[i][j] = .1*DD1[i][j];
          else DD2[i][j] = 0.0;
        else DD2[i][j] = DD1[i][j];
  }
  {
    PetscReal coords[2*m];
    /* forms the element stiffness for the Laplacian and coordinates */
    for (Ii=Istart,ix=0; Ii<Iend; Ii++,ix++) {
      j = Ii/(ne+1); i = Ii%(ne+1);
      /* coords */
      x = h*(Ii % (ne+1)); y = h*(Ii/(ne+1));
      coords[2*ix] = x; coords[2*ix+1] = y;
      if ( i<ne && j<ne ) {
        PetscInt jj,ii,idx[4] = {Ii, Ii+1, Ii + (ne+1) + 1, Ii + (ne+1)};
        /* radius */
        PetscReal radius = PetscSqrtScalar( (x-.5+h/2)*(x-.5+h/2) + (y-.5+h/2)*(y-.5+h/2) );
        PetscReal alpha = 1.0;
        if ( radius < 0.25 ){
          alpha = soft_alpha;
        }

        for (ii=0;ii<4;ii++)for (jj=0;jj<4;jj++) DD[ii][jj] = alpha*DD1[ii][jj];
        ierr = MatSetValues(Pmat,4,idx,4,idx,(const PetscScalar*)DD,ADD_VALUES);CHKERRQ(ierr);
        if ( j>0 ) {
          ierr = MatSetValues(Amat,4,idx,4,idx,(const PetscScalar*)DD,ADD_VALUES);CHKERRQ(ierr);
        }
        else {
          /* a BC */
          for (ii=0;ii<4;ii++)for (jj=0;jj<4;jj++) DD[ii][jj] = alpha*DD2[ii][jj];
          ierr = MatSetValues(Amat,4,idx,4,idx,(const PetscScalar*)DD,ADD_VALUES);CHKERRQ(ierr);
        }
      }
      if ( j>0 ) {
        PetscScalar v = h*h;
        PetscInt jj = Ii;
        ierr = VecSetValues(bb,1,&jj,&v,INSERT_VALUES);      CHKERRQ(ierr);
      }
    }
    ierr = MatAssemblyBegin(Amat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(Amat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyBegin(Pmat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(Pmat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = VecAssemblyBegin(bb);  CHKERRQ(ierr);
    ierr = VecAssemblyEnd(bb);    CHKERRQ(ierr);

    /* Setup solver */
    ierr = KSPCreate(PETSC_COMM_WORLD,&ksp);                    CHKERRQ(ierr);
    ierr = KSPSetType( ksp, KSPCG );                            CHKERRQ(ierr);
    ierr = KSPGetPC(ksp,&pc);                                   CHKERRQ(ierr);
    ierr = PCSetType(pc,PCGAMG);                                CHKERRQ(ierr);
    ierr = KSPSetFromOptions(ksp);                              CHKERRQ(ierr);

    /* finish KSP/PC setup */
    ierr = KSPSetOperators( ksp, Amat, Amat, SAME_NONZERO_PATTERN ); CHKERRQ(ierr);
    ierr = PCSetCoordinates( pc, 2, m, coords );                   CHKERRQ(ierr);
  }

  if ( !PETSC_TRUE ) {
    PetscViewer viewer;
    ierr = PetscViewerASCIIOpen(wcomm, "Amat.m", &viewer);  CHKERRQ(ierr);
    ierr = PetscViewerSetFormat( viewer, PETSC_VIEWER_ASCII_MATLAB);  CHKERRQ(ierr);
    ierr = MatView(Amat,viewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy( &viewer );
  }

  /* solve */
#if defined(PETSC_USE_LOG)
  ierr = PetscLogStageRegister("Solve", &stage);      CHKERRQ(ierr);
  ierr = PetscLogStagePush(stage);      CHKERRQ(ierr);
#endif
  ierr = VecSet(xx,.0);          CHKERRQ(ierr);

  ierr = KSPSolve(ksp,bb,xx);     CHKERRQ(ierr);

#if defined(PETSC_USE_LOG)
  ierr = PetscLogStagePop();      CHKERRQ(ierr);
#endif

  ierr = KSPGetIterationNumber(ksp,&its);CHKERRQ(ierr);

  if ( !PETSC_TRUE ) {
    PetscReal norm,norm2;
    PetscViewer viewer;
    Vec res;
    ierr = PetscViewerASCIIOpen(wcomm, "rhs.m", &viewer);  CHKERRQ(ierr);
    ierr = PetscViewerSetFormat( viewer, PETSC_VIEWER_ASCII_MATLAB);  CHKERRQ(ierr);
    ierr = VecView(bb,viewer); CHKERRQ(ierr);
    ierr = PetscViewerDestroy( &viewer );
    ierr = VecNorm( bb, NORM_2, &norm2 );  CHKERRQ(ierr);

    ierr = PetscViewerASCIIOpen(wcomm, "solution.m", &viewer);  CHKERRQ(ierr);
    ierr = PetscViewerSetFormat( viewer, PETSC_VIEWER_ASCII_MATLAB);  CHKERRQ(ierr);
    ierr = VecView(xx,viewer); CHKERRQ(ierr);
    ierr = PetscViewerDestroy( &viewer );

    ierr = VecDuplicate( xx, &res );   CHKERRQ(ierr);
    ierr = MatMult( Amat, xx, res );   CHKERRQ(ierr);
    ierr = VecAXPY( bb, -1.0, res );  CHKERRQ(ierr);
    ierr = VecDestroy( &res );CHKERRQ(ierr);
    ierr = VecNorm(bb,NORM_2,&norm);  CHKERRQ(ierr);
    PetscPrintf(PETSC_COMM_WORLD,"[%d]%s |b-Ax|/|b|=%e, |b|=%e\n",0,__FUNCT__,norm/norm2,norm2);

    ierr = PetscViewerASCIIOpen(wcomm, "residual.m", &viewer);  CHKERRQ(ierr);
    ierr = PetscViewerSetFormat( viewer, PETSC_VIEWER_ASCII_MATLAB);  CHKERRQ(ierr);
    ierr = VecView(bb,viewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy( &viewer );
  }

  /* Free work space */
  ierr = KSPDestroy(&ksp);CHKERRQ(ierr);
  ierr = VecDestroy(&xx);CHKERRQ(ierr);
  ierr = VecDestroy(&bb);CHKERRQ(ierr);
  ierr = MatDestroy(&Amat);CHKERRQ(ierr);
  ierr = MatDestroy(&Pmat);CHKERRQ(ierr);

  ierr = PetscFinalize();
  return 0;
}

