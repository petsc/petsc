static char help[] = 
"ex56: 3D, bi-linear quadrilateral (Q1), displacement finite element formulation\n\
of plain strain linear elasticity, that uses the GAMG PC.  E=1.0, nu=0.25.\n\
Unit square domain with Dirichelet boundary condition on the y=0 side only.\n\
Load of 1.0 in x direction on all nodes (not a true uniform load).\n\
  -ne <size>      : number of (square) quadrilateral elements in each dimention\n\
  -alpha <v>      : scaling of material coeficient in embedded circle\n\n";

#include <petscksp.h>

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **args)
{
  Mat            Amat,Pmat;
  PetscErrorCode ierr;
  PetscInt       i,m,nn,M,its,Istart,Iend,j,k,Ii,ix,ne=4;
  PetscReal      x,y,z,h;
  Vec            xx,bb;
  KSP            ksp;
  PetscReal      soft_alpha = 1.e-3;
  MPI_Comm       wcomm;
  PetscMPIInt    npe,mype;
  PC pc;
  PetscScalar DD[24][24],DD2[24][24];
#if defined(PETSC_USE_LOG)
  PetscLogStage  stage[2];
#endif
  PetscScalar DD1[24][24];

  PetscInitialize(&argc,&args,(char *)0,help);
  wcomm = PETSC_COMM_WORLD;
  ierr = MPI_Comm_rank( wcomm, &mype );   CHKERRQ(ierr);
  ierr = MPI_Comm_size( wcomm, &npe );    CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-ne",&ne,PETSC_NULL); CHKERRQ(ierr);
  h = 1./ne; nn = ne+1;
  /* ne*ne; number of global elements */
  ierr = PetscOptionsGetReal(PETSC_NULL,"-alpha",&soft_alpha,PETSC_NULL); CHKERRQ(ierr);
  M = 3*nn*nn*nn; /* global number of equations */
  m = nn*nn*nn/npe;
  if(mype==npe-1) m = nn*nn*nn - (npe-1)*m;
  m *= 3; /* number of equeations local*/
  /* create stiffness matrix */
  ierr = MatCreateMPIAIJ(wcomm,m,m,M,M,81,PETSC_NULL,47,PETSC_NULL,&Amat);CHKERRQ(ierr);
  ierr = MatCreateMPIAIJ(wcomm,m,m,M,M,81,PETSC_NULL,47,PETSC_NULL,&Pmat);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(Amat,&Istart,&Iend);CHKERRQ(ierr);
  ierr = MatSetBlockSize(Amat,3);      CHKERRQ(ierr);
  ierr = MatSetBlockSize(Pmat,3);      CHKERRQ(ierr);
  m = Iend - Istart;
  /* Generate vectors */
  ierr = VecCreate(wcomm,&xx);   CHKERRQ(ierr);
  ierr = VecSetSizes(xx,m,M);    CHKERRQ(ierr);
  ierr = VecSetFromOptions(xx);  CHKERRQ(ierr);
  ierr = VecDuplicate(xx,&bb);   CHKERRQ(ierr);
  ierr = VecSet(bb,.0);         CHKERRQ(ierr);
  /* generate element matrices */
  {
    FILE *file;
    char fname[] = "elem_3d_elast_v_25.txt";
    file = fopen(fname, "r");
    if (file == 0) {
      PetscPrintf(PETSC_COMM_WORLD,"\t%s failed to open input file '%s'\n",__FUNCT__,fname);
    }
    else {
      for(i=0;i<24;i++){
        for(j=0;j<24;j++){
          fscanf(file, "%le", &DD1[i][j]);
        }
      }
    }
    /* BC version of element */
    for(i=0;i<24;i++)
      for(j=0;j<24;j++)
        if(i<8 || j < 8)
          if(i==j) DD2[i][j] = .1*DD1[i][j];
          else DD2[i][j] = 0.0;
        else DD2[i][j] = DD1[i][j];
  }
  {
    PetscReal coords[3*m];
    /* forms the element stiffness for the Laplacian and coordinates */
    for (Ii = Istart/3, ix = 0; Ii < Iend/3; Ii++, ix++ ) {
      i = Ii%nn; j = Ii%(nn*nn)/nn; k = Ii/(nn*nn);
      /* coords */
      x = h*i; y = h*j; z = h*k;
      coords[3*ix] = x; coords[3*ix+1] = y; coords[3*ix+2] = z;
      if( i<ne && j<ne && k<ne) {
        PetscInt jj,ii,idx[8] = {Ii, Ii+1, Ii+nn+1, Ii+nn, 
                                 Ii+nn*nn, Ii+1+nn*nn, 
                                 Ii+nn+1+nn*nn, Ii+nn+nn*nn };
        /* radius */
        PetscReal radius = PetscSqrtScalar((x-.5+h/2)*(x-.5+h/2)+(y-.5+h/2)*(y-.5+h/2)+(z-.5+h/2)*(z-.5+h/2));
        PetscReal alpha = 1.0;
        if( radius < 0.25 ){
          alpha = soft_alpha;
        }
        for(ii=0;ii<24;ii++)for(jj=0;jj<24;jj++) DD[ii][jj] = alpha*DD1[ii][jj];
        ierr = MatSetValuesBlocked(Pmat,8,idx,8,idx,(const PetscScalar*)DD,ADD_VALUES);CHKERRQ(ierr);
        if( k>0 ) {
          ierr = MatSetValuesBlocked(Amat,8,idx,8,idx,(const PetscScalar*)DD,ADD_VALUES);CHKERRQ(ierr);
        }
        else {
          /* a BC */
          for(ii=0;ii<24;ii++)for(jj=0;jj<24;jj++) DD[ii][jj] = alpha*DD2[ii][jj];
          ierr = MatSetValuesBlocked(Amat,8,idx,8,idx,(const PetscScalar*)DD,ADD_VALUES);CHKERRQ(ierr);
        }
      }
      if( k>0 ) {
        PetscScalar v = h*h;
        PetscInt jj = 3*Ii; /* load in x direction */
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
    ierr = KSPSetOperators( ksp, Amat, Amat, SAME_NONZERO_PATTERN ); CHKERRQ(ierr);
    ierr = KSPSetType( ksp, KSPCG );                            CHKERRQ(ierr);
    ierr = KSPGetPC( ksp, &pc );                                   CHKERRQ(ierr);
    ierr = PCSetType( pc, PCGAMG );                                CHKERRQ(ierr);
    ierr = PCSetCoordinates( pc, 3, coords );                   CHKERRQ(ierr);
    ierr = KSPSetFromOptions( ksp );                              CHKERRQ(ierr);
  }

  if( !PETSC_TRUE ) {
    PetscViewer viewer;
    ierr = PetscViewerASCIIOpen(wcomm, "Amat.m", &viewer);  CHKERRQ(ierr);
    ierr = PetscViewerSetFormat( viewer, PETSC_VIEWER_ASCII_MATLAB);  CHKERRQ(ierr);
    ierr = MatView(Amat,viewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy( &viewer );
  }

  /* solve */
#if defined(PETSC_USE_LOG)
  ierr = PetscLogStageRegister("Setup", &stage[0]);      CHKERRQ(ierr);
  ierr = PetscLogStageRegister("Solve", &stage[1]);      CHKERRQ(ierr);
  ierr = PetscLogStagePush(stage[0]);                    CHKERRQ(ierr);
#endif
  ierr = KSPSetUp( ksp );         CHKERRQ(ierr);
#if defined(PETSC_USE_LOG)
  ierr = PetscLogStagePop();      CHKERRQ(ierr);
#endif

  ierr = VecSet(xx,.0);           CHKERRQ(ierr);

#if defined(PETSC_USE_LOG)
  ierr = PetscLogStagePush(stage[1]);                    CHKERRQ(ierr);
#endif
  ierr = KSPSolve( ksp, bb, xx );     CHKERRQ(ierr);
#if defined(PETSC_USE_LOG)
  ierr = PetscLogStagePop();      CHKERRQ(ierr);
#endif

  ierr = KSPGetIterationNumber(ksp,&its);CHKERRQ(ierr);

  if( !PETSC_TRUE ) {
    PetscReal norm,norm2;
    PetscViewer viewer;
    Vec res;

    ierr = VecNorm( bb, NORM_2, &norm2 );  CHKERRQ(ierr);

    ierr = VecDuplicate( xx, &res );   CHKERRQ(ierr);
    ierr = MatMult( Amat, xx, res );   CHKERRQ(ierr);
    ierr = VecAXPY( bb, -1.0, res );   CHKERRQ(ierr);
    ierr = VecDestroy( &res );CHKERRQ(ierr);
    ierr = VecNorm( bb, NORM_2, &norm );  CHKERRQ(ierr);
PetscPrintf(PETSC_COMM_WORLD,"[%d]%s |b-Ax|/|b|=%e, |b|=%e\n",0,__FUNCT__,norm/norm2,norm2);
    ierr = PetscViewerASCIIOpen(wcomm, "residual.m", &viewer);  CHKERRQ(ierr);
    ierr = PetscViewerSetFormat(viewer, PETSC_VIEWER_ASCII_MATLAB);  CHKERRQ(ierr);
    ierr = VecView(bb,viewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy( &viewer );


    /* ierr = PetscViewerASCIIOpen(wcomm, "rhs.m", &viewer);  CHKERRQ(ierr); */
    /* ierr = PetscViewerSetFormat( viewer, PETSC_VIEWER_ASCII_MATLAB ); */
    /* CHKERRQ( ierr ); */
    /* ierr = VecView( bb,viewer );           CHKERRQ(ierr); */
    /* ierr = PetscViewerDestroy( &viewer );  CHKERRQ(ierr); */

    /* ierr = PetscViewerASCIIOpen(wcomm, "solution.m", &viewer);  CHKERRQ(ierr); */
    /* ierr = PetscViewerSetFormat( viewer, PETSC_VIEWER_ASCII_MATLAB ); */
    /* CHKERRQ(ierr); */
    /* ierr = VecView( xx, viewer ); CHKERRQ(ierr); */
    /* ierr = PetscViewerDestroy( &viewer ); CHKERRQ(ierr); */
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

