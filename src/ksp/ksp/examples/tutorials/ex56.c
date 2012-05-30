static char help[] = 
"ex56: 3D, bi-linear quadrilateral (Q1), displacement finite element formulation\n\
of plain strain linear elasticity, that uses the GAMG PC.  E=1.0, nu=0.25.\n\
Unit square domain with Dirichelet boundary condition on the y=0 side only.\n\
Load of 1.0 in x direction on all nodes (not a true uniform load).\n\
  -ne <size>      : number of (square) quadrilateral elements in each dimention\n\
  -alpha <v>      : scaling of material coeficient in embedded circle\n\n";

#include <petscksp.h>
#include <assert.h>

#define ADD_STAGES
 
#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **args)
{
  Mat            Amat;
  PetscErrorCode ierr;
  PetscInt       m,nn,M,Istart,Iend,i,j,k,ii,jj,kk,ic,ne=4,id;
  PetscReal      x,y,z,h,*coords,soft_alpha=1.e-3;
  Vec            xx,bb;
  KSP            ksp;
  MPI_Comm       wcomm;
  PetscMPIInt    npe,mype;
  PC pc;
  PetscScalar DD[24][24],DD2[24][24];
#if defined(PETSC_USE_LOG) && defined(ADD_STAGES)
  PetscLogStage  stage[6];
#endif
  PetscScalar DD1[24][24];
  const PCType type;

  PetscInitialize(&argc,&args,(char *)0,help);
  wcomm = PETSC_COMM_WORLD;
  ierr = MPI_Comm_rank( wcomm, &mype );   CHKERRQ(ierr);
  ierr = MPI_Comm_size( wcomm, &npe );    CHKERRQ(ierr);

  /* log */
#if defined(PETSC_USE_LOG) && defined(ADD_STAGES)
  ierr = PetscLogStageRegister("Setup", &stage[0]);      CHKERRQ(ierr);
  ierr = PetscLogStageRegister("Solve", &stage[1]);      CHKERRQ(ierr);
  ierr = PetscLogStageRegister("2nd Setup", &stage[2]);      CHKERRQ(ierr);
  ierr = PetscLogStageRegister("2nd Solve", &stage[3]);      CHKERRQ(ierr);
  ierr = PetscLogStageRegister("3rd Setup", &stage[4]);      CHKERRQ(ierr);
  ierr = PetscLogStageRegister("3rd Solve", &stage[5]);      CHKERRQ(ierr);
#endif
  
  ierr = PetscOptionsGetInt(PETSC_NULL,"-ne",&ne,PETSC_NULL); CHKERRQ(ierr);
  h = 1./ne; nn = ne+1;
  /* ne*ne; number of global elements */
  ierr = PetscOptionsGetReal(PETSC_NULL,"-alpha",&soft_alpha,PETSC_NULL); CHKERRQ(ierr);
  M = 3*nn*nn*nn; /* global number of equations */
  if(npe==2) {
    if(mype==1) m=0;
    else m = nn*nn*nn;
    npe = 1;
  }
  else {
    m = nn*nn*nn/npe;
    if(mype==npe-1) m = nn*nn*nn - (npe-1)*m;
  }
  m *= 3; /* number of equations local*/
  /* Setup solver, get PC type and pc */
  ierr = KSPCreate(PETSC_COMM_WORLD,&ksp);                    CHKERRQ(ierr);
  ierr = KSPSetType( ksp, KSPCG );                            CHKERRQ(ierr);
  ierr = KSPSetComputeSingularValues( ksp, PETSC_TRUE ); CHKERRQ(ierr);
  ierr = KSPGetPC( ksp, &pc );                                   CHKERRQ(ierr);
  ierr = PCSetType( pc, PCGAMG ); CHKERRQ(ierr); /* default */
  ierr = KSPSetFromOptions( ksp );                              CHKERRQ(ierr);
  ierr = PCGetType( pc, &type );                              CHKERRQ(ierr);

  {
    /* configureation */
    const PetscInt NP = (PetscInt)(pow((double)npe,1./3.) + .5);
    if(npe!=NP*NP*NP)SETERRQ1(wcomm,PETSC_ERR_ARG_WRONG, "npe=%d: npe^{1/3} must be integer",npe);
    if(nn!=NP*(nn/NP))SETERRQ1(wcomm,PETSC_ERR_ARG_WRONG, "-ne %d: (ne+1)%(npe^{1/3}) must equal zero",ne);
    const PetscInt ipx = mype%NP, ipy = (mype%(NP*NP))/NP, ipz = mype/(NP*NP);
    const PetscInt Ni0 = ipx*(nn/NP), Nj0 = ipy*(nn/NP), Nk0 = ipz*(nn/NP);
    const PetscInt Ni1 = Ni0 + (m>0 ? (nn/NP) : 0), Nj1 = Nj0 + (nn/NP), Nk1 = Nk0 + (nn/NP);
    const PetscInt NN = nn/NP, id0 = ipz*nn*nn*NN + ipy*nn*NN*NN + ipx*NN*NN*NN;
    PetscInt *d_nnz, *o_nnz,osz[4]={0,9,15,19},nbc;
    PetscScalar vv[24], v2[24];
    
    /* count nnz */
    ierr = PetscMalloc( (m+1)*sizeof(PetscInt), &d_nnz ); CHKERRQ(ierr);
    ierr = PetscMalloc( (m+1)*sizeof(PetscInt), &o_nnz ); CHKERRQ(ierr);
    for(i=Ni0,ic=0;i<Ni1;i++){
      for(j=Nj0;j<Nj1;j++){
	for(k=Nk0;k<Nk1;k++){
	  nbc = 0;
	  if(i==Ni0 || i==Ni1-1)nbc++;
	  if(j==Nj0 || j==Nj1-1)nbc++;
	  if(k==Nk0 || k==Nk1-1)nbc++;
	  for(jj=0;jj<3;jj++,ic++){
	    d_nnz[ic] = 3*(27-osz[nbc]);
	    o_nnz[ic] = 3*osz[nbc];
	  }
	}
      }
    }
    assert(ic==m);
    
    /* create stiffness matrix */
    if( strcmp(type, PCPROMETHEUS) == 0 ){
      /* prometheus needs BAIJ */
      ierr = MatCreateBAIJ(wcomm,3,m,m,M,M,27,PETSC_NULL,19,PETSC_NULL,&Amat);CHKERRQ(ierr);
    }
    else {
      ierr = MatCreate(wcomm,&Amat);CHKERRQ(ierr);
      ierr = MatSetSizes(Amat,m,m,M,M);CHKERRQ(ierr);
      ierr = MatSetBlockSize(Amat,3);CHKERRQ(ierr);
      ierr = MatSetType(Amat,MATAIJ);CHKERRQ(ierr);
      ierr = MatSeqAIJSetPreallocation(Amat,0,d_nnz);CHKERRQ(ierr);
      ierr = MatMPIAIJSetPreallocation(Amat,0,d_nnz,0,o_nnz);CHKERRQ(ierr);
    }
    ierr = PetscFree( d_nnz );  CHKERRQ(ierr);
    ierr = PetscFree( o_nnz );  CHKERRQ(ierr);

    ierr = MatGetOwnershipRange(Amat,&Istart,&Iend);CHKERRQ(ierr);

    assert(m == Iend-Istart);
    /* Generate vectors */
    ierr = VecCreate(wcomm,&xx);   CHKERRQ(ierr);
    ierr = VecSetSizes(xx,m,M);    CHKERRQ(ierr);
    ierr = VecSetBlockSize(xx,3);      CHKERRQ(ierr);
    ierr = VecSetFromOptions(xx);  CHKERRQ(ierr);
    ierr = VecDuplicate(xx,&bb);   CHKERRQ(ierr);
    ierr = VecSet(bb,.0);         CHKERRQ(ierr);
    /* generate element matrices */
    {
      FILE *file;
      char fname[] = "data/elem_3d_elast_v_25.txt";
      file = fopen(fname, "r");
      if (file == 0) {
	PetscPrintf(PETSC_COMM_WORLD,"\t%s failed to open input file '%s'\n",__FUNCT__,fname);
	for(i=0;i<24;i++){
	  for(j=0;j<24;j++){
	    if(i==j)DD1[i][j] = 1.0;
	    else DD1[i][j] = -.25;
	  }
	}
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
	  if(i<12 || j < 12)
	    if(i==j) DD2[i][j] = 0.1*DD1[i][j];
	    else DD2[i][j] = 0.0;
	  else DD2[i][j] = DD1[i][j];
      /* element residual/load vector */
      for(i=0;i<24;i++){
        if(i%3==0) vv[i] = h*h;
        else if(i%3==1) vv[i] = 2.0*h*h;
        else vv[i] = .0;
      }
      for(i=0;i<24;i++){
        if(i%3==0 && i>=12) v2[i] = h*h;
        else if(i%3==1 && i>=12) v2[i] = 2.0*h*h;
        else v2[i] = .0;
      }
    }

    ierr = PetscMalloc( (m+1)*sizeof(PetscReal), &coords ); CHKERRQ(ierr);
    coords[m] = -99.0;

    /* forms the element stiffness for the Laplacian and coordinates */
    for(i=Ni0,ic=0,ii=0;i<Ni1;i++,ii++){
      for(j=Nj0,jj=0;j<Nj1;j++,jj++){
	for(k=Nk0,kk=0;k<Nk1;k++,kk++,ic++){

	  /* coords */
	  x = coords[3*ic] = h*(PetscReal)i; 
	  y = coords[3*ic+1] = h*(PetscReal)j; 
	  z = coords[3*ic+2] = h*(PetscReal)k;
	  /* matrix */
	  id = id0 + ii + NN*jj + NN*NN*kk; 
	  
	  if( i<ne && j<ne && k<ne) {
	    /* radius */
	    PetscReal radius = PetscSqrtScalar((x-.5+h/2)*(x-.5+h/2)+(y-.5+h/2)*(y-.5+h/2)+
					       (z-.5+h/2)*(z-.5+h/2));
	    PetscReal alpha = 1.0;
	    PetscInt jx,ix,idx[8] = { id, id+1, id+NN+1, id+NN, 
				      id        + NN*NN, id+1    + NN*NN, 
				      id+NN+1 + NN*NN, id+NN + NN*NN };

	    /* correct indices */
	    if(i==Ni1-1 && Ni1!=nn){
	      idx[1] += NN*(NN*NN-1);
	      idx[2] += NN*(NN*NN-1);
	      idx[5] += NN*(NN*NN-1);
	      idx[6] += NN*(NN*NN-1);
	    }
	    if(j==Nj1-1 && Nj1!=nn) {
	      idx[2] += NN*NN*(nn-1);
	      idx[3] += NN*NN*(nn-1);
	      idx[6] += NN*NN*(nn-1);
	      idx[7] += NN*NN*(nn-1);
	    }
	    if(k==Nk1-1 && Nk1!=nn) {
	      idx[4] += NN*(nn*nn-NN*NN);
	      idx[5] += NN*(nn*nn-NN*NN);
	      idx[6] += NN*(nn*nn-NN*NN);
	      idx[7] += NN*(nn*nn-NN*NN);
	    }
	    
	    if( radius < 0.25 ){
	      alpha = soft_alpha;
	    }
	    for(ix=0;ix<24;ix++)for(jx=0;jx<24;jx++) DD[ix][jx] = alpha*DD1[ix][jx];
	    if( k>0 ) {
	      ierr = MatSetValuesBlocked(Amat,8,idx,8,idx,(const PetscScalar*)DD,ADD_VALUES);
              CHKERRQ(ierr);
	      ierr = VecSetValuesBlocked(bb,8,idx,(const PetscScalar*)vv,ADD_VALUES); CHKERRQ(ierr);
              
	    }
	    else {
	      /* a BC */
	      for(ix=0;ix<24;ix++)for(jx=0;jx<24;jx++) DD[ix][jx] = alpha*DD2[ix][jx];
	      ierr = MatSetValuesBlocked(Amat,8,idx,8,idx,(const PetscScalar*)DD,ADD_VALUES);
              CHKERRQ(ierr);
              ierr = VecSetValuesBlocked(bb,8,idx,(const PetscScalar*)v2,ADD_VALUES); CHKERRQ(ierr);
	    }
	  }
	}
      }

    }
    ierr = MatAssemblyBegin(Amat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(Amat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = VecAssemblyBegin(bb);  CHKERRQ(ierr);
    ierr = VecAssemblyEnd(bb);    CHKERRQ(ierr);
  }
  
  if( !PETSC_TRUE ) {
    PetscViewer viewer;
    ierr = PetscViewerASCIIOpen(wcomm, "Amat.m", &viewer);  CHKERRQ(ierr);
    ierr = PetscViewerSetFormat( viewer, PETSC_VIEWER_ASCII_MATLAB);  CHKERRQ(ierr);
    ierr = MatView(Amat,viewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy( &viewer );
  }

  /* finish KSP/PC setup */
  ierr = KSPSetOperators( ksp, Amat, Amat, SAME_NONZERO_PATTERN ); CHKERRQ(ierr);
  ierr = PCSetCoordinates( pc, 3, m/3, coords );                   CHKERRQ(ierr);

#if defined(PETSC_USE_LOG) && defined(ADD_STAGES)
  ierr = PetscLogStagePush(stage[0]);                    CHKERRQ(ierr);
#endif

  /* PC setup basically */
  ierr = KSPSetUp( ksp );         CHKERRQ(ierr);

#if defined(PETSC_USE_LOG) && defined(ADD_STAGES)
  ierr = PetscLogStagePop();      CHKERRQ(ierr);
  ierr = PetscLogStagePush(stage[1]);                    CHKERRQ(ierr);
#endif

  /* 1st solve */
  ierr = KSPSolve( ksp, bb, xx );     CHKERRQ(ierr);

#if defined(PETSC_USE_LOG) && defined(ADD_STAGES)
  ierr = PetscLogStagePop();      CHKERRQ(ierr);
#endif

  /* 2nd solve */
/* #define TwoSolve */
#if defined(TwoSolve)
  {
    PetscReal emax, emin;
#if defined(PETSC_USE_LOG) && defined(ADD_STAGES)
    ierr = PetscLogStagePush(stage[2]);                    CHKERRQ(ierr);
#endif
    /* PC setup basically */
    ierr = MatScale( Amat, 100000.0 ); CHKERRQ(ierr);
    ierr = KSPSetOperators( ksp, Amat, Amat, SAME_NONZERO_PATTERN ); CHKERRQ(ierr);
    ierr = KSPSetUp( ksp );         CHKERRQ(ierr);

#if defined(PETSC_USE_LOG) && defined(ADD_STAGES)
    ierr = PetscLogStagePop();      CHKERRQ(ierr);
    ierr = PetscLogStagePush(stage[3]);                    CHKERRQ(ierr);
#endif
    ierr = KSPSolve( ksp, bb, xx );     CHKERRQ(ierr);
    ierr = KSPComputeExtremeSingularValues( ksp, &emax, &emin ); CHKERRQ(ierr);
    
#if defined(PETSC_USE_LOG) && defined(ADD_STAGES)
    ierr = PetscLogStagePop();      CHKERRQ(ierr);
    ierr = PetscLogStagePush(stage[4]);                    CHKERRQ(ierr);
#endif
    
    /* 3rd solve */
    ierr = MatScale( Amat, 100000.0 ); CHKERRQ(ierr);
    ierr = KSPSetOperators( ksp, Amat, Amat, SAME_NONZERO_PATTERN ); CHKERRQ(ierr);
    ierr = KSPSetUp( ksp );         CHKERRQ(ierr);
    
#if defined(PETSC_USE_LOG) && defined(ADD_STAGES)
    ierr = PetscLogStagePop();      CHKERRQ(ierr);
    ierr = PetscLogStagePush(stage[5]);                    CHKERRQ(ierr);
#endif
    
    ierr = KSPSolve( ksp, bb, xx );     CHKERRQ(ierr);
    
#if defined(PETSC_USE_LOG) && defined(ADD_STAGES)
    ierr = PetscLogStagePop();      CHKERRQ(ierr);
#endif
    
    PetscReal norm,norm2;
    /* PetscViewer viewer; */
    Vec res;
    
    ierr = VecNorm( bb, NORM_2, &norm2 );  CHKERRQ(ierr);
    
    ierr = VecDuplicate( xx, &res );   CHKERRQ(ierr);
    ierr = MatMult( Amat, xx, res );   CHKERRQ(ierr);
    ierr = VecAXPY( bb, -1.0, res );   CHKERRQ(ierr);
    ierr = VecDestroy( &res );CHKERRQ(ierr);
    ierr = VecNorm( bb, NORM_2, &norm );  CHKERRQ(ierr);
    PetscPrintf(PETSC_COMM_WORLD,"[%d]%s |b-Ax|/|b|=%e, |b|=%e, emax=%e\n",0,__FUNCT__,norm/norm2,norm2,emax);
    /*ierr = PetscViewerASCIIOpen(wcomm, "residual.m", &viewer);  CHKERRQ(ierr);
     ierr = PetscViewerSetFormat(viewer, PETSC_VIEWER_ASCII_MATLAB);  CHKERRQ(ierr);
     ierr = VecView(bb,viewer);CHKERRQ(ierr);
     ierr = PetscViewerDestroy( &viewer );*/
    
    
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
#endif

  /* Free work space */
#if !defined(foo)
  ierr = KSPDestroy(&ksp);CHKERRQ(ierr);
#endif
  ierr = VecDestroy(&xx);CHKERRQ(ierr);
  ierr = VecDestroy(&bb);CHKERRQ(ierr);
  ierr = MatDestroy(&Amat);CHKERRQ(ierr);
  ierr = PetscFree( coords );  CHKERRQ(ierr);
  
  ierr = PetscFinalize();
  return 0;
}

