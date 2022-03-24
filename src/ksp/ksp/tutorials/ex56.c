static char help[] = "3D, tri-linear quadrilateral (Q1), displacement finite element formulation\n\
of linear elasticity.  E=1.0, nu=0.25.\n\
Unit square domain with Dirichelet boundary condition on the y=0 side only.\n\
Load of 1.0 in x + 2y direction on all nodes (not a true uniform load).\n\
  -ne <size>      : number of (square) quadrilateral elements in each dimension\n\
  -alpha <v>      : scaling of material coeficient in embedded circle\n\n";

#include <petscksp.h>

static PetscBool log_stages = PETSC_TRUE;
static PetscErrorCode MaybeLogStagePush(PetscLogStage stage) { return log_stages ? PetscLogStagePush(stage) : 0; }
static PetscErrorCode MaybeLogStagePop() { return log_stages ? PetscLogStagePop() : 0; }
PetscErrorCode elem_3d_elast_v_25(PetscScalar *);

int main(int argc,char **args)
{
  Mat            Amat;
  PetscErrorCode ierr;
  PetscInt       m,nn,M,Istart,Iend,i,j,k,ii,jj,kk,ic,ne=4,id;
  PetscReal      x,y,z,h,*coords,soft_alpha=1.e-3;
  PetscBool      two_solves=PETSC_FALSE,test_nonzero_cols=PETSC_FALSE,use_nearnullspace=PETSC_FALSE,test_late_bs=PETSC_FALSE;
  Vec            xx,bb;
  KSP            ksp;
  MPI_Comm       comm;
  PetscMPIInt    npe,mype;
  PetscScalar    DD[24][24],DD2[24][24];
  PetscLogStage  stage[6];
  PetscScalar    DD1[24][24];

  CHKERRQ(PetscInitialize(&argc,&args,(char*)0,help));
  comm = PETSC_COMM_WORLD;
  CHKERRMPI(MPI_Comm_rank(comm, &mype));
  CHKERRMPI(MPI_Comm_size(comm, &npe));

  ierr = PetscOptionsBegin(comm,NULL,"3D bilinear Q1 elasticity options","");CHKERRQ(ierr);
  {
    char nestring[256];
    CHKERRQ(PetscSNPrintf(nestring,sizeof nestring,"number of elements in each direction, ne+1 must be a multiple of %D (sizes^{1/3})",(PetscInt)(PetscPowReal((PetscReal)npe,1./3.) + .5)));
    CHKERRQ(PetscOptionsInt("-ne",nestring,"",ne,&ne,NULL));
    CHKERRQ(PetscOptionsBool("-log_stages","Log stages of solve separately","",log_stages,&log_stages,NULL));
    CHKERRQ(PetscOptionsReal("-alpha","material coefficient inside circle","",soft_alpha,&soft_alpha,NULL));
    CHKERRQ(PetscOptionsBool("-two_solves","solve additional variant of the problem","",two_solves,&two_solves,NULL));
    CHKERRQ(PetscOptionsBool("-test_nonzero_cols","nonzero test","",test_nonzero_cols,&test_nonzero_cols,NULL));
    CHKERRQ(PetscOptionsBool("-use_mat_nearnullspace","MatNearNullSpace API test","",use_nearnullspace,&use_nearnullspace,NULL));
    CHKERRQ(PetscOptionsBool("-test_late_bs","","",test_late_bs,&test_late_bs,NULL));
  }
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  if (log_stages) {
    CHKERRQ(PetscLogStageRegister("Setup", &stage[0]));
    CHKERRQ(PetscLogStageRegister("Solve", &stage[1]));
    CHKERRQ(PetscLogStageRegister("2nd Setup", &stage[2]));
    CHKERRQ(PetscLogStageRegister("2nd Solve", &stage[3]));
    CHKERRQ(PetscLogStageRegister("3rd Setup", &stage[4]));
    CHKERRQ(PetscLogStageRegister("3rd Solve", &stage[5]));
  } else {
    for (i=0; i<(PetscInt)(sizeof(stage)/sizeof(stage[0])); i++) stage[i] = -1;
  }

  h = 1./ne; nn = ne+1;
  /* ne*ne; number of global elements */
  M = 3*nn*nn*nn; /* global number of equations */
  if (npe==2) {
    if (mype==1) m=0;
    else m = nn*nn*nn;
    npe = 1;
  } else {
    m = nn*nn*nn/npe;
    if (mype==npe-1) m = nn*nn*nn - (npe-1)*m;
  }
  m *= 3; /* number of equations local*/
  /* Setup solver */
  CHKERRQ(KSPCreate(PETSC_COMM_WORLD,&ksp));
  CHKERRQ(KSPSetComputeSingularValues(ksp, PETSC_TRUE));
  CHKERRQ(KSPSetFromOptions(ksp));
  {
    /* configuration */
    const PetscInt NP = (PetscInt)(PetscPowReal((PetscReal)npe,1./3.) + .5);
    const PetscInt ipx = mype%NP, ipy = (mype%(NP*NP))/NP, ipz = mype/(NP*NP);
    const PetscInt Ni0 = ipx*(nn/NP), Nj0 = ipy*(nn/NP), Nk0 = ipz*(nn/NP);
    const PetscInt Ni1 = Ni0 + (m>0 ? (nn/NP) : 0), Nj1 = Nj0 + (nn/NP), Nk1 = Nk0 + (nn/NP);
    const PetscInt NN  = nn/NP, id0 = ipz*nn*nn*NN + ipy*nn*NN*NN + ipx*NN*NN*NN;
    PetscInt       *d_nnz, *o_nnz,osz[4]={0,9,15,19},nbc;
    PetscScalar    vv[24], v2[24];
    PetscCheckFalse(npe!=NP*NP*NP,comm,PETSC_ERR_ARG_WRONG, "npe=%d: npe^{1/3} must be integer",npe);
    PetscCheckFalse(nn!=NP*(nn/NP),comm,PETSC_ERR_ARG_WRONG, "-ne %d: (ne+1)%(npe^{1/3}) must equal zero",ne);

    /* count nnz */
    CHKERRQ(PetscMalloc1(m+1, &d_nnz));
    CHKERRQ(PetscMalloc1(m+1, &o_nnz));
    for (i=Ni0,ic=0; i<Ni1; i++) {
      for (j=Nj0; j<Nj1; j++) {
        for (k=Nk0; k<Nk1; k++) {
          nbc = 0;
          if (i==Ni0 || i==Ni1-1) nbc++;
          if (j==Nj0 || j==Nj1-1) nbc++;
          if (k==Nk0 || k==Nk1-1) nbc++;
          for (jj=0; jj<3; jj++,ic++) {
            d_nnz[ic] = 3*(27-osz[nbc]);
            o_nnz[ic] = 3*osz[nbc];
          }
        }
      }
    }
    PetscCheckFalse(ic != m,PETSC_COMM_SELF,PETSC_ERR_PLIB,"ic %D does not equal m %D",ic,m);

    /* create stiffness matrix */
    CHKERRQ(MatCreate(comm,&Amat));
    CHKERRQ(MatSetSizes(Amat,m,m,M,M));
    if (!test_late_bs) {
      CHKERRQ(MatSetBlockSize(Amat,3));
    }
    CHKERRQ(MatSetType(Amat,MATAIJ));
    CHKERRQ(MatSetOption(Amat,MAT_SPD,PETSC_TRUE));
    CHKERRQ(MatSetFromOptions(Amat));
    CHKERRQ(MatSeqAIJSetPreallocation(Amat,0,d_nnz));
    CHKERRQ(MatMPIAIJSetPreallocation(Amat,0,d_nnz,0,o_nnz));

    CHKERRQ(PetscFree(d_nnz));
    CHKERRQ(PetscFree(o_nnz));
    CHKERRQ(MatCreateVecs(Amat,&bb,&xx));

    CHKERRQ(MatGetOwnershipRange(Amat,&Istart,&Iend));

    PetscCheckFalse(m != Iend - Istart,PETSC_COMM_SELF,PETSC_ERR_PLIB,"m %D does not equal Iend %D - Istart %D",m,Iend,Istart);
    /* generate element matrices */
    {
      PetscBool hasData = PETSC_TRUE;
      if (!hasData) {
        CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"\t No data is provided\n"));
        for (i=0; i<24; i++) {
          for (j=0; j<24; j++) {
            if (i==j) DD1[i][j] = 1.0;
            else DD1[i][j] = -.25;
          }
        }
      } else {
        /* Get array data */
        CHKERRQ(elem_3d_elast_v_25((PetscScalar*)DD1));
      }

      /* BC version of element */
      for (i=0; i<24; i++) {
        for (j=0; j<24; j++) {
          if (i<12 || (j < 12 && !test_nonzero_cols)) {
            if (i==j) DD2[i][j] = 0.1*DD1[i][j];
            else DD2[i][j] = 0.0;
          } else DD2[i][j] = DD1[i][j];
        }
      }
      /* element residual/load vector */
      for (i=0; i<24; i++) {
        if (i%3==0) vv[i] = h*h;
        else if (i%3==1) vv[i] = 2.0*h*h;
        else vv[i] = .0;
      }
      for (i=0; i<24; i++) {
        if (i%3==0 && i>=12) v2[i] = h*h;
        else if (i%3==1 && i>=12) v2[i] = 2.0*h*h;
        else v2[i] = .0;
      }
    }

    CHKERRQ(PetscMalloc1(m+1, &coords));
    coords[m] = -99.0;

    /* forms the element stiffness and coordinates */
    for (i=Ni0,ic=0,ii=0; i<Ni1; i++,ii++) {
      for (j=Nj0,jj=0; j<Nj1; j++,jj++) {
        for (k=Nk0,kk=0; k<Nk1; k++,kk++,ic++) {
          /* coords */
          x = coords[3*ic] = h*(PetscReal)i;
          y = coords[3*ic+1] = h*(PetscReal)j;
          z = coords[3*ic+2] = h*(PetscReal)k;
          /* matrix */
          id = id0 + ii + NN*jj + NN*NN*kk;
          if (i<ne && j<ne && k<ne) {
            /* radius */
            PetscReal radius = PetscSqrtReal((x-.5+h/2)*(x-.5+h/2)+(y-.5+h/2)*(y-.5+h/2)+(z-.5+h/2)*(z-.5+h/2));
            PetscReal alpha = 1.0;
            PetscInt  jx,ix,idx[8],idx3[24];
            idx[0] = id;
            idx[1] = id+1;
            idx[2] = id+NN+1;
            idx[3] = id+NN;
            idx[4] = id + NN*NN;
            idx[5] = id+1 + NN*NN;
            idx[6] = id+NN+1 + NN*NN;
            idx[7] = id+NN + NN*NN;

            /* correct indices */
            if (i==Ni1-1 && Ni1!=nn) {
              idx[1] += NN*(NN*NN-1);
              idx[2] += NN*(NN*NN-1);
              idx[5] += NN*(NN*NN-1);
              idx[6] += NN*(NN*NN-1);
            }
            if (j==Nj1-1 && Nj1!=nn) {
              idx[2] += NN*NN*(nn-1);
              idx[3] += NN*NN*(nn-1);
              idx[6] += NN*NN*(nn-1);
              idx[7] += NN*NN*(nn-1);
            }
            if (k==Nk1-1 && Nk1!=nn) {
              idx[4] += NN*(nn*nn-NN*NN);
              idx[5] += NN*(nn*nn-NN*NN);
              idx[6] += NN*(nn*nn-NN*NN);
              idx[7] += NN*(nn*nn-NN*NN);
            }

            if (radius < 0.25) alpha = soft_alpha;

            for (ix=0; ix<24; ix++) {
              for (jx=0;jx<24;jx++) DD[ix][jx] = alpha*DD1[ix][jx];
            }
            if (k>0) {
              if (!test_late_bs) {
                CHKERRQ(MatSetValuesBlocked(Amat,8,idx,8,idx,(const PetscScalar*)DD,ADD_VALUES));
                CHKERRQ(VecSetValuesBlocked(bb,8,idx,(const PetscScalar*)vv,ADD_VALUES));
              } else {
                for (ix=0; ix<8; ix++) { idx3[3*ix] = 3*idx[ix]; idx3[3*ix+1] = 3*idx[ix]+1; idx3[3*ix+2] = 3*idx[ix]+2; }
                CHKERRQ(MatSetValues(Amat,24,idx3,24,idx3,(const PetscScalar*)DD,ADD_VALUES));
                CHKERRQ(VecSetValues(bb,24,idx3,(const PetscScalar*)vv,ADD_VALUES));
              }
            } else {
              /* a BC */
              for (ix=0;ix<24;ix++) {
                for (jx=0;jx<24;jx++) DD[ix][jx] = alpha*DD2[ix][jx];
              }
              if (!test_late_bs) {
                CHKERRQ(MatSetValuesBlocked(Amat,8,idx,8,idx,(const PetscScalar*)DD,ADD_VALUES));
                CHKERRQ(VecSetValuesBlocked(bb,8,idx,(const PetscScalar*)v2,ADD_VALUES));
              } else {
                for (ix=0; ix<8; ix++) { idx3[3*ix] = 3*idx[ix]; idx3[3*ix+1] = 3*idx[ix]+1; idx3[3*ix+2] = 3*idx[ix]+2; }
                CHKERRQ(MatSetValues(Amat,24,idx3,24,idx3,(const PetscScalar*)DD,ADD_VALUES));
                CHKERRQ(VecSetValues(bb,24,idx3,(const PetscScalar*)v2,ADD_VALUES));
              }
            }
          }
        }
      }

    }
    CHKERRQ(MatAssemblyBegin(Amat,MAT_FINAL_ASSEMBLY));
    CHKERRQ(MatAssemblyEnd(Amat,MAT_FINAL_ASSEMBLY));
    CHKERRQ(VecAssemblyBegin(bb));
    CHKERRQ(VecAssemblyEnd(bb));
  }
  CHKERRQ(MatAssemblyBegin(Amat,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(Amat,MAT_FINAL_ASSEMBLY));
  CHKERRQ(VecAssemblyBegin(bb));
  CHKERRQ(VecAssemblyEnd(bb));
  if (test_late_bs) {
    CHKERRQ(VecSetBlockSize(xx,3));
    CHKERRQ(VecSetBlockSize(bb,3));
    CHKERRQ(MatSetBlockSize(Amat,3));
  }

  if (!PETSC_TRUE) {
    PetscViewer viewer;
    CHKERRQ(PetscViewerASCIIOpen(comm, "Amat.m", &viewer));
    CHKERRQ(PetscViewerPushFormat(viewer, PETSC_VIEWER_ASCII_MATLAB));
    CHKERRQ(MatView(Amat,viewer));
    CHKERRQ(PetscViewerPopFormat(viewer));
    CHKERRQ(PetscViewerDestroy(&viewer));
  }

  /* finish KSP/PC setup */
  CHKERRQ(KSPSetOperators(ksp, Amat, Amat));
  if (use_nearnullspace) {
    MatNullSpace matnull;
    Vec          vec_coords;
    PetscScalar  *c;

    CHKERRQ(VecCreate(MPI_COMM_WORLD,&vec_coords));
    CHKERRQ(VecSetBlockSize(vec_coords,3));
    CHKERRQ(VecSetSizes(vec_coords,m,PETSC_DECIDE));
    CHKERRQ(VecSetUp(vec_coords));
    CHKERRQ(VecGetArray(vec_coords,&c));
    for (i=0; i<m; i++) c[i] = coords[i]; /* Copy since Scalar type might be Complex */
    CHKERRQ(VecRestoreArray(vec_coords,&c));
    CHKERRQ(MatNullSpaceCreateRigidBody(vec_coords,&matnull));
    CHKERRQ(MatSetNearNullSpace(Amat,matnull));
    CHKERRQ(MatNullSpaceDestroy(&matnull));
    CHKERRQ(VecDestroy(&vec_coords));
  } else {
    PC             pc;
    CHKERRQ(KSPGetPC(ksp,&pc));
    CHKERRQ(PCSetCoordinates(pc, 3, m/3, coords));
  }

  CHKERRQ(MaybeLogStagePush(stage[0]));

  /* PC setup basically */
  CHKERRQ(KSPSetUp(ksp));

  CHKERRQ(MaybeLogStagePop());
  CHKERRQ(MaybeLogStagePush(stage[1]));

  /* test BCs */
  if (test_nonzero_cols) {
    VecZeroEntries(xx);
    if (mype==0) VecSetValue(xx,0,1.0,INSERT_VALUES);
    VecAssemblyBegin(xx);
    VecAssemblyEnd(xx);
    KSPSetInitialGuessNonzero(ksp,PETSC_TRUE);
  }

  /* 1st solve */
  CHKERRQ(KSPSolve(ksp, bb, xx));

  CHKERRQ(MaybeLogStagePop());

  /* 2nd solve */
  if (two_solves) {
    PetscReal emax, emin;
    PetscReal norm,norm2;
    Vec       res;

    CHKERRQ(MaybeLogStagePush(stage[2]));
    /* PC setup basically */
    CHKERRQ(MatScale(Amat, 100000.0));
    CHKERRQ(KSPSetOperators(ksp, Amat, Amat));
    CHKERRQ(KSPSetUp(ksp));

    CHKERRQ(MaybeLogStagePop());
    CHKERRQ(MaybeLogStagePush(stage[3]));
    CHKERRQ(KSPSolve(ksp, bb, xx));
    CHKERRQ(KSPComputeExtremeSingularValues(ksp, &emax, &emin));

    CHKERRQ(MaybeLogStagePop());
    CHKERRQ(MaybeLogStagePush(stage[4]));

    CHKERRQ(MaybeLogStagePop());
    CHKERRQ(MaybeLogStagePush(stage[5]));

    /* 3rd solve */
    CHKERRQ(KSPSolve(ksp, bb, xx));

    CHKERRQ(MaybeLogStagePop());

    CHKERRQ(VecNorm(bb, NORM_2, &norm2));

    CHKERRQ(VecDuplicate(xx, &res));
    CHKERRQ(MatMult(Amat, xx, res));
    CHKERRQ(VecAXPY(bb, -1.0, res));
    CHKERRQ(VecDestroy(&res));
    CHKERRQ(VecNorm(bb, NORM_2, &norm));
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"[%d]%s |b-Ax|/|b|=%e, |b|=%e, emax=%e\n",0,PETSC_FUNCTION_NAME,(double)(norm/norm2),(double)norm2,(double)emax));
  }

  /* Free work space */
  CHKERRQ(KSPDestroy(&ksp));
  CHKERRQ(VecDestroy(&xx));
  CHKERRQ(VecDestroy(&bb));
  CHKERRQ(MatDestroy(&Amat));
  CHKERRQ(PetscFree(coords));

  CHKERRQ(PetscFinalize());
  return 0;
}

/* Data was previously provided in the file data/elem_3d_elast_v_25.tx */
PetscErrorCode elem_3d_elast_v_25(PetscScalar *dd)
{
  PetscScalar    DD[] = {
  0.18981481481481474     ,
  5.27777777777777568E-002,
  5.27777777777777568E-002,
 -5.64814814814814659E-002,
 -1.38888888888889072E-002,
 -1.38888888888889089E-002,
 -8.24074074074073876E-002,
 -5.27777777777777429E-002,
  1.38888888888888725E-002,
  4.90740740740740339E-002,
  1.38888888888889124E-002,
  4.72222222222222071E-002,
  4.90740740740740339E-002,
  4.72222222222221932E-002,
  1.38888888888888968E-002,
 -8.24074074074073876E-002,
  1.38888888888888673E-002,
 -5.27777777777777429E-002,
 -7.87037037037036785E-002,
 -4.72222222222221932E-002,
 -4.72222222222222071E-002,
  1.20370370370370180E-002,
 -1.38888888888888742E-002,
 -1.38888888888888829E-002,
  5.27777777777777568E-002,
  0.18981481481481474     ,
  5.27777777777777568E-002,
  1.38888888888889124E-002,
  4.90740740740740269E-002,
  4.72222222222221932E-002,
 -5.27777777777777637E-002,
 -8.24074074074073876E-002,
  1.38888888888888725E-002,
 -1.38888888888889037E-002,
 -5.64814814814814728E-002,
 -1.38888888888888985E-002,
  4.72222222222221932E-002,
  4.90740740740740478E-002,
  1.38888888888888968E-002,
 -1.38888888888888673E-002,
  1.20370370370370058E-002,
 -1.38888888888888742E-002,
 -4.72222222222221932E-002,
 -7.87037037037036785E-002,
 -4.72222222222222002E-002,
  1.38888888888888742E-002,
 -8.24074074074073598E-002,
 -5.27777777777777568E-002,
  5.27777777777777568E-002,
  5.27777777777777568E-002,
  0.18981481481481474     ,
  1.38888888888889055E-002,
  4.72222222222222002E-002,
  4.90740740740740269E-002,
 -1.38888888888888829E-002,
 -1.38888888888888829E-002,
  1.20370370370370180E-002,
  4.72222222222222002E-002,
  1.38888888888888985E-002,
  4.90740740740740339E-002,
 -1.38888888888888985E-002,
 -1.38888888888888968E-002,
 -5.64814814814814520E-002,
 -5.27777777777777568E-002,
  1.38888888888888777E-002,
 -8.24074074074073876E-002,
 -4.72222222222222002E-002,
 -4.72222222222221932E-002,
 -7.87037037037036646E-002,
  1.38888888888888794E-002,
 -5.27777777777777568E-002,
 -8.24074074074073598E-002,
 -5.64814814814814659E-002,
  1.38888888888889124E-002,
  1.38888888888889055E-002,
  0.18981481481481474     ,
 -5.27777777777777568E-002,
 -5.27777777777777499E-002,
  4.90740740740740269E-002,
 -1.38888888888889072E-002,
 -4.72222222222221932E-002,
 -8.24074074074073876E-002,
  5.27777777777777568E-002,
 -1.38888888888888812E-002,
 -8.24074074074073876E-002,
 -1.38888888888888742E-002,
  5.27777777777777499E-002,
  4.90740740740740269E-002,
 -4.72222222222221863E-002,
 -1.38888888888889089E-002,
  1.20370370370370162E-002,
  1.38888888888888673E-002,
  1.38888888888888742E-002,
 -7.87037037037036785E-002,
  4.72222222222222002E-002,
  4.72222222222222071E-002,
 -1.38888888888889072E-002,
  4.90740740740740269E-002,
  4.72222222222222002E-002,
 -5.27777777777777568E-002,
  0.18981481481481480     ,
  5.27777777777777568E-002,
  1.38888888888889020E-002,
 -5.64814814814814728E-002,
 -1.38888888888888951E-002,
  5.27777777777777637E-002,
 -8.24074074074073876E-002,
  1.38888888888888881E-002,
  1.38888888888888742E-002,
  1.20370370370370232E-002,
 -1.38888888888888812E-002,
 -4.72222222222221863E-002,
  4.90740740740740339E-002,
  1.38888888888888933E-002,
 -1.38888888888888812E-002,
 -8.24074074074073876E-002,
 -5.27777777777777568E-002,
  4.72222222222222071E-002,
 -7.87037037037036924E-002,
 -4.72222222222222140E-002,
 -1.38888888888889089E-002,
  4.72222222222221932E-002,
  4.90740740740740269E-002,
 -5.27777777777777499E-002,
  5.27777777777777568E-002,
  0.18981481481481477     ,
 -4.72222222222222071E-002,
  1.38888888888888968E-002,
  4.90740740740740131E-002,
  1.38888888888888812E-002,
 -1.38888888888888708E-002,
  1.20370370370370267E-002,
  5.27777777777777568E-002,
  1.38888888888888812E-002,
 -8.24074074074073876E-002,
  1.38888888888889124E-002,
 -1.38888888888889055E-002,
 -5.64814814814814589E-002,
 -1.38888888888888812E-002,
 -5.27777777777777568E-002,
 -8.24074074074073737E-002,
  4.72222222222222002E-002,
 -4.72222222222222002E-002,
 -7.87037037037036924E-002,
 -8.24074074074073876E-002,
 -5.27777777777777637E-002,
 -1.38888888888888829E-002,
  4.90740740740740269E-002,
  1.38888888888889020E-002,
 -4.72222222222222071E-002,
  0.18981481481481480     ,
  5.27777777777777637E-002,
 -5.27777777777777637E-002,
 -5.64814814814814728E-002,
 -1.38888888888889037E-002,
  1.38888888888888951E-002,
 -7.87037037037036785E-002,
 -4.72222222222222002E-002,
  4.72222222222221932E-002,
  1.20370370370370128E-002,
 -1.38888888888888725E-002,
  1.38888888888888812E-002,
  4.90740740740740408E-002,
  4.72222222222222002E-002,
 -1.38888888888888951E-002,
 -8.24074074074073876E-002,
  1.38888888888888812E-002,
  5.27777777777777637E-002,
 -5.27777777777777429E-002,
 -8.24074074074073876E-002,
 -1.38888888888888829E-002,
 -1.38888888888889072E-002,
 -5.64814814814814728E-002,
  1.38888888888888968E-002,
  5.27777777777777637E-002,
  0.18981481481481480     ,
 -5.27777777777777568E-002,
  1.38888888888888916E-002,
  4.90740740740740339E-002,
 -4.72222222222222210E-002,
 -4.72222222222221932E-002,
 -7.87037037037036924E-002,
  4.72222222222222002E-002,
  1.38888888888888742E-002,
 -8.24074074074073876E-002,
  5.27777777777777429E-002,
  4.72222222222222002E-002,
  4.90740740740740269E-002,
 -1.38888888888888951E-002,
 -1.38888888888888846E-002,
  1.20370370370370267E-002,
  1.38888888888888916E-002,
  1.38888888888888725E-002,
  1.38888888888888725E-002,
  1.20370370370370180E-002,
 -4.72222222222221932E-002,
 -1.38888888888888951E-002,
  4.90740740740740131E-002,
 -5.27777777777777637E-002,
 -5.27777777777777568E-002,
  0.18981481481481480     ,
 -1.38888888888888968E-002,
 -4.72222222222221932E-002,
  4.90740740740740339E-002,
  4.72222222222221932E-002,
  4.72222222222222071E-002,
 -7.87037037037036646E-002,
 -1.38888888888888742E-002,
  5.27777777777777499E-002,
 -8.24074074074073737E-002,
  1.38888888888888933E-002,
  1.38888888888889020E-002,
 -5.64814814814814589E-002,
  5.27777777777777568E-002,
 -1.38888888888888794E-002,
 -8.24074074074073876E-002,
  4.90740740740740339E-002,
 -1.38888888888889037E-002,
  4.72222222222222002E-002,
 -8.24074074074073876E-002,
  5.27777777777777637E-002,
  1.38888888888888812E-002,
 -5.64814814814814728E-002,
  1.38888888888888916E-002,
 -1.38888888888888968E-002,
  0.18981481481481480     ,
 -5.27777777777777499E-002,
  5.27777777777777707E-002,
  1.20370370370370180E-002,
  1.38888888888888812E-002,
 -1.38888888888888812E-002,
 -7.87037037037036785E-002,
  4.72222222222222002E-002,
 -4.72222222222222071E-002,
 -8.24074074074073876E-002,
 -1.38888888888888742E-002,
 -5.27777777777777568E-002,
  4.90740740740740616E-002,
 -4.72222222222222002E-002,
  1.38888888888888846E-002,
  1.38888888888889124E-002,
 -5.64814814814814728E-002,
  1.38888888888888985E-002,
  5.27777777777777568E-002,
 -8.24074074074073876E-002,
 -1.38888888888888708E-002,
 -1.38888888888889037E-002,
  4.90740740740740339E-002,
 -4.72222222222221932E-002,
 -5.27777777777777499E-002,
  0.18981481481481480     ,
 -5.27777777777777568E-002,
 -1.38888888888888673E-002,
 -8.24074074074073598E-002,
  5.27777777777777429E-002,
  4.72222222222222002E-002,
 -7.87037037037036785E-002,
  4.72222222222222002E-002,
  1.38888888888888708E-002,
  1.20370370370370128E-002,
  1.38888888888888760E-002,
 -4.72222222222222002E-002,
  4.90740740740740478E-002,
 -1.38888888888888951E-002,
  4.72222222222222071E-002,
 -1.38888888888888985E-002,
  4.90740740740740339E-002,
 -1.38888888888888812E-002,
  1.38888888888888881E-002,
  1.20370370370370267E-002,
  1.38888888888888951E-002,
 -4.72222222222222210E-002,
  4.90740740740740339E-002,
  5.27777777777777707E-002,
 -5.27777777777777568E-002,
  0.18981481481481477     ,
  1.38888888888888829E-002,
  5.27777777777777707E-002,
 -8.24074074074073598E-002,
 -4.72222222222222140E-002,
  4.72222222222222140E-002,
 -7.87037037037036646E-002,
 -5.27777777777777707E-002,
 -1.38888888888888829E-002,
 -8.24074074074073876E-002,
 -1.38888888888888881E-002,
  1.38888888888888881E-002,
 -5.64814814814814589E-002,
  4.90740740740740339E-002,
  4.72222222222221932E-002,
 -1.38888888888888985E-002,
 -8.24074074074073876E-002,
  1.38888888888888742E-002,
  5.27777777777777568E-002,
 -7.87037037037036785E-002,
 -4.72222222222221932E-002,
  4.72222222222221932E-002,
  1.20370370370370180E-002,
 -1.38888888888888673E-002,
  1.38888888888888829E-002,
  0.18981481481481469     ,
  5.27777777777777429E-002,
 -5.27777777777777429E-002,
 -5.64814814814814659E-002,
 -1.38888888888889055E-002,
  1.38888888888889055E-002,
 -8.24074074074074153E-002,
 -5.27777777777777429E-002,
 -1.38888888888888760E-002,
  4.90740740740740408E-002,
  1.38888888888888968E-002,
 -4.72222222222222071E-002,
  4.72222222222221932E-002,
  4.90740740740740478E-002,
 -1.38888888888888968E-002,
 -1.38888888888888742E-002,
  1.20370370370370232E-002,
  1.38888888888888812E-002,
 -4.72222222222222002E-002,
 -7.87037037037036924E-002,
  4.72222222222222071E-002,
  1.38888888888888812E-002,
 -8.24074074074073598E-002,
  5.27777777777777707E-002,
  5.27777777777777429E-002,
  0.18981481481481477     ,
 -5.27777777777777499E-002,
  1.38888888888889107E-002,
  4.90740740740740478E-002,
 -4.72222222222221932E-002,
 -5.27777777777777568E-002,
 -8.24074074074074153E-002,
 -1.38888888888888812E-002,
 -1.38888888888888846E-002,
 -5.64814814814814659E-002,
  1.38888888888888812E-002,
  1.38888888888888968E-002,
  1.38888888888888968E-002,
 -5.64814814814814520E-002,
  5.27777777777777499E-002,
 -1.38888888888888812E-002,
 -8.24074074074073876E-002,
  4.72222222222221932E-002,
  4.72222222222222002E-002,
 -7.87037037037036646E-002,
 -1.38888888888888812E-002,
  5.27777777777777429E-002,
 -8.24074074074073598E-002,
 -5.27777777777777429E-002,
 -5.27777777777777499E-002,
  0.18981481481481474     ,
 -1.38888888888888985E-002,
 -4.72222222222221863E-002,
  4.90740740740740339E-002,
  1.38888888888888829E-002,
  1.38888888888888777E-002,
  1.20370370370370249E-002,
 -4.72222222222222002E-002,
 -1.38888888888888933E-002,
  4.90740740740740339E-002,
 -8.24074074074073876E-002,
 -1.38888888888888673E-002,
 -5.27777777777777568E-002,
  4.90740740740740269E-002,
 -4.72222222222221863E-002,
  1.38888888888889124E-002,
  1.20370370370370128E-002,
  1.38888888888888742E-002,
 -1.38888888888888742E-002,
 -7.87037037037036785E-002,
  4.72222222222222002E-002,
 -4.72222222222222140E-002,
 -5.64814814814814659E-002,
  1.38888888888889107E-002,
 -1.38888888888888985E-002,
  0.18981481481481474     ,
 -5.27777777777777499E-002,
  5.27777777777777499E-002,
  4.90740740740740339E-002,
 -1.38888888888889055E-002,
  4.72222222222221932E-002,
 -8.24074074074074153E-002,
  5.27777777777777499E-002,
  1.38888888888888829E-002,
  1.38888888888888673E-002,
  1.20370370370370058E-002,
  1.38888888888888777E-002,
 -4.72222222222221863E-002,
  4.90740740740740339E-002,
 -1.38888888888889055E-002,
 -1.38888888888888725E-002,
 -8.24074074074073876E-002,
  5.27777777777777499E-002,
  4.72222222222222002E-002,
 -7.87037037037036785E-002,
  4.72222222222222140E-002,
 -1.38888888888889055E-002,
  4.90740740740740478E-002,
 -4.72222222222221863E-002,
 -5.27777777777777499E-002,
  0.18981481481481469     ,
 -5.27777777777777499E-002,
  1.38888888888889072E-002,
 -5.64814814814814659E-002,
  1.38888888888889003E-002,
  5.27777777777777429E-002,
 -8.24074074074074153E-002,
 -1.38888888888888812E-002,
 -5.27777777777777429E-002,
 -1.38888888888888742E-002,
 -8.24074074074073876E-002,
 -1.38888888888889089E-002,
  1.38888888888888933E-002,
 -5.64814814814814589E-002,
  1.38888888888888812E-002,
  5.27777777777777429E-002,
 -8.24074074074073737E-002,
 -4.72222222222222071E-002,
  4.72222222222222002E-002,
 -7.87037037037036646E-002,
  1.38888888888889055E-002,
 -4.72222222222221932E-002,
  4.90740740740740339E-002,
  5.27777777777777499E-002,
 -5.27777777777777499E-002,
  0.18981481481481474     ,
  4.72222222222222002E-002,
 -1.38888888888888985E-002,
  4.90740740740740339E-002,
 -1.38888888888888846E-002,
  1.38888888888888812E-002,
  1.20370370370370284E-002,
 -7.87037037037036785E-002,
 -4.72222222222221932E-002,
 -4.72222222222222002E-002,
  1.20370370370370162E-002,
 -1.38888888888888812E-002,
 -1.38888888888888812E-002,
  4.90740740740740408E-002,
  4.72222222222222002E-002,
  1.38888888888888933E-002,
 -8.24074074074073876E-002,
  1.38888888888888708E-002,
 -5.27777777777777707E-002,
 -8.24074074074074153E-002,
 -5.27777777777777568E-002,
  1.38888888888888829E-002,
  4.90740740740740339E-002,
  1.38888888888889072E-002,
  4.72222222222222002E-002,
  0.18981481481481477     ,
  5.27777777777777429E-002,
  5.27777777777777568E-002,
 -5.64814814814814659E-002,
 -1.38888888888888846E-002,
 -1.38888888888888881E-002,
 -4.72222222222221932E-002,
 -7.87037037037036785E-002,
 -4.72222222222221932E-002,
  1.38888888888888673E-002,
 -8.24074074074073876E-002,
 -5.27777777777777568E-002,
  4.72222222222222002E-002,
  4.90740740740740269E-002,
  1.38888888888889020E-002,
 -1.38888888888888742E-002,
  1.20370370370370128E-002,
 -1.38888888888888829E-002,
 -5.27777777777777429E-002,
 -8.24074074074074153E-002,
  1.38888888888888777E-002,
 -1.38888888888889055E-002,
 -5.64814814814814659E-002,
 -1.38888888888888985E-002,
  5.27777777777777429E-002,
  0.18981481481481469     ,
  5.27777777777777429E-002,
  1.38888888888888933E-002,
  4.90740740740740339E-002,
  4.72222222222222071E-002,
 -4.72222222222222071E-002,
 -4.72222222222222002E-002,
 -7.87037037037036646E-002,
  1.38888888888888742E-002,
 -5.27777777777777568E-002,
 -8.24074074074073737E-002,
 -1.38888888888888951E-002,
 -1.38888888888888951E-002,
 -5.64814814814814589E-002,
 -5.27777777777777568E-002,
  1.38888888888888760E-002,
 -8.24074074074073876E-002,
 -1.38888888888888760E-002,
 -1.38888888888888812E-002,
  1.20370370370370249E-002,
  4.72222222222221932E-002,
  1.38888888888889003E-002,
  4.90740740740740339E-002,
  5.27777777777777568E-002,
  5.27777777777777429E-002,
  0.18981481481481474     ,
  1.38888888888888933E-002,
  4.72222222222222071E-002,
  4.90740740740740339E-002,
  1.20370370370370180E-002,
  1.38888888888888742E-002,
  1.38888888888888794E-002,
 -7.87037037037036785E-002,
  4.72222222222222071E-002,
  4.72222222222222002E-002,
 -8.24074074074073876E-002,
 -1.38888888888888846E-002,
  5.27777777777777568E-002,
  4.90740740740740616E-002,
 -4.72222222222222002E-002,
 -1.38888888888888881E-002,
  4.90740740740740408E-002,
 -1.38888888888888846E-002,
 -4.72222222222222002E-002,
 -8.24074074074074153E-002,
  5.27777777777777429E-002,
 -1.38888888888888846E-002,
 -5.64814814814814659E-002,
  1.38888888888888933E-002,
  1.38888888888888933E-002,
  0.18981481481481477     ,
 -5.27777777777777568E-002,
 -5.27777777777777637E-002,
 -1.38888888888888742E-002,
 -8.24074074074073598E-002,
 -5.27777777777777568E-002,
  4.72222222222222002E-002,
 -7.87037037037036924E-002,
 -4.72222222222222002E-002,
  1.38888888888888812E-002,
  1.20370370370370267E-002,
 -1.38888888888888794E-002,
 -4.72222222222222002E-002,
  4.90740740740740478E-002,
  1.38888888888888881E-002,
  1.38888888888888968E-002,
 -5.64814814814814659E-002,
 -1.38888888888888933E-002,
  5.27777777777777499E-002,
 -8.24074074074074153E-002,
  1.38888888888888812E-002,
 -1.38888888888888846E-002,
  4.90740740740740339E-002,
  4.72222222222222071E-002,
 -5.27777777777777568E-002,
  0.18981481481481477     ,
  5.27777777777777637E-002,
 -1.38888888888888829E-002,
 -5.27777777777777568E-002,
 -8.24074074074073598E-002,
  4.72222222222222071E-002,
 -4.72222222222222140E-002,
 -7.87037037037036924E-002,
  5.27777777777777637E-002,
  1.38888888888888916E-002,
 -8.24074074074073876E-002,
  1.38888888888888846E-002,
 -1.38888888888888951E-002,
 -5.64814814814814589E-002,
 -4.72222222222222071E-002,
  1.38888888888888812E-002,
  4.90740740740740339E-002,
  1.38888888888888829E-002,
 -1.38888888888888812E-002,
  1.20370370370370284E-002,
 -1.38888888888888881E-002,
  4.72222222222222071E-002,
  4.90740740740740339E-002,
 -5.27777777777777637E-002,
  5.27777777777777637E-002,
  0.18981481481481477     ,
  };

  PetscFunctionBeginUser;
  CHKERRQ(PetscArraycpy(dd,DD,576));
  PetscFunctionReturn(0);
}

/*TEST

   test:
      nsize: 8
      args: -ne 13 -alpha 1.e-3 -ksp_type cg -pc_type gamg -pc_gamg_agg_nsmooths 1 -pc_gamg_reuse_interpolation true -two_solves -ksp_converged_reason -use_mat_nearnullspace -ksp_view -pc_gamg_square_graph 1 -mg_levels_ksp_max_it 1 -mg_levels_ksp_type chebyshev -mg_levels_ksp_chebyshev_esteig 0,0.2,0,1.05 -pc_gamg_esteig_ksp_max_it 10 -pc_gamg_asm_use_agg true -mg_levels_sub_pc_type lu -mg_levels_pc_asm_overlap 0 -pc_gamg_threshold -0.01 -pc_gamg_coarse_eq_limit 200 -pc_gamg_process_eq_limit 30 -pc_gamg_repartition false -pc_mg_cycle_type v -pc_gamg_use_parallel_coarse_grid_solver -mg_coarse_pc_type jacobi -mg_coarse_ksp_type cg -ksp_monitor_short -pc_gamg_rank_reduction_factors 2,2
      filter: grep -v variant

   test:
      suffix: 2
      nsize: 8
      args: -ne 31 -alpha 1.e-3 -ksp_type cg -pc_type gamg -pc_gamg_agg_nsmooths 1 -pc_gamg_reuse_interpolation true -two_solves -ksp_converged_reason -use_mat_nearnullspace -pc_gamg_square_graph 1 -mg_levels_ksp_max_it 1 -mg_levels_ksp_type chebyshev -mg_levels_ksp_chebyshev_esteig 0,0.2,0,1.05 -pc_gamg_esteig_ksp_max_it 10 -pc_gamg_asm_use_agg true -mg_levels_sub_pc_type lu -mg_levels_pc_asm_overlap 0 -pc_gamg_threshold -0.01 -pc_gamg_coarse_eq_limit 200 -pc_gamg_process_eq_limit 30 -pc_gamg_repartition false -pc_mg_cycle_type v -pc_gamg_use_parallel_coarse_grid_solver -mg_coarse_pc_type jacobi -mg_coarse_ksp_type cg
      filter: grep -v variant

   test:
      suffix: latebs
      filter: grep -v variant
      nsize: 8
      args: -test_late_bs 0 -ne 9 -alpha 1.e-3 -ksp_type cg -pc_type gamg -pc_gamg_agg_nsmooths 1 -pc_gamg_reuse_interpolation true -two_solves -ksp_converged_reason -ksp_view -use_mat_nearnullspace -pc_gamg_square_graph 1 -mg_levels_ksp_max_it 1 -mg_levels_ksp_type chebyshev -mg_levels_ksp_chebyshev_esteig 0,0.2,0,1.05 -pc_gamg_esteig_ksp_max_it 10 -pc_gamg_asm_use_agg true -mg_levels_sub_pc_type lu -mg_levels_pc_asm_overlap 0 -pc_gamg_threshold -0.01 -pc_gamg_coarse_eq_limit 200 -pc_gamg_process_eq_limit 30 -pc_gamg_repartition false -pc_mg_cycle_type v -pc_gamg_use_parallel_coarse_grid_solver -mg_coarse_pc_type jacobi -mg_coarse_ksp_type cg -ksp_monitor_short -ksp_view

   test:
      suffix: latebs-2
      filter: grep -v variant
      nsize: 8
      args: -test_late_bs -ne 9 -alpha 1.e-3 -ksp_type cg -pc_type gamg -pc_gamg_agg_nsmooths 1 -pc_gamg_reuse_interpolation true -two_solves -ksp_converged_reason -ksp_view -use_mat_nearnullspace -pc_gamg_square_graph 1 -mg_levels_ksp_max_it 1 -mg_levels_ksp_type chebyshev -mg_levels_ksp_chebyshev_esteig 0,0.2,0,1.05 -pc_gamg_esteig_ksp_max_it 10 -pc_gamg_asm_use_agg true -mg_levels_sub_pc_type lu -mg_levels_pc_asm_overlap 0 -pc_gamg_threshold -0.01 -pc_gamg_coarse_eq_limit 200 -pc_gamg_process_eq_limit 30 -pc_gamg_repartition false -pc_mg_cycle_type v -pc_gamg_use_parallel_coarse_grid_solver -mg_coarse_pc_type jacobi -mg_coarse_ksp_type cg -ksp_monitor_short -ksp_view

   test:
      suffix: ml
      nsize: 8
      args: -ne 9 -alpha 1.e-3 -ksp_type cg -pc_type ml -mg_levels_ksp_type chebyshev -mg_levels_ksp_chebyshev_esteig 0,0.05,0,1.05 -mg_levels_pc_type sor -ksp_monitor_short
      requires: ml

   test:
      suffix: nns
      args: -ne 9 -alpha 1.e-3 -ksp_converged_reason -ksp_type cg -ksp_max_it 50 -pc_type gamg -pc_gamg_esteig_ksp_type cg -pc_gamg_esteig_ksp_max_it 10 -pc_gamg_type agg -pc_gamg_agg_nsmooths 1 -pc_gamg_coarse_eq_limit 1000 -mg_levels_ksp_type chebyshev -mg_levels_pc_type sor -pc_gamg_reuse_interpolation true -two_solves -use_mat_nearnullspace -pc_gamg_use_sa_esteig 0 -mg_levels_esteig_ksp_max_it 10

   test:
      suffix: nns_telescope
      nsize: 2
      args: -use_mat_nearnullspace -ksp_monitor_short -pc_type telescope -pc_telescope_reduction_factor 2 -telescope_pc_type gamg -telescope_pc_gamg_esteig_ksp_type cg -telescope_pc_gamg_esteig_ksp_max_it 10

   test:
      suffix: seqaijmkl
      nsize: 8
      requires: mkl_sparse
      args: -ne 9 -alpha 1.e-3 -ksp_type cg -pc_type gamg -pc_gamg_agg_nsmooths 1 -pc_gamg_reuse_interpolation true -two_solves -ksp_converged_reason -use_mat_nearnullspace -pc_gamg_square_graph 1 -mg_levels_ksp_max_it 2 -mg_levels_ksp_type chebyshev -mg_levels_pc_type jacobi -mg_levels_ksp_chebyshev_esteig 0,0.2,0,1.05 -pc_gamg_esteig_ksp_max_it 10 -pc_gamg_threshold 0.01 -pc_gamg_coarse_eq_limit 2000 -pc_gamg_process_eq_limit 200 -pc_gamg_repartition false -pc_mg_cycle_type v -ksp_monitor_short -mat_seqaij_type seqaijmkl

TEST*/
