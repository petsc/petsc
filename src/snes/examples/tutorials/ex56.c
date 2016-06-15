static char help[] = "3D, tri-linear quadrilateral (Q1), displacement finite element formulation\n\
of linear elasticity.  E=1.0, nu=0.25.\n\
Unit cube domain with Dirichlet boundary condition at x=0.\n\
Load of [1, 2, 3] at x=1. (not a true uniform load).\n\
  -ne <size>      : number of (square) quadrilateral elements in each dimension\n\
  -alpha <v>      : scaling of material coeficient in embedded circle\n\n";

#include <petscdmplex.h>
#include <petscsnes.h>
#include <petscds.h>
#include <petscdmforest.h>
/* #define MARK_DEBUG */
static PetscBool log_stages = PETSC_TRUE;
static PetscErrorCode MaybeLogStagePush(PetscLogStage stage) { return log_stages ? PetscLogStagePush(stage) : 0; }
static PetscErrorCode MaybeLogStagePop() { return log_stages ? PetscLogStagePop() : 0; }

static PetscReal s_soft_alpha=1.e-3;
static PetscReal s_mu=0.4;
static PetscReal s_lambda=0.4;

#undef __FUNCT__
#define __FUNCT__ "f0_bd_u_3d"
void f0_bd_u_3d(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                PetscReal t, const PetscReal x[], const PetscReal n[], PetscScalar f0[])
{
  PetscReal r[]={x[0]-.5,x[1]-.5,x[2]-.5};
  PetscInt i;
  for (i = 0; i < dim; ++i) f0[i] = 1+i;
  f0[1] += -r[2]; /* add a twist around x-axis */
  f0[2] +=  r[1];
}

#undef __FUNCT__
#define __FUNCT__ "f1_bd_u"
void f1_bd_u(PetscInt dim, PetscInt Nf, PetscInt NfAux,
             const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
             const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
             PetscReal t, const PetscReal x[], const PetscReal n[], PetscScalar f1[])
{
  const PetscInt Ncomp = dim;
  PetscInt       comp, d;
  for (comp = 0; comp < Ncomp; ++comp) {
    for (d = 0; d < dim; ++d) {
      f1[comp*dim+d] = 0.0;
    }
  }
}

#undef __FUNCT__
#define __FUNCT__ "f1_u_3d"
/* gradU[comp*dim+d] = {u_x, u_y} or {u_x, u_y, u_z} */
void f1_u_3d(PetscInt dim, PetscInt Nf, PetscInt NfAux,
          const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
          const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
          PetscReal t, const PetscReal x[], PetscScalar f1[])
{
  PetscReal trace=0,mu=s_mu,lambda=s_lambda,rad;
  PetscInt i,j;
  for (i=0,rad=0.;i<dim;i++) {
    PetscReal t=x[i]-.5;
    rad += t*t;
  }
  rad = PetscSqrtReal(rad);
  if (rad>0.25) {
    mu *= s_soft_alpha;
    /* lambda *= s_soft_alpha; */ /* we could keep the bulk the same like rubberish */
  }
  for (i=0; i < dim; ++i) {
    trace += PetscRealPart(u_x[i]);
  } /* for */
  for (i = 0; i < dim; ++i) {
    f1[i*dim+i] += lambda * trace;
    for (j=0; j < dim; ++j) { /* stress[i*_dim+j] += mu * (disp_x[i*_dim+j] + disp_x[j*_dim+i]  */
      f1[i*dim+j] += mu * (u_x[i*dim+j] + u_x[j*dim+i]);
    } /* for */
  } /* for */
}

#define IDX(ii,jj,kk,ll) (27*ii+9*jj+3*kk+ll)
#undef __FUNCT__
#define __FUNCT__ "g3_uu_3d"
void g3_uu_3d(PetscInt dim, PetscInt Nf, PetscInt NfAux,
          const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
          const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
          PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscScalar g3[])
{
  PetscReal mu=s_mu, lambda=s_lambda,rad;
  PetscInt i;
  for (i=0,rad=0.;i<dim;i++) {
    PetscReal t=x[i]-.5;
    rad += t*t;
  }
  rad = PetscSqrtReal(rad);
  if (rad>0.25) {
    mu *= s_soft_alpha;
    /* lambda *= s_soft_alpha; */ /* we could keep the bulk the same like rubberish */
  }
  if (1) {
    g3[0] += lambda;
    g3[0] += mu;
    g3[0] += mu;
    g3[4] += lambda;
    g3[8] += lambda;
    g3[10] += mu;
    g3[12] += mu;
    g3[20] += mu;
    g3[24] += mu;
    g3[28] += mu;
    g3[30] += mu;
    g3[36] += lambda;
    g3[40] += lambda;
    g3[40] += mu;
    g3[40] += mu;
    g3[44] += lambda;
    g3[50] += mu;
    g3[52] += mu;
    g3[56] += mu;
    g3[60] += mu;
    g3[68] += mu;
    g3[70] += mu;
    g3[72] += lambda;
    g3[76] += lambda;
    g3[80] += lambda;
    g3[80] += mu;
    g3[80] += mu;
  } else {
    static int cc=-1;
    cc++;
    int i,j,k,l;
    for (i = 0; i < 3; ++i) {
      for (j = 0; j < 3; ++j) {
        for (k = 0; k < 3; ++k) {
          for (l = 0; l < 3; ++l) {
            if (k==l && i==j) g3[IDX(i,j,k,l)] += lambda;
            if (i==k && j==l) g3[IDX(i,j,k,l)] += mu;
            if (i==l && j==k) g3[IDX(i,j,k,l)] += mu;
	    if (k==l && i==j && !cc) PetscPrintf(PETSC_COMM_WORLD,"g3[%d] += lambda;\n",IDX(i,j,k,l));
	    if (i==k && j==l && !cc) PetscPrintf(PETSC_COMM_WORLD,"g3[%d] += mu;\n",IDX(i,j,k,l));
	    if (i==l && j==k && !cc) PetscPrintf(PETSC_COMM_WORLD,"g3[%d] += mu;\n",IDX(i,j,k,l));
          }
        }
      }
    }
  }
}

void f0_u(PetscInt dim, PetscInt Nf, PetscInt NfAux,
          const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
          const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
          PetscReal t, const PetscReal x[], PetscScalar f0[])
{
  const    PetscInt Ncomp = dim;
  PetscInt comp;

  for (comp = 0; comp < Ncomp; ++comp) f0[comp] = 0.0;
}

PetscErrorCode zero(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *ctx)
{
  const PetscInt Ncomp = dim;
  PetscInt       comp;

  for (comp = 0; comp < Ncomp; ++comp) u[comp] = 0;
  return 0;
}

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **args)
{
  Mat            Amat;
  PetscErrorCode ierr;
  SNES           snes;
  KSP            ksp;
  MPI_Comm       comm;
  PetscMPIInt    npe,rank;
  PetscLogStage  stage[7];
  PetscBool      two_solves=PETSC_FALSE,test_nonzero_cols=PETSC_FALSE,use_nearnullspace=PETSC_TRUE;
  Vec            xx,bb;
  PetscInt       i,ne=7,dim=3,cells[3]={7,7,7};
  DM             dm,distdm;
  PetscBool      flg,use_snes=PETSC_FALSE;

  PetscFunctionBeginUser;
  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  comm = PETSC_COMM_WORLD;
  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm, &npe);CHKERRQ(ierr);
  ierr = PetscOptionsBegin(comm,NULL,"3D bilinear Q1 elasticity options","");CHKERRQ(ierr);
  {
    ierr = PetscOptionsInt("-ne","number of elements in each direction","",ne,&ne,&flg);CHKERRQ(ierr);
    if (flg) {
      for (i=0; i<dim; i++) cells[i] = ne; /* this can be generalized now */
    }
    ierr = PetscOptionsBool("-log_stages","Log stages of solve separately","",log_stages,&log_stages,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-alpha","material coefficient inside circle","",s_soft_alpha,&s_soft_alpha,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsBool("-two_solves","solve additional variant of the problem","",two_solves,&two_solves,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsBool("-test_nonzero_cols","nonzero test","",test_nonzero_cols,&test_nonzero_cols,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsBool("-use_mat_nearnullspace","MatNearNullSpace API test","",use_nearnullspace,&use_nearnullspace,NULL);CHKERRQ(ierr);
    i = 3;
    ierr = PetscOptionsInt("-mat_block_size","","",i,&i,&flg);CHKERRQ(ierr);
    if (!flg || i!=3) SETERRQ2(PETSC_COMM_WORLD, PETSC_ERR_USER, "'-mat_block_size 3' must be set (%D) and = 3 (%D)",flg,flg? i : 3);
  }
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  if (log_stages) {
    ierr = PetscLogStageRegister("Mesh Setup", &stage[6]);CHKERRQ(ierr);
    ierr = PetscLogStageRegister("1st Setup", &stage[0]);CHKERRQ(ierr);
    ierr = PetscLogStageRegister("1st Solve", &stage[1]);CHKERRQ(ierr);
    ierr = PetscLogStageRegister("2nd Setup (RAP)", &stage[2]);CHKERRQ(ierr);
    ierr = PetscLogStageRegister("2nd Solve", &stage[3]);CHKERRQ(ierr);
    ierr = PetscLogStageRegister("3rd Setup (nothing)", &stage[4]);CHKERRQ(ierr);
    ierr = PetscLogStageRegister("3rd Solve", &stage[5]);CHKERRQ(ierr);
  } else {
    for (i=0; i<(PetscInt)(sizeof(stage)/sizeof(stage[0])); i++) stage[i] = -1;
  }
  /* create DM, Plex calls DMSetup */
  ierr = MaybeLogStagePush(stage[6]);CHKERRQ(ierr);
  ierr = DMPlexCreateHexBoxMesh(comm, dim, cells, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, &dm);CHKERRQ(ierr);
  {
    DMLabel         label;
    IS              is;
    ierr = DMCreateLabel(dm, "boundary");CHKERRQ(ierr);
    ierr = DMGetLabel(dm, "boundary", &label);CHKERRQ(ierr);
    ierr = DMPlexMarkBoundaryFaces(dm, label);CHKERRQ(ierr);
    ierr = DMGetStratumIS(dm, "boundary", 1,  &is);CHKERRQ(ierr);
    ierr = DMCreateLabel(dm,"Faces");CHKERRQ(ierr);
    if (is) {
      PetscInt        d, f, Nf;
      const PetscInt *faces;
      PetscInt        csize;
      PetscSection    cs;
      Vec             coordinates ;
      DM              cdm;
      ierr = ISGetLocalSize(is, &Nf);CHKERRQ(ierr);
      ierr = ISGetIndices(is, &faces);CHKERRQ(ierr);

      ierr = DMGetCoordinatesLocal(dm, &coordinates);CHKERRQ(ierr);
      ierr = DMGetCoordinateDM(dm, &cdm);CHKERRQ(ierr);
      ierr = DMGetDefaultSection(cdm, &cs);CHKERRQ(ierr);

      /* Check for each boundary face if any component of its centroid is either 0.0 or 1.0 */
      for (f = 0; f < Nf; ++f) {
        PetscReal   faceCoord;
        PetscInt    b,v;
        PetscScalar *coords = NULL;
        PetscInt    Nv;

        ierr = DMPlexVecGetClosure(cdm, cs, coordinates, faces[f], &csize, &coords);CHKERRQ(ierr);
        Nv   = csize/dim; /* Calculate mean coordinate vector */
        for (d = 0; d < dim; ++d) {
          faceCoord = 0.0;
          for (v = 0; v < Nv; ++v) faceCoord += PetscRealPart(coords[v*dim+d]);
          faceCoord /= Nv;
          for (b = 0; b < 2; ++b) {
            if (PetscAbs(faceCoord - b*1.0) < PETSC_SMALL) {
              ierr = DMSetLabelValue(dm, "Faces", faces[f], d*2+b+1);CHKERRQ(ierr);
            }
          }
        }
        ierr = DMPlexVecRestoreClosure(cdm, cs, coordinates, faces[f], &csize, &coords);CHKERRQ(ierr);
      }
      ierr = ISRestoreIndices(is, &faces);CHKERRQ(ierr);
    }
    ierr = ISDestroy(&is);CHKERRQ(ierr);
    ierr = DMGetLabel(dm, "Faces", &label);CHKERRQ(ierr);
    ierr = DMPlexLabelComplete(dm, label);CHKERRQ(ierr);
  }
  ierr = PetscObjectSetName( (PetscObject)dm,"Mesh");CHKERRQ(ierr);
  { /* convert to p4est, and distribute */
    char      convType[256];
    PetscBool flg;
    DM newdm;
    ierr = PetscOptionsBegin(comm, "", "Mesh conversion options", "DMPLEX");CHKERRQ(ierr);
    ierr = PetscOptionsFList("-dm_type","Convert DMPlex to another format (should not be Plex!)","ex56.c",DMList,DMPLEX,convType,256,&flg);CHKERRQ(ierr);
    ierr = PetscOptionsEnd();
    if (flg) {
      ierr = DMConvert(dm,convType,&newdm);CHKERRQ(ierr);
      if (newdm) {
        const char *prefix;
        PetscBool isForest;
        ierr = PetscObjectGetOptionsPrefix((PetscObject)dm,&prefix);CHKERRQ(ierr);
        ierr = PetscObjectSetOptionsPrefix((PetscObject)newdm,prefix);CHKERRQ(ierr);
        ierr = DMIsForest(newdm,&isForest);CHKERRQ(ierr);
        if (isForest) {
        } else SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_USER, "Converted to non Forest?");
        ierr = DMDestroy(&dm);CHKERRQ(ierr);
        dm   = newdm;
      } else SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_USER, "Convert failed?");
    } else {
      /* Plex Distribute mesh over processes */
      ierr = DMPlexDistribute(dm, 0, NULL, &distdm);CHKERRQ(ierr);
      if (distdm) {
        ierr = DMDestroy(&dm);CHKERRQ(ierr);
        dm   = distdm;
      }
    }
  }
  ierr = DMSetFromOptions(dm);CHKERRQ(ierr); /* refinement done here in Plex, p4est */
  ierr = DMSetUp(dm);CHKERRQ(ierr); /* plex only */
  ierr = DMViewFromOptions(dm, NULL, "-dm_view");CHKERRQ(ierr);
  /* fem */
  {
    const PetscInt Ncomp = dim;
    const PetscInt components[] = {0,1,2};
    const PetscInt Nfid = 1, Npid = 1;
    const PetscInt fid[] = {1}; /* The fixed faces (x=0) */
    const PetscInt pid[] = {2}; /* The faces with loading (x=1) */
    PetscQuadrature q;
    PetscInt        order;
    PetscFE         feBd,fe;
    PetscDS         prob;
    DM              cdm = dm;

    ierr = PetscFECreateDefault(dm, dim, dim, PETSC_FALSE, NULL, 2, &fe);CHKERRQ(ierr); /* elasticity */
    ierr = PetscObjectSetName((PetscObject) fe, "deformation");CHKERRQ(ierr);
    ierr = PetscFEGetQuadrature(fe, &q);CHKERRQ(ierr);
    ierr = PetscQuadratureGetOrder(q, &order);CHKERRQ(ierr);
    ierr = PetscFECreateDefault(dm, dim-1, dim, PETSC_FALSE, NULL, order, &feBd);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) feBd, "deformation");CHKERRQ(ierr);
    ierr = DMGetDS(dm, &prob);CHKERRQ(ierr);

      /* FEM prob */
    ierr = PetscDSSetDiscretization(prob, 0, (PetscObject) fe);CHKERRQ(ierr);
    ierr = PetscDSSetBdDiscretization(prob, 0, (PetscObject) feBd);CHKERRQ(ierr);
    ierr = PetscDSSetResidual(prob, 0, f0_u, f1_u_3d);CHKERRQ(ierr);
    ierr = PetscDSSetJacobian(prob, 0, 0, NULL, NULL, NULL, g3_uu_3d);CHKERRQ(ierr);
    ierr = PetscDSSetBdResidual(prob, 0, f0_bd_u_3d, f1_bd_u);CHKERRQ(ierr);
    /* bcs */
    ierr = PetscDSAddBoundary(prob, PETSC_TRUE, "fixed", "Faces", 0, Ncomp, components, (void (*)()) zero, Nfid, fid, NULL);CHKERRQ(ierr);
    ierr = PetscDSAddBoundary(prob, PETSC_FALSE, "traction", "Faces", 0, Ncomp, components, NULL, Npid, pid, NULL);CHKERRQ(ierr);
    while (cdm) {
      /* ierr = PetscDSSetBdJacobian(prob, 0, 0, NULL, g1_bd_uu_3d, NULL, NULL);CHKERRQ(ierr); */
      ierr = DMSetDS(cdm,prob);CHKERRQ(ierr);
#if defined(MARK_DEBUG)
      ierr = DMView(cdm,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
#endif
      ierr = DMGetCoarseDM(cdm, &cdm);CHKERRQ(ierr);
    }
    ierr = PetscFEDestroy(&fe);CHKERRQ(ierr);
    ierr = PetscFEDestroy(&feBd);CHKERRQ(ierr);
  }
  if (use_nearnullspace) {
    /* Set up the near null space (a.k.a. rigid body modes) that will be used by the multigrid preconditioner */
    DM           subdm;
    MatNullSpace nearNullSpace;
    PetscInt     fields = 0;
    PetscObject  deformation;
    ierr = DMCreateSubDM(dm, 1, &fields, NULL, &subdm);CHKERRQ(ierr);
    ierr = DMPlexCreateRigidBody(subdm, &nearNullSpace);CHKERRQ(ierr);
    ierr = DMGetField(dm, 0, &deformation);CHKERRQ(ierr);
    ierr = PetscObjectCompose(deformation, "nearnullspace", (PetscObject) nearNullSpace);CHKERRQ(ierr);
    ierr = DMDestroy(&subdm);CHKERRQ(ierr);
    ierr = MatNullSpaceDestroy(&nearNullSpace);CHKERRQ(ierr);
  }
  ierr = MaybeLogStagePop();CHKERRQ(ierr);
  ierr = MaybeLogStagePush(stage[0]);CHKERRQ(ierr);
  /* snes */
  ierr = SNESCreate(comm, &snes);CHKERRQ(ierr);
  ierr = SNESSetDM(snes, dm);CHKERRQ(ierr);
  ierr = DMSetMatType(dm,MATAIJ);CHKERRQ(ierr);
  ierr = SNESSetFromOptions(snes);CHKERRQ(ierr);
  ierr = DMSNESSetFunctionLocal(dm,  (PetscErrorCode (*)(DM,Vec,Vec,void*))DMPlexSNESComputeResidualFEM,NULL);CHKERRQ(ierr);
  ierr = DMSNESSetJacobianLocal(dm,  (PetscErrorCode (*)(DM,Vec,Mat,Mat,void*))DMPlexSNESComputeJacobianFEM,NULL);CHKERRQ(ierr);
  ierr = DMCreateMatrix(dm, &Amat);CHKERRQ(ierr);
  ierr = SNESSetJacobian(snes, Amat, Amat, NULL, NULL);CHKERRQ(ierr);
  ierr = MatCreateVecs(Amat,&xx,&bb);CHKERRQ(ierr);
  ierr = SNESSetUp(snes);CHKERRQ(ierr);
  ierr = SNESGetKSP(snes, &ksp);CHKERRQ(ierr);
  ierr = KSPSetComputeSingularValues( ksp,PETSC_TRUE);CHKERRQ(ierr);
  /* test BCs */
  ierr = VecZeroEntries(xx);CHKERRQ(ierr);
  if (test_nonzero_cols) {
    if (rank==0) ierr = VecSetValue(xx,0,1.0,INSERT_VALUES);CHKERRQ(ierr);
    ierr = VecAssemblyBegin(xx);CHKERRQ(ierr);
    ierr = VecAssemblyEnd(xx);CHKERRQ(ierr);
  }
  ierr = SNESComputeJacobian(snes, bb, Amat, Amat);CHKERRQ(ierr);
  ierr = SNESComputeFunction(snes, xx, bb);CHKERRQ(ierr);
  if (!use_snes) {
    /* PC setup basically */
    ierr = KSPSetOperators(ksp, Amat, Amat);CHKERRQ(ierr);
    ierr = KSPSetUp(ksp);CHKERRQ(ierr);
    ierr = KSPSetInitialGuessNonzero(ksp,PETSC_TRUE);CHKERRQ(ierr);
  }
  ierr = VecGetSize(bb,&i);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"\t[%d]%s %d/%d eqs in vector\n",rank,__FUNCT__,i,i/dim);CHKERRQ(ierr);
#if defined(MARK_DEBUG)
  ierr = VecGetBlockSize(bb,&i);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"\t[%d]%s block size %d\n",rank,__FUNCT__,i,i/dim);CHKERRQ(ierr);
  if (PETSC_FALSE) {
    PetscViewer viewer;
    ierr = PetscViewerASCIIOpen(comm, "Amat.m", &viewer);CHKERRQ(ierr);
    ierr = PetscViewerPushFormat(viewer, PETSC_VIEWER_ASCII_MATLAB);CHKERRQ(ierr);
    ierr = MatView(Amat,viewer);CHKERRQ(ierr);
    ierr = PetscViewerPopFormat(viewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer);
  }
#endif
  ierr = MaybeLogStagePop();CHKERRQ(ierr);
  /* 1st solve */
  ierr = MaybeLogStagePush(stage[1]);CHKERRQ(ierr);
  if (use_snes) {
    ierr = VecZeroEntries(bb);CHKERRQ(ierr);
    ierr = SNESSolve(snes, bb, xx);CHKERRQ(ierr);
  } else {
    ierr = KSPSolve(ksp, bb, xx);CHKERRQ(ierr);
  }
  ierr = MaybeLogStagePop();CHKERRQ(ierr);
  /* 2nd solve */
  if (two_solves) {
    PetscReal emax, emin;
    Vec       res;

    ierr = MaybeLogStagePush(stage[2]);CHKERRQ(ierr);
    /* PC setup basically */
    ierr = MatScale(Amat, 100000.0);CHKERRQ(ierr);
    if (use_snes) {
      ierr = SNESSetUp(snes);CHKERRQ(ierr);
    } else {
      ierr = KSPSetOperators(ksp, Amat, Amat);CHKERRQ(ierr);
      ierr = KSPSetUp(ksp);CHKERRQ(ierr);
    }
    ierr = MaybeLogStagePop();CHKERRQ(ierr);
    ierr = MaybeLogStagePush(stage[3]);CHKERRQ(ierr);
    if (use_snes) {
      ierr = SNESSolve(snes, bb, xx);CHKERRQ(ierr);
    } else {
      ierr = VecZeroEntries(xx);CHKERRQ(ierr);
      ierr = KSPSolve(ksp, bb, xx);CHKERRQ(ierr);
    }
    ierr = MaybeLogStagePop();CHKERRQ(ierr);
    ierr = MaybeLogStagePush(stage[4]);CHKERRQ(ierr);

    /* 3rd solve */
    if (use_snes) {
      ierr = SNESSetUp(snes);CHKERRQ(ierr);
    } else {
      ierr = KSPSetOperators(ksp, Amat, Amat);CHKERRQ(ierr);
      ierr = KSPSetUp(ksp);CHKERRQ(ierr);
    }
    ierr = MaybeLogStagePop();CHKERRQ(ierr);
    ierr = MaybeLogStagePush(stage[5]);CHKERRQ(ierr);
    if (use_snes) {
      ierr = SNESSolve(snes, bb, xx);CHKERRQ(ierr);
    } else {
      ierr = KSPSolve(ksp, bb, xx);CHKERRQ(ierr);
    }
    ierr = MaybeLogStagePop();CHKERRQ(ierr);

    ierr = VecDuplicate(xx, &res);CHKERRQ(ierr);
    ierr = MatMult(Amat, xx, res);CHKERRQ(ierr);
    ierr = VecAXPY(bb, -1.0, res);CHKERRQ(ierr);
    ierr = VecDestroy(&res);CHKERRQ(ierr);
    ierr = KSPComputeExtremeSingularValues(ksp, &emax, &emin);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"[%d]%s emax=%10.4E\n",0,__FUNCT__,emax);CHKERRQ(ierr);
  }

  /* Free work space */
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  ierr = SNESDestroy(&snes);CHKERRQ(ierr);
  ierr = VecDestroy(&xx);CHKERRQ(ierr);
  ierr = VecDestroy(&bb);CHKERRQ(ierr);
  ierr = MatDestroy(&Amat);CHKERRQ(ierr);
 
  ierr = PetscFinalize();
  return ierr;
}
