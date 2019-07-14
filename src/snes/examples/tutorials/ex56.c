static char help[] = "3D, tri-quadratic hexahedra (Q1), displacement finite element formulation\n\
of linear elasticity.  E=1.0, nu=1/3.\n\
Unit cube domain with Dirichlet boundary\n\n";

#include <petscdmplex.h>
#include <petscsnes.h>
#include <petscds.h>
#include <petscdmforest.h>

static PetscReal s_soft_alpha=1.e-3;
static PetscReal s_mu=0.4;
static PetscReal s_lambda=0.4;

static void f0_bd_u_3d(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                       const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                       const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                       PetscReal t, const PetscReal x[], const PetscReal n[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  f0[0] = 1;     /* x direction pull */
  f0[1] = -x[2]; /* add a twist around x-axis */
  f0[2] =  x[1];
}

static void f1_bd_u(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                    const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                    const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                    PetscReal t, const PetscReal x[], const PetscReal n[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f1[])
{
  const PetscInt Ncomp = dim;
  PetscInt       comp, d;
  for (comp = 0; comp < Ncomp; ++comp) {
    for (d = 0; d < dim; ++d) {
      f1[comp*dim+d] = 0.0;
    }
  }
}

/* gradU[comp*dim+d] = {u_x, u_y} or {u_x, u_y, u_z} */
static void f1_u_3d_alpha(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                          const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                          const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                          PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f1[])
{
  PetscReal trace,mu=s_mu,lambda=s_lambda,rad;
  PetscInt i,j;
  for (i=0,rad=0.;i<dim;i++) {
    PetscReal t=x[i];
    rad += t*t;
  }
  rad = PetscSqrtReal(rad);
  if (rad>0.25) {
    mu *= s_soft_alpha;
    lambda *= s_soft_alpha; /* we could keep the bulk the same like rubberish */
  }
  for (i=0,trace=0; i < dim; ++i) {
    trace += PetscRealPart(u_x[i*dim+i]);
  }
  for (i=0; i < dim; ++i) {
    for (j=0; j < dim; ++j) {
      f1[i*dim+j] = mu*(u_x[i*dim+j]+u_x[j*dim+i]);
    }
    f1[i*dim+i] += lambda*trace;
  }
}

/* gradU[comp*dim+d] = {u_x, u_y} or {u_x, u_y, u_z} */
static void f1_u_3d(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                    const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                    const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                    PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f1[])
{
  PetscReal trace,mu=s_mu,lambda=s_lambda;
  PetscInt i,j;
  for (i=0,trace=0; i < dim; ++i) {
    trace += PetscRealPart(u_x[i*dim+i]);
  }
  for (i=0; i < dim; ++i) {
    for (j=0; j < dim; ++j) {
      f1[i*dim+j] = mu*(u_x[i*dim+j]+u_x[j*dim+i]);
    }
    f1[i*dim+i] += lambda*trace;
  }
}

/* 3D elasticity */
#define IDX(ii,jj,kk,ll) (27*ii+9*jj+3*kk+ll)

void g3_uu_3d_private( PetscScalar g3[], const PetscReal mu, const PetscReal lambda)
{
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
    int        i,j,k,l;
    static int cc=-1;
    cc++;
    for (i = 0; i < 3; ++i) {
      for (j = 0; j < 3; ++j) {
        for (k = 0; k < 3; ++k) {
          for (l = 0; l < 3; ++l) {
            if (k==l && i==j) g3[IDX(i,j,k,l)] += lambda;
            if (i==k && j==l) g3[IDX(i,j,k,l)] += mu;
            if (i==l && j==k) g3[IDX(i,j,k,l)] += mu;
	    if (k==l && i==j && !cc) (void) PetscPrintf(PETSC_COMM_WORLD,"g3[%d] += lambda;\n",IDX(i,j,k,l));
	    if (i==k && j==l && !cc) (void) PetscPrintf(PETSC_COMM_WORLD,"g3[%d] += mu;\n",IDX(i,j,k,l));
	    if (i==l && j==k && !cc) (void) PetscPrintf(PETSC_COMM_WORLD,"g3[%d] += mu;\n",IDX(i,j,k,l));
          }
        }
      }
    }
  }
}

static void g3_uu_3d_alpha(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                           const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                           const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                           PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g3[])
{
  PetscReal mu=s_mu, lambda=s_lambda,rad;
  PetscInt i;
  for (i=0,rad=0.;i<dim;i++) {
    PetscReal t=x[i];
    rad += t*t;
  }
  rad = PetscSqrtReal(rad);
  if (rad>0.25) {
    mu *= s_soft_alpha;
    lambda *= s_soft_alpha; /* we could keep the bulk the same like rubberish */
  }
  g3_uu_3d_private(g3,mu,lambda);
}

static void g3_uu_3d(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                     const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                     const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                     PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g3[])
{
  g3_uu_3d_private(g3,s_mu,s_lambda);
}

static void f0_u(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                 const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                 const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                 PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  const    PetscInt Ncomp = dim;
  PetscInt comp;

  for (comp = 0; comp < Ncomp; ++comp) f0[comp] = 0.0;
}

/* PI_i (x_i^4 - x_i^2) */
static void f0_u_x4(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                    const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                    const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                    PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  const    PetscInt Ncomp = dim;
  PetscInt comp,i;

  for (comp = 0; comp < Ncomp; ++comp) {
    f0[comp] = 1e5;
    for (i = 0; i < Ncomp; ++i) {
      f0[comp] *= /* (comp+1)* */(x[i]*x[i]*x[i]*x[i] - x[i]*x[i]); /* assumes (0,1]^D domain */
    }
  }
}

PetscErrorCode zero(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *ctx)
{
  const PetscInt Ncomp = dim;
  PetscInt       comp;

  for (comp = 0; comp < Ncomp; ++comp) u[comp] = 0;
  return 0;
}

int main(int argc,char **args)
{
  Mat            Amat;
  PetscErrorCode ierr;
  SNES           snes;
  KSP            ksp;
  MPI_Comm       comm;
  PetscMPIInt    rank;
  PetscLogStage  stage[17];
  PetscBool      test_nonzero_cols=PETSC_FALSE,use_nearnullspace=PETSC_TRUE,attach_nearnullspace=PETSC_FALSE;
  Vec            xx,bb;
  PetscInt       iter,i,N,dim=3,cells[3]={1,1,1},max_conv_its,local_sizes[7],run_type=1;
  DM             dm,distdm,basedm;
  PetscBool      flg;
  char           convType[256];
  PetscReal      Lx,mdisp[10],err[10];
  const char * const options[10] = {"-ex56_dm_refine 0",
                                    "-ex56_dm_refine 1",
                                    "-ex56_dm_refine 2",
                                    "-ex56_dm_refine 3",
                                    "-ex56_dm_refine 4",
                                    "-ex56_dm_refine 5",
                                    "-ex56_dm_refine 6",
                                    "-ex56_dm_refine 7",
                                    "-ex56_dm_refine 8",
                                    "-ex56_dm_refine 9"};
  PetscFunctionBeginUser;
  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  comm = PETSC_COMM_WORLD;
  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
  /* options */
  ierr = PetscOptionsBegin(comm,NULL,"3D bilinear Q1 elasticity options","");CHKERRQ(ierr);
  {
    i = 3;
    ierr = PetscOptionsIntArray("-cells", "Number of (flux tube) processor in each dimension", "ex56.c", cells, &i, NULL);CHKERRQ(ierr);

    Lx = 1.; /* or ne for rod */
    max_conv_its = 3;
    ierr = PetscOptionsInt("-max_conv_its","Number of iterations in convergence study","",max_conv_its,&max_conv_its,NULL);CHKERRQ(ierr);
    if (max_conv_its<=0 || max_conv_its>7) SETERRQ1(PETSC_COMM_WORLD, PETSC_ERR_USER, "Bad number of iterations for convergence test (%D)",max_conv_its);
    ierr = PetscOptionsReal("-lx","Length of domain","",Lx,&Lx,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-alpha","material coefficient inside circle","",s_soft_alpha,&s_soft_alpha,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsBool("-test_nonzero_cols","nonzero test","",test_nonzero_cols,&test_nonzero_cols,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsBool("-use_mat_nearnullspace","MatNearNullSpace API test","",use_nearnullspace,&use_nearnullspace,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsBool("-attach_mat_nearnullspace","MatNearNullSpace API test (via MatSetNearNullSpace)","",attach_nearnullspace,&attach_nearnullspace,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-run_type","0: twisting load on cantalever, 1: 3rd order accurate convergence test","",run_type,&run_type,NULL);CHKERRQ(ierr);
    i = 3;
    ierr = PetscOptionsInt("-mat_block_size","","",i,&i,&flg);CHKERRQ(ierr);
    if (!flg || i!=3) SETERRQ2(PETSC_COMM_WORLD, PETSC_ERR_USER, "'-mat_block_size 3' must be set (%D) and = 3 (%D)",flg,flg? i : 3);
  }
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  ierr = PetscLogStageRegister("Mesh Setup", &stage[16]);CHKERRQ(ierr);
  for (iter=0 ; iter<max_conv_its ; iter++) {
    char str[] = "Solve 0";
    str[6] += iter;
    ierr = PetscLogStageRegister(str, &stage[iter]);CHKERRQ(ierr);
  }
  /* create DM, Plex calls DMSetup */
  ierr = PetscLogStagePush(stage[16]);CHKERRQ(ierr);
  ierr = DMPlexCreateBoxMesh(comm, dim, PETSC_FALSE, cells, NULL, NULL, NULL, PETSC_TRUE, &dm);CHKERRQ(ierr);
  {
    DMLabel         label;
    IS              is;
    ierr = DMCreateLabel(dm, "boundary");CHKERRQ(ierr);
    ierr = DMGetLabel(dm, "boundary", &label);CHKERRQ(ierr);
    ierr = DMPlexMarkBoundaryFaces(dm, 1, label);CHKERRQ(ierr);
    if (!run_type) {
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
        ierr = DMGetLocalSection(cdm, &cs);CHKERRQ(ierr);
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
              if (PetscAbs(faceCoord - b) < PETSC_SMALL) { /* domain have not been set yet, still [0,1]^3 */
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
  }
  {
    PetscInt    dimEmbed, i;
    PetscInt    nCoords;
    PetscScalar *coords,bounds[] = {0,1,-.5,.5,-.5,.5,}; /* x_min,x_max,y_min,y_max */
    Vec         coordinates;
    bounds[1] = Lx;
    if (run_type==1) {
      for (i = 0; i < 2*dim; i++) bounds[i] = (i%2) ? 1 : 0;
    }
    ierr = DMGetCoordinatesLocal(dm,&coordinates);CHKERRQ(ierr);
    ierr = DMGetCoordinateDim(dm,&dimEmbed);CHKERRQ(ierr);
    if (dimEmbed != dim) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"dimEmbed != dim %D",dimEmbed);
    ierr = VecGetLocalSize(coordinates,&nCoords);CHKERRQ(ierr);
    if (nCoords % dimEmbed) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Coordinate vector the wrong size");
    ierr = VecGetArray(coordinates,&coords);CHKERRQ(ierr);
    for (i = 0; i < nCoords; i += dimEmbed) {
      PetscInt    j;
      PetscScalar *coord = &coords[i];
      for (j = 0; j < dimEmbed; j++) {
        coord[j] = bounds[2 * j] + coord[j] * (bounds[2 * j + 1] - bounds[2 * j]);
      }
    }
    ierr = VecRestoreArray(coordinates,&coords);CHKERRQ(ierr);
    ierr = DMSetCoordinatesLocal(dm,coordinates);CHKERRQ(ierr);
  }

  /* convert to p4est, and distribute */

  ierr = PetscOptionsBegin(comm, "", "Mesh conversion options", "DMPLEX");CHKERRQ(ierr);
  ierr = PetscOptionsFList("-dm_type","Convert DMPlex to another format (should not be Plex!)","ex56.c",DMList,DMPLEX,convType,256,&flg);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();
  if (flg) {
    DM newdm;
    ierr = DMConvert(dm,convType,&newdm);CHKERRQ(ierr);
    if (newdm) {
      const char *prefix;
      PetscBool isForest;
      ierr = PetscObjectGetOptionsPrefix((PetscObject)dm,&prefix);CHKERRQ(ierr);
      ierr = PetscObjectSetOptionsPrefix((PetscObject)newdm,prefix);CHKERRQ(ierr);
      ierr = DMIsForest(newdm,&isForest);CHKERRQ(ierr);
      if (!isForest) SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_USER, "Converted to non Forest?");
      ierr = DMDestroy(&dm);CHKERRQ(ierr);
      dm   = newdm;
    } else SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_USER, "Convert failed?");
  } else {
    PetscPartitioner part;
    /* Plex Distribute mesh over processes */
    ierr = DMPlexGetPartitioner(dm,&part);CHKERRQ(ierr);
    ierr = PetscPartitionerSetFromOptions(part);CHKERRQ(ierr);
    ierr = DMPlexDistribute(dm, 0, NULL, &distdm);CHKERRQ(ierr);
    if (distdm) {
      const char *prefix;
      ierr = PetscObjectGetOptionsPrefix((PetscObject)dm,&prefix);CHKERRQ(ierr);
      ierr = PetscObjectSetOptionsPrefix((PetscObject)distdm,prefix);CHKERRQ(ierr);
      ierr = DMDestroy(&dm);CHKERRQ(ierr);
      dm   = distdm;
    }
  }
  ierr = PetscLogStagePop();CHKERRQ(ierr);
  basedm = dm; dm = NULL;

  for (iter=0 ; iter<max_conv_its ; iter++) {
    ierr = PetscLogStagePush(stage[16]);CHKERRQ(ierr);
    /* make new DM */
    ierr = DMClone(basedm, &dm);CHKERRQ(ierr);
    ierr = PetscObjectSetOptionsPrefix((PetscObject) dm, "ex56_");CHKERRQ(ierr);
    ierr = PetscObjectSetName( (PetscObject)dm,"Mesh");CHKERRQ(ierr);
    ierr = PetscOptionsClearValue(NULL,"-ex56_dm_refine");CHKERRQ(ierr);
    ierr = PetscOptionsInsertString(NULL,options[iter]);CHKERRQ(ierr);
    ierr = DMSetFromOptions(dm);CHKERRQ(ierr); /* refinement done here in Plex, p4est */
    /* snes */
    ierr = SNESCreate(comm, &snes);CHKERRQ(ierr);
    ierr = SNESSetDM(snes, dm);CHKERRQ(ierr);
    /* fem */
    {
      const PetscInt Ncomp = dim;
      const PetscInt components[] = {0,1,2};
      const PetscInt Nfid = 1, Npid = 1;
      const PetscInt fid[] = {1}; /* The fixed faces (x=0) */
      const PetscInt pid[] = {2}; /* The faces with loading (x=L_x) */
      PetscFE        fe;
      PetscDS        prob;
      DM             cdm = dm;

      ierr = PetscFECreateDefault(PetscObjectComm((PetscObject) dm), dim, dim, PETSC_FALSE, NULL, PETSC_DECIDE, &fe);CHKERRQ(ierr); /* elasticity */
      ierr = PetscObjectSetName((PetscObject) fe, "deformation");CHKERRQ(ierr);
      /* FEM prob */
      ierr = DMSetField(dm, 0, NULL, (PetscObject) fe);CHKERRQ(ierr);
      ierr = DMCreateDS(dm);CHKERRQ(ierr);
      ierr = DMGetDS(dm, &prob);CHKERRQ(ierr);
      /* setup problem */
      if (run_type==1) {
        ierr = PetscDSSetJacobian(prob, 0, 0, NULL, NULL, NULL, g3_uu_3d);CHKERRQ(ierr);
        ierr = PetscDSSetResidual(prob, 0, f0_u_x4, f1_u_3d);CHKERRQ(ierr);
      } else {
        ierr = PetscDSSetJacobian(prob, 0, 0, NULL, NULL, NULL, g3_uu_3d_alpha);CHKERRQ(ierr);
        ierr = PetscDSSetResidual(prob, 0, f0_u, f1_u_3d_alpha);CHKERRQ(ierr);
        ierr = PetscDSSetBdResidual(prob, 0, f0_bd_u_3d, f1_bd_u);CHKERRQ(ierr);
      }
      /* bcs */
      if (run_type==1) {
        PetscInt id = 1;
        ierr = DMAddBoundary(dm, DM_BC_ESSENTIAL, "wall", "boundary", 0, 0, NULL, (void (*)(void)) zero, 1, &id, NULL);CHKERRQ(ierr);
      } else {
        ierr = PetscDSAddBoundary(prob, DM_BC_ESSENTIAL, "fixed", "Faces", 0, Ncomp, components, (void (*)(void)) zero, Nfid, fid, NULL);CHKERRQ(ierr);
        ierr = PetscDSAddBoundary(prob, DM_BC_NATURAL, "traction", "Faces", 0, Ncomp, components, NULL, Npid, pid, NULL);CHKERRQ(ierr);
      }
      while (cdm) {
        ierr = DMCopyDisc(dm, cdm);CHKERRQ(ierr);
        ierr = DMGetCoarseDM(cdm, &cdm);CHKERRQ(ierr);
      }
      ierr = PetscFEDestroy(&fe);CHKERRQ(ierr);
    }
    /* vecs & mat */
    ierr = DMCreateGlobalVector(dm,&xx);CHKERRQ(ierr);
    ierr = VecDuplicate(xx, &bb);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) bb, "b");CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) xx, "u");CHKERRQ(ierr);
    ierr = DMCreateMatrix(dm, &Amat);CHKERRQ(ierr);
    ierr = VecGetSize(bb,&N);CHKERRQ(ierr);
    local_sizes[iter] = N;
    ierr = PetscInfo2(snes,"%D global equations, %D vertices\n",N,N/dim);CHKERRQ(ierr);
    if ((use_nearnullspace || attach_nearnullspace) && N/dim > 1) {
      /* Set up the near null space (a.k.a. rigid body modes) that will be used by the multigrid preconditioner */
      DM           subdm;
      MatNullSpace nearNullSpace;
      PetscInt     fields = 0;
      PetscObject  deformation;
      ierr = DMCreateSubDM(dm, 1, &fields, NULL, &subdm);CHKERRQ(ierr);
      ierr = DMPlexCreateRigidBody(subdm, &nearNullSpace);CHKERRQ(ierr);
      ierr = DMGetField(dm, 0, NULL, &deformation);CHKERRQ(ierr);
      ierr = PetscObjectCompose(deformation, "nearnullspace", (PetscObject) nearNullSpace);CHKERRQ(ierr);
      ierr = DMDestroy(&subdm);CHKERRQ(ierr);
      if (attach_nearnullspace) {
        ierr = MatSetNearNullSpace(Amat,nearNullSpace);CHKERRQ(ierr);
      }
      ierr = MatNullSpaceDestroy(&nearNullSpace);CHKERRQ(ierr); /* created by DM and destroyed by Mat */
    }
    ierr = DMPlexSetSNESLocalFEM(dm,NULL,NULL,NULL);CHKERRQ(ierr);
    ierr = SNESSetJacobian(snes, Amat, Amat, NULL, NULL);CHKERRQ(ierr);
    ierr = SNESSetFromOptions(snes);CHKERRQ(ierr);
    ierr = DMSetUp(dm);CHKERRQ(ierr);
    ierr = PetscLogStagePop();CHKERRQ(ierr);
    ierr = PetscLogStagePush(stage[16]);CHKERRQ(ierr);
    /* ksp */
    ierr = SNESGetKSP(snes, &ksp);CHKERRQ(ierr);
    ierr = KSPSetComputeSingularValues(ksp,PETSC_TRUE);CHKERRQ(ierr);
    /* test BCs */
    ierr = VecZeroEntries(xx);CHKERRQ(ierr);
    if (test_nonzero_cols) {
      if (!rank) {
        ierr = VecSetValue(xx,0,1.0,INSERT_VALUES);CHKERRQ(ierr);
      }
      ierr = VecAssemblyBegin(xx);CHKERRQ(ierr);
      ierr = VecAssemblyEnd(xx);CHKERRQ(ierr);
    }
    ierr = VecZeroEntries(bb);CHKERRQ(ierr);
    ierr = VecGetSize(bb,&i);CHKERRQ(ierr);
    local_sizes[iter] = i;
    ierr = PetscInfo2(snes,"%D equations in vector, %D vertices\n",i,i/dim);CHKERRQ(ierr);
    ierr = PetscLogStagePop();CHKERRQ(ierr);
    /* solve */
    ierr = PetscLogStagePush(stage[iter]);CHKERRQ(ierr);
    ierr = SNESSolve(snes, bb, xx);CHKERRQ(ierr);
    ierr = PetscLogStagePop();CHKERRQ(ierr);
    ierr = VecNorm(xx,NORM_INFINITY,&mdisp[iter]);CHKERRQ(ierr);
    ierr = DMViewFromOptions(dm, NULL, "-dm_view");CHKERRQ(ierr);
    {
      PetscViewer       viewer = NULL;
      PetscViewerFormat fmt;
      ierr = PetscOptionsGetViewer(comm,NULL,"ex56_","-vec_view",&viewer,&fmt,&flg);CHKERRQ(ierr);
      if (flg) {
        ierr = PetscViewerPushFormat(viewer,fmt);CHKERRQ(ierr);
        ierr = VecView(xx,viewer);CHKERRQ(ierr);
        ierr = VecView(bb,viewer);CHKERRQ(ierr);
        ierr = PetscViewerPopFormat(viewer);CHKERRQ(ierr);
      }
      ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
    }
    /* Free work space */
    ierr = DMDestroy(&dm);CHKERRQ(ierr);
    ierr = SNESDestroy(&snes);CHKERRQ(ierr);
    ierr = VecDestroy(&xx);CHKERRQ(ierr);
    ierr = VecDestroy(&bb);CHKERRQ(ierr);
    ierr = MatDestroy(&Amat);CHKERRQ(ierr);
  }
  ierr = DMDestroy(&basedm);CHKERRQ(ierr);
  if (run_type==1) err[0] = 59.975208 - mdisp[0]; /* error with what I think is the exact solution */
  else             err[0] = 171.038 - mdisp[0];
  for (iter=1 ; iter<max_conv_its ; iter++) {
    if (run_type==1) err[iter] = 59.975208 - mdisp[iter];
    else             err[iter] = 171.038 - mdisp[iter];
    ierr = PetscPrintf(PETSC_COMM_WORLD,"[%d] %D) N=%12D, max displ=%9.7e, disp diff=%9.2e, error=%4.3e, rate=%3.2g\n",rank,iter,local_sizes[iter],(double)mdisp[iter],
                       (double)(mdisp[iter]-mdisp[iter-1]),(double)err[iter],(double)(PetscLogReal(err[iter-1]/err[iter])/PetscLogReal(2.)));CHKERRQ(ierr);
  }

  ierr = PetscFinalize();
  return ierr;
}

/*TEST

  test:
    suffix: 0
    nsize: 4
    requires: !single
    args: -cells 2,2,1 -max_conv_its 2 -petscspace_degree 2 -snes_max_it 2 -ksp_max_it 100 -ksp_type cg -ksp_rtol 1.e-11 -ksp_norm_type unpreconditioned -snes_rtol 1.e-10 -pc_type gamg -pc_gamg_type agg -pc_gamg_agg_nsmooths 1 -pc_gamg_coarse_eq_limit 1000 -pc_gamg_reuse_interpolation true -pc_gamg_square_graph 1 -pc_gamg_threshold 0.05 -pc_gamg_threshold_scale .0 -ksp_converged_reason -snes_monitor_short -ksp_monitor_short -snes_converged_reason -use_mat_nearnullspace true -mg_levels_ksp_max_it 1 -mg_levels_ksp_type chebyshev -mg_levels_esteig_ksp_type cg -mg_levels_esteig_ksp_max_it 10 -mg_levels_ksp_chebyshev_esteig 0,0.05,0,1.05 -mg_levels_pc_type jacobi -petscpartitioner_type simple -mat_block_size 3 -matptap_via scalable -ex56_dm_view -run_type 1
    timeoutfactor: 2

  # HYPRE PtAP broken with complex numbers
  test:
    suffix: hypre
    requires: hypre !single !complex
    nsize: 4
    args: -cells 2,2,1 -max_conv_its 2 -lx 1. -alpha .01 -ex56_dm_refine 1 -petscspace_degree 2 -ksp_type cg -ksp_monitor_short -ksp_rtol 1.e-8 -pc_type hypre -pc_hypre_type boomeramg -pc_hypre_boomeramg_no_CF true -pc_hypre_boomeramg_agg_nl 1 -pc_hypre_boomeramg_coarsen_type HMIS -pc_hypre_boomeramg_interp_type ext+i -ksp_converged_reason -use_mat_nearnullspace true -mat_block_size 3 -petscpartitioner_type simple

  test:
    suffix: ml
    requires: ml !single
    nsize: 4
    args: -cells 2,2,1 -max_conv_its 2 -lx 1. -alpha .01 -ex56_dm_refine 1 -petscspace_degree 2 -ksp_type cg -ksp_monitor_short -ksp_converged_reason -ksp_rtol 1.e-8 -pc_type ml -mg_levels_ksp_type chebyshev -mg_levels_ksp_max_it 3 -mg_levels_esteig_ksp_max_it 10 -mg_levels_ksp_chebyshev_esteig 0,0.05,0,1.05 -mg_levels_pc_type sor -mg_levels_esteig_ksp_type cg -mat_block_size 3 -petscpartitioner_type simple -use_mat_nearnullspace

  test:
    nsize: 4
    requires: parmetis !single
    args: -cells 2,2,1 -max_conv_its 2 -petscspace_degree 2 -snes_max_it 2 -ksp_max_it 100 -ksp_type cg -ksp_rtol 1.e-11 -ksp_norm_type unpreconditioned -snes_rtol 1.e-10 -pc_type gamg -pc_gamg_type agg -pc_gamg_agg_nsmooths 1 -pc_gamg_coarse_eq_limit 10 -pc_gamg_reuse_interpolation true -pc_gamg_square_graph 1 -pc_gamg_threshold 0.05 -pc_gamg_threshold_scale .0 -snes_converged_reason -use_mat_nearnullspace true -mg_levels_ksp_max_it 1 -mg_levels_ksp_type chebyshev -mg_levels_esteig_ksp_type cg -mg_levels_esteig_ksp_max_it 10 -mg_levels_ksp_chebyshev_esteig 0,0.05,0,1.05 -mg_levels_pc_type jacobi -pc_gamg_mat_partitioning_type parmetis -mat_block_size 3 -run_type 1 -pc_gamg_repartition true

  test:
    suffix: bddc
    nsize: 4
    requires: !single
    args: -cells 2,2,1 -max_conv_its 2 -lx 1. -alpha .01 -ex56_dm_refine 1 -petscspace_degree 2 -ksp_type cg -ksp_monitor_short -ksp_rtol 1.e-8 -ksp_converged_reason -mat_block_size 3 -petscpartitioner_type simple -ex56_dm_mat_type is -matis_localmat_type {{sbaij baij aij}} -pc_type bddc

  testset:
    nsize: 4
    requires: !single
    args: -cells 2,2,1 -max_conv_its 2 -lx 1. -alpha .01 -ex56_dm_refine 1 -petscspace_degree 2 -ksp_type cg -ksp_monitor_short -ksp_rtol 1.e-10 -ksp_converged_reason -mat_block_size 3 -petscpartitioner_type simple -ex56_dm_mat_type is -matis_localmat_type aij -pc_type bddc -attach_mat_nearnullspace {{0 1}separate output}
    test:
      suffix: bddc_approx_gamg
      args: -pc_bddc_switch_static -prefix_push pc_bddc_dirichlet_ -approximate -pc_type gamg -pc_gamg_type agg -pc_gamg_agg_nsmooths 1 -pc_gamg_coarse_eq_limit 10 -pc_gamg_reuse_interpolation true -pc_gamg_square_graph 1 -pc_gamg_threshold 0.05 -pc_gamg_threshold_scale .0 -mg_levels_ksp_max_it 1 -mg_levels_ksp_type chebyshev -mg_levels_esteig_ksp_type cg -mg_levels_esteig_ksp_max_it 10 -prefix_pop -prefix_push pc_bddc_neumann_ -approximate -pc_type gamg -pc_gamg_type agg -pc_gamg_agg_nsmooths 1 -pc_gamg_coarse_eq_limit 10 -pc_gamg_reuse_interpolation true -pc_gamg_square_graph 1 -pc_gamg_threshold 0.05 -pc_gamg_threshold_scale .0 -mg_levels_ksp_max_it 1 -mg_levels_ksp_type chebyshev -mg_levels_esteig_ksp_type cg -mg_levels_esteig_ksp_max_it 10 -prefix_pop
    # HYPRE PtAP broken with complex numbers
    test:
      requires: hypre !complex
      suffix: bddc_approx_hypre
      args: -pc_bddc_switch_static -prefix_push pc_bddc_dirichlet_ -pc_type hypre -pc_hypre_boomeramg_no_CF true -pc_hypre_boomeramg_strong_threshold 0.75 -pc_hypre_boomeramg_agg_nl 1 -pc_hypre_boomeramg_coarsen_type HMIS -pc_hypre_boomeramg_interp_type ext+i -prefix_pop -prefix_push pc_bddc_neumann_ -pc_type hypre -pc_hypre_boomeramg_no_CF true -pc_hypre_boomeramg_strong_threshold 0.75 -pc_hypre_boomeramg_agg_nl 1 -pc_hypre_boomeramg_coarsen_type HMIS -pc_hypre_boomeramg_interp_type ext+i -prefix_pop
    test:
      requires: ml
      suffix: bddc_approx_ml
      args: -pc_bddc_switch_static -prefix_push pc_bddc_dirichlet_ -approximate -pc_type ml -mg_levels_ksp_max_it 1 -mg_levels_ksp_type chebyshev -mg_levels_esteig_ksp_type cg -mg_levels_esteig_ksp_max_it 10 -prefix_pop -prefix_push pc_bddc_neumann_ -approximate -pc_type ml -mg_levels_ksp_max_it 1 -mg_levels_ksp_type chebyshev -mg_levels_esteig_ksp_type cg -mg_levels_esteig_ksp_max_it 10 -prefix_pop

  test:
    suffix: fetidp
    nsize: 4
    requires: !single
    args: -cells 2,2,1 -max_conv_its 2 -lx 1. -alpha .01 -ex56_dm_refine 1 -petscspace_degree 2 -ksp_type fetidp -fetidp_ksp_type cg -ksp_monitor_short -ksp_rtol 1.e-8 -ksp_converged_reason -mat_block_size 3 -petscpartitioner_type simple -ex56_dm_mat_type is -matis_localmat_type {{sbaij baij aij}}

  test:
    suffix: bddc_elast
    nsize: 4
    requires: !single
    args: -cells 2,2,1 -max_conv_its 2 -lx 1. -alpha .01 -ex56_dm_refine 1 -petscspace_degree 2 -ksp_type cg -ksp_monitor_short -ksp_rtol 1.e-8 -ksp_converged_reason -mat_block_size 3 -petscpartitioner_type simple -ex56_dm_mat_type is -matis_localmat_type sbaij -pc_type bddc -pc_bddc_monolithic -attach_mat_nearnullspace

  test:
    suffix: fetidp_elast
    nsize: 4
    requires: !single
    args: -cells 2,2,1 -max_conv_its 2 -lx 1. -alpha .01 -ex56_dm_refine 1 -petscspace_degree 2 -ksp_type fetidp -fetidp_ksp_type cg -ksp_monitor_short -ksp_rtol 1.e-8 -ksp_converged_reason -mat_block_size 3 -petscpartitioner_type simple -ex56_dm_mat_type is -matis_localmat_type sbaij -fetidp_bddc_pc_bddc_monolithic -attach_mat_nearnullspace

  test:
    suffix: cuda
    nsize: 2
    requires: cuda !single
    args: -cells 2,2,1 -max_conv_its 2 -petscspace_degree 2 -snes_max_it 2 -ksp_max_it 100 -ksp_type cg -ksp_rtol 1.e-11 -ksp_norm_type unpreconditioned -snes_rtol 1.e-10 -pc_type gamg -pc_gamg_type agg -pc_gamg_agg_nsmooths 1 -pc_gamg_coarse_eq_limit 10 -pc_gamg_reuse_interpolation true -pc_gamg_square_graph 1 -pc_gamg_threshold 0.05 -pc_gamg_threshold_scale .0 -snes_converged_reason -use_mat_nearnullspace true -mg_levels_ksp_max_it 1 -mg_levels_ksp_type chebyshev -mg_levels_esteig_ksp_type cg -mg_levels_esteig_ksp_max_it 10 -mg_levels_ksp_chebyshev_esteig 0,0.05,0,1.05 -mg_levels_pc_type jacobi -mat_block_size 3 -run_type 1 -ex56_dm_mat_type aijcusparse -ex56_dm_vec_type cuda -ksp_monitor_short

TEST*/
