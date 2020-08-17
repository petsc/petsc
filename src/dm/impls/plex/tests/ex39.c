const char help[] =
  "A test of H-div conforming discretizations on different cell types.\n";

#include <petscdmplex.h>
#include <petscds.h>
#include <petscsnes.h>
#include <petscconvest.h>
#include <petscfe.h>
#include <petsc/private/petscfeimpl.h>

/*
* We are using the system
*
* \vec{u} = \vec{\hat{u}}
* p = \div{\vec{u}} in low degree approximation space
* d = \div{\vec{u}} - p == 0 in higher degree approximation space
*
* That is, we are using the field d to compute the error between \div{\vec{u}}
* computed in a space 1 degree higher than p and the value of p which is
* \div{u} computed in the low degree space. If H-div
* elements are implemented correctly then this should be identically zero since
* the divergence of a function in H(div) should be exactly representable in L_2
* by definition.
*/
static PetscErrorCode zero_func(PetscInt dim,PetscReal time,const PetscReal x[],PetscInt Nc,PetscScalar *u,void *ctx)
{
  PetscInt c;
  for (c = 0; c < Nc; ++c) u[c] = 0;
  return 0;
}
/* Linear Exact Functions
   \vec{u} = \vec{x};
   p = dim;
   */
static PetscErrorCode linear_u(PetscInt dim,PetscReal time,const PetscReal x[],PetscInt Nc,PetscScalar *u,void *ctx)
{
  PetscInt c;
  for (c = 0; c < Nc; ++c) u[c] = x[c];
  return 0;
}
static PetscErrorCode linear_p(PetscInt dim,PetscReal time,const PetscReal x[],PetscInt Nc,PetscScalar *u,void *ctx)
{
  u[0] = dim;
  return 0;
}

/* Sinusoidal Exact Functions
 * u_i = \sin{2*\pi*x_i}
 * p = \Sum_{i=1}^{dim} 2*\pi*cos{2*\pi*x_i}
 * */

static PetscErrorCode sinusoid_u(PetscInt dim,PetscReal time,const PetscReal
                                 x[],PetscInt Nc,PetscScalar *u,void *ctx)
{
  PetscInt c;
  for (c = 0; c< Nc; ++c) u[c] = PetscSinReal(2*PETSC_PI*x[c]);
  return 0;
}
static PetscErrorCode sinusoid_p(PetscInt dim,PetscReal time,const PetscReal
                                 x[],PetscInt Nc,PetscScalar *u,void *ctx)
{
  PetscInt d;
  u[0]=0;
  for (d=0; d<dim; ++d) u[0] += 2*PETSC_PI*PetscCosReal(2*PETSC_PI*x[d]);
  return 0;
}

/* Pointwise residual for u = u*. Need one of these for each possible u* */
static void f0_v_linear(PetscInt dim,PetscInt Nf,PetscInt NfAux,const PetscInt uOff[],const PetscInt uOff_x[],const PetscScalar u[],const PetscScalar u_t[],const PetscScalar u_x[],const PetscInt aOff[],const PetscInt aOff_x[],const PetscScalar a[],const PetscScalar a_t[],const PetscScalar a_x[],PetscReal t,const PetscReal x[],PetscInt numConstants,const PetscScalar constants[],PetscScalar f0[])
{
  PetscInt    i;
  PetscScalar *u_rhs;

  PetscCalloc1(dim,&u_rhs);
  (void) linear_u(dim,t,x,dim,u_rhs,NULL);
  for (i = 0; i < dim; ++i) f0[i] = u[uOff[0]+i]-u_rhs[i];
  PetscFree(u_rhs);
}

static void f0_v_sinusoid(PetscInt dim,PetscInt Nf,PetscInt NfAux,const PetscInt uOff[],const PetscInt uOff_x[],const PetscScalar u[],const PetscScalar u_t[],const PetscScalar u_x[],const PetscInt aOff[],const PetscInt aOff_x[],const PetscScalar a[],const PetscScalar a_t[],const PetscScalar a_x[],PetscReal t,const PetscReal x[],PetscInt numConstants,const PetscScalar constants[],PetscScalar f0[])
{
  PetscInt    i;
  PetscScalar *u_rhs;

  PetscCalloc1(dim,&u_rhs);
  (void) sinusoid_u(dim,t,x,dim,u_rhs,NULL);
  for (i = 0; i < dim; ++i) f0[i] = u[uOff[0]+i]-u_rhs[i];
  PetscFree(u_rhs);
}

/* Residual function for enforcing p = \div{u}. */
static void f0_q(PetscInt dim,PetscInt Nf,PetscInt NfAux,const PetscInt uOff[],const PetscInt uOff_x[],const PetscScalar u[],const PetscScalar u_t[],const PetscScalar u_x[],const PetscInt aOff[],const PetscInt aOff_x[],const PetscScalar a[],const PetscScalar a_t[],const PetscScalar a_x[],PetscReal t,const PetscReal x[],PetscInt numConstants,const PetscScalar constants[],PetscScalar f0[])
{
  PetscInt    i;
  PetscScalar divu;

  divu = 0.;
  for (i = 0; i< dim; ++i) divu += u_x[uOff_x[0]+i*dim+i];
  f0[0] = u[uOff[1]] - divu;
}

/* Residual function for p_err = \div{u} - p. */
static void f0_w(PetscInt dim,PetscInt Nf,PetscInt NfAux,const PetscInt uOff[],const PetscInt uOff_x[],const PetscScalar u[],const PetscScalar u_t[],const PetscScalar u_x[],const PetscInt aOff[],const PetscInt aOff_x[],const PetscScalar a[],const PetscScalar a_t[],const PetscScalar a_x[],PetscReal t,const PetscReal x[],PetscInt numConstants,const PetscScalar constants[],PetscScalar f0[])
{
  PetscInt    i;
  PetscScalar divu;

  divu = 0.;
  for (i = 0; i < dim; ++i) divu += u_x[uOff_x[0] + i*dim +i];
  f0[0] = u[uOff[2]] - u[uOff[1]] + divu;
}

/* Boundary residual for the embedding system. Need one for each form of
 * solution. These enforce u = \hat{u} at the boundary. */
static void f0_bd_u_sinusoid(PetscInt dim,PetscInt Nf,PetscInt NfAux,const PetscInt uOff[],const PetscInt uOff_x[],const PetscScalar u[],const PetscScalar u_t[],const PetscScalar u_x[],const PetscInt aOff[],const PetscInt aOff_x[],const PetscScalar a[],const PetscScalar a_t[],const PetscScalar a_x[],PetscReal t,const PetscReal x[],const PetscReal n[],PetscInt numConstants,const PetscScalar constants[],PetscScalar f0[])
{
  PetscInt    d;
  PetscScalar *u_rhs;
  PetscCalloc1(dim,&u_rhs);
  (void) sinusoid_u(dim,t,x,dim,u_rhs,NULL);

  for (d=0; d<dim; ++d) f0[d] = u_rhs[d];
  PetscFree(u_rhs);

}

static void f0_bd_u_linear(PetscInt dim,PetscInt Nf,PetscInt NfAux,const PetscInt uOff[],const PetscInt uOff_x[],const PetscScalar u[],const PetscScalar u_t[],const PetscScalar u_x[],const PetscInt aOff[],const PetscInt aOff_x[],const PetscScalar a[],const PetscScalar a_t[],const PetscScalar a_x[],PetscReal t,const PetscReal x[],const PetscReal n[],PetscInt numConstants,const PetscScalar constants[],PetscScalar f0[])
{
  PetscInt    d;
  PetscScalar *u_rhs;
  PetscCalloc1(dim,&u_rhs);
  (void) linear_u(dim,t,x,dim,u_rhs,NULL);

  for (d=0; d<dim; ++d) f0[d] = u_rhs[d];
  PetscFree(u_rhs);
}
/* Jacobian functions. For the following, v is the test function associated with
 * u, q the test function associated with p, and w the test function associated
 * with d. */
/* <v, u> */
static void g0_vu(PetscInt dim,PetscInt Nf,PetscInt NfAux,const PetscInt uOff[],const PetscInt uOff_x[],const PetscScalar u[],const PetscScalar u_t[],const PetscScalar u_x[],const PetscInt aOff[],const PetscInt aOff_x[],const PetscScalar a[],const PetscScalar a_t[],const PetscScalar a_x[],PetscReal t,PetscReal u_tShift,const PetscReal x[],PetscInt numConstants,const PetscScalar constants[],PetscScalar g0[])
{
  PetscInt c;
  for (c = 0; c < dim; ++c) g0[c * dim + c] = 1.0;
}

/* <q, p> */
static void g0_qp(PetscInt dim,PetscInt Nf,PetscInt NfAux,const PetscInt uOff[],const PetscInt uOff_x[],const PetscScalar u[],const PetscScalar u_t[],const PetscScalar u_x[],const PetscInt aOff[],const PetscInt aOff_x[],const PetscScalar a[],const PetscScalar a_t[],const PetscScalar a_x[],PetscReal t,PetscReal u_tShift,const PetscReal x[],PetscInt numConstants,const PetscScalar constants[],PetscScalar g0[])
{
  PetscInt d;
  for (d=0; d< dim; ++d) g0[d * dim + d] = 1.0;
}

/* -<q, \div{u}> For the embedded system. This is different from the method of
 * manufactured solution because instead of computing <q,\div{u}> - <q,f> we
 * need <q,p> - <q,\div{u}.*/
static void g1_qu(PetscInt dim,PetscInt Nf,PetscInt NfAux,const PetscInt uOff[],const PetscInt uOff_x[],const PetscScalar u[],const PetscScalar u_t[],const PetscScalar u_x[],const PetscInt aOff[],const PetscInt aOff_x[],const PetscScalar a[],const PetscScalar a_t[],const PetscScalar a_x[],PetscReal t,PetscReal u_tShift,const PetscReal x[],PetscInt numConstants,const PetscScalar constants[],PetscScalar g1[])
{
  PetscInt d;
  for (d = 0; d < dim; ++d) g1[d * dim + d] = -1.0;
}

/* <w, p> This is only used by the embedded system. Where we need to compute
 * <w,d> - <w,p> + <w, \div{u}>*/
static void g0_wp(PetscInt dim,PetscInt Nf,PetscInt NfAux,const PetscInt uOff[],const PetscInt uOff_x[],const PetscScalar u[],const PetscScalar u_t[],const PetscScalar u_x[],const PetscInt aOff[],const PetscInt aOff_x[],const PetscScalar a[],const PetscScalar a_t[],const PetscScalar a_x[],PetscReal t,PetscReal u_tShift,const PetscReal x[],PetscInt numConstants,const PetscScalar constants[],PetscScalar g0[])
{
  PetscInt d;
  for (d=0; d< dim; ++d) g0[d * dim + d] = -1.0;
}

/* <w, d> */
static void g0_wd(PetscInt dim,PetscInt Nf,PetscInt NfAux,const PetscInt uOff[],const PetscInt uOff_x[],const PetscScalar u[],const PetscScalar u_t[],const PetscScalar u_x[],const PetscInt aOff[],const PetscInt aOff_x[],const PetscScalar a[],const PetscScalar a_t[],const PetscScalar a_x[],PetscReal t,PetscReal u_tShift,const PetscReal x[],PetscInt numConstants,const PetscScalar constants[],PetscScalar g0[])
{
  PetscInt c;
  for (c = 0; c < dim; ++c) g0[c*dim+c] = 1.0;
}

/* <w, \div{u}> for the embedded system. */
static void g1_wu(PetscInt dim,PetscInt Nf,PetscInt NfAux,const PetscInt uOff[],const PetscInt uOff_x[],const PetscScalar u[],const PetscScalar u_t[],const PetscScalar u_x[],const PetscInt aOff[],const PetscInt aOff_x[],const PetscScalar a[],const PetscScalar a_t[],const PetscScalar a_x[],PetscReal t,PetscReal u_tShift,const PetscReal x[],PetscInt numConstants,const PetscScalar constants[],PetscScalar g1[])
{
  PetscInt d;
  for (d = 0; d < dim; ++d) g1[d * dim + d] = 1.0;
}

/* Enum and string array for selecting mesh perturbation options */
typedef enum { NONE = 0,PERTURB = 1,SKEW = 2,SKEW_PERTURB = 3 } Transform;
const char* const TransformTypes[] = {"none","perturb","skew","skew_perturb","Perturbation","",NULL};

/* Enum and string array for selecting the form of the exact solution*/
typedef enum
{ LINEAR = 0,SINUSOIDAL = 1 } Solution;
const char* const SolutionTypes[] = {"linear","sinusoidal","Solution","",NULL};

typedef struct
{
  PetscBool simplex;
  PetscInt  dim;
  Transform mesh_transform;
  Solution  sol_form;
} UserCtx;

/* Process command line options and initialize the UserCtx struct */
static PetscErrorCode ProcessOptions(MPI_Comm comm,UserCtx * user)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* Default to  2D, unperturbed triangle mesh and Linear solution.*/
  user->simplex        = PETSC_TRUE;
  user->dim            = 2;
  user->mesh_transform = NONE;
  user->sol_form       = LINEAR;

  ierr = PetscOptionsBegin(comm,"","H-div Test Options","DMPLEX");CHKERRQ(ierr);
  ierr = PetscOptionsBool("-simplex","Whether to use simplices (true) or tensor-product (false) cells in " "the mesh","ex39.c",user->simplex,&user->simplex,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-dim","Number of solution dimensions","ex39.c",user->dim,&user->dim,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnum("-mesh_transform","Method used to perturb the mesh vertices. Options are skew, perturb, skew_perturb,or none","ex39.c",TransformTypes,(PetscEnum) user->mesh_transform,(PetscEnum*) &user->mesh_transform,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnum("-sol_form","Form of the exact solution. Options are Linear or Sinusoidal","ex39.c",SolutionTypes,(PetscEnum) user->sol_form,(PetscEnum*) &user->sol_form,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* Perturb the position of each mesh vertex by a small amount.*/
static PetscErrorCode PerturbMesh(DM *mesh,PetscScalar *coordVals,PetscInt npoints,PetscInt dim)
{
  PetscInt       i,j,k;
  PetscErrorCode ierr;
  PetscReal      minCoords[3],maxCoords[3],maxPert[3],randVal,amp;
  PetscRandom    ran;

  PetscFunctionBegin;
  ierr = DMGetCoordinateDim(*mesh,&dim);CHKERRQ(ierr);
  ierr = DMGetLocalBoundingBox(*mesh,minCoords,maxCoords);CHKERRQ(ierr);
  ierr = PetscRandomCreate(PETSC_COMM_WORLD,&ran);CHKERRQ(ierr);

  /* Compute something approximately equal to half an edge length. This is the
   * most we can perturb points and gaurantee that there won't be any topology
   * issues. */
  for (k = 0; k < dim; ++k) maxPert[k] = 0.025 * (maxCoords[k] - minCoords[k]) / (PetscPowReal(npoints,1. / dim) - 1);
  /* For each mesh vertex */
  for (i = 0; i < npoints; ++i) {
    /* For each coordinate of the vertex */
    for (j = 0; j < dim; ++j) {
      /* Generate a random amplitude in [-0.5*maxPert, 0.5*maxPert] */
      ierr = PetscRandomGetValueReal(ran,&randVal);CHKERRQ(ierr);
      amp  = maxPert[j] * (randVal - 0.5);
      /* Add the perturbation to the vertex*/
      coordVals[dim * i + j] += amp;
    }
  }

  PetscRandomDestroy(&ran);
  PetscFunctionReturn(0);
}

/* Apply a global skew transformation to the mesh. */
static PetscErrorCode SkewMesh(DM * mesh,PetscScalar * coordVals,PetscInt npoints,PetscInt dim)
{
  PetscInt       i,j,k,l;
  PetscErrorCode ierr;
  PetscScalar    * transMat;
  PetscScalar    tmpcoord[3];
  PetscRandom    ran;
  PetscReal      randVal;

  PetscFunctionBegin;
  ierr = PetscCalloc1(dim * dim,&transMat);CHKERRQ(ierr);
  ierr = PetscRandomCreate(PETSC_COMM_WORLD,&ran);CHKERRQ(ierr);

  /* Make a matrix representing a skew transformation */
  for (i = 0; i < dim; ++i) {
    for (j = 0; j < dim; ++j) {
      ierr = PetscRandomGetValueReal(ran,&randVal);CHKERRQ(ierr);
      if (i == j) transMat[i * dim + j] = 1.;
      else if (j < i) transMat[i * dim + j] = 2 * (j + i)*randVal;
      else transMat[i * dim + j] = 0;
    }
  }

  /* Multiply each coordinate vector by our tranformation.*/
  for (i = 0; i < npoints; ++i) {
    for (j = 0; j < dim; ++j) {
      tmpcoord[j] = 0;
      for (k = 0; k < dim; ++k) tmpcoord[j] += coordVals[dim * i + k] * transMat[dim * k + j];
    }
    for (l = 0; l < dim; ++l) coordVals[dim * i + l] = tmpcoord[l];
  }
  ierr = PetscFree(transMat);CHKERRQ(ierr);
  ierr = PetscRandomDestroy(&ran);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* Accesses the mesh coordinate array and performs the transformation operations
 * specified by the user options */
static PetscErrorCode TransformMesh(UserCtx * user,DM * mesh)
{
  PetscErrorCode ierr;
  PetscInt       dim,npoints;
  PetscScalar    * coordVals;
  Vec            coords;

  PetscFunctionBegin;
  ierr    = DMGetCoordinates(*mesh,&coords);CHKERRQ(ierr);
  ierr    = VecGetArray(coords,&coordVals);CHKERRQ(ierr);
  ierr    = VecGetLocalSize(coords,&npoints);CHKERRQ(ierr);
  ierr    = DMGetCoordinateDim(*mesh,&dim);CHKERRQ(ierr);
  npoints = npoints / dim;

  switch (user->mesh_transform) {
  case PERTURB:
    ierr = PerturbMesh(mesh,coordVals,npoints,dim);CHKERRQ(ierr);
    break;
  case SKEW:
    ierr = SkewMesh(mesh,coordVals,npoints,dim);CHKERRQ(ierr);
    break;
  case SKEW_PERTURB:
    ierr = SkewMesh(mesh,coordVals,npoints,dim);CHKERRQ(ierr);
    ierr = PerturbMesh(mesh,coordVals,npoints,dim);CHKERRQ(ierr);
    break;
  default:
    PetscFunctionReturn(-1);
  }
  ierr = VecRestoreArray(coords,&coordVals);CHKERRQ(ierr);
  ierr = DMSetCoordinates(*mesh,coords);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode CreateMesh(MPI_Comm comm,UserCtx * user,DM * mesh)
{
  PetscErrorCode   ierr;
  DMLabel          label;
  const char       *name  = "marker";
  DM               dmDist = NULL;
  PetscPartitioner part;

  PetscFunctionBegin;
  /* Create box mesh from user parameters */
  ierr = DMPlexCreateBoxMesh(comm,user->dim,user->simplex,NULL,NULL,NULL,NULL,PETSC_TRUE,mesh);CHKERRQ(ierr);

  /* Make sure the mesh gets properly distributed if running in parallel */
  ierr = DMPlexGetPartitioner(*mesh,&part);CHKERRQ(ierr);
  ierr = PetscPartitionerSetFromOptions(part);CHKERRQ(ierr);
  ierr = DMPlexDistribute(*mesh,0,NULL,&dmDist);CHKERRQ(ierr);
  if (dmDist) {
    ierr  = DMDestroy(mesh);CHKERRQ(ierr);
    *mesh = dmDist;
  }

  /* Mark the boundaries, we will need this later when setting up the system of
   * equations */
  ierr = DMCreateLabel(*mesh,name);CHKERRQ(ierr);
  ierr = DMGetLabel(*mesh,name,&label);CHKERRQ(ierr);
  ierr = DMPlexMarkBoundaryFaces(*mesh,1,label);CHKERRQ(ierr);
  ierr = DMPlexLabelComplete(*mesh,label);CHKERRQ(ierr);
  ierr = DMLocalizeCoordinates(*mesh);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) *mesh,"Mesh");CHKERRQ(ierr);

  /* Perform any mesh transformations if specified by user */
  if (user->mesh_transform != NONE) {
    ierr = TransformMesh(user,mesh);CHKERRQ(ierr);
  }

  /* Get any other mesh options from the command line */
  ierr = DMSetApplicationContext(*mesh,user);CHKERRQ(ierr);
  ierr = DMSetFromOptions(*mesh);CHKERRQ(ierr);
  ierr = DMViewFromOptions(*mesh,NULL,"-dm_view");CHKERRQ(ierr);

  ierr = DMDestroy(&dmDist);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* Setup the system of equations that we wish to solve */
static PetscErrorCode SetupProblem(DM dm,UserCtx * user)
{
  PetscDS        prob;
  PetscErrorCode ierr;
  const PetscInt id=1;

  PetscFunctionBegin;
  ierr = DMGetDS(dm,&prob);CHKERRQ(ierr);
  /* All of these are independent of the user's choice of solution */
  ierr = PetscDSSetResidual(prob,1,f0_q,NULL);CHKERRQ(ierr);
  ierr = PetscDSSetResidual(prob,2,f0_w,NULL);CHKERRQ(ierr);
  ierr = PetscDSSetJacobian(prob,0,0,g0_vu,NULL,NULL,NULL);CHKERRQ(ierr);
  ierr = PetscDSSetJacobian(prob,1,0,NULL,g1_qu,NULL,NULL);CHKERRQ(ierr);
  ierr = PetscDSSetJacobian(prob,1,1,g0_qp,NULL,NULL,NULL);CHKERRQ(ierr);
  ierr = PetscDSSetJacobian(prob,2,0,NULL,g1_wu,NULL,NULL);CHKERRQ(ierr);
  ierr = PetscDSSetJacobian(prob,2,1,g0_wp,NULL,NULL,NULL);CHKERRQ(ierr);
  ierr = PetscDSSetJacobian(prob,2,2,g0_wd,NULL,NULL,NULL);CHKERRQ(ierr);

  /* Field 2 is the error between \div{u} and pressure in a higher dimensional
   * space. If all is right this should be machine zero. */
  ierr = PetscDSSetExactSolution(prob,2,zero_func,NULL);CHKERRQ(ierr);

  switch (user->sol_form) {
  case LINEAR:
    ierr = PetscDSSetResidual(prob,0,f0_v_linear,NULL);CHKERRQ(ierr);
    ierr = PetscDSSetBdResidual(prob,0,f0_bd_u_linear,NULL);CHKERRQ(ierr);
    ierr = PetscDSSetExactSolution(prob,0,linear_u,NULL);CHKERRQ(ierr);
    ierr = PetscDSSetExactSolution(prob,1,linear_p,NULL);CHKERRQ(ierr);
    break;
  case SINUSOIDAL:
    ierr = PetscDSSetResidual(prob,0,f0_v_sinusoid,NULL);CHKERRQ(ierr);
    ierr = PetscDSSetBdResidual(prob,0,f0_bd_u_sinusoid,NULL);CHKERRQ(ierr);
    ierr = PetscDSSetExactSolution(prob,0,sinusoid_u,NULL);CHKERRQ(ierr);
    ierr = PetscDSSetExactSolution(prob,1,sinusoid_p,NULL);CHKERRQ(ierr);
    break;
  default:
    PetscFunctionReturn(-1);
  }

  ierr = PetscDSAddBoundary(prob,DM_BC_NATURAL,"Boundary Integral","marker",0,0,NULL,(void (*)(void))NULL,NULL,1,&id,user);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* Create the finite element spaces we will use for this system */
static PetscErrorCode SetupDiscretization(DM mesh,PetscErrorCode (*setup)(DM,UserCtx*),UserCtx *user)
{
  DM             cdm = mesh;
  PetscFE        fevel,fepres,fedivErr;
  const PetscInt dim = user->dim;
  PetscErrorCode ierr;


  PetscFunctionBegin;
  /* Create FE objects and give them names so that options can be set from
   * command line */
  ierr = PetscFECreateDefault(PetscObjectComm((PetscObject) mesh),dim,dim,user->simplex,"velocity_",-1,&fevel);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) fevel,"velocity");CHKERRQ(ierr);

  ierr = PetscFECreateDefault(PetscObjectComm((PetscObject) mesh),dim,1,user->simplex,"pressure_",-1,&fepres);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) fepres,"pressure");CHKERRQ(ierr);

  ierr = PetscFECreateDefault(PetscObjectComm((PetscObject)
                                              mesh),dim,1,user->simplex,"divErr_",-1,&fedivErr);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) fedivErr,"divErr");CHKERRQ(ierr);

  ierr = PetscFECopyQuadrature(fevel,fepres);CHKERRQ(ierr);
  ierr = PetscFECopyQuadrature(fevel,fedivErr);CHKERRQ(ierr);

  /* Associate the FE objects with the mesh and setup the system */
  ierr = DMSetField(mesh,0,NULL,(PetscObject) fevel);CHKERRQ(ierr);
  ierr = DMSetField(mesh,1,NULL,(PetscObject) fepres);CHKERRQ(ierr);
  ierr = DMSetField(mesh,2,NULL,(PetscObject) fedivErr);CHKERRQ(ierr);
  ierr = DMCreateDS(mesh);CHKERRQ(ierr);
  ierr = (*setup)(mesh,user);CHKERRQ(ierr);

  while (cdm) {
    ierr = DMCopyDisc(mesh,cdm);CHKERRQ(ierr);
    ierr = DMGetCoarseDM(cdm,&cdm);CHKERRQ(ierr);
  }

  /* The Mesh now owns the fields, so we can destroy the FEs created in this
   * function */
  ierr = PetscFEDestroy(&fevel);CHKERRQ(ierr);
  ierr = PetscFEDestroy(&fepres);CHKERRQ(ierr);
  ierr = PetscFEDestroy(&fedivErr);CHKERRQ(ierr);
  ierr = DMDestroy(&cdm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}



int main(int argc,char **argv)
{
  PetscInt        i;
  UserCtx         user;
  DM              mesh;
  SNES            snes;
  Vec             computed,divErr;
  PetscReal       divErrNorm;
  PetscErrorCode  ierr;
  IS              * fieldIS;
  PetscBool       exampleSuccess = PETSC_FALSE;
  const PetscReal errTol         = 10. * PETSC_SMALL;

  char stdFormat[] = "L2 Norm of the Divergence Error is: %g\n H(div) elements working correctly: %s\n";

  /* Initialize PETSc */
  ierr = PetscInitialize(&argc,&argv,NULL,help);if (ierr) return ierr;
  ierr = ProcessOptions(PETSC_COMM_WORLD,&user);CHKERRQ(ierr);

  /* Set up the system, we need to create a solver and a mesh and then assign
   * the correct spaces into the mesh*/
  ierr = SNESCreate(PETSC_COMM_WORLD,&snes);CHKERRQ(ierr);
  ierr = CreateMesh(PETSC_COMM_WORLD,&user,&mesh);CHKERRQ(ierr);
  ierr = SNESSetDM(snes,mesh);CHKERRQ(ierr);
  ierr = SetupDiscretization(mesh,SetupProblem,&user);CHKERRQ(ierr);
  ierr = DMPlexSetSNESLocalFEM(mesh,&user,&user,&user);CHKERRQ(ierr);
  ierr = SNESSetFromOptions(snes);CHKERRQ(ierr);

  /* Grab field IS so that we can view the solution by field */
  ierr = DMCreateFieldIS(mesh,NULL,NULL,&fieldIS);CHKERRQ(ierr);

  /* Create a vector to store the SNES solution, solve the system and grab the
   * solution from SNES */
  ierr = DMCreateGlobalVector(mesh,&computed);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) computed,"computedSolution");CHKERRQ(ierr);
  ierr = VecSet(computed,0.0);CHKERRQ(ierr);
  ierr = SNESSolve(snes,NULL,computed);CHKERRQ(ierr);
  ierr = SNESGetSolution(snes,&computed);CHKERRQ(ierr);
  ierr = VecViewFromOptions(computed,NULL,"-computedSolution_view");CHKERRQ(ierr);

  /* Now we pull out the portion of the vector corresponding to the 3rd field
   * which is the error between \div{u} computed in a higher dimensional space
   * and p = \div{u} computed in a low dimension space. We report the L2 norm of
   * this vector which should be zero if the H(div) spaces are implemented
   * correctly. */
  ierr           = VecGetSubVector(computed,fieldIS[2],&divErr);CHKERRQ(ierr);
  ierr           = VecNorm(divErr,NORM_2,&divErrNorm);CHKERRQ(ierr);
  ierr           = VecRestoreSubVector(computed,fieldIS[2],&divErr);CHKERRQ(ierr);
  exampleSuccess = (PetscBool)(divErrNorm <= errTol);


  ierr = PetscPrintf(PETSC_COMM_WORLD,stdFormat,divErrNorm,exampleSuccess ? "true" : "false");CHKERRQ(ierr);


  /* Tear down */
  ierr = VecDestroy(&divErr);CHKERRQ(ierr);
  ierr = VecDestroy(&computed);CHKERRQ(ierr);
  for (i = 0; i < 3; ++i) {
    ierr = ISDestroy(&fieldIS[i]);CHKERRQ(ierr);
  }
  ierr = PetscFree(fieldIS);CHKERRQ(ierr);
  ierr = SNESDestroy(&snes);CHKERRQ(ierr);
  ierr = DMDestroy(&mesh);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST
  testset:
    suffix: 2d_bdm
    requires: triangle
    args: -dim 2 \
      -simplex true \
      -velocity_petscfe_default_quadrature_order 1 \
      -velocity_petscspace_degree 1 \
      -velocity_petscdualspace_type bdm \
      -divErr_petscspace_degree 1 \
      -divErr_petscdualspace_lagrange_continuity false \
      -dm_refine 0 \
      -snes_error_if_not_converged \
      -ksp_rtol 1e-10 \
      -ksp_error_if_not_converged \
      -pc_type fieldsplit\
      -pc_fieldsplit_detect_saddle_point\
      -pc_fieldsplit_type schur\
      -pc_fieldsplit_schur_precondition full
    test:
      suffix: linear
      args: -sol_form linear -mesh_transform none
    test:
      suffix: sinusoidal
      args: -sol_form sinusoidal -mesh_transform none
    test:
      suffix: sinusoidal_skew
      args: -sol_form sinusoidal -mesh_transform skew
    test:
      suffix: sinusoidal_perturb
      args: -sol_form sinusoidal -mesh_transform perturb
    test:
      suffix: sinusoidal_skew_perturb
      args: -sol_form sinusoidal -mesh_transform skew_perturb

  testset:
    TODO: broken
    suffix: 2d_bdmq
    args: -dim 2 \
      -simplex false \
      -velocity_petscspace_degree 1 \
      -velocity_petscdualspace_type bdm \
      -velocity_petscdualspace_lagrange_tensor 1 \
      -divErr_petscspace_degree 1 \
      -divErr_petscdualspace_lagrange_continuity false \
      -dm_refine 0 \
      -snes_error_if_not_converged \
      -ksp_rtol 1e-10 \
      -ksp_error_if_not_converged \
      -pc_type fieldsplit\
      -pc_fieldsplit_detect_saddle_point\
      -pc_fieldsplit_type schur\
      -pc_fieldsplit_schur_precondition full
    test:
      suffix: linear
      args: -sol_form linear -mesh_transform none
    test:
      suffix: sinusoidal
      args: -sol_form sinusoidal -mesh_transform none
    test:
      suffix: sinusoidal_skew
      args: -sol_form sinusoidal -mesh_transform skew
    test:
      suffix: sinusoidal_perturb
      args: -sol_form sinusoidal -mesh_transform perturb
    test:
      suffix: sinusoidal_skew_perturb
      args: -sol_form sinusoidal -mesh_transform skew_perturb

  testset:
    suffix: 3d_bdm
    requires: ctetgen
    args: -dim 3 \
      -simplex true \
      -velocity_petscspace_degree 1 \
      -velocity_petscdualspace_type bdm \
      -divErr_petscspace_degree 1 \
      -divErr_petscdualspace_lagrange_continuity false \
      -dm_refine 0 \
      -snes_error_if_not_converged \
      -ksp_rtol 1e-10 \
      -ksp_error_if_not_converged \
      -pc_type fieldsplit \
      -pc_fieldsplit_detect_saddle_point \
      -pc_fieldsplit_type schur \
      -pc_fieldsplit_schur_precondition full
    test:
      suffix: linear
      args: -sol_form linear -mesh_transform none
    test:
      suffix: sinusoidal
      args: -sol_form sinusoidal -mesh_transform none
    test:
      suffix: sinusoidal_skew
      args: -sol_form sinusoidal -mesh_transform skew
    test:
      suffix: sinusoidal_perturb
      args: -sol_form sinusoidal -mesh_transform perturb
    test:
      suffix: sinusoidal_skew_perturb
      args: -sol_form sinusoidal -mesh_transform skew_perturb

  testset:
    TODO: broken
    suffix: 3d_bdmq
    requires: ctetgen
    args: -dim 3 \
      -simplex false \
      -velocity_petscspace_degree 1 \
      -velocity_petscdualspace_type bdm \
      -velocity_petscdualspace_lagrange_tensor 1 \
      -divErr_petscspace_degree 1 \
      -divErr_petscdualspace_lagrange_continuity false \
      -dm_refine 0 \
      -snes_error_if_not_converged \
      -ksp_rtol 1e-10 \
      -ksp_error_if_not_converged \
      -pc_type fieldsplit \
      -pc_fieldsplit_detect_saddle_point \
      -pc_fieldsplit_type schur \
      -pc_fieldsplit_schur_precondition full
    test:
      suffix: linear
      args: -sol_form linear -mesh_transform none
    test:
      suffix: sinusoidal
      args: -sol_form sinusoidal -mesh_transform none
    test:
      suffix: sinusoidal_skew
      args: -sol_form sinusoidal -mesh_transform skew
    test:
      suffix: sinusoidal_perturb
      args: -sol_form sinusoidal -mesh_transform perturb
    test:
      suffix: sinusoidal_skew_perturb
      args: -sol_form sinusoidal -mesh_transform skew_perturb

  test:
    suffix: quad_rt_0
    args: -dim 2 -simplex false -mesh_transform skew \
          -divErr_petscspace_degree 1 \
          -divErr_petscdualspace_lagrange_continuity false \
          -dm_refine 0 \
          -snes_error_if_not_converged \
          -ksp_rtol 1e-10 \
          -ksp_error_if_not_converged \
          -pc_type fieldsplit\
          -pc_fieldsplit_detect_saddle_point\
          -pc_fieldsplit_type schur\
          -pc_fieldsplit_schur_precondition full \
          -velocity_petscfe_default_quadrature_order 1 \
          -velocity_petscspace_type sum \
          -velocity_petscspace_variables 2 \
          -velocity_petscspace_components 2 \
          -velocity_petscspace_sum_spaces 2 \
          -velocity_petscspace_sum_concatenate true \
          -velocity_subspace0_petscspace_variables 2 \
          -velocity_subspace0_petscspace_type tensor \
          -velocity_subspace0_petscspace_tensor_spaces 2 \
          -velocity_subspace0_petscspace_tensor_uniform false \
          -velocity_subspace0_subspace_0_petscspace_degree 1 \
          -velocity_subspace0_subspace_1_petscspace_degree 0 \
          -velocity_subspace1_petscspace_variables 2 \
          -velocity_subspace1_petscspace_type tensor \
          -velocity_subspace1_petscspace_tensor_spaces 2 \
          -velocity_subspace1_petscspace_tensor_uniform false \
          -velocity_subspace1_subspace_0_petscspace_degree 0 \
          -velocity_subspace1_subspace_1_petscspace_degree 1 \
          -velocity_petscdualspace_form_degree -1 \
          -velocity_petscdualspace_order 1 \
          -velocity_petscdualspace_lagrange_trimmed true
TEST*/
