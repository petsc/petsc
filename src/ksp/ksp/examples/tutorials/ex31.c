/*T
   Concepts: KSP^solving a system of linear equations
   Concepts: KSP^semi-implicit
   Processors: n
T*/

/*
This is intended to be a prototypical example of the semi-implicit algorithm for
a PDE. We have three phases:

  1) An explicit predictor step

     u^{k+1/3} = P(u^k)

  2) An implicit corrector step

     \Delta u^{k+2/3} = F(u^{k+1/3})

  3) An explicit update step

     u^{k+1} = C(u^{k+2/3})

We will solve on the unit square with Dirichlet boundary conditions

   u = f(x,y) for x = 0, x = 1, y = 0, y = 1

Although we are using a DMDA, and thus have a structured mesh, we will discretize
the problem with finite elements, splitting each cell of the DMDA into two
triangles.

This uses multigrid to solve the linear system
*/

static char help[] = "Solves 2D compressible Euler using multigrid.\n\n";

#include <petscdmda.h>
#include <petscksp.h>
#include <petscpcmg.h>

typedef struct {
  Vec rho;     /* The mass solution \rho */
  Vec rho_u;   /* The x-momentum solution \rho u */
  Vec rho_v;   /* The y-momentum solution \rho v */
  Vec rho_e;   /* The energy solution \rho e_t */
  Vec p;       /* The pressure solution P */
  Vec t;       /* The temperature solution T */
  Vec u;       /* The x-velocity solution u */
  Vec v;       /* The y-velocity solution v */
} SolutionContext;

typedef struct {
  SolutionContext sol_n;   /* The solution at time t^n */
  SolutionContext sol_phi; /* The element-averaged solution at time t^{n+\phi} */
  SolutionContext sol_np1; /* The solution at time t^{n+1} */
  Vec             mu;      /* The dynamic viscosity \mu(T) at time n */
  Vec             kappa;   /* The thermal conductivity \kappa(T) at time n */
  PetscScalar     phi;     /* The time weighting parameter */
  PetscScalar     dt;      /* The timestep \Delta t */
} UserContext;

extern PetscErrorCode CreateStructures(DM,UserContext*);
extern PetscErrorCode DestroyStructures(DM,UserContext*);
extern PetscErrorCode ComputePredictor(DM,UserContext*);
extern PetscErrorCode ComputeMatrix(KSP,Mat,Mat,MatStructure*,void*);
extern PetscErrorCode ComputeRHS(KSP,Vec,void*);
extern PetscErrorCode ComputeCorrector(DM,Vec,Vec);

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  KSP            ksp;
  DM             da;
  Vec            x, xNew;
  UserContext    user;
  PetscErrorCode ierr;

  PetscInitialize(&argc,&argv,(char *)0,help);

  ierr = KSPCreate(PETSC_COMM_WORLD,&ksp);CHKERRQ(ierr);
  ierr = DMDACreate2d(PETSC_COMM_WORLD, DMDA_BOUNDARY_NONE, DMDA_BOUNDARY_NONE,DMDA_STENCIL_STAR,3,3,PETSC_DECIDE,PETSC_DECIDE,1,1,0,0,&da);CHKERRQ(ierr);
  ierr = DMSetApplicationContext(da, &user);CHKERRQ(ierr);
  ierr = KSPSetDM(ksp, da);CHKERRQ(ierr);

  ierr = PetscOptionsBegin(PETSC_COMM_WORLD, "", "Options for PCICE", "DM");
    user.phi = 0.5;
    ierr = PetscOptionsScalar("-phi", "The time weighting parameter", "ex31.c", user.phi, &user.phi, PETSC_NULL);CHKERRQ(ierr);
    user.dt  = 0.1;
    ierr = PetscOptionsScalar("-dt", "The time step", "ex31.c", user.dt, &user.dt, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();

  ierr = CreateStructures(da, &user);CHKERRQ(ierr);
  ierr = ComputePredictor(da, &user);CHKERRQ(ierr);

  ierr = KSPSetComputeRHS(ksp,ComputeRHS,&user);CHKERRQ(ierr);
  ierr = KSPSetComputeOperators(ksp,ComputeMatrix,&user);CHKERRQ(ierr);
  ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);
  ierr = KSPSolve(ksp, PETSC_NULL, PETSC_NULL);CHKERRQ(ierr);

  ierr = KSPGetSolution(ksp, &x);CHKERRQ(ierr);
  ierr = VecDuplicate(x, &xNew);CHKERRQ(ierr);
  ierr = ComputeCorrector(da, x, xNew);CHKERRQ(ierr);
  ierr = VecDestroy(&xNew);CHKERRQ(ierr);

  ierr = DestroyStructures(da, &user);CHKERRQ(ierr);
  ierr = DMDestroy(&da);CHKERRQ(ierr);
  ierr = KSPDestroy(&ksp);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return 0;
}

#undef __FUNCT__
#define __FUNCT__ "CreateStructures"
PetscErrorCode CreateStructures(DM da, UserContext *user)
{
  const PetscInt *necon;
  PetscInt        ne,nc;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = DMDAGetElements(da,&ne,&nc,&necon);CHKERRQ(ierr);
  ierr = DMDARestoreElements(da,&ne,&nc,&necon);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(da, &user->sol_n.rho);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(da, &user->sol_n.rho_u);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(da, &user->sol_n.rho_v);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(da, &user->sol_n.rho_e);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(da, &user->sol_n.p);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(da, &user->sol_n.u);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(da, &user->sol_n.v);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(da, &user->sol_n.t);CHKERRQ(ierr);
  ierr = VecCreate(PETSC_COMM_WORLD, &user->sol_phi.rho);CHKERRQ(ierr);
  ierr = VecSetSizes(user->sol_phi.rho, ne, PETSC_DECIDE);CHKERRQ(ierr);
  ierr = VecSetType(user->sol_phi.rho,VECMPI);CHKERRQ(ierr);
  ierr = VecDuplicate(user->sol_phi.rho, &user->sol_phi.rho_u);CHKERRQ(ierr);
  ierr = VecDuplicate(user->sol_phi.rho, &user->sol_phi.rho_v);CHKERRQ(ierr);
  ierr = VecDuplicate(user->sol_phi.rho, &user->sol_phi.u);CHKERRQ(ierr);
  ierr = VecDuplicate(user->sol_phi.rho, &user->sol_phi.v);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(da, &user->sol_np1.rho);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(da, &user->sol_np1.rho_u);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(da, &user->sol_np1.rho_v);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(da, &user->sol_np1.rho_e);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(da, &user->sol_np1.p);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(da, &user->sol_np1.u);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(da, &user->sol_np1.v);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(da, &user->mu);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(da, &user->kappa);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DestroyStructures"
PetscErrorCode DestroyStructures(DM da, UserContext   *user)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecDestroy(&user->sol_n.rho);CHKERRQ(ierr);
  ierr = VecDestroy(&user->sol_n.rho_u);CHKERRQ(ierr);
  ierr = VecDestroy(&user->sol_n.rho_v);CHKERRQ(ierr);
  ierr = VecDestroy(&user->sol_n.rho_e);CHKERRQ(ierr);
  ierr = VecDestroy(&user->sol_n.p);CHKERRQ(ierr);
  ierr = VecDestroy(&user->sol_n.u);CHKERRQ(ierr);
  ierr = VecDestroy(&user->sol_n.v);CHKERRQ(ierr);
  ierr = VecDestroy(&user->sol_n.t);CHKERRQ(ierr);
  ierr = VecDestroy(&user->sol_phi.rho);CHKERRQ(ierr);
  ierr = VecDestroy(&user->sol_phi.rho_u);CHKERRQ(ierr);
  ierr = VecDestroy(&user->sol_phi.rho_v);CHKERRQ(ierr);
  ierr = VecDestroy(&user->sol_phi.u);CHKERRQ(ierr);
  ierr = VecDestroy(&user->sol_phi.v);CHKERRQ(ierr);
  ierr = VecDestroy(&user->sol_np1.rho);CHKERRQ(ierr);
  ierr = VecDestroy(&user->sol_np1.rho_u);CHKERRQ(ierr);
  ierr = VecDestroy(&user->sol_np1.rho_v);CHKERRQ(ierr);
  ierr = VecDestroy(&user->sol_np1.rho_e);CHKERRQ(ierr);
  ierr = VecDestroy(&user->sol_np1.p);CHKERRQ(ierr);
  ierr = VecDestroy(&user->sol_np1.u);CHKERRQ(ierr);
  ierr = VecDestroy(&user->sol_np1.v);CHKERRQ(ierr);
  ierr = VecDestroy(&user->mu);CHKERRQ(ierr);
  ierr = VecDestroy(&user->kappa);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "CalculateElementVelocity"
/* Average the velocity (u,v) at time t^n over each element for time n+\phi */
PetscErrorCode CalculateElementVelocity(DM da, UserContext *user)
{
  PetscScalar    *u_n,   *v_n;
  PetscScalar    *u_phi, *v_phi;
  const PetscInt *necon;
  PetscInt        j, e, ne, nc;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = DMDAGetElements(da, &ne, &nc, &necon);CHKERRQ(ierr);
  ierr = VecGetArray(user->sol_n.u, &u_n);CHKERRQ(ierr);
  ierr = VecGetArray(user->sol_n.v, &v_n);CHKERRQ(ierr);
  ierr = PetscMalloc(ne*sizeof(PetscScalar),&u_phi);CHKERRQ(ierr);
  ierr = PetscMalloc(ne*sizeof(PetscScalar),&v_phi);CHKERRQ(ierr);
  for(e = 0; e < ne; e++) {
    u_phi[e] = 0.0;
    v_phi[e] = 0.0;
    for(j = 0; j < 3; j++) {
      u_phi[e] += u_n[necon[3*e+j]];
      v_phi[e] += v_n[necon[3*e+j]];
    }
    u_phi[e] /= 3.0;
    v_phi[e] /= 3.0;
  }
  ierr = PetscFree(u_phi);CHKERRQ(ierr);
  ierr = PetscFree(v_phi);CHKERRQ(ierr);
  ierr = DMDARestoreElements(da, &ne,&nc, &necon);CHKERRQ(ierr);
  ierr = VecRestoreArray(user->sol_n.u, &u_n);CHKERRQ(ierr);
  ierr = VecRestoreArray(user->sol_n.v, &v_n);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TaylorGalerkinStepI"
/* This is equation 32,

   U^{n+\phi}_E = {1\over Vol_E} \left( \int_\Omega [N]{U^n} d\Omega - \phi\Delta t \int_\Omega [\nabla N]\cdot{F^n} d\Omega \right) + \phi\Delta t Q^n

which is really simple for linear elements

   U^{n+\phi}_E = {1\over3} \sum^3_{i=1} U^n_i - \phi\Delta t [\nabla N]\cdot{F^n} + \phi\Delta t Q^n

where

   U^{n+\phi}_E = {\rho  \rho u  \rho v}^{n+\phi}_E

and the x and y components of the convective fluxes F are

   f^n = {\rho u  \rho u^2  \rho uv}^n      g^n = {\rho v  \rho uv  \rho v^2}^n
*/
PetscErrorCode TaylorGalerkinStepI(DM da, UserContext *user)
{
  PetscScalar     phi_dt = user->phi*user->dt;
  PetscScalar    *u_n,     *v_n;
  PetscScalar    *rho_n,   *rho_u_n,   *rho_v_n;
  PetscScalar    *rho_phi, *rho_u_phi, *rho_v_phi;
  PetscScalar     Fx_x, Fy_y;
  PetscScalar     psi_x[3], psi_y[3];
  PetscInt        idx[3];
  PetscReal       hx, hy;
  const PetscInt *necon;
  PetscInt        j, e, ne, nc, mx, my;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = DMDAGetInfo(da, 0, &mx, &my, 0,0,0,0,0,0,0,0,0,0);CHKERRQ(ierr);
  hx   = 1.0 / (PetscReal)(mx-1);
  hy   = 1.0 / (PetscReal)(my-1);
  ierr = VecSet(user->sol_phi.rho,0.0);CHKERRQ(ierr);
  ierr = VecSet(user->sol_phi.rho_u,0.0);CHKERRQ(ierr);
  ierr = VecSet(user->sol_phi.rho_v,0.0);CHKERRQ(ierr);
  ierr = VecGetArray(user->sol_n.u,       &u_n);CHKERRQ(ierr);
  ierr = VecGetArray(user->sol_n.v,       &v_n);CHKERRQ(ierr);
  ierr = VecGetArray(user->sol_n.rho,     &rho_n);CHKERRQ(ierr);
  ierr = VecGetArray(user->sol_n.rho_u,   &rho_u_n);CHKERRQ(ierr);
  ierr = VecGetArray(user->sol_n.rho_v,   &rho_v_n);CHKERRQ(ierr);
  ierr = VecGetArray(user->sol_phi.rho,   &rho_phi);CHKERRQ(ierr);
  ierr = VecGetArray(user->sol_phi.rho_u, &rho_u_phi);CHKERRQ(ierr);
  ierr = VecGetArray(user->sol_phi.rho_v, &rho_v_phi);CHKERRQ(ierr);
  ierr = DMDAGetElements(da, &ne, &nc, &necon);CHKERRQ(ierr);
  for(e = 0; e < ne; e++) {
    /* Average the existing fields over the element */
    for(j = 0; j < 3; j++) {
      idx[j] = necon[3*e+j];
      rho_phi[e]   += rho_n[idx[j]];
      rho_u_phi[e] += rho_u_n[idx[j]];
      rho_v_phi[e] += rho_v_n[idx[j]];
    }
    rho_phi[e]   /= 3.0;
    rho_u_phi[e] /= 3.0;
    rho_v_phi[e] /= 3.0;
    /* Get basis function deriatives (we need the orientation of the element here) */
    if (idx[1] > idx[0]) {
      psi_x[0] = -hy; psi_x[1] =  hy; psi_x[2] = 0.0;
      psi_y[0] = -hx; psi_y[1] = 0.0; psi_y[2] =  hx;
    } else {
      psi_x[0] =  hy; psi_x[1] = -hy; psi_x[2] = 0.0;
      psi_y[0] =  hx; psi_y[1] = 0.0; psi_y[2] = -hx;
    }
    /* Determine the convective fluxes for \rho^{n+\phi} */
    Fx_x = 0.0; Fy_y = 0.0;
    for(j = 0; j < 3; j++) {
      Fx_x += psi_x[j]*rho_u_n[idx[j]];
      Fy_y += psi_y[j]*rho_v_n[idx[j]];
    }
    rho_phi[e] -= phi_dt*(Fx_x + Fy_y);
    /* Determine the convective fluxes for (\rho u)^{n+\phi} */
    Fx_x = 0.0; Fy_y = 0.0;
    for(j = 0; j < 3; j++) {
      Fx_x += psi_x[j]*rho_u_n[idx[j]]*u_n[idx[j]];
      Fy_y += psi_y[j]*rho_v_n[idx[j]]*u_n[idx[j]];
    }
    rho_u_phi[e] -= phi_dt*(Fx_x + Fy_y);
    /* Determine the convective fluxes for (\rho v)^{n+\phi} */
    Fx_x = 0.0; Fy_y = 0.0;
    for(j = 0; j < 3; j++) {
      Fx_x += psi_x[j]*rho_u_n[idx[j]]*v_n[idx[j]];
      Fy_y += psi_y[j]*rho_v_n[idx[j]]*v_n[idx[j]];
    }
    rho_v_phi[e] -= phi_dt*(Fx_x + Fy_y);
  }
  ierr = DMDARestoreElements(da, &ne, &nc, &necon);CHKERRQ(ierr);
  ierr = VecRestoreArray(user->sol_n.u,       &u_n);CHKERRQ(ierr);
  ierr = VecRestoreArray(user->sol_n.v,       &v_n);CHKERRQ(ierr);
  ierr = VecRestoreArray(user->sol_n.rho,     &rho_n);CHKERRQ(ierr);
  ierr = VecRestoreArray(user->sol_n.rho_u,   &rho_u_n);CHKERRQ(ierr);
  ierr = VecRestoreArray(user->sol_n.rho_v,   &rho_v_n);CHKERRQ(ierr);
  ierr = VecRestoreArray(user->sol_phi.rho,   &rho_phi);CHKERRQ(ierr);
  ierr = VecRestoreArray(user->sol_phi.rho_u, &rho_u_phi);CHKERRQ(ierr);
  ierr = VecRestoreArray(user->sol_phi.rho_v, &rho_v_phi);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TaylorGalerkinStepIIMomentum"
/*
The element stiffness matrix for the identity in linear elements is

  1  /2 1 1\
  -  |1 2 1|
  12 \1 1 2/

  no matter what the shape of the triangle. */
PetscErrorCode TaylorGalerkinStepIIMomentum(DM da, UserContext *user)
{
  MPI_Comm        comm;
  KSP             ksp;
  Mat             mat;
  Vec             rhs_u, rhs_v;
  PetscScalar     identity[9] = {0.16666666667, 0.08333333333, 0.08333333333,
                                 0.08333333333, 0.16666666667, 0.08333333333,
                                 0.08333333333, 0.08333333333, 0.16666666667};
  PetscScalar    *u_n,       *v_n,      *mu_n;
  PetscScalar    *u_phi,     *v_phi;
  PetscScalar    *rho_u_phi, *rho_v_phi;
  PetscInt        idx[3];
  PetscScalar     values_u[3];
  PetscScalar     values_v[3];
  PetscScalar     psi_x[3], psi_y[3];
  PetscScalar     mu, tau_xx, tau_xy, tau_yy;
  PetscReal       hx, hy, area;
  const PetscInt *necon;
  PetscInt        j, k, e, ne, nc, mx, my;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject) da, &comm);CHKERRQ(ierr);
  ierr = DMCreateMatrix(da, MATAIJ, &mat);CHKERRQ(ierr);
  ierr = MatSetOption(mat,MAT_NEW_NONZERO_ALLOCATION_ERR,PETSC_FALSE);CHKERRQ(ierr);
  ierr = DMGetGlobalVector(da, &rhs_u);CHKERRQ(ierr);
  ierr = DMGetGlobalVector(da, &rhs_v);CHKERRQ(ierr);
  ierr = KSPCreate(comm, &ksp);CHKERRQ(ierr);
  ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);

  ierr = DMDAGetInfo(da, 0, &mx, &my, 0,0,0,0,0,0,0,0,0,0);CHKERRQ(ierr);
  hx   = 1.0 / (PetscReal)(mx-1);
  hy   = 1.0 / (PetscReal)(my-1);
  area = 0.5*hx*hy;
  ierr = VecGetArray(user->sol_n.u,       &u_n);CHKERRQ(ierr);
  ierr = VecGetArray(user->sol_n.v,       &v_n);CHKERRQ(ierr);
  ierr = VecGetArray(user->mu,            &mu_n);CHKERRQ(ierr);
  ierr = VecGetArray(user->sol_phi.u,     &u_phi);CHKERRQ(ierr);
  ierr = VecGetArray(user->sol_phi.v,     &v_phi);CHKERRQ(ierr);
  ierr = VecGetArray(user->sol_phi.rho_u, &rho_u_phi);CHKERRQ(ierr);
  ierr = VecGetArray(user->sol_phi.rho_v, &rho_v_phi);CHKERRQ(ierr);
  ierr = DMDAGetElements(da, &ne, &nc, &necon);CHKERRQ(ierr);
  for(e = 0; e < ne; e++) {
    for(j = 0; j < 3; j++) {
      idx[j] = necon[3*e+j];
      values_u[j] = 0.0;
      values_v[j] = 0.0;
    }
    /* Get basis function deriatives (we need the orientation of the element here) */
    if (idx[1] > idx[0]) {
      psi_x[0] = -hy; psi_x[1] =  hy; psi_x[2] = 0.0;
      psi_y[0] = -hx; psi_y[1] = 0.0; psi_y[2] =  hx;
    } else {
      psi_x[0] =  hy; psi_x[1] = -hy; psi_x[2] = 0.0;
      psi_y[0] =  hx; psi_y[1] = 0.0; psi_y[2] = -hx;
    }
    /*  <\nabla\psi, F^{n+\phi}_e>: Divergence of the element-averaged convective fluxes */
    for(j = 0; j < 3; j++) {
      values_u[j] += psi_x[j]*rho_u_phi[e]*u_phi[e] + psi_y[j]*rho_u_phi[e]*v_phi[e];
      values_v[j] += psi_x[j]*rho_v_phi[e]*u_phi[e] + psi_y[j]*rho_v_phi[e]*v_phi[e];
    }
    /*  -<\nabla\psi, F^n_v>: Divergence of the viscous fluxes */
    for(j = 0; j < 3; j++) {
      /* \tau_{xx} = 2/3 \mu(T) (2 {\partial u\over\partial x} - {\partial v\over\partial y}) */
      /* \tau_{xy} =     \mu(T) (  {\partial u\over\partial y} + {\partial v\over\partial x}) */
      /* \tau_{yy} = 2/3 \mu(T) (2 {\partial v\over\partial y} - {\partial u\over\partial x}) */
      mu     = 0.0;
      tau_xx = 0.0;
      tau_xy = 0.0;
      tau_yy = 0.0;
      for(k = 0; k < 3; k++) {
        mu     += mu_n[idx[k]];
        tau_xx += 2.0*psi_x[k]*u_n[idx[k]] - psi_y[k]*v_n[idx[k]];
        tau_xy +=     psi_y[k]*u_n[idx[k]] + psi_x[k]*v_n[idx[k]];
        tau_yy += 2.0*psi_y[k]*v_n[idx[k]] - psi_x[k]*u_n[idx[k]];
      }
      mu     /= 3.0;
      tau_xx *= (2.0/3.0)*mu;
      tau_xy *= mu;
      tau_yy *= (2.0/3.0)*mu;
      values_u[j] -= area*(psi_x[j]*tau_xx + psi_y[j]*tau_xy);
      values_v[j] -= area*(psi_x[j]*tau_xy + psi_y[j]*tau_yy);
    }
    /* Accumulate to global structures */
    ierr = VecSetValuesLocal(rhs_u, 3, idx, values_u, ADD_VALUES);CHKERRQ(ierr);
    ierr = VecSetValuesLocal(rhs_v, 3, idx, values_v, ADD_VALUES);CHKERRQ(ierr);
    ierr = MatSetValuesLocal(mat, 3, idx, 3, idx, identity, ADD_VALUES);CHKERRQ(ierr);
  }
  ierr = DMDARestoreElements(da, &ne, &nc, &necon);CHKERRQ(ierr);
  ierr = VecRestoreArray(user->sol_n.u,       &u_n);CHKERRQ(ierr);
  ierr = VecRestoreArray(user->sol_n.v,       &v_n);CHKERRQ(ierr);
  ierr = VecRestoreArray(user->mu,            &mu_n);CHKERRQ(ierr);
  ierr = VecRestoreArray(user->sol_phi.u,     &u_phi);CHKERRQ(ierr);
  ierr = VecRestoreArray(user->sol_phi.v,     &v_phi);CHKERRQ(ierr);
  ierr = VecRestoreArray(user->sol_phi.rho_u, &rho_u_phi);CHKERRQ(ierr);
  ierr = VecRestoreArray(user->sol_phi.rho_v, &rho_v_phi);CHKERRQ(ierr);

  ierr = VecAssemblyBegin(rhs_u);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(rhs_v);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(mat, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(rhs_u);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(rhs_v);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(mat, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = VecScale(rhs_u,user->dt);CHKERRQ(ierr);
  ierr = VecScale(rhs_v,user->dt);CHKERRQ(ierr);

  ierr = KSPSetOperators(ksp, mat, mat, DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
  ierr = KSPSolve(ksp, rhs_u, user->sol_np1.rho_u);CHKERRQ(ierr);
  ierr = KSPSolve(ksp, rhs_v, user->sol_np1.rho_v);CHKERRQ(ierr);
  ierr = KSPDestroy(&ksp);CHKERRQ(ierr);
  ierr = MatDestroy(&mat);CHKERRQ(ierr);
  ierr = DMRestoreGlobalVector(da, &rhs_u);CHKERRQ(ierr);
  ierr = DMRestoreGlobalVector(da, &rhs_v);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TaylorGalerkinStepIIMassEnergy"
/* Notice that this requires the previous momentum solution.

The element stiffness matrix for the identity in linear elements is

  1  /2 1 1\
  -  |1 2 1|
  12 \1 1 2/

  no matter what the shape of the triangle. */
PetscErrorCode TaylorGalerkinStepIIMassEnergy(DM da, UserContext *user)
{
  MPI_Comm        comm;
  Mat             mat;
  Vec             rhs_m, rhs_e;
  PetscScalar     identity[9] = {0.16666666667, 0.08333333333, 0.08333333333,
                                 0.08333333333, 0.16666666667, 0.08333333333,
                                 0.08333333333, 0.08333333333, 0.16666666667};
  PetscScalar    *u_n,       *v_n,     *p_n,     *t_n,     *mu_n,    *kappa_n;
  PetscScalar    *rho_n,     *rho_u_n, *rho_v_n, *rho_e_n;
  PetscScalar    *u_phi,     *v_phi;
  PetscScalar    *rho_u_np1, *rho_v_np1;
  PetscInt        idx[3];
  PetscScalar     psi_x[3], psi_y[3];
  PetscScalar     values_m[3];
  PetscScalar     values_e[3];
  PetscScalar     phi = user->phi;
  PetscScalar     mu, kappa, tau_xx, tau_xy, tau_yy, q_x, q_y;
  PetscReal       hx, hy, area;
  KSP             ksp;
  const PetscInt *necon;
  PetscInt        j, k, e, ne, nc, mx, my;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject) da, &comm);CHKERRQ(ierr);
  ierr = DMCreateMatrix(da, MATAIJ, &mat);CHKERRQ(ierr);
  ierr = MatSetOption(mat,MAT_NEW_NONZERO_ALLOCATION_ERR,PETSC_FALSE);CHKERRQ(ierr);
  ierr = DMGetGlobalVector(da, &rhs_m);CHKERRQ(ierr);
  ierr = DMGetGlobalVector(da, &rhs_e);CHKERRQ(ierr);
  ierr = KSPCreate(comm, &ksp);CHKERRQ(ierr);
  ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);

  ierr = DMDAGetInfo(da, 0, &mx, &my, 0,0,0,0,0,0,0,0,0,0);CHKERRQ(ierr);
  hx   = 1.0 / (PetscReal)(mx-1);
  hy   = 1.0 / (PetscReal)(my-1);
  area = 0.5*hx*hy;
  ierr = VecGetArray(user->sol_n.u,       &u_n);CHKERRQ(ierr);
  ierr = VecGetArray(user->sol_n.v,       &v_n);CHKERRQ(ierr);
  ierr = VecGetArray(user->sol_n.p,       &p_n);CHKERRQ(ierr);
  ierr = VecGetArray(user->sol_n.t,       &t_n);CHKERRQ(ierr);
  ierr = VecGetArray(user->mu,            &mu_n);CHKERRQ(ierr);
  ierr = VecGetArray(user->kappa,         &kappa_n);CHKERRQ(ierr);
  ierr = VecGetArray(user->sol_n.rho,     &rho_n);CHKERRQ(ierr);
  ierr = VecGetArray(user->sol_n.rho_u,   &rho_u_n);CHKERRQ(ierr);
  ierr = VecGetArray(user->sol_n.rho_v,   &rho_v_n);CHKERRQ(ierr);
  ierr = VecGetArray(user->sol_n.rho_e,   &rho_e_n);CHKERRQ(ierr);
  ierr = VecGetArray(user->sol_phi.u,     &u_phi);CHKERRQ(ierr);
  ierr = VecGetArray(user->sol_phi.v,     &v_phi);CHKERRQ(ierr);
  ierr = VecGetArray(user->sol_np1.rho_u, &rho_u_np1);CHKERRQ(ierr);
  ierr = VecGetArray(user->sol_np1.rho_v, &rho_v_np1);CHKERRQ(ierr);
  ierr = DMDAGetElements(da, &ne, &nc, &necon);CHKERRQ(ierr);
  for(e = 0; e < ne; e++) {
    for(j = 0; j < 3; j++) {
      idx[j] = necon[3*e+j];
      values_m[j] = 0.0;
      values_e[j] = 0.0;
    }
    /* Get basis function deriatives (we need the orientation of the element here) */
    if (idx[1] > idx[0]) {
      psi_x[0] = -hy; psi_x[1] =  hy; psi_x[2] = 0.0;
      psi_y[0] = -hx; psi_y[1] = 0.0; psi_y[2] =  hx;
    } else {
      psi_x[0] =  hy; psi_x[1] = -hy; psi_x[2] = 0.0;
      psi_y[0] =  hx; psi_y[1] = 0.0; psi_y[2] = -hx;
    }
    /*  <\nabla\psi, F^*>: Divergence of the predicted convective fluxes */
    for(j = 0; j < 3; j++) {
      values_m[j] += (psi_x[j]*(phi*rho_u_np1[idx[j]] + rho_u_n[idx[j]]) + psi_y[j]*(rho_v_np1[idx[j]] + rho_v_n[idx[j]]))/3.0;
      values_e[j] += values_m[j]*((rho_e_n[idx[j]] + p_n[idx[j]]) / rho_n[idx[j]]);
    }
    /*  -<\nabla\psi, F^n_v>: Divergence of the viscous fluxes */
    for(j = 0; j < 3; j++) {
      /* \tau_{xx} = 2/3 \mu(T) (2 {\partial u\over\partial x} - {\partial v\over\partial y}) */
      /* \tau_{xy} =     \mu(T) (  {\partial u\over\partial y} + {\partial v\over\partial x}) */
      /* \tau_{yy} = 2/3 \mu(T) (2 {\partial v\over\partial y} - {\partial u\over\partial x}) */
      /* q_x       = -\kappa(T) {\partial T\over\partial x} */
      /* q_y       = -\kappa(T) {\partial T\over\partial y} */

      /* above code commeted out - causing ininitialized variables. */
      q_x =0; q_y =0;

      mu     = 0.0;
      kappa  = 0.0;
      tau_xx = 0.0;
      tau_xy = 0.0;
      tau_yy = 0.0;
      for(k = 0; k < 3; k++) {
        mu     += mu_n[idx[k]];
        kappa  += kappa_n[idx[k]];
        tau_xx += 2.0*psi_x[k]*u_n[idx[k]] - psi_y[k]*v_n[idx[k]];
        tau_xy +=     psi_y[k]*u_n[idx[k]] + psi_x[k]*v_n[idx[k]];
        tau_yy += 2.0*psi_y[k]*v_n[idx[k]] - psi_x[k]*u_n[idx[k]];
        q_x    += psi_x[k]*t_n[idx[k]];
        q_y    += psi_y[k]*t_n[idx[k]];
      }
      mu     /= 3.0;
      kappa  /= 3.0;
      tau_xx *= (2.0/3.0)*mu;
      tau_xy *= mu;
      tau_yy *= (2.0/3.0)*mu;
      values_e[j] -= area*(psi_x[j]*(u_phi[e]*tau_xx + v_phi[e]*tau_xy + q_x) + psi_y[j]*(u_phi[e]*tau_xy + v_phi[e]*tau_yy + q_y));
    }
    /* Accumulate to global structures */
    ierr = VecSetValuesLocal(rhs_m, 3, idx, values_m, ADD_VALUES);CHKERRQ(ierr);
    ierr = VecSetValuesLocal(rhs_e, 3, idx, values_e, ADD_VALUES);CHKERRQ(ierr);
    ierr = MatSetValuesLocal(mat, 3, idx, 3, idx, identity, ADD_VALUES);CHKERRQ(ierr);
  }
  ierr = DMDARestoreElements(da, &ne, &nc, &necon);CHKERRQ(ierr);
  ierr = VecRestoreArray(user->sol_n.u,       &u_n);CHKERRQ(ierr);
  ierr = VecRestoreArray(user->sol_n.v,       &v_n);CHKERRQ(ierr);
  ierr = VecRestoreArray(user->sol_n.p,       &p_n);CHKERRQ(ierr);
  ierr = VecRestoreArray(user->sol_n.t,       &t_n);CHKERRQ(ierr);
  ierr = VecRestoreArray(user->mu,            &mu_n);CHKERRQ(ierr);
  ierr = VecRestoreArray(user->kappa,         &kappa_n);CHKERRQ(ierr);
  ierr = VecRestoreArray(user->sol_n.rho,     &rho_n);CHKERRQ(ierr);
  ierr = VecRestoreArray(user->sol_n.rho_u,   &rho_u_n);CHKERRQ(ierr);
  ierr = VecRestoreArray(user->sol_n.rho_v,   &rho_v_n);CHKERRQ(ierr);
  ierr = VecRestoreArray(user->sol_n.rho_e,   &rho_e_n);CHKERRQ(ierr);
  ierr = VecRestoreArray(user->sol_phi.u,     &u_phi);CHKERRQ(ierr);
  ierr = VecRestoreArray(user->sol_phi.v,     &v_phi);CHKERRQ(ierr);
  ierr = VecRestoreArray(user->sol_np1.rho_u, &rho_u_np1);CHKERRQ(ierr);
  ierr = VecRestoreArray(user->sol_np1.rho_v, &rho_v_np1);CHKERRQ(ierr);

  ierr = VecAssemblyBegin(rhs_m);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(rhs_e);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(mat, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(rhs_m);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(rhs_e);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(mat, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = VecScale(rhs_m, user->dt);CHKERRQ(ierr);
  ierr = VecScale(rhs_e, user->dt);CHKERRQ(ierr);

  ierr = KSPSetOperators(ksp, mat, mat, DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
  ierr = KSPSolve(ksp, rhs_m, user->sol_np1.rho);CHKERRQ(ierr);
  ierr = KSPSolve(ksp, rhs_e, user->sol_np1.rho_e);CHKERRQ(ierr);
  ierr = KSPDestroy(&ksp);CHKERRQ(ierr);
  ierr = MatDestroy(&mat);CHKERRQ(ierr);
  ierr = DMRestoreGlobalVector(da, &rhs_m);CHKERRQ(ierr);
  ierr = DMRestoreGlobalVector(da, &rhs_e);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ComputePredictor"
PetscErrorCode ComputePredictor(DM da, UserContext *user)
{
  Vec            uOldLocal, uLocal,uOld;
  PetscScalar   *pOld;
  PetscScalar   *p;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMGetGlobalVector(da, &uOld);CHKERRQ(ierr);
  ierr = DMGetLocalVector(da, &uOldLocal);CHKERRQ(ierr);
  ierr = DMGetLocalVector(da, &uLocal);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(da, uOld, INSERT_VALUES, uOldLocal);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(da, uOld, INSERT_VALUES, uOldLocal);CHKERRQ(ierr);
  ierr = VecGetArray(uOldLocal, &pOld);CHKERRQ(ierr);
  ierr = VecGetArray(uLocal,    &p);CHKERRQ(ierr);

  /* Source terms are all zero right now */
  ierr = CalculateElementVelocity(da, user);CHKERRQ(ierr);
  ierr = TaylorGalerkinStepI(da, user);CHKERRQ(ierr);
  ierr = TaylorGalerkinStepIIMomentum(da, user);CHKERRQ(ierr);
  ierr = TaylorGalerkinStepIIMassEnergy(da, user);CHKERRQ(ierr);
  /* Solve equation (9) for \delta(\rho\vu) and (\rho\vu)^* */
  /* Solve equation (13) for \delta\rho and \rho^* */
  /* Solve equation (15) for \delta(\rho e_t) and (\rho e_t)^* */
  /* Apply artifical dissipation */
  /* Determine the smoothed explicit pressure, \tilde P and temperature \tilde T using the equation of state */


  ierr = VecRestoreArray(uOldLocal, &pOld);CHKERRQ(ierr);
  ierr = VecRestoreArray(uLocal,    &p);CHKERRQ(ierr);
#if 0
  ierr = DMLocalToGlobalBegin(da, uLocal, ADD_VALUES,u);CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd(da, uLocal, ADD_VALUES,u);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(da, &uOldLocal);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(da, &uLocal);CHKERRQ(ierr);
#endif
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ComputeRHS"
/*
  We integrate over each cell

  (i, j+1)----(i+1, j+1)
      | \         |
      |  \        |
      |   \       |
      |    \      |
      |     \     |
      |      \    |
      |       \   |
  (i,   j)----(i+1, j)
*/
PetscErrorCode ComputeRHS(KSP ksp, Vec b, void *ctx)
{
  UserContext   *user = (UserContext*)ctx;
  PetscScalar    phi  = user->phi;
  PetscScalar   *array;
  PetscInt       ne,nc,i;
  const PetscInt *e;
  PetscErrorCode ierr;
  Vec            blocal;
  DM             da;

  PetscFunctionBegin;
  ierr = KSPGetDM(ksp,&da);CHKERRQ(ierr);
  /* access a local vector with room for the ghost points */
  ierr = DMGetLocalVector(da,&blocal);CHKERRQ(ierr);
  ierr = VecGetArray(blocal, (PetscScalar **) &array);CHKERRQ(ierr);

  /* access the list of elements on this processor and loop over them */
  ierr = DMDAGetElements(da,&ne,&nc,&e);CHKERRQ(ierr);
  for (i=0; i<ne; i++) {

    /* this is nonsense, but set each nodal value to phi (will actually do integration over element */
    array[e[3*i]]   = phi;
    array[e[3*i+1]] = phi;
    array[e[3*i+2]] = phi;
  }
  ierr = VecRestoreArray(blocal, (PetscScalar **) &array);CHKERRQ(ierr);
  ierr = DMDARestoreElements(da,&ne,&nc,&e);CHKERRQ(ierr);

  /* add our partial sums over all processors into b */
  ierr = DMLocalToGlobalBegin(da,blocal,ADD_VALUES,b);CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd(da,blocal, ADD_VALUES,b);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(da,&blocal);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ComputeMatrix"
/*
  We integrate over each cell

  (i, j+1)----(i+1, j+1)
      | \         |
      |  \        |
      |   \       |
      |    \      |
      |     \     |
      |      \    |
      |       \   |
  (i,   j)----(i+1, j)

However, the element stiffness matrix for the identity in linear elements is

  1  /2 1 1\
  -  |1 2 1|
  12 \1 1 2/

no matter what the shape of the triangle. The Laplacian stiffness matrix is

  1  /         (x_2 - x_1)^2 + (y_2 - y_1)^2           -(x_2 - x_0)(x_2 - x_1) - (y_2 - y_1)(y_2 - y_0)  (x_1 - x_0)(x_2 - x_1) + (y_1 - y_0)(y_2 - y_1)\
  -  |-(x_2 - x_0)(x_2 - x_1) - (y_2 - y_1)(y_2 - y_0)           (x_2 - x_0)^2 + (y_2 - y_0)^2          -(x_1 - x_0)(x_2 - x_0) - (y_1 - y_0)(y_2 - y_0)|
  A  \ (x_1 - x_0)(x_2 - x_1) + (y_1 - y_0)(y_2 - y_1) -(x_1 - x_0)(x_2 - x_0) - (y_1 - y_0)(y_2 - y_0)           (x_1 - x_0)^2 + (y_1 - y_0)^2         /

where A is the area of the triangle, and (x_i, y_i) is its i'th vertex.
*/
PetscErrorCode ComputeMatrix(KSP ksp, Mat J, Mat jac, MatStructure *flag,void *ctx)
{
  UserContext   *user = (UserContext*)ctx;
  /* not being used!
  PetscScalar    identity[9] = {0.16666666667, 0.08333333333, 0.08333333333,
                                0.08333333333, 0.16666666667, 0.08333333333,
                                0.08333333333, 0.08333333333, 0.16666666667};
  */
  PetscScalar    values[3][3];
  PetscInt       idx[3];
  PetscScalar    hx, hy, hx2, hy2, area,phi_dt2;
  PetscInt       i,mx,my,xm,ym,xs,ys;
  PetscInt       ne,nc;
  const PetscInt *e;
  PetscErrorCode ierr;
  DM             da;

  PetscFunctionBegin;
  ierr = KSPGetDM(ksp,&da);CHKERRQ(ierr);
  ierr = DMDAGetInfo(da, 0, &mx, &my, 0,0,0,0,0,0,0,0,0,0);CHKERRQ(ierr);
  ierr = DMDAGetCorners(da,&xs,&ys,0,&xm,&ym,0);CHKERRQ(ierr);
  hx   = 1.0 / (mx-1);
  hy   = 1.0 / (my-1);
  area = 0.5*hx*hy;
  phi_dt2 = user->phi*user->dt*user->dt;
  hx2     = hx*hx/area*phi_dt2;
  hy2     = hy*hy/area*phi_dt2;

  /* initially all elements have identical geometry so all element stiffness are identical */
  values[0][0] = hx2 + hy2; values[0][1] = -hy2; values[0][2] = -hx2;
  values[1][0] = -hy2;      values[1][1] = hy2;  values[1][2] = 0.0;
  values[2][0] = -hx2;      values[2][1] = 0.0;  values[2][2] = hx2;

  ierr = DMDAGetElements(da,&ne,&nc,&e);CHKERRQ(ierr);
  for (i=0; i<ne; i++) {
    idx[0] = e[3*i];
    idx[1] = e[3*i+1];
    idx[2] = e[3*i+2];
    ierr = MatSetValuesLocal(jac,3,idx,3,idx,(PetscScalar*)values,ADD_VALUES);CHKERRQ(ierr);
  }
  ierr = DMDARestoreElements(da,&ne,&nc,&e);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(jac, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(jac, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ComputeCorrector"
PetscErrorCode ComputeCorrector(DM da, Vec uOld, Vec u)
{
  Vec            uOldLocal, uLocal;
  PetscScalar    *cOld;
  PetscScalar    *c;
  PetscInt       i,ne,nc;
  const PetscInt *e;
  PetscErrorCode ierr;
  
  PetscFunctionBegin;
  ierr = VecSet(u,0.0);CHKERRQ(ierr);
  ierr = DMGetLocalVector(da, &uOldLocal);CHKERRQ(ierr);
  ierr = DMGetLocalVector(da, &uLocal);CHKERRQ(ierr);
  ierr = VecSet(uLocal,0.0);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(da, uOld, INSERT_VALUES, uOldLocal);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(da, uOld, INSERT_VALUES, uOldLocal);CHKERRQ(ierr);
  ierr = VecGetArray(uOldLocal, &cOld);CHKERRQ(ierr);
  ierr = VecGetArray(uLocal,    &c);CHKERRQ(ierr);

  /* access the list of elements on this processor and loop over them */
  ierr = DMDAGetElements(da,&ne,&nc,&e);CHKERRQ(ierr);
  for (i=0; i<ne; i++) {

    /* this is nonsense, but copy each nodal value*/
    c[e[3*i]]   = cOld[e[3*i]];
    c[e[3*i+1]] = cOld[e[3*i+1]];
    c[e[3*i+2]] = cOld[e[3*i+2]];
  }
  ierr = DMDARestoreElements(da,&ne,&nc,&e);CHKERRQ(ierr);
  ierr = VecRestoreArray(uOldLocal, &cOld);CHKERRQ(ierr);
  ierr = VecRestoreArray(uLocal,    &c);CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(da, uLocal, ADD_VALUES,u);CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd(da, uLocal, ADD_VALUES,u);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(da, &uOldLocal);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(da, &uLocal);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
