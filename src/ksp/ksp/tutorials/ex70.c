static char help[] = "------------------------------------------------------------------------------------------------------------------------------ \n\
  Solves the time-dependent incompressible, variable viscosity Stokes equation in 2D driven by buouyancy variations. \n\
  Time-dependence is introduced by evolving the density (rho) and viscosity (eta) according to \n\
    D \\rho / Dt = 0    and    D \\eta / Dt = 0 \n\
  The Stokes problem is discretized using Q1-Q1 finite elements, stabilized with Bochev's polynomial projection method. \n\
  The hyperbolic evolution equation for density is discretized using a variant of the Particle-In-Cell (PIC) method. \n\
  The DMDA object is used to define the FE problem, whilst DMSwarm provides support for the PIC method. \n\
  Material points (particles) store density and viscosity. The particles are advected with the fluid velocity using RK1. \n\
  At each time step, the value of density and viscosity stored on each particle are projected into a Q1 function space \n\
  and then interpolated onto the Gauss quadrature points. \n\
  The model problem defined in this example is the iso-viscous Rayleigh-Taylor instability (case 1a) from: \n\
    \"A comparison of methods for the modeling of thermochemical convection\" \n\
    P.E. van Keken, S.D. King, H. Schmeling, U.R. Christensen, D. Neumeister and M.-P. Doin, \n\
    Journal of Geophysical Research, vol 102 (B10), 477--499 (1997) \n\
  Note that whilst the model problem defined is for an iso-viscous, the implementation in this example supports \n\
  variable viscoity formulations. \n\
  This example is based on src/ksp/ksp/tutorials/ex43.c \n\
  Options: \n\
    -mx        : Number of elements in the x-direction \n\
    -my        : Number of elements in the y-direction \n\
    -mxy       : Number of elements in the x- and y-directions \n\
    -nt        : Number of time steps \n\
    -dump_freq : Frequency of output file creation \n\
    -ppcell    : Number of times the reference cell is sub-divided \n\
    -randomize_coords : Apply a random shift to each particle coordinate in the range [-fac*dh,0.fac*dh] \n\
    -randomize_fac    : Set the scaling factor for the random shift (default = 0.25)\n";

/* Contributed by Dave May */

#include <petscksp.h>
#include <petscdm.h>
#include <petscdmda.h>
#include <petscdmswarm.h>
#include <petsc/private/dmimpl.h>

static PetscErrorCode DMDAApplyBoundaryConditions(DM,Mat,Vec);

#define NSD            2 /* number of spatial dimensions */
#define NODES_PER_EL   4 /* nodes per element */
#define U_DOFS         2 /* degrees of freedom per velocity node */
#define P_DOFS         1 /* degrees of freedom per pressure node */
#define GAUSS_POINTS   4

static void EvaluateBasis_Q1(PetscScalar _xi[],PetscScalar N[])
{
  PetscScalar xi  = _xi[0];
  PetscScalar eta = _xi[1];

  N[0] = 0.25*(1.0-xi)*(1.0-eta);
  N[1] = 0.25*(1.0+xi)*(1.0-eta);
  N[2] = 0.25*(1.0+xi)*(1.0+eta);
  N[3] = 0.25*(1.0-xi)*(1.0+eta);
}

static void EvaluateBasisDerivatives_Q1(PetscScalar _xi[],PetscScalar dN[][NODES_PER_EL])
{
  PetscScalar xi  = _xi[0];
  PetscScalar eta = _xi[1];

  dN[0][0] = -0.25*(1.0-eta);
  dN[0][1] =  0.25*(1.0-eta);
  dN[0][2] =  0.25*(1.0+eta);
  dN[0][3] = -0.25*(1.0+eta);

  dN[1][0] = -0.25*(1.0-xi);
  dN[1][1] = -0.25*(1.0+xi);
  dN[1][2] =  0.25*(1.0+xi);
  dN[1][3] =  0.25*(1.0-xi);
}

static void EvaluateDerivatives(PetscScalar dN[][NODES_PER_EL],PetscScalar dNx[][NODES_PER_EL],PetscScalar coords[],PetscScalar *det_J)
{
  PetscScalar J00,J01,J10,J11,J;
  PetscScalar iJ00,iJ01,iJ10,iJ11;
  PetscInt    i;

  J00 = J01 = J10 = J11 = 0.0;
  for (i = 0; i < NODES_PER_EL; i++) {
    PetscScalar cx = coords[2*i];
    PetscScalar cy = coords[2*i+1];

    J00 += dN[0][i]*cx;      /* J_xx = dx/dxi */
    J01 += dN[0][i]*cy;      /* J_xy = dy/dxi */
    J10 += dN[1][i]*cx;      /* J_yx = dx/deta */
    J11 += dN[1][i]*cy;      /* J_yy = dy/deta */
  }
  J = (J00*J11)-(J01*J10);

  iJ00 =  J11/J;
  iJ01 = -J01/J;
  iJ10 = -J10/J;
  iJ11 =  J00/J;

  for (i = 0; i < NODES_PER_EL; i++) {
    dNx[0][i] = dN[0][i]*iJ00+dN[1][i]*iJ01;
    dNx[1][i] = dN[0][i]*iJ10+dN[1][i]*iJ11;
  }

  if (det_J) *det_J = J;
}

static void CreateGaussQuadrature(PetscInt *ngp,PetscScalar gp_xi[][2],PetscScalar gp_weight[])
{
  *ngp         = 4;
  gp_xi[0][0]  = -0.57735026919; gp_xi[0][1] = -0.57735026919;
  gp_xi[1][0]  = -0.57735026919; gp_xi[1][1] =  0.57735026919;
  gp_xi[2][0]  =  0.57735026919; gp_xi[2][1] =  0.57735026919;
  gp_xi[3][0]  =  0.57735026919; gp_xi[3][1] = -0.57735026919;
  gp_weight[0] = 1.0;
  gp_weight[1] = 1.0;
  gp_weight[2] = 1.0;
  gp_weight[3] = 1.0;
}

static PetscErrorCode DMDAGetElementEqnums_up(const PetscInt element[],PetscInt s_u[],PetscInt s_p[])
{
  PetscInt i;
  PetscFunctionBeginUser;
  for (i=0; i<NODES_PER_EL; i++) {
    /* velocity */
    s_u[NSD*i+0] = 3*element[i];
    s_u[NSD*i+1] = 3*element[i]+1;
    /* pressure */
    s_p[i] = 3*element[i]+2;
  }
  PetscFunctionReturn(0);
}

static PetscInt map_wIwDI_uJuDJ(PetscInt wi,PetscInt wd,PetscInt w_NPE,PetscInt w_dof,PetscInt ui,PetscInt ud,PetscInt u_NPE,PetscInt u_dof)
{
  PetscInt ij,r,c,nc;

  nc = u_NPE*u_dof;
  r = w_dof*wi+wd;
  c = u_dof*ui+ud;
  ij = r*nc+c;
  return(ij);
}

static void BForm_DivT(PetscScalar Ke[],PetscScalar coords[],PetscScalar eta[])
{
  PetscScalar gp_xi[GAUSS_POINTS][NSD],gp_weight[GAUSS_POINTS];
  PetscScalar GNi_p[NSD][NODES_PER_EL],GNx_p[NSD][NODES_PER_EL];
  PetscScalar J_p,tildeD[3];
  PetscScalar B[3][U_DOFS*NODES_PER_EL];
  PetscInt    p,i,j,k,ngp;

  /* define quadrature rule */
  CreateGaussQuadrature(&ngp,gp_xi,gp_weight);

  /* evaluate bilinear form */
  for (p = 0; p < ngp; p++) {
    EvaluateBasisDerivatives_Q1(gp_xi[p],GNi_p);
    EvaluateDerivatives(GNi_p,GNx_p,coords,&J_p);

    for (i = 0; i < NODES_PER_EL; i++) {
      PetscScalar d_dx_i = GNx_p[0][i];
      PetscScalar d_dy_i = GNx_p[1][i];

      B[0][2*i] = d_dx_i;B[0][2*i+1] = 0.0;
      B[1][2*i] = 0.0;B[1][2*i+1] = d_dy_i;
      B[2][2*i] = d_dy_i;B[2][2*i+1] = d_dx_i;
    }

    tildeD[0] = 2.0*gp_weight[p]*J_p*eta[p];
    tildeD[1] = 2.0*gp_weight[p]*J_p*eta[p];
    tildeD[2] =       gp_weight[p]*J_p*eta[p];

    /* form Bt tildeD B */
    /*
    Ke_ij = Bt_ik . D_kl . B_lj
          = B_ki . D_kl . B_lj
          = B_ki . D_kk . B_kj
    */
    for (i = 0; i < 8; i++) {
      for (j = 0; j < 8; j++) {
        for (k = 0; k < 3; k++) { /* Note D is diagonal for stokes */
          Ke[i+8*j] += B[k][i]*tildeD[k]*B[k][j];
        }
      }
    }
  }
}

static void BForm_Grad(PetscScalar Ke[],PetscScalar coords[])
{
  PetscScalar gp_xi[GAUSS_POINTS][NSD],gp_weight[GAUSS_POINTS];
  PetscScalar Ni_p[NODES_PER_EL],GNi_p[NSD][NODES_PER_EL],GNx_p[NSD][NODES_PER_EL];
  PetscScalar J_p,fac;
  PetscInt    p,i,j,di,ngp;

  /* define quadrature rule */
  CreateGaussQuadrature(&ngp,gp_xi,gp_weight);

  /* evaluate bilinear form */
  for (p = 0; p < ngp; p++) {
    EvaluateBasis_Q1(gp_xi[p],Ni_p);
    EvaluateBasisDerivatives_Q1(gp_xi[p],GNi_p);
    EvaluateDerivatives(GNi_p,GNx_p,coords,&J_p);
    fac = gp_weight[p]*J_p;

    for (i = 0; i < NODES_PER_EL; i++) { /* u nodes */
      for (di = 0; di < NSD; di++) { /* u dofs */
        for (j = 0; j < 4; j++) {  /* p nodes, p dofs = 1 (ie no loop) */
          PetscInt IJ;
          IJ = map_wIwDI_uJuDJ(i,di,NODES_PER_EL,2,j,0,NODES_PER_EL,1);

          Ke[IJ] -= GNx_p[di][i]*Ni_p[j]*fac;
        }
      }
    }
  }
}

static void BForm_Div(PetscScalar De[],PetscScalar coords[])
{
  PetscScalar Ge[U_DOFS*NODES_PER_EL*P_DOFS*NODES_PER_EL];
  PetscInt    i,j,nr_g,nc_g;

  PetscMemzero(Ge,sizeof(Ge));
  BForm_Grad(Ge,coords);

  nr_g = U_DOFS*NODES_PER_EL;
  nc_g = P_DOFS*NODES_PER_EL;

  for (i = 0; i < nr_g; i++) {
    for (j = 0; j < nc_g; j++) {
      De[nr_g*j+i] = Ge[nc_g*i+j];
    }
  }
}

static void BForm_Stabilisation(PetscScalar Ke[],PetscScalar coords[],PetscScalar eta[])
{
  PetscScalar gp_xi[GAUSS_POINTS][NSD],gp_weight[GAUSS_POINTS];
  PetscScalar Ni_p[NODES_PER_EL],GNi_p[NSD][NODES_PER_EL],GNx_p[NSD][NODES_PER_EL];
  PetscScalar J_p,fac,eta_avg;
  PetscInt    p,i,j,ngp;

  /* define quadrature rule */
  CreateGaussQuadrature(&ngp,gp_xi,gp_weight);

  /* evaluate bilinear form */
  for (p = 0; p < ngp; p++) {
    EvaluateBasis_Q1(gp_xi[p],Ni_p);
    EvaluateBasisDerivatives_Q1(gp_xi[p],GNi_p);
    EvaluateDerivatives(GNi_p,GNx_p,coords,&J_p);
    fac = gp_weight[p]*J_p;

    for (i = 0; i < NODES_PER_EL; i++) {
      for (j = 0; j < NODES_PER_EL; j++) {
        Ke[NODES_PER_EL*i+j] -= fac*(Ni_p[i]*Ni_p[j]-0.0625);
      }
    }
  }

  /* scale */
  eta_avg = 0.0;
  for (p = 0; p < ngp; p++) eta_avg += eta[p];
  eta_avg = (1.0/((PetscScalar)ngp))*eta_avg;
  fac     = 1.0/eta_avg;
  for (i = 0; i < NODES_PER_EL; i++) {
    for (j = 0; j < NODES_PER_EL; j++) {
      Ke[NODES_PER_EL*i+j] = fac*Ke[NODES_PER_EL*i+j];
    }
  }
}

static void BForm_ScaledMassMatrix(PetscScalar Ke[],PetscScalar coords[],PetscScalar eta[])
{
  PetscScalar gp_xi[GAUSS_POINTS][NSD],gp_weight[GAUSS_POINTS];
  PetscScalar Ni_p[NODES_PER_EL],GNi_p[NSD][NODES_PER_EL],GNx_p[NSD][NODES_PER_EL];
  PetscScalar J_p,fac,eta_avg;
  PetscInt    p,i,j,ngp;

  /* define quadrature rule */
  CreateGaussQuadrature(&ngp,gp_xi,gp_weight);

  /* evaluate bilinear form */
  for (p = 0; p < ngp; p++) {
    EvaluateBasis_Q1(gp_xi[p],Ni_p);
    EvaluateBasisDerivatives_Q1(gp_xi[p],GNi_p);
    EvaluateDerivatives(GNi_p,GNx_p,coords,&J_p);
    fac = gp_weight[p]*J_p;

    for (i = 0; i < NODES_PER_EL; i++) {
      for (j = 0; j < NODES_PER_EL; j++) {
        Ke[NODES_PER_EL*i+j] -= fac*Ni_p[i]*Ni_p[j];
      }
    }
  }

  /* scale */
  eta_avg = 0.0;
  for (p = 0; p < ngp; p++) eta_avg += eta[p];
  eta_avg = (1.0/((PetscScalar)ngp))*eta_avg;
  fac     = 1.0/eta_avg;
  for (i = 0; i < NODES_PER_EL; i++) {
    for (j = 0; j < NODES_PER_EL; j++) {
      Ke[NODES_PER_EL*i+j] *= fac;
    }
  }
}

static void LForm_MomentumRHS(PetscScalar Fe[],PetscScalar coords[],PetscScalar fx[],PetscScalar fy[])
{
  PetscScalar gp_xi[GAUSS_POINTS][NSD],gp_weight[GAUSS_POINTS];
  PetscScalar Ni_p[NODES_PER_EL],GNi_p[NSD][NODES_PER_EL],GNx_p[NSD][NODES_PER_EL];
  PetscScalar J_p,fac;
  PetscInt    p,i,ngp;

  /* define quadrature rule */
  CreateGaussQuadrature(&ngp,gp_xi,gp_weight);

  /* evaluate linear form */
  for (p = 0; p < ngp; p++) {
    EvaluateBasis_Q1(gp_xi[p],Ni_p);
    EvaluateBasisDerivatives_Q1(gp_xi[p],GNi_p);
    EvaluateDerivatives(GNi_p,GNx_p,coords,&J_p);
    fac = gp_weight[p]*J_p;

    for (i = 0; i < NODES_PER_EL; i++) {
      Fe[NSD*i]    = 0.0;
      Fe[NSD*i+1] -= fac*Ni_p[i]*fy[p];
    }
  }
}

static PetscErrorCode GetElementCoords(const PetscScalar _coords[],const PetscInt e2n[],PetscScalar el_coords[])
{
  PetscInt i,d;
  PetscFunctionBeginUser;
  /* get coords for the element */
  for (i=0; i<4; i++) {
    for (d=0; d<NSD; d++) {
      el_coords[NSD*i+d] = _coords[NSD * e2n[i] + d];
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode AssembleStokes_A(Mat A,DM stokes_da,DM quadrature)
{
  DM                     cda;
  Vec                    coords;
  const PetscScalar      *_coords;
  PetscInt               u_eqn[NODES_PER_EL*U_DOFS]; /* 2 degrees of freedom */
  PetscInt               p_eqn[NODES_PER_EL*P_DOFS]; /* 1 degrees of freedom */
  PetscInt               nel,npe,eidx;
  const PetscInt         *element_list;
  PetscScalar            Ae[NODES_PER_EL*U_DOFS*NODES_PER_EL*U_DOFS];
  PetscScalar            Ge[NODES_PER_EL*U_DOFS*NODES_PER_EL*P_DOFS];
  PetscScalar            De[NODES_PER_EL*P_DOFS*NODES_PER_EL*U_DOFS];
  PetscScalar            Ce[NODES_PER_EL*P_DOFS*NODES_PER_EL*P_DOFS];
  PetscScalar            el_coords[NODES_PER_EL*NSD];
  PetscScalar            *q_eta,*prop_eta;

  PetscFunctionBeginUser;
  PetscCall(MatZeroEntries(A));
  /* setup for coords */
  PetscCall(DMGetCoordinateDM(stokes_da,&cda));
  PetscCall(DMGetCoordinatesLocal(stokes_da,&coords));
  PetscCall(VecGetArrayRead(coords,&_coords));

  /* setup for coefficients */
  PetscCall(DMSwarmGetField(quadrature,"eta_q",NULL,NULL,(void**)&q_eta));

  PetscCall(DMDAGetElements(stokes_da,&nel,&npe,&element_list));
  for (eidx = 0; eidx < nel; eidx++) {
    const PetscInt *element = &element_list[npe*eidx];

    /* get coords for the element */
    PetscCall(GetElementCoords(_coords,element,el_coords));

    /* get coefficients for the element */
    prop_eta = &q_eta[GAUSS_POINTS * eidx];

    /* initialise element stiffness matrix */
    PetscCall(PetscMemzero(Ae,sizeof(Ae)));
    PetscCall(PetscMemzero(Ge,sizeof(Ge)));
    PetscCall(PetscMemzero(De,sizeof(De)));
    PetscCall(PetscMemzero(Ce,sizeof(Ce)));

    /* form element stiffness matrix */
    BForm_DivT(Ae,el_coords,prop_eta);
    BForm_Grad(Ge,el_coords);
    BForm_Div(De,el_coords);
    BForm_Stabilisation(Ce,el_coords,prop_eta);

    /* insert element matrix into global matrix */
    PetscCall(DMDAGetElementEqnums_up(element,u_eqn,p_eqn));
    PetscCall(MatSetValuesLocal(A,NODES_PER_EL*U_DOFS,u_eqn,NODES_PER_EL*U_DOFS,u_eqn,Ae,ADD_VALUES));
    PetscCall(MatSetValuesLocal(A,NODES_PER_EL*U_DOFS,u_eqn,NODES_PER_EL*P_DOFS,p_eqn,Ge,ADD_VALUES));
    PetscCall(MatSetValuesLocal(A,NODES_PER_EL*P_DOFS,p_eqn,NODES_PER_EL*U_DOFS,u_eqn,De,ADD_VALUES));
    PetscCall(MatSetValuesLocal(A,NODES_PER_EL*P_DOFS,p_eqn,NODES_PER_EL*P_DOFS,p_eqn,Ce,ADD_VALUES));
  }
  PetscCall(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));

  PetscCall(DMSwarmRestoreField(quadrature,"eta_q",NULL,NULL,(void**)&q_eta));
  PetscCall(VecRestoreArrayRead(coords,&_coords));
  PetscFunctionReturn(0);
}

static PetscErrorCode AssembleStokes_PC(Mat A,DM stokes_da,DM quadrature)
{
  DM                     cda;
  Vec                    coords;
  const PetscScalar      *_coords;
  PetscInt               u_eqn[NODES_PER_EL*U_DOFS]; /* 2 degrees of freedom */
  PetscInt               p_eqn[NODES_PER_EL*P_DOFS]; /* 1 degrees of freedom */
  PetscInt               nel,npe,eidx;
  const PetscInt         *element_list;
  PetscScalar            Ae[NODES_PER_EL*U_DOFS*NODES_PER_EL*U_DOFS];
  PetscScalar            Ge[NODES_PER_EL*U_DOFS*NODES_PER_EL*P_DOFS];
  PetscScalar            De[NODES_PER_EL*P_DOFS*NODES_PER_EL*U_DOFS];
  PetscScalar            Ce[NODES_PER_EL*P_DOFS*NODES_PER_EL*P_DOFS];
  PetscScalar            el_coords[NODES_PER_EL*NSD];
  PetscScalar            *q_eta,*prop_eta;

  PetscFunctionBeginUser;
  PetscCall(MatZeroEntries(A));
  /* setup for coords */
  PetscCall(DMGetCoordinateDM(stokes_da,&cda));
  PetscCall(DMGetCoordinatesLocal(stokes_da,&coords));
  PetscCall(VecGetArrayRead(coords,&_coords));

  /* setup for coefficients */
  PetscCall(DMSwarmGetField(quadrature,"eta_q",NULL,NULL,(void**)&q_eta));

  PetscCall(DMDAGetElements(stokes_da,&nel,&npe,&element_list));
  for (eidx = 0; eidx < nel; eidx++) {
    const PetscInt *element = &element_list[npe*eidx];

    /* get coords for the element */
    PetscCall(GetElementCoords(_coords,element,el_coords));

    /* get coefficients for the element */
    prop_eta = &q_eta[GAUSS_POINTS * eidx];

    /* initialise element stiffness matrix */
    PetscCall(PetscMemzero(Ae,sizeof(Ae)));
    PetscCall(PetscMemzero(Ge,sizeof(Ge)));
    PetscCall(PetscMemzero(De,sizeof(De)));
    PetscCall(PetscMemzero(Ce,sizeof(Ce)));

    /* form element stiffness matrix */
    BForm_DivT(Ae,el_coords,prop_eta);
    BForm_Grad(Ge,el_coords);
    BForm_ScaledMassMatrix(Ce,el_coords,prop_eta);

    /* insert element matrix into global matrix */
    PetscCall(DMDAGetElementEqnums_up(element,u_eqn,p_eqn));
    PetscCall(MatSetValuesLocal(A,NODES_PER_EL*U_DOFS,u_eqn,NODES_PER_EL*U_DOFS,u_eqn,Ae,ADD_VALUES));
    PetscCall(MatSetValuesLocal(A,NODES_PER_EL*U_DOFS,u_eqn,NODES_PER_EL*P_DOFS,p_eqn,Ge,ADD_VALUES));
    PetscCall(MatSetValuesLocal(A,NODES_PER_EL*P_DOFS,p_eqn,NODES_PER_EL*U_DOFS,u_eqn,De,ADD_VALUES));
    PetscCall(MatSetValuesLocal(A,NODES_PER_EL*P_DOFS,p_eqn,NODES_PER_EL*P_DOFS,p_eqn,Ce,ADD_VALUES));
  }
  PetscCall(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));

  PetscCall(DMSwarmRestoreField(quadrature,"eta_q",NULL,NULL,(void**)&q_eta));
  PetscCall(VecRestoreArrayRead(coords,&_coords));

  PetscFunctionReturn(0);
}

static PetscErrorCode AssembleStokes_RHS(Vec F,DM stokes_da,DM quadrature)
{
  DM                     cda;
  Vec                    coords;
  const PetscScalar      *_coords;
  PetscInt               u_eqn[NODES_PER_EL*U_DOFS]; /* 2 degrees of freedom */
  PetscInt               p_eqn[NODES_PER_EL*P_DOFS]; /* 1 degrees of freedom */
  PetscInt               nel,npe,eidx,i;
  const PetscInt         *element_list;
  PetscScalar            Fe[NODES_PER_EL*U_DOFS];
  PetscScalar            He[NODES_PER_EL*P_DOFS];
  PetscScalar            el_coords[NODES_PER_EL*NSD];
  PetscScalar            *q_rhs,*prop_fy;
  Vec                    local_F;
  PetscScalar            *LA_F;

  PetscFunctionBeginUser;
  PetscCall(VecZeroEntries(F));
  /* setup for coords */
  PetscCall(DMGetCoordinateDM(stokes_da,&cda));
  PetscCall(DMGetCoordinatesLocal(stokes_da,&coords));
  PetscCall(VecGetArrayRead(coords,&_coords));

  /* setup for coefficients */
  PetscCall(DMSwarmGetField(quadrature,"rho_q",NULL,NULL,(void**)&q_rhs));

  /* get access to the vector */
  PetscCall(DMGetLocalVector(stokes_da,&local_F));
  PetscCall(VecZeroEntries(local_F));
  PetscCall(VecGetArray(local_F,&LA_F));

  PetscCall(DMDAGetElements(stokes_da,&nel,&npe,&element_list));
  for (eidx = 0; eidx < nel; eidx++) {
    const PetscInt *element = &element_list[npe*eidx];

    /* get coords for the element */
    PetscCall(GetElementCoords(_coords,element,el_coords));

    /* get coefficients for the element */
    prop_fy = &q_rhs[GAUSS_POINTS * eidx];

    /* initialise element stiffness matrix */
    PetscCall(PetscMemzero(Fe,sizeof(Fe)));
    PetscCall(PetscMemzero(He,sizeof(He)));

    /* form element stiffness matrix */
    LForm_MomentumRHS(Fe,el_coords,NULL,prop_fy);

    /* insert element matrix into global matrix */
    PetscCall(DMDAGetElementEqnums_up(element,u_eqn,p_eqn));

    for (i=0; i<NODES_PER_EL*U_DOFS; i++) {
      LA_F[ u_eqn[i] ] += Fe[i];
    }
  }
  PetscCall(DMSwarmRestoreField(quadrature,"rho_q",NULL,NULL,(void**)&q_rhs));
  PetscCall(VecRestoreArrayRead(coords,&_coords));

  PetscCall(VecRestoreArray(local_F,&LA_F));
  PetscCall(DMLocalToGlobalBegin(stokes_da,local_F,ADD_VALUES,F));
  PetscCall(DMLocalToGlobalEnd(stokes_da,local_F,ADD_VALUES,F));
  PetscCall(DMRestoreLocalVector(stokes_da,&local_F));

  PetscFunctionReturn(0);
}

PetscErrorCode DMSwarmPICInsertPointsCellwise(DM dm,DM dmc,PetscInt e,PetscInt npoints,PetscReal xi[],PetscBool proximity_initialization)
{
  PetscInt          dim,nel,npe,q,k,d,ncurr;
  const PetscInt    *element_list;
  Vec               coor;
  const PetscScalar *_coor;
  PetscReal         **basis,*elcoor,*xp;
  PetscReal         *swarm_coor;
  PetscInt          *swarm_cellid;

  PetscFunctionBeginUser;
  PetscCall(DMGetDimension(dm,&dim));
  PetscCall(DMDAGetElements(dmc,&nel,&npe,&element_list));

  PetscCall(PetscMalloc1(dim*npoints,&xp));
  PetscCall(PetscMalloc1(dim*npe,&elcoor));
  PetscCall(PetscMalloc1(npoints,&basis));
  for (q=0; q<npoints; q++) {
    PetscCall(PetscMalloc1(npe,&basis[q]));

    switch (dim) {
      case 1:
        basis[q][0] = 0.5*(1.0 - xi[dim*q+0]);
        basis[q][1] = 0.5*(1.0 + xi[dim*q+0]);
        break;
      case 2:
        basis[q][0] = 0.25*(1.0 - xi[dim*q+0])*(1.0 - xi[dim*q+1]);
        basis[q][1] = 0.25*(1.0 + xi[dim*q+0])*(1.0 - xi[dim*q+1]);
        basis[q][2] = 0.25*(1.0 + xi[dim*q+0])*(1.0 + xi[dim*q+1]);
        basis[q][3] = 0.25*(1.0 - xi[dim*q+0])*(1.0 + xi[dim*q+1]);
        break;

      case 3:
        basis[q][0] = 0.125*(1.0 - xi[dim*q+0])*(1.0 - xi[dim*q+1])*(1.0 - xi[dim*q+2]);
        basis[q][1] = 0.125*(1.0 + xi[dim*q+0])*(1.0 - xi[dim*q+1])*(1.0 - xi[dim*q+2]);
        basis[q][2] = 0.125*(1.0 + xi[dim*q+0])*(1.0 + xi[dim*q+1])*(1.0 - xi[dim*q+2]);
        basis[q][3] = 0.125*(1.0 - xi[dim*q+0])*(1.0 + xi[dim*q+1])*(1.0 - xi[dim*q+2]);
        basis[q][4] = 0.125*(1.0 - xi[dim*q+0])*(1.0 - xi[dim*q+1])*(1.0 + xi[dim*q+2]);
        basis[q][5] = 0.125*(1.0 + xi[dim*q+0])*(1.0 - xi[dim*q+1])*(1.0 + xi[dim*q+2]);
        basis[q][6] = 0.125*(1.0 + xi[dim*q+0])*(1.0 + xi[dim*q+1])*(1.0 + xi[dim*q+2]);
        basis[q][7] = 0.125*(1.0 - xi[dim*q+0])*(1.0 + xi[dim*q+1])*(1.0 + xi[dim*q+2]);
        break;
    }
  }

  PetscCall(DMGetCoordinatesLocal(dmc,&coor));
  PetscCall(VecGetArrayRead(coor,&_coor));
  /* compute and store the coordinates for the new points */
  {
    const PetscInt *element = &element_list[npe*e];

    for (k=0; k<npe; k++) {
      for (d=0; d<dim; d++) {
        elcoor[dim*k+d] = PetscRealPart(_coor[ dim*element[k] + d ]);
      }
    }
    for (q=0; q<npoints; q++) {
      for (d=0; d<dim; d++) {
        xp[dim*q+d] = 0.0;
      }
      for (k=0; k<npe; k++) {
        for (d=0; d<dim; d++) {
          xp[dim*q+d] += basis[q][k] * elcoor[dim*k+d];
        }
      }
    }
  }
  PetscCall(VecRestoreArrayRead(coor,&_coor));
  PetscCall(DMDARestoreElements(dmc,&nel,&npe,&element_list));

  PetscCall(DMSwarmGetLocalSize(dm,&ncurr));
  PetscCall(DMSwarmAddNPoints(dm,npoints));

  if (proximity_initialization) {
    PetscInt  *nnlist;
    PetscReal *coor_q,*coor_qn;
    PetscInt  npoints_e,*plist_e;

    PetscCall(DMSwarmSortGetPointsPerCell(dm,e,&npoints_e,&plist_e));

    PetscCall(PetscMalloc1(npoints,&nnlist));
    /* find nearest neighour points in this cell */
    PetscCall(DMSwarmGetField(dm,DMSwarmPICField_coor,NULL,NULL,(void**)&swarm_coor));
    PetscCall(DMSwarmGetField(dm,DMSwarmPICField_cellid,NULL,NULL,(void**)&swarm_cellid));
    for (q=0; q<npoints; q++) {
      PetscInt  qn,nearest_neighour = -1;
      PetscReal sep,min_sep = PETSC_MAX_REAL;

      coor_q = &xp[dim*q];
      for (qn=0; qn<npoints_e; qn++) {
        coor_qn = &swarm_coor[dim*plist_e[qn]];
        sep = 0.0;
        for (d=0; d<dim; d++) {
          sep += (coor_q[d]-coor_qn[d])*(coor_q[d]-coor_qn[d]);
        }
        if (sep < min_sep) {
          nearest_neighour = plist_e[qn];
          min_sep = sep;
        }
      }
      PetscCheck(nearest_neighour != -1,PETSC_COMM_SELF,PETSC_ERR_USER,"Cell %D is empty - cannot initialize using nearest neighbours",e);
      nnlist[q] = nearest_neighour;
    }
    PetscCall(DMSwarmRestoreField(dm,DMSwarmPICField_cellid,NULL,NULL,(void**)&swarm_cellid));
    PetscCall(DMSwarmRestoreField(dm,DMSwarmPICField_coor,NULL,NULL,(void**)&swarm_coor));

    /* copies the nearest neighbour (nnlist[q]) into the new slot (ncurr+q) */
    for (q=0; q<npoints; q++) {
      PetscCall(DMSwarmCopyPoint(dm,nnlist[q],ncurr+q));
    }
    PetscCall(DMSwarmGetField(dm,DMSwarmPICField_coor,NULL,NULL,(void**)&swarm_coor));
    PetscCall(DMSwarmGetField(dm,DMSwarmPICField_cellid,NULL,NULL,(void**)&swarm_cellid));
    for (q=0; q<npoints; q++) {
      /* set the coordinates */
      for (d=0; d<dim; d++) {
        swarm_coor[dim*(ncurr+q)+d] = xp[dim*q+d];
      }
      /* set the cell index */
      swarm_cellid[ncurr+q] = e;
    }
    PetscCall(DMSwarmRestoreField(dm,DMSwarmPICField_cellid,NULL,NULL,(void**)&swarm_cellid));
    PetscCall(DMSwarmRestoreField(dm,DMSwarmPICField_coor,NULL,NULL,(void**)&swarm_coor));

    PetscCall(PetscFree(plist_e));
    PetscCall(PetscFree(nnlist));
  } else {
    PetscCall(DMSwarmGetField(dm,DMSwarmPICField_coor,NULL,NULL,(void**)&swarm_coor));
    PetscCall(DMSwarmGetField(dm,DMSwarmPICField_cellid,NULL,NULL,(void**)&swarm_cellid));
    for (q=0; q<npoints; q++) {
      /* set the coordinates */
      for (d=0; d<dim; d++) {
        swarm_coor[dim*(ncurr+q)+d] = xp[dim*q+d];
      }
      /* set the cell index */
      swarm_cellid[ncurr+q] = e;
    }
    PetscCall(DMSwarmRestoreField(dm,DMSwarmPICField_cellid,NULL,NULL,(void**)&swarm_cellid));
    PetscCall(DMSwarmRestoreField(dm,DMSwarmPICField_coor,NULL,NULL,(void**)&swarm_coor));
  }

  PetscCall(PetscFree(xp));
  PetscCall(PetscFree(elcoor));
  for (q=0; q<npoints; q++) {
    PetscCall(PetscFree(basis[q]));
  }
  PetscCall(PetscFree(basis));
  PetscFunctionReturn(0);
}

PetscErrorCode MaterialPoint_PopulateCell(DM dm_vp,DM dm_mpoint)
{
  PetscInt        _npe,_nel,e,nel;
  const PetscInt  *element;
  DM              dmc;
  PetscQuadrature quadrature;
  const PetscReal *xi;
  PetscInt        npoints_q,cnt,cnt_g;

  PetscFunctionBeginUser;
  PetscCall(DMDAGetElements(dm_vp,&_nel,&_npe,&element));
  nel = _nel;
  PetscCall(DMDARestoreElements(dm_vp,&_nel,&_npe,&element));

  PetscCall(PetscDTGaussTensorQuadrature(2,1,4,-1.0,1.0,&quadrature));
  PetscCall(PetscQuadratureGetData(quadrature,NULL,NULL,&npoints_q,&xi,NULL));
  PetscCall(DMSwarmGetCellDM(dm_mpoint,&dmc));

  PetscCall(DMSwarmSortGetAccess(dm_mpoint));

  cnt = 0;
  for (e=0; e<nel; e++) {
    PetscInt npoints_per_cell;

    PetscCall(DMSwarmSortGetNumberOfPointsPerCell(dm_mpoint,e,&npoints_per_cell));

    if (npoints_per_cell < 12) {
      PetscCall(DMSwarmPICInsertPointsCellwise(dm_mpoint,dm_vp,e,npoints_q,(PetscReal*)xi,PETSC_TRUE));
      cnt++;
    }
  }
  PetscCallMPI(MPI_Allreduce(&cnt,&cnt_g,1,MPIU_INT,MPI_SUM,PETSC_COMM_WORLD));
  if (cnt_g > 0) {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,".... ....pop cont: adjusted %D cells\n",cnt_g));
  }

  PetscCall(DMSwarmSortRestoreAccess(dm_mpoint));
  PetscCall(PetscQuadratureDestroy(&quadrature));
  PetscFunctionReturn(0);
}

PetscErrorCode MaterialPoint_AdvectRK1(DM dm_vp,Vec vp,PetscReal dt,DM dm_mpoint)
{
  Vec               vp_l,coor_l;
  const PetscScalar *LA_vp;
  PetscInt          i,p,e,npoints,nel,npe;
  PetscInt          *mpfield_cell;
  PetscReal         *mpfield_coor;
  const PetscInt    *element_list;
  const PetscInt    *element;
  PetscScalar       xi_p[NSD],Ni[NODES_PER_EL];
  const PetscScalar *LA_coor;
  PetscScalar       dx[NSD];

  PetscFunctionBeginUser;
  PetscCall(DMGetCoordinatesLocal(dm_vp,&coor_l));
  PetscCall(VecGetArrayRead(coor_l,&LA_coor));

  PetscCall(DMGetLocalVector(dm_vp,&vp_l));
  PetscCall(DMGlobalToLocalBegin(dm_vp,vp,INSERT_VALUES,vp_l));
  PetscCall(DMGlobalToLocalEnd(dm_vp,vp,INSERT_VALUES,vp_l));
  PetscCall(VecGetArrayRead(vp_l,&LA_vp));

  PetscCall(DMDAGetElements(dm_vp,&nel,&npe,&element_list));
  PetscCall(DMSwarmGetLocalSize(dm_mpoint,&npoints));
  PetscCall(DMSwarmGetField(dm_mpoint,DMSwarmPICField_coor,NULL,NULL,(void**)&mpfield_coor));
  PetscCall(DMSwarmGetField(dm_mpoint,DMSwarmPICField_cellid,NULL,NULL,(void**)&mpfield_cell));
  for (p=0; p<npoints; p++) {
    PetscReal         *coor_p;
    PetscScalar       vel_n[NSD*NODES_PER_EL],vel_p[NSD];
    const PetscScalar *x0;
    const PetscScalar *x2;

    e       = mpfield_cell[p];
    coor_p  = &mpfield_coor[NSD*p];
    element = &element_list[NODES_PER_EL*e];

    /* compute local coordinates: (xp-x0)/dx = (xip+1)/2 */
    x0 = &LA_coor[NSD*element[0]];
    x2 = &LA_coor[NSD*element[2]];

    dx[0] = x2[0] - x0[0];
    dx[1] = x2[1] - x0[1];

    xi_p[0] = 2.0 * (coor_p[0] - x0[0])/dx[0] - 1.0;
    xi_p[1] = 2.0 * (coor_p[1] - x0[1])/dx[1] - 1.0;
    PetscCheck(PetscRealPart(xi_p[0]) >= -1.0-PETSC_SMALL,PETSC_COMM_SELF,PETSC_ERR_SUP,"value (xi) too small %1.4e [e=%" PetscInt_FMT "]",(double)PetscRealPart(xi_p[0]),e);
    PetscCheck(PetscRealPart(xi_p[0]) <=  1.0+PETSC_SMALL,PETSC_COMM_SELF,PETSC_ERR_SUP,"value (xi) too large %1.4e [e=%" PetscInt_FMT "]",(double)PetscRealPart(xi_p[0]),e);
    PetscCheck(PetscRealPart(xi_p[1]) >= -1.0-PETSC_SMALL,PETSC_COMM_SELF,PETSC_ERR_SUP,"value (eta) too small %1.4e [e=%" PetscInt_FMT "]",(double)PetscRealPart(xi_p[1]),e);
    PetscCheck(PetscRealPart(xi_p[1]) <=  1.0+PETSC_SMALL,PETSC_COMM_SELF,PETSC_ERR_SUP,"value (eta) too large %1.4e [e=%" PetscInt_FMT "]",(double)PetscRealPart(xi_p[1]),e);

    /* evaluate basis functions */
    EvaluateBasis_Q1(xi_p,Ni);

    /* get cell nodal velocities */
    for (i=0; i<NODES_PER_EL; i++) {
      PetscInt nid;

      nid = element[i];
      vel_n[NSD*i+0] = LA_vp[(NSD+1)*nid+0];
      vel_n[NSD*i+1] = LA_vp[(NSD+1)*nid+1];
    }

    /* interpolate velocity */
    vel_p[0] = vel_p[1] = 0.0;
    for (i=0; i<NODES_PER_EL; i++) {
      vel_p[0] += Ni[i] * vel_n[NSD*i+0];
      vel_p[1] += Ni[i] * vel_n[NSD*i+1];
    }

    coor_p[0] += dt * PetscRealPart(vel_p[0]);
    coor_p[1] += dt * PetscRealPart(vel_p[1]);
  }

  PetscCall(DMSwarmRestoreField(dm_mpoint,DMSwarmPICField_cellid,NULL,NULL,(void**)&mpfield_cell));
  PetscCall(DMSwarmRestoreField(dm_mpoint,DMSwarmPICField_coor,NULL,NULL,(void**)&mpfield_coor));
  PetscCall(DMDARestoreElements(dm_vp,&nel,&npe,&element_list));
  PetscCall(VecRestoreArrayRead(vp_l,&LA_vp));
  PetscCall(DMRestoreLocalVector(dm_vp,&vp_l));
  PetscCall(VecRestoreArrayRead(coor_l,&LA_coor));
  PetscFunctionReturn(0);
}

PetscErrorCode MaterialPoint_Interpolate(DM dm,Vec eta_v,Vec rho_v,DM dm_quadrature)
{
  Vec            eta_l,rho_l;
  PetscScalar    *_eta_l,*_rho_l;
  PetscInt       nqp,npe,nel;
  PetscScalar    qp_xi[GAUSS_POINTS][NSD];
  PetscScalar    qp_weight[GAUSS_POINTS];
  PetscInt       q,k,e;
  PetscScalar    Ni[GAUSS_POINTS][NODES_PER_EL];
  const PetscInt *element_list;
  PetscReal      *q_eta,*q_rhs;

  PetscFunctionBeginUser;
  /* define quadrature rule */
  CreateGaussQuadrature(&nqp,qp_xi,qp_weight);
  for (q=0; q<nqp; q++) {
    EvaluateBasis_Q1(qp_xi[q],Ni[q]);
  }

  PetscCall(DMGetLocalVector(dm,&eta_l));
  PetscCall(DMGetLocalVector(dm,&rho_l));

  PetscCall(DMGlobalToLocalBegin(dm,eta_v,INSERT_VALUES,eta_l));
  PetscCall(DMGlobalToLocalEnd(dm,eta_v,INSERT_VALUES,eta_l));
  PetscCall(DMGlobalToLocalBegin(dm,rho_v,INSERT_VALUES,rho_l));
  PetscCall(DMGlobalToLocalEnd(dm,rho_v,INSERT_VALUES,rho_l));

  PetscCall(VecGetArray(eta_l,&_eta_l));
  PetscCall(VecGetArray(rho_l,&_rho_l));

  PetscCall(DMSwarmGetField(dm_quadrature,"eta_q",NULL,NULL,(void**)&q_eta));
  PetscCall(DMSwarmGetField(dm_quadrature,"rho_q",NULL,NULL,(void**)&q_rhs));

  PetscCall(DMDAGetElements(dm,&nel,&npe,&element_list));
  for (e=0; e<nel; e++) {
    PetscScalar    eta_field_e[NODES_PER_EL];
    PetscScalar    rho_field_e[NODES_PER_EL];
    const PetscInt *element = &element_list[4*e];

    for (k=0; k<NODES_PER_EL; k++) {
      eta_field_e[k] = _eta_l[ element[k] ];
      rho_field_e[k] = _rho_l[ element[k] ];
    }

    for (q=0; q<nqp; q++) {
      PetscScalar eta_q,rho_q;

      eta_q = rho_q = 0.0;
      for (k=0; k<NODES_PER_EL; k++) {
        eta_q += Ni[q][k] * eta_field_e[k];
        rho_q += Ni[q][k] * rho_field_e[k];
      }

      q_eta[nqp*e+q] = PetscRealPart(eta_q);
      q_rhs[nqp*e+q] = PetscRealPart(rho_q);
    }
  }
  PetscCall(DMDARestoreElements(dm,&nel,&npe,&element_list));

  PetscCall(DMSwarmRestoreField(dm_quadrature,"rho_q",NULL,NULL,(void**)&q_rhs));
  PetscCall(DMSwarmRestoreField(dm_quadrature,"eta_q",NULL,NULL,(void**)&q_eta));

  PetscCall(VecRestoreArray(rho_l,&_rho_l));
  PetscCall(VecRestoreArray(eta_l,&_eta_l));
  PetscCall(DMRestoreLocalVector(dm,&rho_l));
  PetscCall(DMRestoreLocalVector(dm,&eta_l));
  PetscFunctionReturn(0);
}

static PetscErrorCode SolveTimeDepStokes(PetscInt mx,PetscInt my)
{
  DM                     dm_stokes,dm_coeff;
  PetscInt               u_dof,p_dof,dof,stencil_width;
  Mat                    A,B;
  PetscInt               nel_local;
  Vec                    eta_v,rho_v;
  Vec                    f,X;
  KSP                    ksp;
  PC                     pc;
  char                   filename[PETSC_MAX_PATH_LEN];
  DM                     dms_quadrature,dms_mpoint;
  PetscInt               nel,npe,npoints;
  const PetscInt         *element_list;
  PetscInt               tk,nt,dump_freq;
  PetscReal              dt,dt_max = 0.0;
  PetscReal              vx[2],vy[2],max_v = 0.0,max_v_step,dh;
  const char             *fieldnames[] = { "eta" , "rho" };
  Vec                    *pfields;
  PetscInt               ppcell = 1;
  PetscReal              time,delta_eta = 1.0;
  PetscBool              randomize_coords = PETSC_FALSE;
  PetscReal              randomize_fac = 0.25;
  PetscBool              no_view = PETSC_FALSE;
  PetscBool              isbddc;

  PetscFunctionBeginUser;
  /*
    Generate the DMDA for the velocity and pressure spaces.
    We use Q1 elements for both fields.
    The Q1 FE basis on a regular mesh has a 9-point stencil (DMDA_STENCIL_BOX)
    The number of nodes in each direction is mx+1, my+1
  */
  u_dof         = U_DOFS; /* Vx, Vy - velocities */
  p_dof         = P_DOFS; /* p - pressure */
  dof           = u_dof + p_dof;
  stencil_width = 1;
  PetscCall(DMDACreate2d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DMDA_STENCIL_BOX,mx+1,my+1,PETSC_DECIDE,PETSC_DECIDE,dof,stencil_width,NULL,NULL,&dm_stokes));
  PetscCall(DMDASetElementType(dm_stokes,DMDA_ELEMENT_Q1));
  PetscCall(DMSetMatType(dm_stokes,MATAIJ));
  PetscCall(DMSetFromOptions(dm_stokes));
  PetscCall(DMSetUp(dm_stokes));
  PetscCall(DMDASetFieldName(dm_stokes,0,"ux"));
  PetscCall(DMDASetFieldName(dm_stokes,1,"uy"));
  PetscCall(DMDASetFieldName(dm_stokes,2,"p"));

  /* unit box [0,0.9142] x [0,1] */
  PetscCall(DMDASetUniformCoordinates(dm_stokes,0.0,0.9142,0.0,1.0,0.,0.));
  dh = 1.0/((PetscReal)(mx));

  /* Get local number of elements */
  {
    PetscCall(DMDAGetElements(dm_stokes,&nel,&npe,&element_list));

    nel_local = nel;

    PetscCall(DMDARestoreElements(dm_stokes,&nel,&npe,&element_list));
  }

  /* Create DMDA for representing scalar fields */
  PetscCall(DMDACreateCompatibleDMDA(dm_stokes,1,&dm_coeff));

  /* Create the swarm for storing quadrature point values */
  PetscCall(DMCreate(PETSC_COMM_WORLD,&dms_quadrature));
  PetscCall(DMSetType(dms_quadrature,DMSWARM));
  PetscCall(DMSetDimension(dms_quadrature,2));
  PetscCall(PetscObjectSetName((PetscObject) dms_quadrature, "Quadrature Swarm"));

  /* Register fields for viscosity and density on the quadrature points */
  PetscCall(DMSwarmRegisterPetscDatatypeField(dms_quadrature,"eta_q",1,PETSC_REAL));
  PetscCall(DMSwarmRegisterPetscDatatypeField(dms_quadrature,"rho_q",1,PETSC_REAL));
  PetscCall(DMSwarmFinalizeFieldRegister(dms_quadrature));
  PetscCall(DMSwarmSetLocalSizes(dms_quadrature,nel_local * GAUSS_POINTS,0));

  /* Create the material point swarm */
  PetscCall(DMCreate(PETSC_COMM_WORLD,&dms_mpoint));
  PetscCall(DMSetType(dms_mpoint,DMSWARM));
  PetscCall(DMSetDimension(dms_mpoint,2));
  PetscCall(PetscObjectSetName((PetscObject) dms_mpoint, "Material Point Swarm"));

  /* Configure the material point swarm to be of type Particle-In-Cell */
  PetscCall(DMSwarmSetType(dms_mpoint,DMSWARM_PIC));

  /*
     Specify the DM to use for point location and projections
     within the context of a PIC scheme
  */
  PetscCall(DMSwarmSetCellDM(dms_mpoint,dm_coeff));

  /* Register fields for viscosity and density */
  PetscCall(DMSwarmRegisterPetscDatatypeField(dms_mpoint,"eta",1,PETSC_REAL));
  PetscCall(DMSwarmRegisterPetscDatatypeField(dms_mpoint,"rho",1,PETSC_REAL));
  PetscCall(DMSwarmFinalizeFieldRegister(dms_mpoint));

  PetscCall(PetscOptionsGetInt(NULL,NULL,"-ppcell",&ppcell,NULL));
  PetscCall(DMSwarmSetLocalSizes(dms_mpoint,nel_local * ppcell,100));

  /*
    Layout the material points in space using the cell DM.
    Particle coordinates are defined by cell wise using different methods.
    - DMSWARMPIC_LAYOUT_GAUSS defines particles coordinates at the positions
                              corresponding to a Gauss quadrature rule with
                              ppcell points in each direction.
    - DMSWARMPIC_LAYOUT_REGULAR defines particle coordinates at the centoid of
                                ppcell x ppcell quadralaterals defined within the
                                reference element.
    - DMSWARMPIC_LAYOUT_SUBDIVISION defines particles coordinates at the centroid
                                    of each quadralateral obtained by sub-dividing
                                    the reference element cell ppcell times.
  */
  PetscCall(DMSwarmInsertPointsUsingCellDM(dms_mpoint,DMSWARMPIC_LAYOUT_SUBDIVISION,ppcell));

  /*
    Defne a high resolution layer of material points across the material interface
  */
  {
    PetscInt  npoints_dir_x[2];
    PetscReal min[2],max[2];

    npoints_dir_x[0] = (PetscInt)(0.9142/(0.05*dh));
    npoints_dir_x[1] = (PetscInt)((0.25-0.15)/(0.05*dh));
    min[0] = 0.0;  max[0] = 0.9142;
    min[1] = 0.05; max[1] = 0.35;
    PetscCall(DMSwarmSetPointsUniformCoordinates(dms_mpoint,min,max,npoints_dir_x,ADD_VALUES));
  }

  /*
    Define a high resolution layer of material points near the surface of the domain
    to deal with weakly compressible Q1-Q1 elements. These elements "self compact"
    when applied to buouyancy driven flow. The error in div(u) is O(h).
  */
  {
    PetscInt  npoints_dir_x[2];
    PetscReal min[2],max[2];

    npoints_dir_x[0] = (PetscInt)(0.9142/(0.25*dh));
    npoints_dir_x[1] = (PetscInt)(3.0*dh/(0.25*dh));
    min[0] = 0.0;          max[0] = 0.9142;
    min[1] = 1.0 - 3.0*dh; max[1] = 1.0-0.0001;
    PetscCall(DMSwarmSetPointsUniformCoordinates(dms_mpoint,min,max,npoints_dir_x,ADD_VALUES));
  }

  PetscCall(DMView(dms_mpoint,PETSC_VIEWER_STDOUT_WORLD));

  /* Define initial material properties on each particle in the material point swarm */
  PetscCall(PetscOptionsGetReal(NULL,NULL,"-delta_eta",&delta_eta,NULL));
  PetscCall(PetscOptionsGetBool(NULL,NULL,"-randomize_coords",&randomize_coords,NULL));
  PetscCall(PetscOptionsGetReal(NULL,NULL,"-randomize_fac",&randomize_fac,NULL));
  PetscCheck(randomize_fac <= 1.0,PETSC_COMM_WORLD,PETSC_ERR_USER,"The value of -randomize_fac should be <= 1.0");
  {
    PetscReal   *array_x,*array_e,*array_r;
    PetscInt    p;
    PetscRandom r;
    PetscMPIInt rank;

    PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));

    PetscCall(PetscRandomCreate(PETSC_COMM_SELF,&r));
    PetscCall(PetscRandomSetInterval(r,-randomize_fac*dh,randomize_fac*dh));
    PetscCall(PetscRandomSetSeed(r,(unsigned long)rank));
    PetscCall(PetscRandomSeed(r));

    PetscCall(DMDAGetElements(dm_stokes,&nel,&npe,&element_list));

    /*
       Fetch the registered data from the material point DMSwarm.
       The fields "eta" and "rho" were registered by this example.
       The field identified by the the variable DMSwarmPICField_coor
       was registered by the DMSwarm implementation when the function
         DMSwarmSetType(dms_mpoint,DMSWARM_PIC)
       was called. The returned array defines the coordinates of each
       material point in the point swarm.
    */
    PetscCall(DMSwarmGetField(dms_mpoint,DMSwarmPICField_coor,NULL,NULL,(void**)&array_x));
    PetscCall(DMSwarmGetField(dms_mpoint,"eta",               NULL,NULL,(void**)&array_e));
    PetscCall(DMSwarmGetField(dms_mpoint,"rho",               NULL,NULL,(void**)&array_r));

    PetscCall(DMSwarmGetLocalSize(dms_mpoint,&npoints));
    for (p = 0; p < npoints; p++) {
      PetscReal x_p[2],rr[2];

      if (randomize_coords) {
        PetscCall(PetscRandomGetValueReal(r,&rr[0]));
        PetscCall(PetscRandomGetValueReal(r,&rr[1]));
        array_x[2*p + 0] += rr[0];
        array_x[2*p + 1] += rr[1];
      }

      /* Get the coordinates of point, p */
      x_p[0] = array_x[2*p + 0];
      x_p[1] = array_x[2*p + 1];

       if (x_p[1] < (0.2 + 0.02*PetscCosReal(PETSC_PI*x_p[0]/0.9142))) {
         /* Material properties below the interface */
         array_e[p] = 1.0 * (1.0/delta_eta);
         array_r[p] = 0.0;
       } else {
         /* Material properties above the interface */
         array_e[p] = 1.0;
         array_r[p] = 1.0;
       }
    }

    /*
       Restore the fetched data fields from the material point DMSwarm.
       Calling the Restore function invalidates the points array_r, array_e, array_x
       by setting them to NULL.
    */
    PetscCall(DMSwarmRestoreField(dms_mpoint,"rho",NULL,NULL,(void**)&array_r));
    PetscCall(DMSwarmRestoreField(dms_mpoint,"eta",NULL,NULL,(void**)&array_e));
    PetscCall(DMSwarmRestoreField(dms_mpoint,DMSwarmPICField_coor,NULL,NULL,(void**)&array_x));

    PetscCall(DMDARestoreElements(dm_stokes,&nel,&npe,&element_list));
    PetscCall(PetscRandomDestroy(&r));
  }

  /*
     If the particle coordinates where randomly shifted, they may have crossed into another
     element, or into another sub-domain. To account for this we call the Migrate function.
  */
  if (randomize_coords) {
    PetscCall(DMSwarmMigrate(dms_mpoint,PETSC_TRUE));
  }

  PetscCall(PetscOptionsGetBool(NULL,NULL,"-no_view",&no_view,NULL));
  if (!no_view) {
    PetscCall(DMSwarmViewXDMF(dms_mpoint,"ic_coeff_dms.xmf"));
  }

  /* project the swarm properties */
  PetscCall(DMSwarmProjectFields(dms_mpoint,2,fieldnames,&pfields,PETSC_FALSE));
  eta_v = pfields[0];
  rho_v = pfields[1];
  PetscCall(PetscObjectSetName((PetscObject)eta_v,"eta"));
  PetscCall(PetscObjectSetName((PetscObject)rho_v,"rho"));
  PetscCall(MaterialPoint_Interpolate(dm_coeff,eta_v,rho_v,dms_quadrature));

  /* view projected coefficients eta and rho */
  if (!no_view) {
    PetscViewer viewer;

    PetscCall(PetscViewerCreate(PETSC_COMM_WORLD,&viewer));
    PetscCall(PetscViewerSetType(viewer,PETSCVIEWERVTK));
    PetscCall(PetscViewerFileSetMode(viewer,FILE_MODE_WRITE));
    PetscCall(PetscViewerFileSetName(viewer,"ic_coeff_dmda.vts"));
    PetscCall(VecView(eta_v,viewer));
    PetscCall(VecView(rho_v,viewer));
    PetscCall(PetscViewerDestroy(&viewer));
  }

  PetscCall(DMCreateMatrix(dm_stokes,&A));
  PetscCall(DMCreateMatrix(dm_stokes,&B));
  PetscCall(DMCreateGlobalVector(dm_stokes,&f));
  PetscCall(DMCreateGlobalVector(dm_stokes,&X));

  PetscCall(AssembleStokes_A(A,dm_stokes,dms_quadrature));
  PetscCall(AssembleStokes_PC(B,dm_stokes,dms_quadrature));
  PetscCall(AssembleStokes_RHS(f,dm_stokes,dms_quadrature));

  PetscCall(DMDAApplyBoundaryConditions(dm_stokes,A,f));
  PetscCall(DMDAApplyBoundaryConditions(dm_stokes,B,NULL));

  PetscCall(KSPCreate(PETSC_COMM_WORLD,&ksp));
  PetscCall(KSPSetOptionsPrefix(ksp,"stokes_"));
  PetscCall(KSPSetDM(ksp,dm_stokes));
  PetscCall(KSPSetDMActive(ksp,PETSC_FALSE));
  PetscCall(KSPSetOperators(ksp,A,B));
  PetscCall(KSPSetFromOptions(ksp));
  PetscCall(KSPGetPC(ksp,&pc));
  PetscCall(PetscObjectTypeCompare((PetscObject)pc,PCBDDC,&isbddc));
  if (isbddc) {
    PetscCall(KSPSetOperators(ksp,A,A));
  }

  /* Define u-v-p indices for fieldsplit */
  {
    PC             pc;
    const PetscInt ufields[] = {0,1},pfields[1] = {2};

    PetscCall(KSPGetPC(ksp,&pc));
    PetscCall(PCFieldSplitSetBlockSize(pc,3));
    PetscCall(PCFieldSplitSetFields(pc,"u",2,ufields,ufields));
    PetscCall(PCFieldSplitSetFields(pc,"p",1,pfields,pfields));
  }

  /* If using a fieldsplit preconditioner, attach a DMDA to the velocity split so that geometric multigrid can be used */
  {
    PC        pc,pc_u;
    KSP       *sub_ksp,ksp_u;
    PetscInt  nsplits;
    DM        dm_u;
    PetscBool is_pcfs;

    PetscCall(KSPGetPC(ksp,&pc));

    is_pcfs = PETSC_FALSE;
    PetscCall(PetscObjectTypeCompare((PetscObject)pc,PCFIELDSPLIT,&is_pcfs));

    if (is_pcfs) {
      PetscCall(KSPSetUp(ksp));
      PetscCall(KSPGetPC(ksp,&pc));
      PetscCall(PCFieldSplitGetSubKSP(pc,&nsplits,&sub_ksp));
      ksp_u = sub_ksp[0];
      PetscCall(PetscFree(sub_ksp));

      if (nsplits == 2) {
        PetscCall(DMDACreateCompatibleDMDA(dm_stokes,2,&dm_u));

        PetscCall(KSPSetDM(ksp_u,dm_u));
        PetscCall(KSPSetDMActive(ksp_u,PETSC_FALSE));
        PetscCall(DMDestroy(&dm_u));

        /* enforce galerkin coarse grids be used */
        PetscCall(KSPGetPC(ksp_u,&pc_u));
        PetscCall(PCMGSetGalerkin(pc_u,PC_MG_GALERKIN_PMAT));
      }
    }
  }

  dump_freq = 10;
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-dump_freq",&dump_freq,NULL));
  nt = 10;
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-nt",&nt,NULL));
  time = 0.0;
  for (tk=1; tk<=nt; tk++) {

    PetscCall(PetscPrintf(PETSC_COMM_WORLD,".... assemble\n"));
    PetscCall(AssembleStokes_A(A,dm_stokes,dms_quadrature));
    PetscCall(AssembleStokes_PC(B,dm_stokes,dms_quadrature));
    PetscCall(AssembleStokes_RHS(f,dm_stokes,dms_quadrature));

    PetscCall(PetscPrintf(PETSC_COMM_WORLD,".... bc imposition\n"));
    PetscCall(DMDAApplyBoundaryConditions(dm_stokes,A,f));
    PetscCall(DMDAApplyBoundaryConditions(dm_stokes,B,NULL));

    PetscCall(PetscPrintf(PETSC_COMM_WORLD,".... solve\n"));
    PetscCall(KSPSetOperators(ksp,A, isbddc ? A : B));
    PetscCall(KSPSolve(ksp,f,X));

    PetscCall(VecStrideMax(X,0,NULL,&vx[1]));
    PetscCall(VecStrideMax(X,1,NULL,&vy[1]));
    PetscCall(VecStrideMin(X,0,NULL,&vx[0]));
    PetscCall(VecStrideMin(X,1,NULL,&vy[0]));

    max_v_step = PetscMax(vx[0],vx[1]);
    max_v_step = PetscMax(max_v_step,vy[0]);
    max_v_step = PetscMax(max_v_step,vy[1]);
    max_v = PetscMax(max_v,max_v_step);

    dt_max = 2.0;
    dt = 0.5 * (dh / max_v_step);
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,".... max v %1.4e , dt %1.4e : [total] max v %1.4e , dt_max %1.4e\n",(double)max_v_step,(double)dt,(double)max_v,(double)dt_max));
    dt = PetscMin(dt_max,dt);

    /* advect */
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,".... advect\n"));
    PetscCall(MaterialPoint_AdvectRK1(dm_stokes,X,dt,dms_mpoint));

    /* migrate */
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,".... migrate\n"));
    PetscCall(DMSwarmMigrate(dms_mpoint,PETSC_TRUE));

    /* update cell population */
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,".... populate cells\n"));
    PetscCall(MaterialPoint_PopulateCell(dm_stokes,dms_mpoint));

    /* update coefficients on quadrature points */
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,".... project\n"));
    PetscCall(DMSwarmProjectFields(dms_mpoint,2,fieldnames,&pfields,PETSC_TRUE));
    eta_v = pfields[0];
    rho_v = pfields[1];
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,".... interp\n"));
    PetscCall(MaterialPoint_Interpolate(dm_coeff,eta_v,rho_v,dms_quadrature));

    if (tk%dump_freq == 0) {
      PetscViewer viewer;

      PetscCall(PetscPrintf(PETSC_COMM_WORLD,".... write XDMF, VTS\n"));
      PetscCall(PetscSNPrintf(filename,PETSC_MAX_PATH_LEN-1,"step%.4D_coeff_dms.xmf",tk));
      PetscCall(DMSwarmViewXDMF(dms_mpoint,filename));

      PetscCall(PetscSNPrintf(filename,PETSC_MAX_PATH_LEN-1,"step%.4D_vp_dm.vts",tk));
      PetscCall(PetscViewerCreate(PETSC_COMM_WORLD,&viewer));
      PetscCall(PetscViewerSetType(viewer,PETSCVIEWERVTK));
      PetscCall(PetscViewerFileSetMode(viewer,FILE_MODE_WRITE));
      PetscCall(PetscViewerFileSetName(viewer,filename));
      PetscCall(VecView(X,viewer));
      PetscCall(PetscViewerDestroy(&viewer));
    }
    time += dt;
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"step %D : time %1.2e \n",tk,(double)time));
  }

  PetscCall(KSPDestroy(&ksp));
  PetscCall(VecDestroy(&X));
  PetscCall(VecDestroy(&f));
  PetscCall(MatDestroy(&A));
  PetscCall(MatDestroy(&B));
  PetscCall(VecDestroy(&eta_v));
  PetscCall(VecDestroy(&rho_v));
  PetscCall(PetscFree(pfields));

  PetscCall(DMDestroy(&dms_mpoint));
  PetscCall(DMDestroy(&dms_quadrature));
  PetscCall(DMDestroy(&dm_coeff));
  PetscCall(DMDestroy(&dm_stokes));
  PetscFunctionReturn(0);
}

/*
 <sequential run>
 ./ex70 -stokes_ksp_type fgmres -stokes_pc_type fieldsplit -stokes_pc_fieldsplit_block_size 3 -stokes_pc_fieldsplit_type SYMMETRIC_MULTIPLICATIVE -stokes_pc_fieldsplit_0_fields 0,1 -stokes_pc_fieldsplit_1_fields 2 -stokes_fieldsplit_0_ksp_type preonly -stokes_fieldsplit_0_pc_type lu -stokes_fieldsplit_1_ksp_type preonly -stokes_fieldsplit_1_pc_type lu  -mx 80 -my 80  -stokes_ksp_converged_reason  -dump_freq 25  -stokes_ksp_rtol 1.0e-8 -build_twosided allreduce  -ppcell 2 -nt 4000 -delta_eta 1.0 -randomize_coords
*/
int main(int argc,char **args)
{
  PetscInt       mx,my;
  PetscBool      set = PETSC_FALSE;

  PetscCall(PetscInitialize(&argc,&args,(char*)0,help));
  mx = my = 10;
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-mx",&mx,NULL));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-my",&my,NULL));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-mxy",&mx,&set));
  if (set) {
    my = mx;
  }
  PetscCall(SolveTimeDepStokes(mx,my));
  PetscCall(PetscFinalize());
  return 0;
}

/* -------------------------- helpers for boundary conditions -------------------------------- */
static PetscErrorCode BCApplyZero_EAST(DM da,PetscInt d_idx,Mat A,Vec b)
{
  DM                     cda;
  Vec                    coords;
  PetscInt               si,sj,nx,ny,i,j;
  PetscInt               M,N;
  DMDACoor2d             **_coords;
  const PetscInt         *g_idx;
  PetscInt               *bc_global_ids;
  PetscScalar            *bc_vals;
  PetscInt               nbcs;
  PetscInt               n_dofs;
  ISLocalToGlobalMapping ltogm;

  PetscFunctionBeginUser;
  PetscCall(DMGetLocalToGlobalMapping(da,&ltogm));
  PetscCall(ISLocalToGlobalMappingGetIndices(ltogm,&g_idx));

  PetscCall(DMGetCoordinateDM(da,&cda));
  PetscCall(DMGetCoordinatesLocal(da,&coords));
  PetscCall(DMDAVecGetArray(cda,coords,&_coords));
  PetscCall(DMDAGetGhostCorners(cda,&si,&sj,0,&nx,&ny,0));
  PetscCall(DMDAGetInfo(da,0,&M,&N,0,0,0,0,&n_dofs,0,0,0,0,0));

  PetscCall(PetscMalloc1(ny*n_dofs,&bc_global_ids));
  PetscCall(PetscMalloc1(ny*n_dofs,&bc_vals));

  /* init the entries to -1 so VecSetValues will ignore them */
  for (i = 0; i < ny*n_dofs; i++) bc_global_ids[i] = -1;

  i = nx-1;
  for (j = 0; j < ny; j++) {
    PetscInt local_id;

    local_id = i+j*nx;

    bc_global_ids[j] = g_idx[n_dofs*local_id+d_idx];

    bc_vals[j] =  0.0;
  }
  PetscCall(ISLocalToGlobalMappingRestoreIndices(ltogm,&g_idx));
  nbcs = 0;
  if ((si+nx) == (M)) nbcs = ny;

  if (b) {
    PetscCall(VecSetValues(b,nbcs,bc_global_ids,bc_vals,INSERT_VALUES));
    PetscCall(VecAssemblyBegin(b));
    PetscCall(VecAssemblyEnd(b));
  }
  if (A) {
    PetscCall(MatZeroRowsColumns(A,nbcs,bc_global_ids,1.0,0,0));
  }

  PetscCall(PetscFree(bc_vals));
  PetscCall(PetscFree(bc_global_ids));

  PetscCall(DMDAVecRestoreArray(cda,coords,&_coords));
  PetscFunctionReturn(0);
}

static PetscErrorCode BCApplyZero_WEST(DM da,PetscInt d_idx,Mat A,Vec b)
{
  DM                     cda;
  Vec                    coords;
  PetscInt               si,sj,nx,ny,i,j;
  PetscInt               M,N;
  DMDACoor2d             **_coords;
  const PetscInt         *g_idx;
  PetscInt               *bc_global_ids;
  PetscScalar            *bc_vals;
  PetscInt               nbcs;
  PetscInt               n_dofs;
  ISLocalToGlobalMapping ltogm;

  PetscFunctionBeginUser;
  PetscCall(DMGetLocalToGlobalMapping(da,&ltogm));
  PetscCall(ISLocalToGlobalMappingGetIndices(ltogm,&g_idx));

  PetscCall(DMGetCoordinateDM(da,&cda));
  PetscCall(DMGetCoordinatesLocal(da,&coords));
  PetscCall(DMDAVecGetArray(cda,coords,&_coords));
  PetscCall(DMDAGetGhostCorners(cda,&si,&sj,0,&nx,&ny,0));
  PetscCall(DMDAGetInfo(da,0,&M,&N,0,0,0,0,&n_dofs,0,0,0,0,0));

  PetscCall(PetscMalloc1(ny*n_dofs,&bc_global_ids));
  PetscCall(PetscMalloc1(ny*n_dofs,&bc_vals));

  /* init the entries to -1 so VecSetValues will ignore them */
  for (i = 0; i < ny*n_dofs; i++) bc_global_ids[i] = -1;

  i = 0;
  for (j = 0; j < ny; j++) {
    PetscInt local_id;

    local_id = i+j*nx;

    bc_global_ids[j] = g_idx[n_dofs*local_id+d_idx];

    bc_vals[j] =  0.0;
  }
  PetscCall(ISLocalToGlobalMappingRestoreIndices(ltogm,&g_idx));
  nbcs = 0;
  if (si == 0) nbcs = ny;

  if (b) {
    PetscCall(VecSetValues(b,nbcs,bc_global_ids,bc_vals,INSERT_VALUES));
    PetscCall(VecAssemblyBegin(b));
    PetscCall(VecAssemblyEnd(b));
  }

  if (A) {
    PetscCall(MatZeroRowsColumns(A,nbcs,bc_global_ids,1.0,0,0));
  }

  PetscCall(PetscFree(bc_vals));
  PetscCall(PetscFree(bc_global_ids));

  PetscCall(DMDAVecRestoreArray(cda,coords,&_coords));
  PetscFunctionReturn(0);
}

static PetscErrorCode BCApplyZero_NORTH(DM da,PetscInt d_idx,Mat A,Vec b)
{
  DM                     cda;
  Vec                    coords;
  PetscInt               si,sj,nx,ny,i,j;
  PetscInt               M,N;
  DMDACoor2d             **_coords;
  const PetscInt         *g_idx;
  PetscInt               *bc_global_ids;
  PetscScalar            *bc_vals;
  PetscInt               nbcs;
  PetscInt               n_dofs;
  ISLocalToGlobalMapping ltogm;

  PetscFunctionBeginUser;
  PetscCall(DMGetLocalToGlobalMapping(da,&ltogm));
  PetscCall(ISLocalToGlobalMappingGetIndices(ltogm,&g_idx));

  PetscCall(DMGetCoordinateDM(da,&cda));
  PetscCall(DMGetCoordinatesLocal(da,&coords));
  PetscCall(DMDAVecGetArray(cda,coords,&_coords));
  PetscCall(DMDAGetGhostCorners(cda,&si,&sj,0,&nx,&ny,0));
  PetscCall(DMDAGetInfo(da,0,&M,&N,0,0,0,0,&n_dofs,0,0,0,0,0));

  PetscCall(PetscMalloc1(nx,&bc_global_ids));
  PetscCall(PetscMalloc1(nx,&bc_vals));

  /* init the entries to -1 so VecSetValues will ignore them */
  for (i = 0; i < nx; i++) bc_global_ids[i] = -1;

  j = ny-1;
  for (i = 0; i < nx; i++) {
    PetscInt local_id;

    local_id = i+j*nx;

    bc_global_ids[i] = g_idx[n_dofs*local_id+d_idx];

    bc_vals[i] =  0.0;
  }
  PetscCall(ISLocalToGlobalMappingRestoreIndices(ltogm,&g_idx));
  nbcs = 0;
  if ((sj+ny) == (N)) nbcs = nx;

  if (b) {
    PetscCall(VecSetValues(b,nbcs,bc_global_ids,bc_vals,INSERT_VALUES));
    PetscCall(VecAssemblyBegin(b));
    PetscCall(VecAssemblyEnd(b));
  }
  if (A) {
    PetscCall(MatZeroRowsColumns(A,nbcs,bc_global_ids,1.0,NULL,NULL));
  }

  PetscCall(PetscFree(bc_vals));
  PetscCall(PetscFree(bc_global_ids));

  PetscCall(DMDAVecRestoreArray(cda,coords,&_coords));
  PetscFunctionReturn(0);
}

static PetscErrorCode BCApplyZero_SOUTH(DM da,PetscInt d_idx,Mat A,Vec b)
{
  DM                     cda;
  Vec                    coords;
  PetscInt               si,sj,nx,ny,i,j;
  PetscInt               M,N;
  DMDACoor2d             **_coords;
  const PetscInt         *g_idx;
  PetscInt               *bc_global_ids;
  PetscScalar            *bc_vals;
  PetscInt               nbcs;
  PetscInt               n_dofs;
  ISLocalToGlobalMapping ltogm;

  PetscFunctionBeginUser;
  PetscCall(DMGetLocalToGlobalMapping(da,&ltogm));
  PetscCall(ISLocalToGlobalMappingGetIndices(ltogm,&g_idx));

  PetscCall(DMGetCoordinateDM(da,&cda));
  PetscCall(DMGetCoordinatesLocal(da,&coords));
  PetscCall(DMDAVecGetArray(cda,coords,&_coords));
  PetscCall(DMDAGetGhostCorners(cda,&si,&sj,0,&nx,&ny,0));
  PetscCall(DMDAGetInfo(da,0,&M,&N,0,0,0,0,&n_dofs,0,0,0,0,0));

  PetscCall(PetscMalloc1(nx,&bc_global_ids));
  PetscCall(PetscMalloc1(nx,&bc_vals));

  /* init the entries to -1 so VecSetValues will ignore them */
  for (i = 0; i < nx; i++) bc_global_ids[i] = -1;

  j = 0;
  for (i = 0; i < nx; i++) {
    PetscInt local_id;

    local_id = i+j*nx;

    bc_global_ids[i] = g_idx[n_dofs*local_id+d_idx];

    bc_vals[i] =  0.0;
  }
  PetscCall(ISLocalToGlobalMappingRestoreIndices(ltogm,&g_idx));
  nbcs = 0;
  if (sj == 0) nbcs = nx;

  if (b) {
    PetscCall(VecSetValues(b,nbcs,bc_global_ids,bc_vals,INSERT_VALUES));
    PetscCall(VecAssemblyBegin(b));
    PetscCall(VecAssemblyEnd(b));
  }
  if (A) {
    PetscCall(MatZeroRowsColumns(A,nbcs,bc_global_ids,1.0,0,0));
  }

  PetscCall(PetscFree(bc_vals));
  PetscCall(PetscFree(bc_global_ids));

  PetscCall(DMDAVecRestoreArray(cda,coords,&_coords));
  PetscFunctionReturn(0);
}

/*
 Impose free slip boundary conditions on the left/right faces: u_i n_i = 0, tau_{ij} t_j = 0
 Impose no slip boundray conditions on the top/bottom faces:   u_i n_i = 0, u_i t_i = 0
*/
static PetscErrorCode DMDAApplyBoundaryConditions(DM dm_stokes,Mat A,Vec f)
{
  PetscFunctionBeginUser;
  PetscCall(BCApplyZero_NORTH(dm_stokes,0,A,f));
  PetscCall(BCApplyZero_NORTH(dm_stokes,1,A,f));
  PetscCall(BCApplyZero_EAST(dm_stokes,0,A,f));
  PetscCall(BCApplyZero_SOUTH(dm_stokes,0,A,f));
  PetscCall(BCApplyZero_SOUTH(dm_stokes,1,A,f));
  PetscCall(BCApplyZero_WEST(dm_stokes,0,A,f));
  PetscFunctionReturn(0);
}

/*TEST

   test:
     suffix: 1
     args: -no_view
     requires: !complex double
     filter: grep -v atomic
     filter_output: grep -v atomic
   test:
     suffix: 1_matis
     requires: !complex double
     args: -no_view -dm_mat_type is
     filter: grep -v atomic
     filter_output: grep -v atomic
   testset:
     nsize: 4
     requires: !complex double
     args: -no_view -dm_mat_type is -stokes_ksp_type fetidp -mx 80 -my 80 -stokes_ksp_converged_reason -stokes_ksp_rtol 1.0e-8 -ppcell 2 -nt 4 -randomize_coords -stokes_ksp_error_if_not_converged
     filter: grep -v atomic
     filter_output: grep -v atomic
     test:
       suffix: fetidp
       args: -stokes_fetidp_bddc_pc_bddc_coarse_redundant_pc_type svd
     test:
       suffix: fetidp_lumped
       args: -stokes_fetidp_bddc_pc_bddc_coarse_redundant_pc_type svd -stokes_fetidp_pc_lumped -stokes_fetidp_bddc_pc_bddc_dirichlet_pc_type none -stokes_fetidp_bddc_pc_bddc_switch_static
     test:
       suffix: fetidp_saddlepoint
       args: -stokes_ksp_fetidp_saddlepoint -stokes_fetidp_ksp_type cg -stokes_ksp_norm_type natural -stokes_fetidp_pc_fieldsplit_schur_fact_type diag -stokes_fetidp_fieldsplit_p_pc_type bjacobi -stokes_fetidp_fieldsplit_lag_ksp_type preonly -stokes_fetidp_fieldsplit_p_ksp_type preonly -stokes_ksp_fetidp_pressure_field 2 -stokes_fetidp_pc_fieldsplit_schur_scale -1
     test:
       suffix: fetidp_saddlepoint_lumped
       args: -stokes_ksp_fetidp_saddlepoint -stokes_fetidp_ksp_type cg -stokes_ksp_norm_type natural -stokes_fetidp_pc_fieldsplit_schur_fact_type diag -stokes_fetidp_fieldsplit_p_pc_type bjacobi -stokes_fetidp_fieldsplit_lag_ksp_type preonly -stokes_fetidp_fieldsplit_p_ksp_type preonly -stokes_ksp_fetidp_pressure_field 2 -stokes_fetidp_pc_fieldsplit_schur_scale -1 -stokes_fetidp_bddc_pc_bddc_dirichlet_pc_type none -stokes_fetidp_bddc_pc_bddc_switch_static -stokes_fetidp_pc_lumped
TEST*/
