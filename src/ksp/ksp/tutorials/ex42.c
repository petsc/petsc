static char help[] = "Solves the incompressible, variable viscosity stokes equation in 3d using Q1Q1 elements, \n\
stabilized with Bochev's polynomial projection method. Note that implementation here assumes \n\
all boundaries are free-slip, i.e. zero normal flow and zero tangential stress \n\
     -mx : number elements in x-direction \n\
     -my : number elements in y-direction \n\
     -mz : number elements in z-direction \n\
     -stokes_ksp_monitor_blocks : active monitor for each component u,v,w,p \n\
     -model : defines viscosity and forcing function. Choose either: 0 (isoviscous), 1 (manufactured-broken), 2 (solcx-2d), 3 (sinker), 4 (subdomain jumps) \n";

/* Contributed by Dave May */

#include <petscksp.h>
#include <petscdmda.h>

#define PROFILE_TIMING
#define ASSEMBLE_LOWER_TRIANGULAR

#define NSD          3 /* number of spatial dimensions */
#define NODES_PER_EL 8 /* nodes per element */
#define U_DOFS       3 /* degrees of freedom per velocity node */
#define P_DOFS       1 /* degrees of freedom per pressure node */
#define GAUSS_POINTS 8

/* Gauss point based evaluation */
typedef struct {
  PetscScalar gp_coords[NSD * GAUSS_POINTS];
  PetscScalar eta[GAUSS_POINTS];
  PetscScalar fx[GAUSS_POINTS];
  PetscScalar fy[GAUSS_POINTS];
  PetscScalar fz[GAUSS_POINTS];
  PetscScalar hc[GAUSS_POINTS];
} GaussPointCoefficients;

typedef struct {
  PetscScalar u_dof;
  PetscScalar v_dof;
  PetscScalar w_dof;
  PetscScalar p_dof;
} StokesDOF;

typedef struct _p_CellProperties *CellProperties;
struct _p_CellProperties {
  PetscInt                ncells;
  PetscInt                mx, my, mz;
  PetscInt                sex, sey, sez;
  GaussPointCoefficients *gpc;
};

/* elements */
PetscErrorCode CellPropertiesCreate(DM da_stokes, CellProperties *C)
{
  CellProperties cells;
  PetscInt       mx, my, mz, sex, sey, sez;

  PetscFunctionBeginUser;
  PetscCall(PetscNew(&cells));

  PetscCall(DMDAGetElementsCorners(da_stokes, &sex, &sey, &sez));
  PetscCall(DMDAGetElementsSizes(da_stokes, &mx, &my, &mz));
  cells->mx     = mx;
  cells->my     = my;
  cells->mz     = mz;
  cells->ncells = mx * my * mz;
  cells->sex    = sex;
  cells->sey    = sey;
  cells->sez    = sez;

  PetscCall(PetscMalloc1(mx * my * mz, &cells->gpc));

  *C = cells;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode CellPropertiesDestroy(CellProperties *C)
{
  CellProperties cells;

  PetscFunctionBeginUser;
  if (!C) PetscFunctionReturn(PETSC_SUCCESS);
  cells = *C;
  PetscCall(PetscFree(cells->gpc));
  PetscCall(PetscFree(cells));
  *C = NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode CellPropertiesGetCell(CellProperties C, PetscInt II, PetscInt J, PetscInt K, GaussPointCoefficients **G)
{
  PetscFunctionBeginUser;
  *G = &C->gpc[(II - C->sex) + (J - C->sey) * C->mx + (K - C->sez) * C->mx * C->my];
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* FEM routines */
/*
 Element: Local basis function ordering
 1-----2
 |     |
 |     |
 0-----3
 */
static void ShapeFunctionQ13D_Evaluate(PetscScalar _xi[], PetscScalar Ni[])
{
  PetscReal xi   = PetscRealPart(_xi[0]);
  PetscReal eta  = PetscRealPart(_xi[1]);
  PetscReal zeta = PetscRealPart(_xi[2]);

  Ni[0] = 0.125 * (1.0 - xi) * (1.0 - eta) * (1.0 - zeta);
  Ni[1] = 0.125 * (1.0 - xi) * (1.0 + eta) * (1.0 - zeta);
  Ni[2] = 0.125 * (1.0 + xi) * (1.0 + eta) * (1.0 - zeta);
  Ni[3] = 0.125 * (1.0 + xi) * (1.0 - eta) * (1.0 - zeta);

  Ni[4] = 0.125 * (1.0 - xi) * (1.0 - eta) * (1.0 + zeta);
  Ni[5] = 0.125 * (1.0 - xi) * (1.0 + eta) * (1.0 + zeta);
  Ni[6] = 0.125 * (1.0 + xi) * (1.0 + eta) * (1.0 + zeta);
  Ni[7] = 0.125 * (1.0 + xi) * (1.0 - eta) * (1.0 + zeta);
}

static void ShapeFunctionQ13D_Evaluate_dxi(PetscScalar _xi[], PetscScalar GNi[][NODES_PER_EL])
{
  PetscReal xi   = PetscRealPart(_xi[0]);
  PetscReal eta  = PetscRealPart(_xi[1]);
  PetscReal zeta = PetscRealPart(_xi[2]);
  /* xi deriv */
  GNi[0][0] = -0.125 * (1.0 - eta) * (1.0 - zeta);
  GNi[0][1] = -0.125 * (1.0 + eta) * (1.0 - zeta);
  GNi[0][2] = 0.125 * (1.0 + eta) * (1.0 - zeta);
  GNi[0][3] = 0.125 * (1.0 - eta) * (1.0 - zeta);

  GNi[0][4] = -0.125 * (1.0 - eta) * (1.0 + zeta);
  GNi[0][5] = -0.125 * (1.0 + eta) * (1.0 + zeta);
  GNi[0][6] = 0.125 * (1.0 + eta) * (1.0 + zeta);
  GNi[0][7] = 0.125 * (1.0 - eta) * (1.0 + zeta);
  /* eta deriv */
  GNi[1][0] = -0.125 * (1.0 - xi) * (1.0 - zeta);
  GNi[1][1] = 0.125 * (1.0 - xi) * (1.0 - zeta);
  GNi[1][2] = 0.125 * (1.0 + xi) * (1.0 - zeta);
  GNi[1][3] = -0.125 * (1.0 + xi) * (1.0 - zeta);

  GNi[1][4] = -0.125 * (1.0 - xi) * (1.0 + zeta);
  GNi[1][5] = 0.125 * (1.0 - xi) * (1.0 + zeta);
  GNi[1][6] = 0.125 * (1.0 + xi) * (1.0 + zeta);
  GNi[1][7] = -0.125 * (1.0 + xi) * (1.0 + zeta);
  /* zeta deriv */
  GNi[2][0] = -0.125 * (1.0 - xi) * (1.0 - eta);
  GNi[2][1] = -0.125 * (1.0 - xi) * (1.0 + eta);
  GNi[2][2] = -0.125 * (1.0 + xi) * (1.0 + eta);
  GNi[2][3] = -0.125 * (1.0 + xi) * (1.0 - eta);

  GNi[2][4] = 0.125 * (1.0 - xi) * (1.0 - eta);
  GNi[2][5] = 0.125 * (1.0 - xi) * (1.0 + eta);
  GNi[2][6] = 0.125 * (1.0 + xi) * (1.0 + eta);
  GNi[2][7] = 0.125 * (1.0 + xi) * (1.0 - eta);
}

static void matrix_inverse_3x3(PetscScalar A[3][3], PetscScalar B[3][3])
{
  PetscScalar t4, t6, t8, t10, t12, t14, t17;

  t4  = A[2][0] * A[0][1];
  t6  = A[2][0] * A[0][2];
  t8  = A[1][0] * A[0][1];
  t10 = A[1][0] * A[0][2];
  t12 = A[0][0] * A[1][1];
  t14 = A[0][0] * A[1][2];
  t17 = 0.1e1 / (t4 * A[1][2] - t6 * A[1][1] - t8 * A[2][2] + t10 * A[2][1] + t12 * A[2][2] - t14 * A[2][1]);

  B[0][0] = (A[1][1] * A[2][2] - A[1][2] * A[2][1]) * t17;
  B[0][1] = -(A[0][1] * A[2][2] - A[0][2] * A[2][1]) * t17;
  B[0][2] = (A[0][1] * A[1][2] - A[0][2] * A[1][1]) * t17;
  B[1][0] = -(-A[2][0] * A[1][2] + A[1][0] * A[2][2]) * t17;
  B[1][1] = (-t6 + A[0][0] * A[2][2]) * t17;
  B[1][2] = -(-t10 + t14) * t17;
  B[2][0] = (-A[2][0] * A[1][1] + A[1][0] * A[2][1]) * t17;
  B[2][1] = -(-t4 + A[0][0] * A[2][1]) * t17;
  B[2][2] = (-t8 + t12) * t17;
}

static void ShapeFunctionQ13D_Evaluate_dx(PetscScalar GNi[][NODES_PER_EL], PetscScalar GNx[][NODES_PER_EL], PetscScalar coords[], PetscScalar *det_J)
{
  PetscScalar J00, J01, J02, J10, J11, J12, J20, J21, J22;
  PetscInt    n;
  PetscScalar iJ[3][3], JJ[3][3];

  J00 = J01 = J02 = 0.0;
  J10 = J11 = J12 = 0.0;
  J20 = J21 = J22 = 0.0;
  for (n = 0; n < NODES_PER_EL; n++) {
    PetscScalar cx = coords[NSD * n + 0];
    PetscScalar cy = coords[NSD * n + 1];
    PetscScalar cz = coords[NSD * n + 2];

    /* J_ij = d(x_j) / d(xi_i) */ /* J_ij = \sum _I GNi[j][I} * x_i */
    J00 = J00 + GNi[0][n] * cx;   /* J_xx */
    J01 = J01 + GNi[0][n] * cy;   /* J_xy = dx/deta */
    J02 = J02 + GNi[0][n] * cz;   /* J_xz = dx/dzeta */

    J10 = J10 + GNi[1][n] * cx; /* J_yx = dy/dxi */
    J11 = J11 + GNi[1][n] * cy; /* J_yy */
    J12 = J12 + GNi[1][n] * cz; /* J_yz */

    J20 = J20 + GNi[2][n] * cx; /* J_zx */
    J21 = J21 + GNi[2][n] * cy; /* J_zy */
    J22 = J22 + GNi[2][n] * cz; /* J_zz */
  }

  JJ[0][0] = J00;
  JJ[0][1] = J01;
  JJ[0][2] = J02;
  JJ[1][0] = J10;
  JJ[1][1] = J11;
  JJ[1][2] = J12;
  JJ[2][0] = J20;
  JJ[2][1] = J21;
  JJ[2][2] = J22;

  matrix_inverse_3x3(JJ, iJ);

  *det_J = J00 * J11 * J22 - J00 * J12 * J21 - J10 * J01 * J22 + J10 * J02 * J21 + J20 * J01 * J12 - J20 * J02 * J11;

  for (n = 0; n < NODES_PER_EL; n++) {
    GNx[0][n] = GNi[0][n] * iJ[0][0] + GNi[1][n] * iJ[0][1] + GNi[2][n] * iJ[0][2];
    GNx[1][n] = GNi[0][n] * iJ[1][0] + GNi[1][n] * iJ[1][1] + GNi[2][n] * iJ[1][2];
    GNx[2][n] = GNi[0][n] * iJ[2][0] + GNi[1][n] * iJ[2][1] + GNi[2][n] * iJ[2][2];
  }
}

static void ConstructGaussQuadrature3D(PetscInt *ngp, PetscScalar gp_xi[][NSD], PetscScalar gp_weight[])
{
  *ngp        = 8;
  gp_xi[0][0] = -0.57735026919;
  gp_xi[0][1] = -0.57735026919;
  gp_xi[0][2] = -0.57735026919;
  gp_xi[1][0] = -0.57735026919;
  gp_xi[1][1] = 0.57735026919;
  gp_xi[1][2] = -0.57735026919;
  gp_xi[2][0] = 0.57735026919;
  gp_xi[2][1] = 0.57735026919;
  gp_xi[2][2] = -0.57735026919;
  gp_xi[3][0] = 0.57735026919;
  gp_xi[3][1] = -0.57735026919;
  gp_xi[3][2] = -0.57735026919;

  gp_xi[4][0] = -0.57735026919;
  gp_xi[4][1] = -0.57735026919;
  gp_xi[4][2] = 0.57735026919;
  gp_xi[5][0] = -0.57735026919;
  gp_xi[5][1] = 0.57735026919;
  gp_xi[5][2] = 0.57735026919;
  gp_xi[6][0] = 0.57735026919;
  gp_xi[6][1] = 0.57735026919;
  gp_xi[6][2] = 0.57735026919;
  gp_xi[7][0] = 0.57735026919;
  gp_xi[7][1] = -0.57735026919;
  gp_xi[7][2] = 0.57735026919;

  gp_weight[0] = 1.0;
  gp_weight[1] = 1.0;
  gp_weight[2] = 1.0;
  gp_weight[3] = 1.0;

  gp_weight[4] = 1.0;
  gp_weight[5] = 1.0;
  gp_weight[6] = 1.0;
  gp_weight[7] = 1.0;
}

/*
 i,j are the element indices
 The unknown is a vector quantity.
 The s[].c is used to indicate the degree of freedom.
 */
static PetscErrorCode DMDAGetElementEqnums3D_up(MatStencil s_u[], MatStencil s_p[], PetscInt i, PetscInt j, PetscInt k)
{
  PetscInt n;

  PetscFunctionBeginUser;
  /* velocity */
  n = 0;
  /* node 0 */
  s_u[n].i = i;
  s_u[n].j = j;
  s_u[n].k = k;
  s_u[n].c = 0;
  n++; /* Vx0 */
  s_u[n].i = i;
  s_u[n].j = j;
  s_u[n].k = k;
  s_u[n].c = 1;
  n++; /* Vy0 */
  s_u[n].i = i;
  s_u[n].j = j;
  s_u[n].k = k;
  s_u[n].c = 2;
  n++; /* Vz0 */

  s_u[n].i = i;
  s_u[n].j = j + 1;
  s_u[n].k = k;
  s_u[n].c = 0;
  n++;
  s_u[n].i = i;
  s_u[n].j = j + 1;
  s_u[n].k = k;
  s_u[n].c = 1;
  n++;
  s_u[n].i = i;
  s_u[n].j = j + 1;
  s_u[n].k = k;
  s_u[n].c = 2;
  n++;

  s_u[n].i = i + 1;
  s_u[n].j = j + 1;
  s_u[n].k = k;
  s_u[n].c = 0;
  n++;
  s_u[n].i = i + 1;
  s_u[n].j = j + 1;
  s_u[n].k = k;
  s_u[n].c = 1;
  n++;
  s_u[n].i = i + 1;
  s_u[n].j = j + 1;
  s_u[n].k = k;
  s_u[n].c = 2;
  n++;

  s_u[n].i = i + 1;
  s_u[n].j = j;
  s_u[n].k = k;
  s_u[n].c = 0;
  n++;
  s_u[n].i = i + 1;
  s_u[n].j = j;
  s_u[n].k = k;
  s_u[n].c = 1;
  n++;
  s_u[n].i = i + 1;
  s_u[n].j = j;
  s_u[n].k = k;
  s_u[n].c = 2;
  n++;

  /* */
  s_u[n].i = i;
  s_u[n].j = j;
  s_u[n].k = k + 1;
  s_u[n].c = 0;
  n++; /* Vx4 */
  s_u[n].i = i;
  s_u[n].j = j;
  s_u[n].k = k + 1;
  s_u[n].c = 1;
  n++; /* Vy4 */
  s_u[n].i = i;
  s_u[n].j = j;
  s_u[n].k = k + 1;
  s_u[n].c = 2;
  n++; /* Vz4 */

  s_u[n].i = i;
  s_u[n].j = j + 1;
  s_u[n].k = k + 1;
  s_u[n].c = 0;
  n++;
  s_u[n].i = i;
  s_u[n].j = j + 1;
  s_u[n].k = k + 1;
  s_u[n].c = 1;
  n++;
  s_u[n].i = i;
  s_u[n].j = j + 1;
  s_u[n].k = k + 1;
  s_u[n].c = 2;
  n++;

  s_u[n].i = i + 1;
  s_u[n].j = j + 1;
  s_u[n].k = k + 1;
  s_u[n].c = 0;
  n++;
  s_u[n].i = i + 1;
  s_u[n].j = j + 1;
  s_u[n].k = k + 1;
  s_u[n].c = 1;
  n++;
  s_u[n].i = i + 1;
  s_u[n].j = j + 1;
  s_u[n].k = k + 1;
  s_u[n].c = 2;
  n++;

  s_u[n].i = i + 1;
  s_u[n].j = j;
  s_u[n].k = k + 1;
  s_u[n].c = 0;
  n++;
  s_u[n].i = i + 1;
  s_u[n].j = j;
  s_u[n].k = k + 1;
  s_u[n].c = 1;
  n++;
  s_u[n].i = i + 1;
  s_u[n].j = j;
  s_u[n].k = k + 1;
  s_u[n].c = 2;
  n++;

  /* pressure */
  n = 0;

  s_p[n].i = i;
  s_p[n].j = j;
  s_p[n].k = k;
  s_p[n].c = 3;
  n++; /* P0 */
  s_p[n].i = i;
  s_p[n].j = j + 1;
  s_p[n].k = k;
  s_p[n].c = 3;
  n++;
  s_p[n].i = i + 1;
  s_p[n].j = j + 1;
  s_p[n].k = k;
  s_p[n].c = 3;
  n++;
  s_p[n].i = i + 1;
  s_p[n].j = j;
  s_p[n].k = k;
  s_p[n].c = 3;
  n++;

  s_p[n].i = i;
  s_p[n].j = j;
  s_p[n].k = k + 1;
  s_p[n].c = 3;
  n++; /* P0 */
  s_p[n].i = i;
  s_p[n].j = j + 1;
  s_p[n].k = k + 1;
  s_p[n].c = 3;
  n++;
  s_p[n].i = i + 1;
  s_p[n].j = j + 1;
  s_p[n].k = k + 1;
  s_p[n].c = 3;
  n++;
  s_p[n].i = i + 1;
  s_p[n].j = j;
  s_p[n].k = k + 1;
  s_p[n].c = 3;
  n++;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode GetElementCoords3D(DMDACoor3d ***coords, PetscInt i, PetscInt j, PetscInt k, PetscScalar el_coord[])
{
  PetscFunctionBeginUser;
  /* get coords for the element */
  el_coord[0] = coords[k][j][i].x;
  el_coord[1] = coords[k][j][i].y;
  el_coord[2] = coords[k][j][i].z;

  el_coord[3] = coords[k][j + 1][i].x;
  el_coord[4] = coords[k][j + 1][i].y;
  el_coord[5] = coords[k][j + 1][i].z;

  el_coord[6] = coords[k][j + 1][i + 1].x;
  el_coord[7] = coords[k][j + 1][i + 1].y;
  el_coord[8] = coords[k][j + 1][i + 1].z;

  el_coord[9]  = coords[k][j][i + 1].x;
  el_coord[10] = coords[k][j][i + 1].y;
  el_coord[11] = coords[k][j][i + 1].z;

  el_coord[12] = coords[k + 1][j][i].x;
  el_coord[13] = coords[k + 1][j][i].y;
  el_coord[14] = coords[k + 1][j][i].z;

  el_coord[15] = coords[k + 1][j + 1][i].x;
  el_coord[16] = coords[k + 1][j + 1][i].y;
  el_coord[17] = coords[k + 1][j + 1][i].z;

  el_coord[18] = coords[k + 1][j + 1][i + 1].x;
  el_coord[19] = coords[k + 1][j + 1][i + 1].y;
  el_coord[20] = coords[k + 1][j + 1][i + 1].z;

  el_coord[21] = coords[k + 1][j][i + 1].x;
  el_coord[22] = coords[k + 1][j][i + 1].y;
  el_coord[23] = coords[k + 1][j][i + 1].z;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode StokesDAGetNodalFields3D(StokesDOF ***field, PetscInt i, PetscInt j, PetscInt k, StokesDOF nodal_fields[])
{
  PetscFunctionBeginUser;
  /* get the nodal fields for u */
  nodal_fields[0].u_dof = field[k][j][i].u_dof;
  nodal_fields[0].v_dof = field[k][j][i].v_dof;
  nodal_fields[0].w_dof = field[k][j][i].w_dof;

  nodal_fields[1].u_dof = field[k][j + 1][i].u_dof;
  nodal_fields[1].v_dof = field[k][j + 1][i].v_dof;
  nodal_fields[1].w_dof = field[k][j + 1][i].w_dof;

  nodal_fields[2].u_dof = field[k][j + 1][i + 1].u_dof;
  nodal_fields[2].v_dof = field[k][j + 1][i + 1].v_dof;
  nodal_fields[2].w_dof = field[k][j + 1][i + 1].w_dof;

  nodal_fields[3].u_dof = field[k][j][i + 1].u_dof;
  nodal_fields[3].v_dof = field[k][j][i + 1].v_dof;
  nodal_fields[3].w_dof = field[k][j][i + 1].w_dof;

  nodal_fields[4].u_dof = field[k + 1][j][i].u_dof;
  nodal_fields[4].v_dof = field[k + 1][j][i].v_dof;
  nodal_fields[4].w_dof = field[k + 1][j][i].w_dof;

  nodal_fields[5].u_dof = field[k + 1][j + 1][i].u_dof;
  nodal_fields[5].v_dof = field[k + 1][j + 1][i].v_dof;
  nodal_fields[5].w_dof = field[k + 1][j + 1][i].w_dof;

  nodal_fields[6].u_dof = field[k + 1][j + 1][i + 1].u_dof;
  nodal_fields[6].v_dof = field[k + 1][j + 1][i + 1].v_dof;
  nodal_fields[6].w_dof = field[k + 1][j + 1][i + 1].w_dof;

  nodal_fields[7].u_dof = field[k + 1][j][i + 1].u_dof;
  nodal_fields[7].v_dof = field[k + 1][j][i + 1].v_dof;
  nodal_fields[7].w_dof = field[k + 1][j][i + 1].w_dof;

  /* get the nodal fields for p */
  nodal_fields[0].p_dof = field[k][j][i].p_dof;
  nodal_fields[1].p_dof = field[k][j + 1][i].p_dof;
  nodal_fields[2].p_dof = field[k][j + 1][i + 1].p_dof;
  nodal_fields[3].p_dof = field[k][j][i + 1].p_dof;

  nodal_fields[4].p_dof = field[k + 1][j][i].p_dof;
  nodal_fields[5].p_dof = field[k + 1][j + 1][i].p_dof;
  nodal_fields[6].p_dof = field[k + 1][j + 1][i + 1].p_dof;
  nodal_fields[7].p_dof = field[k + 1][j][i + 1].p_dof;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscInt ASS_MAP_wIwDI_uJuDJ(PetscInt wi, PetscInt wd, PetscInt w_NPE, PetscInt w_dof, PetscInt ui, PetscInt ud, PetscInt u_NPE, PetscInt u_dof)
{
  PetscInt              ij;
  PETSC_UNUSED PetscInt r, c, nr, nc;

  nr = w_NPE * w_dof;
  nc = u_NPE * u_dof;

  r = w_dof * wi + wd;
  c = u_dof * ui + ud;

  ij = r * nc + c;

  return ij;
}

static PetscErrorCode DMDASetValuesLocalStencil3D_ADD_VALUES(StokesDOF ***fields_F, MatStencil u_eqn[], MatStencil p_eqn[], PetscScalar Fe_u[], PetscScalar Fe_p[])
{
  PetscInt n, II, J, K;

  PetscFunctionBeginUser;
  for (n = 0; n < NODES_PER_EL; n++) {
    II = u_eqn[NSD * n].i;
    J  = u_eqn[NSD * n].j;
    K  = u_eqn[NSD * n].k;

    fields_F[K][J][II].u_dof = fields_F[K][J][II].u_dof + Fe_u[NSD * n];

    II = u_eqn[NSD * n + 1].i;
    J  = u_eqn[NSD * n + 1].j;
    K  = u_eqn[NSD * n + 1].k;

    fields_F[K][J][II].v_dof = fields_F[K][J][II].v_dof + Fe_u[NSD * n + 1];

    II                       = u_eqn[NSD * n + 2].i;
    J                        = u_eqn[NSD * n + 2].j;
    K                        = u_eqn[NSD * n + 2].k;
    fields_F[K][J][II].w_dof = fields_F[K][J][II].w_dof + Fe_u[NSD * n + 2];

    II = p_eqn[n].i;
    J  = p_eqn[n].j;
    K  = p_eqn[n].k;

    fields_F[K][J][II].p_dof = fields_F[K][J][II].p_dof + Fe_p[n];
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static void FormStressOperatorQ13D(PetscScalar Ke[], PetscScalar coords[], PetscScalar eta[])
{
  PetscInt       ngp;
  PetscScalar    gp_xi[GAUSS_POINTS][NSD];
  PetscScalar    gp_weight[GAUSS_POINTS];
  PetscInt       p, i, j, k;
  PetscScalar    GNi_p[NSD][NODES_PER_EL], GNx_p[NSD][NODES_PER_EL];
  PetscScalar    J_p, tildeD[6];
  PetscScalar    B[6][U_DOFS * NODES_PER_EL];
  const PetscInt nvdof = U_DOFS * NODES_PER_EL;

  /* define quadrature rule */
  ConstructGaussQuadrature3D(&ngp, gp_xi, gp_weight);

  /* evaluate integral */
  for (p = 0; p < ngp; p++) {
    ShapeFunctionQ13D_Evaluate_dxi(gp_xi[p], GNi_p);
    ShapeFunctionQ13D_Evaluate_dx(GNi_p, GNx_p, coords, &J_p);

    for (i = 0; i < NODES_PER_EL; i++) {
      PetscScalar d_dx_i = GNx_p[0][i];
      PetscScalar d_dy_i = GNx_p[1][i];
      PetscScalar d_dz_i = GNx_p[2][i];

      B[0][3 * i]     = d_dx_i;
      B[0][3 * i + 1] = 0.0;
      B[0][3 * i + 2] = 0.0;
      B[1][3 * i]     = 0.0;
      B[1][3 * i + 1] = d_dy_i;
      B[1][3 * i + 2] = 0.0;
      B[2][3 * i]     = 0.0;
      B[2][3 * i + 1] = 0.0;
      B[2][3 * i + 2] = d_dz_i;

      B[3][3 * i]     = d_dy_i;
      B[3][3 * i + 1] = d_dx_i;
      B[3][3 * i + 2] = 0.0; /* e_xy */
      B[4][3 * i]     = d_dz_i;
      B[4][3 * i + 1] = 0.0;
      B[4][3 * i + 2] = d_dx_i; /* e_xz */
      B[5][3 * i]     = 0.0;
      B[5][3 * i + 1] = d_dz_i;
      B[5][3 * i + 2] = d_dy_i; /* e_yz */
    }

    tildeD[0] = 2.0 * gp_weight[p] * J_p * eta[p];
    tildeD[1] = 2.0 * gp_weight[p] * J_p * eta[p];
    tildeD[2] = 2.0 * gp_weight[p] * J_p * eta[p];

    tildeD[3] = gp_weight[p] * J_p * eta[p];
    tildeD[4] = gp_weight[p] * J_p * eta[p];
    tildeD[5] = gp_weight[p] * J_p * eta[p];

    /* form Bt tildeD B */
    /*
     Ke_ij = Bt_ik . D_kl . B_lj
     = B_ki . D_kl . B_lj
     = B_ki . D_kk . B_kj
     */
    for (i = 0; i < nvdof; i++) {
      for (j = i; j < nvdof; j++) {
        for (k = 0; k < 6; k++) Ke[i * nvdof + j] += B[k][i] * tildeD[k] * B[k][j];
      }
    }
  }
  /* fill lower triangular part */
#if defined(ASSEMBLE_LOWER_TRIANGULAR)
  for (i = 0; i < nvdof; i++) {
    for (j = i; j < nvdof; j++) Ke[j * nvdof + i] = Ke[i * nvdof + j];
  }
#endif
}

static void FormGradientOperatorQ13D(PetscScalar Ke[], PetscScalar coords[])
{
  PetscInt    ngp;
  PetscScalar gp_xi[GAUSS_POINTS][NSD];
  PetscScalar gp_weight[GAUSS_POINTS];
  PetscInt    p, i, j, di;
  PetscScalar Ni_p[NODES_PER_EL];
  PetscScalar GNi_p[NSD][NODES_PER_EL], GNx_p[NSD][NODES_PER_EL];
  PetscScalar J_p, fac;

  /* define quadrature rule */
  ConstructGaussQuadrature3D(&ngp, gp_xi, gp_weight);

  /* evaluate integral */
  for (p = 0; p < ngp; p++) {
    ShapeFunctionQ13D_Evaluate(gp_xi[p], Ni_p);
    ShapeFunctionQ13D_Evaluate_dxi(gp_xi[p], GNi_p);
    ShapeFunctionQ13D_Evaluate_dx(GNi_p, GNx_p, coords, &J_p);
    fac = gp_weight[p] * J_p;

    for (i = 0; i < NODES_PER_EL; i++) {     /* u nodes */
      for (di = 0; di < NSD; di++) {         /* u dofs */
        for (j = 0; j < NODES_PER_EL; j++) { /* p nodes, p dofs = 1 (ie no loop) */
          PetscInt IJ;
          IJ = ASS_MAP_wIwDI_uJuDJ(i, di, NODES_PER_EL, 3, j, 0, NODES_PER_EL, 1);

          Ke[IJ] = Ke[IJ] - GNx_p[di][i] * Ni_p[j] * fac;
        }
      }
    }
  }
}

static void FormDivergenceOperatorQ13D(PetscScalar De[], PetscScalar coords[])
{
  PetscScalar Ge[U_DOFS * NODES_PER_EL * P_DOFS * NODES_PER_EL];
  PetscInt    i, j;
  PetscInt    nr_g, nc_g;

  PetscCallAbort(PETSC_COMM_SELF, PetscMemzero(Ge, sizeof(Ge)));
  FormGradientOperatorQ13D(Ge, coords);

  nr_g = U_DOFS * NODES_PER_EL;
  nc_g = P_DOFS * NODES_PER_EL;

  for (i = 0; i < nr_g; i++) {
    for (j = 0; j < nc_g; j++) De[nr_g * j + i] = Ge[nc_g * i + j];
  }
}

static void FormStabilisationOperatorQ13D(PetscScalar Ke[], PetscScalar coords[], PetscScalar eta[])
{
  PetscInt    ngp;
  PetscScalar gp_xi[GAUSS_POINTS][NSD];
  PetscScalar gp_weight[GAUSS_POINTS];
  PetscInt    p, i, j;
  PetscScalar Ni_p[NODES_PER_EL];
  PetscScalar GNi_p[NSD][NODES_PER_EL], GNx_p[NSD][NODES_PER_EL];
  PetscScalar J_p, fac, eta_avg;

  /* define quadrature rule */
  ConstructGaussQuadrature3D(&ngp, gp_xi, gp_weight);

  /* evaluate integral */
  for (p = 0; p < ngp; p++) {
    ShapeFunctionQ13D_Evaluate(gp_xi[p], Ni_p);
    ShapeFunctionQ13D_Evaluate_dxi(gp_xi[p], GNi_p);
    ShapeFunctionQ13D_Evaluate_dx(GNi_p, GNx_p, coords, &J_p);
    fac = gp_weight[p] * J_p;
    /*
     for (i = 0; i < NODES_PER_EL; i++) {
     for (j = i; j < NODES_PER_EL; j++) {
     Ke[NODES_PER_EL*i+j] -= fac*(Ni_p[i]*Ni_p[j]-0.015625);
     }
     }
     */

    for (i = 0; i < NODES_PER_EL; i++) {
      for (j = 0; j < NODES_PER_EL; j++) Ke[NODES_PER_EL * i + j] -= fac * (Ni_p[i] * Ni_p[j] - 0.015625);
    }
  }

  /* scale */
  eta_avg = 0.0;
  for (p = 0; p < ngp; p++) eta_avg += eta[p];
  eta_avg = (1.0 / ((PetscScalar)ngp)) * eta_avg;
  fac     = 1.0 / eta_avg;

  /*
   for (i = 0; i < NODES_PER_EL; i++) {
   for (j = i; j < NODES_PER_EL; j++) {
   Ke[NODES_PER_EL*i+j] = fac*Ke[NODES_PER_EL*i+j];
   #if defined(ASSEMBLE_LOWER_TRIANGULAR)
   Ke[NODES_PER_EL*j+i] = Ke[NODES_PER_EL*i+j];
   #endif
   }
   }
   */

  for (i = 0; i < NODES_PER_EL; i++) {
    for (j = 0; j < NODES_PER_EL; j++) Ke[NODES_PER_EL * i + j] = fac * Ke[NODES_PER_EL * i + j];
  }
}

static void FormScaledMassMatrixOperatorQ13D(PetscScalar Ke[], PetscScalar coords[], PetscScalar eta[])
{
  PetscInt    ngp;
  PetscScalar gp_xi[GAUSS_POINTS][NSD];
  PetscScalar gp_weight[GAUSS_POINTS];
  PetscInt    p, i, j;
  PetscScalar Ni_p[NODES_PER_EL];
  PetscScalar GNi_p[NSD][NODES_PER_EL], GNx_p[NSD][NODES_PER_EL];
  PetscScalar J_p, fac, eta_avg;

  /* define quadrature rule */
  ConstructGaussQuadrature3D(&ngp, gp_xi, gp_weight);

  /* evaluate integral */
  for (p = 0; p < ngp; p++) {
    ShapeFunctionQ13D_Evaluate(gp_xi[p], Ni_p);
    ShapeFunctionQ13D_Evaluate_dxi(gp_xi[p], GNi_p);
    ShapeFunctionQ13D_Evaluate_dx(GNi_p, GNx_p, coords, &J_p);
    fac = gp_weight[p] * J_p;

    /*
     for (i = 0; i < NODES_PER_EL; i++) {
     for (j = i; j < NODES_PER_EL; j++) {
     Ke[NODES_PER_EL*i+j] = Ke[NODES_PER_EL*i+j]-fac*Ni_p[i]*Ni_p[j];
     }
     }
     */

    for (i = 0; i < NODES_PER_EL; i++) {
      for (j = 0; j < NODES_PER_EL; j++) Ke[NODES_PER_EL * i + j] = Ke[NODES_PER_EL * i + j] - fac * Ni_p[i] * Ni_p[j];
    }
  }

  /* scale */
  eta_avg = 0.0;
  for (p = 0; p < ngp; p++) eta_avg += eta[p];
  eta_avg = (1.0 / ((PetscScalar)ngp)) * eta_avg;
  fac     = 1.0 / eta_avg;
  /*
   for (i = 0; i < NODES_PER_EL; i++) {
   for (j = i; j < NODES_PER_EL; j++) {
   Ke[NODES_PER_EL*i+j] = fac*Ke[NODES_PER_EL*i+j];
   #if defined(ASSEMBLE_LOWER_TRIANGULAR)
   Ke[NODES_PER_EL*j+i] = Ke[NODES_PER_EL*i+j];
   #endif
   }
   }
   */

  for (i = 0; i < NODES_PER_EL; i++) {
    for (j = 0; j < NODES_PER_EL; j++) Ke[NODES_PER_EL * i + j] = fac * Ke[NODES_PER_EL * i + j];
  }
}

static void FormMomentumRhsQ13D(PetscScalar Fe[], PetscScalar coords[], PetscScalar fx[], PetscScalar fy[], PetscScalar fz[])
{
  PetscInt    ngp;
  PetscScalar gp_xi[GAUSS_POINTS][NSD];
  PetscScalar gp_weight[GAUSS_POINTS];
  PetscInt    p, i;
  PetscScalar Ni_p[NODES_PER_EL];
  PetscScalar GNi_p[NSD][NODES_PER_EL], GNx_p[NSD][NODES_PER_EL];
  PetscScalar J_p, fac;

  /* define quadrature rule */
  ConstructGaussQuadrature3D(&ngp, gp_xi, gp_weight);

  /* evaluate integral */
  for (p = 0; p < ngp; p++) {
    ShapeFunctionQ13D_Evaluate(gp_xi[p], Ni_p);
    ShapeFunctionQ13D_Evaluate_dxi(gp_xi[p], GNi_p);
    ShapeFunctionQ13D_Evaluate_dx(GNi_p, GNx_p, coords, &J_p);
    fac = gp_weight[p] * J_p;

    for (i = 0; i < NODES_PER_EL; i++) {
      Fe[NSD * i] -= fac * Ni_p[i] * fx[p];
      Fe[NSD * i + 1] -= fac * Ni_p[i] * fy[p];
      Fe[NSD * i + 2] -= fac * Ni_p[i] * fz[p];
    }
  }
}

static void FormContinuityRhsQ13D(PetscScalar Fe[], PetscScalar coords[], PetscScalar hc[])
{
  PetscInt    ngp;
  PetscScalar gp_xi[GAUSS_POINTS][NSD];
  PetscScalar gp_weight[GAUSS_POINTS];
  PetscInt    p, i;
  PetscScalar Ni_p[NODES_PER_EL];
  PetscScalar GNi_p[NSD][NODES_PER_EL], GNx_p[NSD][NODES_PER_EL];
  PetscScalar J_p, fac;

  /* define quadrature rule */
  ConstructGaussQuadrature3D(&ngp, gp_xi, gp_weight);

  /* evaluate integral */
  for (p = 0; p < ngp; p++) {
    ShapeFunctionQ13D_Evaluate(gp_xi[p], Ni_p);
    ShapeFunctionQ13D_Evaluate_dxi(gp_xi[p], GNi_p);
    ShapeFunctionQ13D_Evaluate_dx(GNi_p, GNx_p, coords, &J_p);
    fac = gp_weight[p] * J_p;

    for (i = 0; i < NODES_PER_EL; i++) Fe[i] -= fac * Ni_p[i] * hc[p];
  }
}

#define _ZERO_ROWCOL_i(A, i) \
  do { \
    PetscInt    KK; \
    PetscScalar tmp = A[24 * (i) + (i)]; \
    for (KK = 0; KK < 24; KK++) A[24 * (i) + KK] = 0.0; \
    for (KK = 0; KK < 24; KK++) A[24 * KK + (i)] = 0.0; \
    A[24 * (i) + (i)] = tmp; \
  } while (0)

#define _ZERO_ROW_i(A, i) \
  do { \
    PetscInt KK; \
    for (KK = 0; KK < 8; KK++) A[8 * (i) + KK] = 0.0; \
  } while (0)

#define _ZERO_COL_i(A, i) \
  do { \
    PetscInt KK; \
    for (KK = 0; KK < 8; KK++) A[24 * KK + (i)] = 0.0; \
  } while (0)

static PetscErrorCode AssembleA_Stokes(Mat A, DM stokes_da, CellProperties cell_properties)
{
  DM                      cda;
  Vec                     coords;
  DMDACoor3d           ***_coords;
  MatStencil              u_eqn[NODES_PER_EL * U_DOFS];
  MatStencil              p_eqn[NODES_PER_EL * P_DOFS];
  PetscInt                sex, sey, sez, mx, my, mz;
  PetscInt                ei, ej, ek;
  PetscScalar             Ae[NODES_PER_EL * U_DOFS * NODES_PER_EL * U_DOFS];
  PetscScalar             Ge[NODES_PER_EL * U_DOFS * NODES_PER_EL * P_DOFS];
  PetscScalar             De[NODES_PER_EL * P_DOFS * NODES_PER_EL * U_DOFS];
  PetscScalar             Ce[NODES_PER_EL * P_DOFS * NODES_PER_EL * P_DOFS];
  PetscScalar             el_coords[NODES_PER_EL * NSD];
  GaussPointCoefficients *props;
  PetscScalar            *prop_eta;
  PetscInt                n, M, N, P;

  PetscFunctionBeginUser;
  PetscCall(DMDAGetInfo(stokes_da, 0, &M, &N, &P, 0, 0, 0, 0, 0, 0, 0, 0, 0));
  /* setup for coords */
  PetscCall(DMGetCoordinateDM(stokes_da, &cda));
  PetscCall(DMGetCoordinatesLocal(stokes_da, &coords));
  PetscCall(DMDAVecGetArray(cda, coords, &_coords));
  PetscCall(DMDAGetElementsCorners(stokes_da, &sex, &sey, &sez));
  PetscCall(DMDAGetElementsSizes(stokes_da, &mx, &my, &mz));
  for (ek = sez; ek < sez + mz; ek++) {
    for (ej = sey; ej < sey + my; ej++) {
      for (ei = sex; ei < sex + mx; ei++) {
        /* get coords for the element */
        PetscCall(GetElementCoords3D(_coords, ei, ej, ek, el_coords));
        /* get cell properties */
        PetscCall(CellPropertiesGetCell(cell_properties, ei, ej, ek, &props));
        /* get coefficients for the element */
        prop_eta = props->eta;

        /* initialise element stiffness matrix */
        PetscCall(PetscMemzero(Ae, sizeof(Ae)));
        PetscCall(PetscMemzero(Ge, sizeof(Ge)));
        PetscCall(PetscMemzero(De, sizeof(De)));
        PetscCall(PetscMemzero(Ce, sizeof(Ce)));

        /* form element stiffness matrix */
        FormStressOperatorQ13D(Ae, el_coords, prop_eta);
        FormGradientOperatorQ13D(Ge, el_coords);
        /*#if defined(ASSEMBLE_LOWER_TRIANGULAR)*/
        FormDivergenceOperatorQ13D(De, el_coords);
        /*#endif*/
        FormStabilisationOperatorQ13D(Ce, el_coords, prop_eta);

        /* insert element matrix into global matrix */
        PetscCall(DMDAGetElementEqnums3D_up(u_eqn, p_eqn, ei, ej, ek));

        for (n = 0; n < NODES_PER_EL; n++) {
          if ((u_eqn[3 * n].i == 0) || (u_eqn[3 * n].i == M - 1)) {
            _ZERO_ROWCOL_i(Ae, 3 * n);
            _ZERO_ROW_i(Ge, 3 * n);
            _ZERO_COL_i(De, 3 * n);
          }

          if ((u_eqn[3 * n + 1].j == 0) || (u_eqn[3 * n + 1].j == N - 1)) {
            _ZERO_ROWCOL_i(Ae, 3 * n + 1);
            _ZERO_ROW_i(Ge, 3 * n + 1);
            _ZERO_COL_i(De, 3 * n + 1);
          }

          if ((u_eqn[3 * n + 2].k == 0) || (u_eqn[3 * n + 2].k == P - 1)) {
            _ZERO_ROWCOL_i(Ae, 3 * n + 2);
            _ZERO_ROW_i(Ge, 3 * n + 2);
            _ZERO_COL_i(De, 3 * n + 2);
          }
        }
        PetscCall(MatSetValuesStencil(A, NODES_PER_EL * U_DOFS, u_eqn, NODES_PER_EL * U_DOFS, u_eqn, Ae, ADD_VALUES));
        PetscCall(MatSetValuesStencil(A, NODES_PER_EL * U_DOFS, u_eqn, NODES_PER_EL * P_DOFS, p_eqn, Ge, ADD_VALUES));
        PetscCall(MatSetValuesStencil(A, NODES_PER_EL * P_DOFS, p_eqn, NODES_PER_EL * U_DOFS, u_eqn, De, ADD_VALUES));
        PetscCall(MatSetValuesStencil(A, NODES_PER_EL * P_DOFS, p_eqn, NODES_PER_EL * P_DOFS, p_eqn, Ce, ADD_VALUES));
      }
    }
  }
  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));

  PetscCall(DMDAVecRestoreArray(cda, coords, &_coords));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode AssembleA_PCStokes(Mat A, DM stokes_da, CellProperties cell_properties)
{
  DM                      cda;
  Vec                     coords;
  DMDACoor3d           ***_coords;
  MatStencil              u_eqn[NODES_PER_EL * U_DOFS];
  MatStencil              p_eqn[NODES_PER_EL * P_DOFS];
  PetscInt                sex, sey, sez, mx, my, mz;
  PetscInt                ei, ej, ek;
  PetscScalar             Ae[NODES_PER_EL * U_DOFS * NODES_PER_EL * U_DOFS];
  PetscScalar             Ge[NODES_PER_EL * U_DOFS * NODES_PER_EL * P_DOFS];
  PetscScalar             De[NODES_PER_EL * P_DOFS * NODES_PER_EL * U_DOFS];
  PetscScalar             Ce[NODES_PER_EL * P_DOFS * NODES_PER_EL * P_DOFS];
  PetscScalar             el_coords[NODES_PER_EL * NSD];
  GaussPointCoefficients *props;
  PetscScalar            *prop_eta;
  PetscInt                n, M, N, P;

  PetscFunctionBeginUser;
  PetscCall(DMDAGetInfo(stokes_da, 0, &M, &N, &P, 0, 0, 0, 0, 0, 0, 0, 0, 0));
  /* setup for coords */
  PetscCall(DMGetCoordinateDM(stokes_da, &cda));
  PetscCall(DMGetCoordinatesLocal(stokes_da, &coords));
  PetscCall(DMDAVecGetArray(cda, coords, &_coords));

  PetscCall(DMDAGetElementsCorners(stokes_da, &sex, &sey, &sez));
  PetscCall(DMDAGetElementsSizes(stokes_da, &mx, &my, &mz));
  for (ek = sez; ek < sez + mz; ek++) {
    for (ej = sey; ej < sey + my; ej++) {
      for (ei = sex; ei < sex + mx; ei++) {
        /* get coords for the element */
        PetscCall(GetElementCoords3D(_coords, ei, ej, ek, el_coords));
        /* get cell properties */
        PetscCall(CellPropertiesGetCell(cell_properties, ei, ej, ek, &props));
        /* get coefficients for the element */
        prop_eta = props->eta;

        /* initialise element stiffness matrix */
        PetscCall(PetscMemzero(Ae, sizeof(Ae)));
        PetscCall(PetscMemzero(Ge, sizeof(Ge)));
        PetscCall(PetscMemzero(De, sizeof(De)));
        PetscCall(PetscMemzero(Ce, sizeof(Ce)));

        /* form element stiffness matrix */
        FormStressOperatorQ13D(Ae, el_coords, prop_eta);
        FormGradientOperatorQ13D(Ge, el_coords);
        /* FormDivergenceOperatorQ13D(De,el_coords); */
        FormScaledMassMatrixOperatorQ13D(Ce, el_coords, prop_eta);

        /* insert element matrix into global matrix */
        PetscCall(DMDAGetElementEqnums3D_up(u_eqn, p_eqn, ei, ej, ek));

        for (n = 0; n < NODES_PER_EL; n++) {
          if ((u_eqn[3 * n].i == 0) || (u_eqn[3 * n].i == M - 1)) {
            _ZERO_ROWCOL_i(Ae, 3 * n);
            _ZERO_ROW_i(Ge, 3 * n);
            _ZERO_COL_i(De, 3 * n);
          }

          if ((u_eqn[3 * n + 1].j == 0) || (u_eqn[3 * n + 1].j == N - 1)) {
            _ZERO_ROWCOL_i(Ae, 3 * n + 1);
            _ZERO_ROW_i(Ge, 3 * n + 1);
            _ZERO_COL_i(De, 3 * n + 1);
          }

          if ((u_eqn[3 * n + 2].k == 0) || (u_eqn[3 * n + 2].k == P - 1)) {
            _ZERO_ROWCOL_i(Ae, 3 * n + 2);
            _ZERO_ROW_i(Ge, 3 * n + 2);
            _ZERO_COL_i(De, 3 * n + 2);
          }
        }
        PetscCall(MatSetValuesStencil(A, NODES_PER_EL * U_DOFS, u_eqn, NODES_PER_EL * U_DOFS, u_eqn, Ae, ADD_VALUES));
        PetscCall(MatSetValuesStencil(A, NODES_PER_EL * U_DOFS, u_eqn, NODES_PER_EL * P_DOFS, p_eqn, Ge, ADD_VALUES));
        /*PetscCall(MatSetValuesStencil(A,NODES_PER_EL*P_DOFS,p_eqn,NODES_PER_EL*U_DOFS,u_eqn,De,ADD_VALUES));*/
        PetscCall(MatSetValuesStencil(A, NODES_PER_EL * P_DOFS, p_eqn, NODES_PER_EL * P_DOFS, p_eqn, Ce, ADD_VALUES));
      }
    }
  }
  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));

  PetscCall(DMDAVecRestoreArray(cda, coords, &_coords));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode AssembleF_Stokes(Vec F, DM stokes_da, CellProperties cell_properties)
{
  DM                      cda;
  Vec                     coords;
  DMDACoor3d           ***_coords;
  MatStencil              u_eqn[NODES_PER_EL * U_DOFS];
  MatStencil              p_eqn[NODES_PER_EL * P_DOFS];
  PetscInt                sex, sey, sez, mx, my, mz;
  PetscInt                ei, ej, ek;
  PetscScalar             Fe[NODES_PER_EL * U_DOFS];
  PetscScalar             He[NODES_PER_EL * P_DOFS];
  PetscScalar             el_coords[NODES_PER_EL * NSD];
  GaussPointCoefficients *props;
  PetscScalar            *prop_fx, *prop_fy, *prop_fz, *prop_hc;
  Vec                     local_F;
  StokesDOF            ***ff;
  PetscInt                n, M, N, P;

  PetscFunctionBeginUser;
  PetscCall(DMDAGetInfo(stokes_da, 0, &M, &N, &P, 0, 0, 0, 0, 0, 0, 0, 0, 0));
  /* setup for coords */
  PetscCall(DMGetCoordinateDM(stokes_da, &cda));
  PetscCall(DMGetCoordinatesLocal(stokes_da, &coords));
  PetscCall(DMDAVecGetArray(cda, coords, &_coords));

  /* get access to the vector */
  PetscCall(DMGetLocalVector(stokes_da, &local_F));
  PetscCall(VecZeroEntries(local_F));
  PetscCall(DMDAVecGetArray(stokes_da, local_F, &ff));
  PetscCall(DMDAGetElementsCorners(stokes_da, &sex, &sey, &sez));
  PetscCall(DMDAGetElementsSizes(stokes_da, &mx, &my, &mz));
  for (ek = sez; ek < sez + mz; ek++) {
    for (ej = sey; ej < sey + my; ej++) {
      for (ei = sex; ei < sex + mx; ei++) {
        /* get coords for the element */
        PetscCall(GetElementCoords3D(_coords, ei, ej, ek, el_coords));
        /* get cell properties */
        PetscCall(CellPropertiesGetCell(cell_properties, ei, ej, ek, &props));
        /* get coefficients for the element */
        prop_fx = props->fx;
        prop_fy = props->fy;
        prop_fz = props->fz;
        prop_hc = props->hc;

        /* initialise element stiffness matrix */
        PetscCall(PetscMemzero(Fe, sizeof(Fe)));
        PetscCall(PetscMemzero(He, sizeof(He)));

        /* form element stiffness matrix */
        FormMomentumRhsQ13D(Fe, el_coords, prop_fx, prop_fy, prop_fz);
        FormContinuityRhsQ13D(He, el_coords, prop_hc);

        /* insert element matrix into global matrix */
        PetscCall(DMDAGetElementEqnums3D_up(u_eqn, p_eqn, ei, ej, ek));

        for (n = 0; n < NODES_PER_EL; n++) {
          if ((u_eqn[3 * n].i == 0) || (u_eqn[3 * n].i == M - 1)) Fe[3 * n] = 0.0;

          if ((u_eqn[3 * n + 1].j == 0) || (u_eqn[3 * n + 1].j == N - 1)) Fe[3 * n + 1] = 0.0;

          if ((u_eqn[3 * n + 2].k == 0) || (u_eqn[3 * n + 2].k == P - 1)) Fe[3 * n + 2] = 0.0;
        }

        PetscCall(DMDASetValuesLocalStencil3D_ADD_VALUES(ff, u_eqn, p_eqn, Fe, He));
      }
    }
  }
  PetscCall(DMDAVecRestoreArray(stokes_da, local_F, &ff));
  PetscCall(DMLocalToGlobalBegin(stokes_da, local_F, ADD_VALUES, F));
  PetscCall(DMLocalToGlobalEnd(stokes_da, local_F, ADD_VALUES, F));
  PetscCall(DMRestoreLocalVector(stokes_da, &local_F));

  PetscCall(DMDAVecRestoreArray(cda, coords, &_coords));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static void evaluate_MS_FrankKamentski_constants(PetscReal *theta, PetscReal *MX, PetscReal *MY, PetscReal *MZ)
{
  *theta = 0.0;
  *MX    = 2.0 * PETSC_PI;
  *MY    = 2.0 * PETSC_PI;
  *MZ    = 2.0 * PETSC_PI;
}
static void evaluate_MS_FrankKamentski(PetscReal pos[], PetscReal v[], PetscReal *p, PetscReal *eta, PetscReal Fm[], PetscReal *Fc)
{
  PetscReal x, y, z;
  PetscReal theta, MX, MY, MZ;

  evaluate_MS_FrankKamentski_constants(&theta, &MX, &MY, &MZ);
  x = pos[0];
  y = pos[1];
  z = pos[2];
  if (v) {
    /*
     v[0] = PetscPowRealInt(z,3)*PetscExpReal(y)*PetscSinReal(PETSC_PI*x);
     v[1] = z*cos(2.0*PETSC_PI*x)*PetscExpReal(-y)*PetscSinReal(PETSC_PI*y);
     v[2] = -(PetscPowRealInt(x,3) + PetscPowRealInt(y,3))*PetscSinReal(2.0*PETSC_PI*z);
     */
    /*
     v[0] = PetscSinReal(PETSC_PI*x);
     v[1] = PetscSinReal(PETSC_PI*y);
     v[2] = PetscSinReal(PETSC_PI*z);
     */
    v[0] = PetscPowRealInt(z, 3) * PetscExpReal(y) * PetscSinReal(PETSC_PI * x);
    v[1] = z * PetscCosReal(2.0 * PETSC_PI * x) * PetscExpReal(-y) * PetscSinReal(PETSC_PI * y);
    v[2] = PetscPowRealInt(z, 2) * (PetscCosReal(2.0 * PETSC_PI * x) * PetscExpReal(-y) * PetscSinReal(PETSC_PI * y) / 2 - PETSC_PI * PetscCosReal(PETSC_PI * y) * PetscCosReal(2.0 * PETSC_PI * x) * PetscExpReal(-y) / 2) - PETSC_PI * PetscPowRealInt(z, 4) * PetscCosReal(PETSC_PI * x) * PetscExpReal(y) / 4;
  }
  if (p) *p = PetscPowRealInt(x, 2) + PetscPowRealInt(y, 2) + PetscPowRealInt(z, 2);
  if (eta) {
    /* eta = PetscExpReal(-theta*(1.0 - y - 0.1*PetscCosReal(MX*x)*PetscCosReal(MZ*z)*PetscSinReal(MY*y)));*/
    *eta = 1.0;
  }
  if (Fm) {
    /*
     Fm[0] = -2*x - 2.0*PetscPowRealInt(PETSC_PI,2)*PetscPowRealInt(z,3)*PetscExpReal(y)*PetscExpReal(-theta*(1 - y - 0.1*PetscCosReal(MX*x)*PetscCosReal(MZ*z)*PetscSinReal(MY*y)))*PetscSinReal(PETSC_PI*x) - 0.2*MZ*theta*(-1.5*PetscPowRealInt(x,2)*PetscSinReal(2.0*PETSC_PI*z) + 1.5*PetscPowRealInt(z,2)*PetscExpReal(y)*PetscSinReal(PETSC_PI*x))*PetscCosReal(MX*x)*PetscExpReal(-theta*(1 - y - 0.1*PetscCosReal(MX*x)*PetscCosReal(MZ*z)*PetscSinReal(MY*y)))*PetscSinReal(MY*y)*PetscSinReal(MZ*z) - 0.2*PETSC_PI*MX*theta*PetscPowRealInt(z,3)*PetscCosReal(PETSC_PI*x)*PetscCosReal(MZ*z)*PetscExpReal(y)*PetscExpReal(-theta*(1 - y - 0.1*PetscCosReal(MX*x)*PetscCosReal(MZ*z)*PetscSinReal(MY*y)))*PetscSinReal(MX*x)*PetscSinReal(MY*y) + 2.0*(3.0*z*PetscExpReal(y)*PetscSinReal(PETSC_PI*x) - 3.0*PETSC_PI*PetscPowRealInt(x,2)*PetscCosReal(2.0*PETSC_PI*z))*PetscExpReal(-theta*(1 - y - 0.1*PetscCosReal(MX*x)*PetscCosReal(MZ*z)*PetscSinReal(MY*y))) + 2.0*(0.5*PetscPowRealInt(z,3)*PetscExpReal(y)*PetscSinReal(PETSC_PI*x) + PETSC_PI*z*PetscExpReal(-y)*PetscSinReal(PETSC_PI*y)*PetscSinReal(2.0*PETSC_PI*x) - 1.0*z*PetscPowRealInt(PETSC_PI,2)*PetscCosReal(PETSC_PI*y)*PetscExpReal(-y)*PetscSinReal(2.0*PETSC_PI*x))*PetscExpReal(-theta*(1 - y - 0.1*PetscCosReal(MX*x)*PetscCosReal(MZ*z)*PetscSinReal(MY*y))) + 2.0*theta*(1 + 0.1*MY*PetscCosReal(MX*x)*PetscCosReal(MY*y)*PetscCosReal(MZ*z))*(0.5*PetscPowRealInt(z,3)*PetscExpReal(y)*PetscSinReal(PETSC_PI*x) - 1.0*PETSC_PI*z*PetscExpReal(-y)*PetscSinReal(PETSC_PI*y)*PetscSinReal(2.0*PETSC_PI*x))*PetscExpReal(-theta*(1 - y - 0.1*PetscCosReal(MX*x)*PetscCosReal(MZ*z)*PetscSinReal(MY*y))) ;
     Fm[1] = -2*y - 0.2*MX*theta*(0.5*PetscPowRealInt(z,3)*PetscExpReal(y)*PetscSinReal(PETSC_PI*x) - 1.0*PETSC_PI*z*PetscExpReal(-y)*PetscSinReal(PETSC_PI*y)*PetscSinReal(2.0*PETSC_PI*x))*PetscCosReal(MZ*z)*PetscExpReal(-theta*(1 - y - 0.1*PetscCosReal(MX*x)*PetscCosReal(MZ*z)*PetscSinReal(MY*y)))*PetscSinReal(MX*x)*PetscSinReal(MY*y) - 0.2*MZ*theta*(-1.5*PetscPowRealInt(y,2)*PetscSinReal(2.0*PETSC_PI*z) + 0.5*PetscCosReal(2.0*PETSC_PI*x)*PetscExpReal(-y)*PetscSinReal(PETSC_PI*y))*PetscCosReal(MX*x)*PetscExpReal(-theta*(1 - y - 0.1*PetscCosReal(MX*x)*PetscCosReal(MZ*z)*PetscSinReal(MY*y)))*PetscSinReal(MY*y)*PetscSinReal(MZ*z) + 2.0*(-2.0*z*PetscPowRealInt(PETSC_PI,2)*PetscCosReal(2.0*PETSC_PI*x)*PetscExpReal(-y)*PetscSinReal(PETSC_PI*y) + 0.5*PETSC_PI*PetscPowRealInt(z,3)*PetscCosReal(PETSC_PI*x)*PetscExpReal(y))*PetscExpReal(-theta*(1 - y - 0.1*PetscCosReal(MX*x)*PetscCosReal(MZ*z)*PetscSinReal(MY*y))) + 2.0*(z*PetscCosReal(2.0*PETSC_PI*x)*PetscExpReal(-y)*PetscSinReal(PETSC_PI*y) - z*PetscPowRealInt(PETSC_PI,2)*PetscCosReal(2.0*PETSC_PI*x)*PetscExpReal(-y)*PetscSinReal(PETSC_PI*y) - 2*PETSC_PI*z*PetscCosReal(PETSC_PI*y)*PetscCosReal(2.0*PETSC_PI*x)*PetscExpReal(-y))*PetscExpReal(-theta*(1 - y - 0.1*PetscCosReal(MX*x)*PetscCosReal(MZ*z)*PetscSinReal(MY*y))) + 2.0*theta*(1 + 0.1*MY*PetscCosReal(MX*x)*PetscCosReal(MY*y)*PetscCosReal(MZ*z))*(-z*PetscCosReal(2.0*PETSC_PI*x)*PetscExpReal(-y)*PetscSinReal(PETSC_PI*y) + PETSC_PI*z*PetscCosReal(PETSC_PI*y)*PetscCosReal(2.0*PETSC_PI*x)*PetscExpReal(-y))*PetscExpReal(-theta*(1 - y - 0.1*PetscCosReal(MX*x)*PetscCosReal(MZ*z)*PetscSinReal(MY*y))) - 6.0*PETSC_PI*PetscPowRealInt(y,2)*PetscCosReal(2.0*PETSC_PI*z)*PetscExpReal(-theta*(1 - y - 0.1*PetscCosReal(MX*x)*PetscCosReal(MZ*z)*PetscSinReal(MY*y)));
     Fm[2] = -2*z + 8.0*PetscPowRealInt(PETSC_PI,2)*(PetscPowRealInt(x,3) + PetscPowRealInt(y,3))*PetscExpReal(-theta*(1 - y - 0.1*PetscCosReal(MX*x)*PetscCosReal(MZ*z)*PetscSinReal(MY*y)))*PetscSinReal(2.0*PETSC_PI*z) - 0.2*MX*theta*(-1.5*PetscPowRealInt(x,2)*PetscSinReal(2.0*PETSC_PI*z) + 1.5*PetscPowRealInt(z,2)*PetscExpReal(y)*PetscSinReal(PETSC_PI*x))*PetscCosReal(MZ*z)*PetscExpReal(-theta*(1 - y - 0.1*PetscCosReal(MX*x)*PetscCosReal(MZ*z)*PetscSinReal(MY*y)))*PetscSinReal(MX*x)*PetscSinReal(MY*y) + 0.4*PETSC_PI*MZ*theta*(PetscPowRealInt(x,3) + PetscPowRealInt(y,3))*PetscCosReal(MX*x)*PetscCosReal(2.0*PETSC_PI*z)*PetscExpReal(-theta*(1 - y - 0.1*PetscCosReal(MX*x)*PetscCosReal(MZ*z)*PetscSinReal(MY*y)))*PetscSinReal(MY*y)*PetscSinReal(MZ*z) + 2.0*(-3.0*x*PetscSinReal(2.0*PETSC_PI*z) + 1.5*PETSC_PI*PetscPowRealInt(z,2)*PetscCosReal(PETSC_PI*x)*PetscExpReal(y))*PetscExpReal(-theta*(1 - y - 0.1*PetscCosReal(MX*x)*PetscCosReal(MZ*z)*PetscSinReal(MY*y))) + 2.0*(-3.0*y*PetscSinReal(2.0*PETSC_PI*z) - 0.5*PetscCosReal(2.0*PETSC_PI*x)*PetscExpReal(-y)*PetscSinReal(PETSC_PI*y) + 0.5*PETSC_PI*PetscCosReal(PETSC_PI*y)*PetscCosReal(2.0*PETSC_PI*x)*PetscExpReal(-y))*PetscExpReal(-theta*(1 - y - 0.1*PetscCosReal(MX*x)*PetscCosReal(MZ*z)*PetscSinReal(MY*y))) + 2.0*theta*(1 + 0.1*MY*PetscCosReal(MX*x)*PetscCosReal(MY*y)*PetscCosReal(MZ*z))*(-1.5*PetscPowRealInt(y,2)*PetscSinReal(2.0*PETSC_PI*z) + 0.5*PetscCosReal(2.0*PETSC_PI*x)*PetscExpReal(-y)*PetscSinReal(PETSC_PI*y))*PetscExpReal(-theta*(1 - y - 0.1*PetscCosReal(MX*x)*PetscCosReal(MZ*z)*PetscSinReal(MY*y)))  ;
     */
    /*
     Fm[0]=-2*x - 2.0*PetscPowRealInt(PETSC_PI,2)*PetscSinReal(PETSC_PI*x);
     Fm[1]=-2*y - 2.0*PetscPowRealInt(PETSC_PI,2)*PetscSinReal(PETSC_PI*y);
     Fm[2]=-2*z - 2.0*PetscPowRealInt(PETSC_PI,2)*PetscSinReal(PETSC_PI*z);
     */
    /*
     Fm[0] = -2*x + PetscPowRealInt(z,3)*PetscExpReal(y)*PetscSinReal(PETSC_PI*x) + 6.0*z*PetscExpReal(y)*PetscSinReal(PETSC_PI*x) - 6.0*PETSC_PI*PetscPowRealInt(x,2)*PetscCosReal(2.0*PETSC_PI*z) - 2.0*PetscPowRealInt(PETSC_PI,2)*PetscPowRealInt(z,3)*PetscExpReal(y)*PetscSinReal(PETSC_PI*x) + 2.0*PETSC_PI*z*PetscExpReal(-y)*PetscSinReal(PETSC_PI*y)*PetscSinReal(2.0*PETSC_PI*x) - 2.0*z*PetscPowRealInt(PETSC_PI,2)*PetscCosReal(PETSC_PI*y)*PetscExpReal(-y)*PetscSinReal(2.0*PETSC_PI*x) ;
     Fm[1] = -2*y - 6.0*PETSC_PI*PetscPowRealInt(y,2)*PetscCosReal(2.0*PETSC_PI*z) + 2.0*z*PetscCosReal(2.0*PETSC_PI*x)*PetscExpReal(-y)*PetscSinReal(PETSC_PI*y) - 6.0*z*PetscPowRealInt(PETSC_PI,2)*PetscCosReal(2.0*PETSC_PI*x)*PetscExpReal(-y)*PetscSinReal(PETSC_PI*y) + PETSC_PI*PetscPowRealInt(z,3)*PetscCosReal(PETSC_PI*x)*PetscExpReal(y) - 4.0*PETSC_PI*z*PetscCosReal(PETSC_PI*y)*PetscCosReal(2.0*PETSC_PI*x)*PetscExpReal(-y);
     Fm[2] = -2*z - 6.0*x*PetscSinReal(2.0*PETSC_PI*z) - 6.0*y*PetscSinReal(2.0*PETSC_PI*z) - PetscCosReal(2.0*PETSC_PI*x)*PetscExpReal(-y)*PetscSinReal(PETSC_PI*y) + 8.0*PetscPowRealInt(PETSC_PI,2)*(PetscPowRealInt(x,3) + PetscPowRealInt(y,3))*PetscSinReal(2.0*PETSC_PI*z) + 3.0*PETSC_PI*PetscPowRealInt(z,2)*PetscCosReal(PETSC_PI*x)*PetscExpReal(y) + PETSC_PI*PetscCosReal(PETSC_PI*y)*PetscCosReal(2.0*PETSC_PI*x)*PetscExpReal(-y) ;
     */
    Fm[0] = -2 * x + 2 * z * (PetscPowRealInt(PETSC_PI, 2) * PetscCosReal(PETSC_PI * y) * PetscExpReal(-y) * PetscSinReal(2.0 * PETSC_PI * x) - 1.0 * PETSC_PI * PetscExpReal(-y) * PetscSinReal(PETSC_PI * y) * PetscSinReal(2.0 * PETSC_PI * x)) + PetscPowRealInt(z, 3) * PetscExpReal(y) * PetscSinReal(PETSC_PI * x) + 6.0 * z * PetscExpReal(y) * PetscSinReal(PETSC_PI * x) - 1.0 * PetscPowRealInt(PETSC_PI, 2) * PetscPowRealInt(z, 3) * PetscExpReal(y) * PetscSinReal(PETSC_PI * x) + 2.0 * PETSC_PI * z * PetscExpReal(-y) * PetscSinReal(PETSC_PI * y) * PetscSinReal(2.0 * PETSC_PI * x) - 2.0 * z * PetscPowRealInt(PETSC_PI, 2) * PetscCosReal(PETSC_PI * y) * PetscExpReal(-y) * PetscSinReal(2.0 * PETSC_PI * x);
    Fm[1] = -2 * y + 2 * z * (-PetscCosReal(2.0 * PETSC_PI * x) * PetscExpReal(-y) * PetscSinReal(PETSC_PI * y) / 2 + PetscPowRealInt(PETSC_PI, 2) * PetscCosReal(2.0 * PETSC_PI * x) * PetscExpReal(-y) * PetscSinReal(PETSC_PI * y) / 2 + PETSC_PI * PetscCosReal(PETSC_PI * y) * PetscCosReal(2.0 * PETSC_PI * x) * PetscExpReal(-y)) + 2.0 * z * PetscCosReal(2.0 * PETSC_PI * x) * PetscExpReal(-y) * PetscSinReal(PETSC_PI * y) - 6.0 * z * PetscPowRealInt(PETSC_PI, 2) * PetscCosReal(2.0 * PETSC_PI * x) * PetscExpReal(-y) * PetscSinReal(PETSC_PI * y) - 4.0 * PETSC_PI * z * PetscCosReal(PETSC_PI * y) * PetscCosReal(2.0 * PETSC_PI * x) * PetscExpReal(-y);
    Fm[2] = -2 * z + PetscPowRealInt(z, 2) * (-2.0 * PetscPowRealInt(PETSC_PI, 2) * PetscCosReal(2.0 * PETSC_PI * x) * PetscExpReal(-y) * PetscSinReal(PETSC_PI * y) + 2.0 * PetscPowRealInt(PETSC_PI, 3) * PetscCosReal(PETSC_PI * y) * PetscCosReal(2.0 * PETSC_PI * x) * PetscExpReal(-y)) + PetscPowRealInt(z, 2) * (PetscCosReal(2.0 * PETSC_PI * x) * PetscExpReal(-y) * PetscSinReal(PETSC_PI * y) / 2 - 3 * PetscPowRealInt(PETSC_PI, 2) * PetscCosReal(2.0 * PETSC_PI * x) * PetscExpReal(-y) * PetscSinReal(PETSC_PI * y) / 2 + PetscPowRealInt(PETSC_PI, 3) * PetscCosReal(PETSC_PI * y) * PetscCosReal(2.0 * PETSC_PI * x) * PetscExpReal(-y) / 2 - 3 * PETSC_PI * PetscCosReal(PETSC_PI * y) * PetscCosReal(2.0 * PETSC_PI * x) * PetscExpReal(-y) / 2) + 1.0 * PetscCosReal(2.0 * PETSC_PI * x) * PetscExpReal(-y) * PetscSinReal(PETSC_PI * y) + 0.25 * PetscPowRealInt(PETSC_PI, 3) * PetscPowRealInt(z, 4) * PetscCosReal(PETSC_PI * x) * PetscExpReal(y) - 0.25 * PETSC_PI * PetscPowRealInt(z, 4) * PetscCosReal(PETSC_PI * x) * PetscExpReal(y) - 3.0 * PETSC_PI * PetscPowRealInt(z, 2) * PetscCosReal(PETSC_PI * x) * PetscExpReal(y) - 1.0 * PETSC_PI * PetscCosReal(PETSC_PI * y) * PetscCosReal(2.0 * PETSC_PI * x) * PetscExpReal(-y);
  }
  if (Fc) {
    /* Fc = -2.0*PETSC_PI*(PetscPowRealInt(x,3) + PetscPowRealInt(y,3))*PetscCosReal(2.0*PETSC_PI*z) - z*PetscCosReal(2.0*PETSC_PI*x)*PetscExpReal(-y)*PetscSinReal(PETSC_PI*y) + PETSC_PI*PetscPowRealInt(z,3)*PetscCosReal(PETSC_PI*x)*PetscExpReal(y) + PETSC_PI*z*PetscCosReal(PETSC_PI*y)*PetscCosReal(2.0*PETSC_PI*x)*PetscExpReal(-y) ;*/
    /* Fc = PETSC_PI*PetscCosReal(PETSC_PI*x) + PETSC_PI*PetscCosReal(PETSC_PI*y) + PETSC_PI*PetscCosReal(PETSC_PI*z);*/
    /* Fc = -2.0*PETSC_PI*(PetscPowRealInt(x,3) + PetscPowRealInt(y,3))*PetscCosReal(2.0*PETSC_PI*z) - z*PetscCosReal(2.0*PETSC_PI*x)*PetscExpReal(-y)*PetscSinReal(PETSC_PI*y) + PETSC_PI*PetscPowRealInt(z,3)*PetscCosReal(PETSC_PI*x)*PetscExpReal(y) + PETSC_PI*z*PetscCosReal(PETSC_PI*y)*PetscCosReal(2.0*PETSC_PI*x)*PetscExpReal(-y);*/
    *Fc = 0.0;
  }
}

static PetscErrorCode DMDACreateManufacturedSolution(PetscInt mx, PetscInt my, PetscInt mz, DM *_da, Vec *_X)
{
  DM            da, cda;
  Vec           X;
  StokesDOF  ***_stokes;
  Vec           coords;
  DMDACoor3d ***_coords;
  PetscInt      si, sj, sk, ei, ej, ek, i, j, k;

  PetscFunctionBeginUser;
  PetscCall(DMDACreate3d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DMDA_STENCIL_BOX, mx + 1, my + 1, mz + 1, PETSC_DECIDE, PETSC_DECIDE, PETSC_DECIDE, 4, 1, NULL, NULL, NULL, &da));
  PetscCall(DMSetFromOptions(da));
  PetscCall(DMSetUp(da));
  PetscCall(DMDASetFieldName(da, 0, "anlytic_Vx"));
  PetscCall(DMDASetFieldName(da, 1, "anlytic_Vy"));
  PetscCall(DMDASetFieldName(da, 2, "anlytic_Vz"));
  PetscCall(DMDASetFieldName(da, 3, "analytic_P"));

  PetscCall(DMDASetUniformCoordinates(da, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0));

  PetscCall(DMGetCoordinatesLocal(da, &coords));
  PetscCall(DMGetCoordinateDM(da, &cda));
  PetscCall(DMDAVecGetArray(cda, coords, &_coords));

  PetscCall(DMCreateGlobalVector(da, &X));
  PetscCall(DMDAVecGetArray(da, X, &_stokes));

  PetscCall(DMDAGetCorners(da, &si, &sj, &sk, &ei, &ej, &ek));
  for (k = sk; k < sk + ek; k++) {
    for (j = sj; j < sj + ej; j++) {
      for (i = si; i < si + ei; i++) {
        PetscReal pos[NSD], pressure, vel[NSD];

        pos[0] = PetscRealPart(_coords[k][j][i].x);
        pos[1] = PetscRealPart(_coords[k][j][i].y);
        pos[2] = PetscRealPart(_coords[k][j][i].z);

        evaluate_MS_FrankKamentski(pos, vel, &pressure, NULL, NULL, NULL);

        _stokes[k][j][i].u_dof = vel[0];
        _stokes[k][j][i].v_dof = vel[1];
        _stokes[k][j][i].w_dof = vel[2];
        _stokes[k][j][i].p_dof = pressure;
      }
    }
  }
  PetscCall(DMDAVecRestoreArray(da, X, &_stokes));
  PetscCall(DMDAVecRestoreArray(cda, coords, &_coords));

  *_da = da;
  *_X  = X;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMDAIntegrateErrors3D(DM stokes_da, Vec X, Vec X_analytic)
{
  DM            cda;
  Vec           coords, X_analytic_local, X_local;
  DMDACoor3d ***_coords;
  PetscInt      sex, sey, sez, mx, my, mz;
  PetscInt      ei, ej, ek;
  PetscScalar   el_coords[NODES_PER_EL * NSD];
  StokesDOF  ***stokes_analytic, ***stokes;
  StokesDOF     stokes_analytic_e[NODES_PER_EL], stokes_e[NODES_PER_EL];

  PetscScalar GNi_p[NSD][NODES_PER_EL], GNx_p[NSD][NODES_PER_EL];
  PetscScalar Ni_p[NODES_PER_EL];
  PetscInt    ngp;
  PetscScalar gp_xi[GAUSS_POINTS][NSD];
  PetscScalar gp_weight[GAUSS_POINTS];
  PetscInt    p, i;
  PetscScalar J_p, fac;
  PetscScalar h, p_e_L2, u_e_L2, u_e_H1, p_L2, u_L2, u_H1, tp_L2, tu_L2, tu_H1;
  PetscScalar tint_p_ms, tint_p, int_p_ms, int_p;
  PetscInt    M;
  PetscReal   xymin[NSD], xymax[NSD];

  PetscFunctionBeginUser;
  /* define quadrature rule */
  ConstructGaussQuadrature3D(&ngp, gp_xi, gp_weight);

  /* setup for coords */
  PetscCall(DMGetCoordinateDM(stokes_da, &cda));
  PetscCall(DMGetCoordinatesLocal(stokes_da, &coords));
  PetscCall(DMDAVecGetArray(cda, coords, &_coords));

  /* setup for analytic */
  PetscCall(DMCreateLocalVector(stokes_da, &X_analytic_local));
  PetscCall(DMGlobalToLocalBegin(stokes_da, X_analytic, INSERT_VALUES, X_analytic_local));
  PetscCall(DMGlobalToLocalEnd(stokes_da, X_analytic, INSERT_VALUES, X_analytic_local));
  PetscCall(DMDAVecGetArray(stokes_da, X_analytic_local, &stokes_analytic));

  /* setup for solution */
  PetscCall(DMCreateLocalVector(stokes_da, &X_local));
  PetscCall(DMGlobalToLocalBegin(stokes_da, X, INSERT_VALUES, X_local));
  PetscCall(DMGlobalToLocalEnd(stokes_da, X, INSERT_VALUES, X_local));
  PetscCall(DMDAVecGetArray(stokes_da, X_local, &stokes));

  PetscCall(DMDAGetInfo(stokes_da, 0, &M, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0));
  PetscCall(DMGetBoundingBox(stokes_da, xymin, xymax));

  h = (xymax[0] - xymin[0]) / ((PetscReal)(M - 1));

  tp_L2 = tu_L2 = tu_H1 = 0.0;
  tint_p_ms = tint_p = 0.0;

  PetscCall(DMDAGetElementsCorners(stokes_da, &sex, &sey, &sez));
  PetscCall(DMDAGetElementsSizes(stokes_da, &mx, &my, &mz));
  for (ek = sez; ek < sez + mz; ek++) {
    for (ej = sey; ej < sey + my; ej++) {
      for (ei = sex; ei < sex + mx; ei++) {
        /* get coords for the element */
        PetscCall(GetElementCoords3D(_coords, ei, ej, ek, el_coords));
        PetscCall(StokesDAGetNodalFields3D(stokes, ei, ej, ek, stokes_e));
        PetscCall(StokesDAGetNodalFields3D(stokes_analytic, ei, ej, ek, stokes_analytic_e));

        /* evaluate integral */
        for (p = 0; p < ngp; p++) {
          ShapeFunctionQ13D_Evaluate(gp_xi[p], Ni_p);
          ShapeFunctionQ13D_Evaluate_dxi(gp_xi[p], GNi_p);
          ShapeFunctionQ13D_Evaluate_dx(GNi_p, GNx_p, el_coords, &J_p);
          fac = gp_weight[p] * J_p;

          for (i = 0; i < NODES_PER_EL; i++) {
            tint_p_ms = tint_p_ms + fac * Ni_p[i] * stokes_analytic_e[i].p_dof;
            tint_p    = tint_p + fac * Ni_p[i] * stokes_e[i].p_dof;
          }
        }
      }
    }
  }
  PetscCallMPI(MPIU_Allreduce(&tint_p_ms, &int_p_ms, 1, MPIU_SCALAR, MPIU_SUM, PETSC_COMM_WORLD));
  PetscCallMPI(MPIU_Allreduce(&tint_p, &int_p, 1, MPIU_SCALAR, MPIU_SUM, PETSC_COMM_WORLD));

  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\\int P dv %1.4e (h)  %1.4e (ms)\n", (double)PetscRealPart(int_p), (double)PetscRealPart(int_p_ms)));

  /* remove mine and add manufacture one */
  PetscCall(DMDAVecRestoreArray(stokes_da, X_analytic_local, &stokes_analytic));
  PetscCall(DMDAVecRestoreArray(stokes_da, X_local, &stokes));

  {
    PetscInt     k, L, dof;
    PetscScalar *fields;
    PetscCall(DMDAGetInfo(stokes_da, 0, 0, 0, 0, 0, 0, 0, &dof, 0, 0, 0, 0, 0));

    PetscCall(VecGetLocalSize(X_local, &L));
    PetscCall(VecGetArray(X_local, &fields));
    for (k = 0; k < L / dof; k++) fields[dof * k + 3] = fields[dof * k + 3] - int_p + int_p_ms;
    PetscCall(VecRestoreArray(X_local, &fields));

    PetscCall(VecGetLocalSize(X, &L));
    PetscCall(VecGetArray(X, &fields));
    for (k = 0; k < L / dof; k++) fields[dof * k + 3] = fields[dof * k + 3] - int_p + int_p_ms;
    PetscCall(VecRestoreArray(X, &fields));
  }

  PetscCall(DMDAVecGetArray(stokes_da, X_local, &stokes));
  PetscCall(DMDAVecGetArray(stokes_da, X_analytic_local, &stokes_analytic));

  for (ek = sez; ek < sez + mz; ek++) {
    for (ej = sey; ej < sey + my; ej++) {
      for (ei = sex; ei < sex + mx; ei++) {
        /* get coords for the element */
        PetscCall(GetElementCoords3D(_coords, ei, ej, ek, el_coords));
        PetscCall(StokesDAGetNodalFields3D(stokes, ei, ej, ek, stokes_e));
        PetscCall(StokesDAGetNodalFields3D(stokes_analytic, ei, ej, ek, stokes_analytic_e));

        /* evaluate integral */
        p_e_L2 = 0.0;
        u_e_L2 = 0.0;
        u_e_H1 = 0.0;
        for (p = 0; p < ngp; p++) {
          ShapeFunctionQ13D_Evaluate(gp_xi[p], Ni_p);
          ShapeFunctionQ13D_Evaluate_dxi(gp_xi[p], GNi_p);
          ShapeFunctionQ13D_Evaluate_dx(GNi_p, GNx_p, el_coords, &J_p);
          fac = gp_weight[p] * J_p;

          for (i = 0; i < NODES_PER_EL; i++) {
            PetscScalar u_error, v_error, w_error;

            p_e_L2 = p_e_L2 + fac * Ni_p[i] * (stokes_e[i].p_dof - stokes_analytic_e[i].p_dof) * (stokes_e[i].p_dof - stokes_analytic_e[i].p_dof);

            u_error = stokes_e[i].u_dof - stokes_analytic_e[i].u_dof;
            v_error = stokes_e[i].v_dof - stokes_analytic_e[i].v_dof;
            w_error = stokes_e[i].w_dof - stokes_analytic_e[i].w_dof;
            u_e_L2 += fac * Ni_p[i] * (u_error * u_error + v_error * v_error + w_error * w_error);

            u_e_H1 = u_e_H1 + fac * (GNx_p[0][i] * u_error * GNx_p[0][i] * u_error                                                                                                   /* du/dx */
                                     + GNx_p[1][i] * u_error * GNx_p[1][i] * u_error + GNx_p[2][i] * u_error * GNx_p[2][i] * u_error + GNx_p[0][i] * v_error * GNx_p[0][i] * v_error /* dv/dx */
                                     + GNx_p[1][i] * v_error * GNx_p[1][i] * v_error + GNx_p[2][i] * v_error * GNx_p[2][i] * v_error + GNx_p[0][i] * w_error * GNx_p[0][i] * w_error /* dw/dx */
                                     + GNx_p[1][i] * w_error * GNx_p[1][i] * w_error + GNx_p[2][i] * w_error * GNx_p[2][i] * w_error);
          }
        }

        tp_L2 += p_e_L2;
        tu_L2 += u_e_L2;
        tu_H1 += u_e_H1;
      }
    }
  }
  PetscCallMPI(MPIU_Allreduce(&tp_L2, &p_L2, 1, MPIU_SCALAR, MPIU_SUM, PETSC_COMM_WORLD));
  PetscCallMPI(MPIU_Allreduce(&tu_L2, &u_L2, 1, MPIU_SCALAR, MPIU_SUM, PETSC_COMM_WORLD));
  PetscCallMPI(MPIU_Allreduce(&tu_H1, &u_H1, 1, MPIU_SCALAR, MPIU_SUM, PETSC_COMM_WORLD));
  p_L2 = PetscSqrtScalar(p_L2);
  u_L2 = PetscSqrtScalar(u_L2);
  u_H1 = PetscSqrtScalar(u_H1);

  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "%1.4e   %1.4e   %1.4e   %1.4e \n", (double)PetscRealPart(h), (double)PetscRealPart(p_L2), (double)PetscRealPart(u_L2), (double)PetscRealPart(u_H1)));

  PetscCall(DMDAVecRestoreArray(cda, coords, &_coords));

  PetscCall(DMDAVecRestoreArray(stokes_da, X_analytic_local, &stokes_analytic));
  PetscCall(VecDestroy(&X_analytic_local));
  PetscCall(DMDAVecRestoreArray(stokes_da, X_local, &stokes));
  PetscCall(VecDestroy(&X_local));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DAView_3DVTK_StructuredGrid_appended(DM da, Vec FIELD, const char file_prefix[])
{
  char         vtk_filename[PETSC_MAX_PATH_LEN];
  PetscMPIInt  rank;
  MPI_Comm     comm;
  FILE        *vtk_fp     = NULL;
  const char  *byte_order = PetscBinaryBigEndian() ? "BigEndian" : "LittleEndian";
  PetscInt     si, sj, sk, nx, ny, nz, i;
  PetscInt     f, n_fields, N;
  DM           cda;
  Vec          coords;
  Vec          l_FIELD;
  PetscScalar *_L_FIELD;
  PetscInt     memory_offset;
  PetscScalar *buffer;

  PetscFunctionBeginUser;
  /* create file name */
  PetscCall(PetscObjectGetComm((PetscObject)da, &comm));
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  PetscCall(PetscSNPrintf(vtk_filename, sizeof(vtk_filename), "subdomain-%s-p%1.4d.vts", file_prefix, rank));

  /* open file and write header */
  vtk_fp = fopen(vtk_filename, "w");
  PetscCheck(vtk_fp, PETSC_COMM_SELF, PETSC_ERR_SYS, "Cannot open file = %s ", vtk_filename);

  PetscCall(PetscFPrintf(PETSC_COMM_SELF, vtk_fp, "<?xml version=\"1.0\"?>\n"));

  /* coords */
  PetscCall(DMDAGetGhostCorners(da, &si, &sj, &sk, &nx, &ny, &nz));
  N = nx * ny * nz;

  PetscCall(PetscFPrintf(PETSC_COMM_SELF, vtk_fp, "<VTKFile type=\"StructuredGrid\" version=\"0.1\" byte_order=\"%s\">\n", byte_order));
  PetscCall(PetscFPrintf(PETSC_COMM_SELF, vtk_fp, "  <StructuredGrid WholeExtent=\"%" PetscInt_FMT " %" PetscInt_FMT " %" PetscInt_FMT " %" PetscInt_FMT " %" PetscInt_FMT " %" PetscInt_FMT "\">\n", si, si + nx - 1, sj, sj + ny - 1, sk, sk + nz - 1));
  PetscCall(PetscFPrintf(PETSC_COMM_SELF, vtk_fp, "    <Piece Extent=\"%" PetscInt_FMT " %" PetscInt_FMT " %" PetscInt_FMT " %" PetscInt_FMT " %" PetscInt_FMT " %" PetscInt_FMT "\">\n", si, si + nx - 1, sj, sj + ny - 1, sk, sk + nz - 1));

  memory_offset = 0;

  PetscCall(PetscFPrintf(PETSC_COMM_SELF, vtk_fp, "      <CellData></CellData>\n"));

  PetscCall(PetscFPrintf(PETSC_COMM_SELF, vtk_fp, "      <Points>\n"));

  /* copy coordinates */
  PetscCall(DMGetCoordinateDM(da, &cda));
  PetscCall(DMGetCoordinatesLocal(da, &coords));
  PetscCall(PetscFPrintf(PETSC_COMM_SELF, vtk_fp, "        <DataArray type=\"Float64\" NumberOfComponents=\"3\" format=\"appended\" offset=\"%" PetscInt_FMT "\" />\n", memory_offset));
  memory_offset = memory_offset + sizeof(PetscInt) + sizeof(PetscScalar) * N * 3;

  PetscCall(PetscFPrintf(PETSC_COMM_SELF, vtk_fp, "      </Points>\n"));

  PetscCall(PetscFPrintf(PETSC_COMM_SELF, vtk_fp, "      <PointData Scalars=\" "));
  PetscCall(DMDAGetInfo(da, 0, 0, 0, 0, 0, 0, 0, &n_fields, 0, 0, 0, 0, 0));
  for (f = 0; f < n_fields; f++) {
    const char *field_name;
    PetscCall(DMDAGetFieldName(da, f, &field_name));
    PetscCall(PetscFPrintf(PETSC_COMM_SELF, vtk_fp, "%s ", field_name));
  }
  PetscCall(PetscFPrintf(PETSC_COMM_SELF, vtk_fp, "\">\n"));

  for (f = 0; f < n_fields; f++) {
    const char *field_name;

    PetscCall(DMDAGetFieldName(da, f, &field_name));
    PetscCall(PetscFPrintf(PETSC_COMM_SELF, vtk_fp, "        <DataArray type=\"Float64\" Name=\"%s\" format=\"appended\" offset=\"%" PetscInt_FMT "\"/>\n", field_name, memory_offset));
    memory_offset = memory_offset + sizeof(PetscInt) + sizeof(PetscScalar) * N;
  }

  PetscCall(PetscFPrintf(PETSC_COMM_SELF, vtk_fp, "      </PointData>\n"));
  PetscCall(PetscFPrintf(PETSC_COMM_SELF, vtk_fp, "    </Piece>\n"));
  PetscCall(PetscFPrintf(PETSC_COMM_SELF, vtk_fp, "  </StructuredGrid>\n"));

  PetscCall(PetscMalloc1(N, &buffer));
  PetscCall(DMGetLocalVector(da, &l_FIELD));
  PetscCall(DMGlobalToLocalBegin(da, FIELD, INSERT_VALUES, l_FIELD));
  PetscCall(DMGlobalToLocalEnd(da, FIELD, INSERT_VALUES, l_FIELD));
  PetscCall(VecGetArray(l_FIELD, &_L_FIELD));

  PetscCall(PetscFPrintf(PETSC_COMM_SELF, vtk_fp, "  <AppendedData encoding=\"raw\">\n"));
  PetscCall(PetscFPrintf(PETSC_COMM_SELF, vtk_fp, "_"));

  /* write coordinates */
  {
    int          length = (int)(sizeof(PetscScalar) * N * 3);
    PetscScalar *allcoords;

    fwrite(&length, sizeof(int), 1, vtk_fp);
    PetscCall(VecGetArray(coords, &allcoords));
    fwrite(allcoords, sizeof(PetscScalar), 3 * N, vtk_fp);
    PetscCall(VecRestoreArray(coords, &allcoords));
  }
  /* write fields */
  for (f = 0; f < n_fields; f++) {
    int length = (int)(sizeof(PetscScalar) * N);
    fwrite(&length, sizeof(int), 1, vtk_fp);
    /* load */
    for (i = 0; i < N; i++) buffer[i] = _L_FIELD[n_fields * i + f];

    /* write */
    fwrite(buffer, sizeof(PetscScalar), N, vtk_fp);
  }
  PetscCall(PetscFPrintf(PETSC_COMM_SELF, vtk_fp, "\n  </AppendedData>\n"));

  PetscCall(PetscFPrintf(PETSC_COMM_SELF, vtk_fp, "</VTKFile>\n"));

  PetscCall(PetscFree(buffer));
  PetscCall(VecRestoreArray(l_FIELD, &_L_FIELD));
  PetscCall(DMRestoreLocalVector(da, &l_FIELD));

  if (vtk_fp) {
    fclose(vtk_fp);
    vtk_fp = NULL;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DAViewVTK_write_PieceExtend(FILE *vtk_fp, PetscInt indent_level, DM da, const char local_file_prefix[])
{
  PetscMPIInt     size, rank;
  MPI_Comm        comm;
  const PetscInt *lx, *ly, *lz;
  PetscInt        M, N, P, pM, pN, pP, sum, *olx, *oly, *olz;
  PetscInt       *osx, *osy, *osz, *oex, *oey, *oez;
  PetscInt        i, j, k, II, stencil;

  PetscFunctionBeginUser;
  /* create file name */
  PetscCall(PetscObjectGetComm((PetscObject)da, &comm));
  PetscCallMPI(MPI_Comm_size(comm, &size));
  PetscCallMPI(MPI_Comm_rank(comm, &rank));

  PetscCall(DMDAGetInfo(da, 0, &M, &N, &P, &pM, &pN, &pP, 0, &stencil, 0, 0, 0, 0));
  PetscCall(DMDAGetOwnershipRanges(da, &lx, &ly, &lz));

  /* generate start,end list */
  PetscCall(PetscMalloc1(pM + 1, &olx));
  PetscCall(PetscMalloc1(pN + 1, &oly));
  PetscCall(PetscMalloc1(pP + 1, &olz));
  sum = 0;
  for (i = 0; i < pM; i++) {
    olx[i] = sum;
    sum    = sum + lx[i];
  }
  olx[pM] = sum;
  sum     = 0;
  for (i = 0; i < pN; i++) {
    oly[i] = sum;
    sum    = sum + ly[i];
  }
  oly[pN] = sum;
  sum     = 0;
  for (i = 0; i < pP; i++) {
    olz[i] = sum;
    sum    = sum + lz[i];
  }
  olz[pP] = sum;

  PetscCall(PetscMalloc1(pM, &osx));
  PetscCall(PetscMalloc1(pN, &osy));
  PetscCall(PetscMalloc1(pP, &osz));
  PetscCall(PetscMalloc1(pM, &oex));
  PetscCall(PetscMalloc1(pN, &oey));
  PetscCall(PetscMalloc1(pP, &oez));
  for (i = 0; i < pM; i++) {
    osx[i] = olx[i] - stencil;
    oex[i] = olx[i] + lx[i] + stencil;
    if (osx[i] < 0) osx[i] = 0;
    if (oex[i] > M) oex[i] = M;
  }

  for (i = 0; i < pN; i++) {
    osy[i] = oly[i] - stencil;
    oey[i] = oly[i] + ly[i] + stencil;
    if (osy[i] < 0) osy[i] = 0;
    if (oey[i] > M) oey[i] = N;
  }
  for (i = 0; i < pP; i++) {
    osz[i] = olz[i] - stencil;
    oez[i] = olz[i] + lz[i] + stencil;
    if (osz[i] < 0) osz[i] = 0;
    if (oez[i] > P) oez[i] = P;
  }

  for (k = 0; k < pP; k++) {
    for (j = 0; j < pN; j++) {
      for (i = 0; i < pM; i++) {
        char     name[PETSC_MAX_PATH_LEN];
        PetscInt procid = i + j * pM + k * pM * pN; /* convert proc(i,j,k) to pid */
        PetscCall(PetscSNPrintf(name, sizeof(name), "subdomain-%s-p%1.4" PetscInt_FMT ".vts", local_file_prefix, procid));
        for (II = 0; II < indent_level; II++) PetscCall(PetscFPrintf(PETSC_COMM_SELF, vtk_fp, "  "));

        PetscCall(PetscFPrintf(PETSC_COMM_SELF, vtk_fp, "<Piece Extent=\"%" PetscInt_FMT " %" PetscInt_FMT " %" PetscInt_FMT " %" PetscInt_FMT " %" PetscInt_FMT " %" PetscInt_FMT "\"      Source=\"%s\"/>\n", osx[i], oex[i] - 1, osy[j], oey[j] - 1, osz[k], oez[k] - 1, name));
      }
    }
  }
  PetscCall(PetscFree(olx));
  PetscCall(PetscFree(oly));
  PetscCall(PetscFree(olz));
  PetscCall(PetscFree(osx));
  PetscCall(PetscFree(osy));
  PetscCall(PetscFree(osz));
  PetscCall(PetscFree(oex));
  PetscCall(PetscFree(oey));
  PetscCall(PetscFree(oez));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DAView_3DVTK_PStructuredGrid(DM da, const char file_prefix[], const char local_file_prefix[])
{
  MPI_Comm    comm;
  PetscMPIInt size, rank;
  char        vtk_filename[PETSC_MAX_PATH_LEN];
  FILE       *vtk_fp     = NULL;
  const char *byte_order = PetscBinaryBigEndian() ? "BigEndian" : "LittleEndian";
  PetscInt    M, N, P, si, sj, sk, nx, ny, nz;
  PetscInt    i, dofs;

  PetscFunctionBeginUser;
  /* only rank-0 generates this file */
  PetscCall(PetscObjectGetComm((PetscObject)da, &comm));
  PetscCallMPI(MPI_Comm_size(comm, &size));
  PetscCallMPI(MPI_Comm_rank(comm, &rank));

  if (rank != 0) PetscFunctionReturn(PETSC_SUCCESS);

  /* create file name */
  PetscCall(PetscSNPrintf(vtk_filename, sizeof(vtk_filename), "%s.pvts", file_prefix));
  vtk_fp = fopen(vtk_filename, "w");
  PetscCheck(vtk_fp, PETSC_COMM_SELF, PETSC_ERR_SYS, "Cannot open file = %s ", vtk_filename);

  /* (VTK) generate pvts header */
  PetscCall(PetscFPrintf(PETSC_COMM_SELF, vtk_fp, "<?xml version=\"1.0\"?>\n"));

  PetscCall(PetscFPrintf(PETSC_COMM_SELF, vtk_fp, "<VTKFile type=\"PStructuredGrid\" version=\"0.1\" byte_order=\"%s\">\n", byte_order));

  /* define size of the nodal mesh based on the cell DM */
  PetscCall(DMDAGetInfo(da, 0, &M, &N, &P, 0, 0, 0, &dofs, 0, 0, 0, 0, 0));
  PetscCall(DMDAGetGhostCorners(da, &si, &sj, &sk, &nx, &ny, &nz));
  PetscCall(PetscFPrintf(PETSC_COMM_SELF, vtk_fp, "  <PStructuredGrid GhostLevel=\"1\" WholeExtent=\"%d %" PetscInt_FMT " %d %" PetscInt_FMT " %d %" PetscInt_FMT "\">\n", 0, M - 1, 0, N - 1, 0, P - 1)); /* note overlap = 1 for Q1 */

  /* DUMP THE CELL REFERENCES */
  PetscCall(PetscFPrintf(PETSC_COMM_SELF, vtk_fp, "    <PCellData>\n"));
  PetscCall(PetscFPrintf(PETSC_COMM_SELF, vtk_fp, "    </PCellData>\n"));

  PetscCall(PetscFPrintf(PETSC_COMM_SELF, vtk_fp, "    <PPoints>\n"));
  PetscCall(PetscFPrintf(PETSC_COMM_SELF, vtk_fp, "      <PDataArray type=\"Float64\" Name=\"Points\" NumberOfComponents=\"%d\"/>\n", NSD));
  PetscCall(PetscFPrintf(PETSC_COMM_SELF, vtk_fp, "    </PPoints>\n"));

  PetscCall(PetscFPrintf(PETSC_COMM_SELF, vtk_fp, "    <PPointData>\n"));
  for (i = 0; i < dofs; i++) {
    const char *fieldname;
    PetscCall(DMDAGetFieldName(da, i, &fieldname));
    PetscCall(PetscFPrintf(PETSC_COMM_SELF, vtk_fp, "      <PDataArray type=\"Float64\" Name=\"%s\" NumberOfComponents=\"1\"/>\n", fieldname));
  }
  PetscCall(PetscFPrintf(PETSC_COMM_SELF, vtk_fp, "    </PPointData>\n"));

  /* write out the parallel information */
  PetscCall(DAViewVTK_write_PieceExtend(vtk_fp, 2, da, local_file_prefix));

  /* close the file */
  PetscCall(PetscFPrintf(PETSC_COMM_SELF, vtk_fp, "  </PStructuredGrid>\n"));
  PetscCall(PetscFPrintf(PETSC_COMM_SELF, vtk_fp, "</VTKFile>\n"));

  if (vtk_fp) {
    fclose(vtk_fp);
    vtk_fp = NULL;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DAView3DPVTS(DM da, Vec x, const char NAME[])
{
  char vts_filename[PETSC_MAX_PATH_LEN];
  char pvts_filename[PETSC_MAX_PATH_LEN];

  PetscFunctionBeginUser;
  PetscCall(PetscSNPrintf(vts_filename, sizeof(vts_filename), "%s-mesh", NAME));
  PetscCall(DAView_3DVTK_StructuredGrid_appended(da, x, vts_filename));

  PetscCall(PetscSNPrintf(pvts_filename, sizeof(pvts_filename), "%s-mesh", NAME));
  PetscCall(DAView_3DVTK_PStructuredGrid(da, pvts_filename, vts_filename));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode KSPMonitorStokesBlocks(KSP ksp, PetscInt n, PetscReal rnorm, void *dummy)
{
  PetscReal norms[4];
  Vec       Br, v, w;
  Mat       A;

  PetscFunctionBeginUser;
  PetscCall(KSPGetOperators(ksp, &A, NULL));
  PetscCall(MatCreateVecs(A, &w, &v));

  PetscCall(KSPBuildResidual(ksp, v, w, &Br));

  PetscCall(VecStrideNorm(Br, 0, NORM_2, &norms[0]));
  PetscCall(VecStrideNorm(Br, 1, NORM_2, &norms[1]));
  PetscCall(VecStrideNorm(Br, 2, NORM_2, &norms[2]));
  PetscCall(VecStrideNorm(Br, 3, NORM_2, &norms[3]));

  PetscCall(VecDestroy(&v));
  PetscCall(VecDestroy(&w));

  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "%3" PetscInt_FMT " KSP Component U,V,W,P residual norm [ %1.12e, %1.12e, %1.12e, %1.12e ]\n", n, (double)norms[0], (double)norms[1], (double)norms[2], (double)norms[3]));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCMGSetupViaCoarsen(PC pc, DM da_fine)
{
  PetscInt              nlevels, k;
  PETSC_UNUSED PetscInt finest;
  DM                   *da_list, *daclist;
  Mat                   R;

  PetscFunctionBeginUser;
  nlevels = 1;
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-levels", &nlevels, 0));

  PetscCall(PetscMalloc1(nlevels, &da_list));
  for (k = 0; k < nlevels; k++) da_list[k] = NULL;
  PetscCall(PetscMalloc1(nlevels, &daclist));
  for (k = 0; k < nlevels; k++) daclist[k] = NULL;

  /* finest grid is nlevels - 1 */
  finest     = nlevels - 1;
  daclist[0] = da_fine;
  PetscCall(PetscObjectReference((PetscObject)da_fine));
  PetscCall(DMCoarsenHierarchy(da_fine, nlevels - 1, &daclist[1]));
  for (k = 0; k < nlevels; k++) {
    da_list[k] = daclist[nlevels - 1 - k];
    PetscCall(DMDASetUniformCoordinates(da_list[k], 0.0, 1.0, 0.0, 1.0, 0.0, 1.0));
  }

  PetscCall(PCMGSetLevels(pc, nlevels, NULL));
  PetscCall(PCMGSetType(pc, PC_MG_MULTIPLICATIVE));
  PetscCall(PCMGSetGalerkin(pc, PC_MG_GALERKIN_PMAT));

  for (k = 1; k < nlevels; k++) {
    PetscCall(DMCreateInterpolation(da_list[k - 1], da_list[k], &R, NULL));
    PetscCall(PCMGSetInterpolation(pc, k, R));
    PetscCall(MatDestroy(&R));
  }

  /* tidy up */
  for (k = 0; k < nlevels; k++) PetscCall(DMDestroy(&da_list[k]));
  PetscCall(PetscFree(da_list));
  PetscCall(PetscFree(daclist));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode solve_stokes_3d_coupled(PetscInt mx, PetscInt my, PetscInt mz)
{
  DM             da_Stokes;
  PetscInt       u_dof, p_dof, dof, stencil_width;
  Mat            A, B;
  DM             vel_cda;
  Vec            vel_coords;
  PetscInt       p;
  Vec            f, X, X1;
  DMDACoor3d  ***_vel_coords;
  PetscInt       its;
  KSP            ksp_S;
  PetscInt       model_definition = 0;
  PetscInt       ei, ej, ek, sex, sey, sez, Imx, Jmy, Kmz;
  CellProperties cell_properties;
  PetscBool      write_output = PETSC_FALSE, resolve = PETSC_FALSE;

  PetscFunctionBeginUser;
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-resolve", &resolve, NULL));
  /* Generate the da for velocity and pressure */
  /* Num nodes in each direction is mx+1, my+1, mz+1 */
  u_dof         = U_DOFS; /* Vx, Vy - velocities */
  p_dof         = P_DOFS; /* p - pressure */
  dof           = u_dof + p_dof;
  stencil_width = 1;
  PetscCall(DMDACreate3d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DMDA_STENCIL_BOX, mx + 1, my + 1, mz + 1, PETSC_DECIDE, PETSC_DECIDE, PETSC_DECIDE, dof, stencil_width, NULL, NULL, NULL, &da_Stokes));
  PetscCall(DMSetMatType(da_Stokes, MATAIJ));
  PetscCall(DMSetFromOptions(da_Stokes));
  PetscCall(DMSetUp(da_Stokes));
  PetscCall(DMDASetFieldName(da_Stokes, 0, "Vx"));
  PetscCall(DMDASetFieldName(da_Stokes, 1, "Vy"));
  PetscCall(DMDASetFieldName(da_Stokes, 2, "Vz"));
  PetscCall(DMDASetFieldName(da_Stokes, 3, "P"));

  /* unit box [0,1] x [0,1] x [0,1] */
  PetscCall(DMDASetUniformCoordinates(da_Stokes, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0));

  /* create quadrature point info for PDE coefficients */
  PetscCall(CellPropertiesCreate(da_Stokes, &cell_properties));

  /* interpolate the coordinates to quadrature points */
  PetscCall(DMGetCoordinateDM(da_Stokes, &vel_cda));
  PetscCall(DMGetCoordinatesLocal(da_Stokes, &vel_coords));
  PetscCall(DMDAVecGetArray(vel_cda, vel_coords, &_vel_coords));
  PetscCall(DMDAGetElementsCorners(da_Stokes, &sex, &sey, &sez));
  PetscCall(DMDAGetElementsSizes(da_Stokes, &Imx, &Jmy, &Kmz));
  for (ek = sez; ek < sez + Kmz; ek++) {
    for (ej = sey; ej < sey + Jmy; ej++) {
      for (ei = sex; ei < sex + Imx; ei++) {
        /* get coords for the element */
        PetscInt                ngp;
        PetscScalar             gp_xi[GAUSS_POINTS][NSD], gp_weight[GAUSS_POINTS];
        PetscScalar             el_coords[NSD * NODES_PER_EL];
        GaussPointCoefficients *cell;

        PetscCall(CellPropertiesGetCell(cell_properties, ei, ej, ek, &cell));
        PetscCall(GetElementCoords3D(_vel_coords, ei, ej, ek, el_coords));
        ConstructGaussQuadrature3D(&ngp, gp_xi, gp_weight);

        for (p = 0; p < GAUSS_POINTS; p++) {
          PetscScalar xi_p[NSD], Ni_p[NODES_PER_EL];
          PetscScalar gp_x, gp_y, gp_z;
          PetscInt    n;

          xi_p[0] = gp_xi[p][0];
          xi_p[1] = gp_xi[p][1];
          xi_p[2] = gp_xi[p][2];
          ShapeFunctionQ13D_Evaluate(xi_p, Ni_p);

          gp_x = gp_y = gp_z = 0.0;
          for (n = 0; n < NODES_PER_EL; n++) {
            gp_x = gp_x + Ni_p[n] * el_coords[NSD * n];
            gp_y = gp_y + Ni_p[n] * el_coords[NSD * n + 1];
            gp_z = gp_z + Ni_p[n] * el_coords[NSD * n + 2];
          }
          cell->gp_coords[NSD * p]     = gp_x;
          cell->gp_coords[NSD * p + 1] = gp_y;
          cell->gp_coords[NSD * p + 2] = gp_z;
        }
      }
    }
  }

  PetscCall(PetscOptionsGetInt(NULL, NULL, "-model", &model_definition, NULL));

  switch (model_definition) {
  case 0: /* isoviscous */
    for (ek = sez; ek < sez + Kmz; ek++) {
      for (ej = sey; ej < sey + Jmy; ej++) {
        for (ei = sex; ei < sex + Imx; ei++) {
          GaussPointCoefficients *cell;

          PetscCall(CellPropertiesGetCell(cell_properties, ei, ej, ek, &cell));
          for (p = 0; p < GAUSS_POINTS; p++) {
            PetscReal coord_x = PetscRealPart(cell->gp_coords[NSD * p]);
            PetscReal coord_y = PetscRealPart(cell->gp_coords[NSD * p + 1]);
            PetscReal coord_z = PetscRealPart(cell->gp_coords[NSD * p + 2]);

            cell->eta[p] = 1.0;

            cell->fx[p] = 0.0 * coord_x;
            cell->fy[p] = -PetscSinReal(2.2 * PETSC_PI * coord_y) * PetscCosReal(1.0 * PETSC_PI * coord_x);
            cell->fz[p] = 0.0 * coord_z;
            cell->hc[p] = 0.0;
          }
        }
      }
    }
    break;

  case 1: /* manufactured */
    for (ek = sez; ek < sez + Kmz; ek++) {
      for (ej = sey; ej < sey + Jmy; ej++) {
        for (ei = sex; ei < sex + Imx; ei++) {
          PetscReal               eta, Fm[NSD], Fc, pos[NSD];
          GaussPointCoefficients *cell;

          PetscCall(CellPropertiesGetCell(cell_properties, ei, ej, ek, &cell));
          for (p = 0; p < GAUSS_POINTS; p++) {
            PetscReal coord_x = PetscRealPart(cell->gp_coords[NSD * p]);
            PetscReal coord_y = PetscRealPart(cell->gp_coords[NSD * p + 1]);
            PetscReal coord_z = PetscRealPart(cell->gp_coords[NSD * p + 2]);

            pos[0] = coord_x;
            pos[1] = coord_y;
            pos[2] = coord_z;

            evaluate_MS_FrankKamentski(pos, NULL, NULL, &eta, Fm, &Fc);
            cell->eta[p] = eta;

            cell->fx[p] = Fm[0];
            cell->fy[p] = Fm[1];
            cell->fz[p] = Fm[2];
            cell->hc[p] = Fc;
          }
        }
      }
    }
    break;

  case 2: /* solcx */
    for (ek = sez; ek < sez + Kmz; ek++) {
      for (ej = sey; ej < sey + Jmy; ej++) {
        for (ei = sex; ei < sex + Imx; ei++) {
          GaussPointCoefficients *cell;

          PetscCall(CellPropertiesGetCell(cell_properties, ei, ej, ek, &cell));
          for (p = 0; p < GAUSS_POINTS; p++) {
            PetscReal coord_x = PetscRealPart(cell->gp_coords[NSD * p]);
            PetscReal coord_y = PetscRealPart(cell->gp_coords[NSD * p + 1]);
            PetscReal coord_z = PetscRealPart(cell->gp_coords[NSD * p + 2]);

            cell->eta[p] = 1.0;

            cell->fx[p] = 0.0;
            cell->fy[p] = -PetscSinReal(3.0 * PETSC_PI * coord_y) * PetscCosReal(1.0 * PETSC_PI * coord_x);
            cell->fz[p] = 0.0 * coord_z;
            cell->hc[p] = 0.0;
          }
        }
      }
    }
    break;

  case 3: /* sinker */
    for (ek = sez; ek < sez + Kmz; ek++) {
      for (ej = sey; ej < sey + Jmy; ej++) {
        for (ei = sex; ei < sex + Imx; ei++) {
          GaussPointCoefficients *cell;

          PetscCall(CellPropertiesGetCell(cell_properties, ei, ej, ek, &cell));
          for (p = 0; p < GAUSS_POINTS; p++) {
            PetscReal xp = PetscRealPart(cell->gp_coords[NSD * p]);
            PetscReal yp = PetscRealPart(cell->gp_coords[NSD * p + 1]);
            PetscReal zp = PetscRealPart(cell->gp_coords[NSD * p + 2]);

            cell->eta[p] = 1.0e-2;
            cell->fx[p]  = 0.0;
            cell->fy[p]  = 0.0;
            cell->fz[p]  = 0.0;
            cell->hc[p]  = 0.0;

            if ((PetscAbs(xp - 0.5) < 0.2) && (PetscAbs(yp - 0.5) < 0.2) && (PetscAbs(zp - 0.5) < 0.2)) {
              cell->eta[p] = 1.0;
              cell->fz[p]  = 1.0;
            }
          }
        }
      }
    }
    break;

  case 4: /* subdomain jumps */
    for (ek = sez; ek < sez + Kmz; ek++) {
      for (ej = sey; ej < sey + Jmy; ej++) {
        for (ei = sex; ei < sex + Imx; ei++) {
          PetscReal               opts_mag, opts_eta0;
          PetscInt                px, py, pz;
          PetscBool               jump;
          PetscMPIInt             rr;
          GaussPointCoefficients *cell;

          opts_mag  = 1.0;
          opts_eta0 = 1.e-2;

          PetscCall(PetscOptionsGetReal(NULL, NULL, "-jump_eta0", &opts_eta0, NULL));
          PetscCall(PetscOptionsGetReal(NULL, NULL, "-jump_magnitude", &opts_mag, NULL));
          PetscCall(DMDAGetInfo(da_Stokes, NULL, NULL, NULL, NULL, &px, &py, &pz, NULL, NULL, NULL, NULL, NULL, NULL));
          rr = PetscGlobalRank % (px * py);
          if (px % 2) jump = (PetscBool)(rr % 2);
          else jump = (PetscBool)((rr / px) % 2 ? rr % 2 : !(rr % 2));
          rr = PetscGlobalRank / (px * py);
          if (rr % 2) jump = (PetscBool)!jump;
          PetscCall(CellPropertiesGetCell(cell_properties, ei, ej, ek, &cell));
          for (p = 0; p < GAUSS_POINTS; p++) {
            PetscReal xp = PetscRealPart(cell->gp_coords[NSD * p]);
            PetscReal yp = PetscRealPart(cell->gp_coords[NSD * p + 1]);

            cell->eta[p] = jump ? PetscPowReal(10.0, opts_mag) : opts_eta0;
            cell->fx[p]  = 0.0;
            cell->fy[p]  = -PetscSinReal(2.2 * PETSC_PI * yp) * PetscCosReal(1.0 * PETSC_PI * xp);
            cell->fz[p]  = 0.0;
            cell->hc[p]  = 0.0;
          }
        }
      }
    }
    break;
  default:
    SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_SUP, "No default model is supported. Choose either -model {0,1,2,3}");
  }

  PetscCall(DMDAVecRestoreArray(vel_cda, vel_coords, &_vel_coords));

  /* Generate a matrix with the correct non-zero pattern of type AIJ. This will work in parallel and serial */
  PetscCall(DMCreateMatrix(da_Stokes, &A));
  PetscCall(DMCreateMatrix(da_Stokes, &B));
  PetscCall(DMCreateGlobalVector(da_Stokes, &X));
  PetscCall(DMCreateGlobalVector(da_Stokes, &f));

  /* assemble A11 */
  PetscCall(MatZeroEntries(A));
  PetscCall(MatZeroEntries(B));
  PetscCall(VecZeroEntries(f));

  PetscCall(AssembleA_Stokes(A, da_Stokes, cell_properties));
  PetscCall(MatViewFromOptions(A, NULL, "-amat_view"));
  PetscCall(AssembleA_PCStokes(B, da_Stokes, cell_properties));
  PetscCall(MatViewFromOptions(B, NULL, "-bmat_view"));
  /* build force vector */
  PetscCall(AssembleF_Stokes(f, da_Stokes, cell_properties));

  /* SOLVE */
  PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp_S));
  PetscCall(KSPSetOptionsPrefix(ksp_S, "stokes_"));
  PetscCall(KSPSetOperators(ksp_S, A, B));
  PetscCall(KSPSetFromOptions(ksp_S));

  {
    PC             pc;
    const PetscInt ufields[] = {0, 1, 2}, pfields[] = {3};
    PetscCall(KSPGetPC(ksp_S, &pc));
    PetscCall(PCFieldSplitSetBlockSize(pc, 4));
    PetscCall(PCFieldSplitSetFields(pc, "u", 3, ufields, ufields));
    PetscCall(PCFieldSplitSetFields(pc, "p", 1, pfields, pfields));
  }

  {
    PC        pc;
    PetscBool same = PETSC_FALSE;
    PetscCall(KSPGetPC(ksp_S, &pc));
    PetscCall(PetscObjectTypeCompare((PetscObject)pc, PCMG, &same));
    if (same) PetscCall(PCMGSetupViaCoarsen(pc, da_Stokes));
  }

  {
    PC        pc;
    PetscBool same = PETSC_FALSE;
    PetscCall(KSPGetPC(ksp_S, &pc));
    PetscCall(PetscObjectTypeCompare((PetscObject)pc, PCBDDC, &same));
    if (same) PetscCall(KSPSetOperators(ksp_S, A, A));
  }

  {
    PetscBool stokes_monitor = PETSC_FALSE;
    PetscCall(PetscOptionsGetBool(NULL, NULL, "-stokes_ksp_monitor_blocks", &stokes_monitor, 0));
    if (stokes_monitor) PetscCall(KSPMonitorSet(ksp_S, KSPMonitorStokesBlocks, NULL, NULL));
  }

  if (resolve) {
    /* Test changing matrix data structure and resolve */
    PetscCall(VecDuplicate(X, &X1));
  }

  PetscCall(KSPSolve(ksp_S, f, X));
  if (resolve) {
    Mat C;
    PetscCall(MatDuplicate(A, MAT_COPY_VALUES, &C));
    PetscCall(KSPSetOperators(ksp_S, C, C));
    PetscCall(KSPSolve(ksp_S, f, X1));
    PetscCall(MatDestroy(&C));
    PetscCall(VecDestroy(&X1));
  }

  PetscCall(PetscOptionsGetBool(NULL, NULL, "-write_pvts", &write_output, NULL));
  if (write_output) PetscCall(DAView3DPVTS(da_Stokes, X, "up"));
  {
    PetscBool flg = PETSC_FALSE;
    char      filename[PETSC_MAX_PATH_LEN];
    PetscCall(PetscOptionsGetString(NULL, NULL, "-write_binary", filename, sizeof(filename), &flg));
    if (flg) {
      PetscViewer viewer;
      /* PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD,filename[0]?filename:"ex42-binaryoutput",FILE_MODE_WRITE,&viewer)); */
      PetscCall(PetscViewerVTKOpen(PETSC_COMM_WORLD, "ex42.vts", FILE_MODE_WRITE, &viewer));
      PetscCall(VecView(X, viewer));
      PetscCall(PetscViewerDestroy(&viewer));
    }
  }
  PetscCall(KSPGetIterationNumber(ksp_S, &its));

  /* verify */
  if (model_definition == 1) {
    DM  da_Stokes_analytic;
    Vec X_analytic;

    PetscCall(DMDACreateManufacturedSolution(mx, my, mz, &da_Stokes_analytic, &X_analytic));
    if (write_output) PetscCall(DAView3DPVTS(da_Stokes_analytic, X_analytic, "ms"));
    PetscCall(DMDAIntegrateErrors3D(da_Stokes_analytic, X, X_analytic));
    if (write_output) PetscCall(DAView3DPVTS(da_Stokes, X, "up2"));
    PetscCall(DMDestroy(&da_Stokes_analytic));
    PetscCall(VecDestroy(&X_analytic));
  }

  PetscCall(KSPDestroy(&ksp_S));
  PetscCall(VecDestroy(&X));
  PetscCall(VecDestroy(&f));
  PetscCall(MatDestroy(&A));
  PetscCall(MatDestroy(&B));

  PetscCall(CellPropertiesDestroy(&cell_properties));
  PetscCall(DMDestroy(&da_Stokes));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **args)
{
  PetscInt mx, my, mz;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &args, NULL, help));
  mx = my = mz = 10;
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-mx", &mx, NULL));
  my = mx;
  mz = mx;
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-my", &my, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-mz", &mz, NULL));
  PetscCall(solve_stokes_3d_coupled(mx, my, mz));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:
      args: -stokes_ksp_monitor_short -stokes_ksp_converged_reason -stokes_pc_type lu

   test:
      suffix: 2
      nsize: 3
      args: -stokes_ksp_monitor_short -stokes_ksp_converged_reason -stokes_pc_type redundant -stokes_redundant_pc_type lu

   test:
      suffix: bddc_stokes
      nsize: 6
      args: -mx 5 -my 4 -mz 3 -stokes_ksp_monitor_short -stokes_ksp_converged_reason -stokes_pc_type bddc -dm_mat_type is -stokes_pc_bddc_dirichlet_pc_type svd -stokes_pc_bddc_neumann_pc_type svd -stokes_pc_bddc_coarse_redundant_pc_type svd

   test:
      suffix: bddc_stokes_deluxe
      nsize: 6
      args: -mx 5 -my 4 -mz 3 -stokes_ksp_monitor_short -stokes_ksp_converged_reason -stokes_pc_type bddc -dm_mat_type is -stokes_pc_bddc_dirichlet_pc_type svd -stokes_pc_bddc_neumann_pc_type svd -stokes_pc_bddc_coarse_redundant_pc_type svd -stokes_pc_bddc_use_deluxe_scaling -stokes_sub_schurs_posdef 0 -stokes_sub_schurs_symmetric -stokes_sub_schurs_mat_solver_type petsc

   test:
      requires: !single
      suffix: bddc_stokes_subdomainjump_deluxe
      nsize: 9
      args: -model 4 -jump_magnitude 4 -mx 6 -my 6 -mz 2 -stokes_ksp_monitor_short -stokes_ksp_converged_reason -stokes_pc_type bddc -dm_mat_type is -stokes_pc_bddc_use_deluxe_scaling -stokes_sub_schurs_posdef 0 -stokes_sub_schurs_symmetric -stokes_sub_schurs_mat_solver_type petsc -stokes_pc_bddc_schur_layers 1

   test:
      requires: !single
      suffix: 3
      args: -stokes_ksp_converged_reason -stokes_pc_type fieldsplit -resolve

   test:
      suffix: tut_1
      nsize: 4
      requires: !single
      args: -stokes_ksp_monitor

   test:
      suffix: tut_2
      nsize: 4
      requires: !single
      args: -stokes_ksp_monitor -stokes_pc_type fieldsplit -stokes_pc_fieldsplit_type schur

   test:
      suffix: tut_3
      nsize: 4
      requires: !single
      args: -mx 20 -stokes_ksp_monitor -stokes_pc_type fieldsplit -stokes_pc_fieldsplit_type schur

TEST*/
