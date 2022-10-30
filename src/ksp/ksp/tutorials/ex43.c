static char help[] = "Solves the incompressible, variable viscosity Stokes equation in 2d on the unit domain \n\
using Q1Q1 elements, stabilized with Bochev's polynomial projection method. \n\
The models defined utilise free slip boundary conditions on all sides. \n\
Options: \n"
                     "\
     -mx : Number of elements in the x-direction \n\
     -my : Number of elements in the y-direction \n\
     -o : Specify output filename for solution (will be petsc binary format or paraview format if the extension is .vts) \n\
     -gnuplot : Output Gauss point coordinates, coefficients and u,p solution in gnuplot format \n\
     -glvis : Visualizes coefficients and u,p solution through GLVIs (use -viewer_glvis_dmda_bs 2,1 to visualize velocity as a vector)\n\
     -c_str : Indicates the structure of the coefficients to use \n"
                     "\
          -c_str 0 => Coefficient definition for an analytic solution with a vertical jump in viscosity at x = xc \n\
                      This problem is driven by the forcing function f(x,y) = (0, sin(nz pi y)cos(pi x) \n\
                         Parameters: \n\
                              -solcx_eta0  : Viscosity to the left of the interface \n\
                              -solcx_eta1  : Viscosity to the right of the interface \n\
                              -solcx_xc    : Location of the interface \n\
                              -solcx_nz    : Wavenumber in the y direction \n"
                     "\
          -c_str 1 => Coefficient definition for a dense rectangular blob located at the center of the domain \n\
                         Parameters: \n\
                              -sinker_eta0 : Viscosity of the background fluid \n\
                              -sinker_eta1 : Viscosity of the blob \n\
                              -sinker_dx   : Width of the blob \n\
                              -sinker_dy   : Height of the blob \n"
                     "\
          -c_str 2 => Coefficient definition for a dense circular blob located at the center of the domain \n\
                         Parameters: \n\
                              -sinker_eta0 : Viscosity of the background fluid \n\
                              -sinker_eta1 : Viscosity of the blob \n\
                              -sinker_r    : Radius of the blob \n"
                     "\
          -c_str 3 => Coefficient definition for a dense circular and rectangular inclusion (located at the center of the domain) \n\
                              -sinker_eta0 : Viscosity of the background fluid \n\
                              -sinker_eta1 : Viscosity of the two inclusions \n\
                              -sinker_r    : Radius of the circular inclusion \n\
                              -sinker_c0x  : Origin (x-coord) of the circular inclusion \n\
                              -sinker_c0y  : Origin (y-coord) of the circular inclusion \n\
                              -sinker_dx   : Width of the rectangular inclusion \n\
                              -sinker_dy   : Height of the rectangular inclusion \n\
                              -sinker_phi  : Rotation angle of the rectangular inclusion \n"
                     "\
          -c_str 4 => Coefficient definition for checkerboard jumps aligned with the domain decomposition \n\
                              -jump_eta0      : Viscosity for black subdomains \n\
                              -jump_magnitude : Magnitude of jumps. White subdomains will have eta = eta0*10^magnitude \n\
                              -jump_nz        : Wavenumber in the y direction for rhs \n"
                     "\
     -use_gp_coords : Evaluate the viscosity and force term at the global coordinates of each quadrature point \n\
                      By default, the viscosity and force term are evaluated at the element center and applied as a constant over the entire element \n";

/* Contributed by Dave May */

#include <petscksp.h>
#include <petscdm.h>
#include <petscdmda.h>

/* A Maple-generated exact solution created by Mirko Velic (mirko.velic@sci.monash.edu.au) */
#include "ex43-solcx.h"

static PetscErrorCode DMDABCApplyFreeSlip(DM, Mat, Vec);

#define NSD          2 /* number of spatial dimensions */
#define NODES_PER_EL 4 /* nodes per element */
#define U_DOFS       2 /* degrees of freedom per velocity node */
#define P_DOFS       1 /* degrees of freedom per pressure node */
#define GAUSS_POINTS 4

/* Gauss point based evaluation 8+4+4+4 = 20 */
typedef struct {
  PetscScalar gp_coords[2 * GAUSS_POINTS];
  PetscScalar eta[GAUSS_POINTS];
  PetscScalar fx[GAUSS_POINTS];
  PetscScalar fy[GAUSS_POINTS];
} GaussPointCoefficients;

typedef struct {
  PetscScalar u_dof;
  PetscScalar v_dof;
  PetscScalar p_dof;
} StokesDOF;

static PetscErrorCode glvis_extract_eta(PetscObject oV, PetscInt nf, PetscObject oVf[], void *ctx)
{
  DM                       properties_da = (DM)(ctx), stokes_da;
  Vec                      V = (Vec)oV, *Vf = (Vec *)oVf;
  GaussPointCoefficients **props;
  PetscInt                 sex, sey, mx, my;
  PetscInt                 ei, ej, p, cum;
  PetscScalar             *array;

  PetscFunctionBeginUser;
  PetscCall(VecGetDM(Vf[0], &stokes_da));
  PetscCall(DMDAVecGetArrayRead(properties_da, V, &props));
  PetscCall(DMDAGetElementsCorners(stokes_da, &sex, &sey, NULL));
  PetscCall(DMDAGetElementsSizes(stokes_da, &mx, &my, NULL));
  PetscCall(VecGetArray(Vf[0], &array));
  cum = 0;
  for (ej = sey; ej < sey + my; ej++) {
    for (ei = sex; ei < sex + mx; ei++) {
      for (p = 0; p < GAUSS_POINTS; p++) array[cum++] = props[ej][ei].eta[p];
    }
  }
  PetscCall(VecRestoreArray(Vf[0], &array));
  PetscCall(DMDAVecRestoreArrayRead(properties_da, V, &props));
  PetscFunctionReturn(0);
}

/*
 Element: Local basis function ordering
 1-----2
 |     |
 |     |
 0-----3
 */
static void ConstructQ12D_Ni(PetscScalar _xi[], PetscScalar Ni[])
{
  PetscScalar xi  = _xi[0];
  PetscScalar eta = _xi[1];

  Ni[0] = 0.25 * (1.0 - xi) * (1.0 - eta);
  Ni[1] = 0.25 * (1.0 - xi) * (1.0 + eta);
  Ni[2] = 0.25 * (1.0 + xi) * (1.0 + eta);
  Ni[3] = 0.25 * (1.0 + xi) * (1.0 - eta);
}

static void ConstructQ12D_GNi(PetscScalar _xi[], PetscScalar GNi[][NODES_PER_EL])
{
  PetscScalar xi  = _xi[0];
  PetscScalar eta = _xi[1];

  GNi[0][0] = -0.25 * (1.0 - eta);
  GNi[0][1] = -0.25 * (1.0 + eta);
  GNi[0][2] = 0.25 * (1.0 + eta);
  GNi[0][3] = 0.25 * (1.0 - eta);

  GNi[1][0] = -0.25 * (1.0 - xi);
  GNi[1][1] = 0.25 * (1.0 - xi);
  GNi[1][2] = 0.25 * (1.0 + xi);
  GNi[1][3] = -0.25 * (1.0 + xi);
}

static void ConstructQ12D_GNx(PetscScalar GNi[][NODES_PER_EL], PetscScalar GNx[][NODES_PER_EL], PetscScalar coords[], PetscScalar *det_J)
{
  PetscScalar J00, J01, J10, J11, J;
  PetscScalar iJ00, iJ01, iJ10, iJ11;
  PetscInt    i;

  J00 = J01 = J10 = J11 = 0.0;
  for (i = 0; i < NODES_PER_EL; i++) {
    PetscScalar cx = coords[2 * i];
    PetscScalar cy = coords[2 * i + 1];

    J00 = J00 + GNi[0][i] * cx; /* J_xx = dx/dxi */
    J01 = J01 + GNi[0][i] * cy; /* J_xy = dy/dxi */
    J10 = J10 + GNi[1][i] * cx; /* J_yx = dx/deta */
    J11 = J11 + GNi[1][i] * cy; /* J_yy = dy/deta */
  }
  J = (J00 * J11) - (J01 * J10);

  iJ00 = J11 / J;
  iJ01 = -J01 / J;
  iJ10 = -J10 / J;
  iJ11 = J00 / J;

  for (i = 0; i < NODES_PER_EL; i++) {
    GNx[0][i] = GNi[0][i] * iJ00 + GNi[1][i] * iJ01;
    GNx[1][i] = GNi[0][i] * iJ10 + GNi[1][i] * iJ11;
  }

  if (det_J) *det_J = J;
}

static void ConstructGaussQuadrature(PetscInt *ngp, PetscScalar gp_xi[][2], PetscScalar gp_weight[])
{
  *ngp         = 4;
  gp_xi[0][0]  = -0.57735026919;
  gp_xi[0][1]  = -0.57735026919;
  gp_xi[1][0]  = -0.57735026919;
  gp_xi[1][1]  = 0.57735026919;
  gp_xi[2][0]  = 0.57735026919;
  gp_xi[2][1]  = 0.57735026919;
  gp_xi[3][0]  = 0.57735026919;
  gp_xi[3][1]  = -0.57735026919;
  gp_weight[0] = 1.0;
  gp_weight[1] = 1.0;
  gp_weight[2] = 1.0;
  gp_weight[3] = 1.0;
}

/*
i,j are the element indices
The unknown is a vector quantity.
The s[].c is used to indicate the degree of freedom.
*/
static PetscErrorCode DMDAGetElementEqnums_up(MatStencil s_u[], MatStencil s_p[], PetscInt i, PetscInt j)
{
  PetscFunctionBeginUser;
  /* velocity */
  /* node 0 */
  s_u[0].i = i;
  s_u[0].j = j;
  s_u[0].c = 0; /* Vx0 */
  s_u[1].i = i;
  s_u[1].j = j;
  s_u[1].c = 1; /* Vy0 */

  /* node 1 */
  s_u[2].i = i;
  s_u[2].j = j + 1;
  s_u[2].c = 0; /* Vx1 */
  s_u[3].i = i;
  s_u[3].j = j + 1;
  s_u[3].c = 1; /* Vy1 */

  /* node 2 */
  s_u[4].i = i + 1;
  s_u[4].j = j + 1;
  s_u[4].c = 0; /* Vx2 */
  s_u[5].i = i + 1;
  s_u[5].j = j + 1;
  s_u[5].c = 1; /* Vy2 */

  /* node 3 */
  s_u[6].i = i + 1;
  s_u[6].j = j;
  s_u[6].c = 0; /* Vx3 */
  s_u[7].i = i + 1;
  s_u[7].j = j;
  s_u[7].c = 1; /* Vy3 */

  /* pressure */
  s_p[0].i = i;
  s_p[0].j = j;
  s_p[0].c = 2; /* P0 */
  s_p[1].i = i;
  s_p[1].j = j + 1;
  s_p[1].c = 2; /* P1 */
  s_p[2].i = i + 1;
  s_p[2].j = j + 1;
  s_p[2].c = 2; /* P2 */
  s_p[3].i = i + 1;
  s_p[3].j = j;
  s_p[3].c = 2; /* P3 */
  PetscFunctionReturn(0);
}

static PetscErrorCode DMDAGetElementOwnershipRanges2d(DM da, PetscInt **_lx, PetscInt **_ly)
{
  PetscMPIInt  rank;
  PetscInt     proc_I, proc_J;
  PetscInt     cpu_x, cpu_y;
  PetscInt     local_mx, local_my;
  Vec          vlx, vly;
  PetscInt    *LX, *LY, i;
  PetscScalar *_a;
  Vec          V_SEQ;
  VecScatter   ctx;

  PetscFunctionBeginUser;
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));

  PetscCall(DMDAGetInfo(da, 0, 0, 0, 0, &cpu_x, &cpu_y, 0, 0, 0, 0, 0, 0, 0));

  proc_J = rank / cpu_x;
  proc_I = rank - cpu_x * proc_J;

  PetscCall(PetscMalloc1(cpu_x, &LX));
  PetscCall(PetscMalloc1(cpu_y, &LY));

  PetscCall(DMDAGetElementsSizes(da, &local_mx, &local_my, NULL));
  PetscCall(VecCreate(PETSC_COMM_WORLD, &vlx));
  PetscCall(VecSetSizes(vlx, PETSC_DECIDE, cpu_x));
  PetscCall(VecSetFromOptions(vlx));

  PetscCall(VecCreate(PETSC_COMM_WORLD, &vly));
  PetscCall(VecSetSizes(vly, PETSC_DECIDE, cpu_y));
  PetscCall(VecSetFromOptions(vly));

  PetscCall(VecSetValue(vlx, proc_I, (PetscScalar)(local_mx + 1.0e-9), INSERT_VALUES));
  PetscCall(VecSetValue(vly, proc_J, (PetscScalar)(local_my + 1.0e-9), INSERT_VALUES));
  PetscCall(VecAssemblyBegin(vlx); VecAssemblyEnd(vlx));
  PetscCall(VecAssemblyBegin(vly); VecAssemblyEnd(vly));

  PetscCall(VecScatterCreateToAll(vlx, &ctx, &V_SEQ));
  PetscCall(VecScatterBegin(ctx, vlx, V_SEQ, INSERT_VALUES, SCATTER_FORWARD));
  PetscCall(VecScatterEnd(ctx, vlx, V_SEQ, INSERT_VALUES, SCATTER_FORWARD));
  PetscCall(VecGetArray(V_SEQ, &_a));
  for (i = 0; i < cpu_x; i++) LX[i] = (PetscInt)PetscRealPart(_a[i]);
  PetscCall(VecRestoreArray(V_SEQ, &_a));
  PetscCall(VecScatterDestroy(&ctx));
  PetscCall(VecDestroy(&V_SEQ));

  PetscCall(VecScatterCreateToAll(vly, &ctx, &V_SEQ));
  PetscCall(VecScatterBegin(ctx, vly, V_SEQ, INSERT_VALUES, SCATTER_FORWARD));
  PetscCall(VecScatterEnd(ctx, vly, V_SEQ, INSERT_VALUES, SCATTER_FORWARD));
  PetscCall(VecGetArray(V_SEQ, &_a));
  for (i = 0; i < cpu_y; i++) LY[i] = (PetscInt)PetscRealPart(_a[i]);
  PetscCall(VecRestoreArray(V_SEQ, &_a));
  PetscCall(VecScatterDestroy(&ctx));
  PetscCall(VecDestroy(&V_SEQ));

  *_lx = LX;
  *_ly = LY;

  PetscCall(VecDestroy(&vlx));
  PetscCall(VecDestroy(&vly));
  PetscFunctionReturn(0);
}

static PetscErrorCode DMDACoordViewGnuplot2d(DM da, const char prefix[])
{
  DM           cda;
  Vec          coords;
  DMDACoor2d **_coords;
  PetscInt     si, sj, nx, ny, i, j;
  FILE        *fp;
  char         fname[PETSC_MAX_PATH_LEN];
  PetscMPIInt  rank;

  PetscFunctionBeginUser;
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
  PetscCall(PetscSNPrintf(fname, sizeof(fname), "%s-p%1.4d.dat", prefix, rank));
  PetscCall(PetscFOpen(PETSC_COMM_SELF, fname, "w", &fp));
  PetscCheck(fp, PETSC_COMM_SELF, PETSC_ERR_USER, "Cannot open file");

  PetscCall(PetscFPrintf(PETSC_COMM_SELF, fp, "### Element geometry for processor %1.4d ### \n", rank));

  PetscCall(DMGetCoordinateDM(da, &cda));
  PetscCall(DMGetCoordinatesLocal(da, &coords));
  PetscCall(DMDAVecGetArrayRead(cda, coords, &_coords));
  PetscCall(DMDAGetGhostCorners(cda, &si, &sj, 0, &nx, &ny, 0));
  for (j = sj; j < sj + ny - 1; j++) {
    for (i = si; i < si + nx - 1; i++) {
      PetscCall(PetscFPrintf(PETSC_COMM_SELF, fp, "%1.6e %1.6e \n", (double)PetscRealPart(_coords[j][i].x), (double)PetscRealPart(_coords[j][i].y)));
      PetscCall(PetscFPrintf(PETSC_COMM_SELF, fp, "%1.6e %1.6e \n", (double)PetscRealPart(_coords[j + 1][i].x), (double)PetscRealPart(_coords[j + 1][i].y)));
      PetscCall(PetscFPrintf(PETSC_COMM_SELF, fp, "%1.6e %1.6e \n", (double)PetscRealPart(_coords[j + 1][i + 1].x), (double)PetscRealPart(_coords[j + 1][i + 1].y)));
      PetscCall(PetscFPrintf(PETSC_COMM_SELF, fp, "%1.6e %1.6e \n", (double)PetscRealPart(_coords[j][i + 1].x), (double)PetscRealPart(_coords[j][i + 1].y)));
      PetscCall(PetscFPrintf(PETSC_COMM_SELF, fp, "%1.6e %1.6e \n\n", (double)PetscRealPart(_coords[j][i].x), (double)PetscRealPart(_coords[j][i].y)));
    }
  }
  PetscCall(DMDAVecRestoreArrayRead(cda, coords, &_coords));

  PetscCall(PetscFClose(PETSC_COMM_SELF, fp));
  PetscFunctionReturn(0);
}

static PetscErrorCode DMDAViewGnuplot2d(DM da, Vec fields, const char comment[], const char prefix[])
{
  DM           cda;
  Vec          coords, local_fields;
  DMDACoor2d **_coords;
  FILE        *fp;
  char         fname[PETSC_MAX_PATH_LEN];
  PetscMPIInt  rank;
  PetscInt     si, sj, nx, ny, i, j;
  PetscInt     n_dofs, d;
  PetscScalar *_fields;

  PetscFunctionBeginUser;
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
  PetscCall(PetscSNPrintf(fname, sizeof(fname), "%s-p%1.4d.dat", prefix, rank));
  PetscCall(PetscFOpen(PETSC_COMM_SELF, fname, "w", &fp));
  PetscCheck(fp, PETSC_COMM_SELF, PETSC_ERR_USER, "Cannot open file");

  PetscCall(PetscFPrintf(PETSC_COMM_SELF, fp, "### %s (processor %1.4d) ### \n", comment, rank));
  PetscCall(DMDAGetInfo(da, 0, 0, 0, 0, 0, 0, 0, &n_dofs, 0, 0, 0, 0, 0));
  PetscCall(PetscFPrintf(PETSC_COMM_SELF, fp, "### x y "));
  for (d = 0; d < n_dofs; d++) {
    const char *field_name;
    PetscCall(DMDAGetFieldName(da, d, &field_name));
    PetscCall(PetscFPrintf(PETSC_COMM_SELF, fp, "%s ", field_name));
  }
  PetscCall(PetscFPrintf(PETSC_COMM_SELF, fp, "###\n"));

  PetscCall(DMGetCoordinateDM(da, &cda));
  PetscCall(DMGetCoordinatesLocal(da, &coords));
  PetscCall(DMDAVecGetArray(cda, coords, &_coords));
  PetscCall(DMDAGetGhostCorners(cda, &si, &sj, 0, &nx, &ny, 0));

  PetscCall(DMCreateLocalVector(da, &local_fields));
  PetscCall(DMGlobalToLocalBegin(da, fields, INSERT_VALUES, local_fields));
  PetscCall(DMGlobalToLocalEnd(da, fields, INSERT_VALUES, local_fields));
  PetscCall(VecGetArray(local_fields, &_fields));

  for (j = sj; j < sj + ny; j++) {
    for (i = si; i < si + nx; i++) {
      PetscScalar coord_x, coord_y;
      PetscScalar field_d;

      coord_x = _coords[j][i].x;
      coord_y = _coords[j][i].y;

      PetscCall(PetscFPrintf(PETSC_COMM_SELF, fp, "%1.6e %1.6e ", (double)PetscRealPart(coord_x), (double)PetscRealPart(coord_y)));
      for (d = 0; d < n_dofs; d++) {
        field_d = _fields[n_dofs * ((i - si) + (j - sj) * (nx)) + d];
        PetscCall(PetscFPrintf(PETSC_COMM_SELF, fp, "%1.6e ", (double)PetscRealPart(field_d)));
      }
      PetscCall(PetscFPrintf(PETSC_COMM_SELF, fp, "\n"));
    }
  }
  PetscCall(VecRestoreArray(local_fields, &_fields));
  PetscCall(VecDestroy(&local_fields));

  PetscCall(DMDAVecRestoreArray(cda, coords, &_coords));

  PetscCall(PetscFClose(PETSC_COMM_SELF, fp));
  PetscFunctionReturn(0);
}

static PetscErrorCode DMDAViewCoefficientsGnuplot2d(DM da, Vec fields, const char comment[], const char prefix[])
{
  DM                       cda;
  Vec                      local_fields;
  FILE                    *fp;
  char                     fname[PETSC_MAX_PATH_LEN];
  PetscMPIInt              rank;
  PetscInt                 si, sj, nx, ny, i, j, p;
  PetscInt                 n_dofs, d;
  GaussPointCoefficients **_coefficients;

  PetscFunctionBeginUser;
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
  PetscCall(PetscSNPrintf(fname, sizeof(fname), "%s-p%1.4d.dat", prefix, rank));
  PetscCall(PetscFOpen(PETSC_COMM_SELF, fname, "w", &fp));
  PetscCheck(fp, PETSC_COMM_SELF, PETSC_ERR_USER, "Cannot open file");

  PetscCall(PetscFPrintf(PETSC_COMM_SELF, fp, "### %s (processor %1.4d) ### \n", comment, rank));
  PetscCall(DMDAGetInfo(da, 0, 0, 0, 0, 0, 0, 0, &n_dofs, 0, 0, 0, 0, 0));
  PetscCall(PetscFPrintf(PETSC_COMM_SELF, fp, "### x y "));
  for (d = 0; d < n_dofs; d++) {
    const char *field_name;
    PetscCall(DMDAGetFieldName(da, d, &field_name));
    PetscCall(PetscFPrintf(PETSC_COMM_SELF, fp, "%s ", field_name));
  }
  PetscCall(PetscFPrintf(PETSC_COMM_SELF, fp, "###\n"));

  PetscCall(DMGetCoordinateDM(da, &cda));
  PetscCall(DMDAGetGhostCorners(cda, &si, &sj, 0, &nx, &ny, 0));

  PetscCall(DMCreateLocalVector(da, &local_fields));
  PetscCall(DMGlobalToLocalBegin(da, fields, INSERT_VALUES, local_fields));
  PetscCall(DMGlobalToLocalEnd(da, fields, INSERT_VALUES, local_fields));
  PetscCall(DMDAVecGetArray(da, local_fields, &_coefficients));

  for (j = sj; j < sj + ny; j++) {
    for (i = si; i < si + nx; i++) {
      PetscScalar coord_x, coord_y;

      for (p = 0; p < GAUSS_POINTS; p++) {
        coord_x = _coefficients[j][i].gp_coords[2 * p];
        coord_y = _coefficients[j][i].gp_coords[2 * p + 1];

        PetscCall(PetscFPrintf(PETSC_COMM_SELF, fp, "%1.6e %1.6e ", (double)PetscRealPart(coord_x), (double)PetscRealPart(coord_y)));

        PetscCall(PetscFPrintf(PETSC_COMM_SELF, fp, "%1.6e %1.6e %1.6e", (double)PetscRealPart(_coefficients[j][i].eta[p]), (double)PetscRealPart(_coefficients[j][i].fx[p]), (double)PetscRealPart(_coefficients[j][i].fy[p])));
        PetscCall(PetscFPrintf(PETSC_COMM_SELF, fp, "\n"));
      }
    }
  }
  PetscCall(DMDAVecRestoreArray(da, local_fields, &_coefficients));
  PetscCall(VecDestroy(&local_fields));

  PetscCall(PetscFClose(PETSC_COMM_SELF, fp));
  PetscFunctionReturn(0);
}

static PetscInt ASS_MAP_wIwDI_uJuDJ(PetscInt wi, PetscInt wd, PetscInt w_NPE, PetscInt w_dof, PetscInt ui, PetscInt ud, PetscInt u_NPE, PetscInt u_dof)
{
  PetscInt ij;
  PetscInt r, c, nc;

  nc = u_NPE * u_dof;
  r  = w_dof * wi + wd;
  c  = u_dof * ui + ud;
  ij = r * nc + c;
  return ij;
}

/*
 D = [ 2.eta   0   0   ]
     [   0   2.eta 0   ]
     [   0     0   eta ]

 B = [ d_dx   0   ]
     [  0    d_dy ]
     [ d_dy  d_dx ]
*/
static void FormStressOperatorQ1(PetscScalar Ke[], PetscScalar coords[], PetscScalar eta[])
{
  PetscInt    ngp;
  PetscScalar gp_xi[GAUSS_POINTS][2];
  PetscScalar gp_weight[GAUSS_POINTS];
  PetscInt    p, i, j, k;
  PetscScalar GNi_p[NSD][NODES_PER_EL], GNx_p[NSD][NODES_PER_EL];
  PetscScalar J_p, tildeD[3];
  PetscScalar B[3][U_DOFS * NODES_PER_EL];

  /* define quadrature rule */
  ConstructGaussQuadrature(&ngp, gp_xi, gp_weight);

  /* evaluate integral */
  for (p = 0; p < ngp; p++) {
    ConstructQ12D_GNi(gp_xi[p], GNi_p);
    ConstructQ12D_GNx(GNi_p, GNx_p, coords, &J_p);

    for (i = 0; i < NODES_PER_EL; i++) {
      PetscScalar d_dx_i = GNx_p[0][i];
      PetscScalar d_dy_i = GNx_p[1][i];

      B[0][2 * i]     = d_dx_i;
      B[0][2 * i + 1] = 0.0;
      B[1][2 * i]     = 0.0;
      B[1][2 * i + 1] = d_dy_i;
      B[2][2 * i]     = d_dy_i;
      B[2][2 * i + 1] = d_dx_i;
    }

    tildeD[0] = 2.0 * gp_weight[p] * J_p * eta[p];
    tildeD[1] = 2.0 * gp_weight[p] * J_p * eta[p];
    tildeD[2] = gp_weight[p] * J_p * eta[p];

    /* form Bt tildeD B */
    /*
    Ke_ij = Bt_ik . D_kl . B_lj
          = B_ki . D_kl . B_lj
          = B_ki . D_kk . B_kj
    */
    for (i = 0; i < 8; i++) {
      for (j = 0; j < 8; j++) {
        for (k = 0; k < 3; k++) { /* Note D is diagonal for stokes */
          Ke[i + 8 * j] = Ke[i + 8 * j] + B[k][i] * tildeD[k] * B[k][j];
        }
      }
    }
  }
}

static void FormGradientOperatorQ1(PetscScalar Ke[], PetscScalar coords[])
{
  PetscInt    ngp;
  PetscScalar gp_xi[GAUSS_POINTS][2];
  PetscScalar gp_weight[GAUSS_POINTS];
  PetscInt    p, i, j, di;
  PetscScalar Ni_p[NODES_PER_EL];
  PetscScalar GNi_p[NSD][NODES_PER_EL], GNx_p[NSD][NODES_PER_EL];
  PetscScalar J_p, fac;

  /* define quadrature rule */
  ConstructGaussQuadrature(&ngp, gp_xi, gp_weight);

  /* evaluate integral */
  for (p = 0; p < ngp; p++) {
    ConstructQ12D_Ni(gp_xi[p], Ni_p);
    ConstructQ12D_GNi(gp_xi[p], GNi_p);
    ConstructQ12D_GNx(GNi_p, GNx_p, coords, &J_p);
    fac = gp_weight[p] * J_p;

    for (i = 0; i < NODES_PER_EL; i++) { /* u nodes */
      for (di = 0; di < NSD; di++) {     /* u dofs */
        for (j = 0; j < 4; j++) {        /* p nodes, p dofs = 1 (ie no loop) */
          PetscInt IJ;
          IJ = ASS_MAP_wIwDI_uJuDJ(i, di, NODES_PER_EL, 2, j, 0, NODES_PER_EL, 1);

          Ke[IJ] = Ke[IJ] - GNx_p[di][i] * Ni_p[j] * fac;
        }
      }
    }
  }
}

static void FormDivergenceOperatorQ1(PetscScalar De[], PetscScalar coords[])
{
  PetscScalar Ge[U_DOFS * NODES_PER_EL * P_DOFS * NODES_PER_EL];
  PetscInt    i, j;
  PetscInt    nr_g, nc_g;

  PetscMemzero(Ge, sizeof(Ge));
  FormGradientOperatorQ1(Ge, coords);

  nr_g = U_DOFS * NODES_PER_EL;
  nc_g = P_DOFS * NODES_PER_EL;

  for (i = 0; i < nr_g; i++) {
    for (j = 0; j < nc_g; j++) De[nr_g * j + i] = Ge[nc_g * i + j];
  }
}

static void FormStabilisationOperatorQ1(PetscScalar Ke[], PetscScalar coords[], PetscScalar eta[])
{
  PetscInt    ngp;
  PetscScalar gp_xi[GAUSS_POINTS][2];
  PetscScalar gp_weight[GAUSS_POINTS];
  PetscInt    p, i, j;
  PetscScalar Ni_p[NODES_PER_EL];
  PetscScalar GNi_p[NSD][NODES_PER_EL], GNx_p[NSD][NODES_PER_EL];
  PetscScalar J_p, fac, eta_avg;

  /* define quadrature rule */
  ConstructGaussQuadrature(&ngp, gp_xi, gp_weight);

  /* evaluate integral */
  for (p = 0; p < ngp; p++) {
    ConstructQ12D_Ni(gp_xi[p], Ni_p);
    ConstructQ12D_GNi(gp_xi[p], GNi_p);
    ConstructQ12D_GNx(GNi_p, GNx_p, coords, &J_p);
    fac = gp_weight[p] * J_p;

    for (i = 0; i < NODES_PER_EL; i++) {
      for (j = 0; j < NODES_PER_EL; j++) Ke[NODES_PER_EL * i + j] = Ke[NODES_PER_EL * i + j] - fac * (Ni_p[i] * Ni_p[j] - 0.0625);
    }
  }

  /* scale */
  eta_avg = 0.0;
  for (p = 0; p < ngp; p++) eta_avg += eta[p];
  eta_avg = (1.0 / ((PetscScalar)ngp)) * eta_avg;
  fac     = 1.0 / eta_avg;
  for (i = 0; i < NODES_PER_EL; i++) {
    for (j = 0; j < NODES_PER_EL; j++) Ke[NODES_PER_EL * i + j] = fac * Ke[NODES_PER_EL * i + j];
  }
}

static void FormScaledMassMatrixOperatorQ1(PetscScalar Ke[], PetscScalar coords[], PetscScalar eta[])
{
  PetscInt    ngp;
  PetscScalar gp_xi[GAUSS_POINTS][2];
  PetscScalar gp_weight[GAUSS_POINTS];
  PetscInt    p, i, j;
  PetscScalar Ni_p[NODES_PER_EL];
  PetscScalar GNi_p[NSD][NODES_PER_EL], GNx_p[NSD][NODES_PER_EL];
  PetscScalar J_p, fac, eta_avg;

  /* define quadrature rule */
  ConstructGaussQuadrature(&ngp, gp_xi, gp_weight);

  /* evaluate integral */
  for (p = 0; p < ngp; p++) {
    ConstructQ12D_Ni(gp_xi[p], Ni_p);
    ConstructQ12D_GNi(gp_xi[p], GNi_p);
    ConstructQ12D_GNx(GNi_p, GNx_p, coords, &J_p);
    fac = gp_weight[p] * J_p;

    for (i = 0; i < NODES_PER_EL; i++) {
      for (j = 0; j < NODES_PER_EL; j++) Ke[NODES_PER_EL * i + j] = Ke[NODES_PER_EL * i + j] - fac * Ni_p[i] * Ni_p[j];
    }
  }

  /* scale */
  eta_avg = 0.0;
  for (p = 0; p < ngp; p++) eta_avg += eta[p];
  eta_avg = (1.0 / ((PetscScalar)ngp)) * eta_avg;
  fac     = 1.0 / eta_avg;
  for (i = 0; i < NODES_PER_EL; i++) {
    for (j = 0; j < NODES_PER_EL; j++) Ke[NODES_PER_EL * i + j] = fac * Ke[NODES_PER_EL * i + j];
  }
}

static void FormMomentumRhsQ1(PetscScalar Fe[], PetscScalar coords[], PetscScalar fx[], PetscScalar fy[])
{
  PetscInt    ngp;
  PetscScalar gp_xi[GAUSS_POINTS][2];
  PetscScalar gp_weight[GAUSS_POINTS];
  PetscInt    p, i;
  PetscScalar Ni_p[NODES_PER_EL];
  PetscScalar GNi_p[NSD][NODES_PER_EL], GNx_p[NSD][NODES_PER_EL];
  PetscScalar J_p, fac;

  /* define quadrature rule */
  ConstructGaussQuadrature(&ngp, gp_xi, gp_weight);

  /* evaluate integral */
  for (p = 0; p < ngp; p++) {
    ConstructQ12D_Ni(gp_xi[p], Ni_p);
    ConstructQ12D_GNi(gp_xi[p], GNi_p);
    ConstructQ12D_GNx(GNi_p, GNx_p, coords, &J_p);
    fac = gp_weight[p] * J_p;

    for (i = 0; i < NODES_PER_EL; i++) {
      Fe[NSD * i] += fac * Ni_p[i] * fx[p];
      Fe[NSD * i + 1] += fac * Ni_p[i] * fy[p];
    }
  }
}

static PetscErrorCode GetElementCoords(DMDACoor2d **_coords, PetscInt ei, PetscInt ej, PetscScalar el_coords[])
{
  PetscFunctionBeginUser;
  /* get coords for the element */
  el_coords[NSD * 0]     = _coords[ej][ei].x;
  el_coords[NSD * 0 + 1] = _coords[ej][ei].y;
  el_coords[NSD * 1]     = _coords[ej + 1][ei].x;
  el_coords[NSD * 1 + 1] = _coords[ej + 1][ei].y;
  el_coords[NSD * 2]     = _coords[ej + 1][ei + 1].x;
  el_coords[NSD * 2 + 1] = _coords[ej + 1][ei + 1].y;
  el_coords[NSD * 3]     = _coords[ej][ei + 1].x;
  el_coords[NSD * 3 + 1] = _coords[ej][ei + 1].y;
  PetscFunctionReturn(0);
}

static PetscErrorCode AssembleA_Stokes(Mat A, DM stokes_da, DM properties_da, Vec properties)
{
  DM                       cda;
  Vec                      coords;
  DMDACoor2d             **_coords;
  MatStencil               u_eqn[NODES_PER_EL * U_DOFS]; /* 2 degrees of freedom */
  MatStencil               p_eqn[NODES_PER_EL * P_DOFS]; /* 1 degrees of freedom */
  PetscInt                 sex, sey, mx, my;
  PetscInt                 ei, ej;
  PetscScalar              Ae[NODES_PER_EL * U_DOFS * NODES_PER_EL * U_DOFS];
  PetscScalar              Ge[NODES_PER_EL * U_DOFS * NODES_PER_EL * P_DOFS];
  PetscScalar              De[NODES_PER_EL * P_DOFS * NODES_PER_EL * U_DOFS];
  PetscScalar              Ce[NODES_PER_EL * P_DOFS * NODES_PER_EL * P_DOFS];
  PetscScalar              el_coords[NODES_PER_EL * NSD];
  Vec                      local_properties;
  GaussPointCoefficients **props;
  PetscScalar             *prop_eta;

  PetscFunctionBeginUser;
  /* setup for coords */
  PetscCall(DMGetCoordinateDM(stokes_da, &cda));
  PetscCall(DMGetCoordinatesLocal(stokes_da, &coords));
  PetscCall(DMDAVecGetArray(cda, coords, &_coords));

  /* setup for coefficients */
  PetscCall(DMCreateLocalVector(properties_da, &local_properties));
  PetscCall(DMGlobalToLocalBegin(properties_da, properties, INSERT_VALUES, local_properties));
  PetscCall(DMGlobalToLocalEnd(properties_da, properties, INSERT_VALUES, local_properties));
  PetscCall(DMDAVecGetArray(properties_da, local_properties, &props));

  PetscCall(DMDAGetElementsCorners(stokes_da, &sex, &sey, NULL));
  PetscCall(DMDAGetElementsSizes(stokes_da, &mx, &my, NULL));
  for (ej = sey; ej < sey + my; ej++) {
    for (ei = sex; ei < sex + mx; ei++) {
      /* get coords for the element */
      GetElementCoords(_coords, ei, ej, el_coords);

      /* get coefficients for the element */
      prop_eta = props[ej][ei].eta;

      /* initialise element stiffness matrix */
      PetscCall(PetscMemzero(Ae, sizeof(Ae)));
      PetscCall(PetscMemzero(Ge, sizeof(Ge)));
      PetscCall(PetscMemzero(De, sizeof(De)));
      PetscCall(PetscMemzero(Ce, sizeof(Ce)));

      /* form element stiffness matrix */
      FormStressOperatorQ1(Ae, el_coords, prop_eta);
      FormGradientOperatorQ1(Ge, el_coords);
      FormDivergenceOperatorQ1(De, el_coords);
      FormStabilisationOperatorQ1(Ce, el_coords, prop_eta);

      /* insert element matrix into global matrix */
      PetscCall(DMDAGetElementEqnums_up(u_eqn, p_eqn, ei, ej));
      PetscCall(MatSetValuesStencil(A, NODES_PER_EL * U_DOFS, u_eqn, NODES_PER_EL * U_DOFS, u_eqn, Ae, ADD_VALUES));
      PetscCall(MatSetValuesStencil(A, NODES_PER_EL * U_DOFS, u_eqn, NODES_PER_EL * P_DOFS, p_eqn, Ge, ADD_VALUES));
      PetscCall(MatSetValuesStencil(A, NODES_PER_EL * P_DOFS, p_eqn, NODES_PER_EL * U_DOFS, u_eqn, De, ADD_VALUES));
      PetscCall(MatSetValuesStencil(A, NODES_PER_EL * P_DOFS, p_eqn, NODES_PER_EL * P_DOFS, p_eqn, Ce, ADD_VALUES));
    }
  }
  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));

  PetscCall(DMDAVecRestoreArray(cda, coords, &_coords));

  PetscCall(DMDAVecRestoreArray(properties_da, local_properties, &props));
  PetscCall(VecDestroy(&local_properties));
  PetscFunctionReturn(0);
}

static PetscErrorCode AssembleA_PCStokes(Mat A, DM stokes_da, DM properties_da, Vec properties)
{
  DM                       cda;
  Vec                      coords;
  DMDACoor2d             **_coords;
  MatStencil               u_eqn[NODES_PER_EL * U_DOFS]; /* 2 degrees of freedom */
  MatStencil               p_eqn[NODES_PER_EL * P_DOFS]; /* 1 degrees of freedom */
  PetscInt                 sex, sey, mx, my;
  PetscInt                 ei, ej;
  PetscScalar              Ae[NODES_PER_EL * U_DOFS * NODES_PER_EL * U_DOFS];
  PetscScalar              Ge[NODES_PER_EL * U_DOFS * NODES_PER_EL * P_DOFS];
  PetscScalar              De[NODES_PER_EL * P_DOFS * NODES_PER_EL * U_DOFS];
  PetscScalar              Ce[NODES_PER_EL * P_DOFS * NODES_PER_EL * P_DOFS];
  PetscScalar              el_coords[NODES_PER_EL * NSD];
  Vec                      local_properties;
  GaussPointCoefficients **props;
  PetscScalar             *prop_eta;

  PetscFunctionBeginUser;
  /* setup for coords */
  PetscCall(DMGetCoordinateDM(stokes_da, &cda));
  PetscCall(DMGetCoordinatesLocal(stokes_da, &coords));
  PetscCall(DMDAVecGetArray(cda, coords, &_coords));

  /* setup for coefficients */
  PetscCall(DMCreateLocalVector(properties_da, &local_properties));
  PetscCall(DMGlobalToLocalBegin(properties_da, properties, INSERT_VALUES, local_properties));
  PetscCall(DMGlobalToLocalEnd(properties_da, properties, INSERT_VALUES, local_properties));
  PetscCall(DMDAVecGetArray(properties_da, local_properties, &props));

  PetscCall(DMDAGetElementsCorners(stokes_da, &sex, &sey, NULL));
  PetscCall(DMDAGetElementsSizes(stokes_da, &mx, &my, NULL));
  for (ej = sey; ej < sey + my; ej++) {
    for (ei = sex; ei < sex + mx; ei++) {
      /* get coords for the element */
      GetElementCoords(_coords, ei, ej, el_coords);

      /* get coefficients for the element */
      prop_eta = props[ej][ei].eta;

      /* initialise element stiffness matrix */
      PetscCall(PetscMemzero(Ae, sizeof(Ae)));
      PetscCall(PetscMemzero(Ge, sizeof(Ge)));
      PetscCall(PetscMemzero(De, sizeof(De)));
      PetscCall(PetscMemzero(Ce, sizeof(Ce)));

      /* form element stiffness matrix */
      FormStressOperatorQ1(Ae, el_coords, prop_eta);
      FormGradientOperatorQ1(Ge, el_coords);
      FormScaledMassMatrixOperatorQ1(Ce, el_coords, prop_eta);

      /* insert element matrix into global matrix */
      PetscCall(DMDAGetElementEqnums_up(u_eqn, p_eqn, ei, ej));
      PetscCall(MatSetValuesStencil(A, NODES_PER_EL * U_DOFS, u_eqn, NODES_PER_EL * U_DOFS, u_eqn, Ae, ADD_VALUES));
      PetscCall(MatSetValuesStencil(A, NODES_PER_EL * U_DOFS, u_eqn, NODES_PER_EL * P_DOFS, p_eqn, Ge, ADD_VALUES));
      PetscCall(MatSetValuesStencil(A, NODES_PER_EL * P_DOFS, p_eqn, NODES_PER_EL * P_DOFS, p_eqn, Ce, ADD_VALUES));
    }
  }
  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));

  PetscCall(DMDAVecRestoreArray(cda, coords, &_coords));

  PetscCall(DMDAVecRestoreArray(properties_da, local_properties, &props));
  PetscCall(VecDestroy(&local_properties));
  PetscFunctionReturn(0);
}

static PetscErrorCode DMDASetValuesLocalStencil_ADD_VALUES(StokesDOF **fields_F, MatStencil u_eqn[], MatStencil p_eqn[], PetscScalar Fe_u[], PetscScalar Fe_p[])
{
  PetscInt n;

  PetscFunctionBeginUser;
  for (n = 0; n < 4; n++) {
    fields_F[u_eqn[2 * n].j][u_eqn[2 * n].i].u_dof         = fields_F[u_eqn[2 * n].j][u_eqn[2 * n].i].u_dof + Fe_u[2 * n];
    fields_F[u_eqn[2 * n + 1].j][u_eqn[2 * n + 1].i].v_dof = fields_F[u_eqn[2 * n + 1].j][u_eqn[2 * n + 1].i].v_dof + Fe_u[2 * n + 1];
    fields_F[p_eqn[n].j][p_eqn[n].i].p_dof                 = fields_F[p_eqn[n].j][p_eqn[n].i].p_dof + Fe_p[n];
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode AssembleF_Stokes(Vec F, DM stokes_da, DM properties_da, Vec properties)
{
  DM                       cda;
  Vec                      coords;
  DMDACoor2d             **_coords;
  MatStencil               u_eqn[NODES_PER_EL * U_DOFS]; /* 2 degrees of freedom */
  MatStencil               p_eqn[NODES_PER_EL * P_DOFS]; /* 1 degrees of freedom */
  PetscInt                 sex, sey, mx, my;
  PetscInt                 ei, ej;
  PetscScalar              Fe[NODES_PER_EL * U_DOFS];
  PetscScalar              He[NODES_PER_EL * P_DOFS];
  PetscScalar              el_coords[NODES_PER_EL * NSD];
  Vec                      local_properties;
  GaussPointCoefficients **props;
  PetscScalar             *prop_fx, *prop_fy;
  Vec                      local_F;
  StokesDOF              **ff;

  PetscFunctionBeginUser;
  /* setup for coords */
  PetscCall(DMGetCoordinateDM(stokes_da, &cda));
  PetscCall(DMGetCoordinatesLocal(stokes_da, &coords));
  PetscCall(DMDAVecGetArray(cda, coords, &_coords));

  /* setup for coefficients */
  PetscCall(DMGetLocalVector(properties_da, &local_properties));
  PetscCall(DMGlobalToLocalBegin(properties_da, properties, INSERT_VALUES, local_properties));
  PetscCall(DMGlobalToLocalEnd(properties_da, properties, INSERT_VALUES, local_properties));
  PetscCall(DMDAVecGetArray(properties_da, local_properties, &props));

  /* get access to the vector */
  PetscCall(DMGetLocalVector(stokes_da, &local_F));
  PetscCall(VecZeroEntries(local_F));
  PetscCall(DMDAVecGetArray(stokes_da, local_F, &ff));

  PetscCall(DMDAGetElementsCorners(stokes_da, &sex, &sey, NULL));
  PetscCall(DMDAGetElementsSizes(stokes_da, &mx, &my, NULL));
  for (ej = sey; ej < sey + my; ej++) {
    for (ei = sex; ei < sex + mx; ei++) {
      /* get coords for the element */
      GetElementCoords(_coords, ei, ej, el_coords);

      /* get coefficients for the element */
      prop_fx = props[ej][ei].fx;
      prop_fy = props[ej][ei].fy;

      /* initialise element stiffness matrix */
      PetscCall(PetscMemzero(Fe, sizeof(Fe)));
      PetscCall(PetscMemzero(He, sizeof(He)));

      /* form element stiffness matrix */
      FormMomentumRhsQ1(Fe, el_coords, prop_fx, prop_fy);

      /* insert element matrix into global matrix */
      PetscCall(DMDAGetElementEqnums_up(u_eqn, p_eqn, ei, ej));

      PetscCall(DMDASetValuesLocalStencil_ADD_VALUES(ff, u_eqn, p_eqn, Fe, He));
    }
  }

  PetscCall(DMDAVecRestoreArray(stokes_da, local_F, &ff));
  PetscCall(DMLocalToGlobalBegin(stokes_da, local_F, ADD_VALUES, F));
  PetscCall(DMLocalToGlobalEnd(stokes_da, local_F, ADD_VALUES, F));
  PetscCall(DMRestoreLocalVector(stokes_da, &local_F));

  PetscCall(DMDAVecRestoreArray(cda, coords, &_coords));

  PetscCall(DMDAVecRestoreArray(properties_da, local_properties, &props));
  PetscCall(DMRestoreLocalVector(properties_da, &local_properties));
  PetscFunctionReturn(0);
}

static PetscErrorCode DMDACreateSolCx(PetscReal eta0, PetscReal eta1, PetscReal xc, PetscInt nz, PetscInt mx, PetscInt my, DM *_da, Vec *_X)
{
  DM           da, cda;
  Vec          X;
  StokesDOF  **_stokes;
  Vec          coords;
  DMDACoor2d **_coords;
  PetscInt     si, sj, ei, ej, i, j;

  PetscFunctionBeginUser;
  PetscCall(DMDACreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DMDA_STENCIL_BOX, mx + 1, my + 1, PETSC_DECIDE, PETSC_DECIDE, 3, 1, NULL, NULL, &da));
  PetscCall(DMSetFromOptions(da));
  PetscCall(DMSetUp(da));
  PetscCall(DMDASetFieldName(da, 0, "anlytic_Vx"));
  PetscCall(DMDASetFieldName(da, 1, "anlytic_Vy"));
  PetscCall(DMDASetFieldName(da, 2, "analytic_P"));

  PetscCall(DMDASetUniformCoordinates(da, 0.0, 1.0, 0.0, 1.0, 0., 0.));

  PetscCall(DMGetCoordinatesLocal(da, &coords));
  PetscCall(DMGetCoordinateDM(da, &cda));
  PetscCall(DMDAVecGetArray(cda, coords, &_coords));

  PetscCall(DMCreateGlobalVector(da, &X));
  PetscCall(DMDAVecGetArray(da, X, &_stokes));

  PetscCall(DMDAGetCorners(da, &si, &sj, 0, &ei, &ej, 0));
  for (j = sj; j < sj + ej; j++) {
    for (i = si; i < si + ei; i++) {
      PetscReal pos[2], pressure, vel[2], total_stress[3], strain_rate[3];

      pos[0] = PetscRealPart(_coords[j][i].x);
      pos[1] = PetscRealPart(_coords[j][i].y);

      evaluate_solCx(pos, eta0, eta1, xc, nz, vel, &pressure, total_stress, strain_rate);

      _stokes[j][i].u_dof = vel[0];
      _stokes[j][i].v_dof = vel[1];
      _stokes[j][i].p_dof = pressure;
    }
  }
  PetscCall(DMDAVecRestoreArray(da, X, &_stokes));
  PetscCall(DMDAVecRestoreArray(cda, coords, &_coords));

  *_da = da;
  *_X  = X;
  PetscFunctionReturn(0);
}

static PetscErrorCode StokesDAGetNodalFields(StokesDOF **fields, PetscInt ei, PetscInt ej, StokesDOF nodal_fields[])
{
  PetscFunctionBeginUser;
  /* get the nodal fields */
  nodal_fields[0].u_dof = fields[ej][ei].u_dof;
  nodal_fields[0].v_dof = fields[ej][ei].v_dof;
  nodal_fields[0].p_dof = fields[ej][ei].p_dof;
  nodal_fields[1].u_dof = fields[ej + 1][ei].u_dof;
  nodal_fields[1].v_dof = fields[ej + 1][ei].v_dof;
  nodal_fields[1].p_dof = fields[ej + 1][ei].p_dof;
  nodal_fields[2].u_dof = fields[ej + 1][ei + 1].u_dof;
  nodal_fields[2].v_dof = fields[ej + 1][ei + 1].v_dof;
  nodal_fields[2].p_dof = fields[ej + 1][ei + 1].p_dof;
  nodal_fields[3].u_dof = fields[ej][ei + 1].u_dof;
  nodal_fields[3].v_dof = fields[ej][ei + 1].v_dof;
  nodal_fields[3].p_dof = fields[ej][ei + 1].p_dof;
  PetscFunctionReturn(0);
}

static PetscErrorCode DMDAIntegrateErrors(DM stokes_da, Vec X, Vec X_analytic)
{
  DM           cda;
  Vec          coords, X_analytic_local, X_local;
  DMDACoor2d **_coords;
  PetscInt     sex, sey, mx, my;
  PetscInt     ei, ej;
  PetscScalar  el_coords[NODES_PER_EL * NSD];
  StokesDOF  **stokes_analytic, **stokes;
  StokesDOF    stokes_analytic_e[4], stokes_e[4];

  PetscScalar GNi_p[NSD][NODES_PER_EL], GNx_p[NSD][NODES_PER_EL];
  PetscScalar Ni_p[NODES_PER_EL];
  PetscInt    ngp;
  PetscScalar gp_xi[GAUSS_POINTS][2];
  PetscScalar gp_weight[GAUSS_POINTS];
  PetscInt    p, i;
  PetscScalar J_p, fac;
  PetscScalar h, p_e_L2, u_e_L2, u_e_H1, p_L2, u_L2, u_H1, tp_L2, tu_L2, tu_H1;
  PetscInt    M;
  PetscReal   xymin[2], xymax[2];

  PetscFunctionBeginUser;
  /* define quadrature rule */
  ConstructGaussQuadrature(&ngp, gp_xi, gp_weight);

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

  h = (xymax[0] - xymin[0]) / ((PetscReal)M);

  tp_L2 = tu_L2 = tu_H1 = 0.0;

  PetscCall(DMDAGetElementsCorners(stokes_da, &sex, &sey, NULL));
  PetscCall(DMDAGetElementsSizes(stokes_da, &mx, &my, NULL));
  for (ej = sey; ej < sey + my; ej++) {
    for (ei = sex; ei < sex + mx; ei++) {
      /* get coords for the element */
      PetscCall(GetElementCoords(_coords, ei, ej, el_coords));
      PetscCall(StokesDAGetNodalFields(stokes, ei, ej, stokes_e));
      PetscCall(StokesDAGetNodalFields(stokes_analytic, ei, ej, stokes_analytic_e));

      /* evaluate integral */
      p_e_L2 = 0.0;
      u_e_L2 = 0.0;
      u_e_H1 = 0.0;
      for (p = 0; p < ngp; p++) {
        ConstructQ12D_Ni(gp_xi[p], Ni_p);
        ConstructQ12D_GNi(gp_xi[p], GNi_p);
        ConstructQ12D_GNx(GNi_p, GNx_p, el_coords, &J_p);
        fac = gp_weight[p] * J_p;

        for (i = 0; i < NODES_PER_EL; i++) {
          PetscScalar u_error, v_error;

          p_e_L2 = p_e_L2 + fac * Ni_p[i] * (stokes_e[i].p_dof - stokes_analytic_e[i].p_dof) * (stokes_e[i].p_dof - stokes_analytic_e[i].p_dof);

          u_error = stokes_e[i].u_dof - stokes_analytic_e[i].u_dof;
          v_error = stokes_e[i].v_dof - stokes_analytic_e[i].v_dof;
          u_e_L2 += fac * Ni_p[i] * (u_error * u_error + v_error * v_error);

          u_e_H1 = u_e_H1 + fac * (GNx_p[0][i] * u_error * GNx_p[0][i] * u_error     /* du/dx */
                                   + GNx_p[1][i] * u_error * GNx_p[1][i] * u_error   /* du/dy */
                                   + GNx_p[0][i] * v_error * GNx_p[0][i] * v_error   /* dv/dx */
                                   + GNx_p[1][i] * v_error * GNx_p[1][i] * v_error); /* dv/dy */
        }
      }

      tp_L2 += p_e_L2;
      tu_L2 += u_e_L2;
      tu_H1 += u_e_H1;
    }
  }
  PetscCallMPI(MPI_Allreduce(&tp_L2, &p_L2, 1, MPIU_SCALAR, MPIU_SUM, PETSC_COMM_WORLD));
  PetscCallMPI(MPI_Allreduce(&tu_L2, &u_L2, 1, MPIU_SCALAR, MPIU_SUM, PETSC_COMM_WORLD));
  PetscCallMPI(MPI_Allreduce(&tu_H1, &u_H1, 1, MPIU_SCALAR, MPIU_SUM, PETSC_COMM_WORLD));
  p_L2 = PetscSqrtScalar(p_L2);
  u_L2 = PetscSqrtScalar(u_L2);
  u_H1 = PetscSqrtScalar(u_H1);

  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "%1.4e   %1.4e   %1.4e   %1.4e\n", (double)PetscRealPart(h), (double)PetscRealPart(p_L2), (double)PetscRealPart(u_L2), (double)PetscRealPart(u_H1)));

  PetscCall(DMDAVecRestoreArray(cda, coords, &_coords));

  PetscCall(DMDAVecRestoreArray(stokes_da, X_analytic_local, &stokes_analytic));
  PetscCall(VecDestroy(&X_analytic_local));
  PetscCall(DMDAVecRestoreArray(stokes_da, X_local, &stokes));
  PetscCall(VecDestroy(&X_local));
  PetscFunctionReturn(0);
}

static PetscErrorCode solve_stokes_2d_coupled(PetscInt mx, PetscInt my)
{
  DM                       da_Stokes, da_prop;
  PetscInt                 u_dof, p_dof, dof, stencil_width;
  Mat                      A, B;
  DM                       prop_cda, vel_cda;
  Vec                      prop_coords, vel_coords;
  PetscInt                 si, sj, nx, ny, i, j, p;
  Vec                      f, X;
  PetscInt                 prop_dof, prop_stencil_width;
  Vec                      properties, l_properties;
  PetscReal                dx, dy;
  PetscInt                 M, N;
  DMDACoor2d             **_prop_coords, **_vel_coords;
  GaussPointCoefficients **element_props;
  PetscInt                 its;
  KSP                      ksp_S;
  PetscInt                 coefficient_structure = 0;
  PetscInt                 cpu_x, cpu_y, *lx = NULL, *ly = NULL;
  PetscBool                use_gp_coords = PETSC_FALSE, set, output_gnuplot = PETSC_FALSE, glvis = PETSC_FALSE, change = PETSC_FALSE;
  char                     filename[PETSC_MAX_PATH_LEN];

  PetscFunctionBeginUser;

  PetscCall(PetscOptionsGetBool(NULL, NULL, "-gnuplot", &output_gnuplot, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-glvis", &glvis, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-change", &change, NULL));

  /* Generate the da for velocity and pressure */
  /*
  We use Q1 elements for the temperature.
  FEM has a 9-point stencil (BOX) or connectivity pattern
  Num nodes in each direction is mx+1, my+1
  */
  u_dof         = U_DOFS; /* Vx, Vy - velocities */
  p_dof         = P_DOFS; /* p - pressure */
  dof           = u_dof + p_dof;
  stencil_width = 1;
  PetscCall(DMDACreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DMDA_STENCIL_BOX, mx + 1, my + 1, PETSC_DECIDE, PETSC_DECIDE, dof, stencil_width, NULL, NULL, &da_Stokes));

  PetscCall(DMSetMatType(da_Stokes, MATAIJ));
  PetscCall(DMSetFromOptions(da_Stokes));
  PetscCall(DMSetUp(da_Stokes));
  PetscCall(DMDASetFieldName(da_Stokes, 0, "Vx"));
  PetscCall(DMDASetFieldName(da_Stokes, 1, "Vy"));
  PetscCall(DMDASetFieldName(da_Stokes, 2, "P"));

  /* unit box [0,1] x [0,1] */
  PetscCall(DMDASetUniformCoordinates(da_Stokes, 0.0, 1.0, 0.0, 1.0, 0., 0.));

  /* Generate element properties, we will assume all material properties are constant over the element */
  /* !!! IN PARALLEL WE MUST MAKE SURE THE TWO DMDA's ALIGN !!!  */
  PetscCall(DMDAGetInfo(da_Stokes, 0, 0, 0, 0, &cpu_x, &cpu_y, 0, 0, 0, 0, 0, 0, 0));
  PetscCall(DMDAGetElementOwnershipRanges2d(da_Stokes, &lx, &ly));

  prop_dof           = (int)(sizeof(GaussPointCoefficients) / sizeof(PetscScalar)); /* gauss point setup */
  prop_stencil_width = 0;
  PetscCall(DMDACreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DMDA_STENCIL_BOX, mx, my, cpu_x, cpu_y, prop_dof, prop_stencil_width, lx, ly, &da_prop));
  PetscCall(DMSetFromOptions(da_prop));
  PetscCall(DMSetUp(da_prop));
  PetscCall(PetscFree(lx));
  PetscCall(PetscFree(ly));

  /* define centroid positions */
  PetscCall(DMDAGetInfo(da_prop, 0, &M, &N, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0));
  dx = 1.0 / ((PetscReal)(M));
  dy = 1.0 / ((PetscReal)(N));

  PetscCall(DMDASetUniformCoordinates(da_prop, 0.0 + 0.5 * dx, 1.0 - 0.5 * dx, 0.0 + 0.5 * dy, 1.0 - 0.5 * dy, 0., 0));

  /* define coefficients */
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-c_str", &coefficient_structure, NULL));

  PetscCall(DMCreateGlobalVector(da_prop, &properties));
  PetscCall(DMCreateLocalVector(da_prop, &l_properties));
  PetscCall(DMDAVecGetArray(da_prop, l_properties, &element_props));

  PetscCall(DMGetCoordinateDM(da_prop, &prop_cda));
  PetscCall(DMGetCoordinatesLocal(da_prop, &prop_coords));
  PetscCall(DMDAVecGetArray(prop_cda, prop_coords, &_prop_coords));

  PetscCall(DMDAGetGhostCorners(prop_cda, &si, &sj, 0, &nx, &ny, 0));

  PetscCall(DMGetCoordinateDM(da_Stokes, &vel_cda));
  PetscCall(DMGetCoordinatesLocal(da_Stokes, &vel_coords));
  PetscCall(DMDAVecGetArray(vel_cda, vel_coords, &_vel_coords));

  /* interpolate the coordinates */
  for (j = sj; j < sj + ny; j++) {
    for (i = si; i < si + nx; i++) {
      PetscInt    ngp;
      PetscScalar gp_xi[GAUSS_POINTS][2], gp_weight[GAUSS_POINTS];
      PetscScalar el_coords[8];

      PetscCall(GetElementCoords(_vel_coords, i, j, el_coords));
      ConstructGaussQuadrature(&ngp, gp_xi, gp_weight);

      for (p = 0; p < GAUSS_POINTS; p++) {
        PetscScalar gp_x, gp_y;
        PetscInt    n;
        PetscScalar xi_p[2], Ni_p[4];

        xi_p[0] = gp_xi[p][0];
        xi_p[1] = gp_xi[p][1];
        ConstructQ12D_Ni(xi_p, Ni_p);

        gp_x = 0.0;
        gp_y = 0.0;
        for (n = 0; n < NODES_PER_EL; n++) {
          gp_x = gp_x + Ni_p[n] * el_coords[2 * n];
          gp_y = gp_y + Ni_p[n] * el_coords[2 * n + 1];
        }
        element_props[j][i].gp_coords[2 * p]     = gp_x;
        element_props[j][i].gp_coords[2 * p + 1] = gp_y;
      }
    }
  }

  /* define the coefficients */
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-use_gp_coords", &use_gp_coords, NULL));

  for (j = sj; j < sj + ny; j++) {
    for (i = si; i < si + nx; i++) {
      PetscReal centroid_x = PetscRealPart(_prop_coords[j][i].x); /* centroids of cell */
      PetscReal centroid_y = PetscRealPart(_prop_coords[j][i].y);
      PetscReal coord_x, coord_y;

      if (coefficient_structure == 0) {
        PetscReal opts_eta0, opts_eta1, opts_xc;
        PetscInt  opts_nz;

        opts_eta0 = 1.0;
        opts_eta1 = 1.0;
        opts_xc   = 0.5;
        opts_nz   = 1;

        PetscCall(PetscOptionsGetReal(NULL, NULL, "-solcx_eta0", &opts_eta0, NULL));
        PetscCall(PetscOptionsGetReal(NULL, NULL, "-solcx_eta1", &opts_eta1, NULL));
        PetscCall(PetscOptionsGetReal(NULL, NULL, "-solcx_xc", &opts_xc, NULL));
        PetscCall(PetscOptionsGetInt(NULL, NULL, "-solcx_nz", &opts_nz, NULL));

        for (p = 0; p < GAUSS_POINTS; p++) {
          coord_x = centroid_x;
          coord_y = centroid_y;
          if (use_gp_coords) {
            coord_x = PetscRealPart(element_props[j][i].gp_coords[2 * p]);
            coord_y = PetscRealPart(element_props[j][i].gp_coords[2 * p + 1]);
          }

          element_props[j][i].eta[p] = opts_eta0;
          if (coord_x > opts_xc) element_props[j][i].eta[p] = opts_eta1;

          element_props[j][i].fx[p] = 0.0;
          element_props[j][i].fy[p] = PetscSinReal(opts_nz * PETSC_PI * coord_y) * PetscCosReal(1.0 * PETSC_PI * coord_x);
        }
      } else if (coefficient_structure == 1) { /* square sinker */
        PetscReal opts_eta0, opts_eta1, opts_dx, opts_dy;

        opts_eta0 = 1.0;
        opts_eta1 = 1.0;
        opts_dx   = 0.50;
        opts_dy   = 0.50;

        PetscCall(PetscOptionsGetReal(NULL, NULL, "-sinker_eta0", &opts_eta0, NULL));
        PetscCall(PetscOptionsGetReal(NULL, NULL, "-sinker_eta1", &opts_eta1, NULL));
        PetscCall(PetscOptionsGetReal(NULL, NULL, "-sinker_dx", &opts_dx, NULL));
        PetscCall(PetscOptionsGetReal(NULL, NULL, "-sinker_dy", &opts_dy, NULL));

        for (p = 0; p < GAUSS_POINTS; p++) {
          coord_x = centroid_x;
          coord_y = centroid_y;
          if (use_gp_coords) {
            coord_x = PetscRealPart(element_props[j][i].gp_coords[2 * p]);
            coord_y = PetscRealPart(element_props[j][i].gp_coords[2 * p + 1]);
          }

          element_props[j][i].eta[p] = opts_eta0;
          element_props[j][i].fx[p]  = 0.0;
          element_props[j][i].fy[p]  = 0.0;

          if ((coord_x > -0.5 * opts_dx + 0.5) && (coord_x < 0.5 * opts_dx + 0.5)) {
            if ((coord_y > -0.5 * opts_dy + 0.5) && (coord_y < 0.5 * opts_dy + 0.5)) {
              element_props[j][i].eta[p] = opts_eta1;
              element_props[j][i].fx[p]  = 0.0;
              element_props[j][i].fy[p]  = -1.0;
            }
          }
        }
      } else if (coefficient_structure == 2) { /* circular sinker */
        PetscReal opts_eta0, opts_eta1, opts_r, radius2;

        opts_eta0 = 1.0;
        opts_eta1 = 1.0;
        opts_r    = 0.25;

        PetscCall(PetscOptionsGetReal(NULL, NULL, "-sinker_eta0", &opts_eta0, NULL));
        PetscCall(PetscOptionsGetReal(NULL, NULL, "-sinker_eta1", &opts_eta1, NULL));
        PetscCall(PetscOptionsGetReal(NULL, NULL, "-sinker_r", &opts_r, NULL));

        for (p = 0; p < GAUSS_POINTS; p++) {
          coord_x = centroid_x;
          coord_y = centroid_y;
          if (use_gp_coords) {
            coord_x = PetscRealPart(element_props[j][i].gp_coords[2 * p]);
            coord_y = PetscRealPart(element_props[j][i].gp_coords[2 * p + 1]);
          }

          element_props[j][i].eta[p] = opts_eta0;
          element_props[j][i].fx[p]  = 0.0;
          element_props[j][i].fy[p]  = 0.0;

          radius2 = (coord_x - 0.5) * (coord_x - 0.5) + (coord_y - 0.5) * (coord_y - 0.5);
          if (radius2 < opts_r * opts_r) {
            element_props[j][i].eta[p] = opts_eta1;
            element_props[j][i].fx[p]  = 0.0;
            element_props[j][i].fy[p]  = -1.0;
          }
        }
      } else if (coefficient_structure == 3) { /* circular and rectangular inclusion */
        PetscReal opts_eta0, opts_eta1, opts_r, opts_dx, opts_dy, opts_c0x, opts_c0y, opts_s0x, opts_s0y, opts_phi, radius2;

        opts_eta0 = 1.0;
        opts_eta1 = 1.0;
        opts_r    = 0.25;
        opts_c0x  = 0.35; /* circle center */
        opts_c0y  = 0.35;
        opts_s0x  = 0.7; /* square center */
        opts_s0y  = 0.7;
        opts_dx   = 0.25;
        opts_dy   = 0.25;
        opts_phi  = 25;

        PetscCall(PetscOptionsGetReal(NULL, NULL, "-sinker_eta0", &opts_eta0, NULL));
        PetscCall(PetscOptionsGetReal(NULL, NULL, "-sinker_eta1", &opts_eta1, NULL));
        PetscCall(PetscOptionsGetReal(NULL, NULL, "-sinker_r", &opts_r, NULL));
        PetscCall(PetscOptionsGetReal(NULL, NULL, "-sinker_c0x", &opts_c0x, NULL));
        PetscCall(PetscOptionsGetReal(NULL, NULL, "-sinker_c0y", &opts_c0y, NULL));
        PetscCall(PetscOptionsGetReal(NULL, NULL, "-sinker_s0x", &opts_s0x, NULL));
        PetscCall(PetscOptionsGetReal(NULL, NULL, "-sinker_s0y", &opts_s0y, NULL));
        PetscCall(PetscOptionsGetReal(NULL, NULL, "-sinker_dx", &opts_dx, NULL));
        PetscCall(PetscOptionsGetReal(NULL, NULL, "-sinker_dy", &opts_dy, NULL));
        PetscCall(PetscOptionsGetReal(NULL, NULL, "-sinker_phi", &opts_phi, NULL));
        opts_phi *= PETSC_PI / 180;

        for (p = 0; p < GAUSS_POINTS; p++) {
          coord_x = centroid_x;
          coord_y = centroid_y;
          if (use_gp_coords) {
            coord_x = PetscRealPart(element_props[j][i].gp_coords[2 * p]);
            coord_y = PetscRealPart(element_props[j][i].gp_coords[2 * p + 1]);
          }

          element_props[j][i].eta[p] = opts_eta0;
          element_props[j][i].fx[p]  = 0.0;
          element_props[j][i].fy[p]  = -0.2;

          radius2 = PetscSqr(coord_x - opts_c0x) + PetscSqr(coord_y - opts_c0y);
          if (radius2 < opts_r * opts_r || (PetscAbs(+(coord_x - opts_s0x) * PetscCosReal(opts_phi) + (coord_y - opts_s0y) * PetscSinReal(opts_phi)) < opts_dx / 2 && PetscAbs(-(coord_x - opts_s0x) * PetscSinReal(opts_phi) + (coord_y - opts_s0y) * PetscCosReal(opts_phi)) < opts_dy / 2)) {
            element_props[j][i].eta[p] = opts_eta1;
            element_props[j][i].fx[p]  = 0.0;
            element_props[j][i].fy[p]  = -1.0;
          }
        }
      } else if (coefficient_structure == 4) { /* subdomain jump */
        PetscReal opts_mag, opts_eta0;
        PetscInt  opts_nz, px, py;
        PetscBool jump;

        opts_mag  = 1.0;
        opts_eta0 = 1.0;
        opts_nz   = 1;

        PetscCall(PetscOptionsGetReal(NULL, NULL, "-jump_eta0", &opts_eta0, NULL));
        PetscCall(PetscOptionsGetReal(NULL, NULL, "-jump_magnitude", &opts_mag, NULL));
        PetscCall(PetscOptionsGetInt(NULL, NULL, "-jump_nz", &opts_nz, NULL));
        PetscCall(DMDAGetInfo(da_Stokes, NULL, NULL, NULL, NULL, &px, &py, NULL, NULL, NULL, NULL, NULL, NULL, NULL));
        if (px % 2) {
          jump = (PetscBool)(PetscGlobalRank % 2);
        } else {
          jump = (PetscBool)((PetscGlobalRank / px) % 2 ? PetscGlobalRank % 2 : !(PetscGlobalRank % 2));
        }
        for (p = 0; p < GAUSS_POINTS; p++) {
          coord_x = centroid_x;
          coord_y = centroid_y;
          if (use_gp_coords) {
            coord_x = PetscRealPart(element_props[j][i].gp_coords[2 * p]);
            coord_y = PetscRealPart(element_props[j][i].gp_coords[2 * p + 1]);
          }

          element_props[j][i].eta[p] = jump ? PetscPowReal(10.0, opts_mag) : opts_eta0;
          element_props[j][i].fx[p]  = 0.0;
          element_props[j][i].fy[p]  = PetscSinReal(opts_nz * PETSC_PI * coord_y) * PetscCosReal(1.0 * PETSC_PI * coord_x);
        }
      } else SETERRQ(PETSC_COMM_SELF, PETSC_ERR_USER, "Unknown coefficient_structure");
    }
  }
  PetscCall(DMDAVecRestoreArray(prop_cda, prop_coords, &_prop_coords));

  PetscCall(DMDAVecRestoreArray(vel_cda, vel_coords, &_vel_coords));

  PetscCall(DMDAVecRestoreArray(da_prop, l_properties, &element_props));
  PetscCall(DMLocalToGlobalBegin(da_prop, l_properties, ADD_VALUES, properties));
  PetscCall(DMLocalToGlobalEnd(da_prop, l_properties, ADD_VALUES, properties));

  if (output_gnuplot) {
    PetscCall(DMDACoordViewGnuplot2d(da_Stokes, "mesh"));
    PetscCall(DMDAViewCoefficientsGnuplot2d(da_prop, properties, "Coefficients for Stokes eqn.", "properties"));
  }

  if (glvis) {
    Vec         glv_prop, etaf;
    PetscViewer view;
    PetscInt    dim;
    const char *fec = {"FiniteElementCollection: L2_2D_P1"};

    PetscCall(DMGetDimension(da_Stokes, &dim));
    PetscCall(VecCreateSeq(PETSC_COMM_SELF, GAUSS_POINTS * mx * mx, &etaf));
    PetscCall(PetscObjectSetName((PetscObject)etaf, "viscosity"));
    PetscCall(PetscViewerGLVisOpen(PETSC_COMM_WORLD, PETSC_VIEWER_GLVIS_SOCKET, NULL, PETSC_DECIDE, &view));
    PetscCall(PetscViewerGLVisSetFields(view, 1, &fec, &dim, glvis_extract_eta, (PetscObject *)&etaf, da_prop, NULL));
    PetscCall(DMGetLocalVector(da_prop, &glv_prop));
    PetscCall(DMGlobalToLocalBegin(da_prop, properties, INSERT_VALUES, glv_prop));
    PetscCall(DMGlobalToLocalEnd(da_prop, properties, INSERT_VALUES, glv_prop));
    PetscCall(VecSetDM(etaf, da_Stokes));
    PetscCall(VecView(glv_prop, view));
    PetscCall(PetscViewerDestroy(&view));
    PetscCall(DMRestoreLocalVector(da_prop, &glv_prop));
    PetscCall(VecDestroy(&etaf));
  }

  /* Generate a matrix with the correct non-zero pattern of type AIJ. This will work in parallel and serial */
  PetscCall(DMCreateMatrix(da_Stokes, &A));
  PetscCall(DMCreateMatrix(da_Stokes, &B));
  PetscCall(DMCreateGlobalVector(da_Stokes, &f));
  PetscCall(DMCreateGlobalVector(da_Stokes, &X));

  /* assemble A11 */
  PetscCall(MatZeroEntries(A));
  PetscCall(MatZeroEntries(B));
  PetscCall(VecZeroEntries(f));

  PetscCall(AssembleA_Stokes(A, da_Stokes, da_prop, properties));
  PetscCall(AssembleA_PCStokes(B, da_Stokes, da_prop, properties));
  /* build force vector */
  PetscCall(AssembleF_Stokes(f, da_Stokes, da_prop, properties));

  PetscCall(DMDABCApplyFreeSlip(da_Stokes, A, f));
  PetscCall(DMDABCApplyFreeSlip(da_Stokes, B, NULL));

  /* SOLVE */
  PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp_S));
  PetscCall(KSPSetOptionsPrefix(ksp_S, "stokes_"));
  PetscCall(KSPSetDM(ksp_S, da_Stokes));
  PetscCall(KSPSetDMActive(ksp_S, PETSC_FALSE));
  PetscCall(KSPSetOperators(ksp_S, A, B));
  PetscCall(KSPSetFromOptions(ksp_S));
  {
    PC             pc;
    const PetscInt ufields[] = {0, 1}, pfields[1] = {2};

    PetscCall(KSPGetPC(ksp_S, &pc));
    PetscCall(PCFieldSplitSetBlockSize(pc, 3));
    PetscCall(PCFieldSplitSetFields(pc, "u", 2, ufields, ufields));
    PetscCall(PCFieldSplitSetFields(pc, "p", 1, pfields, pfields));
  }

  {
    PC        pc;
    PetscBool same = PETSC_FALSE;
    PetscCall(KSPGetPC(ksp_S, &pc));
    PetscCall(PetscObjectTypeCompare((PetscObject)pc, PCBDDC, &same));
    if (same) {
      PetscBool usedivmat = PETSC_FALSE;
      PetscCall(KSPSetOperators(ksp_S, A, A));

      PetscCall(PetscOptionsGetBool(NULL, NULL, "-stokes_pc_bddc_use_divergence", &usedivmat, NULL));
      if (usedivmat) {
        IS      *fields, vel;
        PetscInt i, nf;

        PetscCall(DMCreateFieldDecomposition(da_Stokes, &nf, NULL, &fields, NULL));
        PetscCall(ISConcatenate(PETSC_COMM_WORLD, 2, fields, &vel));
        PetscCall(MatZeroRowsIS(B, fields[2], 1.0, NULL, NULL)); /* we put 1.0 on the diagonal to pick the pressure average too */
        PetscCall(MatTranspose(B, MAT_INPLACE_MATRIX, &B));
        PetscCall(MatZeroRowsIS(B, vel, 0.0, NULL, NULL));
        PetscCall(ISDestroy(&vel));
        PetscCall(PCBDDCSetDivergenceMat(pc, B, PETSC_FALSE, NULL));
        for (i = 0; i < nf; i++) PetscCall(ISDestroy(&fields[i]));
        PetscCall(PetscFree(fields));
      }
    }
  }

  {
    PC        pc_S;
    KSP      *sub_ksp, ksp_U;
    PetscInt  nsplits;
    DM        da_U;
    PetscBool is_pcfs;

    PetscCall(KSPSetUp(ksp_S));
    PetscCall(KSPGetPC(ksp_S, &pc_S));

    is_pcfs = PETSC_FALSE;
    PetscCall(PetscObjectTypeCompare((PetscObject)pc_S, PCFIELDSPLIT, &is_pcfs));

    if (is_pcfs) {
      PetscCall(PCFieldSplitGetSubKSP(pc_S, &nsplits, &sub_ksp));
      ksp_U = sub_ksp[0];
      PetscCall(PetscFree(sub_ksp));

      if (nsplits == 2) {
        PetscCall(DMDACreateCompatibleDMDA(da_Stokes, 2, &da_U));

        PetscCall(KSPSetDM(ksp_U, da_U));
        PetscCall(KSPSetDMActive(ksp_U, PETSC_FALSE));
        PetscCall(DMDestroy(&da_U));
        if (change) {
          const char opt[] = "-stokes_fieldsplit_u_pc_factor_mat_solver_type mumps";
          PetscCall(PetscOptionsInsertString(NULL, opt));
          PetscCall(KSPSetFromOptions(ksp_U));
        }
        PetscCall(KSPSetUp(ksp_U));
      }
    }
  }

  PetscCall(KSPSolve(ksp_S, f, X));

  PetscCall(PetscOptionsGetString(NULL, NULL, "-o", filename, sizeof(filename), &set));
  if (set) {
    char       *ext;
    PetscViewer viewer;
    PetscBool   flg;
    PetscCall(PetscViewerCreate(PETSC_COMM_WORLD, &viewer));
    PetscCall(PetscStrrchr(filename, '.', &ext));
    PetscCall(PetscStrcmp("vts", ext, &flg));
    if (flg) {
      PetscCall(PetscViewerSetType(viewer, PETSCVIEWERVTK));
    } else {
      PetscCall(PetscViewerSetType(viewer, PETSCVIEWERBINARY));
    }
    PetscCall(PetscViewerFileSetMode(viewer, FILE_MODE_WRITE));
    PetscCall(PetscViewerFileSetName(viewer, filename));
    PetscCall(VecView(X, viewer));
    PetscCall(PetscViewerDestroy(&viewer));
  }
  if (output_gnuplot) PetscCall(DMDAViewGnuplot2d(da_Stokes, X, "Velocity solution for Stokes eqn.", "X"));

  if (glvis) {
    PetscViewer view;

    PetscCall(PetscViewerCreate(PETSC_COMM_WORLD, &view));
    PetscCall(PetscViewerSetType(view, PETSCVIEWERGLVIS));
    PetscCall(VecView(X, view));
    PetscCall(PetscViewerDestroy(&view));
  }

  PetscCall(KSPGetIterationNumber(ksp_S, &its));

  if (coefficient_structure == 0) {
    PetscReal opts_eta0, opts_eta1, opts_xc;
    PetscInt  opts_nz, N;
    DM        da_Stokes_analytic;
    Vec       X_analytic;
    PetscReal nrm1[3], nrm2[3], nrmI[3];

    opts_eta0 = 1.0;
    opts_eta1 = 1.0;
    opts_xc   = 0.5;
    opts_nz   = 1;

    PetscCall(PetscOptionsGetReal(NULL, NULL, "-solcx_eta0", &opts_eta0, NULL));
    PetscCall(PetscOptionsGetReal(NULL, NULL, "-solcx_eta1", &opts_eta1, NULL));
    PetscCall(PetscOptionsGetReal(NULL, NULL, "-solcx_xc", &opts_xc, NULL));
    PetscCall(PetscOptionsGetInt(NULL, NULL, "-solcx_nz", &opts_nz, NULL));

    PetscCall(DMDACreateSolCx(opts_eta0, opts_eta1, opts_xc, opts_nz, mx, my, &da_Stokes_analytic, &X_analytic));
    if (output_gnuplot) PetscCall(DMDAViewGnuplot2d(da_Stokes_analytic, X_analytic, "Analytic solution for Stokes eqn.", "X_analytic"));
    PetscCall(DMDAIntegrateErrors(da_Stokes_analytic, X, X_analytic));

    PetscCall(VecAXPY(X_analytic, -1.0, X));
    PetscCall(VecGetSize(X_analytic, &N));
    N = N / 3;

    PetscCall(VecStrideNorm(X_analytic, 0, NORM_1, &nrm1[0]));
    PetscCall(VecStrideNorm(X_analytic, 0, NORM_2, &nrm2[0]));
    PetscCall(VecStrideNorm(X_analytic, 0, NORM_INFINITY, &nrmI[0]));

    PetscCall(VecStrideNorm(X_analytic, 1, NORM_1, &nrm1[1]));
    PetscCall(VecStrideNorm(X_analytic, 1, NORM_2, &nrm2[1]));
    PetscCall(VecStrideNorm(X_analytic, 1, NORM_INFINITY, &nrmI[1]));

    PetscCall(VecStrideNorm(X_analytic, 2, NORM_1, &nrm1[2]));
    PetscCall(VecStrideNorm(X_analytic, 2, NORM_2, &nrm2[2]));
    PetscCall(VecStrideNorm(X_analytic, 2, NORM_INFINITY, &nrmI[2]));

    PetscCall(DMDestroy(&da_Stokes_analytic));
    PetscCall(VecDestroy(&X_analytic));
  }

  PetscCall(KSPDestroy(&ksp_S));
  PetscCall(VecDestroy(&X));
  PetscCall(VecDestroy(&f));
  PetscCall(MatDestroy(&A));
  PetscCall(MatDestroy(&B));

  PetscCall(DMDestroy(&da_Stokes));
  PetscCall(DMDestroy(&da_prop));

  PetscCall(VecDestroy(&properties));
  PetscCall(VecDestroy(&l_properties));
  PetscFunctionReturn(0);
}

int main(int argc, char **args)
{
  PetscInt mx, my;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &args, (char *)0, help));
  mx = my = 10;
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-mx", &mx, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-my", &my, NULL));
  PetscCall(solve_stokes_2d_coupled(mx, my));
  PetscCall(PetscFinalize());
  return 0;
}

/* -------------------------- helpers for boundary conditions -------------------------------- */
static PetscErrorCode BCApplyZero_EAST(DM da, PetscInt d_idx, Mat A, Vec b)
{
  DM                     cda;
  Vec                    coords;
  PetscInt               si, sj, nx, ny, i, j;
  PetscInt               M, N;
  DMDACoor2d           **_coords;
  const PetscInt        *g_idx;
  PetscInt              *bc_global_ids;
  PetscScalar           *bc_vals;
  PetscInt               nbcs;
  PetscInt               n_dofs;
  ISLocalToGlobalMapping ltogm;

  PetscFunctionBeginUser;
  PetscCall(DMGetLocalToGlobalMapping(da, &ltogm));
  PetscCall(ISLocalToGlobalMappingGetIndices(ltogm, &g_idx));

  PetscCall(DMGetCoordinateDM(da, &cda));
  PetscCall(DMGetCoordinatesLocal(da, &coords));
  PetscCall(DMDAVecGetArray(cda, coords, &_coords));
  PetscCall(DMDAGetGhostCorners(cda, &si, &sj, 0, &nx, &ny, 0));
  PetscCall(DMDAGetInfo(da, 0, &M, &N, 0, 0, 0, 0, &n_dofs, 0, 0, 0, 0, 0));

  PetscCall(PetscMalloc1(ny * n_dofs, &bc_global_ids));
  PetscCall(PetscMalloc1(ny * n_dofs, &bc_vals));

  /* init the entries to -1 so VecSetValues will ignore them */
  for (i = 0; i < ny * n_dofs; i++) bc_global_ids[i] = -1;

  i = nx - 1;
  for (j = 0; j < ny; j++) {
    PetscInt local_id;

    local_id = i + j * nx;

    bc_global_ids[j] = g_idx[n_dofs * local_id + d_idx];

    bc_vals[j] = 0.0;
  }
  PetscCall(ISLocalToGlobalMappingRestoreIndices(ltogm, &g_idx));
  nbcs = 0;
  if ((si + nx) == (M)) nbcs = ny;

  if (b) {
    PetscCall(VecSetValues(b, nbcs, bc_global_ids, bc_vals, INSERT_VALUES));
    PetscCall(VecAssemblyBegin(b));
    PetscCall(VecAssemblyEnd(b));
  }
  if (A) PetscCall(MatZeroRowsColumns(A, nbcs, bc_global_ids, 1.0, 0, 0));

  PetscCall(PetscFree(bc_vals));
  PetscCall(PetscFree(bc_global_ids));

  PetscCall(DMDAVecRestoreArray(cda, coords, &_coords));
  PetscFunctionReturn(0);
}

static PetscErrorCode BCApplyZero_WEST(DM da, PetscInt d_idx, Mat A, Vec b)
{
  DM                     cda;
  Vec                    coords;
  PetscInt               si, sj, nx, ny, i, j;
  PetscInt               M, N;
  DMDACoor2d           **_coords;
  const PetscInt        *g_idx;
  PetscInt              *bc_global_ids;
  PetscScalar           *bc_vals;
  PetscInt               nbcs;
  PetscInt               n_dofs;
  ISLocalToGlobalMapping ltogm;

  PetscFunctionBeginUser;
  PetscCall(DMGetLocalToGlobalMapping(da, &ltogm));
  PetscCall(ISLocalToGlobalMappingGetIndices(ltogm, &g_idx));

  PetscCall(DMGetCoordinateDM(da, &cda));
  PetscCall(DMGetCoordinatesLocal(da, &coords));
  PetscCall(DMDAVecGetArray(cda, coords, &_coords));
  PetscCall(DMDAGetGhostCorners(cda, &si, &sj, 0, &nx, &ny, 0));
  PetscCall(DMDAGetInfo(da, 0, &M, &N, 0, 0, 0, 0, &n_dofs, 0, 0, 0, 0, 0));

  PetscCall(PetscMalloc1(ny * n_dofs, &bc_global_ids));
  PetscCall(PetscMalloc1(ny * n_dofs, &bc_vals));

  /* init the entries to -1 so VecSetValues will ignore them */
  for (i = 0; i < ny * n_dofs; i++) bc_global_ids[i] = -1;

  i = 0;
  for (j = 0; j < ny; j++) {
    PetscInt local_id;

    local_id = i + j * nx;

    bc_global_ids[j] = g_idx[n_dofs * local_id + d_idx];

    bc_vals[j] = 0.0;
  }
  PetscCall(ISLocalToGlobalMappingRestoreIndices(ltogm, &g_idx));
  nbcs = 0;
  if (si == 0) nbcs = ny;

  if (b) {
    PetscCall(VecSetValues(b, nbcs, bc_global_ids, bc_vals, INSERT_VALUES));
    PetscCall(VecAssemblyBegin(b));
    PetscCall(VecAssemblyEnd(b));
  }

  if (A) PetscCall(MatZeroRowsColumns(A, nbcs, bc_global_ids, 1.0, 0, 0));

  PetscCall(PetscFree(bc_vals));
  PetscCall(PetscFree(bc_global_ids));

  PetscCall(DMDAVecRestoreArray(cda, coords, &_coords));
  PetscFunctionReturn(0);
}

static PetscErrorCode BCApplyZero_NORTH(DM da, PetscInt d_idx, Mat A, Vec b)
{
  DM                     cda;
  Vec                    coords;
  PetscInt               si, sj, nx, ny, i, j;
  PetscInt               M, N;
  DMDACoor2d           **_coords;
  const PetscInt        *g_idx;
  PetscInt              *bc_global_ids;
  PetscScalar           *bc_vals;
  PetscInt               nbcs;
  PetscInt               n_dofs;
  ISLocalToGlobalMapping ltogm;

  PetscFunctionBeginUser;
  PetscCall(DMGetLocalToGlobalMapping(da, &ltogm));
  PetscCall(ISLocalToGlobalMappingGetIndices(ltogm, &g_idx));

  PetscCall(DMGetCoordinateDM(da, &cda));
  PetscCall(DMGetCoordinatesLocal(da, &coords));
  PetscCall(DMDAVecGetArray(cda, coords, &_coords));
  PetscCall(DMDAGetGhostCorners(cda, &si, &sj, 0, &nx, &ny, 0));
  PetscCall(DMDAGetInfo(da, 0, &M, &N, 0, 0, 0, 0, &n_dofs, 0, 0, 0, 0, 0));

  PetscCall(PetscMalloc1(nx, &bc_global_ids));
  PetscCall(PetscMalloc1(nx, &bc_vals));

  /* init the entries to -1 so VecSetValues will ignore them */
  for (i = 0; i < nx; i++) bc_global_ids[i] = -1;

  j = ny - 1;
  for (i = 0; i < nx; i++) {
    PetscInt local_id;

    local_id = i + j * nx;

    bc_global_ids[i] = g_idx[n_dofs * local_id + d_idx];

    bc_vals[i] = 0.0;
  }
  PetscCall(ISLocalToGlobalMappingRestoreIndices(ltogm, &g_idx));
  nbcs = 0;
  if ((sj + ny) == (N)) nbcs = nx;

  if (b) {
    PetscCall(VecSetValues(b, nbcs, bc_global_ids, bc_vals, INSERT_VALUES));
    PetscCall(VecAssemblyBegin(b));
    PetscCall(VecAssemblyEnd(b));
  }
  if (A) PetscCall(MatZeroRowsColumns(A, nbcs, bc_global_ids, 1.0, 0, 0));

  PetscCall(PetscFree(bc_vals));
  PetscCall(PetscFree(bc_global_ids));

  PetscCall(DMDAVecRestoreArray(cda, coords, &_coords));
  PetscFunctionReturn(0);
}

static PetscErrorCode BCApplyZero_SOUTH(DM da, PetscInt d_idx, Mat A, Vec b)
{
  DM                     cda;
  Vec                    coords;
  PetscInt               si, sj, nx, ny, i, j;
  PetscInt               M, N;
  DMDACoor2d           **_coords;
  const PetscInt        *g_idx;
  PetscInt              *bc_global_ids;
  PetscScalar           *bc_vals;
  PetscInt               nbcs;
  PetscInt               n_dofs;
  ISLocalToGlobalMapping ltogm;

  PetscFunctionBeginUser;
  PetscCall(DMGetLocalToGlobalMapping(da, &ltogm));
  PetscCall(ISLocalToGlobalMappingGetIndices(ltogm, &g_idx));

  PetscCall(DMGetCoordinateDM(da, &cda));
  PetscCall(DMGetCoordinatesLocal(da, &coords));
  PetscCall(DMDAVecGetArray(cda, coords, &_coords));
  PetscCall(DMDAGetGhostCorners(cda, &si, &sj, 0, &nx, &ny, 0));
  PetscCall(DMDAGetInfo(da, 0, &M, &N, 0, 0, 0, 0, &n_dofs, 0, 0, 0, 0, 0));

  PetscCall(PetscMalloc1(nx, &bc_global_ids));
  PetscCall(PetscMalloc1(nx, &bc_vals));

  /* init the entries to -1 so VecSetValues will ignore them */
  for (i = 0; i < nx; i++) bc_global_ids[i] = -1;

  j = 0;
  for (i = 0; i < nx; i++) {
    PetscInt local_id;

    local_id = i + j * nx;

    bc_global_ids[i] = g_idx[n_dofs * local_id + d_idx];

    bc_vals[i] = 0.0;
  }
  PetscCall(ISLocalToGlobalMappingRestoreIndices(ltogm, &g_idx));
  nbcs = 0;
  if (sj == 0) nbcs = nx;

  if (b) {
    PetscCall(VecSetValues(b, nbcs, bc_global_ids, bc_vals, INSERT_VALUES));
    PetscCall(VecAssemblyBegin(b));
    PetscCall(VecAssemblyEnd(b));
  }
  if (A) PetscCall(MatZeroRowsColumns(A, nbcs, bc_global_ids, 1.0, 0, 0));

  PetscCall(PetscFree(bc_vals));
  PetscCall(PetscFree(bc_global_ids));

  PetscCall(DMDAVecRestoreArray(cda, coords, &_coords));
  PetscFunctionReturn(0);
}

/* Impose free slip boundary conditions; u_{i} n_{i} = 0, tau_{ij} t_j = 0 */
static PetscErrorCode DMDABCApplyFreeSlip(DM da_Stokes, Mat A, Vec f)
{
  PetscFunctionBeginUser;
  PetscCall(BCApplyZero_NORTH(da_Stokes, 1, A, f));
  PetscCall(BCApplyZero_EAST(da_Stokes, 0, A, f));
  PetscCall(BCApplyZero_SOUTH(da_Stokes, 1, A, f));
  PetscCall(BCApplyZero_WEST(da_Stokes, 0, A, f));
  PetscFunctionReturn(0);
}

/*TEST

   build:
      requires: !complex !single

   test:
      args: -stokes_pc_use_amat -stokes_ksp_type fgmres -stokes_pc_type fieldsplit -stokes_pc_fieldsplit_block_size 3 -stokes_pc_fieldsplit_type SYMMETRIC_MULTIPLICATIVE -stokes_pc_fieldsplit_0_fields 0,1 -stokes_pc_fieldsplit_1_fields 2 -stokes_fieldsplit_0_ksp_type preonly -stokes_fieldsplit_0_pc_type lu -stokes_fieldsplit_1_ksp_type preonly -stokes_fieldsplit_1_pc_type jacobi -c_str 0 -solcx_eta0 1.0 -solcx_eta1 1.0e6 -solcx_xc 0.5 -solcx_nz 2 -mx 20 -my 20 -stokes_ksp_monitor_short

   testset:
      args: -stokes_pc_use_amat -stokes_ksp_type fgmres -stokes_pc_type fieldsplit -stokes_pc_fieldsplit_block_size 3 -stokes_pc_fieldsplit_type SYMMETRIC_MULTIPLICATIVE -stokes_fieldsplit_u_ksp_type preonly -stokes_fieldsplit_u_pc_type lu -stokes_fieldsplit_p_ksp_type preonly -stokes_fieldsplit_p_pc_type jacobi -c_str 0 -solcx_eta0 1.0 -solcx_eta1 1.0e6 -solcx_xc 0.5 -solcx_nz 2 -mx 20 -my 20 -stokes_ksp_monitor_short
      test:
         suffix: 2
         args:
         output_file: output/ex43_1.out
      test:
         requires: mumps
         suffix: 2_mumps
         args: -change true -stokes_ksp_view
         output_file: output/ex43_2_mumps.out
         # mumps INFO,INFOG,RINFO,RINFOG may vary on different archs, so keep just a stable one
         filter:  grep -E -v "(INFOG\([^7]|INFO\(|\[0\])"

   test:
      suffix: 3
      nsize: 4
      args: -stokes_pc_use_amat -stokes_ksp_type gcr -stokes_ksp_gcr_restart 60 -stokes_ksp_norm_type unpreconditioned -stokes_ksp_rtol 1.e-2 -c_str 3 -sinker_eta0 1.0 -sinker_eta1 100 -sinker_dx 0.4 -sinker_dy 0.3 -mx 128 -my 128 -stokes_ksp_monitor_short -stokes_pc_type mg -stokes_mg_levels_pc_type fieldsplit -stokes_pc_use_amat false -stokes_pc_mg_galerkin pmat -stokes_mg_levels_pc_fieldsplit_block_size 3 -stokes_mg_levels_pc_fieldsplit_0_fields 0,1 -stokes_mg_levels_pc_fieldsplit_1_fields 2 -stokes_mg_levels_fieldsplit_0_pc_type sor -stokes_mg_levels_fieldsplit_1_pc_type sor -stokes_mg_levels_ksp_type chebyshev -stokes_mg_levels_ksp_max_it 1 -stokes_mg_levels_ksp_chebyshev_esteig 0,0.2,0,1.1 -stokes_pc_mg_levels 4 -stokes_ksp_view

   test:
      suffix: 4
      nsize: 4
      args: -stokes_ksp_type pipegcr -stokes_ksp_pipegcr_mmax 60 -stokes_ksp_pipegcr_unroll_w 1 -stokes_ksp_norm_type natural -c_str 3 -sinker_eta0 1.0 -sinker_eta1 100 -sinker_dx 0.4 -sinker_dy 0.3 -mx 128 -my 128 -stokes_ksp_monitor_short -stokes_pc_type mg -stokes_mg_levels_pc_type fieldsplit -stokes_pc_use_amat false -stokes_pc_mg_galerkin pmat -stokes_mg_levels_pc_fieldsplit_block_size 3 -stokes_mg_levels_pc_fieldsplit_0_fields 0,1 -stokes_mg_levels_pc_fieldsplit_1_fields 2 -stokes_mg_levels_fieldsplit_0_pc_type sor -stokes_mg_levels_fieldsplit_1_pc_type sor -stokes_mg_levels_ksp_type chebyshev -stokes_mg_levels_ksp_max_it 1 -stokes_mg_levels_ksp_chebyshev_esteig 0,0.2,0,1.1 -stokes_pc_mg_levels 4 -stokes_ksp_view

   test:
      suffix: 5
      nsize: 4
      args: -stokes_pc_fieldsplit_off_diag_use_amat -stokes_ksp_type pipegcr -stokes_pc_type fieldsplit -stokes_pc_fieldsplit_block_size 3 -stokes_pc_fieldsplit_type SYMMETRIC_MULTIPLICATIVE -stokes_pc_fieldsplit_0_fields 0,1 -stokes_pc_fieldsplit_1_fields 2 -stokes_fieldsplit_0_ksp_type preonly -stokes_fieldsplit_0_pc_type bjacobi -stokes_fieldsplit_1_ksp_type preonly -stokes_fieldsplit_1_pc_type bjacobi -c_str 0 -solcx_eta0 1.0 -solcx_eta1 1.0e6 -solcx_xc 0.5 -solcx_nz 2 -mx 20 -my 20 -stokes_ksp_monitor_short -stokes_ksp_view

   test:
      suffix: 6
      nsize: 8
      args: -stokes_ksp_view -stokes_pc_type mg -stokes_pc_mg_levels 2 -stokes_mg_coarse_pc_type telescope -stokes_mg_coarse_pc_telescope_reduction_factor 2 -stokes_pc_use_amat false -stokes_pc_mg_galerkin pmat -stokes_mg_coarse_pc_telescope_subcomm_type contiguous

   test:
      suffix: bjacobi
      nsize: 4
      args: -stokes_ksp_rtol 1.e-4 -stokes_pc_type bjacobi -stokes_pc_bjacobi_blocks 2 -dm_mat_type aij -stokes_ksp_converged_reason

   test:
      suffix: bjacobi_baij
      nsize: 4
      args: -stokes_ksp_rtol 1.e-4 -stokes_pc_type bjacobi -stokes_pc_bjacobi_blocks 2 -dm_mat_type baij -stokes_ksp_converged_reason
      output_file: output/ex43_bjacobi.out

   test:
      suffix: nested_gmg
      nsize: 4
      args: -stokes_pc_fieldsplit_off_diag_use_amat -mx 16 -my 16 -stokes_ksp_type fgmres -stokes_pc_type fieldsplit -stokes_fieldsplit_u_pc_type mg -stokes_fieldsplit_u_pc_mg_levels 5 -stokes_fieldsplit_u_pc_mg_galerkin pmat -stokes_fieldsplit_u_ksp_type cg -stokes_fieldsplit_u_ksp_rtol 1.0e-4 -stokes_fieldsplit_u_mg_levels_pc_type jacobi -solcx_eta0 1.0e4 -stokes_fieldsplit_u_ksp_converged_reason -stokes_ksp_converged_reason -stokes_fieldsplit_p_sub_pc_factor_zeropivot 1.e-8

   test:
      suffix: fetidp
      nsize: 8
      args: -dm_mat_type is -stokes_ksp_type fetidp -stokes_ksp_fetidp_saddlepoint -stokes_fetidp_ksp_type cg -stokes_ksp_converged_reason -stokes_fetidp_pc_fieldsplit_schur_fact_type diag -stokes_fetidp_fieldsplit_p_pc_type bjacobi -stokes_fetidp_fieldsplit_lag_ksp_type preonly -stokes_fetidp_fieldsplit_p_ksp_type preonly -stokes_ksp_fetidp_pressure_field 2 -stokes_fetidp_pc_fieldsplit_schur_scale -1

   test:
      suffix: fetidp_unsym
      nsize: 8
      args: -dm_mat_type is -stokes_ksp_type fetidp -stokes_ksp_monitor_true_residual -stokes_ksp_converged_reason -stokes_fetidp_bddc_pc_bddc_coarse_redundant_pc_type svd

   test:
      suffix: bddc_stokes_deluxe
      nsize: 8
      args: -stokes_ksp_monitor_short -stokes_ksp_converged_reason -stokes_pc_type bddc -dm_mat_type is -stokes_pc_bddc_coarse_redundant_pc_type svd -stokes_pc_bddc_use_deluxe_scaling -stokes_sub_schurs_posdef 0 -stokes_sub_schurs_symmetric -stokes_sub_schurs_mat_solver_type petsc

   test:
      suffix: bddc_stokes_subdomainjump_deluxe
      nsize: 9
      args: -c_str 4 -jump_magnitude 3 -stokes_ksp_monitor_short -stokes_ksp_converged_reason -stokes_pc_type bddc -dm_mat_type is -stokes_pc_bddc_coarse_redundant_pc_type svd -stokes_pc_bddc_use_deluxe_scaling -stokes_sub_schurs_posdef 0 -stokes_sub_schurs_symmetric -stokes_sub_schurs_mat_solver_type petsc

TEST*/
