static char help[] =  "   Solves the compressible plane strain elasticity equations in 2d on the unit domain using Q1 finite elements. \n\
   Material properties E (Youngs modulus) and nu (Poisson ratio) may vary as a function of space. \n\
   The model utilises boundary conditions which produce compression in the x direction. \n\
Options: \n"
"\
     -mx : number of elements in x-direction \n\
     -my : number of elements in y-direction \n\
     -c_str : structure of the coefficients to use. \n"
"\
          -c_str 0 => isotropic material with constant coefficients. \n\
                         Parameters: \n\
                             -iso_E  : Youngs modulus \n\
                             -iso_nu : Poisson ratio \n\
          -c_str 1 => step function in the material properties in x. \n\
                         Parameters: \n\
                              -step_E0  : Youngs modulus to the left of the step \n\
                              -step_nu0 : Poisson ratio to the left of the step \n\
                              -step_E1  : Youngs modulus to the right of the step \n\
                              -step_n1  : Poisson ratio to the right of the step \n\
                              -step_xc  : x coordinate of the step \n"
"\
          -c_str 2 => checkerboard material with alternating properties. \n\
                      Repeats the following pattern throughout the domain. For example with 4 materials specified, we would heve \n\
                      -------------------------\n\
                      |  D  |  A  |  B  |  C  |\n\
                      ------|-----|-----|------\n\
                      |  C  |  D  |  A  |  B  |\n\
                      ------|-----|-----|------\n\
                      |  B  |  C  |  D  |  A  |\n\
                      ------|-----|-----|------\n\
                      |  A  |  B  |  C  |  D  |\n\
                      -------------------------\n\
                      \n\
                         Parameters: \n\
                              -brick_E    : a comma separated list of Young's modulii \n\
                              -brick_nu   : a comma separated list of Poisson ratios  \n\
                              -brick_span : the number of elements in x and y each brick will span \n\
          -c_str 3 => sponge-like material with alternating properties. \n\
                      Repeats the following pattern throughout the domain \n"
"\
                      -----------------------------\n\
                      |       [background]        |\n\
                      |          E0,nu0           |\n\
                      |     -----------------     |\n\
                      |     |  [inclusion]  |     |\n\
                      |     |    E1,nu1     |     |\n\
                      |     |               |     |\n\
                      |     | <---- w ----> |     |\n\
                      |     |               |     |\n\
                      |     |               |     |\n\
                      |     -----------------     |\n\
                      |                           |\n\
                      |                           |\n\
                      -----------------------------\n\
                      <--------  t + w + t ------->\n\
                      \n\
                         Parameters: \n\
                              -sponge_E0  : Youngs modulus of the surrounding material \n\
                              -sponge_E1  : Youngs modulus of the inclusion \n\
                              -sponge_nu0 : Poisson ratio of the surrounding material \n\
                              -sponge_nu1 : Poisson ratio of the inclusion \n\
                              -sponge_t   : the number of elements defining the border around each inclusion \n\
                              -sponge_w   : the number of elements in x and y each inclusion will span\n\
     -use_gp_coords : Evaluate the Youngs modulus, Poisson ratio and the body force at the global coordinates of the quadrature points.\n\
     By default, E, nu and the body force are evaulated at the element center and applied as a constant over the entire element.\n\
     -use_nonsymbc : Option to use non-symmetric boundary condition imposition. This choice will use less memory.";

/* Contributed by Dave May */

#include <petscksp.h>
#include <petscdm.h>
#include <petscdmda.h>

static PetscErrorCode DMDABCApplyCompression(DM,Mat,Vec);
static PetscErrorCode DMDABCApplySymmetricCompression(DM elas_da,Mat A,Vec f,IS *dofs,Mat *AA,Vec *ff);

#define NSD            2 /* number of spatial dimensions */
#define NODES_PER_EL   4 /* nodes per element */
#define U_DOFS         2 /* degrees of freedom per displacement node */
#define GAUSS_POINTS   4

/* cell based evaluation */
typedef struct {
  PetscScalar E,nu,fx,fy;
} Coefficients;

/* Gauss point based evaluation 8+4+4+4 = 20 */
typedef struct {
  PetscScalar gp_coords[2*GAUSS_POINTS];
  PetscScalar E[GAUSS_POINTS];
  PetscScalar nu[GAUSS_POINTS];
  PetscScalar fx[GAUSS_POINTS];
  PetscScalar fy[GAUSS_POINTS];
} GaussPointCoefficients;

typedef struct {
  PetscScalar ux_dof;
  PetscScalar uy_dof;
} ElasticityDOF;

/*

 D = E/((1+nu)(1-2nu)) * [ 1-nu   nu        0     ]
                         [  nu   1-nu       0     ]
                         [  0     0   0.5*(1-2nu) ]

 B = [ d_dx   0   ]
     [  0    d_dy ]
     [ d_dy  d_dx ]

 */

/* FEM routines */
/*
 Element: Local basis function ordering
 1-----2
 |     |
 |     |
 0-----3
 */
static void ConstructQ12D_Ni(PetscScalar _xi[],PetscScalar Ni[])
{
  PetscScalar xi  = _xi[0];
  PetscScalar eta = _xi[1];

  Ni[0] = 0.25*(1.0-xi)*(1.0-eta);
  Ni[1] = 0.25*(1.0-xi)*(1.0+eta);
  Ni[2] = 0.25*(1.0+xi)*(1.0+eta);
  Ni[3] = 0.25*(1.0+xi)*(1.0-eta);
}

static void ConstructQ12D_GNi(PetscScalar _xi[],PetscScalar GNi[][NODES_PER_EL])
{
  PetscScalar xi  = _xi[0];
  PetscScalar eta = _xi[1];

  GNi[0][0] = -0.25*(1.0-eta);
  GNi[0][1] = -0.25*(1.0+eta);
  GNi[0][2] =   0.25*(1.0+eta);
  GNi[0][3] =   0.25*(1.0-eta);

  GNi[1][0] = -0.25*(1.0-xi);
  GNi[1][1] =   0.25*(1.0-xi);
  GNi[1][2] =   0.25*(1.0+xi);
  GNi[1][3] = -0.25*(1.0+xi);
}

static void ConstructQ12D_GNx(PetscScalar GNi[][NODES_PER_EL],PetscScalar GNx[][NODES_PER_EL],PetscScalar coords[],PetscScalar *det_J)
{
  PetscScalar J00,J01,J10,J11,J;
  PetscScalar iJ00,iJ01,iJ10,iJ11;
  PetscInt    i;

  J00 = J01 = J10 = J11 = 0.0;
  for (i = 0; i < NODES_PER_EL; i++) {
    PetscScalar cx = coords[2*i+0];
    PetscScalar cy = coords[2*i+1];

    J00 = J00+GNi[0][i]*cx;      /* J_xx = dx/dxi */
    J01 = J01+GNi[0][i]*cy;      /* J_xy = dy/dxi */
    J10 = J10+GNi[1][i]*cx;      /* J_yx = dx/deta */
    J11 = J11+GNi[1][i]*cy;      /* J_yy = dy/deta */
  }
  J = (J00*J11)-(J01*J10);

  iJ00 =  J11/J;
  iJ01 = -J01/J;
  iJ10 = -J10/J;
  iJ11 =  J00/J;

  for (i = 0; i < NODES_PER_EL; i++) {
    GNx[0][i] = GNi[0][i]*iJ00+GNi[1][i]*iJ01;
    GNx[1][i] = GNi[0][i]*iJ10+GNi[1][i]*iJ11;
  }

  if (det_J) *det_J = J;
}

static void ConstructGaussQuadrature(PetscInt *ngp,PetscScalar gp_xi[][2],PetscScalar gp_weight[])
{
  *ngp         = 4;
  gp_xi[0][0]  = -0.57735026919;gp_xi[0][1] = -0.57735026919;
  gp_xi[1][0]  = -0.57735026919;gp_xi[1][1] =  0.57735026919;
  gp_xi[2][0]  =  0.57735026919;gp_xi[2][1] =  0.57735026919;
  gp_xi[3][0]  =  0.57735026919;gp_xi[3][1] = -0.57735026919;
  gp_weight[0] = 1.0;
  gp_weight[1] = 1.0;
  gp_weight[2] = 1.0;
  gp_weight[3] = 1.0;
}

static PetscErrorCode DMDAGetElementOwnershipRanges2d(DM da,PetscInt **_lx,PetscInt **_ly)
{
  PetscMPIInt    rank;
  PetscInt       proc_I,proc_J;
  PetscInt       cpu_x,cpu_y;
  PetscInt       local_mx,local_my;
  Vec            vlx,vly;
  PetscInt       *LX,*LY,i;
  PetscScalar    *_a;
  Vec            V_SEQ;
  VecScatter     ctx;

  PetscFunctionBeginUser;
  MPI_Comm_rank(PETSC_COMM_WORLD,&rank);

  DMDAGetInfo(da,0,0,0,0,&cpu_x,&cpu_y,0,0,0,0,0,0,0);

  proc_J = rank/cpu_x;
  proc_I = rank-cpu_x*proc_J;

  PetscCall(PetscMalloc1(cpu_x,&LX));
  PetscCall(PetscMalloc1(cpu_y,&LY));

  PetscCall(DMDAGetElementsSizes(da,&local_mx,&local_my,NULL));
  PetscCall(VecCreate(PETSC_COMM_WORLD,&vlx));
  PetscCall(VecSetSizes(vlx,PETSC_DECIDE,cpu_x));
  PetscCall(VecSetFromOptions(vlx));

  PetscCall(VecCreate(PETSC_COMM_WORLD,&vly));
  PetscCall(VecSetSizes(vly,PETSC_DECIDE,cpu_y));
  PetscCall(VecSetFromOptions(vly));

  PetscCall(VecSetValue(vlx,proc_I,(PetscScalar)(local_mx+1.0e-9),INSERT_VALUES));
  PetscCall(VecSetValue(vly,proc_J,(PetscScalar)(local_my+1.0e-9),INSERT_VALUES));
  PetscCall(VecAssemblyBegin(vlx);VecAssemblyEnd(vlx));
  PetscCall(VecAssemblyBegin(vly);VecAssemblyEnd(vly));

  PetscCall(VecScatterCreateToAll(vlx,&ctx,&V_SEQ));
  PetscCall(VecScatterBegin(ctx,vlx,V_SEQ,INSERT_VALUES,SCATTER_FORWARD));
  PetscCall(VecScatterEnd(ctx,vlx,V_SEQ,INSERT_VALUES,SCATTER_FORWARD));
  PetscCall(VecGetArray(V_SEQ,&_a));
  for (i = 0; i < cpu_x; i++) LX[i] = (PetscInt)PetscRealPart(_a[i]);
  PetscCall(VecRestoreArray(V_SEQ,&_a));
  PetscCall(VecScatterDestroy(&ctx));
  PetscCall(VecDestroy(&V_SEQ));

  PetscCall(VecScatterCreateToAll(vly,&ctx,&V_SEQ));
  PetscCall(VecScatterBegin(ctx,vly,V_SEQ,INSERT_VALUES,SCATTER_FORWARD));
  PetscCall(VecScatterEnd(ctx,vly,V_SEQ,INSERT_VALUES,SCATTER_FORWARD));
  PetscCall(VecGetArray(V_SEQ,&_a));
  for (i = 0; i < cpu_y; i++) LY[i] = (PetscInt)PetscRealPart(_a[i]);
  PetscCall(VecRestoreArray(V_SEQ,&_a));
  PetscCall(VecScatterDestroy(&ctx));
  PetscCall(VecDestroy(&V_SEQ));

  *_lx = LX;
  *_ly = LY;

  PetscCall(VecDestroy(&vlx));
  PetscCall(VecDestroy(&vly));
  PetscFunctionReturn(0);
}

static PetscErrorCode DMDACoordViewGnuplot2d(DM da,const char prefix[])
{
  DM             cda;
  Vec            coords;
  DMDACoor2d     **_coords;
  PetscInt       si,sj,nx,ny,i,j;
  FILE           *fp;
  char           fname[PETSC_MAX_PATH_LEN];
  PetscMPIInt    rank;

  PetscFunctionBeginUser;
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  PetscCall(PetscSNPrintf(fname,sizeof(fname),"%s-p%1.4d.dat",prefix,rank));
  PetscCall(PetscFOpen(PETSC_COMM_SELF,fname,"w",&fp));
  PetscCheck(fp,PETSC_COMM_SELF,PETSC_ERR_USER,"Cannot open file");
  PetscCall(PetscFPrintf(PETSC_COMM_SELF,fp,"### Element geometry for processor %1.4d ### \n",rank));

  PetscCall(DMGetCoordinateDM(da,&cda));
  PetscCall(DMGetCoordinatesLocal(da,&coords));
  PetscCall(DMDAVecGetArray(cda,coords,&_coords));
  PetscCall(DMDAGetGhostCorners(cda,&si,&sj,0,&nx,&ny,0));
  for (j = sj; j < sj+ny-1; j++) {
    for (i = si; i < si+nx-1; i++) {
      PetscCall(PetscFPrintf(PETSC_COMM_SELF,fp,"%1.6e %1.6e \n",(double)PetscRealPart(_coords[j][i].x),(double)PetscRealPart(_coords[j][i].y)));
      PetscCall(PetscFPrintf(PETSC_COMM_SELF,fp,"%1.6e %1.6e \n",(double)PetscRealPart(_coords[j+1][i].x),(double)PetscRealPart(_coords[j+1][i].y)));
      PetscCall(PetscFPrintf(PETSC_COMM_SELF,fp,"%1.6e %1.6e \n",(double)PetscRealPart(_coords[j+1][i+1].x),(double)PetscRealPart(_coords[j+1][i+1].y)));
      PetscCall(PetscFPrintf(PETSC_COMM_SELF,fp,"%1.6e %1.6e \n",(double)PetscRealPart(_coords[j][i+1].x),(double)PetscRealPart(_coords[j][i+1].y)));
      PetscCall(PetscFPrintf(PETSC_COMM_SELF,fp,"%1.6e %1.6e \n\n",(double)PetscRealPart(_coords[j][i].x),(double)PetscRealPart(_coords[j][i].y)));
    }
  }
  PetscCall(DMDAVecRestoreArray(cda,coords,&_coords));

  PetscCall(PetscFClose(PETSC_COMM_SELF,fp));
  PetscFunctionReturn(0);
}

static PetscErrorCode DMDAViewGnuplot2d(DM da,Vec fields,const char comment[],const char prefix[])
{
  DM             cda;
  Vec            coords,local_fields;
  DMDACoor2d     **_coords;
  FILE           *fp;
  char           fname[PETSC_MAX_PATH_LEN];
  const char     *field_name;
  PetscMPIInt    rank;
  PetscInt       si,sj,nx,ny,i,j;
  PetscInt       n_dofs,d;
  PetscScalar    *_fields;

  PetscFunctionBeginUser;
  MPI_Comm_rank(PETSC_COMM_WORLD,&rank);
  PetscCall(PetscSNPrintf(fname,sizeof(fname),"%s-p%1.4d.dat",prefix,rank));
  PetscCall(PetscFOpen(PETSC_COMM_SELF,fname,"w",&fp));
  PetscCheck(fp,PETSC_COMM_SELF,PETSC_ERR_USER,"Cannot open file");

  PetscCall(PetscFPrintf(PETSC_COMM_SELF,fp,"### %s (processor %1.4d) ### \n",comment,rank));
  PetscCall(DMDAGetInfo(da,0,0,0,0,0,0,0,&n_dofs,0,0,0,0,0));
  PetscCall(PetscFPrintf(PETSC_COMM_SELF,fp,"### x y "));
  for (d = 0; d < n_dofs; d++) {
    PetscCall(DMDAGetFieldName(da,d,&field_name));
    PetscCall(PetscFPrintf(PETSC_COMM_SELF,fp,"%s ",field_name));
  }
  PetscCall(PetscFPrintf(PETSC_COMM_SELF,fp,"###\n"));

  PetscCall(DMGetCoordinateDM(da,&cda));
  PetscCall(DMGetCoordinatesLocal(da,&coords));
  PetscCall(DMDAVecGetArray(cda,coords,&_coords));
  PetscCall(DMDAGetGhostCorners(cda,&si,&sj,0,&nx,&ny,0));

  PetscCall(DMCreateLocalVector(da,&local_fields));
  PetscCall(DMGlobalToLocalBegin(da,fields,INSERT_VALUES,local_fields));
  PetscCall(DMGlobalToLocalEnd(da,fields,INSERT_VALUES,local_fields));
  PetscCall(VecGetArray(local_fields,&_fields));

  for (j = sj; j < sj+ny; j++) {
    for (i = si; i < si+nx; i++) {
      PetscScalar coord_x,coord_y;
      PetscScalar field_d;

      coord_x = _coords[j][i].x;
      coord_y = _coords[j][i].y;

      PetscCall(PetscFPrintf(PETSC_COMM_SELF,fp,"%1.6e %1.6e ",(double)PetscRealPart(coord_x),(double)PetscRealPart(coord_y)));
      for (d = 0; d < n_dofs; d++) {
        field_d = _fields[n_dofs*((i-si)+(j-sj)*(nx))+d];
        PetscCall(PetscFPrintf(PETSC_COMM_SELF,fp,"%1.6e ",(double)PetscRealPart(field_d)));
      }
      PetscCall(PetscFPrintf(PETSC_COMM_SELF,fp,"\n"));
    }
  }
  PetscCall(VecRestoreArray(local_fields,&_fields));
  PetscCall(VecDestroy(&local_fields));

  PetscCall(DMDAVecRestoreArray(cda,coords,&_coords));

  PetscCall(PetscFClose(PETSC_COMM_SELF,fp));
  PetscFunctionReturn(0);
}

static PetscErrorCode DMDAViewCoefficientsGnuplot2d(DM da,Vec fields,const char comment[],const char prefix[])
{
  DM                     cda;
  Vec                    local_fields;
  FILE                   *fp;
  char                   fname[PETSC_MAX_PATH_LEN];
  const char             *field_name;
  PetscMPIInt            rank;
  PetscInt               si,sj,nx,ny,i,j,p;
  PetscInt               n_dofs,d;
  GaussPointCoefficients **_coefficients;

  PetscFunctionBeginUser;
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  PetscCall(PetscSNPrintf(fname,sizeof(fname),"%s-p%1.4d.dat",prefix,rank));
  PetscCall(PetscFOpen(PETSC_COMM_SELF,fname,"w",&fp));
  PetscCheck(fp,PETSC_COMM_SELF,PETSC_ERR_USER,"Cannot open file");

  PetscCall(PetscFPrintf(PETSC_COMM_SELF,fp,"### %s (processor %1.4d) ### \n",comment,rank));
  PetscCall(DMDAGetInfo(da,0,0,0,0,0,0,0,&n_dofs,0,0,0,0,0));
  PetscCall(PetscFPrintf(PETSC_COMM_SELF,fp,"### x y "));
  for (d = 0; d < n_dofs; d++) {
    PetscCall(DMDAGetFieldName(da,d,&field_name));
    PetscCall(PetscFPrintf(PETSC_COMM_SELF,fp,"%s ",field_name));
  }
  PetscCall(PetscFPrintf(PETSC_COMM_SELF,fp,"###\n"));

  PetscCall(DMGetCoordinateDM(da,&cda));
  PetscCall(DMDAGetGhostCorners(cda,&si,&sj,0,&nx,&ny,0));

  PetscCall(DMCreateLocalVector(da,&local_fields));
  PetscCall(DMGlobalToLocalBegin(da,fields,INSERT_VALUES,local_fields));
  PetscCall(DMGlobalToLocalEnd(da,fields,INSERT_VALUES,local_fields));
  PetscCall(DMDAVecGetArray(da,local_fields,&_coefficients));

  for (j = sj; j < sj+ny; j++) {
    for (i = si; i < si+nx; i++) {
      PetscScalar coord_x,coord_y;

      for (p = 0; p < GAUSS_POINTS; p++) {
        coord_x = _coefficients[j][i].gp_coords[2*p];
        coord_y = _coefficients[j][i].gp_coords[2*p+1];

        PetscCall(PetscFPrintf(PETSC_COMM_SELF,fp,"%1.6e %1.6e ",(double)PetscRealPart(coord_x),(double)PetscRealPart(coord_y)));

        PetscCall(PetscFPrintf(PETSC_COMM_SELF,fp,"%1.6e %1.6e %1.6e %1.6e\n",
                               (double)PetscRealPart(_coefficients[j][i].E[p]),(double)PetscRealPart(_coefficients[j][i].nu[p]),
                               (double)PetscRealPart(_coefficients[j][i].fx[p]),(double)PetscRealPart(_coefficients[j][i].fy[p])));
      }
    }
  }
  PetscCall(DMDAVecRestoreArray(da,local_fields,&_coefficients));
  PetscCall(VecDestroy(&local_fields));

  PetscCall(PetscFClose(PETSC_COMM_SELF,fp));
  PetscFunctionReturn(0);
}

static void FormStressOperatorQ1(PetscScalar Ke[],PetscScalar coords[],PetscScalar E[],PetscScalar nu[])
{
  PetscInt    ngp;
  PetscScalar gp_xi[GAUSS_POINTS][2];
  PetscScalar gp_weight[GAUSS_POINTS];
  PetscInt    p,i,j,k,l;
  PetscScalar GNi_p[NSD][NODES_PER_EL],GNx_p[NSD][NODES_PER_EL];
  PetscScalar J_p;
  PetscScalar B[3][U_DOFS*NODES_PER_EL];
  PetscScalar prop_E,prop_nu,factor,constit_D[3][3];

  /* define quadrature rule */
  ConstructGaussQuadrature(&ngp,gp_xi,gp_weight);

  /* evaluate integral */
  for (p = 0; p < ngp; p++) {
    ConstructQ12D_GNi(gp_xi[p],GNi_p);
    ConstructQ12D_GNx(GNi_p,GNx_p,coords,&J_p);

    for (i = 0; i < NODES_PER_EL; i++) {
      PetscScalar d_dx_i = GNx_p[0][i];
      PetscScalar d_dy_i = GNx_p[1][i];

      B[0][2*i] = d_dx_i;  B[0][2*i+1] = 0.0;
      B[1][2*i] = 0.0;     B[1][2*i+1] = d_dy_i;
      B[2][2*i] = d_dy_i;  B[2][2*i+1] = d_dx_i;
    }

    /* form D for the quadrature point */
    prop_E          = E[p];
    prop_nu         = nu[p];
    factor          = prop_E / ((1.0+prop_nu)*(1.0-2.0*prop_nu));
    constit_D[0][0] = 1.0-prop_nu;  constit_D[0][1] = prop_nu;      constit_D[0][2] = 0.0;
    constit_D[1][0] = prop_nu;      constit_D[1][1] = 1.0-prop_nu;  constit_D[1][2] = 0.0;
    constit_D[2][0] = 0.0;          constit_D[2][1] = 0.0;          constit_D[2][2] = 0.5*(1.0-2.0*prop_nu);
    for (i = 0; i < 3; i++) {
      for (j = 0; j < 3; j++) {
        constit_D[i][j] = factor * constit_D[i][j] * gp_weight[p] * J_p;
      }
    }

    /* form Bt tildeD B */
    /*
     Ke_ij = Bt_ik . D_kl . B_lj
     = B_ki . D_kl . B_lj
     */
    for (i = 0; i < 8; i++) {
      for (j = 0; j < 8; j++) {
        for (k = 0; k < 3; k++) {
          for (l = 0; l < 3; l++) {
            Ke[8*i+j] = Ke[8*i+j] + B[k][i] * constit_D[k][l] * B[l][j];
          }
        }
      }
    }

  } /* end quadrature */
}

static void FormMomentumRhsQ1(PetscScalar Fe[],PetscScalar coords[],PetscScalar fx[],PetscScalar fy[])
{
  PetscInt    ngp;
  PetscScalar gp_xi[GAUSS_POINTS][2];
  PetscScalar gp_weight[GAUSS_POINTS];
  PetscInt    p,i;
  PetscScalar Ni_p[NODES_PER_EL];
  PetscScalar GNi_p[NSD][NODES_PER_EL],GNx_p[NSD][NODES_PER_EL];
  PetscScalar J_p,fac;

  /* define quadrature rule */
  ConstructGaussQuadrature(&ngp,gp_xi,gp_weight);

  /* evaluate integral */
  for (p = 0; p < ngp; p++) {
    ConstructQ12D_Ni(gp_xi[p],Ni_p);
    ConstructQ12D_GNi(gp_xi[p],GNi_p);
    ConstructQ12D_GNx(GNi_p,GNx_p,coords,&J_p);
    fac = gp_weight[p]*J_p;

    for (i = 0; i < NODES_PER_EL; i++) {
      Fe[NSD*i]   += fac*Ni_p[i]*fx[p];
      Fe[NSD*i+1] += fac*Ni_p[i]*fy[p];
    }
  }
}

/*
 i,j are the element indices
 The unknown is a vector quantity.
 The s[].c is used to indicate the degree of freedom.
 */
static PetscErrorCode DMDAGetElementEqnums_u(MatStencil s_u[],PetscInt i,PetscInt j)
{
  PetscFunctionBeginUser;
  /* displacement */
  /* node 0 */
  s_u[0].i = i;s_u[0].j = j;s_u[0].c = 0;          /* Ux0 */
  s_u[1].i = i;s_u[1].j = j;s_u[1].c = 1;          /* Uy0 */

  /* node 1 */
  s_u[2].i = i;s_u[2].j = j+1;s_u[2].c = 0;        /* Ux1 */
  s_u[3].i = i;s_u[3].j = j+1;s_u[3].c = 1;        /* Uy1 */

  /* node 2 */
  s_u[4].i = i+1;s_u[4].j = j+1;s_u[4].c = 0;      /* Ux2 */
  s_u[5].i = i+1;s_u[5].j = j+1;s_u[5].c = 1;      /* Uy2 */

  /* node 3 */
  s_u[6].i = i+1;s_u[6].j = j;s_u[6].c = 0;        /* Ux3 */
  s_u[7].i = i+1;s_u[7].j = j;s_u[7].c = 1;        /* Uy3 */
  PetscFunctionReturn(0);
}

static PetscErrorCode GetElementCoords(DMDACoor2d **_coords,PetscInt ei,PetscInt ej,PetscScalar el_coords[])
{
  PetscFunctionBeginUser;
  /* get coords for the element */
  el_coords[NSD*0+0] = _coords[ej][ei].x;      el_coords[NSD*0+1] = _coords[ej][ei].y;
  el_coords[NSD*1+0] = _coords[ej+1][ei].x;    el_coords[NSD*1+1] = _coords[ej+1][ei].y;
  el_coords[NSD*2+0] = _coords[ej+1][ei+1].x;  el_coords[NSD*2+1] = _coords[ej+1][ei+1].y;
  el_coords[NSD*3+0] = _coords[ej][ei+1].x;    el_coords[NSD*3+1] = _coords[ej][ei+1].y;
  PetscFunctionReturn(0);
}

static PetscErrorCode AssembleA_Elasticity(Mat A,DM elas_da,DM properties_da,Vec properties)
{
  DM                     cda;
  Vec                    coords;
  DMDACoor2d             **_coords;
  MatStencil             u_eqn[NODES_PER_EL*U_DOFS]; /* 2 degrees of freedom */
  PetscInt               sex,sey,mx,my;
  PetscInt               ei,ej;
  PetscScalar            Ae[NODES_PER_EL*U_DOFS*NODES_PER_EL*U_DOFS];
  PetscScalar            el_coords[NODES_PER_EL*NSD];
  Vec                    local_properties;
  GaussPointCoefficients **props;
  PetscScalar            *prop_E,*prop_nu;

  PetscFunctionBeginUser;
  /* setup for coords */
  PetscCall(DMGetCoordinateDM(elas_da,&cda));
  PetscCall(DMGetCoordinatesLocal(elas_da,&coords));
  PetscCall(DMDAVecGetArray(cda,coords,&_coords));

  /* setup for coefficients */
  PetscCall(DMCreateLocalVector(properties_da,&local_properties));
  PetscCall(DMGlobalToLocalBegin(properties_da,properties,INSERT_VALUES,local_properties));
  PetscCall(DMGlobalToLocalEnd(properties_da,properties,INSERT_VALUES,local_properties));
  PetscCall(DMDAVecGetArray(properties_da,local_properties,&props));

  PetscCall(DMDAGetElementsCorners(elas_da,&sex,&sey,0));
  PetscCall(DMDAGetElementsSizes(elas_da,&mx,&my,0));
  for (ej = sey; ej < sey+my; ej++) {
    for (ei = sex; ei < sex+mx; ei++) {
      /* get coords for the element */
      GetElementCoords(_coords,ei,ej,el_coords);

      /* get coefficients for the element */
      prop_E  = props[ej][ei].E;
      prop_nu = props[ej][ei].nu;

      /* initialise element stiffness matrix */
      PetscCall(PetscMemzero(Ae,sizeof(Ae)));

      /* form element stiffness matrix */
      FormStressOperatorQ1(Ae,el_coords,prop_E,prop_nu);

      /* insert element matrix into global matrix */
      PetscCall(DMDAGetElementEqnums_u(u_eqn,ei,ej));
      PetscCall(MatSetValuesStencil(A,NODES_PER_EL*U_DOFS,u_eqn,NODES_PER_EL*U_DOFS,u_eqn,Ae,ADD_VALUES));
    }
  }
  PetscCall(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));

  PetscCall(DMDAVecRestoreArray(cda,coords,&_coords));

  PetscCall(DMDAVecRestoreArray(properties_da,local_properties,&props));
  PetscCall(VecDestroy(&local_properties));
  PetscFunctionReturn(0);
}

static PetscErrorCode DMDASetValuesLocalStencil_ADD_VALUES(ElasticityDOF **fields_F,MatStencil u_eqn[],PetscScalar Fe_u[])
{
  PetscInt n;

  PetscFunctionBeginUser;
  for (n = 0; n < 4; n++) {
    fields_F[u_eqn[2*n].j][u_eqn[2*n].i].ux_dof     = fields_F[u_eqn[2*n].j][u_eqn[2*n].i].ux_dof+Fe_u[2*n];
    fields_F[u_eqn[2*n+1].j][u_eqn[2*n+1].i].uy_dof = fields_F[u_eqn[2*n+1].j][u_eqn[2*n+1].i].uy_dof+Fe_u[2*n+1];
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode AssembleF_Elasticity(Vec F,DM elas_da,DM properties_da,Vec properties)
{
  DM                     cda;
  Vec                    coords;
  DMDACoor2d             **_coords;
  MatStencil             u_eqn[NODES_PER_EL*U_DOFS]; /* 2 degrees of freedom */
  PetscInt               sex,sey,mx,my;
  PetscInt               ei,ej;
  PetscScalar            Fe[NODES_PER_EL*U_DOFS];
  PetscScalar            el_coords[NODES_PER_EL*NSD];
  Vec                    local_properties;
  GaussPointCoefficients **props;
  PetscScalar            *prop_fx,*prop_fy;
  Vec                    local_F;
  ElasticityDOF          **ff;

  PetscFunctionBeginUser;
  /* setup for coords */
  PetscCall(DMGetCoordinateDM(elas_da,&cda));
  PetscCall(DMGetCoordinatesLocal(elas_da,&coords));
  PetscCall(DMDAVecGetArray(cda,coords,&_coords));

  /* setup for coefficients */
  PetscCall(DMGetLocalVector(properties_da,&local_properties));
  PetscCall(DMGlobalToLocalBegin(properties_da,properties,INSERT_VALUES,local_properties));
  PetscCall(DMGlobalToLocalEnd(properties_da,properties,INSERT_VALUES,local_properties));
  PetscCall(DMDAVecGetArray(properties_da,local_properties,&props));

  /* get access to the vector */
  PetscCall(DMGetLocalVector(elas_da,&local_F));
  PetscCall(VecZeroEntries(local_F));
  PetscCall(DMDAVecGetArray(elas_da,local_F,&ff));

  PetscCall(DMDAGetElementsCorners(elas_da,&sex,&sey,0));
  PetscCall(DMDAGetElementsSizes(elas_da,&mx,&my,0));
  for (ej = sey; ej < sey+my; ej++) {
    for (ei = sex; ei < sex+mx; ei++) {
      /* get coords for the element */
      GetElementCoords(_coords,ei,ej,el_coords);

      /* get coefficients for the element */
      prop_fx = props[ej][ei].fx;
      prop_fy = props[ej][ei].fy;

      /* initialise element stiffness matrix */
      PetscCall(PetscMemzero(Fe,sizeof(Fe)));

      /* form element stiffness matrix */
      FormMomentumRhsQ1(Fe,el_coords,prop_fx,prop_fy);

      /* insert element matrix into global matrix */
      PetscCall(DMDAGetElementEqnums_u(u_eqn,ei,ej));

      PetscCall(DMDASetValuesLocalStencil_ADD_VALUES(ff,u_eqn,Fe));
    }
  }

  PetscCall(DMDAVecRestoreArray(elas_da,local_F,&ff));
  PetscCall(DMLocalToGlobalBegin(elas_da,local_F,ADD_VALUES,F));
  PetscCall(DMLocalToGlobalEnd(elas_da,local_F,ADD_VALUES,F));
  PetscCall(DMRestoreLocalVector(elas_da,&local_F));

  PetscCall(DMDAVecRestoreArray(cda,coords,&_coords));

  PetscCall(DMDAVecRestoreArray(properties_da,local_properties,&props));
  PetscCall(DMRestoreLocalVector(properties_da,&local_properties));
  PetscFunctionReturn(0);
}

static PetscErrorCode solve_elasticity_2d(PetscInt mx,PetscInt my)
{
  DM                     elas_da,da_prop;
  PetscInt               u_dof,dof,stencil_width;
  Mat                    A;
  PetscInt               mxl,myl;
  DM                     prop_cda,vel_cda;
  Vec                    prop_coords,vel_coords;
  PetscInt               si,sj,nx,ny,i,j,p;
  Vec                    f,X;
  PetscInt               prop_dof,prop_stencil_width;
  Vec                    properties,l_properties;
  MatNullSpace           matnull;
  PetscReal              dx,dy;
  PetscInt               M,N;
  DMDACoor2d             **_prop_coords,**_vel_coords;
  GaussPointCoefficients **element_props;
  KSP                    ksp_E;
  PetscInt               coefficient_structure = 0;
  PetscInt               cpu_x,cpu_y,*lx = NULL,*ly = NULL;
  PetscBool              use_gp_coords = PETSC_FALSE;
  PetscBool              use_nonsymbc  = PETSC_FALSE;
  PetscBool              no_view       = PETSC_FALSE;
  PetscBool              flg;

  PetscFunctionBeginUser;
  /* Generate the da for velocity and pressure */
  /*
   We use Q1 elements for the temperature.
   FEM has a 9-point stencil (BOX) or connectivity pattern
   Num nodes in each direction is mx+1, my+1
   */
  u_dof         = U_DOFS; /* Vx, Vy - velocities */
  dof           = u_dof;
  stencil_width = 1;
  PetscCall(DMDACreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE,DMDA_STENCIL_BOX,mx+1,my+1,PETSC_DECIDE,PETSC_DECIDE,dof,stencil_width,NULL,NULL,&elas_da));

  PetscCall(DMSetMatType(elas_da,MATAIJ));
  PetscCall(DMSetFromOptions(elas_da));
  PetscCall(DMSetUp(elas_da));

  PetscCall(DMDASetFieldName(elas_da,0,"Ux"));
  PetscCall(DMDASetFieldName(elas_da,1,"Uy"));

  /* unit box [0,1] x [0,1] */
  PetscCall(DMDASetUniformCoordinates(elas_da,0.0,1.0,0.0,1.0,0.0,1.0));

  /* Generate element properties, we will assume all material properties are constant over the element */
  /* local number of elements */
  PetscCall(DMDAGetElementsSizes(elas_da,&mxl,&myl,NULL));

  /* !!! IN PARALLEL WE MUST MAKE SURE THE TWO DMDA's ALIGN !!! */
  PetscCall(DMDAGetInfo(elas_da,0,0,0,0,&cpu_x,&cpu_y,0,0,0,0,0,0,0));
  PetscCall(DMDAGetElementOwnershipRanges2d(elas_da,&lx,&ly));

  prop_dof           = (PetscInt)(sizeof(GaussPointCoefficients)/sizeof(PetscScalar)); /* gauss point setup */
  prop_stencil_width = 0;
  PetscCall(DMDACreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE,DMDA_STENCIL_BOX,mx,my,cpu_x,cpu_y,prop_dof,prop_stencil_width,lx,ly,&da_prop));
  PetscCall(DMSetFromOptions(da_prop));
  PetscCall(DMSetUp(da_prop));

  PetscCall(PetscFree(lx));
  PetscCall(PetscFree(ly));

  /* define centroid positions */
  PetscCall(DMDAGetInfo(da_prop,0,&M,&N,0,0,0,0,0,0,0,0,0,0));
  dx   = 1.0/((PetscReal)(M));
  dy   = 1.0/((PetscReal)(N));

  PetscCall(DMDASetUniformCoordinates(da_prop,0.0+0.5*dx,1.0-0.5*dx,0.0+0.5*dy,1.0-0.5*dy,0.0,1.0));

  /* define coefficients */
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-c_str",&coefficient_structure,NULL));

  PetscCall(DMCreateGlobalVector(da_prop,&properties));
  PetscCall(DMCreateLocalVector(da_prop,&l_properties));
  PetscCall(DMDAVecGetArray(da_prop,l_properties,&element_props));

  PetscCall(DMGetCoordinateDM(da_prop,&prop_cda));
  PetscCall(DMGetCoordinatesLocal(da_prop,&prop_coords));
  PetscCall(DMDAVecGetArray(prop_cda,prop_coords,&_prop_coords));

  PetscCall(DMDAGetGhostCorners(prop_cda,&si,&sj,0,&nx,&ny,0));

  PetscCall(DMGetCoordinateDM(elas_da,&vel_cda));
  PetscCall(DMGetCoordinatesLocal(elas_da,&vel_coords));
  PetscCall(DMDAVecGetArray(vel_cda,vel_coords,&_vel_coords));

  /* interpolate the coordinates */
  for (j = sj; j < sj+ny; j++) {
    for (i = si; i < si+nx; i++) {
      PetscInt    ngp;
      PetscScalar gp_xi[GAUSS_POINTS][2],gp_weight[GAUSS_POINTS];
      PetscScalar el_coords[8];

      PetscCall(GetElementCoords(_vel_coords,i,j,el_coords));
      ConstructGaussQuadrature(&ngp,gp_xi,gp_weight);

      for (p = 0; p < GAUSS_POINTS; p++) {
        PetscScalar gp_x,gp_y;
        PetscInt    n;
        PetscScalar xi_p[2],Ni_p[4];

        xi_p[0] = gp_xi[p][0];
        xi_p[1] = gp_xi[p][1];
        ConstructQ12D_Ni(xi_p,Ni_p);

        gp_x = 0.0;
        gp_y = 0.0;
        for (n = 0; n < NODES_PER_EL; n++) {
          gp_x = gp_x+Ni_p[n]*el_coords[2*n];
          gp_y = gp_y+Ni_p[n]*el_coords[2*n+1];
        }
        element_props[j][i].gp_coords[2*p]   = gp_x;
        element_props[j][i].gp_coords[2*p+1] = gp_y;
      }
    }
  }

  /* define the coefficients */
  PetscCall(PetscOptionsGetBool(NULL,NULL,"-use_gp_coords",&use_gp_coords,&flg));

  for (j = sj; j < sj+ny; j++) {
    for (i = si; i < si+nx; i++) {
      PetscScalar              centroid_x = _prop_coords[j][i].x; /* centroids of cell */
      PetscScalar              centroid_y = _prop_coords[j][i].y;
      PETSC_UNUSED PetscScalar coord_x,coord_y;

      if (coefficient_structure == 0) { /* isotropic */
        PetscScalar opts_E,opts_nu;

        opts_E  = 1.0;
        opts_nu = 0.33;
        PetscCall(PetscOptionsGetScalar(NULL,NULL,"-iso_E",&opts_E,&flg));
        PetscCall(PetscOptionsGetScalar(NULL,NULL,"-iso_nu",&opts_nu,&flg));

        for (p = 0; p < GAUSS_POINTS; p++) {
          element_props[j][i].E[p]  = opts_E;
          element_props[j][i].nu[p] = opts_nu;

          element_props[j][i].fx[p] = 0.0;
          element_props[j][i].fy[p] = 0.0;
        }
      } else if (coefficient_structure == 1) { /* step */
        PetscScalar opts_E0,opts_nu0,opts_xc;
        PetscScalar opts_E1,opts_nu1;

        opts_E0  = opts_E1  = 1.0;
        opts_nu0 = opts_nu1 = 0.333;
        opts_xc  = 0.5;
        PetscCall(PetscOptionsGetScalar(NULL,NULL,"-step_E0",&opts_E0,&flg));
        PetscCall(PetscOptionsGetScalar(NULL,NULL,"-step_nu0",&opts_nu0,&flg));
        PetscCall(PetscOptionsGetScalar(NULL,NULL,"-step_E1",&opts_E1,&flg));
        PetscCall(PetscOptionsGetScalar(NULL,NULL,"-step_nu1",&opts_nu1,&flg));
        PetscCall(PetscOptionsGetScalar(NULL,NULL,"-step_xc",&opts_xc,&flg));

        for (p = 0; p < GAUSS_POINTS; p++) {
          coord_x = centroid_x;
          coord_y = centroid_y;
          if (use_gp_coords) {
            coord_x = element_props[j][i].gp_coords[2*p];
            coord_y = element_props[j][i].gp_coords[2*p+1];
          }

          element_props[j][i].E[p]  = opts_E0;
          element_props[j][i].nu[p] = opts_nu0;
          if (PetscRealPart(coord_x) > PetscRealPart(opts_xc)) {
            element_props[j][i].E[p]  = opts_E1;
            element_props[j][i].nu[p] = opts_nu1;
          }

          element_props[j][i].fx[p] = 0.0;
          element_props[j][i].fy[p] = 0.0;
        }
      } else if (coefficient_structure == 2) { /* brick */
        PetscReal values_E[10];
        PetscReal values_nu[10];
        PetscInt  nbricks,maxnbricks;
        PetscInt  index,span;
        PetscInt  jj;

        flg        = PETSC_FALSE;
        maxnbricks = 10;
        PetscCall(PetscOptionsGetRealArray(NULL,NULL, "-brick_E",values_E,&maxnbricks,&flg));
        nbricks    = maxnbricks;
        PetscCheck(flg,PETSC_COMM_SELF,PETSC_ERR_USER,"User must supply a list of E values for each brick");

        flg        = PETSC_FALSE;
        maxnbricks = 10;
        PetscCall(PetscOptionsGetRealArray(NULL,NULL, "-brick_nu",values_nu,&maxnbricks,&flg));
        PetscCheck(flg,PETSC_COMM_SELF,PETSC_ERR_USER,"User must supply a list of nu values for each brick");
        PetscCheck(maxnbricks == nbricks,PETSC_COMM_SELF,PETSC_ERR_USER,"User must supply equal numbers of values for E and nu");

        span = 1;
        PetscCall(PetscOptionsGetInt(NULL,NULL,"-brick_span",&span,&flg));

        /* cycle through the indices so that no two material properties are repeated in lines of x or y */
        jj    = (j/span)%nbricks;
        index = (jj+i/span)%nbricks;
        /*printf("j=%d: index = %d \n", j,index); */

        for (p = 0; p < GAUSS_POINTS; p++) {
          element_props[j][i].E[p]  = values_E[index];
          element_props[j][i].nu[p] = values_nu[index];
        }
      } else if (coefficient_structure == 3) { /* sponge */
        PetscScalar opts_E0,opts_nu0;
        PetscScalar opts_E1,opts_nu1;
        PetscInt    opts_t,opts_w;
        PetscInt    ii,jj,ci,cj;

        opts_E0  = opts_E1  = 1.0;
        opts_nu0 = opts_nu1 = 0.333;
        PetscCall(PetscOptionsGetScalar(NULL,NULL,"-sponge_E0",&opts_E0,&flg));
        PetscCall(PetscOptionsGetScalar(NULL,NULL,"-sponge_nu0",&opts_nu0,&flg));
        PetscCall(PetscOptionsGetScalar(NULL,NULL,"-sponge_E1",&opts_E1,&flg));
        PetscCall(PetscOptionsGetScalar(NULL,NULL,"-sponge_nu1",&opts_nu1,&flg));

        opts_t = opts_w = 1;
        PetscCall(PetscOptionsGetInt(NULL,NULL,"-sponge_t",&opts_t,&flg));
        PetscCall(PetscOptionsGetInt(NULL,NULL,"-sponge_w",&opts_w,&flg));

        ii = (i)/(opts_t+opts_w+opts_t);
        jj = (j)/(opts_t+opts_w+opts_t);

        ci = i - ii*(opts_t+opts_w+opts_t);
        cj = j - jj*(opts_t+opts_w+opts_t);

        for (p = 0; p < GAUSS_POINTS; p++) {
          element_props[j][i].E[p]  = opts_E0;
          element_props[j][i].nu[p] = opts_nu0;
        }
        if ((ci >= opts_t) && (ci < opts_t+opts_w)) {
          if ((cj >= opts_t) && (cj < opts_t+opts_w)) {
            for (p = 0; p < GAUSS_POINTS; p++) {
              element_props[j][i].E[p]  = opts_E1;
              element_props[j][i].nu[p] = opts_nu1;
            }
          }
        }
      } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"Unknown coefficient_structure");
    }
  }
  PetscCall(DMDAVecRestoreArray(prop_cda,prop_coords,&_prop_coords));

  PetscCall(DMDAVecRestoreArray(vel_cda,vel_coords,&_vel_coords));

  PetscCall(DMDAVecRestoreArray(da_prop,l_properties,&element_props));
  PetscCall(DMLocalToGlobalBegin(da_prop,l_properties,ADD_VALUES,properties));
  PetscCall(DMLocalToGlobalEnd(da_prop,l_properties,ADD_VALUES,properties));

  PetscCall(PetscOptionsGetBool(NULL,NULL,"-no_view",&no_view,NULL));
  if (!no_view) {
    PetscCall(DMDAViewCoefficientsGnuplot2d(da_prop,properties,"Coeffcients for elasticity eqn.","properties"));
    PetscCall(DMDACoordViewGnuplot2d(elas_da,"mesh"));
  }

  /* Generate a matrix with the correct non-zero pattern of type AIJ. This will work in parallel and serial */
  PetscCall(DMCreateMatrix(elas_da,&A));
  PetscCall(DMGetCoordinates(elas_da,&vel_coords));
  PetscCall(MatNullSpaceCreateRigidBody(vel_coords,&matnull));
  PetscCall(MatSetNearNullSpace(A,matnull));
  PetscCall(MatNullSpaceDestroy(&matnull));
  PetscCall(MatCreateVecs(A,&f,&X));

  /* assemble A11 */
  PetscCall(MatZeroEntries(A));
  PetscCall(VecZeroEntries(f));

  PetscCall(AssembleA_Elasticity(A,elas_da,da_prop,properties));
  /* build force vector */
  PetscCall(AssembleF_Elasticity(f,elas_da,da_prop,properties));

  PetscCall(KSPCreate(PETSC_COMM_WORLD,&ksp_E));
  PetscCall(KSPSetOptionsPrefix(ksp_E,"elas_"));  /* elasticity */

  PetscCall(PetscOptionsGetBool(NULL,NULL,"-use_nonsymbc",&use_nonsymbc,&flg));
  /* solve */
  if (!use_nonsymbc) {
    Mat        AA;
    Vec        ff,XX;
    IS         is;
    VecScatter scat;

    PetscCall(DMDABCApplySymmetricCompression(elas_da,A,f,&is,&AA,&ff));
    PetscCall(VecDuplicate(ff,&XX));

    PetscCall(KSPSetOperators(ksp_E,AA,AA));
    PetscCall(KSPSetFromOptions(ksp_E));

    PetscCall(KSPSolve(ksp_E,ff,XX));

    /* push XX back into X */
    PetscCall(DMDABCApplyCompression(elas_da,NULL,X));

    PetscCall(VecScatterCreate(XX,NULL,X,is,&scat));
    PetscCall(VecScatterBegin(scat,XX,X,INSERT_VALUES,SCATTER_FORWARD));
    PetscCall(VecScatterEnd(scat,XX,X,INSERT_VALUES,SCATTER_FORWARD));
    PetscCall(VecScatterDestroy(&scat));

    PetscCall(MatDestroy(&AA));
    PetscCall(VecDestroy(&ff));
    PetscCall(VecDestroy(&XX));
    PetscCall(ISDestroy(&is));
  } else {
    PetscCall(DMDABCApplyCompression(elas_da,A,f));

    PetscCall(KSPSetOperators(ksp_E,A,A));
    PetscCall(KSPSetFromOptions(ksp_E));

    PetscCall(KSPSolve(ksp_E,f,X));
  }

  if (!no_view) PetscCall(DMDAViewGnuplot2d(elas_da,X,"Displacement solution for elasticity eqn.","X"));
  PetscCall(KSPDestroy(&ksp_E));

  PetscCall(VecDestroy(&X));
  PetscCall(VecDestroy(&f));
  PetscCall(MatDestroy(&A));

  PetscCall(DMDestroy(&elas_da));
  PetscCall(DMDestroy(&da_prop));

  PetscCall(VecDestroy(&properties));
  PetscCall(VecDestroy(&l_properties));
  PetscFunctionReturn(0);
}

int main(int argc,char **args)
{
  PetscInt       mx,my;

  PetscCall(PetscInitialize(&argc,&args,(char*)0,help));
  mx   = my = 10;
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-mx",&mx,NULL));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-my",&my,NULL));
  PetscCall(solve_elasticity_2d(mx,my));
  PetscCall(PetscFinalize());
  return 0;
}

/* -------------------------- helpers for boundary conditions -------------------------------- */

static PetscErrorCode BCApply_EAST(DM da,PetscInt d_idx,PetscScalar bc_val,Mat A,Vec b)
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
  /* enforce bc's */
  PetscCall(DMGetLocalToGlobalMapping(da,&ltogm));
  PetscCall(ISLocalToGlobalMappingGetIndices(ltogm,&g_idx));

  PetscCall(DMGetCoordinateDM(da,&cda));
  PetscCall(DMGetCoordinatesLocal(da,&coords));
  PetscCall(DMDAVecGetArray(cda,coords,&_coords));
  PetscCall(DMDAGetGhostCorners(cda,&si,&sj,0,&nx,&ny,0));
  PetscCall(DMDAGetInfo(da,0,&M,&N,0,0,0,0,&n_dofs,0,0,0,0,0));

  /* --- */

  PetscCall(PetscMalloc1(ny*n_dofs,&bc_global_ids));
  PetscCall(PetscMalloc1(ny*n_dofs,&bc_vals));

  /* init the entries to -1 so VecSetValues will ignore them */
  for (i = 0; i < ny*n_dofs; i++) bc_global_ids[i] = -1;

  i = nx-1;
  for (j = 0; j < ny; j++) {
    PetscInt                 local_id;
    PETSC_UNUSED PetscScalar coordx,coordy;

    local_id = i+j*nx;

    bc_global_ids[j] = g_idx[n_dofs*local_id+d_idx];

    coordx = _coords[j+sj][i+si].x;
    coordy = _coords[j+sj][i+si].y;

    bc_vals[j] =  bc_val;
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
    PetscCall(MatZeroRows(A,nbcs,bc_global_ids,1.0,0,0));
  }

  PetscCall(PetscFree(bc_vals));
  PetscCall(PetscFree(bc_global_ids));

  PetscCall(DMDAVecRestoreArray(cda,coords,&_coords));
  PetscFunctionReturn(0);
}

static PetscErrorCode BCApply_WEST(DM da,PetscInt d_idx,PetscScalar bc_val,Mat A,Vec b)
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
  /* enforce bc's */
  PetscCall(DMGetLocalToGlobalMapping(da,&ltogm));
  PetscCall(ISLocalToGlobalMappingGetIndices(ltogm,&g_idx));

  PetscCall(DMGetCoordinateDM(da,&cda));
  PetscCall(DMGetCoordinatesLocal(da,&coords));
  PetscCall(DMDAVecGetArray(cda,coords,&_coords));
  PetscCall(DMDAGetGhostCorners(cda,&si,&sj,0,&nx,&ny,0));
  PetscCall(DMDAGetInfo(da,0,&M,&N,0,0,0,0,&n_dofs,0,0,0,0,0));

  /* --- */

  PetscCall(PetscMalloc1(ny*n_dofs,&bc_global_ids));
  PetscCall(PetscMalloc1(ny*n_dofs,&bc_vals));

  /* init the entries to -1 so VecSetValues will ignore them */
  for (i = 0; i < ny*n_dofs; i++) bc_global_ids[i] = -1;

  i = 0;
  for (j = 0; j < ny; j++) {
    PetscInt                 local_id;
    PETSC_UNUSED PetscScalar coordx,coordy;

    local_id = i+j*nx;

    bc_global_ids[j] = g_idx[n_dofs*local_id+d_idx];

    coordx = _coords[j+sj][i+si].x;
    coordy = _coords[j+sj][i+si].y;

    bc_vals[j] =  bc_val;
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
    PetscCall(MatZeroRows(A,nbcs,bc_global_ids,1.0,0,0));
  }

  PetscCall(PetscFree(bc_vals));
  PetscCall(PetscFree(bc_global_ids));

  PetscCall(DMDAVecRestoreArray(cda,coords,&_coords));
  PetscFunctionReturn(0);
}

static PetscErrorCode DMDABCApplyCompression(DM elas_da,Mat A,Vec f)
{
  PetscFunctionBeginUser;
  PetscCall(BCApply_EAST(elas_da,0,-1.0,A,f));
  PetscCall(BCApply_EAST(elas_da,1, 0.0,A,f));
  PetscCall(BCApply_WEST(elas_da,0,1.0,A,f));
  PetscCall(BCApply_WEST(elas_da,1,0.0,A,f));
  PetscFunctionReturn(0);
}

static PetscErrorCode Orthogonalize(PetscInt n,Vec *vecs)
{
  PetscInt       i,j;
  PetscScalar    dot;

  PetscFunctionBegin;
  for (i=0; i<n; i++) {
  PetscCall(VecNormalize(vecs[i],NULL));
     for (j=i+1; j<n; j++) {
       PetscCall(VecDot(vecs[i],vecs[j],&dot));
       PetscCall(VecAXPY(vecs[j],-dot,vecs[i]));
     }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode DMDABCApplySymmetricCompression(DM elas_da,Mat A,Vec f,IS *dofs,Mat *AA,Vec *ff)
{
  PetscInt       start,end,m;
  PetscInt       *unconstrained;
  PetscInt       cnt,i;
  Vec            x;
  PetscScalar    *_x;
  IS             is;
  VecScatter     scat;

  PetscFunctionBeginUser;
  /* push bc's into f and A */
  PetscCall(VecDuplicate(f,&x));
  PetscCall(BCApply_EAST(elas_da,0,-1.0,A,x));
  PetscCall(BCApply_EAST(elas_da,1, 0.0,A,x));
  PetscCall(BCApply_WEST(elas_da,0,1.0,A,x));
  PetscCall(BCApply_WEST(elas_da,1,0.0,A,x));

  /* define which dofs are not constrained */
  PetscCall(VecGetLocalSize(x,&m));
  PetscCall(PetscMalloc1(m,&unconstrained));
  PetscCall(VecGetOwnershipRange(x,&start,&end));
  PetscCall(VecGetArray(x,&_x));
  cnt  = 0;
  for (i = 0; i < m; i+=2) {
    PetscReal val1,val2;

    val1 = PetscRealPart(_x[i]);
    val2 = PetscRealPart(_x[i+1]);
    if (PetscAbs(val1) < 0.1 && PetscAbs(val2) < 0.1) {
      unconstrained[cnt] = start + i;
      cnt++;
      unconstrained[cnt] = start + i + 1;
      cnt++;
    }
  }
  PetscCall(VecRestoreArray(x,&_x));

  PetscCall(ISCreateGeneral(PETSC_COMM_WORLD,cnt,unconstrained,PETSC_COPY_VALUES,&is));
  PetscCall(PetscFree(unconstrained));
  PetscCall(ISSetBlockSize(is,2));

  /* define correction for dirichlet in the rhs */
  PetscCall(MatMult(A,x,f));
  PetscCall(VecScale(f,-1.0));

  /* get new matrix */
  PetscCall(MatCreateSubMatrix(A,is,is,MAT_INITIAL_MATRIX,AA));
  /* get new vector */
  PetscCall(MatCreateVecs(*AA,NULL,ff));

  PetscCall(VecScatterCreate(f,is,*ff,NULL,&scat));
  PetscCall(VecScatterBegin(scat,f,*ff,INSERT_VALUES,SCATTER_FORWARD));
  PetscCall(VecScatterEnd(scat,f,*ff,INSERT_VALUES,SCATTER_FORWARD));

  {                             /* Constrain near-null space */
    PetscInt     nvecs;
    const        Vec *vecs;
    Vec          *uvecs;
    PetscBool    has_const;
    MatNullSpace mnull,unull;

    PetscCall(MatGetNearNullSpace(A,&mnull));
    PetscCall(MatNullSpaceGetVecs(mnull,&has_const,&nvecs,&vecs));
    PetscCall(VecDuplicateVecs(*ff,nvecs,&uvecs));
    for (i=0; i<nvecs; i++) {
      PetscCall(VecScatterBegin(scat,vecs[i],uvecs[i],INSERT_VALUES,SCATTER_FORWARD));
      PetscCall(VecScatterEnd(scat,vecs[i],uvecs[i],INSERT_VALUES,SCATTER_FORWARD));
    }
    PetscCall(Orthogonalize(nvecs,uvecs));
    PetscCall(MatNullSpaceCreate(PetscObjectComm((PetscObject)A),PETSC_FALSE,nvecs,uvecs,&unull));
    PetscCall(MatSetNearNullSpace(*AA,unull));
    PetscCall(MatNullSpaceDestroy(&unull));
    PetscCall(VecDestroyVecs(nvecs,&uvecs));
  }

  PetscCall(VecScatterDestroy(&scat));

  *dofs = is;
  PetscCall(VecDestroy(&x));
  PetscFunctionReturn(0);
}

/*TEST

   build:
      requires: !complex !single

   test:
      args: -mx 20 -my 30 -elas_ksp_monitor_short -no_view -c_str 3 -sponge_E0 1 -sponge_E1 1000 -sponge_nu0 0.4 -sponge_nu1 0.2 -sponge_t 1 -sponge_w 8 -elas_ksp_rtol 5e-3 -elas_ksp_view
      output_file: output/ex49_1.out

   test:
      suffix: 2
      nsize: 4
      args: -mx 20 -my 30 -elas_ksp_monitor_short -no_view -c_str 3 -sponge_E0 1 -sponge_E1 1000 -sponge_nu0 0.4 -sponge_nu1 0.2 -sponge_t 1 -sponge_w 8 -elas_ksp_type gcr -elas_pc_type asm -elas_sub_pc_type lu -elas_ksp_rtol 5e-3

   test:
      suffix: 3
      nsize: 4
      args: -mx 20 -my 30 -elas_ksp_monitor_short -no_view -c_str 2 -brick_E 1,10,1000,100 -brick_nu 0.4,0.2,0.3,0.1 -brick_span 3 -elas_pc_type asm -elas_sub_pc_type lu -elas_ksp_rtol 5e-3

   test:
      suffix: 4
      nsize: 4
      args: -elas_ksp_monitor_short -elas_ksp_converged_reason -elas_ksp_type cg -elas_ksp_norm_type unpreconditioned -mx 40 -my 40 -c_str 2 -brick_E 1,1e-6,1e-2 -brick_nu .3,.2,.4 -brick_span 8 -elas_mg_levels_ksp_type chebyshev -elas_pc_type ml -elas_mg_levels_ksp_chebyshev_esteig 0,0.2,0,1.1 -elas_mg_levels_pc_type pbjacobi -elas_mg_levels_ksp_max_it 3 -use_nonsymbc -elas_pc_ml_nullspace user
      requires: ml

   test:
      suffix: 5
      nsize: 3
      args: -elas_ksp_monitor_short -elas_ksp_converged_reason -elas_ksp_type cg -elas_ksp_norm_type natural -mx 22 -my 22 -c_str 2 -brick_E 1,1e-6,1e-2 -brick_nu .3,.2,.4 -brick_span 8 -elas_pc_type gamg -elas_mg_levels_ksp_type chebyshev -elas_mg_levels_ksp_max_it 1 -elas_mg_levels_ksp_chebyshev_esteig 0.2,1.1 -elas_mg_levels_pc_type jacobi

   test:
      suffix: 6
      nsize: 4
      args: -mx 20 -my 30 -elas_ksp_monitor_short -no_view -c_str 3 -sponge_E0 1 -sponge_E1 1000 -sponge_nu0 0.4 -sponge_nu1 0.2 -sponge_t 1 -sponge_w 8 -elas_ksp_type pipegcr -elas_pc_type asm -elas_sub_pc_type lu

   test:
      suffix: 7
      nsize: 4
      args: -mx 20 -my 30 -elas_ksp_monitor_short -no_view -c_str 3 -sponge_E0 1 -sponge_E1 1000 -sponge_nu0 0.4 -sponge_nu1 0.2 -sponge_t 1 -sponge_w 8 -elas_ksp_type pipegcr -elas_pc_type asm -elas_sub_pc_type ksp -elas_sub_ksp_ksp_type cg -elas_sub_ksp_ksp_max_it 15

   test:
      suffix: 8
      nsize: 4
      args: -mx 20 -my 30 -elas_ksp_monitor_short -no_view -c_str 3 -sponge_E0 1 -sponge_E1 1000 -sponge_nu0 0.4 -sponge_nu1 0.2 -sponge_t 1 -sponge_w 8 -elas_ksp_type pipefgmres -elas_pc_type asm -elas_sub_pc_type ksp -elas_sub_ksp_ksp_type cg -elas_sub_ksp_ksp_max_it 15

   test:
      suffix: hypre_nullspace
      requires: hypre !defined(PETSC_HAVE_HYPRE_DEVICE)
      args: -elas_ksp_monitor_short -elas_ksp_converged_reason -elas_ksp_type cg -elas_ksp_norm_type natural -mx 22 -my 22 -c_str 2 -brick_E 1,1e-6,1e-2 -brick_nu .3,.2,.4 -brick_span 8 -elas_pc_type hypre -elas_pc_hypre_boomeramg_nodal_coarsen 6 -elas_pc_hypre_boomeramg_vec_interp_variant 3 -elas_pc_hypre_boomeramg_interp_type ext+i -elas_ksp_view

   test:
      nsize: 4
      suffix: bddc
      args: -elas_ksp_monitor_short -no_view -elas_ksp_converged_reason -elas_ksp_type cg -elas_ksp_norm_type natural -mx 22 -my 22 -dm_mat_type is -elas_pc_type bddc -elas_pc_bddc_monolithic

   test:
      nsize: 4
      suffix: bddc_unsym
      args: -elas_ksp_monitor_short -no_view -elas_ksp_converged_reason -elas_ksp_type cg -elas_ksp_norm_type natural -mx 22 -my 22 -dm_mat_type is -elas_pc_type bddc -elas_pc_bddc_monolithic -use_nonsymbc -elas_pc_bddc_symmetric 0

   test:
      nsize: 4
      suffix: bddc_unsym_deluxe
      args: -elas_ksp_monitor_short -no_view -elas_ksp_converged_reason -elas_ksp_type cg -elas_ksp_norm_type natural -mx 22 -my 22 -dm_mat_type is -elas_pc_type bddc -elas_pc_bddc_monolithic -use_nonsymbc -elas_pc_bddc_symmetric 0 -elas_pc_bddc_use_deluxe_scaling -elas_sub_schurs_symmetric 0

   test:
      nsize: 4
      suffix: fetidp_unsym_deluxe
      args: -elas_ksp_monitor_short -no_view -elas_ksp_converged_reason -elas_ksp_type fetidp -elas_fetidp_ksp_type cg -elas_ksp_norm_type natural -mx 22 -my 22 -dm_mat_type is -elas_fetidp_bddc_pc_bddc_monolithic -use_nonsymbc -elas_fetidp_bddc_pc_bddc_use_deluxe_scaling -elas_fetidp_bddc_sub_schurs_symmetric 0 -elas_fetidp_bddc_pc_bddc_deluxe_singlemat

   test:
      nsize: 4
      suffix: bddc_layerjump
      args: -mx 40 -my 40 -elas_ksp_monitor_short -no_view -c_str 3 -sponge_E0 1 -sponge_E1 1000 -sponge_nu0 0.4 -sponge_nu1 0.2 -sponge_t 1 -sponge_w 8 -elas_ksp_type cg -elas_pc_type bddc -elas_pc_bddc_monolithic -dm_mat_type is -elas_ksp_norm_type natural

   test:
      nsize: 4
      suffix: bddc_subdomainjump
      args: -mx 40 -my 40 -elas_ksp_monitor_short -no_view -c_str 2 -brick_E 1,1000 -brick_nu 0.4,0.2 -brick_span 20  -elas_ksp_type cg -elas_pc_type bddc -elas_pc_bddc_monolithic -dm_mat_type is -elas_pc_is_use_stiffness_scaling -elas_ksp_norm_type natural

   test:
      nsize: 9
      suffix: bddc_subdomainjump_deluxe
      args: -mx 30 -my 30 -elas_ksp_monitor_short -no_view -c_str 2 -brick_E 1,1000 -brick_nu 0.4,0.2 -brick_span 10  -elas_ksp_type cg -elas_pc_type bddc -elas_pc_bddc_monolithic -dm_mat_type is -elas_pc_bddc_use_deluxe_scaling -elas_ksp_norm_type natural -elas_pc_bddc_schur_layers 1
TEST*/
