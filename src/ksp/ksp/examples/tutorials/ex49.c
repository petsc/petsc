static char help[] =
  "   Solves the compressible plane strain elasticity equations in 2d on the unit domain using Q1 finite elements. \n\
   Material properties E (Youngs moduls) and nu (Poisson ratio) may vary as a function of space. \n\
   The model utilisse boundary conditions which produce compression in the x direction. \n\
Options: \n\
     -mx : number elements in x-direciton \n\
     -my : number elements in y-direciton \n\
     -c_str : indicates the structure of the coefficients to use. \n\
          -c_str 0 => Setup for an isotropic material with constant coefficients. \n\
                         Parameters: \n\
                             -iso_E  : Youngs modulus \n\
                             -iso_nu : Poisson ratio \n\
          -c_str 1 => Setup for a step function in the material properties in x. \n\
                         Parameters: \n\
                              -step_E0  : Youngs modulus to the left of the step \n\
                              -step_nu0 : Poisson ratio to the left of the step \n\
                              -step_E1  : Youngs modulus to the right of the step \n\
                              -step_n1  : Poisson ratio to the right of the step \n\
                              -step_xc  : x coordinate of the step \n\
          -c_str 2 => Setup for a checkerboard material with alternating properties. \n\
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
                              -brick_E    : a comma seperated list of Young's modulii \n\
                              -brick_nu   : a comma seperated list of Poisson ratio's  \n\
                              -brick_span : the number of elements in x and y each brick will span \n\
          -c_str 3 => Setup for a sponge-like material with alternating properties. \n\
                      Repeats the following pattern throughout the domain \n\
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
                              -sponge_E0  : Youngs moduls of the surrounding material \n\
                              -sponge_E1  : Youngs moduls of the inclusio \n\
                              -sponge_nu0 : Poisson ratio of the surrounding material \n\
                              -sponge_nu1 : Poisson ratio of the inclusio \n\
                              -sponge_t   : the number of elements defining the border around each inclusion \n\
                              -sponge_w   : the number of elements in x and y each inclusion will span\n\
     -use_gp_coords : Evaluate the Youngs modulus, Poisson ratio and the body force at the global coordinates of the quadrature points.\n\
     By default, E, nu and the body force are evaulated at the element center and applied as a constant over the entire element.\n\
     -use_nonsymbc : Option to use non-symmetric boundary condition imposition. This choice will use less memory.";

/* Contributed by Dave May */

#include <petscksp.h>
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

 D = E/( (1+nu)(1-2nu) ) * [ 1-nu   nu        0     ]
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
    PetscScalar cx = coords[ 2*i+0 ];
    PetscScalar cy = coords[ 2*i+1 ];

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

  if (det_J != NULL) {
    *det_J = J;
  }
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


/* procs to the left claim the ghost node as their element */
#undef __FUNCT__
#define __FUNCT__ "DMDAGetLocalElementSize"
static PetscErrorCode DMDAGetLocalElementSize(DM da,PetscInt *mxl,PetscInt *myl,PetscInt *mzl)
{
  PetscErrorCode ierr;
  PetscInt m,n,p,M,N,P;
  PetscInt sx,sy,sz;

  PetscFunctionBegin;
  ierr = DMDAGetInfo(da,0,&M,&N,&P,0,0,0,0,0,0,0,0,0);CHKERRQ(ierr);
  ierr = DMDAGetCorners(da,&sx,&sy,&sz,&m,&n,&p);CHKERRQ(ierr);

  if (mxl != PETSC_NULL) {
    *mxl = m;
    if ((sx+m) == M) {  /* last proc */
      *mxl = m-1;
    }
  }
  if (myl != PETSC_NULL) {
    *myl = n;
    if ((sy+n) == N) {  /* last proc */
      *myl = n-1;
    }
  }
  if (mzl != PETSC_NULL) {
    *mzl = p;
    if ((sz+p) == P) {  /* last proc */
      *mzl = p-1;
    }
  }

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMDAGetElementCorners"
static PetscErrorCode DMDAGetElementCorners(DM da,
                                          PetscInt *sx,PetscInt *sy,PetscInt *sz,
                                          PetscInt *mx,PetscInt *my,PetscInt *mz)
{
  PetscErrorCode ierr;
  PetscInt si,sj,sk;

  PetscFunctionBegin;
  ierr = DMDAGetGhostCorners(da,&si,&sj,&sk,0,0,0);CHKERRQ(ierr);

  if (sx != PETSC_NULL) {
    *sx = si;
    if (si != 0) {
      *sx = si+1;
    }
  }
  if (sy != PETSC_NULL) {
    *sy = sj;
    if (sj != 0) {
      *sy = sj+1;
    }
  }

  if (sk != PETSC_NULL) {
    *sz = sk;
    if (sk != 0) {
      *sz = sk+1;
    }
  }

  ierr = DMDAGetLocalElementSize(da,mx,my,mz);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMDAGetElementOwnershipRanges2d"
static PetscErrorCode DMDAGetElementOwnershipRanges2d(DM da,PetscInt **_lx,PetscInt **_ly)
{
  PetscErrorCode ierr;
  PetscMPIInt    rank;
  PetscInt       proc_I,proc_J;
  PetscInt       cpu_x,cpu_y;
  PetscInt       local_mx,local_my;
  Vec            vlx,vly;
  PetscInt       *LX,*LY,i;
  PetscScalar    *_a;
  Vec            V_SEQ;
  VecScatter     ctx;

  PetscFunctionBegin;
  MPI_Comm_rank(PETSC_COMM_WORLD,&rank);

  DMDAGetInfo(da,0,0,0,0,&cpu_x,&cpu_y,0,0,0,0,0,0,0);

  proc_J = rank/cpu_x;
  proc_I = rank-cpu_x*proc_J;

  ierr = PetscMalloc(sizeof(PetscInt)*cpu_x,&LX);CHKERRQ(ierr);
  ierr = PetscMalloc(sizeof(PetscInt)*cpu_y,&LY);CHKERRQ(ierr);

  ierr = DMDAGetLocalElementSize(da,&local_mx,&local_my,PETSC_NULL);CHKERRQ(ierr);
  ierr = VecCreate(PETSC_COMM_WORLD,&vlx);CHKERRQ(ierr);
  ierr = VecSetSizes(vlx,PETSC_DECIDE,cpu_x);CHKERRQ(ierr);
  ierr = VecSetFromOptions(vlx);CHKERRQ(ierr);

  ierr = VecCreate(PETSC_COMM_WORLD,&vly);CHKERRQ(ierr);
  ierr = VecSetSizes(vly,PETSC_DECIDE,cpu_y);CHKERRQ(ierr);
  ierr = VecSetFromOptions(vly);CHKERRQ(ierr);

  ierr = VecSetValue(vlx,proc_I,(PetscScalar)(local_mx+1.0e-9),INSERT_VALUES);CHKERRQ(ierr);
  ierr = VecSetValue(vly,proc_J,(PetscScalar)(local_my+1.0e-9),INSERT_VALUES);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(vlx);VecAssemblyEnd(vlx);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(vly);VecAssemblyEnd(vly);CHKERRQ(ierr);



  ierr = VecScatterCreateToAll(vlx,&ctx,&V_SEQ);CHKERRQ(ierr);
  ierr = VecScatterBegin(ctx,vlx,V_SEQ,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(ctx,vlx,V_SEQ,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecGetArray(V_SEQ,&_a);CHKERRQ(ierr);
  for (i = 0; i < cpu_x; i++) {
    LX[i] = (PetscInt)PetscRealPart(_a[i]);
  }
  ierr = VecRestoreArray(V_SEQ,&_a);CHKERRQ(ierr);
  ierr = VecScatterDestroy(&ctx);CHKERRQ(ierr);
  ierr = VecDestroy(&V_SEQ);CHKERRQ(ierr);

  ierr = VecScatterCreateToAll(vly,&ctx,&V_SEQ);CHKERRQ(ierr);
  ierr = VecScatterBegin(ctx,vly,V_SEQ,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(ctx,vly,V_SEQ,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecGetArray(V_SEQ,&_a);CHKERRQ(ierr);
  for (i = 0; i < cpu_y; i++) {
    LY[i] = (PetscInt)PetscRealPart(_a[i]);
  }
  ierr = VecRestoreArray(V_SEQ,&_a);CHKERRQ(ierr);
  ierr = VecScatterDestroy(&ctx);CHKERRQ(ierr);
  ierr = VecDestroy(&V_SEQ);CHKERRQ(ierr);

  *_lx = LX;
  *_ly = LY;

  ierr = VecDestroy(&vlx);CHKERRQ(ierr);
  ierr = VecDestroy(&vly);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMDACoordViewGnuplot2d"
static PetscErrorCode DMDACoordViewGnuplot2d(DM da,const char prefix[])
{
  DM             cda;
  Vec            coords;
  DMDACoor2d       **_coords;
  PetscInt       si,sj,nx,ny,i,j;
  FILE           *fp;
  char           fname[PETSC_MAX_PATH_LEN];
  PetscMPIInt    rank;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
  ierr = PetscSNPrintf(fname,sizeof fname,"%s-p%1.4d.dat",prefix,rank);CHKERRQ(ierr);
  ierr = PetscFOpen(PETSC_COMM_SELF,fname,"w",&fp);CHKERRQ(ierr);
  if (fp == PETSC_NULL) {
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"Cannot open file");
  }

  ierr = PetscFPrintf(PETSC_COMM_SELF,fp,"### Element geometry for processor %1.4d ### \n",rank);CHKERRQ(ierr);

  ierr = DMDAGetCoordinateDA(da,&cda);CHKERRQ(ierr);
  ierr = DMDAGetGhostedCoordinates(da,&coords);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(cda,coords,&_coords);CHKERRQ(ierr);
  ierr = DMDAGetGhostCorners(cda,&si,&sj,0,&nx,&ny,0);CHKERRQ(ierr);
  for (j = sj; j < sj+ny-1; j++) {
    for (i = si; i < si+nx-1; i++) {
      ierr = PetscFPrintf(PETSC_COMM_SELF,fp,"%1.6e %1.6e \n",PetscRealPart(_coords[j][i].x),PetscRealPart(_coords[j][i].y));CHKERRQ(ierr);
      ierr = PetscFPrintf(PETSC_COMM_SELF,fp,"%1.6e %1.6e \n",PetscRealPart(_coords[j+1][i].x),PetscRealPart(_coords[j+1][i].y));CHKERRQ(ierr);
      ierr = PetscFPrintf(PETSC_COMM_SELF,fp,"%1.6e %1.6e \n",PetscRealPart(_coords[j+1][i+1].x),PetscRealPart(_coords[j+1][i+1].y));CHKERRQ(ierr);
      ierr = PetscFPrintf(PETSC_COMM_SELF,fp,"%1.6e %1.6e \n",PetscRealPart(_coords[j][i+1].x),PetscRealPart(_coords[j][i+1].y));CHKERRQ(ierr);
      ierr = PetscFPrintf(PETSC_COMM_SELF,fp,"%1.6e %1.6e \n\n",PetscRealPart(_coords[j][i].x),PetscRealPart(_coords[j][i].y));CHKERRQ(ierr);
    }
  }
  ierr = DMDAVecRestoreArray(cda,coords,&_coords);CHKERRQ(ierr);

  ierr = PetscFClose(PETSC_COMM_SELF,fp);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMDAViewGnuplot2d"
static PetscErrorCode DMDAViewGnuplot2d(DM da,Vec fields,const char comment[],const char prefix[])
{
  DM             cda;
  Vec            coords,local_fields;
  DMDACoor2d       **_coords;
  FILE           *fp;
  char           fname[PETSC_MAX_PATH_LEN];
  const char     *field_name;
  PetscMPIInt    rank;
  PetscInt       si,sj,nx,ny,i,j;
  PetscInt       n_dofs,d;
  PetscScalar    *_fields;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  MPI_Comm_rank(PETSC_COMM_WORLD,&rank);
  ierr = PetscSNPrintf(fname,sizeof fname,"%s-p%1.4d.dat",prefix,rank);CHKERRQ(ierr);
  ierr = PetscFOpen(PETSC_COMM_SELF,fname,"w",&fp);CHKERRQ(ierr);
  if (fp == PETSC_NULL) {
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"Cannot open file");
  }

  ierr = PetscFPrintf(PETSC_COMM_SELF,fp,"### %s (processor %1.4d) ### \n",comment,rank);CHKERRQ(ierr);
  ierr = DMDAGetInfo(da,0,0,0,0,0,0,0,&n_dofs,0,0,0,0,0);CHKERRQ(ierr);
  ierr = PetscFPrintf(PETSC_COMM_SELF,fp,"### x y ");CHKERRQ(ierr);
  for (d = 0; d < n_dofs; d++) {
    ierr = DMDAGetFieldName(da,d,&field_name);CHKERRQ(ierr);
    ierr = PetscFPrintf(PETSC_COMM_SELF,fp,"%s ",field_name);CHKERRQ(ierr);
  }
  ierr = PetscFPrintf(PETSC_COMM_SELF,fp,"###\n");CHKERRQ(ierr);


  ierr = DMDAGetCoordinateDA(da,&cda);CHKERRQ(ierr);
  ierr = DMDAGetGhostedCoordinates(da,&coords);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(cda,coords,&_coords);CHKERRQ(ierr);
  ierr = DMDAGetGhostCorners(cda,&si,&sj,0,&nx,&ny,0);CHKERRQ(ierr);

  ierr = DMCreateLocalVector(da,&local_fields);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(da,fields,INSERT_VALUES,local_fields);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(da,fields,INSERT_VALUES,local_fields);CHKERRQ(ierr);
  ierr = VecGetArray(local_fields,&_fields);CHKERRQ(ierr);

  for (j = sj; j < sj+ny; j++) {
    for (i = si; i < si+nx; i++) {
      PetscScalar coord_x,coord_y;
      PetscScalar field_d;

      coord_x = _coords[j][i].x;
      coord_y = _coords[j][i].y;

      ierr = PetscFPrintf(PETSC_COMM_SELF,fp,"%1.6e %1.6e ",PetscRealPart(coord_x),PetscRealPart(coord_y));CHKERRQ(ierr);
      for (d = 0; d < n_dofs; d++) {
        field_d = _fields[ n_dofs*((i-si)+(j-sj)*(nx))+d ];
        ierr    = PetscFPrintf(PETSC_COMM_SELF,fp,"%1.6e ",PetscRealPart(field_d));CHKERRQ(ierr);
      }
      ierr = PetscFPrintf(PETSC_COMM_SELF,fp,"\n");CHKERRQ(ierr);
    }
  }
  ierr = VecRestoreArray(local_fields,&_fields);CHKERRQ(ierr);
  ierr = VecDestroy(&local_fields);CHKERRQ(ierr);

  ierr = DMDAVecRestoreArray(cda,coords,&_coords);CHKERRQ(ierr);

  ierr = PetscFClose(PETSC_COMM_SELF,fp);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMDAViewCoefficientsGnuplot2d"
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
  PetscErrorCode         ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
  ierr = PetscSNPrintf(fname,sizeof fname,"%s-p%1.4d.dat",prefix,rank);CHKERRQ(ierr);
  ierr = PetscFOpen(PETSC_COMM_SELF,fname,"w",&fp);CHKERRQ(ierr);
  if (!fp) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"Cannot open file");

  ierr = PetscFPrintf(PETSC_COMM_SELF,fp,"### %s (processor %1.4d) ### \n",comment,rank);CHKERRQ(ierr);
  ierr = DMDAGetInfo(da,0,0,0,0,0,0,0,&n_dofs,0,0,0,0,0);CHKERRQ(ierr);
  ierr = PetscFPrintf(PETSC_COMM_SELF,fp,"### x y ");CHKERRQ(ierr);
  for (d = 0; d < n_dofs; d++) {
    ierr = DMDAGetFieldName(da,d,&field_name);CHKERRQ(ierr);
    ierr = PetscFPrintf(PETSC_COMM_SELF,fp,"%s ",field_name);CHKERRQ(ierr);
  }
  ierr = PetscFPrintf(PETSC_COMM_SELF,fp,"###\n");CHKERRQ(ierr);


  ierr = DMDAGetCoordinateDA(da,&cda);CHKERRQ(ierr);
  ierr = DMDAGetGhostCorners(cda,&si,&sj,0,&nx,&ny,0);CHKERRQ(ierr);

  ierr = DMCreateLocalVector(da,&local_fields);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(da,fields,INSERT_VALUES,local_fields);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(da,fields,INSERT_VALUES,local_fields);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(da,local_fields,&_coefficients);CHKERRQ(ierr);

  for (j = sj; j < sj+ny; j++) {
    for (i = si; i < si+nx; i++) {
      PetscScalar coord_x,coord_y;

      for (p = 0; p < GAUSS_POINTS; p++) {
        coord_x = _coefficients[j][i].gp_coords[2*p];
        coord_y = _coefficients[j][i].gp_coords[2*p+1];

        ierr = PetscFPrintf(PETSC_COMM_SELF,fp,"%1.6e %1.6e ",PetscRealPart(coord_x),PetscRealPart(coord_y));CHKERRQ(ierr);

        ierr = PetscFPrintf(PETSC_COMM_SELF,fp,"%1.6e %1.6e %1.6e %1.6e",
                            PetscRealPart(_coefficients[j][i].E[p]),PetscRealPart(_coefficients[j][i].nu[p]),
                            PetscRealPart(_coefficients[j][i].fx[p]),PetscRealPart(_coefficients[j][i].fy[p]));CHKERRQ(ierr);
        ierr = PetscFPrintf(PETSC_COMM_SELF,fp,"\n");CHKERRQ(ierr);
      }
    }
  }
  ierr = DMDAVecRestoreArray(da,local_fields,&_coefficients);CHKERRQ(ierr);
  ierr = VecDestroy(&local_fields);CHKERRQ(ierr);

  ierr = PetscFClose(PETSC_COMM_SELF,fp);CHKERRQ(ierr);
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
    prop_E  = E[p];
    prop_nu = nu[p];
    factor = prop_E / (  (1.0+prop_nu)*(1.0-2.0*prop_nu)  );
    constit_D[0][0] = 1.0-prop_nu;	constit_D[0][1] = prop_nu; 			constit_D[0][2] = 0.0;
    constit_D[1][0] = prop_nu;		  constit_D[1][1] = 1.0-prop_nu; 	constit_D[1][2] = 0.0;
    constit_D[2][0] = 0.0;			    constit_D[2][1] = 0.0;					constit_D[2][2] = 0.5*(1.0-2.0*prop_nu);
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
      Fe[NSD*i  ] += fac*Ni_p[i]*fx[p];
      Fe[NSD*i+1] += fac*Ni_p[i]*fy[p];
    }
  }
}

/*
 i,j are the element indices
 The unknown is a vector quantity.
 The s[].c is used to indicate the degree of freedom.
 */
#undef __FUNCT__
#define __FUNCT__ "DMDAGetElementEqnums_u"
static PetscErrorCode DMDAGetElementEqnums_u(MatStencil s_u[],PetscInt i,PetscInt j)
{
  PetscFunctionBegin;
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

#undef __FUNCT__
#define __FUNCT__ "GetElementCoords"
static PetscErrorCode GetElementCoords(DMDACoor2d **_coords,PetscInt ei,PetscInt ej,PetscScalar el_coords[])
{
  PetscFunctionBegin;
  /* get coords for the element */
  el_coords[NSD*0+0] = _coords[ej  ][ei  ].x;  el_coords[NSD*0+1] = _coords[ej  ][ei  ].y;
  el_coords[NSD*1+0] = _coords[ej+1][ei  ].x;  el_coords[NSD*1+1] = _coords[ej+1][ei  ].y;
  el_coords[NSD*2+0] = _coords[ej+1][ei+1].x;  el_coords[NSD*2+1] = _coords[ej+1][ei+1].y;
  el_coords[NSD*3+0] = _coords[ej  ][ei+1].x;  el_coords[NSD*3+1] = _coords[ej  ][ei+1].y;

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "AssembleA_Elasticity"
static PetscErrorCode AssembleA_Elasticity(Mat A,DM elas_da,DM properties_da,Vec properties)
{
  DM                     cda;
  Vec                    coords;
  DMDACoor2d               **_coords;
  MatStencil             u_eqn[NODES_PER_EL*U_DOFS]; /* 2 degrees of freedom */
  PetscInt               sex,sey,mx,my;
  PetscInt               ei,ej;
  PetscScalar            Ae[NODES_PER_EL*U_DOFS*NODES_PER_EL*U_DOFS];
  PetscScalar            el_coords[NODES_PER_EL*NSD];
  Vec                    local_properties;
  GaussPointCoefficients **props;
  PetscScalar            *prop_E,*prop_nu;
  PetscErrorCode         ierr;

  PetscFunctionBegin;
  /* setup for coords */
  ierr = DMDAGetCoordinateDA(elas_da,&cda);CHKERRQ(ierr);
  ierr = DMDAGetGhostedCoordinates(elas_da,&coords);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(cda,coords,&_coords);CHKERRQ(ierr);

  /* setup for coefficients */
  ierr = DMCreateLocalVector(properties_da,&local_properties);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(properties_da,properties,INSERT_VALUES,local_properties);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(properties_da,properties,INSERT_VALUES,local_properties);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(properties_da,local_properties,&props);CHKERRQ(ierr);

  ierr = DMDAGetElementCorners(elas_da,&sex,&sey,0,&mx,&my,0);CHKERRQ(ierr);
  for (ej = sey; ej < sey+my; ej++) {
    for (ei = sex; ei < sex+mx; ei++) {
      /* get coords for the element */
      GetElementCoords(_coords,ei,ej,el_coords);

      /* get coefficients for the element */
      prop_E  = props[ej][ei].E;
      prop_nu = props[ej][ei].nu;

      /* initialise element stiffness matrix */
      ierr = PetscMemzero(Ae,sizeof(PetscScalar)*NODES_PER_EL*U_DOFS*NODES_PER_EL*U_DOFS);CHKERRQ(ierr);

      /* form element stiffness matrix */
      FormStressOperatorQ1(Ae,el_coords,prop_E,prop_nu);

      /* insert element matrix into global matrix */
      ierr = DMDAGetElementEqnums_u(u_eqn,ei,ej);CHKERRQ(ierr);
      ierr = MatSetValuesStencil(A,NODES_PER_EL*U_DOFS,u_eqn,NODES_PER_EL*U_DOFS,u_eqn,Ae,ADD_VALUES);CHKERRQ(ierr);
    }
  }
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  ierr = DMDAVecRestoreArray(cda,coords,&_coords);CHKERRQ(ierr);

  ierr = DMDAVecRestoreArray(properties_da,local_properties,&props);CHKERRQ(ierr);
  ierr = VecDestroy(&local_properties);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "DMDASetValuesLocalStencil_ADD_VALUES"
static PetscErrorCode DMDASetValuesLocalStencil_ADD_VALUES(ElasticityDOF **fields_F,MatStencil u_eqn[],PetscScalar Fe_u[])
{
  PetscInt n;

  PetscFunctionBegin;
  for (n = 0; n < 4; n++) {
    fields_F[ u_eqn[2*n  ].j ][ u_eqn[2*n  ].i ].ux_dof = fields_F[ u_eqn[2*n  ].j ][ u_eqn[2*n  ].i ].ux_dof+Fe_u[2*n  ];
    fields_F[ u_eqn[2*n+1].j ][ u_eqn[2*n+1].i ].uy_dof = fields_F[ u_eqn[2*n+1].j ][ u_eqn[2*n+1].i ].uy_dof+Fe_u[2*n+1];
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "AssembleF_Elasticity"
static PetscErrorCode AssembleF_Elasticity(Vec F,DM elas_da,DM properties_da,Vec properties)
{
  DM                     cda;
  Vec                    coords;
  DMDACoor2d               **_coords;
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
  PetscErrorCode         ierr;

  PetscFunctionBegin;
  /* setup for coords */
  ierr = DMDAGetCoordinateDA(elas_da,&cda);CHKERRQ(ierr);
  ierr = DMDAGetGhostedCoordinates(elas_da,&coords);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(cda,coords,&_coords);CHKERRQ(ierr);

  /* setup for coefficients */
  ierr = DMGetLocalVector(properties_da,&local_properties);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(properties_da,properties,INSERT_VALUES,local_properties);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(properties_da,properties,INSERT_VALUES,local_properties);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(properties_da,local_properties,&props);CHKERRQ(ierr);

  /* get acces to the vector */
  ierr = DMGetLocalVector(elas_da,&local_F);CHKERRQ(ierr);
  ierr = VecZeroEntries(local_F);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(elas_da,local_F,&ff);CHKERRQ(ierr);


  ierr = DMDAGetElementCorners(elas_da,&sex,&sey,0,&mx,&my,0);CHKERRQ(ierr);
  for (ej = sey; ej < sey+my; ej++) {
    for (ei = sex; ei < sex+mx; ei++) {
      /* get coords for the element */
      GetElementCoords(_coords,ei,ej,el_coords);

      /* get coefficients for the element */
      prop_fx = props[ej][ei].fx;
      prop_fy = props[ej][ei].fy;

      /* initialise element stiffness matrix */
      ierr = PetscMemzero(Fe,sizeof(PetscScalar)*NODES_PER_EL*U_DOFS);CHKERRQ(ierr);

      /* form element stiffness matrix */
      FormMomentumRhsQ1(Fe,el_coords,prop_fx,prop_fy);

      /* insert element matrix into global matrix */
      ierr = DMDAGetElementEqnums_u(u_eqn,ei,ej);CHKERRQ(ierr);

      ierr = DMDASetValuesLocalStencil_ADD_VALUES(ff,u_eqn,Fe);CHKERRQ(ierr);
    }
  }

  ierr = DMDAVecRestoreArray(elas_da,local_F,&ff);CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(elas_da,local_F,ADD_VALUES,F);CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd(elas_da,local_F,ADD_VALUES,F);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(elas_da,&local_F);CHKERRQ(ierr);

  ierr = DMDAVecRestoreArray(cda,coords,&_coords);CHKERRQ(ierr);

  ierr = DMDAVecRestoreArray(properties_da,local_properties,&props);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(properties_da,&local_properties);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "solve_elasticity_2d"
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
  DMDACoor2d               **_prop_coords,**_vel_coords;
  GaussPointCoefficients **element_props;
  KSP                    ksp_E;
  PetscInt               coefficient_structure = 0;
  PetscInt               cpu_x,cpu_y,*lx = PETSC_NULL,*ly = PETSC_NULL;
  PetscBool              use_gp_coords = PETSC_FALSE;
  PetscBool              use_nonsymbc = PETSC_FALSE;
  PetscBool              no_view = PETSC_FALSE;
  PetscBool              flg;
  PetscErrorCode         ierr;

  PetscFunctionBegin;
  /* Generate the da for velocity and pressure */
  /*
   We use Q1 elements for the temperature.
   FEM has a 9-point stencil (BOX) or connectivity pattern
   Num nodes in each direction is mx+1, my+1
   */
  u_dof         = U_DOFS; /* Vx, Vy - velocities */
  dof           = u_dof;
  stencil_width = 1;
  ierr          = DMDACreate2d(PETSC_COMM_WORLD, DMDA_BOUNDARY_NONE, DMDA_BOUNDARY_NONE,DMDA_STENCIL_BOX,
                             mx+1,my+1,PETSC_DECIDE,PETSC_DECIDE,dof,stencil_width,PETSC_NULL,PETSC_NULL,&elas_da);CHKERRQ(ierr);
  ierr = DMDASetFieldName(elas_da,0,"Ux");CHKERRQ(ierr);
  ierr = DMDASetFieldName(elas_da,1,"Uy");CHKERRQ(ierr);

  /* unit box [0,1] x [0,1] */
  ierr = DMDASetUniformCoordinates(elas_da,0.0,1.0,0.0,1.0,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);


  /* Generate element properties, we will assume all material properties are constant over the element */
  /* local number of elements */
  ierr = DMDAGetLocalElementSize(elas_da,&mxl,&myl,PETSC_NULL);CHKERRQ(ierr);

  /* !!! IN PARALLEL WE MUST MAKE SURE THE TWO DMDA's ALIGN !!! // */
  ierr = DMDAGetInfo(elas_da,0,0,0,0,&cpu_x,&cpu_y,0,0,0,0,0,0,0);CHKERRQ(ierr);
  ierr = DMDAGetElementOwnershipRanges2d(elas_da,&lx,&ly);CHKERRQ(ierr);

  prop_dof           = (PetscInt)(sizeof(GaussPointCoefficients)/sizeof(PetscScalar)); /* gauss point setup */
  prop_stencil_width = 0;
  ierr               = DMDACreate2d(PETSC_COMM_WORLD, DMDA_BOUNDARY_NONE, DMDA_BOUNDARY_NONE,DMDA_STENCIL_BOX,
                                  mx,my,cpu_x,cpu_y,prop_dof,prop_stencil_width,lx,ly,&da_prop);CHKERRQ(ierr);
  ierr = PetscFree(lx);CHKERRQ(ierr);
  ierr = PetscFree(ly);CHKERRQ(ierr);

  /* define centroid positions */
  ierr = DMDAGetInfo(da_prop,0,&M,&N,0,0,0,0,0,0,0,0,0,0);CHKERRQ(ierr);
  dx   = 1.0/((PetscReal)(M));
  dy   = 1.0/((PetscReal)(N));

  ierr = DMDASetUniformCoordinates(da_prop,0.0+0.5*dx,1.0-0.5*dx,0.0+0.5*dy,1.0-0.5*dy,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);

  /* define coefficients */
  ierr = PetscOptionsGetInt(PETSC_NULL,"-c_str",&coefficient_structure,PETSC_NULL);CHKERRQ(ierr);

  ierr = DMCreateGlobalVector(da_prop,&properties);CHKERRQ(ierr);
  ierr = DMCreateLocalVector(da_prop,&l_properties);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(da_prop,l_properties,&element_props);CHKERRQ(ierr);

  ierr = DMDAGetCoordinateDA(da_prop,&prop_cda);CHKERRQ(ierr);
  ierr = DMDAGetGhostedCoordinates(da_prop,&prop_coords);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(prop_cda,prop_coords,&_prop_coords);CHKERRQ(ierr);

  ierr = DMDAGetGhostCorners(prop_cda,&si,&sj,0,&nx,&ny,0);CHKERRQ(ierr);

  ierr = DMDAGetCoordinateDA(elas_da,&vel_cda);CHKERRQ(ierr);
  ierr = DMDAGetGhostedCoordinates(elas_da,&vel_coords);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(vel_cda,vel_coords,&_vel_coords);CHKERRQ(ierr);


  /* interpolate the coordinates */
  for (j = sj; j < sj+ny; j++) {
    for (i = si; i < si+nx; i++) {
      PetscInt    ngp;
      PetscScalar gp_xi[GAUSS_POINTS][2],gp_weight[GAUSS_POINTS];
      PetscScalar el_coords[8];

      ierr = GetElementCoords(_vel_coords,i,j,el_coords);CHKERRQ(ierr);
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
          gp_x = gp_x+Ni_p[n]*el_coords[2*n  ];
          gp_y = gp_y+Ni_p[n]*el_coords[2*n+1];
        }
        element_props[j][i].gp_coords[2*p  ] = gp_x;
        element_props[j][i].gp_coords[2*p+1] = gp_y;
      }
    }
  }

  /* define the coefficients */
  ierr = PetscOptionsGetBool(PETSC_NULL,"-use_gp_coords",&use_gp_coords,&flg);CHKERRQ(ierr);

  for (j = sj; j < sj+ny; j++) {
    for (i = si; i < si+nx; i++) {
      PetscScalar centroid_x = _prop_coords[j][i].x; /* centroids of cell */
      PetscScalar centroid_y = _prop_coords[j][i].y;
      PETSC_UNUSED PetscScalar coord_x,coord_y;


      if (coefficient_structure == 0) { /* isotropic */
        PetscScalar opts_E,opts_nu;

        opts_E  = 1.0;
        opts_nu = 0.33;
        ierr = PetscOptionsGetScalar(PETSC_NULL,"-iso_E",&opts_E,&flg);CHKERRQ(ierr);
        ierr = PetscOptionsGetScalar(PETSC_NULL,"-iso_nu",&opts_nu,&flg);CHKERRQ(ierr);

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
        ierr = PetscOptionsGetScalar(PETSC_NULL,"-step_E0",&opts_E0,&flg);CHKERRQ(ierr);
        ierr = PetscOptionsGetScalar(PETSC_NULL,"-step_nu0",&opts_nu0,&flg);CHKERRQ(ierr);
        ierr = PetscOptionsGetScalar(PETSC_NULL,"-step_E1",&opts_E1,&flg);CHKERRQ(ierr);
        ierr = PetscOptionsGetScalar(PETSC_NULL,"-step_nu1",&opts_nu1,&flg);CHKERRQ(ierr);
        ierr = PetscOptionsGetScalar(PETSC_NULL,"-step_xc",&opts_xc,&flg);CHKERRQ(ierr);

        for (p = 0; p < GAUSS_POINTS; p++) {
          coord_x = centroid_x;
          coord_y = centroid_y;
          if (use_gp_coords == PETSC_TRUE) {
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
        PetscInt nbricks,maxnbricks;
        PetscInt index,span;
        PetscInt jj;

        flg = PETSC_FALSE;
        maxnbricks = 10;
        ierr = PetscOptionsGetRealArray( PETSC_NULL, "-brick_E",values_E,&maxnbricks,&flg);CHKERRQ(ierr);
        nbricks = maxnbricks;
        if (!flg) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"User must supply a list of E values for each brick");CHKERRQ(ierr);

        flg = PETSC_FALSE;
        maxnbricks = 10;
        ierr = PetscOptionsGetRealArray( PETSC_NULL, "-brick_nu",values_nu,&maxnbricks,&flg);CHKERRQ(ierr);
        if (!flg) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"User must supply a list of nu values for each brick");CHKERRQ(ierr);
        if (maxnbricks != nbricks) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"User must supply equal numbers of values for E and nu");CHKERRQ(ierr);

        span = 1;
        ierr = PetscOptionsGetInt(PETSC_NULL,"-brick_span",&span,&flg);CHKERRQ(ierr);

        /* cycle through the indices so that no two material properties are repeated in lines of x or y */
        jj = (j/span)%nbricks;
        index = (jj+i/span)%nbricks;
        /*printf("j=%d: index = %d \n", j,index ); */

        for (p = 0; p < GAUSS_POINTS; p++) {
          element_props[j][i].E[p]  = values_E[index];
          element_props[j][i].nu[p] = values_nu[index];
        }
      } else if (coefficient_structure == 3) { /* sponge */
        PetscScalar opts_E0,opts_nu0;
        PetscScalar opts_E1,opts_nu1;
        PetscInt opts_t,opts_w;
        PetscInt ii,jj,ci,cj;

        opts_E0  = opts_E1  = 1.0;
        opts_nu0 = opts_nu1 = 0.333;
        ierr = PetscOptionsGetScalar(PETSC_NULL,"-sponge_E0",&opts_E0,&flg);CHKERRQ(ierr);
        ierr = PetscOptionsGetScalar(PETSC_NULL,"-sponge_nu0",&opts_nu0,&flg);CHKERRQ(ierr);
        ierr = PetscOptionsGetScalar(PETSC_NULL,"-sponge_E1",&opts_E1,&flg);CHKERRQ(ierr);
        ierr = PetscOptionsGetScalar(PETSC_NULL,"-sponge_nu1",&opts_nu1,&flg);CHKERRQ(ierr);

        opts_t = opts_w = 1;
        ierr = PetscOptionsGetInt(PETSC_NULL,"-sponge_t",&opts_t,&flg);CHKERRQ(ierr);
        ierr = PetscOptionsGetInt(PETSC_NULL,"-sponge_w",&opts_w,&flg);CHKERRQ(ierr);

        ii = (i)/(opts_t+opts_w+opts_t);
        jj = (j)/(opts_t+opts_w+opts_t);

        ci = i - ii*(opts_t+opts_w+opts_t);
        cj = j - jj*(opts_t+opts_w+opts_t);

        for (p = 0; p < GAUSS_POINTS; p++) {
          element_props[j][i].E[p]  = opts_E0;
          element_props[j][i].nu[p] = opts_nu0;
        }
        if ( (ci >= opts_t) && (ci < opts_t+opts_w) ) {
          if ( (cj >= opts_t) && (cj < opts_t+opts_w) ) {
            for (p = 0; p < GAUSS_POINTS; p++) {
              element_props[j][i].E[p]  = opts_E1;
              element_props[j][i].nu[p] = opts_nu1;
            }
          }
        }

      } else {
        SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"Unknown coefficient_structure");
      }
    }
  }
  ierr = DMDAVecRestoreArray(prop_cda,prop_coords,&_prop_coords);CHKERRQ(ierr);

  ierr = DMDAVecRestoreArray(vel_cda,vel_coords,&_vel_coords);CHKERRQ(ierr);

  ierr = DMDAVecRestoreArray(da_prop,l_properties,&element_props);CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(da_prop,l_properties,ADD_VALUES,properties);CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd(da_prop,l_properties,ADD_VALUES,properties);CHKERRQ(ierr);

  ierr = PetscOptionsGetBool(PETSC_NULL,"-no_view",&no_view,PETSC_NULL);CHKERRQ(ierr);
  if (!no_view) {
    ierr = DMDAViewCoefficientsGnuplot2d(da_prop,properties,"Coeffcients for elasticity eqn.","properties");CHKERRQ(ierr);
    ierr = DMDACoordViewGnuplot2d(elas_da,"mesh");CHKERRQ(ierr);
  }

  /* Generate a matrix with the correct non-zero pattern of type AIJ. This will work in parallel and serial */
  ierr = DMCreateMatrix(elas_da,MATAIJ,&A);CHKERRQ(ierr);
  ierr = DMDAGetCoordinates(elas_da,&vel_coords);CHKERRQ(ierr);
  ierr = MatNullSpaceCreateRigidBody(vel_coords,&matnull);CHKERRQ(ierr);
  ierr = MatSetNearNullSpace(A,matnull);CHKERRQ(ierr);
  ierr = MatNullSpaceDestroy(&matnull);CHKERRQ(ierr);
  ierr = MatGetVecs(A,&f,&X);CHKERRQ(ierr);

  /* assemble A11 */
  ierr = MatZeroEntries(A);CHKERRQ(ierr);
  ierr = VecZeroEntries(f);CHKERRQ(ierr);

  ierr = AssembleA_Elasticity(A,elas_da,da_prop,properties);CHKERRQ(ierr);
  /* build force vector */
  ierr = AssembleF_Elasticity(f,elas_da,da_prop,properties);CHKERRQ(ierr);


  ierr = KSPCreate(PETSC_COMM_WORLD,&ksp_E);CHKERRQ(ierr);
  ierr = KSPSetOptionsPrefix(ksp_E,"elas_");CHKERRQ(ierr);  /* elasticity */

  ierr = PetscOptionsGetBool(PETSC_NULL,"-use_nonsymbc",&use_nonsymbc,&flg);CHKERRQ(ierr);
  /* solve */
  if (use_nonsymbc == PETSC_FALSE) {
    Mat AA;
    Vec ff,XX;
    IS is;
    VecScatter scat;

    ierr = DMDABCApplySymmetricCompression(elas_da,A,f,&is,&AA,&ff);CHKERRQ(ierr);
    ierr = VecDuplicate(ff,&XX);CHKERRQ(ierr);

    ierr = KSPSetOperators(ksp_E,AA,AA,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
    ierr = KSPSetFromOptions(ksp_E);CHKERRQ(ierr);

    ierr = KSPSolve(ksp_E,ff,XX);CHKERRQ(ierr);

    /* push XX back into X */
    ierr = DMDABCApplyCompression(elas_da,PETSC_NULL,X);CHKERRQ(ierr);

    ierr = VecScatterCreate(XX,PETSC_NULL,X,is,&scat);CHKERRQ(ierr);
    ierr = VecScatterBegin(scat,XX,X,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterEnd(scat,XX,X,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterDestroy(&scat);CHKERRQ(ierr);

    ierr = MatDestroy(&AA);CHKERRQ(ierr);
    ierr = VecDestroy(&ff);CHKERRQ(ierr);
    ierr = VecDestroy(&XX);CHKERRQ(ierr);
    ierr = ISDestroy(&is);CHKERRQ(ierr);
  } else {
    ierr = DMDABCApplyCompression(elas_da,A,f);CHKERRQ(ierr);

    ierr = KSPSetOperators(ksp_E,A,A,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
    ierr = KSPSetFromOptions(ksp_E);CHKERRQ(ierr);

    ierr = KSPSolve(ksp_E,f,X);CHKERRQ(ierr);
  }

  if (!no_view) {ierr = DMDAViewGnuplot2d(elas_da,X,"Displacement solution for elasticity eqn.","X");CHKERRQ(ierr);}
  ierr = KSPDestroy(&ksp_E);CHKERRQ(ierr);

  ierr = VecDestroy(&X);CHKERRQ(ierr);
  ierr = VecDestroy(&f);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);

  ierr = DMDestroy(&elas_da);CHKERRQ(ierr);
  ierr = DMDestroy(&da_prop);CHKERRQ(ierr);

  ierr = VecDestroy(&properties);CHKERRQ(ierr);
  ierr = VecDestroy(&l_properties);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **args)
{
  PetscErrorCode ierr;
  PetscInt       mx,my;

  ierr = PetscInitialize(&argc,&args,(char *)0,help);CHKERRQ(ierr);

  mx   = my = 10;
  ierr = PetscOptionsGetInt(PETSC_NULL,"-mx",&mx,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-my",&my,PETSC_NULL);CHKERRQ(ierr);

  ierr = solve_elasticity_2d(mx,my);CHKERRQ(ierr);

  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}

/* -------------------------- helpers for boundary conditions -------------------------------- */

#undef __FUNCT__
#define __FUNCT__ "BCApply_EAST"
static PetscErrorCode BCApply_EAST(DM da,PetscInt d_idx,PetscScalar bc_val,Mat A,Vec b)
{
  DM             cda;
  Vec            coords;
  PetscInt       si,sj,nx,ny,i,j;
  PetscInt       M,N;
  DMDACoor2d       **_coords;
  PetscInt       *g_idx;
  PetscInt       *bc_global_ids;
  PetscScalar    *bc_vals;
  PetscInt       nbcs;
  PetscInt       n_dofs;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* enforce bc's */
  ierr = DMDAGetGlobalIndices(da,PETSC_NULL,&g_idx);CHKERRQ(ierr);

  ierr = DMDAGetCoordinateDA(da,&cda);CHKERRQ(ierr);
  ierr = DMDAGetGhostedCoordinates(da,&coords);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(cda,coords,&_coords);CHKERRQ(ierr);
  ierr = DMDAGetGhostCorners(cda,&si,&sj,0,&nx,&ny,0);CHKERRQ(ierr);
  ierr = DMDAGetInfo(da,0,&M,&N,0,0,0,0,&n_dofs,0,0,0,0,0);CHKERRQ(ierr);

  /* /// */

  ierr = PetscMalloc(sizeof(PetscInt)*ny*n_dofs,&bc_global_ids);CHKERRQ(ierr);
  ierr = PetscMalloc(sizeof(PetscScalar)*ny*n_dofs,&bc_vals);CHKERRQ(ierr);

  /* init the entries to -1 so VecSetValues will ignore them */
  for (i = 0; i < ny*n_dofs; i++) {
    bc_global_ids[i] = -1;
  }

  i = nx-1;
  for (j = 0; j < ny; j++) {
    PetscInt    local_id;
    PETSC_UNUSED PetscScalar coordx,coordy;

    local_id = i+j*nx;

    bc_global_ids[j] = g_idx[ n_dofs*local_id+d_idx ];

    coordx = _coords[j+sj][i+si].x;
    coordy = _coords[j+sj][i+si].y;

    bc_vals[j] =  bc_val;
  }
  nbcs = 0;
  if ((si+nx) == (M)) {
    nbcs = ny;
  }

  if (b) {
    ierr = VecSetValues(b,nbcs,bc_global_ids,bc_vals,INSERT_VALUES);CHKERRQ(ierr);
    ierr = VecAssemblyBegin(b);CHKERRQ(ierr);
    ierr = VecAssemblyEnd(b);CHKERRQ(ierr);
  }
  if (A) {
    ierr = MatZeroRows(A,nbcs,bc_global_ids,1.0,0,0);CHKERRQ(ierr);
  }

  ierr = PetscFree(bc_vals);CHKERRQ(ierr);
  ierr = PetscFree(bc_global_ids);CHKERRQ(ierr);

  ierr = DMDAVecRestoreArray(cda,coords,&_coords);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "BCApply_WEST"
static PetscErrorCode BCApply_WEST(DM da,PetscInt d_idx,PetscScalar bc_val,Mat A,Vec b)
{
  DM             cda;
  Vec            coords;
  PetscInt       si,sj,nx,ny,i,j;
  PetscInt       M,N;
  DMDACoor2d       **_coords;
  PetscInt       *g_idx;
  PetscInt       *bc_global_ids;
  PetscScalar    *bc_vals;
  PetscInt       nbcs;
  PetscInt       n_dofs;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* enforce bc's */
  ierr = DMDAGetGlobalIndices(da,PETSC_NULL,&g_idx);CHKERRQ(ierr);

  ierr = DMDAGetCoordinateDA(da,&cda);CHKERRQ(ierr);
  ierr = DMDAGetGhostedCoordinates(da,&coords);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(cda,coords,&_coords);CHKERRQ(ierr);
  ierr = DMDAGetGhostCorners(cda,&si,&sj,0,&nx,&ny,0);CHKERRQ(ierr);
  ierr = DMDAGetInfo(da,0,&M,&N,0,0,0,0,&n_dofs,0,0,0,0,0);CHKERRQ(ierr);

  /* /// */

  ierr = PetscMalloc(sizeof(PetscInt)*ny*n_dofs,&bc_global_ids);CHKERRQ(ierr);
  ierr = PetscMalloc(sizeof(PetscScalar)*ny*n_dofs,&bc_vals);CHKERRQ(ierr);

  /* init the entries to -1 so VecSetValues will ignore them */
  for (i = 0; i < ny*n_dofs; i++) {
    bc_global_ids[i] = -1;
  }

  i = 0;
  for (j = 0; j < ny; j++) {
    PetscInt    local_id;
    PETSC_UNUSED PetscScalar coordx,coordy;

    local_id = i+j*nx;

    bc_global_ids[j] = g_idx[ n_dofs*local_id+d_idx ];

    coordx = _coords[j+sj][i+si].x;
    coordy = _coords[j+sj][i+si].y;

    bc_vals[j] =  bc_val;
  }
  nbcs = 0;
  if (si == 0) {
    nbcs = ny;
  }

  if (b) {
    ierr = VecSetValues(b,nbcs,bc_global_ids,bc_vals,INSERT_VALUES);CHKERRQ(ierr);
    ierr = VecAssemblyBegin(b);CHKERRQ(ierr);
    ierr = VecAssemblyEnd(b);CHKERRQ(ierr);
  }
  if (A) {
    ierr = MatZeroRows(A,nbcs,bc_global_ids,1.0,0,0);CHKERRQ(ierr);
  }

  ierr = PetscFree(bc_vals);CHKERRQ(ierr);
  ierr = PetscFree(bc_global_ids);CHKERRQ(ierr);

  ierr = DMDAVecRestoreArray(cda,coords,&_coords);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMDABCApplyCompression"
static PetscErrorCode DMDABCApplyCompression(DM elas_da,Mat A,Vec f)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = BCApply_EAST(elas_da,0,-1.0,A,f);CHKERRQ(ierr);
  ierr = BCApply_EAST(elas_da,1, 0.0,A,f);CHKERRQ(ierr);
  ierr = BCApply_WEST(elas_da,0,1.0,A,f);CHKERRQ(ierr);
  ierr = BCApply_WEST(elas_da,1,0.0,A,f);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMDABCApplySymmetricCompression"
static PetscErrorCode DMDABCApplySymmetricCompression(DM elas_da,Mat A,Vec f,IS *dofs,Mat *AA,Vec *ff)
{
  PetscErrorCode ierr;
  PetscInt start,end,m;
  PetscInt *unconstrained;
  PetscInt cnt,i;
  Vec x;
  PetscScalar *_x;
  IS is;
  VecScatter scat;

  PetscFunctionBegin;
  /* push bc's into f and A */
  ierr = VecDuplicate(f,&x);CHKERRQ(ierr);
  ierr = BCApply_EAST(elas_da,0,-1.0,A,x);CHKERRQ(ierr);
  ierr = BCApply_EAST(elas_da,1, 0.0,A,x);CHKERRQ(ierr);
  ierr = BCApply_WEST(elas_da,0,1.0,A,x);CHKERRQ(ierr);
  ierr = BCApply_WEST(elas_da,1,0.0,A,x);CHKERRQ(ierr);

  /* define which dofs are not constrained */
  ierr = VecGetLocalSize(x,&m);CHKERRQ(ierr);
  ierr = PetscMalloc(sizeof(PetscInt)*m,&unconstrained);CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(x,&start,&end);CHKERRQ(ierr);
  ierr = VecGetArray(x,&_x);CHKERRQ(ierr);
  cnt = 0;
  for (i = 0; i < m; i++ ) {
    PetscReal val;

    val = PetscRealPart(_x[i]);
    if( fabs(val) < 0.1 ) {
      unconstrained[cnt] = start + i;
      cnt++;
    }
  }
  ierr = VecRestoreArray(x,&_x);CHKERRQ(ierr);

  ierr = ISCreateGeneral(PETSC_COMM_WORLD,cnt,unconstrained,PETSC_COPY_VALUES,&is);CHKERRQ(ierr);
  ierr = PetscFree(unconstrained);CHKERRQ(ierr);

  /* define correction for dirichlet in the rhs */
  ierr = MatMult(A,x,f);CHKERRQ(ierr);
  ierr = VecScale(f,-1.0);CHKERRQ(ierr);

  /* get new matrix */
  ierr = MatGetSubMatrix(A,is,is,MAT_INITIAL_MATRIX,AA);CHKERRQ(ierr);
  /* get new vector */
  ierr = MatGetVecs(*AA,PETSC_NULL,ff);CHKERRQ(ierr);

  ierr = VecScatterCreate(f,is,*ff,PETSC_NULL,&scat);CHKERRQ(ierr);
  ierr = VecScatterBegin(scat,f,*ff,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(scat,f,*ff,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterDestroy(&scat);CHKERRQ(ierr);

  *dofs = is;
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
