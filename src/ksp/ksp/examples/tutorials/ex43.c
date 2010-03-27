static char help[] =
  "Solves the incompressible, variable viscosity stokes equation in 2d on the unit domain \n\
using Q1Q1 elements, stabilized with Bochev's polynomial projection method. \n\
The models defined utilise free slip boundary conditions on all sides. \n\
Options: \n\
     -mx : number elements in x-direciton \n\
     -my : number elements in y-direciton \n\
     -c_str : indicates the structure of the coefficients to use. \n\
          -c_str 0 => Setup for an analytic solution with a vertical jump in viscosity. This problem is driven by the \n\
                         forcing function f = ( 0, sin(n_z pi y )cos( pi x ). \n\
                         Parameters: \n\
                              -solcx_eta0 : the viscosity to the left of the interface \n\
                              -solcx_eta1 : the viscosity to the right of the interface \n\
                              -solcx_xc : the location of the interface \n\
                              -solcx_nz : the wavenumber in the y direction \n\
          -c_str 1 => Setup for a rectangular blob located in the origin of the domain. \n\
                         Parameters: \n\
                              -sinker_eta0 : the viscosity of the background fluid \n\
                              -sinker_eta1 : the viscosity of the blob \n\
                              -sinker_dx : the width of the blob \n\
                              -sinker_dy : the width of the blob \n\
          -c_str 2 => Setup for a circular blob located in the origin of the domain. \n\
                         Parameters: \n\
                              -sinker_eta0 : the viscosity of the background fluid \n\
                              -sinker_eta1 : the viscosity of the blob \n\
                              -sinker_r : radius of the blob \n\
     -use_gp_coords : evaluate the viscosity and the body force at the global coordinates of the quadrature points.\n\
     By default, eta and the body force are evaulated at the element center and applied as a constant over the entire element.\n";

/* Contributed by Dave May */

#include "petscksp.h"
#include "petscda.h"

#include "ex43-solCx.h" /* A Maple-generated exact solution */

static PetscErrorCode DABCApplyFreeSlip(DA,Mat,Vec);


#define NSD            2 /* number of spatial dimensions */
#define NODES_PER_EL   4 /* nodes per element */
#define U_DOFS         2 /* degrees of freedom per velocity node */
#define P_DOFS         1 /* degrees of freedom per pressure node */
#define GAUSS_POINTS   4

/* cell based evaluation */
typedef struct {
  PetscScalar eta,fx,fy;
} Coefficients;

/* Gauss point based evaluation 8+4+4+4 = 20 */
typedef struct {
  PetscScalar gp_coords[2*GAUSS_POINTS];
  PetscScalar eta[GAUSS_POINTS];
  PetscScalar fx[GAUSS_POINTS];
  PetscScalar fy[GAUSS_POINTS];
} GaussPointCoefficients;

typedef struct {
  PetscScalar u_dof;
  PetscScalar v_dof;
  PetscScalar p_dof;
} StokesDOF;


/*

D = [ 2.eta   0   0   ]
[   0   2.eta 0   ]
[   0     0   eta ]

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
#define __FUNCT__ "DAGetLocalElementSize"
static PetscErrorCode DAGetLocalElementSize(DA da,PetscInt *mxl,PetscInt *myl,PetscInt *mzl)
{
  PetscInt m,n,p,M,N,P;
  PetscInt sx,sy,sz;
  PetscInt ml,nl,pl;

  PetscFunctionBegin;
  DAGetInfo(da,0,&M,&N,&P,0,0,0,0,0,0,0);
  DAGetCorners(da,&sx,&sy,&sz,&m,&n,&p);

  ml = nl = pl = 0;
  if (mxl != PETSC_NULL) {
    *mxl = m;
    if ((sx+m) == M) {  /* last proc */
      *mxl = m-1;
    }

    ml = *mxl;
  }
  if (myl != PETSC_NULL) {
    *myl = n;
    if ((sy+n) == N) {  /* last proc */
      *myl = n-1;
    }

    nl = *myl;
  }
  if (mzl != PETSC_NULL) {
    *mzl = p;
    if ((sz+p) == P) {  /* last proc */
      *mzl = p-1;
    }

    pl = *mzl;
  }

  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DAGetElementCorners"
static PetscErrorCode DAGetElementCorners(DA da,
                                          PetscInt *sx,PetscInt *sy,PetscInt *sz,
                                          PetscInt *mx,PetscInt *my,PetscInt *mz)
{
  PetscInt si,sj,sk;

  PetscFunctionBegin;
  DAGetGhostCorners(da,&si,&sj,&sk,0,0,0);

  *sx = si;
  if (si != 0) {
    *sx = si+1;
  }

  *sy = sj;
  if (sj != 0) {
    *sy = sj+1;
  }

  if (sk != PETSC_NULL) {
    *sz = sk;
    if (sk != 0) {
      *sz = sk+1;
    }
  }

  DAGetLocalElementSize(da,mx,my,mz);

  PetscFunctionReturn(0);
}

/*
i,j are the element indices
The unknown is a vector quantity.
The s[].c is used to indicate the degree of freedom.
*/
#undef __FUNCT__  
#define __FUNCT__ "DAGetElementEqnums_up"
static PetscErrorCode DAGetElementEqnums_up(MatStencil s_u[],MatStencil s_p[],PetscInt i,PetscInt j)
{
  PetscFunctionBegin;
  /* velocity */
  /* node 0 */
  s_u[0].i = i;s_u[0].j = j;s_u[0].c = 0;                         /* Vx0 */
  s_u[1].i = i;s_u[1].j = j;s_u[1].c = 1;                         /* Vy0 */

  /* node 1 */
  s_u[2].i = i;s_u[2].j = j+1;s_u[2].c = 0;                         /* Vx1 */
  s_u[3].i = i;s_u[3].j = j+1;s_u[3].c = 1;                         /* Vy1 */

  /* node 2 */
  s_u[4].i = i+1;s_u[4].j = j+1;s_u[4].c = 0;                         /* Vx2 */
  s_u[5].i = i+1;s_u[5].j = j+1;s_u[5].c = 1;                         /* Vy2 */

  /* node 3 */
  s_u[6].i = i+1;s_u[6].j = j;s_u[6].c = 0;                         /* Vx3 */
  s_u[7].i = i+1;s_u[7].j = j;s_u[7].c = 1;                         /* Vy3 */


  /* pressure */
  s_p[0].i = i;s_p[0].j = j;s_p[0].c = 2;                         /* P0 */
  s_p[1].i = i;s_p[1].j = j+1;s_p[1].c = 2;                         /* P0 */
  s_p[2].i = i+1;s_p[2].j = j+1;s_p[2].c = 2;                         /* P1 */
  s_p[3].i = i+1;s_p[3].j = j;s_p[3].c = 2;                         /* P1 */
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DAGetElementOwnershipRanges2d"
static PetscErrorCode DAGetElementOwnershipRanges2d(DA da,PetscInt **_lx,PetscInt **_ly)
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

  DAGetInfo(da,0,0,0,0,&cpu_x,&cpu_y,0,0,0,0,0);

  proc_J = rank/cpu_x;
  proc_I = rank-cpu_x*proc_J;

  ierr = PetscMalloc(sizeof(PetscInt)*cpu_x,&LX);CHKERRQ(ierr);
  ierr = PetscMalloc(sizeof(PetscInt)*cpu_y,&LY);CHKERRQ(ierr);

  ierr = DAGetLocalElementSize(da,&local_mx,&local_my,PETSC_NULL);CHKERRQ(ierr);
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
    LX[i] = (PetscInt)_a[i];
  }
  ierr = VecRestoreArray(V_SEQ,&_a);CHKERRQ(ierr);
  ierr = VecScatterDestroy(ctx);CHKERRQ(ierr);
  ierr = VecDestroy(V_SEQ);CHKERRQ(ierr);

  ierr = VecScatterCreateToAll(vly,&ctx,&V_SEQ);CHKERRQ(ierr);
  ierr = VecScatterBegin(ctx,vly,V_SEQ,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(ctx,vly,V_SEQ,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecGetArray(V_SEQ,&_a);CHKERRQ(ierr);
  for (i = 0; i < cpu_y; i++) {
    LY[i] = (PetscInt)_a[i];
  }
  ierr = VecRestoreArray(V_SEQ,&_a);CHKERRQ(ierr);
  ierr = VecScatterDestroy(ctx);CHKERRQ(ierr);
  ierr = VecDestroy(V_SEQ);CHKERRQ(ierr);



  *_lx = LX;
  *_ly = LY;

  ierr = VecDestroy(vlx);CHKERRQ(ierr);
  ierr = VecDestroy(vly);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DACoordViewGnuplot2d"
static PetscErrorCode DACoordViewGnuplot2d(DA da,const char prefix[])
{
  DA             cda;
  Vec            coords;
  DACoor2d       **_coords;
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
    SETERRQ(PETSC_ERR_USER,"Cannot open file");
  }

  ierr = PetscFPrintf(PETSC_COMM_SELF,fp,"### Element geometry for processor %1.4d ### \n",rank);CHKERRQ(ierr);


  ierr = DAGetCoordinateDA(da,&cda);CHKERRQ(ierr);
  ierr = DAGetGhostedCoordinates(da,&coords);CHKERRQ(ierr);
  ierr = DAVecGetArray(cda,coords,&_coords);CHKERRQ(ierr);
  ierr = DAGetGhostCorners(cda,&si,&sj,0,&nx,&ny,0);CHKERRQ(ierr);
  for (j = sj; j < sj+ny-1; j++) {
    for (i = si; i < si+nx-1; i++) {
      ierr = PetscFPrintf(PETSC_COMM_SELF,fp,"%1.6e %1.6e \n",_coords[j][i].x,_coords[j][i].y);CHKERRQ(ierr);
      ierr = PetscFPrintf(PETSC_COMM_SELF,fp,"%1.6e %1.6e \n",_coords[j+1][i].x,_coords[j+1][i].y);CHKERRQ(ierr);
      ierr = PetscFPrintf(PETSC_COMM_SELF,fp,"%1.6e %1.6e \n",_coords[j+1][i+1].x,_coords[j+1][i+1].y);CHKERRQ(ierr);
      ierr = PetscFPrintf(PETSC_COMM_SELF,fp,"%1.6e %1.6e \n",_coords[j][i+1].x,_coords[j][i+1].y);CHKERRQ(ierr);
      ierr = PetscFPrintf(PETSC_COMM_SELF,fp,"%1.6e %1.6e \n\n",_coords[j][i].x,_coords[j][i].y);CHKERRQ(ierr);
    }
  }
  ierr = DAVecRestoreArray(cda,coords,&_coords);CHKERRQ(ierr);
  ierr = VecDestroy(coords);CHKERRQ(ierr);
  ierr = DADestroy(cda);CHKERRQ(ierr);

  ierr = PetscFClose(PETSC_COMM_SELF,fp);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DAViewGnuplot2d"
static PetscErrorCode DAViewGnuplot2d(DA da,Vec fields,const char comment[],const char prefix[])
{
  DA             cda;
  Vec            coords,local_fields;
  DACoor2d       **_coords;
  FILE           *fp;
  char           fname[PETSC_MAX_PATH_LEN],*field_name;
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
    SETERRQ(PETSC_ERR_USER,"Cannot open file");
  }

  ierr = PetscFPrintf(PETSC_COMM_SELF,fp,"### %s (processor %1.4d) ### \n",comment,rank);CHKERRQ(ierr);
  ierr = DAGetInfo(da,0,0,0,0,0,0,0,&n_dofs,0,0,0);CHKERRQ(ierr);
  ierr = PetscFPrintf(PETSC_COMM_SELF,fp,"### x y ");CHKERRQ(ierr);
  for (d = 0; d < n_dofs; d++) {
    ierr = DAGetFieldName(da,d,&field_name);CHKERRQ(ierr);
    ierr = PetscFPrintf(PETSC_COMM_SELF,fp,"%s ",field_name);CHKERRQ(ierr);
  }
  ierr = PetscFPrintf(PETSC_COMM_SELF,fp,"###\n");CHKERRQ(ierr);


  ierr = DAGetCoordinateDA(da,&cda);CHKERRQ(ierr);
  ierr = DAGetGhostedCoordinates(da,&coords);CHKERRQ(ierr);
  ierr = DAVecGetArray(cda,coords,&_coords);CHKERRQ(ierr);
  ierr = DAGetGhostCorners(cda,&si,&sj,0,&nx,&ny,0);CHKERRQ(ierr);

  ierr = DACreateLocalVector(da,&local_fields);CHKERRQ(ierr);
  ierr = DAGlobalToLocalBegin(da,fields,INSERT_VALUES,local_fields);CHKERRQ(ierr);
  ierr = DAGlobalToLocalEnd(da,fields,INSERT_VALUES,local_fields);CHKERRQ(ierr);
  ierr = VecGetArray(local_fields,&_fields);CHKERRQ(ierr);


  for (j = sj; j < sj+ny; j++) {
    for (i = si; i < si+nx; i++) {
      PetscScalar coord_x,coord_y;
      PetscScalar field_d;

      coord_x = _coords[j][i].x;
      coord_y = _coords[j][i].y;

      ierr = PetscFPrintf(PETSC_COMM_SELF,fp,"%1.6e %1.6e ",coord_x,coord_y);CHKERRQ(ierr);
      for (d = 0; d < n_dofs; d++) {
        field_d = _fields[ n_dofs*((i-si)+(j-sj)*(nx))+d ];
        ierr    = PetscFPrintf(PETSC_COMM_SELF,fp,"%1.6e ",field_d);CHKERRQ(ierr);
      }
      ierr = PetscFPrintf(PETSC_COMM_SELF,fp,"\n");CHKERRQ(ierr);
    }
  }
  ierr = VecRestoreArray(local_fields,&_fields);CHKERRQ(ierr);
  ierr = VecDestroy(local_fields);CHKERRQ(ierr);

  ierr = DAVecRestoreArray(cda,coords,&_coords);CHKERRQ(ierr);
  ierr = VecDestroy(coords);CHKERRQ(ierr);
  ierr = DADestroy(cda);CHKERRQ(ierr);

  ierr = PetscFClose(PETSC_COMM_SELF,fp);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DAViewCoefficientsGnuplot2d"
static PetscErrorCode DAViewCoefficientsGnuplot2d(DA da,Vec fields,const char comment[],const char prefix[])
{
  DA                     cda;
  Vec                    local_fields;
  FILE                   *fp;
  char                   fname[PETSC_MAX_PATH_LEN],*field_name;
  PetscMPIInt            rank;
  PetscInt               si,sj,nx,ny,i,j,p;
  PetscInt               n_dofs,d;
  GaussPointCoefficients **_coefficients;
  PetscErrorCode         ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
  ierr = PetscSNPrintf(fname,sizeof fname,"%s-p%1.4d.dat",prefix,rank);CHKERRQ(ierr);
  ierr = PetscFOpen(PETSC_COMM_SELF,fname,"w",&fp);CHKERRQ(ierr);
  if (fp == PETSC_NULL) {
    SETERRQ(PETSC_ERR_USER,"Cannot open file");
  }

  ierr = PetscFPrintf(PETSC_COMM_SELF,fp,"### %s (processor %1.4d) ### \n",comment,rank);CHKERRQ(ierr);
  ierr = DAGetInfo(da,0,0,0,0,0,0,0,&n_dofs,0,0,0);CHKERRQ(ierr);
  ierr = PetscFPrintf(PETSC_COMM_SELF,fp,"### x y ");CHKERRQ(ierr);
  for (d = 0; d < n_dofs; d++) {
    ierr = DAGetFieldName(da,d,&field_name);CHKERRQ(ierr);
    ierr = PetscFPrintf(PETSC_COMM_SELF,fp,"%s ",field_name);CHKERRQ(ierr);
  }
  ierr = PetscFPrintf(PETSC_COMM_SELF,fp,"###\n");CHKERRQ(ierr);


  ierr = DAGetCoordinateDA(da,&cda);CHKERRQ(ierr);
  ierr = DAGetGhostCorners(cda,&si,&sj,0,&nx,&ny,0);CHKERRQ(ierr);

  ierr = DACreateLocalVector(da,&local_fields);CHKERRQ(ierr);
  ierr = DAGlobalToLocalBegin(da,fields,INSERT_VALUES,local_fields);CHKERRQ(ierr);
  ierr = DAGlobalToLocalEnd(da,fields,INSERT_VALUES,local_fields);CHKERRQ(ierr);
  ierr = DAVecGetArray(da,local_fields,&_coefficients);CHKERRQ(ierr);


  for (j = sj; j < sj+ny; j++) {
    for (i = si; i < si+nx; i++) {
      PetscScalar coord_x,coord_y;

      for (p = 0; p < GAUSS_POINTS; p++) {
        coord_x = _coefficients[j][i].gp_coords[2*p];
        coord_y = _coefficients[j][i].gp_coords[2*p+1];

        ierr = PetscFPrintf(PETSC_COMM_SELF,fp,"%1.6e %1.6e ",coord_x,coord_y);CHKERRQ(ierr);

        ierr = PetscFPrintf(PETSC_COMM_SELF,fp,"%1.6e %1.6e %1.6e",_coefficients[j][i].eta[p],_coefficients[j][i].fx[p],_coefficients[j][i].fy[p]);CHKERRQ(ierr);
        ierr = PetscFPrintf(PETSC_COMM_SELF,fp,"\n");CHKERRQ(ierr);
      }
    }
  }
  ierr = DAVecRestoreArray(da,local_fields,&_coefficients);CHKERRQ(ierr);
  ierr = VecDestroy(local_fields);CHKERRQ(ierr);

  ierr = DADestroy(cda);CHKERRQ(ierr);

  ierr = PetscFClose(PETSC_COMM_SELF,fp);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


static PetscInt ASS_MAP_wIwDI_uJuDJ(
  PetscInt wi,PetscInt wd,PetscInt w_NPE,PetscInt w_dof,
  PetscInt ui,PetscInt ud,PetscInt u_NPE,PetscInt u_dof)
{
  PetscInt ij;
  PetscInt r,c,nr,nc;

  nr = w_NPE*w_dof;
  nc = u_NPE*u_dof;

  r = w_dof*wi+wd;
  c = u_dof*ui+ud;

  ij = r*nc+c;

  return ij;
}

static void FormStressOperatorQ1(PetscScalar Ke[],PetscScalar coords[],PetscScalar eta[])
{
  PetscInt    ngp;
  PetscScalar gp_xi[GAUSS_POINTS][2];
  PetscScalar gp_weight[GAUSS_POINTS];
  PetscInt    p,i,j,k;
  PetscScalar GNi_p[NSD][NODES_PER_EL],GNx_p[NSD][NODES_PER_EL];
  PetscScalar J_p,tildeD[3];
  PetscScalar B[3][U_DOFS*NODES_PER_EL];


  /* define quadrature rule */
  ConstructGaussQuadrature(&ngp,gp_xi,gp_weight);

  /* evaluate integral */
  for (p = 0; p < ngp; p++) {
    ConstructQ12D_GNi(gp_xi[p],GNi_p);
    ConstructQ12D_GNx(GNi_p,GNx_p,coords,&J_p);

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
          Ke[i+8*j] = Ke[i+8*j]+B[k][i]*tildeD[k]*B[k][j];
        }
      }
    }
  }
}

static void FormGradientOperatorQ1(PetscScalar Ke[],PetscScalar coords[])
{
  PetscInt    ngp;
  PetscScalar gp_xi[GAUSS_POINTS][2];
  PetscScalar gp_weight[GAUSS_POINTS];
  PetscInt    p,i,j,di;
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

    for (i = 0; i < NODES_PER_EL; i++) { /* u nodes */
      for (di = 0; di < NSD; di++) { /* u dofs */
        for (j = 0; j < 4; j++) {  /* p nodes, p dofs = 1 (ie no loop) */
          PetscInt IJ;
          /*     Ke[4*u_idx+j] = Ke[4*u_idx+j] - GNx_p[di][i] * Ni_p[j] * fac; */
          IJ = ASS_MAP_wIwDI_uJuDJ(i,di,NODES_PER_EL,2,j,0,NODES_PER_EL,1);

          Ke[IJ] = Ke[IJ]-GNx_p[di][i]*Ni_p[j]*fac;
        }
      }
    }
  }
}

static void FormDivergenceOperatorQ1(PetscScalar De[],PetscScalar coords[])
{
  PetscScalar Ge[U_DOFS*NODES_PER_EL*P_DOFS*NODES_PER_EL];
  PetscInt    i,j;
  PetscInt    nr_g,nc_g;

  PetscMemzero(Ge,sizeof(PetscScalar)*U_DOFS*NODES_PER_EL*P_DOFS*NODES_PER_EL);
  FormGradientOperatorQ1(Ge,coords);

  nr_g = U_DOFS*NODES_PER_EL;
  nc_g = P_DOFS*NODES_PER_EL;

  for (i = 0; i < nr_g; i++) {
    for (j = 0; j < nc_g; j++) {
      De[nr_g*j+i] = Ge[nc_g*i+j];
    }
  }
}

static void FormStabilisationOperatorQ1(PetscScalar Ke[],PetscScalar coords[],PetscScalar eta[])
{
  PetscInt    ngp;
  PetscScalar gp_xi[GAUSS_POINTS][2];
  PetscScalar gp_weight[GAUSS_POINTS];
  PetscInt    p,i,j;
  PetscScalar Ni_p[NODES_PER_EL];
  PetscScalar GNi_p[NSD][NODES_PER_EL],GNx_p[NSD][NODES_PER_EL];
  PetscScalar J_p,fac,eta_avg;


  /* define quadrature rule */
  ConstructGaussQuadrature(&ngp,gp_xi,gp_weight);

  /* evaluate integral */
  for (p = 0; p < ngp; p++) {
    ConstructQ12D_Ni(gp_xi[p],Ni_p);
    ConstructQ12D_GNi(gp_xi[p],GNi_p);
    ConstructQ12D_GNx(GNi_p,GNx_p,coords,&J_p);
    fac = gp_weight[p]*J_p;

    for (i = 0; i < NODES_PER_EL; i++) {
      for (j = 0; j < NODES_PER_EL; j++) {
        Ke[NODES_PER_EL*i+j] = Ke[NODES_PER_EL*i+j]-fac*(Ni_p[i]*Ni_p[j]-0.0625);
      }
    }
  }

  /* scale */
  eta_avg = 0.0;
  for (p = 0; p < ngp; p++) {
    eta_avg += eta[p];
  }
  eta_avg = (1.0/((PetscScalar)ngp))*eta_avg;
  fac     = 1.0/eta_avg;
  for (i = 0; i < NODES_PER_EL; i++) {
    for (j = 0; j < NODES_PER_EL; j++) {
      Ke[NODES_PER_EL*i+j] = fac*Ke[NODES_PER_EL*i+j];
    }
  }
}

static void FormScaledMassMatrixOperatorQ1(PetscScalar Ke[],PetscScalar coords[],PetscScalar eta[])
{
  PetscInt    ngp;
  PetscScalar gp_xi[GAUSS_POINTS][2];
  PetscScalar gp_weight[GAUSS_POINTS];
  PetscInt    p,i,j;
  PetscScalar Ni_p[NODES_PER_EL];
  PetscScalar GNi_p[NSD][NODES_PER_EL],GNx_p[NSD][NODES_PER_EL];
  PetscScalar J_p,fac,eta_avg;


  /* define quadrature rule */
  ConstructGaussQuadrature(&ngp,gp_xi,gp_weight);

  /* evaluate integral */
  for (p = 0; p < ngp; p++) {
    ConstructQ12D_Ni(gp_xi[p],Ni_p);
    ConstructQ12D_GNi(gp_xi[p],GNi_p);
    ConstructQ12D_GNx(GNi_p,GNx_p,coords,&J_p);
    fac = gp_weight[p]*J_p;

    for (i = 0; i < NODES_PER_EL; i++) {
      for (j = 0; j < NODES_PER_EL; j++) {
        Ke[NODES_PER_EL*i+j] = Ke[NODES_PER_EL*i+j]-fac*Ni_p[i]*Ni_p[j];
      }
    }
  }

  /* scale */
  eta_avg = 0.0;
  for (p = 0; p < ngp; p++) {
    eta_avg += eta[p];
  }
  eta_avg = (1.0/((PetscScalar)ngp))*eta_avg;
  fac     = 1.0/eta_avg;
  for (i = 0; i < NODES_PER_EL; i++) {
    for (j = 0; j < NODES_PER_EL; j++) {
      Ke[NODES_PER_EL*i+j] = fac*Ke[NODES_PER_EL*i+j];
    }
  }
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

#undef __FUNCT__  
#define __FUNCT__ "GetElementCoords"
static PetscErrorCode GetElementCoords(DACoor2d **_coords,PetscInt ei,PetscInt ej,PetscScalar el_coords[])
{
  PetscFunctionBegin;
  /* get coords for the element */
  el_coords[NSD*0+0] = _coords[ej][ei].x;el_coords[NSD*0+1] = _coords[ej][ei].y;
  el_coords[NSD*1+0] = _coords[ej+1][ei].x;el_coords[NSD*1+1] = _coords[ej+1][ei].y;
  el_coords[NSD*2+0] = _coords[ej+1][ei+1].x;el_coords[NSD*2+1] = _coords[ej+1][ei+1].y;
  el_coords[NSD*3+0] = _coords[ej][ei+1].x;el_coords[NSD*3+1] = _coords[ej][ei+1].y;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "AssembleA_Stokes"
static PetscErrorCode AssembleA_Stokes(Mat A,DA stokes_da,DA properties_da,Vec properties)
{
  DA                     cda;
  Vec                    coords;
  DACoor2d               **_coords;
  MatStencil             u_eqn[NODES_PER_EL*U_DOFS]; /* 2 degrees of freedom */
  MatStencil             p_eqn[NODES_PER_EL*P_DOFS]; /* 1 degrees of freedom */
  PetscInt               sex,sey,mx,my;
  PetscInt               ei,ej;
  PetscScalar            Ae[NODES_PER_EL*U_DOFS*NODES_PER_EL*U_DOFS];
  PetscScalar            Ge[NODES_PER_EL*U_DOFS*NODES_PER_EL*P_DOFS];
  PetscScalar            De[NODES_PER_EL*P_DOFS*NODES_PER_EL*U_DOFS];
  PetscScalar            Ce[NODES_PER_EL*P_DOFS*NODES_PER_EL*P_DOFS];
  PetscScalar            el_coords[NODES_PER_EL*NSD];
  Vec                    local_properties;
  GaussPointCoefficients **props;
  PetscScalar            *prop_eta;
  PetscErrorCode         ierr;

  PetscFunctionBegin;

  /* setup for coords */
  ierr = DAGetCoordinateDA(stokes_da,&cda);CHKERRQ(ierr);
  ierr = DAGetGhostedCoordinates(stokes_da,&coords);CHKERRQ(ierr);
  ierr = DAVecGetArray(cda,coords,&_coords);CHKERRQ(ierr);

  /* setup for coefficients */
  ierr = DACreateLocalVector(properties_da,&local_properties);CHKERRQ(ierr);
  ierr = DAGlobalToLocalBegin(properties_da,properties,INSERT_VALUES,local_properties);CHKERRQ(ierr);
  ierr = DAGlobalToLocalEnd(properties_da,properties,INSERT_VALUES,local_properties);CHKERRQ(ierr);
  ierr = DAVecGetArray(properties_da,local_properties,&props);CHKERRQ(ierr);

  ierr = DAGetElementCorners(stokes_da,&sex,&sey,0,&mx,&my,0);CHKERRQ(ierr);
  for (ej = sey; ej < sey+my; ej++) {
    for (ei = sex; ei < sex+mx; ei++) {
      /* get coords for the element */
      GetElementCoords(_coords,ei,ej,el_coords);

      /* get coefficients for the element */
      prop_eta = props[ej][ei].eta;

      /* initialise element stiffness matrix */
      ierr = PetscMemzero(Ae,sizeof(PetscScalar)*NODES_PER_EL*U_DOFS*NODES_PER_EL*U_DOFS);CHKERRQ(ierr);
      ierr = PetscMemzero(Ge,sizeof(PetscScalar)*NODES_PER_EL*U_DOFS*NODES_PER_EL*P_DOFS);CHKERRQ(ierr);
      ierr = PetscMemzero(De,sizeof(PetscScalar)*NODES_PER_EL*P_DOFS*NODES_PER_EL*U_DOFS);CHKERRQ(ierr);
      ierr = PetscMemzero(Ce,sizeof(PetscScalar)*NODES_PER_EL*P_DOFS*NODES_PER_EL*P_DOFS);CHKERRQ(ierr);

      /* form element stiffness matrix */
      FormStressOperatorQ1(Ae,el_coords,prop_eta);
      FormGradientOperatorQ1(Ge,el_coords);
      FormDivergenceOperatorQ1(De,el_coords);
      FormStabilisationOperatorQ1(Ce,el_coords,prop_eta);

      /* insert element matrix into global matrix */
      ierr = DAGetElementEqnums_up(u_eqn,p_eqn,ei,ej);CHKERRQ(ierr);
      ierr = MatSetValuesStencil(A,NODES_PER_EL*U_DOFS,u_eqn,NODES_PER_EL*U_DOFS,u_eqn,Ae,ADD_VALUES);CHKERRQ(ierr);
      ierr = MatSetValuesStencil(A,NODES_PER_EL*U_DOFS,u_eqn,NODES_PER_EL*P_DOFS,p_eqn,Ge,ADD_VALUES);CHKERRQ(ierr);
      ierr = MatSetValuesStencil(A,NODES_PER_EL*P_DOFS,p_eqn,NODES_PER_EL*U_DOFS,u_eqn,De,ADD_VALUES);CHKERRQ(ierr);
      ierr = MatSetValuesStencil(A,NODES_PER_EL*P_DOFS,p_eqn,NODES_PER_EL*P_DOFS,p_eqn,Ce,ADD_VALUES);CHKERRQ(ierr);
    }
  }
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  ierr = DAVecRestoreArray(cda,coords,&_coords);CHKERRQ(ierr);
  ierr = VecDestroy(coords);CHKERRQ(ierr);
  ierr = DADestroy(cda);CHKERRQ(ierr);

  ierr = DAVecRestoreArray(properties_da,local_properties,&props);CHKERRQ(ierr);
  ierr = VecDestroy(local_properties);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "AssembleA_PCStokes"
static PetscErrorCode AssembleA_PCStokes(Mat A,DA stokes_da,DA properties_da,Vec properties)
{
  DA                     cda;
  Vec                    coords;
  DACoor2d               **_coords;
  MatStencil             u_eqn[NODES_PER_EL*U_DOFS]; /* 2 degrees of freedom */
  MatStencil             p_eqn[NODES_PER_EL*P_DOFS]; /* 1 degrees of freedom */
  PetscInt               sex,sey,mx,my;
  PetscInt               ei,ej;
  PetscScalar            Ae[NODES_PER_EL*U_DOFS*NODES_PER_EL*U_DOFS];
  PetscScalar            Ge[NODES_PER_EL*U_DOFS*NODES_PER_EL*P_DOFS];
  PetscScalar            De[NODES_PER_EL*P_DOFS*NODES_PER_EL*U_DOFS];
  PetscScalar            Ce[NODES_PER_EL*P_DOFS*NODES_PER_EL*P_DOFS];
  PetscScalar            el_coords[NODES_PER_EL*NSD];
  Vec                    local_properties;
  GaussPointCoefficients **props;
  PetscScalar            *prop_eta;
  PetscErrorCode         ierr;

  PetscFunctionBegin;
  /* setup for coords */
  ierr = DAGetCoordinateDA(stokes_da,&cda);CHKERRQ(ierr);
  ierr = DAGetGhostedCoordinates(stokes_da,&coords);CHKERRQ(ierr);
  ierr = DAVecGetArray(cda,coords,&_coords);CHKERRQ(ierr);

  /* setup for coefficients */
  ierr = DACreateLocalVector(properties_da,&local_properties);CHKERRQ(ierr);
  ierr = DAGlobalToLocalBegin(properties_da,properties,INSERT_VALUES,local_properties);CHKERRQ(ierr);
  ierr = DAGlobalToLocalEnd(properties_da,properties,INSERT_VALUES,local_properties);CHKERRQ(ierr);
  ierr = DAVecGetArray(properties_da,local_properties,&props);CHKERRQ(ierr);

  ierr = DAGetElementCorners(stokes_da,&sex,&sey,0,&mx,&my,0);CHKERRQ(ierr);
  for (ej = sey; ej < sey+my; ej++) {
    for (ei = sex; ei < sex+mx; ei++) {
      /* get coords for the element */
      GetElementCoords(_coords,ei,ej,el_coords);

      /* get coefficients for the element */
      prop_eta = props[ej][ei].eta;

      /* initialise element stiffness matrix */
      ierr = PetscMemzero(Ae,sizeof(PetscScalar)*NODES_PER_EL*U_DOFS*NODES_PER_EL*U_DOFS);CHKERRQ(ierr);
      ierr = PetscMemzero(Ge,sizeof(PetscScalar)*NODES_PER_EL*U_DOFS*NODES_PER_EL*P_DOFS);CHKERRQ(ierr);
      ierr = PetscMemzero(De,sizeof(PetscScalar)*NODES_PER_EL*P_DOFS*NODES_PER_EL*U_DOFS);CHKERRQ(ierr);
      ierr = PetscMemzero(Ce,sizeof(PetscScalar)*NODES_PER_EL*P_DOFS*NODES_PER_EL*P_DOFS);CHKERRQ(ierr);


      /* form element stiffness matrix */
      FormStressOperatorQ1(Ae,el_coords,prop_eta);
      FormGradientOperatorQ1(Ge,el_coords);
      /*               FormDivergenceOperatorQ1( De, el_coords ); */
      FormScaledMassMatrixOperatorQ1(Ce,el_coords,prop_eta);

      /* insert element matrix into global matrix */
      ierr = DAGetElementEqnums_up(u_eqn,p_eqn,ei,ej);CHKERRQ(ierr);
      ierr = MatSetValuesStencil(A,NODES_PER_EL*U_DOFS,u_eqn,NODES_PER_EL*U_DOFS,u_eqn,Ae,ADD_VALUES);CHKERRQ(ierr);
      ierr = MatSetValuesStencil(A,NODES_PER_EL*U_DOFS,u_eqn,NODES_PER_EL*P_DOFS,p_eqn,Ge,ADD_VALUES);CHKERRQ(ierr);
      /*     MatSetValuesStencil( A, NODES_PER_EL*P_DOFS,p_eqn, NODES_PER_EL*U_DOFS,u_eqn, De, ADD_VALUES ); */
      ierr = MatSetValuesStencil(A,NODES_PER_EL*P_DOFS,p_eqn,NODES_PER_EL*P_DOFS,p_eqn,Ce,ADD_VALUES);CHKERRQ(ierr);
    }
  }
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  ierr = DAVecRestoreArray(cda,coords,&_coords);CHKERRQ(ierr);
  ierr = VecDestroy(coords);CHKERRQ(ierr);
  ierr = DADestroy(cda);CHKERRQ(ierr);

  ierr = DAVecRestoreArray(properties_da,local_properties,&props);CHKERRQ(ierr);
  ierr = VecDestroy(local_properties);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DASetValuesLocalStencil_ADD_VALUES"
static PetscErrorCode DASetValuesLocalStencil_ADD_VALUES(StokesDOF **fields_F,MatStencil u_eqn[],MatStencil p_eqn[],PetscScalar Fe_u[],PetscScalar Fe_p[])
{
  PetscInt n;

  PetscFunctionBegin;
  for (n = 0; n < 4; n++) {
    fields_F[ u_eqn[2*n  ].j ][ u_eqn[2*n  ].i ].u_dof = fields_F[ u_eqn[2*n  ].j ][ u_eqn[2*n  ].i ].u_dof+Fe_u[2*n  ];
    fields_F[ u_eqn[2*n+1].j ][ u_eqn[2*n+1].i ].v_dof = fields_F[ u_eqn[2*n+1].j ][ u_eqn[2*n+1].i ].v_dof+Fe_u[2*n+1];
    fields_F[ p_eqn[n].j     ][ p_eqn[n].i     ].p_dof = fields_F[ p_eqn[n].j     ][ p_eqn[n].i     ].p_dof+Fe_p[n    ];
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "AssembleF_Stokes"
static PetscErrorCode AssembleF_Stokes(Vec F,DA stokes_da,DA properties_da,Vec properties)
{
  DA                     cda;
  Vec                    coords;
  DACoor2d               **_coords;
  MatStencil             u_eqn[NODES_PER_EL*U_DOFS]; /* 2 degrees of freedom */
  MatStencil             p_eqn[NODES_PER_EL*P_DOFS]; /* 1 degrees of freedom */
  PetscInt               sex,sey,mx,my;
  PetscInt               ei,ej;
  PetscScalar            Fe[NODES_PER_EL*U_DOFS];
  PetscScalar            He[NODES_PER_EL*P_DOFS];
  PetscScalar            el_coords[NODES_PER_EL*NSD];
  Vec                    local_properties;
  GaussPointCoefficients **props;
  PetscScalar            *prop_fx,*prop_fy;
  Vec                    local_F;
  StokesDOF              **ff;
  PetscErrorCode         ierr;

  PetscFunctionBegin;
  /* setup for coords */
  ierr = DAGetCoordinateDA(stokes_da,&cda);CHKERRQ(ierr);
  ierr = DAGetGhostedCoordinates(stokes_da,&coords);CHKERRQ(ierr);
  ierr = DAVecGetArray(cda,coords,&_coords);CHKERRQ(ierr);

  /* setup for coefficients */
  ierr = DAGetLocalVector(properties_da,&local_properties);CHKERRQ(ierr);
  ierr = DAGlobalToLocalBegin(properties_da,properties,INSERT_VALUES,local_properties);CHKERRQ(ierr);
  ierr = DAGlobalToLocalEnd(properties_da,properties,INSERT_VALUES,local_properties);CHKERRQ(ierr);
  ierr = DAVecGetArray(properties_da,local_properties,&props);CHKERRQ(ierr);

  /* get acces to the vector */
  ierr = DAGetLocalVector(stokes_da,&local_F);CHKERRQ(ierr);
  ierr = VecZeroEntries(local_F);CHKERRQ(ierr);
  ierr = DAVecGetArray(stokes_da,local_F,&ff);CHKERRQ(ierr);


  ierr = DAGetElementCorners(stokes_da,&sex,&sey,0,&mx,&my,0);CHKERRQ(ierr);
  for (ej = sey; ej < sey+my; ej++) {
    for (ei = sex; ei < sex+mx; ei++) {
      /* get coords for the element */
      GetElementCoords(_coords,ei,ej,el_coords);

      /* get coefficients for the element */
      prop_fx = props[ej][ei].fx;
      prop_fy = props[ej][ei].fy;

      /* initialise element stiffness matrix */
      ierr = PetscMemzero(Fe,sizeof(PetscScalar)*NODES_PER_EL*U_DOFS);CHKERRQ(ierr);
      ierr = PetscMemzero(He,sizeof(PetscScalar)*NODES_PER_EL*P_DOFS);CHKERRQ(ierr);


      /* form element stiffness matrix */
      FormMomentumRhsQ1(Fe,el_coords,prop_fx,prop_fy);

      /* insert element matrix into global matrix */
      ierr = DAGetElementEqnums_up(u_eqn,p_eqn,ei,ej);CHKERRQ(ierr);

      ierr = DASetValuesLocalStencil_ADD_VALUES(ff,u_eqn,p_eqn,Fe,He);CHKERRQ(ierr);
    }
  }

  ierr = DAVecRestoreArray(stokes_da,local_F,&ff);CHKERRQ(ierr);
  ierr = DALocalToGlobalBegin(stokes_da,local_F,F);CHKERRQ(ierr);
  ierr = DALocalToGlobalEnd(stokes_da,local_F,F);CHKERRQ(ierr);
  ierr = DARestoreLocalVector(stokes_da,&local_F);CHKERRQ(ierr);


  ierr = DAVecRestoreArray(cda,coords,&_coords);CHKERRQ(ierr);
  ierr = VecDestroy(coords);CHKERRQ(ierr);
  ierr = DADestroy(cda);CHKERRQ(ierr);

  ierr = DAVecRestoreArray(properties_da,local_properties,&props);CHKERRQ(ierr);
  ierr = DARestoreLocalVector(properties_da,&local_properties);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DACreateSolCx"
static PetscErrorCode DACreateSolCx(PetscScalar eta0,PetscScalar eta1,PetscScalar xc,PetscInt nz,
                                    PetscInt mx,PetscInt my,
                                    DA *_da,Vec *_X)
{
  DA             da,cda;
  Vec            X,local_X;
  StokesDOF      **_stokes;
  Vec            coords;
  DACoor2d       **_coords;
  PetscInt       si,sj,ei,ej,i,j;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DACreate2d(PETSC_COMM_WORLD,DA_NONPERIODIC,DA_STENCIL_BOX,
                    mx+1,my+1,PETSC_DECIDE,PETSC_DECIDE,3,1,PETSC_NULL,PETSC_NULL,&da);CHKERRQ(ierr);
  ierr = DASetFieldName(da,0,"anlytic_Vx");CHKERRQ(ierr);
  ierr = DASetFieldName(da,1,"anlytic_Vy");CHKERRQ(ierr);
  ierr = DASetFieldName(da,2,"analytic_P");CHKERRQ(ierr);


  ierr = DASetUniformCoordinates(da,0.0,1.0,0.0,1.0,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);


  ierr = DAGetGhostedCoordinates(da,&coords);CHKERRQ(ierr);
  ierr = DAGetCoordinateDA(da,&cda);CHKERRQ(ierr);
  ierr = DAVecGetArray(cda,coords,&_coords);CHKERRQ(ierr);

  ierr = DACreateGlobalVector(da,&X);CHKERRQ(ierr);
  ierr = DACreateLocalVector(da,&local_X);CHKERRQ(ierr);
  ierr = DAVecGetArray(da,local_X,&_stokes);CHKERRQ(ierr);

  ierr = DAGetGhostCorners(da,&si,&sj,0,&ei,&ej,0);CHKERRQ(ierr);
  for (j = sj; j < sj+ej; j++) {
    for (i = si; i < si+ei; i++) {
      double pos[2],pressure,vel[2],total_stress[3],strain_rate[3];

      pos[0] = _coords[j][i].x;
      pos[1] = _coords[j][i].y;

      evaluate_solCx(pos,eta0,eta1,xc,nz,vel,&pressure,total_stress,strain_rate);

      _stokes[j][i].u_dof = vel[0];
      _stokes[j][i].v_dof = vel[1];
      _stokes[j][i].p_dof = pressure;
    }
  }
  ierr = DAVecRestoreArray(da,local_X,&_stokes);CHKERRQ(ierr);
  ierr = DAVecRestoreArray(cda,coords,&_coords);CHKERRQ(ierr);

  ierr = DALocalToGlobal(da,local_X,INSERT_VALUES,X);CHKERRQ(ierr);

  ierr = DADestroy(cda);CHKERRQ(ierr);
  ierr = VecDestroy(local_X);CHKERRQ(ierr);
  ierr = VecDestroy(coords);CHKERRQ(ierr);

  *_da = da;
  *_X  = X;

  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "StokesDAGetNodalFields"
static PetscErrorCode StokesDAGetNodalFields(StokesDOF **fields,PetscInt ei,PetscInt ej,StokesDOF nodal_fields[])
{
  PetscFunctionBegin;
  /* get the nodal fields */
  nodal_fields[0].u_dof = fields[ej  ][ei  ].u_dof;nodal_fields[0].v_dof = fields[ej  ][ei  ].v_dof;nodal_fields[0].p_dof = fields[ej  ][ei  ].p_dof;
  nodal_fields[1].u_dof = fields[ej+1][ei  ].u_dof;nodal_fields[1].v_dof = fields[ej+1][ei  ].v_dof;nodal_fields[1].p_dof = fields[ej+1][ei  ].p_dof;
  nodal_fields[2].u_dof = fields[ej+1][ei+1].u_dof;nodal_fields[2].v_dof = fields[ej+1][ei+1].v_dof;nodal_fields[2].p_dof = fields[ej+1][ei+1].p_dof;
  nodal_fields[3].u_dof = fields[ej  ][ei+1].u_dof;nodal_fields[3].v_dof = fields[ej  ][ei+1].v_dof;nodal_fields[3].p_dof = fields[ej  ][ei+1].p_dof;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DAIntegrateErrors"
static PetscErrorCode DAIntegrateErrors(DA stokes_da,Vec X,Vec X_analytic)
{
  DA          cda;
  Vec         coords,X_analytic_local,X_local;
  DACoor2d    **_coords;
  PetscInt    sex,sey,mx,my;
  PetscInt    ei,ej;
  PetscScalar el_coords[NODES_PER_EL*NSD];
  StokesDOF   **stokes_analytic,**stokes;
  StokesDOF   stokes_analytic_e[4],stokes_e[4];

  PetscScalar    GNi_p[NSD][NODES_PER_EL],GNx_p[NSD][NODES_PER_EL];
  PetscScalar    Ni_p[NODES_PER_EL];
  PetscInt       ngp;
  PetscScalar    gp_xi[GAUSS_POINTS][2];
  PetscScalar    gp_weight[GAUSS_POINTS];
  PetscInt       p,i;
  PetscScalar    J_p,fac;
  PetscScalar    h,p_e_L2,u_e_L2,u_e_H1,p_L2,u_L2,u_H1;
  PetscInt       M;
  PetscReal      xymin[2],xymax[2];
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* define quadrature rule */
  ConstructGaussQuadrature(&ngp,gp_xi,gp_weight);

  /* setup for coords */
  ierr = DAGetCoordinateDA(stokes_da,&cda);CHKERRQ(ierr);
  ierr = DAGetGhostedCoordinates(stokes_da,&coords);CHKERRQ(ierr);
  ierr = DAVecGetArray(cda,coords,&_coords);CHKERRQ(ierr);

  /* setup for analytic */
  ierr = DACreateLocalVector(stokes_da,&X_analytic_local);CHKERRQ(ierr);
  ierr = DAGlobalToLocalBegin(stokes_da,X_analytic,INSERT_VALUES,X_analytic_local);CHKERRQ(ierr);
  ierr = DAGlobalToLocalEnd(stokes_da,X_analytic,INSERT_VALUES,X_analytic_local);CHKERRQ(ierr);
  ierr = DAVecGetArray(stokes_da,X_analytic_local,&stokes_analytic);CHKERRQ(ierr);

  /* setup for solution */
  ierr = DACreateLocalVector(stokes_da,&X_local);CHKERRQ(ierr);
  ierr = DAGlobalToLocalBegin(stokes_da,X,INSERT_VALUES,X_local);CHKERRQ(ierr);
  ierr = DAGlobalToLocalEnd(stokes_da,X,INSERT_VALUES,X_local);CHKERRQ(ierr);
  ierr = DAVecGetArray(stokes_da,X_local,&stokes);CHKERRQ(ierr);

  ierr = DAGetInfo(stokes_da,0,&M,0,0,0,0,0,0,0,0,0);CHKERRQ(ierr);
  ierr = DAGetBoundingBox(stokes_da,xymin,xymax);CHKERRQ(ierr);

  h = (xymax[0]-xymin[0])/((double)M);

  p_L2 = u_L2 = u_H1 = 0.0;

  ierr = DAGetElementCorners(stokes_da,&sex,&sey,0,&mx,&my,0);CHKERRQ(ierr);
  for (ej = sey; ej < sey+my; ej++) {
    for (ei = sex; ei < sex+mx; ei++) {
      /* get coords for the element */
      ierr = GetElementCoords(_coords,ei,ej,el_coords);CHKERRQ(ierr);
      ierr = StokesDAGetNodalFields(stokes,ei,ej,stokes_e);CHKERRQ(ierr);
      ierr = StokesDAGetNodalFields(stokes_analytic,ei,ej,stokes_analytic_e);CHKERRQ(ierr);

      /* evaluate integral */
      p_e_L2 = 0.0;
      u_e_L2 = 0.0;
      u_e_H1 = 0.0;
      for (p = 0; p < ngp; p++) {
        ConstructQ12D_Ni(gp_xi[p],Ni_p);
        ConstructQ12D_GNi(gp_xi[p],GNi_p);
        ConstructQ12D_GNx(GNi_p,GNx_p,el_coords,&J_p);
        fac = gp_weight[p]*J_p;

        for (i = 0; i < NODES_PER_EL; i++) {
          PetscScalar u_error,v_error;

          p_e_L2 = p_e_L2+fac*Ni_p[i]*(stokes_e[i].p_dof-stokes_analytic_e[i].p_dof)*(stokes_e[i].p_dof-stokes_analytic_e[i].p_dof);

          u_error = stokes_e[i].u_dof-stokes_analytic_e[i].u_dof;
          v_error = stokes_e[i].v_dof-stokes_analytic_e[i].v_dof;
          u_L2    = u_L2+fac*Ni_p[i]*(u_error*u_error+v_error*v_error);

          u_e_H1 = u_e_H1+fac*(GNx_p[0][i]*u_error*GNx_p[0][i]*u_error              /* du/dx */
                               +GNx_p[1][i]*u_error*GNx_p[1][i]*u_error               /* du/dy */
                               +GNx_p[0][i]*v_error*GNx_p[0][i]*v_error               /* dv/dx */
                               +GNx_p[1][i]*v_error*GNx_p[1][i]*v_error);             /* dv/dy */
        }
      }

      p_L2 = p_L2+p_e_L2;
      u_L2 = u_L2+u_e_L2;
      u_H1 = u_H1+u_e_H1;
    }
  }
  p_L2 = sqrt(p_L2);
  u_L2 = sqrt(u_L2);
  u_H1 = sqrt(u_H1);


  ierr = PetscPrintf(PETSC_COMM_WORLD,"%1.4e   %1.4e   %1.4e   %1.4e \n",h,p_L2,u_L2,u_H1);CHKERRQ(ierr);


  ierr = DAVecRestoreArray(cda,coords,&_coords);CHKERRQ(ierr);
  ierr = VecDestroy(coords);CHKERRQ(ierr);
  ierr = DADestroy(cda);CHKERRQ(ierr);

  ierr = DAVecRestoreArray(stokes_da,X_analytic_local,&stokes_analytic);CHKERRQ(ierr);
  ierr = VecDestroy(X_analytic_local);CHKERRQ(ierr);
  ierr = DAVecRestoreArray(stokes_da,X_local,&stokes);CHKERRQ(ierr);
  ierr = VecDestroy(X_local);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "solve_stokes_2d_coupled"
static PetscErrorCode solve_stokes_2d_coupled(PetscInt mx,PetscInt my)
{
  DA                     da_Stokes,da_prop;
  PetscInt               u_dof,p_dof,dof,stencil_width;
  Mat                    A,B;
  PetscInt               mxl,myl;
  DA                     prop_cda,vel_cda;
  Vec                    prop_coords,vel_coords;
  PetscInt               si,sj,nx,ny,i,j,p;
  Vec                    f,X;
  PetscInt               prop_dof,prop_stencil_width;
  Vec                    properties,l_properties;
  PetscScalar            dx,dy;
  PetscInt               M,N;
  DACoor2d               **_prop_coords,**_vel_coords;
  GaussPointCoefficients **element_props;
  PetscInt               its;
  KSP                    ksp_S;
  PetscInt               coefficient_structure = 0;
  PetscInt               cpu_x,cpu_y,*lx = PETSC_NULL,*ly = PETSC_NULL;
  PetscTruth             use_gp_coords = PETSC_FALSE;
  PetscErrorCode         ierr;

  PetscFunctionBegin;
  /* Generate the da for velocity and pressure */
  /*
  We use Q1 elements for the temperature.
  FEM has a 9-point stencil (BOX) or connectivity pattern
  Num nodes in each direction is mx+1, my+1
  */
  u_dof         = U_DOFS; /* Vx, Vy - velocities */
  p_dof         = P_DOFS; /* p - pressure */
  dof           = u_dof+p_dof;
  stencil_width = 1;
  ierr          = DACreate2d(PETSC_COMM_WORLD,DA_NONPERIODIC,DA_STENCIL_BOX,
                             mx+1,my+1,PETSC_DECIDE,PETSC_DECIDE,dof,stencil_width,PETSC_NULL,PETSC_NULL,&da_Stokes);CHKERRQ(ierr);
  ierr = DASetFieldName(da_Stokes,0,"Vx");CHKERRQ(ierr);
  ierr = DASetFieldName(da_Stokes,1,"Vy");CHKERRQ(ierr);
  ierr = DASetFieldName(da_Stokes,2,"P");CHKERRQ(ierr);

  /* unit box [0,1] x [0,1] */
  ierr = DASetUniformCoordinates(da_Stokes,0.0,1.0,0.0,1.0,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);


  /* Generate element properties, we will assume all material properties are constant over the element */
  /* local number of elements */
  ierr = DAGetLocalElementSize(da_Stokes,&mxl,&myl,PETSC_NULL);CHKERRQ(ierr);

  /* !!! IN PARALLEL WE MUST MAKE SURE THE TWO DA's ALIGN !!! // */
  ierr = DAGetInfo(da_Stokes,0,0,0,0,&cpu_x,&cpu_y,0,0,0,0,0);CHKERRQ(ierr);
  ierr = DAGetElementOwnershipRanges2d(da_Stokes,&lx,&ly);CHKERRQ(ierr);

  prop_dof           = (int)(sizeof(GaussPointCoefficients)/sizeof(PetscScalar)); /* gauss point setup */
  prop_stencil_width = 0;
  ierr               = DACreate2d(PETSC_COMM_WORLD,DA_NONPERIODIC,DA_STENCIL_BOX,
                                  mx,my,cpu_x,cpu_y,prop_dof,prop_stencil_width,lx,ly,&da_prop);CHKERRQ(ierr);
  ierr = PetscFree(lx);CHKERRQ(ierr);
  ierr = PetscFree(ly);CHKERRQ(ierr);

  /* define centroid positions */
  ierr = DAGetInfo(da_prop,0,&M,&N,0,0,0,0,0,0,0,0);CHKERRQ(ierr);
  dx   = 1.0/((PetscScalar)(M));
  dy   = 1.0/((PetscScalar)(N));

  ierr = DASetUniformCoordinates(da_prop,0.0+0.5*dx,1.0-0.5*dx,0.0+0.5*dx,1.0-0.5*dx,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);

  /* define coefficients */
  ierr = PetscOptionsGetInt(PETSC_NULL,"-c_str",&coefficient_structure,PETSC_NULL);CHKERRQ(ierr);
  /*     PetscPrintf( PETSC_COMM_WORLD, "Using coeficient structure %D \n", coefficient_structure ); */

  ierr = DACreateGlobalVector(da_prop,&properties);CHKERRQ(ierr);
  ierr = DACreateLocalVector(da_prop,&l_properties);CHKERRQ(ierr);
  ierr = DAVecGetArray(da_prop,l_properties,&element_props);CHKERRQ(ierr);

  ierr = DAGetCoordinateDA(da_prop,&prop_cda);CHKERRQ(ierr);
  ierr = DAGetGhostedCoordinates(da_prop,&prop_coords);CHKERRQ(ierr);
  ierr = DAVecGetArray(prop_cda,prop_coords,&_prop_coords);CHKERRQ(ierr);

  ierr = DAGetGhostCorners(prop_cda,&si,&sj,0,&nx,&ny,0);CHKERRQ(ierr);

  ierr = DAGetCoordinateDA(da_Stokes,&vel_cda);CHKERRQ(ierr);
  ierr = DAGetGhostedCoordinates(da_Stokes,&vel_coords);CHKERRQ(ierr);
  ierr = DAVecGetArray(vel_cda,vel_coords,&_vel_coords);CHKERRQ(ierr);


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
  ierr = PetscOptionsGetTruth(PETSC_NULL,"-use_gp_coords",&use_gp_coords,0);CHKERRQ(ierr);

  for (j = sj; j < sj+ny; j++) {
    for (i = si; i < si+nx; i++) {
      PetscScalar centroid_x = _prop_coords[j][i].x; /* centroids of cell */
      PetscScalar centroid_y = _prop_coords[j][i].y;
      PetscScalar coord_x,coord_y;

      if (coefficient_structure == 0) {
        PetscScalar opts_eta0,opts_eta1,opts_xc;
        PetscInt    opts_nz;

        opts_eta0 = 1.0;
        opts_eta1 = 1.0;
        opts_xc   = 0.5;
        opts_nz   = 1;

        ierr = PetscOptionsGetScalar(PETSC_NULL,"-solcx_eta0",&opts_eta0,0);CHKERRQ(ierr);
        ierr = PetscOptionsGetScalar(PETSC_NULL,"-solcx_eta1",&opts_eta1,0);CHKERRQ(ierr);
        ierr = PetscOptionsGetScalar(PETSC_NULL,"-solcx_xc",&opts_xc,0);CHKERRQ(ierr);
        ierr = PetscOptionsGetInt(PETSC_NULL,"-solcx_nz",&opts_nz,0);CHKERRQ(ierr);

        for (p = 0; p < GAUSS_POINTS; p++) {
          coord_x = centroid_x;
          coord_y = centroid_y;
          if (use_gp_coords == PETSC_TRUE) {
            coord_x = element_props[j][i].gp_coords[2*p];
            coord_y = element_props[j][i].gp_coords[2*p+1];
          }


          element_props[j][i].eta[p] = opts_eta0;
          if (coord_x > opts_xc) {
            element_props[j][i].eta[p] = opts_eta1;
          }

          element_props[j][i].fx[p] = 0.0;
          element_props[j][i].fy[p] = sin((double)opts_nz*M_PI*coord_y)*cos(1.0*M_PI*coord_x);
        }
      } else if (coefficient_structure == 1) { /* square sinker */
        PetscScalar opts_eta0,opts_eta1,opts_dx,opts_dy;

        opts_eta0 = 1.0;
        opts_eta1 = 1.0;
        opts_dx   = 0.50;
        opts_dy   = 0.50;

        ierr = PetscOptionsGetScalar(PETSC_NULL,"-sinker_eta0",&opts_eta0,0);CHKERRQ(ierr);
        ierr = PetscOptionsGetScalar(PETSC_NULL,"-sinker_eta1",&opts_eta1,0);CHKERRQ(ierr);
        ierr = PetscOptionsGetScalar(PETSC_NULL,"-sinker_dx",&opts_dx,0);CHKERRQ(ierr);
        ierr = PetscOptionsGetScalar(PETSC_NULL,"-sinker_dy",&opts_dy,0);CHKERRQ(ierr);


        for (p = 0; p < GAUSS_POINTS; p++) {
          coord_x = centroid_x;
          coord_y = centroid_y;
          if (use_gp_coords == PETSC_TRUE) {
            coord_x = element_props[j][i].gp_coords[2*p];
            coord_y = element_props[j][i].gp_coords[2*p+1];
          }

          element_props[j][i].eta[p] = opts_eta0;
          element_props[j][i].fx[p]  = 0.0;
          element_props[j][i].fy[p]  = 0.0;

          if ((coord_x > -0.5*opts_dx+0.5) && (coord_x < 0.5*opts_dx+0.5)) {
            if ((coord_y > -0.5*opts_dy+0.5) && (coord_y < 0.5*opts_dy+0.5)) {
              element_props[j][i].eta[p] =  opts_eta1;
              element_props[j][i].fx[p]  =  0.0;
              element_props[j][i].fy[p]  = -1.0;
            }
          }
        }
      } else if (coefficient_structure == 2) { /* circular sinker */
        PetscScalar opts_eta0,opts_eta1,opts_r,radius2;

        opts_eta0 = 1.0;
        opts_eta1 = 1.0;
        opts_r    = 0.25;

        ierr = PetscOptionsGetScalar(PETSC_NULL,"-sinker_eta0",&opts_eta0,0);CHKERRQ(ierr);
        ierr = PetscOptionsGetScalar(PETSC_NULL,"-sinker_eta1",&opts_eta1,0);CHKERRQ(ierr);
        ierr = PetscOptionsGetScalar(PETSC_NULL,"-sinker_r",&opts_r,0);CHKERRQ(ierr);

        for (p = 0; p < GAUSS_POINTS; p++) {
          coord_x = centroid_x;
          coord_y = centroid_y;
          if (use_gp_coords == PETSC_TRUE) {
            coord_x = element_props[j][i].gp_coords[2*p];
            coord_y = element_props[j][i].gp_coords[2*p+1];
          }

          element_props[j][i].eta[p] = opts_eta0;
          element_props[j][i].fx[p]  = 0.0;
          element_props[j][i].fy[p]  = 0.0;

          radius2 = (coord_x-0.5)*(coord_x-0.5)+(coord_y-0.5)*(coord_y-0.5);
          if (radius2 < opts_r*opts_r) {
            element_props[j][i].eta[p] =  opts_eta1;
            element_props[j][i].fx[p]  =  0.0;
            element_props[j][i].fy[p]  = -1.0;
          }
        }
      } else {
        SETERRQ(PETSC_ERR_USER,"Unknown coefficient_structure");
      }
    }
  }
  ierr = DAVecRestoreArray(prop_cda,prop_coords,&_prop_coords);CHKERRQ(ierr);
  ierr = VecDestroy(prop_coords);CHKERRQ(ierr);
  ierr = DADestroy(prop_cda);CHKERRQ(ierr);

  ierr = DAVecRestoreArray(vel_cda,vel_coords,&_vel_coords);CHKERRQ(ierr);
  ierr = VecDestroy(vel_coords);CHKERRQ(ierr);
  ierr = DADestroy(vel_cda);CHKERRQ(ierr);

  ierr = DAVecRestoreArray(da_prop,l_properties,&element_props);CHKERRQ(ierr);
  ierr = DALocalToGlobalBegin(da_prop,l_properties,properties);CHKERRQ(ierr);
  ierr = DALocalToGlobalEnd(da_prop,l_properties,properties);CHKERRQ(ierr);


  ierr = DACoordViewGnuplot2d(da_Stokes,"mesh");CHKERRQ(ierr);
  ierr = DAViewCoefficientsGnuplot2d(da_prop,properties,"Coeffcients for Stokes eqn.","properties");CHKERRQ(ierr);


  /* Generate a matrix with the correct non-zero pattern of type AIJ. This will work in parallel and serial */
  ierr = DAGetMatrix(da_Stokes,MATAIJ,&A);CHKERRQ(ierr);
  ierr = DAGetMatrix(da_Stokes,MATAIJ,&B);CHKERRQ(ierr);
  ierr = MatGetVecs(A,&f,&X);CHKERRQ(ierr);

  /* assemble A11 */
  ierr = MatZeroEntries(A);CHKERRQ(ierr);
  ierr = MatZeroEntries(B);CHKERRQ(ierr);
  ierr = VecZeroEntries(f);CHKERRQ(ierr);

  ierr = AssembleA_Stokes(A,da_Stokes,da_prop,properties);CHKERRQ(ierr);
  ierr = AssembleA_PCStokes(B,da_Stokes,da_prop,properties);CHKERRQ(ierr);
  /* build force vector */
  ierr = AssembleF_Stokes(f,da_Stokes,da_prop,properties);CHKERRQ(ierr);

  ierr = DABCApplyFreeSlip(da_Stokes,A,f);CHKERRQ(ierr);
  ierr = DABCApplyFreeSlip(da_Stokes,B,PETSC_NULL);CHKERRQ(ierr);

  /* SOLVE */
  ierr = KSPCreate(PETSC_COMM_WORLD,&ksp_S);CHKERRQ(ierr);
  ierr = KSPSetOptionsPrefix(ksp_S,"stokes_"); /* stokes */ CHKERRQ(ierr);
  ierr = KSPSetOperators(ksp_S,A,B,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
  ierr = KSPSetFromOptions(ksp_S);CHKERRQ(ierr);

  ierr = KSPSolve(ksp_S,f,X);CHKERRQ(ierr);
  ierr = DAViewGnuplot2d(da_Stokes,X,"Velocity solution for Stokes eqn.","X");CHKERRQ(ierr);

  ierr = KSPGetIterationNumber(ksp_S,&its);CHKERRQ(ierr);

  /*
  {
  Vec x;
  PetscInt L;
  VecDuplicate(f,&x);
  MatMult(A,X, x );
  VecAXPY( x, -1.0, f );
  VecNorm( x, NORM_2, &nrm );
  PetscPrintf( PETSC_COMM_WORLD, "Its. %1.4d, norm |AX-f| = %1.5e \n", its, nrm );
  VecDestroy(x);

  VecNorm( X, NORM_2, &nrm );
  VecGetSize( X, &L );
  PetscPrintf( PETSC_COMM_WORLD, "           norm |X|/sqrt{N} = %1.5e \n", nrm/sqrt( (PetscScalar)L ) );
  }
  */

  if (coefficient_structure == 0) {
    PetscScalar opts_eta0,opts_eta1,opts_xc;
    PetscInt    opts_nz,N;
    DA          da_Stokes_analytic;
    Vec         X_analytic;
    PetscReal   nrm1[3],nrm2[3],nrmI[3];

    opts_eta0 = 1.0;
    opts_eta1 = 1.0;
    opts_xc   = 0.5;
    opts_nz   = 1;

    ierr = PetscOptionsGetScalar(PETSC_NULL,"-solcx_eta0",&opts_eta0,0);CHKERRQ(ierr);
    ierr = PetscOptionsGetScalar(PETSC_NULL,"-solcx_eta1",&opts_eta1,0);CHKERRQ(ierr);
    ierr = PetscOptionsGetScalar(PETSC_NULL,"-solcx_xc",&opts_xc,0);CHKERRQ(ierr);
    ierr = PetscOptionsGetInt(PETSC_NULL,"-solcx_nz",&opts_nz,0);CHKERRQ(ierr);


    ierr = DACreateSolCx(opts_eta0,opts_eta1,opts_xc,opts_nz,mx,my,&da_Stokes_analytic,&X_analytic);CHKERRQ(ierr);
    ierr = DAViewGnuplot2d(da_Stokes_analytic,X_analytic,"Analytic solution for Stokes eqn.","X_analytic");CHKERRQ(ierr);

    ierr = DAIntegrateErrors(da_Stokes_analytic,X,X_analytic);CHKERRQ(ierr);


    ierr = VecAXPY(X_analytic,-1.0,X);CHKERRQ(ierr);
    ierr = VecGetSize(X_analytic,&N);CHKERRQ(ierr);
    N    = N/3;

    ierr = VecStrideNorm(X_analytic,0,NORM_1,&nrm1[0]);CHKERRQ(ierr);
    ierr = VecStrideNorm(X_analytic,0,NORM_2,&nrm2[0]);CHKERRQ(ierr);
    ierr = VecStrideNorm(X_analytic,0,NORM_INFINITY,&nrmI[0]);CHKERRQ(ierr);

    ierr = VecStrideNorm(X_analytic,1,NORM_1,&nrm1[1]);CHKERRQ(ierr);
    ierr = VecStrideNorm(X_analytic,1,NORM_2,&nrm2[1]);CHKERRQ(ierr);
    ierr = VecStrideNorm(X_analytic,1,NORM_INFINITY,&nrmI[1]);CHKERRQ(ierr);

    ierr = VecStrideNorm(X_analytic,2,NORM_1,&nrm1[2]);CHKERRQ(ierr);
    ierr = VecStrideNorm(X_analytic,2,NORM_2,&nrm2[2]);CHKERRQ(ierr);
    ierr = VecStrideNorm(X_analytic,2,NORM_INFINITY,&nrmI[2]);CHKERRQ(ierr);

    /*
    PetscPrintf( PETSC_COMM_WORLD, "%1.4e   %1.4e %1.4e %1.4e    %1.4e %1.4e %1.4e    %1.4e %1.4e %1.4e\n",
    1.0/mx,
    nrm1[0]/(double)N, sqrt(nrm2[0]/(double)N), nrmI[0],
    nrm1[1]/(double)N, sqrt(nrm2[1]/(double)N), nrmI[1],
    nrm1[2]/(double)N, sqrt(nrm2[2]/(double)N), nrmI[2] );
    */
    ierr = DADestroy(da_Stokes_analytic);CHKERRQ(ierr);
    ierr = VecDestroy(X_analytic);CHKERRQ(ierr);
  }


  ierr = KSPDestroy(ksp_S);CHKERRQ(ierr);
  ierr = VecDestroy(X);CHKERRQ(ierr);
  ierr = VecDestroy(f);CHKERRQ(ierr);
  ierr = MatDestroy(A);CHKERRQ(ierr);
  ierr = MatDestroy(B);CHKERRQ(ierr);

  ierr = DADestroy(da_Stokes);CHKERRQ(ierr);
  ierr = DADestroy(da_prop);CHKERRQ(ierr);

  ierr = VecDestroy(properties);CHKERRQ(ierr);
  ierr = VecDestroy(l_properties);CHKERRQ(ierr);

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

  solve_stokes_2d_coupled(mx,my);

  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}

/* -------------------------- helpers for boundary conditions -------------------------------- */

#undef __FUNCT__  
#define __FUNCT__ "BCApply_EAST"
static PetscErrorCode BCApply_EAST(DA da,PetscInt d_idx,PetscScalar bc_val,Mat A,Vec b)
{
  DA             cda;
  Vec            coords;
  PetscInt       si,sj,nx,ny,i,j;
  PetscInt       M,N;
  DACoor2d       **_coords;
  PetscInt       *g_idx;
  PetscInt       *bc_global_ids;
  PetscScalar    *bc_vals;
  PetscInt       nbcs;
  PetscInt       n_dofs;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* enforce bc's */
  ierr = DAGetGlobalIndices(da,PETSC_NULL,&g_idx);CHKERRQ(ierr);

  ierr = DAGetCoordinateDA(da,&cda);CHKERRQ(ierr);
  ierr = DAGetGhostedCoordinates(da,&coords);CHKERRQ(ierr);
  ierr = DAVecGetArray(cda,coords,&_coords);CHKERRQ(ierr);
  ierr = DAGetGhostCorners(cda,&si,&sj,0,&nx,&ny,0);CHKERRQ(ierr);
  ierr = DAGetInfo(da,0,&M,&N,0,0,0,0,&n_dofs,0,0,0);CHKERRQ(ierr);

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
    PetscScalar coordx,coordy;

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

  if (b != PETSC_NULL) {
    ierr = VecSetValues(b,nbcs,bc_global_ids,bc_vals,INSERT_VALUES);CHKERRQ(ierr);
    ierr = VecAssemblyBegin(b);CHKERRQ(ierr);
    ierr = VecAssemblyEnd(b);CHKERRQ(ierr);
  }
  if (A != PETSC_NULL) {
    ierr = MatZeroRows(A,nbcs,bc_global_ids,1.0);CHKERRQ(ierr);
  }


  ierr = PetscFree(bc_vals);CHKERRQ(ierr);
  ierr = PetscFree(bc_global_ids);CHKERRQ(ierr);

  ierr = DAVecRestoreArray(cda,coords,&_coords);CHKERRQ(ierr);
  ierr = VecDestroy(coords);CHKERRQ(ierr);
  ierr = DADestroy(cda);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "BCApply_WEST"
static PetscErrorCode BCApply_WEST(DA da,PetscInt d_idx,PetscScalar bc_val,Mat A,Vec b)
{
  DA             cda;
  Vec            coords;
  PetscInt       si,sj,nx,ny,i,j;
  PetscInt       M,N;
  DACoor2d       **_coords;
  PetscInt       *g_idx;
  PetscInt       *bc_global_ids;
  PetscScalar    *bc_vals;
  PetscInt       nbcs;
  PetscInt       n_dofs;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* enforce bc's */
  ierr = DAGetGlobalIndices(da,PETSC_NULL,&g_idx);CHKERRQ(ierr);

  ierr = DAGetCoordinateDA(da,&cda);CHKERRQ(ierr);
  ierr = DAGetGhostedCoordinates(da,&coords);CHKERRQ(ierr);
  ierr = DAVecGetArray(cda,coords,&_coords);CHKERRQ(ierr);
  ierr = DAGetGhostCorners(cda,&si,&sj,0,&nx,&ny,0);CHKERRQ(ierr);
  ierr = DAGetInfo(da,0,&M,&N,0,0,0,0,&n_dofs,0,0,0);CHKERRQ(ierr);

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
    PetscScalar coordx,coordy;

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

  if (b != PETSC_NULL) {
    ierr = VecSetValues(b,nbcs,bc_global_ids,bc_vals,INSERT_VALUES);CHKERRQ(ierr);
    ierr = VecAssemblyBegin(b);CHKERRQ(ierr);
    ierr = VecAssemblyEnd(b);CHKERRQ(ierr);
  }
  if (A != PETSC_NULL) {
    ierr = MatZeroRows(A,nbcs,bc_global_ids,1.0);CHKERRQ(ierr);
  }


  ierr = PetscFree(bc_vals);CHKERRQ(ierr);
  ierr = PetscFree(bc_global_ids);CHKERRQ(ierr);

  ierr = DAVecRestoreArray(cda,coords,&_coords);CHKERRQ(ierr);
  ierr = VecDestroy(coords);CHKERRQ(ierr);
  ierr = DADestroy(cda);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "BCApply_NORTH"
static PetscErrorCode BCApply_NORTH(DA da,PetscInt d_idx,PetscScalar bc_val,Mat A,Vec b)
{
  DA             cda;
  Vec            coords;
  PetscInt       si,sj,nx,ny,i,j;
  PetscInt       M,N;
  DACoor2d       **_coords;
  PetscInt       *g_idx;
  PetscInt       *bc_global_ids;
  PetscScalar    *bc_vals;
  PetscInt       nbcs;
  PetscInt       n_dofs;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* enforce bc's */
  ierr = DAGetGlobalIndices(da,PETSC_NULL,&g_idx);CHKERRQ(ierr);

  ierr = DAGetCoordinateDA(da,&cda);CHKERRQ(ierr);
  ierr = DAGetGhostedCoordinates(da,&coords);CHKERRQ(ierr);
  ierr = DAVecGetArray(cda,coords,&_coords);CHKERRQ(ierr);
  ierr = DAGetGhostCorners(cda,&si,&sj,0,&nx,&ny,0);CHKERRQ(ierr);
  ierr = DAGetInfo(da,0,&M,&N,0,0,0,0,&n_dofs,0,0,0);CHKERRQ(ierr);

  /* /// */

  ierr = PetscMalloc(sizeof(PetscInt)*nx,&bc_global_ids);CHKERRQ(ierr);
  ierr = PetscMalloc(sizeof(PetscScalar)*nx,&bc_vals);CHKERRQ(ierr);

  /* init the entries to -1 so VecSetValues will ignore them */
  for (i = 0; i < nx; i++) {
    bc_global_ids[i] = -1;
  }

  j = ny-1;
  for (i = 0; i < nx; i++) {
    PetscInt    local_id;
    PetscScalar coordx,coordy;

    local_id = i+j*nx;

    bc_global_ids[i] = g_idx[ n_dofs*local_id+d_idx ];

    coordx = _coords[j+sj][i+si].x;
    coordy = _coords[j+sj][i+si].y;

    bc_vals[i] =  bc_val;
  }
  nbcs = 0;
  if ((sj+ny) == (N)) {
    nbcs = nx;
  }

  if (b != PETSC_NULL) {
    ierr = VecSetValues(b,nbcs,bc_global_ids,bc_vals,INSERT_VALUES);CHKERRQ(ierr);
    ierr = VecAssemblyBegin(b);CHKERRQ(ierr);
    ierr = VecAssemblyEnd(b);CHKERRQ(ierr);
  }
  if (A != PETSC_NULL) {
    ierr = MatZeroRows(A,nbcs,bc_global_ids,1.0);CHKERRQ(ierr);
  }


  ierr = PetscFree(bc_vals);CHKERRQ(ierr);
  ierr = PetscFree(bc_global_ids);CHKERRQ(ierr);

  ierr = DAVecRestoreArray(cda,coords,&_coords);CHKERRQ(ierr);
  ierr = VecDestroy(coords);CHKERRQ(ierr);
  ierr = DADestroy(cda);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "BCApply_SOUTH"
static PetscErrorCode BCApply_SOUTH(DA da,PetscInt d_idx,PetscScalar bc_val,Mat A,Vec b)
{
  DA             cda;
  Vec            coords;
  PetscInt       si,sj,nx,ny,i,j;
  PetscInt       M,N;
  DACoor2d       **_coords;
  PetscInt       *g_idx;
  PetscInt       *bc_global_ids;
  PetscScalar    *bc_vals;
  PetscInt       nbcs;
  PetscInt       n_dofs;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* enforce bc's */
  ierr = DAGetGlobalIndices(da,PETSC_NULL,&g_idx);CHKERRQ(ierr);

  ierr = DAGetCoordinateDA(da,&cda);CHKERRQ(ierr);
  ierr = DAGetGhostedCoordinates(da,&coords);CHKERRQ(ierr);
  ierr = DAVecGetArray(cda,coords,&_coords);CHKERRQ(ierr);
  ierr = DAGetGhostCorners(cda,&si,&sj,0,&nx,&ny,0);CHKERRQ(ierr);
  ierr = DAGetInfo(da,0,&M,&N,0,0,0,0,&n_dofs,0,0,0);CHKERRQ(ierr);

  /* /// */

  ierr = PetscMalloc(sizeof(PetscInt)*nx,&bc_global_ids);CHKERRQ(ierr);
  ierr = PetscMalloc(sizeof(PetscScalar)*nx,&bc_vals);CHKERRQ(ierr);

  /* init the entries to -1 so VecSetValues will ignore them */
  for (i = 0; i < nx; i++) {
    bc_global_ids[i] = -1;
  }

  j = 0;
  for (i = 0; i < nx; i++) {
    PetscInt    local_id;
    PetscScalar coordx,coordy;

    local_id = i+j*nx;

    bc_global_ids[i] = g_idx[ n_dofs*local_id+d_idx ];

    coordx = _coords[j+sj][i+si].x;
    coordy = _coords[j+sj][i+si].y;

    bc_vals[i] =  bc_val;
  }
  nbcs = 0;
  if (sj == 0) {
    nbcs = nx;
  }


  if (b != PETSC_NULL) {
    ierr = VecSetValues(b,nbcs,bc_global_ids,bc_vals,INSERT_VALUES);CHKERRQ(ierr);
    ierr = VecAssemblyBegin(b);CHKERRQ(ierr);
    ierr = VecAssemblyEnd(b);CHKERRQ(ierr);
  }
  if (A != PETSC_NULL) {
    ierr = MatZeroRows(A,nbcs,bc_global_ids,1.0);CHKERRQ(ierr);
  }


  ierr = PetscFree(bc_vals);CHKERRQ(ierr);
  ierr = PetscFree(bc_global_ids);CHKERRQ(ierr);

  ierr = DAVecRestoreArray(cda,coords,&_coords);CHKERRQ(ierr);
  ierr = VecDestroy(coords);CHKERRQ(ierr);
  ierr = DADestroy(cda);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DABCApplyFreeSlip"
/*
Free slip sides.
*/
#undef __FUNCT__  
#define __FUNCT__ "DABCApplyFreeSlip"
static PetscErrorCode DABCApplyFreeSlip(DA da_Stokes,Mat A,Vec f)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = BCApply_NORTH(da_Stokes,1,0.0,A,f);CHKERRQ(ierr);
  ierr = BCApply_EAST(da_Stokes,0,0.0,A,f);CHKERRQ(ierr);
  ierr = BCApply_SOUTH(da_Stokes,1,0.0,A,f);CHKERRQ(ierr);
  ierr = BCApply_WEST(da_Stokes,0,0.0,A,f);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
