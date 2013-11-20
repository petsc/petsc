/*T
   Concepts: KSP^solving a system of linear equations
   Concepts: KSP^Laplacian, 2d
   Processors: n
T*/

/*
Added at the request of Marc Garbey.

Inhomogeneous Laplacian in 2D. Modeled by the partial differential equation

   -div \rho grad u = f,  0 < x,y < 1,

with forcing function

   f = e^{-x^2/\nu} e^{-y^2/\nu}

with Dirichlet boundary conditions

   u = f(x,y) for x = 0, x = 1, y = 0, y = 1

or pure Neumman boundary conditions

This uses multigrid to solve the linear system
*/

static char help[] = "\
                      Solves 2D inhomogeneous Laplacian equation with a Gaussian source.\n \
                      Usage: ./ex4 -bc_type dirichlet -nu .01 -n 10\n \
                             ./ex4 -bc_type dirichlet -nu .01 -use_da -da_grid_x 10 -da_grid_y 10\n";


// PETSc includes:
#include <petscksp.h>
#include <petscdmda.h>
#include <petscdmmoab.h>

// MOAB includes:
#if defined (PETSC_HAVE_MOAB)
#  include <moab/Core.hpp>
#  include <moab/ReadUtilIface.hpp>
#  include <MBTagConventions.hpp>

typedef moab::Range* MBRange;
#else
#error You must have MOAB for this example. Reconfigure using --download-moab
#endif

const int NQPTS1D=2;
const int NQPTS=NQPTS1D*NQPTS1D;
const int VPERE=4;

extern PetscErrorCode ComputeMatrix_DA(KSP,Mat,Mat,MatStructure*,void*);
extern PetscErrorCode ComputeRHS_DA(KSP,Vec,void*);
extern PetscErrorCode ComputeMatrix_MOAB(KSP,Mat,Mat,MatStructure*,void*);
extern PetscErrorCode ComputeRHS_MOAB(KSP,Vec,void*);

extern PetscErrorCode Compute_Quad4_Basis ( double coords[3*4], int n, double pts[], double phi[],  double dphidx[], double dphidy[] );
extern PetscErrorCode ComputeQuadraturePointsPhysical(const double verts[VPERE*3], double quad[NQPTS*3], double jxw[NQPTS]);

typedef enum {DIRICHLET, NEUMANN} BCType;

typedef struct {
  PetscInt  dim,n;
  PetscReal rho;
  PetscReal xref,yref;
  PetscReal nu;
  BCType    bcType;
  char filename[PETSC_MAX_PATH_LEN];
} UserContext;

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  KSP            ksp;
  DM             dm;
  
  UserContext    user;
  const char     *bcTypes[2] = {"dirichlet","neumann"};
  const char     *fields[1] = {"Pressure"};
  PetscErrorCode ierr;
  PetscInt       bc;
  Vec            b,x;
  PetscBool      use_extfile;

  PetscInitialize(&argc,&argv,(char*)0,help);

  ierr        = PetscOptionsBegin(PETSC_COMM_WORLD, "", "Options for the inhomogeneous Poisson equation", "DMqq");
  user.dim    = 2;
  ierr        = PetscOptionsInt("-dim", "The dimension of the problem", "ex4.c", user.dim, &user.dim, NULL);CHKERRQ(ierr);
  user.n      = 2;
  ierr        = PetscOptionsInt("-n", "The elements in each direction", "ex4.c", user.n, &user.n, NULL);CHKERRQ(ierr);
  user.rho    = 0.5;
  ierr        = PetscOptionsReal("-rho", "The conductivity", "ex4.c", user.rho, &user.rho, NULL);CHKERRQ(ierr);
  user.xref   = 0.5;
  ierr        = PetscOptionsReal("-xref", "The x-coordinate of Gaussian center", "ex4.c", user.xref, &user.xref, NULL);CHKERRQ(ierr);
  user.yref   = 0.5;
  ierr        = PetscOptionsReal("-yref", "The y-coordinate of Gaussian center", "ex4.c", user.yref, &user.yref, NULL);CHKERRQ(ierr);
  user.nu     = 0.05;
  ierr        = PetscOptionsReal("-nu", "The width of the Gaussian source", "ex4.c", user.nu, &user.nu, NULL);CHKERRQ(ierr);
  bc          = (PetscInt)NEUMANN;
  ierr        = PetscOptionsEList("-bc_type","Type of boundary condition","ex4.c",bcTypes,2,bcTypes[0],&bc,NULL);CHKERRQ(ierr);
  user.bcType = (BCType)bc;
  ierr        = PetscOptionsString("-file", "The mesh file for the problem", "ex4.c", "",user.filename,PETSC_MAX_PATH_LEN,&use_extfile);CHKERRQ(ierr);
  ierr        = PetscOptionsEnd();

  if (use_extfile) {
    ierr = DMMoabLoadFromFile(PETSC_COMM_WORLD, user.dim, user.filename, "", &dm);CHKERRQ(ierr);
  }
  else {
    ierr = DMMoabCreateBoxMesh(PETSC_COMM_WORLD, user.dim, user.n, 1, &dm);CHKERRQ(ierr);
  }
  ierr = DMSetFromOptions(dm);CHKERRQ(ierr);
  ierr = DMMoabSetFields(dm, 1, fields);CHKERRQ(ierr);

  /* SetUp the data structures for DMMOAB */
  ierr = DMSetUp(dm);CHKERRQ(ierr);

  ierr = KSPCreate(PETSC_COMM_WORLD,&ksp);CHKERRQ(ierr);
  ierr = KSPSetComputeRHS(ksp,ComputeRHS_MOAB,&user);CHKERRQ(ierr);
  ierr = KSPSetComputeOperators(ksp,ComputeMatrix_MOAB,&user);CHKERRQ(ierr);
  ierr = KSPSetDM(ksp,dm);CHKERRQ(ierr);

  ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);
  ierr = KSPSetUp(ksp);CHKERRQ(ierr);
  ierr = KSPSolve(ksp,NULL,NULL);CHKERRQ(ierr);
  ierr = KSPGetSolution(ksp,&x);CHKERRQ(ierr);
  ierr = KSPGetRhs(ksp,&b);CHKERRQ(ierr);

  ierr = DMMoabSetFieldVector(dm, 0, x);CHKERRQ(ierr);
  ierr = DMMoabOutput(dm, "ex4.vtk", "");CHKERRQ(ierr);

  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  ierr = KSPDestroy(&ksp);CHKERRQ(ierr);
  ierr = PetscFinalize();

  return 0;
}


#undef __FUNCT__
#define __FUNCT__ "ComputeRho_DA"
PetscErrorCode ComputeRho_DA(PetscInt i, PetscInt j, PetscInt mx, PetscInt my, PetscReal centerRho, PetscReal *rho)
{
  PetscFunctionBeginUser;
  if ((i > mx/3.0) && (i < 2.0*mx/3.0) && (j > my/3.0) && (j < 2.0*my/3.0)) {
    *rho = centerRho;
  } else {
    *rho = 1.0;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ComputeRho_MOAB"
PetscErrorCode ComputeRho_MOAB(PetscReal coords[3], PetscReal centerRho, PetscReal *rho)
{
  PetscFunctionBeginUser;
  if ((coords[0] > 1.0/3.0) && (coords[0] < 2.0/3.0) && (coords[1] > 1.0/3.0) && (coords[1] < 2.0/3.0)) {
    *rho = centerRho;
  } else {
    *rho = 1.0;
  }
  PetscFunctionReturn(0);
}



#undef __FUNCT__
#define __FUNCT__ "ComputeRHS_MOAB"
PetscErrorCode ComputeRHS_MOAB(KSP ksp,Vec b,void *ptr)
{
  UserContext*  user = (UserContext*)ptr;
  DM            dm;
  PetscScalar   Hx,Hy;
  PetscScalar   *array;
  PetscInt dof_indices[VPERE],dbdry[VPERE];
  double vpos[VPERE*3],quadrature[NQPTS*3],jxw[NQPTS];
  PetscInt i,q,num_conn;
  const moab::EntityHandle *connect;
  moab::Range elocal,vlocal,bdvtx;
  moab::Interface*  mbImpl;
  PetscScalar phi[VPERE*NQPTS];
  PetscBool         elem_on_boundary;

  moab::ErrorCode  merr;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  ierr = KSPGetDM(ksp,&dm);CHKERRQ(ierr);
  
  Hx = Hy = 1.0/user->n;

  /* reset the RHS */
  ierr = VecSet(b, 0.0);CHKERRQ(ierr);

  /* Get pointers to vector data 
      -- get the local representation of the arrays from DM */
  ierr = DMMoabVecGetArray(dm, b, &array);CHKERRQ(ierr);

  ierr = DMMoabGetInterface(dm, &mbImpl);CHKERRQ(ierr);
  ierr = DMMoabGetLocalVertices(dm, &vlocal, PETSC_NULL);CHKERRQ(ierr);
  ierr = DMMoabGetLocalElements(dm, &elocal);CHKERRQ(ierr);
  ierr = DMMoabGetBoundaryEntities(dm, &bdvtx, PETSC_NULL);CHKERRQ(ierr);
  
  PetscPrintf(PETSC_COMM_WORLD, "\n MOAB mesh: Found %D vertices and %D elements.\n", vlocal.size(), elocal.size());
  
  /* loop over local elements */
  for(moab::Range::iterator iter = elocal.begin(); iter != elocal.end(); iter++) {
    const moab::EntityHandle ehandle = *iter;

    // Get connectivity information:
    merr = mbImpl->get_connectivity(ehandle, connect, num_conn);MBERRNM(merr); // get the connectivity, in canonical order
    if (num_conn != VPERE) SETERRQ1(PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG, "Only QUAD4 element bases are supported in the current example. Connectivity=%D.\n", num_conn);

    /* compute the mid-point of the element and use a 1-point lumped quadrature */
    merr = mbImpl->get_coords(connect, num_conn, vpos);MBERRNM(merr);

    ierr = DMMoabGetFieldDofs(dm, num_conn, connect, 0, dof_indices);CHKERRQ(ierr);
    
#if 0
    /* Compute function over the locally owned part of the grid */
    for (i=0; i < num_conn; ++i) {
      array[dof_indices[i]] = PetscExpScalar(-(vpos[3*i]*vpos[3*i])/user->nu)*PetscExpScalar(-(vpos[3*i+1]*vpos[3*i+1])/user->nu)*Hx*Hy;
    }
#else

    ierr = ComputeQuadraturePointsPhysical(vpos, quadrature, jxw);CHKERRQ(ierr);

    /* compute the basis functions and the derivatives wrt x and y directions */
    ierr = Compute_Quad4_Basis(vpos, NQPTS, quadrature, phi,  0, 0);CHKERRQ(ierr);

    /* Compute function over the locally owned part of the grid */
    for (q=0; q<NQPTS; ++q) {
      const double xx=(quadrature[3*q]-user->xref)*(quadrature[3*q]-user->xref);
      const double yy=(quadrature[3*q+1]-user->yref)*(quadrature[3*q+1]-user->yref);
      for (i=0; i < VPERE; ++i) {
        array[dof_indices[i]] += jxw[q] * phi[q*VPERE+i] * PetscExpScalar(-(xx+yy)/user->nu)/Hx/Hy;
      }
    }

    /* check if element is on the bundary */
    elem_on_boundary = PETSC_FALSE;
    for (i=0; i < VPERE; ++i) {
      moab::Range::const_iterator giter = bdvtx.find(connect[i]);
      if (giter != bdvtx.end()) {
        dbdry[i] = 1;
        elem_on_boundary = PETSC_TRUE;
      }
      else dbdry[i] = 0;
    }

    /* apply dirichlet boundary conditions */
    if (elem_on_boundary && user->bcType == DIRICHLET) {
      for (i=0; i < VPERE; ++i) {
        if (dbdry[i]) {  /* dirichlet node */
          /* think about strongly imposing dirichlet */
          const double xx=(vpos[3*q]-user->xref)*(vpos[3*q]-user->xref);
          const double yy=(vpos[3*q+1]-user->yref)*(vpos[3*q+1]-user->yref);
          array[dof_indices[i]] = PetscExpScalar(-(xx+yy)/user->nu);
        }
      }
    }

#endif

  }

  /* Restore vectors */
  ierr = DMMoabVecRestoreArray(dm, b, &array);CHKERRQ(ierr);

  /* force right hand side to be consistent for singular matrix */
  /* note this is really a hack, normally the model would provide you with a consistent right handside */
  if (user->bcType == NEUMANN) {
    MatNullSpace nullspace;

    ierr = MatNullSpaceCreate(PETSC_COMM_WORLD,PETSC_TRUE,0,0,&nullspace);CHKERRQ(ierr);
    ierr = MatNullSpaceRemove(nullspace,b);CHKERRQ(ierr);
    ierr = MatNullSpaceDestroy(&nullspace);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "ComputeMatrix_MOAB"
PetscErrorCode ComputeMatrix_MOAB(KSP ksp,Mat J,Mat jac,MatStructure *str,void *ctx)
{
  UserContext    *user = (UserContext*)ctx;
  DM            dm;
 
  PetscInt       i,j,q,num_conn;
  PetscInt dof_indices[VPERE],dbdry[VPERE];
  double vpos[VPERE*3],quadrature[NQPTS*3],jxw[NQPTS];

  const moab::EntityHandle *connect;
  moab::Range elocal,vlocal,bdvtx;;
  moab::Interface*  mbImpl;
  PetscBool         elem_on_boundary;
  PetscScalar  array[VPERE*VPERE];
  double phi[VPERE*NQPTS], dphidx[VPERE*NQPTS], dphidy[VPERE*NQPTS];

  PetscReal      rho;
 
  PetscErrorCode ierr;
  moab::ErrorCode  merr;
 
  PetscFunctionBeginUser;
  ierr      = KSPGetDM(ksp,&dm);CHKERRQ(ierr);

  ierr = DMMoabGetInterface(dm, &mbImpl);CHKERRQ(ierr);
  ierr = DMMoabGetLocalVertices(dm, &vlocal, PETSC_NULL);CHKERRQ(ierr);
  ierr = DMMoabGetLocalElements(dm, &elocal);CHKERRQ(ierr);
  ierr = DMMoabGetBoundaryEntities(dm, &bdvtx, PETSC_NULL);CHKERRQ(ierr);
  
  /* loop over local elements */
  for(moab::Range::iterator iter = elocal.begin(); iter != elocal.end(); iter++) {
    const moab::EntityHandle ehandle = *iter;

    // Get connectivity information:
    merr = mbImpl->get_connectivity(ehandle, connect, num_conn);MBERRNM(merr); // get the connectivity, in canonical order
    if (num_conn != VPERE) SETERRQ1(PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG, "Only QUAD4 element bases are supported in the current example. Connectivity=%D.\n", num_conn);

    /* compute the mid-point of the element and use a 1-point lumped quadrature */
    merr = mbImpl->get_coords(connect, num_conn, vpos);MBERRNM(merr);

    ierr = DMMoabGetFieldDofs(dm, num_conn, connect, 0, dof_indices);CHKERRQ(ierr);

    ierr = ComputeQuadraturePointsPhysical(vpos, quadrature, jxw);CHKERRQ(ierr);

    /* compute the inhomogeneous diffusion coefficient at the first quadrature point 
        -- for large spatial variations, embed this property evaluation inside quadrature loop
    */
    ierr  = ComputeRho_MOAB(quadrature, user->rho, &rho);CHKERRQ(ierr);
 
    /* compute the basis functions and the derivatives wrt x and y directions */
    ierr = Compute_Quad4_Basis(vpos, NQPTS, quadrature, phi,  dphidx, dphidy);CHKERRQ(ierr);

    ierr = PetscMemzero(array, VPERE*VPERE*sizeof(PetscScalar));

    /* Compute function over the locally owned part of the grid */
    for (q=0; q<NQPTS; ++q) {
      for (i=0; i < VPERE; ++i) {
        for (j=0; j < VPERE; ++j) {
          array[i*VPERE+j] += jxw[q] * rho * ( dphidx[q*VPERE+i]*dphidx[q*VPERE+j] + 
                                                              dphidy[q*VPERE+i]*dphidy[q*VPERE+j] );
        }
      }
    }

    /* check if element is on the bundary */
    elem_on_boundary = PETSC_FALSE;
    for (i=0; i < VPERE; ++i) {
      moab::Range::const_iterator giter = bdvtx.find(connect[i]);
      if (giter != bdvtx.end()) {
        dbdry[i] = 1;
        elem_on_boundary = PETSC_TRUE;
      }
      else dbdry[i] = 0;
    }

    /* apply dirichlet boundary conditions */
    if (elem_on_boundary && user->bcType == DIRICHLET) {
      for (i=0; i < VPERE; ++i) {
        if (dbdry[i]) {  /* dirichlet node */
          /* think about strongly imposing dirichlet */
          for (j=0; j < VPERE; ++j) {
            array[i*VPERE+j] = 0.0;
            /* TODO: symmetrize the system - need the RHS */
            /*
            const double xx=(vpos[3*q]-user->xref)*(vpos[3*q]-user->xref);
            const double yy=(vpos[3*q+1]-user->yref)*(vpos[3*q+1]-user->yref);
            barray[j*VPERE] -= array[j*VPERE+i]*barray[i*VPERE];
            array[j*VPERE+i] = 0.0;
            */
          }
          array[i*VPERE+i] = 1.0;
        }
      }
    }

    ierr = MatSetValues(jac, VPERE, dof_indices, VPERE, dof_indices, array, ADD_VALUES);CHKERRQ(ierr);
  }

  ierr = MatAssemblyBegin(jac,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(jac,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  if (user->bcType == NEUMANN) {
    MatNullSpace nullspace;

    ierr = MatNullSpaceCreate(PETSC_COMM_WORLD,PETSC_TRUE,0,0,&nullspace);CHKERRQ(ierr);
    ierr = MatSetNullSpace(jac,nullspace);CHKERRQ(ierr);
    ierr = MatNullSpaceDestroy(&nullspace);CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "ComputeRHS_DA"
PetscErrorCode ComputeRHS_DA(KSP ksp,Vec b,void *ctx)
{
  UserContext    *user = (UserContext*)ctx;
  PetscErrorCode ierr;
  PetscInt       i,j,mx,my,xm,ym,xs,ys;
  PetscScalar    Hx,Hy;
  PetscScalar    **array;
  DM             da;

  PetscFunctionBeginUser;
  ierr = KSPGetDM(ksp,&da);CHKERRQ(ierr);
  ierr = DMDAGetInfo(da, 0, &mx, &my, 0,0,0,0,0,0,0,0,0,0);CHKERRQ(ierr);
  Hx   = 1.0 / (PetscReal)(mx-1);
  Hy   = 1.0 / (PetscReal)(my-1);
  ierr = DMDAGetCorners(da,&xs,&ys,0,&xm,&ym,0);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(da, b, &array);CHKERRQ(ierr);
  for (j=ys; j<ys+ym; j++) {
    for (i=xs; i<xs+xm; i++) {
      array[j][i] = PetscExpScalar(-((PetscReal)i*Hx)*((PetscReal)i*Hx)/user->nu)*PetscExpScalar(-((PetscReal)j*Hy)*((PetscReal)j*Hy)/user->nu)*Hx*Hy;
    }
  }
  ierr = DMDAVecRestoreArray(da, b, &array);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(b);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(b);CHKERRQ(ierr);

  /* force right hand side to be consistent for singular matrix */
  /* note this is really a hack, normally the model would provide you with a consistent right handside */
  if (user->bcType == NEUMANN) {
    MatNullSpace nullspace;

    ierr = MatNullSpaceCreate(PETSC_COMM_WORLD,PETSC_TRUE,0,0,&nullspace);CHKERRQ(ierr);
    ierr = MatNullSpaceRemove(nullspace,b);CHKERRQ(ierr);
    ierr = MatNullSpaceDestroy(&nullspace);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "ComputeMatrix_DA"
PetscErrorCode ComputeMatrix_DA(KSP ksp,Mat J,Mat jac,MatStructure *str,void *ctx)
{
  UserContext    *user = (UserContext*)ctx;
  PetscReal      centerRho;
  PetscErrorCode ierr;
  PetscInt       i,j,mx,my,xm,ym,xs,ys;
  PetscScalar    v[5];
  PetscReal      Hx,Hy,HydHx,HxdHy,rho;
  MatStencil     row, col[5];
  DM             da;

  PetscFunctionBeginUser;
  ierr      = KSPGetDM(ksp,&da);CHKERRQ(ierr);
  centerRho = user->rho;
  ierr      = DMDAGetInfo(da,0,&mx,&my,0,0,0,0,0,0,0,0,0,0);CHKERRQ(ierr);
  Hx        = 1.0 / (PetscReal)(mx-1);
  Hy        = 1.0 / (PetscReal)(my-1);
  HxdHy     = Hx/Hy;
  HydHx     = Hy/Hx;
  ierr      = DMDAGetCorners(da,&xs,&ys,0,&xm,&ym,0);CHKERRQ(ierr);
  for (j=ys; j<ys+ym; j++) {
    for (i=xs; i<xs+xm; i++) {
      row.i = i; row.j = j;
      ierr  = ComputeRho_DA(i, j, mx, my, centerRho, &rho);CHKERRQ(ierr);
      if (i==0 || j==0 || i==mx-1 || j==my-1) {
        if (user->bcType == DIRICHLET) {
          v[0] = 2.0*rho*(HxdHy + HydHx);
          ierr = MatSetValuesStencil(jac,1,&row,1,&row,v,INSERT_VALUES);CHKERRQ(ierr);
        } else if (user->bcType == NEUMANN) {
          PetscInt numx = 0, numy = 0, num = 0;
          if (j!=0) {
            v[num] = -rho*HxdHy;              col[num].i = i;   col[num].j = j-1;
            numy++; num++;
          }
          if (i!=0) {
            v[num] = -rho*HydHx;              col[num].i = i-1; col[num].j = j;
            numx++; num++;
          }
          if (i!=mx-1) {
            v[num] = -rho*HydHx;              col[num].i = i+1; col[num].j = j;
            numx++; num++;
          }
          if (j!=my-1) {
            v[num] = -rho*HxdHy;              col[num].i = i;   col[num].j = j+1;
            numy++; num++;
          }
          v[num] = numx*rho*HydHx + numy*rho*HxdHy; col[num].i = i;   col[num].j = j;
          num++;
          ierr = MatSetValuesStencil(jac,1,&row,num,col,v,INSERT_VALUES);CHKERRQ(ierr);
        }
      } else {
        v[0] = -rho*HxdHy;              col[0].i = i;   col[0].j = j-1;
        v[1] = -rho*HydHx;              col[1].i = i-1; col[1].j = j;
        v[2] = 2.0*rho*(HxdHy + HydHx); col[2].i = i;   col[2].j = j;
        v[3] = -rho*HydHx;              col[3].i = i+1; col[3].j = j;
        v[4] = -rho*HxdHy;              col[4].i = i;   col[4].j = j+1;
        ierr = MatSetValuesStencil(jac,1,&row,5,col,v,INSERT_VALUES);CHKERRQ(ierr);
      }
    }
  }
  ierr = MatAssemblyBegin(jac,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(jac,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  if (user->bcType == NEUMANN) {
    MatNullSpace nullspace;

    ierr = MatNullSpaceCreate(PETSC_COMM_WORLD,PETSC_TRUE,0,0,&nullspace);CHKERRQ(ierr);
    ierr = MatSetNullSpace(jac,nullspace);CHKERRQ(ierr);
    ierr = MatNullSpaceDestroy(&nullspace);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}



#undef __FUNCT__
#define __FUNCT__ "Compute_Quad4_Basis"
PetscErrorCode Compute_Quad4_Basis ( double coords[VPERE*3], PetscInt n, double *pts, PetscScalar *phi, PetscScalar *dphidx, PetscScalar *dphidy)
/*
*
*  Purpose:
*
*    BASIS_MN_Q4: all bases at N points for a Q4 element.
*
*  Discussion:
*
*    The routine is given the coordinates of the vertices of a quadrilateral.
*    It works directly with these coordinates, and does not refer to a 
*    reference element.
*
*    The sides of the element are presumed to lie along coordinate axes.
*
*    The routine evaluates the basis functions associated with each corner,
*    and their derivatives with respect to X and Y.
*
*  Physical Element Q4:
*
*    |
*    |  4-------3
*    |  |       |
*    Y  |       |
*    |  |       |
*    |  1-------2
*    |
*    +-----X------>
*
*  Parameters:
*
*    Input, double Q[3*4], the coordinates of the vertices.
*    It is common to list these points in counter clockwise order.
*
*    Input, int N, the number of evaluation points.
*
*    Input, double P[3*N], the evaluation points.
*
*    Output, double PHI[4*N], the bases at the evaluation points.
*
*    Output, double DPHIDX[4*N], DPHIDY[4*N], the derivatives of the
*    bases at the evaluation points.
*
*  Original Author: John Burkardt (http://people.sc.fsu.edu/~jburkardt/cpp_src/fem2d_pack/fem2d_pack.cpp)
*  Modified by Vijay Mahadevan
*/
{
  PetscScalar ejac;
  int i,j;

  PetscFunctionBegin;
  ejac = ( coords[0+2*3] - coords[0+0*3] ) * ( coords[1+2*3] - coords[1+0*3] );
  ejac = 1.0/(ejac);

  for (j=0;j<n;j++)
  {
    phi[0+j*4] =      ( coords[0+2*3] - pts[0+j*3]  ) * ( coords[1+2*3] - pts[1+j*3]  );
    phi[1+j*4] =      ( pts[0+j*3]  - coords[0+0*3] ) * ( coords[1+2*3] - pts[1+j*3]  );
    phi[2+j*4] =      ( pts[0+j*3]  - coords[0+0*3] ) * ( pts[1+j*3]  - coords[1+0*3] );
    phi[3+j*4] =      ( coords[0+2*3] - pts[0+j*3]  ) * ( pts[1+j*3]  - coords[1+0*3] );

    if (dphidx) {
      dphidx[0+j*4] = - ( coords[1+2*3] - pts[1+j*3] );
      dphidx[1+j*4] =   ( coords[1+2*3] - pts[1+j*3] );
      dphidx[2+j*4] =   ( pts[1+j*3]  - coords[1+0*3] );
      dphidx[3+j*4] = - ( pts[1+j*3]  - coords[1+0*3] );
    }

    if (dphidy) {
      dphidy[0+j*4] = - ( coords[0+2*3] - pts[0+j*3] );
      dphidy[1+j*4] = - ( pts[0+j*3]  - coords[0+0*3] );
      dphidy[2+j*4] =   ( pts[0+j*3]  - coords[0+0*3] );
      dphidy[3+j*4] =   ( coords[0+2*3] - pts[0+j*3] );
    }
  }

  /*  Divide by element jacobian. */
  for ( j = 0; j < n; j++ ) {
    for ( i = 0; i < VPERE; i++ ) {
      phi[i+j*4]    *= ejac;
      if (dphidx) dphidx[i+j*4] *= ejac;
      if (dphidy) dphidy[i+j*4] *= ejac;
    }
  }
#if 0
  /* verify if the computed basis functions are consistent */
  for ( j = 0; j < n; j++ ) {
    double phisum=0,dphixsum=0,dphiysum=0;
    for ( i = 0; i < 4; i++ ) {
      phisum += phi[i+j*4];
      if (dphidx) dphixsum += dphidx[i+j*4];
      if (dphidy) dphiysum += dphidy[i+j*4];
    }
    PetscPrintf(PETSC_COMM_WORLD, "Sum of basis at quadrature point %D = %G, %G, %G\n", j, phisum, dphixsum, dphiysum);
  }

#endif
  PetscFunctionReturn(0);
}

const double GLG_QUAD_P1[1+1] = {0.0, 2.0};
const double GLG_QUAD_P3[2+1] = {-0.577350269189625764509148780502, 0.577350269189625764509148780502, 1.0};
PetscErrorCode ComputeQuadraturePointsPhysical(const double verts[VPERE*3], double quad[NQPTS*3], double jxw[NQPTS])
{
  int i,j;
  double centroid[3];
  const double *x;
  double dx = fabs(verts[0+2*3] - verts[0+0*3])/2, dy = fabs( verts[1+2*3] - verts[1+0*3] )/2;
  double ejac = dx*dy;

  switch(NQPTS1D) {
   case 1:
     x = GLG_QUAD_P1;
     break;
   case 2:
   default:
     x = GLG_QUAD_P3;
     break;
  }
  
  centroid[0] = centroid[1] = centroid[2] = 0.0;
  for (i=0; i<VPERE; ++i) {
    centroid[0] += verts[i*3+0];
    centroid[1] += verts[i*3+1];
    centroid[2] += verts[i*3+2];
  }
  centroid[0] /= 4;
  centroid[1] /= 4;
  centroid[2] /= 4;

  for (i=0; i<NQPTS1D; ++i) {
    for (j=0; j<NQPTS1D; ++j) {
      quad[(i*NQPTS1D+j)*3] = centroid[0]+dx*(x[i]);
      quad[(i*NQPTS1D+j)*3+1] = centroid[1]+dy*(x[j]);
      quad[(i*NQPTS1D+j)*3+2] = centroid[2];
      jxw[i*NQPTS1D+j] = x[NQPTS1D]*ejac;
//      PetscPrintf(PETSC_COMM_WORLD, "Element [%D, %D] : Quadrature [%G, %G]\n", i, j, quad[(i*NQPTS1D+j)*3], quad[(i*NQPTS1D+j)*3+1]);
    }
  }
//  PetscPrintf(PETSC_COMM_WORLD, "Element [%G, %G]  has jacobian = %G\n", centroid[0], centroid[1], 4*ejac*ejac);
  
  PetscFunctionReturn(0);
}


