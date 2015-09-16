/*T
   Concepts: KSP^solving a system of linear equations using a MOAB based DM implementation.
   Concepts: KSP^Laplacian, 2d
   Processors: n
T*/

/*
Inhomogeneous Laplacian in 2D. Modeled by the partial differential equation

   -div \rho grad u = f,  0 < x,y < 1,

with forcing function

   f = e^{-((x-xr)^2+(y-yr)^2)/\nu}

with Dirichlet boundary conditions

   u = f(x,y) for x = 0, x = 1, y = 0, y = 1

or pure Neumman boundary conditions

Usage:
    mpiexec -n 2 ./ex2 -bc_type dirichlet -nu .01 -rho .01 -file input/quad_2p.h5m -dmmb_rw_dbg 0 -n 50

*/

static char help[] = "\
                      Solves 2D inhomogeneous Laplacian equation with a Gaussian source.\n \
                      Usage: ./ex2 -bc_type dirichlet -nu .01 -n 10\n";


/* PETSc includes */
#include <petscksp.h>
#include <petscdmmoab.h>

#define LOCAL_ASSEMBLY

const int NQPTS1D=2;
const int NQPTS=NQPTS1D*NQPTS1D;
const int VPERE=4;

extern PetscErrorCode ComputeMatrix_MOAB(KSP,Mat,Mat,void*);
extern PetscErrorCode ComputeRHS_MOAB(KSP,Vec,void*);

extern PetscErrorCode Compute_Quad4_Basis ( PetscReal coords[3*4], int n, PetscReal pts[], PetscReal *phi, PetscReal *dphidx, PetscReal *dphidy );
extern PetscErrorCode ComputeQuadraturePointsPhysical(const PetscReal verts[VPERE*3], PetscReal quad[NQPTS*3], PetscReal jxw[NQPTS]);

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
  const char     *fields[1] = {"T-Variable"};
  PetscErrorCode ierr;
  PetscInt       bc,np;
  Vec            b,x;
  PetscBool      use_extfile,io;

  PetscInitialize(&argc,&argv,(char*)0,help);

  MPI_Comm_size(PETSC_COMM_WORLD,&np);

  ierr        = PetscOptionsBegin(PETSC_COMM_WORLD, "", "Options for the inhomogeneous Poisson equation", "ex35.c");
  user.dim    = 2;
  ierr        = PetscOptionsInt("-dim", "The dimension of the problem", "ex35.c", user.dim, &user.dim, NULL);CHKERRQ(ierr);
  user.n      = 2;
  ierr        = PetscOptionsInt("-n", "The elements in each direction", "ex35.c", user.n, &user.n, NULL);CHKERRQ(ierr);
  user.rho    = 0.5;
  ierr        = PetscOptionsReal("-rho", "The conductivity", "ex35.c", user.rho, &user.rho, NULL);CHKERRQ(ierr);
  user.xref   = 0.5;
  ierr        = PetscOptionsReal("-xref", "The x-coordinate of Gaussian center", "ex35.c", user.xref, &user.xref, NULL);CHKERRQ(ierr);
  user.yref   = 0.5;
  ierr        = PetscOptionsReal("-yref", "The y-coordinate of Gaussian center", "ex35.c", user.yref, &user.yref, NULL);CHKERRQ(ierr);
  user.nu     = 0.05;
  ierr        = PetscOptionsReal("-nu", "The width of the Gaussian source", "ex35.c", user.nu, &user.nu, NULL);CHKERRQ(ierr);
  io          = PETSC_FALSE;
  ierr        = PetscOptionsBool("-io", "Write out the solution and mesh data", "ex35.c", io, &io, NULL);CHKERRQ(ierr);
  bc          = (PetscInt)DIRICHLET;
  ierr        = PetscOptionsEList("-bc_type","Type of boundary condition","ex35.c",bcTypes,2,bcTypes[0],&bc,NULL);CHKERRQ(ierr);
  user.bcType = (BCType)bc;
  ierr        = PetscOptionsString("-file", "The mesh file for the problem", "ex35.c", "",user.filename,PETSC_MAX_PATH_LEN,&use_extfile);CHKERRQ(ierr);
  ierr        = PetscOptionsEnd();

  /* Create the DM object from either a mesh file or from in-memory structured grid */
  if (use_extfile) {
    ierr = DMMoabLoadFromFile(PETSC_COMM_WORLD, user.dim, user.filename, (np==1 ? "" : ""), &dm);CHKERRQ(ierr);
  }
  else {
    ierr = DMMoabCreateBoxMesh(PETSC_COMM_WORLD, user.dim, PETSC_FALSE, NULL, user.n, 1, &dm);CHKERRQ(ierr);
  }
  ierr = DMSetFromOptions(dm);CHKERRQ(ierr);
  ierr = DMMoabSetFieldNames(dm, 1, fields);CHKERRQ(ierr);

  /* SetUp the data structures for DMMOAB */
  ierr = DMSetUp(dm);CHKERRQ(ierr);

  ierr = KSPCreate(PETSC_COMM_WORLD,&ksp);CHKERRQ(ierr);
  ierr = KSPSetComputeRHS(ksp,ComputeRHS_MOAB,&user);CHKERRQ(ierr);
  ierr = KSPSetComputeOperators(ksp,ComputeMatrix_MOAB,&user);CHKERRQ(ierr);
  ierr = KSPSetDM(ksp,dm);CHKERRQ(ierr);

  ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);

  /* Perform the actual solve */
  ierr = KSPSolve(ksp,NULL,NULL);CHKERRQ(ierr);
  ierr = KSPGetSolution(ksp,&x);CHKERRQ(ierr);
  ierr = KSPGetRhs(ksp,&b);CHKERRQ(ierr);

  if (io) {
    /* Write out the solution along with the mesh */
    ierr = DMMoabSetGlobalFieldVector(dm, x);CHKERRQ(ierr);
#ifdef MOAB_HDF5_H
    ierr = DMMoabOutput(dm, "ex35.h5m", "");CHKERRQ(ierr);
#else
    /* MOAB does not support true parallel writers that aren't HDF5 based
       And so if you are using VTK as the output format in parallel,
       the data could be jumbled due to the order in which the processors
       write out their parts of the mesh and solution tags
    */
    ierr = DMMoabOutput(dm, "ex35.vtk", "");CHKERRQ(ierr);
#endif
  }

  /* Cleanup objects */
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  ierr = KSPDestroy(&ksp);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return 0;
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
  UserContext*      user = (UserContext*)ptr;
  DM                dm;
  PetscInt          dof_indices[VPERE];
  PetscBool         dbdry[VPERE];
  PetscReal         vpos[VPERE*3],quadrature[NQPTS*3],jxw[NQPTS],phi[VPERE*NQPTS];
  PetscInt          i,q,num_conn;
  const moab::EntityHandle *connect;
  const moab::Range *elocal;
  moab::Interface*  mbImpl;
  PetscScalar       localv[VPERE];
  PetscBool         elem_on_boundary;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = KSPGetDM(ksp,&dm);CHKERRQ(ierr);

  /* reset the RHS */
  ierr = VecSet(b, 0.0);CHKERRQ(ierr);

  /* get the essential MOAB mesh related quantities needed for FEM assembly */
  ierr = DMMoabGetInterface(dm, &mbImpl);CHKERRQ(ierr);
  ierr = DMMoabGetLocalElements(dm, &elocal);CHKERRQ(ierr);

  /* loop over local elements */
  for(moab::Range::iterator iter = elocal->begin(); iter != elocal->end(); iter++) {
    const moab::EntityHandle ehandle = *iter;

    ierr = PetscMemzero(localv,sizeof(PetscScalar)*VPERE);CHKERRQ(ierr);

    // Get connectivity information:
    ierr = DMMoabGetElementConnectivity(dm, ehandle, &num_conn, &connect);CHKERRQ(ierr);
    if (num_conn != VPERE) SETERRQ1(PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG, "Only QUAD4 element bases are supported in the current example. n(Connectivity)=%D.\n", num_conn);

    /* compute the mid-point of the element and use a 1-point lumped quadrature */
    ierr = DMMoabGetVertexCoordinates(dm,num_conn,connect,vpos);CHKERRQ(ierr);

    /* get the global DOF number to appropriately set the element contribution in the RHS vector */
#ifdef LOCAL_ASSEMBLY
    ierr = DMMoabGetFieldDofsLocal(dm, num_conn, connect, 0, dof_indices);CHKERRQ(ierr);
#else
    ierr = DMMoabGetFieldDofs(dm, num_conn, connect, 0, dof_indices);CHKERRQ(ierr);    
#endif

    /* compute the quadrature points transformed to the physical space */
    ierr = ComputeQuadraturePointsPhysical(vpos, quadrature, jxw);CHKERRQ(ierr);

    /* compute the basis functions and the derivatives wrt x and y directions */
    ierr = Compute_Quad4_Basis(vpos, NQPTS, quadrature, phi,  0, 0);CHKERRQ(ierr);

    /* Compute function over the locally owned part of the grid */
    for (q=0; q<NQPTS; ++q) {
      const PetscScalar xx=(quadrature[3*q]-user->xref)*(quadrature[3*q]-user->xref);
      const PetscScalar yy=(quadrature[3*q+1]-user->yref)*(quadrature[3*q+1]-user->yref);
      for (i=0; i < VPERE; ++i) {
        localv[i] += jxw[q] * phi[q*VPERE+i] * PetscExpScalar(-(xx+yy)/user->nu);
      }
    }

    /* check if element is on the boundary */
    ierr = DMMoabIsEntityOnBoundary(dm,ehandle,&elem_on_boundary);CHKERRQ(ierr);

    /* apply dirichlet boundary conditions */
    if (elem_on_boundary && user->bcType == DIRICHLET) {

      /* get the list of nodes on boundary so that we can enforce dirichlet conditions strongly */
      ierr = DMMoabCheckBoundaryVertices(dm,num_conn,connect,dbdry);CHKERRQ(ierr);

      for (i=0; i < VPERE; ++i) {
        if (dbdry[i]) {  /* dirichlet node */
          /* think about strongly imposing dirichlet */
          const PetscScalar xx=(vpos[3*i]-user->xref)*(vpos[3*i]-user->xref);
          const PetscScalar yy=(vpos[3*i+1]-user->yref)*(vpos[3*i+1]-user->yref);
          localv[i] = PetscExpScalar(-(xx+yy)/user->nu);
        }
      }
    }

#ifdef LOCAL_ASSEMBLY
    /* set the values directly into appropriate locations. Can alternately use VecSetValues */
    ierr = VecSetValuesLocal(b, VPERE, dof_indices, localv, ADD_VALUES);CHKERRQ(ierr);
#else
    ierr = VecSetValues(b, VPERE, dof_indices, localv, ADD_VALUES);CHKERRQ(ierr);
#endif
  }

  /* force right hand side to be consistent for singular matrix */
  /* note this is really a hack, normally the model would provide you with a consistent right handside */
  if (user->bcType == NEUMANN) {
    MatNullSpace nullspace;

    ierr = MatNullSpaceCreate(PETSC_COMM_WORLD,PETSC_TRUE,0,0,&nullspace);CHKERRQ(ierr);
    ierr = MatNullSpaceRemove(nullspace,b);CHKERRQ(ierr);
    ierr = MatNullSpaceDestroy(&nullspace);CHKERRQ(ierr);
  }

  /* Restore vectors */
  ierr = VecAssemblyBegin(b);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(b);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "ComputeMatrix_MOAB"
PetscErrorCode ComputeMatrix_MOAB(KSP ksp,Mat J,Mat jac,void *ctx)
{
  UserContext       *user = (UserContext*)ctx;
  DM                dm;
  PetscInt          i,j,q,num_conn;
  PetscInt          dof_indices[VPERE];
  PetscReal         vpos[VPERE*3],quadrature[NQPTS*3],jxw[NQPTS];
  PetscBool         dbdry[VPERE];
  const moab::EntityHandle *connect;
  const moab::Range *elocal;
  moab::Interface*  mbImpl;
  PetscBool         elem_on_boundary;
  PetscScalar       array[VPERE*VPERE];
  PetscReal         phi[VPERE*NQPTS], dphidx[VPERE*NQPTS], dphidy[VPERE*NQPTS];
  PetscReal         rho;
  PetscErrorCode    ierr;
 
  PetscFunctionBeginUser;
  ierr      = KSPGetDM(ksp,&dm);CHKERRQ(ierr);

  /* get the essential MOAB mesh related quantities needed for FEM assembly */  
  ierr = DMMoabGetInterface(dm, &mbImpl);CHKERRQ(ierr);
  ierr = DMMoabGetLocalElements(dm, &elocal);CHKERRQ(ierr);

  /* loop over local elements */
  for(moab::Range::iterator iter = elocal->begin(); iter != elocal->end(); iter++) {
    const moab::EntityHandle ehandle = *iter;

    // Get connectivity information:
    ierr = DMMoabGetElementConnectivity(dm, ehandle, &num_conn, &connect);CHKERRQ(ierr);
    if (num_conn != VPERE) SETERRQ1(PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG, "Only QUAD4 element bases are supported in the current example. Connectivity=%D.\n", num_conn);

    /* compute the mid-point of the element and use a 1-point lumped quadrature */
    ierr = DMMoabGetVertexCoordinates(dm,num_conn,connect,vpos);CHKERRQ(ierr);

    /* get the global DOF number to appropriately set the element contribution in the RHS vector */
#ifdef LOCAL_ASSEMBLY
    ierr = DMMoabGetFieldDofsLocal(dm, num_conn, connect, 0, dof_indices);CHKERRQ(ierr);
#else
    ierr = DMMoabGetFieldDofs(dm, num_conn, connect, 0, dof_indices);CHKERRQ(ierr);
#endif

    /* compute the quadrature points transformed to the physical space */
    ierr = ComputeQuadraturePointsPhysical(vpos, quadrature, jxw);CHKERRQ(ierr);

    /* compute the basis functions and the derivatives wrt x and y directions */
    ierr = Compute_Quad4_Basis(vpos, NQPTS, quadrature, phi,  dphidx, dphidy);CHKERRQ(ierr);

    /* compute the inhomogeneous diffusion coefficient at the first quadrature point 
        -- for large spatial variations, embed this property evaluation inside quadrature loop */
    ierr  = ComputeRho_MOAB(quadrature, user->rho, &rho);CHKERRQ(ierr);

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

    /* check if element is on the boundary */
    ierr = DMMoabIsEntityOnBoundary(dm,ehandle,&elem_on_boundary);CHKERRQ(ierr);

    /* apply dirichlet boundary conditions */
    if (elem_on_boundary && user->bcType == DIRICHLET) {

      /* get the list of nodes on boundary so that we can enforce dirichlet conditions strongly */
      ierr = DMMoabCheckBoundaryVertices(dm,num_conn,connect,dbdry);CHKERRQ(ierr);

      for (i=0; i < VPERE; ++i) {
        if (dbdry[i]) {  /* dirichlet node */
          /* think about strongly imposing dirichlet */
          for (j=0; j < VPERE; ++j) {
            /* TODO: symmetrize the system - need the RHS */
            array[i*VPERE+j] = 0.0;
          }
          array[i*VPERE+i] = 1.0;
        }
      }
    }

#ifdef LOCAL_ASSEMBLY
    /* set the values directly into appropriate locations. Can alternately use VecSetValues */
    ierr = MatSetValuesLocal(jac, VPERE, dof_indices, VPERE, dof_indices, array, ADD_VALUES);CHKERRQ(ierr);
#else
    ierr = MatSetValues(jac, VPERE, dof_indices, VPERE, dof_indices, array, ADD_VALUES);CHKERRQ(ierr);
#endif
  }

  ierr = MatAssemblyBegin(jac,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(jac,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  if (user->bcType == NEUMANN) {
    MatNullSpace nullspace;

    ierr = MatNullSpaceCreate(PETSC_COMM_WORLD,PETSC_TRUE,0,0,&nullspace);CHKERRQ(ierr);
    ierr = MatSetNullSpace(J,nullspace);CHKERRQ(ierr);
    ierr = MatNullSpaceDestroy(&nullspace);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}


/*
*  Purpose: BASIS_MN_Q4: all bases at N points for a Q4 element.
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
*    Input, PetscScalar Q[3*4], the coordinates of the vertices.
*    It is common to list these points in counter clockwise order.
*
*    Input, int N, the number of evaluation points.
*
*    Input, PetscScalar P[3*N], the evaluation points.
*
*    Output, PetscScalar PHI[4*N], the bases at the evaluation points.
*
*    Output, PetscScalar DPHIDX[4*N], DPHIDY[4*N], the derivatives of the
*    bases at the evaluation points.
*
*  Original Author: John Burkardt (http://people.sc.fsu.edu/~jburkardt/cpp_src/fem2d_pack/fem2d_pack.cpp)
*  Modified by Vijay Mahadevan
*/
#undef __FUNCT__
#define __FUNCT__ "Compute_Quad4_Basis"
PetscErrorCode Compute_Quad4_Basis ( PetscReal coords[VPERE*3], PetscInt n, PetscReal *pts, PetscReal *phi, PetscReal *dphidx, PetscReal *dphidy)
{
  PetscReal ejac;
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
    PetscScalar phisum=0,dphixsum=0,dphiysum=0;
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


/*
*  Purpose: Compute the quadrature points in the physical space with appropriate transformation for QUAD4 elements.
*
*  Parameters:
*
*    Input, PetscScalar verts[VPERE*4], the coordinates of the vertices.
*    It is common to list these points in counter clockwise order.
*
*    Output, PetscScalar quad[3*NQPTS], the physical quadrature evaluation point.
*
*    Output, PetscScalar jxw[NQPTS], the product of Jacobian of the physical element times the weights at the quadrature points.
*/
#undef __FUNCT__
#define __FUNCT__ "ComputeQuadraturePointsPhysical"
PetscErrorCode ComputeQuadraturePointsPhysical(const PetscReal verts[VPERE*3], PetscReal quad[NQPTS*3], PetscReal jxw[NQPTS])
{
  int i,j;
  PetscReal centroid[3];
  const PetscReal GLG_QUAD[3] = {-0.577350269189625764509148780502, 0.577350269189625764509148780502, 1.0};
  PetscReal dx = fabs(verts[0+2*3] - verts[0+0*3])/2, dy = fabs( verts[1+2*3] - verts[1+0*3] )/2;
  PetscReal ejac = dx*dy;
  
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
      quad[(i*NQPTS1D+j)*3] = centroid[0]+dx*(GLG_QUAD[i]);
      quad[(i*NQPTS1D+j)*3+1] = centroid[1]+dy*(GLG_QUAD[j]);
      quad[(i*NQPTS1D+j)*3+2] = centroid[2];
      jxw[i*NQPTS1D+j] = GLG_QUAD[NQPTS1D]*ejac;
    }
  }
  PetscFunctionReturn(0);
}

