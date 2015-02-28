/*T
   Concepts: KSP^solving a system of linear equations using a MOAB based DM implementation.
   Concepts: KSP^Laplacian, 3d
   Processors: n
T*/

/*
Inhomogeneous Laplacian in 3-D. Modeled by the partial differential equation

   -div \rho grad u = f,  0 < x,y < 1,

with forcing function

   f = e^{-((x-xr)^2+(y-yr)^2)/\nu}

with Dirichlet boundary conditions

   u = f(x,y) for x = 0, x = 1, y = 0, y = 1

or pure Neumman boundary conditions

Usage:
    ./ex36 -bc_type dirichlet -nu .01 -rho .01 -file input/quad_2p.h5m -dmmb_rw_dbg 0 -n 50

*/

static char help[] = "\
                      Solves a three dimensional inhomogeneous Laplacian equation with a Gaussian source.\n \
                      Usage: ./ex36 -bc_type dirichlet -nu .01 -n 10\n";

#define PROBLEM 2

/* PETSc includes */
#include <petscksp.h>
#include <petscdmmoab.h>

static PetscErrorCode ComputeMatrix_MOAB(KSP,Mat,Mat,void*);
static PetscErrorCode ComputeRHS_MOAB(KSP,Vec,void*);

static PetscErrorCode Compute_Basis ( int nverts, PetscReal *coords/*nverts*3*/, PetscInt npts, PetscReal *quad/*npts*3*/, PetscReal *pts/*npts*3*/, 
        PetscReal *jxw/*npts*/, PetscReal *phi/*npts*/, PetscReal *dphidx/*npts*/, PetscReal *dphidy/*npts*/, PetscReal *dphidz/*npts*/);

typedef enum {DIRICHLET, NEUMANN} BCType;

typedef struct {
  PetscInt  dim,n,nlevels;
  PetscReal rho;
  PetscReal xyzref[3];
  PetscReal nu;
  BCType    bcType;
  char filename[PETSC_MAX_PATH_LEN];

  /* Discretization parameters */
  int NQPTS,VPERE;
} UserContext;

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  KSP            ksp;
  DM             dm,dmref;
  DM            *dmhierarchy;

  PC        pc;
  PetscInt  k;
  Mat       R;

  UserContext    user;
  const char     *bcTypes[2] = {"dirichlet","neumann"};
  const char     *fields[1] = {"T-Variable"};
  PetscErrorCode ierr;
  PetscInt       bc,np;
  Vec            b,x;
  PetscBool      use_extfile,io,usesimplex,usemg;

  PetscInitialize(&argc,&argv,(char*)0,help);

  MPI_Comm_size(PETSC_COMM_WORLD,&np);

  ierr        = PetscOptionsBegin(PETSC_COMM_WORLD, "", "Options for the inhomogeneous Poisson equation", "ex36.cxx");
  user.dim    = 3;
  //ierr        = PetscOptionsInt("-dim", "The dimension of the problem", "ex36.cxx", user.dim, &user.dim, NULL);CHKERRQ(ierr);
  user.n      = 2;
  ierr        = PetscOptionsInt("-n", "The elements in each direction", "ex36.cxx", user.n, &user.n, NULL);CHKERRQ(ierr);
  user.nlevels= 2;
  ierr        = PetscOptionsInt("-levels", "Number of levels in the multigrid hierarchy", "ex36.cxx", user.nlevels, &user.nlevels, NULL);CHKERRQ(ierr);
  user.rho    = 0.5;
  ierr        = PetscOptionsReal("-rho", "The conductivity", "ex36.cxx", user.rho, &user.rho, NULL);CHKERRQ(ierr);
  user.xyzref[0]   = user.xyzref[1]   = user.xyzref[2]   = 0.5;
  ierr        = PetscOptionsReal("-xyzref", "The x-coordinate of Gaussian center", "ex36.cxx", user.xyzref[0], &user.xyzref[0], NULL);CHKERRQ(ierr);
  user.xyzref[1]   = user.xyzref[2]   = user.xyzref[0];
  user.nu     = 0.1;
  ierr        = PetscOptionsReal("-nu", "The width of the Gaussian source", "ex36.cxx", user.nu, &user.nu, NULL);CHKERRQ(ierr);
  io          = PETSC_FALSE;
  usemg       = PETSC_FALSE;
  ierr        = PetscOptionsBool("-mg", "Use multigrid preconditioner", "ex36.cxx", usemg, &usemg, NULL);CHKERRQ(ierr);
  ierr        = PetscOptionsBool("-io", "Write out the solution and mesh data", "ex36.cxx", io, &io, NULL);CHKERRQ(ierr);
  bc          = (PetscInt)DIRICHLET;
  ierr        = PetscOptionsEList("-bc_type","Type of boundary condition","ex36.cxx",bcTypes,2,bcTypes[0],&bc,NULL);CHKERRQ(ierr);
  user.bcType = (BCType)bc;
  ierr        = PetscOptionsString("-file", "The mesh file for the problem", "ex36.cxx", "",user.filename,PETSC_MAX_PATH_LEN,&use_extfile);CHKERRQ(ierr);

  usesimplex=PETSC_FALSE;
  ierr        = PetscOptionsBool("-tet", "Use tetrahedra to discretize the unit cube domain", "ex36.cxx", usesimplex, &usesimplex, NULL);CHKERRQ(ierr);
  user.NQPTS  = (usesimplex ? 4:8);
  ierr        = PetscOptionsInt("-nq", "Number of quadrature points to be used in the FEM discretization", "ex36.cxx", user.NQPTS, &user.NQPTS, NULL);CHKERRQ(ierr);
  ierr        = PetscOptionsEnd();

  user.VPERE=(usesimplex ? 4:8);

  /* Create the DM object from either a mesh file or from in-memory structured grid */
  if (use_extfile) {
    ierr = DMMoabLoadFromFile(PETSC_COMM_WORLD, user.dim, user.filename, (np==1 ? "" : ""), &dm);CHKERRQ(ierr);
  }
  else {
    ierr = DMMoabCreateBoxMesh(PETSC_COMM_WORLD, user.dim, usesimplex, NULL, user.n, 1, &dm);CHKERRQ(ierr);
  }
  ierr = DMSetFromOptions(dm);CHKERRQ(ierr);
  ierr = DMMoabSetFieldNames(dm, 1, fields);CHKERRQ(ierr);

  using namespace moab;
  moab::ErrorCode error;
  moab::Range _verts, _edges, _faces, _cells;

  moab::Interface*  mb;
  ierr = DMMoabGetInterface(dm, &mb);CHKERRQ(ierr);
  /* Get all entities by dimension on the rootset with recursion turned on */
  error = mb->get_entities_by_dimension( 0, 0, _verts, true);MB_CHK_ERR(error);
  error = mb->get_entities_by_dimension( 0, 1, _edges, true);MB_CHK_ERR(error);
  error = mb->get_entities_by_dimension( 0, 2, _faces, true);MB_CHK_ERR(error);
  error = mb->get_entities_by_dimension( 0, 3, _cells, true);MB_CHK_ERR(error);
  PetscPrintf(PETSC_COMM_WORLD, "%d %d %d %d\n", _verts.size(), _edges.size(), _faces.size(), _cells.size());

  /* SetUp the data structures for DMMOAB */
  ierr = DMSetUp(dm);CHKERRQ(ierr);

  ierr = DMSetFromOptions(dm);CHKERRQ(ierr);

  ierr = DMSetApplicationContext(dm, &user);CHKERRQ(ierr);

  ierr = KSPCreate(PETSC_COMM_WORLD,&ksp);CHKERRQ(ierr);
  ierr = KSPSetComputeRHS(ksp,ComputeRHS_MOAB,&user);CHKERRQ(ierr);
  ierr = KSPSetComputeOperators(ksp,ComputeMatrix_MOAB,&user);CHKERRQ(ierr);

  if (user.nlevels)
  {
    KSPGetPC(ksp,&pc);
    ierr = PetscMalloc(sizeof(DM)*(user.nlevels+1),&dmhierarchy);
    for (k=0; k<=user.nlevels; k++) dmhierarchy[k] = NULL;

    PetscPrintf(PETSC_COMM_WORLD, "Number of levels: %d\n", user.nlevels);
    ierr = DMMoabGenerateHierarchy(dm,user.nlevels,PETSC_NULL);CHKERRQ(ierr);

    // coarsest grid = 0
    // finest grid = nlevels
    dmhierarchy[0] = dm;
    PetscBool usehierarchy=PETSC_FALSE;
    if (usehierarchy) {
      ierr = DMRefineHierarchy(dm,user.nlevels,&dmhierarchy[1]);CHKERRQ(ierr);
    }
    else {
      PetscPrintf(PETSC_COMM_WORLD, "Level %D\n", 0);
      for (k=1; k<=user.nlevels; k++) {
        PetscPrintf(PETSC_COMM_WORLD, "Level %D\n", k);
        ierr = DMRefine(dmhierarchy[k-1],MPI_COMM_NULL,&dmhierarchy[k]);CHKERRQ(ierr);
      }
    }
    dmref = dmhierarchy[user.nlevels];
    PetscObjectReference((PetscObject)dmref);

    ierr = KSPSetDM(ksp,dmref);CHKERRQ(ierr);
    if (usemg) {
      ierr = PCSetType(pc,PCMG);CHKERRQ(ierr);
      ierr = PCMGSetType(pc,PC_MG_MULTIPLICATIVE);CHKERRQ(ierr);
      ierr = PCMGSetLevels(pc,user.nlevels+1,NULL);CHKERRQ(ierr);
      ierr = PCMGSetGalerkin(pc,PETSC_TRUE);CHKERRQ(ierr);
      ierr = PCMGSetCycleType(pc, PC_MG_CYCLE_V);CHKERRQ(ierr);
      ierr = PCMGSetNumberSmoothUp(pc,2);CHKERRQ(ierr);
      ierr = PCMGSetNumberSmoothDown(pc,2);CHKERRQ(ierr);

      for (k=1; k<=user.nlevels; k++) {
        ierr = DMCreateInterpolation(dmhierarchy[k-1],dmhierarchy[k],&R,NULL);CHKERRQ(ierr);
        //MatView(R,0);
        ierr = PCMGSetInterpolation(pc,k,R);CHKERRQ(ierr);
        //ierr = PCMGSetRestriction(pc,k,R);CHKERRQ(ierr);
        ierr = MatDestroy(&R);CHKERRQ(ierr);
      }
    }

    for (k=1; k<=user.nlevels; k++) {
      ierr = DMDestroy(&dmhierarchy[k]);CHKERRQ(ierr);
    }
    ierr = PetscFree(dmhierarchy);CHKERRQ(ierr);
  }
  else {
    dmref = dm;
    PetscObjectReference((PetscObject)dmref);
    ierr = KSPSetDM(ksp,dmref);CHKERRQ(ierr);
  }

  ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);

  /* Perform the actual solve */
  ierr = KSPSolve(ksp,NULL,NULL);CHKERRQ(ierr);
  ierr = KSPGetSolution(ksp,&x);CHKERRQ(ierr);
  ierr = KSPGetRhs(ksp,&b);CHKERRQ(ierr);

  if (io) {
    /* Write out the solution along with the mesh */
    ierr = DMMoabSetGlobalFieldVector(dmref, x);CHKERRQ(ierr);
#ifdef MOAB_HDF5_H
    ierr = DMMoabOutput(dmref, "ex36.h5m", "");CHKERRQ(ierr);
#else
    /* MOAB does not support true parallel writers that aren't HDF5 based
       And so if you are using VTK as the output format in parallel,
       the data could be jumbled due to the order in which the processors
       write out their parts of the mesh and solution tags
    */
    ierr = DMMoabOutput(dmref, "ex36.vtk", "");CHKERRQ(ierr);
#endif
  }

  /* Cleanup objects */
  ierr = KSPDestroy(&ksp);CHKERRQ(ierr);
  ierr = DMDestroy(&dmref);CHKERRQ(ierr);
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return 0;
}


#undef __FUNCT__
#define __FUNCT__ "ComputeRho_MOAB"
PetscReal ComputeRho_MOAB(PetscReal coords[3], PetscReal centerRho)
{
#if (PROBLEM == 1)
  if ((coords[0] > 1.0/3.0) && (coords[0] < 2.0/3.0) && 
      (coords[1] > 1.0/3.0) && (coords[1] < 2.0/3.0) && 
      (coords[2] > 1.0/3.0) && (coords[2] < 2.0/3.0))
  {
    return centerRho;
  }
  else
#endif
    return 1.0;
}

double forcing_function(PetscReal coords[3], PetscReal nu, PetscReal cdref[3])
{
#if (PROBLEM == 1)
  const PetscScalar xx=(coords[0]-cdref[0])*(coords[0]-cdref[0]);
  const PetscScalar yy=(coords[1]-cdref[1])*(coords[1]-cdref[1]);
  const PetscScalar zz=(coords[2]-cdref[2])*(coords[2]-cdref[2]);
  return PetscExpScalar(-(xx+yy+zz)/nu);
#else
  return PI*PI*(sin(PI*coords[0])+sin(PI*coords[1])+sin(PI*coords[2]));
#endif
}

#undef __FUNCT__
#define __FUNCT__ "ComputeRHS_MOAB"
PetscErrorCode ComputeRHS_MOAB(KSP ksp,Vec b,void *ptr)
{
  UserContext*      user = (UserContext*)ptr;
  DM                dm;
  PetscInt          dof_indices[user->VPERE];
  PetscBool         dbdry[user->VPERE];
  PetscReal         vpos[user->VPERE*3],quadrature[user->NQPTS*3],phypts[user->NQPTS*3];
  PetscReal         jxw[user->NQPTS],phi[user->VPERE*user->NQPTS];
  PetscInt          i,q,num_conn;
  const moab::EntityHandle *connect;
  const moab::Range *elocal;
  moab::Interface*  mbImpl;
  PetscScalar       localv[user->VPERE];
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

    ierr = PetscMemzero(localv,sizeof(PetscScalar)*user->VPERE);CHKERRQ(ierr);

    // Get connectivity information:
    ierr = DMMoabGetElementConnectivity(dm, ehandle, &num_conn, &connect);CHKERRQ(ierr);
    if (num_conn != user->VPERE) SETERRQ1(PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG, "Only HEX8 element bases are supported in the current example. n(Connectivity)=%D.\n", num_conn);

    /* compute the mid-point of the element and use a 1-point lumped quadrature */
    ierr = DMMoabGetVertexCoordinates(dm,num_conn,connect,vpos);CHKERRQ(ierr);

    /* get the global DOF number to appropriately set the element contribution in the RHS vector */
    ierr = DMMoabGetFieldDofsLocal(dm, num_conn, connect, 0, dof_indices);CHKERRQ(ierr);

    /* compute the quadrature points transformed to the physical space and then
       compute the basis functions to compute local operators */
    ierr = Compute_Basis(user->VPERE, vpos, user->NQPTS, quadrature, phypts, jxw, phi, 0, 0, 0);CHKERRQ(ierr);

    /* Compute function over the locally owned part of the grid */
    for (q=0; q<user->NQPTS; ++q) {
      const double ff = forcing_function(&quadrature[3*q], user->nu, user->xyzref);
      for (i=0; i < user->VPERE; ++i) {
        localv[i] += jxw[q] * phi[q*user->VPERE+i] * ff;
      }
    }

    /* check if element is on the boundary */
    //ierr = DMMoabIsEntityOnBoundary(dm,ehandle,&elem_on_boundary);CHKERRQ(ierr);
    ierr = DMMoabCheckBoundaryVertices(dm,num_conn,connect,dbdry);CHKERRQ(ierr);
    elem_on_boundary=PETSC_FALSE;
    for (i=0; i< user->VPERE; ++i)
      if (dbdry[i]) elem_on_boundary=PETSC_TRUE;

    /* apply dirichlet boundary conditions */
    if (elem_on_boundary && user->bcType == DIRICHLET) {

      /* get the list of nodes on boundary so that we can enforce dirichlet conditions strongly */
      //ierr = DMMoabCheckBoundaryVertices(dm,num_conn,connect,dbdry);CHKERRQ(ierr);

      for (i=0; i < user->VPERE; ++i) {
        if (dbdry[i]) {  /* dirichlet node */
          /* think about strongly imposing dirichlet */
          localv[i] = forcing_function(&vpos[3*i], user->nu, user->xyzref);
        }
      }
    }

    /* set the values directly into appropriate locations. Can alternately use VecSetValues */
    ierr = VecSetValuesLocal(b, user->VPERE, dof_indices, localv, ADD_VALUES);CHKERRQ(ierr);
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
  //VecView(b,0);
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "ComputeMatrix_MOAB"
PetscErrorCode ComputeMatrix_MOAB(KSP ksp,Mat J,Mat jac,void *ctx)
{
  UserContext       *user = (UserContext*)ctx;
  DM                dm;
  PetscInt          i,j,q,num_conn;
  PetscInt          dof_indices[user->VPERE];
  PetscReal         vpos[user->VPERE*3],quadrature[user->NQPTS*3],phypts[user->NQPTS*3];
  PetscBool         dbdry[user->VPERE];
  const moab::EntityHandle *connect;
  const moab::Range *elocal;
  moab::Interface*  mbImpl;
  PetscBool         elem_on_boundary;
  PetscScalar       array[user->VPERE*user->VPERE];
  PetscReal         jxw[user->NQPTS],phi[user->VPERE*user->NQPTS];
  PetscReal         dphidx[user->VPERE*user->NQPTS], dphidy[user->VPERE*user->NQPTS], dphidz[user->VPERE*user->NQPTS];
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
    if (num_conn != user->VPERE) SETERRQ1(PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG, "Only HEX8 element bases are supported in the current example. Connectivity=%D.\n", num_conn);

    /* compute the mid-point of the element and use a 1-point lumped quadrature */
    ierr = DMMoabGetVertexCoordinates(dm,user->VPERE,connect,vpos);CHKERRQ(ierr);

    /* get the global DOF number to appropriately set the element contribution in the RHS vector */
    ierr = DMMoabGetFieldDofsLocal(dm, user->VPERE, connect, 0, dof_indices);CHKERRQ(ierr);

    /* compute the quadrature points transformed to the physical space and
       compute the basis functions and the derivatives wrt x, y and z directions */
    ierr = Compute_Basis(user->VPERE, vpos, user->NQPTS, quadrature, phypts, jxw, phi, dphidx, dphidy, dphidz);CHKERRQ(ierr);

    ierr = PetscMemzero(array, user->VPERE*user->VPERE*sizeof(PetscScalar));

    /* Compute function over the locally owned part of the grid */
    for (q=0; q < user->NQPTS; ++q) {

      /* compute the inhomogeneous diffusion coefficient at the first quadrature point 
          -- for large spatial variations, embed this property evaluation inside quadrature loop */
      rho = ComputeRho_MOAB(&phypts[q*3], user->rho);

      for (i=0; i < user->VPERE; ++i) {
        for (j=0; j < user->VPERE; ++j) {
          array[i*user->VPERE+j] += jxw[q] * rho * ( dphidx[q*user->VPERE+i]*dphidx[q*user->VPERE+j] + 
                                               dphidy[q*user->VPERE+i]*dphidy[q*user->VPERE+j] +
                                               dphidz[q*user->VPERE+i]*dphidz[q*user->VPERE+j] );
        }
      }
    }

    /* check if element is on the boundary */
    //ierr = DMMoabIsEntityOnBoundary(dm,ehandle,&elem_on_boundary);CHKERRQ(ierr);
    ierr = DMMoabCheckBoundaryVertices(dm,user->VPERE,connect,dbdry);CHKERRQ(ierr);
    elem_on_boundary=PETSC_FALSE;
    for (i=0; i < user->VPERE; ++i)
      if (dbdry[i]) elem_on_boundary=PETSC_TRUE;

    /* apply dirichlet boundary conditions */
    if (elem_on_boundary && user->bcType == DIRICHLET) {

      /* get the list of nodes on boundary so that we can enforce dirichlet conditions strongly */
      //ierr = DMMoabCheckBoundaryVertices(dm,user->VPERE,connect,dbdry);CHKERRQ(ierr);

      for (i=0; i < user->VPERE; ++i) {
        if (dbdry[i]) {  /* dirichlet node */
          // PetscPrintf (PETSC_COMM_WORLD, "At the boundary:= dof_indices[%d] = %d\n", i, dof_indices[i]);
          /* think about strongly imposing dirichlet */
          for (j=0; j < user->VPERE; ++j) {
            /* TODO: symmetrize the system - need the RHS */
            array[i*user->VPERE+j] = 0.0;
          }
          array[i*user->VPERE+i] = 1.0;
        }
      }
    }

    /* set the values directly into appropriate locations. Can alternately use VecSetValues */
    ierr = MatSetValuesLocal(jac, user->VPERE, dof_indices, user->VPERE, dof_indices, array, ADD_VALUES);CHKERRQ(ierr);
  }

  ierr = MatAssemblyBegin(jac,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(jac,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  if (user->bcType == NEUMANN) {
    MatNullSpace nullspace;

    ierr = MatNullSpaceCreate(PETSC_COMM_WORLD,PETSC_TRUE,0,0,&nullspace);CHKERRQ(ierr);
    ierr = MatSetNullSpace(jac,nullspace);CHKERRQ(ierr);
    ierr = MatNullSpaceDestroy(&nullspace);CHKERRQ(ierr);
  }
  //MatView(jac,0);
  PetscFunctionReturn(0);
}


double determinant_mat_3x3 ( PetscReal inmat[3*3] )
{
  return   inmat[0]*(inmat[8]*inmat[4]-inmat[7]*inmat[5])
         - inmat[3]*(inmat[8]*inmat[1]-inmat[7]*inmat[2])
         + inmat[6]*(inmat[5]*inmat[1]-inmat[4]*inmat[2]);
}

/*
*  Purpose: Compute_Basis: all bases at N points for a HEX8 element.
*
*  Discussion:
*
*    The routine is given the coordinates of the vertices of a hexahedra.
*    It works directly with these coordinates, and does not refer to a 
*    reference element.
*
*    The sides of the element are presumed to lie along coordinate axes.
*
*    The routine evaluates the basis functions associated with each corner,
*    and their derivatives with respect to X and Y.
*
*  Physical Element HEX8:
*
*      8------7        t  s
*     /|     /|        | /
*    5------6 |        |/
*    | |    | |        0-------r
*    | 4----|-3        
*    |/     |/        
*    1------2        
*     
*
*  Parameters:
*
*    Input, PetscScalar Q[3*8], the coordinates of the vertices.
*    It is common to list these points in counter clockwise order.
*
*    Input, int N, the number of evaluation points.
*
*    Input, PetscScalar P[3*N], the evaluation points.
*
*    Output, PetscScalar PHI[8*N], the bases at the evaluation points.
*
*    Output, PetscScalar DPHIDX[8*N], DPHIDY[8*N], the derivatives of the
*    bases at the evaluation points.
*
*  Original Author: John Burkardt (http://people.sc.fsu.edu/~jburkardt/cpp_src/fem3d_pack/fem3d_pack.cpp)
*  Modified by Vijay Mahadevan
*/
#undef __FUNCT__
#define __FUNCT__ "Compute_Basis"
PetscErrorCode Compute_Basis ( int nverts, PetscReal *coords/*nverts*3*/, PetscInt npts, PetscReal *quad/*npts*3*/, PetscReal *pts/*npts*3*/, 
        PetscReal *jxw/*npts*/, PetscReal *phi/*npts*/, PetscReal *dphidx/*npts*/, PetscReal *dphidy/*npts*/, PetscReal *dphidz/*npts*/)
{
  PetscReal ejac;
  int i,j,k;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  if (nverts == 8) {

    /* 3-D 2-point tensor product Gaussian quadrature */
    quad[0]=-0.5773502691896257; quad[1]=-0.5773502691896257;  quad[2]=-0.5773502691896257;
    quad[3]=-0.5773502691896257; quad[4]=0.5773502691896257;   quad[5]=-0.5773502691896257;
    quad[6]=0.5773502691896257;  quad[7]=-0.5773502691896257;  quad[8]=-0.5773502691896257;
    quad[9]=0.5773502691896257;  quad[10]=0.5773502691896257;  quad[11]=-0.5773502691896257;

    quad[12]=-0.5773502691896257; quad[13]=-0.5773502691896257;  quad[14]=0.5773502691896257;
    quad[15]=-0.5773502691896257; quad[16]=0.5773502691896257;   quad[17]=0.5773502691896257;
    quad[18]=0.5773502691896257;  quad[19]=-0.5773502691896257;  quad[20]=0.5773502691896257;
    quad[21]=0.5773502691896257;  quad[22]=0.5773502691896257;   quad[23]=0.5773502691896257;
    /* transform quadrature bounds: [-1, 1] => [0, 1] */
    for (i=0; i<npts*3; ++i) {
      quad[i]=0.5*quad[i]+0.5;
      pts[i] = 0.0;
    }
    ierr = PetscMemzero(pts,npts*3*sizeof(PetscReal));CHKERRQ(ierr);

    for (j=0;j<npts;j++)
    {
      const int offset = j*nverts;
      const double& r = quad[j*3+0];
      const double& s = quad[j*3+1];
      const double& t = quad[j*3+2];

      phi[offset+0] = ( 1.0 - r ) * ( 1.0 - s ) * ( 1.0 - t );
      phi[offset+1] =         r   * ( 1.0 - s ) * ( 1.0 - t );
      phi[offset+2] =         r   *         s   * ( 1.0 - t );
      phi[offset+3] = ( 1.0 - r ) *         s   * ( 1.0 - t );
      phi[offset+4] = ( 1.0 - r ) * ( 1.0 - s ) *         t;
      phi[offset+5] =         r   * ( 1.0 - s ) *         t;
      phi[offset+6] =         r   *         s   *         t;
      phi[offset+7] = ( 1.0 - r ) *         s   *         t;

      const double dNi_dxi[8]  = { - ( 1.0 - s ) * ( 1.0 - t ),
                                     ( 1.0 - s ) * ( 1.0 - t ),
                                             s   * ( 1.0 - t ),
                                   -         s   * ( 1.0 - t ),
                                   - ( 1.0 - s ) *         t,
                                     ( 1.0 - s ) *         t,
                                             s   *         t,
                                   -         s   *         t };

      const double dNi_deta[8]  = { - ( 1.0 - r ) * ( 1.0 - t ),
                                    -         r   * ( 1.0 - t ),
                                              r   * ( 1.0 - t ),
                                      ( 1.0 - r ) * ( 1.0 - t ),
                                    - ( 1.0 - r ) *         t,
                                    -         r   *         t,
                                              r   *         t,
                                      ( 1.0 - r ) *         t};

      const double dNi_dzeta[8]  = { - ( 1.0 - r ) * ( 1.0 - s ),
                                     -         r   * ( 1.0 - s ),
                                     -         r   *         s  ,
                                     - ( 1.0 - r ) *         s  ,
                                       ( 1.0 - r ) * ( 1.0 - s ),
                                               r   * ( 1.0 - s ),
                                               r   *         s  ,
                                       ( 1.0 - r ) *         s  };

      /* Reset arrays. */
      for ( i = 0; i < nverts; i++ ) {
        if (dphidx) dphidx[i+offset] = 0.0;
        if (dphidy) dphidy[i+offset] = 0.0;
        if (dphidz) dphidz[i+offset] = 0.0;
      }

      double jacobian[9];
      for ( i = 0; i < 9; i++ ) jacobian[i]=0.0;

      for (i = 0; i < nverts; ++i) {
        const PetscScalar* vertex = coords+i*3;
        jacobian[0] += dNi_dxi[i]   * vertex[0];
        jacobian[3] += dNi_dxi[i]   * vertex[1];
        jacobian[6] += dNi_dxi[i]   * vertex[2];
        jacobian[1] += dNi_deta[i]  * vertex[0];
        jacobian[4] += dNi_deta[i]  * vertex[1];
        jacobian[7] += dNi_deta[i]  * vertex[2];
        jacobian[2] += dNi_dzeta[i] * vertex[0];
        jacobian[5] += dNi_dzeta[i] * vertex[1];
        jacobian[8] += dNi_dzeta[i] * vertex[2];
        for (int k = 0; k < 3; ++k) pts[3*j+k] += phi[i+offset] * vertex[k];
      }
      ejac = determinant_mat_3x3 (jacobian);
      jxw[j] = ejac/(nverts);

      /*  Divide by element jacobian. */
      for ( i = 0; i < nverts; i++ ) {
        if (dphidx) dphidx[i+offset] = dNi_dxi[i] / pow(ejac,1.0/3.0);
        if (dphidy) dphidy[i+offset] = dNi_deta[i] / pow(ejac,1.0/3.0);
        if (dphidz) dphidz[i+offset] = dNi_dzeta[i] / pow(ejac,1.0/3.0);
      }
    }
  }
  else if (nverts == 4) {
    // KEAST rule 2, order 4
    quad[0]=0.5854101966249685; quad[1]=0.1381966011250105; quad[2]=0.1381966011250105;
    quad[3]=0.1381966011250105; quad[4]=0.1381966011250105; quad[5]=0.1381966011250105;
    quad[6]=0.1381966011250105; quad[7]=0.1381966011250105; quad[8]=0.5854101966249685;
    quad[9]=0.1381966011250105; quad[10]=0.5854101966249685; quad[11]=0.1381966011250105;

    ierr = PetscMemzero(pts,npts*3*sizeof(PetscReal));CHKERRQ(ierr);

    //
    //           | x1 x2 x3 x4 |
    //  Volume = | y1 y2 y3 y4 |
    //           | z1 z2 z3 z4 |
    //           |  1  1  1  1 |
    //
    ejac =
        -(coords[0+0*3] * (
          coords[1+1*3] * ( coords[2+2*3] - coords[2+3*3] )   
        - coords[1+2*3] * ( coords[2+1*3] - coords[2+3*3] )   
        + coords[1+3*3] * ( coords[2+1*3] - coords[2+2*3] ) ) 
      - coords[0+1*3] * (
          coords[1+0*3] * ( coords[2+2*3] - coords[2+3*3] )   
        - coords[1+2*3] * ( coords[2+0*3] - coords[2+3*3] )   
        + coords[1+3*3] * ( coords[2+0*3] - coords[2+2*3] ) ) 
      + coords[0+2*3] * (
          coords[1+0*3] * ( coords[2+1*3] - coords[2+3*3] )   
        - coords[1+1*3] * ( coords[2+0*3] - coords[2+3*3] )   
        + coords[1+3*3] * ( coords[2+0*3] - coords[2+1*3] ) ) 
      - coords[0+3*3] * (
          coords[1+0*3] * ( coords[2+1*3] - coords[2+2*3] )   
        - coords[1+1*3] * ( coords[2+0*3] - coords[2+2*3] )   
        + coords[1+2*3] * ( coords[2+0*3] - coords[2+1*3] ) ));

    if ( ejac < 1e-8 ) SETERRQ1(PETSC_COMM_WORLD,PETSC_ERR_ARG_OUTOFRANGE,"Tetrahedral element has zero volume: %g. Degenerate element or invalid connectivity\n", ejac);

    for ( j = 0; j < npts; j++ )
    {
      const int offset = j*nverts;
      phi[offset+0] = 1.0 - quad[offset+0] - quad[offset+1] - quad[offset+2];
      phi[offset+1] = quad[offset+0];
      phi[offset+2] = quad[offset+1];
      phi[offset+3] = quad[offset+2];

      jxw[j] = ejac/(nverts)/6.0;

      const double dNi_dxi[4]  = { -1.0, 1.0, 0.0, 0.0 } ;
      const double dNi_deta[4]  = { -1.0, 0.0, 1.0, 0.0 } ;
      const double dNi_dzeta[4]  = { -1.0, 0.0, 0.0, 1.0 } ;

      /*  Divide by element jacobian. */
      for ( i = 0; i < nverts; i++ ) {
        if (dphidx) dphidx[i+offset] = dNi_dxi[i] / ejac / 24;
        if (dphidy) dphidy[i+offset] = dNi_deta[i] / ejac / 24;
        if (dphidz) dphidz[i+offset] = dNi_dzeta[i] / ejac / 24;
        const PetscScalar* vertex = coords+i*3;
        for (k = 0; k < 3; ++k) 
          pts[3*j+k] += phi[i+offset] * vertex[k];
      }
      //PetscPrintf(PETSC_COMM_WORLD, "\t Physical [%d, %g]: [x, y, z] = %g, %g, %g\n", 3*j, ejac, pts[3*j+0], pts[3*j+1], pts[3*j+2]);

/*
  //
  //             | xp x2 x3 x4 |
  //  Phi(1,P) = | yp y2 y3 y4 | / volume
  //             | zp z2 z3 z4 |
  //             |  1  1  1  1 |
  //
        phi[0+offset] = -(
          quad[0+j*3] * (
            coords[1+1*3] * ( coords[2+2*3] - coords[2+3*3] )
          - coords[1+2*3] * ( coords[2+1*3] - coords[2+3*3] )
          + coords[1+3*3] * ( coords[2+1*3] - coords[2+2*3] ) )
        - coords[0+1*3] * (
            quad[1+j*3] * ( coords[2+2*3] - coords[2+3*3] )
          - coords[1+2*3] * ( quad[2+j*3] - coords[2+3*3] )
          + coords[1+3*3] * ( quad[2+j*3] - coords[2+2*3] ) )
        + coords[0+2*3] * (
            quad[1+j*3] * ( coords[2+1*3] - coords[2+3*3] )
          - coords[1+1*3] * ( quad[2+j*3] - coords[2+3*3] )
          + coords[1+3*3] * ( quad[2+j*3] - coords[2+1*3] ) )
        - coords[0+3*3] * (
            quad[1+j*3] * ( coords[2+1*3] - coords[2+2*3] )
          - coords[1+1*3] * ( quad[2+j*3] - coords[2+2*3] )
          + coords[1+2*3] * ( quad[2+j*3] - coords[2+1*3] ) ) ) / volume;
  //
  //             | x1 xp x3 x4 |
  //  Phi(2,P) = | y1 yp y3 y4 | / volume
  //             | z1 zp z3 z4 |
  //             |  1  1  1  1 |
  //
      phi[1+offset] = -(
          coords[0+0*3] * (
            quad[1+j*3] * ( coords[2+2*3] - coords[2+3*3] )
          - coords[1+2*3] * ( quad[2+j*3] - coords[2+3*3] )
          + coords[1+3*3] * ( quad[2+j*3] - coords[2+2*3] ) )
        - quad[0+j*3]   * (
            coords[1+0*3] * ( coords[2+2*3] - coords[2+3*3] )
          - coords[1+2*3] * ( coords[2+0*3] - coords[2+3*3] )
          + coords[1+3*3] * ( coords[2+0*3] - coords[2+2*3] ) )
        + coords[0+2*3] * (
            coords[1+0*3] * ( quad[2+j*3] - coords[2+3*3] )
          - quad[1+j*3] * ( coords[2+0*3] - coords[2+3*3] )
          + coords[1+3*3] * ( coords[2+0*3] - quad[2+j*3] ) )
        - coords[0+3*3] * (
            coords[1+0*3] * ( quad[2+j*3] - coords[2+2*3] )
          - quad[1+j*3] * ( coords[2+0*3] - coords[2+2*3] )
          + coords[1+2*3] * ( coords[2+0*3] - quad[2+j*3] ) ) ) / volume;
  //
  //             | x1 x2 xp x4 |
  //  Phi(3,P) = | y1 y2 yp y4 | / volume
  //             | z1 z2 zp z4 |
  //             |  1  1  1  1 |
  //
      phi[2+offset] = -(
          coords[0+0*3] * (
            coords[1+1*3] * ( quad[2+j*3] - coords[2+3*3] )
          - quad[1+j*3] * ( coords[2+1*3] - coords[2+3*3] )
          + coords[1+3*3] * ( coords[2+1*3] - quad[2+j*3] ) )
        - coords[0+1*3] * (
            coords[1+0*3] * ( quad[2+j*3] - coords[2+3*3] )
          - quad[1+j*3] * ( coords[2+0*3] - coords[2+3*3] )
          + coords[1+3*3] * ( coords[2+0*3] - quad[2+j*3] ) )
        + quad[0+j*3] * (
            coords[1+0*3] * ( coords[2+1*3] - coords[2+3*3] )
          - coords[1+1*3] * ( coords[2+0*3] - coords[2+3*3] )
          + coords[1+3*3] * ( coords[2+0*3] - coords[2+1*3] ) )
        - coords[0+3*3] * (
            coords[1+0*3] * ( coords[2+1*3] - quad[2+j*3] )
          - coords[1+1*3] * ( coords[2+0*3] - quad[2+j*3] )
          + quad[1+j*3] * ( coords[2+0*3] - coords[2+1*3] ) ) ) / volume;
  //
  //             | x1 x2 x3 xp |
  //  Phi(4,P) = | y1 y2 y3 yp | / volume
  //             | z1 z2 z3 zp |
  //             |  1  1  1  1 |
  //
      phi[3+offset] = -(
          coords[0+0*3] * (
            coords[1+1*3] * ( coords[2+2*3] - quad[2+j*3] )
          - coords[1+2*3] * ( coords[2+1*3] - quad[2+j*3] )
          + quad[1+j*3] * ( coords[2+1*3] - coords[2+2*3] ) )
        - coords[0+1*3] * (
            coords[1+0*3] * ( coords[2+2*3] - quad[2+j*3] )
          - coords[1+2*3] * ( coords[2+0*3] - quad[2+j*3] )
          + quad[1+j*3] * ( coords[2+0*3] - coords[2+2*3] ) )
        + coords[0+2*3] * (
            coords[1+0*3] * ( coords[2+1*3] - quad[2+j*3] )
          - coords[1+1*3] * ( coords[2+0*3] - quad[2+j*3] )
          + quad[1+j*3] * ( coords[2+0*3] - coords[2+1*3] ) )
        - quad[0+j*3] * (
            coords[1+0*3] * ( coords[2+1*3] - coords[2+2*3] )
          - coords[1+1*3] * ( coords[2+0*3] - coords[2+2*3] )
          + coords[1+2*3] * ( coords[2+0*3] - coords[2+1*3] ) ) ) / volume;
*/

    }
  }
  else
  {
    SETERRQ1(PETSC_COMM_WORLD,PETSC_ERR_ARG_OUTOFRANGE,"The number of entity vertices are invalid. Currently only support HEX8: %D",nverts);
  }
#if 0
  /* verify if the computed basis functions are consistent */
  for ( j = 0; j < npts; j++ ) {
    PetscScalar phisum=0,dphixsum=0,dphiysum=0,dphizsum=0;
    const int offset = j*nverts;
    for ( i = 0; i < nverts; i++ ) {
      phisum += phi[i+offset];
      if (dphidx) dphixsum += dphidx[i+offset];
      if (dphidy) dphiysum += dphidy[i+offset];
      if (dphidz) dphizsum += dphidz[i+offset];
      if (dphidx) PetscPrintf(PETSC_COMM_WORLD, "\t Values [%d]: [phi, dphidx, dphidy, dphidz] = %g, %g, %g, %g\n", j, phi[i+offset], dphidx[i+offset], dphidy[i+offset], dphidz[i+offset]);
    }
    PetscPrintf(PETSC_COMM_WORLD, "Sum of basis at quadrature point %D = %g, %g, %g, %g\n", j, phisum, dphixsum, dphiysum, dphizsum);
  }
#endif
  PetscFunctionReturn(0);
}
