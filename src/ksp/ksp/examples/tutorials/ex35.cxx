/*T
   Concepts: KSP^solving a system of linear equations using a MOAB based DM implementation.
   Concepts: KSP^Laplacian, 2d
   Processors: n
T*/

/*
Inhomogeneous Laplacian in 2D. Modeled by the partial differential equation

   -div \rho grad u = f,  0 < x,y < 1,

Problem 1: (Default)

  Use the forcing function

     f = e^{-((x-xr)^2+(y-yr)^2)/\nu}

  with Dirichlet boundary conditions

     u = f(x,y) for x = 0, x = 1, y = 0, y = 1

  or pure Neumman boundary conditions

Problem 2:

  Use the following exact solution with Dirichlet boundary condition

    u = sin(pi*x)*sin(pi*y)

  and generate an appropriate forcing function to measure convergence.

Usage:

  Run with different values of \rho and \nu (problem 1) to control diffusion and gaussian source spread. This uses the internal mesh generator implemented in DMMoab.

    mpiexec -n $NP ./ex35 -n 20 -nu 0.02 -rho 0.01
    mpiexec -n $NP ./ex35 -n 40 -nu 0.01 -rho 0.005 -io -ksp_monitor
    mpiexec -n $NP ./ex35 -n 80 -nu 0.01 -rho 0.005 -io -ksp_monitor -pc_type hypre
    mpiexec -n $NP ./ex35 -n 160 -bc_type neumann -nu 0.005 -rho 0.01 -io
    mpiexec -n $NP ./ex35 -n 320 -bc_type neumann -nu 0.001 -rho 1 -io

  Measure convergence rate with uniform refinement with the options: "-problem 2 -error".

    mpiexec -n $NP ./ex35 -problem 2 -error -n 16 
    mpiexec -n $NP ./ex35 -problem 2 -error -n 32
    mpiexec -n $NP ./ex35 -problem 2 -error -n 64
    mpiexec -n $NP ./ex35 -problem 2 -error -n 128
    mpiexec -n $NP ./ex35 -problem 2 -error -n 256
    mpiexec -n $NP ./ex35 -problem 2 -error -n 512

  Now, you could alternately load an external MOAB mesh that discretizes the unit square and use that to run the solver.

    mpiexec -n $NP ./ex35 -problem 1 -file ./external_mesh.h5m 
    mpiexec -n $NP ./ex35 -problem 2 -file ./external_mesh.h5m -error
*/

static char help[] = "\
                      Solves 2D inhomogeneous Laplacian equation with a Gaussian source.\n \
                      Usage: ./ex35 -bc_type dirichlet -nu .01 -n 10\n";


/* PETSc includes */
#include <petscksp.h>
#include <petscdmmoab.h>

#define LOCAL_ASSEMBLY

const int NQPTS1D=2;
const int NQPTS=NQPTS1D*NQPTS1D;
const int VPERE=4;

extern PetscErrorCode ComputeMatrix(KSP,Mat,Mat,void*);
extern PetscErrorCode ComputeRHS(KSP,Vec,void*);

static PetscErrorCode Compute_Quad4_Basis ( PetscInt n, PetscReal *verts, PetscReal quad[NQPTS*3], PetscReal phypts[NQPTS*3], PetscReal jxw[NQPTS],
                                     PetscReal *phi, PetscReal *dphidx, PetscReal *dphidy);

typedef enum {DIRICHLET, NEUMANN} BCType;

typedef struct {
  PetscInt  dim,n,problem;
  PetscReal rho;
  PetscReal x,y;
  PetscReal xref,yref;
  PetscReal nu;
  BCType    bcType;
  char filename[PETSC_MAX_PATH_LEN];
} UserContext;

static PetscErrorCode ComputeDiscreteL2Error(KSP ksp,Vec err,UserContext *user);

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  KSP             ksp;
  DM              dm;
  UserContext     user;
  const char      *bcTypes[2] = {"dirichlet","neumann"};
  const char      *fields[1] = {"T-Variable"};
  PetscErrorCode  ierr;
  PetscInt        bc,np;
  Vec             b,x,err=0;
  PetscBool       use_extfile,io,error;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);CHKERRQ(ierr);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&np);CHKERRQ(ierr);

  ierr        = PetscOptionsBegin(PETSC_COMM_WORLD, "", "Options for the inhomogeneous Poisson equation", "ex35.cxx");
  user.dim    = 2;
  ierr        = PetscOptionsInt("-dim", "The dimension of the problem", "ex35.cxx", user.dim, &user.dim, NULL);CHKERRQ(ierr);
  user.problem= 1;
  ierr        = PetscOptionsInt("-problem", "The type of problem being solved (controls forcing function)", "ex35.cxx", user.problem, &user.problem, NULL);CHKERRQ(ierr);
  user.n      = 2;
  ierr        = PetscOptionsInt("-n", "The elements in each direction", "ex35.cxx", user.n, &user.n, NULL);CHKERRQ(ierr);
  user.rho    = 0.1;
  ierr        = PetscOptionsReal("-rho", "The conductivity", "ex35.cxx", user.rho, &user.rho, NULL);CHKERRQ(ierr);
  user.x      = 1.0;
  ierr        = PetscOptionsReal("-x", "The domain size in x-direction", "ex35.cxx", user.x, &user.x, NULL);CHKERRQ(ierr);
  user.y      = 1.0;
  ierr        = PetscOptionsReal("-y", "The domain size in y-direction", "ex35.cxx", user.y, &user.y, NULL);CHKERRQ(ierr);
  user.xref   = user.x/2;
  ierr        = PetscOptionsReal("-xref", "The x-coordinate of Gaussian center (for -problem 1)", "ex35.cxx", user.xref, &user.xref, NULL);CHKERRQ(ierr);
  user.yref   = user.y/2;
  ierr        = PetscOptionsReal("-yref", "The y-coordinate of Gaussian center (for -problem 1)", "ex35.cxx", user.yref, &user.yref, NULL);CHKERRQ(ierr);
  user.nu     = 0.05;
  ierr        = PetscOptionsReal("-nu", "The width of the Gaussian source (for -problem 1)", "ex35.cxx", user.nu, &user.nu, NULL);CHKERRQ(ierr);
  io          = PETSC_FALSE;
  ierr        = PetscOptionsBool("-io", "Write out the solution and mesh data", "ex35.cxx", io, &io, NULL);CHKERRQ(ierr);
  error       = PETSC_FALSE;
  ierr        = PetscOptionsBool("-error", "Compute the discrete L_2 and L_inf errors of the solution", "ex35.cxx", error, &error, NULL);CHKERRQ(ierr);
  bc          = (PetscInt)DIRICHLET;
  ierr        = PetscOptionsEList("-bc_type","Type of boundary condition","ex35.cxx",bcTypes,2,bcTypes[0],&bc,NULL);CHKERRQ(ierr);
  user.bcType = (BCType)bc;
  ierr        = PetscOptionsString("-file", "The mesh file for the problem", "ex35.cxx", "",user.filename,PETSC_MAX_PATH_LEN,&use_extfile);CHKERRQ(ierr);
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
  ierr = KSPSetComputeRHS(ksp,ComputeRHS,&user);CHKERRQ(ierr);
  ierr = KSPSetComputeOperators(ksp,ComputeMatrix,&user);CHKERRQ(ierr);
  ierr = KSPSetDM(ksp,dm);CHKERRQ(ierr);

  ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);

  /* Perform the actual solve */
  ierr = KSPSolve(ksp,NULL,NULL);CHKERRQ(ierr);
  ierr = KSPGetSolution(ksp,&x);CHKERRQ(ierr);
  ierr = KSPGetRhs(ksp,&b);CHKERRQ(ierr);

  if (error) {
    ierr = DMGetGlobalVector(dm, &err);CHKERRQ(ierr);
    ierr = ComputeDiscreteL2Error(ksp, err, &user);CHKERRQ(ierr);
    ierr = DMRestoreGlobalVector(dm, &err);CHKERRQ(ierr);
  }

  if (io) {
    /* Write out the solution along with the mesh */
    ierr = DMMoabSetGlobalFieldVector(dm, x);CHKERRQ(ierr);
#ifdef MOAB_HDF5_H
    ierr = DMMoabOutput(dm, "ex35.h5m", NULL);CHKERRQ(ierr);
#else
    /* MOAB does not support true parallel writers that aren't HDF5 based
       And so if you are using VTK as the output format in parallel,
       the data could be jumbled due to the order in which the processors
       write out their parts of the mesh and solution tags
    */
    ierr = DMMoabOutput(dm, "ex35.vtk", NULL);CHKERRQ(ierr);
#endif
  }

  /* Cleanup objects */
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  ierr = KSPDestroy(&ksp);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

#undef __FUNCT__
#define __FUNCT__ "ComputeRho"
PetscScalar ComputeRho(PetscReal coords[3], UserContext* user)
{
  switch(user->problem) {
    case 2:
      return user->rho;
    case 1:
    default:
      if ((coords[0] > user->x/3.0) && (coords[0] < 2.0*user->x/3.0) && (coords[1] > user->y/3.0) && (coords[1] < 2.0*user->y/3.0)) {
        return user->rho;
      } else {
        return 1.0;
      }
  }
}

#undef __FUNCT__
#define __FUNCT__ "ExactSolution"
PetscScalar ExactSolution(PetscReal coords[3], UserContext* user)
{
  switch(user->problem) {
    case 2:
      return sin(PI*coords[0]/user->x)*sin(PI*coords[1]/user->y);
    case 1:
    default:
      SETERRQ1(PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "Exact solution for -problem = [%D] is not available.\n", user->problem);
  }
}

#undef __FUNCT__
#define __FUNCT__ "ComputeForcingFunction"
PetscScalar ComputeForcingFunction(PetscReal coords[3], UserContext* user)
{
  switch(user->problem) {
    case 2:
      return PI*PI*ComputeRho(coords, user)*(1.0/user->x/user->x+1.0/user->y/user->y)*sin(PI*coords[0]/user->x)*sin(PI*coords[1]/user->y);
    case 1:
    default:
      const PetscScalar xx=(coords[0]-user->xref)*(coords[0]-user->xref);
      const PetscScalar yy=(coords[1]-user->yref)*(coords[1]-user->yref);
      return PetscExpScalar(-(xx+yy)/user->nu);
  }
}


#undef __FUNCT__
#define __FUNCT__ "ComputeRHS"
PetscErrorCode ComputeRHS(KSP ksp,Vec b,void *ptr)
{
  UserContext*      user = (UserContext*)ptr;
  DM                dm;
  PetscInt          dof_indices[VPERE];
  PetscBool         dbdry[VPERE];
  PetscReal         vpos[VPERE*3],quadrature[NQPTS*3],phypts[NQPTS*3],jxw[NQPTS];
  PetscScalar       ff;
  PetscInt          i,q,num_conn;
  const moab::EntityHandle *connect;
  const moab::Range *elocal;
  moab::Interface*  mbImpl;
  PetscScalar       phi[VPERE*NQPTS],localv[VPERE];
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

    /* compute the basis functions and the derivatives wrt x and y directions */
    /* compute the quadrature points transformed to the physical space */
    ierr = Compute_Quad4_Basis(NQPTS, vpos, quadrature, phypts, jxw, phi, 0, 0);CHKERRQ(ierr);

    /* Compute function over the locally owned part of the grid */
    for (q=0; q<NQPTS; ++q) {
      ff = ComputeForcingFunction(&phypts[3*q], user);

      for (i=0; i < VPERE; ++i) {
        localv[i] += jxw[q] * phi[q*VPERE+i] * ff;
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
          localv[i] = ComputeForcingFunction(&vpos[3*i], user);
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
#define __FUNCT__ "ComputeMatrix"
PetscErrorCode ComputeMatrix(KSP ksp,Mat J,Mat jac,void *ctx)
{
  UserContext       *user = (UserContext*)ctx;
  DM                dm;
  PetscInt          i,j,q,num_conn;
  PetscInt          dof_indices[VPERE];
  PetscReal         vpos[VPERE*3],quadrature[NQPTS*3],phypts[NQPTS*3],jxw[NQPTS],rho;
  PetscBool         dbdry[VPERE];
  const moab::EntityHandle *connect;
  const moab::Range *elocal;
  moab::Interface*  mbImpl;
  PetscBool         elem_on_boundary;
  PetscScalar       array[VPERE*VPERE];
  PetscScalar       phi[VPERE*NQPTS], dphidx[VPERE*NQPTS], dphidy[VPERE*NQPTS];
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
    /* compute the basis functions and the derivatives wrt x and y directions */
    ierr = Compute_Quad4_Basis(NQPTS, vpos, quadrature, phypts, jxw, phi,  dphidx, dphidy);CHKERRQ(ierr);

    ierr = PetscMemzero(array, VPERE*VPERE*sizeof(PetscScalar));

    /* Compute function over the locally owned part of the grid */
    for (q=0; q<NQPTS; ++q) {
      /* compute the inhomogeneous diffusion coefficient at the first quadrature point 
         -- for large spatial variations, embed this property evaluation inside quadrature loop */
      rho = ComputeRho(&phypts[q*3], user);

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


#undef __FUNCT__
#define __FUNCT__ "ComputeDiscreteL2Error"
PetscErrorCode ComputeDiscreteL2Error(KSP ksp,Vec err,UserContext *user)
{
  DM                dm;
  Vec               sol;
  PetscScalar       vpos[3];
  const PetscScalar *x;
  PetscScalar       *e;
  PetscReal         l2err=0.0,linferr=0.0,global_l2,global_linf;
  PetscInt          dof_index,N;
  const moab::Range *ownedvtx;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = KSPGetDM(ksp,&dm);CHKERRQ(ierr);

  /* get the solution vector */
  ierr = KSPGetSolution(ksp, &sol);CHKERRQ(ierr);

  /* Get the internal reference to the vector arrays */
  ierr = DMMoabVecGetArrayRead(dm, sol, &x);CHKERRQ(ierr);
  ierr = VecGetSize(sol, &N);CHKERRQ(ierr);
  if (err) {
    /* reset the error vector */
    ierr = VecSet(err, 0.0);CHKERRQ(ierr);
    /* get array reference */
    ierr = DMMoabVecGetArray(dm, err, &e);CHKERRQ(ierr);
  }

  ierr = DMMoabGetLocalVertices(dm, &ownedvtx, NULL);CHKERRQ(ierr);

  /* Compute function over the locally owned part of the grid */
  for(moab::Range::iterator iter = ownedvtx->begin(); iter != ownedvtx->end(); iter++) {
    const moab::EntityHandle vhandle = *iter;
    ierr = DMMoabGetDofsBlockedLocal(dm, 1, &vhandle, &dof_index);CHKERRQ(ierr);

    /* compute the mid-point of the element and use a 1-point lumped quadrature */
    ierr = DMMoabGetVertexCoordinates(dm,1,&vhandle,vpos);CHKERRQ(ierr);

    /* compute the discrete L2 error against the exact solution */
    const PetscScalar lerr = (ExactSolution(vpos, user) - x[dof_index]);
    l2err += lerr*lerr;
    if (linferr<fabs(lerr))
      linferr=fabs(lerr);

    if (err) {
      /* set the discrete L2 error against the exact solution */
      e[dof_index] = l2err;
    }
  }

  ierr = MPI_Allreduce(&l2err, &global_l2, 1, MPI_DOUBLE, MPI_SUM, PETSC_COMM_WORLD);CHKERRQ(ierr);
  ierr = MPI_Allreduce(&linferr, &global_linf, 1, MPI_DOUBLE, MPI_MAX, PETSC_COMM_WORLD);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD, "Computed Errors: L_2 = %f, L_inf = %f\n", sqrt(global_l2/N), global_linf);CHKERRQ(ierr);

  /* Restore vectors */
  ierr = DMMoabVecRestoreArrayRead(dm, sol, &x);CHKERRQ(ierr);
  if (err) {
    ierr = DMMoabVecRestoreArray(dm, err, &e);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}


PetscReal determinant_mat_2x2 ( PetscReal inmat[2*2] )
{
  return  inmat[0]*inmat[3]-inmat[1]*inmat[2];
}

PetscErrorCode invert_mat_2x2 (PetscReal *inmat, PetscReal *outmat, PetscReal *determinant)
{
  if (!inmat) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_POINTER,"Invalid input matrix specified for 2x2 inversion.");
  PetscReal det = determinant_mat_2x2(inmat);
  if (outmat) {
    outmat[0]= inmat[3]/det;
    outmat[1]=-inmat[1]/det;
    outmat[2]=-inmat[2]/det;
    outmat[3]= inmat[0]/det;
  }
  if (determinant) *determinant=det;
  PetscFunctionReturn(0);
}


/*
*  Compute_Quad4_Basis: computes the bilinear Lagrange bases at N points for a Q4 element.
*
*  Discussion:
*
*    The routine is given the coordinates of the vertices of a quadrilateral.
*    It works with these coordinates, and evaluates the basis, derivatives,
*    quadrature on reference, transformed physics quadrature points etc.
*
*    The sides of the element are presumed to lie along coordinate axes.
*    But the implementation is general enough to tackle skewed quads.
*
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
*/
#undef __FUNCT__
#define __FUNCT__ "Compute_Quad4_Basis"
PetscErrorCode Compute_Quad4_Basis ( PetscInt n, PetscReal *verts, PetscReal quad[NQPTS*3], PetscReal phypts[NQPTS*3], PetscReal jxw[NQPTS],
                                     PetscReal *phi, PetscReal *dphidx, PetscReal *dphidy)
{
  int i,j;
  PetscReal jacobian[4];
  PetscErrorCode    ierr;
  PetscFunctionBegin;

  /* 2-D 2-point tensor product Gaussian quadrature */
  quad[0]=-0.5773502691896257; quad[1]=-0.5773502691896257;
  quad[3]=-0.5773502691896257; quad[4]=0.5773502691896257;
  quad[6]=0.5773502691896257; quad[7]=-0.5773502691896257;
  quad[9]=0.5773502691896257; quad[10]=0.5773502691896257;
  /* transform quadrature bounds: [-1, 1] => [0, 1] */
  for (i=0; i<VPERE*3; ++i) {
    quad[i]=0.5*quad[i]+0.5;
  }

  ierr = PetscMemzero(phypts,NQPTS*3*sizeof(PetscReal));CHKERRQ(ierr);
  for (j=0;j<n;j++)
  {
    const int offset=j*VPERE;
    const double r = quad[0+j*3];
    const double s = quad[1+j*3];

    phi[0+offset] = ( 1.0 - r ) * ( 1.0 - s );
    phi[1+offset] =         r   * ( 1.0 - s );
    phi[2+offset] =         r   *         s;
    phi[3+offset] = ( 1.0 - r ) *         s;

    const double dNi_dxi[4]  = { -1.0 + s, 1.0 - s, s, -s };
    const double dNi_deta[4] = { -1.0 + r, -r, r, 1.0 - r };

    ierr = PetscMemzero(jacobian,4*sizeof(PetscReal));CHKERRQ(ierr);
    for (i = 0; i < VPERE; ++i) {
      const PetscScalar* vertices = verts+i*3;
      jacobian[0] += dNi_dxi[i] * vertices[0];
      jacobian[2] += dNi_dxi[i] * vertices[1];
      jacobian[1] += dNi_deta[i] * vertices[0];
      jacobian[3] += dNi_deta[i] * vertices[1];
      for (int k = 0; k < 3; ++k)
        phypts[3*j+k] += phi[i+offset] * vertices[k];
    }

    const double jacobiandet = determinant_mat_2x2 (jacobian);
    jxw[j] = jacobiandet/(NQPTS);

    /*  Divide by element jacobian. */
    for ( i = 0; i < VPERE; i++ ) {
      if (dphidx) dphidx[i+offset] = dNi_dxi[i] / sqrt(jacobiandet);
      if (dphidy) dphidy[i+offset] = dNi_deta[i] / sqrt(jacobiandet);
    }

  }
#if 0
  /* verify if the computed basis functions are consistent */
  for ( j = 0; j < n; j++ ) {
    PetscScalar phisum=0,dphixsum=0,dphiysum=0;
    for ( i = 0; i < VPERE; i++ ) {
      phisum += phi[i+j*VPERE];
      if (dphidx) dphixsum += dphidx[i+j*VPERE];
      if (dphidy) dphiysum += dphidy[i+j*VPERE];
    }
    PetscPrintf(PETSC_COMM_WORLD, "Sum of basis at quadrature point %D = %g, %g, %g\n", j, phisum, dphixsum, dphiysum);
  }
#endif
  PetscFunctionReturn(0);
}

