static const char help[] = "Time-dependent Brusselator reaction-diffusion PDE in 1d. Demonstrates IMEX methods and uses MOAB.\n";
/*
   u_t - alpha u_xx = A + u^2 v - (B+1) u
   v_t - alpha v_xx = B u - u^2 v
   0 < x < 1;
   A = 1, B = 3, alpha = 1/50

   Initial conditions:
   u(x,0) = 1 + sin(2 pi x)
   v(x,0) = 3

   Boundary conditions:
   u(0,t) = u(1,t) = 1
   v(0,t) = v(1,t) = 3
*/

// PETSc includes:
#include <petscts.h>
#include <petscdmmoab.h>

// MOAB includes:
#if defined (PETSC_HAVE_MOAB)
#  include <moab/Core.hpp>
#  include <moab/ReadUtilIface.hpp>
#  include <MBTagConventions.hpp>
#else
#error You must have MOAB for this example. Reconfigure using --download-moab
#endif

typedef moab::Range* MBRange;

typedef struct {
  PetscScalar u,v;
} Field;

struct pUserCtx {
  PetscReal A,B;        /* Reaction coefficients */
  PetscReal alpha;      /* Diffusion coefficient */
  Field leftbc;         /* Dirichlet boundary conditions at left boundary */
  Field rightbc;        /* Dirichlet boundary conditions at right boundary */
  PetscInt  npts;       /* Number of mesh points */
  PetscInt  ntsteps;    /* Number of time steps */
  PetscInt nvars;       /* Number of variables in the equation system */
  PetscInt ftype;       /* The type of function assembly routine to use in residual calculation
                           0 (default) = MOAB-Ops, 1 = Block-Ops, 2 = Ghosted-Ops  */

  moab::ParallelComm *pcomm;
  moab::Interface    *mbint;
  moab::Tag           solndofs;
  MBRange             ownedelms;
  MBRange             allvtx,ownedvtx,ghostvtx;
  moab::EntityHandle  rootset, partnset;
};
typedef pUserCtx* UserCtx;

#undef __FUNCT__
#define __FUNCT__ "Initialize_AppContext"
PetscErrorCode Initialize_AppContext(UserCtx *puser)
{
  UserCtx           user;
  moab::ErrorCode   merr;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = PetscNew(struct pUserCtx, &user);CHKERRQ(ierr);

  user->mbint = new moab::Core();

  // Create root sets for each mesh.  Then pass these
  // to the load_file functions to be populated.
  merr = user->mbint->create_meshset(moab::MESHSET_SET, user->rootset);
  MBERR("Creating root set failed", merr);
  merr = user->mbint->create_meshset(moab::MESHSET_SET, user->partnset);
  MBERR("Creating partition set failed", merr);

  // Create the parallel communicator object with the partition handle associated with MOAB
  user->pcomm = moab::ParallelComm::get_pcomm(user->mbint, user->partnset, &PETSC_COMM_WORLD);

  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,PETSC_NULL,"Advection-reaction options","");
  {
    user->nvars  = 2;
    user->A      = 1;
    user->B      = 3;
    user->alpha  = 0.02;
    user->leftbc.u  = 1;
    user->rightbc.u = 1;
    user->leftbc.v  = 3;
    user->rightbc.v = 3;
    user->npts   = 11;
    user->ntsteps = 10000;
    user->ftype = 0;
    ierr = PetscOptionsReal("-A","Reaction rate","",user->A,&user->A,PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-B","Reaction rate","",user->B,&user->B,PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-alpha","Diffusion coefficient","",user->alpha,&user->alpha,PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-uleft","Dirichlet boundary condition","",user->leftbc.u,&user->leftbc.u,PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-uright","Dirichlet boundary condition","",user->rightbc.u,&user->rightbc.u,PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-vleft","Dirichlet boundary condition","",user->leftbc.v,&user->leftbc.v,PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-vright","Dirichlet boundary condition","",user->rightbc.v,&user->rightbc.v,PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-npts","Number of mesh points","",user->npts,&user->npts,PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-ndt","Number of time steps","",user->ntsteps,&user->ntsteps,PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-ftype","Type of function evaluation model for FEM assembly","",user->ftype,&user->ftype,PETSC_NULL);CHKERRQ(ierr);
  }
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  user->ownedvtx = new moab::Range();
  user->ownedelms = new moab::Range();
  user->ghostvtx = new moab::Range();
  user->allvtx = new moab::Range();

  *puser = user;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "Destroy_AppContext"
PetscErrorCode Destroy_AppContext(UserCtx *user)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  delete (*user)->pcomm;
  delete (*user)->mbint;
  delete (*user)->ownedvtx;
  delete (*user)->ownedelms;
  delete (*user)->allvtx;
  delete (*user)->ghostvtx;

  ierr = PetscFree(*user);CHKERRQ(ierr);
  user = PETSC_NULL;
  PetscFunctionReturn(0);
}

static PetscErrorCode FormRHSFunction(TS,PetscReal,Vec,Vec,void*);
static PetscErrorCode FormIFunctionGhosted(TS,PetscReal,Vec,Vec,Vec,void*);
static PetscErrorCode FormIFunctionGlobalBlocked(TS,PetscReal,Vec,Vec,Vec,void*);
static PetscErrorCode FormIFunctionMOAB(TS,PetscReal,Vec,Vec,Vec,void*);
static PetscErrorCode FormIJacobian(TS,PetscReal,Vec,Vec,PetscReal,Mat*,Mat*,MatStructure*,void*);
static PetscErrorCode FormInitialSolution(TS,Vec,void*);
static PetscErrorCode CreateMesh(UserCtx);

/****************
 *              *
 *     MAIN     *
 *              *
 ****************/
#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  TS                ts;         /* nonlinear solver */
  Vec               X;          /* solution, residual vectors */
  Mat               J;          /* Jacobian matrix */
  PetscInt          steps;
  PetscErrorCode    ierr;
  PetscReal         hx,dt,ftime;
  UserCtx           user;       /* user-defined work context */
  TSConvergedReason reason;
  DM                dm;

  ierr = PetscInitialize(&argc,&argv,(char *)0,help);CHKERRQ(ierr);

  // Initialize the user context struct
  ierr = Initialize_AppContext(&user);CHKERRQ(ierr);

  // Fill in the user defined work context:
  ierr = CreateMesh(user);CHKERRQ(ierr);

  // Create the solution vector:
  ierr = DMCreate(PETSC_COMM_WORLD, &dm);CHKERRQ(ierr);
  ierr = DMSetType(dm, DMMOAB);CHKERRQ(ierr);
  ierr = DMMoabSetParallelComm(dm, user->pcomm);CHKERRQ(ierr);
  ierr = DMMoabSetBlockSize(dm, user->nvars);CHKERRQ(ierr);
  ierr = DMSetFromOptions(dm);CHKERRQ(ierr);

  /* SetUp the data structures for DMMOAB */
  ierr = DMSetUp(dm);CHKERRQ(ierr);

  // print some information if -info is enabled
  PetscInfo1(dm, "Number of owned elements = %D\n", user->ownedelms->size());
  PetscInfo1(dm, "Number of owned vertices = %D\n", user->ownedvtx->size());
  PetscInfo1(dm, "Number of shared vertices = %D\n", user->ghostvtx->size());
  PetscInfo1(dm, "Number of vertices = %D\n", user->allvtx->size());

  //  Create timestepping solver context
  ierr = TSCreate(PETSC_COMM_WORLD,&ts);CHKERRQ(ierr);
  ierr = TSSetDM(ts, dm);CHKERRQ(ierr);
  ierr = TSSetType(ts,TSARKIMEX);CHKERRQ(ierr);
  ierr = TSSetRHSFunction(ts,PETSC_NULL,FormRHSFunction,user);CHKERRQ(ierr);
  if (user->ftype == 1) {
    ierr = TSSetIFunction(ts,PETSC_NULL,FormIFunctionGlobalBlocked,user);CHKERRQ(ierr);
  } else if(user->ftype == 2) {
    ierr = TSSetIFunction(ts,PETSC_NULL,FormIFunctionGhosted,user);CHKERRQ(ierr);  
  } else {
    ierr = TSSetIFunction(ts,PETSC_NULL,FormIFunctionMOAB,user);CHKERRQ(ierr);
  }
  ierr = DMCreateMatrix(dm, MATBAIJ, &J);CHKERRQ(ierr);
  ierr = TSSetIJacobian(ts,J,J,FormIJacobian,user);CHKERRQ(ierr);

  ftime = 10.0;
  ierr = TSSetDuration(ts,user->ntsteps,ftime);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create the solution vector and set the initial conditions
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  /* Use the call to DMMoabCreateVector for creating a named global MOAB Vec object.
     Alternately, use the following call to DM for creating an unnamed (anonymous) global 
     MOAB Vec object.

         ierr = DMCreateGlobalVector(dm, &X);CHKERRQ(ierr);
  */
  ierr = DMMoabCreateVector(dm, user->solndofs, 1, user->ownedvtx, PETSC_TRUE, PETSC_FALSE,
                              &X);CHKERRQ(ierr);

  ierr = FormInitialSolution(ts,X,user);CHKERRQ(ierr);
  ierr = TSSetSolution(ts,X);CHKERRQ(ierr);
  hx = 1.0/(user->npts-1);
  dt = 0.4 * PetscSqr(hx) / user->alpha; /* Diffusive stability limit */
  ierr = TSSetInitialTimeStep(ts,0.0,dt);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set runtime options
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = TSSetFromOptions(ts);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Solve nonlinear system
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = TSSolve(ts,X);CHKERRQ(ierr);
  ierr = TSGetSolveTime(ts,&ftime);CHKERRQ(ierr);
  ierr = TSGetTimeStepNumber(ts,&steps);CHKERRQ(ierr);
  ierr = TSGetConvergedReason(ts,&reason);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"%s at time %G after %D steps\n",TSConvergedReasons[reason],ftime,steps);CHKERRQ(ierr);

  /* Print the numerical solution to screen and then dump to file */
//  ierr = VecView(X,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  // Write out the solution along with the mesh
//  merr = user->mbint->write_file("ex2.h5m");MBERRNM(merr);

  // Free work space.
  // Free all PETSc related resources:
  ierr = MatDestroy(&J);CHKERRQ(ierr);
  ierr = VecDestroy(&X);CHKERRQ(ierr);
  ierr = TSDestroy(&ts);CHKERRQ(ierr);
  ierr = DMDestroy(&dm);CHKERRQ(ierr);

  // Free all MOAB related resources:
  ierr = Destroy_AppContext(&user);CHKERRQ(ierr);

  ierr = PetscFinalize();
  return 0;
}

#undef __FUNCT__
#define __FUNCT__ "FormRHSFunction"
static PetscErrorCode FormRHSFunction(TS ts,PetscReal t,Vec X,Vec F,void *ptr)
{
  UserCtx           user = (UserCtx)ptr;
  PetscReal         hx;
  const Field       *x;
  Field             *f;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  hx = 1.0/(PetscReal)(user->npts-1);

  /* Get pointers to vector data */
  ierr = VecGetArrayRead(X,(const PetscScalar**)&x);CHKERRQ(ierr);
  ierr = VecGetArray(F,(PetscScalar**)&f);CHKERRQ(ierr);

  /* Compute function over the locally owned part of the grid */
  moab::Range::iterator iter = user->ownedvtx->begin();
  const moab::EntityHandle first_vertex = *iter;
  for(; iter != user->ownedvtx->end(); iter++) {
    moab::EntityHandle i = *iter - first_vertex;
    PetscScalar u = x[i].u, v = x[i].v;
    f[i].u = hx*(user->A + u*u*v - (user->B+1)*u);
    f[i].v = hx*(user->B*u - u*u*v);
  }

  /* Restore vectors */
  ierr = VecRestoreArrayRead(X,(const PetscScalar**)&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(F,(PetscScalar**)&f);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


/*
  IJacobian - Compute IJacobian = dF/dU + a dF/dUdot
*/
#undef __FUNCT__
#define __FUNCT__ "FormIJacobian"
PetscErrorCode FormIJacobian(TS ts,PetscReal t,Vec X,Vec Xdot,PetscReal a,Mat *J,Mat *Jpre,MatStructure *str,void *ptr)
{
  UserCtx             user = (UserCtx)ptr;
  PetscErrorCode      ierr;
  PetscInt            verts_per_entity=2,count;
  PetscReal           hx;
  PetscInt           *vertex_ids,rank;
  moab::Tag           id_tag;
  const moab::EntityHandle *connect;
  DM                  dm;
  moab::ErrorCode     merr;

  PetscFunctionBegin;
  ierr = TSGetDM(ts, &dm);CHKERRQ(ierr);
  MPI_Comm_rank( PETSC_COMM_WORLD,&rank );

  // Get the global IDs on all vertexes:
  ierr = DMMoabGetLocalToGlobalTag(dm, &id_tag);CHKERRQ(ierr);
  merr = user->mbint->tag_iterate(id_tag,user->allvtx->begin(),user->allvtx->end(),
  				  count,reinterpret_cast<void*&>(vertex_ids));MBERRNM(merr);

  /* zero out the discrete operator */
  ierr = MatZeroEntries(*Jpre);CHKERRQ(ierr);

  /* compute local element sizes */
  hx = 1.0/(PetscReal)(user->npts-1);

  /* Compute function over the locally owned part of the grid 
     Assemble the operator by looping over edges and computing
     contribution for each vertex dof                         */
  for(moab::Range::iterator iter = user->ownedelms->begin(); iter != user->ownedelms->end(); iter++) {
    
    merr = user->mbint->get_connectivity((*iter), connect, verts_per_entity);MBERRNM(merr); // get the connectivity, in canonical order
    const int idl  = vertex_ids[connect[0]-1];
    const int idr  = vertex_ids[connect[1]-1];

    const PetscInt    lcols[] = {idl,idr}, rcols[] = {idr, idl};
    const PetscScalar dxxL = user->alpha/hx,dxxR = -user->alpha/hx;

    if (idl == 0) {
      // Boundary conditions...
      const PetscScalar lvals[2][2] = {{hx,0},{0,hx}};
      ierr = MatSetValuesBlocked(*Jpre,1,&idl,1,&idl,&lvals[0][0],ADD_VALUES);CHKERRQ(ierr);

      const PetscScalar vals_u[2][2][2] = {{{a *hx/2+dxxL,0},{dxxR,0}},
                                         {{0,a*hx/2+dxxL},{0,dxxR}}};

      ierr = MatSetValuesBlocked(*Jpre,1,&idr,2,rcols,&vals_u[0][0][0],ADD_VALUES);CHKERRQ(ierr);
    }
    else if(idr == user->npts-1) {
      // Boundary conditions...
      const PetscScalar rvals[2][2] = {{hx,0},{0,hx}};
      ierr = MatSetValuesBlocked(*Jpre,1,&idr,1,&idr,&rvals[0][0],ADD_VALUES);CHKERRQ(ierr);

      const PetscScalar vals_u[2][2][2] = {{{a *hx/2+dxxL,0},{dxxR,0}},
                                         {{0,a*hx/2+dxxL},{0,dxxR}}};

      ierr = MatSetValuesBlocked(*Jpre,1,&idl,2,lcols,&vals_u[0][0][0],ADD_VALUES);CHKERRQ(ierr);
    }
    else {
      const PetscScalar vals_u[2][2][2] = {{{a *hx/2+dxxL,0},{dxxR,0}},
                                         {{0,a*hx/2+dxxL},{0,dxxR}}};
      const PetscScalar vals_v[2][2][2] = {{{a *hx/2+dxxL,0},{dxxR,0}},
                                         {{0,a*hx/2+dxxL},{0,dxxR}}};
      
      ierr = MatSetValuesBlocked(*Jpre,1,&idr,2,rcols,&vals_u[0][0][0],ADD_VALUES);CHKERRQ(ierr);
      ierr = MatSetValuesBlocked(*Jpre,1,&idl,2,lcols,&vals_v[0][0][0],ADD_VALUES);CHKERRQ(ierr);
    }
  }

  ierr = MatAssemblyBegin(*Jpre,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*Jpre,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  if (*J != *Jpre) {
    ierr = MatAssemblyBegin(*J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(*J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "Create_1D_Mesh"
PetscErrorCode Create_1D_Mesh(moab::ParallelComm* pcomm,int npts,int nghost)
{
  moab::ErrorCode merr;
  PetscInt rank,nprocs;
  moab::ReadUtilIface* readMeshIface;
  moab::Range ownedvtx,ownedelms;
  moab::Tag id_tag;

  // Determine which elements (cells) this process owns:
  const PetscInt nele = npts-1;
  PetscInt my_nele; // Number of elements owned by this process
  PetscInt vstart;    // The starting element for this process

  PetscFunctionBegin;
  MPI_Comm_size( PETSC_COMM_WORLD,&nprocs );
  MPI_Comm_rank( PETSC_COMM_WORLD,&rank );

  // Begin with some error checking:
  if(npts < 2) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_OUTOFRANGE,"Number of points must be >= 2");
  if(nprocs >= npts) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_OUTOFRANGE,"Number of processors must be less than number of points");

  // No errors,proceed with building the mesh:
  moab::Interface *mbint = pcomm->get_moab();
  mbint->query_interface(readMeshIface);

  const PetscInt fraction = nele / nprocs;
  const PetscInt remainder = nele % nprocs;

  if(rank < remainder) {
    // This process gets "fraction+1" vertexes
    my_nele = fraction + 1;
    vstart = rank * (fraction+1);
  } else {
    // This process gets "fraction" vertexes
    my_nele = fraction;
    vstart = remainder*(fraction+1) + fraction*(rank-remainder);
  }

  // Create the local portion of the mesh:
  const PetscInt my_npts = my_nele + 1;

  // Create vertexes and set the coodinate of each vertex:
  moab::EntityHandle vertex_first;
  std::vector<double*> vertex_coords;
  const int sequence_size = (my_nele + 2) + 1;
  merr = readMeshIface->get_node_coords(3,my_npts,1,vertex_first,vertex_coords,sequence_size);MBERRNM(merr);

  // Get the global ID tag. The global ID tag is applied to each
  // vertex. It acts as an global identifier which MOAB uses to
  // assemble the individual pieces of the mesh:
  merr = mbint->tag_get_handle(GLOBAL_ID_TAG_NAME,id_tag);MBERRNM(merr);

  std::vector<int> global_ids(my_npts);
  const double hx = 1.0/nele;  // domain = [0 1]
  for (int i = 0; i < my_npts; i++) {
    vertex_coords[0][i] = ((i+vstart))*hx;
    vertex_coords[1][i] = 0.0;
    vertex_coords[2][i] = 0.0;
    global_ids[i] = i+vstart;
  }

  merr = mbint->get_entities_by_type(0,moab::MBVERTEX,ownedvtx,true);MBERRNM(merr);

  // Create elements between mesh points. This is done so that VisIt
  // will interpret the output as a mesh that can be plotted...
  moab::EntityHandle edge_first;
  moab::EntityHandle *connectivity = 0;

  merr = readMeshIface->get_element_connect (my_nele,2,moab::MBEDGE,1,edge_first,connectivity);MBERRNM(merr);
  for (int i = 0; i < my_nele; i+=1) {
    connectivity[2*i]   = vertex_first + i;
    connectivity[2*i+1] = vertex_first + (i+1);
  }
  merr = readMeshIface->update_adjacencies(edge_first,my_nele,2,connectivity);MBERRNM(merr);
  
  // set the global id for all the owned vertices
  merr = mbint->tag_set_data(id_tag,ownedvtx,global_ids.data());MBERRNM(merr);
  
  // resolve the shared entities by exchanging information to adjacent processors
  merr = mbint->get_entities_by_type(0,moab::MBEDGE,ownedelms,true);MBERRNM(merr);
  merr = pcomm->resolve_shared_ents(0,ownedelms,1,0);MBERRNM(merr);

  // Reassign global IDs on all entities.
  merr = pcomm->exchange_ghost_cells( 1,0,nghost,0,true);MBERRNM(merr);

  // Everything is set up, now just do a tag exchange to update tags
  // on all of the ghost vertexes:
  merr = pcomm->exchange_tags(id_tag,ownedvtx);MBERRNM(merr);
  merr = pcomm->exchange_tags(id_tag,ownedelms);MBERRNM(merr);

  // set the dimension of the mesh
  merr = mbint->set_dimension(1);MBERRNM(merr);
  PetscFunctionReturn(0);
}



#undef __FUNCT__
#define __FUNCT__ "CreateMesh"
PetscErrorCode CreateMesh(UserCtx user)
{
  PetscErrorCode ierr;
  moab::ErrorCode merr;
  PetscScalar     deflt[2]={0.0,0.0};
  void            *data;
  PetscInt        count;

  PetscFunctionBegin;
  ierr = Create_1D_Mesh(user->pcomm,user->npts,1); CHKERRQ(ierr);
  
  /* Set the edge/vertex range once so we don't have to do it
     again. To do this we get all of the edges/vertexes then filter so
     we have only the owned entities in the ranges */
   merr = user->mbint->get_entities_by_type(0,moab::MBVERTEX,*user->allvtx,true);MBERRNM(merr);

  merr = user->mbint->get_entities_by_type(0,moab::MBEDGE,*user->ownedelms,true);MBERRNM(merr);
  merr = user->pcomm->filter_pstatus(*user->ownedelms,PSTATUS_NOT_OWNED,PSTATUS_NOT);MBERRNM(merr);

  *user->ownedvtx = *user->allvtx;
  /* filter based on Pstatus flag to get only owned vertices */
  merr = user->pcomm->filter_pstatus(*user->ownedvtx,PSTATUS_NOT_OWNED,PSTATUS_NOT);MBERRNM(merr);

  /* subtract owned vertices from all the local vertices (which includes ghost vertices) */
  *user->ghostvtx = moab::subtract(*user->allvtx, *user->ownedvtx);
 
  /* Do some error checking...make sure that tag_data is in a single sequence */
  merr = user->mbint->tag_get_handle("UNKNOWNS",2,moab::MB_TYPE_DOUBLE,user->solndofs,
                                  moab::MB_TAG_DENSE | moab::MB_TAG_CREAT,deflt);MBERRNM(merr);
  merr = user->mbint->tag_iterate(user->solndofs,user->allvtx->begin(),user->allvtx->end(),count,data);MBERRNM(merr);

  if((unsigned int) count != user->allvtx->size()) {
    SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_LIB,"Tag data not laid out contiguously %i %i",count,user->allvtx->size());
  }
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "FormInitialSolution"
PetscErrorCode FormInitialSolution(TS ts,Vec X,void *ctx)
{
  UserCtx           user = (UserCtx)ctx;
  PetscReal         hx;
  DM                dm;
  Field             *x;
  PetscInt          *vertex_ids,count;
  moab::Tag         id_tag;
  moab::ErrorCode   merr;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  hx = 1.0/(PetscReal)(user->npts-1);

  // Get the global IDs on all vertexes:
  ierr = VecGetDM(X, &dm);CHKERRQ(ierr);
  ierr = DMMoabGetLocalToGlobalTag(dm, &id_tag);CHKERRQ(ierr);
  merr = user->mbint->tag_iterate(id_tag,user->allvtx->begin(),user->allvtx->end(),
  				  count,reinterpret_cast<void*&>(vertex_ids));MBERRNM(merr);

  ierr = VecSet(X, 0.0);CHKERRQ(ierr);

  /* Get pointers to vector data */
  ierr = VecGetArray(X,(PetscScalar**)&x);CHKERRQ(ierr);

  /* Compute function over the locally owned part of the grid */
  const int first_vertex = vertex_ids[*user->ownedvtx->begin()-1];
  moab::Range::iterator iter;
  for(iter = user->ownedvtx->begin(); iter != user->ownedvtx->end(); iter++) {
    const int i = vertex_ids[*iter-1]-first_vertex ;
    PetscReal xi = (i+first_vertex)*hx;
    x[i].u = user->leftbc.u*(1.-xi) + user->rightbc.u*xi + sin(2.*PETSC_PI*xi);
    x[i].v = user->leftbc.v*(1.-xi) + user->rightbc.v*xi;
  }

  /* Restore vectors */
  ierr = VecRestoreArray(X,(PetscScalar**)&x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "FormIFunctionMOAB"
static PetscErrorCode FormIFunctionMOAB(TS ts,PetscReal t,Vec X,Vec Xdot,Vec F,void *ctx)
{
  UserCtx         user = (UserCtx)ctx;
  DM              dm;
  PetscInt        i;
  Field           *x,*xdot,*f;
  PetscReal       hx;
  PetscErrorCode  ierr;
  moab::ErrorCode merr;
  PetscInt        *vertex_ids;
  moab::Tag       id_tag,x_tag,xdot_tag,f_tag;

  PetscFunctionBegin;
  hx = 1.0/(PetscReal)(user->npts-1);
  ierr = TSGetDM(ts, &dm);CHKERRQ(ierr);

  // Get connectivity information:
  int verts_per_entity;
  int count,num_edges;
  moab::EntityHandle *connect;
  moab::Range::iterator iter = user->ownedelms->begin();
  merr = user->mbint->connect_iterate(iter,user->ownedelms->end(),connect,verts_per_entity,num_edges);MBERRNM(merr);

  // get tag data for solution 
  ierr = DMMoabGetVecTag(X,&x_tag);CHKERRQ(ierr);
  merr = user->mbint->tag_iterate(x_tag,user->allvtx->begin(),user->allvtx->end(),
				  count,reinterpret_cast<void*&>(x));MBERRNM(merr);

  // get the tag data for solution derivative
  ierr = DMMoabGetVecTag(Xdot,&xdot_tag);CHKERRQ(ierr);
  merr = user->mbint->tag_iterate(xdot_tag,user->allvtx->begin(),user->allvtx->end(),
				  count,reinterpret_cast<void*&>(xdot));MBERRNM(merr);

  // get the residual tag
  ierr = DMMoabGetVecTag(F,&f_tag);CHKERRQ(ierr);
  merr = user->mbint->tag_iterate(f_tag,user->allvtx->begin(),user->allvtx->end(),
				  count,reinterpret_cast<void*&>(f));MBERRNM(merr);

  // Get the global IDs on all vertexes:
  ierr = DMMoabGetLocalToGlobalTag(dm,&id_tag);CHKERRQ(ierr);
  merr = user->mbint->tag_iterate(id_tag,user->allvtx->begin(),user->allvtx->end(),
  				  count,reinterpret_cast<void*&>(vertex_ids));MBERRNM(merr);

  // Exchange tags that are needed for assembly:
  merr = user->pcomm->exchange_tags(x_tag,*user->allvtx);MBERRNM(merr);
  merr = user->pcomm->exchange_tags(xdot_tag,*user->allvtx);MBERRNM(merr);

  /* reset the residual vector */
  ierr = PetscMemzero(f,sizeof(Field)*user->allvtx->size());CHKERRQ(ierr);

  const moab::EntityHandle first_vertex = *user->allvtx->begin();

  for (i = 0; i < num_edges; i++) {
    const moab::EntityHandle idx_left  = connect[2*i]-first_vertex;
    const moab::EntityHandle idx_right = connect[2*i+1]-first_vertex;

    const int id_left  = vertex_ids[idx_left ];
    const int id_right = vertex_ids[idx_right];

    if (id_left == 0) {
      // Apply left BC
      f[idx_left].u += hx * (x[idx_left].u - user->leftbc.u);
      f[idx_left].v += hx * (x[idx_left].v - user->leftbc.v);
    } else {
      f[idx_left].u += hx * xdot[idx_left].u + user->alpha*(x[idx_left].u - x[idx_right].u)/hx;
      f[idx_left].v += hx * xdot[idx_left].v + user->alpha*(x[idx_left].v - x[idx_right].v)/hx;
    }

    if (id_right == user->npts-1) {
      // Apply right BC
      f[idx_right].u += hx * (x[idx_right].u - user->rightbc.u);
      f[idx_right].v += hx * (x[idx_right].v - user->rightbc.v);
    } else {
      f[idx_right].u += user->alpha*(x[idx_right].u-x[idx_left].u)/hx;
      f[idx_right].v += user->alpha*(x[idx_right].v-x[idx_left].v)/hx;
    }
  }

  // Add tags on shared vertexes:
  merr = user->pcomm->reduce_tags(f_tag,MPI_SUM,*user->ghostvtx);MBERRNM(merr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "FormIFunctionGlobalBlocked"
static PetscErrorCode FormIFunctionGlobalBlocked(TS ts,PetscReal t,Vec X,Vec Xdot,Vec F,void *ptr)
{
  UserCtx             user = (UserCtx)ptr;
  PetscInt            i,count,rank,rstart,rend;
  Field               *x,*xdot;
  Vec                 xltmp, xdtmp, xlocal,xdotlocal;
  PetscReal           hx;
  PetscErrorCode      ierr;
  DM                  dm;
  moab::ErrorCode     merr;
  PetscInt            *vertex_ids;
  moab::Tag           id_tag;
  PetscInt            verts_per_entity=2;
  const PetscInt      left=0,right=1;
  const moab::EntityHandle  *connect;

  PetscFunctionBegin;
  hx = 1.0/(PetscReal)(user->npts-1);

  MPI_Comm_rank( PETSC_COMM_WORLD,&rank );
  ierr = TSGetDM(ts, &dm);CHKERRQ(ierr);

  ierr = DMGetLocalVector(dm,&xlocal);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(dm,X,INSERT_VALUES,xlocal);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(dm,X,INSERT_VALUES,xlocal);CHKERRQ(ierr);
  ierr = VecGhostGetLocalForm(xlocal,&xltmp);CHKERRQ(ierr);
  ierr = VecGhostUpdateBegin(xlocal,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecGhostUpdateEnd(xlocal,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);

  ierr = DMGetLocalVector(dm,&xdotlocal);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(dm,Xdot,INSERT_VALUES,xdotlocal);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(dm,Xdot,INSERT_VALUES,xdotlocal);CHKERRQ(ierr);
  ierr = VecGhostGetLocalForm(xdotlocal,&xdtmp);CHKERRQ(ierr);
  ierr = VecGhostUpdateBegin(xdotlocal,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecGhostUpdateEnd(xdotlocal,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);

  // reset the residual vector before assembly
  ierr = VecSet(F, 0.0);CHKERRQ(ierr);

  // Get the global IDs on all vertexes:
  ierr = DMMoabGetLocalToGlobalTag(dm, &id_tag);CHKERRQ(ierr);
  merr = user->mbint->tag_iterate(id_tag,user->allvtx->begin(),user->allvtx->end(),
  				  count,reinterpret_cast<void*&>(vertex_ids));MBERRNM(merr);

  int xsiz,xlocsiz;
  VecGetSize(xlocal, &xsiz);
  VecGetLocalSize(xlocal, &xlocsiz);

  const int first_vertex = vertex_ids[*user->ownedvtx->begin()-1];

  ierr = VecGetArrayRead(xltmp, (const PetscScalar**)&x);CHKERRQ(ierr);
  ierr = VecGetArrayRead(xdtmp, (const PetscScalar**)&xdot);CHKERRQ(ierr);
  
  VecGetOwnershipRange(X,&rstart,&rend);
  int rsize = (rend-rstart)/2;

  /* Compute function over the locally owned part of the grid 
     Assemble the operator by looping over edges and computing
     contribution for each vertex dof                         */
  i=0;
  /* loop over local elements */
  for(moab::Range::iterator iter = user->ownedelms->begin(); iter != user->ownedelms->end(); iter++, i++) {
    
    merr = user->mbint->get_connectivity((*iter), connect, verts_per_entity);MBERRNM(merr); // get the connectivity, in canonical order
    int idl  = vertex_ids[connect[left]-1]-first_vertex;
    int idr  = vertex_ids[connect[right]-1]-first_vertex;

    if (idl < 0 || idr < 0) {
      // Did we hit a ghost that is before the start ?!
      if (idl<0) idl = ((rend-rstart)/2-(idl+1));
      if (idr<0) idr = ((idr+first_vertex)*2-rend)/2;
    }
    if (idl > rsize || idr > rsize) {
      PetscPrintf(PETSC_COMM_SELF, "\n [%D] Found large left, or right indices not between [%D-%D]: %D - %D", rank, rstart/2, rend/2, idl, idr);      
    }

    const int cols[2] = {vertex_ids[connect[left]-1], vertex_ids[connect[right]-1]};

    if (idl+first_vertex == 0) {
      const double vals[4] = { hx * (x[idl].u - user->leftbc.u),
                               hx * (x[idl].v - user->leftbc.v),
                               hx/2 * xdot[idr].u + user->alpha * ( x[idr].u - x[idl].u ) / hx,
                               hx/2 * xdot[idr].v + user->alpha * ( x[idr].v - x[idl].v ) / hx};

      ierr = VecSetValuesBlocked(F, 2, cols, vals, ADD_VALUES);CHKERRQ(ierr);
      
    } else if (idr+first_vertex == user->npts-1) {
      const double vals[4] = { hx/2 * xdot[idl].u + user->alpha * ( x[idl].u - x[idr].u ) / hx,
                               hx/2 * xdot[idl].v + user->alpha * ( x[idl].v - x[idr].v ) / hx,
                               hx * (x[idr].u - user->rightbc.u),
                               hx * (x[idr].v - user->rightbc.v) };

      ierr = VecSetValuesBlocked(F, 2, cols, vals, ADD_VALUES);CHKERRQ(ierr);
            
    } else {
      const double vals[4] = { hx/2 * xdot[idl].u + user->alpha * (x[idl].u - x[idr].u)/ hx,
                               hx/2 * xdot[idl].v + user->alpha * (x[idl].v - x[idr].v)/ hx,
                               hx/2 * xdot[idr].u + user->alpha * (x[idr].u - x[idl].u)/ hx,
                               hx/2 * xdot[idr].v + user->alpha * (x[idr].v - x[idl].v)/ hx };

      ierr = VecSetValuesBlocked(F, 2, cols, vals, ADD_VALUES);CHKERRQ(ierr);
    }
  }

  // Restore all the local vector data array references
  ierr = VecRestoreArrayRead(xltmp, (const PetscScalar**)&x);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(xdtmp, (const PetscScalar**)&xdot);CHKERRQ(ierr);

  ierr = VecGhostRestoreLocalForm(xlocal, &xltmp);CHKERRQ(ierr);
  ierr = VecGhostRestoreLocalForm(xdotlocal, &xdtmp);CHKERRQ(ierr);

  ierr = DMRestoreLocalVector(dm, &xlocal);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dm, &xdotlocal);CHKERRQ(ierr);

  ierr = VecAssemblyBegin(F);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(F);CHKERRQ(ierr);  
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "FormIFunctionGhosted"
static PetscErrorCode FormIFunctionGhosted(TS ts,PetscReal t,Vec X,Vec Xdot,Vec F,void *ptr)
{
  UserCtx             user = (UserCtx)ptr;
  PetscInt            i,count,rank,rstart,rend;
  Field               *x,*xdot;
  Field               *f;
  Vec                 xltmp, xdtmp, xlocal,xdotlocal, flocal,fltmp;
  PetscReal           hx;
  PetscErrorCode      ierr;
  DM                  dm;
  moab::ErrorCode     merr;
  PetscInt            *vertex_ids;
  moab::Tag           id_tag;
  PetscInt            verts_per_entity=2;
  const PetscInt      left=0,right=1;
  const moab::EntityHandle  *connect;

  PetscFunctionBegin;
  hx = 1.0/(PetscReal)(user->npts-1);

  MPI_Comm_rank( PETSC_COMM_WORLD,&rank );
  ierr = TSGetDM(ts, &dm);CHKERRQ(ierr);

  ierr = DMGetLocalVector(dm,&xlocal);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(dm,X,INSERT_VALUES,xlocal);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(dm,X,INSERT_VALUES,xlocal);CHKERRQ(ierr);
  ierr = VecGhostGetLocalForm(xlocal,&xltmp);CHKERRQ(ierr);
  ierr = VecGhostUpdateBegin(xlocal,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecGhostUpdateEnd(xlocal,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);

  ierr = DMGetLocalVector(dm,&xdotlocal);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(dm,Xdot,INSERT_VALUES,xdotlocal);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(dm,Xdot,INSERT_VALUES,xdotlocal);CHKERRQ(ierr);
  ierr = VecGhostGetLocalForm(xdotlocal,&xdtmp);CHKERRQ(ierr);
  ierr = VecGhostUpdateBegin(xdotlocal,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecGhostUpdateEnd(xdotlocal,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);

  ierr = DMGetLocalVector(dm,&flocal);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(dm,F,INSERT_VALUES,flocal);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(dm,F,INSERT_VALUES,flocal);CHKERRQ(ierr);
  ierr = VecGhostGetLocalForm(flocal,&fltmp);CHKERRQ(ierr);
  // reset the residual vector before assembly
  ierr = VecSet(fltmp, 0.0);CHKERRQ(ierr);

  // Get the global IDs on all vertexes:
  ierr = DMMoabGetLocalToGlobalTag(dm, &id_tag);CHKERRQ(ierr);
  merr = user->mbint->tag_iterate(id_tag,user->allvtx->begin(),user->allvtx->end(),
  				  count,reinterpret_cast<void*&>(vertex_ids));MBERRNM(merr);

  int xsiz,xlocsiz;
  VecGetSize(xlocal, &xsiz);
  VecGetLocalSize(xlocal, &xlocsiz);

  const int first_vertex = vertex_ids[*user->ownedvtx->begin()-1];

  ierr = VecGetArrayRead(xltmp, (const PetscScalar**)&x);CHKERRQ(ierr);
  ierr = VecGetArrayRead(xdtmp, (const PetscScalar**)&xdot);CHKERRQ(ierr);
  ierr = VecGetArray(fltmp, (PetscScalar**)&f);CHKERRQ(ierr);

  VecGetOwnershipRange(X,&rstart,&rend);
  int rsize = (rend-rstart)/2;

  /* Compute function over the locally owned part of the grid 
     Assemble the operator by looping over edges and computing
     contribution for each vertex dof                         */
  i=0;
  /* loop over local elements */
  for(moab::Range::iterator iter = user->ownedelms->begin(); iter != user->ownedelms->end(); iter++, i++) {
    
    merr = user->mbint->get_connectivity((*iter), connect, verts_per_entity);MBERRNM(merr); // get the connectivity, in canonical order
    int idl  = vertex_ids[connect[left]-1]-first_vertex;
    int idr  = vertex_ids[connect[right]-1]-first_vertex;

    if (idl < 0 || idr < 0) {
      // Did we hit a ghost that is before the start ?!
      if (idl<0) idl = ((rend-rstart)/2-(idl+1));
      if (idr<0) idr = ((idr+first_vertex)*2-rend)/2;
    }
    if (idl > rsize || idr > rsize) {
      PetscPrintf(PETSC_COMM_SELF, "\n [%D] Found large left, or right indices not between [%D-%D]: %D - %D", rank, rstart/2, rend/2, idl, idr);      
    }

    if (idl+first_vertex == 0) {      
      f[idl].u += hx * (x[idl].u - user->leftbc.u);
      f[idl].v += hx * (x[idl].v - user->leftbc.v);
      f[idr].u += hx/2 * xdot[idr].u + user->alpha * ( x[idr].u - x[idl].u ) / hx;
      f[idr].v += hx/2 * xdot[idr].v + user->alpha * ( x[idr].v - x[idl].v ) / hx;
    } else if (idr+first_vertex == user->npts-1) {      
      f[idr].u += hx * (x[idr].u - user->rightbc.u);
      f[idr].v += hx * (x[idr].v - user->rightbc.v);
      f[idl].u += hx/2 * xdot[idl].u + user->alpha * ( x[idl].u - x[idr].u ) / hx;
      f[idl].v += hx/2 * xdot[idl].v + user->alpha * ( x[idl].v - x[idr].v ) / hx;
    } else {      
      f[idl].u += hx/2 * xdot[idl].u + user->alpha * (x[idl].u - x[idr].u)/ hx;
      f[idl].v += hx/2 * xdot[idl].v + user->alpha * (x[idl].v - x[idr].v)/ hx;
      f[idr].u += hx/2 * xdot[idr].u + user->alpha * (x[idr].u - x[idl].u)/ hx;
      f[idr].v += hx/2 * xdot[idr].v + user->alpha * (x[idr].v - x[idl].v)/ hx;
    }
  }

  // Restore all the local vector data array references
  ierr = VecRestoreArrayRead(xltmp, (const PetscScalar**)&x);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(xdtmp, (const PetscScalar**)&xdot);CHKERRQ(ierr);
  ierr = VecRestoreArray(fltmp, (PetscScalar**)&f);CHKERRQ(ierr);

  ierr = VecGhostUpdateBegin(flocal,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  ierr = VecGhostUpdateEnd(flocal,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);

  ierr = VecGhostRestoreLocalForm(xlocal, &xltmp);CHKERRQ(ierr);
  ierr = VecGhostRestoreLocalForm(xdotlocal, &xdtmp);CHKERRQ(ierr);
  ierr = VecGhostRestoreLocalForm(flocal,&fltmp);CHKERRQ(ierr);

  ierr = DMLocalToGlobalBegin(dm,flocal,INSERT_VALUES,F);CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd(dm,flocal,INSERT_VALUES,F);CHKERRQ(ierr);

  ierr = DMRestoreLocalVector(dm, &xlocal);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dm, &xdotlocal);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dm, &flocal);CHKERRQ(ierr);  
  PetscFunctionReturn(0);
}

