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

typedef struct {
  PetscScalar u,v;
} Field;

typedef struct _User *User;
struct _User {
  PetscReal A,B;                /* Reaction coefficients */
  PetscReal alpha;              /* Diffusion coefficient */
  PetscReal uleft,uright;       /* Dirichlet boundary conditions */
  PetscReal vleft,vright;       /* Dirichlet boundary conditions */
  PetscInt  npts;               /* Number of mesh points */

  moab::ParallelComm *pcomm;
  moab::Interface *mbint;
  moab::Range *owned_vertexes;
  moab::Range *owned_edges;
  moab::Range *all_vertexes;
  moab::Range *shared_vertexes;
  moab::Tag    unknowns_tag;
  PetscInt     unknowns_tag_size;
  moab::Tag    id_tag;
};

static PetscErrorCode FormRHSFunction(TS,PetscReal,Vec,Vec,void*);
static PetscErrorCode FormIFunction(TS,PetscReal,Vec,Vec,Vec,void*);
static PetscErrorCode FormIJacobian(TS,PetscReal,Vec,Vec,PetscReal,Mat*,Mat*,MatStructure*,void*);

PetscErrorCode create_app_data(_User& user);
PetscErrorCode destroy_app_data(_User& user);
PetscErrorCode create_matrix(_User &user, DM &dm, Mat *J);

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
  PetscInt          steps,maxsteps,mx;
  PetscErrorCode    ierr;
  PetscReal         ftime,hx,dt;
  _User             user;       /* user-defined work context */
  TSConvergedReason reason;
  DM                dm;
  moab::ErrorCode   merr;

  PetscInitialize(&argc,&argv,(char *)0,help);

  // Fill in the user defined work context (creates mesh, solution field on mesh)
  ierr = create_app_data(user);CHKERRQ(ierr);

    // Create the DM to manage the mesh
  ierr = DMMoabCreateMoab(user.pcomm->comm(),user.mbint,user.pcomm,user.id_tag,user.owned_vertexes,&dm);CHKERRQ(ierr);

  // Create the solution vector:
  ierr = DMMoabCreateVector(dm,user.unknowns_tag,user.unknowns_tag_size,*user.owned_vertexes,PETSC_FALSE,PETSC_FALSE,&X);CHKERRQ(ierr);
    // Create the matrix
  ierr = create_matrix(user, dm, &J);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create timestepping solver context
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = TSCreate(PETSC_COMM_WORLD,&ts);CHKERRQ(ierr);
  ierr = TSSetType(ts,TSARKIMEX);CHKERRQ(ierr);

  ierr = TSSetRHSFunction(ts,PETSC_NULL,FormRHSFunction,&user);CHKERRQ(ierr);
  ierr = TSSetIFunction(ts,PETSC_NULL,FormIFunction,&user);CHKERRQ(ierr);
  ierr = TSSetIJacobian(ts,J,J,FormIJacobian,&user);CHKERRQ(ierr);

  ftime = 10.0;
  maxsteps = 10000;
  ierr = TSSetDuration(ts,maxsteps,ftime);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set initial conditions
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = TSSetSolution(ts,X);CHKERRQ(ierr);
  ierr = VecGetSize(X,&mx);CHKERRQ(ierr);
  hx = 1.0/(PetscReal)(mx/2-1);
  dt = 0.4 * PetscSqr(hx) / user.alpha; /* Diffusive stability limit */
  ierr = TSSetInitialTimeStep(ts,0.0,dt);CHKERRQ(ierr);

  // /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  //    Set runtime options
  //  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = TSSetFromOptions(ts);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Solve nonlinear system
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = TSSolve(ts,X);CHKERRQ(ierr);
  ierr = TSGetSolveTime(ts,&ftime);CHKERRQ(ierr);
  ierr = TSGetTimeStepNumber(ts,&steps);CHKERRQ(ierr);
  ierr = TSGetConvergedReason(ts,&reason);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"%s at time %G after %D steps\n",TSConvergedReasons[reason],ftime,steps);CHKERRQ(ierr);


  // Print the solution:
  ierr = VecView(X,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Write out the final mesh
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  merr = user.mbint->write_file("ex30-final.h5m");MBERRNM(merr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Free work space.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  // Free all PETSc related resources:
  ierr = MatDestroy(&J);CHKERRQ(ierr);
  ierr = VecDestroy(&X);CHKERRQ(ierr);
  ierr = TSDestroy(&ts);CHKERRQ(ierr);
  ierr = DMDestroy(&dm);CHKERRQ(ierr);

  // Free all MOAB related resources:
  ierr = destroy_app_data(user);CHKERRQ(ierr);

  ierr = PetscFinalize();
  return 0;
}

#undef __FUNCT__
#define __FUNCT__ "FormIFunction"
static PetscErrorCode FormIFunction(TS ts,PetscReal t,Vec X,Vec Xdot,Vec F,void *ptr)
{
  User            user = (User)ptr;
  PetscInt        i;
  Field           *x,*xdot,*f;
  PetscReal       hx;
  PetscErrorCode  ierr;
  moab::ErrorCode merr;
  int             *vertex_ids;

  PetscFunctionBegin;
  hx = 1.0/(PetscReal)(user->npts-1);

  // Get connectivity information:
  int verts_per_entity;
  int count,num_edges;
  moab::EntityHandle *connect;
  moab::Range::iterator iter = user->owned_edges->begin();
  merr = user->mbint->connect_iterate(iter,user->owned_edges->end(),connect,verts_per_entity,num_edges);MBERRNM(merr);

  // Get tag data:
  moab::Tag x_tag;
  ierr = DMMoabGetVecTag(X,&x_tag);CHKERRQ(ierr);
  merr = user->mbint->tag_iterate(x_tag,user->all_vertexes->begin(),user->all_vertexes->end(),
				  count,reinterpret_cast<void*&>(x));MBERRNM(merr);

  moab::Tag xdot_tag;
  ierr = DMMoabGetVecTag(Xdot,&xdot_tag);CHKERRQ(ierr);
  merr = user->mbint->tag_iterate(xdot_tag,user->all_vertexes->begin(),user->all_vertexes->end(),
				  count,reinterpret_cast<void*&>(xdot));MBERRNM(merr);

  moab::Tag f_tag;
  ierr = DMMoabGetVecTag(F,&f_tag);CHKERRQ(ierr);
  merr = user->mbint->tag_iterate(f_tag,user->all_vertexes->begin(),user->all_vertexes->end(),
				  count,reinterpret_cast<void*&>(f));MBERRNM(merr);

  // Get the global IDs on all vertexes:
  moab::Tag id_tag;
  ierr = user->mbint->tag_get_handle(GLOBAL_ID_TAG_NAME,id_tag);CHKERRQ(ierr);
  merr = user->mbint->tag_iterate(id_tag,user->all_vertexes->begin(),user->all_vertexes->end(),
  				  count,reinterpret_cast<void*&>(vertex_ids));MBERRNM(merr);

  // Exchange tags that are needed for assembly:
  merr = user->pcomm->exchange_tags(x_tag,*user->all_vertexes);MBERRNM(merr);
  merr = user->pcomm->exchange_tags(xdot_tag,*user->all_vertexes);MBERRNM(merr);

  // Compute f:
  const Field zero_field = {0.0,0.0};
  std::fill(f,f+user->all_vertexes->size(),zero_field);

  const moab::EntityHandle first_vertex = *user->all_vertexes->begin();

  for (i = 0; i < num_edges; i++) {
    const moab::EntityHandle idx_left  = connect[2*i]-first_vertex;
    const moab::EntityHandle idx_right = connect[2*i+1]-first_vertex;

    const int id_left  = vertex_ids[idx_left ];
    const int id_right = vertex_ids[idx_right];

    if (id_left == 0) {
      // Apply left BC
      f[idx_left].u += hx * (x[idx_left].u - user->uleft);
      f[idx_left].v += hx * (x[idx_left].v - user->vleft);
    } else {
      f[idx_left].u += hx * xdot[idx_left].u - user->alpha*(-2*x[idx_left].u + x[idx_right].u)/hx;
      f[idx_left].v += hx * xdot[idx_left].v - user->alpha*(-2*x[idx_left].v + x[idx_right].v)/hx;
    }

    if (id_right == user->npts-1) {
      // Apply right BC
      f[idx_right].u += hx * (x[idx_right].u - user->uright);
      f[idx_right].v += hx * (x[idx_right].v - user->vright);
    } else {
      f[idx_right].u -= user->alpha*x[idx_left].u/hx;
      f[idx_right].v -= user->alpha*x[idx_left].v/hx;
    }
  }

  // Add tags on shared vertexes:
  merr = user->pcomm->reduce_tags(f_tag,MPI_SUM,*user->shared_vertexes);MBERRNM(merr);

  // Print vectors for debugging:
  // ierr = PetscPrintf(PETSC_COMM_WORLD, "X\n");CHKERRQ(ierr);
  // ierr = VecView(X,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  // ierr = PetscPrintf(PETSC_COMM_WORLD, "Xdot\n");CHKERRQ(ierr);
  // ierr = VecView(Xdot,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  // ierr = PetscPrintf(PETSC_COMM_WORLD, "F\n");CHKERRQ(ierr);
  // ierr = VecView(F,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FormRHSFunction"
static PetscErrorCode FormRHSFunction(TS ts,PetscReal t,Vec X,Vec F,void *ptr)
{
  User            user = (User)ptr;
  PetscReal       hx;
  Field           *x,*f;
  PetscErrorCode  ierr;
  moab::Tag       id_tag;
  moab::ErrorCode merr;

  PetscFunctionBegin;
  hx = 1.0/(PetscReal)(user->npts-1);

  merr = user->mbint->tag_get_handle(GLOBAL_ID_TAG_NAME,id_tag);MBERRNM(merr);

  /* Get pointers to vector data */
  ierr = VecGetArray(X,reinterpret_cast<PetscScalar**>(&x));CHKERRQ(ierr);
  ierr = VecGetArray(F,reinterpret_cast<PetscScalar**>(&f));CHKERRQ(ierr);

  /* Compute function over the locally owned part of the grid */
  const moab::EntityHandle first_vertex = *user->owned_vertexes->begin();
  moab::Range::iterator iter;
  for(iter = user->owned_vertexes->begin(); iter != user->owned_vertexes->end(); iter++) {
    moab::EntityHandle i = *iter - first_vertex;
    PetscScalar u = x[i].u,v = x[i].v;
    f[i].u = hx*(user->A + u*u*v - (user->B+1)*u);
    f[i].v = hx*(user->B*u - u*u*v);
  }

  /* Restore vectors */
  ierr = VecRestoreArray(X,reinterpret_cast<PetscScalar**>(&x));CHKERRQ(ierr);
  ierr = VecRestoreArray(F,reinterpret_cast<PetscScalar**>(&f));CHKERRQ(ierr);

  // Print vectors for debugging:
  /* ierr = PetscPrintf(PETSC_COMM_WORLD,"RHS");CHKERRQ(ierr); */
  /* ierr = VecView(F,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr); */

  PetscFunctionReturn(0);
}

/* --------------------------------------------------------------------- */
/*
  IJacobian - Compute IJacobian = dF/dU + a dF/dUdot
*/
#undef __FUNCT__
#define __FUNCT__ "FormIJacobian"
PetscErrorCode FormIJacobian(TS ts,PetscReal t,Vec X,Vec Xdot,PetscReal a,Mat *J,Mat *Jpre,MatStructure *str,void *ptr)
{
  User            user = (User)ptr;
  PetscErrorCode  ierr;
  PetscInt        i;
  PetscReal       hx;
  moab::Tag       id_tag;
  moab::ErrorCode merr;

  PetscFunctionBegin;
  hx = 1.0/(PetscReal)(user->npts-1);

  merr = user->mbint->tag_get_handle(GLOBAL_ID_TAG_NAME,id_tag);MBERRNM(merr);

  /* Compute function over the locally owned part of the grid */
  moab::Range::iterator iter;
  for(iter = user->owned_vertexes->begin(); iter != user->owned_vertexes->end(); iter++) {
    merr = user->mbint->tag_get_data(id_tag,&(*iter),1,&i);MBERRNM(merr);

    if (i == 0 || i == user->npts-1) {
      // Boundary conditions...
      const PetscInt row = i,col = i;
      const PetscScalar vals[2][2] = {{hx,0},{0,hx}};
      ierr = MatSetValuesBlocked(*Jpre,1,&row,1,&col,&vals[0][0],INSERT_VALUES);CHKERRQ(ierr);
    } else {
      //
      const PetscInt row = i,col[] = {i-1,i,i+1};
      const PetscScalar dxxL = -user->alpha/hx,dxx0 = 2.*user->alpha/hx,dxxR = -user->alpha/hx;
      const PetscScalar vals[2][3][2] = {{{dxxL,0},{a*hx+dxx0,0},{dxxR,0}},
                                         {{0,dxxL},{0,a*hx+dxx0},{0,dxxR}}};
      ierr = MatSetValuesBlocked(*Jpre,1,&row,3,col,&vals[0][0][0],INSERT_VALUES);CHKERRQ(ierr);
    }
  }

  ierr = MatAssemblyBegin(*Jpre,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*Jpre,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  if (*J != *Jpre) {
    ierr = MatAssemblyBegin(*J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(*J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }

  /* ierr = MatView(*J,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr); */

  PetscFunctionReturn(0);
}


/********************************
 *                              *
 *     initialize_moab_mesh     *
 *                              *
 ********************************/

#undef __FUNCT__
#define __FUNCT__ "initialize_moab_mesh"
PetscErrorCode initialize_moab_mesh(moab::ParallelComm* pcomm,int npts,int nghost,moab::Tag &unknowns_tag,PetscInt &unknowns_tag_size,moab::Tag &id_tag)
{
  moab::ErrorCode merr;
  PetscInt num_procs;
  PetscInt rank;

  PetscFunctionBegin;
  MPI_Comm_size( pcomm->comm(),&num_procs );
  MPI_Comm_rank( pcomm->comm(),&rank );

  // Begin with some error checking:
  if(npts < 2) {
    SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_OUTOFRANGE,"npts must be >= 2");
  }

  if(num_procs >= npts) {
    SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_OUTOFRANGE,"Num procs must be < npts");
  }

  // No errors,proceed with building the mesh:
  moab::Interface *mbint = pcomm->get_moab();
  moab::ReadUtilIface* readMeshIface;
  mbint->query_interface(readMeshIface);

  // Determine which elements (cells) this process owns:
  const PetscInt nele = npts-1;
  PetscInt my_nele = nele / num_procs; // Number of elements owned by this process
  const PetscInt extra = nele % num_procs;
  PetscInt vstart = rank * my_nele;    // The starting element for this process
  if(rank < extra) my_nele++;
  vstart += std::min(extra, rank);
  const PetscInt my_npts = my_nele + 1;

  // Create the local portion of the mesh:

  // Create vertexes and set the coodinate of each vertex:
  moab::EntityHandle vertex_first;
  std::vector<double*> vertex_coords;
  const int sequence_size = (my_nele + 2) + 1;
  merr = readMeshIface->get_node_coords(3,my_npts,1,vertex_first,vertex_coords,sequence_size);MBERRNM(merr);

  const double xs = 0.0, xe = 1.0;
  const double dx = (xe - xs) / nele;
  for (int i = 0; i < my_npts; i++) {
    vertex_coords[0][i] = (i+vstart)*dx;
    vertex_coords[1][i] = vertex_coords[2][i] = 0.0;
  }

  moab::Range owned_vertexes;
  merr = mbint->get_entities_by_type(0,moab::MBVERTEX,owned_vertexes);MBERRNM(merr);

  // Create elements between mesh points. This is done so that VisIt
  // will interpret the output as a mesh that can be plotted...
  moab::EntityHandle edge_first;
  moab::EntityHandle *connectivity = 0;
  merr = readMeshIface->get_element_connect(my_nele,2,moab::MBEDGE,1,edge_first,connectivity);MBERRNM(merr);

  for (int i = 0; i < my_nele; i+=1) {
    connectivity[2*i]   = vertex_first + i;
    connectivity[2*i+1] = vertex_first + (i+1);
  }

  merr = readMeshIface->update_adjacencies(edge_first,my_nele,2,connectivity);MBERRNM(merr);

  // Set tags on all of the vertexes...

  // Create tag handle to represent the unknowns, u and v and
  // initialize them. We will create a single tag whose type is an
  // array of two doubles and whose name is "unknowns"
  Field default_value = {0.0,0.0};
  unknowns_tag_size = sizeof(Field)/sizeof(PetscScalar);
  merr = mbint->tag_get_handle("unknowns",unknowns_tag_size,moab::MB_TYPE_DOUBLE,unknowns_tag,
                              moab::MB_TAG_DENSE | moab::MB_TAG_CREAT,&default_value);MBERRNM(merr);

  // Apply the "unknowns" tag to the vertexes with some initial value...
  std::vector<Field> tag_data(my_npts);
  for (int i = 0; i < my_npts; i++) {
    tag_data[i].u = 1+sin(2*PETSC_PI*vertex_coords[0][i]);
    tag_data[i].v = 3;
  }

  merr = mbint->tag_set_data(unknowns_tag,owned_vertexes,tag_data.data());MBERRNM(merr);

  // Get the global ID tag. The global ID tag is applied to each
  // vertex. It acts as an global identifier which MOAB uses to
  // assemble the individual pieces of the mesh:
  merr = mbint->tag_get_handle(GLOBAL_ID_TAG_NAME,id_tag);MBERRNM(merr);

  std::vector<int> global_ids(my_npts);
  for (int i = 0; i < my_npts; i++) {
    global_ids[i] = i+vstart;
  }

  merr = mbint->tag_set_data(id_tag,owned_vertexes,global_ids.data());MBERRNM(merr);

  moab::Range owned_edges;
  merr = mbint->get_entities_by_type(0,moab::MBEDGE,owned_edges);MBERRNM(merr);
  merr = pcomm->resolve_shared_ents(0,owned_edges,1,0);MBERRNM(merr);

  // Reassign global IDs on all entities.
  merr = pcomm->assign_global_ids(0,1,0,false);MBERRNM(merr);
  merr = pcomm->exchange_ghost_cells( 1,0,nghost,0,true);MBERRNM(merr);

  // Everything is set up, now just do a tag exchange to update tags
  // on all of the shared vertexes:
  merr = pcomm->exchange_tags(id_tag,owned_vertexes);MBERRNM(merr);

  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "create_app_data"
PetscErrorCode create_app_data(_User& user)
{
  PetscErrorCode ierr;
  moab::ErrorCode merr;

  PetscFunctionBegin;

  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,PETSC_NULL,"Advection-reaction options",""); {
    user.A      = 1;
    user.B      = 3;
    user.alpha  = 0.02;
    user.uleft  = 1;
    user.uright = 1;
    user.vleft  = 3;
    user.vright = 3;
    user.npts   = 11;
    ierr = PetscOptionsReal("-A","Reaction rate","",user.A,&user.A,PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-B","Reaction rate","",user.B,&user.B,PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-alpha","Diffusion coefficient","",user.alpha,&user.alpha,PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-uleft","Dirichlet boundary condition","",user.uleft,&user.uleft,PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-uright","Dirichlet boundary condition","",user.uright,&user.uright,PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-vleft","Dirichlet boundary condition","",user.vleft,&user.vleft,PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-vright","Dirichlet boundary condition","",user.vright,&user.vright,PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-npts","Number of mesh points","",user.npts,&user.npts,PETSC_NULL);CHKERRQ(ierr);
  } ierr = PetscOptionsEnd();CHKERRQ(ierr);

  user.mbint = new moab::Core;
  user.pcomm = new moab::ParallelComm(user.mbint,PETSC_COMM_WORLD);

  ierr = initialize_moab_mesh(user.pcomm,user.npts,1,user.unknowns_tag,user.unknowns_tag_size,user.id_tag); CHKERRQ(ierr); // creates moab mesh and field of unknowns on vertices

  // Set the edge/vertex range once so we don't have to do it
  // again. To do this we get all of the edges/vertexes then filter so
  // we have only the owned entities in the ranges:
  user.owned_edges = new moab::Range;
  merr = user.mbint->get_entities_by_type(0,moab::MBEDGE,*user.owned_edges);MBERRNM(merr);

  user.owned_vertexes = new moab::Range;
  merr = user.mbint->get_entities_by_type(0,moab::MBVERTEX,*user.owned_vertexes);MBERRNM(merr);
  user.all_vertexes = new moab::Range(*user.owned_vertexes);

  user.shared_vertexes = new moab::Range;
  merr = user.mbint->get_entities_by_type(0,moab::MBVERTEX,*user.shared_vertexes);MBERRNM(merr);

  merr = user.pcomm->filter_pstatus(*user.owned_edges,PSTATUS_NOT_OWNED,PSTATUS_NOT);MBERRNM(merr);

  merr = user.pcomm->filter_pstatus(*user.owned_vertexes,PSTATUS_NOT_OWNED,PSTATUS_NOT);MBERRNM(merr);

  merr = user.pcomm->filter_pstatus(*user.shared_vertexes,PSTATUS_SHARED,PSTATUS_OR);MBERRNM(merr);

  // Do some error checking...make sure that tag_data is in a single
  // sequence:
  int count;
  void *data;
  moab::Tag tag;
  merr = user.mbint->tag_get_handle("unknowns",tag);MBERRNM(merr);
  merr = user.mbint->tag_iterate(tag,user.all_vertexes->begin(),user.all_vertexes->end(),
				 count,data);MBERRNM(merr);
  if((unsigned int) count != user.all_vertexes->size()) {
    SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_LIB,"Tag data not laid out contiguously %i %i",
	     count,user.all_vertexes->size());
  }

  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "create_matrix"
PetscErrorCode create_matrix(_User &user, DM &dm, Mat *J)
{
  PetscErrorCode ierr;
  moab::ErrorCode merr;
  moab::Tag ltog_tag;
  moab::Range range;

  PetscFunctionBegin;
  ierr = DMMoabGetLocalToGlobalTag(dm,&ltog_tag);CHKERRQ(ierr);
  ierr = DMMoabGetRange(dm,&range);CHKERRQ(ierr);

  // Create the matrix:
  ierr = MatCreateBAIJ(user.pcomm->comm(),2,2*range.size(),2*range.size(),
  		       PETSC_DECIDE,PETSC_DECIDE,3, NULL, 3, NULL, J);CHKERRQ(ierr);

  // Set local to global numbering using the ltog_tag:
  ISLocalToGlobalMapping ltog;
  PetscInt               *gindices = new PetscInt[range.size()];
  merr = user.pcomm->get_moab()->tag_get_data(ltog_tag, range, gindices);MBERRNM(merr);

  ierr = ISLocalToGlobalMappingCreate(PETSC_COMM_SELF, range.size(), gindices,PETSC_COPY_VALUES, &ltog);CHKERRQ(ierr);
  ierr = MatSetLocalToGlobalMappingBlock(*J,ltog,ltog);CHKERRQ(ierr);

  // Clean up:
  ierr = ISLocalToGlobalMappingDestroy(&ltog);CHKERRQ(ierr);
  delete [] gindices;

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "destroy_app_data"
PetscErrorCode destroy_app_data(_User& user)
{
  PetscFunctionBegin;

  delete user.owned_edges;
  delete user.owned_vertexes;
  delete user.all_vertexes;
  delete user.shared_vertexes;
  delete user.pcomm;
  delete user.mbint;

  PetscFunctionReturn(0);
}
