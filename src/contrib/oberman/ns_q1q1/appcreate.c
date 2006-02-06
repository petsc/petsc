#include "appctx.h"

/*
         -  Generates the "global" parallel vector to contain the 
	    right hand side and solution.
         -  Generates "ghosted" local vectors for local computations etc.
         -  Generates scatter context for updating ghost points etc.
*/
#undef __FUNC__
#define __FUNC__ "AppCxtCreateVector"
int AppCtxCreateVector(AppCtx* appctx)
{
/********* Collect context informatrion ***********/
  AppGrid                *grid = &appctx->grid;
  AppAlgebra             *algebra = &appctx->algebra;
  MPI_Comm               comm = appctx->comm;
  /* The local to global mapping */
 ISLocalToGlobalMapping ltog = grid->ltog;
 /* number of vertices on this processor */
  int   vertex_n = grid->vertex_n;
  /* number of vertices including ghosted ones */
 int vertex_n_ghosted = grid->vertex_n_ghosted;
 /* global number of each vertex on the processor */
 IS  vertex_global = grid->vertex_global;
/* blocked global number of each vertex on the processor */
 IS  vertex_global_blocked   = grid->vertex_global_blocked;
 /************* Variables to set ************/
/* global to local mapping for vectors */
 VecScatter    gtol;
 VecScatter    dgtol; /* for the nonlinear funtion */
 Vec x, z;  /* x,z for nonzero structure,*/
 Vec x_local, w_local, z_local; /* used for determining nonzero structure of matriz */
Vec b;  /* b for holding rhs */
Vec f; /* nonlinear function */
Vec g; /*for solution, and initial guess */
Vec *solnv; /* for solution at each time step */
 Vec f_local; /* used for nonlinear function */
 /********** Internal Variables **********/
int ierr,its, i;

  PetscFunctionBegin;
  /*  Create vector to contain load, nonlinear function, and initial guess  */
  ierr = VecCreateMPI(comm,DF*vertex_n,PETSC_DECIDE,&b);CHKERRQ(ierr);
  ierr = VecSetBlockSize(b , DF);  CHKERRQ(ierr);
  ierr = VecSetLocalToGlobalMappingBlocked(b,ltog);CHKERRQ(ierr);

  /* NNEEEDDD TO FIGURE THIS OUT */
 ierr = VecSetLocalToGlobalMapping(b, grid->dltog);CHKERRQ(ierr);
 /*use the blocked ltog for the larger index set */

  /* For load and solution vectors.  Duplicated vectors inherit the blocking */
  ierr = VecDuplicate(b,&f);CHKERRQ(ierr);/*  the nonlinear function */
  ierr = VecDuplicate(b,&g);CHKERRQ(ierr);/*  the initial guess  */
  ierr = VecDuplicateVecs(b, NSTEPS+1, &solnv);CHKERRQ(ierr);  /* the soln at each time step */
  /* later dynamically make block of size 16 of solution vectors for dynamic time stepping */

  /* set it */
  algebra->b       = b;
  algebra->f       = f;
  algebra->g       = g;
  algebra->solnv = solnv;

  ierr = VecCreateSeq(PETSC_COMM_SELF,DF*vertex_n_ghosted,&f_local);CHKERRQ(ierr);
  ierr = VecScatterCreate(f,vertex_global_blocked,f_local,0,&dgtol);CHKERRQ(ierr);
  /* for vecscatter, second argument is the IS to scatter.  
 Use the  blocked form created in appload.c */  

  /* set variables */
  algebra->f_local = f_local;
  algebra->dgtol = dgtol;

  /* Create work vectors for MatCreate */
  ierr = VecCreateMPI(comm,vertex_n,PETSC_DECIDE,&x);CHKERRQ(ierr);
  ierr = VecSetLocalToGlobalMapping(x,ltog);CHKERRQ(ierr);
  ierr = VecDuplicate(x,&z);
 
  /* Create local work vectors for MatCreate */
  ierr = VecCreateSeq(PETSC_COMM_SELF,vertex_n_ghosted,&w_local);CHKERRQ(ierr);
  ierr = VecDuplicate(w_local,&x_local);CHKERRQ(ierr);
  ierr = VecDuplicate(w_local,&z_local);CHKERRQ(ierr);
  /* only need to create one scatter, it works on duplicate  vectors */ 
  ierr = VecScatterCreate(x,vertex_global,w_local,0,&gtol);CHKERRQ(ierr);

  /* set variables */
  algebra->x       = x;
  algebra->z       = z;
  algebra->w_local = w_local;
  algebra->x_local = x_local;
  algebra->z_local = z_local;
  algebra->gtol    = gtol;

  PetscFunctionReturn(0);
}

/*
     Creates the sparse matrix (with the correct nonzero pattern) that will
  be later filled with the stiffness matrix
*/
#undef __FUNC__
#define __FUNC__ "AppCxtCreateMatrix"
int AppCtxCreateMatrix(AppCtx* appctx)
{
/********* Collect context informatrion ***********/
  AppAlgebra             *algebra = &appctx->algebra;
  AppGrid                *grid    = &appctx->grid;
  /* these vectors should all have one space per node */ 
 Vec x = algebra->x,  z = algebra->z;
  Vec w_local = algebra->w_local, x_local = algebra->x_local; 
  Vec z_local = algebra->z_local;
  VecScatter             gtol = algebra->gtol;
  MPI_Comm               comm = appctx->comm;
  /* The local to global mapping */
 ISLocalToGlobalMapping ltog = grid->ltog;
/* the blocked ltod, for by hand blocking */
  ISLocalToGlobalMapping dltog = grid->dltog;
  /* The IS of the (blocked) vertices */
IS vertex_global_blocked = grid->vertex_global_blocked;
 /* number of vertices on this processor */
  int   vertex_n = grid->vertex_n;
  /* number of cells on this processor */
  int    cell_n = grid->cell_n;
  /* neighbours of the cell */
  int  *cell_cell = grid->cell_cell;
  /* vertices of the cell (in local numbering) */
  int  *cell_vertex = grid->cell_vertex;
  /* number of vertices on each cell (=4) */
int NV = grid->NV;

 /************* Variables to set ************/
  Mat A;
  Mat J;

 /********** Internal Variables **********/
  double *sdnz, *sonz;  /* non-zero entries on this processor, non-zero entries off this processor */
   int *onz, *dnz;
   int rank; double srank;  /* copies of the integer variables */
   const int four = 4;
   double *procs; 
   double  wght,  zero = 0.0, one = 1.0;
   int   ierr, cproc, i, j;
   int  *cells,*vertices; 
   int *vertex_gb;
   double *idvalues;
 
  /* We are doing everything blocked, so just use the vectors with one values
 per vertex, same for one degree of freedom per node, then CreateBlocked.  */
  PetscFunctionBegin;
 /* ------------------------------------------------
      Determine non-zero structure of the matrix 
      --------------------------------------------*/
  /* 1) make proc[] contain the processor number of each ghosted vertex */

  MPI_Comm_rank(comm,&rank);    /* Get the index of this processor */
  srank = rank;
   /* set all values of x to the index of this processor */
  ierr = VecSet(x,srank);CHKERRQ(ierr);           

  /* w_local contains all vertices, including ghosted that this processor uses */
  ierr = VecScatterBegin(x,w_local,INSERT_VALUES,SCATTER_FORWARD,gtol);CHKERRQ(ierr);
  ierr = VecScatterEnd(x,w_local,INSERT_VALUES,SCATTER_FORWARD,gtol);CHKERRQ(ierr);

  /* copy w_local into the array procs */
  ierr = VecGetArray(w_local,&procs);CHKERRQ(ierr);
  /* make an array the size x_local ( total number of vertices, including ghosted) ,
 this is for the elements on this processor */ 
  ierr = VecSet(x_local,zero);CHKERRQ(ierr);   
  ierr = VecGetArray(x_local,&sdnz);CHKERRQ(ierr);  
  /* make an array of appropriate size, for the  vertices off this processor */
  ierr = VecSet(z_local,zero);CHKERRQ(ierr); 
  ierr = VecGetArray(z_local,&sonz);CHKERRQ(ierr);

  /* 2) loop over local elements; count matrix nonzeros */

  /*  For each vertex, we count the number of nonzero entries in the matrix.  This is done by looking at how many other vertices are adjacent,  at least in the current case of billinear elements we only have elements on the vertices.  We compute this efficiently, by looping over cells, the vertices, and weighting with .5 those vertices which are adjacen and have nieghbouring element and so will be counted twice.  For data management purposes we need to know if the elements are on or off - processor, so we put the count into sdnz, or donz respectively.  */

  /* loop over cells */
  for ( i=0; i<cell_n; i++ ) {
    vertices = cell_vertex + four*i;
    cells    = cell_cell   + four*i;
    /* loop over vertices */
    for ( j=0; j<four; j += 1 ) {
      cproc = PetscReal(procs[vertices[j]]);
      /* 1st neighbor, -adjacent */
      if (cells[j] >= 0) wght = .5; else wght = 1.0;
      if (cproc == procs[vertices[(j+1) % four ]]) { /* on diagonal part */
        sdnz[vertices[j]] += wght;} 
      else { sonz[vertices[j]] += wght;}
      /* 2nd neighbor - diagonally opposite*/
     if (cproc == procs[vertices[(j+2) % four]]) { /* on diagonal part */
        sdnz[vertices[j]] += 1.0;}
     else{ sonz[vertices[j]] += 1.0;}
      /* 3rd neighbor  - adjacent */
      if (cells[(j+3)% four ] >= 0) wght = .5; else wght = 1.0; /* check if it has an adjacent cell */
      if (cproc == procs[vertices[(j+3) % four]]) { /* on diagonal part */
        sdnz[vertices[j]] += wght; } 
      else {sonz[vertices[j]] += wght;}
    }
  }

  /* Put the values back into the vectors */
  ierr = VecRestoreArray(x_local,&sdnz);CHKERRQ(ierr);
  ierr = VecRestoreArray(z_local,&sonz);CHKERRQ(ierr);
  ierr = VecRestoreArray(w_local,&procs);CHKERRQ(ierr);

  /* copy the local values up into x. */
  ierr = VecSet(x,zero);CHKERRQ(ierr);
  ierr = VecScatterBegin(x_local,x,ADD_VALUES,SCATTER_REVERSE,gtol);CHKERRQ(ierr);
  ierr = VecScatterEnd(x_local,x,ADD_VALUES,SCATTER_REVERSE,gtol);CHKERRQ(ierr);
  ierr = VecGetArray(x,&sdnz);CHKERRQ(ierr);
  /* copy the local values up into z. */
  ierr = VecSet(z,zero);CHKERRQ(ierr);
  ierr = VecScatterBegin(z_local,z,ADD_VALUES,SCATTER_REVERSE,gtol);CHKERRQ(ierr);
  ierr = VecScatterEnd(z_local,z,ADD_VALUES,SCATTER_REVERSE,gtol);CHKERRQ(ierr);
  ierr = VecGetArray(z,&sonz);CHKERRQ(ierr);

  /* now copy values into and integer array, adding one for the diagonal entry */
  dnz  = (int *) PetscMalloc((vertex_n+1)*sizeof(int));CHKPTRQ(dnz);
  onz  = (int *) PetscMalloc((vertex_n+1)*sizeof(int));CHKPTRQ(onz);
  for ( i=0; i<vertex_n; i++ ) {
    dnz[i] = 1 + (int) PetscReal(sdnz[i]);
    onz[i] = (int) PetscReal(sonz[i]);
  }
  ierr = VecRestoreArray(x,&sdnz);CHKERRQ(ierr);
  ierr = VecRestoreArray(z,&sonz);CHKERRQ(ierr);

  /* now create the matrix */
  ierr = MatCreateMPIBAIJ(comm, DF, DF*vertex_n,DF*vertex_n,PETSC_DETERMINE,PETSC_DETERMINE,0,dnz,0,onz,&A); CHKERRQ(ierr);
  ierr = MatSetLocalToGlobalMappingBlocked(A,ltog);CHKERRQ(ierr);

  /* Dupicate the matrix for now.  Later the Jacobian will not have the same nonzero structure  */
   ierr = MatCreateMPIBAIJ(comm, DF, DF*vertex_n,DF*vertex_n,PETSC_DETERMINE,PETSC_DETERMINE,0,dnz,0,onz,&J); CHKERRQ(ierr);
  ierr = MatSetLocalToGlobalMappingBlocked(J,ltog);CHKERRQ(ierr); 
 ierr = MatSetLocalToGlobalMapping(J,dltog);CHKERRQ(ierr); 

  ierr = PetscFree(dnz);CHKERRQ(ierr);
  ierr = PetscFree(onz);CHKERRQ(ierr);
  algebra->A = A;
  algebra->J = J;
  PetscFunctionReturn(0);
}
