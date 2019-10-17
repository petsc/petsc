
/*
   Include "petscsnes.h" so that we can use SNES solvers.  Note that this
   file automatically includes:
     petscsys.h       - base PETSc routines   petscvec.h - vectors
     petscmat.h - matrices
     petscis.h     - index sets            petscksp.h - Krylov subspace methods
     petscviewer.h - viewers               petscpc.h  - preconditioners
     petscksp.h   - linear solvers
*/
#include <petscsnes.h>
#include <petscao.h>

static char help[] = "An Unstructured Grid Example.\n\
This example demonstrates how to solve a nonlinear system in parallel\n\
with SNES for an unstructured mesh. The mesh and partitioning information\n\
is read in an application defined ordering,which is later transformed\n\
into another convenient ordering (called the local ordering). The local\n\
ordering, apart from being efficient on cpu cycles and memory, allows\n\
the use of the SPMD model of parallel programming. After partitioning\n\
is done, scatters are created between local (sequential)and global\n\
(distributed) vectors. Finally, we set up the nonlinear solver context\n\
in the usual way as a structured grid  (see\n\
petsc/src/snes/examples/tutorials/ex5.c).\n\
This example also illustrates the use of parallel matrix coloring.\n\
The command line options include:\n\
  -vert <Nv>, where Nv is the global number of nodes\n\
  -elem <Ne>, where Ne is the global number of elements\n\
  -nl_par <lambda>, where lambda is the multiplier for the non linear term (u*u) term\n\
  -lin_par <alpha>, where alpha is the multiplier for the linear term (u)\n\
  -fd_jacobian_coloring -mat_coloring_type lf\n";

/*T
   Concepts: SNES^unstructured grid
   Concepts: AO^application to PETSc ordering or vice versa;
   Concepts: VecScatter^using vector scatter operations;
   Processors: n
T*/



/* ------------------------------------------------------------------------

   PDE Solved : L(u) + lambda*u*u + alpha*u = 0 where L(u) is the Laplacian.

   The Laplacian is approximated in the following way: each edge is given a weight
   of one meaning that the diagonal term will have the weight equal to the degree
   of a node. The off diagonal terms will get a weight of -1.

   -----------------------------------------------------------------------*/


#define MAX_ELEM      500  /* Maximum number of elements */
#define MAX_VERT      100  /* Maximum number of vertices */
#define MAX_VERT_ELEM   3  /* Vertices per element       */

/*
  Application-defined context for problem specific data
*/
typedef struct {
  PetscInt   Nvglobal,Nvlocal;              /* global and local number of vertices */
  PetscInt   Neglobal,Nelocal;              /* global and local number of vertices */
  PetscInt   AdjM[MAX_VERT][50];            /* adjacency list of a vertex */
  PetscInt   itot[MAX_VERT];                /* total number of neighbors for a vertex */
  PetscInt   icv[MAX_ELEM][MAX_VERT_ELEM];  /* vertices belonging to an element */
  PetscInt   v2p[MAX_VERT];                 /* processor number for a vertex */
  PetscInt   *locInd,*gloInd;               /* local and global orderings for a node */
  Vec        localX,localF;                 /* local solution (u) and f(u) vectors */
  PetscReal  non_lin_param;                 /* nonlinear parameter for the PDE */
  PetscReal  lin_param;                     /* linear parameter for the PDE */
  VecScatter scatter;                       /* scatter context for the local and
                                               distributed vectors */
} AppCtx;

/*
  User-defined routines
*/
PetscErrorCode FormJacobian(SNES,Vec,Mat,Mat,void*);
PetscErrorCode FormFunction(SNES,Vec,Vec,void*);
PetscErrorCode FormInitialGuess(AppCtx*,Vec);

int main(int argc,char **argv)
{
  SNES                   snes;                 /* SNES context */
  SNESType               type = SNESNEWTONLS;  /* default nonlinear solution method */
  Vec                    x,r;                  /* solution, residual vectors */
  Mat                    Jac;                  /* Jacobian matrix */
  AppCtx                 user;                 /* user-defined application context */
  AO                     ao;                   /* Application Ordering object */
  IS                     isglobal,islocal;     /* global and local index sets */
  PetscMPIInt            rank,size;            /* rank of a process, number of processors */
  PetscInt               rstart;               /* starting index of PETSc ordering for a processor */
  PetscInt               nfails;               /* number of unsuccessful Newton steps */
  PetscInt               bs = 1;               /* block size for multicomponent systems */
  PetscInt               nvertices;            /* number of local plus ghost nodes of a processor */
  PetscInt               *pordering;           /* PETSc ordering */
  PetscInt               *vertices;            /* list of all vertices (incl. ghost ones) on a processor */
  PetscInt               *verticesmask;
  PetscInt               *tmp;
  PetscInt               i,j,jstart,inode,nb,nbrs,Nvneighborstotal = 0;
  PetscErrorCode         ierr;
  PetscInt               its,N;
  PetscScalar            *xx;
  char                   str[256],form[256],part_name[256];
  FILE                   *fptr,*fptr1;
  ISLocalToGlobalMapping isl2g;
  int                    dtmp;
#if defined(UNUSED_VARIABLES)
  PetscDraw              draw;                 /* drawing context */
  PetscScalar            *ff,*gg;
  PetscReal              tiny = 1.0e-10,zero = 0.0,one = 1.0,big = 1.0e+10;
  PetscInt               *tmp1,*tmp2;
#endif
  MatFDColoring          matfdcoloring = 0;
  PetscBool              fd_jacobian_coloring = PETSC_FALSE;

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Initialize program
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr =  PetscInitialize(&argc,&argv,"options.inf",help);if (ierr) return ierr;
  ierr = MPI_Comm_rank(MPI_COMM_WORLD,&rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(MPI_COMM_WORLD,&size);CHKERRQ(ierr);

  /* The current input file options.inf is for 2 proc run only */
  if (size != 2) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_WRONG_MPI_SIZE,"This example currently runs on 2 procs only.");

  /*
     Initialize problem parameters
  */
  user.Nvglobal = 16;      /*Global # of vertices  */
  user.Neglobal = 18;      /*Global # of elements  */

  ierr = PetscOptionsGetInt(NULL,NULL,"-vert",&user.Nvglobal,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-elem",&user.Neglobal,NULL);CHKERRQ(ierr);

  user.non_lin_param = 0.06;

  ierr = PetscOptionsGetReal(NULL,NULL,"-nl_par",&user.non_lin_param,NULL);CHKERRQ(ierr);

  user.lin_param = -1.0;

  ierr = PetscOptionsGetReal(NULL,NULL,"-lin_par",&user.lin_param,NULL);CHKERRQ(ierr);

  user.Nvlocal = 0;
  user.Nelocal = 0;

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      Read the mesh and partitioning information
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /*
     Read the mesh and partitioning information from 'adj.in'.
     The file format is as follows.
     For each line the first entry is the processor rank where the
     current node belongs. The second entry is the number of
     neighbors of a node. The rest of the line is the adjacency
     list of a node. Currently this file is set up to work on two
     processors.

     This is not a very good example of reading input. In the future,
     we will put an example that shows the style that should be
     used in a real application, where partitioning will be done
     dynamically by calling partitioning routines (at present, we have
     a  ready interface to ParMeTiS).
   */
  fptr = fopen("adj.in","r");
  if (!fptr) SETERRQ(PETSC_COMM_SELF,0,"Could not open adj.in");

  /*
     Each processor writes to the file output.<rank> where rank is the
     processor's rank.
  */
  sprintf(part_name,"output.%d",rank);
  fptr1 = fopen(part_name,"w");
  if (!fptr1) SETERRQ(PETSC_COMM_SELF,0,"Could no open output file");
  ierr = PetscMalloc1(user.Nvglobal,&user.gloInd);CHKERRQ(ierr);
  ierr = PetscFPrintf(PETSC_COMM_SELF,fptr1,"Rank is %d\n",rank);CHKERRQ(ierr);
  for (inode = 0; inode < user.Nvglobal; inode++) {
    if (!fgets(str,256,fptr)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_FILE_READ,"fgets read failed");
    sscanf(str,"%d",&dtmp);user.v2p[inode] = dtmp;
    if (user.v2p[inode] == rank) {
      ierr = PetscFPrintf(PETSC_COMM_SELF,fptr1,"Node %D belongs to processor %D\n",inode,user.v2p[inode]);CHKERRQ(ierr);

      user.gloInd[user.Nvlocal] = inode;
      sscanf(str,"%*d %d",&dtmp);
      nbrs = dtmp;
      ierr = PetscFPrintf(PETSC_COMM_SELF,fptr1,"Number of neighbors for the vertex %D is %D\n",inode,nbrs);CHKERRQ(ierr);

      user.itot[user.Nvlocal] = nbrs;
      Nvneighborstotal       += nbrs;
      for (i = 0; i < user.itot[user.Nvlocal]; i++) {
        form[0]='\0';
        for (j=0; j < i+2; j++) {
          ierr = PetscStrlcat(form,"%*d ",sizeof(form));CHKERRQ(ierr);
        }
        ierr = PetscStrlcat(form,"%d",sizeof(form));CHKERRQ(ierr);

        sscanf(str,form,&dtmp);
        user.AdjM[user.Nvlocal][i] = dtmp;

        ierr = PetscFPrintf(PETSC_COMM_SELF,fptr1,"%D ",user.AdjM[user.Nvlocal][i]);CHKERRQ(ierr);
      }
      ierr = PetscFPrintf(PETSC_COMM_SELF,fptr1,"\n");CHKERRQ(ierr);
      user.Nvlocal++;
    }
  }
  ierr = PetscFPrintf(PETSC_COMM_SELF,fptr1,"Total # of Local Vertices is %D \n",user.Nvlocal);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create different orderings
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /*
    Create the local ordering list for vertices. First a list using the PETSc global
    ordering is created. Then we use the AO object to get the PETSc-to-application and
    application-to-PETSc mappings. Each vertex also gets a local index (stored in the
    locInd array).
  */
  ierr    = MPI_Scan(&user.Nvlocal,&rstart,1,MPIU_INT,MPI_SUM,PETSC_COMM_WORLD);CHKERRQ(ierr);
  rstart -= user.Nvlocal;
  ierr    = PetscMalloc1(user.Nvlocal,&pordering);CHKERRQ(ierr);

  for (i=0; i < user.Nvlocal; i++) pordering[i] = rstart + i;

  /*
    Create the AO object
  */
  ierr = AOCreateBasic(MPI_COMM_WORLD,user.Nvlocal,user.gloInd,pordering,&ao);CHKERRQ(ierr);
  ierr = PetscFree(pordering);CHKERRQ(ierr);

  /*
    Keep the global indices for later use
  */
  ierr = PetscMalloc1(user.Nvlocal,&user.locInd);CHKERRQ(ierr);
  ierr = PetscMalloc1(Nvneighborstotal,&tmp);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Demonstrate the use of AO functionality
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = PetscFPrintf(PETSC_COMM_SELF,fptr1,"Before AOApplicationToPetsc, local indices are : \n");CHKERRQ(ierr);
  for (i=0; i < user.Nvlocal; i++) {
    ierr = PetscFPrintf(PETSC_COMM_SELF,fptr1," %D ",user.gloInd[i]);CHKERRQ(ierr);

    user.locInd[i] = user.gloInd[i];
  }
  ierr   = PetscFPrintf(PETSC_COMM_SELF,fptr1,"\n");CHKERRQ(ierr);
  jstart = 0;
  for (i=0; i < user.Nvlocal; i++) {
    ierr = PetscFPrintf(PETSC_COMM_SELF,fptr1,"Neghbors of local vertex %D are : ",user.gloInd[i]);CHKERRQ(ierr);
    for (j=0; j < user.itot[i]; j++) {
      ierr = PetscFPrintf(PETSC_COMM_SELF,fptr1,"%D ",user.AdjM[i][j]);CHKERRQ(ierr);

      tmp[j + jstart] = user.AdjM[i][j];
    }
    jstart += user.itot[i];
    ierr    = PetscFPrintf(PETSC_COMM_SELF,fptr1,"\n");CHKERRQ(ierr);
  }

  /*
    Now map the vlocal and neighbor lists to the PETSc ordering
  */
  ierr = AOApplicationToPetsc(ao,user.Nvlocal,user.locInd);CHKERRQ(ierr);
  ierr = AOApplicationToPetsc(ao,Nvneighborstotal,tmp);CHKERRQ(ierr);
  ierr = AODestroy(&ao);CHKERRQ(ierr);

  ierr = PetscFPrintf(PETSC_COMM_SELF,fptr1,"After AOApplicationToPetsc, local indices are : \n");CHKERRQ(ierr);
  for (i=0; i < user.Nvlocal; i++) {
    ierr = PetscFPrintf(PETSC_COMM_SELF,fptr1," %D ",user.locInd[i]);CHKERRQ(ierr);
  }
  ierr = PetscFPrintf(PETSC_COMM_SELF,fptr1,"\n");CHKERRQ(ierr);

  jstart = 0;
  for (i=0; i < user.Nvlocal; i++) {
    ierr = PetscFPrintf(PETSC_COMM_SELF,fptr1,"Neghbors of local vertex %D are : ",user.locInd[i]);CHKERRQ(ierr);
    for (j=0; j < user.itot[i]; j++) {
      user.AdjM[i][j] = tmp[j+jstart];

      ierr = PetscFPrintf(PETSC_COMM_SELF,fptr1,"%D ",user.AdjM[i][j]);CHKERRQ(ierr);
    }
    jstart += user.itot[i];
    ierr    = PetscFPrintf(PETSC_COMM_SELF,fptr1,"\n");CHKERRQ(ierr);
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Extract the ghost vertex information for each processor
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  /*
   Next, we need to generate a list of vertices required for this processor
   and a local numbering scheme for all vertices required on this processor.
      vertices - integer array of all vertices needed on this processor in PETSc
                 global numbering; this list consists of first the "locally owned"
                 vertices followed by the ghost vertices.
      verticesmask - integer array that for each global vertex lists its local
                     vertex number (in vertices) + 1. If the global vertex is not
                     represented on this processor, then the corresponding
                     entry in verticesmask is zero

      Note: vertices and verticesmask are both Nvglobal in length; this may
    sound terribly non-scalable, but in fact is not so bad for a reasonable
    number of processors. Importantly, it allows us to use NO SEARCHING
    in setting up the data structures.
  */
  ierr      = PetscMalloc1(user.Nvglobal,&vertices);CHKERRQ(ierr);
  ierr      = PetscCalloc1(user.Nvglobal,&verticesmask);CHKERRQ(ierr);
  nvertices = 0;

  /*
    First load "owned vertices" into list
  */
  for (i=0; i < user.Nvlocal; i++) {
    vertices[nvertices++]        = user.locInd[i];
    verticesmask[user.locInd[i]] = nvertices;
  }

  /*
    Now load ghost vertices into list
  */
  for (i=0; i < user.Nvlocal; i++) {
    for (j=0; j < user.itot[i]; j++) {
      nb = user.AdjM[i][j];
      if (!verticesmask[nb]) {
        vertices[nvertices++] = nb;
        verticesmask[nb]      = nvertices;
      }
    }
  }

  ierr = PetscFPrintf(PETSC_COMM_SELF,fptr1,"\n");CHKERRQ(ierr);
  ierr = PetscFPrintf(PETSC_COMM_SELF,fptr1,"The array vertices is :\n");CHKERRQ(ierr);
  for (i=0; i < nvertices; i++) {
    ierr = PetscFPrintf(PETSC_COMM_SELF,fptr1,"%D ",vertices[i]);CHKERRQ(ierr);
  }
  ierr = PetscFPrintf(PETSC_COMM_SELF,fptr1,"\n");CHKERRQ(ierr);

  /*
     Map the vertices listed in the neighbors to the local numbering from
    the global ordering that they contained initially.
  */
  ierr = PetscFPrintf(PETSC_COMM_SELF,fptr1,"\n");CHKERRQ(ierr);
  ierr = PetscFPrintf(PETSC_COMM_SELF,fptr1,"After mapping neighbors in the local contiguous ordering\n");CHKERRQ(ierr);
  for (i=0; i<user.Nvlocal; i++) {
    ierr = PetscFPrintf(PETSC_COMM_SELF,fptr1,"Neghbors of local vertex %D are :\n",i);CHKERRQ(ierr);
    for (j = 0; j < user.itot[i]; j++) {
      nb              = user.AdjM[i][j];
      user.AdjM[i][j] = verticesmask[nb] - 1;

      ierr = PetscFPrintf(PETSC_COMM_SELF,fptr1,"%D ",user.AdjM[i][j]);CHKERRQ(ierr);
    }
    ierr = PetscFPrintf(PETSC_COMM_SELF,fptr1,"\n");CHKERRQ(ierr);
  }

  N = user.Nvglobal;

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create vector and matrix data structures
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /*
    Create vector data structures
  */
  ierr = VecCreate(MPI_COMM_WORLD,&x);CHKERRQ(ierr);
  ierr = VecSetSizes(x,user.Nvlocal,N);CHKERRQ(ierr);
  ierr = VecSetFromOptions(x);CHKERRQ(ierr);
  ierr = VecDuplicate(x,&r);CHKERRQ(ierr);
  ierr = VecCreateSeq(MPI_COMM_SELF,bs*nvertices,&user.localX);CHKERRQ(ierr);
  ierr = VecDuplicate(user.localX,&user.localF);CHKERRQ(ierr);

  /*
    Create the scatter between the global representation and the
    local representation
  */
  ierr = ISCreateStride(MPI_COMM_SELF,bs*nvertices,0,1,&islocal);CHKERRQ(ierr);
  ierr = ISCreateBlock(MPI_COMM_SELF,bs,nvertices,vertices,PETSC_COPY_VALUES,&isglobal);CHKERRQ(ierr);
  ierr = VecScatterCreate(x,isglobal,user.localX,islocal,&user.scatter);CHKERRQ(ierr);
  ierr = ISDestroy(&isglobal);CHKERRQ(ierr);
  ierr = ISDestroy(&islocal);CHKERRQ(ierr);

  /*
     Create matrix data structure; Just to keep the example simple, we have not done any
     preallocation of memory for the matrix. In real application code with big matrices,
     preallocation should always be done to expedite the matrix creation.
  */
  ierr = MatCreate(MPI_COMM_WORLD,&Jac);CHKERRQ(ierr);
  ierr = MatSetSizes(Jac,PETSC_DECIDE,PETSC_DECIDE,N,N);CHKERRQ(ierr);
  ierr = MatSetFromOptions(Jac);CHKERRQ(ierr);
  ierr = MatSetUp(Jac);CHKERRQ(ierr);

  /*
    The following routine allows us to set the matrix values in local ordering
  */
  ierr = ISLocalToGlobalMappingCreate(MPI_COMM_SELF,bs,nvertices,vertices,PETSC_COPY_VALUES,&isl2g);CHKERRQ(ierr);
  ierr = MatSetLocalToGlobalMapping(Jac,isl2g,isl2g);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create nonlinear solver context
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = SNESCreate(MPI_COMM_WORLD,&snes);CHKERRQ(ierr);
  ierr = SNESSetType(snes,type);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Set routines for function and Jacobian evaluation
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = SNESSetFunction(snes,r,FormFunction,(void*)&user);CHKERRQ(ierr);

  ierr = PetscOptionsGetBool(NULL,NULL,"-fd_jacobian_coloring",&fd_jacobian_coloring,0);CHKERRQ(ierr);
  if (!fd_jacobian_coloring) {
    ierr = SNESSetJacobian(snes,Jac,Jac,FormJacobian,(void*)&user);CHKERRQ(ierr);
  } else {  /* Use matfdcoloring */
    ISColoring   iscoloring;
    MatColoring  mc;

    /* Get the data structure of Jac */
    ierr = FormJacobian(snes,x,Jac,Jac,&user);CHKERRQ(ierr);
    /* Create coloring context */
    ierr = MatColoringCreate(Jac,&mc);CHKERRQ(ierr);
    ierr = MatColoringSetType(mc,MATCOLORINGSL);CHKERRQ(ierr);
    ierr = MatColoringSetFromOptions(mc);CHKERRQ(ierr);
    ierr = MatColoringApply(mc,&iscoloring);CHKERRQ(ierr);
    ierr = MatColoringDestroy(&mc);CHKERRQ(ierr);
    ierr = MatFDColoringCreate(Jac,iscoloring,&matfdcoloring);CHKERRQ(ierr);
    ierr = MatFDColoringSetFunction(matfdcoloring,(PetscErrorCode (*)(void))FormFunction,&user);CHKERRQ(ierr);
    ierr = MatFDColoringSetFromOptions(matfdcoloring);CHKERRQ(ierr);
    ierr = MatFDColoringSetUp(Jac,iscoloring,matfdcoloring);CHKERRQ(ierr);
    /* ierr = MatFDColoringView(matfdcoloring,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr); */
    ierr = SNESSetJacobian(snes,Jac,Jac,SNESComputeJacobianDefaultColor,matfdcoloring);CHKERRQ(ierr);
    ierr = ISColoringDestroy(&iscoloring);CHKERRQ(ierr);
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Customize nonlinear solver; set runtime options
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = SNESSetFromOptions(snes);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Evaluate initial guess; then solve nonlinear system
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /*
     Note: The user should initialize the vector, x, with the initial guess
     for the nonlinear solver prior to calling SNESSolve().  In particular,
     to employ an initial guess of zero, the user should explicitly set
     this vector to zero by calling VecSet().
  */
  ierr = FormInitialGuess(&user,x);CHKERRQ(ierr);

  /*
    Print the initial guess
  */
  ierr = VecGetArray(x,&xx);CHKERRQ(ierr);
  for (inode = 0; inode < user.Nvlocal; inode++) {
    ierr = PetscFPrintf(PETSC_COMM_SELF,fptr1,"Initial Solution at node %D is %f \n",inode,xx[inode]);CHKERRQ(ierr);
  }
  ierr = VecRestoreArray(x,&xx);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Now solve the nonlinear system
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = SNESSolve(snes,NULL,x);CHKERRQ(ierr);
  ierr = SNESGetIterationNumber(snes,&its);CHKERRQ(ierr);
  ierr = SNESGetNonlinearStepFailures(snes,&nfails);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Print the output : solution vector and other information
     Each processor writes to the file output.<rank> where rank is the
     processor's rank.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = VecGetArray(x,&xx);CHKERRQ(ierr);
  for (inode = 0; inode < user.Nvlocal; inode++) {
    ierr = PetscFPrintf(PETSC_COMM_SELF,fptr1,"Solution at node %D is %f \n",inode,xx[inode]);CHKERRQ(ierr);
  }
  ierr = VecRestoreArray(x,&xx);CHKERRQ(ierr);
  fclose(fptr1);
  ierr = PetscPrintf(MPI_COMM_WORLD,"number of SNES iterations = %D, ",its);CHKERRQ(ierr);
  ierr = PetscPrintf(MPI_COMM_WORLD,"number of unsuccessful steps = %D\n",nfails);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Free work space.  All PETSc objects should be destroyed when they
     are no longer needed.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = PetscFree(user.gloInd);CHKERRQ(ierr);
  ierr = PetscFree(user.locInd);CHKERRQ(ierr);
  ierr = PetscFree(vertices);CHKERRQ(ierr);
  ierr = PetscFree(verticesmask);CHKERRQ(ierr);
  ierr = PetscFree(tmp);CHKERRQ(ierr);
  ierr = VecScatterDestroy(&user.scatter);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingDestroy(&isl2g);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&r);CHKERRQ(ierr);
  ierr = VecDestroy(&user.localX);CHKERRQ(ierr);
  ierr = VecDestroy(&user.localF);CHKERRQ(ierr);
  ierr = MatDestroy(&Jac);CHKERRQ(ierr);  ierr = SNESDestroy(&snes);CHKERRQ(ierr);
  /*ierr = PetscDrawDestroy(draw);CHKERRQ(ierr);*/
  if (fd_jacobian_coloring) {
    ierr = MatFDColoringDestroy(&matfdcoloring);CHKERRQ(ierr);
  }
  ierr = PetscFinalize();
  return ierr;
}
/* --------------------  Form initial approximation ----------------- */

/*
   FormInitialGuess - Forms initial approximation.

   Input Parameters:
   user - user-defined application context
   X - vector

   Output Parameter:
   X - vector
 */
PetscErrorCode FormInitialGuess(AppCtx *user,Vec X)
{
  PetscInt    i,Nvlocal,ierr;
  PetscInt    *gloInd;
  PetscScalar *x;
#if defined(UNUSED_VARIABLES)
  PetscReal temp1,temp,hx,hy,hxdhy,hydhx,sc;
  PetscInt  Neglobal,Nvglobal,j,row;
  PetscReal alpha,lambda;

  Nvglobal = user->Nvglobal;
  Neglobal = user->Neglobal;
  lambda   = user->non_lin_param;
  alpha    = user->lin_param;
#endif

  Nvlocal = user->Nvlocal;
  gloInd  = user->gloInd;

  /*
     Get a pointer to vector data.
       - For default PETSc vectors, VecGetArray() returns a pointer to
         the data array.  Otherwise, the routine is implementation dependent.
       - You MUST call VecRestoreArray() when you no longer need access to
         the array.
  */
  ierr = VecGetArray(X,&x);CHKERRQ(ierr);

  /*
     Compute initial guess over the locally owned part of the grid
  */
  for (i=0; i < Nvlocal; i++) x[i] = (PetscReal)gloInd[i];

  /*
     Restore vector
  */
  ierr = VecRestoreArray(X,&x);CHKERRQ(ierr);
  return 0;
}
/* --------------------  Evaluate Function F(x) --------------------- */
/*
   FormFunction - Evaluates nonlinear function, F(x).

   Input Parameters:
.  snes - the SNES context
.  X - input vector
.  ptr - optional user-defined context, as set by SNESSetFunction()

   Output Parameter:
.  F - function vector
 */
PetscErrorCode FormFunction(SNES snes,Vec X,Vec F,void *ptr)
{
  AppCtx         *user = (AppCtx*)ptr;
  PetscErrorCode ierr;
  PetscInt       i,j,Nvlocal;
  PetscReal      alpha,lambda;
  PetscScalar    *x,*f;
  VecScatter     scatter;
  Vec            localX = user->localX;
#if defined(UNUSED_VARIABLES)
  PetscScalar ut,ub,ul,ur,u,*g,sc,uyy,uxx;
  PetscReal   hx,hy,hxdhy,hydhx;
  PetscReal   two = 2.0,one = 1.0;
  PetscInt    Nvglobal,Neglobal,row;
  PetscInt    *gloInd;

  Nvglobal = user->Nvglobal;
  Neglobal = user->Neglobal;
  gloInd   = user->gloInd;
#endif

  Nvlocal = user->Nvlocal;
  lambda  = user->non_lin_param;
  alpha   = user->lin_param;
  scatter = user->scatter;

  /*
     PDE : L(u) + lambda*u*u +alpha*u = 0 where L(u) is the approximate Laplacian as
     described in the beginning of this code

     First scatter the distributed vector X into local vector localX (that includes
     values for ghost nodes. If we wish,we can put some other work between
     VecScatterBegin() and VecScatterEnd() to overlap the communication with
     computation.
 */
  ierr = VecScatterBegin(scatter,X,localX,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(scatter,X,localX,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);

  /*
     Get pointers to vector data
  */
  ierr = VecGetArray(localX,&x);CHKERRQ(ierr);
  ierr = VecGetArray(F,&f);CHKERRQ(ierr);

  /*
    Now compute the f(x). As mentioned earlier, the computed Laplacian is just an
    approximate one chosen for illustrative purpose only. Another point to notice
    is that this is a local (completly parallel) calculation. In practical application
    codes, function calculation time is a dominat portion of the overall execution time.
  */
  for (i=0; i < Nvlocal; i++) {
    f[i] = (user->itot[i] - alpha)*x[i] - lambda*x[i]*x[i];
    for (j = 0; j < user->itot[i]; j++) f[i] -= x[user->AdjM[i][j]];
  }

  /*
     Restore vectors
  */
  ierr = VecRestoreArray(localX,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(F,&f);CHKERRQ(ierr);
  /*ierr = VecView(F,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);*/

  return 0;
}

/* --------------------  Evaluate Jacobian F'(x) -------------------- */
/*
   FormJacobian - Evaluates Jacobian matrix.

   Input Parameters:
.  snes - the SNES context
.  X - input vector
.  ptr - optional user-defined context, as set by SNESSetJacobian()

   Output Parameters:
.  A - Jacobian matrix
.  B - optionally different preconditioning matrix
.  flag - flag indicating matrix structure

*/
PetscErrorCode FormJacobian(SNES snes,Vec X,Mat J,Mat jac,void *ptr)
{
  AppCtx      *user = (AppCtx*)ptr;
  PetscInt    i,j,Nvlocal,col[50],ierr;
  PetscScalar alpha,lambda,value[50];
  Vec         localX = user->localX;
  VecScatter  scatter;
  PetscScalar *x;
#if defined(UNUSED_VARIABLES)
  PetscScalar two = 2.0,one = 1.0;
  PetscInt    row,Nvglobal,Neglobal;
  PetscInt    *gloInd;

  Nvglobal = user->Nvglobal;
  Neglobal = user->Neglobal;
  gloInd   = user->gloInd;
#endif

  /*printf("Entering into FormJacobian \n");*/
  Nvlocal = user->Nvlocal;
  lambda  = user->non_lin_param;
  alpha   =  user->lin_param;
  scatter = user->scatter;

  /*
     PDE : L(u) + lambda*u*u +alpha*u = 0 where L(u) is the approximate Laplacian as
     described in the beginning of this code

     First scatter the distributed vector X into local vector localX (that includes
     values for ghost nodes. If we wish, we can put some other work between
     VecScatterBegin() and VecScatterEnd() to overlap the communication with
     computation.
  */
  ierr = VecScatterBegin(scatter,X,localX,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(scatter,X,localX,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);

  /*
     Get pointer to vector data
  */
  ierr = VecGetArray(localX,&x);CHKERRQ(ierr);

  for (i=0; i < Nvlocal; i++) {
    col[0]   = i;
    value[0] = user->itot[i] - 2.0*lambda*x[i] - alpha;
    for (j = 0; j < user->itot[i]; j++) {
      col[j+1]   = user->AdjM[i][j];
      value[j+1] = -1.0;
    }

    /*
      Set the matrix values in the local ordering. Note that in order to use this
      feature we must call the routine MatSetLocalToGlobalMapping() after the
      matrix has been created.
    */
    ierr = MatSetValuesLocal(jac,1,&i,1+user->itot[i],col,value,INSERT_VALUES);CHKERRQ(ierr);
  }

  /*
     Assemble matrix, using the 2-step process:
       MatAssemblyBegin(), MatAssemblyEnd().
     Between these two calls, the pointer to vector data has been restored to
     demonstrate the use of overlapping communicationn with computation.
  */
  ierr = MatAssemblyBegin(jac,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = VecRestoreArray(localX,&x);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(jac,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  /*
     Tell the matrix we will never add a new nonzero location to the
     matrix. If we do, it will generate an error.
  */
  ierr = MatSetOption(jac,MAT_NEW_NONZERO_LOCATION_ERR,PETSC_TRUE);CHKERRQ(ierr);
  /* MatView(jac,PETSC_VIEWER_STDOUT_SELF); */
  return 0;
}



/*TEST

   build:
      requires: !complex

   test:
      nsize: 2
      args: -snes_monitor_short
      localrunfiles: options.inf adj.in

   test:
      suffix: 2
      nsize: 2
      args: -snes_monitor_short -fd_jacobian_coloring
      localrunfiles: options.inf adj.in

TEST*/
