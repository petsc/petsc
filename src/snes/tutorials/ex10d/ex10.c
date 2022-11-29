
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
petsc/src/snes/tutorials/ex5.c).\n\
This example also illustrates the use of parallel matrix coloring.\n\
The command line options include:\n\
  -vert <Nv>, where Nv is the global number of nodes\n\
  -elem <Ne>, where Ne is the global number of elements\n\
  -nl_par <lambda>, where lambda is the multiplier for the non linear term (u*u) term\n\
  -lin_par <alpha>, where alpha is the multiplier for the linear term (u)\n\
  -fd_jacobian_coloring -mat_coloring_type lf\n";

/* ------------------------------------------------------------------------

   PDE Solved : L(u) + lambda*u*u + alpha*u = 0 where L(u) is the Laplacian.

   The Laplacian is approximated in the following way: each edge is given a weight
   of one meaning that the diagonal term will have the weight equal to the degree
   of a node. The off diagonal terms will get a weight of -1.

   -----------------------------------------------------------------------*/

#define MAX_ELEM      500 /* Maximum number of elements */
#define MAX_VERT      100 /* Maximum number of vertices */
#define MAX_VERT_ELEM 3   /* Vertices per element       */

/*
  Application-defined context for problem specific data
*/
typedef struct {
  PetscInt   Nvglobal, Nvlocal;            /* global and local number of vertices */
  PetscInt   Neglobal, Nelocal;            /* global and local number of vertices */
  PetscInt   AdjM[MAX_VERT][50];           /* adjacency list of a vertex */
  PetscInt   itot[MAX_VERT];               /* total number of neighbors for a vertex */
  PetscInt   icv[MAX_ELEM][MAX_VERT_ELEM]; /* vertices belonging to an element */
  PetscInt   v2p[MAX_VERT];                /* processor number for a vertex */
  PetscInt  *locInd, *gloInd;              /* local and global orderings for a node */
  Vec        localX, localF;               /* local solution (u) and f(u) vectors */
  PetscReal  non_lin_param;                /* nonlinear parameter for the PDE */
  PetscReal  lin_param;                    /* linear parameter for the PDE */
  VecScatter scatter;                      /* scatter context for the local and
                                               distributed vectors */
} AppCtx;

/*
  User-defined routines
*/
PetscErrorCode FormJacobian(SNES, Vec, Mat, Mat, void *);
PetscErrorCode FormFunction(SNES, Vec, Vec, void *);
PetscErrorCode FormInitialGuess(AppCtx *, Vec);

int main(int argc, char **argv)
{
  SNES                   snes;                /* SNES context */
  SNESType               type = SNESNEWTONLS; /* default nonlinear solution method */
  Vec                    x, r;                /* solution, residual vectors */
  Mat                    Jac;                 /* Jacobian matrix */
  AppCtx                 user;                /* user-defined application context */
  AO                     ao;                  /* Application Ordering object */
  IS                     isglobal, islocal;   /* global and local index sets */
  PetscMPIInt            rank, size;          /* rank of a process, number of processors */
  PetscInt               rstart;              /* starting index of PETSc ordering for a processor */
  PetscInt               nfails;              /* number of unsuccessful Newton steps */
  PetscInt               bs = 1;              /* block size for multicomponent systems */
  PetscInt               nvertices;           /* number of local plus ghost nodes of a processor */
  PetscInt              *pordering;           /* PETSc ordering */
  PetscInt              *vertices;            /* list of all vertices (incl. ghost ones) on a processor */
  PetscInt              *verticesmask;
  PetscInt              *tmp;
  PetscInt               i, j, jstart, inode, nb, nbrs, Nvneighborstotal = 0;
  PetscInt               its, N;
  PetscScalar           *xx;
  char                   str[256], form[256], part_name[256];
  FILE                  *fptr, *fptr1;
  ISLocalToGlobalMapping isl2g;
  int                    dtmp;
#if defined(UNUSED_VARIABLES)
  PetscDraw    draw; /* drawing context */
  PetscScalar *ff, *gg;
  PetscReal    tiny = 1.0e-10, zero = 0.0, one = 1.0, big = 1.0e+10;
  PetscInt    *tmp1, *tmp2;
#endif
  MatFDColoring matfdcoloring        = 0;
  PetscBool     fd_jacobian_coloring = PETSC_FALSE;

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Initialize program
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, "options.inf", help));
  PetscCallMPI(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
  PetscCallMPI(MPI_Comm_size(MPI_COMM_WORLD, &size));

  /* The current input file options.inf is for 2 proc run only */
  PetscCheck(size == 2, PETSC_COMM_WORLD, PETSC_ERR_WRONG_MPI_SIZE, "This example currently runs on 2 procs only.");

  /*
     Initialize problem parameters
  */
  user.Nvglobal = 16; /* Global # of vertices  */
  user.Neglobal = 18; /* Global # of elements  */

  PetscCall(PetscOptionsGetInt(NULL, NULL, "-vert", &user.Nvglobal, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-elem", &user.Neglobal, NULL));

  user.non_lin_param = 0.06;

  PetscCall(PetscOptionsGetReal(NULL, NULL, "-nl_par", &user.non_lin_param, NULL));

  user.lin_param = -1.0;

  PetscCall(PetscOptionsGetReal(NULL, NULL, "-lin_par", &user.lin_param, NULL));

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
  fptr = fopen("adj.in", "r");
  PetscCheck(fptr, PETSC_COMM_SELF, PETSC_ERR_FILE_OPEN, "Could not open adj.in");

  /*
     Each processor writes to the file output.<rank> where rank is the
     processor's rank.
  */
  sprintf(part_name, "output.%d", rank);
  fptr1 = fopen(part_name, "w");
  PetscCheck(fptr1, PETSC_COMM_SELF, PETSC_ERR_FILE_OPEN, "Could no open output file");
  PetscCall(PetscMalloc1(user.Nvglobal, &user.gloInd));
  PetscCall(PetscFPrintf(PETSC_COMM_SELF, fptr1, "Rank is %d\n", rank));
  for (inode = 0; inode < user.Nvglobal; inode++) {
    PetscCheck(fgets(str, 256, fptr), PETSC_COMM_SELF, PETSC_ERR_FILE_READ, "fgets read failed");
    sscanf(str, "%d", &dtmp);
    user.v2p[inode] = dtmp;
    if (user.v2p[inode] == rank) {
      PetscCall(PetscFPrintf(PETSC_COMM_SELF, fptr1, "Node %" PetscInt_FMT " belongs to processor %" PetscInt_FMT "\n", inode, user.v2p[inode]));

      user.gloInd[user.Nvlocal] = inode;
      sscanf(str, "%*d %d", &dtmp);
      nbrs = dtmp;
      PetscCall(PetscFPrintf(PETSC_COMM_SELF, fptr1, "Number of neighbors for the vertex %" PetscInt_FMT " is %" PetscInt_FMT "\n", inode, nbrs));

      user.itot[user.Nvlocal] = nbrs;
      Nvneighborstotal += nbrs;
      for (i = 0; i < user.itot[user.Nvlocal]; i++) {
        form[0] = '\0';
        for (j = 0; j < i + 2; j++) PetscCall(PetscStrlcat(form, "%*d ", sizeof(form)));
        PetscCall(PetscStrlcat(form, "%d", sizeof(form)));

        sscanf(str, form, &dtmp);
        user.AdjM[user.Nvlocal][i] = dtmp;

        PetscCall(PetscFPrintf(PETSC_COMM_SELF, fptr1, "%" PetscInt_FMT " ", user.AdjM[user.Nvlocal][i]));
      }
      PetscCall(PetscFPrintf(PETSC_COMM_SELF, fptr1, "\n"));
      user.Nvlocal++;
    }
  }
  PetscCall(PetscFPrintf(PETSC_COMM_SELF, fptr1, "Total # of Local Vertices is %" PetscInt_FMT " \n", user.Nvlocal));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create different orderings
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /*
    Create the local ordering list for vertices. First a list using the PETSc global
    ordering is created. Then we use the AO object to get the PETSc-to-application and
    application-to-PETSc mappings. Each vertex also gets a local index (stored in the
    locInd array).
  */
  PetscCallMPI(MPI_Scan(&user.Nvlocal, &rstart, 1, MPIU_INT, MPI_SUM, PETSC_COMM_WORLD));
  rstart -= user.Nvlocal;
  PetscCall(PetscMalloc1(user.Nvlocal, &pordering));

  for (i = 0; i < user.Nvlocal; i++) pordering[i] = rstart + i;

  /*
    Create the AO object
  */
  PetscCall(AOCreateBasic(MPI_COMM_WORLD, user.Nvlocal, user.gloInd, pordering, &ao));
  PetscCall(PetscFree(pordering));

  /*
    Keep the global indices for later use
  */
  PetscCall(PetscMalloc1(user.Nvlocal, &user.locInd));
  PetscCall(PetscMalloc1(Nvneighborstotal, &tmp));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Demonstrate the use of AO functionality
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  PetscCall(PetscFPrintf(PETSC_COMM_SELF, fptr1, "Before AOApplicationToPetsc, local indices are : \n"));
  for (i = 0; i < user.Nvlocal; i++) {
    PetscCall(PetscFPrintf(PETSC_COMM_SELF, fptr1, " %" PetscInt_FMT " ", user.gloInd[i]));

    user.locInd[i] = user.gloInd[i];
  }
  PetscCall(PetscFPrintf(PETSC_COMM_SELF, fptr1, "\n"));
  jstart = 0;
  for (i = 0; i < user.Nvlocal; i++) {
    PetscCall(PetscFPrintf(PETSC_COMM_SELF, fptr1, "Neghbors of local vertex %" PetscInt_FMT " are : ", user.gloInd[i]));
    for (j = 0; j < user.itot[i]; j++) {
      PetscCall(PetscFPrintf(PETSC_COMM_SELF, fptr1, "%" PetscInt_FMT " ", user.AdjM[i][j]));

      tmp[j + jstart] = user.AdjM[i][j];
    }
    jstart += user.itot[i];
    PetscCall(PetscFPrintf(PETSC_COMM_SELF, fptr1, "\n"));
  }

  /*
    Now map the vlocal and neighbor lists to the PETSc ordering
  */
  PetscCall(AOApplicationToPetsc(ao, user.Nvlocal, user.locInd));
  PetscCall(AOApplicationToPetsc(ao, Nvneighborstotal, tmp));
  PetscCall(AODestroy(&ao));

  PetscCall(PetscFPrintf(PETSC_COMM_SELF, fptr1, "After AOApplicationToPetsc, local indices are : \n"));
  for (i = 0; i < user.Nvlocal; i++) PetscCall(PetscFPrintf(PETSC_COMM_SELF, fptr1, " %" PetscInt_FMT " ", user.locInd[i]));
  PetscCall(PetscFPrintf(PETSC_COMM_SELF, fptr1, "\n"));

  jstart = 0;
  for (i = 0; i < user.Nvlocal; i++) {
    PetscCall(PetscFPrintf(PETSC_COMM_SELF, fptr1, "Neghbors of local vertex %" PetscInt_FMT " are : ", user.locInd[i]));
    for (j = 0; j < user.itot[i]; j++) {
      user.AdjM[i][j] = tmp[j + jstart];

      PetscCall(PetscFPrintf(PETSC_COMM_SELF, fptr1, "%" PetscInt_FMT " ", user.AdjM[i][j]));
    }
    jstart += user.itot[i];
    PetscCall(PetscFPrintf(PETSC_COMM_SELF, fptr1, "\n"));
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
  PetscCall(PetscMalloc1(user.Nvglobal, &vertices));
  PetscCall(PetscCalloc1(user.Nvglobal, &verticesmask));
  nvertices = 0;

  /*
    First load "owned vertices" into list
  */
  for (i = 0; i < user.Nvlocal; i++) {
    vertices[nvertices++]        = user.locInd[i];
    verticesmask[user.locInd[i]] = nvertices;
  }

  /*
    Now load ghost vertices into list
  */
  for (i = 0; i < user.Nvlocal; i++) {
    for (j = 0; j < user.itot[i]; j++) {
      nb = user.AdjM[i][j];
      if (!verticesmask[nb]) {
        vertices[nvertices++] = nb;
        verticesmask[nb]      = nvertices;
      }
    }
  }

  PetscCall(PetscFPrintf(PETSC_COMM_SELF, fptr1, "\n"));
  PetscCall(PetscFPrintf(PETSC_COMM_SELF, fptr1, "The array vertices is :\n"));
  for (i = 0; i < nvertices; i++) PetscCall(PetscFPrintf(PETSC_COMM_SELF, fptr1, "%" PetscInt_FMT " ", vertices[i]));
  PetscCall(PetscFPrintf(PETSC_COMM_SELF, fptr1, "\n"));

  /*
     Map the vertices listed in the neighbors to the local numbering from
    the global ordering that they contained initially.
  */
  PetscCall(PetscFPrintf(PETSC_COMM_SELF, fptr1, "\n"));
  PetscCall(PetscFPrintf(PETSC_COMM_SELF, fptr1, "After mapping neighbors in the local contiguous ordering\n"));
  for (i = 0; i < user.Nvlocal; i++) {
    PetscCall(PetscFPrintf(PETSC_COMM_SELF, fptr1, "Neghbors of local vertex %" PetscInt_FMT " are :\n", i));
    for (j = 0; j < user.itot[i]; j++) {
      nb              = user.AdjM[i][j];
      user.AdjM[i][j] = verticesmask[nb] - 1;

      PetscCall(PetscFPrintf(PETSC_COMM_SELF, fptr1, "%" PetscInt_FMT " ", user.AdjM[i][j]));
    }
    PetscCall(PetscFPrintf(PETSC_COMM_SELF, fptr1, "\n"));
  }

  N = user.Nvglobal;

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create vector and matrix data structures
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /*
    Create vector data structures
  */
  PetscCall(VecCreate(MPI_COMM_WORLD, &x));
  PetscCall(VecSetSizes(x, user.Nvlocal, N));
  PetscCall(VecSetFromOptions(x));
  PetscCall(VecDuplicate(x, &r));
  PetscCall(VecCreateSeq(MPI_COMM_SELF, bs * nvertices, &user.localX));
  PetscCall(VecDuplicate(user.localX, &user.localF));

  /*
    Create the scatter between the global representation and the
    local representation
  */
  PetscCall(ISCreateStride(MPI_COMM_SELF, bs * nvertices, 0, 1, &islocal));
  PetscCall(ISCreateBlock(MPI_COMM_SELF, bs, nvertices, vertices, PETSC_COPY_VALUES, &isglobal));
  PetscCall(VecScatterCreate(x, isglobal, user.localX, islocal, &user.scatter));
  PetscCall(ISDestroy(&isglobal));
  PetscCall(ISDestroy(&islocal));

  /*
     Create matrix data structure; Just to keep the example simple, we have not done any
     preallocation of memory for the matrix. In real application code with big matrices,
     preallocation should always be done to expedite the matrix creation.
  */
  PetscCall(MatCreate(MPI_COMM_WORLD, &Jac));
  PetscCall(MatSetSizes(Jac, PETSC_DECIDE, PETSC_DECIDE, N, N));
  PetscCall(MatSetFromOptions(Jac));
  PetscCall(MatSetUp(Jac));

  /*
    The following routine allows us to set the matrix values in local ordering
  */
  PetscCall(ISLocalToGlobalMappingCreate(MPI_COMM_SELF, bs, nvertices, vertices, PETSC_COPY_VALUES, &isl2g));
  PetscCall(MatSetLocalToGlobalMapping(Jac, isl2g, isl2g));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create nonlinear solver context
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  PetscCall(SNESCreate(MPI_COMM_WORLD, &snes));
  PetscCall(SNESSetType(snes, type));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Set routines for function and Jacobian evaluation
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(SNESSetFunction(snes, r, FormFunction, (void *)&user));

  PetscCall(PetscOptionsGetBool(NULL, NULL, "-fd_jacobian_coloring", &fd_jacobian_coloring, 0));
  if (!fd_jacobian_coloring) {
    PetscCall(SNESSetJacobian(snes, Jac, Jac, FormJacobian, (void *)&user));
  } else { /* Use matfdcoloring */
    ISColoring  iscoloring;
    MatColoring mc;

    /* Get the data structure of Jac */
    PetscCall(FormJacobian(snes, x, Jac, Jac, &user));
    /* Create coloring context */
    PetscCall(MatColoringCreate(Jac, &mc));
    PetscCall(MatColoringSetType(mc, MATCOLORINGSL));
    PetscCall(MatColoringSetFromOptions(mc));
    PetscCall(MatColoringApply(mc, &iscoloring));
    PetscCall(MatColoringDestroy(&mc));
    PetscCall(MatFDColoringCreate(Jac, iscoloring, &matfdcoloring));
    PetscCall(MatFDColoringSetFunction(matfdcoloring, (PetscErrorCode(*)(void))FormFunction, &user));
    PetscCall(MatFDColoringSetFromOptions(matfdcoloring));
    PetscCall(MatFDColoringSetUp(Jac, iscoloring, matfdcoloring));
    /* PetscCall(MatFDColoringView(matfdcoloring,PETSC_VIEWER_STDOUT_WORLD)); */
    PetscCall(SNESSetJacobian(snes, Jac, Jac, SNESComputeJacobianDefaultColor, matfdcoloring));
    PetscCall(ISColoringDestroy(&iscoloring));
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Customize nonlinear solver; set runtime options
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  PetscCall(SNESSetFromOptions(snes));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Evaluate initial guess; then solve nonlinear system
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /*
     Note: The user should initialize the vector, x, with the initial guess
     for the nonlinear solver prior to calling SNESSolve().  In particular,
     to employ an initial guess of zero, the user should explicitly set
     this vector to zero by calling VecSet().
  */
  PetscCall(FormInitialGuess(&user, x));

  /*
    Print the initial guess
  */
  PetscCall(VecGetArray(x, &xx));
  for (inode = 0; inode < user.Nvlocal; inode++) PetscCall(PetscFPrintf(PETSC_COMM_SELF, fptr1, "Initial Solution at node %" PetscInt_FMT " is %f \n", inode, (double)PetscRealPart(xx[inode])));
  PetscCall(VecRestoreArray(x, &xx));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Now solve the nonlinear system
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  PetscCall(SNESSolve(snes, NULL, x));
  PetscCall(SNESGetIterationNumber(snes, &its));
  PetscCall(SNESGetNonlinearStepFailures(snes, &nfails));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Print the output : solution vector and other information
     Each processor writes to the file output.<rank> where rank is the
     processor's rank.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  PetscCall(VecGetArray(x, &xx));
  for (inode = 0; inode < user.Nvlocal; inode++) PetscCall(PetscFPrintf(PETSC_COMM_SELF, fptr1, "Solution at node %" PetscInt_FMT " is %f \n", inode, (double)PetscRealPart(xx[inode])));
  PetscCall(VecRestoreArray(x, &xx));
  fclose(fptr1);
  PetscCall(PetscPrintf(MPI_COMM_WORLD, "number of SNES iterations = %" PetscInt_FMT ", ", its));
  PetscCall(PetscPrintf(MPI_COMM_WORLD, "number of unsuccessful steps = %" PetscInt_FMT "\n", nfails));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Free work space.  All PETSc objects should be destroyed when they
     are no longer needed.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(PetscFree(user.gloInd));
  PetscCall(PetscFree(user.locInd));
  PetscCall(PetscFree(vertices));
  PetscCall(PetscFree(verticesmask));
  PetscCall(PetscFree(tmp));
  PetscCall(VecScatterDestroy(&user.scatter));
  PetscCall(ISLocalToGlobalMappingDestroy(&isl2g));
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&r));
  PetscCall(VecDestroy(&user.localX));
  PetscCall(VecDestroy(&user.localF));
  PetscCall(SNESDestroy(&snes));
  PetscCall(MatDestroy(&Jac));
  /* PetscCall(PetscDrawDestroy(draw));*/
  if (fd_jacobian_coloring) PetscCall(MatFDColoringDestroy(&matfdcoloring));
  PetscCall(PetscFinalize());
  return 0;
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
PetscErrorCode FormInitialGuess(AppCtx *user, Vec X)
{
  PetscInt     i, Nvlocal;
  PetscInt    *gloInd;
  PetscScalar *x;
#if defined(UNUSED_VARIABLES)
  PetscReal temp1, temp, hx, hy, hxdhy, hydhx, sc;
  PetscInt  Neglobal, Nvglobal, j, row;
  PetscReal alpha, lambda;

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
  PetscCall(VecGetArray(X, &x));

  /*
     Compute initial guess over the locally owned part of the grid
  */
  for (i = 0; i < Nvlocal; i++) x[i] = (PetscReal)gloInd[i];

  /*
     Restore vector
  */
  PetscCall(VecRestoreArray(X, &x));
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
PetscErrorCode FormFunction(SNES snes, Vec X, Vec F, void *ptr)
{
  AppCtx      *user = (AppCtx *)ptr;
  PetscInt     i, j, Nvlocal;
  PetscReal    alpha, lambda;
  PetscScalar *x, *f;
  VecScatter   scatter;
  Vec          localX = user->localX;
#if defined(UNUSED_VARIABLES)
  PetscScalar ut, ub, ul, ur, u, *g, sc, uyy, uxx;
  PetscReal   hx, hy, hxdhy, hydhx;
  PetscReal   two = 2.0, one = 1.0;
  PetscInt    Nvglobal, Neglobal, row;
  PetscInt   *gloInd;

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
  PetscCall(VecScatterBegin(scatter, X, localX, INSERT_VALUES, SCATTER_FORWARD));
  PetscCall(VecScatterEnd(scatter, X, localX, INSERT_VALUES, SCATTER_FORWARD));

  /*
     Get pointers to vector data
  */
  PetscCall(VecGetArray(localX, &x));
  PetscCall(VecGetArray(F, &f));

  /*
    Now compute the f(x). As mentioned earlier, the computed Laplacian is just an
    approximate one chosen for illustrative purpose only. Another point to notice
    is that this is a local (completly parallel) calculation. In practical application
    codes, function calculation time is a dominat portion of the overall execution time.
  */
  for (i = 0; i < Nvlocal; i++) {
    f[i] = (user->itot[i] - alpha) * x[i] - lambda * x[i] * x[i];
    for (j = 0; j < user->itot[i]; j++) f[i] -= x[user->AdjM[i][j]];
  }

  /*
     Restore vectors
  */
  PetscCall(VecRestoreArray(localX, &x));
  PetscCall(VecRestoreArray(F, &f));
  /* PetscCall(VecView(F,PETSC_VIEWER_STDOUT_WORLD)); */

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
PetscErrorCode FormJacobian(SNES snes, Vec X, Mat J, Mat jac, void *ptr)
{
  AppCtx      *user = (AppCtx *)ptr;
  PetscInt     i, j, Nvlocal, col[50];
  PetscScalar  alpha, lambda, value[50];
  Vec          localX = user->localX;
  VecScatter   scatter;
  PetscScalar *x;
#if defined(UNUSED_VARIABLES)
  PetscScalar two = 2.0, one = 1.0;
  PetscInt    row, Nvglobal, Neglobal;
  PetscInt   *gloInd;

  Nvglobal = user->Nvglobal;
  Neglobal = user->Neglobal;
  gloInd   = user->gloInd;
#endif

  /*printf("Entering into FormJacobian \n");*/
  Nvlocal = user->Nvlocal;
  lambda  = user->non_lin_param;
  alpha   = user->lin_param;
  scatter = user->scatter;

  /*
     PDE : L(u) + lambda*u*u +alpha*u = 0 where L(u) is the approximate Laplacian as
     described in the beginning of this code

     First scatter the distributed vector X into local vector localX (that includes
     values for ghost nodes. If we wish, we can put some other work between
     VecScatterBegin() and VecScatterEnd() to overlap the communication with
     computation.
  */
  PetscCall(VecScatterBegin(scatter, X, localX, INSERT_VALUES, SCATTER_FORWARD));
  PetscCall(VecScatterEnd(scatter, X, localX, INSERT_VALUES, SCATTER_FORWARD));

  /*
     Get pointer to vector data
  */
  PetscCall(VecGetArray(localX, &x));

  for (i = 0; i < Nvlocal; i++) {
    col[0]   = i;
    value[0] = user->itot[i] - 2.0 * lambda * x[i] - alpha;
    for (j = 0; j < user->itot[i]; j++) {
      col[j + 1]   = user->AdjM[i][j];
      value[j + 1] = -1.0;
    }

    /*
      Set the matrix values in the local ordering. Note that in order to use this
      feature we must call the routine MatSetLocalToGlobalMapping() after the
      matrix has been created.
    */
    PetscCall(MatSetValuesLocal(jac, 1, &i, 1 + user->itot[i], col, value, INSERT_VALUES));
  }

  /*
     Assemble matrix, using the 2-step process:
       MatAssemblyBegin(), MatAssemblyEnd().
     Between these two calls, the pointer to vector data has been restored to
     demonstrate the use of overlapping communicationn with computation.
  */
  PetscCall(MatAssemblyBegin(jac, MAT_FINAL_ASSEMBLY));
  PetscCall(VecRestoreArray(localX, &x));
  PetscCall(MatAssemblyEnd(jac, MAT_FINAL_ASSEMBLY));

  /*
     Tell the matrix we will never add a new nonzero location to the
     matrix. If we do, it will generate an error.
  */
  PetscCall(MatSetOption(jac, MAT_NEW_NONZERO_LOCATION_ERR, PETSC_TRUE));
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
