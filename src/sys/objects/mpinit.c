

#include "petsc.h"        /*I  "petsc.h"   I*/
#include "petscsys.h"

static MPI_Comm saved_PETSC_COMM_WORLD = 0;
MPI_Comm PETSC_COMM_LOCAL_WORLD        = 0;        /* comm for a single node (local set of processes) */
static PetscTruth used_PetscOpenMP     = PETSC_FALSE;

extern PetscErrorCode PETSC_DLLEXPORT PetscOpenMPHandle(MPI_Comm);

#if defined(PETSC_HAVE_MPI_COMM_SPAWN)
#undef __FUNCT__  
#define __FUNCT__ "PetscOpenMPSpawn"
/*@C
   PetscOpenMPSpawn - Initialize additional processes to be used as "worker" processes. This is not generally 
     called by users. One should use -openmp_spawn_size <n> to indicate that you wish to have n-1 new MPI 
     processes spawned for each current process.

   Not Collective (could make collective on MPI_COMM_WORLD, generate one huge comm and then split it up)

   Input Parameter:
.  nodesize - size of each compute node that will share processors

   Options Database:
.   -openmp_spawn_size nodesize

   Notes: This is only supported on systems with an MPI 2 implementation that includes the MPI_Comm_Spawn() routine.

$    Comparison of two approaches for OpenMP usage (MPI started with N processes)
$
$    -openmp_spawn_size <n> requires MPI 2, results in n*N total processes with N directly used by application code
$                                           and n-1 worker processes (used by PETSc) for each application node.
$                           You MUST launch MPI so that only ONE MPI process is created for each hardware node.
$
$    -openmp_merge_size <n> results in N total processes, N/n used by the application code and the rest worker processes
$                            (used by PETSc)
$                           You MUST launch MPI so that n MPI processes are created for each hardware node.
$
$    petscmpirun -np 2 ./ex1 -openmp_spawn_size 3 gives 2 application nodes (and 4 PETSc worker nodes)
$    petscmpirun -np 6 ./ex1 -openmp_merge_size 3 gives the SAME 2 application nodes and 4 PETSc worker nodes
$       This is what would use if each of the computers hardware nodes had 3 CPUs.
$
$      These are intended to be used in conjunction with USER OpenMP code. The user will have 1 process per
$   computer (hardware) node (where the computer node has p cpus), the user's code will use threads to fully
$   utilize all the CPUs on the node. The PETSc code will have p processes to fully use the compute node for 
$   PETSc calculations. The user THREADS and PETSc PROCESSES will NEVER run at the same time so the p CPUs 
$   are always working on p task, never more than p.
$
$    See PCOPENMP for a PETSc preconditioner that can use this functionality
$
   Level: developer

   Concepts: OpenMP
   
.seealso: PetscFinalize(), PetscInitializeFortran(), PetscGetArgs(), PetscOpenMPFinalize(), PetscInitialize(), PetscOpenMPMerge()

@*/
PetscErrorCode PETSC_DLLEXPORT PetscOpenMPSpawn(PetscMPIInt nodesize)
{
  PetscErrorCode ierr;
  PetscMPIInt    size;
  MPI_Comm       parent,children;
							   
  PetscFunctionBegin;
  ierr = MPI_Comm_get_parent(&parent);CHKERRQ(ierr);
  if (parent == MPI_COMM_NULL) {
    char programname[PETSC_MAX_PATH_LEN];
    char **argv;

    ierr = PetscGetProgramName(programname,PETSC_MAX_PATH_LEN);CHKERRQ(ierr);
    ierr = PetscGetArguments(&argv);CHKERRQ(ierr);
    ierr = MPI_Comm_spawn(programname,argv,nodesize-1,MPI_INFO_NULL,0,PETSC_COMM_SELF,&children,MPI_ERRCODES_IGNORE);CHKERRQ(ierr);
    ierr = PetscFreeArguments(argv);CHKERRQ(ierr);
    ierr = MPI_Intercomm_merge(children,0,&PETSC_COMM_LOCAL_WORLD);CHKERRQ(ierr); 

    ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
    ierr = PetscInfo2(0,"PETSc OpenMP successfully spawned: number of nodes = %d node size = %d\n",size,nodesize);CHKERRQ(ierr);
    saved_PETSC_COMM_WORLD = PETSC_COMM_WORLD;
    used_PetscOpenMP       = PETSC_TRUE;
  } else { /* worker nodes that get spawned */
    ierr             = MPI_Intercomm_merge(parent,1,&PETSC_COMM_LOCAL_WORLD);CHKERRQ(ierr); 
    ierr             = PetscOpenMPHandle(PETSC_COMM_LOCAL_WORLD);CHKERRQ(ierr);
    used_PetscOpenMP = PETSC_FALSE; /* so that PetscOpenMPFinalize() will not attempt a broadcast from this process */
    ierr             = PetscEnd();  /* cannot continue into user code */
  }
  PetscFunctionReturn(0);
}
#endif

#undef __FUNCT__  
#define __FUNCT__ "PetscOpenMPMerge"
/*@C
   PetscOpenMPMerge - Initializes the PETSc and MPI to work with OpenMP. This is not usually called
      by the user. One should use -openmp_merge_size <n> to indicate the node size of merged communicator
      to be.

   Collective on MPI_COMM_WORLD or PETSC_COMM_WORLD if it has been set

   Input Parameter:
.  nodesize - size of each compute node that will share processors

   Options Database:
.   -openmp_merge_size <n>

   Level: developer

$    Comparison of two approaches for OpenMP usage (MPI started with N processes)
$
$    -openmp_spawn_size <n> requires MPI 2, results in n*N total processes with N directly used by application code
$                                           and n-1 worker processes (used by PETSc) for each application node.
$                           You MUST launch MPI so that only ONE MPI process is created for each hardware node.
$
$    -openmp_merge_size <n> results in N total processes, N/n used by the application code and the rest worker processes
$                            (used by PETSc)
$                           You MUST launch MPI so that n MPI processes are created for each hardware node.
$
$    petscmpirun -np 2 ./ex1 -openmp_spawn_size 3 gives 2 application nodes (and 4 PETSc worker nodes)
$    petscmpirun -np 6 ./ex1 -openmp_merge_size 3 gives the SAME 2 application nodes and 4 PETSc worker nodes
$       This is what would use if each of the computers hardware nodes had 3 CPUs.
$
$      These are intended to be used in conjunction with USER OpenMP code. The user will have 1 process per
$   computer (hardware) node (where the computer node has p cpus), the user's code will use threads to fully
$   utilize all the CPUs on the node. The PETSc code will have p processes to fully use the compute node for 
$   PETSc calculations. The user THREADS and PETSc PROCESSES will NEVER run at the same time so the p CPUs 
$   are always working on p task, never more than p.
$
$    See PCOPENMP for a PETSc preconditioner that can use this functionality
$

   Concepts: OpenMP
   
.seealso: PetscFinalize(), PetscInitializeFortran(), PetscGetArgs(), PetscOpenMPFinalize(), PetscInitialize()

@*/
PetscErrorCode PETSC_DLLEXPORT PetscOpenMPMerge(PetscMPIInt nodesize)
{
  PetscErrorCode ierr;
  PetscMPIInt    size,rank,*ranks,i;
  MPI_Group      group,newgroup;

  PetscFunctionBegin;
  saved_PETSC_COMM_WORLD = PETSC_COMM_WORLD;

  ierr = MPI_Comm_size(saved_PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
  if (size % nodesize) SETERRQ2(PETSC_ERR_ARG_SIZ,"Total number of process nodes %d is not divisible by number of processes per node %d",size,nodesize);
  ierr = MPI_Comm_rank(saved_PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);


  /* create two communicators 
      *) one that contains the first process from each node: 0,nodesize,2*nodesize,...
      *) one that contains all processes in a node:  (0,1,2...,nodesize-1), (nodesize,nodesize+1,...2*nodesize-), ...
  */
  ierr = MPI_Comm_group(saved_PETSC_COMM_WORLD,&group);CHKERRQ(ierr);
  ierr = PetscMalloc((size/nodesize)*sizeof(PetscMPIInt),&ranks);CHKERRQ(ierr);
  for (i=0; i<(size/nodesize); i++) ranks[i] = i*nodesize;
  ierr = MPI_Group_incl(group,size/nodesize,ranks,&newgroup);CHKERRQ(ierr);
  ierr = PetscFree(ranks);CHKERRQ(ierr);
  ierr = MPI_Comm_create(saved_PETSC_COMM_WORLD,newgroup,&PETSC_COMM_WORLD);CHKERRQ(ierr);
  if (rank % nodesize) PETSC_COMM_WORLD = 0; /* mark invalid processes for easy debugging */
  ierr = MPI_Group_free(&group);CHKERRQ(ierr);
  ierr = MPI_Group_free(&newgroup);CHKERRQ(ierr);

  ierr = MPI_Comm_split(saved_PETSC_COMM_WORLD,rank/nodesize,rank % nodesize,&PETSC_COMM_LOCAL_WORLD);CHKERRQ(ierr);

  ierr = PetscInfo2(0,"PETSc OpenMP successfully started: number of nodes = %d node size = %d\n",size/nodesize,nodesize);CHKERRQ(ierr);
  ierr = PetscInfo1(0,"PETSc OpenMP process %sactive\n",(rank % nodesize) ? "in" : "");CHKERRQ(ierr);

  used_PetscOpenMP = PETSC_TRUE;
  /* 
     All process not involved in user application code wait here
  */
  if (!PETSC_COMM_WORLD) {
    ierr             = PetscOpenMPHandle(PETSC_COMM_LOCAL_WORLD);CHKERRQ(ierr);
    PETSC_COMM_WORLD = saved_PETSC_COMM_WORLD;
    used_PetscOpenMP = PETSC_FALSE; /* so that PetscOpenMPIFinalize() will not attempt a broadcast from this process */
    ierr             = PetscEnd();  /* cannot continue into user code */
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscOpenMPFinalize"
/*@C
   PetscOpenMPFinalizes - Finalizes the PETSc and MPI to work with OpenMP. Called by PetscFinalize() cannot
       be called by user.

   Collective on the entire system

   Level: developer
           
.seealso: PetscFinalize(), PetscInitializeFortran(), PetscGetArgs(), PetscOpenMPMerge()

@*/
PetscErrorCode PETSC_DLLEXPORT PetscOpenMPFinalize(void)
{
  PetscErrorCode ierr = 0;
  PetscInt       command = 3;

  PetscFunctionBegin;
  if (!used_PetscOpenMP) PetscFunctionReturn(0);
  ierr = MPI_Bcast(&command,1,MPIU_INT,0,PETSC_COMM_LOCAL_WORLD);CHKERRQ(ierr);
  PETSC_COMM_WORLD = saved_PETSC_COMM_WORLD;
  PetscFunctionReturn(ierr);
}

static PetscInt numberobjects = 0;
static void     *objects[100];

#undef __FUNCT__  
#define __FUNCT__ "PetscOpenMPHandle"
/*@C
   PetscOpenMPHandle - Receives commands from the master node and processes them

   Collective on MPI_Comm

   Level: developer
           
.seealso: PetscOpenMPMerge()

@*/
PetscErrorCode PETSC_DLLEXPORT PetscOpenMPHandle(MPI_Comm comm)
{
  PetscErrorCode ierr;
  PetscInt       command;
  PetscTruth     exitwhileloop = PETSC_FALSE;

  PetscFunctionBegin;
  while (!exitwhileloop) {
    ierr = MPI_Bcast(&command,1,MPIU_INT,0,comm);CHKERRQ(ierr);
    switch (command) {
    case 0: { 
      size_t n;
      void   *ptr;
      ierr = MPI_Bcast(&n,1,MPI_INT,0,comm);CHKERRQ(ierr); /* may be wrong size here */
      /* cannot use PetscNew() cause it requires struct argument */
      ierr = PetscMalloc(n,&ptr);CHKERRQ(ierr);
      ierr = PetscMemzero(ptr,n);CHKERRQ(ierr);
      objects[numberobjects++] = ptr;
      break;
    }
    case 1: {
      PetscInt i;
      ierr = MPI_Bcast(&i,1,MPIU_INT,0,comm);CHKERRQ(ierr);
      ierr = PetscFree(objects[i]);CHKERRQ(ierr);
      objects[i] = 0;
      break;
    }
    case 2: {
      PetscInt       i;
      PetscErrorCode (*f)(MPI_Comm,void*);
      ierr = MPI_Bcast(&i,1,MPIU_INT,0,comm);CHKERRQ(ierr);
      ierr = MPI_Bcast(&f,1,MPIU_INT,0,comm);CHKERRQ(ierr);
      ierr = (*f)(comm,objects[i]);CHKERRQ(ierr);
      break;
    }
    case 3: {
      exitwhileloop = PETSC_TRUE;
      break;
    }
    default:
      SETERRQ1(PETSC_ERR_PLIB,"Unknown OpenMP command %D",command);
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscOpenMPNew"
/*@C
   PetscOpenMPNew - Creates a "c struct" on all nodes of an OpenMP communicator

   Collective on MPI_Comm

   Level: developer
           
.seealso: PetscOpenMPMerge()

@*/
PetscErrorCode PETSC_DLLEXPORT PetscOpenMPNew(MPI_Comm comm,size_t n,void **ptr)
{
  PetscErrorCode ierr;
  PetscInt       command = 0;

  PetscFunctionBegin;
  if (!used_PetscOpenMP) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not using OpenMP feature of PETSc");

  ierr = MPI_Bcast(&command,1,MPIU_INT,0,comm);CHKERRQ(ierr);
  ierr = MPI_Bcast(&n,1,MPI_INT,0,comm);CHKERRQ(ierr); /* may be wrong size here since size_t */
  /* cannot use PetscNew() cause it requires struct argument */
  ierr = PetscMalloc(n,ptr);CHKERRQ(ierr);
  ierr = PetscMemzero(*ptr,n);CHKERRQ(ierr);
  objects[numberobjects++] = *ptr;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscOpenMPFree"
/*@C
   PetscOpenMPFree - Frees a "c struct" on all nodes of an OpenMP communicator

   Collective on MPI_Comm

   Level: developer
           
.seealso: PetscOpenMPMerge(), PetscOpenMPNew()

@*/
PetscErrorCode PETSC_DLLEXPORT PetscOpenMPFree(MPI_Comm comm,void *ptr)
{
  PetscErrorCode ierr;
  PetscInt       command = 1,i;

  PetscFunctionBegin;
  if (!used_PetscOpenMP) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not using OpenMP feature of PETSc");

  ierr = MPI_Bcast(&command,1,MPIU_INT,0,comm);CHKERRQ(ierr);
  for (i=0; i<numberobjects; i++) {
    if (objects[i] == ptr) {
      ierr = MPI_Bcast(&i,1,MPIU_INT,0,comm);CHKERRQ(ierr);
      ierr = PetscFree(ptr);CHKERRQ(ierr);
      objects[i] = 0;
      PetscFunctionReturn(0);
    }
  }
  SETERRQ(PETSC_ERR_ARG_WRONG,"Pointer does not appear to have been created with PetscOpenMPNew()");
  PetscFunctionReturn(ierr);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscOpenMPRun"
/*@C
   PetscOpenMPRun - runs a function on all the processes of a node

   Collective on MPI_Comm

   Level: developer
           
.seealso: PetscOpenMPMerge(), PetscOpenMPNew(), PetscOpenMPFree()

@*/
PetscErrorCode PETSC_DLLEXPORT PetscOpenMPRun(MPI_Comm comm,PetscErrorCode (*f)(MPI_Comm,void *),void *ptr)
{
  PetscErrorCode ierr;
  PetscInt       command = 2,i;

  PetscFunctionBegin;
  if (!used_PetscOpenMP) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not using OpenMP feature of PETSc");

  ierr = MPI_Bcast(&command,1,MPIU_INT,0,comm);CHKERRQ(ierr);
  for (i=0; i<numberobjects; i++) {
    if (objects[i] == ptr) {
      ierr = MPI_Bcast(&i,1,MPIU_INT,0,comm);CHKERRQ(ierr);
      ierr = MPI_Bcast(&f,1,MPIU_INT,0,comm);CHKERRQ(ierr);
      ierr = (*f)(comm,ptr);CHKERRQ(ierr);
      PetscFunctionReturn(0);
    }
  }
  SETERRQ(PETSC_ERR_ARG_WRONG,"Pointer does not appear to have been created with PetscOpenMPNew()");
  PetscFunctionReturn(ierr);
}
