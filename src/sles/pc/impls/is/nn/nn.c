/*$Id: nn.c,v 1.13 2001/08/07 03:03:41 balay Exp $*/

#include "src/sles/pc/impls/is/nn/nn.h"

/* -------------------------------------------------------------------------- */
/*
   PCSetUp_NN - Prepares for the use of the NN preconditioner
                    by setting data structures and options.   

   Input Parameter:
.  pc - the preconditioner context

   Application Interface Routine: PCSetUp()

   Notes:
   The interface routine PCSetUp() is not usually called directly by
   the user, but instead is called by PCApply() if necessary.
*/
#undef __FUNCT__  
#define __FUNCT__ "PCSetUp_NN"
static int PCSetUp_NN(PC pc)
{
  int ierr;
  
  PetscFunctionBegin;
  if (!pc->setupcalled) {
    /* Set up all the "iterative substructuring" common block */
    ierr = PCISSetUp(pc);CHKERRQ(ierr);
    /* Create the coarse matrix. */
    ierr = PCNNCreateCoarseMatrix(pc);CHKERRQ(ierr);
  } 
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
/*
   PCApply_NN - Applies the NN preconditioner to a vector.

   Input Parameters:
.  pc - the preconditioner context
.  r - input vector (global)

   Output Parameter:
.  z - output vector (global)

   Application Interface Routine: PCApply()
 */
#undef __FUNCT__  
#define __FUNCT__ "PCApply_NN"
static int PCApply_NN(PC pc,Vec r,Vec z)
{
  PC_IS       *pcis = (PC_IS*)(pc->data);
  int         ierr;
  PetscScalar m_one = -1.0;
  Vec         w = pcis->vec1_global;

  PetscFunctionBegin;

  /*
    Dirichlet solvers.
    Solving $ B_I^{(i)}r_I^{(i)} $ at each processor.
    Storing the local results at vec2_D
  */
  ierr = VecScatterBegin(r,pcis->vec1_D,INSERT_VALUES,SCATTER_FORWARD,pcis->global_to_D);CHKERRQ(ierr);
  ierr = VecScatterEnd  (r,pcis->vec1_D,INSERT_VALUES,SCATTER_FORWARD,pcis->global_to_D);CHKERRQ(ierr);
  ierr = SLESSolve(pcis->sles_D,pcis->vec1_D,pcis->vec2_D);CHKERRQ(ierr);
  
  /*
    Computing $ r_B - \sum_j \tilde R_j^T A_{BI}^{(j)} (B_I^{(j)}r_I^{(j)}) $ .
    Storing the result in the interface portion of the global vector w.
  */
  ierr = MatMult(pcis->A_BI,pcis->vec2_D,pcis->vec1_B);CHKERRQ(ierr);
  ierr = VecScale(&m_one,pcis->vec1_B);CHKERRQ(ierr);
  ierr = VecCopy(r,w);CHKERRQ(ierr);
  ierr = VecScatterBegin(pcis->vec1_B,w,ADD_VALUES,SCATTER_REVERSE,pcis->global_to_B);CHKERRQ(ierr);
  ierr = VecScatterEnd  (pcis->vec1_B,w,ADD_VALUES,SCATTER_REVERSE,pcis->global_to_B);CHKERRQ(ierr);

  /*
    Apply the interface preconditioner
  */
  ierr = PCNNApplyInterfacePreconditioner(pc,w,z,pcis->work_N,pcis->vec1_B,pcis->vec2_B,pcis->vec3_B,pcis->vec1_D,
                                          pcis->vec3_D,pcis->vec1_N,pcis->vec2_N);CHKERRQ(ierr);

  /*
    Computing $ t_I^{(i)} = A_{IB}^{(i)} \tilde R_i z_B $
    The result is stored in vec1_D.
  */
  ierr = VecScatterBegin(z,pcis->vec1_B,INSERT_VALUES,SCATTER_FORWARD,pcis->global_to_B);CHKERRQ(ierr);
  ierr = VecScatterEnd  (z,pcis->vec1_B,INSERT_VALUES,SCATTER_FORWARD,pcis->global_to_B);CHKERRQ(ierr);
  ierr = MatMult(pcis->A_IB,pcis->vec1_B,pcis->vec1_D);CHKERRQ(ierr);

  /*
    Dirichlet solvers.
    Computing $ B_I^{(i)}t_I^{(i)} $ and sticking into the global vector the blocks
    $ B_I^{(i)}r_I^{(i)} - B_I^{(i)}t_I^{(i)} $.
  */
  ierr = VecScatterBegin(pcis->vec2_D,z,INSERT_VALUES,SCATTER_REVERSE,pcis->global_to_D);CHKERRQ(ierr);
  ierr = VecScatterEnd  (pcis->vec2_D,z,INSERT_VALUES,SCATTER_REVERSE,pcis->global_to_D);CHKERRQ(ierr);
  ierr = SLESSolve(pcis->sles_D,pcis->vec1_D,pcis->vec2_D);CHKERRQ(ierr);
  ierr = VecScale(&m_one,pcis->vec2_D);CHKERRQ(ierr);
  ierr = VecScatterBegin(pcis->vec2_D,z,ADD_VALUES,SCATTER_REVERSE,pcis->global_to_D);CHKERRQ(ierr);
  ierr = VecScatterEnd  (pcis->vec2_D,z,ADD_VALUES,SCATTER_REVERSE,pcis->global_to_D);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
/*
   PCDestroy_NN - Destroys the private context for the NN preconditioner
   that was created with PCCreate_NN().

   Input Parameter:
.  pc - the preconditioner context

   Application Interface Routine: PCDestroy()
*/
#undef __FUNCT__  
#define __FUNCT__ "PCDestroy_NN"
static int PCDestroy_NN(PC pc)
{
  PC_NN *pcnn = (PC_NN*)pc->data;
  int   ierr;

  PetscFunctionBegin;

  ierr = PCISDestroy(pc);CHKERRQ(ierr);

  if (pcnn->coarse_mat)  {ierr = MatDestroy(pcnn->coarse_mat);CHKERRQ(ierr);}
  if (pcnn->coarse_x)    {ierr = VecDestroy(pcnn->coarse_x);CHKERRQ(ierr);}
  if (pcnn->coarse_b)    {ierr = VecDestroy(pcnn->coarse_b);CHKERRQ(ierr);}
  if (pcnn->sles_coarse) {ierr = SLESDestroy(pcnn->sles_coarse);CHKERRQ(ierr);}
  if (pcnn->DZ_IN) {
    if (pcnn->DZ_IN[0]) {ierr = PetscFree(pcnn->DZ_IN[0]);CHKERRQ(ierr);}
    ierr = PetscFree(pcnn->DZ_IN);CHKERRQ(ierr);
  }

  /*
      Free the private data structure that was hanging off the PC
  */
  ierr = PetscFree(pcnn);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
/*
   PCCreate_NN - Creates a NN preconditioner context, PC_NN, 
   and sets this as the private data within the generic preconditioning 
   context, PC, that was created within PCCreate().

   Input Parameter:
.  pc - the preconditioner context

   Application Interface Routine: PCCreate()
*/
EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PCCreate_NN"
int PCCreate_NN(PC pc)
{
  int   ierr;
  PC_NN *pcnn;

  PetscFunctionBegin;

  /*
     Creates the private data structure for this preconditioner and
     attach it to the PC object.
  */
  ierr      = PetscNew(PC_NN,&pcnn);CHKERRQ(ierr);
  pc->data  = (void*)pcnn;

  /*
     Logs the memory usage; this is not needed but allows PETSc to 
     monitor how much memory is being used for various purposes.
  */
  PetscLogObjectMemory(pc,sizeof(PC_NN)+sizeof(PC_IS)); /* Is this the right thing to do? */

  ierr = PCISCreate(pc);CHKERRQ(ierr);
  pcnn->coarse_mat  = 0;
  pcnn->coarse_x    = 0;
  pcnn->coarse_b    = 0;
  pcnn->sles_coarse = 0;
  pcnn->DZ_IN       = 0;

  /*
      Set the pointers for the functions that are provided above.
      Now when the user-level routines (such as PCApply(), PCDestroy(), etc.)
      are called, they will automatically call these functions.  Note we
      choose not to provide a couple of these functions since they are
      not needed.
  */
  pc->ops->apply               = PCApply_NN;
  pc->ops->applytranspose      = 0;
  pc->ops->setup               = PCSetUp_NN;
  pc->ops->destroy             = PCDestroy_NN;
  pc->ops->view                = 0;
  pc->ops->applyrichardson     = 0;
  pc->ops->applysymmetricleft  = 0;
  pc->ops->applysymmetricright = 0;

  PetscFunctionReturn(0);
}
EXTERN_C_END


/* -------------------------------------------------------------------------- */
/*
   PCNNCreateCoarseMatrix - 
*/
#undef __FUNCT__  
#define __FUNCT__ "PCNNCreateCoarseMatrix"
int PCNNCreateCoarseMatrix (PC pc)
{
  MPI_Request    *send_request, *recv_request;
  int            i, j, k, ierr;

  PetscScalar*   mat;    /* Sub-matrix with this subdomain's contribution to the coarse matrix             */
  PetscScalar**  DZ_OUT; /* proc[k].DZ_OUT[i][] = bit of vector to be sent from processor k to processor i */

  /* aliasing some names */
  PC_IS*         pcis     = (PC_IS*)(pc->data);
  PC_NN*         pcnn     = (PC_NN*)pc->data;
  int            n_neigh  = pcis->n_neigh;
  int*           neigh    = pcis->neigh;
  int*           n_shared = pcis->n_shared;
  int**          shared   = pcis->shared;  
  PetscScalar**  DZ_IN;   /* Must be initialized after memory allocation. */

  PetscFunctionBegin;

  /* Allocate memory for mat (the +1 is to handle the case n_neigh equal to zero) */
  ierr = PetscMalloc((n_neigh*n_neigh+1)*sizeof(PetscScalar),&mat);CHKERRQ(ierr);

  /* Allocate memory for DZ */
  /* Notice that DZ_OUT[0] is allocated some space that is never used. */
  /* This is just in order to DZ_OUT and DZ_IN to have exactly the same form. */
  {
    int size_of_Z = 0;
    ierr  = PetscMalloc ((n_neigh+1)*sizeof(PetscScalar*),&pcnn->DZ_IN);CHKERRQ(ierr);
    DZ_IN = pcnn->DZ_IN;
    ierr  = PetscMalloc ((n_neigh+1)*sizeof(PetscScalar*),&DZ_OUT);CHKERRQ(ierr);
    for (i=0; i<n_neigh; i++) {
      size_of_Z += n_shared[i];
    }
    ierr = PetscMalloc ((size_of_Z+1)*sizeof(PetscScalar),&DZ_IN[0]);CHKERRQ(ierr);
    ierr = PetscMalloc ((size_of_Z+1)*sizeof(PetscScalar),&DZ_OUT[0]);CHKERRQ(ierr);
  }
  for (i=1; i<n_neigh; i++) {
    DZ_IN[i]  = DZ_IN [i-1] + n_shared[i-1];
    DZ_OUT[i] = DZ_OUT[i-1] + n_shared[i-1];
  }

  /* Set the values of DZ_OUT, in order to send this info to the neighbours */
  /* First, set the auxiliary array pcis->work_N. */
  ierr = PCISScatterArrayNToVecB(pcis->work_N,pcis->D,INSERT_VALUES,SCATTER_REVERSE,pc);CHKERRQ(ierr);
  for (i=1; i<n_neigh; i++){
    for (j=0; j<n_shared[i]; j++) {
      DZ_OUT[i][j] = pcis->work_N[shared[i][j]];
    }
  }

  /* Non-blocking send/receive the common-interface chunks of scaled nullspaces */
  /* Notice that send_request[] and recv_request[] could have one less element. */
  /* We make them longer to have request[i] corresponding to neigh[i].          */
  {
    int tag;
    ierr = PetscObjectGetNewTag((PetscObject)pc,&tag);CHKERRQ(ierr);
    ierr = PetscMalloc((2*(n_neigh)+1)*sizeof(MPI_Request),&send_request);CHKERRQ(ierr);
    recv_request = send_request + (n_neigh);
    for (i=1; i<n_neigh; i++) {
      ierr = MPI_Isend((void*)(DZ_OUT[i]),n_shared[i],MPIU_SCALAR,neigh[i],tag,pc->comm,&(send_request[i]));CHKERRQ(ierr);
      ierr = MPI_Irecv((void*)(DZ_IN [i]),n_shared[i],MPIU_SCALAR,neigh[i],tag,pc->comm,&(recv_request[i]));CHKERRQ(ierr);
    }
  }

  /* Set DZ_IN[0][] (recall that neigh[0]==rank, always) */
  for(j=0; j<n_shared[0]; j++) {
    DZ_IN[0][j] = pcis->work_N[shared[0][j]];
  }

  /* Start computing with local D*Z while communication goes on.    */
  /* Apply Schur complement. The result is "stored" in vec (more    */
  /* precisely, vec points to the result, stored in pc_nn->vec1_B)  */
  /* and also scattered to pcnn->work_N.                            */
  ierr = PCNNApplySchurToChunk(pc,n_shared[0],shared[0],DZ_IN[0],pcis->work_N,pcis->vec1_B,
                               pcis->vec2_B,pcis->vec1_D,pcis->vec2_D);CHKERRQ(ierr);

  /* Compute the first column, while completing the receiving. */
  for (i=0; i<n_neigh; i++) {
    MPI_Status stat;
    int ind=0;
    if (i>0) { ierr = MPI_Waitany(n_neigh-1,recv_request+1,&ind,&stat);CHKERRQ(ierr); ind++;}
    mat[ind*n_neigh+0] = 0.0;
    for (k=0; k<n_shared[ind]; k++) {
      mat[ind*n_neigh+0] += DZ_IN[ind][k] * pcis->work_N[shared[ind][k]];
    }
  }

  /* Compute the remaining of the columns */
  for (j=1; j<n_neigh; j++) {
    ierr = PCNNApplySchurToChunk(pc,n_shared[j],shared[j],DZ_IN[j],pcis->work_N,pcis->vec1_B,
                                 pcis->vec2_B,pcis->vec1_D,pcis->vec2_D);CHKERRQ(ierr);
    for (i=0; i<n_neigh; i++) {
      mat[i*n_neigh+j] = 0.0;
      for (k=0; k<n_shared[i]; k++) {
	mat[i*n_neigh+j] += DZ_IN[i][k] * pcis->work_N[shared[i][k]];
      }
    }
  }

  /* Complete the sending. */
  if (n_neigh>1) {
    MPI_Status *stat;
    ierr = PetscMalloc((n_neigh-1)*sizeof(MPI_Status),&stat);CHKERRQ(ierr);
    ierr = MPI_Waitall(n_neigh-1,&(send_request[1]),stat);CHKERRQ(ierr);
    ierr = PetscFree(stat);CHKERRQ(ierr);
  }

  /* Free the memory for the MPI requests */
  ierr = PetscFree(send_request);CHKERRQ(ierr);

  /* Free the memory for DZ_OUT */
  if (DZ_OUT) {
    if (DZ_OUT[0]) { ierr = PetscFree(DZ_OUT[0]);CHKERRQ(ierr); }
    ierr = PetscFree(DZ_OUT);CHKERRQ(ierr);
  }

  {
    int size,n_neigh_m1;
    ierr = MPI_Comm_size(pc->comm,&size);CHKERRQ(ierr);
    n_neigh_m1 = (n_neigh) ? n_neigh-1 : 0;
    /* Create the global coarse vectors (rhs and solution). */
    ierr = VecCreateMPI(pc->comm,1,size,&(pcnn->coarse_b));CHKERRQ(ierr);
    ierr = VecDuplicate(pcnn->coarse_b,&(pcnn->coarse_x));CHKERRQ(ierr);
    /* Create and set the global coarse matrix. */
    ierr = MatCreateMPIAIJ(pc->comm,1,1,size,size,1,PETSC_NULL,n_neigh_m1,PETSC_NULL,&(pcnn->coarse_mat));CHKERRQ(ierr);
    ierr = MatSetValues(pcnn->coarse_mat,n_neigh,neigh,n_neigh,neigh,mat,ADD_VALUES);CHKERRQ(ierr);
    ierr = MatAssemblyBegin(pcnn->coarse_mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd  (pcnn->coarse_mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }

  {
    int         rank;
    PetscScalar one = 1.0; 
    IS          is;
    ierr = MPI_Comm_rank(pc->comm,&rank);CHKERRQ(ierr);
    /* "Zero out" rows of not-purely-Neumann subdomains */
    if (pcis->pure_neumann) {  /* does NOT zero the row; create an empty index set. The reason is that MatZeroRows() is collective. */
      ierr = ISCreateStride(pc->comm,0,0,0,&is);CHKERRQ(ierr);
    } else { /* here it DOES zero the row, since it's not a floating subdomain. */
      ierr = ISCreateStride(pc->comm,1,rank,0,&is);CHKERRQ(ierr);
    }
    ierr = MatZeroRows(pcnn->coarse_mat,is,&one);CHKERRQ(ierr);
    ierr = ISDestroy(is);CHKERRQ(ierr);
  }

  /* Create the coarse linear solver context */
  {
    PC  pc_ctx, inner_pc;
    KSP ksp_ctx;
    ierr = SLESCreate(pc->comm,&pcnn->sles_coarse);CHKERRQ(ierr);
    ierr = SLESSetOperators(pcnn->sles_coarse,pcnn->coarse_mat,pcnn->coarse_mat,SAME_PRECONDITIONER);CHKERRQ(ierr);
    ierr = SLESGetKSP(pcnn->sles_coarse,&ksp_ctx);CHKERRQ(ierr);
    ierr = KSPGetPC(ksp_ctx,&pc_ctx);CHKERRQ(ierr);
    ierr = PCSetType(pc_ctx,PCREDUNDANT);CHKERRQ(ierr);                
    ierr = KSPSetType(ksp_ctx,KSPPREONLY);CHKERRQ(ierr);               
    ierr = PCRedundantGetPC(pc_ctx,&inner_pc);CHKERRQ(ierr);           
    ierr = PCSetType(inner_pc,PCLU);CHKERRQ(ierr);                     
    ierr = SLESSetOptionsPrefix(pcnn->sles_coarse,"coarse_");CHKERRQ(ierr);
    ierr = SLESSetFromOptions(pcnn->sles_coarse);CHKERRQ(ierr);
    /* the vectors in the following line are dummy arguments, just telling the SLES the vector size. Values are not used */
    ierr = SLESSetUp(pcnn->sles_coarse,pcnn->coarse_x,pcnn->coarse_b);CHKERRQ(ierr);
  }

  /* Free the memory for mat */
  ierr = PetscFree(mat);CHKERRQ(ierr);

  /* for DEBUGGING, save the coarse matrix to a file. */
  {
    PetscTruth flg;
    ierr = PetscOptionsHasName(PETSC_NULL,"-save_coarse_matrix",&flg);CHKERRQ(ierr);
    if (flg) {
      PetscViewer viewer;
      ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD,"coarse.m",&viewer);CHKERRQ(ierr);
      ierr = PetscViewerSetFormat(viewer,PETSC_VIEWER_ASCII_MATLAB);CHKERRQ(ierr);
      ierr = MatView(pcnn->coarse_mat,viewer);CHKERRQ(ierr);
      ierr = PetscViewerDestroy(viewer);CHKERRQ(ierr);
    }
  }

  /*  Set the variable pcnn->factor_coarse_rhs. */
  pcnn->factor_coarse_rhs = (pcis->pure_neumann) ? 1.0 : 0.0;

  /* See historical note 02, at the bottom of this file. */

  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
/*
   PCNNApplySchurToChunk - 

   Input parameters:
.  pcnn
.  n - size of chunk
.  idx - indices of chunk
.  chunk - values

   Output parameters:
.  array_N - result of Schur complement applied to chunk, scattered to big array
.  vec1_B  - result of Schur complement applied to chunk
.  vec2_B  - garbage (used as work space)
.  vec1_D  - garbage (used as work space)
.  vec2_D  - garbage (used as work space)

*/
#undef __FUNCT__  
#define __FUNCT__ "PCNNApplySchurToChunk"
int PCNNApplySchurToChunk(PC pc, int n, int* idx, PetscScalar *chunk, PetscScalar* array_N, Vec vec1_B, Vec vec2_B, Vec vec1_D, Vec vec2_D)
{
  int   i, ierr;
  PC_IS *pcis = (PC_IS*)(pc->data);

  PetscFunctionBegin;

  ierr = PetscMemzero((void*)array_N, pcis->n*sizeof(PetscScalar));CHKERRQ(ierr);
  for (i=0; i<n; i++) { array_N[idx[i]] = chunk[i]; }
  ierr = PCISScatterArrayNToVecB(array_N,vec2_B,INSERT_VALUES,SCATTER_FORWARD,pc);CHKERRQ(ierr);
  ierr = PCISApplySchur(pc,vec2_B,vec1_B,(Vec)0,vec1_D,vec2_D);CHKERRQ(ierr);
  ierr = PCISScatterArrayNToVecB(array_N,vec1_B,INSERT_VALUES,SCATTER_REVERSE,pc);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
/*
   PCNNApplyInterfacePreconditioner - Apply the interface preconditioner, i.e., 
                                      the preconditioner for the Schur complement.

   Input parameter:
.  r - global vector of interior and interface nodes. The values on the interior nodes are NOT used.

   Output parameters:
.  z - global vector of interior and interface nodes. The values on the interface are the result of
       the application of the interface preconditioner to the interface part of r. The values on the
       interior nodes are garbage.
.  work_N - array of local nodes (interior and interface, including ghosts); returns garbage (used as work space)
.  vec1_B - vector of local interface nodes (including ghosts); returns garbage (used as work space)
.  vec2_B - vector of local interface nodes (including ghosts); returns garbage (used as work space)
.  vec3_B - vector of local interface nodes (including ghosts); returns garbage (used as work space)
.  vec1_D - vector of local interior nodes; returns garbage (used as work space)
.  vec2_D - vector of local interior nodes; returns garbage (used as work space)
.  vec1_N - vector of local nodes (interior and interface, including ghosts); returns garbage (used as work space)
.  vec2_N - vector of local nodes (interior and interface, including ghosts); returns garbage (used as work space)

*/
#undef __FUNCT__
#define __FUNCT__ "PCNNApplyInterfacePreconditioner"
int PCNNApplyInterfacePreconditioner (PC pc, Vec r, Vec z, PetscScalar* work_N, Vec vec1_B, Vec vec2_B, Vec vec3_B, Vec vec1_D,
                                      Vec vec2_D, Vec vec1_N, Vec vec2_N)
{
  int    ierr;
  PC_IS* pcis = (PC_IS*)(pc->data);

  PetscFunctionBegin;

  /*
    First balancing step.
  */
  {
    PetscTruth flg;
    ierr = PetscOptionsHasName(PETSC_NULL,"-turn_off_first_balancing",&flg);CHKERRQ(ierr);
    if (!flg) {
      ierr = PCNNBalancing(pc,r,(Vec)0,z,vec1_B,vec2_B,(Vec)0,vec1_D,vec2_D,work_N);CHKERRQ(ierr);
    } else {
      ierr = VecCopy(r,z);CHKERRQ(ierr);
    }
  }

  /*
    Extract the local interface part of z and scale it by D 
  */
  ierr = VecScatterBegin(z,vec1_B,INSERT_VALUES,SCATTER_FORWARD,pcis->global_to_B);CHKERRQ(ierr);
  ierr = VecScatterEnd  (z,vec1_B,INSERT_VALUES,SCATTER_FORWARD,pcis->global_to_B);CHKERRQ(ierr);
  ierr = VecPointwiseMult(pcis->D,vec1_B,vec2_B);CHKERRQ(ierr);

  /* Neumann Solver */
  ierr = PCISApplyInvSchur(pc,vec2_B,vec1_B,vec1_N,vec2_N);CHKERRQ(ierr);

  /*
    Second balancing step.
  */
  {
    PetscTruth flg;
    ierr = PetscOptionsHasName(PETSC_NULL,"-turn_off_second_balancing",&flg);CHKERRQ(ierr);
    if (!flg) {
      ierr = PCNNBalancing(pc,r,vec1_B,z,vec2_B,vec3_B,(Vec)0,vec1_D,vec2_D,work_N);CHKERRQ(ierr);
    } else {
      PetscScalar zero = 0.0;
      ierr = VecPointwiseMult(pcis->D,vec1_B,vec2_B);CHKERRQ(ierr);
      ierr = VecSet(&zero,z);CHKERRQ(ierr);
      ierr = VecScatterBegin(vec2_B,z,ADD_VALUES,SCATTER_REVERSE,pcis->global_to_B);CHKERRQ(ierr);
      ierr = VecScatterEnd  (vec2_B,z,ADD_VALUES,SCATTER_REVERSE,pcis->global_to_B);CHKERRQ(ierr);
    }
  }

  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
/*
   PCNNBalancing - Computes z, as given in equations (15) and (16) (if the
                   input argument u is provided), or s, as given in equations
                   (12) and (13), if the input argument u is a null vector.
                   Notice that the input argument u plays the role of u_i in
                   equation (14). The equation numbers refer to [Man93].

   Input Parameters:
.  pcnn - NN preconditioner context.
.  r - MPI vector of all nodes (interior and interface). It's preserved.
.  u - (Optional) sequential vector of local interface nodes. It's preserved UNLESS vec3_B is null.

   Output Parameters:
.  z - MPI vector of interior and interface nodes. Returns s or z (see description above).
.  vec1_B - Sequential vector of local interface nodes. Workspace.
.  vec2_B - Sequential vector of local interface nodes. Workspace.
.  vec3_B - (Optional) sequential vector of local interface nodes. Workspace.
.  vec1_D - Sequential vector of local interior nodes. Workspace.
.  vec2_D - Sequential vector of local interior nodes. Workspace.
.  work_N - Array of all local nodes (interior and interface). Workspace.

*/
#undef __FUNCT__  
#define __FUNCT__ "PCNNBalancing"
int PCNNBalancing (PC pc, Vec r, Vec u, Vec z, Vec vec1_B, Vec vec2_B, Vec vec3_B,
                   Vec vec1_D, Vec vec2_D, PetscScalar *work_N)
{
  int            k, ierr;
  PetscScalar    zero     =  0.0;
  PetscScalar    m_one    = -1.0;
  PetscScalar    value;
  PetscScalar*   lambda;
  PC_NN*         pcnn     = (PC_NN*)(pc->data);
  PC_IS*         pcis     = (PC_IS*)(pc->data);

  PetscFunctionBegin;
  ierr = PetscLogEventBegin(PC_ApplyCoarse,0,0,0,0);CHKERRQ(ierr);

  if (u) { 
    if (!vec3_B) { vec3_B = u; }
    ierr = VecPointwiseMult(pcis->D,u,vec1_B);CHKERRQ(ierr);
    ierr = VecSet(&zero,z);CHKERRQ(ierr);
    ierr = VecScatterBegin(vec1_B,z,ADD_VALUES,SCATTER_REVERSE,pcis->global_to_B);CHKERRQ(ierr);
    ierr = VecScatterEnd  (vec1_B,z,ADD_VALUES,SCATTER_REVERSE,pcis->global_to_B);CHKERRQ(ierr);
    ierr = VecScatterBegin(z,vec2_B,INSERT_VALUES,SCATTER_FORWARD,pcis->global_to_B);CHKERRQ(ierr);
    ierr = VecScatterEnd  (z,vec2_B,INSERT_VALUES,SCATTER_FORWARD,pcis->global_to_B);CHKERRQ(ierr);
    ierr = PCISApplySchur(pc,vec2_B,vec3_B,(Vec)0,vec1_D,vec2_D);CHKERRQ(ierr);
    ierr = VecScale(&m_one,vec3_B);CHKERRQ(ierr);
    ierr = VecCopy(r,z);CHKERRQ(ierr);
    ierr = VecScatterBegin(vec3_B,z,ADD_VALUES,SCATTER_REVERSE,pcis->global_to_B);CHKERRQ(ierr);
    ierr = VecScatterEnd  (vec3_B,z,ADD_VALUES,SCATTER_REVERSE,pcis->global_to_B);CHKERRQ(ierr);
  } else {
    ierr = VecCopy(r,z);CHKERRQ(ierr);
  }
  ierr = VecScatterBegin(z,vec2_B,INSERT_VALUES,SCATTER_FORWARD,pcis->global_to_B);CHKERRQ(ierr);
  ierr = VecScatterEnd  (z,vec2_B,INSERT_VALUES,SCATTER_FORWARD,pcis->global_to_B);CHKERRQ(ierr);
  ierr = PCISScatterArrayNToVecB(work_N,vec2_B,INSERT_VALUES,SCATTER_REVERSE,pc);CHKERRQ(ierr);
  for (k=0, value=0.0; k<pcis->n_shared[0]; k++) { value += pcnn->DZ_IN[0][k] * work_N[pcis->shared[0][k]]; }
  value *= pcnn->factor_coarse_rhs;  /* This factor is set in CreateCoarseMatrix(). */
  {
    int rank;
    ierr = MPI_Comm_rank(pc->comm,&rank);CHKERRQ(ierr);
    ierr = VecSetValue(pcnn->coarse_b,rank,value,INSERT_VALUES);CHKERRQ(ierr);
    /*
       Since we are only inserting local values (one value actually) we don't need to do the 
       reduction that tells us there is no data that needs to be moved. Hence we comment out these
       ierr = VecAssemblyBegin(pcnn->coarse_b);CHKERRQ(ierr); 
       ierr = VecAssemblyEnd  (pcnn->coarse_b);CHKERRQ(ierr);
    */
  }
  ierr = SLESSolve(pcnn->sles_coarse,pcnn->coarse_b,pcnn->coarse_x);CHKERRQ(ierr);
  if (!u) { ierr = VecScale(&m_one,pcnn->coarse_x);CHKERRQ(ierr); }
  ierr = VecGetArray(pcnn->coarse_x,&lambda);CHKERRQ(ierr);
  for (k=0; k<pcis->n_shared[0]; k++) { work_N[pcis->shared[0][k]] = *lambda * pcnn->DZ_IN[0][k]; }
  ierr = VecRestoreArray(pcnn->coarse_x,&lambda);CHKERRQ(ierr);
  ierr = PCISScatterArrayNToVecB(work_N,vec2_B,INSERT_VALUES,SCATTER_FORWARD,pc);CHKERRQ(ierr);
  ierr = VecSet(&zero,z);CHKERRQ(ierr);
  ierr = VecScatterBegin(vec2_B,z,ADD_VALUES,SCATTER_REVERSE,pcis->global_to_B);CHKERRQ(ierr);
  ierr = VecScatterEnd  (vec2_B,z,ADD_VALUES,SCATTER_REVERSE,pcis->global_to_B);CHKERRQ(ierr);
  if (!u) {
    ierr = VecScatterBegin(z,vec2_B,INSERT_VALUES,SCATTER_FORWARD,pcis->global_to_B);CHKERRQ(ierr);
    ierr = VecScatterEnd  (z,vec2_B,INSERT_VALUES,SCATTER_FORWARD,pcis->global_to_B);CHKERRQ(ierr);
    ierr = PCISApplySchur(pc,vec2_B,vec1_B,(Vec)0,vec1_D,vec2_D);CHKERRQ(ierr);
    ierr = VecCopy(r,z);CHKERRQ(ierr);
  }
  ierr = VecScatterBegin(vec1_B,z,ADD_VALUES,SCATTER_REVERSE,pcis->global_to_B);CHKERRQ(ierr);
  ierr = VecScatterEnd  (vec1_B,z,ADD_VALUES,SCATTER_REVERSE,pcis->global_to_B);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(PC_ApplyCoarse,0,0,0,0);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNCT__



/*  -------   E N D   O F   T H E   C O D E   -------  */
/*                                                     */
/*  From now on, "footnotes" (or "historical notes").  */
/*                                                     */
/*  -------------------------------------------------  */


#ifdef __HISTORICAL_NOTES___do_not_compile__

/* --------------------------------------------------------------------------
   Historical note 01 
   -------------------------------------------------------------------------- */
/*
   We considered the possibility of an alternative D_i that would still
   provide a partition of unity (i.e., $ \sum_i  N_i D_i N_i^T = I $).
   The basic principle was still the pseudo-inverse of the counting
   function; the difference was that we would not count subdomains
   that do not contribute to the coarse space (i.e., not pure-Neumann
   subdomains).

   This turned out to be a bad idea:  we would solve trivial Neumann
   problems in the not pure-Neumann subdomains, since we would be scaling
   the balanced residual by zero.
*/

    {
      PetscTruth flg;
      ierr = PetscOptionsHasName(PETSC_NULL,"-pcnn_new_scaling",&flg);CHKERRQ(ierr);
      if (flg) {
        Vec    counter;
        PetscScalar one=1.0, zero=0.0;
        ierr = VecDuplicate(pc->vec,&counter);CHKERRQ(ierr);
        ierr = VecSet(&zero,counter);CHKERRQ(ierr);
        if (pcnn->pure_neumann) {
          ierr = VecSet(&one,pcnn->D);CHKERRQ(ierr);
        } else {
          ierr = VecSet(&zero,pcnn->D);CHKERRQ(ierr);
        }
        ierr = VecScatterBegin(pcnn->D,counter,ADD_VALUES,SCATTER_REVERSE,pcnn->global_to_B);CHKERRQ(ierr);
        ierr = VecScatterEnd  (pcnn->D,counter,ADD_VALUES,SCATTER_REVERSE,pcnn->global_to_B);CHKERRQ(ierr);
        ierr = VecScatterBegin(counter,pcnn->D,INSERT_VALUES,SCATTER_FORWARD,pcnn->global_to_B);CHKERRQ(ierr);
        ierr = VecScatterEnd  (counter,pcnn->D,INSERT_VALUES,SCATTER_FORWARD,pcnn->global_to_B);CHKERRQ(ierr);
        ierr = VecDestroy(counter);CHKERRQ(ierr);
        if (pcnn->pure_neumann) {
          ierr = VecReciprocal(pcnn->D);CHKERRQ(ierr);
        } else {
          ierr = VecSet(&zero,pcnn->D);CHKERRQ(ierr);
        }
      }
    }



/* --------------------------------------------------------------------------
   Historical note 02 
   -------------------------------------------------------------------------- */
/*
   We tried an alternative coarse problem, that would eliminate exactly a
   constant error. Turned out not to improve the overall convergence.
*/

  /*  Set the variable pcnn->factor_coarse_rhs. */
  {
    PetscTruth flg;
    ierr = PetscOptionsHasName(PETSC_NULL,"-enforce_preserving_constants",&flg);CHKERRQ(ierr);
    if (!flg) { pcnn->factor_coarse_rhs = (pcnn->pure_neumann) ? 1.0 : 0.0; }
    else {
      PetscScalar zero = 0.0, one = 1.0;
      ierr = VecSet(&one,pcnn->vec1_B);
      ierr = ApplySchurComplement(pcnn,pcnn->vec1_B,pcnn->vec2_B,(Vec)0,pcnn->vec1_D,pcnn->vec2_D);CHKERRQ(ierr);
      ierr = VecSet(&zero,pcnn->vec1_global);CHKERRQ(ierr);
      ierr = VecScatterBegin(pcnn->vec2_B,pcnn->vec1_global,ADD_VALUES,SCATTER_REVERSE,pcnn->global_to_B);CHKERRQ(ierr);
      ierr = VecScatterEnd  (pcnn->vec2_B,pcnn->vec1_global,ADD_VALUES,SCATTER_REVERSE,pcnn->global_to_B);CHKERRQ(ierr);
      ierr = VecScatterBegin(pcnn->vec1_global,pcnn->vec1_B,INSERT_VALUES,SCATTER_FORWARD,pcnn->global_to_B);CHKERRQ(ierr);
      ierr = VecScatterEnd  (pcnn->vec1_global,pcnn->vec1_B,INSERT_VALUES,SCATTER_FORWARD,pcnn->global_to_B);CHKERRQ(ierr);
      if (pcnn->pure_neumann) { pcnn->factor_coarse_rhs = 1.0; }
      else {
        ierr = ScatterArrayNToVecB(pcnn->work_N,pcnn->vec1_B,INSERT_VALUES,SCATTER_REVERSE,pcnn);CHKERRQ(ierr);
        for (k=0, pcnn->factor_coarse_rhs=0.0; k<pcnn->n_shared[0]; k++) {
          pcnn->factor_coarse_rhs += pcnn->work_N[pcnn->shared[0][k]] * pcnn->DZ_IN[0][k];
        }
        if (pcnn->factor_coarse_rhs) { pcnn->factor_coarse_rhs = 1.0 / pcnn->factor_coarse_rhs; }
        else { SETERRQ(1,"Constants cannot be preserved. Remove \"-enforce_preserving_constants\" option."); }
      }
    }
  }

#endif /* __HISTORICAL_NOTES___do_not_compile */
