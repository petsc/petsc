#define PETSCKSP_DLL

#include "../src/ksp/pc/impls/is/pcis.h"

/* -------------------------------------------------------------------------- */
/*
   PCISSetUp - 
*/
#undef __FUNCT__  
#define __FUNCT__ "PCISSetUp"
PetscErrorCode PETSCKSP_DLLEXPORT PCISSetUp(PC pc)
{
  PC_IS           *pcis = (PC_IS*)(pc->data);
  Mat_IS          *matis = (Mat_IS*)pc->mat->data; 
  PetscInt        i;
  PetscErrorCode  ierr;
  PetscTruth      flg;
  
  PetscFunctionBegin;
  ierr = PetscTypeCompare((PetscObject)pc->mat,MATIS,&flg);CHKERRQ(ierr);
  if (!flg){
    SETERRQ(PETSC_ERR_ARG_WRONG,"Preconditioner type of Neumann Neumman requires matrix of type MATIS");
  }

  pcis->pure_neumann = matis->pure_neumann;

  /*
    Creating the local vector vec1_N, containing the inverse of the number
    of subdomains to which each local node (either owned or ghost)
    pertains. To accomplish that, we scatter local vectors of 1's to
    a global vector (adding the values); scatter the result back to
    local vectors and finally invert the result.
  */
  {
    Vec    counter;
    ierr = VecDuplicate(matis->x,&pcis->vec1_N);CHKERRQ(ierr);
    ierr = MatGetVecs(pc->pmat,&counter,0);CHKERRQ(ierr); /* temporary auxiliar vector */
    ierr = VecSet(counter,0.0);CHKERRQ(ierr);
    ierr = VecSet(pcis->vec1_N,1.0);CHKERRQ(ierr);
    ierr = VecScatterBegin(matis->ctx,pcis->vec1_N,counter,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
    ierr = VecScatterEnd  (matis->ctx,pcis->vec1_N,counter,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
    ierr = VecScatterBegin(matis->ctx,counter,pcis->vec1_N,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterEnd  (matis->ctx,counter,pcis->vec1_N,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecDestroy(counter);CHKERRQ(ierr);
  }
  /*
    Creating local and global index sets for interior and
    inteface nodes. Notice that interior nodes have D[i]==1.0.
  */
  {
    PetscInt     n_I;
    PetscInt    *idx_I_local,*idx_B_local,*idx_I_global,*idx_B_global;
    PetscScalar *array;
    /* Identifying interior and interface nodes, in local numbering */
    ierr = VecGetSize(pcis->vec1_N,&pcis->n);CHKERRQ(ierr);
    ierr = VecGetArray(pcis->vec1_N,&array);CHKERRQ(ierr);
    ierr = PetscMalloc(pcis->n*sizeof(PetscInt),&idx_I_local);CHKERRQ(ierr);
    ierr = PetscMalloc(pcis->n*sizeof(PetscInt),&idx_B_local);CHKERRQ(ierr);
    for (i=0, pcis->n_B=0, n_I=0; i<pcis->n; i++) {
      if (array[i] == 1.0) { idx_I_local[n_I]       = i; n_I++;       }
      else                 { idx_B_local[pcis->n_B] = i; pcis->n_B++; }
    }
    /* Getting the global numbering */
    idx_B_global = idx_I_local + n_I; /* Just avoiding allocating extra memory, since we have vacant space */
    idx_I_global = idx_B_local + pcis->n_B;
    ierr = ISLocalToGlobalMappingApply(matis->mapping,pcis->n_B,idx_B_local,idx_B_global);CHKERRQ(ierr);
    ierr = ISLocalToGlobalMappingApply(matis->mapping,n_I,      idx_I_local,idx_I_global);CHKERRQ(ierr);
    /* Creating the index sets. */
    ierr = ISCreateGeneral(MPI_COMM_SELF,pcis->n_B,idx_B_local, &pcis->is_B_local);CHKERRQ(ierr);
    ierr = ISCreateGeneral(MPI_COMM_SELF,pcis->n_B,idx_B_global,&pcis->is_B_global);CHKERRQ(ierr);
    ierr = ISCreateGeneral(MPI_COMM_SELF,n_I      ,idx_I_local, &pcis->is_I_local);CHKERRQ(ierr);
    ierr = ISCreateGeneral(MPI_COMM_SELF,n_I      ,idx_I_global,&pcis->is_I_global);CHKERRQ(ierr);
    /* Freeing memory and restoring arrays */
    ierr = PetscFree(idx_B_local);CHKERRQ(ierr);
    ierr = PetscFree(idx_I_local);CHKERRQ(ierr);
    ierr = VecRestoreArray(pcis->vec1_N,&array);CHKERRQ(ierr);
  }

  /*
    Extracting the blocks A_II, A_BI, A_IB and A_BB from A. If the numbering
    is such that interior nodes come first than the interface ones, we have

    [           |      ]
    [    A_II   | A_IB ]
    A = [           |      ]
    [-----------+------]
    [    A_BI   | A_BB ]
  */

  ierr = MatGetSubMatrix(matis->A,pcis->is_I_local,pcis->is_I_local,MAT_INITIAL_MATRIX,&pcis->A_II);CHKERRQ(ierr);
  ierr = MatGetSubMatrix(matis->A,pcis->is_I_local,pcis->is_B_local,MAT_INITIAL_MATRIX,&pcis->A_IB);CHKERRQ(ierr);
  ierr = MatGetSubMatrix(matis->A,pcis->is_B_local,pcis->is_I_local,MAT_INITIAL_MATRIX,&pcis->A_BI);CHKERRQ(ierr);
  ierr = MatGetSubMatrix(matis->A,pcis->is_B_local,pcis->is_B_local,MAT_INITIAL_MATRIX,&pcis->A_BB);CHKERRQ(ierr);

  /*
    Creating work vectors and arrays
  */
  /* pcis->vec1_N has already been created */
  ierr = VecDuplicate(pcis->vec1_N,&pcis->vec2_N);CHKERRQ(ierr);
  ierr = VecCreateSeq(PETSC_COMM_SELF,pcis->n-pcis->n_B,&pcis->vec1_D);CHKERRQ(ierr);
  ierr = VecDuplicate(pcis->vec1_D,&pcis->vec2_D);CHKERRQ(ierr);
  ierr = VecDuplicate(pcis->vec1_D,&pcis->vec3_D);CHKERRQ(ierr);
  ierr = VecCreateSeq(PETSC_COMM_SELF,pcis->n_B,&pcis->vec1_B);CHKERRQ(ierr);
  ierr = VecDuplicate(pcis->vec1_B,&pcis->vec2_B);CHKERRQ(ierr);
  ierr = VecDuplicate(pcis->vec1_B,&pcis->vec3_B);CHKERRQ(ierr);
  ierr = MatGetVecs(pc->pmat,&pcis->vec1_global,0);CHKERRQ(ierr);
  ierr = PetscMalloc((pcis->n)*sizeof(PetscScalar),&pcis->work_N);CHKERRQ(ierr);

  /* Creating the scatter contexts */
  ierr = VecScatterCreate(pcis->vec1_global,pcis->is_I_global,pcis->vec1_D,(IS)0,&pcis->global_to_D);CHKERRQ(ierr);
  ierr = VecScatterCreate(pcis->vec1_N,pcis->is_B_local,pcis->vec1_B,(IS)0,&pcis->N_to_B);CHKERRQ(ierr);
  ierr = VecScatterCreate(pcis->vec1_global,pcis->is_B_global,pcis->vec1_B,(IS)0,&pcis->global_to_B);CHKERRQ(ierr);

  /* Creating scaling "matrix" D, from information in vec1_N */
  ierr = VecDuplicate(pcis->vec1_B,&pcis->D);CHKERRQ(ierr);
  ierr = VecScatterBegin(pcis->N_to_B,pcis->vec1_N,pcis->D,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd  (pcis->N_to_B,pcis->vec1_N,pcis->D,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecReciprocal(pcis->D);CHKERRQ(ierr);    

  /* See historical note 01, at the bottom of this file. */

  /*
    Creating the KSP contexts for the local Dirichlet and Neumann problems.
  */
  {
    PC  pc_ctx;
    /* Dirichlet */
    ierr = KSPCreate(PETSC_COMM_SELF,&pcis->ksp_D);CHKERRQ(ierr);
    ierr = PetscObjectIncrementTabLevel((PetscObject)pcis->ksp_D,(PetscObject)pc,1);CHKERRQ(ierr);
    ierr = KSPSetOperators(pcis->ksp_D,pcis->A_II,pcis->A_II,SAME_PRECONDITIONER);CHKERRQ(ierr);
    ierr = KSPSetOptionsPrefix(pcis->ksp_D,"is_localD_");CHKERRQ(ierr);
    ierr = KSPGetPC(pcis->ksp_D,&pc_ctx);CHKERRQ(ierr);
    ierr = PCSetType(pc_ctx,PCLU);CHKERRQ(ierr);
    ierr = KSPSetType(pcis->ksp_D,KSPPREONLY);CHKERRQ(ierr);
    ierr = KSPSetFromOptions(pcis->ksp_D);CHKERRQ(ierr);
    /* the vectors in the following line are dummy arguments, just telling the KSP the vector size. Values are not used */
    ierr = KSPSetUp(pcis->ksp_D);CHKERRQ(ierr);
    /* Neumann */
    ierr = KSPCreate(PETSC_COMM_SELF,&pcis->ksp_N);CHKERRQ(ierr);
    ierr = PetscObjectIncrementTabLevel((PetscObject)pcis->ksp_N,(PetscObject)pc,1);CHKERRQ(ierr);
    ierr = KSPSetOperators(pcis->ksp_N,matis->A,matis->A,SAME_PRECONDITIONER);CHKERRQ(ierr);
    ierr = KSPSetOptionsPrefix(pcis->ksp_N,"is_localN_");CHKERRQ(ierr);
    ierr = KSPGetPC(pcis->ksp_N,&pc_ctx);CHKERRQ(ierr);
    ierr = PCSetType(pc_ctx,PCLU);CHKERRQ(ierr);
    ierr = KSPSetType(pcis->ksp_N,KSPPREONLY);CHKERRQ(ierr);
    ierr = KSPSetFromOptions(pcis->ksp_N);CHKERRQ(ierr);
    {
      PetscTruth damp_fixed = PETSC_FALSE,
                 remove_nullspace_fixed = PETSC_FALSE,
                 set_damping_factor_floating = PETSC_FALSE,
                 not_damp_floating = PETSC_FALSE,
                 not_remove_nullspace_floating = PETSC_FALSE;
      PetscReal  fixed_factor,
                 floating_factor;

      ierr = PetscOptionsGetReal(((PetscObject)pc_ctx)->prefix,"-pc_is_damp_fixed",&fixed_factor,&damp_fixed);CHKERRQ(ierr);
      if (!damp_fixed) { fixed_factor = 0.0; }
      ierr = PetscOptionsGetTruth(((PetscObject)pc_ctx)->prefix,"-pc_is_damp_fixed",&damp_fixed,PETSC_NULL);CHKERRQ(ierr);

      ierr = PetscOptionsGetTruth(((PetscObject)pc_ctx)->prefix,"-pc_is_remove_nullspace_fixed",&remove_nullspace_fixed,PETSC_NULL);CHKERRQ(ierr);

      ierr = PetscOptionsGetReal(((PetscObject)pc_ctx)->prefix,"-pc_is_set_damping_factor_floating",
			      &floating_factor,&set_damping_factor_floating);CHKERRQ(ierr);
      if (!set_damping_factor_floating) { floating_factor = 0.0; }
      ierr = PetscOptionsGetTruth(((PetscObject)pc_ctx)->prefix,"-pc_is_set_damping_factor_floating",&set_damping_factor_floating,PETSC_NULL);CHKERRQ(ierr);
      if (!set_damping_factor_floating) { floating_factor = 1.e-12; }

      ierr = PetscOptionsGetTruth(((PetscObject)pc_ctx)->prefix,"-pc_is_not_damp_floating",&not_damp_floating,PETSC_NULL);CHKERRQ(ierr);

      ierr = PetscOptionsGetTruth(((PetscObject)pc_ctx)->prefix,"-pc_is_not_remove_nullspace_floating",&not_remove_nullspace_floating,PETSC_NULL);CHKERRQ(ierr);

      if (pcis->pure_neumann) {  /* floating subdomain */ 
	if (!(not_damp_floating)) {
          ierr = PCFactorSetShiftType(pc_ctx,MAT_SHIFT_NONZERO);CHKERRQ(ierr);
          ierr = PCFactorSetShiftAmount(pc_ctx,floating_factor);CHKERRQ(ierr);
	}
	if (!(not_remove_nullspace_floating)){
	  MatNullSpace nullsp;
	  ierr = MatNullSpaceCreate(PETSC_COMM_SELF,PETSC_TRUE,0,PETSC_NULL,&nullsp);CHKERRQ(ierr);
	  ierr = KSPSetNullSpace(pcis->ksp_N,nullsp);CHKERRQ(ierr);
	  ierr = MatNullSpaceDestroy(nullsp);CHKERRQ(ierr);
	}
      } else {  /* fixed subdomain */
	if (damp_fixed) {
          ierr = PCFactorSetShiftType(pc_ctx,MAT_SHIFT_NONZERO);CHKERRQ(ierr);
          ierr = PCFactorSetShiftAmount(pc_ctx,floating_factor);CHKERRQ(ierr);
	}
	if (remove_nullspace_fixed) {
	  MatNullSpace nullsp;
	  ierr = MatNullSpaceCreate(PETSC_COMM_SELF,PETSC_TRUE,0,PETSC_NULL,&nullsp);CHKERRQ(ierr);
	  ierr = KSPSetNullSpace(pcis->ksp_N,nullsp);CHKERRQ(ierr);
	  ierr = MatNullSpaceDestroy(nullsp);CHKERRQ(ierr);
	}
      }
    }
    /* the vectors in the following line are dummy arguments, just telling the KSP the vector size. Values are not used */
    ierr = KSPSetUp(pcis->ksp_N);CHKERRQ(ierr);
  }

  ierr = ISLocalToGlobalMappingGetInfo(((Mat_IS*)(pc->mat->data))->mapping,&(pcis->n_neigh),&(pcis->neigh),&(pcis->n_shared),&(pcis->shared));CHKERRQ(ierr);
  pcis->ISLocalToGlobalMappingGetInfoWasCalled = PETSC_TRUE;
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
/*
   PCISDestroy -
*/
#undef __FUNCT__  
#define __FUNCT__ "PCISDestroy"
PetscErrorCode PETSCKSP_DLLEXPORT PCISDestroy(PC pc)
{
  PC_IS          *pcis = (PC_IS*)(pc->data);
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (pcis->is_B_local)  {ierr = ISDestroy(pcis->is_B_local);CHKERRQ(ierr);}
  if (pcis->is_I_local)  {ierr = ISDestroy(pcis->is_I_local);CHKERRQ(ierr);}
  if (pcis->is_B_global) {ierr = ISDestroy(pcis->is_B_global);CHKERRQ(ierr);}
  if (pcis->is_I_global) {ierr = ISDestroy(pcis->is_I_global);CHKERRQ(ierr);}
  if (pcis->A_II)        {ierr = MatDestroy(pcis->A_II);CHKERRQ(ierr);}
  if (pcis->A_IB)        {ierr = MatDestroy(pcis->A_IB);CHKERRQ(ierr);}
  if (pcis->A_BI)        {ierr = MatDestroy(pcis->A_BI);CHKERRQ(ierr);}
  if (pcis->A_BB)        {ierr = MatDestroy(pcis->A_BB);CHKERRQ(ierr);}
  if (pcis->D)           {ierr = VecDestroy(pcis->D);CHKERRQ(ierr);}
  if (pcis->ksp_N)      {ierr = KSPDestroy(pcis->ksp_N);CHKERRQ(ierr);}
  if (pcis->ksp_D)      {ierr = KSPDestroy(pcis->ksp_D);CHKERRQ(ierr);}
  if (pcis->vec1_N)      {ierr = VecDestroy(pcis->vec1_N);CHKERRQ(ierr);}
  if (pcis->vec2_N)      {ierr = VecDestroy(pcis->vec2_N);CHKERRQ(ierr);}
  if (pcis->vec1_D)      {ierr = VecDestroy(pcis->vec1_D);CHKERRQ(ierr);}
  if (pcis->vec2_D)      {ierr = VecDestroy(pcis->vec2_D);CHKERRQ(ierr);}
  if (pcis->vec3_D)      {ierr = VecDestroy(pcis->vec3_D);CHKERRQ(ierr);}
  if (pcis->vec1_B)      {ierr = VecDestroy(pcis->vec1_B);CHKERRQ(ierr);}
  if (pcis->vec2_B)      {ierr = VecDestroy(pcis->vec2_B);CHKERRQ(ierr);}
  if (pcis->vec3_B)      {ierr = VecDestroy(pcis->vec3_B);CHKERRQ(ierr);}
  if (pcis->vec1_global) {ierr = VecDestroy(pcis->vec1_global);CHKERRQ(ierr);}
  if (pcis->global_to_D) {ierr = VecScatterDestroy(pcis->global_to_D);CHKERRQ(ierr);}
  if (pcis->N_to_B)      {ierr = VecScatterDestroy(pcis->N_to_B);CHKERRQ(ierr);}
  if (pcis->global_to_B) {ierr = VecScatterDestroy(pcis->global_to_B);CHKERRQ(ierr);}
  ierr = PetscFree(pcis->work_N);CHKERRQ(ierr);
  if (pcis->ISLocalToGlobalMappingGetInfoWasCalled) {
    ierr = ISLocalToGlobalMappingRestoreInfo((ISLocalToGlobalMapping)0,&(pcis->n_neigh),&(pcis->neigh),&(pcis->n_shared),&(pcis->shared));CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
/*
   PCISCreate - 
*/
#undef __FUNCT__  
#define __FUNCT__ "PCISCreate"
PetscErrorCode PETSCKSP_DLLEXPORT PCISCreate(PC pc)
{
  PC_IS *pcis = (PC_IS*)(pc->data);

  PetscFunctionBegin;
  pcis->is_B_local  = 0;
  pcis->is_I_local  = 0;
  pcis->is_B_global = 0;
  pcis->is_I_global = 0;
  pcis->A_II        = 0;
  pcis->A_IB        = 0;
  pcis->A_BI        = 0;
  pcis->A_BB        = 0;
  pcis->D           = 0;
  pcis->ksp_N      = 0;
  pcis->ksp_D      = 0;
  pcis->vec1_N      = 0;
  pcis->vec2_N      = 0;
  pcis->vec1_D      = 0;
  pcis->vec2_D      = 0;
  pcis->vec3_D      = 0;
  pcis->vec1_B      = 0;
  pcis->vec2_B      = 0;
  pcis->vec3_B      = 0;
  pcis->vec1_global = 0;
  pcis->work_N      = 0;
  pcis->global_to_D = 0;
  pcis->N_to_B      = 0;
  pcis->global_to_B = 0;
  pcis->ISLocalToGlobalMappingGetInfoWasCalled = PETSC_FALSE;
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
/*
   PCISApplySchur -

   Input parameters:
.  pc - preconditioner context
.  v - vector to which the Schur complement is to be applied (it is NOT modified inside this function, UNLESS vec2_B is null)

   Output parameters:
.  vec1_B - result of Schur complement applied to chunk
.  vec2_B - garbage (used as work space), or null (and v is used as workspace)
.  vec1_D - garbage (used as work space)
.  vec2_D - garbage (used as work space)

*/
#undef __FUNCT__  
#define __FUNCT__ "PCIterSuApplySchur"
PetscErrorCode PETSCKSP_DLLEXPORT PCISApplySchur(PC pc, Vec v, Vec vec1_B, Vec vec2_B, Vec vec1_D, Vec vec2_D)
{
  PetscErrorCode ierr;
  PC_IS          *pcis = (PC_IS*)(pc->data);

  PetscFunctionBegin;
  if (!vec2_B) { vec2_B = v; }

  ierr = MatMult(pcis->A_BB,v,vec1_B);CHKERRQ(ierr);
  ierr = MatMult(pcis->A_IB,v,vec1_D);CHKERRQ(ierr);
  ierr = KSPSolve(pcis->ksp_D,vec1_D,vec2_D);CHKERRQ(ierr);
  ierr = MatMult(pcis->A_BI,vec2_D,vec2_B);CHKERRQ(ierr);
  ierr = VecAXPY(vec1_B,-1.0,vec2_B);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
/*
   PCISScatterArrayNToVecB - Scatters interface node values from a big array (of all local nodes, interior or interface,
   including ghosts) into an interface vector, when in SCATTER_FORWARD mode, or vice-versa, when in SCATTER_REVERSE
   mode.

   Input parameters:
.  pc - preconditioner context
.  array_N - [when in SCATTER_FORWARD mode] Array to be scattered into the vector
.  v_B - [when in SCATTER_REVERSE mode] Vector to be scattered into the array

   Output parameter:
.  array_N - [when in SCATTER_REVERSE mode] Array to receive the scattered vector
.  v_B - [when in SCATTER_FORWARD mode] Vector to receive the scattered array

   Notes:
   The entries in the array that do not correspond to interface nodes remain unaltered.
*/
#undef __FUNCT__
#define __FUNCT__ "PCISScatterArrayNToVecB"
PetscErrorCode PETSCKSP_DLLEXPORT PCISScatterArrayNToVecB (PetscScalar *array_N, Vec v_B, InsertMode imode, ScatterMode smode, PC pc)
{
  PetscInt       i;
  const PetscInt *idex;
  PetscErrorCode ierr;
  PetscScalar    *array_B;
  PC_IS          *pcis = (PC_IS*)(pc->data);

  PetscFunctionBegin;
  ierr = VecGetArray(v_B,&array_B);CHKERRQ(ierr);
  ierr = ISGetIndices(pcis->is_B_local,&idex);CHKERRQ(ierr);

  if (smode == SCATTER_FORWARD) {
    if (imode == INSERT_VALUES) {
      for (i=0; i<pcis->n_B; i++) { array_B[i]  = array_N[idex[i]]; }
    } else {  /* ADD_VALUES */
      for (i=0; i<pcis->n_B; i++) { array_B[i] += array_N[idex[i]]; }
    }
  } else {  /* SCATTER_REVERSE */
    if (imode == INSERT_VALUES) {
      for (i=0; i<pcis->n_B; i++) { array_N[idex[i]]  = array_B[i]; }
    } else {  /* ADD_VALUES */
      for (i=0; i<pcis->n_B; i++) { array_N[idex[i]] += array_B[i]; }
    }
  }
  ierr = ISRestoreIndices(pcis->is_B_local,&idex);CHKERRQ(ierr);
  ierr = VecRestoreArray(v_B,&array_B);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
/*
   PCISApplyInvSchur - Solves the Neumann problem related to applying the inverse of the Schur complement.
   More precisely, solves the problem:
                                        [ A_II  A_IB ] [ . ]   [ 0 ]
                                        [            ] [   ] = [   ]
                                        [ A_BI  A_BB ] [ x ]   [ b ]

   Input parameters:
.  pc - preconditioner context
.  b - vector of local interface nodes (including ghosts)

   Output parameters:
.  x - vector of local interface nodes (including ghosts); returns the application of the inverse of the Schur
       complement to b
.  vec1_N - vector of local nodes (interior and interface, including ghosts); returns garbage (used as work space)
.  vec2_N - vector of local nodes (interior and interface, including ghosts); returns garbage (used as work space)

*/
#undef __FUNCT__
#define __FUNCT__ "PCISApplyInvSchur"
PetscErrorCode PETSCKSP_DLLEXPORT PCISApplyInvSchur (PC pc, Vec b, Vec x, Vec vec1_N, Vec vec2_N)
{
  PetscErrorCode ierr;
  PC_IS          *pcis = (PC_IS*)(pc->data);

  PetscFunctionBegin;
  /*
    Neumann solvers. 
    Applying the inverse of the local Schur complement, i.e, solving a Neumann
    Problem with zero at the interior nodes of the RHS and extracting the interface
    part of the solution. inverse Schur complement is applied to b and the result
    is stored in x.
  */
  /* Setting the RHS vec1_N */
  ierr = VecSet(vec1_N,0.0);CHKERRQ(ierr);
  ierr = VecScatterBegin(pcis->N_to_B,b,vec1_N,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  ierr = VecScatterEnd  (pcis->N_to_B,b,vec1_N,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  /* Checking for consistency of the RHS */
  {
    PetscTruth flg = PETSC_FALSE;
    ierr = PetscOptionsGetTruth(PETSC_NULL,"-pc_is_check_consistency",&flg,PETSC_NULL);CHKERRQ(ierr);
    if (flg) {
      PetscScalar average;
      PetscViewer viewer;
      ierr = PetscViewerASCIIGetStdout(((PetscObject)pc)->comm,&viewer);CHKERRQ(ierr);

      ierr = VecSum(vec1_N,&average);CHKERRQ(ierr);
      average = average / ((PetscReal)pcis->n);
      if (pcis->pure_neumann) {

        ierr = PetscViewerASCIISynchronizedPrintf(viewer,"Subdomain %04d is floating. Average = % 1.14e\n",
                                             PetscGlobalRank,PetscAbsScalar(average));CHKERRQ(ierr);
      } else {
        ierr = PetscViewerASCIISynchronizedPrintf(viewer,"Subdomain %04d is fixed.    Average = % 1.14e\n",
                                             PetscGlobalRank,PetscAbsScalar(average));CHKERRQ(ierr);
      }
      ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
    }
  }
  /* Solving the system for vec2_N */
  ierr = KSPSolve(pcis->ksp_N,vec1_N,vec2_N);CHKERRQ(ierr);
  /* Extracting the local interface vector out of the solution */
  ierr = VecScatterBegin(pcis->N_to_B,vec2_N,x,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd  (pcis->N_to_B,vec2_N,x,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}









