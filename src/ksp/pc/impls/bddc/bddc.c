
#include "bddc.h"

/*
   Implementation of BDDC preconditioner based on:
   C. Dohrmann "An approximate BDDC preconditioner", Numerical Linear Algebra with Applications Volume 14, Issue 2, pages 149-168, March 2007
*/

/* -------------------------------------------------------------------------- */
#undef __FUNCT__  
#define __FUNCT__ "PCSetFromOptions_BDDC"
PetscErrorCode PCSetFromOptions_BDDC(PC pc)
{
  PC_BDDC         *pcbddc = (PC_BDDC*)pc->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscOptionsHead("BDDC options");CHKERRQ(ierr);
  /* Verbose debugging of main data structures */
  ierr = PetscOptionsBool("-pc_bddc_check_all"       ,"Verbose (debugging) output for PCBDDC"                            ,"none",pcbddc->check_flag      ,&pcbddc->check_flag      ,PETSC_NULL);CHKERRQ(ierr);
  /* Some customization for default primal space */
  ierr = PetscOptionsBool("-pc_bddc_vertices_only"   ,"Use vertices only in coarse space (i.e. discard constraints)","none",pcbddc->vertices_flag   ,&pcbddc->vertices_flag   ,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-pc_bddc_constraints_only","Use constraints only in coarse space (i.e. discard vertices)","none",pcbddc->constraints_flag,&pcbddc->constraints_flag,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-pc_bddc_faces_only"      ,"Use faces only in coarse space (i.e. discard edges)"         ,"none",pcbddc->faces_flag      ,&pcbddc->faces_flag      ,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-pc_bddc_edges_only"      ,"Use edges only in coarse space (i.e. discard faces)"         ,"none",pcbddc->edges_flag      ,&pcbddc->edges_flag      ,PETSC_NULL);CHKERRQ(ierr);
  /* Coarse solver context */
  static const char *avail_coarse_problems[] = {"sequential","replicated","parallel","multilevel",""}; //order of choiches depends on ENUM defined in bddc.h
  ierr = PetscOptionsEnum("-pc_bddc_coarse_problem_type","Set coarse problem type","none",avail_coarse_problems,(PetscEnum)pcbddc->coarse_problem_type,(PetscEnum*)&pcbddc->coarse_problem_type,PETSC_NULL);CHKERRQ(ierr);
  /* Two different application of BDDC to the whole set of dofs, internal and interface */
  ierr = PetscOptionsBool("-pc_bddc_switch_preconditioning_type","Switch between M_2 (default) and M_3 preconditioners (as defined by Dohrmann)","none",pcbddc->prec_type,&pcbddc->prec_type,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-pc_bddc_coarsening_ratio","Set coarsening ratio used in multilevel coarsening","none",pcbddc->coarsening_ratio,&pcbddc->coarsening_ratio,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PCBDDCSetCoarseProblemType_BDDC"
PetscErrorCode PCBDDCSetCoarseProblemType_BDDC(PC pc, CoarseProblemType CPT)
{
  PC_BDDC  *pcbddc = (PC_BDDC*)pc->data;

  PetscFunctionBegin;
  pcbddc->coarse_problem_type = CPT; 
  PetscFunctionReturn(0);
}
EXTERN_C_END

/* -------------------------------------------------------------------------- */
#undef __FUNCT__  
#define __FUNCT__ "PCBDDCSetCoarseProblemType"
/*
SZ INSERT COMMENT HERE 
*/
PetscErrorCode PCBDDCSetCoarseProblemType(PC pc, CoarseProblemType CPT)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  ierr = PetscTryMethod(pc,"PCBDDCSetCoarseProblemType_C",(PC,CoarseProblemType),(pc,CPT));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PCBDDCSetNeumannBoundaries_BDDC"
PetscErrorCode PCBDDCSetNeumannBoundaries_BDDC(PC pc,Vec input_vec)
{
  PC_BDDC  *pcbddc = (PC_BDDC*)pc->data;

  PetscFunctionBegin;
  VecDuplicate(input_vec,&pcbddc->Vec_Neumann);
  VecCopy(input_vec,pcbddc->Vec_Neumann);
  PetscFunctionReturn(0);
}
EXTERN_C_END

/* -------------------------------------------------------------------------- */
#undef __FUNCT__  
#define __FUNCT__ "PCBDDCSetNeumannBoundaries"
/*@
   PCBDDCSetNeumannBoundaries - brief explanation

   Collective on PC

   Input Parameters:
+  pc - the preconditioning context
-  input_vec - pick a better name and explain what this is

   Level: intermediate

   Notes:
   usage notes, perhaps an example

.seealso: PCBDDC
@*/
PetscErrorCode PCBDDCSetNeumannBoundaries(PC pc,Vec input_vec)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  ierr = PetscTryMethod(pc,"PCBDDCSetNeumannBoundaries_C",(PC,Vec),(pc,input_vec));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
/*
   PCSetUp_BDDC - Prepares for the use of the BDDC preconditioner
                    by setting data structures and options.   

   Input Parameter:
.  pc - the preconditioner context

   Application Interface Routine: PCSetUp()

   Notes:
   The interface routine PCSetUp() is not usually called directly by
   the user, but instead is called by PCApply() if necessary.
*/
#undef __FUNCT__  
#define __FUNCT__ "PCSetUp_BDDC"
static PetscErrorCode PCSetUp_BDDC(PC pc)
{
  PetscErrorCode ierr;
  PC_BDDC*       pcbddc   = (PC_BDDC*)pc->data;
  PC_IS            *pcis = (PC_IS*)(pc->data);

  PetscFunctionBegin;
  if (!pc->setupcalled) {

    /* For BDDC we need to define a local "Neumann" to that defined in PCISSetup
       So, we set to pcnone the Neumann problem of pcid in order to avoid unneeded computation
       Also, we decide to directly build the (same) Dirichlet problem */
    ierr = PetscOptionsSetValue("-is_localN_pc_type","none");CHKERRQ(ierr);
    ierr = PetscOptionsSetValue("-is_localD_pc_type","none");CHKERRQ(ierr);
    /* Set up all the "iterative substructuring" common block */
    ierr = PCISSetUp(pc);CHKERRQ(ierr);
    /* Destroy some PC_IS data which is not needed by BDDC */
    //if (pcis->ksp_D)  {ierr = KSPDestroy(&pcis->ksp_D);CHKERRQ(ierr);}
    //if (pcis->ksp_N)  {ierr = KSPDestroy(&pcis->ksp_N);CHKERRQ(ierr);}
    //if (pcis->vec2_B) {ierr = VecDestroy(&pcis->vec2_B);CHKERRQ(ierr);}
    //if (pcis->vec3_B) {ierr = VecDestroy(&pcis->vec3_B);CHKERRQ(ierr);}
    //pcis->ksp_D  = 0;
    //pcis->ksp_N  = 0;
    //pcis->vec2_B = 0;
    //pcis->vec3_B = 0;

    //TODO MOVE CODE FRAGMENT
    PetscInt im_active=0;
    if(pcis->n) im_active = 1;
    MPI_Allreduce(&im_active,&pcbddc->active_procs,1,MPIU_INT,MPI_SUM,((PetscObject)pc)->comm);
    //printf("Calling PCBDDC MANAGE with active procs %d (im_active = %d)\n",pcbddc->active_procs,im_active);
    /* Set up grid quantities for BDDC */
    //TODO only active procs must call this -> remove synchronized print inside (the only point of sync)
    ierr = PCBDDCManageLocalBoundaries(pc);CHKERRQ(ierr); 

    /* Create coarse and local stuffs used for evaluating action of preconditioner */
    ierr = PCBDDCCoarseSetUp(pc);CHKERRQ(ierr);

    if ( !pcis->n_neigh ) pcis->ISLocalToGlobalMappingGetInfoWasCalled=PETSC_FALSE; //processes fakely involved in multilevel should not call ISLocalToGlobalMappingRestoreInfo  
    /* We no more need this matrix */
    //if (pcis->A_BB)  {ierr = MatDestroy(&pcis->A_BB);CHKERRQ(ierr);}
    //pcis->A_BB   = 0;

    //printf("Using coarse problem type %d\n",pcbddc->coarse_problem_type);
    //printf("Using coarse communications type %d\n",pcbddc->coarse_communications_type);
    //printf("Using coarsening ratio %d\n",pcbddc->coarsening_ratio);
  }

  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
/*
   PCApply_BDDC - Applies the BDDC preconditioner to a vector.

   Input Parameters:
.  pc - the preconditioner context
.  r - input vector (global)

   Output Parameter:
.  z - output vector (global)

   Application Interface Routine: PCApply()
 */
#undef __FUNCT__  
#define __FUNCT__ "PCApply_BDDC"
static PetscErrorCode PCApply_BDDC(PC pc,Vec r,Vec z)
{
  PC_IS          *pcis = (PC_IS*)(pc->data);
  PC_BDDC        *pcbddc = (PC_BDDC*)(pc->data);
  PetscErrorCode ierr;
  PetscScalar    one = 1.0;
  PetscScalar    m_one = -1.0;

/* This code is similar to that provided in nn.c for PCNN
   NN interface preconditioner changed to BDDC
   Added support for M_3 preconditioenr in the reference article (code is active if pcbddc->prec_type = PETSC_TRUE) */

  PetscFunctionBegin;
  /* First Dirichlet solve */
  ierr = VecScatterBegin(pcis->global_to_D,r,pcis->vec1_D,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd  (pcis->global_to_D,r,pcis->vec1_D,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = PCApply(pcbddc->pc_D,pcis->vec1_D,pcis->vec2_D);CHKERRQ(ierr);
  /*
    Assembling right hand side for BDDC operator
    - vec1_D for the Dirichlet part (if needed, i.e. prec_flag=PETSC_TRUE)
    - the interface part of the global vector z
  */
  //ierr = VecScatterBegin(pcis->global_to_B,r,pcis->vec1_B,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  //ierr = VecScatterEnd  (pcis->global_to_B,r,pcis->vec1_B,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScale(pcis->vec2_D,m_one);CHKERRQ(ierr);
  ierr = MatMult(pcis->A_BI,pcis->vec2_D,pcis->vec1_B);CHKERRQ(ierr);
  //ierr = MatMultAdd(pcis->A_BI,pcis->vec2_D,pcis->vec1_B,pcis->vec1_B);CHKERRQ(ierr);
  if(pcbddc->prec_type) { ierr = MatMultAdd(pcis->A_II,pcis->vec2_D,pcis->vec1_D,pcis->vec1_D);CHKERRQ(ierr); }
  ierr = VecScale(pcis->vec2_D,m_one);CHKERRQ(ierr);
  ierr = VecCopy(r,z);CHKERRQ(ierr);
  ierr = VecScatterBegin(pcis->global_to_B,pcis->vec1_B,z,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  ierr = VecScatterEnd  (pcis->global_to_B,pcis->vec1_B,z,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);

  /*
    Apply interface preconditioner
    Results are stored in:
    -  vec1_D (if needed, i.e. with prec_type = PETSC_TRUE)
    -  the interface part of the global vector z
  */
  ierr = PCBDDCApplyInterfacePreconditioner(pc,z); CHKERRQ(ierr);

  /* Second Dirichlet solves and assembling of output */
  ierr = VecScatterBegin(pcis->global_to_B,z,pcis->vec1_B,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd  (pcis->global_to_B,z,pcis->vec1_B,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = MatMult(pcis->A_IB,pcis->vec1_B,pcis->vec3_D);CHKERRQ(ierr);
  if(pcbddc->prec_type) { ierr = MatMultAdd(pcis->A_II,pcis->vec1_D,pcis->vec3_D,pcis->vec3_D);CHKERRQ(ierr); }
  ierr = PCApply(pcbddc->pc_D,pcis->vec3_D,pcbddc->vec4_D);CHKERRQ(ierr);
  ierr = VecScale(pcbddc->vec4_D,m_one);CHKERRQ(ierr);
  if(pcbddc->prec_type) { ierr = VecAXPY (pcbddc->vec4_D,one,pcis->vec1_D);CHKERRQ(ierr); } 
  ierr = VecAXPY (pcis->vec2_D,one,pcbddc->vec4_D);CHKERRQ(ierr); 
  ierr = VecScatterBegin(pcis->global_to_D,pcis->vec2_D,z,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  ierr = VecScatterEnd  (pcis->global_to_D,pcis->vec2_D,z,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);

  PetscFunctionReturn(0);

}

/* -------------------------------------------------------------------------- */
/*
   PCBDDCApplyInterfacePreconditioner - Apply the BDDC preconditioner at the interface. 
    
*/
#undef __FUNCT__
#define __FUNCT__ "PCBDDCApplyInterfacePreconditioner"
PetscErrorCode  PCBDDCApplyInterfacePreconditioner (PC pc, Vec z)
{ 
  PetscErrorCode ierr;
  PC_BDDC*       pcbddc = (PC_BDDC*)(pc->data);
  PC_IS*         pcis =   (PC_IS*)  (pc->data);
  PetscScalar    zero = 0.0;

  PetscFunctionBegin;
  /* Get Local boundary and apply partition of unity */
  ierr = VecScatterBegin(pcis->global_to_B,z,pcis->vec1_B,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd  (pcis->global_to_B,z,pcis->vec1_B,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecPointwiseMult(pcis->vec1_B,pcis->D,pcis->vec1_B);CHKERRQ(ierr);

  /* Application of PHI^T  */
  ierr = MatMultTranspose(pcbddc->coarse_phi_B,pcis->vec1_B,pcbddc->vec1_P);CHKERRQ(ierr);
  if(pcbddc->prec_type) { ierr = MatMultTransposeAdd(pcbddc->coarse_phi_D,pcis->vec1_D,pcbddc->vec1_P,pcbddc->vec1_P);CHKERRQ(ierr); }

  /* Scatter data of coarse_rhs */
  if(pcbddc->coarse_rhs) ierr = VecSet(pcbddc->coarse_rhs,zero);CHKERRQ(ierr);
  ierr = PCBDDCScatterCoarseDataBegin(pc,pcbddc->vec1_P,pcbddc->coarse_rhs,ADD_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);

  /* Local solution on R nodes */
  ierr = VecSet(pcbddc->vec1_R,zero);CHKERRQ(ierr);
  ierr = VecScatterBegin(pcbddc->R_to_B,pcis->vec1_B,pcbddc->vec1_R,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  ierr = VecScatterEnd  (pcbddc->R_to_B,pcis->vec1_B,pcbddc->vec1_R,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  if(pcbddc->prec_type) {
    ierr = VecScatterBegin(pcbddc->R_to_D,pcis->vec1_D,pcbddc->vec1_R,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
    ierr = VecScatterEnd  (pcbddc->R_to_D,pcis->vec1_D,pcbddc->vec1_R,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  }
  ierr = PCBDDCSolveSaddlePoint(pc);CHKERRQ(ierr);
  ierr = VecSet(pcis->vec1_B,zero);CHKERRQ(ierr);
  ierr = VecScatterBegin(pcbddc->R_to_B,pcbddc->vec2_R,pcis->vec1_B,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd  (pcbddc->R_to_B,pcbddc->vec2_R,pcis->vec1_B,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  if(pcbddc->prec_type) {
    ierr = VecScatterBegin(pcbddc->R_to_D,pcbddc->vec2_R,pcis->vec1_D,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterEnd  (pcbddc->R_to_D,pcbddc->vec2_R,pcis->vec1_D,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  }

  /* Coarse solution */
  ierr = PCBDDCScatterCoarseDataEnd(pc,pcbddc->vec1_P,pcbddc->coarse_rhs,ADD_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  if(pcbddc->coarse_rhs) ierr = PCApply(pcbddc->coarse_pc,pcbddc->coarse_rhs,pcbddc->coarse_vec);CHKERRQ(ierr);
  ierr = PCBDDCScatterCoarseDataBegin(pc,pcbddc->coarse_vec,pcbddc->vec1_P,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  ierr = PCBDDCScatterCoarseDataEnd  (pc,pcbddc->coarse_vec,pcbddc->vec1_P,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);

  /* Sum contributions from two levels */
  /* Apply partition of unity and sum boundary values */
  ierr = MatMultAdd(pcbddc->coarse_phi_B,pcbddc->vec1_P,pcis->vec1_B,pcis->vec1_B);CHKERRQ(ierr);
  ierr = VecPointwiseMult(pcis->vec1_B,pcis->D,pcis->vec1_B);CHKERRQ(ierr);
  if(pcbddc->prec_type) { ierr = MatMultAdd(pcbddc->coarse_phi_D,pcbddc->vec1_P,pcis->vec1_D,pcis->vec1_D);CHKERRQ(ierr); }
  ierr = VecSet(z,zero);CHKERRQ(ierr);
  ierr = VecScatterBegin(pcis->global_to_B,pcis->vec1_B,z,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  ierr = VecScatterEnd  (pcis->global_to_B,pcis->vec1_B,z,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}


/* -------------------------------------------------------------------------- */
/*
   PCBDDCSolveSaddlePoint 
    
*/
#undef __FUNCT__
#define __FUNCT__ "PCBDDCSolveSaddlePoint"
PetscErrorCode  PCBDDCSolveSaddlePoint(PC pc)

{ 
  PetscErrorCode ierr;
  PC_BDDC*       pcbddc = (PC_BDDC*)(pc->data);

  PetscFunctionBegin;

  ierr = PCApply(pcbddc->pc_R,pcbddc->vec1_R,pcbddc->vec2_R);CHKERRQ(ierr);
  if(pcbddc->n_constraints) {
    ierr = MatMult(pcbddc->local_auxmat1,pcbddc->vec2_R,pcbddc->vec1_C);CHKERRQ(ierr);
    ierr = MatMultAdd(pcbddc->local_auxmat2,pcbddc->vec1_C,pcbddc->vec2_R,pcbddc->vec2_R);CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);

}
/* -------------------------------------------------------------------------- */
/* -------------------------------------------------------------------------- */
/*
   PCBDDCScatterCoarseDataBegin  
    
*/
#undef __FUNCT__
#define __FUNCT__ "PCBDDCScatterCoarseDataBegin"
PetscErrorCode  PCBDDCScatterCoarseDataBegin(PC pc,Vec vec_from, Vec vec_to, InsertMode imode, ScatterMode smode)
{ 
  PetscErrorCode ierr;
  PC_BDDC*       pcbddc = (PC_BDDC*)(pc->data);

  PetscFunctionBegin;

  switch(pcbddc->coarse_communications_type){
    case SCATTERS_BDDC:
      ierr = VecScatterBegin(pcbddc->coarse_loc_to_glob,vec_from,vec_to,imode,smode);CHKERRQ(ierr);
      break;
    case GATHERS_BDDC:
      break;
  }
  PetscFunctionReturn(0);

}
/* -------------------------------------------------------------------------- */
/*
   PCBDDCScatterCoarseDataEnd  
    
*/
#undef __FUNCT__
#define __FUNCT__ "PCBDDCScatterCoarseDataEnd"
PetscErrorCode  PCBDDCScatterCoarseDataEnd(PC pc,Vec vec_from, Vec vec_to, InsertMode imode, ScatterMode smode)
{ 
  PetscErrorCode ierr;
  PC_BDDC*       pcbddc = (PC_BDDC*)(pc->data);
  PetscScalar*   array_to;
  PetscScalar*   array_from;
  MPI_Comm       comm=((PetscObject)pc)->comm;
  PetscInt i;

  PetscFunctionBegin;

  switch(pcbddc->coarse_communications_type){
    case SCATTERS_BDDC:
      ierr = VecScatterEnd(pcbddc->coarse_loc_to_glob,vec_from,vec_to,imode,smode);CHKERRQ(ierr);
      break;
    case GATHERS_BDDC:
      if(vec_from) VecGetArray(vec_from,&array_from);
      if(vec_to)   VecGetArray(vec_to,&array_to);
      switch(pcbddc->coarse_problem_type){
        case SEQUENTIAL_BDDC:
          if(smode == SCATTER_FORWARD) {
            MPI_Gatherv(&array_from[0],pcbddc->local_primal_size,MPIU_SCALAR,&pcbddc->replicated_local_primal_values[0],pcbddc->local_primal_sizes,pcbddc->local_primal_displacements,MPIU_SCALAR,0,comm);
            if(vec_to) {
              for(i=0;i<pcbddc->replicated_primal_size;i++)
                array_to[pcbddc->replicated_local_primal_indices[i]]+=pcbddc->replicated_local_primal_values[i];
            }
          } else {
            if(vec_from)
              for(i=0;i<pcbddc->replicated_primal_size;i++)
                pcbddc->replicated_local_primal_values[i]=array_from[pcbddc->replicated_local_primal_indices[i]];
            MPI_Scatterv(&pcbddc->replicated_local_primal_values[0],pcbddc->local_primal_sizes,pcbddc->local_primal_displacements,MPIU_SCALAR,&array_to[0],pcbddc->local_primal_size,MPIU_SCALAR,0,comm);
          }
          break;
        case REPLICATED_BDDC:
          if(smode == SCATTER_FORWARD) {
            MPI_Allgatherv(&array_from[0],pcbddc->local_primal_size,MPIU_SCALAR,&pcbddc->replicated_local_primal_values[0],pcbddc->local_primal_sizes,pcbddc->local_primal_displacements,MPIU_SCALAR,comm);
            for(i=0;i<pcbddc->replicated_primal_size;i++)
              array_to[pcbddc->replicated_local_primal_indices[i]]+=pcbddc->replicated_local_primal_values[i];
          } else { /* no communications needed for SCATTER_REVERSE since needed data is already present */
            for(i=0;i<pcbddc->local_primal_size;i++)
              array_to[i]=array_from[pcbddc->local_primal_indices[i]];
          }
          break;
      }
      if(vec_from) VecRestoreArray(vec_from,&array_from);
      if(vec_to)   VecRestoreArray(vec_to,&array_to);
      break;
  }
  PetscFunctionReturn(0);

}

/* -------------------------------------------------------------------------- */
/*
   PCDestroy_BDDC - Destroys the private context for the NN preconditioner
   that was created with PCCreate_BDDC().

   Input Parameter:
.  pc - the preconditioner context

   Application Interface Routine: PCDestroy()
*/
#undef __FUNCT__  
#define __FUNCT__ "PCDestroy_BDDC"
static PetscErrorCode PCDestroy_BDDC(PC pc)
{
  PC_BDDC          *pcbddc = (PC_BDDC*)pc->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* free data created by PCIS */
  ierr = PCISDestroy(pc);CHKERRQ(ierr);
  /* free BDDC data  */
  if (pcbddc->coarse_vec)         {ierr = VecDestroy(&pcbddc->coarse_vec);CHKERRQ(ierr);}
  if (pcbddc->coarse_rhs)         {ierr = VecDestroy(&pcbddc->coarse_rhs);CHKERRQ(ierr);}
  if (pcbddc->coarse_pc)          {ierr = PCDestroy(&pcbddc->coarse_pc);CHKERRQ(ierr);}
  if (pcbddc->coarse_mat)         {ierr = MatDestroy(&pcbddc->coarse_mat);CHKERRQ(ierr);}
  if (pcbddc->coarse_phi_B)       {ierr = MatDestroy(&pcbddc->coarse_phi_B);CHKERRQ(ierr);}
  if (pcbddc->coarse_phi_D)       {ierr = MatDestroy(&pcbddc->coarse_phi_D);CHKERRQ(ierr);}
  if (pcbddc->vec1_P)             {ierr = VecDestroy(&pcbddc->vec1_P);CHKERRQ(ierr);}
  if (pcbddc->vec1_C)             {ierr = VecDestroy(&pcbddc->vec1_C);CHKERRQ(ierr);}
  if (pcbddc->local_auxmat1)      {ierr = MatDestroy(&pcbddc->local_auxmat1);CHKERRQ(ierr);}
  if (pcbddc->local_auxmat2)      {ierr = MatDestroy(&pcbddc->local_auxmat2);CHKERRQ(ierr);}
  if (pcbddc->vec1_R)             {ierr = VecDestroy(&pcbddc->vec1_R);CHKERRQ(ierr);}
  if (pcbddc->vec2_R)             {ierr = VecDestroy(&pcbddc->vec2_R);CHKERRQ(ierr);}
  if (pcbddc->vec4_D)             {ierr = VecDestroy(&pcbddc->vec4_D);CHKERRQ(ierr);}
  if (pcbddc->R_to_B)             {ierr = VecScatterDestroy(&pcbddc->R_to_B);CHKERRQ(ierr);}
  if (pcbddc->R_to_D)             {ierr = VecScatterDestroy(&pcbddc->R_to_D);CHKERRQ(ierr);}
  if (pcbddc->coarse_loc_to_glob) {ierr = VecScatterDestroy(&pcbddc->coarse_loc_to_glob);CHKERRQ(ierr);}
  if (pcbddc->pc_D)               {ierr = PCDestroy(&pcbddc->pc_D);CHKERRQ(ierr);}
  if (pcbddc->pc_R)               {ierr = PCDestroy(&pcbddc->pc_R);CHKERRQ(ierr);}
  if (pcbddc->Vec_Neumann)        {ierr = VecDestroy(&pcbddc->Vec_Neumann);CHKERRQ(ierr);}
  if (pcbddc->vertices)           {ierr = PetscFree(pcbddc->vertices);CHKERRQ(ierr);}
  if (pcbddc->local_primal_indices)              { ierr = PetscFree(pcbddc->local_primal_indices);CHKERRQ(ierr);}
  if (pcbddc->replicated_local_primal_indices)   { ierr = PetscFree(pcbddc->replicated_local_primal_indices);CHKERRQ(ierr);}
  //if (pcbddc->replicated_local_primal_values)    { ierr = PetscFree(pcbddc->replicated_local_primal_values);CHKERRQ(ierr);}
  if (pcbddc->replicated_local_primal_values)    { free(pcbddc->replicated_local_primal_values); }
  if (pcbddc->local_primal_displacements)        { ierr = PetscFree(pcbddc->local_primal_displacements);CHKERRQ(ierr);}
  if (pcbddc->local_primal_sizes)                { ierr = PetscFree(pcbddc->local_primal_sizes);CHKERRQ(ierr);}
  if (pcbddc->n_constraints) {
    ierr = PetscFree(pcbddc->indices_to_constraint[0]);CHKERRQ(ierr);
    ierr = PetscFree(pcbddc->indices_to_constraint);CHKERRQ(ierr);
    ierr = PetscFree(pcbddc->quadrature_constraint[0]);CHKERRQ(ierr);
    ierr = PetscFree(pcbddc->quadrature_constraint);CHKERRQ(ierr);
    ierr = PetscFree(pcbddc->sizes_of_constraint);CHKERRQ(ierr);
  }
  /* Free the private data structure that was hanging off the PC */
  ierr = PetscFree(pcbddc);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
/*MC
   PCBDDC - Balancing Domain Decomposition by Constraints.

   Options Database Keys:
.    -pc_is_damp_fixed <fact> -
.    -pc_is_remove_nullspace_fixed -
.    -pc_is_set_damping_factor_floating <fact> -
.    -pc_is_not_damp_floating -

   Level: intermediate

   Notes: The matrix used with this preconditioner must be of type MATIS 

          Unlike more 'conventional' interface preconditioners, this iterates over ALL the
          degrees of freedom, NOT just those on the interface (this allows the use of approximate solvers
          on the subdomains).

          Options for the coarse grid preconditioner can be set with -pc_bddc_coarse_pc_xxx
          Options for the Dirichlet subproblem can be set with -pc_bddc_localD_xxx
          Options for the Neumann subproblem can be set with -pc_bddc_localN_xxx

   Contributed by Stefano Zampini

.seealso:  PCCreate(), PCSetType(), PCType (for list of available types), PC,  MATIS
M*/
EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PCCreate_BDDC"
PetscErrorCode PCCreate_BDDC(PC pc)
{
  PetscErrorCode ierr;
  PC_BDDC          *pcbddc;

  PetscFunctionBegin;
  /* Creates the private data structure for this preconditioner and attach it to the PC object. */
  ierr      = PetscNewLog(pc,PC_BDDC,&pcbddc);CHKERRQ(ierr);
  pc->data  = (void*)pcbddc;
  /* create PCIS data structure */
  ierr = PCISCreate(pc);CHKERRQ(ierr);
  /* BDDC specific */
  pcbddc->coarse_vec                 = 0;
  pcbddc->coarse_rhs                 = 0;
  pcbddc->coarse_pc                  = 0;
  pcbddc->coarse_phi_B               = 0;
  pcbddc->coarse_phi_D               = 0;
  pcbddc->vec1_P                     = 0;          
  pcbddc->vec1_R                     = 0; 
  pcbddc->vec2_R                     = 0; 
  pcbddc->local_auxmat1              = 0;
  pcbddc->local_auxmat2              = 0;
  pcbddc->R_to_B                     = 0;
  pcbddc->R_to_D                     = 0;
  pcbddc->pc_D                       = 0;
  pcbddc->pc_R                       = 0;
  pcbddc->n_constraints              = 0;
  pcbddc->n_vertices                 = 0;
  pcbddc->vertices                   = 0;
  pcbddc->indices_to_constraint      = 0;
  pcbddc->quadrature_constraint      = 0;
  pcbddc->sizes_of_constraint        = 0;
  pcbddc->local_primal_indices       = 0;
  pcbddc->prec_type                  = PETSC_FALSE;
  pcbddc->Vec_Neumann                = 0;
  pcbddc->local_primal_sizes         = 0;
  pcbddc->local_primal_displacements = 0;
  pcbddc->replicated_local_primal_indices = 0;
  pcbddc->replicated_local_primal_values  = 0;
  pcbddc->coarse_loc_to_glob         = 0;
  pcbddc->check_flag                 = PETSC_FALSE;
  pcbddc->vertices_flag              = PETSC_FALSE;
  pcbddc->constraints_flag           = PETSC_FALSE;
  pcbddc->faces_flag                 = PETSC_FALSE;
  pcbddc->edges_flag                 = PETSC_FALSE;
  pcbddc->coarsening_ratio           = 8;
  /* function pointers */
  pc->ops->apply               = PCApply_BDDC;
  pc->ops->applytranspose      = 0;
  pc->ops->setup               = PCSetUp_BDDC;
  pc->ops->destroy             = PCDestroy_BDDC;
  pc->ops->setfromoptions      = PCSetFromOptions_BDDC;
  pc->ops->view                = 0;
  pc->ops->applyrichardson     = 0;
  pc->ops->applysymmetricleft  = 0;
  pc->ops->applysymmetricright = 0;
  /* composing function */
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCBDDCSetNeumannBoundaries_C","PCBDDCSetNeumannBoundaries_BDDC",
                    PCBDDCSetNeumannBoundaries_BDDC);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCBDDCSetCoarseProblemType_C","PCBDDCSetCoarseProblemType_BDDC",
                    PCBDDCSetCoarseProblemType_BDDC);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END

/* -------------------------------------------------------------------------- */
/*  
   PCBDDCCoarseSetUp - 
*/
#undef __FUNCT__  
#define __FUNCT__ "PCBDDCCoarseSetUp"
PetscErrorCode PCBDDCCoarseSetUp(PC pc)
{   
  PetscErrorCode  ierr;

  PC_IS*            pcis = (PC_IS*)(pc->data);
  PC_BDDC*          pcbddc = (PC_BDDC*)pc->data;
  Mat_IS            *matis = (Mat_IS*)pc->pmat->data; 
  IS                is_R_local;
  IS                is_V_local;
  IS                is_C_local;
  IS                is_aux1;
  IS                is_aux2;
  PetscViewer       viewer;
  PetscBool         check_flag=pcbddc->check_flag;
  const VecType     impVecType;
  const MatType     impMatType;
  PetscInt          n_R=0;
  PetscInt          n_D=0;
  PetscInt          n_B=0;
  PetscMPIInt       totprocs;
  PetscScalar       zero=0.0;
  PetscScalar       one=1.0;
  PetscScalar       m_one=-1.0;
  PetscScalar*      array;
  PetscScalar       *coarse_submat_vals;
  PetscInt          *idx_R_local;
  PetscInt          *idx_V_B;
  PetscScalar       *coarsefunctions_errors;
  PetscScalar       *constraints_errors;
  /* auxiliary indices */
  PetscInt s,i,j,k;
  
  PetscFunctionBegin;
//  if ( !pcbddc->n_vertices && !pcbddc->n_constraints ) {
//    SETERRQ(((PetscObject)pc)->comm,PETSC_ERR_ARG_WRONG,"BDDC preconditioner needs vertices and/or constraints!");
//  }
  if(pcbddc->check_flag) {
    ierr = PetscViewerASCIIGetStdout(((PetscObject)pc)->comm,&viewer);CHKERRQ(ierr);
    ierr = PetscViewerASCIISynchronizedAllow(viewer,PETSC_TRUE);CHKERRQ(ierr);
  }

  /* Set Non-overlapping dimensions */
  n_B = pcis->n_B; n_D = pcis->n - n_B;
  /* Set local primal size */
  pcbddc->local_primal_size = pcbddc->n_vertices + pcbddc->n_constraints;
  /* Dohrmann's notation: dofs splitted in R (Remaining: all dofs but the vertices) and V (Vertices) */
  ierr = VecSet(pcis->vec1_N,one);CHKERRQ(ierr);
  ierr = VecGetArray(pcis->vec1_N,&array);CHKERRQ(ierr);
  for (i=0;i<pcbddc->n_vertices;i++) { array[ pcbddc->vertices[i] ] = zero; }
  ierr = PetscMalloc(( pcis->n - pcbddc->n_vertices )*sizeof(PetscInt),&idx_R_local);CHKERRQ(ierr);
  for (i=0, n_R=0; i<pcis->n; i++) { if (array[i] == one) { idx_R_local[n_R] = i; n_R++; } } 
  ierr = VecRestoreArray(pcis->vec1_N,&array);CHKERRQ(ierr);
  if(check_flag) {
    ierr = PetscViewerASCIIPrintf(viewer,"--------------------------------------------------\n");CHKERRQ(ierr);
    ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
    ierr = PetscViewerASCIISynchronizedPrintf(viewer,"Subdomain %04d local dimensions\n",PetscGlobalRank);CHKERRQ(ierr);
    ierr = PetscViewerASCIISynchronizedPrintf(viewer,"local_size = %d, dirichlet_size = %d, boundary_size = %d\n",pcis->n,n_D,n_B);CHKERRQ(ierr);
    ierr = PetscViewerASCIISynchronizedPrintf(viewer,"r_size = %d, v_size = %d, constraints = %d, local_primal_size = %d\n",n_R,pcbddc->n_vertices,pcbddc->n_constraints,pcbddc->local_primal_size);CHKERRQ(ierr);
    ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
  }
  /* Allocate needed vectors */
  /* Set Mat type for local matrices needed by BDDC precondtioner */
//  ierr = MatGetType(matis->A,&impMatType);CHKERRQ(ierr);
  //ierr = VecGetType(pcis->vec1_N,&impVecType);CHKERRQ(ierr);
  impMatType = MATSEQDENSE;
  impVecType = VECSEQ;
  ierr = VecDuplicate(pcis->vec1_D,&pcbddc->vec4_D);CHKERRQ(ierr);
  ierr = VecDuplicate(pcis->vec1_N,&pcis->vec2_N);CHKERRQ(ierr);
  ierr = VecCreate(PETSC_COMM_SELF,&pcbddc->vec1_R);CHKERRQ(ierr);
  ierr = VecSetSizes(pcbddc->vec1_R,n_R,n_R);CHKERRQ(ierr);
  ierr = VecSetType(pcbddc->vec1_R,impVecType);CHKERRQ(ierr);
  ierr = VecDuplicate(pcbddc->vec1_R,&pcbddc->vec2_R);
  ierr = VecCreate(PETSC_COMM_SELF,&pcbddc->vec1_P);CHKERRQ(ierr);
  ierr = VecSetSizes(pcbddc->vec1_P,pcbddc->local_primal_size,pcbddc->local_primal_size);CHKERRQ(ierr);
  ierr = VecSetType(pcbddc->vec1_P,impVecType);CHKERRQ(ierr);

  /* Creating some index sets needed  */
  /* For submatrices */
  ierr = ISCreateGeneral(PETSC_COMM_SELF,n_R,idx_R_local,PETSC_COPY_VALUES,&is_R_local);CHKERRQ(ierr);
  if(pcbddc->n_vertices)    { ierr = ISCreateGeneral(PETSC_COMM_SELF,pcbddc->n_vertices,pcbddc->vertices,PETSC_COPY_VALUES,&is_V_local);CHKERRQ(ierr); }
  if(pcbddc->n_constraints) { ierr = ISCreateStride (PETSC_COMM_SELF,pcbddc->n_constraints,pcbddc->n_vertices,1,&is_C_local);CHKERRQ(ierr); }
  /* For VecScatters pcbddc->R_to_B and (optionally) pcbddc->R_to_D */
  {
    PetscInt   *aux_array1;
    PetscInt   *aux_array2;
    PetscScalar      value;

    ierr = PetscMalloc( (pcis->n_B-pcbddc->n_vertices)*sizeof(PetscInt),&aux_array1);CHKERRQ(ierr);
    ierr = PetscMalloc( (pcis->n_B-pcbddc->n_vertices)*sizeof(PetscInt),&aux_array2);CHKERRQ(ierr);

    ierr = VecSet(pcis->vec1_global,zero);
    ierr = VecScatterBegin(matis->ctx,pcis->vec1_N,pcis->vec1_global,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
    ierr = VecScatterEnd  (matis->ctx,pcis->vec1_N,pcis->vec1_global,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
    ierr = VecScatterBegin(matis->ctx,pcis->vec1_global,pcis->vec1_N,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterEnd  (matis->ctx,pcis->vec1_global,pcis->vec1_N,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterBegin(pcis->N_to_B,pcis->vec1_N,pcis->vec1_B,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterEnd  (pcis->N_to_B,pcis->vec1_N,pcis->vec1_B,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);    
    ierr = VecGetArray(pcis->vec1_N,&array);CHKERRQ(ierr);
    for (i=0, s=0; i<n_R; i++) { if (array[idx_R_local[i]] > one) { aux_array1[s] = i; s++; } }
    ierr = VecRestoreArray(pcis->vec1_N,&array);CHKERRQ(ierr);
    ierr = ISCreateGeneral(PETSC_COMM_SELF,s,aux_array1,PETSC_COPY_VALUES,&is_aux1);CHKERRQ(ierr);
    ierr = VecGetArray(pcis->vec1_B,&array);CHKERRQ(ierr);
    for (i=0, s=0; i<n_B; i++) { if (array[i] > one) { aux_array2[s] = i; s++; } }
    ierr = VecRestoreArray(pcis->vec1_N,&array);CHKERRQ(ierr);
    ierr = ISCreateGeneral(PETSC_COMM_SELF,s,aux_array2,PETSC_COPY_VALUES,&is_aux2);CHKERRQ(ierr);
    ierr = VecScatterCreate(pcbddc->vec1_R,is_aux1,pcis->vec1_B,is_aux2,&pcbddc->R_to_B);CHKERRQ(ierr);
    ierr = PetscFree(aux_array1);CHKERRQ(ierr);
    ierr = PetscFree(aux_array2);CHKERRQ(ierr);
    ierr = ISDestroy(&is_aux1);CHKERRQ(ierr);
    ierr = ISDestroy(&is_aux2);CHKERRQ(ierr);

    if(pcbddc->prec_type || check_flag ) {
      ierr = PetscMalloc(n_D*sizeof(PetscInt),&aux_array1);CHKERRQ(ierr);
      ierr = VecGetArray(pcis->vec1_N,&array);CHKERRQ(ierr);
      for (i=0, s=0; i<n_R; i++) { if (array[idx_R_local[i]] == one) { aux_array1[s] = i; s++; } }
      ierr = VecRestoreArray(pcis->vec1_N,&array);CHKERRQ(ierr);
      ierr = ISCreateGeneral(PETSC_COMM_SELF,s,aux_array1,PETSC_COPY_VALUES,&is_aux1);CHKERRQ(ierr);
      ierr = VecScatterCreate(pcbddc->vec1_R,is_aux1,pcis->vec1_D,(IS)0,&pcbddc->R_to_D);CHKERRQ(ierr);
      ierr = PetscFree(aux_array1);CHKERRQ(ierr);
      ierr = ISDestroy(&is_aux1);CHKERRQ(ierr);
    }

    /* Check scatters */
    if(check_flag) {
      
      Vec            vec_aux;

      ierr = PetscViewerASCIIPrintf(viewer,"--------------------------------------------------\n");CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"Checking pcbddc->R_to_B scatter\n");CHKERRQ(ierr);
      ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
      ierr = VecSetRandom(pcbddc->vec1_R,PETSC_NULL);
      ierr = VecSetRandom(pcis->vec1_B,PETSC_NULL);
      ierr = VecDuplicate(pcbddc->vec1_R,&vec_aux);
      ierr = VecCopy(pcbddc->vec1_R,vec_aux);
      ierr = VecScatterBegin(pcbddc->R_to_B,pcbddc->vec1_R,pcis->vec1_B,INSERT_VALUES,SCATTER_FORWARD);
      ierr = VecScatterEnd  (pcbddc->R_to_B,pcbddc->vec1_R,pcis->vec1_B,INSERT_VALUES,SCATTER_FORWARD);
      ierr = VecScatterBegin(pcbddc->R_to_B,pcis->vec1_B,vec_aux,INSERT_VALUES,SCATTER_REVERSE);
      ierr = VecScatterEnd  (pcbddc->R_to_B,pcis->vec1_B,vec_aux,INSERT_VALUES,SCATTER_REVERSE);
      ierr = VecAXPY(vec_aux,m_one,pcbddc->vec1_R);
      ierr = VecNorm(vec_aux,NORM_INFINITY,&value);
      ierr = PetscViewerASCIISynchronizedPrintf(viewer,"Subdomain %04d R_to_B FORWARD error = % 1.14e\n",PetscGlobalRank,value);CHKERRQ(ierr);
      ierr = VecDestroy(&vec_aux);

      ierr = VecSetRandom(pcbddc->vec1_R,PETSC_NULL);
      ierr = VecSetRandom(pcis->vec1_B,PETSC_NULL);
      ierr = VecDuplicate(pcis->vec1_B,&vec_aux);
      ierr = VecCopy(pcis->vec1_B,vec_aux);
      ierr = VecScatterBegin(pcbddc->R_to_B,pcis->vec1_B,pcbddc->vec1_R,INSERT_VALUES,SCATTER_REVERSE);
      ierr = VecScatterEnd  (pcbddc->R_to_B,pcis->vec1_B,pcbddc->vec1_R,INSERT_VALUES,SCATTER_REVERSE);
      ierr = VecScatterBegin(pcbddc->R_to_B,pcbddc->vec1_R,vec_aux,INSERT_VALUES,SCATTER_FORWARD);
      ierr = VecScatterEnd  (pcbddc->R_to_B,pcbddc->vec1_R,vec_aux,INSERT_VALUES,SCATTER_FORWARD);
      ierr = VecAXPY(vec_aux,m_one,pcis->vec1_B);
      ierr = VecNorm(vec_aux,NORM_INFINITY,&value);
      ierr = PetscViewerASCIISynchronizedPrintf(viewer,"Subdomain %04d R_to_B REVERSE error = % 1.14e\n",PetscGlobalRank,value);CHKERRQ(ierr);
      ierr = VecDestroy(&vec_aux);

      ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"Checking pcbddc->R_to_D scatter\n");CHKERRQ(ierr);
      ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);

      ierr = VecSetRandom(pcbddc->vec1_R,PETSC_NULL);
      ierr = VecSetRandom(pcis->vec1_D,PETSC_NULL);
      ierr = VecDuplicate(pcbddc->vec1_R,&vec_aux);
      ierr = VecCopy(pcbddc->vec1_R,vec_aux);
      ierr = VecScatterBegin(pcbddc->R_to_D,pcbddc->vec1_R,pcis->vec1_D,INSERT_VALUES,SCATTER_FORWARD);
      ierr = VecScatterEnd  (pcbddc->R_to_D,pcbddc->vec1_R,pcis->vec1_D,INSERT_VALUES,SCATTER_FORWARD);
      ierr = VecScatterBegin(pcbddc->R_to_D,pcis->vec1_D,vec_aux,INSERT_VALUES,SCATTER_REVERSE);
      ierr = VecScatterEnd  (pcbddc->R_to_D,pcis->vec1_D,vec_aux,INSERT_VALUES,SCATTER_REVERSE);
      ierr = VecAXPY(vec_aux,m_one,pcbddc->vec1_R);
      ierr = VecNorm(vec_aux,NORM_INFINITY,&value);
      ierr = PetscViewerASCIISynchronizedPrintf(viewer,"Subdomain %04d R_to_D FORWARD error = % 1.14e\n",PetscGlobalRank,value);CHKERRQ(ierr);
      ierr = VecDestroy(&vec_aux);

      ierr = VecSetRandom(pcbddc->vec1_R,PETSC_NULL);
      ierr = VecSetRandom(pcis->vec1_D,PETSC_NULL);
      ierr = VecDuplicate(pcis->vec1_D,&vec_aux);
      ierr = VecCopy(pcis->vec1_D,vec_aux);
      ierr = VecScatterBegin(pcbddc->R_to_D,pcis->vec1_D,pcbddc->vec1_R,INSERT_VALUES,SCATTER_REVERSE);
      ierr = VecScatterEnd  (pcbddc->R_to_D,pcis->vec1_D,pcbddc->vec1_R,INSERT_VALUES,SCATTER_REVERSE);
      ierr = VecScatterBegin(pcbddc->R_to_D,pcbddc->vec1_R,vec_aux,INSERT_VALUES,SCATTER_FORWARD);
      ierr = VecScatterEnd  (pcbddc->R_to_D,pcbddc->vec1_R,vec_aux,INSERT_VALUES,SCATTER_FORWARD);
      ierr = VecAXPY(vec_aux,m_one,pcis->vec1_D);
      ierr = VecNorm(vec_aux,NORM_INFINITY,&value);
      ierr = PetscViewerASCIISynchronizedPrintf(viewer,"Subdomain %04d R_to_D REVERSE error = % 1.14e\n",PetscGlobalRank,value);CHKERRQ(ierr);
      ierr = VecDestroy(&vec_aux);
      ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);

    }
  }

  /* vertices in boundary numbering */
  if(pcbddc->n_vertices) {
    ierr = VecSet(pcis->vec1_N,m_one);
    ierr = VecGetArray(pcis->vec1_N,&array);CHKERRQ(ierr);
    for (i=0; i<pcbddc->n_vertices; i++) { array[ pcbddc->vertices[i] ] = i; }
    ierr = VecRestoreArray(pcis->vec1_N,&array);CHKERRQ(ierr);
    ierr = VecScatterBegin(pcis->N_to_B,pcis->vec1_N,pcis->vec1_B,INSERT_VALUES,SCATTER_FORWARD);
    ierr = VecScatterEnd  (pcis->N_to_B,pcis->vec1_N,pcis->vec1_B,INSERT_VALUES,SCATTER_FORWARD);
    ierr = PetscMalloc(pcbddc->n_vertices*sizeof(PetscInt),&idx_V_B);CHKERRQ(ierr);
    ierr = VecGetArray(pcis->vec1_B,&array);CHKERRQ(ierr);
    //printf("vertices in boundary numbering\n");
    for (i=0; i<pcbddc->n_vertices; i++) {
      s=0;
      while (array[s] != i ) {s++;}
      idx_V_B[i]=s;
      //printf("idx_V_B[%d]=%d\n",i,s);
    }
    ierr = VecRestoreArray(pcis->vec1_B,&array);CHKERRQ(ierr);
  }


  /* Assign global numbering to coarse dofs */
  // TODO move this block before calling SetupCoarseEnvironment
  {
    PetscScalar    coarsesum;
    PetscMPIInt    *auxlocal_primal;
    PetscMPIInt    *auxglobal_primal;
    PetscMPIInt    *all_auxglobal_primal;
    PetscMPIInt    *all_auxglobal_primal_type;  /* dummy */

    ierr = MPI_Comm_size(((PetscObject)pc)->comm,&totprocs);CHKERRQ(ierr);
    /* Construct needed data structures for message passing */
    ierr = PetscMalloc( pcbddc->local_primal_size*sizeof(PetscMPIInt),&pcbddc->local_primal_indices);CHKERRQ(ierr);
    ierr = PetscMalloc(          totprocs*sizeof(PetscMPIInt),          &pcbddc->local_primal_sizes);CHKERRQ(ierr);
    ierr = PetscMalloc(          totprocs*sizeof(PetscMPIInt),  &pcbddc->local_primal_displacements);CHKERRQ(ierr);
    /* Gather local_primal_size information to all processes  */
    ierr = MPI_Allgather(&pcbddc->local_primal_size,1,MPIU_INT,&pcbddc->local_primal_sizes[0],1,MPIU_INT, ((PetscObject)pc)->comm );CHKERRQ(ierr);
    pcbddc->replicated_primal_size = 0;
    for (i=0; i<totprocs; i++) {
      pcbddc->local_primal_displacements[i] = pcbddc->replicated_primal_size ;
      pcbddc->replicated_primal_size  += pcbddc->local_primal_sizes[i];
    }
    /* allocate some auxiliary space */
    ierr = PetscMalloc( (pcbddc->local_primal_size)*sizeof(PetscMPIInt),          &auxlocal_primal);CHKERRQ(ierr);
    ierr = PetscMalloc( (pcbddc->local_primal_size)*sizeof(PetscMPIInt),         &auxglobal_primal);CHKERRQ(ierr);
    ierr = PetscMalloc( (pcbddc->replicated_primal_size)*sizeof(PetscMPIInt),     &all_auxglobal_primal);CHKERRQ(ierr);
    ierr = PetscMalloc( (pcbddc->replicated_primal_size)*sizeof(PetscMPIInt),&all_auxglobal_primal_type);CHKERRQ(ierr);

    /* First let's count coarse dofs: note that we allow to have a constraint on a subdomain and not its counterpart on the neighbour subdomain (if user wants)
       This code fragment assumes that the number of local constraints per connected component
       is not greater than the number of nodes on the connected component (for each dof)
       (otherwise we will surely have linear dependence between constraints and thus a singular coarse problem) */
    /* auxlocal_primal      : primal indices in local nodes numbering (internal and interface) */ 
    ierr = VecSet(pcis->vec1_N,zero);
    ierr = VecGetArray(pcis->vec1_N,&array);CHKERRQ(ierr);
    for(i=0;i<pcbddc->n_vertices;i++) {
      array[ pcbddc->vertices[i] ] = one;
      auxlocal_primal[i] = pcbddc->vertices[i];
    }
    for(i=0;i<pcbddc->n_constraints;i++) {
      for (s=0; s<pcbddc->sizes_of_constraint[i]; s++) {
        k = pcbddc->indices_to_constraint[i][s];
        if( array[k] == zero ) {
          array[k] = one;
          auxlocal_primal[i+pcbddc->n_vertices] = k;
          break;
        }
      }
    }
    ierr = VecRestoreArray(pcis->vec1_N,&array);CHKERRQ(ierr);
    ierr = VecSet(pcis->vec1_global,zero);CHKERRQ(ierr);
    ierr = VecScatterBegin(matis->ctx,pcis->vec1_N,pcis->vec1_global,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
    ierr = VecScatterEnd  (matis->ctx,pcis->vec1_N,pcis->vec1_global,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
    ierr = VecScatterBegin(matis->ctx,pcis->vec1_global,pcis->vec1_N,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterEnd  (matis->ctx,pcis->vec1_global,pcis->vec1_N,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecGetArray(pcis->vec1_N,&array);CHKERRQ(ierr);
    for(i=0;i<pcis->n;i++) {
      if(array[i]) { array[i] = one/array[i]; }
    }
    ierr = VecRestoreArray(pcis->vec1_N,&array);CHKERRQ(ierr);
    ierr = VecSet(pcis->vec1_global,zero);CHKERRQ(ierr);
    ierr = VecScatterBegin(matis->ctx,pcis->vec1_N,pcis->vec1_global,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
    ierr = VecScatterEnd  (matis->ctx,pcis->vec1_N,pcis->vec1_global,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);

    ierr = VecSum(pcis->vec1_global,&coarsesum);CHKERRQ(ierr);
    pcbddc->coarse_size = (PetscInt) coarsesum;
    if(check_flag) {
      ierr = PetscViewerASCIIPrintf(viewer,"--------------------------------------------------\n");CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"Size of coarse problem = %d\n",pcbddc->coarse_size);CHKERRQ(ierr);
      ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
    }
    /* Now assign them a global numbering */
    /* auxglobal_primal contains indices in global nodes numbering (internal and interface) */
    ierr = ISLocalToGlobalMappingApply(matis->mapping,pcbddc->local_primal_size,auxlocal_primal,auxglobal_primal);CHKERRQ(ierr);
    /* all_auxglobal_primal contains all primal nodes indices in global nodes numbering (internal and interface) */
    ierr = MPI_Allgatherv(&auxglobal_primal[0],pcbddc->local_primal_size,MPIU_INT,&all_auxglobal_primal[0],pcbddc->local_primal_sizes,pcbddc->local_primal_displacements,MPIU_INT, ((PetscObject)pc)->comm );CHKERRQ(ierr);
    /* aux_global_type is a dummy argument (PetscSortMPIInt doesn't exist!) */
    ierr = PetscSortMPIIntWithArray( pcbddc->replicated_primal_size,all_auxglobal_primal,all_auxglobal_primal_type);CHKERRQ(ierr);
    k=1;
    j=all_auxglobal_primal[0];  /* first dof in global numbering */
    for(i=1;i< pcbddc->replicated_primal_size ;i++) {
      if(j != all_auxglobal_primal[i] ) {
        all_auxglobal_primal[k]=all_auxglobal_primal[i];
        k++;
        j=all_auxglobal_primal[i];
      }
    }
    /* At this point all_auxglobal_primal should contains one copy of each primal node's indices in global nodes numbering */
    /* We need only the indices from 0 to pcbddc->coarse_size. Remaning elements of array are garbage. */
    /* Now get global coarse numbering of local primal nodes */
    for(i=0;i<pcbddc->local_primal_size;i++) {
      k=0;
      while( all_auxglobal_primal[k] != auxglobal_primal[i] ) { k++;}
      pcbddc->local_primal_indices[i]=k;
    }
    if(check_flag) {
      ierr = PetscViewerASCIIPrintf(viewer,"--------------------------------------------------\n");CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"Distribution of local primal indices\n");CHKERRQ(ierr);
      ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
      ierr = PetscViewerASCIISynchronizedPrintf(viewer,"Subdomain %04d\n",PetscGlobalRank);CHKERRQ(ierr);
      for(i=0;i<pcbddc->local_primal_size;i++) {
        ierr = PetscViewerASCIISynchronizedPrintf(viewer,"local_primal_indices[%d]=%d \n",i,pcbddc->local_primal_indices[i]);
      }
      ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
    }
    /* free allocated memory */
    ierr = PetscFree(          auxlocal_primal);CHKERRQ(ierr);
    ierr = PetscFree(         auxglobal_primal);CHKERRQ(ierr);
    ierr = PetscFree(     all_auxglobal_primal);CHKERRQ(ierr);
    ierr = PetscFree(all_auxglobal_primal_type);CHKERRQ(ierr);

  }

  /* Creating PC contexts for local Dirichlet and Neumann problems */
  {
    Mat  A_RR;
    /* Matrix for Dirichlet problem is A_II -> we already have it from pcis.c code */
    ierr = PCCreate(PETSC_COMM_SELF,&pcbddc->pc_D);CHKERRQ(ierr);
    ierr = PetscObjectIncrementTabLevel((PetscObject)pcbddc->pc_D,(PetscObject)pc,1);CHKERRQ(ierr);
    ierr = PCSetOperators(pcbddc->pc_D,pcis->A_II,pcis->A_II,SAME_PRECONDITIONER);CHKERRQ(ierr);
    ierr = PCSetOptionsPrefix(pcbddc->pc_D,"pc_bddc_localD_");CHKERRQ(ierr);
    /* default */
    ierr = PCSetType(pcbddc->pc_D,PCLU);CHKERRQ(ierr);
    /* Allow user's customization */
    ierr = PCSetFromOptions(pcbddc->pc_D);CHKERRQ(ierr);
    /* Set Up PC for Dirichlet problem of BDDC */
    ierr = PCSetUp(pcbddc->pc_D);CHKERRQ(ierr);
    /* Matrix for Neumann problem is A_RR -> we need to create it */
    ierr = MatGetSubMatrix(matis->A,is_R_local,is_R_local,MAT_INITIAL_MATRIX,&A_RR);CHKERRQ(ierr);
    ierr = PCCreate(PETSC_COMM_SELF,&pcbddc->pc_R);CHKERRQ(ierr);
    ierr = PetscObjectIncrementTabLevel((PetscObject)pcbddc->pc_R,(PetscObject)pc,1);CHKERRQ(ierr);
    ierr = PCSetOperators(pcbddc->pc_R,A_RR,A_RR,SAME_PRECONDITIONER);CHKERRQ(ierr);
    ierr = PCSetOptionsPrefix(pcbddc->pc_R,"pc_bddc_localN_");CHKERRQ(ierr);
    /* default */
    ierr = PCSetType(pcbddc->pc_R,PCLU);CHKERRQ(ierr);
    /* Allow user's customization */
    ierr = PCSetFromOptions(pcbddc->pc_R);CHKERRQ(ierr);
    /* Set Up PC for Neumann problem of BDDC */
    ierr = PCSetUp(pcbddc->pc_R);CHKERRQ(ierr);
    /* check Neumann solve */
    if(pcbddc->check_flag) {
      Vec temp_vec;
      PetscScalar value;

      ierr = VecDuplicate(pcbddc->vec1_R,&temp_vec);
      ierr = VecSetRandom(pcbddc->vec1_R,PETSC_NULL);
      ierr = MatMult(A_RR,pcbddc->vec1_R,pcbddc->vec2_R);
      ierr = PCApply(pcbddc->pc_R,pcbddc->vec2_R,temp_vec);
      ierr = VecAXPY(temp_vec,m_one,pcbddc->vec1_R);
      ierr = VecNorm(temp_vec,NORM_INFINITY,&value);
      ierr = PetscViewerFlush(viewer);
      ierr = PetscViewerASCIIPrintf(viewer,"--------------------------------------------------\n");CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"Checking solution of Neumann problem\n");CHKERRQ(ierr);
      ierr = PetscViewerASCIISynchronizedPrintf(viewer,"Subdomain %04d infinity error for Neumann solve = % 1.14e \n",PetscGlobalRank,value);CHKERRQ(ierr);
      ierr = PetscViewerFlush(viewer);
      ierr = VecDestroy(&temp_vec);
    }
    /* free Neumann problem's matrix */
    ierr = MatDestroy(&A_RR);CHKERRQ(ierr);
  }

  /* Assemble all remaining stuff needed to apply BDDC  */
  {
    Mat          A_RV,A_VR,A_VV;
    Mat          M1,M2;
    Mat          C_CR;
    Mat          CMAT,AUXMAT;
    Vec          vec1_C;
    Vec          vec2_C;
    Vec          vec1_V;
    Vec          vec2_V;
    PetscInt     *nnz;
    PetscInt     *auxindices;
    PetscInt     index[0];
    PetscScalar* array2;
    MatFactorInfo matinfo;

    /* Allocating some extra storage just to be safe */
    ierr = PetscMalloc (pcis->n*sizeof(PetscInt),&nnz);CHKERRQ(ierr);
    ierr = PetscMalloc (pcis->n*sizeof(PetscInt),&auxindices);CHKERRQ(ierr);
    for(i=0;i<pcis->n;i++) {auxindices[i]=i;}

    /* some work vectors on vertices and/or constraints */
    if(pcbddc->n_vertices) {
      ierr = VecCreate(PETSC_COMM_SELF,&vec1_V);CHKERRQ(ierr);
      ierr = VecSetSizes(vec1_V,pcbddc->n_vertices,pcbddc->n_vertices);CHKERRQ(ierr);
      ierr = VecSetType(vec1_V,impVecType);CHKERRQ(ierr);
      ierr = VecDuplicate(vec1_V,&vec2_V);CHKERRQ(ierr);
    }
    if(pcbddc->n_constraints) {
      ierr = VecCreate(PETSC_COMM_SELF,&vec1_C);CHKERRQ(ierr);
      ierr = VecSetSizes(vec1_C,pcbddc->n_constraints,pcbddc->n_constraints);CHKERRQ(ierr);
      ierr = VecSetType(vec1_C,impVecType);CHKERRQ(ierr);
      ierr = VecDuplicate(vec1_C,&vec2_C); CHKERRQ(ierr);
      ierr = VecDuplicate(vec1_C,&pcbddc->vec1_C); CHKERRQ(ierr);
    }
    /* Create C matrix [I 0; 0 const] */
    ierr = MatCreate(PETSC_COMM_SELF,&CMAT);
    ierr = MatSetType(CMAT,MATSEQAIJ);CHKERRQ(ierr);
    ierr = MatSetSizes(CMAT,pcbddc->local_primal_size,pcis->n,pcbddc->local_primal_size,pcis->n);CHKERRQ(ierr);
    /* nonzeros */
    for(i=0;i<pcbddc->n_vertices;i++) { nnz[i]= 1; }
    for(i=0;i<pcbddc->n_constraints;i++) { nnz[i+pcbddc->n_vertices]=pcbddc->sizes_of_constraint[i];}
    ierr = MatSeqAIJSetPreallocation(CMAT,0,nnz);CHKERRQ(ierr);
    for(i=0;i<pcbddc->n_vertices;i++) {
      ierr = MatSetValue(CMAT,i,pcbddc->vertices[i],1.0,INSERT_VALUES); CHKERRQ(ierr);
    }
    for(i=0;i<pcbddc->n_constraints;i++) {
      index[0]=i+pcbddc->n_vertices;
      ierr = MatSetValues(CMAT,1,index,pcbddc->sizes_of_constraint[i],pcbddc->indices_to_constraint[i],pcbddc->quadrature_constraint[i],INSERT_VALUES); CHKERRQ(ierr);
    }
    //if(check_flag) printf("CMAT assembling\n");
    ierr = MatAssemblyBegin(CMAT,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    ierr = MatAssemblyEnd(CMAT,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    //ierr = MatView(CMAT,PETSC_VIEWER_STDOUT_SELF);

    /* Precompute stuffs needed for preprocessing and application of BDDC*/

    if(pcbddc->n_constraints) {
      /* some work vectors */
      ierr = MatCreate(PETSC_COMM_SELF,&pcbddc->local_auxmat2);CHKERRQ(ierr);
      ierr = MatSetSizes(pcbddc->local_auxmat2,n_R,pcbddc->n_constraints,n_R,pcbddc->n_constraints);CHKERRQ(ierr);
      ierr = MatSetType(pcbddc->local_auxmat2,impMatType);CHKERRQ(ierr); 

      /* Assemble local_auxmat2 = - A_{RR}^{-1} C^T_{CR} needed by BDDC application */
      for(i=0;i<pcbddc->n_constraints;i++) {
        ierr = VecSet(pcis->vec1_N,zero);
        ierr = VecGetArray(pcis->vec1_N,&array);CHKERRQ(ierr);
        for(j=0;j<pcbddc->sizes_of_constraint[i];j++) { array[ pcbddc->indices_to_constraint[i][j] ] = - pcbddc->quadrature_constraint[i][j]; }
        ierr = VecGetArray(pcbddc->vec1_R,&array2);CHKERRQ(ierr);
        for(j=0;j<n_R;j++) { array2[j] = array[ idx_R_local[j] ]; }
        ierr = VecRestoreArray(pcis->vec1_N,&array);CHKERRQ(ierr);
        ierr = VecRestoreArray(pcbddc->vec1_R,&array2);CHKERRQ(ierr);
        ierr = PCApply(pcbddc->pc_R,pcbddc->vec1_R,pcbddc->vec2_R);CHKERRQ(ierr);
        ierr = VecGetArray(pcbddc->vec2_R,&array);CHKERRQ(ierr);
        index[0]=i;
        ierr = MatSetValues(pcbddc->local_auxmat2,n_R,auxindices,1,index,array,INSERT_VALUES); CHKERRQ(ierr);
        ierr = VecRestoreArray(pcbddc->vec2_R,&array);CHKERRQ(ierr);
      }
      //if(check_flag) printf("pcbddc->local_auxmat2 assembling\n");
      ierr = MatAssemblyBegin(pcbddc->local_auxmat2,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
      ierr = MatAssemblyEnd  (pcbddc->local_auxmat2,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

      /* Create Constraint matrix on R nodes: C_{CR}  */
      ierr = MatGetSubMatrix(CMAT,is_C_local,is_R_local,MAT_INITIAL_MATRIX,&C_CR);CHKERRQ(ierr);
      ierr = ISDestroy(&is_C_local); CHKERRQ(ierr);

        /* Assemble AUXMAT = ( LUFactor )( -C_{CR} A_{RR}^{-1} C^T_{CR} )^{-1} */
      ierr = MatMatMult(C_CR,pcbddc->local_auxmat2,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&AUXMAT); CHKERRQ(ierr);
      ierr = MatFactorInfoInitialize(&matinfo);
      ierr = ISCreateStride (PETSC_COMM_SELF,pcbddc->n_constraints,0,1,&is_aux1);CHKERRQ(ierr);
      ierr = MatLUFactor(AUXMAT,is_aux1,is_aux1,&matinfo);CHKERRQ(ierr);
      ierr = ISDestroy(&is_aux1); CHKERRQ(ierr);

        /* Assemble explicitly M1 = ( C_{CR} A_{RR}^{-1} C^T_{CR} )^{-1} needed in preproc (should be dense) */
      ierr = MatCreate(PETSC_COMM_SELF,&M1);
      ierr = MatSetSizes(M1,pcbddc->n_constraints,pcbddc->n_constraints,pcbddc->n_constraints,pcbddc->n_constraints);CHKERRQ(ierr);
      ierr = MatSetType(M1,impMatType);CHKERRQ(ierr);
      for(i=0;i<pcbddc->n_constraints;i++) {
        ierr = VecSet(vec1_C,zero);CHKERRQ(ierr);
        ierr = VecSetValue(vec1_C,i,one,INSERT_VALUES);CHKERRQ(ierr);
        ierr = VecAssemblyBegin(vec1_C);CHKERRQ(ierr);
        ierr = VecAssemblyEnd(vec1_C);CHKERRQ(ierr);
        ierr = MatSolve(AUXMAT,vec1_C,vec2_C);CHKERRQ(ierr);
        ierr = VecScale(vec2_C,m_one);CHKERRQ(ierr);
        ierr = VecGetArray(vec2_C,&array);CHKERRQ(ierr);
        index[0]=i;
        ierr = MatSetValues(M1,pcbddc->n_constraints,auxindices,1,index,array,INSERT_VALUES);CHKERRQ(ierr);
        ierr = VecRestoreArray(vec2_C,&array);CHKERRQ(ierr);
      }
      //if(check_flag) printf("M1 assembling\n");
      ierr = MatAssemblyBegin(M1,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
      ierr = MatAssemblyEnd  (M1,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
      ierr = MatDestroy(&AUXMAT);CHKERRQ(ierr);
      /* Assemble local_auxmat1 = M1*C_{CR} needed by BDDC application in KSP and in preproc */
      //if(check_flag) printf("pcbddc->local_auxmat1 computing and assembling\n");
      ierr = MatMatMult(M1,C_CR,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&pcbddc->local_auxmat1); CHKERRQ(ierr);

    }

    /* Get submatrices from subdomain matrix */
    if(pcbddc->n_vertices){
      ierr = MatGetSubMatrix(matis->A,is_R_local,is_V_local,MAT_INITIAL_MATRIX,&A_RV);CHKERRQ(ierr);
      ierr = MatGetSubMatrix(matis->A,is_V_local,is_R_local,MAT_INITIAL_MATRIX,&A_VR);CHKERRQ(ierr);
      ierr = MatGetSubMatrix(matis->A,is_V_local,is_V_local,MAT_INITIAL_MATRIX,&A_VV);CHKERRQ(ierr);
      /* Assemble M2 = A_RR^{-1}A_RV */
      ierr = MatCreate(PETSC_COMM_SELF,&M2);
      ierr = MatSetSizes(M2,n_R,pcbddc->n_vertices,n_R,pcbddc->n_vertices);CHKERRQ(ierr);
      ierr = MatSetType(M2,impMatType);CHKERRQ(ierr);
      for(i=0;i<pcbddc->n_vertices;i++) {
        ierr = VecSet(vec1_V,zero);CHKERRQ(ierr);
        ierr = VecSetValue(vec1_V,i,one,INSERT_VALUES);CHKERRQ(ierr);
        ierr = VecAssemblyBegin(vec1_V);CHKERRQ(ierr);
        ierr = VecAssemblyEnd(vec1_V);CHKERRQ(ierr);
        ierr = MatMult(A_RV,vec1_V,pcbddc->vec1_R);CHKERRQ(ierr);
        ierr = PCApply(pcbddc->pc_R,pcbddc->vec1_R,pcbddc->vec2_R);CHKERRQ(ierr);
        index[0]=i;
        ierr = VecGetArray(pcbddc->vec2_R,&array);CHKERRQ(ierr);
        ierr = MatSetValues(M2,n_R,auxindices,1,index,array,INSERT_VALUES);CHKERRQ(ierr);
        ierr = VecRestoreArray(pcbddc->vec2_R,&array);CHKERRQ(ierr);
      }
      //if(check_flag) printf("M2 assembling\n");
      ierr = MatAssemblyBegin(M2,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
      ierr = MatAssemblyEnd  (M2,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    }

/* Matrix of coarse basis functions (local) */
    ierr = MatCreate(PETSC_COMM_SELF,&pcbddc->coarse_phi_B);
    ierr = MatSetSizes(pcbddc->coarse_phi_B,n_B,pcbddc->local_primal_size,n_B,pcbddc->local_primal_size);CHKERRQ(ierr);
    ierr = MatSetType(pcbddc->coarse_phi_B,impMatType);CHKERRQ(ierr);
    if(pcbddc->prec_type || check_flag ) {
      ierr = MatCreate(PETSC_COMM_SELF,&pcbddc->coarse_phi_D);
      ierr = MatSetSizes(pcbddc->coarse_phi_D,n_D,pcbddc->local_primal_size,n_D,pcbddc->local_primal_size);CHKERRQ(ierr);
      ierr = MatSetType(pcbddc->coarse_phi_D,impMatType);CHKERRQ(ierr);
    }

/* Subdomain contribution (Non-overlapping) to coarse matrix  */
    if(check_flag) {
      ierr = PetscMalloc( pcbddc->local_primal_size*sizeof(PetscScalar),&coarsefunctions_errors);CHKERRQ(ierr);
      ierr = PetscMalloc( pcbddc->local_primal_size*sizeof(PetscScalar),&constraints_errors);CHKERRQ(ierr);
    }
    ierr = PetscMalloc ((pcbddc->local_primal_size)*(pcbddc->local_primal_size)*sizeof(PetscScalar),&coarse_submat_vals);CHKERRQ(ierr);

    /* We are now ready to evaluate coarse basis functions and subdomain contribution to coarse problem */
    for(i=0;i<pcbddc->n_vertices;i++){
      ierr = VecSet(vec1_V,zero);CHKERRQ(ierr);
      ierr = VecSetValue(vec1_V,i,one,INSERT_VALUES);CHKERRQ(ierr);
      ierr = VecAssemblyBegin(vec1_V);CHKERRQ(ierr);
      ierr = VecAssemblyEnd(vec1_V);CHKERRQ(ierr);
      /* solution of saddle point problem */
      ierr = MatMult(M2,vec1_V,pcbddc->vec1_R);CHKERRQ(ierr);
      ierr = VecScale(pcbddc->vec1_R,m_one);CHKERRQ(ierr);
      if(pcbddc->n_constraints) {
        ierr = MatMult(pcbddc->local_auxmat1,pcbddc->vec1_R,vec1_C);CHKERRQ(ierr);
        ierr = MatMultAdd(pcbddc->local_auxmat2,vec1_C,pcbddc->vec1_R,pcbddc->vec1_R);CHKERRQ(ierr);
        ierr = VecScale(vec1_C,m_one);CHKERRQ(ierr);
      }
      ierr = MatMult(A_VR,pcbddc->vec1_R,vec2_V);CHKERRQ(ierr);
      ierr = MatMultAdd(A_VV,vec1_V,vec2_V,vec2_V);CHKERRQ(ierr);

      /* Set values in coarse basis function and subdomain part of coarse_mat */
      /* coarse basis functions */
      index[0]=i;
      ierr = VecSet(pcis->vec1_B,zero);CHKERRQ(ierr);
      ierr = VecScatterBegin(pcbddc->R_to_B,pcbddc->vec1_R,pcis->vec1_B,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
      ierr = VecScatterEnd  (pcbddc->R_to_B,pcbddc->vec1_R,pcis->vec1_B,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
      ierr = VecGetArray(pcis->vec1_B,&array);CHKERRQ(ierr);
      ierr = MatSetValues(pcbddc->coarse_phi_B,n_B,auxindices,1,index,array,INSERT_VALUES);CHKERRQ(ierr);
      ierr = VecRestoreArray(pcis->vec1_B,&array);CHKERRQ(ierr);
      ierr = MatSetValue(pcbddc->coarse_phi_B,idx_V_B[i],i,one,INSERT_VALUES);CHKERRQ(ierr);
      if( pcbddc->prec_type || check_flag  ) {
        ierr = VecScatterBegin(pcbddc->R_to_D,pcbddc->vec1_R,pcis->vec1_D,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
        ierr = VecScatterEnd(pcbddc->R_to_D,pcbddc->vec1_R,pcis->vec1_D,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
        ierr = VecGetArray(pcis->vec1_D,&array);CHKERRQ(ierr);
        ierr = MatSetValues(pcbddc->coarse_phi_D,n_D,auxindices,1,index,array,INSERT_VALUES);CHKERRQ(ierr);
        ierr = VecRestoreArray(pcis->vec1_D,&array);CHKERRQ(ierr);
      } 
      /* subdomain contribution to coarse matrix */
      ierr = VecGetArray(vec2_V,&array);CHKERRQ(ierr);
      for(j=0;j<pcbddc->n_vertices;j++) {coarse_submat_vals[i*pcbddc->local_primal_size+j]=array[j];} //WARNING -> column major ordering
      ierr = VecRestoreArray(vec2_V,&array);CHKERRQ(ierr);
      if( pcbddc->n_constraints) {
        ierr = VecGetArray(vec1_C,&array);CHKERRQ(ierr);
        for(j=0;j<pcbddc->n_constraints;j++) {coarse_submat_vals[i*pcbddc->local_primal_size+j+pcbddc->n_vertices]=array[j];} //WARNING -> column major ordering
        ierr = VecRestoreArray(vec1_C,&array);CHKERRQ(ierr);
      }
 
      if( check_flag ) {
        /* assemble subdomain vector on nodes */
        ierr = VecSet(pcis->vec1_N,zero);
        ierr = VecGetArray(pcis->vec1_N,&array);CHKERRQ(ierr);
        ierr = VecGetArray(pcbddc->vec1_R,&array2);CHKERRQ(ierr);
        for(j=0;j<n_R;j++) { array[idx_R_local[j]] = array2[j]; }
        array[ pcbddc->vertices[i] ] = one;
        ierr = VecRestoreArray(pcbddc->vec1_R,&array2);CHKERRQ(ierr);
        ierr = VecRestoreArray(pcis->vec1_N,&array);CHKERRQ(ierr);
        /* assemble subdomain vector of lagrange multipliers (i.e. primal nodes) */
        ierr = VecSet(pcbddc->vec1_P,zero);
        ierr = VecGetArray(pcbddc->vec1_P,&array2);CHKERRQ(ierr);
        ierr = VecGetArray(vec2_V,&array);CHKERRQ(ierr);
        for(j=0;j<pcbddc->n_vertices;j++) { array2[j]=array[j]; }
        ierr = VecRestoreArray(vec2_V,&array);CHKERRQ(ierr);
        if(pcbddc->n_constraints) {
          ierr = VecGetArray(vec1_C,&array);CHKERRQ(ierr);
          for(j=0;j<pcbddc->n_constraints;j++) { array2[j+pcbddc->n_vertices]=array[j]; }
          ierr = VecRestoreArray(vec1_C,&array);CHKERRQ(ierr);
        } 
        ierr = VecRestoreArray(pcbddc->vec1_P,&array2);CHKERRQ(ierr);
        ierr = VecScale(pcbddc->vec1_P,m_one);CHKERRQ(ierr);
        /* check saddle point solution */
        ierr = MatMult(matis->A,pcis->vec1_N,pcis->vec2_N); CHKERRQ(ierr);
        ierr = MatMultTransposeAdd(CMAT,pcbddc->vec1_P,pcis->vec2_N,pcis->vec2_N);
        ierr = VecNorm(pcis->vec2_N,NORM_INFINITY,&coarsefunctions_errors[index[0]]); CHKERRQ(ierr);
        ierr = MatMult(CMAT,pcis->vec1_N,pcbddc->vec1_P); CHKERRQ(ierr);
        ierr = VecGetArray(pcbddc->vec1_P,&array);CHKERRQ(ierr);
        array[index[0]]=array[index[0]]+m_one;  /* shift by the identity matrix */
        ierr = VecRestoreArray(pcbddc->vec1_P,&array);CHKERRQ(ierr);
        ierr = VecNorm(pcbddc->vec1_P,NORM_INFINITY,&constraints_errors[index[0]]); CHKERRQ(ierr);
      }
    }
 
    for(i=0;i<pcbddc->n_constraints;i++){
      ierr = VecSet(vec2_C,zero);
      ierr = VecSetValue(vec2_C,i,m_one,INSERT_VALUES);CHKERRQ(ierr);
      ierr = VecAssemblyBegin(vec2_C);CHKERRQ(ierr);
      ierr = VecAssemblyEnd(vec2_C);CHKERRQ(ierr);
      /* solution of saddle point problem */
      ierr = MatMult(M1,vec2_C,vec1_C);CHKERRQ(ierr);
      ierr = MatMult(pcbddc->local_auxmat2,vec1_C,pcbddc->vec1_R);CHKERRQ(ierr);
      ierr = VecScale(vec1_C,m_one);CHKERRQ(ierr);
      if(pcbddc->n_vertices) { ierr = MatMult(A_VR,pcbddc->vec1_R,vec2_V);CHKERRQ(ierr); }
      /* Set values in coarse basis function and subdomain part of coarse_mat */
      /* coarse basis functions */
      index[0]=i+pcbddc->n_vertices;
      ierr = VecSet(pcis->vec1_B,zero);CHKERRQ(ierr);
      ierr = VecScatterBegin(pcbddc->R_to_B,pcbddc->vec1_R,pcis->vec1_B,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
      ierr = VecScatterEnd  (pcbddc->R_to_B,pcbddc->vec1_R,pcis->vec1_B,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
      ierr = VecGetArray(pcis->vec1_B,&array);CHKERRQ(ierr);
      ierr = MatSetValues(pcbddc->coarse_phi_B,n_B,auxindices,1,index,array,INSERT_VALUES);CHKERRQ(ierr);
      ierr = VecRestoreArray(pcis->vec1_B,&array);CHKERRQ(ierr);
      if( pcbddc->prec_type || check_flag ) {
        ierr = VecScatterBegin(pcbddc->R_to_D,pcbddc->vec1_R,pcis->vec1_D,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
        ierr = VecScatterEnd(pcbddc->R_to_D,pcbddc->vec1_R,pcis->vec1_D,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
        ierr = VecGetArray(pcis->vec1_D,&array);CHKERRQ(ierr);
        ierr = MatSetValues(pcbddc->coarse_phi_D,n_D,auxindices,1,index,array,INSERT_VALUES);CHKERRQ(ierr);
        ierr = VecRestoreArray(pcis->vec1_D,&array);CHKERRQ(ierr);
      }
      /* subdomain contribution to coarse matrix */
      if(pcbddc->n_vertices) {
        ierr = VecGetArray(vec2_V,&array);CHKERRQ(ierr);
        for(j=0;j<pcbddc->n_vertices;j++) {coarse_submat_vals[index[0]*pcbddc->local_primal_size+j]=array[j];} //WARNING -> column major ordering
        ierr = VecRestoreArray(vec2_V,&array);CHKERRQ(ierr);
      }
      ierr = VecGetArray(vec1_C,&array);CHKERRQ(ierr);
      for(j=0;j<pcbddc->n_constraints;j++) {coarse_submat_vals[index[0]*pcbddc->local_primal_size+j+pcbddc->n_vertices]=array[j];} //WARNING -> column major ordering
      ierr = VecRestoreArray(vec1_C,&array);CHKERRQ(ierr);
 
      if( check_flag ) {
        /* assemble subdomain vector on nodes */
        ierr = VecSet(pcis->vec1_N,zero);
        ierr = VecGetArray(pcis->vec1_N,&array);CHKERRQ(ierr);
        ierr = VecGetArray(pcbddc->vec1_R,&array2);CHKERRQ(ierr);
        for(j=0;j<n_R;j++){ array[ idx_R_local[j] ] = array2[j]; }
        ierr = VecRestoreArray(pcbddc->vec1_R,&array2);CHKERRQ(ierr);
        ierr = VecRestoreArray(pcis->vec1_N,&array);CHKERRQ(ierr);
        /* assemble subdomain vector of lagrange multipliers */
        ierr = VecSet(pcbddc->vec1_P,zero);
        ierr = VecGetArray(pcbddc->vec1_P,&array2);CHKERRQ(ierr);
        if( pcbddc->n_vertices) {
          ierr = VecGetArray(vec2_V,&array);CHKERRQ(ierr);
          for(j=0;j<pcbddc->n_vertices;j++) {array2[j]=-array[j];}
          ierr = VecRestoreArray(vec2_V,&array);CHKERRQ(ierr);
        }
        ierr = VecGetArray(vec1_C,&array);CHKERRQ(ierr);
        for(j=0;j<pcbddc->n_constraints;j++) {array2[j+pcbddc->n_vertices]=-array[j];}
        ierr = VecRestoreArray(vec1_C,&array);CHKERRQ(ierr);
        ierr = VecRestoreArray(pcbddc->vec1_P,&array2);CHKERRQ(ierr);
        /* check saddle point solution */
        ierr = MatMult(matis->A,pcis->vec1_N,pcis->vec2_N); CHKERRQ(ierr);
        ierr = MatMultTransposeAdd(CMAT,pcbddc->vec1_P,pcis->vec2_N,pcis->vec2_N);
        ierr = VecNorm(pcis->vec2_N,NORM_INFINITY,&coarsefunctions_errors[index[0]]); CHKERRQ(ierr);
        ierr = MatMult(CMAT,pcis->vec1_N,pcbddc->vec1_P); CHKERRQ(ierr);
        ierr = VecGetArray(pcbddc->vec1_P,&array);CHKERRQ(ierr);
        array[index[0]]=array[index[0]]+m_one; /* shift by the identity matrix */
        ierr = VecRestoreArray(pcbddc->vec1_P,&array);CHKERRQ(ierr);
        ierr = VecNorm(pcbddc->vec1_P,NORM_INFINITY,&constraints_errors[index[0]]); CHKERRQ(ierr);
      }
    }
    ierr = MatAssemblyBegin(pcbddc->coarse_phi_B,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    ierr = MatAssemblyEnd  (pcbddc->coarse_phi_B,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    if( pcbddc->prec_type || check_flag ) {
      ierr = MatAssemblyBegin(pcbddc->coarse_phi_D,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
      ierr = MatAssemblyEnd  (pcbddc->coarse_phi_D,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    }
    /* Checking coarse_sub_mat and coarse basis functios */
    /* It shuld be \Phi^{(j)^T} A^{(j)} \Phi^{(j)}=coarse_sub_mat */
    if(check_flag) {

      Mat coarse_sub_mat;
      Mat TM1,TM2,TM3,TM4;
      Mat coarse_phi_D,coarse_phi_B,A_II,A_BB,A_IB,A_BI;
      const MatType checkmattype;
      PetscScalar      value;
      PetscInt bs;

      ierr = MatGetType(matis->A,&checkmattype);CHKERRQ(ierr);
      ierr = MatGetBlockSize(matis->A,&bs);CHKERRQ(ierr);
      //printf("Mat type local ismat: %s\n",checkmattype);
      //printf("Mat bs   local ismat: %d\n",bs);
      ierr = MatGetType(pcis->A_II,&checkmattype);CHKERRQ(ierr);
      ierr = MatGetBlockSize(pcis->A_II,&bs);CHKERRQ(ierr);
      //printf("Mat type local is D : %s\n",checkmattype);
      //printf("Mat bs   local is D : %d\n",bs);
      checkmattype = MATSEQAIJ;
      MatConvert(pcis->A_II,checkmattype,MAT_INITIAL_MATRIX,&A_II);
      MatConvert(pcis->A_IB,checkmattype,MAT_INITIAL_MATRIX,&A_IB);
      MatConvert(pcis->A_BI,checkmattype,MAT_INITIAL_MATRIX,&A_BI);
      MatConvert(pcis->A_BB,checkmattype,MAT_INITIAL_MATRIX,&A_BB);
      MatConvert(pcbddc->coarse_phi_D,checkmattype,MAT_INITIAL_MATRIX,&coarse_phi_D);
      MatConvert(pcbddc->coarse_phi_B,checkmattype,MAT_INITIAL_MATRIX,&coarse_phi_B);
      MatCreateSeqDense(PETSC_COMM_SELF,pcbddc->local_primal_size,pcbddc->local_primal_size,coarse_submat_vals,&coarse_sub_mat);CHKERRQ(ierr);
      MatConvert(coarse_sub_mat,checkmattype,MAT_REUSE_MATRIX,&coarse_sub_mat);

      ierr = PetscViewerASCIIPrintf(viewer,"--------------------------------------------------\n");CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"Check coarse sub mat and local basis functions\n");CHKERRQ(ierr);
      ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
      ierr = MatPtAP(A_II,coarse_phi_D,MAT_INITIAL_MATRIX,1.0,&TM1);
      ierr = MatPtAP(A_BB,coarse_phi_B,MAT_INITIAL_MATRIX,1.0,&TM2);
      ierr = MatMatMult(A_IB,coarse_phi_B,MAT_INITIAL_MATRIX,1.0,&AUXMAT);
      ierr = MatMatTransposeMult(coarse_phi_D,AUXMAT,MAT_INITIAL_MATRIX,1.0,&TM3);
      ierr = MatDestroy(&AUXMAT);
      ierr = MatMatMult(A_BI,coarse_phi_D,MAT_INITIAL_MATRIX,1.0,&AUXMAT);
      ierr = MatMatTransposeMult(coarse_phi_B,AUXMAT,MAT_INITIAL_MATRIX,1.0,&TM4);
      ierr = MatDestroy(&AUXMAT);
      ierr = MatAXPY(TM1,one,TM2,DIFFERENT_NONZERO_PATTERN);
      ierr = MatAXPY(TM1,one,TM3,DIFFERENT_NONZERO_PATTERN);
      ierr = MatAXPY(TM1,one,TM4,DIFFERENT_NONZERO_PATTERN);
      ierr = MatAXPY(TM1,m_one,coarse_sub_mat,DIFFERENT_NONZERO_PATTERN);
      ierr = MatNorm(TM1,NORM_INFINITY,&value);
      ierr = PetscViewerASCIISynchronizedPrintf(viewer,"----------------------------------\n");CHKERRQ(ierr);
      ierr = PetscViewerASCIISynchronizedPrintf(viewer,"Subdomain %04d \n",PetscGlobalRank);CHKERRQ(ierr);
      ierr = PetscViewerASCIISynchronizedPrintf(viewer,"matrix error = % 1.14e\n",value);CHKERRQ(ierr);
      ierr = PetscViewerASCIISynchronizedPrintf(viewer,"coarse functions errors\n");CHKERRQ(ierr);
      for(i=0;i<pcbddc->local_primal_size;i++) {
        ierr = PetscViewerASCIISynchronizedPrintf(viewer,"local %02d-th function error = % 1.14e\n",i,coarsefunctions_errors[i]);CHKERRQ(ierr);
      }
      ierr = PetscViewerASCIISynchronizedPrintf(viewer,"constraints errors\n");CHKERRQ(ierr);
      for(i=0;i<pcbddc->local_primal_size;i++) {
        ierr = PetscViewerASCIISynchronizedPrintf(viewer,"local %02d-th function error = % 1.14e\n",i,constraints_errors[i]);CHKERRQ(ierr);
      }
      ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
      ierr = MatDestroy(&A_II);
      ierr = MatDestroy(&A_BB);
      ierr = MatDestroy(&A_IB);
      ierr = MatDestroy(&A_BI);
      ierr = MatDestroy(&TM1);
      ierr = MatDestroy(&TM2);
      ierr = MatDestroy(&TM3);
      ierr = MatDestroy(&TM4);
      ierr = MatDestroy(&coarse_phi_D);
      ierr = MatDestroy(&coarse_sub_mat);
      ierr = MatDestroy(&coarse_phi_B);
      ierr = PetscFree(coarsefunctions_errors);CHKERRQ(ierr);
      ierr = PetscFree(constraints_errors);CHKERRQ(ierr);
    }

    /* create coarse matrix and data structures for message passing associated actual choice of coarse problem type */
    ierr = PCBDDCSetupCoarseEnvironment(pc,coarse_submat_vals);CHKERRQ(ierr);
    /* free memory */ 
    ierr = PetscFree(coarse_submat_vals);CHKERRQ(ierr);
    ierr = PetscFree(auxindices);CHKERRQ(ierr);
    ierr = PetscFree(nnz);CHKERRQ(ierr);
    if(pcbddc->n_vertices) {
      ierr = VecDestroy(&vec1_V);CHKERRQ(ierr);
      ierr = VecDestroy(&vec2_V);CHKERRQ(ierr);
      ierr = MatDestroy(&M2);CHKERRQ(ierr);
      ierr = MatDestroy(&A_RV);CHKERRQ(ierr);
      ierr = MatDestroy(&A_VR);CHKERRQ(ierr);
      ierr = MatDestroy(&A_VV);CHKERRQ(ierr);
    }
    if(pcbddc->n_constraints) {
      ierr = VecDestroy(&vec1_C);CHKERRQ(ierr);
      ierr = VecDestroy(&vec2_C);CHKERRQ(ierr);
      ierr = MatDestroy(&M1);CHKERRQ(ierr);
      ierr = MatDestroy(&C_CR); CHKERRQ(ierr);
    }
    ierr = MatDestroy(&CMAT); CHKERRQ(ierr);
  }
  /* free memory */ 
  if(pcbddc->n_vertices) {
    ierr = PetscFree(idx_V_B);CHKERRQ(ierr);
    ierr = ISDestroy(&is_V_local);CHKERRQ(ierr);
  }
  ierr = PetscFree(idx_R_local);CHKERRQ(ierr);
  ierr = ISDestroy(&is_R_local);CHKERRQ(ierr);
  //ierr = VecDestroy(&pcis->vec2_N);CHKERRQ(ierr);
  //pcis->vec2_N = 0;

  PetscFunctionReturn(0);

}

/* -------------------------------------------------------------------------- */

#undef __FUNCT__  
#define __FUNCT__ "PCBDDCSetupCoarseEnvironment"
PetscErrorCode PCBDDCSetupCoarseEnvironment(PC pc,PetscScalar* coarse_submat_vals)
{

 
  Mat_IS    *matis    = (Mat_IS*)pc->pmat->data; 
  PC_BDDC   *pcbddc   = (PC_BDDC*)pc->data;
  PC_IS     *pcis     = (PC_IS*)pc->data;
  MPI_Comm  prec_comm = ((PetscObject)pc)->comm;
  MPI_Comm  coarse_comm;

  /* common to all choiches */
  PetscScalar *temp_coarse_mat_vals;
  PetscScalar *ins_coarse_mat_vals;
  PetscInt    *ins_local_primal_indices;
  PetscMPIInt *localsizes2,*localdispl2;
  PetscMPIInt size_prec_comm;
  PetscMPIInt rank_prec_comm;
  PetscMPIInt active_rank=MPI_PROC_NULL;
  PetscMPIInt master_proc=0;
  PetscInt    ins_local_primal_size;
  /* specific to MULTILEVEL_BDDC */
  PetscMPIInt *ranks_recv;
  PetscMPIInt count_recv=0;
  PetscMPIInt rank_coarse_proc_send_to;
  PetscMPIInt coarse_color = MPI_UNDEFINED;
  ISLocalToGlobalMapping coarse_ISLG;
  /* some other variables */
  PetscErrorCode ierr;
  const MatType coarse_mat_type;
  const PCType  coarse_pc_type;
  PetscInt i,j,k,bs;
  
  PetscFunctionBegin;

  ins_local_primal_indices = 0;
  ins_coarse_mat_vals      = 0;
  localsizes2              = 0;
  localdispl2              = 0;
  temp_coarse_mat_vals     = 0;
  coarse_ISLG              = 0;

  MPI_Comm_size(prec_comm,&size_prec_comm);
  MPI_Comm_rank(prec_comm,&rank_prec_comm);
  ierr = MatGetBlockSize(matis->A,&bs);CHKERRQ(ierr);

  /* adapt coarse problem type */
  if(pcbddc->coarse_problem_type == MULTILEVEL_BDDC && pcbddc->active_procs < MIN_PROCS_FOR_BDDC )
    pcbddc->coarse_problem_type = PARALLEL_BDDC;

  switch(pcbddc->coarse_problem_type){

    case(MULTILEVEL_BDDC):   //we define a coarse mesh where subdomains are elements
    {
      /* we need additional variables */
      MetisInt   n_subdomains,n_parts,objval,ncon,faces_nvtxs;
      MetisInt   *metis_coarse_subdivision;
      MetisInt   options[METIS_NOPTIONS];
      PetscMPIInt size_coarse_comm,rank_coarse_comm;
      PetscMPIInt procs_jumps_coarse_comm;
      PetscMPIInt *coarse_subdivision;
      PetscMPIInt *total_count_recv;
      PetscMPIInt *total_ranks_recv;
      PetscMPIInt *displacements_recv;
      PetscMPIInt *my_faces_connectivity;
      PetscMPIInt *petsc_faces_adjncy;
      MetisInt    *faces_adjncy;
      MetisInt    *faces_xadj;
      PetscMPIInt *number_of_faces;
      PetscMPIInt *faces_displacements;
      PetscInt    *array_int;
      PetscMPIInt my_faces=0;
      PetscMPIInt total_faces=0;

      /* this code has a bug (see below) for more then three levels -> I can solve it quickly */

      /* define some quantities */
      pcbddc->coarse_communications_type = SCATTERS_BDDC;
      coarse_mat_type = MATIS;
      coarse_pc_type  = PCBDDC;

      /* details of coarse decomposition */
      n_subdomains = pcbddc->active_procs;
      n_parts      = n_subdomains/pcbddc->coarsening_ratio;
      procs_jumps_coarse_comm = pcbddc->coarsening_ratio*(size_prec_comm/pcbddc->active_procs);

      ierr = PetscMalloc (pcbddc->replicated_primal_size*sizeof(PetscMPIInt),&pcbddc->replicated_local_primal_indices);CHKERRQ(ierr);
      MPI_Allgatherv(&pcbddc->local_primal_indices[0],pcbddc->local_primal_size,MPIU_INT,&pcbddc->replicated_local_primal_indices[0],pcbddc->local_primal_sizes,pcbddc->local_primal_displacements,MPIU_INT,prec_comm);

      /* build CSR graph of subdomains' connectivity through faces */
      ierr = PetscMalloc (pcis->n*sizeof(PetscInt),&array_int);CHKERRQ(ierr);
      PetscMemzero(array_int,pcis->n*sizeof(PetscInt));
      for(i=1;i<pcis->n_neigh;i++){/* i=1 so I don't count myself -> faces nodes counts to 1 */
        for(j=0;j<pcis->n_shared[i];j++){
          array_int[ pcis->shared[i][j] ]+=1;
        }
      }
      for(i=1;i<pcis->n_neigh;i++){
        for(j=0;j<pcis->n_shared[i];j++){
          if(array_int[ pcis->shared[i][j] ] == 1 ){
            my_faces++;
            break;
          }
        }
      }
      //printf("I found %d faces.\n",my_faces);

      MPI_Reduce(&my_faces,&total_faces,1,MPIU_INT,MPI_SUM,master_proc,prec_comm);
      ierr = PetscMalloc (my_faces*sizeof(PetscInt),&my_faces_connectivity);CHKERRQ(ierr);
      my_faces=0;
      for(i=1;i<pcis->n_neigh;i++){
        for(j=0;j<pcis->n_shared[i];j++){
          if(array_int[ pcis->shared[i][j] ] == 1 ){
            my_faces_connectivity[my_faces]=pcis->neigh[i];
            my_faces++;
            break;
          }
        }
      }
      if(rank_prec_comm == master_proc) {
        //printf("I found %d total faces.\n",total_faces);
        ierr = PetscMalloc (total_faces*sizeof(PetscMPIInt),&petsc_faces_adjncy);CHKERRQ(ierr);
        ierr = PetscMalloc (size_prec_comm*sizeof(PetscMPIInt),&number_of_faces);CHKERRQ(ierr);
        ierr = PetscMalloc (total_faces*sizeof(MetisInt),&faces_adjncy);CHKERRQ(ierr);
        ierr = PetscMalloc ((n_subdomains+1)*sizeof(MetisInt),&faces_xadj);CHKERRQ(ierr);
        ierr = PetscMalloc ((size_prec_comm+1)*sizeof(PetscMPIInt),&faces_displacements);CHKERRQ(ierr);
      }
      MPI_Gather(&my_faces,1,MPIU_INT,&number_of_faces[0],1,MPIU_INT,master_proc,prec_comm);
      if(rank_prec_comm == master_proc) {
        faces_xadj[0]=0;
        faces_displacements[0]=0;
        j=0;
        for(i=1;i<size_prec_comm+1;i++) {
          faces_displacements[i]=faces_displacements[i-1]+number_of_faces[i-1];
          if(number_of_faces[i-1]) {
            j++;
            faces_xadj[j]=faces_xadj[j-1]+number_of_faces[i-1];
          }
        }
        printf("The J I count is %d and should be %d\n",j,n_subdomains);
        printf("Total faces seem %d and should be %d\n",faces_xadj[j],total_faces);
      }
      MPI_Gatherv(&my_faces_connectivity[0],my_faces,MPIU_INT,&petsc_faces_adjncy[0],number_of_faces,faces_displacements,MPIU_INT,master_proc,prec_comm);
      ierr = PetscFree(my_faces_connectivity);CHKERRQ(ierr);
      ierr = PetscFree(array_int);CHKERRQ(ierr);
      if(rank_prec_comm == master_proc) {
        for(i=0;i<total_faces;i++) faces_adjncy[i]=(MetisInt)(petsc_faces_adjncy[i]); ///procs_jumps_coarse_comm); // cast to MetisInt
        printf("This is the face connectivity (%d)\n",procs_jumps_coarse_comm);
        for(i=0;i<n_subdomains;i++){
          printf("proc %d is connected with \n",i);
          for(j=faces_xadj[i];j<faces_xadj[i+1];j++)
            printf("%d ",faces_adjncy[j]);
          printf("\n");
        }
        ierr = PetscFree(faces_displacements);CHKERRQ(ierr);
        ierr = PetscFree(number_of_faces);CHKERRQ(ierr);
        ierr = PetscFree(petsc_faces_adjncy);CHKERRQ(ierr);
      }

      if( rank_prec_comm == master_proc ) {

        ncon=1;
        faces_nvtxs=n_subdomains;
        /* partition graoh induced by face connectivity */
        ierr = PetscMalloc (n_subdomains*sizeof(MetisInt),&metis_coarse_subdivision);CHKERRQ(ierr);
        ierr = METIS_SetDefaultOptions(options);
        /* we need a contiguous partition of the coarse mesh */
        options[METIS_OPTION_CONTIG]=1;
        options[METIS_OPTION_DBGLVL]=1;
        options[METIS_OPTION_OBJTYPE]=METIS_OBJTYPE_CUT; 
        options[METIS_OPTION_IPTYPE]=METIS_IPTYPE_EDGE;
        options[METIS_OPTION_NITER]=30;
        //options[METIS_OPTION_NCUTS]=1;
        //printf("METIS PART GRAPH\n");
        /* BUG: faces_xadj and faces_adjncy content must be adapted using the coarsening factor*/
        ierr = METIS_PartGraphKway(&faces_nvtxs,&ncon,faces_xadj,faces_adjncy,NULL,NULL,NULL,&n_parts,NULL,NULL,options,&objval,metis_coarse_subdivision);
        //printf("OKOKOKOKOKOKOKOK\n");
        if(ierr != METIS_OK) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in METIS_PartGraphKway (metis error code %D) called from PCBDDCSetupCoarseEnvironment\n",ierr);
        ierr = PetscFree(faces_xadj);CHKERRQ(ierr);
        ierr = PetscFree(faces_adjncy);CHKERRQ(ierr);
        coarse_subdivision = (PetscMPIInt*)calloc(size_prec_comm,sizeof(PetscMPIInt)); /* calloc for contiguous memory since we need to scatter these values later */
        /* copy/cast values avoiding possible type conflicts between PETSc, MPI and METIS */
        for(i=0;i<size_prec_comm;i++) coarse_subdivision[i]=MPI_PROC_NULL;;
        k=size_prec_comm/pcbddc->active_procs;
        for(i=0;i<n_subdomains;i++) coarse_subdivision[k*i]=(PetscInt)(metis_coarse_subdivision[i]);
        ierr = PetscFree(metis_coarse_subdivision);CHKERRQ(ierr);

      }

      /* Create new communicator for coarse problem splitting the old one */
      if( !(rank_prec_comm%procs_jumps_coarse_comm) && rank_prec_comm < procs_jumps_coarse_comm*n_parts ){
        coarse_color=0;              //for communicator splitting
        active_rank=rank_prec_comm;  //for insertion of matrix values
      }
      // procs with coarse_color = MPI_UNDEFINED will have coarse_comm = MPI_COMM_NULL (from mpi standards)
      // key = rank_prec_comm -> keep same ordering of ranks from the old to the new communicator
      MPI_Comm_split(prec_comm,coarse_color,rank_prec_comm,&coarse_comm);

      if( coarse_color == 0 ) {
        MPI_Comm_size(coarse_comm,&size_coarse_comm);
        MPI_Comm_rank(coarse_comm,&rank_coarse_comm);
        //printf("Details of coarse comm\n");
        //printf("size = %d, myrank = %d\n",size_coarse_comm,rank_coarse_comm);
        //printf("jumps = %d, coarse_color = %d, n_parts = %d\n",procs_jumps_coarse_comm,coarse_color,n_parts);
      } else {
        rank_coarse_comm = MPI_PROC_NULL;
      }

      /* master proc take care of arranging and distributing coarse informations */
      if(rank_coarse_comm == master_proc) {
        ierr = PetscMalloc (size_coarse_comm*sizeof(PetscMPIInt),&displacements_recv);CHKERRQ(ierr);
        //ierr = PetscMalloc (size_coarse_comm*sizeof(PetscMPIInt),&total_count_recv);CHKERRQ(ierr);
        //ierr = PetscMalloc (n_subdomains*sizeof(PetscMPIInt),&total_ranks_recv);CHKERRQ(ierr);
        total_count_recv = (PetscMPIInt*)calloc(size_prec_comm,sizeof(PetscMPIInt));
        total_ranks_recv = (PetscMPIInt*)calloc(n_subdomains,sizeof(PetscMPIInt));
        /* some initializations */
        displacements_recv[0]=0;
        //PetscMemzero(total_count_recv,size_coarse_comm*sizeof(PetscMPIInt)); not needed -> calloc initializes to zero
        /* count from how many processes the j-th process of the coarse decomposition will receive data */
        for(j=0;j<size_coarse_comm;j++) 
          for(i=0;i<n_subdomains;i++) 
            if(coarse_subdivision[i]==j) 
              total_count_recv[j]++;
        /* displacements needed for scatterv of total_ranks_recv */
        for(i=1;i<size_coarse_comm;i++) displacements_recv[i]=displacements_recv[i-1]+total_count_recv[i-1];
        /* Now fill properly total_ranks_recv -> each coarse process will receive the ranks (in prec_comm communicator) of its friend (sending) processes */
        ierr = PetscMemzero(total_count_recv,size_coarse_comm*sizeof(PetscMPIInt));CHKERRQ(ierr);
        for(j=0;j<size_coarse_comm;j++) {
          for(i=0;i<n_subdomains;i++) {
            if(coarse_subdivision[i]==j) {
              total_ranks_recv[displacements_recv[j]+total_count_recv[j]]=i;
              total_count_recv[j]=total_count_recv[j]+1;
            }
          }
        }
        //for(j=0;j<size_coarse_comm;j++) {
        //  printf("process %d in new rank will receive from %d processes (ranks follows)\n",j,total_count_recv[j]);
        //  for(i=0;i<total_count_recv[j];i++) {
        //    printf("%d ",total_ranks_recv[displacements_recv[j]+i]);
        //  }
        //  printf("\n");
       // }

        /* identify new decomposition in terms of ranks in the old communicator */
        k=size_prec_comm/pcbddc->active_procs;
        for(i=0;i<n_subdomains;i++) coarse_subdivision[k*i]=coarse_subdivision[k*i]*procs_jumps_coarse_comm;
        printf("coarse_subdivision in old end new ranks\n");
        for(i=0;i<size_prec_comm;i++)
          printf("(%d %d) ",coarse_subdivision[i],coarse_subdivision[i]/procs_jumps_coarse_comm);
        printf("\n");
      }

      /* Scatter new decomposition for send details */
      MPI_Scatter(&coarse_subdivision[0],1,MPIU_INT,&rank_coarse_proc_send_to,1,MPIU_INT,master_proc,prec_comm);
      /* Scatter receiving details to members of coarse decomposition */
      if( coarse_color == 0) {
        MPI_Scatter(&total_count_recv[0],1,MPIU_INT,&count_recv,1,MPIU_INT,master_proc,coarse_comm);
        ierr = PetscMalloc (count_recv*sizeof(PetscMPIInt),&ranks_recv);CHKERRQ(ierr);
        MPI_Scatterv(&total_ranks_recv[0],total_count_recv,displacements_recv,MPIU_INT,&ranks_recv[0],count_recv,MPIU_INT,master_proc,coarse_comm);
      }

      //printf("I will send my matrix data to proc  %d\n",rank_coarse_proc_send_to);
      //if(coarse_color == 0) {
      //  printf("I will receive some matrix data from %d processes (ranks follows)\n",count_recv);
      //  for(i=0;i<count_recv;i++)
      //    printf("%d ",ranks_recv[i]);
      //  printf("\n");
      //}

      if(rank_prec_comm == master_proc) {
        //ierr = PetscFree(coarse_subdivision);CHKERRQ(ierr);
        //ierr = PetscFree(total_count_recv);CHKERRQ(ierr);
        //ierr = PetscFree(total_ranks_recv);CHKERRQ(ierr);
        free(coarse_subdivision);
        free(total_count_recv);
        free(total_ranks_recv);
        ierr = PetscFree(displacements_recv);CHKERRQ(ierr);
      }
      break;
    }

    case(REPLICATED_BDDC):

      pcbddc->coarse_communications_type = GATHERS_BDDC;
      coarse_mat_type = MATSEQAIJ;
      coarse_pc_type  = PCLU;
      coarse_comm = PETSC_COMM_SELF;
      active_rank = rank_prec_comm;
      break;

    case(PARALLEL_BDDC):

      pcbddc->coarse_communications_type = SCATTERS_BDDC;
      coarse_mat_type = MATMPIAIJ;
      coarse_pc_type  = PCREDUNDANT;
      coarse_comm = prec_comm;
      active_rank = rank_prec_comm;
      break;

    case(SEQUENTIAL_BDDC):
      pcbddc->coarse_communications_type = GATHERS_BDDC;
      coarse_mat_type = MATSEQAIJ;
      coarse_pc_type = PCLU;
      coarse_comm = PETSC_COMM_SELF;
      active_rank = master_proc;
      break;
  }

  switch(pcbddc->coarse_communications_type){

    case(SCATTERS_BDDC):
      {
        if(pcbddc->coarse_problem_type==MULTILEVEL_BDDC) {

          PetscMPIInt send_size;
          PetscInt    *aux_ins_indices;
          PetscInt    ii,jj;
          MPI_Request *requests;

          /* allocate auxiliary space */
          ierr = PetscMalloc ( pcbddc->coarse_size*sizeof(PetscInt),&aux_ins_indices);CHKERRQ(ierr);
          ierr = PetscMemzero(aux_ins_indices,pcbddc->coarse_size*sizeof(PetscInt));CHKERRQ(ierr);
          /* allocate stuffs for message massing */
          ierr = PetscMalloc ( (count_recv+1)*sizeof(MPI_Request),&requests);CHKERRQ(ierr);
          for(i=0;i<count_recv+1;i++) requests[i]=MPI_REQUEST_NULL;
          ierr = PetscMalloc ( count_recv*sizeof(PetscMPIInt),&localsizes2);CHKERRQ(ierr);
          ierr = PetscMalloc ( count_recv*sizeof(PetscMPIInt),&localdispl2);CHKERRQ(ierr);
          /* fill up quantities */
          j=0;
          for(i=0;i<count_recv;i++){
            ii = ranks_recv[i];
            localsizes2[i]=pcbddc->local_primal_sizes[ii]*pcbddc->local_primal_sizes[ii];
            localdispl2[i]=j;
            j+=localsizes2[i];
            jj = pcbddc->local_primal_displacements[ii];
            for(k=0;k<pcbddc->local_primal_sizes[ii];k++) aux_ins_indices[pcbddc->replicated_local_primal_indices[jj+k]]+=1;  // it counts the coarse subdomains sharing the coarse node
          }
          //printf("aux_ins_indices 1\n");
          //for(i=0;i<pcbddc->coarse_size;i++)
          //  printf("%d ",aux_ins_indices[i]);
          //printf("\n");
          /* temp_coarse_mat_vals used to store temporarly received matrix values */
          ierr = PetscMalloc ( j*sizeof(PetscScalar),&temp_coarse_mat_vals);CHKERRQ(ierr);
          /* evaluate how many values I will insert in coarse mat */
          ins_local_primal_size=0;
          for(i=0;i<pcbddc->coarse_size;i++)
            if(aux_ins_indices[i])
              ins_local_primal_size++;
          /* evaluate indices I will insert in coarse mat */
          ierr = PetscMalloc ( ins_local_primal_size*sizeof(PetscInt),&ins_local_primal_indices);CHKERRQ(ierr);
          j=0;
          for(i=0;i<pcbddc->coarse_size;i++)
            if(aux_ins_indices[i])
              ins_local_primal_indices[j++]=i;
          /* use aux_ins_indices to realize a global to local mapping */
          j=0;
          for(i=0;i<pcbddc->coarse_size;i++){
            if(aux_ins_indices[i]==0){
              aux_ins_indices[i]=-1;
            } else {
              aux_ins_indices[i]=j;
              j++;
            }
          }         

          //printf("New details localsizes2 localdispl2\n");
          //for(i=0;i<count_recv;i++)
          //  printf("(%d %d) ",localsizes2[i],localdispl2[i]);
          //printf("\n");
          //printf("aux_ins_indices 2\n");
          //for(i=0;i<pcbddc->coarse_size;i++)
          //  printf("%d ",aux_ins_indices[i]);
          //printf("\n");
          //printf("ins_local_primal_indices\n");
          //for(i=0;i<ins_local_primal_size;i++)
          //  printf("%d ",ins_local_primal_indices[i]);
          //printf("\n");
          //printf("coarse_submat_vals\n");
          //for(i=0;i<pcbddc->local_primal_size;i++)
          //  for(j=0;j<pcbddc->local_primal_size;j++)
          //    printf("(%lf %d %d)\n",coarse_submat_vals[j*pcbddc->local_primal_size+i],pcbddc->local_primal_indices[i],pcbddc->local_primal_indices[j]);
          //printf("\n");
 
          /* processes partecipating in coarse problem receive matrix data from their friends */
          for(i=0;i<count_recv;i++) MPI_Irecv(&temp_coarse_mat_vals[localdispl2[i]],localsizes2[i],MPIU_SCALAR,ranks_recv[i],666,prec_comm,&requests[i]);
          if(rank_coarse_proc_send_to != MPI_PROC_NULL ) {
            send_size=pcbddc->local_primal_size*pcbddc->local_primal_size;
            MPI_Isend(&coarse_submat_vals[0],send_size,MPIU_SCALAR,rank_coarse_proc_send_to,666,prec_comm,&requests[count_recv]);
          }
          MPI_Waitall(count_recv+1,requests,MPI_STATUSES_IGNORE);

          //if(coarse_color == 0) {
          //  printf("temp_coarse_mat_vals\n");
          //  for(k=0;k<count_recv;k++){
          //    printf("---- %d ----\n",ranks_recv[k]);
          //    for(i=0;i<pcbddc->local_primal_sizes[ranks_recv[k]];i++)
          //      for(j=0;j<pcbddc->local_primal_sizes[ranks_recv[k]];j++)
          //        printf("(%lf %d %d)\n",temp_coarse_mat_vals[localdispl2[k]+j*pcbddc->local_primal_sizes[ranks_recv[k]]+i],pcbddc->replicated_local_primal_indices[pcbddc->local_primal_displacements[ranks_recv[k]]+i],pcbddc->replicated_local_primal_indices[pcbddc->local_primal_displacements[ranks_recv[k]]+j]);
          //    printf("\n");
          //  }
          //}
          /* calculate data to insert in coarse mat */
          ierr = PetscMalloc ( ins_local_primal_size*ins_local_primal_size*sizeof(PetscScalar),&ins_coarse_mat_vals);CHKERRQ(ierr);
          PetscMemzero(ins_coarse_mat_vals,ins_local_primal_size*ins_local_primal_size*sizeof(PetscScalar));

          PetscMPIInt rr,kk,lps,lpd;
          PetscInt row_ind,col_ind;
          for(k=0;k<count_recv;k++){
            rr = ranks_recv[k];
            kk = localdispl2[k];
            lps = pcbddc->local_primal_sizes[rr];
            lpd = pcbddc->local_primal_displacements[rr];
            //printf("Inserting the following indices (received from %d)\n",rr);
            for(j=0;j<lps;j++){
              col_ind=aux_ins_indices[pcbddc->replicated_local_primal_indices[lpd+j]];
              for(i=0;i<lps;i++){
                row_ind=aux_ins_indices[pcbddc->replicated_local_primal_indices[lpd+i]];
                //printf("%d %d\n",row_ind,col_ind);
                ins_coarse_mat_vals[col_ind*ins_local_primal_size+row_ind]+=temp_coarse_mat_vals[kk+j*lps+i];
              }
            }
          }
          ierr = PetscFree(requests);CHKERRQ(ierr);
          ierr = PetscFree(aux_ins_indices);CHKERRQ(ierr);
          ierr = PetscFree(temp_coarse_mat_vals);CHKERRQ(ierr);
          if(coarse_color == 0) { ierr = PetscFree(ranks_recv);CHKERRQ(ierr); }

            /* create local to global mapping needed by coarse MATIS */
          {
            IS coarse_IS;
            if(coarse_comm != MPI_COMM_NULL ) MPI_Comm_free(&coarse_comm);
            coarse_comm = prec_comm;
            active_rank=rank_prec_comm;
            ierr = ISCreateGeneral(coarse_comm,ins_local_primal_size,ins_local_primal_indices,PETSC_COPY_VALUES,&coarse_IS);CHKERRQ(ierr);
            //ierr = ISSetBlockSize(coarse_IS,bs);CHKERRQ(ierr);
            ierr = ISLocalToGlobalMappingCreateIS(coarse_IS,&coarse_ISLG);CHKERRQ(ierr);
            ierr = ISDestroy(&coarse_IS);CHKERRQ(ierr);
          }
        }
        if(pcbddc->coarse_problem_type==PARALLEL_BDDC) {
          /* arrays for values insertion */
          ins_local_primal_size = pcbddc->local_primal_size;
          ierr = PetscMalloc ( ins_local_primal_size*sizeof(PetscMPIInt),&ins_local_primal_indices);CHKERRQ(ierr);
          ierr = PetscMalloc ( ins_local_primal_size*ins_local_primal_size*sizeof(PetscScalar),&ins_coarse_mat_vals);CHKERRQ(ierr);
          for(j=0;j<ins_local_primal_size;j++){
            ins_local_primal_indices[j]=pcbddc->local_primal_indices[j];
            for(i=0;i<ins_local_primal_size;i++) ins_coarse_mat_vals[j*ins_local_primal_size+i]=coarse_submat_vals[j*ins_local_primal_size+i];
          }
        }
        break;
        
    }

    case(GATHERS_BDDC):
      {

        PetscMPIInt mysize,mysize2;

        if(rank_prec_comm==active_rank) {
          ierr = PetscMalloc ( pcbddc->replicated_primal_size*sizeof(PetscMPIInt),&pcbddc->replicated_local_primal_indices);CHKERRQ(ierr);
          pcbddc->replicated_local_primal_values = (PetscScalar*)calloc(pcbddc->replicated_primal_size,sizeof(PetscScalar));
          ierr = PetscMalloc ( size_prec_comm*sizeof(PetscMPIInt),&localsizes2);CHKERRQ(ierr);
          ierr = PetscMalloc ( size_prec_comm*sizeof(PetscMPIInt),&localdispl2);CHKERRQ(ierr);
          /* arrays for values insertion */
          ins_local_primal_size = pcbddc->coarse_size;
          ierr = PetscMalloc ( ins_local_primal_size*sizeof(PetscMPIInt),&ins_local_primal_indices);CHKERRQ(ierr);
          ierr = PetscMalloc ( ins_local_primal_size*ins_local_primal_size*sizeof(PetscScalar),&ins_coarse_mat_vals);CHKERRQ(ierr);
          for(i=0;i<size_prec_comm;i++) localsizes2[i]=pcbddc->local_primal_sizes[i]*pcbddc->local_primal_sizes[i];
          localdispl2[0]=0;
          for(i=1;i<size_prec_comm;i++) localdispl2[i]=localsizes2[i-1]+localdispl2[i-1];
          j=0;
          for(i=0;i<size_prec_comm;i++) j+=localsizes2[i];
          ierr = PetscMalloc ( j*sizeof(PetscScalar),&temp_coarse_mat_vals);CHKERRQ(ierr);
        }

        mysize=pcbddc->local_primal_size;
        mysize2=pcbddc->local_primal_size*pcbddc->local_primal_size;
        if(pcbddc->coarse_problem_type == SEQUENTIAL_BDDC){
          MPI_Gatherv(&pcbddc->local_primal_indices[0],mysize,MPIU_INT,&pcbddc->replicated_local_primal_indices[0],pcbddc->local_primal_sizes,pcbddc->local_primal_displacements,MPIU_INT,master_proc,prec_comm);
          MPI_Gatherv(&coarse_submat_vals[0],mysize2,MPIU_SCALAR,&temp_coarse_mat_vals[0],localsizes2,localdispl2,MPIU_SCALAR,master_proc,prec_comm);
        } else {
          MPI_Allgatherv(&pcbddc->local_primal_indices[0],mysize,MPIU_INT,&pcbddc->replicated_local_primal_indices[0],pcbddc->local_primal_sizes,pcbddc->local_primal_displacements,MPIU_INT,prec_comm); 
          MPI_Allgatherv(&coarse_submat_vals[0],mysize2,MPIU_SCALAR,&temp_coarse_mat_vals[0],localsizes2,localdispl2,MPIU_SCALAR,prec_comm);
        }

  /* free data structures no longer needed and allocate some space which will be needed in BDDC application */
        if(rank_prec_comm==active_rank) {
          PetscInt offset,offset2,row_ind,col_ind;
          for(j=0;j<ins_local_primal_size;j++){
            ins_local_primal_indices[j]=j;
            for(i=0;i<ins_local_primal_size;i++) ins_coarse_mat_vals[j*ins_local_primal_size+i]=0.0;
          }
          for(k=0;k<size_prec_comm;k++){
            offset=pcbddc->local_primal_displacements[k];
            offset2=localdispl2[k];
            for(j=0;j<pcbddc->local_primal_sizes[k];j++){
              col_ind=pcbddc->replicated_local_primal_indices[offset+j];
              for(i=0;i<pcbddc->local_primal_sizes[k];i++){
                row_ind=pcbddc->replicated_local_primal_indices[offset+i];
                ins_coarse_mat_vals[col_ind*pcbddc->coarse_size+row_ind]+=temp_coarse_mat_vals[offset2+j*pcbddc->local_primal_sizes[k]+i];
              }
            }
          }
        }
        break;
      }//switch on coarse problem and communications associated with finished
  }

  /* Now create and fill up coarse matrix */
  if( rank_prec_comm == active_rank ) {
    if(pcbddc->coarse_problem_type != MULTILEVEL_BDDC) {
      ierr = MatCreate(coarse_comm,&pcbddc->coarse_mat);CHKERRQ(ierr);
      ierr = MatSetSizes(pcbddc->coarse_mat,PETSC_DECIDE,PETSC_DECIDE,pcbddc->coarse_size,pcbddc->coarse_size);CHKERRQ(ierr);
      ierr = MatSetType(pcbddc->coarse_mat,coarse_mat_type);CHKERRQ(ierr);
      ierr = MatSetOption(pcbddc->coarse_mat,MAT_IGNORE_ZERO_ENTRIES,PETSC_TRUE);CHKERRQ(ierr);
      ierr = MatSetOption(pcbddc->coarse_mat,MAT_ROW_ORIENTED,PETSC_FALSE);CHKERRQ(ierr); //local values stored in column major
    } else {
      Mat matis_coarse_local_mat;
      ierr = MatCreateIS(coarse_comm,PETSC_DECIDE,PETSC_DECIDE,pcbddc->coarse_size,pcbddc->coarse_size,coarse_ISLG,&pcbddc->coarse_mat);CHKERRQ(ierr);
      ierr = MatISGetLocalMat(pcbddc->coarse_mat,&matis_coarse_local_mat);CHKERRQ(ierr);
      ierr = MatSetOption(matis_coarse_local_mat,MAT_ROW_ORIENTED,PETSC_FALSE);CHKERRQ(ierr); //local values stored in column major
      ierr = MatSetOption(matis_coarse_local_mat,MAT_IGNORE_ZERO_ENTRIES,PETSC_TRUE);CHKERRQ(ierr); 
      ierr = MatSetOption(matis_coarse_local_mat,MAT_USE_INODES,PETSC_FALSE);CHKERRQ(ierr);
    }
    ierr = MatSetValues(pcbddc->coarse_mat,ins_local_primal_size,ins_local_primal_indices,ins_local_primal_size,ins_local_primal_indices,ins_coarse_mat_vals,ADD_VALUES);CHKERRQ(ierr);
    ierr = MatAssemblyBegin(pcbddc->coarse_mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(pcbddc->coarse_mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    if(pcbddc->coarse_problem_type == MULTILEVEL_BDDC) {
      Mat matis_coarse_local_mat;
      printf("Setting bs %d\n",bs);
      ierr = MatISGetLocalMat(pcbddc->coarse_mat,&matis_coarse_local_mat);CHKERRQ(ierr);
      ierr = MatSetBlockSize(matis_coarse_local_mat,bs);CHKERRQ(ierr);
    } 

    ierr = MatGetVecs(pcbddc->coarse_mat,&pcbddc->coarse_vec,&pcbddc->coarse_rhs);CHKERRQ(ierr);
    /* Preconditioner for coarse problem */
    ierr = PCCreate(coarse_comm,&pcbddc->coarse_pc);CHKERRQ(ierr);
    ierr = PetscObjectIncrementTabLevel((PetscObject)pcbddc->coarse_pc,(PetscObject)pc,1);CHKERRQ(ierr);
    ierr = PCSetOperators(pcbddc->coarse_pc,pcbddc->coarse_mat,pcbddc->coarse_mat,SAME_PRECONDITIONER);CHKERRQ(ierr);
    ierr = PCSetType(pcbddc->coarse_pc,coarse_pc_type);CHKERRQ(ierr);
    ierr = PCSetOptionsPrefix(pcbddc->coarse_pc,"pc_bddc_coarse_");CHKERRQ(ierr);
    /* Allow user's customization */
    ierr = PCSetFromOptions(pcbddc->coarse_pc);CHKERRQ(ierr);
    /* Set Up PC for coarse problem BDDC */
    //if(pcbddc->check_flag && pcbddc->coarse_problem_type == MULTILEVEL_BDDC) {
    if(pcbddc->coarse_problem_type == MULTILEVEL_BDDC) PetscPrintf(PETSC_COMM_WORLD,"----------------Setting up a new level---------------\n");
    if(pcbddc->coarse_problem_type == MULTILEVEL_BDDC) PCBDDCSetCoarseProblemType(pcbddc->coarse_pc,MULTILEVEL_BDDC);
    ierr = PCSetUp(pcbddc->coarse_pc);CHKERRQ(ierr);
  }
  if(pcbddc->coarse_communications_type == SCATTERS_BDDC) {
     IS local_IS,global_IS;
     ierr = ISCreateStride(PETSC_COMM_SELF,pcbddc->local_primal_size,0,1,&local_IS);CHKERRQ(ierr);
     ierr = ISCreateGeneral(PETSC_COMM_SELF,pcbddc->local_primal_size,pcbddc->local_primal_indices,PETSC_COPY_VALUES,&global_IS);CHKERRQ(ierr);
     ierr = VecScatterCreate(pcbddc->vec1_P,local_IS,pcbddc->coarse_vec,global_IS,&pcbddc->coarse_loc_to_glob);CHKERRQ(ierr);
     ierr = ISDestroy(&local_IS);CHKERRQ(ierr);
     ierr = ISDestroy(&global_IS);CHKERRQ(ierr);
  }


  /* Check condition number of coarse problem */
  /* How to destroy KSP without destroying PC associated with? */ 
/*  if( rank_prec_comm == active_rank ) {
    KSP coarseksp;
    PetscScalar m_one=-1.0;
    PetscReal infty_error,lambda_min,lambda_max;

    KSPCreate(coarse_comm,&coarseksp);
    KSPSetType(coarseksp,KSPCG);
    KSPSetOperators(coarseksp,pcbddc->coarse_mat,pcbddc->coarse_mat,SAME_PRECONDITIONER);
    KSPSetPC(coarseksp,pcbddc->coarse_pc);
    KSPSetComputeSingularValues(coarseksp,PETSC_TRUE);
    VecSetRandom(pcbddc->coarse_rhs,PETSC_NULL);
    MatMult(pcbddc->coarse_mat,pcbddc->coarse_rhs,pcbddc->coarse_vec);
    MatMult(pcbddc->coarse_mat,pcbddc->coarse_vec,pcbddc->coarse_rhs);
    KSPSolve(coarseksp,pcbddc->coarse_rhs,pcbddc->coarse_rhs);
    KSPComputeExtremeSingularValues(coarseksp,&lambda_max,&lambda_min);
    VecAXPY(pcbddc->coarse_rhs,m_one,pcbddc->coarse_vec);
    VecNorm(pcbddc->coarse_rhs,NORM_INFINITY,&infty_error);
    printf("eigenvalues: % 1.14e %1.14e\n",lambda_min,lambda_max);
    printf("Error on coarse ksp %1.14e\n",infty_error);
        
  }*/

  /* free data structures no longer needed */
  if(coarse_ISLG)                { ierr = ISLocalToGlobalMappingDestroy(&coarse_ISLG);CHKERRQ(ierr); }
  if(ins_local_primal_indices)   { ierr = PetscFree(ins_local_primal_indices);CHKERRQ(ierr);  }
  if(ins_coarse_mat_vals)        { ierr = PetscFree(ins_coarse_mat_vals);CHKERRQ(ierr);}
  if(localsizes2)                { ierr = PetscFree(localsizes2);CHKERRQ(ierr);}
  if(localdispl2)                { ierr = PetscFree(localdispl2);CHKERRQ(ierr);}
  if(temp_coarse_mat_vals)       { ierr = PetscFree(temp_coarse_mat_vals);CHKERRQ(ierr);}

  PetscFunctionReturn(0);

}

#undef __FUNCT__  
#define __FUNCT__ "PCBDDCManageLocalBoundaries"
PetscErrorCode PCBDDCManageLocalBoundaries(PC pc)
{

  PC_BDDC  *pcbddc = (PC_BDDC*)pc->data;
  PC_IS      *pcis = (PC_IS*)pc->data;
  Mat_IS   *matis  = (Mat_IS*)pc->pmat->data; 
  PetscInt *distinct_values;
  PetscInt **array_int;
  PetscInt bs,ierr,i,j,s,k;
  PetscInt total_counts;
  PetscBool flg_row;
  PCBDDCGraph mat_graph;
  PetscScalar *array;
  Mat        mat_adj;

  PetscFunctionBegin;
  
  ierr = MatGetBlockSize(matis->A,&bs);CHKERRQ(ierr);
  // allocate and initialize needed graph structure
  ierr = PetscMalloc(sizeof(*mat_graph),&mat_graph);CHKERRQ(ierr);
  ierr = MatConvert(matis->A,MATMPIADJ,MAT_INITIAL_MATRIX,&mat_adj);CHKERRQ(ierr);
  ierr = MatGetRowIJ(mat_adj,0,PETSC_FALSE,PETSC_FALSE,&mat_graph->nvtxs,&mat_graph->xadj,&mat_graph->adjncy,&flg_row);CHKERRQ(ierr);
  if(!flg_row) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Error in MatGetRowIJ called from PCBDDCManageLocalBoundaries.\n");
  ierr = PetscMalloc(mat_graph->nvtxs*sizeof(PetscInt),&mat_graph->where);CHKERRQ(ierr);
  ierr = PetscMalloc(mat_graph->nvtxs*sizeof(PetscInt),&mat_graph->count);CHKERRQ(ierr);
  ierr = PetscMalloc(mat_graph->nvtxs*sizeof(PetscInt),&mat_graph->which_dof);CHKERRQ(ierr);
  ierr = PetscMalloc(mat_graph->nvtxs*sizeof(PetscInt),&mat_graph->queue);CHKERRQ(ierr);
  ierr = PetscMalloc((mat_graph->nvtxs+1)*sizeof(PetscInt),&mat_graph->cptr);CHKERRQ(ierr);
  ierr = PetscMalloc(mat_graph->nvtxs*sizeof(PetscBool),&mat_graph->touched);CHKERRQ(ierr);
  for(i=0;i<mat_graph->nvtxs;i++){
    mat_graph->count[i]=0;
    mat_graph->touched[i]=PETSC_FALSE;
  }
  for(i=0;i<mat_graph->nvtxs/bs;i++) {
    for(s=0;s<bs;s++) {
      mat_graph->which_dof[i*bs+s]=s;
    }
  }
  //printf("nvtxs = %d\n",mat_graph->nvtxs);
  //printf("these are my IS data with n_neigh = %d\n",pcis->n_neigh);
  //for(i=0;i<pcis->n_neigh;i++){
  //  printf("number of shared nodes with rank %d is %d \n",pcis->neigh[i],pcis->n_shared[i]);
  // }

  total_counts=0;
  for(i=0;i<pcis->n_neigh;i++){
    s=pcis->n_shared[i];
    total_counts+=s;
    //printf("computing neigh %d (rank = %d, n_shared = %d)\n",i,pcis->neigh[i],s);
    for(j=0;j<s;j=j++){
      mat_graph->count[pcis->shared[i][j]] += 1;
    }
  }
  /* Take into account Neumann data incrementing number of sharing subdomains for all but faces nodes lying on the interface */
  if(pcbddc->Vec_Neumann) {
    ierr = VecGetArray(pcbddc->Vec_Neumann,&array);CHKERRQ(ierr);
    for(i=0;i<pcis->n;i++){
      if(array[i] > 0.0  && mat_graph->count[i] > 2){
        mat_graph->count[i]=mat_graph->count[i]+1;
        total_counts++;
      }
    }
    ierr = VecRestoreArray(pcbddc->Vec_Neumann,&array);CHKERRQ(ierr);
  }
 
  ierr = PetscMalloc(mat_graph->nvtxs*sizeof(PetscInt*),&array_int);CHKERRQ(ierr);
  if(mat_graph->nvtxs) ierr = PetscMalloc(total_counts*sizeof(PetscInt),&array_int[0]);CHKERRQ(ierr);
  for(i=1;i<mat_graph->nvtxs;i++) array_int[i]=array_int[i-1]+mat_graph->count[i-1];

  for(i=0;i<mat_graph->nvtxs;i++) mat_graph->count[i]=0;
  for(i=0;i<pcis->n_neigh;i++){
    s=pcis->n_shared[i];
    for(j=0;j<s;j++) {
      k=pcis->shared[i][j];
      array_int[k][mat_graph->count[k]] = pcis->neigh[i];
      mat_graph->count[k]+=1;
    }
  }
  /* set -1 fake neighbour */
  if(pcbddc->Vec_Neumann) {
    ierr = VecGetArray(pcbddc->Vec_Neumann,&array);CHKERRQ(ierr);
    for(i=0;i<pcis->n;i++){
      if( array[i] > 0.0 && mat_graph->count[i] > 2){
        array_int[i][mat_graph->count[i]] = -1; //An additional fake neighbour (with rank -1) 
        mat_graph->count[i]+=1;
      }
    }
    ierr = VecRestoreArray(pcbddc->Vec_Neumann,&array);CHKERRQ(ierr);
  }

  /* sort sharing subdomains */
  for(i=0;i<mat_graph->nvtxs;i++) { ierr = PetscSortInt(mat_graph->count[i],array_int[i]);CHKERRQ(ierr); }

  // Prepare for FindConnectedComponents
  // Vertices will be eliminated later (if needed)
  PetscInt nodes_touched=0;
  for(i=0;i<mat_graph->nvtxs;i++){
    if(!mat_graph->count[i]){  //internal nodes
      mat_graph->touched[i]=PETSC_TRUE;
      mat_graph->where[i]=0;
      nodes_touched++;
    }
    if(pcbddc->faces_flag){
      if(mat_graph->count[i]>2){  //all but faces
        mat_graph->touched[i]=PETSC_TRUE;
        mat_graph->where[i]=0;
        nodes_touched++;
      }
    }
    if(pcbddc->edges_flag){
      if(mat_graph->count[i]==2){  //faces
        mat_graph->touched[i]=PETSC_TRUE;
        mat_graph->where[i]=0;
        nodes_touched++;
      }
    }
  } 

  PetscInt rvalue=1;
  PetscBool same_set;
  mat_graph->ncmps = 0;
  while(nodes_touched<mat_graph->nvtxs) {
    // find first untouched node in local ordering
    i=0;
    while(mat_graph->touched[i]) i++;
    mat_graph->touched[i]=PETSC_TRUE;
    mat_graph->where[i]=rvalue;
    nodes_touched++;
    // now find other connected nodes shared by the same set of subdomains
    for(j=i+1;j<mat_graph->nvtxs;j++){
      // check for same number of sharing subdomains and dof number
      if(mat_graph->count[i]==mat_graph->count[j] && mat_graph->which_dof[i] == mat_graph->which_dof[j] ){
        // check for same set of sharing subdomains
        same_set=PETSC_TRUE;
        for(k=0;k<mat_graph->count[j];k++){
          if(array_int[i][k]!=array_int[j][k]) {
            same_set=PETSC_FALSE;
          }
        }
        // OK, I found a friend of mine
        if(same_set) {
          mat_graph->where[j]=rvalue;
          mat_graph->touched[j]=PETSC_TRUE;
          nodes_touched++;
        }
      }
    }
    rvalue++;
  }
//  printf("where and count contains %d distinct values\n",rvalue);
//  for(j=0;j<mat_graph->nvtxs;j++)
//    printf("[%d %d %d]\n",j,mat_graph->where[j],mat_graph->count[j]);

  if(mat_graph->nvtxs) {
    ierr = PetscFree(array_int[0]);CHKERRQ(ierr);
    ierr = PetscFree(array_int);CHKERRQ(ierr);
  }

  rvalue--;
  ierr  = PetscMalloc ( rvalue*sizeof(PetscInt),&distinct_values);CHKERRQ(ierr);
  for(i=0;i<rvalue;i++) distinct_values[i]=i+1;  //initializiation
  if(rvalue) ierr = PCBDDCFindConnectedComponents(mat_graph, rvalue, distinct_values);
  //printf("total number of connected components %d \n",mat_graph->ncmps);
  //for (i=0; i<mat_graph->ncmps; i++) {
  //  printf("[queue num %d] ptr %d, length %d, start index %d\n",i,mat_graph->cptr[i],mat_graph->cptr[i+1]-mat_graph->cptr[i],mat_graph->queue[mat_graph->cptr[i]]);
  //}
  PetscInt nfc=0;
  PetscInt nec=0;
  PetscInt nvc=0;
  for (i=0; i<mat_graph->ncmps; i++) {
    // sort each connected component (by local ordering)
    ierr = PetscSortInt(mat_graph->cptr[i+1]-mat_graph->cptr[i],&mat_graph->queue[mat_graph->cptr[i]]);CHKERRQ(ierr);
    // count edge and faces
    if( !pcbddc->vertices_flag ) {
      if( mat_graph->cptr[i+1]-mat_graph->cptr[i] > 1 ){
        if(mat_graph->count[mat_graph->queue[mat_graph->cptr[i]]]==2){
          nfc++;
        } else {
          nec++;
        }
      }
    }
    // count vertices
    if( !pcbddc->constraints_flag ){
      if( mat_graph->cptr[i+1]-mat_graph->cptr[i] == 1 ){
        nvc++;
      }
    }
  }
  
  pcbddc->n_constraints = nec+nfc;
  pcbddc->n_vertices    = nvc;

  if(pcbddc->n_constraints){
    /* allocate space for local constraints of BDDC */
    ierr  = PetscMalloc (pcbddc->n_constraints*sizeof(PetscInt*),&pcbddc->indices_to_constraint);CHKERRQ(ierr);
    ierr  = PetscMalloc (pcbddc->n_constraints*sizeof(PetscScalar*),&pcbddc->quadrature_constraint);CHKERRQ(ierr);
    ierr  = PetscMalloc (pcbddc->n_constraints*sizeof(PetscInt),&pcbddc->sizes_of_constraint);CHKERRQ(ierr);
    k=0;
    for (i=0; i<mat_graph->ncmps; i++) {
      if( mat_graph->cptr[i+1]-mat_graph->cptr[i] > 1 ){
        pcbddc->sizes_of_constraint[k] = mat_graph->cptr[i+1]-mat_graph->cptr[i];
        k++;
      }
    }
//    printf("check constraints %d (should be %d)\n",k,pcbddc->n_constraints);
//    for(i=0;i<k;i++)
//      printf("%d ",pcbddc->sizes_of_constraint[i]);
//    printf("\n");
    k=0;
    for (i=0; i<pcbddc->n_constraints; i++) k+=pcbddc->sizes_of_constraint[i];
    ierr = PetscMalloc (k*sizeof(PetscInt),&pcbddc->indices_to_constraint[0]);CHKERRQ(ierr);
    ierr = PetscMalloc (k*sizeof(PetscScalar),&pcbddc->quadrature_constraint[0]);CHKERRQ(ierr);
    for (i=1; i<pcbddc->n_constraints; i++) {
      pcbddc->indices_to_constraint[i]  = pcbddc->indices_to_constraint[i-1] + pcbddc->sizes_of_constraint[i-1];
      pcbddc->quadrature_constraint[i]  = pcbddc->quadrature_constraint[i-1] + pcbddc->sizes_of_constraint[i-1];
    }
    k=0;
    PetscScalar quad_value;
    for (i=0; i<mat_graph->ncmps; i++) {
      if( mat_graph->cptr[i+1]-mat_graph->cptr[i] > 1 ){
        quad_value=1.0/( (PetscScalar) (mat_graph->cptr[i+1]-mat_graph->cptr[i]) );
        for(j=0;j<mat_graph->cptr[i+1]-mat_graph->cptr[i];j++) {
          pcbddc->indices_to_constraint[k][j] = mat_graph->queue[mat_graph->cptr[i]+j];
          pcbddc->quadrature_constraint[k][j] = quad_value;
        }
        k++;
      }
    } 
  }
  if(pcbddc->n_vertices){
    /* allocate space for local vertices of BDDC */
    ierr  = PetscMalloc (pcbddc->n_vertices*sizeof(PetscInt),&pcbddc->vertices);CHKERRQ(ierr);
    k=0;
    for (i=0; i<mat_graph->ncmps; i++) {
      if( mat_graph->cptr[i+1]-mat_graph->cptr[i] == 1 ){
        pcbddc->vertices[k] = mat_graph->queue[mat_graph->cptr[i]]; 
        k++;
      }
    }
    // sort vertex set (by local ordering)
    ierr = PetscSortInt(pcbddc->n_vertices,pcbddc->vertices);CHKERRQ(ierr);
  }

  if(pcbddc->check_flag) {
    PetscViewer     viewer;
    PetscViewerASCIIGetStdout(((PetscObject)pc)->comm,&viewer);
    PetscViewerASCIISynchronizedAllow(viewer,PETSC_TRUE);
    PetscViewerASCIISynchronizedPrintf(viewer,"--------------------------------------------------------------\n");
    PetscViewerASCIISynchronizedPrintf(viewer,"Details from PCBDDCManageLocalBoundaries for subdomain %04d\n",PetscGlobalRank);
    PetscViewerASCIISynchronizedPrintf(viewer,"--------------------------------------------------------------\n");
//    PetscViewerASCIISynchronizedPrintf(viewer,"Graph (adjacency structure) of local Neumann mat\n");
//    PetscViewerASCIISynchronizedPrintf(viewer,"--------------------------------------------------------------\n");
//    for(i=0;i<mat_graph->nvtxs;i++) {
//      PetscViewerASCIISynchronizedPrintf(viewer,"Nodes connected to node number %d are %d\n",i,mat_graph->xadj[i+1]-mat_graph->xadj[i]);
//      for(j=mat_graph->xadj[i];j<mat_graph->xadj[i+1];j++){
//        PetscViewerASCIISynchronizedPrintf(viewer,"%d ",mat_graph->adjncy[j]);
//      }
//      PetscViewerASCIISynchronizedPrintf(viewer,"\n--------------------------------------------------------------\n");
//    }
    // TODO: APPLY Local to Global Mapping from IS object?
    PetscViewerASCIISynchronizedPrintf(viewer,"Matrix graph has %d connected components", mat_graph->ncmps);
    for(i=0;i<mat_graph->ncmps;i++) {
      PetscViewerASCIISynchronizedPrintf(viewer,"\nSize and count for connected component %02d : %04d %01d\n", i,mat_graph->cptr[i+1]-mat_graph->cptr[i],mat_graph->count[mat_graph->queue[mat_graph->cptr[i]]]);
      for (j=mat_graph->cptr[i]; j<mat_graph->cptr[i+1]; j++){
        PetscViewerASCIISynchronizedPrintf(viewer,"%d ",mat_graph->queue[j]);
      }
    }
    PetscViewerASCIISynchronizedPrintf(viewer,"\n--------------------------------------------------------------\n");
    if( pcbddc->n_vertices ) PetscViewerASCIISynchronizedPrintf(viewer,"Subdomain %04d detected %02d local vertices\n",PetscGlobalRank,pcbddc->n_vertices);
    if( nfc )                PetscViewerASCIISynchronizedPrintf(viewer,"Subdomain %04d detected %02d local faces\n",PetscGlobalRank,nfc);
    if( nec )                PetscViewerASCIISynchronizedPrintf(viewer,"Subdomain %04d detected %02d local edges\n",PetscGlobalRank,nec);
    if( pcbddc->n_vertices ) PetscViewerASCIISynchronizedPrintf(viewer,"Subdomain vertices follows\n",PetscGlobalRank,pcbddc->n_vertices);
    for(i=0;i<pcbddc->n_vertices;i++){
                             PetscViewerASCIISynchronizedPrintf(viewer,"%d ",pcbddc->vertices[i]);
    }
    if( pcbddc->n_vertices ) PetscViewerASCIISynchronizedPrintf(viewer,"\n");
//    if( pcbddc->n_constraints ) PetscViewerASCIISynchronizedPrintf(viewer,"Indices and quadrature constraints");
//    for(i=0;i<pcbddc->n_constraints;i++){
//      PetscViewerASCIISynchronizedPrintf(viewer,"\nConstraint number %d\n",i);
//      for(j=0;j<pcbddc->sizes_of_constraint[i];j++) {
//        PetscViewerASCIISynchronizedPrintf(viewer,"(%d, %f) ",pcbddc->indices_to_constraint[i][j],pcbddc->quadrature_constraint[i][j]);
//      }
//    }
//    if( pcbddc->n_constraints ) PetscViewerASCIISynchronizedPrintf(viewer,"\n");
    PetscViewerFlush(viewer);
  }

  // Restore CSR structure into sequantial matrix and free memory space no longer needed
  ierr = MatRestoreRowIJ(mat_adj,0,PETSC_FALSE,PETSC_TRUE,&mat_graph->nvtxs,&mat_graph->xadj,&mat_graph->adjncy,&flg_row);CHKERRQ(ierr);
  ierr = MatDestroy(&mat_adj);CHKERRQ(ierr);
  if(!flg_row) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Error in MatRestoreRowIJ called from PCBDDCManageLocalBoundaries.\n");
  ierr = PetscFree(distinct_values);CHKERRQ(ierr);
  // Free graph structure
  if(mat_graph->nvtxs){
    ierr = PetscFree(mat_graph->where);CHKERRQ(ierr);
    ierr = PetscFree(mat_graph->touched);CHKERRQ(ierr);
    ierr = PetscFree(mat_graph->which_dof);CHKERRQ(ierr);
    ierr = PetscFree(mat_graph->queue);CHKERRQ(ierr);
    ierr = PetscFree(mat_graph->cptr);CHKERRQ(ierr);
    ierr = PetscFree(mat_graph->count);CHKERRQ(ierr);
  }
  ierr = PetscFree(mat_graph);CHKERRQ(ierr);

  PetscFunctionReturn(0);

}

/* -------------------------------------------------------------------------- */

/* The following code has been adapted from function IsConnectedSubdomain contained 
   in source file contig.c of METIS library (version 5.0.1)                           */
                                
#undef __FUNCT__  
#define __FUNCT__ "PCBDDCFindConnectedComponents"
PetscErrorCode PCBDDCFindConnectedComponents(PCBDDCGraph graph, PetscInt n_dist, PetscInt *dist_vals)
{
  PetscInt i, j, k, nvtxs, first, last, nleft, ncmps,pid,cum_queue,n,ncmps_pid;
  PetscInt *xadj, *adjncy, *where, *queue;
  PetscInt *cptr;
  PetscBool *touched;
  
  PetscFunctionBegin;

  nvtxs   = graph->nvtxs;
  xadj    = graph->xadj;
  adjncy  = graph->adjncy;
  where   = graph->where;
  touched = graph->touched;
  queue   = graph->queue;
  cptr    = graph->cptr;

  for (i=0; i<nvtxs; i++) 
    touched[i] = PETSC_FALSE;

  cum_queue=0;
  ncmps=0;

  for(n=0; n<n_dist; n++) {

    pid = dist_vals[n]; 
    nleft = 0;
    for (i=0; i<nvtxs; i++) {
      if (where[i] == pid)
        nleft++;
    }
    for (i=0; i<nvtxs; i++) {
      if (where[i] == pid)
        break;
    }
    
    touched[i] = PETSC_TRUE;
    queue[cum_queue] = i;
    first = 0; last = 1;

    cptr[ncmps] = cum_queue;  /* This actually points to queue */
    ncmps_pid = 0;
    while (first != nleft) {
      if (first == last) { /* Find another starting vertex */
        cptr[++ncmps] = first+cum_queue;
        ncmps_pid++;
        for (i=0; i<nvtxs; i++) {
          if (where[i] == pid && !touched[i])
            break;
        }
        queue[cum_queue+last] = i;
        last++;
        touched[i] = PETSC_TRUE;
      }

      i = queue[cum_queue+first];
      first++;
      for (j=xadj[i]; j<xadj[i+1]; j++) {
        k = adjncy[j];
        if (where[k] == pid && !touched[k]) {
          queue[cum_queue+last] = k;
          last++;
          touched[k] = PETSC_TRUE;
        }
      }
    }
    cptr[++ncmps] = first+cum_queue;
    ncmps_pid++;
    cum_queue=cptr[ncmps];

    //printf("The graph has %d connected components in partition %d\n", ncmps_pid, pid);
  }
  graph->ncmps = ncmps;

  PetscFunctionReturn(0);
}

