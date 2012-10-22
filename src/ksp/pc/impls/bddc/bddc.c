/* TODOLIST
   DofSplitting and DM attached to pc?
   Change SetNeumannBoundaries to SetNeumannBoundariesLocal and provide new SetNeumannBoundaries (same Dirichlet)
     - change prec_type to switch_inexact_prec_type
   Inexact solvers: global preconditioner application is ready, ask to developers (Jed?) on how to best implement Dohrmann's approach (PCSHELL?)
   change how to deal with the coarse problem (PCBDDCSetCoarseEnvironment):
     - simplify coarse problem structure -> PCBDDC or PCREDUDANT, nothing else -> same comm for all levels?
     - remove coarse enums and allow use of PCBDDCGetCoarseKSP
     - remove metis dependency -> use MatPartitioning for multilevel -> Assemble serial adjacency in ManageLocalBoundaries?
   code refactoring:
     - pick up better names for static functions
   change options structure:
     - insert BDDC into MG framework?
   provide other ops? Ask to developers
   remove all unused printf
   man pages
*/

/* ---------------------------------------------------------------------------------------------------------------------------------------------- 
   Implementation of BDDC preconditioner based on:
   C. Dohrmann "An approximate BDDC preconditioner", Numerical Linear Algebra with Applications Volume 14, Issue 2, pages 149-168, March 2007
   ---------------------------------------------------------------------------------------------------------------------------------------------- */

#include "bddc.h" /*I "petscpc.h" I*/  /* includes for fortran wrappers */
#include <petscblaslapack.h>
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
  ierr = PetscOptionsBool("-pc_bddc_check_all"       ,"Verbose (debugging) output for PCBDDC"                       ,"none",pcbddc->dbg_flag      ,&pcbddc->dbg_flag      ,PETSC_NULL);CHKERRQ(ierr);
  /* Some customization for default primal space */
  ierr = PetscOptionsBool("-pc_bddc_vertices_only"   ,"Use only vertices in coarse space (i.e. discard constraints)","none",pcbddc->vertices_flag   ,&pcbddc->vertices_flag   ,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-pc_bddc_constraints_only","Use only constraints in coarse space (i.e. discard vertices)","none",pcbddc->constraints_flag,&pcbddc->constraints_flag,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-pc_bddc_faces_only"      ,"Use only faces among constraints of coarse space (i.e. discard edges)"         ,"none",pcbddc->faces_flag      ,&pcbddc->faces_flag      ,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-pc_bddc_edges_only"      ,"Use only edges among constraints of coarse space (i.e. discard faces)"         ,"none",pcbddc->edges_flag      ,&pcbddc->edges_flag      ,PETSC_NULL);CHKERRQ(ierr);
  /* Coarse solver context */
  static const char * const avail_coarse_problems[] = {"sequential","replicated","parallel","multilevel","CoarseProblemType","PC_BDDC_",0}; /*order of choiches depends on ENUM defined in bddc.h */
  ierr = PetscOptionsEnum("-pc_bddc_coarse_problem_type","Set coarse problem type","none",avail_coarse_problems,(PetscEnum)pcbddc->coarse_problem_type,(PetscEnum*)&pcbddc->coarse_problem_type,PETSC_NULL);CHKERRQ(ierr);
  /* Two different application of BDDC to the whole set of dofs, internal and interface */
  ierr = PetscOptionsBool("-pc_bddc_switch_preconditioning_type","Switch between M_2 (default) and M_3 preconditioners (as defined by Dohrmann)","none",pcbddc->prec_type,&pcbddc->prec_type,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-pc_bddc_use_change_of_basis","Use change of basis approach for primal space","none",pcbddc->usechangeofbasis,&pcbddc->usechangeofbasis,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-pc_bddc_use_change_on_faces","Use change of basis approach for face constraints","none",pcbddc->usechangeonfaces,&pcbddc->usechangeonfaces,PETSC_NULL);CHKERRQ(ierr);
  pcbddc->usechangeonfaces = pcbddc->usechangeonfaces && pcbddc->usechangeofbasis;
  ierr = PetscOptionsInt("-pc_bddc_coarsening_ratio","Set coarsening ratio used in multilevel coarsening","none",pcbddc->coarsening_ratio,&pcbddc->coarsening_ratio,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-pc_bddc_max_levels","Set maximum number of levels for multilevel","none",pcbddc->max_levels,&pcbddc->max_levels,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
/* -------------------------------------------------------------------------- */
EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PCBDDCSetCoarseProblemType_BDDC"
static PetscErrorCode PCBDDCSetCoarseProblemType_BDDC(PC pc, CoarseProblemType CPT)
{
  PC_BDDC  *pcbddc = (PC_BDDC*)pc->data;

  PetscFunctionBegin;
  pcbddc->coarse_problem_type = CPT; 
  PetscFunctionReturn(0);
}
EXTERN_C_END
#undef __FUNCT__  
#define __FUNCT__ "PCBDDCSetCoarseProblemType"
/*@
 PCBDDCSetCoarseProblemType - Set coarse problem type in PCBDDC.

   Not collective

   Input Parameters:
+  pc - the preconditioning context
-  CoarseProblemType - pick a better name and explain what this is

   Level: intermediate

   Notes:
   Not collective but all procs must call with same arguments.

.seealso: PCBDDC
@*/
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
#define __FUNCT__ "PCBDDCSetCoarseningRatio_BDDC"
static PetscErrorCode PCBDDCSetCoarseningRatio_BDDC(PC pc,PetscInt k)
{
  PC_BDDC  *pcbddc = (PC_BDDC*)pc->data;

  PetscFunctionBegin;
  pcbddc->coarsening_ratio=k;
  PetscFunctionReturn(0);
}
EXTERN_C_END
#undef __FUNCT__  
#define __FUNCT__ "PCBDDCSetCoarseningRatio"
/*@
 PCBDDCSetCoarseningRatio - Set coarsening ratio used in multilevel coarsening

   Logically collective on PC

   Input Parameters:
+  pc - the preconditioning context
-  k - coarsening ratio

   Approximatively k subdomains at the finer level will be aggregated into a single subdomain at the coarser level.

   Level: intermediate

   Notes:

.seealso: PCBDDC
@*/
PetscErrorCode PCBDDCSetCoarseningRatio(PC pc,PetscInt k)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  ierr = PetscTryMethod(pc,"PCBDDCSetCoarseningRatio_C",(PC,PetscInt),(pc,k));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
/* -------------------------------------------------------------------------- */
EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PCBDDCSetMaxLevels_BDDC"
static PetscErrorCode PCBDDCSetMaxLevels_BDDC(PC pc,PetscInt max_levels)
{
  PC_BDDC  *pcbddc = (PC_BDDC*)pc->data;

  PetscFunctionBegin;
  pcbddc->max_levels=max_levels;
  PetscFunctionReturn(0);
}
EXTERN_C_END
#undef __FUNCT__  
#define __FUNCT__ "PCBDDCSetMaxLevels"
/*@
 PCBDDCSetMaxLevels - Sets the maximum number of levels within the multilevel approach.

   Logically collective on PC

   Input Parameters:
+  pc - the preconditioning context
-  max_levels - the maximum number of levels

   Default value is 1, i.e. coarse problem will be solved inexactly with one application
   of PCBDDC preconditioner if the multilevel approach is requested.

   Level: intermediate

   Notes:

.seealso: PCBDDC
@*/
PetscErrorCode PCBDDCSetMaxLevels(PC pc,PetscInt max_levels)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  ierr = PetscTryMethod(pc,"PCBDDCSetMaxLevels_C",(PC,PetscInt),(pc,max_levels));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
/* -------------------------------------------------------------------------- */
EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PCBDDCSetNullSpace_BDDC"
static PetscErrorCode PCBDDCSetNullSpace_BDDC(PC pc,MatNullSpace NullSpace)
{
  PC_BDDC  *pcbddc = (PC_BDDC*)pc->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectReference((PetscObject)NullSpace);CHKERRQ(ierr);
  ierr = MatNullSpaceDestroy(&pcbddc->NullSpace);CHKERRQ(ierr);
  pcbddc->NullSpace=NullSpace;
  PetscFunctionReturn(0);
}
EXTERN_C_END
#undef __FUNCT__  
#define __FUNCT__ "PCBDDCSetNullSpace"
/*@
 PCBDDCSetNullSpace - Set NullSpace of global operator of BDDC preconditioned mat.

   Logically collective on PC and MatNullSpace

   Input Parameters:
+  pc - the preconditioning context
-  NullSpace - Null space of the linear operator to be preconditioned.

   Level: intermediate

   Notes:

.seealso: PCBDDC
@*/
PetscErrorCode PCBDDCSetNullSpace(PC pc,MatNullSpace NullSpace)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  ierr = PetscTryMethod(pc,"PCBDDCSetNullSpace_C",(PC,MatNullSpace),(pc,NullSpace));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
/* -------------------------------------------------------------------------- */
EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PCBDDCSetDirichletBoundaries_BDDC"
static PetscErrorCode PCBDDCSetDirichletBoundaries_BDDC(PC pc,IS DirichletBoundaries)
{
  PC_BDDC  *pcbddc = (PC_BDDC*)pc->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = ISDestroy(&pcbddc->DirichletBoundaries);CHKERRQ(ierr);
  ierr = PetscObjectReference((PetscObject)DirichletBoundaries);CHKERRQ(ierr);
  pcbddc->DirichletBoundaries=DirichletBoundaries;
  PetscFunctionReturn(0);
}
EXTERN_C_END
#undef __FUNCT__  
#define __FUNCT__ "PCBDDCSetDirichletBoundaries"
/*@
 PCBDDCSetDirichletBoundaries - Set index set defining subdomain part (in local ordering)
                              of Dirichlet boundaries for the global problem.

   Not collective

   Input Parameters:
+  pc - the preconditioning context
-  DirichletBoundaries - sequential index set defining the subdomain part of Dirichlet boundaries (can be PETSC_NULL)

   Level: intermediate

   Notes:

.seealso: PCBDDC
@*/
PetscErrorCode PCBDDCSetDirichletBoundaries(PC pc,IS DirichletBoundaries)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  ierr = PetscTryMethod(pc,"PCBDDCSetDirichletBoundaries_C",(PC,IS),(pc,DirichletBoundaries));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
/* -------------------------------------------------------------------------- */
EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PCBDDCSetNeumannBoundaries_BDDC"
static PetscErrorCode PCBDDCSetNeumannBoundaries_BDDC(PC pc,IS NeumannBoundaries)
{
  PC_BDDC  *pcbddc = (PC_BDDC*)pc->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = ISDestroy(&pcbddc->NeumannBoundaries);CHKERRQ(ierr);
  ierr = PetscObjectReference((PetscObject)NeumannBoundaries);CHKERRQ(ierr);
  pcbddc->NeumannBoundaries=NeumannBoundaries;
  PetscFunctionReturn(0);
}
EXTERN_C_END
#undef __FUNCT__  
#define __FUNCT__ "PCBDDCSetNeumannBoundaries"
/*@
 PCBDDCSetNeumannBoundaries - Set index set defining subdomain part (in local ordering)
                              of Neumann boundaries for the global problem.

   Not collective

   Input Parameters:
+  pc - the preconditioning context
-  NeumannBoundaries - sequential index set defining the subdomain part of Neumann boundaries (can be PETSC_NULL)

   Level: intermediate

   Notes:

.seealso: PCBDDC
@*/
PetscErrorCode PCBDDCSetNeumannBoundaries(PC pc,IS NeumannBoundaries)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  ierr = PetscTryMethod(pc,"PCBDDCSetNeumannBoundaries_C",(PC,IS),(pc,NeumannBoundaries));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
/* -------------------------------------------------------------------------- */
EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PCBDDCGetDirichletBoundaries_BDDC"
static PetscErrorCode PCBDDCGetDirichletBoundaries_BDDC(PC pc,IS *DirichletBoundaries)
{
  PC_BDDC  *pcbddc = (PC_BDDC*)pc->data;

  PetscFunctionBegin;
  *DirichletBoundaries = pcbddc->DirichletBoundaries;
  PetscFunctionReturn(0);
}
EXTERN_C_END
#undef __FUNCT__  
#define __FUNCT__ "PCBDDCGetDirichletBoundaries"
/*@
 PCBDDCGetDirichletBoundaries - Get index set defining subdomain part (in local ordering)
                                of Dirichlet boundaries for the global problem.

   Not collective

   Input Parameters:
+  pc - the preconditioning context

   Output Parameters:
+  DirichletBoundaries - index set defining the subdomain part of Dirichlet boundaries

   Level: intermediate

   Notes:

.seealso: PCBDDC
@*/
PetscErrorCode PCBDDCGetDirichletBoundaries(PC pc,IS *DirichletBoundaries)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  ierr = PetscUseMethod(pc,"PCBDDCGetDirichletBoundaries_C",(PC,IS*),(pc,DirichletBoundaries));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
/* -------------------------------------------------------------------------- */
EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PCBDDCGetNeumannBoundaries_BDDC"
static PetscErrorCode PCBDDCGetNeumannBoundaries_BDDC(PC pc,IS *NeumannBoundaries)
{
  PC_BDDC  *pcbddc = (PC_BDDC*)pc->data;

  PetscFunctionBegin;
  *NeumannBoundaries = pcbddc->NeumannBoundaries;
  PetscFunctionReturn(0);
}
EXTERN_C_END
#undef __FUNCT__  
#define __FUNCT__ "PCBDDCGetNeumannBoundaries"
/*@
 PCBDDCGetNeumannBoundaries - Get index set defining subdomain part (in local ordering)
                              of Neumann boundaries for the global problem.

   Not collective

   Input Parameters:
+  pc - the preconditioning context

   Output Parameters:
+  NeumannBoundaries - index set defining the subdomain part of Neumann boundaries

   Level: intermediate

   Notes:

.seealso: PCBDDC
@*/
PetscErrorCode PCBDDCGetNeumannBoundaries(PC pc,IS *NeumannBoundaries)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  ierr = PetscUseMethod(pc,"PCBDDCGetNeumannBoundaries_C",(PC,IS*),(pc,NeumannBoundaries));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
/* -------------------------------------------------------------------------- */
EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PCBDDCSetLocalAdjacencyGraph_BDDC"
static PetscErrorCode PCBDDCSetLocalAdjacencyGraph_BDDC(PC pc, PetscInt nvtxs,const PetscInt xadj[],const PetscInt adjncy[], PetscCopyMode copymode)
{
  PC_BDDC        *pcbddc = (PC_BDDC*)pc->data;
  PCBDDCGraph    mat_graph=pcbddc->mat_graph;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  mat_graph->nvtxs=nvtxs;
  ierr = PetscFree(mat_graph->xadj);CHKERRQ(ierr);
  ierr = PetscFree(mat_graph->adjncy);CHKERRQ(ierr);
  if (copymode == PETSC_COPY_VALUES) {
    ierr = PetscMalloc((mat_graph->nvtxs+1)*sizeof(PetscInt),&mat_graph->xadj);CHKERRQ(ierr);
    ierr = PetscMalloc(xadj[mat_graph->nvtxs]*sizeof(PetscInt),&mat_graph->adjncy);CHKERRQ(ierr);
    ierr = PetscMemcpy(mat_graph->xadj,xadj,(mat_graph->nvtxs+1)*sizeof(PetscInt));CHKERRQ(ierr);
    ierr = PetscMemcpy(mat_graph->adjncy,adjncy,xadj[mat_graph->nvtxs]*sizeof(PetscInt));CHKERRQ(ierr);
  } else if (copymode == PETSC_OWN_POINTER) {
    mat_graph->xadj = (PetscInt*)xadj;
    mat_graph->adjncy = (PetscInt*)adjncy;
  } else {
    SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_SUP,"Unsupported copy mode %d in %s\n",copymode,__FUNCT__);
  }
  PetscFunctionReturn(0);
}
EXTERN_C_END
#undef __FUNCT__  
#define __FUNCT__ "PCBDDCSetLocalAdjacencyGraph"
/*@
 PCBDDCSetLocalAdjacencyGraph - Set CSR graph of local matrix for use of PCBDDC.

   Not collective

   Input Parameters:
+  pc - the preconditioning context
-  nvtxs - number of local vertices of the graph
-  xadj, adjncy - the CSR graph
-  copymode - either PETSC_COPY_VALUES or PETSC_OWN_POINTER. In the former case the user must free the array passed in;
                                                             in the latter case, memory must be obtained with PetscMalloc.

   Level: intermediate

   Notes:

.seealso: PCBDDC
@*/
PetscErrorCode PCBDDCSetLocalAdjacencyGraph(PC pc,PetscInt nvtxs,const PetscInt xadj[],const PetscInt adjncy[], PetscCopyMode copymode)
{
  PetscInt       nrows,ncols;
  Mat_IS         *matis = (Mat_IS*)pc->pmat->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  ierr = MatGetSize(matis->A,&nrows,&ncols);CHKERRQ(ierr);
  if (nvtxs != nrows) {
    SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Local adjacency size %d passed in %s differs from local problem size %d!\n",nvtxs,__FUNCT__,nrows);
  } else {
    ierr = PetscTryMethod(pc,"PCBDDCSetLocalAdjacencyGraph_C",(PC,PetscInt,const PetscInt[],const PetscInt[],PetscCopyMode),(pc,nvtxs,xadj,adjncy,copymode));CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
/* -------------------------------------------------------------------------- */
EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PCBDDCSetDofsSplitting_BDDC"
static PetscErrorCode PCBDDCSetDofsSplitting_BDDC(PC pc,PetscInt n_is, IS ISForDofs[])
{
  PC_BDDC  *pcbddc = (PC_BDDC*)pc->data;
  PetscInt i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* Destroy ISes if they were already set */
  for (i=0;i<pcbddc->n_ISForDofs;i++) {
    ierr = ISDestroy(&pcbddc->ISForDofs[i]);CHKERRQ(ierr);
  }
  ierr = PetscFree(pcbddc->ISForDofs);CHKERRQ(ierr);
  /* allocate space then set */
  ierr = PetscMalloc(n_is*sizeof(IS),&pcbddc->ISForDofs);CHKERRQ(ierr);
  for (i=0;i<n_is;i++) {
    ierr = PetscObjectReference((PetscObject)ISForDofs[i]);CHKERRQ(ierr);
    pcbddc->ISForDofs[i]=ISForDofs[i];
  }
  pcbddc->n_ISForDofs=n_is;
  PetscFunctionReturn(0);
}
EXTERN_C_END
#undef __FUNCT__  
#define __FUNCT__ "PCBDDCSetDofsSplitting"
/*@
 PCBDDCSetDofsSplitting - Set index sets defining fields of local mat.

   Not collective

   Input Parameters:
+  pc - the preconditioning context
-  n - number of index sets defining the fields
-  IS[] - array of IS describing the fields

   Level: intermediate

   Notes:

.seealso: PCBDDC
@*/
PetscErrorCode PCBDDCSetDofsSplitting(PC pc,PetscInt n_is, IS ISForDofs[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  ierr = PetscTryMethod(pc,"PCBDDCSetDofsSplitting_C",(PC,PetscInt,IS[]),(pc,n_is,ISForDofs));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
/* -------------------------------------------------------------------------- */
#undef __FUNCT__  
#define __FUNCT__ "PCPreSolve_BDDC"
/* -------------------------------------------------------------------------- */
/*
   PCPreSolve_BDDC - Changes the right hand side and (if necessary) the initial
                     guess if a transformation of basis approach has been selected.

   Input Parameter:
+  pc - the preconditioner contex

   Application Interface Routine: PCPreSolve()

   Notes:
   The interface routine PCPreSolve() is not usually called directly by
   the user, but instead is called by KSPSolve().
*/
static PetscErrorCode PCPreSolve_BDDC(PC pc, KSP ksp, Vec rhs, Vec x)
{
  PetscErrorCode ierr;
  PC_BDDC        *pcbddc = (PC_BDDC*)pc->data;
  PC_IS          *pcis = (PC_IS*)(pc->data);
  Mat_IS         *matis = (Mat_IS*)pc->pmat->data;
  Mat            temp_mat;
  IS             dirIS;
  PetscInt       dirsize,i,*is_indices;
  PetscScalar    *array_x,*array_diagonal;
  Vec            used_vec;
  PetscBool      guess_nonzero;

  PetscFunctionBegin;
  if (x) {
    ierr = PetscObjectReference((PetscObject)x);CHKERRQ(ierr);
    used_vec = x;
  } else {
    ierr = PetscObjectReference((PetscObject)pcbddc->temp_solution);CHKERRQ(ierr);
    used_vec = pcbddc->temp_solution;
    ierr = VecSet(used_vec,0.0);CHKERRQ(ierr);
  }
  /* hack into ksp data structure PCPreSolve comes earlier in src/ksp/ksp/interface/itfunc.c */
  if (ksp) {
    ierr = KSPGetInitialGuessNonzero(ksp,&guess_nonzero);CHKERRQ(ierr);
    if ( !guess_nonzero ) {
      ierr = VecSet(used_vec,0.0);CHKERRQ(ierr);
    }
  }
  /* store the original rhs */
  ierr = VecCopy(rhs,pcbddc->original_rhs);CHKERRQ(ierr);

  /* Take into account zeroed rows -> change rhs and store solution removed */    
  ierr = MatGetDiagonal(pc->pmat,pcis->vec1_global);CHKERRQ(ierr);
  ierr = VecPointwiseDivide(pcis->vec1_global,rhs,pcis->vec1_global);CHKERRQ(ierr);
  ierr = VecScatterBegin(matis->ctx,pcis->vec1_global,pcis->vec2_N,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd  (matis->ctx,pcis->vec1_global,pcis->vec2_N,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterBegin(matis->ctx,used_vec,pcis->vec1_N,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd  (matis->ctx,used_vec,pcis->vec1_N,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = PCBDDCGetDirichletBoundaries(pc,&dirIS);CHKERRQ(ierr);
  if (dirIS) {
    ierr = ISGetSize(dirIS,&dirsize);CHKERRQ(ierr);
    ierr = VecGetArray(pcis->vec1_N,&array_x);CHKERRQ(ierr);
    ierr = VecGetArray(pcis->vec2_N,&array_diagonal);CHKERRQ(ierr);
    ierr = ISGetIndices(dirIS,(const PetscInt**)&is_indices);CHKERRQ(ierr);
    for (i=0;i<dirsize;i++) {
      array_x[is_indices[i]]=array_diagonal[is_indices[i]];
    }
    ierr = ISRestoreIndices(dirIS,(const PetscInt**)&is_indices);CHKERRQ(ierr);
    ierr = VecRestoreArray(pcis->vec2_N,&array_diagonal);CHKERRQ(ierr);
    ierr = VecRestoreArray(pcis->vec1_N,&array_x);CHKERRQ(ierr);
  }
  ierr = VecScatterBegin(matis->ctx,pcis->vec1_N,used_vec,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  ierr = VecScatterEnd  (matis->ctx,pcis->vec1_N,used_vec,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);

  /* remove the computed solution from the rhs */
  ierr = VecScale(used_vec,-1.0);CHKERRQ(ierr);
  ierr = MatMultAdd(pc->pmat,used_vec,rhs,rhs);CHKERRQ(ierr);
  ierr = VecScale(used_vec,-1.0);CHKERRQ(ierr);

  /* store partially computed solution and set initial guess */
  if (x) {
    ierr = VecCopy(used_vec,pcbddc->temp_solution);CHKERRQ(ierr);
    ierr = VecSet(used_vec,0.0);CHKERRQ(ierr);
    if (pcbddc->use_exact_dirichlet) {
      ierr = VecScatterBegin(pcis->global_to_D,rhs,pcis->vec1_D,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
      ierr = VecScatterEnd  (pcis->global_to_D,rhs,pcis->vec1_D,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
      ierr = KSPSolve(pcbddc->ksp_D,pcis->vec1_D,pcis->vec2_D);CHKERRQ(ierr);
      ierr = VecScatterBegin(pcis->global_to_D,pcis->vec2_D,used_vec,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
      ierr = VecScatterEnd  (pcis->global_to_D,pcis->vec2_D,used_vec,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
      if (ksp) {
        ierr = KSPSetInitialGuessNonzero(ksp,PETSC_TRUE);CHKERRQ(ierr);
      }
    }
  }

  /* rhs change of basis */
  if (pcbddc->usechangeofbasis) {
    /* swap pointers for local matrices */
    temp_mat = matis->A;
    matis->A = pcbddc->local_mat;
    pcbddc->local_mat = temp_mat;
    /* Get local rhs and apply transformation of basis */
    ierr = VecScatterBegin(pcis->global_to_B,rhs,pcis->vec1_B,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterEnd  (pcis->global_to_B,rhs,pcis->vec1_B,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    /* from original basis to modified basis */
    ierr = MatMultTranspose(pcbddc->ChangeOfBasisMatrix,pcis->vec1_B,pcis->vec2_B);CHKERRQ(ierr);
    /* put back modified values into the global vec using INSERT_VALUES copy mode */
    ierr = VecScatterBegin(pcis->global_to_B,pcis->vec2_B,rhs,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
    ierr = VecScatterEnd  (pcis->global_to_B,pcis->vec2_B,rhs,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
    if (ksp && pcbddc->NullSpace) {
      ierr = MatNullSpaceRemove(pcbddc->NullSpace,used_vec,PETSC_NULL);CHKERRQ(ierr);
      ierr = MatNullSpaceRemove(pcbddc->NullSpace,rhs,PETSC_NULL);CHKERRQ(ierr);
    }
  }
  ierr = VecDestroy(&used_vec);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
/* -------------------------------------------------------------------------- */
#undef __FUNCT__  
#define __FUNCT__ "PCPostSolve_BDDC"
/* -------------------------------------------------------------------------- */
/*
   PCPostSolve_BDDC - Changes the computed solution if a transformation of basis
                     approach has been selected. Also, restores rhs to its original state.

   Input Parameter:
+  pc - the preconditioner contex

   Application Interface Routine: PCPostSolve()

   Notes:
   The interface routine PCPostSolve() is not usually called directly by
   the user, but instead is called by KSPSolve().
*/
static PetscErrorCode PCPostSolve_BDDC(PC pc, KSP ksp, Vec rhs, Vec x)
{
  PetscErrorCode ierr;
  PC_BDDC        *pcbddc = (PC_BDDC*)pc->data;
  PC_IS          *pcis = (PC_IS*)(pc->data);
  Mat_IS         *matis = (Mat_IS*)pc->pmat->data;
  Mat            temp_mat;

  PetscFunctionBegin;
  if (pcbddc->usechangeofbasis) {
    /* swap pointers for local matrices */
    temp_mat = matis->A;
    matis->A = pcbddc->local_mat;
    pcbddc->local_mat = temp_mat;
    /* restore rhs to its original state */
    if (rhs) {
      ierr = VecCopy(pcbddc->original_rhs,rhs);CHKERRQ(ierr);
    }
    /* Get Local boundary and apply transformation of basis to solution vector */
    ierr = VecScatterBegin(pcis->global_to_B,x,pcis->vec1_B,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterEnd  (pcis->global_to_B,x,pcis->vec1_B,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    /* from modified basis to original basis */
    ierr = MatMult(pcbddc->ChangeOfBasisMatrix,pcis->vec1_B,pcis->vec2_B);CHKERRQ(ierr);
    /* put back modified values into the global vec using INSERT_VALUES copy mode */
    ierr = VecScatterBegin(pcis->global_to_B,pcis->vec2_B,x,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
    ierr = VecScatterEnd  (pcis->global_to_B,pcis->vec2_B,x,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  }
  /* add solution removed in presolve */
  if (x) {
    ierr = VecAXPY(x,1.0,pcbddc->temp_solution);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
/* -------------------------------------------------------------------------- */
#undef __FUNCT__  
#define __FUNCT__ "PCSetUp_BDDC"
/* -------------------------------------------------------------------------- */
/*
   PCSetUp_BDDC - Prepares for the use of the BDDC preconditioner
                  by setting data structures and options.   

   Input Parameter:
+  pc - the preconditioner context

   Application Interface Routine: PCSetUp()

   Notes:
   The interface routine PCSetUp() is not usually called directly by
   the user, but instead is called by PCApply() if necessary.
*/
PetscErrorCode PCSetUp_BDDC(PC pc)
{
  PetscErrorCode ierr;
  PC_BDDC*       pcbddc   = (PC_BDDC*)pc->data;

  PetscFunctionBegin;
  if (!pc->setupcalled) {
    /* For BDDC we need to define a local "Neumann" problem different to that defined in PCISSetup
       So, we set to pcnone the Neumann problem of pcis in order to avoid unneeded computation
       Also, we decide to directly build the (same) Dirichlet problem */
    ierr = PetscOptionsSetValue("-is_localN_pc_type","none");CHKERRQ(ierr);
    ierr = PetscOptionsSetValue("-is_localD_pc_type","none");CHKERRQ(ierr);
    /* Set up all the "iterative substructuring" common block */
    
    ierr = PCISSetUp(pc);CHKERRQ(ierr);
    /* Get stdout for dbg */
    if (pcbddc->dbg_flag) {
      ierr = PetscViewerASCIIGetStdout(((PetscObject)pc)->comm,&pcbddc->dbg_viewer);CHKERRQ(ierr);
      ierr = PetscViewerASCIISynchronizedAllow(pcbddc->dbg_viewer,PETSC_TRUE);CHKERRQ(ierr);
    }
    /* Analyze local interface */
    ierr = PCBDDCManageLocalBoundaries(pc);CHKERRQ(ierr); 
    /* Set up local constraint matrix */
    ierr = PCBDDCCreateConstraintMatrix(pc);CHKERRQ(ierr);
    /* Create coarse and local stuffs used for evaluating action of preconditioner */
    ierr = PCBDDCCoarseSetUp(pc);CHKERRQ(ierr);
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
PetscErrorCode PCApply_BDDC(PC pc,Vec r,Vec z)
{
  PC_IS             *pcis = (PC_IS*)(pc->data);
  PC_BDDC           *pcbddc = (PC_BDDC*)(pc->data);
  PetscErrorCode    ierr;
  const PetscScalar one = 1.0;
  const PetscScalar m_one = -1.0;
  const PetscScalar zero = 0.0;

/* This code is similar to that provided in nn.c for PCNN
   NN interface preconditioner changed to BDDC
   Added support for M_3 preconditioenr in the reference article (code is active if pcbddc->prec_type = PETSC_TRUE) */

  PetscFunctionBegin;
  if (!pcbddc->use_exact_dirichlet) {
    /* First Dirichlet solve */
    ierr = VecScatterBegin(pcis->global_to_D,r,pcis->vec1_D,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterEnd  (pcis->global_to_D,r,pcis->vec1_D,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = KSPSolve(pcbddc->ksp_D,pcis->vec1_D,pcis->vec2_D);CHKERRQ(ierr);
    /*
      Assembling right hand side for BDDC operator
      - vec1_D for the Dirichlet part (if needed, i.e. prec_flag=PETSC_TRUE)
      - the interface part of the global vector z
    */
    ierr = VecScale(pcis->vec2_D,m_one);CHKERRQ(ierr);
    ierr = MatMult(pcis->A_BI,pcis->vec2_D,pcis->vec1_B);CHKERRQ(ierr);
    if (pcbddc->prec_type) { ierr = MatMultAdd(pcis->A_II,pcis->vec2_D,pcis->vec1_D,pcis->vec1_D);CHKERRQ(ierr); }
    ierr = VecScale(pcis->vec2_D,m_one);CHKERRQ(ierr);
    ierr = VecCopy(r,z);CHKERRQ(ierr);
    ierr = VecScatterBegin(pcis->global_to_B,pcis->vec1_B,z,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
    ierr = VecScatterEnd  (pcis->global_to_B,pcis->vec1_B,z,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
    ierr = VecScatterBegin(pcis->global_to_B,z,pcis->vec1_B,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterEnd  (pcis->global_to_B,z,pcis->vec1_B,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  } else {
    ierr = VecScatterBegin(pcis->global_to_B,r,pcis->vec1_B,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterEnd  (pcis->global_to_B,r,pcis->vec1_B,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecSet(pcis->vec1_D,zero);CHKERRQ(ierr);
    ierr = VecSet(pcis->vec2_D,zero);CHKERRQ(ierr);
  }

  /* Apply partition of unity */
  ierr = VecPointwiseMult(pcis->vec1_B,pcis->D,pcis->vec1_B);CHKERRQ(ierr);

  /* Apply interface preconditioner
     input/output vecs: pcis->vec1_B and pcis->vec1_D */
  ierr = PCBDDCApplyInterfacePreconditioner(pc);CHKERRQ(ierr);

  /* Apply partition of unity and sum boundary values */
  ierr = VecPointwiseMult(pcis->vec1_B,pcis->D,pcis->vec1_B);CHKERRQ(ierr);
  ierr = VecSet(z,zero);CHKERRQ(ierr);
  ierr = VecScatterBegin(pcis->global_to_B,pcis->vec1_B,z,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  ierr = VecScatterEnd  (pcis->global_to_B,pcis->vec1_B,z,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);

  /* Second Dirichlet solve and assembling of output */
  ierr = VecScatterBegin(pcis->global_to_B,z,pcis->vec1_B,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd  (pcis->global_to_B,z,pcis->vec1_B,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = MatMult(pcis->A_IB,pcis->vec1_B,pcis->vec3_D);CHKERRQ(ierr);
  if (pcbddc->prec_type) { ierr = MatMultAdd(pcis->A_II,pcis->vec1_D,pcis->vec3_D,pcis->vec3_D);CHKERRQ(ierr); }
  ierr = KSPSolve(pcbddc->ksp_D,pcis->vec3_D,pcbddc->vec4_D);CHKERRQ(ierr);
  ierr = VecScale(pcbddc->vec4_D,m_one);CHKERRQ(ierr);
  if (pcbddc->prec_type) { ierr = VecAXPY (pcbddc->vec4_D,one,pcis->vec1_D);CHKERRQ(ierr); } 
  ierr = VecAXPY (pcis->vec2_D,one,pcbddc->vec4_D);CHKERRQ(ierr); 
  ierr = VecScatterBegin(pcis->global_to_D,pcis->vec2_D,z,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  ierr = VecScatterEnd  (pcis->global_to_D,pcis->vec2_D,z,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  PetscFunctionReturn(0);

}
/* -------------------------------------------------------------------------- */
#undef __FUNCT__  
#define __FUNCT__ "PCDestroy_BDDC"
PetscErrorCode PCDestroy_BDDC(PC pc)
{
  PC_BDDC        *pcbddc = (PC_BDDC*)pc->data;
  PetscInt       i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* free data created by PCIS */
  ierr = PCISDestroy(pc);CHKERRQ(ierr);
  /* free BDDC data  */
  ierr = MatNullSpaceDestroy(&pcbddc->CoarseNullSpace);CHKERRQ(ierr);
  ierr = MatNullSpaceDestroy(&pcbddc->NullSpace);CHKERRQ(ierr);
  ierr = VecDestroy(&pcbddc->temp_solution);CHKERRQ(ierr);
  ierr = VecDestroy(&pcbddc->original_rhs);CHKERRQ(ierr);
  ierr = MatDestroy(&pcbddc->local_mat);CHKERRQ(ierr);
  ierr = MatDestroy(&pcbddc->ChangeOfBasisMatrix);CHKERRQ(ierr);
  ierr = VecDestroy(&pcbddc->coarse_vec);CHKERRQ(ierr);
  ierr = VecDestroy(&pcbddc->coarse_rhs);CHKERRQ(ierr);
  ierr = KSPDestroy(&pcbddc->coarse_ksp);CHKERRQ(ierr);
  ierr = MatDestroy(&pcbddc->coarse_mat);CHKERRQ(ierr);
  ierr = MatDestroy(&pcbddc->coarse_phi_B);CHKERRQ(ierr);
  ierr = MatDestroy(&pcbddc->coarse_phi_D);CHKERRQ(ierr);
  ierr = VecDestroy(&pcbddc->vec1_P);CHKERRQ(ierr);
  ierr = VecDestroy(&pcbddc->vec1_C);CHKERRQ(ierr);
  ierr = MatDestroy(&pcbddc->local_auxmat1);CHKERRQ(ierr);
  ierr = MatDestroy(&pcbddc->local_auxmat2);CHKERRQ(ierr);
  ierr = VecDestroy(&pcbddc->vec1_R);CHKERRQ(ierr);
  ierr = VecDestroy(&pcbddc->vec2_R);CHKERRQ(ierr);
  ierr = VecDestroy(&pcbddc->vec4_D);CHKERRQ(ierr);
  ierr = VecScatterDestroy(&pcbddc->R_to_B);CHKERRQ(ierr);
  ierr = VecScatterDestroy(&pcbddc->R_to_D);CHKERRQ(ierr);
  ierr = VecScatterDestroy(&pcbddc->coarse_loc_to_glob);CHKERRQ(ierr);
  ierr = KSPDestroy(&pcbddc->ksp_D);CHKERRQ(ierr);
  ierr = KSPDestroy(&pcbddc->ksp_R);CHKERRQ(ierr);
  ierr = ISDestroy(&pcbddc->NeumannBoundaries);CHKERRQ(ierr);
  ierr = ISDestroy(&pcbddc->DirichletBoundaries);CHKERRQ(ierr);
  ierr = MatDestroy(&pcbddc->ConstraintMatrix);CHKERRQ(ierr);
  ierr = PetscFree(pcbddc->local_primal_indices);CHKERRQ(ierr);
  ierr = PetscFree(pcbddc->replicated_local_primal_indices);CHKERRQ(ierr);
  ierr = PetscFree(pcbddc->replicated_local_primal_values);CHKERRQ(ierr);
  ierr = PetscFree(pcbddc->local_primal_displacements);CHKERRQ(ierr);
  ierr = PetscFree(pcbddc->local_primal_sizes);CHKERRQ(ierr);
  for (i=0;i<pcbddc->n_ISForDofs;i++) { ierr = ISDestroy(&pcbddc->ISForDofs[i]);CHKERRQ(ierr); }
  ierr = PetscFree(pcbddc->ISForDofs);CHKERRQ(ierr);
  for (i=0;i<pcbddc->n_ISForFaces;i++) { ierr = ISDestroy(&pcbddc->ISForFaces[i]);CHKERRQ(ierr); }
  ierr = PetscFree(pcbddc->ISForFaces);CHKERRQ(ierr);
  for (i=0;i<pcbddc->n_ISForEdges;i++) { ierr = ISDestroy(&pcbddc->ISForEdges[i]);CHKERRQ(ierr); }
  ierr = PetscFree(pcbddc->ISForEdges);CHKERRQ(ierr);
  ierr = ISDestroy(&pcbddc->ISForVertices);CHKERRQ(ierr);
  /* Free graph structure */
  ierr = PetscFree(pcbddc->mat_graph->xadj);CHKERRQ(ierr);
  ierr = PetscFree(pcbddc->mat_graph->adjncy);CHKERRQ(ierr);
  if (pcbddc->mat_graph->nvtxs) {
    ierr = PetscFree(pcbddc->mat_graph->neighbours_set[0]);CHKERRQ(ierr);
  }
  ierr = PetscFree(pcbddc->mat_graph->neighbours_set);CHKERRQ(ierr);
  ierr = PetscFree4(pcbddc->mat_graph->where,pcbddc->mat_graph->count,pcbddc->mat_graph->cptr,pcbddc->mat_graph->queue);CHKERRQ(ierr);
  ierr = PetscFree2(pcbddc->mat_graph->which_dof,pcbddc->mat_graph->touched);CHKERRQ(ierr);
  ierr = PetscFree(pcbddc->mat_graph->where_ncmps);CHKERRQ(ierr);
  ierr = PetscFree(pcbddc->mat_graph);CHKERRQ(ierr);
  /* remove functions */
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCBDDCSetCoarseningRatio_C","",PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCBDDCSetMaxLevels_C","",PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCBDDCSetNullSpace_C","",PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCBDDCSetDirichletBoundaries_C","",PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCBDDCSetNeumannBoundaries_C","",PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCBDDCGetDirichletBoundaries_C","",PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCBDDCGetNeumannBoundaries_C","",PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCBDDCSetCoarseProblemType_C","",PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCBDDCSetDofsSplitting_C","",PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCBDDCSetLocalAdjacencyGraph_C","",PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCPreSolve_C","",PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCPostSolve_C","",PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCBDDCCreateFETIDPOperators_C","",PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCBDDCMatFETIDPGetRHS_C","",PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCBDDCMatFETIDPGetSolution_C","",PETSC_NULL);CHKERRQ(ierr);
  /* Free the private data structure that was hanging off the PC */
  ierr = PetscFree(pcbddc);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
/* -------------------------------------------------------------------------- */
EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PCBDDCMatFETIDPGetRHS_BDDC"
static PetscErrorCode PCBDDCMatFETIDPGetRHS_BDDC(Mat fetidp_mat, Vec standard_rhs, Vec fetidp_flux_rhs)
{
  FETIDPMat_ctx  *mat_ctx;
  PC_IS*         pcis;
  PC_BDDC*       pcbddc;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatShellGetContext(fetidp_mat,&mat_ctx);CHKERRQ(ierr);
  pcis = (PC_IS*)mat_ctx->pc->data;
  pcbddc = (PC_BDDC*)mat_ctx->pc->data;

  /* change of basis for physical rhs if needed
     It also changes the rhs in case of dirichlet boundaries */
  (*mat_ctx->pc->ops->presolve)(mat_ctx->pc,PETSC_NULL,standard_rhs,PETSC_NULL);
  /* store vectors for computation of fetidp final solution */
  ierr = VecScatterBegin(pcis->global_to_D,standard_rhs,mat_ctx->temp_solution_D,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd  (pcis->global_to_D,standard_rhs,mat_ctx->temp_solution_D,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterBegin(pcis->global_to_B,standard_rhs,mat_ctx->temp_solution_B,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd  (pcis->global_to_B,standard_rhs,mat_ctx->temp_solution_B,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  /* scale rhs since it should be unassembled */
  ierr = VecPointwiseMult(mat_ctx->temp_solution_B,pcis->D,mat_ctx->temp_solution_B);CHKERRQ(ierr);
  if (!pcbddc->prec_type) {
    /* compute partially subassembled Schur complement right-hand side */
    ierr = KSPSolve(pcbddc->ksp_D,mat_ctx->temp_solution_D,pcis->vec1_D);CHKERRQ(ierr);
    ierr = MatMult(pcis->A_BI,pcis->vec1_D,pcis->vec1_B);CHKERRQ(ierr);
    ierr = VecAXPY(mat_ctx->temp_solution_B,-1.0,pcis->vec1_B);CHKERRQ(ierr);
    ierr = VecSet(standard_rhs,0.0);CHKERRQ(ierr);
    ierr = VecScatterBegin(pcis->global_to_B,mat_ctx->temp_solution_B,standard_rhs,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
    ierr = VecScatterEnd  (pcis->global_to_B,mat_ctx->temp_solution_B,standard_rhs,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
    ierr = VecScatterBegin(pcis->global_to_B,standard_rhs,mat_ctx->temp_solution_B,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterEnd  (pcis->global_to_B,standard_rhs,mat_ctx->temp_solution_B,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecPointwiseMult(mat_ctx->temp_solution_B,pcis->D,mat_ctx->temp_solution_B);CHKERRQ(ierr);
  }
  /* BDDC rhs */
  ierr = VecCopy(mat_ctx->temp_solution_B,pcis->vec1_B);CHKERRQ(ierr);
  if (pcbddc->prec_type) {
    ierr = VecCopy(mat_ctx->temp_solution_D,pcis->vec1_D);CHKERRQ(ierr);
  }
  /* apply BDDC */
  ierr = PCBDDCApplyInterfacePreconditioner(mat_ctx->pc);CHKERRQ(ierr);
  /* Application of B_delta and assembling of rhs for fetidp fluxes */
  ierr = VecSet(fetidp_flux_rhs,0.0);CHKERRQ(ierr);
  ierr = MatMult(mat_ctx->B_delta,pcis->vec1_B,mat_ctx->lambda_local);CHKERRQ(ierr);
  ierr = VecScatterBegin(mat_ctx->l2g_lambda,mat_ctx->lambda_local,fetidp_flux_rhs,ADD_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd  (mat_ctx->l2g_lambda,mat_ctx->lambda_local,fetidp_flux_rhs,ADD_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  /* restore original rhs */
  ierr = VecCopy(pcbddc->original_rhs,standard_rhs);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END
#undef __FUNCT__  
#define __FUNCT__ "PCBDDCMatFETIDPGetRHS"
/*@
 PCBDDCMatFETIDPGetRHS - Get rhs for FETIDP linear system.

   Collective

   Input Parameters:
+  fetidp_mat   - the FETIDP mat obtained by a call to PCBDDCCreateFETIDPOperators
+  standard_rhs - the rhs of your linear system 
  
   Output Parameters:
+  fetidp_flux_rhs   - the rhs of the FETIDP linear system

   Level: developer

   Notes:

.seealso: PCBDDC
@*/
PetscErrorCode PCBDDCMatFETIDPGetRHS(Mat fetidp_mat, Vec standard_rhs, Vec fetidp_flux_rhs)
{
  FETIDPMat_ctx  *mat_ctx;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatShellGetContext(fetidp_mat,&mat_ctx);CHKERRQ(ierr);
  ierr = PetscTryMethod(mat_ctx->pc,"PCBDDCMatFETIDPGetRHS_C",(Mat,Vec,Vec),(fetidp_mat,standard_rhs,fetidp_flux_rhs));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
/* -------------------------------------------------------------------------- */
EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PCBDDCMatFETIDPGetSolution_BDDC"
static PetscErrorCode PCBDDCMatFETIDPGetSolution_BDDC(Mat fetidp_mat, Vec fetidp_flux_sol, Vec standard_sol)
{
  FETIDPMat_ctx  *mat_ctx;
  PC_IS*         pcis;
  PC_BDDC*       pcbddc;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatShellGetContext(fetidp_mat,&mat_ctx);CHKERRQ(ierr);
  pcis = (PC_IS*)mat_ctx->pc->data;
  pcbddc = (PC_BDDC*)mat_ctx->pc->data;

  /* apply B_delta^T */
  ierr = VecScatterBegin(mat_ctx->l2g_lambda,fetidp_flux_sol,mat_ctx->lambda_local,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  ierr = VecScatterEnd  (mat_ctx->l2g_lambda,fetidp_flux_sol,mat_ctx->lambda_local,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  ierr = MatMultTranspose(mat_ctx->B_delta,mat_ctx->lambda_local,pcis->vec1_B);CHKERRQ(ierr);
  /* compute rhs for BDDC application */
  ierr = VecAYPX(pcis->vec1_B,-1.0,mat_ctx->temp_solution_B);CHKERRQ(ierr);
  if (pcbddc->prec_type) {
    ierr = VecCopy(mat_ctx->temp_solution_D,pcis->vec1_D);CHKERRQ(ierr);
  }
  /* apply BDDC */
  ierr = PCBDDCApplyInterfacePreconditioner(mat_ctx->pc);CHKERRQ(ierr);
  /* put values into standard global vector */
  ierr = VecScatterBegin(pcis->global_to_B,pcis->vec1_B,standard_sol,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  ierr = VecScatterEnd  (pcis->global_to_B,pcis->vec1_B,standard_sol,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  if (!pcbddc->prec_type) {
    /* compute values into the interior if solved for the partially subassembled Schur complement */
    ierr = MatMult(pcis->A_IB,pcis->vec1_B,pcis->vec1_D);CHKERRQ(ierr);
    ierr = VecAXPY(mat_ctx->temp_solution_D,-1.0,pcis->vec1_D);CHKERRQ(ierr);
    ierr = KSPSolve(pcbddc->ksp_D,mat_ctx->temp_solution_D,pcis->vec1_D);CHKERRQ(ierr);
  }
  ierr = VecScatterBegin(pcis->global_to_D,pcis->vec1_D,standard_sol,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  ierr = VecScatterEnd  (pcis->global_to_D,pcis->vec1_D,standard_sol,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  /* final change of basis if needed
     Is also sums the dirichlet part removed during RHS assembling */
  (*mat_ctx->pc->ops->postsolve)(mat_ctx->pc,PETSC_NULL,PETSC_NULL,standard_sol);
  PetscFunctionReturn(0);

}
EXTERN_C_END
#undef __FUNCT__  
#define __FUNCT__ "PCBDDCMatFETIDPGetSolution"
/*@
 PCBDDCMatFETIDPGetSolution - Get Solution for FETIDP linear system.

   Collective

   Input Parameters:
+  fetidp_mat        - the FETIDP mat obtained by a call to PCBDDCCreateFETIDPOperators
+  fetidp_flux_sol - the solution of the FETIDP linear system
  
   Output Parameters:
+  standard_sol      - the solution on the global domain

   Level: developer

   Notes:

.seealso: PCBDDC
@*/
PetscErrorCode PCBDDCMatFETIDPGetSolution(Mat fetidp_mat, Vec fetidp_flux_sol, Vec standard_sol)
{
  FETIDPMat_ctx  *mat_ctx;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatShellGetContext(fetidp_mat,&mat_ctx);CHKERRQ(ierr);
  ierr = PetscTryMethod(mat_ctx->pc,"PCBDDCMatFETIDPGetSolution_C",(Mat,Vec,Vec),(fetidp_mat,fetidp_flux_sol,standard_sol));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
/* -------------------------------------------------------------------------- */
EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PCBDDCCreateFETIDPOperators_BDDC"
static PetscErrorCode PCBDDCCreateFETIDPOperators_BDDC(PC pc, Mat *fetidp_mat, PC *fetidp_pc)
{
  PETSC_EXTERN PetscErrorCode FETIDPMatMult(Mat,Vec,Vec);
  PETSC_EXTERN PetscErrorCode PCBDDCDestroyFETIDPMat(Mat);
  PETSC_EXTERN PetscErrorCode FETIDPPCApply(PC,Vec,Vec);
  PETSC_EXTERN PetscErrorCode PCBDDCDestroyFETIDPPC(PC);

  FETIDPMat_ctx  *fetidpmat_ctx;
  Mat            newmat;
  FETIDPPC_ctx  *fetidppc_ctx;
  PC             newpc;
  MPI_Comm       comm = ((PetscObject)pc)->comm;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* FETIDP linear matrix */
  ierr = PCBDDCCreateFETIDPMatContext(pc, &fetidpmat_ctx);CHKERRQ(ierr);
  ierr = PCBDDCSetupFETIDPMatContext(fetidpmat_ctx);CHKERRQ(ierr);
  ierr = MatCreateShell(comm,PETSC_DECIDE,PETSC_DECIDE,fetidpmat_ctx->n_lambda,fetidpmat_ctx->n_lambda,fetidpmat_ctx,&newmat);CHKERRQ(ierr);
  ierr = MatShellSetOperation(newmat,MATOP_MULT,(void (*)(void))FETIDPMatMult);CHKERRQ(ierr);
  ierr = MatShellSetOperation(newmat,MATOP_DESTROY,(void (*)(void))PCBDDCDestroyFETIDPMat);CHKERRQ(ierr);
  ierr = MatSetUp(newmat);CHKERRQ(ierr);
  /* FETIDP preconditioner */
  ierr = PCBDDCCreateFETIDPPCContext(pc, &fetidppc_ctx);CHKERRQ(ierr);
  ierr = PCBDDCSetupFETIDPPCContext(newmat,fetidppc_ctx);CHKERRQ(ierr);
  ierr = PCCreate(comm,&newpc);CHKERRQ(ierr);
  ierr = PCSetType(newpc,PCSHELL);CHKERRQ(ierr);
  ierr = PCShellSetContext(newpc,fetidppc_ctx);CHKERRQ(ierr);
  ierr = PCShellSetApply(newpc,FETIDPPCApply);CHKERRQ(ierr);
  ierr = PCShellSetDestroy(newpc,PCBDDCDestroyFETIDPPC);CHKERRQ(ierr);
  ierr = PCSetOperators(newpc,newmat,newmat,SAME_PRECONDITIONER);CHKERRQ(ierr);
  ierr = PCSetUp(newpc);CHKERRQ(ierr);
  /* return pointers for objects created */
  *fetidp_mat=newmat;
  *fetidp_pc=newpc;

  PetscFunctionReturn(0);
}
EXTERN_C_END
#undef __FUNCT__  
#define __FUNCT__ "PCBDDCCreateFETIDPOperators"
/*@
 PCBDDCCreateFETIDPOperators - Create operators for FETIDP.

   Collective

   Input Parameters:
+  pc - the BDDC preconditioning context (setup must be already called)

   Level: developer

   Notes:

.seealso: PCBDDC
@*/
PetscErrorCode PCBDDCCreateFETIDPOperators(PC pc, Mat *fetidp_mat, PC *fetidp_pc)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  if (pc->setupcalled) {
    ierr = PetscTryMethod(pc,"PCBDDCCreateFETIDPOperators_C",(PC,Mat*,PC*),(pc,fetidp_mat,fetidp_pc));CHKERRQ(ierr);
  } else {
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_SUP,"You must call PCSetup_BDDC before calling %s\n",__FUNCT__);
  }
  PetscFunctionReturn(0);
}
/* -------------------------------------------------------------------------- */
/*MC
   PCBDDC - Balancing Domain Decomposition by Constraints.

   Options Database Keys:
.    -pcbddc ??? -

   Level: intermediate

   Notes: The matrix used with this preconditioner must be of type MATIS 

          Unlike more 'conventional' interface preconditioners, this iterates over ALL the
          degrees of freedom, NOT just those on the interface (this allows the use of approximate solvers
          on the subdomains).

          Options for the coarse grid preconditioner can be set with -
          Options for the Dirichlet subproblem can be set with -
          Options for the Neumann subproblem can be set with -

   Contributed by Stefano Zampini

.seealso:  PCCreate(), PCSetType(), PCType (for list of available types), PC,  MATIS
M*/
EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PCCreate_BDDC"
PetscErrorCode PCCreate_BDDC(PC pc)
{
  PetscErrorCode ierr;
  PC_BDDC        *pcbddc;
  PCBDDCGraph    mat_graph;

  PetscFunctionBegin;
  /* Creates the private data structure for this preconditioner and attach it to the PC object. */
  ierr      = PetscNewLog(pc,PC_BDDC,&pcbddc);CHKERRQ(ierr);
  pc->data  = (void*)pcbddc;

  /* create PCIS data structure */
  ierr = PCISCreate(pc);CHKERRQ(ierr);

  /* BDDC specific */
  pcbddc->CoarseNullSpace            = 0;
  pcbddc->NullSpace                  = 0;
  pcbddc->temp_solution              = 0;
  pcbddc->original_rhs               = 0;
  pcbddc->local_mat                  = 0;
  pcbddc->ChangeOfBasisMatrix        = 0;
  pcbddc->usechangeofbasis           = PETSC_TRUE;
  pcbddc->usechangeonfaces           = PETSC_FALSE;
  pcbddc->coarse_vec                 = 0;
  pcbddc->coarse_rhs                 = 0;
  pcbddc->coarse_ksp                 = 0;
  pcbddc->coarse_phi_B               = 0;
  pcbddc->coarse_phi_D               = 0;
  pcbddc->vec1_P                     = 0;          
  pcbddc->vec1_R                     = 0; 
  pcbddc->vec2_R                     = 0; 
  pcbddc->local_auxmat1              = 0;
  pcbddc->local_auxmat2              = 0;
  pcbddc->R_to_B                     = 0;
  pcbddc->R_to_D                     = 0;
  pcbddc->ksp_D                      = 0;
  pcbddc->ksp_R                      = 0;
  pcbddc->local_primal_indices       = 0;
  pcbddc->prec_type                  = PETSC_FALSE;
  pcbddc->NeumannBoundaries          = 0;
  pcbddc->ISForDofs                  = 0;
  pcbddc->ISForVertices              = 0;
  pcbddc->n_ISForFaces               = 0;
  pcbddc->n_ISForEdges               = 0;
  pcbddc->ConstraintMatrix           = 0;
  pcbddc->use_nnsp_true              = PETSC_FALSE;
  pcbddc->local_primal_sizes         = 0;
  pcbddc->local_primal_displacements = 0;
  pcbddc->replicated_local_primal_indices = 0;
  pcbddc->replicated_local_primal_values  = 0;
  pcbddc->coarse_loc_to_glob         = 0;
  pcbddc->dbg_flag                   = PETSC_FALSE;
  pcbddc->coarsening_ratio           = 8;
  pcbddc->use_exact_dirichlet        = PETSC_TRUE;
  pcbddc->current_level              = 0;
  pcbddc->max_levels                 = 1;

  /* allocate and initialize needed graph structure */
  ierr = PetscMalloc(sizeof(*mat_graph),&pcbddc->mat_graph);CHKERRQ(ierr);
  pcbddc->mat_graph->xadj            = 0;
  pcbddc->mat_graph->adjncy          = 0;

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
  pc->ops->presolve            = PCPreSolve_BDDC;
  pc->ops->postsolve           = PCPostSolve_BDDC;

  /* composing function */
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCBDDCSetCoarseningRatio_C","PCBDDCSetCoarseningRatio_BDDC",
                    PCBDDCSetCoarseningRatio_BDDC);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCBDDCSetMaxLevels_C","PCBDDCSetMaxLevels_BDDC",
                    PCBDDCSetMaxLevels_BDDC);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCBDDCSetNullSpace_C","PCBDDCSetNullSpace_BDDC",
                    PCBDDCSetNullSpace_BDDC);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCBDDCSetDirichletBoundaries_C","PCBDDCSetDirichletBoundaries_BDDC",
                    PCBDDCSetDirichletBoundaries_BDDC);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCBDDCSetNeumannBoundaries_C","PCBDDCSetNeumannBoundaries_BDDC",
                    PCBDDCSetNeumannBoundaries_BDDC);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCBDDCGetDirichletBoundaries_C","PCBDDCGetDirichletBoundaries_BDDC",
                    PCBDDCGetDirichletBoundaries_BDDC);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCBDDCGetNeumannBoundaries_C","PCBDDCGetNeumannBoundaries_BDDC",
                    PCBDDCGetNeumannBoundaries_BDDC);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCBDDCSetCoarseProblemType_C","PCBDDCSetCoarseProblemType_BDDC",
                    PCBDDCSetCoarseProblemType_BDDC);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCBDDCSetDofsSplitting_C","PCBDDCSetDofsSplitting_BDDC",
                    PCBDDCSetDofsSplitting_BDDC);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCBDDCSetLocalAdjacencyGraph_C","PCBDDCSetLocalAdjacencyGraph_BDDC",
                    PCBDDCSetLocalAdjacencyGraph_BDDC);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCPreSolve_C","PCPreSolve_BDDC",
                    PCPreSolve_BDDC);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCPostSolve_C","PCPostSolve_BDDC",
                    PCPostSolve_BDDC);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCBDDCCreateFETIDPOperators_C","PCBDDCCreateFETIDPOperators_BDDC",
                    PCBDDCCreateFETIDPOperators_BDDC);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCBDDCMatFETIDPGetRHS_C","PCBDDCMatFETIDPGetRHS_BDDC",
                    PCBDDCMatFETIDPGetRHS_BDDC);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCBDDCMatFETIDPGetSolution_C","PCBDDCMatFETIDPGetSolution_BDDC",
                    PCBDDCMatFETIDPGetSolution_BDDC);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END

/* -------------------------------------------------------------------------- */
/* All static functions from now on                                           */
/* -------------------------------------------------------------------------- */
#undef __FUNCT__  
#define __FUNCT__ "PCBDDCSetUseExactDirichlet"
static PetscErrorCode PCBDDCSetUseExactDirichlet(PC pc,PetscBool use)
{
  PC_BDDC  *pcbddc = (PC_BDDC*)pc->data;

  PetscFunctionBegin;
  pcbddc->use_exact_dirichlet=use;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCBDDCSetLevel"
static PetscErrorCode PCBDDCSetLevel(PC pc,PetscInt level)
{
  PC_BDDC  *pcbddc = (PC_BDDC*)pc->data;

  PetscFunctionBegin;
  pcbddc->current_level=level;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCBDDCAdaptNullSpace"
static PetscErrorCode PCBDDCAdaptNullSpace(PC pc)
{
  PC_IS*         pcis = (PC_IS*)  (pc->data);
  PC_BDDC*       pcbddc = (PC_BDDC*)(pc->data);
  KSP            inv_change;
  PC             pc_change;
  const Vec      *nsp_vecs;
  Vec            *new_nsp_vecs;
  PetscInt       i,nsp_size,new_nsp_size,start_new;
  PetscBool      nsp_has_cnst;
  MatNullSpace   new_nsp;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatNullSpaceGetVecs(pcbddc->NullSpace,&nsp_has_cnst,&nsp_size,&nsp_vecs);CHKERRQ(ierr);
  ierr = KSPCreate(PETSC_COMM_SELF,&inv_change);CHKERRQ(ierr);
  ierr = KSPSetOperators(inv_change,pcbddc->ChangeOfBasisMatrix,pcbddc->ChangeOfBasisMatrix,SAME_PRECONDITIONER);CHKERRQ(ierr);
  ierr = KSPSetType(inv_change,KSPPREONLY);CHKERRQ(ierr);
  ierr = KSPGetPC(inv_change,&pc_change);CHKERRQ(ierr);
  ierr = PCSetType(pc_change,PCLU);CHKERRQ(ierr);
  ierr = KSPSetUp(inv_change);CHKERRQ(ierr);
  new_nsp_size = nsp_size;
  if (nsp_has_cnst) { new_nsp_size++; }
  ierr = PetscMalloc(new_nsp_size*sizeof(Vec),&new_nsp_vecs);CHKERRQ(ierr);
  for (i=0;i<new_nsp_size;i++) { ierr = VecDuplicate(pcis->vec1_global,&new_nsp_vecs[i]);CHKERRQ(ierr); }
  start_new = 0;
  if (nsp_has_cnst) {
    start_new = 1;
    ierr = VecSet(new_nsp_vecs[0],1.0);CHKERRQ(ierr);
    ierr = VecSet(pcis->vec1_B,1.0);CHKERRQ(ierr);
    ierr = KSPSolve(inv_change,pcis->vec1_B,pcis->vec1_B);
    ierr = VecScatterBegin(pcis->global_to_B,pcis->vec1_B,new_nsp_vecs[0],INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
    ierr = VecScatterEnd  (pcis->global_to_B,pcis->vec1_B,new_nsp_vecs[0],INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  }
  for (i=0;i<nsp_size;i++) {
    ierr = VecCopy(nsp_vecs[i],new_nsp_vecs[i+start_new]);CHKERRQ(ierr);
    ierr = VecScatterBegin(pcis->global_to_B,nsp_vecs[i],pcis->vec1_B,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterEnd  (pcis->global_to_B,nsp_vecs[i],pcis->vec1_B,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = KSPSolve(inv_change,pcis->vec1_B,pcis->vec1_B);
    ierr = VecScatterBegin(pcis->global_to_B,pcis->vec1_B,new_nsp_vecs[i+start_new],INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
    ierr = VecScatterEnd  (pcis->global_to_B,pcis->vec1_B,new_nsp_vecs[i+start_new],INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  }
  ierr = VecNormalize(new_nsp_vecs[0],PETSC_NULL);CHKERRQ(ierr);
  /* TODO : Orthonormalize vecs when new_nsp_size > 0! */
  
  /*PetscBool nsp_t=PETSC_FALSE;
  ierr = MatNullSpaceTest(pcbddc->NullSpace,pc->pmat,&nsp_t);CHKERRQ(ierr);
  printf("Original Null Space test: %d\n",nsp_t);
  Mat temp_mat;
  Mat_IS* matis = (Mat_IS*)pc->pmat->data;
    temp_mat = matis->A;
    matis->A = pcbddc->local_mat;
    pcbddc->local_mat = temp_mat;
  ierr = MatNullSpaceTest(pcbddc->NullSpace,pc->pmat,&nsp_t);CHKERRQ(ierr);
  printf("Original Null Space, mat changed test: %d\n",nsp_t);
  {
    PetscReal test_norm;
    for (i=0;i<new_nsp_size;i++) {
      ierr = MatMult(pc->pmat,new_nsp_vecs[i],pcis->vec1_global);CHKERRQ(ierr);
      ierr = VecNorm(pcis->vec1_global,NORM_2,&test_norm);CHKERRQ(ierr);
      if (test_norm > 1.e-12) {
        printf("------------ERROR VEC %d------------------\n",i);
        ierr = VecView(pcis->vec1_global,PETSC_VIEWER_STDOUT_WORLD);
        printf("------------------------------------------\n");
      }
    }
  }*/

  ierr = KSPDestroy(&inv_change);CHKERRQ(ierr);
  ierr = MatNullSpaceCreate(((PetscObject)pc)->comm,PETSC_FALSE,new_nsp_size,new_nsp_vecs,&new_nsp);CHKERRQ(ierr);
  ierr = PCBDDCSetNullSpace(pc,new_nsp);CHKERRQ(ierr);
  ierr = MatNullSpaceDestroy(&new_nsp);CHKERRQ(ierr);
  /*
  ierr = MatNullSpaceTest(pcbddc->NullSpace,pc->pmat,&nsp_t);CHKERRQ(ierr);
  printf("New Null Space, mat changed: %d\n",nsp_t);
    temp_mat = matis->A;
    matis->A = pcbddc->local_mat;
    pcbddc->local_mat = temp_mat;
  ierr = MatNullSpaceTest(pcbddc->NullSpace,pc->pmat,&nsp_t);CHKERRQ(ierr);
  printf("New Null Space, mat original: %d\n",nsp_t);*/

  for (i=0;i<new_nsp_size;i++) { ierr = VecDestroy(&new_nsp_vecs[i]);CHKERRQ(ierr); }
  ierr = PetscFree(new_nsp_vecs);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCBDDCCreateFETIDPMatContext"
static PetscErrorCode PCBDDCCreateFETIDPMatContext(PC pc, FETIDPMat_ctx **fetidpmat_ctx)
{
  FETIDPMat_ctx  *newctx;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscMalloc(sizeof(*newctx),&newctx);CHKERRQ(ierr);
  newctx->lambda_local    = 0;
  newctx->temp_solution_B = 0;
  newctx->temp_solution_D = 0;
  newctx->B_delta         = 0;
  newctx->B_Ddelta        = 0; /* theoretically belongs to the FETIDP preconditioner */
  newctx->l2g_lambda      = 0;
  /* increase the reference count for BDDC preconditioner */
  ierr = PetscObjectReference((PetscObject)pc);CHKERRQ(ierr);
  newctx->pc              = pc;
  *fetidpmat_ctx          = newctx;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCBDDCCreateFETIDPPCContext"
static PetscErrorCode PCBDDCCreateFETIDPPCContext(PC pc, FETIDPPC_ctx **fetidppc_ctx)
{
  FETIDPPC_ctx  *newctx;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscMalloc(sizeof(*newctx),&newctx);CHKERRQ(ierr);
  newctx->lambda_local    = 0;
  newctx->B_Ddelta        = 0;
  newctx->l2g_lambda      = 0;
  /* increase the reference count for BDDC preconditioner */
  ierr = PetscObjectReference((PetscObject)pc);CHKERRQ(ierr);
  newctx->pc              = pc;
  *fetidppc_ctx           = newctx;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCBDDCDestroyFETIDPMat"
static PetscErrorCode PCBDDCDestroyFETIDPMat(Mat A)
{
  FETIDPMat_ctx  *mat_ctx;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatShellGetContext(A,(void**)&mat_ctx);CHKERRQ(ierr);
  ierr = VecDestroy(&mat_ctx->lambda_local);CHKERRQ(ierr);
  ierr = VecDestroy(&mat_ctx->temp_solution_D);CHKERRQ(ierr);
  ierr = VecDestroy(&mat_ctx->temp_solution_B);CHKERRQ(ierr);
  ierr = MatDestroy(&mat_ctx->B_delta);CHKERRQ(ierr);
  ierr = MatDestroy(&mat_ctx->B_Ddelta);CHKERRQ(ierr);
  ierr = VecScatterDestroy(&mat_ctx->l2g_lambda);CHKERRQ(ierr);
  ierr = PCDestroy(&mat_ctx->pc);CHKERRQ(ierr); /* actually it does not destroy BDDC, only decrease its reference count */
  ierr = PetscFree(mat_ctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCBDDCDestroyFETIDPPC"
static PetscErrorCode PCBDDCDestroyFETIDPPC(PC pc)
{
  FETIDPPC_ctx  *pc_ctx;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PCShellGetContext(pc,(void**)&pc_ctx);CHKERRQ(ierr);
  ierr = VecDestroy(&pc_ctx->lambda_local);CHKERRQ(ierr);
  ierr = MatDestroy(&pc_ctx->B_Ddelta);CHKERRQ(ierr);
  ierr = VecScatterDestroy(&pc_ctx->l2g_lambda);CHKERRQ(ierr);
  ierr = PCDestroy(&pc_ctx->pc);CHKERRQ(ierr); /* actually it does not destroy BDDC, only decrease its reference count */
  ierr = PetscFree(pc_ctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCBDDCSetupFETIDPMatContext"
static PetscErrorCode PCBDDCSetupFETIDPMatContext(FETIDPMat_ctx *fetidpmat_ctx )
{
  PetscErrorCode ierr;
  PC_IS          *pcis=(PC_IS*)fetidpmat_ctx->pc->data;
  PC_BDDC        *pcbddc=(PC_BDDC*)fetidpmat_ctx->pc->data;
  PCBDDCGraph    mat_graph=pcbddc->mat_graph;
  Mat_IS         *matis  = (Mat_IS*)fetidpmat_ctx->pc->pmat->data;
  MPI_Comm       comm = ((PetscObject)(fetidpmat_ctx->pc))->comm;

  Mat            ScalingMat;
  Vec            lambda_global;
  IS             IS_l2g_lambda;

  PetscBool      skip_node,fully_redundant;
  PetscInt       i,j,k,s,n_boundary_dofs,n_global_lambda,n_vertices,partial_sum;
  PetscInt       n_local_lambda,n_lambda_for_dof,dual_size,n_neg_values,n_pos_values;
  PetscMPIInt    rank,nprocs;
  PetscScalar    scalar_value;

  PetscInt       *vertex_indices,*temp_indices;
  PetscInt       *dual_dofs_boundary_indices,*aux_local_numbering_1,*aux_global_numbering;
  PetscInt       *aux_sums,*cols_B_delta,*l2g_indices;
  PetscScalar    *array,*scaling_factors,*vals_B_delta;
  PetscInt       *aux_local_numbering_2,*dof_sizes,*dof_displs;
  PetscInt       first_index,old_index;
  PetscBool      first_found = PETSC_FALSE;

  /* For communication of scaling factors */
  PetscInt       *ptrs_buffer,neigh_position;
  PetscScalar    **all_factors,*send_buffer,*recv_buffer;
  MPI_Request    *send_reqs,*recv_reqs;

  /* tests */
  Vec            test_vec;
  PetscBool      test_fetidp;
  PetscViewer    viewer;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm,&nprocs);CHKERRQ(ierr);

  /* Default type of lagrange multipliers is non-redundant */
  fully_redundant = PETSC_FALSE;
  ierr = PetscOptionsGetBool(PETSC_NULL,"-fetidp_fullyredundant",&fully_redundant,PETSC_NULL);CHKERRQ(ierr);

  /* Evaluate local and global number of lagrange multipliers */
  ierr = VecSet(pcis->vec1_N,0.0);CHKERRQ(ierr);
  n_local_lambda = 0;
  partial_sum = 0;
  n_boundary_dofs = 0;
  s = 0;
  n_vertices = 0;
  /* Get Vertices used to define the BDDC */
  ierr = PetscMalloc(pcbddc->local_primal_size*sizeof(*vertex_indices),&vertex_indices);CHKERRQ(ierr);
  for (i=0;i<pcbddc->local_primal_size;i++) {
    ierr = MatGetRow(pcbddc->ConstraintMatrix,i,&j,(const PetscInt**)&temp_indices,PETSC_NULL);CHKERRQ(ierr);
    if (j == 1) {
      vertex_indices[n_vertices]=temp_indices[0];
      n_vertices++;
    }
    ierr = MatRestoreRow(pcbddc->ConstraintMatrix,i,&j,(const PetscInt**)&temp_indices,PETSC_NULL);CHKERRQ(ierr);
  }
  dual_size = pcis->n_B-n_vertices;
  ierr = PetscSortInt(n_vertices,vertex_indices);CHKERRQ(ierr);
  ierr = PetscMalloc(dual_size*sizeof(*dual_dofs_boundary_indices),&dual_dofs_boundary_indices);CHKERRQ(ierr);
  ierr = PetscMalloc(dual_size*sizeof(*aux_local_numbering_1),&aux_local_numbering_1);CHKERRQ(ierr);
  ierr = PetscMalloc(dual_size*sizeof(*aux_local_numbering_2),&aux_local_numbering_2);CHKERRQ(ierr);

  ierr = VecGetArray(pcis->vec1_N,&array);CHKERRQ(ierr);
  for (i=0;i<pcis->n;i++){
    j = mat_graph->count[i]; /* RECALL: mat_graph->count[i] does not count myself */
    k = 0;
    if (j > 0) { 
      k = (mat_graph->neighbours_set[i][0] == -1 ?  1 : 0);
    }
    j = j - k ;
    if ( j > 0 ) { n_boundary_dofs++; }

    skip_node = PETSC_FALSE;
    if ( s < n_vertices && vertex_indices[s]==i) { /* it works for a sorted set of vertices */
      skip_node = PETSC_TRUE;
      s++;
    }
    if (j < 1) {skip_node = PETSC_TRUE;}
    if ( !skip_node ) {
      if (fully_redundant) {
        /* fully redundant set of lagrange multipliers */
        n_lambda_for_dof = (j*(j+1))/2;
      } else {
        n_lambda_for_dof = j;
      }
      n_local_lambda += j;
      /* needed to evaluate global number of lagrange multipliers */
      array[i]=(1.0*n_lambda_for_dof)/(j+1.0); /* already scaled for the next global sum */
      /* store some data needed */
      dual_dofs_boundary_indices[partial_sum] = n_boundary_dofs-1;
      aux_local_numbering_1[partial_sum] = i;
      aux_local_numbering_2[partial_sum] = n_lambda_for_dof;
      partial_sum++;
    }
  }
  /*printf("I found %d local lambda dofs\n",n_local_lambda);
  printf("I found %d boundary dofs (should be %d)\n",n_boundary_dofs,pcis->n_B);
  printf("Partial sum %d should be %d\n",partial_sum,dual_size);*/
  ierr = VecRestoreArray(pcis->vec1_N,&array);CHKERRQ(ierr);

  ierr = VecSet(pcis->vec1_global,0.0);CHKERRQ(ierr);
  ierr = VecScatterBegin(matis->ctx,pcis->vec1_N,pcis->vec1_global,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  ierr = VecScatterEnd  (matis->ctx,pcis->vec1_N,pcis->vec1_global,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  ierr = VecSum(pcis->vec1_global,&scalar_value);CHKERRQ(ierr);
  fetidpmat_ctx->n_lambda = (PetscInt) scalar_value; 
  /* printf("I found %d global multipliers (%f)\n",fetidpmat_ctx->n_lambda,scalar_value); */

  /* compute global ordering of lagrange multipliers and associate l2g map */
  ierr = VecSet(pcis->vec1_global,0.0);CHKERRQ(ierr);
  ierr = VecSet(pcis->vec1_N,0.0);CHKERRQ(ierr);
  ierr = VecGetArray(pcis->vec1_N,&array);CHKERRQ(ierr);
  for (i=0;i<dual_size;i++) {
    array[aux_local_numbering_1[i]] = aux_local_numbering_2[i];
  }
  ierr = VecRestoreArray(pcis->vec1_N,&array);CHKERRQ(ierr);
  ierr = VecScatterBegin(matis->ctx,pcis->vec1_N,pcis->vec1_global,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  ierr = VecScatterEnd  (matis->ctx,pcis->vec1_N,pcis->vec1_global,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  ierr = VecSum(pcis->vec1_global,&scalar_value);CHKERRQ(ierr);
  if (pcbddc->dbg_flag && (PetscInt)scalar_value != fetidpmat_ctx->n_lambda) {
    SETERRQ3(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"Error in %s: global number of multipliers mismatch! (%d!=%d)\n",__FUNCT__,(PetscInt)scalar_value,fetidpmat_ctx->n_lambda);
  }

  /* Fill pcis->vec1_global with cumulative function for global numbering */
  ierr = VecGetArray(pcis->vec1_global,&array);CHKERRQ(ierr);
  ierr = VecGetLocalSize(pcis->vec1_global,&s);CHKERRQ(ierr);
  k = 0;
  first_index = -1;
  for (i=0;i<s;i++) {
    if (!first_found && array[i] > 0.0) {
      first_found = PETSC_TRUE;
      first_index = i;
    }
    k += (PetscInt)array[i];
  }
  j = ( !rank ? nprocs : 0);
  ierr = PetscMalloc(j*sizeof(*dof_sizes),&dof_sizes);CHKERRQ(ierr);
  ierr = PetscMalloc(j*sizeof(*dof_displs),&dof_displs);CHKERRQ(ierr);
  ierr = MPI_Gather(&k,1,MPIU_INT,dof_sizes,1,MPIU_INT,0,comm);CHKERRQ(ierr);
  if (!rank) {
    dof_displs[0]=0;
    for (i=1;i<nprocs;i++) {
      dof_displs[i] = dof_displs[i-1]+dof_sizes[i-1];
    }
  }
  ierr = MPI_Scatter(dof_displs,1,MPIU_INT,&k,1,MPIU_INT,0,comm);CHKERRQ(ierr);
  if (first_found) {
    array[first_index] += k;
    old_index = first_index;
    for (i=first_index+1;i<s;i++) {
      if (array[i] > 0.0) {
        array[i] += array[old_index];
        old_index = i;
      }
    }
  }
  ierr = VecRestoreArray(pcis->vec1_global,&array);CHKERRQ(ierr);
  ierr = VecSet(pcis->vec1_N,0.0);CHKERRQ(ierr);
  ierr = VecScatterBegin(matis->ctx,pcis->vec1_global,pcis->vec1_N,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd  (matis->ctx,pcis->vec1_global,pcis->vec1_N,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = PetscMalloc(dual_size*sizeof(*aux_global_numbering),&aux_global_numbering);CHKERRQ(ierr);
  ierr = VecGetArray(pcis->vec1_N,&array);CHKERRQ(ierr);
  for (i=0;i<dual_size;i++) {
    aux_global_numbering[i] = (PetscInt)array[aux_local_numbering_1[i]]-aux_local_numbering_2[i];
  }
  ierr = VecRestoreArray(pcis->vec1_N,&array);CHKERRQ(ierr);
  ierr = PetscFree(aux_local_numbering_2);CHKERRQ(ierr);
  ierr = PetscFree(dof_displs);CHKERRQ(ierr);
  ierr = PetscFree(dof_sizes);CHKERRQ(ierr);

  /* init data for scaling factors exchange */
  partial_sum = 0;
  j = 0;
  ierr = PetscMalloc(pcis->n_neigh*sizeof(PetscInt),&ptrs_buffer);CHKERRQ(ierr);
  ierr = PetscMalloc((pcis->n_neigh-1)*sizeof(MPI_Request),&send_reqs);CHKERRQ(ierr);
  ierr = PetscMalloc((pcis->n_neigh-1)*sizeof(MPI_Request),&recv_reqs);CHKERRQ(ierr);
  ierr = PetscMalloc(pcis->n*sizeof(PetscScalar*),&all_factors);CHKERRQ(ierr);
  ptrs_buffer[0]=0;
  for (i=1;i<pcis->n_neigh;i++) {
    partial_sum += pcis->n_shared[i];
    ptrs_buffer[i] = ptrs_buffer[i-1]+pcis->n_shared[i];
  }
  ierr = PetscMalloc( partial_sum*sizeof(PetscScalar),&send_buffer);CHKERRQ(ierr);
  ierr = PetscMalloc( partial_sum*sizeof(PetscScalar),&recv_buffer);CHKERRQ(ierr);
  ierr = PetscMalloc( partial_sum*sizeof(PetscScalar),&all_factors[0]);CHKERRQ(ierr);
  for (i=0;i<pcis->n-1;i++) {
    j = mat_graph->count[i];
    if (j>0) {
      k = (mat_graph->neighbours_set[i][0] == -1 ?  1 : 0);
      j = j - k;
    }
    all_factors[i+1]=all_factors[i]+j;
  }
  /* scatter B scaling to N vec */
  ierr = VecScatterBegin(pcis->N_to_B,pcis->D,pcis->vec1_N,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  ierr = VecScatterEnd  (pcis->N_to_B,pcis->D,pcis->vec1_N,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  /* communications */
  ierr = VecGetArray(pcis->vec1_N,&array);CHKERRQ(ierr);
  for (i=1;i<pcis->n_neigh;i++) {
    for (j=0;j<pcis->n_shared[i];j++) {
      send_buffer[ptrs_buffer[i-1]+j]=array[pcis->shared[i][j]];
    }
    j = ptrs_buffer[i]-ptrs_buffer[i-1];
    ierr = MPI_Isend(&send_buffer[ptrs_buffer[i-1]],j,MPIU_SCALAR,pcis->neigh[i],0,comm,&send_reqs[i-1]);CHKERRQ(ierr);
    ierr = MPI_Irecv(&recv_buffer[ptrs_buffer[i-1]],j,MPIU_SCALAR,pcis->neigh[i],0,comm,&recv_reqs[i-1]);CHKERRQ(ierr);
  }
  ierr = VecRestoreArray(pcis->vec1_N,&array);CHKERRQ(ierr);
  ierr = MPI_Waitall((pcis->n_neigh-1),recv_reqs,MPI_STATUSES_IGNORE);CHKERRQ(ierr);
  /* put values in correct places */
  for (i=1;i<pcis->n_neigh;i++) {
    for (j=0;j<pcis->n_shared[i];j++) {
      k = pcis->shared[i][j];
      neigh_position = 0;
      while(mat_graph->neighbours_set[k][neigh_position] != pcis->neigh[i]) {neigh_position++;}
      s = (mat_graph->neighbours_set[k][0] == -1 ?  1 : 0);
      neigh_position = neigh_position - s;
      all_factors[k][neigh_position]=recv_buffer[ptrs_buffer[i-1]+j];
    }
  }
  ierr = MPI_Waitall((pcis->n_neigh-1),send_reqs,MPI_STATUSES_IGNORE);CHKERRQ(ierr);
  ierr = PetscFree(send_reqs);CHKERRQ(ierr);
  ierr = PetscFree(recv_reqs);CHKERRQ(ierr);
  ierr = PetscFree(send_buffer);CHKERRQ(ierr);
  ierr = PetscFree(recv_buffer);CHKERRQ(ierr);
  ierr = PetscFree(ptrs_buffer);CHKERRQ(ierr);

  /* Compute B and B_delta (local actions) */
  ierr = PetscMalloc(pcis->n_neigh*sizeof(*aux_sums),&aux_sums);CHKERRQ(ierr);
  ierr = PetscMalloc(n_local_lambda*sizeof(*l2g_indices),&l2g_indices);CHKERRQ(ierr);
  ierr = PetscMalloc(n_local_lambda*sizeof(*vals_B_delta),&vals_B_delta);CHKERRQ(ierr);
  ierr = PetscMalloc(n_local_lambda*sizeof(*cols_B_delta),&cols_B_delta);CHKERRQ(ierr);
  ierr = PetscMalloc(n_local_lambda*sizeof(*scaling_factors),&scaling_factors);CHKERRQ(ierr);
  n_global_lambda=0;
  partial_sum=0;
  for (i=0;i<dual_size;i++) {
    n_global_lambda = aux_global_numbering[i];
    j = mat_graph->count[aux_local_numbering_1[i]];
    k = (mat_graph->neighbours_set[aux_local_numbering_1[i]][0] == -1 ?  1 : 0);
    j = j - k;
    aux_sums[0]=0;
    for (s=1;s<j;s++) {
      aux_sums[s]=aux_sums[s-1]+j-s+1;
    }
    array = all_factors[aux_local_numbering_1[i]];
    n_neg_values = 0;
    while(n_neg_values < j && mat_graph->neighbours_set[aux_local_numbering_1[i]][n_neg_values+k] < rank) {n_neg_values++;}
    n_pos_values = j - n_neg_values;
    if (fully_redundant) {
      for (s=0;s<n_neg_values;s++) {
        l2g_indices    [partial_sum+s]=aux_sums[s]+n_neg_values-s-1+n_global_lambda;
        cols_B_delta   [partial_sum+s]=dual_dofs_boundary_indices[i];
        vals_B_delta   [partial_sum+s]=-1.0;
        scaling_factors[partial_sum+s]=array[s];
      }
      for (s=0;s<n_pos_values;s++) {
        l2g_indices    [partial_sum+s+n_neg_values]=aux_sums[n_neg_values]+s+n_global_lambda;
        cols_B_delta   [partial_sum+s+n_neg_values]=dual_dofs_boundary_indices[i];
        vals_B_delta   [partial_sum+s+n_neg_values]=1.0;
        scaling_factors[partial_sum+s+n_neg_values]=array[s+n_neg_values];
      }
      partial_sum += j;
    } else {
      /* l2g_indices and default cols and vals of B_delta */
      for (s=0;s<j;s++) {
        l2g_indices    [partial_sum+s]=n_global_lambda+s;
        cols_B_delta   [partial_sum+s]=dual_dofs_boundary_indices[i];
        vals_B_delta   [partial_sum+s]=0.0;
      }
      /* B_delta */
      if ( n_neg_values > 0 ) { /* there's a rank next to me to the left */
        vals_B_delta   [partial_sum+n_neg_values-1]=-1.0;
      }
      if ( n_neg_values < j ) { /* there's a rank next to me to the right */
        vals_B_delta   [partial_sum+n_neg_values]=1.0;
      }
      /* scaling as in Klawonn-Widlund 1999*/
      for (s=0;s<n_neg_values;s++) {
        scalar_value = 0.0;
        for (k=0;k<s+1;k++) {
          scalar_value += array[k];
        }
        scaling_factors[partial_sum+s] = -scalar_value;
      }
      for (s=0;s<n_pos_values;s++) {
        scalar_value = 0.0;
        for (k=s+n_neg_values;k<j;k++) {
          scalar_value += array[k];
        }
        scaling_factors[partial_sum+s+n_neg_values] = scalar_value;
      }
      partial_sum += j;
    }
  }
  ierr = PetscFree(aux_global_numbering);CHKERRQ(ierr);
  ierr = PetscFree(aux_sums);CHKERRQ(ierr);
  ierr = PetscFree(aux_local_numbering_1);CHKERRQ(ierr);
  ierr = PetscFree(dual_dofs_boundary_indices);CHKERRQ(ierr);
  ierr = PetscFree(all_factors[0]);CHKERRQ(ierr);
  ierr = PetscFree(all_factors);CHKERRQ(ierr);
  /* printf("I found %d local lambda dofs when numbering them (should be %d)\n",partial_sum,n_local_lambda); */

  /* Local to global mapping of fetidpmat */
  ierr = VecCreate(PETSC_COMM_SELF,&fetidpmat_ctx->lambda_local);CHKERRQ(ierr);
  ierr = VecSetSizes(fetidpmat_ctx->lambda_local,n_local_lambda,n_local_lambda);CHKERRQ(ierr);
  ierr = VecSetType(fetidpmat_ctx->lambda_local,VECSEQ);CHKERRQ(ierr);
  ierr = VecCreate(comm,&lambda_global);CHKERRQ(ierr);
  ierr = VecSetSizes(lambda_global,PETSC_DECIDE,fetidpmat_ctx->n_lambda);CHKERRQ(ierr);
  ierr = VecSetType(lambda_global,VECMPI);CHKERRQ(ierr);
  ierr = ISCreateGeneral(comm,n_local_lambda,l2g_indices,PETSC_OWN_POINTER,&IS_l2g_lambda);CHKERRQ(ierr);
  ierr = VecScatterCreate(fetidpmat_ctx->lambda_local,(IS)0,lambda_global,IS_l2g_lambda,&fetidpmat_ctx->l2g_lambda);CHKERRQ(ierr);
  ierr = ISDestroy(&IS_l2g_lambda);CHKERRQ(ierr);

  /* Create local part of B_delta */
  ierr = MatCreate(PETSC_COMM_SELF,&fetidpmat_ctx->B_delta);
  ierr = MatSetSizes(fetidpmat_ctx->B_delta,n_local_lambda,pcis->n_B,n_local_lambda,pcis->n_B);CHKERRQ(ierr);
  ierr = MatSetType(fetidpmat_ctx->B_delta,MATSEQAIJ);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(fetidpmat_ctx->B_delta,1,PETSC_NULL);CHKERRQ(ierr);
  ierr = MatSetOption(fetidpmat_ctx->B_delta,MAT_IGNORE_ZERO_ENTRIES,PETSC_TRUE);CHKERRQ(ierr);
  for (i=0;i<n_local_lambda;i++) {
    ierr = MatSetValue(fetidpmat_ctx->B_delta,i,cols_B_delta[i],vals_B_delta[i],INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = PetscFree(vals_B_delta);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(fetidpmat_ctx->B_delta,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd  (fetidpmat_ctx->B_delta,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  if (fully_redundant) {
    ierr = MatCreate(PETSC_COMM_SELF,&ScalingMat);
    ierr = MatSetSizes(ScalingMat,n_local_lambda,n_local_lambda,n_local_lambda,n_local_lambda);CHKERRQ(ierr);
    ierr = MatSetType(ScalingMat,MATSEQAIJ);CHKERRQ(ierr);
    ierr = MatSeqAIJSetPreallocation(ScalingMat,1,PETSC_NULL);CHKERRQ(ierr);
    for (i=0;i<n_local_lambda;i++) {
      ierr = MatSetValue(ScalingMat,i,i,scaling_factors[i],INSERT_VALUES);CHKERRQ(ierr);
    }
    ierr = MatAssemblyBegin(ScalingMat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd  (ScalingMat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatMatMult(ScalingMat,fetidpmat_ctx->B_delta,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&fetidpmat_ctx->B_Ddelta);CHKERRQ(ierr);
    ierr = MatDestroy(&ScalingMat);CHKERRQ(ierr);
  } else {
    ierr = MatCreate(PETSC_COMM_SELF,&fetidpmat_ctx->B_Ddelta);
    ierr = MatSetSizes(fetidpmat_ctx->B_Ddelta,n_local_lambda,pcis->n_B,n_local_lambda,pcis->n_B);CHKERRQ(ierr);
    ierr = MatSetType(fetidpmat_ctx->B_Ddelta,MATSEQAIJ);CHKERRQ(ierr);
    ierr = MatSeqAIJSetPreallocation(fetidpmat_ctx->B_Ddelta,1,PETSC_NULL);CHKERRQ(ierr);
    for (i=0;i<n_local_lambda;i++) {
      ierr = MatSetValue(fetidpmat_ctx->B_Ddelta,i,cols_B_delta[i],scaling_factors[i],INSERT_VALUES);CHKERRQ(ierr);
    }
    ierr = MatAssemblyBegin(fetidpmat_ctx->B_Ddelta,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd  (fetidpmat_ctx->B_Ddelta,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }
  ierr = PetscFree(scaling_factors);CHKERRQ(ierr);
  ierr = PetscFree(cols_B_delta);CHKERRQ(ierr);

  /* Create some vectors needed by fetidp */
  ierr = VecDuplicate(pcis->vec1_B,&fetidpmat_ctx->temp_solution_B);CHKERRQ(ierr);
  ierr = VecDuplicate(pcis->vec1_D,&fetidpmat_ctx->temp_solution_D);CHKERRQ(ierr);

  test_fetidp = PETSC_FALSE;
  ierr = PetscOptionsGetBool(PETSC_NULL,"-fetidp_check",&test_fetidp,PETSC_NULL);CHKERRQ(ierr);

  if (test_fetidp) {
    
    ierr = PetscViewerASCIIGetStdout(((PetscObject)(fetidpmat_ctx->pc))->comm,&viewer);CHKERRQ(ierr);
    ierr = PetscViewerASCIISynchronizedAllow(viewer,PETSC_TRUE);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"----------FETI_DP TESTS--------------\n");CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"All tests should return zero!\n");CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"FETIDP MAT context in the ");CHKERRQ(ierr);
    if (fully_redundant) {
      ierr = PetscViewerASCIIPrintf(viewer,"fully redundant case for lagrange multipliers.\n");CHKERRQ(ierr);
    } else {
      ierr = PetscViewerASCIIPrintf(viewer,"Non-fully redundant case for lagrange multiplier.\n");CHKERRQ(ierr);
    }
    ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);

    /******************************************************************/
    /* TEST A/B: Test numbering of global lambda dofs             */
    /******************************************************************/

    ierr = VecDuplicate(fetidpmat_ctx->lambda_local,&test_vec);CHKERRQ(ierr);
    ierr = VecSet(lambda_global,1.0);CHKERRQ(ierr);
    ierr = VecSet(test_vec,1.0);CHKERRQ(ierr);
    ierr = VecScatterBegin(fetidpmat_ctx->l2g_lambda,lambda_global,fetidpmat_ctx->lambda_local,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
    ierr = VecScatterEnd  (fetidpmat_ctx->l2g_lambda,lambda_global,fetidpmat_ctx->lambda_local,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
    scalar_value = -1.0;
    ierr = VecAXPY(test_vec,scalar_value,fetidpmat_ctx->lambda_local);CHKERRQ(ierr);
    ierr = VecNorm(test_vec,NORM_INFINITY,&scalar_value);CHKERRQ(ierr);
    ierr = VecDestroy(&test_vec);CHKERRQ(ierr);
    ierr = PetscViewerASCIISynchronizedPrintf(viewer,"A[%04d]: CHECK glob to loc: % 1.14e\n",rank,scalar_value);CHKERRQ(ierr);
    ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
    if (fully_redundant) {
      ierr = VecSet(lambda_global,0.0);CHKERRQ(ierr);
      ierr = VecSet(fetidpmat_ctx->lambda_local,0.5);CHKERRQ(ierr);
      ierr = VecScatterBegin(fetidpmat_ctx->l2g_lambda,fetidpmat_ctx->lambda_local,lambda_global,ADD_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
      ierr = VecScatterEnd  (fetidpmat_ctx->l2g_lambda,fetidpmat_ctx->lambda_local,lambda_global,ADD_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
      ierr = VecSum(lambda_global,&scalar_value);CHKERRQ(ierr);
      ierr = PetscViewerASCIISynchronizedPrintf(viewer,"B[%04d]: CHECK loc to glob: % 1.14e\n",rank,scalar_value-fetidpmat_ctx->n_lambda);CHKERRQ(ierr);
      ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
    }

    /******************************************************************/
    /* TEST C: It should holds B_delta*w=0, w\in\widehat{W}           */
    /* This is the meaning of the B matrix                            */
    /******************************************************************/

    ierr = VecSetRandom(pcis->vec1_N,PETSC_NULL);CHKERRQ(ierr);
    ierr = VecSet(pcis->vec1_global,0.0);CHKERRQ(ierr);
    ierr = VecScatterBegin(matis->ctx,pcis->vec1_N,pcis->vec1_global,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
    ierr = VecScatterEnd  (matis->ctx,pcis->vec1_N,pcis->vec1_global,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
    ierr = VecScatterBegin(matis->ctx,pcis->vec1_global,pcis->vec1_N,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterEnd  (matis->ctx,pcis->vec1_global,pcis->vec1_N,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterBegin(pcis->N_to_B,pcis->vec1_N,pcis->vec1_B,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterEnd  (pcis->N_to_B,pcis->vec1_N,pcis->vec1_B,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    /* Action of B_delta */
    ierr = MatMult(fetidpmat_ctx->B_delta,pcis->vec1_B,fetidpmat_ctx->lambda_local);CHKERRQ(ierr);
    ierr = VecSet(lambda_global,0.0);CHKERRQ(ierr);
    ierr = VecScatterBegin(fetidpmat_ctx->l2g_lambda,fetidpmat_ctx->lambda_local,lambda_global,ADD_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterEnd  (fetidpmat_ctx->l2g_lambda,fetidpmat_ctx->lambda_local,lambda_global,ADD_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecNorm(lambda_global,NORM_INFINITY,&scalar_value);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"C[coll]: CHECK infty norm of B_delta*w (w continuous): % 1.14e\n",scalar_value);CHKERRQ(ierr);
    ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);

    /******************************************************************/
    /* TEST D: It should holds E_Dw = w - P_Dw w\in\widetilde{W}     */
    /* E_D = R_D^TR                                                   */
    /* P_D = B_{D,delta}^T B_{delta}                                  */
    /* eq.44 Mandel Tezaur and Dohrmann 2005                          */
    /******************************************************************/

    /* compute a random vector in \widetilde{W} */
    ierr = VecSetRandom(pcis->vec1_N,PETSC_NULL);CHKERRQ(ierr);
    scalar_value = 0.0;  /* set zero at vertices */
    ierr = VecGetArray(pcis->vec1_N,&array);CHKERRQ(ierr);
    for (i=0;i<n_vertices;i++) { array[vertex_indices[i]]=scalar_value; }
    ierr = VecRestoreArray(pcis->vec1_N,&array);CHKERRQ(ierr);
    /* store w for final comparison */
    ierr = VecDuplicate(pcis->vec1_B,&test_vec);CHKERRQ(ierr);
    ierr = VecScatterBegin(pcis->N_to_B,pcis->vec1_N,test_vec,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterEnd  (pcis->N_to_B,pcis->vec1_N,test_vec,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);

    /* Jump operator P_D : results stored in pcis->vec1_B */

    ierr = VecScatterBegin(pcis->N_to_B,pcis->vec1_N,pcis->vec1_B,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterEnd  (pcis->N_to_B,pcis->vec1_N,pcis->vec1_B,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    /* Action of B_delta */
    ierr = MatMult(fetidpmat_ctx->B_delta,pcis->vec1_B,fetidpmat_ctx->lambda_local);CHKERRQ(ierr);
    ierr = VecSet(lambda_global,0.0);CHKERRQ(ierr);
    ierr = VecScatterBegin(fetidpmat_ctx->l2g_lambda,fetidpmat_ctx->lambda_local,lambda_global,ADD_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterEnd  (fetidpmat_ctx->l2g_lambda,fetidpmat_ctx->lambda_local,lambda_global,ADD_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    /* Action of B_Ddelta^T */
    ierr = VecScatterBegin(fetidpmat_ctx->l2g_lambda,lambda_global,fetidpmat_ctx->lambda_local,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
    ierr = VecScatterEnd  (fetidpmat_ctx->l2g_lambda,lambda_global,fetidpmat_ctx->lambda_local,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
    ierr = MatMultTranspose(fetidpmat_ctx->B_Ddelta,fetidpmat_ctx->lambda_local,pcis->vec1_B);CHKERRQ(ierr);

    /* Average operator E_D : results stored in pcis->vec2_B */

    ierr = VecScatterBegin(pcis->N_to_B,pcis->vec1_N,pcis->vec2_B,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterEnd  (pcis->N_to_B,pcis->vec1_N,pcis->vec2_B,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecPointwiseMult(pcis->vec2_B,pcis->D,pcis->vec2_B);CHKERRQ(ierr);
    ierr = VecScatterBegin(pcis->N_to_B,pcis->vec2_B,pcis->vec1_N,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
    ierr = VecScatterEnd  (pcis->N_to_B,pcis->vec2_B,pcis->vec1_N,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
    ierr = VecSet(pcis->vec1_global,0.0);CHKERRQ(ierr);
    ierr = VecScatterBegin(matis->ctx,pcis->vec1_N,pcis->vec1_global,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
    ierr = VecScatterEnd  (matis->ctx,pcis->vec1_N,pcis->vec1_global,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
    ierr = VecScatterBegin(matis->ctx,pcis->vec1_global,pcis->vec1_N,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterEnd  (matis->ctx,pcis->vec1_global,pcis->vec1_N,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterBegin(pcis->N_to_B,pcis->vec1_N,pcis->vec2_B,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterEnd  (pcis->N_to_B,pcis->vec1_N,pcis->vec2_B,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);

    /* test E_D=I-P_D */
    scalar_value = 1.0;
    ierr = VecAXPY(pcis->vec1_B,scalar_value,pcis->vec2_B);CHKERRQ(ierr);
    scalar_value = -1.0;
    ierr = VecAXPY(pcis->vec1_B,scalar_value,test_vec);CHKERRQ(ierr);
    ierr = VecNorm(pcis->vec1_B,NORM_INFINITY,&scalar_value);CHKERRQ(ierr);
    ierr = VecDestroy(&test_vec);CHKERRQ(ierr);
    ierr = PetscViewerASCIISynchronizedPrintf(viewer,"D[%04d] CHECK infty norm of E_D + P_D - I: % 1.14e\n",rank,scalar_value);CHKERRQ(ierr);
    ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);

    /******************************************************************/
    /* TEST E: It should holds R_D^TP_Dw=0 w\in\widetilde{W}          */
    /* eq.48 Mandel Tezaur and Dohrmann 2005                          */
    /******************************************************************/

    ierr = VecSetRandom(pcis->vec1_N,PETSC_NULL);CHKERRQ(ierr);
    ierr = VecGetArray(pcis->vec1_N,&array);CHKERRQ(ierr);
    scalar_value = 0.0;  /* set zero at vertices */
    for (i=0;i<n_vertices;i++) { array[vertex_indices[i]]=scalar_value; }
    ierr = VecRestoreArray(pcis->vec1_N,&array);CHKERRQ(ierr);

    /* Jump operator P_D : results stored in pcis->vec1_B */

    ierr = VecScatterBegin(pcis->N_to_B,pcis->vec1_N,pcis->vec1_B,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterEnd  (pcis->N_to_B,pcis->vec1_N,pcis->vec1_B,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    /* Action of B_delta */
    ierr = MatMult(fetidpmat_ctx->B_delta,pcis->vec1_B,fetidpmat_ctx->lambda_local);CHKERRQ(ierr);
    ierr = VecSet(lambda_global,0.0);CHKERRQ(ierr);
    ierr = VecScatterBegin(fetidpmat_ctx->l2g_lambda,fetidpmat_ctx->lambda_local,lambda_global,ADD_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterEnd  (fetidpmat_ctx->l2g_lambda,fetidpmat_ctx->lambda_local,lambda_global,ADD_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    /* Action of B_Ddelta^T */
    ierr = VecScatterBegin(fetidpmat_ctx->l2g_lambda,lambda_global,fetidpmat_ctx->lambda_local,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
    ierr = VecScatterEnd  (fetidpmat_ctx->l2g_lambda,lambda_global,fetidpmat_ctx->lambda_local,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
    ierr = MatMultTranspose(fetidpmat_ctx->B_Ddelta,fetidpmat_ctx->lambda_local,pcis->vec1_B);CHKERRQ(ierr);
    /* diagonal scaling */
    ierr = VecPointwiseMult(pcis->vec1_B,pcis->D,pcis->vec1_B);CHKERRQ(ierr);
    /* sum on the interface */
    ierr = VecSet(pcis->vec1_N,0.0);CHKERRQ(ierr);
    ierr = VecScatterBegin(pcis->N_to_B,pcis->vec1_B,pcis->vec1_N,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
    ierr = VecScatterEnd  (pcis->N_to_B,pcis->vec1_B,pcis->vec1_N,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
    ierr = VecSet(pcis->vec1_global,0.0);CHKERRQ(ierr);
    ierr = VecScatterBegin(matis->ctx,pcis->vec1_N,pcis->vec1_global,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
    ierr = VecScatterEnd  (matis->ctx,pcis->vec1_N,pcis->vec1_global,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
    ierr = VecNorm(pcis->vec1_global,NORM_INFINITY,&scalar_value);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"E[coll]: CHECK infty norm of R^T_D P_D: % 1.14e\n",scalar_value);CHKERRQ(ierr);
    ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);

    if (!fully_redundant) {
      /******************************************************************/
      /* TEST F: It should holds B_{delta}B^T_{D,delta}=I               */
      /* Corollary thm 14 Mandel Tezaur and Dohrmann 2005               */
      /******************************************************************/
      ierr = VecDuplicate(lambda_global,&test_vec);CHKERRQ(ierr);
      ierr = VecSetRandom(lambda_global,PETSC_NULL);CHKERRQ(ierr);
      /* Action of B_Ddelta^T */
      ierr = VecScatterBegin(fetidpmat_ctx->l2g_lambda,lambda_global,fetidpmat_ctx->lambda_local,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
      ierr = VecScatterEnd  (fetidpmat_ctx->l2g_lambda,lambda_global,fetidpmat_ctx->lambda_local,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
      ierr = MatMultTranspose(fetidpmat_ctx->B_Ddelta,fetidpmat_ctx->lambda_local,pcis->vec1_B);CHKERRQ(ierr);
      /* Action of B_delta */
      ierr = MatMult(fetidpmat_ctx->B_delta,pcis->vec1_B,fetidpmat_ctx->lambda_local);CHKERRQ(ierr);
      ierr = VecSet(test_vec,0.0);CHKERRQ(ierr);
      ierr = VecScatterBegin(fetidpmat_ctx->l2g_lambda,fetidpmat_ctx->lambda_local,test_vec,ADD_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
      ierr = VecScatterEnd  (fetidpmat_ctx->l2g_lambda,fetidpmat_ctx->lambda_local,test_vec,ADD_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
      scalar_value = -1.0;
      ierr = VecAXPY(lambda_global,scalar_value,test_vec);CHKERRQ(ierr);
      ierr = VecNorm(lambda_global,NORM_INFINITY,&scalar_value);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"E[coll]: CHECK infty norm of P^T_D - I: % 1.14e\n",scalar_value);CHKERRQ(ierr);
      ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
      ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
      ierr = VecDestroy(&test_vec);CHKERRQ(ierr);
    }
  }
  /* final cleanup */
  ierr = PetscFree(vertex_indices);CHKERRQ(ierr);
  ierr = VecDestroy(&lambda_global);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCBDDCSetupFETIDPPCContext"
static PetscErrorCode PCBDDCSetupFETIDPPCContext(Mat fetimat, FETIDPPC_ctx *fetidppc_ctx)
{
  FETIDPMat_ctx  *mat_ctx;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatShellGetContext(fetimat,&mat_ctx);CHKERRQ(ierr);
  /* get references from objects created when setting up feti mat context */
  ierr = PetscObjectReference((PetscObject)mat_ctx->lambda_local);CHKERRQ(ierr);
  fetidppc_ctx->lambda_local = mat_ctx->lambda_local;
  ierr = PetscObjectReference((PetscObject)mat_ctx->B_Ddelta);CHKERRQ(ierr);
  fetidppc_ctx->B_Ddelta = mat_ctx->B_Ddelta;
  ierr = PetscObjectReference((PetscObject)mat_ctx->l2g_lambda);CHKERRQ(ierr);
  fetidppc_ctx->l2g_lambda = mat_ctx->l2g_lambda;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "FETIDPMatMult"
static PetscErrorCode FETIDPMatMult(Mat fetimat, Vec x, Vec y)
{
  FETIDPMat_ctx  *mat_ctx;
  PC_IS          *pcis;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatShellGetContext(fetimat,&mat_ctx);CHKERRQ(ierr);
  pcis = (PC_IS*)mat_ctx->pc->data;
  /* Application of B_delta^T */
  ierr = VecScatterBegin(mat_ctx->l2g_lambda,x,mat_ctx->lambda_local,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  ierr = VecScatterEnd(mat_ctx->l2g_lambda,x,mat_ctx->lambda_local,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  ierr = MatMultTranspose(mat_ctx->B_delta,mat_ctx->lambda_local,pcis->vec1_B);CHKERRQ(ierr);
  /* Application of \widetilde{S}^-1 */
  ierr = VecSet(pcis->vec1_D,0.0);CHKERRQ(ierr);
  ierr = PCBDDCApplyInterfacePreconditioner(mat_ctx->pc);CHKERRQ(ierr);
  /* Application of B_delta */
  ierr = MatMult(mat_ctx->B_delta,pcis->vec1_B,mat_ctx->lambda_local);CHKERRQ(ierr);
  ierr = VecSet(y,0.0);CHKERRQ(ierr);
  ierr = VecScatterBegin(mat_ctx->l2g_lambda,mat_ctx->lambda_local,y,ADD_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(mat_ctx->l2g_lambda,mat_ctx->lambda_local,y,ADD_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "FETIDPPCApply"
static PetscErrorCode FETIDPPCApply(PC fetipc, Vec x, Vec y)
{
  FETIDPPC_ctx   *pc_ctx;
  PC_IS          *pcis;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PCShellGetContext(fetipc,(void**)&pc_ctx);
  pcis = (PC_IS*)pc_ctx->pc->data;
  /* Application of B_Ddelta^T */
  ierr = VecScatterBegin(pc_ctx->l2g_lambda,x,pc_ctx->lambda_local,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  ierr = VecScatterEnd(pc_ctx->l2g_lambda,x,pc_ctx->lambda_local,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  ierr = VecSet(pcis->vec2_B,0.0);CHKERRQ(ierr);
  ierr = MatMultTranspose(pc_ctx->B_Ddelta,pc_ctx->lambda_local,pcis->vec2_B);CHKERRQ(ierr);
  /* Application of S */
  ierr = PCISApplySchur(pc_ctx->pc,pcis->vec2_B,pcis->vec1_B,(Vec)0,pcis->vec1_D,pcis->vec2_D);CHKERRQ(ierr);
  /* Application of B_Ddelta */
  ierr = MatMult(pc_ctx->B_Ddelta,pcis->vec1_B,pc_ctx->lambda_local);CHKERRQ(ierr);
  ierr = VecSet(y,0.0);CHKERRQ(ierr);
  ierr = VecScatterBegin(pc_ctx->l2g_lambda,pc_ctx->lambda_local,y,ADD_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(pc_ctx->l2g_lambda,pc_ctx->lambda_local,y,ADD_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCBDDCSetupLocalAdjacencyGraph"
static PetscErrorCode PCBDDCSetupLocalAdjacencyGraph(PC pc)
{
  PC_BDDC        *pcbddc = (PC_BDDC*)pc->data;
  Mat_IS         *matis = (Mat_IS*)pc->pmat->data;
  PetscInt       nvtxs;
  const PetscInt *xadj,*adjncy;
  Mat            mat_adj; 
  PetscBool      symmetrize_rowij=PETSC_TRUE,compressed_rowij=PETSC_FALSE,flg_row=PETSC_TRUE;
  PCBDDCGraph    mat_graph=pcbddc->mat_graph;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* get CSR adjacency from local matrix if user has not yet provided local graph using PCBDDCSetLocalAdjacencyGraph function */
  if (!mat_graph->xadj) {
    ierr = MatConvert(matis->A,MATMPIADJ,MAT_INITIAL_MATRIX,&mat_adj);CHKERRQ(ierr);
    ierr = MatGetRowIJ(mat_adj,0,symmetrize_rowij,compressed_rowij,&nvtxs,&xadj,&adjncy,&flg_row);CHKERRQ(ierr);
    if (!flg_row) {
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Error in MatGetRowIJ called in %s\n",__FUNCT__);
    }
    /* Get adjacency into BDDC workspace */
    ierr = PCBDDCSetLocalAdjacencyGraph(pc,nvtxs,xadj,adjncy,PETSC_COPY_VALUES);CHKERRQ(ierr);
    ierr = MatRestoreRowIJ(mat_adj,0,symmetrize_rowij,compressed_rowij,&nvtxs,&xadj,&adjncy,&flg_row);CHKERRQ(ierr);
    if (!flg_row) {
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Error in MatRestoreRowIJ called in %s\n",__FUNCT__);
    }
    ierr = MatDestroy(&mat_adj);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
/* -------------------------------------------------------------------------- */
#undef __FUNCT__
#define __FUNCT__ "PCBDDCApplyInterfacePreconditioner"
static PetscErrorCode  PCBDDCApplyInterfacePreconditioner(PC pc)
{ 
  PetscErrorCode ierr;
  PC_BDDC*        pcbddc = (PC_BDDC*)(pc->data);
  PC_IS*            pcis = (PC_IS*)  (pc->data);
  const PetscScalar zero = 0.0;

  PetscFunctionBegin;
  /* Application of PHI^T  */
  ierr = MatMultTranspose(pcbddc->coarse_phi_B,pcis->vec1_B,pcbddc->vec1_P);CHKERRQ(ierr);
  if (pcbddc->prec_type) { ierr = MatMultTransposeAdd(pcbddc->coarse_phi_D,pcis->vec1_D,pcbddc->vec1_P,pcbddc->vec1_P);CHKERRQ(ierr); }

  /* Scatter data of coarse_rhs */
  if (pcbddc->coarse_rhs) { ierr = VecSet(pcbddc->coarse_rhs,zero);CHKERRQ(ierr); }
  ierr = PCBDDCScatterCoarseDataBegin(pc,pcbddc->vec1_P,pcbddc->coarse_rhs,ADD_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);

  /* Local solution on R nodes */
  ierr = VecSet(pcbddc->vec1_R,zero);CHKERRQ(ierr);
  ierr = VecScatterBegin(pcbddc->R_to_B,pcis->vec1_B,pcbddc->vec1_R,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  ierr = VecScatterEnd  (pcbddc->R_to_B,pcis->vec1_B,pcbddc->vec1_R,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  if (pcbddc->prec_type) {
    ierr = VecScatterBegin(pcbddc->R_to_D,pcis->vec1_D,pcbddc->vec1_R,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
    ierr = VecScatterEnd  (pcbddc->R_to_D,pcis->vec1_D,pcbddc->vec1_R,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  }
  ierr = PCBDDCSolveSaddlePoint(pc);CHKERRQ(ierr);
  ierr = VecSet(pcis->vec1_B,zero);CHKERRQ(ierr);
  ierr = VecScatterBegin(pcbddc->R_to_B,pcbddc->vec2_R,pcis->vec1_B,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd  (pcbddc->R_to_B,pcbddc->vec2_R,pcis->vec1_B,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  if (pcbddc->prec_type) {
    ierr = VecScatterBegin(pcbddc->R_to_D,pcbddc->vec2_R,pcis->vec1_D,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterEnd  (pcbddc->R_to_D,pcbddc->vec2_R,pcis->vec1_D,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  }

  /* Coarse solution */
  ierr = PCBDDCScatterCoarseDataEnd(pc,pcbddc->vec1_P,pcbddc->coarse_rhs,ADD_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  if (pcbddc->coarse_rhs) {
    if (pcbddc->CoarseNullSpace) {
      ierr = MatNullSpaceRemove(pcbddc->CoarseNullSpace,pcbddc->coarse_rhs,PETSC_NULL);CHKERRQ(ierr);
    }
    ierr = KSPSolve(pcbddc->coarse_ksp,pcbddc->coarse_rhs,pcbddc->coarse_vec);CHKERRQ(ierr);
    if (pcbddc->CoarseNullSpace) {
      ierr = MatNullSpaceRemove(pcbddc->CoarseNullSpace,pcbddc->coarse_vec,PETSC_NULL);CHKERRQ(ierr);
    }
  }
  ierr = PCBDDCScatterCoarseDataBegin(pc,pcbddc->coarse_vec,pcbddc->vec1_P,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  ierr = PCBDDCScatterCoarseDataEnd  (pc,pcbddc->coarse_vec,pcbddc->vec1_P,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);

  /* Sum contributions from two levels */
  ierr = MatMultAdd(pcbddc->coarse_phi_B,pcbddc->vec1_P,pcis->vec1_B,pcis->vec1_B);CHKERRQ(ierr);
  if (pcbddc->prec_type) { ierr = MatMultAdd(pcbddc->coarse_phi_D,pcbddc->vec1_P,pcis->vec1_D,pcis->vec1_D);CHKERRQ(ierr); }
  PetscFunctionReturn(0);
}
/* -------------------------------------------------------------------------- */
#undef __FUNCT__
#define __FUNCT__ "PCBDDCSolveSaddlePoint"
static PetscErrorCode  PCBDDCSolveSaddlePoint(PC pc)
{ 
  PetscErrorCode ierr;
  PC_BDDC*       pcbddc = (PC_BDDC*)(pc->data);

  PetscFunctionBegin;
  ierr = KSPSolve(pcbddc->ksp_R,pcbddc->vec1_R,pcbddc->vec2_R);CHKERRQ(ierr);
  if (pcbddc->local_auxmat1) {
    ierr = MatMult(pcbddc->local_auxmat1,pcbddc->vec2_R,pcbddc->vec1_C);CHKERRQ(ierr);
    ierr = MatMultAdd(pcbddc->local_auxmat2,pcbddc->vec1_C,pcbddc->vec2_R,pcbddc->vec2_R);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
/* -------------------------------------------------------------------------- */
#undef __FUNCT__
#define __FUNCT__ "PCBDDCScatterCoarseDataBegin"
static PetscErrorCode  PCBDDCScatterCoarseDataBegin(PC pc,Vec vec_from, Vec vec_to, InsertMode imode, ScatterMode smode)
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
#undef __FUNCT__
#define __FUNCT__ "PCBDDCScatterCoarseDataEnd"
static PetscErrorCode  PCBDDCScatterCoarseDataEnd(PC pc,Vec vec_from, Vec vec_to, InsertMode imode, ScatterMode smode)
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
      if (vec_from) VecGetArray(vec_from,&array_from);
      if (vec_to)   VecGetArray(vec_to,&array_to);
      switch(pcbddc->coarse_problem_type){
        case SEQUENTIAL_BDDC:
          if (smode == SCATTER_FORWARD) {
            ierr = MPI_Gatherv(&array_from[0],pcbddc->local_primal_size,MPIU_SCALAR,&pcbddc->replicated_local_primal_values[0],pcbddc->local_primal_sizes,pcbddc->local_primal_displacements,MPIU_SCALAR,0,comm);CHKERRQ(ierr);
            if (vec_to) {
              if (imode == ADD_VALUES) {
                for (i=0;i<pcbddc->replicated_primal_size;i++) {
                  array_to[pcbddc->replicated_local_primal_indices[i]]+=pcbddc->replicated_local_primal_values[i];
                }
              } else {
                for (i=0;i<pcbddc->replicated_primal_size;i++) {
                  array_to[pcbddc->replicated_local_primal_indices[i]]=pcbddc->replicated_local_primal_values[i];
                }
              }
            }
          } else {
            if (vec_from) {
              if (imode == ADD_VALUES) {
                printf("Scatter mode %d, insert mode %d for case %d not implemented!\n",smode,imode,pcbddc->coarse_problem_type);
              }
              for (i=0;i<pcbddc->replicated_primal_size;i++) {
                pcbddc->replicated_local_primal_values[i]=array_from[pcbddc->replicated_local_primal_indices[i]];
              }
            }
            ierr = MPI_Scatterv(&pcbddc->replicated_local_primal_values[0],pcbddc->local_primal_sizes,pcbddc->local_primal_displacements,MPIU_SCALAR,&array_to[0],pcbddc->local_primal_size,MPIU_SCALAR,0,comm);CHKERRQ(ierr);
          }
          break;
        case REPLICATED_BDDC:
          if (smode == SCATTER_FORWARD) {
            ierr = MPI_Allgatherv(&array_from[0],pcbddc->local_primal_size,MPIU_SCALAR,&pcbddc->replicated_local_primal_values[0],pcbddc->local_primal_sizes,pcbddc->local_primal_displacements,MPIU_SCALAR,comm);CHKERRQ(ierr);
            if (imode == ADD_VALUES) {
              for (i=0;i<pcbddc->replicated_primal_size;i++) {
                array_to[pcbddc->replicated_local_primal_indices[i]]+=pcbddc->replicated_local_primal_values[i];
              }
            } else {
              for (i=0;i<pcbddc->replicated_primal_size;i++) {
                array_to[pcbddc->replicated_local_primal_indices[i]]=pcbddc->replicated_local_primal_values[i];
              }
            }
          } else { /* no communications needed for SCATTER_REVERSE since needed data is already present */
            if (imode == ADD_VALUES) {
              for (i=0;i<pcbddc->local_primal_size;i++) {
                array_to[i]+=array_from[pcbddc->local_primal_indices[i]]; 
              }
            } else {
              for (i=0;i<pcbddc->local_primal_size;i++) {
                array_to[i]=array_from[pcbddc->local_primal_indices[i]]; 
              }
            }
          }
          break;
        case MULTILEVEL_BDDC:
          break;
        case PARALLEL_BDDC:
          break;
      }
      if (vec_from) VecRestoreArray(vec_from,&array_from);
      if (vec_to)   VecRestoreArray(vec_to,&array_to);
      break;
  }
  PetscFunctionReturn(0);
}
/* -------------------------------------------------------------------------- */
#undef __FUNCT__  
#define __FUNCT__ "PCBDDCCreateConstraintMatrix"
static PetscErrorCode PCBDDCCreateConstraintMatrix(PC pc)
{   
  PetscErrorCode ierr;
  PC_IS*         pcis = (PC_IS*)(pc->data);
  PC_BDDC*       pcbddc = (PC_BDDC*)pc->data;
  Mat_IS         *matis = (Mat_IS*)pc->pmat->data;
  PetscInt       *nnz,*is_indices;
  PetscScalar    *temp_quadrature_constraint;
  PetscInt       *temp_indices,*temp_indices_to_constraint,*temp_indices_to_constraint_B,*local_to_B;
  PetscInt       local_primal_size,i,j,k,total_counts,max_size_of_constraint;
  PetscInt       n_constraints,n_vertices,size_of_constraint;
  PetscScalar    quad_value;
  PetscBool      nnsp_has_cnst=PETSC_FALSE,use_nnsp_true=pcbddc->use_nnsp_true;
  PetscInt       nnsp_size=0,nnsp_addone=0,temp_constraints,temp_start_ptr;
  IS             *used_IS;
  MatType        impMatType=MATSEQAIJ;
  PetscBLASInt   Bs,Bt,lwork,lierr;
  PetscReal      tol=1.0e-8;
  MatNullSpace   nearnullsp;
  const Vec      *nearnullvecs;
  Vec            *localnearnullsp;
  PetscScalar    *work,*temp_basis,*array_vector,*correlation_mat;
  PetscReal      *rwork,*singular_vals;
  PetscBLASInt   Bone=1,*ipiv;
  Vec            temp_vec;
  Mat            temp_mat;
  KSP            temp_ksp;
  PC             temp_pc;
  PetscInt       s,start_constraint,dual_dofs;
  PetscBool      compute_submatrix,useksp=PETSC_FALSE;
  PetscInt       *aux_primal_permutation,*aux_primal_numbering;
  PetscBool      boolforface,*change_basis;
/* some ugly conditional declarations */
#if defined(PETSC_MISSING_LAPACK_GESVD)
  PetscScalar    dot_result;
  PetscScalar    one=1.0,zero=0.0;
  PetscInt       ii;
  PetscScalar    *singular_vectors;
  PetscBLASInt   *iwork,*ifail;
  PetscReal      dummy_real,abs_tol;
  PetscBLASInt   eigs_found;
#if defined(PETSC_USE_COMPLEX)
  PetscScalar    val1,val2;
#endif
#endif
  PetscBLASInt   dummy_int;
  PetscScalar    dummy_scalar;

  PetscFunctionBegin;
  /* check if near null space is attached to global mat */
  ierr = MatGetNearNullSpace(pc->pmat,&nearnullsp);CHKERRQ(ierr);
  if (nearnullsp) {
    ierr = MatNullSpaceGetVecs(nearnullsp,&nnsp_has_cnst,&nnsp_size,&nearnullvecs);CHKERRQ(ierr);
  } else { /* if near null space is not provided it uses constants */ 
    nnsp_has_cnst = PETSC_TRUE;
    use_nnsp_true = PETSC_TRUE; 
  }
  if (nnsp_has_cnst) {
    nnsp_addone = 1;
  }
  /*
       Evaluate maximum storage size needed by the procedure
       - temp_indices will contain start index of each constraint stored as follows
       - temp_indices_to_constraint  [temp_indices[i],...,temp[indices[i+1]-1] will contain the indices (in local numbering) on which the constraint acts
       - temp_indices_to_constraint_B[temp_indices[i],...,temp[indices[i+1]-1] will contain the indices (in boundary numbering) on which the constraint acts
       - temp_quadrature_constraint  [temp_indices[i],...,temp[indices[i+1]-1] will contain the scalars representing the constraint itself
                                                                                                                                                         */

  total_counts = pcbddc->n_ISForFaces+pcbddc->n_ISForEdges;
  total_counts *= (nnsp_addone+nnsp_size);
  ierr = ISGetSize(pcbddc->ISForVertices,&n_vertices);CHKERRQ(ierr);
  total_counts += n_vertices;
  ierr = PetscMalloc((total_counts+1)*sizeof(PetscInt),&temp_indices);CHKERRQ(ierr);
  ierr = PetscMalloc((total_counts+1)*sizeof(PetscBool),&change_basis);CHKERRQ(ierr);
  total_counts = 0;
  max_size_of_constraint = 0;
  for (i=0;i<pcbddc->n_ISForEdges+pcbddc->n_ISForFaces;i++){
    if (i<pcbddc->n_ISForEdges){
      used_IS = &pcbddc->ISForEdges[i];
    } else {
      used_IS = &pcbddc->ISForFaces[i-pcbddc->n_ISForEdges];
    }
    ierr = ISGetSize(*used_IS,&j);CHKERRQ(ierr); 
    total_counts += j;
    if (j>max_size_of_constraint) max_size_of_constraint=j;
  }
  total_counts *= (nnsp_addone+nnsp_size);
  total_counts += n_vertices;
  ierr = PetscMalloc(total_counts*sizeof(PetscScalar),&temp_quadrature_constraint);CHKERRQ(ierr);
  ierr = PetscMalloc(total_counts*sizeof(PetscInt),&temp_indices_to_constraint);CHKERRQ(ierr);
  ierr = PetscMalloc(total_counts*sizeof(PetscInt),&temp_indices_to_constraint_B);CHKERRQ(ierr);
  ierr = PetscMalloc(pcis->n*sizeof(PetscInt),&local_to_B);CHKERRQ(ierr);
  ierr = ISGetIndices(pcis->is_B_local,(const PetscInt**)&is_indices);CHKERRQ(ierr);
  for (i=0;i<pcis->n;i++) {
    local_to_B[i]=-1;
  }
  for (i=0;i<pcis->n_B;i++) {
    local_to_B[is_indices[i]]=i;
  }
  ierr = ISRestoreIndices(pcis->is_B_local,(const PetscInt**)&is_indices);CHKERRQ(ierr);

  /* First we issue queries to allocate optimal workspace for LAPACKgesvd or LAPACKsyev/LAPACKheev */
  rwork = 0;
  work = 0;
  singular_vals = 0;
  temp_basis = 0;
  correlation_mat = 0;
  if (!pcbddc->use_nnsp_true) {
    PetscScalar temp_work;
#if defined(PETSC_MISSING_LAPACK_GESVD)
    /* POD */
    PetscInt max_n;
    max_n = nnsp_addone+nnsp_size;
    /* using some techniques borrowed from Proper Orthogonal Decomposition */
    ierr = PetscMalloc(max_n*max_n*sizeof(PetscScalar),&correlation_mat);CHKERRQ(ierr);
    ierr = PetscMalloc(max_n*max_n*sizeof(PetscScalar),&singular_vectors);CHKERRQ(ierr);
    ierr = PetscMalloc(max_n*sizeof(PetscReal),&singular_vals);CHKERRQ(ierr);
    ierr = PetscMalloc(max_size_of_constraint*(nnsp_addone+nnsp_size)*sizeof(PetscScalar),&temp_basis);CHKERRQ(ierr);
#if defined(PETSC_USE_COMPLEX)
    ierr = PetscMalloc(3*max_n*sizeof(PetscReal),&rwork);CHKERRQ(ierr);
#endif
    ierr = PetscMalloc(5*max_n*sizeof(PetscBLASInt),&iwork);CHKERRQ(ierr);
    ierr = PetscMalloc(max_n*sizeof(PetscBLASInt),&ifail);CHKERRQ(ierr);
    /* now we evaluate the optimal workspace using query with lwork=-1 */
    Bt = PetscBLASIntCast(max_n);
    lwork=-1;
    ierr = PetscFPTrapPush(PETSC_FP_TRAP_OFF);CHKERRQ(ierr);
#if !defined(PETSC_USE_COMPLEX)
    abs_tol=1.e-8;
/*    LAPACKsyev_("V","U",&Bt,correlation_mat,&Bt,singular_vals,&temp_work,&lwork,&lierr); */
    LAPACKsyevx_("V","A","U",&Bt,correlation_mat,&Bt,&dummy_real,&dummy_real,&dummy_int,&dummy_int,
                 &abs_tol,&eigs_found,singular_vals,singular_vectors,&Bt,&temp_work,&lwork,iwork,ifail,&lierr);
#else
/*    LAPACKsyev_("V","U",&Bt,correlation_mat,&Bt,singular_vals,&temp_work,&lwork,rwork,&lierr); */
/*  LAPACK call is missing here! TODO */
    SETERRQ(((PetscObject) pc)->comm, PETSC_ERR_SUP, "Not yet implemented for complexes when PETSC_MISSING_GESVD = 1");
#endif
    if ( lierr ) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in query to SYEVX Lapack routine %d",(int)lierr);
    ierr = PetscFPTrapPop();CHKERRQ(ierr);
#else /* on missing GESVD */
    /* SVD */
    PetscInt max_n,min_n;
    max_n = max_size_of_constraint;
    min_n = nnsp_addone+nnsp_size;
    if (max_size_of_constraint < ( nnsp_addone+nnsp_size ) ) {
      min_n = max_size_of_constraint;
      max_n = nnsp_addone+nnsp_size;
    }
    ierr = PetscMalloc(min_n*sizeof(PetscReal),&singular_vals);CHKERRQ(ierr);
#if defined(PETSC_USE_COMPLEX)
    ierr = PetscMalloc(5*min_n*sizeof(PetscReal),&rwork);CHKERRQ(ierr);
#endif
    /* now we evaluate the optimal workspace using query with lwork=-1 */
    lwork=-1;
    Bs = PetscBLASIntCast(max_n);
    Bt = PetscBLASIntCast(min_n);
    dummy_int = Bs;
    ierr = PetscFPTrapPush(PETSC_FP_TRAP_OFF);CHKERRQ(ierr);
#if !defined(PETSC_USE_COMPLEX)
    LAPACKgesvd_("O","N",&Bs,&Bt,&temp_quadrature_constraint[0],&Bs,singular_vals,
                 &dummy_scalar,&dummy_int,&dummy_scalar,&dummy_int,&temp_work,&lwork,&lierr);
#else
    LAPACKgesvd_("O","N",&Bs,&Bt,&temp_quadrature_constraint[0],&Bs,singular_vals,
                 &dummy_scalar,&dummy_int,&dummy_scalar,&dummy_int,&temp_work,&lwork,rwork,&lierr);
#endif
    if ( lierr ) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in query to SVD Lapack routine %d",(int)lierr);
    ierr = PetscFPTrapPop();CHKERRQ(ierr);
#endif
    /* Allocate optimal workspace */
    lwork = PetscBLASIntCast((PetscInt)PetscRealPart(temp_work));
    total_counts = (PetscInt)lwork;
    ierr = PetscMalloc(total_counts*sizeof(PetscScalar),&work);CHKERRQ(ierr);
  }
  /* get local part of global near null space vectors */
  ierr = PetscMalloc(nnsp_size*sizeof(Vec),&localnearnullsp);CHKERRQ(ierr);
  for (k=0;k<nnsp_size;k++) {
    ierr = VecDuplicate(pcis->vec1_N,&localnearnullsp[k]);CHKERRQ(ierr);
    ierr = VecScatterBegin(matis->ctx,nearnullvecs[k],localnearnullsp[k],INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterEnd  (matis->ctx,nearnullvecs[k],localnearnullsp[k],INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  }
  /* Now we can loop on constraining sets */
  total_counts=0;
  temp_indices[0]=0;
  /* vertices */
  PetscBool used_vertex;
  ierr = ISGetIndices(pcbddc->ISForVertices,(const PetscInt**)&is_indices);CHKERRQ(ierr);
  if (nnsp_has_cnst) { /* consider all vertices */
    for (i=0;i<n_vertices;i++) {
      temp_indices_to_constraint[temp_indices[total_counts]]=is_indices[i];
      temp_indices_to_constraint_B[temp_indices[total_counts]]=local_to_B[is_indices[i]];
      temp_quadrature_constraint[temp_indices[total_counts]]=1.0;
      temp_indices[total_counts+1]=temp_indices[total_counts]+1;
      change_basis[total_counts]=PETSC_FALSE; 
      total_counts++;
    }
  } else { /* consider vertices for which exist at least a localnearnullsp which is not null there */
    for (i=0;i<n_vertices;i++) {
      used_vertex=PETSC_FALSE;
      k=0;
      while(!used_vertex && k<nnsp_size) {
        ierr = VecGetArrayRead(localnearnullsp[k],(const PetscScalar**)&array_vector);CHKERRQ(ierr);
        if (PetscAbsScalar(array_vector[is_indices[i]])>0.0) {
          temp_indices_to_constraint[temp_indices[total_counts]]=is_indices[i];
          temp_indices_to_constraint_B[temp_indices[total_counts]]=local_to_B[is_indices[i]];
          temp_quadrature_constraint[temp_indices[total_counts]]=1.0;
          temp_indices[total_counts+1]=temp_indices[total_counts]+1;
          change_basis[total_counts]=PETSC_FALSE; 
          total_counts++; 
          used_vertex=PETSC_TRUE;
        }
        ierr = VecRestoreArrayRead(localnearnullsp[k],(const PetscScalar**)&array_vector);CHKERRQ(ierr);
        k++;
      }
    }
  }
  ierr = ISRestoreIndices(pcbddc->ISForVertices,(const PetscInt**)&is_indices);CHKERRQ(ierr);
  n_vertices=total_counts;
  /* edges and faces */
  for (i=0;i<pcbddc->n_ISForEdges+pcbddc->n_ISForFaces;i++){
    if (i<pcbddc->n_ISForEdges){
      used_IS = &pcbddc->ISForEdges[i];
      boolforface = pcbddc->usechangeofbasis;
    } else {
      used_IS = &pcbddc->ISForFaces[i-pcbddc->n_ISForEdges];
      boolforface = pcbddc->usechangeonfaces;
    }
    temp_constraints = 0;          /* zero the number of constraints I have on this conn comp */
    temp_start_ptr = total_counts; /* need to know the starting index of constraints stored */
    ierr = ISGetSize(*used_IS,&size_of_constraint);CHKERRQ(ierr);
    ierr = ISGetIndices(*used_IS,(const PetscInt**)&is_indices);CHKERRQ(ierr);
    if (nnsp_has_cnst) {
      temp_constraints++;
      quad_value = (PetscScalar) (1.0/PetscSqrtReal((PetscReal)size_of_constraint));
      for (j=0;j<size_of_constraint;j++) {
        temp_indices_to_constraint[temp_indices[total_counts]+j]=is_indices[j];
        temp_indices_to_constraint_B[temp_indices[total_counts]+j]=local_to_B[is_indices[j]];
        temp_quadrature_constraint[temp_indices[total_counts]+j]=quad_value;
      }
      temp_indices[total_counts+1]=temp_indices[total_counts]+size_of_constraint;  /* store new starting point */
      change_basis[total_counts]=boolforface;
      total_counts++; 
    }
    for (k=0;k<nnsp_size;k++) {
      ierr = VecGetArrayRead(localnearnullsp[k],(const PetscScalar**)&array_vector);CHKERRQ(ierr);
      for (j=0;j<size_of_constraint;j++) {
        temp_indices_to_constraint[temp_indices[total_counts]+j]=is_indices[j];
        temp_indices_to_constraint_B[temp_indices[total_counts]+j]=local_to_B[is_indices[j]];
        temp_quadrature_constraint[temp_indices[total_counts]+j]=array_vector[is_indices[j]];
      }
      ierr = VecRestoreArrayRead(localnearnullsp[k],(const PetscScalar**)&array_vector);CHKERRQ(ierr);
      quad_value = 1.0;
      if ( use_nnsp_true ) { /* check if array is null on the connected component in case use_nnsp_true has been requested */
        Bs = PetscBLASIntCast(size_of_constraint);
        quad_value = BLASasum_(&Bs,&temp_quadrature_constraint[temp_indices[total_counts]],&Bone);
      }
      if ( quad_value > 0.0 ) { /* keep indices and values */
        temp_constraints++;
        temp_indices[total_counts+1]=temp_indices[total_counts]+size_of_constraint;  /* store new starting point */
        change_basis[total_counts]=boolforface;
        total_counts++;
      }
    }
    ierr = ISRestoreIndices(*used_IS,(const PetscInt**)&is_indices);CHKERRQ(ierr);
    /* perform SVD on the constraint if use_nnsp_true has not be requested by the user */
    if (!use_nnsp_true) {

      Bs = PetscBLASIntCast(size_of_constraint);
      Bt = PetscBLASIntCast(temp_constraints);

#if defined(PETSC_MISSING_LAPACK_GESVD)
      ierr = PetscMemzero(correlation_mat,Bt*Bt*sizeof(PetscScalar));CHKERRQ(ierr);
      /* Store upper triangular part of correlation matrix */
      for (j=0;j<temp_constraints;j++) {
        for (k=0;k<j+1;k++) {
#if defined(PETSC_USE_COMPLEX)
          /* hand made complex dot product -> replace */
          dot_result = 0.0;
          for (ii=0; ii<size_of_constraint; ii++) {
            val1 = temp_quadrature_constraint[temp_indices[temp_start_ptr+j]+ii];
            val2 = temp_quadrature_constraint[temp_indices[temp_start_ptr+k]+ii];
            dot_result += val1*PetscConj(val2);
          }
#else
          dot_result = BLASdot_(&Bs,&temp_quadrature_constraint[temp_indices[temp_start_ptr+j]],&Bone,
                                    &temp_quadrature_constraint[temp_indices[temp_start_ptr+k]],&Bone);
#endif
          correlation_mat[j*temp_constraints+k]=dot_result;
        }
      }
      ierr = PetscFPTrapPush(PETSC_FP_TRAP_OFF);CHKERRQ(ierr);
#if !defined(PETSC_USE_COMPLEX)
/*      LAPACKsyev_("V","U",&Bt,correlation_mat,&Bt,singular_vals,work,&lwork,&lierr); */
      LAPACKsyevx_("V","A","U",&Bt,correlation_mat,&Bt,&dummy_real,&dummy_real,&dummy_int,&dummy_int,
                 &abs_tol,&eigs_found,singular_vals,singular_vectors,&Bt,work,&lwork,iwork,ifail,&lierr);
#else
/*  LAPACK call is missing here! TODO */
      SETERRQ(((PetscObject) pc)->comm, PETSC_ERR_SUP, "Not yet implemented for complexes when PETSC_MISSING_GESVD = 1");
#endif   
      if ( lierr ) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in SYEVX Lapack routine %d",(int)lierr);
      ierr = PetscFPTrapPop();CHKERRQ(ierr);
      /* retain eigenvalues greater than tol: note that lapack SYEV gives eigs in ascending order */
      j=0;
      while( j < Bt && singular_vals[j] < tol) j++;
      total_counts=total_counts-j;
      if (j<temp_constraints) {
        for (k=j;k<Bt;k++) { singular_vals[k]=1.0/PetscSqrtReal(singular_vals[k]); }
        ierr = PetscFPTrapPush(PETSC_FP_TRAP_OFF);CHKERRQ(ierr);
        BLASgemm_("N","N",&Bs,&Bt,&Bt,&one,&temp_quadrature_constraint[temp_indices[temp_start_ptr]],&Bs,correlation_mat,&Bt,&zero,temp_basis,&Bs);
        ierr = PetscFPTrapPop();CHKERRQ(ierr);
        /* copy POD basis into used quadrature memory */
        for (k=0;k<Bt-j;k++) {
          for (ii=0;ii<size_of_constraint;ii++) {
            temp_quadrature_constraint[temp_indices[temp_start_ptr+k]+ii]=singular_vals[Bt-1-k]*temp_basis[(Bt-1-k)*size_of_constraint+ii];
          }
        }
      }

#else  /* on missing GESVD */
      PetscInt min_n = temp_constraints;
      if (min_n > size_of_constraint) min_n = size_of_constraint;
      dummy_int = Bs;
      ierr = PetscFPTrapPush(PETSC_FP_TRAP_OFF);CHKERRQ(ierr);
#if !defined(PETSC_USE_COMPLEX)
      LAPACKgesvd_("O","N",&Bs,&Bt,&temp_quadrature_constraint[temp_indices[temp_start_ptr]],&Bs,singular_vals,
                   &dummy_scalar,&dummy_int,&dummy_scalar,&dummy_int,work,&lwork,&lierr);
#else
      LAPACKgesvd_("O","N",&Bs,&Bt,&temp_quadrature_constraint[temp_indices[temp_start_ptr]],&Bs,singular_vals,
                   &dummy_scalar,&dummy_int,&dummy_scalar,&dummy_int,work,&lwork,rwork,&lierr);
#endif
      if ( lierr ) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in SVD Lapack routine %d",(int)lierr);
      ierr = PetscFPTrapPop();CHKERRQ(ierr);
      /* retain eigenvalues greater than tol: note that lapack SVD gives eigs in descending order */
      j=0;
      while( j < min_n && singular_vals[min_n-j-1] < tol) j++;
      total_counts = total_counts-(PetscInt)Bt+(min_n-j);
#endif
    }
  }

  n_constraints=total_counts-n_vertices;
  local_primal_size = total_counts;
  /* set quantities in pcbddc data structure */
  pcbddc->n_vertices = n_vertices;
  pcbddc->n_constraints = n_constraints;
  pcbddc->local_primal_size = local_primal_size;

  /* Create constraint matrix */
  /* The constraint matrix is used to compute the l2g map of primal dofs */
  /* so we need to set it up properly either with or without change of basis */
  ierr = MatCreate(PETSC_COMM_SELF,&pcbddc->ConstraintMatrix);CHKERRQ(ierr);
  ierr = MatSetType(pcbddc->ConstraintMatrix,impMatType);CHKERRQ(ierr);
  ierr = MatSetSizes(pcbddc->ConstraintMatrix,local_primal_size,pcis->n,local_primal_size,pcis->n);CHKERRQ(ierr);
  /* compute a local numbering of constraints : vertices first then constraints */
  ierr = VecSet(pcis->vec1_N,0.0);CHKERRQ(ierr);
  ierr = VecGetArray(pcis->vec1_N,&array_vector);CHKERRQ(ierr);
  ierr = PetscMalloc(local_primal_size*sizeof(PetscInt),&aux_primal_numbering);CHKERRQ(ierr);
  ierr = PetscMalloc(local_primal_size*sizeof(PetscInt),&aux_primal_permutation);CHKERRQ(ierr);
  total_counts=0;
  /* find vertices: subdomain corners plus dofs with basis changed */
  for (i=0;i<local_primal_size;i++) {
    size_of_constraint=temp_indices[i+1]-temp_indices[i];
    if (change_basis[i] || size_of_constraint == 1) {
      k=0;
      while(k < size_of_constraint && array_vector[temp_indices_to_constraint[temp_indices[i]+size_of_constraint-k-1]] != 0.0) {
        k=k+1;
      }
      j=temp_indices_to_constraint[temp_indices[i]+size_of_constraint-k-1];
      array_vector[j] = 1.0;
      aux_primal_numbering[total_counts]=j;
      aux_primal_permutation[total_counts]=total_counts;
      total_counts++;
    }
  }
  ierr = VecRestoreArray(pcis->vec1_N,&array_vector);CHKERRQ(ierr);
  /* permute indices in order to have a sorted set of vertices */
  ierr = PetscSortIntWithPermutation(total_counts,aux_primal_numbering,aux_primal_permutation);
  /* nonzero structure */
  ierr = PetscMalloc(local_primal_size*sizeof(PetscInt),&nnz);CHKERRQ(ierr);
  for (i=0;i<total_counts;i++) {
    nnz[i]=1;
  }
  j=total_counts;
  for (i=n_vertices;i<local_primal_size;i++) {
    if (!change_basis[i]) {
      nnz[j]=temp_indices[i+1]-temp_indices[i];
      j++;
    }
  }
  ierr = MatSeqAIJSetPreallocation(pcbddc->ConstraintMatrix,0,nnz);CHKERRQ(ierr);
  ierr = PetscFree(nnz);CHKERRQ(ierr);
  /* set values in constraint matrix */
  for (i=0;i<total_counts;i++) {
    j = aux_primal_permutation[i];
    k = aux_primal_numbering[j];
    ierr = MatSetValue(pcbddc->ConstraintMatrix,i,k,1.0,INSERT_VALUES);CHKERRQ(ierr);
  }
  for (i=n_vertices;i<local_primal_size;i++) {
    if (!change_basis[i]) {
      size_of_constraint=temp_indices[i+1]-temp_indices[i];
      ierr = MatSetValues(pcbddc->ConstraintMatrix,1,&total_counts,size_of_constraint,&temp_indices_to_constraint[temp_indices[i]],&temp_quadrature_constraint[temp_indices[i]],INSERT_VALUES);CHKERRQ(ierr);
      total_counts++;
    }
  }
  ierr = PetscFree(aux_primal_numbering);CHKERRQ(ierr);
  ierr = PetscFree(aux_primal_permutation);CHKERRQ(ierr);
  /* assembling */
  ierr = MatAssemblyBegin(pcbddc->ConstraintMatrix,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(pcbddc->ConstraintMatrix,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  /* Create matrix for change of basis. We don't need it in case pcbddc->usechangeofbasis is FALSE */
  if (pcbddc->usechangeofbasis) {
    ierr = MatCreate(PETSC_COMM_SELF,&pcbddc->ChangeOfBasisMatrix);CHKERRQ(ierr);
    ierr = MatSetType(pcbddc->ChangeOfBasisMatrix,impMatType);CHKERRQ(ierr);
    ierr = MatSetSizes(pcbddc->ChangeOfBasisMatrix,pcis->n_B,pcis->n_B,pcis->n_B,pcis->n_B);CHKERRQ(ierr);
    /* work arrays */
    /* we need to reuse these arrays, so we free them */
    ierr = PetscFree(temp_basis);CHKERRQ(ierr);
    ierr = PetscFree(work);CHKERRQ(ierr);
    ierr = PetscMalloc(pcis->n_B*sizeof(PetscInt),&nnz);CHKERRQ(ierr);
    ierr = PetscMalloc((nnsp_addone+nnsp_size)*(nnsp_addone+nnsp_size)*sizeof(PetscScalar),&temp_basis);CHKERRQ(ierr);
    ierr = PetscMalloc((nnsp_addone+nnsp_size)*sizeof(PetscScalar),&work);CHKERRQ(ierr);
    ierr = PetscMalloc((nnsp_addone+nnsp_size)*sizeof(PetscBLASInt),&ipiv);CHKERRQ(ierr);
    for (i=0;i<pcis->n_B;i++) {
      nnz[i]=1;
    }
    /* Overestimated nonzeros per row */
    k=1;
    for (i=pcbddc->n_vertices;i<local_primal_size;i++) {
      if (change_basis[i]) {
        size_of_constraint = temp_indices[i+1]-temp_indices[i];
        if (k < size_of_constraint) {
          k = size_of_constraint;
        }
        for (j=0;j<size_of_constraint;j++) {
          nnz[temp_indices_to_constraint_B[temp_indices[i]+j]] = size_of_constraint;
        }
      }
    }
    ierr = MatSeqAIJSetPreallocation(pcbddc->ChangeOfBasisMatrix,0,nnz);CHKERRQ(ierr);
    ierr = PetscFree(nnz);CHKERRQ(ierr);
    /* Temporary array to store indices */
    ierr = PetscMalloc(k*sizeof(PetscInt),&is_indices);CHKERRQ(ierr);
    /* Set initial identity in the matrix */
    for (i=0;i<pcis->n_B;i++) {
      ierr = MatSetValue(pcbddc->ChangeOfBasisMatrix,i,i,1.0,INSERT_VALUES);CHKERRQ(ierr);
    }
    /* Now we loop on the constraints which need a change of basis */
    /* Change of basis matrix is evaluated as the FIRST APPROACH in */
    /* Klawonn and Widlund, Dual-primal FETI-DP methods for linear elasticity, (6.2.1) */
    temp_constraints = 0;
    if (pcbddc->n_vertices < local_primal_size) {
      temp_start_ptr = temp_indices_to_constraint_B[temp_indices[pcbddc->n_vertices]];
    }
    for (i=pcbddc->n_vertices;i<local_primal_size;i++) {
      if (change_basis[i]) {
        compute_submatrix = PETSC_FALSE;
        useksp = PETSC_FALSE;
        if (temp_start_ptr == temp_indices_to_constraint_B[temp_indices[i]]) {
          temp_constraints++;
          if (i == local_primal_size -1 ||  temp_start_ptr != temp_indices_to_constraint_B[temp_indices[i+1]]) {
            compute_submatrix = PETSC_TRUE;
          }
        } 
        if (compute_submatrix) {
          if (temp_constraints > 1 || pcbddc->use_nnsp_true) {
            useksp = PETSC_TRUE;
          }
          size_of_constraint = temp_indices[i+1]-temp_indices[i];
          if (useksp) { /* experimental */
            ierr = MatCreate(PETSC_COMM_SELF,&temp_mat);CHKERRQ(ierr);
            ierr = MatSetType(temp_mat,impMatType);CHKERRQ(ierr);
            ierr = MatSetSizes(temp_mat,size_of_constraint,size_of_constraint,size_of_constraint,size_of_constraint);CHKERRQ(ierr);
            ierr = MatSeqAIJSetPreallocation(temp_mat,size_of_constraint,PETSC_NULL);CHKERRQ(ierr);
          }
          /* First _size_of_constraint-temp_constraints_ columns */
          dual_dofs = size_of_constraint-temp_constraints;
          start_constraint = i+1-temp_constraints;
          for (s=0;s<dual_dofs;s++) {
            is_indices[0] = s;
            for (j=0;j<temp_constraints;j++) {
              for (k=0;k<temp_constraints;k++) { 
                temp_basis[j*temp_constraints+k]=temp_quadrature_constraint[temp_indices[start_constraint+k]+s+j+1];
              }
              work[j]=-temp_quadrature_constraint[temp_indices[start_constraint+j]+s];
              is_indices[j+1]=s+j+1;
            }
            Bt = temp_constraints;
            ierr = PetscFPTrapPush(PETSC_FP_TRAP_OFF);CHKERRQ(ierr);
            LAPACKgesv_(&Bt,&Bone,temp_basis,&Bt,ipiv,work,&Bt,&lierr);
            if ( lierr ) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in GESV Lapack routine %d",(int)lierr);          
            ierr = PetscFPTrapPop();CHKERRQ(ierr);
            j = temp_indices_to_constraint_B[temp_indices[start_constraint]+s];
            ierr = MatSetValues(pcbddc->ChangeOfBasisMatrix,temp_constraints,&temp_indices_to_constraint_B[temp_indices[start_constraint]+s+1],1,&j,work,INSERT_VALUES);CHKERRQ(ierr);
            if (useksp) {
              /* temp mat with transposed rows and columns */
              ierr = MatSetValues(temp_mat,1,&s,temp_constraints,&is_indices[1],work,INSERT_VALUES);CHKERRQ(ierr);
              ierr = MatSetValue(temp_mat,is_indices[0],is_indices[0],1.0,INSERT_VALUES);CHKERRQ(ierr);
            }
          }
          if (useksp) {
            /* last rows of temp_mat */
            for (j=0;j<size_of_constraint;j++) {
              is_indices[j] = j;
            }
            for (s=0;s<temp_constraints;s++) {
              k = s + dual_dofs;
              ierr = MatSetValues(temp_mat,1,&k,size_of_constraint,is_indices,&temp_quadrature_constraint[temp_indices[start_constraint+s]],INSERT_VALUES);CHKERRQ(ierr);
            }
            ierr = MatAssemblyBegin(temp_mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
            ierr = MatAssemblyEnd(temp_mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
            ierr = MatGetVecs(temp_mat,&temp_vec,PETSC_NULL);CHKERRQ(ierr);
            ierr = KSPCreate(PETSC_COMM_SELF,&temp_ksp);CHKERRQ(ierr);
            ierr = KSPSetOperators(temp_ksp,temp_mat,temp_mat,SAME_PRECONDITIONER);CHKERRQ(ierr);
            ierr = KSPSetType(temp_ksp,KSPPREONLY);CHKERRQ(ierr);
            ierr = KSPGetPC(temp_ksp,&temp_pc);CHKERRQ(ierr);
            ierr = PCSetType(temp_pc,PCLU);CHKERRQ(ierr);
            ierr = KSPSetUp(temp_ksp);CHKERRQ(ierr);
            for (s=0;s<temp_constraints;s++) {
              ierr = VecSet(temp_vec,0.0);CHKERRQ(ierr);
              ierr = VecSetValue(temp_vec,s+dual_dofs,1.0,INSERT_VALUES);CHKERRQ(ierr);
              ierr = VecAssemblyBegin(temp_vec);CHKERRQ(ierr);
              ierr = VecAssemblyEnd(temp_vec);CHKERRQ(ierr);
              ierr = KSPSolve(temp_ksp,temp_vec,temp_vec);CHKERRQ(ierr);
              ierr = VecGetArray(temp_vec,&array_vector);CHKERRQ(ierr);
              j = temp_indices_to_constraint_B[temp_indices[start_constraint+s]+size_of_constraint-s-1];
              /* last columns of change of basis matrix associated to new primal dofs */
              ierr = MatSetValues(pcbddc->ChangeOfBasisMatrix,size_of_constraint,&temp_indices_to_constraint_B[temp_indices[start_constraint+s]],1,&j,array_vector,INSERT_VALUES);CHKERRQ(ierr);
              ierr = VecRestoreArray(temp_vec,&array_vector);CHKERRQ(ierr);
            }
            ierr = MatDestroy(&temp_mat);CHKERRQ(ierr);
            ierr = KSPDestroy(&temp_ksp);CHKERRQ(ierr);
            ierr = VecDestroy(&temp_vec);CHKERRQ(ierr);
          } else {
            /* last columns of change of basis matrix associated to new primal dofs */
            for (s=0;s<temp_constraints;s++) {
              j = temp_indices_to_constraint_B[temp_indices[start_constraint+s]+size_of_constraint-s-1];
              ierr = MatSetValues(pcbddc->ChangeOfBasisMatrix,size_of_constraint,&temp_indices_to_constraint_B[temp_indices[start_constraint+s]],1,&j,&temp_quadrature_constraint[temp_indices[start_constraint+s]],INSERT_VALUES);CHKERRQ(ierr);
            }
          }
          /* prepare for the next cycle */
          temp_constraints = 0;
          if (i != local_primal_size -1 ) { 
            temp_start_ptr = temp_indices_to_constraint_B[temp_indices[i+1]];
          }
        }
      }
    }
    /* assembling */
    ierr = MatAssemblyBegin(pcbddc->ChangeOfBasisMatrix,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(pcbddc->ChangeOfBasisMatrix,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = PetscFree(ipiv);CHKERRQ(ierr);
    ierr = PetscFree(is_indices);CHKERRQ(ierr);
  }
  /* free workspace no longer needed */ 
  ierr = PetscFree(rwork);CHKERRQ(ierr);
  ierr = PetscFree(work);CHKERRQ(ierr);
  ierr = PetscFree(temp_basis);CHKERRQ(ierr);
  ierr = PetscFree(singular_vals);CHKERRQ(ierr);
  ierr = PetscFree(correlation_mat);CHKERRQ(ierr);
  ierr = PetscFree(temp_indices);CHKERRQ(ierr);
  ierr = PetscFree(change_basis);CHKERRQ(ierr);
  ierr = PetscFree(temp_indices_to_constraint);CHKERRQ(ierr);
  ierr = PetscFree(temp_indices_to_constraint_B);CHKERRQ(ierr);
  ierr = PetscFree(local_to_B);CHKERRQ(ierr);
  ierr = PetscFree(temp_quadrature_constraint);CHKERRQ(ierr);
#if defined(PETSC_MISSING_LAPACK_GESVD)
  ierr = PetscFree(iwork);CHKERRQ(ierr);
  ierr = PetscFree(ifail);CHKERRQ(ierr);
  ierr = PetscFree(singular_vectors);CHKERRQ(ierr);
#endif
  for (k=0;k<nnsp_size;k++) {
    ierr = VecDestroy(&localnearnullsp[k]);CHKERRQ(ierr);
  }
  ierr = PetscFree(localnearnullsp);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
/* -------------------------------------------------------------------------- */
#undef __FUNCT__  
#define __FUNCT__ "PCBDDCCoarseSetUp"
static PetscErrorCode PCBDDCCoarseSetUp(PC pc)
{   
  PetscErrorCode  ierr;

  PC_IS*            pcis = (PC_IS*)(pc->data);
  PC_BDDC*          pcbddc = (PC_BDDC*)pc->data;
  Mat_IS            *matis = (Mat_IS*)pc->pmat->data;
  Mat               change_mat_all;
  IS                is_R_local;
  IS                is_V_local;
  IS                is_C_local;
  IS                is_aux1;
  IS                is_aux2;
  VecType           impVecType;
  MatType           impMatType;
  PetscInt          n_R=0;
  PetscInt          n_D=0;
  PetscInt          n_B=0;
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
  PetscInt          i,j,k;
  /* for verbose output of bddc */
  PetscViewer       viewer=pcbddc->dbg_viewer;
  PetscBool         dbg_flag=pcbddc->dbg_flag;
  /* for counting coarse dofs */
  PetscInt          n_vertices,n_constraints;
  PetscInt          size_of_constraint;
  PetscInt          *row_cmat_indices;
  PetscScalar       *row_cmat_values;
  PetscInt          *vertices,*nnz,*is_indices,*temp_indices;

  PetscFunctionBegin;
  /* Set Non-overlapping dimensions */
  n_B = pcis->n_B; n_D = pcis->n - n_B;
  /* Set types for local objects needed by BDDC precondtioner */
  impMatType = MATSEQDENSE;
  impVecType = VECSEQ;
  /* get vertex indices from constraint matrix */
  ierr = PetscMalloc(pcbddc->local_primal_size*sizeof(PetscInt),&vertices);CHKERRQ(ierr);
  n_vertices=0;
  for (i=0;i<pcbddc->local_primal_size;i++) {
    ierr = MatGetRow(pcbddc->ConstraintMatrix,i,&size_of_constraint,(const PetscInt**)&row_cmat_indices,PETSC_NULL);CHKERRQ(ierr);
    if (size_of_constraint == 1) {
      vertices[n_vertices]=row_cmat_indices[0];
      n_vertices++;
    }
    ierr = MatRestoreRow(pcbddc->ConstraintMatrix,i,&size_of_constraint,(const PetscInt**)&row_cmat_indices,PETSC_NULL);CHKERRQ(ierr);
  }
  /* Set number of constraints */
  n_constraints = pcbddc->local_primal_size-n_vertices;

  /* vertices in boundary numbering */
  if (n_vertices) {
    ierr = VecSet(pcis->vec1_N,m_one);CHKERRQ(ierr);
    ierr = VecGetArray(pcis->vec1_N,&array);CHKERRQ(ierr);
    for (i=0; i<n_vertices; i++) { array[ vertices[i] ] = i; }
    ierr = VecRestoreArray(pcis->vec1_N,&array);CHKERRQ(ierr);
    ierr = VecScatterBegin(pcis->N_to_B,pcis->vec1_N,pcis->vec1_B,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterEnd  (pcis->N_to_B,pcis->vec1_N,pcis->vec1_B,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = PetscMalloc(n_vertices*sizeof(PetscInt),&idx_V_B);CHKERRQ(ierr);
    ierr = VecGetArray(pcis->vec1_B,&array);CHKERRQ(ierr);
    for (i=0; i<n_vertices; i++) {
      j=0;
      while (array[j] != i ) {j++;}
      idx_V_B[i]=j;
    }
    ierr = VecRestoreArray(pcis->vec1_B,&array);CHKERRQ(ierr);
  }

  /* transform local matrices if needed */
  if (pcbddc->usechangeofbasis) {
    ierr = PetscMalloc(pcis->n*sizeof(PetscInt),&nnz);CHKERRQ(ierr);
    ierr = ISGetIndices(pcis->is_I_local,(const PetscInt**)&is_indices);CHKERRQ(ierr);
    for (i=0;i<n_D;i++) {
      nnz[is_indices[i]]=1;
    }
    ierr = ISRestoreIndices(pcis->is_I_local,(const PetscInt**)&is_indices);CHKERRQ(ierr);
    ierr = ISGetIndices(pcis->is_B_local,(const PetscInt**)&is_indices);CHKERRQ(ierr);
    k=1;
    for (i=0;i<n_B;i++) {
      ierr = MatGetRow(pcbddc->ChangeOfBasisMatrix,i,&j,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
      nnz[is_indices[i]]=j;
      if ( k < j) {
        k = j;
      }
      ierr = MatRestoreRow(pcbddc->ChangeOfBasisMatrix,i,&j,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
    }
    ierr = ISRestoreIndices(pcis->is_B_local,(const PetscInt**)&is_indices);CHKERRQ(ierr);
    /* assemble change of basis matrix on the whole set of local dofs */
    ierr = PetscMalloc(k*sizeof(PetscInt),&temp_indices);CHKERRQ(ierr);
    ierr = MatCreate(PETSC_COMM_SELF,&change_mat_all);CHKERRQ(ierr);
    ierr = MatSetSizes(change_mat_all,pcis->n,pcis->n,pcis->n,pcis->n);CHKERRQ(ierr);
    ierr = MatSetType(change_mat_all,MATSEQAIJ);CHKERRQ(ierr);
    ierr = MatSeqAIJSetPreallocation(change_mat_all,0,nnz);CHKERRQ(ierr);
    ierr = ISGetIndices(pcis->is_I_local,(const PetscInt**)&is_indices);CHKERRQ(ierr);
    for (i=0;i<n_D;i++) {
      ierr = MatSetValue(change_mat_all,is_indices[i],is_indices[i],1.0,INSERT_VALUES);CHKERRQ(ierr);
    }
    ierr = ISRestoreIndices(pcis->is_I_local,(const PetscInt**)&is_indices);CHKERRQ(ierr);
    ierr = ISGetIndices(pcis->is_B_local,(const PetscInt**)&is_indices);CHKERRQ(ierr);
    for (i=0;i<n_B;i++) {
      ierr = MatGetRow(pcbddc->ChangeOfBasisMatrix,i,&j,(const PetscInt**)&row_cmat_indices,(const PetscScalar**)&row_cmat_values);CHKERRQ(ierr);
      for (k=0;k<j;k++) {
        temp_indices[k]=is_indices[row_cmat_indices[k]];
      }
      ierr = MatSetValues(change_mat_all,1,&is_indices[i],j,temp_indices,row_cmat_values,INSERT_VALUES);CHKERRQ(ierr);
      ierr = MatRestoreRow(pcbddc->ChangeOfBasisMatrix,i,&j,(const PetscInt**)&row_cmat_indices,(const PetscScalar**)&row_cmat_values);CHKERRQ(ierr);
    }
    ierr = MatAssemblyBegin(change_mat_all,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(change_mat_all,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatPtAP(matis->A,change_mat_all,MAT_INITIAL_MATRIX,1.0,&pcbddc->local_mat);CHKERRQ(ierr);
    ierr = MatDestroy(&pcis->A_IB);CHKERRQ(ierr);
    ierr = MatDestroy(&pcis->A_BI);CHKERRQ(ierr);
    ierr = MatDestroy(&pcis->A_BB);CHKERRQ(ierr);
    ierr = MatGetSubMatrix(pcbddc->local_mat,pcis->is_I_local,pcis->is_B_local,MAT_INITIAL_MATRIX,&pcis->A_IB);CHKERRQ(ierr);
    ierr = MatGetSubMatrix(pcbddc->local_mat,pcis->is_B_local,pcis->is_I_local,MAT_INITIAL_MATRIX,&pcis->A_BI);CHKERRQ(ierr);
    ierr = MatGetSubMatrix(pcbddc->local_mat,pcis->is_B_local,pcis->is_B_local,MAT_INITIAL_MATRIX,&pcis->A_BB);CHKERRQ(ierr);
    ierr = MatDestroy(&change_mat_all);CHKERRQ(ierr);
    ierr = PetscFree(nnz);CHKERRQ(ierr);
    ierr = PetscFree(temp_indices);CHKERRQ(ierr);
  } else {
    /* without change of basis, the local matrix is unchanged */
    ierr = PetscObjectReference((PetscObject)matis->A);CHKERRQ(ierr);
    pcbddc->local_mat = matis->A;
  }
  /* Change global null space passed in by the user if change of basis has been performed */
  if (pcbddc->NullSpace && pcbddc->usechangeofbasis) {
    ierr = PCBDDCAdaptNullSpace(pc);CHKERRQ(ierr);
  }

  /* Dohrmann's notation: dofs splitted in R (Remaining: all dofs but the vertices) and V (Vertices) */
  ierr = VecSet(pcis->vec1_N,one);CHKERRQ(ierr);
  ierr = VecGetArray(pcis->vec1_N,&array);CHKERRQ(ierr);
  for (i=0;i<n_vertices;i++) { array[ vertices[i] ] = zero; }
  ierr = PetscMalloc(( pcis->n - n_vertices )*sizeof(PetscInt),&idx_R_local);CHKERRQ(ierr);
  for (i=0, n_R=0; i<pcis->n; i++) { if (array[i] == one) { idx_R_local[n_R] = i; n_R++; } } 
  ierr = VecRestoreArray(pcis->vec1_N,&array);CHKERRQ(ierr);
  if (dbg_flag) {
    ierr = PetscViewerASCIIPrintf(viewer,"--------------------------------------------------\n");CHKERRQ(ierr);
    ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
    ierr = PetscViewerASCIISynchronizedPrintf(viewer,"Subdomain %04d local dimensions\n",PetscGlobalRank);CHKERRQ(ierr);
    ierr = PetscViewerASCIISynchronizedPrintf(viewer,"local_size = %d, dirichlet_size = %d, boundary_size = %d\n",pcis->n,n_D,n_B);CHKERRQ(ierr);
    ierr = PetscViewerASCIISynchronizedPrintf(viewer,"r_size = %d, v_size = %d, constraints = %d, local_primal_size = %d\n",n_R,n_vertices,n_constraints,pcbddc->local_primal_size);CHKERRQ(ierr);
    ierr = PetscViewerASCIISynchronizedPrintf(viewer,"pcbddc->n_vertices = %d, pcbddc->n_constraints = %d\n",pcbddc->n_vertices,pcbddc->n_constraints);CHKERRQ(ierr);
    ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
  }

  /* Allocate needed vectors */
  ierr = VecDuplicate(pcis->vec1_global,&pcbddc->original_rhs);CHKERRQ(ierr);
  ierr = VecDuplicate(pcis->vec1_global,&pcbddc->temp_solution);CHKERRQ(ierr);
  ierr = VecDuplicate(pcis->vec1_D,&pcbddc->vec4_D);CHKERRQ(ierr);
  ierr = VecCreate(PETSC_COMM_SELF,&pcbddc->vec1_R);CHKERRQ(ierr);
  ierr = VecSetSizes(pcbddc->vec1_R,n_R,n_R);CHKERRQ(ierr);
  ierr = VecSetType(pcbddc->vec1_R,impVecType);CHKERRQ(ierr);
  ierr = VecDuplicate(pcbddc->vec1_R,&pcbddc->vec2_R);CHKERRQ(ierr);
  ierr = VecCreate(PETSC_COMM_SELF,&pcbddc->vec1_P);CHKERRQ(ierr);
  ierr = VecSetSizes(pcbddc->vec1_P,pcbddc->local_primal_size,pcbddc->local_primal_size);CHKERRQ(ierr);
  ierr = VecSetType(pcbddc->vec1_P,impVecType);CHKERRQ(ierr);

  /* Creating some index sets needed  */
  /* For submatrices */
  ierr = ISCreateGeneral(PETSC_COMM_SELF,n_R,idx_R_local,PETSC_OWN_POINTER,&is_R_local);CHKERRQ(ierr);
  if (n_vertices)    {
    ierr = ISCreateGeneral(PETSC_COMM_SELF,n_vertices,vertices,PETSC_OWN_POINTER,&is_V_local);CHKERRQ(ierr);
  }
  if (n_constraints) {
    ierr = ISCreateStride(PETSC_COMM_SELF,n_constraints,n_vertices,1,&is_C_local);CHKERRQ(ierr);
  }

  /* For VecScatters pcbddc->R_to_B and (optionally) pcbddc->R_to_D */
  {
    PetscInt   *aux_array1;
    PetscInt   *aux_array2;
    PetscInt   *idx_I_local;

    ierr = PetscMalloc( (pcis->n_B-n_vertices)*sizeof(PetscInt),&aux_array1);CHKERRQ(ierr);
    ierr = PetscMalloc( (pcis->n_B-n_vertices)*sizeof(PetscInt),&aux_array2);CHKERRQ(ierr);

    ierr = ISGetIndices(pcis->is_I_local,(const PetscInt**)&idx_I_local);CHKERRQ(ierr);
    ierr = VecGetArray(pcis->vec1_N,&array);CHKERRQ(ierr);
    for (i=0; i<n_D; i++) { array[idx_I_local[i]] = 0; }
    ierr = ISRestoreIndices(pcis->is_I_local,(const PetscInt**)&idx_I_local);CHKERRQ(ierr);
    for (i=0, j=0; i<n_R; i++) { if ( array[idx_R_local[i]] == one ) { aux_array1[j] = i; j++; } }
    ierr = VecRestoreArray(pcis->vec1_N,&array);CHKERRQ(ierr);
    ierr = ISCreateGeneral(PETSC_COMM_SELF,j,aux_array1,PETSC_COPY_VALUES,&is_aux1);CHKERRQ(ierr);
    ierr = VecScatterBegin(pcis->N_to_B,pcis->vec1_N,pcis->vec1_B,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterEnd  (pcis->N_to_B,pcis->vec1_N,pcis->vec1_B,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecGetArray(pcis->vec1_B,&array);CHKERRQ(ierr);
    for (i=0, j=0; i<n_B; i++) { if ( array[i] == one ) { aux_array2[j] = i; j++; } }
    ierr = VecRestoreArray(pcis->vec1_B,&array);CHKERRQ(ierr);
    ierr = ISCreateGeneral(PETSC_COMM_SELF,j,aux_array2,PETSC_COPY_VALUES,&is_aux2);CHKERRQ(ierr);
    ierr = VecScatterCreate(pcbddc->vec1_R,is_aux1,pcis->vec1_B,is_aux2,&pcbddc->R_to_B);CHKERRQ(ierr);
    ierr = PetscFree(aux_array1);CHKERRQ(ierr);
    ierr = PetscFree(aux_array2);CHKERRQ(ierr);
    ierr = ISDestroy(&is_aux1);CHKERRQ(ierr);
    ierr = ISDestroy(&is_aux2);CHKERRQ(ierr);

    if (pcbddc->prec_type || dbg_flag ) {
      ierr = PetscMalloc(n_D*sizeof(PetscInt),&aux_array1);CHKERRQ(ierr);
      ierr = VecGetArray(pcis->vec1_N,&array);CHKERRQ(ierr);
      for (i=0, j=0; i<n_R; i++) { if (array[idx_R_local[i]] == zero) { aux_array1[j] = i; j++; } }
      ierr = VecRestoreArray(pcis->vec1_N,&array);CHKERRQ(ierr);
      ierr = ISCreateGeneral(PETSC_COMM_SELF,j,aux_array1,PETSC_COPY_VALUES,&is_aux1);CHKERRQ(ierr);
      ierr = VecScatterCreate(pcbddc->vec1_R,is_aux1,pcis->vec1_D,(IS)0,&pcbddc->R_to_D);CHKERRQ(ierr);
      ierr = PetscFree(aux_array1);CHKERRQ(ierr);
      ierr = ISDestroy(&is_aux1);CHKERRQ(ierr);
    }
  }

  /* Creating PC contexts for local Dirichlet and Neumann problems */
  {
    Mat  A_RR;
    PC   pc_temp;
    /* Matrix for Dirichlet problem is A_II -> we already have it from pcis.c code */
    ierr = KSPCreate(PETSC_COMM_SELF,&pcbddc->ksp_D);CHKERRQ(ierr);
    ierr = PetscObjectIncrementTabLevel((PetscObject)pcbddc->ksp_D,(PetscObject)pc,1);CHKERRQ(ierr);
    ierr = KSPSetOperators(pcbddc->ksp_D,pcis->A_II,pcis->A_II,SAME_PRECONDITIONER);CHKERRQ(ierr);
    ierr = KSPSetType(pcbddc->ksp_D,KSPPREONLY);CHKERRQ(ierr);
    ierr = KSPSetOptionsPrefix(pcbddc->ksp_D,"dirichlet_");CHKERRQ(ierr);
    /* default */
    ierr = KSPGetPC(pcbddc->ksp_D,&pc_temp);CHKERRQ(ierr);
    ierr = PCSetType(pc_temp,PCLU);CHKERRQ(ierr);
    /* Allow user's customization */
    ierr = KSPSetFromOptions(pcbddc->ksp_D);CHKERRQ(ierr);
    /* umfpack interface has a bug when matrix dimension is zero */
    if (!n_D) {
      ierr = PCSetType(pc_temp,PCNONE);CHKERRQ(ierr);
    }
    /* Set Up KSP for Dirichlet problem of BDDC */
    ierr = KSPSetUp(pcbddc->ksp_D);CHKERRQ(ierr);
    /* set ksp_D into pcis data */
    ierr = KSPDestroy(&pcis->ksp_D);CHKERRQ(ierr);
    ierr = PetscObjectReference((PetscObject)pcbddc->ksp_D);CHKERRQ(ierr);
    pcis->ksp_D = pcbddc->ksp_D;
    /* Matrix for Neumann problem is A_RR -> we need to create it */
    ierr = MatGetSubMatrix(pcbddc->local_mat,is_R_local,is_R_local,MAT_INITIAL_MATRIX,&A_RR);CHKERRQ(ierr);
    ierr = KSPCreate(PETSC_COMM_SELF,&pcbddc->ksp_R);CHKERRQ(ierr);
    ierr = PetscObjectIncrementTabLevel((PetscObject)pcbddc->ksp_R,(PetscObject)pc,1);CHKERRQ(ierr);
    ierr = KSPSetOperators(pcbddc->ksp_R,A_RR,A_RR,SAME_PRECONDITIONER);CHKERRQ(ierr);
    ierr = KSPSetType(pcbddc->ksp_R,KSPPREONLY);CHKERRQ(ierr);
    ierr = KSPSetOptionsPrefix(pcbddc->ksp_R,"neumann_");CHKERRQ(ierr);
    /* default */
    ierr = KSPGetPC(pcbddc->ksp_R,&pc_temp);CHKERRQ(ierr);
    ierr = PCSetType(pc_temp,PCLU);CHKERRQ(ierr);
    /* Allow user's customization */
    ierr = KSPSetFromOptions(pcbddc->ksp_R);CHKERRQ(ierr);
    /* umfpack interface has a bug when matrix dimension is zero */
    if (!pcis->n) {
      ierr = PCSetType(pc_temp,PCNONE);CHKERRQ(ierr);
    }
    /* Set Up KSP for Neumann problem of BDDC */
    ierr = KSPSetUp(pcbddc->ksp_R);CHKERRQ(ierr);
    /* check Dirichlet and Neumann solvers */
    {
      Vec         temp_vec;
      PetscReal   value;
      PetscMPIInt use_exact,use_exact_reduced;

      ierr = VecDuplicate(pcis->vec1_D,&temp_vec);CHKERRQ(ierr);
      ierr = VecSetRandom(pcis->vec1_D,PETSC_NULL);CHKERRQ(ierr);
      ierr = MatMult(pcis->A_II,pcis->vec1_D,pcis->vec2_D);CHKERRQ(ierr);
      ierr = KSPSolve(pcbddc->ksp_D,pcis->vec2_D,temp_vec);CHKERRQ(ierr);
      ierr = VecAXPY(temp_vec,m_one,pcis->vec1_D);CHKERRQ(ierr);
      ierr = VecNorm(temp_vec,NORM_INFINITY,&value);CHKERRQ(ierr);
      use_exact = 1;
      if (PetscAbsReal(value) > 1.e-4) {
        use_exact = 0;
      }
      ierr = MPI_Allreduce(&use_exact,&use_exact_reduced,1,MPIU_INT,MPI_LAND,((PetscObject)pc)->comm);CHKERRQ(ierr);
      pcbddc->use_exact_dirichlet = (PetscBool) use_exact_reduced;
      ierr = VecDestroy(&temp_vec);CHKERRQ(ierr);
      if (dbg_flag) {
        ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
        ierr = PetscViewerASCIIPrintf(viewer,"--------------------------------------------------\n");CHKERRQ(ierr);
        ierr = PetscViewerASCIIPrintf(viewer,"Checking solution of Dirichlet and Neumann problems\n");CHKERRQ(ierr);
        ierr = PetscViewerASCIISynchronizedPrintf(viewer,"Subdomain %04d infinity error for Dirichlet solve = % 1.14e \n",PetscGlobalRank,value);CHKERRQ(ierr);
        ierr = VecDuplicate(pcbddc->vec1_R,&temp_vec);CHKERRQ(ierr);
        ierr = VecSetRandom(pcbddc->vec1_R,PETSC_NULL);CHKERRQ(ierr);
        ierr = MatMult(A_RR,pcbddc->vec1_R,pcbddc->vec2_R);CHKERRQ(ierr);
        ierr = KSPSolve(pcbddc->ksp_R,pcbddc->vec2_R,temp_vec);CHKERRQ(ierr);
        ierr = VecAXPY(temp_vec,m_one,pcbddc->vec1_R);CHKERRQ(ierr);
        ierr = VecNorm(temp_vec,NORM_INFINITY,&value);CHKERRQ(ierr);
        ierr = VecDestroy(&temp_vec);CHKERRQ(ierr);
        ierr = PetscViewerASCIISynchronizedPrintf(viewer,"Subdomain %04d infinity error for  Neumann  solve = % 1.14e \n",PetscGlobalRank,value);CHKERRQ(ierr);
        ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
      }
    }
    /* free Neumann problem's matrix */
    ierr = MatDestroy(&A_RR);CHKERRQ(ierr);
  }

  /* Assemble all remaining stuff needed to apply BDDC  */
  {
    Mat          A_RV,A_VR,A_VV;
    Mat          M1;
    Mat          C_CR;
    Mat          AUXMAT;
    Vec          vec1_C;
    Vec          vec2_C;
    Vec          vec1_V;
    Vec          vec2_V;
    PetscInt     *nnz;
    PetscInt     *auxindices;
    PetscInt     index;
    PetscScalar* array2;
    MatFactorInfo matinfo;

    /* Allocating some extra storage just to be safe */
    ierr = PetscMalloc (pcis->n*sizeof(PetscInt),&nnz);CHKERRQ(ierr);
    ierr = PetscMalloc (pcis->n*sizeof(PetscInt),&auxindices);CHKERRQ(ierr);
    for (i=0;i<pcis->n;i++) {auxindices[i]=i;}

    /* some work vectors on vertices and/or constraints */
    if (n_vertices) {
      ierr = VecCreate(PETSC_COMM_SELF,&vec1_V);CHKERRQ(ierr);
      ierr = VecSetSizes(vec1_V,n_vertices,n_vertices);CHKERRQ(ierr);
      ierr = VecSetType(vec1_V,impVecType);CHKERRQ(ierr);
      ierr = VecDuplicate(vec1_V,&vec2_V);CHKERRQ(ierr);
    }
    if (n_constraints) {
      ierr = VecCreate(PETSC_COMM_SELF,&vec1_C);CHKERRQ(ierr);
      ierr = VecSetSizes(vec1_C,n_constraints,n_constraints);CHKERRQ(ierr);
      ierr = VecSetType(vec1_C,impVecType);CHKERRQ(ierr);
      ierr = VecDuplicate(vec1_C,&vec2_C);CHKERRQ(ierr);
      ierr = VecDuplicate(vec1_C,&pcbddc->vec1_C);CHKERRQ(ierr);
    }
    /* Precompute stuffs needed for preprocessing and application of BDDC*/
    if (n_constraints) {
      ierr = MatCreate(PETSC_COMM_SELF,&pcbddc->local_auxmat2);CHKERRQ(ierr);
      ierr = MatSetSizes(pcbddc->local_auxmat2,n_R,n_constraints,n_R,n_constraints);CHKERRQ(ierr);
      ierr = MatSetType(pcbddc->local_auxmat2,impMatType);CHKERRQ(ierr);
      ierr = MatSeqDenseSetPreallocation(pcbddc->local_auxmat2,PETSC_NULL);CHKERRQ(ierr);

      /* Create Constraint matrix on R nodes: C_{CR}  */
      ierr = MatGetSubMatrix(pcbddc->ConstraintMatrix,is_C_local,is_R_local,MAT_INITIAL_MATRIX,&C_CR);CHKERRQ(ierr);
      ierr = ISDestroy(&is_C_local);CHKERRQ(ierr);

      /* Assemble local_auxmat2 = - A_{RR}^{-1} C^T_{CR} needed by BDDC application */
      for (i=0;i<n_constraints;i++) {
        ierr = VecSet(pcbddc->vec1_R,zero);CHKERRQ(ierr);
        /* Get row of constraint matrix in R numbering */
        ierr = VecGetArray(pcbddc->vec1_R,&array);CHKERRQ(ierr);
        ierr = MatGetRow(C_CR,i,&size_of_constraint,(const PetscInt**)&row_cmat_indices,(const PetscScalar**)&row_cmat_values);CHKERRQ(ierr);
        for (j=0;j<size_of_constraint;j++) { array[ row_cmat_indices[j] ] = - row_cmat_values[j]; }
        ierr = MatRestoreRow(C_CR,i,&size_of_constraint,(const PetscInt**)&row_cmat_indices,(const PetscScalar**)&row_cmat_values);CHKERRQ(ierr);
        ierr = VecRestoreArray(pcbddc->vec1_R,&array);CHKERRQ(ierr);
        /* Solve for row of constraint matrix in R numbering */
        ierr = KSPSolve(pcbddc->ksp_R,pcbddc->vec1_R,pcbddc->vec2_R);CHKERRQ(ierr);
        /* Set values */
        ierr = VecGetArray(pcbddc->vec2_R,&array);CHKERRQ(ierr);
        ierr = MatSetValues(pcbddc->local_auxmat2,n_R,auxindices,1,&i,array,INSERT_VALUES);CHKERRQ(ierr);
        ierr = VecRestoreArray(pcbddc->vec2_R,&array);CHKERRQ(ierr);
      }
      ierr = MatAssemblyBegin(pcbddc->local_auxmat2,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
      ierr = MatAssemblyEnd(pcbddc->local_auxmat2,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

      /* Assemble AUXMAT = ( LUFactor )( -C_{CR} A_{RR}^{-1} C^T_{CR} )^{-1} */
      ierr = MatMatMult(C_CR,pcbddc->local_auxmat2,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&AUXMAT);CHKERRQ(ierr);
      ierr = MatFactorInfoInitialize(&matinfo);CHKERRQ(ierr);
      ierr = ISCreateStride(PETSC_COMM_SELF,n_constraints,0,1,&is_aux1);CHKERRQ(ierr);
      ierr = MatLUFactor(AUXMAT,is_aux1,is_aux1,&matinfo);CHKERRQ(ierr);
      ierr = ISDestroy(&is_aux1);CHKERRQ(ierr);

      /* Assemble explicitly M1 = ( C_{CR} A_{RR}^{-1} C^T_{CR} )^{-1} needed in preproc  */
      ierr = MatCreate(PETSC_COMM_SELF,&M1);CHKERRQ(ierr);
      ierr = MatSetSizes(M1,n_constraints,n_constraints,n_constraints,n_constraints);CHKERRQ(ierr);
      ierr = MatSetType(M1,impMatType);CHKERRQ(ierr);
      ierr = MatSeqDenseSetPreallocation(M1,PETSC_NULL);CHKERRQ(ierr);
      for (i=0;i<n_constraints;i++) {
        ierr = VecSet(vec1_C,zero);CHKERRQ(ierr);
        ierr = VecSetValue(vec1_C,i,one,INSERT_VALUES);CHKERRQ(ierr);
        ierr = VecAssemblyBegin(vec1_C);CHKERRQ(ierr);
        ierr = VecAssemblyEnd(vec1_C);CHKERRQ(ierr);
        ierr = MatSolve(AUXMAT,vec1_C,vec2_C);CHKERRQ(ierr);
        ierr = VecScale(vec2_C,m_one);CHKERRQ(ierr);
        ierr = VecGetArray(vec2_C,&array);CHKERRQ(ierr);
        ierr = MatSetValues(M1,n_constraints,auxindices,1,&i,array,INSERT_VALUES);CHKERRQ(ierr);
        ierr = VecRestoreArray(vec2_C,&array);CHKERRQ(ierr);
      }
      ierr = MatAssemblyBegin(M1,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
      ierr = MatAssemblyEnd(M1,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
      ierr = MatDestroy(&AUXMAT);CHKERRQ(ierr);
      /* Assemble local_auxmat1 = M1*C_{CR} needed by BDDC application in KSP and in preproc */
      ierr = MatMatMult(M1,C_CR,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&pcbddc->local_auxmat1);CHKERRQ(ierr);

    }

    /* Get submatrices from subdomain matrix */
    if (n_vertices){
      ierr = MatGetSubMatrix(pcbddc->local_mat,is_R_local,is_V_local,MAT_INITIAL_MATRIX,&A_RV);CHKERRQ(ierr);
      ierr = MatGetSubMatrix(pcbddc->local_mat,is_V_local,is_R_local,MAT_INITIAL_MATRIX,&A_VR);CHKERRQ(ierr);
      ierr = MatGetSubMatrix(pcbddc->local_mat,is_V_local,is_V_local,MAT_INITIAL_MATRIX,&A_VV);CHKERRQ(ierr);
    }

    /* Matrix of coarse basis functions (local) */
    ierr = MatCreate(PETSC_COMM_SELF,&pcbddc->coarse_phi_B);CHKERRQ(ierr);
    ierr = MatSetSizes(pcbddc->coarse_phi_B,n_B,pcbddc->local_primal_size,n_B,pcbddc->local_primal_size);CHKERRQ(ierr);
    ierr = MatSetType(pcbddc->coarse_phi_B,impMatType);CHKERRQ(ierr);
    ierr = MatSeqDenseSetPreallocation(pcbddc->coarse_phi_B,PETSC_NULL);CHKERRQ(ierr);
    if (pcbddc->prec_type || dbg_flag ) {
      ierr = MatCreate(PETSC_COMM_SELF,&pcbddc->coarse_phi_D);CHKERRQ(ierr);
      ierr = MatSetSizes(pcbddc->coarse_phi_D,n_D,pcbddc->local_primal_size,n_D,pcbddc->local_primal_size);CHKERRQ(ierr);
      ierr = MatSetType(pcbddc->coarse_phi_D,impMatType);CHKERRQ(ierr);
      ierr = MatSeqDenseSetPreallocation(pcbddc->coarse_phi_D,PETSC_NULL);CHKERRQ(ierr);
    }

    if (dbg_flag) {
      ierr = PetscMalloc( pcbddc->local_primal_size*sizeof(PetscScalar),&coarsefunctions_errors);CHKERRQ(ierr);
      ierr = PetscMalloc( pcbddc->local_primal_size*sizeof(PetscScalar),&constraints_errors);CHKERRQ(ierr);
    }
    /* Subdomain contribution (Non-overlapping) to coarse matrix  */
    ierr = PetscMalloc ((pcbddc->local_primal_size)*(pcbddc->local_primal_size)*sizeof(PetscScalar),&coarse_submat_vals);CHKERRQ(ierr);

    /* We are now ready to evaluate coarse basis functions and subdomain contribution to coarse problem */
    for (i=0;i<n_vertices;i++){
      ierr = VecSet(vec1_V,zero);CHKERRQ(ierr);
      ierr = VecSetValue(vec1_V,i,one,INSERT_VALUES);CHKERRQ(ierr);
      ierr = VecAssemblyBegin(vec1_V);CHKERRQ(ierr);
      ierr = VecAssemblyEnd(vec1_V);CHKERRQ(ierr);
      /* solution of saddle point problem */
      ierr = MatMult(A_RV,vec1_V,pcbddc->vec1_R);CHKERRQ(ierr);
      ierr = KSPSolve(pcbddc->ksp_R,pcbddc->vec1_R,pcbddc->vec1_R);CHKERRQ(ierr);
      ierr = VecScale(pcbddc->vec1_R,m_one);CHKERRQ(ierr);
      if (n_constraints) {
        ierr = MatMult(pcbddc->local_auxmat1,pcbddc->vec1_R,vec1_C);CHKERRQ(ierr);
        ierr = MatMultAdd(pcbddc->local_auxmat2,vec1_C,pcbddc->vec1_R,pcbddc->vec1_R);CHKERRQ(ierr);
        ierr = VecScale(vec1_C,m_one);CHKERRQ(ierr);
      }
      ierr = MatMult(A_VR,pcbddc->vec1_R,vec2_V);CHKERRQ(ierr);
      ierr = MatMultAdd(A_VV,vec1_V,vec2_V,vec2_V);CHKERRQ(ierr);

      /* Set values in coarse basis function and subdomain part of coarse_mat */
      /* coarse basis functions */
      ierr = VecSet(pcis->vec1_B,zero);CHKERRQ(ierr);
      ierr = VecScatterBegin(pcbddc->R_to_B,pcbddc->vec1_R,pcis->vec1_B,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
      ierr = VecScatterEnd  (pcbddc->R_to_B,pcbddc->vec1_R,pcis->vec1_B,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
      ierr = VecGetArray(pcis->vec1_B,&array);CHKERRQ(ierr);
      ierr = MatSetValues(pcbddc->coarse_phi_B,n_B,auxindices,1,&i,array,INSERT_VALUES);CHKERRQ(ierr);
      ierr = VecRestoreArray(pcis->vec1_B,&array);CHKERRQ(ierr);
      ierr = MatSetValue(pcbddc->coarse_phi_B,idx_V_B[i],i,one,INSERT_VALUES);CHKERRQ(ierr);
      if ( pcbddc->prec_type || dbg_flag  ) {
        ierr = VecScatterBegin(pcbddc->R_to_D,pcbddc->vec1_R,pcis->vec1_D,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
        ierr = VecScatterEnd(pcbddc->R_to_D,pcbddc->vec1_R,pcis->vec1_D,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
        ierr = VecGetArray(pcis->vec1_D,&array);CHKERRQ(ierr);
        ierr = MatSetValues(pcbddc->coarse_phi_D,n_D,auxindices,1,&i,array,INSERT_VALUES);CHKERRQ(ierr);
        ierr = VecRestoreArray(pcis->vec1_D,&array);CHKERRQ(ierr);
      } 
      /* subdomain contribution to coarse matrix */
      ierr = VecGetArray(vec2_V,&array);CHKERRQ(ierr);
      for (j=0;j<n_vertices;j++) { coarse_submat_vals[i*pcbddc->local_primal_size+j] = array[j]; } /* WARNING -> column major ordering */
      ierr = VecRestoreArray(vec2_V,&array);CHKERRQ(ierr);
      if (n_constraints) {
        ierr = VecGetArray(vec1_C,&array);CHKERRQ(ierr);
        for (j=0;j<n_constraints;j++) { coarse_submat_vals[i*pcbddc->local_primal_size+j+n_vertices] = array[j]; } /* WARNING -> column major ordering */
        ierr = VecRestoreArray(vec1_C,&array);CHKERRQ(ierr);
      }
 
      if ( dbg_flag ) {
        /* assemble subdomain vector on nodes */
        ierr = VecSet(pcis->vec1_N,zero);CHKERRQ(ierr);
        ierr = VecGetArray(pcis->vec1_N,&array);CHKERRQ(ierr);
        ierr = VecGetArray(pcbddc->vec1_R,&array2);CHKERRQ(ierr);
        for (j=0;j<n_R;j++) { array[idx_R_local[j]] = array2[j]; }
        array[ vertices[i] ] = one;
        ierr = VecRestoreArray(pcbddc->vec1_R,&array2);CHKERRQ(ierr);
        ierr = VecRestoreArray(pcis->vec1_N,&array);CHKERRQ(ierr);
        /* assemble subdomain vector of lagrange multipliers (i.e. primal nodes) */
        ierr = VecSet(pcbddc->vec1_P,zero);CHKERRQ(ierr);
        ierr = VecGetArray(pcbddc->vec1_P,&array2);CHKERRQ(ierr);
        ierr = VecGetArray(vec2_V,&array);CHKERRQ(ierr);
        for (j=0;j<n_vertices;j++) { array2[j]=array[j]; }
        ierr = VecRestoreArray(vec2_V,&array);CHKERRQ(ierr);
        if (n_constraints) {
          ierr = VecGetArray(vec1_C,&array);CHKERRQ(ierr);
          for (j=0;j<n_constraints;j++) { array2[j+n_vertices]=array[j]; }
          ierr = VecRestoreArray(vec1_C,&array);CHKERRQ(ierr);
        } 
        ierr = VecRestoreArray(pcbddc->vec1_P,&array2);CHKERRQ(ierr);
        ierr = VecScale(pcbddc->vec1_P,m_one);CHKERRQ(ierr);
        /* check saddle point solution */
        ierr = MatMult(pcbddc->local_mat,pcis->vec1_N,pcis->vec2_N);CHKERRQ(ierr);
        ierr = MatMultTransposeAdd(pcbddc->ConstraintMatrix,pcbddc->vec1_P,pcis->vec2_N,pcis->vec2_N);CHKERRQ(ierr);
        ierr = VecNorm(pcis->vec2_N,NORM_INFINITY,&coarsefunctions_errors[i]);CHKERRQ(ierr);
        ierr = MatMult(pcbddc->ConstraintMatrix,pcis->vec1_N,pcbddc->vec1_P);CHKERRQ(ierr);
        ierr = VecGetArray(pcbddc->vec1_P,&array);CHKERRQ(ierr);
        array[i]=array[i]+m_one;  /* shift by the identity matrix */
        ierr = VecRestoreArray(pcbddc->vec1_P,&array);CHKERRQ(ierr);
        ierr = VecNorm(pcbddc->vec1_P,NORM_INFINITY,&constraints_errors[i]);CHKERRQ(ierr);
      }
    }
 
    for (i=0;i<n_constraints;i++){
      ierr = VecSet(vec2_C,zero);CHKERRQ(ierr);
      ierr = VecSetValue(vec2_C,i,m_one,INSERT_VALUES);CHKERRQ(ierr);
      ierr = VecAssemblyBegin(vec2_C);CHKERRQ(ierr);
      ierr = VecAssemblyEnd(vec2_C);CHKERRQ(ierr);
      /* solution of saddle point problem */
      ierr = MatMult(M1,vec2_C,vec1_C);CHKERRQ(ierr);
      ierr = MatMult(pcbddc->local_auxmat2,vec1_C,pcbddc->vec1_R);CHKERRQ(ierr);
      ierr = VecScale(vec1_C,m_one);CHKERRQ(ierr);
      if (n_vertices) { ierr = MatMult(A_VR,pcbddc->vec1_R,vec2_V);CHKERRQ(ierr); }
      /* Set values in coarse basis function and subdomain part of coarse_mat */
      /* coarse basis functions */
      index=i+n_vertices;
      ierr = VecSet(pcis->vec1_B,zero);CHKERRQ(ierr);
      ierr = VecScatterBegin(pcbddc->R_to_B,pcbddc->vec1_R,pcis->vec1_B,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
      ierr = VecScatterEnd  (pcbddc->R_to_B,pcbddc->vec1_R,pcis->vec1_B,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
      ierr = VecGetArray(pcis->vec1_B,&array);CHKERRQ(ierr);
      ierr = MatSetValues(pcbddc->coarse_phi_B,n_B,auxindices,1,&index,array,INSERT_VALUES);CHKERRQ(ierr);
      ierr = VecRestoreArray(pcis->vec1_B,&array);CHKERRQ(ierr);
      if ( pcbddc->prec_type || dbg_flag ) {
        ierr = VecScatterBegin(pcbddc->R_to_D,pcbddc->vec1_R,pcis->vec1_D,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
        ierr = VecScatterEnd(pcbddc->R_to_D,pcbddc->vec1_R,pcis->vec1_D,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
        ierr = VecGetArray(pcis->vec1_D,&array);CHKERRQ(ierr);
        ierr = MatSetValues(pcbddc->coarse_phi_D,n_D,auxindices,1,&index,array,INSERT_VALUES);CHKERRQ(ierr);
        ierr = VecRestoreArray(pcis->vec1_D,&array);CHKERRQ(ierr);
      }
      /* subdomain contribution to coarse matrix */
      if (n_vertices) {
        ierr = VecGetArray(vec2_V,&array);CHKERRQ(ierr);
        for (j=0;j<n_vertices;j++) {coarse_submat_vals[index*pcbddc->local_primal_size+j]=array[j];} /* WARNING -> column major ordering */
        ierr = VecRestoreArray(vec2_V,&array);CHKERRQ(ierr);
      }
      ierr = VecGetArray(vec1_C,&array);CHKERRQ(ierr);
      for (j=0;j<n_constraints;j++) {coarse_submat_vals[index*pcbddc->local_primal_size+j+n_vertices]=array[j];} /* WARNING -> column major ordering */
      ierr = VecRestoreArray(vec1_C,&array);CHKERRQ(ierr);
 
      if ( dbg_flag ) {
        /* assemble subdomain vector on nodes */
        ierr = VecSet(pcis->vec1_N,zero);CHKERRQ(ierr);
        ierr = VecGetArray(pcis->vec1_N,&array);CHKERRQ(ierr);
        ierr = VecGetArray(pcbddc->vec1_R,&array2);CHKERRQ(ierr);
        for (j=0;j<n_R;j++){ array[ idx_R_local[j] ] = array2[j]; }
        ierr = VecRestoreArray(pcbddc->vec1_R,&array2);CHKERRQ(ierr);
        ierr = VecRestoreArray(pcis->vec1_N,&array);CHKERRQ(ierr);
        /* assemble subdomain vector of lagrange multipliers */
        ierr = VecSet(pcbddc->vec1_P,zero);CHKERRQ(ierr);
        ierr = VecGetArray(pcbddc->vec1_P,&array2);CHKERRQ(ierr);
        if ( n_vertices) {
          ierr = VecGetArray(vec2_V,&array);CHKERRQ(ierr);
          for (j=0;j<n_vertices;j++) {array2[j]=-array[j];}
          ierr = VecRestoreArray(vec2_V,&array);CHKERRQ(ierr);
        }
        ierr = VecGetArray(vec1_C,&array);CHKERRQ(ierr);
        for (j=0;j<n_constraints;j++) {array2[j+n_vertices]=-array[j];}
        ierr = VecRestoreArray(vec1_C,&array);CHKERRQ(ierr);
        ierr = VecRestoreArray(pcbddc->vec1_P,&array2);CHKERRQ(ierr);
        /* check saddle point solution */
        ierr = MatMult(pcbddc->local_mat,pcis->vec1_N,pcis->vec2_N);CHKERRQ(ierr);
        ierr = MatMultTransposeAdd(pcbddc->ConstraintMatrix,pcbddc->vec1_P,pcis->vec2_N,pcis->vec2_N);CHKERRQ(ierr);
        ierr = VecNorm(pcis->vec2_N,NORM_INFINITY,&coarsefunctions_errors[index]);CHKERRQ(ierr);
        ierr = MatMult(pcbddc->ConstraintMatrix,pcis->vec1_N,pcbddc->vec1_P);CHKERRQ(ierr);
        ierr = VecGetArray(pcbddc->vec1_P,&array);CHKERRQ(ierr);
        array[index]=array[index]+m_one; /* shift by the identity matrix */
        ierr = VecRestoreArray(pcbddc->vec1_P,&array);CHKERRQ(ierr);
        ierr = VecNorm(pcbddc->vec1_P,NORM_INFINITY,&constraints_errors[index]);CHKERRQ(ierr);
      }
    }
    ierr = MatAssemblyBegin(pcbddc->coarse_phi_B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd  (pcbddc->coarse_phi_B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    if ( pcbddc->prec_type || dbg_flag ) {
      ierr = MatAssemblyBegin(pcbddc->coarse_phi_D,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
      ierr = MatAssemblyEnd  (pcbddc->coarse_phi_D,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    }
    /* Checking coarse_sub_mat and coarse basis functios */
    /* It shuld be \Phi^{(j)^T} A^{(j)} \Phi^{(j)}=coarse_sub_mat */
    if (dbg_flag) {
      Mat         coarse_sub_mat;
      Mat         TM1,TM2,TM3,TM4;
      Mat         coarse_phi_D,coarse_phi_B,A_II,A_BB,A_IB,A_BI;
      MatType     checkmattype=MATSEQAIJ;
      PetscScalar value;

      ierr = MatConvert(pcis->A_II,checkmattype,MAT_INITIAL_MATRIX,&A_II);CHKERRQ(ierr);
      ierr = MatConvert(pcis->A_IB,checkmattype,MAT_INITIAL_MATRIX,&A_IB);CHKERRQ(ierr);
      ierr = MatConvert(pcis->A_BI,checkmattype,MAT_INITIAL_MATRIX,&A_BI);CHKERRQ(ierr);
      ierr = MatConvert(pcis->A_BB,checkmattype,MAT_INITIAL_MATRIX,&A_BB);CHKERRQ(ierr);
      ierr = MatConvert(pcbddc->coarse_phi_D,checkmattype,MAT_INITIAL_MATRIX,&coarse_phi_D);CHKERRQ(ierr);
      ierr = MatConvert(pcbddc->coarse_phi_B,checkmattype,MAT_INITIAL_MATRIX,&coarse_phi_B);CHKERRQ(ierr);
      ierr = MatCreateSeqDense(PETSC_COMM_SELF,pcbddc->local_primal_size,pcbddc->local_primal_size,coarse_submat_vals,&coarse_sub_mat);CHKERRQ(ierr);
      ierr = MatConvert(coarse_sub_mat,checkmattype,MAT_REUSE_MATRIX,&coarse_sub_mat);CHKERRQ(ierr);

      /*PetscViewer view_out;
      PetscMPIInt myrank;
      char filename[256];
      MPI_Comm_rank(((PetscObject)pc)->comm,&myrank);
      sprintf(filename,"coarsesubmat_%04d.m",myrank);
      ierr = PetscViewerASCIIOpen(PETSC_COMM_SELF,filename,&view_out);CHKERRQ(ierr);
      ierr = PetscViewerSetFormat(view_out,PETSC_VIEWER_ASCII_MATLAB);CHKERRQ(ierr);
      ierr = MatView(coarse_sub_mat,view_out);CHKERRQ(ierr);
      ierr = PetscViewerDestroy(&view_out);CHKERRQ(ierr);*/

      ierr = PetscViewerASCIIPrintf(viewer,"--------------------------------------------------\n");CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"Check coarse sub mat and local basis functions\n");CHKERRQ(ierr);
      ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
      ierr = MatPtAP(A_II,coarse_phi_D,MAT_INITIAL_MATRIX,1.0,&TM1);CHKERRQ(ierr);
      ierr = MatPtAP(A_BB,coarse_phi_B,MAT_INITIAL_MATRIX,1.0,&TM2);CHKERRQ(ierr);
      ierr = MatMatMult(A_IB,coarse_phi_B,MAT_INITIAL_MATRIX,1.0,&AUXMAT);CHKERRQ(ierr);
      ierr = MatTransposeMatMult(coarse_phi_D,AUXMAT,MAT_INITIAL_MATRIX,1.0,&TM3);CHKERRQ(ierr);
      ierr = MatDestroy(&AUXMAT);CHKERRQ(ierr);
      ierr = MatMatMult(A_BI,coarse_phi_D,MAT_INITIAL_MATRIX,1.0,&AUXMAT);CHKERRQ(ierr);
      ierr = MatTransposeMatMult(coarse_phi_B,AUXMAT,MAT_INITIAL_MATRIX,1.0,&TM4);CHKERRQ(ierr);
      ierr = MatDestroy(&AUXMAT);CHKERRQ(ierr);
      ierr = MatAXPY(TM1,one,TM2,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
      ierr = MatAXPY(TM1,one,TM3,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
      ierr = MatAXPY(TM1,one,TM4,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
      ierr = MatAXPY(TM1,m_one,coarse_sub_mat,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
      ierr = MatNorm(TM1,NORM_INFINITY,&value);CHKERRQ(ierr);
      ierr = PetscViewerASCIISynchronizedPrintf(viewer,"----------------------------------\n");CHKERRQ(ierr);
      ierr = PetscViewerASCIISynchronizedPrintf(viewer,"Subdomain %04d \n",PetscGlobalRank);CHKERRQ(ierr);
      ierr = PetscViewerASCIISynchronizedPrintf(viewer,"matrix error = % 1.14e\n",value);CHKERRQ(ierr);
      ierr = PetscViewerASCIISynchronizedPrintf(viewer,"coarse functions errors\n");CHKERRQ(ierr);
      for (i=0;i<pcbddc->local_primal_size;i++) { ierr = PetscViewerASCIISynchronizedPrintf(viewer,"local %02d-th function error = % 1.14e\n",i,coarsefunctions_errors[i]);CHKERRQ(ierr); }
      ierr = PetscViewerASCIISynchronizedPrintf(viewer,"constraints errors\n");CHKERRQ(ierr);
      for (i=0;i<pcbddc->local_primal_size;i++) { ierr = PetscViewerASCIISynchronizedPrintf(viewer,"local %02d-th function error = % 1.14e\n",i,constraints_errors[i]);CHKERRQ(ierr); }
      ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
      ierr = MatDestroy(&A_II);CHKERRQ(ierr);
      ierr = MatDestroy(&A_BB);CHKERRQ(ierr);
      ierr = MatDestroy(&A_IB);CHKERRQ(ierr);
      ierr = MatDestroy(&A_BI);CHKERRQ(ierr);
      ierr = MatDestroy(&TM1);CHKERRQ(ierr);
      ierr = MatDestroy(&TM2);CHKERRQ(ierr);
      ierr = MatDestroy(&TM3);CHKERRQ(ierr);
      ierr = MatDestroy(&TM4);CHKERRQ(ierr);
      ierr = MatDestroy(&coarse_phi_D);CHKERRQ(ierr);
      ierr = MatDestroy(&coarse_sub_mat);CHKERRQ(ierr);
      ierr = MatDestroy(&coarse_phi_B);CHKERRQ(ierr);
      ierr = PetscFree(coarsefunctions_errors);CHKERRQ(ierr);
      ierr = PetscFree(constraints_errors);CHKERRQ(ierr);
    }
    /* free memory */ 
    if (n_vertices) {
      ierr = VecDestroy(&vec1_V);CHKERRQ(ierr);
      ierr = VecDestroy(&vec2_V);CHKERRQ(ierr);
      ierr = MatDestroy(&A_RV);CHKERRQ(ierr);
      ierr = MatDestroy(&A_VR);CHKERRQ(ierr);
      ierr = MatDestroy(&A_VV);CHKERRQ(ierr);
    }
    if (n_constraints) {
      ierr = VecDestroy(&vec1_C);CHKERRQ(ierr);
      ierr = VecDestroy(&vec2_C);CHKERRQ(ierr);
      ierr = MatDestroy(&M1);CHKERRQ(ierr);
      ierr = MatDestroy(&C_CR);CHKERRQ(ierr);
    }
    ierr = PetscFree(auxindices);CHKERRQ(ierr);
    ierr = PetscFree(nnz);CHKERRQ(ierr);
    /* create coarse matrix and data structures for message passing associated actual choice of coarse problem type */
    ierr = PCBDDCSetupCoarseEnvironment(pc,coarse_submat_vals);CHKERRQ(ierr);
    ierr = PetscFree(coarse_submat_vals);CHKERRQ(ierr);
  }
  /* free memory */ 
  if (n_vertices) {
    ierr = PetscFree(idx_V_B);CHKERRQ(ierr);
    ierr = ISDestroy(&is_V_local);CHKERRQ(ierr);
  }
  ierr = ISDestroy(&is_R_local);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */

#undef __FUNCT__  
#define __FUNCT__ "PCBDDCSetupCoarseEnvironment"
static PetscErrorCode PCBDDCSetupCoarseEnvironment(PC pc,PetscScalar* coarse_submat_vals)
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
  MatType coarse_mat_type;
  PCType  coarse_pc_type;
  KSPType coarse_ksp_type;
  PC pc_temp;
  PetscInt i,j,k;
  PetscInt max_it_coarse_ksp=1;  /* don't increase this value */
  /* verbose output viewer */
  PetscViewer viewer=pcbddc->dbg_viewer;
  PetscBool   dbg_flag=pcbddc->dbg_flag;

  PetscInt      offset,offset2;
  PetscMPIInt   im_active,active_procs;
  PetscInt      *dnz,*onz;

  PetscBool     setsym,issym=PETSC_FALSE;

  PetscFunctionBegin;
  ins_local_primal_indices = 0;
  ins_coarse_mat_vals      = 0;
  localsizes2              = 0;
  localdispl2              = 0;
  temp_coarse_mat_vals     = 0;
  coarse_ISLG              = 0;

  ierr = MPI_Comm_size(prec_comm,&size_prec_comm);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(prec_comm,&rank_prec_comm);CHKERRQ(ierr);
  ierr = MatIsSymmetricKnown(pc->pmat,&setsym,&issym);CHKERRQ(ierr);
 
  /* Assign global numbering to coarse dofs */
  {
    PetscInt     *auxlocal_primal;
    PetscInt     *row_cmat_indices;
    PetscInt     *aux_ordering;
    PetscInt     *row_cmat_global_indices;
    PetscInt     *dof_sizes,*dof_displs;
    PetscInt     size_of_constraint;
    PetscBool    *array_bool;
    PetscBool    first_found;
    PetscInt     first_index,old_index,s;
    PetscMPIInt  mpi_local_primal_size;
    PetscScalar  coarsesum,*array;

    mpi_local_primal_size = (PetscMPIInt)pcbddc->local_primal_size;

    /* Construct needed data structures for message passing */
    ierr = PetscMalloc(pcbddc->local_primal_size*sizeof(PetscInt),&pcbddc->local_primal_indices);CHKERRQ(ierr);
    j = 0;
    if (rank_prec_comm == 0 || pcbddc->coarse_problem_type == REPLICATED_BDDC || pcbddc->coarse_problem_type == MULTILEVEL_BDDC) {
      j = size_prec_comm;
    } 
    ierr = PetscMalloc(j*sizeof(PetscMPIInt),&pcbddc->local_primal_sizes);CHKERRQ(ierr);
    ierr = PetscMalloc(j*sizeof(PetscMPIInt),&pcbddc->local_primal_displacements);CHKERRQ(ierr);
    /* Gather local_primal_size information for all processes  */
    if (pcbddc->coarse_problem_type == REPLICATED_BDDC || pcbddc->coarse_problem_type == MULTILEVEL_BDDC) {
      ierr = MPI_Allgather(&mpi_local_primal_size,1,MPIU_INT,&pcbddc->local_primal_sizes[0],1,MPIU_INT,prec_comm);CHKERRQ(ierr);
    } else {
      ierr = MPI_Gather(&mpi_local_primal_size,1,MPIU_INT,&pcbddc->local_primal_sizes[0],1,MPIU_INT,0,prec_comm);CHKERRQ(ierr);
    }
    pcbddc->replicated_primal_size = 0;
    for (i=0; i<j; i++) {
      pcbddc->local_primal_displacements[i] = pcbddc->replicated_primal_size ;
      pcbddc->replicated_primal_size += pcbddc->local_primal_sizes[i];
    }

    /* First let's count coarse dofs.
       This code fragment assumes that the number of local constraints per connected component
       is not greater than the number of nodes defined for the connected component 
       (otherwise we will surely have linear dependence between constraints and thus a singular coarse problem) */
    ierr = PetscMalloc(pcbddc->local_primal_size*sizeof(PetscInt),&auxlocal_primal);CHKERRQ(ierr);
    j = 0;
    for (i=0;i<pcbddc->local_primal_size;i++) {
      ierr = MatGetRow(pcbddc->ConstraintMatrix,i,&size_of_constraint,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
      if ( j < size_of_constraint ) {
        j = size_of_constraint;
      }
      ierr = MatRestoreRow(pcbddc->ConstraintMatrix,i,&size_of_constraint,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
    }
    ierr = PetscMalloc(j*sizeof(PetscInt),&aux_ordering);CHKERRQ(ierr);
    ierr = PetscMalloc(j*sizeof(PetscInt),&row_cmat_global_indices);CHKERRQ(ierr);
    ierr = PetscMalloc(pcis->n*sizeof(PetscBool),&array_bool);CHKERRQ(ierr);
    for (i=0;i<pcis->n;i++) {
      array_bool[i] = PETSC_FALSE;
    }
    for (i=0;i<pcbddc->local_primal_size;i++) {
      ierr = MatGetRow(pcbddc->ConstraintMatrix,i,&size_of_constraint,(const PetscInt**)&row_cmat_indices,PETSC_NULL);CHKERRQ(ierr);
      for (j=0; j<size_of_constraint; j++) {
        aux_ordering[j] = j;
      }
      ierr = ISLocalToGlobalMappingApply(matis->mapping,size_of_constraint,row_cmat_indices,row_cmat_global_indices);CHKERRQ(ierr);
      ierr = PetscSortIntWithPermutation(size_of_constraint,row_cmat_global_indices,aux_ordering);CHKERRQ(ierr);
      for (j=0; j<size_of_constraint; j++) {
        k = row_cmat_indices[aux_ordering[j]];
        if ( !array_bool[k] ) {
          array_bool[k] = PETSC_TRUE;
          auxlocal_primal[i] = k;
          break;
        }
      }
      ierr = MatRestoreRow(pcbddc->ConstraintMatrix,i,&size_of_constraint,(const PetscInt**)&row_cmat_indices,PETSC_NULL);CHKERRQ(ierr);
    }
    ierr = PetscFree(aux_ordering);CHKERRQ(ierr);
    ierr = PetscFree(array_bool);CHKERRQ(ierr);
    ierr = PetscFree(row_cmat_global_indices);CHKERRQ(ierr);

    /* Compute number of coarse dofs */
    ierr = VecSet(pcis->vec1_N,0.0);CHKERRQ(ierr);
    ierr = VecGetArray(pcis->vec1_N,&array);CHKERRQ(ierr);
    for (i=0;i<pcbddc->local_primal_size;i++) {
      array[auxlocal_primal[i]]=1.0;
    }
    ierr = VecRestoreArray(pcis->vec1_N,&array);CHKERRQ(ierr);
    ierr = VecSet(pcis->vec1_global,0.0);CHKERRQ(ierr);
    ierr = VecScatterBegin(matis->ctx,pcis->vec1_N,pcis->vec1_global,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
    ierr = VecScatterEnd  (matis->ctx,pcis->vec1_N,pcis->vec1_global,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
    ierr = VecSum(pcis->vec1_global,&coarsesum);CHKERRQ(ierr);
    pcbddc->coarse_size = (PetscInt)coarsesum;
 
    /* Fill pcis->vec1_global with cumulative function for global numbering */
    ierr = VecGetArray(pcis->vec1_global,&array);CHKERRQ(ierr);
    ierr = VecGetLocalSize(pcis->vec1_global,&s);CHKERRQ(ierr);
    k = 0;
    first_index = -1;
    first_found = PETSC_FALSE;
    for (i=0;i<s;i++) {
      if (!first_found && array[i] > 0.0) {
        first_found = PETSC_TRUE;
        first_index = i;
      }
      k += (PetscInt)array[i];
    }
    j = ( !rank_prec_comm ? size_prec_comm : 0);
    ierr = PetscMalloc(j*sizeof(*dof_sizes),&dof_sizes);CHKERRQ(ierr);
    ierr = PetscMalloc(j*sizeof(*dof_displs),&dof_displs);CHKERRQ(ierr);
    ierr = MPI_Gather(&k,1,MPIU_INT,dof_sizes,1,MPIU_INT,0,prec_comm);CHKERRQ(ierr);
    if (!rank_prec_comm) {
      dof_displs[0]=0;
      for (i=1;i<size_prec_comm;i++) {
        dof_displs[i] = dof_displs[i-1]+dof_sizes[i-1];
      }
    }
    ierr = MPI_Scatter(dof_displs,1,MPIU_INT,&k,1,MPIU_INT,0,prec_comm);CHKERRQ(ierr);
    if (first_found) {
      array[first_index] += k;
      old_index = first_index;
      for (i=first_index+1;i<s;i++) {
        if (array[i] > 0.0) {
          array[i] += array[old_index];
          old_index = i;
        }
      }
    }
    ierr = VecRestoreArray(pcis->vec1_global,&array);CHKERRQ(ierr);
    ierr = VecSet(pcis->vec1_N,0.0);CHKERRQ(ierr);
    ierr = VecScatterBegin(matis->ctx,pcis->vec1_global,pcis->vec1_N,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterEnd  (matis->ctx,pcis->vec1_global,pcis->vec1_N,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecGetArray(pcis->vec1_N,&array);CHKERRQ(ierr);
    for (i=0;i<pcbddc->local_primal_size;i++) {
      pcbddc->local_primal_indices[i] = (PetscInt)array[auxlocal_primal[i]]-1;
    }
    ierr = VecRestoreArray(pcis->vec1_N,&array);CHKERRQ(ierr);
    ierr = PetscFree(dof_displs);CHKERRQ(ierr);
    ierr = PetscFree(dof_sizes);CHKERRQ(ierr);

    if (dbg_flag) {
      ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"--------------------------------------------------\n");CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"Check coarse indices\n");CHKERRQ(ierr);
      ierr = VecSet(pcis->vec1_N,0.0);CHKERRQ(ierr);
      ierr = VecGetArray(pcis->vec1_N,&array);CHKERRQ(ierr);
      for (i=0;i<pcbddc->local_primal_size;i++) {
        array[auxlocal_primal[i]]=1.0;
      }
      ierr = VecRestoreArray(pcis->vec1_N,&array);CHKERRQ(ierr);
      ierr = VecSet(pcis->vec1_global,0.0);CHKERRQ(ierr);
      ierr = VecScatterBegin(matis->ctx,pcis->vec1_N,pcis->vec1_global,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
      ierr = VecScatterEnd  (matis->ctx,pcis->vec1_N,pcis->vec1_global,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
      ierr = VecScatterBegin(matis->ctx,pcis->vec1_global,pcis->vec1_N,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
      ierr = VecScatterEnd  (matis->ctx,pcis->vec1_global,pcis->vec1_N,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
      ierr = VecGetArray(pcis->vec1_N,&array);CHKERRQ(ierr);
      for (i=0;i<pcis->n;i++) { 
        if (array[i] == 1.0) {
          ierr = ISLocalToGlobalMappingApply(matis->mapping,1,&i,&j);CHKERRQ(ierr);
          ierr = PetscViewerASCIISynchronizedPrintf(viewer,"Subdomain %04d: WRONG COARSE INDEX %d (local %d)\n",PetscGlobalRank,j,i);CHKERRQ(ierr);
        }
      }
      ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
      for (i=0;i<pcis->n;i++) {
        if( array[i] > 0.0) {
          array[i] = 1.0/array[i];
        }
      }
      ierr = VecRestoreArray(pcis->vec1_N,&array);CHKERRQ(ierr);
      ierr = VecSet(pcis->vec1_global,0.0);CHKERRQ(ierr);
      ierr = VecScatterBegin(matis->ctx,pcis->vec1_N,pcis->vec1_global,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
      ierr = VecScatterEnd  (matis->ctx,pcis->vec1_N,pcis->vec1_global,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
      ierr = VecSum(pcis->vec1_global,&coarsesum);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"Size of coarse problem SHOULD be %lf\n",coarsesum);CHKERRQ(ierr);
      ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
    }
    ierr = PetscFree(auxlocal_primal);CHKERRQ(ierr);
  }

  if (dbg_flag) {
    ierr = PetscViewerASCIIPrintf(viewer,"Size of coarse problem is %d\n",pcbddc->coarse_size);CHKERRQ(ierr);
    /*ierr = PetscViewerASCIIPrintf(viewer,"Distribution of local primal indices\n");CHKERRQ(ierr);
    ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
    ierr = PetscViewerASCIISynchronizedPrintf(viewer,"Subdomain %04d\n",PetscGlobalRank);CHKERRQ(ierr);
    for (i=0;i<pcbddc->local_primal_size;i++) {
      ierr = PetscViewerASCIISynchronizedPrintf(viewer,"local_primal_indices[%d]=%d \n",i,pcbddc->local_primal_indices[i]);
    }*/
    ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
  }

  im_active = 0;
  if (pcis->n) { im_active = 1; }
  ierr = MPI_Allreduce(&im_active,&active_procs,1,MPIU_INT,MPI_SUM,prec_comm);CHKERRQ(ierr);

  /* adapt coarse problem type */
  if (pcbddc->coarse_problem_type == MULTILEVEL_BDDC) {
    if (pcbddc->current_level < pcbddc->max_levels) {
      if ( (active_procs/pcbddc->coarsening_ratio) < 2 ) {
        if (dbg_flag) {
          ierr = PetscViewerASCIIPrintf(viewer,"Not enough active processes on level %d (active %d,ratio %d). Parallel direct solve for coarse problem\n",pcbddc->current_level,active_procs,pcbddc->coarsening_ratio);CHKERRQ(ierr);
         ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
        }
        pcbddc->coarse_problem_type = PARALLEL_BDDC;
      }
    } else {
      if (dbg_flag) {
        ierr = PetscViewerASCIIPrintf(viewer,"Max number of levels reached. Using parallel direct solve for coarse problem\n",pcbddc->max_levels,active_procs,pcbddc->coarsening_ratio);CHKERRQ(ierr);
        ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
      }
      pcbddc->coarse_problem_type = PARALLEL_BDDC;
    }
  }  

  switch(pcbddc->coarse_problem_type){

    case(MULTILEVEL_BDDC):   /* we define a coarse mesh where subdomains are elements */
    {
      /* we need additional variables */
      MetisInt    n_subdomains,n_parts,objval,ncon,faces_nvtxs;
      MetisInt    *metis_coarse_subdivision;
      MetisInt    options[METIS_NOPTIONS];
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
      PetscInt    ranks_stretching_ratio;

      /* define some quantities */
      pcbddc->coarse_communications_type = SCATTERS_BDDC;
      coarse_mat_type = MATIS;
      coarse_pc_type  = PCBDDC;
      coarse_ksp_type = KSPRICHARDSON;

      /* details of coarse decomposition */
      n_subdomains = active_procs;
      n_parts      = n_subdomains/pcbddc->coarsening_ratio;
      ranks_stretching_ratio = size_prec_comm/active_procs;
      procs_jumps_coarse_comm = pcbddc->coarsening_ratio*ranks_stretching_ratio;

#if 0
      PetscMPIInt *old_ranks;
      PetscInt    *new_ranks,*jj,*ii;
      MatPartitioning mat_part;
      IS coarse_new_decomposition,is_numbering;
      PetscViewer viewer_test;
      MPI_Comm    test_coarse_comm;
      PetscMPIInt test_coarse_color;
      Mat         mat_adj;
      /* Create new communicator for coarse problem splitting the old one */
      /* procs with coarse_color = MPI_UNDEFINED will have coarse_comm = MPI_COMM_NULL (from mpi standards)
         key = rank_prec_comm -> keep same ordering of ranks from the old to the new communicator */
      test_coarse_color = ( im_active ? 0 : MPI_UNDEFINED );
      test_coarse_comm = MPI_COMM_NULL;
      ierr = MPI_Comm_split(prec_comm,test_coarse_color,rank_prec_comm,&test_coarse_comm);CHKERRQ(ierr);
      if (im_active) {
        ierr = PetscMalloc(n_subdomains*sizeof(PetscMPIInt),&old_ranks);
        ierr = PetscMalloc(size_prec_comm*sizeof(PetscInt),&new_ranks);
        ierr = MPI_Comm_rank(test_coarse_comm,&rank_coarse_comm);CHKERRQ(ierr);
        ierr = MPI_Comm_size(test_coarse_comm,&j);CHKERRQ(ierr);
        ierr = MPI_Allgather(&rank_prec_comm,1,MPIU_INT,old_ranks,1,MPIU_INT,test_coarse_comm);CHKERRQ(ierr);
        for (i=0;i<size_prec_comm;i++) {
          new_ranks[i] = -1;
        }
        for (i=0;i<n_subdomains;i++) {
          new_ranks[old_ranks[i]] = i;
        }
        ierr = PetscViewerASCIIOpen(test_coarse_comm,"test_mat_part.out",&viewer_test);CHKERRQ(ierr);
        k = pcis->n_neigh-1;
        ierr = PetscMalloc(2*sizeof(PetscInt),&ii);
        ii[0]=0;
        ii[1]=k;
        ierr = PetscMalloc(k*sizeof(PetscInt),&jj);
        for (i=0;i<k;i++) {
          jj[i]=new_ranks[pcis->neigh[i+1]];
        }
        ierr = PetscSortInt(k,jj);CHKERRQ(ierr); 
        ierr = MatCreateMPIAdj(test_coarse_comm,1,n_subdomains,ii,jj,PETSC_NULL,&mat_adj);CHKERRQ(ierr);
        ierr = MatView(mat_adj,viewer_test);CHKERRQ(ierr);
        ierr = MatPartitioningCreate(test_coarse_comm,&mat_part);CHKERRQ(ierr);
        ierr = MatPartitioningSetAdjacency(mat_part,mat_adj);CHKERRQ(ierr);
        ierr = MatPartitioningSetFromOptions(mat_part);CHKERRQ(ierr);
        printf("Setting Nparts %d\n",n_parts);
        ierr = MatPartitioningSetNParts(mat_part,n_parts);CHKERRQ(ierr);
        ierr = MatPartitioningView(mat_part,viewer_test);CHKERRQ(ierr);
        ierr = MatPartitioningApply(mat_part,&coarse_new_decomposition);CHKERRQ(ierr);
        ierr = ISView(coarse_new_decomposition,viewer_test);CHKERRQ(ierr);
        ierr = ISPartitioningToNumbering(coarse_new_decomposition,&is_numbering);CHKERRQ(ierr);
        ierr = ISView(is_numbering,viewer_test);CHKERRQ(ierr);
        ierr = PetscViewerDestroy(&viewer_test);CHKERRQ(ierr);
        ierr = ISDestroy(&coarse_new_decomposition);CHKERRQ(ierr);
        ierr = ISDestroy(&is_numbering);CHKERRQ(ierr);
        ierr = MatPartitioningDestroy(&mat_part);CHKERRQ(ierr);
        ierr = PetscFree(old_ranks);CHKERRQ(ierr);
        ierr = PetscFree(new_ranks);CHKERRQ(ierr);
        ierr = MPI_Comm_free(&test_coarse_comm);CHKERRQ(ierr);
      }
#endif

      /* build CSR graph of subdomains' connectivity */
      ierr = PetscMalloc (pcis->n*sizeof(PetscInt),&array_int);CHKERRQ(ierr);
      ierr = PetscMemzero(array_int,pcis->n*sizeof(PetscInt));CHKERRQ(ierr);
      for (i=1;i<pcis->n_neigh;i++){/* i=1 so I don't count myself -> faces nodes counts to 1 */
        for (j=0;j<pcis->n_shared[i];j++){
          array_int[ pcis->shared[i][j] ]+=1;
        }
      }
      for (i=1;i<pcis->n_neigh;i++){
        for (j=0;j<pcis->n_shared[i];j++){
          if (array_int[ pcis->shared[i][j] ] > 0 ){
            my_faces++;
            break;
          }
        }
      }

      ierr = MPI_Reduce(&my_faces,&total_faces,1,MPIU_INT,MPI_SUM,master_proc,prec_comm);CHKERRQ(ierr);
      ierr = PetscMalloc (my_faces*sizeof(PetscInt),&my_faces_connectivity);CHKERRQ(ierr);
      my_faces=0;
      for (i=1;i<pcis->n_neigh;i++){
        for (j=0;j<pcis->n_shared[i];j++){
          if (array_int[ pcis->shared[i][j] ] > 0 ){
            my_faces_connectivity[my_faces]=pcis->neigh[i];
            my_faces++;
            break;
          }
        }
      }
      if (rank_prec_comm == master_proc) {
        ierr = PetscMalloc (total_faces*sizeof(PetscMPIInt),&petsc_faces_adjncy);CHKERRQ(ierr);
        ierr = PetscMalloc (size_prec_comm*sizeof(PetscMPIInt),&number_of_faces);CHKERRQ(ierr);
        ierr = PetscMalloc (total_faces*sizeof(MetisInt),&faces_adjncy);CHKERRQ(ierr);
        ierr = PetscMalloc ((n_subdomains+1)*sizeof(MetisInt),&faces_xadj);CHKERRQ(ierr);
        ierr = PetscMalloc ((size_prec_comm+1)*sizeof(PetscMPIInt),&faces_displacements);CHKERRQ(ierr);
      }
      ierr = MPI_Gather(&my_faces,1,MPIU_INT,&number_of_faces[0],1,MPIU_INT,master_proc,prec_comm);CHKERRQ(ierr);
      if (rank_prec_comm == master_proc) {
        faces_xadj[0]=0;
        faces_displacements[0]=0;
        j=0;
        for (i=1;i<size_prec_comm+1;i++) {
          faces_displacements[i]=faces_displacements[i-1]+number_of_faces[i-1];
          if (number_of_faces[i-1]) {
            j++;
            faces_xadj[j]=faces_xadj[j-1]+number_of_faces[i-1];
          }
        }
      }
      ierr = MPI_Gatherv(&my_faces_connectivity[0],my_faces,MPIU_INT,&petsc_faces_adjncy[0],number_of_faces,faces_displacements,MPIU_INT,master_proc,prec_comm);CHKERRQ(ierr);
      ierr = PetscFree(my_faces_connectivity);CHKERRQ(ierr);
      ierr = PetscFree(array_int);CHKERRQ(ierr);
      if (rank_prec_comm == master_proc) {
        for (i=0;i<total_faces;i++) faces_adjncy[i]=(MetisInt)(petsc_faces_adjncy[i]/ranks_stretching_ratio); /* cast to MetisInt */
        /*printf("This is the face connectivity (actual ranks)\n");
        for (i=0;i<n_subdomains;i++){
          printf("proc %d is connected with \n",i);
          for (j=faces_xadj[i];j<faces_xadj[i+1];j++)
            printf("%d ",faces_adjncy[j]);
          printf("\n");
        }*/
        ierr = PetscFree(faces_displacements);CHKERRQ(ierr);
        ierr = PetscFree(number_of_faces);CHKERRQ(ierr);
        ierr = PetscFree(petsc_faces_adjncy);CHKERRQ(ierr);
      }

      if ( rank_prec_comm == master_proc ) {

        PetscInt heuristic_for_metis=3;

        ncon=1;
        faces_nvtxs=n_subdomains;
        /* partition graoh induced by face connectivity */
        ierr = PetscMalloc (n_subdomains*sizeof(MetisInt),&metis_coarse_subdivision);CHKERRQ(ierr);
        ierr = METIS_SetDefaultOptions(options);
        /* we need a contiguous partition of the coarse mesh */
        options[METIS_OPTION_CONTIG]=1;
        options[METIS_OPTION_NITER]=30;
        if (pcbddc->coarsening_ratio > 1) {
          if (n_subdomains>n_parts*heuristic_for_metis) {
            options[METIS_OPTION_IPTYPE]=METIS_IPTYPE_EDGE;
            options[METIS_OPTION_OBJTYPE]=METIS_OBJTYPE_CUT; 
            ierr = METIS_PartGraphKway(&faces_nvtxs,&ncon,faces_xadj,faces_adjncy,NULL,NULL,NULL,&n_parts,NULL,NULL,options,&objval,metis_coarse_subdivision);
            if (ierr != METIS_OK) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in METIS_PartGraphKway (metis error code %D) called from PCBDDCSetupCoarseEnvironment\n",ierr);
          } else {
            ierr = METIS_PartGraphRecursive(&faces_nvtxs,&ncon,faces_xadj,faces_adjncy,NULL,NULL,NULL,&n_parts,NULL,NULL,options,&objval,metis_coarse_subdivision);
            if (ierr != METIS_OK) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in METIS_PartGraphRecursive (metis error code %D) called from PCBDDCSetupCoarseEnvironment\n",ierr);
          }
        } else {
          for (i=0;i<n_subdomains;i++) {
            metis_coarse_subdivision[i]=i;
          }
        }
        ierr = PetscFree(faces_xadj);CHKERRQ(ierr);
        ierr = PetscFree(faces_adjncy);CHKERRQ(ierr);
        ierr = PetscMalloc(size_prec_comm*sizeof(PetscMPIInt),&coarse_subdivision);CHKERRQ(ierr);
        /* copy/cast values avoiding possible type conflicts between PETSc, MPI and METIS */
        for (i=0;i<size_prec_comm;i++) { coarse_subdivision[i]=MPI_PROC_NULL; }
        for (i=0;i<n_subdomains;i++)   { coarse_subdivision[ranks_stretching_ratio*i]=(PetscInt)(metis_coarse_subdivision[i]); }
        ierr = PetscFree(metis_coarse_subdivision);CHKERRQ(ierr);
      }

      /* Create new communicator for coarse problem splitting the old one */
      if ( !(rank_prec_comm%procs_jumps_coarse_comm) && rank_prec_comm < procs_jumps_coarse_comm*n_parts ){
        coarse_color=0;              /* for communicator splitting */
        active_rank=rank_prec_comm;  /* for insertion of matrix values */
      }
      /* procs with coarse_color = MPI_UNDEFINED will have coarse_comm = MPI_COMM_NULL (from mpi standards)
         key = rank_prec_comm -> keep same ordering of ranks from the old to the new communicator */
      ierr = MPI_Comm_split(prec_comm,coarse_color,rank_prec_comm,&coarse_comm);CHKERRQ(ierr);

      if ( coarse_color == 0 ) {
        ierr = MPI_Comm_size(coarse_comm,&size_coarse_comm);CHKERRQ(ierr);
        ierr = MPI_Comm_rank(coarse_comm,&rank_coarse_comm);CHKERRQ(ierr);
      } else {
        rank_coarse_comm = MPI_PROC_NULL;
      }

      /* master proc take care of arranging and distributing coarse information */
      if (rank_coarse_comm == master_proc) {
        ierr = PetscMalloc (size_coarse_comm*sizeof(PetscMPIInt),&displacements_recv);CHKERRQ(ierr);
        ierr = PetscMalloc (size_coarse_comm*sizeof(PetscMPIInt),&total_count_recv);CHKERRQ(ierr);
        ierr = PetscMalloc (n_subdomains*sizeof(PetscMPIInt),&total_ranks_recv);CHKERRQ(ierr);
        /* some initializations */
        displacements_recv[0]=0;
        ierr = PetscMemzero(total_count_recv,size_coarse_comm*sizeof(PetscMPIInt));CHKERRQ(ierr);
        /* count from how many processes the j-th process of the coarse decomposition will receive data */
        for (j=0;j<size_coarse_comm;j++) {
          for (i=0;i<size_prec_comm;i++) {
            if (coarse_subdivision[i]==j) { 
              total_count_recv[j]++;
            }
          }
        }
        /* displacements needed for scatterv of total_ranks_recv */
        for (i=1;i<size_coarse_comm;i++) { displacements_recv[i]=displacements_recv[i-1]+total_count_recv[i-1]; }
        /* Now fill properly total_ranks_recv -> each coarse process will receive the ranks (in prec_comm communicator) of its friend (sending) processes */
        ierr = PetscMemzero(total_count_recv,size_coarse_comm*sizeof(PetscMPIInt));CHKERRQ(ierr);
        for (j=0;j<size_coarse_comm;j++) {
          for (i=0;i<size_prec_comm;i++) {
            if (coarse_subdivision[i]==j) {
              total_ranks_recv[displacements_recv[j]+total_count_recv[j]]=i;
              total_count_recv[j]+=1;
            }
          }
        }
        /*for (j=0;j<size_coarse_comm;j++) {
          printf("process %d in new rank will receive from %d processes (original ranks follows)\n",j,total_count_recv[j]);
          for (i=0;i<total_count_recv[j];i++) {
            printf("%d ",total_ranks_recv[displacements_recv[j]+i]);
          }
          printf("\n");
        }*/

        /* identify new decomposition in terms of ranks in the old communicator */
        for (i=0;i<n_subdomains;i++) { 
          coarse_subdivision[ranks_stretching_ratio*i]=coarse_subdivision[ranks_stretching_ratio*i]*procs_jumps_coarse_comm;
        }
        /*printf("coarse_subdivision in old end new ranks\n");
        for (i=0;i<size_prec_comm;i++)
          if (coarse_subdivision[i]!=MPI_PROC_NULL) { 
            printf("%d=(%d %d), ",i,coarse_subdivision[i],coarse_subdivision[i]/procs_jumps_coarse_comm);
          } else {
            printf("%d=(%d %d), ",i,coarse_subdivision[i],coarse_subdivision[i]);
          }
        printf("\n");*/
      }

      /* Scatter new decomposition for send details */
      ierr = MPI_Scatter(&coarse_subdivision[0],1,MPIU_INT,&rank_coarse_proc_send_to,1,MPIU_INT,master_proc,prec_comm);CHKERRQ(ierr);
      /* Scatter receiving details to members of coarse decomposition */
      if ( coarse_color == 0) {
        ierr = MPI_Scatter(&total_count_recv[0],1,MPIU_INT,&count_recv,1,MPIU_INT,master_proc,coarse_comm);CHKERRQ(ierr);
        ierr = PetscMalloc (count_recv*sizeof(PetscMPIInt),&ranks_recv);CHKERRQ(ierr);
        ierr = MPI_Scatterv(&total_ranks_recv[0],total_count_recv,displacements_recv,MPIU_INT,&ranks_recv[0],count_recv,MPIU_INT,master_proc,coarse_comm);CHKERRQ(ierr);
      }

      /*printf("I will send my matrix data to proc  %d\n",rank_coarse_proc_send_to);
      if (coarse_color == 0) {
        printf("I will receive some matrix data from %d processes (ranks follows)\n",count_recv);
        for (i=0;i<count_recv;i++)
          printf("%d ",ranks_recv[i]);
        printf("\n");
      }*/

      if (rank_prec_comm == master_proc) {
        ierr = PetscFree(coarse_subdivision);CHKERRQ(ierr);
        ierr = PetscFree(total_count_recv);CHKERRQ(ierr);
        ierr = PetscFree(total_ranks_recv);CHKERRQ(ierr);
        ierr = PetscFree(displacements_recv);CHKERRQ(ierr);
      }
      break;
    }

    case(REPLICATED_BDDC):

      pcbddc->coarse_communications_type = GATHERS_BDDC;
      coarse_mat_type = MATSEQAIJ;
      coarse_pc_type  = PCLU;
      coarse_ksp_type  = KSPPREONLY;
      coarse_comm = PETSC_COMM_SELF;
      active_rank = rank_prec_comm;
      break;

    case(PARALLEL_BDDC):

      pcbddc->coarse_communications_type = SCATTERS_BDDC;
      coarse_mat_type = MATMPIAIJ;
      coarse_pc_type  = PCREDUNDANT;
      coarse_ksp_type  = KSPPREONLY;
      coarse_comm = prec_comm;
      active_rank = rank_prec_comm;
      break;

    case(SEQUENTIAL_BDDC):
      pcbddc->coarse_communications_type = GATHERS_BDDC;
      coarse_mat_type = MATSEQAIJ;
      coarse_pc_type = PCLU;
      coarse_ksp_type  = KSPPREONLY;
      coarse_comm = PETSC_COMM_SELF;
      active_rank = master_proc;
      break;
  }

  switch(pcbddc->coarse_communications_type){

    case(SCATTERS_BDDC):
      {
        if (pcbddc->coarse_problem_type==MULTILEVEL_BDDC) {

          IS coarse_IS;

          if(pcbddc->coarsening_ratio == 1) {
            ins_local_primal_size = pcbddc->local_primal_size;
            ins_local_primal_indices = pcbddc->local_primal_indices;
            if (coarse_color == 0) { ierr = PetscFree(ranks_recv);CHKERRQ(ierr); }
            /* nonzeros */
            ierr = PetscMalloc(ins_local_primal_size*sizeof(PetscInt),&dnz);CHKERRQ(ierr);
            ierr = PetscMemzero(dnz,ins_local_primal_size*sizeof(PetscInt));CHKERRQ(ierr);
            for (i=0;i<ins_local_primal_size;i++) {
              dnz[i] = ins_local_primal_size;
            }
          } else {
            PetscMPIInt send_size;
            PetscMPIInt *send_buffer;
            PetscInt    *aux_ins_indices;
            PetscInt    ii,jj;
            MPI_Request *requests;

            ierr = PetscMalloc(count_recv*sizeof(PetscMPIInt),&localdispl2);CHKERRQ(ierr);
            /* reusing pcbddc->local_primal_displacements and pcbddc->replicated_primal_size */
            ierr = PetscFree(pcbddc->local_primal_displacements);CHKERRQ(ierr);
            ierr = PetscMalloc((count_recv+1)*sizeof(PetscMPIInt),&pcbddc->local_primal_displacements);CHKERRQ(ierr);
            pcbddc->replicated_primal_size = count_recv;
            j = 0;
            for (i=0;i<count_recv;i++) {
              pcbddc->local_primal_displacements[i] = j;
              j += pcbddc->local_primal_sizes[ranks_recv[i]];
            }
            pcbddc->local_primal_displacements[count_recv] = j;
            ierr = PetscMalloc(j*sizeof(PetscMPIInt),&pcbddc->replicated_local_primal_indices);CHKERRQ(ierr);
            /* allocate auxiliary space */
            ierr = PetscMalloc(count_recv*sizeof(PetscMPIInt),&localsizes2);CHKERRQ(ierr);
            ierr = PetscMalloc(pcbddc->coarse_size*sizeof(PetscInt),&aux_ins_indices);CHKERRQ(ierr);
            ierr = PetscMemzero(aux_ins_indices,pcbddc->coarse_size*sizeof(PetscInt));CHKERRQ(ierr);
            /* allocate stuffs for message massing */
            ierr = PetscMalloc((count_recv+1)*sizeof(MPI_Request),&requests);CHKERRQ(ierr);
            for (i=0;i<count_recv+1;i++) { requests[i]=MPI_REQUEST_NULL; }
            /* send indices to be inserted */
            for (i=0;i<count_recv;i++) {
              send_size = pcbddc->local_primal_sizes[ranks_recv[i]];
              ierr = MPI_Irecv(&pcbddc->replicated_local_primal_indices[pcbddc->local_primal_displacements[i]],send_size,MPIU_INT,ranks_recv[i],999,prec_comm,&requests[i]);CHKERRQ(ierr);
            }
            if (rank_coarse_proc_send_to != MPI_PROC_NULL ) {
              send_size = pcbddc->local_primal_size;
              ierr = PetscMalloc(send_size*sizeof(PetscMPIInt),&send_buffer);CHKERRQ(ierr);
              for (i=0;i<send_size;i++) {
                send_buffer[i]=(PetscMPIInt)pcbddc->local_primal_indices[i];
              }
              ierr = MPI_Isend(send_buffer,send_size,MPIU_INT,rank_coarse_proc_send_to,999,prec_comm,&requests[count_recv]);CHKERRQ(ierr);
            }
            ierr = MPI_Waitall(count_recv+1,requests,MPI_STATUSES_IGNORE);CHKERRQ(ierr);
            if (rank_coarse_proc_send_to != MPI_PROC_NULL ) { 
              ierr = PetscFree(send_buffer);CHKERRQ(ierr);
            }
            j = 0;
            for (i=0;i<count_recv;i++) {
              ii = pcbddc->local_primal_displacements[i+1]-pcbddc->local_primal_displacements[i];
              localsizes2[i] = ii*ii;
              localdispl2[i] = j;
              j += localsizes2[i];
              jj = pcbddc->local_primal_displacements[i];
              /* it counts the coarse subdomains sharing the coarse node */
              for (k=0;k<ii;k++) {
                aux_ins_indices[pcbddc->replicated_local_primal_indices[jj+k]] += 1;
              }
            }
            /* temp_coarse_mat_vals used to store matrix values to be received */
            ierr = PetscMalloc(j*sizeof(PetscScalar),&temp_coarse_mat_vals);CHKERRQ(ierr);
            /* evaluate how many values I will insert in coarse mat */
            ins_local_primal_size = 0;
            for (i=0;i<pcbddc->coarse_size;i++) {
              if (aux_ins_indices[i]) {
                ins_local_primal_size++;
              }
            }
            /* evaluate indices I will insert in coarse mat */
            ierr = PetscMalloc(ins_local_primal_size*sizeof(PetscInt),&ins_local_primal_indices);CHKERRQ(ierr);
            j = 0;
            for(i=0;i<pcbddc->coarse_size;i++) {
              if(aux_ins_indices[i]) {
                ins_local_primal_indices[j] = i;
                j++;
              }
            }
            /* processes partecipating in coarse problem receive matrix data from their friends */
            for (i=0;i<count_recv;i++) { 
              ierr = MPI_Irecv(&temp_coarse_mat_vals[localdispl2[i]],localsizes2[i],MPIU_SCALAR,ranks_recv[i],666,prec_comm,&requests[i]);CHKERRQ(ierr);
            }
            if (rank_coarse_proc_send_to != MPI_PROC_NULL ) {
              send_size = pcbddc->local_primal_size*pcbddc->local_primal_size;
              ierr = MPI_Isend(&coarse_submat_vals[0],send_size,MPIU_SCALAR,rank_coarse_proc_send_to,666,prec_comm,&requests[count_recv]);CHKERRQ(ierr);
            }
            ierr = MPI_Waitall(count_recv+1,requests,MPI_STATUSES_IGNORE);CHKERRQ(ierr);
            /* nonzeros */
            ierr = PetscMalloc(ins_local_primal_size*sizeof(PetscInt),&dnz);CHKERRQ(ierr);
            ierr = PetscMemzero(dnz,ins_local_primal_size*sizeof(PetscInt));CHKERRQ(ierr);
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
            for (i=0;i<count_recv;i++) {
              j = pcbddc->local_primal_sizes[ranks_recv[i]];
              for (k=0;k<j;k++) {
                dnz[aux_ins_indices[pcbddc->replicated_local_primal_indices[pcbddc->local_primal_displacements[i]+k]]] += j;
              }
            }
            /* check */
            for (i=0;i<ins_local_primal_size;i++) {
              if (dnz[i] > ins_local_primal_size) {
                dnz[i] = ins_local_primal_size;
              }
            }
            ierr = PetscFree(requests);CHKERRQ(ierr);
            ierr = PetscFree(aux_ins_indices);CHKERRQ(ierr);
            if (coarse_color == 0) { ierr = PetscFree(ranks_recv);CHKERRQ(ierr); }
          }
          /* create local to global mapping needed by coarse MATIS */
          if (coarse_comm != MPI_COMM_NULL ) {ierr = MPI_Comm_free(&coarse_comm);CHKERRQ(ierr);}
          coarse_comm = prec_comm;
          active_rank = rank_prec_comm;
          ierr = ISCreateGeneral(coarse_comm,ins_local_primal_size,ins_local_primal_indices,PETSC_COPY_VALUES,&coarse_IS);CHKERRQ(ierr);
          ierr = ISLocalToGlobalMappingCreateIS(coarse_IS,&coarse_ISLG);CHKERRQ(ierr);
          ierr = ISDestroy(&coarse_IS);CHKERRQ(ierr); 
        } else if (pcbddc->coarse_problem_type==PARALLEL_BDDC) {
          /* arrays for values insertion */
          ins_local_primal_size = pcbddc->local_primal_size;
          ierr = PetscMalloc(ins_local_primal_size*sizeof(PetscInt),&ins_local_primal_indices);CHKERRQ(ierr);
          ierr = PetscMalloc(ins_local_primal_size*ins_local_primal_size*sizeof(PetscScalar),&ins_coarse_mat_vals);CHKERRQ(ierr);
          for (j=0;j<ins_local_primal_size;j++){
            ins_local_primal_indices[j]=pcbddc->local_primal_indices[j];
            for (i=0;i<ins_local_primal_size;i++) {
              ins_coarse_mat_vals[j*ins_local_primal_size+i]=coarse_submat_vals[j*ins_local_primal_size+i];
            }
          }
        }
        break;
        
    }

    case(GATHERS_BDDC):
      {

        PetscMPIInt mysize,mysize2;
        PetscMPIInt *send_buffer;

        if (rank_prec_comm==active_rank) {
          ierr = PetscMalloc ( pcbddc->replicated_primal_size*sizeof(PetscMPIInt),&pcbddc->replicated_local_primal_indices);CHKERRQ(ierr);
          ierr = PetscMalloc ( pcbddc->replicated_primal_size*sizeof(PetscScalar),&pcbddc->replicated_local_primal_values);CHKERRQ(ierr);
          ierr = PetscMalloc ( size_prec_comm*sizeof(PetscMPIInt),&localsizes2);CHKERRQ(ierr);
          ierr = PetscMalloc ( size_prec_comm*sizeof(PetscMPIInt),&localdispl2);CHKERRQ(ierr);
          /* arrays for values insertion */
          for (i=0;i<size_prec_comm;i++) { localsizes2[i]=pcbddc->local_primal_sizes[i]*pcbddc->local_primal_sizes[i]; }
          localdispl2[0]=0;
          for (i=1;i<size_prec_comm;i++) { localdispl2[i]=localsizes2[i-1]+localdispl2[i-1]; }
          j=0;
          for (i=0;i<size_prec_comm;i++) { j+=localsizes2[i]; }
          ierr = PetscMalloc ( j*sizeof(PetscScalar),&temp_coarse_mat_vals);CHKERRQ(ierr);
        }

        mysize=pcbddc->local_primal_size;
        mysize2=pcbddc->local_primal_size*pcbddc->local_primal_size;
        ierr = PetscMalloc(mysize*sizeof(PetscMPIInt),&send_buffer);CHKERRQ(ierr);
        for (i=0;i<mysize;i++) {
          send_buffer[i]=(PetscMPIInt)pcbddc->local_primal_indices[i];
        }
        if (pcbddc->coarse_problem_type == SEQUENTIAL_BDDC){
          ierr = MPI_Gatherv(send_buffer,mysize,MPIU_INT,&pcbddc->replicated_local_primal_indices[0],pcbddc->local_primal_sizes,pcbddc->local_primal_displacements,MPIU_INT,master_proc,prec_comm);CHKERRQ(ierr);
          ierr = MPI_Gatherv(&coarse_submat_vals[0],mysize2,MPIU_SCALAR,&temp_coarse_mat_vals[0],localsizes2,localdispl2,MPIU_SCALAR,master_proc,prec_comm);CHKERRQ(ierr);
        } else {
          ierr = MPI_Allgatherv(send_buffer,mysize,MPIU_INT,&pcbddc->replicated_local_primal_indices[0],pcbddc->local_primal_sizes,pcbddc->local_primal_displacements,MPIU_INT,prec_comm);CHKERRQ(ierr);
          ierr = MPI_Allgatherv(&coarse_submat_vals[0],mysize2,MPIU_SCALAR,&temp_coarse_mat_vals[0],localsizes2,localdispl2,MPIU_SCALAR,prec_comm);CHKERRQ(ierr);
        }
        ierr = PetscFree(send_buffer);CHKERRQ(ierr);
        break;
      }/* switch on coarse problem and communications associated with finished */
  }

  /* Now create and fill up coarse matrix */
  if ( rank_prec_comm == active_rank ) {

    Mat matis_coarse_local_mat;

    if (pcbddc->coarse_problem_type != MULTILEVEL_BDDC) {
      ierr = MatCreate(coarse_comm,&pcbddc->coarse_mat);CHKERRQ(ierr);
      ierr = MatSetSizes(pcbddc->coarse_mat,PETSC_DECIDE,PETSC_DECIDE,pcbddc->coarse_size,pcbddc->coarse_size);CHKERRQ(ierr);
      ierr = MatSetType(pcbddc->coarse_mat,coarse_mat_type);CHKERRQ(ierr);
      ierr = MatSetUp(pcbddc->coarse_mat);CHKERRQ(ierr);
      ierr = MatSetOption(pcbddc->coarse_mat,MAT_ROW_ORIENTED,PETSC_FALSE);CHKERRQ(ierr); /* local values stored in column major */
      ierr = MatSetOption(pcbddc->coarse_mat,MAT_IGNORE_ZERO_ENTRIES,PETSC_TRUE);CHKERRQ(ierr);
    } else {
      ierr = MatCreateIS(coarse_comm,1,PETSC_DECIDE,PETSC_DECIDE,pcbddc->coarse_size,pcbddc->coarse_size,coarse_ISLG,&pcbddc->coarse_mat);CHKERRQ(ierr);
      ierr = MatSetUp(pcbddc->coarse_mat);CHKERRQ(ierr);
      ierr = MatISGetLocalMat(pcbddc->coarse_mat,&matis_coarse_local_mat);CHKERRQ(ierr);
      ierr = MatSetUp(matis_coarse_local_mat);CHKERRQ(ierr);
      ierr = MatSetOption(matis_coarse_local_mat,MAT_ROW_ORIENTED,PETSC_FALSE);CHKERRQ(ierr); /* local values stored in column major */
      ierr = MatSetOption(matis_coarse_local_mat,MAT_IGNORE_ZERO_ENTRIES,PETSC_TRUE);CHKERRQ(ierr); 
    }
    /* preallocation */
    if (pcbddc->coarse_problem_type != MULTILEVEL_BDDC) {

      PetscInt lrows,lcols;

      ierr = MatGetLocalSize(pcbddc->coarse_mat,&lrows,&lcols);CHKERRQ(ierr);
      ierr = MatPreallocateInitialize(coarse_comm,lrows,lcols,dnz,onz);CHKERRQ(ierr);

      if (pcbddc->coarse_problem_type == PARALLEL_BDDC) {

        Vec         vec_dnz,vec_onz;
        PetscScalar *my_dnz,*my_onz,*array;
        PetscInt    *mat_ranges,*row_ownership;
        PetscInt    coarse_index_row,coarse_index_col,owner;

        ierr = VecCreate(prec_comm,&vec_dnz);CHKERRQ(ierr);
        ierr = VecSetSizes(vec_dnz,PETSC_DECIDE,pcbddc->coarse_size);CHKERRQ(ierr);
        ierr = VecSetType(vec_dnz,VECMPI);CHKERRQ(ierr);
        ierr = VecDuplicate(vec_dnz,&vec_onz);CHKERRQ(ierr);

        ierr = PetscMalloc(pcbddc->local_primal_size*sizeof(PetscScalar),&my_dnz);CHKERRQ(ierr);
        ierr = PetscMalloc(pcbddc->local_primal_size*sizeof(PetscScalar),&my_onz);CHKERRQ(ierr);
        ierr = PetscMemzero(my_dnz,pcbddc->local_primal_size*sizeof(PetscScalar));CHKERRQ(ierr);
        ierr = PetscMemzero(my_onz,pcbddc->local_primal_size*sizeof(PetscScalar));CHKERRQ(ierr);

        ierr = PetscMalloc(pcbddc->coarse_size*sizeof(PetscInt),&row_ownership);CHKERRQ(ierr);
        ierr = MatGetOwnershipRanges(pcbddc->coarse_mat,(const PetscInt**)&mat_ranges);CHKERRQ(ierr);
        for (i=0;i<size_prec_comm;i++) {
          for (j=mat_ranges[i];j<mat_ranges[i+1];j++) {
            row_ownership[j]=i;
          }
        }

        for (i=0;i<pcbddc->local_primal_size;i++) {
          coarse_index_row = pcbddc->local_primal_indices[i];
          owner = row_ownership[coarse_index_row];
          for (j=i;j<pcbddc->local_primal_size;j++) {
            owner = row_ownership[coarse_index_row];
            coarse_index_col = pcbddc->local_primal_indices[j];
            if (coarse_index_col > mat_ranges[owner]-1 && coarse_index_col < mat_ranges[owner+1] ) {
              my_dnz[i] += 1.0;
            } else {
              my_onz[i] += 1.0;
            }
            if (i != j) {
              owner = row_ownership[coarse_index_col];
              if (coarse_index_row > mat_ranges[owner]-1 && coarse_index_row < mat_ranges[owner+1] ) {
                my_dnz[j] += 1.0;
              } else {
                my_onz[j] += 1.0;
              }
            }
          }
        }
        ierr = VecSet(vec_dnz,0.0);CHKERRQ(ierr);
        ierr = VecSet(vec_onz,0.0);CHKERRQ(ierr);
        if (pcbddc->local_primal_size) {
          ierr = VecSetValues(vec_dnz,pcbddc->local_primal_size,pcbddc->local_primal_indices,my_dnz,ADD_VALUES);CHKERRQ(ierr);
          ierr = VecSetValues(vec_onz,pcbddc->local_primal_size,pcbddc->local_primal_indices,my_onz,ADD_VALUES);CHKERRQ(ierr);
        }
        ierr = VecAssemblyBegin(vec_dnz);CHKERRQ(ierr);
        ierr = VecAssemblyBegin(vec_onz);CHKERRQ(ierr);
        ierr = VecAssemblyEnd(vec_dnz);CHKERRQ(ierr);
        ierr = VecAssemblyEnd(vec_onz);CHKERRQ(ierr);
        j = mat_ranges[rank_prec_comm+1]-mat_ranges[rank_prec_comm];
        ierr = VecGetArray(vec_dnz,&array);CHKERRQ(ierr);
        for (i=0;i<j;i++) {
          dnz[i] = (PetscInt)array[i];
        }
        ierr = VecRestoreArray(vec_dnz,&array);CHKERRQ(ierr);
        ierr = VecGetArray(vec_onz,&array);CHKERRQ(ierr);
        for (i=0;i<j;i++) {
          onz[i] = (PetscInt)array[i];
        }
        ierr = VecRestoreArray(vec_onz,&array);CHKERRQ(ierr);
        ierr = PetscFree(my_dnz);CHKERRQ(ierr);
        ierr = PetscFree(my_onz);CHKERRQ(ierr);
        ierr = PetscFree(row_ownership);CHKERRQ(ierr);
        ierr = VecDestroy(&vec_dnz);CHKERRQ(ierr);
        ierr = VecDestroy(&vec_onz);CHKERRQ(ierr);
      } else {
        for (k=0;k<size_prec_comm;k++){
          offset=pcbddc->local_primal_displacements[k];
          offset2=localdispl2[k];
          ins_local_primal_size = pcbddc->local_primal_sizes[k];
          ierr = PetscMalloc(ins_local_primal_size*sizeof(PetscInt),&ins_local_primal_indices);CHKERRQ(ierr);
          for (j=0;j<ins_local_primal_size;j++){
            ins_local_primal_indices[j]=(PetscInt)pcbddc->replicated_local_primal_indices[offset+j];
          }
          for (j=0;j<ins_local_primal_size;j++) {
            ierr = MatPreallocateSet(ins_local_primal_indices[j],ins_local_primal_size,ins_local_primal_indices,dnz,onz);CHKERRQ(ierr);
          }
          ierr = PetscFree(ins_local_primal_indices);CHKERRQ(ierr);
        }
      }
      /* check */
      for (i=0;i<lrows;i++) {
        if (dnz[i]>lcols) {
          dnz[i]=lcols;
        }
        if (onz[i]>pcbddc->coarse_size-lcols) {
          onz[i]=pcbddc->coarse_size-lcols;
        }
      }
      ierr = MatSeqAIJSetPreallocation(pcbddc->coarse_mat,PETSC_NULL,dnz);CHKERRQ(ierr);  
      ierr = MatMPIAIJSetPreallocation(pcbddc->coarse_mat,PETSC_NULL,dnz,PETSC_NULL,onz);CHKERRQ(ierr);  
      ierr = MatPreallocateFinalize(dnz,onz);CHKERRQ(ierr);
    } else {
      ierr = MatSeqAIJSetPreallocation(matis_coarse_local_mat,0,dnz);CHKERRQ(ierr);
      ierr = PetscFree(dnz);CHKERRQ(ierr);
    }
    /* insert values */
    if (pcbddc->coarse_problem_type == PARALLEL_BDDC) {
      ierr = MatSetValues(pcbddc->coarse_mat,ins_local_primal_size,ins_local_primal_indices,ins_local_primal_size,ins_local_primal_indices,ins_coarse_mat_vals,ADD_VALUES);CHKERRQ(ierr);
    } else if (pcbddc->coarse_problem_type == MULTILEVEL_BDDC) {
      if (pcbddc->coarsening_ratio == 1) {
        ins_coarse_mat_vals = coarse_submat_vals;
        ierr = MatSetValues(pcbddc->coarse_mat,ins_local_primal_size,ins_local_primal_indices,ins_local_primal_size,ins_local_primal_indices,ins_coarse_mat_vals,INSERT_VALUES);CHKERRQ(ierr);
      } else {
        ierr = PetscFree(ins_local_primal_indices);CHKERRQ(ierr);
        for (k=0;k<pcbddc->replicated_primal_size;k++) {
          offset = pcbddc->local_primal_displacements[k];
          offset2 = localdispl2[k];
          ins_local_primal_size = pcbddc->local_primal_displacements[k+1]-pcbddc->local_primal_displacements[k];
          ierr = PetscMalloc(ins_local_primal_size*sizeof(PetscInt),&ins_local_primal_indices);CHKERRQ(ierr);
          for (j=0;j<ins_local_primal_size;j++){
            ins_local_primal_indices[j]=(PetscInt)pcbddc->replicated_local_primal_indices[offset+j];
          }
          ins_coarse_mat_vals = &temp_coarse_mat_vals[offset2];
          ierr = MatSetValues(pcbddc->coarse_mat,ins_local_primal_size,ins_local_primal_indices,ins_local_primal_size,ins_local_primal_indices,ins_coarse_mat_vals,ADD_VALUES);CHKERRQ(ierr);
          ierr = PetscFree(ins_local_primal_indices);CHKERRQ(ierr);
        }
      }
      ins_local_primal_indices = 0;
      ins_coarse_mat_vals = 0;
    } else {
      for (k=0;k<size_prec_comm;k++){
        offset=pcbddc->local_primal_displacements[k];
        offset2=localdispl2[k];
        ins_local_primal_size = pcbddc->local_primal_sizes[k];
        ierr = PetscMalloc(ins_local_primal_size*sizeof(PetscInt),&ins_local_primal_indices);CHKERRQ(ierr);
        for (j=0;j<ins_local_primal_size;j++){
          ins_local_primal_indices[j]=(PetscInt)pcbddc->replicated_local_primal_indices[offset+j];
        }
        ins_coarse_mat_vals = &temp_coarse_mat_vals[offset2];
        ierr = MatSetValues(pcbddc->coarse_mat,ins_local_primal_size,ins_local_primal_indices,ins_local_primal_size,ins_local_primal_indices,ins_coarse_mat_vals,ADD_VALUES);CHKERRQ(ierr);
        ierr = PetscFree(ins_local_primal_indices);CHKERRQ(ierr);
      }
      ins_local_primal_indices = 0;
      ins_coarse_mat_vals = 0;
    }
    ierr = MatAssemblyBegin(pcbddc->coarse_mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(pcbddc->coarse_mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    /* symmetry of coarse matrix */
    if (issym) {
      ierr = MatSetOption(pcbddc->coarse_mat,MAT_SYMMETRIC,PETSC_TRUE);CHKERRQ(ierr);
    }
    ierr = MatGetVecs(pcbddc->coarse_mat,&pcbddc->coarse_vec,&pcbddc->coarse_rhs);CHKERRQ(ierr);
  }

  /* create loc to glob scatters if needed */
  if (pcbddc->coarse_communications_type == SCATTERS_BDDC) {
     IS local_IS,global_IS;
     ierr = ISCreateStride(PETSC_COMM_SELF,pcbddc->local_primal_size,0,1,&local_IS);CHKERRQ(ierr);
     ierr = ISCreateGeneral(PETSC_COMM_SELF,pcbddc->local_primal_size,pcbddc->local_primal_indices,PETSC_COPY_VALUES,&global_IS);CHKERRQ(ierr);
     ierr = VecScatterCreate(pcbddc->vec1_P,local_IS,pcbddc->coarse_vec,global_IS,&pcbddc->coarse_loc_to_glob);CHKERRQ(ierr);
     ierr = ISDestroy(&local_IS);CHKERRQ(ierr);
     ierr = ISDestroy(&global_IS);CHKERRQ(ierr);
  }

  /* free memory no longer needed */
  if (coarse_ISLG)              { ierr = ISLocalToGlobalMappingDestroy(&coarse_ISLG);CHKERRQ(ierr); }
  if (ins_local_primal_indices) { ierr = PetscFree(ins_local_primal_indices);CHKERRQ(ierr); }
  if (ins_coarse_mat_vals)      { ierr = PetscFree(ins_coarse_mat_vals);CHKERRQ(ierr); }
  if (localsizes2)              { ierr = PetscFree(localsizes2);CHKERRQ(ierr); }
  if (localdispl2)              { ierr = PetscFree(localdispl2);CHKERRQ(ierr); }
  if (temp_coarse_mat_vals)     { ierr = PetscFree(temp_coarse_mat_vals);CHKERRQ(ierr); }

  /* Eval coarse null space */
  if (pcbddc->NullSpace) {
    const Vec      *nsp_vecs;
    PetscInt       nsp_size,coarse_nsp_size;
    PetscBool      nsp_has_cnst;
    PetscReal      test_null;
    Vec            *coarse_nsp_vecs;
 
    coarse_nsp_size = 0;
    coarse_nsp_vecs = 0;
    ierr = MatNullSpaceGetVecs(pcbddc->NullSpace,&nsp_has_cnst,&nsp_size,&nsp_vecs);CHKERRQ(ierr);
    if (rank_prec_comm == active_rank) {
      ierr = PetscMalloc((nsp_size+1)*sizeof(Vec),&coarse_nsp_vecs);CHKERRQ(ierr);
      for (i=0;i<nsp_size+1;i++) {
        ierr = VecDuplicate(pcbddc->coarse_vec,&coarse_nsp_vecs[i]);CHKERRQ(ierr);
      }
    }
    if (nsp_has_cnst) {
      ierr = VecSet(pcis->vec1_N,1.0);CHKERRQ(ierr);
      ierr = MatMult(pcbddc->ConstraintMatrix,pcis->vec1_N,pcbddc->vec1_P);CHKERRQ(ierr);
      ierr = PCBDDCScatterCoarseDataBegin(pc,pcbddc->vec1_P,pcbddc->coarse_vec,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
      ierr = PCBDDCScatterCoarseDataEnd(pc,pcbddc->vec1_P,pcbddc->coarse_vec,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
      if (rank_prec_comm == active_rank) {
        ierr = MatMult(pcbddc->coarse_mat,pcbddc->coarse_vec,pcbddc->coarse_rhs);CHKERRQ(ierr);
        ierr = VecNorm(pcbddc->coarse_rhs,NORM_INFINITY,&test_null);CHKERRQ(ierr);
        if (test_null > 1.0e-12 && pcbddc->dbg_flag ) {
          ierr = PetscViewerASCIIPrintf(pcbddc->dbg_viewer,"Constant coarse null space error % 1.14e\n",test_null);CHKERRQ(ierr);
        }
        ierr = VecCopy(pcbddc->coarse_vec,coarse_nsp_vecs[coarse_nsp_size]);CHKERRQ(ierr);
        coarse_nsp_size++;
      }
    }
    for (i=0;i<nsp_size;i++)  {
      ierr = VecScatterBegin(matis->ctx,nsp_vecs[i],pcis->vec1_N,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
      ierr = VecScatterEnd  (matis->ctx,nsp_vecs[i],pcis->vec1_N,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
      ierr = MatMult(pcbddc->ConstraintMatrix,pcis->vec1_N,pcbddc->vec1_P);CHKERRQ(ierr);
      ierr = PCBDDCScatterCoarseDataBegin(pc,pcbddc->vec1_P,pcbddc->coarse_vec,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
      ierr = PCBDDCScatterCoarseDataEnd(pc,pcbddc->vec1_P,pcbddc->coarse_vec,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
      if (rank_prec_comm == active_rank) {
        ierr = MatMult(pcbddc->coarse_mat,pcbddc->coarse_vec,pcbddc->coarse_rhs);CHKERRQ(ierr);
        ierr = VecNorm(pcbddc->coarse_rhs,NORM_2,&test_null);CHKERRQ(ierr);
        if (test_null > 1.0e-12 && pcbddc->dbg_flag ) {
          ierr = PetscViewerASCIIPrintf(pcbddc->dbg_viewer,"Vec %d coarse null space error % 1.14e\n",i,test_null);CHKERRQ(ierr);
        }
        ierr = VecCopy(pcbddc->coarse_vec,coarse_nsp_vecs[coarse_nsp_size]);CHKERRQ(ierr);
        coarse_nsp_size++;
      }
    }
    if (coarse_nsp_size > 0) {
      /* TODO orthonormalize vecs */
      ierr = VecNormalize(coarse_nsp_vecs[0],PETSC_NULL);CHKERRQ(ierr);
      ierr = MatNullSpaceCreate(coarse_comm,PETSC_FALSE,coarse_nsp_size,coarse_nsp_vecs,&pcbddc->CoarseNullSpace);CHKERRQ(ierr);
      for (i=0;i<nsp_size+1;i++) {
        ierr = VecDestroy(&coarse_nsp_vecs[i]);CHKERRQ(ierr);
      }
    }
    ierr = PetscFree(coarse_nsp_vecs);CHKERRQ(ierr);
  }

  /* KSP for coarse problem */
  if (rank_prec_comm == active_rank) {
    PetscBool isbddc=PETSC_FALSE;

    ierr = KSPCreate(coarse_comm,&pcbddc->coarse_ksp);CHKERRQ(ierr);
    ierr = PetscObjectIncrementTabLevel((PetscObject)pcbddc->coarse_ksp,(PetscObject)pc,1);CHKERRQ(ierr);
    ierr = KSPSetOperators(pcbddc->coarse_ksp,pcbddc->coarse_mat,pcbddc->coarse_mat,SAME_PRECONDITIONER);CHKERRQ(ierr);
    ierr = KSPSetTolerances(pcbddc->coarse_ksp,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT,max_it_coarse_ksp);CHKERRQ(ierr);
    ierr = KSPSetType(pcbddc->coarse_ksp,coarse_ksp_type);CHKERRQ(ierr);
    ierr = KSPGetPC(pcbddc->coarse_ksp,&pc_temp);CHKERRQ(ierr);
    ierr = PCSetType(pc_temp,coarse_pc_type);CHKERRQ(ierr);
    /* Allow user's customization */
    ierr = KSPSetOptionsPrefix(pcbddc->coarse_ksp,"coarse_");CHKERRQ(ierr);
    /* Set Up PC for coarse problem BDDC */
    if (pcbddc->coarse_problem_type == MULTILEVEL_BDDC) { 
      i = pcbddc->current_level+1;
      ierr = PCBDDCSetLevel(pc_temp,i);CHKERRQ(ierr);
      ierr = PCBDDCSetCoarseningRatio(pc_temp,pcbddc->coarsening_ratio);CHKERRQ(ierr);
      ierr = PCBDDCSetMaxLevels(pc_temp,pcbddc->max_levels);CHKERRQ(ierr);
      ierr = PCBDDCSetCoarseProblemType(pc_temp,MULTILEVEL_BDDC);CHKERRQ(ierr);
      if (pcbddc->CoarseNullSpace) { ierr = PCBDDCSetNullSpace(pc_temp,pcbddc->CoarseNullSpace);CHKERRQ(ierr); }
      if (dbg_flag) { 
        ierr = PetscViewerASCIIPrintf(viewer,"----------------Level %d: Setting up level %d---------------\n",pcbddc->current_level,i);CHKERRQ(ierr);
        ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
      }
    }
    ierr = KSPSetFromOptions(pcbddc->coarse_ksp);CHKERRQ(ierr);
    ierr = KSPSetUp(pcbddc->coarse_ksp);CHKERRQ(ierr);

    ierr = KSPGetTolerances(pcbddc->coarse_ksp,PETSC_NULL,PETSC_NULL,PETSC_NULL,&j);CHKERRQ(ierr);
    ierr = KSPGetPC(pcbddc->coarse_ksp,&pc_temp);CHKERRQ(ierr);
    ierr = PetscObjectTypeCompare((PetscObject)pc_temp,PCBDDC,&isbddc);CHKERRQ(ierr);
    if (j == 1) {
      ierr = KSPSetNormType(pcbddc->coarse_ksp,KSP_NORM_NONE);CHKERRQ(ierr);
      if (isbddc) {
        ierr = PCBDDCSetUseExactDirichlet(pc_temp,PETSC_FALSE);CHKERRQ(ierr);
      }
    }
  }
  /* Check coarse problem if requested */
  if ( dbg_flag && rank_prec_comm == active_rank ) {
    KSP check_ksp;
    PC  check_pc;
    Vec check_vec;
    PetscReal   abs_infty_error,infty_error,lambda_min,lambda_max;
    KSPType check_ksp_type;

    /* Create ksp object suitable for extreme eigenvalues' estimation */
    ierr = KSPCreate(coarse_comm,&check_ksp);CHKERRQ(ierr);
    ierr = KSPSetOperators(check_ksp,pcbddc->coarse_mat,pcbddc->coarse_mat,SAME_PRECONDITIONER);CHKERRQ(ierr);
    ierr = KSPSetTolerances(check_ksp,1.e-12,1.e-12,PETSC_DEFAULT,pcbddc->coarse_size);CHKERRQ(ierr);
    if (pcbddc->coarse_problem_type == MULTILEVEL_BDDC) {
      if (issym) {
        check_ksp_type = KSPCG;
      } else {
        check_ksp_type = KSPGMRES;
      }
      ierr = KSPSetComputeSingularValues(check_ksp,PETSC_TRUE);CHKERRQ(ierr);
    } else {
      check_ksp_type = KSPPREONLY;
    }
    ierr = KSPSetType(check_ksp,check_ksp_type);CHKERRQ(ierr);
    ierr = KSPGetPC(pcbddc->coarse_ksp,&check_pc);CHKERRQ(ierr);
    ierr = KSPSetPC(check_ksp,check_pc);CHKERRQ(ierr);
    ierr = KSPSetUp(check_ksp);CHKERRQ(ierr);
    /* create random vec */
    ierr = VecDuplicate(pcbddc->coarse_vec,&check_vec);CHKERRQ(ierr);
    ierr = VecSetRandom(check_vec,PETSC_NULL);CHKERRQ(ierr);
    if (pcbddc->CoarseNullSpace) { ierr = MatNullSpaceRemove(pcbddc->CoarseNullSpace,check_vec,PETSC_NULL);CHKERRQ(ierr); }
    ierr = MatMult(pcbddc->coarse_mat,check_vec,pcbddc->coarse_rhs);CHKERRQ(ierr);
    /* solve coarse problem */
    ierr = KSPSolve(check_ksp,pcbddc->coarse_rhs,pcbddc->coarse_vec);CHKERRQ(ierr);
    if (pcbddc->CoarseNullSpace) { ierr = MatNullSpaceRemove(pcbddc->CoarseNullSpace,pcbddc->coarse_vec,PETSC_NULL);CHKERRQ(ierr); }
    /* check coarse problem residual error */
    ierr = VecAXPY(check_vec,-1.0,pcbddc->coarse_vec);CHKERRQ(ierr);
    ierr = VecNorm(check_vec,NORM_INFINITY,&infty_error);CHKERRQ(ierr);
    ierr = MatMult(pcbddc->coarse_mat,check_vec,pcbddc->coarse_rhs);CHKERRQ(ierr);
    ierr = VecNorm(pcbddc->coarse_rhs,NORM_INFINITY,&abs_infty_error);CHKERRQ(ierr);
    ierr = VecDestroy(&check_vec);CHKERRQ(ierr);
    /* get eigenvalue estimation if inexact */
    if (pcbddc->coarse_problem_type == MULTILEVEL_BDDC) {
      ierr = KSPComputeExtremeSingularValues(check_ksp,&lambda_max,&lambda_min);CHKERRQ(ierr);
      ierr = KSPGetIterationNumber(check_ksp,&k);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"Coarse problem eigenvalues estimated with %d iterations of %s.\n",k,check_ksp_type);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"Coarse problem eigenvalues: % 1.14e %1.14e\n",lambda_min,lambda_max);CHKERRQ(ierr);
    }
    ierr = PetscViewerASCIIPrintf(viewer,"Coarse problem exact infty_error   : %1.14e\n",infty_error);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"Coarse problem residual infty_error: %1.14e\n",abs_infty_error);CHKERRQ(ierr);
    ierr = KSPDestroy(&check_ksp);CHKERRQ(ierr);
  }
  if (dbg_flag) { ierr = PetscViewerFlush(viewer);CHKERRQ(ierr); }

  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCBDDCManageLocalBoundaries"
static PetscErrorCode PCBDDCManageLocalBoundaries(PC pc)
{

  PC_BDDC     *pcbddc = (PC_BDDC*)pc->data;
  PC_IS         *pcis = (PC_IS*)pc->data;
  Mat_IS      *matis  = (Mat_IS*)pc->pmat->data; 
  PCBDDCGraph mat_graph=pcbddc->mat_graph;
  PetscInt    *is_indices,*auxis;
  PetscInt    bs,ierr,i,j,s,k,iindex,neumann_bsize,dirichlet_bsize;
  PetscInt    total_counts,nodes_touched,where_values=1,vertex_size;
  PetscMPIInt adapt_interface=0,adapt_interface_reduced=0,NEUMANNCNT=0;
  PetscBool   same_set;
  MPI_Comm    interface_comm=((PetscObject)pc)->comm;
  PetscBool   use_faces=PETSC_FALSE,use_edges=PETSC_FALSE;
  const PetscInt *neumann_nodes;
  const PetscInt *dirichlet_nodes;
  IS          used_IS,*custom_ISForDofs;
  PetscScalar *array;
  PetscScalar *array2;
  PetscViewer viewer=pcbddc->dbg_viewer;
  PetscInt    *queue_in_global_numbering;

  PetscFunctionBegin;
  /* Setup local adjacency graph */
  mat_graph->nvtxs=pcis->n;
  if (!mat_graph->xadj) { NEUMANNCNT = 1; }
  ierr = PCBDDCSetupLocalAdjacencyGraph(pc);CHKERRQ(ierr);
  i = mat_graph->nvtxs;
  ierr = PetscMalloc4(i,PetscInt,&mat_graph->where,i,PetscInt,&mat_graph->count,i+1,PetscInt,&mat_graph->cptr,i,PetscInt,&mat_graph->queue);CHKERRQ(ierr);
  ierr = PetscMalloc2(i,PetscInt,&mat_graph->which_dof,i,PetscBool,&mat_graph->touched);CHKERRQ(ierr);
  ierr = PetscMemzero(mat_graph->where,mat_graph->nvtxs*sizeof(PetscInt));CHKERRQ(ierr);
  ierr = PetscMemzero(mat_graph->count,mat_graph->nvtxs*sizeof(PetscInt));CHKERRQ(ierr);
  ierr = PetscMemzero(mat_graph->which_dof,mat_graph->nvtxs*sizeof(PetscInt));CHKERRQ(ierr);
  ierr = PetscMemzero(mat_graph->queue,mat_graph->nvtxs*sizeof(PetscInt));CHKERRQ(ierr);
  ierr = PetscMemzero(mat_graph->cptr,(mat_graph->nvtxs+1)*sizeof(PetscInt));CHKERRQ(ierr);
  
  /* Setting dofs splitting in mat_graph->which_dof
     Get information about dofs' splitting if provided by the user
     Otherwise it assumes a constant block size */
  vertex_size=0;
  if (!pcbddc->n_ISForDofs) {
    ierr = MatGetBlockSize(matis->A,&bs);CHKERRQ(ierr);
    ierr = PetscMalloc(bs*sizeof(IS),&custom_ISForDofs);CHKERRQ(ierr);
    for (i=0;i<bs;i++) {
      ierr = ISCreateStride(PETSC_COMM_SELF,pcis->n/bs,i,bs,&custom_ISForDofs[i]);CHKERRQ(ierr);
    }
    ierr = PCBDDCSetDofsSplitting(pc,bs,custom_ISForDofs);CHKERRQ(ierr);
    vertex_size=1;
    /* remove my references to IS objects */
    for (i=0;i<bs;i++) {
      ierr = ISDestroy(&custom_ISForDofs[i]);CHKERRQ(ierr);
    }
    ierr = PetscFree(custom_ISForDofs);CHKERRQ(ierr);
  }
  for (i=0;i<pcbddc->n_ISForDofs;i++) {
    ierr = ISGetSize(pcbddc->ISForDofs[i],&k);CHKERRQ(ierr);
    ierr = ISGetIndices(pcbddc->ISForDofs[i],(const PetscInt**)&is_indices);CHKERRQ(ierr);
    for (j=0;j<k;j++) {
      mat_graph->which_dof[is_indices[j]]=i;
    }
    ierr = ISRestoreIndices(pcbddc->ISForDofs[i],(const PetscInt**)&is_indices);CHKERRQ(ierr);
  }
  /* use mat block size as vertex size if it has not yet set */
  if (!vertex_size) {
    ierr = MatGetBlockSize(matis->A,&vertex_size);CHKERRQ(ierr);
  }

  /* count number of neigh per node */
  total_counts=0;
  for (i=1;i<pcis->n_neigh;i++){
    s=pcis->n_shared[i];
    total_counts+=s;
    for (j=0;j<s;j++){
      mat_graph->count[pcis->shared[i][j]] += 1;
    }
  }
  /* Take into account Neumann data -> it increments number of sharing subdomains for nodes lying on the interface */
  ierr = PCBDDCGetNeumannBoundaries(pc,&used_IS);CHKERRQ(ierr);
  ierr = VecSet(pcis->vec1_N,0.0);CHKERRQ(ierr);
  ierr = VecGetArray(pcis->vec1_N,&array);CHKERRQ(ierr);
  if (used_IS) {
    ierr = ISGetSize(used_IS,&neumann_bsize);CHKERRQ(ierr);
    ierr = ISGetIndices(used_IS,&neumann_nodes);CHKERRQ(ierr);
    for (i=0;i<neumann_bsize;i++){
      iindex = neumann_nodes[i];
      if (mat_graph->count[iindex] > NEUMANNCNT && array[iindex]==0.0){ 
        mat_graph->count[iindex]+=1;
        total_counts++;
        array[iindex]=array[iindex]+1.0;
      } else if (array[iindex]>0.0) {
        SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_USER,"Error for neumann nodes provided to BDDC! They must be uniquely listed! Found duplicate node %d\n",iindex);
      }
    }
  }
  ierr = VecRestoreArray(pcis->vec1_N,&array);CHKERRQ(ierr);
  /* allocate space for storing the set of neighbours for each node */ 
  ierr = PetscMalloc(mat_graph->nvtxs*sizeof(PetscInt*),&mat_graph->neighbours_set);CHKERRQ(ierr);
  if (mat_graph->nvtxs) { ierr = PetscMalloc(total_counts*sizeof(PetscInt),&mat_graph->neighbours_set[0]);CHKERRQ(ierr); }
  for (i=1;i<mat_graph->nvtxs;i++) mat_graph->neighbours_set[i]=mat_graph->neighbours_set[i-1]+mat_graph->count[i-1];
  ierr = PetscMemzero(mat_graph->count,mat_graph->nvtxs*sizeof(PetscInt));CHKERRQ(ierr);
  for (i=1;i<pcis->n_neigh;i++){
    s=pcis->n_shared[i];
    for (j=0;j<s;j++) {
      k=pcis->shared[i][j];
      mat_graph->neighbours_set[k][mat_graph->count[k]] = pcis->neigh[i];
      mat_graph->count[k]+=1;
    }
  }
  /* Check consistency of Neumann nodes */
  ierr = VecSet(pcis->vec1_global,0.0);CHKERRQ(ierr);
  ierr = VecScatterBegin(matis->ctx,pcis->vec1_N,pcis->vec1_global,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  ierr = VecScatterEnd  (matis->ctx,pcis->vec1_N,pcis->vec1_global,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  ierr = VecScatterBegin(matis->ctx,pcis->vec1_global,pcis->vec1_N,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd  (matis->ctx,pcis->vec1_global,pcis->vec1_N,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecGetArray(pcis->vec1_N,&array);CHKERRQ(ierr);
  /* set -1 fake neighbour to mimic Neumann boundary */
  if (used_IS) {
    for (i=0;i<neumann_bsize;i++){
      iindex = neumann_nodes[i];
      if (mat_graph->count[iindex] > NEUMANNCNT){
        if (mat_graph->count[iindex]+1 != (PetscInt)array[iindex]) {
          SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_USER,"Neumann nodes provided to BDDC must be consistent among neighbours!\nNode %d: number of sharing subdomains %d != number of subdomains for which it is a neumann node %d\n",iindex,mat_graph->count[iindex]+1,(PetscInt)array[iindex]);
        } 
        mat_graph->neighbours_set[iindex][mat_graph->count[iindex]] = -1;
        mat_graph->count[iindex]+=1;
      }
    }
    ierr = ISRestoreIndices(used_IS,&neumann_nodes);CHKERRQ(ierr);
  }
  ierr = VecRestoreArray(pcis->vec1_N,&array);CHKERRQ(ierr);
  /* sort set of sharing subdomains */
  for (i=0;i<mat_graph->nvtxs;i++) { ierr = PetscSortInt(mat_graph->count[i],mat_graph->neighbours_set[i]);CHKERRQ(ierr); }
  /* remove interior nodes and dirichlet boundary nodes from the next search into the graph */
  for (i=0;i<mat_graph->nvtxs;i++){mat_graph->touched[i]=PETSC_FALSE;}
  nodes_touched=0;
  ierr = PCBDDCGetDirichletBoundaries(pc,&used_IS);CHKERRQ(ierr);
  ierr = VecSet(pcis->vec2_N,0.0);CHKERRQ(ierr);
  ierr = VecGetArray(pcis->vec1_N,&array);CHKERRQ(ierr);
  ierr = VecGetArray(pcis->vec2_N,&array2);CHKERRQ(ierr);
  if (used_IS) {
    ierr = ISGetSize(used_IS,&dirichlet_bsize);CHKERRQ(ierr);
    if (dirichlet_bsize && matis->pure_neumann) {
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"Dirichlet boundaries are intended to be used with matrices with zeroed rows!\n");
    }
    ierr = ISGetIndices(used_IS,&dirichlet_nodes);CHKERRQ(ierr);
    for (i=0;i<dirichlet_bsize;i++){
      iindex=dirichlet_nodes[i];
      if (mat_graph->count[iindex] && !mat_graph->touched[iindex]) {
        if (array[iindex]>0.0) {
          SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_USER,"BDDC cannot have nodes which are marked as Neumann and Dirichlet at the same time! Wrong node %d\n",iindex);
        }
        mat_graph->touched[iindex]=PETSC_TRUE;
        mat_graph->where[iindex]=0;
        nodes_touched++;
        array2[iindex]=array2[iindex]+1.0;
      } 
    }
    ierr = ISRestoreIndices(used_IS,&dirichlet_nodes);CHKERRQ(ierr);
  }
  ierr = VecRestoreArray(pcis->vec1_N,&array);CHKERRQ(ierr);
  ierr = VecRestoreArray(pcis->vec2_N,&array2);CHKERRQ(ierr);
  /* Check consistency of Dirichlet nodes */
  ierr = VecSet(pcis->vec1_N,1.0);CHKERRQ(ierr);
  ierr = VecSet(pcis->vec1_global,0.0);CHKERRQ(ierr);
  ierr = VecScatterBegin(matis->ctx,pcis->vec1_N,pcis->vec1_global,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  ierr = VecScatterEnd  (matis->ctx,pcis->vec1_N,pcis->vec1_global,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  ierr = VecScatterBegin(matis->ctx,pcis->vec1_global,pcis->vec1_N,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd  (matis->ctx,pcis->vec1_global,pcis->vec1_N,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecSet(pcis->vec1_global,0.0);CHKERRQ(ierr);
  ierr = VecScatterBegin(matis->ctx,pcis->vec2_N,pcis->vec1_global,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  ierr = VecScatterEnd  (matis->ctx,pcis->vec2_N,pcis->vec1_global,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  ierr = VecScatterBegin(matis->ctx,pcis->vec1_global,pcis->vec2_N,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd  (matis->ctx,pcis->vec1_global,pcis->vec2_N,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecGetArray(pcis->vec1_N,&array);CHKERRQ(ierr);
  ierr = VecGetArray(pcis->vec2_N,&array2);CHKERRQ(ierr);
  if (used_IS) {
    ierr = ISGetSize(used_IS,&dirichlet_bsize);CHKERRQ(ierr);
    ierr = ISGetIndices(used_IS,&dirichlet_nodes);CHKERRQ(ierr);
    for (i=0;i<dirichlet_bsize;i++){
      iindex=dirichlet_nodes[i];
      if (array[iindex]>1.0 && array[iindex]!=array2[iindex] ) {
         SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_USER,"Dirichlet nodes provided to BDDC must be consistent among neighbours!\nNode %d: number of sharing subdomains %d != number of subdomains for which it is a neumann node %d\n",iindex,(PetscInt)array[iindex],(PetscInt)array2[iindex]);
      } 
    }
    ierr = ISRestoreIndices(used_IS,&dirichlet_nodes);CHKERRQ(ierr);
  }
  ierr = VecRestoreArray(pcis->vec1_N,&array);CHKERRQ(ierr);
  ierr = VecRestoreArray(pcis->vec2_N,&array2);CHKERRQ(ierr);

  for (i=0;i<mat_graph->nvtxs;i++){
    if (!mat_graph->count[i]){  /* interior nodes */
      mat_graph->touched[i]=PETSC_TRUE;
      mat_graph->where[i]=0;
      nodes_touched++;
    }
  } 
  mat_graph->ncmps = 0;
  i=0;
  while(nodes_touched<mat_graph->nvtxs) {
    /*  find first untouched node in local ordering */
    while(mat_graph->touched[i]) i++;
    mat_graph->touched[i]=PETSC_TRUE;
    mat_graph->where[i]=where_values;
    nodes_touched++;
    /* now find all other nodes having the same set of sharing subdomains */
    for (j=i+1;j<mat_graph->nvtxs;j++){
      /* check for same number of sharing subdomains and dof number */
      if (!mat_graph->touched[j] && mat_graph->count[i]==mat_graph->count[j] && mat_graph->which_dof[i] == mat_graph->which_dof[j] ){
        /* check for same set of sharing subdomains */
        same_set=PETSC_TRUE;
        for (k=0;k<mat_graph->count[j];k++){
          if (mat_graph->neighbours_set[i][k]!=mat_graph->neighbours_set[j][k]) {
            same_set=PETSC_FALSE;
          }
        }
        /* I found a friend of mine */
        if (same_set) {
          mat_graph->where[j]=where_values;
          mat_graph->touched[j]=PETSC_TRUE;
          nodes_touched++;
        }
      }
    }
    where_values++;
  }
  where_values--; if (where_values<0) where_values=0;
  ierr = PetscMalloc(where_values*sizeof(PetscMPIInt),&mat_graph->where_ncmps);CHKERRQ(ierr);
  /* Find connected components defined on the shared interface */
  if (where_values) {
    ierr = PCBDDCFindConnectedComponents(mat_graph, where_values); 
  }
  ierr = PetscMalloc(mat_graph->nvtxs*sizeof(PetscInt),&queue_in_global_numbering);CHKERRQ(ierr);
  /* check consistency of connected components among neighbouring subdomains -> it adapt them in case it is needed */
  for (i=0;i<where_values;i++) {
    /* We are not sure that on a given subset of the local interface,
       two connected components will be the same among sharing subdomains */
    if (mat_graph->where_ncmps[i]>1) { 
      adapt_interface=1;
      break;
    }
  }
  ierr = MPI_Allreduce(&adapt_interface,&adapt_interface_reduced,1,MPIU_INT,MPI_LOR,interface_comm);CHKERRQ(ierr);
  if (pcbddc->dbg_flag && adapt_interface_reduced) {
    ierr = PetscViewerASCIIPrintf(viewer,"Adapting interface\n");CHKERRQ(ierr);
    ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
  }
  if (where_values && adapt_interface_reduced) {

    PetscInt sum_requests=0,my_rank;
    PetscInt buffer_size,start_of_recv,size_of_recv,start_of_send;
    PetscInt temp_buffer_size,ins_val,global_where_counter;
    PetscInt *cum_recv_counts;
    PetscInt *where_to_nodes_indices;
    PetscInt *petsc_buffer;
    PetscMPIInt *recv_buffer;
    PetscMPIInt *recv_buffer_where;
    PetscMPIInt *send_buffer;
    PetscMPIInt size_of_send;
    PetscInt *sizes_of_sends;
    MPI_Request *send_requests;
    MPI_Request *recv_requests;
    PetscInt *where_cc_adapt;
    PetscInt **temp_buffer;
    PetscInt *nodes_to_temp_buffer_indices;
    PetscInt *add_to_where;
    PetscInt *aux_new_xadj,*new_xadj,*new_adjncy;

    /* Retrict adjacency graph using information from connected components */
    ierr = PetscMalloc(mat_graph->nvtxs*sizeof(PetscInt),&aux_new_xadj);CHKERRQ(ierr);
    for (i=0;i<mat_graph->nvtxs;i++) {
      aux_new_xadj[i]=1;
    }
    for (i=0;i<mat_graph->ncmps;i++) {
      k = mat_graph->cptr[i+1]-mat_graph->cptr[i];
      for (j=0;j<k;j++) {
        aux_new_xadj[mat_graph->queue[mat_graph->cptr[i]+j]]=k;
      }
    }
    j = 0;
    for (i=0;i<mat_graph->nvtxs;i++) {
      j += aux_new_xadj[i];
    }
    ierr = PetscMalloc((mat_graph->nvtxs+1)*sizeof(PetscInt),&new_xadj);CHKERRQ(ierr);
    ierr = PetscMalloc(j*sizeof(PetscInt),&new_adjncy);CHKERRQ(ierr);
    new_xadj[0]=0;
    for (i=0;i<mat_graph->nvtxs;i++) {
      new_xadj[i+1]=new_xadj[i]+aux_new_xadj[i];
      if (aux_new_xadj[i]==1) {
        new_adjncy[new_xadj[i]]=i;
      }
    }
    ierr = PetscFree(aux_new_xadj);CHKERRQ(ierr); 
    for (i=0;i<mat_graph->ncmps;i++) {
      k = mat_graph->cptr[i+1]-mat_graph->cptr[i];
      for (j=0;j<k;j++) {
        ierr = PetscMemcpy(&new_adjncy[new_xadj[mat_graph->queue[mat_graph->cptr[i]+j]]],&mat_graph->queue[mat_graph->cptr[i]],k*sizeof(PetscInt));CHKERRQ(ierr);
      }
    }
    ierr = PCBDDCSetLocalAdjacencyGraph(pc,mat_graph->nvtxs,new_xadj,new_adjncy,PETSC_OWN_POINTER);CHKERRQ(ierr);
    /* For consistency among neughbouring procs, I need to sort (by global ordering) each connected component */
    for (i=0;i<mat_graph->ncmps;i++) {
      k = mat_graph->cptr[i+1]-mat_graph->cptr[i];
      ierr = ISLocalToGlobalMappingApply(matis->mapping,k,&mat_graph->queue[mat_graph->cptr[i]],&queue_in_global_numbering[mat_graph->cptr[i]]);CHKERRQ(ierr);
      ierr = PetscSortIntWithArray(k,&queue_in_global_numbering[mat_graph->cptr[i]],&mat_graph->queue[mat_graph->cptr[i]]);CHKERRQ(ierr);
    }
    /* allocate some space */
    ierr = MPI_Comm_rank(interface_comm,&my_rank);CHKERRQ(ierr);
    ierr = PetscMalloc((where_values+1)*sizeof(PetscInt),&cum_recv_counts);CHKERRQ(ierr);
    ierr = PetscMemzero(cum_recv_counts,(where_values+1)*sizeof(PetscInt));CHKERRQ(ierr);
    ierr = PetscMalloc(where_values*sizeof(PetscInt),&where_to_nodes_indices);CHKERRQ(ierr);
    /* first count how many neighbours per connected component I will receive from */
    cum_recv_counts[0]=0;
    for (i=1;i<where_values+1;i++){
      j=0;
      while(mat_graph->where[j] != i) { j++; }
      where_to_nodes_indices[i-1]=j;
      if (mat_graph->neighbours_set[j][0]!=-1) { cum_recv_counts[i]=cum_recv_counts[i-1]+mat_graph->count[j]; } /* We don't want sends/recvs_to/from_self -> here I don't count myself  */
      else { cum_recv_counts[i]=cum_recv_counts[i-1]+mat_graph->count[j]-1; }
    }
    ierr = PetscMalloc(2*cum_recv_counts[where_values]*sizeof(PetscMPIInt),&recv_buffer_where);CHKERRQ(ierr);
    ierr = PetscMalloc(cum_recv_counts[where_values]*sizeof(MPI_Request),&send_requests);CHKERRQ(ierr);
    ierr = PetscMalloc(cum_recv_counts[where_values]*sizeof(MPI_Request),&recv_requests);CHKERRQ(ierr);
    for (i=0;i<cum_recv_counts[where_values];i++) { 
      send_requests[i]=MPI_REQUEST_NULL;
      recv_requests[i]=MPI_REQUEST_NULL;
    }
    /* exchange with my neighbours the number of my connected components on the shared interface */
    for (i=0;i<where_values;i++){
      j=where_to_nodes_indices[i];
      k = (mat_graph->neighbours_set[j][0] == -1 ?  1 : 0);
      for (;k<mat_graph->count[j];k++){
        ierr = MPI_Isend(&mat_graph->where_ncmps[i],1,MPIU_INT,mat_graph->neighbours_set[j][k],(my_rank+1)*mat_graph->count[j],interface_comm,&send_requests[sum_requests]);CHKERRQ(ierr); 
        ierr = MPI_Irecv(&recv_buffer_where[sum_requests],1,MPIU_INT,mat_graph->neighbours_set[j][k],(mat_graph->neighbours_set[j][k]+1)*mat_graph->count[j],interface_comm,&recv_requests[sum_requests]);CHKERRQ(ierr); 
        sum_requests++;
      }
    }
    ierr = MPI_Waitall(sum_requests,recv_requests,MPI_STATUSES_IGNORE);CHKERRQ(ierr);
    ierr = MPI_Waitall(sum_requests,send_requests,MPI_STATUSES_IGNORE);CHKERRQ(ierr);
    /* determine the connected component I need to adapt */
    ierr = PetscMalloc(where_values*sizeof(PetscInt),&where_cc_adapt);CHKERRQ(ierr);
    ierr = PetscMemzero(where_cc_adapt,where_values*sizeof(PetscInt));CHKERRQ(ierr);
    for (i=0;i<where_values;i++){
      for (j=cum_recv_counts[i];j<cum_recv_counts[i+1];j++){
        /* The first condition is natural (i.e someone has a different number of cc than me), the second one is just to be safe */
        if ( mat_graph->where_ncmps[i]!=recv_buffer_where[j] || mat_graph->where_ncmps[i] > 1 ) { 
          where_cc_adapt[i]=PETSC_TRUE;
          break;
        }
      }
    }
    buffer_size = 0;
    for (i=0;i<where_values;i++) {
      if (where_cc_adapt[i]) {
        for (j=i;j<mat_graph->ncmps;j++) {
          if (mat_graph->where[mat_graph->queue[mat_graph->cptr[j]]] == i+1) { /* WARNING -> where values goes from 1 to where_values included */
            buffer_size += 1 + mat_graph->cptr[j+1]-mat_graph->cptr[j];
          }
        }
      }
    }  
    ierr = PetscMalloc(buffer_size*sizeof(PetscMPIInt),&send_buffer);CHKERRQ(ierr);
    /* now get from neighbours their ccs (in global numbering) and adapt them (in case it is needed) */
    /* first determine how much data to send (size of each queue plus the global indices) and communicate it to neighbours */
    ierr = PetscMalloc(where_values*sizeof(PetscInt),&sizes_of_sends);CHKERRQ(ierr);
    ierr = PetscMemzero(sizes_of_sends,where_values*sizeof(PetscInt));CHKERRQ(ierr);
    sum_requests=0;
    start_of_send=0;
    start_of_recv=cum_recv_counts[where_values];
    for (i=0;i<where_values;i++) {
      if (where_cc_adapt[i]) {
        size_of_send=0;
        for (j=i;j<mat_graph->ncmps;j++) {
          if (mat_graph->where[mat_graph->queue[mat_graph->cptr[j]]] == i+1) { /* WARNING -> where values goes from 1 to where_values included */
            send_buffer[start_of_send+size_of_send]=mat_graph->cptr[j+1]-mat_graph->cptr[j];
            size_of_send+=1;
            for (k=0;k<mat_graph->cptr[j+1]-mat_graph->cptr[j];k++) {
              send_buffer[start_of_send+size_of_send+k]=queue_in_global_numbering[mat_graph->cptr[j]+k];
            }
            size_of_send=size_of_send+mat_graph->cptr[j+1]-mat_graph->cptr[j];
          }
        }
        j = where_to_nodes_indices[i];
        k = (mat_graph->neighbours_set[j][0] == -1 ?  1 : 0);
        sizes_of_sends[i]=size_of_send;
        for (;k<mat_graph->count[j];k++){
          ierr = MPI_Isend(&sizes_of_sends[i],1,MPIU_INT,mat_graph->neighbours_set[j][k],(my_rank+1)*mat_graph->count[j],interface_comm,&send_requests[sum_requests]);CHKERRQ(ierr); 
          ierr = MPI_Irecv(&recv_buffer_where[sum_requests+start_of_recv],1,MPIU_INT,mat_graph->neighbours_set[j][k],(mat_graph->neighbours_set[j][k]+1)*mat_graph->count[j],interface_comm,&recv_requests[sum_requests]);CHKERRQ(ierr);
          sum_requests++;
        }
        start_of_send+=size_of_send;
      }
    }
    ierr = MPI_Waitall(sum_requests,send_requests,MPI_STATUSES_IGNORE);CHKERRQ(ierr);
    ierr = MPI_Waitall(sum_requests,recv_requests,MPI_STATUSES_IGNORE);CHKERRQ(ierr);
    buffer_size=0;
    for (k=0;k<sum_requests;k++) { buffer_size+=recv_buffer_where[start_of_recv+k]; }
    ierr = PetscMalloc(buffer_size*sizeof(PetscMPIInt),&recv_buffer);CHKERRQ(ierr);
    /* now exchange the data */
    start_of_recv=0;
    start_of_send=0;
    sum_requests=0;
    for (i=0;i<where_values;i++) {
      if (where_cc_adapt[i]) {
        size_of_send = sizes_of_sends[i];
        j = where_to_nodes_indices[i];
        k = (mat_graph->neighbours_set[j][0] == -1 ?  1 : 0);
        for (;k<mat_graph->count[j];k++){
          ierr = MPI_Isend(&send_buffer[start_of_send],size_of_send,MPIU_INT,mat_graph->neighbours_set[j][k],(my_rank+1)*mat_graph->count[j],interface_comm,&send_requests[sum_requests]);CHKERRQ(ierr);
          size_of_recv=recv_buffer_where[cum_recv_counts[where_values]+sum_requests];
          ierr = MPI_Irecv(&recv_buffer[start_of_recv],size_of_recv,MPIU_INT,mat_graph->neighbours_set[j][k],(mat_graph->neighbours_set[j][k]+1)*mat_graph->count[j],interface_comm,&recv_requests[sum_requests]);CHKERRQ(ierr);
          start_of_recv+=size_of_recv;
          sum_requests++;
        }
        start_of_send+=size_of_send;
      }
    }
    ierr = MPI_Waitall(sum_requests,recv_requests,MPI_STATUSES_IGNORE);CHKERRQ(ierr);
    ierr = MPI_Waitall(sum_requests,send_requests,MPI_STATUSES_IGNORE);CHKERRQ(ierr);
    ierr = PetscMalloc(buffer_size*sizeof(PetscInt),&petsc_buffer);CHKERRQ(ierr);
    for (k=0;k<start_of_recv;k++) { petsc_buffer[k]=(PetscInt)recv_buffer[k]; }
    for (j=0;j<buffer_size;) {
       ierr = ISGlobalToLocalMappingApply(matis->mapping,IS_GTOLM_MASK,petsc_buffer[j],&petsc_buffer[j+1],&petsc_buffer[j],&petsc_buffer[j+1]);CHKERRQ(ierr);
       k=petsc_buffer[j]+1;
       j+=k;
    }
    sum_requests=cum_recv_counts[where_values];
    start_of_recv=0;
    ierr = PetscMalloc(mat_graph->nvtxs*sizeof(PetscInt),&nodes_to_temp_buffer_indices);CHKERRQ(ierr);
    global_where_counter=0;
    for (i=0;i<where_values;i++){
      if (where_cc_adapt[i]){
        temp_buffer_size=0;
        /* find nodes on the shared interface we need to adapt */
        for (j=0;j<mat_graph->nvtxs;j++){
          if (mat_graph->where[j]==i+1) {
            nodes_to_temp_buffer_indices[j]=temp_buffer_size;
            temp_buffer_size++;
          } else {
            nodes_to_temp_buffer_indices[j]=-1;
          } 
        }
        /* allocate some temporary space */
        ierr = PetscMalloc(temp_buffer_size*sizeof(PetscInt*),&temp_buffer);CHKERRQ(ierr);
        ierr = PetscMalloc(temp_buffer_size*(cum_recv_counts[i+1]-cum_recv_counts[i])*sizeof(PetscInt),&temp_buffer[0]);CHKERRQ(ierr);
        ierr = PetscMemzero(temp_buffer[0],temp_buffer_size*(cum_recv_counts[i+1]-cum_recv_counts[i])*sizeof(PetscInt));CHKERRQ(ierr);
        for (j=1;j<temp_buffer_size;j++){
          temp_buffer[j]=temp_buffer[j-1]+cum_recv_counts[i+1]-cum_recv_counts[i];
        }
        /* analyze contributions from neighbouring subdomains for i-th conn comp 
           temp buffer structure:
           supposing part of the interface has dimension 5 (global nodes 0,1,2,3,4)
           3 neighs procs with structured connected components:
             neigh 0: [0 1 4], [2 3];  (2 connected components)
             neigh 1: [0 1], [2 3 4];  (2 connected components)
             neigh 2: [0 4], [1], [2 3]; (3 connected components)
           tempbuffer (row-oriented) should be filled as:
             [ 0, 0, 0;
               0, 0, 1;
               1, 1, 2;
               1, 1, 2;
               0, 1, 0; ];
           This way we can simply recover the resulting structure account for possible intersections of ccs among neighs.
           The mat_graph->where array will be modified to reproduce the following 4 connected components [0], [1], [2 3], [4];
                                                                                                                                   */     
        for (j=0;j<cum_recv_counts[i+1]-cum_recv_counts[i];j++) {
          ins_val=0;
          size_of_recv=recv_buffer_where[sum_requests];  /* total size of recv from neighs */
          for (buffer_size=0;buffer_size<size_of_recv;) {  /* loop until all data from neighs has been taken into account */
            for (k=1;k<petsc_buffer[buffer_size+start_of_recv]+1;k++) { /* filling properly temp_buffer using data from a single recv */
              temp_buffer[ nodes_to_temp_buffer_indices[ petsc_buffer[ start_of_recv+buffer_size+k ] ] ][j]=ins_val;
            }
            buffer_size+=k;
            ins_val++;
          }
          start_of_recv+=size_of_recv;
          sum_requests++;
        }
        ierr = PetscMalloc(temp_buffer_size*sizeof(PetscInt),&add_to_where);CHKERRQ(ierr);
        ierr = PetscMemzero(add_to_where,temp_buffer_size*sizeof(PetscInt));CHKERRQ(ierr);
        for (j=0;j<temp_buffer_size;j++){
          if (!add_to_where[j]){ /* found a new cc  */
            global_where_counter++;
            add_to_where[j]=global_where_counter;
            for (k=j+1;k<temp_buffer_size;k++){ /* check for other nodes in new cc */
              same_set=PETSC_TRUE;
              for (s=0;s<cum_recv_counts[i+1]-cum_recv_counts[i];s++){
                if (temp_buffer[j][s]!=temp_buffer[k][s]) {
                  same_set=PETSC_FALSE;
                  break;
                }
              }
              if (same_set) { add_to_where[k]=global_where_counter; }
            }
          }
        }
        /* insert new data in where array */
        temp_buffer_size=0;
        for (j=0;j<mat_graph->nvtxs;j++){
          if (mat_graph->where[j]==i+1) {
            mat_graph->where[j]=where_values+add_to_where[temp_buffer_size];
            temp_buffer_size++;
          }
        }
        ierr = PetscFree(temp_buffer[0]);CHKERRQ(ierr);
        ierr = PetscFree(temp_buffer);CHKERRQ(ierr);
        ierr = PetscFree(add_to_where);CHKERRQ(ierr);
      }
    }
    ierr = PetscFree(nodes_to_temp_buffer_indices);CHKERRQ(ierr);
    ierr = PetscFree(sizes_of_sends);CHKERRQ(ierr);
    ierr = PetscFree(send_requests);CHKERRQ(ierr);
    ierr = PetscFree(recv_requests);CHKERRQ(ierr);
    ierr = PetscFree(petsc_buffer);CHKERRQ(ierr);
    ierr = PetscFree(recv_buffer);CHKERRQ(ierr);
    ierr = PetscFree(recv_buffer_where);CHKERRQ(ierr);
    ierr = PetscFree(send_buffer);CHKERRQ(ierr);
    ierr = PetscFree(cum_recv_counts);CHKERRQ(ierr);
    ierr = PetscFree(where_to_nodes_indices);CHKERRQ(ierr);
    ierr = PetscFree(where_cc_adapt);CHKERRQ(ierr);
    /* We are ready to evaluate consistent connected components on each part of the shared interface */
    if (global_where_counter) {
      for (i=0;i<mat_graph->nvtxs;i++){ mat_graph->touched[i]=PETSC_FALSE; }
      global_where_counter=0;
      for (i=0;i<mat_graph->nvtxs;i++){
        if (mat_graph->where[i] && !mat_graph->touched[i]) {
          global_where_counter++;
          for (j=i+1;j<mat_graph->nvtxs;j++){
            if (!mat_graph->touched[j] && mat_graph->where[j]==mat_graph->where[i]) {
              mat_graph->where[j]=global_where_counter;
              mat_graph->touched[j]=PETSC_TRUE;
            }
          }
          mat_graph->where[i]=global_where_counter;
          mat_graph->touched[i]=PETSC_TRUE;
        }
      }
      where_values=global_where_counter;
    }
    if (global_where_counter) {
      ierr = PetscMemzero(mat_graph->cptr,(mat_graph->nvtxs+1)*sizeof(PetscInt));CHKERRQ(ierr);
      ierr = PetscMemzero(mat_graph->queue,mat_graph->nvtxs*sizeof(PetscInt));CHKERRQ(ierr);
      ierr = PetscFree(mat_graph->where_ncmps);CHKERRQ(ierr);
      ierr = PetscMalloc(where_values*sizeof(PetscMPIInt),&mat_graph->where_ncmps);CHKERRQ(ierr);
      ierr = PCBDDCFindConnectedComponents(mat_graph, where_values); 
    }
  } /* Finished adapting interface */
  /* For consistency among neughbouring procs, I need to sort (by global ordering) each connected component */
  for (i=0;i<mat_graph->ncmps;i++) {
    k = mat_graph->cptr[i+1]-mat_graph->cptr[i];
    ierr = ISLocalToGlobalMappingApply(matis->mapping,k,&mat_graph->queue[mat_graph->cptr[i]],&queue_in_global_numbering[mat_graph->cptr[i]]);CHKERRQ(ierr);
    ierr = PetscSortIntWithArray(k,&queue_in_global_numbering[mat_graph->cptr[i]],&mat_graph->queue[mat_graph->cptr[i]]);CHKERRQ(ierr);
  }

  PetscInt nfc=0;
  PetscInt nec=0;
  PetscInt nvc=0;
  PetscBool twodim_flag=PETSC_FALSE;
  for (i=0; i<mat_graph->ncmps; i++) {
    if ( mat_graph->cptr[i+1]-mat_graph->cptr[i] > vertex_size ){
      if (mat_graph->count[mat_graph->queue[mat_graph->cptr[i]]]==1){ /* 1 neigh Neumann fake included */
        nfc++;
      } else { /* note that nec will be zero in 2d */
        nec++;
      }
    } else {
      nvc+=mat_graph->cptr[i+1]-mat_graph->cptr[i];
    }
  }
  if (!nec) { /* we are in a 2d case -> no faces, only edges */
    nec = nfc;
    nfc = 0;
    twodim_flag = PETSC_TRUE;
  }
  /* allocate IS arrays for faces, edges. Vertices need a single index set. */
  k=0;
  for (i=0; i<mat_graph->ncmps; i++) {
    j=mat_graph->cptr[i+1]-mat_graph->cptr[i];
    if ( j > k) {
      k=j;
    } 
    if (j<=vertex_size) {
      k+=vertex_size;
    }
  }
  ierr = PetscMalloc(k*sizeof(PetscInt),&auxis);CHKERRQ(ierr);
  if (!pcbddc->vertices_flag && !pcbddc->edges_flag) {
    ierr = PetscMalloc(nfc*sizeof(IS),&pcbddc->ISForFaces);CHKERRQ(ierr);
    use_faces=PETSC_TRUE;
  }
  if (!pcbddc->vertices_flag && !pcbddc->faces_flag) {
    ierr = PetscMalloc(nec*sizeof(IS),&pcbddc->ISForEdges);CHKERRQ(ierr);
    use_edges=PETSC_TRUE;
  }
  nfc=0;
  nec=0;
  for (i=0; i<mat_graph->ncmps; i++) {
    if ( mat_graph->cptr[i+1]-mat_graph->cptr[i] > vertex_size ){
      for (j=0;j<mat_graph->cptr[i+1]-mat_graph->cptr[i];j++) {
        auxis[j]=mat_graph->queue[mat_graph->cptr[i]+j];
      }
      if (mat_graph->count[mat_graph->queue[mat_graph->cptr[i]]]==1){
        if (twodim_flag) {
          if (use_edges) {
            ierr = ISCreateGeneral(PETSC_COMM_SELF,j,auxis,PETSC_COPY_VALUES,&pcbddc->ISForEdges[nec]);CHKERRQ(ierr);
            nec++;
          }
        } else {
          if (use_faces) {
            ierr = ISCreateGeneral(PETSC_COMM_SELF,j,auxis,PETSC_COPY_VALUES,&pcbddc->ISForFaces[nfc]);CHKERRQ(ierr);
            nfc++;
          }
        } 
      } else {
        if (use_edges) {
          ierr = ISCreateGeneral(PETSC_COMM_SELF,j,auxis,PETSC_COPY_VALUES,&pcbddc->ISForEdges[nec]);CHKERRQ(ierr);
          nec++;
        }
      }
    }
  }
  pcbddc->n_ISForFaces=nfc;
  pcbddc->n_ISForEdges=nec;
  nvc=0;
  if ( !pcbddc->constraints_flag ) {
    for (i=0; i<mat_graph->ncmps; i++) {
      if ( mat_graph->cptr[i+1]-mat_graph->cptr[i] <= vertex_size ){
        for ( j=mat_graph->cptr[i];j<mat_graph->cptr[i+1];j++) {
          auxis[nvc]=mat_graph->queue[j];
          nvc++;
        }
      }
    }
  }
  /* sort vertex set (by local ordering) */
  ierr = PetscSortInt(nvc,auxis);CHKERRQ(ierr); 
  ierr = ISCreateGeneral(PETSC_COMM_SELF,nvc,auxis,PETSC_COPY_VALUES,&pcbddc->ISForVertices);CHKERRQ(ierr); 
  if (pcbddc->dbg_flag) {
    ierr = PetscViewerASCIISynchronizedPrintf(viewer,"--------------------------------------------------------------\n");CHKERRQ(ierr);
    ierr = PetscViewerASCIISynchronizedPrintf(viewer,"Details from PCBDDCManageLocalBoundaries for subdomain %04d\n",PetscGlobalRank);CHKERRQ(ierr);
    ierr = PetscViewerASCIISynchronizedPrintf(viewer,"Matrix graph has %d connected components", mat_graph->ncmps);CHKERRQ(ierr);
    for (i=0;i<mat_graph->ncmps;i++) {
      ierr = PetscViewerASCIISynchronizedPrintf(viewer,"\nDetails for connected component number %02d: size %04d, count %01d. Nodes follow.\n",
             i,mat_graph->cptr[i+1]-mat_graph->cptr[i],mat_graph->count[mat_graph->queue[mat_graph->cptr[i]]]);CHKERRQ(ierr);
      ierr = PetscViewerASCIISynchronizedPrintf(viewer,"subdomains: ");
      for (j=0;j<mat_graph->count[mat_graph->queue[mat_graph->cptr[i]]]; j++) {
        ierr = PetscViewerASCIISynchronizedPrintf(viewer,"%d ",mat_graph->neighbours_set[mat_graph->queue[mat_graph->cptr[i]]][j]);
      }
      ierr = PetscViewerASCIISynchronizedPrintf(viewer,"\n");
      for (j=mat_graph->cptr[i]; j<mat_graph->cptr[i+1]; j++){
        ierr = PetscViewerASCIISynchronizedPrintf(viewer,"%d (%d), ",mat_graph->queue[j],queue_in_global_numbering[j]);CHKERRQ(ierr);
      }
    }
    ierr = PetscViewerASCIISynchronizedPrintf(viewer,"\n--------------------------------------------------------------\n");CHKERRQ(ierr);
    ierr = PetscViewerASCIISynchronizedPrintf(viewer,"Subdomain %04d detected %02d local vertices\n",PetscGlobalRank,nvc);CHKERRQ(ierr);
    ierr = PetscViewerASCIISynchronizedPrintf(viewer,"Subdomain %04d detected %02d local faces\n",PetscGlobalRank,nfc);CHKERRQ(ierr);
    ierr = PetscViewerASCIISynchronizedPrintf(viewer,"Subdomain %04d detected %02d local edges\n",PetscGlobalRank,nec);CHKERRQ(ierr);
    ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
  }
  ierr = PetscFree(auxis);CHKERRQ(ierr);
  ierr = PetscFree(queue_in_global_numbering);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */

/* The following code has been adapted from function IsConnectedSubdomain contained 
   in source file contig.c of METIS library (version 5.0.1)
   It finds connected components of each partition labeled from 1 to n_dist  */
                                
#undef __FUNCT__  
#define __FUNCT__ "PCBDDCFindConnectedComponents"
static PetscErrorCode PCBDDCFindConnectedComponents(PCBDDCGraph graph, PetscInt n_dist )
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

  for (i=0; i<nvtxs; i++) { 
    touched[i] = PETSC_FALSE;
  }

  cum_queue=0;
  ncmps=0;

  for (n=0; n<n_dist; n++) {
    pid = n+1;  /* partition labeled by 0 is discarded */
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
    graph->where_ncmps[n] = ncmps_pid;
  }
  graph->ncmps = ncmps;

  PetscFunctionReturn(0);
}
