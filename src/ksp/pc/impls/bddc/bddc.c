/* TODOLIST
   DofSplitting and DM attached to pc?
   Change SetNeumannBoundaries to SetNeumannBoundariesLocal and provide new SetNeumannBoundaries (same Dirichlet)
   change how to deal with the coarse problem (PCBDDCSetCoarseEnvironment):
     - simplify coarse problem structure -> PCBDDC or PCREDUDANT, nothing else -> same comm for all levels?
     - remove coarse enums and allow use of PCBDDCGetCoarseKSP
     - remove metis dependency -> use MatPartitioning for multilevel -> Assemble serial adjacency in PCBDDCAnalyzeInterface?
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
#include "bddcprivate.h"
#include <petscblaslapack.h>

/* prototypes for static functions contained in bddc.c */
static PetscErrorCode PCBDDCSetUseExactDirichlet(PC,PetscBool);
static PetscErrorCode PCBDDCSetLevel(PC,PetscInt);
static PetscErrorCode PCBDDCCoarseSetUp(PC);
static PetscErrorCode PCBDDCSetUpCoarseEnvironment(PC,PetscScalar*);

/* -------------------------------------------------------------------------- */
#undef __FUNCT__
#define __FUNCT__ "PCSetFromOptions_BDDC"
PetscErrorCode PCSetFromOptions_BDDC(PC pc)
{
  PC_BDDC        *pcbddc = (PC_BDDC*)pc->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscOptionsHead("BDDC options");CHKERRQ(ierr);
  /* Verbose debugging of main data structures */
  ierr = PetscOptionsInt("-pc_bddc_check_level"       ,"Verbose (debugging) output for PCBDDC"                       ,"none",pcbddc->dbg_flag      ,&pcbddc->dbg_flag      ,NULL);CHKERRQ(ierr);
  /* Some customization for default primal space */
  ierr = PetscOptionsBool("-pc_bddc_vertices_only"   ,"Use only vertices in coarse space (i.e. discard constraints)","none",pcbddc->vertices_flag   ,&pcbddc->vertices_flag   ,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-pc_bddc_constraints_only","Use only constraints in coarse space (i.e. discard vertices)","none",pcbddc->constraints_flag,&pcbddc->constraints_flag,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-pc_bddc_faces_only"      ,"Use only faces among constraints of coarse space (i.e. discard edges)"         ,"none",pcbddc->faces_flag      ,&pcbddc->faces_flag      ,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-pc_bddc_edges_only"      ,"Use only edges among constraints of coarse space (i.e. discard faces)"         ,"none",pcbddc->edges_flag      ,&pcbddc->edges_flag      ,NULL);CHKERRQ(ierr);
  /* Coarse solver context */
  static const char * const avail_coarse_problems[] = {"sequential","replicated","parallel","multilevel","CoarseProblemType","PC_BDDC_",0}; /*order of choiches depends on ENUM defined in bddc.h */
  ierr = PetscOptionsEnum("-pc_bddc_coarse_problem_type","Set coarse problem type","none",avail_coarse_problems,(PetscEnum)pcbddc->coarse_problem_type,(PetscEnum*)&pcbddc->coarse_problem_type,NULL);CHKERRQ(ierr);
  /* Two different application of BDDC to the whole set of dofs, internal and interface */
  ierr = PetscOptionsBool("-pc_bddc_switch_preconditioning_type","Switch between M_2 (default) and M_3 preconditioners (as defined by Dohrmann)","none",pcbddc->inexact_prec_type,&pcbddc->inexact_prec_type,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-pc_bddc_use_change_of_basis","Use change of basis approach for primal space","none",pcbddc->use_change_of_basis,&pcbddc->use_change_of_basis,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-pc_bddc_use_change_on_faces","Use change of basis approach for face constraints","none",pcbddc->use_change_on_faces,&pcbddc->use_change_on_faces,NULL);CHKERRQ(ierr);
  if (!pcbddc->use_change_of_basis) {
    pcbddc->use_change_on_faces = PETSC_FALSE;
  }
  ierr = PetscOptionsInt("-pc_bddc_coarsening_ratio","Set coarsening ratio used in multilevel coarsening","none",pcbddc->coarsening_ratio,&pcbddc->coarsening_ratio,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-pc_bddc_max_levels","Set maximum number of levels for multilevel","none",pcbddc->max_levels,&pcbddc->max_levels,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-pc_bddc_use_deluxe_scaling","Use deluxe scaling for BDDC","none",pcbddc->use_deluxe_scaling,&pcbddc->use_deluxe_scaling,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
/* -------------------------------------------------------------------------- */
#undef __FUNCT__
#define __FUNCT__ "PCBDDCSetPrimalVerticesLocalIS_BDDC"
static PetscErrorCode PCBDDCSetPrimalVerticesLocalIS_BDDC(PC pc, IS PrimalVertices)
{
  PC_BDDC        *pcbddc = (PC_BDDC*)pc->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = ISDestroy(&pcbddc->user_primal_vertices);CHKERRQ(ierr);
  ierr = PetscObjectReference((PetscObject)PrimalVertices);CHKERRQ(ierr);
  pcbddc->user_primal_vertices = PrimalVertices;
  PetscFunctionReturn(0);
}
#undef __FUNCT__
#define __FUNCT__ "PCBDDCSetPrimalVerticesLocalIS"
/*@
 PCBDDCSetPrimalVerticesLocalIS - Set user defined primal vertices in PCBDDC.

   Not collective

   Input Parameters:
+  pc - the preconditioning context
-  PrimalVertices - index sets of primal vertices in local numbering

   Level: intermediate

   Notes:

.seealso: PCBDDC
@*/
PetscErrorCode PCBDDCSetPrimalVerticesLocalIS(PC pc, IS PrimalVertices)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscValidHeaderSpecific(PrimalVertices,IS_CLASSID,2);
  ierr = PetscTryMethod(pc,"PCBDDCSetPrimalVerticesLocalIS_C",(PC,IS),(pc,PrimalVertices));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
/* -------------------------------------------------------------------------- */
#undef __FUNCT__
#define __FUNCT__ "PCBDDCSetCoarseProblemType_BDDC"
static PetscErrorCode PCBDDCSetCoarseProblemType_BDDC(PC pc, CoarseProblemType CPT)
{
  PC_BDDC  *pcbddc = (PC_BDDC*)pc->data;

  PetscFunctionBegin;
  pcbddc->coarse_problem_type = CPT;
  PetscFunctionReturn(0);
}

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
#undef __FUNCT__
#define __FUNCT__ "PCBDDCSetCoarseningRatio_BDDC"
static PetscErrorCode PCBDDCSetCoarseningRatio_BDDC(PC pc,PetscInt k)
{
  PC_BDDC  *pcbddc = (PC_BDDC*)pc->data;

  PetscFunctionBegin;
  pcbddc->coarsening_ratio=k;
  PetscFunctionReturn(0);
}

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

#undef __FUNCT__
#define __FUNCT__ "PCBDDCSetMaxLevels_BDDC"
static PetscErrorCode PCBDDCSetMaxLevels_BDDC(PC pc,PetscInt max_levels)
{
  PC_BDDC  *pcbddc = (PC_BDDC*)pc->data;

  PetscFunctionBegin;
  pcbddc->max_levels=max_levels;
  PetscFunctionReturn(0);
}

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
  PetscValidHeaderSpecific(NullSpace,MAT_NULLSPACE_CLASSID,2);
  ierr = PetscTryMethod(pc,"PCBDDCSetNullSpace_C",(PC,MatNullSpace),(pc,NullSpace));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
/* -------------------------------------------------------------------------- */

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

#undef __FUNCT__
#define __FUNCT__ "PCBDDCSetDirichletBoundaries"
/*@
 PCBDDCSetDirichletBoundaries - Set index set defining subdomain part (in local ordering)
                              of Dirichlet boundaries for the global problem.

   Not collective

   Input Parameters:
+  pc - the preconditioning context
-  DirichletBoundaries - sequential index set defining the subdomain part of Dirichlet boundaries (can be NULL)

   Level: intermediate

   Notes:

.seealso: PCBDDC
@*/
PetscErrorCode PCBDDCSetDirichletBoundaries(PC pc,IS DirichletBoundaries)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscValidHeaderSpecific(DirichletBoundaries,IS_CLASSID,2);
  ierr = PetscTryMethod(pc,"PCBDDCSetDirichletBoundaries_C",(PC,IS),(pc,DirichletBoundaries));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
/* -------------------------------------------------------------------------- */

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

#undef __FUNCT__
#define __FUNCT__ "PCBDDCSetNeumannBoundaries"
/*@
 PCBDDCSetNeumannBoundaries - Set index set defining subdomain part (in local ordering)
                              of Neumann boundaries for the global problem.

   Not collective

   Input Parameters:
+  pc - the preconditioning context
-  NeumannBoundaries - sequential index set defining the subdomain part of Neumann boundaries (can be NULL)

   Level: intermediate

   Notes:

.seealso: PCBDDC
@*/
PetscErrorCode PCBDDCSetNeumannBoundaries(PC pc,IS NeumannBoundaries)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscValidHeaderSpecific(NeumannBoundaries,IS_CLASSID,2);
  ierr = PetscTryMethod(pc,"PCBDDCSetNeumannBoundaries_C",(PC,IS),(pc,NeumannBoundaries));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
/* -------------------------------------------------------------------------- */

#undef __FUNCT__
#define __FUNCT__ "PCBDDCGetDirichletBoundaries_BDDC"
static PetscErrorCode PCBDDCGetDirichletBoundaries_BDDC(PC pc,IS *DirichletBoundaries)
{
  PC_BDDC  *pcbddc = (PC_BDDC*)pc->data;

  PetscFunctionBegin;
  *DirichletBoundaries = pcbddc->DirichletBoundaries;
  PetscFunctionReturn(0);
}

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

#undef __FUNCT__
#define __FUNCT__ "PCBDDCGetNeumannBoundaries_BDDC"
static PetscErrorCode PCBDDCGetNeumannBoundaries_BDDC(PC pc,IS *NeumannBoundaries)
{
  PC_BDDC  *pcbddc = (PC_BDDC*)pc->data;

  PetscFunctionBegin;
  *NeumannBoundaries = pcbddc->NeumannBoundaries;
  PetscFunctionReturn(0);
}

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

#undef __FUNCT__
#define __FUNCT__ "PCBDDCSetLocalAdjacencyGraph_BDDC"
static PetscErrorCode PCBDDCSetLocalAdjacencyGraph_BDDC(PC pc, PetscInt nvtxs,const PetscInt xadj[],const PetscInt adjncy[], PetscCopyMode copymode)
{
  PC_BDDC        *pcbddc = (PC_BDDC*)pc->data;
  PCBDDCGraph    mat_graph = pcbddc->mat_graph;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* free old CSR */
  ierr = PCBDDCGraphResetCSR(mat_graph);CHKERRQ(ierr);
  /* get CSR into graph structure */
  if (copymode == PETSC_COPY_VALUES) {
    ierr = PetscMalloc((nvtxs+1)*sizeof(PetscInt),&mat_graph->xadj);CHKERRQ(ierr);
    ierr = PetscMalloc(xadj[nvtxs]*sizeof(PetscInt),&mat_graph->adjncy);CHKERRQ(ierr);
    ierr = PetscMemcpy(mat_graph->xadj,xadj,(nvtxs+1)*sizeof(PetscInt));CHKERRQ(ierr);
    ierr = PetscMemcpy(mat_graph->adjncy,adjncy,xadj[nvtxs]*sizeof(PetscInt));CHKERRQ(ierr);
  } else if (copymode == PETSC_OWN_POINTER) {
    mat_graph->xadj = (PetscInt*)xadj;
    mat_graph->adjncy = (PetscInt*)adjncy;
  }
  mat_graph->nvtxs_csr = nvtxs;
  PetscFunctionReturn(0);
}

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
  void (*f)(void) = 0;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscValidIntPointer(xadj,3);
  PetscValidIntPointer(xadj,4);
  if (copymode != PETSC_COPY_VALUES && copymode != PETSC_OWN_POINTER) {
    SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_SUP,"Unsupported copy mode %d in %s\n",copymode,__FUNCT__);
  }
  ierr = PetscTryMethod(pc,"PCBDDCSetLocalAdjacencyGraph_C",(PC,PetscInt,const PetscInt[],const PetscInt[],PetscCopyMode),(pc,nvtxs,xadj,adjncy,copymode));CHKERRQ(ierr);
  /* free arrays if PCBDDC is not the PC type */
  ierr = PetscObjectQueryFunction((PetscObject)pc,"PCBDDCSetLocalAdjacencyGraph_C",&f);CHKERRQ(ierr);
  if (!f && copymode == PETSC_OWN_POINTER) {
    ierr = PetscFree(xadj);CHKERRQ(ierr);
    ierr = PetscFree(adjncy);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
/* -------------------------------------------------------------------------- */

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
  ierr = VecScatterEnd(matis->ctx,pcis->vec1_global,pcis->vec2_N,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterBegin(matis->ctx,used_vec,pcis->vec1_N,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(matis->ctx,used_vec,pcis->vec1_N,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = PCBDDCGetDirichletBoundaries(pc,&dirIS);CHKERRQ(ierr);
  if (dirIS) {
    ierr = ISGetSize(dirIS,&dirsize);CHKERRQ(ierr);
    ierr = VecGetArray(pcis->vec1_N,&array_x);CHKERRQ(ierr);
    ierr = VecGetArray(pcis->vec2_N,&array_diagonal);CHKERRQ(ierr);
    ierr = ISGetIndices(dirIS,(const PetscInt**)&is_indices);CHKERRQ(ierr);
    for (i=0; i<dirsize; i++) array_x[is_indices[i]] = array_diagonal[is_indices[i]];
    ierr = ISRestoreIndices(dirIS,(const PetscInt**)&is_indices);CHKERRQ(ierr);
    ierr = VecRestoreArray(pcis->vec2_N,&array_diagonal);CHKERRQ(ierr);
    ierr = VecRestoreArray(pcis->vec1_N,&array_x);CHKERRQ(ierr);
  }
  ierr = VecScatterBegin(matis->ctx,pcis->vec1_N,used_vec,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  ierr = VecScatterEnd(matis->ctx,pcis->vec1_N,used_vec,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);

  /* remove the computed solution from the rhs */
  ierr = VecScale(used_vec,-1.0);CHKERRQ(ierr);
  ierr = MatMultAdd(pc->pmat,used_vec,rhs,rhs);CHKERRQ(ierr);
  ierr = VecScale(used_vec,-1.0);CHKERRQ(ierr);

  /* store partially computed solution and set initial guess */
  if (x) {
    ierr = VecCopy(used_vec,pcbddc->temp_solution);CHKERRQ(ierr);
    ierr = VecSet(used_vec,0.0);CHKERRQ(ierr);
    if (pcbddc->use_exact_dirichlet && !pcbddc->coarse_psi_B) {
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
  if (pcbddc->use_change_of_basis) {
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
  }
  if (ksp && pcbddc->NullSpace) {
    ierr = MatNullSpaceRemove(pcbddc->NullSpace,used_vec);CHKERRQ(ierr);
    ierr = MatNullSpaceRemove(pcbddc->NullSpace,rhs);CHKERRQ(ierr);
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
  PC_IS          *pcis   = (PC_IS*)(pc->data);
  Mat_IS         *matis = (Mat_IS*)pc->pmat->data;
  Mat            temp_mat;

  PetscFunctionBegin;
  if (pcbddc->use_change_of_basis) {
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
  PC_BDDC*       pcbddc = (PC_BDDC*)pc->data;
  MatStructure   flag;
  PetscBool      computeis,computetopography,computesolvers;

  PetscFunctionBegin;
  /* the following lines of code should be replaced by a better logic between PCIS, PCNN, PCBDDC and other nonoverlapping preconditioners */
  /* For BDDC we need to define a local "Neumann" problem different to that defined in PCISSetup
     So, we set to pcnone the Neumann problem of pcis in order to avoid unneeded computation
     Also, we decide to directly build the (same) Dirichlet problem */
  ierr = PetscOptionsSetValue("-is_localN_pc_type","none");CHKERRQ(ierr);
  ierr = PetscOptionsSetValue("-is_localD_pc_type","none");CHKERRQ(ierr);
  /* Get stdout for dbg */
  if (pcbddc->dbg_flag && !pcbddc->dbg_viewer) {
    ierr = PetscViewerASCIIGetStdout(PetscObjectComm((PetscObject)pc),&pcbddc->dbg_viewer);CHKERRQ(ierr);
    ierr = PetscViewerASCIISynchronizedAllow(pcbddc->dbg_viewer,PETSC_TRUE);CHKERRQ(ierr);
  }
  /* first attempt to split work */
  if (pc->setupcalled) {
    computeis = PETSC_FALSE;
    ierr = PCGetOperators(pc,NULL,NULL,&flag);CHKERRQ(ierr);
    if (flag == SAME_PRECONDITIONER) {
      computetopography = PETSC_FALSE;
      computesolvers = PETSC_FALSE;
    } else if (flag == SAME_NONZERO_PATTERN) {
      computetopography = PETSC_FALSE;
      computesolvers = PETSC_TRUE;
    } else { /* DIFFERENT_NONZERO_PATTERN */
      computetopography = PETSC_TRUE;
      computesolvers = PETSC_TRUE;
    }
  } else {
    computeis = PETSC_TRUE;
    computetopography = PETSC_TRUE;
    computesolvers = PETSC_TRUE;
  }
  /* Set up all the "iterative substructuring" common block */
  if (computeis) {
    ierr = PCISSetUp(pc);CHKERRQ(ierr);
  }
  /* Analyze interface and set up local constraint and change of basis matrices */
  if (computetopography) {
    /* reset data */
    ierr = PCBDDCResetTopography(pc);CHKERRQ(ierr);
    ierr = PCBDDCAnalyzeInterface(pc);CHKERRQ(ierr);
    ierr = PCBDDCConstraintsSetUp(pc);CHKERRQ(ierr);
  }
  if (computesolvers) {
    /* reset data */
    ierr = PCBDDCResetSolvers(pc);CHKERRQ(ierr);
    ierr = PCBDDCScalingDestroy(pc);CHKERRQ(ierr);
    /* Create coarse and local stuffs used for evaluating action of preconditioner */
    ierr = PCBDDCCoarseSetUp(pc);CHKERRQ(ierr);
    ierr = PCBDDCScalingSetUp(pc);CHKERRQ(ierr);
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
   Added support for M_3 preconditioner in the reference article (code is active if pcbddc->inexact_prec_type = PETSC_TRUE) */

  PetscFunctionBegin;
  if (!pcbddc->use_exact_dirichlet || pcbddc->coarse_psi_B) {
    /* First Dirichlet solve */
    ierr = VecScatterBegin(pcis->global_to_D,r,pcis->vec1_D,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterEnd  (pcis->global_to_D,r,pcis->vec1_D,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = KSPSolve(pcbddc->ksp_D,pcis->vec1_D,pcis->vec2_D);CHKERRQ(ierr);
    /*
      Assembling right hand side for BDDC operator
      - pcis->vec1_D for the Dirichlet part (if needed, i.e. prec_flag=PETSC_TRUE)
      - pcis->vec1_B the interface part of the global vector z
    */
    ierr = VecScale(pcis->vec2_D,m_one);CHKERRQ(ierr);
    ierr = MatMult(pcis->A_BI,pcis->vec2_D,pcis->vec1_B);CHKERRQ(ierr);
    if (pcbddc->inexact_prec_type) { ierr = MatMultAdd(pcis->A_II,pcis->vec2_D,pcis->vec1_D,pcis->vec1_D);CHKERRQ(ierr); }
    ierr = VecScale(pcis->vec2_D,m_one);CHKERRQ(ierr);
    ierr = VecCopy(r,z);CHKERRQ(ierr);
    ierr = VecScatterBegin(pcis->global_to_B,pcis->vec1_B,z,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
    ierr = VecScatterEnd  (pcis->global_to_B,pcis->vec1_B,z,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
    ierr = PCBDDCScalingRestriction(pc,z,pcis->vec1_B);CHKERRQ(ierr);
  } else {
    ierr = VecSet(pcis->vec1_D,zero);CHKERRQ(ierr);
    ierr = VecSet(pcis->vec2_D,zero);CHKERRQ(ierr);
    ierr = PCBDDCScalingRestriction(pc,r,pcis->vec1_B);CHKERRQ(ierr);
  }

  /* Apply interface preconditioner
     input/output vecs: pcis->vec1_B and pcis->vec1_D */
  ierr = PCBDDCApplyInterfacePreconditioner(pc);CHKERRQ(ierr);

  /* Apply transpose of partition of unity operator */
  ierr = PCBDDCScalingExtension(pc,pcis->vec1_B,z);CHKERRQ(ierr);

  /* Second Dirichlet solve and assembling of output */
  ierr = VecScatterBegin(pcis->global_to_B,z,pcis->vec1_B,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd  (pcis->global_to_B,z,pcis->vec1_B,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = MatMult(pcis->A_IB,pcis->vec1_B,pcis->vec3_D);CHKERRQ(ierr);
  if (pcbddc->inexact_prec_type) { ierr = MatMultAdd(pcis->A_II,pcis->vec1_D,pcis->vec3_D,pcis->vec3_D);CHKERRQ(ierr); }
  ierr = KSPSolve(pcbddc->ksp_D,pcis->vec3_D,pcbddc->vec4_D);CHKERRQ(ierr);
  ierr = VecScale(pcbddc->vec4_D,m_one);CHKERRQ(ierr);
  if (pcbddc->inexact_prec_type) { ierr = VecAXPY (pcbddc->vec4_D,one,pcis->vec1_D);CHKERRQ(ierr); }
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* free data created by PCIS */
  ierr = PCISDestroy(pc);CHKERRQ(ierr);
  /* free BDDC custom data  */
  ierr = PCBDDCResetCustomization(pc);CHKERRQ(ierr);
  /* destroy objects related to topography */
  ierr = PCBDDCResetTopography(pc);CHKERRQ(ierr);
  /* free allocated graph structure */
  ierr = PetscFree(pcbddc->mat_graph);CHKERRQ(ierr);
  /* free data for scaling operator */
  ierr = PCBDDCScalingDestroy(pc);CHKERRQ(ierr);
  /* free solvers stuff */
  ierr = PCBDDCResetSolvers(pc);CHKERRQ(ierr);
  /* remove functions */
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCBDDCSetPrimalVerticesLocalIS_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCBDDCSetCoarseningRatio_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCBDDCSetMaxLevels_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCBDDCSetNullSpace_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCBDDCSetDirichletBoundaries_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCBDDCSetNeumannBoundaries_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCBDDCGetDirichletBoundaries_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCBDDCGetNeumannBoundaries_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCBDDCSetCoarseProblemType_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCBDDCSetDofsSplitting_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCBDDCSetLocalAdjacencyGraph_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCBDDCCreateFETIDPOperators_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCBDDCMatFETIDPGetRHS_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCBDDCMatFETIDPGetSolution_C",NULL);CHKERRQ(ierr);
  /* Free the private data structure */
  ierr = PetscFree(pc->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
/* -------------------------------------------------------------------------- */

#undef __FUNCT__
#define __FUNCT__ "PCBDDCMatFETIDPGetRHS_BDDC"
static PetscErrorCode PCBDDCMatFETIDPGetRHS_BDDC(Mat fetidp_mat, Vec standard_rhs, Vec fetidp_flux_rhs)
{
  FETIDPMat_ctx  mat_ctx;
  PC_IS*         pcis;
  PC_BDDC*       pcbddc;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatShellGetContext(fetidp_mat,&mat_ctx);CHKERRQ(ierr);
  pcis = (PC_IS*)mat_ctx->pc->data;
  pcbddc = (PC_BDDC*)mat_ctx->pc->data;

  /* change of basis for physical rhs if needed
     It also changes the rhs in case of dirichlet boundaries */
  (*mat_ctx->pc->ops->presolve)(mat_ctx->pc,NULL,standard_rhs,NULL);
  /* store vectors for computation of fetidp final solution */
  ierr = VecScatterBegin(pcis->global_to_D,standard_rhs,mat_ctx->temp_solution_D,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(pcis->global_to_D,standard_rhs,mat_ctx->temp_solution_D,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  /* scale rhs since it should be unassembled : TODO use counter scaling? (also below) */
  ierr = VecScatterBegin(pcis->global_to_B,standard_rhs,mat_ctx->temp_solution_B,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(pcis->global_to_B,standard_rhs,mat_ctx->temp_solution_B,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  /* Apply partition of unity */
  ierr = VecPointwiseMult(mat_ctx->temp_solution_B,pcis->D,mat_ctx->temp_solution_B);CHKERRQ(ierr);
  /* ierr = PCBDDCScalingRestriction(mat_ctx->pc,standard_rhs,mat_ctx->temp_solution_B);CHKERRQ(ierr); */
  if (!pcbddc->inexact_prec_type) {
    /* compute partially subassembled Schur complement right-hand side */
    ierr = KSPSolve(pcbddc->ksp_D,mat_ctx->temp_solution_D,pcis->vec1_D);CHKERRQ(ierr);
    ierr = MatMult(pcis->A_BI,pcis->vec1_D,pcis->vec1_B);CHKERRQ(ierr);
    ierr = VecAXPY(mat_ctx->temp_solution_B,-1.0,pcis->vec1_B);CHKERRQ(ierr);
    ierr = VecSet(standard_rhs,0.0);CHKERRQ(ierr);
    ierr = VecScatterBegin(pcis->global_to_B,mat_ctx->temp_solution_B,standard_rhs,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
    ierr = VecScatterEnd(pcis->global_to_B,mat_ctx->temp_solution_B,standard_rhs,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
    /* ierr = PCBDDCScalingRestriction(mat_ctx->pc,standard_rhs,mat_ctx->temp_solution_B);CHKERRQ(ierr); */
    ierr = VecScatterBegin(pcis->global_to_B,standard_rhs,mat_ctx->temp_solution_B,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterEnd(pcis->global_to_B,standard_rhs,mat_ctx->temp_solution_B,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecPointwiseMult(mat_ctx->temp_solution_B,pcis->D,mat_ctx->temp_solution_B);CHKERRQ(ierr);
  }
  /* BDDC rhs */
  ierr = VecCopy(mat_ctx->temp_solution_B,pcis->vec1_B);CHKERRQ(ierr);
  if (pcbddc->inexact_prec_type) {
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
  FETIDPMat_ctx  mat_ctx;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatShellGetContext(fetidp_mat,&mat_ctx);CHKERRQ(ierr);
  ierr = PetscTryMethod(mat_ctx->pc,"PCBDDCMatFETIDPGetRHS_C",(Mat,Vec,Vec),(fetidp_mat,standard_rhs,fetidp_flux_rhs));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
/* -------------------------------------------------------------------------- */

#undef __FUNCT__
#define __FUNCT__ "PCBDDCMatFETIDPGetSolution_BDDC"
static PetscErrorCode PCBDDCMatFETIDPGetSolution_BDDC(Mat fetidp_mat, Vec fetidp_flux_sol, Vec standard_sol)
{
  FETIDPMat_ctx  mat_ctx;
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
  if (pcbddc->inexact_prec_type) {
    ierr = VecCopy(mat_ctx->temp_solution_D,pcis->vec1_D);CHKERRQ(ierr);
  }
  /* apply BDDC */
  ierr = PCBDDCApplyInterfacePreconditioner(mat_ctx->pc);CHKERRQ(ierr);
  /* put values into standard global vector */
  ierr = VecScatterBegin(pcis->global_to_B,pcis->vec1_B,standard_sol,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  ierr = VecScatterEnd  (pcis->global_to_B,pcis->vec1_B,standard_sol,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  if (!pcbddc->inexact_prec_type) {
    /* compute values into the interior if solved for the partially subassembled Schur complement */
    ierr = MatMult(pcis->A_IB,pcis->vec1_B,pcis->vec1_D);CHKERRQ(ierr);
    ierr = VecAXPY(mat_ctx->temp_solution_D,-1.0,pcis->vec1_D);CHKERRQ(ierr);
    ierr = KSPSolve(pcbddc->ksp_D,mat_ctx->temp_solution_D,pcis->vec1_D);CHKERRQ(ierr);
  }
  ierr = VecScatterBegin(pcis->global_to_D,pcis->vec1_D,standard_sol,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  ierr = VecScatterEnd  (pcis->global_to_D,pcis->vec1_D,standard_sol,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  /* final change of basis if needed
     Is also sums the dirichlet part removed during RHS assembling */
  (*mat_ctx->pc->ops->postsolve)(mat_ctx->pc,NULL,NULL,standard_sol);
  PetscFunctionReturn(0);

}

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
  FETIDPMat_ctx  mat_ctx;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatShellGetContext(fetidp_mat,&mat_ctx);CHKERRQ(ierr);
  ierr = PetscTryMethod(mat_ctx->pc,"PCBDDCMatFETIDPGetSolution_C",(Mat,Vec,Vec),(fetidp_mat,fetidp_flux_sol,standard_sol));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
/* -------------------------------------------------------------------------- */

extern PetscErrorCode FETIDPMatMult(Mat,Vec,Vec);
extern PetscErrorCode PCBDDCDestroyFETIDPMat(Mat);
extern PetscErrorCode FETIDPPCApply(PC,Vec,Vec);
extern PetscErrorCode PCBDDCDestroyFETIDPPC(PC);

#undef __FUNCT__
#define __FUNCT__ "PCBDDCCreateFETIDPOperators_BDDC"
static PetscErrorCode PCBDDCCreateFETIDPOperators_BDDC(PC pc, Mat *fetidp_mat, PC *fetidp_pc)
{

  FETIDPMat_ctx  fetidpmat_ctx;
  Mat            newmat;
  FETIDPPC_ctx   fetidppc_ctx;
  PC             newpc;
  MPI_Comm       comm;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)pc,&comm);CHKERRQ(ierr);
  /* FETIDP linear matrix */
  ierr = PCBDDCCreateFETIDPMatContext(pc,&fetidpmat_ctx);CHKERRQ(ierr);
  ierr = PCBDDCSetupFETIDPMatContext(fetidpmat_ctx);CHKERRQ(ierr);
  ierr = MatCreateShell(comm,PETSC_DECIDE,PETSC_DECIDE,fetidpmat_ctx->n_lambda,fetidpmat_ctx->n_lambda,fetidpmat_ctx,&newmat);CHKERRQ(ierr);
  ierr = MatShellSetOperation(newmat,MATOP_MULT,(void (*)(void))FETIDPMatMult);CHKERRQ(ierr);
  ierr = MatShellSetOperation(newmat,MATOP_DESTROY,(void (*)(void))PCBDDCDestroyFETIDPMat);CHKERRQ(ierr);
  ierr = MatSetUp(newmat);CHKERRQ(ierr);
  /* FETIDP preconditioner */
  ierr = PCBDDCCreateFETIDPPCContext(pc,&fetidppc_ctx);CHKERRQ(ierr);
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
  } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"You must call PCSetup_BDDC() first \n");
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

#undef __FUNCT__
#define __FUNCT__ "PCCreate_BDDC"
PETSC_EXTERN PetscErrorCode PCCreate_BDDC(PC pc)
{
  PetscErrorCode      ierr;
  PC_BDDC             *pcbddc;

  PetscFunctionBegin;
  /* Creates the private data structure for this preconditioner and attach it to the PC object. */
  ierr      = PetscNewLog(pc,PC_BDDC,&pcbddc);CHKERRQ(ierr);
  pc->data  = (void*)pcbddc;

  /* create PCIS data structure */
  ierr = PCISCreate(pc);CHKERRQ(ierr);

  /* BDDC specific */
  pcbddc->user_primal_vertices       = 0;
  pcbddc->NullSpace                  = 0;
  pcbddc->temp_solution              = 0;
  pcbddc->original_rhs               = 0;
  pcbddc->local_mat                  = 0;
  pcbddc->ChangeOfBasisMatrix        = 0;
  pcbddc->use_change_of_basis        = PETSC_TRUE;
  pcbddc->use_change_on_faces        = PETSC_FALSE;
  pcbddc->coarse_vec                 = 0;
  pcbddc->coarse_rhs                 = 0;
  pcbddc->coarse_ksp                 = 0;
  pcbddc->coarse_phi_B               = 0;
  pcbddc->coarse_phi_D               = 0;
  pcbddc->coarse_psi_B               = 0;
  pcbddc->coarse_psi_D               = 0;
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
  pcbddc->inexact_prec_type          = PETSC_FALSE;
  pcbddc->NeumannBoundaries          = 0;
  pcbddc->ISForDofs                  = 0;
  pcbddc->ConstraintMatrix           = 0;
  pcbddc->use_nnsp_true              = PETSC_FALSE;
  pcbddc->local_primal_sizes         = 0;
  pcbddc->local_primal_displacements = 0;
  pcbddc->coarse_loc_to_glob         = 0;
  pcbddc->dbg_flag                   = 0;
  pcbddc->coarsening_ratio           = 8;
  pcbddc->use_exact_dirichlet        = PETSC_TRUE;
  pcbddc->current_level              = 0;
  pcbddc->max_levels                 = 1;
  pcbddc->replicated_local_primal_indices = 0;
  pcbddc->replicated_local_primal_values  = 0;

  /* create local graph structure */
  ierr = PCBDDCGraphCreate(&pcbddc->mat_graph);CHKERRQ(ierr);

  /* scaling */
  pcbddc->use_deluxe_scaling         = PETSC_FALSE;
  pcbddc->work_scaling               = 0;

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
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCBDDCSetPrimalVerticesLocalIS_C",PCBDDCSetPrimalVerticesLocalIS_BDDC);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCBDDCSetCoarseningRatio_C",PCBDDCSetCoarseningRatio_BDDC);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCBDDCSetMaxLevels_C",PCBDDCSetMaxLevels_BDDC);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCBDDCSetNullSpace_C",PCBDDCSetNullSpace_BDDC);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCBDDCSetDirichletBoundaries_C",PCBDDCSetDirichletBoundaries_BDDC);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCBDDCSetNeumannBoundaries_C",PCBDDCSetNeumannBoundaries_BDDC);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCBDDCGetDirichletBoundaries_C",PCBDDCGetDirichletBoundaries_BDDC);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCBDDCGetNeumannBoundaries_C",PCBDDCGetNeumannBoundaries_BDDC);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCBDDCSetCoarseProblemType_C",PCBDDCSetCoarseProblemType_BDDC);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCBDDCSetDofsSplitting_C",PCBDDCSetDofsSplitting_BDDC);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCBDDCSetLocalAdjacencyGraph_C",PCBDDCSetLocalAdjacencyGraph_BDDC);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCBDDCCreateFETIDPOperators_C",PCBDDCCreateFETIDPOperators_BDDC);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCBDDCMatFETIDPGetRHS_C",PCBDDCMatFETIDPGetRHS_BDDC);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCBDDCMatFETIDPGetSolution_C",PCBDDCMatFETIDPGetSolution_BDDC);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

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

/* -------------------------------------------------------------------------- */
#undef __FUNCT__
#define __FUNCT__ "PCBDDCCoarseSetUp"
static PetscErrorCode PCBDDCCoarseSetUp(PC pc)
{
  PetscErrorCode  ierr;

  PC_IS*            pcis = (PC_IS*)(pc->data);
  PC_BDDC*          pcbddc = (PC_BDDC*)pc->data;
  Mat_IS            *matis = (Mat_IS*)pc->pmat->data;
  IS                is_R_local;
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
  PetscReal         *coarsefunctions_errors,*constraints_errors;
  /* auxiliary indices */
  PetscInt          i,j,k;
  /* for verbose output of bddc */
  PetscViewer       viewer=pcbddc->dbg_viewer;
  PetscInt          dbg_flag=pcbddc->dbg_flag;
  /* for counting coarse dofs */
  PetscInt          n_vertices,n_constraints;
  PetscInt          size_of_constraint;
  PetscInt          *row_cmat_indices;
  PetscScalar       *row_cmat_values;
  PetscInt          *vertices;

  PetscFunctionBegin;
  /* Set Non-overlapping dimensions */
  n_B = pcis->n_B; n_D = pcis->n - n_B;

  /* transform local matrices if needed */
  if (pcbddc->use_change_of_basis) {
    Mat      change_mat_all;
    PetscInt *nnz,*is_indices,*temp_indices;

    ierr = PetscMalloc(pcis->n*sizeof(PetscInt),&nnz);CHKERRQ(ierr);
    ierr = ISGetIndices(pcis->is_I_local,(const PetscInt**)&is_indices);CHKERRQ(ierr);
    for (i=0;i<n_D;i++) nnz[is_indices[i]] = 1;
    ierr = ISRestoreIndices(pcis->is_I_local,(const PetscInt**)&is_indices);CHKERRQ(ierr);
    ierr = ISGetIndices(pcis->is_B_local,(const PetscInt**)&is_indices);CHKERRQ(ierr);
    k=1;
    for (i=0;i<n_B;i++) {
      ierr = MatGetRow(pcbddc->ChangeOfBasisMatrix,i,&j,NULL,NULL);CHKERRQ(ierr);
      nnz[is_indices[i]]=j;
      if (k < j) k = j;
      ierr = MatRestoreRow(pcbddc->ChangeOfBasisMatrix,i,&j,NULL,NULL);CHKERRQ(ierr);
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
      for (k=0; k<j; k++) temp_indices[k]=is_indices[row_cmat_indices[k]];
      ierr = MatSetValues(change_mat_all,1,&is_indices[i],j,temp_indices,row_cmat_values,INSERT_VALUES);CHKERRQ(ierr);
      ierr = MatRestoreRow(pcbddc->ChangeOfBasisMatrix,i,&j,(const PetscInt**)&row_cmat_indices,(const PetscScalar**)&row_cmat_values);CHKERRQ(ierr);
    }
    ierr = MatAssemblyBegin(change_mat_all,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(change_mat_all,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    /* TODO: HOW TO WORK WITH BAIJ? PtAP not provided */
    ierr = MatGetBlockSize(matis->A,&i);CHKERRQ(ierr);
    if (i==1) {
      ierr = MatPtAP(matis->A,change_mat_all,MAT_INITIAL_MATRIX,1.0,&pcbddc->local_mat);CHKERRQ(ierr);
    } else {
      Mat work_mat;
      ierr = MatConvert(matis->A,MATSEQAIJ,MAT_INITIAL_MATRIX,&work_mat);CHKERRQ(ierr);
      ierr = MatPtAP(work_mat,change_mat_all,MAT_INITIAL_MATRIX,1.0,&pcbddc->local_mat);CHKERRQ(ierr);
      ierr = MatDestroy(&work_mat);CHKERRQ(ierr);
    }
    ierr = MatDestroy(&change_mat_all);CHKERRQ(ierr);
    ierr = PetscFree(nnz);CHKERRQ(ierr);
    ierr = PetscFree(temp_indices);CHKERRQ(ierr);
  } else {
    /* without change of basis, the local matrix is unchanged */
    ierr = PetscObjectReference((PetscObject)matis->A);CHKERRQ(ierr);
    pcbddc->local_mat = matis->A;
  }
  /* need to rebuild PCIS matrices during SNES or TS -> TODO move this to PCIS code */
  ierr = MatDestroy(&pcis->A_IB);CHKERRQ(ierr);
  ierr = MatDestroy(&pcis->A_BI);CHKERRQ(ierr);
  ierr = MatDestroy(&pcis->A_BB);CHKERRQ(ierr);
  ierr = MatGetSubMatrix(pcbddc->local_mat,pcis->is_I_local,pcis->is_B_local,MAT_INITIAL_MATRIX,&pcis->A_IB);CHKERRQ(ierr);
  ierr = MatGetSubMatrix(pcbddc->local_mat,pcis->is_B_local,pcis->is_I_local,MAT_INITIAL_MATRIX,&pcis->A_BI);CHKERRQ(ierr);
  ierr = MatGetSubMatrix(pcbddc->local_mat,pcis->is_B_local,pcis->is_B_local,MAT_INITIAL_MATRIX,&pcis->A_BB);CHKERRQ(ierr);
  /* Change global null space passed in by the user if change of basis has been requested */
  if (pcbddc->NullSpace && pcbddc->use_change_of_basis) {
    ierr = PCBDDCNullSpaceAdaptGlobal(pc);CHKERRQ(ierr);
  }

  /* Set types for local objects needed by BDDC precondtioner */
  impMatType = MATSEQDENSE;
  impVecType = VECSEQ;
  /* get vertex indices from constraint matrix */
  ierr = PCBDDCGetPrimalVerticesLocalIdx(pc,&n_vertices,&vertices);CHKERRQ(ierr);
  /* Set number of constraints */
  n_constraints = pcbddc->local_primal_size-n_vertices;
  /* Dohrmann's notation: dofs splitted in R (Remaining: all dofs but the vertices) and V (Vertices) */
  ierr = VecSet(pcis->vec1_N,one);CHKERRQ(ierr);
  ierr = VecGetArray(pcis->vec1_N,&array);CHKERRQ(ierr);
  for (i=0;i<n_vertices;i++) array[vertices[i]] = zero;
  ierr = PetscMalloc((pcis->n-n_vertices)*sizeof(PetscInt),&idx_R_local);CHKERRQ(ierr);
  for (i=0, n_R=0; i<pcis->n; i++) {
    if (array[i] == one) {
      idx_R_local[n_R] = i;
      n_R++;
    }
  }
  ierr = VecRestoreArray(pcis->vec1_N,&array);CHKERRQ(ierr);
  ierr = PetscFree(vertices);CHKERRQ(ierr);
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

  /* For VecScatters pcbddc->R_to_B and (optionally) pcbddc->R_to_D */
  {
    IS         is_aux1,is_aux2;
    PetscInt   *aux_array1;
    PetscInt   *aux_array2;
    PetscInt   *idx_I_local;

    ierr = PetscMalloc((pcis->n_B-n_vertices)*sizeof(PetscInt),&aux_array1);CHKERRQ(ierr);
    ierr = PetscMalloc((pcis->n_B-n_vertices)*sizeof(PetscInt),&aux_array2);CHKERRQ(ierr);

    ierr = ISGetIndices(pcis->is_I_local,(const PetscInt**)&idx_I_local);CHKERRQ(ierr);
    ierr = VecGetArray(pcis->vec1_N,&array);CHKERRQ(ierr);
    for (i=0; i<n_D; i++) array[idx_I_local[i]] = 0;
    ierr = ISRestoreIndices(pcis->is_I_local,(const PetscInt**)&idx_I_local);CHKERRQ(ierr);
    for (i=0, j=0; i<n_R; i++) {
      if (array[idx_R_local[i]] == one) {
        aux_array1[j] = i;
        j++;
      }
    }
    ierr = VecRestoreArray(pcis->vec1_N,&array);CHKERRQ(ierr);
    ierr = ISCreateGeneral(PETSC_COMM_SELF,j,aux_array1,PETSC_COPY_VALUES,&is_aux1);CHKERRQ(ierr);
    ierr = VecScatterBegin(pcis->N_to_B,pcis->vec1_N,pcis->vec1_B,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterEnd  (pcis->N_to_B,pcis->vec1_N,pcis->vec1_B,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecGetArray(pcis->vec1_B,&array);CHKERRQ(ierr);
    for (i=0, j=0; i<n_B; i++) {
      if (array[i] == one) {
        aux_array2[j] = i; j++;
      }
    }
    ierr = VecRestoreArray(pcis->vec1_B,&array);CHKERRQ(ierr);
    ierr = ISCreateGeneral(PETSC_COMM_SELF,j,aux_array2,PETSC_COPY_VALUES,&is_aux2);CHKERRQ(ierr);
    ierr = VecScatterCreate(pcbddc->vec1_R,is_aux1,pcis->vec1_B,is_aux2,&pcbddc->R_to_B);CHKERRQ(ierr);
    ierr = PetscFree(aux_array1);CHKERRQ(ierr);
    ierr = PetscFree(aux_array2);CHKERRQ(ierr);
    ierr = ISDestroy(&is_aux1);CHKERRQ(ierr);
    ierr = ISDestroy(&is_aux2);CHKERRQ(ierr);

    if (pcbddc->inexact_prec_type || dbg_flag ) {
      ierr = PetscMalloc(n_D*sizeof(PetscInt),&aux_array1);CHKERRQ(ierr);
      ierr = VecGetArray(pcis->vec1_N,&array);CHKERRQ(ierr);
      for (i=0, j=0; i<n_R; i++) {
        if (array[idx_R_local[i]] == zero) {
          aux_array1[j] = i;
          j++;
        }
      }
      ierr = VecRestoreArray(pcis->vec1_N,&array);CHKERRQ(ierr);
      ierr = ISCreateGeneral(PETSC_COMM_SELF,j,aux_array1,PETSC_COPY_VALUES,&is_aux1);CHKERRQ(ierr);
      ierr = VecScatterCreate(pcbddc->vec1_R,is_aux1,pcis->vec1_D,(IS)0,&pcbddc->R_to_D);CHKERRQ(ierr);
      ierr = PetscFree(aux_array1);CHKERRQ(ierr);
      ierr = ISDestroy(&is_aux1);CHKERRQ(ierr);
    }
  }

  /* Creating PC contexts for local Dirichlet and Neumann problems */
  {
    Mat          A_RR;
    PC           pc_temp;
    MatStructure matstruct;
    /* Matrix for Dirichlet problem is A_II */
    /* HACK (TODO) A_II can be changed between nonlinear iterations */
    if (pc->setupcalled) { /* we dont need to rebuild dirichlet problem the first time we build BDDC */
      ierr = PCGetOperators(pc,NULL,NULL,&matstruct);CHKERRQ(ierr);
      if (matstruct == SAME_NONZERO_PATTERN) {
        ierr = MatGetSubMatrix(matis->A,pcis->is_I_local,pcis->is_I_local,MAT_REUSE_MATRIX,&pcis->A_II);CHKERRQ(ierr);
      } else {
        ierr = MatDestroy(&pcis->A_II);CHKERRQ(ierr);
        ierr = MatGetSubMatrix(matis->A,pcis->is_I_local,pcis->is_I_local,MAT_INITIAL_MATRIX,&pcis->A_II);CHKERRQ(ierr);
      }
    }
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
    if (!n_R) {
      ierr = PCSetType(pc_temp,PCNONE);CHKERRQ(ierr);
    }
    /* Set Up KSP for Neumann problem of BDDC */
    ierr = KSPSetUp(pcbddc->ksp_R);CHKERRQ(ierr);
    /* check Dirichlet and Neumann solvers and adapt them if a nullspace correction is needed */
    {
      Vec         temp_vec;
      PetscReal   value;
      PetscInt    use_exact,use_exact_reduced;

      ierr = VecDuplicate(pcis->vec1_D,&temp_vec);CHKERRQ(ierr);
      ierr = VecSetRandom(pcis->vec1_D,NULL);CHKERRQ(ierr);
      ierr = MatMult(pcis->A_II,pcis->vec1_D,pcis->vec2_D);CHKERRQ(ierr);
      ierr = KSPSolve(pcbddc->ksp_D,pcis->vec2_D,temp_vec);CHKERRQ(ierr);
      ierr = VecAXPY(temp_vec,m_one,pcis->vec1_D);CHKERRQ(ierr);
      ierr = VecNorm(temp_vec,NORM_INFINITY,&value);CHKERRQ(ierr);
      ierr = VecDestroy(&temp_vec);CHKERRQ(ierr);
      use_exact = 1;
      if (PetscAbsReal(value) > 1.e-4) use_exact = 0;
      ierr = MPI_Allreduce(&use_exact,&use_exact_reduced,1,MPIU_INT,MPI_LAND,PetscObjectComm((PetscObject)pc));CHKERRQ(ierr);
      ierr = PCBDDCSetUseExactDirichlet(pc,(PetscBool)use_exact_reduced);CHKERRQ(ierr);
      if (dbg_flag) {
        ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
        ierr = PetscViewerASCIIPrintf(viewer,"--------------------------------------------------\n");CHKERRQ(ierr);
        ierr = PetscViewerASCIIPrintf(viewer,"Checking solution of Dirichlet and Neumann problems\n");CHKERRQ(ierr);
        ierr = PetscViewerASCIISynchronizedPrintf(viewer,"Subdomain %04d infinity error for Dirichlet solve = % 1.14e \n",PetscGlobalRank,value);CHKERRQ(ierr);
        ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
      }
      if (n_D && pcbddc->NullSpace && !use_exact_reduced && !pcbddc->inexact_prec_type) {
        ierr = PCBDDCNullSpaceAssembleCorrection(pc,pcis->is_I_local);
      }
      ierr = VecDuplicate(pcbddc->vec1_R,&temp_vec);CHKERRQ(ierr);
      ierr = VecSetRandom(pcbddc->vec1_R,NULL);CHKERRQ(ierr);
      ierr = MatMult(A_RR,pcbddc->vec1_R,pcbddc->vec2_R);CHKERRQ(ierr);
      ierr = KSPSolve(pcbddc->ksp_R,pcbddc->vec2_R,temp_vec);CHKERRQ(ierr);
      ierr = VecAXPY(temp_vec,m_one,pcbddc->vec1_R);CHKERRQ(ierr);
      ierr = VecNorm(temp_vec,NORM_INFINITY,&value);CHKERRQ(ierr);
      ierr = VecDestroy(&temp_vec);CHKERRQ(ierr);
      use_exact = 1;
      if (PetscAbsReal(value) > 1.e-4) use_exact = 0;
      ierr = MPI_Allreduce(&use_exact,&use_exact_reduced,1,MPIU_INT,MPI_LAND,PetscObjectComm((PetscObject)pc));CHKERRQ(ierr);
      if (dbg_flag) {
        ierr = PetscViewerASCIISynchronizedPrintf(viewer,"Subdomain %04d infinity error for  Neumann  solve = % 1.14e \n",PetscGlobalRank,value);CHKERRQ(ierr);
        ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
      }
      if (n_R && pcbddc->NullSpace && !use_exact_reduced) { /* is it the right logic? */
        ierr = PCBDDCNullSpaceAssembleCorrection(pc,is_R_local);
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
    IS           is_C_local,is_V_local,is_aux1;
    ISLocalToGlobalMapping BtoNmap;
    PetscInt     *nnz;
    PetscInt     *idx_V_B;
    PetscInt     *auxindices;
    PetscInt     index;
    PetscScalar* array2;
    MatFactorInfo matinfo;
    PetscBool    setsym=PETSC_FALSE,issym=PETSC_FALSE;

    /* Allocating some extra storage just to be safe */
    ierr = PetscMalloc (pcis->n*sizeof(PetscInt),&nnz);CHKERRQ(ierr);
    ierr = PetscMalloc (pcis->n*sizeof(PetscInt),&auxindices);CHKERRQ(ierr);
    for (i=0;i<pcis->n;i++) auxindices[i]=i;

    ierr = PCBDDCGetPrimalVerticesLocalIdx(pc,&n_vertices,&vertices);CHKERRQ(ierr);
    /* vertices in boundary numbering */
    ierr = PetscMalloc(n_vertices*sizeof(PetscInt),&idx_V_B);CHKERRQ(ierr);
    ierr = ISLocalToGlobalMappingCreateIS(pcis->is_B_local,&BtoNmap);CHKERRQ(ierr);
    ierr = ISGlobalToLocalMappingApply(BtoNmap,IS_GTOLM_DROP,n_vertices,vertices,&i,idx_V_B);CHKERRQ(ierr);
    ierr = ISLocalToGlobalMappingDestroy(&BtoNmap);CHKERRQ(ierr);
    if (i != n_vertices) {
      SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_SUP,"Error in boundary numbering for BDDC vertices! %d != %d\n",n_vertices,i);
    }

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
      ierr = MatSeqDenseSetPreallocation(pcbddc->local_auxmat2,NULL);CHKERRQ(ierr);

      /* Create Constraint matrix on R nodes: C_{CR}  */
      ierr = ISCreateStride(PETSC_COMM_SELF,n_constraints,n_vertices,1,&is_C_local);CHKERRQ(ierr);
      ierr = MatGetSubMatrix(pcbddc->ConstraintMatrix,is_C_local,is_R_local,MAT_INITIAL_MATRIX,&C_CR);CHKERRQ(ierr);
      ierr = ISDestroy(&is_C_local);CHKERRQ(ierr);

      /* Assemble local_auxmat2 = - A_{RR}^{-1} C^T_{CR} needed by BDDC application */
      for (i=0;i<n_constraints;i++) {
        ierr = VecSet(pcbddc->vec1_R,zero);CHKERRQ(ierr);
        /* Get row of constraint matrix in R numbering */
        ierr = VecGetArray(pcbddc->vec1_R,&array);CHKERRQ(ierr);
        ierr = MatGetRow(C_CR,i,&size_of_constraint,(const PetscInt**)&row_cmat_indices,(const PetscScalar**)&row_cmat_values);CHKERRQ(ierr);
        for (j=0;j<size_of_constraint;j++) array[row_cmat_indices[j]] = -row_cmat_values[j];
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
      ierr = MatSeqDenseSetPreallocation(M1,NULL);CHKERRQ(ierr);
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
    if (n_vertices) {
      ierr = ISCreateGeneral(PETSC_COMM_SELF,n_vertices,vertices,PETSC_COPY_VALUES,&is_V_local);CHKERRQ(ierr);
      ierr = MatGetSubMatrix(pcbddc->local_mat,is_R_local,is_V_local,MAT_INITIAL_MATRIX,&A_RV);CHKERRQ(ierr);
      ierr = MatGetSubMatrix(pcbddc->local_mat,is_V_local,is_R_local,MAT_INITIAL_MATRIX,&A_VR);CHKERRQ(ierr);
      ierr = MatGetSubMatrix(pcbddc->local_mat,is_V_local,is_V_local,MAT_INITIAL_MATRIX,&A_VV);CHKERRQ(ierr);
      ierr = ISDestroy(&is_V_local);CHKERRQ(ierr);
    }

    /* Matrix of coarse basis functions (local) */
    ierr = MatCreate(PETSC_COMM_SELF,&pcbddc->coarse_phi_B);CHKERRQ(ierr);
    ierr = MatSetSizes(pcbddc->coarse_phi_B,n_B,pcbddc->local_primal_size,n_B,pcbddc->local_primal_size);CHKERRQ(ierr);
    ierr = MatSetType(pcbddc->coarse_phi_B,impMatType);CHKERRQ(ierr);
    ierr = MatSeqDenseSetPreallocation(pcbddc->coarse_phi_B,NULL);CHKERRQ(ierr);
    if (pcbddc->inexact_prec_type || dbg_flag ) {
      ierr = MatCreate(PETSC_COMM_SELF,&pcbddc->coarse_phi_D);CHKERRQ(ierr);
      ierr = MatSetSizes(pcbddc->coarse_phi_D,n_D,pcbddc->local_primal_size,n_D,pcbddc->local_primal_size);CHKERRQ(ierr);
      ierr = MatSetType(pcbddc->coarse_phi_D,impMatType);CHKERRQ(ierr);
      ierr = MatSeqDenseSetPreallocation(pcbddc->coarse_phi_D,NULL);CHKERRQ(ierr);
    }

    if (dbg_flag) {
      ierr = PetscMalloc(2*pcbddc->local_primal_size*sizeof(*coarsefunctions_errors),&coarsefunctions_errors);CHKERRQ(ierr);
      ierr = PetscMalloc(2*pcbddc->local_primal_size*sizeof(*constraints_errors),&constraints_errors);CHKERRQ(ierr);
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
      if ( pcbddc->inexact_prec_type || dbg_flag  ) {
        ierr = VecScatterBegin(pcbddc->R_to_D,pcbddc->vec1_R,pcis->vec1_D,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
        ierr = VecScatterEnd(pcbddc->R_to_D,pcbddc->vec1_R,pcis->vec1_D,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
        ierr = VecGetArray(pcis->vec1_D,&array);CHKERRQ(ierr);
        ierr = MatSetValues(pcbddc->coarse_phi_D,n_D,auxindices,1,&i,array,INSERT_VALUES);CHKERRQ(ierr);
        ierr = VecRestoreArray(pcis->vec1_D,&array);CHKERRQ(ierr);
      }
      /* subdomain contribution to coarse matrix */
      ierr = VecGetArray(vec2_V,&array);CHKERRQ(ierr);
      for (j=0; j<n_vertices; j++) coarse_submat_vals[i*pcbddc->local_primal_size+j] = array[j];   /* WARNING -> column major ordering */
      ierr = VecRestoreArray(vec2_V,&array);CHKERRQ(ierr);
      if (n_constraints) {
        ierr = VecGetArray(vec1_C,&array);CHKERRQ(ierr);
        for (j=0; j<n_constraints; j++) coarse_submat_vals[i*pcbddc->local_primal_size+j+n_vertices] = array[j];   /* WARNING -> column major ordering */
        ierr = VecRestoreArray(vec1_C,&array);CHKERRQ(ierr);
      }

      if ( dbg_flag ) {
        /* assemble subdomain vector on nodes */
        ierr = VecSet(pcis->vec1_N,zero);CHKERRQ(ierr);
        ierr = VecGetArray(pcis->vec1_N,&array);CHKERRQ(ierr);
        ierr = VecGetArray(pcbddc->vec1_R,&array2);CHKERRQ(ierr);
        for (j=0;j<n_R;j++) array[idx_R_local[j]] = array2[j];
        array[ vertices[i] ] = one;
        ierr = VecRestoreArray(pcbddc->vec1_R,&array2);CHKERRQ(ierr);
        ierr = VecRestoreArray(pcis->vec1_N,&array);CHKERRQ(ierr);
        /* assemble subdomain vector of lagrange multipliers (i.e. primal nodes) */
        ierr = VecSet(pcbddc->vec1_P,zero);CHKERRQ(ierr);
        ierr = VecGetArray(pcbddc->vec1_P,&array2);CHKERRQ(ierr);
        ierr = VecGetArray(vec2_V,&array);CHKERRQ(ierr);
        for (j=0;j<n_vertices;j++) array2[j]=array[j];
        ierr = VecRestoreArray(vec2_V,&array);CHKERRQ(ierr);
        if (n_constraints) {
          ierr = VecGetArray(vec1_C,&array);CHKERRQ(ierr);
          for (j=0;j<n_constraints;j++) array2[j+n_vertices]=array[j];
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
      ierr = VecScatterEnd(pcbddc->R_to_B,pcbddc->vec1_R,pcis->vec1_B,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
      ierr = VecGetArray(pcis->vec1_B,&array);CHKERRQ(ierr);
      ierr = MatSetValues(pcbddc->coarse_phi_B,n_B,auxindices,1,&index,array,INSERT_VALUES);CHKERRQ(ierr);
      ierr = VecRestoreArray(pcis->vec1_B,&array);CHKERRQ(ierr);
      if ( pcbddc->inexact_prec_type || dbg_flag ) {
        ierr = VecScatterBegin(pcbddc->R_to_D,pcbddc->vec1_R,pcis->vec1_D,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
        ierr = VecScatterEnd(pcbddc->R_to_D,pcbddc->vec1_R,pcis->vec1_D,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
        ierr = VecGetArray(pcis->vec1_D,&array);CHKERRQ(ierr);
        ierr = MatSetValues(pcbddc->coarse_phi_D,n_D,auxindices,1,&index,array,INSERT_VALUES);CHKERRQ(ierr);
        ierr = VecRestoreArray(pcis->vec1_D,&array);CHKERRQ(ierr);
      }
      /* subdomain contribution to coarse matrix */
      if (n_vertices) {
        ierr = VecGetArray(vec2_V,&array);CHKERRQ(ierr);
        for (j=0; j<n_vertices; j++) coarse_submat_vals[index*pcbddc->local_primal_size+j]=array[j]; /* WARNING -> column major ordering */
        ierr = VecRestoreArray(vec2_V,&array);CHKERRQ(ierr);
      }
      ierr = VecGetArray(vec1_C,&array);CHKERRQ(ierr);
      for (j=0; j<n_constraints; j++) coarse_submat_vals[index*pcbddc->local_primal_size+j+n_vertices]=array[j]; /* WARNING -> column major ordering */
      ierr = VecRestoreArray(vec1_C,&array);CHKERRQ(ierr);

      if ( dbg_flag ) {
        /* assemble subdomain vector on nodes */
        ierr = VecSet(pcis->vec1_N,zero);CHKERRQ(ierr);
        ierr = VecGetArray(pcis->vec1_N,&array);CHKERRQ(ierr);
        ierr = VecGetArray(pcbddc->vec1_R,&array2);CHKERRQ(ierr);
        for (j=0;j<n_R;j++) array[idx_R_local[j]] = array2[j];
        ierr = VecRestoreArray(pcbddc->vec1_R,&array2);CHKERRQ(ierr);
        ierr = VecRestoreArray(pcis->vec1_N,&array);CHKERRQ(ierr);
        /* assemble subdomain vector of lagrange multipliers */
        ierr = VecSet(pcbddc->vec1_P,zero);CHKERRQ(ierr);
        ierr = VecGetArray(pcbddc->vec1_P,&array2);CHKERRQ(ierr);
        if ( n_vertices) {
          ierr = VecGetArray(vec2_V,&array);CHKERRQ(ierr);
          for (j=0;j<n_vertices;j++) array2[j]=-array[j];
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
    ierr = MatAssemblyEnd(pcbddc->coarse_phi_B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    if ( pcbddc->inexact_prec_type || dbg_flag ) {
      ierr = MatAssemblyBegin(pcbddc->coarse_phi_D,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
      ierr = MatAssemblyEnd(pcbddc->coarse_phi_D,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    }
    /* compute other basis functions for non-symmetric problems */
    ierr = MatIsSymmetricKnown(pc->pmat,&setsym,&issym);CHKERRQ(ierr);
    if ( !setsym || (setsym && !issym) ) {
      ierr = MatCreate(PETSC_COMM_SELF,&pcbddc->coarse_psi_B);CHKERRQ(ierr);
      ierr = MatSetSizes(pcbddc->coarse_psi_B,n_B,pcbddc->local_primal_size,n_B,pcbddc->local_primal_size);CHKERRQ(ierr);
      ierr = MatSetType(pcbddc->coarse_psi_B,impMatType);CHKERRQ(ierr);
      ierr = MatSeqDenseSetPreallocation(pcbddc->coarse_psi_B,NULL);CHKERRQ(ierr);
      if (pcbddc->inexact_prec_type || dbg_flag ) {
        ierr = MatCreate(PETSC_COMM_SELF,&pcbddc->coarse_psi_D);CHKERRQ(ierr);
        ierr = MatSetSizes(pcbddc->coarse_psi_D,n_D,pcbddc->local_primal_size,n_D,pcbddc->local_primal_size);CHKERRQ(ierr);
        ierr = MatSetType(pcbddc->coarse_psi_D,impMatType);CHKERRQ(ierr);
        ierr = MatSeqDenseSetPreallocation(pcbddc->coarse_psi_D,NULL);CHKERRQ(ierr);
      }
      for (i=0;i<pcbddc->local_primal_size;i++) {
        if (n_constraints) {
          ierr = VecSet(vec1_C,zero);CHKERRQ(ierr);
          ierr = VecGetArray(vec1_C,&array);CHKERRQ(ierr);
          for (j=0;j<n_constraints;j++) {
            array[j]=coarse_submat_vals[(j+n_vertices)*pcbddc->local_primal_size+i];
          } 
          ierr = VecRestoreArray(vec1_C,&array);CHKERRQ(ierr);
        }
        if (i<n_vertices) {
          ierr = VecSet(vec1_V,zero);CHKERRQ(ierr);
          ierr = VecSetValue(vec1_V,i,m_one,INSERT_VALUES);CHKERRQ(ierr);
          ierr = VecAssemblyBegin(vec1_V);CHKERRQ(ierr);
          ierr = VecAssemblyEnd(vec1_V);CHKERRQ(ierr);
          ierr = MatMultTranspose(A_VR,vec1_V,pcbddc->vec1_R);CHKERRQ(ierr);
          if (n_constraints) {
            ierr = MatMultTransposeAdd(C_CR,vec1_C,pcbddc->vec1_R,pcbddc->vec1_R);CHKERRQ(ierr);
          }
        } else {
          ierr = MatMultTranspose(C_CR,vec1_C,pcbddc->vec1_R);CHKERRQ(ierr);
        }
        ierr = KSPSolveTranspose(pcbddc->ksp_R,pcbddc->vec1_R,pcbddc->vec1_R);CHKERRQ(ierr);
        ierr = VecSet(pcis->vec1_B,zero);CHKERRQ(ierr);
        ierr = VecScatterBegin(pcbddc->R_to_B,pcbddc->vec1_R,pcis->vec1_B,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
        ierr = VecScatterEnd(pcbddc->R_to_B,pcbddc->vec1_R,pcis->vec1_B,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
        ierr = VecGetArray(pcis->vec1_B,&array);CHKERRQ(ierr);
        ierr = MatSetValues(pcbddc->coarse_psi_B,n_B,auxindices,1,&i,array,INSERT_VALUES);CHKERRQ(ierr);
        ierr = VecRestoreArray(pcis->vec1_B,&array);CHKERRQ(ierr);
        if (i<n_vertices) {
          ierr = MatSetValue(pcbddc->coarse_psi_B,idx_V_B[i],i,one,INSERT_VALUES);CHKERRQ(ierr);
        }
        if ( pcbddc->inexact_prec_type || dbg_flag  ) {
          ierr = VecScatterBegin(pcbddc->R_to_D,pcbddc->vec1_R,pcis->vec1_D,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
          ierr = VecScatterEnd(pcbddc->R_to_D,pcbddc->vec1_R,pcis->vec1_D,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
          ierr = VecGetArray(pcis->vec1_D,&array);CHKERRQ(ierr);
          ierr = MatSetValues(pcbddc->coarse_psi_D,n_D,auxindices,1,&i,array,INSERT_VALUES);CHKERRQ(ierr);
          ierr = VecRestoreArray(pcis->vec1_D,&array);CHKERRQ(ierr);
        }

        if ( dbg_flag ) {
          /* assemble subdomain vector on nodes */
          ierr = VecSet(pcis->vec1_N,zero);CHKERRQ(ierr);
          ierr = VecGetArray(pcis->vec1_N,&array);CHKERRQ(ierr);
          ierr = VecGetArray(pcbddc->vec1_R,&array2);CHKERRQ(ierr);
          for (j=0;j<n_R;j++) array[idx_R_local[j]] = array2[j];
          if (i<n_vertices) array[vertices[i]] = one;
          ierr = VecRestoreArray(pcbddc->vec1_R,&array2);CHKERRQ(ierr);
          ierr = VecRestoreArray(pcis->vec1_N,&array);CHKERRQ(ierr);
          /* assemble subdomain vector of lagrange multipliers */
          ierr = VecGetArray(pcbddc->vec1_P,&array);CHKERRQ(ierr);
          for (j=0;j<pcbddc->local_primal_size;j++) {
            array[j]=-coarse_submat_vals[j*pcbddc->local_primal_size+i];
          } 
          ierr = VecRestoreArray(pcbddc->vec1_P,&array);CHKERRQ(ierr);
          /* check saddle point solution */
          ierr = MatMultTranspose(pcbddc->local_mat,pcis->vec1_N,pcis->vec2_N);CHKERRQ(ierr);
          ierr = MatMultTransposeAdd(pcbddc->ConstraintMatrix,pcbddc->vec1_P,pcis->vec2_N,pcis->vec2_N);CHKERRQ(ierr);
          ierr = VecNorm(pcis->vec2_N,NORM_INFINITY,&coarsefunctions_errors[i+pcbddc->local_primal_size]);CHKERRQ(ierr);
          ierr = MatMult(pcbddc->ConstraintMatrix,pcis->vec1_N,pcbddc->vec1_P);CHKERRQ(ierr);
          ierr = VecGetArray(pcbddc->vec1_P,&array);CHKERRQ(ierr);
          array[i]=array[i]+m_one; /* shift by the identity matrix */
          ierr = VecRestoreArray(pcbddc->vec1_P,&array);CHKERRQ(ierr);
          ierr = VecNorm(pcbddc->vec1_P,NORM_INFINITY,&constraints_errors[i+pcbddc->local_primal_size]);CHKERRQ(ierr);
        }
      }
      ierr = MatAssemblyBegin(pcbddc->coarse_psi_B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
      ierr = MatAssemblyEnd(pcbddc->coarse_psi_B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
      if ( pcbddc->inexact_prec_type || dbg_flag ) {
        ierr = MatAssemblyBegin(pcbddc->coarse_psi_D,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
        ierr = MatAssemblyEnd(pcbddc->coarse_psi_D,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
      }
    }
    ierr = PetscFree(idx_V_B);CHKERRQ(ierr);
    /* Checking coarse_sub_mat and coarse basis functios */
    /* Symmetric case     : It should be \Phi^{(j)^T} A^{(j)} \Phi^{(j)}=coarse_sub_mat */
    /* Non-symmetric case : It should be \Psi^{(j)^T} A^{(j)} \Phi^{(j)}=coarse_sub_mat */
    if (dbg_flag) {
      Mat         coarse_sub_mat;
      Mat         TM1,TM2,TM3,TM4;
      Mat         coarse_phi_D,coarse_phi_B;
      Mat         coarse_psi_D,coarse_psi_B;
      Mat         A_II,A_BB,A_IB,A_BI;
      MatType     checkmattype=MATSEQAIJ;
      PetscReal   real_value;

      ierr = MatConvert(pcis->A_II,checkmattype,MAT_INITIAL_MATRIX,&A_II);CHKERRQ(ierr);
      ierr = MatConvert(pcis->A_IB,checkmattype,MAT_INITIAL_MATRIX,&A_IB);CHKERRQ(ierr);
      ierr = MatConvert(pcis->A_BI,checkmattype,MAT_INITIAL_MATRIX,&A_BI);CHKERRQ(ierr);
      ierr = MatConvert(pcis->A_BB,checkmattype,MAT_INITIAL_MATRIX,&A_BB);CHKERRQ(ierr);
      ierr = MatConvert(pcbddc->coarse_phi_D,checkmattype,MAT_INITIAL_MATRIX,&coarse_phi_D);CHKERRQ(ierr);
      ierr = MatConvert(pcbddc->coarse_phi_B,checkmattype,MAT_INITIAL_MATRIX,&coarse_phi_B);CHKERRQ(ierr);
      if (pcbddc->coarse_psi_B) {
        ierr = MatConvert(pcbddc->coarse_psi_D,checkmattype,MAT_INITIAL_MATRIX,&coarse_psi_D);CHKERRQ(ierr);
        ierr = MatConvert(pcbddc->coarse_psi_B,checkmattype,MAT_INITIAL_MATRIX,&coarse_psi_B);CHKERRQ(ierr);
      }
      ierr = MatCreateSeqDense(PETSC_COMM_SELF,pcbddc->local_primal_size,pcbddc->local_primal_size,coarse_submat_vals,&coarse_sub_mat);CHKERRQ(ierr);

      ierr = PetscViewerASCIIPrintf(viewer,"--------------------------------------------------\n");CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"Check coarse sub mat and local basis functions\n");CHKERRQ(ierr);
      ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
      if (pcbddc->coarse_psi_B) {
        ierr = MatMatMult(A_II,coarse_phi_D,MAT_INITIAL_MATRIX,1.0,&AUXMAT);CHKERRQ(ierr);
        ierr = MatTransposeMatMult(coarse_psi_D,AUXMAT,MAT_INITIAL_MATRIX,1.0,&TM1);CHKERRQ(ierr);
        ierr = MatDestroy(&AUXMAT);CHKERRQ(ierr);
        ierr = MatMatMult(A_BB,coarse_phi_B,MAT_INITIAL_MATRIX,1.0,&AUXMAT);CHKERRQ(ierr);
        ierr = MatTransposeMatMult(coarse_psi_B,AUXMAT,MAT_INITIAL_MATRIX,1.0,&TM2);CHKERRQ(ierr);
        ierr = MatDestroy(&AUXMAT);CHKERRQ(ierr);
        ierr = MatMatMult(A_IB,coarse_phi_B,MAT_INITIAL_MATRIX,1.0,&AUXMAT);CHKERRQ(ierr);
        ierr = MatTransposeMatMult(coarse_psi_D,AUXMAT,MAT_INITIAL_MATRIX,1.0,&TM3);CHKERRQ(ierr);
        ierr = MatDestroy(&AUXMAT);CHKERRQ(ierr);
        ierr = MatMatMult(A_BI,coarse_phi_D,MAT_INITIAL_MATRIX,1.0,&AUXMAT);CHKERRQ(ierr);
        ierr = MatTransposeMatMult(coarse_psi_B,AUXMAT,MAT_INITIAL_MATRIX,1.0,&TM4);CHKERRQ(ierr);
        ierr = MatDestroy(&AUXMAT);CHKERRQ(ierr);
      } else {
        ierr = MatPtAP(A_II,coarse_phi_D,MAT_INITIAL_MATRIX,1.0,&TM1);CHKERRQ(ierr);
        ierr = MatPtAP(A_BB,coarse_phi_B,MAT_INITIAL_MATRIX,1.0,&TM2);CHKERRQ(ierr);
        ierr = MatMatMult(A_IB,coarse_phi_B,MAT_INITIAL_MATRIX,1.0,&AUXMAT);CHKERRQ(ierr);
        ierr = MatTransposeMatMult(coarse_phi_D,AUXMAT,MAT_INITIAL_MATRIX,1.0,&TM3);CHKERRQ(ierr);
        ierr = MatDestroy(&AUXMAT);CHKERRQ(ierr);
        ierr = MatMatMult(A_BI,coarse_phi_D,MAT_INITIAL_MATRIX,1.0,&AUXMAT);CHKERRQ(ierr);
        ierr = MatTransposeMatMult(coarse_phi_B,AUXMAT,MAT_INITIAL_MATRIX,1.0,&TM4);CHKERRQ(ierr);
        ierr = MatDestroy(&AUXMAT);CHKERRQ(ierr);
      }
      ierr = MatAXPY(TM1,one,TM2,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
      ierr = MatAXPY(TM1,one,TM3,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
      ierr = MatAXPY(TM1,one,TM4,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
      ierr = MatConvert(TM1,MATSEQDENSE,MAT_REUSE_MATRIX,&TM1);CHKERRQ(ierr);
      ierr = MatAXPY(TM1,m_one,coarse_sub_mat,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
      ierr = MatNorm(TM1,NORM_INFINITY,&real_value);CHKERRQ(ierr);
      ierr = PetscViewerASCIISynchronizedPrintf(viewer,"----------------------------------\n");CHKERRQ(ierr);
      ierr = PetscViewerASCIISynchronizedPrintf(viewer,"Subdomain %04d \n",PetscGlobalRank);CHKERRQ(ierr);
      ierr = PetscViewerASCIISynchronizedPrintf(viewer,"matrix error = % 1.14e\n",real_value);CHKERRQ(ierr);
      ierr = PetscViewerASCIISynchronizedPrintf(viewer,"coarse functions (phi) errors\n");CHKERRQ(ierr);
      for (i=0;i<pcbddc->local_primal_size;i++) {
        ierr = PetscViewerASCIISynchronizedPrintf(viewer,"local %02d-th function error = % 1.14e\n",i,coarsefunctions_errors[i]);CHKERRQ(ierr);
      }
      ierr = PetscViewerASCIISynchronizedPrintf(viewer,"constraints (phi) errors\n");CHKERRQ(ierr);
      for (i=0;i<pcbddc->local_primal_size;i++) {
        ierr = PetscViewerASCIISynchronizedPrintf(viewer,"local %02d-th function error = % 1.14e\n",i,constraints_errors[i]);CHKERRQ(ierr);
      }
      if (pcbddc->coarse_psi_B) {
        ierr = PetscViewerASCIISynchronizedPrintf(viewer,"coarse functions (psi) errors\n");CHKERRQ(ierr);
        for (i=pcbddc->local_primal_size;i<2*pcbddc->local_primal_size;i++) {
          ierr = PetscViewerASCIISynchronizedPrintf(viewer,"local %02d-th function error = % 1.14e\n",i-pcbddc->local_primal_size,coarsefunctions_errors[i]);CHKERRQ(ierr);
        }
        ierr = PetscViewerASCIISynchronizedPrintf(viewer,"constraints (psi) errors\n");CHKERRQ(ierr);
        for (i=pcbddc->local_primal_size;i<2*pcbddc->local_primal_size;i++) {
          ierr = PetscViewerASCIISynchronizedPrintf(viewer,"local %02d-th function error = % 1.14e\n",i-pcbddc->local_primal_size,constraints_errors[i]);CHKERRQ(ierr);
        }
      }
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
      ierr = MatDestroy(&coarse_phi_B);CHKERRQ(ierr);
      if (pcbddc->coarse_psi_B) {
        ierr = MatDestroy(&coarse_psi_D);CHKERRQ(ierr);
        ierr = MatDestroy(&coarse_psi_B);CHKERRQ(ierr);
      }
      ierr = MatDestroy(&coarse_sub_mat);CHKERRQ(ierr);
      ierr = PetscFree(coarsefunctions_errors);CHKERRQ(ierr);
      ierr = PetscFree(constraints_errors);CHKERRQ(ierr);
    }
    /* free memory */
    if (n_vertices) {
      ierr = PetscFree(vertices);CHKERRQ(ierr);
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
    ierr = PCBDDCSetUpCoarseEnvironment(pc,coarse_submat_vals);CHKERRQ(ierr);
    ierr = PetscFree(coarse_submat_vals);CHKERRQ(ierr);
  }
  /* free memory */
  ierr = ISDestroy(&is_R_local);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */

/* BDDC requires metis 5.0.1 for multilevel */
#if defined(PETSC_HAVE_METIS)
#include "metis.h"
#define MetisInt    idx_t
#define MetisScalar real_t
#endif

#undef __FUNCT__
#define __FUNCT__ "PCBDDCSetUpCoarseEnvironment"
static PetscErrorCode PCBDDCSetUpCoarseEnvironment(PC pc,PetscScalar* coarse_submat_vals)
{


  Mat_IS    *matis    = (Mat_IS*)pc->pmat->data;
  PC_BDDC   *pcbddc   = (PC_BDDC*)pc->data;
  PC_IS     *pcis     = (PC_IS*)pc->data;
  MPI_Comm  prec_comm;
  MPI_Comm  coarse_comm;

  MatNullSpace CoarseNullSpace;

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
  PetscMPIInt *ranks_recv=0;
  PetscMPIInt count_recv=0;
  PetscMPIInt rank_coarse_proc_send_to=-1;
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
  PetscInt    dbg_flag=pcbddc->dbg_flag;

  PetscInt      offset,offset2;
  PetscMPIInt   im_active,active_procs;
  PetscInt      *dnz,*onz;

  PetscBool     setsym,issym=PETSC_FALSE;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)pc,&prec_comm);CHKERRQ(ierr);
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
    PetscInt     *auxlocal_primal,*aux_idx;
    PetscMPIInt  mpi_local_primal_size;
    PetscScalar  coarsesum,*array;

    mpi_local_primal_size = (PetscMPIInt)pcbddc->local_primal_size;

    /* Construct needed data structures for message passing */
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
    ierr = PCBDDCGetPrimalVerticesLocalIdx(pc,&i,&aux_idx);CHKERRQ(ierr);
    ierr = PetscMemcpy(auxlocal_primal,aux_idx,i*sizeof(PetscInt));CHKERRQ(ierr);
    ierr = PetscFree(aux_idx);CHKERRQ(ierr);
    ierr = PCBDDCGetPrimalConstraintsLocalIdx(pc,&j,&aux_idx);CHKERRQ(ierr);
    ierr = PetscMemcpy(&auxlocal_primal[i],aux_idx,j*sizeof(PetscInt));CHKERRQ(ierr);
    ierr = PetscFree(aux_idx);CHKERRQ(ierr);
    /* Compute number of coarse dofs */
    ierr = PCBDDCSubsetNumbering(prec_comm,matis->mapping,pcbddc->local_primal_size,auxlocal_primal,NULL,&pcbddc->coarse_size,&pcbddc->local_primal_indices);CHKERRQ(ierr);

    if (dbg_flag) {
      ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"--------------------------------------------------\n");CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"Check coarse indices\n");CHKERRQ(ierr);
      ierr = VecSet(pcis->vec1_N,0.0);CHKERRQ(ierr);
      ierr = VecGetArray(pcis->vec1_N,&array);CHKERRQ(ierr);
      for (i=0;i<pcbddc->local_primal_size;i++) array[auxlocal_primal[i]]=1.0;
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
        if (PetscRealPart(array[i]) > 0.0) array[i] = 1.0/PetscRealPart(array[i]);
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
    if (dbg_flag > 1) {
      ierr = PetscViewerASCIIPrintf(viewer,"Distribution of local primal indices\n");CHKERRQ(ierr);
      ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
      ierr = PetscViewerASCIISynchronizedPrintf(viewer,"Subdomain %04d\n",PetscGlobalRank);CHKERRQ(ierr);
      for (i=0;i<pcbddc->local_primal_size;i++) {
        ierr = PetscViewerASCIISynchronizedPrintf(viewer,"local_primal_indices[%d]=%d \n",i,pcbddc->local_primal_indices[i]);
      }
      ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
    }
  }

  im_active = 0;
  if (pcis->n) im_active = 1;
  ierr = MPI_Allreduce(&im_active,&active_procs,1,MPIU_INT,MPI_SUM,prec_comm);CHKERRQ(ierr);

  /* adapt coarse problem type */
#if defined(PETSC_HAVE_METIS)
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
#else
  pcbddc->coarse_problem_type = PARALLEL_BDDC;
#endif

  switch(pcbddc->coarse_problem_type){

    case(MULTILEVEL_BDDC):   /* we define a coarse mesh where subdomains are elements */
    {
#if defined(PETSC_HAVE_METIS)
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
        for (i=0; i<size_prec_comm; i++) new_ranks[i] = -1;
        for (i=0; i<n_subdomains; i++) new_ranks[old_ranks[i]] = i;
        ierr = PetscViewerASCIIOpen(test_coarse_comm,"test_mat_part.out",&viewer_test);CHKERRQ(ierr);
        k = pcis->n_neigh-1;
        ierr = PetscMalloc(2*sizeof(PetscInt),&ii);
        ii[0]=0;
        ii[1]=k;
        ierr = PetscMalloc(k*sizeof(PetscInt),&jj);
        for (i=0; i<k; i++) jj[i]=new_ranks[pcis->neigh[i+1]];
        ierr = PetscSortInt(k,jj);CHKERRQ(ierr);
        ierr = MatCreateMPIAdj(test_coarse_comm,1,n_subdomains,ii,jj,NULL,&mat_adj);CHKERRQ(ierr);
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
            if (ierr != METIS_OK) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in METIS_PartGraphKway (metis error code %D) called from PCBDDCSetUpCoarseEnvironment\n",ierr);
          } else {
            ierr = METIS_PartGraphRecursive(&faces_nvtxs,&ncon,faces_xadj,faces_adjncy,NULL,NULL,NULL,&n_parts,NULL,NULL,options,&objval,metis_coarse_subdivision);
            if (ierr != METIS_OK) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in METIS_PartGraphRecursive (metis error code %D) called from PCBDDCSetUpCoarseEnvironment\n",ierr);
          }
        } else {
          for (i=0;i<n_subdomains;i++) metis_coarse_subdivision[i]=i;
        }
        ierr = PetscFree(faces_xadj);CHKERRQ(ierr);
        ierr = PetscFree(faces_adjncy);CHKERRQ(ierr);
        ierr = PetscMalloc(size_prec_comm*sizeof(PetscMPIInt),&coarse_subdivision);CHKERRQ(ierr);

        /* copy/cast values avoiding possible type conflicts between PETSc, MPI and METIS */
        for (i=0;i<size_prec_comm;i++) coarse_subdivision[i]=MPI_PROC_NULL;
        for (i=0;i<n_subdomains;i++) coarse_subdivision[ranks_stretching_ratio*i]=(PetscInt)(metis_coarse_subdivision[i]);
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
          if (coarse_subdivision[i]==j) total_count_recv[j]++;
          }
        }
        /* displacements needed for scatterv of total_ranks_recv */
      for (i=1; i<size_coarse_comm; i++) displacements_recv[i]=displacements_recv[i-1]+total_count_recv[i-1];

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
#endif
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
      coarse_mat_type = MATAIJ;
      coarse_pc_type  = PCREDUNDANT;
      coarse_ksp_type  = KSPPREONLY;
      coarse_comm = prec_comm;
      active_rank = rank_prec_comm;
      break;

    case(SEQUENTIAL_BDDC):
      pcbddc->coarse_communications_type = GATHERS_BDDC;
      coarse_mat_type = MATAIJ;
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
      for (i=0;i<size_prec_comm;i++) localsizes2[i]=pcbddc->local_primal_sizes[i]*pcbddc->local_primal_sizes[i];
          localdispl2[0]=0;
      for (i=1;i<size_prec_comm;i++) localdispl2[i]=localsizes2[i-1]+localdispl2[i-1];
          j=0;
      for (i=0;i<size_prec_comm;i++) j+=localsizes2[i];
          ierr = PetscMalloc ( j*sizeof(PetscScalar),&temp_coarse_mat_vals);CHKERRQ(ierr);
        }

        mysize=pcbddc->local_primal_size;
        mysize2=pcbddc->local_primal_size*pcbddc->local_primal_size;
        ierr = PetscMalloc(mysize*sizeof(PetscMPIInt),&send_buffer);CHKERRQ(ierr);
    for (i=0; i<mysize; i++) send_buffer[i]=(PetscMPIInt)pcbddc->local_primal_indices[i];

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
      ierr = MatSetOptionsPrefix(pcbddc->coarse_mat,"coarse_");CHKERRQ(ierr);
      ierr = MatSetFromOptions(pcbddc->coarse_mat);CHKERRQ(ierr);
      ierr = MatSetUp(pcbddc->coarse_mat);CHKERRQ(ierr);
      ierr = MatSetOption(pcbddc->coarse_mat,MAT_ROW_ORIENTED,PETSC_FALSE);CHKERRQ(ierr); /* local values stored in column major */
      ierr = MatSetOption(pcbddc->coarse_mat,MAT_IGNORE_ZERO_ENTRIES,PETSC_TRUE);CHKERRQ(ierr);
    } else {
      ierr = MatCreateIS(coarse_comm,1,PETSC_DECIDE,PETSC_DECIDE,pcbddc->coarse_size,pcbddc->coarse_size,coarse_ISLG,&pcbddc->coarse_mat);CHKERRQ(ierr);
      ierr = MatSetUp(pcbddc->coarse_mat);CHKERRQ(ierr);
      ierr = MatISGetLocalMat(pcbddc->coarse_mat,&matis_coarse_local_mat);CHKERRQ(ierr);
      ierr = MatSetOptionsPrefix(pcbddc->coarse_mat,"coarse_");CHKERRQ(ierr);
      ierr = MatSetFromOptions(pcbddc->coarse_mat);CHKERRQ(ierr);
      ierr = MatSetUp(matis_coarse_local_mat);CHKERRQ(ierr);
      ierr = MatSetOption(matis_coarse_local_mat,MAT_ROW_ORIENTED,PETSC_FALSE);CHKERRQ(ierr); /* local values stored in column major */
      ierr = MatSetOption(matis_coarse_local_mat,MAT_IGNORE_ZERO_ENTRIES,PETSC_TRUE);CHKERRQ(ierr);
    }
    /* preallocation */
    if (pcbddc->coarse_problem_type != MULTILEVEL_BDDC) {

      PetscInt lrows,lcols,bs;

      ierr = MatGetLocalSize(pcbddc->coarse_mat,&lrows,&lcols);CHKERRQ(ierr);
      ierr = MatPreallocateInitialize(coarse_comm,lrows,lcols,dnz,onz);CHKERRQ(ierr);
      ierr = MatGetBlockSize(pcbddc->coarse_mat,&bs);CHKERRQ(ierr);

      if (pcbddc->coarse_problem_type == PARALLEL_BDDC) {

        Vec         vec_dnz,vec_onz;
        PetscScalar *my_dnz,*my_onz,*array;
        PetscInt    *mat_ranges,*row_ownership;
        PetscInt    coarse_index_row,coarse_index_col,owner;

        ierr = VecCreate(prec_comm,&vec_dnz);CHKERRQ(ierr);
        ierr = VecSetBlockSize(vec_dnz,bs);CHKERRQ(ierr);
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
        for (i=0; i<j; i++) dnz[i] = (PetscInt)PetscRealPart(array[i]);

        ierr = VecRestoreArray(vec_dnz,&array);CHKERRQ(ierr);
        ierr = VecGetArray(vec_onz,&array);CHKERRQ(ierr);
        for (i=0;i<j;i++) onz[i] = (PetscInt)PetscRealPart(array[i]);

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
        if (dnz[i]>lcols) dnz[i]=lcols;
        if (onz[i]>pcbddc->coarse_size-lcols) onz[i]=pcbddc->coarse_size-lcols;
      }
      ierr = MatSeqAIJSetPreallocation(pcbddc->coarse_mat,0,dnz);CHKERRQ(ierr);
      ierr = MatMPIAIJSetPreallocation(pcbddc->coarse_mat,0,dnz,0,onz);CHKERRQ(ierr);
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

  /* Compute coarse null space */
  CoarseNullSpace = 0;
  if (pcbddc->NullSpace) {
    ierr = PCBDDCNullSpaceAssembleCoarse(pc,&CoarseNullSpace);CHKERRQ(ierr);
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
      if (CoarseNullSpace) {
        ierr = PCBDDCSetNullSpace(pc_temp,CoarseNullSpace);CHKERRQ(ierr);
      }
      if (dbg_flag) {
        ierr = PetscViewerASCIIPrintf(viewer,"----------------Level %d: Setting up level %d---------------\n",pcbddc->current_level,i);CHKERRQ(ierr);
        ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
      }
    } else {
      if (CoarseNullSpace) {
        ierr = KSPSetNullSpace(pcbddc->coarse_ksp,CoarseNullSpace);CHKERRQ(ierr);
      }
    }
    ierr = KSPSetFromOptions(pcbddc->coarse_ksp);CHKERRQ(ierr);
    ierr = KSPSetUp(pcbddc->coarse_ksp);CHKERRQ(ierr);

    ierr = KSPGetTolerances(pcbddc->coarse_ksp,NULL,NULL,NULL,&j);CHKERRQ(ierr);
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
      if (issym) check_ksp_type = KSPCG;
      else check_ksp_type = KSPGMRES;
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
    ierr = VecSetRandom(check_vec,NULL);CHKERRQ(ierr);
    if (CoarseNullSpace) {
      ierr = MatNullSpaceRemove(CoarseNullSpace,check_vec);CHKERRQ(ierr);
    }
    ierr = MatMult(pcbddc->coarse_mat,check_vec,pcbddc->coarse_rhs);CHKERRQ(ierr);
    /* solve coarse problem */
    ierr = KSPSolve(check_ksp,pcbddc->coarse_rhs,pcbddc->coarse_vec);CHKERRQ(ierr);
    if (CoarseNullSpace) {
      ierr = MatNullSpaceRemove(CoarseNullSpace,pcbddc->coarse_vec);CHKERRQ(ierr);
    }
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
  if (dbg_flag) {
    ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
  }
  ierr = MatNullSpaceDestroy(&CoarseNullSpace);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

