/* TODOLIST

   ConstraintsSetup
   - tolerances for constraints as an option (take care of single precision!)
   - Can MAT_IGNORE_ZERO_ENTRIES be used for Constraints Matrix?

   Solvers
   - Add support for reuse fill and cholecky factor for coarse solver (similar to local solvers)
   - Propagate ksp prefixes for solvers to mat objects?
   - Propagate nearnullspace info among levels

   User interface
   - Change SetNeumannBoundaries to SetNeumannBoundariesLocal and provide new SetNeumannBoundaries (same Dirichlet)
   - Negative indices in dirichlet and Neumann ISs should be skipped (now they cause out-of-bounds access)
   - Provide PCApplyTranpose_BDDC
   - DofSplitting and DM attached to pc?

   Debugging output
   - Better management of verbosity levels of debugging output

   Build
   - make runexe59

   Extra
   - Is it possible to work with PCBDDCGraph on boundary indices only (less memory consumed)?
   - Why options for "pc_bddc_coarse" solver gets propagated to "pc_bddc_coarse_1" solver?
   - add support for computing h,H and related using coordinates?
   - Change of basis approach does not work with my nonlinear mechanics example. why? (seems not an issue with l2gmap)
   - Better management in PCIS code
   - BDDC with MG framework?

   FETIDP
   - Move FETIDP code to its own classes

   MATIS related operations contained in BDDC code
   - Provide general case for subassembling
   - Preallocation routines in MatISGetMPIAXAIJ

*/

/* ----------------------------------------------------------------------------------------------------------------------------------------------
   Implementation of BDDC preconditioner based on:
   C. Dohrmann "An approximate BDDC preconditioner", Numerical Linear Algebra with Applications Volume 14, Issue 2, pages 149-168, March 2007
   ---------------------------------------------------------------------------------------------------------------------------------------------- */

#include "bddc.h" /*I "petscpc.h" I*/  /* includes for fortran wrappers */
#include "bddcprivate.h"
#include <petscblaslapack.h>

/* -------------------------------------------------------------------------- */
#undef __FUNCT__
#define __FUNCT__ "PCSetFromOptions_BDDC"
PetscErrorCode PCSetFromOptions_BDDC(PC pc)
{
  PC_BDDC        *pcbddc = (PC_BDDC*)pc->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscOptionsHead("BDDC options");CHKERRQ(ierr);
  /* Verbose debugging */
  ierr = PetscOptionsInt("-pc_bddc_check_level","Verbose output for PCBDDC (intended for debug)","none",pcbddc->dbg_flag,&pcbddc->dbg_flag,NULL);CHKERRQ(ierr);
  /* Primal space cumstomization */
  ierr = PetscOptionsBool("-pc_bddc_use_vertices","Use or not corner dofs in coarse space","none",pcbddc->use_vertices,&pcbddc->use_vertices,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-pc_bddc_use_edges","Use or not edge constraints in coarse space","none",pcbddc->use_edges,&pcbddc->use_edges,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-pc_bddc_use_faces","Use or not face constraints in coarse space","none",pcbddc->use_faces,&pcbddc->use_faces,NULL);CHKERRQ(ierr);
  /* Change of basis */
  ierr = PetscOptionsBool("-pc_bddc_use_change_of_basis","Use or not change of basis on local edge nodes","none",pcbddc->use_change_of_basis,&pcbddc->use_change_of_basis,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-pc_bddc_use_change_on_faces","Use or not change of basis on local face nodes","none",pcbddc->use_change_on_faces,&pcbddc->use_change_on_faces,NULL);CHKERRQ(ierr);
  if (!pcbddc->use_change_of_basis) {
    pcbddc->use_change_on_faces = PETSC_FALSE;
  }
  /* Switch between M_2 (default) and M_3 preconditioners (as defined by C. Dohrmann in the ref. article) */
  ierr = PetscOptionsBool("-pc_bddc_switch_static","Switch on static condensation ops around the interface preconditioner","none",pcbddc->switch_static,&pcbddc->switch_static,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-pc_bddc_coarsening_ratio","Set coarsening ratio used in multilevel coarsening","none",pcbddc->coarsening_ratio,&pcbddc->coarsening_ratio,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-pc_bddc_levels","Set maximum number of levels for multilevel","none",pcbddc->max_levels,&pcbddc->max_levels,NULL);CHKERRQ(ierr);
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
 PCBDDCSetPrimalVerticesLocalIS - Set additional user defined primal vertices in PCBDDC

   Not collective

   Input Parameters:
+  pc - the preconditioning context
-  PrimalVertices - index set of primal vertices in local numbering

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
#define __FUNCT__ "PCBDDCSetCoarseningRatio_BDDC"
static PetscErrorCode PCBDDCSetCoarseningRatio_BDDC(PC pc,PetscInt k)
{
  PC_BDDC  *pcbddc = (PC_BDDC*)pc->data;

  PetscFunctionBegin;
  pcbddc->coarsening_ratio = k;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCBDDCSetCoarseningRatio"
/*@
 PCBDDCSetCoarseningRatio - Set coarsening ratio used in multilevel

   Logically collective on PC

   Input Parameters:
+  pc - the preconditioning context
-  k - coarsening ratio (H/h at the coarser level)

   Approximatively k subdomains at the finer level will be aggregated into a single subdomain at the coarser level

   Level: intermediate

   Notes:

.seealso: PCBDDC
@*/
PetscErrorCode PCBDDCSetCoarseningRatio(PC pc,PetscInt k)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscValidLogicalCollectiveInt(pc,k,2);
  ierr = PetscTryMethod(pc,"PCBDDCSetCoarseningRatio_C",(PC,PetscInt),(pc,k));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* The following functions (PCBDDCSetUseExactDirichlet PCBDDCSetLevel) are not public */
#undef __FUNCT__
#define __FUNCT__ "PCBDDCSetUseExactDirichlet_BDDC"
static PetscErrorCode PCBDDCSetUseExactDirichlet_BDDC(PC pc,PetscBool flg)
{
  PC_BDDC  *pcbddc = (PC_BDDC*)pc->data;

  PetscFunctionBegin;
  pcbddc->use_exact_dirichlet_trick = flg;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCBDDCSetUseExactDirichlet"
PetscErrorCode PCBDDCSetUseExactDirichlet(PC pc,PetscBool flg)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscValidLogicalCollectiveBool(pc,flg,2);
  ierr = PetscTryMethod(pc,"PCBDDCSetUseExactDirichlet_C",(PC,PetscBool),(pc,flg));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCBDDCSetLevel_BDDC"
static PetscErrorCode PCBDDCSetLevel_BDDC(PC pc,PetscInt level)
{
  PC_BDDC  *pcbddc = (PC_BDDC*)pc->data;

  PetscFunctionBegin;
  pcbddc->current_level = level;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCBDDCSetLevel"
PetscErrorCode PCBDDCSetLevel(PC pc,PetscInt level)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscValidLogicalCollectiveInt(pc,level,2);
  ierr = PetscTryMethod(pc,"PCBDDCSetLevel_C",(PC,PetscInt),(pc,level));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCBDDCSetLevels_BDDC"
static PetscErrorCode PCBDDCSetLevels_BDDC(PC pc,PetscInt levels)
{
  PC_BDDC  *pcbddc = (PC_BDDC*)pc->data;

  PetscFunctionBegin;
  pcbddc->max_levels = levels;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCBDDCSetLevels"
/*@
 PCBDDCSetLevels - Sets the maximum number of levels for multilevel

   Logically collective on PC

   Input Parameters:
+  pc - the preconditioning context
-  levels - the maximum number of levels (max 9)

   Default value is 0, i.e. traditional one-level BDDC

   Level: intermediate

   Notes:

.seealso: PCBDDC
@*/
PetscErrorCode PCBDDCSetLevels(PC pc,PetscInt levels)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscValidLogicalCollectiveInt(pc,levels,2);
  ierr = PetscTryMethod(pc,"PCBDDCSetLevels_C",(PC,PetscInt),(pc,levels));CHKERRQ(ierr);
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
  pcbddc->NullSpace = NullSpace;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCBDDCSetNullSpace"
/*@
 PCBDDCSetNullSpace - Set nullspace for BDDC operator

   Logically collective on PC and MatNullSpace

   Input Parameters:
+  pc - the preconditioning context
-  NullSpace - Null space of the linear operator to be preconditioned (Pmat)

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
  PetscCheckSameComm(pc,1,NullSpace,2);
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
  pcbddc->recompute_topography = PETSC_TRUE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCBDDCSetDirichletBoundaries"
/*@
 PCBDDCSetDirichletBoundaries - Set IS defining Dirichlet boundaries for the global problem.

   Not collective

   Input Parameters:
+  pc - the preconditioning context
-  DirichletBoundaries - sequential IS defining the subdomain part of Dirichlet boundaries (in local ordering)

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
  pcbddc->recompute_topography = PETSC_TRUE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCBDDCSetNeumannBoundaries"
/*@
 PCBDDCSetNeumannBoundaries - Set IS defining Neumann boundaries for the global problem.

   Not collective

   Input Parameters:
+  pc - the preconditioning context
-  NeumannBoundaries - sequential IS defining the subdomain part of Neumann boundaries (in local ordering)

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
 PCBDDCGetDirichletBoundaries - Get IS for local Dirichlet boundaries

   Not collective

   Input Parameters:
.  pc - the preconditioning context

   Output Parameters:
.  DirichletBoundaries - index set defining the subdomain part of Dirichlet boundaries

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
 PCBDDCGetNeumannBoundaries - Get IS for local Neumann boundaries

   Not collective

   Input Parameters:
.  pc - the preconditioning context

   Output Parameters:
.  NeumannBoundaries - index set defining the subdomain part of Neumann boundaries

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
  /* TODO: PCBDDCGraphSetAdjacency */
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
 PCBDDCSetLocalAdjacencyGraph - Set adjacency structure (CSR graph) of the local Neumann matrix

   Not collective

   Input Parameters:
+  pc - the preconditioning context
.  nvtxs - number of local vertices of the graph (i.e., the local size of your problem)
.  xadj, adjncy - the CSR graph
-  copymode - either PETSC_COPY_VALUES or PETSC_OWN_POINTER.

   Level: intermediate

   Notes:

.seealso: PCBDDC,PetscCopyMode
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
  pcbddc->user_provided_isfordofs = PETSC_TRUE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCBDDCSetDofsSplitting"
/*@
 PCBDDCSetDofsSplitting - Set index sets defining fields of the local Neumann matrix

   Not collective

   Input Parameters:
+  pc - the preconditioning context
-  n_is - number of index sets defining the fields
.  ISForDofs - array of IS describing the fields

   Level: intermediate

   Notes:

.seealso: PCBDDC
@*/
PetscErrorCode PCBDDCSetDofsSplitting(PC pc,PetscInt n_is, IS ISForDofs[])
{
  PetscInt       i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  for (i=0;i<n_is;i++) {
    PetscValidHeaderSpecific(ISForDofs[i],IS_CLASSID,2);
  }
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
  PetscBool      guess_nonzero,flg,bddc_has_dirichlet_boundaries;

  PetscFunctionBegin;
  /* if we are working with cg, one dirichlet solve can be avoided during Krylov iterations */
  if (ksp) {
    PetscBool iscg;
    ierr = PetscObjectTypeCompare((PetscObject)ksp,KSPCG,&iscg);CHKERRQ(ierr);
    if (!iscg) {
      ierr = PCBDDCSetUseExactDirichlet(pc,PETSC_FALSE);CHKERRQ(ierr);
    }
  }
  /* Creates parallel work vectors used in presolve */
  if (!pcbddc->original_rhs) {
    ierr = VecDuplicate(pcis->vec1_global,&pcbddc->original_rhs);CHKERRQ(ierr);
  }
  if (!pcbddc->temp_solution) {
    ierr = VecDuplicate(pcis->vec1_global,&pcbddc->temp_solution);CHKERRQ(ierr);
  }
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
    if (!guess_nonzero) {
      ierr = VecSet(used_vec,0.0);CHKERRQ(ierr);
    }
  }

  /* TODO: remove when Dirichlet boundaries will be shared */
  ierr = PCBDDCGetDirichletBoundaries(pc,&dirIS);CHKERRQ(ierr);
  flg = PETSC_FALSE;
  if (dirIS) flg = PETSC_TRUE;
  ierr = MPI_Allreduce(&flg,&bddc_has_dirichlet_boundaries,1,MPIU_BOOL,MPI_LOR,PetscObjectComm((PetscObject)pc));CHKERRQ(ierr);

  /* store the original rhs */
  ierr = VecCopy(rhs,pcbddc->original_rhs);CHKERRQ(ierr);

  /* Take into account zeroed rows -> change rhs and store solution removed */
  if (rhs && bddc_has_dirichlet_boundaries) {
    ierr = MatGetDiagonal(pc->pmat,pcis->vec1_global);CHKERRQ(ierr);
    ierr = VecPointwiseDivide(pcis->vec1_global,rhs,pcis->vec1_global);CHKERRQ(ierr);
    ierr = VecScatterBegin(matis->ctx,pcis->vec1_global,pcis->vec2_N,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterEnd(matis->ctx,pcis->vec1_global,pcis->vec2_N,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterBegin(matis->ctx,used_vec,pcis->vec1_N,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterEnd(matis->ctx,used_vec,pcis->vec1_N,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
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
  }

  /* store partially computed solution and set initial guess */
  if (x) {
    ierr = VecCopy(used_vec,pcbddc->temp_solution);CHKERRQ(ierr);
    ierr = VecSet(used_vec,0.0);CHKERRQ(ierr);
    if (pcbddc->use_exact_dirichlet_trick) {
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

  /* prepare MatMult and rhs for solver */
  if (pcbddc->use_change_of_basis) {
    /* swap pointers for local matrices */
    temp_mat = matis->A;
    matis->A = pcbddc->local_mat;
    pcbddc->local_mat = temp_mat;
    if (rhs) {
      /* Get local rhs and apply transformation of basis */
      ierr = VecScatterBegin(pcis->global_to_B,rhs,pcis->vec1_B,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
      ierr = VecScatterEnd  (pcis->global_to_B,rhs,pcis->vec1_B,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
      /* from original basis to modified basis */
      ierr = MatMultTranspose(pcbddc->ChangeOfBasisMatrix,pcis->vec1_B,pcis->vec2_B);CHKERRQ(ierr);
      /* put back modified values into the global vec using INSERT_VALUES copy mode */
      ierr = VecScatterBegin(pcis->global_to_B,pcis->vec2_B,rhs,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
      ierr = VecScatterEnd  (pcis->global_to_B,pcis->vec2_B,rhs,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
    }
  }

  /* remove nullspace if present */
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
  }
  if (pcbddc->use_change_of_basis && x) {
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
  /* restore rhs to its original state */
  if (rhs) {
    ierr = VecCopy(pcbddc->original_rhs,rhs);CHKERRQ(ierr);
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
  PetscErrorCode   ierr;
  PC_BDDC*         pcbddc = (PC_BDDC*)pc->data;
  MatNullSpace     nearnullspace;
  MatStructure     flag;
  PetscBool        computeis,computetopography,computesolvers;
  PetscBool        new_nearnullspace_provided;

  PetscFunctionBegin;
  /* the following lines of code should be replaced by a better logic between PCIS, PCNN, PCBDDC and other future nonoverlapping preconditioners */
  /* PCIS does not support MatStructure flags different from SAME_PRECONDITIONER */
  /* For BDDC we need to define a local "Neumann" problem different to that defined in PCISSetup
     Also, BDDC directly build the Dirichlet problem */

  /* split work */
  if (pc->setupcalled) {
    computeis = PETSC_FALSE;
    ierr = PCGetOperators(pc,NULL,NULL,&flag);CHKERRQ(ierr);
    if (flag == SAME_PRECONDITIONER) {
      PetscFunctionReturn(0);
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
  if (pcbddc->recompute_topography) {
    computetopography = PETSC_TRUE;
  }

  /* Get stdout for dbg */
  if (pcbddc->dbg_flag && !pcbddc->dbg_viewer) {
    ierr = PetscViewerASCIIGetStdout(PetscObjectComm((PetscObject)pc),&pcbddc->dbg_viewer);CHKERRQ(ierr);
    ierr = PetscViewerASCIISynchronizedAllow(pcbddc->dbg_viewer,PETSC_TRUE);CHKERRQ(ierr);
    if (pcbddc->current_level) {
      ierr = PetscViewerASCIIAddTab(pcbddc->dbg_viewer,2);CHKERRQ(ierr);
    }
  }

  /* Set up all the "iterative substructuring" common block without computing solvers */
  if (computeis) {
    /* HACK INTO PCIS */
    PC_IS* pcis = (PC_IS*)pc->data;
    pcis->computesolvers = PETSC_FALSE;
    ierr = PCISSetUp(pc);CHKERRQ(ierr);
    ierr = ISLocalToGlobalMappingCreateIS(pcis->is_B_local,&pcbddc->BtoNmap);CHKERRQ(ierr);
  }

  /* Analyze interface */
  if (computetopography) {
    ierr = PCBDDCAnalyzeInterface(pc);CHKERRQ(ierr);
  }

  /* infer if NullSpace object attached to Mat via MatSetNearNullSpace has changed */
  new_nearnullspace_provided = PETSC_FALSE;
  ierr = MatGetNearNullSpace(pc->pmat,&nearnullspace);CHKERRQ(ierr);
  if (pcbddc->onearnullspace) { /* already used nearnullspace */
    if (!nearnullspace) { /* near null space attached to mat has been destroyed */
      new_nearnullspace_provided = PETSC_TRUE;
    } else {
      /* determine if the two nullspaces are different (should be lightweight) */
      if (nearnullspace != pcbddc->onearnullspace) {
        new_nearnullspace_provided = PETSC_TRUE;
      } else { /* maybe the user has changed the content of the nearnullspace so check vectors ObjectStateId */
        PetscInt         i;
        const Vec        *nearnullvecs;
        PetscObjectState state;
        PetscInt         nnsp_size;
        ierr = MatNullSpaceGetVecs(nearnullspace,NULL,&nnsp_size,&nearnullvecs);CHKERRQ(ierr);
        for (i=0;i<nnsp_size;i++) {
          ierr = PetscObjectStateGet((PetscObject)nearnullvecs[i],&state);CHKERRQ(ierr);
          if (pcbddc->onearnullvecs_state[i] != state) {
            new_nearnullspace_provided = PETSC_TRUE;
            break;
          }
        }
      }
    }
  } else {
    if (!nearnullspace) { /* both nearnullspaces are null */
      new_nearnullspace_provided = PETSC_FALSE;
    } else { /* nearnullspace attached later */
      new_nearnullspace_provided = PETSC_TRUE;
    }
  }

  /* Setup constraints and related work vectors */
  /* reset primal space flags */
  pcbddc->new_primal_space = PETSC_FALSE;
  pcbddc->new_primal_space_local = PETSC_FALSE;
  if (computetopography || new_nearnullspace_provided) {
    /* It also sets the primal space flags */
    ierr = PCBDDCConstraintsSetUp(pc);CHKERRQ(ierr);
    /* Allocate needed local vectors (which depends on quantities defined during ConstraintsSetUp) */
    ierr = PCBDDCSetUpLocalWorkVectors(pc);CHKERRQ(ierr);
  }

  if (computesolvers || pcbddc->new_primal_space) {
    /* reset data */
    ierr = PCBDDCScalingDestroy(pc);CHKERRQ(ierr);
    /* Create coarse and local stuffs */
    ierr = PCBDDCSetUpSolvers(pc);CHKERRQ(ierr);
    ierr = PCBDDCScalingSetUp(pc);CHKERRQ(ierr);
  }
  if (pcbddc->dbg_flag && pcbddc->current_level) {
    ierr = PetscViewerASCIISubtractTab(pcbddc->dbg_viewer,2);CHKERRQ(ierr);
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
   Added support for M_3 preconditioner in the reference article (code is active if pcbddc->switch_static = PETSC_TRUE) */

  PetscFunctionBegin;
  if (!pcbddc->use_exact_dirichlet_trick) {
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
    if (pcbddc->switch_static) { ierr = MatMultAdd(pcis->A_II,pcis->vec2_D,pcis->vec1_D,pcis->vec1_D);CHKERRQ(ierr); }
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
  if (pcbddc->switch_static) { ierr = MatMultAdd(pcis->A_II,pcis->vec1_D,pcis->vec3_D,pcis->vec3_D);CHKERRQ(ierr); }
  ierr = KSPSolve(pcbddc->ksp_D,pcis->vec3_D,pcis->vec4_D);CHKERRQ(ierr);
  ierr = VecScale(pcis->vec4_D,m_one);CHKERRQ(ierr);
  if (pcbddc->switch_static) { ierr = VecAXPY (pcis->vec4_D,one,pcis->vec1_D);CHKERRQ(ierr); }
  ierr = VecAXPY (pcis->vec2_D,one,pcis->vec4_D);CHKERRQ(ierr);
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
  /* free global vectors needed in presolve */
  ierr = VecDestroy(&pcbddc->temp_solution);CHKERRQ(ierr);
  ierr = VecDestroy(&pcbddc->original_rhs);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingDestroy(&pcbddc->BtoNmap);CHKERRQ(ierr);
  /* remove functions */
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCBDDCSetPrimalVerticesLocalIS_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCBDDCSetCoarseningRatio_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCBDDCSetLevel_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCBDDCSetUseExactDirichlet_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCBDDCSetLevels_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCBDDCSetNullSpace_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCBDDCSetDirichletBoundaries_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCBDDCSetNeumannBoundaries_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCBDDCGetDirichletBoundaries_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCBDDCGetNeumannBoundaries_C",NULL);CHKERRQ(ierr);
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
  ierr = PCPreSolve_BDDC(mat_ctx->pc,NULL,standard_rhs,NULL);CHKERRQ(ierr);
  /* store vectors for computation of fetidp final solution */
  ierr = VecScatterBegin(pcis->global_to_D,standard_rhs,mat_ctx->temp_solution_D,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(pcis->global_to_D,standard_rhs,mat_ctx->temp_solution_D,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  /* scale rhs since it should be unassembled */
  /* TODO use counter scaling? (also below) */
  ierr = VecScatterBegin(pcis->global_to_B,standard_rhs,mat_ctx->temp_solution_B,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(pcis->global_to_B,standard_rhs,mat_ctx->temp_solution_B,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  /* Apply partition of unity */
  ierr = VecPointwiseMult(mat_ctx->temp_solution_B,pcis->D,mat_ctx->temp_solution_B);CHKERRQ(ierr);
  /* ierr = PCBDDCScalingRestriction(mat_ctx->pc,standard_rhs,mat_ctx->temp_solution_B);CHKERRQ(ierr); */
  if (!pcbddc->switch_static) {
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
  if (pcbddc->switch_static) {
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
 PCBDDCMatFETIDPGetRHS - Compute the right-hand side for FETIDP linear system

   Collective

   Input Parameters:
+  fetidp_mat   - the FETIDP matrix object obtained by calling PCBDDCCreateFETIDPOperators
.  standard_rhs - the right-hand side for your linear system

   Output Parameters:
-  fetidp_flux_rhs   - the right-hand side for the FETIDP linear system

   Level: developer

   Notes:

.seealso: PCBDDC,PCBDDCCreateFETIDPOperators
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
  if (pcbddc->switch_static) {
    ierr = VecCopy(mat_ctx->temp_solution_D,pcis->vec1_D);CHKERRQ(ierr);
  }
  /* apply BDDC */
  ierr = PCBDDCApplyInterfacePreconditioner(mat_ctx->pc);CHKERRQ(ierr);
  /* put values into standard global vector */
  ierr = VecScatterBegin(pcis->global_to_B,pcis->vec1_B,standard_sol,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  ierr = VecScatterEnd  (pcis->global_to_B,pcis->vec1_B,standard_sol,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  if (!pcbddc->switch_static) {
    /* compute values into the interior if solved for the partially subassembled Schur complement */
    ierr = MatMult(pcis->A_IB,pcis->vec1_B,pcis->vec1_D);CHKERRQ(ierr);
    ierr = VecAXPY(mat_ctx->temp_solution_D,-1.0,pcis->vec1_D);CHKERRQ(ierr);
    ierr = KSPSolve(pcbddc->ksp_D,mat_ctx->temp_solution_D,pcis->vec1_D);CHKERRQ(ierr);
  }
  ierr = VecScatterBegin(pcis->global_to_D,pcis->vec1_D,standard_sol,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  ierr = VecScatterEnd  (pcis->global_to_D,pcis->vec1_D,standard_sol,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  /* final change of basis if needed
     Is also sums the dirichlet part removed during RHS assembling */
  ierr = PCPostSolve_BDDC(mat_ctx->pc,NULL,NULL,standard_sol);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCBDDCMatFETIDPGetSolution"
/*@
 PCBDDCMatFETIDPGetSolution - Compute the physical solution from the solution of the FETIDP linear system

   Collective

   Input Parameters:
+  fetidp_mat        - the FETIDP matrix obtained by calling PCBDDCCreateFETIDPOperators
.  fetidp_flux_sol - the solution of the FETIDP linear system

   Output Parameters:
-  standard_sol      - the solution defined on the physical domain

   Level: developer

   Notes:

.seealso: PCBDDC,PCBDDCCreateFETIDPOperators
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
 PCBDDCCreateFETIDPOperators - Create operators for FETIDP

   Collective

   Input Parameters:
+  pc - the BDDC preconditioning context already setup

   Output Parameters:
.  fetidp_mat - shell FETIDP matrix object
.  fetidp_pc  - shell Dirichlet preconditioner for FETIDP matrix

   Options Database Keys:
-    -fetidp_fullyredundant: use or not a fully redundant set of Lagrange multipliers

   Level: developer

   Notes:
     Currently the only operation provided for FETIDP matrix is MatMult

.seealso: PCBDDC
@*/
PetscErrorCode PCBDDCCreateFETIDPOperators(PC pc, Mat *fetidp_mat, PC *fetidp_pc)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  if (pc->setupcalled) {
    ierr = PetscUseMethod(pc,"PCBDDCCreateFETIDPOperators_C",(PC,Mat*,PC*),(pc,fetidp_mat,fetidp_pc));CHKERRQ(ierr);
  } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"You must call PCSetup_BDDC() first \n");
  PetscFunctionReturn(0);
}
/* -------------------------------------------------------------------------- */
/*MC
   PCBDDC - Balancing Domain Decomposition by Constraints.

   An implementation of the BDDC preconditioner based on

.vb
   [1] C. R. Dohrmann. "An approximate BDDC preconditioner", Numerical Linear Algebra with Applications Volume 14, Issue 2, pages 149-168, March 2007
   [2] A. Klawonn and O. B. Widlund. "Dual-Primal FETI Methods for Linear Elasticity", http://cs.nyu.edu/csweb/Research/TechReports/TR2004-855/TR2004-855.pdf
   [3] J. Mandel, B. Sousedik, C. R. Dohrmann. "Multispace and Multilevel BDDC", http://arxiv.org/abs/0712.3977
.ve

   The matrix to be preconditioned (Pmat) must be of type MATIS.

   Currently works with MATIS matrices with local Neumann matrices of type MATSEQAIJ, MATSEQBAIJ or MATSEQSBAIJ, either with real or complex numbers.

   It also works with unsymmetric and indefinite problems.

   Unlike 'conventional' interface preconditioners, PCBDDC iterates over all degrees of freedom, not just those on the interface. This allows the use of approximate solvers on the subdomains.

   Approximate local solvers are automatically adapted for singular linear problems (see [1]) if the user has provided the nullspace using PCBDDCSetNullSpace

   Boundary nodes are split in vertices, edges and faces using information from the local to global mapping of dofs and the local connectivity graph of nodes. The latter can be customized by using PCBDDCSetLocalAdjacencyGraph

   Constraints can be customized by attaching a MatNullSpace object to the MATIS matrix via MatSetNearNullSpace.

   Change of basis is performed similarly to [2] when requested. When more the one constraint is present on a single connected component (i.e. an edge or a face), a robust method based on local QR factorizations is used.

   The PETSc implementation also supports multilevel BDDC [3]. Coarse grids are partitioned using MatPartitioning object.

   Options Database Keys:

.    -pc_bddc_use_vertices <1> - use or not vertices in primal space
.    -pc_bddc_use_edges <1> - use or not edges in primal space
.    -pc_bddc_use_faces <0> - use or not faces in primal space
.    -pc_bddc_use_change_of_basis <0> - use change of basis approach (on edges only)
.    -pc_bddc_use_change_on_faces <0> - use change of basis approach on faces if change of basis has been requested
.    -pc_bddc_switch_static <0> - switches from M_2 to M_3 operator (see reference article [1])
.    -pc_bddc_levels <0> - maximum number of levels for multilevel
.    -pc_bddc_coarsening_ratio - H/h ratio at the coarser level
-    -pc_bddc_check_level <0> - set verbosity level of debugging output

   Options for Dirichlet, Neumann or coarse solver can be set with
.vb
      -pc_bddc_dirichlet_
      -pc_bddc_neumann_
      -pc_bddc_coarse_
.ve
   e.g -pc_bddc_dirichlet_ksp_type richardson -pc_bddc_dirichlet_pc_type gamg

   When using a multilevel approach, solvers' options at the N-th level can be specified as
.vb
      -pc_bddc_dirichlet_N_
      -pc_bddc_neumann_N_
      -pc_bddc_coarse_N_
.ve
   Note that level number ranges from the finest 0 to the coarsest N

   Level: intermediate

   Developer notes:
     Currently does not work with KSPBCGS and other KSPs requiring the specialization of PCApplyTranspose

     New deluxe scaling operator will be available soon.

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

  /* BDDC customization */
  pcbddc->use_vertices        = PETSC_TRUE;
  pcbddc->use_edges           = PETSC_TRUE;
  pcbddc->use_faces           = PETSC_FALSE;
  pcbddc->use_change_of_basis = PETSC_FALSE;
  pcbddc->use_change_on_faces = PETSC_FALSE;
  pcbddc->switch_static       = PETSC_FALSE;
  pcbddc->use_nnsp_true       = PETSC_FALSE; /* not yet exposed */
  pcbddc->dbg_flag            = 0;

  pcbddc->BtoNmap                    = 0;
  pcbddc->local_primal_size          = 0;
  pcbddc->n_vertices                 = 0;
  pcbddc->n_actual_vertices          = 0;
  pcbddc->n_constraints              = 0;
  pcbddc->primal_indices_local_idxs  = 0;
  pcbddc->recompute_topography       = PETSC_FALSE;
  pcbddc->coarse_size                = 0;
  pcbddc->new_primal_space           = PETSC_FALSE;
  pcbddc->new_primal_space_local     = PETSC_FALSE;
  pcbddc->global_primal_indices      = 0;
  pcbddc->onearnullspace             = 0;
  pcbddc->onearnullvecs_state        = 0;
  pcbddc->user_primal_vertices       = 0;
  pcbddc->NullSpace                  = 0;
  pcbddc->temp_solution              = 0;
  pcbddc->original_rhs               = 0;
  pcbddc->local_mat                  = 0;
  pcbddc->ChangeOfBasisMatrix        = 0;
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
  pcbddc->NeumannBoundaries          = 0;
  pcbddc->user_provided_isfordofs    = PETSC_FALSE;
  pcbddc->n_ISForDofs                = 0;
  pcbddc->ISForDofs                  = 0;
  pcbddc->ConstraintMatrix           = 0;
  pcbddc->use_exact_dirichlet_trick  = PETSC_TRUE;
  pcbddc->coarse_loc_to_glob         = 0;
  pcbddc->coarsening_ratio           = 8;
  pcbddc->current_level              = 0;
  pcbddc->max_levels                 = 0;

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
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCBDDCSetLevel_C",PCBDDCSetLevel_BDDC);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCBDDCSetUseExactDirichlet_C",PCBDDCSetUseExactDirichlet_BDDC);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCBDDCSetLevels_C",PCBDDCSetLevels_BDDC);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCBDDCSetNullSpace_C",PCBDDCSetNullSpace_BDDC);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCBDDCSetDirichletBoundaries_C",PCBDDCSetDirichletBoundaries_BDDC);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCBDDCSetNeumannBoundaries_C",PCBDDCSetNeumannBoundaries_BDDC);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCBDDCGetDirichletBoundaries_C",PCBDDCGetDirichletBoundaries_BDDC);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCBDDCGetNeumannBoundaries_C",PCBDDCGetNeumannBoundaries_BDDC);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCBDDCSetDofsSplitting_C",PCBDDCSetDofsSplitting_BDDC);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCBDDCSetLocalAdjacencyGraph_C",PCBDDCSetLocalAdjacencyGraph_BDDC);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCBDDCCreateFETIDPOperators_C",PCBDDCCreateFETIDPOperators_BDDC);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCBDDCMatFETIDPGetRHS_C",PCBDDCMatFETIDPGetRHS_BDDC);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCBDDCMatFETIDPGetSolution_C",PCBDDCMatFETIDPGetSolution_BDDC);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

