/* TODOLIST

   Solvers
   - Add support for cholesky for coarse solver (similar to local solvers)
   - Propagate ksp prefixes for solvers to mat objects?

   User interface
   - ** DM attached to pc?

   Debugging output
   - * Better management of verbosity levels of debugging output

   Extra
   - *** Is it possible to work with PCBDDCGraph on boundary indices only (less memory consumed)?
   - BDDC with MG framework?

   FETIDP
   - Move FETIDP code to its own classes

   MATIS related operations contained in BDDC code
   - Provide general case for subassembling

*/

#include <../src/ksp/pc/impls/bddc/bddc.h> /*I "petscpc.h" I*/  /* includes for fortran wrappers */
#include <../src/ksp/pc/impls/bddc/bddcprivate.h>
#include <petscblaslapack.h>

/* -------------------------------------------------------------------------- */
#undef __FUNCT__
#define __FUNCT__ "PCSetFromOptions_BDDC"
PetscErrorCode PCSetFromOptions_BDDC(PetscOptionItems *PetscOptionsObject,PC pc)
{
  PC_BDDC        *pcbddc = (PC_BDDC*)pc->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscOptionsHead(PetscOptionsObject,"BDDC options");CHKERRQ(ierr);
  /* Verbose debugging */
  ierr = PetscOptionsInt("-pc_bddc_check_level","Verbose output for PCBDDC (intended for debug)","none",pcbddc->dbg_flag,&pcbddc->dbg_flag,NULL);CHKERRQ(ierr);
  /* Primal space customization */
  ierr = PetscOptionsBool("-pc_bddc_use_local_mat_graph","Use or not adjacency graph of local mat for interface analysis","none",pcbddc->use_local_adj,&pcbddc->use_local_adj,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-pc_bddc_use_vertices","Use or not corner dofs in coarse space","none",pcbddc->use_vertices,&pcbddc->use_vertices,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-pc_bddc_use_edges","Use or not edge constraints in coarse space","none",pcbddc->use_edges,&pcbddc->use_edges,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-pc_bddc_use_faces","Use or not face constraints in coarse space","none",pcbddc->use_faces,&pcbddc->use_faces,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-pc_bddc_use_true_nnsp","Use near null space attached to the matrix without modifications","none",pcbddc->use_nnsp_true,&pcbddc->use_nnsp_true,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-pc_bddc_use_qr_single","Use QR factorization for single constraints on cc (QR is used when multiple constraints are present)","none",pcbddc->use_qr_single,&pcbddc->use_qr_single,NULL);CHKERRQ(ierr);
  /* Change of basis */
  ierr = PetscOptionsBool("-pc_bddc_use_change_of_basis","Use or not internal change of basis on local edge nodes","none",pcbddc->use_change_of_basis,&pcbddc->use_change_of_basis,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-pc_bddc_use_change_on_faces","Use or not internal change of basis on local face nodes","none",pcbddc->use_change_on_faces,&pcbddc->use_change_on_faces,NULL);CHKERRQ(ierr);
  if (!pcbddc->use_change_of_basis) {
    pcbddc->use_change_on_faces = PETSC_FALSE;
  }
  /* Switch between M_2 (default) and M_3 preconditioners (as defined by C. Dohrmann in the ref. article) */
  ierr = PetscOptionsBool("-pc_bddc_switch_static","Switch on static condensation ops around the interface preconditioner","none",pcbddc->switch_static,&pcbddc->switch_static,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-pc_bddc_coarse_redistribute","Number of procs where to redistribute coarse problem","none",pcbddc->redistribute_coarse,&pcbddc->redistribute_coarse,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-pc_bddc_coarsening_ratio","Set coarsening ratio used in multilevel coarsening","none",pcbddc->coarsening_ratio,&pcbddc->coarsening_ratio,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-pc_bddc_levels","Set maximum number of levels for multilevel","none",pcbddc->max_levels,&pcbddc->max_levels,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-pc_bddc_use_coarse_estimates","Use estimated eigenvalues for coarse problem","none",pcbddc->use_coarse_estimates,&pcbddc->use_coarse_estimates,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-pc_bddc_use_deluxe_scaling","Use deluxe scaling for BDDC","none",pcbddc->use_deluxe_scaling,&pcbddc->use_deluxe_scaling,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-pc_bddc_schur_rebuild","Whether or not the interface graph for Schur principal minors has to be rebuilt (i.e. define the interface without any adjacency)","none",pcbddc->sub_schurs_rebuild,&pcbddc->sub_schurs_rebuild,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-pc_bddc_schur_layers","Number of dofs' layers for the computation of principal minors (i.e. -1 uses all dofs)","none",pcbddc->sub_schurs_layers,&pcbddc->sub_schurs_layers,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-pc_bddc_schur_use_useradj","Whether or not the CSR graph specified by the user should be used for computing successive layers (default is to use adj of local mat)","none",pcbddc->sub_schurs_use_useradj,&pcbddc->sub_schurs_use_useradj,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-pc_bddc_deluxe_faster","Faster application of deluxe scaling (requires extra work during setup)","none",pcbddc->faster_deluxe,&pcbddc->faster_deluxe,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-pc_bddc_adaptive_threshold","Threshold to be used for adaptive selection of constraints","none",pcbddc->adaptive_threshold,&pcbddc->adaptive_threshold,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-pc_bddc_adaptive_nmin","Minimum number of constraints per connected components","none",pcbddc->adaptive_nmin,&pcbddc->adaptive_nmin,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-pc_bddc_adaptive_nmax","Maximum number of constraints per connected components","none",pcbddc->adaptive_nmax,&pcbddc->adaptive_nmax,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-pc_bddc_symmetric","Symmetric computation of primal basis functions","none",pcbddc->symmetric_primal,&pcbddc->symmetric_primal,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-pc_bddc_coarse_adj","Number of processors where to map the coarse adjacency list","none",pcbddc->coarse_adj_red,&pcbddc->coarse_adj_red,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
#undef __FUNCT__
#define __FUNCT__ "PCView_BDDC"
static PetscErrorCode PCView_BDDC(PC pc,PetscViewer viewer)
{
  PC_BDDC              *pcbddc = (PC_BDDC*)pc->data;
  PetscErrorCode       ierr;
  PetscBool            isascii,isstring;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&isascii);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERSTRING,&isstring);CHKERRQ(ierr);

  /* In a braindead way, print out anything which the user can control from the command line,
     cribbing from PCSetFromOptions_BDDC */

  /* Nothing printed for the String viewer */

  /* ASCII viewer */
  if (isascii) {
    /* Verbose debugging */
    ierr = PetscViewerASCIIPrintf(viewer,    "  BDDC: Use verbose output: %d\n",pcbddc->dbg_flag);CHKERRQ(ierr);

    /* Primal space customization */
    ierr = PetscViewerASCIIPrintf(viewer,    "  BDDC: Use local mat graph: %d\n",pcbddc->use_local_adj);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,    "  BDDC: Use vertices: %d\n",pcbddc->use_vertices);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,    "  BDDC: Use edges: %d\n",pcbddc->use_edges);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,    "  BDDC: Use faces: %d\n",pcbddc->use_faces);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,    "  BDDC: Use true near null space: %d\n",pcbddc->use_nnsp_true);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,    "  BDDC: Use QR for single constraints on cc: %d\n",pcbddc->use_qr_single);CHKERRQ(ierr);

    /* Change of basis */
    ierr = PetscViewerASCIIPrintf(viewer,    "  BDDC: Use change of basis on local edge nodes: %d\n",pcbddc->use_change_of_basis);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,    "  BDDC: Use change of basis on local face nodes: %d\n",pcbddc->use_change_on_faces);CHKERRQ(ierr);

    /* Switch between M_2 (default) and M_3 preconditioners (as defined by C. Dohrmann in the ref. article) */
    ierr = PetscViewerASCIIPrintf(viewer,    "  BDDC: Switch on static condensation ops around the interface preconditioner: %d\n",pcbddc->switch_static);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,    "  BDDC: Coarse problem restribute procs: %d\n",pcbddc->redistribute_coarse);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,    "  BDDC: Multilevel coarsening ratio: %d\n",pcbddc->coarsening_ratio);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,    "  BDDC: Multilevel max levels: %d\n",pcbddc->max_levels);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,    "  BDDC: Use estimated eigs for coarse problem: %d\n",pcbddc->use_coarse_estimates);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,    "  BDDC: Use deluxe scaling: %d\n",pcbddc->use_deluxe_scaling);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,    "  BDDC: Rebuild interface graph for Schur principal minors: %d\n",pcbddc->sub_schurs_rebuild);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,    "  BDDC: Number of dofs' layers for the computation of principal minors: %d\n",pcbddc->sub_schurs_layers);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,    "  BDDC: Use user CSR graph to compute successive layers: %d\n",pcbddc->sub_schurs_use_useradj);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,    "  BDDC: Fast deluxe scaling: %d\n",pcbddc->faster_deluxe);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,    "  BDDC: Adaptive constraint selection threshold: %g\n",pcbddc->adaptive_threshold);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,    "  BDDC: Min constraints / connected component: %d\n",pcbddc->adaptive_nmin);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,    "  BDDC: Max constraints / connected component: %d\n",pcbddc->adaptive_nmax);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,    "  BDDC: Symmetric computation of primal basis functions: %d\n",pcbddc->symmetric_primal);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,    "  BDDC: Num. Procs. to map coarse adjacency list: %d\n",pcbddc->coarse_adj_red);CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
#undef __FUNCT__
#define __FUNCT__ "PCBDDCSetChangeOfBasisMat_BDDC"
static PetscErrorCode PCBDDCSetChangeOfBasisMat_BDDC(PC pc, Mat change)
{
  PC_BDDC        *pcbddc = (PC_BDDC*)pc->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatDestroy(&pcbddc->user_ChangeOfBasisMatrix);CHKERRQ(ierr);
  ierr = PetscObjectReference((PetscObject)change);CHKERRQ(ierr);
  pcbddc->user_ChangeOfBasisMatrix = change;
  PetscFunctionReturn(0);
}
#undef __FUNCT__
#define __FUNCT__ "PCBDDCSetChangeOfBasisMat"
/*@
 PCBDDCSetChangeOfBasisMat - Set user defined change of basis for dofs

   Collective on PC

   Input Parameters:
+  pc - the preconditioning context
-  change - the change of basis matrix

   Level: intermediate

   Notes:

.seealso: PCBDDC
@*/
PetscErrorCode PCBDDCSetChangeOfBasisMat(PC pc, Mat change)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscValidHeaderSpecific(change,MAT_CLASSID,2);
  PetscCheckSameComm(pc,1,change,2);
  if (pc->mat) {
    PetscInt rows_c,cols_c,rows,cols;
    ierr = MatGetSize(pc->mat,&rows,&cols);CHKERRQ(ierr);
    ierr = MatGetSize(change,&rows_c,&cols_c);CHKERRQ(ierr);
    if (rows_c != rows) SETERRQ2(PetscObjectComm((PetscObject)pc),PETSC_ERR_SUP,"Invalid number of rows for change of basis matrix! %d != %d",rows_c,rows);
    if (cols_c != cols) SETERRQ2(PetscObjectComm((PetscObject)pc),PETSC_ERR_SUP,"Invalid number of columns for change of basis matrix! %d != %d",cols_c,cols);
    ierr = MatGetLocalSize(pc->mat,&rows,&cols);CHKERRQ(ierr);
    ierr = MatGetLocalSize(change,&rows_c,&cols_c);CHKERRQ(ierr);
    if (rows_c != rows) SETERRQ2(PetscObjectComm((PetscObject)pc),PETSC_ERR_SUP,"Invalid number of local rows for change of basis matrix! %d != %d",rows_c,rows);
    if (cols_c != cols) SETERRQ2(PetscObjectComm((PetscObject)pc),PETSC_ERR_SUP,"Invalid number of local columns for change of basis matrix! %d != %d",cols_c,cols);
  }
  ierr = PetscTryMethod(pc,"PCBDDCSetChangeOfBasisMat_C",(PC,Mat),(pc,change));CHKERRQ(ierr);
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

   Collective

   Input Parameters:
+  pc - the preconditioning context
-  PrimalVertices - index set of primal vertices in local numbering (can be empty)

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
  PetscCheckSameComm(pc,1,PrimalVertices,2);
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

   Options Database Keys:
.    -pc_bddc_coarsening_ratio

   Level: intermediate

   Notes:
     Approximatively k subdomains at the finer level will be aggregated into a single subdomain at the coarser level

.seealso: PCBDDC, PCBDDCSetLevels()
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

   Options Database Keys:
.    -pc_bddc_levels

   Level: intermediate

   Notes:
     Default value is 0, i.e. traditional one-level BDDC

.seealso: PCBDDC, PCBDDCSetCoarseningRatio()
@*/
PetscErrorCode PCBDDCSetLevels(PC pc,PetscInt levels)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscValidLogicalCollectiveInt(pc,levels,2);
  if (levels > 99) SETERRQ(PetscObjectComm((PetscObject)pc),PETSC_ERR_SUP,"Maximum number of levels for bddc is 99\n");
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
  /* last user setting takes precendence -> destroy any other customization */
  ierr = ISDestroy(&pcbddc->DirichletBoundariesLocal);CHKERRQ(ierr);
  ierr = ISDestroy(&pcbddc->DirichletBoundaries);CHKERRQ(ierr);
  ierr = PetscObjectReference((PetscObject)DirichletBoundaries);CHKERRQ(ierr);
  pcbddc->DirichletBoundaries = DirichletBoundaries;
  pcbddc->recompute_topography = PETSC_TRUE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCBDDCSetDirichletBoundaries"
/*@
 PCBDDCSetDirichletBoundaries - Set IS defining Dirichlet boundaries for the global problem.

   Collective

   Input Parameters:
+  pc - the preconditioning context
-  DirichletBoundaries - parallel IS defining the Dirichlet boundaries

   Level: intermediate

   Notes:
     Provide the information if you used MatZeroRows/Columns routines. Any process can list any global node

.seealso: PCBDDC, PCBDDCSetDirichletBoundariesLocal(), MatZeroRows(), MatZeroRowsColumns()
@*/
PetscErrorCode PCBDDCSetDirichletBoundaries(PC pc,IS DirichletBoundaries)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscValidHeaderSpecific(DirichletBoundaries,IS_CLASSID,2);
  PetscCheckSameComm(pc,1,DirichletBoundaries,2);
  ierr = PetscTryMethod(pc,"PCBDDCSetDirichletBoundaries_C",(PC,IS),(pc,DirichletBoundaries));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
/* -------------------------------------------------------------------------- */

#undef __FUNCT__
#define __FUNCT__ "PCBDDCSetDirichletBoundariesLocal_BDDC"
static PetscErrorCode PCBDDCSetDirichletBoundariesLocal_BDDC(PC pc,IS DirichletBoundaries)
{
  PC_BDDC  *pcbddc = (PC_BDDC*)pc->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* last user setting takes precendence -> destroy any other customization */
  ierr = ISDestroy(&pcbddc->DirichletBoundariesLocal);CHKERRQ(ierr);
  ierr = ISDestroy(&pcbddc->DirichletBoundaries);CHKERRQ(ierr);
  ierr = PetscObjectReference((PetscObject)DirichletBoundaries);CHKERRQ(ierr);
  pcbddc->DirichletBoundariesLocal = DirichletBoundaries;
  pcbddc->recompute_topography = PETSC_TRUE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCBDDCSetDirichletBoundariesLocal"
/*@
 PCBDDCSetDirichletBoundariesLocal - Set IS defining Dirichlet boundaries for the global problem in local ordering.

   Collective

   Input Parameters:
+  pc - the preconditioning context
-  DirichletBoundaries - parallel IS defining the Dirichlet boundaries (in local ordering)

   Level: intermediate

   Notes:

.seealso: PCBDDC, PCBDDCSetDirichletBoundaries(), MatZeroRows(), MatZeroRowsColumns()
@*/
PetscErrorCode PCBDDCSetDirichletBoundariesLocal(PC pc,IS DirichletBoundaries)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscValidHeaderSpecific(DirichletBoundaries,IS_CLASSID,2);
  PetscCheckSameComm(pc,1,DirichletBoundaries,2);
  ierr = PetscTryMethod(pc,"PCBDDCSetDirichletBoundariesLocal_C",(PC,IS),(pc,DirichletBoundaries));CHKERRQ(ierr);
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
  /* last user setting takes precendence -> destroy any other customization */
  ierr = ISDestroy(&pcbddc->NeumannBoundariesLocal);CHKERRQ(ierr);
  ierr = ISDestroy(&pcbddc->NeumannBoundaries);CHKERRQ(ierr);
  ierr = PetscObjectReference((PetscObject)NeumannBoundaries);CHKERRQ(ierr);
  pcbddc->NeumannBoundaries = NeumannBoundaries;
  pcbddc->recompute_topography = PETSC_TRUE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCBDDCSetNeumannBoundaries"
/*@
 PCBDDCSetNeumannBoundaries - Set IS defining Neumann boundaries for the global problem.

   Collective

   Input Parameters:
+  pc - the preconditioning context
-  NeumannBoundaries - parallel IS defining the Neumann boundaries

   Level: intermediate

   Notes:
     Any process can list any global node

.seealso: PCBDDC, PCBDDCSetNeumannBoundariesLocal()
@*/
PetscErrorCode PCBDDCSetNeumannBoundaries(PC pc,IS NeumannBoundaries)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscValidHeaderSpecific(NeumannBoundaries,IS_CLASSID,2);
  PetscCheckSameComm(pc,1,NeumannBoundaries,2);
  ierr = PetscTryMethod(pc,"PCBDDCSetNeumannBoundaries_C",(PC,IS),(pc,NeumannBoundaries));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
/* -------------------------------------------------------------------------- */

#undef __FUNCT__
#define __FUNCT__ "PCBDDCSetNeumannBoundariesLocal_BDDC"
static PetscErrorCode PCBDDCSetNeumannBoundariesLocal_BDDC(PC pc,IS NeumannBoundaries)
{
  PC_BDDC  *pcbddc = (PC_BDDC*)pc->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* last user setting takes precendence -> destroy any other customization */
  ierr = ISDestroy(&pcbddc->NeumannBoundariesLocal);CHKERRQ(ierr);
  ierr = ISDestroy(&pcbddc->NeumannBoundaries);CHKERRQ(ierr);
  ierr = PetscObjectReference((PetscObject)NeumannBoundaries);CHKERRQ(ierr);
  pcbddc->NeumannBoundariesLocal = NeumannBoundaries;
  pcbddc->recompute_topography = PETSC_TRUE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCBDDCSetNeumannBoundariesLocal"
/*@
 PCBDDCSetNeumannBoundariesLocal - Set IS defining Neumann boundaries for the global problem in local ordering.

   Collective

   Input Parameters:
+  pc - the preconditioning context
-  NeumannBoundaries - parallel IS defining the subdomain part of Neumann boundaries (in local ordering)

   Level: intermediate

   Notes:

.seealso: PCBDDC, PCBDDCSetNeumannBoundaries()
@*/
PetscErrorCode PCBDDCSetNeumannBoundariesLocal(PC pc,IS NeumannBoundaries)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscValidHeaderSpecific(NeumannBoundaries,IS_CLASSID,2);
  PetscCheckSameComm(pc,1,NeumannBoundaries,2);
  ierr = PetscTryMethod(pc,"PCBDDCSetNeumannBoundariesLocal_C",(PC,IS),(pc,NeumannBoundaries));CHKERRQ(ierr);
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
 PCBDDCGetDirichletBoundaries - Get parallel IS for Dirichlet boundaries

   Collective

   Input Parameters:
.  pc - the preconditioning context

   Output Parameters:
.  DirichletBoundaries - index set defining the Dirichlet boundaries

   Level: intermediate

   Notes:
     The IS returned (if any) is the same passed in earlier by the user with PCBDDCSetDirichletBoundaries

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
#define __FUNCT__ "PCBDDCGetDirichletBoundariesLocal_BDDC"
static PetscErrorCode PCBDDCGetDirichletBoundariesLocal_BDDC(PC pc,IS *DirichletBoundaries)
{
  PC_BDDC  *pcbddc = (PC_BDDC*)pc->data;

  PetscFunctionBegin;
  *DirichletBoundaries = pcbddc->DirichletBoundariesLocal;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCBDDCGetDirichletBoundariesLocal"
/*@
 PCBDDCGetDirichletBoundariesLocal - Get parallel IS for Dirichlet boundaries (in local ordering)

   Collective

   Input Parameters:
.  pc - the preconditioning context

   Output Parameters:
.  DirichletBoundaries - index set defining the subdomain part of Dirichlet boundaries

   Level: intermediate

   Notes:
     The IS returned could be the same passed in earlier by the user (if provided with PCBDDCSetDirichletBoundariesLocal) or a global-to-local map of the global IS (if provided with PCBDDCSetDirichletBoundaries).
          In the latter case, the IS will be available after PCSetUp.

.seealso: PCBDDC
@*/
PetscErrorCode PCBDDCGetDirichletBoundariesLocal(PC pc,IS *DirichletBoundaries)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  ierr = PetscUseMethod(pc,"PCBDDCGetDirichletBoundariesLocal_C",(PC,IS*),(pc,DirichletBoundaries));CHKERRQ(ierr);
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
 PCBDDCGetNeumannBoundaries - Get parallel IS for Neumann boundaries

   Collective

   Input Parameters:
.  pc - the preconditioning context

   Output Parameters:
.  NeumannBoundaries - index set defining the Neumann boundaries

   Level: intermediate

   Notes:
     The IS returned (if any) is the same passed in earlier by the user with PCBDDCSetNeumannBoundaries

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
#define __FUNCT__ "PCBDDCGetNeumannBoundariesLocal_BDDC"
static PetscErrorCode PCBDDCGetNeumannBoundariesLocal_BDDC(PC pc,IS *NeumannBoundaries)
{
  PC_BDDC  *pcbddc = (PC_BDDC*)pc->data;

  PetscFunctionBegin;
  *NeumannBoundaries = pcbddc->NeumannBoundariesLocal;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCBDDCGetNeumannBoundariesLocal"
/*@
 PCBDDCGetNeumannBoundariesLocal - Get parallel IS for Neumann boundaries (in local ordering)

   Collective

   Input Parameters:
.  pc - the preconditioning context

   Output Parameters:
.  NeumannBoundaries - index set defining the subdomain part of Neumann boundaries

   Level: intermediate

   Notes:
     The IS returned could be the same passed in earlier by the user (if provided with PCBDDCSetNeumannBoundariesLocal) or a global-to-local map of the global IS (if provided with PCBDDCSetNeumannBoundaries).
          In the latter case, the IS will be available after PCSetUp.

.seealso: PCBDDC
@*/
PetscErrorCode PCBDDCGetNeumannBoundariesLocal(PC pc,IS *NeumannBoundaries)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  ierr = PetscUseMethod(pc,"PCBDDCGetNeumannBoundariesLocal_C",(PC,IS*),(pc,NeumannBoundaries));CHKERRQ(ierr);
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
    ierr = PetscMalloc1(nvtxs+1,&mat_graph->xadj);CHKERRQ(ierr);
    ierr = PetscMalloc1(xadj[nvtxs],&mat_graph->adjncy);CHKERRQ(ierr);
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
 PCBDDCSetLocalAdjacencyGraph - Set adjacency structure (CSR graph) of the local matrix

   Not collective

   Input Parameters:
+  pc - the preconditioning context
.  nvtxs - number of local vertices of the graph (i.e., the size of the local problem)
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
  PetscValidIntPointer(adjncy,4);
  if (copymode != PETSC_COPY_VALUES && copymode != PETSC_OWN_POINTER)  SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_SUP,"Unsupported copy mode %d",copymode);
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
#define __FUNCT__ "PCBDDCSetDofsSplittingLocal_BDDC"
static PetscErrorCode PCBDDCSetDofsSplittingLocal_BDDC(PC pc,PetscInt n_is, IS ISForDofs[])
{
  PC_BDDC  *pcbddc = (PC_BDDC*)pc->data;
  PetscInt i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* Destroy ISes if they were already set */
  for (i=0;i<pcbddc->n_ISForDofsLocal;i++) {
    ierr = ISDestroy(&pcbddc->ISForDofsLocal[i]);CHKERRQ(ierr);
  }
  ierr = PetscFree(pcbddc->ISForDofsLocal);CHKERRQ(ierr);
  /* last user setting takes precendence -> destroy any other customization */
  for (i=0;i<pcbddc->n_ISForDofs;i++) {
    ierr = ISDestroy(&pcbddc->ISForDofs[i]);CHKERRQ(ierr);
  }
  ierr = PetscFree(pcbddc->ISForDofs);CHKERRQ(ierr);
  pcbddc->n_ISForDofs = 0;
  /* allocate space then set */
  if (n_is) {
    ierr = PetscMalloc1(n_is,&pcbddc->ISForDofsLocal);CHKERRQ(ierr);
  }
  for (i=0;i<n_is;i++) {
    ierr = PetscObjectReference((PetscObject)ISForDofs[i]);CHKERRQ(ierr);
    pcbddc->ISForDofsLocal[i]=ISForDofs[i];
  }
  pcbddc->n_ISForDofsLocal=n_is;
  if (n_is) pcbddc->user_provided_isfordofs = PETSC_TRUE;
  pcbddc->recompute_topography = PETSC_TRUE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCBDDCSetDofsSplittingLocal"
/*@
 PCBDDCSetDofsSplittingLocal - Set index sets defining fields of the local subdomain matrix

   Collective

   Input Parameters:
+  pc - the preconditioning context
.  n_is - number of index sets defining the fields
-  ISForDofs - array of IS describing the fields in local ordering

   Level: intermediate

   Notes:
     n_is should be the same among processes. Not all nodes need to be listed: unlisted nodes will belong to the complement field.

.seealso: PCBDDC
@*/
PetscErrorCode PCBDDCSetDofsSplittingLocal(PC pc,PetscInt n_is, IS ISForDofs[])
{
  PetscInt       i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscValidLogicalCollectiveInt(pc,n_is,2);
  for (i=0;i<n_is;i++) {
    PetscCheckSameComm(pc,1,ISForDofs[i],3);
    PetscValidHeaderSpecific(ISForDofs[i],IS_CLASSID,3);
  }
  ierr = PetscTryMethod(pc,"PCBDDCSetDofsSplittingLocal_C",(PC,PetscInt,IS[]),(pc,n_is,ISForDofs));CHKERRQ(ierr);
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
  /* last user setting takes precendence -> destroy any other customization */
  for (i=0;i<pcbddc->n_ISForDofsLocal;i++) {
    ierr = ISDestroy(&pcbddc->ISForDofsLocal[i]);CHKERRQ(ierr);
  }
  ierr = PetscFree(pcbddc->ISForDofsLocal);CHKERRQ(ierr);
  pcbddc->n_ISForDofsLocal = 0;
  /* allocate space then set */
  if (n_is) {
    ierr = PetscMalloc1(n_is,&pcbddc->ISForDofs);CHKERRQ(ierr);
  }
  for (i=0;i<n_is;i++) {
    ierr = PetscObjectReference((PetscObject)ISForDofs[i]);CHKERRQ(ierr);
    pcbddc->ISForDofs[i]=ISForDofs[i];
  }
  pcbddc->n_ISForDofs=n_is;
  if (n_is) pcbddc->user_provided_isfordofs = PETSC_TRUE;
  pcbddc->recompute_topography = PETSC_TRUE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCBDDCSetDofsSplitting"
/*@
 PCBDDCSetDofsSplitting - Set index sets defining fields of the global matrix

   Collective

   Input Parameters:
+  pc - the preconditioning context
.  n_is - number of index sets defining the fields
-  ISForDofs - array of IS describing the fields in global ordering

   Level: intermediate

   Notes:
     Any process can list any global node. Not all nodes need to be listed: unlisted nodes will belong to the complement field.

.seealso: PCBDDC
@*/
PetscErrorCode PCBDDCSetDofsSplitting(PC pc,PetscInt n_is, IS ISForDofs[])
{
  PetscInt       i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscValidLogicalCollectiveInt(pc,n_is,2);
  for (i=0;i<n_is;i++) {
    PetscCheckSameComm(pc,1,ISForDofs[i],3);
    PetscValidHeaderSpecific(ISForDofs[i],IS_CLASSID,3);
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
  Vec            used_vec;
  PetscBool      copy_rhs = PETSC_TRUE;

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
  } else { /* it can only happen when calling PCBDDCMatFETIDPGetRHS */
    ierr = PetscObjectReference((PetscObject)pcbddc->temp_solution);CHKERRQ(ierr);
    used_vec = pcbddc->temp_solution;
    ierr = VecSet(used_vec,0.0);CHKERRQ(ierr);
  }

  /* hack into ksp data structure since PCPreSolve comes earlier than setting to zero the guess in src/ksp/ksp/interface/itfunc.c */
  if (ksp) {
    /* store the flag for the initial guess since it will be restored back during PCPostSolve_BDDC */
    ierr = KSPGetInitialGuessNonzero(ksp,&pcbddc->ksp_guess_nonzero);CHKERRQ(ierr);
    if (!pcbddc->ksp_guess_nonzero) {
      ierr = VecSet(used_vec,0.0);CHKERRQ(ierr);
    }
  }

  pcbddc->rhs_change = PETSC_FALSE;

  /* Take into account zeroed rows -> change rhs and store solution removed */
  if (rhs) {
    IS dirIS = NULL;

    /* DirichletBoundariesLocal may not be consistent among neighbours; gets a dirichlet dofs IS from graph (may be cached) */
    ierr = PCBDDCGraphGetDirichletDofs(pcbddc->mat_graph,&dirIS);CHKERRQ(ierr);
    if (dirIS) {
      Mat_IS            *matis = (Mat_IS*)pc->pmat->data;
      PetscInt          dirsize,i,*is_indices;
      PetscScalar       *array_x;
      const PetscScalar *array_diagonal;

      ierr = MatGetDiagonal(pc->pmat,pcis->vec1_global);CHKERRQ(ierr);
      ierr = VecPointwiseDivide(pcis->vec1_global,rhs,pcis->vec1_global);CHKERRQ(ierr);
      ierr = VecScatterBegin(matis->rctx,pcis->vec1_global,pcis->vec2_N,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
      ierr = VecScatterEnd(matis->rctx,pcis->vec1_global,pcis->vec2_N,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
      ierr = VecScatterBegin(matis->rctx,used_vec,pcis->vec1_N,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
      ierr = VecScatterEnd(matis->rctx,used_vec,pcis->vec1_N,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
      ierr = ISGetLocalSize(dirIS,&dirsize);CHKERRQ(ierr);
      ierr = VecGetArray(pcis->vec1_N,&array_x);CHKERRQ(ierr);
      ierr = VecGetArrayRead(pcis->vec2_N,&array_diagonal);CHKERRQ(ierr);
      ierr = ISGetIndices(dirIS,(const PetscInt**)&is_indices);CHKERRQ(ierr);
      for (i=0; i<dirsize; i++) array_x[is_indices[i]] = array_diagonal[is_indices[i]];
      ierr = ISRestoreIndices(dirIS,(const PetscInt**)&is_indices);CHKERRQ(ierr);
      ierr = VecRestoreArrayRead(pcis->vec2_N,&array_diagonal);CHKERRQ(ierr);
      ierr = VecRestoreArray(pcis->vec1_N,&array_x);CHKERRQ(ierr);
      ierr = VecScatterBegin(matis->rctx,pcis->vec1_N,used_vec,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
      ierr = VecScatterEnd(matis->rctx,pcis->vec1_N,used_vec,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
      pcbddc->rhs_change = PETSC_TRUE;
      ierr = ISDestroy(&dirIS);CHKERRQ(ierr);
    }
  }

  /* remove the computed solution or the initial guess from the rhs */
  if (pcbddc->rhs_change || (ksp && pcbddc->ksp_guess_nonzero) ) {
    /* store the original rhs */
    if (copy_rhs) {
      ierr = VecCopy(rhs,pcbddc->original_rhs);CHKERRQ(ierr);
      copy_rhs = PETSC_FALSE;
    }
    pcbddc->rhs_change = PETSC_TRUE;
    ierr = VecScale(used_vec,-1.0);CHKERRQ(ierr);
    ierr = MatMultAdd(pc->pmat,used_vec,rhs,rhs);CHKERRQ(ierr);
    ierr = VecScale(used_vec,-1.0);CHKERRQ(ierr);
    ierr = VecCopy(used_vec,pcbddc->temp_solution);CHKERRQ(ierr);
    if (ksp) {
      ierr = KSPSetInitialGuessNonzero(ksp,PETSC_FALSE);CHKERRQ(ierr);
    }
  }
  ierr = VecDestroy(&used_vec);CHKERRQ(ierr);

  /* store partially computed solution and set initial guess */
  if (x && pcbddc->use_exact_dirichlet_trick) {
    ierr = VecSet(x,0.0);CHKERRQ(ierr);
    ierr = VecScatterBegin(pcis->global_to_D,rhs,pcis->vec1_D,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterEnd(pcis->global_to_D,rhs,pcis->vec1_D,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = KSPSolve(pcbddc->ksp_D,pcis->vec1_D,pcis->vec2_D);CHKERRQ(ierr);
    ierr = VecScatterBegin(pcis->global_to_D,pcis->vec2_D,x,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
    ierr = VecScatterEnd(pcis->global_to_D,pcis->vec2_D,x,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
    if (ksp) {
      ierr = KSPSetInitialGuessNonzero(ksp,PETSC_TRUE);CHKERRQ(ierr);
    }
  }

  if (pcbddc->ChangeOfBasisMatrix) {
    PCBDDCChange_ctx change_ctx;

    /* get change ctx */
    ierr = MatShellGetContext(pcbddc->new_global_mat,&change_ctx);CHKERRQ(ierr);

    /* set current iteration matrix inside change context (change of basis has been already set into the ctx during PCSetUp) */
    ierr = MatDestroy(&change_ctx->original_mat);CHKERRQ(ierr);
    ierr = PetscObjectReference((PetscObject)pc->mat);CHKERRQ(ierr);
    change_ctx->original_mat = pc->mat;

    /* change iteration matrix */
    ierr = MatDestroy(&pc->mat);CHKERRQ(ierr);
    ierr = PetscObjectReference((PetscObject)pcbddc->new_global_mat);CHKERRQ(ierr);
    pc->mat = pcbddc->new_global_mat;

    /* store the original rhs */
    if (copy_rhs) {
      ierr = VecCopy(rhs,pcbddc->original_rhs);CHKERRQ(ierr);
      copy_rhs = PETSC_FALSE;
    }

    /* change rhs */
    ierr = MatMultTranspose(change_ctx->global_change,rhs,pcis->vec1_global);CHKERRQ(ierr);
    ierr = VecCopy(pcis->vec1_global,rhs);CHKERRQ(ierr);
    pcbddc->rhs_change = PETSC_TRUE;
  }

  /* remove nullspace if present */
  if (ksp && x && pcbddc->NullSpace) {
    ierr = MatNullSpaceRemove(pcbddc->NullSpace,x);CHKERRQ(ierr);
    /* store the original rhs */
    if (copy_rhs) {
      ierr = VecCopy(rhs,pcbddc->original_rhs);CHKERRQ(ierr);
    }
    pcbddc->rhs_change = PETSC_TRUE;
    ierr = MatNullSpaceRemove(pcbddc->NullSpace,rhs);CHKERRQ(ierr);
  }
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

  PetscFunctionBegin;
  if (pcbddc->ChangeOfBasisMatrix) {
    PCBDDCChange_ctx change_ctx;

    /* get change ctx */
    ierr = MatShellGetContext(pcbddc->new_global_mat,&change_ctx);CHKERRQ(ierr);

    /* restore iteration matrix */
    ierr = MatDestroy(&pc->mat);CHKERRQ(ierr);
    ierr = PetscObjectReference((PetscObject)change_ctx->original_mat);CHKERRQ(ierr);
    pc->mat = change_ctx->original_mat;

    /* get solution in original basis */
    if (x) {
      PC_IS *pcis = (PC_IS*)(pc->data);
      ierr = MatMult(change_ctx->global_change,x,pcis->vec1_global);CHKERRQ(ierr);
      ierr = VecCopy(pcis->vec1_global,x);CHKERRQ(ierr);
    }
  }

  /* add solution removed in presolve */
  if (x && pcbddc->rhs_change) {
    ierr = VecAXPY(x,1.0,pcbddc->temp_solution);CHKERRQ(ierr);
  }

  /* restore rhs to its original state */
  if (rhs && pcbddc->rhs_change) {
    ierr = VecCopy(pcbddc->original_rhs,rhs);CHKERRQ(ierr);
  }
  pcbddc->rhs_change = PETSC_FALSE;

  /* restore ksp guess state */
  if (ksp) {
    ierr = KSPSetInitialGuessNonzero(ksp,pcbddc->ksp_guess_nonzero);CHKERRQ(ierr);
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
  Mat_IS*        matis;
  MatNullSpace   nearnullspace;
  PetscInt       nrows,ncols;
  PetscBool      computetopography,computesolvers,computesubschurs;
  PetscBool      computeconstraintsmatrix;
  PetscBool      new_nearnullspace_provided,ismatis;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)pc->pmat,MATIS,&ismatis);CHKERRQ(ierr);
  if (!ismatis) SETERRQ(PetscObjectComm((PetscObject)pc),PETSC_ERR_ARG_WRONG,"PCBDDC preconditioner requires matrix of type MATIS");
  ierr = MatGetSize(pc->pmat,&nrows,&ncols);CHKERRQ(ierr);
  if (nrows != ncols) SETERRQ(PetscObjectComm((PetscObject)pc),PETSC_ERR_SUP,"PCBDDC preconditioner requires a square preconditioning matrix");
  matis = (Mat_IS*)pc->pmat->data;
  /* the following lines of code should be replaced by a better logic between PCIS, PCNN, PCBDDC and other future nonoverlapping preconditioners */
  /* For BDDC we need to define a local "Neumann" problem different to that defined in PCISSetup
     Also, BDDC directly build the Dirichlet problem */
  /* split work */
  if (pc->setupcalled) {
    if (pc->flag == SAME_NONZERO_PATTERN) {
      computetopography = PETSC_FALSE;
      computesolvers = PETSC_TRUE;
    } else { /* DIFFERENT_NONZERO_PATTERN */
      computetopography = PETSC_TRUE;
      computesolvers = PETSC_TRUE;
    }
  } else {
    computetopography = PETSC_TRUE;
    computesolvers = PETSC_TRUE;
  }
  if (pcbddc->recompute_topography) {
    computetopography = PETSC_TRUE;
  }
  computeconstraintsmatrix = PETSC_FALSE;
  if (pcbddc->adaptive_threshold > 0.0 && !pcbddc->use_deluxe_scaling) SETERRQ(PetscObjectComm((PetscObject)pc),PETSC_ERR_SUP,"Cannot compute adaptive constraints without deluxe scaling. Rerun with -pc_bddc_use_deluxe_scaling");
  pcbddc->adaptive_selection = (PetscBool)(pcbddc->adaptive_threshold > 0.0 && pcbddc->use_deluxe_scaling);
  if (pcbddc->adaptive_selection) pcbddc->use_faces = PETSC_TRUE;

  computesubschurs = (PetscBool)(pcbddc->adaptive_selection || pcbddc->use_deluxe_scaling);
  if (pcbddc->faster_deluxe && pcbddc->adaptive_selection && pcbddc->use_change_of_basis) SETERRQ(PetscObjectComm((PetscObject)pc),PETSC_ERR_SUP,"Cannot compute faster deluxe if adaptivity and change of basis are both requested. Rerun with -pc_bddc_deluxe_faster false");

  /* Get stdout for dbg */
  if (pcbddc->dbg_flag) {
    if (!pcbddc->dbg_viewer) {
      pcbddc->dbg_viewer = PETSC_VIEWER_STDOUT_(PetscObjectComm((PetscObject)pc));
      ierr = PetscViewerASCIIPushSynchronized(pcbddc->dbg_viewer);CHKERRQ(ierr);
    }
    ierr = PetscViewerASCIIAddTab(pcbddc->dbg_viewer,2*pcbddc->current_level);CHKERRQ(ierr);
  }

  if (pcbddc->user_ChangeOfBasisMatrix) {
    /* use_change_of_basis flag is used to automatically compute a change of basis from constraints */
    pcbddc->use_change_of_basis = PETSC_FALSE;
    ierr = PCBDDCComputeLocalMatrix(pc,pcbddc->user_ChangeOfBasisMatrix);CHKERRQ(ierr);
  } else {
    ierr = MatDestroy(&pcbddc->local_mat);CHKERRQ(ierr);
    ierr = PetscObjectReference((PetscObject)matis->A);CHKERRQ(ierr);
    pcbddc->local_mat = matis->A;
  }

  /* workaround for reals */
#if !defined(PETSC_USE_COMPLEX)
  if (matis->A->symmetric_set) {
    ierr = MatSetOption(pcbddc->local_mat,MAT_HERMITIAN,matis->A->symmetric);CHKERRQ(ierr);
  }
#endif

  /* Set up all the "iterative substructuring" common block without computing solvers */
  {
    Mat temp_mat;

    temp_mat = matis->A;
    matis->A = pcbddc->local_mat;
    ierr = PCISSetUp(pc,PETSC_FALSE);CHKERRQ(ierr);
    pcbddc->local_mat = matis->A;
    matis->A = temp_mat;
  }

  /* Analyze interface and setup sub_schurs data */
  if (computetopography) {
    ierr = PCBDDCAnalyzeInterface(pc);CHKERRQ(ierr);
    computeconstraintsmatrix = PETSC_TRUE;
  }

  /* Setup local dirichlet solver ksp_D and sub_schurs solvers */
  if (computesolvers) {
    PCBDDCSubSchurs sub_schurs=pcbddc->sub_schurs;

    if (computesubschurs && computetopography) {
      ierr = PCBDDCInitSubSchurs(pc);CHKERRQ(ierr);
    }
    if (sub_schurs->use_mumps) {
      if (computesubschurs) {
        ierr = PCBDDCSetUpSubSchurs(pc);CHKERRQ(ierr);
      }
      ierr = PCBDDCSetUpLocalSolvers(pc,PETSC_TRUE,PETSC_FALSE);CHKERRQ(ierr);
    } else {
      ierr = PCBDDCSetUpLocalSolvers(pc,PETSC_TRUE,PETSC_FALSE);CHKERRQ(ierr);
      if (computesubschurs) {
        ierr = PCBDDCSetUpSubSchurs(pc);CHKERRQ(ierr);
      }
    }
    if (pcbddc->adaptive_selection) {
      ierr = PCBDDCAdaptiveSelection(pc);CHKERRQ(ierr);
      computeconstraintsmatrix = PETSC_TRUE;
    }
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
  if (computeconstraintsmatrix || new_nearnullspace_provided) {
    /* It also sets the primal space flags */
    ierr = PCBDDCConstraintsSetUp(pc);CHKERRQ(ierr);
    /* Allocate needed local vectors (which depends on quantities defined during ConstraintsSetUp) */
    ierr = PCBDDCSetUpLocalWorkVectors(pc);CHKERRQ(ierr);
  }

  if (computesolvers || pcbddc->new_primal_space) {
    if (pcbddc->use_change_of_basis) {
      PC_IS *pcis = (PC_IS*)(pc->data);

      ierr = PCBDDCComputeLocalMatrix(pc,pcbddc->ChangeOfBasisMatrix);CHKERRQ(ierr);
      /* get submatrices */
      ierr = MatDestroy(&pcis->A_IB);CHKERRQ(ierr);
      ierr = MatDestroy(&pcis->A_BI);CHKERRQ(ierr);
      ierr = MatDestroy(&pcis->A_BB);CHKERRQ(ierr);
      ierr = MatGetSubMatrix(pcbddc->local_mat,pcis->is_B_local,pcis->is_B_local,MAT_INITIAL_MATRIX,&pcis->A_BB);CHKERRQ(ierr);
      ierr = MatGetSubMatrix(pcbddc->local_mat,pcis->is_I_local,pcis->is_B_local,MAT_INITIAL_MATRIX,&pcis->A_IB);CHKERRQ(ierr);
      ierr = MatGetSubMatrix(pcbddc->local_mat,pcis->is_B_local,pcis->is_I_local,MAT_INITIAL_MATRIX,&pcis->A_BI);CHKERRQ(ierr);
      /* set flag in pcis to not reuse submatrices during PCISCreate */
      pcis->reusesubmatrices = PETSC_FALSE;
    } else if (!pcbddc->user_ChangeOfBasisMatrix) {
      ierr = MatDestroy(&pcbddc->local_mat);CHKERRQ(ierr);
      ierr = PetscObjectReference((PetscObject)matis->A);CHKERRQ(ierr);
      pcbddc->local_mat = matis->A;
    }
    /* SetUp coarse and local Neumann solvers */
    ierr = PCBDDCSetUpSolvers(pc);CHKERRQ(ierr);
    /* SetUp Scaling operator */
    ierr = PCBDDCScalingSetUp(pc);CHKERRQ(ierr);
  }

  if (pcbddc->dbg_flag) {
    ierr = PetscViewerASCIISubtractTab(pcbddc->dbg_viewer,2*pcbddc->current_level);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
/*
   PCApply_BDDC - Applies the BDDC operator to a vector.

   Input Parameters:
+  pc - the preconditioner context
-  r - input vector (global)

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
  PetscInt          n_B = pcis->n_B, n_D = pcis->n - n_B;
  PetscErrorCode    ierr;
  const PetscScalar one = 1.0;
  const PetscScalar m_one = -1.0;
  const PetscScalar zero = 0.0;

/* This code is similar to that provided in nn.c for PCNN
   NN interface preconditioner changed to BDDC
   Added support for M_3 preconditioner in the reference article (code is active if pcbddc->switch_static == PETSC_TRUE) */

  PetscFunctionBegin;
  if (!pcbddc->use_exact_dirichlet_trick) {
    ierr = VecCopy(r,z);CHKERRQ(ierr);
    /* First Dirichlet solve */
    ierr = VecScatterBegin(pcis->global_to_D,r,pcis->vec1_D,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterEnd(pcis->global_to_D,r,pcis->vec1_D,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    /*
      Assembling right hand side for BDDC operator
      - pcis->vec1_D for the Dirichlet part (if needed, i.e. pcbddc->switch_static == PETSC_TRUE)
      - pcis->vec1_B the interface part of the global vector z
    */
    if (n_D) {
      ierr = KSPSolve(pcbddc->ksp_D,pcis->vec1_D,pcis->vec2_D);CHKERRQ(ierr);
      ierr = VecScale(pcis->vec2_D,m_one);CHKERRQ(ierr);
      if (pcbddc->switch_static) { ierr = MatMultAdd(pcis->A_II,pcis->vec2_D,pcis->vec1_D,pcis->vec1_D);CHKERRQ(ierr); }
      ierr = MatMult(pcis->A_BI,pcis->vec2_D,pcis->vec1_B);CHKERRQ(ierr);
    } else {
      ierr = VecSet(pcis->vec1_B,zero);CHKERRQ(ierr);
    }
    ierr = VecScatterBegin(pcis->global_to_B,pcis->vec1_B,z,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
    ierr = VecScatterEnd(pcis->global_to_B,pcis->vec1_B,z,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
    ierr = PCBDDCScalingRestriction(pc,z,pcis->vec1_B);CHKERRQ(ierr);
  } else {
    if (pcbddc->switch_static) {
      ierr = VecSet(pcis->vec1_D,zero);CHKERRQ(ierr);
    }
    ierr = PCBDDCScalingRestriction(pc,r,pcis->vec1_B);CHKERRQ(ierr);
  }

  /* Apply interface preconditioner
     input/output vecs: pcis->vec1_B and pcis->vec1_D */
  ierr = PCBDDCApplyInterfacePreconditioner(pc,PETSC_FALSE);CHKERRQ(ierr);

  /* Apply transpose of partition of unity operator */
  ierr = PCBDDCScalingExtension(pc,pcis->vec1_B,z);CHKERRQ(ierr);

  /* Second Dirichlet solve and assembling of output */
  ierr = VecScatterBegin(pcis->global_to_B,z,pcis->vec1_B,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(pcis->global_to_B,z,pcis->vec1_B,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  if (n_B) {
    ierr = MatMult(pcis->A_IB,pcis->vec1_B,pcis->vec3_D);CHKERRQ(ierr);
    if (pcbddc->switch_static) { ierr = MatMultAdd(pcis->A_II,pcis->vec1_D,pcis->vec3_D,pcis->vec3_D);CHKERRQ(ierr); }
  } else if (pcbddc->switch_static) {
    ierr = MatMult(pcis->A_II,pcis->vec1_D,pcis->vec3_D);CHKERRQ(ierr);
  }
  ierr = KSPSolve(pcbddc->ksp_D,pcis->vec3_D,pcis->vec4_D);CHKERRQ(ierr);
  if (!pcbddc->use_exact_dirichlet_trick) {
    if (pcbddc->switch_static) {
      ierr = VecAXPBYPCZ(pcis->vec2_D,m_one,one,m_one,pcis->vec4_D,pcis->vec1_D);CHKERRQ(ierr);
    } else {
      ierr = VecAXPBY(pcis->vec2_D,m_one,m_one,pcis->vec4_D);CHKERRQ(ierr);
    }
    ierr = VecScatterBegin(pcis->global_to_D,pcis->vec2_D,z,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
    ierr = VecScatterEnd(pcis->global_to_D,pcis->vec2_D,z,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  } else {
    if (pcbddc->switch_static) {
      ierr = VecAXPBY(pcis->vec4_D,one,m_one,pcis->vec1_D);CHKERRQ(ierr);
    } else {
      ierr = VecScale(pcis->vec4_D,m_one);CHKERRQ(ierr);
    }
    ierr = VecScatterBegin(pcis->global_to_D,pcis->vec4_D,z,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
    ierr = VecScatterEnd(pcis->global_to_D,pcis->vec4_D,z,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
/*
   PCApplyTranspose_BDDC - Applies the transpose of the BDDC operator to a vector.

   Input Parameters:
+  pc - the preconditioner context
-  r - input vector (global)

   Output Parameter:
.  z - output vector (global)

   Application Interface Routine: PCApplyTranspose()
 */
#undef __FUNCT__
#define __FUNCT__ "PCApplyTranspose_BDDC"
PetscErrorCode PCApplyTranspose_BDDC(PC pc,Vec r,Vec z)
{
  PC_IS             *pcis = (PC_IS*)(pc->data);
  PC_BDDC           *pcbddc = (PC_BDDC*)(pc->data);
  PetscInt          n_B = pcis->n_B, n_D = pcis->n - n_B;
  PetscErrorCode    ierr;
  const PetscScalar one = 1.0;
  const PetscScalar m_one = -1.0;
  const PetscScalar zero = 0.0;

  PetscFunctionBegin;
  if (!pcbddc->use_exact_dirichlet_trick) {
    ierr = VecCopy(r,z);CHKERRQ(ierr);
    /* First Dirichlet solve */
    ierr = VecScatterBegin(pcis->global_to_D,r,pcis->vec1_D,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterEnd(pcis->global_to_D,r,pcis->vec1_D,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    /*
      Assembling right hand side for BDDC operator
      - pcis->vec1_D for the Dirichlet part (if needed, i.e. pcbddc->switch_static == PETSC_TRUE)
      - pcis->vec1_B the interface part of the global vector z
    */
    if (n_D) {
      ierr = KSPSolveTranspose(pcbddc->ksp_D,pcis->vec1_D,pcis->vec2_D);CHKERRQ(ierr);
      ierr = VecScale(pcis->vec2_D,m_one);CHKERRQ(ierr);
      if (pcbddc->switch_static) { ierr = MatMultTransposeAdd(pcis->A_II,pcis->vec2_D,pcis->vec1_D,pcis->vec1_D);CHKERRQ(ierr); }
      ierr = MatMultTranspose(pcis->A_IB,pcis->vec2_D,pcis->vec1_B);CHKERRQ(ierr);
    } else {
      ierr = VecSet(pcis->vec1_B,zero);CHKERRQ(ierr);
    }
    ierr = VecScatterBegin(pcis->global_to_B,pcis->vec1_B,z,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
    ierr = VecScatterEnd(pcis->global_to_B,pcis->vec1_B,z,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
    ierr = PCBDDCScalingRestriction(pc,z,pcis->vec1_B);CHKERRQ(ierr);
  } else {
    if (pcbddc->switch_static) {
      ierr = VecSet(pcis->vec1_D,zero);CHKERRQ(ierr);
    }
    ierr = PCBDDCScalingRestriction(pc,r,pcis->vec1_B);CHKERRQ(ierr);
  }

  /* Apply interface preconditioner
     input/output vecs: pcis->vec1_B and pcis->vec1_D */
  ierr = PCBDDCApplyInterfacePreconditioner(pc,PETSC_TRUE);CHKERRQ(ierr);

  /* Apply transpose of partition of unity operator */
  ierr = PCBDDCScalingExtension(pc,pcis->vec1_B,z);CHKERRQ(ierr);

  /* Second Dirichlet solve and assembling of output */
  ierr = VecScatterBegin(pcis->global_to_B,z,pcis->vec1_B,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(pcis->global_to_B,z,pcis->vec1_B,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  if (n_B) {
    ierr = MatMultTranspose(pcis->A_BI,pcis->vec1_B,pcis->vec3_D);CHKERRQ(ierr);
    if (pcbddc->switch_static) { ierr = MatMultTransposeAdd(pcis->A_II,pcis->vec1_D,pcis->vec3_D,pcis->vec3_D);CHKERRQ(ierr); }
  } else if (pcbddc->switch_static) {
    ierr = MatMultTranspose(pcis->A_II,pcis->vec1_D,pcis->vec3_D);CHKERRQ(ierr);
  }
  ierr = KSPSolveTranspose(pcbddc->ksp_D,pcis->vec3_D,pcis->vec4_D);CHKERRQ(ierr);
  if (!pcbddc->use_exact_dirichlet_trick) {
    if (pcbddc->switch_static) {
      ierr = VecAXPBYPCZ(pcis->vec2_D,m_one,one,m_one,pcis->vec4_D,pcis->vec1_D);CHKERRQ(ierr);
    } else {
      ierr = VecAXPBY(pcis->vec2_D,m_one,m_one,pcis->vec4_D);CHKERRQ(ierr);
    }
    ierr = VecScatterBegin(pcis->global_to_D,pcis->vec2_D,z,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
    ierr = VecScatterEnd(pcis->global_to_D,pcis->vec2_D,z,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  } else {
    if (pcbddc->switch_static) {
      ierr = VecAXPBY(pcis->vec4_D,one,m_one,pcis->vec1_D);CHKERRQ(ierr);
    } else {
      ierr = VecScale(pcis->vec4_D,m_one);CHKERRQ(ierr);
    }
    ierr = VecScatterBegin(pcis->global_to_D,pcis->vec4_D,z,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
    ierr = VecScatterEnd(pcis->global_to_D,pcis->vec4_D,z,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  }
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
  /* free allocated sub schurs structure */
  ierr = PetscFree(pcbddc->sub_schurs);CHKERRQ(ierr);
  /* destroy objects for scaling operator */
  ierr = PCBDDCScalingDestroy(pc);CHKERRQ(ierr);
  ierr = PetscFree(pcbddc->deluxe_ctx);CHKERRQ(ierr);
  /* free solvers stuff */
  ierr = PCBDDCResetSolvers(pc);CHKERRQ(ierr);
  /* free global vectors needed in presolve */
  ierr = VecDestroy(&pcbddc->temp_solution);CHKERRQ(ierr);
  ierr = VecDestroy(&pcbddc->original_rhs);CHKERRQ(ierr);
  /* free stuff for change of basis hooks */
  if (pcbddc->new_global_mat) {
    PCBDDCChange_ctx change_ctx;
    ierr = MatShellGetContext(pcbddc->new_global_mat,&change_ctx);CHKERRQ(ierr);
    ierr = MatDestroy(&change_ctx->original_mat);CHKERRQ(ierr);
    ierr = MatDestroy(&change_ctx->global_change);CHKERRQ(ierr);
    ierr = VecDestroyVecs(2,&change_ctx->work);CHKERRQ(ierr);
    ierr = PetscFree(change_ctx);CHKERRQ(ierr);
  }
  ierr = MatDestroy(&pcbddc->new_global_mat);CHKERRQ(ierr);
  /* remove functions */
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCBDDCSetChangeOfBasisMat_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCBDDCSetPrimalVerticesLocalIS_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCBDDCSetCoarseningRatio_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCBDDCSetLevel_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCBDDCSetUseExactDirichlet_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCBDDCSetLevels_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCBDDCSetNullSpace_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCBDDCSetDirichletBoundaries_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCBDDCSetDirichletBoundariesLocal_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCBDDCSetNeumannBoundaries_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCBDDCSetNeumannBoundariesLocal_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCBDDCGetDirichletBoundaries_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCBDDCGetDirichletBoundariesLocal_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCBDDCGetNeumannBoundaries_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCBDDCGetNeumannBoundariesLocal_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCBDDCSetDofsSplitting_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCBDDCSetDofsSplittingLocal_C",NULL);CHKERRQ(ierr);
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
  Vec            copy_standard_rhs;
  PC_IS*         pcis;
  PC_BDDC*       pcbddc;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatShellGetContext(fetidp_mat,&mat_ctx);CHKERRQ(ierr);
  pcis = (PC_IS*)mat_ctx->pc->data;
  pcbddc = (PC_BDDC*)mat_ctx->pc->data;

  /*
     change of basis for physical rhs if needed
     It also changes the rhs in case of dirichlet boundaries
     TODO: better management when FETIDP will have its own class
  */
  ierr = VecDuplicate(standard_rhs,&copy_standard_rhs);CHKERRQ(ierr);
  ierr = VecCopy(standard_rhs,copy_standard_rhs);CHKERRQ(ierr);
  ierr = PCPreSolve_BDDC(mat_ctx->pc,NULL,copy_standard_rhs,NULL);CHKERRQ(ierr);
  /* store vectors for computation of fetidp final solution */
  ierr = VecScatterBegin(pcis->global_to_D,copy_standard_rhs,mat_ctx->temp_solution_D,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(pcis->global_to_D,copy_standard_rhs,mat_ctx->temp_solution_D,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  /* scale rhs since it should be unassembled */
  /* TODO use counter scaling? (also below) */
  ierr = VecScatterBegin(pcis->global_to_B,copy_standard_rhs,mat_ctx->temp_solution_B,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(pcis->global_to_B,copy_standard_rhs,mat_ctx->temp_solution_B,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  /* Apply partition of unity */
  ierr = VecPointwiseMult(mat_ctx->temp_solution_B,pcis->D,mat_ctx->temp_solution_B);CHKERRQ(ierr);
  /* ierr = PCBDDCScalingRestriction(mat_ctx->pc,copy_standard_rhs,mat_ctx->temp_solution_B);CHKERRQ(ierr); */
  if (!pcbddc->switch_static) {
    /* compute partially subassembled Schur complement right-hand side */
    ierr = KSPSolve(pcbddc->ksp_D,mat_ctx->temp_solution_D,pcis->vec1_D);CHKERRQ(ierr);
    ierr = MatMult(pcis->A_BI,pcis->vec1_D,pcis->vec1_B);CHKERRQ(ierr);
    ierr = VecAXPY(mat_ctx->temp_solution_B,-1.0,pcis->vec1_B);CHKERRQ(ierr);
    ierr = VecSet(copy_standard_rhs,0.0);CHKERRQ(ierr);
    ierr = VecScatterBegin(pcis->global_to_B,mat_ctx->temp_solution_B,copy_standard_rhs,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
    ierr = VecScatterEnd(pcis->global_to_B,mat_ctx->temp_solution_B,copy_standard_rhs,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
    /* ierr = PCBDDCScalingRestriction(mat_ctx->pc,copy_standard_rhs,mat_ctx->temp_solution_B);CHKERRQ(ierr); */
    ierr = VecScatterBegin(pcis->global_to_B,copy_standard_rhs,mat_ctx->temp_solution_B,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterEnd(pcis->global_to_B,copy_standard_rhs,mat_ctx->temp_solution_B,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecPointwiseMult(mat_ctx->temp_solution_B,pcis->D,mat_ctx->temp_solution_B);CHKERRQ(ierr);
  }
  ierr = VecDestroy(&copy_standard_rhs);CHKERRQ(ierr);
  /* BDDC rhs */
  ierr = VecCopy(mat_ctx->temp_solution_B,pcis->vec1_B);CHKERRQ(ierr);
  if (pcbddc->switch_static) {
    ierr = VecCopy(mat_ctx->temp_solution_D,pcis->vec1_D);CHKERRQ(ierr);
  }
  /* apply BDDC */
  ierr = PCBDDCApplyInterfacePreconditioner(mat_ctx->pc,PETSC_FALSE);CHKERRQ(ierr);
  /* Application of B_delta and assembling of rhs for fetidp fluxes */
  ierr = VecSet(fetidp_flux_rhs,0.0);CHKERRQ(ierr);
  ierr = MatMult(mat_ctx->B_delta,pcis->vec1_B,mat_ctx->lambda_local);CHKERRQ(ierr);
  ierr = VecScatterBegin(mat_ctx->l2g_lambda,mat_ctx->lambda_local,fetidp_flux_rhs,ADD_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(mat_ctx->l2g_lambda,mat_ctx->lambda_local,fetidp_flux_rhs,ADD_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCBDDCMatFETIDPGetRHS"
/*@
 PCBDDCMatFETIDPGetRHS - Compute the right-hand side for FETI-DP linear system using the physical right-hand side

   Collective

   Input Parameters:
+  fetidp_mat      - the FETI-DP matrix object obtained by a call to PCBDDCCreateFETIDPOperators
-  standard_rhs    - the right-hand side of the original linear system

   Output Parameters:
.  fetidp_flux_rhs - the right-hand side for the FETI-DP linear system

   Level: developer

   Notes:

.seealso: PCBDDC, PCBDDCCreateFETIDPOperators, PCBDDCMatFETIDPGetSolution
@*/
PetscErrorCode PCBDDCMatFETIDPGetRHS(Mat fetidp_mat, Vec standard_rhs, Vec fetidp_flux_rhs)
{
  FETIDPMat_ctx  mat_ctx;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatShellGetContext(fetidp_mat,&mat_ctx);CHKERRQ(ierr);
  ierr = PetscUseMethod(mat_ctx->pc,"PCBDDCMatFETIDPGetRHS_C",(Mat,Vec,Vec),(fetidp_mat,standard_rhs,fetidp_flux_rhs));CHKERRQ(ierr);
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
  ierr = PCBDDCApplyInterfacePreconditioner(mat_ctx->pc,PETSC_FALSE);CHKERRQ(ierr);
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
 PCBDDCMatFETIDPGetSolution - Compute the physical solution using the solution of the FETI-DP linear system

   Collective

   Input Parameters:
+  fetidp_mat      - the FETI-DP matrix obtained by a call to PCBDDCCreateFETIDPOperators
-  fetidp_flux_sol - the solution of the FETI-DP linear system

   Output Parameters:
.  standard_sol    - the solution defined on the physical domain

   Level: developer

   Notes:

.seealso: PCBDDC, PCBDDCCreateFETIDPOperators, PCBDDCMatFETIDPGetRHS
@*/
PetscErrorCode PCBDDCMatFETIDPGetSolution(Mat fetidp_mat, Vec fetidp_flux_sol, Vec standard_sol)
{
  FETIDPMat_ctx  mat_ctx;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatShellGetContext(fetidp_mat,&mat_ctx);CHKERRQ(ierr);
  ierr = PetscUseMethod(mat_ctx->pc,"PCBDDCMatFETIDPGetSolution_C",(Mat,Vec,Vec),(fetidp_mat,fetidp_flux_sol,standard_sol));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
/* -------------------------------------------------------------------------- */

extern PetscErrorCode FETIDPMatMult(Mat,Vec,Vec);
extern PetscErrorCode FETIDPMatMultTranspose(Mat,Vec,Vec);
extern PetscErrorCode PCBDDCDestroyFETIDPMat(Mat);
extern PetscErrorCode FETIDPPCApply(PC,Vec,Vec);
extern PetscErrorCode FETIDPPCApplyTranspose(PC,Vec,Vec);
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
  ierr = MatShellSetOperation(newmat,MATOP_MULT_TRANSPOSE,(void (*)(void))FETIDPMatMultTranspose);CHKERRQ(ierr);
  ierr = MatShellSetOperation(newmat,MATOP_DESTROY,(void (*)(void))PCBDDCDestroyFETIDPMat);CHKERRQ(ierr);
  ierr = MatSetUp(newmat);CHKERRQ(ierr);
  /* FETIDP preconditioner */
  ierr = PCBDDCCreateFETIDPPCContext(pc,&fetidppc_ctx);CHKERRQ(ierr);
  ierr = PCBDDCSetupFETIDPPCContext(newmat,fetidppc_ctx);CHKERRQ(ierr);
  ierr = PCCreate(comm,&newpc);CHKERRQ(ierr);
  ierr = PCSetType(newpc,PCSHELL);CHKERRQ(ierr);
  ierr = PCShellSetContext(newpc,fetidppc_ctx);CHKERRQ(ierr);
  ierr = PCShellSetApply(newpc,FETIDPPCApply);CHKERRQ(ierr);
  ierr = PCShellSetApplyTranspose(newpc,FETIDPPCApplyTranspose);CHKERRQ(ierr);
  ierr = PCShellSetDestroy(newpc,PCBDDCDestroyFETIDPPC);CHKERRQ(ierr);
  ierr = PCSetOperators(newpc,newmat,newmat);CHKERRQ(ierr);
  ierr = PCSetUp(newpc);CHKERRQ(ierr);
  /* return pointers for objects created */
  *fetidp_mat=newmat;
  *fetidp_pc=newpc;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCBDDCCreateFETIDPOperators"
/*@
 PCBDDCCreateFETIDPOperators - Create FETI-DP operators

   Collective

   Input Parameters:
.  pc - the BDDC preconditioning context (setup should have been called before)

   Output Parameters:
+  fetidp_mat - shell FETI-DP matrix object
-  fetidp_pc  - shell Dirichlet preconditioner for FETI-DP matrix

   Options Database Keys:
.    -fetidp_fullyredundant <false> - use or not a fully redundant set of Lagrange multipliers

   Level: developer

   Notes:
     Currently the only operations provided for FETI-DP matrix are MatMult and MatMultTranspose

.seealso: PCBDDC, PCBDDCMatFETIDPGetRHS, PCBDDCMatFETIDPGetSolution
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
   [4] C. Pechstein and C. R. Dohrmann. "Modern domain decomposition methods BDDC, deluxe scaling, and an algebraic approach", Seminar talk, Linz, December 2013, http://people.ricam.oeaw.ac.at/c.pechstein/pechstein-bddc2013.pdf
.ve

   The matrix to be preconditioned (Pmat) must be of type MATIS.

   Currently works with MATIS matrices with local matrices of type MATSEQAIJ, MATSEQBAIJ or MATSEQSBAIJ, either with real or complex numbers.

   It also works with unsymmetric and indefinite problems.

   Unlike 'conventional' interface preconditioners, PCBDDC iterates over all degrees of freedom, not just those on the interface. This allows the use of approximate solvers on the subdomains.

   Approximate local solvers are automatically adapted for singular linear problems (see [1]) if the user has provided the nullspace using PCBDDCSetNullSpace()

   Boundary nodes are split in vertices, edges and faces classes using information from the local to global mapping of dofs and the local connectivity graph of nodes. The latter can be customized by using PCBDDCSetLocalAdjacencyGraph()
   Additional information on dofs can be provided by using PCBDDCSetDofsSplitting(), PCBDDCSetDirichletBoundaries(), PCBDDCSetNeumannBoundaries(), and PCBDDCSetPrimalVerticesLocalIS()

   Constraints can be customized by attaching a MatNullSpace object to the MATIS matrix via MatSetNearNullSpace(). Non-singular modes are retained via SVD.

   Change of basis is performed similarly to [2] when requested. When more than one constraint is present on a single connected component (i.e. an edge or a face), a robust method based on local QR factorizations is used.
   User defined change of basis can be passed to PCBDDC by using PCBDDCSetChangeOfBasisMat()

   The PETSc implementation also supports multilevel BDDC [3]. Coarse grids are partitioned using a MatPartitioning object.

   Adaptive selection of primal constraints [4] is supported for SPD systems with high-contrast in the coefficients if MUMPS is present. Future versions of the code will also consider using MKL_PARDISO or PASTIX.

   An experimental interface to the FETI-DP method is available. FETI-DP operators could be created using PCBDDCCreateFETIDPOperators(). A stand-alone class for the FETI-DP method will be provided in the next releases.
   Deluxe scaling is not supported yet for FETI-DP.

   Options Database Keys (some of them, run with -h for a complete list):

.    -pc_bddc_use_vertices <true> - use or not vertices in primal space
.    -pc_bddc_use_edges <true> - use or not edges in primal space
.    -pc_bddc_use_faces <false> - use or not faces in primal space
.    -pc_bddc_symmetric <true> - symmetric computation of primal basis functions. Specify false for unsymmetric problems
.    -pc_bddc_use_change_of_basis <false> - use change of basis approach (on edges only)
.    -pc_bddc_use_change_on_faces <false> - use change of basis approach on faces if change of basis has been requested
.    -pc_bddc_switch_static <false> - switches from M_2 (default) to M_3 operator (see reference article [1])
.    -pc_bddc_levels <0> - maximum number of levels for multilevel
.    -pc_bddc_coarsening_ratio <8> - number of subdomains which will be aggregated together at the coarser level (e.g. H/h ratio at the coarser level, significative only in the multilevel case)
.    -pc_bddc_redistribute <0> - size of a subset of processors where the coarse problem will be remapped (the value is ignored if not at the coarsest level)
.    -pc_bddc_use_deluxe_scaling <false> - use deluxe scaling
.    -pc_bddc_schur_layers <-1> - select the economic version of deluxe scaling by specifying the number of layers (-1 corresponds to the original deluxe scaling)
.    -pc_bddc_adaptive_threshold <0.0> - when a value greater than one is specified, adaptive selection of constraints is performed on edges and faces (requires deluxe scaling and MUMPS installed)
-    -pc_bddc_check_level <0> - set verbosity level of debugging output

   Options for Dirichlet, Neumann or coarse solver can be set with
.vb
      -pc_bddc_dirichlet_
      -pc_bddc_neumann_
      -pc_bddc_coarse_
.ve
   e.g -pc_bddc_dirichlet_ksp_type richardson -pc_bddc_dirichlet_pc_type gamg. PCBDDC uses by default KPSPREONLY and PCLU.

   When using a multilevel approach, solvers' options at the N-th level (N > 1) can be specified as
.vb
      -pc_bddc_dirichlet_lN_
      -pc_bddc_neumann_lN_
      -pc_bddc_coarse_lN_
.ve
   Note that level number ranges from the finest (0) to the coarsest (N).
   In order to specify options for the BDDC operators at the coarser levels (and not for the solvers), prepend -pc_bddc_coarse_ or -pc_bddc_coarse_l to the option, e.g.
.vb
     -pc_bddc_coarse_pc_bddc_adaptive_threshold 5 -pc_bddc_coarse_l1_pc_bddc_redistribute 3
.ve
   will use a threshold of 5 for constraints' selection at the first coarse level and will redistribute the coarse problem of the first coarse level on 3 processors

   Level: intermediate

   Developer notes:

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
  ierr      = PetscNewLog(pc,&pcbddc);CHKERRQ(ierr);
  pc->data  = (void*)pcbddc;

  /* create PCIS data structure */
  ierr = PCISCreate(pc);CHKERRQ(ierr);

  /* BDDC customization */
  pcbddc->use_local_adj       = PETSC_TRUE;
  pcbddc->use_vertices        = PETSC_TRUE;
  pcbddc->use_edges           = PETSC_TRUE;
  pcbddc->use_faces           = PETSC_FALSE;
  pcbddc->use_change_of_basis = PETSC_FALSE;
  pcbddc->use_change_on_faces = PETSC_FALSE;
  pcbddc->switch_static       = PETSC_FALSE;
  pcbddc->use_nnsp_true       = PETSC_FALSE;
  pcbddc->use_qr_single       = PETSC_FALSE;
  pcbddc->symmetric_primal    = PETSC_TRUE;
  pcbddc->dbg_flag            = 0;
  /* private */
  pcbddc->local_primal_size          = 0;
  pcbddc->local_primal_size_cc       = 0;
  pcbddc->local_primal_ref_node      = 0;
  pcbddc->local_primal_ref_mult      = 0;
  pcbddc->n_vertices                 = 0;
  pcbddc->primal_indices_local_idxs  = 0;
  pcbddc->recompute_topography       = PETSC_FALSE;
  pcbddc->coarse_size                = -1;
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
  pcbddc->user_ChangeOfBasisMatrix   = 0;
  pcbddc->new_global_mat             = 0;
  pcbddc->coarse_vec                 = 0;
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
  pcbddc->NeumannBoundariesLocal     = 0;
  pcbddc->DirichletBoundaries        = 0;
  pcbddc->DirichletBoundariesLocal   = 0;
  pcbddc->user_provided_isfordofs    = PETSC_FALSE;
  pcbddc->n_ISForDofs                = 0;
  pcbddc->n_ISForDofsLocal           = 0;
  pcbddc->ISForDofs                  = 0;
  pcbddc->ISForDofsLocal             = 0;
  pcbddc->ConstraintMatrix           = 0;
  pcbddc->use_exact_dirichlet_trick  = PETSC_TRUE;
  pcbddc->coarse_loc_to_glob         = 0;
  pcbddc->coarsening_ratio           = 8;
  pcbddc->coarse_adj_red             = 0;
  pcbddc->current_level              = 0;
  pcbddc->max_levels                 = 0;
  pcbddc->use_coarse_estimates       = PETSC_FALSE;
  pcbddc->redistribute_coarse        = 0;
  pcbddc->coarse_subassembling       = 0;
  pcbddc->coarse_subassembling_init  = 0;

  /* create local graph structure */
  ierr = PCBDDCGraphCreate(&pcbddc->mat_graph);CHKERRQ(ierr);

  /* scaling */
  pcbddc->work_scaling          = 0;
  pcbddc->use_deluxe_scaling    = PETSC_FALSE;
  pcbddc->faster_deluxe         = PETSC_FALSE;

  /* create sub schurs structure */
  ierr = PCBDDCSubSchursCreate(&pcbddc->sub_schurs);CHKERRQ(ierr);
  pcbddc->sub_schurs_rebuild     = PETSC_FALSE;
  pcbddc->sub_schurs_layers      = -1;
  pcbddc->sub_schurs_use_useradj = PETSC_FALSE;

  pcbddc->computed_rowadj = PETSC_FALSE;

  /* adaptivity */
  pcbddc->adaptive_threshold      = 0.0;
  pcbddc->adaptive_nmax           = 0;
  pcbddc->adaptive_nmin           = 0;

  /* function pointers */
  pc->ops->apply               = PCApply_BDDC;
  pc->ops->applytranspose      = PCApplyTranspose_BDDC;
  pc->ops->setup               = PCSetUp_BDDC;
  pc->ops->destroy             = PCDestroy_BDDC;
  pc->ops->setfromoptions      = PCSetFromOptions_BDDC;
  pc->ops->view                = PCView_BDDC;
  pc->ops->applyrichardson     = 0;
  pc->ops->applysymmetricleft  = 0;
  pc->ops->applysymmetricright = 0;
  pc->ops->presolve            = PCPreSolve_BDDC;
  pc->ops->postsolve           = PCPostSolve_BDDC;

  /* composing function */
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCBDDCSetChangeOfBasisMat_C",PCBDDCSetChangeOfBasisMat_BDDC);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCBDDCSetPrimalVerticesLocalIS_C",PCBDDCSetPrimalVerticesLocalIS_BDDC);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCBDDCSetCoarseningRatio_C",PCBDDCSetCoarseningRatio_BDDC);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCBDDCSetLevel_C",PCBDDCSetLevel_BDDC);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCBDDCSetUseExactDirichlet_C",PCBDDCSetUseExactDirichlet_BDDC);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCBDDCSetLevels_C",PCBDDCSetLevels_BDDC);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCBDDCSetNullSpace_C",PCBDDCSetNullSpace_BDDC);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCBDDCSetDirichletBoundaries_C",PCBDDCSetDirichletBoundaries_BDDC);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCBDDCSetDirichletBoundariesLocal_C",PCBDDCSetDirichletBoundariesLocal_BDDC);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCBDDCSetNeumannBoundaries_C",PCBDDCSetNeumannBoundaries_BDDC);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCBDDCSetNeumannBoundariesLocal_C",PCBDDCSetNeumannBoundariesLocal_BDDC);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCBDDCGetDirichletBoundaries_C",PCBDDCGetDirichletBoundaries_BDDC);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCBDDCGetDirichletBoundariesLocal_C",PCBDDCGetDirichletBoundariesLocal_BDDC);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCBDDCGetNeumannBoundaries_C",PCBDDCGetNeumannBoundaries_BDDC);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCBDDCGetNeumannBoundariesLocal_C",PCBDDCGetNeumannBoundariesLocal_BDDC);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCBDDCSetDofsSplitting_C",PCBDDCSetDofsSplitting_BDDC);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCBDDCSetDofsSplittingLocal_C",PCBDDCSetDofsSplittingLocal_BDDC);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCBDDCSetLocalAdjacencyGraph_C",PCBDDCSetLocalAdjacencyGraph_BDDC);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCBDDCCreateFETIDPOperators_C",PCBDDCCreateFETIDPOperators_BDDC);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCBDDCMatFETIDPGetRHS_C",PCBDDCMatFETIDPGetRHS_BDDC);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCBDDCMatFETIDPGetSolution_C",PCBDDCMatFETIDPGetSolution_BDDC);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

