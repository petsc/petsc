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

   MATIS related operations contained in BDDC code
   - Provide general case for subassembling

*/

#include <../src/ksp/pc/impls/bddc/bddc.h> /*I "petscpc.h" I*/  /* includes for fortran wrappers */
#include <../src/ksp/pc/impls/bddc/bddcprivate.h>
#include <petscblaslapack.h>

static PetscBool PCBDDCPackageInitialized = PETSC_FALSE;

static PetscBool  cited = PETSC_FALSE;
static const char citation[] =
"@article{ZampiniPCBDDC,\n"
"author = {Stefano Zampini},\n"
"title = {{PCBDDC}: A Class of Robust Dual-Primal Methods in {PETS}c},\n"
"journal = {SIAM Journal on Scientific Computing},\n"
"volume = {38},\n"
"number = {5},\n"
"pages = {S282-S306},\n"
"year = {2016},\n"
"doi = {10.1137/15M1025785},\n"
"URL = {http://dx.doi.org/10.1137/15M1025785},\n"
"eprint = {http://dx.doi.org/10.1137/15M1025785}\n"
"}\n";

PetscLogEvent PC_BDDC_Topology[PETSC_PCBDDC_MAXLEVELS];
PetscLogEvent PC_BDDC_LocalSolvers[PETSC_PCBDDC_MAXLEVELS];
PetscLogEvent PC_BDDC_LocalWork[PETSC_PCBDDC_MAXLEVELS];
PetscLogEvent PC_BDDC_CorrectionSetUp[PETSC_PCBDDC_MAXLEVELS];
PetscLogEvent PC_BDDC_ApproxSetUp[PETSC_PCBDDC_MAXLEVELS];
PetscLogEvent PC_BDDC_ApproxApply[PETSC_PCBDDC_MAXLEVELS];
PetscLogEvent PC_BDDC_CoarseSetUp[PETSC_PCBDDC_MAXLEVELS];
PetscLogEvent PC_BDDC_CoarseSolver[PETSC_PCBDDC_MAXLEVELS];
PetscLogEvent PC_BDDC_AdaptiveSetUp[PETSC_PCBDDC_MAXLEVELS];
PetscLogEvent PC_BDDC_Scaling[PETSC_PCBDDC_MAXLEVELS];
PetscLogEvent PC_BDDC_Schurs[PETSC_PCBDDC_MAXLEVELS];
PetscLogEvent PC_BDDC_Solves[PETSC_PCBDDC_MAXLEVELS][3];

const char *const PCBDDCInterfaceExtTypes[] = {"DIRICHLET","LUMP","PCBDDCInterfaceExtType","PC_BDDC_INTERFACE_EXT_",NULL};

PetscErrorCode PCApply_BDDC(PC,Vec,Vec);

PetscErrorCode PCSetFromOptions_BDDC(PetscOptionItems *PetscOptionsObject,PC pc)
{
  PC_BDDC        *pcbddc = (PC_BDDC*)pc->data;
  PetscInt       nt,i;

  PetscFunctionBegin;
  CHKERRQ(PetscOptionsHead(PetscOptionsObject,"BDDC options"));
  /* Verbose debugging */
  CHKERRQ(PetscOptionsInt("-pc_bddc_check_level","Verbose output for PCBDDC (intended for debug)","none",pcbddc->dbg_flag,&pcbddc->dbg_flag,NULL));
  /* Approximate solvers */
  CHKERRQ(PetscOptionsEnum("-pc_bddc_interface_ext_type","Use DIRICHLET or LUMP to extend interface corrections to interior","PCBDDCSetInterfaceExtType",PCBDDCInterfaceExtTypes,(PetscEnum)pcbddc->interface_extension,(PetscEnum*)&pcbddc->interface_extension,NULL));
  if (pcbddc->interface_extension == PC_BDDC_INTERFACE_EXT_DIRICHLET) {
    CHKERRQ(PetscOptionsBool("-pc_bddc_dirichlet_approximate","Inform PCBDDC that we are using approximate Dirichlet solvers","none",pcbddc->NullSpace_corr[0],&pcbddc->NullSpace_corr[0],NULL));
    CHKERRQ(PetscOptionsBool("-pc_bddc_dirichlet_approximate_scale","Inform PCBDDC that we need to scale the Dirichlet solve","none",pcbddc->NullSpace_corr[1],&pcbddc->NullSpace_corr[1],NULL));
  } else {
    /* This flag is needed/implied by lumping */
    pcbddc->switch_static = PETSC_TRUE;
  }
  CHKERRQ(PetscOptionsBool("-pc_bddc_neumann_approximate","Inform PCBDDC that we are using approximate Neumann solvers","none",pcbddc->NullSpace_corr[2],&pcbddc->NullSpace_corr[2],NULL));
  CHKERRQ(PetscOptionsBool("-pc_bddc_neumann_approximate_scale","Inform PCBDDC that we need to scale the Neumann solve","none",pcbddc->NullSpace_corr[3],&pcbddc->NullSpace_corr[3],NULL));
  /* Primal space customization */
  CHKERRQ(PetscOptionsBool("-pc_bddc_use_local_mat_graph","Use or not adjacency graph of local mat for interface analysis","none",pcbddc->use_local_adj,&pcbddc->use_local_adj,NULL));
  CHKERRQ(PetscOptionsInt("-pc_bddc_graph_maxcount","Maximum number of shared subdomains for a connected component","none",pcbddc->graphmaxcount,&pcbddc->graphmaxcount,NULL));
  CHKERRQ(PetscOptionsBool("-pc_bddc_corner_selection","Activates face-based corner selection","none",pcbddc->corner_selection,&pcbddc->corner_selection,NULL));
  CHKERRQ(PetscOptionsBool("-pc_bddc_use_vertices","Use or not corner dofs in coarse space","none",pcbddc->use_vertices,&pcbddc->use_vertices,NULL));
  CHKERRQ(PetscOptionsBool("-pc_bddc_use_edges","Use or not edge constraints in coarse space","none",pcbddc->use_edges,&pcbddc->use_edges,NULL));
  CHKERRQ(PetscOptionsBool("-pc_bddc_use_faces","Use or not face constraints in coarse space","none",pcbddc->use_faces,&pcbddc->use_faces,NULL));
  CHKERRQ(PetscOptionsInt("-pc_bddc_vertex_size","Connected components smaller or equal to vertex size will be considered as primal vertices","none",pcbddc->vertex_size,&pcbddc->vertex_size,NULL));
  CHKERRQ(PetscOptionsBool("-pc_bddc_use_nnsp","Use near null space attached to the matrix to compute constraints","none",pcbddc->use_nnsp,&pcbddc->use_nnsp,NULL));
  CHKERRQ(PetscOptionsBool("-pc_bddc_use_nnsp_true","Use near null space attached to the matrix to compute constraints as is","none",pcbddc->use_nnsp_true,&pcbddc->use_nnsp_true,NULL));
  CHKERRQ(PetscOptionsBool("-pc_bddc_use_qr_single","Use QR factorization for single constraints on cc (QR is always used when multiple constraints are present)","none",pcbddc->use_qr_single,&pcbddc->use_qr_single,NULL));
  /* Change of basis */
  CHKERRQ(PetscOptionsBool("-pc_bddc_use_change_of_basis","Use or not internal change of basis on local edge nodes","none",pcbddc->use_change_of_basis,&pcbddc->use_change_of_basis,NULL));
  CHKERRQ(PetscOptionsBool("-pc_bddc_use_change_on_faces","Use or not internal change of basis on local face nodes","none",pcbddc->use_change_on_faces,&pcbddc->use_change_on_faces,NULL));
  if (!pcbddc->use_change_of_basis) {
    pcbddc->use_change_on_faces = PETSC_FALSE;
  }
  /* Switch between M_2 (default) and M_3 preconditioners (as defined by C. Dohrmann in the ref. article) */
  CHKERRQ(PetscOptionsBool("-pc_bddc_switch_static","Switch on static condensation ops around the interface preconditioner","none",pcbddc->switch_static,&pcbddc->switch_static,NULL));
  CHKERRQ(PetscOptionsInt("-pc_bddc_coarse_eqs_per_proc","Target number of equations per process for coarse problem redistribution (significant only at the coarsest level)","none",pcbddc->coarse_eqs_per_proc,&pcbddc->coarse_eqs_per_proc,NULL));
  i    = pcbddc->coarsening_ratio;
  CHKERRQ(PetscOptionsInt("-pc_bddc_coarsening_ratio","Set coarsening ratio used in multilevel coarsening","PCBDDCSetCoarseningRatio",i,&i,NULL));
  CHKERRQ(PCBDDCSetCoarseningRatio(pc,i));
  i    = pcbddc->max_levels;
  CHKERRQ(PetscOptionsInt("-pc_bddc_levels","Set maximum number of levels for multilevel","PCBDDCSetLevels",i,&i,NULL));
  CHKERRQ(PCBDDCSetLevels(pc,i));
  CHKERRQ(PetscOptionsInt("-pc_bddc_coarse_eqs_limit","Set maximum number of equations on coarsest grid to aim for","none",pcbddc->coarse_eqs_limit,&pcbddc->coarse_eqs_limit,NULL));
  CHKERRQ(PetscOptionsBool("-pc_bddc_use_coarse_estimates","Use estimated eigenvalues for coarse problem","none",pcbddc->use_coarse_estimates,&pcbddc->use_coarse_estimates,NULL));
  CHKERRQ(PetscOptionsBool("-pc_bddc_use_deluxe_scaling","Use deluxe scaling for BDDC","none",pcbddc->use_deluxe_scaling,&pcbddc->use_deluxe_scaling,NULL));
  CHKERRQ(PetscOptionsBool("-pc_bddc_schur_rebuild","Whether or not the interface graph for Schur principal minors has to be rebuilt (i.e. define the interface without any adjacency)","none",pcbddc->sub_schurs_rebuild,&pcbddc->sub_schurs_rebuild,NULL));
  CHKERRQ(PetscOptionsInt("-pc_bddc_schur_layers","Number of dofs' layers for the computation of principal minors (i.e. -1 uses all dofs)","none",pcbddc->sub_schurs_layers,&pcbddc->sub_schurs_layers,NULL));
  CHKERRQ(PetscOptionsBool("-pc_bddc_schur_use_useradj","Whether or not the CSR graph specified by the user should be used for computing successive layers (default is to use adj of local mat)","none",pcbddc->sub_schurs_use_useradj,&pcbddc->sub_schurs_use_useradj,NULL));
  CHKERRQ(PetscOptionsBool("-pc_bddc_schur_exact","Whether or not to use the exact Schur complement instead of the reduced one (which excludes size 1 cc)","none",pcbddc->sub_schurs_exact_schur,&pcbddc->sub_schurs_exact_schur,NULL));
  CHKERRQ(PetscOptionsBool("-pc_bddc_deluxe_zerorows","Zero rows and columns of deluxe operators associated with primal dofs","none",pcbddc->deluxe_zerorows,&pcbddc->deluxe_zerorows,NULL));
  CHKERRQ(PetscOptionsBool("-pc_bddc_deluxe_singlemat","Collapse deluxe operators","none",pcbddc->deluxe_singlemat,&pcbddc->deluxe_singlemat,NULL));
  CHKERRQ(PetscOptionsBool("-pc_bddc_adaptive_userdefined","Use user-defined constraints (should be attached via MatSetNearNullSpace to pmat) in addition to those adaptively generated","none",pcbddc->adaptive_userdefined,&pcbddc->adaptive_userdefined,NULL));
  nt   = 2;
  CHKERRQ(PetscOptionsRealArray("-pc_bddc_adaptive_threshold","Thresholds to be used for adaptive selection of constraints","none",pcbddc->adaptive_threshold,&nt,NULL));
  if (nt == 1) pcbddc->adaptive_threshold[1] = pcbddc->adaptive_threshold[0];
  CHKERRQ(PetscOptionsInt("-pc_bddc_adaptive_nmin","Minimum number of constraints per connected components","none",pcbddc->adaptive_nmin,&pcbddc->adaptive_nmin,NULL));
  CHKERRQ(PetscOptionsInt("-pc_bddc_adaptive_nmax","Maximum number of constraints per connected components","none",pcbddc->adaptive_nmax,&pcbddc->adaptive_nmax,NULL));
  CHKERRQ(PetscOptionsBool("-pc_bddc_symmetric","Symmetric computation of primal basis functions","none",pcbddc->symmetric_primal,&pcbddc->symmetric_primal,NULL));
  CHKERRQ(PetscOptionsInt("-pc_bddc_coarse_adj","Number of processors where to map the coarse adjacency list","none",pcbddc->coarse_adj_red,&pcbddc->coarse_adj_red,NULL));
  CHKERRQ(PetscOptionsBool("-pc_bddc_benign_trick","Apply the benign subspace trick to saddle point problems with discontinuous pressures","none",pcbddc->benign_saddle_point,&pcbddc->benign_saddle_point,NULL));
  CHKERRQ(PetscOptionsBool("-pc_bddc_benign_change","Compute the pressure change of basis explicitly","none",pcbddc->benign_change_explicit,&pcbddc->benign_change_explicit,NULL));
  CHKERRQ(PetscOptionsBool("-pc_bddc_benign_compute_correction","Compute the benign correction during PreSolve","none",pcbddc->benign_compute_correction,&pcbddc->benign_compute_correction,NULL));
  CHKERRQ(PetscOptionsBool("-pc_bddc_nonetflux","Automatic computation of no-net-flux quadrature weights","none",pcbddc->compute_nonetflux,&pcbddc->compute_nonetflux,NULL));
  CHKERRQ(PetscOptionsBool("-pc_bddc_detect_disconnected","Detects disconnected subdomains","none",pcbddc->detect_disconnected,&pcbddc->detect_disconnected,NULL));
  CHKERRQ(PetscOptionsBool("-pc_bddc_detect_disconnected_filter","Filters out small entries in the local matrix when detecting disconnected subdomains","none",pcbddc->detect_disconnected_filter,&pcbddc->detect_disconnected_filter,NULL));
  CHKERRQ(PetscOptionsBool("-pc_bddc_eliminate_dirichlet","Whether or not we want to eliminate dirichlet dofs during presolve","none",pcbddc->eliminate_dirdofs,&pcbddc->eliminate_dirdofs,NULL));
  CHKERRQ(PetscOptionsTail());
  PetscFunctionReturn(0);
}

static PetscErrorCode PCView_BDDC(PC pc,PetscViewer viewer)
{
  PC_BDDC              *pcbddc = (PC_BDDC*)pc->data;
  PC_IS                *pcis = (PC_IS*)pc->data;
  PetscBool            isascii;
  PetscSubcomm         subcomm;
  PetscViewer          subviewer;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&isascii));
  /* ASCII viewer */
  if (isascii) {
    PetscMPIInt   color,rank,size;
    PetscInt64    loc[7],gsum[6],gmax[6],gmin[6],totbenign;
    PetscScalar   interface_size;
    PetscReal     ratio1=0.,ratio2=0.;
    Vec           counter;

    if (!pc->setupcalled) {
      CHKERRQ(PetscViewerASCIIPrintf(viewer,"  Partial information available: preconditioner has not been setup yet\n"));
    }
    CHKERRQ(PetscViewerASCIIPrintf(viewer,"  Use verbose output: %D\n",pcbddc->dbg_flag));
    CHKERRQ(PetscViewerASCIIPrintf(viewer,"  Use user-defined CSR: %d\n",!!pcbddc->mat_graph->nvtxs_csr));
    CHKERRQ(PetscViewerASCIIPrintf(viewer,"  Use local mat graph: %d\n",pcbddc->use_local_adj && !pcbddc->mat_graph->nvtxs_csr));
    if (pcbddc->mat_graph->twodim) {
      CHKERRQ(PetscViewerASCIIPrintf(viewer,"  Connectivity graph topological dimension: 2\n"));
    } else {
      CHKERRQ(PetscViewerASCIIPrintf(viewer,"  Connectivity graph topological dimension: 3\n"));
    }
    if (pcbddc->graphmaxcount != PETSC_MAX_INT) {
      CHKERRQ(PetscViewerASCIIPrintf(viewer,"  Graph max count: %D\n",pcbddc->graphmaxcount));
    }
    CHKERRQ(PetscViewerASCIIPrintf(viewer,"  Corner selection: %d (selected %d)\n",pcbddc->corner_selection,pcbddc->corner_selected));
    CHKERRQ(PetscViewerASCIIPrintf(viewer,"  Use vertices: %d (vertex size %D)\n",pcbddc->use_vertices,pcbddc->vertex_size));
    CHKERRQ(PetscViewerASCIIPrintf(viewer,"  Use edges: %d\n",pcbddc->use_edges));
    CHKERRQ(PetscViewerASCIIPrintf(viewer,"  Use faces: %d\n",pcbddc->use_faces));
    CHKERRQ(PetscViewerASCIIPrintf(viewer,"  Use true near null space: %d\n",pcbddc->use_nnsp_true));
    CHKERRQ(PetscViewerASCIIPrintf(viewer,"  Use QR for single constraints on cc: %d\n",pcbddc->use_qr_single));
    CHKERRQ(PetscViewerASCIIPrintf(viewer,"  Use change of basis on local edge nodes: %d\n",pcbddc->use_change_of_basis));
    CHKERRQ(PetscViewerASCIIPrintf(viewer,"  Use change of basis on local face nodes: %d\n",pcbddc->use_change_on_faces));
    CHKERRQ(PetscViewerASCIIPrintf(viewer,"  User defined change of basis matrix: %d\n",!!pcbddc->user_ChangeOfBasisMatrix));
    CHKERRQ(PetscViewerASCIIPrintf(viewer,"  Has change of basis matrix: %d\n",!!pcbddc->ChangeOfBasisMatrix));
    CHKERRQ(PetscViewerASCIIPrintf(viewer,"  Eliminate dirichlet boundary dofs: %d\n",pcbddc->eliminate_dirdofs));
    CHKERRQ(PetscViewerASCIIPrintf(viewer,"  Switch on static condensation ops around the interface preconditioner: %d\n",pcbddc->switch_static));
    CHKERRQ(PetscViewerASCIIPrintf(viewer,"  Use exact dirichlet trick: %d\n",pcbddc->use_exact_dirichlet_trick));
    CHKERRQ(PetscViewerASCIIPrintf(viewer,"  Interface extension: %s\n",PCBDDCInterfaceExtTypes[pcbddc->interface_extension]));
    CHKERRQ(PetscViewerASCIIPrintf(viewer,"  Multilevel max levels: %D\n",pcbddc->max_levels));
    CHKERRQ(PetscViewerASCIIPrintf(viewer,"  Multilevel coarsening ratio: %D\n",pcbddc->coarsening_ratio));
    CHKERRQ(PetscViewerASCIIPrintf(viewer,"  Use estimated eigs for coarse problem: %d\n",pcbddc->use_coarse_estimates));
    CHKERRQ(PetscViewerASCIIPrintf(viewer,"  Use deluxe scaling: %d\n",pcbddc->use_deluxe_scaling));
    CHKERRQ(PetscViewerASCIIPrintf(viewer,"  Use deluxe zerorows: %d\n",pcbddc->deluxe_zerorows));
    CHKERRQ(PetscViewerASCIIPrintf(viewer,"  Use deluxe singlemat: %d\n",pcbddc->deluxe_singlemat));
    CHKERRQ(PetscViewerASCIIPrintf(viewer,"  Rebuild interface graph for Schur principal minors: %d\n",pcbddc->sub_schurs_rebuild));
    CHKERRQ(PetscViewerASCIIPrintf(viewer,"  Number of dofs' layers for the computation of principal minors: %D\n",pcbddc->sub_schurs_layers));
    CHKERRQ(PetscViewerASCIIPrintf(viewer,"  Use user CSR graph to compute successive layers: %d\n",pcbddc->sub_schurs_use_useradj));
    if (pcbddc->adaptive_threshold[1] != pcbddc->adaptive_threshold[0]) {
      CHKERRQ(PetscViewerASCIIPrintf(viewer,"  Adaptive constraint selection thresholds (active %d, userdefined %d): %g,%g\n",pcbddc->adaptive_selection,pcbddc->adaptive_userdefined,pcbddc->adaptive_threshold[0],pcbddc->adaptive_threshold[1]));
    } else {
      CHKERRQ(PetscViewerASCIIPrintf(viewer,"  Adaptive constraint selection threshold (active %d, userdefined %d): %g\n",pcbddc->adaptive_selection,pcbddc->adaptive_userdefined,pcbddc->adaptive_threshold[0]));
    }
    CHKERRQ(PetscViewerASCIIPrintf(viewer,"  Min constraints / connected component: %D\n",pcbddc->adaptive_nmin));
    CHKERRQ(PetscViewerASCIIPrintf(viewer,"  Max constraints / connected component: %D\n",pcbddc->adaptive_nmax));
    CHKERRQ(PetscViewerASCIIPrintf(viewer,"  Invert exact Schur complement for adaptive selection: %d\n",pcbddc->sub_schurs_exact_schur));
    CHKERRQ(PetscViewerASCIIPrintf(viewer,"  Symmetric computation of primal basis functions: %d\n",pcbddc->symmetric_primal));
    CHKERRQ(PetscViewerASCIIPrintf(viewer,"  Num. Procs. to map coarse adjacency list: %D\n",pcbddc->coarse_adj_red));
    CHKERRQ(PetscViewerASCIIPrintf(viewer,"  Coarse eqs per proc (significant at the coarsest level): %D\n",pcbddc->coarse_eqs_per_proc));
    CHKERRQ(PetscViewerASCIIPrintf(viewer,"  Detect disconnected: %d (filter %d)\n",pcbddc->detect_disconnected,pcbddc->detect_disconnected_filter));
    CHKERRQ(PetscViewerASCIIPrintf(viewer,"  Benign subspace trick: %d (change explicit %d)\n",pcbddc->benign_saddle_point,pcbddc->benign_change_explicit));
    CHKERRQ(PetscViewerASCIIPrintf(viewer,"  Benign subspace trick is active: %d\n",pcbddc->benign_have_null));
    CHKERRQ(PetscViewerASCIIPrintf(viewer,"  Algebraic computation of no-net-flux: %d\n",pcbddc->compute_nonetflux));
    if (!pc->setupcalled) PetscFunctionReturn(0);

    /* compute interface size */
    CHKERRQ(VecSet(pcis->vec1_B,1.0));
    CHKERRQ(MatCreateVecs(pc->pmat,&counter,NULL));
    CHKERRQ(VecSet(counter,0.0));
    CHKERRQ(VecScatterBegin(pcis->global_to_B,pcis->vec1_B,counter,INSERT_VALUES,SCATTER_REVERSE));
    CHKERRQ(VecScatterEnd(pcis->global_to_B,pcis->vec1_B,counter,INSERT_VALUES,SCATTER_REVERSE));
    CHKERRQ(VecSum(counter,&interface_size));
    CHKERRQ(VecDestroy(&counter));

    /* compute some statistics on the domain decomposition */
    gsum[0] = 1;
    gsum[1] = gsum[2] = gsum[3] = gsum[4] = gsum[5] = 0;
    loc[0]  = !!pcis->n;
    loc[1]  = pcis->n - pcis->n_B;
    loc[2]  = pcis->n_B;
    loc[3]  = pcbddc->local_primal_size;
    loc[4]  = pcis->n;
    loc[5]  = pcbddc->n_local_subs > 0 ? pcbddc->n_local_subs : (pcis->n ? 1 : 0);
    loc[6]  = pcbddc->benign_n;
    CHKERRMPI(MPI_Reduce(loc,gsum,6,MPIU_INT64,MPI_SUM,0,PetscObjectComm((PetscObject)pc)));
    if (!loc[0]) loc[1] = loc[2] = loc[3] = loc[4] = loc[5] = -1;
    CHKERRMPI(MPI_Reduce(loc,gmax,6,MPIU_INT64,MPI_MAX,0,PetscObjectComm((PetscObject)pc)));
    if (!loc[0]) loc[1] = loc[2] = loc[3] = loc[4] = loc[5] = PETSC_MAX_INT;
    CHKERRMPI(MPI_Reduce(loc,gmin,6,MPIU_INT64,MPI_MIN,0,PetscObjectComm((PetscObject)pc)));
    CHKERRMPI(MPI_Reduce(&loc[6],&totbenign,1,MPIU_INT64,MPI_SUM,0,PetscObjectComm((PetscObject)pc)));
    if (pcbddc->coarse_size) {
      ratio1 = pc->pmat->rmap->N/(1.*pcbddc->coarse_size);
      ratio2 = PetscRealPart(interface_size)/pcbddc->coarse_size;
    }
    CHKERRQ(PetscViewerASCIIPrintf(viewer,"********************************** STATISTICS AT LEVEL %d **********************************\n",pcbddc->current_level));
    CHKERRQ(PetscViewerASCIIPrintf(viewer,"  Global dofs sizes: all %D interface %D coarse %D\n",pc->pmat->rmap->N,(PetscInt)PetscRealPart(interface_size),pcbddc->coarse_size));
    CHKERRQ(PetscViewerASCIIPrintf(viewer,"  Coarsening ratios: all/coarse %D interface/coarse %D\n",(PetscInt)ratio1,(PetscInt)ratio2));
    CHKERRQ(PetscViewerASCIIPrintf(viewer,"  Active processes : %D\n",(PetscInt)gsum[0]));
    CHKERRQ(PetscViewerASCIIPrintf(viewer,"  Total subdomains : %D\n",(PetscInt)gsum[5]));
    if (pcbddc->benign_have_null) {
      CHKERRQ(PetscViewerASCIIPrintf(viewer,"  Benign subs      : %D\n",(PetscInt)totbenign));
    }
    CHKERRQ(PetscViewerASCIIPrintf(viewer,"  Dofs type        :\tMIN\tMAX\tMEAN\n"));
    CHKERRQ(PetscViewerASCIIPrintf(viewer,"  Interior  dofs   :\t%D\t%D\t%D\n",(PetscInt)gmin[1],(PetscInt)gmax[1],(PetscInt)(gsum[1]/gsum[0])));
    CHKERRQ(PetscViewerASCIIPrintf(viewer,"  Interface dofs   :\t%D\t%D\t%D\n",(PetscInt)gmin[2],(PetscInt)gmax[2],(PetscInt)(gsum[2]/gsum[0])));
    CHKERRQ(PetscViewerASCIIPrintf(viewer,"  Primal    dofs   :\t%D\t%D\t%D\n",(PetscInt)gmin[3],(PetscInt)gmax[3],(PetscInt)(gsum[3]/gsum[0])));
    CHKERRQ(PetscViewerASCIIPrintf(viewer,"  Local     dofs   :\t%D\t%D\t%D\n",(PetscInt)gmin[4],(PetscInt)gmax[4],(PetscInt)(gsum[4]/gsum[0])));
    CHKERRQ(PetscViewerASCIIPrintf(viewer,"  Local     subs   :\t%D\t%D\n"    ,(PetscInt)gmin[5],(PetscInt)gmax[5]));
    CHKERRQ(PetscViewerFlush(viewer));

    CHKERRMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)pc),&rank));

    /* local solvers */
    CHKERRQ(PetscViewerGetSubViewer(viewer,PetscObjectComm((PetscObject)pcbddc->ksp_D),&subviewer));
    if (rank == 0) {
      CHKERRQ(PetscViewerASCIIPrintf(subviewer,"--- Interior solver (rank 0)\n"));
      CHKERRQ(PetscViewerASCIIPushTab(subviewer));
      CHKERRQ(KSPView(pcbddc->ksp_D,subviewer));
      CHKERRQ(PetscViewerASCIIPopTab(subviewer));
      CHKERRQ(PetscViewerASCIIPrintf(subviewer,"--- Correction solver (rank 0)\n"));
      CHKERRQ(PetscViewerASCIIPushTab(subviewer));
      CHKERRQ(KSPView(pcbddc->ksp_R,subviewer));
      CHKERRQ(PetscViewerASCIIPopTab(subviewer));
      CHKERRQ(PetscViewerFlush(subviewer));
    }
    CHKERRQ(PetscViewerRestoreSubViewer(viewer,PetscObjectComm((PetscObject)pcbddc->ksp_D),&subviewer));
    CHKERRQ(PetscViewerFlush(viewer));

    /* the coarse problem can be handled by a different communicator */
    if (pcbddc->coarse_ksp) color = 1;
    else color = 0;
    CHKERRMPI(MPI_Comm_size(PetscObjectComm((PetscObject)pc),&size));
    CHKERRQ(PetscSubcommCreate(PetscObjectComm((PetscObject)pc),&subcomm));
    CHKERRQ(PetscSubcommSetNumber(subcomm,PetscMin(size,2)));
    CHKERRQ(PetscSubcommSetTypeGeneral(subcomm,color,rank));
    CHKERRQ(PetscViewerGetSubViewer(viewer,PetscSubcommChild(subcomm),&subviewer));
    if (color == 1) {
      CHKERRQ(PetscViewerASCIIPrintf(subviewer,"--- Coarse solver\n"));
      CHKERRQ(PetscViewerASCIIPushTab(subviewer));
      CHKERRQ(KSPView(pcbddc->coarse_ksp,subviewer));
      CHKERRQ(PetscViewerASCIIPopTab(subviewer));
      CHKERRQ(PetscViewerFlush(subviewer));
    }
    CHKERRQ(PetscViewerRestoreSubViewer(viewer,PetscSubcommChild(subcomm),&subviewer));
    CHKERRQ(PetscSubcommDestroy(&subcomm));
    CHKERRQ(PetscViewerFlush(viewer));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PCBDDCSetDiscreteGradient_BDDC(PC pc, Mat G, PetscInt order, PetscInt field, PetscBool global, PetscBool conforming)
{
  PC_BDDC        *pcbddc = (PC_BDDC*)pc->data;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectReference((PetscObject)G));
  CHKERRQ(MatDestroy(&pcbddc->discretegradient));
  pcbddc->discretegradient = G;
  pcbddc->nedorder         = order > 0 ? order : -order;
  pcbddc->nedfield         = field;
  pcbddc->nedglobal        = global;
  pcbddc->conforming       = conforming;
  PetscFunctionReturn(0);
}

/*@
 PCBDDCSetDiscreteGradient - Sets the discrete gradient

   Collective on PC

   Input Parameters:
+  pc         - the preconditioning context
.  G          - the discrete gradient matrix (should be in AIJ format)
.  order      - the order of the Nedelec space (1 for the lowest order)
.  field      - the field id of the Nedelec dofs (not used if the fields have not been specified)
.  global     - the type of global ordering for the rows of G
-  conforming - whether the mesh is conforming or not

   Level: advanced

   Notes:
    The discrete gradient matrix G is used to analyze the subdomain edges, and it should not contain any zero entry.
          For variable order spaces, the order should be set to zero.
          If global is true, the rows of G should be given in global ordering for the whole dofs;
          if false, the ordering should be global for the Nedelec field.
          In the latter case, it should hold gid[i] < gid[j] iff geid[i] < geid[j], with gid the global orderding for all the dofs
          and geid the one for the Nedelec field.

.seealso: PCBDDC,PCBDDCSetDofsSplitting(),PCBDDCSetDofsSplittingLocal()
@*/
PetscErrorCode PCBDDCSetDiscreteGradient(PC pc, Mat G, PetscInt order, PetscInt field, PetscBool global, PetscBool conforming)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscValidHeaderSpecific(G,MAT_CLASSID,2);
  PetscValidLogicalCollectiveInt(pc,order,3);
  PetscValidLogicalCollectiveInt(pc,field,4);
  PetscValidLogicalCollectiveBool(pc,global,5);
  PetscValidLogicalCollectiveBool(pc,conforming,6);
  PetscCheckSameComm(pc,1,G,2);
  CHKERRQ(PetscTryMethod(pc,"PCBDDCSetDiscreteGradient_C",(PC,Mat,PetscInt,PetscInt,PetscBool,PetscBool),(pc,G,order,field,global,conforming)));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCBDDCSetDivergenceMat_BDDC(PC pc, Mat divudotp, PetscBool trans, IS vl2l)
{
  PC_BDDC        *pcbddc = (PC_BDDC*)pc->data;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectReference((PetscObject)divudotp));
  CHKERRQ(MatDestroy(&pcbddc->divudotp));
  pcbddc->divudotp = divudotp;
  pcbddc->divudotp_trans = trans;
  pcbddc->compute_nonetflux = PETSC_TRUE;
  if (vl2l) {
    CHKERRQ(PetscObjectReference((PetscObject)vl2l));
    CHKERRQ(ISDestroy(&pcbddc->divudotp_vl2l));
    pcbddc->divudotp_vl2l = vl2l;
  }
  PetscFunctionReturn(0);
}

/*@
 PCBDDCSetDivergenceMat - Sets the linear operator representing \int_\Omega \div {\bf u} \cdot p dx

   Collective on PC

   Input Parameters:
+  pc - the preconditioning context
.  divudotp - the matrix (must be of type MATIS)
.  trans - if trans if false (resp. true), then pressures are in the test (trial) space and velocities are in the trial (test) space.
-  vl2l - optional index set describing the local (wrt the local matrix in divudotp) to local (wrt the local matrix in the preconditioning matrix) map for the velocities

   Level: advanced

   Notes:
    This auxiliary matrix is used to compute quadrature weights representing the net-flux across subdomain boundaries
          If vl2l is NULL, the local ordering for velocities in divudotp should match that of the preconditioning matrix

.seealso: PCBDDC
@*/
PetscErrorCode PCBDDCSetDivergenceMat(PC pc, Mat divudotp, PetscBool trans, IS vl2l)
{
  PetscBool      ismatis;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscValidHeaderSpecific(divudotp,MAT_CLASSID,2);
  PetscCheckSameComm(pc,1,divudotp,2);
  PetscValidLogicalCollectiveBool(pc,trans,3);
  if (vl2l) PetscValidHeaderSpecific(vl2l,IS_CLASSID,4);
  CHKERRQ(PetscObjectTypeCompare((PetscObject)divudotp,MATIS,&ismatis));
  PetscCheckFalse(!ismatis,PetscObjectComm((PetscObject)pc),PETSC_ERR_ARG_WRONG,"Divergence matrix needs to be of type MATIS");
  CHKERRQ(PetscTryMethod(pc,"PCBDDCSetDivergenceMat_C",(PC,Mat,PetscBool,IS),(pc,divudotp,trans,vl2l)));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCBDDCSetChangeOfBasisMat_BDDC(PC pc, Mat change, PetscBool interior)
{
  PC_BDDC        *pcbddc = (PC_BDDC*)pc->data;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectReference((PetscObject)change));
  CHKERRQ(MatDestroy(&pcbddc->user_ChangeOfBasisMatrix));
  pcbddc->user_ChangeOfBasisMatrix = change;
  pcbddc->change_interior = interior;
  PetscFunctionReturn(0);
}
/*@
 PCBDDCSetChangeOfBasisMat - Set user defined change of basis for dofs

   Collective on PC

   Input Parameters:
+  pc - the preconditioning context
.  change - the change of basis matrix
-  interior - whether or not the change of basis modifies interior dofs

   Level: intermediate

   Notes:

.seealso: PCBDDC
@*/
PetscErrorCode PCBDDCSetChangeOfBasisMat(PC pc, Mat change, PetscBool interior)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscValidHeaderSpecific(change,MAT_CLASSID,2);
  PetscCheckSameComm(pc,1,change,2);
  if (pc->mat) {
    PetscInt rows_c,cols_c,rows,cols;
    CHKERRQ(MatGetSize(pc->mat,&rows,&cols));
    CHKERRQ(MatGetSize(change,&rows_c,&cols_c));
    PetscCheckFalse(rows_c != rows,PetscObjectComm((PetscObject)pc),PETSC_ERR_SUP,"Invalid number of rows for change of basis matrix! %D != %D",rows_c,rows);
    PetscCheckFalse(cols_c != cols,PetscObjectComm((PetscObject)pc),PETSC_ERR_SUP,"Invalid number of columns for change of basis matrix! %D != %D",cols_c,cols);
    CHKERRQ(MatGetLocalSize(pc->mat,&rows,&cols));
    CHKERRQ(MatGetLocalSize(change,&rows_c,&cols_c));
    PetscCheckFalse(rows_c != rows,PetscObjectComm((PetscObject)pc),PETSC_ERR_SUP,"Invalid number of local rows for change of basis matrix! %D != %D",rows_c,rows);
    PetscCheckFalse(cols_c != cols,PetscObjectComm((PetscObject)pc),PETSC_ERR_SUP,"Invalid number of local columns for change of basis matrix! %D != %D",cols_c,cols);
  }
  CHKERRQ(PetscTryMethod(pc,"PCBDDCSetChangeOfBasisMat_C",(PC,Mat,PetscBool),(pc,change,interior)));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCBDDCSetPrimalVerticesIS_BDDC(PC pc, IS PrimalVertices)
{
  PC_BDDC        *pcbddc = (PC_BDDC*)pc->data;
  PetscBool      isequal = PETSC_FALSE;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectReference((PetscObject)PrimalVertices));
  if (pcbddc->user_primal_vertices) {
    CHKERRQ(ISEqual(PrimalVertices,pcbddc->user_primal_vertices,&isequal));
  }
  CHKERRQ(ISDestroy(&pcbddc->user_primal_vertices));
  CHKERRQ(ISDestroy(&pcbddc->user_primal_vertices_local));
  pcbddc->user_primal_vertices = PrimalVertices;
  if (!isequal) pcbddc->recompute_topography = PETSC_TRUE;
  PetscFunctionReturn(0);
}

/*@
 PCBDDCSetPrimalVerticesIS - Set additional user defined primal vertices in PCBDDC

   Collective

   Input Parameters:
+  pc - the preconditioning context
-  PrimalVertices - index set of primal vertices in global numbering (can be empty)

   Level: intermediate

   Notes:
     Any process can list any global node

.seealso: PCBDDC, PCBDDCGetPrimalVerticesIS(), PCBDDCSetPrimalVerticesLocalIS(), PCBDDCGetPrimalVerticesLocalIS()
@*/
PetscErrorCode PCBDDCSetPrimalVerticesIS(PC pc, IS PrimalVertices)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscValidHeaderSpecific(PrimalVertices,IS_CLASSID,2);
  PetscCheckSameComm(pc,1,PrimalVertices,2);
  CHKERRQ(PetscTryMethod(pc,"PCBDDCSetPrimalVerticesIS_C",(PC,IS),(pc,PrimalVertices)));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCBDDCGetPrimalVerticesIS_BDDC(PC pc, IS *is)
{
  PC_BDDC *pcbddc = (PC_BDDC*)pc->data;

  PetscFunctionBegin;
  *is = pcbddc->user_primal_vertices;
  PetscFunctionReturn(0);
}

/*@
 PCBDDCGetPrimalVerticesIS - Get user defined primal vertices set with PCBDDCSetPrimalVerticesIS()

   Collective

   Input Parameters:
.  pc - the preconditioning context

   Output Parameters:
.  is - index set of primal vertices in global numbering (NULL if not set)

   Level: intermediate

   Notes:

.seealso: PCBDDC, PCBDDCSetPrimalVerticesIS(), PCBDDCSetPrimalVerticesLocalIS(), PCBDDCGetPrimalVerticesLocalIS()
@*/
PetscErrorCode PCBDDCGetPrimalVerticesIS(PC pc, IS *is)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscValidPointer(is,2);
  CHKERRQ(PetscUseMethod(pc,"PCBDDCGetPrimalVerticesIS_C",(PC,IS*),(pc,is)));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCBDDCSetPrimalVerticesLocalIS_BDDC(PC pc, IS PrimalVertices)
{
  PC_BDDC        *pcbddc = (PC_BDDC*)pc->data;
  PetscBool      isequal = PETSC_FALSE;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectReference((PetscObject)PrimalVertices));
  if (pcbddc->user_primal_vertices_local) {
    CHKERRQ(ISEqual(PrimalVertices,pcbddc->user_primal_vertices_local,&isequal));
  }
  CHKERRQ(ISDestroy(&pcbddc->user_primal_vertices));
  CHKERRQ(ISDestroy(&pcbddc->user_primal_vertices_local));
  pcbddc->user_primal_vertices_local = PrimalVertices;
  if (!isequal) pcbddc->recompute_topography = PETSC_TRUE;
  PetscFunctionReturn(0);
}

/*@
 PCBDDCSetPrimalVerticesLocalIS - Set additional user defined primal vertices in PCBDDC

   Collective

   Input Parameters:
+  pc - the preconditioning context
-  PrimalVertices - index set of primal vertices in local numbering (can be empty)

   Level: intermediate

   Notes:

.seealso: PCBDDC, PCBDDCSetPrimalVerticesIS(), PCBDDCGetPrimalVerticesIS(), PCBDDCGetPrimalVerticesLocalIS()
@*/
PetscErrorCode PCBDDCSetPrimalVerticesLocalIS(PC pc, IS PrimalVertices)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscValidHeaderSpecific(PrimalVertices,IS_CLASSID,2);
  PetscCheckSameComm(pc,1,PrimalVertices,2);
  CHKERRQ(PetscTryMethod(pc,"PCBDDCSetPrimalVerticesLocalIS_C",(PC,IS),(pc,PrimalVertices)));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCBDDCGetPrimalVerticesLocalIS_BDDC(PC pc, IS *is)
{
  PC_BDDC *pcbddc = (PC_BDDC*)pc->data;

  PetscFunctionBegin;
  *is = pcbddc->user_primal_vertices_local;
  PetscFunctionReturn(0);
}

/*@
 PCBDDCGetPrimalVerticesLocalIS - Get user defined primal vertices set with PCBDDCSetPrimalVerticesLocalIS()

   Collective

   Input Parameters:
.  pc - the preconditioning context

   Output Parameters:
.  is - index set of primal vertices in local numbering (NULL if not set)

   Level: intermediate

   Notes:

.seealso: PCBDDC, PCBDDCSetPrimalVerticesIS(), PCBDDCGetPrimalVerticesIS(), PCBDDCSetPrimalVerticesLocalIS()
@*/
PetscErrorCode PCBDDCGetPrimalVerticesLocalIS(PC pc, IS *is)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscValidPointer(is,2);
  CHKERRQ(PetscUseMethod(pc,"PCBDDCGetPrimalVerticesLocalIS_C",(PC,IS*),(pc,is)));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCBDDCSetCoarseningRatio_BDDC(PC pc,PetscInt k)
{
  PC_BDDC  *pcbddc = (PC_BDDC*)pc->data;

  PetscFunctionBegin;
  pcbddc->coarsening_ratio = k;
  PetscFunctionReturn(0);
}

/*@
 PCBDDCSetCoarseningRatio - Set coarsening ratio used in multilevel

   Logically collective on PC

   Input Parameters:
+  pc - the preconditioning context
-  k - coarsening ratio (H/h at the coarser level)

   Options Database Keys:
.    -pc_bddc_coarsening_ratio <int> - Set coarsening ratio used in multilevel coarsening

   Level: intermediate

   Notes:
     Approximatively k subdomains at the finer level will be aggregated into a single subdomain at the coarser level

.seealso: PCBDDC, PCBDDCSetLevels()
@*/
PetscErrorCode PCBDDCSetCoarseningRatio(PC pc,PetscInt k)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscValidLogicalCollectiveInt(pc,k,2);
  CHKERRQ(PetscTryMethod(pc,"PCBDDCSetCoarseningRatio_C",(PC,PetscInt),(pc,k)));
  PetscFunctionReturn(0);
}

/* The following functions (PCBDDCSetUseExactDirichlet PCBDDCSetLevel) are not public */
static PetscErrorCode PCBDDCSetUseExactDirichlet_BDDC(PC pc,PetscBool flg)
{
  PC_BDDC  *pcbddc = (PC_BDDC*)pc->data;

  PetscFunctionBegin;
  pcbddc->use_exact_dirichlet_trick = flg;
  PetscFunctionReturn(0);
}

PetscErrorCode PCBDDCSetUseExactDirichlet(PC pc,PetscBool flg)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscValidLogicalCollectiveBool(pc,flg,2);
  CHKERRQ(PetscTryMethod(pc,"PCBDDCSetUseExactDirichlet_C",(PC,PetscBool),(pc,flg)));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCBDDCSetLevel_BDDC(PC pc,PetscInt level)
{
  PC_BDDC  *pcbddc = (PC_BDDC*)pc->data;

  PetscFunctionBegin;
  pcbddc->current_level = level;
  PetscFunctionReturn(0);
}

PetscErrorCode PCBDDCSetLevel(PC pc,PetscInt level)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscValidLogicalCollectiveInt(pc,level,2);
  CHKERRQ(PetscTryMethod(pc,"PCBDDCSetLevel_C",(PC,PetscInt),(pc,level)));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCBDDCSetLevels_BDDC(PC pc,PetscInt levels)
{
  PC_BDDC  *pcbddc = (PC_BDDC*)pc->data;

  PetscFunctionBegin;
  PetscCheckFalse(levels > PETSC_PCBDDC_MAXLEVELS-1,PetscObjectComm((PetscObject)pc),PETSC_ERR_SUP,"Maximum number of additional levels for BDDC is %d",PETSC_PCBDDC_MAXLEVELS-1);
  pcbddc->max_levels = levels;
  PetscFunctionReturn(0);
}

/*@
 PCBDDCSetLevels - Sets the maximum number of additional levels allowed for multilevel BDDC

   Logically collective on PC

   Input Parameters:
+  pc - the preconditioning context
-  levels - the maximum number of levels

   Options Database Keys:
.    -pc_bddc_levels <int> - Set maximum number of levels for multilevel

   Level: intermediate

   Notes:
     The default value is 0, that gives the classical two-levels BDDC

.seealso: PCBDDC, PCBDDCSetCoarseningRatio()
@*/
PetscErrorCode PCBDDCSetLevels(PC pc,PetscInt levels)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscValidLogicalCollectiveInt(pc,levels,2);
  CHKERRQ(PetscTryMethod(pc,"PCBDDCSetLevels_C",(PC,PetscInt),(pc,levels)));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCBDDCSetDirichletBoundaries_BDDC(PC pc,IS DirichletBoundaries)
{
  PC_BDDC        *pcbddc = (PC_BDDC*)pc->data;
  PetscBool      isequal = PETSC_FALSE;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectReference((PetscObject)DirichletBoundaries));
  if (pcbddc->DirichletBoundaries) {
    CHKERRQ(ISEqual(DirichletBoundaries,pcbddc->DirichletBoundaries,&isequal));
  }
  /* last user setting takes precedence -> destroy any other customization */
  CHKERRQ(ISDestroy(&pcbddc->DirichletBoundariesLocal));
  CHKERRQ(ISDestroy(&pcbddc->DirichletBoundaries));
  pcbddc->DirichletBoundaries = DirichletBoundaries;
  if (!isequal) pcbddc->recompute_topography = PETSC_TRUE;
  PetscFunctionReturn(0);
}

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
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscValidHeaderSpecific(DirichletBoundaries,IS_CLASSID,2);
  PetscCheckSameComm(pc,1,DirichletBoundaries,2);
  CHKERRQ(PetscTryMethod(pc,"PCBDDCSetDirichletBoundaries_C",(PC,IS),(pc,DirichletBoundaries)));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCBDDCSetDirichletBoundariesLocal_BDDC(PC pc,IS DirichletBoundaries)
{
  PC_BDDC        *pcbddc = (PC_BDDC*)pc->data;
  PetscBool      isequal = PETSC_FALSE;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectReference((PetscObject)DirichletBoundaries));
  if (pcbddc->DirichletBoundariesLocal) {
    CHKERRQ(ISEqual(DirichletBoundaries,pcbddc->DirichletBoundariesLocal,&isequal));
  }
  /* last user setting takes precedence -> destroy any other customization */
  CHKERRQ(ISDestroy(&pcbddc->DirichletBoundariesLocal));
  CHKERRQ(ISDestroy(&pcbddc->DirichletBoundaries));
  pcbddc->DirichletBoundariesLocal = DirichletBoundaries;
  if (!isequal) pcbddc->recompute_topography = PETSC_TRUE;
  PetscFunctionReturn(0);
}

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
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscValidHeaderSpecific(DirichletBoundaries,IS_CLASSID,2);
  PetscCheckSameComm(pc,1,DirichletBoundaries,2);
  CHKERRQ(PetscTryMethod(pc,"PCBDDCSetDirichletBoundariesLocal_C",(PC,IS),(pc,DirichletBoundaries)));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCBDDCSetNeumannBoundaries_BDDC(PC pc,IS NeumannBoundaries)
{
  PC_BDDC        *pcbddc = (PC_BDDC*)pc->data;
  PetscBool      isequal = PETSC_FALSE;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectReference((PetscObject)NeumannBoundaries));
  if (pcbddc->NeumannBoundaries) {
    CHKERRQ(ISEqual(NeumannBoundaries,pcbddc->NeumannBoundaries,&isequal));
  }
  /* last user setting takes precedence -> destroy any other customization */
  CHKERRQ(ISDestroy(&pcbddc->NeumannBoundariesLocal));
  CHKERRQ(ISDestroy(&pcbddc->NeumannBoundaries));
  pcbddc->NeumannBoundaries = NeumannBoundaries;
  if (!isequal) pcbddc->recompute_topography = PETSC_TRUE;
  PetscFunctionReturn(0);
}

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
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscValidHeaderSpecific(NeumannBoundaries,IS_CLASSID,2);
  PetscCheckSameComm(pc,1,NeumannBoundaries,2);
  CHKERRQ(PetscTryMethod(pc,"PCBDDCSetNeumannBoundaries_C",(PC,IS),(pc,NeumannBoundaries)));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCBDDCSetNeumannBoundariesLocal_BDDC(PC pc,IS NeumannBoundaries)
{
  PC_BDDC        *pcbddc = (PC_BDDC*)pc->data;
  PetscBool      isequal = PETSC_FALSE;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectReference((PetscObject)NeumannBoundaries));
  if (pcbddc->NeumannBoundariesLocal) {
    CHKERRQ(ISEqual(NeumannBoundaries,pcbddc->NeumannBoundariesLocal,&isequal));
  }
  /* last user setting takes precedence -> destroy any other customization */
  CHKERRQ(ISDestroy(&pcbddc->NeumannBoundariesLocal));
  CHKERRQ(ISDestroy(&pcbddc->NeumannBoundaries));
  pcbddc->NeumannBoundariesLocal = NeumannBoundaries;
  if (!isequal) pcbddc->recompute_topography = PETSC_TRUE;
  PetscFunctionReturn(0);
}

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
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscValidHeaderSpecific(NeumannBoundaries,IS_CLASSID,2);
  PetscCheckSameComm(pc,1,NeumannBoundaries,2);
  CHKERRQ(PetscTryMethod(pc,"PCBDDCSetNeumannBoundariesLocal_C",(PC,IS),(pc,NeumannBoundaries)));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCBDDCGetDirichletBoundaries_BDDC(PC pc,IS *DirichletBoundaries)
{
  PC_BDDC  *pcbddc = (PC_BDDC*)pc->data;

  PetscFunctionBegin;
  *DirichletBoundaries = pcbddc->DirichletBoundaries;
  PetscFunctionReturn(0);
}

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
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  CHKERRQ(PetscUseMethod(pc,"PCBDDCGetDirichletBoundaries_C",(PC,IS*),(pc,DirichletBoundaries)));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCBDDCGetDirichletBoundariesLocal_BDDC(PC pc,IS *DirichletBoundaries)
{
  PC_BDDC  *pcbddc = (PC_BDDC*)pc->data;

  PetscFunctionBegin;
  *DirichletBoundaries = pcbddc->DirichletBoundariesLocal;
  PetscFunctionReturn(0);
}

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
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  CHKERRQ(PetscUseMethod(pc,"PCBDDCGetDirichletBoundariesLocal_C",(PC,IS*),(pc,DirichletBoundaries)));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCBDDCGetNeumannBoundaries_BDDC(PC pc,IS *NeumannBoundaries)
{
  PC_BDDC  *pcbddc = (PC_BDDC*)pc->data;

  PetscFunctionBegin;
  *NeumannBoundaries = pcbddc->NeumannBoundaries;
  PetscFunctionReturn(0);
}

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
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  CHKERRQ(PetscUseMethod(pc,"PCBDDCGetNeumannBoundaries_C",(PC,IS*),(pc,NeumannBoundaries)));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCBDDCGetNeumannBoundariesLocal_BDDC(PC pc,IS *NeumannBoundaries)
{
  PC_BDDC  *pcbddc = (PC_BDDC*)pc->data;

  PetscFunctionBegin;
  *NeumannBoundaries = pcbddc->NeumannBoundariesLocal;
  PetscFunctionReturn(0);
}

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
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  CHKERRQ(PetscUseMethod(pc,"PCBDDCGetNeumannBoundariesLocal_C",(PC,IS*),(pc,NeumannBoundaries)));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCBDDCSetLocalAdjacencyGraph_BDDC(PC pc, PetscInt nvtxs,const PetscInt xadj[],const PetscInt adjncy[], PetscCopyMode copymode)
{
  PC_BDDC        *pcbddc = (PC_BDDC*)pc->data;
  PCBDDCGraph    mat_graph = pcbddc->mat_graph;
  PetscBool      same_data = PETSC_FALSE;

  PetscFunctionBegin;
  if (!nvtxs) {
    if (copymode == PETSC_OWN_POINTER) {
      CHKERRQ(PetscFree(xadj));
      CHKERRQ(PetscFree(adjncy));
    }
    CHKERRQ(PCBDDCGraphResetCSR(mat_graph));
    PetscFunctionReturn(0);
  }
  if (mat_graph->nvtxs == nvtxs && mat_graph->freecsr) { /* we own the data */
    if (mat_graph->xadj == xadj && mat_graph->adjncy == adjncy) same_data = PETSC_TRUE;
    if (!same_data && mat_graph->xadj[nvtxs] == xadj[nvtxs]) {
      CHKERRQ(PetscArraycmp(xadj,mat_graph->xadj,nvtxs+1,&same_data));
      if (same_data) {
        CHKERRQ(PetscArraycmp(adjncy,mat_graph->adjncy,xadj[nvtxs],&same_data));
      }
    }
  }
  if (!same_data) {
    /* free old CSR */
    CHKERRQ(PCBDDCGraphResetCSR(mat_graph));
    /* get CSR into graph structure */
    if (copymode == PETSC_COPY_VALUES) {
      CHKERRQ(PetscMalloc1(nvtxs+1,&mat_graph->xadj));
      CHKERRQ(PetscMalloc1(xadj[nvtxs],&mat_graph->adjncy));
      CHKERRQ(PetscArraycpy(mat_graph->xadj,xadj,nvtxs+1));
      CHKERRQ(PetscArraycpy(mat_graph->adjncy,adjncy,xadj[nvtxs]));
      mat_graph->freecsr = PETSC_TRUE;
    } else if (copymode == PETSC_OWN_POINTER) {
      mat_graph->xadj    = (PetscInt*)xadj;
      mat_graph->adjncy  = (PetscInt*)adjncy;
      mat_graph->freecsr = PETSC_TRUE;
    } else if (copymode == PETSC_USE_POINTER) {
      mat_graph->xadj    = (PetscInt*)xadj;
      mat_graph->adjncy  = (PetscInt*)adjncy;
      mat_graph->freecsr = PETSC_FALSE;
    } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Unsupported copy mode %D",copymode);
    mat_graph->nvtxs_csr = nvtxs;
    pcbddc->recompute_topography = PETSC_TRUE;
  }
  PetscFunctionReturn(0);
}

/*@
 PCBDDCSetLocalAdjacencyGraph - Set adjacency structure (CSR graph) of the local degrees of freedom.

   Not collective

   Input Parameters:
+  pc - the preconditioning context.
.  nvtxs - number of local vertices of the graph (i.e., the number of local dofs).
.  xadj, adjncy - the connectivity of the dofs in CSR format.
-  copymode - supported modes are PETSC_COPY_VALUES, PETSC_USE_POINTER or PETSC_OWN_POINTER.

   Level: intermediate

   Notes:
    A dof is considered connected with all local dofs if xadj[dof+1]-xadj[dof] == 1 and adjncy[xadj[dof]] is negative.

.seealso: PCBDDC,PetscCopyMode
@*/
PetscErrorCode PCBDDCSetLocalAdjacencyGraph(PC pc,PetscInt nvtxs,const PetscInt xadj[],const PetscInt adjncy[], PetscCopyMode copymode)
{
  void (*f)(void) = NULL;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  if (nvtxs) {
    PetscValidIntPointer(xadj,3);
    if (xadj[nvtxs]) PetscValidIntPointer(adjncy,4);
  }
  CHKERRQ(PetscTryMethod(pc,"PCBDDCSetLocalAdjacencyGraph_C",(PC,PetscInt,const PetscInt[],const PetscInt[],PetscCopyMode),(pc,nvtxs,xadj,adjncy,copymode)));
  /* free arrays if PCBDDC is not the PC type */
  CHKERRQ(PetscObjectQueryFunction((PetscObject)pc,"PCBDDCSetLocalAdjacencyGraph_C",&f));
  if (!f && copymode == PETSC_OWN_POINTER) {
    CHKERRQ(PetscFree(xadj));
    CHKERRQ(PetscFree(adjncy));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PCBDDCSetDofsSplittingLocal_BDDC(PC pc,PetscInt n_is, IS ISForDofs[])
{
  PC_BDDC        *pcbddc = (PC_BDDC*)pc->data;
  PetscInt       i;
  PetscBool      isequal = PETSC_FALSE;

  PetscFunctionBegin;
  if (pcbddc->n_ISForDofsLocal == n_is) {
    for (i=0;i<n_is;i++) {
      PetscBool isequalt;
      CHKERRQ(ISEqual(ISForDofs[i],pcbddc->ISForDofsLocal[i],&isequalt));
      if (!isequalt) break;
    }
    if (i == n_is) isequal = PETSC_TRUE;
  }
  for (i=0;i<n_is;i++) {
    CHKERRQ(PetscObjectReference((PetscObject)ISForDofs[i]));
  }
  /* Destroy ISes if they were already set */
  for (i=0;i<pcbddc->n_ISForDofsLocal;i++) {
    CHKERRQ(ISDestroy(&pcbddc->ISForDofsLocal[i]));
  }
  CHKERRQ(PetscFree(pcbddc->ISForDofsLocal));
  /* last user setting takes precedence -> destroy any other customization */
  for (i=0;i<pcbddc->n_ISForDofs;i++) {
    CHKERRQ(ISDestroy(&pcbddc->ISForDofs[i]));
  }
  CHKERRQ(PetscFree(pcbddc->ISForDofs));
  pcbddc->n_ISForDofs = 0;
  /* allocate space then set */
  if (n_is) {
    CHKERRQ(PetscMalloc1(n_is,&pcbddc->ISForDofsLocal));
  }
  for (i=0;i<n_is;i++) {
    pcbddc->ISForDofsLocal[i] = ISForDofs[i];
  }
  pcbddc->n_ISForDofsLocal = n_is;
  if (n_is) pcbddc->user_provided_isfordofs = PETSC_TRUE;
  if (!isequal) pcbddc->recompute_topography = PETSC_TRUE;
  PetscFunctionReturn(0);
}

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

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscValidLogicalCollectiveInt(pc,n_is,2);
  for (i=0;i<n_is;i++) {
    PetscCheckSameComm(pc,1,ISForDofs[i],3);
    PetscValidHeaderSpecific(ISForDofs[i],IS_CLASSID,3);
  }
  CHKERRQ(PetscTryMethod(pc,"PCBDDCSetDofsSplittingLocal_C",(PC,PetscInt,IS[]),(pc,n_is,ISForDofs)));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCBDDCSetDofsSplitting_BDDC(PC pc,PetscInt n_is, IS ISForDofs[])
{
  PC_BDDC        *pcbddc = (PC_BDDC*)pc->data;
  PetscInt       i;
  PetscBool      isequal = PETSC_FALSE;

  PetscFunctionBegin;
  if (pcbddc->n_ISForDofs == n_is) {
    for (i=0;i<n_is;i++) {
      PetscBool isequalt;
      CHKERRQ(ISEqual(ISForDofs[i],pcbddc->ISForDofs[i],&isequalt));
      if (!isequalt) break;
    }
    if (i == n_is) isequal = PETSC_TRUE;
  }
  for (i=0;i<n_is;i++) {
    CHKERRQ(PetscObjectReference((PetscObject)ISForDofs[i]));
  }
  /* Destroy ISes if they were already set */
  for (i=0;i<pcbddc->n_ISForDofs;i++) {
    CHKERRQ(ISDestroy(&pcbddc->ISForDofs[i]));
  }
  CHKERRQ(PetscFree(pcbddc->ISForDofs));
  /* last user setting takes precedence -> destroy any other customization */
  for (i=0;i<pcbddc->n_ISForDofsLocal;i++) {
    CHKERRQ(ISDestroy(&pcbddc->ISForDofsLocal[i]));
  }
  CHKERRQ(PetscFree(pcbddc->ISForDofsLocal));
  pcbddc->n_ISForDofsLocal = 0;
  /* allocate space then set */
  if (n_is) {
    CHKERRQ(PetscMalloc1(n_is,&pcbddc->ISForDofs));
  }
  for (i=0;i<n_is;i++) {
    pcbddc->ISForDofs[i] = ISForDofs[i];
  }
  pcbddc->n_ISForDofs = n_is;
  if (n_is) pcbddc->user_provided_isfordofs = PETSC_TRUE;
  if (!isequal) pcbddc->recompute_topography = PETSC_TRUE;
  PetscFunctionReturn(0);
}

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

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscValidLogicalCollectiveInt(pc,n_is,2);
  for (i=0;i<n_is;i++) {
    PetscValidHeaderSpecific(ISForDofs[i],IS_CLASSID,3);
    PetscCheckSameComm(pc,1,ISForDofs[i],3);
  }
  CHKERRQ(PetscTryMethod(pc,"PCBDDCSetDofsSplitting_C",(PC,PetscInt,IS[]),(pc,n_is,ISForDofs)));
  PetscFunctionReturn(0);
}

/*
   PCPreSolve_BDDC - Changes the right hand side and (if necessary) the initial
                     guess if a transformation of basis approach has been selected.

   Input Parameter:
+  pc - the preconditioner context

   Application Interface Routine: PCPreSolve()

   Notes:
     The interface routine PCPreSolve() is not usually called directly by
   the user, but instead is called by KSPSolve().
*/
static PetscErrorCode PCPreSolve_BDDC(PC pc, KSP ksp, Vec rhs, Vec x)
{
  PC_BDDC        *pcbddc = (PC_BDDC*)pc->data;
  PC_IS          *pcis = (PC_IS*)(pc->data);
  Vec            used_vec;
  PetscBool      iscg = PETSC_FALSE, save_rhs = PETSC_TRUE, benign_correction_computed;

  PetscFunctionBegin;
  /* if we are working with CG, one dirichlet solve can be avoided during Krylov iterations */
  if (ksp) {
    PetscBool isgroppcg, ispipecg, ispipelcg, ispipecgrr;

    CHKERRQ(PetscObjectTypeCompare((PetscObject)ksp,KSPCG,&iscg));
    CHKERRQ(PetscObjectTypeCompare((PetscObject)ksp,KSPGROPPCG,&isgroppcg));
    CHKERRQ(PetscObjectTypeCompare((PetscObject)ksp,KSPPIPECG,&ispipecg));
    CHKERRQ(PetscObjectTypeCompare((PetscObject)ksp,KSPPIPECG,&ispipelcg));
    CHKERRQ(PetscObjectTypeCompare((PetscObject)ksp,KSPPIPECGRR,&ispipecgrr));
    iscg = (PetscBool)(iscg || isgroppcg || ispipecg || ispipelcg || ispipecgrr);
    if (pcbddc->benign_apply_coarse_only || pcbddc->switch_static || !iscg || pc->mat != pc->pmat) {
      CHKERRQ(PCBDDCSetUseExactDirichlet(pc,PETSC_FALSE));
    }
  }
  if (pcbddc->benign_apply_coarse_only || pcbddc->switch_static || pc->mat != pc->pmat) {
    CHKERRQ(PCBDDCSetUseExactDirichlet(pc,PETSC_FALSE));
  }

  /* Creates parallel work vectors used in presolve */
  if (!pcbddc->original_rhs) {
    CHKERRQ(VecDuplicate(pcis->vec1_global,&pcbddc->original_rhs));
  }
  if (!pcbddc->temp_solution) {
    CHKERRQ(VecDuplicate(pcis->vec1_global,&pcbddc->temp_solution));
  }

  pcbddc->temp_solution_used = PETSC_FALSE;
  if (x) {
    CHKERRQ(PetscObjectReference((PetscObject)x));
    used_vec = x;
  } else { /* it can only happen when calling PCBDDCMatFETIDPGetRHS */
    CHKERRQ(PetscObjectReference((PetscObject)pcbddc->temp_solution));
    used_vec = pcbddc->temp_solution;
    CHKERRQ(VecSet(used_vec,0.0));
    pcbddc->temp_solution_used = PETSC_TRUE;
    CHKERRQ(VecCopy(rhs,pcbddc->original_rhs));
    save_rhs = PETSC_FALSE;
    pcbddc->eliminate_dirdofs = PETSC_TRUE;
  }

  /* hack into ksp data structure since PCPreSolve comes earlier than setting to zero the guess in src/ksp/ksp/interface/itfunc.c */
  if (ksp) {
    /* store the flag for the initial guess since it will be restored back during PCPostSolve_BDDC */
    CHKERRQ(KSPGetInitialGuessNonzero(ksp,&pcbddc->ksp_guess_nonzero));
    if (!pcbddc->ksp_guess_nonzero) {
      CHKERRQ(VecSet(used_vec,0.0));
    }
  }

  pcbddc->rhs_change = PETSC_FALSE;
  /* Take into account zeroed rows -> change rhs and store solution removed */
  if (rhs && pcbddc->eliminate_dirdofs) {
    IS dirIS = NULL;

    /* DirichletBoundariesLocal may not be consistent among neighbours; gets a dirichlet dofs IS from graph (may be cached) */
    CHKERRQ(PCBDDCGraphGetDirichletDofs(pcbddc->mat_graph,&dirIS));
    if (dirIS) {
      Mat_IS            *matis = (Mat_IS*)pc->pmat->data;
      PetscInt          dirsize,i,*is_indices;
      PetscScalar       *array_x;
      const PetscScalar *array_diagonal;

      CHKERRQ(MatGetDiagonal(pc->pmat,pcis->vec1_global));
      CHKERRQ(VecPointwiseDivide(pcis->vec1_global,rhs,pcis->vec1_global));
      CHKERRQ(VecScatterBegin(matis->rctx,pcis->vec1_global,pcis->vec2_N,INSERT_VALUES,SCATTER_FORWARD));
      CHKERRQ(VecScatterEnd(matis->rctx,pcis->vec1_global,pcis->vec2_N,INSERT_VALUES,SCATTER_FORWARD));
      CHKERRQ(VecScatterBegin(matis->rctx,used_vec,pcis->vec1_N,INSERT_VALUES,SCATTER_FORWARD));
      CHKERRQ(VecScatterEnd(matis->rctx,used_vec,pcis->vec1_N,INSERT_VALUES,SCATTER_FORWARD));
      CHKERRQ(ISGetLocalSize(dirIS,&dirsize));
      CHKERRQ(VecGetArray(pcis->vec1_N,&array_x));
      CHKERRQ(VecGetArrayRead(pcis->vec2_N,&array_diagonal));
      CHKERRQ(ISGetIndices(dirIS,(const PetscInt**)&is_indices));
      for (i=0; i<dirsize; i++) array_x[is_indices[i]] = array_diagonal[is_indices[i]];
      CHKERRQ(ISRestoreIndices(dirIS,(const PetscInt**)&is_indices));
      CHKERRQ(VecRestoreArrayRead(pcis->vec2_N,&array_diagonal));
      CHKERRQ(VecRestoreArray(pcis->vec1_N,&array_x));
      CHKERRQ(VecScatterBegin(matis->rctx,pcis->vec1_N,used_vec,INSERT_VALUES,SCATTER_REVERSE));
      CHKERRQ(VecScatterEnd(matis->rctx,pcis->vec1_N,used_vec,INSERT_VALUES,SCATTER_REVERSE));
      pcbddc->rhs_change = PETSC_TRUE;
      CHKERRQ(ISDestroy(&dirIS));
    }
  }

  /* remove the computed solution or the initial guess from the rhs */
  if (pcbddc->rhs_change || (ksp && pcbddc->ksp_guess_nonzero)) {
    /* save the original rhs */
    if (save_rhs) {
      CHKERRQ(VecSwap(rhs,pcbddc->original_rhs));
      save_rhs = PETSC_FALSE;
    }
    pcbddc->rhs_change = PETSC_TRUE;
    CHKERRQ(VecScale(used_vec,-1.0));
    CHKERRQ(MatMultAdd(pc->mat,used_vec,pcbddc->original_rhs,rhs));
    CHKERRQ(VecScale(used_vec,-1.0));
    CHKERRQ(VecCopy(used_vec,pcbddc->temp_solution));
    pcbddc->temp_solution_used = PETSC_TRUE;
    if (ksp) {
      CHKERRQ(KSPSetInitialGuessNonzero(ksp,PETSC_FALSE));
    }
  }
  CHKERRQ(VecDestroy(&used_vec));

  /* compute initial vector in benign space if needed
     and remove non-benign solution from the rhs */
  benign_correction_computed = PETSC_FALSE;
  if (rhs && pcbddc->benign_compute_correction && (pcbddc->benign_have_null || pcbddc->benign_apply_coarse_only)) {
    /* compute u^*_h using ideas similar to those in Xuemin Tu's PhD thesis (see Section 4.8.1)
       Recursively apply BDDC in the multilevel case */
    if (!pcbddc->benign_vec) {
      CHKERRQ(VecDuplicate(rhs,&pcbddc->benign_vec));
    }
    /* keep applying coarse solver unless we no longer have benign subdomains */
    pcbddc->benign_apply_coarse_only = pcbddc->benign_have_null ? PETSC_TRUE : PETSC_FALSE;
    if (!pcbddc->benign_skip_correction) {
      CHKERRQ(PCApply_BDDC(pc,rhs,pcbddc->benign_vec));
      benign_correction_computed = PETSC_TRUE;
      if (pcbddc->temp_solution_used) {
        CHKERRQ(VecAXPY(pcbddc->temp_solution,1.0,pcbddc->benign_vec));
      }
      CHKERRQ(VecScale(pcbddc->benign_vec,-1.0));
      /* store the original rhs if not done earlier */
      if (save_rhs) {
        CHKERRQ(VecSwap(rhs,pcbddc->original_rhs));
      }
      if (pcbddc->rhs_change) {
        CHKERRQ(MatMultAdd(pc->mat,pcbddc->benign_vec,rhs,rhs));
      } else {
        CHKERRQ(MatMultAdd(pc->mat,pcbddc->benign_vec,pcbddc->original_rhs,rhs));
      }
      pcbddc->rhs_change = PETSC_TRUE;
    }
    pcbddc->benign_apply_coarse_only = PETSC_FALSE;
  } else {
    CHKERRQ(VecDestroy(&pcbddc->benign_vec));
  }

  /* dbg output */
  if (pcbddc->dbg_flag && benign_correction_computed) {
    Vec v;

    CHKERRQ(VecDuplicate(pcis->vec1_global,&v));
    if (pcbddc->ChangeOfBasisMatrix) {
      CHKERRQ(MatMultTranspose(pcbddc->ChangeOfBasisMatrix,rhs,v));
    } else {
      CHKERRQ(VecCopy(rhs,v));
    }
    CHKERRQ(PCBDDCBenignGetOrSetP0(pc,v,PETSC_TRUE));
    CHKERRQ(PetscViewerASCIIPrintf(pcbddc->dbg_viewer,"LEVEL %D: is the correction benign?\n",pcbddc->current_level));
    CHKERRQ(PetscScalarView(pcbddc->benign_n,pcbddc->benign_p0,pcbddc->dbg_viewer));
    CHKERRQ(PetscViewerFlush(pcbddc->dbg_viewer));
    CHKERRQ(VecDestroy(&v));
  }

  /* set initial guess if using PCG */
  pcbddc->exact_dirichlet_trick_app = PETSC_FALSE;
  if (x && pcbddc->use_exact_dirichlet_trick) {
    CHKERRQ(VecSet(x,0.0));
    if (pcbddc->ChangeOfBasisMatrix && pcbddc->change_interior) {
      if (benign_correction_computed) { /* we have already saved the changed rhs */
        CHKERRQ(VecLockReadPop(pcis->vec1_global));
      } else {
        CHKERRQ(MatMultTranspose(pcbddc->ChangeOfBasisMatrix,rhs,pcis->vec1_global));
      }
      CHKERRQ(VecScatterBegin(pcis->global_to_D,pcis->vec1_global,pcis->vec1_D,INSERT_VALUES,SCATTER_FORWARD));
      CHKERRQ(VecScatterEnd(pcis->global_to_D,pcis->vec1_global,pcis->vec1_D,INSERT_VALUES,SCATTER_FORWARD));
    } else {
      CHKERRQ(VecScatterBegin(pcis->global_to_D,rhs,pcis->vec1_D,INSERT_VALUES,SCATTER_FORWARD));
      CHKERRQ(VecScatterEnd(pcis->global_to_D,rhs,pcis->vec1_D,INSERT_VALUES,SCATTER_FORWARD));
    }
    CHKERRQ(PetscLogEventBegin(PC_BDDC_Solves[pcbddc->current_level][0],pc,0,0,0));
    CHKERRQ(KSPSolve(pcbddc->ksp_D,pcis->vec1_D,pcis->vec2_D));
    CHKERRQ(PetscLogEventEnd(PC_BDDC_Solves[pcbddc->current_level][0],pc,0,0,0));
    CHKERRQ(KSPCheckSolve(pcbddc->ksp_D,pc,pcis->vec2_D));
    if (pcbddc->ChangeOfBasisMatrix && pcbddc->change_interior) {
      CHKERRQ(VecSet(pcis->vec1_global,0.));
      CHKERRQ(VecScatterBegin(pcis->global_to_D,pcis->vec2_D,pcis->vec1_global,INSERT_VALUES,SCATTER_REVERSE));
      CHKERRQ(VecScatterEnd(pcis->global_to_D,pcis->vec2_D,pcis->vec1_global,INSERT_VALUES,SCATTER_REVERSE));
      CHKERRQ(MatMult(pcbddc->ChangeOfBasisMatrix,pcis->vec1_global,x));
    } else {
      CHKERRQ(VecScatterBegin(pcis->global_to_D,pcis->vec2_D,x,INSERT_VALUES,SCATTER_REVERSE));
      CHKERRQ(VecScatterEnd(pcis->global_to_D,pcis->vec2_D,x,INSERT_VALUES,SCATTER_REVERSE));
    }
    if (ksp) {
      CHKERRQ(KSPSetInitialGuessNonzero(ksp,PETSC_TRUE));
    }
    pcbddc->exact_dirichlet_trick_app = PETSC_TRUE;
  } else if (pcbddc->ChangeOfBasisMatrix && pcbddc->change_interior && benign_correction_computed && pcbddc->use_exact_dirichlet_trick) {
    CHKERRQ(VecLockReadPop(pcis->vec1_global));
  }
  PetscFunctionReturn(0);
}

/*
   PCPostSolve_BDDC - Changes the computed solution if a transformation of basis
                     approach has been selected. Also, restores rhs to its original state.

   Input Parameter:
+  pc - the preconditioner context

   Application Interface Routine: PCPostSolve()

   Notes:
     The interface routine PCPostSolve() is not usually called directly by
     the user, but instead is called by KSPSolve().
*/
static PetscErrorCode PCPostSolve_BDDC(PC pc, KSP ksp, Vec rhs, Vec x)
{
  PC_BDDC        *pcbddc = (PC_BDDC*)pc->data;

  PetscFunctionBegin;
  /* add solution removed in presolve */
  if (x && pcbddc->rhs_change) {
    if (pcbddc->temp_solution_used) {
      CHKERRQ(VecAXPY(x,1.0,pcbddc->temp_solution));
    } else if (pcbddc->benign_compute_correction && pcbddc->benign_vec) {
      CHKERRQ(VecAXPY(x,-1.0,pcbddc->benign_vec));
    }
    /* restore to original state (not for FETI-DP) */
    if (ksp) pcbddc->temp_solution_used = PETSC_FALSE;
  }

  /* restore rhs to its original state (not needed for FETI-DP) */
  if (rhs && pcbddc->rhs_change) {
    CHKERRQ(VecSwap(rhs,pcbddc->original_rhs));
    pcbddc->rhs_change = PETSC_FALSE;
  }
  /* restore ksp guess state */
  if (ksp) {
    CHKERRQ(KSPSetInitialGuessNonzero(ksp,pcbddc->ksp_guess_nonzero));
    /* reset flag for exact dirichlet trick */
    pcbddc->exact_dirichlet_trick_app = PETSC_FALSE;
  }
  PetscFunctionReturn(0);
}

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
  PC_BDDC*        pcbddc = (PC_BDDC*)pc->data;
  PCBDDCSubSchurs sub_schurs;
  Mat_IS*         matis;
  MatNullSpace    nearnullspace;
  Mat             lA;
  IS              lP,zerodiag = NULL;
  PetscInt        nrows,ncols;
  PetscMPIInt     size;
  PetscBool       computesubschurs;
  PetscBool       computeconstraintsmatrix;
  PetscBool       new_nearnullspace_provided,ismatis,rl;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectTypeCompare((PetscObject)pc->pmat,MATIS,&ismatis));
  PetscCheckFalse(!ismatis,PetscObjectComm((PetscObject)pc),PETSC_ERR_ARG_WRONG,"PCBDDC preconditioner requires matrix of type MATIS");
  CHKERRQ(MatGetSize(pc->pmat,&nrows,&ncols));
  PetscCheckFalse(nrows != ncols,PetscObjectComm((PetscObject)pc),PETSC_ERR_SUP,"PCBDDC preconditioner requires a square preconditioning matrix");
  CHKERRMPI(MPI_Comm_size(PetscObjectComm((PetscObject)pc),&size));

  matis = (Mat_IS*)pc->pmat->data;
  /* the following lines of code should be replaced by a better logic between PCIS, PCNN, PCBDDC and other future nonoverlapping preconditioners */
  /* For BDDC we need to define a local "Neumann" problem different to that defined in PCISSetup
     Also, BDDC builds its own KSP for the Dirichlet problem */
  rl = pcbddc->recompute_topography;
  if (!pc->setupcalled || pc->flag == DIFFERENT_NONZERO_PATTERN) rl = PETSC_TRUE;
  CHKERRMPI(MPIU_Allreduce(&rl,&pcbddc->recompute_topography,1,MPIU_BOOL,MPI_LOR,PetscObjectComm((PetscObject)pc)));
  if (pcbddc->recompute_topography) {
    pcbddc->graphanalyzed    = PETSC_FALSE;
    computeconstraintsmatrix = PETSC_TRUE;
  } else {
    computeconstraintsmatrix = PETSC_FALSE;
  }

  /* check parameters' compatibility */
  if (!pcbddc->use_deluxe_scaling) pcbddc->deluxe_zerorows = PETSC_FALSE;
  pcbddc->adaptive_selection   = (PetscBool)(pcbddc->adaptive_threshold[0] != 0.0 || pcbddc->adaptive_threshold[1] != 0.0);
  pcbddc->use_deluxe_scaling   = (PetscBool)(pcbddc->use_deluxe_scaling && size > 1);
  pcbddc->adaptive_selection   = (PetscBool)(pcbddc->adaptive_selection && size > 1);
  pcbddc->adaptive_userdefined = (PetscBool)(pcbddc->adaptive_selection && pcbddc->adaptive_userdefined);
  if (pcbddc->adaptive_selection) pcbddc->use_faces = PETSC_TRUE;

  computesubschurs = (PetscBool)(pcbddc->adaptive_selection || pcbddc->use_deluxe_scaling);

  /* activate all connected components if the netflux has been requested */
  if (pcbddc->compute_nonetflux) {
    pcbddc->use_vertices = PETSC_TRUE;
    pcbddc->use_edges    = PETSC_TRUE;
    pcbddc->use_faces    = PETSC_TRUE;
  }

  /* Get stdout for dbg */
  if (pcbddc->dbg_flag) {
    if (!pcbddc->dbg_viewer) {
      pcbddc->dbg_viewer = PETSC_VIEWER_STDOUT_(PetscObjectComm((PetscObject)pc));
    }
    CHKERRQ(PetscViewerASCIIPushSynchronized(pcbddc->dbg_viewer));
    CHKERRQ(PetscViewerASCIIAddTab(pcbddc->dbg_viewer,2*pcbddc->current_level));
  }

  /* process topology information */
  CHKERRQ(PetscLogEventBegin(PC_BDDC_Topology[pcbddc->current_level],pc,0,0,0));
  if (pcbddc->recompute_topography) {
    CHKERRQ(PCBDDCComputeLocalTopologyInfo(pc));
    if (pcbddc->discretegradient) {
      CHKERRQ(PCBDDCNedelecSupport(pc));
    }
  }
  if (pcbddc->corner_selected) pcbddc->use_vertices = PETSC_TRUE;

  /* change basis if requested by the user */
  if (pcbddc->user_ChangeOfBasisMatrix) {
    /* use_change_of_basis flag is used to automatically compute a change of basis from constraints */
    pcbddc->use_change_of_basis = PETSC_FALSE;
    CHKERRQ(PCBDDCComputeLocalMatrix(pc,pcbddc->user_ChangeOfBasisMatrix));
  } else {
    CHKERRQ(MatDestroy(&pcbddc->local_mat));
    CHKERRQ(PetscObjectReference((PetscObject)matis->A));
    pcbddc->local_mat = matis->A;
  }

  /*
     Compute change of basis on local pressures (aka zerodiag dofs) with the benign trick
     This should come earlier then PCISSetUp for extracting the correct subdomain matrices
  */
  CHKERRQ(PCBDDCBenignShellMat(pc,PETSC_TRUE));
  if (pcbddc->benign_saddle_point) {
    PC_IS* pcis = (PC_IS*)pc->data;

    if (pcbddc->user_ChangeOfBasisMatrix || pcbddc->use_change_of_basis || !computesubschurs) pcbddc->benign_change_explicit = PETSC_TRUE;
    /* detect local saddle point and change the basis in pcbddc->local_mat */
    CHKERRQ(PCBDDCBenignDetectSaddlePoint(pc,(PetscBool)(!pcbddc->recompute_topography),&zerodiag));
    /* pop B0 mat from local mat */
    CHKERRQ(PCBDDCBenignPopOrPushB0(pc,PETSC_TRUE));
    /* give pcis a hint to not reuse submatrices during PCISCreate */
    if (pc->flag == SAME_NONZERO_PATTERN && pcis->reusesubmatrices == PETSC_TRUE) {
      if (pcbddc->benign_n && (pcbddc->benign_change_explicit || pcbddc->dbg_flag)) {
        pcis->reusesubmatrices = PETSC_FALSE;
      } else {
        pcis->reusesubmatrices = PETSC_TRUE;
      }
    } else {
      pcis->reusesubmatrices = PETSC_FALSE;
    }
  }

  /* propagate relevant information */
  if (matis->A->symmetric_set) {
    CHKERRQ(MatSetOption(pcbddc->local_mat,MAT_SYMMETRIC,matis->A->symmetric));
  }
  if (matis->A->spd_set) {
    CHKERRQ(MatSetOption(pcbddc->local_mat,MAT_SPD,matis->A->spd));
  }

  /* Set up all the "iterative substructuring" common block without computing solvers */
  {
    Mat temp_mat;

    temp_mat = matis->A;
    matis->A = pcbddc->local_mat;
    CHKERRQ(PCISSetUp(pc,PETSC_TRUE,PETSC_FALSE));
    pcbddc->local_mat = matis->A;
    matis->A = temp_mat;
  }

  /* Analyze interface */
  if (!pcbddc->graphanalyzed) {
    CHKERRQ(PCBDDCAnalyzeInterface(pc));
    computeconstraintsmatrix = PETSC_TRUE;
    if (pcbddc->adaptive_selection && !pcbddc->use_deluxe_scaling && !pcbddc->mat_graph->twodim) {
      SETERRQ(PetscObjectComm((PetscObject)pc),PETSC_ERR_SUP,"Cannot compute the adaptive primal space for a problem with 3D edges without deluxe scaling");
    }
    if (pcbddc->compute_nonetflux) {
      MatNullSpace nnfnnsp;

      PetscCheckFalse(!pcbddc->divudotp,PetscObjectComm((PetscObject)pc),PETSC_ERR_SUP,"Missing divudotp operator");
      CHKERRQ(PCBDDCComputeNoNetFlux(pc->pmat,pcbddc->divudotp,pcbddc->divudotp_trans,pcbddc->divudotp_vl2l,pcbddc->mat_graph,&nnfnnsp));
      /* TODO what if a nearnullspace is already attached? */
      if (nnfnnsp) {
        CHKERRQ(MatSetNearNullSpace(pc->pmat,nnfnnsp));
        CHKERRQ(MatNullSpaceDestroy(&nnfnnsp));
      }
    }
  }
  CHKERRQ(PetscLogEventEnd(PC_BDDC_Topology[pcbddc->current_level],pc,0,0,0));

  /* check existence of a divergence free extension, i.e.
     b(v_I,p_0) = 0 for all v_I (raise error if not).
     Also, check that PCBDDCBenignGetOrSetP0 works */
  if (pcbddc->benign_saddle_point && pcbddc->dbg_flag > 1) {
    CHKERRQ(PCBDDCBenignCheck(pc,zerodiag));
  }
  CHKERRQ(ISDestroy(&zerodiag));

  /* Setup local dirichlet solver ksp_D and sub_schurs solvers */
  if (computesubschurs && pcbddc->recompute_topography) {
    CHKERRQ(PCBDDCInitSubSchurs(pc));
  }
  /* SetUp Scaling operator (scaling matrices could be needed in SubSchursSetUp)*/
  if (!pcbddc->use_deluxe_scaling) {
    CHKERRQ(PCBDDCScalingSetUp(pc));
  }

  /* finish setup solvers and do adaptive selection of constraints */
  sub_schurs = pcbddc->sub_schurs;
  if (sub_schurs && sub_schurs->schur_explicit) {
    if (computesubschurs) {
      CHKERRQ(PCBDDCSetUpSubSchurs(pc));
    }
    CHKERRQ(PCBDDCSetUpLocalSolvers(pc,PETSC_TRUE,PETSC_FALSE));
  } else {
    CHKERRQ(PCBDDCSetUpLocalSolvers(pc,PETSC_TRUE,PETSC_FALSE));
    if (computesubschurs) {
      CHKERRQ(PCBDDCSetUpSubSchurs(pc));
    }
  }
  if (pcbddc->adaptive_selection) {
    CHKERRQ(PCBDDCAdaptiveSelection(pc));
    computeconstraintsmatrix = PETSC_TRUE;
  }

  /* infer if NullSpace object attached to Mat via MatSetNearNullSpace has changed */
  new_nearnullspace_provided = PETSC_FALSE;
  CHKERRQ(MatGetNearNullSpace(pc->pmat,&nearnullspace));
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
        CHKERRQ(MatNullSpaceGetVecs(nearnullspace,NULL,&nnsp_size,&nearnullvecs));
        for (i=0;i<nnsp_size;i++) {
          CHKERRQ(PetscObjectStateGet((PetscObject)nearnullvecs[i],&state));
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
  CHKERRQ(PetscLogEventBegin(PC_BDDC_LocalWork[pcbddc->current_level],pc,0,0,0));
  pcbddc->new_primal_space = PETSC_FALSE;
  pcbddc->new_primal_space_local = PETSC_FALSE;
  if (computeconstraintsmatrix || new_nearnullspace_provided) {
    /* It also sets the primal space flags */
    CHKERRQ(PCBDDCConstraintsSetUp(pc));
  }
  /* Allocate needed local vectors (which depends on quantities defined during ConstraintsSetUp) */
  CHKERRQ(PCBDDCSetUpLocalWorkVectors(pc));

  if (pcbddc->use_change_of_basis) {
    PC_IS *pcis = (PC_IS*)(pc->data);

    CHKERRQ(PCBDDCComputeLocalMatrix(pc,pcbddc->ChangeOfBasisMatrix));
    if (pcbddc->benign_change) {
      CHKERRQ(MatDestroy(&pcbddc->benign_B0));
      /* pop B0 from pcbddc->local_mat */
      CHKERRQ(PCBDDCBenignPopOrPushB0(pc,PETSC_TRUE));
    }
    /* get submatrices */
    CHKERRQ(MatDestroy(&pcis->A_IB));
    CHKERRQ(MatDestroy(&pcis->A_BI));
    CHKERRQ(MatDestroy(&pcis->A_BB));
    CHKERRQ(MatCreateSubMatrix(pcbddc->local_mat,pcis->is_B_local,pcis->is_B_local,MAT_INITIAL_MATRIX,&pcis->A_BB));
    CHKERRQ(MatCreateSubMatrix(pcbddc->local_mat,pcis->is_I_local,pcis->is_B_local,MAT_INITIAL_MATRIX,&pcis->A_IB));
    CHKERRQ(MatCreateSubMatrix(pcbddc->local_mat,pcis->is_B_local,pcis->is_I_local,MAT_INITIAL_MATRIX,&pcis->A_BI));
    /* set flag in pcis to not reuse submatrices during PCISCreate */
    pcis->reusesubmatrices = PETSC_FALSE;
  } else if (!pcbddc->user_ChangeOfBasisMatrix && !pcbddc->benign_change) {
    CHKERRQ(MatDestroy(&pcbddc->local_mat));
    CHKERRQ(PetscObjectReference((PetscObject)matis->A));
    pcbddc->local_mat = matis->A;
  }

  /* interface pressure block row for B_C */
  CHKERRQ(PetscObjectQuery((PetscObject)pc,"__KSPFETIDP_lP" ,(PetscObject*)&lP));
  CHKERRQ(PetscObjectQuery((PetscObject)pc,"__KSPFETIDP_lA" ,(PetscObject*)&lA));
  if (lA && lP) {
    PC_IS*    pcis = (PC_IS*)pc->data;
    Mat       B_BI,B_BB,Bt_BI,Bt_BB;
    PetscBool issym;
    CHKERRQ(MatIsSymmetric(lA,PETSC_SMALL,&issym));
    if (issym) {
      CHKERRQ(MatCreateSubMatrix(lA,lP,pcis->is_I_local,MAT_INITIAL_MATRIX,&B_BI));
      CHKERRQ(MatCreateSubMatrix(lA,lP,pcis->is_B_local,MAT_INITIAL_MATRIX,&B_BB));
      CHKERRQ(MatCreateTranspose(B_BI,&Bt_BI));
      CHKERRQ(MatCreateTranspose(B_BB,&Bt_BB));
    } else {
      CHKERRQ(MatCreateSubMatrix(lA,lP,pcis->is_I_local,MAT_INITIAL_MATRIX,&B_BI));
      CHKERRQ(MatCreateSubMatrix(lA,lP,pcis->is_B_local,MAT_INITIAL_MATRIX,&B_BB));
      CHKERRQ(MatCreateSubMatrix(lA,pcis->is_I_local,lP,MAT_INITIAL_MATRIX,&Bt_BI));
      CHKERRQ(MatCreateSubMatrix(lA,pcis->is_B_local,lP,MAT_INITIAL_MATRIX,&Bt_BB));
    }
    CHKERRQ(PetscObjectCompose((PetscObject)pc,"__KSPFETIDP_B_BI",(PetscObject)B_BI));
    CHKERRQ(PetscObjectCompose((PetscObject)pc,"__KSPFETIDP_B_BB",(PetscObject)B_BB));
    CHKERRQ(PetscObjectCompose((PetscObject)pc,"__KSPFETIDP_Bt_BI",(PetscObject)Bt_BI));
    CHKERRQ(PetscObjectCompose((PetscObject)pc,"__KSPFETIDP_Bt_BB",(PetscObject)Bt_BB));
    CHKERRQ(MatDestroy(&B_BI));
    CHKERRQ(MatDestroy(&B_BB));
    CHKERRQ(MatDestroy(&Bt_BI));
    CHKERRQ(MatDestroy(&Bt_BB));
  }
  CHKERRQ(PetscLogEventEnd(PC_BDDC_LocalWork[pcbddc->current_level],pc,0,0,0));

  /* SetUp coarse and local Neumann solvers */
  CHKERRQ(PCBDDCSetUpSolvers(pc));
  /* SetUp Scaling operator */
  if (pcbddc->use_deluxe_scaling) {
    CHKERRQ(PCBDDCScalingSetUp(pc));
  }

  /* mark topography as done */
  pcbddc->recompute_topography = PETSC_FALSE;

  /* wrap pcis->A_IB and pcis->A_BI if we did not change explicitly the variables on the pressures */
  CHKERRQ(PCBDDCBenignShellMat(pc,PETSC_FALSE));

  if (pcbddc->dbg_flag) {
    CHKERRQ(PetscViewerASCIISubtractTab(pcbddc->dbg_viewer,2*pcbddc->current_level));
    CHKERRQ(PetscViewerASCIIPopSynchronized(pcbddc->dbg_viewer));
  }
  PetscFunctionReturn(0);
}

/*
   PCApply_BDDC - Applies the BDDC operator to a vector.

   Input Parameters:
+  pc - the preconditioner context
-  r - input vector (global)

   Output Parameter:
.  z - output vector (global)

   Application Interface Routine: PCApply()
 */
PetscErrorCode PCApply_BDDC(PC pc,Vec r,Vec z)
{
  PC_IS             *pcis = (PC_IS*)(pc->data);
  PC_BDDC           *pcbddc = (PC_BDDC*)(pc->data);
  Mat               lA = NULL;
  PetscInt          n_B = pcis->n_B, n_D = pcis->n - n_B;
  const PetscScalar one = 1.0;
  const PetscScalar m_one = -1.0;
  const PetscScalar zero = 0.0;
/* This code is similar to that provided in nn.c for PCNN
   NN interface preconditioner changed to BDDC
   Added support for M_3 preconditioner in the reference article (code is active if pcbddc->switch_static == PETSC_TRUE) */

  PetscFunctionBegin;
  CHKERRQ(PetscCitationsRegister(citation,&cited));
  if (pcbddc->switch_static) {
    CHKERRQ(MatISGetLocalMat(pc->useAmat ? pc->mat : pc->pmat,&lA));
  }

  if (pcbddc->ChangeOfBasisMatrix) {
    Vec swap;

    CHKERRQ(MatMultTranspose(pcbddc->ChangeOfBasisMatrix,r,pcbddc->work_change));
    swap = pcbddc->work_change;
    pcbddc->work_change = r;
    r = swap;
    /* save rhs so that we don't need to apply the change of basis for the exact dirichlet trick in PreSolve */
    if (pcbddc->benign_apply_coarse_only && pcbddc->use_exact_dirichlet_trick && pcbddc->change_interior) {
      CHKERRQ(VecCopy(r,pcis->vec1_global));
      CHKERRQ(VecLockReadPush(pcis->vec1_global));
    }
  }
  if (pcbddc->benign_have_null) { /* get p0 from r */
    CHKERRQ(PCBDDCBenignGetOrSetP0(pc,r,PETSC_TRUE));
  }
  if (pcbddc->interface_extension == PC_BDDC_INTERFACE_EXT_DIRICHLET && !pcbddc->exact_dirichlet_trick_app && !pcbddc->benign_apply_coarse_only) {
    CHKERRQ(VecCopy(r,z));
    /* First Dirichlet solve */
    CHKERRQ(VecScatterBegin(pcis->global_to_D,r,pcis->vec1_D,INSERT_VALUES,SCATTER_FORWARD));
    CHKERRQ(VecScatterEnd(pcis->global_to_D,r,pcis->vec1_D,INSERT_VALUES,SCATTER_FORWARD));
    /*
      Assembling right hand side for BDDC operator
      - pcis->vec1_D for the Dirichlet part (if needed, i.e. pcbddc->switch_static == PETSC_TRUE)
      - pcis->vec1_B the interface part of the global vector z
    */
    if (n_D) {
      CHKERRQ(PetscLogEventBegin(PC_BDDC_Solves[pcbddc->current_level][0],pc,0,0,0));
      CHKERRQ(KSPSolve(pcbddc->ksp_D,pcis->vec1_D,pcis->vec2_D));
      CHKERRQ(PetscLogEventEnd(PC_BDDC_Solves[pcbddc->current_level][0],pc,0,0,0));
      CHKERRQ(KSPCheckSolve(pcbddc->ksp_D,pc,pcis->vec2_D));
      CHKERRQ(VecScale(pcis->vec2_D,m_one));
      if (pcbddc->switch_static) {
        CHKERRQ(VecSet(pcis->vec1_N,0.));
        CHKERRQ(VecScatterBegin(pcis->N_to_D,pcis->vec2_D,pcis->vec1_N,INSERT_VALUES,SCATTER_REVERSE));
        CHKERRQ(VecScatterEnd(pcis->N_to_D,pcis->vec2_D,pcis->vec1_N,INSERT_VALUES,SCATTER_REVERSE));
        if (!pcbddc->switch_static_change) {
          CHKERRQ(MatMult(lA,pcis->vec1_N,pcis->vec2_N));
        } else {
          CHKERRQ(MatMult(pcbddc->switch_static_change,pcis->vec1_N,pcis->vec2_N));
          CHKERRQ(MatMult(lA,pcis->vec2_N,pcis->vec1_N));
          CHKERRQ(MatMultTranspose(pcbddc->switch_static_change,pcis->vec1_N,pcis->vec2_N));
        }
        CHKERRQ(VecScatterBegin(pcis->N_to_D,pcis->vec2_N,pcis->vec1_D,ADD_VALUES,SCATTER_FORWARD));
        CHKERRQ(VecScatterEnd(pcis->N_to_D,pcis->vec2_N,pcis->vec1_D,ADD_VALUES,SCATTER_FORWARD));
        CHKERRQ(VecScatterBegin(pcis->N_to_B,pcis->vec2_N,pcis->vec1_B,INSERT_VALUES,SCATTER_FORWARD));
        CHKERRQ(VecScatterEnd(pcis->N_to_B,pcis->vec2_N,pcis->vec1_B,INSERT_VALUES,SCATTER_FORWARD));
      } else {
        CHKERRQ(MatMult(pcis->A_BI,pcis->vec2_D,pcis->vec1_B));
      }
    } else {
      CHKERRQ(VecSet(pcis->vec1_B,zero));
    }
    CHKERRQ(VecScatterBegin(pcis->global_to_B,pcis->vec1_B,z,ADD_VALUES,SCATTER_REVERSE));
    CHKERRQ(VecScatterEnd(pcis->global_to_B,pcis->vec1_B,z,ADD_VALUES,SCATTER_REVERSE));
    CHKERRQ(PCBDDCScalingRestriction(pc,z,pcis->vec1_B));
  } else {
    if (!pcbddc->benign_apply_coarse_only) {
      CHKERRQ(PCBDDCScalingRestriction(pc,r,pcis->vec1_B));
    }
  }
  if (pcbddc->interface_extension == PC_BDDC_INTERFACE_EXT_LUMP) {
    PetscCheckFalse(!pcbddc->switch_static,PetscObjectComm((PetscObject)pc),PETSC_ERR_SUP,"You forgot to pass -pc_bddc_switch_static");
    CHKERRQ(VecScatterBegin(pcis->global_to_D,r,pcis->vec1_D,INSERT_VALUES,SCATTER_FORWARD));
    CHKERRQ(VecScatterEnd(pcis->global_to_D,r,pcis->vec1_D,INSERT_VALUES,SCATTER_FORWARD));
  }

  /* Apply interface preconditioner
     input/output vecs: pcis->vec1_B and pcis->vec1_D */
  CHKERRQ(PCBDDCApplyInterfacePreconditioner(pc,PETSC_FALSE));

  /* Apply transpose of partition of unity operator */
  CHKERRQ(PCBDDCScalingExtension(pc,pcis->vec1_B,z));
  if (pcbddc->interface_extension == PC_BDDC_INTERFACE_EXT_LUMP) {
    CHKERRQ(VecScatterBegin(pcis->global_to_D,pcis->vec1_D,z,INSERT_VALUES,SCATTER_REVERSE));
    CHKERRQ(VecScatterEnd(pcis->global_to_D,pcis->vec1_D,z,INSERT_VALUES,SCATTER_REVERSE));
    PetscFunctionReturn(0);
  }
  /* Second Dirichlet solve and assembling of output */
  CHKERRQ(VecScatterBegin(pcis->global_to_B,z,pcis->vec1_B,INSERT_VALUES,SCATTER_FORWARD));
  CHKERRQ(VecScatterEnd(pcis->global_to_B,z,pcis->vec1_B,INSERT_VALUES,SCATTER_FORWARD));
  if (n_B) {
    if (pcbddc->switch_static) {
      CHKERRQ(VecScatterBegin(pcis->N_to_D,pcis->vec1_D,pcis->vec1_N,INSERT_VALUES,SCATTER_REVERSE));
      CHKERRQ(VecScatterEnd(pcis->N_to_D,pcis->vec1_D,pcis->vec1_N,INSERT_VALUES,SCATTER_REVERSE));
      CHKERRQ(VecScatterBegin(pcis->N_to_B,pcis->vec1_B,pcis->vec1_N,INSERT_VALUES,SCATTER_REVERSE));
      CHKERRQ(VecScatterEnd(pcis->N_to_B,pcis->vec1_B,pcis->vec1_N,INSERT_VALUES,SCATTER_REVERSE));
      if (!pcbddc->switch_static_change) {
        CHKERRQ(MatMult(lA,pcis->vec1_N,pcis->vec2_N));
      } else {
        CHKERRQ(MatMult(pcbddc->switch_static_change,pcis->vec1_N,pcis->vec2_N));
        CHKERRQ(MatMult(lA,pcis->vec2_N,pcis->vec1_N));
        CHKERRQ(MatMultTranspose(pcbddc->switch_static_change,pcis->vec1_N,pcis->vec2_N));
      }
      CHKERRQ(VecScatterBegin(pcis->N_to_D,pcis->vec2_N,pcis->vec3_D,INSERT_VALUES,SCATTER_FORWARD));
      CHKERRQ(VecScatterEnd(pcis->N_to_D,pcis->vec2_N,pcis->vec3_D,INSERT_VALUES,SCATTER_FORWARD));
    } else {
      CHKERRQ(MatMult(pcis->A_IB,pcis->vec1_B,pcis->vec3_D));
    }
  } else if (pcbddc->switch_static) { /* n_B is zero */
    if (!pcbddc->switch_static_change) {
      CHKERRQ(MatMult(lA,pcis->vec1_D,pcis->vec3_D));
    } else {
      CHKERRQ(MatMult(pcbddc->switch_static_change,pcis->vec1_D,pcis->vec1_N));
      CHKERRQ(MatMult(lA,pcis->vec1_N,pcis->vec2_N));
      CHKERRQ(MatMultTranspose(pcbddc->switch_static_change,pcis->vec2_N,pcis->vec3_D));
    }
  }
  CHKERRQ(PetscLogEventBegin(PC_BDDC_Solves[pcbddc->current_level][0],pc,0,0,0));
  CHKERRQ(KSPSolve(pcbddc->ksp_D,pcis->vec3_D,pcis->vec4_D));
  CHKERRQ(PetscLogEventEnd(PC_BDDC_Solves[pcbddc->current_level][0],pc,0,0,0));
  CHKERRQ(KSPCheckSolve(pcbddc->ksp_D,pc,pcis->vec4_D));

  if (!pcbddc->exact_dirichlet_trick_app && !pcbddc->benign_apply_coarse_only) {
    if (pcbddc->switch_static) {
      CHKERRQ(VecAXPBYPCZ(pcis->vec2_D,m_one,one,m_one,pcis->vec4_D,pcis->vec1_D));
    } else {
      CHKERRQ(VecAXPBY(pcis->vec2_D,m_one,m_one,pcis->vec4_D));
    }
    CHKERRQ(VecScatterBegin(pcis->global_to_D,pcis->vec2_D,z,INSERT_VALUES,SCATTER_REVERSE));
    CHKERRQ(VecScatterEnd(pcis->global_to_D,pcis->vec2_D,z,INSERT_VALUES,SCATTER_REVERSE));
  } else {
    if (pcbddc->switch_static) {
      CHKERRQ(VecAXPBY(pcis->vec4_D,one,m_one,pcis->vec1_D));
    } else {
      CHKERRQ(VecScale(pcis->vec4_D,m_one));
    }
    CHKERRQ(VecScatterBegin(pcis->global_to_D,pcis->vec4_D,z,INSERT_VALUES,SCATTER_REVERSE));
    CHKERRQ(VecScatterEnd(pcis->global_to_D,pcis->vec4_D,z,INSERT_VALUES,SCATTER_REVERSE));
  }
  if (pcbddc->benign_have_null) { /* set p0 (computed in PCBDDCApplyInterface) */
    if (pcbddc->benign_apply_coarse_only) {
      CHKERRQ(PetscArrayzero(pcbddc->benign_p0,pcbddc->benign_n));
    }
    CHKERRQ(PCBDDCBenignGetOrSetP0(pc,z,PETSC_FALSE));
  }

  if (pcbddc->ChangeOfBasisMatrix) {
    pcbddc->work_change = r;
    CHKERRQ(VecCopy(z,pcbddc->work_change));
    CHKERRQ(MatMult(pcbddc->ChangeOfBasisMatrix,pcbddc->work_change,z));
  }
  PetscFunctionReturn(0);
}

/*
   PCApplyTranspose_BDDC - Applies the transpose of the BDDC operator to a vector.

   Input Parameters:
+  pc - the preconditioner context
-  r - input vector (global)

   Output Parameter:
.  z - output vector (global)

   Application Interface Routine: PCApplyTranspose()
 */
PetscErrorCode PCApplyTranspose_BDDC(PC pc,Vec r,Vec z)
{
  PC_IS             *pcis = (PC_IS*)(pc->data);
  PC_BDDC           *pcbddc = (PC_BDDC*)(pc->data);
  Mat               lA = NULL;
  PetscInt          n_B = pcis->n_B, n_D = pcis->n - n_B;
  const PetscScalar one = 1.0;
  const PetscScalar m_one = -1.0;
  const PetscScalar zero = 0.0;

  PetscFunctionBegin;
  CHKERRQ(PetscCitationsRegister(citation,&cited));
  if (pcbddc->switch_static) {
    CHKERRQ(MatISGetLocalMat(pc->useAmat ? pc->mat : pc->pmat,&lA));
  }
  if (pcbddc->ChangeOfBasisMatrix) {
    Vec swap;

    CHKERRQ(MatMultTranspose(pcbddc->ChangeOfBasisMatrix,r,pcbddc->work_change));
    swap = pcbddc->work_change;
    pcbddc->work_change = r;
    r = swap;
    /* save rhs so that we don't need to apply the change of basis for the exact dirichlet trick in PreSolve */
    if (pcbddc->benign_apply_coarse_only && pcbddc->exact_dirichlet_trick_app && pcbddc->change_interior) {
      CHKERRQ(VecCopy(r,pcis->vec1_global));
      CHKERRQ(VecLockReadPush(pcis->vec1_global));
    }
  }
  if (pcbddc->benign_have_null) { /* get p0 from r */
    CHKERRQ(PCBDDCBenignGetOrSetP0(pc,r,PETSC_TRUE));
  }
  if (!pcbddc->exact_dirichlet_trick_app && !pcbddc->benign_apply_coarse_only) {
    CHKERRQ(VecCopy(r,z));
    /* First Dirichlet solve */
    CHKERRQ(VecScatterBegin(pcis->global_to_D,r,pcis->vec1_D,INSERT_VALUES,SCATTER_FORWARD));
    CHKERRQ(VecScatterEnd(pcis->global_to_D,r,pcis->vec1_D,INSERT_VALUES,SCATTER_FORWARD));
    /*
      Assembling right hand side for BDDC operator
      - pcis->vec1_D for the Dirichlet part (if needed, i.e. pcbddc->switch_static == PETSC_TRUE)
      - pcis->vec1_B the interface part of the global vector z
    */
    if (n_D) {
      CHKERRQ(PetscLogEventBegin(PC_BDDC_Solves[pcbddc->current_level][0],pc,0,0,0));
      CHKERRQ(KSPSolveTranspose(pcbddc->ksp_D,pcis->vec1_D,pcis->vec2_D));
      CHKERRQ(PetscLogEventEnd(PC_BDDC_Solves[pcbddc->current_level][0],pc,0,0,0));
      CHKERRQ(KSPCheckSolve(pcbddc->ksp_D,pc,pcis->vec2_D));
      CHKERRQ(VecScale(pcis->vec2_D,m_one));
      if (pcbddc->switch_static) {
        CHKERRQ(VecSet(pcis->vec1_N,0.));
        CHKERRQ(VecScatterBegin(pcis->N_to_D,pcis->vec2_D,pcis->vec1_N,INSERT_VALUES,SCATTER_REVERSE));
        CHKERRQ(VecScatterEnd(pcis->N_to_D,pcis->vec2_D,pcis->vec1_N,INSERT_VALUES,SCATTER_REVERSE));
        if (!pcbddc->switch_static_change) {
          CHKERRQ(MatMultTranspose(lA,pcis->vec1_N,pcis->vec2_N));
        } else {
          CHKERRQ(MatMult(pcbddc->switch_static_change,pcis->vec1_N,pcis->vec2_N));
          CHKERRQ(MatMultTranspose(lA,pcis->vec2_N,pcis->vec1_N));
          CHKERRQ(MatMultTranspose(pcbddc->switch_static_change,pcis->vec1_N,pcis->vec2_N));
        }
        CHKERRQ(VecScatterBegin(pcis->N_to_D,pcis->vec2_N,pcis->vec1_D,ADD_VALUES,SCATTER_FORWARD));
        CHKERRQ(VecScatterEnd(pcis->N_to_D,pcis->vec2_N,pcis->vec1_D,ADD_VALUES,SCATTER_FORWARD));
        CHKERRQ(VecScatterBegin(pcis->N_to_B,pcis->vec2_N,pcis->vec1_B,INSERT_VALUES,SCATTER_FORWARD));
        CHKERRQ(VecScatterEnd(pcis->N_to_B,pcis->vec2_N,pcis->vec1_B,INSERT_VALUES,SCATTER_FORWARD));
      } else {
        CHKERRQ(MatMultTranspose(pcis->A_IB,pcis->vec2_D,pcis->vec1_B));
      }
    } else {
      CHKERRQ(VecSet(pcis->vec1_B,zero));
    }
    CHKERRQ(VecScatterBegin(pcis->global_to_B,pcis->vec1_B,z,ADD_VALUES,SCATTER_REVERSE));
    CHKERRQ(VecScatterEnd(pcis->global_to_B,pcis->vec1_B,z,ADD_VALUES,SCATTER_REVERSE));
    CHKERRQ(PCBDDCScalingRestriction(pc,z,pcis->vec1_B));
  } else {
    CHKERRQ(PCBDDCScalingRestriction(pc,r,pcis->vec1_B));
  }

  /* Apply interface preconditioner
     input/output vecs: pcis->vec1_B and pcis->vec1_D */
  CHKERRQ(PCBDDCApplyInterfacePreconditioner(pc,PETSC_TRUE));

  /* Apply transpose of partition of unity operator */
  CHKERRQ(PCBDDCScalingExtension(pc,pcis->vec1_B,z));

  /* Second Dirichlet solve and assembling of output */
  CHKERRQ(VecScatterBegin(pcis->global_to_B,z,pcis->vec1_B,INSERT_VALUES,SCATTER_FORWARD));
  CHKERRQ(VecScatterEnd(pcis->global_to_B,z,pcis->vec1_B,INSERT_VALUES,SCATTER_FORWARD));
  if (n_B) {
    if (pcbddc->switch_static) {
      CHKERRQ(VecScatterBegin(pcis->N_to_D,pcis->vec1_D,pcis->vec1_N,INSERT_VALUES,SCATTER_REVERSE));
      CHKERRQ(VecScatterEnd(pcis->N_to_D,pcis->vec1_D,pcis->vec1_N,INSERT_VALUES,SCATTER_REVERSE));
      CHKERRQ(VecScatterBegin(pcis->N_to_B,pcis->vec1_B,pcis->vec1_N,INSERT_VALUES,SCATTER_REVERSE));
      CHKERRQ(VecScatterEnd(pcis->N_to_B,pcis->vec1_B,pcis->vec1_N,INSERT_VALUES,SCATTER_REVERSE));
      if (!pcbddc->switch_static_change) {
        CHKERRQ(MatMultTranspose(lA,pcis->vec1_N,pcis->vec2_N));
      } else {
        CHKERRQ(MatMult(pcbddc->switch_static_change,pcis->vec1_N,pcis->vec2_N));
        CHKERRQ(MatMultTranspose(lA,pcis->vec2_N,pcis->vec1_N));
        CHKERRQ(MatMultTranspose(pcbddc->switch_static_change,pcis->vec1_N,pcis->vec2_N));
      }
      CHKERRQ(VecScatterBegin(pcis->N_to_D,pcis->vec2_N,pcis->vec3_D,INSERT_VALUES,SCATTER_FORWARD));
      CHKERRQ(VecScatterEnd(pcis->N_to_D,pcis->vec2_N,pcis->vec3_D,INSERT_VALUES,SCATTER_FORWARD));
    } else {
      CHKERRQ(MatMultTranspose(pcis->A_BI,pcis->vec1_B,pcis->vec3_D));
    }
  } else if (pcbddc->switch_static) { /* n_B is zero */
    if (!pcbddc->switch_static_change) {
      CHKERRQ(MatMultTranspose(lA,pcis->vec1_D,pcis->vec3_D));
    } else {
      CHKERRQ(MatMult(pcbddc->switch_static_change,pcis->vec1_D,pcis->vec1_N));
      CHKERRQ(MatMultTranspose(lA,pcis->vec1_N,pcis->vec2_N));
      CHKERRQ(MatMultTranspose(pcbddc->switch_static_change,pcis->vec2_N,pcis->vec3_D));
    }
  }
  CHKERRQ(PetscLogEventBegin(PC_BDDC_Solves[pcbddc->current_level][0],pc,0,0,0));
  CHKERRQ(KSPSolveTranspose(pcbddc->ksp_D,pcis->vec3_D,pcis->vec4_D));
  CHKERRQ(PetscLogEventEnd(PC_BDDC_Solves[pcbddc->current_level][0],pc,0,0,0));
  CHKERRQ(KSPCheckSolve(pcbddc->ksp_D,pc,pcis->vec4_D));
  if (!pcbddc->exact_dirichlet_trick_app && !pcbddc->benign_apply_coarse_only) {
    if (pcbddc->switch_static) {
      CHKERRQ(VecAXPBYPCZ(pcis->vec2_D,m_one,one,m_one,pcis->vec4_D,pcis->vec1_D));
    } else {
      CHKERRQ(VecAXPBY(pcis->vec2_D,m_one,m_one,pcis->vec4_D));
    }
    CHKERRQ(VecScatterBegin(pcis->global_to_D,pcis->vec2_D,z,INSERT_VALUES,SCATTER_REVERSE));
    CHKERRQ(VecScatterEnd(pcis->global_to_D,pcis->vec2_D,z,INSERT_VALUES,SCATTER_REVERSE));
  } else {
    if (pcbddc->switch_static) {
      CHKERRQ(VecAXPBY(pcis->vec4_D,one,m_one,pcis->vec1_D));
    } else {
      CHKERRQ(VecScale(pcis->vec4_D,m_one));
    }
    CHKERRQ(VecScatterBegin(pcis->global_to_D,pcis->vec4_D,z,INSERT_VALUES,SCATTER_REVERSE));
    CHKERRQ(VecScatterEnd(pcis->global_to_D,pcis->vec4_D,z,INSERT_VALUES,SCATTER_REVERSE));
  }
  if (pcbddc->benign_have_null) { /* set p0 (computed in PCBDDCApplyInterface) */
    CHKERRQ(PCBDDCBenignGetOrSetP0(pc,z,PETSC_FALSE));
  }
  if (pcbddc->ChangeOfBasisMatrix) {
    pcbddc->work_change = r;
    CHKERRQ(VecCopy(z,pcbddc->work_change));
    CHKERRQ(MatMult(pcbddc->ChangeOfBasisMatrix,pcbddc->work_change,z));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PCReset_BDDC(PC pc)
{
  PC_BDDC        *pcbddc = (PC_BDDC*)pc->data;
  PC_IS          *pcis = (PC_IS*)pc->data;
  KSP            kspD,kspR,kspC;

  PetscFunctionBegin;
  /* free BDDC custom data  */
  CHKERRQ(PCBDDCResetCustomization(pc));
  /* destroy objects related to topography */
  CHKERRQ(PCBDDCResetTopography(pc));
  /* destroy objects for scaling operator */
  CHKERRQ(PCBDDCScalingDestroy(pc));
  /* free solvers stuff */
  CHKERRQ(PCBDDCResetSolvers(pc));
  /* free global vectors needed in presolve */
  CHKERRQ(VecDestroy(&pcbddc->temp_solution));
  CHKERRQ(VecDestroy(&pcbddc->original_rhs));
  /* free data created by PCIS */
  CHKERRQ(PCISDestroy(pc));

  /* restore defaults */
  kspD = pcbddc->ksp_D;
  kspR = pcbddc->ksp_R;
  kspC = pcbddc->coarse_ksp;
  CHKERRQ(PetscMemzero(pc->data,sizeof(*pcbddc)));
  pcis->n_neigh                     = -1;
  pcis->scaling_factor              = 1.0;
  pcis->reusesubmatrices            = PETSC_TRUE;
  pcbddc->use_local_adj             = PETSC_TRUE;
  pcbddc->use_vertices              = PETSC_TRUE;
  pcbddc->use_edges                 = PETSC_TRUE;
  pcbddc->symmetric_primal          = PETSC_TRUE;
  pcbddc->vertex_size               = 1;
  pcbddc->recompute_topography      = PETSC_TRUE;
  pcbddc->coarse_size               = -1;
  pcbddc->use_exact_dirichlet_trick = PETSC_TRUE;
  pcbddc->coarsening_ratio          = 8;
  pcbddc->coarse_eqs_per_proc       = 1;
  pcbddc->benign_compute_correction = PETSC_TRUE;
  pcbddc->nedfield                  = -1;
  pcbddc->nedglobal                 = PETSC_TRUE;
  pcbddc->graphmaxcount             = PETSC_MAX_INT;
  pcbddc->sub_schurs_layers         = -1;
  pcbddc->ksp_D                     = kspD;
  pcbddc->ksp_R                     = kspR;
  pcbddc->coarse_ksp                = kspC;
  PetscFunctionReturn(0);
}

PetscErrorCode PCDestroy_BDDC(PC pc)
{
  PC_BDDC        *pcbddc = (PC_BDDC*)pc->data;

  PetscFunctionBegin;
  CHKERRQ(PCReset_BDDC(pc));
  CHKERRQ(KSPDestroy(&pcbddc->ksp_D));
  CHKERRQ(KSPDestroy(&pcbddc->ksp_R));
  CHKERRQ(KSPDestroy(&pcbddc->coarse_ksp));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)pc,"PCBDDCSetDiscreteGradient_C",NULL));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)pc,"PCBDDCSetDivergenceMat_C",NULL));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)pc,"PCBDDCSetChangeOfBasisMat_C",NULL));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)pc,"PCBDDCSetPrimalVerticesLocalIS_C",NULL));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)pc,"PCBDDCSetPrimalVerticesIS_C",NULL));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)pc,"PCBDDCSetCoarseningRatio_C",NULL));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)pc,"PCBDDCSetLevel_C",NULL));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)pc,"PCBDDCSetUseExactDirichlet_C",NULL));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)pc,"PCBDDCSetLevels_C",NULL));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)pc,"PCBDDCSetDirichletBoundaries_C",NULL));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)pc,"PCBDDCSetDirichletBoundariesLocal_C",NULL));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)pc,"PCBDDCSetNeumannBoundaries_C",NULL));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)pc,"PCBDDCSetNeumannBoundariesLocal_C",NULL));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)pc,"PCBDDCGetDirichletBoundaries_C",NULL));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)pc,"PCBDDCGetDirichletBoundariesLocal_C",NULL));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)pc,"PCBDDCGetNeumannBoundaries_C",NULL));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)pc,"PCBDDCGetNeumannBoundariesLocal_C",NULL));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)pc,"PCBDDCSetDofsSplitting_C",NULL));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)pc,"PCBDDCSetDofsSplittingLocal_C",NULL));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)pc,"PCBDDCSetLocalAdjacencyGraph_C",NULL));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)pc,"PCBDDCCreateFETIDPOperators_C",NULL));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)pc,"PCBDDCMatFETIDPGetRHS_C",NULL));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)pc,"PCBDDCMatFETIDPGetSolution_C",NULL));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)pc,"PCPreSolveChangeRHS_C",NULL));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)pc,"PCSetCoordinates_C",NULL));
  CHKERRQ(PetscFree(pc->data));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCSetCoordinates_BDDC(PC pc, PetscInt dim, PetscInt nloc, PetscReal *coords)
{
  PC_BDDC        *pcbddc = (PC_BDDC*)pc->data;
  PCBDDCGraph    mat_graph = pcbddc->mat_graph;

  PetscFunctionBegin;
  CHKERRQ(PetscFree(mat_graph->coords));
  CHKERRQ(PetscMalloc1(nloc*dim,&mat_graph->coords));
  CHKERRQ(PetscArraycpy(mat_graph->coords,coords,nloc*dim));
  mat_graph->cnloc = nloc;
  mat_graph->cdim  = dim;
  mat_graph->cloc  = PETSC_FALSE;
  /* flg setup */
  pcbddc->recompute_topography = PETSC_TRUE;
  pcbddc->corner_selected = PETSC_FALSE;
  PetscFunctionReturn(0);
}

static PetscErrorCode PCPreSolveChangeRHS_BDDC(PC pc, PetscBool* change)
{
  PetscFunctionBegin;
  *change = PETSC_TRUE;
  PetscFunctionReturn(0);
}

static PetscErrorCode PCBDDCMatFETIDPGetRHS_BDDC(Mat fetidp_mat, Vec standard_rhs, Vec fetidp_flux_rhs)
{
  FETIDPMat_ctx  mat_ctx;
  Vec            work;
  PC_IS*         pcis;
  PC_BDDC*       pcbddc;

  PetscFunctionBegin;
  CHKERRQ(MatShellGetContext(fetidp_mat,&mat_ctx));
  pcis = (PC_IS*)mat_ctx->pc->data;
  pcbddc = (PC_BDDC*)mat_ctx->pc->data;

  CHKERRQ(VecSet(fetidp_flux_rhs,0.0));
  /* copy rhs since we may change it during PCPreSolve_BDDC */
  if (!pcbddc->original_rhs) {
    CHKERRQ(VecDuplicate(pcis->vec1_global,&pcbddc->original_rhs));
  }
  if (mat_ctx->rhs_flip) {
    CHKERRQ(VecPointwiseMult(pcbddc->original_rhs,standard_rhs,mat_ctx->rhs_flip));
  } else {
    CHKERRQ(VecCopy(standard_rhs,pcbddc->original_rhs));
  }
  if (mat_ctx->g2g_p) {
    /* interface pressure rhs */
    CHKERRQ(VecScatterBegin(mat_ctx->g2g_p,fetidp_flux_rhs,pcbddc->original_rhs,INSERT_VALUES,SCATTER_REVERSE));
    CHKERRQ(VecScatterEnd(mat_ctx->g2g_p,fetidp_flux_rhs,pcbddc->original_rhs,INSERT_VALUES,SCATTER_REVERSE));
    CHKERRQ(VecScatterBegin(mat_ctx->g2g_p,standard_rhs,fetidp_flux_rhs,INSERT_VALUES,SCATTER_FORWARD));
    CHKERRQ(VecScatterEnd(mat_ctx->g2g_p,standard_rhs,fetidp_flux_rhs,INSERT_VALUES,SCATTER_FORWARD));
    if (!mat_ctx->rhs_flip) {
      CHKERRQ(VecScale(fetidp_flux_rhs,-1.));
    }
  }
  /*
     change of basis for physical rhs if needed
     It also changes the rhs in case of dirichlet boundaries
  */
  CHKERRQ(PCPreSolve_BDDC(mat_ctx->pc,NULL,pcbddc->original_rhs,NULL));
  if (pcbddc->ChangeOfBasisMatrix) {
    CHKERRQ(MatMultTranspose(pcbddc->ChangeOfBasisMatrix,pcbddc->original_rhs,pcbddc->work_change));
    work = pcbddc->work_change;
   } else {
    work = pcbddc->original_rhs;
  }
  /* store vectors for computation of fetidp final solution */
  CHKERRQ(VecScatterBegin(pcis->global_to_D,work,mat_ctx->temp_solution_D,INSERT_VALUES,SCATTER_FORWARD));
  CHKERRQ(VecScatterEnd(pcis->global_to_D,work,mat_ctx->temp_solution_D,INSERT_VALUES,SCATTER_FORWARD));
  /* scale rhs since it should be unassembled */
  /* TODO use counter scaling? (also below) */
  CHKERRQ(VecScatterBegin(pcis->global_to_B,work,mat_ctx->temp_solution_B,INSERT_VALUES,SCATTER_FORWARD));
  CHKERRQ(VecScatterEnd(pcis->global_to_B,work,mat_ctx->temp_solution_B,INSERT_VALUES,SCATTER_FORWARD));
  /* Apply partition of unity */
  CHKERRQ(VecPointwiseMult(mat_ctx->temp_solution_B,pcis->D,mat_ctx->temp_solution_B));
  /* CHKERRQ(PCBDDCScalingRestriction(mat_ctx->pc,work,mat_ctx->temp_solution_B)); */
  if (!pcbddc->switch_static) {
    /* compute partially subassembled Schur complement right-hand side */
    CHKERRQ(PetscLogEventBegin(PC_BDDC_Solves[pcbddc->current_level][0],mat_ctx->pc,0,0,0));
    CHKERRQ(KSPSolve(pcbddc->ksp_D,mat_ctx->temp_solution_D,pcis->vec1_D));
    CHKERRQ(PetscLogEventEnd(PC_BDDC_Solves[pcbddc->current_level][0],mat_ctx->pc,0,0,0));
    /* Cannot propagate up error in KSPSolve() because there is no access to the PC */
    CHKERRQ(MatMult(pcis->A_BI,pcis->vec1_D,pcis->vec1_B));
    CHKERRQ(VecAXPY(mat_ctx->temp_solution_B,-1.0,pcis->vec1_B));
    CHKERRQ(VecSet(work,0.0));
    CHKERRQ(VecScatterBegin(pcis->global_to_B,mat_ctx->temp_solution_B,work,ADD_VALUES,SCATTER_REVERSE));
    CHKERRQ(VecScatterEnd(pcis->global_to_B,mat_ctx->temp_solution_B,work,ADD_VALUES,SCATTER_REVERSE));
    /* CHKERRQ(PCBDDCScalingRestriction(mat_ctx->pc,work,mat_ctx->temp_solution_B)); */
    CHKERRQ(VecScatterBegin(pcis->global_to_B,work,mat_ctx->temp_solution_B,INSERT_VALUES,SCATTER_FORWARD));
    CHKERRQ(VecScatterEnd(pcis->global_to_B,work,mat_ctx->temp_solution_B,INSERT_VALUES,SCATTER_FORWARD));
    CHKERRQ(VecPointwiseMult(mat_ctx->temp_solution_B,pcis->D,mat_ctx->temp_solution_B));
  }
  /* BDDC rhs */
  CHKERRQ(VecCopy(mat_ctx->temp_solution_B,pcis->vec1_B));
  if (pcbddc->switch_static) {
    CHKERRQ(VecCopy(mat_ctx->temp_solution_D,pcis->vec1_D));
  }
  /* apply BDDC */
  CHKERRQ(PetscArrayzero(pcbddc->benign_p0,pcbddc->benign_n));
  CHKERRQ(PCBDDCApplyInterfacePreconditioner(mat_ctx->pc,PETSC_FALSE));
  CHKERRQ(PetscArrayzero(pcbddc->benign_p0,pcbddc->benign_n));

  /* Application of B_delta and assembling of rhs for fetidp fluxes */
  CHKERRQ(MatMult(mat_ctx->B_delta,pcis->vec1_B,mat_ctx->lambda_local));
  CHKERRQ(VecScatterBegin(mat_ctx->l2g_lambda,mat_ctx->lambda_local,fetidp_flux_rhs,ADD_VALUES,SCATTER_FORWARD));
  CHKERRQ(VecScatterEnd(mat_ctx->l2g_lambda,mat_ctx->lambda_local,fetidp_flux_rhs,ADD_VALUES,SCATTER_FORWARD));
  /* Add contribution to interface pressures */
  if (mat_ctx->l2g_p) {
    CHKERRQ(MatMult(mat_ctx->B_BB,pcis->vec1_B,mat_ctx->vP));
    if (pcbddc->switch_static) {
      CHKERRQ(MatMultAdd(mat_ctx->B_BI,pcis->vec1_D,mat_ctx->vP,mat_ctx->vP));
    }
    CHKERRQ(VecScatterBegin(mat_ctx->l2g_p,mat_ctx->vP,fetidp_flux_rhs,ADD_VALUES,SCATTER_FORWARD));
    CHKERRQ(VecScatterEnd(mat_ctx->l2g_p,mat_ctx->vP,fetidp_flux_rhs,ADD_VALUES,SCATTER_FORWARD));
  }
  PetscFunctionReturn(0);
}

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

  PetscFunctionBegin;
  PetscValidHeaderSpecific(fetidp_mat,MAT_CLASSID,1);
  PetscValidHeaderSpecific(standard_rhs,VEC_CLASSID,2);
  PetscValidHeaderSpecific(fetidp_flux_rhs,VEC_CLASSID,3);
  CHKERRQ(MatShellGetContext(fetidp_mat,&mat_ctx));
  CHKERRQ(PetscUseMethod(mat_ctx->pc,"PCBDDCMatFETIDPGetRHS_C",(Mat,Vec,Vec),(fetidp_mat,standard_rhs,fetidp_flux_rhs)));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCBDDCMatFETIDPGetSolution_BDDC(Mat fetidp_mat, Vec fetidp_flux_sol, Vec standard_sol)
{
  FETIDPMat_ctx  mat_ctx;
  PC_IS*         pcis;
  PC_BDDC*       pcbddc;
  Vec            work;

  PetscFunctionBegin;
  CHKERRQ(MatShellGetContext(fetidp_mat,&mat_ctx));
  pcis = (PC_IS*)mat_ctx->pc->data;
  pcbddc = (PC_BDDC*)mat_ctx->pc->data;

  /* apply B_delta^T */
  CHKERRQ(VecSet(pcis->vec1_B,0.));
  CHKERRQ(VecScatterBegin(mat_ctx->l2g_lambda,fetidp_flux_sol,mat_ctx->lambda_local,INSERT_VALUES,SCATTER_REVERSE));
  CHKERRQ(VecScatterEnd(mat_ctx->l2g_lambda,fetidp_flux_sol,mat_ctx->lambda_local,INSERT_VALUES,SCATTER_REVERSE));
  CHKERRQ(MatMultTranspose(mat_ctx->B_delta,mat_ctx->lambda_local,pcis->vec1_B));
  if (mat_ctx->l2g_p) {
    CHKERRQ(VecScatterBegin(mat_ctx->l2g_p,fetidp_flux_sol,mat_ctx->vP,INSERT_VALUES,SCATTER_REVERSE));
    CHKERRQ(VecScatterEnd(mat_ctx->l2g_p,fetidp_flux_sol,mat_ctx->vP,INSERT_VALUES,SCATTER_REVERSE));
    CHKERRQ(MatMultAdd(mat_ctx->Bt_BB,mat_ctx->vP,pcis->vec1_B,pcis->vec1_B));
  }

  /* compute rhs for BDDC application */
  CHKERRQ(VecAYPX(pcis->vec1_B,-1.0,mat_ctx->temp_solution_B));
  if (pcbddc->switch_static) {
    CHKERRQ(VecCopy(mat_ctx->temp_solution_D,pcis->vec1_D));
    if (mat_ctx->l2g_p) {
      CHKERRQ(VecScale(mat_ctx->vP,-1.));
      CHKERRQ(MatMultAdd(mat_ctx->Bt_BI,mat_ctx->vP,pcis->vec1_D,pcis->vec1_D));
    }
  }

  /* apply BDDC */
  CHKERRQ(PetscArrayzero(pcbddc->benign_p0,pcbddc->benign_n));
  CHKERRQ(PCBDDCApplyInterfacePreconditioner(mat_ctx->pc,PETSC_FALSE));

  /* put values into global vector */
  if (pcbddc->ChangeOfBasisMatrix) work = pcbddc->work_change;
  else work = standard_sol;
  CHKERRQ(VecScatterBegin(pcis->global_to_B,pcis->vec1_B,work,INSERT_VALUES,SCATTER_REVERSE));
  CHKERRQ(VecScatterEnd(pcis->global_to_B,pcis->vec1_B,work,INSERT_VALUES,SCATTER_REVERSE));
  if (!pcbddc->switch_static) {
    /* compute values into the interior if solved for the partially subassembled Schur complement */
    CHKERRQ(MatMult(pcis->A_IB,pcis->vec1_B,pcis->vec1_D));
    CHKERRQ(VecAYPX(pcis->vec1_D,-1.0,mat_ctx->temp_solution_D));
    CHKERRQ(PetscLogEventBegin(PC_BDDC_Solves[pcbddc->current_level][0],mat_ctx->pc,0,0,0));
    CHKERRQ(KSPSolve(pcbddc->ksp_D,pcis->vec1_D,pcis->vec1_D));
    CHKERRQ(PetscLogEventEnd(PC_BDDC_Solves[pcbddc->current_level][0],mat_ctx->pc,0,0,0));
    /* Cannot propagate up error in KSPSolve() because there is no access to the PC */
  }

  CHKERRQ(VecScatterBegin(pcis->global_to_D,pcis->vec1_D,work,INSERT_VALUES,SCATTER_REVERSE));
  CHKERRQ(VecScatterEnd(pcis->global_to_D,pcis->vec1_D,work,INSERT_VALUES,SCATTER_REVERSE));
  /* add p0 solution to final solution */
  CHKERRQ(PCBDDCBenignGetOrSetP0(mat_ctx->pc,work,PETSC_FALSE));
  if (pcbddc->ChangeOfBasisMatrix) {
    CHKERRQ(MatMult(pcbddc->ChangeOfBasisMatrix,work,standard_sol));
  }
  CHKERRQ(PCPostSolve_BDDC(mat_ctx->pc,NULL,NULL,standard_sol));
  if (mat_ctx->g2g_p) {
    CHKERRQ(VecScatterBegin(mat_ctx->g2g_p,fetidp_flux_sol,standard_sol,INSERT_VALUES,SCATTER_REVERSE));
    CHKERRQ(VecScatterEnd(mat_ctx->g2g_p,fetidp_flux_sol,standard_sol,INSERT_VALUES,SCATTER_REVERSE));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PCView_BDDCIPC(PC pc, PetscViewer viewer)
{
  BDDCIPC_ctx    bddcipc_ctx;
  PetscBool      isascii;

  PetscFunctionBegin;
  CHKERRQ(PCShellGetContext(pc,&bddcipc_ctx));
  CHKERRQ(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&isascii));
  if (isascii) {
    CHKERRQ(PetscViewerASCIIPrintf(viewer,"BDDC interface preconditioner\n"));
  }
  CHKERRQ(PetscViewerASCIIPushTab(viewer));
  CHKERRQ(PCView(bddcipc_ctx->bddc,viewer));
  CHKERRQ(PetscViewerASCIIPopTab(viewer));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCSetUp_BDDCIPC(PC pc)
{
  BDDCIPC_ctx    bddcipc_ctx;
  PetscBool      isbddc;
  Vec            vv;
  IS             is;
  PC_IS          *pcis;

  PetscFunctionBegin;
  CHKERRQ(PCShellGetContext(pc,&bddcipc_ctx));
  CHKERRQ(PetscObjectTypeCompare((PetscObject)bddcipc_ctx->bddc,PCBDDC,&isbddc));
  PetscCheckFalse(!isbddc,PetscObjectComm((PetscObject)pc),PETSC_ERR_SUP,"Invalid type %s. Must be of type bddc",((PetscObject)bddcipc_ctx->bddc)->type_name);
  CHKERRQ(PCSetUp(bddcipc_ctx->bddc));

  /* create interface scatter */
  pcis = (PC_IS*)(bddcipc_ctx->bddc->data);
  CHKERRQ(VecScatterDestroy(&bddcipc_ctx->g2l));
  CHKERRQ(MatCreateVecs(pc->pmat,&vv,NULL));
  CHKERRQ(ISRenumber(pcis->is_B_global,NULL,NULL,&is));
  CHKERRQ(VecScatterCreate(vv,is,pcis->vec1_B,NULL,&bddcipc_ctx->g2l));
  CHKERRQ(ISDestroy(&is));
  CHKERRQ(VecDestroy(&vv));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCApply_BDDCIPC(PC pc, Vec r, Vec x)
{
  BDDCIPC_ctx    bddcipc_ctx;
  PC_IS          *pcis;
  VecScatter     tmps;

  PetscFunctionBegin;
  CHKERRQ(PCShellGetContext(pc,&bddcipc_ctx));
  pcis = (PC_IS*)(bddcipc_ctx->bddc->data);
  tmps = pcis->global_to_B;
  pcis->global_to_B = bddcipc_ctx->g2l;
  CHKERRQ(PCBDDCScalingRestriction(bddcipc_ctx->bddc,r,pcis->vec1_B));
  CHKERRQ(PCBDDCApplyInterfacePreconditioner(bddcipc_ctx->bddc,PETSC_FALSE));
  CHKERRQ(PCBDDCScalingExtension(bddcipc_ctx->bddc,pcis->vec1_B,x));
  pcis->global_to_B = tmps;
  PetscFunctionReturn(0);
}

static PetscErrorCode PCApplyTranspose_BDDCIPC(PC pc, Vec r, Vec x)
{
  BDDCIPC_ctx    bddcipc_ctx;
  PC_IS          *pcis;
  VecScatter     tmps;

  PetscFunctionBegin;
  CHKERRQ(PCShellGetContext(pc,&bddcipc_ctx));
  pcis = (PC_IS*)(bddcipc_ctx->bddc->data);
  tmps = pcis->global_to_B;
  pcis->global_to_B = bddcipc_ctx->g2l;
  CHKERRQ(PCBDDCScalingRestriction(bddcipc_ctx->bddc,r,pcis->vec1_B));
  CHKERRQ(PCBDDCApplyInterfacePreconditioner(bddcipc_ctx->bddc,PETSC_TRUE));
  CHKERRQ(PCBDDCScalingExtension(bddcipc_ctx->bddc,pcis->vec1_B,x));
  pcis->global_to_B = tmps;
  PetscFunctionReturn(0);
}

static PetscErrorCode PCDestroy_BDDCIPC(PC pc)
{
  BDDCIPC_ctx    bddcipc_ctx;

  PetscFunctionBegin;
  CHKERRQ(PCShellGetContext(pc,&bddcipc_ctx));
  CHKERRQ(PCDestroy(&bddcipc_ctx->bddc));
  CHKERRQ(VecScatterDestroy(&bddcipc_ctx->g2l));
  CHKERRQ(PetscFree(bddcipc_ctx));
  PetscFunctionReturn(0);
}

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

  PetscFunctionBegin;
  PetscValidHeaderSpecific(fetidp_mat,MAT_CLASSID,1);
  PetscValidHeaderSpecific(fetidp_flux_sol,VEC_CLASSID,2);
  PetscValidHeaderSpecific(standard_sol,VEC_CLASSID,3);
  CHKERRQ(MatShellGetContext(fetidp_mat,&mat_ctx));
  CHKERRQ(PetscUseMethod(mat_ctx->pc,"PCBDDCMatFETIDPGetSolution_C",(Mat,Vec,Vec),(fetidp_mat,fetidp_flux_sol,standard_sol)));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCBDDCCreateFETIDPOperators_BDDC(PC pc, PetscBool fully_redundant, const char* prefix, Mat *fetidp_mat, PC *fetidp_pc)
{

  FETIDPMat_ctx  fetidpmat_ctx;
  Mat            newmat;
  FETIDPPC_ctx   fetidppc_ctx;
  PC             newpc;
  MPI_Comm       comm;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectGetComm((PetscObject)pc,&comm));
  /* FETI-DP matrix */
  CHKERRQ(PCBDDCCreateFETIDPMatContext(pc,&fetidpmat_ctx));
  fetidpmat_ctx->fully_redundant = fully_redundant;
  CHKERRQ(PCBDDCSetupFETIDPMatContext(fetidpmat_ctx));
  CHKERRQ(MatCreateShell(comm,fetidpmat_ctx->n,fetidpmat_ctx->n,fetidpmat_ctx->N,fetidpmat_ctx->N,fetidpmat_ctx,&newmat));
  CHKERRQ(PetscObjectSetName((PetscObject)newmat,!fetidpmat_ctx->l2g_lambda_only ? "F" : "G"));
  CHKERRQ(MatShellSetOperation(newmat,MATOP_MULT,(void (*)(void))FETIDPMatMult));
  CHKERRQ(MatShellSetOperation(newmat,MATOP_MULT_TRANSPOSE,(void (*)(void))FETIDPMatMultTranspose));
  CHKERRQ(MatShellSetOperation(newmat,MATOP_DESTROY,(void (*)(void))PCBDDCDestroyFETIDPMat));
  /* propagate MatOptions */
  {
    PC_BDDC   *pcbddc = (PC_BDDC*)fetidpmat_ctx->pc->data;
    PetscBool issym;

    CHKERRQ(MatGetOption(pc->mat,MAT_SYMMETRIC,&issym));
    if (issym || pcbddc->symmetric_primal) {
      CHKERRQ(MatSetOption(newmat,MAT_SYMMETRIC,PETSC_TRUE));
    }
  }
  CHKERRQ(MatSetOptionsPrefix(newmat,prefix));
  CHKERRQ(MatAppendOptionsPrefix(newmat,"fetidp_"));
  CHKERRQ(MatSetUp(newmat));
  /* FETI-DP preconditioner */
  CHKERRQ(PCBDDCCreateFETIDPPCContext(pc,&fetidppc_ctx));
  CHKERRQ(PCBDDCSetupFETIDPPCContext(newmat,fetidppc_ctx));
  CHKERRQ(PCCreate(comm,&newpc));
  CHKERRQ(PCSetOperators(newpc,newmat,newmat));
  CHKERRQ(PCSetOptionsPrefix(newpc,prefix));
  CHKERRQ(PCAppendOptionsPrefix(newpc,"fetidp_"));
  CHKERRQ(PCSetErrorIfFailure(newpc,pc->erroriffailure));
  if (!fetidpmat_ctx->l2g_lambda_only) { /* standard FETI-DP */
    CHKERRQ(PCSetType(newpc,PCSHELL));
    CHKERRQ(PCShellSetName(newpc,"FETI-DP multipliers"));
    CHKERRQ(PCShellSetContext(newpc,fetidppc_ctx));
    CHKERRQ(PCShellSetApply(newpc,FETIDPPCApply));
    CHKERRQ(PCShellSetApplyTranspose(newpc,FETIDPPCApplyTranspose));
    CHKERRQ(PCShellSetView(newpc,FETIDPPCView));
    CHKERRQ(PCShellSetDestroy(newpc,PCBDDCDestroyFETIDPPC));
  } else { /* saddle-point FETI-DP */
    Mat       M;
    PetscInt  psize;
    PetscBool fake = PETSC_FALSE, isfieldsplit;

    CHKERRQ(ISViewFromOptions(fetidpmat_ctx->lagrange,NULL,"-lag_view"));
    CHKERRQ(ISViewFromOptions(fetidpmat_ctx->pressure,NULL,"-press_view"));
    CHKERRQ(PetscObjectQuery((PetscObject)pc,"__KSPFETIDP_PPmat",(PetscObject*)&M));
    CHKERRQ(PCSetType(newpc,PCFIELDSPLIT));
    CHKERRQ(PCFieldSplitSetIS(newpc,"lag",fetidpmat_ctx->lagrange));
    CHKERRQ(PCFieldSplitSetIS(newpc,"p",fetidpmat_ctx->pressure));
    CHKERRQ(PCFieldSplitSetType(newpc,PC_COMPOSITE_SCHUR));
    CHKERRQ(PCFieldSplitSetSchurFactType(newpc,PC_FIELDSPLIT_SCHUR_FACT_DIAG));
    CHKERRQ(ISGetSize(fetidpmat_ctx->pressure,&psize));
    if (psize != M->rmap->N) {
      Mat      M2;
      PetscInt lpsize;

      fake = PETSC_TRUE;
      CHKERRQ(ISGetLocalSize(fetidpmat_ctx->pressure,&lpsize));
      CHKERRQ(MatCreate(comm,&M2));
      CHKERRQ(MatSetType(M2,MATAIJ));
      CHKERRQ(MatSetSizes(M2,lpsize,lpsize,psize,psize));
      CHKERRQ(MatSetUp(M2));
      CHKERRQ(MatAssemblyBegin(M2,MAT_FINAL_ASSEMBLY));
      CHKERRQ(MatAssemblyEnd(M2,MAT_FINAL_ASSEMBLY));
      CHKERRQ(PCFieldSplitSetSchurPre(newpc,PC_FIELDSPLIT_SCHUR_PRE_USER,M2));
      CHKERRQ(MatDestroy(&M2));
    } else {
      CHKERRQ(PCFieldSplitSetSchurPre(newpc,PC_FIELDSPLIT_SCHUR_PRE_USER,M));
    }
    CHKERRQ(PCFieldSplitSetSchurScale(newpc,1.0));

    /* we need to setfromoptions and setup here to access the blocks */
    CHKERRQ(PCSetFromOptions(newpc));
    CHKERRQ(PCSetUp(newpc));

    /* user may have changed the type (e.g. -fetidp_pc_type none) */
    CHKERRQ(PetscObjectTypeCompare((PetscObject)newpc,PCFIELDSPLIT,&isfieldsplit));
    if (isfieldsplit) {
      KSP       *ksps;
      PC        ppc,lagpc;
      PetscInt  nn;
      PetscBool ismatis,matisok = PETSC_FALSE,check = PETSC_FALSE;

      /* set the solver for the (0,0) block */
      CHKERRQ(PCFieldSplitSchurGetSubKSP(newpc,&nn,&ksps));
      if (!nn) { /* not of type PC_COMPOSITE_SCHUR */
        CHKERRQ(PCFieldSplitGetSubKSP(newpc,&nn,&ksps));
        if (!fake) { /* pass pmat to the pressure solver */
          Mat F;

          CHKERRQ(KSPGetOperators(ksps[1],&F,NULL));
          CHKERRQ(KSPSetOperators(ksps[1],F,M));
        }
      } else {
        PetscBool issym;
        Mat       S;

        CHKERRQ(PCFieldSplitSchurGetS(newpc,&S));

        CHKERRQ(MatGetOption(newmat,MAT_SYMMETRIC,&issym));
        if (issym) {
          CHKERRQ(MatSetOption(S,MAT_SYMMETRIC,PETSC_TRUE));
        }
      }
      CHKERRQ(KSPGetPC(ksps[0],&lagpc));
      CHKERRQ(PCSetType(lagpc,PCSHELL));
      CHKERRQ(PCShellSetName(lagpc,"FETI-DP multipliers"));
      CHKERRQ(PCShellSetContext(lagpc,fetidppc_ctx));
      CHKERRQ(PCShellSetApply(lagpc,FETIDPPCApply));
      CHKERRQ(PCShellSetApplyTranspose(lagpc,FETIDPPCApplyTranspose));
      CHKERRQ(PCShellSetView(lagpc,FETIDPPCView));
      CHKERRQ(PCShellSetDestroy(lagpc,PCBDDCDestroyFETIDPPC));

      /* Olof's idea: interface Schur complement preconditioner for the mass matrix */
      CHKERRQ(KSPGetPC(ksps[1],&ppc));
      if (fake) {
        BDDCIPC_ctx    bddcipc_ctx;
        PetscContainer c;

        matisok = PETSC_TRUE;

        /* create inner BDDC solver */
        CHKERRQ(PetscNew(&bddcipc_ctx));
        CHKERRQ(PCCreate(comm,&bddcipc_ctx->bddc));
        CHKERRQ(PCSetType(bddcipc_ctx->bddc,PCBDDC));
        CHKERRQ(PCSetOperators(bddcipc_ctx->bddc,M,M));
        CHKERRQ(PetscObjectQuery((PetscObject)pc,"__KSPFETIDP_pCSR",(PetscObject*)&c));
        CHKERRQ(PetscObjectTypeCompare((PetscObject)M,MATIS,&ismatis));
        if (c && ismatis) {
          Mat      lM;
          PetscInt *csr,n;

          CHKERRQ(MatISGetLocalMat(M,&lM));
          CHKERRQ(MatGetSize(lM,&n,NULL));
          CHKERRQ(PetscContainerGetPointer(c,(void**)&csr));
          CHKERRQ(PCBDDCSetLocalAdjacencyGraph(bddcipc_ctx->bddc,n,csr,csr + (n + 1),PETSC_COPY_VALUES));
          CHKERRQ(MatISRestoreLocalMat(M,&lM));
        }
        CHKERRQ(PCSetOptionsPrefix(bddcipc_ctx->bddc,((PetscObject)ksps[1])->prefix));
        CHKERRQ(PCSetErrorIfFailure(bddcipc_ctx->bddc,pc->erroriffailure));
        CHKERRQ(PCSetFromOptions(bddcipc_ctx->bddc));

        /* wrap the interface application */
        CHKERRQ(PCSetType(ppc,PCSHELL));
        CHKERRQ(PCShellSetName(ppc,"FETI-DP pressure"));
        CHKERRQ(PCShellSetContext(ppc,bddcipc_ctx));
        CHKERRQ(PCShellSetSetUp(ppc,PCSetUp_BDDCIPC));
        CHKERRQ(PCShellSetApply(ppc,PCApply_BDDCIPC));
        CHKERRQ(PCShellSetApplyTranspose(ppc,PCApplyTranspose_BDDCIPC));
        CHKERRQ(PCShellSetView(ppc,PCView_BDDCIPC));
        CHKERRQ(PCShellSetDestroy(ppc,PCDestroy_BDDCIPC));
      }

      /* determine if we need to assemble M to construct a preconditioner */
      if (!matisok) {
        CHKERRQ(PetscObjectTypeCompare((PetscObject)M,MATIS,&ismatis));
        CHKERRQ(PetscObjectTypeCompareAny((PetscObject)ppc,&matisok,PCBDDC,PCJACOBI,PCNONE,PCMG,""));
        if (ismatis && !matisok) {
          CHKERRQ(MatConvert(M,MATAIJ,MAT_INPLACE_MATRIX,&M));
        }
      }

      /* run the subproblems to check convergence */
      CHKERRQ(PetscOptionsGetBool(NULL,((PetscObject)newmat)->prefix,"-check_saddlepoint",&check,NULL));
      if (check) {
        PetscInt i;

        for (i=0;i<nn;i++) {
          KSP       kspC;
          PC        pc;
          Mat       F,pF;
          Vec       x,y;
          PetscBool isschur,prec = PETSC_TRUE;

          CHKERRQ(KSPCreate(PetscObjectComm((PetscObject)ksps[i]),&kspC));
          CHKERRQ(KSPSetOptionsPrefix(kspC,((PetscObject)ksps[i])->prefix));
          CHKERRQ(KSPAppendOptionsPrefix(kspC,"check_"));
          CHKERRQ(KSPGetOperators(ksps[i],&F,&pF));
          CHKERRQ(PetscObjectTypeCompare((PetscObject)F,MATSCHURCOMPLEMENT,&isschur));
          if (isschur) {
            KSP  kspS,kspS2;
            Mat  A00,pA00,A10,A01,A11;
            char prefix[256];

            CHKERRQ(MatSchurComplementGetKSP(F,&kspS));
            CHKERRQ(MatSchurComplementGetSubMatrices(F,&A00,&pA00,&A01,&A10,&A11));
            CHKERRQ(MatCreateSchurComplement(A00,pA00,A01,A10,A11,&F));
            CHKERRQ(MatSchurComplementGetKSP(F,&kspS2));
            CHKERRQ(PetscSNPrintf(prefix,sizeof(prefix),"%sschur_",((PetscObject)kspC)->prefix));
            CHKERRQ(KSPSetOptionsPrefix(kspS2,prefix));
            CHKERRQ(KSPGetPC(kspS2,&pc));
            CHKERRQ(PCSetType(pc,PCKSP));
            CHKERRQ(PCKSPSetKSP(pc,kspS));
            CHKERRQ(KSPSetFromOptions(kspS2));
            CHKERRQ(KSPGetPC(kspS2,&pc));
            CHKERRQ(PCSetUseAmat(pc,PETSC_TRUE));
          } else {
            CHKERRQ(PetscObjectReference((PetscObject)F));
          }
          CHKERRQ(KSPSetFromOptions(kspC));
          CHKERRQ(PetscOptionsGetBool(NULL,((PetscObject)kspC)->prefix,"-preconditioned",&prec,NULL));
          if (prec)  {
            CHKERRQ(KSPGetPC(ksps[i],&pc));
            CHKERRQ(KSPSetPC(kspC,pc));
          }
          CHKERRQ(KSPSetOperators(kspC,F,pF));
          CHKERRQ(MatCreateVecs(F,&x,&y));
          CHKERRQ(VecSetRandom(x,NULL));
          CHKERRQ(MatMult(F,x,y));
          CHKERRQ(KSPSolve(kspC,y,x));
          CHKERRQ(KSPCheckSolve(kspC,pc,x));
          CHKERRQ(KSPDestroy(&kspC));
          CHKERRQ(MatDestroy(&F));
          CHKERRQ(VecDestroy(&x));
          CHKERRQ(VecDestroy(&y));
        }
      }
      CHKERRQ(PetscFree(ksps));
    }
  }
  /* return pointers for objects created */
  *fetidp_mat = newmat;
  *fetidp_pc  = newpc;
  PetscFunctionReturn(0);
}

/*@C
 PCBDDCCreateFETIDPOperators - Create FETI-DP operators

   Collective

   Input Parameters:
+  pc - the BDDC preconditioning context (setup should have been called before)
.  fully_redundant - true for a fully redundant set of Lagrange multipliers
-  prefix - optional options database prefix for the objects to be created (can be NULL)

   Output Parameters:
+  fetidp_mat - shell FETI-DP matrix object
-  fetidp_pc  - shell Dirichlet preconditioner for FETI-DP matrix

   Level: developer

   Notes:
     Currently the only operations provided for FETI-DP matrix are MatMult and MatMultTranspose

.seealso: PCBDDC, PCBDDCMatFETIDPGetRHS, PCBDDCMatFETIDPGetSolution
@*/
PetscErrorCode PCBDDCCreateFETIDPOperators(PC pc, PetscBool fully_redundant, const char *prefix, Mat *fetidp_mat, PC *fetidp_pc)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  if (pc->setupcalled) {
    CHKERRQ(PetscUseMethod(pc,"PCBDDCCreateFETIDPOperators_C",(PC,PetscBool,const char*,Mat*,PC*),(pc,fully_redundant,prefix,fetidp_mat,fetidp_pc)));
  } else SETERRQ(PetscObjectComm((PetscObject)pc),PETSC_ERR_SUP,"You must call PCSetup_BDDC() first");
  PetscFunctionReturn(0);
}
/* -------------------------------------------------------------------------- */
/*MC
   PCBDDC - Balancing Domain Decomposition by Constraints.

   An implementation of the BDDC preconditioner based on the bibliography found below.

   The matrix to be preconditioned (Pmat) must be of type MATIS.

   Currently works with MATIS matrices with local matrices of type MATSEQAIJ, MATSEQBAIJ or MATSEQSBAIJ, either with real or complex numbers.

   It also works with unsymmetric and indefinite problems.

   Unlike 'conventional' interface preconditioners, PCBDDC iterates over all degrees of freedom, not just those on the interface. This allows the use of approximate solvers on the subdomains.

   Approximate local solvers are automatically adapted (see [1]) if the user has attached a nullspace object to the subdomain matrices, and informed BDDC of using approximate solvers (via the command line).

   Boundary nodes are split in vertices, edges and faces classes using information from the local to global mapping of dofs and the local connectivity graph of nodes. The latter can be customized by using PCBDDCSetLocalAdjacencyGraph()
   Additional information on dofs can be provided by using PCBDDCSetDofsSplitting(), PCBDDCSetDirichletBoundaries(), PCBDDCSetNeumannBoundaries(), and PCBDDCSetPrimalVerticesIS() and their local counterparts.

   Constraints can be customized by attaching a MatNullSpace object to the MATIS matrix via MatSetNearNullSpace(). Non-singular modes are retained via SVD.

   Change of basis is performed similarly to [2] when requested. When more than one constraint is present on a single connected component (i.e. an edge or a face), a robust method based on local QR factorizations is used.
   User defined change of basis can be passed to PCBDDC by using PCBDDCSetChangeOfBasisMat()

   The PETSc implementation also supports multilevel BDDC [3]. Coarse grids are partitioned using a MatPartitioning object.

   Adaptive selection of primal constraints [4] is supported for SPD systems with high-contrast in the coefficients if MUMPS or MKL_PARDISO are present. Future versions of the code will also consider using PASTIX.

   An experimental interface to the FETI-DP method is available. FETI-DP operators could be created using PCBDDCCreateFETIDPOperators(). A stand-alone class for the FETI-DP method will be provided in the next releases.

   Options Database Keys (some of them, run with -help for a complete list):

+    -pc_bddc_use_vertices <true> - use or not vertices in primal space
.    -pc_bddc_use_edges <true> - use or not edges in primal space
.    -pc_bddc_use_faces <false> - use or not faces in primal space
.    -pc_bddc_symmetric <true> - symmetric computation of primal basis functions. Specify false for unsymmetric problems
.    -pc_bddc_use_change_of_basis <false> - use change of basis approach (on edges only)
.    -pc_bddc_use_change_on_faces <false> - use change of basis approach on faces if change of basis has been requested
.    -pc_bddc_switch_static <false> - switches from M_2 (default) to M_3 operator (see reference article [1])
.    -pc_bddc_levels <0> - maximum number of levels for multilevel
.    -pc_bddc_coarsening_ratio <8> - number of subdomains which will be aggregated together at the coarser level (e.g. H/h ratio at the coarser level, significative only in the multilevel case)
.    -pc_bddc_coarse_redistribute <0> - size of a subset of processors where the coarse problem will be remapped (the value is ignored if not at the coarsest level)
.    -pc_bddc_use_deluxe_scaling <false> - use deluxe scaling
.    -pc_bddc_schur_layers <\-1> - select the economic version of deluxe scaling by specifying the number of layers (-1 corresponds to the original deluxe scaling)
.    -pc_bddc_adaptive_threshold <0.0> - when a value different than zero is specified, adaptive selection of constraints is performed on edges and faces (requires deluxe scaling and MUMPS or MKL_PARDISO installed)
-    -pc_bddc_check_level <0> - set verbosity level of debugging output

   Options for Dirichlet, Neumann or coarse solver can be set with
.vb
      -pc_bddc_dirichlet_
      -pc_bddc_neumann_
      -pc_bddc_coarse_
.ve
   e.g. -pc_bddc_dirichlet_ksp_type richardson -pc_bddc_dirichlet_pc_type gamg. PCBDDC uses by default KSPPREONLY and PCLU.

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

   References:
+  * - C. R. Dohrmann. "An approximate BDDC preconditioner", Numerical Linear Algebra with Applications Volume 14, Issue 2, pages 149--168, March 2007
.  * - A. Klawonn and O. B. Widlund. "Dual-Primal FETI Methods for Linear Elasticity", Communications on Pure and Applied Mathematics Volume 59, Issue 11, pages 1523--1572, November 2006
.  * - J. Mandel, B. Sousedik, C. R. Dohrmann. "Multispace and Multilevel BDDC", Computing Volume 83, Issue 2--3, pages 55--85, November 2008
-  * - C. Pechstein and C. R. Dohrmann. "Modern domain decomposition methods BDDC, deluxe scaling, and an algebraic approach", Seminar talk, Linz, December 2013, http://people.ricam.oeaw.ac.at/c.pechstein/pechstein-bddc2013.pdf

   Level: intermediate

   Developer Notes:

   Contributed by Stefano Zampini

.seealso:  PCCreate(), PCSetType(), PCType (for list of available types), PC,  MATIS
M*/

PETSC_EXTERN PetscErrorCode PCCreate_BDDC(PC pc)
{
  PC_BDDC             *pcbddc;

  PetscFunctionBegin;
  CHKERRQ(PetscNewLog(pc,&pcbddc));
  pc->data = pcbddc;

  /* create PCIS data structure */
  CHKERRQ(PCISCreate(pc));

  /* create local graph structure */
  CHKERRQ(PCBDDCGraphCreate(&pcbddc->mat_graph));

  /* BDDC nonzero defaults */
  pcbddc->use_nnsp                  = PETSC_TRUE;
  pcbddc->use_local_adj             = PETSC_TRUE;
  pcbddc->use_vertices              = PETSC_TRUE;
  pcbddc->use_edges                 = PETSC_TRUE;
  pcbddc->symmetric_primal          = PETSC_TRUE;
  pcbddc->vertex_size               = 1;
  pcbddc->recompute_topography      = PETSC_TRUE;
  pcbddc->coarse_size               = -1;
  pcbddc->use_exact_dirichlet_trick = PETSC_TRUE;
  pcbddc->coarsening_ratio          = 8;
  pcbddc->coarse_eqs_per_proc       = 1;
  pcbddc->benign_compute_correction = PETSC_TRUE;
  pcbddc->nedfield                  = -1;
  pcbddc->nedglobal                 = PETSC_TRUE;
  pcbddc->graphmaxcount             = PETSC_MAX_INT;
  pcbddc->sub_schurs_layers         = -1;
  pcbddc->adaptive_threshold[0]     = 0.0;
  pcbddc->adaptive_threshold[1]     = 0.0;

  /* function pointers */
  pc->ops->apply               = PCApply_BDDC;
  pc->ops->applytranspose      = PCApplyTranspose_BDDC;
  pc->ops->setup               = PCSetUp_BDDC;
  pc->ops->destroy             = PCDestroy_BDDC;
  pc->ops->setfromoptions      = PCSetFromOptions_BDDC;
  pc->ops->view                = PCView_BDDC;
  pc->ops->applyrichardson     = NULL;
  pc->ops->applysymmetricleft  = NULL;
  pc->ops->applysymmetricright = NULL;
  pc->ops->presolve            = PCPreSolve_BDDC;
  pc->ops->postsolve           = PCPostSolve_BDDC;
  pc->ops->reset               = PCReset_BDDC;

  /* composing function */
  CHKERRQ(PetscObjectComposeFunction((PetscObject)pc,"PCBDDCSetDiscreteGradient_C",PCBDDCSetDiscreteGradient_BDDC));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)pc,"PCBDDCSetDivergenceMat_C",PCBDDCSetDivergenceMat_BDDC));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)pc,"PCBDDCSetChangeOfBasisMat_C",PCBDDCSetChangeOfBasisMat_BDDC));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)pc,"PCBDDCSetPrimalVerticesLocalIS_C",PCBDDCSetPrimalVerticesLocalIS_BDDC));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)pc,"PCBDDCSetPrimalVerticesIS_C",PCBDDCSetPrimalVerticesIS_BDDC));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)pc,"PCBDDCGetPrimalVerticesLocalIS_C",PCBDDCGetPrimalVerticesLocalIS_BDDC));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)pc,"PCBDDCGetPrimalVerticesIS_C",PCBDDCGetPrimalVerticesIS_BDDC));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)pc,"PCBDDCSetCoarseningRatio_C",PCBDDCSetCoarseningRatio_BDDC));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)pc,"PCBDDCSetLevel_C",PCBDDCSetLevel_BDDC));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)pc,"PCBDDCSetUseExactDirichlet_C",PCBDDCSetUseExactDirichlet_BDDC));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)pc,"PCBDDCSetLevels_C",PCBDDCSetLevels_BDDC));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)pc,"PCBDDCSetDirichletBoundaries_C",PCBDDCSetDirichletBoundaries_BDDC));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)pc,"PCBDDCSetDirichletBoundariesLocal_C",PCBDDCSetDirichletBoundariesLocal_BDDC));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)pc,"PCBDDCSetNeumannBoundaries_C",PCBDDCSetNeumannBoundaries_BDDC));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)pc,"PCBDDCSetNeumannBoundariesLocal_C",PCBDDCSetNeumannBoundariesLocal_BDDC));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)pc,"PCBDDCGetDirichletBoundaries_C",PCBDDCGetDirichletBoundaries_BDDC));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)pc,"PCBDDCGetDirichletBoundariesLocal_C",PCBDDCGetDirichletBoundariesLocal_BDDC));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)pc,"PCBDDCGetNeumannBoundaries_C",PCBDDCGetNeumannBoundaries_BDDC));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)pc,"PCBDDCGetNeumannBoundariesLocal_C",PCBDDCGetNeumannBoundariesLocal_BDDC));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)pc,"PCBDDCSetDofsSplitting_C",PCBDDCSetDofsSplitting_BDDC));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)pc,"PCBDDCSetDofsSplittingLocal_C",PCBDDCSetDofsSplittingLocal_BDDC));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)pc,"PCBDDCSetLocalAdjacencyGraph_C",PCBDDCSetLocalAdjacencyGraph_BDDC));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)pc,"PCBDDCCreateFETIDPOperators_C",PCBDDCCreateFETIDPOperators_BDDC));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)pc,"PCBDDCMatFETIDPGetRHS_C",PCBDDCMatFETIDPGetRHS_BDDC));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)pc,"PCBDDCMatFETIDPGetSolution_C",PCBDDCMatFETIDPGetSolution_BDDC));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)pc,"PCPreSolveChangeRHS_C",PCPreSolveChangeRHS_BDDC));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)pc,"PCSetCoordinates_C",PCSetCoordinates_BDDC));
  PetscFunctionReturn(0);
}

/*@C
 PCBDDCInitializePackage - This function initializes everything in the PCBDDC package. It is called
    from PCInitializePackage().

 Level: developer

 .seealso: PetscInitialize()
@*/
PetscErrorCode PCBDDCInitializePackage(void)
{
  int            i;

  PetscFunctionBegin;
  if (PCBDDCPackageInitialized) PetscFunctionReturn(0);
  PCBDDCPackageInitialized = PETSC_TRUE;
  CHKERRQ(PetscRegisterFinalize(PCBDDCFinalizePackage));

  /* general events */
  CHKERRQ(PetscLogEventRegister("PCBDDCTopo",PC_CLASSID,&PC_BDDC_Topology[0]));
  CHKERRQ(PetscLogEventRegister("PCBDDCLKSP",PC_CLASSID,&PC_BDDC_LocalSolvers[0]));
  CHKERRQ(PetscLogEventRegister("PCBDDCLWor",PC_CLASSID,&PC_BDDC_LocalWork[0]));
  CHKERRQ(PetscLogEventRegister("PCBDDCCorr",PC_CLASSID,&PC_BDDC_CorrectionSetUp[0]));
  CHKERRQ(PetscLogEventRegister("PCBDDCASet",PC_CLASSID,&PC_BDDC_ApproxSetUp[0]));
  CHKERRQ(PetscLogEventRegister("PCBDDCAApp",PC_CLASSID,&PC_BDDC_ApproxApply[0]));
  CHKERRQ(PetscLogEventRegister("PCBDDCCSet",PC_CLASSID,&PC_BDDC_CoarseSetUp[0]));
  CHKERRQ(PetscLogEventRegister("PCBDDCCKSP",PC_CLASSID,&PC_BDDC_CoarseSolver[0]));
  CHKERRQ(PetscLogEventRegister("PCBDDCAdap",PC_CLASSID,&PC_BDDC_AdaptiveSetUp[0]));
  CHKERRQ(PetscLogEventRegister("PCBDDCScal",PC_CLASSID,&PC_BDDC_Scaling[0]));
  CHKERRQ(PetscLogEventRegister("PCBDDCSchr",PC_CLASSID,&PC_BDDC_Schurs[0]));
  CHKERRQ(PetscLogEventRegister("PCBDDCDirS",PC_CLASSID,&PC_BDDC_Solves[0][0]));
  CHKERRQ(PetscLogEventRegister("PCBDDCNeuS",PC_CLASSID,&PC_BDDC_Solves[0][1]));
  CHKERRQ(PetscLogEventRegister("PCBDDCCoaS",PC_CLASSID,&PC_BDDC_Solves[0][2]));
  for (i=1;i<PETSC_PCBDDC_MAXLEVELS;i++) {
    char ename[32];

    CHKERRQ(PetscSNPrintf(ename,sizeof(ename),"PCBDDCTopo l%02d",i));
    CHKERRQ(PetscLogEventRegister(ename,PC_CLASSID,&PC_BDDC_Topology[i]));
    CHKERRQ(PetscSNPrintf(ename,sizeof(ename),"PCBDDCLKSP l%02d",i));
    CHKERRQ(PetscLogEventRegister(ename,PC_CLASSID,&PC_BDDC_LocalSolvers[i]));
    CHKERRQ(PetscSNPrintf(ename,sizeof(ename),"PCBDDCLWor l%02d",i));
    CHKERRQ(PetscLogEventRegister(ename,PC_CLASSID,&PC_BDDC_LocalWork[i]));
    CHKERRQ(PetscSNPrintf(ename,sizeof(ename),"PCBDDCCorr l%02d",i));
    CHKERRQ(PetscLogEventRegister(ename,PC_CLASSID,&PC_BDDC_CorrectionSetUp[i]));
    CHKERRQ(PetscSNPrintf(ename,sizeof(ename),"PCBDDCASet l%02d",i));
    CHKERRQ(PetscLogEventRegister(ename,PC_CLASSID,&PC_BDDC_ApproxSetUp[i]));
    CHKERRQ(PetscSNPrintf(ename,sizeof(ename),"PCBDDCAApp l%02d",i));
    CHKERRQ(PetscLogEventRegister(ename,PC_CLASSID,&PC_BDDC_ApproxApply[i]));
    CHKERRQ(PetscSNPrintf(ename,sizeof(ename),"PCBDDCCSet l%02d",i));
    CHKERRQ(PetscLogEventRegister(ename,PC_CLASSID,&PC_BDDC_CoarseSetUp[i]));
    CHKERRQ(PetscSNPrintf(ename,sizeof(ename),"PCBDDCCKSP l%02d",i));
    CHKERRQ(PetscLogEventRegister(ename,PC_CLASSID,&PC_BDDC_CoarseSolver[i]));
    CHKERRQ(PetscSNPrintf(ename,sizeof(ename),"PCBDDCAdap l%02d",i));
    CHKERRQ(PetscLogEventRegister(ename,PC_CLASSID,&PC_BDDC_AdaptiveSetUp[i]));
    CHKERRQ(PetscSNPrintf(ename,sizeof(ename),"PCBDDCScal l%02d",i));
    CHKERRQ(PetscLogEventRegister(ename,PC_CLASSID,&PC_BDDC_Scaling[i]));
    CHKERRQ(PetscSNPrintf(ename,sizeof(ename),"PCBDDCSchr l%02d",i));
    CHKERRQ(PetscLogEventRegister(ename,PC_CLASSID,&PC_BDDC_Schurs[i]));
    CHKERRQ(PetscSNPrintf(ename,sizeof(ename),"PCBDDCDirS l%02d",i));
    CHKERRQ(PetscLogEventRegister(ename,PC_CLASSID,&PC_BDDC_Solves[i][0]));
    CHKERRQ(PetscSNPrintf(ename,sizeof(ename),"PCBDDCNeuS l%02d",i));
    CHKERRQ(PetscLogEventRegister(ename,PC_CLASSID,&PC_BDDC_Solves[i][1]));
    CHKERRQ(PetscSNPrintf(ename,sizeof(ename),"PCBDDCCoaS l%02d",i));
    CHKERRQ(PetscLogEventRegister(ename,PC_CLASSID,&PC_BDDC_Solves[i][2]));
  }
  PetscFunctionReturn(0);
}

/*@C
 PCBDDCFinalizePackage - This function frees everything from the PCBDDC package. It is
    called from PetscFinalize() automatically.

 Level: developer

 .seealso: PetscFinalize()
@*/
PetscErrorCode PCBDDCFinalizePackage(void)
{
  PetscFunctionBegin;
  PCBDDCPackageInitialized = PETSC_FALSE;
  PetscFunctionReturn(0);
}
