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
  PetscOptionsHeadBegin(PetscOptionsObject,"BDDC options");
  /* Verbose debugging */
  PetscCall(PetscOptionsInt("-pc_bddc_check_level","Verbose output for PCBDDC (intended for debug)","none",pcbddc->dbg_flag,&pcbddc->dbg_flag,NULL));
  /* Approximate solvers */
  PetscCall(PetscOptionsEnum("-pc_bddc_interface_ext_type","Use DIRICHLET or LUMP to extend interface corrections to interior","PCBDDCSetInterfaceExtType",PCBDDCInterfaceExtTypes,(PetscEnum)pcbddc->interface_extension,(PetscEnum*)&pcbddc->interface_extension,NULL));
  if (pcbddc->interface_extension == PC_BDDC_INTERFACE_EXT_DIRICHLET) {
    PetscCall(PetscOptionsBool("-pc_bddc_dirichlet_approximate","Inform PCBDDC that we are using approximate Dirichlet solvers","none",pcbddc->NullSpace_corr[0],&pcbddc->NullSpace_corr[0],NULL));
    PetscCall(PetscOptionsBool("-pc_bddc_dirichlet_approximate_scale","Inform PCBDDC that we need to scale the Dirichlet solve","none",pcbddc->NullSpace_corr[1],&pcbddc->NullSpace_corr[1],NULL));
  } else {
    /* This flag is needed/implied by lumping */
    pcbddc->switch_static = PETSC_TRUE;
  }
  PetscCall(PetscOptionsBool("-pc_bddc_neumann_approximate","Inform PCBDDC that we are using approximate Neumann solvers","none",pcbddc->NullSpace_corr[2],&pcbddc->NullSpace_corr[2],NULL));
  PetscCall(PetscOptionsBool("-pc_bddc_neumann_approximate_scale","Inform PCBDDC that we need to scale the Neumann solve","none",pcbddc->NullSpace_corr[3],&pcbddc->NullSpace_corr[3],NULL));
  /* Primal space customization */
  PetscCall(PetscOptionsBool("-pc_bddc_use_local_mat_graph","Use or not adjacency graph of local mat for interface analysis","none",pcbddc->use_local_adj,&pcbddc->use_local_adj,NULL));
  PetscCall(PetscOptionsInt("-pc_bddc_graph_maxcount","Maximum number of shared subdomains for a connected component","none",pcbddc->graphmaxcount,&pcbddc->graphmaxcount,NULL));
  PetscCall(PetscOptionsBool("-pc_bddc_corner_selection","Activates face-based corner selection","none",pcbddc->corner_selection,&pcbddc->corner_selection,NULL));
  PetscCall(PetscOptionsBool("-pc_bddc_use_vertices","Use or not corner dofs in coarse space","none",pcbddc->use_vertices,&pcbddc->use_vertices,NULL));
  PetscCall(PetscOptionsBool("-pc_bddc_use_edges","Use or not edge constraints in coarse space","none",pcbddc->use_edges,&pcbddc->use_edges,NULL));
  PetscCall(PetscOptionsBool("-pc_bddc_use_faces","Use or not face constraints in coarse space","none",pcbddc->use_faces,&pcbddc->use_faces,NULL));
  PetscCall(PetscOptionsInt("-pc_bddc_vertex_size","Connected components smaller or equal to vertex size will be considered as primal vertices","none",pcbddc->vertex_size,&pcbddc->vertex_size,NULL));
  PetscCall(PetscOptionsBool("-pc_bddc_use_nnsp","Use near null space attached to the matrix to compute constraints","none",pcbddc->use_nnsp,&pcbddc->use_nnsp,NULL));
  PetscCall(PetscOptionsBool("-pc_bddc_use_nnsp_true","Use near null space attached to the matrix to compute constraints as is","none",pcbddc->use_nnsp_true,&pcbddc->use_nnsp_true,NULL));
  PetscCall(PetscOptionsBool("-pc_bddc_use_qr_single","Use QR factorization for single constraints on cc (QR is always used when multiple constraints are present)","none",pcbddc->use_qr_single,&pcbddc->use_qr_single,NULL));
  /* Change of basis */
  PetscCall(PetscOptionsBool("-pc_bddc_use_change_of_basis","Use or not internal change of basis on local edge nodes","none",pcbddc->use_change_of_basis,&pcbddc->use_change_of_basis,NULL));
  PetscCall(PetscOptionsBool("-pc_bddc_use_change_on_faces","Use or not internal change of basis on local face nodes","none",pcbddc->use_change_on_faces,&pcbddc->use_change_on_faces,NULL));
  if (!pcbddc->use_change_of_basis) {
    pcbddc->use_change_on_faces = PETSC_FALSE;
  }
  /* Switch between M_2 (default) and M_3 preconditioners (as defined by C. Dohrmann in the ref. article) */
  PetscCall(PetscOptionsBool("-pc_bddc_switch_static","Switch on static condensation ops around the interface preconditioner","none",pcbddc->switch_static,&pcbddc->switch_static,NULL));
  PetscCall(PetscOptionsInt("-pc_bddc_coarse_eqs_per_proc","Target number of equations per process for coarse problem redistribution (significant only at the coarsest level)","none",pcbddc->coarse_eqs_per_proc,&pcbddc->coarse_eqs_per_proc,NULL));
  i    = pcbddc->coarsening_ratio;
  PetscCall(PetscOptionsInt("-pc_bddc_coarsening_ratio","Set coarsening ratio used in multilevel coarsening","PCBDDCSetCoarseningRatio",i,&i,NULL));
  PetscCall(PCBDDCSetCoarseningRatio(pc,i));
  i    = pcbddc->max_levels;
  PetscCall(PetscOptionsInt("-pc_bddc_levels","Set maximum number of levels for multilevel","PCBDDCSetLevels",i,&i,NULL));
  PetscCall(PCBDDCSetLevels(pc,i));
  PetscCall(PetscOptionsInt("-pc_bddc_coarse_eqs_limit","Set maximum number of equations on coarsest grid to aim for","none",pcbddc->coarse_eqs_limit,&pcbddc->coarse_eqs_limit,NULL));
  PetscCall(PetscOptionsBool("-pc_bddc_use_coarse_estimates","Use estimated eigenvalues for coarse problem","none",pcbddc->use_coarse_estimates,&pcbddc->use_coarse_estimates,NULL));
  PetscCall(PetscOptionsBool("-pc_bddc_use_deluxe_scaling","Use deluxe scaling for BDDC","none",pcbddc->use_deluxe_scaling,&pcbddc->use_deluxe_scaling,NULL));
  PetscCall(PetscOptionsBool("-pc_bddc_schur_rebuild","Whether or not the interface graph for Schur principal minors has to be rebuilt (i.e. define the interface without any adjacency)","none",pcbddc->sub_schurs_rebuild,&pcbddc->sub_schurs_rebuild,NULL));
  PetscCall(PetscOptionsInt("-pc_bddc_schur_layers","Number of dofs' layers for the computation of principal minors (i.e. -1 uses all dofs)","none",pcbddc->sub_schurs_layers,&pcbddc->sub_schurs_layers,NULL));
  PetscCall(PetscOptionsBool("-pc_bddc_schur_use_useradj","Whether or not the CSR graph specified by the user should be used for computing successive layers (default is to use adj of local mat)","none",pcbddc->sub_schurs_use_useradj,&pcbddc->sub_schurs_use_useradj,NULL));
  PetscCall(PetscOptionsBool("-pc_bddc_schur_exact","Whether or not to use the exact Schur complement instead of the reduced one (which excludes size 1 cc)","none",pcbddc->sub_schurs_exact_schur,&pcbddc->sub_schurs_exact_schur,NULL));
  PetscCall(PetscOptionsBool("-pc_bddc_deluxe_zerorows","Zero rows and columns of deluxe operators associated with primal dofs","none",pcbddc->deluxe_zerorows,&pcbddc->deluxe_zerorows,NULL));
  PetscCall(PetscOptionsBool("-pc_bddc_deluxe_singlemat","Collapse deluxe operators","none",pcbddc->deluxe_singlemat,&pcbddc->deluxe_singlemat,NULL));
  PetscCall(PetscOptionsBool("-pc_bddc_adaptive_userdefined","Use user-defined constraints (should be attached via MatSetNearNullSpace to pmat) in addition to those adaptively generated","none",pcbddc->adaptive_userdefined,&pcbddc->adaptive_userdefined,NULL));
  nt   = 2;
  PetscCall(PetscOptionsRealArray("-pc_bddc_adaptive_threshold","Thresholds to be used for adaptive selection of constraints","none",pcbddc->adaptive_threshold,&nt,NULL));
  if (nt == 1) pcbddc->adaptive_threshold[1] = pcbddc->adaptive_threshold[0];
  PetscCall(PetscOptionsInt("-pc_bddc_adaptive_nmin","Minimum number of constraints per connected components","none",pcbddc->adaptive_nmin,&pcbddc->adaptive_nmin,NULL));
  PetscCall(PetscOptionsInt("-pc_bddc_adaptive_nmax","Maximum number of constraints per connected components","none",pcbddc->adaptive_nmax,&pcbddc->adaptive_nmax,NULL));
  PetscCall(PetscOptionsBool("-pc_bddc_symmetric","Symmetric computation of primal basis functions","none",pcbddc->symmetric_primal,&pcbddc->symmetric_primal,NULL));
  PetscCall(PetscOptionsInt("-pc_bddc_coarse_adj","Number of processors where to map the coarse adjacency list","none",pcbddc->coarse_adj_red,&pcbddc->coarse_adj_red,NULL));
  PetscCall(PetscOptionsBool("-pc_bddc_benign_trick","Apply the benign subspace trick to saddle point problems with discontinuous pressures","none",pcbddc->benign_saddle_point,&pcbddc->benign_saddle_point,NULL));
  PetscCall(PetscOptionsBool("-pc_bddc_benign_change","Compute the pressure change of basis explicitly","none",pcbddc->benign_change_explicit,&pcbddc->benign_change_explicit,NULL));
  PetscCall(PetscOptionsBool("-pc_bddc_benign_compute_correction","Compute the benign correction during PreSolve","none",pcbddc->benign_compute_correction,&pcbddc->benign_compute_correction,NULL));
  PetscCall(PetscOptionsBool("-pc_bddc_nonetflux","Automatic computation of no-net-flux quadrature weights","none",pcbddc->compute_nonetflux,&pcbddc->compute_nonetflux,NULL));
  PetscCall(PetscOptionsBool("-pc_bddc_detect_disconnected","Detects disconnected subdomains","none",pcbddc->detect_disconnected,&pcbddc->detect_disconnected,NULL));
  PetscCall(PetscOptionsBool("-pc_bddc_detect_disconnected_filter","Filters out small entries in the local matrix when detecting disconnected subdomains","none",pcbddc->detect_disconnected_filter,&pcbddc->detect_disconnected_filter,NULL));
  PetscCall(PetscOptionsBool("-pc_bddc_eliminate_dirichlet","Whether or not we want to eliminate dirichlet dofs during presolve","none",pcbddc->eliminate_dirdofs,&pcbddc->eliminate_dirdofs,NULL));
  PetscOptionsHeadEnd();
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
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&isascii));
  /* ASCII viewer */
  if (isascii) {
    PetscMPIInt   color,rank,size;
    PetscInt64    loc[7],gsum[6],gmax[6],gmin[6],totbenign;
    PetscScalar   interface_size;
    PetscReal     ratio1=0.,ratio2=0.;
    Vec           counter;

    if (!pc->setupcalled) {
      PetscCall(PetscViewerASCIIPrintf(viewer,"  Partial information available: preconditioner has not been setup yet\n"));
    }
    PetscCall(PetscViewerASCIIPrintf(viewer,"  Use verbose output: %D\n",pcbddc->dbg_flag));
    PetscCall(PetscViewerASCIIPrintf(viewer,"  Use user-defined CSR: %d\n",!!pcbddc->mat_graph->nvtxs_csr));
    PetscCall(PetscViewerASCIIPrintf(viewer,"  Use local mat graph: %d\n",pcbddc->use_local_adj && !pcbddc->mat_graph->nvtxs_csr));
    if (pcbddc->mat_graph->twodim) {
      PetscCall(PetscViewerASCIIPrintf(viewer,"  Connectivity graph topological dimension: 2\n"));
    } else {
      PetscCall(PetscViewerASCIIPrintf(viewer,"  Connectivity graph topological dimension: 3\n"));
    }
    if (pcbddc->graphmaxcount != PETSC_MAX_INT) {
      PetscCall(PetscViewerASCIIPrintf(viewer,"  Graph max count: %D\n",pcbddc->graphmaxcount));
    }
    PetscCall(PetscViewerASCIIPrintf(viewer,"  Corner selection: %d (selected %d)\n",pcbddc->corner_selection,pcbddc->corner_selected));
    PetscCall(PetscViewerASCIIPrintf(viewer,"  Use vertices: %d (vertex size %D)\n",pcbddc->use_vertices,pcbddc->vertex_size));
    PetscCall(PetscViewerASCIIPrintf(viewer,"  Use edges: %d\n",pcbddc->use_edges));
    PetscCall(PetscViewerASCIIPrintf(viewer,"  Use faces: %d\n",pcbddc->use_faces));
    PetscCall(PetscViewerASCIIPrintf(viewer,"  Use true near null space: %d\n",pcbddc->use_nnsp_true));
    PetscCall(PetscViewerASCIIPrintf(viewer,"  Use QR for single constraints on cc: %d\n",pcbddc->use_qr_single));
    PetscCall(PetscViewerASCIIPrintf(viewer,"  Use change of basis on local edge nodes: %d\n",pcbddc->use_change_of_basis));
    PetscCall(PetscViewerASCIIPrintf(viewer,"  Use change of basis on local face nodes: %d\n",pcbddc->use_change_on_faces));
    PetscCall(PetscViewerASCIIPrintf(viewer,"  User defined change of basis matrix: %d\n",!!pcbddc->user_ChangeOfBasisMatrix));
    PetscCall(PetscViewerASCIIPrintf(viewer,"  Has change of basis matrix: %d\n",!!pcbddc->ChangeOfBasisMatrix));
    PetscCall(PetscViewerASCIIPrintf(viewer,"  Eliminate dirichlet boundary dofs: %d\n",pcbddc->eliminate_dirdofs));
    PetscCall(PetscViewerASCIIPrintf(viewer,"  Switch on static condensation ops around the interface preconditioner: %d\n",pcbddc->switch_static));
    PetscCall(PetscViewerASCIIPrintf(viewer,"  Use exact dirichlet trick: %d\n",pcbddc->use_exact_dirichlet_trick));
    PetscCall(PetscViewerASCIIPrintf(viewer,"  Interface extension: %s\n",PCBDDCInterfaceExtTypes[pcbddc->interface_extension]));
    PetscCall(PetscViewerASCIIPrintf(viewer,"  Multilevel max levels: %D\n",pcbddc->max_levels));
    PetscCall(PetscViewerASCIIPrintf(viewer,"  Multilevel coarsening ratio: %D\n",pcbddc->coarsening_ratio));
    PetscCall(PetscViewerASCIIPrintf(viewer,"  Use estimated eigs for coarse problem: %d\n",pcbddc->use_coarse_estimates));
    PetscCall(PetscViewerASCIIPrintf(viewer,"  Use deluxe scaling: %d\n",pcbddc->use_deluxe_scaling));
    PetscCall(PetscViewerASCIIPrintf(viewer,"  Use deluxe zerorows: %d\n",pcbddc->deluxe_zerorows));
    PetscCall(PetscViewerASCIIPrintf(viewer,"  Use deluxe singlemat: %d\n",pcbddc->deluxe_singlemat));
    PetscCall(PetscViewerASCIIPrintf(viewer,"  Rebuild interface graph for Schur principal minors: %d\n",pcbddc->sub_schurs_rebuild));
    PetscCall(PetscViewerASCIIPrintf(viewer,"  Number of dofs' layers for the computation of principal minors: %D\n",pcbddc->sub_schurs_layers));
    PetscCall(PetscViewerASCIIPrintf(viewer,"  Use user CSR graph to compute successive layers: %d\n",pcbddc->sub_schurs_use_useradj));
    if (pcbddc->adaptive_threshold[1] != pcbddc->adaptive_threshold[0]) {
      PetscCall(PetscViewerASCIIPrintf(viewer,"  Adaptive constraint selection thresholds (active %d, userdefined %d): %g,%g\n",pcbddc->adaptive_selection,pcbddc->adaptive_userdefined,pcbddc->adaptive_threshold[0],pcbddc->adaptive_threshold[1]));
    } else {
      PetscCall(PetscViewerASCIIPrintf(viewer,"  Adaptive constraint selection threshold (active %d, userdefined %d): %g\n",pcbddc->adaptive_selection,pcbddc->adaptive_userdefined,pcbddc->adaptive_threshold[0]));
    }
    PetscCall(PetscViewerASCIIPrintf(viewer,"  Min constraints / connected component: %D\n",pcbddc->adaptive_nmin));
    PetscCall(PetscViewerASCIIPrintf(viewer,"  Max constraints / connected component: %D\n",pcbddc->adaptive_nmax));
    PetscCall(PetscViewerASCIIPrintf(viewer,"  Invert exact Schur complement for adaptive selection: %d\n",pcbddc->sub_schurs_exact_schur));
    PetscCall(PetscViewerASCIIPrintf(viewer,"  Symmetric computation of primal basis functions: %d\n",pcbddc->symmetric_primal));
    PetscCall(PetscViewerASCIIPrintf(viewer,"  Num. Procs. to map coarse adjacency list: %D\n",pcbddc->coarse_adj_red));
    PetscCall(PetscViewerASCIIPrintf(viewer,"  Coarse eqs per proc (significant at the coarsest level): %D\n",pcbddc->coarse_eqs_per_proc));
    PetscCall(PetscViewerASCIIPrintf(viewer,"  Detect disconnected: %d (filter %d)\n",pcbddc->detect_disconnected,pcbddc->detect_disconnected_filter));
    PetscCall(PetscViewerASCIIPrintf(viewer,"  Benign subspace trick: %d (change explicit %d)\n",pcbddc->benign_saddle_point,pcbddc->benign_change_explicit));
    PetscCall(PetscViewerASCIIPrintf(viewer,"  Benign subspace trick is active: %d\n",pcbddc->benign_have_null));
    PetscCall(PetscViewerASCIIPrintf(viewer,"  Algebraic computation of no-net-flux: %d\n",pcbddc->compute_nonetflux));
    if (!pc->setupcalled) PetscFunctionReturn(0);

    /* compute interface size */
    PetscCall(VecSet(pcis->vec1_B,1.0));
    PetscCall(MatCreateVecs(pc->pmat,&counter,NULL));
    PetscCall(VecSet(counter,0.0));
    PetscCall(VecScatterBegin(pcis->global_to_B,pcis->vec1_B,counter,INSERT_VALUES,SCATTER_REVERSE));
    PetscCall(VecScatterEnd(pcis->global_to_B,pcis->vec1_B,counter,INSERT_VALUES,SCATTER_REVERSE));
    PetscCall(VecSum(counter,&interface_size));
    PetscCall(VecDestroy(&counter));

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
    PetscCallMPI(MPI_Reduce(loc,gsum,6,MPIU_INT64,MPI_SUM,0,PetscObjectComm((PetscObject)pc)));
    if (!loc[0]) loc[1] = loc[2] = loc[3] = loc[4] = loc[5] = -1;
    PetscCallMPI(MPI_Reduce(loc,gmax,6,MPIU_INT64,MPI_MAX,0,PetscObjectComm((PetscObject)pc)));
    if (!loc[0]) loc[1] = loc[2] = loc[3] = loc[4] = loc[5] = PETSC_MAX_INT;
    PetscCallMPI(MPI_Reduce(loc,gmin,6,MPIU_INT64,MPI_MIN,0,PetscObjectComm((PetscObject)pc)));
    PetscCallMPI(MPI_Reduce(&loc[6],&totbenign,1,MPIU_INT64,MPI_SUM,0,PetscObjectComm((PetscObject)pc)));
    if (pcbddc->coarse_size) {
      ratio1 = pc->pmat->rmap->N/(1.*pcbddc->coarse_size);
      ratio2 = PetscRealPart(interface_size)/pcbddc->coarse_size;
    }
    PetscCall(PetscViewerASCIIPrintf(viewer,"********************************** STATISTICS AT LEVEL %d **********************************\n",pcbddc->current_level));
    PetscCall(PetscViewerASCIIPrintf(viewer,"  Global dofs sizes: all %D interface %D coarse %D\n",pc->pmat->rmap->N,(PetscInt)PetscRealPart(interface_size),pcbddc->coarse_size));
    PetscCall(PetscViewerASCIIPrintf(viewer,"  Coarsening ratios: all/coarse %D interface/coarse %D\n",(PetscInt)ratio1,(PetscInt)ratio2));
    PetscCall(PetscViewerASCIIPrintf(viewer,"  Active processes : %D\n",(PetscInt)gsum[0]));
    PetscCall(PetscViewerASCIIPrintf(viewer,"  Total subdomains : %D\n",(PetscInt)gsum[5]));
    if (pcbddc->benign_have_null) {
      PetscCall(PetscViewerASCIIPrintf(viewer,"  Benign subs      : %D\n",(PetscInt)totbenign));
    }
    PetscCall(PetscViewerASCIIPrintf(viewer,"  Dofs type        :\tMIN\tMAX\tMEAN\n"));
    PetscCall(PetscViewerASCIIPrintf(viewer,"  Interior  dofs   :\t%D\t%D\t%D\n",(PetscInt)gmin[1],(PetscInt)gmax[1],(PetscInt)(gsum[1]/gsum[0])));
    PetscCall(PetscViewerASCIIPrintf(viewer,"  Interface dofs   :\t%D\t%D\t%D\n",(PetscInt)gmin[2],(PetscInt)gmax[2],(PetscInt)(gsum[2]/gsum[0])));
    PetscCall(PetscViewerASCIIPrintf(viewer,"  Primal    dofs   :\t%D\t%D\t%D\n",(PetscInt)gmin[3],(PetscInt)gmax[3],(PetscInt)(gsum[3]/gsum[0])));
    PetscCall(PetscViewerASCIIPrintf(viewer,"  Local     dofs   :\t%D\t%D\t%D\n",(PetscInt)gmin[4],(PetscInt)gmax[4],(PetscInt)(gsum[4]/gsum[0])));
    PetscCall(PetscViewerASCIIPrintf(viewer,"  Local     subs   :\t%D\t%D\n"    ,(PetscInt)gmin[5],(PetscInt)gmax[5]));
    PetscCall(PetscViewerFlush(viewer));

    PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)pc),&rank));

    /* local solvers */
    PetscCall(PetscViewerGetSubViewer(viewer,PetscObjectComm((PetscObject)pcbddc->ksp_D),&subviewer));
    if (rank == 0) {
      PetscCall(PetscViewerASCIIPrintf(subviewer,"--- Interior solver (rank 0)\n"));
      PetscCall(PetscViewerASCIIPushTab(subviewer));
      PetscCall(KSPView(pcbddc->ksp_D,subviewer));
      PetscCall(PetscViewerASCIIPopTab(subviewer));
      PetscCall(PetscViewerASCIIPrintf(subviewer,"--- Correction solver (rank 0)\n"));
      PetscCall(PetscViewerASCIIPushTab(subviewer));
      PetscCall(KSPView(pcbddc->ksp_R,subviewer));
      PetscCall(PetscViewerASCIIPopTab(subviewer));
      PetscCall(PetscViewerFlush(subviewer));
    }
    PetscCall(PetscViewerRestoreSubViewer(viewer,PetscObjectComm((PetscObject)pcbddc->ksp_D),&subviewer));
    PetscCall(PetscViewerFlush(viewer));

    /* the coarse problem can be handled by a different communicator */
    if (pcbddc->coarse_ksp) color = 1;
    else color = 0;
    PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)pc),&size));
    PetscCall(PetscSubcommCreate(PetscObjectComm((PetscObject)pc),&subcomm));
    PetscCall(PetscSubcommSetNumber(subcomm,PetscMin(size,2)));
    PetscCall(PetscSubcommSetTypeGeneral(subcomm,color,rank));
    PetscCall(PetscViewerGetSubViewer(viewer,PetscSubcommChild(subcomm),&subviewer));
    if (color == 1) {
      PetscCall(PetscViewerASCIIPrintf(subviewer,"--- Coarse solver\n"));
      PetscCall(PetscViewerASCIIPushTab(subviewer));
      PetscCall(KSPView(pcbddc->coarse_ksp,subviewer));
      PetscCall(PetscViewerASCIIPopTab(subviewer));
      PetscCall(PetscViewerFlush(subviewer));
    }
    PetscCall(PetscViewerRestoreSubViewer(viewer,PetscSubcommChild(subcomm),&subviewer));
    PetscCall(PetscSubcommDestroy(&subcomm));
    PetscCall(PetscViewerFlush(viewer));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PCBDDCSetDiscreteGradient_BDDC(PC pc, Mat G, PetscInt order, PetscInt field, PetscBool global, PetscBool conforming)
{
  PC_BDDC        *pcbddc = (PC_BDDC*)pc->data;

  PetscFunctionBegin;
  PetscCall(PetscObjectReference((PetscObject)G));
  PetscCall(MatDestroy(&pcbddc->discretegradient));
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
  PetscTryMethod(pc,"PCBDDCSetDiscreteGradient_C",(PC,Mat,PetscInt,PetscInt,PetscBool,PetscBool),(pc,G,order,field,global,conforming));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCBDDCSetDivergenceMat_BDDC(PC pc, Mat divudotp, PetscBool trans, IS vl2l)
{
  PC_BDDC        *pcbddc = (PC_BDDC*)pc->data;

  PetscFunctionBegin;
  PetscCall(PetscObjectReference((PetscObject)divudotp));
  PetscCall(MatDestroy(&pcbddc->divudotp));
  pcbddc->divudotp = divudotp;
  pcbddc->divudotp_trans = trans;
  pcbddc->compute_nonetflux = PETSC_TRUE;
  if (vl2l) {
    PetscCall(PetscObjectReference((PetscObject)vl2l));
    PetscCall(ISDestroy(&pcbddc->divudotp_vl2l));
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
  PetscCall(PetscObjectTypeCompare((PetscObject)divudotp,MATIS,&ismatis));
  PetscCheck(ismatis,PetscObjectComm((PetscObject)pc),PETSC_ERR_ARG_WRONG,"Divergence matrix needs to be of type MATIS");
  PetscTryMethod(pc,"PCBDDCSetDivergenceMat_C",(PC,Mat,PetscBool,IS),(pc,divudotp,trans,vl2l));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCBDDCSetChangeOfBasisMat_BDDC(PC pc, Mat change, PetscBool interior)
{
  PC_BDDC        *pcbddc = (PC_BDDC*)pc->data;

  PetscFunctionBegin;
  PetscCall(PetscObjectReference((PetscObject)change));
  PetscCall(MatDestroy(&pcbddc->user_ChangeOfBasisMatrix));
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
    PetscCall(MatGetSize(pc->mat,&rows,&cols));
    PetscCall(MatGetSize(change,&rows_c,&cols_c));
    PetscCheck(rows_c == rows,PetscObjectComm((PetscObject)pc),PETSC_ERR_SUP,"Invalid number of rows for change of basis matrix! %D != %D",rows_c,rows);
    PetscCheck(cols_c == cols,PetscObjectComm((PetscObject)pc),PETSC_ERR_SUP,"Invalid number of columns for change of basis matrix! %D != %D",cols_c,cols);
    PetscCall(MatGetLocalSize(pc->mat,&rows,&cols));
    PetscCall(MatGetLocalSize(change,&rows_c,&cols_c));
    PetscCheck(rows_c == rows,PetscObjectComm((PetscObject)pc),PETSC_ERR_SUP,"Invalid number of local rows for change of basis matrix! %D != %D",rows_c,rows);
    PetscCheck(cols_c == cols,PetscObjectComm((PetscObject)pc),PETSC_ERR_SUP,"Invalid number of local columns for change of basis matrix! %D != %D",cols_c,cols);
  }
  PetscTryMethod(pc,"PCBDDCSetChangeOfBasisMat_C",(PC,Mat,PetscBool),(pc,change,interior));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCBDDCSetPrimalVerticesIS_BDDC(PC pc, IS PrimalVertices)
{
  PC_BDDC        *pcbddc = (PC_BDDC*)pc->data;
  PetscBool      isequal = PETSC_FALSE;

  PetscFunctionBegin;
  PetscCall(PetscObjectReference((PetscObject)PrimalVertices));
  if (pcbddc->user_primal_vertices) {
    PetscCall(ISEqual(PrimalVertices,pcbddc->user_primal_vertices,&isequal));
  }
  PetscCall(ISDestroy(&pcbddc->user_primal_vertices));
  PetscCall(ISDestroy(&pcbddc->user_primal_vertices_local));
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
  PetscTryMethod(pc,"PCBDDCSetPrimalVerticesIS_C",(PC,IS),(pc,PrimalVertices));
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
  PetscUseMethod(pc,"PCBDDCGetPrimalVerticesIS_C",(PC,IS*),(pc,is));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCBDDCSetPrimalVerticesLocalIS_BDDC(PC pc, IS PrimalVertices)
{
  PC_BDDC        *pcbddc = (PC_BDDC*)pc->data;
  PetscBool      isequal = PETSC_FALSE;

  PetscFunctionBegin;
  PetscCall(PetscObjectReference((PetscObject)PrimalVertices));
  if (pcbddc->user_primal_vertices_local) {
    PetscCall(ISEqual(PrimalVertices,pcbddc->user_primal_vertices_local,&isequal));
  }
  PetscCall(ISDestroy(&pcbddc->user_primal_vertices));
  PetscCall(ISDestroy(&pcbddc->user_primal_vertices_local));
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
  PetscTryMethod(pc,"PCBDDCSetPrimalVerticesLocalIS_C",(PC,IS),(pc,PrimalVertices));
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
  PetscUseMethod(pc,"PCBDDCGetPrimalVerticesLocalIS_C",(PC,IS*),(pc,is));
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
  PetscTryMethod(pc,"PCBDDCSetCoarseningRatio_C",(PC,PetscInt),(pc,k));
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
  PetscTryMethod(pc,"PCBDDCSetUseExactDirichlet_C",(PC,PetscBool),(pc,flg));
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
  PetscTryMethod(pc,"PCBDDCSetLevel_C",(PC,PetscInt),(pc,level));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCBDDCSetLevels_BDDC(PC pc,PetscInt levels)
{
  PC_BDDC  *pcbddc = (PC_BDDC*)pc->data;

  PetscFunctionBegin;
  PetscCheck(levels < PETSC_PCBDDC_MAXLEVELS,PetscObjectComm((PetscObject)pc),PETSC_ERR_SUP,"Maximum number of additional levels for BDDC is %d",PETSC_PCBDDC_MAXLEVELS-1);
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
  PetscTryMethod(pc,"PCBDDCSetLevels_C",(PC,PetscInt),(pc,levels));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCBDDCSetDirichletBoundaries_BDDC(PC pc,IS DirichletBoundaries)
{
  PC_BDDC        *pcbddc = (PC_BDDC*)pc->data;
  PetscBool      isequal = PETSC_FALSE;

  PetscFunctionBegin;
  PetscCall(PetscObjectReference((PetscObject)DirichletBoundaries));
  if (pcbddc->DirichletBoundaries) {
    PetscCall(ISEqual(DirichletBoundaries,pcbddc->DirichletBoundaries,&isequal));
  }
  /* last user setting takes precedence -> destroy any other customization */
  PetscCall(ISDestroy(&pcbddc->DirichletBoundariesLocal));
  PetscCall(ISDestroy(&pcbddc->DirichletBoundaries));
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
  PetscTryMethod(pc,"PCBDDCSetDirichletBoundaries_C",(PC,IS),(pc,DirichletBoundaries));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCBDDCSetDirichletBoundariesLocal_BDDC(PC pc,IS DirichletBoundaries)
{
  PC_BDDC        *pcbddc = (PC_BDDC*)pc->data;
  PetscBool      isequal = PETSC_FALSE;

  PetscFunctionBegin;
  PetscCall(PetscObjectReference((PetscObject)DirichletBoundaries));
  if (pcbddc->DirichletBoundariesLocal) {
    PetscCall(ISEqual(DirichletBoundaries,pcbddc->DirichletBoundariesLocal,&isequal));
  }
  /* last user setting takes precedence -> destroy any other customization */
  PetscCall(ISDestroy(&pcbddc->DirichletBoundariesLocal));
  PetscCall(ISDestroy(&pcbddc->DirichletBoundaries));
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
  PetscTryMethod(pc,"PCBDDCSetDirichletBoundariesLocal_C",(PC,IS),(pc,DirichletBoundaries));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCBDDCSetNeumannBoundaries_BDDC(PC pc,IS NeumannBoundaries)
{
  PC_BDDC        *pcbddc = (PC_BDDC*)pc->data;
  PetscBool      isequal = PETSC_FALSE;

  PetscFunctionBegin;
  PetscCall(PetscObjectReference((PetscObject)NeumannBoundaries));
  if (pcbddc->NeumannBoundaries) {
    PetscCall(ISEqual(NeumannBoundaries,pcbddc->NeumannBoundaries,&isequal));
  }
  /* last user setting takes precedence -> destroy any other customization */
  PetscCall(ISDestroy(&pcbddc->NeumannBoundariesLocal));
  PetscCall(ISDestroy(&pcbddc->NeumannBoundaries));
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
  PetscTryMethod(pc,"PCBDDCSetNeumannBoundaries_C",(PC,IS),(pc,NeumannBoundaries));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCBDDCSetNeumannBoundariesLocal_BDDC(PC pc,IS NeumannBoundaries)
{
  PC_BDDC        *pcbddc = (PC_BDDC*)pc->data;
  PetscBool      isequal = PETSC_FALSE;

  PetscFunctionBegin;
  PetscCall(PetscObjectReference((PetscObject)NeumannBoundaries));
  if (pcbddc->NeumannBoundariesLocal) {
    PetscCall(ISEqual(NeumannBoundaries,pcbddc->NeumannBoundariesLocal,&isequal));
  }
  /* last user setting takes precedence -> destroy any other customization */
  PetscCall(ISDestroy(&pcbddc->NeumannBoundariesLocal));
  PetscCall(ISDestroy(&pcbddc->NeumannBoundaries));
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
  PetscTryMethod(pc,"PCBDDCSetNeumannBoundariesLocal_C",(PC,IS),(pc,NeumannBoundaries));
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
  PetscUseMethod(pc,"PCBDDCGetDirichletBoundaries_C",(PC,IS*),(pc,DirichletBoundaries));
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
  PetscUseMethod(pc,"PCBDDCGetDirichletBoundariesLocal_C",(PC,IS*),(pc,DirichletBoundaries));
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
  PetscUseMethod(pc,"PCBDDCGetNeumannBoundaries_C",(PC,IS*),(pc,NeumannBoundaries));
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
  PetscUseMethod(pc,"PCBDDCGetNeumannBoundariesLocal_C",(PC,IS*),(pc,NeumannBoundaries));
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
      PetscCall(PetscFree(xadj));
      PetscCall(PetscFree(adjncy));
    }
    PetscCall(PCBDDCGraphResetCSR(mat_graph));
    PetscFunctionReturn(0);
  }
  if (mat_graph->nvtxs == nvtxs && mat_graph->freecsr) { /* we own the data */
    if (mat_graph->xadj == xadj && mat_graph->adjncy == adjncy) same_data = PETSC_TRUE;
    if (!same_data && mat_graph->xadj[nvtxs] == xadj[nvtxs]) {
      PetscCall(PetscArraycmp(xadj,mat_graph->xadj,nvtxs+1,&same_data));
      if (same_data) {
        PetscCall(PetscArraycmp(adjncy,mat_graph->adjncy,xadj[nvtxs],&same_data));
      }
    }
  }
  if (!same_data) {
    /* free old CSR */
    PetscCall(PCBDDCGraphResetCSR(mat_graph));
    /* get CSR into graph structure */
    if (copymode == PETSC_COPY_VALUES) {
      PetscCall(PetscMalloc1(nvtxs+1,&mat_graph->xadj));
      PetscCall(PetscMalloc1(xadj[nvtxs],&mat_graph->adjncy));
      PetscCall(PetscArraycpy(mat_graph->xadj,xadj,nvtxs+1));
      PetscCall(PetscArraycpy(mat_graph->adjncy,adjncy,xadj[nvtxs]));
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
  PetscTryMethod(pc,"PCBDDCSetLocalAdjacencyGraph_C",(PC,PetscInt,const PetscInt[],const PetscInt[],PetscCopyMode),(pc,nvtxs,xadj,adjncy,copymode));
  /* free arrays if PCBDDC is not the PC type */
  PetscCall(PetscObjectQueryFunction((PetscObject)pc,"PCBDDCSetLocalAdjacencyGraph_C",&f));
  if (!f && copymode == PETSC_OWN_POINTER) {
    PetscCall(PetscFree(xadj));
    PetscCall(PetscFree(adjncy));
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
      PetscCall(ISEqual(ISForDofs[i],pcbddc->ISForDofsLocal[i],&isequalt));
      if (!isequalt) break;
    }
    if (i == n_is) isequal = PETSC_TRUE;
  }
  for (i=0;i<n_is;i++) {
    PetscCall(PetscObjectReference((PetscObject)ISForDofs[i]));
  }
  /* Destroy ISes if they were already set */
  for (i=0;i<pcbddc->n_ISForDofsLocal;i++) {
    PetscCall(ISDestroy(&pcbddc->ISForDofsLocal[i]));
  }
  PetscCall(PetscFree(pcbddc->ISForDofsLocal));
  /* last user setting takes precedence -> destroy any other customization */
  for (i=0;i<pcbddc->n_ISForDofs;i++) {
    PetscCall(ISDestroy(&pcbddc->ISForDofs[i]));
  }
  PetscCall(PetscFree(pcbddc->ISForDofs));
  pcbddc->n_ISForDofs = 0;
  /* allocate space then set */
  if (n_is) {
    PetscCall(PetscMalloc1(n_is,&pcbddc->ISForDofsLocal));
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
  PetscTryMethod(pc,"PCBDDCSetDofsSplittingLocal_C",(PC,PetscInt,IS[]),(pc,n_is,ISForDofs));
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
      PetscCall(ISEqual(ISForDofs[i],pcbddc->ISForDofs[i],&isequalt));
      if (!isequalt) break;
    }
    if (i == n_is) isequal = PETSC_TRUE;
  }
  for (i=0;i<n_is;i++) {
    PetscCall(PetscObjectReference((PetscObject)ISForDofs[i]));
  }
  /* Destroy ISes if they were already set */
  for (i=0;i<pcbddc->n_ISForDofs;i++) {
    PetscCall(ISDestroy(&pcbddc->ISForDofs[i]));
  }
  PetscCall(PetscFree(pcbddc->ISForDofs));
  /* last user setting takes precedence -> destroy any other customization */
  for (i=0;i<pcbddc->n_ISForDofsLocal;i++) {
    PetscCall(ISDestroy(&pcbddc->ISForDofsLocal[i]));
  }
  PetscCall(PetscFree(pcbddc->ISForDofsLocal));
  pcbddc->n_ISForDofsLocal = 0;
  /* allocate space then set */
  if (n_is) {
    PetscCall(PetscMalloc1(n_is,&pcbddc->ISForDofs));
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
  PetscTryMethod(pc,"PCBDDCSetDofsSplitting_C",(PC,PetscInt,IS[]),(pc,n_is,ISForDofs));
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

    PetscCall(PetscObjectTypeCompare((PetscObject)ksp,KSPCG,&iscg));
    PetscCall(PetscObjectTypeCompare((PetscObject)ksp,KSPGROPPCG,&isgroppcg));
    PetscCall(PetscObjectTypeCompare((PetscObject)ksp,KSPPIPECG,&ispipecg));
    PetscCall(PetscObjectTypeCompare((PetscObject)ksp,KSPPIPECG,&ispipelcg));
    PetscCall(PetscObjectTypeCompare((PetscObject)ksp,KSPPIPECGRR,&ispipecgrr));
    iscg = (PetscBool)(iscg || isgroppcg || ispipecg || ispipelcg || ispipecgrr);
    if (pcbddc->benign_apply_coarse_only || pcbddc->switch_static || !iscg || pc->mat != pc->pmat) {
      PetscCall(PCBDDCSetUseExactDirichlet(pc,PETSC_FALSE));
    }
  }
  if (pcbddc->benign_apply_coarse_only || pcbddc->switch_static || pc->mat != pc->pmat) {
    PetscCall(PCBDDCSetUseExactDirichlet(pc,PETSC_FALSE));
  }

  /* Creates parallel work vectors used in presolve */
  if (!pcbddc->original_rhs) {
    PetscCall(VecDuplicate(pcis->vec1_global,&pcbddc->original_rhs));
  }
  if (!pcbddc->temp_solution) {
    PetscCall(VecDuplicate(pcis->vec1_global,&pcbddc->temp_solution));
  }

  pcbddc->temp_solution_used = PETSC_FALSE;
  if (x) {
    PetscCall(PetscObjectReference((PetscObject)x));
    used_vec = x;
  } else { /* it can only happen when calling PCBDDCMatFETIDPGetRHS */
    PetscCall(PetscObjectReference((PetscObject)pcbddc->temp_solution));
    used_vec = pcbddc->temp_solution;
    PetscCall(VecSet(used_vec,0.0));
    pcbddc->temp_solution_used = PETSC_TRUE;
    PetscCall(VecCopy(rhs,pcbddc->original_rhs));
    save_rhs = PETSC_FALSE;
    pcbddc->eliminate_dirdofs = PETSC_TRUE;
  }

  /* hack into ksp data structure since PCPreSolve comes earlier than setting to zero the guess in src/ksp/ksp/interface/itfunc.c */
  if (ksp) {
    /* store the flag for the initial guess since it will be restored back during PCPostSolve_BDDC */
    PetscCall(KSPGetInitialGuessNonzero(ksp,&pcbddc->ksp_guess_nonzero));
    if (!pcbddc->ksp_guess_nonzero) {
      PetscCall(VecSet(used_vec,0.0));
    }
  }

  pcbddc->rhs_change = PETSC_FALSE;
  /* Take into account zeroed rows -> change rhs and store solution removed */
  if (rhs && pcbddc->eliminate_dirdofs) {
    IS dirIS = NULL;

    /* DirichletBoundariesLocal may not be consistent among neighbours; gets a dirichlet dofs IS from graph (may be cached) */
    PetscCall(PCBDDCGraphGetDirichletDofs(pcbddc->mat_graph,&dirIS));
    if (dirIS) {
      Mat_IS            *matis = (Mat_IS*)pc->pmat->data;
      PetscInt          dirsize,i,*is_indices;
      PetscScalar       *array_x;
      const PetscScalar *array_diagonal;

      PetscCall(MatGetDiagonal(pc->pmat,pcis->vec1_global));
      PetscCall(VecPointwiseDivide(pcis->vec1_global,rhs,pcis->vec1_global));
      PetscCall(VecScatterBegin(matis->rctx,pcis->vec1_global,pcis->vec2_N,INSERT_VALUES,SCATTER_FORWARD));
      PetscCall(VecScatterEnd(matis->rctx,pcis->vec1_global,pcis->vec2_N,INSERT_VALUES,SCATTER_FORWARD));
      PetscCall(VecScatterBegin(matis->rctx,used_vec,pcis->vec1_N,INSERT_VALUES,SCATTER_FORWARD));
      PetscCall(VecScatterEnd(matis->rctx,used_vec,pcis->vec1_N,INSERT_VALUES,SCATTER_FORWARD));
      PetscCall(ISGetLocalSize(dirIS,&dirsize));
      PetscCall(VecGetArray(pcis->vec1_N,&array_x));
      PetscCall(VecGetArrayRead(pcis->vec2_N,&array_diagonal));
      PetscCall(ISGetIndices(dirIS,(const PetscInt**)&is_indices));
      for (i=0; i<dirsize; i++) array_x[is_indices[i]] = array_diagonal[is_indices[i]];
      PetscCall(ISRestoreIndices(dirIS,(const PetscInt**)&is_indices));
      PetscCall(VecRestoreArrayRead(pcis->vec2_N,&array_diagonal));
      PetscCall(VecRestoreArray(pcis->vec1_N,&array_x));
      PetscCall(VecScatterBegin(matis->rctx,pcis->vec1_N,used_vec,INSERT_VALUES,SCATTER_REVERSE));
      PetscCall(VecScatterEnd(matis->rctx,pcis->vec1_N,used_vec,INSERT_VALUES,SCATTER_REVERSE));
      pcbddc->rhs_change = PETSC_TRUE;
      PetscCall(ISDestroy(&dirIS));
    }
  }

  /* remove the computed solution or the initial guess from the rhs */
  if (pcbddc->rhs_change || (ksp && pcbddc->ksp_guess_nonzero)) {
    /* save the original rhs */
    if (save_rhs) {
      PetscCall(VecSwap(rhs,pcbddc->original_rhs));
      save_rhs = PETSC_FALSE;
    }
    pcbddc->rhs_change = PETSC_TRUE;
    PetscCall(VecScale(used_vec,-1.0));
    PetscCall(MatMultAdd(pc->mat,used_vec,pcbddc->original_rhs,rhs));
    PetscCall(VecScale(used_vec,-1.0));
    PetscCall(VecCopy(used_vec,pcbddc->temp_solution));
    pcbddc->temp_solution_used = PETSC_TRUE;
    if (ksp) {
      PetscCall(KSPSetInitialGuessNonzero(ksp,PETSC_FALSE));
    }
  }
  PetscCall(VecDestroy(&used_vec));

  /* compute initial vector in benign space if needed
     and remove non-benign solution from the rhs */
  benign_correction_computed = PETSC_FALSE;
  if (rhs && pcbddc->benign_compute_correction && (pcbddc->benign_have_null || pcbddc->benign_apply_coarse_only)) {
    /* compute u^*_h using ideas similar to those in Xuemin Tu's PhD thesis (see Section 4.8.1)
       Recursively apply BDDC in the multilevel case */
    if (!pcbddc->benign_vec) {
      PetscCall(VecDuplicate(rhs,&pcbddc->benign_vec));
    }
    /* keep applying coarse solver unless we no longer have benign subdomains */
    pcbddc->benign_apply_coarse_only = pcbddc->benign_have_null ? PETSC_TRUE : PETSC_FALSE;
    if (!pcbddc->benign_skip_correction) {
      PetscCall(PCApply_BDDC(pc,rhs,pcbddc->benign_vec));
      benign_correction_computed = PETSC_TRUE;
      if (pcbddc->temp_solution_used) {
        PetscCall(VecAXPY(pcbddc->temp_solution,1.0,pcbddc->benign_vec));
      }
      PetscCall(VecScale(pcbddc->benign_vec,-1.0));
      /* store the original rhs if not done earlier */
      if (save_rhs) {
        PetscCall(VecSwap(rhs,pcbddc->original_rhs));
      }
      if (pcbddc->rhs_change) {
        PetscCall(MatMultAdd(pc->mat,pcbddc->benign_vec,rhs,rhs));
      } else {
        PetscCall(MatMultAdd(pc->mat,pcbddc->benign_vec,pcbddc->original_rhs,rhs));
      }
      pcbddc->rhs_change = PETSC_TRUE;
    }
    pcbddc->benign_apply_coarse_only = PETSC_FALSE;
  } else {
    PetscCall(VecDestroy(&pcbddc->benign_vec));
  }

  /* dbg output */
  if (pcbddc->dbg_flag && benign_correction_computed) {
    Vec v;

    PetscCall(VecDuplicate(pcis->vec1_global,&v));
    if (pcbddc->ChangeOfBasisMatrix) {
      PetscCall(MatMultTranspose(pcbddc->ChangeOfBasisMatrix,rhs,v));
    } else {
      PetscCall(VecCopy(rhs,v));
    }
    PetscCall(PCBDDCBenignGetOrSetP0(pc,v,PETSC_TRUE));
    PetscCall(PetscViewerASCIIPrintf(pcbddc->dbg_viewer,"LEVEL %D: is the correction benign?\n",pcbddc->current_level));
    PetscCall(PetscScalarView(pcbddc->benign_n,pcbddc->benign_p0,pcbddc->dbg_viewer));
    PetscCall(PetscViewerFlush(pcbddc->dbg_viewer));
    PetscCall(VecDestroy(&v));
  }

  /* set initial guess if using PCG */
  pcbddc->exact_dirichlet_trick_app = PETSC_FALSE;
  if (x && pcbddc->use_exact_dirichlet_trick) {
    PetscCall(VecSet(x,0.0));
    if (pcbddc->ChangeOfBasisMatrix && pcbddc->change_interior) {
      if (benign_correction_computed) { /* we have already saved the changed rhs */
        PetscCall(VecLockReadPop(pcis->vec1_global));
      } else {
        PetscCall(MatMultTranspose(pcbddc->ChangeOfBasisMatrix,rhs,pcis->vec1_global));
      }
      PetscCall(VecScatterBegin(pcis->global_to_D,pcis->vec1_global,pcis->vec1_D,INSERT_VALUES,SCATTER_FORWARD));
      PetscCall(VecScatterEnd(pcis->global_to_D,pcis->vec1_global,pcis->vec1_D,INSERT_VALUES,SCATTER_FORWARD));
    } else {
      PetscCall(VecScatterBegin(pcis->global_to_D,rhs,pcis->vec1_D,INSERT_VALUES,SCATTER_FORWARD));
      PetscCall(VecScatterEnd(pcis->global_to_D,rhs,pcis->vec1_D,INSERT_VALUES,SCATTER_FORWARD));
    }
    PetscCall(PetscLogEventBegin(PC_BDDC_Solves[pcbddc->current_level][0],pc,0,0,0));
    PetscCall(KSPSolve(pcbddc->ksp_D,pcis->vec1_D,pcis->vec2_D));
    PetscCall(PetscLogEventEnd(PC_BDDC_Solves[pcbddc->current_level][0],pc,0,0,0));
    PetscCall(KSPCheckSolve(pcbddc->ksp_D,pc,pcis->vec2_D));
    if (pcbddc->ChangeOfBasisMatrix && pcbddc->change_interior) {
      PetscCall(VecSet(pcis->vec1_global,0.));
      PetscCall(VecScatterBegin(pcis->global_to_D,pcis->vec2_D,pcis->vec1_global,INSERT_VALUES,SCATTER_REVERSE));
      PetscCall(VecScatterEnd(pcis->global_to_D,pcis->vec2_D,pcis->vec1_global,INSERT_VALUES,SCATTER_REVERSE));
      PetscCall(MatMult(pcbddc->ChangeOfBasisMatrix,pcis->vec1_global,x));
    } else {
      PetscCall(VecScatterBegin(pcis->global_to_D,pcis->vec2_D,x,INSERT_VALUES,SCATTER_REVERSE));
      PetscCall(VecScatterEnd(pcis->global_to_D,pcis->vec2_D,x,INSERT_VALUES,SCATTER_REVERSE));
    }
    if (ksp) {
      PetscCall(KSPSetInitialGuessNonzero(ksp,PETSC_TRUE));
    }
    pcbddc->exact_dirichlet_trick_app = PETSC_TRUE;
  } else if (pcbddc->ChangeOfBasisMatrix && pcbddc->change_interior && benign_correction_computed && pcbddc->use_exact_dirichlet_trick) {
    PetscCall(VecLockReadPop(pcis->vec1_global));
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
      PetscCall(VecAXPY(x,1.0,pcbddc->temp_solution));
    } else if (pcbddc->benign_compute_correction && pcbddc->benign_vec) {
      PetscCall(VecAXPY(x,-1.0,pcbddc->benign_vec));
    }
    /* restore to original state (not for FETI-DP) */
    if (ksp) pcbddc->temp_solution_used = PETSC_FALSE;
  }

  /* restore rhs to its original state (not needed for FETI-DP) */
  if (rhs && pcbddc->rhs_change) {
    PetscCall(VecSwap(rhs,pcbddc->original_rhs));
    pcbddc->rhs_change = PETSC_FALSE;
  }
  /* restore ksp guess state */
  if (ksp) {
    PetscCall(KSPSetInitialGuessNonzero(ksp,pcbddc->ksp_guess_nonzero));
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
  PetscCall(PetscObjectTypeCompare((PetscObject)pc->pmat,MATIS,&ismatis));
  PetscCheck(ismatis,PetscObjectComm((PetscObject)pc),PETSC_ERR_ARG_WRONG,"PCBDDC preconditioner requires matrix of type MATIS");
  PetscCall(MatGetSize(pc->pmat,&nrows,&ncols));
  PetscCheck(nrows == ncols,PetscObjectComm((PetscObject)pc),PETSC_ERR_SUP,"PCBDDC preconditioner requires a square preconditioning matrix");
  PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)pc),&size));

  matis = (Mat_IS*)pc->pmat->data;
  /* the following lines of code should be replaced by a better logic between PCIS, PCNN, PCBDDC and other future nonoverlapping preconditioners */
  /* For BDDC we need to define a local "Neumann" problem different to that defined in PCISSetup
     Also, BDDC builds its own KSP for the Dirichlet problem */
  rl = pcbddc->recompute_topography;
  if (!pc->setupcalled || pc->flag == DIFFERENT_NONZERO_PATTERN) rl = PETSC_TRUE;
  PetscCall(MPIU_Allreduce(&rl,&pcbddc->recompute_topography,1,MPIU_BOOL,MPI_LOR,PetscObjectComm((PetscObject)pc)));
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
    PetscCall(PetscViewerASCIIPushSynchronized(pcbddc->dbg_viewer));
    PetscCall(PetscViewerASCIIAddTab(pcbddc->dbg_viewer,2*pcbddc->current_level));
  }

  /* process topology information */
  PetscCall(PetscLogEventBegin(PC_BDDC_Topology[pcbddc->current_level],pc,0,0,0));
  if (pcbddc->recompute_topography) {
    PetscCall(PCBDDCComputeLocalTopologyInfo(pc));
    if (pcbddc->discretegradient) {
      PetscCall(PCBDDCNedelecSupport(pc));
    }
  }
  if (pcbddc->corner_selected) pcbddc->use_vertices = PETSC_TRUE;

  /* change basis if requested by the user */
  if (pcbddc->user_ChangeOfBasisMatrix) {
    /* use_change_of_basis flag is used to automatically compute a change of basis from constraints */
    pcbddc->use_change_of_basis = PETSC_FALSE;
    PetscCall(PCBDDCComputeLocalMatrix(pc,pcbddc->user_ChangeOfBasisMatrix));
  } else {
    PetscCall(MatDestroy(&pcbddc->local_mat));
    PetscCall(PetscObjectReference((PetscObject)matis->A));
    pcbddc->local_mat = matis->A;
  }

  /*
     Compute change of basis on local pressures (aka zerodiag dofs) with the benign trick
     This should come earlier then PCISSetUp for extracting the correct subdomain matrices
  */
  PetscCall(PCBDDCBenignShellMat(pc,PETSC_TRUE));
  if (pcbddc->benign_saddle_point) {
    PC_IS* pcis = (PC_IS*)pc->data;

    if (pcbddc->user_ChangeOfBasisMatrix || pcbddc->use_change_of_basis || !computesubschurs) pcbddc->benign_change_explicit = PETSC_TRUE;
    /* detect local saddle point and change the basis in pcbddc->local_mat */
    PetscCall(PCBDDCBenignDetectSaddlePoint(pc,(PetscBool)(!pcbddc->recompute_topography),&zerodiag));
    /* pop B0 mat from local mat */
    PetscCall(PCBDDCBenignPopOrPushB0(pc,PETSC_TRUE));
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
    PetscCall(MatSetOption(pcbddc->local_mat,MAT_SYMMETRIC,matis->A->symmetric));
  }
  if (matis->A->spd_set) {
    PetscCall(MatSetOption(pcbddc->local_mat,MAT_SPD,matis->A->spd));
  }

  /* Set up all the "iterative substructuring" common block without computing solvers */
  {
    Mat temp_mat;

    temp_mat = matis->A;
    matis->A = pcbddc->local_mat;
    PetscCall(PCISSetUp(pc,PETSC_TRUE,PETSC_FALSE));
    pcbddc->local_mat = matis->A;
    matis->A = temp_mat;
  }

  /* Analyze interface */
  if (!pcbddc->graphanalyzed) {
    PetscCall(PCBDDCAnalyzeInterface(pc));
    computeconstraintsmatrix = PETSC_TRUE;
    if (pcbddc->adaptive_selection && !pcbddc->use_deluxe_scaling && !pcbddc->mat_graph->twodim) {
      SETERRQ(PetscObjectComm((PetscObject)pc),PETSC_ERR_SUP,"Cannot compute the adaptive primal space for a problem with 3D edges without deluxe scaling");
    }
    if (pcbddc->compute_nonetflux) {
      MatNullSpace nnfnnsp;

      PetscCheck(pcbddc->divudotp,PetscObjectComm((PetscObject)pc),PETSC_ERR_SUP,"Missing divudotp operator");
      PetscCall(PCBDDCComputeNoNetFlux(pc->pmat,pcbddc->divudotp,pcbddc->divudotp_trans,pcbddc->divudotp_vl2l,pcbddc->mat_graph,&nnfnnsp));
      /* TODO what if a nearnullspace is already attached? */
      if (nnfnnsp) {
        PetscCall(MatSetNearNullSpace(pc->pmat,nnfnnsp));
        PetscCall(MatNullSpaceDestroy(&nnfnnsp));
      }
    }
  }
  PetscCall(PetscLogEventEnd(PC_BDDC_Topology[pcbddc->current_level],pc,0,0,0));

  /* check existence of a divergence free extension, i.e.
     b(v_I,p_0) = 0 for all v_I (raise error if not).
     Also, check that PCBDDCBenignGetOrSetP0 works */
  if (pcbddc->benign_saddle_point && pcbddc->dbg_flag > 1) {
    PetscCall(PCBDDCBenignCheck(pc,zerodiag));
  }
  PetscCall(ISDestroy(&zerodiag));

  /* Setup local dirichlet solver ksp_D and sub_schurs solvers */
  if (computesubschurs && pcbddc->recompute_topography) {
    PetscCall(PCBDDCInitSubSchurs(pc));
  }
  /* SetUp Scaling operator (scaling matrices could be needed in SubSchursSetUp)*/
  if (!pcbddc->use_deluxe_scaling) {
    PetscCall(PCBDDCScalingSetUp(pc));
  }

  /* finish setup solvers and do adaptive selection of constraints */
  sub_schurs = pcbddc->sub_schurs;
  if (sub_schurs && sub_schurs->schur_explicit) {
    if (computesubschurs) {
      PetscCall(PCBDDCSetUpSubSchurs(pc));
    }
    PetscCall(PCBDDCSetUpLocalSolvers(pc,PETSC_TRUE,PETSC_FALSE));
  } else {
    PetscCall(PCBDDCSetUpLocalSolvers(pc,PETSC_TRUE,PETSC_FALSE));
    if (computesubschurs) {
      PetscCall(PCBDDCSetUpSubSchurs(pc));
    }
  }
  if (pcbddc->adaptive_selection) {
    PetscCall(PCBDDCAdaptiveSelection(pc));
    computeconstraintsmatrix = PETSC_TRUE;
  }

  /* infer if NullSpace object attached to Mat via MatSetNearNullSpace has changed */
  new_nearnullspace_provided = PETSC_FALSE;
  PetscCall(MatGetNearNullSpace(pc->pmat,&nearnullspace));
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
        PetscCall(MatNullSpaceGetVecs(nearnullspace,NULL,&nnsp_size,&nearnullvecs));
        for (i=0;i<nnsp_size;i++) {
          PetscCall(PetscObjectStateGet((PetscObject)nearnullvecs[i],&state));
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
  PetscCall(PetscLogEventBegin(PC_BDDC_LocalWork[pcbddc->current_level],pc,0,0,0));
  pcbddc->new_primal_space = PETSC_FALSE;
  pcbddc->new_primal_space_local = PETSC_FALSE;
  if (computeconstraintsmatrix || new_nearnullspace_provided) {
    /* It also sets the primal space flags */
    PetscCall(PCBDDCConstraintsSetUp(pc));
  }
  /* Allocate needed local vectors (which depends on quantities defined during ConstraintsSetUp) */
  PetscCall(PCBDDCSetUpLocalWorkVectors(pc));

  if (pcbddc->use_change_of_basis) {
    PC_IS *pcis = (PC_IS*)(pc->data);

    PetscCall(PCBDDCComputeLocalMatrix(pc,pcbddc->ChangeOfBasisMatrix));
    if (pcbddc->benign_change) {
      PetscCall(MatDestroy(&pcbddc->benign_B0));
      /* pop B0 from pcbddc->local_mat */
      PetscCall(PCBDDCBenignPopOrPushB0(pc,PETSC_TRUE));
    }
    /* get submatrices */
    PetscCall(MatDestroy(&pcis->A_IB));
    PetscCall(MatDestroy(&pcis->A_BI));
    PetscCall(MatDestroy(&pcis->A_BB));
    PetscCall(MatCreateSubMatrix(pcbddc->local_mat,pcis->is_B_local,pcis->is_B_local,MAT_INITIAL_MATRIX,&pcis->A_BB));
    PetscCall(MatCreateSubMatrix(pcbddc->local_mat,pcis->is_I_local,pcis->is_B_local,MAT_INITIAL_MATRIX,&pcis->A_IB));
    PetscCall(MatCreateSubMatrix(pcbddc->local_mat,pcis->is_B_local,pcis->is_I_local,MAT_INITIAL_MATRIX,&pcis->A_BI));
    /* set flag in pcis to not reuse submatrices during PCISCreate */
    pcis->reusesubmatrices = PETSC_FALSE;
  } else if (!pcbddc->user_ChangeOfBasisMatrix && !pcbddc->benign_change) {
    PetscCall(MatDestroy(&pcbddc->local_mat));
    PetscCall(PetscObjectReference((PetscObject)matis->A));
    pcbddc->local_mat = matis->A;
  }

  /* interface pressure block row for B_C */
  PetscCall(PetscObjectQuery((PetscObject)pc,"__KSPFETIDP_lP" ,(PetscObject*)&lP));
  PetscCall(PetscObjectQuery((PetscObject)pc,"__KSPFETIDP_lA" ,(PetscObject*)&lA));
  if (lA && lP) {
    PC_IS*    pcis = (PC_IS*)pc->data;
    Mat       B_BI,B_BB,Bt_BI,Bt_BB;
    PetscBool issym;
    PetscCall(MatIsSymmetric(lA,PETSC_SMALL,&issym));
    if (issym) {
      PetscCall(MatCreateSubMatrix(lA,lP,pcis->is_I_local,MAT_INITIAL_MATRIX,&B_BI));
      PetscCall(MatCreateSubMatrix(lA,lP,pcis->is_B_local,MAT_INITIAL_MATRIX,&B_BB));
      PetscCall(MatCreateTranspose(B_BI,&Bt_BI));
      PetscCall(MatCreateTranspose(B_BB,&Bt_BB));
    } else {
      PetscCall(MatCreateSubMatrix(lA,lP,pcis->is_I_local,MAT_INITIAL_MATRIX,&B_BI));
      PetscCall(MatCreateSubMatrix(lA,lP,pcis->is_B_local,MAT_INITIAL_MATRIX,&B_BB));
      PetscCall(MatCreateSubMatrix(lA,pcis->is_I_local,lP,MAT_INITIAL_MATRIX,&Bt_BI));
      PetscCall(MatCreateSubMatrix(lA,pcis->is_B_local,lP,MAT_INITIAL_MATRIX,&Bt_BB));
    }
    PetscCall(PetscObjectCompose((PetscObject)pc,"__KSPFETIDP_B_BI",(PetscObject)B_BI));
    PetscCall(PetscObjectCompose((PetscObject)pc,"__KSPFETIDP_B_BB",(PetscObject)B_BB));
    PetscCall(PetscObjectCompose((PetscObject)pc,"__KSPFETIDP_Bt_BI",(PetscObject)Bt_BI));
    PetscCall(PetscObjectCompose((PetscObject)pc,"__KSPFETIDP_Bt_BB",(PetscObject)Bt_BB));
    PetscCall(MatDestroy(&B_BI));
    PetscCall(MatDestroy(&B_BB));
    PetscCall(MatDestroy(&Bt_BI));
    PetscCall(MatDestroy(&Bt_BB));
  }
  PetscCall(PetscLogEventEnd(PC_BDDC_LocalWork[pcbddc->current_level],pc,0,0,0));

  /* SetUp coarse and local Neumann solvers */
  PetscCall(PCBDDCSetUpSolvers(pc));
  /* SetUp Scaling operator */
  if (pcbddc->use_deluxe_scaling) {
    PetscCall(PCBDDCScalingSetUp(pc));
  }

  /* mark topography as done */
  pcbddc->recompute_topography = PETSC_FALSE;

  /* wrap pcis->A_IB and pcis->A_BI if we did not change explicitly the variables on the pressures */
  PetscCall(PCBDDCBenignShellMat(pc,PETSC_FALSE));

  if (pcbddc->dbg_flag) {
    PetscCall(PetscViewerASCIISubtractTab(pcbddc->dbg_viewer,2*pcbddc->current_level));
    PetscCall(PetscViewerASCIIPopSynchronized(pcbddc->dbg_viewer));
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
  PetscCall(PetscCitationsRegister(citation,&cited));
  if (pcbddc->switch_static) {
    PetscCall(MatISGetLocalMat(pc->useAmat ? pc->mat : pc->pmat,&lA));
  }

  if (pcbddc->ChangeOfBasisMatrix) {
    Vec swap;

    PetscCall(MatMultTranspose(pcbddc->ChangeOfBasisMatrix,r,pcbddc->work_change));
    swap = pcbddc->work_change;
    pcbddc->work_change = r;
    r = swap;
    /* save rhs so that we don't need to apply the change of basis for the exact dirichlet trick in PreSolve */
    if (pcbddc->benign_apply_coarse_only && pcbddc->use_exact_dirichlet_trick && pcbddc->change_interior) {
      PetscCall(VecCopy(r,pcis->vec1_global));
      PetscCall(VecLockReadPush(pcis->vec1_global));
    }
  }
  if (pcbddc->benign_have_null) { /* get p0 from r */
    PetscCall(PCBDDCBenignGetOrSetP0(pc,r,PETSC_TRUE));
  }
  if (pcbddc->interface_extension == PC_BDDC_INTERFACE_EXT_DIRICHLET && !pcbddc->exact_dirichlet_trick_app && !pcbddc->benign_apply_coarse_only) {
    PetscCall(VecCopy(r,z));
    /* First Dirichlet solve */
    PetscCall(VecScatterBegin(pcis->global_to_D,r,pcis->vec1_D,INSERT_VALUES,SCATTER_FORWARD));
    PetscCall(VecScatterEnd(pcis->global_to_D,r,pcis->vec1_D,INSERT_VALUES,SCATTER_FORWARD));
    /*
      Assembling right hand side for BDDC operator
      - pcis->vec1_D for the Dirichlet part (if needed, i.e. pcbddc->switch_static == PETSC_TRUE)
      - pcis->vec1_B the interface part of the global vector z
    */
    if (n_D) {
      PetscCall(PetscLogEventBegin(PC_BDDC_Solves[pcbddc->current_level][0],pc,0,0,0));
      PetscCall(KSPSolve(pcbddc->ksp_D,pcis->vec1_D,pcis->vec2_D));
      PetscCall(PetscLogEventEnd(PC_BDDC_Solves[pcbddc->current_level][0],pc,0,0,0));
      PetscCall(KSPCheckSolve(pcbddc->ksp_D,pc,pcis->vec2_D));
      PetscCall(VecScale(pcis->vec2_D,m_one));
      if (pcbddc->switch_static) {
        PetscCall(VecSet(pcis->vec1_N,0.));
        PetscCall(VecScatterBegin(pcis->N_to_D,pcis->vec2_D,pcis->vec1_N,INSERT_VALUES,SCATTER_REVERSE));
        PetscCall(VecScatterEnd(pcis->N_to_D,pcis->vec2_D,pcis->vec1_N,INSERT_VALUES,SCATTER_REVERSE));
        if (!pcbddc->switch_static_change) {
          PetscCall(MatMult(lA,pcis->vec1_N,pcis->vec2_N));
        } else {
          PetscCall(MatMult(pcbddc->switch_static_change,pcis->vec1_N,pcis->vec2_N));
          PetscCall(MatMult(lA,pcis->vec2_N,pcis->vec1_N));
          PetscCall(MatMultTranspose(pcbddc->switch_static_change,pcis->vec1_N,pcis->vec2_N));
        }
        PetscCall(VecScatterBegin(pcis->N_to_D,pcis->vec2_N,pcis->vec1_D,ADD_VALUES,SCATTER_FORWARD));
        PetscCall(VecScatterEnd(pcis->N_to_D,pcis->vec2_N,pcis->vec1_D,ADD_VALUES,SCATTER_FORWARD));
        PetscCall(VecScatterBegin(pcis->N_to_B,pcis->vec2_N,pcis->vec1_B,INSERT_VALUES,SCATTER_FORWARD));
        PetscCall(VecScatterEnd(pcis->N_to_B,pcis->vec2_N,pcis->vec1_B,INSERT_VALUES,SCATTER_FORWARD));
      } else {
        PetscCall(MatMult(pcis->A_BI,pcis->vec2_D,pcis->vec1_B));
      }
    } else {
      PetscCall(VecSet(pcis->vec1_B,zero));
    }
    PetscCall(VecScatterBegin(pcis->global_to_B,pcis->vec1_B,z,ADD_VALUES,SCATTER_REVERSE));
    PetscCall(VecScatterEnd(pcis->global_to_B,pcis->vec1_B,z,ADD_VALUES,SCATTER_REVERSE));
    PetscCall(PCBDDCScalingRestriction(pc,z,pcis->vec1_B));
  } else {
    if (!pcbddc->benign_apply_coarse_only) {
      PetscCall(PCBDDCScalingRestriction(pc,r,pcis->vec1_B));
    }
  }
  if (pcbddc->interface_extension == PC_BDDC_INTERFACE_EXT_LUMP) {
    PetscCheck(pcbddc->switch_static,PetscObjectComm((PetscObject)pc),PETSC_ERR_SUP,"You forgot to pass -pc_bddc_switch_static");
    PetscCall(VecScatterBegin(pcis->global_to_D,r,pcis->vec1_D,INSERT_VALUES,SCATTER_FORWARD));
    PetscCall(VecScatterEnd(pcis->global_to_D,r,pcis->vec1_D,INSERT_VALUES,SCATTER_FORWARD));
  }

  /* Apply interface preconditioner
     input/output vecs: pcis->vec1_B and pcis->vec1_D */
  PetscCall(PCBDDCApplyInterfacePreconditioner(pc,PETSC_FALSE));

  /* Apply transpose of partition of unity operator */
  PetscCall(PCBDDCScalingExtension(pc,pcis->vec1_B,z));
  if (pcbddc->interface_extension == PC_BDDC_INTERFACE_EXT_LUMP) {
    PetscCall(VecScatterBegin(pcis->global_to_D,pcis->vec1_D,z,INSERT_VALUES,SCATTER_REVERSE));
    PetscCall(VecScatterEnd(pcis->global_to_D,pcis->vec1_D,z,INSERT_VALUES,SCATTER_REVERSE));
    PetscFunctionReturn(0);
  }
  /* Second Dirichlet solve and assembling of output */
  PetscCall(VecScatterBegin(pcis->global_to_B,z,pcis->vec1_B,INSERT_VALUES,SCATTER_FORWARD));
  PetscCall(VecScatterEnd(pcis->global_to_B,z,pcis->vec1_B,INSERT_VALUES,SCATTER_FORWARD));
  if (n_B) {
    if (pcbddc->switch_static) {
      PetscCall(VecScatterBegin(pcis->N_to_D,pcis->vec1_D,pcis->vec1_N,INSERT_VALUES,SCATTER_REVERSE));
      PetscCall(VecScatterEnd(pcis->N_to_D,pcis->vec1_D,pcis->vec1_N,INSERT_VALUES,SCATTER_REVERSE));
      PetscCall(VecScatterBegin(pcis->N_to_B,pcis->vec1_B,pcis->vec1_N,INSERT_VALUES,SCATTER_REVERSE));
      PetscCall(VecScatterEnd(pcis->N_to_B,pcis->vec1_B,pcis->vec1_N,INSERT_VALUES,SCATTER_REVERSE));
      if (!pcbddc->switch_static_change) {
        PetscCall(MatMult(lA,pcis->vec1_N,pcis->vec2_N));
      } else {
        PetscCall(MatMult(pcbddc->switch_static_change,pcis->vec1_N,pcis->vec2_N));
        PetscCall(MatMult(lA,pcis->vec2_N,pcis->vec1_N));
        PetscCall(MatMultTranspose(pcbddc->switch_static_change,pcis->vec1_N,pcis->vec2_N));
      }
      PetscCall(VecScatterBegin(pcis->N_to_D,pcis->vec2_N,pcis->vec3_D,INSERT_VALUES,SCATTER_FORWARD));
      PetscCall(VecScatterEnd(pcis->N_to_D,pcis->vec2_N,pcis->vec3_D,INSERT_VALUES,SCATTER_FORWARD));
    } else {
      PetscCall(MatMult(pcis->A_IB,pcis->vec1_B,pcis->vec3_D));
    }
  } else if (pcbddc->switch_static) { /* n_B is zero */
    if (!pcbddc->switch_static_change) {
      PetscCall(MatMult(lA,pcis->vec1_D,pcis->vec3_D));
    } else {
      PetscCall(MatMult(pcbddc->switch_static_change,pcis->vec1_D,pcis->vec1_N));
      PetscCall(MatMult(lA,pcis->vec1_N,pcis->vec2_N));
      PetscCall(MatMultTranspose(pcbddc->switch_static_change,pcis->vec2_N,pcis->vec3_D));
    }
  }
  PetscCall(PetscLogEventBegin(PC_BDDC_Solves[pcbddc->current_level][0],pc,0,0,0));
  PetscCall(KSPSolve(pcbddc->ksp_D,pcis->vec3_D,pcis->vec4_D));
  PetscCall(PetscLogEventEnd(PC_BDDC_Solves[pcbddc->current_level][0],pc,0,0,0));
  PetscCall(KSPCheckSolve(pcbddc->ksp_D,pc,pcis->vec4_D));

  if (!pcbddc->exact_dirichlet_trick_app && !pcbddc->benign_apply_coarse_only) {
    if (pcbddc->switch_static) {
      PetscCall(VecAXPBYPCZ(pcis->vec2_D,m_one,one,m_one,pcis->vec4_D,pcis->vec1_D));
    } else {
      PetscCall(VecAXPBY(pcis->vec2_D,m_one,m_one,pcis->vec4_D));
    }
    PetscCall(VecScatterBegin(pcis->global_to_D,pcis->vec2_D,z,INSERT_VALUES,SCATTER_REVERSE));
    PetscCall(VecScatterEnd(pcis->global_to_D,pcis->vec2_D,z,INSERT_VALUES,SCATTER_REVERSE));
  } else {
    if (pcbddc->switch_static) {
      PetscCall(VecAXPBY(pcis->vec4_D,one,m_one,pcis->vec1_D));
    } else {
      PetscCall(VecScale(pcis->vec4_D,m_one));
    }
    PetscCall(VecScatterBegin(pcis->global_to_D,pcis->vec4_D,z,INSERT_VALUES,SCATTER_REVERSE));
    PetscCall(VecScatterEnd(pcis->global_to_D,pcis->vec4_D,z,INSERT_VALUES,SCATTER_REVERSE));
  }
  if (pcbddc->benign_have_null) { /* set p0 (computed in PCBDDCApplyInterface) */
    if (pcbddc->benign_apply_coarse_only) {
      PetscCall(PetscArrayzero(pcbddc->benign_p0,pcbddc->benign_n));
    }
    PetscCall(PCBDDCBenignGetOrSetP0(pc,z,PETSC_FALSE));
  }

  if (pcbddc->ChangeOfBasisMatrix) {
    pcbddc->work_change = r;
    PetscCall(VecCopy(z,pcbddc->work_change));
    PetscCall(MatMult(pcbddc->ChangeOfBasisMatrix,pcbddc->work_change,z));
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
  PetscCall(PetscCitationsRegister(citation,&cited));
  if (pcbddc->switch_static) {
    PetscCall(MatISGetLocalMat(pc->useAmat ? pc->mat : pc->pmat,&lA));
  }
  if (pcbddc->ChangeOfBasisMatrix) {
    Vec swap;

    PetscCall(MatMultTranspose(pcbddc->ChangeOfBasisMatrix,r,pcbddc->work_change));
    swap = pcbddc->work_change;
    pcbddc->work_change = r;
    r = swap;
    /* save rhs so that we don't need to apply the change of basis for the exact dirichlet trick in PreSolve */
    if (pcbddc->benign_apply_coarse_only && pcbddc->exact_dirichlet_trick_app && pcbddc->change_interior) {
      PetscCall(VecCopy(r,pcis->vec1_global));
      PetscCall(VecLockReadPush(pcis->vec1_global));
    }
  }
  if (pcbddc->benign_have_null) { /* get p0 from r */
    PetscCall(PCBDDCBenignGetOrSetP0(pc,r,PETSC_TRUE));
  }
  if (!pcbddc->exact_dirichlet_trick_app && !pcbddc->benign_apply_coarse_only) {
    PetscCall(VecCopy(r,z));
    /* First Dirichlet solve */
    PetscCall(VecScatterBegin(pcis->global_to_D,r,pcis->vec1_D,INSERT_VALUES,SCATTER_FORWARD));
    PetscCall(VecScatterEnd(pcis->global_to_D,r,pcis->vec1_D,INSERT_VALUES,SCATTER_FORWARD));
    /*
      Assembling right hand side for BDDC operator
      - pcis->vec1_D for the Dirichlet part (if needed, i.e. pcbddc->switch_static == PETSC_TRUE)
      - pcis->vec1_B the interface part of the global vector z
    */
    if (n_D) {
      PetscCall(PetscLogEventBegin(PC_BDDC_Solves[pcbddc->current_level][0],pc,0,0,0));
      PetscCall(KSPSolveTranspose(pcbddc->ksp_D,pcis->vec1_D,pcis->vec2_D));
      PetscCall(PetscLogEventEnd(PC_BDDC_Solves[pcbddc->current_level][0],pc,0,0,0));
      PetscCall(KSPCheckSolve(pcbddc->ksp_D,pc,pcis->vec2_D));
      PetscCall(VecScale(pcis->vec2_D,m_one));
      if (pcbddc->switch_static) {
        PetscCall(VecSet(pcis->vec1_N,0.));
        PetscCall(VecScatterBegin(pcis->N_to_D,pcis->vec2_D,pcis->vec1_N,INSERT_VALUES,SCATTER_REVERSE));
        PetscCall(VecScatterEnd(pcis->N_to_D,pcis->vec2_D,pcis->vec1_N,INSERT_VALUES,SCATTER_REVERSE));
        if (!pcbddc->switch_static_change) {
          PetscCall(MatMultTranspose(lA,pcis->vec1_N,pcis->vec2_N));
        } else {
          PetscCall(MatMult(pcbddc->switch_static_change,pcis->vec1_N,pcis->vec2_N));
          PetscCall(MatMultTranspose(lA,pcis->vec2_N,pcis->vec1_N));
          PetscCall(MatMultTranspose(pcbddc->switch_static_change,pcis->vec1_N,pcis->vec2_N));
        }
        PetscCall(VecScatterBegin(pcis->N_to_D,pcis->vec2_N,pcis->vec1_D,ADD_VALUES,SCATTER_FORWARD));
        PetscCall(VecScatterEnd(pcis->N_to_D,pcis->vec2_N,pcis->vec1_D,ADD_VALUES,SCATTER_FORWARD));
        PetscCall(VecScatterBegin(pcis->N_to_B,pcis->vec2_N,pcis->vec1_B,INSERT_VALUES,SCATTER_FORWARD));
        PetscCall(VecScatterEnd(pcis->N_to_B,pcis->vec2_N,pcis->vec1_B,INSERT_VALUES,SCATTER_FORWARD));
      } else {
        PetscCall(MatMultTranspose(pcis->A_IB,pcis->vec2_D,pcis->vec1_B));
      }
    } else {
      PetscCall(VecSet(pcis->vec1_B,zero));
    }
    PetscCall(VecScatterBegin(pcis->global_to_B,pcis->vec1_B,z,ADD_VALUES,SCATTER_REVERSE));
    PetscCall(VecScatterEnd(pcis->global_to_B,pcis->vec1_B,z,ADD_VALUES,SCATTER_REVERSE));
    PetscCall(PCBDDCScalingRestriction(pc,z,pcis->vec1_B));
  } else {
    PetscCall(PCBDDCScalingRestriction(pc,r,pcis->vec1_B));
  }

  /* Apply interface preconditioner
     input/output vecs: pcis->vec1_B and pcis->vec1_D */
  PetscCall(PCBDDCApplyInterfacePreconditioner(pc,PETSC_TRUE));

  /* Apply transpose of partition of unity operator */
  PetscCall(PCBDDCScalingExtension(pc,pcis->vec1_B,z));

  /* Second Dirichlet solve and assembling of output */
  PetscCall(VecScatterBegin(pcis->global_to_B,z,pcis->vec1_B,INSERT_VALUES,SCATTER_FORWARD));
  PetscCall(VecScatterEnd(pcis->global_to_B,z,pcis->vec1_B,INSERT_VALUES,SCATTER_FORWARD));
  if (n_B) {
    if (pcbddc->switch_static) {
      PetscCall(VecScatterBegin(pcis->N_to_D,pcis->vec1_D,pcis->vec1_N,INSERT_VALUES,SCATTER_REVERSE));
      PetscCall(VecScatterEnd(pcis->N_to_D,pcis->vec1_D,pcis->vec1_N,INSERT_VALUES,SCATTER_REVERSE));
      PetscCall(VecScatterBegin(pcis->N_to_B,pcis->vec1_B,pcis->vec1_N,INSERT_VALUES,SCATTER_REVERSE));
      PetscCall(VecScatterEnd(pcis->N_to_B,pcis->vec1_B,pcis->vec1_N,INSERT_VALUES,SCATTER_REVERSE));
      if (!pcbddc->switch_static_change) {
        PetscCall(MatMultTranspose(lA,pcis->vec1_N,pcis->vec2_N));
      } else {
        PetscCall(MatMult(pcbddc->switch_static_change,pcis->vec1_N,pcis->vec2_N));
        PetscCall(MatMultTranspose(lA,pcis->vec2_N,pcis->vec1_N));
        PetscCall(MatMultTranspose(pcbddc->switch_static_change,pcis->vec1_N,pcis->vec2_N));
      }
      PetscCall(VecScatterBegin(pcis->N_to_D,pcis->vec2_N,pcis->vec3_D,INSERT_VALUES,SCATTER_FORWARD));
      PetscCall(VecScatterEnd(pcis->N_to_D,pcis->vec2_N,pcis->vec3_D,INSERT_VALUES,SCATTER_FORWARD));
    } else {
      PetscCall(MatMultTranspose(pcis->A_BI,pcis->vec1_B,pcis->vec3_D));
    }
  } else if (pcbddc->switch_static) { /* n_B is zero */
    if (!pcbddc->switch_static_change) {
      PetscCall(MatMultTranspose(lA,pcis->vec1_D,pcis->vec3_D));
    } else {
      PetscCall(MatMult(pcbddc->switch_static_change,pcis->vec1_D,pcis->vec1_N));
      PetscCall(MatMultTranspose(lA,pcis->vec1_N,pcis->vec2_N));
      PetscCall(MatMultTranspose(pcbddc->switch_static_change,pcis->vec2_N,pcis->vec3_D));
    }
  }
  PetscCall(PetscLogEventBegin(PC_BDDC_Solves[pcbddc->current_level][0],pc,0,0,0));
  PetscCall(KSPSolveTranspose(pcbddc->ksp_D,pcis->vec3_D,pcis->vec4_D));
  PetscCall(PetscLogEventEnd(PC_BDDC_Solves[pcbddc->current_level][0],pc,0,0,0));
  PetscCall(KSPCheckSolve(pcbddc->ksp_D,pc,pcis->vec4_D));
  if (!pcbddc->exact_dirichlet_trick_app && !pcbddc->benign_apply_coarse_only) {
    if (pcbddc->switch_static) {
      PetscCall(VecAXPBYPCZ(pcis->vec2_D,m_one,one,m_one,pcis->vec4_D,pcis->vec1_D));
    } else {
      PetscCall(VecAXPBY(pcis->vec2_D,m_one,m_one,pcis->vec4_D));
    }
    PetscCall(VecScatterBegin(pcis->global_to_D,pcis->vec2_D,z,INSERT_VALUES,SCATTER_REVERSE));
    PetscCall(VecScatterEnd(pcis->global_to_D,pcis->vec2_D,z,INSERT_VALUES,SCATTER_REVERSE));
  } else {
    if (pcbddc->switch_static) {
      PetscCall(VecAXPBY(pcis->vec4_D,one,m_one,pcis->vec1_D));
    } else {
      PetscCall(VecScale(pcis->vec4_D,m_one));
    }
    PetscCall(VecScatterBegin(pcis->global_to_D,pcis->vec4_D,z,INSERT_VALUES,SCATTER_REVERSE));
    PetscCall(VecScatterEnd(pcis->global_to_D,pcis->vec4_D,z,INSERT_VALUES,SCATTER_REVERSE));
  }
  if (pcbddc->benign_have_null) { /* set p0 (computed in PCBDDCApplyInterface) */
    PetscCall(PCBDDCBenignGetOrSetP0(pc,z,PETSC_FALSE));
  }
  if (pcbddc->ChangeOfBasisMatrix) {
    pcbddc->work_change = r;
    PetscCall(VecCopy(z,pcbddc->work_change));
    PetscCall(MatMult(pcbddc->ChangeOfBasisMatrix,pcbddc->work_change,z));
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
  PetscCall(PCBDDCResetCustomization(pc));
  /* destroy objects related to topography */
  PetscCall(PCBDDCResetTopography(pc));
  /* destroy objects for scaling operator */
  PetscCall(PCBDDCScalingDestroy(pc));
  /* free solvers stuff */
  PetscCall(PCBDDCResetSolvers(pc));
  /* free global vectors needed in presolve */
  PetscCall(VecDestroy(&pcbddc->temp_solution));
  PetscCall(VecDestroy(&pcbddc->original_rhs));
  /* free data created by PCIS */
  PetscCall(PCISDestroy(pc));

  /* restore defaults */
  kspD = pcbddc->ksp_D;
  kspR = pcbddc->ksp_R;
  kspC = pcbddc->coarse_ksp;
  PetscCall(PetscMemzero(pc->data,sizeof(*pcbddc)));
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
  PetscCall(PCReset_BDDC(pc));
  PetscCall(KSPDestroy(&pcbddc->ksp_D));
  PetscCall(KSPDestroy(&pcbddc->ksp_R));
  PetscCall(KSPDestroy(&pcbddc->coarse_ksp));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc,"PCBDDCSetDiscreteGradient_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc,"PCBDDCSetDivergenceMat_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc,"PCBDDCSetChangeOfBasisMat_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc,"PCBDDCSetPrimalVerticesLocalIS_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc,"PCBDDCSetPrimalVerticesIS_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc,"PCBDDCSetCoarseningRatio_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc,"PCBDDCSetLevel_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc,"PCBDDCSetUseExactDirichlet_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc,"PCBDDCSetLevels_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc,"PCBDDCSetDirichletBoundaries_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc,"PCBDDCSetDirichletBoundariesLocal_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc,"PCBDDCSetNeumannBoundaries_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc,"PCBDDCSetNeumannBoundariesLocal_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc,"PCBDDCGetDirichletBoundaries_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc,"PCBDDCGetDirichletBoundariesLocal_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc,"PCBDDCGetNeumannBoundaries_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc,"PCBDDCGetNeumannBoundariesLocal_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc,"PCBDDCSetDofsSplitting_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc,"PCBDDCSetDofsSplittingLocal_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc,"PCBDDCSetLocalAdjacencyGraph_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc,"PCBDDCCreateFETIDPOperators_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc,"PCBDDCMatFETIDPGetRHS_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc,"PCBDDCMatFETIDPGetSolution_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc,"PCPreSolveChangeRHS_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc,"PCSetCoordinates_C",NULL));
  PetscCall(PetscFree(pc->data));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCSetCoordinates_BDDC(PC pc, PetscInt dim, PetscInt nloc, PetscReal *coords)
{
  PC_BDDC        *pcbddc = (PC_BDDC*)pc->data;
  PCBDDCGraph    mat_graph = pcbddc->mat_graph;

  PetscFunctionBegin;
  PetscCall(PetscFree(mat_graph->coords));
  PetscCall(PetscMalloc1(nloc*dim,&mat_graph->coords));
  PetscCall(PetscArraycpy(mat_graph->coords,coords,nloc*dim));
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
  PetscCall(MatShellGetContext(fetidp_mat,&mat_ctx));
  pcis = (PC_IS*)mat_ctx->pc->data;
  pcbddc = (PC_BDDC*)mat_ctx->pc->data;

  PetscCall(VecSet(fetidp_flux_rhs,0.0));
  /* copy rhs since we may change it during PCPreSolve_BDDC */
  if (!pcbddc->original_rhs) {
    PetscCall(VecDuplicate(pcis->vec1_global,&pcbddc->original_rhs));
  }
  if (mat_ctx->rhs_flip) {
    PetscCall(VecPointwiseMult(pcbddc->original_rhs,standard_rhs,mat_ctx->rhs_flip));
  } else {
    PetscCall(VecCopy(standard_rhs,pcbddc->original_rhs));
  }
  if (mat_ctx->g2g_p) {
    /* interface pressure rhs */
    PetscCall(VecScatterBegin(mat_ctx->g2g_p,fetidp_flux_rhs,pcbddc->original_rhs,INSERT_VALUES,SCATTER_REVERSE));
    PetscCall(VecScatterEnd(mat_ctx->g2g_p,fetidp_flux_rhs,pcbddc->original_rhs,INSERT_VALUES,SCATTER_REVERSE));
    PetscCall(VecScatterBegin(mat_ctx->g2g_p,standard_rhs,fetidp_flux_rhs,INSERT_VALUES,SCATTER_FORWARD));
    PetscCall(VecScatterEnd(mat_ctx->g2g_p,standard_rhs,fetidp_flux_rhs,INSERT_VALUES,SCATTER_FORWARD));
    if (!mat_ctx->rhs_flip) {
      PetscCall(VecScale(fetidp_flux_rhs,-1.));
    }
  }
  /*
     change of basis for physical rhs if needed
     It also changes the rhs in case of dirichlet boundaries
  */
  PetscCall(PCPreSolve_BDDC(mat_ctx->pc,NULL,pcbddc->original_rhs,NULL));
  if (pcbddc->ChangeOfBasisMatrix) {
    PetscCall(MatMultTranspose(pcbddc->ChangeOfBasisMatrix,pcbddc->original_rhs,pcbddc->work_change));
    work = pcbddc->work_change;
   } else {
    work = pcbddc->original_rhs;
  }
  /* store vectors for computation of fetidp final solution */
  PetscCall(VecScatterBegin(pcis->global_to_D,work,mat_ctx->temp_solution_D,INSERT_VALUES,SCATTER_FORWARD));
  PetscCall(VecScatterEnd(pcis->global_to_D,work,mat_ctx->temp_solution_D,INSERT_VALUES,SCATTER_FORWARD));
  /* scale rhs since it should be unassembled */
  /* TODO use counter scaling? (also below) */
  PetscCall(VecScatterBegin(pcis->global_to_B,work,mat_ctx->temp_solution_B,INSERT_VALUES,SCATTER_FORWARD));
  PetscCall(VecScatterEnd(pcis->global_to_B,work,mat_ctx->temp_solution_B,INSERT_VALUES,SCATTER_FORWARD));
  /* Apply partition of unity */
  PetscCall(VecPointwiseMult(mat_ctx->temp_solution_B,pcis->D,mat_ctx->temp_solution_B));
  /* PetscCall(PCBDDCScalingRestriction(mat_ctx->pc,work,mat_ctx->temp_solution_B)); */
  if (!pcbddc->switch_static) {
    /* compute partially subassembled Schur complement right-hand side */
    PetscCall(PetscLogEventBegin(PC_BDDC_Solves[pcbddc->current_level][0],mat_ctx->pc,0,0,0));
    PetscCall(KSPSolve(pcbddc->ksp_D,mat_ctx->temp_solution_D,pcis->vec1_D));
    PetscCall(PetscLogEventEnd(PC_BDDC_Solves[pcbddc->current_level][0],mat_ctx->pc,0,0,0));
    /* Cannot propagate up error in KSPSolve() because there is no access to the PC */
    PetscCall(MatMult(pcis->A_BI,pcis->vec1_D,pcis->vec1_B));
    PetscCall(VecAXPY(mat_ctx->temp_solution_B,-1.0,pcis->vec1_B));
    PetscCall(VecSet(work,0.0));
    PetscCall(VecScatterBegin(pcis->global_to_B,mat_ctx->temp_solution_B,work,ADD_VALUES,SCATTER_REVERSE));
    PetscCall(VecScatterEnd(pcis->global_to_B,mat_ctx->temp_solution_B,work,ADD_VALUES,SCATTER_REVERSE));
    /* PetscCall(PCBDDCScalingRestriction(mat_ctx->pc,work,mat_ctx->temp_solution_B)); */
    PetscCall(VecScatterBegin(pcis->global_to_B,work,mat_ctx->temp_solution_B,INSERT_VALUES,SCATTER_FORWARD));
    PetscCall(VecScatterEnd(pcis->global_to_B,work,mat_ctx->temp_solution_B,INSERT_VALUES,SCATTER_FORWARD));
    PetscCall(VecPointwiseMult(mat_ctx->temp_solution_B,pcis->D,mat_ctx->temp_solution_B));
  }
  /* BDDC rhs */
  PetscCall(VecCopy(mat_ctx->temp_solution_B,pcis->vec1_B));
  if (pcbddc->switch_static) {
    PetscCall(VecCopy(mat_ctx->temp_solution_D,pcis->vec1_D));
  }
  /* apply BDDC */
  PetscCall(PetscArrayzero(pcbddc->benign_p0,pcbddc->benign_n));
  PetscCall(PCBDDCApplyInterfacePreconditioner(mat_ctx->pc,PETSC_FALSE));
  PetscCall(PetscArrayzero(pcbddc->benign_p0,pcbddc->benign_n));

  /* Application of B_delta and assembling of rhs for fetidp fluxes */
  PetscCall(MatMult(mat_ctx->B_delta,pcis->vec1_B,mat_ctx->lambda_local));
  PetscCall(VecScatterBegin(mat_ctx->l2g_lambda,mat_ctx->lambda_local,fetidp_flux_rhs,ADD_VALUES,SCATTER_FORWARD));
  PetscCall(VecScatterEnd(mat_ctx->l2g_lambda,mat_ctx->lambda_local,fetidp_flux_rhs,ADD_VALUES,SCATTER_FORWARD));
  /* Add contribution to interface pressures */
  if (mat_ctx->l2g_p) {
    PetscCall(MatMult(mat_ctx->B_BB,pcis->vec1_B,mat_ctx->vP));
    if (pcbddc->switch_static) {
      PetscCall(MatMultAdd(mat_ctx->B_BI,pcis->vec1_D,mat_ctx->vP,mat_ctx->vP));
    }
    PetscCall(VecScatterBegin(mat_ctx->l2g_p,mat_ctx->vP,fetidp_flux_rhs,ADD_VALUES,SCATTER_FORWARD));
    PetscCall(VecScatterEnd(mat_ctx->l2g_p,mat_ctx->vP,fetidp_flux_rhs,ADD_VALUES,SCATTER_FORWARD));
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
  PetscCall(MatShellGetContext(fetidp_mat,&mat_ctx));
  PetscUseMethod(mat_ctx->pc,"PCBDDCMatFETIDPGetRHS_C",(Mat,Vec,Vec),(fetidp_mat,standard_rhs,fetidp_flux_rhs));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCBDDCMatFETIDPGetSolution_BDDC(Mat fetidp_mat, Vec fetidp_flux_sol, Vec standard_sol)
{
  FETIDPMat_ctx  mat_ctx;
  PC_IS*         pcis;
  PC_BDDC*       pcbddc;
  Vec            work;

  PetscFunctionBegin;
  PetscCall(MatShellGetContext(fetidp_mat,&mat_ctx));
  pcis = (PC_IS*)mat_ctx->pc->data;
  pcbddc = (PC_BDDC*)mat_ctx->pc->data;

  /* apply B_delta^T */
  PetscCall(VecSet(pcis->vec1_B,0.));
  PetscCall(VecScatterBegin(mat_ctx->l2g_lambda,fetidp_flux_sol,mat_ctx->lambda_local,INSERT_VALUES,SCATTER_REVERSE));
  PetscCall(VecScatterEnd(mat_ctx->l2g_lambda,fetidp_flux_sol,mat_ctx->lambda_local,INSERT_VALUES,SCATTER_REVERSE));
  PetscCall(MatMultTranspose(mat_ctx->B_delta,mat_ctx->lambda_local,pcis->vec1_B));
  if (mat_ctx->l2g_p) {
    PetscCall(VecScatterBegin(mat_ctx->l2g_p,fetidp_flux_sol,mat_ctx->vP,INSERT_VALUES,SCATTER_REVERSE));
    PetscCall(VecScatterEnd(mat_ctx->l2g_p,fetidp_flux_sol,mat_ctx->vP,INSERT_VALUES,SCATTER_REVERSE));
    PetscCall(MatMultAdd(mat_ctx->Bt_BB,mat_ctx->vP,pcis->vec1_B,pcis->vec1_B));
  }

  /* compute rhs for BDDC application */
  PetscCall(VecAYPX(pcis->vec1_B,-1.0,mat_ctx->temp_solution_B));
  if (pcbddc->switch_static) {
    PetscCall(VecCopy(mat_ctx->temp_solution_D,pcis->vec1_D));
    if (mat_ctx->l2g_p) {
      PetscCall(VecScale(mat_ctx->vP,-1.));
      PetscCall(MatMultAdd(mat_ctx->Bt_BI,mat_ctx->vP,pcis->vec1_D,pcis->vec1_D));
    }
  }

  /* apply BDDC */
  PetscCall(PetscArrayzero(pcbddc->benign_p0,pcbddc->benign_n));
  PetscCall(PCBDDCApplyInterfacePreconditioner(mat_ctx->pc,PETSC_FALSE));

  /* put values into global vector */
  if (pcbddc->ChangeOfBasisMatrix) work = pcbddc->work_change;
  else work = standard_sol;
  PetscCall(VecScatterBegin(pcis->global_to_B,pcis->vec1_B,work,INSERT_VALUES,SCATTER_REVERSE));
  PetscCall(VecScatterEnd(pcis->global_to_B,pcis->vec1_B,work,INSERT_VALUES,SCATTER_REVERSE));
  if (!pcbddc->switch_static) {
    /* compute values into the interior if solved for the partially subassembled Schur complement */
    PetscCall(MatMult(pcis->A_IB,pcis->vec1_B,pcis->vec1_D));
    PetscCall(VecAYPX(pcis->vec1_D,-1.0,mat_ctx->temp_solution_D));
    PetscCall(PetscLogEventBegin(PC_BDDC_Solves[pcbddc->current_level][0],mat_ctx->pc,0,0,0));
    PetscCall(KSPSolve(pcbddc->ksp_D,pcis->vec1_D,pcis->vec1_D));
    PetscCall(PetscLogEventEnd(PC_BDDC_Solves[pcbddc->current_level][0],mat_ctx->pc,0,0,0));
    /* Cannot propagate up error in KSPSolve() because there is no access to the PC */
  }

  PetscCall(VecScatterBegin(pcis->global_to_D,pcis->vec1_D,work,INSERT_VALUES,SCATTER_REVERSE));
  PetscCall(VecScatterEnd(pcis->global_to_D,pcis->vec1_D,work,INSERT_VALUES,SCATTER_REVERSE));
  /* add p0 solution to final solution */
  PetscCall(PCBDDCBenignGetOrSetP0(mat_ctx->pc,work,PETSC_FALSE));
  if (pcbddc->ChangeOfBasisMatrix) {
    PetscCall(MatMult(pcbddc->ChangeOfBasisMatrix,work,standard_sol));
  }
  PetscCall(PCPostSolve_BDDC(mat_ctx->pc,NULL,NULL,standard_sol));
  if (mat_ctx->g2g_p) {
    PetscCall(VecScatterBegin(mat_ctx->g2g_p,fetidp_flux_sol,standard_sol,INSERT_VALUES,SCATTER_REVERSE));
    PetscCall(VecScatterEnd(mat_ctx->g2g_p,fetidp_flux_sol,standard_sol,INSERT_VALUES,SCATTER_REVERSE));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PCView_BDDCIPC(PC pc, PetscViewer viewer)
{
  BDDCIPC_ctx    bddcipc_ctx;
  PetscBool      isascii;

  PetscFunctionBegin;
  PetscCall(PCShellGetContext(pc,&bddcipc_ctx));
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&isascii));
  if (isascii) {
    PetscCall(PetscViewerASCIIPrintf(viewer,"BDDC interface preconditioner\n"));
  }
  PetscCall(PetscViewerASCIIPushTab(viewer));
  PetscCall(PCView(bddcipc_ctx->bddc,viewer));
  PetscCall(PetscViewerASCIIPopTab(viewer));
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
  PetscCall(PCShellGetContext(pc,&bddcipc_ctx));
  PetscCall(PetscObjectTypeCompare((PetscObject)bddcipc_ctx->bddc,PCBDDC,&isbddc));
  PetscCheck(isbddc,PetscObjectComm((PetscObject)pc),PETSC_ERR_SUP,"Invalid type %s. Must be of type bddc",((PetscObject)bddcipc_ctx->bddc)->type_name);
  PetscCall(PCSetUp(bddcipc_ctx->bddc));

  /* create interface scatter */
  pcis = (PC_IS*)(bddcipc_ctx->bddc->data);
  PetscCall(VecScatterDestroy(&bddcipc_ctx->g2l));
  PetscCall(MatCreateVecs(pc->pmat,&vv,NULL));
  PetscCall(ISRenumber(pcis->is_B_global,NULL,NULL,&is));
  PetscCall(VecScatterCreate(vv,is,pcis->vec1_B,NULL,&bddcipc_ctx->g2l));
  PetscCall(ISDestroy(&is));
  PetscCall(VecDestroy(&vv));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCApply_BDDCIPC(PC pc, Vec r, Vec x)
{
  BDDCIPC_ctx    bddcipc_ctx;
  PC_IS          *pcis;
  VecScatter     tmps;

  PetscFunctionBegin;
  PetscCall(PCShellGetContext(pc,&bddcipc_ctx));
  pcis = (PC_IS*)(bddcipc_ctx->bddc->data);
  tmps = pcis->global_to_B;
  pcis->global_to_B = bddcipc_ctx->g2l;
  PetscCall(PCBDDCScalingRestriction(bddcipc_ctx->bddc,r,pcis->vec1_B));
  PetscCall(PCBDDCApplyInterfacePreconditioner(bddcipc_ctx->bddc,PETSC_FALSE));
  PetscCall(PCBDDCScalingExtension(bddcipc_ctx->bddc,pcis->vec1_B,x));
  pcis->global_to_B = tmps;
  PetscFunctionReturn(0);
}

static PetscErrorCode PCApplyTranspose_BDDCIPC(PC pc, Vec r, Vec x)
{
  BDDCIPC_ctx    bddcipc_ctx;
  PC_IS          *pcis;
  VecScatter     tmps;

  PetscFunctionBegin;
  PetscCall(PCShellGetContext(pc,&bddcipc_ctx));
  pcis = (PC_IS*)(bddcipc_ctx->bddc->data);
  tmps = pcis->global_to_B;
  pcis->global_to_B = bddcipc_ctx->g2l;
  PetscCall(PCBDDCScalingRestriction(bddcipc_ctx->bddc,r,pcis->vec1_B));
  PetscCall(PCBDDCApplyInterfacePreconditioner(bddcipc_ctx->bddc,PETSC_TRUE));
  PetscCall(PCBDDCScalingExtension(bddcipc_ctx->bddc,pcis->vec1_B,x));
  pcis->global_to_B = tmps;
  PetscFunctionReturn(0);
}

static PetscErrorCode PCDestroy_BDDCIPC(PC pc)
{
  BDDCIPC_ctx    bddcipc_ctx;

  PetscFunctionBegin;
  PetscCall(PCShellGetContext(pc,&bddcipc_ctx));
  PetscCall(PCDestroy(&bddcipc_ctx->bddc));
  PetscCall(VecScatterDestroy(&bddcipc_ctx->g2l));
  PetscCall(PetscFree(bddcipc_ctx));
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
  PetscCall(MatShellGetContext(fetidp_mat,&mat_ctx));
  PetscUseMethod(mat_ctx->pc,"PCBDDCMatFETIDPGetSolution_C",(Mat,Vec,Vec),(fetidp_mat,fetidp_flux_sol,standard_sol));
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
  PetscCall(PetscObjectGetComm((PetscObject)pc,&comm));
  /* FETI-DP matrix */
  PetscCall(PCBDDCCreateFETIDPMatContext(pc,&fetidpmat_ctx));
  fetidpmat_ctx->fully_redundant = fully_redundant;
  PetscCall(PCBDDCSetupFETIDPMatContext(fetidpmat_ctx));
  PetscCall(MatCreateShell(comm,fetidpmat_ctx->n,fetidpmat_ctx->n,fetidpmat_ctx->N,fetidpmat_ctx->N,fetidpmat_ctx,&newmat));
  PetscCall(PetscObjectSetName((PetscObject)newmat,!fetidpmat_ctx->l2g_lambda_only ? "F" : "G"));
  PetscCall(MatShellSetOperation(newmat,MATOP_MULT,(void (*)(void))FETIDPMatMult));
  PetscCall(MatShellSetOperation(newmat,MATOP_MULT_TRANSPOSE,(void (*)(void))FETIDPMatMultTranspose));
  PetscCall(MatShellSetOperation(newmat,MATOP_DESTROY,(void (*)(void))PCBDDCDestroyFETIDPMat));
  /* propagate MatOptions */
  {
    PC_BDDC   *pcbddc = (PC_BDDC*)fetidpmat_ctx->pc->data;
    PetscBool issym;

    PetscCall(MatGetOption(pc->mat,MAT_SYMMETRIC,&issym));
    if (issym || pcbddc->symmetric_primal) {
      PetscCall(MatSetOption(newmat,MAT_SYMMETRIC,PETSC_TRUE));
    }
  }
  PetscCall(MatSetOptionsPrefix(newmat,prefix));
  PetscCall(MatAppendOptionsPrefix(newmat,"fetidp_"));
  PetscCall(MatSetUp(newmat));
  /* FETI-DP preconditioner */
  PetscCall(PCBDDCCreateFETIDPPCContext(pc,&fetidppc_ctx));
  PetscCall(PCBDDCSetupFETIDPPCContext(newmat,fetidppc_ctx));
  PetscCall(PCCreate(comm,&newpc));
  PetscCall(PCSetOperators(newpc,newmat,newmat));
  PetscCall(PCSetOptionsPrefix(newpc,prefix));
  PetscCall(PCAppendOptionsPrefix(newpc,"fetidp_"));
  PetscCall(PCSetErrorIfFailure(newpc,pc->erroriffailure));
  if (!fetidpmat_ctx->l2g_lambda_only) { /* standard FETI-DP */
    PetscCall(PCSetType(newpc,PCSHELL));
    PetscCall(PCShellSetName(newpc,"FETI-DP multipliers"));
    PetscCall(PCShellSetContext(newpc,fetidppc_ctx));
    PetscCall(PCShellSetApply(newpc,FETIDPPCApply));
    PetscCall(PCShellSetApplyTranspose(newpc,FETIDPPCApplyTranspose));
    PetscCall(PCShellSetView(newpc,FETIDPPCView));
    PetscCall(PCShellSetDestroy(newpc,PCBDDCDestroyFETIDPPC));
  } else { /* saddle-point FETI-DP */
    Mat       M;
    PetscInt  psize;
    PetscBool fake = PETSC_FALSE, isfieldsplit;

    PetscCall(ISViewFromOptions(fetidpmat_ctx->lagrange,NULL,"-lag_view"));
    PetscCall(ISViewFromOptions(fetidpmat_ctx->pressure,NULL,"-press_view"));
    PetscCall(PetscObjectQuery((PetscObject)pc,"__KSPFETIDP_PPmat",(PetscObject*)&M));
    PetscCall(PCSetType(newpc,PCFIELDSPLIT));
    PetscCall(PCFieldSplitSetIS(newpc,"lag",fetidpmat_ctx->lagrange));
    PetscCall(PCFieldSplitSetIS(newpc,"p",fetidpmat_ctx->pressure));
    PetscCall(PCFieldSplitSetType(newpc,PC_COMPOSITE_SCHUR));
    PetscCall(PCFieldSplitSetSchurFactType(newpc,PC_FIELDSPLIT_SCHUR_FACT_DIAG));
    PetscCall(ISGetSize(fetidpmat_ctx->pressure,&psize));
    if (psize != M->rmap->N) {
      Mat      M2;
      PetscInt lpsize;

      fake = PETSC_TRUE;
      PetscCall(ISGetLocalSize(fetidpmat_ctx->pressure,&lpsize));
      PetscCall(MatCreate(comm,&M2));
      PetscCall(MatSetType(M2,MATAIJ));
      PetscCall(MatSetSizes(M2,lpsize,lpsize,psize,psize));
      PetscCall(MatSetUp(M2));
      PetscCall(MatAssemblyBegin(M2,MAT_FINAL_ASSEMBLY));
      PetscCall(MatAssemblyEnd(M2,MAT_FINAL_ASSEMBLY));
      PetscCall(PCFieldSplitSetSchurPre(newpc,PC_FIELDSPLIT_SCHUR_PRE_USER,M2));
      PetscCall(MatDestroy(&M2));
    } else {
      PetscCall(PCFieldSplitSetSchurPre(newpc,PC_FIELDSPLIT_SCHUR_PRE_USER,M));
    }
    PetscCall(PCFieldSplitSetSchurScale(newpc,1.0));

    /* we need to setfromoptions and setup here to access the blocks */
    PetscCall(PCSetFromOptions(newpc));
    PetscCall(PCSetUp(newpc));

    /* user may have changed the type (e.g. -fetidp_pc_type none) */
    PetscCall(PetscObjectTypeCompare((PetscObject)newpc,PCFIELDSPLIT,&isfieldsplit));
    if (isfieldsplit) {
      KSP       *ksps;
      PC        ppc,lagpc;
      PetscInt  nn;
      PetscBool ismatis,matisok = PETSC_FALSE,check = PETSC_FALSE;

      /* set the solver for the (0,0) block */
      PetscCall(PCFieldSplitSchurGetSubKSP(newpc,&nn,&ksps));
      if (!nn) { /* not of type PC_COMPOSITE_SCHUR */
        PetscCall(PCFieldSplitGetSubKSP(newpc,&nn,&ksps));
        if (!fake) { /* pass pmat to the pressure solver */
          Mat F;

          PetscCall(KSPGetOperators(ksps[1],&F,NULL));
          PetscCall(KSPSetOperators(ksps[1],F,M));
        }
      } else {
        PetscBool issym;
        Mat       S;

        PetscCall(PCFieldSplitSchurGetS(newpc,&S));

        PetscCall(MatGetOption(newmat,MAT_SYMMETRIC,&issym));
        if (issym) {
          PetscCall(MatSetOption(S,MAT_SYMMETRIC,PETSC_TRUE));
        }
      }
      PetscCall(KSPGetPC(ksps[0],&lagpc));
      PetscCall(PCSetType(lagpc,PCSHELL));
      PetscCall(PCShellSetName(lagpc,"FETI-DP multipliers"));
      PetscCall(PCShellSetContext(lagpc,fetidppc_ctx));
      PetscCall(PCShellSetApply(lagpc,FETIDPPCApply));
      PetscCall(PCShellSetApplyTranspose(lagpc,FETIDPPCApplyTranspose));
      PetscCall(PCShellSetView(lagpc,FETIDPPCView));
      PetscCall(PCShellSetDestroy(lagpc,PCBDDCDestroyFETIDPPC));

      /* Olof's idea: interface Schur complement preconditioner for the mass matrix */
      PetscCall(KSPGetPC(ksps[1],&ppc));
      if (fake) {
        BDDCIPC_ctx    bddcipc_ctx;
        PetscContainer c;

        matisok = PETSC_TRUE;

        /* create inner BDDC solver */
        PetscCall(PetscNew(&bddcipc_ctx));
        PetscCall(PCCreate(comm,&bddcipc_ctx->bddc));
        PetscCall(PCSetType(bddcipc_ctx->bddc,PCBDDC));
        PetscCall(PCSetOperators(bddcipc_ctx->bddc,M,M));
        PetscCall(PetscObjectQuery((PetscObject)pc,"__KSPFETIDP_pCSR",(PetscObject*)&c));
        PetscCall(PetscObjectTypeCompare((PetscObject)M,MATIS,&ismatis));
        if (c && ismatis) {
          Mat      lM;
          PetscInt *csr,n;

          PetscCall(MatISGetLocalMat(M,&lM));
          PetscCall(MatGetSize(lM,&n,NULL));
          PetscCall(PetscContainerGetPointer(c,(void**)&csr));
          PetscCall(PCBDDCSetLocalAdjacencyGraph(bddcipc_ctx->bddc,n,csr,csr + (n + 1),PETSC_COPY_VALUES));
          PetscCall(MatISRestoreLocalMat(M,&lM));
        }
        PetscCall(PCSetOptionsPrefix(bddcipc_ctx->bddc,((PetscObject)ksps[1])->prefix));
        PetscCall(PCSetErrorIfFailure(bddcipc_ctx->bddc,pc->erroriffailure));
        PetscCall(PCSetFromOptions(bddcipc_ctx->bddc));

        /* wrap the interface application */
        PetscCall(PCSetType(ppc,PCSHELL));
        PetscCall(PCShellSetName(ppc,"FETI-DP pressure"));
        PetscCall(PCShellSetContext(ppc,bddcipc_ctx));
        PetscCall(PCShellSetSetUp(ppc,PCSetUp_BDDCIPC));
        PetscCall(PCShellSetApply(ppc,PCApply_BDDCIPC));
        PetscCall(PCShellSetApplyTranspose(ppc,PCApplyTranspose_BDDCIPC));
        PetscCall(PCShellSetView(ppc,PCView_BDDCIPC));
        PetscCall(PCShellSetDestroy(ppc,PCDestroy_BDDCIPC));
      }

      /* determine if we need to assemble M to construct a preconditioner */
      if (!matisok) {
        PetscCall(PetscObjectTypeCompare((PetscObject)M,MATIS,&ismatis));
        PetscCall(PetscObjectTypeCompareAny((PetscObject)ppc,&matisok,PCBDDC,PCJACOBI,PCNONE,PCMG,""));
        if (ismatis && !matisok) {
          PetscCall(MatConvert(M,MATAIJ,MAT_INPLACE_MATRIX,&M));
        }
      }

      /* run the subproblems to check convergence */
      PetscCall(PetscOptionsGetBool(NULL,((PetscObject)newmat)->prefix,"-check_saddlepoint",&check,NULL));
      if (check) {
        PetscInt i;

        for (i=0;i<nn;i++) {
          KSP       kspC;
          PC        pc;
          Mat       F,pF;
          Vec       x,y;
          PetscBool isschur,prec = PETSC_TRUE;

          PetscCall(KSPCreate(PetscObjectComm((PetscObject)ksps[i]),&kspC));
          PetscCall(KSPSetOptionsPrefix(kspC,((PetscObject)ksps[i])->prefix));
          PetscCall(KSPAppendOptionsPrefix(kspC,"check_"));
          PetscCall(KSPGetOperators(ksps[i],&F,&pF));
          PetscCall(PetscObjectTypeCompare((PetscObject)F,MATSCHURCOMPLEMENT,&isschur));
          if (isschur) {
            KSP  kspS,kspS2;
            Mat  A00,pA00,A10,A01,A11;
            char prefix[256];

            PetscCall(MatSchurComplementGetKSP(F,&kspS));
            PetscCall(MatSchurComplementGetSubMatrices(F,&A00,&pA00,&A01,&A10,&A11));
            PetscCall(MatCreateSchurComplement(A00,pA00,A01,A10,A11,&F));
            PetscCall(MatSchurComplementGetKSP(F,&kspS2));
            PetscCall(PetscSNPrintf(prefix,sizeof(prefix),"%sschur_",((PetscObject)kspC)->prefix));
            PetscCall(KSPSetOptionsPrefix(kspS2,prefix));
            PetscCall(KSPGetPC(kspS2,&pc));
            PetscCall(PCSetType(pc,PCKSP));
            PetscCall(PCKSPSetKSP(pc,kspS));
            PetscCall(KSPSetFromOptions(kspS2));
            PetscCall(KSPGetPC(kspS2,&pc));
            PetscCall(PCSetUseAmat(pc,PETSC_TRUE));
          } else {
            PetscCall(PetscObjectReference((PetscObject)F));
          }
          PetscCall(KSPSetFromOptions(kspC));
          PetscCall(PetscOptionsGetBool(NULL,((PetscObject)kspC)->prefix,"-preconditioned",&prec,NULL));
          if (prec)  {
            PetscCall(KSPGetPC(ksps[i],&pc));
            PetscCall(KSPSetPC(kspC,pc));
          }
          PetscCall(KSPSetOperators(kspC,F,pF));
          PetscCall(MatCreateVecs(F,&x,&y));
          PetscCall(VecSetRandom(x,NULL));
          PetscCall(MatMult(F,x,y));
          PetscCall(KSPSolve(kspC,y,x));
          PetscCall(KSPCheckSolve(kspC,pc,x));
          PetscCall(KSPDestroy(&kspC));
          PetscCall(MatDestroy(&F));
          PetscCall(VecDestroy(&x));
          PetscCall(VecDestroy(&y));
        }
      }
      PetscCall(PetscFree(ksps));
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
    PetscUseMethod(pc,"PCBDDCCreateFETIDPOperators_C",(PC,PetscBool,const char*,Mat*,PC*),(pc,fully_redundant,prefix,fetidp_mat,fetidp_pc));
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
  PetscCall(PetscNewLog(pc,&pcbddc));
  pc->data = pcbddc;

  /* create PCIS data structure */
  PetscCall(PCISCreate(pc));

  /* create local graph structure */
  PetscCall(PCBDDCGraphCreate(&pcbddc->mat_graph));

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
  PetscCall(PetscObjectComposeFunction((PetscObject)pc,"PCBDDCSetDiscreteGradient_C",PCBDDCSetDiscreteGradient_BDDC));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc,"PCBDDCSetDivergenceMat_C",PCBDDCSetDivergenceMat_BDDC));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc,"PCBDDCSetChangeOfBasisMat_C",PCBDDCSetChangeOfBasisMat_BDDC));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc,"PCBDDCSetPrimalVerticesLocalIS_C",PCBDDCSetPrimalVerticesLocalIS_BDDC));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc,"PCBDDCSetPrimalVerticesIS_C",PCBDDCSetPrimalVerticesIS_BDDC));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc,"PCBDDCGetPrimalVerticesLocalIS_C",PCBDDCGetPrimalVerticesLocalIS_BDDC));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc,"PCBDDCGetPrimalVerticesIS_C",PCBDDCGetPrimalVerticesIS_BDDC));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc,"PCBDDCSetCoarseningRatio_C",PCBDDCSetCoarseningRatio_BDDC));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc,"PCBDDCSetLevel_C",PCBDDCSetLevel_BDDC));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc,"PCBDDCSetUseExactDirichlet_C",PCBDDCSetUseExactDirichlet_BDDC));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc,"PCBDDCSetLevels_C",PCBDDCSetLevels_BDDC));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc,"PCBDDCSetDirichletBoundaries_C",PCBDDCSetDirichletBoundaries_BDDC));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc,"PCBDDCSetDirichletBoundariesLocal_C",PCBDDCSetDirichletBoundariesLocal_BDDC));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc,"PCBDDCSetNeumannBoundaries_C",PCBDDCSetNeumannBoundaries_BDDC));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc,"PCBDDCSetNeumannBoundariesLocal_C",PCBDDCSetNeumannBoundariesLocal_BDDC));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc,"PCBDDCGetDirichletBoundaries_C",PCBDDCGetDirichletBoundaries_BDDC));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc,"PCBDDCGetDirichletBoundariesLocal_C",PCBDDCGetDirichletBoundariesLocal_BDDC));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc,"PCBDDCGetNeumannBoundaries_C",PCBDDCGetNeumannBoundaries_BDDC));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc,"PCBDDCGetNeumannBoundariesLocal_C",PCBDDCGetNeumannBoundariesLocal_BDDC));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc,"PCBDDCSetDofsSplitting_C",PCBDDCSetDofsSplitting_BDDC));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc,"PCBDDCSetDofsSplittingLocal_C",PCBDDCSetDofsSplittingLocal_BDDC));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc,"PCBDDCSetLocalAdjacencyGraph_C",PCBDDCSetLocalAdjacencyGraph_BDDC));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc,"PCBDDCCreateFETIDPOperators_C",PCBDDCCreateFETIDPOperators_BDDC));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc,"PCBDDCMatFETIDPGetRHS_C",PCBDDCMatFETIDPGetRHS_BDDC));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc,"PCBDDCMatFETIDPGetSolution_C",PCBDDCMatFETIDPGetSolution_BDDC));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc,"PCPreSolveChangeRHS_C",PCPreSolveChangeRHS_BDDC));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc,"PCSetCoordinates_C",PCSetCoordinates_BDDC));
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
  PetscCall(PetscRegisterFinalize(PCBDDCFinalizePackage));

  /* general events */
  PetscCall(PetscLogEventRegister("PCBDDCTopo",PC_CLASSID,&PC_BDDC_Topology[0]));
  PetscCall(PetscLogEventRegister("PCBDDCLKSP",PC_CLASSID,&PC_BDDC_LocalSolvers[0]));
  PetscCall(PetscLogEventRegister("PCBDDCLWor",PC_CLASSID,&PC_BDDC_LocalWork[0]));
  PetscCall(PetscLogEventRegister("PCBDDCCorr",PC_CLASSID,&PC_BDDC_CorrectionSetUp[0]));
  PetscCall(PetscLogEventRegister("PCBDDCASet",PC_CLASSID,&PC_BDDC_ApproxSetUp[0]));
  PetscCall(PetscLogEventRegister("PCBDDCAApp",PC_CLASSID,&PC_BDDC_ApproxApply[0]));
  PetscCall(PetscLogEventRegister("PCBDDCCSet",PC_CLASSID,&PC_BDDC_CoarseSetUp[0]));
  PetscCall(PetscLogEventRegister("PCBDDCCKSP",PC_CLASSID,&PC_BDDC_CoarseSolver[0]));
  PetscCall(PetscLogEventRegister("PCBDDCAdap",PC_CLASSID,&PC_BDDC_AdaptiveSetUp[0]));
  PetscCall(PetscLogEventRegister("PCBDDCScal",PC_CLASSID,&PC_BDDC_Scaling[0]));
  PetscCall(PetscLogEventRegister("PCBDDCSchr",PC_CLASSID,&PC_BDDC_Schurs[0]));
  PetscCall(PetscLogEventRegister("PCBDDCDirS",PC_CLASSID,&PC_BDDC_Solves[0][0]));
  PetscCall(PetscLogEventRegister("PCBDDCNeuS",PC_CLASSID,&PC_BDDC_Solves[0][1]));
  PetscCall(PetscLogEventRegister("PCBDDCCoaS",PC_CLASSID,&PC_BDDC_Solves[0][2]));
  for (i=1;i<PETSC_PCBDDC_MAXLEVELS;i++) {
    char ename[32];

    PetscCall(PetscSNPrintf(ename,sizeof(ename),"PCBDDCTopo l%02d",i));
    PetscCall(PetscLogEventRegister(ename,PC_CLASSID,&PC_BDDC_Topology[i]));
    PetscCall(PetscSNPrintf(ename,sizeof(ename),"PCBDDCLKSP l%02d",i));
    PetscCall(PetscLogEventRegister(ename,PC_CLASSID,&PC_BDDC_LocalSolvers[i]));
    PetscCall(PetscSNPrintf(ename,sizeof(ename),"PCBDDCLWor l%02d",i));
    PetscCall(PetscLogEventRegister(ename,PC_CLASSID,&PC_BDDC_LocalWork[i]));
    PetscCall(PetscSNPrintf(ename,sizeof(ename),"PCBDDCCorr l%02d",i));
    PetscCall(PetscLogEventRegister(ename,PC_CLASSID,&PC_BDDC_CorrectionSetUp[i]));
    PetscCall(PetscSNPrintf(ename,sizeof(ename),"PCBDDCASet l%02d",i));
    PetscCall(PetscLogEventRegister(ename,PC_CLASSID,&PC_BDDC_ApproxSetUp[i]));
    PetscCall(PetscSNPrintf(ename,sizeof(ename),"PCBDDCAApp l%02d",i));
    PetscCall(PetscLogEventRegister(ename,PC_CLASSID,&PC_BDDC_ApproxApply[i]));
    PetscCall(PetscSNPrintf(ename,sizeof(ename),"PCBDDCCSet l%02d",i));
    PetscCall(PetscLogEventRegister(ename,PC_CLASSID,&PC_BDDC_CoarseSetUp[i]));
    PetscCall(PetscSNPrintf(ename,sizeof(ename),"PCBDDCCKSP l%02d",i));
    PetscCall(PetscLogEventRegister(ename,PC_CLASSID,&PC_BDDC_CoarseSolver[i]));
    PetscCall(PetscSNPrintf(ename,sizeof(ename),"PCBDDCAdap l%02d",i));
    PetscCall(PetscLogEventRegister(ename,PC_CLASSID,&PC_BDDC_AdaptiveSetUp[i]));
    PetscCall(PetscSNPrintf(ename,sizeof(ename),"PCBDDCScal l%02d",i));
    PetscCall(PetscLogEventRegister(ename,PC_CLASSID,&PC_BDDC_Scaling[i]));
    PetscCall(PetscSNPrintf(ename,sizeof(ename),"PCBDDCSchr l%02d",i));
    PetscCall(PetscLogEventRegister(ename,PC_CLASSID,&PC_BDDC_Schurs[i]));
    PetscCall(PetscSNPrintf(ename,sizeof(ename),"PCBDDCDirS l%02d",i));
    PetscCall(PetscLogEventRegister(ename,PC_CLASSID,&PC_BDDC_Solves[i][0]));
    PetscCall(PetscSNPrintf(ename,sizeof(ename),"PCBDDCNeuS l%02d",i));
    PetscCall(PetscLogEventRegister(ename,PC_CLASSID,&PC_BDDC_Solves[i][1]));
    PetscCall(PetscSNPrintf(ename,sizeof(ename),"PCBDDCCoaS l%02d",i));
    PetscCall(PetscLogEventRegister(ename,PC_CLASSID,&PC_BDDC_Solves[i][2]));
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
