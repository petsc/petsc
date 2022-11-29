#include <petsc/private/pcmgimpl.h>
#include <petsc/private/pcbddcimpl.h>
#include <petsc/private/pcbddcprivateimpl.h>

static PetscErrorCode PCMGGDSWSetUp(PC pc, PetscInt l, DM dm, KSP smooth, PetscInt Nc, Mat A, PetscInt *ns, Mat **sA_IG_n, KSP **sksp_n, IS **sI_n, IS **sG_n, Mat **sGf_n, IS **sGi_n, IS **sGiM_n)
{
  KSP                   *sksp;
  PC                     pcbddc = NULL, smoothpc;
  PC_BDDC               *ipcbddc;
  PC_IS                 *ipcis;
  Mat                   *sA_IG, *sGf, cmat, lA;
  ISLocalToGlobalMapping l2g;
  IS                    *sI, *sG, *sGi, *sGiM, cref;
  PCBDDCSubSchurs        sub_schurs = NULL;
  PCBDDCGraph            graph;
  const char            *prefix;
  const PetscScalar     *tdata;
  PetscScalar           *data, *cdata;
  PetscReal              tol = 0.0, otol;
  const PetscInt        *ia, *ja;
  PetscInt              *ccii, *cridx;
  PetscInt               i, j, ngct, ng, dbg = 0, odbg, minmax[2] = {0, PETSC_MAX_INT}, ominmax[2], vsize;
  PetscBool              flg, userdefined = PETSC_TRUE, reuse_solver = PETSC_TRUE, reduced = PETSC_FALSE;

  PetscFunctionBegin;
  PetscCall(MatGetBlockSize(A, &vsize));
  PetscCall(KSPGetOptionsPrefix(smooth, &prefix));
  PetscOptionsBegin(PetscObjectComm((PetscObject)smooth), prefix, "GDSW options", "PC");
  PetscCall(PetscOptionsReal("-gdsw_tolerance", "Tolerance for eigenvalue problem", NULL, tol, &tol, NULL));
  PetscCall(PetscOptionsBool("-gdsw_userdefined", "Use user-defined functions in addition to those adaptively generated", NULL, userdefined, &userdefined, NULL));
  PetscCall(PetscOptionsIntArray("-gdsw_minmax", "Minimum and maximum number of basis functions per connected component for adaptive GDSW", NULL, minmax, (i = 2, &i), NULL));
  PetscCall(PetscOptionsInt("-gdsw_vertex_size", "Connected components smaller or equal to vertex size will be considered as vertices", NULL, vsize, &vsize, NULL));
  PetscCall(PetscOptionsBool("-gdsw_reuse", "Reuse interior solver from Schur complement computations", NULL, reuse_solver, &reuse_solver, NULL));
  PetscCall(PetscOptionsBool("-gdsw_reduced", "Reduced GDSW", NULL, reduced, &reduced, NULL));
  PetscCall(PetscOptionsInt("-gdsw_debug", "Debug output", NULL, dbg, &dbg, NULL));
  PetscOptionsEnd();

  PetscCall(PetscObjectTypeCompare((PetscObject)A, MATIS, &flg));
  if (!flg) {
    MatNullSpace nnsp;

    PetscCall(MatGetNearNullSpace(A, &nnsp));
    PetscObjectReference((PetscObject)nnsp);
    PetscCall(MatConvert(A, MATIS, MAT_INITIAL_MATRIX, &A));
    PetscCall(MatSetNearNullSpace(A, nnsp));
    PetscCall(MatNullSpaceDestroy(&nnsp));
  } else PetscCall(PetscObjectReference((PetscObject)A));

  /* TODO Multi sub */
  *ns = 1;
  PetscCall(PetscMalloc1(*ns, &sA_IG));
  PetscCall(PetscMalloc1(*ns, &sksp));
  PetscCall(PetscMalloc1(*ns, &sI));
  PetscCall(PetscMalloc1(*ns, &sG));
  PetscCall(PetscMalloc1(*ns, &sGf));
  PetscCall(PetscMalloc1(*ns, &sGi));
  PetscCall(PetscMalloc1(*ns, &sGiM));
  *sA_IG_n = sA_IG;
  *sksp_n  = sksp;
  *sI_n    = sI;
  *sG_n    = sG;
  *sGf_n   = sGf;
  *sGi_n   = sGi;
  *sGiM_n  = sGiM;

  /* submatrices and solvers */
  PetscCall(KSPGetPC(smooth, &smoothpc));
  PetscCall(PetscObjectTypeCompareAny((PetscObject)smoothpc, &flg, PCBDDC, ""));
  if (!flg) {
    Mat smoothA;

    PetscCall(PCGetOperators(smoothpc, &smoothA, NULL));
    PetscCall(PCCreate(PetscObjectComm((PetscObject)A), &pcbddc));
    PetscCall(PCSetType(pcbddc, PCBDDC));
    PetscCall(PCSetOperators(pcbddc, smoothA, A));
    PetscCall(PCISSetUp(pcbddc, PETSC_TRUE, PETSC_FALSE));
  } else {
    PetscCall(PetscObjectReference((PetscObject)smoothpc));
    pcbddc = smoothpc;
  }
  ipcis   = (PC_IS *)pcbddc->data;
  ipcbddc = (PC_BDDC *)pcbddc->data;
  PetscCall(PetscObjectReference((PetscObject)ipcis->A_IB));
  PetscCall(PetscObjectReference((PetscObject)ipcis->is_I_global));
  PetscCall(PetscObjectReference((PetscObject)ipcis->is_B_global));
  sA_IG[0] = ipcis->A_IB;
  sI[0]    = ipcis->is_I_global;
  sG[0]    = ipcis->is_B_global;

  PetscCall(KSPCreate(PetscObjectComm((PetscObject)ipcis->A_II), &sksp[0]));
  PetscCall(KSPSetOperators(sksp[0], ipcis->A_II, ipcis->pA_II));
  PetscCall(KSPSetOptionsPrefix(sksp[0], prefix));
  PetscCall(KSPAppendOptionsPrefix(sksp[0], "gdsw_"));
  PetscCall(KSPSetFromOptions(sksp[0]));

  /* analyze interface */
  PetscCall(MatISGetLocalMat(A, &lA));
  graph = ipcbddc->mat_graph;
  if (!flg) {
    PetscInt N;

    PetscCall(MatISGetLocalToGlobalMapping(A, &l2g, NULL));
    PetscCall(MatGetSize(A, &N, NULL));
    graph->commsizelimit = 0; /* don't use the COMM_SELF variant of the graph */
    PetscCall(PCBDDCGraphInit(graph, l2g, N, PETSC_MAX_INT));
    PetscCall(MatGetRowIJ(lA, 0, PETSC_TRUE, PETSC_FALSE, &graph->nvtxs_csr, (const PetscInt **)&graph->xadj, (const PetscInt **)&graph->adjncy, &flg));
    PetscCall(PCBDDCGraphSetUp(graph, vsize, NULL, NULL, 0, NULL, NULL));
    PetscCall(MatRestoreRowIJ(lA, 0, PETSC_TRUE, PETSC_FALSE, &graph->nvtxs_csr, (const PetscInt **)&graph->xadj, (const PetscInt **)&graph->adjncy, &flg));
    PetscCall(PCBDDCGraphComputeConnectedComponents(graph));
  }
  l2g = graph->l2gmap;
  if (reduced) {
    PetscContainer        gcand;
    PCBDDCGraphCandidates cand;
    PetscErrorCode (*rgdsw)(DM, PetscInt *, IS **);

    PetscCall(PetscObjectQueryFunction((PetscObject)dm, "DMComputeLocalRGDSWSets", &rgdsw));
    PetscCheck(rgdsw, PETSC_COMM_SELF, PETSC_ERR_SUP, "Not supported");
    PetscCall(PetscNew(&cand));
    PetscCall((*rgdsw)(dm, &cand->nfc, &cand->Faces));
    /* filter interior (if any) and guarantee IS are ordered by global numbering */
    for (i = 0; i < cand->nfc; i++) {
      IS is, is2;

      PetscCall(ISLocalToGlobalMappingApplyIS(l2g, cand->Faces[i], &is));
      PetscCall(ISDestroy(&cand->Faces[i]));
      PetscCall(ISSort(is));
      PetscCall(ISGlobalToLocalMappingApplyIS(l2g, IS_GTOLM_DROP, is, &is2));
      PetscCall(ISDestroy(&is));
      PetscCall(ISGlobalToLocalMappingApplyIS(ipcis->BtoNmap, IS_GTOLM_DROP, is2, &is));
      PetscCall(ISDestroy(&is2));
      PetscCall(ISLocalToGlobalMappingApplyIS(ipcis->BtoNmap, is, &cand->Faces[i]));
      PetscCall(ISDestroy(&is));
    }
    PetscCall(PetscContainerCreate(PETSC_COMM_SELF, &gcand));
    PetscCall(PetscContainerSetPointer(gcand, cand));
    PetscCall(PetscContainerSetUserDestroy(gcand, PCBDDCDestroyGraphCandidatesIS));
    PetscCall(PetscObjectCompose((PetscObject)l2g, "_PCBDDCGraphCandidatesIS", (PetscObject)gcand));
    PetscCall(PetscContainerDestroy(&gcand));
  }

  /* interface functions */
  otol                           = ipcbddc->adaptive_threshold[1];
  odbg                           = ipcbddc->dbg_flag;
  ominmax[0]                     = ipcbddc->adaptive_nmin;
  ominmax[1]                     = ipcbddc->adaptive_nmax;
  ipcbddc->adaptive_threshold[1] = tol;
  ipcbddc->dbg_flag              = dbg;
  ipcbddc->adaptive_nmin         = minmax[0];
  ipcbddc->adaptive_nmax         = minmax[1];
  if (tol != 0.0) { /* adaptive */
    Mat lS;

    PetscCall(MatCreateSchurComplement(ipcis->A_II, ipcis->pA_II, ipcis->A_IB, ipcis->A_BI, ipcis->A_BB, &lS));
    PetscCall(KSPGetOptionsPrefix(sksp[0], &prefix));
    PetscCall(PCBDDCSubSchursCreate(&sub_schurs));
    PetscCall(PCBDDCSubSchursInit(sub_schurs, prefix, ipcis->is_I_local, ipcis->is_B_local, graph, ipcis->BtoNmap, PETSC_FALSE, PETSC_TRUE));
    if (userdefined) PetscCall(PCBDDCComputeFakeChange(pcbddc, PETSC_FALSE, graph, NULL, &cmat, &cref, NULL, &flg));
    else {
      cmat = NULL;
      cref = NULL;
    }
    PetscCall(PCBDDCSubSchursSetUp(sub_schurs, lA, lS, PETSC_TRUE, NULL, NULL, -1, NULL, PETSC_TRUE, reuse_solver, PETSC_FALSE, 0, NULL, NULL, cmat, cref));
    PetscCall(MatDestroy(&lS));
    PetscCall(MatDestroy(&cmat));
    PetscCall(ISDestroy(&cref));
    if (sub_schurs->reuse_solver) {
      PetscCall(KSPSetPC(sksp[0], sub_schurs->reuse_solver->interior_solver));
      PetscCall(PCDestroy(&sub_schurs->reuse_solver->interior_solver));
      sub_schurs->reuse_solver = NULL;
    }
  }
  PetscCall(PCBDDCComputeFakeChange(pcbddc, PETSC_TRUE, graph, sub_schurs, &cmat, &cref, &sGiM[0], NULL));
  PetscCall(PCBDDCSubSchursDestroy(&sub_schurs));
  ipcbddc->adaptive_threshold[1] = otol;
  ipcbddc->dbg_flag              = odbg;
  ipcbddc->adaptive_nmin         = ominmax[0];
  ipcbddc->adaptive_nmax         = ominmax[1];

  PetscCall(ISLocalToGlobalMappingApplyIS(l2g, cref, &sGi[0]));
  PetscCall(ISDestroy(&cref));

  PetscCall(MatSeqAIJGetArrayRead(cmat, &tdata));
  PetscCall(MatGetRowIJ(cmat, 0, PETSC_FALSE, PETSC_FALSE, &ngct, &ia, &ja, &flg));
  PetscCheck(flg, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Error in MatGetRowIJ");

  PetscCall(PetscMalloc1(ngct + 1, &ccii));
  PetscCall(PetscMalloc1(ia[ngct], &cridx));
  PetscCall(PetscMalloc1(ia[ngct], &cdata));

  PetscCall(PetscArraycpy(ccii, ia, ngct + 1));
  PetscCall(PetscArraycpy(cdata, tdata, ia[ngct]));
  PetscCall(ISGlobalToLocalMappingApply(ipcis->BtoNmap, IS_GTOLM_DROP, ia[ngct], ja, &i, cridx));
  PetscCheck(i == ia[ngct], PETSC_COMM_SELF, PETSC_ERR_PLIB, "Error in G2L");

  PetscCall(MatRestoreRowIJ(cmat, 0, PETSC_FALSE, PETSC_FALSE, &i, &ia, &ja, &flg));
  PetscCall(MatSeqAIJRestoreArrayRead(cmat, &tdata));
  PetscCall(MatDestroy(&cmat));

  /* populate dense matrix */
  PetscCall(ISGetLocalSize(sG[0], &ng));
  PetscCall(MatCreateSeqDense(PETSC_COMM_SELF, ng, ngct, NULL, &sGf[0]));
  PetscCall(MatDenseGetArrayWrite(sGf[0], &data));
  for (i = 0; i < ngct; i++)
    for (j = ccii[i]; j < ccii[i + 1]; j++) data[ng * i + cridx[j]] = cdata[j];
  PetscCall(MatDenseRestoreArrayWrite(sGf[0], &data));

  PetscCall(PetscFree(cdata));
  PetscCall(PetscFree(ccii));
  PetscCall(PetscFree(cridx));
  PetscCall(PCDestroy(&pcbddc));
  PetscCall(MatDestroy(&A));
  PetscFunctionReturn(0);
}

PetscErrorCode PCMGGDSWCreateCoarseSpace_Private(PC pc, PetscInt l, DM dm, KSP smooth, PetscInt Nc, Mat guess, Mat *cspace)
{
  KSP            *sksp;
  Mat             A, *sA_IG, *sGf, preallocator;
  IS              Gidx, GidxMult, cG;
  IS             *sI, *sG, *sGi, *sGiM;
  const PetscInt *cidx;
  PetscInt        NG, ns, n, i, c, rbs, cbs[2];
  PetscBool       flg;
  MatType         ptype;

  PetscFunctionBegin;
  *cspace = NULL;
  if (!l) PetscFunctionReturn(0);
  if (pc->useAmat) {
    PetscCall(KSPGetOperatorsSet(smooth, &flg, NULL));
    PetscCheck(flg, PetscObjectComm((PetscObject)smooth), PETSC_ERR_ORDER, "Amat not set");
    PetscCall(KSPGetOperators(smooth, &A, NULL));
  } else {
    PetscCall(KSPGetOperatorsSet(smooth, NULL, &flg));
    PetscCheck(flg, PetscObjectComm((PetscObject)smooth), PETSC_ERR_ORDER, "Pmat not set");
    PetscCall(KSPGetOperators(smooth, NULL, &A));
  }

  /* Setup (also setup smoother here) */
  if (!pc->setupcalled) PetscCall(KSPSetFromOptions(smooth));
  PetscCall(KSPSetUp(smooth));
  PetscCall(KSPSetUpOnBlocks(smooth));
  PetscCall(PCMGGDSWSetUp(pc, l, dm, smooth, Nc, A, &ns, &sA_IG, &sksp, &sI, &sG, &sGf, &sGi, &sGiM));

  /* Number GDSW basis functions */
  PetscCall(ISConcatenate(PetscObjectComm((PetscObject)A), ns, sGi, &Gidx));
  PetscCall(ISConcatenate(PetscObjectComm((PetscObject)A), ns, sGiM, &GidxMult));
  PetscCall(ISRenumber(Gidx, GidxMult, &NG, &cG));
  PetscCall(ISDestroy(&Gidx));

  /* Detect column block size */
  PetscCall(ISGetMinMax(GidxMult, &cbs[0], &cbs[1]));
  PetscCall(PetscGlobalMinMaxInt(PetscObjectComm((PetscObject)A), cbs, cbs));
  PetscCall(ISDestroy(&GidxMult));

  /* Construct global interpolation matrix */
  PetscCall(MatGetLocalSize(A, NULL, &n));
  PetscCall(MatCreate(PetscObjectComm((PetscObject)A), &preallocator));
  PetscCall(MatSetSizes(preallocator, n, PETSC_DECIDE, PETSC_DECIDE, NG));
  PetscCall(MatSetType(preallocator, MATPREALLOCATOR));
  PetscCall(MatSetUp(preallocator));
  PetscCall(ISGetIndices(cG, &cidx));
  for (i = 0, c = 0; i < ns; i++) {
    const PetscInt *ri, *rg;
    PetscInt        nri, nrg, ncg;

    PetscCall(ISGetLocalSize(sI[i], &nri));
    PetscCall(ISGetLocalSize(sG[i], &nrg));
    PetscCall(ISGetIndices(sI[i], &ri));
    PetscCall(ISGetIndices(sG[i], &rg));
    PetscCall(MatGetSize(sGf[i], NULL, &ncg));
    PetscCall(MatSetValues(preallocator, nri, ri, ncg, cidx + c, NULL, INSERT_VALUES));
    PetscCall(MatSetValues(preallocator, nrg, rg, ncg, cidx + c, NULL, INSERT_VALUES));
    PetscCall(ISRestoreIndices(sI[i], &ri));
    PetscCall(ISRestoreIndices(sG[i], &rg));
  }
  PetscCall(MatAssemblyBegin(preallocator, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(preallocator, MAT_FINAL_ASSEMBLY));

  ptype = MATAIJ;
  if (PetscDefined(HAVE_DEVICE)) {
    PetscCall(MatBoundToCPU(A, &flg));
    if (!flg) {
      VecType vtype;
      char   *found;

      PetscCall(MatGetVecType(A, &vtype));
      PetscCall(PetscStrstr(vtype, "cuda", &found));
      if (found) ptype = MATAIJCUSPARSE;
    }
  }
  PetscCall(MatCreate(PetscObjectComm((PetscObject)A), cspace));
  PetscCall(MatSetSizes(*cspace, n, PETSC_DECIDE, PETSC_DECIDE, NG));
  PetscCall(MatSetType(*cspace, ptype));
  PetscCall(MatGetBlockSizes(A, NULL, &rbs));
  PetscCall(MatSetBlockSizes(*cspace, rbs, cbs[0] == cbs[1] ? cbs[0] : 1));
  PetscCall(MatPreallocatorPreallocate(preallocator, PETSC_FALSE, *cspace));
  PetscCall(MatDestroy(&preallocator));
  PetscCall(MatSetOption(*cspace, MAT_ROW_ORIENTED, PETSC_FALSE));

  for (i = 0, c = 0; i < ns; i++) {
    Mat                X, Y;
    const PetscScalar *v;
    const PetscInt    *ri, *rg;
    PetscInt           nri, nrg, ncg;

    PetscCall(MatMatMult(sA_IG[i], sGf[i], MAT_INITIAL_MATRIX, PETSC_DEFAULT, &Y));
    PetscCall(MatScale(Y, -1.0));
    PetscCall(MatDuplicate(Y, MAT_DO_NOT_COPY_VALUES, &X));
    PetscCall(KSPMatSolve(sksp[i], Y, X));

    PetscCall(ISGetLocalSize(sI[i], &nri));
    PetscCall(ISGetLocalSize(sG[i], &nrg));
    PetscCall(ISGetIndices(sI[i], &ri));
    PetscCall(ISGetIndices(sG[i], &rg));
    PetscCall(MatGetSize(sGf[i], NULL, &ncg));

    PetscCall(MatDenseGetArrayRead(X, &v));
    PetscCall(MatSetValues(*cspace, nri, ri, ncg, cidx + c, v, INSERT_VALUES));
    PetscCall(MatDenseRestoreArrayRead(X, &v));
    PetscCall(MatDenseGetArrayRead(sGf[i], &v));
    PetscCall(MatSetValues(*cspace, nrg, rg, ncg, cidx + c, v, INSERT_VALUES));
    PetscCall(MatDenseRestoreArrayRead(sGf[i], &v));
    PetscCall(ISRestoreIndices(sI[i], &ri));
    PetscCall(ISRestoreIndices(sG[i], &rg));
    PetscCall(MatDestroy(&Y));
    PetscCall(MatDestroy(&X));
  }
  PetscCall(ISRestoreIndices(cG, &cidx));
  PetscCall(ISDestroy(&cG));
  PetscCall(MatAssemblyBegin(*cspace, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(*cspace, MAT_FINAL_ASSEMBLY));

  for (i = 0; i < ns; i++) {
    PetscCall(KSPDestroy(&sksp[i]));
    PetscCall(ISDestroy(&sI[i]));
    PetscCall(ISDestroy(&sG[i]));
    PetscCall(ISDestroy(&sGi[i]));
    PetscCall(ISDestroy(&sGiM[i]));
    PetscCall(MatDestroy(&sGf[i]));
    PetscCall(MatDestroy(&sA_IG[i]));
  }
  PetscCall(PetscFree(sksp));
  PetscCall(PetscFree(sI));
  PetscCall(PetscFree(sG));
  PetscCall(PetscFree(sGi));
  PetscCall(PetscFree(sGiM));
  PetscCall(PetscFree(sGf));
  PetscCall(PetscFree(sA_IG));
  PetscFunctionReturn(0);
}
