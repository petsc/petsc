
#include <petscfe.h>
#include <petscdmplex.h>
#include <petsc/private/hashmap.h>
#include <petsc/private/dmpleximpl.h>
#include <petsc/private/petscfeimpl.h>

const char help[] = "Test PETSCDUALSPACELAGRANGE\n";

typedef struct _PetscHashLagKey
{
  PetscInt  dim;
  PetscInt  order;
  PetscInt  formDegree;
  PetscBool trimmed;
  PetscInt  tensor;
  PetscBool continuous;
} PetscHashLagKey;

#define PetscHashLagKeyHash(key) \
  PetscHashCombine(PetscHashCombine(PetscHashCombine(PetscHashInt((key).dim), \
                                                     PetscHashInt((key).order)), \
                                    PetscHashInt((key).formDegree)), \
                   PetscHashCombine(PetscHashCombine(PetscHashInt((key).trimmed), \
                                                     PetscHashInt((key).tensor)), \
                                    PetscHashInt((key).continuous)))

#define PetscHashLagKeyEqual(k1,k2) \
  (((k1).dim == (k2).dim) ? \
   ((k1).order == (k2).order) ? \
   ((k1).formDegree == (k2).formDegree) ? \
   ((k1).trimmed == (k2).trimmed) ? \
   ((k1).tensor == (k2).tensor) ? \
   ((k1).continuous == (k2).continuous) : 0 : 0 : 0 : 0 : 0)

PETSC_HASH_MAP(HashLag, PetscHashLagKey, PetscInt, PetscHashLagKeyHash, PetscHashLagKeyEqual, 0)

static PetscErrorCode ExpectedNumDofs_Total(PetscInt dim, PetscInt order, PetscInt formDegree, PetscBool trimmed, PetscInt tensor, PetscInt nCopies, PetscInt *nDofs);
static PetscErrorCode ExpectedNumDofs_Interior(PetscInt dim, PetscInt order, PetscInt formDegree, PetscBool trimmed, PetscInt tensor, PetscInt nCopies, PetscInt *nDofs);

static PetscErrorCode ExpectedNumDofs_Total(PetscInt dim, PetscInt order, PetscInt formDegree, PetscBool trimmed, PetscInt tensor, PetscInt nCopies, PetscInt *nDofs)
{
  PetscFunctionBegin;
  formDegree = PetscAbsInt(formDegree);
  /* see femtable.org for the source of most of these values */
  *nDofs = -1;
  if (tensor == 0) { /* simplex spaces */
    if (trimmed) {
      PetscInt rnchooserk;
      PetscInt rkm1choosek;

      PetscCall(PetscDTBinomialInt(order + dim, order + formDegree, &rnchooserk));
      PetscCall(PetscDTBinomialInt(order + formDegree - 1, formDegree, &rkm1choosek));
      *nDofs = rnchooserk * rkm1choosek * nCopies;
    } else {
      PetscInt rnchooserk;
      PetscInt rkchoosek;

      PetscCall(PetscDTBinomialInt(order + dim, order + formDegree, &rnchooserk));
      PetscCall(PetscDTBinomialInt(order + formDegree, formDegree, &rkchoosek));
      *nDofs = rnchooserk * rkchoosek * nCopies;
    }
  } else if (tensor == 1) { /* hypercubes */
    if (trimmed) {
      PetscInt nchoosek;
      PetscInt rpowk, rp1pownmk;

      PetscCall(PetscDTBinomialInt(dim, formDegree, &nchoosek));
      rpowk = PetscPowInt(order, formDegree);
      rp1pownmk = PetscPowInt(order + 1, dim - formDegree);
      *nDofs = nchoosek * rpowk * rp1pownmk * nCopies;
    } else {
      PetscInt nchoosek;
      PetscInt rp1pown;

      PetscCall(PetscDTBinomialInt(dim, formDegree, &nchoosek));
      rp1pown = PetscPowInt(order + 1, dim);
      *nDofs = nchoosek * rp1pown * nCopies;
    }
  } else { /* prism */
    PetscInt tracek = 0;
    PetscInt tracekm1 = 0;
    PetscInt fiber0 = 0;
    PetscInt fiber1 = 0;

    if (formDegree < dim) {
      PetscCall(ExpectedNumDofs_Total(dim - 1, order, formDegree, trimmed, 0, 1, &tracek));
      PetscCall(ExpectedNumDofs_Total(1, order, 0, trimmed, 0, 1, &fiber0));
    }
    if (formDegree > 0) {
      PetscCall(ExpectedNumDofs_Total(dim - 1, order, formDegree - 1, trimmed, 0, 1, &tracekm1));
      PetscCall(ExpectedNumDofs_Total(1, order, 1, trimmed, 0, 1, &fiber1));
    }
    *nDofs = (tracek * fiber0 + tracekm1 * fiber1) * nCopies;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode ExpectedNumDofs_Interior(PetscInt dim, PetscInt order, PetscInt formDegree, PetscBool trimmed,
                                               PetscInt tensor, PetscInt nCopies, PetscInt *nDofs)
{
  PetscFunctionBegin;
  formDegree = PetscAbsInt(formDegree);
  /* see femtable.org for the source of most of these values */
  *nDofs = -1;
  if (tensor == 0) { /* simplex spaces */
    if (trimmed) {
      if (order + formDegree > dim) {
        PetscInt eorder = order + formDegree - dim - 1;
        PetscInt eformDegree = dim - formDegree;
        PetscInt rnchooserk;
        PetscInt rkchoosek;

        PetscCall(PetscDTBinomialInt(eorder + dim, eorder + eformDegree, &rnchooserk));
        PetscCall(PetscDTBinomialInt(eorder + eformDegree, eformDegree, &rkchoosek));
        *nDofs = rnchooserk * rkchoosek * nCopies;
      } else {
        *nDofs = 0;
      }

    } else {
      if (order + formDegree > dim) {
        PetscInt eorder = order + formDegree - dim;
        PetscInt eformDegree = dim - formDegree;
        PetscInt rnchooserk;
        PetscInt rkm1choosek;

        PetscCall(PetscDTBinomialInt(eorder + dim, eorder + eformDegree, &rnchooserk));
        PetscCall(PetscDTBinomialInt(eorder + eformDegree - 1, eformDegree, &rkm1choosek));
        *nDofs = rnchooserk * rkm1choosek * nCopies;
      } else {
        *nDofs = 0;
      }
    }
  } else if (tensor == 1) { /* hypercubes */
    if (dim < 2) {
      PetscCall(ExpectedNumDofs_Interior(dim, order, formDegree, trimmed, 0, nCopies, nDofs));
    } else {
      PetscInt tracek = 0;
      PetscInt tracekm1 = 0;
      PetscInt fiber0 = 0;
      PetscInt fiber1 = 0;

      if (formDegree < dim) {
        PetscCall(ExpectedNumDofs_Interior(dim - 1, order, formDegree, trimmed, dim > 2 ? 1 : 0, 1, &tracek));
        PetscCall(ExpectedNumDofs_Interior(1, order, 0, trimmed, 0, 1, &fiber0));
      }
      if (formDegree > 0) {
        PetscCall(ExpectedNumDofs_Interior(dim - 1, order, formDegree - 1, trimmed, dim > 2 ? 1 : 0, 1, &tracekm1));
        PetscCall(ExpectedNumDofs_Interior(1, order, 1, trimmed, 0, 1, &fiber1));
      }
      *nDofs = (tracek * fiber0 + tracekm1 * fiber1) * nCopies;
    }
  } else { /* prism */
    PetscInt tracek = 0;
    PetscInt tracekm1 = 0;
    PetscInt fiber0 = 0;
    PetscInt fiber1 = 0;

    if (formDegree < dim) {
      PetscCall(ExpectedNumDofs_Interior(dim - 1, order, formDegree, trimmed, 0, 1, &tracek));
      PetscCall(ExpectedNumDofs_Interior(1, order, 0, trimmed, 0, 1, &fiber0));
    }
    if (formDegree > 0) {
      PetscCall(ExpectedNumDofs_Interior(dim - 1, order, formDegree - 1, trimmed, 0, 1, &tracekm1));
      PetscCall(ExpectedNumDofs_Interior(1, order, 1, trimmed, 0, 1, &fiber1));
    }
    *nDofs = (tracek * fiber0 + tracekm1 * fiber1) * nCopies;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode testLagrange(PetscHashLag lagTable, DM K, PetscInt dim, PetscInt order, PetscInt formDegree, PetscBool trimmed, PetscInt tensor, PetscBool continuous, PetscInt nCopies)
{
  PetscDualSpace  sp;
  MPI_Comm        comm = PETSC_COMM_SELF;
  PetscInt        Nk;
  PetscHashLagKey key;
  PetscHashIter   iter;
  PetscBool       missing;
  PetscInt        spdim, spintdim, exspdim, exspintdim;

  PetscFunctionBegin;
  PetscCall(PetscDTBinomialInt(dim, PetscAbsInt(formDegree), &Nk));
  PetscCall(PetscDualSpaceCreate(comm, &sp));
  PetscCall(PetscDualSpaceSetType(sp, PETSCDUALSPACELAGRANGE));
  PetscCall(PetscDualSpaceSetDM(sp, K));
  PetscCall(PetscDualSpaceSetOrder(sp, order));
  PetscCall(PetscDualSpaceSetFormDegree(sp, formDegree));
  PetscCall(PetscDualSpaceSetNumComponents(sp, nCopies * Nk));
  PetscCall(PetscDualSpaceLagrangeSetContinuity(sp, continuous));
  PetscCall(PetscDualSpaceLagrangeSetTensor(sp, (PetscBool) tensor));
  PetscCall(PetscDualSpaceLagrangeSetTrimmed(sp, trimmed));
  PetscCall(PetscInfo(NULL, "Input: dim %D, order %D, trimmed %D, tensor %D, continuous %D, formDegree %D, nCopies %D\n", dim, order, (PetscInt) trimmed, tensor, (PetscInt) continuous, formDegree, nCopies));
  PetscCall(ExpectedNumDofs_Total(dim, order, formDegree, trimmed, tensor, nCopies, &exspdim));
  if (continuous && dim > 0 && order > 0) {
    PetscCall(ExpectedNumDofs_Interior(dim, order, formDegree, trimmed, tensor, nCopies, &exspintdim));
  } else {
    exspintdim = exspdim;
  }
  PetscCall(PetscDualSpaceSetUp(sp));
  PetscCall(PetscDualSpaceGetDimension(sp, &spdim));
  PetscCall(PetscDualSpaceGetInteriorDimension(sp, &spintdim));
  PetscCheck(spdim == exspdim,PETSC_COMM_SELF, PETSC_ERR_PLIB, "Expected dual space dimension %D, got %D", exspdim, spdim);
  PetscCheck(spintdim == exspintdim,PETSC_COMM_SELF, PETSC_ERR_PLIB, "Expected dual space interior dimension %D, got %D", exspintdim, spintdim);
  key.dim = dim;
  key.formDegree = formDegree;
  PetscCall(PetscDualSpaceGetOrder(sp, &key.order));
  PetscCall(PetscDualSpaceLagrangeGetContinuity(sp, &key.continuous));
  if (tensor == 2) {
    key.tensor = 2;
  } else {
    PetscBool bTensor;

    PetscCall(PetscDualSpaceLagrangeGetTensor(sp, &bTensor));
    key.tensor = bTensor;
  }
  PetscCall(PetscDualSpaceLagrangeGetTrimmed(sp, &key.trimmed));
  PetscCall(PetscInfo(NULL, "After setup:  order %D, trimmed %D, tensor %D, continuous %D\n", key.order, (PetscInt) key.trimmed, key.tensor, (PetscInt) key.continuous));
  PetscCall(PetscHashLagPut(lagTable, key, &iter, &missing));
  if (missing) {
    DMPolytopeType type;

    PetscCall(DMPlexGetCellType(K, 0, &type));
    PetscCall(PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_SELF, "New space: %s, order %D, trimmed %D, tensor %D, continuous %D, form degree %D\n", DMPolytopeTypes[type], order, (PetscInt) trimmed, tensor, (PetscInt) continuous, formDegree));
    PetscCall(PetscViewerASCIIPushTab(PETSC_VIEWER_STDOUT_SELF));
    {
      PetscQuadrature intNodes, allNodes;
      Mat intMat, allMat;
      MatInfo info;
      PetscInt i, j, nodeIdxDim, nodeVecDim, nNodes;
      const PetscInt *nodeIdx;
      const PetscReal *nodeVec;

      PetscDualSpace_Lag *lag = (PetscDualSpace_Lag *) sp->data;

      PetscCall(PetscLagNodeIndicesGetData_Internal(lag->allNodeIndices, &nodeIdxDim, &nodeVecDim, &nNodes, &nodeIdx, &nodeVec));
      PetscCheck(nodeVecDim == Nk,PETSC_COMM_SELF, PETSC_ERR_PLIB, "Incorrect nodeVecDim");
      PetscCheck(nNodes == spdim,PETSC_COMM_SELF, PETSC_ERR_PLIB, "Incorrect nNodes");

      PetscCall(PetscDualSpaceGetAllData(sp, &allNodes, &allMat));

      PetscCall(PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_SELF, "All nodes:\n"));
      PetscCall(PetscViewerASCIIPushTab(PETSC_VIEWER_STDOUT_SELF));
      PetscCall(PetscQuadratureView(allNodes, PETSC_VIEWER_STDOUT_SELF));
      PetscCall(PetscViewerASCIIPopTab(PETSC_VIEWER_STDOUT_SELF));
      PetscCall(PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_SELF, "All node indices:\n"));
      for (i = 0; i < spdim; i++) {
        PetscCall(PetscPrintf(PETSC_COMM_SELF, "("));
        for (j = 0; j < nodeIdxDim; j++) {
          PetscCall(PetscPrintf(PETSC_COMM_SELF, " %D,", nodeIdx[i * nodeIdxDim + j]));
        }
        PetscCall(PetscPrintf(PETSC_COMM_SELF, "): ["));
        for (j = 0; j < nodeVecDim; j++) {
          PetscCall(PetscPrintf(PETSC_COMM_SELF, " %g,", (double) nodeVec[i * nodeVecDim + j]));
        }
        PetscCall(PetscPrintf(PETSC_COMM_SELF, "]\n"));
      }

      PetscCall(MatGetInfo(allMat, MAT_LOCAL, &info));
      PetscCall(PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_SELF, "All matrix: %D nonzeros\n", (PetscInt) info.nz_used));

      PetscCall(PetscDualSpaceGetInteriorData(sp, &intNodes, &intMat));
      if (intMat && intMat != allMat) {
        PetscInt        intNodeIdxDim, intNodeVecDim, intNnodes;
        const PetscInt  *intNodeIdx;
        const PetscReal *intNodeVec;
        PetscBool       same;

        PetscCall(PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_SELF, "Interior nodes:\n"));
        PetscCall(PetscViewerASCIIPushTab(PETSC_VIEWER_STDOUT_SELF));
        PetscCall(PetscQuadratureView(intNodes, PETSC_VIEWER_STDOUT_SELF));
        PetscCall(PetscViewerASCIIPopTab(PETSC_VIEWER_STDOUT_SELF));

        PetscCall(MatGetInfo(intMat, MAT_LOCAL, &info));
        PetscCall(PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_SELF, "Interior matrix: %D nonzeros\n", (PetscInt) info.nz_used));
        PetscCall(PetscLagNodeIndicesGetData_Internal(lag->intNodeIndices, &intNodeIdxDim, &intNodeVecDim, &intNnodes, &intNodeIdx, &intNodeVec));
        PetscCheckFalse(intNodeIdxDim != nodeIdxDim || intNodeVecDim != nodeVecDim,PETSC_COMM_SELF, PETSC_ERR_PLIB, "Interior node indices not the same shale as all node indices");
        PetscCheck(intNnodes == spintdim,PETSC_COMM_SELF, PETSC_ERR_PLIB, "Incorrect interior nNodes");
        PetscCall(PetscArraycmp(intNodeIdx, nodeIdx, nodeIdxDim * intNnodes, &same));
        PetscCheck(same,PETSC_COMM_SELF, PETSC_ERR_PLIB, "Interior node indices not the same as start of all node indices");
        PetscCall(PetscArraycmp(intNodeVec, nodeVec, nodeVecDim * intNnodes, &same));
        PetscCheck(same,PETSC_COMM_SELF, PETSC_ERR_PLIB, "Interior node vectors not the same as start of all node vectors");
      } else if (intMat) {
        PetscCall(PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_SELF, "Interior data is the same as all data\n"));
        PetscCheck(intNodes == allNodes,PETSC_COMM_SELF, PETSC_ERR_PLIB, "Interior nodes should be the same as all nodes");
        PetscCheck(lag->intNodeIndices == lag->allNodeIndices,PETSC_COMM_SELF, PETSC_ERR_PLIB, "Interior node indices should be the same as all node indices");
      }
    }
    if (dim <= 2 && spintdim) {
      PetscInt numFaces, o;

      {
        DMPolytopeType ct;
        /* The number of arrangements is no longer based on the number of faces */
        PetscCall(DMPlexGetCellType(K, 0, &ct));
        numFaces = DMPolytopeTypeGetNumArrangments(ct) / 2;
      }
      for (o = -numFaces; o < numFaces; ++o) {
        Mat symMat;

        PetscCall(PetscDualSpaceCreateInteriorSymmetryMatrix_Lagrange(sp, o, &symMat));
        PetscCall(PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_SELF, "Interior node symmetry matrix for orientation %D:\n", o));
        PetscCall(PetscViewerASCIIPushTab(PETSC_VIEWER_STDOUT_SELF));
        PetscCall(MatView(symMat, PETSC_VIEWER_STDOUT_SELF));
        PetscCall(PetscViewerASCIIPopTab(PETSC_VIEWER_STDOUT_SELF));
        PetscCall(MatDestroy(&symMat));
      }
    }
    PetscCall(PetscViewerASCIIPopTab(PETSC_VIEWER_STDOUT_SELF));
  }
  PetscCall(PetscDualSpaceDestroy(&sp));
  PetscFunctionReturn(0);
}

int main (int argc, char **argv)
{
  PetscInt        dim;
  PetscHashLag    lagTable;
  PetscInt        tensorCell;
  PetscInt        order, ordermin, ordermax;
  PetscBool       continuous;
  PetscBool       trimmed;
  DM              dm;

  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  dim = 3;
  tensorCell = 0;
  continuous = PETSC_FALSE;
  trimmed = PETSC_FALSE;
  PetscOptionsBegin(PETSC_COMM_WORLD,"","Options for PETSCDUALSPACELAGRANGE test","none");
  PetscCall(PetscOptionsRangeInt("-dim", "The spatial dimension","ex1.c",dim,&dim,NULL,0,3));
  PetscCall(PetscOptionsRangeInt("-tensor", "(0) simplex (1) hypercube (2) wedge","ex1.c",tensorCell,&tensorCell,NULL,0,2));
  PetscCall(PetscOptionsBool("-continuous", "Whether the dual space has continuity","ex1.c",continuous,&continuous,NULL));
  PetscCall(PetscOptionsBool("-trimmed", "Whether the dual space matches a trimmed polynomial space","ex1.c",trimmed,&trimmed,NULL));
  PetscOptionsEnd();
  PetscCall(PetscHashLagCreate(&lagTable));

  if (tensorCell < 2) {
    PetscCall(DMPlexCreateReferenceCell(PETSC_COMM_SELF, DMPolytopeTypeSimpleShape(dim, (PetscBool) !tensorCell), &dm));
  } else {
    PetscCall(DMPlexCreateReferenceCell(PETSC_COMM_SELF, DM_POLYTOPE_TRI_PRISM, &dm));
  }
  ordermin = trimmed ? 1 : 0;
  ordermax = tensorCell == 2 ? 4 : tensorCell == 1 ? 3 : dim + 2;
  for (order = ordermin; order <= ordermax; order++) {
    PetscInt formDegree;

    for (formDegree = PetscMin(0,-dim+1); formDegree <= dim; formDegree++) {
      PetscInt nCopies;

      for (nCopies = 1; nCopies <= 3; nCopies++) {
        PetscCall(testLagrange(lagTable, dm, dim, order, formDegree, trimmed, (PetscBool) tensorCell, continuous, nCopies));
      }
    }
  }
  PetscCall(DMDestroy(&dm));
  PetscCall(PetscHashLagDestroy(&lagTable));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

 test:
   suffix: 0
   args: -dim 0

 test:
   suffix: 1_discontinuous_full
   args: -dim 1 -continuous 0 -trimmed 0

 test:
   suffix: 1_continuous_full
   args: -dim 1 -continuous 1 -trimmed 0

 test:
   suffix: 2_simplex_discontinuous_full
   args: -dim 2 -tensor 0 -continuous 0 -trimmed 0

 test:
   suffix: 2_simplex_continuous_full
   args: -dim 2 -tensor 0 -continuous 1 -trimmed 0

 test:
   suffix: 2_tensor_discontinuous_full
   args: -dim 2 -tensor 1 -continuous 0 -trimmed 0

 test:
   suffix: 2_tensor_continuous_full
   args: -dim 2 -tensor 1 -continuous 1 -trimmed 0

 test:
   suffix: 3_simplex_discontinuous_full
   args: -dim 3 -tensor 0 -continuous 0 -trimmed 0

 test:
   suffix: 3_simplex_continuous_full
   args: -dim 3 -tensor 0 -continuous 1 -trimmed 0

 test:
   suffix: 3_tensor_discontinuous_full
   args: -dim 3 -tensor 1 -continuous 0 -trimmed 0

 test:
   suffix: 3_tensor_continuous_full
   args: -dim 3 -tensor 1 -continuous 1 -trimmed 0

 test:
   suffix: 3_wedge_discontinuous_full
   args: -dim 3 -tensor 2 -continuous 0 -trimmed 0

 test:
   suffix: 3_wedge_continuous_full
   args: -dim 3 -tensor 2 -continuous 1 -trimmed 0

 test:
   suffix: 1_discontinuous_trimmed
   args: -dim 1 -continuous 0 -trimmed 1

 test:
   suffix: 1_continuous_trimmed
   args: -dim 1 -continuous 1 -trimmed 1

 test:
   suffix: 2_simplex_discontinuous_trimmed
   args: -dim 2 -tensor 0 -continuous 0 -trimmed 1

 test:
   suffix: 2_simplex_continuous_trimmed
   args: -dim 2 -tensor 0 -continuous 1 -trimmed 1

 test:
   suffix: 2_tensor_discontinuous_trimmed
   args: -dim 2 -tensor 1 -continuous 0 -trimmed 1

 test:
   suffix: 2_tensor_continuous_trimmed
   args: -dim 2 -tensor 1 -continuous 1 -trimmed 1

 test:
   suffix: 3_simplex_discontinuous_trimmed
   args: -dim 3 -tensor 0 -continuous 0 -trimmed 1

 test:
   suffix: 3_simplex_continuous_trimmed
   args: -dim 3 -tensor 0 -continuous 1 -trimmed 1

 test:
   suffix: 3_tensor_discontinuous_trimmed
   args: -dim 3 -tensor 1 -continuous 0 -trimmed 1

 test:
   suffix: 3_tensor_continuous_trimmed
   args: -dim 3 -tensor 1 -continuous 1 -trimmed 1

 test:
   suffix: 3_wedge_discontinuous_trimmed
   args: -dim 3 -tensor 2 -continuous 0 -trimmed 1

 test:
   suffix: 3_wedge_continuous_trimmed
   args: -dim 3 -tensor 2 -continuous 1 -trimmed 1

TEST*/
