# Changes: Development

<!---
% STYLE GUIDELINES:
% * Capitalize sentences
% * Use imperative, e.g., Add, Improve, Change, etc.
% * Don't use a period (.) at the end of entries
% * If multiple sentences are needed, use a period or semicolon to divide sentences, but not at the end of the final sentence
--->

## General


## Configure/Build


## Sys


## Event Logging


## PetscViewer


## PetscDraw


## AO


## IS


## VecScatter / PetscSF


## PF


## Vec


## PetscSection


## PetscPartitioner


## Mat

- Add `MATPRODUCT_PtAP` support for `MATDIAGONAL`

## MatCoarsen


## PC


## KSP


## SNES


## SNESLineSearch


## TS


## TAO

- Add `TaoGetDM()` and `TaoSetDM()`

## TaoTerm


## PetscRegressor


## PetscDA


## DM


## DMSwarm

- Add `DMSwarmProjectFields()` and `DMSwarmProjectGradientFields()`
- Add `DMSwarmSort` class
- Add `DMSwarmSortDestroy()` and `DMSwarmSortView()`
- Allow `DMSwarmCellDMSetSort()` to take in `NULL` and clear the sort
- Add `DMSwarmPreallocateMassMatrix()` and `DMSwarmFillMassMatrix()`

## DMPlex

- Add `DMPlexSetClosurePermutationLexicographic()`
- Add `DMPlexDrawCell()`
- Add `DMPlexLabelCompleteStar()`
- Add `DMPlexVecGetClosureAtDepth()`
- Add an extra communicator argument to `DMPlexFilter()` to allow extracting local meshes
- Add `DMPlexCopyFlags()`
- Add `DMPlexRebalanceSharedLabelPoints()`
- Add `DMPlexCheckLabel()` and `DMPlexReconcileLabel()`
- Change CGNS viewer to use multi-component read/write interface for better performance

## FE/FV


## DMNetwork


## DMStag


## DT


## Fortran
