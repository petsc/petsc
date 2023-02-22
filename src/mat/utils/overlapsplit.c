/*
 * Increase the overlap of a 'big' subdomain across several processor cores
 *
 * Author: Fande Kong <fdkong.jd@gmail.com>
 */

#include <petscsf.h>
#include <petsc/private/matimpl.h>

/*
 * Increase overlap for the sub-matrix across sub communicator
 * sub-matrix could be a graph or numerical matrix
 * */
PetscErrorCode MatIncreaseOverlapSplit_Single(Mat mat, IS *is, PetscInt ov)
{
  PetscInt        i, nindx, *indices_sc, *indices_ov, localsize, *localsizes_sc, localsize_tmp;
  PetscInt       *indices_ov_rd, nroots, nleaves, *localoffsets, *indices_recv, *sources_sc, *sources_sc_rd;
  const PetscInt *indices;
  PetscMPIInt     srank, ssize, issamecomm, k, grank;
  IS              is_sc, allis_sc, partitioning;
  MPI_Comm        gcomm, dcomm, scomm;
  PetscSF         sf;
  PetscSFNode    *remote;
  Mat            *smat;
  MatPartitioning part;

  PetscFunctionBegin;
  /* get a sub communicator before call individual MatIncreaseOverlap
   * since the sub communicator may be changed.
   * */
  PetscCall(PetscObjectGetComm((PetscObject)(*is), &dcomm));
  /* make a copy before the original one is deleted */
  PetscCall(PetscCommDuplicate(dcomm, &scomm, NULL));
  /* get a global communicator, where mat should be a global matrix  */
  PetscCall(PetscObjectGetComm((PetscObject)mat, &gcomm));
  PetscUseTypeMethod(mat, increaseoverlap, 1, is, ov);
  PetscCallMPI(MPI_Comm_compare(gcomm, scomm, &issamecomm));
  /* if the sub-communicator is the same as the global communicator,
   * user does not want to use a sub-communicator
   * */
  if (issamecomm == MPI_IDENT || issamecomm == MPI_CONGRUENT) {
    PetscCall(PetscCommDestroy(&scomm));
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  /* if the sub-communicator is petsc_comm_self,
   * user also does not care the sub-communicator
   * */
  PetscCallMPI(MPI_Comm_compare(scomm, PETSC_COMM_SELF, &issamecomm));
  if (issamecomm == MPI_IDENT || issamecomm == MPI_CONGRUENT) {
    PetscCall(PetscCommDestroy(&scomm));
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  PetscCallMPI(MPI_Comm_rank(scomm, &srank));
  PetscCallMPI(MPI_Comm_size(scomm, &ssize));
  PetscCallMPI(MPI_Comm_rank(gcomm, &grank));
  /* create a new IS based on sub-communicator
   * since the old IS is often based on petsc_comm_self
   * */
  PetscCall(ISGetLocalSize(*is, &nindx));
  PetscCall(PetscMalloc1(nindx, &indices_sc));
  PetscCall(ISGetIndices(*is, &indices));
  PetscCall(PetscArraycpy(indices_sc, indices, nindx));
  PetscCall(ISRestoreIndices(*is, &indices));
  /* we do not need any more */
  PetscCall(ISDestroy(is));
  /* create a index set based on the sub communicator  */
  PetscCall(ISCreateGeneral(scomm, nindx, indices_sc, PETSC_OWN_POINTER, &is_sc));
  /* gather all indices within  the sub communicator */
  PetscCall(ISAllGather(is_sc, &allis_sc));
  PetscCall(ISDestroy(&is_sc));
  /* gather local sizes */
  PetscCall(PetscMalloc1(ssize, &localsizes_sc));
  /* get individual local sizes for all index sets */
  PetscCallMPI(MPI_Gather(&nindx, 1, MPIU_INT, localsizes_sc, 1, MPIU_INT, 0, scomm));
  /* only root does these computations */
  if (!srank) {
    /* get local size for the big index set */
    PetscCall(ISGetLocalSize(allis_sc, &localsize));
    PetscCall(PetscCalloc2(localsize, &indices_ov, localsize, &sources_sc));
    PetscCall(PetscCalloc2(localsize, &indices_ov_rd, localsize, &sources_sc_rd));
    PetscCall(ISGetIndices(allis_sc, &indices));
    PetscCall(PetscArraycpy(indices_ov, indices, localsize));
    PetscCall(ISRestoreIndices(allis_sc, &indices));
    PetscCall(ISDestroy(&allis_sc));
    /* assign corresponding sources */
    localsize_tmp = 0;
    for (k = 0; k < ssize; k++) {
      for (i = 0; i < localsizes_sc[k]; i++) sources_sc[localsize_tmp++] = k;
    }
    /* record where indices come from */
    PetscCall(PetscSortIntWithArray(localsize, indices_ov, sources_sc));
    /* count local sizes for reduced indices */
    PetscCall(PetscArrayzero(localsizes_sc, ssize));
    /* initialize the first entity */
    if (localsize) {
      indices_ov_rd[0] = indices_ov[0];
      sources_sc_rd[0] = sources_sc[0];
      localsizes_sc[sources_sc[0]]++;
    }
    localsize_tmp = 1;
    /* remove duplicate integers */
    for (i = 1; i < localsize; i++) {
      if (indices_ov[i] != indices_ov[i - 1]) {
        indices_ov_rd[localsize_tmp]   = indices_ov[i];
        sources_sc_rd[localsize_tmp++] = sources_sc[i];
        localsizes_sc[sources_sc[i]]++;
      }
    }
    PetscCall(PetscFree2(indices_ov, sources_sc));
    PetscCall(PetscCalloc1(ssize + 1, &localoffsets));
    for (k = 0; k < ssize; k++) localoffsets[k + 1] = localoffsets[k] + localsizes_sc[k];
    nleaves = localoffsets[ssize];
    PetscCall(PetscArrayzero(localoffsets, ssize + 1));
    nroots = localsizes_sc[srank];
    PetscCall(PetscMalloc1(nleaves, &remote));
    for (i = 0; i < nleaves; i++) {
      remote[i].rank  = sources_sc_rd[i];
      remote[i].index = localoffsets[sources_sc_rd[i]]++;
    }
    PetscCall(PetscFree(localoffsets));
  } else {
    PetscCall(ISDestroy(&allis_sc));
    /* Allocate a 'zero' pointer to avoid using uninitialized variable  */
    PetscCall(PetscCalloc1(0, &remote));
    nleaves       = 0;
    indices_ov_rd = NULL;
    sources_sc_rd = NULL;
  }
  /* scatter sizes to everybody */
  PetscCallMPI(MPI_Scatter(localsizes_sc, 1, MPIU_INT, &nroots, 1, MPIU_INT, 0, scomm));
  PetscCall(PetscFree(localsizes_sc));
  PetscCall(PetscCalloc1(nroots, &indices_recv));
  /* set data back to every body */
  PetscCall(PetscSFCreate(scomm, &sf));
  PetscCall(PetscSFSetType(sf, PETSCSFBASIC));
  PetscCall(PetscSFSetFromOptions(sf));
  PetscCall(PetscSFSetGraph(sf, nroots, nleaves, NULL, PETSC_OWN_POINTER, remote, PETSC_OWN_POINTER));
  PetscCall(PetscSFReduceBegin(sf, MPIU_INT, indices_ov_rd, indices_recv, MPI_REPLACE));
  PetscCall(PetscSFReduceEnd(sf, MPIU_INT, indices_ov_rd, indices_recv, MPI_REPLACE));
  PetscCall(PetscSFDestroy(&sf));
  PetscCall(PetscFree2(indices_ov_rd, sources_sc_rd));
  PetscCall(ISCreateGeneral(scomm, nroots, indices_recv, PETSC_OWN_POINTER, &is_sc));
  PetscCall(MatCreateSubMatricesMPI(mat, 1, &is_sc, &is_sc, MAT_INITIAL_MATRIX, &smat));
  PetscCall(ISDestroy(&allis_sc));
  /* create a partitioner to repartition the sub-matrix */
  PetscCall(MatPartitioningCreate(scomm, &part));
  PetscCall(MatPartitioningSetAdjacency(part, smat[0]));
#if defined(PETSC_HAVE_PARMETIS)
  /* if there exists a ParMETIS installation, we try to use ParMETIS
   * because a repartition routine possibly work better
   * */
  PetscCall(MatPartitioningSetType(part, MATPARTITIONINGPARMETIS));
  /* try to use reparition function, instead of partition function */
  PetscCall(MatPartitioningParmetisSetRepartition(part));
#else
  /* we at least provide a default partitioner to rebalance the computation  */
  PetscCall(MatPartitioningSetType(part, MATPARTITIONINGAVERAGE));
#endif
  /* user can pick up any partitioner by using an option */
  PetscCall(MatPartitioningSetFromOptions(part));
  PetscCall(MatPartitioningApply(part, &partitioning));
  PetscCall(MatPartitioningDestroy(&part));
  PetscCall(MatDestroy(&(smat[0])));
  PetscCall(PetscFree(smat));
  /* get local rows including  overlap */
  PetscCall(ISBuildTwoSided(partitioning, is_sc, is));
  PetscCall(ISDestroy(&is_sc));
  PetscCall(ISDestroy(&partitioning));
  PetscCall(PetscCommDestroy(&scomm));
  PetscFunctionReturn(PETSC_SUCCESS);
}
