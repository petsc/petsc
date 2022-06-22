
#include <petsc/private/petscimpl.h>
#include <petscis.h> /*I "petscis.h" I*/

/* This is the bitonic merge that works on non-power-of-2 sizes found at http://www.iti.fh-flensburg.de/lang/algorithmen/sortieren/bitonic/oddn.htm */
static PetscErrorCode PetscParallelSortInt_Bitonic_Merge(MPI_Comm comm, PetscMPIInt tag, PetscMPIInt rankStart, PetscMPIInt rankEnd, PetscMPIInt rank, PetscMPIInt n, PetscInt keys[], PetscInt buffer[], PetscBool forward)
{
  PetscInt       diff;
  PetscInt       split, mid, partner;

  PetscFunctionBegin;
  diff = rankEnd - rankStart;
  if (diff <= 0) PetscFunctionReturn(0);
  if (diff == 1) {
    if (forward) {
      PetscCall(PetscSortInt((PetscInt) n, keys));
    } else {
      PetscCall(PetscSortReverseInt((PetscInt) n, keys));
    }
    PetscFunctionReturn(0);
  }
  split = 1;
  while (2 * split < diff) split *= 2;
  mid = rankStart + split;
  if (rank < mid) {
    partner = rank + split;
  } else {
    partner = rank - split;
  }
  if (partner < rankEnd) {
    PetscMPIInt i;

    PetscCallMPI(MPI_Sendrecv(keys, n, MPIU_INT, partner, tag, buffer, n, MPIU_INT, partner, tag, comm, MPI_STATUS_IGNORE));
    if ((rank < partner) == (forward == PETSC_TRUE)) {
      for (i = 0; i < n; i++) {
        keys[i] = (keys[i] <= buffer[i]) ? keys[i] : buffer[i];
      }
    } else {
      for (i = 0; i < n; i++) {
        keys[i] = (keys[i] > buffer[i]) ? keys[i] : buffer[i];
      }
    }
  }
  /* divide and conquer */
  if (rank < mid) {
    PetscCall(PetscParallelSortInt_Bitonic_Merge(comm, tag, rankStart, mid, rank, n, keys, buffer, forward));
  } else {
    PetscCall(PetscParallelSortInt_Bitonic_Merge(comm, tag, mid, rankEnd, rank, n, keys, buffer, forward));
  }
  PetscFunctionReturn(0);
}

/* This is the bitonic sort that works on non-power-of-2 sizes found at http://www.iti.fh-flensburg.de/lang/algorithmen/sortieren/bitonic/oddn.htm */
static PetscErrorCode PetscParallelSortInt_Bitonic_Recursive(MPI_Comm comm, PetscMPIInt tag, PetscMPIInt rankStart, PetscMPIInt rankEnd, PetscMPIInt rank, PetscMPIInt n, PetscInt keys[], PetscInt buffer[], PetscBool forward)
{
  PetscInt       diff;
  PetscInt       mid;

  PetscFunctionBegin;
  diff = rankEnd - rankStart;
  if (diff <= 0) PetscFunctionReturn(0);
  if (diff == 1) {
    if (forward) {
      PetscCall(PetscSortInt(n, keys));
    } else {
      PetscCall(PetscSortReverseInt(n, keys));
    }
    PetscFunctionReturn(0);
  }
  mid = rankStart + diff / 2;
  /* divide and conquer */
  if (rank < mid) {
    PetscCall(PetscParallelSortInt_Bitonic_Recursive(comm, tag, rankStart, mid, rank, n, keys, buffer, (PetscBool) !forward));
  } else {
    PetscCall(PetscParallelSortInt_Bitonic_Recursive(comm, tag, mid, rankEnd, rank, n, keys, buffer, forward));
  }
  /* bitonic merge */
  PetscCall(PetscParallelSortInt_Bitonic_Merge(comm, tag, rankStart, rankEnd, rank, n, keys, buffer, forward));
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscParallelSortInt_Bitonic(MPI_Comm comm, PetscInt n, PetscInt keys[])
{
  PetscMPIInt size, rank, tag, mpin;
  PetscInt       *buffer;

  PetscFunctionBegin;
  PetscValidIntPointer(keys, 3);
  PetscCall(PetscCommGetNewTag(comm, &tag));
  PetscCallMPI(MPI_Comm_size(comm, &size));
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  PetscCall(PetscMPIIntCast(n, &mpin));
  PetscCall(PetscMalloc1(n, &buffer));
  PetscCall(PetscParallelSortInt_Bitonic_Recursive(comm, tag, 0, size, rank, mpin, keys, buffer, PETSC_TRUE));
  PetscCall(PetscFree(buffer));
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscParallelSampleSelect(PetscLayout mapin, PetscLayout mapout, PetscInt keysin[], PetscInt *outpivots[])
{
  PetscMPIInt    size, rank;
  PetscInt       *pivots, *finalpivots, i;
  PetscInt       non_empty, my_first, count;
  PetscMPIInt    *keys_per, max_keys_per;

  PetscFunctionBegin;
  PetscCallMPI(MPI_Comm_size(mapin->comm, &size));
  PetscCallMPI(MPI_Comm_rank(mapin->comm, &rank));

  /* Choose P - 1 pivots that would be ideal for the distribution on this process */
  PetscCall(PetscMalloc1(size - 1, &pivots));
  for (i = 0; i < size - 1; i++) {
    PetscInt index;

    if (!mapin->n) {
      /* if this rank is empty, put "infinity" in the list.  each process knows how many empty
       * processes are in the layout, so those values will be ignored from the end of the sorted
       * pivots */
      pivots[i] = PETSC_MAX_INT;
    } else {
      /* match the distribution to the desired output described by mapout by getting the index
       * that is approximately the appropriate fraction through the list */
      index = ((PetscReal) mapout->range[i + 1]) * ((PetscReal) mapin->n) / ((PetscReal) mapout->N);
      index = PetscMin(index, (mapin->n - 1));
      index = PetscMax(index, 0);
      pivots[i] = keysin[index];
    }
  }
  /* sort the pivots in parallel */
  PetscCall(PetscParallelSortInt_Bitonic(mapin->comm, size - 1, pivots));
  if (PetscDefined(USE_DEBUG)) {
    PetscBool sorted;

    PetscCall(PetscParallelSortedInt(mapin->comm, size - 1, pivots, &sorted));
    PetscCheck(sorted,mapin->comm, PETSC_ERR_PLIB, "bitonic sort failed");
  }

  /* if there are Z nonempty processes, we have (P - 1) * Z real pivots, and we want to select
   * at indices Z - 1, 2*Z - 1, ... (P - 1) * Z - 1 */
  non_empty = size;
  for (i = 0; i < size; i++) if (mapout->range[i] == mapout->range[i+1]) non_empty--;
  PetscCall(PetscCalloc1(size + 1, &keys_per));
  my_first = -1;
  if (non_empty) {
    for (i = 0; i < size - 1; i++) {
      size_t sample = (i + 1) * non_empty - 1;
      size_t sample_rank = sample / (size - 1);

      keys_per[sample_rank]++;
      if (my_first < 0 && (PetscMPIInt) sample_rank == rank) {
        my_first = (PetscInt) (sample - sample_rank * (size - 1));
      }
    }
  }
  for (i = 0, max_keys_per = 0; i < size; i++) max_keys_per = PetscMax(keys_per[i], max_keys_per);
  PetscCall(PetscMalloc1(size * max_keys_per, &finalpivots));
  /* now that we know how many pivots each process will provide, gather the selected pivots at the start of the array
   * and allgather them */
  for (i = 0; i < keys_per[rank]; i++) pivots[i] = pivots[my_first + i * non_empty];
  for (i = keys_per[rank]; i < max_keys_per; i++) pivots[i] = PETSC_MAX_INT;
  PetscCallMPI(MPI_Allgather(pivots, max_keys_per, MPIU_INT, finalpivots, max_keys_per, MPIU_INT, mapin->comm));
  for (i = 0, count = 0; i < size; i++) {
    PetscInt j;

    for (j = 0; j < max_keys_per; j++) {
      if (j < keys_per[i]) {
        finalpivots[count++] = finalpivots[i * max_keys_per + j];
      }
    }
  }
  *outpivots = finalpivots;
  PetscCall(PetscFree(keys_per));
  PetscCall(PetscFree(pivots));
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscParallelRedistribute(PetscLayout map, PetscInt n, PetscInt arrayin[], PetscInt arrayout[])
{
  PetscMPIInt  size, rank;
  PetscInt     myOffset, nextOffset;
  PetscInt     i;
  PetscMPIInt  total, filled;
  PetscMPIInt  sender, nfirst, nsecond;
  PetscMPIInt  firsttag, secondtag;
  MPI_Request  firstreqrcv;
  MPI_Request *firstreqs;
  MPI_Request *secondreqs;
  MPI_Status   firststatus;

  PetscFunctionBegin;
  PetscCallMPI(MPI_Comm_size(map->comm, &size));
  PetscCallMPI(MPI_Comm_rank(map->comm, &rank));
  PetscCall(PetscCommGetNewTag(map->comm, &firsttag));
  PetscCall(PetscCommGetNewTag(map->comm, &secondtag));
  myOffset = 0;
  PetscCall(PetscMalloc2(size, &firstreqs, size, &secondreqs));
  PetscCallMPI(MPI_Scan(&n, &nextOffset, 1, MPIU_INT, MPI_SUM, map->comm));
  myOffset = nextOffset - n;
  total = map->range[rank + 1] - map->range[rank];
  if (total > 0) {
    PetscCallMPI(MPI_Irecv(arrayout, total, MPIU_INT, MPI_ANY_SOURCE, firsttag, map->comm, &firstreqrcv));
  }
  for (i = 0, nsecond = 0, nfirst = 0; i < size; i++) {
    PetscInt itotal;
    PetscInt overlap, oStart, oEnd;

    itotal = map->range[i + 1] - map->range[i];
    if (itotal <= 0) continue;
    oStart = PetscMax(myOffset, map->range[i]);
    oEnd   = PetscMin(nextOffset, map->range[i + 1]);
    overlap = oEnd - oStart;
    if (map->range[i] >= myOffset && map->range[i] < nextOffset) {
      /* send first message */
      PetscCallMPI(MPI_Isend(&arrayin[map->range[i] - myOffset], overlap, MPIU_INT, i, firsttag, map->comm, &(firstreqs[nfirst++])));
    } else if (overlap > 0) {
      /* send second message */
      PetscCallMPI(MPI_Isend(&arrayin[oStart - myOffset], overlap, MPIU_INT, i, secondtag, map->comm, &(secondreqs[nsecond++])));
    } else if (overlap == 0 && myOffset > map->range[i] && myOffset < map->range[i + 1]) {
      /* send empty second message */
      PetscCallMPI(MPI_Isend(&arrayin[oStart - myOffset], 0, MPIU_INT, i, secondtag, map->comm, &(secondreqs[nsecond++])));
    }
  }
  filled = 0;
  sender = -1;
  if (total > 0) {
    PetscCallMPI(MPI_Wait(&firstreqrcv, &firststatus));
    sender = firststatus.MPI_SOURCE;
    PetscCallMPI(MPI_Get_count(&firststatus, MPIU_INT, &filled));
  }
  while (filled < total) {
    PetscMPIInt mfilled;
    MPI_Status stat;

    sender++;
    PetscCallMPI(MPI_Recv(&arrayout[filled], total - filled, MPIU_INT, sender, secondtag, map->comm, &stat));
    PetscCallMPI(MPI_Get_count(&stat, MPIU_INT, &mfilled));
    filled += mfilled;
  }
  PetscCallMPI(MPI_Waitall(nfirst, firstreqs, MPI_STATUSES_IGNORE));
  PetscCallMPI(MPI_Waitall(nsecond, secondreqs, MPI_STATUSES_IGNORE));
  PetscCall(PetscFree2(firstreqs, secondreqs));
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscParallelSortInt_Samplesort(PetscLayout mapin, PetscLayout mapout, PetscInt keysin[], PetscInt keysout[])
{
  PetscMPIInt    size, rank;
  PetscInt       *pivots = NULL, *buffer;
  PetscInt       i, j;
  PetscMPIInt    *keys_per_snd, *keys_per_rcv, *offsets_snd, *offsets_rcv, nrecv;

  PetscFunctionBegin;
  PetscCallMPI(MPI_Comm_size(mapin->comm, &size));
  PetscCallMPI(MPI_Comm_rank(mapin->comm, &rank));
  PetscCall(PetscMalloc4(size, &keys_per_snd, size, &keys_per_rcv, size + 1, &offsets_snd, size + 1, &offsets_rcv));
  /* sort locally */
  PetscCall(PetscSortInt(mapin->n, keysin));
  /* get P - 1 pivots */
  PetscCall(PetscParallelSampleSelect(mapin, mapout, keysin, &pivots));
  /* determine which entries in the sorted array go to which other processes based on the pivots */
  for (i = 0, j = 0; i < size - 1; i++) {
    PetscInt prev = j;

    while ((j < mapin->n) && (keysin[j] < pivots[i])) j++;
    offsets_snd[i] = prev;
    keys_per_snd[i] = j - prev;
  }
  offsets_snd[size - 1] = j;
  keys_per_snd[size - 1] = mapin->n - j;
  offsets_snd[size] = mapin->n;
  /* get the incoming sizes */
  PetscCallMPI(MPI_Alltoall(keys_per_snd, 1, MPI_INT, keys_per_rcv, 1, MPI_INT, mapin->comm));
  offsets_rcv[0] = 0;
  for (i = 0; i < size; i++) {
    offsets_rcv[i+1] = offsets_rcv[i] + keys_per_rcv[i];
  }
  nrecv = offsets_rcv[size];
  /* all to all exchange */
  PetscCall(PetscMalloc1(nrecv, &buffer));
  PetscCallMPI(MPI_Alltoallv(keysin, keys_per_snd, offsets_snd, MPIU_INT, buffer, keys_per_rcv, offsets_rcv, MPIU_INT, mapin->comm));
  PetscCall(PetscFree(pivots));
  PetscCall(PetscFree4(keys_per_snd, keys_per_rcv, offsets_snd, offsets_rcv));

  /* local sort */
  PetscCall(PetscSortInt(nrecv, buffer));
#if defined(PETSC_USE_DEBUG)
  {
    PetscBool sorted;

    PetscCall(PetscParallelSortedInt(mapin->comm, nrecv, buffer, &sorted));
    PetscCheck(sorted,mapin->comm, PETSC_ERR_PLIB, "samplesort (pre-redistribute) sort failed");
  }
#endif

  /* redistribute to the desired order */
  PetscCall(PetscParallelRedistribute(mapout, nrecv, buffer, keysout));
  PetscCall(PetscFree(buffer));
  PetscFunctionReturn(0);
}

/*@
  PetscParallelSortInt - Globally sort a distributed array of integers

  Collective

  Input Parameters:
+ mapin - PetscLayout describing the distribution of the input keys
. mapout - PetscLayout describing the distribution of the output keys
- keysin - the pre-sorted array of integers

  Output Parameter:
. keysout - the array in which the sorted integers will be stored.  If mapin == mapout, then keysin may be equal to keysout.

  Level: developer

  Notes: This implements a distributed samplesort, which, with local array sizes n_in and n_out, global size N, and global number of processes P, does:

  - sorts locally
  - chooses pivots by sorting (in parallel) (P-1) pivot suggestions from each process using bitonic sort and allgathering a subset of (P-1) of those
  - using to the pivots to repartition the keys by all-to-all exchange
  - sorting the repartitioned keys locally (the array is now globally sorted, but does not match the mapout layout)
  - redistributing to match the mapout layout

  If keysin != keysout, then keysin will not be changed during PetscParallelSortInt.

.seealso: `PetscParallelSortedInt()`
@*/
PetscErrorCode PetscParallelSortInt(PetscLayout mapin, PetscLayout mapout, PetscInt keysin[], PetscInt keysout[])
{
  PetscMPIInt    size;
  PetscMPIInt    result;
  PetscInt       *keysincopy = NULL;

  PetscFunctionBegin;
  PetscValidPointer(mapin, 1);
  PetscValidPointer(mapout, 2);
  PetscCallMPI(MPI_Comm_compare(mapin->comm, mapout->comm, &result));
  PetscCheck(result == MPI_IDENT || result == MPI_CONGRUENT,mapin->comm, PETSC_ERR_ARG_NOTSAMECOMM, "layouts are not on the same communicator");
  PetscCall(PetscLayoutSetUp(mapin));
  PetscCall(PetscLayoutSetUp(mapout));
  if (mapin->n) PetscValidIntPointer(keysin, 3);
  if (mapout->n) PetscValidIntPointer(keysout, 4);
  PetscCheck(mapin->N == mapout->N,mapin->comm, PETSC_ERR_ARG_SIZ, "Input and output layouts have different global sizes (%" PetscInt_FMT " != %" PetscInt_FMT ")", mapin->N, mapout->N);
  PetscCallMPI(MPI_Comm_size(mapin->comm, &size));
  if (size == 1) {
    if (keysout != keysin) {
      PetscCall(PetscMemcpy(keysout, keysin, mapin->n * sizeof(PetscInt)));
    }
    PetscCall(PetscSortInt(mapout->n, keysout));
    if (size == 1) PetscFunctionReturn(0);
  }
  if (keysout != keysin) {
    PetscCall(PetscMalloc1(mapin->n, &keysincopy));
    PetscCall(PetscMemcpy(keysincopy, keysin, mapin->n * sizeof(PetscInt)));
    keysin = keysincopy;
  }
  PetscCall(PetscParallelSortInt_Samplesort(mapin, mapout, keysin, keysout));
#if defined(PETSC_USE_DEBUG)
  {
    PetscBool sorted;

    PetscCall(PetscParallelSortedInt(mapout->comm, mapout->n, keysout, &sorted));
    PetscCheck(sorted,mapout->comm, PETSC_ERR_PLIB, "samplesort sort failed");
  }
#endif
  PetscCall(PetscFree(keysincopy));
  PetscFunctionReturn(0);
}
