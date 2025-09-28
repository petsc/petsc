#include <petscsf.h>
#include <petscdmplex.h>
#include <petsc/private/dmimpl.h>
#include <petsc/private/dmpleximpl.h>
#include <petsc/private/partitionerimpl.h> /*I "petscpartitioner.h" I*/

PetscBool  MSPartitionerCite       = PETSC_FALSE;
const char MSPartitionerCitation[] = "@article{PETScMSPart2021,\n"
                                     "  author  = {Parsani, Matteo and Boukharfane, Radouan and Nolasco, Irving Reyna and Fern{\'a}ndez, David C Del Rey and Zampini, Stefano and Hadri, Bilel and Dalcin, Lisandro},\n"
                                     "  title   = {High-order accurate entropy-stable discontinuous collocated Galerkin methods with the summation-by-parts property for compressible {CFD} frameworks: Scalable {SSDC} algorithms and flow solver},\n"
                                     "  journal = {Journal of Computational Physics},\n"
                                     "  volume  = {424},\n"
                                     "  pages   = {109844},\n"
                                     "  year    = {2021}\n"
                                     "}\n";

PetscLogEvent PetscPartitioner_MS_SetUp;
PetscLogEvent PetscPartitioner_MS_Stage[PETSCPARTITIONER_MS_NUMSTAGE];

typedef struct {
  PetscInt   levels;
  MPI_Group *lgroup;
  /* Need access to the DM in inner stages */
  PetscInt stage;
  DM       stagedm;
  /* Stage partitioners */
  PetscPartitioner *ppart;
  /* Diagnostic */
  PetscInt *edgeCut;
  /* Target partition weights */
  PetscBool     view_tpwgs;
  PetscSection *tpwgs;
  PetscBool     compute_tpwgs;
} PetscPartitioner_MS;

const char *const PetscPartitionerMultistageStrategyList[] = {"NODE", "MSECTION", "PetscPartitionerMultistageStrategy", "PETSCPARTITIONER_MS_", NULL};

static void PetscLCM_Local(void *in, void *out, PetscMPIInt *cnt, MPI_Datatype *datatype)
{
  PetscInt  count = *cnt;
  PetscInt *xin = (PetscInt *)in, *xout = (PetscInt *)out;

  PetscFunctionBegin;
  if (*datatype != MPIU_INT) {
    (void)(*PetscErrorPrintf)("Can only handle MPIU_INT");
    PETSCABORT(MPI_COMM_SELF, PETSC_ERR_ARG_WRONG);
  }
  for (PetscInt i = 0; i < count; i++) xout[i] = PetscLCM(xin[i], xout[i]);
  PetscFunctionReturnVoid();
}

static PetscErrorCode PetscPartitionerView_Multistage(PetscPartitioner part, PetscViewer viewer)
{
  PetscPartitioner_MS *p = (PetscPartitioner_MS *)part->data;
  PetscViewerFormat    format;
  PetscBool            iascii;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &iascii));
  PetscCall(PetscViewerGetFormat(viewer, &format));
  if (iascii) {
    MPI_Comm  comm;
    MPI_Group group;

    PetscCall(PetscViewerASCIIPushTab(viewer));
    if (!p->stagedm || p->stage == 0) PetscCall(PetscViewerASCIIPrintf(viewer, "Multistage graph partitioner: total stages %" PetscInt_FMT "\n", p->levels));
    comm = PetscObjectComm((PetscObject)part);
    PetscCallMPI(MPI_Comm_group(comm, &group));
    for (PetscInt l = 0; l < p->levels; l++) {
      PetscPartitioner ppart  = p->ppart[l];
      MPI_Comm         pcomm  = PetscObjectComm((PetscObject)ppart);
      MPI_Group        pgroup = MPI_GROUP_EMPTY;

      PetscCall(PetscViewerASCIIPushTab(viewer));
      if (l) {
        if (pcomm != MPI_COMM_NULL) PetscCallMPI(MPI_Comm_group(pcomm, &pgroup));
      } else {
        pgroup = p->lgroup[0];
      }

      if (l) {
        IS          is;
        PetscMPIInt gr, gem = 1;
        PetscInt    uniq;

        PetscCallMPI(MPI_Group_size(group, &gr));
        if (pgroup != MPI_GROUP_EMPTY) {
          gem = 0;
          PetscCallMPI(MPI_Group_translate_ranks(pgroup, 1, &gem, group, &gr));
        }
        PetscCall(ISCreateStride(PetscObjectComm((PetscObject)part), 1, gr, 1, &is));
        PetscCall(ISRenumber(is, NULL, &uniq, NULL));
        PetscCall(ISDestroy(&is));
        PetscCallMPI(MPIU_Allreduce(MPI_IN_PLACE, &gem, 1, MPI_INT, MPI_SUM, PetscObjectComm((PetscObject)part)));
        if (gem) uniq--;
        if (!p->stagedm || l == p->stage) PetscCall(PetscViewerASCIIPrintf(viewer, "Stage %" PetscInt_FMT " partitioners (%" PetscInt_FMT " unique groups, %d empty processes)\n", l, uniq, gem));
      } else {
        PetscMPIInt psize;
        PetscCallMPI(MPI_Group_size(pgroup, &psize));
        if (!p->stagedm || l == p->stage) PetscCall(PetscViewerASCIIPrintf(viewer, "Stage %" PetscInt_FMT " partitioner to %d processes\n", l, psize));
      }
      PetscCall(PetscViewerFlush(viewer));
      if (format == PETSC_VIEWER_ASCII_INFO_DETAIL && (!p->stagedm || l == p->stage)) {
        PetscViewer pviewer;

        if (pcomm == MPI_COMM_NULL) pcomm = PETSC_COMM_SELF;
        PetscCall(PetscViewerGetSubViewer(viewer, pcomm, &pviewer));
        if (ppart) {
          PetscMPIInt size, *ranks, *granks;
          char        tstr[16], strranks[3072]; /* I'm lazy: max 12 chars (including spaces) per rank -> 256 ranks max*/

          PetscCallMPI(MPI_Group_size(pgroup, &size));
          PetscCall(PetscMalloc2(size, &ranks, size, &granks));
          for (PetscMPIInt i = 0; i < size; i++) ranks[i] = i;
          PetscCallMPI(MPI_Group_translate_ranks(pgroup, size, ranks, group, granks));
          if (size <= 256) {
            PetscCall(PetscStrncpy(strranks, "", sizeof(strranks)));
            for (PetscInt i = 0; i < size; i++) {
              PetscCall(PetscSNPrintf(tstr, sizeof(tstr), " %d", granks[i]));
              PetscCall(PetscStrlcat(strranks, tstr, sizeof(strranks)));
            }
          } else PetscCall(PetscStrncpy(strranks, " not showing > 256", sizeof(strranks))); /* LCOV_EXCL_LINE */
          PetscCall(PetscFree2(ranks, granks));
          if (!l) {
            PetscCall(PetscViewerASCIIPrintf(pviewer, "Destination ranks:%s\n", strranks));
            PetscCall(PetscPartitionerView(ppart, pviewer));
          } else {
            PetscCall(PetscPartitionerView(ppart, pviewer));
            PetscCall(PetscViewerASCIIPrintf(pviewer, "  Participating ranks:%s\n", strranks));
          }
          if (p->view_tpwgs) PetscCall(PetscSectionView(p->tpwgs[l], pviewer));
        }
        PetscCall(PetscViewerRestoreSubViewer(viewer, pcomm, &pviewer));
      }
      PetscCall(PetscViewerFlush(viewer));

      if (l && pgroup != MPI_GROUP_EMPTY) PetscCallMPI(MPI_Group_free(&pgroup));
    }
    for (PetscInt l = 0; l < p->levels; l++) PetscCall(PetscViewerASCIIPopTab(viewer));
    PetscCall(PetscViewerASCIIPopTab(viewer));
    PetscCallMPI(MPI_Group_free(&group));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscPartitionerMultistage_CreateStages(MPI_Comm comm, const PetscInt *options, PetscInt *nlevels, MPI_Group *levels[])
{
  MPI_Comm     ncomm;
  MPI_Group   *lgroup;
  MPI_Group    ggroup, ngroup;
  PetscMPIInt *ranks, *granks;
  PetscMPIInt  size, nsize, isize, rank, nrank, i, l, n, m;
  PetscInt     strategy = options ? options[0] : (PetscInt)PETSCPARTITIONER_MS_STRATEGY_NODE;

  PetscFunctionBegin;
  PetscCallMPI(MPI_Comm_size(comm, &size));
  PetscCallMPI(MPI_Comm_rank(comm, &rank));

  if (size == 1) {
    PetscCall(PetscMalloc1(1, &lgroup));
    PetscCallMPI(MPI_Comm_group(comm, &lgroup[0]));
    *nlevels = 1;
    *levels  = lgroup;
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  switch (strategy) {
  case PETSCPARTITIONER_MS_STRATEGY_NODE:
    /* create groups (in global rank ordering of comm) for the 2-level partitioner */
    if (options && options[1] > 0) {
      PetscMPIInt node_size = (PetscMPIInt)options[1];
      if (node_size > 1) {
        PetscMPIInt color;
        if (options[2]) { /* interleaved */
          PetscMPIInt ngroups = size / node_size;

          if (size % node_size) ngroups += 1;
          color = rank % ngroups;
        } else {
          color = rank / node_size;
        }
        PetscCallMPI(MPI_Comm_split(comm, color, rank, &ncomm));
      } else {
        PetscCallMPI(MPI_Comm_dup(comm, &ncomm));
      }
    } else {
#if defined(PETSC_HAVE_MPI_PROCESS_SHARED_MEMORY)
      PetscCallMPI(MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &ncomm));
#else
      /* if users do not specify the node size and MPI_Comm_split_type is not available, defaults to same comm */
      PetscCallMPI(MPI_Comm_dup(comm, &ncomm));
#endif
    }

    PetscCallMPI(MPI_Comm_size(ncomm, &nsize));
    if (nsize == size) { /* one node */
      PetscCall(PetscMalloc1(1, &lgroup));
      PetscCallMPI(MPI_Comm_group(ncomm, &lgroup[0]));
      PetscCallMPI(MPI_Comm_free(&ncomm));

      *nlevels = 1;
      *levels  = lgroup;
      break;
    }

    PetscCall(PetscMalloc1(2, &lgroup));

    /* intranode group (in terms of global rank) */
    PetscCall(PetscMalloc2(size, &ranks, nsize, &granks));
    for (i = 0; i < nsize; i++) ranks[i] = i;
    PetscCallMPI(MPI_Comm_group(comm, &ggroup));
    PetscCallMPI(MPI_Comm_group(ncomm, &ngroup));
    PetscCallMPI(MPI_Group_translate_ranks(ngroup, nsize, ranks, ggroup, granks));
    PetscCallMPI(MPI_Group_incl(ggroup, nsize, granks, &lgroup[1]));

    /* internode group (master processes on the nodes only)
       this group must be specified by all processes in the comm
       we need to gather those master ranks */
    PetscCallMPI(MPI_Group_rank(ngroup, &nrank));
    granks[0] = !nrank ? rank : -1;
    PetscCallMPI(MPI_Allgather(granks, 1, MPI_INT, ranks, 1, MPI_INT, comm));
    for (i = 0, isize = 0; i < size; i++)
      if (ranks[i] >= 0) ranks[isize++] = ranks[i];
    PetscCallMPI(MPI_Group_incl(ggroup, isize, ranks, &lgroup[0]));

    PetscCall(PetscFree2(ranks, granks));
    PetscCallMPI(MPI_Group_free(&ggroup));
    PetscCallMPI(MPI_Group_free(&ngroup));
    PetscCallMPI(MPI_Comm_free(&ncomm));

    *nlevels = 2;
    *levels  = lgroup;
    break;

  case PETSCPARTITIONER_MS_STRATEGY_MSECTION:
    /* recursive m-section (m=2 bisection) */
    m = options ? (PetscMPIInt)options[1] : 2;
    PetscCheck(m >= 2, comm, PETSC_ERR_SUP, "Invalid split %d, must be greater than one", m);
    l = 0;
    n = 1;
    while (n <= size) {
      l++;
      n *= m;
    }
    l -= 1;
    n /= m;

    PetscCheck(l != 0, comm, PETSC_ERR_SUP, "Invalid split %d with communicator size %d", m, size);
    PetscCall(PetscMalloc1(l, &lgroup));
    for (i = 1; i < l; i++) lgroup[i] = MPI_GROUP_EMPTY;

    if (l > 1) {
      PetscMPIInt rem = size - n;
      PetscCall(PetscMalloc1((m + rem), &ranks));
      for (i = 0; i < m; i++) ranks[i] = i * (n / m);
      for (i = 0; i < rem; i++) ranks[i + m] = n + i;
      PetscCallMPI(MPI_Comm_group(comm, &ggroup));
      PetscCallMPI(MPI_Group_incl(ggroup, m + rem, ranks, &lgroup[0]));
      if (rank < n) {
        for (i = 1; i < l; i++) {
          PetscMPIInt inc = (PetscMPIInt)PetscPowInt(m, l - i - 1);
          if (rank % inc == 0) {
            PetscMPIInt anch = (rank - rank % (inc * m)), r;
            for (r = 0; r < m; r++) ranks[r] = anch + r * inc;
            PetscCallMPI(MPI_Group_incl(ggroup, m, ranks, &lgroup[i]));
          }
        }
      }
      PetscCall(PetscFree(ranks));
      PetscCallMPI(MPI_Group_free(&ggroup));
    } else {
      PetscCallMPI(MPI_Comm_group(comm, &lgroup[0]));
    }

    *nlevels = l;
    *levels  = lgroup;
    break;

  default:
    SETERRQ(comm, PETSC_ERR_SUP, "Not implemented"); /* LCOV_EXCL_LINE */
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscPartitionerMultistage_DestroyStages(PetscInt nstages, MPI_Group *groups[])
{
  PetscFunctionBegin;
  for (PetscInt l = 0; l < nstages; l++) {
    MPI_Group group = (*groups)[l];
    if (group != MPI_GROUP_EMPTY) PetscCallMPI(MPI_Group_free(&group));
  }
  PetscCall(PetscFree(*groups));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscPartitionerReset_Multistage(PetscPartitioner part)
{
  PetscPartitioner_MS *p       = (PetscPartitioner_MS *)part->data;
  PetscInt             nstages = p->levels;

  PetscFunctionBegin;
  p->levels = 0;
  PetscCall(PetscPartitionerMultistage_DestroyStages(nstages, &p->lgroup));
  for (PetscInt l = 0; l < nstages; l++) {
    PetscCall(PetscPartitionerDestroy(&p->ppart[l]));
    PetscCall(PetscSectionDestroy(&p->tpwgs[l]));
  }
  PetscCall(PetscFree(p->ppart));
  PetscCall(PetscFree(p->tpwgs));
  PetscCall(PetscFree(p->edgeCut));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscPartitionerDestroy_Multistage(PetscPartitioner part)
{
  PetscFunctionBegin;
  PetscCall(PetscPartitionerReset_Multistage(part));
  PetscCall(PetscFree(part->data));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscPartitionerMultistageSetStages - Sets stages for the partitioning

  Collective

  Input Parameters:
+ part   - the `PetscPartitioner` object.
. levels - the number of stages.
- lgroup - array of `MPI_Group`s for each stage.

  Level: advanced

  Notes:
  `MPI_Comm_create(comm, lgroup[l], &lcomm)` is used to compute the communicator for the stage partitioner at level `l`.
  The groups must be specified in the process numbering of the partitioner communicator.
  `lgroup[0]` must be collectively specified and it must represent a proper subset of the communicator associated with the original partitioner.
  For each level, ranks can be listed in one group only (but they can be listed on different levels)

.seealso: `PetscPartitionerSetType()`, `PetscPartitionerDestroy()`, `PETSCPARTITIONERMULTISTAGE`
@*/
PetscErrorCode PetscPartitionerMultistageSetStages(PetscPartitioner part, PetscInt levels, MPI_Group lgroup[])
{
  PetscPartitioner_MS *p = (PetscPartitioner_MS *)part->data;
  MPI_Comm             comm;
  MPI_Group            wgroup;
  PetscMPIInt        **lparts = NULL;
  PetscMPIInt          rank, size;
  PetscBool            touched, isms;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(part, PETSCPARTITIONER_CLASSID, 1);
  PetscValidLogicalCollectiveInt(part, levels, 2);
  PetscCheck(levels >= 0, PetscObjectComm((PetscObject)part), PETSC_ERR_ARG_OUTOFRANGE, "Number of levels must be non-negative");
  if (levels) PetscAssertPointer(lgroup, 3);
  PetscCall(PetscObjectTypeCompare((PetscObject)part, PETSCPARTITIONERMULTISTAGE, &isms));
  if (!isms) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(PetscLogEventBegin(PetscPartitioner_MS_SetUp, part, 0, 0, 0));

  PetscCall(PetscPartitionerReset_Multistage(part));

  PetscCall(PetscObjectGetComm((PetscObject)part, &comm));
  PetscCallMPI(MPI_Comm_group(comm, &wgroup));
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  PetscCallMPI(MPI_Comm_size(comm, &size));

  p->levels = levels;
  PetscCall(PetscCalloc1(p->levels, &p->ppart));
  PetscCall(PetscCalloc1(p->levels, &p->lgroup));
  PetscCall(PetscCalloc1(p->levels, &p->tpwgs));
  PetscCall(PetscCalloc1(p->levels, &p->edgeCut));

  /* support for target partition weights */
  touched = PETSC_FALSE;
  if (p->compute_tpwgs) PetscCall(PetscMalloc1(p->levels, &lparts));

  for (PetscInt l = 0; l < p->levels; l++) {
    const char *prefix;
    char        aprefix[256];
    MPI_Comm    lcomm;

    if (l) { /* let MPI complain/hang if the user did not specify the groups properly */
      PetscCallMPI(MPI_Comm_create(comm, lgroup[l], &lcomm));
    } else { /* in debug mode, we check that the initial group must be consistently (collectively) specified on comm */
#if defined(PETSC_USE_DEBUG)
      MPI_Group    group, igroup = lgroup[0];
      PetscMPIInt *ranks, *granks;
      PetscMPIInt  b[2], b2[2], csize, gsize;

      PetscCallMPI(MPI_Group_size(igroup, &gsize));
      b[0] = -gsize;
      b[1] = +gsize;
      PetscCallMPI(MPIU_Allreduce(b, b2, 2, MPI_INT, MPI_MAX, comm));
      PetscCheck(-b2[0] == b2[1], comm, PETSC_ERR_ARG_WRONG, "Initial group must be collectively specified");
      PetscCallMPI(MPI_Comm_group(comm, &group));
      PetscCallMPI(MPI_Group_size(group, &csize));
      PetscCall(PetscMalloc2(gsize, &ranks, (csize * gsize), &granks));
      for (PetscMPIInt i = 0; i < gsize; i++) granks[i] = i;
      PetscCallMPI(MPI_Group_translate_ranks(igroup, gsize, granks, group, ranks));
      PetscCallMPI(MPI_Group_free(&group));
      PetscCallMPI(MPI_Allgather(ranks, gsize, MPI_INT, granks, gsize, MPI_INT, comm));
      for (PetscMPIInt i = 0; i < gsize; i++) {
        PetscMPIInt chkr = granks[i];
        for (PetscMPIInt j = 0; j < csize; j++) {
          PetscMPIInt shft = j * gsize + i;
          PetscCheck(chkr == granks[shft], comm, PETSC_ERR_ARG_WRONG, "Initial group must be collectively specified");
        }
      }
      PetscCall(PetscFree2(ranks, granks));
#endif
      lcomm = comm;
    }
    PetscCall(PetscObjectGetOptionsPrefix((PetscObject)part, &prefix));
    if (lcomm != MPI_COMM_NULL) {
      /* MPI_Group_dup */
      PetscCallMPI(MPI_Group_union(lgroup[l], MPI_GROUP_EMPTY, &p->lgroup[l]));
      PetscCall(PetscPartitionerCreate(lcomm, &p->ppart[l]));
      PetscCall(PetscObjectSetOptionsPrefix((PetscObject)p->ppart[l], prefix));
      PetscCall(PetscSNPrintf(aprefix, sizeof(aprefix), "petscpartitioner_multistage_levels_%" PetscInt_FMT "_", l));
      PetscCall(PetscObjectAppendOptionsPrefix((PetscObject)p->ppart[l], aprefix));
    } else {
      PetscCheck(l != 0, PetscObjectComm((PetscObject)part), PETSC_ERR_USER, "Invalid first group");
      p->lgroup[l] = MPI_GROUP_EMPTY;
    }
    if (lcomm != comm && lcomm != MPI_COMM_NULL) PetscCallMPI(MPI_Comm_free(&lcomm));

    /* compute number of partitions per level and detect if a process is part of the process (at any level) */
    if (p->compute_tpwgs) {
      PetscMPIInt gsize;
      PetscCall(PetscMalloc1(size, &lparts[l]));
      PetscCallMPI(MPI_Group_size(p->lgroup[l], &gsize));
      if (!l) {
        PetscMPIInt tr;
        PetscCallMPI(MPI_Group_translate_ranks(wgroup, 1, &rank, p->lgroup[0], &tr));
        if (tr == MPI_UNDEFINED) gsize = 0;
      }
      if (touched && !gsize) gsize = 1;
      PetscCallMPI(MPI_Allgather(&gsize, 1, MPI_INT, lparts[l], 1, MPI_INT, comm));
      if (lparts[l][rank]) touched = PETSC_TRUE;
    }
  }

  /* determine weights (bottom-up) */
  if (p->compute_tpwgs) {
    PetscMPIInt *tranks;
    PetscInt    *lwgts, wgt;
    MPI_Op       MPIU_LCM; /* XXX this is actually recreated at every setup */

    PetscCall(PetscMalloc1((2 * size), &tranks));

    /* we need to compute the least common multiple across processes */
    PetscCallMPI(MPI_Op_create(PetscLCM_Local, 1, &MPIU_LCM));

    /* final target has to have all ones as weights (if the process gets touched) */
    wgt = touched ? 1 : 0;
    PetscCall(PetscMalloc1(size, &lwgts));
    PetscCallMPI(MPI_Allgather(&wgt, 1, MPIU_INT, lwgts, 1, MPIU_INT, comm));

    /* now go up the hierarchy and populate the PetscSection describing the partition weights */
    for (PetscInt l = p->levels - 1; l >= 0; l--) {
      MPI_Comm    pcomm;
      MPI_Group   igroup;
      PetscMPIInt isize, isizer = 0;
      PetscInt    a, b, wgtr;
      PetscMPIInt gsize;
      PetscBool   usep = PETSC_FALSE;

      if (p->ppart[l]) {
        PetscCall(PetscObjectGetComm((PetscObject)p->ppart[l], &pcomm));
      } else {
        pcomm = PETSC_COMM_SELF;
      }

      PetscCallMPI(MPI_Group_size(p->lgroup[l], &gsize));
      if (gsize) {
        usep = PETSC_TRUE;
        for (PetscMPIInt i = 0; i < gsize; i++) tranks[i] = i;
        PetscCallMPI(MPI_Group_translate_ranks(p->lgroup[l], gsize, tranks, wgroup, tranks + size));
      } else gsize = lparts[l][rank];
      PetscCall(PetscFree(lparts[l]));
      PetscCall(PetscSectionCreate(pcomm, &p->tpwgs[l]));
      PetscCall(PetscObjectSetName((PetscObject)p->tpwgs[l], "Target partition weights"));
      PetscCall(PetscSectionSetChart(p->tpwgs[l], 0, gsize));

      if (usep) {
        PetscMPIInt *tr = tranks + size;
        for (PetscMPIInt i = 0; i < gsize; i++) PetscCall(PetscSectionSetDof(p->tpwgs[l], i, lwgts[tr[i]]));
      } else if (gsize) {
        PetscCall(PetscSectionSetDof(p->tpwgs[l], 0, 1));
      }
      PetscCall(PetscSectionSetUp(p->tpwgs[l]));
      if (!l) break;

      /* determine number of processes shared by two consecutive levels */
      PetscCallMPI(MPI_Group_intersection(p->lgroup[l], p->lgroup[l - 1], &igroup));
      PetscCallMPI(MPI_Group_size(igroup, &isize));
      PetscCallMPI(MPI_Group_free(&igroup));
      if (!p->ppart[l] && touched) isize = 1; /* if this process gets touched, needs to propagate its data from one level to the other */

      /* reduce on level partitioner comm the size of the max size of the igroup */
      PetscCallMPI(MPIU_Allreduce(&isize, &isizer, 1, MPI_INT, MPI_MAX, pcomm));

      /* sum previously computed partition weights on the level comm */
      wgt  = lwgts[rank];
      wgtr = wgt;
      PetscCallMPI(MPIU_Allreduce(&wgt, &wgtr, 1, MPIU_INT, MPI_SUM, pcomm));

      /* partition weights are given with integers; to properly compute these and be able to propagate them to the next level,
         we need to compute the least common multiple of isizer across the global comm */
      a = isizer ? isizer : 1;
      b = a;
      PetscCallMPI(MPIU_Allreduce(&a, &b, 1, MPIU_INT, MPIU_LCM, comm));

      /* finally share this process weight with all the other processes */
      wgt = isizer ? (b * wgtr) / isizer : 0;
      PetscCallMPI(MPI_Allgather(&wgt, 1, MPIU_INT, lwgts, 1, MPIU_INT, comm));
    }
    PetscCall(PetscFree(lwgts));
    PetscCall(PetscFree(tranks));
    PetscCall(PetscFree(lparts));
    PetscCallMPI(MPI_Op_free(&MPIU_LCM));
  }
  PetscCallMPI(MPI_Group_free(&wgroup));
  PetscCall(PetscLogEventEnd(PetscPartitioner_MS_SetUp, part, 0, 0, 0));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PetscPartitionerMultistageGetStages_Multistage(PetscPartitioner part, PetscInt *levels, MPI_Group *lgroup[])
{
  PetscPartitioner_MS *p = (PetscPartitioner_MS *)part->data;

  PetscFunctionBegin;
  PetscCheckTypeName(part, PETSCPARTITIONERMULTISTAGE);
  PetscValidHeaderSpecific(part, PETSCPARTITIONER_CLASSID, 1);
  if (levels) *levels = p->levels;
  if (lgroup) *lgroup = p->lgroup;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PetscPartitionerMultistageSetStage_Multistage(PetscPartitioner part, PetscInt stage, PetscObject odm)
{
  DM                   dm = (DM)odm;
  PetscPartitioner_MS *p  = (PetscPartitioner_MS *)part->data;

  PetscFunctionBegin;
  PetscCheckTypeName(part, PETSCPARTITIONERMULTISTAGE);
  PetscCheck(p->levels, PetscObjectComm((PetscObject)part), PETSC_ERR_ORDER, "Number of stages not set yet");
  PetscCheck(stage >= 0 && stage < p->levels, PetscObjectComm((PetscObject)part), PETSC_ERR_ARG_OUTOFRANGE, "Invalid stage index %" PetscInt_FMT ", not in [0,%" PetscInt_FMT ")", stage, p->levels);

  p->stage = stage;
  PetscCall(PetscObjectReference((PetscObject)dm));
  PetscCall(DMDestroy(&p->stagedm));
  p->stagedm = dm;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PetscPartitionerMultistageGetStage_Multistage(PetscPartitioner part, PetscInt *stage, PetscObject *odm)
{
  PetscPartitioner_MS *p = (PetscPartitioner_MS *)part->data;

  PetscFunctionBegin;
  PetscCheckTypeName(part, PETSCPARTITIONERMULTISTAGE);
  if (stage) *stage = p->stage;
  if (odm) *odm = (PetscObject)p->stagedm;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscPartitionerSetUp_Multistage(PetscPartitioner part)
{
  PetscPartitioner_MS *p    = (PetscPartitioner_MS *)part->data;
  MPI_Comm             comm = PetscObjectComm((PetscObject)part);
  PetscInt             nstages;
  MPI_Group           *groups;

  PetscFunctionBegin;
  if (p->levels) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(PetscPartitionerMultistage_CreateStages(comm, NULL, &nstages, &groups));
  PetscCall(PetscPartitionerMultistageSetStages(part, nstages, groups));
  PetscCall(PetscPartitionerMultistage_DestroyStages(nstages, &groups));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* targetSection argument unused, target partition weights are computed internally */
static PetscErrorCode PetscPartitionerPartition_Multistage(PetscPartitioner part, PetscInt nparts, PetscInt numVertices, PetscInt start[], PetscInt adjacency[], PetscSection vertSection, PetscSection edgeSection, PETSC_UNUSED PetscSection targetSection, PetscSection partSection, IS *partition)
{
  PetscPartitioner_MS *p = (PetscPartitioner_MS *)part->data;
  PetscPartitioner     ppart;
  PetscSection         ppartSection = NULL;
  IS                   lpartition;
  const PetscInt      *idxs;
  MPI_Comm             comm, pcomm;
  MPI_Group            group, lgroup, pgroup;
  PetscInt            *pstart, *padjacency;
  PetscInt             nid, i, pedgeCut;
  PetscMPIInt          sameComm, pparts;
  PetscBool            freeadj = PETSC_FALSE;

  PetscFunctionBegin;
  PetscCheck(p->levels, PetscObjectComm((PetscObject)part), PETSC_ERR_ORDER, "Number of stages not set yet");
  PetscCheck(p->stage >= 0 && p->stage < p->levels, PetscObjectComm((PetscObject)part), PETSC_ERR_ARG_OUTOFRANGE, "Invalid stage index %" PetscInt_FMT ", not in [0,%" PetscInt_FMT ")", p->stage, p->levels);
  PetscCall(PetscLogEventBegin(PetscPartitioner_MS_Stage[PetscMin(p->stage, PETSCPARTITIONER_MS_MAXSTAGE)], part, 0, 0, 0));

  /* Group for current stage (size of the group defines number of "local" partitions) */
  lgroup = p->lgroup[p->stage];
  PetscCallMPI(MPI_Group_size(lgroup, &pparts));

  /* Current stage partitioner */
  ppart = p->ppart[p->stage];
  if (ppart) {
    PetscCall(PetscObjectGetComm((PetscObject)ppart, &pcomm));
    PetscCallMPI(MPI_Comm_group(pcomm, &pgroup));
  } else {
    pcomm  = PETSC_COMM_SELF;
    pgroup = MPI_GROUP_EMPTY;
    pparts = -1;
  }

  /* Global comm of partitioner */
  PetscCall(PetscObjectGetComm((PetscObject)part, &comm));
  PetscCallMPI(MPI_Comm_group(comm, &group));

  /* Get adjacency */
  PetscCallMPI(MPI_Group_compare(group, pgroup, &sameComm));
  if (sameComm != MPI_UNEQUAL) {
    pstart     = start;
    padjacency = adjacency;
  } else { /* restrict to partitioner comm */
    ISLocalToGlobalMapping l2g, g2l;
    IS                     gid, rid;
    const PetscInt        *idxs1;
    PetscInt              *idxs2;
    PetscInt               cStart, cEnd, cum;
    DM                     dm = p->stagedm;
    PetscSF                sf;

    PetscCheck(dm, PetscObjectComm((PetscObject)part), PETSC_ERR_PLIB, "Missing stage DM");
    PetscCall(DMGetPointSF(dm, &sf));
    PetscCall(DMPlexGetHeightStratum(dm, part->height, &cStart, &cEnd));
    PetscCall(DMPlexCreateNumbering_Plex(dm, cStart, cEnd, part->height, NULL, sf, &gid));
    /* filter overlapped local cells (if any) */
    PetscCall(ISGetIndices(gid, &idxs1));
    PetscCall(ISGetLocalSize(gid, &cum));
    PetscCall(PetscMalloc1(cum, &idxs2));
    for (i = cStart, cum = 0; i < cEnd; i++) {
      if (idxs1[i - cStart] < 0) continue;
      idxs2[cum++] = idxs1[i - cStart];
    }
    PetscCall(ISRestoreIndices(gid, &idxs1));

    PetscCheck(numVertices == cum, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Unexpected %" PetscInt_FMT " != %" PetscInt_FMT, numVertices, cum);
    PetscCall(ISDestroy(&gid));

    /*
       g2l from full numbering to local numbering
       l2g from local numbering to restricted numbering
    */
    PetscCall(ISCreateGeneral(pcomm, numVertices, idxs2, PETSC_OWN_POINTER, &gid));
    PetscCall(ISRenumber(gid, NULL, NULL, &rid));
    PetscCall(ISLocalToGlobalMappingCreateIS(gid, &g2l));
    PetscCall(ISLocalToGlobalMappingSetType(g2l, ISLOCALTOGLOBALMAPPINGHASH));
    PetscCall(ISLocalToGlobalMappingCreateIS(rid, &l2g));
    PetscCall(ISDestroy(&gid));
    PetscCall(ISDestroy(&rid));

    PetscCall(PetscMalloc2(numVertices + 1, &pstart, start[numVertices], &padjacency));
    pstart[0] = 0;
    for (i = 0; i < numVertices; i++) {
      PetscCall(ISGlobalToLocalMappingApply(g2l, IS_GTOLM_DROP, start[i + 1] - start[i], adjacency + start[i], &pstart[i + 1], padjacency + pstart[i]));
      PetscCall(ISLocalToGlobalMappingApply(l2g, pstart[i + 1], padjacency + pstart[i], padjacency + pstart[i]));
      pstart[i + 1] += pstart[i];
    }
    PetscCall(ISLocalToGlobalMappingDestroy(&l2g));
    PetscCall(ISLocalToGlobalMappingDestroy(&g2l));

    freeadj = PETSC_TRUE;
  }

  /* Compute partitioning */
  pedgeCut = 0;
  PetscCall(PetscSectionCreate(pcomm, &ppartSection));
  if (ppart) {
    PetscMPIInt prank;

    PetscCheck(ppart->ops->partition, PetscObjectComm((PetscObject)ppart), PETSC_ERR_ARG_WRONGSTATE, "PetscPartitioner has no partitioning method on stage %" PetscInt_FMT, p->stage);
    PetscCall(PetscPartitionerPartition(ppart, pparts, numVertices, pstart, padjacency, vertSection, edgeSection, p->tpwgs[p->stage], ppartSection, &lpartition));
    PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)ppart), &prank));
    if (!prank) pedgeCut = ppart->edgeCut; /* only the master rank will reduce */
  } else {                                 /* not participating */
    PetscCall(ISCreateStride(PETSC_COMM_SELF, numVertices, 0, 1, &lpartition));
    pparts = numVertices > 0 ? 1 : 0;
  }
  if (freeadj) PetscCall(PetscFree2(pstart, padjacency));

  /* Init final partition (output) */
  PetscCall(PetscSectionReset(partSection));
  PetscCall(PetscSectionSetChart(partSection, 0, nparts));

  /* We need to map the section back to the global comm numbering */
  for (i = 0; i < pparts; i++) {
    PetscInt    dof;
    PetscMPIInt mp, mpt;

    if (lgroup != MPI_GROUP_EMPTY) {
      PetscCall(PetscMPIIntCast(i, &mp));
      PetscCall(PetscSectionGetDof(ppartSection, i, &dof));
      PetscCallMPI(MPI_Group_translate_ranks(lgroup, 1, &mp, group, &mpt));
    } else {
      PetscCheck(pparts == 1, PETSC_COMM_SELF, PETSC_ERR_LIB, "Unexpected pparts %d", pparts);
      PetscCallMPI(MPI_Comm_rank(comm, &mpt));
      PetscCall(ISGetLocalSize(lpartition, &dof));
    }
    PetscCall(PetscSectionSetDof(partSection, mpt, dof));
  }
  PetscCall(PetscSectionSetUp(partSection));
  PetscCall(PetscSectionDestroy(&ppartSection));

  /* No need to translate the "partition" output, as it is in local cell numbering
     we only change the comm of the index set */
  PetscCall(ISGetIndices(lpartition, &idxs));
  PetscCall(ISGetLocalSize(lpartition, &nid));
  PetscCall(ISCreateGeneral(comm, nid, idxs, PETSC_COPY_VALUES, partition));
  PetscCall(ISRestoreIndices(lpartition, &idxs));
  PetscCall(ISDestroy(&lpartition));

  PetscCall(PetscSectionViewFromOptions(partSection, (PetscObject)part, "-petscpartitioner_multistage_partition_view"));
  PetscCall(ISViewFromOptions(*partition, (PetscObject)part, "-petscpartitioner_multistage_partition_view"));

  /* Update edge-cut */
  p->edgeCut[p->stage] = pedgeCut;
  for (i = p->stage + 1; i < p->levels; i++) p->edgeCut[i] = 0;
  for (i = 0; i < p->stage; i++) pedgeCut += p->edgeCut[i];
  part->edgeCut = -1;
  PetscCallMPI(MPI_Reduce(&pedgeCut, &part->edgeCut, 1, MPIU_INT, MPI_SUM, 0, comm));

  PetscCallMPI(MPI_Group_free(&group));
  if (pgroup != MPI_GROUP_EMPTY) PetscCallMPI(MPI_Group_free(&pgroup));
  PetscCall(PetscLogEventEnd(PetscPartitioner_MS_Stage[PetscMin(p->stage, PETSCPARTITIONER_MS_MAXSTAGE)], part, 0, 0, 0));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscPartitionerSetFromOptions_Multistage(PetscPartitioner part, PetscOptionItems PetscOptionsObject)
{
  PetscPartitioner_MS *p        = (PetscPartitioner_MS *)part->data;
  PetscEnum            strategy = (PetscEnum)PETSCPARTITIONER_MS_STRATEGY_NODE;
  PetscBool            set, roundrobin;
  PetscInt             options[3] = {PETSC_DETERMINE, PETSC_DETERMINE, PETSC_DETERMINE};

  PetscFunctionBegin;
  PetscOptionsHeadBegin(PetscOptionsObject, "PetscPartitioner Multistate Options");
  PetscCall(PetscOptionsEnum("-petscpartitioner_multistage_strategy", "Default partitioning strategy", "", PetscPartitionerMultistageStrategyList, strategy, &strategy, &set));
  if (set || !p->levels) {
    options[0] = (PetscInt)strategy;
    switch (options[0]) {
    case PETSCPARTITIONER_MS_STRATEGY_NODE:
      options[1] = PETSC_DETERMINE;
      roundrobin = PETSC_FALSE;
      PetscCall(PetscOptionsInt("-petscpartitioner_multistage_node_size", "Number of processes per node", "", options[1], &options[1], NULL));
      PetscCall(PetscOptionsBool("-petscpartitioner_multistage_node_interleaved", "Use round robin rank assignments", "", roundrobin, &roundrobin, NULL));
      options[2] = (PetscInt)roundrobin;
      break;
    case PETSCPARTITIONER_MS_STRATEGY_MSECTION:
      options[1] = 2;
      PetscCall(PetscOptionsInt("-petscpartitioner_multistage_msection", "Number of splits per level", "", options[1], &options[1], NULL));
      break;
    default:
      break; /* LCOV_EXCL_LINE */
    }
  }
  PetscCall(PetscOptionsBool("-petscpartitioner_multistage_tpwgts", "Use target partition weights in stage partitioners", "", p->compute_tpwgs, &p->compute_tpwgs, NULL));
  PetscCall(PetscOptionsBool("-petscpartitioner_multistage_viewtpwgts", "View target partition weights", "", p->view_tpwgs, &p->view_tpwgs, NULL));
  PetscOptionsHeadEnd();

  if (options[0] != PETSC_DETERMINE) {
    MPI_Comm   comm = PetscObjectComm((PetscObject)part);
    PetscInt   nstages;
    MPI_Group *groups;

    PetscCall(PetscPartitionerMultistage_CreateStages(comm, options, &nstages, &groups));
    PetscCall(PetscPartitionerMultistageSetStages(part, nstages, groups));
    PetscCall(PetscPartitionerMultistage_DestroyStages(nstages, &groups));
  }

  {
    PetscInt nstages = p->levels, l;
    for (l = 0; l < nstages; l++) {
      if (p->ppart[l]) PetscCall(PetscPartitionerSetFromOptions(p->ppart[l]));
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
  PETSCPARTITIONERMULTISTAGE = "multistage" - A PetscPartitioner object using a multistage distribution strategy

  Level: intermediate

  Options Database Keys:
+  -petscpartitioner_multistage_strategy <strategy> - Either `PETSCPARTITIONER_MS_STRATEGY_NODE`, or `PETSCPARTITIONER_MS_STRATEGY_MSECTION`
.  -petscpartitioner_multistage_node_size <int> - Number of processes per computing node (or `PETSC_DECIDE`)
.  -petscpartitioner_multistage_node_interleaved <bool> - Assign ranks round-robin.
.  -petscpartitioner_multistage_msection <int> - Number of splits per level in recursive m-section splits (use `2` for bisection)
-  -petscpartitioner_multistage_tpwgts <bool> - Use target partition weights in stage partitioner

  Notes:
  The default multistage strategy use `PETSCPARTITIONER_MS_STRATEGY_NODE` and automatically discovers node information using `MPI_Comm_split_type`.
  `PETSCPARTITIONER_MS_STRATEGY_MSECTION` is more for testing purposes.
  Options for single stage partitioners are prefixed by `-petscpartitioner_multistage_levels_`.
  For example, to use parmetis in all stages, `-petscpartitioner_multistage_levels_petscpartitioner_type parmetis`
  Finer grained control is also possible: `-petscpartitioner_multistage_levels_0_petscpartitioner_type parmetis`, `-petscpartitioner_multistage_levels_1_petscpartitioner_type simple`

.seealso: `PetscPartitionerType`, `PetscPartitionerCreate()`, `PetscPartitionerSetType()`
M*/

PETSC_EXTERN PetscErrorCode PetscPartitionerCreate_Multistage(PetscPartitioner part)
{
  PetscPartitioner_MS *p;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(part, PETSCPARTITIONER_CLASSID, 1);
  PetscCall(PetscNew(&p));
  p->compute_tpwgs = PETSC_TRUE;

  part->data                = p;
  part->ops->view           = PetscPartitionerView_Multistage;
  part->ops->destroy        = PetscPartitionerDestroy_Multistage;
  part->ops->partition      = PetscPartitionerPartition_Multistage;
  part->ops->setfromoptions = PetscPartitionerSetFromOptions_Multistage;
  part->ops->setup          = PetscPartitionerSetUp_Multistage;

  PetscCall(PetscCitationsRegister(MSPartitionerCitation, &MSPartitionerCite));
  PetscFunctionReturn(PETSC_SUCCESS);
}
