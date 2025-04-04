#include <petsc/private/sfimpl.h>

#if !defined(PETSC_HAVE_MPI_COMBINER_DUP) && !defined(MPI_COMBINER_DUP) /* We have no way to interpret output of MPI_Type_get_envelope without this. */
  #define MPI_COMBINER_DUP 0
#endif
#if !defined(PETSC_HAVE_MPI_COMBINER_NAMED) && !defined(MPI_COMBINER_NAMED)
  #define MPI_COMBINER_NAMED -2
#endif
#if !defined(PETSC_HAVE_MPI_COMBINER_CONTIGUOUS) && !defined(MPI_COMBINER_CONTIGUOUS) && MPI_VERSION < 2
  #define MPI_COMBINER_CONTIGUOUS -1
#endif

static PetscErrorCode MPIPetsc_Type_free(MPI_Datatype *a)
{
  MPIU_Count  nints, naddrs, ncounts, ntypes;
  PetscMPIInt combiner;

  PetscFunctionBegin;
  PetscCallMPI(MPIPetsc_Type_get_envelope(*a, &nints, &naddrs, &ncounts, &ntypes, &combiner));

  if (combiner != MPI_COMBINER_NAMED) PetscCallMPI(MPI_Type_free(a));

  *a = MPI_DATATYPE_NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}

// PETSc wrapper for MPI_Type_get_envelope_c using MPIU_Count arguments; works even when MPI large count is not available
PetscErrorCode MPIPetsc_Type_get_envelope(MPI_Datatype datatype, MPIU_Count *nints, MPIU_Count *naddrs, MPIU_Count *ncounts, MPIU_Count *ntypes, PetscMPIInt *combiner)
{
  PetscFunctionBegin;
#if defined(PETSC_HAVE_MPI_LARGE_COUNT) && !defined(PETSC_HAVE_MPIUNI) // MPIUNI does not really support large counts in datatype creation
  PetscCallMPI(MPI_Type_get_envelope_c(datatype, nints, naddrs, ncounts, ntypes, combiner));
#else
  PetscMPIInt mints, maddrs, mtypes;
  // As of 2024/09/12, MPI Forum has yet to decide whether it is legal to call MPI_Type_get_envelope() on types created by, e.g.,
  // MPI_Type_contiguous_c(4, MPI_DOUBLE, &newtype). We just let the MPI being used play out (i.e., return error or not)
  PetscCallMPI(MPI_Type_get_envelope(datatype, &mints, &maddrs, &mtypes, combiner));
  *nints   = mints;
  *naddrs  = maddrs;
  *ncounts = 0;
  *ntypes  = mtypes;
#endif
  PetscFunctionReturn(PETSC_SUCCESS);
}

// PETSc wrapper for MPI_Type_get_contents_c using MPIU_Count arguments; works even when MPI large count is not available
PetscErrorCode MPIPetsc_Type_get_contents(MPI_Datatype datatype, MPIU_Count nints, MPIU_Count naddrs, MPIU_Count ncounts, MPIU_Count ntypes, int intarray[], MPI_Aint addrarray[], MPIU_Count countarray[], MPI_Datatype typearray[])
{
  PetscFunctionBegin;
#if defined(PETSC_HAVE_MPI_LARGE_COUNT) && !defined(PETSC_HAVE_MPIUNI) // MPI-4.0, so MPIU_Count is MPI_Count
  PetscCallMPI(MPI_Type_get_contents_c(datatype, nints, naddrs, ncounts, ntypes, intarray, addrarray, countarray, typearray));
#else
  PetscCheck(nints <= PETSC_MPI_INT_MAX && naddrs <= PETSC_MPI_INT_MAX && ntypes <= PETSC_MPI_INT_MAX && ncounts == 0, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "The input derived MPI datatype is created with large counts, but PETSc is configured with an MPI without the large count support");
  PetscCallMPI(MPI_Type_get_contents(datatype, (PetscMPIInt)nints, (PetscMPIInt)naddrs, (PetscMPIInt)ntypes, intarray, addrarray, typearray));
#endif
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  Unwrap an MPI datatype recursively in case it is dupped or MPI_Type_contiguous(1,...)'ed from another type.

   Input Parameter:
.  a  - the datatype to be unwrapped

   Output Parameters:
+ atype - the unwrapped datatype, which is either equal(=) to a or equivalent to a.
- flg   - true if atype != a, which implies caller should MPIPetsc_Type_free(atype) after use. Note atype might be MPI builtin.
*/
PetscErrorCode MPIPetsc_Type_unwrap(MPI_Datatype a, MPI_Datatype *atype, PetscBool *flg)
{
  MPIU_Count   nints = 0, naddrs = 0, ncounts = 0, ntypes = 0, counts[1] = {0};
  PetscMPIInt  combiner, ints[1] = {0};
  MPI_Aint     addrs[1] = {0};
  MPI_Datatype types[1] = {MPI_INT};

  PetscFunctionBegin;
  *flg   = PETSC_FALSE;
  *atype = a;
  if (a == MPIU_INT || a == MPIU_REAL || a == MPIU_SCALAR) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(MPIPetsc_Type_get_envelope(a, &nints, &naddrs, &ncounts, &ntypes, &combiner));
  if (combiner == MPI_COMBINER_DUP) {
    PetscCheck(nints == 0 && naddrs == 0 && ncounts == 0 && ntypes == 1, PETSC_COMM_SELF, PETSC_ERR_LIB, "Unexpected returns from MPI_Type_get_envelope()");
    PetscCallMPI(MPIPetsc_Type_get_contents(a, nints, naddrs, ncounts, ntypes, ints, addrs, counts, types));
    /* Recursively unwrap dupped types. */
    PetscCall(MPIPetsc_Type_unwrap(types[0], atype, flg));
    if (*flg) {
      /* If the recursive call returns a new type, then that means that atype[0] != types[0] and we're on the hook to
       * free types[0].  Note that this case occurs if combiner(types[0]) is MPI_COMBINER_DUP, so we're safe to
       * directly call MPI_Type_free rather than MPIPetsc_Type_free here. */
      PetscCallMPI(MPI_Type_free(&types[0]));
    }
    /* In any case, it's up to the caller to free the returned type in this case. */
    *flg = PETSC_TRUE;
  } else if (combiner == MPI_COMBINER_CONTIGUOUS) {
    PetscCheck((nints + ncounts == 1) && naddrs == 0 && ntypes == 1, PETSC_COMM_SELF, PETSC_ERR_LIB, "Unexpected returns from MPI_Type_get_envelope()");
    PetscCallMPI(MPIPetsc_Type_get_contents(a, nints, naddrs, ncounts, ntypes, ints, addrs, counts, types));
    if ((nints == 1 && ints[0] == 1) || (ncounts == 1 && counts[0] == 1)) { /* If a is created by MPI_Type_contiguous/_c(1,..) */
      PetscCall(MPIPetsc_Type_unwrap(types[0], atype, flg));
      if (*flg) PetscCall(MPIPetsc_Type_free(&types[0]));
      *flg = PETSC_TRUE;
    } else {
      PetscCall(MPIPetsc_Type_free(&types[0]));
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MPIPetsc_Type_compare(MPI_Datatype a, MPI_Datatype b, PetscBool *match)
{
  MPI_Datatype atype, btype;
  MPIU_Count   aintcount, aaddrcount, acountcount, atypecount;
  MPIU_Count   bintcount, baddrcount, bcountcount, btypecount;
  PetscMPIInt  acombiner, bcombiner;
  PetscBool    freeatype, freebtype;

  PetscFunctionBegin;
  if (a == b) { /* this is common when using MPI builtin datatypes */
    *match = PETSC_TRUE;
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  PetscCall(MPIPetsc_Type_unwrap(a, &atype, &freeatype));
  PetscCall(MPIPetsc_Type_unwrap(b, &btype, &freebtype));
  *match = PETSC_FALSE;
  if (atype == btype) {
    *match = PETSC_TRUE;
    goto free_types;
  }
  PetscCall(MPIPetsc_Type_get_envelope(atype, &aintcount, &aaddrcount, &acountcount, &atypecount, &acombiner));
  PetscCall(MPIPetsc_Type_get_envelope(btype, &bintcount, &baddrcount, &bcountcount, &btypecount, &bcombiner));
  if (acombiner == bcombiner && aintcount == bintcount && aaddrcount == baddrcount && acountcount == bcountcount && atypecount == btypecount && (aintcount > 0 || aaddrcount > 0 || acountcount > 0 || atypecount > 0)) {
    PetscMPIInt  *aints, *bints;
    MPI_Aint     *aaddrs, *baddrs;
    MPIU_Count   *acounts, *bcounts;
    MPI_Datatype *atypes, *btypes;
    PetscInt      i;
    PetscBool     same;

    PetscCall(PetscMalloc4(aintcount, &aints, aaddrcount, &aaddrs, acountcount, &acounts, atypecount, &atypes));
    PetscCall(PetscMalloc4(bintcount, &bints, baddrcount, &baddrs, bcountcount, &bcounts, btypecount, &btypes));
    PetscCall(MPIPetsc_Type_get_contents(atype, aintcount, aaddrcount, acountcount, atypecount, aints, aaddrs, acounts, atypes));
    PetscCall(MPIPetsc_Type_get_contents(btype, bintcount, baddrcount, bcountcount, btypecount, bints, baddrs, bcounts, btypes));
    PetscCall(PetscArraycmp(aints, bints, aintcount, &same));
    if (same) {
      PetscCall(PetscArraycmp(aaddrs, baddrs, aaddrcount, &same));
      if (same) {
        PetscCall(PetscArraycmp(acounts, bcounts, acountcount, &same));
        if (same) {
          /* Check for identity first */
          PetscCall(PetscArraycmp(atypes, btypes, atypecount, &same));
          if (!same) {
            /* If the atype or btype were not predefined data types, then the types returned from MPI_Type_get_contents
           * will merely be equivalent to the types used in the construction, so we must recursively compare. */
            for (i = 0; i < atypecount; i++) {
              PetscCall(MPIPetsc_Type_compare(atypes[i], btypes[i], &same));
              if (!same) break;
            }
          }
        }
      }
    }
    for (i = 0; i < atypecount; i++) {
      PetscCall(MPIPetsc_Type_free(&atypes[i]));
      PetscCall(MPIPetsc_Type_free(&btypes[i]));
    }
    PetscCall(PetscFree4(aints, aaddrs, acounts, atypes));
    PetscCall(PetscFree4(bints, baddrs, bcounts, btypes));
    if (same) *match = PETSC_TRUE;
  }
free_types:
  if (freeatype) PetscCall(MPIPetsc_Type_free(&atype));
  if (freebtype) PetscCall(MPIPetsc_Type_free(&btype));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Check whether a was created via MPI_Type_contiguous from b
 *
 */
PetscErrorCode MPIPetsc_Type_compare_contig(MPI_Datatype a, MPI_Datatype b, PetscInt *n)
{
  MPI_Datatype atype, btype;
  MPIU_Count   aintcount, aaddrcount, acountcount, atypecount;
  PetscMPIInt  acombiner;
  PetscBool    freeatype, freebtype;

  PetscFunctionBegin;
  if (a == b) {
    *n = 1;
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  *n = 0;
  PetscCall(MPIPetsc_Type_unwrap(a, &atype, &freeatype));
  PetscCall(MPIPetsc_Type_unwrap(b, &btype, &freebtype));
  PetscCall(MPIPetsc_Type_get_envelope(atype, &aintcount, &aaddrcount, &acountcount, &atypecount, &acombiner));
  if (acombiner == MPI_COMBINER_CONTIGUOUS && (aintcount >= 1 || acountcount >= 1)) {
    PetscMPIInt  *aints;
    MPI_Aint     *aaddrs;
    MPIU_Count   *acounts;
    MPI_Datatype *atypes;
    PetscBool     same;
    PetscCall(PetscMalloc4(aintcount, &aints, aaddrcount, &aaddrs, acountcount, &acounts, atypecount, &atypes));
    PetscCall(MPIPetsc_Type_get_contents(atype, aintcount, aaddrcount, acountcount, atypecount, aints, aaddrs, acounts, atypes));
    /* Check for identity first. */
    if (atypes[0] == btype) {
      if (aintcount) *n = aints[0];
      else PetscCall(PetscIntCast(acounts[0], n)); // Yet to support real big count values
    } else {
      /* atypes[0] merely has to be equivalent to the type used to create atype. */
      PetscCall(MPIPetsc_Type_compare(atypes[0], btype, &same));
      if (same) {
        if (aintcount) *n = aints[0];
        else PetscCall(PetscIntCast(acounts[0], n)); // Yet to support real big count values
      }
    }
    for (MPIU_Count i = 0; i < atypecount; i++) PetscCall(MPIPetsc_Type_free(&atypes[i]));
    PetscCall(PetscFree4(aints, aaddrs, acounts, atypes));
  }

  if (freeatype) PetscCall(MPIPetsc_Type_free(&atype));
  if (freebtype) PetscCall(MPIPetsc_Type_free(&btype));
  PetscFunctionReturn(PETSC_SUCCESS);
}
