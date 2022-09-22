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
  PetscMPIInt nints, naddrs, ntypes, combiner;

  PetscFunctionBegin;
  PetscCallMPI(MPI_Type_get_envelope(*a, &nints, &naddrs, &ntypes, &combiner));

  if (combiner != MPI_COMBINER_NAMED) PetscCallMPI(MPI_Type_free(a));

  *a = MPI_DATATYPE_NULL;
  PetscFunctionReturn(0);
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
  PetscMPIInt  nints, naddrs, ntypes, combiner, ints[1];
  MPI_Aint     addrs[1];
  MPI_Datatype types[1];

  PetscFunctionBegin;
  *flg   = PETSC_FALSE;
  *atype = a;
  if (a == MPIU_INT || a == MPIU_REAL || a == MPIU_SCALAR) PetscFunctionReturn(0);
  PetscCallMPI(MPI_Type_get_envelope(a, &nints, &naddrs, &ntypes, &combiner));
  if (combiner == MPI_COMBINER_DUP) {
    PetscCheck(nints == 0 && naddrs == 0 && ntypes == 1, PETSC_COMM_SELF, PETSC_ERR_LIB, "Unexpected returns from MPI_Type_get_envelope()");
    PetscCallMPI(MPI_Type_get_contents(a, 0, 0, 1, ints, addrs, types));
    /* Recursively unwrap dupped types. */
    PetscCall(MPIPetsc_Type_unwrap(types[0], atype, flg));
    if (*flg) {
      /* If the recursive call returns a new type, then that means that atype[0] != types[0] and we're on the hook to
       * free types[0].  Note that this case occurs if combiner(types[0]) is MPI_COMBINER_DUP, so we're safe to
       * directly call MPI_Type_free rather than MPIPetsc_Type_free here. */
      PetscCallMPI(MPI_Type_free(&(types[0])));
    }
    /* In any case, it's up to the caller to free the returned type in this case. */
    *flg = PETSC_TRUE;
  } else if (combiner == MPI_COMBINER_CONTIGUOUS) {
    PetscCheck(nints == 1 && naddrs == 0 && ntypes == 1, PETSC_COMM_SELF, PETSC_ERR_LIB, "Unexpected returns from MPI_Type_get_envelope()");
    PetscCallMPI(MPI_Type_get_contents(a, 1, 0, 1, ints, addrs, types));
    if (ints[0] == 1) { /* If a is created by MPI_Type_contiguous(1,..) */
      PetscCall(MPIPetsc_Type_unwrap(types[0], atype, flg));
      if (*flg) PetscCall(MPIPetsc_Type_free(&(types[0])));
      *flg = PETSC_TRUE;
    } else {
      PetscCall(MPIPetsc_Type_free(&(types[0])));
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MPIPetsc_Type_compare(MPI_Datatype a, MPI_Datatype b, PetscBool *match)
{
  MPI_Datatype atype, btype;
  PetscMPIInt  aintcount, aaddrcount, atypecount, acombiner;
  PetscMPIInt  bintcount, baddrcount, btypecount, bcombiner;
  PetscBool    freeatype, freebtype;

  PetscFunctionBegin;
  if (a == b) { /* this is common when using MPI builtin datatypes */
    *match = PETSC_TRUE;
    PetscFunctionReturn(0);
  }
  PetscCall(MPIPetsc_Type_unwrap(a, &atype, &freeatype));
  PetscCall(MPIPetsc_Type_unwrap(b, &btype, &freebtype));
  *match = PETSC_FALSE;
  if (atype == btype) {
    *match = PETSC_TRUE;
    goto free_types;
  }
  PetscCallMPI(MPI_Type_get_envelope(atype, &aintcount, &aaddrcount, &atypecount, &acombiner));
  PetscCallMPI(MPI_Type_get_envelope(btype, &bintcount, &baddrcount, &btypecount, &bcombiner));
  if (acombiner == bcombiner && aintcount == bintcount && aaddrcount == baddrcount && atypecount == btypecount && (aintcount > 0 || aaddrcount > 0 || atypecount > 0)) {
    PetscMPIInt  *aints, *bints;
    MPI_Aint     *aaddrs, *baddrs;
    MPI_Datatype *atypes, *btypes;
    PetscInt      i;
    PetscBool     same;
    PetscCall(PetscMalloc6(aintcount, &aints, bintcount, &bints, aaddrcount, &aaddrs, baddrcount, &baddrs, atypecount, &atypes, btypecount, &btypes));
    PetscCallMPI(MPI_Type_get_contents(atype, aintcount, aaddrcount, atypecount, aints, aaddrs, atypes));
    PetscCallMPI(MPI_Type_get_contents(btype, bintcount, baddrcount, btypecount, bints, baddrs, btypes));
    PetscCall(PetscArraycmp(aints, bints, aintcount, &same));
    if (same) {
      PetscCall(PetscArraycmp(aaddrs, baddrs, aaddrcount, &same));
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
    for (i = 0; i < atypecount; i++) {
      PetscCall(MPIPetsc_Type_free(&(atypes[i])));
      PetscCall(MPIPetsc_Type_free(&(btypes[i])));
    }
    PetscCall(PetscFree6(aints, bints, aaddrs, baddrs, atypes, btypes));
    if (same) *match = PETSC_TRUE;
  }
free_types:
  if (freeatype) PetscCall(MPIPetsc_Type_free(&atype));
  if (freebtype) PetscCall(MPIPetsc_Type_free(&btype));
  PetscFunctionReturn(0);
}

/* Check whether a was created via MPI_Type_contiguous from b
 *
 */
PetscErrorCode MPIPetsc_Type_compare_contig(MPI_Datatype a, MPI_Datatype b, PetscInt *n)
{
  MPI_Datatype atype, btype;
  PetscMPIInt  aintcount, aaddrcount, atypecount, acombiner;
  PetscBool    freeatype, freebtype;

  PetscFunctionBegin;
  if (a == b) {
    *n = 1;
    PetscFunctionReturn(0);
  }
  *n = 0;
  PetscCall(MPIPetsc_Type_unwrap(a, &atype, &freeatype));
  PetscCall(MPIPetsc_Type_unwrap(b, &btype, &freebtype));
  PetscCallMPI(MPI_Type_get_envelope(atype, &aintcount, &aaddrcount, &atypecount, &acombiner));
  if (acombiner == MPI_COMBINER_CONTIGUOUS && aintcount >= 1) {
    PetscMPIInt  *aints;
    MPI_Aint     *aaddrs;
    MPI_Datatype *atypes;
    PetscInt      i;
    PetscBool     same;
    PetscCall(PetscMalloc3(aintcount, &aints, aaddrcount, &aaddrs, atypecount, &atypes));
    PetscCallMPI(MPI_Type_get_contents(atype, aintcount, aaddrcount, atypecount, aints, aaddrs, atypes));
    /* Check for identity first. */
    if (atypes[0] == btype) {
      *n = aints[0];
    } else {
      /* atypes[0] merely has to be equivalent to the type used to create atype. */
      PetscCall(MPIPetsc_Type_compare(atypes[0], btype, &same));
      if (same) *n = aints[0];
    }
    for (i = 0; i < atypecount; i++) PetscCall(MPIPetsc_Type_free(&(atypes[i])));
    PetscCall(PetscFree3(aints, aaddrs, atypes));
  }

  if (freeatype) PetscCall(MPIPetsc_Type_free(&atype));
  if (freebtype) PetscCall(MPIPetsc_Type_free(&btype));
  PetscFunctionReturn(0);
}
