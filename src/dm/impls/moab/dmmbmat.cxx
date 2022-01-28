#include <petsc/private/dmmbimpl.h> /*I  "petscdmmoab.h"   I*/
#include <petsc/private/vecimpl.h>

#include <petscdmmoab.h>
#include <MBTagConventions.hpp>
#include <moab/NestedRefine.hpp>

PETSC_EXTERN PetscErrorCode DMMoab_Compute_NNZ_From_Connectivity(DM, PetscInt*, PetscInt*, PetscInt*, PetscInt*, PetscBool);

PETSC_EXTERN PetscErrorCode DMCreateMatrix_Moab(DM dm, Mat *J)
{
  PetscErrorCode  ierr;
  PetscInt        innz = 0, ionz = 0, nlsiz;
  DM_Moab         *dmmoab = (DM_Moab*)dm->data;
  PetscInt        *nnz = 0, *onz = 0;
  char            *tmp = 0;
  Mat             A;
  MatType         mtype;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(J, 3);

  /* next, need to allocate the non-zero arrays to enable pre-allocation */
  mtype = dm->mattype;
  ierr = PetscStrstr(mtype, MATBAIJ, &tmp);CHKERRQ(ierr);
  nlsiz = (tmp ? dmmoab->nloc : dmmoab->nloc * dmmoab->numFields);

  /* allocate the nnz, onz arrays based on block size and local nodes */
  ierr = PetscCalloc2(nlsiz, &nnz, nlsiz, &onz);CHKERRQ(ierr);

  /* compute the nonzero pattern based on MOAB connectivity data for local elements */
  ierr = DMMoab_Compute_NNZ_From_Connectivity(dm, &innz, nnz, &ionz, onz, (tmp ? PETSC_TRUE : PETSC_FALSE));CHKERRQ(ierr);

  /* create the Matrix and set its type as specified by user */
  ierr = MatCreate((((PetscObject)dm)->comm), &A);CHKERRQ(ierr);
  ierr = MatSetSizes(A, dmmoab->nloc * dmmoab->numFields, dmmoab->nloc * dmmoab->numFields, PETSC_DETERMINE, PETSC_DETERMINE);CHKERRQ(ierr);
  ierr = MatSetType(A, mtype);CHKERRQ(ierr);
  ierr = MatSetBlockSize(A, dmmoab->bs);CHKERRQ(ierr);
  ierr = MatSetDM(A, dm);CHKERRQ(ierr); /* set DM reference */
  ierr = MatSetFromOptions(A);CHKERRQ(ierr);

  PetscAssertFalse(!dmmoab->ltog_map,(((PetscObject)dm)->comm), PETSC_ERR_ORDER, "Cannot create a DMMoab Mat without calling DMSetUp first.");
  ierr = MatSetLocalToGlobalMapping(A, dmmoab->ltog_map, dmmoab->ltog_map);CHKERRQ(ierr);

  /* set preallocation based on different supported Mat types */
  ierr = MatSeqAIJSetPreallocation(A, innz, nnz);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(A, innz, nnz, ionz, onz);CHKERRQ(ierr);
  ierr = MatSeqBAIJSetPreallocation(A, dmmoab->bs, innz, nnz);CHKERRQ(ierr);
  ierr = MatMPIBAIJSetPreallocation(A, dmmoab->bs, innz, nnz, ionz, onz);CHKERRQ(ierr);

  /* clean up temporary memory */
  ierr = PetscFree2(nnz, onz);CHKERRQ(ierr);

  /* set up internal matrix data-structures */
  ierr = MatSetUp(A);CHKERRQ(ierr);

  /* MatSetOption(A, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE); */

  *J = A;
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode DMMoab_Compute_NNZ_From_Connectivity(DM dm, PetscInt* innz, PetscInt* nnz, PetscInt* ionz, PetscInt* onz, PetscBool isbaij)
{
  PetscInt        i, f, nloc, vpere, bs, n_nnz, n_onz, ivtx = 0;
  PetscInt        ibs, jbs, inbsize, iobsize, nfields, nlsiz;
  DM_Moab         *dmmoab = (DM_Moab*)dm->data;
  moab::Range     found;
  std::vector<moab::EntityHandle> adjs, storage;
  PetscBool isinterlaced;
  moab::EntityHandle vtx;
  moab::ErrorCode merr;

  PetscFunctionBegin;
  bs = dmmoab->bs;
  nloc = dmmoab->nloc;
  nfields = dmmoab->numFields;
  isinterlaced = (isbaij || bs == nfields ? PETSC_TRUE : PETSC_FALSE);
  nlsiz = (isinterlaced ? nloc : nloc * nfields);

  /* loop over the locally owned vertices and figure out the NNZ pattern using connectivity information */
  for (moab::Range::const_iterator iter = dmmoab->vowned->begin(); iter != dmmoab->vowned->end(); iter++, ivtx++) {

    vtx = *iter;
    /* Get adjacency information for current vertex - i.e., all elements of dimension (dim) that connects
       to the current vertex. We can then decipher if a vertex is ghosted or not and compute the
       non-zero pattern accordingly. */
    adjs.clear();
    if (dmmoab->hlevel && (dmmoab->pcomm->size() == 1)) {
      merr = dmmoab->hierarchy->get_adjacencies(vtx, dmmoab->dim, adjs); MBERRNM(merr);
    }
    else {
      merr = dmmoab->mbiface->get_adjacencies(&vtx, 1, dmmoab->dim, true, adjs, moab::Interface::UNION); MBERRNM(merr);
    }

    /* reset counters */
    n_nnz = n_onz = 0;
    found.clear();

    /* loop over vertices and update the number of connectivity */
    for (unsigned jter = 0; jter < adjs.size(); ++jter) {

      /* Get connectivity information in canonical ordering for the local element */
      const moab::EntityHandle *connect;
      std::vector<moab::EntityHandle> cconnect;
      merr = dmmoab->mbiface->get_connectivity(adjs[jter], connect, vpere, false, &storage); MBERRNM(merr);

      /* loop over each element connected to the adjacent vertex and update as needed */
      for (i = 0; i < vpere; ++i) {
        /* find the truly user-expected layer of ghosted entities to decipher NNZ pattern */
        if (connect[i] == vtx || found.find(connect[i]) != found.end()) continue; /* make sure we don't double count shared vertices */
        if (dmmoab->vghost->find(connect[i]) != dmmoab->vghost->end()) n_onz++; /* update out-of-proc onz */
        else n_nnz++; /* else local vertex */
        found.insert(connect[i]);
      }
    }
    storage.clear();

    if (isbaij) {
      nnz[ivtx] = n_nnz;  /* leave out self to avoid repeats -> node shared by multiple elements */
      if (onz) {
        onz[ivtx] = n_onz; /* add ghost non-owned nodes */
      }
    }
    else { /* AIJ matrices */
      if (!isinterlaced) {
        for (f = 0; f < nfields; f++) {
          nnz[f * nloc + ivtx] = n_nnz; /* leave out self to avoid repeats -> node shared by multiple elements */
          if (onz)
            onz[f * nloc + ivtx] = n_onz; /* add ghost non-owned nodes */
        }
      }
      else {
        for (f = 0; f < nfields; f++) {
          nnz[nfields * ivtx + f] = n_nnz; /* leave out self to avoid repeats -> node shared by multiple elements */
          if (onz)
            onz[nfields * ivtx + f] = n_onz; /* add ghost non-owned nodes */
        }
      }
    }
  }

  for (i = 0; i < nlsiz; i++)
    nnz[i] += 1; /* self count the node */

  for (ivtx = 0; ivtx < nloc; ivtx++) {
    if (!isbaij) {
      for (ibs = 0; ibs < nfields; ibs++) {
        if (dmmoab->dfill) {  /* first address the diagonal block */
          /* just add up the ints -- easier/faster rather than branching based on "1" */
          for (jbs = 0, inbsize = 0; jbs < nfields; jbs++)
            inbsize += dmmoab->dfill[ibs * nfields + jbs];
        }
        else inbsize = nfields; /* dense coupling since user didn't specify the component fill explicitly */
        if (isinterlaced) nnz[ivtx * nfields + ibs] *= inbsize;
        else nnz[ibs * nloc + ivtx] *= inbsize;

        if (onz) {
          if (dmmoab->ofill) {  /* next address the off-diagonal block */
            /* just add up the ints -- easier/faster rather than branching based on "1" */
            for (jbs = 0, iobsize = 0; jbs < nfields; jbs++)
              iobsize += dmmoab->dfill[ibs * nfields + jbs];
          }
          else iobsize = nfields; /* dense coupling since user didn't specify the component fill explicitly */
          if (isinterlaced) onz[ivtx * nfields + ibs] *= iobsize;
          else onz[ibs * nloc + ivtx] *= iobsize;
        }
      }
    }
    else {
      /* check if we got overzealous in our nnz and onz computations */
      nnz[ivtx] = (nnz[ivtx] > dmmoab->nloc ? dmmoab->nloc : nnz[ivtx]);
      if (onz) onz[ivtx] = (onz[ivtx] > dmmoab->nloc ? dmmoab->nloc : onz[ivtx]);
    }
  }
  /* update innz and ionz based on local maxima */
  if (innz || ionz) {
    if (innz) *innz = 0;
    if (ionz) *ionz = 0;
    for (i = 0; i < nlsiz; i++) {
      if (innz && (nnz[i] > *innz)) *innz = nnz[i];
      if ((ionz && onz) && (onz[i] > *ionz)) *ionz = onz[i];
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode DMMoabSetBlockFills_Private(PetscInt w, const PetscInt *fill, PetscInt **rfill)
{
  PetscErrorCode ierr;
  PetscInt       i, j, *ifill;

  PetscFunctionBegin;
  if (!fill) PetscFunctionReturn(0);
  ierr  = PetscMalloc1(w * w, &ifill);CHKERRQ(ierr);
  for (i = 0; i < w; i++) {
    for (j = 0; j < w; j++)
      ifill[i * w + j] = fill[i * w + j];
  }

  *rfill = ifill;
  PetscFunctionReturn(0);
}

/*@C
    DMMoabSetBlockFills - Sets the fill pattern in each block for a multi-component problem
    of the matrix returned by DMCreateMatrix().

    Logically Collective on da

    Input Parameters:
+   dm - the DMMoab object
.   dfill - the fill pattern in the diagonal block (may be NULL, means use dense block)
-   ofill - the fill pattern in the off-diagonal blocks

    Level: developer

    Notes:
    This only makes sense when you are doing multicomponent problems but using the
       MPIAIJ matrix format

           The format for dfill and ofill is a 2 dimensional dof by dof matrix with 1 entries
       representing coupling and 0 entries for missing coupling. For example
$             dfill[9] = {1, 0, 0,
$                         1, 1, 0,
$                         0, 1, 1}
       means that row 0 is coupled with only itself in the diagonal block, row 1 is coupled with
       itself and row 0 (in the diagonal block) and row 2 is coupled with itself and row 1 (in the
       diagonal block).

     DMDASetGetMatrix() allows you to provide general code for those more complicated nonzero patterns then
     can be represented in the dfill, ofill format

   Contributed by Glenn Hammond

.seealso DMCreateMatrix(), DMDASetGetMatrix(), DMSetMatrixPreallocateOnly()

@*/
PetscErrorCode  DMMoabSetBlockFills(DM dm, const PetscInt *dfill, const PetscInt *ofill)
{
  DM_Moab       *dmmoab = (DM_Moab*)dm->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  ierr = DMMoabSetBlockFills_Private(dmmoab->numFields, dfill, &dmmoab->dfill);CHKERRQ(ierr);
  ierr = DMMoabSetBlockFills_Private(dmmoab->numFields, ofill, &dmmoab->ofill);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
