#define PETSCMAT_DLL
#include <petsc-private/matimpl.h>
#include <../src/mat/impls/ij/stashij.h>
#include <../src/sys/utils/hash.h>

/*MC
 MATIJ - MATIJ = "ij".
          A matrix class encoding a PseudoGraph -- a directed graph that admits multiple edges between its vertices.
          The underlying pseudograph, and therefore the matrix, can be interpreted as a multiset-valued or array-valued
          map from vertices to vertices: each vertex v is mapped to the multiset or array of the vertices that terminate
          the edges originating at v.

          Vertices, edges, and local sizes:
          Pseudograph vertices are identified with indices -- nonnegative integers of type PetscInt:
            - domain indices, from which the edges emanate
            - range  or codomain indices, at which the edges terminate
          Each processor owns the domain indices falling within the local ownership range (see MatGetOwnershipRange()).

          Edges emanating from a local domain index correspond to the matrix entries in the corresponding local row.
          Indices terminating the local edges can have any value in [0,N) (where N is Mat's global column size).
          Since any global index can be the target of any local edge, or even of multiple local edges with the same
          source index, the matrix column size does not reflect row sizes.  In particular, the number of edges with the
          same local source can be greater than N (where n is the global column size). As with MatMPIADJ, there is no
          particular distinction attached to the local column size n.


          Map, support, image(s):
          The interpretation as an array-valued map allows MATIJ to define its action on indices or indexed arrays.
          An array of indices with entries within the local ownership range can be mapped to the index array obtained by
          a concatenation of the images of all of the input indices.  Likewise, an indexed array of weights -- scalars,
          integers or integer-scalar pairs -- can be mapped to a similar indexed array with the indices replaced by
          their images, and the weights duplicated, if necessary.

          Using the above map interpretation of MATIJ, the indices within the local ownership range and  nonempty
          images constitute the local support of the Mat -- an array of size m0 <= m.  The indices that belong to any of
          the images of the locally-supported indices constitute the local image of size n0 <= N.

  Level: advanced
M*/

typedef struct {
  PetscBool multivalued;  /* Whether the underlying pseudograph is not a graph. */
  /* The following data structures are using for stashing. */
  MatStashMPIIJ stash;
  /* The following data structures are used for mapping. */
  PetscHashI hsupp; /* local support in a hash table */
  PetscInt m,n;
  PetscInt *ij;      /* concatenated local images of ALL local input elements (i.e., all indices from the local ownership range), sorted within each image */
  PetscInt *ijlen;   /* image segment boundaries for each local input index */
  PetscInt *image;   /* local image */
  PetscInt minijlen, maxijlen; /* min and max image segment size */
  /* The following data structures are used for binning. */
  PetscInt *binoffsets, *binsizes;
} Mat_IJ;


#define MatIJCheckAssembled(mat,needassembled,arg)                                            \
  do {                                                                                            \
    if (!((mat)->assembled) && (needassembled)) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, \
                                  "MatIJ not assembled");                                     \
    if (((mat)->assembled) && !(needassembled)) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, \
                                  "Mat already assembled");                                 \
  } while(0)



#define MatIJGetSuppIndex_Private(A,mode,i,ii)                          \
do                                                                      \
 {                                                                      \
  (ii) = -1;                                                            \
  if ((mode) == MATIJ_LOCAL) {                                           \
    (ii) = i;                                                           \
  }                                                                     \
  else {                                                                \
    Mat_IJ *_13_ij = (Mat_IJ*)((A)->data);                              \
    if (!_13_ij->hsupp) {                                                \
      if ((ii) < (A)->rmap->rend && (ii) >= (i) - A->rmap->rstart)       \
        (ii) = (i) - (A)->rmap->rstart;                                 \
      else                                                              \
        (ii) = -1;                                                      \
    }                                                                   \
    else  {                                                             \
      PetscHashIMap(_13_ij->hsupp,(i),(ii));                            \
    }                                                                   \
  }                                                                     \
 }                                                                      \
while(0)

#define MatIJGetIndexImage_Private(A,mode,i,ii)                         \
{                                                                       \
    if ((mode) == MATIJ_LOCAL) {                                         \
      /* Assume image has been "localized". */                          \
      ii = i;                                                           \
    }                                                                   \
    else  {                                                             \
      ii = !((Mat_IJ*)((A)->data))->image?i:(((Mat_IJ*)((A)->data))->image)[i]; \
    }                                                                   \
}                                                                       \




static PetscErrorCode MatIJLocalizeImage_Private(Mat);

/*@C
   MatIJMap      - map an array of global indices (inidxi) with index (inidxj) and scalar (inval) weights, by pushing
                   the indices along the edges of the underlying pseudograph (see MATIJ).
                     Each locally-owned global index i from inidxi is replaced by the array of global indices terminating
                   the Mat's pseudograph edges that emanate from i, in the order the edges were provided to
                   MatIJSetEdges() or MatIJSetEdgesIS(); the individual image arrays are concatenated. inidxi ndices
                   outside the local ownership range or the local support are silently ignored -- replaced by
                   empty arrays. The weights from the domain indices are attached to the corresponding images, with
                   duplication, if necessary.

   Not collective.

   Input Parameters:
+  A        - pseudograph
.  intype   - (MATIJ_LOCAL | MATIJ_GLOBAL) meaning of inidxi: local support numbers or global indices
.  insize   - size of the input index and weight arrays; PETSC_NULL indicates _all_ support indices
.  inidxi   - array (of size insize) of global indices
.  inidxj   - array (of size insize) of index weights
.  inval    - array (of size insize) of scalar weights
-  outtype  - (MATIJ_LOCAL | MATIJ_GLOBAL) desired meaning of outdxi: local support numbers or global indices

   Output Parameters:
+  outsize  - size of the output index and weight arrays
.  outidxi  - array (of size outsize) of the global indices adjacent to the indices in inidxi
.  outidxj  - array (of size outsize) of the index weights inherited by outidxi from inidxi
.  outval   - array (of size outsize) of the scalar weights inherited by outidxi from inidxi
-  imgsizes - array (of size insize) of the sizes of image segments within outidxi for each i from inidxi

   Level: advanced
.seealso: MatIJBin(), MatIJBinMap(), MatIJGetSupport()
@*/
#undef __FUNCT__
#define __FUNCT__ "MatIJMap"
PetscErrorCode MatIJMap(Mat A, MatIJIndexType intype, PetscInt insize, const PetscInt *inidxi, const PetscInt *inidxj, const PetscScalar *inval, MatIJIndexType outtype, PetscInt *outsize, PetscInt **outidxi, PetscInt **outidxj, PetscScalar **outval, PetscInt **outsizes)
{
  PetscErrorCode ierr;
  Mat_IJ *pg = (Mat_IJ*)A->data;
  PetscInt i,j,k,indi=0,indj,outsize_ = -1,*outidxi_ = PETSC_NULL, *outidxj_ = PETSC_NULL, *outsizes_ = PETSC_NULL;
  PetscScalar *outval_ = PETSC_NULL;
  PetscFunctionBegin;

  if ((outidxi && !*outidxi) || (outidxj && !*outidxj) || (outval && !*outval)) {
    ierr = MatIJMap(A,intype,insize,inidxi,inidxj,inval,outtype,&outsize_,PETSC_NULL,PETSC_NULL,PETSC_NULL,PETSC_NULL); CHKERRQ(ierr);
  }
  if (insize == PETSC_DETERMINE)
    inidxi = PETSC_NULL;
  else if (insize < 0)
    SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Invalid input array size: %D", insize);
  if (outidxi) {
    if (!*outidxi) {
      ierr = PetscMalloc(sizeof(PetscInt)*outsize_, outidxi);    CHKERRQ(ierr);
    }
    outidxi_ = *outidxi;
  }
  if (outidxj) {
    if (!*outidxj) {
      ierr = PetscMalloc(sizeof(PetscInt)*outsize_, outidxj);    CHKERRQ(ierr);
    }
    outidxj_= *outidxj;
  }
  if (outval) {
    if (!*outval) {
      ierr = PetscMalloc(sizeof(PetscScalar)*outsize_, outval);  CHKERRQ(ierr);
    }
    outval_ = *outval;
  }
  if (outsizes_) {
    if (!*outsizes) {
      ierr = PetscMalloc(sizeof(PetscInt)*insize, outsizes);    CHKERRQ(ierr);
    }
    outsizes_ = *outsizes;
  }

  if (intype == MATIJ_LOCAL && !pg->image) {
    ierr = MatIJLocalizeImage_Private(A); CHKERRQ(ierr);
  }
  j = 0;
  for (i = 0; i < insize; ++i) {
    if (!inidxi) {
      indi = i;
    }
    else {
      /* Convert to local. */
      MatIJGetSuppIndex_Private(A,intype,inidxi[i],indi);
      if ((indi < 0 || indi >= pg->m)){
        /* drop */
        if (outsizes_) outsizes_[i] = 0;
        continue;
      }
    }
    if (outidxi_ || (inval && outval_) || (inidxj && outidxj_) ) {
      for (k = pg->ijlen[indi]; k < pg->ijlen[indi+1]; ++k) {
        MatIJGetIndexImage_Private(A,outtype,pg->ij[k],indj);
        if (outidxi_)         outidxi_[j] = indj;
        if (inidxj&&outidxj_) outidxj_[j] = inidxj[i];
        if (inval&&outval_)   outval_[j]  = inval[i];
        ++j;
      }
    }
    else {
      j += pg->ijlen[indi+1]-pg->ijlen[indi];
    }
    if (outsizes_) outsizes_[i] = (pg->ijlen[indi+1]-pg->ijlen[indi]);
  }/* for (i = 0; i < len; ++i) */
  if (outsize) *outsize = j;
  PetscFunctionReturn(0);
}



/*@C
   MatIJBin     - bin an array of global indices (inidxi) along with index (inidxj) and scalar (inval) weights by pushing the indices
                   along the edges of the underlying pseudograph (see MATIJ).
                     Each locally-owned global index i from inidxi is put in the arrays corresponding to the global indices
                   terminating the Mat's pseudograph edges that emanate from i. The bin arrays are ordered by the terminating
                   index. inidxi ndices outside the local ownership range or the local support are silently ignored --
                   contribute to no bins. The index weights in inidxj and inval are arranged into bins of their own, exactly mirroring
                   the binning of inidxi.


   Not collective.

   Input Parameters:
+  A        - pseudograph
.  intype   - (MATIJ_LOCAL | MATIJ_GLOBAL) meaning of inidxi: local support numbers or global indices
.  insize   - size of the input index and weight arrays; PETSC_NULL indicates _all_ support indices
.  inidxi   - array (of size insize) of global indices
.  inidxj   - array (of size insize) of index weights
-  inval    - array (of size insize) of scalar weights


   Output Parameters:
+  outsize  - size of the array of concatenated bins
.  outidxi  - array (of size outsize) containing the binned indices from inidxi
.  outidxj  - array (of size outsize) containing the binned index weights from inidxj
.  outval   - array (of size outsize) containing the binned scalar weights from inval
-  binsizes - array (of size n) of bin sizes

   Note: n0 is the local image size -- the number of indices terminating the locally-supported indices
         (see MATIJ) -- and can be obtained with MatIJGetImageSize().

   Level: advanced
.seealso: MatIJMap(), MatIJBinMap(), MatIJGetSupport(), MatIJGetImageSize()
@*/
#undef __FUNCT__
#define __FUNCT__ "MatIJBin"
PetscErrorCode MatIJBin(Mat A, MatIJIndexType intype, PetscInt insize, const PetscInt *inidxi, const PetscInt *inidxj, const PetscScalar *inval, PetscInt *outsize, PetscInt **outidxi, PetscInt **outidxj, PetscScalar **outval, PetscInt **binsizes)
{
  PetscErrorCode ierr;
  Mat_IJ *pg = (Mat_IJ*)A->data;
  PetscInt i,j,k,indi=0,outsize_ = -1,*outidxi_ = PETSC_NULL, *outidxj_ = PETSC_NULL, *binsizes_ = PETSC_NULL;
  PetscScalar *outval_ = PETSC_NULL;
  PetscFunctionBegin;

  /* Binning requires a localized image. */
  ierr = MatIJLocalizeImage_Private(A); CHKERRQ(ierr);
  if ((outidxi && !*outidxi) || (outidxj && !*outidxj) || (outval && !*outval)) {
    ierr = MatIJBin(A,intype,insize,inidxi,PETSC_NULL,PETSC_NULL,&outsize_,PETSC_NULL,PETSC_NULL,PETSC_NULL,PETSC_NULL); CHKERRQ(ierr);
  }
  if (insize == PETSC_DETERMINE){
    insize = pg->m;
    inidxi = PETSC_NULL;
  }
  else if (insize < 0)
    SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Invalid input array size: %D", insize);
  if (outidxi) {
    if (!*outidxi) {
      ierr = PetscMalloc(sizeof(PetscInt)*outsize_,outidxi);    CHKERRQ(ierr);
    }
    outidxi_ = *outidxi;
  }
  if (outidxj) {
    if (!*outidxj) {
      ierr = PetscMalloc(sizeof(PetscInt)*outsize_,outidxj);    CHKERRQ(ierr);
    }
    outidxj_ = *outidxj;
  }
  if (outval) {
    if (!*outval) {
      ierr = PetscMalloc(sizeof(PetscScalar)*outsize_,outval);  CHKERRQ(ierr);
    }
    outval_ = *outval;
  }
  if (binsizes) {
    if (!*binsizes) {
      ierr = PetscMalloc(sizeof(PetscInt)*(pg->n), binsizes);    CHKERRQ(ierr);
    }
    binsizes_ = *binsizes;
  }

  /* We'll need to count the contributions to each "bin" and the offset of each bin in outidx. */
  /* Allocate the bin offset array, if necessary. */
  if (!pg->binoffsets) {
    ierr = PetscMalloc((pg->n+1)*sizeof(PetscInt), &(pg->binoffsets)); CHKERRQ(ierr);
  }
  /* Initialize bin offsets */
  for (j = 0; j <= pg->n; ++j) {
    pg->binoffsets[j] = 0;
  }
  for (i = 0; i < insize; ++i) {
    if (!inidxi) {
      indi = i;
    }
    else {
      MatIJGetSuppIndex_Private(A,intype,inidxi[i],indi);
      if ((indi < 0 || indi >=pg->m)){
        /* drop */
        continue;
      }
    }
    for (k = pg->ijlen[indi]; k < pg->ijlen[indi+1]; ++k) {
      ++(pg->binoffsets[pg->ij[k]+1]);
    }
  }/* for (i = 0; i < insize; ++i) */
  /* Convert bin sizes into bin offsets. */
  for (j = 0; j < pg->n; ++j) {
    pg->binoffsets[j+1] += pg->binoffsets[j];
  }
  /* Now bin the input indices and values. */
  if (outidxi_ || (inval && outval) || (inidxj && outidxj_) ) {
    /* Allocate the bin size array, if necessary. */
    if (!binsizes_) {
      if (!pg->binsizes) {
        ierr = PetscMalloc((pg->n)*sizeof(PetscInt), &(pg->binsizes)); CHKERRQ(ierr);
      }
      binsizes_ = pg->binsizes;
    }
    /* Initialize bin sizes to zero. */
    for (j = 0; j < pg->n; ++j) {
      binsizes_[j] = 0;
    }
    for (i = 0; i < insize; ++i) {
      if (!inidxi) {
        indi = i;
      }
      else {
        /* Convert to local. */
        MatIJGetSuppIndex_Private(A,intype,inidxi[i],indi);
      }
      if ((indi < 0 || indi >= pg->m)){
        /* drop */
        continue;
      }
      for (k = pg->ijlen[indi]; k < pg->ijlen[indi+1]; ++k) {
        j = pg->ij[k];
        if (outidxi_)            outidxi_[pg->binoffsets[j]+binsizes_[j]] = inidxi[i];
        if (outval_ && inval)    outval_ [pg->binoffsets[j]+binsizes_[j]] = inval[i];
        if (outidxj_ && inidxj)  outidxj_[pg->binoffsets[j]+binsizes_[j]] = inidxj[i];
        ++binsizes[j];
      }
    }/* for (i = 0; i < insize; ++i) */
  }/* if (outidxi_ || (inval && outval_) || (inidxj && outidxj_) ) */
  if (outsize) *outsize = pg->binoffsets[pg->n];
  PetscFunctionReturn(0);
}


/*@C
   MatIJBinMap     - simultaneously bin and map an  array of indices (inidxi) along with index (inidxj) and scalar (inval) weights
                      by pushing the indices along the edges of two pseudographs (see MATIJ, MatIJMap(), MatIJBin()).
                        Each locally-supported index i from inidxi is assigned to the arrays (bins) corresponding to the global
                      indices terminating the A's pseudograph edges that emanate from i. i's location in each bin is occupied
                      by the index terminating the corresponding pseudograph edge that emanate from i in B. Thus, A and B must be
                      compatible in the following sense: they must  have the same local suppors and local image sizes.
                         inidxi indices outside the local support are silently ignored -- contribute to no bins. The index (inidxj)
                      and scalar (inval) weights are arranged in bins of their own, exactly mirroring the binning of inidxi.

   Not collective.

   Input Parameters:
+  A        - binning pseudograph
.  B        - mapping pseudograph
.  intype   - (MATIJ_LOCAL | MATIJ_GLOBAL) meaning of inidxi: local support numbers or global indices
.  insize   - size of the input index and weight arrays
.  inidxi   - array (of size insize) of global indices
.  inidxj   - array (of size insize) of index  weights
.  inval    - array (of size insize) of scalar weights
-  outtype  - (MATIJ_LOCAL | MATIJ_GLOBAL) desired meaning of inidxi: local image numbers or global indices


   Output Parameters:
+  outsize  - size of the array of concatenated bins
.  outidxi  - array (of size outsize) containing the binned images of the indices from inidxi
.  outidxj  - array (of size outsize) containing the binned index weights from inidxj
.  outval   - array (of size outsize) containing the binned scalar weights from inval
-  binsizes - array (of size n0) of bin sizes

   Note:
+  The idea behind MatIJBinMap is that the binning is done by A, while what is actually binned comes from B.
   Pseudographs A and B are structurally isomorphic. Moreover, they only differ in the terminating indices:
   edges i-e->jA and i-eB->jB are in a one-to-one correspondence, if their positions in the ordering of i's
   images are the same. Each source index i is assigned to every bin jA labeled by each of the indices
   attached to i in A by some eA, but the binned value is jB --  the index attached to i in B by eB, which
   corresponds to eA.
-  Another way of viewing the pseudograph pair A and B is as a single pseudograph (B), whose edges are
   colored (by A's terminating indices), and ordered on the color within each originating index's image.


   Level: advanced
.seealso: MATIJ, MatIJBin(), MatIJMap(), MatIJGetSupport(), MatIJGetImage(), MatIJGetRowSizes()
@*/
#undef __FUNCT__
#define __FUNCT__ "MatIJBinMap"
PetscErrorCode MatIJBinMap(Mat A, Mat B, MatIJIndexType intype, PetscInt insize, const PetscInt *inidxi, const PetscInt *inidxj, const PetscScalar *inval, MatIJIndexType outtype, PetscInt *outsize, PetscInt **outidxi, PetscInt **outidxj, PetscScalar **outval, PetscInt **binsizes)
{
  PetscErrorCode ierr;
  Mat_IJ *pga = (Mat_IJ*)A->data;
  Mat_IJ *pgb = (Mat_IJ*)B->data;
  PetscBool isij;
  PetscInt indi = -1, indj, i,j,k,outsize_ = -1,*outidxi_ = PETSC_NULL, *outidxj_ = PETSC_NULL, *binsizes_ = PETSC_NULL;

  PetscScalar *outval_ = PETSC_NULL;
  PetscFunctionBegin;

  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  ierr = PetscObjectTypeCompare((PetscObject)A,MATIJ,&isij); CHKERRQ(ierr);
  if (!isij) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Matrix 1 not of type MATIJ: %s", ((PetscObject)A)->type);
  PetscValidHeaderSpecific(B,MAT_CLASSID,2);
  ierr = PetscObjectTypeCompare((PetscObject)B,MATIJ,&isij); CHKERRQ(ierr);
  if (!isij) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Matrix 2 not of type MATIJ: %s", ((PetscObject)B)->type);
  PetscCheckSameComm((PetscObject)A,1,(PetscObject)B,2);

  if (A->rmap->n != B->rmap->n) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Incompatible local row sizes: %D and %D", A->rmap->n, B->rmap->n);
  if (A->rmap->N != B->rmap->N) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Incompatible global row sizes: %D and %D", A->rmap->N, B->rmap->N);
  if (A->cmap->n != B->cmap->n) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Incompatible local column sizes: %D and %D", A->cmap->n, B->cmap->n);
  if (A->cmap->N != B->cmap->N) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Incompatible global column sizes: %D and %D", A->cmap->N, B->cmap->N);

  if (pga->m != pgb->m) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Incompatible local support sizes: %D and %D", pga->m, pgb->m);
  if (pga->n != pgb->n) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Incompatible local image sizes: %D and %D", pga->n, pgb->n);

  /* Binning requires a localized image. */
  ierr = MatIJLocalizeImage_Private(A); CHKERRQ(ierr);
  if ((outidxi && !*outidxi) || (outidxj && !*outidxj) || (outval && !*outval)) {
    ierr = MatIJBinMap(A,B,intype,insize,inidxi,inidxj,inval,outtype,&outsize_,PETSC_NULL,PETSC_NULL,PETSC_NULL,PETSC_NULL); CHKERRQ(ierr);
  }
  if (insize == PETSC_DETERMINE){
    insize = pga->m;
    inidxi = PETSC_NULL;
  }
  else if (insize < 0)
    SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Invalid input array size: %D", insize);
  if (outidxi) {
    if (!*outidxi) {
      ierr = PetscMalloc(sizeof(PetscInt)*outsize_, outidxi);    CHKERRQ(ierr);
    }
    outidxi_ = *outidxi;
  }
  if (outidxj) {
    if (!*outidxj) {
      ierr = PetscMalloc(sizeof(PetscInt)*outsize_, outidxj);    CHKERRQ(ierr);
    }
    outidxj_ = *outidxj;
  }
  if (outval) {
    if (!*outval) {
      ierr = PetscMalloc(sizeof(PetscScalar)*outsize_, outval);  CHKERRQ(ierr);
    }
    outval_ = *outval;
  }
  if (binsizes) {
    if (!*binsizes) {
      ierr = PetscMalloc(sizeof(PetscInt)*(pga->n), binsizes);    CHKERRQ(ierr);
    }
    binsizes_ = *binsizes;
  }

  /* We'll need to count the contributions to each "bin" and the offset of each bin in outidx_. */
  /* Allocate the bin offset array, if necessary. */
  if (!pga->binoffsets) {
    ierr = PetscMalloc((pga->n+1)*sizeof(PetscInt), &(pga->binoffsets)); CHKERRQ(ierr);
  }
  /* Allocate the bin size array, if necessary. */
  if (!binsizes_) {
    if (!pga->binsizes) {
      ierr = PetscMalloc((pga->n)*sizeof(PetscInt), &(pga->binsizes)); CHKERRQ(ierr);
    }
    binsizes_ = pga->binsizes;
  }
  /* Initialize bin offsets */
  for (j = 0; j <= pga->n; ++j) {
    pga->binoffsets[j] = 0;
  }
  for (i = 0; i < insize; ++i) {
    if (!inidxi) {
      indi = i;
    }
    else {
      /* Convert to local. */
      MatIJGetSuppIndex_Private(A,intype,inidxi[i],indi);
      if ((indi < 0 || indi >= pga->m)){
        /* drop */
        continue;
      }
    }
    if (pga->ijlen[indi] != pgb->ijlen[indi])
      SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Image sizes different for local index %D = indi: %D and %D", indi, pga->ijlen[indi], pgb->ijlen[indi]);
    for (k = pga->ijlen[indi]; k < pga->ijlen[indi+1]; ++k) {
      ++(pga->binoffsets[pga->ij[k]+1]);
    }
  }/* for (i = 0; i < insize; ++i) */
  /* Convert bin sizes into bin offsets. */
  for (j = 0; j < pga->n; ++j) {
    pga->binoffsets[j+1] += pga->binoffsets[j];
  }
  /* Now bin the input indices and values. */
  if (outidxi_ || (inval && outval_) || (inidxj && outidxj_) ) {
    /* Initialize bin sizes to zero. */
    for (j = 0; j < pga->n; ++j) {
      binsizes_[j] = 0;
    }
    for (i = 0; i < insize; ++i) {
      if (!inidxi) {
        indi = inidxi[i];
      }
      else {
        /* Convert to local. */
        MatIJGetSuppIndex_Private(A,intype,inidxi[i],indi);
        if ((indi < 0 || indi >= pga->m)){
          /* drop */
          continue;
        }
      }
      for (k = pga->ijlen[indi]; k < pga->ijlen[indi+1]; ++k) {
        j = pga->ij[k];
        MatIJGetIndexImage_Private(A,outtype,pgb->ij[k],indj);
        if (outidxi_)            outidxi_[pga->binoffsets[j]+binsizes_[j]] = indj;
        if (outval_ && inval)    outval_[pga->binoffsets[j] +binsizes_[j]] = inval[i];
        if (outidxj_ && inidxj)  outidxj_[pga->binoffsets[j]+binsizes_[j]] = inidxj[i];
        ++binsizes_[j];
      }
    }/* for (i = 0; i < insize; ++i) */
  }/* if (outidxi_ || (inval && outval_) || (inidxj && outidxj_) ) */
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatGetRow_IJ"
PetscErrorCode MatGetRow_IJ(Mat A, PetscInt row, PetscInt *rowsize, PetscInt *cols[], PetscScalar *vals[]) {
  PetscInt off,len,i,r;
  PetscErrorCode ierr;
  Mat_IJ *pg = (Mat_IJ*)A->data;

  PetscFunctionBegin;
  /* It is easy to implement this, but will only be done, if there is demand. */
  if (rowsize) *rowsize = 0;
  if (cols)    *cols    = PETSC_NULL;
  if (vals)    *vals    = PETSC_NULL;

  /* Convert to local. */
  MatIJGetSuppIndex_Private(A,MATIJ_GLOBAL,row,r);
  if ((r >= 0 && r < pg->m)){
    off = pg->ijlen[r];
    len = pg->ijlen[r+1]-pg->ijlen[r];
    if (cols) *cols = pg->ij+off;
    if (rowsize) *rowsize = len;
    if (vals) {
      ierr = PetscMalloc(sizeof(PetscScalar)*len, vals); CHKERRQ(ierr);
      for (i = 0; i < len; ++i) {
        (*vals)[i] = (PetscScalar)1.0;
      }
    }
  }
  PetscFunctionReturn(0);
 }

#undef __FUNCT__
#define __FUNCT__ "MatRestoreRow_IJ"
PetscErrorCode MatRestoreRow_IJ(Mat A, PetscInt row, PetscInt *rowsize, PetscInt *cols[], PetscScalar *vals[]) {

  PetscInt r;
  PetscErrorCode ierr;
  Mat_IJ *pg = (Mat_IJ*)A->data;

  PetscFunctionBegin;

  /* Convert to local. */
  MatIJGetSuppIndex_Private(A,MATIJ_GLOBAL,row,r);
  if ((r >= 0 && r < pg->m)){
    if (vals) {
      ierr = PetscFree(*vals); CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}


/*@C
   MatIJSetMultivalued - indicates whether the underlying pseudograph is a multivalued or not (a graph).

   Not collective.

   Input arguments:
+  A           -  pseudograph
-  multivalued -  whether the matrix encodes a multivalued (pseudo)graph.

   Level: advanced

.seealso: MatIJGetMultivalued(), MatIJSetEdges(), MatIJGetEdges(), MatIJGetSupport(), MatIJGetImage()
 @*/
#undef  __FUNCT__
#define __FUNCT__ "MatIJSetMultivalued"
PetscErrorCode MatIJSetMultivalued(Mat A, PetscBool multivalued)
{
  PetscErrorCode ierr;
  Mat_IJ *pg = (Mat_IJ *)(A->data);
  PetscBool isij;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  ierr = PetscObjectTypeCompare((PetscObject)A,MATIJ,&isij); CHKERRQ(ierr);
  if (!isij) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Matrix not of type MATIJ: %s", ((PetscObject)A)->type);
  MatIJCheckAssembled(A,PETSC_FALSE,1);
  ierr = MatStashMPIIJSetMultivalued_Private(pg->stash,multivalued); CHKERRQ(ierr);
  pg->multivalued = multivalued;
  PetscFunctionReturn(0);
}

/*@C
   MatIJGetMultivalued - return a flag indicating whether the underlying pseudograph is a multivalued or not (a graph).

   Not collective.

   Input arguments:
.  A           -    pseudograph

   Output arguments:
.  multivalued -  whether the matrix encodes a multivalued (pseudo)graph.

   Level: advanced

.seealso: MatIJSetMultivalued(), MatIJSetEdges(), MatIJGetEdges(), MatIJGetSupport(), MatIJGetImage()
 @*/
#undef  __FUNCT__
#define __FUNCT__ "MatIJGetMultivalued"
PetscErrorCode MatIJGetMultivalued(Mat A, PetscBool *multivalued)
{
  PetscErrorCode ierr;
  Mat_IJ *pg = (Mat_IJ *)(A->data);
  PetscBool isij;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  ierr = PetscObjectTypeCompare((PetscObject)A,MATIJ,&isij); CHKERRQ(ierr);
  if (!isij) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Matrix not of type MATIJ: %s", ((PetscObject)A)->type);
  PetscValidPointer(multivalued,2);
  *multivalued = pg->multivalued;
  PetscFunctionReturn(0);
}


/* Clears everything, but the stash and multivalued. */
#undef __FUNCT__
#define __FUNCT__ "MatIJClear_Private"
static PetscErrorCode MatIJClear_Private(Mat mat)
{
  Mat_IJ   *pg  = (Mat_IJ*)(mat->data);
  PetscErrorCode ierr;
  PetscFunctionBegin;
  if (pg->hsupp) {
    PetscHashIDestroy((pg->hsupp));
  }
  if (pg->image) {
    ierr = PetscFree(pg->image);  CHKERRQ(ierr);
  }
  if (pg->ij) {
    ierr = PetscFree(pg->ij);     CHKERRQ(ierr);
  }
  if (pg->ijlen) {
    ierr = PetscFree(pg->ijlen);  CHKERRQ(ierr);
  }
  if (pg->binoffsets) {
    ierr = PetscFree(pg->binoffsets); CHKERRQ(ierr);
  }
  pg->m = pg->minijlen = pg->maxijlen = 0;
  pg->n = PETSC_DETERMINE;
  PetscFunctionReturn(0);
}

/*@C
   MatIJSetEdgesIS - sets the edges in the pseudograph matrix.
                     The edges are specified as two index sets of vertices of equal length:
                     outgoing and incoming vertices (ix -> iy).

   Not collective

   Input parameters:
+  A  -    pseudograph
.  ix -    list of outgoing vertices
-  iy -    list of incoming vertices

   Note:
+  This will cause the matrix to be be put in an unassembled state.
.  Edges are assembled during MatAssembly -- moved to the processor owning the outgoing vertex.
-  Communicators of the IS objects must match that of MatIJ.

   Level: intermediate

.seealso: MATIJ, MatIJSetEdges(), MatIJGetEdgesIS(), MatIJGetSupportIS(), MatIJGetImageIS()
@*/
#undef  __FUNCT__
#define __FUNCT__ "MatIJSetEdgesIS"
PetscErrorCode MatIJSetEdgesIS(Mat A, IS ix, IS iy)
{
  IS iix, iiy;
  PetscInt nix, niy;
  const PetscInt *ixidx, *iyidx;
  PetscErrorCode ierr;
  PetscBool      isij;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  ierr = PetscObjectTypeCompare((PetscObject)A,MATIJ,&isij); CHKERRQ(ierr);
  if (!isij) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Matrix not of type MATIJ: %s", ((PetscObject)A)->type);

  PetscValidHeaderSpecific(ix,IS_CLASSID,2);
  PetscValidHeaderSpecific(iy,IS_CLASSID,3);

  PetscCheckSameComm(A,1,ix,2);
  PetscCheckSameComm(ix,2,iy,3);

  if (!ix) {
    ierr = ISCreateStride(((PetscObject)A)->comm, A->rmap->n, A->rmap->rstart, 1, &(iix)); CHKERRQ(ierr);
    nix = A->rmap->n;
  }
  else
    iix = ix;
  if (!iy) {
    ierr = ISCreateStride(((PetscObject)A)->comm, A->cmap->n, A->cmap->rstart, 1, &(iiy)); CHKERRQ(ierr);
    niy = A->cmap->n;
  }
  else
    iiy = iy;
  ierr = ISGetLocalSize(iix,&nix); CHKERRQ(ierr);
  ierr = ISGetLocalSize(iiy,&niy); CHKERRQ(ierr);
  if (nix != niy) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Incompatible IS sizes: %D and %D", nix, niy);

  ierr = ISGetIndices(iix, &ixidx);  CHKERRQ(ierr);
  ierr = ISGetIndices(iiy, &iyidx);  CHKERRQ(ierr);
  MatIJSetEdges(A,nix,ixidx, iyidx); CHKERRQ(ierr);
  ierr = ISRestoreIndices(iix, &ixidx);  CHKERRQ(ierr);
  ierr = ISRestoreIndices(iiy, &iyidx);  CHKERRQ(ierr);
  if (!ix) {
    ierr = ISDestroy(&iix); CHKERRQ(ierr);
  }
  if (!iy) {
    ierr = ISDestroy(&iiy); CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*@C
   MatIJSetEdges - sets the edges in the pseudograph matrix.
                     The edges are specified as two integer arrays of vertices of equal length:
                     outgoing and incoming vertices (ix -> iy).

   Not collective

   Input parameters:
+  A     -    pseudograph
.  len   -    length of vertex arrays
.  ixidx -    list of outgoing vertices
-  iyidx -    list of incoming vertices

   Note:
+  This will cause the matrix to be be put in an unassembled state.
-  Edges are assembled during MatAssembly -- moved to the processor owning the outgoing vertex.

   Level: intermediate

.seealso: MATIJ, MatIJSetEdgesIS(), MatIJGetEdges(), MatIJGetSupport(), MatIJGetImage()
@*/
#undef  __FUNCT__
#define __FUNCT__ "MatIJSetEdges"
PetscErrorCode MatIJSetEdges(Mat A, PetscInt len, const PetscInt *ixidx, const PetscInt *iyidx)
{
  Mat_IJ *pg = (Mat_IJ*)(A->data);
  PetscInt *iixidx = PETSC_NULL, *iiyidx = PETSC_NULL, k;
  PetscErrorCode ierr;
  PetscBool      isij;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  ierr = PetscObjectTypeCompare((PetscObject)A,MATIJ,&isij); CHKERRQ(ierr);
  if (!isij) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Matrix not of type MATIJ: %s", ((PetscObject)A)->type);

  if (len < 0) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Negative edge array length: %D", len);

  if (!ixidx){
    if (len != A->rmap->n)
      SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "The length of an empty source array %D must equal the local row size %D", len, A->rmap->n);
    ierr = PetscMalloc(len*sizeof(PetscInt), &iixidx); CHKERRQ(ierr);
    for (k = 0; k < len; ++k) {
      iixidx[k] = A->rmap->rstart + k;
    }
    ixidx = iixidx;
  }
  if (!iyidx) {
    if (len != A->cmap->n)
      SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "The length of an empty target array %D must equal the local column size %D", len, A->cmap->n);
    for (k = 0; k < len; ++k) {
      iiyidx[k] = A->cmap->rstart + k;
    }
    iyidx = iiyidx;
  }
  ierr = MatStashMPIIJExtend_Private(pg->stash, len, ixidx, iyidx); CHKERRQ(ierr);
  if (!iixidx) {
    ierr = PetscFree(iixidx); CHKERRQ(ierr);
  }
  if (!iiyidx) {
    ierr = PetscFree(iiyidx); CHKERRQ(ierr);
  }
  A->was_assembled = A->assembled;
  A->assembled = PETSC_FALSE;
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "MatIJGetAssembledEdges_Private"
static PetscErrorCode MatIJGetAssembledEdges_Private(Mat A, PetscInt *len, PetscInt **ixidx, PetscInt **iyidx)
{
  PetscErrorCode ierr;
  Mat_IJ   *pg = (Mat_IJ *)(A->data);
  PetscInt len_, *ixidx_ = PETSC_NULL,*iyidx_ = PETSC_NULL, k,i, ii;
  PetscHashIIter hi;
  PetscBool isij;
  PetscFunctionBegin;

  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  ierr = PetscObjectTypeCompare((PetscObject)A,MATIJ,&isij); CHKERRQ(ierr);
  if (!isij) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Matrix not of type MATIJ: %s", ((PetscObject)A)->type);

  if (!len && !ixidx && !iyidx) PetscFunctionReturn(0);

  len_ = pg->ijlen[pg->m];
  if (len) *len = len_;
  if (!ixidx && !iyidx) PetscFunctionReturn(0);
  if (ixidx) {
    if (!*ixidx) {
      ierr = PetscMalloc(sizeof(PetscInt)*len_, ixidx); CHKERRQ(ierr);
    }
    ixidx_ = *ixidx;
  }
  if (iyidx) {
    if (!*iyidx) {
      ierr = PetscMalloc(sizeof(PetscInt)*len_, iyidx); CHKERRQ(ierr);
    }
    iyidx_ = *iyidx;
  }
  if (pg->hsupp) {
    PetscHashIIterBegin(pg->hsupp,hi);
    while(!PetscHashIIterAtEnd(pg->hsupp,hi)){
      PetscHashIIterGetKeyVal(pg->hsupp,hi,ii,i);
      for (k = pg->ijlen[i]; k < pg->ijlen[i+1]; ++k) {
        if (ixidx_) ixidx_[k] = ii;
        if (iyidx_) iyidx_[k] = pg->image[pg->ij[k]];
      }
      PetscHashIIterNext(pg->hsupp,hi);
    }
  }
  else {
    for (i = 0; i < pg->m; ++i) {
      for (k = pg->ijlen[i]; k < pg->ijlen[i+1]; ++k) {
        if (ixidx_) ixidx_[k] = i + A->rmap->rstart;
        if (iyidx_) iyidx_[k] = pg->image[pg->ij[k]];
      }
    }
  }
  PetscFunctionReturn(0);
}


/*@C
   MatIJGetEdges -   retrieves the edges from a pseudograph matrix.
                     The edges are specified as two integer arrays of vertices of equal length:
                     outgoing and incoming vertices (ix -> iy).

   Not collective

   Input parameters:
.  A     -    pseudograph

   Output parameters:
+  len   -    length of vertex arrays
.  ixidx -    list of outgoing vertices
-  iyidx -    list of incoming vertices


   Notes:
+  Both assembled and unassembled edges are returned.
-  For an assembled matrix the retrieved outgoing vertices are guaranteed to be locally-owned.

   Level: advanced

.seealso: MATIJ, MatIJSetEdges(), MatIJGetEdgesIS(), MatIJGetSupport(), MatIJGetImage()
@*/
#undef  __FUNCT__
#define __FUNCT__ "MatIJGetEdges"
PetscErrorCode MatIJGetEdges(Mat A, PetscInt *len, PetscInt **ixidx, PetscInt **iyidx)
{
  PetscErrorCode ierr;
  Mat_IJ   *pg = (Mat_IJ *)(A->data);
  PetscInt len_, lenI, lenII;
  PetscInt *ixidxI = PETSC_NULL, *iyidxI = PETSC_NULL, *ixidxII = PETSC_NULL, *iyidxII = PETSC_NULL;
  PetscBool isij;
  PetscFunctionBegin;

  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  ierr = PetscObjectTypeCompare((PetscObject)A,MATIJ,&isij); CHKERRQ(ierr);
  if (!isij) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Matrix not of type MATIJ: %s", ((PetscObject)A)->type);

  if (!len && !ixidx && !iyidx) PetscFunctionReturn(0);

  ierr = MatIJGetAssembledEdges_Private(A, &lenI, PETSC_NULL, PETSC_NULL);              CHKERRQ(ierr);
  ierr = MatStashMPIIJGetIndicesMerged_Private(pg->stash, &lenII, PETSC_NULL, PETSC_NULL); CHKERRQ(ierr);

  len_ = lenI + lenII;
  if (len) *len = len_;

  if (!len_ || (!ixidx && !iyidx)) PetscFunctionReturn(0);

  if (!ixidx || !iyidx) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Vertex array pointers must be null or non-null together");

  if (!lenI) {
    /* Only stash indices need to be returned. */
    ierr = MatStashMPIIJGetIndicesMerged_Private(pg->stash, len, ixidx, iyidx); CHKERRQ(ierr);
  }
  else if (!lenII) {
    /* Only assembled edges must be returned. */
    ierr = MatIJGetAssembledEdges_Private(A, len, ixidx, iyidx); CHKERRQ(ierr);
  }
  else {
    /* Retrieve the two sets of indices. */
    ierr = MatIJGetAssembledEdges_Private(A, &lenI, &ixidxI, &iyidxI);                CHKERRQ(ierr);
    ierr = MatStashMPIIJGetIndicesMerged_Private(pg->stash, &lenII, &ixidxII, &iyidxII); CHKERRQ(ierr);
    /* Merge. */
    ierr = PetscMergeIntArrayPair(lenI,ixidxI,iyidxI,lenII,ixidxII,iyidxII,len,ixidx,iyidx); CHKERRQ(ierr);
    /* Clean up. */
    ierr = PetscFree(ixidxI);  CHKERRQ(ierr);
    ierr = PetscFree(iyidxI);  CHKERRQ(ierr);
    ierr = PetscFree(ixidxII); CHKERRQ(ierr);
    ierr = PetscFree(iyidxII); CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*@C
   MatIJGetEdgesIS - retrieves the edges in the boolean matrix graph.
                     The edges are specified as two index sets of vertices of equal length:
                     outgoing and incoming vertices (ix -> iy).

   Not collective

   Input parameters:
.  A     -    pseudograph

   Output parameters:
+  ix    -    IS of outgoing vertices
-  iy    -    IS of incoming vertices

   Note:
+  Both assembled and unassembled edges are returned.
.  For an assembled matrix the retrieved outgoing vertices are guaranteed to be locally-owned.
-  ix and iy will have the same communicator as MatIJ and will have the same length.


   Level: intermediate

.seealso: MATIJ, MatIJSetEdgesIS(), MatIJGetEdges(), MatIJGetSupportIS(), MatIJGetImageIS()
@*/
#undef  __FUNCT__
#define __FUNCT__ "MatIJGetEdgesIS"
PetscErrorCode MatIJGetEdgesIS(Mat A, IS *ix, IS *iy)
{
  PetscErrorCode ierr;
  PetscInt   len, *ixidx = PETSC_NULL, *iyidx = PETSC_NULL, **_ixidx = PETSC_NULL, **_iyidx = PETSC_NULL;
  PetscBool  isij;
  PetscFunctionBegin;

  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  ierr = PetscObjectTypeCompare((PetscObject)A,MATIJ,&isij); CHKERRQ(ierr);
  if (!isij) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Matrix not of type MATIJ: %s", ((PetscObject)A)->type);
  if (ix){
    _ixidx = &ixidx;
  }
  if (iy) {
    _iyidx = &iyidx;
  }
  ierr = MatIJGetEdges(A, &len, _ixidx, _iyidx); CHKERRQ(ierr);
  if (ix) {
    ierr = ISCreateGeneral(A->rmap->comm, len, ixidx, PETSC_OWN_POINTER, ix); CHKERRQ(ierr);
  }
  if (iy) {
    ierr = ISCreateGeneral(A->cmap->comm, len, iyidx, PETSC_OWN_POINTER, iy); CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*
 Sort iy and store the unique in a (temporary) hash table to determine the local image.
 Endow the global image indices with a local number, then replace global indices in ij
 with the local numbers.  Store the global image in an array: each local number will
 naturally serve as the index into the array for the corresponding global index.
*/
#undef __FUNCT__
#define __FUNCT__ "MatIJLocalizeImage_Private"
static PetscErrorCode MatIJLocalizeImage_Private(Mat A)
{
  PetscErrorCode ierr;
  Mat_IJ *pg = (Mat_IJ *) A->data;
  PetscInt i,j,k,n,totalnij,*image;
  PetscHashI himage;
  PetscFunctionBegin;
  if (pg->image) PetscFunctionReturn(0);

  /* (A) Construct image -- the set of unique ij indices. */
  /* (B) Endow the image with a local numbering.  */
  /* (C) Then convert ij to this local numbering. */

  /* (A) */
  /* Insert all of the ij into himage. */
  PetscHashICreate(himage);
  for (i = 0; i < pg->m; ++i) {
    for (k = pg->ijlen[i]; pg->ijlen[i+1]; ++i) {
      PetscHashIAdd(himage,pg->ij[k],0);
    }
  }
  /* (B) */
  /* Endow the image with a local numbering: retrieve and sort its elements. */
  PetscHashISize(himage,n);
  ierr = PetscMalloc(n*sizeof(PetscInt), &image); CHKERRQ(ierr);
  PetscHashIGetKeys(himage,n,image);
  ierr = PetscSortInt(n,image); CHKERRQ(ierr);
  /* (C) */
  /*
   Convert ij to local numbering: insert image elements into an emptied and resized himage, mapping them to their local numbers.
   Then remap all of ij using himage.
   */
  PetscHashIClear(himage);
  PetscHashIResize(himage,n);
  for (j = 0; j < n; ++j) {
    PetscHashIAdd(himage,image[j],j);
  }
  totalnij = 0;
  PetscHashIMapArray(himage,pg->ijlen[pg->m],pg->ij,totalnij,pg->ij); CHKERRQ(ierr);
  if (totalnij!=pg->ijlen[pg->m]) {
    SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Number of image indices before %D and after %D localization do not match", pg->ijlen[pg->m],totalnij);
  }
  /* Store the newly computed image array. */
  pg->image = image;
  /* Clean up. */
  PetscHashIDestroy(himage);
  PetscFunctionReturn(0);
}

/*
 Indices are assumed sorted on ix.
 If !multivalued, remove iy duplicates from each ix's image segment.
 Record the number of images in ijlen.
 Store unique ix in a hash table along with their local numbers.
 Sort iy and store the unique in a (temporary) hash table to determine the local image.
 Note:  this routine takes ownership of ("steals the reference to") ixidx and iyidx.
*/
#undef __FUNCT__
#define __FUNCT__ "MatIJSetEdgesLocal_Private"
static PetscErrorCode MatIJSetEdgesLocal_Private(Mat A, const PetscInt len, PetscInt *ixidx, PetscInt *iyidx)
{
  PetscErrorCode ierr;
  Mat_IJ *pg = (Mat_IJ *) A->data;
  PetscInt start, end, totalnij;
  PetscInt i,j,minnij, maxnij, nij,m,*ij, *ijlen, *supp;
  PetscHashI hsupp = PETSC_NULL;
  PetscFunctionBegin;

  if (!len) {
    PetscFunctionReturn(0);
  }

  m        = 0;
  totalnij = 0;
  maxnij   = 0;
  minnij   = len;
  start = 0;
  while (start < len) {
    end = start+1;
    while (end < len && ixidx[end] == ixidx[start]) ++end;
    ++(m); /* count all of ixidx[start:end-1] as a single occurence of an idx index */
    nij = 1; /* At least one element in the image. */
    if (end - 1 > start) { /* found 2 or more of ixidx[start] in a row */
      /* sort the relevant portion of iy */
      ierr = PetscSortInt(end-start,iyidx+start);CHKERRQ(ierr);
      if (pg->multivalued) {
        nij = end-start;
      }
      else {
        /* count unique elements in iyidx[start,end-1] */
        for (j=start+1; j < end; ++j){
          if (iyidx[j] > iyidx[j-1]) ++nij;
        }
      }
    }
    totalnij += nij;
    minnij = PetscMin(minnij, nij);
    maxnij = PetscMax(maxnij, nij);
    start = end;
  }
  /*
   Now we know the size of the support -- m, and the total size of concatenated image segments -- totalnij.
   Allocate an array for recording the support indices -- supp.
   Allocate an array for recording the images of each support index -- ij.
   Allocate an array for counting the number of images for each support index -- ijlen.
   */
  ierr = PetscMalloc(sizeof(PetscInt)*(m+1), &ijlen);  CHKERRQ(ierr);
  ierr = PetscMalloc(sizeof(PetscInt)*(totalnij),&ij); CHKERRQ(ierr);
  ierr = PetscMalloc(sizeof(PetscInt)*m, &supp); CHKERRQ(ierr);

  /*
     We now record in supp only the unique ixidx indices, and in ij the iyidx indices in each of the image segments.
   */
  if (m < A->rmap->n)
    PetscHashICreate(hsupp);
  i = 0;
  j = 0;
  start = 0;
  ijlen[0] = 0;
  while (start < len) {
    end = start+1;
    while (end < len && ixidx[end] == ixidx[start]) ++end;
    if (hsupp) {
      PetscHashIAdd(hsupp,ixidx[start],i); CHKERRQ(ierr);
    }
    ++i;
    /* the relevant portion of iy is already sorted. */
    ij[j++] = iyidx[start++];
    while(start < end) {
      if (pg->multivalued || iyidx[start] > iyidx[start-1])
        ij[j++] = iyidx[start];
      ++start;
    }
    ijlen[i] = j;
  }

  ierr = PetscFree(ixidx); CHKERRQ(ierr);
  ierr = PetscFree(iyidx); CHKERRQ(ierr);
  /* Record the changes. */
  pg->hsupp    = hsupp;
  pg->m        = m;
  pg->ij       = ij;
  pg->ijlen    = ijlen;
  pg->minijlen = minnij;
  pg->maxijlen = maxnij;

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatAssemblyBegin_IJ"
PetscErrorCode MatAssemblyBegin_IJ(Mat A, MatAssemblyType type)
{
  Mat_IJ   *ij  = (Mat_IJ*)(A->data);
  PetscInt len, *ixidx, *iyidx;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatStashMPIIJAssemble_Private(ij->stash);              CHKERRQ(ierr);
  if (type == MAT_FINAL_ASSEMBLY) {
    ierr = MatIJGetEdges(A, &len, &ixidx, &iyidx);           CHKERRQ(ierr);
    ierr = MatStashMPIIJClear_Private(ij->stash);               CHKERRQ(ierr);

    ierr = MatStashMPIIJSetPreallocation_Private(ij->stash, 0,0); CHKERRQ(ierr);
    ierr = MatIJClear_Private(A);                            CHKERRQ(ierr);
    ierr = MatIJSetEdgesLocal_Private(A, len, ixidx, iyidx); CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "MatAssemblyEnd_IJ"
PetscErrorCode MatAssemblyEnd_IJ(Mat A, MatAssemblyType type)
{
  PetscFunctionBegin;
  /* Currently a noop */
  PetscFunctionReturn(0);
}


/*@C
   MatIJGetSupport - retrieves the global indices of the graph's vertices of nonzero outdegree
                     (i.e., the global indices of this processor's nonzero rows).
                     If the graph is regarded as a multivalued map on integers, this is
                     the support of the map (i.e., the set of indices with nonempty images).


   Not collective

   Input parameters:
.  A     -    pseudograph

   Output parameters:
+  len   -    the length of the support array
-  supp  -    the support array

   Note:
+  This operation fails for a nonassembled matrix.
.  In general, the returned indices are unsorted; use PetscSortInt, if necessary.
-  The caller is responsible for freeing the support array.


   Level: intermediate

.seealso: MATIJ, MatIJGetSupportIS(), MatIJGetImage()
@*/
#undef  __FUNCT__
#define __FUNCT__ "MatIJGetSupport"
PetscErrorCode MatIJGetSupport(Mat A, PetscInt *len, PetscInt **supp)
{
  PetscErrorCode ierr;
  Mat_IJ *pg = (Mat_IJ *)(A->data);
  PetscBool isij;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  ierr = PetscObjectTypeCompare((PetscObject)A,MATIJ,&isij); CHKERRQ(ierr);
  if (!isij) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Matrix not of type MATIJ: %s", ((PetscObject)A)->type);
  MatIJCheckAssembled(A,PETSC_TRUE,1);
  if (!len && !supp) PetscFunctionReturn(0);
  if (len) *len = pg->m;
  if (!supp) PetscFunctionReturn(0);
  if (!*supp) {
    ierr = PetscMalloc(sizeof(PetscInt)*pg->m, supp);          CHKERRQ(ierr);
  }
  if (!pg->hsupp) {
    PetscInt i;
    for (i = 0; i < pg->m; ++i) {
      (*supp)[i] = i + A->rmap->rstart;
    }
  }
  *len = 0;
  PetscHashIGetKeys(pg->hsupp, *len, *supp);                CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   MatIJGetSupportIS - retrieves the global indices of the graph's vertices of nonzero outdegree
                     (i.e., the global indices of this processor's nonzero rows).
                     If the graph is regarded as a multivalued map on integers, this is
                     the support of the map (i.e., the set of indices with nonempty images).

   Not collective

   Input parameters:
.  A     -    pseudograph

   Output parameters:
.  supp  -    the support IS

   Note:
+  This operation fails for a nonassembled matrix.
-  The caller is responsible for destroying the support IS.


   Level: intermediate

.seealso: MATIJ, MatIJGetSupport(), MatIJGetSupportISSupported(), MatIJGetImage()
@*/
#undef  __FUNCT__
#define __FUNCT__ "MatIJGetSupportIS"
PetscErrorCode MatIJGetSupportIS(Mat A, IS *supp)
{
  PetscErrorCode ierr;
  Mat_IJ         *pg = (Mat_IJ *)(A->data);
  PetscBool      isij;
  PetscInt       ilen, *isupp;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  ierr = PetscObjectTypeCompare((PetscObject)A,MATIJ,&isij); CHKERRQ(ierr);
  if (!isij) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Matrix not of type MATIJ: %s", ((PetscObject)A)->type);
  MatIJCheckAssembled(A,PETSC_TRUE,1);
  if (!supp) PetscFunctionReturn(0);
  if (pg->hsupp) {
    ierr = MatIJGetSupport(A, &ilen, &isupp); CHKERRQ(ierr);
    ierr = ISCreateGeneral(A->rmap->comm, ilen, isupp, PETSC_OWN_POINTER, supp); CHKERRQ(ierr);
  }
  else {
    ierr = ISCreateStride(A->rmap->comm, A->rmap->n, A->rmap->rstart,1,supp);          CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}




/*@C
   MatIJGetImage - retrieves the global indices of the graph's vertices of nonzero indegree
                     on this processor (i.e., the global indices of this processor's nonzero columns).
                     If the graph is regarded as a multivalued map on integers, this is
                     the image of the map (the union of the images of this processor's support indices).

   Not collective

   Input parameters:
.  A     -    pseudograph

   Output parameters:
+  len   -    the length of the image array
-  image -    the image array

   Note:
+  This operation fails for a nonassembled matrix.
-  The caller is responsible for freeing the image array.


   Level: intermediate

.seealso: MATIJ, MatIJGetImageIS(), MatIJGetSupport(), MatIJGetMaxRowSize()
@*/
#undef  __FUNCT__
#define __FUNCT__ "MatIJGetImage"
PetscErrorCode MatIJGetImage(Mat A, PetscInt *len, PetscInt **image)
{
  PetscErrorCode ierr;
  Mat_IJ *pg = (Mat_IJ *)(A->data);
  PetscBool isij;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  ierr = PetscObjectTypeCompare((PetscObject)A,MATIJ,&isij); CHKERRQ(ierr);
  if (!isij) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Matrix not of type MATIJ: %s", ((PetscObject)A)->type);
  MatIJCheckAssembled(A,PETSC_TRUE,1);
  ierr = MatIJLocalizeImage_Private(A); CHKERRQ(ierr);
  if (len) *len = pg->n;
  if (image) {
    ierr = PetscMalloc(sizeof(PetscInt)*pg->n, image);             CHKERRQ(ierr);
    ierr = PetscMemcpy(*image, pg->image, sizeof(PetscInt)*pg->n); CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}



/*@C
  MatIJGetMaxRowSize - returns the largest number of nonzero columns in all of this processor's rows.
                             If MatIJ (equivalently, the underlying graph) is regarded as a multivalued
                           mapping on integers, then the result is the size of the largest set among
                           the images of this processor's indices.
  Not collective.

  Input parameters:
. A        -    pseudograph

  Output parameters:
. maxsize  - the size of the largest image set

  Level: advanced

  Notes:
+ This routine is useful for preallocating arrays to hold the images of the local indices:
  if an array of the largest image size is allocated, it can be used for repeatedly computing
  the images of the local indices.
- This routine will fail for an unassembled matrix.

.seealso: MATIJ, MatIJGetImage(), MatIJGetSupport(), MatIJMapI(), MatIJMapIJ(), MatIJMapIW(), MatIJMapIJW()
 @*/
#undef __FUNCT__
#define __FUNCT__ "MatIJGetMaxRowSize"
PetscErrorCode MatIJGetMaxRowSize(Mat A, PetscInt *maxsize)
{
  PetscErrorCode ierr;
  Mat_IJ *pg = (Mat_IJ *)(A->data);
  PetscBool isij;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  ierr = PetscObjectTypeCompare((PetscObject)A,MATIJ,&isij); CHKERRQ(ierr);
  if (!isij) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Matrix not of type MATIJ: %s", ((PetscObject)A)->type);
  MatIJCheckAssembled(A,PETSC_TRUE,1);
  *maxsize = pg->maxijlen;
  PetscFunctionReturn(0);
}

/*@C
  MatIJGetMinRowSize -   returns the largest number of nonzero columns in all of this processor's rows nonzero rows,
                           or zero if no local nonzero rows exist.
                             If MatIJ (equivalently, the underlying graph) is regarded as a multivalued
                           mapping on integers, then the result is the size of the smallest nonempty set among
                           the images of this processor's indices.  If all images are empty, the result is zero.
  Not collective.

  Input parameters:
. A        -    pseudograph

  Output parameters:
. minsize  - the size of the smallest nonempty image set

  Level: advanced

.seealso: MatIJGetMinRowSize()
 @*/
#undef __FUNCT__
#define __FUNCT__ "MatIJGetMinRowSize"
PetscErrorCode MatIJGetMinRowSize(Mat A, PetscInt *minsize)
{
  PetscErrorCode ierr;
  Mat_IJ *pg = (Mat_IJ *)(A->data);
  PetscBool isij;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  ierr = PetscObjectTypeCompare((PetscObject)A,MATIJ,&isij); CHKERRQ(ierr);
  if (!isij) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Matrix not of type MATIJ: %s", ((PetscObject)A)->type);
  MatIJCheckAssembled(A,PETSC_TRUE,1);
  *minsize = pg->minijlen;
  PetscFunctionReturn(0);
}




#undef  __FUNCT__
#define __FUNCT__ "MatDuplicate_IJ"
PetscErrorCode MatDuplicate_IJ(Mat A, MatDuplicateOption op, Mat *B)
{
  PetscErrorCode ierr;
  Mat_IJ* aij = (Mat_IJ*)(A->data), *bij;
  PetscFunctionBegin;
  MatIJCheckAssembled(A,PETSC_TRUE,1);
  ierr = MatCreate(((PetscObject)A)->comm, B);                            CHKERRQ(ierr);
  ierr = MatSetSizes(*B, A->rmap->n, A->cmap->n, A->rmap->N, A->cmap->N); CHKERRQ(ierr);
  ierr = MatSetType(*B, MATIJ);                                           CHKERRQ(ierr);
  bij = (Mat_IJ*)((*B)->data);
  if (aij->hsupp) PetscHashIDuplicate(aij->hsupp, bij->hsupp);             CHKERRQ(ierr);
  bij->m = aij->m;
  ierr = PetscMemcpy(bij->ijlen, aij->ijlen, sizeof(PetscInt)*(bij->m+1));     CHKERRQ(ierr);
  ierr = PetscMemcpy(bij->ij, aij->ij, sizeof(PetscInt)*(bij->ijlen[bij->m])); CHKERRQ(ierr);
  bij->n = aij->n;
  if (aij->image) {
    ierr = PetscMemcpy(bij->image, aij->image, sizeof(PetscInt)*bij->n);         CHKERRQ(ierr);
  }
  bij->maxijlen = aij->maxijlen;
  bij->minijlen = aij->minijlen;
  PetscFunctionReturn(0);
}

/*@C
   MatIJGetImageIS - retrieves the global indices of the graph's vertices of nonzero indegree
                     on this processor (i.e., the global indices of this processor's nonzero columns).
                     If the graph is regarded as a multivalued map on integers, this is
                     the image of the map (the union of the images of this processor's support indices).

   Not collective

   Input parameters:
.  A     -    pseudograph

   Output parameters:
.  image -    image IS

   Note:
+  This operation fails for a nonassembled matrix.
-  The caller is responsible for freeing the image IS.


   Level: intermediate

.seealso: MatIJGetImage(), MatIJGetSupportIS(), MatIJGetMaxRowSize()
@*/
#undef  __FUNCT__
#define __FUNCT__ "MatIJGetImageIS"
PetscErrorCode MatIJGetImageIS(Mat A, IS *image)
{
  PetscErrorCode ierr;
  Mat_IJ      *pg = (Mat_IJ *)(A->data);
  PetscBool      isij;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  ierr = PetscObjectTypeCompare((PetscObject)A,MATIJ,&isij); CHKERRQ(ierr);
  if (!isij) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Matrix not of type MATIJ: %s", ((PetscObject)A)->type);
  MatIJCheckAssembled(A,PETSC_TRUE,1);
  if (image) {
    ierr = ISCreateGeneral(A->cmap->comm, pg->n, pg->image, PETSC_COPY_VALUES, image); CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}


/*@C
   MatIJGetRowSizes - retrieves the numbers of edges emanating from the each of the supplied global indices,
                          provided they fall into the local ownership range.  Other indices result in an error.

   Not collective

   Input parameters:
+  A      -    pseudograph
.  intype -    (MATIJ_LOCAL | MATIJ_GLOBAL) meaning of inidxi: local support numbers or global indices
.  len    -    the length of the index list, or PETSC_DETERMINE to indicate all of the locally supported indices
-  inidxi -    array (of length len) of global indices whose image sizes are sought; ignored if len == PETSC_DETERMINE,
               which is equivalent to using all of the supported indices.

   Output parameters:
.  sizes  -    array (of length len) of image sizes of the global indices in inidxi

   Note:
+  This operation fails for a nonassembled matrix.
.  If len is PETSC_DEFAULT, inidxi must be PETSC_DEFAULT, and vice versa.
-  The caller is responsible for freeing sizes.


   Level: intermediate

.seealso: MatIJ, MatIJGetRowSizesSupported(), MatIJGetImage(), MatIJGetImageIS(), MatIJGetMaxRowSize()
@*/
#undef  __FUNCT__
#define __FUNCT__ "MatIJGetRowSizes"
PetscErrorCode MatIJGetRowSizes(Mat A, MatIJIndexType intype, PetscInt len, const PetscInt *inidxi, PetscInt **sizes)
{
  PetscErrorCode ierr;
  PetscBool      isij;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  ierr = PetscObjectTypeCompare((PetscObject)A,MATIJ,&isij); CHKERRQ(ierr);
  if (!isij) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Matrix not of type MATIJ: %s", ((PetscObject)A)->type);
  MatIJCheckAssembled(A,PETSC_TRUE,1);
  ierr = MatIJMap(A,intype,len,inidxi,PETSC_NULL,PETSC_NULL,MATIJ_GLOBAL,PETSC_NULL,PETSC_NULL,PETSC_NULL,PETSC_NULL,sizes); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   MatIJGetSupportSize - retrieves the total numbers of nonzero outdegree indices in the local ownership range
                         (the number of nonzero local rows).

   Not collective.

   Input parameters:
.  A      -    pseudograph

   Output parameters:
.  size   -    local support size

   Note:
.  This operation fails for a nonassembled matrix.


   Level: intermediate

.seealso: MATIJ, MatIJGetRowSizes(), MatIJGetImage(), MatIJGetImageIS(), MatIJGetMinRowSize(), MatIJGetMaxRowSize(), MatIJGetSupportSize()
@*/
#undef  __FUNCT__
#define __FUNCT__ "MatIJGetSupportSize"
PetscErrorCode MatIJGetSupportSize(Mat A, PetscInt *size)
{
  PetscErrorCode ierr;
  PetscBool      isij;
  Mat_IJ *pg = (Mat_IJ*)A->data;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  ierr = PetscObjectTypeCompare((PetscObject)A,MATIJ,&isij); CHKERRQ(ierr);
  if (!isij) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Matrix not of type MATIJ: %s", ((PetscObject)A)->type);
  PetscValidIntPointer(size,2);
  MatIJCheckAssembled(A,PETSC_TRUE,1);
  *size = pg->m;
  PetscFunctionReturn(0);
}


/*@C
   MatIJGetImageSize - retrieves the total numbers of target indices adjacent to the source indices in the local ownership
                       range (the number of nonzero local columns).

   Not collective.

   Input parameters:
.  A      -    pseudograph

   Output parameters:
.  size   -    local image size

   Note:
.  This operation fails for a nonassembled matrix.


   Level: intermediate

.seealso: MATIJ, MatIJGetSupportSize(), MatIJGetSupport(), MatIJGetSupportIS()
@*/
#undef  __FUNCT__
#define __FUNCT__ "MatIJGetImageSize"
PetscErrorCode MatIJGetImageSize(Mat A, PetscInt *size)
{
  PetscErrorCode ierr;
  PetscBool      isij;
  Mat_IJ *pg = (Mat_IJ*)A->data;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  ierr = PetscObjectTypeCompare((PetscObject)A,MATIJ,&isij); CHKERRQ(ierr);
  if (!isij) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Matrix not of type MATIJ: %s", ((PetscObject)A)->type);
  PetscValidIntPointer(size,2);
  MatIJCheckAssembled(A,PETSC_TRUE,1);
  ierr = MatIJLocalizeImage_Private(A); CHKERRQ(ierr);
  *size = pg->n;
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "MatIJBinRenumberLocal_Private"
PetscErrorCode MatIJBinRenumberLocal_Private(Mat A, MatIJIndexType intype, PetscInt insize, const PetscInt *inidxi, PetscInt *_outsize, PetscInt **_outidxi, PetscInt **_binsizes)
{
  PetscErrorCode ierr;
  PetscInt indi = -1, i,j,k,outsize = -1, *outidxi = PETSC_NULL, *binsizes = PETSC_NULL;
  Mat_IJ *pg = (Mat_IJ*)A->data;
  PetscFunctionBegin;


  /* We bin all of the input, and in the process assign bin-wise local numbers to them. */
  /* We'll need to count the contributions to each "bin" and the offset of each bin in outidx. */
  /* Binning requires a localized image. */
  ierr = MatIJLocalizeImage_Private(A); CHKERRQ(ierr);
  if ((_outidxi && !*_outidxi)) {
    ierr = MatIJBinRenumberLocal_Private(A,intype,insize,inidxi,&outsize,PETSC_NULL,PETSC_NULL); CHKERRQ(ierr);
  }
  if (insize == PETSC_DETERMINE){
    insize = pg->m;
    inidxi = PETSC_NULL;
  }
  else if (insize < 0)
    SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Invalid input array size: %D", insize);
  if (_outidxi) {
    if (!*_outidxi) {
      ierr = PetscMalloc(sizeof(PetscInt)*outsize, _outidxi);    CHKERRQ(ierr);
    }
    outidxi = *_outidxi;
  }
  if (_binsizes) {
    if (!*_binsizes) {
      ierr = PetscMalloc(sizeof(PetscInt)*(pg->n), _binsizes);    CHKERRQ(ierr);
    }
    binsizes = *_binsizes;
  }
  /* Allocate the bin offset array, if necessary. */
  if (!pg->binoffsets) {
    ierr = PetscMalloc((pg->n+1)*sizeof(PetscInt), &(pg->binoffsets)); CHKERRQ(ierr);
  }
  /* Initialize bin offsets */
  for (j = 0; j <= pg->n; ++j) {
    pg->binoffsets[j] = 0;
  }

  for (i = 0; i < insize; ++i) {
    if (!inidxi) {
      indi = i;
    }
    else {
    /* Convert to local. */
      MatIJGetSuppIndex_Private(A,intype,inidxi[i],indi);
    }
    if ((indi < 0 || indi > pg->m)){
      /* drop */
      continue;
    }
    for (k = pg->ijlen[indi]; k < pg->ijlen[indi+1]; ++k) {
      ++(pg->binoffsets[pg->ij[k]+1]);
    }
  }/* for (i = 0; i < insize; ++i) */
  /* Convert bin sizes into bin offsets. */
  for (j = 0; j < pg->n; ++j) {
    pg->binoffsets[j+1] += pg->binoffsets[j];
  }
  /* Now bin the input indices and values. */
  if (outidxi) {
    /* Allocate the bin size array, if necessary. */
    if (!binsizes) {
      if (!pg->binsizes) {
        ierr = PetscMalloc((pg->n)*sizeof(PetscInt), &(pg->binsizes)); CHKERRQ(ierr);
      }
      binsizes = pg->binsizes;
    }
    /* Initialize bin sizes to zero. */
    for (j = 0; j < pg->n; ++j) {
      binsizes[j] = 0;
    }
    for (i = 0; i < insize; ++i) {
      if (!inidxi) {
        indi = i;
      }
      else {
        /* Convert to local. */
        MatIJGetSuppIndex_Private(A,intype,inidxi[i],indi);
        if ((indi < 0 || indi > pg->m)){
          /* drop */
          continue;
        }
      }
      for (k = pg->ijlen[indi]; k < pg->ijlen[indi+1]; ++k) {
        j = pg->ij[k];
        outidxi[pg->binoffsets[j]+binsizes[j]] = binsizes[j];
        ++binsizes[j];
      }
    }/* for (i = 0; i < insize; ++i) */
  }/* if (outidxi) */
  if (_outsize) *_outsize = pg->binoffsets[pg->n];
  PetscFunctionReturn(0);
}


/*@C
   MatIJBinRenumber      - map the support indices to their global numbers within their image bins.
                             If the image indices are interpreted as colors labeling subdomains, then
                             each global subdomain is given a new contiguous zero-based numbering
                             uniquely defined by the following: the new vertex numbers increase with the
                             owning processor's rank, and within each rank they are arranged according
                             to their order in the local portion of the bin.


   Collective on A.

   Input arguments:
.  A           - pseudograph

   Output arguments:
.  B           - renumbering pseudograph

   Level:    advanced

   Notes: observe that each local support index might be mapped to the same global index multiple times,
          if it happens to have the same number within different bins. In order to decide which color
          each of the new numbers refers to, it is useful to use the result B in conjunction with
          the original pseudograph A as the second and first argument to MatIJBinMap(), respectively.
          By construction, B is compatible to A in the sense of MatIJBinMap().

.keywords: pseudograph, coloring, binning, numbering
.seealso:  MatIJBinMap()
@*/
#undef  __FUNCT__
#define __FUNCT__ "MatIJBinRenumber"
PetscErrorCode MatIJBinRenumber(Mat A, Mat *B)
{
  PetscErrorCode ierr;
  Mat_IJ *pg = (Mat_IJ*)A->data;
  PetscInt iysize = -1, len, *ixidx, *iyidx = PETSC_NULL, *bsizes = PETSC_NULL, N = 0, g,b,blow,bhigh,j;
  PetscMPIInt rank, size, NN, bsize,goffset;

  PetscFunctionBegin;
  ierr = MatIJLocalizeImage_Private(A); CHKERRQ(ierr);
  ierr = MatIJBinRenumberLocal_Private(A, MATIJ_LOCAL, pg->m, PETSC_NULL, &iysize, &iyidx, &bsizes); CHKERRQ(ierr);
  ierr = MPI_Comm_size(((PetscObject)A)->comm, &size); CHKERRQ(ierr);
  ierr = MPI_Comm_rank(((PetscObject)A)->comm, &rank); CHKERRQ(ierr);
  /*
   Since the new mapping is to the global bin numberings, we need to adjust the numberings further,
   by determining the local offsets for each bin: the sizes of each bin on the preceeding
   processors within the communicator. This is accomplished by a series of scan ops, one per bin.
   */
  /*
   In addition, we need to know the largest global bin size N, which can be determined by looking at
   bin offset of the process with the largest rank in the comm, adding the local bin size to it,
   taking the max across the bins and then broadcasting it to the other processes in the comm.
   */
  /* Loop over the global bins g; count the locally-supported bins b. */
  b = 0;
  blow = bhigh = 0;
  for (g = 0, b = 0; g < A->cmap->N; ++g) {
    if (pg->image[b] == g) {
      bsize = PetscMPIIntCast(bsizes[b]);
    }
    else {
      bsize = 0;
    }
    ierr = MPI_Scan(&bsize,&goffset,1,MPI_INT, MPI_SUM,((PetscObject)A)->comm); CHKERRQ(ierr);
    if (pg->image[b] == g) {
      blow = bhigh;
      bhigh = blow + bsizes[b];
      /* Shift the indices by the goffset. */
      for (j = blow; j < bhigh; ++j)  iyidx[j] += goffset;
      /* Compute the maximum partial global bin size, up to and including this proc's portion. */
      NN = PetscMPIIntCast(PetscMax(NN,goffset + bsizes[b]));
      ++b;
    }
    else {
      /* Compute the maximum partial global bin size, up to and including this proc's portion. */
      NN = PetscMPIIntCast(PetscMax(NN,goffset));
    }
  }/* Loop over the global bins. */
  if (b != pg->n) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Traversed %D local bins, not the same as expected: %D", b, pg->n);
  /* Broadcast from the last rank the largest global bin size. */
  ierr = MPI_Bcast(&NN,1,MPI_INT, rank-1,((PetscObject)A)->comm); CHKERRQ(ierr);
  N = NN;
  /* Now construct the new pseudograph. */
  /* The number of edges and the source vertices are the same as in the old pseudograph. */
  ierr = MatIJGetEdges(A,&len,&ixidx,PETSC_NULL); CHKERRQ(ierr);
  /* But instead of terminating at bins, the edges terminate at the local numbers within the bin. */
#if defined PETSC_USE_DEBUG
  {
    PetscInt k,blen = 0;
    for (k = 0; k < pg->n; ++k) blen += bsizes[k];
  if (len != blen)
    SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Number of edges in the original pseudograph %D and the renumbering pseudograph %D do not match", len, blen);
  }
#endif
  ierr = MatCreate(((PetscObject)A)->comm, B); CHKERRQ(ierr);
  ierr = MatSetSizes(*B, A->rmap->n, PETSC_DETERMINE, PETSC_DETERMINE, N); CHKERRQ(ierr);
  ierr = MatIJSetEdges(*B, len, ixidx, iyidx); CHKERRQ(ierr);
  /* All ixidx indices are within the local ownership range, so no parallel assembly is required. */
  ierr = MatIJSetEdgesLocal_Private(*B, len, ixidx, iyidx); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "MatTranspose_IJ"
PetscErrorCode MatTranspose_IJ(Mat A, MatReuse reuse, Mat *B)
{
  PetscErrorCode ierr;
  PetscBool multivalued;
  IS ix, iy;
  PetscFunctionBegin;
  ierr = MatCreate(((PetscObject)A)->comm, B);                             CHKERRQ(ierr);
  ierr = MatSetSizes(*B, A->cmap->n, A->rmap->n, A->cmap->N, A->rmap->N);  CHKERRQ(ierr);
  ierr = MatSetType(*B, MATIJ);                                            CHKERRQ(ierr);
  ierr = MatIJGetMultivalued(A,&multivalued);                              CHKERRQ(ierr);
  ierr = MatIJGetEdgesIS(A, &ix, &iy);                                     CHKERRQ(ierr);
  ierr = MatIJSetEdgesIS(*B, iy,ix);                                       CHKERRQ(ierr);
  ierr = ISDestroy(&(ix));                                                 CHKERRQ(ierr);
  ierr = ISDestroy(&(iy));                                                 CHKERRQ(ierr);
  ierr = MatAssemblyBegin(*B, MAT_FINAL_ASSEMBLY);                         CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*B, MAT_FINAL_ASSEMBLY);                           CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "MatTransposeMatMult_IJ_IJ"
PetscErrorCode MatTransposeMatMult_IJ_IJ(Mat A, Mat B, MatReuse reuse, PetscReal fill, Mat *CC)
{
  PetscErrorCode ierr;
  Mat C;
  PetscInt nsupp1, nsupp2, nsupp3, *supp1 = PETSC_NULL, *supp2 = PETSC_NULL, *supp3, imgsize1, *imgsizes1 = PETSC_NULL,
           imgsize2, *imgsizes2 = PETSC_NULL, *image1 = PETSC_NULL, *image2 = PETSC_NULL,
           *ixidx, *iyidx, count, i1,i2,i1low,i1high,i2low,i2high,k;
  PetscFunctionBegin;
  PetscCheckSameComm(A,1,B,2);
  /*
                                                        C     _
                                                       ...    |
      |-----|                                |-----|  ------> |
         ^                                      ^             |
         |                    ======>           |             -
    A    |                                  A   |
   ...   |       B      _                  ...  |
         |      ...     |                       |
      |-----|  ------>  |                    |-----|
                        |
                        -
   */
  ierr = MatIJGetSupport(A,  &nsupp1, &supp1);  CHKERRQ(ierr);
  ierr = MatIJGetSupport(B,  &nsupp2, &supp2);  CHKERRQ(ierr);
  /* Avoid computing the intersection, which may be unscalable in storage. */
  /*
   Count the number of images of the intersection of supports under the "upward" (A) and "rightward" (B) maps.
   It is done this way: supp1 is mapped by B obtaining offsets2, and supp2 is mapped by A obtaining offsets1.
   */
  ierr = MatIJMap(A,MATIJ_GLOBAL,nsupp2,supp2,PETSC_NULL,PETSC_NULL,MATIJ_GLOBAL,&imgsize1,PETSC_NULL,PETSC_NULL,PETSC_NULL,&imgsizes1); CHKERRQ(ierr);
  ierr = MatIJMap(B,MATIJ_GLOBAL,nsupp1,supp1,PETSC_NULL,PETSC_NULL,MATIJ_GLOBAL,&imgsize2,PETSC_NULL,PETSC_NULL,PETSC_NULL,&imgsizes2); CHKERRQ(ierr);
  /* Count the number of supp1 indices with nonzero images in B -- that's the size of the intersection. */
  nsupp3 = 0;
  for (k = 0; k < nsupp1; ++k) nsupp3 += (imgsizes1[k]>0);
#if defined(PETSC_USE_DEBUG)
  /* Now count the number of supp2 indices with nonzero images in map1: should be the same. */
  {
    PetscInt nsupp3_2 = 0;
    for (k = 0; k < nsupp2; ++k) nsupp3_2 += (imgsizes2[k]>0);
    if (nsupp3 != nsupp3_2) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Intersections supports different in map1: %D and map2: %D", nsupp3, nsupp3_2);
  }
#endif
  /* Allocate indices for the intersection. */
  ierr = PetscMalloc(sizeof(PetscInt)*nsupp3, &supp3); CHKERRQ(ierr);
  nsupp3 = 0;
  for (k = 0; k < nsupp2; ++k) {
    if (imgsizes1[k]) {
      supp3[nsupp3] =  supp2[k];
      ++nsupp3;
    }
  }
  ierr = PetscFree(supp1);                      CHKERRQ(ierr);
  ierr = PetscFree(supp2);                      CHKERRQ(ierr);

  /*
   Now obtain the "up" (A) and "right" (B) images of supp3.
   Recall that imgsizes1 are allocated for lsupp2, and imgsizes2 for lsupp1.
   */
  /* Reuse imgsizes1 and imgsizes2, even though they are a bit bigger, than necessary now and full of junk from the previous calls. Saves malloc/free.*/
  ierr = MatIJMap(A,MATIJ_GLOBAL,nsupp3,supp3,PETSC_NULL,PETSC_NULL,MATIJ_GLOBAL,&imgsize1,&image1,PETSC_NULL,PETSC_NULL,&imgsizes1);  CHKERRQ(ierr);
  ierr = MatIJMap(B,MATIJ_GLOBAL,nsupp3,supp3,PETSC_NULL,PETSC_NULL,MATIJ_GLOBAL,&imgsize2,&image2,PETSC_NULL,PETSC_NULL,&imgsizes2);  CHKERRQ(ierr);
  ierr = PetscFree(supp3);  CHKERRQ(ierr);

  /* Count the total number of arrows to add to the pushed forward MatIJ. */
  count = 0;
  for (k = 0; k < nsupp3; ++k) {
    count += (imgsizes1[k])*(imgsizes2[k]);
  }
  /* Allocate storage for the composed indices. */
  ierr = PetscMalloc2(count, PetscInt, &ixidx, count, PetscInt, &iyidx); CHKERRQ(ierr);
  count= 0;
  i1low = 0;
  i2low = 0;
  for (k = 0; k < nsupp3; ++k) {
    i1high = i1low + imgsizes1[k];
    i2high = i2low + imgsizes2[k];
    for (i1 = i1low; i1 < i1high; ++i1) {
      for (i2 = i2low; i1 < i2high; ++i2) {
        ixidx[count] = image1[i1];
        iyidx[count] = image2[i2];
        ++count;
      }
    }
    i1low = i1high;
    i2low = i2high;
  }
  ierr = PetscFree(image1);    CHKERRQ(ierr);
  ierr = PetscFree(image2);    CHKERRQ(ierr);
  ierr = PetscFree(imgsizes1); CHKERRQ(ierr);
  ierr = PetscFree(imgsizes2); CHKERRQ(ierr);
  /* Now construct the new MatIJ. */
  ierr = MatCreate(((PetscObject)A)->comm, &C);                          CHKERRQ(ierr);
  ierr = MatSetSizes(C, A->cmap->n, B->cmap->n, A->cmap->N, B->cmap->N); CHKERRQ(ierr);
  ierr = MatSetType(C, MATIJ);                                           CHKERRQ(ierr);
  ierr = MatIJSetEdges(C,count,ixidx,iyidx);                             CHKERRQ(ierr);
  ierr = MatAssemblyBegin(C, MAT_FINAL_ASSEMBLY);                        CHKERRQ(ierr);
  ierr = MatAssemblyEnd(C, MAT_FINAL_ASSEMBLY);                          CHKERRQ(ierr);
  ierr = PetscFree2(ixidx,iyidx);                                        CHKERRQ(ierr);

  *CC = C;
  PetscFunctionReturn(0);
}





#undef  __FUNCT__
#define __FUNCT__ "MatMatMult_IJ_IJ"
PetscErrorCode MatMatMult_IJ_IJ(Mat A, Mat B, MatReuse reuse, PetscReal fill, Mat *CC)
{
  PetscErrorCode ierr;
  Mat At,C;

  PetscFunctionBegin;
  PetscCheckSameComm(A,1,B,2);
  /*

                  B     _                              B     _
                 ...    |                             ...    |
       |-----|  ------> |                 |-----|    ------> |
          ^             |                    ^               |
          |             -                    |               -
      A   |                ======>      A    |
     ...  |                            ...   |         C     _
          |                                  |        ...    |
       |-----|                            |-----|    ------> |
                                                             |
                                                             -
   Convert this to a pushforward by transposing A  to At and then pushing B forward along At
   (reflect the second diagram with respect to a horizontal axis and then compare with Pushforward,
    of just push forward "downward".)

                  B     _                             B     _                            B     _
                 ...    |                            ...    |                           ...    |
       |-----|  ------> |                  |-----|  ------> |                |-----|   ------> |
          ^             |                     |             |                   |              |
          |             -                     |             -                   |              -
      A   |                ======>       At   |                 ======>   At    |
     ...  |                              ...  |                           ...   |        C     _
          |                                   V                                 V       ...    |
       |-----|                             |-----|                           |-----|   ------> |
                                                                                               |
                                                                                               -
   */
  ierr = MatTranspose(A, MAT_INITIAL_MATRIX, &At);                CHKERRQ(ierr);
  ierr = MatTransposeMatMult(At, B, MAT_INITIAL_MATRIX, 1.0, &C); CHKERRQ(ierr);
  ierr = MatDestroy(&At);                                         CHKERRQ(ierr);
  C->ops->matmult = MatMatMult_IJ_IJ;
  *CC = C;
  PetscFunctionReturn(0);
}


#undef  __FUNCT__
#define __FUNCT__ "MatZeroEntries_IJ"
PetscErrorCode MatZeroEntries_IJ(Mat A)
{
  Mat_IJ *pg = (Mat_IJ*) A->data;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = MatIJClear_Private(A); CHKERRQ(ierr);
  ierr = MatStashMPIIJClear_Private(pg->stash); CHKERRQ(ierr);
  A->was_assembled = A->assembled;
  A->assembled = PETSC_FALSE;
  PetscFunctionReturn(0);
}


EXTERN_C_BEGIN
#undef  __FUNCT__
#define __FUNCT__ "MatView_IJ"
PetscErrorCode MatView_IJ(Mat A, PetscViewer v)
{
  Mat_IJ *pg = (Mat_IJ*) A->data;
  PetscBool      isij, isascii;
  PetscInt indi, indj,i=-1,j;
  PetscHashIIter it=0;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  ierr = PetscObjectTypeCompare((PetscObject)A,MATIJ,&isij); CHKERRQ(ierr);
  if (!isij) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Matrix not of type MATIJ: %s", ((PetscObject)A)->type);
  MatIJCheckAssembled(A,PETSC_TRUE,1);
  ierr = PetscObjectTypeCompare((PetscObject)v,PETSCVIEWERASCII,&isascii);CHKERRQ(ierr);
  if (!isascii) {
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_SUP,"Viewer type %s not supported",((PetscObject)v)->type_name);
  }
  if (!pg->hsupp) {
    i = A->rmap->rstart;
  }
  else {
    PetscHashIIterBegin(pg->hsupp,it);
  }
  while(1) {
    if (pg->hsupp) {
      if (PetscHashIIterAtEnd(pg->hsupp,it)) break;
      else {
        PetscHashIIterGetKeyVal(pg->hsupp,it,i,indi);
      }
    }
    else {
      if (i == A->rmap->rend)
        break;
      else
        indi = i - A->rmap->rstart;
    }
    ierr = PetscViewerASCIISynchronizedPrintf(v, "%D --> ", i); CHKERRQ(ierr);
    for (indj = pg->ijlen[indi]; indj < pg->ijlen[indi+1]; ++indj) {
      MatIJGetIndexImage_Private(A,MATIJ_GLOBAL,indj,j);
      ierr = PetscViewerASCIISynchronizedPrintf(v, "%D ", j);     CHKERRQ(ierr);
    }
  }
  ierr = PetscViewerFlush(v); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef  __FUNCT__
#define __FUNCT__ "MatDestroy_IJ"
PetscErrorCode MatDestroy_IJ(Mat A) {
  PetscErrorCode ierr;
  Mat_IJ          *pg = (Mat_IJ *)(A->data);

  PetscFunctionBegin;
  ierr = MatIJClear_Private(A); CHKERRQ(ierr);
  ierr = MatStashMPIIJDestroy_Private(&(pg->stash)); CHKERRQ(ierr);
  A->data = PETSC_NULL;

  ierr = PetscObjectComposeFunction((PetscObject)A,"MatMatMult_ij_ij_C", "",PETSC_NULL); CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatTransposeMatMult_ij_ij_C", "",PETSC_NULL); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "MatCreate_IJ"
PetscErrorCode MatCreate_IJ(Mat A)
{
  PetscErrorCode ierr;
  Mat_IJ         *pg;

  PetscFunctionBegin;
  ierr = PetscNewLog(A, Mat_IJ, &pg); CHKERRQ(ierr);
  A->data = (void*)pg;

  ierr = PetscLayoutSetUp(A->rmap);         CHKERRQ(ierr);
  ierr = PetscLayoutSetUp(A->cmap);         CHKERRQ(ierr);

  ierr = MatStashMPIIJCreate_Private(A->rmap, &(pg->stash));  CHKERRQ(ierr);

  ierr = PetscMemzero(A->ops,sizeof(*(A->ops)));CHKERRQ(ierr);
  A->ops->assemblybegin         = MatAssemblyBegin_IJ;
  A->ops->assemblyend           = MatAssemblyEnd_IJ;
  A->ops->zeroentries           = MatZeroEntries_IJ;
  A->ops->getrow                = MatGetRow_IJ;
  A->ops->restorerow            = MatRestoreRow_IJ;
  A->ops->duplicate             = MatDuplicate_IJ;
  A->ops->destroy               = MatDestroy_IJ;
  A->ops->view                  = MatView_IJ;

  ierr = MatRegisterOp(((PetscObject)A)->comm, PETSC_NULL, (PetscVoidFunction)MatMatMult_IJ_IJ, "MatMatMult", 2, MATIJ,MATIJ);  CHKERRQ(ierr);
  ierr = MatRegisterOp(((PetscObject)A)->comm, PETSC_NULL, (PetscVoidFunction)MatTransposeMatMult_IJ_IJ, "MatTransposeMatMult", 2, MATIJ,MATIJ);  CHKERRQ(ierr);
  ierr = PetscObjectChangeTypeName((PetscObject)A, MATIJ); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END
