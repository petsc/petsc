/*
   Implementation of DMStag, defining dimension-independent functions in the
   DM API. stag1d.c, stag2d.c, and stag3d.c may include dimension-specific
   implementations of DM API functions, and other files here contain additional
   DMStag-specific API functions, as well as internal functions.
*/
#include <petsc/private/dmstagimpl.h>
#include <petscsf.h>

static PetscErrorCode DMCreateFieldDecomposition_Stag(DM dm, PetscInt *len,char ***namelist, IS **islist, DM **dmlist)
{
  PetscErrorCode ierr;
  PetscInt       f0,f1,f2,f3,dof0,dof1,dof2,dof3,n_entries,k,d,cnt,n_fields,dim;
  DMStagStencil  *stencil0,*stencil1,*stencil2,*stencil3;

  PetscFunctionBegin;
  ierr = DMGetDimension(dm,&dim);CHKERRQ(ierr);
  ierr = DMStagGetDOF(dm,&dof0,&dof1,&dof2,&dof3);CHKERRQ(ierr);
  ierr = DMStagGetEntriesPerElement(dm,&n_entries);CHKERRQ(ierr);

  f0 = 1;
  f1 = f2 = f3 = 0;
  if (dim == 1) {
    f1 = 1;
  } else if (dim == 2) {
    f1 = 2;
    f2 = 1;
  } else if (dim == 3) {
    f1 = 3;
    f2 = 3;
    f3 = 1;
  } else SETERRQ1(PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"Unsupported dimension %" PetscInt_FMT,dim);

  ierr = PetscCalloc1(f0*dof0,&stencil0);CHKERRQ(ierr);
  ierr = PetscCalloc1(f1*dof1,&stencil1);CHKERRQ(ierr);
  if (dim >= 2) {
    ierr = PetscCalloc1(f2*dof2,&stencil2);CHKERRQ(ierr);
  }
  if (dim >= 3) {
    ierr = PetscCalloc1(f3*dof3,&stencil3);CHKERRQ(ierr);
  }
  for (k=0; k<f0; ++k) {
    for (d=0; d<dof0; ++d) {
      stencil0[dof0*k + d].i = 0; stencil0[dof0*k + d].j = 0; stencil0[dof0*k + d].j = 0;
    }
  }
  for (k=0; k<f1; ++k) {
    for (d=0; d<dof1; ++d) {
      stencil1[dof1*k + d].i = 0; stencil1[dof1*k + d].j = 0; stencil1[dof1*k + d].j = 0;
    }
  }
  if (dim >= 2) {
    for (k=0; k<f2; ++k) {
      for (d=0; d<dof2; ++d) {
        stencil2[dof2*k + d].i = 0; stencil2[dof2*k + d].j = 0; stencil2[dof2*k + d].j = 0;
      }
    }
  }
  if (dim >= 3) {
    for (k=0; k<f3; ++k) {
      for (d=0; d<dof3; ++d) {
        stencil3[dof3*k + d].i = 0; stencil3[dof3*k + d].j = 0; stencil3[dof3*k + d].j = 0;
      }
    }
  }

  n_fields = 0;
  if (dof0 != 0) ++n_fields;
  if (dof1 != 0) ++n_fields;
  if (dim >=2 && dof2 != 0) ++n_fields;
  if (dim >=3 && dof3 != 0) ++n_fields;
  if (len) *len = n_fields;

  if (islist) {
    ierr = PetscMalloc1(n_fields,islist);CHKERRQ(ierr);

    if (dim == 1) {
      /* face, element */
      for (d=0; d<dof0; ++d) {
        stencil0[d].loc = DMSTAG_LEFT;
        stencil0[d].c = d;
      }
      for (d=0; d<dof1; ++d) {
        stencil1[d].loc = DMSTAG_ELEMENT;
        stencil1[d].c = d;
      }
    } else if (dim == 2) {
      /* vertex, edge(down,left), element */
      for (d=0; d<dof0; ++d) {
        stencil0[d].loc = DMSTAG_DOWN_LEFT;
        stencil0[d].c = d;
      }
      /* edge */
      cnt = 0;
      for (d=0; d<dof1; ++d) {
        stencil1[cnt].loc = DMSTAG_DOWN;  stencil1[cnt].c = d;
        ++cnt;
      }
      for (d=0; d<dof1; ++d) {
        stencil1[cnt].loc = DMSTAG_LEFT;  stencil1[cnt].c = d;
        ++cnt;
      }
      /* element */
      for (d=0; d<dof2; ++d) {
        stencil2[d].loc = DMSTAG_ELEMENT;
        stencil2[d].c = d;
      }
    } else if (dim == 3) {
      /* vertex, edge(down,left), face(down,left,back), element */
      for (d=0; d<dof0; ++d) {
        stencil0[d].loc = DMSTAG_BACK_DOWN_LEFT;
        stencil0[d].c = d;
      }
      /* edges */
      cnt = 0;
      for (d=0; d<dof1; ++d) {
        stencil1[cnt].loc = DMSTAG_BACK_DOWN;  stencil1[cnt].c = d;
        ++cnt;
      }
      for (d=0; d<dof1; ++d) {
        stencil1[cnt].loc = DMSTAG_BACK_LEFT;  stencil1[cnt].c = d;
        ++cnt;
      }
      for (d=0; d<dof1; ++d) {
        stencil1[cnt].loc = DMSTAG_DOWN_LEFT;  stencil1[cnt].c = d;
        ++cnt;
      }
      /* faces */
      cnt = 0;
      for (d=0; d<dof2; ++d) {
        stencil2[cnt].loc = DMSTAG_BACK;  stencil2[cnt].c = d;
        ++cnt;
      }
      for (d=0; d<dof2; ++d) {
        stencil2[cnt].loc = DMSTAG_DOWN;  stencil2[cnt].c = d;
        ++cnt;
      }
      for (d=0; d<dof2; ++d) {
        stencil2[cnt].loc = DMSTAG_LEFT;  stencil2[cnt].c = d;
        ++cnt;
      }
      /* elements */
      for (d=0; d<dof3; ++d) {
        stencil3[d].loc = DMSTAG_ELEMENT;
        stencil3[d].c = d;
      }
    } else SETERRQ1(PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"Unsupported dimension %" PetscInt_FMT,dim);

    cnt = 0;
    if (dof0 != 0) {
      ierr = DMStagCreateISFromStencils(dm,f0*dof0,stencil0,&(*islist)[cnt]);CHKERRQ(ierr);CHKERRQ(ierr);
      ++cnt;
    }
    if (dof1 != 0) {
      ierr = DMStagCreateISFromStencils(dm,f1*dof1,stencil1,&(*islist)[cnt]);CHKERRQ(ierr);CHKERRQ(ierr);
      ++cnt;
    }
    if (dim >= 2 && dof2 != 0) {
      ierr = DMStagCreateISFromStencils(dm,f2*dof2,stencil2,&(*islist)[cnt]);CHKERRQ(ierr);CHKERRQ(ierr);
      ++cnt;
    }
    if (dim >= 3 && dof3 != 0) {
      ierr = DMStagCreateISFromStencils(dm,f3*dof3,stencil3,&(*islist)[cnt]);CHKERRQ(ierr);CHKERRQ(ierr);
      ++cnt;
    }
  }

  if (namelist) {
    ierr = PetscMalloc1(n_fields,namelist);CHKERRQ(ierr);
    cnt = 0;
    if (dim == 1) {
      if (dof0 != 0) {
        PetscStrallocpy("vertex",&(*namelist)[cnt]);
        ++cnt;
      }
      if (dof1 != 0) {
        PetscStrallocpy("element",&(*namelist)[cnt]);
        ++cnt;
      }
    } else if (dim == 2) {
      if (dof0 != 0) {
        PetscStrallocpy("vertex",&(*namelist)[cnt]);
        ++cnt;
      }
      if (dof1 != 0) {
        PetscStrallocpy("face",&(*namelist)[cnt]);
        ++cnt;
      }
      if (dof2 != 0) {
        PetscStrallocpy("element",&(*namelist)[cnt]);
        ++cnt;
      }
    } else if (dim == 3) {
      if (dof0 != 0) {
        PetscStrallocpy("vertex",&(*namelist)[cnt]);
        ++cnt;
      }
      if (dof1 != 0) {
        PetscStrallocpy("edge",&(*namelist)[cnt]);
        ++cnt;
      }
      if (dof2 != 0) {
        PetscStrallocpy("face",&(*namelist)[cnt]);
        ++cnt;
      }
      if (dof3 != 0) {
        PetscStrallocpy("element",&(*namelist)[cnt]);
        ++cnt;
      }
    }
  } else SETERRQ1(PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"Unsupported dimension %" PetscInt_FMT,dim);
  if (dmlist) {
    ierr = PetscMalloc1(n_fields,dmlist);CHKERRQ(ierr);
    cnt = 0;
    if (dof0 != 0) {
      ierr = DMStagCreateCompatibleDMStag(dm,dof0,0,0,0,&(*dmlist)[cnt]);CHKERRQ(ierr);
      ++cnt;
    }
    if (dof1 != 0) {
      ierr = DMStagCreateCompatibleDMStag(dm,0,dof1,0,0,&(*dmlist)[cnt]);CHKERRQ(ierr);
      ++cnt;
    }
    if (dim >= 2 && dof2 != 0) {
      ierr = DMStagCreateCompatibleDMStag(dm,0,0,dof2,0,&(*dmlist)[cnt]);CHKERRQ(ierr);
      ++cnt;
    }
    if (dim >= 3 && dof3 != 0) {
      ierr = DMStagCreateCompatibleDMStag(dm,0,0,0,dof3,&(*dmlist)[cnt]);CHKERRQ(ierr);
      ++cnt;
    }
  }
  ierr = PetscFree(stencil0);CHKERRQ(ierr);
  ierr = PetscFree(stencil1);CHKERRQ(ierr);
  if (dim >= 2) {
    ierr = PetscFree(stencil2);CHKERRQ(ierr);
  }
  if (dim >= 3) {
    ierr = PetscFree(stencil3);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode DMClone_Stag(DM dm,DM *newdm)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* Destroy the DM created by generic logic in DMClone() */
  if (*newdm) {
    ierr = DMDestroy(newdm);CHKERRQ(ierr);
  }
  ierr = DMStagDuplicateWithoutSetup(dm,PetscObjectComm((PetscObject)dm),newdm);CHKERRQ(ierr);
  ierr = DMSetUp(*newdm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode DMDestroy_Stag(DM dm)
{
  PetscErrorCode ierr;
  DM_Stag        *stag;
  PetscInt       i;

  PetscFunctionBegin;
  stag = (DM_Stag*)dm->data;
  for (i=0; i<DMSTAG_MAX_DIM; ++i) {
    ierr = PetscFree(stag->l[i]);CHKERRQ(ierr);
  }
  ierr = VecScatterDestroy(&stag->gtol);CHKERRQ(ierr);
  ierr = VecScatterDestroy(&stag->ltog_injective);CHKERRQ(ierr);
  ierr = PetscFree(stag->neighbors);CHKERRQ(ierr);
  ierr = PetscFree(stag->locationOffsets);CHKERRQ(ierr);
  ierr = PetscFree(stag->coordinateDMType);CHKERRQ(ierr);
  ierr = PetscFree(stag);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode DMCreateGlobalVector_Stag(DM dm,Vec *vec)
{
  PetscErrorCode  ierr;
  DM_Stag * const stag = (DM_Stag*)dm->data;

  PetscFunctionBegin;
  if (PetscUnlikely(!dm->setupcalled)) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_WRONGSTATE,"This function must be called after DMSetUp()");
  ierr = VecCreateMPI(PetscObjectComm((PetscObject)dm),stag->entries,PETSC_DECIDE,vec);CHKERRQ(ierr);
  ierr = VecSetDM(*vec,dm);CHKERRQ(ierr);
  /* Could set some ops, as DMDA does */
  ierr = VecSetLocalToGlobalMapping(*vec,dm->ltogmap);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode DMCreateLocalVector_Stag(DM dm,Vec *vec)
{
  PetscErrorCode  ierr;
  DM_Stag * const stag = (DM_Stag*)dm->data;

  PetscFunctionBegin;
  if (PetscUnlikely(!dm->setupcalled)) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_WRONGSTATE,"This function must be called after DMSetUp()");
  ierr = VecCreateSeq(PETSC_COMM_SELF,stag->entriesGhost,vec);CHKERRQ(ierr);
  ierr = VecSetBlockSize(*vec,stag->entriesPerElement);CHKERRQ(ierr);
  ierr = VecSetDM(*vec,dm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode DMCreateMatrix_Stag(DM dm,Mat *mat)
{
  PetscErrorCode         ierr;
  MatType                mat_type;
  PetscBool              is_shell,is_aij;
  PetscInt               dim,entries;
  ISLocalToGlobalMapping ltogmap;

  PetscFunctionBegin;
  if (PetscUnlikely(!dm->setupcalled)) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_WRONGSTATE,"This function must be called after DMSetUp()");
  ierr = DMGetDimension(dm,&dim);CHKERRQ(ierr);
  ierr = DMGetMatType(dm,&mat_type);CHKERRQ(ierr);
  ierr = DMStagGetEntries(dm,&entries);CHKERRQ(ierr);
  ierr = MatCreate(PetscObjectComm((PetscObject)dm),mat);CHKERRQ(ierr);
  ierr = MatSetSizes(*mat,entries,entries,PETSC_DETERMINE,PETSC_DETERMINE);CHKERRQ(ierr);
  ierr = MatSetType(*mat,mat_type);CHKERRQ(ierr);
  ierr = MatSetUp(*mat);CHKERRQ(ierr);
  ierr = DMGetLocalToGlobalMapping(dm,&ltogmap);CHKERRQ(ierr);
  ierr = MatSetLocalToGlobalMapping(*mat,ltogmap,ltogmap);CHKERRQ(ierr);
  ierr = MatSetDM(*mat,dm);CHKERRQ(ierr);

  /* Compare to similar and perhaps superior logic in DMCreateMatrix_DA, which creates
     the matrix first and then performs this logic by checking for preallocation functions */
  ierr = PetscStrcmp(mat_type,MATAIJ,&is_aij);CHKERRQ(ierr);
  if (!is_aij) {
    ierr = PetscStrcmp(mat_type,MATSEQAIJ,&is_aij);CHKERRQ(ierr);
  } else if (!is_aij) {
    ierr = PetscStrcmp(mat_type,MATMPIAIJ,&is_aij);CHKERRQ(ierr);
  }
  ierr = PetscStrcmp(mat_type,MATSHELL,&is_shell);CHKERRQ(ierr);
  if (is_aij) {
    Mat             preallocator;
    PetscInt        m,n;
    const PetscBool fill_with_zeros = PETSC_FALSE;

    ierr = MatCreate(PetscObjectComm((PetscObject)dm),&preallocator);CHKERRQ(ierr);
    ierr = MatSetType(preallocator,MATPREALLOCATOR);CHKERRQ(ierr);
    ierr = MatGetLocalSize(*mat,&m,&n);CHKERRQ(ierr);
    ierr = MatSetSizes(preallocator,m,n,PETSC_DECIDE,PETSC_DECIDE);CHKERRQ(ierr);
    ierr = MatSetLocalToGlobalMapping(preallocator,ltogmap,ltogmap);CHKERRQ(ierr);
    ierr = MatSetUp(preallocator);CHKERRQ(ierr);
    switch (dim) {
      case 1:
        ierr = DMCreateMatrix_Stag_1D_AIJ_Assemble(dm,preallocator);CHKERRQ(ierr);
        break;
      case 2:
        ierr = DMCreateMatrix_Stag_2D_AIJ_Assemble(dm,preallocator);CHKERRQ(ierr);
        break;
      case 3:
        ierr = DMCreateMatrix_Stag_3D_AIJ_Assemble(dm,preallocator);CHKERRQ(ierr);
        break;
      default: SETERRQ1(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_OUTOFRANGE,"Unsupported dimension %" PetscInt_FMT,dim);
    }
    ierr = MatPreallocatorPreallocate(preallocator,fill_with_zeros,*mat);CHKERRQ(ierr);
    ierr = MatDestroy(&preallocator);CHKERRQ(ierr);

    if (!dm->prealloc_only) {
      switch (dim) {
        case 1:
          ierr = DMCreateMatrix_Stag_1D_AIJ_Assemble(dm,*mat);CHKERRQ(ierr);
          break;
        case 2:
          ierr = DMCreateMatrix_Stag_2D_AIJ_Assemble(dm,*mat);CHKERRQ(ierr);
          break;
        case 3:
          ierr = DMCreateMatrix_Stag_3D_AIJ_Assemble(dm,*mat);CHKERRQ(ierr);
          break;
        default: SETERRQ1(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_OUTOFRANGE,"Unsupported dimension %" PetscInt_FMT,dim);
      }
    }
    /* Note: GPU-related logic, e.g. at the end of DMCreateMatrix_DA_1d_MPIAIJ, is not included here
       but might be desirable */
  } else if (is_shell) {
    /* nothing more to do */
  } else SETERRQ1(PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"Not implemented for Mattype %s",mat_type);
  PetscFunctionReturn(0);
}

static PetscErrorCode DMGetCompatibility_Stag(DM dm,DM dm2,PetscBool *compatible,PetscBool *set)
{
  PetscErrorCode  ierr;
  const DM_Stag * const stag  = (DM_Stag*)dm->data;
  const DM_Stag * const stag2 = (DM_Stag*)dm2->data;
  PetscInt              dim,dim2,i;
  MPI_Comm              comm;
  PetscMPIInt           sameComm;
  DMType                type2;
  PetscBool             sameType;

  PetscFunctionBegin;
  ierr = DMGetType(dm2,&type2);CHKERRQ(ierr);
  ierr = PetscStrcmp(DMSTAG,type2,&sameType);CHKERRQ(ierr);
  if (!sameType) {
    ierr = PetscInfo1((PetscObject)dm,"DMStag compatibility check not implemented with DM of type %s\n",type2);CHKERRQ(ierr);
    *set = PETSC_FALSE;
    PetscFunctionReturn(0);
  }

  ierr = PetscObjectGetComm((PetscObject)dm,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_compare(comm,PetscObjectComm((PetscObject)dm2),&sameComm);CHKERRMPI(ierr);
  if (sameComm != MPI_IDENT) {
    ierr = PetscInfo2((PetscObject)dm,"DMStag objects have different communicators: %d != %d\n",comm,PetscObjectComm((PetscObject)dm2));CHKERRQ(ierr);
    *set = PETSC_FALSE;
    PetscFunctionReturn(0);
  }
  ierr = DMGetDimension(dm ,&dim);CHKERRQ(ierr);
  ierr = DMGetDimension(dm2,&dim2);CHKERRQ(ierr);
  if (dim != dim2) {
    ierr = PetscInfo((PetscObject)dm,"DMStag objects have different dimensions");CHKERRQ(ierr);
    *set = PETSC_TRUE;
    *compatible = PETSC_FALSE;
    PetscFunctionReturn(0);
  }
  for (i=0; i<dim; ++i) {
    if (stag->N[i] != stag2->N[i]) {
      ierr = PetscInfo3((PetscObject)dm,"DMStag objects have different global numbers of elements in dimension %D: %D != %D\n",i,stag->n[i],stag2->n[i]);CHKERRQ(ierr);
      *set = PETSC_TRUE;
      *compatible = PETSC_FALSE;
      PetscFunctionReturn(0);
    }
    if (stag->n[i] != stag2->n[i]) {
      ierr = PetscInfo3((PetscObject)dm,"DMStag objects have different local numbers of elements in dimension %D: %D != %D\n",i,stag->n[i],stag2->n[i]);CHKERRQ(ierr);
      *set = PETSC_TRUE;
      *compatible = PETSC_FALSE;
      PetscFunctionReturn(0);
    }
    if (stag->boundaryType[i] != stag2->boundaryType[i]) {
      ierr = PetscInfo3((PetscObject)dm,"DMStag objects have different boundary types in dimension %d: %s != %s\n",i,stag->boundaryType[i],stag2->boundaryType[i]);CHKERRQ(ierr);
      *set = PETSC_TRUE;
      *compatible = PETSC_FALSE;
      PetscFunctionReturn(0);
    }
  }
  /* Note: we include stencil type and width in the notion of compatibility, as this affects
     the "atlas" (local subdomains). This might be irritating in legitimate cases
     of wanting to transfer between two other-wise compatible DMs with different
     stencil characteristics. */
  if (stag->stencilType != stag2->stencilType) {
    ierr = PetscInfo2((PetscObject)dm,"DMStag objects have different ghost stencil types: %s != %s\n",DMStagStencilTypes[stag->stencilType],DMStagStencilTypes[stag2->stencilType]);CHKERRQ(ierr);
    *set = PETSC_TRUE;
    *compatible = PETSC_FALSE;
    PetscFunctionReturn(0);
  }
  if (stag->stencilWidth != stag2->stencilWidth) {
    ierr = PetscInfo2((PetscObject)dm,"DMStag objects have different ghost stencil widths: %D != %D\n",stag->stencilWidth,stag->stencilWidth);CHKERRQ(ierr);
    *set = PETSC_TRUE;
    *compatible = PETSC_FALSE;
    PetscFunctionReturn(0);
  }
  *set = PETSC_TRUE;
  *compatible = PETSC_TRUE;
  PetscFunctionReturn(0);
}

/*
Note there are several orderings in play here.
In all cases, non-element dof are associated with the element that they are below/left/behind, and the order in 2D proceeds vertex/bottom edge/left edge/element (with all dof on each together).
Also in all cases, only subdomains which are the last in their dimension have partial elements.

1) "Natural" Ordering (not used). Number adding each full or partial (on the right or top) element, starting at the bottom left (i=0,j=0) and proceeding across the entire domain, row by row to get a global numbering.
2) Global ("PETSc") ordering. The same as natural, but restricted to each domain. So, traverse all elements (again starting at the bottom left and going row-by-row) on rank 0, then continue numbering with rank 1, and so on.
3) Local ordering. Including ghost elements (both interior and on the right/top/front to complete partial elements), use the same convention to create a local numbering.
*/

static PetscErrorCode DMLocalToGlobalBegin_Stag(DM dm,Vec l,InsertMode mode,Vec g)
{
  PetscErrorCode  ierr;
  DM_Stag * const stag = (DM_Stag*)dm->data;

  PetscFunctionBegin;
  if (mode == ADD_VALUES) {
    ierr = VecScatterBegin(stag->gtol,l,g,mode,SCATTER_REVERSE);CHKERRQ(ierr);
  } else if (mode == INSERT_VALUES) {
    if (stag->ltog_injective) {
      ierr = VecScatterBegin(stag->ltog_injective,l,g,mode,SCATTER_FORWARD);CHKERRQ(ierr);
    } else {
      ierr = VecScatterBegin(stag->gtol,l,g,mode,SCATTER_REVERSE_LOCAL);CHKERRQ(ierr);
    }
  } else SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"Unsupported InsertMode");
  PetscFunctionReturn(0);
}

static PetscErrorCode DMLocalToGlobalEnd_Stag(DM dm,Vec l,InsertMode mode,Vec g)
{
  PetscErrorCode  ierr;
  DM_Stag * const stag = (DM_Stag*)dm->data;

  PetscFunctionBegin;
  if (mode == ADD_VALUES) {
    ierr = VecScatterEnd(stag->gtol,l,g,mode,SCATTER_REVERSE);CHKERRQ(ierr);
  } else if (mode == INSERT_VALUES) {
    if (stag->ltog_injective) {
      ierr = VecScatterEnd(stag->ltog_injective,l,g,mode,SCATTER_FORWARD);CHKERRQ(ierr);
    } else {
      ierr = VecScatterEnd(stag->gtol,l,g,mode,SCATTER_REVERSE_LOCAL);CHKERRQ(ierr);
    }
  } else SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"Unsupported InsertMode");
  PetscFunctionReturn(0);
}

static PetscErrorCode DMGlobalToLocalBegin_Stag(DM dm,Vec g,InsertMode mode,Vec l)
{
  PetscErrorCode  ierr;
  DM_Stag * const stag = (DM_Stag*)dm->data;

  PetscFunctionBegin;
  ierr = VecScatterBegin(stag->gtol,g,l,mode,SCATTER_FORWARD);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode DMGlobalToLocalEnd_Stag(DM dm,Vec g,InsertMode mode,Vec l)
{
  PetscErrorCode  ierr;
  DM_Stag * const stag = (DM_Stag*)dm->data;

  PetscFunctionBegin;
  ierr = VecScatterEnd(stag->gtol,g,l,mode,SCATTER_FORWARD);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
If a stratum is active (non-zero dof), make it active in the coordinate DM.
*/
static PetscErrorCode DMCreateCoordinateDM_Stag(DM dm,DM *dmc)
{
  PetscErrorCode  ierr;
  DM_Stag * const stag = (DM_Stag*)dm->data;
  PetscInt        dim;
  PetscBool       isstag,isproduct;

  PetscFunctionBegin;

  if (!stag->coordinateDMType) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_OUTOFRANGE,"Before creating a coordinate DM, a type must be specified with DMStagSetCoordinateDMType()");

  ierr = DMGetDimension(dm,&dim);CHKERRQ(ierr);
  ierr = PetscStrcmp(stag->coordinateDMType,DMSTAG,&isstag);CHKERRQ(ierr);
  ierr = PetscStrcmp(stag->coordinateDMType,DMPRODUCT,&isproduct);CHKERRQ(ierr);
  if (isstag) {
    ierr = DMStagCreateCompatibleDMStag(dm,
        stag->dof[0] > 0 ? dim : 0,
        stag->dof[1] > 0 ? dim : 0,
        stag->dof[2] > 0 ? dim : 0,
        stag->dof[3] > 0 ? dim : 0,
        dmc);CHKERRQ(ierr);
  } else if (isproduct) {
    ierr = DMCreate(PETSC_COMM_WORLD,dmc);CHKERRQ(ierr);
    ierr = DMSetType(*dmc,DMPRODUCT);CHKERRQ(ierr);
    ierr = DMSetDimension(*dmc,dim);CHKERRQ(ierr);
  } else SETERRQ1(PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"Unsupported coordinate DM type %s",stag->coordinateDMType);
  PetscFunctionReturn(0);
}

static PetscErrorCode DMGetNeighbors_Stag(DM dm,PetscInt *nRanks,const PetscMPIInt *ranks[])
{
  PetscErrorCode  ierr;
  DM_Stag * const stag = (DM_Stag*)dm->data;
  PetscInt        dim;

  PetscFunctionBegin;
  ierr = DMGetDimension(dm,&dim);CHKERRQ(ierr);
  switch (dim) {
    case 1: *nRanks = 3; break;
    case 2: *nRanks = 9; break;
    case 3: *nRanks = 27; break;
    default : SETERRQ1(PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"Get neighbors not implemented for dim = %D",dim);
  }
  *ranks = stag->neighbors;
  PetscFunctionReturn(0);
}

static PetscErrorCode DMView_Stag(DM dm,PetscViewer viewer)
{
  PetscErrorCode  ierr;
  DM_Stag * const stag = (DM_Stag*)dm->data;
  PetscBool       isascii,viewAllRanks;
  PetscMPIInt     rank,size;
  PetscInt        dim,maxRanksToView,i;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(PetscObjectComm((PetscObject)dm),&rank);CHKERRMPI(ierr);
  ierr = MPI_Comm_size(PetscObjectComm((PetscObject)dm),&size);CHKERRMPI(ierr);
  ierr = DMGetDimension(dm,&dim);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&isascii);CHKERRQ(ierr);
  if (isascii) {
    ierr = PetscViewerASCIIPrintf(viewer,"Dimension: %D\n",dim);CHKERRQ(ierr);
    switch (dim) {
      case 1:
        ierr = PetscViewerASCIIPrintf(viewer,"Global size: %D\n",stag->N[0]);CHKERRQ(ierr);
        break;
      case 2:
        ierr = PetscViewerASCIIPrintf(viewer,"Global sizes: %D x %D\n",stag->N[0],stag->N[1]);CHKERRQ(ierr);
        ierr = PetscViewerASCIIPrintf(viewer,"Parallel decomposition: %D x %D ranks\n",stag->nRanks[0],stag->nRanks[1]);CHKERRQ(ierr);
        break;
      case 3:
        ierr = PetscViewerASCIIPrintf(viewer,"Global sizes: %D x %D x %D\n",stag->N[0],stag->N[1],stag->N[2]);CHKERRQ(ierr);
        ierr = PetscViewerASCIIPrintf(viewer,"Parallel decomposition: %D x %D x %D ranks\n",stag->nRanks[0],stag->nRanks[1],stag->nRanks[2]);CHKERRQ(ierr);
        break;
      default: SETERRQ1(PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"not implemented for dim==%D",dim);
    }
    ierr = PetscViewerASCIIPrintf(viewer,"Boundary ghosting:");CHKERRQ(ierr);
    for (i=0; i<dim; ++i) {
      ierr = PetscViewerASCIIPrintf(viewer," %s",DMBoundaryTypes[stag->boundaryType[i]]);CHKERRQ(ierr);
    }
    ierr = PetscViewerASCIIPrintf(viewer,"\n");CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"Elementwise ghost stencil: %s",DMStagStencilTypes[stag->stencilType]);CHKERRQ(ierr);
    if (stag->stencilType != DMSTAG_STENCIL_NONE) {
      ierr = PetscViewerASCIIPrintf(viewer,", width %D\n",stag->stencilWidth);CHKERRQ(ierr);
    } else {
      ierr = PetscViewerASCIIPrintf(viewer,"\n");CHKERRQ(ierr);
    }
    ierr = PetscViewerASCIIPrintf(viewer,"%D DOF per vertex (0D)\n",stag->dof[0]);CHKERRQ(ierr);
    if (dim == 3) {
      ierr = PetscViewerASCIIPrintf(viewer,"%D DOF per edge (1D)\n",stag->dof[1]);CHKERRQ(ierr);
    }
    if (dim > 1) {
      ierr = PetscViewerASCIIPrintf(viewer,"%D DOF per face (%DD)\n",stag->dof[dim-1],dim-1);CHKERRQ(ierr);
    }
    ierr = PetscViewerASCIIPrintf(viewer,"%D DOF per element (%DD)\n",stag->dof[dim],dim);CHKERRQ(ierr);
    if (dm->coordinateDM) {
      ierr = PetscViewerASCIIPrintf(viewer,"Has coordinate DM\n");CHKERRQ(ierr);
    }
    maxRanksToView = 16;
    viewAllRanks = (PetscBool)(size <= maxRanksToView);
    if (viewAllRanks) {
      ierr = PetscViewerASCIIPushSynchronized(viewer);CHKERRQ(ierr);
      switch (dim) {
        case 1:
          ierr = PetscViewerASCIISynchronizedPrintf(viewer,"[%d] Local elements : %D (%D with ghosts)\n",rank,stag->n[0],stag->nGhost[0]);CHKERRQ(ierr);
          break;
        case 2:
          ierr = PetscViewerASCIISynchronizedPrintf(viewer,"[%d] Rank coordinates (%d,%d)\n",rank,stag->rank[0],stag->rank[1]);CHKERRQ(ierr);
          ierr = PetscViewerASCIISynchronizedPrintf(viewer,"[%d] Local elements : %D x %D (%D x %D with ghosts)\n",rank,stag->n[0],stag->n[1],stag->nGhost[0],stag->nGhost[1]);CHKERRQ(ierr);
          break;
        case 3:
          ierr = PetscViewerASCIISynchronizedPrintf(viewer,"[%d] Rank coordinates (%d,%d,%d)\n",rank,stag->rank[0],stag->rank[1],stag->rank[2]);CHKERRQ(ierr);
          ierr = PetscViewerASCIISynchronizedPrintf(viewer,"[%d] Local elements : %D x %D x %D (%D x %D x %D with ghosts)\n",rank,stag->n[0],stag->n[1],stag->n[2],stag->nGhost[0],stag->nGhost[1],stag->nGhost[2]);CHKERRQ(ierr);
          break;
        default: SETERRQ1(PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"not implemented for dim==%D",dim);
      }
      ierr = PetscViewerASCIISynchronizedPrintf(viewer,"[%d] Local native entries: %d\n",rank,stag->entries);CHKERRQ(ierr);
      ierr = PetscViewerASCIISynchronizedPrintf(viewer,"[%d] Local entries total : %d\n",rank,stag->entriesGhost);CHKERRQ(ierr);
      ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPopSynchronized(viewer);CHKERRQ(ierr);
    } else {
      ierr = PetscViewerASCIIPrintf(viewer,"(Per-rank information omitted since >%D ranks used)\n",maxRanksToView);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode DMSetFromOptions_Stag(PetscOptionItems *PetscOptionsObject,DM dm)
{
  PetscErrorCode  ierr;
  DM_Stag * const stag = (DM_Stag*)dm->data;
  PetscInt        dim;

  PetscFunctionBegin;
  ierr = DMGetDimension(dm,&dim);CHKERRQ(ierr);
  ierr = PetscOptionsHead(PetscOptionsObject,"DMStag Options");CHKERRQ(ierr);
  ierr = PetscOptionsInt("-stag_grid_x","Number of grid points in x direction","DMStagSetGlobalSizes",stag->N[0],&stag->N[0],NULL);CHKERRQ(ierr);
  if (dim > 1) { ierr = PetscOptionsInt("-stag_grid_y","Number of grid points in y direction","DMStagSetGlobalSizes",stag->N[1],&stag->N[1],NULL);CHKERRQ(ierr); }
  if (dim > 2) { ierr = PetscOptionsInt("-stag_grid_z","Number of grid points in z direction","DMStagSetGlobalSizes",stag->N[2],&stag->N[2],NULL);CHKERRQ(ierr); }
  ierr = PetscOptionsInt("-stag_ranks_x","Number of ranks in x direction","DMStagSetNumRanks",stag->nRanks[0],&stag->nRanks[0],NULL);CHKERRQ(ierr);
  if (dim > 1) { ierr = PetscOptionsInt("-stag_ranks_y","Number of ranks in y direction","DMStagSetNumRanks",stag->nRanks[1],&stag->nRanks[1],NULL);CHKERRQ(ierr); }
  if (dim > 2) { ierr = PetscOptionsInt("-stag_ranks_z","Number of ranks in z direction","DMStagSetNumRanks",stag->nRanks[2],&stag->nRanks[2],NULL);CHKERRQ(ierr); }
  ierr = PetscOptionsInt("-stag_stencil_width","Elementwise stencil width","DMStagSetStencilWidth",stag->stencilWidth,&stag->stencilWidth,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnum("-stag_stencil_type","Elementwise stencil stype","DMStagSetStencilType",DMStagStencilTypes,(PetscEnum)stag->stencilType,(PetscEnum*)&stag->stencilType,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnum("-stag_boundary_type_x","Treatment of (physical) boundaries in x direction","DMStagSetBoundaryTypes",DMBoundaryTypes,(PetscEnum)stag->boundaryType[0],(PetscEnum*)&stag->boundaryType[0],NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnum("-stag_boundary_type_y","Treatment of (physical) boundaries in y direction","DMStagSetBoundaryTypes",DMBoundaryTypes,(PetscEnum)stag->boundaryType[1],(PetscEnum*)&stag->boundaryType[1],NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnum("-stag_boundary_type_z","Treatment of (physical) boundaries in z direction","DMStagSetBoundaryTypes",DMBoundaryTypes,(PetscEnum)stag->boundaryType[2],(PetscEnum*)&stag->boundaryType[2],NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-stag_dof_0","Number of dof per 0-cell (vertex)","DMStagSetDOF",stag->dof[0],&stag->dof[0],NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-stag_dof_1","Number of dof per 1-cell (element in 1D, face in 2D, edge in 3D)","DMStagSetDOF",stag->dof[1],&stag->dof[1],NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-stag_dof_2","Number of dof per 2-cell (element in 2D, face in 3D)","DMStagSetDOF",stag->dof[2],&stag->dof[2],NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-stag_dof_3","Number of dof per 3-cell (element in 3D)","DMStagSetDOF",stag->dof[3],&stag->dof[3],NULL);CHKERRQ(ierr);
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*MC
  DMSTAG = "stag" - A DM object representing a "staggered grid" or a structured cell complex.

  This implementation parallels the DMDA implementation in many ways, but allows degrees of freedom
  to be associated with all "strata" in a logically-rectangular grid.

  Each stratum can be characterized by the dimension of the entities ("points", to borrow the DMPLEX
  terminology), from 0- to 3-dimensional.

  In some cases this numbering is used directly, for example with DMStagGetDOF().
  To allow easier reading and to some extent more similar code between different-dimensional implementations
  of the same problem, we associate canonical names for each type of point, for each dimension of DMStag.

  1-dimensional DMStag objects have vertices (0D) and elements (1D).

  2-dimensional DMStag objects have vertices (0D), faces (1D), and elements (2D).

  3-dimensional DMStag objects have vertices (0D), edges (1D), faces (2D), and elements (3D).

  This naming is reflected when viewing a DMStag object with DMView() , and in forming
  convenient options prefixes when creating a decomposition with DMCreateFieldDecomposition().

  Level: beginner

.seealso: DM, DMPRODUCT, DMDA, DMPLEX, DMStagCreate1d(), DMStagCreate2d(), DMStagCreate3d(), DMType, DMCreate(), DMSetType()
M*/

PETSC_EXTERN PetscErrorCode DMCreate_Stag(DM dm)
{
  PetscErrorCode ierr;
  DM_Stag        *stag;
  PetscInt       i,dim;

  PetscFunctionBegin;
  PetscValidPointer(dm,1);
  ierr = PetscNewLog(dm,&stag);CHKERRQ(ierr);
  dm->data = stag;

  stag->gtol                                          = NULL;
  stag->ltog_injective                                = NULL;
  for (i=0; i<DMSTAG_MAX_STRATA; ++i) stag->dof[i]    = 0;
  for (i=0; i<DMSTAG_MAX_DIM;    ++i) stag->l[i]      = NULL;
  stag->stencilType                                   = DMSTAG_STENCIL_NONE;
  stag->stencilWidth                                  = 0;
  for (i=0; i<DMSTAG_MAX_DIM;    ++i) stag->nRanks[i] = -1;
  stag->coordinateDMType                              = NULL;

  ierr = DMGetDimension(dm,&dim);CHKERRQ(ierr);
  if (dim != 1 && dim != 2 && dim != 3) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_WRONGSTATE,"DMSetDimension() must be called to set a dimension with value 1, 2, or 3");

  ierr = PetscMemzero(dm->ops,sizeof(*(dm->ops)));CHKERRQ(ierr);
  dm->ops->createcoordinatedm       = DMCreateCoordinateDM_Stag;
  dm->ops->createglobalvector       = DMCreateGlobalVector_Stag;
  dm->ops->createinterpolation      = NULL;
  dm->ops->createlocalvector        = DMCreateLocalVector_Stag;
  dm->ops->creatematrix             = DMCreateMatrix_Stag;
  dm->ops->destroy                  = DMDestroy_Stag;
  dm->ops->getneighbors             = DMGetNeighbors_Stag;
  dm->ops->globaltolocalbegin       = DMGlobalToLocalBegin_Stag;
  dm->ops->globaltolocalend         = DMGlobalToLocalEnd_Stag;
  dm->ops->localtoglobalbegin       = DMLocalToGlobalBegin_Stag;
  dm->ops->localtoglobalend         = DMLocalToGlobalEnd_Stag;
  dm->ops->setfromoptions           = DMSetFromOptions_Stag;
  switch (dim) {
    case 1: dm->ops->setup          = DMSetUp_Stag_1d; break;
    case 2: dm->ops->setup          = DMSetUp_Stag_2d; break;
    case 3: dm->ops->setup          = DMSetUp_Stag_3d; break;
    default : SETERRQ1(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_OUTOFRANGE,"Unsupported dimension %d",dim);
  }
  dm->ops->clone                    = DMClone_Stag;
  dm->ops->view                     = DMView_Stag;
  dm->ops->getcompatibility         = DMGetCompatibility_Stag;
  dm->ops->createfielddecomposition = DMCreateFieldDecomposition_Stag;
  PetscFunctionReturn(0);
}
