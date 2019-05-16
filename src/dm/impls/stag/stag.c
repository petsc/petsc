/*
   Implementation of DMStag, defining dimension-independent functions in the
   DM API. stag1d.c, stag2d.c, and stag3d.c may include dimension-specific
   implementations of DM API functions, and other files here contain additional
   DMStag-specific API functions (and internal functions).
*/
#include <petsc/private/dmstagimpl.h>
#include <petscsf.h>

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
  ierr = VecCreateSeq(PETSC_COMM_SELF,stag->entriesGhost,vec);CHKERRQ(ierr);
  ierr = VecSetBlockSize(*vec,stag->entriesPerElement);CHKERRQ(ierr);
  ierr = VecSetDM(*vec,dm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode DMCreateMatrix_Stag(DM dm,Mat *mat)
{
  PetscErrorCode         ierr;
  const DM_Stag * const  stag = (DM_Stag*)dm->data;
  MatType                matType;
  PetscBool              isaij,isshell;
  PetscInt               width,nNeighbors,dim;
  ISLocalToGlobalMapping ltogmap;

  PetscFunctionBegin;
  ierr = DMGetDimension(dm,&dim);CHKERRQ(ierr);
  ierr = DMGetMatType(dm,&matType);CHKERRQ(ierr);
  ierr = PetscStrcmp(matType,MATAIJ,&isaij);CHKERRQ(ierr);
  ierr = PetscStrcmp(matType,MATSHELL,&isshell);CHKERRQ(ierr);

  if (isaij) {
    /* This implementation gives a very dense stencil, which is likely unsuitable for
       real applications. */
    switch (stag->stencilType) {
      case DMSTAG_STENCIL_NONE:
        nNeighbors = 1;
        break;
      case DMSTAG_STENCIL_STAR:
        switch (dim) {
          case 1 :
            nNeighbors = 2*stag->stencilWidth + 1;
            break;
          case 2 :
            nNeighbors = 4*stag->stencilWidth + 3;
            break;
          case 3 :
            nNeighbors = 6*stag->stencilWidth + 5;
            break;
          default : SETERRQ1(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_OUTOFRANGE,"Unsupported dimension %d",dim);
        }
        break;
      case DMSTAG_STENCIL_BOX:
        switch (dim) {
          case 1 :
            nNeighbors = (2*stag->stencilWidth + 1);
            break;
          case 2 :
            nNeighbors = (2*stag->stencilWidth + 1) * (2*stag->stencilWidth + 1);
            break;
          case 3 :
            nNeighbors = (2*stag->stencilWidth + 1) * (2*stag->stencilWidth + 1) * (2*stag->stencilWidth + 1);
            break;
          default : SETERRQ1(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_OUTOFRANGE,"Unsupported dimension %d",dim);
        }
        break;
      default : SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_OUTOFRANGE,"Unsupported stencil");
    }
    width = (stag->dof[0] + stag->dof[1] + stag->dof[2] + stag->dof[3]) * nNeighbors;
    ierr = MatCreateAIJ(PETSC_COMM_WORLD,stag->entries,stag->entries,PETSC_DETERMINE,PETSC_DETERMINE,width,NULL,width,NULL,mat);CHKERRQ(ierr);
  } else if (isshell) {
    ierr = MatCreate(PetscObjectComm((PetscObject)dm),mat);CHKERRQ(ierr);
    ierr = MatSetSizes(*mat,stag->entries,stag->entries,PETSC_DETERMINE,PETSC_DETERMINE);CHKERRQ(ierr);
    ierr = MatSetType(*mat,MATSHELL);CHKERRQ(ierr);
    ierr = MatSetUp(*mat);CHKERRQ(ierr);
  } else SETERRQ1(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Not implemented for Mattype %s",matType);

  ierr = DMGetLocalToGlobalMapping(dm,&ltogmap);CHKERRQ(ierr);
  ierr = MatSetLocalToGlobalMapping(*mat,ltogmap,ltogmap);CHKERRQ(ierr);
  ierr = MatSetDM(*mat,dm);CHKERRQ(ierr);
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
  ierr = MPI_Comm_compare(comm,PetscObjectComm((PetscObject)dm2),&sameComm);CHKERRQ(ierr);
  if (sameComm != MPI_IDENT) {
    ierr = PetscInfo2((PetscObject)dm,"DMStag objects have different communicators: %d != %d\n",comm,PetscObjectComm((PetscObject)dm2));CHKERRQ(ierr);
    *set = PETSC_FALSE;
    PetscFunctionReturn(0);
  }
  ierr = DMGetDimension(dm ,&dim );CHKERRQ(ierr);
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
  ierr = MPI_Comm_rank(PetscObjectComm((PetscObject)dm),&rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(PetscObjectComm((PetscObject)dm),&size);CHKERRQ(ierr);
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
    ierr = PetscViewerASCIIPrintf(viewer,"Elementwise ghost stencil: %s, width %D\n",DMStagStencilTypes[stag->stencilType],stag->stencilWidth);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"Stratum dof:");CHKERRQ(ierr);
    for (i=0; i<dim+1; ++i) {
      ierr = PetscViewerASCIIPrintf(viewer," %D:%D",i,stag->dof[i]);CHKERRQ(ierr);
    }
    ierr = PetscViewerASCIIPrintf(viewer,"\n");CHKERRQ(ierr);
    if(dm->coordinateDM) {
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
      ierr = PetscViewerASCIISynchronizedPrintf(viewer,"[%d] Local native entries: %d\n",rank,stag->entries     );CHKERRQ(ierr);
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
  ierr = PetscOptionsInt("-stag_dof_0","Number of dof per 0-cell (vertex/corner)","DMStagSetDOF",stag->dof[0],&stag->dof[0],NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-stag_dof_1","Number of dof per 1-cell (edge)",         "DMStagSetDOF",stag->dof[1],&stag->dof[1],NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-stag_dof_2","Number of dof per 2-cell (face)",         "DMStagSetDOF",stag->dof[2],&stag->dof[2],NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-stag_dof_3","Number of dof per 3-cell (hexahedron)",   "DMStagSetDOF",stag->dof[3],&stag->dof[3],NULL);CHKERRQ(ierr);
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*MC
  DMSTAG = "stag" - A DM object representing a "staggered grid" or a structured cell complex.

  This implementation parallels the DMDA implementation in many ways, but allows degrees of freedom
  to be associated with all "strata" in a logically-rectangular grid: vertices, edges, faces, and elements.

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

  ierr = PetscMemzero(dm->ops,sizeof(*(dm->ops)));CHKERRQ(ierr);
  dm->ops->createcoordinatedm  = DMCreateCoordinateDM_Stag;
  dm->ops->createglobalvector  = DMCreateGlobalVector_Stag;
  dm->ops->createinterpolation = NULL;
  dm->ops->createlocalvector   = DMCreateLocalVector_Stag;
  dm->ops->creatematrix        = DMCreateMatrix_Stag;
  dm->ops->destroy             = DMDestroy_Stag;
  dm->ops->getneighbors        = DMGetNeighbors_Stag;
  dm->ops->globaltolocalbegin  = DMGlobalToLocalBegin_Stag;
  dm->ops->globaltolocalend    = DMGlobalToLocalEnd_Stag;
  dm->ops->localtoglobalbegin  = DMLocalToGlobalBegin_Stag;
  dm->ops->localtoglobalend    = DMLocalToGlobalEnd_Stag;
  dm->ops->setfromoptions      = DMSetFromOptions_Stag;
  switch (dim) {
    case 1: dm->ops->setup     = DMSetUp_Stag_1d; break;
    case 2: dm->ops->setup     = DMSetUp_Stag_2d; break;
    case 3: dm->ops->setup     = DMSetUp_Stag_3d; break;
    default : SETERRQ1(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_OUTOFRANGE,"Unsupported dimension %d",dim);
  }
  dm->ops->clone               = DMClone_Stag;
  dm->ops->view                = DMView_Stag;
  dm->ops->getcompatibility    = DMGetCompatibility_Stag;
  PetscFunctionReturn(0);
}
