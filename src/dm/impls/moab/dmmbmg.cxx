#include <petsc-private/dmmbimpl.h> /*I  "petscdmmoab.h"   I*/

#include <petscdmmoab.h>
#include <MBTagConventions.hpp>
#include <moab/NestedRefine.hpp>

#undef __FUNCT__
#define __FUNCT__ "DMMoabGenerateHierarchy"
/*@
  DMMoabGenerateHierarchy - Generate a multi-level uniform refinement hierarchy
  by succesively refining a coarse mesh, already defined in the DM object
  provided by the user.

  Collective on MPI_Comm

  Input Parameter:
+ dmb  - The DMMoab object

  Output Parameter:
+ nlevels   - The number of levels of refinement needed to generate the hierarchy
. ldegrees  - The degree of refinement at each level in the hierarchy

  Level: beginner

.keywords: DMMoab, create, refinement
@*/
PetscErrorCode DMMoabGenerateHierarchy(DM dm,PetscInt nlevels,PetscInt *ldegrees)
{
  DM_Moab        *dmmoab;
  PetscErrorCode  ierr;
  moab::ErrorCode merr;
  PetscInt *pdegrees,i;
  std::vector<moab::EntityHandle> hsets;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  dmmoab = (DM_Moab*)(dm)->data;

  if (!ldegrees) {
    ierr = PetscMalloc1(nlevels,&pdegrees);CHKERRQ(ierr);
    for (i=0; i<nlevels; i++) pdegrees[i]=2; /* default = Degree 2 refinement */
  }
  else pdegrees=ldegrees;

  /* initialize set level refinement data for hierarchy */
  dmmoab->nhlevels=nlevels;

  /* Instantiate the nested refinement class */
  dmmoab->hierarchy = new moab::NestedRefine(dynamic_cast<moab::Core*>(dmmoab->mbiface), dmmoab->pcomm, dmmoab->fileset);

  ierr = PetscMalloc1(nlevels+1,&dmmoab->hsets);CHKERRQ(ierr);
  hsets.resize(nlevels+1);

  /* generate the mesh hierarchy */
  merr = dmmoab->hierarchy->generate_mesh_hierarchy(nlevels, pdegrees, hsets);MBERRNM(merr);

  merr = dmmoab->hierarchy->exchange_ghosts(hsets, dmmoab->nghostrings);MBERRNM(merr);

  /* copy the mesh sets for nested refinement hierarchy */
  for (i=0; i<=nlevels; i++)
      dmmoab->hsets[i]=hsets[i];

  if (dmmoab->nghostrings && false) {
    PetscInfo2(NULL, "Exchanging ghost cells (dim %d) with %d rings\n",dmmoab->dim,dmmoab->nghostrings);
    // for (i=1; i<=nlevels; i++) {
    //   /* resolve the shared entities by exchanging information to adjacent processors */
    //   merr = dmmoab->pcomm->exchange_ghost_cells(dmmoab->dim,0,dmmoab->nghostrings,dmmoab->dim,true,false,&hsets[i]);MBERRV(dmmoab->mbiface,merr);
    // }
    merr = dmmoab->pcomm->exchange_ghost_cells(dmmoab->dim,0,dmmoab->nghostrings,dmmoab->dim,true,false);MBERRV(dmmoab->mbiface,merr);

    // moab::Range vtxall, elmsall;
    // merr = dmmoab->mbiface->get_entities_by_dimension(0, 0, vtxall, true);MBERRNM(merr);
    // merr = dmmoab->mbiface->get_entities_by_dimension(0, dmmoab->dim, elmsall, true);MBERRNM(merr);
    
  }

  hsets.clear();
  if (!ldegrees) {
    ierr = PetscFree(pdegrees);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMRefineHierarchy_Moab"
/*@
  DMRefineHierarchy_Moab - Generate a multi-level DM hierarchy
  by succesively refining a coarse mesh.

  Collective on MPI_Comm

  Input Parameter:
+ dm  - The DMMoab object

  Output Parameter:
+ nlevels   - The number of levels of refinement needed to generate the hierarchy
. dmf  - The DM objects after successive refinement of the hierarchy

  Level: beginner

.keywords: DMMoab, generate, refinement
@*/
PetscErrorCode  DMRefineHierarchy_Moab(DM dm,PetscInt nlevels,DM dmf[])
{
  PetscErrorCode  ierr;
  PetscInt        i;

  PetscFunctionBegin;

  ierr = DMRefine(dm,PetscObjectComm((PetscObject)dm),&dmf[0]);CHKERRQ(ierr);
  for (i=1; i<nlevels; i++) {
    ierr = DMRefine(dmf[i-1],PetscObjectComm((PetscObject)dm),&dmf[i]);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMCoarsenHierarchy_Moab"
/*@
  DMCoarsenHierarchy_Moab - Generate a multi-level DM hierarchy
  by succesively coarsening a refined mesh.

  Collective on MPI_Comm

  Input Parameter:
+ dm  - The DMMoab object

  Output Parameter:
+ nlevels   - The number of levels of refinement needed to generate the hierarchy
. dmc  - The DM objects after successive coarsening of the hierarchy

  Level: beginner

.keywords: DMMoab, generate, coarsening
@*/
PetscErrorCode DMCoarsenHierarchy_Moab(DM dm,PetscInt nlevels,DM dmc[])
{
  PetscErrorCode  ierr;
  PetscInt        i;

  PetscFunctionBegin;

  ierr = DMCoarsen(dm,PetscObjectComm((PetscObject)dm),&dmc[0]);CHKERRQ(ierr);
  for (i=1; i<nlevels; i++) {
    ierr = DMCoarsen(dmc[i-1],PetscObjectComm((PetscObject)dm),&dmc[i]);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "DMCreateInterpolation_Moab"
/*@
  DMCreateInterpolation_Moab - Generate the interpolation operators to transform
  operators (matrices, vectors) from parent level to child level as defined by
  the DM inputs provided by the user.

  Collective on MPI_Comm

  Input Parameter:
+ dm1  - The DMMoab object
- dm2  - the second, finer DMMoab object

  Output Parameter:
+ interpl  - The interpolation operator for transferring data between the levels
- vec      - The scaling vector (optional)

  Level: developer

.keywords: DMMoab, create, refinement
@*/
PetscErrorCode DMCreateInterpolation_Moab(DM dm1,DM dm2,Mat* interpl,Vec* vec)
{
  DM_Moab         *dmb1, *dmb2;
  PetscErrorCode   ierr;
  moab::ErrorCode  merr;
  PetscInt         dim;
  PetscReal        factor;
  PetscBool        eonbnd;
  PetscInt         innz, *nnz, ionz, *onz;
  PetscInt         nlsiz1, nlsiz2, nlghsiz1, nlghsiz2, ngsiz1, ngsiz2;
  std::vector<int> bndrows;
  std::vector<PetscBool> dbdry;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm1,DM_CLASSID,1);
  PetscValidHeaderSpecific(dm2,DM_CLASSID,2);
  dmb1 = (DM_Moab*)(dm1)->data;
  dmb2 = (DM_Moab*)(dm2)->data;
  nlsiz1 = dmb1->nloc*dmb1->numFields;
  nlsiz2 = dmb2->nloc*dmb2->numFields;
  ngsiz1 = dmb1->n*dmb1->numFields;
  ngsiz2 = dmb2->n*dmb2->numFields;
  nlghsiz1 = (dmb1->nloc+dmb1->nghost)*dmb1->numFields;
  nlghsiz2 = (dmb2->nloc+dmb2->nghost)*dmb2->numFields;

  int rank = dmb1->pcomm->rank();

  PetscInfo4(dm1,"Creating interpolation matrix %D X %D to apply transformation between levels %D -> %D.\n",ngsiz2,ngsiz1,dmb1->hlevel,dmb2->hlevel);

  PetscPrintf(PETSC_COMM_SELF, "[%d] Local matrix: %D X %D\n", rank, nlsiz2, nlsiz1);

  /* allocate the nnz, onz arrays based on block size and local nodes */
  ierr = PetscCalloc2(nlghsiz2,&nnz,nlghsiz2,&onz);CHKERRQ(ierr);

  PetscPrintf(PETSC_COMM_SELF, "[%d] Coarse Local elements: %D, vertices: %D\n", rank, dmb1->elocal->size(), dmb1->vlocal->size());
  PetscPrintf(PETSC_COMM_SELF, "[%d] Fine   Local elements: %D, vertices: %D\n", rank, dmb2->elocal->size(), dmb2->vlocal->size());
  PetscPrintf(PETSC_COMM_SELF, "[%d] Sequence Start: %D, End: %D\n", rank, dmb2->seqstart, dmb2->seqend);
  
  /* Loop through the local elements and compute the relation between the current parent and the refined_level. */
  for(moab::Range::iterator iter = dmb1->elocal->begin(); iter!= dmb1->elocal->end(); iter++) {

    const moab::EntityHandle ehandle = *iter;
    std::vector<moab::EntityHandle> children;
    std::vector<moab::EntityHandle> connp, connc;

    /* Get the relation between the current (coarse) parent and its corresponding (finer) children elements */
    merr = dmb1->hierarchy->parent_to_child(ehandle, dmb1->hlevel, dmb2->hlevel, children);MBERRNM(merr);

    /* Get connectivity and coordinates of the parent vertices */
    merr = dmb1->hierarchy->get_connectivity(ehandle, dmb1->hlevel, connp);MBERRNM(merr);
    for (unsigned ic=0; ic < children.size(); ic++) {
      std::vector<moab::EntityHandle> tconnc;
      /* Get handles of the parent vertices in canonical order and intersect */
      merr = dmb2->hierarchy->get_connectivity(children[ic], dmb2->hlevel, tconnc);MBERRNM(merr);
      for (unsigned tc=0; tc<tconnc.size(); tc++) {
        if (std::find(connc.begin(), connc.end(), tconnc[tc]) == connc.end())
          connc.push_back(tconnc[tc]);
      }
    }
    //PetscPrintf(PETSC_COMM_SELF, "[%d] EntityHandle %d, children = %d, total intersection = %d\n", rank, ehandle, children.size(), connc.size());

    std::vector<int> dofsp(connp.size()), dofsc(connc.size());
    /* TODO: specific to scalar system - use GetDofs */
    //ierr = DMMoabGetFieldDofs(dm1, connp.size(), &connp[0], 0, &dofsp[0]);CHKERRQ(ierr);
    ierr = DMMoabGetFieldDofsLocal(dm2, connc.size(), &connc[0], 0, &dofsc[0]);CHKERRQ(ierr);

    //PetscPrintf(PETSC_COMM_SELF, "[%d] EntityHandle %d, dofs [4] = %d, %d, %d, %d\n", rank, ehandle, dofsc[0], dofsc[1], dofsc[2], dofsc[3]);
    // if (rank == 1) {
    //   for (unsigned tp=0;tp<connc.size(); tp++)
    //     PetscPrintf(PETSC_COMM_SELF, "[%d] EntityHandle %d \t %d  -- dofs [%d] = %d, %d\n", rank, ehandle, connc[tp]-dmb2->seqstart, tp, dmb2->gidmap[(PetscInt)connc[tp]-dmb2->seqstart], dofsc[tp]);
    // }
    for (unsigned tp=0;tp<connp.size(); tp++) {
      // ierr = MatSetValues(*interpl, connc.size(), &dofsc[0], 1, &dofsp[i], &values_phi[0], ADD_VALUES);CHKERRQ(ierr);
      if (dmb1->vowned->find(connp[tp]) != dmb1->vowned->end()) {
        for (unsigned tc=0;tc<connc.size(); tc++) {
          nnz[dofsc[tc]]++;
          //PetscPrintf(PETSC_COMM_SELF, "[%d] Found nnz coupling for %D = %d\n", rank, connp[tp], dofsc[tc]);
        }
      }
      else if (dmb1->vghost->find(connp[tp]) != dmb1->vghost->end()) {
        for (unsigned tc=0;tc<connc.size(); tc++) {
          onz[dofsc[tc]]++;
          //PetscPrintf(PETSC_COMM_SELF, "[%d] Found onz coupling for %D = %d\n", rank, connp[tp], dofsc[tc]);
        }
      }
      else SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Invalid entity in parent level %D\n", connc[tp]);
    }
    for(unsigned tc = 0; tc < connc.size(); tc++) {
      if (dmb2->vowned->find(connc[tc]) != dmb2->vowned->end()) nnz[dofsc[tc]]++;
      else if (dmb2->vghost->find(connc[tc]) != dmb2->vghost->end()) onz[dofsc[tc]]++;
      else SETERRQ1(PETSC_COMM_WORLD,PETSC_ERR_ARG_OUTOFRANGE,"Invalid entity in child level %D\n", connc[tc]);
    }
  }

/*
  int i=0;
  std::vector<moab::EntityHandle> adjs;
  for(moab::Range::iterator iter = dmb1->vowned->begin(); iter!= dmb1->vowned->end(); iter++, i++) {
    merr = dmb1->hierarchy->get_adjacencies(*iter, 0, adjs);MBERRNM(merr);
    nnz[i] -= adjs.size();
    adjs.clear();
  }
  i=0;
  for(moab::Range::iterator iter = dmb1->vghost->begin(); iter!= dmb1->vghost->end(); iter++, i++) {
    //merr = dmb1->hierarchy->get_adjacencies(&(*iter), 1, 0, false, adjs, moab::Interface::UNION);MBERRNM(merr);
    merr = dmb1->hierarchy->get_adjacencies(*iter, 0, adjs);MBERRNM(merr);
    onz[i] -= adjs.size();
    adjs.clear();
  }
*/

  PetscInt* ldofs = dmb2->lidmap;
  PetscInt* gdofs = dmb2->gidmap;
  ionz=onz[0];
  innz=nnz[0];
  for (int tc=0; tc < nlsiz2; tc++) {
    // check for maximum allowed sparsity = fully dense
    nnz[tc] = std::min(nlsiz1,nnz[tc]);
    onz[tc] = std::min(nlsiz1,onz[tc]);

    innz = (innz < nnz[tc] ? nnz[tc] : innz);
    ionz = (ionz < onz[tc] ? onz[tc] : ionz);
    PetscPrintf(PETSC_COMM_SELF, "[%D]: %D NNZ = %D, ONZ = %D\n", rank, gdofs[ldofs[tc]], nnz[tc], onz[tc]);
  }
  PetscPrintf(PETSC_COMM_SELF, "[%D]: Final: INNZ = %D, IONZ = %D\n", rank, innz, ionz);

  MPI_Barrier(PETSC_COMM_WORLD);

  /* create interpolation matrix */
  PetscPrintf(PETSC_COMM_SELF, "[%D]: Creating matrix\n", rank);
  ierr = MatCreate(PetscObjectComm((PetscObject)dm2), interpl);CHKERRQ(ierr);
  PetscPrintf(PETSC_COMM_SELF, "[%D]: set sizes\n", rank);
  ierr = MatSetSizes(*interpl, nlsiz2, ngsiz1, ngsiz2, ngsiz1);CHKERRQ(ierr);
  //ierr = MatSetType(*interpl, dm1->mattype);CHKERRQ(ierr);
  PetscPrintf(PETSC_COMM_SELF, "[%D]: set type\n", rank);
  ierr = MatSetType(*interpl,MATAIJ);CHKERRQ(ierr);
  PetscPrintf(PETSC_COMM_SELF, "[%D]: set from opts\n", rank);
  ierr = MatSetFromOptions(*interpl);CHKERRQ(ierr);

  PetscPrintf(PETSC_COMM_SELF, "[%D]: Setting prealloc\n", rank);
  ierr = MatSeqAIJSetPreallocation(*interpl,innz,nnz);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(*interpl,innz,nnz,ionz,onz);CHKERRQ(ierr);
  //ierr = MatMPIAIJSetPreallocation(*interpl,innz,0,ionz,0);CHKERRQ(ierr);

  PetscPrintf(PETSC_COMM_SELF, "[%D]: Cleaning arrays\n", rank);
  /* clean up temporary memory */
  ierr = PetscFree2(nnz,onz);CHKERRQ(ierr);

  /* set up internal matrix data-structures */
  ierr = MatSetUp(*interpl);CHKERRQ(ierr);
  //ierr = MatZeroEntries(*interpl);CHKERRQ(ierr);

  ierr = DMGetDimension(dm1, &dim);CHKERRQ(ierr);

  factor = std::pow(2.0 /*degree_P_for_refinement*/,(dmb2->hlevel-dmb1->hlevel)*dmb1->dim*1.0);

  ierr = MatSetOption(*interpl, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE);CHKERRQ(ierr);

  /* Loop through the remaining vertices. These vertices appear only on the current refined_level. */
  for(moab::Range::iterator iter = dmb1->elocal->begin(); iter!= dmb1->elocal->end(); iter++) {

    const moab::EntityHandle ehandle = *iter;
    std::vector<moab::EntityHandle> children;
    std::vector<moab::EntityHandle> connp, connc;

    /* Get the relation between the current (coarse) parent and its corresponding (finer) children elements */
    merr = dmb1->hierarchy->parent_to_child(ehandle, dmb1->hlevel, dmb2->hlevel, children);MBERRNM(merr);

    /* Get connectivity and coordinates of the parent vertices */
    merr = dmb1->hierarchy->get_connectivity(ehandle, dmb1->hlevel, connp);MBERRNM(merr);
    for (unsigned ic=0; ic < children.size(); ic++) {
      std::vector<moab::EntityHandle> tconnc;
      /* Get coordinates of the parent vertices in canonical order */
      merr = dmb1->hierarchy->get_connectivity(children[ic], dmb2->hlevel, tconnc);MBERRNM(merr);
      for (unsigned tc=0; tc<tconnc.size(); tc++) {
        connc.push_back(tconnc[tc]);
      }
    }

    std::vector<double> pcoords(connp.size()*3), ccoords(connc.size()*3), values_phi(connc.size());
    /* Get coordinates for connectivity entities in canonical order for both coarse and finer levels */
    merr = dmb1->hierarchy->get_coordinates(&connp[0], connp.size(), dmb1->hlevel, &pcoords[0]);MBERRNM(merr);
    merr = dmb2->hierarchy->get_coordinates(&connc[0], connc.size(), dmb2->hlevel, &ccoords[0]);MBERRNM(merr);

    std::vector<int> dofsp(connp.size()), dofsc(connc.size());
    /* TODO: specific to scalar system - use GetDofs */
    ierr = DMMoabGetFieldDofsLocal(dm1, connp.size(), &connp[0], 0, &dofsp[0]);CHKERRQ(ierr);
    ierr = DMMoabGetFieldDofsLocal(dm2, connc.size(), &connc[0], 0, &dofsc[0]);CHKERRQ(ierr);

    /* Compute the interpolation weights by determining distance of 1-ring 
       neighbor vertices from current vertex */
    for (unsigned i=0;i<connp.size(); i++) {
      double normsum=0.0;
      for (unsigned j=0;j<connc.size(); j++) {
        values_phi[j] = 0.0;
        for (unsigned k=0;k<3; k++)
          values_phi[j] += std::pow(pcoords[i*3+k]-ccoords[k+j*3], dim);
        if (values_phi[j] < 1e-12) {
          values_phi[j] = 1e12;
        }
        else {
          //values_phi[j] = std::pow(values_phi[j], -1.0/dim);
          values_phi[j] = std::pow(values_phi[j], -1.0);
          normsum += values_phi[j];
        }
      }
      for (unsigned j=0;j<connc.size(); j++) {
        if (values_phi[j] > 1e11)
          values_phi[j] = factor*0.5/connc.size();
        else
          values_phi[j] = factor*values_phi[j]*0.5/(connc.size()*normsum);
      }
      ierr = MatSetValues(*interpl, connc.size(), &dofsc[0], 1, &dofsp[i], &values_phi[0], ADD_VALUES);CHKERRQ(ierr);
    }

    /* check if element is on the boundary */
    //ierr = DMMoabIsEntityOnBoundary(dm1,ehandle,&eonbnd);CHKERRQ(ierr);
    dbdry.resize(connc.size());
    ierr = DMMoabCheckBoundaryVertices(dm2,connc.size(),&connc[0],dbdry.data());CHKERRQ(ierr);
    eonbnd=PETSC_FALSE;
    for (unsigned i=0; i< connc.size(); ++i)
      if (dbdry[i]) eonbnd=PETSC_TRUE;

    values_phi.clear();
    values_phi.resize(connp.size());
    /* apply dirichlet boundary conditions */
    if (eonbnd) {

      ierr = MatAssemblyBegin(*interpl,MAT_FLUSH_ASSEMBLY);CHKERRQ(ierr);
      ierr = MatAssemblyEnd(*interpl,MAT_FLUSH_ASSEMBLY);CHKERRQ(ierr);
      /* get the list of nodes on boundary so that we can enforce dirichlet conditions strongly */
      //ierr = DMMoabCheckBoundaryVertices(dm2,connc.size(),&connc[0],dbdry);CHKERRQ(ierr);
      for (unsigned i=0; i < connc.size(); i++) {
        if (dbdry[i]) {  /* dirichlet node */
          /* think about strongly imposing dirichlet */
          //bndrows.push_back(dofsc[i]);

          ierr = MatSetValues(*interpl, 1, &dofsc[i], connp.size(), &dofsp[0], &values_phi[0], INSERT_VALUES);CHKERRQ(ierr);
          //values_phi[0]=1.0;
          //ierr = MatSetValues(*interpl, 1, &dofsc[i], 1, &dofsc[i], &values_phi[0], INSERT_VALUES);CHKERRQ(ierr);
        }
      }

      ierr = MatAssemblyBegin(*interpl,MAT_FLUSH_ASSEMBLY);CHKERRQ(ierr);
      ierr = MatAssemblyEnd(*interpl,MAT_FLUSH_ASSEMBLY);CHKERRQ(ierr);
    }

    //get interpolation weights
    //ierr = Compute_Quad4_Basis(pcoords, 1, coord, values_phi);CHKERRQ(ierr);
    // for (int j=0;j<dofs_per_element; j++)
    //  std::cout<<"values "<<values_phi[j]<<std::endl;

    //get row and column indices, zero weights are ignored
    /*
    int nz_ind = 0;
    idx = dmb2->vowned->index(vhandle);
    for (int j=0;j<dofs_per_element; j++){
      idy[nz_ind] = dmb1->vowned->index(connectivity[j]);
      PetscPrintf(PETSC_COMM_WORLD, "Finding coarse connectivity vertex %D associated with [%D, %D] - set to %D\n", connectivity[j], parent.size(), vhandle, idy[nz_ind]);
      //values_phi[nz_ind] = values_phi[j];
      nz_ind = nz_ind+1;
    }
    */

    //ierr = MatSetValues(*interpl, nz_ind, idy, 1, &idx, values_phi, INSERT_VALUES);CHKERRQ(ierr);
    //ierr = MatSetValues(*interpl, connp.size(), dofsp, connc.size(), dofsc, &values_phi[0], INSERT_VALUES);CHKERRQ(ierr);
  }

  //PetscPrintf(PETSC_COMM_WORLD, "[Boundary vertices = %D] :: A few: %D %D %D %D \n", bndrows.size(), bndrows[0], bndrows[1], bndrows[2], bndrows[3]);
  //ierr = MatZeroRows(*interpl, bndrows.size(), &bndrows[0], 1.0, 0, 0);CHKERRQ(ierr);

  ierr = MatAssemblyBegin(*interpl,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*interpl,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMCreateInjection_Moab"
/*@
  DMCreateInjection_Moab - Generate a multi-level uniform refinement hierarchy
  by succesively refining a coarse mesh, already defined in the DM object
  provided by the user.

  Collective on MPI_Comm

  Input Parameter:
. dmb  - The DMMoab object

  Output Parameter:
. nlevels   - The number of levels of refinement needed to generate the hierarchy
+ ldegrees  - The degree of refinement at each level in the hierarchy

  Level: beginner

.keywords: DMMoab, create, refinement
@*/
PetscErrorCode DMCreateInjection_Moab(DM dm1,DM dm2,VecScatter* ctx)
{
  //DM_Moab        *dmmoab;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm1,DM_CLASSID,1);
  PetscValidHeaderSpecific(dm2,DM_CLASSID,2);
  //dmmoab = (DM_Moab*)(dm1)->data;

  PetscPrintf(PETSC_COMM_WORLD, "[DMCreateInjection_Moab] :: Placeholder\n");
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "DM_UMR_Moab_Private"
PetscErrorCode  DM_UMR_Moab_Private(DM dm,MPI_Comm comm,PetscBool refine,DM *dmref)
{
  PetscErrorCode  ierr;
  PetscInt        i,dim;
  DM              dm2;
  moab::ErrorCode merr;
  DM_Moab        *dmb = (DM_Moab*)dm->data,*dd2;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscValidPointer(dmref,4);

  if ( (dmb->hlevel == dmb->nhlevels && refine) || (dmb->hlevel == 0 && !refine) ) {
    if (dmb->hlevel+1 > dmb->nhlevels && refine) PetscInfo2(NULL,"Invalid multigrid refinement hierarchy level specified (%D). MOAB UMR max levels = %D. Creating a NULL object.\n",dmb->hlevel+1,dmb->nhlevels);
    if (dmb->hlevel-1 < 0 && !refine) PetscInfo1(NULL,"Invalid multigrid coarsen hierarchy level specified (%D). Creating a NULL object.\n",dmb->hlevel-1);
    *dmref = PETSC_NULL;
    PetscFunctionReturn(0);
  }

  ierr = DMMoabCreate(PetscObjectComm((PetscObject)dm), &dm2);CHKERRQ(ierr);
  dd2 = (DM_Moab*)dm2->data;

  dd2->mbiface = dmb->mbiface;
  dd2->pcomm = dmb->pcomm;
  dd2->icreatedinstance = PETSC_FALSE;
  dd2->nghostrings=dmb->nghostrings;

  /* set the new level based on refinement/coarsening */
  if (refine) {
    dd2->hlevel=dmb->hlevel+1;
  }
  else {
    dd2->hlevel=dmb->hlevel-1;
  }

  /* Copy the multilevel hierarchy pointers in MOAB */
  dd2->hierarchy = dmb->hierarchy;
  dd2->nhlevels = dmb->nhlevels;
  ierr = PetscMalloc1(dd2->nhlevels+1,&dd2->hsets);CHKERRQ(ierr);
  for (i=0; i<=dd2->nhlevels; i++) {
    dd2->hsets[i]=dmb->hsets[i];
  }
  dd2->fileset = dd2->hsets[dd2->hlevel];

  /* do the remaining initializations for DMMoab */
  dd2->bs = dmb->bs;
  dd2->numFields = dmb->numFields;
  dd2->rw_dbglevel = dmb->rw_dbglevel;
  dd2->partition_by_rank = dmb->partition_by_rank;
  ierr = PetscStrcpy(dd2->extra_read_options, dmb->extra_read_options);CHKERRQ(ierr);
  ierr = PetscStrcpy(dd2->extra_write_options, dmb->extra_write_options);CHKERRQ(ierr);
  dd2->read_mode = dmb->read_mode;
  dd2->write_mode = dmb->write_mode;

  /* set global ID tag handle */
  ierr = DMMoabSetLocalToGlobalTag(dm2, dmb->ltog_tag);CHKERRQ(ierr);

  merr = dd2->mbiface->tag_get_handle(MATERIAL_SET_TAG_NAME, dd2->material_tag);MBERRNM(merr);

  ierr = DMSetOptionsPrefix(dm2,((PetscObject)dm)->prefix);CHKERRQ(ierr);
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMSetDimension(dm2,dim);CHKERRQ(ierr);

  /* allow overloaded (user replaced) operations to be inherited by refinement clones */
  dm2->ops->creatematrix = dm->ops->creatematrix;

  /* copy fill information if given */
  ierr = DMMoabSetBlockFills(dm2, dmb->dfill, dmb->ofill);CHKERRQ(ierr);

  /* copy vector type information */
  ierr = DMSetMatType(dm2,dm->mattype);CHKERRQ(ierr);
  ierr = DMSetVecType(dm2,dm->vectype);CHKERRQ(ierr);
  dd2->numFields = dmb->numFields;
  if (dmb->numFields) {
    ierr = DMMoabSetFieldNames(dm2,dmb->numFields,dmb->fieldNames);CHKERRQ(ierr);
  }

  ierr = DMSetFromOptions(dm2);CHKERRQ(ierr);

  /* recreate Dof numbering for the refined DM and make sure the distribution is correctly populated */
  ierr = DMSetUp(dm2);CHKERRQ(ierr);

  *dmref = dm2;
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "DMRefine_Moab"
/*@
  DMRefine_Moab - Generate a multi-level uniform refinement hierarchy
  by succesively refining a coarse mesh, already defined in the DM object
  provided by the user.

  Collective on DM

  Input Parameter:
+ dm  - The DMMoab object
- comm - the communicator to contain the new DM object (or MPI_COMM_NULL)

  Output Parameter:
. dmf - the refined DM, or NULL

  Note: If no refinement was done, the return value is NULL

  Level: developer

.keywords: DMMoab, create, refinement
@*/
PetscErrorCode DMRefine_Moab(DM dm,MPI_Comm comm,DM* dmf)
{
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);

  ierr = DM_UMR_Moab_Private(dm,comm,PETSC_TRUE,dmf);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMCoarsen_Moab"
/*@
  DMCoarsen_Moab - Generate a multi-level uniform refinement hierarchy
  by succesively refining a coarse mesh, already defined in the DM object
  provided by the user.

  Collective on DM

  Input Parameter:
. dm  - The DMMoab object
- comm - the communicator to contain the new DM object (or MPI_COMM_NULL)

  Output Parameter:
. dmf - the coarsened DM, or NULL

  Note: If no coarsening was done, the return value is NULL

  Level: developer

.keywords: DMMoab, create, coarsening
@*/
PetscErrorCode DMCoarsen_Moab(DM dm,MPI_Comm comm,DM* dmc)
{
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);

  ierr = DM_UMR_Moab_Private(dm,comm,PETSC_FALSE,dmc);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
