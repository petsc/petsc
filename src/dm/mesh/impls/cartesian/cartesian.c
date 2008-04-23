#include "private/meshimpl.h"
#include <CartesianSieve.hh>

#undef __FUNCT__  
#define __FUNCT__ "MeshCartesianGetMesh"
/*@C
    MeshCartesianGetMesh - Gets the internal mesh object

    Not collective

    Input Parameter:
.    mesh - the mesh object

    Output Parameter:
.    m - the internal mesh object
 
    Level: advanced

.seealso MeshCreate(), MeshCartesianSetMesh()

@*/
PetscErrorCode PETSCDM_DLLEXPORT MeshCartesianGetMesh(Mesh mesh, ALE::Obj<ALE::CartesianMesh>& m)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mesh, MESH_COOKIE, 1);
  m = *((ALE::Obj<ALE::CartesianMesh> *) mesh->data);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MeshCartesianSetMesh"
/*@C
    MeshCartesianSetMesh - Sets the internal mesh object

    Not collective

    Input Parameters:
+    mesh - the mesh object
-    m - the internal mesh object
 
    Level: advanced

.seealso MeshCreate(), MeshCartesianGetMesh()

@*/
PetscErrorCode PETSCDM_DLLEXPORT MeshCartesianSetMesh(Mesh mesh, const ALE::Obj<ALE::CartesianMesh>& m)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mesh, MESH_COOKIE, 1);
  *((ALE::Obj<ALE::CartesianMesh> *) mesh->data) = m;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MeshDestroy_Cartesian"
PetscErrorCode PETSCDM_DLLEXPORT MeshDestroy_Cartesian(Mesh mesh)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (mesh->data) {
    *((ALE::Obj<ALE::CartesianMesh> *) mesh->data) = PETSC_NULL;
    ierr = PetscFree(mesh->data);CHKERRQ(ierr);
    mesh->data    = PETSC_NULL;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MeshView_Cartesian_Ascii"
PetscErrorCode MeshView_Cartesian_Ascii(const ALE::Obj<ALE::CartesianMesh>& mesh, PetscViewer viewer)
{
  PetscViewerFormat format;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = PetscViewerGetFormat(viewer, &format);CHKERRQ(ierr);
  if (format == PETSC_VIEWER_ASCII_VTK) {
#if 0
    ierr = VTKViewer::writeHeader(viewer);CHKERRQ(ierr);
    ierr = VTKViewer::writeVertices(mesh, viewer);CHKERRQ(ierr);
    ierr = VTKViewer::writeElements(mesh, viewer);CHKERRQ(ierr);
#endif
  } else {
    int dim = mesh->getDimension();

    ierr = PetscViewerASCIIPrintf(viewer, "Mesh in %d dimensions:\n", dim);CHKERRQ(ierr);
    // FIX: Need to globalize
    ierr = PetscViewerASCIIPrintf(viewer, "  %d vertices\n", mesh->getSieve()->getNumVertices());CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer, "  %d cells\n",    mesh->getSieve()->getNumCells());CHKERRQ(ierr);
  }
  ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PETSCDM_DLLEXPORT MeshView_Cartesian(Mesh mesh, PetscViewer viewer)
{
  PetscTruth     iascii, isbinary, isdraw;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscTypeCompare((PetscObject) viewer, PETSC_VIEWER_ASCII, &iascii);CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject) viewer, PETSC_VIEWER_BINARY, &isbinary);CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject) viewer, PETSC_VIEWER_DRAW, &isdraw);CHKERRQ(ierr);

  if (iascii){
    ierr = MeshView_Cartesian_Ascii(*((ALE::Obj<ALE::CartesianMesh> *) mesh->data), viewer);CHKERRQ(ierr);
  } else if (isbinary) {
    SETERRQ(PETSC_ERR_SUP, "Binary viewer not implemented for Cartesian Mesh");
  } else if (isdraw){ 
    SETERRQ(PETSC_ERR_SUP, "Draw viewer not implemented for Cartesian Mesh");
  } else {
    SETERRQ1(PETSC_ERR_SUP,"Viewer type %s not supported by this mesh object", ((PetscObject)viewer)->type_name);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MeshGetInterpolation_Cartesian"
PetscErrorCode MeshGetInterpolation_Cartesian(Mesh fineMesh, Mesh coarseMesh, Mat *interpolation, Vec *scaling)
{
  ALE::Obj<ALE::CartesianMesh> coarse;
  ALE::Obj<ALE::CartesianMesh> fine;
  Mat                          P;
  PetscErrorCode               ierr;

  PetscFunctionBegin;
  ierr = MeshCartesianGetMesh(fineMesh,   fine);CHKERRQ(ierr);
  ierr = MeshCartesianGetMesh(coarseMesh, coarse);CHKERRQ(ierr);
#if 0
  const ALE::Obj<ALE::Mesh::real_section_type>& coarseCoordinates = coarse->getRealSection("coordinates");
  const ALE::Obj<ALE::Mesh::real_section_type>& fineCoordinates   = fine->getRealSection("coordinates");
  const ALE::Obj<ALE::Mesh::label_sequence>&    vertices          = fine->depthStratum(0);
  const ALE::Obj<ALE::Mesh::real_section_type>& sCoarse           = coarse->getRealSection("default");
  const ALE::Obj<ALE::Mesh::real_section_type>& sFine             = fine->getRealSection("default");
  const ALE::Obj<ALE::Mesh::order_type>&        coarseOrder = coarse->getFactory()->getGlobalOrder(coarse, "default", sCoarse);
  const ALE::Obj<ALE::Mesh::order_type>&        fineOrder   = fine->getFactory()->getGlobalOrder(fine, "default", sFine);
  const int dim = coarse->getDimension();
  double *v0, *J, *invJ, detJ, *refCoords, *values;
#endif

  ierr = MatCreate(fine->comm(), &P);CHKERRQ(ierr);
#if 0
  ierr = MatSetSizes(P, sFine->size(), sCoarse->size(), PETSC_DETERMINE, PETSC_DETERMINE);CHKERRQ(ierr);
  ierr = MatSetFromOptions(P);CHKERRQ(ierr);
  ierr = PetscMalloc5(dim,double,&v0,dim*dim,double,&J,dim*dim,double,&invJ,dim,double,&refCoords,dim+1,double,&values);CHKERRQ(ierr);
  for(ALE::Mesh::label_sequence::iterator v_iter = vertices->begin(); v_iter != vertices->end(); ++v_iter) {
    const ALE::Mesh::real_section_type::value_type *coords     = fineCoordinates->restrictPoint(*v_iter);
    const ALE::Mesh::point_type                     coarseCell = coarse->locatePoint(coords);

    coarse->computeElementGeometry(coarseCoordinates, coarseCell, v0, J, invJ, detJ);
    for(int d = 0; d < dim; ++d) {
      refCoords[d] = 0.0;
      for(int e = 0; e < dim; ++e) {
        refCoords[d] += invJ[d*dim+e]*(coords[e] - v0[e]);
      }
      refCoords[d] -= 1.0;
    }
    values[0] = 1.0/3.0 - (refCoords[0] + refCoords[1])/3.0;
    values[1] = 0.5*(refCoords[0] + 1.0);
    values[2] = 0.5*(refCoords[1] + 1.0);
    ierr = updateOperatorGeneral(P, fine, sFine, fineOrder, *v_iter, sCoarse, coarseOrder, coarseCell, values, INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = PetscFree5(v0,J,invJ,refCoords,values);CHKERRQ(ierr);
#endif
  ierr = MatAssemblyBegin(P, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(P, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  *interpolation = P;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MeshRefine_Cartesian"
PetscErrorCode MeshRefine_Cartesian(Mesh mesh, MPI_Comm comm, Mesh *refinedMesh)
{
  ALE::Obj<ALE::CartesianMesh> oldMesh;
  PetscErrorCode               ierr;

  PetscFunctionBegin;
  ierr = MeshCartesianGetMesh(mesh, oldMesh);CHKERRQ(ierr);
  ierr = MeshCreate(comm, refinedMesh);CHKERRQ(ierr);
#if 0
  ALE::Obj<ALE::Mesh> newMesh = ALE::Generator::refineMesh(oldMesh, refinementLimit, false);
  ierr = MeshCartesianSetMesh(*refinedMesh, newMesh);CHKERRQ(ierr);
  const ALE::Obj<ALE::CartesianMesh::real_section_type>& s = newMesh->getRealSection("default");

  newMesh->setDiscretization(oldMesh->getDiscretization());
  newMesh->setBoundaryCondition(oldMesh->getBoundaryCondition());
  newMesh->setupField(s);
#else
  SETERRQ(PETSC_ERR_SUP, "Not yet implemented");
#endif
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MeshCoarsen_Cartesian"
PetscErrorCode MeshCoarsen_Cartesian(Mesh mesh, MPI_Comm comm, Mesh *coarseMesh)
{
  ALE::Obj<ALE::CartesianMesh> oldMesh;
  PetscErrorCode               ierr;

  PetscFunctionBegin;
  ierr = MeshCartesianGetMesh(mesh, oldMesh);CHKERRQ(ierr);
  ierr = MeshCreate(comm, coarseMesh);CHKERRQ(ierr);
#if 0
  ALE::Obj<ALE::Mesh> newMesh = ALE::Generator::refineMesh(oldMesh, refinementLimit, false);
  ierr = MeshCartesianSetMesh(*coarseMesh, newMesh);CHKERRQ(ierr);
  const ALE::Obj<ALE::CartesianMesh::real_section_type>& s = newMesh->getRealSection("default");

  newMesh->setDiscretization(oldMesh->getDiscretization());
  newMesh->setBoundaryCondition(oldMesh->getBoundaryCondition());
  newMesh->setupField(s);
#else
  SETERRQ(PETSC_ERR_SUP, "Not yet implemented");
#endif
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MeshGetSectionReal_Cartesian"
PetscErrorCode MeshGetSectionReal_Cartesian(Mesh mesh, const char name[], SectionReal *section)
{
  ALE::Obj<ALE::CartesianMesh> m;
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  ierr = MeshCartesianGetMesh(mesh, m);CHKERRQ(ierr);
  ierr = SectionRealCreate(m->comm(), section);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) *section, name);CHKERRQ(ierr);
#if 0
  ierr = SectionRealSetSection(*section, m->getRealSection(std::string(name)));CHKERRQ(ierr);
  ierr = SectionRealSetBundle(*section, m);CHKERRQ(ierr);
#endif
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "MeshCreate_Cartesian"
PetscErrorCode PETSCDM_DLLEXPORT MeshCreate_Cartesian(Mesh mesh)
{
  ALE::Obj<ALE::CartesianMesh> *cm;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mesh,MESH_COOKIE,1);

  ierr = PetscMalloc(sizeof(ALE::Obj<ALE::CartesianMesh>), &cm);CHKERRQ(ierr);
  ierr = PetscLogObjectMemory(mesh, sizeof(ALE::Obj<ALE::CartesianMesh>));CHKERRQ(ierr);
  mesh->ops->view               = MeshView_Cartesian;
  mesh->ops->destroy            = MeshDestroy_Cartesian;
  mesh->ops->createglobalvector = MeshCreateGlobalVector;
  mesh->ops->getcoloring        = PETSC_NULL;
  mesh->ops->getmatrix          = MeshGetMatrix;
  mesh->ops->getinterpolation   = MeshGetInterpolation_Cartesian;
  mesh->ops->getinjection       = PETSC_NULL;
  mesh->ops->refine             = MeshRefine_Cartesian;
  mesh->ops->coarsen            = MeshCoarsen_Cartesian;
  mesh->ops->refinehierarchy    = PETSC_NULL;
  mesh->ops->coarsenhierarchy   = PETSC_NULL;

  mesh->m             = PETSC_NULL;
  mesh->globalScatter = PETSC_NULL;
  mesh->lf            = PETSC_NULL;
  mesh->lj            = PETSC_NULL;
  mesh->data          = cm;
  PetscFunctionReturn(0);
}
EXTERN_C_END
