#include <petsc-private/meshimpl.h>   /*I      "petscdmmesh.h"   I*/
#include <CartesianSieve.hh>

#undef __FUNCT__
#define __FUNCT__ "DMCartesianGetMesh"
/*@C
  DMCartesianGetMesh - Gets the internal mesh object

  Not collective

  Input Parameter:
. dm - the mesh object

  Output Parameter:
. m - the internal mesh object

  Level: advanced

.seealso MeshCreate(), MeshCartesianSetMesh()
@*/
PetscErrorCode DMCartesianGetMesh(DM dm, ALE::Obj<ALE::CartesianMesh>& m)
{
  DM_Cartesian *c = (DM_Cartesian *) dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  m = c->m;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMCartesianSetMesh"
/*@C
  DMCartesianSetMesh - Sets the internal mesh object

  Not collective

  Input Parameters:
+ mesh - the mesh object
- m - the internal mesh object

  Level: advanced

.seealso MeshCreate(), MeshCartesianGetMesh()
@*/
PetscErrorCode DMCartesianSetMesh(DM dm, const ALE::Obj<ALE::CartesianMesh>& m)
{
  DM_Cartesian *c = (DM_Cartesian *) dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  c->m = m;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMDestroy_Cartesian"
PetscErrorCode  DMDestroy_Cartesian(DM dm)
{
  DM_Cartesian *c = (DM_Cartesian *) dm->data;

  PetscFunctionBegin;
  c->m = PETSC_NULL;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMView_Cartesian_Ascii"
PetscErrorCode DMView_Cartesian_Ascii(const ALE::Obj<ALE::CartesianMesh>& mesh, PetscViewer viewer)
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
    /* FIX: Need to globalize */
    ierr = PetscViewerASCIIPrintf(viewer, "  %d vertices\n", mesh->getSieve()->getNumVertices());CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer, "  %d cells\n",    mesh->getSieve()->getNumCells());CHKERRQ(ierr);
  }
  ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode DMView_Cartesian(DM dm, PetscViewer viewer)
{
  ALE::Obj<ALE::CartesianMesh> m;
  PetscBool      iascii, isbinary, isdraw;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERASCII, &iascii);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERBINARY, &isbinary);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERDRAW, &isdraw);CHKERRQ(ierr);

  ierr = DMCartesianGetMesh(dm, m);CHKERRQ(ierr);
  if (iascii){
    ierr = DMView_Cartesian_Ascii(m, viewer);CHKERRQ(ierr);
  } else if (isbinary) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP, "Binary viewer not implemented for Cartesian Mesh");
  else if (isdraw) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP, "Draw viewer not implemented for Cartesian Mesh");
  else SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_SUP,"Viewer type %s not supported by this mesh object", ((PetscObject)viewer)->type_name);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMCreateInterpolation_Cartesian"
PetscErrorCode DMCreateInterpolation_Cartesian(DM fineMesh, DM coarseMesh, Mat *interpolation, Vec *scaling)
{
  ALE::Obj<ALE::CartesianMesh> coarse;
  ALE::Obj<ALE::CartesianMesh> fine;
  Mat                          P;
  PetscErrorCode               ierr;

  PetscFunctionBegin;
  ierr = DMCartesianGetMesh(fineMesh,   fine);CHKERRQ(ierr);
  ierr = DMCartesianGetMesh(coarseMesh, coarse);CHKERRQ(ierr);
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
#define __FUNCT__ "DMRefine_Cartesian"
PetscErrorCode DMRefine_Cartesian(DM mesh, MPI_Comm comm, DM *refinedMesh)
{
  ALE::Obj<ALE::CartesianMesh> oldMesh;
  PetscErrorCode               ierr;

  PetscFunctionBegin;
  ierr = DMCartesianGetMesh(mesh, oldMesh);CHKERRQ(ierr);
  ierr = DMCartesianCreate(comm, refinedMesh);CHKERRQ(ierr);
#if 0
  ALE::Obj<ALE::Mesh> newMesh = ALE::Generator::refineMesh(oldMesh, refinementLimit, false);
  ierr = MeshCartesianSetMesh(*refinedMesh, newMesh);CHKERRQ(ierr);
  const ALE::Obj<ALE::CartesianMesh::real_section_type>& s = newMesh->getRealSection("default");

  newMesh->setDiscretization(oldMesh->getDiscretization());
  newMesh->setBoundaryCondition(oldMesh->getBoundaryCondition());
  newMesh->setupField(s);
#else
  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP, "Not yet implemented");
#endif
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMCoarsen_Cartesian"
PetscErrorCode DMCoarsen_Cartesian(DM mesh, MPI_Comm comm, DM *coarseMesh)
{
  ALE::Obj<ALE::CartesianMesh> oldMesh;
  PetscErrorCode               ierr;

  PetscFunctionBegin;
  if (comm == MPI_COMM_NULL) comm = ((PetscObject)mesh)->comm;
  ierr = DMCartesianGetMesh(mesh, oldMesh);CHKERRQ(ierr);
  ierr = DMCartesianCreate(comm, coarseMesh);CHKERRQ(ierr);
#if 0
  ALE::Obj<ALE::Mesh> newMesh = ALE::Generator::refineMesh(oldMesh, refinementLimit, false);
  ierr = MeshCartesianSetMesh(*coarseMesh, newMesh);CHKERRQ(ierr);
  const ALE::Obj<ALE::CartesianMesh::real_section_type>& s = newMesh->getRealSection("default");

  newMesh->setDiscretization(oldMesh->getDiscretization());
  newMesh->setBoundaryCondition(oldMesh->getBoundaryCondition());
  newMesh->setupField(s);
#else
  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP, "Not yet implemented");
#endif
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMCartesianGetSectionReal"
PetscErrorCode DMCartesianGetSectionReal(DM dm, const char name[], SectionReal *section)
{
  ALE::Obj<ALE::CartesianMesh> m;
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  ierr = DMCartesianGetMesh(dm, m);CHKERRQ(ierr);
  ierr = SectionRealCreate(m->comm(), section);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) *section, name);CHKERRQ(ierr);
#if 0
  ierr = SectionRealSetSection(*section, m->getRealSection(std::string(name)));CHKERRQ(ierr);
  ierr = SectionRealSetBundle(*section, m);CHKERRQ(ierr);
#endif
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMSetFromOptions_Cartesian"
PetscErrorCode  DMSetFromOptions_Cartesian(DM dm)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  ierr = PetscOptionsHead("DMCartesian Options");CHKERRQ(ierr);
    /* Handle DMCartesian refinement */
    /* Handle associated vectors */
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "DMCreate_Cartesian"
PetscErrorCode DMCreate_Cartesian(DM dm)
{
  DM_Cartesian  *mesh;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  ierr = PetscNewLog(dm, DM_Cartesian, &mesh);CHKERRQ(ierr);
  dm->data = mesh;

  new(&mesh->m) ALE::Obj<ALE::CartesianMesh>(PETSC_NULL);

  ierr = PetscStrallocpy(VECSTANDARD, &dm->vectype);CHKERRQ(ierr);
  dm->ops->globaltolocalbegin = 0;
  dm->ops->globaltolocalend   = 0;
  dm->ops->localtoglobalbegin = 0;
  dm->ops->localtoglobalend   = 0;
  dm->ops->createglobalvector = 0; /* DMCreateGlobalVector_Cartesian; */
  dm->ops->createlocalvector  = 0; /* DMCreateLocalVector_Cartesian; */
  dm->ops->createinterpolation   = DMCreateInterpolation_Cartesian;
  dm->ops->getcoloring        = 0;
  dm->ops->creatematrix          = 0; /* DMCreateMatrix_Cartesian; */
  dm->ops->refine             = DMRefine_Cartesian;
  dm->ops->coarsen            = DMCoarsen_Cartesian;
  dm->ops->refinehierarchy    = 0;
  dm->ops->coarsenhierarchy   = 0;
  dm->ops->getinjection       = 0;
  dm->ops->getaggregates      = 0;
  dm->ops->destroy            = DMDestroy_Cartesian;
  dm->ops->view               = DMView_Cartesian;
  dm->ops->setfromoptions     = DMSetFromOptions_Cartesian;
  dm->ops->setup              = 0;
  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__
#define __FUNCT__ "DMCartesianCreate"
/*@
  DMCartesianCreate - Creates a DMCartesian object.

  Collective on MPI_Comm

  Input Parameter:
. comm - The communicator for the DMCartesian object

  Output Parameter:
. mesh  - The DMCartesian object

  Level: beginner

.keywords: DMCartesian, create
@*/
PetscErrorCode DMCartesianCreate(MPI_Comm comm, DM *mesh)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidPointer(mesh,2);
  ierr = DMCreate(comm, mesh);CHKERRQ(ierr);
  ierr = DMSetType(*mesh, DMCARTESIAN);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
