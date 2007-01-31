 
#include "src/dm/mesh/meshimpl.h"   /*I      "petscmesh.h"   I*/
#include "src/dm/mesh/meshvtk.h"

/* Logging support */
PetscCookie PETSCDM_DLLEXPORT SECTIONREAL_COOKIE = 0;
PetscEvent  SectionReal_View = 0;
PetscCookie PETSCDM_DLLEXPORT SECTIONINT_COOKIE = 0;
PetscEvent  SectionInt_View = 0;
PetscCookie PETSCDM_DLLEXPORT SECTIONPAIR_COOKIE = 0;
PetscEvent  SectionPair_View = 0;

#undef __FUNCT__  
#define __FUNCT__ "SectionView_Sieve_Ascii"
template<typename Section>
PetscErrorCode SectionView_Sieve_Ascii(const Obj<Section>& s, const char name[], PetscViewer viewer)
{
  // state 0: No header has been output
  // state 1: Only POINT_DATA has been output
  // state 2: Only CELL_DATA has been output
  // state 3: Output both, POINT_DATA last
  // state 4: Output both, CELL_DATA last
  const ALE::Obj<ALE::Mesh::topology_type>& topology = s->getTopology();
  PetscViewerFormat format;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = PetscViewerGetFormat(viewer, &format);CHKERRQ(ierr);
  if (format == PETSC_VIEWER_ASCII_VTK || format == PETSC_VIEWER_ASCII_VTK_CELL) {
    static PetscInt   stateId     = -1;
    PetscInt          doOutput    = 0;
    PetscInt          outputState = 0;
    PetscTruth        hasState;

    if (stateId < 0) {
      ierr = PetscObjectComposedDataRegister(&stateId);CHKERRQ(ierr);
      ierr = PetscObjectComposedDataSetInt((PetscObject) viewer, stateId, 0);CHKERRQ(ierr);
    }
    ierr = PetscObjectComposedDataGetInt((PetscObject) viewer, stateId, outputState, hasState);CHKERRQ(ierr);
    if (format == PETSC_VIEWER_ASCII_VTK) {
      if (outputState == 0) {
        outputState = 1;
        doOutput = 1;
      } else if (outputState == 1) {
        doOutput = 0;
      } else if (outputState == 2) {
        outputState = 3;
        doOutput = 1;
      } else if (outputState == 3) {
        doOutput = 0;
      } else if (outputState == 4) {
        SETERRQ(PETSC_ERR_ARG_WRONGSTATE, "Tried to output POINT_DATA again after intervening CELL_DATA");
      }
      const typename Section::patch_type         patch     = topology->getPatches().begin()->first;
      const ALE::Obj<ALE::Mesh::numbering_type>& numbering = ALE::Mesh::NumberingFactory::singleton(s->debug())->getNumbering(topology, patch, 0);
      PetscInt fiberDim = std::abs(s->getFiberDimension(patch, *topology->depthStratum(patch, 0)->begin()));

      if (doOutput) {
        ierr = PetscViewerASCIIPrintf(viewer, "POINT_DATA %d\n", numbering->getGlobalSize());CHKERRQ(ierr);
      }
      VTKViewer::writeField(s, std::string(name), fiberDim, numbering, viewer);
    } else {
      if (outputState == 0) {
        outputState = 2;
        doOutput = 1;
      } else if (outputState == 1) {
        outputState = 4;
        doOutput = 1;
      } else if (outputState == 2) {
        doOutput = 0;
      } else if (outputState == 3) {
        SETERRQ(PETSC_ERR_ARG_WRONGSTATE, "Tried to output CELL_DATA again after intervening POINT_DATA");
      } else if (outputState == 4) {
        doOutput = 0;
      }
      const typename Section::patch_type         patch     = topology->getPatches().begin()->first;
      const ALE::Obj<ALE::Mesh::numbering_type>& numbering = ALE::Mesh::NumberingFactory::singleton(s->debug())->getNumbering(topology, patch, topology->depth());
      PetscInt fiberDim = s->getFiberDimension(patch, *topology->heightStratum(patch, 0)->begin());

      if (doOutput) {
        ierr = PetscViewerASCIIPrintf(viewer, "CELL_DATA %d\n", numbering->getGlobalSize());CHKERRQ(ierr);
      }
      VTKViewer::writeField(s, std::string(name), fiberDim, numbering, viewer);
    }
    ierr = PetscObjectComposedDataSetInt((PetscObject) viewer, stateId, outputState);CHKERRQ(ierr);
  } else {
    s->view(name);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SectionRealView_Sieve"
PetscErrorCode SectionRealView_Sieve(SectionReal section, PetscViewer viewer)
{
  PetscTruth     iascii, isbinary, isdraw;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscTypeCompare((PetscObject) viewer, PETSC_VIEWER_ASCII, &iascii);CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject) viewer, PETSC_VIEWER_BINARY, &isbinary);CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject) viewer, PETSC_VIEWER_DRAW, &isdraw);CHKERRQ(ierr);

  if (iascii){
    ALE::Obj<ALE::Mesh::real_section_type> s;
    const char                            *name;

    ierr = SectionRealGetSection(section, s);CHKERRQ(ierr);
    ierr = PetscObjectGetName((PetscObject) section, &name);CHKERRQ(ierr);
    ierr = SectionView_Sieve_Ascii(s, name, viewer);CHKERRQ(ierr);
  } else if (isbinary) {
    SETERRQ(PETSC_ERR_SUP, "Binary viewer not implemented for Section");
  } else if (isdraw){ 
    SETERRQ(PETSC_ERR_SUP, "Draw viewer not implemented for Section");
  } else {
    SETERRQ1(PETSC_ERR_SUP,"Viewer type %s not supported by this section object", ((PetscObject)viewer)->type_name);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SectionRealView"
/*@C
   SectionRealView - Views a Section object. 

   Collective on Section

   Input Parameters:
+  section - the Section
-  viewer - an optional visualization context

   Notes:
   The available visualization contexts include
+     PETSC_VIEWER_STDOUT_SELF - standard output (default)
-     PETSC_VIEWER_STDOUT_WORLD - synchronized standard
         output where only the first processor opens
         the file.  All other processors send their 
         data to the first processor to print. 

   You can change the format the section is printed using the 
   option PetscViewerSetFormat().

   The user can open alternative visualization contexts with
+    PetscViewerASCIIOpen() - Outputs section to a specified file
.    PetscViewerBinaryOpen() - Outputs section in binary to a
         specified file; corresponding input uses SectionLoad()
.    PetscViewerDrawOpen() - Outputs section to an X window display

   The user can call PetscViewerSetFormat() to specify the output
   format of ASCII printed objects (when using PETSC_VIEWER_STDOUT_SELF,
   PETSC_VIEWER_STDOUT_WORLD and PetscViewerASCIIOpen).  Available formats include
+    PETSC_VIEWER_ASCII_DEFAULT - default, prints section information
-    PETSC_VIEWER_ASCII_VTK - outputs a VTK file describing the section

   Level: beginner

   Concepts: section^printing
   Concepts: section^saving to disk

.seealso: VecView(), PetscViewerASCIIOpen(), PetscViewerDrawOpen(), PetscViewerBinaryOpen(), PetscViewerCreate()
@*/
PetscErrorCode SectionRealView(SectionReal section, PetscViewer viewer)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(section, SECTIONREAL_COOKIE, 1);
  PetscValidType(section, 1);
  if (!viewer) {
    ierr = PetscViewerASCIIGetStdout(section->comm,&viewer);CHKERRQ(ierr);
  }
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_COOKIE, 2);
  PetscCheckSameComm(section, 1, viewer, 2);

  ierr = PetscLogEventBegin(SectionReal_View,0,0,0,0);CHKERRQ(ierr);
  ierr = (*section->ops->view)(section, viewer);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(SectionReal_View,0,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SectionRealDuplicate"
/*@C
  SectionRealDuplicate - Create an equivalent Section object

  Not collective

  Input Parameter:
. section - the section object

  Output Parameter:
. newSection - the duplicate
 
  Level: advanced

.seealso SectionRealCreate(), SectionRealSetSection()
@*/
PetscErrorCode PETSCDM_DLLEXPORT SectionRealDuplicate(SectionReal section, SectionReal *newSection)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(section, SECTIONREAL_COOKIE, 1);
  PetscValidPointer(newSection, 2);
  const ALE::Obj<ALE::Mesh::real_section_type>& s = section->s;
  ALE::Obj<ALE::Mesh::real_section_type>        t = new ALE::Mesh::real_section_type(s->getTopology());

  t->setAtlas(s->getAtlas());
  t->allocateStorage();
  t->copyBC(s);
  ierr = SectionRealCreate(s->comm(), newSection);CHKERRQ(ierr);
  ierr = SectionRealSetSection(*newSection, t);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SectionRealGetSection"
/*@C
  SectionRealGetSection - Gets the internal section object

  Not collective

  Input Parameter:
. section - the section object

  Output Parameter:
. s - the internal section object
 
  Level: advanced

.seealso SectionRealCreate(), SectionRealSetSection()
@*/
PetscErrorCode PETSCDM_DLLEXPORT SectionRealGetSection(SectionReal section, ALE::Obj<ALE::Mesh::real_section_type>& s)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(section, SECTIONREAL_COOKIE, 1);
  s = section->s;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SectionRealSetSection"
/*@C
  SectionRealSetSection - Sets the internal section object

  Not collective

  Input Parameters:
+ section - the section object
- s - the internal section object
 
  Level: advanced

.seealso SectionRealCreate(), SectionRealGetSection()
@*/
PetscErrorCode PETSCDM_DLLEXPORT SectionRealSetSection(SectionReal section, const ALE::Obj<ALE::Mesh::real_section_type>& s)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(section, SECTIONREAL_COOKIE, 1);
  section->s = s;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SectionRealGetTopology"
/*@C
  SectionRealGetTopology - Gets the internal section topology

  Not collective

  Input Parameter:
. section - the section object

  Output Parameter:
. t - the internal section topology
 
  Level: advanced

.seealso SectionRealCreate(), SectionRealGetSection(), SectionRealSetSection()
@*/
PetscErrorCode PETSCDM_DLLEXPORT SectionRealGetTopology(SectionReal section, ALE::Obj<ALE::Mesh::topology_type>& t)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(section, SECTIONREAL_COOKIE, 1);
  t = section->s->getTopology();
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SectionRealSetTopology"
/*@C
  SectionRealSetTopology - Sets the internal section topology

  Not collective

  Input Parameters:
+ section - the section object
- t - the internal section topology
 
  Level: advanced

.seealso SectionRealCreate(), SectionRealGetSection(), SectionRealSetSection()
@*/
PetscErrorCode PETSCDM_DLLEXPORT SectionRealSetTopology(SectionReal section, const ALE::Obj<ALE::Mesh::topology_type>& t)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(section, SECTIONREAL_COOKIE, 1);
  section->s->setTopology(t);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SectionRealCreate"
/*@C
  SectionRealCreate - Creates a Section object, used to manage data for an unstructured problem
  described by a Sieve.

  Collective on MPI_Comm

  Input Parameter:
. comm - the processors that will share the global section

  Output Parameters:
. section - the section object

  Level: advanced

.seealso SectionRealDestroy(), SectionRealView()
@*/
PetscErrorCode PETSCDM_DLLEXPORT SectionRealCreate(MPI_Comm comm, SectionReal *section)
{
  PetscErrorCode ierr;
  SectionReal    s;

  PetscFunctionBegin;
  PetscValidPointer(section,2);
  *section = PETSC_NULL;

  ierr = PetscHeaderCreate(s,_p_SectionReal,struct _SectionRealOps,SECTIONREAL_COOKIE,0,"SectionReal",comm,SectionRealDestroy,0);CHKERRQ(ierr);
  s->ops->view     = SectionRealView_Sieve;
  s->ops->restrict = SectionRealRestrict;
  s->ops->update   = SectionRealUpdate;

  ierr = PetscObjectChangeTypeName((PetscObject) s, "sieve");CHKERRQ(ierr);

  s->s             = new ALE::Mesh::real_section_type(comm);
  *section = s;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SectionRealDestroy"
/*@C
  SectionRealDestroy - Destroys a section.

  Collective on Section

  Input Parameter:
. section - the section object

  Level: advanced

.seealso SectionRealCreate(), SectionRealView()
@*/
PetscErrorCode PETSCDM_DLLEXPORT SectionRealDestroy(SectionReal section)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(section, SECTIONREAL_COOKIE, 1);
  if (--section->refct > 0) PetscFunctionReturn(0);
  section->s = PETSC_NULL;
  ierr = PetscHeaderDestroy(section);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SectionRealDistribute"
/*@C
  SectionRealDistribute - Distributes the sections.

  Not Collective

  Input Parameters:
+ serialSection - The original Section object
- parallelMesh - The parallel Mesh

  Output Parameter:
. parallelSection - The distributed Section object

  Level: intermediate

.keywords: mesh, section, distribute
.seealso: MeshCreate()
@*/
PetscErrorCode SectionRealDistribute(SectionReal serialSection, Mesh parallelMesh, SectionReal *parallelSection)
{
  ALE::Obj<ALE::Mesh::real_section_type> oldSection;
  ALE::Obj<ALE::Mesh>               m;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = SectionRealGetSection(serialSection, oldSection);CHKERRQ(ierr);
  ierr = MeshGetMesh(parallelMesh, m);CHKERRQ(ierr);
  ierr = SectionRealCreate(oldSection->comm(), parallelSection);CHKERRQ(ierr);
  ALE::Obj<ALE::Mesh::real_section_type> newSection = ALE::New::Distribution<ALE::Mesh::topology_type>::distributeSection(oldSection, m->getTopology(), m->getTopology()->getDistSendOverlap(), m->getTopology()->getDistRecvOverlap());
  ierr = SectionRealSetSection(*parallelSection, newSection);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SectionRealRestrict"
/*@C
  SectionRealRestrict - Restricts the SectionReal to a subset of the topology, returning an array of values.

  Not collective

  Input Parameters:
+ section - the section object
- point - the Sieve point

  Output Parameter:
. values - The values associated with the submesh

  Level: advanced

.seealso SectionUpdate(), SectionCreate(), SectionView()
@*/
PetscErrorCode PETSCDM_DLLEXPORT SectionRealRestrict(SectionReal section, PetscInt point, PetscScalar *values[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(section, SECTIONREAL_COOKIE, 1);
  PetscValidScalarPointer(values,3);
  *values = (PetscScalar *) section->s->restrict(0, point);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SectionRealUpdate"
/*@C
  SectionRealUpdate - Updates the array of values associated to a subset of the topology in this Section.

  Not collective

  Input Parameters:
+ section - the section object
. point - the Sieve point
- values - The values associated with the submesh

  Level: advanced

.seealso SectionRealRestrict(), SectionRealCreate(), SectionRealView()
@*/
PetscErrorCode PETSCDM_DLLEXPORT SectionRealUpdate(SectionReal section, PetscInt point, const PetscScalar values[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(section, SECTIONREAL_COOKIE, 1);
  PetscValidScalarPointer(values,3);
  section->s->update(0, point, values);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SectionRealUpdateAdd"
/*@C
  SectionRealUpdateAdd - Updates the array of values associated to a subset of the topology in this Section.

  Not collective

  Input Parameters:
+ section - the section object
. point - the Sieve point
- values - The values associated with the submesh

  Level: advanced

.seealso SectionRealRestrict(), SectionRealCreate(), SectionRealView()
@*/
PetscErrorCode PETSCDM_DLLEXPORT SectionRealUpdateAdd(SectionReal section, PetscInt point, const PetscScalar values[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(section, SECTIONREAL_COOKIE, 1);
  PetscValidScalarPointer(values,3);
  section->s->updateAdd(0, point, values);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SectionRealComplete"
/*@C
  SectionRealComplete - Exchanges data across the mesh overlap.

  Not collective

  Input Parameter:
. section - the section object

  Level: advanced

.seealso SectionRealRestrict(), SectionRealCreate(), SectionRealView()
@*/
PetscErrorCode SectionRealComplete(SectionReal section)
{
  Obj<ALE::Mesh::real_section_type> s;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = SectionRealGetSection(section, s);CHKERRQ(ierr);
  ALE::New::Distribution<ALE::Mesh::topology_type>::completeSection(s);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SectionRealGetLocalVector"
/*@C
  SectionRealGetLocalVector - Retrieves the local section storage as a Vec

  Not collective

  Input Parameter:
. section - the section object

  Output Parameter:
. lv - the local vector

  Level: advanced

.seealso SectionRealRestrict(), SectionRealCreate(), SectionRealView()
@*/
PetscErrorCode SectionRealGetLocalVector(SectionReal section, Vec *lv)
{
  const ALE::Mesh::real_section_type::patch_type patch = 0;
  Obj<ALE::Mesh::real_section_type> s;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = SectionRealGetSection(section, s);CHKERRQ(ierr);
  ierr = VecCreateSeqWithArray(PETSC_COMM_SELF, s->size(patch), s->restrict(patch), lv);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SectionRealZero"
/*@C
  SectionRealZero - Zero out the entries

  Not collective

  Input Parameter:
. section - the section object

  Level: advanced

.seealso SectionRealRestrict(), SectionRealCreate(), SectionRealView()
@*/
PetscErrorCode SectionRealZero(SectionReal section)
{
  const ALE::Mesh::real_section_type::patch_type patch = 0;
  Obj<ALE::Mesh::real_section_type> s;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = SectionRealGetSection(section, s);CHKERRQ(ierr);
  if (s->hasPatch(patch)) {
    s->zero(patch);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MeshGetVertexSectionReal"
/*@C
  MeshGetVertexSectionReal - Create a Section over the vertices with the specified fiber dimension

  Collective on Mesh

  Input Parameters:
+ mesh - The Mesh object
- fiberDim - The section name

  Output Parameter:
. section - The section

  Level: intermediate

.keywords: mesh, section, vertex
.seealso: MeshCreate(), SectionRealCreate()
@*/
PetscErrorCode MeshGetVertexSectionReal(Mesh mesh, PetscInt fiberDim, SectionReal *section)
{
  ALE::Obj<ALE::Mesh> m;
  ALE::Obj<ALE::Mesh::real_section_type> s;
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  ierr = MeshGetMesh(mesh, m);CHKERRQ(ierr);
  ierr = SectionRealCreate(m->comm(), section);CHKERRQ(ierr);
  ierr = SectionRealSetTopology(*section, m->getTopology());CHKERRQ(ierr);
  ierr = SectionRealGetSection(*section, s);CHKERRQ(ierr);
  s->setFiberDimensionByDepth(0, 0, fiberDim);
  s->allocate();
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MeshGetCellSectionReal"
/*@C
  MeshGetCellSectionReal - Create a Section over the cells with the specified fiber dimension

  Collective on Mesh

  Input Parameters:
+ mesh - The Mesh object
- fiberDim - The section name

  Output Parameter:
. section - The section

  Level: intermediate

.keywords: mesh, section, cell
.seealso: MeshCreate(), SectionRealCreate()
@*/
PetscErrorCode MeshGetCellSectionReal(Mesh mesh, PetscInt fiberDim, SectionReal *section)
{
  ALE::Obj<ALE::Mesh> m;
  ALE::Obj<ALE::Mesh::real_section_type> s;
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  ierr = MeshGetMesh(mesh, m);CHKERRQ(ierr);
  ierr = SectionRealCreate(m->comm(), section);CHKERRQ(ierr);
  ierr = SectionRealSetTopology(*section, m->getTopology());CHKERRQ(ierr);
  ierr = SectionRealGetSection(*section, s);CHKERRQ(ierr);
  s->setFiberDimensionByHeight(0, 0, fiberDim);
  s->allocate();
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SectionIntView_Sieve"
PetscErrorCode SectionIntView_Sieve(SectionInt section, PetscViewer viewer)
{
  PetscTruth     iascii, isbinary, isdraw;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscTypeCompare((PetscObject) viewer, PETSC_VIEWER_ASCII, &iascii);CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject) viewer, PETSC_VIEWER_BINARY, &isbinary);CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject) viewer, PETSC_VIEWER_DRAW, &isdraw);CHKERRQ(ierr);

  if (iascii){
    ALE::Obj<ALE::Mesh::int_section_type> s;
    const char                           *name;

    ierr = SectionIntGetSection(section, s);CHKERRQ(ierr);
    ierr = PetscObjectGetName((PetscObject) section, &name);CHKERRQ(ierr);
    ierr = SectionView_Sieve_Ascii(s, name, viewer);CHKERRQ(ierr);
  } else if (isbinary) {
    SETERRQ(PETSC_ERR_SUP, "Binary viewer not implemented for Section");
  } else if (isdraw){ 
    SETERRQ(PETSC_ERR_SUP, "Draw viewer not implemented for Section");
  } else {
    SETERRQ1(PETSC_ERR_SUP,"Viewer type %s not supported by this section object", ((PetscObject)viewer)->type_name);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SectionIntView"
/*@C
   SectionIntView - Views a Section object. 

   Collective on Section

   Input Parameters:
+  section - the Section
-  viewer - an optional visualization context

   Notes:
   The available visualization contexts include
+     PETSC_VIEWER_STDOUT_SELF - standard output (default)
-     PETSC_VIEWER_STDOUT_WORLD - synchronized standard
         output where only the first processor opens
         the file.  All other processors send their 
         data to the first processor to print. 

   You can change the format the section is printed using the 
   option PetscViewerSetFormat().

   The user can open alternative visualization contexts with
+    PetscViewerASCIIOpen() - Outputs section to a specified file
.    PetscViewerBinaryOpen() - Outputs section in binary to a
         specified file; corresponding input uses SectionLoad()
.    PetscViewerDrawOpen() - Outputs section to an X window display

   The user can call PetscViewerSetFormat() to specify the output
   format of ASCII printed objects (when using PETSC_VIEWER_STDOUT_SELF,
   PETSC_VIEWER_STDOUT_WORLD and PetscViewerASCIIOpen).  Available formats include
+    PETSC_VIEWER_ASCII_DEFAULT - default, prints section information
-    PETSC_VIEWER_ASCII_VTK - outputs a VTK file describing the section

   Level: beginner

   Concepts: section^printing
   Concepts: section^saving to disk

.seealso: VecView(), PetscViewerASCIIOpen(), PetscViewerDrawOpen(), PetscViewerBinaryOpen(), PetscViewerCreate()
@*/
PetscErrorCode SectionIntView(SectionInt section, PetscViewer viewer)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(section, SECTIONINT_COOKIE, 1);
  PetscValidType(section, 1);
  if (!viewer) {
    ierr = PetscViewerASCIIGetStdout(section->comm,&viewer);CHKERRQ(ierr);
  }
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_COOKIE, 2);
  PetscCheckSameComm(section, 1, viewer, 2);

  ierr = PetscLogEventBegin(SectionInt_View,0,0,0,0);CHKERRQ(ierr);
  ierr = (*section->ops->view)(section, viewer);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(SectionInt_View,0,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SectionIntGetSection"
/*@C
  SectionIntGetSection - Gets the internal section object

  Not collective

  Input Parameter:
. section - the section object

  Output Parameter:
. s - the internal section object
 
  Level: advanced

.seealso SectionIntCreate(), SectionIntSetSection()
@*/
PetscErrorCode PETSCDM_DLLEXPORT SectionIntGetSection(SectionInt section, ALE::Obj<ALE::Mesh::int_section_type>& s)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(section, SECTIONINT_COOKIE, 1);
  s = section->s;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SectionIntSetSection"
/*@C
  SectionIntSetSection - Sets the internal section object

  Not collective

  Input Parameters:
+ section - the section object
- s - the internal section object
 
  Level: advanced

.seealso SectionIntCreate(), SectionIntGetSection()
@*/
PetscErrorCode PETSCDM_DLLEXPORT SectionIntSetSection(SectionInt section, const ALE::Obj<ALE::Mesh::int_section_type>& s)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(section, SECTIONINT_COOKIE, 1);
  section->s = s;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SectionIntGetTopology"
/*@C
  SectionIntGetTopology - Gets the internal section topology

  Not collective

  Input Parameter:
. section - the section object

  Output Parameter:
. t - the internal section topology
 
  Level: advanced

.seealso SectionIntCreate(), SectionIntGetSection(), SectionIntSetSection()
@*/
PetscErrorCode PETSCDM_DLLEXPORT SectionIntGetTopology(SectionInt section, ALE::Obj<ALE::Mesh::topology_type>& t)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(section, SECTIONINT_COOKIE, 1);
  t = section->s->getTopology();
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SectionIntSetTopology"
/*@C
  SectionIntSetTopology - Sets the internal section topology

  Not collective

  Input Parameters:
+ section - the section object
- t - the internal section topology
 
  Level: advanced

.seealso SectionIntCreate(), SectionIntGetSection(), SectionIntSetSection()
@*/
PetscErrorCode PETSCDM_DLLEXPORT SectionIntSetTopology(SectionInt section, const ALE::Obj<ALE::Mesh::topology_type>& t)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(section, SECTIONINT_COOKIE, 1);
  section->s->setTopology(t);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SectionIntCreate"
/*@C
  SectionIntCreate - Creates a Section object, used to manage data for an unstructured problem
  described by a Sieve.

  Collective on MPI_Comm

  Input Parameter:
. comm - the processors that will share the global section

  Output Parameters:
. section - the section object

  Level: advanced

.seealso SectionIntDestroy(), SectionIntView()
@*/
PetscErrorCode PETSCDM_DLLEXPORT SectionIntCreate(MPI_Comm comm, SectionInt *section)
{
  PetscErrorCode ierr;
  SectionInt    s;

  PetscFunctionBegin;
  PetscValidPointer(section,2);
  *section = PETSC_NULL;

  ierr = PetscHeaderCreate(s,_p_SectionInt,struct _SectionIntOps,SECTIONINT_COOKIE,0,"SectionInt",comm,SectionIntDestroy,0);CHKERRQ(ierr);
  s->ops->view     = SectionIntView_Sieve;
  s->ops->restrict = SectionIntRestrict;
  s->ops->update   = SectionIntUpdate;

  ierr = PetscObjectChangeTypeName((PetscObject) s, "sieve");CHKERRQ(ierr);

  s->s             = new ALE::Mesh::int_section_type(comm);
  *section = s;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SectionIntDestroy"
/*@C
  SectionIntDestroy - Destroys a section.

  Collective on Section

  Input Parameter:
. section - the section object

  Level: advanced

.seealso SectionIntCreate(), SectionIntView()
@*/
PetscErrorCode PETSCDM_DLLEXPORT SectionIntDestroy(SectionInt section)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(section, SECTIONINT_COOKIE, 1);
  if (--section->refct > 0) PetscFunctionReturn(0);
  section->s = PETSC_NULL;
  ierr = PetscHeaderDestroy(section);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SectionIntDistribute"
/*@C
  SectionIntDistribute - Distributes the sections.

  Not Collective

  Input Parameters:
+ serialSection - The original Section object
- parallelMesh - The parallel Mesh

  Output Parameter:
. parallelSection - The distributed Section object

  Level: intermediate

.keywords: mesh, section, distribute
.seealso: MeshCreate()
@*/
PetscErrorCode SectionIntDistribute(SectionInt serialSection, Mesh parallelMesh, SectionInt *parallelSection)
{
  ALE::Obj<ALE::Mesh::int_section_type> oldSection;
  ALE::Obj<ALE::Mesh> m;
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  ierr = SectionIntGetSection(serialSection, oldSection);CHKERRQ(ierr);
  ierr = MeshGetMesh(parallelMesh, m);CHKERRQ(ierr);
  ierr = SectionIntCreate(oldSection->comm(), parallelSection);CHKERRQ(ierr);
  ALE::Obj<ALE::Mesh::int_section_type> newSection = ALE::New::Distribution<ALE::Mesh::topology_type>::distributeSection(oldSection, m->getTopology(), m->getTopology()->getDistSendOverlap(), m->getTopology()->getDistRecvOverlap());
  ierr = SectionIntSetSection(*parallelSection, newSection);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SectionIntRestrict"
/*@C
  SectionIntRestrict - Restricts the SectionInt to a subset of the topology, returning an array of values.

  Not collective

  Input Parameters:
+ section - the section object
- point - the Sieve point

  Output Parameter:
. values - The values associated with the submesh

  Level: advanced

.seealso SectionIntUpdate(), SectionIntCreate(), SectionIntView()
@*/
PetscErrorCode PETSCDM_DLLEXPORT SectionIntRestrict(SectionInt section, PetscInt point, PetscInt *values[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(section, SECTIONINT_COOKIE, 1);
  PetscValidPointer(values,3);
  *values = (PetscInt *) section->s->restrict(0, point);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SectionIntUpdate"
/*@C
  SectionIntUpdate - Updates the array of values associated to a subset of the topology in this Section.

  Not collective

  Input Parameters:
+ section - the section object
. point - the Sieve point
- values - The values associated with the submesh

  Level: advanced

.seealso SectionIntRestrict(), SectionIntCreate(), SectionIntView()
@*/
PetscErrorCode PETSCDM_DLLEXPORT SectionIntUpdate(SectionInt section, PetscInt point, const PetscInt values[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(section, SECTIONINT_COOKIE, 1);
  PetscValidIntPointer(values,3);
  section->s->update(0, point, values);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SectionIntUpdateAdd"
/*@C
  SectionIntUpdateAdd - Updates the array of values associated to a subset of the topology in this Section.

  Not collective

  Input Parameters:
+ section - the section object
. point - the Sieve point
- values - The values associated with the submesh

  Level: advanced

.seealso SectionIntRestrict(), SectionIntCreate(), SectionIntView()
@*/
PetscErrorCode PETSCDM_DLLEXPORT SectionIntUpdateAdd(SectionInt section, PetscInt point, const PetscInt values[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(section, SECTIONREAL_COOKIE, 1);
  PetscValidScalarPointer(values,3);
  section->s->updateAdd(0, point, values);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SectionIntComplete"
/*@C
  SectionIntComplete - Exchanges data across the mesh overlap.

  Not collective

  Input Parameter:
. section - the section object

  Level: advanced

.seealso SectionIntRestrict(), SectionIntCreate(), SectionIntView()
@*/
PetscErrorCode SectionIntComplete(SectionInt section)
{
  Obj<ALE::Mesh::int_section_type> s;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = SectionIntGetSection(section, s);CHKERRQ(ierr);
  ALE::New::Distribution<ALE::Mesh::topology_type>::completeSection(s);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MeshGetVertexSectionInt"
/*@C
  MeshGetVertexSectionInt - Create a Section over the vertices with the specified fiber dimension

  Collective on Mesh

  Input Parameters:
+ mesh - The Mesh object
- fiberDim - The section name

  Output Parameter:
. section - The section

  Level: intermediate

.keywords: mesh, section, vertex
.seealso: MeshCreate(), SectionIntCreate()
@*/
PetscErrorCode MeshGetVertexSectionInt(Mesh mesh, PetscInt fiberDim, SectionInt *section)
{
  ALE::Obj<ALE::Mesh> m;
  ALE::Obj<ALE::Mesh::int_section_type> s;
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  ierr = MeshGetMesh(mesh, m);CHKERRQ(ierr);
  ierr = SectionIntCreate(m->comm(), section);CHKERRQ(ierr);
  ierr = SectionIntSetTopology(*section, m->getTopology());CHKERRQ(ierr);
  ierr = SectionIntGetSection(*section, s);CHKERRQ(ierr);
  s->setFiberDimensionByDepth(0, 0, fiberDim);
  s->allocate();
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MeshGetCellSectionInt"
/*@C
  MeshGetCellSectionInt - Create a Section over the cells with the specified fiber dimension

  Collective on Mesh

  Input Parameters:
+ mesh - The Mesh object
- fiberDim - The section name

  Output Parameter:
. section - The section

  Level: intermediate

.keywords: mesh, section, cell
.seealso: MeshCreate(), SectionIntCreate()
@*/
PetscErrorCode MeshGetCellSectionInt(Mesh mesh, PetscInt fiberDim, SectionInt *section)
{
  ALE::Obj<ALE::Mesh> m;
  ALE::Obj<ALE::Mesh::int_section_type> s;
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  ierr = MeshGetMesh(mesh, m);CHKERRQ(ierr);
  ierr = SectionIntCreate(m->comm(), section);CHKERRQ(ierr);
  ierr = SectionIntSetTopology(*section, m->getTopology());CHKERRQ(ierr);
  ierr = SectionIntGetSection(*section, s);CHKERRQ(ierr);
  s->setFiberDimensionByHeight(0, 0, fiberDim);
  s->allocate();
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SectionPairView_Sieve"
PetscErrorCode SectionPairView_Sieve(SectionPair section, PetscViewer viewer)
{
  PetscTruth     iascii, isbinary, isdraw;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscTypeCompare((PetscObject) viewer, PETSC_VIEWER_ASCII, &iascii);CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject) viewer, PETSC_VIEWER_BINARY, &isbinary);CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject) viewer, PETSC_VIEWER_DRAW, &isdraw);CHKERRQ(ierr);

  if (iascii){
    ALE::Obj<ALE::Mesh::pair_section_type> s;
    const char                            *name;

    ierr = SectionPairGetSection(section, s);CHKERRQ(ierr);
    ierr = PetscObjectGetName((PetscObject) section, &name);CHKERRQ(ierr);
    //FIX ierr = SectionView_Sieve_Ascii(s, name, viewer);CHKERRQ(ierr);
  } else if (isbinary) {
    SETERRQ(PETSC_ERR_SUP, "Binary viewer not implemented for Section");
  } else if (isdraw){ 
    SETERRQ(PETSC_ERR_SUP, "Draw viewer not implemented for Section");
  } else {
    SETERRQ1(PETSC_ERR_SUP,"Viewer type %s not supported by this section object", ((PetscObject)viewer)->type_name);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SectionPairView"
/*@C
   SectionPairView - Views a Section object. 

   Collective on Section

   Input Parameters:
+  section - the Section
-  viewer - an optional visualization context

   Notes:
   The available visualization contexts include
+     PETSC_VIEWER_STDOUT_SELF - standard output (default)
-     PETSC_VIEWER_STDOUT_WORLD - synchronized standard
         output where only the first processor opens
         the file.  All other processors send their 
         data to the first processor to print. 

   You can change the format the section is printed using the 
   option PetscViewerSetFormat().

   The user can open alternative visualization contexts with
+    PetscViewerASCIIOpen() - Outputs section to a specified file
.    PetscViewerBinaryOpen() - Outputs section in binary to a
         specified file; corresponding input uses SectionLoad()
.    PetscViewerDrawOpen() - Outputs section to an X window display

   The user can call PetscViewerSetFormat() to specify the output
   format of ASCII printed objects (when using PETSC_VIEWER_STDOUT_SELF,
   PETSC_VIEWER_STDOUT_WORLD and PetscViewerASCIIOpen).  Available formats include
+    PETSC_VIEWER_ASCII_DEFAULT - default, prints section information
-    PETSC_VIEWER_ASCII_VTK - outputs a VTK file describing the section

   Level: beginner

   Concepts: section^printing
   Concepts: section^saving to disk

.seealso: VecView(), PetscViewerASCIIOpen(), PetscViewerDrawOpen(), PetscViewerBinaryOpen(), PetscViewerCreate()
@*/
PetscErrorCode SectionPairView(SectionPair section, PetscViewer viewer)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(section, SECTIONPAIR_COOKIE, 1);
  PetscValidType(section, 1);
  if (!viewer) {
    ierr = PetscViewerASCIIGetStdout(section->comm,&viewer);CHKERRQ(ierr);
  }
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_COOKIE, 2);
  PetscCheckSameComm(section, 1, viewer, 2);

  ierr = PetscLogEventBegin(SectionPair_View,0,0,0,0);CHKERRQ(ierr);
  ierr = (*section->ops->view)(section, viewer);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(SectionPair_View,0,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SectionPairGetSection"
/*@C
  SectionPairGetSection - Gets the internal section object

  Not collective

  Input Parameter:
. section - the section object

  Output Parameter:
. s - the internal section object
 
  Level: advanced

.seealso SectionPairCreate(), SectionPairSetSection()
@*/
PetscErrorCode PETSCDM_DLLEXPORT SectionPairGetSection(SectionPair section, ALE::Obj<ALE::Mesh::pair_section_type>& s)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(section, SECTIONPAIR_COOKIE, 1);
  s = section->s;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SectionPairSetSection"
/*@C
  SectionPairSetSection - Sets the internal section object

  Not collective

  Input Parameters:
+ section - the section object
- s - the internal section object
 
  Level: advanced

.seealso SectionPairCreate(), SectionPairGetSection()
@*/
PetscErrorCode PETSCDM_DLLEXPORT SectionPairSetSection(SectionPair section, const ALE::Obj<ALE::Mesh::pair_section_type>& s)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(section, SECTIONPAIR_COOKIE, 1);
  section->s = s;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SectionPairGetTopology"
/*@C
  SectionPairGetTopology - Gets the internal section topology

  Not collective

  Input Parameter:
. section - the section object

  Output Parameter:
. t - the internal section topology
 
  Level: advanced

.seealso SectionPairCreate(), SectionPairGetSection(), SectionPairSetSection()
@*/
PetscErrorCode PETSCDM_DLLEXPORT SectionPairGetTopology(SectionPair section, ALE::Obj<ALE::Mesh::topology_type>& t)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(section, SECTIONPAIR_COOKIE, 1);
  t = section->s->getTopology();
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SectionPairSetTopology"
/*@C
  SectionPairSetTopology - Sets the internal section topology

  Not collective

  Input Parameters:
+ section - the section object
- t - the internal section topology
 
  Level: advanced

.seealso SectionPairCreate(), SectionPairGetSection(), SectionPairSetSection()
@*/
PetscErrorCode PETSCDM_DLLEXPORT SectionPairSetTopology(SectionPair section, const ALE::Obj<ALE::Mesh::topology_type>& t)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(section, SECTIONPAIR_COOKIE, 1);
  section->s->setTopology(t);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SectionPairCreate"
/*@C
  SectionPairCreate - Creates a Section object, used to manage data for an unstructured problem
  described by a Sieve.

  Collective on MPI_Comm

  Input Parameter:
. comm - the processors that will share the global section

  Output Parameters:
. section - the section object

  Level: advanced

.seealso SectionPairDestroy(), SectionPairView()
@*/
PetscErrorCode PETSCDM_DLLEXPORT SectionPairCreate(MPI_Comm comm, SectionPair *section)
{
  PetscErrorCode ierr;
  SectionPair    s;

  PetscFunctionBegin;
  PetscValidPointer(section,2);
  *section = PETSC_NULL;

  ierr = PetscHeaderCreate(s,_p_SectionPair,struct _SectionPairOps,SECTIONPAIR_COOKIE,0,"SectionPair",comm,SectionPairDestroy,0);CHKERRQ(ierr);
  s->ops->view     = SectionPairView_Sieve;
  s->ops->restrict = SectionPairRestrict;
  s->ops->update   = SectionPairUpdate;

  ierr = PetscObjectChangeTypeName((PetscObject) s, "sieve");CHKERRQ(ierr);

  s->s             = new ALE::Mesh::pair_section_type(comm);
  *section = s;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SectionPairDestroy"
/*@C
  SectionPairDestroy - Destroys a section.

  Collective on Section

  Input Parameter:
. section - the section object

  Level: advanced

.seealso SectionPairCreate(), SectionPairView()
@*/
PetscErrorCode PETSCDM_DLLEXPORT SectionPairDestroy(SectionPair section)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(section, SECTIONPAIR_COOKIE, 1);
  if (--section->refct > 0) PetscFunctionReturn(0);
  section->s = PETSC_NULL;
  ierr = PetscHeaderDestroy(section);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SectionPairDistribute"
/*@C
  SectionPairDistribute - Distributes the sections.

  Not Collective

  Input Parameters:
+ serialSection - The original Section object
- parallelMesh - The parallel Mesh

  Output Parameter:
. parallelSection - The distributed Section object

  Level: intermediate

.keywords: mesh, section, distribute
.seealso: MeshCreate()
@*/
PetscErrorCode SectionPairDistribute(SectionPair serialSection, Mesh parallelMesh, SectionPair *parallelSection)
{
  ALE::Obj<ALE::Mesh::pair_section_type> oldSection;
  ALE::Obj<ALE::Mesh> m;
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  ierr = SectionPairGetSection(serialSection, oldSection);CHKERRQ(ierr);
  ierr = MeshGetMesh(parallelMesh, m);CHKERRQ(ierr);
  ierr = SectionPairCreate(oldSection->comm(), parallelSection);CHKERRQ(ierr);
  ALE::Obj<ALE::Mesh::pair_section_type> newSection = ALE::New::Distribution<ALE::Mesh::topology_type>::distributeSection(oldSection, m->getTopology(), m->getTopology()->getDistSendOverlap(), m->getTopology()->getDistRecvOverlap());
  ierr = SectionPairSetSection(*parallelSection, newSection);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SectionPairRestrict"
/*@C
  SectionPairRestrict - Restricts the SectionPair to a subset of the topology, returning an array of values.

  Not collective

  Input Parameters:
+ section - the section object
- point - the Sieve point

  Output Parameter:
. values - The values associated with the submesh

  Level: advanced

.seealso SectionUpdate(), SectionCreate(), SectionView()
@*/
PetscErrorCode PETSCDM_DLLEXPORT SectionPairRestrict(SectionPair section, PetscInt point, PetscPair *values[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(section, SECTIONPAIR_COOKIE, 1);
  PetscValidPointer(values,3);
  *values = (PetscPair *) section->s->restrict(0, point);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SectionPairUpdate"
/*@C
  SectionPairUpdate - Updates the array of values associated to a subset of the topology in this Section.

  Not collective

  Input Parameters:
+ section - the section object
. point - the Sieve point
- values - The values associated with the submesh

  Level: advanced

.seealso SectionPairRestrict(), SectionPairCreate(), SectionPairView()
@*/
PetscErrorCode PETSCDM_DLLEXPORT SectionPairUpdate(SectionPair section, PetscInt point, const PetscPair values[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(section, SECTIONPAIR_COOKIE, 1);
  PetscValidPointer(values,3);
  section->s->update(0, point, (ALE::pair<int, ALE::Mesh::split_value> *) values);
  PetscFunctionReturn(0);
}
