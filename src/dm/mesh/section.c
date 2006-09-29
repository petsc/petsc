 
#include "src/dm/mesh/meshimpl.h"   /*I      "petscmesh.h"   I*/
#include "src/dm/mesh/meshvtk.h"

/* Logging support */
PetscCookie PETSCDM_DLLEXPORT SECTION_COOKIE = 0;
PetscEvent  Section_View = 0;

#undef __FUNCT__  
#define __FUNCT__ "SectionView_Sieve_Ascii"
PetscErrorCode SectionView_Sieve_Ascii(Section section, PetscViewer viewer)
{
  // state 0: No header has been output
  // state 1: Only POINT_DATA has been output
  // state 2: Only CELL_DATA has been output
  // state 3: Output both, POINT_DATA last
  // state 4: Output both, CELL_DATA last
  ALE::Obj<ALE::Mesh::section_type>  s;
  ALE::Obj<ALE::Mesh::topology_type> topology;
  PetscViewerFormat format;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = PetscViewerGetFormat(viewer, &format);CHKERRQ(ierr);
  ierr = SectionGetSection(section, s);CHKERRQ(ierr);
  ierr = SectionGetTopology(section, topology);CHKERRQ(ierr);
  if (format == PETSC_VIEWER_ASCII_VTK || format == PETSC_VIEWER_ASCII_VTK_CELL) {
    static PetscInt   stateId     = -1;
    PetscInt          doOutput    = 0;
    PetscInt          outputState = 0;
    PetscInt          fiberDim    = 0;
    PetscTruth        hasState;
    const char       *name;

    ierr = PetscObjectGetName((PetscObject) section, &name);CHKERRQ(ierr);
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
      const ALE::Mesh::section_type::patch_type  patch     = topology->getPatches().begin()->first;
      const ALE::Obj<ALE::Mesh::numbering_type>& numbering = ALE::Mesh::NumberingFactory::singleton(s->debug())->getNumbering(topology, patch, 0);

      if (doOutput) {
        fiberDim = s->getFiberDimension(patch, *topology->depthStratum(patch, 0)->begin());
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
      ALE::Mesh::section_type::patch_type patch = topology->getPatches().begin()->first;
      const ALE::Obj<ALE::Mesh::numbering_type>& numbering = ALE::Mesh::NumberingFactory::singleton(s->debug())->getNumbering(topology, patch, topology->depth());

      if (doOutput) {
        fiberDim = s->getFiberDimension(patch, *topology->heightStratum(patch, 0)->begin());
        ierr = PetscViewerASCIIPrintf(viewer, "CELL_DATA %d\n", numbering->getGlobalSize());CHKERRQ(ierr);
      }
      VTKViewer::writeField(s, std::string(name), fiberDim, numbering, viewer);
    }
    ierr = PetscObjectComposedDataSetInt((PetscObject) viewer, stateId, outputState);CHKERRQ(ierr);
  } else {
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SectionView_Sieve"
PetscErrorCode SectionView_Sieve(Section section, PetscViewer viewer)
{
  PetscTruth     iascii, isbinary, isdraw;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscTypeCompare((PetscObject) viewer, PETSC_VIEWER_ASCII, &iascii);CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject) viewer, PETSC_VIEWER_BINARY, &isbinary);CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject) viewer, PETSC_VIEWER_DRAW, &isdraw);CHKERRQ(ierr);

  if (iascii){
    ierr = SectionView_Sieve_Ascii(section, viewer);CHKERRQ(ierr);
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
#define __FUNCT__ "SectionView"
/*@C
   SectionView - Views a Section object. 

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
PetscErrorCode SectionView(Section section, PetscViewer viewer)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(section, SECTION_COOKIE, 1);
  PetscValidType(section, 1);
  if (!viewer) viewer = PETSC_VIEWER_STDOUT_(section->comm);
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_COOKIE, 2);
  PetscCheckSameComm(section, 1, viewer, 2);

  ierr = PetscLogEventBegin(Section_View,0,0,0,0);CHKERRQ(ierr);
  ierr = (*section->ops->view)(section, viewer);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(Section_View,0,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SectionGetSection"
/*@C
  SectionGetSection - Gets the internal section object

  Not collective

  Input Parameter:
. section - the section object

  Output Parameter:
. s - the internal section object
 
  Level: advanced

.seealso SectionCreate(), SectionSetSection()
@*/
PetscErrorCode PETSCDM_DLLEXPORT SectionGetSection(Section section, ALE::Obj<ALE::Mesh::section_type>& s)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(section, SECTION_COOKIE, 1);
  s = section->s;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SectionSetSection"
/*@C
  SectionSetSection - Sets the internal section object

  Not collective

  Input Parameters:
+ section - the section object
- s - the internal section object
 
  Level: advanced

.seealso SectionCreate(), SectionGetSection()
@*/
PetscErrorCode PETSCDM_DLLEXPORT SectionSetSection(Section section, const ALE::Obj<ALE::Mesh::section_type>& s)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(section, SECTION_COOKIE, 1);
  section->s = s;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SectionGetTopology"
/*@C
  SectionGetTopology - Gets the internal section topology

  Not collective

  Input Parameter:
. section - the section object

  Output Parameter:
. t - the internal section topology
 
  Level: advanced

.seealso SectionCreate(), SectionGetSection(), SectionSetSection()
@*/
PetscErrorCode PETSCDM_DLLEXPORT SectionGetTopology(Section section, ALE::Obj<ALE::Mesh::topology_type>& t)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(section, SECTION_COOKIE, 1);
  t = section->s->getTopology();
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SectionSetTopology"
/*@C
  SectionSetTopology - Sets the internal section topology

  Not collective

  Input Parameters:
+ section - the section object
- t - the internal section topology
 
  Level: advanced

.seealso SectionCreate(), SectionGetSection(), SectionSetSection()
@*/
PetscErrorCode PETSCDM_DLLEXPORT SectionSetTopology(Section section, const ALE::Obj<ALE::Mesh::topology_type>& t)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(section, SECTION_COOKIE, 1);
  section->s->setTopology(t);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SectionCreate"
/*@C
  SectionCreate - Creates a Section object, used to manage data for an unstructured problem
  described by a Sieve.

  Collective on MPI_Comm

  Input Parameter:
. comm - the processors that will share the global section

  Output Parameters:
. section - the section object

  Level: advanced

.seealso SectionDestroy(), SectionView()
@*/
PetscErrorCode PETSCDM_DLLEXPORT SectionCreate(MPI_Comm comm, Section *section)
{
  PetscErrorCode ierr;
  Section        s;

  PetscFunctionBegin;
  PetscValidPointer(section,2);
  *section = PETSC_NULL;
#ifndef PETSC_USE_DYNAMIC_LIBRARIES
  ierr = DMInitializePackage(PETSC_NULL);CHKERRQ(ierr);
#endif

  ierr = PetscHeaderCreate(s,_p_Section,struct _SectionOps,SECTION_COOKIE,0,"Section",comm,SectionDestroy,0);CHKERRQ(ierr);
  s->ops->view     = SectionView_Sieve;
  s->ops->restrict = SectionRestrict;
  s->ops->update   = SectionUpdate;

  ierr = PetscObjectChangeTypeName((PetscObject) s, "sieve");CHKERRQ(ierr);

  s->s             = new ALE::Mesh::section_type(comm);
  *section = s;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SectionDestroy"
/*@C
  SectionDestroy - Destroys a section.

  Collective on Section

  Input Parameter:
. section - the section object

  Level: advanced

.seealso SectionCreate(), SectionView()
@*/
PetscErrorCode PETSCDM_DLLEXPORT SectionDestroy(Section section)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(section, SECTION_COOKIE, 1);
  if (--section->refct > 0) PetscFunctionReturn(0);
  section->s = PETSC_NULL;
  ierr = PetscHeaderDestroy(section);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SectionRestrict"
/*@C
  SectionRestrict - Restricts the Section to a subset of the topology, returning an array of values.

  Not collective

  Input Parameters:
+ section - the section object
- point - the Sieve point

  Output Parameter:
. values - The values associated with the submesh

  Level: advanced

.seealso SectionUpdate(), SectionCreate(), SectionView()
@*/
PetscErrorCode PETSCDM_DLLEXPORT SectionRestrict(Section section, PetscInt point, PetscScalar *values[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(section, SECTION_COOKIE, 1);
  PetscValidScalarPointer(values,3);
  *values = (PetscScalar *) section->s->restrict(0, point);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SectionUpdate"
/*@C
  SectionUpdate - Updates the array of values associated to a subset of the topology in this Section.

  Not collective

  Input Parameters:
+ section - the section object
. point - the Sieve point
- values - The values associated with the submesh

  Level: advanced

.seealso SectionRestrict(), SectionCreate(), SectionView()
@*/
PetscErrorCode PETSCDM_DLLEXPORT SectionUpdate(Section section, PetscInt point, const PetscScalar values[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(section, SECTION_COOKIE, 1);
  PetscValidScalarPointer(values,3);
  section->s->update(0, point, values);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MeshGetVertexSection"
/*@C
  MeshGetVertexSection - Create a Section over the vertices with the specified fiber dimension

  Collective on Mesh

  Input Parameters:
+ mesh - The Mesh object
- fiberDim - The section name

  Output Parameter:
. section - The section

  Level: intermediate

.keywords: mesh, section, vertex
.seealso: MeshCreate(), SectionCreate()
@*/
PetscErrorCode MeshGetVertexSection(Mesh mesh, PetscInt fiberDim, Section *section)
{
  ALE::Obj<ALE::Mesh> m;
  ALE::Obj<ALE::Mesh::section_type> s;
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  ierr = MeshGetMesh(mesh, m);CHKERRQ(ierr);
  ierr = SectionCreate(m->comm(), section);CHKERRQ(ierr);
  ierr = SectionSetTopology(*section, m->getTopology());CHKERRQ(ierr);
  ierr = SectionGetSection(*section, s);CHKERRQ(ierr);
  s->setFiberDimensionByDepth(0, 0, fiberDim);
  s->allocate();
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MeshGetCellSection"
/*@C
  MeshGetCellSection - Create a Section over the cells with the specified fiber dimension

  Collective on Mesh

  Input Parameters:
+ mesh - The Mesh object
- fiberDim - The section name

  Output Parameter:
. section - The section

  Level: intermediate

.keywords: mesh, section, cell
.seealso: MeshCreate(), SectionCreate()
@*/
PetscErrorCode MeshGetCellSection(Mesh mesh, PetscInt fiberDim, Section *section)
{
  ALE::Obj<ALE::Mesh> m;
  ALE::Obj<ALE::Mesh::section_type> s;
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  ierr = MeshGetMesh(mesh, m);CHKERRQ(ierr);
  ierr = SectionCreate(m->comm(), section);CHKERRQ(ierr);
  ierr = SectionSetTopology(*section, m->getTopology());CHKERRQ(ierr);
  ierr = SectionGetSection(*section, s);CHKERRQ(ierr);
  s->setFiberDimensionByHeight(0, 0, fiberDim);
  s->allocate();
  PetscFunctionReturn(0);
}
