#include <petsc-private/meshimpl.h>   /*I      "petscdmmesh.h"   I*/
#include <petscdmmesh_viewers.hh>

/* Logging support */
PetscClassId  SECTIONREAL_CLASSID;
PetscLogEvent  SectionReal_View;
PetscClassId  SECTIONINT_CLASSID;
PetscLogEvent  SectionInt_View;
PetscClassId  SECTIONPAIR_CLASSID;
PetscLogEvent  SectionPair_View;

#undef __FUNCT__
#define __FUNCT__ "SectionRealView_Sieve"
PetscErrorCode SectionRealView_Sieve(SectionReal section, PetscViewer viewer)
{
  PetscBool      iascii, isbinary, isdraw;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscTypeCompare((PetscObject) viewer, PETSCVIEWERASCII, &iascii);CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject) viewer, PETSCVIEWERBINARY, &isbinary);CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject) viewer, PETSCVIEWERDRAW, &isdraw);CHKERRQ(ierr);

  if (iascii){
    ALE::Obj<PETSC_MESH_TYPE::real_section_type> s;
    ALE::Obj<PETSC_MESH_TYPE>                    b;
    const char                                   *name;

    ierr = SectionRealGetSection(section, s);CHKERRQ(ierr);
    ierr = SectionRealGetBundle(section, b);CHKERRQ(ierr);
    ierr = PetscObjectGetName((PetscObject) section, &name);CHKERRQ(ierr);
    ierr = SectionView_Sieve_Ascii(b, s, name, viewer);CHKERRQ(ierr);
  } else if (isbinary) {
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP, "Binary viewer not implemented for Section");
  } else if (isdraw){ 
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP, "Draw viewer not implemented for Section");
  } else {
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_SUP,"Viewer type %s not supported by this section object", ((PetscObject)viewer)->type_name);
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
+    PETSC_VIEWER_DEFAULT - default, prints section information
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
  PetscValidHeaderSpecific(section, SECTIONREAL_CLASSID, 1);
  PetscValidType(section, 1);
  if (!viewer) {
    ierr = PetscViewerASCIIGetStdout(((PetscObject)section)->comm,&viewer);CHKERRQ(ierr);
  }
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 2);
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
PetscErrorCode  SectionRealDuplicate(SectionReal section, SectionReal *newSection)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(section, SECTIONREAL_CLASSID, 1);
  PetscValidPointer(newSection, 2);
  const ALE::Obj<PETSC_MESH_TYPE::real_section_type>& s = section->s;
  ALE::Obj<PETSC_MESH_TYPE::real_section_type>        t = new PETSC_MESH_TYPE::real_section_type(s->comm(), s->debug());

  t->setAtlas(s->getAtlas());
  t->allocateStorage();
  t->copyBC(s);
  ierr = SectionRealCreate(s->comm(), newSection);CHKERRQ(ierr);
  ierr = SectionRealSetSection(*newSection, t);CHKERRQ(ierr);
  ierr = SectionRealSetBundle(*newSection, section->b);CHKERRQ(ierr);
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
PetscErrorCode  SectionRealGetSection(SectionReal section, ALE::Obj<PETSC_MESH_TYPE::real_section_type>& s)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(section, SECTIONREAL_CLASSID, 1);
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
PetscErrorCode  SectionRealSetSection(SectionReal section, const ALE::Obj<PETSC_MESH_TYPE::real_section_type>& s)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(section, SECTIONREAL_CLASSID, 1);
  if (!s.isNull()) {ierr = PetscObjectSetName((PetscObject) section, s->getName().c_str());CHKERRQ(ierr);}
  section->s = s;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SectionRealGetBundle"
/*@C
  SectionRealGetBundle - Gets the section bundle

  Not collective

  Input Parameter:
. section - the section object

  Output Parameter:
. b - the section bundle
 
  Level: advanced

.seealso SectionRealCreate(), SectionRealGetSection(), SectionRealSetSection()
@*/
PetscErrorCode  SectionRealGetBundle(SectionReal section, ALE::Obj<PETSC_MESH_TYPE>& b)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(section, SECTIONREAL_CLASSID, 1);
  b = section->b;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SectionRealSetBundle"
/*@C
  SectionRealSetBundle - Sets the section bundle

  Not collective

  Input Parameters:
+ section - the section object
- b - the section bundle
 
  Level: advanced

.seealso SectionRealCreate(), SectionRealGetSection(), SectionRealSetSection()
@*/
PetscErrorCode  SectionRealSetBundle(SectionReal section, const ALE::Obj<PETSC_MESH_TYPE>& b)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(section, SECTIONREAL_CLASSID, 1);
  section->b = b;
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
PetscErrorCode  SectionRealCreate(MPI_Comm comm, SectionReal *section)
{
  PetscErrorCode ierr;
  SectionReal    s;

  PetscFunctionBegin;
  PetscValidPointer(section,2);
  *section = PETSC_NULL;

  ierr = PetscHeaderCreate(s,_p_SectionReal,struct _SectionRealOps,SECTIONREAL_CLASSID,0,"SectionReal","Section","DM",comm,SectionRealDestroy,0);CHKERRQ(ierr);
  s->ops->view     = SectionRealView_Sieve;
  s->ops->restrictClosure = SectionRealRestrict;
  s->ops->update   = SectionRealUpdate;

  ierr = PetscObjectChangeTypeName((PetscObject) s, "sieve");CHKERRQ(ierr);

  new(&s->s) ALE::Obj<PETSC_MESH_TYPE::real_section_type>(PETSC_MESH_TYPE::real_section_type(comm));
  new(&s->b) ALE::Obj<PETSC_MESH_TYPE>(PETSC_NULL);
  *section = s;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SectionRealDestroy"
/*@
  SectionRealDestroy - Destroys a section.

  Collective on Section

  Input Parameter:
. section - the section object

  Level: advanced

.seealso SectionRealCreate(), SectionRealView()
@*/
PetscErrorCode  SectionRealDestroy(SectionReal *section)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!*section) PetscFunctionReturn(0);
  PetscValidHeaderSpecific(*section, SECTIONREAL_CLASSID, 1);
  if (--((PetscObject)(*section))->refct > 0) PetscFunctionReturn(0);
  ierr = PetscHeaderDestroy(section);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SectionRealDistribute"
/*@
  SectionRealDistribute - Distributes the sections.

  Not Collective

  Input Parameters:
+ serialSection - The original Section object
- parallelMesh - The parallel DMMesh

  Output Parameter:
. parallelSection - The distributed Section object

  Level: intermediate

.keywords: mesh, section, distribute
.seealso: DMMeshCreate()
@*/
PetscErrorCode SectionRealDistribute(SectionReal serialSection, DM parallelMesh, SectionReal *parallelSection)
{
  ALE::Obj<PETSC_MESH_TYPE::real_section_type> oldSection;
  ALE::Obj<PETSC_MESH_TYPE>               m;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = SectionRealGetSection(serialSection, oldSection);CHKERRQ(ierr);
  ierr = DMMeshGetMesh(parallelMesh, m);CHKERRQ(ierr);
  ierr = SectionRealCreate(oldSection->comm(), parallelSection);CHKERRQ(ierr);
#ifdef PETSC_OPT_SIEVE
  ALE::Obj<PETSC_MESH_TYPE::real_section_type> newSection;

  // We assume all integer sections are complete sections
  newSection->setName(oldSection->getName());
  newSection->setChart(m->getSieve()->getChart());
  //distributeSection(oldSection, partition, m->getRenumbering(), m->getDistSendOverlap(), m->getDistRecvOverlap(), newSection);
  ierr = SectionRealSetSection(*parallelSection, newSection);CHKERRQ(ierr);
  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP, "Not working because the partition is unavailable");
#else
  ALE::Obj<PETSC_MESH_TYPE::real_section_type> newSection = ALE::Distribution<PETSC_MESH_TYPE>::distributeSection(oldSection, m, m->getDistSendOverlap(), m->getDistRecvOverlap());
  ierr = SectionRealSetSection(*parallelSection, newSection);CHKERRQ(ierr);
#endif
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
PetscErrorCode  SectionRealRestrict(SectionReal section, PetscInt point, PetscScalar *values[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(section, SECTIONREAL_CLASSID, 1);
  PetscValidScalarPointer(values,3);
  try {
    *values = (PetscScalar *) section->s->restrictPoint(point);
  } catch(ALE::Exception e) {
    SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Invalid argument: %s", e.message());
  }
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
. values - The values associated with the submesh
- mode - The insertion mode

  Level: advanced

.seealso SectionRealRestrict(), SectionRealCreate(), SectionRealView()
@*/
PetscErrorCode  SectionRealUpdate(SectionReal section, PetscInt point, const PetscScalar values[], InsertMode mode)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(section, SECTIONREAL_CLASSID, 1);
  PetscValidScalarPointer(values,3);
#ifdef PETSC_USE_COMPLEX
  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP, "SectionReal does not support complex updates");
#else
  try {
    if (mode == INSERT_VALUES) {
      section->b->update(section->s, point, values);
    } else if (mode == ADD_VALUES) {
      section->b->updateAdd(section->s, point, values);
    } else {
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG, "Invalid insertion mode: %d", mode);
    }
  } catch(ALE::Exception e) {
    SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Invalid argument: %s", e.message());
  }
#endif
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SectionRealRestrictClosure"
/*@C
  SectionRealRestrictClosure - Returns an array with the values in a given closure

  Not Collective

  Input Parameters:
+ section - The section
. mesh    - The DMMesh object
- point   - The sieve point

  Output Parameter:
. array - The array full of values in the closure

  Level: intermediate

.keywords: mesh, elements
.seealso: DMMeshCreate()
@*/
PetscErrorCode SectionRealRestrictClosure(SectionReal section, DM dm, PetscInt point, const PetscScalar *values[])
{
  ALE::Obj<PETSC_MESH_TYPE> m;
  ALE::Obj<PETSC_MESH_TYPE::real_section_type> s;
  PetscErrorCode            ierr;

  PetscFunctionBegin;
  ierr = DMMeshGetMesh(dm, m);CHKERRQ(ierr);
  ierr = SectionRealGetSection(section, s);CHKERRQ(ierr);
#ifdef PETSC_USE_COMPLEX
  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP, "SectionReal does not support complex restriction");
#else
  try {
    *values = m->restrictClosure(s, point);
  } catch(ALE::Exception e) {
    SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Invalid argument: %s", e.message());
  }
#endif
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SectionRealRestrictClosure"
/*@C
  SectionRealRestrictClosure - Returns an array with the values in a given closure

  Not Collective

  Input Parameters:
+ section - The section
. mesh    - The DMMesh object
. point   - The sieve point
. n       - The array size
- array   - The array to fill up

  Output Parameter:
. array - The array full of values in the closure

  Level: intermediate

.keywords: mesh, elements
.seealso: DMMeshCreate()
@*/
PetscErrorCode SectionRealRestrictClosure(SectionReal section, DM dm, PetscInt point, PetscInt n, PetscScalar values[])
{
  ALE::Obj<PETSC_MESH_TYPE> m;
  ALE::Obj<PETSC_MESH_TYPE::real_section_type> s;
  PetscErrorCode            ierr;

  PetscFunctionBegin;
  ierr = DMMeshGetMesh(dm, m);CHKERRQ(ierr);
  ierr = SectionRealGetSection(section, s);CHKERRQ(ierr);
#ifdef PETSC_USE_COMPLEX
  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP, "SectionReal does not support complex restriction");
#else
  try {
    m->restrictClosure(s, point, values, n);
  } catch(ALE::Exception e) {
    SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Invalid argument: %s", e.message());
  }
#endif
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SectionRealUpdateClosure"
/*@C
  SectionRealUpdateClosure - Updates the values in a given closure from the array

  Not Collective

  Input Parameters:
+ section - The section
. mesh    - The DMMesh object
. point   - The sieve point
. array   - The array to fill up
- mode    - The insertion mode

  Output Parameter:
. array - The array full of values in the closure

  Level: intermediate

.keywords: mesh, elements
.seealso: DMMeshCreate()
@*/
PetscErrorCode SectionRealUpdateClosure(SectionReal section, DM dm, PetscInt point, PetscScalar values[], InsertMode mode)
{
  ALE::Obj<PETSC_MESH_TYPE> m;
  ALE::Obj<PETSC_MESH_TYPE::real_section_type> s;
  PetscErrorCode            ierr;

  PetscFunctionBegin;
  ierr = DMMeshGetMesh(dm, m);CHKERRQ(ierr);
  ierr = SectionRealGetSection(section, s);CHKERRQ(ierr);
#ifdef PETSC_USE_COMPLEX
  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP, "SectionReal does not support complex update");
#else
  try {
    if (mode == INSERT_VALUES) {
      m->update(s, point, values);
    } else if (mode == ADD_VALUES) {
      m->updateAdd(s, point, values);
    } else {
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG, "Invalid insertion mode: %d", mode);
    }
  } catch(ALE::Exception e) {
    SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Invalid argument: %s", e.message());
  }
#endif
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SectionRealComplete"
/*@
  SectionRealComplete - Exchanges data across the mesh overlap.

  Not collective

  Input Parameter:
. section - the section object

  Level: advanced

.seealso SectionRealRestrict(), SectionRealCreate(), SectionRealView()
@*/
PetscErrorCode SectionRealComplete(SectionReal section)
{
  Obj<PETSC_MESH_TYPE::real_section_type> s;
  Obj<PETSC_MESH_TYPE>                    b;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = SectionRealGetSection(section, s);CHKERRQ(ierr);
  ierr = SectionRealGetBundle(section, b);CHKERRQ(ierr);
#if 0
  ALE::Distribution<PETSC_MESH_TYPE>::completeSection(b, s);
#else
  ALE::Completion::completeSectionAdd(b->getSendOverlap(), b->getRecvOverlap(), s, s);
#endif
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SectionRealZero"
/*@
  SectionRealZero - Zero out the entries

  Not collective

  Input Parameter:
. section - the section object

  Level: advanced

.seealso SectionRealRestrict(), SectionRealCreate(), SectionRealView()
@*/
PetscErrorCode SectionRealZero(SectionReal section)
{
  Obj<PETSC_MESH_TYPE::real_section_type> s;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = SectionRealGetSection(section, s);CHKERRQ(ierr);
  s->zero();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SectionRealGetFiberDimension"
/*@
  SectionRealGetFiberDimension - Get the size of the vector space attached to the point

  Not collective

  Input Parameters:
+ section - the section object
- point - the Sieve point

  Output Parameters:
. size - The fiber dimension

  Level: advanced

.seealso SectionRealRestrict(), SectionRealCreate(), SectionRealView()
@*/
PetscErrorCode  SectionRealGetFiberDimension(SectionReal section, PetscInt point, PetscInt *size)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(section, SECTIONREAL_CLASSID, 1);
  *size = section->s->getFiberDimension(point);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SectionRealSetFiberDimension"
/*@
  SectionRealSetFiberDimension - Set the size of the vector space attached to the point

  Not collective

  Input Parameters:
+ section - the section object
. point - the Sieve point
- size - The fiber dimension

  Level: advanced

.seealso SectionRealSetFiberDimensionField(), SectionRealRestrict(), SectionRealCreate(), SectionRealView()
@*/
PetscErrorCode  SectionRealSetFiberDimension(SectionReal section, PetscInt point, const PetscInt size)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(section, SECTIONREAL_CLASSID, 1);
  section->s->setFiberDimension(point, size);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SectionRealSetFiberDimensionField"
/*@
  SectionRealSetFiberDimensionField - Set the size of the vector space attached to the point for a given field

  Not collective

  Input Parameters:
+ section - the section object
. point - the Sieve point
. size - The fiber dimension
- field - The field number

  Level: advanced

.seealso SectionRealSetFiberDimension(), SectionRealRestrict(), SectionRealCreate(), SectionRealView()
@*/
PetscErrorCode  SectionRealSetFiberDimensionField(SectionReal section, PetscInt point, const PetscInt size, const PetscInt field)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(section, SECTIONREAL_CLASSID, 1);
  section->s->setFiberDimension(point, size, field);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SectionRealGetSize"
/*@
  SectionRealGetSize - Gets the number of local dofs in this Section

  Not collective

  Input Parameter:
. section - the section object

  Output Parameter:
. size - the section size

  Level: advanced

.seealso SectionRealRestrict(), SectionRealCreate(), SectionRealView()
@*/
PetscErrorCode SectionRealGetSize(SectionReal section, PetscInt *size)
{
  Obj<PETSC_MESH_TYPE::real_section_type> s;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidIntPointer(size,2);
  ierr = SectionRealGetSection(section, s);CHKERRQ(ierr);
  *size = s->size();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SectionRealAllocate"
/*@
  SectionRealAllocate - Allocate storage for this section

  Not collective

  Input Parameter:
. section - the section object

  Level: advanced

.seealso SectionRealRestrict(), SectionRealCreate(), SectionRealView()
@*/
PetscErrorCode  SectionRealAllocate(SectionReal section)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(section, SECTIONREAL_CLASSID, 1);
  section->b->allocate(section->s);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SectionRealCreateLocalVector"
/*@
  SectionRealCreateLocalVector - Creates a vector with the local piece of the Section

  Collective on DMMesh

  Input Parameter:
. section - the Section  

  Output Parameter:
. localVec - the local vector

  Level: advanced

  Notes: The vector can safely be destroyed using VecDestroy().
.seealso DMMeshDestroy(), DMMeshCreate()
@*/
PetscErrorCode  SectionRealCreateLocalVector(SectionReal section, Vec *localVec)
{
  ALE::Obj<PETSC_MESH_TYPE::real_section_type> s;
  PetscErrorCode ierr;

  PetscFunctionBegin;
#ifdef PETSC_USE_COMPLEX
  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP, "SectionReal does not support complex Vec");
#else
  ierr = SectionRealGetSection(section, s);CHKERRQ(ierr);
  ierr = VecCreateSeqWithArray(PETSC_COMM_SELF,1, s->getStorageSize(), s->restrictSpace(), localVec);CHKERRQ(ierr);
#endif
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SectionRealAddSpace"
/*@
  SectionRealAddSpace - Add another field to this section

  Collective on DMMesh

  Input Parameter:
. section - the Section

  Level: advanced

.seealso SectionRealCreate(), SectionRealGetFibration()
@*/
PetscErrorCode  SectionRealAddSpace(SectionReal section)
{
  ALE::Obj<PETSC_MESH_TYPE::real_section_type> s;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = SectionRealGetSection(section, s);CHKERRQ(ierr);
  s->addSpace();
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SectionRealGetFibration"
/*@
  SectionRealGetFibration - Creates a section for only the data associated with the given field

  Collective on DMMesh

  Input Parameter:
+ section - the Section
- field- The field number

  Output Parameter:
. subsection - the section of the given field

  Level: advanced

.seealso SectionRealCreate()
@*/
PetscErrorCode  SectionRealGetFibration(SectionReal section, const PetscInt field, SectionReal *subsection)
{
  ALE::Obj<PETSC_MESH_TYPE>                    b;
  ALE::Obj<PETSC_MESH_TYPE::real_section_type> s;
  ALE::Obj<PETSC_MESH_TYPE::real_section_type> t;
  MPI_Comm       comm;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject) section, &comm);CHKERRQ(ierr);
  ierr = SectionRealGetBundle(section, b);CHKERRQ(ierr);
  ierr = SectionRealGetSection(section, s);CHKERRQ(ierr);
  ierr = SectionRealCreate(comm, subsection);CHKERRQ(ierr);
  ierr = SectionRealSetBundle(*subsection, b);CHKERRQ(ierr);
  t    = s->getFibration(field);
  ierr = SectionRealSetSection(*subsection, t);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SectionRealToVec"
/*@C
  SectionRealToVec - Maps the given section to a Vec

  Collective on Section

  Input Parameters:
+ section - the real Section
- mesh - The DMMesh

  Output Parameter:
. vec - the Vec

  Level: intermediate

.seealso VecCreate(), SectionRealCreate()
@*/
PetscErrorCode  SectionRealToVec(SectionReal section, DM dm, ScatterMode mode, Vec vec)
{
  Vec            localVec;
  VecScatter     scatter;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(section, SECTIONREAL_CLASSID, 1);
  ierr = SectionRealCreateLocalVector(section, &localVec);CHKERRQ(ierr);
  ierr = DMMeshGetGlobalScatter(dm, &scatter);CHKERRQ(ierr);
  if (mode == SCATTER_FORWARD) {
    ierr = VecScatterBegin(scatter, localVec, vec, INSERT_VALUES, mode);CHKERRQ(ierr);
    ierr = VecScatterEnd(scatter, localVec, vec, INSERT_VALUES, mode);CHKERRQ(ierr);
  } else {
    ierr = VecScatterBegin(scatter, vec, localVec, INSERT_VALUES, mode);CHKERRQ(ierr);
    ierr = VecScatterEnd(scatter, vec, localVec, INSERT_VALUES, mode);CHKERRQ(ierr);
  }
  ierr = VecDestroy(&localVec);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SectionRealToVec"
/*@
  SectionRealToVec - Map between unassembled local Section storage and a globally assembled Vec

  Collective on VecScatter

  Input Parameters:
+ section - the Section
. scatter - the scatter from the Section to the Vec
. mode - the mode, SCATTER_FORWARD (Section to Vec) or SCATTER_REVERSE (Vec to Section)
- vec - the Vec

  Level: advanced

.seealso SectionRealRestrict(), SectionRealCreate(), SectionRealView()
@*/
PetscErrorCode  SectionRealToVec(SectionReal section, VecScatter scatter, ScatterMode mode, Vec vec)
{
  Vec            localVec;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(section, SECTIONREAL_CLASSID, 1);
  ierr = SectionRealCreateLocalVector(section, &localVec);CHKERRQ(ierr);
  if (mode == SCATTER_FORWARD) {
    ierr = VecScatterBegin(scatter, localVec, vec, INSERT_VALUES, mode);CHKERRQ(ierr);
    ierr = VecScatterEnd(scatter, localVec, vec, INSERT_VALUES, mode);CHKERRQ(ierr);
  } else {
    ierr = VecScatterBegin(scatter, vec, localVec, INSERT_VALUES, mode);CHKERRQ(ierr);
    ierr = VecScatterEnd(scatter, vec, localVec, INSERT_VALUES, mode);CHKERRQ(ierr);
  }
  ierr = VecDestroy(&localVec);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SectionRealClear"
/*@
  SectionRealClear - Dellocate storage for this section

  Not collective

  Input Parameter:
. section - the section object

  Level: advanced

.seealso SectionRealRestrict(), SectionRealCreate(), SectionRealView()
@*/
PetscErrorCode  SectionRealClear(SectionReal section)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(section, SECTIONREAL_CLASSID, 1);
  section->s->clear();
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SectionRealSet"
/*@
  SectionRealSet - Sets all the values to the given value

  Not collective

  Input Parameters:
+ section - the real Section
- val - the value

  Level: intermediate

.seealso VecNorm(), SectionRealCreate()
@*/
PetscErrorCode  SectionRealSet(SectionReal section, PetscReal val)
{
  Obj<PETSC_MESH_TYPE::real_section_type> s;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = SectionRealGetSection(section, s);CHKERRQ(ierr);
  s->set(val);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SectionRealNorm"
/*@C
  SectionRealNorm - Computes the vector norm.

  Collective on Section

  Input Parameters:
+  section - the real Section
-  type - one of NORM_1, NORM_2, NORM_INFINITY.  Also available
          NORM_1_AND_2, which computes both norms and stores them
          in a two element array.

  Output Parameter:
. val - the norm

  Notes:
$     NORM_1 denotes sum_i |x_i|
$     NORM_2 denotes sqrt(sum_i (x_i)^2)
$     NORM_INFINITY denotes max_i |x_i|

  Level: intermediate

.seealso VecNorm(), SectionRealCreate()
@*/
PetscErrorCode  SectionRealNorm(SectionReal section, DM dm, NormType type, PetscReal *val)
{
  Obj<PETSC_MESH_TYPE> m;
  Obj<PETSC_MESH_TYPE::real_section_type> s;
  Vec            v;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(section, SECTIONREAL_CLASSID, 1);
  ierr = DMMeshGetMesh(dm, m);CHKERRQ(ierr);
  ierr = SectionRealGetSection(section, s);CHKERRQ(ierr);
  const ALE::Obj<PETSC_MESH_TYPE::order_type>& order = m->getFactory()->getGlobalOrder(m, s->getName(), s);
  ierr = VecCreate(m->comm(), &v);CHKERRQ(ierr);
  ierr = VecSetSizes(v, order->getLocalSize(), order->getGlobalSize());CHKERRQ(ierr);
  ierr = VecSetFromOptions(v);CHKERRQ(ierr);
  ierr = SectionRealToVec(section, dm, SCATTER_FORWARD, v);CHKERRQ(ierr);
  ierr = VecNorm(v, type, val);CHKERRQ(ierr);
  ierr = VecDestroy(&v);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SectionRealAXPY"
/*@
  SectionRealAXPY - 

  Collective on Section

  Input Parameters:
+  section - the real Section
.  alpha - a scalar
-  X - the other real Section

  Output Parameter:
. section - the difference

  Level: intermediate

.seealso VecNorm(), SectionRealCreate()
@*/
PetscErrorCode  SectionRealAXPY(SectionReal section, DM dm, PetscScalar alpha, SectionReal X)
{
  Obj<PETSC_MESH_TYPE> m;
  Obj<PETSC_MESH_TYPE::real_section_type> s;
  Obj<PETSC_MESH_TYPE::real_section_type> sX;
  Vec            v, x;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(section, SECTIONREAL_CLASSID, 1);
  ierr = DMMeshGetMesh(dm, m);CHKERRQ(ierr);
  ierr = SectionRealGetSection(section, s);CHKERRQ(ierr);
  ierr = SectionRealGetSection(X, sX);CHKERRQ(ierr);
  const ALE::Obj<PETSC_MESH_TYPE::order_type>& order = m->getFactory()->getGlobalOrder(m, s->getName(), s);
  ierr = VecCreate(m->comm(), &v);CHKERRQ(ierr);
  ierr = VecSetSizes(v, order->getLocalSize(), order->getGlobalSize());CHKERRQ(ierr);
  ierr = VecSetFromOptions(v);CHKERRQ(ierr);
  ierr = VecDuplicate(v, &x);CHKERRQ(ierr);
  ierr = SectionRealToVec(section, dm, SCATTER_FORWARD, v);CHKERRQ(ierr);
  ierr = SectionRealToVec(X,       dm, SCATTER_FORWARD, x);CHKERRQ(ierr);
  ierr = VecAXPY(v, alpha, x);CHKERRQ(ierr);
  ierr = SectionRealToVec(section, dm, SCATTER_REVERSE, v);CHKERRQ(ierr);
  ierr = VecDestroy(&v);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMMeshGetVertexSectionReal"
/*@C
  DMMeshGetVertexSectionReal - Create a Section over the vertices with the specified fiber dimension

  Collective on DMMesh

  Input Parameters:
+ mesh - The DMMesh object
. name - The name of the section
- fiberDim - The number of degrees of freedom per vertex

  Output Parameter:
. section - The section

  Level: intermediate

.keywords: mesh, section, vertex
.seealso: DMMeshCreate(), SectionRealCreate()
@*/
PetscErrorCode DMMeshGetVertexSectionReal(DM dm, const char name[], PetscInt fiberDim, SectionReal *section)
{
  ALE::Obj<PETSC_MESH_TYPE> m;
  ALE::Obj<PETSC_MESH_TYPE::real_section_type> s;
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  ierr = DMMeshGetMesh(dm, m);CHKERRQ(ierr);
  ierr = SectionRealCreate(m->comm(), section);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) *section, name);CHKERRQ(ierr);
  ierr = SectionRealSetBundle(*section, m);CHKERRQ(ierr);
  ierr = SectionRealGetSection(*section, s);CHKERRQ(ierr);
  s->setChart(m->getSieve()->getChart());
  s->setName(name);
  s->setFiberDimension(m->depthStratum(0), fiberDim);
  m->allocate(s);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMMeshGetCellSectionReal"
/*@C
  DMMeshGetCellSectionReal - Create a Section over the cells with the specified fiber dimension

  Collective on DMMesh

  Input Parameters:
+ mesh - The DMMesh object
. name - The name of the section
- fiberDim - The number of degrees of freedom per cell

  Output Parameter:
. section - The section

  Level: intermediate

.keywords: mesh, section, cell
.seealso: DMMeshCreate(), SectionRealCreate(), DMMeshGetVertexSectionReal(), DMMeshCreateSectionRealIS()
@*/
PetscErrorCode DMMeshGetCellSectionReal(DM dm, const char name[], PetscInt fiberDim, SectionReal *section)
{
  ALE::Obj<PETSC_MESH_TYPE> m;
  ALE::Obj<PETSC_MESH_TYPE::real_section_type> s;
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  ierr = DMMeshGetMesh(dm, m);CHKERRQ(ierr);
  ierr = SectionRealCreate(m->comm(), section);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) *section, name);CHKERRQ(ierr);
  ierr = SectionRealSetBundle(*section, m);CHKERRQ(ierr);
  ierr = SectionRealGetSection(*section, s);CHKERRQ(ierr);
  s->setChart(m->getSieve()->getChart());
  s->setName(name);
  s->setFiberDimension(m->heightStratum(0), fiberDim);
  m->allocate(s);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMMeshCreateSectionRealIS"
/*@C
  DMMeshCreateSectionRealIS - Create a Section over the points specified in an IS.
  
  Collective on DMMesh
  
  Input Parameters:
  + dm - the DMMesh object
  . is - The IS describing the points associated with the degrees of freedom
  . name - The name of the section
  - fiberDim - The number of degrees of freedom per point of the IS
  
  Output Parameter:
  . section - The section
  
  Level: intermediate

.keywords: mesh, section, cell
.seealso: DMMeshCreate(), SectionRealCreate(), DMMeshGetVertexSectionReal(), DMMeshGetCellSectionReal()

@*/
PetscErrorCode DMMeshCreateSectionRealIS(DM dm,IS is,const char name[],PetscInt fiberDim,SectionReal *section)
{
  MPI_Comm        comm;
  PetscSection    s;
  PetscInt        pStart,pEnd;
  const PetscInt *points;
  PetscInt        numpoints,p;
  PetscErrorCode  ierr;
  
  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject) dm,&comm);CHKERRQ(ierr);
  
  ierr = PetscSectionCreate(comm,&s);CHKERRQ(ierr);
  ierr = DMMeshGetChart(dm,&pStart,&pEnd);CHKERRQ(ierr);    
  ierr = PetscSectionSetChart(s,pStart,pEnd);CHKERRQ(ierr);
  
  ierr = ISGetLocalSize(is,&numpoints);CHKERRQ(ierr);
  ierr = ISGetIndices(is,&points);CHKERRQ(ierr);
  for (p =0; p < numpoints; p++) {
    ierr = PetscSectionSetDof(s,points[p],fiberDim);CHKERRQ(ierr);
  }
  ierr = ISRestoreIndices(is,&points);CHKERRQ(ierr);
  ierr = DMMeshSetSection(dm,name,s);CHKERRQ(ierr);
  ierr = PetscSectionDestroy(&s);
  ierr = DMMeshGetSectionReal(dm,name,section);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "DMMeshCreateGlobalRealVector"
/*@C
  DMMeshCreateGlobalRealVector - Creates a vector of the correct size to be gathered into by the mesh.

  Collective on DMMesh

  Input Parameters:
+ mesh - the mesh object
- section - The SectionReal

  Output Parameters:
. gvec - the global vector

  Level: advanced

.seealso DMMeshDestroy(), DMMeshCreate(), DMMeshCreateGlobalRealVector()
@*/
PetscErrorCode DMMeshCreateGlobalRealVector(DM dm, SectionReal section, Vec *gvec)
{
  ALE::Obj<PETSC_MESH_TYPE> m;
  ALE::Obj<PETSC_MESH_TYPE::real_section_type> s;
  const char    *name;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMMeshGetMesh(dm, m);CHKERRQ(ierr);
  ierr = SectionRealGetSection(section, s);CHKERRQ(ierr);
  ierr = PetscObjectGetName((PetscObject) section, &name);CHKERRQ(ierr);
  const ALE::Obj<PETSC_MESH_TYPE::order_type>& order = m->getFactory()->getGlobalOrder(m, name, s);

  ierr = VecCreate(m->comm(), gvec);CHKERRQ(ierr);
  ierr = VecSetSizes(*gvec, order->getLocalSize(), order->getGlobalSize());CHKERRQ(ierr);
  ierr = VecSetFromOptions(*gvec);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SectionIntView_Sieve"
PetscErrorCode SectionIntView_Sieve(SectionInt section, PetscViewer viewer)
{
  PetscBool      iascii, isbinary, isdraw;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscTypeCompare((PetscObject) viewer, PETSCVIEWERASCII, &iascii);CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject) viewer, PETSCVIEWERBINARY, &isbinary);CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject) viewer, PETSCVIEWERDRAW, &isdraw);CHKERRQ(ierr);

  if (iascii){
    ALE::Obj<PETSC_MESH_TYPE::int_section_type> s;
    ALE::Obj<PETSC_MESH_TYPE>                   b;
    const char                                  *name;

    ierr = SectionIntGetSection(section, s);CHKERRQ(ierr);
    ierr = SectionIntGetBundle(section, b);CHKERRQ(ierr);
    ierr = PetscObjectGetName((PetscObject) section, &name);CHKERRQ(ierr);
    ierr = SectionView_Sieve_Ascii(b, s, name, viewer);CHKERRQ(ierr);
  } else if (isbinary) {
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP, "Binary viewer not implemented for Section");
  } else if (isdraw){ 
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP, "Draw viewer not implemented for Section");
  } else {
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_SUP,"Viewer type %s not supported by this section object", ((PetscObject)viewer)->type_name);
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
+    PETSC_VIEWER_DEFAULT - default, prints section information
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
  PetscValidHeaderSpecific(section, SECTIONINT_CLASSID, 1);
  PetscValidType(section, 1);
  if (!viewer) {
    ierr = PetscViewerASCIIGetStdout(((PetscObject)section)->comm,&viewer);CHKERRQ(ierr);
  }
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 2);
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
PetscErrorCode  SectionIntGetSection(SectionInt section, ALE::Obj<PETSC_MESH_TYPE::int_section_type>& s)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(section, SECTIONINT_CLASSID, 1);
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
PetscErrorCode  SectionIntSetSection(SectionInt section, const ALE::Obj<PETSC_MESH_TYPE::int_section_type>& s)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(section, SECTIONINT_CLASSID, 1);
  if (!s.isNull()) {ierr = PetscObjectSetName((PetscObject) section, s->getName().c_str());CHKERRQ(ierr);}
  section->s = s;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SectionIntGetBundle"
/*@C
  SectionIntGetBundle - Gets the section bundle

  Not collective

  Input Parameter:
. section - the section object

  Output Parameter:
. b - the section bundle
 
  Level: advanced

.seealso SectionIntCreate(), SectionIntGetSection(), SectionIntSetSection()
@*/
PetscErrorCode  SectionIntGetBundle(SectionInt section, ALE::Obj<PETSC_MESH_TYPE>& b)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(section, SECTIONINT_CLASSID, 1);
  b = section->b;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SectionIntSetBundle"
/*@C
  SectionIntSetBundle - Sets the section bundle

  Not collective

  Input Parameters:
+ section - the section object
- b - the section bundle
 
  Level: advanced

.seealso SectionIntCreate(), SectionIntGetSection(), SectionIntSetSection()
@*/
PetscErrorCode  SectionIntSetBundle(SectionInt section, const ALE::Obj<PETSC_MESH_TYPE>& b)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(section, SECTIONINT_CLASSID, 1);
  section->b = b;
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
PetscErrorCode  SectionIntCreate(MPI_Comm comm, SectionInt *section)
{
  PetscErrorCode ierr;
  SectionInt    s;

  PetscFunctionBegin;
  PetscValidPointer(section,2);
  *section = PETSC_NULL;

  ierr = PetscHeaderCreate(s,_p_SectionInt,struct _SectionIntOps,SECTIONINT_CLASSID,0,"SectionInt","Section","DM",comm,SectionIntDestroy,0);CHKERRQ(ierr);
  s->ops->view     = SectionIntView_Sieve;
  s->ops->restrictClosure = SectionIntRestrict;
  s->ops->update   = SectionIntUpdate;

  ierr = PetscObjectChangeTypeName((PetscObject) s, "sieve");CHKERRQ(ierr);

  new(&s->s) ALE::Obj<PETSC_MESH_TYPE::int_section_type>(PETSC_MESH_TYPE::int_section_type(comm));
  new(&s->b) ALE::Obj<PETSC_MESH_TYPE>(PETSC_NULL);
  *section = s;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SectionIntDestroy"
/*@
  SectionIntDestroy - Destroys a section.

  Collective on Section

  Input Parameter:
. section - the section object

  Level: advanced

.seealso SectionIntCreate(), SectionIntView()
@*/
PetscErrorCode  SectionIntDestroy(SectionInt *section)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!*section) PetscFunctionReturn(0);
  PetscValidHeaderSpecific(*section, SECTIONINT_CLASSID, 1);
  if (--((PetscObject)(*section))->refct > 0) PetscFunctionReturn(0);
  ierr = PetscHeaderDestroy(section);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SectionIntDistribute"
/*@
  SectionIntDistribute - Distributes the sections.

  Not Collective

  Input Parameters:
+ serialSection - The original Section object
- parallelMesh - The parallel DMMesh

  Output Parameter:
. parallelSection - The distributed Section object

  Level: intermediate

.keywords: mesh, section, distribute
.seealso: DMMeshCreate()
@*/
PetscErrorCode SectionIntDistribute(SectionInt serialSection, DM parallelMesh, SectionInt *parallelSection)
{
  ALE::Obj<PETSC_MESH_TYPE::int_section_type> oldSection;
  ALE::Obj<PETSC_MESH_TYPE> m;
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  ierr = SectionIntGetSection(serialSection, oldSection);CHKERRQ(ierr);
  ierr = DMMeshGetMesh(parallelMesh, m);CHKERRQ(ierr);
  ierr = SectionIntCreate(oldSection->comm(), parallelSection);CHKERRQ(ierr);
#ifdef PETSC_OPT_SIEVE
  ALE::Obj<PETSC_MESH_TYPE::int_section_type> newSection;

  // We assume all integer sections are complete sections
  newSection->setName(oldSection->getName());
  newSection->setChart(m->getSieve()->getChart());
  //distributeSection(oldSection, partition, m->getRenumbering(), m->getDistSendOverlap(), m->getDistRecvOverlap(), newSection);
  ierr = SectionIntSetSection(*parallelSection, newSection);CHKERRQ(ierr);
  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP, "Not working because the partition is unavailable");
#else
  ALE::Obj<PETSC_MESH_TYPE::int_section_type> newSection = ALE::Distribution<PETSC_MESH_TYPE>::distributeSection(oldSection, m, m->getDistSendOverlap(), m->getDistRecvOverlap());
  ierr = SectionIntSetSection(*parallelSection, newSection);CHKERRQ(ierr);
#endif
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
PetscErrorCode  SectionIntRestrict(SectionInt section, PetscInt point, PetscInt *values[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(section, SECTIONINT_CLASSID, 1);
  PetscValidPointer(values,3);
  *values = (PetscInt *) section->b->restrictClosure(section->s, point);
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
. values - The values associated with the submesh
- mode - The insertion mode

  Level: advanced

.seealso SectionIntRestrict(), SectionIntCreate(), SectionIntView()
@*/
PetscErrorCode  SectionIntUpdate(SectionInt section, PetscInt point, const PetscInt values[], InsertMode mode)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(section, SECTIONINT_CLASSID, 1);
  PetscValidIntPointer(values,3);
  if (mode == INSERT_VALUES) {
    section->b->update(section->s, point, values);
  } else if (mode == ADD_VALUES) {
    section->b->updateAdd(section->s, point, values);
  } else {
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG, "Invalid insertion mode: %d", mode);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SectionIntRestrictClosure"
/*@C
  SectionIntRestrictClosure - Returns an array with the values in a given closure

  Not Collective

  Input Parameters:
+ section - The section
. mesh    - The DMMesh object
. point   - The sieve point
. n       - The array size
- array   - The array to fill up

  Output Parameter:
. array - The array full of values in the closure

  Level: intermediate

.keywords: mesh, elements
.seealso: DMMeshCreate()
@*/
PetscErrorCode SectionIntRestrictClosure(SectionInt section, DM dm, PetscInt point, PetscInt n, PetscInt values[])
{
  ALE::Obj<PETSC_MESH_TYPE> m;
  ALE::Obj<PETSC_MESH_TYPE::int_section_type> s;
  PetscErrorCode            ierr;

  PetscFunctionBegin;
  ierr = DMMeshGetMesh(dm, m);CHKERRQ(ierr);
  ierr = SectionIntGetSection(section, s);CHKERRQ(ierr);
  m->restrictClosure(s, point, values, n);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SectionIntUpdateClosure"
/*@C
  SectionIntUpdateClosure - Updates the values in a given closure from the array

  Not Collective

  Input Parameters:
+ section - The section
. mesh    - The DMMesh object
. point   - The sieve point
. array   - The array to fill up
- mode    - The insertion mode

  Output Parameter:
. array - The array full of values in the closure

  Level: intermediate

.keywords: mesh, elements
.seealso: DMMeshCreate()
@*/
PetscErrorCode SectionIntUpdateClosure(SectionInt section, DM dm, PetscInt point, PetscInt values[], InsertMode mode)
{
  ALE::Obj<PETSC_MESH_TYPE> m;
  ALE::Obj<PETSC_MESH_TYPE::int_section_type> s;
  PetscErrorCode            ierr;

  PetscFunctionBegin;
  ierr = DMMeshGetMesh(dm, m);CHKERRQ(ierr);
  ierr = SectionIntGetSection(section, s);CHKERRQ(ierr);
  if (mode == INSERT_VALUES) {
    m->update(s, point, values);
  } else if (mode == ADD_VALUES) {
    m->updateAdd(s, point, values);
  } else {
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG, "Invalid insertion mode: %d", mode);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SectionIntComplete"
/*@
  SectionIntComplete - Exchanges data across the mesh overlap.

  Not collective

  Input Parameter:
. section - the section object

  Level: advanced

.seealso SectionIntRestrict(), SectionIntCreate(), SectionIntView()
@*/
PetscErrorCode SectionIntComplete(SectionInt section)
{
  Obj<PETSC_MESH_TYPE::int_section_type> s;
  Obj<PETSC_MESH_TYPE>                   b;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = SectionIntGetSection(section, s);CHKERRQ(ierr);
  ierr = SectionIntGetBundle(section, b);CHKERRQ(ierr);
#if 0
  ALE::Distribution<PETSC_MESH_TYPE>::completeSection(b, s);
#else
  ALE::Completion::completeSectionAdd(b->getSendOverlap(), b->getRecvOverlap(), s, s);
#endif
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SectionIntZero"
/*@
  SectionIntZero - Zero out the entries

  Not collective

  Input Parameter:
. section - the section object

  Level: advanced

.seealso SectionIntRestrict(), SectionIntCreate(), SectionIntView()
@*/
PetscErrorCode SectionIntZero(SectionInt section)
{
  Obj<PETSC_MESH_TYPE::int_section_type> s;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = SectionIntGetSection(section, s);CHKERRQ(ierr);
  s->zero();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SectionIntGetFiberDimension"
/*@
  SectionIntGetFiberDimension - Get the size of the vector space attached to the point

  Not collective

  Input Parameters:
+ section - the section object
- point - the Sieve point

  Output Parameters:
. size - The fiber dimension

  Level: advanced

.seealso SectionRealRestrict(), SectionRealCreate(), SectionRealView()
@*/
PetscErrorCode  SectionIntGetFiberDimension(SectionInt section, PetscInt point, PetscInt *size)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(section, SECTIONINT_CLASSID, 1);
  *size = section->s->getFiberDimension(point);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SectionIntSetFiberDimension"
/*@
  SectionIntSetFiberDimension - Set the size of the vector space attached to the point

  Not collective

  Input Parameters:
+ section - the section object
. point - the Sieve point
- size - The fiber dimension

  Level: advanced

.seealso SectionIntRestrict(), SectionIntCreate(), SectionIntView()
@*/
PetscErrorCode  SectionIntSetFiberDimension(SectionInt section, PetscInt point, const PetscInt size)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(section, SECTIONINT_CLASSID, 1);
  section->s->setFiberDimension(point, size);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SectionIntSetFiberDimensionField"
/*@
  SectionIntSetFiberDimensionField - Set the size of the vector space attached to the point for a given field

  Not collective

  Input Parameters:
+ section - the section object
. point - the Sieve point
. size - The fiber dimension
- field - The field number

  Level: advanced

.seealso SectionIntSetFiberDimension(), SectionIntRestrict(), SectionIntCreate(), SectionIntView()
@*/
PetscErrorCode  SectionIntSetFiberDimensionField(SectionInt section, PetscInt point, const PetscInt size, const PetscInt field)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(section, SECTIONINT_CLASSID, 1);
  section->s->setFiberDimension(point, size, field);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SectionIntGetSize"
/*@
  SectionIntGetSize - Gets the number of local dofs in this Section

  Not collective

  Input Parameter:
. section - the section object

  Output Parameter:
. size - the section size

  Level: advanced

.seealso SectionIntRestrict(), SectionIntCreate(), SectionIntView()
@*/
PetscErrorCode SectionIntGetSize(SectionInt section, PetscInt *size)
{
  Obj<PETSC_MESH_TYPE::int_section_type> s;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidIntPointer(size,2);
  ierr = SectionIntGetSection(section, s);CHKERRQ(ierr);
  *size = s->size();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SectionIntAllocate"
/*@
  SectionIntAllocate - Allocate storage for this section

  Not collective

  Input Parameter:
. section - the section object

  Level: advanced

.seealso SectionIntRestrict(), SectionIntCreate(), SectionIntView()
@*/
PetscErrorCode  SectionIntAllocate(SectionInt section)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(section, SECTIONINT_CLASSID, 1);
  section->b->allocate(section->s);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SectionIntClear"
/*@C
  SectionIntClear - Dellocate storage for this section

  Not collective

  Input Parameter:
. section - the section object

  Level: advanced

.seealso SectionIntRestrict(), SectionIntCreate(), SectionIntView()
@*/
PetscErrorCode  SectionIntClear(SectionInt section)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(section, SECTIONINT_CLASSID, 1);
  section->s->clear();
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SectionIntSet"
/*@
  SectionIntSet - Sets all the values to the given value

  Not collective

  Input Parameters:
+ section - the real Section
- val - the value

  Level: intermediate

.seealso VecNorm(), SectionIntCreate()
@*/
PetscErrorCode  SectionIntSet(SectionInt section, PetscInt val)
{
  Obj<PETSC_MESH_TYPE::int_section_type> s;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = SectionIntGetSection(section, s);CHKERRQ(ierr);
  s->set(val);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SectionIntAddSpace"
/*@
  SectionIntAddSpace - Add another field to this section

  Collective on DMMesh

  Input Parameter:
. section - the Section

  Level: advanced

.seealso SectionIntCreate(), SectionIntGetFibration()
@*/
PetscErrorCode  SectionIntAddSpace(SectionInt section)
{
  ALE::Obj<PETSC_MESH_TYPE::int_section_type> s;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = SectionIntGetSection(section, s);CHKERRQ(ierr);
  s->addSpace();
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SectionIntGetFibration"
/*@
  SectionIntGetFibration - Creates a section for only the data associated with the given field

  Collective on DMMesh

  Input Parameter:
+ section - the Section
- field- The field number

  Output Parameter:
. subsection - the section of the given field

  Level: advanced

.seealso SectionIntCreate(), SectionIntAddSpace()
@*/
PetscErrorCode  SectionIntGetFibration(SectionInt section, const PetscInt field, SectionInt *subsection)
{
  ALE::Obj<PETSC_MESH_TYPE>                   b;
  ALE::Obj<PETSC_MESH_TYPE::int_section_type> s;
  ALE::Obj<PETSC_MESH_TYPE::int_section_type> t;
  MPI_Comm       comm;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject) section, &comm);CHKERRQ(ierr);
  ierr = SectionIntGetBundle(section, b);CHKERRQ(ierr);
  ierr = SectionIntGetSection(section, s);CHKERRQ(ierr);
  ierr = SectionIntCreate(comm, subsection);CHKERRQ(ierr);
  ierr = SectionIntSetBundle(*subsection, b);CHKERRQ(ierr);
  t    = s->getFibration(field);
  ierr = SectionIntSetSection(*subsection, t);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMMeshGetVertexSectionInt"
/*@C
  DMMeshGetVertexSectionInt - Create a Section over the vertices with the specified fiber dimension

  Collective on DMMesh

  Input Parameters:
+ mesh - The DMMesh object
- fiberDim - The section name

  Output Parameter:
. section - The section

  Level: intermediate

.keywords: mesh, section, vertex
.seealso: DMMeshCreate(), SectionIntCreate()
@*/
PetscErrorCode DMMeshGetVertexSectionInt(DM dm, const char name[], PetscInt fiberDim, SectionInt *section)
{
  ALE::Obj<PETSC_MESH_TYPE> m;
  ALE::Obj<PETSC_MESH_TYPE::int_section_type> s;
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  ierr = DMMeshGetMesh(dm, m);CHKERRQ(ierr);
  ierr = SectionIntCreate(m->comm(), section);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) *section, name);CHKERRQ(ierr);
  ierr = SectionIntSetBundle(*section, m);CHKERRQ(ierr);
  ierr = SectionIntGetSection(*section, s);CHKERRQ(ierr);
#ifdef PETSC_OPT_SIEVE
  s->setChart(m->getSieve()->getChart());
#endif
  s->setName(name);
  s->setFiberDimension(m->depthStratum(0), fiberDim);
  m->allocate(s);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMMeshGetCellSectionInt"
/*@C
  DMMeshGetCellSectionInt - Create a Section over the cells with the specified fiber dimension

  Collective on DMMesh

  Input Parameters:
+ mesh - The DMMesh object
- fiberDim - The section name

  Output Parameter:
. section - The section

  Level: intermediate

.keywords: mesh, section, cell
.seealso: DMMeshCreate(), SectionIntCreate()
@*/
PetscErrorCode DMMeshGetCellSectionInt(DM dm, const char name[], PetscInt fiberDim, SectionInt *section)
{
  ALE::Obj<PETSC_MESH_TYPE> m;
  ALE::Obj<PETSC_MESH_TYPE::int_section_type> s;
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  ierr = DMMeshGetMesh(dm, m);CHKERRQ(ierr);
  ierr = SectionIntCreate(m->comm(), section);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) *section, name);CHKERRQ(ierr);
  ierr = SectionIntSetBundle(*section, m);CHKERRQ(ierr);
  ierr = SectionIntGetSection(*section, s);CHKERRQ(ierr);
#ifdef PETSC_OPT_SIEVE
  s->setChart(m->getSieve()->getChart());
#endif
  s->setName(name);
  s->setFiberDimension(m->heightStratum(0), fiberDim);
  m->allocate(s);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMMeshSetupSection"
/*@C
  DMMeshSetupSection - Layout Section based upon discretization and boundary condition information in the Mesh

  Collective on DMMesh

  Input Parameters:
. mesh - The DMMesh object

  Output Parameter:
. section - The section

  Level: intermediate

.keywords: mesh, section, cell
.seealso: DMMeshCreate(), SectionRealCreate()
@*/
PetscErrorCode DMMeshSetupSection(DM dm, SectionReal section)
{
  ALE::Obj<PETSC_MESH_TYPE> m;
  ALE::Obj<PETSC_MESH_TYPE::real_section_type> s;
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  ierr = DMMeshGetMesh(dm, m);CHKERRQ(ierr);
  ierr = SectionRealGetSection(section, s);CHKERRQ(ierr);
  try {
    m->setupField(s);
  } catch(ALE::Exception e) {
    SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "%s", e.message());
  }
  PetscFunctionReturn(0);
}
