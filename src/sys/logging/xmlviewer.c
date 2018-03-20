/*************************************************************************************
 *    M A R I T I M E  R E S E A R C H  I N S T I T U T E  N E T H E R L A N D S     *
 *************************************************************************************
 *    authors: Koos Huijssen, Christiaan M. Klaij                                    *
 *************************************************************************************
 *    content: Viewer for writing XML output                                         *
 *************************************************************************************/
#include <petscviewer.h>
#include <petsc/private/logimpl.h>
#include <petsc/private/fortranimpl.h>
#include "../src/sys/logging/xmlviewer.h"

#if defined(PETSC_USE_LOG)

static int XMLSectionDepth            = 0;

PetscErrorCode PetscViewerXMLStartSection(PetscViewer viewer, const char *name, const char *desc)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!desc) {
    ierr = PetscViewerASCIIPrintf(viewer, "%*s<%s>\n", XMLSectionDepth, "", name);CHKERRQ(ierr);
  } else {
    ierr = PetscViewerASCIIPrintf(viewer, "%*s<%s desc=\"%s\">\n", XMLSectionDepth, "", name, desc);CHKERRQ(ierr);
  }
  XMLSectionDepth += 2;
  PetscFunctionReturn(0);
}

/* Initialize a viewer to XML, and initialize the XMLDepth static parameter */
PetscErrorCode PetscViewerInitASCII_XML(PetscViewer viewer)
{
  PetscErrorCode ierr;
  MPI_Comm       comm;
  char           PerfScript[PETSC_MAX_PATH_LEN+40];

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)viewer,&comm);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer, "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");CHKERRQ(ierr);
  ierr = PetscStrreplace(comm,"<?xml-stylesheet type=\"text/xsl\" href=\"performance_xml2html.xsl\"?>",PerfScript,sizeof(PerfScript));CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer, "%s\n",PerfScript);CHKERRQ(ierr);
  XMLSectionDepth = 0;
  ierr = PetscViewerXMLStartSection(viewer, "root", NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* Initialize a viewer to XML, and initialize the XMLDepth static parameter */
PetscErrorCode PetscViewerFinalASCII_XML(PetscViewer viewer)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscViewerXMLEndSection(viewer, "root");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscViewerXMLEndSection(PetscViewer viewer, const char *name)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  XMLSectionDepth -= 2;
  if (XMLSectionDepth<0) XMLSectionDepth = 0;
  ierr = PetscViewerASCIIPrintf(viewer, "%*s</%s>\n", XMLSectionDepth, "", name);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscViewerXMLPutString(PetscViewer viewer, const char *name, const char *desc, const char *value)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!desc) {
    ierr = PetscViewerASCIIPrintf(viewer, "%*s<%s>%s</%s>\n", XMLSectionDepth, "", name, value, name);CHKERRQ(ierr);
  } else {
    ierr = PetscViewerASCIIPrintf(viewer, "%*s<%s desc=\"%s\">%s</%s>\n", XMLSectionDepth, "", name, desc, value, name);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PetscViewerXMLPutInt(PetscViewer viewer, const char *name, const char *desc, int value)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!desc) {
    ierr = PetscViewerASCIIPrintf(viewer, "%*s<%s>%d</%s>\n", XMLSectionDepth, "", name, value, name);CHKERRQ(ierr);
  } else {
    ierr = PetscViewerASCIIPrintf(viewer, "%*s<%s desc=\"%s\">%d</%s>\n", XMLSectionDepth, "", name, desc, value, name);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PetscViewerXMLPutDouble(PetscViewer viewer, const char *name, const char *desc, PetscLogDouble value, const char *format)
{
  PetscErrorCode ierr;
  char           buffer[1024];

  PetscFunctionBegin;
  if (!desc) {
    ierr = PetscSNPrintf(buffer,sizeof(buffer), "%*s<%s>%s</%s>\n", XMLSectionDepth, "", name, format, name);CHKERRQ(ierr);
  } else {
    ierr = PetscSNPrintf(buffer,sizeof(buffer), "%*s<%s desc=\"%s\">%s</%s>\n", XMLSectionDepth, "", name, desc, format, name);CHKERRQ(ierr);
  }
  ierr = PetscViewerASCIIPrintf(viewer, buffer, value);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#endif
