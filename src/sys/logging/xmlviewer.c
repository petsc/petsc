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
  PetscFunctionBegin;
  if (!desc) {
    CHKERRQ(PetscViewerASCIIPrintf(viewer, "%*s<%s>\n", XMLSectionDepth, "", name));
  } else {
    CHKERRQ(PetscViewerASCIIPrintf(viewer, "%*s<%s desc=\"%s\">\n", XMLSectionDepth, "", name, desc));
  }
  XMLSectionDepth += 2;
  PetscFunctionReturn(0);
}

/* Initialize a viewer to XML, and initialize the XMLDepth static parameter */
PetscErrorCode PetscViewerInitASCII_XML(PetscViewer viewer)
{
  MPI_Comm       comm;
  char           PerfScript[PETSC_MAX_PATH_LEN+40];

  PetscFunctionBegin;
  CHKERRQ(PetscObjectGetComm((PetscObject)viewer,&comm));
  CHKERRQ(PetscViewerASCIIPrintf(viewer, "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"));
  CHKERRQ(PetscStrreplace(comm,"<?xml-stylesheet type=\"text/xsl\" href=\"performance_xml2html.xsl\"?>",PerfScript,sizeof(PerfScript)));
  CHKERRQ(PetscViewerASCIIPrintf(viewer, "%s\n",PerfScript));
  XMLSectionDepth = 0;
  CHKERRQ(PetscViewerXMLStartSection(viewer, "root", NULL));
  PetscFunctionReturn(0);
}

/* Initialize a viewer to XML, and initialize the XMLDepth static parameter */
PetscErrorCode PetscViewerFinalASCII_XML(PetscViewer viewer)
{
  PetscFunctionBegin;
  CHKERRQ(PetscViewerXMLEndSection(viewer, "root"));
  PetscFunctionReturn(0);
}

PetscErrorCode PetscViewerXMLEndSection(PetscViewer viewer, const char *name)
{
  PetscFunctionBegin;
  XMLSectionDepth -= 2;
  if (XMLSectionDepth<0) XMLSectionDepth = 0;
  CHKERRQ(PetscViewerASCIIPrintf(viewer, "%*s</%s>\n", XMLSectionDepth, "", name));
  PetscFunctionReturn(0);
}

PetscErrorCode PetscViewerXMLPutString(PetscViewer viewer, const char *name, const char *desc, const char *value)
{
  PetscFunctionBegin;
  if (!desc) {
    CHKERRQ(PetscViewerASCIIPrintf(viewer, "%*s<%s>%s</%s>\n", XMLSectionDepth, "", name, value, name));
  } else {
    CHKERRQ(PetscViewerASCIIPrintf(viewer, "%*s<%s desc=\"%s\">%s</%s>\n", XMLSectionDepth, "", name, desc, value, name));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PetscViewerXMLPutInt(PetscViewer viewer, const char *name, const char *desc, int value)
{
  PetscFunctionBegin;
  if (!desc) {
    CHKERRQ(PetscViewerASCIIPrintf(viewer, "%*s<%s>%d</%s>\n", XMLSectionDepth, "", name, value, name));
  } else {
    CHKERRQ(PetscViewerASCIIPrintf(viewer, "%*s<%s desc=\"%s\">%d</%s>\n", XMLSectionDepth, "", name, desc, value, name));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PetscViewerXMLPutDouble(PetscViewer viewer, const char *name, const char *desc, PetscLogDouble value, const char *format)
{
  char           buffer[1024];

  PetscFunctionBegin;
  if (!desc) {
    CHKERRQ(PetscSNPrintf(buffer,sizeof(buffer), "%*s<%s>%s</%s>\n", XMLSectionDepth, "", name, format, name));
  } else {
    CHKERRQ(PetscSNPrintf(buffer,sizeof(buffer), "%*s<%s desc=\"%s\">%s</%s>\n", XMLSectionDepth, "", name, desc, format, name));
  }
  CHKERRQ(PetscViewerASCIIPrintf(viewer, buffer, value));
  PetscFunctionReturn(0);
}

#endif
