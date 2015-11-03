#if !defined(__XMLVIEWER_H)
#define __XMLVIEWER_H

PetscErrorCode PetscViewerInitASCII_XML(PetscViewer);
PetscErrorCode PetscViewerFinalASCII_XML(PetscViewer);
PetscErrorCode PetscViewerCheckASCII_XMLFormat(PetscViewer, PetscBool *);
PetscErrorCode PetscViewerXMLStartSection(PetscViewer, const char *, const char *);
PetscErrorCode PetscViewerXMLEndSection(PetscViewer, const char *);
PetscErrorCode PetscViewerXMLPutString(PetscViewer, const char *, const char *, const char *);
PetscErrorCode PetscViewerXMLPutInt(PetscViewer, const char *, const char *, int);
PetscErrorCode PetscViewerXMLPutDouble(PetscViewer, const char *, const char *, PetscLogDouble, const char *);
#endif
