#ifndef PETSCXMLVIEWER_H
#define PETSCXMLVIEWER_H

PETSC_INTERN PetscErrorCode PetscViewerInitASCII_XML(PetscViewer);
PETSC_INTERN PetscErrorCode PetscViewerFinalASCII_XML(PetscViewer);
PETSC_INTERN PetscErrorCode PetscViewerXMLStartSection(PetscViewer, const char *, const char *);
PETSC_INTERN PetscErrorCode PetscViewerXMLEndSection(PetscViewer, const char *);
PETSC_INTERN PetscErrorCode PetscViewerXMLPutString(PetscViewer, const char *, const char *, const char *);
PETSC_INTERN PetscErrorCode PetscViewerXMLPutInt(PetscViewer, const char *, const char *, int);
PETSC_INTERN PetscErrorCode PetscViewerXMLPutDouble(PetscViewer, const char *, const char *, PetscLogDouble, const char *);

#endif
