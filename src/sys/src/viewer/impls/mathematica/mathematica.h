/* $Id: mathematica.h,v 1.2 1999/06/01 20:45:51 knepley Exp $ */
/* 
   This is the definition of the Mathematica viewer structure.
*/

#include "src/sys/src/viewer/viewerimpl.h"   /*I  "petsc.h"  I*/
#include "petscsys.h" 
#ifdef PETSC_HAVE_MATHEMATICA
#include "mathlink.h"
#endif

typedef enum {GRAPHICS_MOTIF, GRAPHICS_PS_FILE, GRAPHICS_PS_STDOUT} GraphicsType;
typedef enum {MATHEMATICA_TRIANGULATION_PLOT, MATHEMATICA_VECTOR_TRIANGULATION_PLOT,
              MATHEMATICA_SURFACE_PLOT,       MATHEMATICA_VECTOR_PLOT} PlotType;
typedef enum {MATHEMATICA_LINK_CREATE, MATHEMATICA_LINK_CONNECT, MATHEMATICA_LINK_LAUNCH} LinkMode;

typedef struct {
#ifdef PETSC_HAVE_MATHEMATICA
  MLINK        link;         /* The link to Mathematica */
#endif
  char        *linkname;     /* The name to link to Mathematica on (usually a port) */
  char        *linkhost;     /* The host to link to Mathematica on */
  LinkMode     linkmode;     /* The link mode */
  GraphicsType graphicsType; /* The system to use for display */
  PlotType     plotType;     /* The type of plot to make */
  const char  *objName;      /* The name for the next object passed to Mathematica */
} PetscViewer_Mathematica;

EXTERN int PetscViewerMathematicaSetFromOptions(PetscViewer);

EXTERN int PetscViewerMathematicaSetLinkName(PetscViewer, const char *);
EXTERN int PetscViewerMathematicaSetLinkPort(PetscViewer, int);
EXTERN int PetscViewerMathematicaSetLinkHost(PetscViewer, const char *);
EXTERN int PetscViewerMathematicaSetLinkMode(PetscViewer, LinkMode);
