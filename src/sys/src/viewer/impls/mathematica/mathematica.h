/* $Id: mathematica.h,v 1.2 1999/06/01 20:45:51 knepley Exp $ */
/* 
   This is the definition of the Mathematica viewer structure.
*/

#include "src/sys/src/viewer/viewerimpl.h"   /*I  "petsc.h"  I*/
#include "petscsys.h" 

typedef enum {GRAPHICS_MOTIF, GRAPHICS_PS_FILE, GRAPHICS_PS_STDOUT} GraphicsType;
typedef enum {MATHEMATICA_TRIANGULATION_PLOT, MATHEMATICA_VECTOR_TRIANGULATION_PLOT,
              MATHEMATICA_SURFACE_PLOT,       MATHEMATICA_VECTOR_PLOT} PlotType;

typedef struct {
#ifdef PETSC_HAVE_MATHEMATICA
  MLINK        link;         /* The link to Mathematica */
#endif
  GraphicsType graphicsType; /* The system to use for display */
  PlotType     plotType;     /* The type of plot to make */
  char        *objName;      /* The name for the next object passed to Mathematica */
} PetscViewer_Mathematica;
