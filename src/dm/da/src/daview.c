#ifndef lint
static char vcid[] = "$Id: da2.c,v 1.32 1996/01/27 04:56:41 bsmith Exp $";
#endif
 
/*
  Code for manipulating distributed regular arrays in parallel.
*/

#include "daimpl.h"    /*I   "da.h"   I*/

/*@
   DAView - Visualizes a distributed array object.

   Input Parameters:
.  da - the distributed array
.  ptr - an optional visualization context

   Notes:
   The available visualization contexts include
$     STDOUT_VIEWER_SELF - standard output (default)
$     STDOUT_VIEWER_WORLD - synchronized standard
$       output where only the first processor opens
$       the file.  All other processors send their 
$       data to the first processor to print. 

   The user can open alternative vistualization contexts with
$    ViewerFileOpenASCII() - output to a specified file
$    DrawOpenX() - output nonzero matrix structure to 
$         an X window display

   Default Output Format:
$ (for 3d arrays):
$   Processor [proc] M  N  P  m  n  p  w  s
$   where
$      M,N,P - global dimension in each direction of the array
$      m,n,p - corresponding number of procs in each dimension 
$      w - number of degress of freedom per node
$      s - stencil width

.keywords: distributed array, view, visualize

.seealso: ViewerFileOpenASCII(), DrawOpenX(), 
@*/
int DAView(DA da, Viewer v)
{
  PETSCVALIDHEADERSPECIFIC(da,DA_COOKIE);
  return (*da->view)((PetscObject)da,v);
}  

