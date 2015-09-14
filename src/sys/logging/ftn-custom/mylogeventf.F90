module myLogEventF
use iso_c_binding
#include "finclude/petscdef.h"
implicit none

   interface  

     subroutine PetscLogViewNested(viewer, ierr) bind(c,name='PetscLogViewNestedRoutine')
     use iso_c_binding
     implicit none
     PetscViewer     :: viewer
     integer(c_int)  :: ierr
     end subroutine PetscLogViewNested
 
     function PetscLogInitializeNested() bind(c,name='PetscLogInitializeNested') result(ierr)
     use iso_c_binding
     implicit none
     integer(c_int) :: ierr
     end function PetscLogInitializeNested
     
     function PetscLogFreeNested() bind(c,name='PetscLogFreeNested') result(ierr)
     use iso_c_binding
     implicit none
     integer(c_int) :: ierr
     end function PetscLogFreeNested

     function PetscLogSetThreshold(newThresh,oldThresh) bind(c,name='PetscLogSetThreshold') result(ierr)
     use iso_c_binding
     implicit none
     PetscLogDouble, intent(in), value :: newThresh
     PetscLogDouble, intent(out) :: oldThresh
     integer(c_int) :: ierr
     end function PetscLogSetThreshold
   end interface
end module myLogEventF

