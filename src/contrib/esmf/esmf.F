!
!    "$Id: esmf.F,v 1.1 2000/09/08 03:08:03 bsmith Exp bsmith $"
!
      module esmf
#include "finclude/petsc.h"

!--------------------------------------------------------------
!
!    Base object: all nontrivial ESMF objects contain this as their first entry
!
      type esmfbase
        character*(64) typename 
        integer        refcount
        MPI_Comm       comm
        integer        threadcount
      end type esmfbase

!--------------------------------------------------------------
!
!   Record of current time (and possibly related material) that indicates  
!   the time of field data
!
      type esmftimestamp
        double precision currenttime
      end type esmftimestamp

!--------------------------------------------------------------
!
!   An abstract representation of an IO device etc. that fields,
!  grid etc may be read from an written to.
!
!   Currently has dummy data in it
!
      type esmfio
        type (esmfbase)  base
        character*(256) filename
      end type esmfio

!--------------------------------------------------------------
!--------------------------------------------------------------
!
!  Currently no polymorphic function
!
!
!--------------------------------------------------------------
!--------------------------------------------------------------
!
!  Currently no functions
!
      end module esmf












