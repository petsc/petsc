!
!    Application context data, stored in common block
!
      double precision h,nrm_2,nrm_max,nox
      Vec    localwork,csolution
      DA     da
      PetscViewer viewer1,viewer2
      PetscInt    M
      common /tsctx/   h,nrm_2,nrm_max,nox
      common /tsctx/   viewer1,viewer2,localwork,csolution,da,M

