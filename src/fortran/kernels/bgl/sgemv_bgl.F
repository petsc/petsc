!
!    Fortran kernel for gemv() BLAS operation. This version supports
!  matrix array stored in single precision but vectors in double
!
#include "include/finclude/petscdef.h"
!
      subroutine MSGemv_BGL(bs,ncols,A,x,y)
      implicit none
      PetscInt          bs,ncols
      MatScalar        A(bs,ncols)
      PetscScalar      x(ncols),y(bs)

      PetscScalar      bgl_size_0_precision
      PetscInt         i,j  

      if( kind(bgl_size_0_precision) .eq. 4) then
        if(sizeof(bgl_size_0_precision) .eq. 4 ) then
           call SGEMV('N', bs, ncols, 1, A, bs, x, 1, 0, y, 1)
	   return
        else if(sizeof(bgl_size_0_precision) .eq. 8 ) then
           call CGEMV('N', bs, ncols, 1, A, bs, x, 1, 0, y, 1)
           return
        endif
      else if (kind(bgl_size_0_precision) .eq. 8) then
        if(sizeof(bgl_size_0_precision) .eq. 8 ) then
           call DGEMV('N', bs, ncols, 1, A, bs, x, 1, 0, y, 1)
	   return
        else if(sizeof(bgl_size_0_precision) .eq. 16 ) then
           call ZGEMV('N', bs, ncols, 1, A, bs, x, 1, 0, y, 1)
           return
        endif
      endif

      write(6,*)'Error in MSGemv: unavailable type'            
      

      do 5, j=1,bs
        y(j) = 0.0d0
 5    continue

      do 10, i=1,ncols
        do 20, j=1,bs
          y(j) = y(j) + A(j,i)*x(i)
 20     continue
 10   continue

      return 
      end

      
      subroutine MSGemvp_BGL(bs,ncols,A,x,y)
      implicit none
      PetscInt          bs,ncols
      MatScalar        A(bs,ncols)
      PetscScalar      x(ncols),y(bs)

      PetscInt         i, j
      PetscScalar      bgl_size_0_precision

      if( kind(bgl_size_0_precision) .eq. 4) then
        if(sizeof(bgl_size_0_precision) .eq. 4 ) then
           call SGEMV('N', bs, ncols, 1, A, bs, x, 1, 1, y, 1)
	   return
        else if(sizeof(bgl_size_0_precision) .eq. 8 ) then
           call CGEMV('N', bs, ncols, 1, A, bs, x, 1, 1, y, 1)
           return
        endif
      else if (kind(bgl_size_0_precision) .eq. 8) then
        if(sizeof(bgl_size_0_precision) .eq. 8 ) then
           call DGEMV('N', bs, ncols, 1, A, bs, x, 1, 1, y, 1)
	   return
        else if(sizeof(bgl_size_0_precision) .eq. 16 ) then
           call ZGEMV('N', bs, ncols, 1, A, bs, x, 1, 1, y, 1)
           return
        endif
      endif
 
      write(6,*)'Error in MSGemvp: unavailable type'            

      do 10, i=1,ncols
        do 20, j=1,bs
          y(j) = y(j) + A(j,i)*x(i)
 20     continue
 10   continue


      return 
      end


      subroutine MSGemvm_BGL(bs,ncols,A,x,y)
      implicit none
      PetscInt          bs,ncols
      MatScalar        A(bs,ncols)
      PetscScalar      x(ncols),y(bs)

      PetscInt         i, j
      PetscScalar      bgl_size_0_precision

      if( kind(bgl_size_0_precision) .eq. 4) then
        if(sizeof(bgl_size_0_precision) .eq. 4 ) then
           call SGEMV('N', bs, ncols, 1, A, bs, x, 1, -1, y, 1)
	   return
        else if(sizeof(bgl_size_0_precision) .eq. 8 ) then
           call CGEMV('N', bs, ncols, 1, A, bs, x, 1, -1, y, 1)
           return
        endif
      else if (kind(bgl_size_0_precision) .eq. 8) then
        if(sizeof(bgl_size_0_precision) .eq. 8 ) then
           call DGEMV('N', bs, ncols, 1, A, bs, x, 1, -1, y, 1)
	   return
        else if(sizeof(bgl_size_0_precision) .eq. 16 ) then
           call ZGEMV('N', bs, ncols, 1, A, bs, x, 1, -1, y, 1)
           return
        endif
      endif

      write(6,*)'Error in MSGemvm: unavailable type'            

      do 10, i=1,ncols
        do 20, j=1,bs
          y(j) = y(j) - A(j,i)*x(i)
 20     continue
 10   continue

      return 
      end


      subroutine MSGemvt_BGL(bs,ncols,A,x,y)
      implicit none
      PetscInt          bs,ncols
      MatScalar        A(bs,ncols)
      PetscScalar      x(bs),y(ncols)

      PetscInt          i,j
      PetscScalar      sum
      PetscScalar      bgl_size_0_precision

      if( kind(bgl_size_0_precision) .eq. 4) then
        if(sizeof(bgl_size_0_precision) .eq. 4 ) then
           call SGEMV('T', bs, ncols, 1, A, bs, x, 1, y, 1)
	   return
        else if(sizeof(bgl_size_0_precision) .eq. 8 ) then
           call CGEMV('T', bs, ncols, 1, A, bs, x, 1, y, 1)
           return
        endif
      else if (kind(bgl_size_0_precision) .eq. 8) then
        if(sizeof(bgl_size_0_precision) .eq. 8 ) then
           call DGEMV('T', bs, ncols, 1, A, bs, x, 1, y, 1)
	   return
        else if(sizeof(bgl_size_0_precision) .eq. 16 ) then
           call ZGEMV('T', bs, ncols, 1, A, bs, x, 1, y, 1)
           return
        endif
      endif

      write(6,*)'Error in MSGemvt: unavailable type'            
      
      do 10, i=1,ncols
        sum = y(i)
        do 20, j=1,bs
          sum = sum + A(j,i)*x(j)
 20     continue
        y(i) = sum
 10   continue

      return 
      end

      subroutine MSGemm_BGL(bs,A,B,C)
      implicit none
      PetscInt          bs
      MatScalar        A(bs,bs),B(bs,bs),C(bs,bs)
      PetscScalar      sum
      PetscInt          i,j,k

      PetscScalar      bgl_size_0_precision

      if( kind(bgl_size_0_precision) .eq. 4) then
        if(sizeof(bgl_size_0_precision) .eq. 4 ) then
           call SGEMM('N', 'N', bs, bs, bs, -1, B, bs, C, bs, 1, A, bs)
	   return
        else if(sizeof(bgl_size_0_precision) .eq. 8 ) then
           call CGEMM('N', 'N', bs, bs, bs, -1, B, bs, C, bs, 1, A, bs)
           return
        endif
      else if (kind(bgl_size_0_precision) .eq. 8) then
        if(sizeof(bgl_size_0_precision) .eq. 8 ) then
           call DGEMM('N', 'N', bs, bs, bs, -1, B, bs, C, bs, 1, A, bs)
	   return
        else if(sizeof(bgl_size_0_precision) .eq. 16 ) then
           call ZGEMM('N', 'N', bs, bs, bs, -1, B, bs, C, bs, 1, A, bs)
           return
        endif
      endif


      write(6,*)'Error in MSGemm: unavailable type'            

      do 10, i=1,bs
        do 20, j=1,bs
          sum = A(i,j)
          do 30, k=1,bs
            sum = sum - B(i,k)*C(k,j)
 30       continue
          A(i,j) = sum
 20     continue
 10   continue

      return 
      end


      subroutine MSGemmi_BGL(bs,A,C,B)
      implicit none
      PetscInt          bs
      MatScalar        A(bs,bs),B(bs,bs),C(bs,bs)
      PetscScalar      sum

      PetscInt          i,j,k

      PetscScalar      bgl_size_0_precision

      if( kind(bgl_size_0_precision) .eq. 4) then
        if(sizeof(bgl_size_0_precision) .eq. 4 ) then
           call SGEMM('N', 'N', bs, bs, bs, 1, B, bs, C, bs, 0, A, bs)
	   return
        else if(sizeof(bgl_size_0_precision) .eq. 8 ) then
           call CGEMM('N', 'N', bs, bs, bs, 1, B, bs, C, bs, 0, A, bs)
           return
        endif
      else if (kind(bgl_size_0_precision) .eq. 8) then
        if(sizeof(bgl_size_0_precision) .eq. 8 ) then
           call DGEMM('N', 'N', bs, bs, bs, 1, B, bs, C, bs, 0, A, bs)
	   return
        else if(sizeof(bgl_size_0_precision) .eq. 16 ) then
           call ZGEMM('N', 'N', bs, bs, bs, 1, B, bs, C, bs, 0, A, bs)
           return
        endif
      endif

      write(6,*)'Error in MSGemmi: unavailable type'            

      do 10, i=1,bs
        do 20, j=1,bs
          sum = 0.0d0
          do 30, k=1,bs
            sum = sum + B(i,k)*C(k,j)
 30       continue
          A(i,j) = sum
 20     continue
 10   continue

      return 
      end
