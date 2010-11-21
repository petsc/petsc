classdef PetscMat < PetscObject
  properties (Constant)
    FLUSH_ASSEMBLY=1;
    FINAL_ASSEMBLY=0;

    SAME_NONZERO_PATTERN=0;
    DIFFERENT_NONZERO_PATTERN=1;
    SAME_PRECONDITIONER=2;
    SUBSET_NONZERO_PATTERN=3;
  end
  methods
    function obj = PetscMat(array,flg)
      if (nargin > 1) 
        %  PetscMat(pid,'pobj') uses an already existing PETSc Mat object
        obj.pobj = array;
        return
      end
      [err,obj.pobj] = calllib('libpetsc', 'MatCreate', 0,0);
    end
    function err = SetType(obj,name)
      err = calllib('libpetsc', 'MatSetType', obj.pobj,name);
    end
    function err = SetFromOptions(obj)
      err = calllib('libpetsc', 'MatSetSetFromOptions', obj.pobj);
    end
    function err = SetSizes(obj,m,n,M,N)
      err = calllib('libpetsc', 'MatSetSizes', obj.pobj,m,n,M,N);
    end
    function err = SetValues(obj,idx,idy,values,insertmode)
      idx = idx - 1;
      idy = idy - 1;
      if (nargin < 5) 
        insertmode = PetscObject.INSERT_VALUES;
      end
      err = calllib('libpetsc', 'MatSetValues', obj.pobj,length(idx),idx,length(idy),idy,values,insertmode);
    end
    function err = AssemblyBegin(obj,mode)
      err = calllib('libpetsc', 'MatAssemblyBegin', obj.pobj,mode);
    end
    function err = AssemblyEnd(obj,mode)
      err = calllib('libpetsc', 'MatAssemblyEnd', obj.pobj,mode);
    end
    function err = View(obj,viewer)
      err = calllib('libpetsc', 'MatView', obj.pobj,viewer.pobj);
    end
    function err = Destroy(obj)
      err = calllib('libpetsc', 'MatDestroy', obj.pobj);
    end
    function err = SetValuesStencil(obj,row,col,values,insertmode)
      if (nargin < 5) 
        insertmode = PetscObject.INSERT_VALUES;
      end
      ndim = isfield(row,'i') + isfield(row,'j') + isfield(row,'k'); 
      nrow = length(row);
      ncol = length(col);
      if (ndim == 1)  %% 1D DM
	for (m=1:nrow)
            row(m).i = row(m).i - 1;
        end
        for (m = 1:ncol)
             col(m).i = col(m).i - 1;
        end
      elseif (ndim == 2)  %% 2D DM
        for (m = 1:nrow)
          row(m).i = row(m).i - 1;
          row(m).j = row(m).j - 1;
        end
        for (m = 1:ncol)
          col(m).i = col(m).i - 1;
          col(m).j = col(m).j - 1;
        end 
      elseif (ndim == 3)  %% 3D DM
        for (m = 1:nrow)
          row(m).i = row(m).i - 1;
          row(m).j = row(m).j - 1;
          row(m).k = row(m).k - 1;
        end
        for (m = 1:ncol)
          col(m).i = col(m).i - 1;
          col(m).j = col(m).j - 1;
          col(m).k = col(m).k - 1;
        end 
      end
      err = calllib('libpetsc','MatSetValuesStencil',obj.pobj,nrow,row,ncol,col,values,insertmode);  
    end
  end
end

 
