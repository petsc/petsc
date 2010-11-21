classdef PetscVec < PetscObject
  properties
    VecFromDM=0;
    DM = [];
  end
  methods
    function [obj] = SetVecfromDM(obj,value)
      obj.VecFromDM = value;
    end
    function [obj] = SetDM(obj,DM)
        obj.DM = DM;
    end
    function obj = PetscVec(array,flg)
      if (nargin > 1) 
        %  PetscVec(pid,'pobj') uses an already existing PETSc Vec object
        obj.pobj = array;
        return
      end
      [err,obj.pobj] = calllib('libpetsc', 'VecCreate', 0,0);
      if (nargin > 0) 
        % Vec(array) creates a Vec initialized with the given array
        err = calllib('libpetsc', 'VecSetType', obj.pobj,'seq');
        err = calllib('libpetsc', 'VecSetSizes', obj.pobj,length(array),length(array));
        idx = 0:length(array)-1;
        err = calllib('libpetsc', 'VecSetValues', obj.pobj,length(idx),idx,array,PetscObject.INSERT_VALUES);
        err = calllib('libpetsc', 'VecAssemblyBegin', obj.pobj);
        err = calllib('libpetsc', 'VecAssemblyEnd', obj.pobj);
      end
    end
    function err = SetFromOptions(obj)
      err = calllib('libpetsc', 'VecSetFromOptions', obj.pobj);
    end
    function err = SetType(obj,name)
      err = calllib('libpetsc', 'VecSetType', obj.pobj,name);
    end
    function err = SetSizes(obj,m,n)
      err = calllib('libpetsc', 'VecSetSizes', obj.pobj,m,n);
    end
    function [n,err] = GetSize(obj)
      n = 0;
      [err,n] = calllib('libpetsc', 'VecGetLocalSize', obj.pobj,n);
    end
    function err = SetValues(obj,idx,values,insertmode)
      if (ischar(idx)) % assume it is ':' 
        [n,err] = GetSize(obj);
        idx = 1:n;
      end
      if (nargin < 3) 
        values = idx;
      end
      if (nargin  < 4) 
        insertmode = PetscObject.INSERT_VALUES;
      end
      idx = idx - 1;
      err = calllib('libpetsc', 'VecSetValues', obj.pobj,length(idx),idx,values,insertmode);
    end
    function [values,err] = GetValues(obj,idx)
      if (ischar(idx)) % assume it is ':' 
        [n,err] = GetSize(obj);
        idx = 1:n;
      end
      idx = idx - 1;
      values = zeros(1,length(idx));
      [err,idx,values] = calllib('libpetsc', 'VecGetValues', obj.pobj,length(idx),idx,values);
    end
    function err = AssemblyBegin(obj)
      err = calllib('libpetsc', 'VecAssemblyBegin', obj.pobj);
    end
    function err = AssemblyEnd(obj)
      err = calllib('libpetsc', 'VecAssemblyEnd', obj.pobj);
    end
    function [vec,err] = Duplicate(obj)
      [err,pid] = calllib('libpetsc', 'VecDuplicate', obj.pobj,0);
      vec = PetscVec(pid,'pobj');
    end
    function err = Copy(obj,v)
      err = calllib('libpetsc', 'VecCopy', obj.pobj,v.pobj);
    end
    function err = Set(obj,v)
      err = calllib('libpetsc', 'VecSet', obj.pobj,v);
    end
    function err = View(obj,viewer)
      err = calllib('libpetsc', 'VecView', obj.pobj,viewer.pobj);
    end
    function err = Destroy(obj)
      err = calllib('libpetsc', 'VecDestroy', obj.pobj);
    end
%
%   The following overload a = x(idx) and x(idx) = a
%
    function varargout = subsref(obj,S)
      %  Matlab design of subsref is MORONIC
      %  To overload () it automatically overloads . which is used
      %  for method calls so we need to force the method calls to use
      %  the 'regular' subsref
      if (S(1).type == '.')
        [varargout{1:nargout}] = builtin('subsref', obj, S);
      else
        if (obj.VecFromDM)
          varargout = {obj.GetValues_DM(S)};
        else
          varargout = {obj.GetValues(S.subs{1})};
        end
      end
    end
    function varargout = GetValues_DM(obj,S)
      M = obj.DM.M; N = obj.DM.N; P = obj.DM.P;
      ndim = obj.DM.ndim;
      if (ndim == 1)  %% 1D DM
        idx = S.subs{1};
      elseif (ndim == 2)  %% 2D DM
        idx = M*(S.subs{2}-1) + S.subs{1};  
      elseif (ndim == 3)  %% 3D DM
        idx = N*(S.subs{3}-1) + M*(S.subs{2}-1) + S.subs{1};
      end
      varargout = {obj.GetValues(idx)};      
    end
    function obj = subsasgn(obj,S,value)
      if (S(1).type ~= '.')
        if (obj.VecFromDM)
          [obj] = obj.SetValues_DM(S,value);
        else
          obj.SetValues(S.subs{1},value);
        end
      end
    end
    function [obj] = SetValues_DM(obj,S,value)
      M = obj.DM.M; N = obj.DM.N; P = obj.DM.P;
      ndim = obj.DM.ndim;
      if (ndim == 1)  %% 1D DM
        idx = S.subs{1};
      elseif (ndim == 2)  %% 2D DM
        idx = M*(S.subs{2}-1) + S.subs{1};  
      elseif (ndim == 3)  %% 3D DM
        idx = N*(S.subs{3}-1) + M*(S.subs{2}-1) + S.subs{1};
      end
      obj.SetValues(idx,value);      
    end
  end
end

 
