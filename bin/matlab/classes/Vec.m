classdef Vec < PetscObject
  methods
    function obj = Vec(array)
      [err,obj.pobj] = calllib('libpetsc', 'VecCreate', 0,0);
      if (nargin > 0) 
        err = calllib('libpetsc', 'VecSetType', obj.pobj,'seq');
        err = calllib('libpetsc', 'VecSetSizes', obj.pobj,length(array),length(array));
        idx = 0:length(array)-1;
        err = calllib('libpetsc', 'VecSetValues', obj.pobj,length(idx),idx,array,PetscObject.INSERT_VALUES);
        err = calllib('libpetsc', 'VecAssemblyBegin', obj.pobj);
        err = calllib('libpetsc', 'VecAssemblyEnd', obj.pobj);
      end
    end
    function SetType(obj,name)
      err = calllib('libpetsc', 'VecSetType', obj.pobj,name);
    end
    function SetSizes(obj,m,n)
      err = calllib('libpetsc', 'VecSetSizes', obj.pobj,m,n);
    end
    function SetValues(obj,idx,values,insertmode)
      if (nargin < 3) 
        values = idx;
        idx = 0:length(values)-1;
      end
      if (nargin  < 4) 
        insertmode = PetscObject.INSERT_VALUES;
      end
      err = calllib('libpetsc', 'VecSetValues', obj.pobj,length(idx),idx,values,insertmode);
    end
    function values = GetValues(obj,idx)
      if (ischar(idx)) % assume it is ':' 
        idx = 0:2;
      end
      values = zeros(1,length(idx));
      [err,idx,values] = calllib('libpetsc', 'VecGetValues', obj.pobj,length(idx),idx,values);
    end
    function AssemblyBegin(obj)
      err = calllib('libpetsc', 'VecAssemblyBegin', obj.pobj);
    end
    function AssemblyEnd(obj)
      err = calllib('libpetsc', 'VecAssemblyEnd', obj.pobj);
    end
    function View(obj,viewer)
      err = calllib('libpetsc', 'VecView', obj.pobj,viewer.pobj);
    end
    function Destroy(obj)
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
        varargout = {obj.GetValues(S.subs{1})};
      end
    end
    function obj = subsasgn(obj,S,value)
      if (S(1).type ~= '.')
        obj.SetValues(S.subs{1},value);
      end
    end
  end
end

 
