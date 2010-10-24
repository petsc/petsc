classdef KSP < PetscObject
  methods
    function obj = KSP()
      [err,obj.pobj] = calllib('libpetsc', 'KSPCreate', 0,0);
    end
    function err = SetType(obj,name)
      err = calllib('libpetsc', 'KSPSetType', obj.pobj,name);
    end
    function err = SetDM(obj,da)
      err = calllib('libpetsc', 'KSPSetDM', obj.pobj,da.pobj);
    end
    function err = SetFromOptions(obj)
      err = calllib('libpetsc', 'KSPSetFromOptions', obj.pobj);
    end
    function err = SetUp(obj)
      err = calllib('libpetsc', 'KSPSetUp', obj.pobj);
    end
    function err = Solve(obj,b,x)
      if (nargin == 1) 
        b = 0;
        x = 0;
      end
      if (b ~= 0) 
        b = b.pobj;
      end
      if (x ~= 0)
        x = x.pobj;
      end
      err = calllib('libpetsc', 'KSPSolve', obj.pobj,b,x);
    end
    function err = SetOperators(obj,A,B,pattern)
      err = calllib('libpetsc', 'KSPSetOperators', obj.pobj,A.pobj,B.pobj,pattern);
    end
    function [x,err] = GetSolution(obj)
      [err,pid] = calllib('libpetsc', 'KSPGetSolution', obj.pobj,0);
      x = Vec(pid,'pobj');
    end
    function err = View(obj,viewer)
      err = calllib('libpetsc', 'KSPView', obj.pobj,viewer.pobj);
    end
    function err = Destroy(obj)
      err = calllib('libpetsc', 'KSPDestroy', obj.pobj);
    end
  end
end

 
