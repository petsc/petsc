classdef SNES < PetscObject
  methods
    function obj = SNES()
      [err,obj.pobj] = calllib('libpetsc', 'SNESCreate', 0,0);
    end
    function err = SetType(obj,name)
      err = calllib('libpetsc', 'SNESSetType', obj.pobj,name);
    end
    function err = SetFromOptions(obj)
      err = calllib('libpetsc', 'SNESSetFromOptions', obj.pobj);
    end
    function err = SetUp(obj)
      err = calllib('libpetsc', 'SNESSetUp', obj.pobj);
    end
    function err = Solve(obj,b,x)
      if (nargin < 3) 
        err = calllib('libpetsc', 'SNESSolve', obj.pobj,0,b.pobj);
      else
        err = calllib('libpetsc', 'SNESSolve', obj.pobj,b.pboj,x.pobj);
      end
    end
    function err = SetFunction(obj,f,func)
      if ~isa(func,'function_handle')
	disp('Requires function handle argument')
        err = 1
        return 
      end
      err = calllib('libpetsc', 'SNESSetFunctionMatlab', obj.pobj,f.pobj,func2str(func));
    end
    function err = View(obj,viewer)
      err = calllib('libpetsc', 'SNESView', obj.pobj,viewer.pobj);
    end
    function err = Destroy(obj)
      err = calllib('libpetsc', 'SNESDestroy', obj.pobj);
    end
  end
end

 
