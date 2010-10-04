classdef KSP < PetscObject
  methods
    function obj = KSP()
      [err,obj.pobj] = calllib('libpetsc', 'KSPCreate', 0,0);
    end
    function err = SetType(obj,name)
      err = calllib('libpetsc', 'KSPSetType', obj.pobj,name);
    end
    function err = SetUp(obj)
      err = calllib('libpetsc', 'KSPSetUp', obj.pobj);
    end
    function err = Solve(obj,b,x)
      err = calllib('libpetsc', 'KSPSolve', obj.pobj,b.pobj,x.pobj);
    end
    function err = SetOperators(obj,A,B,pattern)
      err = calllib('libpetsc', 'KSPSetOperators', obj.pobj,A.pobj,B.pobj,pattern);
    end
    function err = View(obj,viewer)
      err = calllib('libpetsc', 'KSPView', obj.pobj,viewer.pobj);
    end
    function err = Destroy(obj)
      err = calllib('libpetsc', 'KSPDestroy', obj.pobj);
    end
  end
end

 
