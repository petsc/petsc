classdef PetscSNES < PetscObject
  methods
    function obj = PetscSNES(pid,flg)
      if (nargin > 1) 
        %  PetscSNES(pid,'pobj') uses an already existing PETSc SNES object
        obj.pobj = pid;
        return
      end
      [err,obj.pobj] = calllib('libpetsc', 'SNESCreate', 0,0);
    end
    function err = SetType(obj,name)
      err = calllib('libpetsc', 'SNESSetType', obj.pobj,name);
    end
    function err = SetDM(obj,da)
      err = calllib('libpetsc', 'SNESSetDM', obj.pobj,da.pobj);
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
    function err = VISetVariableBounds(obj,xl,xb)
      err = calllib('libpetsc', 'SNESVISetVariableBounds', obj.pobj,xl.pobj,xb.pobj);
    end
    function err = SetFunction(obj,f,func,arg)
      if (nargin < 4) 
        arg = 0;
      end
      err = calllib('libpetsc', 'SNESSetFunctionMatlab', obj.pobj,f.pobj,func,arg);
    end
    function err = SetJacobian(obj,A,B,func,arg)
      if (nargin < 5) 
        arg = 0;
      end
      err = calllib('libpetsc', 'SNESSetJacobianMatlab', obj.pobj,A.pobj,B.pobj,func,arg);
    end
    function err = View(obj,viewer)
      err = calllib('libpetsc', 'SNESView', obj.pobj,viewer.pobj);
    end
    function err = Destroy(obj)
      err = calllib('libpetsc', 'SNESDestroy', obj.pobj);
    end
  end
end

 
