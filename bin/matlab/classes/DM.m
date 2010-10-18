classdef DM < PetscObject
  properties (Constant)
    STENCIL_STAR = 0;
    STENCIL_BOX = 1;

    NONPERIODIC = 0;
    XPERIODIC   = 1;
    YPERIODIC   = 2;
    XYPERIODIC  = 3;
    XYZPERIODIC = 4;
    XZPERIODIC  = 5;
    YZPERIODIC  = 6;
    ZPERIODIC   = 7;
    XYZGHOSTED  = 8;

    Q0 = 0;
    Q1 = 1;
  end
  methods
    function obj = DM(pid,flg)
      if (nargin > 1) 
        %  DM(pid,'pobj') uses an already existing PETSc DM object
        obj.pobj = pid;
        return
      end
      [err,obj.pobj] = calllib('libpetsc', 'DMCreate', 0,0);
    end
    function err = SetType(obj,name)
      err = calllib('libpetsc', 'DMSetType', obj.pobj,name);
    end
    function err = SetDim(obj,dim)
      err = calllib('libpetsc', 'DMDASetDim', obj.pobj,dim);
    end
    function err = SetSizes(obj,sizes)
      err = calllib('libpetsc', 'DMDASetSizes', obj.pobj,sizes(1),sizes(2),sizes(3));
    end
    function err = SetVecType(obj,vtype)
      err = calllib('libpetsc', 'DMSetVecType', obj.pobj,vtype);
    end
    function err = SetPeriodicity(obj,periodicity)
      err = calllib('libpetsc', 'DMDASetPeriodicity', obj.pobj,periodicity);
    end
    function err = SetDof(obj,dof)
      err = calllib('libpetsc', 'DMDASetDof', obj.pobj,dof);
    end
    function err = SetStencilWidth(obj,width)
      err = calllib('libpetsc', 'DMDASetStencilWidth', obj.pobj,width);
    end
    function err = SetStencilType(obj,type)
      err = calllib('libpetsc', 'DMDASetStencilType', obj.pobj,type);
    end
    function err = SetFromOptions(obj)
      err = calllib('libpetsc', 'DMSetFromOptions', obj.pobj);
    end
    function err = SetUp(obj)
      err = calllib('libpetsc', 'DMSetUp', obj.pobj);
    end
    function err = View(obj,viewer)
      err = calllib('libpetsc', 'DMView', obj.pobj,viewer.pobj);
    end
    function err = Destroy(obj)
      err = calllib('libpetsc', 'DMDestroy', obj.pobj);
    end
  end
end

 
