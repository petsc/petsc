classdef PetscDM < PetscObject
  properties(Constant)
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
  properties
      ndim = [];
      M    = 0;
      N    = 0;
      P    = 0;
      dof  = 0;
      s    = 0;
  end
  methods
    function obj = PetscDM(pid,flg)
      if (nargin > 1) 
        %  PetscDM(pid,'pobj') uses an already existing PETSc DM object
        obj.pobj = pid;
        return
      end
      [err,obj.pobj] = calllib('libpetsc', 'DMCreate', 0,0);PetscCHKERRQ(err);
    end
    function err = SetType(obj,name)
      err = calllib('libpetsc', 'DMSetType', obj.pobj,name);PetscCHKERRQ(err);
    end
    function err = SetFunction(obj,func)
      err = calllib('libpetsc', 'DMSetFunctionMatlab', obj.pobj,func);PetscCHKERRQ(err);
    end
    function err = SetJacobian(obj,func)
      err = calllib('libpetsc', 'DMSetJacobianMatlab', obj.pobj,func);PetscCHKERRQ(err);
    end
    function err = SetDim(obj,dim)
      err = calllib('libpetsc', 'DMDASetDim', obj.pobj,dim);PetscCHKERRQ(err);
    end
    function err = SetSizes(obj,sizes)
      err = calllib('libpetsc', 'DMDASetSizes', obj.pobj,sizes(1),sizes(2),sizes(3));PetscCHKERRQ(err);
    end
    function err = SetVecType(obj,vtype)
      err = calllib('libpetsc', 'DMSetVecType', obj.pobj,vtype);PetscCHKERRQ(err);
    end
    function err = SetPeriodicity(obj,periodicity)
      err = calllib('libpetsc', 'DMDASetPeriodicity', obj.pobj,periodicity);PetscCHKERRQ(err);
    end
    function err = SetDof(obj,dof)
      err = calllib('libpetsc', 'DMDASetDof', obj.pobj,dof);PetscCHKERRQ(err);
    end
    function err = SetStencilWidth(obj,width)
      err = calllib('libpetsc', 'DMDASetStencilWidth', obj.pobj,width);PetscCHKERRQ(err);
    end
    function err = SetStencilType(obj,type)
      err = calllib('libpetsc', 'DMDASetStencilType', obj.pobj,type);PetscCHKERRQ(err);
    end
    function err = SetFromOptions(obj)
      err = calllib('libpetsc', 'DMSetFromOptions', obj.pobj);PetscCHKERRQ(err);
    end
    function [v,err] = CreateGlobalVector(obj)
      [err,pidv] = calllib('libpetsc', 'DMCreateGlobalVector', obj.pobj,0);PetscCHKERRQ(err);
      v = PetscVec(pidv,'pobj');
    end
    function [v,err] = GetMatrix(obj,name)
      [err,name,pidv] = calllib('libpetsc', 'DMGetMatrix', obj.pobj,name,0);PetscCHKERRQ(err);
      v = PetscMat(pidv,'pobj');
    end
    function err = SetUp(obj)
      err = calllib('libpetsc', 'DMSetUp', obj.pobj);PetscCHKERRQ(err);PetscCHKERRQ(err);
    end
    function err = View(obj,viewer)
      err = calllib('libpetsc', 'DMView', obj.pobj,viewer.pobj);PetscCHKERRQ(err);
    end
    function err = Destroy(obj)
      err = calllib('libpetsc', 'DMDestroy', obj.pobj);PetscCHKERRQ(err);
    end
    function [ndim,M,N,P,dof,s,err] = GetInfo(obj)
      [err,ndim,M,N,P,m,n,p,dof,s,w,st] = calllib('libpetsc','DMDAGetInfo',obj.pobj,0,0,0,0,0,0,0,0,0,0,0);PetscCHKERRQ(err);
    end
    function [obj] = SetInfo(obj)
      [obj.ndim,obj.M,obj.N,obj.P,obj.dof,obj.s,err] = obj.GetInfo();
    end
    function [dmvec,err] = VecGetArray(obj,vec)
      dmvec = PetscVec(vec.pobj,'pobj');
      dmvec.SetVecfromDM(1);
      if (isempty(obj.ndim))
         %% Temporary hack to fill info in the original DM,this should go to
         %% some DM function and not VecGetArray
         obj = obj.SetInfo();
      end
      dmvec = dmvec.SetDM(obj);
    end
  end
end

 
