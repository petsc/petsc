

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% getalp.m %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% function [alp,lba,uba,ier]=getalp(alpu,alpo,gTp,pTGp);
% get minimizer alp in [alpu,alpo] for a univariate quadratic
%	q(alp)=alp*gTp+0.5*alp^2*pTGp
% lba	lower bound active
% uba	upper bound active
%
% ier	 0 (finite minimizer) 
%	 1 (unbounded minimum)
%
function [alp,lba,uba,ier]=getalp(alpu,alpo,gTp,pTGp);

lba=0;
uba=0;

% determine unboundedness
ier=0;
if alpu==-inf & ( pTGp<0 | (pTGp==0 & gTp>0) ),
  ier=1; lba=1;
end;
if alpo==inf & (pTGp<0 | (pTGp==0 & gTp<0) ),
  ier=1; uba=1; 
end;
if ier, alp=NaN; return; end;
       
% determine activity
if pTGp==0 & gTp==0, 
  alp=0;
elseif pTGp<=0,
  % concave case minimal at a bound
  if alpu==-inf,     lba=0;
  elseif alpo== inf, lba=1;
  else               lba = (2*gTp+(alpu+alpo)*pTGp>0); 
  end;
  uba = ~lba;
else
  alp=-gTp/pTGp;          % unconstrained optimal step
  lba = (alp <= alpu);    % lower bound active
  uba = (alp >= alpo);    % upper bound active
end;

if lba, alp=alpu; end;
if uba, alp=alpo; end;

    
% print?
if abs(alp)==inf, gTp,pTGp,alpu,alpo,alp,lba,uba,ier, end;   
