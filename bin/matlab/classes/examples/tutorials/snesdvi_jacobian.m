function [flg,err] = snesdvi_jacobian(snes,xsol,A,B,user)
%
%  Jacobain for minimal surface are problem as written in
%  snes/src/examples/tests/ex8.c
%
err = 0;
flg = PetscMat.SAME_NONZERO_PATTERN;

mx = user.mx; my = user.my;
hx = 1/(mx+1); hy = 1/(my+1);
hydhx = hy/hx; hxdhy = hx/hy;

x = user.dm.VecGetArray(xsol);

for (i = 1:mx)
  for (j = 1:my)        
    xc = x(i,j);
    xlt = xc; xr = xc;
    xrb = xc; xb = xc;
    xl = xc; xt  = xc;
        
    if (i == 1) % left side
      xl = user.left(j+1);
      xlt = user.left(j+2);
    else
      xl = x(i-1,j);
    end
        
    if (j == 1) % bottom side
      xb = user.bottom(i+1);
      xrb = user.bottom(i+2);
    else
      xb = x(i,j-1);
    end
      
    if (i == mx) %right side
      xr  = user.right(j+1);
      xrb = user.right(j);
    else
      xr = x(i+1,j);
    end
        
    if (j == my) % top side
      xt  = user.top(i+1);
      xlt = user.top(i);
    else
      xt = x(i,j+1);
    end
        
    if (i > 1 & j<my)
      xlt = x(i-1,j+1);
    end
        
    if (j > 1 & i<mx)
      xrb = x(i+1,j-1);
    end
        
    d1 = (xc-xl)/hx;
    d2 = (xc-xr)/hx;
    d3 = (xc-xt)/hy;
    d4 = (xc-xb)/hy;
    d5 = (xrb-xr)/hy;
    d6 = (xrb-xb)/hx;
    d7 = (xlt-xl)/hy;
    d8 = (xlt-xt)/hx;

    f1 = sqrt( 1.0 + d1*d1 + d7*d7);
    f2 = sqrt( 1.0 + d1*d1 + d4*d4);
    f3 = sqrt( 1.0 + d3*d3 + d8*d8);
    f4 = sqrt( 1.0 + d3*d3 + d2*d2);
    f5 = sqrt( 1.0 + d2*d2 + d5*d5);
    f6 = sqrt( 1.0 + d4*d4 + d6*d6);
        
    hl = (-hydhx*(1.0+d7*d7)+d1*d7)/(f1*f1*f1)+ ...
         (-hydhx*(1.0+d4*d4)+d1*d4)/(f2*f2*f2);
    hr = (-hydhx*(1.0+d5*d5)+d2*d5)/(f5*f5*f5)+ ...
         (-hydhx*(1.0+d3*d3)+d2*d3)/(f4*f4*f4);
    ht = (-hxdhy*(1.0+d8*d8)+d3*d8)/(f3*f3*f3)+ ...
         (-hxdhy*(1.0+d2*d2)+d2*d3)/(f4*f4*f4);
    hb = (-hxdhy*(1.0+d6*d6)+d4*d6)/(f6*f6*f6)+ ...
         (-hxdhy*(1.0+d1*d1)+d1*d4)/(f2*f2*f2);

    hbr = -d2*d5/(f5*f5*f5) - d4*d6/(f6*f6*f6);
    htl = -d1*d7/(f1*f1*f1) - d3*d8/(f3*f3*f3);

    hc = hydhx*(1.0+d7*d7)/(f1*f1*f1) + hxdhy*(1.0+d8*d8)/(f3*f3*f3) + ...
         hydhx*(1.0+d5*d5)/(f5*f5*f5) + hxdhy*(1.0+d6*d6)/(f6*f6*f6) + ...
         (hxdhy*(1.0+d1*d1)+hydhx*(1.0+d4*d4)-2*d1*d4)/(f2*f2*f2) + ...
         (hxdhy*(1.0+d2*d2)+hydhx*(1.0+d3*d3)-2*d2*d3)/(f4*f4*f4);
        
    hl = hl/2.0; hr  = hr/2.0;  ht  = ht/2.0; 
    hb = hb/2.0; hbr = hbr/2.0; htl = htl/2.0;  hc = hc/2.0;
         
         
    k = 1;
    v = []; col = [];
    row.i = i; row.j = j;
    if (j>1)
      v(k) = hb; 
      col(k).i = i; col(k).j = j-1;
      k=k+1;
    end

    if (j>1 & i < mx)
      v(k)=hbr;
      col(k).i = i+1; col(k).j = j-1;
      k=k+1;
    end

    if (i>1)
      v(k)= hl; 
      col(k).i = i-1; col(k).j = j;
      k = k + 1;
    end

    v(k)= hc; 
    col(k).i = i; col(k).j = j;
    k = k + 1;

    if (i < mx )
      v(k)= hr; 
      col(k).i = i+1;
      col(k).j = j;
      k = k + 1;
    end

    if (i>1 & j < my )
      v(k)= htl; 
      col(k).i = i-1; col(k).j = j+1;
      k = k + 1;
    end

    if (j < my )
      v(k)= ht; 
      col(k).i = i; col(k).j = j+1;
      k = k + 1;
    end
         
    A.SetValuesStencil(row,col,v);
  end
end

err = B.AssemblyBegin(PetscMat.FINAL_ASSEMBLY);
err = B.AssemblyEnd(PetscMat.FINAL_ASSEMBLY);
err = A.AssemblyBegin(PetscMat.FINAL_ASSEMBLY);
err = A.AssemblyEnd(PetscMat.FINAL_ASSEMBLY);
