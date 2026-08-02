function generate_reference
root=fileparts(mfilename('fullpath'));
raw=readmatrix(fullfile(root,'35PN230.tab'),'FileType','text');
raw=raw(all(isfinite(raw),2),:);
H=raw(:,1); B=raw(:,2);
[B,H,slope,passes]=prepareCurve(B,H);
assert(passes==0,'35PN230 reference unexpectedly required smoothing.');
samples=[B;(B(1:end-1)+B(2:end))/2;B(end)+0.2572;-1.66];
fid=fopen(fullfile(root,'reference_samples.txt'),'wt');
assert(fid>=0,'Could not write reference_samples.txt.');
cleanup=onCleanup(@() fclose(fid));
fprintf(fid,'# B_T H_A_per_m dH_dB H_over_B dv_dB2\n');
for index=1:numel(samples)
    [h,dh,v,dv]=evaluateCurve(samples(index),B,H,slope);
    fprintf(fid,'%.17g %.17g %.17g %.17g %.17g\n', ...
        samples(index),h,dh,v,dv);
end
clear cleanup;

syntheticB=[0;1;2;3];
syntheticH=[0;1;1.01;10];
[processedB,processedH,processedSlope,passes]=prepareCurve(syntheticB,syntheticH);
assert(passes>0,'Synthetic curve did not exercise smoothing.');
fid=fopen(fullfile(root,'smoothing_reference.txt'),'wt');
assert(fid>=0,'Could not write smoothing_reference.txt.');
cleanup=onCleanup(@() fclose(fid));
fprintf(fid,'# smoothing_passes B_T H_A_per_m dH_dB\n');
fprintf(fid,'%d\n',passes);
for index=1:numel(processedB)
    fprintf(fid,'%.17g %.17g %.17g\n',processedB(index), ...
        processedH(index),processedSlope(index));
end
clear cleanup;
end

function [B,H,slope,smoothCount]=prepareCurve(B,H)
n=numel(B);
smoothCount=0;
while true
    M=zeros(n); rhs=zeros(n,1);
    d=B(2)-B(1);
    M(1,1)=4/d; M(1,2)=2/d;
    rhs(1)=6*(H(2)-H(1))/d^2;
    d=B(n)-B(n-1);
    M(n,n)=4/d; M(n,n-1)=2/d;
    rhs(n)=6*(H(n)-H(n-1))/d^2;
    for index=2:n-1
        left=B(index)-B(index-1);
        right=B(index+1)-B(index);
        M(index,index-1)=2/left;
        M(index,index)=4*(left+right)/(left*right);
        M(index,index+1)=2/right;
        rhs(index)=6*(H(index)-H(index-1))/left^2+ ...
            6*(H(index+1)-H(index))/right^2;
    end
    slope=M\rhs;
    curveOK=true;
    for index=2:n
        segment=B(index)-B(index-1);
        d0=slope(index-1); d1=slope(index);
        h0=H(index-1); h1=H(index);
        c0=d0;
        c1=-2*(2*d0*segment+d1*segment+3*h0-3*h1)/segment^2;
        c2=3*(d0*segment+d1*segment+2*h0-2*h1)/segment^3;
        x0=-1; x1=-1;
        discriminant=c1^2-4*c0*c2;
        if c2==0
            if c1~=0, x0=-c0/c1; end
        elseif discriminant>0
            root=sqrt(discriminant);
            x0=-(c1+root)/(2*c2);
            x1=(-c1+root)/(2*c2);
        end
        if (x0>=0 && x0<=segment) || (x1>=0 && x1<=segment)
            curveOK=false;
        end
    end
    if curveOK, return; end
    nextB=B; nextH=H;
    for index=2:n-1
        nextB(index)=(B(index-1)+B(index)+B(index+1))/3;
        nextH(index)=(H(index-1)+H(index)+H(index+1))/3;
    end
    B=nextB; H=nextH;
    smoothCount=smoothCount+1;
    assert(smoothCount<=1024,'Smoothing did not converge.');
end
end

function [h,dh,v,dv]=evaluateCurve(fieldB,B,H,slope)
b=abs(fieldB);
if b==0
    h=0; dh=slope(1); v=slope(1); dv=0;
    return;
end
if b>B(end)
    h=H(end)+slope(end)*(b-B(end));
    dh=slope(end);
else
    index=find(b>=B(1:end-1) & b<=B(2:end),1);
    segment=B(index+1)-B(index);
    z=(b-B(index))/segment; z2=z*z;
    h=(1-3*z2+2*z2*z)*H(index)+ ...
        z*(1-2*z+z2)*segment*slope(index)+ ...
        z2*(3-2*z)*H(index+1)+ ...
        z2*(z-1)*segment*slope(index+1);
    dh=6*z*(z-1)*H(index)/segment+ ...
        (1-4*z+3*z*z)*slope(index)+ ...
        6*z*(1-z)*H(index+1)/segment+ ...
        z*(3*z-2)*slope(index+1);
end
v=h/b;
dv=0.5*(dh/b^2-h/b^3);
end
