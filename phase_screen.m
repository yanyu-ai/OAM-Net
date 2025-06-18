function phi=phase_screen (lamda,z,Cn2)
N=1001;%傅立叶个数
%lamda=1.550e-6;
k=2*pi/lamda;%光束波长和波数
% z=1e3;%传输距离
Numz=20;%把传播距离分成Numz段
dz=z/Numz;%每段的距离4
% dz=50;
% Numz=z/dz;
L=70e-2;%窗口宽度
% m=[-N/2:N/2-1];
m1=1:N;
%  m=[-N+1:0];
df=1/L;
[fx,fy]=meshgrid(m1*df);%空间频率
fr=sqrt(fx.^2+fy.^2);
kx=2*pi*fx;ky=2*pi*fy;kr=2*pi*fr;%空间波数
%Cn2=1e-15;%refractive-index structure parameter;
L0=20;%outer scale of turbulence
l0=5e-3;%inner scale of turbulence
kl=3.3/l0;
% f=exp(-(kr.^2)/kl^2).*(1+1.802*sqrt(kr.^2)/kl-0.254*(sqrt(kr.^2)/kl)^(7/6));
% phin=0.033*Cn2*(kr.^2+(2*pi/L0)^2).^(-11/6).*f;%Kolmogorov spectrum
% phi0=2*pi*k^2*dz*phin;
% Gau=normrnd(0,1/sqrt(2),N)+i*normrnd(0,1/sqrt(2),N);%产生均值为0，方差为1的复随机矩阵
% phi=fft2(Gau*(2*pi/a).*sqrt(phi0),N,N);%经高斯滤波后，逆傅里叶变换得到相位屏fy

r0=0.185*(lamda^2/(dz*Cn2))^(3/5);
phi0=2*pi/L*0.0241*r0^(-5/6)*(fr.^2+1/L0^2).^(-11/12);
Gau=(randn(N,N)+sqrt(-1)*randn(N,N))/sqrt(2);%产生均值为0，方差为1的复随机矩阵
% load Gau.mat;
% Gau=fftshift(fft2(Gau));%经高斯滤波后，逆傅里叶变换得到相位屏fy
% phi=ifft2(ifftshift(Gau.*phi0));
phi=(fft2(Gau.*phi0));
phi=abs(phi);
% figure
% mesh(abs(phi))；