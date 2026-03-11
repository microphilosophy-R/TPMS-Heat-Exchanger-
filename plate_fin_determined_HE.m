clear
clc
%% 热边界条件设置
Lhe_sum=[]; Th_sum=[]; Tc_sum=[]; xh_sum=[]; in_nonequil_tube_sum=[]; ans_sum=[]; Ph_sum=[];
wf1='ortho';
wf2='parahyd';
wf3='equil';
wf4='hydrogen';
wf5='helium';
g=9.81;
%热流体
mh=1/1000;%kg/s;
Thin=66.3;
Phin=1130;
xhin_eq=0.1*(exp(-175/Thin)+0.1)^(-1)-7.06*10^(-9)*Thin^3+3.42*10^(-6)*Thin^2-6.2*10^(-5)*Thin-0.00227;
xhin=0.452;
in_nonequil=(xhin_eq-xhin)/xhin_eq;
%冷流体
mratio=2;
mc=mh*mratio;
Tcin=43.5;
Pcin=540;
wc=wf5;
% 催化剂参数设置
eta_bed=0.5048;
dp=600*10^(-6);
rhos=4078.1;
lamdas=0.58;
cps=700;
%% 板翅式换热器尺寸参数设置（），还包括管的材料
h_hf=9.5*10^(-3);%热侧通道翅片高度
s_hf=3.2*10^(-3);%热侧通道翅片间距
thick_t_hf=0.6*10^(-3);%热侧通道翅片厚度
thick_p=1.2*10^(-3);%冷热通道间隔板厚度
x_hf=s_hf-thick_t_hf;%翅片内距
y_hf=h_hf-thick_t_hf;%翅片内高
AR_hf=x_hf/y_hf;%热侧通道的纵横比
Ac_hf_unit=x_hf*y_hf;%单个小通道的自由流动区域
Pw_hf_unit=2*x_hf+2*y_hf;%单个小通道的润湿周长
Dh_hf=2*x_hf*y_hf/(x_hf+y_hf);%当量等效直径

h_cf=9.5*10^(-3);%冷侧通道翅片高度
s_cf=1*10^(-3);%冷侧通道翅片间距
thick_t_cf=0.2*10^(-3);%冷侧通道翅片厚度
x_cf=s_cf-thick_t_cf;%翅片内距
y_cf=h_cf-thick_t_cf;%翅片内高
AR_cf=x_cf/y_cf;%热侧通道的纵横比
Ac_cf_unit=x_cf*y_cf;%单个小通道的自由流动区域
Pw_cf_unit=2*x_cf+2*y_cf;%单个小通道的润湿周长
Dh_cf=2*x_cf*y_cf/(x_cf+y_cf);%当量等效直径《Thermohydraulic Engineering of Plate-Fin Surfaces for Heat Exchangers Subject to Required Dimensions》

number_fin_hf=47;%热侧翅片数量
width_HE=number_fin_hf*s_hf;%换热器有效宽度
number_fin_cf=width_HE/s_cf;%冷侧翅片数量

N_layer_hf=1;%热侧和冷侧各自的层数，假设为相等，依次交错排列
N_layer_cf=2;

Ac_hf=Ac_hf_unit*number_fin_hf*N_layer_hf;
Ac_cf=Ac_cf_unit*number_fin_cf*N_layer_cf;


Kwall=237;%w/(m*K);铜400；铝237
%% 给定温度，压力，浓度初场
L_heat_exchanger_set=0.94;
N_element=10;
Thout=Tcin+10;

Tcout=Thin-5;
xhout_eq=0.1*(exp(-175/Thout)+0.1)^(-1)-7.06*10^(-9)*Thout^3+3.42*10^(-6)*Thout^2-6.2*10^(-5)*Thout-0.00227;
for i=1:N_element
    L_set(i)=L_heat_exchanger_set/N_element;
end
for i=1:N_element+1
    Th(i)=Thin-(Thin-Thout)/N_element*(i-1);
    Ph(i)=Phin;
    xh(i)=xhin+(xhout_eq-xhin_eq)/N_element*(i-1);
    Tc(i)=Tcin+(Tcout-Tcin)/N_element*(i-1);
    Pc(i)=Pcin;
end

error=10;

while error>0.001
    %热侧氢气的物性
    for i=1:N_element+1
        hh_ortho(i)=h2property(Th(i),Ph(i),wf1,'h');
        hh_para(i)=refpropm('H','T',Th(i),'P',Ph(i),wf2);
        hh(i)=(1-xh(i))*hh_ortho(i)+xh(i)*hh_para(i);

        sh_ortho(i)=h2property(Th(i),Ph(i),wf1,'s');
        sh_para(i)=refpropm('S','T',Th(i),'P',Ph(i),wf2);
        sh(i)=(1-xh(i))*sh_ortho(i)+xh(i)*sh_para(i);

        rhoh_ortho(i)=h2property(Th(i),Ph(i),wf1,'r');
        rhoh_para(i)=refpropm('D','T',Th(i),'P',Ph(i),wf2);
        rhoh(i)=(1-xh(i))*rhoh_ortho(i)+xh(i)*rhoh_para(i);

        miuh_ortho(i)=h2property(Th(i),Ph(i),wf1,'m');
        miuh_para(i)=refpropm('V','T',Th(i),'P',Ph(i),wf2);
        miuh(i)=(1-xh(i))*miuh_ortho(i)+xh(i)*miuh_para(i);

        lamdah_ortho(i)=h2property(Th(i),Ph(i),wf1,'l');
        lamdah_para(i)=refpropm('L','T',Th(i),'P',Ph(i),wf2);
        lamdah(i)=(1-xh(i))*lamdah_ortho(i)+xh(i)*lamdah_para(i);

        cph_ortho(i)=h2property(Th(i),Ph(i),wf1,'c');
        cph_para(i)=refpropm('C','T',Th(i),'P',Ph(i),wf2);
        cph(i)=(1-xh(i))*cph_ortho(i)+xh(i)*cph_para(i);
        Cph(i)=cph(i)*mh;
    end
    for i=1:N_element
        Qh(i)=mh*(hh(i)-hh(i+1));
    end
    %冷侧氢气的物性
%     Tc(1)=Tcin;
%     hc(1)=refpropm('H','T',Tc(1),'P',Pc(1),wc);
%     sc(1)=refpropm('S','T',Tc(1),'P',Pc(1),wc);
%     rhoc(1)=refpropm('D','T',Tc(1),'P',Pc(1),wc);
%     miuc(1)=refpropm('V','T',Tc(1),'P',Pc(1),wc);
%     lamdac(1)=refpropm('L','T',Tc(1),'P',Pc(1),wc);
%     cpc(1)=refpropm('C','T',Tc(1),'P',Pc(1),wc);
%     Cpc(1)=cpc(1)*mc;
%     for i=1:N_element
%         hc(i+1)=Qh(N_element+1-i)/mc+hc(i);
%         Tc(i+1)=refpropm('T','P',Pc(i+1),'H',hc(i+1),wc);
%     end
    for i=1:N_element+1
        hc(i)=refpropm('H','T',Tc(i),'P',Pc(i),wc);
        sc(i)=refpropm('S','T',Tc(i),'P',Pc(i),wc);
        rhoc(i)=refpropm('D','T',Tc(i),'P',Pc(i),wc);
        miuc(i)=refpropm('V','T',Tc(i),'P',Pc(i),wc);
        lamdac(i)=refpropm('L','T',Tc(i),'P',Pc(i),wc);
        cpc(i)=refpropm('C','T',Tc(i),'P',Pc(i),wc);
        Cpc(i)=cpc(i)*mc;
    end
    %微元的换热温差计算
    for i=1:N_element
        deltaTh(i)=Th(i)-Tc(N_element+2-i);
        deltaTc(i)=Th(i+1)-Tc(N_element+1-i);
        deltaTmax(i)=max(deltaTh(i),deltaTc(i));
        deltaTmin(i)=min(deltaTh(i),deltaTc(i));
        if deltaTmax(i)>0 && deltaTmin(i)>0
            logdeltaT(i)=(deltaTmax(i)-deltaTmin(i))/log(deltaTmax(i)/deltaTmin(i));
        elseif deltaTmax(i)<0 && deltaTmin(i)<0
            logdeltaT(i)=(deltaTmin(i)-deltaTmax(i))/log(deltaTmin(i)/deltaTmax(i));
        else
            logdeltaT(i)=(Th(i)+Th(i+1))/2-(Tc(N_element+1-i)+Tc(N_element+2-i))/2;
        end
    end
 

    %冷侧换热
    for i=1:N_element+1
        %冷侧换热
       Gc=mc/Ac_cf;
        uc(i)=mc/rhoc(i)/Ac_cf;
        Rec(i)=Gc*Dh_cf/miuc(i);
        Prc(i)=miuc(i)*cpc(i)/lamdac(i);
        Nuc(i)=(0.233*Rec(i)^(-0.48)*(s_cf/h_cf)^(0.192)*(thick_t_cf/h_cf)^(-0.14))*Rec(i)*Prc(i)^(1/3);
%       jc(i)=exp(-0.0264136*(log(Rec(i)))^3+0.55584*(log(Rec(i)))^2-4.09241*log(Rec(i))+6.21681);
%       Nuc(i)=jc(i)*Rec(i)*Prc(i)^(1/3);
ho(i)=Nuc(i)*lamdac(i)/Dh_cf;
 
    end
    % 热侧换热
    for i=1:N_element+1
        uh(i)=mh/rhoh(i)/Ac_hf;
        Rep(i)=dp*uh(i)*rhoh(i)/miuh(i);
        Prh(i)=miuh(i)*cph(i)/lamdah(i);
        Nuw0(i)=20;%取球和圆柱的中位数（Effective heat transfers in packed bed: Experimental and model investigation）
        Num(i)=0.054*Rep(i)*Prh(i);
        Nuww(i)=0.3*Rep(i)^0.75*Prh(i)^(1/3);
        Nuw(i)=Nuw0(i)+1/(1/Num(i)+1/Nuww(i));
        hw(i)=Nuw(i)*lamdah(i)/dp;
        nh=0.28-0.757*log(eta_bed)-0.057*log(lamdas/lamdah(i));%Experimental and numerical investigation of dynamic heat transfer parameters in packed bed
        kr0(i)=lamdah(i)*(lamdas/lamdah(i))^nh;
        Per(i)=1/(0.11+20.64/Rep(i));%颗粒雷诺数，径向佩克雷特数(FIXED BED CATALYTIC REACTOR MODELLING THE HEAT TRANSFER PROBLE)
        kr(i)=kr0(i)+Rep(i)*Prh(i)*lamdah(i)/Per(i);
% K=(8/1.78*(2-(1-2/Dh_hf*dp)^2))^(-1);
% kr(i)=lamdah(i)*(lamdas/lamdah(i)+K*Rep(i)*Prh(i));
% Nuh(i)=(1.3+5/Dh_hf*dp)*(lamdas/lamdah(i))+0.19*Rep(i)^0.75*Prh(i)^0.33;
% hw(i)=Nuh(i)*lamdah(i)/dp;
        Bi(i)=hw(i)*Dh_hf/2/kr(i);
        hi(i)=1/(1/hw(i)+Dh_hf/6/kr(i)*(Bi(i)+3)/(Bi(i)+4));%An improved equation for the overall heat transfer coefficient in packed beds
    end

    %求解长度
     for i=1:N_element

        n_hf=1/s_hf;%热侧每单位宽度的翅片数量
        Ap_hf_unit=2*(1-n_hf*thick_t_hf);%单位宽度里的一次面积《Thermal design of large plate-fin heat exchanger for cryogenic air separation unit based on multiple dynamic equilibriums》《Heat Exchanger Design Handbook 2nd Edition Kuppan》
        As_hf_unit=2*n_hf*(h_hf-thick_t_hf);%单位宽度里的二次面积
        Atot_hf_unit=Ap_hf_unit+As_hf_unit;
        faih=Atot_hf_unit/1;

        n_cf=1/s_cf;%冷侧每单位宽度的翅片数量
        Ap_cf_unit=2*(1-n_hf*2*thick_t_cf);
        As_cf_unit=2*n_cf*(h_cf-thick_t_cf);
        Atot_cf_unit=Ap_cf_unit+As_cf_unit;
        faic=Atot_cf_unit/1;

        m_hf(i)=(2*hi(i)/(lamdah(i)*thick_t_hf))^0.5;
        l_hf=h_hf/2-thick_t_hf;
        eta_fin_hf(i)=(tanh(m_hf(i)*l_hf))/(m_hf(i)*l_hf);

        etas_hf(i)=1-(As_hf_unit/Atot_hf_unit)*(1-eta_fin_hf(i));


        m_cf(i)=(2*ho(i)/(lamdac(i)*thick_t_cf))^0.5;
        l_cf=h_cf/2-thick_t_cf;
        eta_fin_cf(i)=(tanh(m_cf(i)*l_cf))/(m_cf(i)*l_cf);

        etas_cf(i)=1-(As_cf_unit/Atot_cf_unit)*(1-eta_fin_cf(i));
    end

  for i=1:N_element
        U(i)=1/(1/(etas_hf(i)*faih*hi(i))+1/(etas_cf(N_element+1-i)*faic*ho(N_element+1-i))+thick_p/Kwall);
        A(i)=L_set(i)*width_HE;
        Qc(i)=U(i)*logdeltaT(i)*A(i);
%          Qh(i)=U(i)*logdeltaT(i)*A(i)*2;
    end
    Lt=sum(L_set);

    %冷侧压降
    Pc_new(1)=Pc(1);
    for i=1:N_element
fc(i)=0.029*Rec(i)^(-0.09)*(s_cf/h_cf)^(-0.169)*(thick_t_cf/h_cf)^(0.034);
deltaPc(i)=2*rhoc(i)*uc(i)*L_set(N_element+1-i)*fc(i)/Dh_cf;

        Pc_new(i+1)=Pc_new(i)-deltaPc(i);
    end
    %热侧压降
    Ph_new(1)=Ph(1);
    for i=1:N_element
        A1=1+2/(3*(Dh_hf/dp)*(1-eta_bed));
        B1=(1.15*(dp/Dh_hf)^2+0.87)^2;
deltaPh(i)=(L_set(i)*rhoh(i)*uh(i)^2/dp*(154*A1^2/Rep(i)*(1-eta_bed)^2/eta_bed^3+A1/B1*(1-eta_bed)/eta_bed^3))/1000;

        Ph_new(i+1)=Ph_new(i)-deltaPh(i);
    end

    %能量平衡
    for i=1:N_element
    hc(i+1)=hc(i)+Qc(N_element+1-i)/mc;
    hh(i+1)=hh(i)-Qc(i)/mh; 
%         Qh(i)=(xh(i+1)-xh(i))*mh*(hh_ortho(i)-hh_para(i+1));
    end

    [xh_new]=first_order_rate(L_set,Th,Ph,xh,mh,Ac_hf);

    %% 迭代计算
    for i=1:N_element+1
        Tc_new(i)=refpropm('T','P',Pc(i),'H',hc(i),wc);
    end
    for i=1:N_element+1
        Tc_new(i)=refpropm('T','P',Pc(i),'H',hc(i),wc);
        T_h2=Th(i);
        P_h2=Ph(i);
        h_h2=hh(i);
        x_h2=xh_new(i);

              %         Th_new(i)=fminbnd(@(T_h2)solveT(T_h2,P_h2,h_h2,x_h2),13.957,300);
        Th_new(i)=fsolve(@(T_h2)solveT(T_h2,P_h2,h_h2,x_h2),Thin);
    end


    error=0;
    for i=1:N_element+1
        error=error+(Tc(i)-Tc_new(i))^2+(Ph(i)-Ph_new(i))^2+(Pc(i)-Pc_new(i))^2+(Th(i)-Th_new(i))^2+(xh(i)-xh_new(i))^2;
    end
    alpha=0.1;
    for i=1:N_element+1
        Th(i)=Th(i)+alpha*(Th_new(i)-Th(i));
        Tc(i)=Tc(i)+alpha*(Tc_new(i)-Tc(i));
                Ph(i)=Ph(i)+alpha*(Ph_new(i)-Ph(i));
        Pc(i)=Pc(i)+alpha*(Pc_new(i)-Pc(i));
        xh(i)=xh(i)+alpha*(xh_new(i)-xh(i));
    end
end
% 
% 
function f=solveT(T_h2,P_h2,h_h2,x_h2)

M_H2=2.0159; %g/mol
R=8.314; T_ref=300;  P_ref=1000;
Rg_H2=R/M_H2*1000;    %J/kg-k
%% 以para氢的焓熵值为基准，计算normal氢的补偿值
%计算normal氢对para氢在参考状态的补偿
h_np_offset_ref=(3.41607-3.39382)*Rg_H2*T_ref;
s_np_offset_ref=(15.72626-14.32057)*Rg_H2;
%normal氢在参考状态下补偿后的焓熵值
h_para_ref=refpropm('H','T',T_ref,'P',P_ref,'parahyd');
s_para_ref=refpropm('S','T',T_ref,'P',P_ref,'parahyd');
h_normal_ref=h_np_offset_ref+h_para_ref;
s_normal_ref=s_np_offset_ref+s_para_ref;
%计算normal氢的焓熵补偿值
h_normal_ref_raw=refpropm('H','T',T_ref,'P',P_ref,'hydrogen');
s_normal_ref_raw=refpropm('S','T',T_ref,'P',P_ref,'hydrogen');
h_offset=h_normal_ref-h_normal_ref_raw;
s_offset=s_normal_ref-s_normal_ref_raw;
% 从Refprop直接读取normal氢和para氢的物性数据
h_normal_ori=refpropm('H','T',T_h2,'P',P_h2,'hydrogen');
s_normal_ori=refpropm('S','T',T_h2,'P',P_h2,'hydrogen');
h_para_ori=refpropm('H','T',T_h2,'P',P_h2,'parahyd');
s_para_ori=refpropm('S','T',T_h2,'P',P_h2,'parahyd');

% 统一基准值确定normal氢和para氢正确的物性数据
h_normal=h_normal_ori+h_offset;
s_normal=s_normal_ori+s_offset;

h_para=h_para_ori;
s_para=s_para_ori;
%% 计算正氢的物性
h_ortho=(h_normal-0.25*h_para)*4/3;
s_ortho=(s_normal+R*(0.75*log(0.75)+0.25*log(0.25))-0.25*s_para)*4/3;

f=h_h2-((1-x_h2)*h_ortho+x_h2*h_para);

end
