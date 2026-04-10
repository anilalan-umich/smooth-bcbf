%% main.m
% Smoothed BCBF on a double integrator with a SMOOTH-SATURATED backup.
%
% Pipeline:
%   1. Setup
%   2. Vectorised batch RK4 over a grid: slice values b_i(x) and gradients
%   3. Estimate M, r, d on the grid
%   4. Compute theta*
%   5. Compute btil_theta and L_Fb btil_theta on the grid
%   6. Compute mu, L_min, gamma*
%   7. Closed-loop simulations (sweep over gamma, classical and smoothed)
%   8. Plots:
%       Fig.1 - sets only
%       Fig.2 - classical BCBF trajectories for gamma in {0.1, 0.5, 1, 5}
%       Fig.3 - smoothed BCBF trajectories for the same gamma values
%
% Requires: Optimization Toolbox (quadprog), Control Toolbox (lyap).

clear; clc; 
close all;

%% 1. Setup ----------------------------------------------------------------
par.A    = [0 1; 0 0];
par.Bu   = [0; 1];
par.umax = 2;

par.tau_h = 2;
par.c_h   = 1;
par.dh    = [1; par.tau_h];
par.h_fun = @(y) y(1,:) + par.tau_h*y(2,:) + par.c_h;

par.Kb  = [1 2];

par.sat   = @(r) par.umax * tanh(r / par.umax);
par.sat_d = @(r) 1 - tanh(r / par.umax).^2;

Acl_lin = par.A - par.Bu*par.Kb;
par.P   = lyap(Acl_lin', eye(2));

par.rho = 0.1;

par.T    = 2;
par.dtau = 0.1;
par.tau  = 0:par.dtau:par.T;
par.N    = numel(par.tau);

par.n_sub = 5;

fprintf('=== Smoothed BCBF with smooth-saturated backup ===\n');
fprintf('  T = %.2f, dtau = %.2f, N = %d, rho = %.3f, umax = %.2f\n\n', ...
        par.T, par.dtau, par.N, par.rho, par.umax);

%% 2. Batch RK4 over grid: slice values and gradients ---------------------
ngrid = 561;
x1g = linspace(-4.2, 4, ngrid);
x2g = linspace(-4, 4, ngrid);
[X1, X2] = meshgrid(x1g, x2g);
ngp = numel(X1);

Xc  = [X1(:)'; X2(:)'];
S11 = ones(1, ngp);  S12 = zeros(1, ngp);
S21 = zeros(1, ngp); S22 = ones(1, ngp);

bv_all  = zeros(par.N, ngp);
db1_all = zeros(par.N, ngp);
db2_all = zeros(par.N, ngp);
yt1     = zeros(par.N, ngp);
yt2     = zeros(par.N, ngp);

yt1(1,:) = Xc(1,:);
yt2(1,:) = Xc(2,:);
bv_all(1,:)  = par.h_fun(Xc);
db1_all(1,:) = S11 + par.tau_h * S21;
db2_all(1,:) = S12 + par.tau_h * S22;

dt_int = par.dtau / par.n_sub;
fprintf('Integrating %d sensitivity trajectories ... ', ngp);
tic;
for i = 1:par.N-1
    for ss = 1:par.n_sub
        [Xc, S11, S12, S21, S22] = rk4_step(Xc, S11, S12, S21, S22, dt_int, par);
    end
    yt1(i+1,:) = Xc(1,:);
    yt2(i+1,:) = Xc(2,:);
    bv_all(i+1,:)  = par.h_fun(Xc);
    db1_all(i+1,:) = S11 + par.tau_h * S21;
    db2_all(i+1,:) = S12 + par.tau_h * S22;
end
fprintf('done (%.1f s)\n', toc);

y1 = yt1(end,:); y2 = yt2(end,:);
bv_all(end,:) = par.rho - ( par.P(1,1)*y1.^2 ...
                          + 2*par.P(1,2)*y1.*y2 ...
                          + par.P(2,2)*y2.^2 );
v1 = par.P(1,1)*y1 + par.P(1,2)*y2;
v2 = par.P(2,1)*y1 + par.P(2,2)*y2;
db1_all(end,:) = -2*(S11.*v1 + S21.*v2);
db2_all(end,:) = -2*(S12.*v1 + S22.*v2);

[bhat, ~] = min(bv_all, [], 1);
in_Bhat   = bhat >= 0;

v_grid = -X1(:)' - 2*X2(:)';
Fb1    = X2(:)';
Fb2    = par.sat(v_grid);

LFb_all = db1_all .* Fb1 + db2_all .* Fb2;

%% 3. Estimate M, r, d on the grid ----------------------------------------
eps_tube = par.rho;

in_Tube = in_Bhat & (bhat <= eps_tube);
M_est = max( max( abs(LFb_all(:, in_Tube)) ) );

tol_boundary = 0.005;
on_bnd = in_Bhat & (bhat <= tol_boundary);

tol_active = 0.01;
gap_all   = bv_all - bhat;
active    = gap_all <= tol_active;
inactive  = ~active;

LFb_b   = LFb_all(:, on_bnd);
act_b   = active(:, on_bnd);
r0      = min( LFb_b(act_b) );

gap_b   = gap_all(:, on_bnd);
inact_b = inactive(:, on_bnd);
if any(inact_b(:))
    d0 = min( gap_b(inact_b) );
else
    d0 = NaN;
end

r_est = r0/2;
d_est = d0/2;

fprintf('\nEstimated bounds (over the grid):\n');
fprintf('   M = %.4f, r = %.4f, d = %.4f, eps = %.4f, N = %d\n', ...
        M_est, r_est, d_est, eps_tube, par.N);

%% 4. theta* ---------------------------------------------------------------
theta_first = log(par.N) / eps_tube;
theta_core  = (1/d_est) * log( par.N * (r_est + M_est) / r_est );
theta_star  = max(theta_first, theta_core);
theta_use   = ceil(theta_star * 1.05);

fprintf('   theta* = %.2f, using theta = %d\n', theta_star, theta_use);

%% 5. btil_theta and L_Fb btil_theta on grid ------------------------------
Z   = -theta_use * bv_all;
zmx = max(Z, [], 1);
ez  = exp(Z - zmx);
Sm  = sum(ez, 1);
W   = ez ./ Sm;
btil = -(1/theta_use) * (log(Sm) + zmx);

dbtil1 = sum(W .* db1_all, 1);
dbtil2 = sum(W .* db2_all, 1);

LFb_btil = dbtil1 .* Fb1 + dbtil2 .* Fb2;

in_Btil  = in_Bhat & (btil >= 0);
in_TubeS = in_Btil & (btil <= eps_tube);
in_ExtS  = in_Btil & (btil >  eps_tube);

%% 6. mu, L_min, gamma* ---------------------------------------------------
if any(in_TubeS), mu = min(LFb_btil(in_TubeS)); else, mu = NaN; end
if any(in_ExtS),  L_min = min(LFb_btil(in_ExtS)); else, L_min = 0; end

K_required = max(1e-6, -L_min);
gamma_star = K_required / eps_tube;

fprintf('\n   mu = %+g, L_min = %+g, gamma* = %g\n', mu, L_min, gamma_star);

%% 7. Closed-loop simulations: sweep over gamma ---------------------------
x0   = [-2.6; 3];
[bv0, ~, ~] = slice_vals_point(x0, par);
fprintf('\nIC x0 = (%.2f, %.2f),  bhat(x0) = %+.4f\n', ...
        x0(1), x0(2), min(bv0));

Tsim  = 6;
dt    = 0.005;
tgrid = 0:dt:Tsim;
nT    = numel(tgrid);
kdes  = @(t,x) -par.umax;

gamma_list = [0.01, 0.1, 0.25, 0.5, 1];
nG = numel(gamma_list);

% Pre-allocate storage for both filters: Xc{g}, Uc{g}, Fc{g}
Xc_all = cell(nG,1); Uc_all = cell(nG,1); Fc_all = cell(nG,1);
Xs_all = cell(nG,1); Us_all = cell(nG,1); Fs_all = cell(nG,1);

for gi = 1:nG
    g = gamma_list(gi);
    fprintf('\n--- Simulating gamma = %g ---\n', g);

    % Classical (nonsmooth)
    Xs_c = zeros(2, nT); Us_c = zeros(1, nT); Fs_c = zeros(1, nT);
    Xs_c(:,1) = x0;
    for k = 1:nT-1
        [u, flag, ~] = sf_classical(Xs_c(:,k), kdes(tgrid(k),Xs_c(:,k)), g, par);
        if isnan(u) || flag <= 0
            Us_c(k) = par.sat(-par.Kb * Xs_c(:,k));
            Fs_c(k) = 1;
        else
            Us_c(k) = u;
        end
        Xs_c(:,k+1) = Xs_c(:,k) + dt*(par.A*Xs_c(:,k) + par.Bu*Us_c(k));
    end
    Us_c(nT) = Us_c(nT-1);
    Xc_all{gi} = Xs_c; Uc_all{gi} = Us_c; Fc_all{gi} = Fs_c;
    fprintf('   classical: %d / %d steps infeasible\n', nnz(Fs_c), nT-1);

    % Smoothed
    Xs_s = zeros(2, nT); Us_s = zeros(1, nT); Fs_s = zeros(1, nT);
    Xs_s(:,1) = x0;
    for k = 1:nT-1
        [u, flag, ~] = sf_smoothed(Xs_s(:,k), kdes(tgrid(k),Xs_s(:,k)), g, theta_use, par);
        if isnan(u) || flag <= 0
            Us_s(k) = par.sat(-par.Kb * Xs_s(:,k));
            Fs_s(k) = 1;
        else
            Us_s(k) = u;
        end
        Xs_s(:,k+1) = Xs_s(:,k) + dt*(par.A*Xs_s(:,k) + par.Bu*Us_s(k));
    end
    Us_s(nT) = Us_s(nT-1);
    Xs_all{gi} = Xs_s; Us_all{gi} = Us_s; Fs_all{gi} = Fs_s;
    fprintf('   smoothed:  %d / %d steps infeasible\n', nnz(Fs_s), nT-1);
end

%% 8. Plots ---------------------------------------------------------------
B_hat_grid = reshape(double(in_Bhat),  size(X1));
B_til_grid = reshape(double(in_Btil),  size(X1));

% Common helpers used in all phase plots
x2line = linspace(-4, 4, 200);
x1line = -par.tau_h*x2line - par.c_h;

thg = linspace(0, 2*pi, 300);
Lch = chol(par.P/par.rho, 'lower');
ell = Lch' \ [cos(thg); sin(thg)];

% Color map for the four gamma values
cmap = [
    0.0000 0.0000 0.0000   % black
    0.1216 0.4667 0.7059   % blue
    0.1725 0.6275 0.1725   % green
    0.8392 0.1529 0.1569   % red
    0.5804 0.4039 0.7412   % purple
];

%--- Figure 1: -------------------------------------------------
figure('Name','Sets','Position',[80 80 640 560]); hold on; box on;
LFb_btil_grid = reshape(LFb_btil, size(X1));
LFb_btil_grid(~in_Btil) = NaN;
pcolor(X1, X2, LFb_btil_grid); colormap(turbo); caxis([-2 2]); shading flat;
[~,hBhat] = contour(X1, X2, B_hat_grid, [0.5 0.5], 'k-',  'LineWidth', 2);
[~,hBtil] = contour(X1, X2, B_til_grid, [0.5 0.5], 'm--', 'LineWidth', 2);
hdS = plot(x1line, x2line, 'r-', 'LineWidth', 2);
hSb = plot(ell(1,:), ell(2,:), 'b-', 'LineWidth', 2);
% trajectories
plot_split_trajectory(Xc_all{nG}, Fc_all{nG}, cmap(gi,:));
plot_split_trajectory(Xs_all{nG}, Fs_all{nG}, cmap(gi,:));
plot(x0(1), x0(2), 'ko', 'MarkerFaceColor','k','MarkerSize',6);
xlabel('$x_1$','interpreter','latex');
ylabel('$x_2$','interpreter','latex');
axis equal; xlim([-4.2 2]); ylim([-2 4]);
set(gca, 'Color', [0.95 0.95 0.95]);


%--- Figure 2: ------------------------------------------------------
figure('Name','Infeasibility plot','Position',[160 160 720 600]);

% Top: classic nonsmooth
subplot(2,1,1); hold on; box on;
for gi = 1:nG
    stairs(tgrid, Fc_all{gi}, 'LineWidth', 2, 'Color', cmap(gi,:));
end
xlabel('$t$ (s)','interpreter','latex');
ylim([-0.1 1.1]);
yticks([0 1]);

% Bottom: smoothed
subplot(2,1,2); hold on; box on;
for gi = 1:nG
    stairs(tgrid, Fs_all{gi}, 'LineWidth', 2, 'Color', cmap(gi,:));
end
xlabel('$t$ (s)','interpreter','latex');
ylim([-0.1 1.1]);
yticks([0 1]);

%% =========================== local functions ============================
function hLegend = plot_split_trajectory(X, ~, col)
    hLegend = plot(X(1,:), X(2,:), '-', 'Color', col, 'LineWidth', 2.0);
end

function [X, S11, S12, S21, S22] = rk4_step(X, S11, S12, S21, S22, dt, par)
    [k1X, k1a, k1b, k1c, k1d] = rhs(X, S11, S12, S21, S22, par);
    Xt   = X   + 0.5*dt*k1X;
    a    = S11 + 0.5*dt*k1a; b = S12 + 0.5*dt*k1b;
    c    = S21 + 0.5*dt*k1c; d = S22 + 0.5*dt*k1d;
    [k2X, k2a, k2b, k2c, k2d] = rhs(Xt, a, b, c, d, par);
    Xt   = X   + 0.5*dt*k2X;
    a    = S11 + 0.5*dt*k2a; b = S12 + 0.5*dt*k2b;
    c    = S21 + 0.5*dt*k2c; d = S22 + 0.5*dt*k2d;
    [k3X, k3a, k3b, k3c, k3d] = rhs(Xt, a, b, c, d, par);
    Xt   = X   + dt*k3X;
    a    = S11 + dt*k3a; b = S12 + dt*k3b;
    c    = S21 + dt*k3c; d = S22 + dt*k3d;
    [k4X, k4a, k4b, k4c, k4d] = rhs(Xt, a, b, c, d, par);
    X    = X   + dt/6*(k1X + 2*k2X + 2*k3X + k4X);
    S11  = S11 + dt/6*(k1a + 2*k2a + 2*k3a + k4a);
    S12  = S12 + dt/6*(k1b + 2*k2b + 2*k3b + k4b);
    S21  = S21 + dt/6*(k1c + 2*k2c + 2*k3c + k4c);
    S22  = S22 + dt/6*(k1d + 2*k2d + 2*k3d + k4d);
end

function [dX, dS11, dS12, dS21, dS22] = rhs(X, S11, S12, S21, S22, par)
    v   = -X(1,:) - 2*X(2,:);
    th  = tanh(v / par.umax);
    s   = 1 - th.^2;
    dX  = [X(2,:); par.umax * th];
    dS11 = S21;
    dS12 = S22;
    dS21 = -s.*S11 - 2*s.*S21;
    dS22 = -s.*S12 - 2*s.*S22;
end

function [bv, db1v, db2v] = slice_vals_point(x, par)
    bv  = zeros(par.N,1);
    db1v = zeros(par.N,1);
    db2v = zeros(par.N,1);
    Xc = x;
    S11 = 1; S12 = 0; S21 = 0; S22 = 1;
    bv(1)  = par.h_fun(Xc);
    db1v(1) = S11 + par.tau_h * S21;
    db2v(1) = S12 + par.tau_h * S22;
    dt_int = par.dtau / par.n_sub;
    for i = 1:par.N-1
        for ss = 1:par.n_sub
            [Xc, S11, S12, S21, S22] = rk4_step(Xc, S11, S12, S21, S22, dt_int, par);
        end
        bv(i+1)  = par.h_fun(Xc);
        db1v(i+1) = S11 + par.tau_h * S21;
        db2v(i+1) = S12 + par.tau_h * S22;
    end
    y1 = Xc(1); y2 = Xc(2);
    bv(end) = par.rho - (par.P(1,1)*y1^2 + 2*par.P(1,2)*y1*y2 + par.P(2,2)*y2^2);
    v1 = par.P(1,1)*y1 + par.P(1,2)*y2;
    v2 = par.P(2,1)*y1 + par.P(2,2)*y2;
    db1v(end) = -2*(S11*v1 + S21*v2);
    db2v(end) = -2*(S12*v1 + S22*v2);
end

function [u, flag, t_qp] = sf_classical(x, kdes, gamma, par)
    [bv, db1v, db2v] = slice_vals_point(x, par);
    Lf = db1v * x(2);
    Lg = db2v;
    Aineq = [-Lg; 1; -1];
    bineq = [gamma*bv + Lf; par.umax; par.umax];
    H = 2; f = -2*kdes;
    opts = optimoptions('quadprog','Display','off');
    tic;
    [u,~,flag] = quadprog(H, f, Aineq, bineq, [],[],[],[],[],opts);
    t_qp = toc;
    if isempty(u), u = NaN; end
end

function [u, flag, t_qp] = sf_smoothed(x, kdes, gamma, theta, par)
    [bv, db1v, db2v] = slice_vals_point(x, par);
    z   = -theta*bv;
    zmx = max(z);
    ez  = exp(z - zmx);
    sm  = sum(ez);
    w   = ez/sm;
    btil   = -(1/theta)*(log(sm) + zmx);
    dbtil1 = sum(w.*db1v);
    dbtil2 = sum(w.*db2v);
    Lf = dbtil1*x(2);
    Lg = dbtil2;
    Aineq = [-Lg; 1; -1];
    bineq = [gamma*btil + Lf; par.umax; par.umax];
    H = 2; f = -2*kdes;
    opts = optimoptions('quadprog','Display','off');
    tic;
    [u,~,flag] = quadprog(H, f, Aineq, bineq, [],[],[],[],[],opts);
    t_qp = toc;
    if isempty(u), u = NaN; end
end