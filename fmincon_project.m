%% Joint dynamic multi slot optimization using fmincon
clc; clear; close all;

%% 1. System Parameters (Table I)
fc          = 28e9;
c           = 3e8;
lambda      = c / fc;
alpha_const = c / (4*pi*fc);

n_eff       = 1.4;
lambda0     = lambda / n_eff;

sigma2_dBm  = -90;
sigma2      = 10^((sigma2_dBm - 30)/10);

Gamma_th_dB = 10;
Gamma_th    = 10^(Gamma_th_dB/10);

delta       = lambda/2;

M           = 6;
N_pinching  = 3;
K           = 1;

T           = 15;
Delta_t     = 1.0;

d_ant_height    = 3.0;
pin_feed_point_x = 0;

AREA_MAX_X = 150;
AREA_MAX_Y = 150;

%% Fixed values
P0_dBm_current = 20;    % transmit power upper bound in dBm
E_max_current  = 6;     % energy budget in Joules

P0_max = 10^((P0_dBm_current - 30)/10);   % in Watts

%% Seed sweep

Seed_array = [202];

results_table = table('Size',[0 7], ...
    'VariableTypes',{'double','double','double','double','double','double','double'}, ...
    'VariableNames',{'P0_dBm','E_Max','Seed','TotalRate','MinSNR','ExitFlag','TimeSec'});

options = optimoptions('fmincon','Display','iter','Algorithm','sqp', ...
    'MaxIterations',500,'MaxFunctionEvaluations',1e5, ...
    'TolCon',1e-6,'TolFun',1e-6,'TolX',1e-6);

best_rate = -inf;
best_combination = struct();

%% Sweep seeds
for s = Seed_array

    seed_current = s;

    % users and targets are fixed over all time slots
    [users_all, targets_all] = generate_random_positions_static( ...
        seed_current, M, K, T, AREA_MAX_X, AREA_MAX_Y);
    % users_all:   M x 3 x T
    % targets_all: K x 3 x T

    % decision vector x = [ p(:) ; x_pin(:) ]
    % p is M x T, x_pin is N x T
    nVars = M*T + N_pinching*T; %#ok<NASGU>

    % initial guess for powers
    p0 = 0.25 * P0_max * ones(M,T);

    % initial guess for pinching positions
    xpin0 = zeros(N_pinching,T);
    base_lin = linspace(10, AREA_MAX_X-10, N_pinching);
    for tt = 1:T
        cand = base_lin' + randn(N_pinching,1)*1.0;
        cand = min(max(cand, 0), AREA_MAX_X);
        cand = project_positions_min_spacing(cand, delta, AREA_MAX_X);
        xpin0(:,tt) = cand;
    end
    x0 = [p0(:); xpin0(:)];

    % bounds
    lb_p   = zeros(M*T,1);
    ub_p   = P0_max * ones(M*T,1);
    lb_pin = zeros(N_pinching*T,1);
    ub_pin = AREA_MAX_X * ones(N_pinching*T,1);
    lb     = [lb_p; lb_pin];
    ub     = [ub_p; ub_pin];

    % no linear constraints
    A = []; b = []; Aeq = []; beq = [];

    % objective (negative total rate)
    fun = @(x) -total_rate_multi_slot(x, users_all, targets_all, ...
                alpha_const, lambda, lambda0, sigma2, ...
                d_ant_height, pin_feed_point_x, ...
                M, N_pinching, T, K);

    % nonlinear constraints
    nonlcon = @(x) constraints_multi_slot(x, users_all, targets_all, ...
                Gamma_th, delta, E_max_current, Delta_t, ...
                alpha_const, lambda, lambda0, sigma2, ...
                d_ant_height, pin_feed_point_x, ...
                M, N_pinching, T, K);

    t_start = tic;
    try
        [x_opt, fval, exitflag, output] = fmincon( ...
            fun, x0, A, b, Aeq, beq, lb, ub, nonlcon, options);
    catch ME
        disp('fmincon crashed:');
        disp(ME.message);
        exitflag = -999;
        x_opt    = x0;
        fval     = NaN;
        output.message = ME.message;
    end
    elapsed_time = toc(t_start);

    % evaluate solution
    if exitflag > 0
        totalRate = -fval;
        [~, SNRs_slot] = total_rate_multi_slot(x_opt, users_all, targets_all, ...
                    alpha_const, lambda, lambda0, sigma2, ...
                    d_ant_height, pin_feed_point_x, ...
                    M, N_pinching, T, K);
        minSNR = min(SNRs_slot(:));
    else
        totalRate = NaN;
        try
            [~, SNRs_slot] = total_rate_multi_slot(x_opt, users_all, targets_all, ...
                        alpha_const, lambda, lambda0, sigma2, ...
                        d_ant_height, pin_feed_point_x, ...
                        M, N_pinching, T, K);
            minSNR = min(SNRs_slot(:));
        catch
            minSNR = NaN;
        end
    end

    % store result
    new_row = {P0_dBm_current, E_max_current, seed_current, ...
               totalRate, minSNR, exitflag, elapsed_time};
    results_table = [results_table; new_row]; %#ok<AGROW>

    % track best
    if exitflag > 0 && ~isnan(totalRate) && totalRate > best_rate
        best_rate = totalRate;
        best_combination.P0_dBm = P0_dBm_current;
        best_combination.E_max  = E_max_current;
        best_combination.Seed   = seed_current;
        best_combination.x_opt  = x_opt;
        best_combination.users  = users_all(:,:,1);     % fixed over time
        best_combination.targets = targets_all(:,:,1);  % fixed over time
    end

    fprintf('Seed = %d  Rate = %.4f  minSNR = %.3f  exit = %d  time = %.2f s\n', ...
        seed_current, totalRate, minSNR, exitflag, elapsed_time);

end

%% Results
disp('==============================================');
disp('SWEEP COMPLETE OVER SEEDS');
disp('==============================================');
disp(results_table);
fprintf('\nBest Rate Found: %.6f bits per s per Hz\n', best_rate);
if isfield(best_combination,'Seed')
    fprintf('Fixed P0: %d dBm\n', best_combination.P0_dBm);
    fprintf('Fixed E: %.2f J\n', best_combination.E_max);
    fprintf('Best Seed: %d\n', best_combination.Seed);
end

%% Print best positions and powers
if isfield(best_combination,'x_opt')
    fprintf('\n========= Best UE Positions (M x 3) =========\n');
    % columns are x y z
    disp(best_combination.users);

    fprintf('\n========= Best Target Positions (K x 3) =========\n');
    disp(best_combination.targets);

    x_opt = best_combination.x_opt;
    x_pin_best = reshape(x_opt(M*T+1:end), N_pinching, T);
    fprintf('\n========= Optimal Pinching x positions per slot (N_pinching x T) =========\n');
    disp(x_pin_best);

    p_best = reshape(x_opt(1:M*T), M, T);
    fprintf('\n========= Optimal UE Power Allocation p(m,t) in Watts (M x T) =========\n');
    disp(p_best);
end
disp('==============================================');


%% =========================================================
%  Local functions
%% =========================================================

function [users_all, targets_all] = generate_random_positions_static( ...
    seed, M, K, T, max_x, max_y)
% Generate random user and target locations
% Fixed over all time slots

    users_all   = zeros(M,3,T);
    targets_all = zeros(K,3,T);
    rng(seed);

    users_once   = [rand(M,1)*max_x, rand(M,1)*max_y, zeros(M,1)];
    targets_once = [rand(K,1)*max_x, rand(K,1)*max_y, zeros(K,1)];

    for t = 1:T
        users_all(:,:,t)   = users_once;
        targets_all(:,:,t) = targets_once;
    end
end


function [totalRate, SNRs_slot] = total_rate_multi_slot(x, users_all, targets_all, ...
        alpha_const, lambda, lambda0, sigma2, ...
        d_ant_height, pin_feed_point_x, ...
        M, N_pinching, T, K)
% Compute total communication rate and sensing SNRs for all slots
% SNR formula implements equation 7
% SNRs_slot(k,t) is max over users of Gamma_{k,t,m}

    p     = reshape(x(1:M*T), M, T);               % user powers
    x_pin = reshape(x(M*T+1:end), N_pinching, T);  % pinching positions

    totalRate = 0;
    SNRs_slot = zeros(K,T);    % SNR per target and slot

    epsd = 1e-9;  % small value to avoid divide by zero

    for t = 1:T

        % pinching antenna positions in 3D for this slot (3 x N_pinching)
        pin_pos = [ x_pin(:,t), zeros(N_pinching,1), d_ant_height*ones(N_pinching,1) ]';

        % phase offsets in waveguide
        dist_from_feed = abs(x_pin(:,t) - pin_feed_point_x);
        theta_n = 2*pi*dist_from_feed / lambda0;

        % user channels H_m for this slot
        Hm_vals = complex(zeros(M,1));

        % communication rate for all users in slot t
        for m = 1:M
            user_pos = users_all(m,:,t)';                 % 3 x 1
            dist_mn  = vecnorm(user_pos - pin_pos) + epsd;  % 1 x N_pinching

            H_m = sum(alpha_const * ...
                      exp(-1j*2*pi./lambda .* dist_mn) ./ dist_mn .* ...
                      exp(-1j*theta_n.'));
            Hm_vals(m) = H_m;

            Rm = (1/M) * log2( 1 + ...
                (abs(H_m)^2 * p(m,t)) / (N_pinching * sigma2) );
            totalRate = totalRate + real(Rm);
        end

        % sensing SNR for each target in slot t
        for k = 1:K
            tgt_pos = targets_all(k,:,t)';    % 3 x 1
            dist_kn = vecnorm(tgt_pos - pin_pos) + epsd;

            Hk = sum(alpha_const * ...
                     exp(-1j*2*pi./lambda .* dist_kn) ./ dist_kn .* ...
                     exp(-1j*theta_n.'));

            % compute Gamma_{k,t,m} for all users m using equation 7
            SNR_m = zeros(M,1);
            for m = 1:M
                pm = p(m,t);
                num   = abs(Hk)^2 * pm / N_pinching;
                denom = abs(Hm_vals(m))^2 * pm / N_pinching + sigma2;
                SNR_m(m) = num / denom;
            end

            % slot SNR for target k is maximum over users
            SNRs_slot(k,t) = max(SNR_m);
        end
    end
end


function [c, ceq] = constraints_multi_slot(x, users_all, targets_all, ...
    Gamma_th, delta, E_max, Delta_t, ...
    alpha_const, lambda, lambda0, sigma2, ...
    d_ant_height, pin_feed_point_x, ...
    M, N_pinching, T, K)
% Nonlinear constraints:
%   sensing SNR constraints  Gamma_th - Gamma_{k,t} <= 0  for all k,t
%   total energy constraint  sum p * Delta_t <= E_max
%   pairwise spacing constraints per slot  |x_n - x_m| >= delta

    p     = reshape(x(1:M*T), M, T);
    x_pin = reshape(x(M*T+1:end), N_pinching, T);

    % SNRs per target and slot
    [~, SNRs_slot] = total_rate_multi_slot(x, users_all, targets_all, ...
        alpha_const, lambda, lambda0, sigma2, ...
        d_ant_height, pin_feed_point_x, ...
        M, N_pinching, T, K);

    % SNR constraints: Gamma_th - Gamma_{k,t} <= 0
    c1 = [];
    for t = 1:T
        for k = 1:K
            c1 = [c1; Gamma_th - SNRs_slot(k,t)]; %#ok<AGROW>
        end
    end

    % energy constraint: sum p(m,t)*Delta_t - E_max <= 0
    c2 = sum(p(:)) * Delta_t - E_max;

    % pairwise spacing constraints: for each slot, for all pairs n<m
    c3 = [];
    for t = 1:T
        for n = 1:N_pinching
            for m = n+1:N_pinching
                c3 = [c3; delta - abs(x_pin(n,t) - x_pin(m,t))]; %#ok<AGROW>
            end
        end
    end

    c   = [c1; c2; c3];
    ceq = [];
end


function proj = project_positions_min_spacing(pos, delta, x_max)
% Project one dimensional positions pos (column vector) to satisfy minimum spacing delta,
% while keeping them inside [0, x_max]. A simple greedy projection

    N = length(pos);
    [p_sorted, idx] = sort(pos);

    p_sorted = max(p_sorted, 0);
    p_sorted = min(p_sorted, x_max);

    for i = 2:N
        if p_sorted(i) - p_sorted(i-1) < delta
            p_sorted(i) = p_sorted(i-1) + delta;
        end
    end

    for i = N-1:-1:1
        if p_sorted(i+1) - p_sorted(i) < delta
            p_sorted(i) = p_sorted(i+1) - delta;
        end
    end

    p_sorted = max(p_sorted, 0);
    p_sorted = min(p_sorted, x_max);

    proj = zeros(size(pos));
    proj(idx) = p_sorted;
end
