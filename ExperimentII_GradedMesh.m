function ExperimentII_GradedMesh
% EXPERIMENT II: Uniform versus graded temporal meshes
% MATLAB R2021a compatible.
%
% Model:
%   CaputoD^alpha y(t) + d*y(t) = g(t),  y(0)=0
% Exact solution:
%   y_exact(t) = t^sigma, sigma = alpha
%
% The nonuniform L1 weights are calculated from the actual time nodes.

    clear; clc; close all;
    format short e;

    Tf = 1.0;
    d = 1.0;
    alphas = [0.4, 0.8];
    Nvalues = [40, 80, 160, 320, 640];

    % Results columns:
    % alpha, grid code (1=uniform, 2=graded), N, r, error, order
    Results = [];

    for ia = 1:numel(alphas)
        alpha = alphas(ia);
        sigma = alpha;

        % Predeclared grading parameter
        rGraded = 2.0/alpha;

        for igrid = 1:2
            if igrid == 1
                r = 1.0;
                gridName = 'Uniform';
            else
                r = rGraded;
                gridName = 'Graded';
            end

            errors = zeros(numel(Nvalues),1);

            for j = 1:numel(Nvalues)
                N = Nvalues(j);

                % Actual temporal grid
                t = Tf*( (0:N)'/N ).^r;

                % Exact solution and manufactured forcing
                yExact = t.^sigma;
                g = zeros(N+1,1);

                for n = 2:N+1
                    tn = t(n);
                    caputoExact = gamma(sigma+1) ...
                        / gamma(sigma+1-alpha) ...
                        * tn^(sigma-alpha);

                    g(n) = caputoExact + d*tn^sigma;
                end

                % Nonuniform L1 solver
                y = nonuniform_L1_solver(alpha,d,t,g);

                % Maximum-in-time error, including t=0
                errors(j) = max(abs(y-yExact));

                if j == 1
                    pObserved = NaN;
                else
                    pObserved = log(errors(j-1)/errors(j)) ...
                        / log(Nvalues(j)/Nvalues(j-1));
                end

                Results = [Results; ...
                    alpha, igrid, N, r, errors(j), pObserved]; %#ok<AGROW>

                if isnan(pObserved)
                    fprintf('%5.1f  %-7s  %4d  %6.2f  %13.6e  %8s\n', ...
                        alpha,gridName,N,r,errors(j),'--');
                else
                    fprintf('%5.1f  %-7s  %4d  %6.2f  %13.6e  %8.4f\n', ...
                        alpha,gridName,N,r,errors(j),pObserved);
                end
            end

            fprintf('\n');
        end
    end

    % Export numerical results
    ResultTable = array2table(Results, ...
        'VariableNames', ...
        {'alpha','gridCode','N','r','Einf','pObserved'});

    ResultTable.Grid = repmat({''},height(ResultTable),1);
    ResultTable.Grid(ResultTable.gridCode == 1) = {'Uniform'};
    ResultTable.Grid(ResultTable.gridCode == 2) = {'Graded'};

    ResultTable = movevars(ResultTable,'Grid','After','gridCode');
    writetable(ResultTable,'ExperimentII_GradedMesh_Results.csv');

    % Convergence figure
    figure('Color','w','Position',[100 100 1100 440]);

    for ia = 1:numel(alphas)
        alpha = alphas(ia);
        subplot(1,2,ia);
        hold on; box on; grid on;

        for igrid = 1:2
            mask = Results(:,1) == alpha & Results(:,2) == igrid;
            Ns = Results(mask,3);
            Err = Results(mask,5);

            if igrid == 1
                loglog(Ns,Err,'o-','LineWidth',1.6, ...
                    'MarkerSize',6,'DisplayName','Uniform');
            else
                loglog(Ns,Err,'s-','LineWidth',1.6, ...
                    'MarkerSize',6,'DisplayName','Graded');
            end
        end

        xlabel('$N$','Interpreter','latex');
        ylabel('$E_\infty$','Interpreter','latex');
        title(sprintf('$\\alpha=%.1f$',alpha), ...
            'Interpreter','latex');
        legend('Location','southwest');
        set(gca,'FontSize',11);
    end

    exportgraphics(gcf,'ExperimentII_GradedMesh_Convergence.png', ...
        'Resolution',300);

    disp(' ');
    disp('Complete results:');
    disp(ResultTable);
    disp('Saved CSV: ExperimentII_GradedMesh_Results.csv');
    disp('Saved figure: ExperimentII_GradedMesh_Convergence.png');
end


function y = nonuniform_L1_solver(alpha,d,t,g)
% Nonuniform L1 method for
%   CaputoD^alpha y + d*y = g, 0 < alpha < 1.
%
% MATLAB indices:
%   y(1) corresponds to physical time t(1)=0.
%   y(n+1) corresponds to physical time t_n.
%
% At each t_n, the L1 approximation is
% 1/Gamma(2-alpha) * sum_{k=1}^n
% [(t_n-t_{k-1})^(1-alpha)-(t_n-t_k)^(1-alpha)]
% /(t_k-t_{k-1}) * (y_k-y_{k-1}).

    if alpha <= 0 || alpha >= 1
        error('alpha must satisfy 0 < alpha < 1.');
    end
    if d <= 0
        error('d must be positive.');
    end

    t = t(:);
    g = g(:);

    Nt = numel(t)-1;
    if numel(g) ~= Nt+1
        error('The vectors t and g must have equal lengths.');
    end
    if any(diff(t) <= 0)
        error('Time nodes must be strictly increasing.');
    end

    y = zeros(Nt+1,1);
    c = 1/gamma(2-alpha);

    for n = 1:Nt
        tn = t(n+1);

        % L1 weights for increments k=1,...,n.
        k = (1:n)';
        dt = t(k+1)-t(k);

        weights = ((tn-t(k)).^(1-alpha) ...
                 - (tn-t(k+1)).^(1-alpha))./dt;

        % Contributions from already computed increments.
        if n > 1
            increments = y(2:n+1)-y(1:n);
            history = sum(weights(1:n-1).*increments(1:n-1));
        else
            history = 0;
        end

        % Current increment is y_{n}-y_{n-1};
        % in MATLAB indexing these are y(n+1)-y(n).
        wCurrent = weights(n);

        y(n+1) = (g(n+1)/c - history + wCurrent*y(n)) ...
                 /(wCurrent + d/c);
    end
end