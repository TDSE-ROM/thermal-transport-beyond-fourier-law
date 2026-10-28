function ExperimentII_Alpha_Comparison
% EXPERIMENT II: Effect of fractional order on the solution
% MATLAB R2021a compatible.
%
% Equation:
%   CaputoD^alpha y(t) + d*y(t) = g(t),  y(0)=0
%
% For alpha < 1, manufactured solution:
%   y_exact(t) = t^alpha
%   g(t) = Gamma(alpha+1) + t^alpha
%
% At alpha = 1, solve the corresponding classical ODE:
%   y'(t) + d*y(t) = g(t), y(0)=0
%
% Numerical method: nonuniform-capable L1 scheme on a uniform grid.
% Distinct colours are used for the six curves.

    clear; clc; close all;

    % Parameters
    alphaValues = [0.5 0.6 0.7 0.8 0.9 1.0];
    Tf = 1.0;
    d = 0.5; %1.0;
    Nt = 1200;

    t = linspace(0,Tf,Nt+1)';
    dt = Tf/Nt;

    % Distinct RGB colours
    curveColors = [
        0.0000 0.4470 0.7410
        0.8500 0.3250 0.0980
        0.9290 0.6940 0.1250
        0.4940 0.1840 0.5560
        0.4660 0.6740 0.1880
        0.3010 0.7450 0.9330
    ];

    lineStyles = {'-','--','-.',':','-','--'};

    Y = zeros(Nt+1,numel(alphaValues));

    for ia = 1:numel(alphaValues)
        alpha = alphaValues(ia);

        if abs(alpha-1.0) < 1e-12
            % Classical first-order ODE:
            % y' + d*y = 1 + t, y(0)=0.
            % Use the exact solution for this case.
            %
            % For d=1:
            % y(t) = t - 1 + exp(-t)
            Y(:,ia) = t - 1 + exp(-t);
        else
            % Manufactured fractional solution y_exact = t^alpha.
            % Forcing: g(t) = Gamma(alpha+1) + d*t^alpha.
            g = gamma(alpha+1)*ones(Nt+1,1) + d*t.^alpha;

            Y(:,ia) = uniform_L1_solver(alpha,d,dt,g);
        end
    end

    % Plot all solutions
    figure('Color','w','Position',[100 100 900 600]);
    hold on; box on; grid on;

    for ia = 1:numel(alphaValues)
        plot(t,Y(:,ia), ...
            'Color',curveColors(ia,:), ...
            'LineStyle',lineStyles{ia}, ...
            'LineWidth',2.0, ...
            'DisplayName',sprintf('\\alpha = %.1f',alphaValues(ia)));
    end

    xlabel('Time, $t$','Interpreter','latex','FontSize',13);
    ylabel('Solution, $y(t)$','Interpreter','latex','FontSize',13);
    title('Solution profiles for different fractional orders', ...
        'FontSize',14,'FontWeight','bold');

    legend('Location','northwest','FontSize',11);
    xlim([0 Tf]);
    set(gca,'FontSize',11,'LineWidth',1.0);

    exportgraphics(gcf,'ExperimentII_Alpha_Comparison.png', ...
        'Resolution',300);

    % Save numerical solution data
    Data = array2table([t Y], ...
        'VariableNames', ...
        {'t','alpha_05','alpha_06','alpha_07', ...
         'alpha_08','alpha_09','alpha_10'});

    writetable(Data,'ExperimentII_Alpha_Comparison.csv');

    fprintf('Plot saved as ExperimentII_Alpha_Comparison.png\n');
    fprintf('Solution data saved as ExperimentII_Alpha_Comparison.csv\n');
end


function y = uniform_L1_solver(alpha,d,dt,g)
% Uniform-grid L1 solver for
%   CaputoD^alpha y + d*y = g, y(0)=0, 0<alpha<1.

    g = g(:);
    Nt = numel(g)-1;

    if alpha <= 0 || alpha >= 1
        error('The L1 solver requires 0 < alpha < 1.');
    end

    y = zeros(Nt+1,1);

    % L1 convolution coefficients
    k = (0:Nt-1)';
    a = (k+1).^(1-alpha)-k.^(1-alpha);

    c = 1/(gamma(2-alpha)*dt^alpha);

    for n = 1:Nt
        if n == 1
            history = 0;
        else
            % Contributions from previously computed increments
            j = 1:n-1;
            weights = a(n-j+1)-a(n-j);
            increments = y(j+1)-y(j);
            history = sum(weights(:).*increments(:));
        end

        % Current time step
        y(n+1) = (g(n+1)-c*history+c*y(n))/(c+d);
    end
end
