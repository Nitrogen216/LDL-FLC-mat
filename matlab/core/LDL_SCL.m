function label_pre = LDL_SCL(x_train, y_train, x_test, y_test, lambda1, lambda2, lambda3, c, reg)
% MATLAB port of SCL.py core procedure
    iters = 300; batch = 50;
    rho1 = 0.9; rho2 = 0.999; delta = 1e-8; epsilon = 1e-3;
    code_len = c + 1;
    [n, d] = size(x_train); l = size(y_train,2);
    s1 = zeros(d,l); r1 = zeros(d,l);
    s2 = zeros(code_len,l); r2 = zeros(code_len,l);
    mu = 1;
    theta1 = ones(d,l); w1 = ones(code_len,l);
    [c1, p1] = cluster_ave(y_train, code_len);
    s3 = zeros(size(c1)); r3 = zeros(size(c1));
    loss1 = optimize_func(x_train, theta1, c1, w1, y_train, p1, lambda1, lambda2, lambda3, mu);
    loss = loss1;
    for i=1:iters
        [xb, yb, cb] = next_batch(batch, x_train, y_train, c1);
        g1 = gradient_theta(xb, theta1, cb, w1, yb, p1, lambda1, lambda2, lambda3);
        s1 = rho1*s1 + (1-rho1)*g1; s1h = s1/(1-rho1^(i)); % i=1,2,... matches Python i+1 with i=0,1,...
        r1 = rho2*r1 + (1-rho2)*(g1.^2); r1h = r1/(1-rho2^(i));

        g2 = gradient_w(xb, theta1, cb, w1, yb, p1, lambda1, lambda2, lambda3);
        s2 = rho1*s2 + (1-rho1)*g2; s2h = s2/(1-rho1^(i));
        r2 = rho2*r2 + (1-rho2)*(g2.^2); r2h = r2/(1-rho2^(i));

        g3 = gradient_c(x_train, code_len, theta1, c1, w1, y_train, p1, lambda1, lambda2, lambda3, mu);
        s3 = rho1*s3 + (1-rho1)*g3; s3h = s3/(1-rho1^(i));
        r3 = rho2*r3 + (1-rho2)*(g3.^2); r3h = r3/(1-rho2^(i));

        theta1 = theta1 - epsilon * s1h ./ (sqrt(r1h)+delta);
        w1 = w1 - epsilon * s2h ./ (sqrt(r2h)+delta);
        c1 = c1 - epsilon * s3h ./ (sqrt(r3h)+delta);

        loss2 = optimize_func(x_train, theta1, c1, w1, y_train, p1, lambda1, lambda2, lambda3, mu);
        if i>5 && abs(loss2 - loss) < 1e-4, break; else, mu = 0.1*mu; loss = loss2; end
    end
    % regress codes for test
    regression = cell(1, code_len);
    if nargin < 9 || isempty(reg)
        for i=1:code_len
            regression{i} = fitlm(x_train, c1(:,i));
        end
    else
        for i=1:code_len
            regression{i} = fitrlinear(x_train, c1(:,i), 'Learner','leastsquares','Lambda',reg,'Regularization','ridge');
        end
    end
    codes = zeros(size(x_test,1), code_len);
    for i=1:code_len
        if isa(regression{i}, 'LinearModel')
            codes(:,i) = predict(regression{i}, x_test);
        else
            codes(:,i) = predict(regression{i}, x_test);
        end
    end
    label_pre = predict_func(x_test, theta1, codes, w1);
end

function [data_shuffle, labels_shuffle, codes_shuffle] = next_batch(num, data, labels, codes)
    idx = randperm(size(data,1), num);
    data_shuffle = data(idx,:);
    labels_shuffle = labels(idx,:);
    codes_shuffle = codes(idx,:);
end

function [c, p] = cluster_ave(labels_train, n)
    train_len = size(labels_train,1);
    % kmeans on label space
    predict = kmeans(labels_train, n, 'MaxIter', 1000, 'Replicates', 1);
    classification = cell(1,n);
    c = zeros(train_len, n) + 1e-6;
    for i=1:train_len
        c(i, predict(i)) = 1;
        classification{predict(i)} = [classification{predict(i)}; labels_train(i,:)]; %#ok<AGROW>
    end
    p = zeros(n, size(labels_train,2));
    for i=1:n
        p(i,:) = mean(classification{i}, 1);
    end
end

function P = predict_func(x, theta, c, w)
    M = x*theta + c*w;
    M = M - max(M,[],2);
    E = exp(M);
    P = E ./ sum(E,2);
end

function val = optimize_func(x, theta, c, w, label_real, p, lambda1, lambda2, lambda3, mu)
    label_predict = predict_func(x, theta, c, w);
    label_real = max(min(label_real,1), 1e-6);
    label_predict = max(min(label_predict,1), 1e-6);
    term1 = sum(label_real .* log(label_real ./ label_predict), 'all');
    term2 = sum(theta.^2,'all');
    term3 = sum(w.^2,'all');
    dist = zeros(size(label_predict,1), size(p,1));
    for i=1:size(p,1)
        diff = label_predict - p(i,:);
        dist(:,i) = sum(diff.^2,2);
    end
    term4 = sum(c .* dist, 'all');
    term5 = sum(1 ./ c, 'all');
    val = term1 + lambda1*term2 + lambda2*term3 + lambda3*term4 + mu*term5;
end

function g = gradient_theta(x, theta, c, w, label_real, P, lambda1, lambda2, lambda3)
    g1 = x' * (predict_func(x, theta, c, w) - label_real);
    g2 = 2*lambda1*theta;
    p_tmp = exp(x*theta + c*w);
    p = (p_tmp' ./ sum(p_tmp,2)')';
    g3 = x' * (((size(P,1)*sum(c,2).*p)' - c*P)'.*(p-p.^2));
    g3 = 2*lambda3*g3;
    g = g1 + g2 + g3;
end

function g = gradient_w(x, theta, c, w, label_real, P, lambda1, lambda2, lambda3)
    g1 = c' * (predict_func(x, theta, c, w) - label_real);
    g2 = 2*lambda2*w;
    p_tmp = exp(x*theta + c*w);
    p = (p_tmp' ./ sum(p_tmp,2)')';
    g3 = c' * (((size(P,1)*sum(c,2).*p)' - c*P)'.*(p-p.^2));
    g3 = 2*lambda3*g3;
    g = g1 + g2 + g3;
end

function g = gradient_c(x, code_len, theta, c, w, label_real, P, lambda1, lambda2, lambda3, mu)
    g1 = -label_real * w';
    p_tmp = exp(x*theta + c*w);
    p = (p_tmp' ./ sum(p_tmp,2)')';
    numerator = p_tmp * w';
    denominator = sum(p_tmp,2);
    g2 = (numerator' ./ denominator')';
    % approximate g3/g4 per python version
    g3 = zeros(size(c));
    for m=1:size(x,1)
        for n=1:code_len
            grad=0;
            for L=1:size(label_real,2)
                grad = grad + (p(m,L) - P(n,L)) * p(m,L) * (w(n,L) - g2(m,n));
            end
            g3(m,n) = 2*lambda3*c(m,n)*grad;
        end
    end
    a = sum(p.*p,2); b = sum(P.*P,2)'; ab = p*P';
    g4 = lambda3 * abs(a + b - 2*ab);
    g5 = -mu * (c.^(-2));
    g = g1 + g2 + g3 + g4 + g5;
end

