sigma = 1.6;

k = @(x2) -((sigma^2 * log(3))/3) - x2;

fun0 = @(x1,x2) ((1/(2*pi*(sigma^2))).*exp(-(((x1-1.5).^2) + ((x2-1.5).^2))/...
        (2*(sigma^2))))*0.75;

fun1 = @(x1,x2) ((1/(2*pi*(sigma^2))).*exp(-(((x1+1.5).^2) + ((x2+1.5).^2))/...
        (2*(sigma^2))))*0.25;
    
Err0 = integral2(fun0, -Inf, +Inf, -Inf, k);

Err1 = integral2(fun1, -Inf, +Inf, k, +Inf);

total_error = Err0 + Err1;