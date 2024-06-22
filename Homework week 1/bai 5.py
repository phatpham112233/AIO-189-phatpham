def md_nre_single_sample(y, y_hat, n, P):
    y_root = y** (1/n)
    y_hat_root = y_hat ** (1/n)
    
    difference = abs(y_root - y_hat_root)
    loss = difference ** P
    return loss

y = 2
y_hat = 9
n = 2
p = 1

result = md_nre_single_sample(y, y_hat, n ,p)
print(f"MD_nRE(y={y}, y_hat={y_hat}, n={1}, p={p}) = {result}")

print(md_nre_single_sample(50, 49.5, 2, 1))
print(md_nre_single_sample(20, 19.5, 2, 1))
print(md_nre_single_sample(5.5, 5.0, 2, 1))
print(md_nre_single_sample(1.0, 0.5, 2, 1))
print(md_nre_single_sample(0.6, 0.1, 2, 1))