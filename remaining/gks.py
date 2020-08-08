t = int(input())
for i in range(t):
    n = int(input())
    alist = list(map(int,input().split()))
    total = 0
    maxtill = -1
    for k in range(n):
        if alist[k]>maxtill and (k == n-1 or alist[k] > alist[k+1]):
            total+=1
        maxtill = max(maxtill , alist[k])
    print("Case #{}: {}".format(i+1,total))
