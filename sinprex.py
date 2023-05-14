import numpy as np
'''
シンプレックス表(例)
f:目的関数の最適値, a:係数, b:定数項, x:変数, λ:スラック変数
     |  x1  x2  λ1  λ2  λ3   b
---------------------------------
     | a00 a01 a02 a03 a04  -f
   λ1| a10 a11 a12 a13 a14  b1 
   λ2| a20 a21 a22 a23 a24  b2
   λ3| a30 a31 a32 a33 a34  b3
'''

##入力##
min_max = 1  #最小化問題なら0, 最大化問題なら1
l = [[-30, -20, 0, 0, 0, 0],
     [2, 0, 1, 0, 0, 4],
     [2, 1, 0, 1, 0, 8],
     [0, 1, 0, 0, 1, 6]]
# l = [[-2, -1, 0, 0, 0],  #最小化問題ならそのまま, 最大化問題なら目的関数に-1をかける
    #  [1, 1, 1, 0, 7],
    #  [4, 1, 0, 1, 16]]
print(np.matrix(l))



while True:  #目的関数の係数の最小値が0以上であれば終了
    #列選択#
    m_L = np.min(l[0])   #最上行の中から最小なものがある列Lを選択#
    L = l[0].index(m_L)  #####################################
    

    #行選択#
    count_a = 10000
    for i in range(1, len(l)):  #len(l)=4
        if l[i][L] == 0:
            continue
        elif l[i][L] < 0:
            x = -l[i][L]
        else:
            x = l[i][L]
        c = l[i][-1]
        C = c / x
        if count_a > C:
            count_a = C
            K = i

    P = l[K][L]                               #K行L列の要素を出力
    if P != 0:
        l[K] = list(map(lambda xx: xx/P, l[K]))   #K行をPで除算, l[K][L]を1にする

    #掃き出しを行う#
    for i in range(0, len(l)):                      
        if K == i:
            continue
        z = l[K]
        z = list(map(lambda xx: xx*(-l[i][L]), z))   #整数倍する
        l[i] = [x + y for x, y in zip(l[i], z)]
    
    print(K)  #ピボット行の出力
    print(L)  #ピボット列の出力
    print(np.matrix(l))  #シンプレックス表の更新の出力
    print()
    
    if np.min(l[0]) >= 0.0:
        break
    

for i in range(1, len(l)):  #変数が2つの場合
    if l[i][0] == 1:
        x1 = l[i][-1]
    if l[i][1] == 1:
        x2 = l[i][-1]

if min_max == 0:    
    print('最適解はx1={}, x2={}, f={}'.format(x1, x2, -l[0][-1]))
else:
    print('最適解はx1={}, x2={}, f={}'.format(x1, x2, l[0][-1]))





