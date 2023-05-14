#include <stdio.h>
#define I 4
#define J 6
/*
シンプレックス表(例)
f:目的関数の最適値, a:係数, b:定数項, x:変数, λ:スラック変数
     |  x1  x2  λ1  λ2  λ3   b
---------------------------------
     | a00 a01 a02 a03 a04  -f
   λ1| a10 a11 a12 a13 a14  b1 
   λ2| a20 a21 a22 a23 a24  b2
   λ3| a30 a31 a32 a33 a34  b3
*/
void output(double l[I][J]){
    int i, j;
    
    for (int i=0;i<I;i++){
        for(int j=0;j<J;j++){
            printf("%.1lf ",l[i][j]);
        }
        printf("\n");
    }
    
}

int row_index(double l[I][J]){  //行の最小値の列を出力する関数
    int j, min_index=0;
    double min = l[0][0];
    for(int j=1;j<J;j++){
        if (min > l[0][j]){
            min = l[0][j];
            printf("%.1lf \n",min);
            min_index = j;
            printf("%d",j);///////////////////ここから
        }

    }
    return min_index;
}
int line_min(double l[I][J]){  //行の最小値を出力する関数
    int j, min_index=0;
    double min = l[0][0];
    for(int j=1;j<J;j++){
        if (min > l[0][j]){
            min = l[0][j];
            
        }

    }
    return min;
}

int line_index(double l[I][J], int L){
    int K=1;
    double x, c, C;
    double count_a = l[0][L];
    for(int i=1;i<I;i++){
        if (l[i][L] == 0){
            continue;
        }
        else if(l[i][L] < 0){
            x = -l[i][L];
        }
        else{
            x = l[i][L];
        }
        c = l[i][-1];
        C = c / x;
        if (count_a > C){
            count_a = C;
            K = i;
        }

    }
    return K;
}



int main(void){
    double l[I][J] = {{-30, -20, 0, 0, 0, 0},
                      {2, 0, 1, 0, 0, 4},
                      {2, 1, 0, 1, 0, 8},
                      {0, 1, 0, 0, 1, 6}};
    // output(l);
    for(int b=0;b<2;b++){
        int L = row_index(l);
        int K = line_index(l, L);

        double P = l[K][L];
        if (l[K][L] != 0){
            for(int a=0;a<J;a++){
                l[K][a] /= P;
                // printf("%.1lf ", l[K][a]);
            }
        }
        
        /*掃き出しを行う*/
        for(int i=0;i<I;i++){
            if (K == i){
                continue;
            }
            double xx = l[K][L]*(-l[i][L]);
            for(int j=0;j<J;j++){
                l[i][j] += l[K][j]*xx; 
            }
        
        }
        
       
        // output(l);
        // printf("\n");

        if (line_min(l) >= 0.0){
            break;
        }
        

        
    }

}