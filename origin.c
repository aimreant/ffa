#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>


#define RND          (rand()/(RAND_MAX + 1.0))
#define PI           (3.14159265359)

/* 问题相关 */
#define X_DIM       10
#define gsonum      100           /* 种群大小 */





double domx[2] = { -100, 100 };
double get_y( double x[X_DIM] )
{
    register int i;
    register double sum;

    sum = 0.0;
    for ( i = 0; i < X_DIM; i ++ )
    {
        sum += x[i] * x[i];
    }
    return sum;
}



/* 萤火虫结构 */
typedef struct tag_gso
{
    double x[X_DIM];
    double l;
    double rd;
    double y;
} gso_t, *gso_ptr;


/* 参数 */
const double initl = 5.0;          /* 初始萤光素值 */
const double rho = 0.4;            /* 萤光素挥发系数 */
const double alpha = 0.6;          /* 适应度影响因子 */
const double beta = 0.08;          /* 邻域变化率 */
const double s = 0.8;              /* 移动步长 */
const int nt = 5;                  /* 邻域阀值 */
const double initr = 400;          /* 初始决策半径 */
const double rs = 650;             /* 最大决策半径 */


/* 数据定义 */
gso_t gsos[gsonum];                /* 种群 */
gso_t optimum;                     /* 最优个体 */
double gsodist[gsonum][gsonum];    /* 距离矩阵 */
int tabu[gsonum];                  /* 禁忌表 */
int ninum[gsonum];                 /* 邻域集 */
int maxiter;                       /* 最大迭代次数 */




double dist_gso( int idx1, int idx2 )
{
    register int i;
    register double sum, sum1;

    sum = 0.0;
    for ( i = 0; i < X_DIM; i ++ )
    {
        sum1 = gsos[idx1].x[i] - gsos[idx2].x[i];
        sum += sum1 * sum1;
    }

    return sqrt( sum );
}



void init_single_gso( int idx )
{
    register int i;

    for ( i = 0; i < X_DIM; i ++ )
    {
        gsos[idx].x[i] = domx[0] + ( domx[1] - domx[0] ) * RND;
    }
    gsos[idx].y = get_y( gsos[idx].x );
    gsos[idx].l = initl;
    gsos[idx].rd = initr;

    if ( gsos[idx].y < optimum.y )
    {
        memcpy( &optimum, gsos + idx, sizeof( gso_t ) );
    }
}



void init_gsos()
{
    register int i;

    for ( i = 0; i < gsonum; i ++ )
    {
        init_single_gso( i );
    }
}


void move_gso( int idx1, int idx2 )
{
    register int i;

    for ( i = 0; i < X_DIM; i ++ )
    {
        gsos[idx1].x[i] += s * ( gsos[idx2].x[i] - gsos[idx1].x[i] ) / gsodist[idx1][idx2];

        if ( gsos[idx1].x[i] < domx[0] )
        {
            gsos[idx1].x[i] = domx[0];
        }
        else if ( gsos[idx1].x[i] > domx[1] )
        {
            gsos[idx1].x[i] = domx[1];
        }
    }

    gsos[idx1].y = get_y( gsos[idx1].x );
    if ( gsos[idx1].y < optimum.y )
    {
        memcpy( &optimum, gsos + idx1, sizeof( gso_t ) );
    }
}



int main()
{
    int i, j, k;
    int iter;
    double sum, partsum[gsonum + 1];
    double rnd;

    /* 初始化随机数发生器 */
    srand( ( unsigned int )time( NULL ) );

    /* 无穷大 */
    optimum.y = 1000000000000;

    /* 初始化种群 */
    init_gsos();

    /* 置迭代次数 */
    maxiter = 5000;


    /* 迭代 */
    for ( iter = 0; iter < maxiter; iter ++ )
    {
        /* 更新荧光素 */
        for ( i = 0; i < gsonum; i ++ )
        {
            gsos[i].l = ( 1 - rho ) * gsos[i].l + alpha / gsos[i].y;
        }

        /* 清ni集计数 */
        memset( ninum, 0, sizeof( ninum ) );

        for ( i = 0; i < gsonum; i ++ )
        {
            /* 构建ni */
            memset( tabu, 0, sizeof( tabu ) );

            for ( j = 0; j < gsonum; j ++ )
            {
                if ( i != j && gsos[i].l < gsos[j].l )
                {
                    gsodist[i][j] = dist_gso( i, j );
                    if ( gsodist[i][j] < gsos[i].rd )
                    {
                        tabu[j] = 1;
                        ninum[i] ++;
                    }
                }
            }

            if ( ninum[i] > 0 )
            {
                /* 轮盘赌 */
                sum = 0.0;
                for ( j = 0; j < gsonum; j ++ )
                {
                    if ( tabu[j] == 1 )
                    {
                        sum += gsos[j].l - gsos[i].l;
                    }
                }

                k = 1;
                partsum[0] = 0.0;
                for ( j = 0; j < gsonum; j ++ )
                {
                    if ( tabu[j] == 1 )
                    {
                        partsum[k] = partsum[k - 1] + ( gsos[j].l - gsos[i].l ) / sum;
                        k ++;
                    }
                }
                partsum[k - 1] = 1.1;

                rnd = RND;
                k = 1;
                for ( j = 0; j < gsonum; j ++ )
                {
                    if ( tabu[j] == 1 && rnd < partsum[k ++] )
                    {
                        break;
                    }
                }

                /* 选出j移动位置 */
                move_gso( i, j );
            }

            gsos[i].rd += beta * ( nt - ninum[i] );
            if ( gsos[i].rd < 0 )
            {
                gsos[i].rd = 0;
            }
            else if ( gsos[i].rd > rs )
            {
                gsos[i].rd = rs;
            }
        }

        fprintf( stdout, "best=%.15f\n", optimum.y );
    }

    return 0;
}