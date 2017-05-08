/* find the most probable sequence */
void viterbi(int *data, int len, int nstates,int nobvs, double *prior, double *trans, double *obvs)
{
    double lambda[len][nstates];
    int backtrace[len][nstates];
    int stack[len];

    size_t i, j, k;
    double p;

    for (i = 0; i < len; i++) {
        for (j = 0; j < nstates; j++) {
            lambda[i][j] = - INFINITY;
        }
    }

    for (i = 0; i < nstates; i++) {
        lambda[0][i] = prior[i] + obvs[IDX(i,data[0],nobvs)];
        backtrace[0][i] = -1;       /* -1 is starting point */
    }
    for (i = 1; i < len; i++) {
        for (j = 0; j < nstates; j++) {
            for (k = 0; k < nstates; k++) {
                p = lambda[i-1][k] + trans[IDX(k,j,nstates)] + obvs[IDX(j,data[i],nobvs)];
                if (p > lambda[i][j]) {
                    lambda[i][j] = p;
                    backtrace[i][j] = k;
                }
            }
        }
    }

    /* backtrace */
    for (i = 0; i < nstates; i++) {
        if (i == 0 || lambda[len-1][i] > p) {
            p = lambda[len-1][i];
            k = i;
        }
    }
    stack[len - 1] = k;
    for (i = 1; i < len; i++) {
        stack[len - 1 - i] = backtrace[len - i][stack[len - i]];
    }
    for (i = 0; i < len; i++) {
        printf("%d ", stack[i]);
    }
    printf("\n");
}
