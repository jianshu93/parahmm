/* find the most probable sequence */
void viterbi(int *data, int len, int nstates,int nobvs, double *prior, double *trans, double *obvs)
{
    double *lambda = (double *)malloc(len * nstates * sizeof(double));
    int *backtrace = (int *)malloc(len * nstates * sizeof(int));
    int *stack = (int *)malloc(len * sizeof(int));

    double p;

    for (int i = 0; i < len; i++) {
        for (int j = 0; j < nstates; j++) {
            lambda[i * nstates + j] = - INFINITY;
        }
    }

    clock_t start = clock(), diff;
    for (int i = 0; i < nstates; i++) {
        lambda[i] = prior[i] + obvs[IDX(i,data[0],nobvs)];
        backtrace[i] = -1;       /* -1 is starting point */
    }
    for (int i = 1; i < len; i++) {
        for (int j = 0; j < nstates; j++) {
            for (int k = 0; k < nstates; k++) {
                p = lambda[(i-1) * nstates + k] + trans[IDX(k,j,nstates)] + obvs[IDX(j,data[i],nobvs)];
                if (p > lambda[i * nstates + j]) {
                    lambda[i * nstates + j] = p;
                    backtrace[i * nstates + j] = k;
                }
            }
        }
    }

    int k = 0;
    /* backtrace */
    for (int i = 0; i < nstates; i++) {
        if (i == 0 || lambda[(len-1) * nstates + i] > p) {
            p = lambda[(len-1) * nstates + i];
            k = i;
        }
    }
    stack[len - 1] = k;
    for (int i = 1; i < len; i++) {
        stack[len - 1 - i] = backtrace[(len - i) * nstates + stack[len - i]];
    }
    for (int i = 0; i < len; i++) {
        printf("%d ", stack[i]);
    }
    printf("\n");
    diff = clock() - start;
    int msec = diff * 1000 / CLOCKS_PER_SEC;
    printf("Time taken %d milliseconds\n",  msec);

    free(lambda);
    free(backtrace);
    free(stack);
}
