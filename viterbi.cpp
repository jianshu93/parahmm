/* find the most probable sequence */
void viterbi(int *data, int len, int nstates,int nobvs, float *prior, float *trans, float *obvs)
{
    float *lambda = (float *)malloc(len * nstates * sizeof(float));
    int *backtrace = (int *)malloc(len * nstates * sizeof(int));
    int *stack = (int *)malloc(len * sizeof(int));

    float p;

    double startTime = CycleTimer::currentSeconds();
    //#pragma omp parallel
    //{
//#pragma omp parallel for
    for (int i = 0; i < len; i++) {
        for (int j = 0; j < nstates; j++) {
            lambda[i * nstates + j] = - INFINITY;
        }
    }

//#pragma omp parallel for
    for (int i = 0; i < nstates; i++) {
        lambda[i] = prior[i] + obvs[IDX(i,data[0],nobvs)];
        backtrace[i] = -1;       /* -1 is starting point */
    }
    for (int i = 1; i < len; i++) {
        //#pragma omp parallel for
        for (int j = 0; j < nstates; j++) {
            for (int k = 0; k < nstates; k++) {
                p = lambda[(i-1) * nstates + k] + trans[IDXT(k,j,nstates)] + obvs[IDX(j,data[i],nobvs)];
                if (p > lambda[i * nstates + j]) {
                    lambda[i * nstates + j] = p;
                    backtrace[i * nstates + j] = k;
                }
            }
        }
    }


    int k = 0;
    /* backtrace */
//#pragma omp parallel for
    for (int i = 0; i < nstates; i++) {
        if (i == 0 || lambda[(len-1) * nstates + i] > p) {
            p = lambda[(len-1) * nstates + i];
            k = i;
        }
    }
    //  }
    stack[len - 1] = k;
    for (int i = 1; i < len; i++) {
        stack[len - 1 - i] = backtrace[(len - i) * nstates + stack[len - i]];
    }
    for (int i = 0; i < len; i++) {
        printf("%d ", stack[i]);
    }
    printf("\n");
    double endTime = CycleTimer::currentSeconds();
    printf("Time taken %0.4f milliseconds\n",  (endTime - startTime) * 1000);

    free(lambda);
    free(backtrace);
    free(stack);
}
