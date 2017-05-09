/* forward algoritm: return observation likelihood */
double forward(int *data, int len, int nstates, int nobvs,
        double *prior, double * trans, double *obvs)
{
    /* construct trellis */
    // double alpha[len][nstates];
    // double beta[len][nstates];
    double *alpha = (double *)malloc(len * nstates * sizeof(double));
    double *beta = (double *)malloc(len * nstates * sizeof(double));

    double loglik;

    for (int i = 0; i < len; i++) {
        for (int j = 0; j < nstates; j++) {
            alpha[i * nstates + j] = - INFINITY;
            beta[i * nstates + j] = - INFINITY;
        }
    }

    double startTime = CycleTimer::currentSeconds();
    /* forward pass */
    for (int i = 0; i < nstates; i++) {
        alpha[i] = prior[i] + obvs[IDX(i,data[0],nobvs)];
    }

    for (int i = 1; i < len; i++) {
        for (int j = 0; j < nstates; j++) {
            for (int k = 0; k < nstates; k++) {
                double p = alpha[(i-1) * nstates + k] + trans[IDX(k,j,nstates)] + obvs[IDX(j,data[i],nobvs)];
                alpha[i * nstates + j] = logadd(alpha[i * nstates + j], p);
            }
        }
    }
    loglik = -INFINITY;
    for (int i = 0; i < nstates; i++) {
        loglik = logadd(loglik, alpha[(len-1) * nstates + i]);
    }
    double endTime = CycleTimer::currentSeconds();
    printf("Time taken %.4f milliseconds\n",  (endTime - startTime) * 1000);

    return loglik;
}
