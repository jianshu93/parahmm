double *gmm = NULL;             /* gamma */
double *xi = NULL;              /* xi */
double *pi = NULL;              /* pi */

/* forward backward algoritm: return observation likelihood */
double forward_backward(int *data, int len, int nstates, int nobvs, double *prior, double *trans, double *obvs)
{
    /* construct trellis */
    double *alpha = (double *)malloc(len * nstates * sizeof(double));
    double *beta = (double *)malloc(len * nstates * sizeof(double));

    double loglik;

    for (int i = 0; i < len; i++) {
        for (int j = 0; j < nstates; j++) {
            alpha[i*nstates + j] = - INFINITY;
            beta[i*nstates + j] = - INFINITY;
        }
    }

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

    /* backward pass & update counts */
    for (int i = 0; i < nstates; i++) {
        beta[(len-1) * nstates + i] = 0;         /* 0 = log (1.0) */
    }
    for (int i = 1; i < len; i++) {
        for (int j = 0; j < nstates; j++) {

            double e = alpha[(len-i) * nstates + j] + beta[(len-i) * nstates + j] - loglik;
            gmm[IDX(j,data[len-i],nobvs)] = logadd(gmm[IDX(j,data[len-i],nobvs)], e);

            for (int k = 0; k < nstates; k++) {
                double p = beta[(len-i) * nstates + k] + trans[IDX(j,k,nstates)] + obvs[IDX(k,data[len-i],nobvs)];
                beta[(len-1-i) * nstates + j] = logadd(beta[(len-1-i) * nstates + j], p);

                e = alpha[(len-1-i) * nstates + j] + beta[(len-i) * nstates + k]
                    + trans[IDX(j,k,nstates)] + obvs[IDX(k,data[len-i],nobvs)] - loglik;
                xi[IDX(j,k,nstates)] = logadd(xi[IDX(j,k,nstates)], e);
            }
        }
    }
    double p = -INFINITY;
    for (int i = 0; i < nstates; i++) {
        p = logadd(p, prior[i] + beta[i] + obvs[IDX(i,data[0],nobvs)]);

        double e = alpha[i] + beta[i] - loglik;
        gmm[IDX(i,data[0],nobvs)] = logadd(gmm[IDX(i,data[0],nobvs)], e);

        pi[i] = logadd(pi[i], e);
    }

#ifdef DEBUG
    /* verify if forward prob == backward prob */
    if (fabs(p - loglik) > 1e-5) {
        fprintf(stderr, "Error: forward and backward incompatible: %lf, %lf\n", loglik, p);
    }
#endif

    return loglik;
}

void baum_welch(int *data, int nseq, int iterations, int length, int nstates, int nobvs, double *prior, double *trans, double *obvs)
{
    double *loglik = (double *) malloc(sizeof(double) * nseq);
    if (loglik == NULL) handle_error("malloc");
    for (int i = 0; i < iterations; i++) {
        double startTime = CycleTimer::currentSeconds();
        init_count();
        for (int j = 0; j < nseq; j++) {
            loglik[j] = forward_backward(data + length * j, length, nstates, nobvs, prior, trans, obvs);
        }
        double p = sum(loglik, nseq);

        update_prob();

        printf("iteration %d log-likelihood: %.4lf\n", i + 1, p);
        printf("updated parameters:\n");
        //printf("# initial state probability\n");
        //for (int j = 0; j < nstates; j++) {
        //    printf(" %.4f", exp(prior[j]));
        //}
        //printf("\n");
        //printf("# state transition probability\n");
        //for (int j = 0; j < nstates; j++) {
        //    for (int k = 0; k < nstates; k++) {
        //        printf(" %.4f", exp(trans[IDX(j,k,nstates)]));
        //    }
        //    printf("\n");
        //}
        //printf("# state output probility\n");
        //for (int j = 0; j < nstates; j++) {
        //    for (int k = 0; k < nobvs; k++) {
        //        printf(" %.4f", exp(obvs[IDX(j,k,nobvs)]));
        //    }
        //    printf("\n");
        //}
        //printf("\n");
        double endTime = CycleTimer::currentSeconds();
        printf("Time taken %.4f milliseconds\n",  (endTime - startTime) * 1000);
    }
    free(loglik);
}
