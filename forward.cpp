/* forward algoritm: return observation likelihood */
float forward(int *data, int len, int nstates, int nobvs,
        float *prior, float * trans, float *obvs)
{
    /* construct trellis */
    // float alpha[len][nstates];
    // float beta[len][nstates];
    float *alpha = (float *)aligned_alloc(32, len * nstates * sizeof(float));
    float *beta = (float *)aligned_alloc(32, len * nstates * sizeof(float));

    float loglik;

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
    
    __m256 alpha_AVX, trans_AVX, obvs_AVX;
    __m256 result; 

    for (int i = 1; i < len; i++) {
        for (int j = 0; j < nstates; j++) {
            obvs_AVX = _mm256_set1_ps(obvs[IDX(j,data[i],nobvs)]);
            for (int k = 0; k < nstates; k+=8) {
                alpha_AVX = _mm256_load_ps(alpha + (i-1) * nstates + k);
                trans_AVX = _mm256_load_ps(trans + j * nstates + k);
                result = _mm256_add_ps(alpha_AVX, trans_AVX);
                result = _mm256_add_ps(result, obvs_AVX);
                float* p = (float*)&result;
                for (int m = 0; m < 8; m++) {
                    alpha[i * nstates + j] = logadd(alpha[i * nstates + j], p[m]);
                }
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
