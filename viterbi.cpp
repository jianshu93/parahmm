/* find the most probable sequence */
void viterbi(int *data, int len, int nstates,int nobvs, float *prior, float *trans, float *obvs)
{
    float *lambda = (float *)aligned_alloc(32, len * nstates * sizeof(float));
    int *backtrace = (int *)malloc(len * nstates * sizeof(int));
    int *stack = (int *)malloc(len * sizeof(int));

    for (int i = 0; i < len; i++) {
        for (int j = 0; j < nstates; j++) {
            lambda[i * nstates + j] = - INFINITY;
        }
    }

    double startTime = CycleTimer::currentSeconds();
    for (int i = 0; i < nstates; i++) {
        lambda[i] = prior[i] + obvs[IDX(i,data[0],nobvs)];
        backtrace[i] = -1;       /* -1 is starting point */
    }
    
    __m256 lambda_AVX, trans_AVX, obvs_AVX;
    __m256 result; 

    for (int i = 1; i < len; i++) {
        for (int j = 0; j < nstates; j++) {
            obvs_AVX = _mm256_set1_ps(obvs[IDX(j,data[i],nobvs)]);
            for (int k = 0; k < nstates; k+=8) {
                lambda_AVX = _mm256_load_ps(lambda + (i-1) * nstates + k);
                trans_AVX = _mm256_load_ps(trans + j * nstates + k);
                result = _mm256_add_ps(lambda_AVX, trans_AVX);
                result = _mm256_add_ps(result, obvs_AVX);
                float* p = (float*)&result;
                for (int m = 0; m < 8; m++) {
                    if (p[m] > lambda[i * nstates + j]) {
                        lambda[i * nstates + j] = p[m];
                        backtrace[i * nstates + j] = k+m;
                    }
                }
            }
        }
    }

    int k = 0;
    float p = 0;
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
    double endTime = CycleTimer::currentSeconds();
    printf("Time taken %0.4f milliseconds\n",  (endTime - startTime) * 1000);

    free(lambda);
    free(backtrace);
    free(stack);
}
