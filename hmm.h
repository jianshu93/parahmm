double logadd(double, double);
double sum(double *, int);
double forward(int *, int, int, int, double *, double *, double *);
double forward_backward(int *, int, int, int, double *, double *, double *);
void viterbi(int *, int, int, int, double *, double *, double *);
void baum_welch(int *data, int nseq, int iterations, int length, int nstates, int nobvs,
        double *, double * , double *);
void init_count();
void update_prob();
void usage();
void freeall();

#define IDX(i,j,d) (((i)*(d))+(j))
#define handle_error(msg) \
    do { perror(msg); exit(EXIT_FAILURE); } while (0)
