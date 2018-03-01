#define RK_STATE_LEN 624

typedef struct rk_state_
{
    unsigned long key[RK_STATE_LEN];
    int pos;
}
rk_state;

extern unsigned long rk_random(rk_state *state);
extern double rk_double(rk_state *state);
extern double rk_uniform(rk_state *state, double loc, double scale);
extern void rk_seed(unsigned long seed, rk_state *state);

