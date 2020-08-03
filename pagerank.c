#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <ctype.h>
#include <time.h>

#define D 0.85f
#define NSEC_IN_SEC 1000000000
#define NUM_ITERATIONS 100
#define NUM_TRIALS(graph_size) (fmin(100, fmax(3, 50000 / (graph_size))))

void time_graph_size(char* filename);
void run_pagerank(int num_nodes, int num_edges, uint16_t *src, uint16_t *dest);
void print_avg_time(struct timespec tick, struct timespec tock, int num_trials);

int main() {
    char* filenames[] = {
        "10.txt",
        "30.txt",
        "100.txt",
        "300.txt",
        "1000.txt",
        "3000.txt",
        "10000.txt",
        "30000.txt",
    };
    for (int i = 0; i < 8; i++) {
        time_graph_size(filenames[i]);
    }
}

void time_graph_size(char* filename) {
    int num_nodes, num_edges;
    freopen(filename, "r", stdin);
    scanf("%d %d", &num_nodes, &num_edges);
    // printf("%d nodes; %d edges\n", num_nodes, num_edges);

    uint16_t *src = malloc(num_edges * sizeof(uint16_t));
    uint16_t *dest = malloc(num_edges * sizeof(uint16_t));

    for (uint16_t *src_i = src, *dest_i = dest;
         scanf("%hi %hi", src_i, dest_i) != EOF;
         src_i++, dest_i++) {}

    struct timespec tick, tock;
    clock_gettime(CLOCK_REALTIME, &tick);
    int trials = NUM_TRIALS(num_nodes);
    for (int i = 0; i < trials; i++) {
        run_pagerank(num_nodes, num_edges, src, dest);
    }
    clock_gettime(CLOCK_REALTIME, &tock);
    print_avg_time(tick, tock, trials);

}

void run_pagerank(int num_nodes, int num_edges, uint16_t *src, uint16_t *dest) {
    uint16_t *neighbor_counts = malloc(num_edges * sizeof(uint16_t));
    float *scores = malloc(num_nodes * sizeof(float));
    float *new_scores = malloc(num_nodes * sizeof(float));

    for (int e = 0; e < num_edges; e++) {
        neighbor_counts[src[e]] += 1;
    }

    for (int n = 0; n < num_nodes; n++) {
        scores[n] = 1.0 / num_nodes;
        new_scores[n] = 0;
    }

    for (int i = 0; i < NUM_ITERATIONS; i++) {
        float *swap;
        for (int n = 0; n < num_nodes; n++) {
            scores[n] /= neighbor_counts[n];
        }
        for (int e = 0; e < num_edges; e++) {
            new_scores[dest[e]] += scores[src[e]];
        }
        for (int n = 0; n < num_nodes; n++) {
            new_scores[n] *= D;
            new_scores[n] += (1.0f - D) / num_nodes;
        }
        memset(scores, 0, num_nodes * sizeof(*scores));

        swap = scores;
        scores = new_scores;
        new_scores = swap;
    }
    // for (int n = 0; n < num_nodes; n++) {
    //     printf("%0.3f ", scores[n]);
    // }
}

void print_avg_time(struct timespec tick, struct timespec tock, int num_trials) {
    long int second_diff = tock.tv_sec - tick.tv_sec;
    long int nanosecond_diff = tock.tv_nsec - tick.tv_nsec;
    if (nanosecond_diff < 0) {
                          //123456789
        nanosecond_diff += NSEC_IN_SEC;
        second_diff -= 1;
    }
    //printf("%ld.%.9ld\n", second_diff, nanosecond_diff);
    ldiv_t divmod = ldiv(second_diff, num_trials);
    second_diff = divmod.quot;
    nanosecond_diff += divmod.rem * NSEC_IN_SEC;
    nanosecond_diff /= num_trials;
    printf("%ld.%.9ld\n", second_diff, nanosecond_diff);
}