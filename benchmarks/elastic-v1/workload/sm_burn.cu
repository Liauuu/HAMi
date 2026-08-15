/**
 * sm_burn.cu — controllable SM load for elastic-v1 S1/S2/S3 harness.
 *
 * Modes:
 *   busy   continuous kernel launches, report iter/s
 *   idle   sleep only (no kernels) — becomes IDLE for monitor
 *   noisy  short busy bursts then sleep (oscillation probe)
 *   owner  idle until signal file appears, then busy (S3 reclaim)
 *
 * Launch path: CUDA Driver API cuLaunchKernel (not runtime <<<>>>).
 * cudart caches real libcuda pointers via dlsym and bypasses LD_PRELOAD
 * interposition; calling cuLaunchKernel from this binary hits libvgpu's
 * hooked export so mark_compute_active / last_launch_ns work.
 *
 * Example:
 *   LD_PRELOAD=./libvgpu.so CUDA_REDIRECT=./libvgpu.so \
 *     ./sm_burn --mode=busy --duration=30 --report-ms=200
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <errno.h>
#include <getopt.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#define CHECK_CUDA(call)                                                       \
  do {                                                                         \
    cudaError_t _e = (call);                                                   \
    if (_e != cudaSuccess) {                                                   \
      fprintf(stderr, "CUDA %s:%d: %s\n", __FILE__, __LINE__,                  \
              cudaGetErrorString(_e));                                         \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

#define CHECK_DRV(call)                                                        \
  do {                                                                         \
    CUresult _e = (call);                                                      \
    if (_e != CUDA_SUCCESS) {                                                  \
      const char *_s = NULL;                                                   \
      cuGetErrorString(_e, &_s);                                               \
      fprintf(stderr, "CUDA driver %s:%d: %s\n", __FILE__, __LINE__,           \
              _s ? _s : "unknown");                                            \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

static double now_s(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

__global__ void burn_kernel(float *data, int n, int iters) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n)
    return;
  float x = data[tid];
  for (int i = 0; i < iters; i++) {
    x = x * 1.000001f + 0.000001f;
    x = sqrtf(fabsf(x));
  }
  data[tid] = x;
}

typedef enum { MODE_BUSY, MODE_IDLE, MODE_NOISY, MODE_OWNER } mode_t;

static mode_t parse_mode(const char *s) {
  if (!strcmp(s, "busy"))
    return MODE_BUSY;
  if (!strcmp(s, "idle"))
    return MODE_IDLE;
  if (!strcmp(s, "noisy"))
    return MODE_NOISY;
  if (!strcmp(s, "owner"))
    return MODE_OWNER;
  fprintf(stderr, "unknown mode %s\n", s);
  exit(2);
}

static int file_exists(const char *path) {
  return path && path[0] && access(path, F_OK) == 0;
}

static void emit_tick(const char *name, const char *mode, long long iters,
                      double ips, double t) {
  printf("{\"event\":\"tick\",\"name\":\"%s\",\"mode\":\"%s\",\"iters\":%lld,"
         "\"iters_per_s\":%.3f,\"t_s\":%.3f}\n",
         name, mode, iters, ips, t);
  fflush(stdout);
}

static void emit_mode(const char *name, const char *mode, double t) {
  printf("{\"event\":\"mode\",\"name\":\"%s\",\"mode\":\"%s\",\"t_s\":%.3f}\n",
         name, mode, t);
  fflush(stdout);
}

static void launch_burn(CUfunction fn, float *d, int n, int ker_iters,
                        int blocks, int threads) {
  void *args[] = {&d, &n, &ker_iters};
  CHECK_DRV(cuLaunchKernel(fn, (unsigned)blocks, 1, 1, (unsigned)threads, 1, 1,
                           0, NULL, args, NULL));
  CHECK_CUDA(cudaDeviceSynchronize());
}

int main(int argc, char **argv) {
  const char *name = "worker";
  const char *mode_s = "busy";
  const char *signal_file = NULL;
  double duration = 30.0;
  double report_ms = 200.0;
  double idle_sec = 8.0;
  double noisy_on = 0.2;
  double noisy_off = 0.8;
  int n = 1 << 20; /* 1M floats ~4MiB */
  int ker_iters = 64;
  int device = 0;

  static struct option opts[] = {
      {"name", required_argument, 0, 'n'},
      {"mode", required_argument, 0, 'm'},
      {"duration", required_argument, 0, 'd'},
      {"report-ms", required_argument, 0, 'r'},
      {"signal-file", required_argument, 0, 's'},
      {"idle-sec", required_argument, 0, 'i'},
      {"noisy-on", required_argument, 0, 'o'},
      {"noisy-off", required_argument, 0, 'f'},
      {"device", required_argument, 0, 'g'},
      {0, 0, 0, 0},
  };

  int c;
  while ((c = getopt_long(argc, argv, "", opts, NULL)) != -1) {
    switch (c) {
    case 'n':
      name = optarg;
      break;
    case 'm':
      mode_s = optarg;
      break;
    case 'd':
      duration = atof(optarg);
      break;
    case 'r':
      report_ms = atof(optarg);
      break;
    case 's':
      signal_file = optarg;
      break;
    case 'i':
      idle_sec = atof(optarg);
      break;
    case 'o':
      noisy_on = atof(optarg);
      break;
    case 'f':
      noisy_off = atof(optarg);
      break;
    case 'g':
      device = atoi(optarg);
      break;
    default:
      return 2;
    }
  }

  mode_t mode = parse_mode(mode_s);
  CHECK_CUDA(cudaSetDevice(device));

  float *d;
  CHECK_CUDA(cudaMalloc(&d, (size_t)n * sizeof(float)));
  CHECK_CUDA(cudaMemset(d, 0, (size_t)n * sizeof(float)));
  int threads = 256;
  int blocks = (n + threads - 1) / threads;

  /* Resolve runtime kernel to a CUfunction, then launch via Driver API. */
  CUfunction burn_fn = NULL;
  CHECK_CUDA(cudaGetFuncBySymbol(&burn_fn, (const void *)burn_kernel));
  if (burn_fn == NULL) {
    fprintf(stderr, "cudaGetFuncBySymbol returned NULL\n");
    return 1;
  }

  double t0 = now_s();
  double last_report = t0;
  long long iters = 0;
  long long iters_window = 0;
  int burning = (mode == MODE_BUSY);
  const char *cur_mode = mode_s;

  if (mode == MODE_OWNER || mode == MODE_IDLE) {
    burning = 0;
    cur_mode = "idle";
    emit_mode(name, cur_mode, 0.0);
  } else {
    emit_mode(name, cur_mode, 0.0);
  }

  while (now_s() - t0 < duration) {
    double t = now_s() - t0;

    if (mode == MODE_OWNER) {
      int want = (t >= idle_sec) || file_exists(signal_file);
      if (want && !burning) {
        burning = 1;
        cur_mode = "busy";
        emit_mode(name, cur_mode, t);
      }
    } else if (mode == MODE_NOISY) {
      double cycle = noisy_on + noisy_off;
      double phase = fmod(t, cycle);
      int want = phase < noisy_on;
      if (want != burning) {
        burning = want;
        cur_mode = burning ? "busy" : "idle";
        emit_mode(name, cur_mode, t);
      }
    } else if (mode == MODE_IDLE) {
      burning = 0;
    } else {
      burning = 1;
    }

    if (burning) {
      launch_burn(burn_fn, d, n, ker_iters, blocks, threads);
      iters++;
      iters_window++;
    } else {
      usleep(20 * 1000);
    }

    double now = now_s();
    if ((now - last_report) * 1000.0 >= report_ms) {
      double dt = now - last_report;
      emit_tick(name, cur_mode, iters, (double)iters_window / dt, now - t0);
      iters_window = 0;
      last_report = now;
    }
  }

  CHECK_CUDA(cudaFree(d));
  printf("{\"event\":\"done\",\"name\":\"%s\",\"iters\":%lld,\"t_s\":%.3f}\n",
         name, iters, now_s() - t0);
  fflush(stdout);
  return 0;
}
