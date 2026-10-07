#include "net/link-stats.h"
#include "net/nbr-table.h"
#include "net/routing/rpl-lite/rpl.h"
#include "sys/energest.h"
#include "sys/rtimer.h"

/* Log configuration */
#include "sys/log.h"
#define LOG_MODULE "RPL"
#define LOG_LEVEL LOG_LEVEL_RPL

/* MLOF is only used when the multi-metric DAG Metric Container is advertised.
 * Otherwise RPL_MULTIPLE_METRICS is 0 and this file compiles to an empty
 * object (the rpl_mlof symbol is simply absent). rpl-conf.h sets
 * RPL_MULTIPLE_METRICS when RPL_OF_OCP is MLOF or RPL_CONF_MULTIPLE_METRICS
 * is set; add &rpl_mlof to RPL_SUPPORTED_OFS only in those builds. */
#if RPL_MULTIPLE_METRICS

#define MAX_LINK_METRIC 512
#define MAX_PATH_COST 32768 /* Eq path ETX of 256 */
#define TIME_THRESHOLD (10 * 60 * CLOCK_SECOND)

#define NUM_PDR_STEP 32 // 1 step ~ 3.125% in PDR
#define PDR_STEP_VALUE 64
#define MLOF_PDR_RANGE_SCALED ((uint32_t)NUM_PDR_STEP * PDR_STEP_VALUE)

#define RANK_THRESHOLD 64 // ~ 2.5 ETX (no PDR) or 6.69% in PDR (no ETX)

#define MLOF_MODEL_SVM 0
#define MLOF_MODEL_LINEAR 1
#define MLOF_MODEL_DTREE 2
#define MLOF_MODEL_LGBM 3

#ifdef MLOF_CONF_MODEL
#define MLOF_MODEL MLOF_CONF_MODEL
#else
#define MLOF_MODEL MLOF_MODEL_DTREE
#endif

#if MLOF_MODEL == MLOF_MODEL_LINEAR
#include "mlof-linear.h"
#elif MLOF_MODEL == MLOF_MODEL_DTREE
#include "mlof-dtree.h"
#elif MLOF_MODEL == MLOF_MODEL_LGBM
#include "mlof-lgbm.h"
#else
#include "mlof-svm.h"
#endif

#ifdef MLOF_CONF_PATH_W_PDR
#define MLOF_PATH_W_PDR MLOF_CONF_PATH_W_PDR
#else
#define MLOF_PATH_W_PDR 12
#endif

/* Neighbors advertising a larger hop count (e.g. counting up in a loop, or
   0xff = unknown) are not usable. */
#define MLOF_MAX_HOP_COUNT 127

static uint16_t predict_pdr(rpl_nbr_t *nbr, int is_new);

static uint16_t scale_pdr_to_etx_range(uint16_t pdr) {
  return (uint16_t)(MLOF_PDR_RANGE_SCALED -
                    (MLOF_PDR_RANGE_SCALED * pdr) / 0xffff);
}

/*---------------------------------------------------------------------------*/
static void reset(void) { LOG_INFO("reset MLOF\n"); }
/*---------------------------------------------------------------------------*/
static uint16_t nbr_link_metric(rpl_nbr_t *nbr) {
  const struct link_stats *stats = rpl_neighbor_get_link_stats(nbr);
  return stats != NULL ? stats->etx : 0xffff;
}
/*---------------------------------------------------------------------------*/
static uint16_t link_metric_to_rank(uint16_t etx) { return etx; }
/*---------------------------------------------------------------------------*/
static uint16_t nbr_path_cost(rpl_nbr_t *nbr) {
  uint16_t base_rank;
  uint16_t etx;
  uint16_t pdr;
  uint16_t pdr_etx;
  uint32_t rank_increase;

  if (nbr == NULL) {
    return 0xffff;
  }

  base_rank = nbr->rank;

  pdr = predict_pdr(nbr, nbr != curr_instance.dag.preferred_parent);
  pdr_etx = scale_pdr_to_etx_range(pdr);
  etx = nbr_link_metric(nbr);

  rank_increase = ((16 - MLOF_PATH_W_PDR) * link_metric_to_rank(etx) +
                   MLOF_PATH_W_PDR * pdr_etx) /
                  16;

  return (uint16_t)MIN((uint32_t)base_rank + rank_increase, 0xffff);
}
/*---------------------------------------------------------------------------*/
static rpl_rank_t rank_via_nbr(rpl_nbr_t *nbr) {
  uint16_t min_hoprankinc;
  uint16_t path_cost;

  if (nbr == NULL) {
    return RPL_INFINITE_RANK;
  }

  min_hoprankinc = curr_instance.min_hoprankinc;
  path_cost = nbr_path_cost(nbr);

  /* Rank lower-bound: nbr rank + min_hoprankinc */
  return MAX(MIN((uint32_t)nbr->rank + min_hoprankinc, RPL_INFINITE_RANK),
             path_cost);
}
/*---------------------------------------------------------------------------*/
static int nbr_has_usable_link(rpl_nbr_t *nbr) {
  uint16_t link_metric = nbr_link_metric(nbr);
  /* Exclude links with too high link metrics or hop count */
  return link_metric <= MAX_LINK_METRIC &&
         nbr->mlof.hop_count <= MLOF_MAX_HOP_COUNT;
}
/*---------------------------------------------------------------------------*/
static int nbr_is_acceptable_parent(rpl_nbr_t *nbr) {
  /* Exclude links with too high link metrics or path cost (RFC6719, 3.2.2).
     The link check goes first: it is cheap, while nbr_path_cost() runs the
     model. */
  return nbr_has_usable_link(nbr) && nbr_path_cost(nbr) <= MAX_PATH_COST;
}
/*---------------------------------------------------------------------------*/
/* Path costs are passed in rather than recomputed: nbr_path_cost() runs the
 * model, and best_parent() already has both costs. */
static int within_hysteresis(rpl_nbr_t *nbr, uint16_t path_cost,
                             uint16_t parent_path_cost) {
  int within_rank_hysteresis = path_cost + RANK_THRESHOLD > parent_path_cost;
  int within_time_hysteresis =
      nbr->better_parent_since == 0 ||
      (clock_time() - nbr->better_parent_since) <= TIME_THRESHOLD;

  /* As we want to consider neighbors that are either beyond the rank or time
  hystereses, return 1 here iff the neighbor is within both hystereses. */
  return within_rank_hysteresis && within_time_hysteresis;
}
/*---------------------------------------------------------------------------*/
/* Path cost of nbr if it is an acceptable parent (usable link, cost within
 * MAX_PATH_COST), evaluating the model at most once. Returns 0 if not
 * acceptable, in which case *path_cost is unset. */
static int acceptable_path_cost(rpl_nbr_t *nbr, uint16_t *path_cost) {
  if (nbr == NULL || !nbr_has_usable_link(nbr)) {
    return 0;
  }
  *path_cost = nbr_path_cost(nbr);
  return *path_cost <= MAX_PATH_COST;
}
/*---------------------------------------------------------------------------*/
static rpl_nbr_t *best_parent(rpl_nbr_t *nbr1, rpl_nbr_t *nbr2) {
  uint16_t cost1 = 0;
  uint16_t cost2 = 0;
  int nbr1_is_acceptable = acceptable_path_cost(nbr1, &cost1);
  int nbr2_is_acceptable = acceptable_path_cost(nbr2, &cost2);

  if (!nbr1_is_acceptable) {
    return nbr2_is_acceptable ? nbr2 : NULL;
  }
  if (!nbr2_is_acceptable) {
    return nbr1;
  }

  /* Maintain stability of the preferred parent. Switch only if the gain
  is greater than RANK_THRESHOLD, or if the neighbor has been better than the
  current parent for at more than TIME_THRESHOLD. */
  if (nbr1 == curr_instance.dag.preferred_parent &&
      within_hysteresis(nbr2, cost2, cost1)) {
    return nbr1;
  }
  if (nbr2 == curr_instance.dag.preferred_parent &&
      within_hysteresis(nbr1, cost1, cost2)) {
    return nbr2;
  }

  return cost1 < cost2 ? nbr1 : nbr2;
}
/*---------------------------------------------------------------------------*/
/* 1-byte MLOF metrics keep real values in the lower half of the byte
 * (0..0x7f); 0x80..0xfe are deliberately left unused so the "unknown" sentinel
 * 0xff stays well separated from any legitimate maximum. */
#define MLOF_U8_REAL_MAX 0x7f
#define MLOF_U8_UNKNOWN 0xff

/* num * scale / den with a single 32-bit divide, for 64-bit counters that avoid
 * a __udivdi3 call on 32-bit MCUs. If num * scale or den would not fit in 32
 * bits, both are shifted right first (the ratio is preserved, only the lowest
 * bits are lost). Saturates to UINT32_MAX if den shifts down to 0, i.e. the
 * ratio is huge. den must be non-zero on entry. */
static uint32_t scaled_ratio(uint64_t num, uint64_t den, uint32_t scale) {
  const uint64_t num_max = UINT32_MAX / scale;

  while (num > num_max || den > UINT32_MAX) {
    num >>= 1;
    den >>= 1;
  }
  if (den == 0) {
    return UINT32_MAX;
  }
  return ((uint32_t)num * scale) / (uint32_t)den;
}

/* EWMA of this node's own samples (CPU usage, ppm, drop_rate), weighting the
 * current sample 5/16 so the divide is a shift. */
#define MLOF_EWMA(avg, cur) (((uint32_t)(avg) * 11 + (uint32_t)(cur) * 5) >> 4)

/* CPU-usage fixed-point unit, mirroring the ETX divisor scheme: the utilization
 * fraction f in [0,1] is carried as (uint8_t)(f * this). With unit 128 a value
 * of 128 would be 100%, but real values are capped at MLOF_CPU_USAGE_MAX
 * (0x7f, ~99.2%); 0xff is the "unknown" sentinel. */
#define MLOF_CPU_USAGE_UNIT 128
#define MLOF_CPU_USAGE_MAX MLOF_U8_REAL_MAX
#define MLOF_CPU_USAGE_UNKNOWN MLOF_U8_UNKNOWN
/* Local CPU usage over the interval since the previous call, as a fixed-point
 * fraction with divisor MLOF_CPU_USAGE_UNIT (same scheme as ETX):
 * delta(CPU ticks) * MLOF_CPU_USAGE_UNIT / delta(total ticks). Total ticks =
 * CPU + LPM + DEEP_LPM (ENERGEST_GET_TOTAL_TIME). The first call measures since
 * boot. Each sample is EWMA-smoothed (see MLOF_EWMA) and the smoothed value
 * is returned. Returns 0 when Energest is disabled (ENERGEST_CONF_ON == 0). */
static uint8_t cpu_usage_percent(void) {
#if ENERGEST_CONF_ON
  static uint64_t last_cpu = 0;
  static uint64_t last_total = 0;
  static uint8_t cpu_usage = 0;
  uint64_t cpu, total, delta_cpu, delta_total;

  energest_flush();
  cpu = energest_type_time(ENERGEST_TYPE_CPU);
  total = ENERGEST_GET_TOTAL_TIME();

  delta_cpu = cpu - last_cpu;
  delta_total = total - last_total;
  last_cpu = cpu;
  last_total = total;

  if (delta_total == 0) {
    return cpu_usage;
  }
  cpu_usage = (uint8_t)MLOF_EWMA(
      cpu_usage, MIN(scaled_ratio(delta_cpu, delta_total, MLOF_CPU_USAGE_UNIT),
                     MLOF_CPU_USAGE_MAX));
  return cpu_usage;
#else  /* ENERGEST_CONF_ON */
  return 0;
#endif /* ENERGEST_CONF_ON */
}

/* Blend weights out of 16 (5/16 ~= 0.31, 11/16 ~= 0.69) so the average is a
 * shift instead of a division. Shared by every self/parent path-metric blend
 * below (CPU usage, ppm). */
#define MLOF_BLEND_W_SELF 5
#define MLOF_BLEND_W_PARENT 11
#define MLOF_BLEND_W_SHIFT 4

static uint32_t blend_with_parent(uint32_t self_val, uint32_t parent_val) {
  return (MLOF_BLEND_W_SELF * self_val + MLOF_BLEND_W_PARENT * parent_val +
          (1 << (MLOF_BLEND_W_SHIFT - 1))) >>
         MLOF_BLEND_W_SHIFT;
}

static uint8_t weighted_cpu_usage(uint8_t self_cpu_usage) {
  rpl_nbr_t *parent = curr_instance.dag.preferred_parent;

  if (rpl_dag_root_is_root()) {
    return 0;
  }
  if (parent == NULL ||
      parent->mlof.weighted_cpu_usage == MLOF_CPU_USAGE_UNKNOWN) {
    /* No usable ancestor term (no parent, or parent still advertising the
       "unknown" sentinel, e.g. from stale firmware): report our own load
       rather than blending a bogus value. */
    return self_cpu_usage;
  }

  return (uint8_t)MIN(
      blend_with_parent(self_cpu_usage, parent->mlof.weighted_cpu_usage),
      MLOF_CPU_USAGE_MAX);
}

#define MLOF_TRAFFIC_MIN_WINDOW (30 * CLOCK_SECOND)
#define MLOF_DROP_RATE_MAX MLOF_U8_REAL_MAX

/* This node's own ppm blended with its preferred parent's last-advertised
 * ppm, mirroring weighted_cpu_usage(): the advertised value becomes a path
 * metric (an exponentially-decaying average along the route to the root)
 * instead of a single-hop reading, so a congested link further up the path is
 * visible to nodes several hops below it. ppm does not define an "unknown"
 * sentinel (0 from a real but quiet parent is meaningful), so only a missing
 * parent is special-cased. */
static uint16_t weighted_ppm(uint16_t self_ppm) {
  rpl_nbr_t *parent = curr_instance.dag.preferred_parent;

  if (rpl_dag_root_is_root()) {
    return 0;
  }
  if (parent == NULL) {
    return self_ppm;
  }
  return (uint16_t)MIN(blend_with_parent(self_ppm, parent->mlof.weighted_ppm),
                       0xffff);
}

static void traffic_metrics(uint16_t *ppm, uint8_t *drop_rate) {
  static uint32_t prev_tx = 0;
  static uint32_t prev_drops = 0;
  static clock_time_t prev_time = 0;

  static uint16_t last_ppm = 0;
  static uint8_t last_drop_rate = 0;

  if (rpl_dag_root_is_root()) {
    *ppm = 0;
    *drop_rate = 0;
    return;
  }

  clock_time_t now = clock_time();
  clock_time_t time_delta = now - prev_time;
  if (time_delta <= MLOF_TRAFFIC_MIN_WINDOW) {
    *ppm = last_ppm;
    *drop_rate = last_drop_rate;
    return;
  }

  uint32_t tx_now = link_stats_tx_count();
  uint32_t drops_now = link_stats_drop_count();
  uint32_t tx_delta = tx_now - prev_tx;
  uint32_t drops_delta = drops_now - prev_drops;

  /* One new sample per window, EWMA-smoothed. ppm starts from the current
     sample rather than blending with 0. */
  uint16_t cur_ppm = (uint16_t)MIN(
      scaled_ratio(tx_delta, time_delta, 128 * CLOCK_SECOND), 0xffff);
  last_ppm =
      last_ppm == 0 ? cur_ppm : (uint16_t)MLOF_EWMA(last_ppm, cur_ppm);
  last_drop_rate = (uint8_t)MLOF_EWMA(
      last_drop_rate,
      drops_delta == 0 ? 0 : MIN(tx_delta / drops_delta, MLOF_DROP_RATE_MAX));

  prev_time = now;
  prev_tx = tx_now;
  prev_drops = drops_now;

  *ppm = last_ppm;
  *drop_rate = last_drop_rate;
}

static uint8_t hop_count_via_parent(void) {
  rpl_nbr_t *parent = curr_instance.dag.preferred_parent;

  if (rpl_dag_root_is_root()) {
    return 0;
  }
  if (parent == NULL || parent->mlof.hop_count >= MLOF_U8_REAL_MAX) {
    return MLOF_U8_UNKNOWN;
  }
  return parent->mlof.hop_count + 1;
}

/* This node's own CPU usage/drop_rate (fixed point; single-hop, i.e. not
 * blended with the parent's) as last sampled for the metric container.
 * Cached so the parent-switch callback can report them without resampling,
 * and so predict_pdr() can use the single-hop reading for its own "cpu"/
 * "drop_rate" features - out->weighted_cpu_usage, by contrast, is the
 * path-blended value advertised on the wire (see fill_multiple_metrics()). */
static uint8_t last_self_cpu_usage;
static uint8_t last_self_drop_rate;

/* Total rtimer ticks spent in predict_pdr() and number of calls since boot,
 * printed by the client's metrics log to get the average run time. */
uint32_t mlof_predict_ticks;
uint32_t mlof_predict_count;

static uint16_t predict_pdr(rpl_nbr_t *nbr, int is_new) {
  rtimer_clock_t start = RTIMER_NOW();
  uint16_t pdr;
#if MLOF_MODEL == MLOF_MODEL_LINEAR
  pdr = mlof_predict_pdr_linear((uint8_t)is_new, last_self_cpu_usage,
                                nbr->mlof.weighted_cpu_usage,
                                last_self_drop_rate, nbr->mlof.weighted_ppm,
                                nbr->mlof.hop_count);
#elif MLOF_MODEL == MLOF_MODEL_DTREE
  pdr = mlof_predict_pdr_dtree((uint8_t)is_new, last_self_cpu_usage,
                               nbr->mlof.weighted_cpu_usage,
                               last_self_drop_rate, nbr->mlof.weighted_ppm,
                               nbr->mlof.hop_count);
#elif MLOF_MODEL == MLOF_MODEL_LGBM
  pdr = mlof_predict_pdr_lgbm((uint8_t)is_new, last_self_cpu_usage,
                              nbr->mlof.weighted_cpu_usage,
                              last_self_drop_rate, nbr->mlof.weighted_ppm,
                              nbr->mlof.hop_count);
#else /* MLOF_MODEL == MLOF_MODEL_SVM */
  pdr = mlof_predict_pdr_svm((uint8_t)is_new, last_self_cpu_usage,
                             nbr->mlof.weighted_cpu_usage,
                             last_self_drop_rate, nbr->mlof.weighted_ppm,
                             nbr->mlof.hop_count);
#endif

  mlof_predict_ticks += (rtimer_clock_t)(RTIMER_NOW() - start);
  mlof_predict_count++;
  return pdr;
}

static void fill_multiple_metrics(void) {
  rpl_mlof_mc_t *out = &curr_instance.mc.mlof;
  uint8_t self_cpu_usage = cpu_usage_percent();
  uint16_t self_ppm;
  uint8_t self_drop_rate;
  traffic_metrics(&self_ppm, &self_drop_rate);

  last_self_cpu_usage = self_cpu_usage;
  last_self_drop_rate = self_drop_rate;

  /* This node's own CPU usage and ppm (measured locally on the link to its
     parent), each blended with the parent's advertised value into a path
     metric advertised on the wire. */
  out->weighted_cpu_usage = weighted_cpu_usage(self_cpu_usage);
  out->weighted_ppm = weighted_ppm(self_ppm);
  out->hop_count = hop_count_via_parent();
}
/*---------------------------------------------------------------------------*/
#if MLOF_LOG_TRAINING_DATA
void rpl_mlof_callback_parent_switch(rpl_nbr_t *old, rpl_nbr_t *parent,
                                     int is_new) {
  const linkaddr_t *lla;
  unsigned parent_id;
  (void)old;

  if (parent == NULL) {
    LOG_PRINT("MLOF metrics: null parent\n");
    return;
  }

  /* Node id of the new parent, derived from its link-layer address the same way
     node_id_init() derives our own (last two bytes, big endian). */
  lla = rpl_neighbor_get_lladdr(parent);
  parent_id = lla == NULL ? 0
                          : lla->u8[LINKADDR_SIZE - 1] +
                                (lla->u8[LINKADDR_SIZE - 2] << 8);

  LOG_PRINT("MLOF metrics: is_new=%d parent_id=%u cpu=%u p_cpu=%u "
            "drop_rate=%u parent_ppm=%u hop_count=%u\n",
            is_new, parent_id, (unsigned)last_self_cpu_usage,
            (unsigned)parent->mlof.weighted_cpu_usage,
            (unsigned)last_self_drop_rate,
            (unsigned)parent->mlof.weighted_ppm,
            (unsigned)parent->mlof.hop_count);
}

/* 2-arg adapter wired as RPL_CALLBACK_PARENT_SWITCH: every call through it is a
 * real switch to a new preferred parent. */
void rpl_mlof_of_callback_parent_switch(rpl_nbr_t *old, rpl_nbr_t *new) {
  rpl_mlof_callback_parent_switch(old, new, 1);
}
#endif /* MLOF_LOG_TRAINING_DATA */
/*---------------------------------------------------------------------------*/
static void update_metric_container(void) {
  curr_instance.mc.type = RPL_DAG_MC_MLOF;
  fill_multiple_metrics();
}

/*---------------------------------------------------------------------------*/
rpl_of_t rpl_mlof = {reset,
                     nbr_link_metric,
                     nbr_has_usable_link,
                     nbr_is_acceptable_parent,
                     nbr_path_cost,
                     rank_via_nbr,
                     best_parent,
                     update_metric_container,
                     RPL_OCP_MLOF};

#endif /* RPL_MULTIPLE_METRICS */
