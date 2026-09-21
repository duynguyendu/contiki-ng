#include "net/link-stats.h"
#include "net/nbr-table.h"
#include "net/routing/rpl-lite/rpl.h"
#include "sys/energest.h"

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

#define RANK_THRESHOLD 96 // ~ 2.5 ETX (no PDR) or 6.69% in PDR (no ETX)

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
  /* Exclude links with too high link metrics  */
  return link_metric <= MAX_LINK_METRIC;
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
 * boot. Returns 0 when Energest is disabled (ENERGEST_CONF_ON == 0). */
static uint8_t cpu_usage_percent(void) {
#if ENERGEST_CONF_ON
  static uint64_t last_cpu = 0;
  static uint64_t last_total = 0;
  uint64_t cpu, total, delta_cpu, delta_total;

  energest_flush();
  cpu = energest_type_time(ENERGEST_TYPE_CPU);
  total = ENERGEST_GET_TOTAL_TIME();

  delta_cpu = cpu - last_cpu;
  delta_total = total - last_total;
  last_cpu = cpu;
  last_total = total;

  if (delta_total == 0) {
    return 0;
  }
  return (uint8_t)MIN(scaled_ratio(delta_cpu, delta_total, MLOF_CPU_USAGE_UNIT),
                      MLOF_CPU_USAGE_MAX);
#else  /* ENERGEST_CONF_ON */
  return 0;
#endif /* ENERGEST_CONF_ON */
}

/* ETX and RSSI to the preferred parent, from its link statistics. Both are 0 at
 * the root and INT16_MAX when unavailable: no preferred parent yet, or the
 * statistic has not been measured (ETX == 0, RSSI == LINK_STATS_RSSI_UNKNOWN,
 * which is itself INT16_MAX). The int16_t RSSI is carried in the uint16_t field
 * as-is. One shared root/link-stats check for both values. */
static void parent_link_metrics(uint16_t *etx, int16_t *rssi) {
  rpl_nbr_t *parent = curr_instance.dag.preferred_parent;
  const struct link_stats *stats;

  if (rpl_dag_root_is_root()) {
    *etx = 0;
    *rssi = 0;
    return;
  }
  stats = parent == NULL ? NULL : rpl_neighbor_get_link_stats(parent);
  if (stats == NULL) {
    *etx = (uint16_t)INT16_MAX;
    *rssi = INT16_MAX;
    return;
  }

  *etx = stats->etx == 0 ? (uint16_t)INT16_MAX : stats->etx;
  *rssi = stats->rssi == LINK_STATS_RSSI_UNKNOWN ? INT16_MAX : stats->rssi;
}

/* Blend weights out of 16 (5/16 ~= 0.31, 11/16 ~= 0.69) so the average is a
 * shift instead of a division. */
#define MLOF_CPU_W_SELF 5
#define MLOF_CPU_W_PARENT 11
#define MLOF_CPU_W_SHIFT 4

static uint8_t weighted_cpu_usage(uint8_t self_cpu_usage) {
  rpl_nbr_t *parent = curr_instance.dag.preferred_parent;
  unsigned blend;

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

  blend = (MLOF_CPU_W_SELF * self_cpu_usage +
           MLOF_CPU_W_PARENT * parent->mlof.weighted_cpu_usage +
           (1 << (MLOF_CPU_W_SHIFT - 1))) >>
          MLOF_CPU_W_SHIFT;
  return (uint8_t)MIN(blend, MLOF_CPU_USAGE_MAX);
}

#define MLOF_TRAFFIC_MIN_WINDOW (30 * CLOCK_SECOND)
#define MLOF_DROP_RATE_MAX MLOF_U8_REAL_MAX
#define MLOF_DROP_RATE_UNKNOWN MLOF_U8_UNKNOWN

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

  last_ppm = (uint16_t)MIN(
      scaled_ratio(tx_delta, time_delta, 128 * CLOCK_SECOND), 0xffff);
  last_drop_rate = drops_delta == 0 ? 0
                                    : (uint8_t)MIN(tx_delta / drops_delta,
                                                   MLOF_DROP_RATE_MAX);

  prev_time = now;
  prev_tx = tx_now;
  prev_drops = drops_now;

  *ppm = last_ppm;
  *drop_rate = last_drop_rate;
}

/* Preferred parent's own ppm/drop_rate/cpu_usage, as last advertised in its DIO
 * (i.e. fetched straight from its stored MLOF_MC, no recomputation) - lets a
 * node see how loaded its parent's own uplink and CPU already are, one hop
 * further up. 0 at the root, INT16_MAX / 0xff ("unknown") when there is no
 * parent. None of these are re-advertised on the wire (see rpl-icmp6.c). */
static void parent_advertised_metrics(uint16_t *parent_ppm,
                                      uint8_t *parent_drop_rate,
                                      uint8_t *parent_cpu_usage) {
  rpl_nbr_t *parent = curr_instance.dag.preferred_parent;

  if (rpl_dag_root_is_root()) {
    *parent_ppm = 0;
    *parent_drop_rate = 0;
    *parent_cpu_usage = 0;
    return;
  }
  if (parent == NULL) {
    *parent_ppm = (uint16_t)INT16_MAX;
    *parent_drop_rate = MLOF_DROP_RATE_UNKNOWN;
    *parent_cpu_usage = MLOF_CPU_USAGE_UNKNOWN;
    return;
  }

  *parent_ppm = parent->mlof.ppm;
  *parent_drop_rate = parent->mlof.drop_rate;
  *parent_cpu_usage = parent->mlof.weighted_cpu_usage;
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

/* This node's own CPU usage (fixed point, divisor MLOF_CPU_USAGE_UNIT) as last
 * sampled for the metric container. Cached so the parent-switch callback can
 * report it without calling cpu_usage_percent() again (that call consumes the
 * sampling interval). */
static uint8_t last_self_cpu_usage;

static uint16_t predict_pdr(rpl_nbr_t *nbr, int is_new) {
  uint16_t parent_ppm = nbr->mlof.ppm;
  uint8_t parent_drop_rate = nbr->mlof.drop_rate;
  int16_t rssi = nbr->mlof.rssi;
  uint8_t hop_count = nbr->mlof.hop_count;
  uint8_t cpu = last_self_cpu_usage;
  uint8_t p_cpu = nbr->mlof.weighted_cpu_usage;
  uint16_t etx = nbr->mlof.etx;
  uint16_t ppm = curr_instance.mc.mlof.ppm;
  uint8_t drop_rate =
      curr_instance.mc.mlof.drop_rate;

#if MLOF_MODEL == MLOF_MODEL_LINEAR
  return mlof_predict_pdr_linear(parent_ppm, parent_drop_rate, rssi, hop_count,
                                 (uint8_t)is_new, cpu, p_cpu, etx, ppm,
                                 drop_rate);
#elif MLOF_MODEL == MLOF_MODEL_DTREE
  return mlof_predict_pdr_dtree(parent_ppm, parent_drop_rate, rssi, hop_count,
                                (uint8_t)is_new, cpu, p_cpu, etx, ppm,
                                drop_rate);
#elif MLOF_MODEL == MLOF_MODEL_LGBM
  return mlof_predict_pdr_lgbm(is_new, cpu, p_cpu, etx, rssi, ppm, drop_rate,
                               parent_ppm, parent_drop_rate, hop_count);
#else /* MLOF_MODEL == MLOF_MODEL_SVM */
  return mlof_predict_pdr_svm(parent_ppm, parent_drop_rate, rssi, hop_count,
                              (uint8_t)is_new, cpu, p_cpu, etx, ppm, drop_rate);
#endif
}

static void fill_multiple_metrics(void) {
  rpl_mlof_mc_t *out = &curr_instance.mc.mlof;
  uint8_t self_cpu_usage = cpu_usage_percent();

  last_self_cpu_usage = self_cpu_usage;

  /* out->weighted_cpu_usage: this node's own weighted path metric, advertised
     on the wire. The weighted_cpu_usage() function is used only here, for that
     advertised value - out->parent_cpu_usage below is a plain fetch, never
     re-blended. */
  out->weighted_cpu_usage = weighted_cpu_usage(self_cpu_usage);
  parent_link_metrics(&out->etx, &out->rssi);
  /* This node's own ppm/drop_rate, measured locally on the link to its parent.
   */
  traffic_metrics(&out->ppm, &out->drop_rate);
  /* The parent's own ppm/drop_rate/cpu_usage, fetched from its last DIO. */
  parent_advertised_metrics(&out->parent_ppm, &out->parent_drop_rate,
                            &out->parent_cpu_usage);
  out->hop_count = hop_count_via_parent();
  out->nbr_count = (uint8_t)MIN(rpl_neighbor_count(), 0xff);
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

  /* new->mlof.{ppm,drop_rate} are the parent's own self-measured values (its
     traffic to its own parent) - i.e., from here,
     "parent_ppm"/"parent_drop_rate". curr_instance.mc.mlof.{ppm,drop_rate} are
     this node's own, as last computed by fill_multiple_metrics() (a plain
     stored value, safe to re-read here). */
  LOG_PRINT("MLOF metrics: is_new=%d parent_id=%u cpu=%u p_cpu=%u etx=%u "
            "rssi=%d ppm=%u drop_rate=%u parent_ppm=%u parent_drop_rate=%u "
            "hop_count=%u nbr_count=%u\n",
            is_new, parent_id, (unsigned)last_self_cpu_usage,
            (unsigned)parent->mlof.weighted_cpu_usage,
            (unsigned)parent->mlof.etx, (int)parent->mlof.rssi,
            (unsigned)curr_instance.mc.mlof.ppm,
            (unsigned)curr_instance.mc.mlof.drop_rate,
            (unsigned)parent->mlof.ppm, (unsigned)parent->mlof.drop_rate,
            (unsigned)parent->mlof.hop_count, (unsigned)parent->mlof.nbr_count);
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
