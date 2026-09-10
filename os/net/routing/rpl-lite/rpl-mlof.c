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
#define RANK_THRESHOLD 192  /* Eq ETX of 1.5 */
#define TIME_THRESHOLD (10 * 60 * CLOCK_SECOND)

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
  uint16_t base;

  if (nbr == NULL) {
    return 0xffff;
  }

  base = nbr->rank;

  /* path cost upper bound: 0xffff */
  return MIN((uint32_t)base + link_metric_to_rank(nbr_link_metric(nbr)),
             0xffff);
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
  uint16_t path_cost = nbr_path_cost(nbr);
  /* Exclude links with too high link metrics or path cost (RFC6719, 3.2.2) */
  return nbr_has_usable_link(nbr) && path_cost <= MAX_PATH_COST;
}
/*---------------------------------------------------------------------------*/
static int within_hysteresis(rpl_nbr_t *nbr) {
  // TODO: update this hysteresis for new metrics
  uint16_t path_cost = nbr_path_cost(nbr);
  uint16_t parent_path_cost = nbr_path_cost(curr_instance.dag.preferred_parent);

  int within_rank_hysteresis = path_cost + RANK_THRESHOLD > parent_path_cost;
  int within_time_hysteresis =
      nbr->better_parent_since == 0 ||
      (clock_time() - nbr->better_parent_since) <= TIME_THRESHOLD;

  /* As we want to consider neighbors that are either beyond the rank or time
  hystereses, return 1 here iff the neighbor is within both hystereses. */
  return within_rank_hysteresis && within_time_hysteresis;
}
/*---------------------------------------------------------------------------*/
static rpl_nbr_t *best_parent(rpl_nbr_t *nbr1, rpl_nbr_t *nbr2) {
  int nbr1_is_acceptable;
  int nbr2_is_acceptable;

  nbr1_is_acceptable = nbr1 != NULL && nbr_is_acceptable_parent(nbr1);
  nbr2_is_acceptable = nbr2 != NULL && nbr_is_acceptable_parent(nbr2);

  if (!nbr1_is_acceptable) {
    return nbr2_is_acceptable ? nbr2 : NULL;
  }
  if (!nbr2_is_acceptable) {
    return nbr1_is_acceptable ? nbr1 : NULL;
  }

  /* Maintain stability of the preferred parent. Switch only if the gain
  is greater than RANK_THRESHOLD, or if the neighbor has been better than the
  current parent for at more than TIME_THRESHOLD. */
  // TODO: Calculate the path cost of nbr1 and nbr2
  if (nbr1 == curr_instance.dag.preferred_parent && within_hysteresis(nbr2)) {
    return nbr1;
  }
  if (nbr2 == curr_instance.dag.preferred_parent && within_hysteresis(nbr1)) {
    return nbr2;
  }

  // TODO: add custom OF here

  return nbr_path_cost(nbr1) < nbr_path_cost(nbr2) ? nbr1 : nbr2;
}
/*---------------------------------------------------------------------------*/
/* CPU-usage fixed-point unit, mirroring the ETX divisor scheme: the utilization
 * fraction f in [0,1] is carried as (uint8_t)(f * this). The container field is
 * one byte, so real values are capped at MLOF_CPU_USAGE_MAX (0xfe, ~99.6%) and
 * 0xff is reserved as the "unknown" sentinel. */
#define MLOF_CPU_USAGE_UNIT 256
#define MLOF_CPU_USAGE_MAX 0xfe
#define MLOF_CPU_USAGE_UNKNOWN 0xff

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
  return (uint8_t)MIN((delta_cpu * MLOF_CPU_USAGE_UNIT) / delta_total,
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

#define MLOF_CPU_W_SELF 3
#define MLOF_CPU_W_PARENT 7

static uint8_t weighted_cpu_usage(uint8_t self_cpu_usage) {
  rpl_nbr_t *parent = curr_instance.dag.preferred_parent;
  unsigned blend;

  if (rpl_dag_root_is_root()) {
    return 0;
  }
  if (parent == NULL || parent->mlof.cpu_usage == MLOF_CPU_USAGE_UNKNOWN) {
    return MLOF_CPU_USAGE_UNKNOWN;
  }

  blend = (MLOF_CPU_W_SELF * self_cpu_usage +
           MLOF_CPU_W_PARENT * parent->mlof.cpu_usage + 5) /
          10;
  return (uint8_t)MIN(blend, MLOF_CPU_USAGE_MAX);
}

#define MLOF_TRAFFIC_MIN_WINDOW (30 * CLOCK_SECOND)
#define MLOF_DROP_RATE_UNKNOWN 0xff

/* Traffic metrics on the link to the preferred parent. Both are sampled over a
 * sliding window of at least MLOF_TRAFFIC_MIN_WINDOW and share one link-stats
 * baseline (prev_tx / prev_drops / prev_time) that is reset whenever the
 * preferred parent changes, is lost, or a link-stats counter wraps. Between
 * windows the last computed values are returned; until the first full window
 * after a (re)start they read "unknown" (INT16_MAX / 0xff).
 *  - ppm:       tx attempts on the parent link, packets/second scaled by 128
 *               (same fixed-point convention as the old send_rate value).
 *  - drop_rate: tx attempts per queue drop over the window; 0 = no drops in
 *               the window, 0xff = unknown.
 * Both are 0 at the root. Requires link-stats packet counters, which
 * contiki-default-conf.h enables automatically when MLOF is the OF. */
static void parent_traffic_metrics(uint16_t *ppm, uint8_t *drop_rate) {
  static const rpl_nbr_t *prev_parent = NULL;
  static uint16_t prev_tx = 0;
  static uint16_t prev_drops = 0;
  static clock_time_t prev_time = 0;
  static uint16_t last_ppm = (uint16_t)INT16_MAX;
  static uint8_t last_drop_rate = MLOF_DROP_RATE_UNKNOWN;

  rpl_nbr_t *parent = curr_instance.dag.preferred_parent;
  const struct link_stats *stats;
  clock_time_t now = clock_time();
  uint16_t tx_now, drops_now, tx_delta, drops_delta;
  clock_time_t time_delta;

  if (rpl_dag_root_is_root()) {
    *ppm = 0;
    *drop_rate = 0;
    return;
  }

  stats = parent == NULL ? NULL : rpl_neighbor_get_link_stats(parent);

  /* (Re)start the window on a parent change, a lost parent, or a counter wrap.
   * Report "unknown" until the next full window has elapsed. */
  if (parent != prev_parent || stats == NULL ||
      stats->cnt_current.num_packets_tx < prev_tx ||
      stats->cnt_current.num_queue_drops < prev_drops) {
    prev_parent = parent;
    prev_time = now;
    prev_tx = stats ? stats->cnt_current.num_packets_tx : 0;
    prev_drops = stats ? stats->cnt_current.num_queue_drops : 0;
    last_ppm = (uint16_t)INT16_MAX;
    last_drop_rate = MLOF_DROP_RATE_UNKNOWN;
    *ppm = last_ppm;
    *drop_rate = last_drop_rate;
    return;
  }

  time_delta = now - prev_time;
  if (time_delta <= MLOF_TRAFFIC_MIN_WINDOW) {
    *ppm = last_ppm;
    *drop_rate = last_drop_rate;
    return;
  }

  tx_now = stats->cnt_current.num_packets_tx;
  drops_now = stats->cnt_current.num_queue_drops;
  tx_delta = tx_now - prev_tx;
  drops_delta = drops_now - prev_drops;

  last_ppm = (uint16_t)MIN(
      ((uint32_t)tx_delta * 128 * CLOCK_SECOND) / time_delta, 0xffff);
  last_drop_rate =
      drops_delta == 0 ? 0 : (uint8_t)MIN(tx_delta / drops_delta, 0xff);

  prev_tx = tx_now;
  prev_drops = drops_now;
  prev_time = now;

  LOG_PRINT("MLOF traffic: parent_tx=%u tx_delta=%u drops_delta=%u "
            "time_delta=%lu ppm=%u drop_rate=%u\n",
            (unsigned)tx_now, (unsigned)tx_delta, (unsigned)drops_delta,
            (unsigned long)time_delta, (unsigned)last_ppm,
            (unsigned)last_drop_rate);

  *ppm = last_ppm;
  *drop_rate = last_drop_rate;
}

static uint8_t hop_count_via_parent(void) {
  rpl_nbr_t *parent = curr_instance.dag.preferred_parent;

  if (rpl_dag_root_is_root()) {
    return 0;
  }
  if (parent == NULL || parent->mlof.hop_count >= 0xfe) {
    return 0xff;
  }
  return parent->mlof.hop_count + 1;
}

/* This node's own CPU usage (fixed point, divisor MLOF_CPU_USAGE_UNIT) as last
 * sampled for the metric container. Cached so the parent-switch callback can
 * report it without calling cpu_usage_percent() again (that call consumes the
 * sampling interval). */
static uint8_t last_self_cpu_usage;

static void fill_multiple_metrics(void) {
  rpl_mlof_mc_t *out = &curr_instance.mc.mlof;
  uint8_t self_cpu_usage = cpu_usage_percent();

  last_self_cpu_usage = self_cpu_usage;

  out->cpu_usage = weighted_cpu_usage(self_cpu_usage);
  parent_link_metrics(&out->etx, &out->rssi);
  parent_traffic_metrics(&out->ppm, &out->drop_rate);
  out->hop_count = hop_count_via_parent();
  out->nbr_count = (uint8_t)MIN(rpl_neighbor_count(), 0xff);
}
/*---------------------------------------------------------------------------*/
void rpl_mlof_callback_parent_switch(rpl_nbr_t *old, rpl_nbr_t *new) {
  (void)old;

  if (new == NULL) {
    LOG_PRINT("MLOF metrics: null new\n");
    return;
  }

  LOG_PRINT("MLOF metrics: cpu=%u p_cpu=%u etx=%u rssi=%d ppm=%u drop_rate=%u "
            "hop_count=%u nbr_count=%u\n",
            (unsigned)last_self_cpu_usage, (unsigned)new->mlof.cpu_usage,
            (unsigned)new->mlof.etx, (int)new->mlof.rssi,
            (unsigned)new->mlof.ppm, (unsigned)new->mlof.drop_rate,
            (unsigned)new->mlof.hop_count, (unsigned)new->mlof.nbr_count);
}
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
