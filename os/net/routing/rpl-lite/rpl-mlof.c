/**
 * \addtogroup rpl-lite
 * @{
 *
 * \file
 *         The Minimum Rank with Hysteresis Objective Function (MRHOF), RFC6719
 *
 *         This implementation uses the estimated number of
 *         transmissions (ETX) as the additive routing metric,
 *         and also provides stubs for the energy metric.
 *
 * \author Joakim Eriksson <joakime@sics.se>, Nicolas Tsiftes <nvt@sics.se>
 *  Simon Duquennoy <simon.duquennoy@inria.fr>
 */

#include "net/link-stats.h"
#include "net/nbr-table.h"
#include "net/routing/rpl-lite/rpl.h"
#if RPL_MULTIPLE_METRICS
#include "sys/energest.h"
#endif /* RPL_MULTIPLE_METRICS */

/* Log configuration */
#include "sys/log.h"
#define LOG_MODULE "RPL"
#define LOG_LEVEL LOG_LEVEL_RPL

/* Configuration parameters of RFC6719. Reject parents that have a higher
 * link metric than the following. The default value is 512. */
#ifdef RPL_MRHOF_CONF_MAX_LINK_METRIC
#define MAX_LINK_METRIC RPL_MRHOF_CONF_MAX_LINK_METRIC
#else                       /* RPL_MRHOF_CONF_MAX_LINK_METRIC */
#define MAX_LINK_METRIC 512 /* Eq ETX of 4 */
#endif                      /* RPL_MRHOF_CONF_MAX_LINK_METRIC */

/* Reject parents that have a higher path cost than the following. */
#ifdef RPL_MRHOF_CONF_MAX_PATH_COST
#define MAX_PATH_COST RPL_MRHOF_CONF_MAX_PATH_COST
#else                       /*  RPL_MRHOF_CONF_MAX_PATH_COST */
#define MAX_PATH_COST 32768 /* Eq path ETX of 256 */
#endif                      /* RPL_MRHOF_CONF_MAX_PATH_COST */

#define RANK_THRESHOLD 192 /* Eq ETX of 1.5 */

/* Additional, custom hysteresis based on time. If a neighbor was consistently
 * better than our preferred parent for at least TIME_THRESHOLD, switch to
 * this neighbor regardless of RANK_THRESHOLD. */
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
#if RPL_MULTIPLE_METRICS
/* Local CPU usage in percent over the interval since the previous call:
 * delta(CPU ticks) / delta(total ticks) * 100. Total ticks = CPU + LPM +
 * DEEP_LPM (ENERGEST_GET_TOTAL_TIME). The first call measures since boot.
 * Returns 0 when Energest is disabled (ENERGEST_CONF_ON == 0). */
static uint16_t cpu_usage_percent(void) {
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
  return (uint16_t)MIN((delta_cpu * 100) / delta_total, 100);
#else  /* ENERGEST_CONF_ON */
  return 0;
#endif /* ENERGEST_CONF_ON */
}

/* ETX / RSSI to the preferred parent, taken from its link statistics.
 * Returns 0 at the root, and INT16_MAX when the value is unavailable: no
 * preferred parent yet, or the statistic has not been measured (ETX == 0,
 * RSSI == LINK_STATS_RSSI_UNKNOWN, which is itself INT16_MAX). The int16_t
 * RSSI is carried in the uint16_t field as-is (reinterpret on the receiver). */
typedef enum { PARENT_METRIC_ETX, PARENT_METRIC_RSSI } parent_metric_t;

static uint16_t parent_link_metric(parent_metric_t which) {
  rpl_nbr_t *parent = curr_instance.dag.preferred_parent;
  const struct link_stats *stats;
  uint16_t value, unknown;

  if (rpl_dag_root_is_root()) {
    return 0;
  }
  stats = parent == NULL ? NULL : rpl_neighbor_get_link_stats(parent);
  if (stats == NULL) {
    return (uint16_t)INT16_MAX;
  }

  if (which == PARENT_METRIC_RSSI) {
    value = (uint16_t)stats->rssi;
    unknown = (uint16_t)LINK_STATS_RSSI_UNKNOWN;
  } else {
    value = stats->etx;
    unknown = 0;
  }
  return value == unknown ? (uint16_t)INT16_MAX : value;
}

static void fill_multiple_metrics(void) {
  rpl_mlof_mc_t *out = &curr_instance.mc.mlof;

  /* CPU usage is a local node property, advertised as-is by every node. */
  out->cpu_usage = cpu_usage_percent();
  /* ETX and RSSI to the preferred parent (0 at the root). */
  out->etx = parent_link_metric(PARENT_METRIC_ETX);
  out->rssi = parent_link_metric(PARENT_METRIC_RSSI);
}
#endif /* RPL_MULTIPLE_METRICS */
/*---------------------------------------------------------------------------*/
static void update_metric_container(void) {
  curr_instance.mc.type = RPL_DAG_MC_NONE;
#if RPL_MULTIPLE_METRICS
  fill_multiple_metrics();
#endif /* RPL_MULTIPLE_METRICS */
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

/** @}*/
