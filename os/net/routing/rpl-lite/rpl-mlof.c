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
  if (nbr1 == curr_instance.dag.preferred_parent && within_hysteresis(nbr2)) {
    return nbr1;
  }
  if (nbr2 == curr_instance.dag.preferred_parent && within_hysteresis(nbr1)) {
    return nbr2;
  }

  return nbr_path_cost(nbr1) < nbr_path_cost(nbr2) ? nbr1 : nbr2;
}
/*---------------------------------------------------------------------------*/
#if RPL_MULTIPLE_METRICS
/* Default value of each metric at the root.
 * TODO: replace with real per-metric measurements. */
#define RPL_METRIC_DEFAULT_VALUE 123

/* Value the root advertises; bumped by one every time it is refreshed. */
static uint16_t root_metric_value = RPL_METRIC_DEFAULT_VALUE;

static void fill_multiple_metrics(void) {
  const uint8_t types[] = {RPL_DAG_MC_ENERGY, RPL_DAG_MC_ETX, RPL_DAG_MC_RSSI};
  rpl_metric_set_t *set = &curr_instance.mc.metrics;
  int is_root = rpl_dag_root_is_root();
  unsigned t;

  set->num_metrics = 0;
  for (t = 0; t < sizeof(types) && set->num_metrics < RPL_MC_MAX_METRICS; t++) {
    uint16_t value;
    if (is_root) {
      value = root_metric_value;
    } else {
      uint16_t received;
      // TODO: maybe have a fixed order of metrics and extract by index to
      // increase speed
      if (!rpl_icmp6_last_received_metric(types[t], &received)) {
        received = RPL_METRIC_DEFAULT_VALUE;
      }
      value = received + RPL_MIN_HOPRANKINC;
    }
    set->metrics[set->num_metrics].type = types[t];
    set->metrics[set->num_metrics].value = value;
    set->num_metrics++;
  }

  if (is_root) {
    root_metric_value++;
  }
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
