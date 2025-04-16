#!/usr/bin/env bash

BASEDIR=$(realpath $(dirname "$0"))

THREADS=$(nproc)
DEFAULTFIGURES="3a 3b 3c 3d 4 6 7 8 9 10 11"
FIGURES="${DEFAULTFIGURES}"

while getopts "t:f:" opt; do
  case ${opt} in
    t)
      THREADS=${OPTARG}
      ;;
    f)
      FIGURES=${OPTARG}
      ;;
    ?)
      echo "Unknown option"
      exit 1
      ;;
    h)
      echo "./genfigures.sh -f [FIGURES] -t [NUMTHREADS]"
      exit 0
      ;;
  esac
done

cd ${BASEDIR}

for fig in ${FIGURES}; do
  PLOTPARAMS=
  case ${fig} in
    3a)
      LEAKFILES=(perfect_hint)
      SETFILES=(hints)
      RESULTS=(perfect_hint_hints)
      PLOTPARAMS="-y bikz_avg -x p -p solver nfac p"
      ;;
    3b)
      LEAKFILES=(perfect_hint_bino perfect_hint_bino)
      SETFILES=(hints bino_hints)
      RESULTS=(perfect_hint_bino_hints perfect_hint_bino_bino_hints)
      PLOTPARAMS="-y bikz_avg -x p -p solver nfac p"
      ;;
    3c)
      LEAKFILES=(approximate_hint)
      SETFILES=(approximate_hint_greedy)
      RESULTS=(approximate_hint_approximate_hint_greedy)
      PLOTPARAMS="-y bikz_avg -x sigma -p solver nfac sigma"
      ;;
    3d)
      LEAKFILES=(approximate_hint_bino)
      SETFILES=(approximate_hint)
      RESULTS=(approximate_hint_bino_approximate_hint)
      PLOTPARAMS="-y bikz_avg -x sigma -p solver nfac sigma"
      ;;
    4)
      LEAKFILES=(approximate_hint approximate_hint_bino)
      SETFILES=(approximate_hint_single_greedy approximate_hint_single)
      RESULTS=(approximate_hint_approximate_hint_single_greedy approximate_hint_bino_approximate_hint_single)
      PLOTPARAMS="-y bikz_avg bikz_min bikz_max -x nfac -p solver nfac sigma"
      ;;
    6)
      LEAKFILES=(nt_leakage_value nt_leakage_value)
      SETFILES=(nt_vl_bp nt_vl_greedy)
      RESULTS=(nt_leakage_value_nt_vl_bp nt_leakage_value_nt_vl_greedy)
      PLOTPARAMS="-y bikz_avg bikz_min bikz_max -x nfac -p solver nfac sigma"
      ;;
    7)
      LEAKFILES=(nt_leakage_hw nt_leakage_hw)
      SETFILES=(nt_hw_bp nt_hw_greedy)
      RESULTS=(nt_leakage_hw_nt_hw_bp nt_leakage_hw_nt_hw_greedy)
      PLOTPARAMS_nt_leakage_hw_nt_hw_bp="-y bikz_avg bikz_min bikz_max -x sigma -p solver sigma nfac"
      PLOTPARAMS_nt_leakage_hw_nt_hw_greedy="-y bikz_avg bikz_min bikz_max -x sigma -p solver sigma nfac conv"
      ;;
    8)
      LEAKFILES=(nt_leakage_intt nt_leakage_intt)
      SETFILES=(nt_hw_intt_bp nt_hw_intt_greedy)
      RESULTS=(nt_leakage_intt_nt_hw_intt_bp nt_leakage_intt_nt_hw_intt_greedy)
      PLOTPARAMS_nt_leakage_intt_nt_hw_intt_bp="-y bikz_avg bikz_min bikz_max -x sigma -p solver sigma nfac"
      PLOTPARAMS_nt_leakage_intt_nt_hw_intt_greedy="-y bikz_avg bikz_min bikz_max -x sigma -p solver sigma nfac nd"
      ;;
    9)
      LEAKFILES=(nt_leakage_hw nt_leakage_hw)
      SETFILES=(numerical_bp nt_hw_greedy)
      RESULTS=(nt_leakage_hw_numerical_bp nt_leakage_hw_nt_hw_greedy)
      PLOTPARAMS_nt_leakage_hw_numerical_bp="-y bikz_avg bikz_min bikz_max -x sigma -p solver sigma nfac"
      PLOTPARAMS_nt_leakage_hw_nt_hw_greedy="-y bikz_avg bikz_min bikz_max -x sigma -p solver sigma nfac conv"
      ;;
    10)
      LEAKFILES=(nt_leakage_hw nt_leakage_hw)
      SETFILES=(conv_eval_bp conv_eval_gr)
      RESULTS=(nt_leakage_hw_conv_eval_bp nt_leakage_hw_conv_eval_gr)
      PLOTPARAMS="-y bikz_avg bikz_min bikz_max -x conv -p solver sigma nfac conv"
      ;;
    11)
      LEAKFILES=(modular_hint modular_hint_bino)
      SETFILES=(mod_hints_uni mod_hints_bino)
      RESULTS=(modular_hint_mod_hints_uni modular_hint_bino_mod_hints_bino)
      PLOTPARAMS="-y bikz_avg -x nfac -p solver nfac"
      ;;
    *)
      echo "Unknown figure"
      exit 1
  esac

  echo figure=${fig}
  for (( i=0; i<${#LEAKFILES[*]}; ++i )); do
    leakfile=${LEAKFILES[$i]}
    setfile=${SETFILES[$i]}
    result=${RESULTS[$i]}
    echo leakfile=${leakfile} settingsfile=${setfile} result=${result}
    if [[ -z "${PLOTPARAMS}" ]]; then
      n="PLOTPARAMS_${result}"
      ALTPLOTPARAMS=${!n}
    fi
    cd src/
    python3 eval.py --threads ${THREADS} ../configs/leakage/${leakfile}.json ../configs/settings/${setfile}.json
    cd ../
    python3 scripts/plot.py ${PLOTPARAMS}${ALTPLOTPARAMS} --use-all-jsons --sorted results/${result} | sed -e "/@@OUTPUT@@/r /dev/stdin" -e "/@@OUTPUT@@/d" scripts/fig.tex > figure${fig}.tex
  done
done

